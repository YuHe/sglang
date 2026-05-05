# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/state-spaces/mamba/blob/60dadf2e0ee730ac337035d5533de10bc26e4847/mamba_ssm/ops/triton/layernorm_gated.py

# 门控 LayerNorm / RMSNorm Triton kernel 实现。
# 支持 LayerNorm 和 RMSNorm 两种归一化模式，以及可选的门控（z branch）。

import torch
import triton
import triton.language as tl


# heuristics 自动推断 HAS_BIAS（是否存在偏置 B）和 HAS_Z（是否存在门控分支 Z）
@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X,  # pointer to the input     输入张量指针
    Y,  # pointer to the output    输出张量指针
    W,  # pointer to the weights   归一化缩放权重 (gamma)
    B,  # pointer to the biases    归一化偏置 (beta)，可为 None
    Z,  # pointer to the other branch  门控分支张量，可为 None
    Mean,  # pointer to the mean   存储均值的指针（LayerNorm 用）
    Rstd,  # pointer to the 1/std  存储标准差倒数 1/std 的指针
    stride_x_row: tl.int64,  # X 行步长（token 间距）
    stride_y_row: tl.int64,  # Y 行步长
    stride_z_row: tl.int64,  # Z 行步长
    M: tl.int64,  # number of rows in X   行数（token 数）
    N: tl.int64,  # number of columns in X 列数（每组特征数）
    eps,  # epsilon to avoid division by zero  数值稳定 epsilon
    BLOCK_N: tl.constexpr,  # 每个 Triton block 处理的列数（2 的幂）
    HAS_BIAS: tl.constexpr,    # 是否有 bias（自动推断）
    HAS_Z: tl.constexpr,       # 是否有门控分支（自动推断）
    NORM_BEFORE_GATE: tl.constexpr,  # True: 先归一化再乘 gate；False: 先乘 gate 再归一化
    IS_RMS_NORM: tl.constexpr,       # True: 使用 RMSNorm；False: 使用 LayerNorm
):
    # Map the program id to the row of X and Y it should compute.
    # 将 program id 映射到对应的 token 行和归一化组
    row = tl.program_id(0)
    group = tl.program_id(1)
    # 根据行/组偏移定位各指针起始位置
    X += row * stride_x_row + group * N
    Y += row * stride_y_row + group * N
    if HAS_Z:
        Z += row * stride_z_row + group * N
    if not IS_RMS_NORM:
        Mean += group * M
    Rstd += group * M
    W += group * N
    if HAS_BIAS:
        B += group * N
    # Compute mean and variance
    # 计算均值和方差（或直接计算 RMS）
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    # 若先用 gate（NORM_BEFORE_GATE=False）：x = x * SiLU(z) = x * z * sigmoid(z)
    if HAS_Z and not NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=cols < N).to(tl.float32)
        x *= z * tl.sigmoid(z)
    if not IS_RMS_NORM:
        # LayerNorm：计算均值，然后方差
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        xbar = tl.where(cols < N, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        # RMSNorm：只计算 RMS（均方根），不需要减均值
        xbar = tl.where(cols < N, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    # rstd = 1 / sqrt(var + eps)（倒数标准差，用于归一化）
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    # 归一化并应用线性变换（y = (x - mean) * rstd * w + b）
    mask = cols < N
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w + b if HAS_BIAS else x_hat * w
    # 若 NORM_BEFORE_GATE=True：先归一化，再乘 gate
    if HAS_Z and NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=mask).to(tl.float32)
        y *= z * tl.sigmoid(z)
    # Write output
    # 写出归一化结果
    tl.store(Y + cols, y, mask=mask)


def _layer_norm_fwd(
    x,
    weight,
    bias,
    eps,
    z=None,          # 门控分支张量（可选）
    out=None,        # 预分配的输出张量（可选）
    group_size=None, # 每个归一化组的大小（None 表示整个特征维）
    norm_before_gate=True,  # 是否在 gate 前做归一化
    is_rms_norm=False,       # 是否使用 RMSNorm（True）还是 LayerNorm（False）
):
    # 推导基本形状参数
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    # allocate output
    # 分配或复用输出 tensor
    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1
    # 为 LayerNorm 分配均值存储；RMSNorm 不需要均值
    mean = (
        torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
        if not is_rms_norm
        else None
    )
    # 分配 rstd（1/std）存储
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    # 每特征维 < 64KB 时使用融合 kernel；超出则报错
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    # 按特征维大小启发式确定 warp 数（每 256 列用 1 个 warp，最多 8 个）
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    # grid：(行数, 归一化组数)
    grid = (M, ngroups)
    with torch.get_device_module(x.device).device(x.device.index):
        _layer_norm_fwd_1pass_kernel[grid](
            x,
            out,
            weight,
            bias,
            z,
            mean,
            rstd,
            x.stride(0),
            out.stride(0),
            z.stride(0) if z is not None else 0,
            M,
            group_size,
            eps,
            BLOCK_N=BLOCK_N,
            NORM_BEFORE_GATE=norm_before_gate,
            IS_RMS_NORM=is_rms_norm,
            num_warps=num_warps,
        )
    return out, mean, rstd


def rms_norm_gated(
    x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True,
    is_rms_norm=True
):
    # 保存原始形状，用于最终 reshape
    x_shape_og = x.shape
    # reshape input data into 2D tensor
    # 将任意高维输入展平为 2D（token, features）
    x = x.reshape(-1, x.shape[-1])
    if x.stride(-1) != 1:
        x = x.contiguous()
    if z is not None:
        assert z.shape == x_shape_og
        z = z.reshape(-1, z.shape[-1])
        if z.stride(-1) != 1:
            z = z.contiguous()
    # 确保权重和偏置连续
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    # 调用 Triton forward kernel，使用 RMSNorm 模式
    y, _, _ = _layer_norm_fwd(
        x,
        weight,
        bias,
        eps,
        z=z,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=True,
    )

    # 还原为原始形状并返回
    return y.reshape(x_shape_og)
