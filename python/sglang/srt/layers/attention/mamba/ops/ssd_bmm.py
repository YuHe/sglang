# Adapted from: https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/mamba/ops/ssd_bmm.py

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_bmm.py

# SSM 分块批量矩阵乘法 Triton kernel：在 SSD 算法中计算 chunk 内注意力分数矩阵 L = Q * K^T

# ruff: noqa: E501,SIM102

import math

import torch
import triton
import triton.language as tl


# Triton JIT kernel：在 SSM 分块内执行批量矩阵乘法（out = a @ b^T），支持因果掩码和序列 ID 过滤
@triton.jit
def _bmm_chunk_fwd_kernel(
    # Pointers to matrices
    a_ptr,       # 矩阵 A 的内存指针（对应 C 矩阵，形状 batch×seqlen×[ngroups×]K）
    b_ptr,       # 矩阵 B 的内存指针（对应 B 矩阵，形状同 A）
    out_ptr,     # 输出矩阵指针（形状 batch×nchunks×[ngroups×]chunk_size×chunk_size）
    seq_idx_ptr, # 序列 ID 指针（用于标记 batch 内不同序列的边界，可为 None）
    # Matrix dimensions
    seqlen,      # 序列总长度
    chunk_size,  # 每个 chunk 的大小（SSM 分块处理单位）
    K,           # 内积维度（SSM 状态空间维度 d_state）
    ngroups,     # head/group 数（多头 SSM 时的组数）
    stride_a_batch,    # A 的 batch 维步长
    stride_a_seqlen,   # A 的 seqlen 维步长
    stride_a_head,     # A 的 head/group 维步长
    stride_ak,         # A 的 K 维步长
    stride_b_batch,    # B 的 batch 维步长
    stride_b_seqlen,   # B 的 seqlen 维步长
    stride_b_head,     # B 的 head/group 维步长
    stride_bk,         # B 的 K 维步长
    stride_out_batch,  # 输出的 batch 维步长
    stride_out_chunk,  # 输出的 chunk 维步长
    stride_out_head,   # 输出的 head 维步长
    stride_outm,       # 输出的行步长（chunk_size 方向）
    stride_outn,       # 输出的列步长（chunk_size 方向）
    stride_seq_idx_batch,   # seq_idx 的 batch 步长
    stride_seq_idx_seqlen,  # seq_idx 的 seqlen 步长
    # Meta-parameters
    IS_CAUSAL: tl.constexpr,    # 是否使用因果掩码（只计算 i<=j 的部分，对应下三角）
    dot_dtype: tl.constexpr,    # 矩阵乘法内部精度（bf16/fp16/fp32）
    HAS_SEQ_IDX: tl.constexpr,  # 是否启用序列 ID 边界清零（跨序列乘积置 0）
    BLOCK_SIZE_M: tl.constexpr = 16,  # M 维（行）的分块大小
    BLOCK_SIZE_N: tl.constexpr = 16,  # N 维（列）的分块大小
    BLOCK_SIZE_K: tl.constexpr = 16,  # K 维（内积）的分块大小
):
    # axis=1: batch 索引；axis=2: chunk×group 组合索引
    pid_b = tl.program_id(axis=1)
    pid_ch = tl.program_id(axis=2).to(tl.int64)
    # 从组合索引中还原 chunk 索引和 group/head 索引
    pid_c = pid_ch // ngroups   # chunk 索引（第几个 chunk）
    pid_h = pid_ch - pid_c * ngroups  # group/head 索引
    # axis=0 编码了 (pid_m, pid_n) 二维块索引
    num_pid_n = tl.cdiv(chunk_size, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n  # 输出矩阵行块索引
    pid_n = tl.program_id(axis=0) % num_pid_n   # 输出矩阵列块索引
    # IS_CAUSAL：只计算因果有效部分（i<=j），跳过上三角块
    if IS_CAUSAL:
        if pid_n * BLOCK_SIZE_N >= (pid_m + 1) * BLOCK_SIZE_M:
            return
    # 定位当前 block 在 A、B 矩阵中对应 chunk 的起始指针
    a_ptr += (
        pid_b * stride_a_batch
        + pid_c * chunk_size * stride_a_seqlen  # 跳到当前 chunk 的起始行
        + pid_h * stride_a_head                 # 跳到当前 head/group
    )
    b_ptr += (
        pid_b * stride_b_batch
        + pid_c * chunk_size * stride_b_seqlen
        + pid_h * stride_b_head
    )
    # 若有序列 ID，也定位到当前 chunk 的起始位置
    if HAS_SEQ_IDX:
        seq_idx_ptr += (
            pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen
        )

    # 计算当前块 M/N/K 维度的偏移量（用于寻址）
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # A 指针矩阵：a[offs_m, offs_k]；B 指针矩阵：b[offs_k, offs_n]（注意 B 转置）
    a_ptrs = a_ptr + (offs_m[:, None] * stride_a_seqlen + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_b_seqlen)
    # 当前 chunk 的实际有效长度（最后一个 chunk 可能不足 chunk_size）
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    # 累加器初始化为 float32，确保数值精度
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # 沿 K 维分块迭代，执行分块矩阵乘法
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 加载 A 的当前 K 块，越界位置填 0（使用 mask 保护越界访问）
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < chunk_size_limit)
            & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        ).to(dot_dtype)
        # 加载 B 的当前 K 块（B^T 的列对应 B 的行，即 seqlen 方向）
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K)
            & (offs_n[None, :] < chunk_size_limit),
            other=0.0,
        ).to(dot_dtype)
        # tl.dot：block-level 矩阵乘法，结果累加到 acc（相当于 acc += a @ b）
        acc += tl.dot(a, b)
        # 推进 K 维分块指针
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 重新计算输出块偏移（矩阵乘法后需要对应输出位置）
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # HAS_SEQ_IDX：对来自不同序列的 token 对（i, j），将其乘积清零
    if HAS_SEQ_IDX:
        chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
        # 加载行和列对应的序列 ID
        seq_idx_m = tl.load(
            seq_idx_ptr + offs_m * stride_seq_idx_seqlen,
            mask=offs_m < chunk_size_limit,
            other=-1,   # 越界位置赋 -1（不会与有效 seq_idx 匹配）
        )
        seq_idx_n = tl.load(
            seq_idx_ptr + offs_n * stride_seq_idx_seqlen,
            mask=offs_n < chunk_size_limit,
            other=-2,   # 越界位置赋 -2（与 -1 不同，防止行列越界位置意外匹配）
        )
        # 只保留同一序列内的乘积，跨序列置 0
        acc = tl.where(seq_idx_m[:, None] == seq_idx_n[None, :], acc, 0.0)
    # 将 float32 累加结果转换为输出张量的目标 dtype
    out = acc.to(out_ptr.dtype.element_ty)

    # 定位输出张量的当前 batch/chunk/head 起始位置
    out_ptr += (
        pid_b * stride_out_batch + pid_c * stride_out_chunk + pid_h * stride_out_head
    )
    out_ptrs = out_ptr + (stride_outm * offs_m[:, None] + offs_n[None, :] * stride_outn)
    # 写出矩阵乘法结果（mask 保护边界，防止越界写入）
    tl.store(
        out_ptrs,
        out,
        mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size),
    )


# Python 封装：将输入 reshape 并调用 Triton kernel，返回 chunk 内矩阵乘法结果
def _bmm_chunk_fwd(a, b, chunk_size, seq_idx=None, causal=False, output_dtype=None):
    """
    Argument:
        a: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
        b: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
        seq_idx: (batch, seqlen) or None. out[i, j] for seq_idx[i] != seq_idx[j] will be zeroed out.
        causal: if True, then out[i, j] for i > j will be arbitrary, only out[i, j] for i <= j are
            guaranteed to be correct.
    Return:
        out: (batch, nchunks, chunk_size, chunk_size) or (batch, nchunks, ngroups, chunk_size, chunk_size)
    """
    # Check constraints.
    # 判断是否有 group 维度（4D: batch×seqlen×ngroups×K 或 3D: batch×seqlen×K）
    has_groups = a.dim() == 4
    if not has_groups:
        batch, seqlen, k = a.shape
    else:
        batch, seqlen, ngroups, k = a.shape
    assert b.shape == a.shape
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    # 确保内存连续（Triton kernel 要求 stride(-1)==1 或 stride(1)==1）
    if a.stride(-1) != 1 and a.stride(1) != 1:
        a = a.contiguous()
    if b.stride(-1) != 1 and b.stride(1) != 1:
        b = b.contiguous()
    # 计算 chunk 数（向上取整）
    nchunks = math.ceil(seqlen / chunk_size)
    # Allocates output.
    # 推断输出 dtype（默认与 a 相同，可通过 output_dtype 覆盖）
    out_dtype = a.dtype if output_dtype is None else output_dtype
    # 分配输出张量：无 group 时 4D，有 group 时 5D
    out = torch.empty(
        (
            (batch, nchunks, chunk_size, chunk_size)
            if not has_groups
            else (batch, nchunks, ngroups, chunk_size, chunk_size)
        ),
        device=a.device,
        dtype=out_dtype,
    )
    # 根据输入 dtype 选择内积精度（bf16 > fp16 > fp32 优先级）
    dot_dtype = (
        tl.bfloat16
        if a.dtype == torch.bfloat16 or b.dtype == torch.bfloat16
        else (
            tl.float16
            if a.dtype == torch.float16 or b.dtype == torch.float16
            else tl.float32
        )
    )
    # 启动 grid：axis0 = M块数*N块数，axis1 = batch，axis2 = nchunks*ngroups
    grid = lambda META: (
        triton.cdiv(chunk_size, META["BLOCK_SIZE_M"])
        * triton.cdiv(chunk_size, META["BLOCK_SIZE_N"]),
        batch,
        nchunks if not has_groups else nchunks * ngroups,
    )
    with torch.get_device_module(a.device).device(a.device.index):
        _bmm_chunk_fwd_kernel[grid](
            a,
            b,
            out,
            seq_idx,
            seqlen,
            chunk_size,
            k,
            ngroups if has_groups else 1,
            a.stride(0),
            a.stride(1),
            0 if not has_groups else a.stride(2),  # head stride（无 group 时为 0）
            a.stride(-1),
            b.stride(0),
            b.stride(1),
            0 if not has_groups else b.stride(2),
            b.stride(-1),
            out.stride(0),
            out.stride(1),
            0 if not has_groups else out.stride(2),
            out.stride(-2),
            out.stride(-1),
            *(
                (seq_idx.stride(0), seq_idx.stride(1))
                if seq_idx is not None
                else (0, 0)
            ),
            causal,
            dot_dtype,
            HAS_SEQ_IDX=seq_idx is not None,
        )
    return out
