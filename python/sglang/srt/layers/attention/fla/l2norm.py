# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/l2norm.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# L2 归一化模块：使用 Triton kernel 对最后一维进行 L2 归一化（x / ||x||）

from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.utils import input_guard

# 候选 BT（每个 block 的 token 数）列表，用于 autotune
BT_LIST = [8, 16, 32, 64, 128]


# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8, 16, 32]
#     ],
#     key=["D"],
# )
# Triton kernel（单行版）：对特征维度 D 较大时逐行做 L2 归一化，每个程序实例处理一行
@triton.jit
def l2norm_fwd_kernel1(
    x,             # 输入张量指针
    y,             # 输出张量指针
    D,             # 特征维度大小
    BD: tl.constexpr,  # 每个 block 的特征维度（2 的幂次，>=D）
    eps,           # 防止除零的小量
):
    # 每个程序实例处理第 i_t 行
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    # Compute mean and variance
    # 加载当前行数据，越界位置填 0
    cols = tl.arange(0, BD)
    mask = cols < D
    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    # 计算 L2 范数的平方，再求倒数标准差
    b_var = tl.sum(b_x * b_x, axis=0)
    b_rstd = 1 / tl.sqrt(b_var + eps)
    # tl.store(Rstd + i_t, rstd)
    # Normalize and apply linear transformation
    # 对每个元素做归一化并存储
    b_y = b_x * b_rstd
    tl.store(y + cols, b_y, mask=mask)


# @triton.autotune(
#     configs=[
#         triton.Config({"BT": BT}, num_warps=num_warps)
#         for num_warps in [1, 2, 4, 8, 16]
#         for BT in BT_LIST
#     ],
#     key=["D", "NB"],
# )
# Triton kernel（批量版）：使用 block pointer 同时处理 BT 行，适合 D <= 512 的小特征维度
@triton.jit
def l2norm_fwd_kernel(
    x,             # 输入张量指针
    y,             # 输出张量指针
    eps,           # 防止除零的小量
    NB: tl.constexpr,  # batch 分块数（编译时常量）
    T: tl.constexpr,   # token 总数（行数）
    D: tl.constexpr,   # 特征维度
    BT: tl.constexpr,  # 每个 block 处理的行数
    BD: tl.constexpr,  # 每个 block 处理的列数（>=D 的 2 幂次）
):
    i_t = tl.program_id(0)
    # 使用 block pointer 加载 [BT, BD] 的数据块，超出边界自动填 0
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    # 沿特征维度计算每行的 L2 范数平方，并归一化
    b_var = tl.sum(b_x * b_x, axis=1)
    b_y = b_x / tl.sqrt(b_var + eps)[:, None]
    # 将归一化结果写回输出张量
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


# Python 封装：根据特征维度大小选择合适的 Triton kernel 执行 L2 归一化前向计算
def l2norm_fwd(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None
):
    # 将输入展平为二维矩阵（T, D）
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    # allocate output
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    # rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    # 限制 block 列数不超过 64KB / 元素大小
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    if D <= 512:
        # 小特征维度：使用批量处理 kernel，每 block 处理 BT=16 行
        NB = triton.cdiv(T, 2048)

        def grid(meta):
            return (triton.cdiv(T, meta["BT"]),)

        l2norm_fwd_kernel[grid](
            x,
            y,
            eps,
            NB=NB,
            T=T,
            D=D,
            BD=BD,
            BT=16,
            num_warps=8,
            num_stages=3,
        )
    else:
        # 大特征维度（>512）：逐行处理，每个程序实例负责一行
        l2norm_fwd_kernel1[(T,)](
            x,
            y,
            eps=eps,
            D=D,
            BD=BD,
            num_warps=8,
            num_stages=3,
        )

    # 恢复原始张量形状
    return y.view(x_shape_og)


# 自定义 autograd Function：封装 L2 归一化前向传播（推理时无需反向）
class L2NormFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(ctx, x, eps=1e-6, output_dtype=None):
        # 调用 Triton 加速的前向计算
        return l2norm_fwd(x, eps, output_dtype)


# 函数式接口：对张量 x 做 L2 归一化
def l2norm(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    return L2NormFunction.apply(x, eps, output_dtype)


# 别名：l2_norm 与 l2norm 等价
l2_norm = l2norm


# nn.Module 封装：将 L2 归一化包装为可配置模块
class L2Norm(nn.Module):

    def __init__(self, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.eps = eps
        self.output_dtype = output_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 调用函数式 l2norm 并返回归一化结果
        return l2norm(x, self.eps, self.output_dtype)
