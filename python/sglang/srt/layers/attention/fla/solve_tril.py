# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/solve_tril.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# 本模块实现严格下三角矩阵的逆矩阵计算：(I + A)^{-1}，其中 A 为严格下三角矩阵
# 算法：采用分层块合并策略（16->32->64），先求各 16x16 对角块的逆，再合并

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.index import prepare_chunk_indices
from sglang.srt.layers.attention.fla.utils import input_guard


# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps, num_stages=num_stages)
#         for num_warps in [1, 2, 4, 8]
#         for num_stages in [2, 3, 4, 5]
#     ],
#     key=["BT"],
# )
# Triton kernel：对每个 16x16 对角块进行前代换，计算 (I + A_diag)^{-1}
@triton.jit(do_not_specialize=["T"])
def solve_tril_16x16_kernel(
    A,             # 输入矩阵指针 [B, T, H, BT]（严格下三角）
    Ad,            # 输出对角块逆矩阵指针 [B, T, H, 16]
    cu_seqlens,    # 变长序列累积长度
    chunk_indices, # 变长序列 chunk 索引
    T,             # 序列长度
    H: tl.constexpr,          # 注意力头数
    BT: tl.constexpr,         # 完整 chunk 大小（16/32/64）
    IS_VARLEN: tl.constexpr,  # 是否为变长序列
):
    # 每个程序实例处理第 (i_t, i_bh) 个 16x16 对角块
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # 调整指针到当前序列的对角块位置
    A = A + (bos * H + i_h) * BT
    Ad = Ad + (bos * H + i_h) * 16

    # offset：当前 16x16 块在 BT 维度内的列偏移（对角块在 A 矩阵中的列起始位置）
    offset = (i_t * 16) % BT
    p_A = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * 16, offset), (16, 16), (1, 0)
    )
    p_Ai = tl.make_block_ptr(Ad, (T, 16), (H * 16, 1), (i_t * 16, 0), (16, 16), (1, 0))
    # 加载 16x16 对角块，并取严格下三角部分（取反，因为求解 (I+A)^{-1}）
    b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
    b_A = -tl.where(tl.arange(0, 16)[:, None] > tl.arange(0, 16)[None, :], b_A, 0)

    o_i = tl.arange(0, 16)
    # 前代换：逐行更新逆矩阵（每次处理一个新行）
    for i in range(1, min(16, T - i_t * 16)):
        # 提取第 i 行的原始 A 数据（从 HBM 读取）
        b_a = -tl.load(A + (i_t * 16 + i) * H * BT + o_i + offset)
        # 用已计算的逆矩阵行更新当前行
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
        mask = o_i == i
        # 将第 i 行写入中间逆矩阵
        b_A = tl.where(mask[:, None], b_a, b_A)
    # 加上单位矩阵（对角线置 1）
    b_A += o_i[:, None] == o_i[None, :]
    tl.store(
        p_Ai,
        b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps, num_stages=num_stages)
#         for num_warps in [1, 2, 4, 8]
#         for num_stages in [2, 3, 4, 5]
#     ],
#     key=["H", "BT", "IS_VARLEN"],
# )
# Triton kernel：将两个 16x16 逆矩阵块合并为 32x32 的逆矩阵
# 使用分块矩阵求逆公式：Ai_21 = -Ai_22 @ A_21 @ Ai_11
@triton.jit(do_not_specialize=["T"])
def merge_16x16_to_32x32_inverse_kernel(
    A,             # 原始矩阵指针（含跨块部分）
    Ad,            # 对角块逆矩阵指针 [B, T, H, 16]
    Ai,            # 输出完整逆矩阵指针 [B, T, H, 32]
    cu_seqlens,    # 变长序列累积长度
    chunk_indices, # chunk 索引
    T,             # 序列长度
    H: tl.constexpr,
    BT: tl.constexpr,         # = 32
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # 调整指针到当前序列的起始位置
    A += (bos * H + i_h) * 32
    Ad += (bos * H + i_h) * 16
    Ai += (bos * H + i_h) * 32

    # 加载 32x32 块的跨块部分 A_21（下半块对上半块的注意力）
    p_A_21 = tl.make_block_ptr(
        A, (T, 32), (H * 32, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0)
    )
    # 加载两个 16x16 对角块的逆
    p_Ad_11 = tl.make_block_ptr(
        Ad, (T, 16), (H * 16, 1), (i_t * 32, 0), (16, 16), (1, 0)
    )
    p_Ad_22 = tl.make_block_ptr(
        Ad, (T, 16), (H * 16, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0)
    )
    # 输出 32x32 逆矩阵的三个子块指针（上三角为 0，只需存下三角和对角）
    p_Ai_11 = tl.make_block_ptr(
        Ai, (T, 32), (H * 32, 1), (i_t * 32, 0), (16, 16), (1, 0)
    )
    p_Ai_22 = tl.make_block_ptr(
        Ai, (T, 32), (H * 32, 1), (i_t * 32 + 16, 16), (16, 16), (1, 0)
    )
    p_Ai_21 = tl.make_block_ptr(
        Ai, (T, 32), (H * 32, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0)
    )

    A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    Ai_11 = tl.load(p_Ad_11, boundary_check=(0, 1)).to(tl.float32)
    Ai_22 = tl.load(p_Ad_22, boundary_check=(0, 1)).to(tl.float32)
    # 块合并公式：Ai_21 = -Ai_22 @ A_21 @ Ai_11
    Ai_21 = -tl.dot(
        tl.dot(Ai_22, A_21, input_precision="ieee"), Ai_11, input_precision="ieee"
    )
    tl.store(
        p_Ai_11,
        Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_22,
        Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_21,
        Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps, num_stages=num_stages)
#         for num_warps in [2, 4, 8]
#         for num_stages in [2, 3, 4, 5]
#     ],
#     key=["H", "BT", "IS_VARLEN"],
# )
# Triton kernel：将四个 16x16 逆矩阵块合并为 64x64 的完整逆矩阵
# 使用分块矩阵求逆的递归公式（一阶->二阶->三阶跨块项）
@triton.jit(do_not_specialize=["T"])
def merge_16x16_to_64x64_inverse_kernel(
    A,             # 原始矩阵指针（含所有跨块部分）
    Ad,            # 对角块逆矩阵指针 [B, T, H, 16]（4 个 16x16 块）
    Ai,            # 输出完整逆矩阵指针 [B, T, H, 64]
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,         # = 64
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    A += (bos * H + i_h) * 64
    Ad += (bos * H + i_h) * 16
    Ai += (bos * H + i_h) * 64

    # 加载 6 个跨块部分 A_ij（i>j 的严格下三角块）
    p_A_21 = tl.make_block_ptr(
        A, (T, 64), (H * 64, 1), (i_t * 64 + 16, 0), (16, 16), (1, 0)
    )
    p_A_32 = tl.make_block_ptr(
        A, (T, 64), (H * 64, 1), (i_t * 64 + 32, 16), (16, 16), (1, 0)
    )
    p_A_31 = tl.make_block_ptr(
        A, (T, 64), (H * 64, 1), (i_t * 64 + 32, 0), (16, 16), (1, 0)
    )
    p_A_43 = tl.make_block_ptr(
        A, (T, 64), (H * 64, 1), (i_t * 64 + 48, 32), (16, 16), (1, 0)
    )
    p_A_42 = tl.make_block_ptr(
        A, (T, 64), (H * 64, 1), (i_t * 64 + 48, 16), (16, 16), (1, 0)
    )
    p_A_41 = tl.make_block_ptr(
        A, (T, 64), (H * 64, 1), (i_t * 64 + 48, 0), (16, 16), (1, 0)
    )
    # 加载 4 个 16x16 对角块的逆矩阵
    p_Ad_11 = tl.make_block_ptr(
        Ad, (T, 16), (H * 16, 1), (i_t * 64, 0), (16, 16), (1, 0)
    )
    p_Ad_22 = tl.make_block_ptr(
        Ad, (T, 16), (H * 16, 1), (i_t * 64 + 16, 0), (16, 16), (1, 0)
    )
    p_Ad_33 = tl.make_block_ptr(
        Ad, (T, 16), (H * 16, 1), (i_t * 64 + 32, 0), (16, 16), (1, 0)
    )
    p_Ad_44 = tl.make_block_ptr(
        Ad, (T, 16), (H * 16, 1), (i_t * 64 + 48, 0), (16, 16), (1, 0)
    )

    A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    A_32 = tl.load(p_A_32, boundary_check=(0, 1)).to(tl.float32)
    A_31 = tl.load(p_A_31, boundary_check=(0, 1)).to(tl.float32)
    A_43 = tl.load(p_A_43, boundary_check=(0, 1)).to(tl.float32)
    A_42 = tl.load(p_A_42, boundary_check=(0, 1)).to(tl.float32)
    A_41 = tl.load(p_A_41, boundary_check=(0, 1)).to(tl.float32)

    Ai_11 = tl.load(p_Ad_11, boundary_check=(0, 1)).to(tl.float32)
    Ai_22 = tl.load(p_Ad_22, boundary_check=(0, 1)).to(tl.float32)
    Ai_33 = tl.load(p_Ad_33, boundary_check=(0, 1)).to(tl.float32)
    Ai_44 = tl.load(p_Ad_44, boundary_check=(0, 1)).to(tl.float32)

    # 一阶跨块逆：Ai_ij = -Ai_ii @ A_ij @ Ai_jj（相邻块）
    Ai_21 = -tl.dot(
        tl.dot(Ai_22, A_21, input_precision="ieee"), Ai_11, input_precision="ieee"
    )
    Ai_32 = -tl.dot(
        tl.dot(Ai_33, A_32, input_precision="ieee"), Ai_22, input_precision="ieee"
    )
    Ai_43 = -tl.dot(
        tl.dot(Ai_44, A_43, input_precision="ieee"), Ai_33, input_precision="ieee"
    )

    # 二阶跨块逆（间隔一块）：需利用一阶结果
    Ai_31 = -tl.dot(
        Ai_33,
        tl.dot(A_31, Ai_11, input_precision="ieee")
        + tl.dot(A_32, Ai_21, input_precision="ieee"),
        input_precision="ieee",
    )
    Ai_42 = -tl.dot(
        Ai_44,
        tl.dot(A_42, Ai_22, input_precision="ieee")
        + tl.dot(A_43, Ai_32, input_precision="ieee"),
        input_precision="ieee",
    )
    # 三阶跨块逆（间隔两块）：需利用一阶和二阶结果
    Ai_41 = -tl.dot(
        Ai_44,
        tl.dot(A_41, Ai_11, input_precision="ieee")
        + tl.dot(A_42, Ai_21, input_precision="ieee")
        + tl.dot(A_43, Ai_31, input_precision="ieee"),
        input_precision="ieee",
    )

    # 存储 10 个非零块（4 个对角 + 6 个下三角跨块）
    p_Ai_11 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64, 0), (16, 16), (1, 0)
    )
    p_Ai_22 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 16, 16), (16, 16), (1, 0)
    )
    p_Ai_33 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 32, 32), (16, 16), (1, 0)
    )
    p_Ai_44 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 48, 48), (16, 16), (1, 0)
    )
    p_Ai_21 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 16, 0), (16, 16), (1, 0)
    )
    p_Ai_31 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 32, 0), (16, 16), (1, 0)
    )
    p_Ai_32 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 32, 16), (16, 16), (1, 0)
    )
    p_Ai_41 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 48, 0), (16, 16), (1, 0)
    )
    p_Ai_42 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 48, 16), (16, 16), (1, 0)
    )
    p_Ai_43 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 48, 32), (16, 16), (1, 0)
    )
    tl.store(
        p_Ai_11,
        Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_22,
        Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_33,
        Ai_33.to(p_Ai_33.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_44,
        Ai_44.to(p_Ai_44.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_21,
        Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_31,
        Ai_31.to(p_Ai_31.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_32,
        Ai_32.to(p_Ai_32.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_41,
        Ai_41.to(p_Ai_41.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_42,
        Ai_42.to(p_Ai_42.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_43,
        Ai_43.to(p_Ai_43.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )

    # 将上三角部分（6 个零块）显式写入输出（确保内存初始化正确）
    fill_zeros = tl.zeros((16, 16), dtype=tl.float32)
    p_Ai_12 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64, 16), (16, 16), (1, 0)
    )
    p_Ai_13 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64, 32), (16, 16), (1, 0)
    )
    p_Ai_14 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64, 48), (16, 16), (1, 0)
    )
    p_Ai_23 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 16, 32), (16, 16), (1, 0)
    )
    p_Ai_24 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 16, 48), (16, 16), (1, 0)
    )
    p_Ai_34 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 32, 48), (16, 16), (1, 0)
    )
    tl.store(
        p_Ai_12,
        fill_zeros.to(p_Ai_12.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_13,
        fill_zeros.to(p_Ai_13.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_14,
        fill_zeros.to(p_Ai_14.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_23,
        fill_zeros.to(p_Ai_23.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_24,
        fill_zeros.to(p_Ai_24.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_34,
        fill_zeros.to(p_Ai_34.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


# Python 封装：调用分层 Triton kernel 计算下三角矩阵逆 (I + A)^{-1}
@input_guard
def solve_tril(
    A: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """
    Compute the inverse of the lower triangular matrix
    A should be strictly lower triangular, i.e., A.triu() == 0.

    Args:
        A (torch.Tensor):
            [B, T, H, K]
        cu_seqlens (torch.Tensor):
            The cumulative sequence lengths of the input tensor.
            Default: None.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float`

    Returns:
        (I + A)^-1 with the same shape as A
    """
    # 当前仅支持 BT = 16/32/64
    assert A.shape[-1] in [16, 32, 64]

    B, T, H, BT = A.shape
    # 第一步：计算各 16x16 对角块的逆（BT=16 时直接返回）
    Ad = torch.empty(
        B, T, H, 16, device=A.device, dtype=torch.float if BT != 16 else output_dtype
    )

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, 16) if cu_seqlens is not None else None
    )
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, 16)
    solve_tril_16x16_kernel[NT, B * H](
        A=A,
        Ad=Ad,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=1,
        num_stages=4,
    )
    # BT=16 时无需块合并，直接返回
    if BT == 16:
        return Ad

    # 第二步：块合并，将多个 16x16 逆矩阵组合为完整的 BT x BT 逆矩阵
    Ai = torch.empty(B, T, H, BT, device=A.device, dtype=output_dtype)
    # 根据 BT 大小选择合并 kernel：32x32 或 64x64
    merge_fn = (
        merge_16x16_to_32x32_inverse_kernel
        if BT == 32
        else merge_16x16_to_64x64_inverse_kernel
    )
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    merge_fn[NT, B * H](
        A=A,
        Ad=Ad,
        Ai=Ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=4,
        num_stages=3,
    )
    return Ai
