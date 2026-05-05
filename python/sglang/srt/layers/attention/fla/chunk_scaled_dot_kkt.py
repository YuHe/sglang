# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_scaled_dot_kkt.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# 本模块计算 chunk 内 beta * K * K^T，用于 FLA 的 delta 规则更新中的 intra-chunk 注意力矩阵

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.index import prepare_chunk_indices
from sglang.srt.layers.attention.fla.op import safe_exp


# @triton.autotune(
#     configs=[
#         triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)
#         for BK in [32, 64, 128]
#         for num_warps in [2, 4, 8]
#         for num_stages in [2, 3, 4]
#     ],
#     key=["H", "K", "BT", "IS_VARLEN"],
# )
# Triton kernel：计算每个 chunk 内的 beta * K * K^T（下三角掩码），支持变长序列和门控衰减
@triton.jit(do_not_specialize=["T"])
def chunk_scaled_dot_kkt_fwd_kernel(
    k,             # key 张量指针，形状 [B, T, Hg, K]
    beta,          # 缩放系数指针，形状 [B, T, H]
    g_cumsum,      # 门控累积和指针，形状 [B, T, H]，可选
    A,             # 输出注意力矩阵指针，形状 [B, T, H, BT]
    cu_seqlens,    # 变长序列的累积长度指针
    chunk_indices, # 变长序列时的 chunk 索引指针
    T,             # 序列总长度（非编译时常量）
    H: tl.constexpr,   # 注意力头数
    Hg: tl.constexpr,  # key/value 的头数（可能使用 GQA）
    K: tl.constexpr,   # key 特征维度
    BT: tl.constexpr,  # chunk 大小（block token 数）
    BK: tl.constexpr,  # key 特征分块大小
    IS_VARLEN: tl.constexpr,  # 是否为变长序列
    USE_G: tl.constexpr,      # 是否使用门控累积和
):
    # 获取当前 chunk 索引和 batch*head 索引
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        # 变长序列：从 chunk_indices 中读取样本索引和 chunk 内偏移
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        # 获取当前样本的起始和结束位置
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        # 定长序列：直接按 batch 索引计算偏移
        bos, eos = i_b * T, i_b * T + T
    # chunk 内 token 位置索引，用于构造下三角掩码
    o_t = tl.arange(0, BT)

    # 加载当前 chunk 的 beta 缩放系数
    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))

    # 分块累积计算 K * K^T，避免一次性加载全部 K 维度
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        # 加载 key 分块 [BT, BK]，支持 GQA（多个 head 共享同一个 key head）
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # 累积点积：b_A += b_k @ b_k^T，形状 [BT, BT]
        b_A += tl.dot(b_k, tl.trans(b_k))

    if USE_G:
        # 加载门控累积和，并计算行列差值（衰减因子 g[i] - g[j]）
        p_g = tl.make_block_ptr(
            g_cumsum + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
        )
        b_g = tl.load(p_g, boundary_check=(0,))
        # 计算指数衰减矩阵：exp(g[i] - g[j])，对应时序门控
        b_g_diff = b_g[:, None] - b_g[None, :]
        b_A = b_A * safe_exp(b_g_diff)

    # 乘以 beta 缩放系数，并应用严格下三角掩码（只保留 i > j 的位置）
    b_A *= b_beta[:, None]
    b_A = tl.where(o_t[:, None] > o_t[None, :], b_A, 0)
    # 将结果写回输出张量 A
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (BT * H, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


# Python 封装：调用 Triton kernel 计算 chunk 内 beta * K * K^T
def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""
    Compute beta * K * K^T.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, H]`.
        g_cumsum (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H]`.
            Default: None
        cu_seqlens (torch.LongTensor):
            The cumulative sequence lengths of the input tensor.
            Default: None
        chunk_size (int):
            The chunk size. Default: 64.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float32`

    Returns:
        beta * K * K^T of shape `[B, T, H, BT]` where `BT` is the chunk size.
    """

    # 解析输入形状：B=batch, T=序列长度, Hg=key头数, K=key维度
    B, T, Hg, K = k.shape

    # H 为 beta 的头数（可能 H > Hg，即 GQA 场景）
    H = beta.shape[-1]
    BT = chunk_size
    # 变长序列时预先计算每个 chunk 的索引映射
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    # 计算总 chunk 数
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    # 分配输出张量 A，形状 [B, T, H, BT]
    A = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)
    # 启动 kernel，grid 为 (NT, B*H)
    chunk_scaled_dot_kkt_fwd_kernel[(NT, B * H)](
        k=k,
        beta=beta,
        g_cumsum=g_cumsum,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        BT=BT,
        BK=64,
        IS_VARLEN=cu_seqlens is not None,
        USE_G=g_cumsum is not None,
        num_warps=8,
        num_stages=3,
    )
    return A
