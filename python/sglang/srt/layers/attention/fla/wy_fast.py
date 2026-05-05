# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/wy_fast.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# WY 分解快速前向模块：重新计算 Delta Rule 中的 w（加权 key）和 u（加权 value）
# 利用预计算的 chunk 内下三角注意力矩阵 A 融合门控累积求和

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.index import prepare_chunk_indices


# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps, num_stages=num_stages)
#         for num_warps in [2, 4, 8]
#         for num_stages in [2, 3, 4]
#     ],
#     key=["H", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
# )
# Triton kernel：根据 chunk 内注意力矩阵 A 重新计算 w 和 u
# w = A @ (beta * g * k)，u = A @ (beta * v)，支持变长序列（IS_VARLEN）
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    k,             # key 张量指针
    v,             # value 张量指针
    beta,          # 门控衰减系数 beta
    w,             # 输出：加权 key（w = A @ (beta*g*k)）
    u,             # 输出：加权 value（u = A @ (beta*v)）
    A,             # chunk 内下三角注意力矩阵
    g,             # 门控累积对数（对数形式）
    cu_seqlens,    # 变长序列的累积长度（IS_VARLEN 时使用）
    chunk_indices, # 变长序列的 chunk 索引表
    T,             # 固定序列长度（IS_VARLEN=False 时有效）
    H: tl.constexpr,   # 注意力头数
    Hg: tl.constexpr,  # 门控头数（可与 H 不同，支持 GQA）
    K: tl.constexpr,   # key/value 特征维度
    V: tl.constexpr,   # value 特征维度
    BT: tl.constexpr,  # chunk 大小（时间块）
    BK: tl.constexpr,  # key 维度分块大小
    BV: tl.constexpr,  # value 维度分块大小
    IS_VARLEN: tl.constexpr,  # 是否为变长序列批次
):
    # 二维 grid：(chunk_id, batch * head)
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        # 变长模式：从 chunk_indices 表中读取序列编号和 chunk 局部编号
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        # 读取该序列的起止 token 位置
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        # 固定长度模式：按 batch index 计算起止位置
        bos, eos = i_b * T, i_b * T + T
    # 构造 beta 的 block pointer（形状：[BT]）
    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    # 构造累积门控 g 的 block pointer（取 exp 后作为乘性门控）
    p_g = tl.make_block_ptr(g + (bos * H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    # 构造 chunk 内注意力矩阵 A 的 block pointer（形状：[BT, BT]）
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    # 将对数门控转换为线性乘性门控
    b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))

    # 分块遍历 V 维度：计算 u = A @ (beta * v)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # 将 beta 作为逐行缩放因子应用到 v 上
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        # 矩阵乘法：u = A @ (beta * v)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    # 分块遍历 K 维度：计算 w = A @ (beta * g * k)
    for i_k in range(tl.cdiv(K, BK)):
        # GQA 支持：key 头数可能少于 value 头数，通过整除映射
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # 将 beta 和门控 g 同时应用到 k 上
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        # 矩阵乘法：w = A @ (beta * g * k)
        b_w = tl.dot(b_A, b_kb)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


# Python 封装：配置 grid 并调用 recompute_w_u_fwd_kernel，返回 w 和 u
def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
    chunk_indices: torch.LongTensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 解析张量维度
    B, T, Hg, K, V = *k.shape, v.shape[-1]
    H = v.shape[-2]
    BT = A.shape[-1]  # chunk 大小从 A 矩阵末维推断

    # 如有必要，预计算变长序列的 chunk 索引
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    # 计算总 chunk 数：定长序列用 T/BT，变长序列用 chunk_indices 行数
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = 64
    BV = 64
    # 分配输出缓冲区
    u = torch.empty_like(v)
    w = k.new_empty(B, T, H, K)
    # 启动 kernel：grid = (chunk总数, batch*head数)
    recompute_w_u_fwd_kernel[(NT, B * H)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=4,
        num_stages=3,
    )
    return w, u


# 函数别名：fwd_recompute_w_u 与 recompute_w_u_fwd 等价
fwd_recompute_w_u = recompute_w_u_fwd
