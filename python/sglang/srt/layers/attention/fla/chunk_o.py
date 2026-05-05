# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_o.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# chunk_o：计算 FLA chunk 级别的输出 o = scale * (q @ h^T * exp(g) + A * v)
# 其中 h 为跨 chunk 传递的隐状态，A 为 chunk 内因果注意力矩阵（含门控）

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.index import prepare_chunk_indices
from sglang.srt.layers.attention.fla.op import exp, safe_exp
from sglang.srt.layers.attention.fla.utils import check_shared_mem, is_nvidia_hopper

# 根据共享内存大小和 GPU 架构选择合适的分块尺寸和 warp 数
BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]
NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8]


# @triton.autotune(
#     configs=[
#         triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
#         for BK in BKV_LIST
#         for BV in BKV_LIST
#         for num_warps in NUM_WARPS
#         for num_stages in [2, 3, 4]
#     ],
#     key=["H", "K", "V", "BT"],
# )
# Triton kernel：计算单个 chunk 的输出 o
# o = scale * (q @ h^T * exp(g) + (causal_mask * q @ k^T * safe_exp(g_diff)) @ v)
@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_o(
    q,             # query 张量指针
    k,             # key 张量指针
    v,             # value 张量指针
    h,             # 跨 chunk 隐状态张量指针（形状：[NT*H, K, V]）
    g,             # 累积门控对数（cumsum of log decay），可选
    o,             # 输出张量指针
    cu_seqlens,    # 变长序列边界（IS_VARLEN 时使用）
    chunk_indices, # 变长序列 chunk 索引表
    scale,         # 缩放因子（通常为 K^{-0.5}）
    T,             # 序列长度
    H: tl.constexpr,   # 注意力头数
    Hg: tl.constexpr,  # 门控头数（GQA 支持）
    K: tl.constexpr,   # key 特征维度
    V: tl.constexpr,   # value 特征维度
    BT: tl.constexpr,  # chunk 大小
    BK: tl.constexpr,  # key 分块大小
    BV: tl.constexpr,  # value 分块大小
    USE_G: tl.constexpr,      # 是否使用门控
    IS_VARLEN: tl.constexpr,  # 是否为变长序列
):
    # 三维 grid：(value块, chunk, batch*head)
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        # 变长模式：从 chunk_indices 中读取序列和 chunk 局部编号
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        # 固定长度模式：按 batch index 线性映射
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    # 将各张量指针移动到当前 (batch, head) 对应的起始位置
    q += (bos * Hg + i_h // (H // Hg)) * K
    k += (bos * Hg + i_h // (H // Hg)) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    # 隐状态 h 按 (全局chunk编号 * 头数 + 头编号) 寻址
    h += (i_tg * H + i_h).to(tl.int64) * V * K

    # 初始化输出块和 chunk 内注意力矩阵
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    # 沿 K 维分块累积：b_o += q @ h^T，b_A += q @ k^T
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k, (K, T), (1, Hg * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1)
        )
        p_h = tl.make_block_ptr(
            h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0)
        )
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))

        # [BT, BK] @ [BK, BV] -> [BT, BV]
        # q 与隐状态 h 的乘积（跨 chunk 贡献）
        b_o += tl.dot(b_q, tl.trans(b_h))
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        # chunk 内 q@k^T（因果注意力矩阵）
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        # 应用门控：b_o *= exp(g)，b_A *= safe_exp(g_i - g_j)（因果差分门控）
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_o = b_o * exp(b_g)[:, None]
        # safe_exp 确保 g_i < g_j 时置零（因果性保证）
        b_A = b_A * safe_exp(b_g[:, None] - b_g[None, :])

    # 应用因果掩码：只保留下三角部分（i >= j）
    o_i = tl.arange(0, BT)
    m_A = o_i[:, None] >= o_i[None, :]
    b_A = tl.where(m_A, b_A, 0)

    p_v = tl.make_block_ptr(
        v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    p_o = tl.make_block_ptr(
        o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    b_v = tl.load(p_v, boundary_check=(0, 1))

    # to fix mma -> mma layout conversion
    # already solved by triton v3.2 or higher
    # 最终输出：scale * (跨chunk贡献 + chunk内注意力贡献)
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


# Python 封装：分配输出张量并启动 chunk_fwd_kernel_o
def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: Optional[torch.Tensor] = None,  # cumsum of log decay
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    # 解析张量形状
    B, T, Hg, K, V = *q.shape, v.shape[-1]
    H = v.shape[-2]
    # chunk 大小取 chunk_size 与 T 的最小 2 幂次的较小值
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    # 变长序列时预计算 chunk 索引
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        # 默认缩放因子为 K^{-0.5}
        scale = k.shape[-1] ** -0.5

    o = torch.zeros_like(v)

    # grid = (V分块数, chunk总数, batch*头数)
    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * H)

    chunk_fwd_kernel_o[grid](
        q,
        k,
        v,
        h,
        g,
        o,
        cu_seqlens,
        chunk_indices,
        scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=128,
        BV=64,
        USE_G=g is not None,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=4,
        num_stages=2,
    )
    return o
