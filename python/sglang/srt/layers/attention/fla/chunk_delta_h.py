# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_delta_h.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# 本模块计算 Gated Delta Rule 的跨 chunk 递归状态更新（h）
# 核心算法：h_{t+1} = g_t * h_t + k_t^T * (v_t - w_t * h_t)，支持 K<=256 的分块实现

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from sglang.srt.layers.attention.fla.op import exp, safe_exp
from sglang.srt.layers.attention.fla.utils import is_nvidia_hopper

# Hopper 架构的 warp 配置不同（更少选项），其他架构支持更多 warp 数
NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8, 16]
CHUNK_SIZE = 64


# @triton.autotune(
#     configs=[
#         triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
#         for num_warps in [2, 4]
#         for num_stages in [2, 3, 4]
#         for BV in [32, 64]
#     ],
#     key=["H", "K", "V", "BT", "USE_G"],
#     use_cuda_graph=use_cuda_graph,
# )
# Triton kernel：以 K 维度 64 为固定分块，计算 delta rule 的跨 chunk 递归状态 h
# 每个程序实例负责一个 (value 分块, batch*head) 对
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,              # key 张量指针 [B, T, Hg, K]
    v,              # value/u 张量指针 [B, T, H, V]
    w,              # 写权重张量指针 [B, T, H, K]
    v_new,          # 修正后的 value 输出指针（可选保存）
    g,              # 门控累积和指针 [B, T, H]，可选
    gk,             # 按 key 维度的门控指针，可选
    h,              # 输出状态指针 [B, NT, H, V, K]
    initial_state,  # 初始状态指针 [N, H, V, K]
    initial_state_indices,  # 初始状态索引指针
    cu_seqlens,     # 变长序列累积长度
    chunk_offsets,  # 变长序列的 chunk 偏移
    T,              # 序列长度
    H: tl.constexpr,   # 注意力头数
    Hg: tl.constexpr,  # key/value 头数（GQA）
    K: tl.constexpr,   # key 维度
    V: tl.constexpr,   # value 维度
    BT: tl.constexpr,  # chunk 大小
    BV: tl.constexpr,  # value 分块大小
    USE_G: tl.constexpr,              # 是否使用标量门控
    USE_GK: tl.constexpr,             # 是否使用 key 维度门控
    USE_INITIAL_STATE: tl.constexpr,  # 是否使用初始状态
    INPLACE_UPDATE: tl.constexpr,     # 是否原地更新最终状态
    SAVE_NEW_VALUE: tl.constexpr,     # 是否保存修正后的 value
    IS_VARLEN: tl.constexpr,          # 是否为变长序列
):
    # 获取当前 value 分块索引和 batch*head 索引
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        # 变长序列：读取当前样本的起止位置和 chunk 基础偏移
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BV, BK] 初始化递归状态 h，按 K 维度分成最多 4 个 64 宽分块
    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([BV, 64], dtype=tl.float32)

    # calculate offset 计算各张量的起始内存偏移
    h += ((boh * H + i_h) * V * K).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    if SAVE_NEW_VALUE:
        v_new += ((bos * H + i_h) * V).to(tl.int64)
    # 各张量的 token 步长（沿 T 维度的跳跃大小）
    stride_v = H * V
    stride_h = H * V * K
    stride_k = Hg * K
    stride_w = H * K

    # 读取当前样本对应的初始/最终状态指针
    index = tl.load(initial_state_indices + i_n).to(tl.int32)
    h0 = initial_state + index * stride_h
    ht = initial_state + index * stride_h
    if USE_INITIAL_STATE:
        h0 = h0 + i_h * V * K
    if INPLACE_UPDATE:
        ht = ht + i_h * V * K

    # load initial state 加载初始状态到 b_h 寄存器（最多 4 个 K 分块）
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
            )
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
            )
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # main recurrence 主递归循环：逐 chunk 更新状态 h
    for i_t in range(NT):
        # 将当前状态 b_h 存储到输出张量 h（时间步 i_t 的状态快照）
        p_h1 = tl.make_block_ptr(
            h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)
        )
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(
                h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(
                h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
            )
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(
                h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
            )
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        # 计算 v_delta = v - w @ h（delta rule 的修正量），分多 K 块累积
        p_w = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
        # 加载原始 v，计算 v_delta = v - w @ h
        p_v = tl.make_block_ptr(
            v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        # 可选：保存修正后的 value（v_new = v - w @ h）
        if SAVE_NEW_VALUE:
            p_v = tl.make_block_ptr(
                v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            )
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        # 获取当前 chunk 最后一个有效位置的索引
        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            # 加载 chunk 末尾的标量门控值 g_last
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,))
            # 对 v_delta 施加相对时序衰减：exp(g_last - g[t])
            b_v = b_v * safe_exp(b_g_last - b_g)[:, None]
            b_g_last = exp(b_g_last)
            # 对状态 h 施加全局门控衰减：h *= exp(g_last)
            b_h1 = b_h1 * b_g_last
            if K > 64:
                b_h2 = b_h2 * b_g_last
            if K > 128:
                b_h3 = b_h3 * b_g_last
            if K > 192:
                b_h4 = b_h4 * b_g_last

        if USE_GK:
            # 按 key 维度分别加载门控并对状态 h 施加衰减
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k1,
                mask=(o_k1 < K),
                other=0.0,
            )
            b_h1 *= exp(b_gk_last1)[None, :]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                    mask=(o_k2 < K),
                    other=0.0,
                )
                b_h2 *= exp(b_gk_last2)[None, :]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k3,
                    mask=(o_k3 < K),
                    other=0.0,
                )
                b_h3 *= exp(b_gk_last3)[None, :]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k4,
                    mask=(o_k4 < K),
                    other=0.0,
                )
                b_h4 *= exp(b_gk_last4)[None, :]
        b_v = b_v.to(k.dtype.element_ty)

        # 更新状态 h += k^T @ v_delta（外积累积）
        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.trans(tl.dot(b_k, b_v))
        if K > 64:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.trans(tl.dot(b_k, b_v))
        if K > 128:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.trans(tl.dot(b_k, b_v))
        if K > 192:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.trans(tl.dot(b_k, b_v))

    # epilogue 尾声：将最终状态写回 initial_state（原地更新）
    if INPLACE_UPDATE:
        p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


# Python 封装：调用 Triton kernel 完成跨 chunk 的状态更新和 v_new 计算
def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,                            # key 张量 [B, T, Hg, K]
    w: torch.Tensor,                            # 写权重张量 [B, T, H, K]
    u: torch.Tensor,                            # 更新值张量 [B, T, H, V]
    g: Optional[torch.Tensor] = None,           # 标量门控
    gk: Optional[torch.Tensor] = None,          # key 维度门控
    initial_state: Optional[torch.Tensor] = None,             # 初始状态
    initial_state_indices: Optional[torch.Tensor] = None,     # 初始状态索引
    save_new_value: bool = True,                # 是否保存 v_new
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 解析输入形状，H 为总头数，Hg 为 key/value 头数（GQA）
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = CHUNK_SIZE

    # 预计算 chunk 索引和 chunk 偏移
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            prepare_chunk_offsets(cu_seqlens, BT),
        )
    # 当前 kernel 仅支持 K <= 256 的情况
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    # 分配输出状态张量 h，形状 [B, NT, H, V, K]
    h = k.new_empty(B, NT, H, V, K)

    # 可选：分配修正 value 张量 v_new
    v_new = torch.empty_like(u) if save_new_value else None

    # 动态 grid：第一维按 value 分块大小划分
    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,        # 注意：u 作为 v 传入 kernel
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BV=32,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_INITIAL_STATE=initial_state is not None,
        INPLACE_UPDATE=True,
        SAVE_NEW_VALUE=v_new is not None,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=4,
        num_stages=2,
    )
    return h, v_new
