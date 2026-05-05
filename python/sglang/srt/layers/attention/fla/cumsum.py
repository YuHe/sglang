# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/cumsum.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# 本模块提供 chunk 内局部前缀累积和（cumsum）操作，支持标量和向量两种模式
# 用于 FLA 中计算门控衰减的 chunk 级别累积和，支持正向和反向累积

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.index import prepare_chunk_indices
from sglang.srt.layers.attention.fla.utils import check_shared_mem, input_guard

# 根据共享内存大小选择向量累积分块候选值
BS_LIST = [32, 64] if check_shared_mem() else [16, 32]


# @triton.autotune(
#     configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
#     key=["B", "H", "BT", "IS_VARLEN", "REVERSE"],
# )
# Triton kernel：对标量门控值（每个 token 一个值）在 chunk 内做前缀累积和
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_scalar_kernel(
    s,             # 输入张量指针
    o,             # 输出张量指针
    scale,         # 可选的缩放系数
    cu_seqlens,    # 变长序列的累积长度指针
    chunk_indices, # 变长序列时的 chunk 索引指针
    T,             # 序列总长度（非编译时常量）
    B: tl.constexpr,           # batch 大小
    H: tl.constexpr,           # 注意力头数
    BT: tl.constexpr,          # chunk 大小
    REVERSE: tl.constexpr,     # 是否反向累积
    HAS_SCALE: tl.constexpr,   # 是否应用缩放
    IS_VARLEN: tl.constexpr,   # 是否为变长序列
    HEAD_FIRST: tl.constexpr,  # 输入是否为 head-first 格式 [B,H,T]
):
    # 获取当前 chunk 和 batch*head 的索引
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        # 变长序列：从 chunk_indices 中解析样本和 chunk 偏移
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        # 定长序列：按 batch 索引计算偏移
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        # head-first 格式：内存布局为 [B, H, T]，步长为 1
        p_s = tl.make_block_ptr(
            s + bos * H + i_h * T, (T,), (1,), (i_t * BT,), (BT,), (0,)
        )
        p_o = tl.make_block_ptr(
            o + bos * H + i_h * T, (T,), (1,), (i_t * BT,), (BT,), (0,)
        )
    else:
        # time-first 格式：内存布局为 [B, T, H]，步长为 H
        p_s = tl.make_block_ptr(s + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    # [BT] 加载当前 chunk 的标量门控值并计算前缀累积和
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    if REVERSE:
        # 反向累积：将正向累积转为从末尾开始的后缀累积
        b_z = tl.sum(b_s, axis=0)
        b_o = -b_o + b_z[None] + b_s
    if HAS_SCALE:
        # 可选缩放
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


# autotune 配置：对向量模式的分块大小和 warp 数进行自动调优
@triton.autotune(
    configs=[
        triton.Config({"BS": BS}, num_warps=num_warps)
        for BS in BS_LIST
        for num_warps in [2, 4, 8]
    ],
    key=["B", "H", "S", "BT", "IS_VARLEN", "REVERSE", "HAS_SCALE"],
)
# Triton kernel：对向量门控值（每个 token 含 S 维向量）在 chunk 内做前缀累积和
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_vector_kernel(
    s,             # 输入张量指针
    o,             # 输出张量指针
    scale,         # 可选缩放系数
    cu_seqlens,    # 变长序列累积长度
    chunk_indices, # 变长序列 chunk 索引
    T,             # 序列总长度
    B: tl.constexpr,           # batch 大小
    H: tl.constexpr,           # 注意力头数
    S: tl.constexpr,           # 向量维度
    BT: tl.constexpr,          # chunk 大小
    BS: tl.constexpr,          # 向量分块大小（autotune）
    REVERSE: tl.constexpr,     # 是否反向累积
    HAS_SCALE: tl.constexpr,   # 是否应用缩放
    IS_VARLEN: tl.constexpr,   # 是否为变长序列
    HEAD_FIRST: tl.constexpr,  # 输入是否为 head-first 格式
):
    # 三维 grid：(向量分块索引, chunk索引, batch*head索引)
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
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

    # 构造累积掩码矩阵：m_s[i,j]=1 表示位置 j 对位置 i 有贡献
    o_i = tl.arange(0, BT)
    if REVERSE:
        # 反向模式：保留上三角（包含对角线），即 j >= i 的位置
        m_s = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)
    else:
        # 正向模式：保留下三角（包含对角线），即 j <= i 的位置
        m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)

    if HEAD_FIRST:
        # head-first 格式的 block pointer
        p_s = tl.make_block_ptr(
            s + (bos * H + i_h * T) * S,
            (T, S),
            (S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
        p_o = tl.make_block_ptr(
            o + (bos * H + i_h * T) * S,
            (T, S),
            (S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
    else:
        # time-first 格式的 block pointer
        p_s = tl.make_block_ptr(
            s + (bos * H + i_h) * S,
            (T, S),
            (H * S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
        p_o = tl.make_block_ptr(
            o + (bos * H + i_h) * S,
            (T, S),
            (H * S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
    # [BT, BS] 使用矩阵乘法实现累积和：m_s @ b_s 等价于前缀累积
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_o = tl.dot(m_s, b_s, allow_tf32=False)
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


# Python 封装：对标量门控（形状 [B,T,H] 或 [B,H,T]）执行 chunk 内局部前缀累积和
def chunk_local_cumsum_scalar(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float,
    chunk_indices: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    # 解析输入形状，支持 head-first 和 time-first 两种格式
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    # 验证 chunk_size 必须是 2 的幂次
    assert chunk_size == 2 ** (
        chunk_size.bit_length() - 1
    ), "chunk_size must be a power of 2"
    BT = chunk_size
    # 变长序列时预计算 chunk 索引
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    # 保留原始输入，创建与输出类型匹配的空张量
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    grid = (NT, B * H)
    chunk_local_cumsum_scalar_kernel[grid](
        s=g_org,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        BT=BT,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
        HAS_SCALE=scale is not None,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=8,
        num_stages=3,
    )
    return g


# Python 封装：对向量门控（形状 [B,T,H,S] 或 [B,H,T,S]）执行 chunk 内局部前缀累积和
def chunk_local_cumsum_vector(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float,
    chunk_indices: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    # 解析输入形状，S 为向量维度
    if head_first:
        B, H, T, S = g.shape
    else:
        B, T, H, S = g.shape
    BT = chunk_size
    # 变长序列时预计算 chunk 索引
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    assert chunk_size == 2 ** (
        chunk_size.bit_length() - 1
    ), "chunk_size must be a power of 2"

    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)

    # 动态 grid：根据向量分块大小 BS 确定第一维大小
    def grid(meta):
        return (triton.cdiv(meta["S"], meta["BS"]), NT, B * H)

    # keep cumulative normalizer in fp32
    # this kernel is equivalent to
    # g = g.view(B, H, NT, BT, -1).cumsum(-2).view(B, H, T, -1)
    chunk_local_cumsum_vector_kernel[grid](
        s=g_org,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        S=S,
        BT=BT,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
        HAS_SCALE=scale is not None,
        IS_VARLEN=cu_seqlens is not None,
    )
    return g


# 统一入口：根据输入维度自动选择标量或向量累积模式
@input_guard
def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float,
    chunk_indices: Optional[torch.LongTensor] = None,
    **kwargs,
) -> torch.Tensor:
    # 变长序列时仅支持 batch_size=1
    if cu_seqlens is not None:
        assert (
            g.shape[0] == 1
        ), "Only batch size 1 is supported when cu_seqlens are provided"
    if len(g.shape) == 3:
        # 3D 张量：标量门控模式
        return chunk_local_cumsum_scalar(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
            chunk_indices=chunk_indices,
        )
    elif len(g.shape) == 4:
        # 4D 张量：向量门控模式
        return chunk_local_cumsum_vector(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
            chunk_indices=chunk_indices,
        )
    else:
        raise ValueError(
            f"Unsupported input shape {g.shape}, "
            f"which should be (B, T, H, D) if `head_first=False` "
            f"or (B, H, T, D) otherwise"
        )
