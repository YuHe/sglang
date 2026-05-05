# Adapted from https://github.com/vllm-project/vllm/blob/0384aa7150c4c9778efca041ffd1beb3ad2bd694/vllm/model_executor/layers/fla/ops/kda.py
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# 本模块实现 KDA（Key-gated Delta Attention）的前向推理接口
# KDA 是 GDN 的向量化 gate 变体：每个 K 维度有独立的门控衰减（而非 GDN 的标量 gate）
# 提供两种计算路径：
#   - fused_recurrent_kda: 融合递归（短序列/decode 场景）
#   - chunk_kda: 分块 chunk 计算（长序列/prefill 场景）
# 核心算子：kda_gate_chunk_cumsum、chunk_kda_fwd_intra、chunk_gated_delta_rule_fwd_h

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from sglang.srt.layers.attention.fla.chunk_intra import chunk_kda_fwd_intra
from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
from sglang.srt.layers.attention.fla.fused_norm_gate import layer_norm_gated_fwd
from sglang.srt.layers.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_fwd_kernel,
)
from sglang.srt.layers.attention.fla.index import prepare_chunk_indices
from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd
from sglang.srt.layers.attention.fla.op import exp, log
from sglang.srt.layers.attention.fla.utils import check_shared_mem

# BS_LIST：根据 GPU 共享内存大小选择 chunk 块大小候选集
BS_LIST = [32, 64] if check_shared_mem() else [16, 32]


# 向上取整整数除法
def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


# 计算不小于 n 的最小 2 的幂次（用于 Triton 块大小对齐）
def next_power_of_2(n: int) -> int:
    """The next power of 2 (inclusive)"""
    if n < 1:
        return 1
    return 1 << (n - 1).bit_length()


# Python 封装：KDA 融合递归前向（reuse GDN kernel，开启 IS_KDA=True 路径）
def fused_recurrent_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    # ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    # BV 限制为 8，减少寄存器压力（decode 路径 T 短）
    BK, BV = next_power_of_2(K), min(next_power_of_2(V), 8)
    NK, NV = cdiv(K, BK), cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    # 分配输出；inplace_final_state=True 时与 initial_state 共享内存
    o = q.new_empty(NK, *v.shape)
    if inplace_final_state:
        final_state = initial_state
    else:
        final_state = q.new_empty(N, HV, V, K, dtype=initial_state.dtype)

    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = final_state.stride(0)

    # if ssm_state_indices is None:
    #     stride_indices_seq, stride_indices_tok = 1, 1
    # elif ssm_state_indices.ndim == 1:
    #     stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    # else:
    #     stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    # grid = (K分块数, V分块数, 序列数*HV)
    grid = (NK, NV, N * HV)
    # 复用 GDN fwd kernel，IS_KDA=True 激活向量 gate 路径
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        # ssm_state_indices=ssm_state_indices,
        # num_accepted_tokens=num_accepted_tokens,
        scale=scale,
        # N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        # stride_init_state_token=stride_init_state_token,
        # stride_final_state_token=stride_final_state_token,
        # stride_indices_seq=stride_indices_seq,
        # stride_indices_tok=stride_indices_tok,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=final_state is not None,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_VARLEN=cu_seqlens is not None,
        # INPLACE_FINAL_STATE=inplace_final_state,
        IS_KDA=True,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # 压缩 NK 维度并返回 (o, final_state)
    o = o.squeeze(0)
    return o, final_state


# 公开 API：KDA 融合递归注意力（decode/短序列场景）
def fused_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,  # KDA 默认对 q/k 做 L2 归一化
    cu_seqlens: torch.LongTensor | None = None,
    # ssm_state_indices: torch.LongTensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
            f"Please flatten variable-length inputs before processing."
        )
    # 默认缩放因子 1/sqrt(K)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    # 确保输入内存连续（Triton kernel 要求最后一维步长为 1）
    o, final_state = fused_recurrent_kda_fwd(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=scale,
        initial_state=initial_state,
        inplace_final_state=inplace_final_state,
        cu_seqlens=cu_seqlens,
        # ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    return o, final_state


# 带门控的 RMSNorm：调用 fused_norm_gate 的 Triton kernel 实现
# 支持 prenorm 模式（同时返回残差）和可选 residual 输入
def rms_norm_gated(
    x: torch.Tensor,
    g: torch.Tensor,        # 门控张量（silu/swish 激活后与归一化结果相乘）
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = "swish",
    residual: torch.Tensor | None = None,
    prenorm: bool = False,  # 是否同时返回归一化前的 residual
    residual_in_fp32: bool = False,
    eps: float = 1e-6,
):
    x_shape_og = x.shape
    # 展平为 2D [M, N] 后调用 Triton kernel（要求最后一维连续）
    x = x.contiguous().reshape(-1, x.shape[-1])
    g = g.contiguous().reshape(-1, g.shape[-1])
    if residual is not None:
        assert residual.shape == x_shape_og
        residual = residual.contiguous().reshape(-1, residual.shape[-1])
    # 确定 residual 精度（float32 或保持输入类型）
    residual_dtype = (
        residual.dtype
        if residual is not None
        else (torch.float if residual_in_fp32 else None)
    )
    # 调用 Triton 融合门控层归一化 kernel（is_rms_norm=True 跳过均值计算）
    y, _, _, residual_out = layer_norm_gated_fwd(
        x=x,
        g=g,
        weight=weight,
        bias=bias,
        activation=activation,
        eps=eps,
        residual=residual,
        residual_dtype=residual_dtype,
        is_rms_norm=True,
    )
    # 恢复原始形状后返回；prenorm=True 时额外返回 residual
    y = y.reshape(x_shape_og)
    return y if not prenorm else (y, residual_out.reshape(x_shape_og))


# autotune：对 BK 和 warp 数进行调优
@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BC", "IS_VARLEN"],
)
# Triton kernel：计算 chunk 内跨子块的 K*K^T（inter 子块部分）
# 同时计算 Akk（beta*k*k^T，用于 solve_tril）和 Aqk（scale*q*k^T，用于输出）
@triton.jit(do_not_specialize=["T"])
def chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter(
    q,           # query 张量指针 [B, T, H, K]
    k,           # key 张量指针 [B, T, H, K]
    g,           # KDA 向量 gate（累积和）指针 [B, T, H, K]
    beta,        # beta 缩放系数指针 [B, T, H]
    A,           # 输出：Akk 矩阵（beta*K*K^T）指针 [B, T, H, BT]
    Aqk,         # 输出：Aqk 矩阵（scale*Q*K^T）指针 [B, T, H, BT]
    scale,       # 注意力缩放因子
    cu_seqlens,  # 变长序列累积长度
    chunk_indices,  # chunk 索引映射
    T,           # 序列长度
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,   # chunk 大小
    BC: tl.constexpr,   # 子块大小（BT/NC）
    BK: tl.constexpr,   # K 分块大小（autotune）
    NC: tl.constexpr,   # 每个 chunk 内子块数（BT/BC）
    IS_VARLEN: tl.constexpr,
):
    # 三维 grid：(chunk数, NC*NC 子块对, batch*H)
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    # 解析子块对 (i_i, i_j)：i_i 为行子块，i_j 为列子块
    i_i, i_j = i_c // NC, i_c % NC
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # 超出序列长度或非下三角位置（i_i <= i_j）直接返回
    if i_t * BT + i_i * BC >= T:
        return
    if i_i <= i_j:
        return

    # 调整指针到当前序列起始位置
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    A += (bos * H + i_h) * BT
    Aqk += (bos * H + i_h) * BT

    # 加载行子块（i_i）的 beta 缩放系数
    p_b = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT + i_i * BC,), (BC,), (0,)
    )
    b_b = tl.load(p_b, boundary_check=(0,))

    # 分块累积 K*K^T（同时计算 Akk 和 Aqk，共享 K 分块加载）
    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q, (T, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k, (T, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0)
        )
        p_g = tl.make_block_ptr(
            g, (T, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0)
        )
        b_kt = tl.make_block_ptr(
            k, (K, T), (1, H * K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1)
        )
        p_gk = tl.make_block_ptr(
            g, (K, T), (1, H * K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1)
        )

        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        # 加载行子块起始位置的 gate 作为归一化基准（数值稳定性）
        # [BK,]
        b_gn = tl.load(g + (i_t * BT + i_i * BC) * H * K + o_k, mask=m_k, other=0)
        # [BC, BK] 行子块 k，按 gate 归一化：k * exp(g - gn)
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1)) * exp(b_g - b_gn[None, :])
        # [BK, BC] 列子块 k（转置），按 gate 归一化
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kt = tl.load(b_kt, boundary_check=(0, 1))
        # [BC, BC] 跨子块的 k*k^T（含 gate 衰减）
        b_ktg = b_kt * exp(b_gn[:, None] - b_gk)
        b_A += tl.dot(b_k, b_ktg)

        # 同时计算 Aqk = scale * q * k^T（共享 b_ktg）
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_qg = b_q * exp(b_g - b_gn[None, :]) * scale
        b_Aqk += tl.dot(b_qg, b_ktg)

    # 按 beta 缩放并存储 Akk 和 Aqk 到输出
    b_A *= b_b[:, None]

    p_A = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0)
    )
    tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))
    p_Aqk = tl.make_block_ptr(
        Aqk, (T, BT), (H * BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0)
    )
    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))


# autotune：对 warp 数进行调优
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
    key=["BK", "BT", "IS_VARLEN"],
)
# Triton kernel：计算 chunk 内对角子块的 K*K^T（intra 子块部分）
# 逐列迭代：每次处理子块内第 j 列（严格下三角），更新 A 和 Aqk
@triton.jit(do_not_specialize=["T"])
def chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra(
    q,           # query 张量指针 [B, T, H, K]
    k,           # key 张量指针 [B, T, H, K]
    g,           # KDA 向量 gate（累积和）指针 [B, T, H, K]
    beta,        # beta 缩放系数指针 [B, T, H]
    A,           # 输出：Akk 矩阵指针 [B, T, H, BT]
    Aqk,         # 输出：Aqk 矩阵指针 [B, T, H, BT]
    scale,       # 注意力缩放因子
    cu_seqlens,  # 变长序列累积长度
    chunk_indices,  # chunk 索引映射
    T,           # 序列长度
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,   # 子块大小
    BK: tl.constexpr,   # K 特征维度（固定为 next_power_of_2(K)，无 autotune）
    IS_VARLEN: tl.constexpr,
):
    # 三维 grid：(chunk数, 子块数(NC), batch*H)
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # 超出序列长度直接返回
    if i_t * BT + i_i * BC >= T:
        return

    # 构造行索引、列索引、行掩码和输出偏移
    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    # 行掩码：当前子块内哪些行在序列范围内
    m_A = (i_t * BT + i_i * BC + o_i) < T
    # 每行在 A/Aqk 张量中的起始偏移（列索引 = i_i * BC）
    o_A = (bos + i_t * BT + i_i * BC + o_i) * H * BT + i_h * BT + i_i * BC

    # 加载子块内的 q/k/g（[BC, BK] 格式）
    p_q = tl.make_block_ptr(
        q + (bos * H + i_h) * K,
        (T, K),
        (H * K, 1),
        (i_t * BT + i_i * BC, 0),
        (BC, BK),
        (1, 0),
    )
    p_k = tl.make_block_ptr(
        k + (bos * H + i_h) * K,
        (T, K),
        (H * K, 1),
        (i_t * BT + i_i * BC, 0),
        (BC, BK),
        (1, 0),
    )
    p_g = tl.make_block_ptr(
        g + (bos * H + i_h) * K,
        (T, K),
        (H * K, 1),
        (i_t * BT + i_i * BC, 0),
        (BC, BK),
        (1, 0),
    )
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))

    # 加载 beta 并缩放 k（[BC, BK]），用于 Akk 计算
    p_b = beta + (bos + i_t * BT + i_i * BC + o_i) * H + i_h
    b_k = b_k * tl.load(p_b, mask=m_A, other=0)[:, None]

    # 子块内逐列迭代（j < i）：严格下三角部分
    p_kt = k + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
    p_gk = g + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k

    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        # 加载第 j 列 k 和 gate（标量指针逐步推进）
        b_kt = tl.load(p_kt, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        # [BC, BK] gate 归一化：k_row * exp(g - gk)
        b_ktg = b_kt[None, :] * exp(b_g - b_gk[None, :])
        # 计算严格下三角 A[i, j]（i > j）
        b_A = tl.sum(b_k * b_ktg, 1)
        b_A = tl.where(o_i > j, b_A, 0.0)
        # 计算 Aqk[i, j]（i >= j，对角线也计算）
        b_Aqk = tl.sum(b_q * b_ktg, 1)
        b_Aqk = tl.where(o_i >= j, b_Aqk * scale, 0.0)
        # 写入 A 和 Aqk 的第 j 列
        tl.store(A + o_A + j, b_A, mask=m_A)
        tl.store(Aqk + o_A + j, b_Aqk, mask=m_A)
        # 推进列指针到下一行
        p_kt += H * K
        p_gk += H * K


# Python 封装：计算 KDA 的 beta*K*K^T 矩阵（同时计算 Aqk=scale*Q*K^T）
# 分两步：step1 处理跨子块（inter），step2 处理对角子块（intra）
def chunk_kda_scaled_dot_kkt_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Compute beta * K * K^T.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, H]`.
        gk (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H, K]` applied to the key tensor. Default: `None`.
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
    B, T, H, K = k.shape
    assert K <= 256
    BT = chunk_size
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    BC = min(16, BT)  # 子块大小（BC <= 16，控制 grid 粒度）
    NC = cdiv(BT, BC)  # 每个 chunk 内子块数
    BK = max(next_power_of_2(K), 16)  # K 维度对齐 16
    # 分配输出张量（A 和 Aqk 形状相同）
    A = torch.zeros(B, T, H, BT, device=k.device, dtype=output_dtype)
    Aqk = torch.zeros(B, T, H, BT, device=k.device, dtype=output_dtype)
    # step1：跨子块部分（NC*NC 子块对，仅处理 i_i > i_j 的下三角）
    grid = (NT, NC * NC, B * H)
    chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter[grid](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        A=A,
        Aqk=Aqk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        NC=NC,
        IS_VARLEN=cu_seqlens is not None,
    )

    # step2：对角子块部分（每个子块独立处理内部严格下三角）
    grid = (NT, NC, B * H)
    chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra[grid](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        A=A,
        Aqk=Aqk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        IS_VARLEN=cu_seqlens is not None,
    )
    return A, Aqk


# autotune：对 warp 数和 stages 进行调优
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
)
# Triton kernel：使用求解后的 A 矩阵重计算 w（写权重）和 u（更新值）
# w = A @ (beta * k * exp(gk))，u = A @ (beta * v)
# 可选同时计算并存储 qg = q * exp(gk)（STORE_QG）和 kg = k * exp(gn - gk)（STORE_KG）
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    q,           # query 张量指针（可选，STORE_QG 时使用）
    k,           # key 张量指针 [B, T, H, K]
    qg,          # qg 输出指针（q * exp(gk)，STORE_QG 时写入）
    kg,          # kg 输出指针（k * exp(gn - gk)，STORE_KG 时写入）
    v,           # value 张量指针 [B, T, H, V]
    beta,        # beta 缩放系数指针 [B, T, H]
    w,           # 写权重输出指针 [B, T, H, K]
    u,           # 更新值输出指针 [B, T, H, V]
    A,           # 求解后的 (I+A)^{-1} 矩阵 [B, T, H, BT]
    gk,          # KDA 向量 gate（累积和）指针 [B, T, H, K]
    cu_seqlens,  # 变长序列累积长度
    chunk_indices,  # chunk 索引映射
    T,           # 序列长度
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_QG: tl.constexpr,  # 是否写回 qg = q * exp(gk)
    STORE_KG: tl.constexpr,  # 是否写回 kg = k * exp(gn - gk)
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,  # 矩阵乘法精度（"ieee" 或 "tf32"）
):
    # 二维 grid：(chunk数, batch*H)
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    # 加载 chunk 内的 beta [BT]
    p_b = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    # 加载求解后的 A 矩阵 [BT, BT]
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    b_A = tl.load(p_A, boundary_check=(0, 1))

    # 计算 u = A @ (beta * v)（沿 V 维度分块）
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
        # b_vb = beta * v（广播 beta 到 V 维度）
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, input_precision=DOT_PRECISION)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    # 计算 w = A @ (beta * k * exp(gk))（沿 K 维度分块）
    for i_k in range(tl.cdiv(K, BK)):
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # b_kb = beta * k（广播 beta 到 K 维度）
        b_kb = b_k * b_b[:, None]

        p_gk = tl.make_block_ptr(
            gk + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        # b_kb *= exp(gk)：施加向量 gate 衰减
        b_kb *= exp(b_gk)
        if STORE_QG:
            # 计算 qg = q * exp(gk) 并存储（用于 chunk_gla_fwd_o_gk）
            p_q = tl.make_block_ptr(
                q + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            p_qg = tl.make_block_ptr(
                qg + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_qg = b_q * exp(b_gk)
            tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty), boundary_check=(0, 1))
        if STORE_KG:
            # 计算 kg = k * exp(gn - gk)（gn = chunk 末尾的 gate，用于跨 chunk 传播）
            last_idx = min(i_t * BT + BT, T) - 1

            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            # 加载 chunk 末尾的 gate 值作为基准（gn）
            b_gn = tl.load(
                gk + ((bos + last_idx) * H + i_h) * K + o_k, mask=m_k, other=0.0
            )
            b_kg = b_k * exp(b_gn - b_gk)

            p_kg = tl.make_block_ptr(
                kg + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))

        # 计算 w = A @ (beta * k * exp(gk)) 并存储
        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


# Python 封装：调用 recompute_w_u_fwd_kernel，返回 (w, u, None, kg)
def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,         # 求解后的 (I+A)^{-1} 矩阵
    q: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,  # KDA 向量 gate（None 时不计算 qg/kg）
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]
    # BK/BV 固定为 64（autotune 候选集内的常用值）
    BK = 64
    BV = 64

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # 分配输出张量
    w = torch.empty_like(k)
    u = torch.empty_like(v)
    # kg 仅在有向量 gate 时计算（用于跨 chunk 状态传播）
    kg = torch.empty_like(k) if gk is not None else None
    recompute_w_u_fwd_kernel[(NT, B * H)](
        q=q,
        k=k,
        qg=None,
        kg=kg,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        STORE_QG=False,
        STORE_KG=kg is not None,
        IS_VARLEN=cu_seqlens is not None,
        DOT_PRECISION="ieee",
    )
    # 返回 (w, u, None, kg)：第三项保留位（兼容 GDN 接口）
    return w, u, None, kg


# autotune：对 BK/BV 和 warp 数进行调优
@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BT", "IS_VARLEN"],
)
# Triton kernel：计算 chunk 内注意力输出（inter chunk 部分）
# o = qg @ h^T + Aqk @ v，其中 qg = q * exp(gk)，h 是从上个 chunk 传来的隐状态
@triton.jit(do_not_specialize=["T"])
def chunk_gla_fwd_kernel_o(
    q,           # query 张量指针（gate 归一化后）[B, T, H, K]
    v,           # value 张量指针 [B, T, H, V]
    g,           # KDA 向量 gate（累积和）指针 [B, T, H, K]
    h,           # 跨 chunk 隐状态指针 [NT, H, V, K]
    o,           # 输出注意力值指针 [B, T, H, V]
    A,           # Aqk 矩阵（intra attention score）[B, T, H, BT]
    cu_seqlens,  # 变长序列累积长度
    chunk_indices,  # chunk 索引映射
    scale,       # 注意力缩放因子
    T,           # 序列长度
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # 三维 grid：(V分块数, chunk数, batch*H)
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        # i_tg：全局 chunk 索引（用于索引 h 张量）
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # 下三角掩码（intra attention 只看当前位置之前的 token）
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    # inter chunk 部分：o += qg @ h^T（跨 chunk 隐状态贡献）
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_g = tl.make_block_ptr(
            g + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_h = tl.make_block_ptr(
            h + (i_tg * H + i_h) * V * K,
            (V, K),
            (K, 1),
            (i_v * BV, i_k * BK),
            (BV, BK),
            (1, 0),
        )

        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, BK]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        # [BT, BK] qg = q * exp(g)（应用向量 gate 衰减）
        b_qg = (b_q * exp(b_g)).to(b_q.dtype)
        # [BK, BV] 跨 chunk 隐状态
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # works but dkw, owing to divine benevolence
        # [BT, BV] o += qg @ h^T
        if i_k >= 0:
            b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
    # intra chunk 部分：o += Aqk @ v（chunk 内 attention score 加权求和）
    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_o = tl.make_block_ptr(
        o + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    # [BT, BT] Aqk（已含 scale），掩码到下三角
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where(m_s, b_A, 0.0).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


# Python 封装：计算 KDA chunk 输出（inter + intra 部分求和）
def chunk_gla_fwd_o_gk(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,          # KDA 向量 gate（累积和）
    A: torch.Tensor,          # Aqk 矩阵（intra attention score）
    h: torch.Tensor,          # 跨 chunk 隐状态 [NT, H, V, K]
    o: torch.Tensor,          # 输出张量（inplace 写入）
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # 动态 grid：V 分块数随 BV autotune 变化
    def grid(meta):
        return (cdiv(V, meta["BV"]), NT, B * H)

    chunk_gla_fwd_kernel_o[grid](
        q=q,
        v=v,
        g=g,
        h=h,
        o=o,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        IS_VARLEN=cu_seqlens is not None,
    )
    return o


# Triton JIT 函数：数值稳定的 softplus(x) = log(1 + exp(x))
# x > 20 时直接返回 x，避免 exp 溢出
@triton.jit
def softplus_fwd(x):
    """Standard softplus: log(1 + exp(x)), with linear approx for large x."""
    return tl.where(x < 20.0, log(1.0 + exp(x)), x)


# 启发式规则：自动检测是否有 dt_bias、scale、变长序列和 lower_bound
@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["dt_bias"] is not None,
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)
# autotune：对 BS（S 维度分块大小）和 warp 数进行调优
@triton.autotune(
    configs=[
        triton.Config({"BS": BS}, num_warps=num_warps)
        for BS in BS_LIST
        for num_warps in [2, 4, 8]
    ],
    key=["H", "S", "BT", "IS_VARLEN"],
)
# Triton kernel：KDA gate 激活 + chunk 内累积和（融合两步为一步）
# 输出：g = cumsum(-exp(A_log) * softplus(s + dt_bias)) 或 cumsum(lower_bound * sigmoid(exp(A_log) * s))
@triton.jit(do_not_specialize=["T"])
def kda_gate_chunk_cumsum_vector_kernel(
    s,           # 原始 gate 输入指针 [B, T, H, S]
    A_log,       # 对数衰减参数指针 [H]
    dt_bias,     # dt 偏置指针（可选）[H, S]
    o,           # 输出累积 gate 指针 [B, T, H, S]
    scale,       # 可选缩放因子
    cu_seqlens,  # 变长序列累积长度
    chunk_indices,  # chunk 索引映射
    lower_bound, # safe gate 模式的下界（USE_LOWER_BOUND 时使用）
    T,           # 序列长度
    H: tl.constexpr,
    S: tl.constexpr,  # gate 特征维度（等于 K）
    BT: tl.constexpr, # chunk 大小
    BS: tl.constexpr, # S 分块大小（autotune）
    HAS_BIAS: tl.constexpr,        # 是否有 dt_bias
    HAS_SCALE: tl.constexpr,       # 是否有 scale
    IS_VARLEN: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr, # 是否使用 safe gate（lower_bound 模式）
):
    # 三维 grid：(S分块数, chunk数, batch*H)
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # 加载原始 gate [BT, BS]
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
    # [BT, BS]
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)

    if HAS_BIAS:
        # 加载 dt_bias 并广播到 [BT, BS]
        p_b = tl.make_block_ptr(
            dt_bias + i_h * S,
            (S,),
            (1,),
            (i_s * BS,),
            (BS,),
            (0,),
        )
        b_bias = tl.load(p_b, boundary_check=(0,)).to(tl.float32)
        b_s = b_s + b_bias[None, :]

    b_A = tl.load(A_log + i_h).to(tl.float32)
    if not USE_LOWER_BOUND:
        # 标准 gate：g = -exp(A_log) * softplus(s + bias)
        b_gate = -exp(b_A) * softplus_fwd(b_s)
    else:
        # Safe gate：g = lower_bound * sigmoid(exp(A_log) * s)（数值更稳定）
        b_gate = lower_bound * tl.sigmoid(exp(b_A) * b_s)

    # 沿时间轴（axis=0）计算 chunk 内累积和
    b_o = tl.cumsum(b_gate, axis=0)

    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def kda_gate_chunk_cumsum(
    g: torch.Tensor,
    A_log: torch.Tensor,
    chunk_size: int,
    scale: float = None,
    dt_bias: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    output_dtype: Optional[torch.dtype] = torch.float,
    chunk_indices: Optional[torch.LongTensor] = None,
    lower_bound: Optional[float] = None,
) -> torch.Tensor:
    """
    Fused KDA gate activation + chunk-local cumulative sum.

    Combines two memory-bound kernels into one:
      1. Gate activation: g = -exp(A_log) * softplus(raw_g + dt_bias)
      2. Chunk-local cumsum along the time axis

    Args:
        g: Raw gate tensor of shape [B, T, H, K] (before activation).
        A_log: Per-head log-scale parameter, [H] elements (any shape, numel=H).
        chunk_size: Chunk size for cumsum (must be power of 2).
        scale: Optional scale factor applied to output.
        dt_bias: Optional per-head bias, flat [H*K] elements.
        cu_seqlens: Cumulative sequence lengths for variable-length input.
        output_dtype: Output dtype (default float32).
        chunk_indices: Pre-computed chunk indices for varlen mode.
        lower_bound: If set, use safe gate: lower_bound * sigmoid(exp(A_log) * g).

    Returns:
        Cumulative-summed gated tensor of shape [B, T, H, K].
    """
    if cu_seqlens is not None:
        assert (
            g.shape[0] == 1
        ), "Only batch size 1 is supported when cu_seqlens are provided"
    assert len(g.shape) == 4
    B, T, H, S = g.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    # chunk_size 必须是 2 的幂次（tl.cumsum 要求）
    assert chunk_size == 2 ** (
        chunk_size.bit_length() - 1
    ), "chunk_size must be a power of 2"

    # 分配输出张量（与输入同形状）
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)

    # 动态 grid：S 分块数随 BS autotune 变化
    def grid(meta):
        return (cdiv(meta["S"], meta["BS"]), NT, B * H)

    kda_gate_chunk_cumsum_vector_kernel[grid](
        s=g_org,
        A_log=A_log,
        dt_bias=dt_bias,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        lower_bound=lower_bound,
        T=T,
        H=H,
        S=S,
        BT=BT,
    )
    return g


# Python 封装：KDA chunk 前向计算（prefill/长序列场景）
# 流程：gate_cumsum → chunk_kda_fwd_intra → chunk_gated_delta_rule_fwd_h → chunk_gla_fwd_o_gk
def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,          # 原始 gate（A_log 不为 None 时由 kernel 激活；否则已激活）
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,  # 每条序列的 SSM 状态索引
    cu_seqlens: Optional[torch.LongTensor] = None,
    A_log: Optional[torch.Tensor] = None,  # 对数衰减参数（非 None 时触发融合 gate 激活）
    dt_bias: Optional[torch.Tensor] = None,
    lower_bound: Optional[float] = None,   # safe gate 下界
):
    chunk_size = 64
    # 预计算 chunk 索引（变长序列），避免各子 kernel 重复计算
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, chunk_size)
        if cu_seqlens is not None
        else None
    )

    if A_log is not None:
        # A_log 不为 None：融合 gate 激活 + chunk 内累积和（一次 kernel 调用）
        # g 是原始 gate；A_log/dt_bias 驱动激活函数
        g = kda_gate_chunk_cumsum(
            g,
            A_log=A_log,
            chunk_size=chunk_size,
            dt_bias=dt_bias,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            lower_bound=lower_bound,
        )
    else:
        # g 已由调用方激活，只需做 chunk 内累积和
        g = chunk_local_cumsum(
            g,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    # 步骤2：融合 scaled_dot_kkt + solve_tril + recompute_w_u（intra chunk 计算）
    # Fused: scaled_dot_kkt + solve_tril + recompute_w_u
    w, u, _, kg, Aqk, _ = chunk_kda_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )

    # 步骤3：跨 chunk 隐状态传播（chunk_gated_delta_rule_fwd_h）
    h, v_new = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        gk=g,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    # 步骤4：计算最终输出（intra + inter chunk 注意力求和）
    del w, u, kg
    o = chunk_gla_fwd_o_gk(
        q=q,
        v=v_new,
        g=g,
        A=Aqk,
        h=h,
        o=v,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )
    del Aqk, v_new, h
    return o


# 公开 API：KDA chunk 注意力（prefill/长序列场景）
def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    initial_state_indices: torch.Tensor = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    lower_bound: Optional[float] = None,
    **kwargs,
):
    # 默认缩放因子 1/sqrt(K)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    # 可选：在进入 chunk 计算前先做 L2 归一化（数值稳定性）
    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q.contiguous())
        k = l2norm_fwd(k.contiguous())

    o = chunk_kda_fwd(
        q=q,
        k=k,
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=scale,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
    )
    return o
