# Adapted from: https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/mamba/ops/ssd_combined.py

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_combined.py

# Mamba2 SSD (State Space Duality) 组合前向计算：协调 5 个子 kernel 完成 chunked SSM 扫描

# ruff: noqa: E501

import torch
import triton
from einops import rearrange
from packaging import version

# 导入 5 个子算子（分别负责 SSD 算法的各个计算阶段）
from .ssd_bmm import _bmm_chunk_fwd                          # chunk 内批量矩阵乘法
from .ssd_chunk_scan import _chunk_scan_fwd                  # chunk 内因果扫描（含 D、z 跳连）
from .ssd_chunk_state import _chunk_cumsum_fwd, _chunk_state_fwd, chunk_state_varlen
                                                             # dA cumsum、chunk 状态聚合、varlen 状态
from .ssd_state_passing import _state_passing_fwd            # 跨 chunk 状态传递（递推）

# 检查 Triton 版本 >= 2.2，某些 kernel 特性依赖该版本
TRITON_22 = version.parse(triton.__version__) >= version.parse("2.2.0")


# 工具函数：判断整数 n 是否为 2 的幂（chunk_size 必须满足此约束）
def is_int_pow_2(n):
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0


# Mamba2 chunked SSM 前向计算核心函数：执行 5 步子计算，返回输出和中间状态
def _mamba_chunk_scan_combined_fwd(
    x,               # 输入序列，(batch, seqlen, nheads, headdim)
    dt,              # 时间步参数 ∆，(batch, seqlen, nheads)
    A,               # SSM 状态转移矩阵（对角线），(nheads,)，负实数
    B,               # SSM 输入矩阵，(batch, seqlen, ngroups, dstate)
    C,               # SSM 输出矩阵，(batch, seqlen, ngroups, dstate)
    chunk_size,      # 分块大小（必须是 2 的幂）
    D=None,          # 跳跃连接权重，(nheads, headdim) 或 (nheads,)
    z=None,          # 门控分支，(batch, seqlen, nheads, headdim)
    dt_bias=None,    # ∆ 偏置，(nheads,)
    initial_states=None,  # 初始 SSM 状态（radix cache 命中时），(batch, nheads, headdim, dstate)
    seq_idx=None,    # 序列 ID，(batch, seqlen)，连续批处理时使用
    chunk_indices=None,   # 逻辑 chunk 索引数组（连续批处理时辅助定位）
    chunk_offsets=None,   # 逻辑 chunk 偏移（连续批处理时辅助 dA_cumsum 修正）
    cu_seqlens=None,      # 序列累计长度（varlen 模式，用于计算 varlen_states）
    dt_softplus=False,    # 是否对 ∆ 应用 softplus 激活
    dt_limit=(0.0, float("inf")),  # ∆ 的有效范围限制
    state_dtype=None,     # SSM 状态张量的 dtype（默认使用 C.dtype）
    out=None,             # 预分配的输出张量（可选）
):
    assert is_int_pow_2(chunk_size), "chunk_size must be integer power of 2"
    # 解析输入张量的形状参数
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0   # nheads 必须是 ngroups 的倍数
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, seqlen, nheads)
    assert A.shape == (nheads,)
    assert C.shape == B.shape
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        # D 可以是每个 head 独立的向量（headdim），或每个 head 一个标量
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    # 确保关键张量内存连续（Triton kernel 要求 stride(-1)==1）
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if (
        x.stride(-1) != 1 and x.stride(1) != 1
    ):  # Either M or K dimension should be contiguous
        x = x.contiguous()
    if (
        z is not None and z.stride(-1) != 1 and z.stride(1) != 1
    ):  # Either M or K dimension should be contiguous
        z = z.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()
    if initial_states is not None:
        if cu_seqlens is None:
            # 常规批处理：每个 batch 有自己的初始状态
            assert initial_states.shape == (batch, nheads, headdim, dstate)
        else:
            # varlen 模式：每个序列有自己的初始状态（序列数 = len(cu_seqlens)-1）
            assert initial_states.shape == (
                len(cu_seqlens) - 1,
                nheads,
                headdim,
                dstate,
            )

    # This function executes 5 sub-functions for computing mamba
    # - a good resource is the blog https://goombalab.github.io/blog/2024/mamba2-part3-algorithm/
    #   which has a minimal implementation to understand the below operations
    # - as explained by the blog, mamba is a special case of causal attention
    # - the idea is to chunk the attention matrix and compute each
    #   submatrix separately using different optimizations.
    # - see the blog and paper for a visualization of the submatrices
    #   which we refer to in the comments below

    # 1. Compute chunked cumsum of A * dt
    # - here dt may go through a softplus activation
    # 第 1 步：计算 dA 累积和（∆ × A 的 cumsum），同时对 ∆ 施加 softplus 和 dt_bias
    dA_cumsum, dt = _chunk_cumsum_fwd(
        dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit
    )

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    # 第 2 步：计算每个 chunk 内的局部 SSM 状态增量（对应 SSD 分解中的 B 项贡献）
    states = _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    # - for handling chunked prefill, this requires i) initial_states
    #   ii) seq_idx iii) is_cont_batched and (iv) chunk_offsets to be all specified.
    # - When a new seq_idx is detected, we will stop passing the prev_state
    #   and switch accordingly to the init_state corresponding to the new seq_idx.
    # - We will also make sure that the dA_cumsum is taken only from the start of the
    #   sequence (hence we need the full dA_cumsum tensor and not just the values at chunk boundaries)
    # - this will ensure that states will be updated with the rightmost flushed seq_idx
    #   of the previous chunk. This implies that the first chunk of states is either 0
    #   or equal to init_states of the first example.
    # 第 3 步：跨 chunk 传递 SSM 状态（对应 SSD 分解中的 A 项递推），生成每个 chunk 的初始全局状态
    states, final_states = _state_passing_fwd(
        rearrange(states, "... p n -> ... (p n)"),  # 将 (headdim, dstate) 展平为 (headdim*dstate)
        dA_cumsum,
        initial_states=(
            rearrange(initial_states, "... p n -> ... (p n)")
            if initial_states is not None
            else None
        ),
        seq_idx=seq_idx,
        chunk_size=chunk_size,
        out_dtype=state_dtype if state_dtype is not None else C.dtype,
        is_cont_batched=cu_seqlens is not None,  # varlen 模式视为连续批处理
        chunk_offsets=chunk_offsets,
    )
    # 将展平的状态还原为 (batch, nchunks, nheads, headdim, dstate) 形状
    states, final_states = (
        rearrange(t, "... (p n) -> ... p n", n=dstate) for t in [states, final_states]
    )

    # 4. Compute batched matrix multiply for C_j^T B_i terms
    # 第 4 步：计算 chunk 内 C^T B 矩阵乘法（对应 SSD 中对角块的注意力分数矩阵）
    CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)

    # 5. Scan and compute the diagonal blocks, taking into
    #    account past causal states.
    # - if initial states are provided, then states information will be
    #   augmented with initial_states.
    # - to do this properly, we need to account for example changes in
    #   the continuous batch, therefore we introduce pseudo chunks, which is
    #   a chunk that is split up each time an example changes.
    # - in each (pseudo) chunk, we detect if the previous (pseudo) chunk had
    #   a seq_idx change, in which case we take states information from
    #   init_states.
    # 第 5 步：chunk 内因果扫描，整合 off-diagonal 项（跨 chunk 状态）和 diagonal 项（chunk 内状态）
    out_x = _chunk_scan_fwd(
        CB,
        x,
        dt,
        dA_cumsum,
        C,
        states,
        D=D,
        z=z,
        seq_idx=seq_idx,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        initial_states=initial_states,
        out=out,
    )
    if cu_seqlens is None:
        # 普通模式：返回 (输出, dt, dA_cumsum, 中间状态, 最终状态)
        return out_x, dt, dA_cumsum, states, final_states
    else:
        # varlen 模式：还需要计算各序列的独立最终状态 varlen_states
        assert (
            batch == 1
        ), "passing cu_seqlens to get the varlen states is only supported if batch dimension is 1"
        # 计算每个序列的 varlen 状态（用于推理时缓存各序列的 SSM 状态）
        varlen_states = chunk_state_varlen(
            B.squeeze(0),
            x.squeeze(0),
            dt.squeeze(0),
            dA_cumsum.squeeze(0),
            cu_seqlens,
            states.squeeze(0),
            initial_states=initial_states,
        )
        return out_x, dt, dA_cumsum, states, final_states, varlen_states


# 公开 API：Mamba2 chunked SSM 扫描，支持多种返回模式（中间状态/最终状态/varlen 状态）
def mamba_chunk_scan_combined(
    x,               # 输入序列，(batch, seqlen, nheads, headdim)
    dt,              # 时间步 ∆，(batch, seqlen, nheads)
    A,               # 状态转移矩阵对角线，(nheads,)
    B,               # SSM 输入矩阵，(batch, seqlen, ngroups, dstate)
    C,               # SSM 输出矩阵，(batch, seqlen, ngroups, dstate)
    chunk_size,      # chunk 大小（2 的幂）
    D=None,          # 跳跃连接 D，(nheads, headdim) 或 (nheads,)
    z=None,          # 门控分支 z，(batch, seqlen, nheads, headdim)
    dt_bias=None,    # ∆ 偏置，(nheads,)
    initial_states=None,  # 初始 SSM 状态
    seq_idx=None,         # 序列 ID
    chunk_indices=None,   # 逻辑 chunk 索引
    chunk_offsets=None,   # 逻辑 chunk 偏移
    cu_seqlens=None,      # 序列累计长度（varlen 模式）
    dt_softplus=False,    # 是否对 ∆ 应用 softplus
    dt_limit=(0.0, float("inf")),  # ∆ 的值域限制
    out=None,             # 预分配输出张量
    return_final_states=False,        # 是否返回最终 SSM 状态
    return_varlen_states=False,       # 是否返回 varlen 模式的各序列状态
    return_intermediate_states=False, # 是否返回中间 chunk 状态序列
    state_dtype=None,     # SSM 状态 dtype
):
    """
    Argument:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads)
        A: (nheads)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        chunk_size: int
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
        dt_bias: (nheads,)
        initial_states: (batch, nheads, headdim, dstate)
        seq_idx: (batch, seqlen)
        cu_seqlens: (num_sequences + 1) or None, only used if return_varlen_states is True
        dt_softplus: Whether to apply softplus to dt
        out: Preallocated output tensor
        state_dtype: The data type of the ssm state
    """

    # 若不需要 varlen_states，忽略 cu_seqlens（避免不必要的 varlen 计算）
    if not return_varlen_states:
        cu_seqlens = None
    else:
        assert (
            cu_seqlens is not None
        ), "cu_seqlens must be provided if return_varlen_states is True"
    # 调用内部函数，解包 5 个（或 6 个）返回值
    out_x, dt_out, dA_cumsum, states, final_states, *rest = (
        _mamba_chunk_scan_combined_fwd(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            D=D,
            z=z,
            dt_bias=dt_bias,
            initial_states=initial_states,
            seq_idx=seq_idx,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            cu_seqlens=cu_seqlens,
            dt_softplus=dt_softplus,
            dt_limit=dt_limit,
            out=out,
            state_dtype=state_dtype,
        )
    )
    # 根据调用方的需求，选择返回内容
    if return_intermediate_states:
        # 返回中间状态序列（每个 chunk 的初始状态）
        if return_varlen_states:
            varlen_states = rest[0]
            if return_final_states:
                return states, final_states, varlen_states
            else:
                return states, varlen_states
        else:
            if return_final_states:
                return states, final_states
            else:
                return states

    if not return_varlen_states:
        if not return_final_states:
            return    # 仅计算输出（不返回任何状态）
        else:
            return final_states  # 仅返回最终状态（prefill 后的 decode cache 初始化）
    else:
        varlen_states = rest[0]
        return (
            (varlen_states)
            if not return_final_states
            else (final_states, varlen_states)
        )
