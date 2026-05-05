# Adapted from: https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/mamba/ops/ssd_chunk_state.py

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_chunk_state.py

# SSD chunk 状态计算 Triton kernel 集合：
# 1. _chunk_cumsum_fwd_kernel：计算 dt×A 的 chunk 内累积和（dA_cumsum）
# 2. _chunk_state_fwd_kernel：聚合 chunk 内 SSM 局部状态（∑ exp(dA) * dt * B * x）
# 3. _chunk_state_varlen_kernel：varlen 模式的序列末尾状态计算（cu_seqlens 支持）

# ruff: noqa: E501

import math

import torch
import triton
import triton.language as tl

from .mamba_ssm import softplus  # 导入 softplus 激活函数（Triton JIT 实现）


# Triton JIT kernel：计算每个 chunk 的 dA 累积和，同时处理 dt 的 softplus 和 clamp
# grid = (batch, nchunks, nheads_blocks)
@triton.jit
def _chunk_cumsum_fwd_kernel(
    # Pointers to matrices
    dt_ptr,           # 输入时间步 ∆，(batch, seqlen, nheads)
    A_ptr,            # 状态转移矩阵对角线 A，(nheads,)，负实数
    dt_bias_ptr,      # ∆ 偏置，(nheads,)，可为 None
    dt_out_ptr,       # 输出处理后的 ∆，(batch, nheads, nchunks, chunk_size)
    dA_cumsum_ptr,    # 输出 dA 累积和，(batch, nheads, nchunks, chunk_size)
    # Matrix dimension
    batch,       # batch 大小
    seqlen,      # 序列长度
    nheads,      # head 数
    chunk_size,  # chunk 大小
    dt_min,      # ∆ 下界（dt_limit 的第一个元素）
    dt_max,      # ∆ 上界（dt_limit 的第二个元素）
    # Strides
    stride_dt_batch,      # dt 的 batch 步长
    stride_dt_seqlen,     # dt 的 seqlen 步长
    stride_dt_head,       # dt 的 head 步长
    stride_A_head,        # A 的 head 步长
    stride_dt_bias_head,  # dt_bias 的 head 步长
    stride_dt_out_batch,  # dt_out 的 batch 步长
    stride_dt_out_chunk,  # dt_out 的 chunk 步长
    stride_dt_out_head,   # dt_out 的 head 步长
    stride_dt_out_csize,  # dt_out 的 chunk_size 内步长
    stride_dA_cs_batch,   # dA_cumsum 的 batch 步长
    stride_dA_cs_chunk,   # dA_cumsum 的 chunk 步长
    stride_dA_cs_head,    # dA_cumsum 的 head 步长
    stride_dA_cs_csize,   # dA_cumsum 的 chunk_size 内步长
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,         # 是否对 ∆ 应用 softplus
    HAS_DT_BIAS: tl.constexpr,         # 是否有 ∆ 偏置
    BLOCK_SIZE_CHUNK: tl.constexpr,    # chunk_size 的下一个 2 的幂（用于向量化）
    BLOCK_SIZE_H: tl.constexpr = 16,   # head 维度的分块大小
):
    pid_b = tl.program_id(axis=0)

    # if dt is long, may cause problems, so use 64 bit
    # https://github.com/triton-lang/triton/issues/1058
    # 使用 int64 避免大序列时 chunk 索引溢出
    pid_c = tl.program_id(axis=1).to(tl.int64)
    pid_h = tl.program_id(axis=2)
    # 定位到当前 (batch, chunk) 的 dt 起始位置
    dt_ptr += pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen
    dt_out_ptr += pid_b * stride_dt_out_batch + pid_c * stride_dt_out_chunk
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk

    # head 分块偏移和 chunk 内位置偏移
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    # dt 指针矩阵：(BLOCK_SIZE_H, BLOCK_SIZE_CHUNK) 块
    dt_ptrs = dt_ptr + (
        offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_seqlen
    )
    A_ptrs = A_ptr + offs_h * stride_A_head
    dt_out_ptrs = dt_out_ptr + (
        offs_h[:, None] * stride_dt_out_head + offs_c[None, :] * stride_dt_out_csize
    )
    dA_cs_ptrs = dA_cumsum_ptr + (
        offs_h[:, None] * stride_dA_cs_head + offs_c[None, :] * stride_dA_cs_csize
    )
    # 当前 chunk 实际有效长度（最后一个 chunk 可能不足 chunk_size）
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    # 加载 ∆（仅加载有效范围内的值，越界置 0）
    dt = tl.load(
        dt_ptrs,
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit),
        other=0.0,
    ).to(tl.float32)
    if HAS_DT_BIAS:
        # 加载 ∆ 偏置并累加
        dt_bias = tl.load(
            dt_bias_ptr + offs_h * stride_dt_bias_head, mask=offs_h < nheads, other=0.0
        ).to(tl.float32)
        dt += dt_bias[:, None]
    if DT_SOFTPLUS:
        # 对 ∆ 应用 softplus（仅当 dt <= 20 时，避免数值溢出）
        dt = tl.where(dt <= 20.0, softplus(dt), dt)
    # As of Triton 2.2.0, tl.clamp is not available yet
    # dt = tl.clamp(dt, dt_min, dt_max)
    # 手动实现 clamp：将 ∆ 限制在 [dt_min, dt_max] 范围内
    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
    # 越界位置置 0（确保 padding 不影响累积和）
    dt = tl.where(
        (offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), dt, 0.0
    )
    # 写出处理后的 ∆（供后续 SSM 状态计算使用）
    tl.store(
        dt_out_ptrs,
        dt,
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size),
    )
    # 加载 A 矩阵对角线（每个 head 一个值）
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
    # dA = dt * A：离散化增量（每个时间步的状态衰减量）
    dA = dt * A[:, None]
    # tl.cumsum：沿 chunk 方向计算累积和（前缀和），得到 dA_cumsum
    dA_cs = tl.cumsum(dA, axis=1)
    # 写出 dA 累积和
    tl.store(
        dA_cs_ptrs,
        dA_cs,
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size),
    )


# Triton JIT kernel：聚合 chunk 内的局部 SSM 状态（S_c = ∑_t exp(dA_cs[-1] - dA_cs[t]) * dt[t] * B[t] * x[t]^T）
# grid = (hdim块*dstate块, batch*nchunks, nheads)
@triton.jit
def _chunk_state_fwd_kernel(
    # Pointers to matrices
    x_ptr,         # 输入 x，(batch, seqlen, nheads, hdim)
    b_ptr,         # SSM 输入矩阵 B，(batch, seqlen, ngroups, dstate)
    states_ptr,    # 输出局部 SSM 状态，(batch, nchunks, nheads, hdim, dstate)
    dt_ptr,        # 时间步 ∆（处理后），(batch, nheads, nchunks, chunk_size)
    dA_cumsum_ptr, # dA 累积和，(batch, nheads, nchunks, chunk_size)
    seq_idx_ptr,   # 序列 ID，(batch, seqlen)，连续批处理时使用
    # Matrix dimensions
    hdim,       # head 维度（headdim）
    dstate,     # SSM 状态维度
    chunk_size, # chunk 大小
    batch,      # batch 大小
    seqlen,     # 序列长度
    nheads_ngroups_ratio,  # nheads // ngroups
    # Strides
    stride_x_batch,        # x 的 batch 步长
    stride_x_seqlen,       # x 的 seqlen 步长
    stride_x_head,         # x 的 head 步长
    stride_x_hdim,         # x 的 hdim 步长
    stride_b_batch,        # B 的 batch 步长
    stride_b_seqlen,       # B 的 seqlen 步长
    stride_b_head,         # B 的 group/head 步长
    stride_b_dstate,       # B 的 dstate 步长
    stride_states_batch,   # states 的 batch 步长
    stride_states_chunk,   # states 的 chunk 步长
    stride_states_head,    # states 的 head 步长
    stride_states_hdim,    # states 的 hdim 步长
    stride_states_dstate,  # states 的 dstate 步长
    stride_dt_batch,       # dt 的 batch 步长
    stride_dt_chunk,       # dt 的 chunk 步长
    stride_dt_head,        # dt 的 head 步长
    stride_dt_csize,       # dt 的 chunk_size 内步长
    stride_dA_cs_batch,    # dA_cumsum 的 batch 步长
    stride_dA_cs_chunk,    # dA_cumsum 的 chunk 步长
    stride_dA_cs_head,     # dA_cumsum 的 head 步长
    stride_dA_cs_csize,    # dA_cumsum 的 chunk_size 内步长
    stride_seq_idx_batch,  # seq_idx 的 batch 步长
    stride_seq_idx_seqlen, # seq_idx 的 seqlen 步长
    # Meta-parameters
    HAS_SEQ_IDX: tl.constexpr,       # 是否启用序列 ID（连续批处理）
    BLOCK_SIZE_M: tl.constexpr = 16,  # hdim 维度分块大小
    BLOCK_SIZE_N: tl.constexpr = 16,  # dstate 维度分块大小
    BLOCK_SIZE_K: tl.constexpr = 16,  # chunk_size 维度分块大小
):
    # axis=1: batch*chunk 组合索引；axis=2: head；axis=0: hdim块*dstate块
    pid_bc = tl.program_id(axis=1).to(tl.int64)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n  # hdim 方向块索引
    pid_n = tl.program_id(axis=0) % num_pid_n   # dstate 方向块索引
    # 定位各指针到当前 (batch, chunk, head) 的起始位置
    b_ptr += (
        pid_b * stride_b_batch
        + pid_c * chunk_size * stride_b_seqlen
        + (pid_h // nheads_ngroups_ratio) * stride_b_head  # B 按 group 共享
    )
    x_ptr += (
        pid_b * stride_x_batch
        + pid_c * chunk_size * stride_x_seqlen
        + pid_h * stride_x_head
    )
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += (
        pid_b * stride_dA_cs_batch
        + pid_c * stride_dA_cs_chunk
        + pid_h * stride_dA_cs_head
    )
    if HAS_SEQ_IDX:
        seq_idx_ptr += (
            pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen
        )

    # hdim、dstate、chunk_size 方向的分块偏移
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # x_ptrs：(hdim_M, chunk_K) 的转置访问（x 按 hdim 行、seqlen 列）
    x_ptrs = x_ptr + (
        offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen
    )
    # b_ptrs：(chunk_K, dstate_N) 访问
    b_ptrs = b_ptr + (
        offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen
    )
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    # 加载 chunk 末尾的 dA_cumsum（作为归一化基准值）
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(
        tl.float32
    )
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    if HAS_SEQ_IDX:
        seq_idx_ptrs = seq_idx_ptr + offs_k * stride_seq_idx_seqlen

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    if HAS_SEQ_IDX:
        # 加载 chunk 末尾的序列 ID（用于跨序列边界清零）
        seq_idx_last = tl.load(
            seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen
        )

    # 累加器：局部 SSM 状态 S_c = ∑_t scale_t * B_t^T * x_t
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # 沿 chunk_size 方向分块迭代，聚合 chunk 内状态
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        # 加载 x 块（hdim × K 块）
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k),
            other=0.0,
        )
        # 加载 B 块（K × dstate 块），转为 float32 计算
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate),
            other=0.0,
        ).to(tl.float32)
        # 加载当前 k 步的 dA_cumsum
        dA_cs_k = tl.load(
            dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0
        ).to(tl.float32)
        if HAS_SEQ_IDX:
            # 加载序列 ID（用于跨序列边界清零）
            seq_idx_k = tl.load(
                seq_idx_ptrs, mask=offs_k < chunk_size_limit - k, other=-1
            )
        # 加载 ∆（时间步大小）
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(
            tl.float32
        )
        if not HAS_SEQ_IDX:
            # scale = exp(dA_cs_last - dA_cs_k) * dt_k（从时间步 k 到 chunk 末尾的传播衰减）
            scale = tl.exp(dA_cs_last - dA_cs_k) * dt_k
        else:
            # 跨序列边界置 0：只累积同一序列内的贡献
            scale = tl.where(
                seq_idx_k == seq_idx_last, tl.exp(dA_cs_last - dA_cs_k) * dt_k, 0.0
            )
        b *= scale[:, None]  # B_t 乘以时间衰减 scale（dB = B * dt * exp(...)）
        b = b.to(x_ptr.dtype.element_ty)
        # tl.dot：x @ B（hdim × dstate），累加到局部状态
        acc += tl.dot(x, b)
        # 推进各指针到下一个 K 块
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize
        if HAS_SEQ_IDX:
            seq_idx_ptrs += BLOCK_SIZE_K * stride_seq_idx_seqlen
    # 将累加结果转换为输出 dtype
    states = acc.to(states_ptr.dtype.element_ty)

    # 定位输出指针并写出局部 SSM 状态
    states_ptr += (
        pid_b * stride_states_batch
        + pid_c * stride_states_chunk
        + pid_h * stride_states_head
    )
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (
        offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate
    )
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    # tl.store：写出当前 chunk 的局部 SSM 状态
    tl.store(states_ptrs, states, mask=c_mask)


# Triton JIT kernel：varlen 模式的序列末尾 SSM 状态计算
# 每个序列独立计算其最后一个 chunk 内的状态贡献，并与跨 chunk 传递的状态合并
# grid = (hdim块*dstate块, batch, nheads)
@triton.jit
def _chunk_state_varlen_kernel(
    # Pointers to matrices
    x_ptr,              # 输入 x（去掉 batch 维），(total_seqlen, nheads, hdim)
    b_ptr,              # SSM 输入矩阵 B，(total_seqlen, ngroups, dstate)
    dt_ptr,             # 时间步 ∆，(nheads, nchunks, chunk_size)
    dA_cumsum_ptr,      # dA 累积和，(nheads, nchunks, chunk_size)
    chunk_states_ptr,   # 跨 chunk 传递的局部状态，(nchunks, nheads, hdim, dstate)
    cu_seqlens_ptr,     # 序列累计长度（前缀和），(batch+1,)
    states_ptr,         # 输出：各序列末尾的 SSM 状态，(batch, nheads, hdim, dstate)
    initstates_ptr,     # 初始 SSM 状态（radix cache），(batch, nheads, hdim, dstate)
    # Matrix dimensions
    hdim,          # head 维度
    dstate,        # SSM 状态维度
    chunk_size,    # chunk 大小
    seqlen,        # 总序列长度
    nheads_ngroups_ratio,  # nheads // ngroups
    # Strides
    stride_x_seqlen,       # x 的 seqlen 步长
    stride_x_head,         # x 的 head 步长
    stride_x_hdim,         # x 的 hdim 步长
    stride_b_seqlen,       # B 的 seqlen 步长
    stride_b_head,         # B 的 group 步长
    stride_b_dstate,       # B 的 dstate 步长
    stride_dt_chunk,       # dt 的 chunk 步长
    stride_dt_head,        # dt 的 head 步长
    stride_dt_csize,       # dt 的 chunk_size 内步长
    stride_dA_cs_chunk,    # dA_cumsum 的 chunk 步长
    stride_dA_cs_head,     # dA_cumsum 的 head 步长
    stride_dA_cs_csize,    # dA_cumsum 的 chunk_size 内步长
    stride_chunk_states_chunk,   # chunk_states 的 chunk 步长
    stride_chunk_states_head,    # chunk_states 的 head 步长
    stride_chunk_states_hdim,    # chunk_states 的 hdim 步长
    stride_chunk_states_dstate,  # chunk_states 的 dstate 步长
    stride_states_batch,   # states 的 batch 步长
    stride_states_head,    # states 的 head 步长
    stride_states_hdim,    # states 的 hdim 步长
    stride_states_dstate,  # states 的 dstate 步长
    stride_init_states_batch,   # initstates 的 batch 步长
    stride_init_states_head,    # initstates 的 head 步长
    stride_init_states_hdim,    # initstates 的 hdim 步长
    stride_init_states_dstate,  # initstates 的 dstate 步长
    # Meta-parameters
    HAS_INITSTATES: tl.constexpr,     # 是否有初始状态（radix cache）
    BLOCK_SIZE_M: tl.constexpr = 16,  # hdim 分块大小
    BLOCK_SIZE_N: tl.constexpr = 16,  # dstate 分块大小
    BLOCK_SIZE_K: tl.constexpr = 16,  # chunk_size 分块大小
):
    # axis=1: batch（每个序列一个 block）；axis=2: head；axis=0: hdim块*dstate块
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    # 从 cu_seqlens 读取当前序列的末尾位置（全局绝对位置）
    end_idx = tl.load(cu_seqlens_ptr + pid_b + 1)
    # 计算序列末尾所在的 chunk 索引
    pid_c = (end_idx - 1) // chunk_size
    # 定位各指针到序列末尾所在的 chunk
    b_ptr += (
        pid_c * chunk_size * stride_b_seqlen
        + (pid_h // nheads_ngroups_ratio) * stride_b_head
    )
    x_ptr += pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    chunk_states_ptr += (
        pid_c * stride_chunk_states_chunk + pid_h * stride_chunk_states_head
    )

    if HAS_INITSTATES:
        # if there are init states provided, we differentiate between states (which
        # are boundary conditions at a chunk boundary) and initstates (which are boundary
        # conditions when a new example in a cont batch starts)
        # 有初始状态：分别处理 chunk 边界状态（states）和 initstates（新序列起始状态）
        initstates_ptr += pid_h * stride_init_states_head

    # 各维度的分块偏移
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (
        offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen
    )
    b_ptrs = b_ptr + (
        offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen
    )
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    # 加载序列末尾所在位置的 dA_cumsum（作为累积和归一化基准）
    dA_cs_last = tl.load(
        dA_cumsum_ptr + (end_idx - pid_c * chunk_size - 1) * stride_dA_cs_csize
    ).to(tl.float32)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize

    # 当前序列在末尾 chunk 内的有效范围：[start_idx_cur, end_idx - pid_c * chunk_size)
    chunk_size_limit = end_idx - pid_c * chunk_size
    start_idx = tl.load(cu_seqlens_ptr + pid_b)  # 序列起始的全局绝对位置
    start_idx_cur = tl.maximum(start_idx - pid_c * chunk_size, 0)  # 序列在当前 chunk 内的起始偏移

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # 在当前序列范围内聚合状态（mask 确保只处理当前序列的部分）
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < hdim)
            & (offs_k[None, :] < chunk_size_limit - k)
            & (offs_k[None, :] >= start_idx_cur - k),  # 只取当前序列范围
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < chunk_size_limit - k)
            & (offs_n[None, :] < dstate)
            & (offs_k[:, None] >= start_idx_cur - k),
            other=0.0,
        ).to(tl.float32)
        dA_cs_k = tl.load(
            dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0
        ).to(tl.float32)
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(
            tl.float32
        )
        # 时间衰减 scale（只在序列有效范围内计算，越界置 0）
        scale = tl.where(
            (offs_k >= start_idx_cur - k) & (offs_k < chunk_size_limit - k),
            tl.exp(dA_cs_last - dA_cs_k) * dt_k,
            0.0,
        )
        b *= scale[:, None]
        b = b.to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, b)  # 累加 chunk 内状态贡献
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    # If the sequence starts after the last chunk idx, we don't need to add the contribution from the last chunk
    # If HAS_INITSTATES==True need to consider two possibilities
    # - if start_idx < pid_c * chunk_size, then we need to take the past_states_ptrs
    # - if state_idx >= pid * chunk_size, then we need to insert initstates
    if (start_idx < pid_c * chunk_size) or (HAS_INITSTATES):  # first chunk
        # 序列起始在当前 chunk 之前（或有 initstates）：需要加入跨 chunk 传递的过去状态

        dA_cs_boundary = 0.0  # default

        if not HAS_INITSTATES:
            # 无初始状态：直接使用 chunk_states（跨 chunk 传递的 SSM 状态）
            past_states_ptrs = chunk_states_ptr + (
                offs_m[:, None] * stride_chunk_states_hdim
                + offs_n[None, :] * stride_chunk_states_dstate
            )
        else:

            # - this seems repetitive, buts its to help the compiler
            if start_idx < pid_c * chunk_size:
                # 序列起始早于当前 chunk：使用 chunk_states
                past_states_ptrs = chunk_states_ptr + (
                    offs_m[:, None] * stride_chunk_states_hdim
                    + offs_n[None, :] * stride_chunk_states_dstate
                )
            else:
                # 序列起始在当前 chunk 内：使用 initstates（radix cache 前缀状态）
                past_states_ptrs = initstates_ptr + (
                    pid_b * stride_init_states_batch
                    + offs_m[:, None] * stride_init_states_hdim
                    + offs_n[None, :] * stride_init_states_dstate
                )

                # need to adjust the boundary
                # 序列不从 chunk 起始开始：需修正 dA_cumsum 边界（减去起始偏移前的累积值）
                if start_idx > pid_c * chunk_size:
                    dA_cs_boundary = tl.load(
                        dA_cumsum_ptr
                        + (start_idx - pid_c * chunk_size - 1) * stride_dA_cs_csize
                    ).to(tl.float32)

        # 加载过去状态（来自 chunk_states 或 initstates）
        past_states = tl.load(
            past_states_ptrs,
            mask=(offs_m[:, None] < hdim) & (offs_n[None, :] < dstate),
            other=0.0,
        ).to(tl.float32)

        # 将过去状态传播到序列末尾（exp(dA_cs_last - dA_cs_boundary) 为衰减因子）
        scale = tl.exp(dA_cs_last - dA_cs_boundary)
        acc += past_states * scale  # 加入跨 chunk 传递的状态贡献

    states = acc.to(states_ptr.dtype.element_ty)

    # 写出当前序列的最终 SSM 状态
    states_ptr += pid_b * stride_states_batch + pid_h * stride_states_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (
        offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate
    )
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    # tl.store：写出序列最终 SSM 状态（varlen 模式的输出）
    tl.store(states_ptrs, states, mask=c_mask)
    start_idx_cur = tl.maximum(start_idx - pid_c * chunk_size, 0)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < hdim)
            & (offs_k[None, :] < chunk_size_limit - k)
            & (offs_k[None, :] >= start_idx_cur - k),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < chunk_size_limit - k)
            & (offs_n[None, :] < dstate)
            & (offs_k[:, None] >= start_idx_cur - k),
            other=0.0,
        ).to(tl.float32)
        dA_cs_k = tl.load(
            dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0
        ).to(tl.float32)
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(
            tl.float32
        )
        scale = tl.where(
            (offs_k >= start_idx_cur - k) & (offs_k < chunk_size_limit - k),
            tl.exp(dA_cs_last - dA_cs_k) * dt_k,
            0.0,
        )
        b *= scale[:, None]
        b = b.to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, b)
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    # If the sequence starts after the last chunk idx, we don't need to add the contribution from the last chunk
    # If HAS_INITSTATES==True need to consider two possibilities
    # - if start_idx < pid_c * chunk_size, then we need to take the past_states_ptrs
    # - if state_idx >= pid * chunk_size, then we need to insert initstates
    if (start_idx < pid_c * chunk_size) or (HAS_INITSTATES):  # first chunk

        dA_cs_boundary = 0.0  # default

        if not HAS_INITSTATES:
            past_states_ptrs = chunk_states_ptr + (
                offs_m[:, None] * stride_chunk_states_hdim
                + offs_n[None, :] * stride_chunk_states_dstate
            )
        else:

            # - this seems repetitive, buts its to help the compiler
            if start_idx < pid_c * chunk_size:
                past_states_ptrs = chunk_states_ptr + (
                    offs_m[:, None] * stride_chunk_states_hdim
                    + offs_n[None, :] * stride_chunk_states_dstate
                )
            else:
                past_states_ptrs = initstates_ptr + (
                    pid_b * stride_init_states_batch
                    + offs_m[:, None] * stride_init_states_hdim
                    + offs_n[None, :] * stride_init_states_dstate
                )

                # need to adjust the boundary
                if start_idx > pid_c * chunk_size:
                    dA_cs_boundary = tl.load(
                        dA_cumsum_ptr
                        + (start_idx - pid_c * chunk_size - 1) * stride_dA_cs_csize
                    ).to(tl.float32)

        past_states = tl.load(
            past_states_ptrs,
            mask=(offs_m[:, None] < hdim) & (offs_n[None, :] < dstate),
            other=0.0,
        ).to(tl.float32)

        scale = tl.exp(dA_cs_last - dA_cs_boundary)
        acc += past_states * scale

    states = acc.to(states_ptr.dtype.element_ty)

    states_ptr += pid_b * stride_states_batch + pid_h * stride_states_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (
        offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate
    )
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)



# Python 封装：计算 dA 累积和，返回 (dA_cumsum, dt_out)
def _chunk_cumsum_fwd(
    dt, A, chunk_size, dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf"))
):
    batch, seqlen, nheads = dt.shape
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
    nchunks = math.ceil(seqlen / chunk_size)
    # 分配输出 dt_out（处理后的 ∆）和 dA_cumsum（累积和），均为 float32
    dt_out = torch.empty(
        batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32
    )
    dA_cumsum = torch.empty(
        batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32
    )
    # grid：(batch, nchunks, nheads_blocks)
    grid_chunk_cs = lambda META: (
        batch,
        nchunks,
        triton.cdiv(nheads, META["BLOCK_SIZE_H"]),
    )
    with torch.get_device_module(dt.device).device(dt.device.index):
        _chunk_cumsum_fwd_kernel[grid_chunk_cs](
            dt,
            A,
            dt_bias,
            dt_out,
            dA_cumsum,
            batch,
            seqlen,
            nheads,
            chunk_size,
            dt_limit[0],   # dt_min
            dt_limit[1],   # dt_max
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            A.stride(0),
            dt_bias.stride(0) if dt_bias is not None else 0,
            dt_out.stride(0),
            dt_out.stride(2),  # 注意：dt_out 维度为 (batch, nheads, nchunks, chunk_size)
            dt_out.stride(1),
            dt_out.stride(3),
            dA_cumsum.stride(0),
            dA_cumsum.stride(2),
            dA_cumsum.stride(1),
            dA_cumsum.stride(3),
            dt_softplus,
            HAS_DT_BIAS=dt_bias is not None,
            BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),
        )
    return dA_cumsum, dt_out


# Python 封装：聚合每个 chunk 内的局部 SSM 状态
def _chunk_state_fwd(
    B, x, dt, dA_cumsum, seq_idx=None, states=None, states_in_fp32=True
):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if states is not None:
        assert states.shape == (batch, nchunks, nheads, headdim, dstate)
    else:
        # 分配输出 states 张量（默认 float32 保证精度）
        states_dtype = torch.float32 if states_in_fp32 else B.dtype
        states = torch.empty(
            (batch, nchunks, nheads, headdim, dstate),
            device=x.device,
            dtype=states_dtype,
        )
    # grid：(hdim块*dstate块, batch*nchunks, nheads)
    grid = lambda META: (
        triton.cdiv(headdim, META["BLOCK_SIZE_M"])
        * triton.cdiv(dstate, META["BLOCK_SIZE_N"]),
        batch * nchunks,
        nheads,
    )
    with torch.get_device_module(x.device).device(x.device.index):
        _chunk_state_fwd_kernel[grid](
            x,
            B,
            states,
            dt,
            dA_cumsum,
            seq_idx,
            headdim,
            dstate,
            chunk_size,
            batch,
            seqlen,
            nheads // ngroups,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            B.stride(-1),
            states.stride(0),
            states.stride(1),
            states.stride(2),
            states.stride(3),
            states.stride(4),
            dt.stride(0),
            dt.stride(2),   # 注意：dt 维度为 (batch, nheads, nchunks, chunk_size)
            dt.stride(1),
            dt.stride(3),
            dA_cumsum.stride(0),
            dA_cumsum.stride(2),
            dA_cumsum.stride(1),
            dA_cumsum.stride(3),
            *(
                (seq_idx.stride(0), seq_idx.stride(1))
                if seq_idx is not None
                else (0, 0)
            ),
            HAS_SEQ_IDX=seq_idx is not None,
        )
    return states


# Python 封装：varlen 模式下计算每个序列的最终 SSM 状态
def chunk_state_varlen(
    B, x, dt, dA_cumsum, cu_seqlens, chunk_states, initial_states=None
):
    total_seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    batch = cu_seqlens.shape[0] - 1  # 序列数 = cu_seqlens 长度 - 1
    cu_seqlens = cu_seqlens.contiguous()
    assert nheads % ngroups == 0
    assert B.shape == (total_seqlen, ngroups, dstate)
    assert dt.shape == (nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert chunk_states.shape == (nchunks, nheads, headdim, dstate)

    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)

    # 分配输出：每个序列的最终 SSM 状态
    states = torch.empty(
        batch,
        nheads,
        headdim,
        dstate,
        dtype=chunk_states.dtype,
        device=chunk_states.device,
    )
    # grid：(hdim块*dstate块, batch, nheads)
    grid = lambda META: (
        triton.cdiv(headdim, META["BLOCK_SIZE_M"])
        * triton.cdiv(dstate, META["BLOCK_SIZE_N"]),
        batch,
        nheads,
    )
    with torch.get_device_module(x.device).device(x.device.index):
        _chunk_state_varlen_kernel[grid](
            x,
            B,
            dt,
            dA_cumsum,
            chunk_states,
            cu_seqlens,
            states,
            initial_states,
            headdim,
            dstate,
            chunk_size,
            total_seqlen,
            nheads // ngroups,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            dt.stride(1),   # 注意：dt 维度为 (nheads, nchunks, chunk_size)
            dt.stride(0),
            dt.stride(2),
            dA_cumsum.stride(1),
            dA_cumsum.stride(0),
            dA_cumsum.stride(2),
            chunk_states.stride(0),
            chunk_states.stride(1),
            chunk_states.stride(2),
            chunk_states.stride(3),
            states.stride(0),
            states.stride(1),
            states.stride(2),
            states.stride(3),
            *(
                (
                    initial_states.stride(0),
                    initial_states.stride(1),
                    initial_states.stride(2),
                    initial_states.stride(3),
                )
                if initial_states is not None
                else (0, 0, 0, 0)
            ),
            HAS_INITSTATES=initial_states is not None,
        )
    return states
