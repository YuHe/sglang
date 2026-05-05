# Adapted from: https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/mamba/ops/ssd_state_passing.py

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_state_passing.py

# SSM 状态传递 Triton kernel：跨 chunk 传播 SSM 隐藏状态 h_t，支持连续批处理和 radix cache 初始状态

# ruff: noqa: E501

import torch
import triton
import triton.language as tl


# Triton JIT kernel：按 chunk 顺序传播 SSM 隐藏状态
# 每个 thread block 处理一个 (batch, head) 的所有 chunk 在 dim 维度的某一分块
@triton.jit
def _state_passing_fwd_kernel(
    # Pointers to matrices
    states_ptr,       # 各 chunk 的局部 SSM 状态，形状 (batch, nchunks, nheads, dim)
    out_ptr,          # 输出：每个 chunk 开始时的全局状态（用于 chunk 内 scan 的初始值）
    final_states_ptr, # 输出：整个序列结束后的最终 SSM 状态（用于 decode 缓存）
    dA_cs_ptr,        # dA 的 cumsum（累积和），形状 (batch, nheads, nchunks, chunk_size)
    initstates_ptr,   # 初始状态（radix cache 命中时预填充的状态），可为 None
    seq_idx_ptr,      # 序列 ID，标记 batch 内不同序列的边界（连续批处理用）
    chunk_offsets_ptr, # 逻辑 chunk 偏移，用于修正跨序列边界的 dA_cumsum
    chunk_meta_num,   # chunk_offsets 的长度
    # Matrix dimensions
    dim,       # SSM 状态向量维度（d_state）
    nchunks,   # chunk 总数（nchunks = ceil(seqlen / chunk_size)）
    seqlen,    # 序列总长度
    chunk_size, # 每个 chunk 的大小
    # Strides
    stride_states_batch,  # states 的 batch 步长
    stride_states_chunk,  # states 的 chunk 步长
    stride_states_head,   # states 的 head 步长
    stride_states_dim,    # states 的 dim 步长
    stride_out_batch,     # out 的 batch 步长
    stride_out_chunk,     # out 的 chunk 步长
    stride_out_head,      # out 的 head 步长
    stride_out_dim,       # out 的 dim 步长
    stride_final_states_batch,  # final_states 的 batch 步长
    stride_final_states_head,   # final_states 的 head 步长
    stride_final_states_dim,    # final_states 的 dim 步长
    stride_dA_cs_batch,   # dA_cumsum 的 batch 步长
    stride_dA_cs_chunk,   # dA_cumsum 的 chunk 步长
    stride_dA_cs_head,    # dA_cumsum 的 head 步长
    stride_dA_cs_csize,   # dA_cumsum 的 chunk_size 内步长
    stride_initstates_batch,  # initstates 的 batch 步长（连续批处理时为序列索引）
    stride_initstates_head,   # initstates 的 head 步长
    stride_initstates_dim,    # initstates 的 dim 步长
    stride_seq_idx_batch,     # seq_idx 的 batch 步长
    stride_seq_idx_seqlen,    # seq_idx 的 seqlen 步长
    # Meta-parameters
    HAS_INITSTATES: tl.constexpr,   # 是否有初始状态（radix cache 命中时为 True）
    HAS_SEQ_IDX: tl.constexpr,      # 是否启用序列 ID（连续批处理时为 True）
    IS_CONT_BATCHED: tl.constexpr,  # 是否为连续批处理（多序列拼接在一个 batch 维度）
    BLOCK_SIZE: tl.constexpr = 16,  # dim 维度的分块大小
):
    # axis=1: batch 索引；axis=2: head 索引；axis=0: dim 分块索引
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)
    # 定位当前 (batch, head) 的指针起始位置
    states_ptr += pid_b * stride_states_batch + pid_h * stride_states_head
    # dA_cs_ptr 指向当前 chunk 末位（cumsum 的最后一个位置）
    dA_cs_ptr += (
        pid_b * stride_dA_cs_batch
        + pid_h * stride_dA_cs_head
        + (chunk_size - 1) * stride_dA_cs_csize  # 定位到 chunk 内的最后一个时间步
    )
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head
    final_states_ptr += (
        pid_b * stride_final_states_batch + pid_h * stride_final_states_head
    )
    # 初始化状态指针（连续批处理时不按 batch 偏移，而是按序列 ID 索引）
    if HAS_INITSTATES:
        initstates_ptr += pid_h * stride_initstates_head
        if not IS_CONT_BATCHED:
            # 普通批处理：按 batch 维度取各自的初始状态
            initstates_ptr += pid_b * stride_initstates_batch

    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch

    # 当前 dim 分块的偏移量（每个 thread block 处理 dim 的一个 BLOCK_SIZE 切片）
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    states_ptrs = states_ptr + offs_m * stride_states_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim
    final_states_ptrs = final_states_ptr + offs_m * stride_final_states_dim

    # - states will be the past state of the sequence that continues on the current check
    # 初始化 SSM 状态：无初始状态时归零，否则从 initstates 加载
    if not HAS_INITSTATES:
        states = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    else:
        initstates_ptr += offs_m * stride_initstates_dim
        initstates_ptrs = initstates_ptr
        # - for cont batches, for the first chunk mean it will be the first batch's
        #   init state
        # 加载第 0 序列（或当前 batch）的初始 SSM 状态
        states = tl.load(initstates_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)

    # 将第 0 个 chunk 的初始状态写入 out（chunk 0 以当前 states 作为其初始值）
    tl.store(out_ptrs, states, mask=offs_m < dim)
    out_ptrs += stride_out_chunk  # 推进到下一个 chunk 的输出位置
    prev_seq_idx_chunk_end = 0    # 记录上一 chunk 末尾的序列 ID（用于检测序列边界）
    logical_chunk_idx = 0         # 逻辑 chunk 索引（跨序列边界时需要单独计数）
    # 逐 chunk 迭代，传播 SSM 状态
    for c in range(nchunks):
        # 加载当前 chunk 产生的局部 SSM 状态增量（由 chunk scan 计算得到）
        new_states = tl.load(states_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        # 加载当前 chunk 末位的 dA 累积和（= sum(dt * A) over the chunk）
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        scale_mask = True  # 默认传播上一 chunk 的状态（乘以 exp(dA_cs)）
        if HAS_SEQ_IDX:
            # - the seq to pass forward is the one that is flushed to the right
            #   boundary.
            # - that is given by seq_idx_chunk_end below: the sequence index at the end of the chunk.
            # 当前 chunk 末尾位置的序列 ID（用于检测序列边界）
            seq_idx_chunk_end = tl.load(
                seq_idx_ptr
                + (min((c + 1) * chunk_size, seqlen) - 1) * stride_seq_idx_seqlen
            )
            if HAS_INITSTATES:
                if IS_CONT_BATCHED and prev_seq_idx_chunk_end != seq_idx_chunk_end:
                    # this means in the current chunk the rightmost flushed seq
                    # has changed.
                    # - so we do not propagate the state from previous chunk
                    # - but rather we load that sequence's init state
                    # 序列边界发生变化：放弃上一序列状态，改用新序列的初始状态
                    initstates_ptrs = (
                        initstates_ptr + seq_idx_chunk_end * stride_initstates_batch
                    )

                    # - update state with seq_idx_new's init state
                    # 加载新序列的 initstate（用于 radix cache 命中时传入的前缀状态）
                    states = tl.load(initstates_ptrs, mask=offs_m < dim, other=0.0).to(
                        tl.float32
                    )

                    # - we need to consider the cumsum only of the last sequence in the chunk
                    # - find its starting position (given by c_off of the logical chunk index)
                    # - and subtract the cumsum just before that position from the total cumsum
                    # - first, update the logical chunk index (add the number of sequences in the current physical chunk):
                    # sequence index at the start of the current chunk
                    # 计算当前 chunk 中包含了多少个序列，更新逻辑 chunk 索引
                    seq_idx_chunk_start = tl.load(
                        seq_idx_ptr
                        + min(c * chunk_size, seqlen) * stride_seq_idx_seqlen
                    )
                    logical_chunk_idx += seq_idx_chunk_end - seq_idx_chunk_start
                    # - load the chunk offset:
                    # 加载逻辑 chunk 在物理 chunk 内的偏移量（新序列开始位置）
                    c_off = tl.load(
                        chunk_offsets_ptr + logical_chunk_idx,
                        mask=logical_chunk_idx < chunk_meta_num,
                        other=0,
                    )
                    # - if offset is 0, then the sequence starts at the beginning of the chunk, and we don't need to subtract anything
                    if c_off > 0:
                        # - dA_cs_ptr currently points to the cumsum at the end of the chunk - subtract the chunk size and add the offset
                        # 序列不从 chunk 起始开始：需从 dA_cumsum 中减去边界前的部分，
                        # 保证只累积当前序列范围内的 dA_cumsum
                        dA_cs_boundary = tl.load(
                            dA_cs_ptr
                            - (chunk_size - 1) * stride_dA_cs_csize
                            + (c_off - 1) * stride_dA_cs_csize,
                            mask=(c_off - 1) > -1 and c_off < chunk_size,
                            other=0.0,
                        )
                        dA_cs -= dA_cs_boundary  # 减去边界值，得到该序列段的 dA_cumsum

                # - increment logical chunk index for every physical chunk
                logical_chunk_idx += 1  # 每处理一个物理 chunk，逻辑 chunk 索引自增 1
            else:
                # 无初始状态的连续批处理：跨序列边界时 scale_mask=False，不传播状态
                scale_mask = seq_idx_chunk_end == prev_seq_idx_chunk_end
            prev_seq_idx_chunk_end = seq_idx_chunk_end  # 更新上一 chunk 末尾序列 ID

        # scale = exp(dA_cs) 如果在同一序列内传播状态，否则 scale = 0（不传播）
        scale = tl.where(scale_mask, tl.exp(dA_cs), 0.0)
        # SSM 状态递推：h_{c+1} = exp(dA_cs) * h_c + new_states
        states = scale * states + new_states
        if c < nchunks - 1:
            # 中间 chunk：将传播后的状态写入 out（作为下一 chunk 的初始状态）
            tl.store(out_ptrs, states, mask=offs_m < dim)
        else:
            # 最后一个 chunk：写入 final_states（用于 decode 阶段缓存）
            tl.store(final_states_ptrs, states, mask=offs_m < dim)
        # 推进 states/dA_cs/out 指针到下一个 chunk
        states_ptrs += stride_states_chunk
        dA_cs_ptr += stride_dA_cs_chunk
        out_ptrs += stride_out_chunk


# Python 封装：验证张量形状并调用 Triton kernel，返回 (chunk 初始状态序列, 最终状态)
def _state_passing_fwd(
    states,               # 各 chunk 的局部 SSM 状态，(batch, nchunks, nheads, dim)
    dA_cumsum,            # dA 累积和，(batch, nheads, nchunks, chunk_size)
    initial_states=None,  # 初始 SSM 状态，(batch, nheads, dim) 或 (nseqs, nheads, dim)
    seq_idx=None,         # 序列 ID，(batch, seqlen)，连续批处理时提供
    chunk_size=None,      # chunk 大小（默认从 dA_cumsum 推导）
    out_dtype=None,       # 输出 dtype（默认与 states 相同）
    is_cont_batched=False, # 是否为连续批处理模式
    chunk_offsets=None,   # 逻辑 chunk 偏移，连续批处理时必须提供
):
    batch, nchunks, nheads, dim = states.shape
    if chunk_size is None:
        chunk_size = dA_cumsum.shape[-1]
    else:
        assert chunk_size == dA_cumsum.shape[-1]
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    if initial_states is not None:
        if is_cont_batched:
            # - if cu_seqlens is provided, then the initial states
            #   are used for continuous batching. In which case we
            #   require seq_idx to be provided
            # 连续批处理模式：需要 seq_idx 标识序列边界
            assert (
                seq_idx is not None
            ), "seq_idx must be provided for continuous batching"
            # - we also need chunk_offsets to be provided, to account
            #   for computation of dA_cumsum from the start of the
            #   sequence
            # 同时需要 chunk_offsets 用于修正跨序列边界的 dA_cumsum
            assert (
                chunk_offsets is not None
            ), "chunk_offsets must be provided for continuous batching"
        else:
            # - this is the regular batching case, where initial
            #   states are used are for each example of the batch.
            # 常规批处理：初始状态逐 batch 独立提供
            assert initial_states.shape == (batch, nheads, dim)

    if seq_idx is not None:
        seqlen = seq_idx.shape[-1]
        assert seq_idx.shape == (batch, seqlen)
    # 输出 dtype，默认复用 states 的 dtype
    out_dtype = states.dtype if out_dtype is None else out_dtype
    # out：存储每个 chunk 开始时刻的全局 SSM 状态（chunk_scan 的初始值来源）
    out = torch.empty(
        (batch, nchunks, nheads, dim), device=states.device, dtype=out_dtype
    )
    # final_states：存储序列结束时的最终 SSM 状态（decode 阶段 cache）
    final_states = torch.empty(
        (batch, nheads, dim), device=states.device, dtype=torch.float32
    )
    # 启动 grid：axis0 = dim 分块数，axis1 = batch，axis2 = nheads
    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE"]), batch, nheads)
    with torch.get_device_module(states.device).device(states.device.index):
        _state_passing_fwd_kernel[grid](
            states,
            out,
            final_states,
            dA_cumsum,
            initial_states,
            seq_idx,
            chunk_offsets,
            len(chunk_offsets) if chunk_offsets is not None else 0,
            dim,
            nchunks,
            seqlen if seq_idx is not None else 0,  # 无 seq_idx 时 seqlen 为 0
            chunk_size,
            states.stride(0),
            states.stride(1),
            states.stride(2),
            states.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            final_states.stride(0),
            final_states.stride(1),
            final_states.stride(2),
            dA_cumsum.stride(0),
            dA_cumsum.stride(2),  # 注意：dim 1 是 nheads，dim 2 是 nchunks
            dA_cumsum.stride(1),
            dA_cumsum.stride(3),
            *(
                (
                    initial_states.stride(0),  # batch/seq 维度步长
                    initial_states.stride(1),  # head 维度步长
                    initial_states.stride(2),  # dim 维度步长
                )
                if initial_states is not None
                else (0, 0, 0)
            ),
            *(
                (seq_idx.stride(0), seq_idx.stride(1))
                if seq_idx is not None
                else (0, 0)
            ),
            HAS_INITSTATES=initial_states is not None,
            HAS_SEQ_IDX=seq_idx is not None,
            IS_CONT_BATCHED=is_cont_batched,
        )
    return out, final_states
