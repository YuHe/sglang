# Adapted from: https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/mamba/ops/ssd_chunk_scan.py

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_chunk_scan.py

# SSD 分块扫描 Triton kernel：计算 SSM 对角块输出（整合 off-diagonal 跨 chunk 状态和 diagonal chunk 内状态）

# ruff: noqa: E501,SIM102

import torch
import triton
import triton.language as tl
from packaging import version

# 检查 Triton 版本 >= 2.2（影响某些 layout 转换的编译行为）
TRITON_22 = version.parse(triton.__version__) >= version.parse("2.2.0")


# Triton JIT kernel：chunk 内因果扫描，计算 SSM 对角块及跨块状态贡献
# grid = (M块*N块, batch*nchunks 或 len(chunk_offsets), nheads)
@triton.jit
def _chunk_scan_fwd_kernel(
    # Pointers to matrices
    cb_ptr,           # C^T B 矩阵指针，(batch, nchunks, ngroups, chunk_size, chunk_size)
    x_ptr,            # 输入 x 指针，(batch, seqlen, nheads, hdim)
    z_ptr,            # 门控分支 z 指针，(batch, seqlen, nheads, hdim)，可为 None
    out_ptr,          # 输出指针（最终结果，可能含 z）
    out_x_ptr,        # 未加 z 的中间输出指针（仅 HAS_Z 时使用）
    dt_ptr,           # 时间步 ∆ 指针，(batch, nheads, nchunks, chunk_size)
    dA_cumsum_ptr,    # dA 累积和指针，(batch, nheads, nchunks, chunk_size)
    seq_idx_ptr,      # 序列 ID 指针，(batch, seqlen)，连续批处理时使用
    C_ptr,            # SSM 输出矩阵 C，(batch, seqlen, ngroups, dstate)
    states_ptr,       # 跨 chunk 传递的全局 SSM 状态，(batch, nchunks, nheads, hdim, dstate)
    D_ptr,            # 跳跃连接权重 D，(nheads, hdim) 或 (nheads,)
    initstates_ptr,   # 初始 SSM 状态（radix cache 用），(nseqs, nheads, hdim, dstate)
    chunk_indices_ptr,  # 逻辑 chunk 索引数组（连续批处理用）
    chunk_offsets_ptr,  # 逻辑 chunk 偏移（每个 chunk 在物理 chunk 内的起始位置）
    chunk_meta_num,   # chunk_indices/chunk_offsets 的长度
    # Matrix dimensions
    chunk_size,       # 每个 chunk 的大小
    hdim,             # head 维度（headdim）
    dstate,           # SSM 状态维度
    batch,            # batch 大小
    seqlen,           # 序列总长度
    nheads_ngroups_ratio,  # nheads // ngroups（B/C group 索引用）
    # Strides
    stride_cb_batch,    # cb 的 batch 步长
    stride_cb_chunk,    # cb 的 chunk 步长
    stride_cb_head,     # cb 的 head 步长
    stride_cb_csize_m,  # cb 的 m 维步长（行方向）
    stride_cb_csize_k,  # cb 的 k 维步长（列方向）
    stride_x_batch,     # x 的 batch 步长
    stride_x_seqlen,    # x 的 seqlen 步长
    stride_x_head,      # x 的 head 步长
    stride_x_hdim,      # x 的 hdim 步长
    stride_z_batch,     # z 的 batch 步长
    stride_z_seqlen,    # z 的 seqlen 步长
    stride_z_head,      # z 的 head 步长
    stride_z_hdim,      # z 的 hdim 步长
    stride_out_batch,   # out 的 batch 步长
    stride_out_seqlen,  # out 的 seqlen 步长
    stride_out_head,    # out 的 head 步长
    stride_out_hdim,    # out 的 hdim 步长
    stride_dt_batch,    # dt 的 batch 步长
    stride_dt_chunk,    # dt 的 chunk 步长
    stride_dt_head,     # dt 的 head 步长
    stride_dt_csize,    # dt 的 chunk_size 内步长
    stride_dA_cs_batch,  # dA_cumsum 的 batch 步长
    stride_dA_cs_chunk,  # dA_cumsum 的 chunk 步长
    stride_dA_cs_head,   # dA_cumsum 的 head 步长
    stride_dA_cs_csize,  # dA_cumsum 的 chunk_size 内步长
    stride_seq_idx_batch,    # seq_idx 的 batch 步长
    stride_seq_idx_seqlen,   # seq_idx 的 seqlen 步长
    stride_C_batch,     # C 的 batch 步长
    stride_C_seqlen,    # C 的 seqlen 步长
    stride_C_head,      # C 的 head 步长
    stride_C_dstate,    # C 的 dstate 步长
    stride_states_batch,   # states 的 batch 步长
    stride_states_chunk,   # states 的 chunk 步长
    stride_states_head,    # states 的 head 步长
    stride_states_hdim,    # states 的 hdim 步长
    stride_states_dstate,  # states 的 dstate 步长
    stride_init_states_batch,   # initstates 的 batch 步长（按 seq_idx 索引）
    stride_init_states_head,    # initstates 的 head 步长
    stride_init_states_hdim,    # initstates 的 hdim 步长
    stride_init_states_dstate,  # initstates 的 dstate 步长
    stride_D_head,    # D 的 head 步长
    # Meta-parameters
    IS_CAUSAL: tl.constexpr,       # 是否使用因果掩码（chunk 内下三角注意力）
    HAS_D: tl.constexpr,           # 是否有跳跃连接 D
    D_HAS_HDIM: tl.constexpr,      # D 是否有 hdim 维度（(nheads, hdim) 而非 (nheads,)）
    HAS_Z: tl.constexpr,           # 是否有门控分支 z
    HAS_SEQ_IDX: tl.constexpr,     # 是否启用序列 ID
    BLOCK_SIZE_DSTATE: tl.constexpr,  # dstate 维度分块大小
    IS_TRITON_22: tl.constexpr,    # Triton 版本 >= 2.2 的标志
    HAS_INITSTATES: tl.constexpr,  # 是否有初始 SSM 状态（连续批处理+radix cache）
    BLOCK_SIZE_M: tl.constexpr = 16,  # chunk_size 方向（行）的分块大小
    BLOCK_SIZE_N: tl.constexpr = 16,  # hdim 方向（列）的分块大小
    BLOCK_SIZE_K: tl.constexpr = 16,  # 内积维度的分块大小
):
    # axis=1: batch*chunk 组合索引；axis=2: head 索引；axis=0: M块*N块 组合索引
    pid_bc = tl.program_id(axis=1).to(tl.int64)
    pid_c = pid_bc // batch    # chunk 索引（可能是逻辑 chunk 索引）
    pid_b = pid_bc - pid_c * batch
    if not HAS_INITSTATES:
        # 无初始状态：逻辑 chunk 索引直接等于物理 chunk 索引，偏移为 0
        c_idx = pid_c
        c_off = 0
    else:
        # 有初始状态（连续批处理）：从 chunk_indices 读取物理 chunk 索引
        c_idx = tl.load(chunk_indices_ptr + pid_c, mask=pid_c > -1, other=0)
        # 从 chunk_offsets 读取当前逻辑 chunk 在物理 chunk 内的起始偏移
        c_off = tl.load(chunk_offsets_ptr + pid_c, mask=pid_c > -1, other=0)

    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n  # chunk_size 方向的块索引（行）
    pid_n = tl.program_id(axis=0) % num_pid_n   # hdim 方向的块索引（列）
    # 定位各指针到当前 (batch, chunk, head) 的起始位置
    cb_ptr += (
        pid_b * stride_cb_batch
        + c_idx * stride_cb_chunk
        + (pid_h // nheads_ngroups_ratio) * stride_cb_head  # B/C 按 group 共享
    )
    x_ptr += (
        pid_b * stride_x_batch
        + c_idx * chunk_size * stride_x_seqlen  # 跳到当前物理 chunk 的起始行
        + pid_h * stride_x_head
    )
    dt_ptr += pid_b * stride_dt_batch + c_idx * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += (
        pid_b * stride_dA_cs_batch
        + c_idx * stride_dA_cs_chunk
        + pid_h * stride_dA_cs_head
    )
    C_ptr += (
        pid_b * stride_C_batch
        + c_idx * chunk_size * stride_C_seqlen
        + (pid_h // nheads_ngroups_ratio) * stride_C_head
    )

    # M-block offsets and prev states
    #  - logic in next block may override these if there is an active offset
    # 当前 M 块的行偏移（c_off 保证从序列起始位置开始，而不是 chunk 物理起始）
    offs_m = pid_m * BLOCK_SIZE_M + c_off + tl.arange(0, BLOCK_SIZE_M)
    # 默认使用跨 chunk 传递来的全局状态作为初始状态
    prev_states_ptr = (
        states_ptr
        + pid_b * stride_states_batch
        + c_idx * stride_states_chunk
        + pid_h * stride_states_head
    )
    prev_states_hdim = stride_states_hdim
    prev_states_dstate = stride_states_dstate

    # 当前 chunk 实际有效长度（最后一个 chunk 可能不足 chunk_size）
    chunk_size_limit = min(chunk_size, seqlen - c_idx * chunk_size)
    if HAS_SEQ_IDX:
        seq_idx_ptr += (
            pid_b * stride_seq_idx_batch + c_idx * chunk_size * stride_seq_idx_seqlen
        )

        # - we only need seq_idx_prev to be aligned to chunk boundary
        # 加载前一个 chunk 末尾的序列 ID（用于检测序列边界）
        seq_idx_prev = tl.load(
            seq_idx_ptr - stride_seq_idx_seqlen, mask=c_idx >= 1, other=0
        )

        if HAS_INITSTATES:
            # if there are init states, we only need seq_idx_m to point
            # what is the current seq_idx

            # get current seq idx
            if (pid_m * BLOCK_SIZE_M + c_off) < chunk_size_limit:
                # 加载当前 M 块起始位置的序列 ID
                seq_idx_m = tl.load(
                    seq_idx_ptr
                    + (pid_m * BLOCK_SIZE_M + c_off) * stride_seq_idx_seqlen,
                )

                # - recall that in ssd_state_passing, for the case c_off == 0
                # i.e., the very first sequence, we made states_ptr hold its initial state
                # so this edge case is taken care of
                if (
                    (c_off == 0)
                    and (
                        seq_idx_prev != seq_idx_m
                    )  # if a seq is changed exactly on boundary
                    or (c_off > 0)  # implies a new example (pseudo chunk)
                ):
                    # 序列发生边界变化：用 initstates 代替全局状态（radix cache 命中）
                    # - replace prev_states_ptr with init_states
                    prev_states_ptr = (
                        initstates_ptr
                        + seq_idx_m * stride_init_states_batch  # 按 seq_idx 索引
                        + pid_h * stride_init_states_head
                    )
                    prev_states_hdim = stride_init_states_hdim  # override strides
                    prev_states_dstate = stride_init_states_dstate

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # 加载当前 M 块行的 dA_cumsum（= exp 后即状态传播衰减因子）
    dA_cs_m = tl.load(
        dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size, other=0.0
    ).to(tl.float32)

    # - handle chunk state limit
    if HAS_INITSTATES:

        # have to split this if otherwise compilation will have problems
        # 有初始状态时，需要修正 dA_cumsum 的边界（从 c_off 开始，而非 chunk 起始）
        dA_cs_m_boundary = 0.0

        # get the c_idx for the next (logica) chunk
        # 查询下一个逻辑 chunk 对应的物理 chunk 索引（用于检测是否有相邻序列）
        c_idx_n = tl.load(
            chunk_indices_ptr + (pid_c + 1),
            mask=pid_c > -1 and (pid_c + 1) < chunk_meta_num,
            other=-1,  # to trigger different chunk
        )

        # - there are things to consider
        # A. if c_off > 0 then we need to move the dA_cs boundary to ensure correct
        #    contribution of past states
        # B. if c_off_n < chunk_size_limit, then we need to adjust this so as not to
        #    encroach into the next sequence, where c_off_n is the offset of the next
        #    (logical) chunk.
        # An equivalent check for B is c_idx == c_idx_n, where there is repetition in
        # (logical) chunk indices.

        if (c_idx == c_idx_n) or c_off > 0:

            # get the next offset
            # 加载下一个逻辑 chunk 的偏移（用于限制当前 chunk 的有效范围）
            c_off_n = tl.load(
                chunk_offsets_ptr + (pid_c + 1),
                mask=pid_c > -1 and (pid_c + 1) < chunk_meta_num,
                other=chunk_size,
            )

            # in this case, adjust down the chunk_size_limit
            if c_idx == c_idx_n:
                # 与下一个逻辑 chunk 共享同一物理 chunk：限制到下一 chunk 起始位置
                chunk_size_limit = min(c_off_n, chunk_size_limit)

            # get the cs at the offset boundary
            # - c_off == 0 is a passthrough
            # - We need dA_cs at the boundary, defined by c_off - no need
            #   to increase pointer by pid_m (it is a constant offset,
            #   i.e. the same for all blocks)
            # 加载 c_off 位置的 dA_cumsum（用于归一化：减去起始偏移前的累积值）
            dA_cs_m_boundary = tl.load(
                dA_cumsum_ptr + (c_off - 1) * stride_dA_cs_csize,
                mask=(((c_off - 1) > -1) and ((c_off) < chunk_size)),
                other=0.0,
            ).to(tl.float32)

    if HAS_SEQ_IDX:
        # - handle seq idx when HAS_INITSTATES==False
        # 无初始状态的连续批处理：加载行的序列 ID（用于跨序列 scale 清零）
        if not HAS_INITSTATES:
            seq_idx_m = tl.load(
                seq_idx_ptr + offs_m * stride_seq_idx_seqlen,
                mask=offs_m < chunk_size_limit,
                other=-1,
            )

    # 初始化累加器（float32 精度）
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Without the if (pid_c > -1), with Triton 2.1.0, I get
    # Assertion `!(srcMmaLayout && dstMmaLayout) && "Unexpected mma -> mm a layout conversion"' failed.
    # With Triton 2.2.0, this works
    if IS_TRITON_22 or c_idx > -1:
        # Faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size 128
        # 一次性加载全部 dstate 维度（若 dstate <= 128），否则分块迭代
        offs_k_dstate = tl.arange(
            0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K
        )
        # C_ptrs：(chunk_size_M, dstate) 块的指针
        C_ptrs = C_ptr + (
            offs_m[:, None] * stride_C_seqlen + offs_k_dstate[None, :] * stride_C_dstate
        )

        # prev_states_ptrs：(dstate, hdim_N) 块的指针（转置访问）
        prev_states_ptrs = prev_states_ptr + (
            offs_n[None, :] * prev_states_hdim
            + offs_k_dstate[:, None] * prev_states_dstate
        )
        if HAS_SEQ_IDX:

            if not HAS_INITSTATES:
                # - this is for continuous batching where there is no init states
                # 无初始状态：跨序列边界时 scale=0（不传播状态）
                scale_m = tl.where(seq_idx_m == seq_idx_prev, tl.exp(dA_cs_m), 0.0)
            else:
                # - if there is initstates, we will rely on prev_states, no zeroing
                #   required.
                # 有初始状态：使用修正后的 dA_cumsum（减去边界值）
                scale_m = tl.exp(dA_cs_m - dA_cs_m_boundary)
        else:
            # 无序列 ID：直接使用 exp(dA_cs_m) 作为状态传播衰减因子
            scale_m = tl.exp(dA_cs_m)
        if BLOCK_SIZE_DSTATE <= 128:
            # dstate 较小：一次性加载 C 和 prev_states，执行 tl.dot
            C = tl.load(
                C_ptrs,
                mask=(offs_m[:, None] < chunk_size_limit)
                & (offs_k_dstate[None, :] < dstate),
                other=0.0,
            )

            prev_states = tl.load(
                prev_states_ptrs,
                mask=(offs_k_dstate[:, None] < dstate) & (offs_n[None, :] < hdim),
                other=0.0,
            )
            prev_states = prev_states.to(C_ptr.dtype.element_ty)
            # off-diagonal 贡献：acc = C @ prev_states * scale_m（跨 chunk 状态投影到输出）
            acc = tl.dot(C, prev_states) * scale_m[:, None]
        else:
            # dstate 较大：分块迭代 dstate 维度
            for k in range(0, dstate, BLOCK_SIZE_K):
                C = tl.load(
                    C_ptrs,
                    mask=(offs_m[:, None] < chunk_size_limit)
                    & (offs_k_dstate[None, :] < dstate - k),
                    other=0.0,
                )
                # C = (C * scale_m[:, None]).to(C_ptr.dtype.element_ty)
                prev_states = tl.load(
                    prev_states_ptrs,
                    mask=(offs_k_dstate[:, None] < dstate - k)
                    & (offs_n[None, :] < hdim),
                    other=0.0,
                )
                prev_states = prev_states.to(C_ptr.dtype.element_ty)
                acc += tl.dot(C, prev_states)  # 分块矩阵乘法累加
                C_ptrs += BLOCK_SIZE_K
                prev_states_ptrs += BLOCK_SIZE_K
            acc *= scale_m[:, None]  # 最后统一乘以传播衰减因子

    # chunk 内对角块计算：acc += CB * diag_scale * dt * x
    offs_k = tl.arange(0, BLOCK_SIZE_K) + c_off
    cb_ptrs = cb_ptr + (
        offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k
    )
    x_ptrs = x_ptr + (
        offs_k[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim
    )
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    # K_MAX：因果掩码时只计算下三角（i <= m 的 k 步）；否则处理全部有效长度
    K_MAX = (
        chunk_size_limit
        if not IS_CAUSAL
        else min((pid_m + 1) * BLOCK_SIZE_M, chunk_size_limit)
    )
    # 沿 chunk_size 方向分块迭代（对角块贡献：cb_ij * exp(dA_cs_i - dA_cs_j) * dt_j * x_j）
    for k in range(0, K_MAX, BLOCK_SIZE_K):
        # 加载 C^T B 矩阵块（chunk 内注意力分数，已在 ssd_bmm 中预计算）
        cb = tl.load(
            cb_ptrs,
            mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < chunk_size - k),
            other=0.0,
        ).to(tl.float32)
        # 加载当前 k 步的 dA_cumsum（用于计算时间衰减 exp(dA_cs_m - dA_cs_k)）
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(
            tl.float32
        )
        # If there's seq_idx, we already set cb[i, j] = 0 for seq_idx[i] != seq_idx[j].
        # So we don't need masking wrt seq_idx here.
        # 时间衰减：cb_ij *= exp(dA_cs_m_i - dA_cs_k_j)（从 j 到 i 的时间传播衰减）
        cb *= tl.exp(dA_cs_m[:, None] - dA_cs_k[None, :])
        # 加载 ∆ 并乘入（离散化 SSM 的 ∆B 项）
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
        cb *= dt_k
        if IS_CAUSAL:
            # 因果掩码：只保留 offs_m >= k + offs_k（下三角）的贡献
            mask = offs_m[:, None] >= k + offs_k[None, :]
            cb = tl.where(mask, cb, 0.0)
        cb = cb.to(x_ptr.dtype.element_ty)
        # 加载 x 块
        x = tl.load(
            x_ptrs,
            mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < hdim),
            other=0.0,
        )
        # tl.dot：对角块矩阵乘法 acc += cb @ x（CB 权重 × 输入 x）
        acc += tl.dot(cb, x)
        # 推进 chunk_size 方向的各指针
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    # 输出位置偏移（含 c_off 保证逻辑起始正确）
    offs_out_m = pid_m * BLOCK_SIZE_M + c_off + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if HAS_D:
        if D_HAS_HDIM:
            # D 有 hdim 维度：逐 hdim 元素加载
            D = tl.load(
                D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0
            ).to(tl.float32)
        else:
            # D 为标量（per-head）：直接加载一个值
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        # 加载对应的输入 x（用于 D 跳跃连接：y += D * x）
        x_residual = tl.load(
            x_ptr
            + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim),
            mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim),
            other=0.0,
        ).to(tl.float32)
        acc += x_residual * D  # 跳跃连接：y += D * x

    if HAS_Z:
        # 有门控分支时，先将 SSM 输出（无 z）存储到 out_x（供后续使用）
        out_x_ptr += (
            pid_b * stride_out_batch
            + c_idx * chunk_size * stride_out_seqlen
            + pid_h * stride_out_head
        )
        out_x_ptrs = out_x_ptr + (
            stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :]
        )
        # tl.store：写出未加 z 的中间 SSM 输出
        tl.store(
            out_x_ptrs,
            acc,
            mask=(offs_out_m[:, None] < chunk_size_limit)
            & (offs_out_n[None, :] < hdim),
        )

        # 加载门控分支 z
        z_ptr += (
            pid_b * stride_z_batch
            + c_idx * chunk_size * stride_z_seqlen
            + pid_h * stride_z_head
        )
        z_ptrs = z_ptr + (
            stride_z_seqlen * offs_out_m[:, None] + stride_z_hdim * offs_out_n[None, :]
        )
        z = tl.load(
            z_ptrs,
            mask=(offs_out_m[:, None] < chunk_size_limit)
            & (offs_out_n[None, :] < hdim),
            other=0.0,
        ).to(tl.float32)
        # 应用 SiLU 门控：y *= z * sigmoid(z)
        acc *= z * tl.sigmoid(z)

    # 定位最终输出指针并写出结果
    out_ptr += (
        pid_b * stride_out_batch
        + c_idx * chunk_size * stride_out_seqlen
        + pid_h * stride_out_head
    )
    out_ptrs = out_ptr + (
        stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :] * stride_out_hdim
    )
    # tl.store：写出最终 chunk scan 输出（含 D 跳连和 z 门控）
    tl.store(
        out_ptrs,
        acc,
        mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim),
    )


# Python 封装：验证形状、配置 grid 并启动 Triton kernel
def _chunk_scan_fwd(
    cb,            # C^T B 矩阵，(batch, nchunks, ngroups, chunk_size, chunk_size)
    x,             # 输入 x，(batch, seqlen, nheads, headdim)
    dt,            # 时间步 ∆，(batch, nheads, nchunks, chunk_size)
    dA_cumsum,     # dA 累积和，(batch, nheads, nchunks, chunk_size)
    C,             # SSM 输出矩阵 C，(batch, seqlen, ngroups, dstate)
    states,        # 跨 chunk 传递的全局 SSM 状态，(batch, nchunks, nheads, headdim, dstate)
    D=None,        # 跳跃连接 D，(nheads, headdim) 或 (nheads,)
    z=None,        # 门控分支 z，(batch, seqlen, nheads, headdim)
    seq_idx=None,  # 序列 ID，(batch, seqlen)
    chunk_indices=None,  # 逻辑 chunk 索引（连续批处理）
    chunk_offsets=None,  # 逻辑 chunk 偏移（连续批处理）
    initial_states=None, # 初始 SSM 状态（连续批处理+radix cache）
    out=None,            # 预分配输出张量（原位更新）
):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = C.shape
    assert nheads % ngroups == 0
    assert C.shape == (batch, seqlen, ngroups, dstate)
    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert states.shape == (batch, nchunks, nheads, headdim, dstate)

    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)

        if initial_states is not None:
            # with initial states, we need to take care of how
            # seq_idx crosses the boundaries
            # 有初始状态时只支持 batch=1（连续批处理+初始状态的组合较复杂）
            assert batch == 1, "chunk scan only supports initial states with batch 1"
            assert (
                chunk_indices is not None and chunk_offsets is not None
            ), "chunk_indices and chunk_offsets should have been set"
        else:
            chunk_indices, chunk_offsets = None, None
    else:
        chunk_indices, chunk_offsets = None, None

    assert out.shape == x.shape

    if z is not None:
        # 有门控时分配中间输出缓冲区（存储未加 z 的 SSM 输出）
        out_x = torch.empty_like(x)
        assert out_x.stride() == out.stride()
    else:
        out_x = None

    # 启动 grid：axis0 = M块*N块，axis1 = batch*nchunks 或逻辑 chunk 数，axis2 = nheads
    grid = lambda META: (
        triton.cdiv(chunk_size, META["BLOCK_SIZE_M"])
        * triton.cdiv(headdim, META["BLOCK_SIZE_N"]),
        batch * nchunks if chunk_offsets is None else len(chunk_offsets),
        nheads,
    )
    z_strides = (
        (z.stride(0), z.stride(1), z.stride(2), z.stride(3))
        if z is not None
        else (0, 0, 0, 0)
    )
    _chunk_scan_fwd_kernel[grid](
        cb,
        x,
        z,
        out,
        out_x,
        dt,
        dA_cumsum,
        seq_idx,
        C,
        states,
        D,
        initial_states,
        chunk_indices,
        chunk_offsets,
        len(chunk_indices) if chunk_indices is not None else 0,  # chunk_meta_num
        chunk_size,
        headdim,
        dstate,
        batch,
        seqlen,
        nheads // ngroups,   # nheads_ngroups_ratio
        cb.stride(0),
        cb.stride(1),
        cb.stride(2),
        cb.stride(3),
        cb.stride(4),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        z_strides[0],
        z_strides[1],
        z_strides[2],
        z_strides[3],
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        dt.stride(0),
        dt.stride(2),    # 注意：dt 维度顺序为 (batch, nheads, nchunks, chunk_size)
        dt.stride(1),
        dt.stride(3),
        dA_cumsum.stride(0),
        dA_cumsum.stride(2),
        dA_cumsum.stride(1),
        dA_cumsum.stride(3),
        *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        C.stride(3),
        states.stride(0),
        states.stride(1),
        states.stride(2),
        states.stride(3),
        states.stride(4),
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
        D.stride(0) if D is not None else 0,
        True,              # IS_CAUSAL：始终使用因果掩码
        D is not None,
        D.dim() == 2 if D is not None else True,  # D_HAS_HDIM
        BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        HAS_Z=z is not None,
        HAS_SEQ_IDX=seq_idx is not None,
        IS_TRITON_22=TRITON_22,
        HAS_INITSTATES=initial_states is not None,
    )
    return out_x
