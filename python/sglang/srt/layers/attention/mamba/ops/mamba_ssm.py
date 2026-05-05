# Adapted from: https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/mamba/ops/mamba_ssm.py

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/selective_state_update.py

# Mamba2 Triton selective_state_update kernel：decode 阶段对每个 token 执行一步 SSM 状态更新
# 支持连续批处理（state_batch_indices）、投机解码（disable_state_update）、
# EAGLE tree attention（retrieve_parent_token）、中间状态缓存（intermediate_states_buffer）

import torch
import triton
import triton.language as tl
from packaging import version

# 填充槽位标识符（连续批处理中跳过无效序列的标记值）
PAD_SLOT_ID = -1

# 检查 Triton 版本 >= 3.0（不同版本的 log1p 实现不同）
TRITON3 = version.parse(triton.__version__) >= version.parse("3.0.0")

if TRITON3:
    # Triton >= 3.0：使用 log(exp(dt) + 1) 实现 softplus（精度更高）
    @triton.jit
    def softplus(dt):
        dt = tl.where(dt <= 20.0, tl.math.log(tl.math.exp(dt) + 1), dt)
        return dt

else:
    # Triton < 3.0：使用 log1p(exp(dt)) 实现 softplus
    @triton.jit
    def softplus(dt):
        dt = tl.where(dt <= 20.0, tl.math.log1p(tl.exp(dt)), dt)
        return dt


# heuristics 自动推断各可选参数是否存在，减少 kernel 分支开销
@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics(
    {
        "HAS_STATE_BATCH_INDICES": lambda args: args["state_batch_indices_ptr"]
        is not None
    }
)
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])}
)
@triton.heuristics(
    {
        "CACHE_INTERMEDIATE_STATES": lambda args: args["intermediate_states_buffer"]
        is not None
    }
)
@triton.heuristics(
    {
        "HAS_EAGLE_TREE_CUSTOM_ATTN_MASK": lambda args: args[
            "retrieve_parent_token_ptr"
        ]
        is not None
    }
)
@triton.heuristics(
    {
        "HAS_INTERMEDIATE_STATE_INDICES": lambda args: args[
            "intermediate_state_indices_ptr"
        ]
        is not None
    }
)
# do_not_specialize=["T"]：T（步数）为运行时值，不作为编译常量（避免频繁重编译）
@triton.jit(do_not_specialize=["T"])
def _selective_scan_update_kernel(
    # Pointers to matrices
    state_ptr,       # SSM 状态张量指针，(batch, nheads, dim, dstate)
    x_ptr,           # 输入 x 指针，(batch, T, nheads, dim)
    dt_ptr,          # 时间步 ∆ 指针，(batch, T, nheads, dim)
    dt_bias_ptr,     # ∆ 偏置指针，(nheads, dim)，可为 None
    A_ptr,           # 状态转移矩阵对角线 A，(nheads, dim, dstate)，负实数
    B_ptr,           # SSM 输入矩阵 B，(batch, T, ngroups, dstate)
    C_ptr,           # SSM 输出矩阵 C，(batch, T, ngroups, dstate)
    D_ptr,           # 跳跃连接权重 D，(nheads, dim)，可为 None
    z_ptr,           # 门控分支 z，(batch, T, nheads, dim)，可为 None
    out_ptr,         # 输出指针，(batch, T, nheads, dim)，原位更新
    state_batch_indices_ptr,  # 连续批处理中 state 的 batch 索引，(batch,)，可为 None
    pad_slot_id,              # 填充槽位 ID（跳过无效序列）
    intermediate_states_buffer,  # 中间状态缓存 buffer（EAGLE 投机解码用）
    cache_steps,              # buffer 缓存的步数总量
    retrieve_parent_token_ptr,   # EAGLE tree attention 父 token 索引，(batch, T)
    intermediate_state_indices_ptr,  # buffer 操作的自定义索引，(batch,)
    # Matrix dimensions
    batch,       # batch 大小
    T,           # 每个 batch 的时间步数（通常为 1，EAGLE 时为树宽）
    nheads,      # head 数
    dim,         # 每个 head 的特征维度（headdim）
    dstate,      # SSM 状态维度（d_state）
    nheads_ngroups_ratio,  # nheads // ngroups（B/C 的 group 索引用）
    # Strides
    stride_state_batch,   # state 的 batch 步长
    stride_state_head,    # state 的 head 步长
    stride_state_dim,     # state 的 dim 步长
    stride_state_dstate,  # state 的 dstate 步长
    stride_x_batch,       # x 的 batch 步长
    stride_x_T,           # x 的 T 步长（时间维度）
    stride_x_head,        # x 的 head 步长
    stride_x_dim,         # x 的 dim 步长
    stride_dt_batch,      # ∆ 的 batch 步长
    stride_dt_T,          # ∆ 的 T 步长
    stride_dt_head,       # ∆ 的 head 步长
    stride_dt_dim,        # ∆ 的 dim 步长
    stride_dt_bias_head,  # dt_bias 的 head 步长
    stride_dt_bias_dim,   # dt_bias 的 dim 步长
    stride_A_head,        # A 的 head 步长
    stride_A_dim,         # A 的 dim 步长
    stride_A_dstate,      # A 的 dstate 步长
    stride_B_batch,       # B 的 batch 步长
    stride_B_T,           # B 的 T 步长
    stride_B_group,       # B 的 group 步长
    stride_B_dstate,      # B 的 dstate 步长
    stride_C_batch,       # C 的 batch 步长
    stride_C_T,           # C 的 T 步长
    stride_C_group,       # C 的 group 步长
    stride_C_dstate,      # C 的 dstate 步长
    stride_D_head,        # D 的 head 步长
    stride_D_dim,         # D 的 dim 步长
    stride_z_batch,       # z 的 batch 步长
    stride_z_T,           # z 的 T 步长
    stride_z_head,        # z 的 head 步长
    stride_z_dim,         # z 的 dim 步长
    stride_out_batch,     # out 的 batch 步长
    stride_out_T,         # out 的 T 步长
    stride_out_head,      # out 的 head 步长
    stride_out_dim,       # out 的 dim 步长
    stride_retrieve_parent_token_batch,  # retrieve_parent_token 的 batch 步长
    stride_retrieve_parent_token_T,      # retrieve_parent_token 的 T 步长
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,          # 是否对 ∆ 应用 softplus
    TIE_HDIM: tl.constexpr,             # True: 所有 dim 共享同一 ∆/A（即 A 为标量）
    BLOCK_SIZE_M: tl.constexpr,         # dim 维度的分块大小
    HAS_DT_BIAS: tl.constexpr,          # 是否有 ∆ 偏置
    HAS_D: tl.constexpr,                # 是否有跳跃连接 D
    HAS_Z: tl.constexpr,                # 是否有门控分支 z
    HAS_STATE_BATCH_INDICES: tl.constexpr,         # 是否有连续批处理 batch 索引
    DISABLE_STATE_UPDATE: tl.constexpr,            # True 时不回写 state（投机验证用）
    CACHE_INTERMEDIATE_STATES: tl.constexpr,       # 是否缓存中间状态
    HAS_EAGLE_TREE_CUSTOM_ATTN_MASK: tl.constexpr, # 是否启用 EAGLE tree attention
    HAS_INTERMEDIATE_STATE_INDICES: tl.constexpr,  # 是否有自定义 buffer 索引
    BLOCK_SIZE_DSTATE: tl.constexpr,    # dstate 维度的分块大小（2 的幂，自动推断）
):
    # axis=0: dim 分块；axis=1: batch；axis=2: head
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    # If HAS_STATE_BATCH_INDICES is true, then the ssm state's batch coordinate
    # is taken from the state_batch_indices_ptr Otherwise, the state coordinate
    # is the same as the batch id.
    # 连续批处理：从 state_batch_indices 读取状态对应的 batch 索引（非线性映射）
    if HAS_STATE_BATCH_INDICES:
        state_batch_indices_ptr += pid_b
        state_batch_idx = tl.load(state_batch_indices_ptr).to(tl.int64)
        state_ptr += state_batch_idx * stride_state_batch + pid_h * stride_state_head
    else:
        # 普通批处理：batch 索引与 pid_b 一致
        state_ptr += pid_b * stride_state_batch + pid_h * stride_state_head

    # 定位各指针到当前 (batch, head) 的起始位置
    x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_h * stride_dt_head
    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_head
    A_ptr += pid_h * stride_A_head
    # B/C 按 group 共享（nheads_ngroups_ratio 个 head 共享一个 group 的 B/C）
    B_ptr += pid_b * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    C_ptr += pid_b * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    if HAS_Z:
        z_ptr += pid_b * stride_z_batch + pid_h * stride_z_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head

    # dim/dstate 维度的偏移量数组（每个 thread block 处理一个 BLOCK_SIZE_M × BLOCK_SIZE_DSTATE 块）
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    state_ptrs = state_ptr + (
        offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
    )

    # 加载 SSM 状态（mask 保护越界，pad slot 不加载实际状态）
    mask = (offs_m[:, None] < dim) & (offs_n[None, :] < dstate)
    if HAS_STATE_BATCH_INDICES:
        mask &= state_batch_idx != pad_slot_id  # pad slot 强制 mask，不读状态
    state = tl.load(state_ptrs, mask=mask, other=0.0).to(tl.float32)

    if HAS_DT_BIAS:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
    if HAS_D:
        D_ptr += pid_h * stride_D_head
        D_ptrs = D_ptr + offs_m * stride_D_dim
    # A 矩阵指针：(dim, dstate) 二维寻址
    A_ptrs = A_ptr + offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate

    # 计算 cache_idx（中间状态 buffer 的写入位置索引）
    cache_idx = -1
    if CACHE_INTERMEDIATE_STATES:
        if HAS_INTERMEDIATE_STATE_INDICES:
            # 使用自定义索引（与 state_batch_indices 可能不同）
            intermediate_state_idx = tl.load(intermediate_state_indices_ptr + pid_b).to(
                tl.int64
            )
            cache_idx = intermediate_state_idx
        elif HAS_STATE_BATCH_INDICES:
            # 使用 state 的 batch 索引
            cache_idx = state_batch_idx
        else:
            cache_idx = pid_b  # 默认使用 pid_b

    # 逐时间步迭代（T 步）
    current_step_idx = 0
    for _ in range(T):
        if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
            # EAGLE tree attention：从父 token 的 buffer 中恢复对应的 SSM 状态
            if current_step_idx != 0 and cache_idx >= 0:
                parent_ptr = (
                    retrieve_parent_token_ptr
                    + pid_b * stride_retrieve_parent_token_batch
                    + current_step_idx * stride_retrieve_parent_token_T
                )
                # 加载当前 token 的父 token 步骤索引
                parent_step_idx = tl.load(parent_ptr).to(tl.int32)

                if parent_step_idx >= 0 and parent_step_idx < T:
                    # 从 buffer 恢复父 token 时刻的 SSM 状态（实现树状因果注意力）
                    step_offset = parent_step_idx * nheads * dim * dstate
                    cache_ptr = (
                        intermediate_states_buffer
                        + cache_idx * cache_steps * nheads * dim * dstate
                        + step_offset
                        + pid_h * dim * dstate
                        + offs_m[:, None] * dstate
                        + offs_n[None, :]
                    )
                    # 加载父 token 的状态作为当前 token 的初始状态
                    state = tl.load(cache_ptr, mask=mask, other=0.0).to(tl.float32)

        # 计算当前步的各输入/输出指针（相对于当前 T 步偏移）
        x_ptrs = x_ptr + offs_m * stride_x_dim
        dt_ptrs = dt_ptr + offs_m * stride_dt_dim
        B_ptrs = B_ptr + offs_n * stride_B_dstate
        C_ptrs = C_ptr + offs_n * stride_C_dstate
        if HAS_Z:
            z_ptrs = z_ptr + offs_m * stride_z_dim
        out_ptrs = out_ptr + offs_m * stride_out_dim

        # 加载当前步的输入 x（(dim,) 切片）
        x = tl.load(x_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if not TIE_HDIM:
            # 非 TIE_HDIM 模式：每个 dim 有独立的 ∆ 和 A
            dt = tl.load(dt_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            if DT_SOFTPLUS:
                dt = softplus(dt)
            # 加载 A 矩阵 (dim × dstate)
            A = tl.load(
                A_ptrs,
                mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate),
                other=0.0,
            ).to(tl.float32)
            # dA = exp(A * ∆)：状态转移衰减因子（(dim, dstate) 矩阵）
            dA = tl.exp(A * dt[:, None])
        else:
            # TIE_HDIM 模式：所有 dim 共享同一个标量 ∆ 和 A
            dt = tl.load(dt_ptr).to(tl.float32)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptr).to(tl.float32)
            if DT_SOFTPLUS:
                dt = softplus(dt)
            A = tl.load(A_ptr).to(tl.float32)
            dA = tl.exp(A * dt)  # scalar, not a matrix

        # 加载 B (dstate,) 和 C (dstate,) 向量
        B = tl.load(B_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
        C = tl.load(C_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
        if HAS_D:
            D = tl.load(D_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if HAS_Z:
            z = tl.load(z_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)

        # dB = B * ∆：输入投影（与 ∆ 耦合，shape 为 (dim, dstate) 或标量）
        dB = B[None, :] * dt[:, None] if not TIE_HDIM else B * dt
        # SSM 状态更新：h_{t+1} = dA * h_t + dB * x_t（离散化 SSM 递推公式）
        state = state * dA + dB * x[:, None]

        # 若启用中间状态缓存，将当前步的状态写入 buffer（供 EAGLE tree attention 回溯）
        if CACHE_INTERMEDIATE_STATES:
            if HAS_STATE_BATCH_INDICES:
                if state_batch_idx != pad_slot_id:
                    cache_ptr_base = (
                        intermediate_states_buffer
                        + cache_idx * cache_steps * nheads * dim * dstate
                        + current_step_idx * nheads * dim * dstate
                        + pid_h * dim * dstate
                    )
                    cache_ptrs = cache_ptr_base + (
                        offs_m[:, None] * dstate + offs_n[None, :]
                    )
                    # tl.store：将当前步的 SSM 状态写入中间状态 buffer
                    tl.store(
                        cache_ptrs, state.to(cache_ptrs.dtype.element_ty), mask=mask
                    )

        # 计算输出：y_t = C^T h_{t+1}（对 dstate 维度求内积）
        out = tl.sum(state * C[None, :], axis=1)
        if HAS_D:
            out += x * D  # 跳跃连接：y += D * x
        if HAS_Z:
            out *= z * tl.sigmoid(z)  # 门控：y *= SiLU(z) = z * sigmoid(z)
        # 写出当前步的输出
        tl.store(out_ptrs, out, mask=offs_m < dim)

        current_step_idx += 1

        # 推进各指针到下一个时间步
        x_ptr += stride_x_T
        dt_ptr += stride_dt_T
        B_ptr += stride_B_T
        C_ptr += stride_C_T
        out_ptr += stride_out_T
        if HAS_Z:
            z_ptr += stride_z_T

    # 若未禁用状态回写，将最终 SSM 状态写回 state 张量（用于下次 decode 步）
    if not DISABLE_STATE_UPDATE:
        tl.store(state_ptrs, state.to(state_ptrs.dtype.element_ty), mask=mask)


# Python 封装：对各维度进行 unsqueeze 标准化，然后调用 Triton kernel
def selective_state_update(
    state,           # SSM 状态，(batch, dim, dstate) 或 (batch, nheads, dim, dstate)
    x,               # 输入，(batch, dim) 或 (batch, nheads, dim) 或 (batch, T, nheads, dim)
    dt,              # 时间步 ∆
    A,               # 状态转移矩阵对角线 A
    B,               # SSM 输入矩阵 B
    C,               # SSM 输出矩阵 C
    D=None,          # 跳跃连接 D（可选）
    z=None,          # 门控分支 z（可选）
    dt_bias=None,    # ∆ 偏置（可选）
    dt_softplus=False,           # 是否对 ∆ 应用 softplus
    state_batch_indices=None,    # 连续批处理 batch 索引
    pad_slot_id=PAD_SLOT_ID,     # 填充槽位 ID
    out=None,                    # 预分配输出张量（原位更新）
    disable_state_update=False,  # True 时不回写 state
    intermediate_states_buffer=None,  # 中间状态缓存 buffer
    cache_steps=None,                 # buffer 缓存步数
    retrieve_parent_token=None,       # EAGLE 父 token 索引
    intermediate_state_indices=None,  # buffer 自定义索引
):
    """
    Argument:
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        x: (batch, dim) or (batch, nheads, dim) for single-token or (batch, T, nheads, dim) for multi-token
        dt: (batch, dim) or (batch, nheads, dim)
        A: (dim, dstate) or (nheads, dim, dstate)
        B: (batch, dstate) or (batch, ngroups, dstate) for single-token or (batch, T, ngroups, dstate) for multi-token
        C: (batch, dstate) or (batch, ngroups, dstate)
        D: (dim,) or (nheads, dim)
        z: (batch, dim) or (batch, nheads, dim)
        dt_bias: (dim,) or (nheads, dim)
        pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
        out: Preallocated ssm output tensor. Assume same shape as x.
             In-place updated.
        disable_state_update: If True, don't write back to state (for speculative verify)
        intermediate_states_buffer: Buffer to cache intermediate states
        cache_steps: Total number of steps in the buffer
        retrieve_parent_token: (batch, T) tensor of parent token indices for EAGLE tree attention
        intermediate_state_indices: (batch,) tensor of indices for intermediate_states_buffer operations.
            If provided, uses these indices instead of state_batch_indices for the buffer.
    """
    # 将低维输入 unsqueeze 到统一的 4D 形状：(batch, T, nheads, dim)
    if state.dim() == 3:
        state = state.unsqueeze(1)   # (batch, dim, dstate) -> (batch, 1, dim, dstate)
    if x.dim() == 2:
        x = x.unsqueeze(1)           # (batch, dim) -> (batch, 1, dim)
    if x.dim() == 3:
        x = x.unsqueeze(1)           # (batch, nheads, dim) -> (batch, 1, nheads, dim)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)    # (batch, dim) -> (batch, 1, dim)
    if dt.dim() == 3:
        dt = dt.unsqueeze(1)    # (batch, nheads, dim) -> (batch, 1, nheads, dim)
    if A.dim() == 2:
        A = A.unsqueeze(0)      # (dim, dstate) -> (1, dim, dstate)
    if B.dim() == 2:
        B = B.unsqueeze(1)      # (batch, dstate) -> (batch, 1, dstate)
    if B.dim() == 3:
        B = B.unsqueeze(1)      # (batch, ngroups, dstate) -> (batch, 1, ngroups, dstate)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if C.dim() == 3:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)      # (dim,) -> (1, dim)
    if z is not None:
        if z.dim() == 2:
            z = z.unsqueeze(1)
        if z.dim() == 3:
            z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    if out.dim() == 2:
        out = out.unsqueeze(1)
    if out.dim() == 3:
        out = out.unsqueeze(1)

    # 解析标准化后的张量形状
    _, nheads, dim, dstate = state.shape
    batch, T, _, _ = x.shape

    # 验证各张量形状一致性
    assert x.shape == (batch, T, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[2]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, T, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
    if state_batch_indices is not None:
        assert state_batch_indices.shape == (batch,)
    assert out.shape == x.shape

    # 启动 grid：axis0 = dim 分块，axis1 = batch，axis2 = nheads
    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE_M"]), batch, nheads)
    z_strides = (
        (z.stride(0), z.stride(1), z.stride(2), z.stride(3))
        if z is not None
        else (0, 0, 0, 0)
    )
    # We don't want autotune since it will overwrite the state
    # We instead tune by hand.
    # 手动调优 BLOCK_SIZE_M 和 num_warps（根据 dstate 大小分级选择）
    BLOCK_SIZE_M, num_warps = (
        (32, 4)
        if dstate <= 16
        else (
            (16, 4)
            if dstate <= 32
            else ((8, 4) if dstate <= 64 else ((4, 4) if dstate <= 128 else ((4, 8))))
        )
    )
    # TIE_HDIM：若 A/dt/dt_bias 的 dim 步长为 0，则所有 dim 共享同一值（绑定模式）
    tie_hdim = (
        A.stride(-1) == 0
        and A.stride(-2) == 0
        and dt.stride(-1) == 0
        and dt_bias.stride(-1) == 0
    )

    retrieve_parent_token_strides = (
        (retrieve_parent_token.stride(0), retrieve_parent_token.stride(1))
        if retrieve_parent_token is not None
        else (0, 0)
    )

    with torch.get_device_module(x.device).device(x.device.index):
        _selective_scan_update_kernel[grid](
            state,
            x,
            dt,
            dt_bias,
            A,
            B,
            C,
            D,
            z,
            out,
            state_batch_indices,
            pad_slot_id,
            intermediate_states_buffer,
            cache_steps if cache_steps is not None else 0,  # None 转 0
            retrieve_parent_token,
            intermediate_state_indices,
            batch,
            T,
            nheads,
            dim,
            dstate,
            nheads // ngroups,   # nheads_ngroups_ratio
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            dt.stride(3),
            *(dt_bias.stride(0), dt_bias.stride(1)) if dt_bias is not None else 0,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            B.stride(3),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            C.stride(3),
            *(D.stride(0), D.stride(1)) if D is not None else 0,
            z_strides[0],
            z_strides[1],
            z_strides[2],
            z_strides[3],
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            retrieve_parent_token_strides[0],
            retrieve_parent_token_strides[1],
            dt_softplus,
            tie_hdim,
            BLOCK_SIZE_M,
            DISABLE_STATE_UPDATE=disable_state_update,
            num_warps=num_warps,
        )
