# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import triton
import triton.language as tl


# Triton JIT 核函数：将 input_ids 原地左移（丢弃头部，尾部写入新 token）
# 用于多层 EAGLE 中在同一缓冲区内更新当前投机步的输入 token 序列
@triton.jit
def rotate_input_ids_kernel(
    input_ids_ptr,         # 输入 token ID 缓冲区（原地修改）
    extend_start_loc_ptr,  # 每个请求在展平序列中的起始位置
    extend_seq_lens_ptr,   # 每个请求的扩展序列长度
    topk_index_ptr,        # 每个请求当前步选中的 top-k token ID
    select_index_ptr,      # 可选：指定写入新 token 的绝对位置（用于非连续布局）
    BLOCK_SIZE: tl.constexpr,
):
    # 每个 program 处理一个请求
    pid = tl.program_id(0)

    start_loc = tl.load(extend_start_loc_ptr + pid)
    seq_len = tl.load(extend_seq_lens_ptr + pid)
    # 取出本步要写入的新 token
    new_token = tl.load(topk_index_ptr + pid)

    # 需要左移的元素数（把 [1..seq_len-1] 移到 [0..seq_len-2]）
    num_elements_to_shift = seq_len - 1

    # 分块左移：将 input_ids[start+1:start+seq_len] 复制到 input_ids[start:start+seq_len-1]
    for off in range(0, num_elements_to_shift, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements_to_shift

        read_ptr = input_ids_ptr + start_loc + offsets + 1
        val = tl.load(read_ptr, mask=mask)
        tl.debug_barrier()  # 确保读写不产生竞态

        write_ptr = input_ids_ptr + start_loc + offsets
        tl.store(write_ptr, val, mask=mask)
        tl.debug_barrier()

    # 将新 token 写入序列末尾（或通过 select_index 指定的绝对位置）
    if seq_len > 0:
        if select_index_ptr is not None:
            last_pos_ptr = input_ids_ptr + tl.load(select_index_ptr + pid)
        else:
            last_pos_ptr = input_ids_ptr + start_loc + seq_len - 1
        tl.store(last_pos_ptr, new_token)


def rotate_input_ids_triton(
    input_ids, extend_start_loc, extend_seq_lens, topk_index, select_index=None
):
    # 计算 batch size 并确定 BLOCK_SIZE（有 select_index 时序列可能更长，用大块）
    batch_size = extend_seq_lens.shape[0]
    BLOCK_SIZE = 4096 if select_index is not None else 8
    grid = (batch_size,)

    # 启动 Triton 核函数，每个请求一个 CUDA block
    rotate_input_ids_kernel[grid](
        input_ids,
        extend_start_loc,
        extend_seq_lens,
        topk_index,
        select_index,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return input_ids


# Triton JIT 核函数：将上一步的草稿状态迁移到下一步的缓冲区
# 同时从隐状态池（hidden_states_pool）中加载历史隐状态，构建新的扩展输入
@triton.jit
def assign_new_state_kernel(
    # Source pointers
    # 上一步的输入 token、位置、隐状态、KV 缓存位置及序列长度/起始位置
    old_input_ids_ptr,
    old_positions_ptr,
    old_hidden_states_ptr,
    old_out_cache_loc_ptr,
    old_extend_seq_lens_ptr,
    old_extend_start_loc_ptr,
    # Destination pointers
    # 新的输入 token、位置、隐状态、KV 缓存位置及序列长度/起始位置
    input_ids_ptr,
    positions_ptr,
    hidden_states_ptr,
    out_cache_loc_ptr,
    extend_seq_lens_ptr,
    extend_start_loc_ptr,
    # Auxiliary data pointers
    # 下一步的 token ID、当前序列长度、填充长度、请求池索引
    # req_to_token: 请求→token 映射表；req_to_hidden_states_pool: 请求→隐状态池
    next_token_ids_ptr,
    seq_lens_ptr,
    padding_lens_ptr,
    req_pool_indices_ptr,
    req_to_token_ptr,
    req_to_hidden_states_pool_ptr,
    # Scalars and Strides
    # step: 当前投机步序号；各张量的 stride 用于计算指针偏移
    step,
    stride_hidden_seq,
    stride_hidden_dim,  # hidden_states strides
    stride_pool_req,
    stride_pool_step,
    stride_pool_dim,  # pool strides
    stride_req_token_0,
    stride_req_token_1,  # req_to_token strides
    # Meta-parameters
    HIDDEN_DIM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_HID: tl.constexpr,
):
    # 每个 program 处理一个请求
    pid = tl.program_id(0)

    seq_len: tl.tensor = tl.load(seq_lens_ptr + pid)
    old_extend_len = tl.load(old_extend_seq_lens_ptr + pid)
    old_start = tl.load(old_extend_start_loc_ptr + pid)
    # 新的 extend 长度比上一步多 1（新增一个 token）
    new_extend_len = old_extend_len + 1
    # 新起始位置：旧起始 + pid（各请求在展平缓冲区中间隔 1 存放）
    new_start = old_start + pid

    # 写入新的序列长度和起始位置
    tl.store(extend_seq_lens_ptr + pid, new_extend_len)
    tl.store(extend_start_loc_ptr + pid, new_start)

    offs_seq = tl.arange(0, BLOCK_SEQ)
    mask_seq = offs_seq < old_extend_len

    # 将旧的 input_ids 复制到新缓冲区（偏移 0 处，为新 token 腾出位置 0）
    old_ids = tl.load(old_input_ids_ptr + old_start + offs_seq, mask=mask_seq)
    tl.store(input_ids_ptr + new_start + offs_seq, old_ids, mask=mask_seq)
    # 在去除 padding 的位置写入下一个 token ID
    padding_len = tl.load(padding_lens_ptr + pid)
    tl.store(
        input_ids_ptr + new_start + old_extend_len - padding_len,
        tl.load(next_token_ids_ptr + pid),
    )

    # 将旧的位置编码复制到新缓冲区（整体后移 1，位置 0 留给新 token 的前驱）
    old_pos = tl.load(old_positions_ptr + old_start + offs_seq, mask=mask_seq)
    tl.store(positions_ptr + new_start + 1 + offs_seq, old_pos, mask=mask_seq)
    # 新位置 0 = 旧起始位置 - 1（确保不小于 0）
    tl.store(
        positions_ptr + new_start, max(tl.load(old_positions_ptr + old_start) - 1, 0)
    )

    # 将旧的 KV 缓存位置复制到新缓冲区（整体后移 1）
    old_cache = tl.load(old_out_cache_loc_ptr + old_start + offs_seq, mask=mask_seq)
    tl.store(out_cache_loc_ptr + new_start + 1 + offs_seq, old_cache, mask=mask_seq)

    # 从 req_to_token 表中读取历史 KV 缓存位置，填充位置 0
    req_idx = tl.load(req_pool_indices_ptr + pid)
    token_idx_col = seq_len - old_extend_len - 1
    if token_idx_col >= 0:
        req_token_ptr_loc = (
            req_to_token_ptr
            + (req_idx * stride_req_token_0)
            + (token_idx_col * stride_req_token_1)
        )
        last_cache_loc = tl.load(req_token_ptr_loc)
        tl.store(out_cache_loc_ptr + new_start, last_cache_loc)

    # 计算隐状态池中的偏移（按请求索引和当前步序号寻址）
    pool_vec_offset_base = ((req_idx + 1) * stride_pool_req) + (
        -(step + 1) * stride_pool_step
    )

    # 分块复制旧的隐状态（shift 一位），并从池中加载位置 0 的隐状态
    for off_h in range(0, HIDDEN_DIM, BLOCK_HID):
        offs_h = off_h + tl.arange(0, BLOCK_HID)
        mask_h = offs_h < HIDDEN_DIM

        # 将旧隐状态复制到新缓冲区的 [new_start+1 : new_start+1+old_extend_len]
        for i in range(BLOCK_SEQ):
            if i < old_extend_len:
                old_h_ptr = (
                    old_hidden_states_ptr
                    + (old_start + i) * stride_hidden_seq
                    + (offs_h * stride_hidden_dim)
                )
                new_h_ptr = (
                    hidden_states_ptr
                    + (new_start + 1 + i) * stride_hidden_seq
                    + (offs_h * stride_hidden_dim)
                )

                chunk_old = tl.load(old_h_ptr, mask=mask_h)
                tl.store(new_h_ptr, chunk_old, mask=mask_h)

        # 从隐状态池中加载本步的起始隐状态，写入新缓冲区位置 0
        pool_ptrs = (
            req_to_hidden_states_pool_ptr
            + pool_vec_offset_base
            + (offs_h * stride_pool_dim)
        )
        pool_val = tl.load(pool_ptrs, mask=mask_h)

        new_h_start_ptrs = (
            hidden_states_ptr
            + (new_start * stride_hidden_seq)
            + (offs_h * stride_hidden_dim)
        )
        tl.store(new_h_start_ptrs, pool_val, mask=mask_h)


def assign_new_state_triton(
    next_token_ids: torch.Tensor,
    old_input_ids: torch.Tensor,
    old_positions: torch.Tensor,
    old_hidden_states: torch.Tensor,
    old_out_cache_loc: torch.Tensor,
    old_extend_seq_lens: torch.Tensor,
    old_extend_start_loc: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    out_cache_loc: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    padding_lens: torch.Tensor,
    num_seqs: int,
    step: int,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    req_to_hidden_states_pool: torch.Tensor,
):
    """
    Wrapper function to calculate offsets and launch the Triton kernel.
    """
    # 读取隐状态维度
    hidden_dim = hidden_states.shape[1]

    # BLOCK_SEQ: 序列方向分块大小（对应 EAGLE 多步的最大扩展长度）
    BLOCK_SEQ = 8
    # BLOCK_HID: 隐状态维度方向分块大小（影响寄存器占用）
    BLOCK_HID = 64

    # 每个请求对应一个 CUDA block
    grid = (num_seqs,)

    assign_new_state_kernel[grid](
        # Pointers
        old_input_ids,
        old_positions,
        old_hidden_states,
        old_out_cache_loc,
        old_extend_seq_lens,
        old_extend_start_loc,
        input_ids,
        positions,
        hidden_states,
        out_cache_loc,
        extend_seq_lens,
        extend_start_loc,
        next_token_ids,
        seq_lens,
        padding_lens,
        req_pool_indices,
        req_to_token,
        req_to_hidden_states_pool,
        # Constants/Strides
        step,
        old_hidden_states.stride(0),
        old_hidden_states.stride(1),
        req_to_hidden_states_pool.stride(0),
        req_to_hidden_states_pool.stride(1),
        req_to_hidden_states_pool.stride(2),
        req_to_token.stride(0),
        req_to_token.stride(1),
        # Meta
        HIDDEN_DIM=hidden_dim,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_HID=BLOCK_HID,
    )


# Triton JIT 核函数：将当前步末尾若干个隐状态写入请求级别的隐状态池
# 隐状态池用于多层 EAGLE 中跨步传递上下文信息
@triton.jit
def assign_hidden_states_pool_kernel(
    hidden_states_ptr,              # 当前步的隐状态张量
    req_pool_indices_ptr,           # 每个请求的池索引
    req_to_hidden_states_pool_ptr,  # 隐状态池（shape: [num_reqs+1, pool_size, hidden_dim]）
    extend_seq_lens_ptr,            # 每个请求的扩展长度
    extend_start_loc_ptr,           # 每个请求的起始位置
    stride_hidden_seq,              # hidden_states 在序列维度的 stride
    stride_hidden_dim,              # hidden_states 在维度方向的 stride
    stride_pool_req,                # 池在请求维度的 stride
    stride_pool_step,               # 池在步数维度的 stride
    stride_pool_dim,                # 池在维度方向的 stride
    HIDDEN_DIM: tl.constexpr,
    pool_size: tl.constexpr,        # 池大小（保存最近 pool_size 步的隐状态）
    BLOCK_HID: tl.constexpr,
):
    pid = tl.program_id(0)

    extend_len = tl.load(extend_seq_lens_ptr + pid)
    start_loc = tl.load(extend_start_loc_ptr + pid)
    end_loc = start_loc + extend_len

    req_idx = tl.load(req_pool_indices_ptr + pid)
    # 计算该请求在池中的基础偏移
    pool_vec_offset_base = req_idx * stride_pool_req

    # 将末尾 pool_size 个隐状态依次写入池
    for i in range(pool_size):
        for off_h in range(0, HIDDEN_DIM, BLOCK_HID):
            offs_h = off_h + tl.arange(0, BLOCK_HID)
            mask_h = offs_h < HIDDEN_DIM

            # 读取第 (end_loc - pool_size + i) 位置的隐状态
            hid_ptr = (
                hidden_states_ptr
                + (end_loc - pool_size + i) * stride_hidden_seq
                + offs_h * stride_hidden_dim
            )
            hid_val = tl.load(hid_ptr, mask=mask_h)

            # 写入池的对应位置
            pool_ptr = (
                req_to_hidden_states_pool_ptr
                + pool_vec_offset_base
                + i * stride_pool_step
                + offs_h * stride_pool_dim
            )
            tl.store(pool_ptr, hid_val, mask=mask_h)


def assign_hidden_states_pool_triton(
    hidden_states: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_hidden_states_pool: torch.Tensor,
    pool_size: int,
    num_seqs: int,
    extend_seq_lens: torch.Tensor,
    extend_start_loc: torch.Tensor,
):
    # 每个请求一个 CUDA block
    grid = (num_seqs,)
    assign_hidden_states_pool_kernel[grid](
        hidden_states,
        req_pool_indices,
        req_to_hidden_states_pool,
        extend_seq_lens,
        extend_start_loc,
        hidden_states.stride(0),
        hidden_states.stride(1),
        req_to_hidden_states_pool.stride(0),
        req_to_hidden_states_pool.stride(1),
        req_to_hidden_states_pool.stride(2),
        HIDDEN_DIM=hidden_states.shape[1],
        pool_size=pool_size,
        BLOCK_HID=64,
    )


def assign_hidden_states_pool_torch(
    hidden_states: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_hidden_states_pool: torch.Tensor,
    pool_size: int,
    num_seqs: int,
    extend_seq_lens: torch.Tensor,
    extend_start_loc: torch.Tensor,
):
    # PyTorch 参考实现（调试用），与 Triton 版本功能等价
    for req in range(num_seqs):
        pool_idx = req_pool_indices[req]
        extend_len = extend_seq_lens[req]
        start_loc = extend_start_loc[req]
        end_loc = start_loc + extend_len
        # 将末尾 pool_size 个隐状态拷贝到对应请求的池中
        req_to_hidden_states_pool[pool_idx, :pool_size, :].copy_(
            hidden_states[end_loc - pool_size : end_loc, :]
        )
