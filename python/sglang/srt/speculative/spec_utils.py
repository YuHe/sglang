from __future__ import annotations

# 推测解码通用工具模块：Triton 内核、树遍历、token 映射加载、调试检查等
import logging
import os
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Optional

import torch
import triton
import triton.language as tl
# snapshot_download 用于从 HuggingFace Hub 下载 hot token 映射文件
from huggingface_hub import snapshot_download

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.distributed.parallel_state import (
    GroupCoordinator,
    patch_tensor_parallel_group,
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.common import get_last_loc
from sglang.srt.server_args import ServerArgs, get_global_server_args
# 平台检测工具函数和 next_power_of_2（用于 Triton 内核 constexpr 上界）
from sglang.srt.utils import is_cuda, is_hip, is_musa, is_npu, next_power_of_2

# 模块级平台标志，避免重复调用检测函数
_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_musa = is_musa()

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_info import EagleVerifyInput


# 按平台选择 fast_topk 实现：CUDA/HIP 用 sgl_kernel 的高性能版本，其余用纯 Python
if _is_cuda:
    from sgl_kernel import fast_topk
elif _is_hip:
    from sgl_kernel import fast_topk
else:
    from sglang.srt.utils.common import fast_topk


logger = logging.getLogger(__name__)


# Simulate acceptance length for benchmarking purposes
# 模拟接受长度（< 0 时关闭），用于基准测试时绕过真实验证
SIMULATE_ACC_LEN = envs.SGLANG_SIMULATE_ACC_LEN.get()  # turn off if < 0
# 模拟方法：'multinomial'（正态分布取整）或 'match-expected'（概率插值）
SIMULATE_ACC_METHOD = envs.SGLANG_SIMULATE_ACC_METHOD.get()

# 树遍历时间告警阈值（秒），超过时打印 warning
TREE_TRAVERSE_TIME_THRESHOLD = 1  # TODO: set this properly
# 树形推测采样内核目前仅支持 CUDA 和 MUSA
TREE_SPEC_KERNEL_AVAILABLE = (
    _is_cuda or _is_musa
)  # This kernel is only available for CUDA and MUSA now


def spec_need_hidden_states(server_args: Optional[ServerArgs] = None) -> bool:
    # multi-layer EAGLE 使用独立 draft 模型，不需要捕获目标模型隐藏状态
    if server_args is None:
        server_args = get_global_server_args()

    # TODO(lsyin): also skip when 1) step = 1 or 2) standalone draft model
    return not server_args.enable_multi_layer_eagle


@triton.jit
def create_extend_after_decode_spec_info(
    verified_id,       # 当前批次已接受的 token id，展平存储
    seq_lens,          # 每条请求当前的序列长度
    accept_lens,       # 每条请求本轮接受的 token 数（含 bonus token）
    positions,         # 输出：每个被接受 token 在序列中的绝对位置
    new_verified_id,   # 输出：每条请求最后一个被接受的 token id（下一步的输入）
    bs_upper: tl.constexpr,  # bs 向上取 2 的幂，用于 constexpr mask
):
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, bs_upper)
    seq_length = tl.load(seq_lens + pid)
    # `accept_lens` includes the bonus token; load this req's value.
    accept_len = tl.load(accept_lens + pid)

    # 计算前缀请求的接受长度累积和，确定本请求输出在 positions 中的写入起点
    accept_len_cumsum = tl.sum(
        tl.load(accept_lens + offsets, mask=offsets < pid, other=0)
    )
    positions_ptr = positions + accept_len_cumsum
    mask = offsets < accept_len
    # 将 accept_len 个 token 的位置写入：从 seq_length - accept_len 开始递增
    tl.store(positions_ptr + offsets, seq_length - accept_len + offsets, mask)

    # 取最后一个被接受 token（即 bonus token）作为 new_verified_id
    accept_len_cumsum += accept_len - 1
    verified_id_data = tl.load(verified_id + accept_len_cumsum)
    tl.store(new_verified_id + pid, verified_id_data)


@triton.jit
def assign_req_to_token_pool(
    req_pool_indices,   # [bs]，每条请求在 req_to_token 池中的行索引
    req_to_token,       # [num_reqs, pool_len]，请求→KV slot 映射表
    start_offset,       # [bs]，本轮每条请求需要写入的起始 KV 位置
    end_offset,         # [bs]，本轮每条请求需要写入的结束 KV 位置
    out_cache_loc,      # 输入：已分配的 KV slot 列表（顺序排列）
    pool_len: tl.constexpr,   # req_to_token 的列数（每请求最大 KV 容量）
    bs_upper: tl.constexpr,   # bs 向上取 2 的幂
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    # 计算前缀请求的总分配量，确定本请求在 out_cache_loc 中的读取起点
    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0)
    out_offset = tl.sum(end - start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset

    save_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    load_offset = tl.arange(0, BLOCK_SIZE)

    # 分块循环写入，将 out_cache_loc 中的 slot 复制到 req_to_token 对应行
    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = save_offset < kv_end
        data = tl.load(out_cache_ptr + load_offset, mask=mask)
        tl.store(token_pool + save_offset, data, mask=mask)
        save_offset += BLOCK_SIZE
        load_offset += BLOCK_SIZE


def assign_req_to_token_pool_func(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
    batch_size: int,
):
    # Grid=(batch_size,)，每个 block 处理一条请求
    assign_req_to_token_pool[(batch_size,)](
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        out_cache_loc,
        req_to_token.shape[1],          # pool_len constexpr
        next_power_of_2(batch_size),    # bs_upper constexpr，用于 mask
    )


@triton.jit
def assign_draft_cache_locs(
    req_pool_indices,           # [bs]，请求行索引
    req_to_token,               # [num_reqs, pool_len]，请求→KV slot 映射表
    seq_lens,                   # [bs]，当前 prefix 长度（不含草稿 token）
    extend_lens,                # [bs]，paged 模式下每请求需写入的总 slot 数
    num_new_pages_per_topk,     # [bs]，paged 模式下每个 topk 需要的新页数
    out_cache_loc,              # 已分配的 KV slot 列表（同时作为 Part3 的输出）
    source_cache_loc,           # 输出：需复制的源 slot（最后一页重复部分）
    target_cache_loc,           # 输出：目标 slot（各 topk 分支对应位置）
    last_page_lens_cumsum,      # [bs]，各请求最后一页长度的前缀和
    duplicate_cache_len: tl.constexpr,  # 需要复制的最后页 token 总数
    pool_len: tl.constexpr,
    topk: tl.constexpr,
    speculative_num_steps: tl.constexpr,
    page_size: tl.constexpr,
    bs_upper: tl.constexpr,
    iter_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 128
    pid = tl.program_id(axis=0)

    # Part 1 输出目标：page_size=1 或 topk=1 时 out_cache_ptr 按 pid * topk * steps 对齐
    if page_size == 1 or topk == 1:
        copy_len = topk * speculative_num_steps
        out_cache_ptr = out_cache_loc + pid * topk * speculative_num_steps
    else:
        # paged 模式：copy_len 来自 extend_lens，起点由前缀累积和决定
        bs_offset = tl.arange(0, bs_upper)
        copy_len = tl.load(extend_lens + pid)
        cum_copy_len = tl.sum(tl.load(extend_lens + bs_offset, mask=bs_offset < pid))
        out_cache_ptr = out_cache_loc + cum_copy_len

    # Part 1: Copy from out_cache_loc to req_to_token
    # 将 out_cache_loc 中的 slot 索引写入 req_to_token，从 seq_lens[pid] 位置开始
    kv_start = tl.load(seq_lens + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len
    num_loop = tl.cdiv(copy_len, BLOCK_SIZE)
    for i in range(num_loop):
        copy_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = copy_offset < copy_len
        data = tl.load(out_cache_ptr + copy_offset, mask=mask)
        tl.store(token_pool + kv_start + copy_offset, data, mask=mask)
    # XXX (MUSA): Triton issue: chained boolean operators (A or B or C) are not supported.
    if (page_size != 1 and topk != 1) and duplicate_cache_len > 0:
        # Part 2: Copy indices into source_cache_loc and target_cache_loc
        # Expected output: src:[8,9,10,8,9,10...] tgt:[16,17,18,24,25,26...]
        # 在 paged topk 模式下，各 topk 分支共享 prefix 最后一页的 slot，
        # 需要将这些 slot 复制到各分支独立页，source/target_cache_loc 记录拷贝信息
        prefix_len = tl.load(seq_lens + pid)
        last_page_len = prefix_len % page_size
        offsets = tl.arange(0, page_size)
        mask = offsets < last_page_len
        num_new_pages_per_topk_ = tl.load(num_new_pages_per_topk + pid)
        prefix_base = token_pool + prefix_len - last_page_len
        src_indices = tl.load(prefix_base + offsets, mask=mask)
        last_page_lens_cumsum_ = tl.load(last_page_lens_cumsum + pid)
        # Skip the first one since no copy is needed
        for topk_id in range(1, topk):
            tl.store(
                source_cache_loc
                + (topk - 1) * (last_page_lens_cumsum_ - last_page_len)
                + (topk_id - 1) * last_page_len
                + offsets,
                src_indices,
                mask=mask,
            )
            tgt_indices = tl.load(
                prefix_base + topk_id * num_new_pages_per_topk_ * page_size + offsets,
                mask=mask,
            )
            tl.store(
                target_cache_loc
                + (topk - 1) * (last_page_lens_cumsum_ - last_page_len)
                + (topk_id - 1) * last_page_len
                + offsets,
                tgt_indices,
                mask=mask,
            )
        # Part 3: Copy and remove the used indices for duplication
        # speculative_num_steps=5, page_size=4, num_new_pages_per_topk_=2, last_page_len=1
        #  - xxxxx .. | - xxxxx .. |
        #   topk=0        topk=1
        #  "-" means prefix tokens
        #  "x" means speculative draft tokens
        #  "." means padded tokens
        # we only want to copy the "x" part.
        # 将每个 topk 分支中实际的草稿 slot（排除 last_page 的 padding）
        # 压缩写回 out_cache_loc，供后续 EAGLE 前向使用
        iter_offset = tl.arange(0, iter_upper)
        for topk_id in range(topk):
            mask_upper = iter_offset < (speculative_num_steps + last_page_len)
            mask_lower = iter_offset >= last_page_len
            combined_mask = mask_upper & mask_lower
            indices = tl.load(
                prefix_base
                + topk_id * num_new_pages_per_topk_ * page_size
                + iter_offset,
                mask=combined_mask,
                other=0,
            )
            # Shift from previous batches
            ptr_offset = pid * speculative_num_steps * topk
            # Subtract last_page_len to fill the gap of duplicated last page tokens.
            # For example, token pool is (1, 2, 3, 4 ,5) and last page is 1,
            # we write 2, 3, 4 to the front of out_cache_loc.
            tl.store(
                out_cache_loc
                + ptr_offset
                + topk_id * speculative_num_steps
                - last_page_len
                + iter_offset,
                indices,
                mask=combined_mask,
            )


@triton.jit
def generate_draft_decode_kv_indices(
    req_pool_indices,           # [bs]
    req_to_token,               # [num_reqs, pool_len]
    paged_kernel_lens,          # [bs]，每条请求的 prefix 长度
    kv_indices,                 # 输出：FlashInfer 所需的 KV slot 索引（含草稿 token）
    kv_indptr,                  # 输出：每条请求 KV 块的指针（CSR 格式）
    positions,                  # [bs * topk]，各 (req, topk) 组合的 KV 个数
    pool_len: tl.constexpr,
    kv_indices_stride: tl.constexpr,   # 相邻推测步之间 kv_indices 的行偏移
    kv_indptr_stride: tl.constexpr,    # 相邻推测步之间 kv_indptr 的行偏移
    bs_upper: tl.constexpr,
    iter_upper: tl.constexpr,          # num_steps 向上取 2 的幂
    num_tokens_upper: tl.constexpr,    # bs * topk 向上取 2 的幂
    page_size: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 128
    # 三维 Grid：(推测步, batch_id, topk_id)
    iters = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    topk_id = tl.program_id(axis=2)

    num_steps = tl.num_programs(axis=0)
    num_seqs = tl.num_programs(axis=1)
    topk = tl.num_programs(axis=2)

    # 按推测步偏移 kv_indices/kv_indptr 基指针
    kv_indices += kv_indices_stride * iters
    kv_indptr += kv_indptr_stride * iters
    iters += 1  # 推测步变为 1-based，用于计算 KV 偏移

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(paged_kernel_lens + load_offset, mask=load_offset < bid, other=0)
    seq_len = tl.load(paged_kernel_lens + bid)
    # 前缀请求的 KV 总量（用于计算本请求在 kv_indices 中的写入起点）
    cum_seq_len = tl.sum(seq_lens)

    # Update kv_indices
    # kv_offset：在 kv_indices 中的二维偏移，包含前序请求的 prefix KV、各步的 topk 偏移
    kv_offset = cum_seq_len * topk + bid * iters * topk + topk_id * (seq_len + iters)
    kv_ptr = kv_indices + kv_offset
    token_pool_ptr = req_to_token + tl.load(req_pool_indices + bid) * pool_len

    # 先写入 prefix 部分的 KV slot（从 req_to_token 读取）
    kv_offset = tl.arange(0, BLOCK_SIZE)
    num_loop = tl.cdiv(seq_len, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = kv_offset < seq_len
        data = tl.load(token_pool_ptr + kv_offset, mask=mask)
        tl.store(kv_ptr + kv_offset, data, mask=mask)
        kv_offset += BLOCK_SIZE

    # 再追加草稿 token 的 KV slot（来自草稿推理阶段分配的位置）
    extend_offset = tl.arange(0, iter_upper)
    if page_size == 1 or topk == 1:
        # page_size=1 路径：草稿 slot 连续存储，按 topk_id * num_steps + step_id 索引
        extend_data = tl.load(
            token_pool_ptr + seq_len + topk_id * num_steps + tl.arange(0, iter_upper),
            mask=extend_offset < iters,
        )
    else:
        # paged 路径：需要跳过 last_page padding，从正确偏移读取草稿 slot
        prefix_len = seq_len
        last_page_len = prefix_len % page_size
        num_new_pages_per_topk = (
            last_page_len + num_steps + page_size - 1
        ) // page_size
        prefix_base = seq_len // page_size * page_size
        start = (
            prefix_base + topk_id * num_new_pages_per_topk * page_size + last_page_len
        )
        extend_data = tl.load(
            token_pool_ptr + start + extend_offset,
            mask=extend_offset < iters,
        )

    tl.store(kv_ptr + seq_len + extend_offset, extend_data, mask=extend_offset < iters)

    # Update kv_indptr
    # 计算 (bid, topk_id) 对应的全局 token 索引 zid
    bs_offset = tl.arange(0, num_tokens_upper)

    zid = bid * topk + topk_id
    if zid == 0:
        # zid=0 时写入最后一个元素（kv_indptr 为 CSR，长度 = bs*topk + 1）
        zid = num_seqs * topk
    positions = tl.load(positions + bs_offset, mask=bs_offset < zid, other=0)
    base = tl.sum(positions)
    # kv_indptr[zid] = 前序所有请求的 KV 总数 + zid * iters（每请求 iters 个草稿 token）
    tl.store(kv_indptr + zid, base + zid * iters)


@triton.jit
def align_evict_mask_to_page_size(
    seq_lens,            # [bs]，当前 prefix 长度
    evict_mask,          # [bs, num_draft_tokens]，True 表示该 slot 可释放
    page_size: tl.constexpr,
    num_draft_tokens: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 对 evict_mask 进行页对齐：不能在一页中间释放，需要整页释放
    t_range = tl.arange(0, BLOCK_SIZE)

    bid = tl.program_id(axis=0)
    seq_len = tl.load(seq_lens + bid)
    io_mask = t_range < num_draft_tokens
    mask_row = tl.load(
        evict_mask + bid * num_draft_tokens + t_range, mask=io_mask, other=0
    )

    num_trues = tl.sum(mask_row)
    num_false = num_draft_tokens - num_trues

    # 计算对齐后的起始位置：确保释放范围从页边界开始
    start = (seq_len + num_false - 1) // page_size * page_size - seq_len
    # 将 [start, start+page_size) 范围内的 mask 清零（对齐到页边界，不分裂页）
    for i in range(max(start, 0), min(start + page_size, num_draft_tokens)):
        tl.store(evict_mask + bid * num_draft_tokens + i, False)


@triton.jit
def get_target_cache_loc(
    tgt_cache_loc,          # 输出：最终保留的 KV slot（已接受 token + bonus）
    to_free_slots,          # 输出：需要释放的 KV slot
    num_accepted_drafts,    # [bs]，各请求接受的草稿 token 数（不含 bonus）
    to_free_num_slots,      # [bs]，各请求需释放的 slot 数
    out_cache_loc,          # 输入：分配的 KV slot 列表（每请求 num_verify_tokens 个）
    num_verify_tokens: tl.constexpr,       # 每请求分配的 slot 数（= draft_token_num + 1）
    num_verify_tokens_upper: tl.constexpr, # 向上取 2 的幂
    bs_upper: tl.constexpr,
):
    bid = tl.program_id(axis=0)
    offset = tl.arange(0, num_verify_tokens_upper)
    bs_offset = tl.arange(0, bs_upper)

    # write the first part to tgt_cache_loc
    # tgt_cache_loc 的写入起点 = 前缀请求的接受总数 + bid（每请求含 1 个 bonus slot）
    accept_len_all = tl.load(num_accepted_drafts + bs_offset, mask=bs_offset < bid)
    tgt_cache_loc_start = tl.sum(accept_len_all) + bid
    copy_len = tl.load(num_accepted_drafts + bid) + 1
    out_cache_loc_row = tl.load(
        out_cache_loc + bid * num_verify_tokens + offset, mask=offset < copy_len
    )
    tl.store(
        tgt_cache_loc + tgt_cache_loc_start + offset,
        out_cache_loc_row,
        mask=offset < copy_len,
    )

    # write the second part to to_free_num_pages
    # to_free_slots 的写入起点 = 前缀请求的待释放 slot 累积总数
    to_free_num_slots_all = tl.load(to_free_num_slots + bs_offset, mask=bs_offset < bid)
    to_free_num_slots_cur = tl.load(to_free_num_slots + bid)
    # 从 out_cache_loc 末尾读取待释放 slot（位于已接受 slot 之后）
    out_cache_loc_start = num_verify_tokens - to_free_num_slots_cur
    to_free_slots_start = tl.sum(to_free_num_slots_all)

    copy_len = to_free_num_slots_cur
    out_cache_loc_row = tl.load(
        out_cache_loc + bid * num_verify_tokens + out_cache_loc_start + offset,
        mask=offset < copy_len,
    )
    tl.store(
        to_free_slots + to_free_slots_start + offset,
        out_cache_loc_row,
        mask=offset < copy_len,
    )


@torch.compile(dynamic=True, disable=_is_npu)
def get_src_tgt_cache_loc(
    seq_lens: torch.Tensor,            # [bs]，prefix 长度
    out_cache_loc: torch.Tensor,       # [bs, draft_token_num+1]，已分配 KV slot
    accept_index: torch.Tensor,        # [total_accepted]，接受 token 的 slot 索引
    num_accepted_drafts: torch.Tensor, # [bs]，各请求接受的草稿 token 数
    draft_token_num: int,              # 草稿 token 总数（= speculative_num_steps * topk）
    page_size: int,
):
    # src_cache_loc：被接受 token 当前所在的 KV slot（按 accept_index 索引）
    src_cache_loc = out_cache_loc[accept_index]
    tgt_cache_loc = torch.empty_like(src_cache_loc)
    extended_len = seq_lens + draft_token_num
    # keep_len：页对齐后实际需要保留的 slot 数（向上对齐到页边界，不超过 extended_len）
    keep_len = torch.minimum(
        (seq_lens + num_accepted_drafts + 1 + page_size - 1) // page_size * page_size,
        extended_len,
    )
    # to_free_num_slots：已分配但实际不需要的 slot 数（超出对齐保留量的部分）
    to_free_num_slots = extended_len - keep_len
    return src_cache_loc, tgt_cache_loc, to_free_num_slots


@triton.jit
def filter_finished_cache_loc_kernel(
    out_cache_loc,                  # 输出：过滤后已完成请求的 tgt_cache_loc 写入目标
    tgt_cache_loc,                  # 输入：全部请求的 tgt_cache_loc（含已完成的）
    num_accepted_drafts,            # [bs]，全部请求的接受草稿 token 数
    num_accepted_drafts_filter,     # [bs]，过滤后仅含未完成请求的接受数（已完成为 0）
    bs_upper: tl.constexpr,
    num_verify_tokens_upper: tl.constexpr,
):
    bid = tl.program_id(0)
    bs_offset = tl.arange(0, bs_upper)

    # 原始 tgt_cache_loc 中本请求的起点：所有请求接受数的前缀和 + bid（bonus）
    num_accepted_drafts_all = tl.load(
        num_accepted_drafts + bs_offset, mask=bs_offset < bid
    )
    old_start = tl.sum(num_accepted_drafts_all) + bid

    # 过滤后 tgt_cache_loc（out_cache_loc）中本请求的写入起点
    num_accepted_drafts_filter_all = tl.load(
        num_accepted_drafts_filter + bs_offset, mask=bs_offset < bid
    )
    new_start = tl.sum(num_accepted_drafts_filter_all)

    # 只复制未完成请求的 tgt_cache_loc 条目（copy_len = filter 中的值，已完成请求为 0）
    copy_len = tl.load(num_accepted_drafts_filter + bid)
    copy_offset = tl.arange(0, num_verify_tokens_upper)
    value = tl.load(
        tgt_cache_loc + old_start + copy_offset, mask=copy_offset < copy_len
    )
    tl.store(
        out_cache_loc + new_start + copy_offset, value, mask=copy_offset < copy_len
    )


@torch.compile(dynamic=True, disable=_is_npu)
def create_num_accepted_drafts_filter(
    num_accepted_drafts: torch.Tensor,        # [bs]，全部请求接受草稿数
    unfinished_index_device: torch.Tensor,    # 未完成请求的索引（GPU tensor）
    seq_lens: torch.Tensor,                   # [bs]，原地更新 seq_lens
):
    # 已完成请求在 filter 中置 0（不写入 out_cache_loc），未完成请求保留 accept+1
    num_accepted_drafts_filter = torch.zeros_like(num_accepted_drafts)
    num_accepted_drafts_filter[unfinished_index_device] = (
        num_accepted_drafts[unfinished_index_device] + 1
    )
    # 原地更新所有请求的 seq_lens（无论是否完成都要推进序列长度）
    seq_lens.add_(num_accepted_drafts + 1)
    return num_accepted_drafts_filter


@torch.compile(dynamic=True, disable=_is_npu)
def select_top_k_tokens(
    i: int,                   # 当前推测步（0-based）
    topk_p: torch.Tensor,     # 当前步 top-k 概率，shape 依步次而异
    topk_index: torch.Tensor, # 当前步 top-k token id
    hidden_states: torch.Tensor,  # 当前步输入的隐藏状态
    scores: torch.Tensor,     # 累积路径概率（树形推测解码中的路径分数）
    topk: int,
):
    if i == 0:
        # The first step after extend
        # 第一步：草稿 token 直接展平为 input_ids，hidden_states 按 topk 重复
        input_ids = topk_index.flatten()
        if hidden_states is not None:
            hidden_states = hidden_states.repeat_interleave(topk, dim=0)
        scores = topk_p  # shape: (b, topk)

        # tree_info 记录树结构信息：(路径概率, token id, 父节点索引)
        tree_info = (
            topk_p.unsqueeze(1),  # shape: (b, 1, topk)
            topk_index,  # shape: (b, topk)
            torch.arange(-1, topk, dtype=torch.long, device=input_ids.device)
            .unsqueeze(0)
            .repeat(topk_p.shape[0], 1),  # shape: (b, topk + 1)
        )
    else:
        # The later decode steps
        # 后续步骤：将上一步的累积分数 scores 与当前步的 topk_p 相乘，选出全局 topk 路径
        expand_scores = torch.mul(
            scores.unsqueeze(2), topk_p.reshape(-1, topk, topk)
        )  # (b, topk, 1) x (b, topk ,topk) -> (b, topk, topk)
        topk_cs_p, topk_cs_index = fast_topk(
            expand_scores.flatten(start_dim=1), topk, dim=-1
        )  # (b, topk)
        scores = topk_cs_p  # shape: (b, topk)

        # 从展平的 topk^2 候选中按全局最优索引取出 input_ids
        topk_index = topk_index.reshape(-1, topk**2)
        input_ids = torch.gather(topk_index, index=topk_cs_index, dim=1).flatten()

        if hidden_states.shape[0] > 0:
            # 从上一步的 hidden_states 中选出对应父节点的隐藏状态
            selected_input_index = topk_cs_index.flatten() // topk + torch.arange(
                0, hidden_states.shape[0], step=topk, device=topk_index.device
            ).repeat_interleave(topk)
            hidden_states = hidden_states[selected_input_index, :]

        tree_info = (
            expand_scores,  # shape: (b, topk, topk)
            topk_index,  # shape: (b, topk * topk)
            topk_cs_index + (topk**2 * (i - 1) + topk),  # shape: (b, topk)
        )

    return input_ids, hidden_states, scores, tree_info


def generate_simulated_accept_index(
    accept_index,
    predict,
    num_accepted_drafts,
    bs,
    spec_steps,
    simulate_acc_len: float = SIMULATE_ACC_LEN,
    simulate_acc_method: str = SIMULATE_ACC_METHOD,
):
    # 仅在 SIMULATE_ACC_LEN > 0 时调用，用于基准测试中绕过真实验证
    assert simulate_acc_len > 0.0

    if simulate_acc_method == "multinomial":
        # 以 simulate_acc_len 为均值、std=1.0 生成正态分布随机接受长度
        simulated_values = torch.normal(
            mean=simulate_acc_len,
            std=1.0,
            size=(1,),
            device="cpu",
        )
        # clamp simulated values to be between 1 and self.spec_steps
        simulated_values = torch.clamp(simulated_values, min=1.0, max=spec_steps + 1)
        simulate_acc_len = int(simulated_values.round().item())
    elif simulate_acc_method == "match-expected":
        # multinomial sampling does not match the expected length
        # we keep it for the sake of compatibility of existing tests
        # but it's better to use "match-expected" for the cases that need to
        # match the expected length, One caveat is that this will only sample
        # either round down or round up of the expected length
        # 以概率插值方式确保期望值与 simulate_acc_len 精确匹配
        simulate_acc_len = max(1.0, min(spec_steps + 1, simulate_acc_len))
        lower = int(simulate_acc_len // 1)
        upper = lower + 1 if lower < spec_steps + 1 else lower
        if lower == upper:
            simulate_acc_len = lower
        else:
            # 按小数部分决定上下界的权重，然后随机采样
            weight_upper = simulate_acc_len - lower
            weight_lower = 1.0 - weight_upper
            probs = torch.tensor([weight_lower, weight_upper], device="cpu")
            sampled_index = torch.multinomial(probs, num_samples=1)
            simulate_acc_len = lower if sampled_index == 0 else upper
    else:
        raise ValueError(f"Invalid simulate_acc_method: {SIMULATE_ACC_METHOD}")

    # 构造模拟的 accept_index：连续 simulate_acc_len 个槽位（从 accept_index 第一列开始）
    accept_indx_first_col = accept_index[:, 0].view(-1, 1)
    sim_accept_index = torch.full(
        (bs, spec_steps + 1), -1, dtype=torch.int32, device="cuda"
    )
    sim_accept_index[:, :simulate_acc_len] = accept_indx_first_col + torch.arange(
        simulate_acc_len, device=accept_index.device
    )
    # 将所有请求的接受数设为 simulate_acc_len - 1（不含 bonus）
    num_accepted_drafts.fill_(simulate_acc_len - 1)
    predict.fill_(100)  # some legit token id
    return sim_accept_index


def traverse_tree(
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    draft_tokens: torch.Tensor,
    grammar: BaseGrammarObject,
    allocate_token_bitmask: torch.Tensor,
    vocab_size: Optional[int] = None,
):
    """
    Traverse the tree constructed by the draft model to generate the logits mask.
    """
    assert (
        retrieve_next_token.shape == retrieve_next_sibling.shape == draft_tokens.shape
    )

    def dfs(
        curr: int,
        retrieve_next_token: torch.Tensor,
        retrieve_next_sibling: torch.Tensor,
        parent_pos: int,
    ):
        if curr == 0:
            # the first token generated by the target model, and thus it is always
            # accepted from the previous iteration
            # 树根节点（索引 0）是上一轮目标模型输出，不需要语法检查
            accepted = True
        else:
            # 检查当前草稿 token 是否在父节点的 grammar bitmask 中被允许
            parent_bitmask = allocate_token_bitmask[parent_pos]
            curr_token_id = draft_tokens[curr]
            if vocab_size and curr_token_id >= vocab_size:
                # token id 超出词汇表范围时直接拒绝
                accepted = False
            else:
                # 32 boolean bitmask values are packed into 32-bit integers
                # 从 int32 位掩码中查询对应 bit（32 个 bool 压缩为 1 个 int32）
                accepted = (
                    parent_bitmask[curr_token_id // 32] & (1 << (curr_token_id % 32))
                ) != 0

        if accepted:
            if curr != 0:
                # Accept the current token
                grammar.accept_token(draft_tokens[curr])
            if not grammar.is_terminated():
                # Generate the bitmask for the current token
                # 为当前 token 生成其子节点的 grammar 约束 bitmask
                grammar.fill_vocab_mask(allocate_token_bitmask, curr)
                if retrieve_next_token[curr] != -1:
                    # Visit the child node
                    dfs(
                        retrieve_next_token[curr],
                        retrieve_next_token,
                        retrieve_next_sibling,
                        curr,
                    )

            if curr != 0:
                # Rollback the current token
                # DFS 回溯：撤销已 accept 的 token，以便探索兄弟分支
                grammar.rollback(1)

        if retrieve_next_sibling[curr] != -1:
            # Visit the sibling node
            # 探索当前节点的兄弟节点（父节点相同的下一个 topk 分支）
            dfs(
                retrieve_next_sibling[curr],
                retrieve_next_token,
                retrieve_next_sibling,
                parent_pos,
            )

    dfs(0, retrieve_next_token, retrieve_next_sibling, -1)


def generate_token_bitmask(
    reqs: List[Req],
    verify_input: EagleVerifyInput,
    retrieve_next_token_cpu: torch.Tensor,
    retrieve_next_sibling_cpu: torch.Tensor,
    draft_tokens_cpu: torch.Tensor,
    vocab_size: int,
):
    """
    Generate the logit mask for structured output.
    Draft model's token can be either valid or invalid with respect to the grammar.
    We need to perform DFS to
    1. figure out which tokens are accepted by the grammar.
    2. if so, what is the corresponding logit mask.
    """

    num_draft_tokens = draft_tokens_cpu.shape[-1]

    # allocate_token_bitmask 延迟分配，首次遇到有 grammar 的请求时创建
    allocate_token_bitmask = None
    assert len(reqs) == retrieve_next_token_cpu.shape[0]
    grammar = None
    for i, req in enumerate(reqs):
        if req.grammar is not None:
            if allocate_token_bitmask is None:
                # 分配 [bs * num_draft_tokens, vocab_size/32] 大小的 bitmask（CPU tensor）
                allocate_token_bitmask = req.grammar.allocate_vocab_mask(
                    vocab_size=vocab_size,
                    batch_size=draft_tokens_cpu.numel(),
                    device="cpu",
                )
            grammar = req.grammar
            s = time.perf_counter()
            # 对第 i 条请求进行树遍历，填充 bitmask 的对应切片
            traverse_tree(
                retrieve_next_token_cpu[i],
                retrieve_next_sibling_cpu[i],
                draft_tokens_cpu[i],
                req.grammar,
                allocate_token_bitmask[
                    i * num_draft_tokens : (i + 1) * num_draft_tokens
                ],
                vocab_size=vocab_size,
            )
            tree_traverse_time = time.perf_counter() - s
            # 树遍历耗时超过阈值时发出警告（有助于发现超大树或低效 grammar）
            if tree_traverse_time > TREE_TRAVERSE_TIME_THRESHOLD:
                logger.warning(
                    f"Bit mask generation took {tree_traverse_time} seconds with "
                    f"grammar: {req.grammar}"
                )

    # 将末尾的 grammar 对象记录到 verify_input，供后续解码使用
    verify_input.grammar = grammar
    return allocate_token_bitmask


def load_token_map(token_map_path: str) -> List[int]:
    # 若本地文件不存在，尝试从 HuggingFace Hub 或 ModelScope 下载
    if not os.path.exists(token_map_path):
        repo_id = os.path.dirname(token_map_path)
        file_name = os.path.basename(token_map_path)

        cache_dir = None
        if envs.SGLANG_USE_MODELSCOPE.get():
            from modelscope.utils.file_utils import get_model_cache_root

            # 先检查 ModelScope 本地缓存
            cached_repo_path = os.path.join(get_model_cache_root(), repo_id)
            if os.path.exists(cached_repo_path):
                cache_dir = cached_repo_path

        if cache_dir is None:
            # 按平台选择下载函数（ModelScope 或 HuggingFace）
            if envs.SGLANG_USE_MODELSCOPE.get():
                from modelscope.hub.snapshot_download import (
                    snapshot_download as download_func,
                )
            else:
                download_func = snapshot_download
            # 只下载 token map 文件，跳过大型模型权重
            cache_dir = download_func(
                repo_id,
                ignore_patterns=["*.bin", "*.safetensors"],
            )

        token_map_path = os.path.join(cache_dir, file_name)
    # 加载 hot token id 列表并转为 int64 tensor
    hot_token_id = torch.load(token_map_path, weights_only=True)
    return torch.tensor(hot_token_id, dtype=torch.int64)


@contextmanager
def draft_tp_context(tp_group: GroupCoordinator):
    # Draft model doesn't use dp and has its own tp group.
    # We disable mscclpp now because it doesn't support 2 comm groups.
    # 切换到草稿模型的 TP 通信组（草稿模型有独立 TP，与目标模型隔离）
    with patch_tensor_parallel_group(tp_group):
        yield


def maybe_detect_nan(tensor: torch.Tensor, msg: str = ""):
    """Async NaN check — no GPU-CPU sync, error surfaces at next sync point."""
    # 仅在环境变量 SGLANG_SPEC_NAN_DETECTION 开启时执行，不引入同步开销
    if not envs.SGLANG_SPEC_NAN_DETECTION.get():
        return
    torch._assert_async(~torch.any(torch.isnan(tensor)), f"NaN detected! {msg}")


def maybe_detect_oob(indices: torch.Tensor, low: int, high: int, msg: str):
    """Async OOB check — no GPU-CPU sync, error surfaces at next sync point."""
    # 仅在 SGLANG_SPEC_OOB_DETECTION 开启时执行越界检查
    if not envs.SGLANG_SPEC_OOB_DETECTION.get():
        return
    if indices.numel() == 0:
        return
    torch._assert_async(
        (indices.min() >= low) & (indices.max() < high),
        f"OOB indices not in [{low}, {high}): {msg}",
    )


# Disable torch.compile for this function because it will be
# even slower.
# @torch.compile(dynamic=True)
def get_last_loc_large_page_size_large_top_k(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    speculative_num_steps: int,
    topk: int,
    page_size: int,
):
    # paged topk 模式下，每条请求需要为 topk 个分支各分配若干新页
    prefix_lens = seq_lens
    last_page_lens = prefix_lens % page_size
    # 每个 topk 分支所需的新页数：向上取整，确保能放下 last_page + num_steps 个 token
    num_new_pages_per_topk = (
        last_page_lens + speculative_num_steps + page_size - 1
    ) // page_size
    # 扩展后的序列长度：prefix 页对齐部分 + topk 个分支各占 num_new_pages_per_topk 页
    seq_lens = prefix_lens // page_size * page_size + num_new_pages_per_topk * (
        page_size * topk
    )
    extend_lens = seq_lens - prefix_lens
    # 获取每条请求最后一个已分配 KV slot 的位置（用于后续写入草稿 slot）
    last_loc = get_last_loc(
        req_to_token,
        req_pool_indices,
        prefix_lens,
    )

    return (
        prefix_lens,
        seq_lens,
        last_loc,
        num_new_pages_per_topk,
        extend_lens,
        last_page_lens,
    )
