from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

# 导入 FlashInfer KV 索引构建工具（用于分页 KV 缓存的地址映射）
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.dflash_utils import (
    compute_dflash_accept_len_and_bonus,
    compute_dflash_sampling_accept_len_and_bonus,
    is_dflash_sampling_verify_available,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func


def _compute_paged_keep_slots(
    *,
    prefix_lens: torch.Tensor,
    commit_lens: torch.Tensor,
    draft_token_num: int,
    page_size: int,
) -> torch.Tensor:
    """Compute how many draft slots per request must remain allocated.

    The allocator frees at page granularity for paged mode, so we can only release
    full pages from the tail after verify.
    """
    # 分页模式下，释放 KV 缓存只能以页为单位；
    # 本函数计算每个请求在验证后需保留的草稿 slot 数（向上对齐到页边界）

    if page_size <= 1:
        raise ValueError(f"Expected page_size > 1, got {page_size}.")

    seq_dtype = prefix_lens.dtype
    # extended_lens: 验证前分配的最大序列长度（前缀 + 全部草稿 token）
    extended_lens = prefix_lens + int(draft_token_num)
    # new_lens: 验证后实际提交的序列长度（前缀 + 接受的 token 数）
    new_lens = prefix_lens + commit_lens.to(seq_dtype)
    # aligned_new_lens: 向上对齐到页边界（确保不释放当前页中的已用 slot）
    aligned_new_lens = ((new_lens + page_size - 1) // page_size) * page_size
    # keep_lens: 实际保留的长度（不超过已分配的最大长度）
    keep_lens = torch.minimum(aligned_new_lens, extended_lens)
    # keep_slots: 需保留的草稿 slot 数（相对于前缀末尾的偏移量）
    keep_slots = (keep_lens - prefix_lens).to(torch.int64)
    keep_slots.clamp_(min=0, max=int(draft_token_num))
    return keep_slots


@dataclass
class DFlashDraftInput(SpecInput):
    """Per-batch DFlash draft state for spec-v1 (non-overlap) scheduling.

    This object is stored on `ScheduleBatch.spec_info` between decode iterations.
    It is NOT sent to model attention backends; the DFlash worker uses it to run
    the draft model and to track draft-side cache progress.

    When draft windowing is disabled, `draft_seq_lens` matches the committed target
    prefix length already materialized in the draft KV cache. When windowing is
    enabled, `draft_seq_lens` is the logical resident length in the draft worker's
    compact req-to-token mapping. In paged mode this may exceed the requested
    window by up to `page_size - 1` so the local page table remains valid. `ctx_lens`
    tracks newly committed target tokens that still need draft KV materialization.
    """

    # 每个请求下一个 DFlash block 的起始 token（上一步验证接受的最后一个 token）
    verified_id: torch.Tensor

    # 扁平化的上下文特征张量，需写入草稿 KV 缓存
    # 形状：[sum(ctx_lens), K * hidden_size]，K 为每个 token 拼接的目标层隐状态数量
    # （由 dflash_config.target_layer_ids 决定，旧 checkpoint 默认 K == draft_num_layers）
    target_hidden: torch.Tensor

    # 每个请求的上下文长度（用于切片 target_hidden），设备张量 int32
    ctx_lens: torch.Tensor

    # 每个请求在草稿 worker 中可见的已提交 token 数（草稿侧序列长度）
    draft_seq_lens: torch.Tensor

    def __post_init__(self):
        # 初始化父类，设置 spec_input_type 为 DFLASH_DRAFT
        super().__init__(spec_input_type=SpecInputType.DFLASH_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        # 草稿状态不改变 token 计数，返回 (1, 1) 表示乘数为 1
        return (1, 1)

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        # 保存旧的 ctx_lens 和 target_hidden，以便后续向量化重建
        old_ctx_lens = self.ctx_lens
        old_target_hidden = self.target_hidden

        # 按照新索引过滤各字段
        self.verified_id = self.verified_id[new_indices]
        self.ctx_lens = old_ctx_lens[new_indices]
        self.draft_seq_lens = self.draft_seq_lens[new_indices]

        # target_hidden 为空则直接保留原值，无需重建
        if old_target_hidden is None or old_target_hidden.numel() == 0:
            self.target_hidden = old_target_hidden
            return

        # 向量化重建 target_hidden：通过 cumsum 计算每个请求的起始偏移量
        old_bs = int(old_ctx_lens.shape[0])
        offsets = torch.zeros(
            (old_bs + 1,), dtype=torch.int64, device=old_ctx_lens.device
        )
        # offsets[1:] 存储各请求在扁平化 target_hidden 中的起始位置
        offsets[1:].copy_(old_ctx_lens.to(torch.int64).cumsum(0))

        # 取过滤后各请求的起始偏移和长度
        start = offsets[:-1]
        seg_start = start[new_indices]
        seg_lens = old_ctx_lens[new_indices].to(torch.int64)

        # 若所有过滤后请求均无上下文 token，返回空张量
        max_len = int(seg_lens.max().item()) if seg_lens.numel() > 0 else 0
        if max_len <= 0:
            self.target_hidden = old_target_hidden[:0]
            return

        # 构建 2D 位置索引矩阵 [new_bs, max_len]，用 mask 过滤越界位置
        r = torch.arange(max_len, device=old_ctx_lens.device, dtype=torch.int64)[
            None, :
        ]
        pos2d = seg_start[:, None] + r
        mask = r < seg_lens[:, None]
        # flat_pos: 所有有效 token 在原 target_hidden 中的绝对索引
        flat_pos = pos2d[mask]
        # index_select 按 flat_pos 从旧 target_hidden 中重建过滤后的版本
        self.target_hidden = (
            old_target_hidden.index_select(0, flat_pos)
            if flat_pos.numel() > 0
            else old_target_hidden[:0]
        )

    def merge_batch(self, spec_info: "DFlashDraftInput"):
        # 合并两个 DFlashDraftInput：将各字段在 batch 维度拼接
        self.verified_id = torch.cat([self.verified_id, spec_info.verified_id], dim=0)
        self.ctx_lens = torch.cat([self.ctx_lens, spec_info.ctx_lens], dim=0)
        self.draft_seq_lens = torch.cat(
            [self.draft_seq_lens, spec_info.draft_seq_lens], dim=0
        )
        # target_hidden 为空时直接替换，否则拼接
        if self.target_hidden is None or self.target_hidden.numel() == 0:
            self.target_hidden = spec_info.target_hidden
        elif (
            spec_info.target_hidden is not None and spec_info.target_hidden.numel() > 0
        ):
            self.target_hidden = torch.cat(
                [self.target_hidden, spec_info.target_hidden], dim=0
            )


@dataclass
class DFlashVerifyInput(SpecInput):
    """Inputs for a target-model verify forward in DFlash (spec-v1).

    The verify forward is run with `ForwardMode.TARGET_VERIFY` so that the target
    model returns logits for all tokens in the block, enabling accept-length
    computation.
    """

    # 本次验证的草稿 token 序列（展平格式：[bs * draft_token_num]）
    draft_token: torch.Tensor
    # 草稿 token 的位置编码索引（展平格式：[bs * draft_token_num]）
    positions: torch.Tensor
    # 每次验证的草稿 token 数量（固定步长，线性验证，非树形）
    draft_token_num: int
    # 与 EAGLE 后端兼容的 topk 字段；DFlash 为线性验证，始终为 1
    topk: int = 1
    # TARGET_VERIFY 时的自定义因果注意力掩码（True 表示允许 (q, k) 对）
    # 部分后端（如 triton）需要此掩码；flashinfer 等不需要
    custom_mask: torch.Tensor | None = None
    # 控制目标模型是否捕获隐状态（DFlash 验证需要 FULL 模式）
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    # DP 注意力 / CUDA Graph padding 所需的 batch token 数（-1 表示由 draft_token_num 自动设置）
    num_tokens_per_batch: int = -1

    def __post_init__(self):
        # 初始化父类，设置 spec_input_type 为 DFLASH_VERIFY
        super().__init__(spec_input_type=SpecInputType.DFLASH_VERIFY)
        # 未显式设置时，num_tokens_per_batch 默认等于 draft_token_num
        if self.num_tokens_per_batch == -1:
            self.num_tokens_per_batch = int(self.draft_token_num)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        # DFlash 验证阶段：每个序列位置对应 draft_token_num 个 token（线性块）
        return self.draft_token_num, self.draft_token_num

    def prepare_for_verify(
        self,
        batch: ScheduleBatch,
        page_size: int,
        *,
        build_custom_mask: bool = True,
    ):
        # idle 模式下无需准备，直接返回
        if batch.forward_mode.is_idle():
            return

        # 将草稿 token 设置为本轮验证的输入 token
        batch.input_ids = self.draft_token

        if page_size == 1:
            # 非分页模式：直接分配扁平 token slot
            batch.out_cache_loc = alloc_token_slots(
                batch.tree_cache, len(batch.input_ids)
            )
            # 目标序列长度 = 当前前缀 + 全部草稿 token
            end_offset = batch.seq_lens + self.draft_token_num
        else:
            # 分页模式：基于已有最后 loc 扩展分配分页 slot
            prefix_lens = batch.seq_lens
            prefix_lens_cpu = batch.seq_lens_cpu
            end_offset = prefix_lens + self.draft_token_num
            end_offset_cpu = prefix_lens_cpu + self.draft_token_num
            # 获取每个请求当前最后一个已分配 token 的位置
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                prefix_lens,
            )
            # 以页粒度扩展分配 draft_token_num 个新 slot
            batch.out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                prefix_lens,
                prefix_lens_cpu,
                end_offset,
                end_offset_cpu,
                last_loc,
                len(batch.input_ids),
            )
            # 保存 last_loc 供验证后的 paged keep-slot 计算使用
            self.last_loc = last_loc

        bs = batch.batch_size()
        # 将新分配的 KV slot 写入 req_to_token_pool 的页表映射
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )

        # 若后端不需要 custom_mask（如 flashinfer），直接置 None 并返回
        if not build_custom_mask:
            self.custom_mask = None
            return

        if self.draft_token_num <= 0:
            raise ValueError(
                f"DFLASH draft_token_num must be positive, got {self.draft_token_num}."
            )
        # 为每个请求构建因果注意力掩码：草稿 token 可以看到完整前缀 + 当前位置及之前的草稿 token
        mask_chunks: List[torch.Tensor] = []
        q_len = int(self.draft_token_num)
        # q_idx: [draft_token_num, 1]，表示每个草稿 token 的相对位置（0-based）
        q_idx = torch.arange(q_len, device=batch.device, dtype=torch.int32).unsqueeze(1)
        for prefix_len in batch.seq_lens_cpu.tolist():
            prefix_len_i = int(prefix_len)
            kv_len = prefix_len_i + q_len
            # k_idx: [1, kv_len]，表示 KV 中每个 token 的绝对位置
            k_idx = torch.arange(
                kv_len, device=batch.device, dtype=torch.int32
            ).unsqueeze(0)
            # 允许 (q, k) 对：k 在前缀范围内，或 k 在草稿范围内且 k <= prefix + q（因果）
            allow = k_idx <= (prefix_len_i + q_idx)
            mask_chunks.append(allow.flatten())
        # 将所有请求的掩码拼接为一维张量
        self.custom_mask = (
            torch.cat(mask_chunks, dim=0)
            if mask_chunks
            else torch.empty((0,), dtype=torch.bool, device=batch.device)
        )

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        device = req_pool_indices.device
        bs = len(req_pool_indices)

        # qo_indptr: 每个请求在展平 Q 序列中的起始偏移，步长为 draft_token_num
        qo_indptr = torch.arange(
            0,
            (bs + 1) * self.draft_token_num,
            step=self.draft_token_num,
            dtype=torch.int32,
            device=device,
        )

        # cum_kv_seq_len: 每个请求的 KV 序列长度（前缀 + draft_token_num）的累积和
        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        # 将草稿 token 数加入 KV 长度（paged_kernel_lens 原为前缀长度）
        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        # kv_indices: 存储每个请求所有 KV token 在 token pool 中的物理地址
        kv_indices = torch.empty(
            paged_kernel_lens_sum + self.draft_token_num * bs,
            dtype=torch.int32,
            device=device,
        )
        # 调用 Triton kernel 填充 kv_indices（将分页地址展开为连续索引）
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        mask = self.custom_mask
        if mask is not None:
            # 计算 custom_mask 应有的元素数：(前缀长度之和 + draft^2 * bs)
            mask_numel = (
                paged_kernel_lens_sum * self.draft_token_num
                + (self.draft_token_num**2) * bs
            )
            if mask.numel() < mask_numel:
                # FIXME(attn): temporary fix for custom mask padding with cuda graph
                # CUDA Graph 固定形状要求：将 mask 填充至所需大小（填充值为 True，表示允许）
                mask = torch.cat(
                    [
                        mask,
                        torch.full(
                            (mask_numel - mask.numel(),),
                            True,
                            dtype=torch.bool,
                            device=device,
                        ),
                    ],
                    dim=0,
                )
                self.custom_mask = mask
        return kv_indices, cum_kv_seq_len, qo_indptr, mask

    def verify(
        self,
        *,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        page_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """DFlash verification for greedy and non-greedy sampling.

        Returns:
            new_verified_id: int64 tensor [bs] (the new current token per request)
            commit_lens: int32 tensor [bs] (how many verify-input tokens are committed)
            next_target_hidden: tensor [sum(commit_lens), feature_dim]
            num_accepted_drafts_per_req_cpu: list[int] (accepted draft tokens per request)
        """
        # idle 模式下直接返回空张量
        if batch.forward_mode.is_idle():
            empty = torch.empty((0,), dtype=torch.int64, device=batch.device)
            return empty, empty.to(torch.int32), empty, []

        bs = batch.batch_size()
        device = logits_output.next_token_logits.device

        sampling_info = batch.sampling_info
        if sampling_info is not None:
            if len(sampling_info) != bs:
                raise RuntimeError(
                    "DFLASH verify sampling_info size mismatch: "
                    f"len(sampling_info)={len(sampling_info)}, bs={bs}."
                )

            # 应用自定义 logit 处理器（与普通采样路径保持一致）
            if sampling_info.has_custom_logit_processor:
                apply_custom_logit_processor(
                    logits_output.next_token_logits,
                    sampling_info,
                    num_tokens_in_batch=self.draft_token_num,
                )

            # 若有惩罚项或 logit_bias，构建线性惩罚张量并叠加到 logits
            if (
                sampling_info.penalizer_orchestrator.is_required
                or sampling_info.logit_bias is not None
            ):
                linear_penalty = torch.zeros(
                    (bs, logits_output.next_token_logits.shape[1]),
                    dtype=torch.float32,
                    device=device,
                )
                sampling_info.apply_logits_bias(linear_penalty)
                # 将 [bs, vocab] 的惩罚项展开为 [bs * draft_token_num, vocab] 后叠加
                logits_output.next_token_logits.add_(
                    torch.repeat_interleave(linear_penalty, self.draft_token_num, dim=0)
                )

        # candidates: [bs, draft_token_num]，草稿 token 矩阵
        candidates = self.draft_token.view(bs, self.draft_token_num)
        if (
            sampling_info is not None
            and not sampling_info.is_all_greedy
            and is_dflash_sampling_verify_available()
        ):
            # 非贪心采样：使用基于概率分布的接受/拒绝采样验证
            accept_len, bonus = compute_dflash_sampling_accept_len_and_bonus(
                candidates=candidates,
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
            )
        else:
            # 贪心验证：argmax 选出目标预测 token，与草稿 token 逐位比对
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).view(
                bs, self.draft_token_num
            )
            accept_len, bonus = compute_dflash_accept_len_and_bonus(
                candidates=candidates,
                target_predict=target_predict,
            )

        # 单次 D2H 传输：将 candidates[:,1:]、accept_len、bonus 打包一起转到 CPU
        # 避免多次小传输导致 GPU/CPU 同步开销
        packed = torch.cat(
            [candidates[:, 1:], accept_len.unsqueeze(1), bonus.unsqueeze(1)], dim=1
        ).cpu()

        max_acc = self.draft_token_num - 1
        num_accepted_drafts_per_req_cpu: List[int] = []
        commit_lens_cpu: List[int] = []
        new_verified_list: List[int] = []

        # 逐请求处理验证结果：确定接受的 token 数，更新请求的 output_ids
        for i, req in enumerate(batch.reqs):
            # acc_len: 本请求接受的草稿 token 数（不含 bonus token）
            acc_len = int(packed[i, max_acc].item())
            # proposed: 接受的草稿 token 序列 + bonus token（目标模型补充的一个新 token）
            proposed = packed[i, :acc_len].tolist() + [
                int(packed[i, max_acc + 1].item())
            ]

            appended = 0
            for token_id in proposed:
                token_id = int(token_id)
                req.output_ids.append(token_id)
                appended += 1
                # 检查请求是否已满足终止条件（EOS、max_len 等）
                req.check_finished()
                if req.finished():
                    break
                # 结构化输出：更新 grammar 状态机接受当前 token
                if req.grammar is not None:
                    req.grammar.accept_token(token_id)

            # 确定本步验证后该请求的 "当前 token"（供下一步草稿模型使用）
            if req.output_ids:
                new_verified_token = int(req.output_ids[-1])
            elif req.origin_input_ids:
                # 本步验证未追加任何 token，保持当前 token 不变
                new_verified_token = int(req.origin_input_ids[-1])
            else:
                raise RuntimeError(
                    "DFLASH verify cannot determine current token: both output_ids and origin_input_ids are empty."
                )

            commit_lens_cpu.append(appended)
            new_verified_list.append(new_verified_token)
            # 接受的草稿 token 数 = appended - 1（扣除 bonus token）
            num_accepted_drafts_per_req_cpu.append(max(0, appended - 1))
            req.spec_verify_ct += 1
            req.spec_accepted_drafts += num_accepted_drafts_per_req_cpu[-1]

        commit_lens = torch.tensor(commit_lens_cpu, dtype=torch.int32, device=device)
        new_verified_id = torch.tensor(
            new_verified_list, dtype=torch.int64, device=device
        )

        # 释放未提交的 KV 缓存 slot，并压缩 out_cache_loc
        if page_size == 1:
            # 非分页模式：精确按 commit_lens 释放多余 slot
            out_cache_loc = batch.out_cache_loc.view(bs, self.draft_token_num)
            keep_mask = (
                torch.arange(self.draft_token_num, device=device)[None, :]
                < commit_lens[:, None]
            )
            batch.token_to_kv_pool_allocator.free(out_cache_loc[~keep_mask])
            batch.out_cache_loc = out_cache_loc[keep_mask]
        else:
            # 分页模式：只能以页为单位释放，keep_slots 向上对齐到页边界
            out_cache_loc = batch.out_cache_loc.view(bs, self.draft_token_num)
            row_offsets = torch.arange(self.draft_token_num, device=device)[None, :]
            keep_slots = _compute_paged_keep_slots(
                prefix_lens=batch.seq_lens,
                commit_lens=commit_lens,
                draft_token_num=self.draft_token_num,
                page_size=page_size,
            )
            # 释放 row_offsets >= keep_slots 的 slot（超出保留边界的都要释放）
            free_mask = row_offsets >= keep_slots[:, None]
            batch.token_to_kv_pool_allocator.free(out_cache_loc[free_mask])

            # out_cache_loc 只保留已提交的 slot（row_offsets < commit_lens）
            keep_mask = row_offsets < commit_lens[:, None]
            batch.out_cache_loc = out_cache_loc[keep_mask]

        # 更新每个请求的 KV 缓存账本
        for req, commit_len in zip(batch.reqs, commit_lens_cpu, strict=True):
            req.kv_committed_len += commit_len
            req.kv_allocated_len = req.kv_committed_len

        # 将新提交的 token 写入 req_to_token_pool 的页表映射
        end_offset = batch.seq_lens + commit_lens.to(batch.seq_lens.dtype)
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )

        # 更新 batch 的序列长度统计（seq_lens、seq_lens_cpu、seq_lens_sum）
        batch.seq_lens.add_(commit_lens.to(batch.seq_lens.dtype))
        batch.seq_lens_cpu.add_(
            torch.tensor(commit_lens_cpu, dtype=batch.seq_lens_cpu.dtype)
        )
        # 保持 seq_lens_sum 与 flashinfer 索引更新器的缓冲区大小计算同步
        batch.seq_lens_sum += sum(commit_lens_cpu)

        # 从目标模型隐状态中提取已提交 token 的特征，作为下一步草稿 KV 物化的输入
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError(
                "DFLASH verify requires target hidden states, but got None."
            )
        # 将 hidden 重塑为 [bs, draft_token_num, feature_dim] 便于按请求切片
        hidden = hidden.view(bs, self.draft_token_num, -1)
        segments: List[torch.Tensor] = []
        for i, ln in enumerate(commit_lens_cpu):
            if ln > 0:
                # 只取前 ln 个 token 的隐状态（对应已提交的 token）
                segments.append(hidden[i, :ln, :])
        # 拼接所有请求的隐状态段，形状为 [sum(commit_lens), feature_dim]
        next_target_hidden = torch.cat(segments, dim=0) if segments else hidden[:0]

        # 清空 hidden_states，避免下游消费者误用（spec-v1 decode 不使用此字段）
        logits_output.hidden_states = None

        return (
            new_verified_id,
            commit_lens,
            next_target_hidden,
            num_accepted_drafts_per_req_cpu,
        )
