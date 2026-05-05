# eagle_info.py
# EAGLE 投机解码的核心数据结构：EagleVerifyInput / EagleDraftInput / EagleVerifyOutput
# 负责 token 树的构建、KV cache 分配、accept/reject 判断以及草稿状态更新
import logging
from copy import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

# 语法约束 backend（JSON schema / EBNF 等）
from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.distributed import get_tp_group
from sglang.srt.environ import envs
# FlashInfer KV indices 构建 Triton kernel
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    is_dp_attention_enabled,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.managers.overlap_utils import FutureIndices
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,   # 分页 KV 扩展分配
    alloc_token_slots,                 # 非分页 KV 分配
    get_last_loc,                      # 获取序列最后 KV 位置（用于分页续写）
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.server_args import get_global_server_args
# V2 Mixin 提供 plan_stream / prepare_for_v2_verify 等接口
from sglang.srt.speculative.eagle_info_v2 import (
    EagleDraftInputV2Mixin,
    EagleVerifyInputV2Mixin,
)
from sglang.srt.speculative.eagle_utils import verify_tree_greedy_func
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import (
    SIMULATE_ACC_LEN,                   # 模拟接受长度（调试用）
    TREE_SPEC_KERNEL_AVAILABLE,         # 是否有 tree speculative sampling kernel（AMD 无）
    align_evict_mask_to_page_size,      # 按页大小对齐驱逐 mask
    assign_req_to_token_pool_func,      # 将 KV 位置写入 req_to_token 表
    create_extend_after_decode_spec_info,
    create_num_accepted_drafts_filter,
    filter_finished_cache_loc_kernel,
    generate_simulated_accept_index,
    get_src_tgt_cache_loc,
    get_target_cache_loc,
)
from sglang.srt.utils import is_cuda, is_musa, next_power_of_2

# CUDA/MUSA 环境才有 sgl_kernel 的 speculative sampling 实现
if is_cuda() or is_musa():
    from sgl_kernel import (
        top_k_renorm_prob,                          # top-k 概率归一化
        top_p_renorm_prob,                          # top-p 概率归一化
        tree_speculative_sampling_target_only,      # token 树投机采样 kernel
    )

logger = logging.getLogger(__name__)


@dataclass
class EagleVerifyInput(SpecInput, EagleVerifyInputV2Mixin):
    # 验证阶段输入：包含草稿 token 树、注意力 mask、检索索引等
    draft_token: torch.Tensor          # 草稿 token 序列（展平的 token 树）
    custom_mask: torch.Tensor          # 注意力自定义 mask（树状结构的因果 mask）
    positions: torch.Tensor            # 草稿 token 的 RoPE 位置
    retrieve_index: torch.Tensor       # 树节点检索索引（[bs, draft_token_num]）
    retrieve_next_token: torch.Tensor  # 树中每个节点的下一个候选 token 索引
    retrieve_next_sibling: torch.Tensor # 树中同级相邻节点索引（用于兄弟节点遍历）
    retrieve_cum_len: torch.Tensor     # 累积长度（用于变长 token 树）
    spec_steps: int                    # 投机解码步数
    topk: int                          # 每步保留的 top-k 候选
    draft_token_num: int               # 每个请求的草稿 token 数（= spec_steps * topk）
    capture_hidden_mode: CaptureHiddenMode  # hidden states 捕获模式
    seq_lens_sum: int                  # 批次总 token 数（用于 FlashInfer 注意力计算）
    seq_lens_cpu: torch.Tensor         # CPU 端序列长度（避免 GPU-CPU 同步）
    grammar: BaseGrammarObject = None  # 语法约束对象（可选）

    # Shape info for padding
    # 用于 CUDA graph padding 的每请求 token 数（-1 时自动填为 draft_token_num）
    num_tokens_per_req: int = -1  # -1 auto-fills from draft_token_num.

    def __post_init__(self):
        super().__init__(SpecInputType.EAGLE_VERIFY)
        if self.num_tokens_per_req < 0:
            self.num_tokens_per_req = self.draft_token_num

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        # 返回 (draft_token_num, draft_token_num) 供 token 计数调整
        return self.draft_token_num, self.draft_token_num

    @classmethod
    def create_idle_input(cls, topk: int, spec_steps: int, num_verify_tokens: int):
        # 创建空的验证输入（用于 IDLE 批次占位，避免 None 判断）
        return cls(
            draft_token=torch.empty((0,), dtype=torch.long, device="cuda"),
            custom_mask=torch.full((0,), True, dtype=torch.bool, device="cuda"),
            positions=torch.empty((0,), dtype=torch.int64, device="cuda"),
            retrieve_index=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device="cuda"
            ),
            retrieve_next_token=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device="cuda"
            ),
            retrieve_next_sibling=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device="cuda"
            ),
            retrieve_cum_len=None,
            topk=topk,
            draft_token_num=num_verify_tokens,
            spec_steps=spec_steps,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=0,
            seq_lens_cpu=torch.empty((0,), dtype=torch.int32),
        )

    def prepare_for_verify(self, batch: ScheduleBatch, page_size: int):
        # 验证前准备：分配草稿 token 的 KV cache 位置，写入 req_to_token 表

        if batch.forward_mode.is_idle():
            return

        # 将草稿 token 序列设置为 batch 的 input_ids
        batch.input_ids = self.draft_token

        if page_size == 1:
            # 非分页：直接分配连续 KV 槽位（draft_token_num 个）
            batch.out_cache_loc = alloc_token_slots(
                batch.tree_cache,
                len(batch.input_ids),
            )
            end_offset = batch.seq_lens + self.draft_token_num
        else:
            # 分页：需要保证在已有页的基础上续写（alloc_paged_token_slots_extend）
            prefix_lens = batch.seq_lens
            prefix_lens_cpu = batch.seq_lens_cpu
            end_offset = prefix_lens + self.draft_token_num
            end_offset_cpu = prefix_lens_cpu + self.draft_token_num
            # 获取每个请求当前最后一个 KV 槽位（用于分页续写）
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                prefix_lens,
            )
            batch.out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                prefix_lens,
                prefix_lens_cpu,
                end_offset,
                end_offset_cpu,
                last_loc,
                len(batch.input_ids),
            )
            self.last_loc = last_loc

        bs = batch.batch_size()
        # 将草稿 token 的 KV 位置写入 req_to_token 表（[seq_lens, end_offset)）
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )

        # Mamba 状态追踪：为 SSM 模型准备 ping-pong 缓冲区索引
        if get_global_server_args().enable_mamba_extra_buffer():
            batch.mamba_track_indices = torch.tensor(
                [
                    req.mamba_ping_pong_track_buffer[req.mamba_next_track_idx]
                    for req in batch.reqs
                ],
                dtype=torch.int64,
                device=batch.device,
            )
            batch.mamba_track_mask = None
            batch.mamba_track_seqlens = None

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        # 生成 FlashInfer prefill 注意力参数：qo_indptr / kv_indices / cum_kv_seq_len / custom_mask
        # TARGET_VERIFY 模式复用 prefill 注意力（而非 decode 注意力），故需要此方法
        device = req_pool_indices.device
        batch_size = len(req_pool_indices)
        # qo_indptr：每个请求查询 token 的起始指针（步长 = draft_token_num）
        qo_indptr = torch.arange(
            0,
            (1 + batch_size) * self.draft_token_num,
            step=self.draft_token_num,
            dtype=torch.int32,
            device=device,
        )
        cum_kv_seq_len = torch.zeros(
            (batch_size + 1,), dtype=torch.int32, device=device
        )

        # KV 长度 = 前缀长度 + 草稿 token 数（验证时草稿 token 也是 KV 的一部分）
        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        # 构建 KV indices（用于 paged attention 的物理页索引查找）
        kv_indices = torch.empty(
            paged_kernel_lens_sum + self.draft_token_num * batch_size,
            dtype=torch.int32,
            device=device,
        )
        create_flashinfer_kv_indices_triton[(batch_size,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        # custom_mask 总大小：前缀 KV × draft_token_num + 树内注意力 mask
        mask_numel = (
            paged_kernel_lens_sum * self.draft_token_num
            + (self.draft_token_num**2) * batch_size
        )
        if self.custom_mask.numel() < mask_numel:
            # FIXME(attn): temporary fix for custom mask padding with cuda graph
            # CUDA graph 场景需要 mask 大小固定，不足时用 True 填充（允许所有注意力）
            self.custom_mask = torch.cat(
                [
                    self.custom_mask,
                    torch.full(
                        (mask_numel - self.custom_mask.numel(),),
                        True,
                        dtype=torch.bool,
                        device=device,
                    ),
                ],
                dim=0,
            )

        return kv_indices, cum_kv_seq_len, qo_indptr, self.custom_mask

    def verify(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
        vocab_mask: Optional[torch.Tensor] = None,  # For grammar
    ) -> torch.Tensor:
        """
        Verify and find accepted tokens based on logits output and batch
        (which contains spec decoding information).

        WARNING: This API in-place modifies the states of logits_output

        This API updates values inside logits_output based on the accepted
        tokens. I.e., logits_output.next_token_logits only contains
        accepted token logits.
        """
        # 验证核心：对比草稿 token 与目标模型 logits，执行 accept/reject
        if batch.forward_mode.is_idle():
            # IDLE 批次：返回空输出（保持 draft_input 结构）
            return EagleVerifyOutput(
                draft_input=EagleDraftInput.create_idle_input(
                    device=batch.device,
                    hidden_size=batch.model_config.spec_hidden_size,
                    dtype=batch.model_config.dtype,
                    topk=self.topk,
                    capture_hidden_mode=CaptureHiddenMode.LAST,
                ),
                logits_output=logits_output,
                verified_id=torch.empty(0, dtype=torch.long, device=batch.device),
                num_accepted_drafts_per_req_cpu=[],
                accepted_indices=torch.full(
                    (0, self.spec_steps + 1),
                    -1,
                    dtype=torch.int32,
                    device=batch.device,
                ),
            )

        bs = self.retrieve_index.shape[0]
        # candidates[bs, draft_token_num]：草稿 token 候选矩阵
        candidates = self.draft_token.reshape(bs, self.draft_token_num)
        sampling_info = batch.sampling_info

        predict_shape = list(logits_output.next_token_logits.shape)[:-1]
        predict_shape[-1] += 1
        # predict：最终采样结果 [bs, spec_steps+1]（含目标模型的额外 token）
        predict = torch.empty(predict_shape, dtype=torch.int32, device=batch.device)
        # accept_index：记录每步接受的草稿 token 位置（-1 表示拒绝）
        accept_index = torch.full(
            (bs, self.spec_steps + 1), -1, dtype=torch.int32, device=batch.device
        )
        num_accepted_drafts = torch.empty((bs,), dtype=torch.int32, device=batch.device)

        if bs != len(sampling_info):
            # 当批次中有些请求已经 idle 时，需要过滤 sampling_info 以匹配实际批次
            sampling_info = copy.deepcopy(sampling_info)
            # NOTE: retrieve_index are the indices of the requests that are kept.
            sampling_info.filter_batch(
                self.retrieve_index.tolist(), self.retrieve_index
            )

        # Apply the custom logit processors if registered in the sampling info.
        # 应用用户注册的自定义 logit processor（如 bias、阻止某些 token 等）
        if sampling_info.has_custom_logit_processor:
            apply_custom_logit_processor(
                logits_output.next_token_logits,
                sampling_info,
                num_tokens_in_batch=self.draft_token_num,
            )

        # Apply penalty
        # 应用频率/存在惩罚和 logit_bias（投机解码的宽松版本：重复应用 draft_token_num 次）
        if (
            sampling_info.penalizer_orchestrator.is_required
            or sampling_info.logit_bias is not None
        ):
            # This is a relaxed version of penalties for speculative decoding.
            sampling_info.penalizer_orchestrator.apply(
                logits_output.next_token_logits, repeat=self.draft_token_num
            )
            if sampling_info.logit_bias is not None:
                logits_output.next_token_logits.add_(
                    torch.repeat_interleave(
                        sampling_info.logit_bias, self.draft_token_num, dim=0
                    )
                )

        # Apply grammar mask
        # 应用语法约束 mask（如 JSON schema 限制）
        if vocab_mask is not None:
            assert self.grammar is not None
            self.grammar.apply_vocab_mask(
                logits=logits_output.next_token_logits, vocab_mask=vocab_mask
            )

        # Sample tokens. Force greedy sampling on AMD
        # greedy 采样：AMD/HIP 构建不支持 tree_speculative_sampling_target_only，强制 greedy
        is_all_greedy = sampling_info.is_all_greedy
        if (not is_all_greedy) and (not TREE_SPEC_KERNEL_AVAILABLE):
            logger.warning(
                "Tree speculative sampling kernel unavailable (likely AMD/HIP build). "
                "Falling back to greedy verification."
            )

        if is_all_greedy or not TREE_SPEC_KERNEL_AVAILABLE:
            # Greedy 验证：argmax 得到目标预测，与草稿 token 逐步比较
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1)
            target_predict = target_predict.reshape(bs, self.draft_token_num)
            predict, accept_index, num_accepted_drafts = verify_tree_greedy_func(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=num_accepted_drafts,  # mutable
                candidates=candidates,
                retrieve_index=self.retrieve_index,
                retrieve_next_token=self.retrieve_next_token,
                retrieve_next_sibling=self.retrieve_next_sibling,
                target_predict=target_predict,
                topk=self.topk,
            )

        else:
            # apply temperature and get target probs
            # 非 greedy：计算目标分布，使用拒绝采样（rejection sampling）做验证
            expanded_temperature = torch.repeat_interleave(
                sampling_info.temperatures, self.draft_token_num, dim=0
            )  # (bs * draft_token_num, 1)

            # 目标模型的 softmax 概率分布（带 temperature 缩放）
            target_probs = F.softmax(
                logits_output.next_token_logits / expanded_temperature, dim=-1
            )  # (bs * draft_token_num, vocab_size)
            # top-k renorm：仅保留 top-k 概率并归一化
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ks, self.draft_token_num, dim=0
                ),
            )  # (bs * draft_token_num, vocab_size)
            if sampling_info.need_top_p_sampling:
                target_probs = top_p_renorm_prob(
                    target_probs,
                    torch.repeat_interleave(
                        sampling_info.top_ps, self.draft_token_num, dim=0
                    ),
                )
            target_probs = target_probs.reshape(bs, self.draft_token_num, -1)

            # draft_probs 对于 target_only 采样全为 0（草稿模型概率不参与拒绝采样）
            draft_probs = torch.zeros(
                target_probs.shape, dtype=torch.float32, device=batch.device
            )

            # coins for rejection sampling
            # 拒绝采样随机数：每个 token 位置生成一个均匀分布随机数
            coins = torch.rand_like(
                candidates, dtype=torch.float32, device=batch.device
            )
            # coins for final sampling
            # 最终 token 采样随机数（生成被拒绝后的替换 token 用）
            coins_for_final_sampling = torch.rand(
                (bs,), dtype=torch.float32, device=batch.device
            )
            # 调用 sgl_kernel 的 tree speculative sampling kernel（支持 top-k/p + 拒绝采样）
            tree_speculative_sampling_target_only(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=num_accepted_drafts,  # mutable
                candidates=candidates,
                # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
                retrive_index=self.retrieve_index,
                retrive_next_token=self.retrieve_next_token,
                retrive_next_sibling=self.retrieve_next_sibling,
                uniform_samples=coins,
                uniform_samples_for_final_sampling=coins_for_final_sampling,
                target_probs=target_probs,
                draft_probs=draft_probs,
                threshold_single=get_global_server_args().speculative_accept_threshold_single,
                threshold_acc=get_global_server_args().speculative_accept_threshold_acc,
                deterministic=True,
            )

            # Sync sampling results across TP ranks: different GPUs may
            # produce slightly different target_probs due to floating-point
            # non-determinism in softmax/top_k/top_p, causing different
            # sampled tokens. Broadcast from rank 0 to ensure consistency.
            # TP>1 时，各 rank 的 softmax 可能因浮点非确定性产生不同结果
            # 从 rank 0 广播以确保所有 rank 的采样结果一致
            tp_group = (
                get_attention_tp_group()
                if is_dp_attention_enabled()
                else get_tp_group()
            )
            if tp_group.world_size > 1:
                tp_group.broadcast(predict, src=0)
                tp_group.broadcast(accept_index, src=0)
                tp_group.broadcast(num_accepted_drafts, src=0)

        if SIMULATE_ACC_LEN > 0.0:
            # Do simulation
            # 调试用：模拟固定接受长度（忽略真实采样结果，用于性能基准）
            accept_index = generate_simulated_accept_index(
                accept_index=accept_index,
                predict=predict,  # mutable
                num_accepted_drafts=num_accepted_drafts,  # mutable
                bs=bs,
                spec_steps=self.spec_steps,
            )

        # 收集未完成的请求索引（用于后续 KV cache 整理和草稿状态更新）
        unfinished_index = []
        unfinished_accept_index = []
        accept_index_cpu = accept_index.tolist()
        predict_cpu = predict.tolist()
        has_finished = False
        think_end_id = batch.model_config.think_end_id

        # Iterate every accepted token and check if req has finished after append the token
        # should be checked BEFORE free kv cache slots
        # 逐请求检查接受 token 序列，更新输出 IDs 并判断是否结束
        # 必须在释放 KV cache 之前执行（否则后续操作引用无效槽位）
        for i, (req, accept_index_row) in enumerate(zip(batch.reqs, accept_index_cpu)):
            num_accepted = 0
            for j, idx in enumerate(accept_index_row):
                if idx == -1:
                    break
                num_accepted += 1
                id = predict_cpu[idx]
                req.output_ids.append(id)
                # 思维链模式：追踪 think_end_id 以标记推理结束
                if req.require_reasoning and think_end_id is not None:
                    req.update_reasoning_tokens(id, think_end_id)
                req.check_finished()
                if not req.finished() and req.grammar is not None:
                    try:
                        # 语法约束：接受新 token 并检查是否生成完整的语法结构
                        req.grammar.accept_token(id)
                    except ValueError as e:
                        logger.info(
                            f"{i=}, {req=}\n" f"{accept_index=}\n" f"{predict=}\n"
                        )
                        raise e
                    req.check_finished()
                if req.finished():
                    has_finished = True
                    # set all tokens after finished token to -1 and break
                    # 标记后续 token 为 -1（已结束的请求不需要更多 token）
                    accept_index[i, j + 1 :] = -1
                    break
            # Update KV cache tracking for the accepted tokens
            # 更新 KV cache 追踪计数（committed_len = 已写入且永久保留的长度）
            req.kv_committed_len += num_accepted
            req.kv_allocated_len = req.kv_committed_len
            if not req.finished():
                unfinished_index.append(i)
                if idx == -1:
                    unfinished_accept_index.append(accept_index[i, :j])
                else:
                    unfinished_accept_index.append(accept_index[i])
            req.spec_verify_ct += 1
            # 统计草稿接受率（不含目标模型强制生成的最后一个 token）
            accepted_draft_tokens = sum(1 for idx in accept_index_row if idx != -1) - 1
            req.spec_accepted_drafts += accepted_draft_tokens
            req.update_spec_acceptance_histogram(accepted_draft_tokens)

        if has_finished:
            # 有请求结束时重新计算 num_accepted_drafts（不能复用之前的值）
            num_accepted_drafts = (accept_index != -1).sum(dim=1) - 1

        # Free the KV cache for unaccepted tokens
        # TODO: fuse them
        # 释放被拒绝 token 占用的 KV cache 槽位
        accept_index = accept_index[accept_index != -1]
        verified_id = predict[accept_index]
        # evict_mask：True = 被拒绝（需释放），False = 被接受（保留）
        evict_mask = torch.full_like(self.draft_token, True, dtype=torch.bool)
        evict_mask[accept_index] = False
        num_accepted_drafts_cpu = num_accepted_drafts.cpu()
        num_accepted_tokens_cpu = num_accepted_drafts_cpu + 1
        # FIXME: this `tolist()` fixes the numerical calculation consistency
        # try to unify the tensor representation and list representation
        num_accepted_drafts_list = num_accepted_drafts_cpu.tolist()
        num_accepted_tokens_list = num_accepted_tokens_cpu.tolist()

        if page_size == 1:
            # TODO: boolean array index leads to a device sync. Remove it.
            # 非分页：直接用布尔 mask 释放被拒绝 token 的 KV 槽位
            token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
        else:
            if self.topk == 1:
                # Only evict full empty page. Do not evict partial empty page
                # topk=1（线性序列）：只驱逐完全为空的页，保留部分填充的最后页
                align_evict_mask_to_page_size[len(batch.seq_lens),](
                    batch.seq_lens,
                    evict_mask,
                    page_size,
                    self.draft_token_num,
                    next_power_of_2(self.draft_token_num),
                )
                token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
            else:
                # Shift the accepted tokens to the beginning.
                # Only evict the last part
                # topk>1（树状候选）：将接受的 token 移到前面，释放后部被拒绝的页
                src_cache_loc, tgt_cache_loc, to_free_num_slots = get_src_tgt_cache_loc(
                    batch.seq_lens,
                    batch.out_cache_loc,
                    accept_index,
                    num_accepted_drafts,
                    self.draft_token_num,
                    page_size,
                )
                to_free_slots = torch.empty(
                    (to_free_num_slots.sum().item(),),
                    dtype=torch.int64,
                    device=to_free_num_slots.device,
                )

                # out_cache_loc: [0  1  2,  3  4  5,  6  7  8]
                # accept_index:  [0 -1  2,  3  4 -1,  6 -1 -1]
                # tgt_cache_loc: [0  1   ,  3  4   ,  6      ]
                # to_free_slots: [      2,        5,     7  8]
                # to_free_slots also needs to be page-aligned without the first partial page
                #
                # split each row of out_cache_loc into two parts.
                # 1. the first part goes to tgt_cache_loc. length = num_accepted_drafts[i] + 1
                # 2. the second part goes to to_free_slots.
                # 将 out_cache_loc 分割：前半部分（接受）→ tgt_cache_loc；后半部分（拒绝）→ to_free_slots
                get_target_cache_loc[(bs,)](
                    tgt_cache_loc,
                    to_free_slots,
                    num_accepted_drafts,
                    to_free_num_slots,
                    batch.out_cache_loc,
                    self.draft_token_num,
                    next_power_of_2(self.draft_token_num),
                    next_power_of_2(bs),
                )

                # Free the kv cache
                # 释放被拒绝的 KV cache 槽位
                token_to_kv_pool_allocator.free(to_free_slots)

                # Copy the kv cache
                # 将接受的 KV cache 移到连续的位置（紧凑化）
                batch.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
                    tgt_cache_loc, src_cache_loc
                )

        # Construct EagleVerifyOutput
        # 构造 EagleVerifyOutput 返回结果
        if not has_finished:
            # 没有请求结束：简单路径，更新所有请求的 KV 位置和序列长度
            if page_size == 1 or self.topk == 1:
                batch.out_cache_loc = batch.out_cache_loc[accept_index]
                assign_req_to_token_pool_func(
                    batch.req_pool_indices,
                    batch.req_to_token_pool.req_to_token,
                    batch.seq_lens,
                    batch.seq_lens + num_accepted_drafts + 1,
                    batch.out_cache_loc,
                    bs,
                )
            else:
                batch.out_cache_loc = tgt_cache_loc
            batch.seq_lens.add_(num_accepted_drafts + 1)
            batch.seq_lens_cpu.add_(num_accepted_tokens_cpu)

            # 创建下一轮草稿输入（hidden_states 来自被接受位置）
            draft_input = EagleDraftInput(
                hidden_states=batch.spec_info.hidden_states[accept_index],
                verified_id=verified_id,
                num_accepted_drafts=num_accepted_drafts,
                num_accepted_tokens=num_accepted_drafts + 1,
                num_accepted_drafts_cpu=num_accepted_drafts_list,
                num_accepted_tokens_cpu=num_accepted_tokens_list,
                seq_lens_for_draft_extend=batch.seq_lens,
                seq_lens_for_draft_extend_cpu=batch.seq_lens_cpu,
                req_pool_indices_for_draft_extend=batch.req_pool_indices,
            )

            return EagleVerifyOutput(
                draft_input=draft_input,
                logits_output=logits_output,
                verified_id=verified_id,
                num_accepted_drafts_per_req_cpu=draft_input.num_accepted_drafts_cpu,
                accepted_indices=accept_index,
            )
        else:
            # 有请求结束：需要区分已完成和未完成的请求，分别处理
            if page_size == 1 or self.topk == 1:
                assign_req_to_token_pool_func(
                    batch.req_pool_indices,
                    batch.req_to_token_pool.req_to_token,
                    batch.seq_lens,
                    batch.seq_lens + num_accepted_drafts + 1,
                    batch.out_cache_loc[accept_index],
                    bs,
                )
                batch.seq_lens.add_(num_accepted_drafts + 1)
                batch.seq_lens_cpu.add_(num_accepted_tokens_cpu)

            if len(unfinished_accept_index) > 0:
                # 仍有未完成请求：为其构建草稿输入
                unfinished_accept_index = torch.cat(unfinished_accept_index)
                unfinished_index_device = torch.tensor(
                    unfinished_index, dtype=torch.int64, device=predict.device
                )
                draft_input_num_accepted_drafts_cpu = [
                    num_accepted_drafts_list[i] for i in unfinished_index
                ]
                draft_input_num_accepted_tokens_cpu = [
                    num_accepted_tokens_list[i] for i in unfinished_index
                ]
                if page_size == 1 or self.topk == 1:
                    batch.out_cache_loc = batch.out_cache_loc[unfinished_accept_index]
                else:
                    # topk>1 + 有请求结束时：需要重新整理 tgt_cache_loc（仅保留未完成请求的部分）
                    batch.out_cache_loc = torch.empty(
                        len(unfinished_index)
                        + sum(draft_input_num_accepted_drafts_cpu),
                        dtype=torch.int64,
                        device=predict.device,
                    )
                    num_accepted_drafts_filter = create_num_accepted_drafts_filter(
                        num_accepted_drafts,
                        unfinished_index_device,
                        batch.seq_lens,
                    )
                    batch.seq_lens_cpu.add_(num_accepted_tokens_cpu)
                    filter_finished_cache_loc_kernel[(bs,)](
                        batch.out_cache_loc,
                        tgt_cache_loc,
                        num_accepted_drafts,
                        num_accepted_drafts_filter,
                        next_power_of_2(bs),
                        next_power_of_2(self.draft_token_num),
                    )

                unfinished_num_accepted_drafts = num_accepted_drafts[
                    unfinished_index_device
                ]
                # 创建未完成请求的草稿输入（hidden_states 来自 unfinished_accept_index）
                draft_input = EagleDraftInput(
                    hidden_states=batch.spec_info.hidden_states[
                        unfinished_accept_index
                    ],
                    verified_id=predict[unfinished_accept_index],
                    num_accepted_drafts_cpu=draft_input_num_accepted_drafts_cpu,
                    num_accepted_tokens_cpu=draft_input_num_accepted_tokens_cpu,
                    num_accepted_drafts=unfinished_num_accepted_drafts,
                    num_accepted_tokens=unfinished_num_accepted_drafts + 1,
                    seq_lens_for_draft_extend=batch.seq_lens[unfinished_index_device],
                    seq_lens_for_draft_extend_cpu=batch.seq_lens_cpu[unfinished_index],
                    req_pool_indices_for_draft_extend=batch.req_pool_indices[
                        unfinished_index_device
                    ],
                )
            else:
                # 所有请求均已结束：返回 IDLE 草稿输入
                draft_input = EagleDraftInput.create_idle_input(
                    device=batch.device,
                    hidden_size=batch.model_config.spec_hidden_size,
                    dtype=batch.model_config.dtype,
                    topk=self.topk,
                    capture_hidden_mode=CaptureHiddenMode.LAST,
                )

            return EagleVerifyOutput(
                draft_input=draft_input,
                logits_output=logits_output,
                verified_id=verified_id,
                num_accepted_drafts_per_req_cpu=num_accepted_drafts_list,
                accepted_indices=accept_index,
            )


@dataclass
class EagleDraftInput(SpecInput, EagleDraftInputV2Mixin):
    # The inputs for decode
    # 解码（草稿生成）阶段的输入数据
    # shape: (b, topk)
    # topk_p / topk_index：每步 topk 采样的概率和 token 索引（用于多路径树展开）
    topk_p: torch.Tensor = None
    topk_index: torch.Tensor = None
    # shape: (b, hidden_size)
    # 目标模型最后一层 hidden states（EAGLE 草稿模型的输入特征）
    hidden_states: torch.Tensor = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    # Inputs for extend
    # shape: (b,)
    # `num_accepted_drafts` and `num_accepted_tokens` are kept in sync:
    # `num_accepted_tokens = num_accepted_drafts + 1` (per-req, one bonus per req).
    # Storing both avoids repeated `+ 1` at every consumer (attn backends, kernels).
    # num_accepted_drafts + 1 = num_accepted_tokens（多一个目标模型强制生成的 bonus token）
    verified_id: torch.Tensor = None          # 上一轮验证后接受的 token ID
    num_accepted_drafts: torch.Tensor = None  # 每个请求本轮接受的草稿 token 数
    num_accepted_tokens: torch.Tensor = None  # 每个请求本轮总接受 token 数（= drafts + 1）
    num_accepted_drafts_cpu: List[int] = None # CPU 端接受草稿数（供调度器统计）
    num_accepted_tokens_cpu: List[int] = None # CPU 端总接受 token 数（供调度器统计）

    # Inputs for the attention backends
    # shape: (b + 1,)
    # kv_indptr / kv_indices：草稿 extend 注意力的 KV 指针和索引（非分页时使用）
    kv_indptr: torch.Tensor = None
    kv_indices: torch.Tensor = None

    # Shape info for padding
    # CUDA graph padding 用的每请求 token 数
    num_tokens_per_req: int = -1
    num_tokens_for_logprob_per_req: int = -1

    # Inputs for draft extend
    # shape: (b,)
    # 草稿 extend 阶段需要的序列信息（accept 后序列已更新，专门传给草稿 worker）
    seq_lens_for_draft_extend: torch.Tensor = None
    seq_lens_for_draft_extend_cpu: torch.Tensor = None
    req_pool_indices_for_draft_extend: torch.Tensor = None

    # Inputs for V2 overlap worker
    # V2 plan_stream 重叠执行相关：FutureIndices 用于懒惰获取 draft extend 后的 KV 索引
    future_indices: Optional[FutureIndices] = None
    new_seq_lens: Optional[torch.Tensor] = None
    verify_done: Optional[torch.cuda.Event] = None  # 验证完成的 CUDA event（用于同步）

    def __post_init__(self):
        super().__init__(SpecInputType.EAGLE_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        # 返回 (num_tokens_per_req, num_tokens_for_logprob_per_req)（CUDA graph padding 用）
        return self.num_tokens_per_req, self.num_tokens_for_logprob_per_req

    def prepare_for_extend(self, batch: ScheduleBatch):
        # 草稿 extend 准备：将 input_ids 调整为 "左移 + verified_id 追加" 的形式
        # EAGLE 草稿模型的输入是：[token_1, ..., token_{n-1}, verified_id]

        if batch.forward_mode.is_idle():
            return

        # Prefill only generate 1 token.
        assert len(self.verified_id) == len(batch.seq_lens)

        pt = 0
        for i, extend_len in enumerate(batch.extend_lens):
            # 将当前请求的 input_ids 左移一位，末尾追加 verified_id（滑动窗口）
            input_ids = batch.input_ids[pt : pt + extend_len]
            batch.input_ids[pt : pt + extend_len] = torch.cat(
                (input_ids[1:], self.verified_id[i].reshape(1))
            )
            pt += extend_len

    @classmethod
    def create_idle_input(
        cls,
        device: torch.device,
        hidden_size: int,
        dtype: torch.dtype,
        topk: int,
        capture_hidden_mode: CaptureHiddenMode,
    ):
        # 创建空的草稿输入（用于 IDLE 批次）
        return cls(
            verified_id=torch.empty((0,), device=device, dtype=torch.int32),
            hidden_states=torch.empty((0, hidden_size), device=device, dtype=dtype),
            topk_p=torch.empty((0, topk), device=device, dtype=torch.float32),
            topk_index=torch.empty((0, topk), device=device, dtype=torch.int64),
            capture_hidden_mode=capture_hidden_mode,
            new_seq_lens=torch.empty((0,), device=device, dtype=torch.int32),
            num_accepted_drafts=torch.empty((0,), device=device, dtype=torch.int32),
            num_accepted_tokens=torch.empty((0,), device=device, dtype=torch.int32),
            num_accepted_drafts_cpu=[],
            num_accepted_tokens_cpu=[],
        )

    def prepare_extend_after_decode(
        self,
        batch: ScheduleBatch,
        speculative_num_steps: int,
    ):
        # 验证完成后，准备草稿 extend：设置 batch 的序列信息为验证后的新状态
        # 草稿 extend 相当于对所有接受的 token 做一次前向（填充草稿模型 KV cache）

        if batch.forward_mode.is_idle():
            return

        # input_ids = verified_id（被接受的所有 token，包含 bonus）
        batch.input_ids = self.verified_id
        batch.extend_lens = batch.spec_info.num_accepted_tokens_cpu
        batch.extend_num_tokens = sum(batch.extend_lens)
        batch.seq_lens = batch.spec_info.seq_lens_for_draft_extend
        batch.seq_lens_cpu = batch.spec_info.seq_lens_for_draft_extend_cpu
        batch.req_pool_indices = batch.spec_info.req_pool_indices_for_draft_extend
        batch.return_logprob = False
        batch.return_hidden_states = False

        # 只需要捕获最后一个位置的 hidden state（草稿下一步的起点）
        self.capture_hidden_mode = CaptureHiddenMode.LAST
        self.positions = torch.empty_like(batch.input_ids, dtype=torch.long)
        self.verified_id = torch.empty_like(self.num_accepted_tokens, dtype=torch.int32)

        # 计算 extend 的 RoPE 位置和对应的 verified_id（Triton kernel 实现）
        create_extend_after_decode_spec_info[(len(batch.seq_lens),)](
            batch.input_ids,
            batch.seq_lens,
            self.num_accepted_tokens,
            self.positions,
            self.verified_id,
            next_power_of_2(max(speculative_num_steps + 1, len(batch.seq_lens))),
        )

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        # 草稿 extend 的注意力参数：每请求的 QO 数量等于 num_accepted_tokens（变长）
        device = req_pool_indices.device
        bs = self.num_accepted_drafts.numel()
        # qo_indptr：各请求查询 token 的起始指针（步长 = num_accepted_tokens[i]）
        qo_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        qo_indptr[1:] = torch.cumsum(self.num_accepted_tokens, dim=0)
        # cum_kv_seq_len：各请求 KV 序列的累积长度（前缀 KV 数量）
        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        if paged_kernel_lens_sum is None:
            paged_kernel_lens_sum = cum_kv_seq_len[-1]

        kv_indices = torch.empty(
            paged_kernel_lens_sum, dtype=torch.int32, device=device
        )

        # 构建 KV indices（物理页索引）
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        return kv_indices, cum_kv_seq_len, qo_indptr, None

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        # 过滤批次：保留 new_indices 指定的请求（用于完成请求后的批次缩减）
        if self.future_indices is not None:
            # V2 重叠模式：只更新 future_indices（延迟加载 KV 索引）
            self.future_indices.indices = self.future_indices.indices[new_indices]
            return

        strict_check = envs.SGLANG_SPEC_ENABLE_STRICT_FILTER_CHECK.get()
        if has_been_filtered:
            # in eagle_utils.py:verify, we have already filtered the batch by `unfinished_index`
            # therefore, we don't need to filter the batch again in scheduler
            # 已在 verify 中通过 unfinished_index 过滤，此处仅截断（不再索引过滤）
            error_msg = f"length of new_indices: {len(new_indices)} != length of topk_p: {len(self.topk_p)}, this should not happen"
            if len(new_indices) != len(self.topk_p):
                if strict_check:
                    raise ValueError(error_msg)
                else:
                    logger.warning(error_msg)

            self.topk_p = self.topk_p[: len(new_indices)]
            self.topk_index = self.topk_index[: len(new_indices)]
            self.hidden_states = self.hidden_states[: len(new_indices)]
            self.verified_id = self.verified_id[: len(new_indices)]
        else:
            # in some cases(e.g draft_extend), we have not filtered the batch by `unfinished_index`
            # 未提前过滤时（如 draft_extend 场景），按 new_indices 索引
            self.topk_p = self.topk_p[new_indices]
            self.topk_index = self.topk_index[new_indices]
            self.hidden_states = self.hidden_states[new_indices]
            self.verified_id = self.verified_id[new_indices]

    def merge_batch(self, spec_info: "EagleDraftInput"):
        # 合并两个 EagleDraftInput（用于连续批次调度：将新到来的请求合并到当前批次）
        if self.future_indices is not None:
            assert spec_info.future_indices is not None
            # V2 重叠模式：拼接 future_indices
            self.future_indices = FutureIndices(
                indices=torch.cat(
                    [self.future_indices.indices, spec_info.future_indices.indices]
                )
            )
            return

        if self.hidden_states is None:
            # self 为空批次：直接使用 spec_info 的数据
            self.hidden_states = spec_info.hidden_states
            self.verified_id = spec_info.verified_id
            self.topk_p = spec_info.topk_p
            self.topk_index = spec_info.topk_index
            return
        if spec_info.hidden_states is None:
            # 新请求为空批次：无需合并
            return
        # 拼接两个批次的核心张量（cat 沿 batch 维度）
        self.hidden_states = torch.cat(
            [self.hidden_states, spec_info.hidden_states], axis=0
        )
        self.verified_id = torch.cat([self.verified_id, spec_info.verified_id], axis=0)
        self.topk_p = torch.cat([self.topk_p, spec_info.topk_p])
        self.topk_index = torch.cat([self.topk_index, spec_info.topk_index])


@dataclass
class EagleVerifyOutput:
    # Draft input batch
    # 下一轮草稿生成的输入（含 hidden_states / verified_id / num_accepted 等）
    draft_input: EagleDraftInput
    # Logit outputs from target worker
    # 目标模型的 logits 输出（next_token_logits 已被原地更新为接受 token 对应的值）
    logits_output: LogitsProcessorOutput
    # Accepted token ids including the bonus token
    # 验证后最终接受的 token IDs（包含目标模型强制生成的 bonus token）
    verified_id: torch.Tensor
    # Accepted token length per sequence in a batch in CPU.
    # CPU 端每个请求接受的草稿 token 数（不含 bonus），供调度器统计
    num_accepted_drafts_per_req_cpu: List[int]
    # Accepted indices from logits_output.next_token_logits
    # 接受 token 在 logits_output 中的索引（用于 logprob 计算）
    accepted_indices: torch.Tensor
