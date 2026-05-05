from __future__ import annotations

import copy
import logging
from typing import Optional, Tuple

import torch
import triton

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.server_args import get_global_server_args

logger = logging.getLogger(__name__)

from dataclasses import dataclass

import torch.nn.functional as F

# 环境变量（如 SGLANG_NGRAM_FORCE_GREEDY_VERIFY）
from sglang.srt.environ import envs
# FlashInfer KV 索引构建工具，用于分页 KV 缓存地址映射
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
# 自定义 logit 处理器（结构化输出约束等）
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import (
    # 标志：树形投机采样 CUDA kernel 是否可用（AMD/HIP 可能不支持）
    TREE_SPEC_KERNEL_AVAILABLE,
    # 将 KV 缓存位置写入 req_to_token_pool 的 Triton kernel
    assign_req_to_token_pool,
    # 分页模式下：将接受 token 的 loc 迁移到连续位置，返回需释放的 slot 数
    get_src_tgt_cache_loc,
    # 分页模式下：实际执行 loc 重排和释放
    get_target_cache_loc,
)
from sglang.srt.utils import is_cuda, is_hip, is_musa, next_power_of_2

# CUDA/MUSA 平台：加载完整的采样和验证 kernel
if is_cuda() or is_musa():
    from sgl_kernel import (
        top_k_renorm_prob,                         # top-k 概率归一化
        top_p_renorm_prob,                         # top-p 概率归一化
        tree_speculative_sampling_target_only,     # 树形投机采样（仅目标概率）
        verify_tree_greedy,                        # 贪心树验证 kernel
    )
elif is_hip():
    # AMD/HIP 平台：仅支持贪心验证，不支持采样验证
    from sgl_kernel import verify_tree_greedy


@dataclass
class NgramVerifyInput(SpecInput):
    def __init__(
        self,
        draft_token: torch.Tensor,       # 展平的草稿 token 序列 [bs * draft_token_num]
        tree_mask: torch.Tensor,         # 树形注意力掩码（FULL_MASK 时含历史前缀）
        positions: torch.Tensor,         # 每个草稿 token 的绝对位置编码
        retrieve_index: torch.Tensor,    # 树遍历：每个请求的接受路径索引
        retrieve_next_token: torch.Tensor,  # 树遍历：每个草稿 token 的下一个 token 位置
        retrieve_next_sibling: torch.Tensor,  # 树遍历：每个草稿 token 的兄弟节点位置
        draft_token_num: int,            # 每个请求的草稿 token 数量
        grammar: BaseGrammarObject = None,  # 结构化输出语法约束对象（可选）
    ):
        # 初始化父类，标记为 NGRAM 验证输入
        super().__init__(SpecInputType.NGRAM_VERIFY)
        self.draft_token = draft_token
        # custom_mask 即树形注意力掩码（与 DFlash/EAGLE 接口兼容）
        self.custom_mask = tree_mask
        self.positions = positions
        self.retrieve_index = retrieve_index
        self.retrieve_next_token = retrieve_next_token
        self.retrieve_next_sibling = retrieve_next_sibling
        self.draft_token_num = draft_token_num
        self.device = self.custom_mask.device
        self.grammar = grammar

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        # NGRAM 验证：每个序列位置对应 draft_token_num 个 token
        return self.draft_token_num, self.draft_token_num

    def prepare_for_verify(self, batch: ScheduleBatch, page_size: int):
        # idle 模式下无需准备
        if batch.forward_mode.is_idle():
            return

        # 将草稿 token 设置为本轮验证的输入
        batch.input_ids = self.draft_token

        if page_size == 1:
            # 非分页模式：直接分配扁平 token slot
            batch.out_cache_loc = alloc_token_slots(
                batch.tree_cache,
                len(batch.input_ids),
            )
            end_offset = batch.seq_lens + self.draft_token_num
        else:
            # TODO(lsyin): add prefix lens cpu here to support page size > 1
            # 分页模式：以页粒度扩展分配 draft_token_num 个新 slot
            prefix_lens = batch.seq_lens
            prefix_lens_cpu = batch.seq_lens_cpu
            end_offset = prefix_lens + self.draft_token_num
            end_offset_cpu = prefix_lens_cpu + self.draft_token_num
            # 获取每个请求当前最后一个已分配 token 的物理位置
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
        # 将新分配的 KV slot 写入 req_to_token_pool 的页表
        assign_req_to_token_pool[(bs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            triton.next_power_of_2(bs),
        )

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        bs = len(req_pool_indices)

        # cum_kv_seq_len: 每个请求 KV 序列长度（前缀 + draft_token_num）的累积和
        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=self.device)

        # 将草稿 token 数加入每个请求的 KV 长度
        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        # qo_indptr: 每个请求在展平 Q 序列中的起始偏移，步长为 draft_token_num
        self.qo_indptr = (
            torch.arange(0, bs + 1, dtype=torch.int32, device=self.device)
            * self.draft_token_num
        )

        # kv_indices: 存储每个请求所有 KV token 在 token pool 中的物理地址
        kv_indices = torch.empty(
            cum_kv_seq_len[-1], dtype=torch.int32, device=self.device
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
        return kv_indices, cum_kv_seq_len, self.qo_indptr, self.custom_mask

    def _fill_requests(
        self,
        batch: ScheduleBatch,
        logits_output: torch.Tensor,
    ):
        # 将 accepted_indices 和 predict 转到 CPU，避免后续循环中的 D2H 同步
        accept_index_cpu = self.accepted_indices.tolist()
        predict_cpu = self.predict.tolist()
        has_finished = False
        think_end_id = batch.model_config.think_end_id

        # 遍历每个接受的 token，更新请求的 output_ids 并检查终止条件
        # 注意：必须在释放 KV 缓存 slot 之前执行，避免 use-after-free
        for i, (req, accept_index_row) in enumerate(zip(batch.reqs, accept_index_cpu)):
            for j, idx in enumerate(accept_index_row):
                # idx == -1 表示该位置无接受 token（路径到达叶节点）
                if idx == -1:
                    break
                id = predict_cpu[idx]
                req.output_ids.append(id)
                # 如果请求需要 reasoning 模式，更新 reasoning token 计数
                if req.require_reasoning and think_end_id is not None:
                    req.update_reasoning_tokens(id, think_end_id)
                req.check_finished()
                if req.finished():
                    has_finished = True
                    # 请求已完成：将当前位置之后的所有接受索引置 -1（不再提交）
                    self.accepted_indices[i, j + 1 :] = -1
                    break
                else:
                    # 结构化输出：通知 grammar 状态机接受当前 token
                    if req.grammar is not None:
                        try:
                            req.grammar.accept_token(id)
                        except ValueError as e:
                            logger.info(
                                f"{i=}, {req=}\n"
                                f"{self.accepted_indices=}\n"
                                f"{self.predict=}\n"
                            )
                            raise e
            req.spec_verify_ct += 1
            # 接受的草稿 token 数 = 非 -1 索引数 - 1（扣除 bonus token）
            accepted_draft_tokens = sum(1 for idx in accept_index_row if idx != -1) - 1
            req.spec_accepted_drafts += accepted_draft_tokens
            req.update_spec_acceptance_histogram(accepted_draft_tokens)

        if has_finished:
            # 有请求提前终止：重新计算每个请求的有效接受 token 数
            self.num_accepted_drafts = (self.accepted_indices != -1).sum(dim=1) - 1
        # 压缩 accepted_indices：去除 -1，得到有效 token 的全局展平索引
        self.accepted_indices = self.accepted_indices[self.accepted_indices != -1]

        # 只保留接受 token 对应的 logits（供后续 logprob 计算）
        logits_output.next_token_logits = logits_output.next_token_logits[
            self.accepted_indices
        ]
        if logits_output.hidden_states:
            logits_output.hidden_states = logits_output.hidden_states[
                self.accepted_indices
            ]
        # verified_id: 每个接受位置对应的真实预测 token ID
        self.verified_id = self.predict[self.accepted_indices]

    def _free_cache(
        self,
        batch: ScheduleBatch,
        page_size: int,
        num_accepted_drafts_cpu: torch.Tensor,
    ):
        bs = batch.batch_size()
        # 释放未被接受的 token 对应的 KV 缓存 slot
        if page_size == 1:
            # TODO: boolean array index leads to a device sync. Remove it.
            # 非分页模式：精确按 accepted_indices 掩码释放多余 slot
            evict_mask = torch.full_like(self.draft_token, True, dtype=torch.bool)
            evict_mask[self.accepted_indices] = False
            batch.token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
            batch.out_cache_loc = batch.out_cache_loc[self.accepted_indices]
        else:
            # 分页模式：只能以页粒度释放；需要将接受 token 迁移到连续位置，释放尾部
            # out_cache_loc: [0  1  2,  3  4  5,  6  7  8]
            # accept_index:  [0 -1  2,  3  4 -1,  6 -1 -1]
            # tgt_cache_loc: [0  1   ,  3  4   ,  6      ]
            # to_free_slots: [      2,        5,     7  8]
            # to_free_slots also needs to be page-aligned without the first partial page
            #
            # 将每行 out_cache_loc 分为两部分：
            # 1. 前 num_accepted_drafts[i]+1 个 slot 移到 tgt_cache_loc
            # 2. 剩余 slot 收集到 to_free_slots（最终释放）
            src_cache_loc, tgt_cache_loc, to_free_num_slots = get_src_tgt_cache_loc(
                batch.seq_lens,
                batch.out_cache_loc,
                self.accepted_indices,
                self.num_accepted_drafts,
                self.draft_token_num,
                page_size,
            )
            to_free_slots = torch.empty(
                (to_free_num_slots.sum().item(),),
                dtype=torch.int64,
                device=to_free_num_slots.device,
            )

            # Triton kernel：执行实际的 loc 重排，输出 tgt_cache_loc 和 to_free_slots
            get_target_cache_loc[(bs,)](
                tgt_cache_loc,
                to_free_slots,
                self.num_accepted_drafts,
                to_free_num_slots,
                batch.out_cache_loc,
                self.draft_token_num,
                next_power_of_2(self.draft_token_num),
                next_power_of_2(bs),
            )

            # 释放已整理出的多余 KV 缓存 slot
            batch.token_to_kv_pool_allocator.free(to_free_slots)

            # 将接受 token 的 KV 数据从 src_cache_loc 搬移到 tgt_cache_loc（紧凑化）
            batch.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
                tgt_cache_loc, src_cache_loc
            )
            batch.out_cache_loc = tgt_cache_loc

        # 更新每个请求的 KV 缓存账本（提交长度 += 本轮接受的 token 数 + 1）
        num_accepted_drafts_list = num_accepted_drafts_cpu.tolist()
        for i, req in enumerate(batch.reqs):
            req.kv_committed_len += num_accepted_drafts_list[i] + 1
            req.kv_allocated_len = req.kv_committed_len

        # 将新提交的 token 写入 req_to_token_pool 的页表映射
        assign_req_to_token_pool[(bs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + self.num_accepted_tokens,
            batch.out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            triton.next_power_of_2(bs),
        )

    def _greedy_verify(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
    ):
        bs = batch.batch_size()
        # 贪心验证：对目标模型输出取 argmax，得到目标预测 token
        target_predict = torch.argmax(logits_output.next_token_logits, dim=-1)
        target_predict = target_predict.reshape(bs, self.draft_token_num)

        # candidates: [bs, draft_token_num]，草稿 token 矩阵
        candidates = self.draft_token.reshape(bs, self.draft_token_num)
        # predict 形状：[total_tokens + bs]，保存每个位置的最终预测（含 bonus token）
        predict_shape = list(logits_output.next_token_logits.shape)[:-1]
        predict_shape[-1] += 1
        self.predict = torch.empty(predict_shape, dtype=torch.int32, device=self.device)
        # accepted_indices: [bs, draft_token_num]，-1 表示该位置未被接受
        self.accepted_indices = torch.full(
            (bs, self.draft_token_num), -1, dtype=torch.int32, device=self.device
        )
        # num_accepted_drafts: [bs]，每个请求接受的草稿 token 数（不含 bonus）
        self.num_accepted_drafts = torch.empty(
            (bs,), dtype=torch.int32, device=self.device
        )

        # 调用 CUDA/HIP/MUSA kernel 执行贪心树验证，就地写入 predict/accept_index/accept_token_num
        verify_tree_greedy(
            predicts=self.predict,  # mutable
            accept_index=self.accepted_indices,  # mutable
            accept_token_num=self.num_accepted_drafts,  # mutable
            candidates=candidates,
            # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
            retrive_index=self.retrieve_index,
            retrive_next_token=self.retrieve_next_token,
            retrive_next_sibling=self.retrieve_next_sibling,
            target_predict=target_predict,
        )

    def _sampling_verify(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
    ):
        bs = batch.batch_size()
        # candidates: [bs, draft_token_num]，草稿 token 矩阵
        candidates = self.draft_token.reshape(bs, self.draft_token_num)
        predict_shape = list(logits_output.next_token_logits.shape)[:-1]
        predict_shape[-1] += 1
        self.predict = torch.empty(predict_shape, dtype=torch.int32, device=self.device)
        self.accepted_indices = torch.full(
            (bs, self.draft_token_num), -1, dtype=torch.int32, device=self.device
        )
        self.num_accepted_drafts = torch.empty(
            (bs,), dtype=torch.int32, device=self.device
        )
        # 将 temperature 展开到每个草稿 token 对应的位置：[bs * draft_token_num, 1]
        expanded_temperature = torch.repeat_interleave(
            sampling_info.temperatures, self.draft_token_num, dim=0
        )  # (bs * draft_token_num, 1)

        # 对目标模型 logits 应用 temperature，计算归一化概率分布
        target_probs = F.softmax(
            logits_output.next_token_logits / expanded_temperature, dim=-1
        )  # (bs * draft_token_num, vocab_size)

        # NOTE: The test shows that top_p_renorm_prob and top_k_renorm_prob are the key factors
        # contributing to the poor performance of _sampling_verify.
        # 应用 top-k 截断并重归一化概率
        target_probs = top_k_renorm_prob(
            target_probs,
            torch.repeat_interleave(sampling_info.top_ks, self.draft_token_num, dim=0),
        )  # (bs * draft_token_num, vocab_size)

        if sampling_info.need_top_p_sampling:
            # logger.info("Using top-p sampling in speculative decoding verification.")
            # 需要 top-p 时：进一步按 nucleus 概率截断并重归一化
            target_probs = top_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ps, self.draft_token_num, dim=0
                ),
            )

        # 将概率重塑为 [bs, draft_token_num, vocab_size] 供 kernel 使用
        target_probs = target_probs.reshape(bs, self.draft_token_num, -1)
        # draft_probs 全零（target_only 模式下不使用草稿模型概率）
        draft_probs = torch.zeros(
            target_probs.shape, dtype=torch.float32, device=self.device
        )

        # 为拒绝采样生成均匀随机数：每个草稿 token 位置一个
        coins = torch.rand_like(candidates, dtype=torch.float32, device=self.device)
        # 为最终采样（bonus token）生成均匀随机数：每个请求一个
        coins_for_final_sampling = torch.rand(
            (bs,), dtype=torch.float32, device=self.device
        )
        # 调用 CUDA kernel 执行树形投机采样验证（仅用目标概率，无草稿模型概率）
        tree_speculative_sampling_target_only(
            predicts=self.predict,  # mutable
            accept_index=self.accepted_indices,  # mutable
            accept_token_num=self.num_accepted_drafts,  # mutable
            candidates=candidates.to(torch.int64),
            # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
            retrive_index=self.retrieve_index.to(torch.int64),
            retrive_next_token=self.retrieve_next_token.to(torch.int64),
            retrive_next_sibling=self.retrieve_next_sibling.to(torch.int64),
            uniform_samples=coins,
            uniform_samples_for_final_sampling=coins_for_final_sampling,
            target_probs=target_probs,
            draft_probs=draft_probs,
            # 接受阈值：单 token 接受概率阈值和累积接受概率阈值
            threshold_single=get_global_server_args().speculative_accept_threshold_single,
            threshold_acc=get_global_server_args().speculative_accept_threshold_acc,
            deterministic=True,
        )

    def verify(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        page_size: int,
        vocab_mask: Optional[torch.Tensor] = None,  # For grammar
    ) -> torch.Tensor:
        # retrieve_index 的第 0 维为实际参与验证的请求数
        bs = self.retrieve_index.shape[0]
        sampling_info = batch.sampling_info

        if bs != len(sampling_info):
            # 若 batch size 与 sampling_info 不一致（有请求被过滤），需要深拷贝并过滤
            sampling_info = copy.deepcopy(sampling_info)
            # NOTE: retrieve_index are the indices of the requests that are kept.
            sampling_info.filter_batch(
                self.retrieve_index.tolist(), self.retrieve_index
            )

        # 应用自定义 logit 处理器（结构化输出约束等）
        if sampling_info.has_custom_logit_processor:
            apply_custom_logit_processor(
                logits_output.next_token_logits,
                sampling_info,
                num_tokens_in_batch=self.draft_token_num,
            )

        # 应用惩罚项（repetition penalty、frequency penalty 等）
        if (
            sampling_info.penalizer_orchestrator.is_required
            or sampling_info.logit_bias is not None
        ):
            # 投机解码的惩罚是宽松版本（relaxed），重复使用同一惩罚向量
            sampling_info.penalizer_orchestrator.apply(
                logits_output.next_token_logits, repeat=self.draft_token_num
            )
            if sampling_info.logit_bias is not None:
                logits_output.next_token_logits.add_(
                    torch.repeat_interleave(
                        sampling_info.logit_bias, self.draft_token_num, dim=0
                    )
                )

        # 应用 grammar 词表掩码（结构化输出）
        if vocab_mask is not None:
            assert self.grammar is not None
            self.grammar.apply_vocab_mask(
                logits=logits_output.next_token_logits, vocab_mask=vocab_mask
            )

        # 根据采样策略选择贪心验证或采样验证
        # AMD/HIP 平台强制使用贪心验证（采样 kernel 不可用）
        is_all_greedy = (
            sampling_info.is_all_greedy or envs.SGLANG_NGRAM_FORCE_GREEDY_VERIFY.get()
        )
        if (not is_all_greedy) and (not TREE_SPEC_KERNEL_AVAILABLE):
            logger.warning(
                "Tree speculative sampling kernel unavailable (likely AMD/HIP build). "
                "Falling back to greedy verification."
            )

        if is_all_greedy or not TREE_SPEC_KERNEL_AVAILABLE:
            # 贪心路径：argmax 比对草稿与目标预测
            self._greedy_verify(batch, logits_output)
        else:
            # NOTE: Compared with greedy_verify, the performance of _sampling_verify is relatively poor.
            # 采样路径：基于概率分布的拒绝采样验证（性能略低于贪心）
            self._sampling_verify(batch, logits_output, sampling_info)

        # 更新 output_ids、grammar 状态，压缩 accepted_indices
        self._fill_requests(batch, logits_output)

        # num_accepted_tokens = num_accepted_drafts + 1（含 bonus token）
        # 必须在 _fill_requests 完成后（num_accepted_drafts 已最终确定）计算
        self.num_accepted_tokens = self.num_accepted_drafts + 1

        # 转到 CPU 供后续账本更新和统计使用
        num_accepted_drafts_cpu = self.num_accepted_drafts.cpu()
        num_accepted_tokens_cpu = num_accepted_drafts_cpu + 1
        num_accepted_drafts = num_accepted_drafts_cpu.sum().item()

        # 释放未接受 token 的 KV 缓存 slot，更新 out_cache_loc 和 req_to_token_pool
        self._free_cache(batch, page_size, num_accepted_drafts_cpu)

        # 更新 batch 的序列长度统计
        batch.seq_lens.add_(self.num_accepted_tokens)
        batch.seq_lens_cpu.add_(num_accepted_tokens_cpu)

        return logits_output, self.verified_id, num_accepted_drafts

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        # NGRAM 验证输入不支持 filter_batch（每步重新生成）
        pass

    def merge_batch(self, spec_info: NgramVerifyInput):
        # NGRAM 验证输入不支持 merge_batch
        pass
