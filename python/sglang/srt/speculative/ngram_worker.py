import logging
from typing import List, Optional

import numpy as np
import torch
# 从 sgl_kernel 导入高效的树掩码索引重建内核
from sgl_kernel.speculative import reconstruct_indices_from_tree_mask

from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v1
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardMode
# 用于记录 batch 中各请求的时间统计
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.observability.trace import get_global_tracing_enabled
from sglang.srt.server_args import ServerArgs
# NgramCorpus: N-gram 语料库，基于后缀自动机（SAM）进行 token 匹配
from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import generate_token_bitmask

logger = logging.getLogger(__name__)


# 是否使用完整掩码（FULL_MASK）：覆盖历史 token + 草稿 token 的完整注意力关系
# 设为 True 则精度更高但显存占用更大；QLEN_MASK 更快但需对应后端支持
USE_FULL_MASK = True


# NGRAMWorker: 基于 N-gram 匹配的投机解码 Worker
# 不使用独立草稿模型，而是通过历史 token 的 N-gram 匹配生成草稿 token
class NGRAMWorker:
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # 持有目标模型 Worker 的引用（用于验证阶段）
        self.target_worker = target_worker
        # 复用目标模型的 model_runner（N-gram 不需要独立草稿模型）
        self.model_runner = target_worker.model_runner
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size
        # 每次生成的草稿 token 数量
        self.draft_token_num: int = server_args.speculative_num_draft_tokens
        # N-gram 匹配的最大 trie 树深度（查找窗口长度）
        self.max_trie_depth: int = server_args.speculative_ngram_max_trie_depth

        self.max_batch_size = target_worker.max_running_requests
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"

        # 预分配推理所需的各种张量缓冲区
        self._init_preallocated_tensors()

        # 初始化 N-gram 语料库（内部使用 C++ 后缀自动机实现高效匹配）
        self.ngram_corpus = NgramCorpus(
            min_bfs_breadth=server_args.speculative_ngram_min_bfs_breadth,
            max_bfs_breadth=server_args.speculative_ngram_max_bfs_breadth,
            match_type=server_args.speculative_ngram_match_type,
            capacity=server_args.speculative_ngram_capacity,
            max_trie_depth=server_args.speculative_ngram_max_trie_depth,
            draft_token_num=server_args.speculative_num_draft_tokens,
            external_sam_budget=server_args.speculative_ngram_external_sam_budget,
            external_corpus_max_tokens=server_args.speculative_ngram_external_corpus_max_tokens,
        )
        # 若配置了外部语料文件，启动时加载到 SAM 中
        if server_args.speculative_ngram_external_corpus_path is not None:
            from sglang.srt.speculative.cpp_ngram.external_corpus import (
                iter_external_corpus_chunks,
            )

            corpus_path = server_args.speculative_ngram_external_corpus_path
            # 分块读取外部 JSONL 语料并 tokenize
            chunks = list(
                iter_external_corpus_chunks(
                    corpus_path,
                    target_worker.tokenizer,
                    server_args.speculative_ngram_external_corpus_max_tokens,
                )
            )
            # 将 token 块加载到命名语料库中
            loaded = self.add_external_corpus(corpus_path, chunks)
            # 提交加载完成（使 SAM 生效）
            self.commit_corpus_load(corpus_path, loaded)
            logger.info(
                "Loaded external ngram corpus '%s' (%d tokens).",
                corpus_path,
                loaded,
            )

    def clear_cache_pool(self):
        # 清空 N-gram 语料库（重置 SAM，释放内存）
        self.ngram_corpus.reset()

    def add_external_corpus(self, corpus_id: str, token_chunks: list[list[int]]) -> int:
        # 将外部语料的 token 块流式写入 SAM，返回成功加载的 token 数
        return self.ngram_corpus.load_external_corpus_named(corpus_id, token_chunks)

    def commit_corpus_load(self, corpus_id: str, loaded_token_count: int) -> None:
        # 提交外部语料加载完成，使 SAM 索引生效
        self.ngram_corpus.commit_external_corpus_load(corpus_id, loaded_token_count)

    def remove_external_corpus(self, corpus_id: str) -> None:
        # 从 SAM 中移除指定外部语料
        self.ngram_corpus.remove_external_corpus(corpus_id)

    def list_external_corpora(self) -> dict[str, int]:
        # 列出所有已加载的外部语料及其 token 数量
        return self.ngram_corpus.list_external_corpora()

    def _efficient_concat_last_n(self, seq1: List[int], seq2: List[int], n: int):
        # 高效取两个序列拼接后的最后 n 个元素，避免完整拼接大序列
        seq2_len = len(seq2)
        if seq2_len >= n:
            return seq2[-n:]

        need_from_seq1 = n - seq2_len
        return seq1[-need_from_seq1:] + seq2

    def _init_preallocated_tensors(self):
        # 预分配所有推理所需张量，避免推理时反复申请显存
        max_total_drafts = self.max_batch_size * self.draft_token_num
        max_total_mask_size = (
            self.max_batch_size * self.draft_token_num * self.draft_token_num
        )

        # 草稿 token 缓冲区（展平格式：[batch * draft_token_num]）
        self.draft_tokens = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        # 树遍历索引：用于在验证后重建接受路径
        self.retrieve_indexes = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        # 每个草稿 token 的下一个兄弟 token（树遍历辅助）
        self.retrieve_next_token = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        # 每个草稿 token 的下一个兄弟节点位置（树遍历辅助）
        self.retrieve_next_sibling = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        # 草稿 token 的位置编码（相对于序列起始的绝对位置）
        self.positions = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        # 树注意力掩码（展平格式：[batch * draft^2]，bool 类型）
        self.tree_mask = torch.empty(
            (max_total_mask_size,), dtype=torch.bool, device=self.device
        )

        # 按 batch size 预切片视图，避免推理时重复 slice 操作
        self.draft_tokens_batch = []
        self.tree_mask_batch = []
        self.retrieve_indexes_batch = []
        self.retrieve_next_token_batch = []
        self.retrieve_next_sibling_batch = []
        self.positions_batch = []

        for bs in range(0, self.max_batch_size + 1):
            self.retrieve_indexes_batch.append(self.retrieve_indexes[:bs, :])
            self.retrieve_next_token_batch.append(self.retrieve_next_token[:bs, :])
            self.retrieve_next_sibling_batch.append(self.retrieve_next_sibling[:bs, :])
            self.positions_batch.append(self.positions[: bs * self.draft_token_num])
            self.draft_tokens_batch.append(
                self.draft_tokens[: bs * self.draft_token_num]
            )
            self.tree_mask_batch.append(
                self.tree_mask[: bs * self.draft_token_num * self.draft_token_num]
            )

    def _prepare_draft_tokens(
        self, batch: ScheduleBatch
    ) -> tuple[np.ndarray, np.ndarray]:
        bs = batch.batch_size()

        # 等待 SAM 上次异步操作完成，确保语料库状态一致
        self.ngram_corpus.synchronize()
        req_ids = []
        batch_tokens = []
        total_lens = []
        for req in batch.reqs:
            # 取每个请求最近 max_trie_depth 个 token 作为 N-gram 查询键
            check_token = self._efficient_concat_last_n(
                req.origin_input_ids, req.output_ids, self.max_trie_depth
            )
            req_ids.append(req.rid)
            batch_tokens.append(check_token)
            # 总序列长度 = 原始输入长度 + 已生成输出长度
            total_lens.append(len(req.origin_input_ids) + len(req.output_ids))
        # 批量从语料库中查询 N-gram 匹配的草稿 token 和对应的树掩码
        req_drafts, mask = self.ngram_corpus.batch_get(
            req_ids, batch_tokens, total_lens
        )
        total_draft_token_num = len(req_drafts)

        # Check if speculative decoding is needed; here we always enforce it
        # 验证草稿 token 数量是否符合预期（每个请求恰好 draft_token_num 个）
        assert (
            total_draft_token_num == bs * self.draft_token_num
        ), f"{total_draft_token_num=}, {bs=}, {self.draft_token_num=}"
        return req_drafts, mask

    def _prepare_for_speculative_decoding(self, batch: ScheduleBatch):
        # extend 阶段（prefill）不需要投机解码，直接返回
        if batch.forward_mode.is_extend():
            return

        bs = batch.batch_size()

        # 取对应 batch size 的预分配视图
        retrieve_index = self.retrieve_indexes_batch[bs]
        retrieve_next_token = self.retrieve_next_token_batch[bs]
        retrieve_next_sibling = self.retrieve_next_sibling_batch[bs]
        positions = self.positions_batch[bs]
        tree_mask = self.tree_mask_batch[bs]
        draft_tokens = self.draft_tokens_batch[bs]

        # 从语料库中获取草稿 token 和树掩码（numpy 格式）
        req_drafts, mask = self._prepare_draft_tokens(batch)
        # 异步将 numpy 数组拷贝到 GPU
        tree_mask.copy_(torch.from_numpy(mask), non_blocking=True)
        draft_tokens.copy_(torch.from_numpy(req_drafts), non_blocking=True)

        # 从树掩码重建遍历所需的辅助索引（就地写入可变参数）
        reconstruct_indices_from_tree_mask(
            tree_mask,
            batch.seq_lens,
            positions,  # mutable
            retrieve_index,  # mutable
            retrieve_next_token,  # mutable
            retrieve_next_sibling,  # mutable
            bs,
            self.draft_token_num,
        )

        # NOTE: QLEN_MASK is faster than FULL_MASK, but requires corresponding changes in flashinfer.
        # Testing shows about 8% performance improvement (the effect is roughly proportional to batch size).
        if USE_FULL_MASK:
            # 构建完整注意力树掩码：[draft_token_num, seq_len + draft_token_num]
            tree_mask = []
            mask = mask.reshape(
                batch.batch_size(), self.draft_token_num, self.draft_token_num
            )
            for i, req in enumerate(batch.reqs):
                seq_len = len(req.origin_input_ids) + len(req.output_ids)
                # 历史 token 部分全为 True（草稿 token 可看到所有历史）
                req_mask = torch.ones((self.draft_token_num, seq_len - 1)).cuda()
                # 拼接 N-gram 生成的树内掩码（草稿 token 间的因果关系）
                req_mask = torch.cat(
                    (req_mask, torch.from_numpy(mask[i]).cuda()), dim=1
                ).to(torch.bool)
                tree_mask.append(req_mask.flatten())
            tree_mask = torch.cat(tree_mask, dim=0)

        # 将 batch 标记为 NGRAM 投机算法并设置验证模式
        batch.spec_algorithm = SpeculativeAlgorithm.NGRAM
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = NgramVerifyInput(
            draft_tokens,
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            self.draft_token_num,
        )
        # 为验证阶段准备 KV 缓存位置等信息
        batch.spec_info.prepare_for_verify(batch, self.page_size)

    def _update_ngram_corpus(self, batch: ScheduleBatch):
        # 验证完成后将最新 token 写入语料库，更新 SAM 以便后续查询
        batch_tokens = []
        for req in batch.reqs:
            # FIXME: Whether to insert 'extend' into the cache or not, after testing,
            # there is not much difference, so we will not insert it for now.
            # if batch.forward_mode.is_extend():
            #     put_ids = req.origin_input_ids + req.output_ids
            # else:
            # 仅取最近 max_trie_depth 个 token 插入（节省 SAM 空间）
            put_ids = self._efficient_concat_last_n(
                req.origin_input_ids, req.output_ids, self.max_trie_depth
            )
            batch_tokens.append(put_ids)
        self.ngram_corpus.batch_put(batch_tokens)

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        # 记录草稿生成阶段的开始时间
        set_time_batch(batch.reqs, "set_spec_draft_start_time", trace_only=True)

        # 准备草稿 token 和树掩码，设置 batch 为验证模式
        self._prepare_for_speculative_decoding(batch)

        # 记录草稿生成阶段的结束时间
        set_time_batch(batch.reqs, "set_spec_draft_end_time", trace_only=True)

        model_worker_batch = batch.get_model_worker_batch()
        spec_info = model_worker_batch.spec_info
        num_accepted_drafts = 0
        accept_lens = None
        num_accepted_drafts_per_req_cpu = None

        if model_worker_batch.forward_mode.is_target_verify():
            # 验证模式：先在 CPU 上提取结构化输出所需的辅助数据（与 GPU 前向并行）
            if batch.has_grammar:
                retrieve_next_token_cpu = spec_info.retrieve_next_token.cpu()
                retrieve_next_sibling_cpu = spec_info.retrieve_next_sibling.cpu()
                draft_tokens_cpu = spec_info.draft_token.view(
                    spec_info.retrieve_next_token.shape
                ).cpu()

            # 记录目标模型验证的开始时间
            set_time_batch(batch.reqs, "set_spec_verify_start_time", trace_only=True)

            # 调用目标模型前向传播（验证所有草稿 token）
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch, is_verify=True
            )
            logits_output, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.can_run_cuda_graph,
            )

            verify_input: NgramVerifyInput = model_worker_batch.spec_info
            vocab_mask = None
            if batch.has_grammar:
                # Generate the logit mask for structured output.
                # Overlap the CPU operations for bitmask generation with the forward pass.
                # 生成结构化输出的词表掩码，与 GPU 前向计算重叠执行
                vocab_mask = generate_token_bitmask(
                    batch.reqs,
                    verify_input,
                    retrieve_next_token_cpu,
                    retrieve_next_sibling_cpu,
                    draft_tokens_cpu,
                    batch.sampling_info.vocab_size,
                )

                if vocab_mask is not None:
                    assert verify_input.grammar is not None
                    vocab_mask = vocab_mask.to(verify_input.retrieve_next_token.device)
                    # NOTE (sk): otherwise, this vocab mask will be the one from the previous extend stage
                    # and will be applied to produce wrong results
                    # 清空 batch 级别的掩码，避免与上次 extend 阶段的掩码混淆
                    batch.sampling_info.vocab_mask = None

            # 验证草稿 token：比对目标模型输出与草稿，确定可接受的最长前缀
            logits_output, next_token_ids, num_accepted_drafts = verify_input.verify(
                batch, logits_output, self.page_size, vocab_mask
            )
            # 将每个请求的接受数量转到 CPU，用于统计和自适应步数调整
            num_accepted_drafts_per_req_cpu = (
                verify_input.num_accepted_drafts.cpu().tolist()
            )

            if get_global_tracing_enabled():
                for idx, req in enumerate(batch.reqs):
                    accepted = (
                        verify_input.num_accepted_drafts[idx].item()
                        if verify_input.num_accepted_drafts is not None
                        else 0
                    )
                    req.time_stats.set_spec_verify_end_time(accepted_tokens=accepted)

            # Store accept_lens for per-request metrics
            # 保存每请求接受长度，用于延迟统计
            accept_lens = verify_input.num_accepted_drafts
            if batch.return_logprob:
                # 若需要返回 logprob，在接受的 token 上补全 logprob 信息
                add_output_logprobs_for_spec_v1(batch, verify_input, logits_output)
            # 将本次接受的 token 写入语料库
            self._update_ngram_corpus(batch)
            # Clean up per-request match state for finished/retracted requests.
            # State entries are created in _prepare_draft_tokens and cleaned here.
            # If a request is removed without passing through verify, the entry
            # persists until reset(); this is acceptable because MatchState is small.
            # 清理已完成或被撤回请求的匹配状态（避免内存泄漏）
            finished_req_ids = []
            for req in batch.reqs:
                if req.finished() or req.is_retracted:
                    finished_req_ids.append(req.rid)
            if finished_req_ids:
                self.ngram_corpus.erase_match_state(finished_req_ids)
            # 验证完成后将 forward_mode 恢复为 DECODE
            batch.forward_mode = ForwardMode.DECODE

        else:
            # 非验证模式（extend/decode 阶段）：直接调用目标模型前向
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch
            )
            logits_output, next_token_ids, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.next_token_ids,
                batch_result.can_run_cuda_graph,
            )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            num_accepted_drafts=num_accepted_drafts,
            num_accepted_drafts_per_req_cpu=num_accepted_drafts_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
            accept_lens=accept_lens,
        )
