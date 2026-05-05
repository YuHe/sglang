# decode 侧调度批次 Mixin：为 PD 分离场景中 decode 节点的 ScheduleBatch 添加 prebuilt 处理能力
# prebuilt 模式：prefill 节点已完成 KV 缓存传输，decode 节点直接接收预构建的 KV 状态
from __future__ import annotations

import logging
from http import HTTPStatus
from typing import TYPE_CHECKING

import torch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.server_args import ServerArgs


# decode 侧 ScheduleBatch 的 PD 分离扩展 Mixin，提供 prebuilt 模式的准备和处理方法
class ScheduleBatchDisaggregationDecodeMixin:

    def prepare_for_prebuilt(self: ScheduleBatch):
        """
        Prepare a prebuilt extend by populate metadata
        Adapted from .prepare_for_extend().
        """
        # 将 forward 模式设置为 PREBUILT，表示 KV 缓存已由 prefill 端预构建并传输完毕
        self.forward_mode = ForwardMode.PREBUILT
        reqs = self.reqs
        # 取每个请求中前缀之后的新增 token id
        input_ids = [r.fill_ids[len(r.prefix_indices) :] for r in reqs]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = []
        pre_lens = []
        req_pool_indices = []

        # Pre-calculate total size
        # 预先计算所有请求的 extend_input_len 之和，分配输出缓存位置张量
        total_size = sum(req.extend_input_len for req in reqs)
        out_cache_loc = torch.empty(total_size, dtype=torch.int64, device=self.device)

        # Fill the tensor in one pass
        # 一次遍历填充 out_cache_loc，避免多次 cat 操作
        offset = 0
        for i, req in enumerate(reqs):
            req_pool_indices.append(req.req_pool_idx)

            # 取请求在 req_to_token_pool 中对应的 token 位置（extend 长度范围内）
            chunk = self.req_to_token_pool.req_to_token[req.req_pool_idx][
                : req.extend_input_len
            ]
            assert (
                offset + req.extend_input_len <= total_size
            ), f"Exceeds total size: offset={offset}, req.extend_input_len={req.extend_input_len}, total_size={total_size}"
            out_cache_loc[offset : offset + req.extend_input_len] = chunk
            offset += req.extend_input_len

            pre_len = len(req.prefix_indices)
            # seq_len：原始输入长度 + 已生成 token 数（不含最后一个，避免重复计算）
            seq_len = len(req.origin_input_ids) + max(0, len(req.output_ids) - 1)
            seq_lens.append(seq_len)
            if len(req.output_ids) == 0:
                assert (
                    seq_len - pre_len == req.extend_input_len
                ), f"seq_len={seq_len}, pre_len={pre_len}, req.extend_input_len={req.extend_input_len}"

            # 更新已缓存 token 数和已计算位置（对非 retracted 请求）
            if not req.retracted_stain:
                req.cached_tokens += pre_len - req.already_computed
                req.already_computed = seq_len
            req.is_retracted = False
            pre_lens.append(pre_len)
            req.extend_logprob_start_len = 0

        extend_input_logprob_token_ids = None

        # Set fields
        # 构建批次所需的各类张量字段
        self.input_ids = torch.tensor(
            sum(input_ids, []), dtype=torch.int32, device=self.device
        )
        self.req_pool_indices = torch.tensor(
            req_pool_indices, dtype=torch.int64, device=self.device
        )
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=self.device)
        self.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        self.orig_seq_lens = torch.tensor(
            seq_lens, dtype=torch.int32, device=self.device
        )
        self.out_cache_loc = out_cache_loc
        self.seq_lens_sum = sum(seq_lens)

        # 若需要返回 logprob，收集各请求的 top logprob 配置
        if self.return_logprob:
            self.top_logprobs_nums = [r.top_logprobs_num for r in reqs]
            self.token_ids_logprobs = [r.token_ids_logprob for r in reqs]

        self.extend_num_tokens = extend_num_tokens
        self.prefix_lens = [len(r.prefix_indices) for r in reqs]
        self.extend_lens = [r.extend_input_len for r in reqs]
        self.extend_logprob_start_lens = [r.extend_logprob_start_len for r in reqs]
        self.extend_input_logprob_token_ids = extend_input_logprob_token_ids
        self.multimodal_inputs = [r.multimodal_inputs for r in reqs]

        # Build sampling info
        # 构建采样信息（温度/top-p 等参数），供后续 decode forward 使用
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    def process_prebuilt(
        self: ScheduleBatch,
        server_args: ServerArgs,
        future_map: FutureMap,
    ):
        """Assign the buffered last input id to schedule batch"""
        # 将每个请求最后一个 output token 收集为批次输出，并更新缓存树
        self.output_ids = []
        for req in self.reqs:
            self.output_ids.append(req.output_ids[-1])
            # 将未完成请求的已生成部分写入 tree_cache
            self.tree_cache.cache_unfinished_req(req)
            if req.grammar is not None:
                # FIXME: this try-except block is for handling unexpected xgrammar issue.
                try:
                    # if it is not None, then the grammar is from a retracted request, and we should not
                    # accept the token as it's already accepted
                    # 非 retracted 请求需调用 grammar.accept_token 推进语法状态机
                    if req.grammar.current_token is None:
                        req.grammar.accept_token(req.output_ids[-1])
                except ValueError as e:
                    from sglang.srt.managers.schedule_batch import FINISH_ABORT

                    # Grammar accept_token can raise ValueError if the token is not in the grammar.
                    # This can happen if the grammar is not set correctly or the token is invalid.
                    # Use to_finish (not finished_reason) so that process_batch_result_prebuilt
                    # handles the release via check_finished -> release_kv_cache in one place.
                    # 语法 accept 失败，标记请求为 ABORT，由后续 process_batch_result_prebuilt 统一释放 KV
                    error_message = f"Grammar accept_token failed for req {req.rid} with token {req.output_ids[-1]}: {e}"
                    req.to_finish = FINISH_ABORT(
                        error_message, HTTPStatus.INTERNAL_SERVER_ERROR
                    )
                req.grammar.finished = req.finished()
        self.output_ids = torch.tensor(self.output_ids, device=self.device)

        # Simulate the eagle run.
        # EAGLE 推测解码模式：在 prebuilt 场景下模拟 eagle draft 阶段
        if self.spec_algorithm.is_eagle():
            num_states = server_args.speculative_eagle_topk
            # 多层 EAGLE 时，状态数按步数倍增
            if server_args.enable_multi_layer_eagle:
                num_states *= server_args.speculative_num_steps
            # 收集每个请求的 topk 概率和索引
            topk_p = torch.stack(
                [
                    torch.as_tensor(
                        req.output_topk_p[:num_states],
                        device=self.device,
                        dtype=torch.float32,
                    )
                    for req in self.reqs
                ],
                dim=0,
            )
            topk_index = torch.stack(
                [
                    torch.as_tensor(
                        req.output_topk_index[:num_states],
                        device=self.device,
                        dtype=torch.int64,
                    )
                    for req in self.reqs
                ],
                dim=0,
            )

            # 收集每个请求在 prefill 端计算得到的 hidden states
            hidden_states_list = [req.hidden_states_tensor for req in self.reqs]
            hidden_states = torch.stack(hidden_states_list, dim=0).to(self.device)

            # local import to avoid circular import
            from sglang.srt.speculative.eagle_info import EagleDraftInput

            # 构建 EagleDraftInput，准备 decode 侧的推测 draft 前向计算
            spec_info = EagleDraftInput(
                topk_p=topk_p,
                topk_index=topk_index,
                hidden_states=hidden_states,
                verified_id=self.output_ids,
                new_seq_lens=self.seq_lens,
            )
            spec_info.prepare_for_extend(self)
            # 仅捕获最后一个 hidden state，用于下一步推测
            spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
            if self.enable_overlap:
                # 开启 overlap 时，预分配 future_indices 以支持异步流水
                spec_info.future_indices = future_map.alloc_future_indices(
                    len(self.seq_lens)
                )
                future_map.store_to_map_for_new_batch(
                    spec_info.future_indices, spec_info
                )
            self.spec_info = spec_info
