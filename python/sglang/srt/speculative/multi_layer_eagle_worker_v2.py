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
# multi_layer_eagle_worker_v2.py
# EAGLE 多层 MTP（Multi-Token Prediction）V2 版本：支持多个草稿模型层同时预测多 token
# 基于 BaseDraftWorker/BaseSpecWorker 框架，支持 plan_stream 重叠执行

import contextlib
import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import speculative_moe_backend_context
from sglang.srt.layers.utils.logprob import compute_spec_v2_logprobs
from sglang.srt.managers.io_struct import (
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromIPCReqInput,
)
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseDraftWorker, BaseSpecWorker
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_info_v2 import fill_new_verified_id
from sglang.srt.speculative.eagle_utils import TreeMaskMode, build_tree_kernel_efficient
from sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner import (
    MultiLayerEagleMultiStepDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.multi_layer_eagle_utils import (
    assign_hidden_states_pool_triton,  # 将 hidden states 写入持久化池（用于 KV 回滚）
    rotate_input_ids_triton,           # 旋转 input_ids（链式 MTP 的输入滑窗）
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    draft_tp_context,        # DP attention 所需的 draft TP context
    maybe_detect_nan,        # 调试用：检测 NaN
    maybe_detect_oob,        # 调试用：检测越界
    select_top_k_tokens,     # 从 topk 候选中选最优路径（token 树展开）
)
from sglang.srt.utils.common import empty_context, fast_topk

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner, ModelRunnerOutput


logger = logging.getLogger(__name__)


def _get_plan_stream(
    device: str,
) -> Tuple[any, contextlib.AbstractContextManager]:
    # 创建独立的 CUDA stream 用于元数据准备（与 GPU 计算重叠）
    # 如果未启用 overlap plan stream，则返回 nullcontext（同步执行）
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        plan_stream = torch.get_device_module(device).Stream()
        plan_stream_ctx = torch.get_device_module(device).stream(plan_stream)
        return plan_stream, plan_stream_ctx
    else:
        return None, contextlib.nullcontext()


class MultiLayerEagleDraftWorker(BaseDraftWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: int,
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # copy args
        # 保存服务参数和并行化配置
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.nccl_port = nccl_port
        self.target_worker = target_worker
        self.draft_extend_attn_backend_list = []
        self.model_config = target_worker.model_config

        # Args for easy access
        # 从 server_args 提取常用超参（topk / num_steps / num_draft_tokens）
        self.device = server_args.device
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Set constant
        # 每次 decode 轮次分配的 KV slot 上限 = max(steps * topk, num_draft_tokens)
        EagleDraftInput.ALLOC_LEN_PER_DECODE = max(
            self.speculative_num_steps * self.topk, self.speculative_num_draft_tokens
        )

        # Do not capture cuda graph in `TpModelWorker` init,
        # will capture later with init_cuda_graphs()
        # 暂时禁用 CUDA graph（避免 TpModelWorker 初始化时提前 capture）
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True

        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        # 草稿与目标 worker 共享同一个 req_to_token_pool 分配器（节省内存）
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        with empty_context(), speculative_moe_backend_context():
            # Init draft worker
            # 初始化草稿 TpModelWorker（多层 MTP 架构，is_multi_layer_eagle=True）
            self.draft_worker = TpModelWorker(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,  # FIXME
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                memory_pool_config=target_worker.model_runner.memory_pool_config,
                is_multi_layer_eagle=True,
            )

        # Alias for better readability
        # draft_runner_list：每个 MTP 步骤对应一个 ModelRunner（共 speculative_num_steps 个）
        self.draft_runner_list: List[ModelRunner] = self.draft_worker.model_runner_list

        # Chain-style MTP: each step propagates its own output hidden states to the
        # next step.  Non-chain: each step uses the target model's hidden states.
        # chain_mtp：下一步使用上一步的输出 hidden states；非 chain：每步都用目标模型的 hidden
        draft_arch = self.draft_worker.model_config.hf_config.architectures[0]
        self.chain_mtp_hidden_states = draft_arch in ["Step3p5MTP"]

        self.init_lm_head()

        # Used for KV Cache reversion
        # req_to_hidden_states_pool：持久化存储中间 hidden states（用于接受长度回滚时恢复 KV cache）
        # 形状：[pool_size, num_steps-1, hidden_size]
        self.req_to_hidden_states_pool = torch.empty(
            (
                self.req_to_token_pool.size,
                self.speculative_num_steps - 1,
                self.model_config.hidden_size,
            ),
            dtype=self.model_config.dtype,
            device=self.device,
        )

        # Init attention backend and cuda graphs
        for i in range(self.speculative_num_steps):
            self.draft_runner_list[i].server_args.disable_cuda_graph = (
                backup_disable_cuda_graph
            )
        # DP attention 时需要切换到 draft_tp_context（使用草稿模型的 TP group）
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        with self.draft_tp_context(
            self.draft_runner_list[0].tp_group
        ), speculative_moe_backend_context():
            self.init_attention_backend()
            self.init_cuda_graphs()

        self.tree_mask_mode = TreeMaskMode.FULL_MASK

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

    def mtp_model_runner(self, step: int):
        # 获取第 step 步的 ModelRunner
        return self.draft_runner_list[step]

    def init_lm_head(self):
        # 从目标模型获取 embedding 和 lm_head，共享给所有草稿步骤的 ModelRunner
        # EAGLE 架构：草稿模型复用目标模型的 embedding 和 lm_head（节省参数）
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        # Share the embedding and lm_head
        for i in range(self.speculative_num_steps):
            self.draft_runner_list[i].model.set_embed_and_head(embed, head)

    def init_attention_backend(self):
        # Create attn backends
        # 为每个草稿步骤创建独立的注意力 backend（支持 draft extend 模式）
        self.draft_extend_attn_backend_list = []
        for step in range(self.speculative_num_steps):
            draft_backend_factory = DraftBackendFactory(
                self.server_args,
                self.draft_runner_list[step],
                self.topk,
                self.speculative_num_steps,
            )
            self.draft_extend_attn_backend_list.append(
                draft_backend_factory.create_draft_extend_backend()
            )
            # 将创建的 backend 设置到对应 ModelRunner
            self.draft_runner_list[step].attn_backend = (
                self.draft_extend_attn_backend_list[-1]
            )

    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        # 初始化 CUDA graph runner（仅 draft extend 需要，decode 不单独 capture）
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        if self.server_args.disable_cuda_graph:
            return

        # 创建多层草稿 extend 的 CUDA graph runner
        self.cuda_graph_runner_for_draft_extend = (
            MultiLayerEagleMultiStepDraftExtendCudaGraphRunner(self)
        )

    def reset_cuda_graph_buffers(self, forward_batch, batch_result):
        # 重置 CUDA graph 缓冲区（每次 decode 前调用，清理上一轮状态）
        if self.cuda_graph_runner_for_draft_extend:
            self.cuda_graph_runner_for_draft_extend.reset_buffers(
                forward_batch, batch_result
            )

    def draft(self, model_worker_batch: ModelWorkerBatch):
        # 草稿生成主入口：准备输入 → 多步前向 → 构建 token 树 → 返回 EagleVerifyInput
        draft_input: EagleDraftInput = model_worker_batch.spec_info
        # 准备 V2 草稿前向批次（设置 seq_lens / kv_indices / 注意力参数等）
        forward_batch, can_cuda_graph = draft_input.prepare_for_v2_draft(
            self.req_to_token_pool,
            model_worker_batch,
            self.cuda_graph_runner,
            self.draft_runner_list[0],
            self.topk,
            self.speculative_num_steps,
        )

        # Run draft
        # 执行多步草稿前向，返回父节点列表、topk 排序索引和草稿 token
        parent_list, top_scores_index, draft_tokens = self.draft_forward(forward_batch)

        if model_worker_batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        # Build tree mask
        # Directly write to cuda graph buffers for verify attn
        # 直接写入验证注意力的 CUDA graph 缓冲区（避免额外复制）
        tree_mask_buf, position_buf = (
            self.target_worker.model_runner.attn_backend.get_verify_buffers_to_fill_after_draft()
        )
        # 构建 token 树的注意力 mask、位置索引、检索索引等
        (
            tree_mask,
            position,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            draft_input.verified_id,
            parent_list,
            top_scores_index,
            draft_tokens,
            model_worker_batch.seq_lens,
            model_worker_batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            self.tree_mask_mode,
            tree_mask_buf,
            position_buf,
        )

        return EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=None,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )

    def draft_forward(self, forward_batch: ForwardBatch):
        # 多步草稿前向：使用多个 MTP 层逐步扩展 token 树
        # Parse args
        spec_info: EagleDraftInput = forward_batch.spec_info
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        maybe_detect_nan(topk_p, "draft_forward: NaN in initial topk_p from spec_info")

        # Return values
        # 返回值：每步的分数列表、token 列表、父节点列表
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        # Forward multiple steps
        # 第 0 步：从 topk_p/topk_index 选取初始 token（产出分数 + 树信息）
        scores = None
        _, hidden_states, scores, tree_info = select_top_k_tokens(
            0, topk_p, topk_index, hidden_states, scores, self.topk
        )
        if self.speculative_num_steps == 1:
            # 单步时，tree_info 直接包含 scores / tokens / parents
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])
        else:
            # 多步时，tree_info 包含所有步骤的联合 topk 信息（shape: [bs, topk, num_steps]）
            for i in range(self.speculative_num_steps):
                score_list.append(tree_info[0][:, :, i].unsqueeze(-1))
                token_index = tree_info[1][:, i].unsqueeze(-1)
                token_list.append(token_index)
                if i == 0:
                    parents_list.append(tree_info[2])
                else:
                    # 非首步的父节点为上一步索引（线性链式）
                    parents_list.append(
                        torch.full(
                            (tree_info[2].size(0), 1),
                            i,
                            dtype=torch.long,
                            device="cuda",
                        )
                    )

        # Organize the results
        # 拼接所有步骤的分数和 token，选出 top-(num_draft_tokens-1) 路径
        score_list = torch.cat(score_list, dim=1).flatten(
            1
        )  # b, n, topk; n= 1 + (num_steps-1) * self.topk
        ss_token_list = torch.cat(
            token_list, dim=1
        )  # b, (self.topk + (num_steps-1) * self.topk)
        # 从分数矩阵中选取最优的 (num_draft_tokens-1) 个路径索引
        top_scores = torch.topk(
            score_list, self.speculative_num_draft_tokens - 1, dim=-1
        )
        top_scores_index = top_scores.indices
        # 排序以保持 token 树的拓扑顺序（build_tree_kernel 要求有序）
        top_scores_index = torch.sort(top_scores_index).values
        maybe_detect_oob(
            top_scores_index,
            0,
            ss_token_list.shape[1],
            "draft_forward: top_scores_index OOB for gather on ss_token_list",
        )
        # 从 ss_token_list 中 gather 出最优路径对应的 token ID
        draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

        # 拼接除最后一步外的所有父节点列表（最后一步不需要父节点）
        if len(parents_list) > 1:
            parent_list = torch.cat(parents_list[:-1], dim=1)
        else:
            batch_size = parents_list[0].shape[0]
            parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

        return parent_list, top_scores_index, draft_tokens

    def draft_extend(self):
        # 占位：多层 MTP 的 draft extend 通过 _draft_extend_for_prefill/_decode 分别处理
        pass

    def _draft_extend_for_prefill(
        self,
        batch: ModelWorkerBatch,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
    ):
        """
        Run draft model extend to correctly fill the KV cache.

        Args:
            batch: The batch to run.
            target_hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
        """
        # Prefill 后的草稿 extend：填充草稿模型 KV cache，并为下一轮 decode 初始化 topk 候选
        # Construct spec_info
        next_draft_input = EagleDraftInput(
            hidden_states=target_hidden_states,
            verified_id=next_token_ids,
            new_seq_lens=batch.seq_lens,
            # draft mode is same with decode mode, only 1 token per req
            # 草稿 extend 每请求处理 1 个 token（prefill 完成后的位置）
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )

        batch.spec_info = next_draft_input

        # Run forward
        # 初始化 ForwardBatch（EXTEND 模式，基于 draft_runner_list[0] 的参数）
        forward_batch = ForwardBatch.init_new(batch, self.draft_runner_list[0])
        forward_batch.return_hidden_states_before_norm = True

        # Construct input_ids
        # 旋转 input_ids：将 next_token_ids 追加到每个请求的序列末尾（草稿模型的输入）
        if not batch.forward_mode.is_idle():
            rotate_input_ids_triton(
                forward_batch.input_ids,
                forward_batch.extend_start_loc,
                forward_batch.extend_seq_lens,
                next_token_ids,
            )

        # 逐步运行所有 MTP 层，收集每步的 topk 概率和索引
        topk_p_list = []
        topk_index_list = []
        for step in range(self.speculative_num_steps):
            # 每个步骤使用自己的 req_to_token_pool（各 MTP 层 KV 独立）
            forward_batch.req_to_token_pool = self.draft_runner_list[
                step
            ].req_to_token_pool
            output: ModelRunnerOutput = self.draft_runner_list[step].forward(
                forward_batch
            )
            maybe_detect_nan(
                output.logits_output.next_token_logits,
                f"draft_extend_for_prefill step {step}",
            )
            probs = torch.softmax(output.logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            topk_p_list.append(topk_p)
            topk_index_list.append(topk_index)
            # Chain-style: use this step's output hidden_states as next step's input
            # 链式 MTP：将当前步输出的 hidden states 传给下一步作为输入
            if (
                self.chain_mtp_hidden_states
                and step < self.speculative_num_steps - 1
                and output.logits_output.hidden_states is not None
            ):
                forward_batch.spec_info.hidden_states = (
                    output.logits_output.hidden_states
                )
            # 旋转 input_ids：用当前步的 topk_index 更新序列（为下一步准备输入）
            if forward_batch.extend_seq_lens is not None:
                rotate_input_ids_triton(
                    forward_batch.input_ids,
                    forward_batch.extend_start_loc,
                    forward_batch.extend_seq_lens,
                    topk_index,
                )
        # 拼接所有步骤的 topk 概率和索引（[bs, num_steps * topk]）
        next_draft_input.topk_p = torch.cat(topk_p_list, dim=1)
        next_draft_input.topk_index = torch.cat(topk_index_list, dim=1)

        # Update req_to_hidden_states_pool for KV Cache reversion
        # 将 target hidden states 写入持久化池（用于后续 KV cache 回滚）
        if forward_batch.extend_seq_lens is not None:
            assign_hidden_states_pool_triton(
                target_hidden_states,
                forward_batch.req_pool_indices,
                self.req_to_hidden_states_pool,
                self.speculative_num_steps - 1,
                forward_batch.batch_size,
                forward_batch.extend_seq_lens,
                forward_batch.extend_start_loc,
            )
        return next_draft_input

    def _draft_extend_for_decode(
        self, batch: ModelWorkerBatch, batch_result: GenerationBatchResult
    ):
        # Decode 后的草稿 extend：基于验证接受的 token，更新草稿模型 KV cache 并准备下一轮 topk
        # Batch 2: Draft extend
        draft_input = EagleDraftInput(
            hidden_states=batch_result.logits_output.hidden_states,
            num_tokens_per_req=self.speculative_num_steps + 1,
            num_tokens_for_logprob_per_req=1,
        )

        # Prepare for draft extend in a separate stream
        # Notice that here we use batch_result.next_token_ids as the input ids
        # 在 plan_stream 中准备 draft extend 元数据（与 GPU 计算重叠）
        with self.plan_stream_ctx:
            forward_batch = draft_input.prepare_for_extend_to_fill_draft_kvcache(
                batch,
                batch_result.next_token_ids,
                self.speculative_num_draft_tokens,
                self.draft_runner_list[0],
                self.cuda_graph_runner_for_draft_extend,
            )
            forward_batch.return_hidden_states_before_norm = True

        # 等待 plan_stream 完成（同步到主计算流）
        if self.plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )
        # Run draft extend batch in the main compute stream
        # 在主计算流中运行草稿 extend（可使用 CUDA graph 加速）
        can_cuda_graph = (
            self.cuda_graph_runner_for_draft_extend
            and self.cuda_graph_runner_for_draft_extend.can_run(forward_batch)
        )
        ret_topk_p_list = []
        ret_topk_index_list = []
        # 备份 next_token_ids（草稿 extend 会修改 forward_batch 中的 input_ids）
        next_token_ids_backup = batch_result.next_token_ids.clone()

        if can_cuda_graph:
            self.reset_cuda_graph_buffers(forward_batch, batch_result)
        else:
            logger.warning_once(
                f"can't use cuda graph for draft extend! may have correctness issue!"
            )
            # 非 CUDA graph：手动计算每个请求最后一个接受 token 的位置索引
            select_index = (
                torch.arange(len(batch.seq_lens), device=self.device)
                * self.speculative_num_draft_tokens
                + batch_result.accept_lens
                - 1
            )

        for step in range(self.speculative_num_steps):
            # log_info_on_rank0(logger, f"step: {step}, forward_batch.input_ids: {forward_batch.input_ids}")
            if can_cuda_graph:
                # CUDA graph 路径：replay 对应步骤的 graph
                draft_logits_output = (
                    self.cuda_graph_runner_for_draft_extend.get_runner(step).replay(
                        forward_batch, init_state=(step == 0)
                    )
                )
                ret_topk_p, ret_topk_index = (
                    draft_logits_output.topk_p,
                    draft_logits_output.topk_index,
                )
            else:
                forward_batch.req_to_token_pool = self.draft_runner_list[
                    step
                ].req_to_token_pool
                draft_logits_output = self.draft_runner_list[step].forward(
                    forward_batch, skip_attn_backend_init=True
                )
                probs = torch.softmax(
                    draft_logits_output.logits_output.next_token_logits[select_index],
                    dim=-1,
                )
                ret_topk_p, ret_topk_index = fast_topk(probs, self.topk, dim=-1)
                # Chain-style: use this step's output hidden_states as next step's input
                if (
                    self.chain_mtp_hidden_states
                    and step < self.speculative_num_steps - 1
                    and draft_logits_output.logits_output.hidden_states is not None
                ):
                    forward_batch.spec_info.hidden_states = (
                        draft_logits_output.logits_output.hidden_states
                    )
                if forward_batch.extend_seq_lens is not None:
                    rotate_input_ids_triton(
                        forward_batch.input_ids,
                        forward_batch.extend_start_loc,
                        forward_batch.extend_seq_lens,
                        ret_topk_index,
                        select_index,
                    )
            ret_topk_p_list.append(ret_topk_p)
            ret_topk_index_list.append(ret_topk_index)

        # Update req_to_hidden_states_pool for KV Cache reversion
        # 更新持久化 hidden states 池（供后续 KV cache 回滚使用）
        if (
            forward_batch.extend_seq_lens is not None
            and self.cuda_graph_runner_for_draft_extend is not None
        ):
            if can_cuda_graph:
                # CUDA graph 路径：从 last_runner 的缓冲区读取 hidden states
                last_runner = self.cuda_graph_runner_for_draft_extend.get_last_runner()
                hidden_states = last_runner.buffers.hidden_states
                req_pool_indices = last_runner.buffers.req_pool_indices
                extend_seq_lens = last_runner.buffers.extend_seq_lens
                extend_start_loc = last_runner.buffers.extend_start_loc
            else:
                hidden_states = draft_logits_output.logits_output.hidden_states
                req_pool_indices = forward_batch.req_pool_indices
                extend_seq_lens = forward_batch.extend_seq_lens
                extend_start_loc = forward_batch.extend_start_loc
            assign_hidden_states_pool_triton(
                hidden_states,
                req_pool_indices,
                self.req_to_hidden_states_pool,
                self.speculative_num_steps - 1,
                forward_batch.batch_size,
                extend_seq_lens,
                extend_start_loc,
            )

        # Reorganize the spec info for the next batch
        # draft_logits_output.next_token_logits = draft_logits_output.next_token_logits[
        #     select_index
        # ]
        # draft_logits_output.hidden_states = draft_logits_output.hidden_states[
        #     select_index
        # ]
        # 恢复 next_token_ids（draft extend 可能修改了它）
        batch_result.next_token_ids = next_token_ids_backup
        # Construct the return values
        # 将多步 topk 结果写入 next_draft_input（用于下一轮草稿生成）
        next_draft_input = batch_result.next_draft_input
        (
            next_draft_input.topk_p,
            next_draft_input.topk_index,
            next_draft_input.hidden_states,
        ) = (
            torch.cat(ret_topk_p_list, dim=1).clone(),
            torch.cat(ret_topk_index_list, dim=1).clone(),
            None,
        )


class MultiLayerEagleWorkerV2(BaseSpecWorker):
    # MultiLayerEagleWorkerV2：多层 EAGLE（MTP）投机解码的顶层 Worker
    # 负责协调 target model 和 draft model 的 prefill / verify / decode 流程
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
        # ── 基本参数解析 ──────────────────────────────────────────────────────
        # Parse arguments
        self.server_args = server_args
        # EAGLE topk：每个 draft step 保留的候选 token 数
        self.topk = server_args.speculative_eagle_topk
        # draft 展开的总步数（即 speculation depth）
        self.speculative_num_steps = server_args.speculative_num_steps
        # 每次投机解码中最大的 draft token 数（topk ^ num_steps 的路径总数）
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        # GPU 设备 ID
        self.gpu_id = gpu_id
        # 推理设备（'cuda:N'）
        self.device = server_args.device
        # 保存 target worker 引用（target model 的 TpModelWorker）
        self._target_worker = target_worker
        # KV cache 页大小（paged attention 分页单位）
        self.page_size = server_args.page_size
        # 投机算法枚举（EAGLE / EAGLE2 / MTP 等）
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # ── 共享内存池 ────────────────────────────────────────────────────────
        # req_to_token_pool 和 token_to_kv_pool_allocator 与 target worker 共享
        # draft model 直接使用同一块 paged KV cache 分配器
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # ── Draft Worker 初始化 ───────────────────────────────────────────────
        # Override the context length of the draft model to be the same as the target model.
        # 将 draft model 的上下文长度对齐到 target model，避免 KV cache 边界不一致
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # 构造多层 EAGLE draft worker（包含 speculative_num_steps 个 ModelRunner）
        self._draft_worker = MultiLayerEagleDraftWorker(
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            attn_cp_rank,
            moe_dp_rank,
            nccl_port,
            target_worker,
        )

        # ── 辅助张量 ──────────────────────────────────────────────────────────
        # Some dummy tensors
        # 标量占位张量：记录每个 topk 路径新分配的 KV page 数（CUDA graph 中固定形状）
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        # 标量占位张量：记录 extend 长度（CUDA graph 中固定形状）
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        # ── Plan stream（重叠执行）─────────────────────────────────────────────
        # 创建可选的 CUDA plan stream，用于将元数据准备与 GPU 计算流水线重叠
        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

    @property
    def target_worker(self):
        # 只读属性：返回 target model 的 TpModelWorker
        return self._target_worker

    @property
    def draft_worker(self):
        # 只读属性：返回多层 EAGLE draft worker
        return self._draft_worker

    def clear_cache_pool(self):
        # KV cache 分配器与 target worker 共享，由 scheduler 统一清理，这里无需操作
        # allocator and kv cache pool are shared with target worker, which are cleared in scheduler
        pass

    def forward_batch_generation(self, model_worker_batch: ModelWorkerBatch):
        # ── 判断当前批次是否含有 prefill（extend）请求 ─────────────────────────
        if (
            model_worker_batch.forward_mode.is_extend()
            or model_worker_batch.is_extend_in_batch
        ):
            # ── Prefill 路径：target model 先做 full prefill，再做 draft extend ──
            # Target prefill：要求捕获所有 token 的 hidden states（FULL 模式）
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
            batch_output = self.target_worker.forward_batch_generation(
                model_worker_batch
            )

            # Chain-style MTP needs FULL to get all-token hidden states;
            # non-chain only needs LAST (the target model's hidden states).
            # chain MTP：每步 draft 用前一步输出的 hidden states，需要 FULL 模式
            # 非 chain MTP：每步均用 target model 的最后 token hidden state，只需 LAST
            model_worker_batch.capture_hidden_mode = (
                CaptureHiddenMode.FULL
                if self.draft_worker.chain_mtp_hidden_states
                else CaptureHiddenMode.LAST
            )
            # 调用 draft worker 执行 prefill 后的 draft KV 填充（预热 draft model）
            batch_output.next_draft_input = self.draft_worker._draft_extend_for_prefill(
                model_worker_batch,
                batch_output.logits_output.hidden_states,
                batch_output.next_token_ids,
            )
            return batch_output
        else:
            # ── Decode 路径：投机解码 draft → verify → draft extend ───────────
            if model_worker_batch.spec_info is None:
                # 若 spec_info 为空（首次 decode），构造空的 idle draft input
                model_worker_batch.spec_info = EagleDraftInput.create_idle_input(
                    device=self.device,
                    hidden_size=self.target_worker.model_config.spec_hidden_size,
                    dtype=self.target_worker.model_config.dtype,
                    topk=self.topk * self.speculative_num_steps,
                    capture_hidden_mode=CaptureHiddenMode.LAST,
                )
            # 取出当前 draft input（包含上一轮 draft 生成的 topk token 树）
            draft_input: EagleDraftInput = model_worker_batch.spec_info
            # 调用 draft worker 生成候选 token 树，返回 EagleVerifyInput
            verify_input: EagleVerifyInput = self.draft_worker.draft(model_worker_batch)
            assert verify_input.is_verify_input()
            # Record a CUDA event after draft() GPU work is dispatched.
            # 在 draft GPU 工作下发后记录 CUDA event，用于 plan stream 等待 draft 完成
            if self.plan_stream:
                self._draft_done_event = torch.get_device_module(self.device).Event()
                self._draft_done_event.record()
            # 将 verify_input 写回 batch，供 verify 步骤使用
            model_worker_batch.spec_info = verify_input
            # target model 对候选 token 树做并行 verify，返回 accept/reject 结果
            batch_output = self.verify(model_worker_batch)
            # verify 完成后，更新 draft KV cache（为下一轮 draft 做准备）
            self.draft_worker._draft_extend_for_decode(model_worker_batch, batch_output)
            return batch_output

    def verify(
        self,
        batch: ModelWorkerBatch,
    ):
        # ── 流同步：防止 plan stream 的 GPU 内存被 PyTorch GC 提前释放 ──────
        # Since batch.seq_lens is allocated in another stream, we need
        # record_stream() to prevent pytorch gc and reuse the gpu memory
        # while forward_stream is still running.
        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )

        # ── 解析参数 ──────────────────────────────────────────────────────────
        # Parse args
        verify_input: EagleVerifyInput = batch.spec_info
        # batch size（请求数）
        bs = len(batch.seq_lens)

        # ── Batch 1：Target model verify（plan stream 流水线重叠）────────────
        # Batch 1: Target verify
        # Prepare for target verify in a separate stream
        with self.plan_stream_ctx:
            # Wait for the draft CUDA graph to finish before plan_stream
            # begins its work.
            # 等待 draft CUDA graph 执行完毕，再让 plan stream 开始元数据准备
            if self.plan_stream and hasattr(self, "_draft_done_event"):
                self.plan_stream.wait_event(self._draft_done_event)
            # 在 plan stream 中准备 verify forward batch（构造 attn mask、position 等）
            verify_forward_batch, can_run_cuda_graph = (
                verify_input.prepare_for_v2_verify(
                    self.req_to_token_pool,
                    batch,
                    self.target_worker,
                )
            )

        # ── 修正 plan stream 重叠期间产生的 buffer 误差 ───────────────────────
        # Correct some buffers due to the overlap plan
        if self.plan_stream:
            # 等待 plan stream 完成，再继续主计算流
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )

            # Some values such as custom_mask and position depend on the output of draft,
            # so the previous plan step used the wrong values. Here, we need to run the related
            # computation again to update them to the correct values.
            # custom_mask 和 position 依赖 draft 输出，plan stream 中用了旧值，
            # 此处重新计算以修正这些 buffer（填充 draft 产生的真实值）
            self.target_worker.model_runner.attn_backend.update_verify_buffers_to_fill_after_draft(
                verify_input,
                (
                    self.target_worker.model_runner.graph_runner.bs
                    if can_run_cuda_graph
                    else None
                ),
            )
        # ── 在主计算流执行 target model verify forward ────────────────────────
        # Run target verify batch in the main compute stream
        forward_batch_output = self.target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            # attn backend 已由 plan stream 初始化，跳过重复初始化
            skip_attn_backend_init=True,
        )
        logits_output = forward_batch_output.logits_output

        # ── 采样：对 target logits 做 accept/reject 决策 ─────────────────────
        # Sample
        # 检测 logits 中的 NaN（调试用）
        maybe_detect_nan(logits_output.next_token_logits, "verify: target model logits")
        # verify_input.sample 执行 token 树的 accept/reject 采样：
        #   predict：每条请求最终接受的 token 序列（含 bonus token）
        #   accept_lens：每条请求实际接受的 draft token 数
        #   accept_index：accept 的 token 在 token 树中的绝对索引
        (
            predict,
            accept_lens,
            accept_index,
        ) = verify_input.sample(batch, logits_output)
        # 计算 verify 后各请求的新序列长度（seq_len += 接受数）
        new_seq_lens = batch.seq_lens + accept_lens
        # 记录 verify 完成的 CUDA event，供后续 draft extend 等待
        verify_done = torch.get_device_module(self.device).Event()
        verify_done.record()

        # ── 收集 verified token IDs ───────────────────────────────────────────
        if not batch.forward_mode.is_idle():
            # 从所有 accept token 中提取每条请求接受的最后一个 token（即 verified_id）
            all_verified_id = predict[accept_index]
            verified_id = torch.empty_like(accept_lens, dtype=torch.int32)
            # fill_new_verified_id：按 accept_lens 从 all_verified_id 中取最终接受的 token
            fill_new_verified_id[(bs,)](
                all_verified_id,
                accept_lens,
                verified_id,
                self.speculative_num_draft_tokens,
            )
        else:
            # idle batch（无真实请求）返回空张量
            verified_id = torch.empty((0,), device=self.device, dtype=torch.int32)

        # ── 计算 log probability（如需要）────────────────────────────────────
        if batch.return_logprob and not batch.forward_mode.is_idle():
            # 计算投机解码 v2 的 logprob（按接受路径汇总各步概率）
            compute_spec_v2_logprobs(
                batch, logits_output, predict, accept_index, self.speculative_num_steps
            )

        # ── 构造下一轮 draft 的输入（EagleDraftInput）────────────────────────
        # Construct the next draft input
        # verified_id：此轮 verify 最终接受的 token，作为下一轮 draft 的起始 token
        # new_seq_lens：更新后的序列长度，用于下一轮 KV cache slot 分配
        # verify_done：CUDA event，确保 draft extend 在 verify 完成后再执行
        next_draft_input = EagleDraftInput(
            verified_id=verified_id,
            new_seq_lens=new_seq_lens,
            verify_done=verify_done,
        )
        # 返回 GenerationBatchResult，包含 logits、采样结果、accept 信息和下一轮 draft input
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            accept_lens=accept_lens,
            routed_experts_output=forward_batch_output.routed_experts_output,
        )

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        # 从磁盘加载新权重：遍历所有 draft step 的 ModelRunner，逐一更新
        for i in range(self.speculative_num_steps):
            success, message = self._draft_worker.draft_runner_list[
                i
            ].update_weights_from_disk(
                recv_req.model_path,
                recv_req.load_format,
                # 是否需要重新 capture CUDA graph（权重更新后模型行为可能变化）
                recapture_cuda_graph=recv_req.recapture_cuda_graph,
            )
            # 任一步失败则立即返回错误
            if not success:
                return success, message
        return True, "Succeeded to update model weights."

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        # 通过 IPC 共享内存更新权重：遍历所有 draft step 的 ModelRunner，逐一更新
        for i in range(self.speculative_num_steps):
            success, message = self._draft_worker.draft_runner_list[
                i
            ].update_weights_from_ipc(recv_req)
            # 任一步失败则立即返回错误
            if not success:
                return success, message
        return True, "Succeeded to update model weights."
