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
# multi_layer_eagle_worker.py
# MultiLayerEagleWorker：多层 EAGLE（MTP）投机解码 Worker（V1 版本）
# 本文件实现了基于 TpModelWorker 继承的多步 EAGLE draft + verify 流程：
#   1. forward_target_extend — target model 做 prefill，捕获全部 hidden states
#   2. forward_draft_extend  — 用 target hidden states 填充 draft KV cache（prefill 路径）
#   3. draft                 — 多步 topk token 树生成，构建 EagleVerifyInput
#   4. verify                — target model 对候选 token 树做并行 verify + accept/reject
#   5. forward_draft_extend_after_decode — verify 后更新 draft KV cache（decode 路径）
# 与 V2（multi_layer_eagle_worker_v2.py）的主要区别：
#   - V1 将 draft + target 合并在同一个 Worker 类（继承 TpModelWorker）
#   - V1 直接在 ScheduleBatch 上操作，V2 使用 ModelWorkerBatch 抽象
#   - V1 不使用独立 plan stream 进行流水线重叠

import logging
import time
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

# 分布式通信：获取 TP group 用于 all_reduce 同步
from sglang.srt.distributed import get_tp_group
# DP attention 的 TP group（用于 enable_dp_attention 模式）
from sglang.srt.layers.dp_attention import get_attention_tp_group
# LogitsProcessorOutput：模型 forward 的输出类型（logits + hidden_states）
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
# MoE 后端上下文管理器（draft model 使用 speculative MoE backend）
from sglang.srt.layers.moe.utils import speculative_moe_backend_context
# 投机解码 v1 的 logprob 计算工具
from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v1
# ScheduleBatch：调度批次，包含请求信息和 spec_info
from sglang.srt.managers.schedule_batch import ScheduleBatch
# GenerationBatchResult：forward 的统一返回结构
from sglang.srt.managers.scheduler import GenerationBatchResult
# TpModelWorker：张量并行 model worker 基类
from sglang.srt.managers.tp_worker import TpModelWorker
# ForwardBatch：实际 GPU forward 使用的批次结构
# CaptureHiddenMode：控制是否捕获 hidden states（FULL/LAST/NONE）
# ForwardMode：DECODE / EXTEND / TARGET_VERIFY / DRAFT_EXTEND / IDLE
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
# ServerArgs：服务启动参数（speculative_num_steps, topk, etc.）
from sglang.srt.server_args import ServerArgs
# DraftBackendFactory：为每个 draft step 创建 attn backend
from sglang.srt.speculative.draft_utils import DraftBackendFactory
# EagleDraftInput / EagleVerifyInput / EagleVerifyOutput：核心数据类
from sglang.srt.speculative.eagle_info import (
    EagleDraftInput,
    EagleVerifyInput,
    EagleVerifyOutput,
)
# build_tree_kernel_efficient：构建 token 树 mask / position / retrieve_index
# organize_draft_results：将 multi-step topk 结果整理成树结构
from sglang.srt.speculative.eagle_utils import (
    build_tree_kernel_efficient,
    organize_draft_results,
)
# MultiLayerEagleDraftExtendCudaGraphRunner：draft extend 的 CUDA graph capture/replay
from sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner import (
    MultiLayerEagleDraftExtendCudaGraphRunner,
)
# SpeculativeAlgorithm：投机算法类型枚举（EAGLE / EAGLE2 / EAGLE3 / MTP）
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
# 投机解码工具函数
from sglang.srt.speculative.spec_utils import (
    draft_tp_context,          # DP attention 模式下切换 draft TP group 的上下文
    fast_topk,                  # 高效 topk 采样
    generate_token_bitmask,     # 结构化输出的 logit mask 生成
    load_token_map,             # 加载 hot token map（EAGLE3 / speculative_token_map）
    maybe_detect_nan,           # NaN 检测（调试）
    select_top_k_tokens,        # 从 topk_p/topk_index 中选出下一步 input_ids
)
# empty_context：无操作的上下文管理器；GPU 内存工具；平台检测
from sglang.srt.utils import empty_context, get_available_gpu_memory, is_cuda, is_npu

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

# 平台检测：NPU 不支持 CUDA graph
_is_npu = is_npu()

if is_cuda():
    # segment_packbits：用于 grammar-guided 结构化输出的 bitmask 压缩
    from sgl_kernel import segment_packbits  # noqa: F401

logger = logging.getLogger(__name__)


class MultiLayerEagleWorker(TpModelWorker):
    # MultiLayerEagleWorker：多层 EAGLE（MTP）投机解码 Worker（V1 版本）
    # 继承自 TpModelWorker，在同一对象中同时管理 draft model 和对 target worker 的引用
    # draft model 以 speculative_num_steps 个独立 ModelRunner 实现多步 topk 采样

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
        # draft 展开的总步数（speculation depth）
        self.speculative_num_steps = server_args.speculative_num_steps
        # 每轮投机解码中候选 token 的最大总数（topk^num_steps 路径裁剪后）
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        # GPU 设备 ID
        self.gpu_id = gpu_id
        # 推理设备（'cuda:N'）
        self.device = server_args.device
        # 保存 target worker 引用，draft model 与 target model 共享 embed + lm_head
        self.target_worker = target_worker
        # KV cache 分页单位
        self.page_size = server_args.page_size
        # 投机算法枚举（EAGLE / EAGLE2 / EAGLE3 / MTP）
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        # draft extend attn backend 列表（每个 draft step 一个）
        self.draft_extend_attn_backend_list = []

        # ── 对齐 draft model 上下文长度 ───────────────────────────────────────
        # Override the context length of the draft model to be the same as the target model.
        # 避免 draft / target KV cache 边界不一致
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # ── 临时禁用 CUDA graph 捕获，稍后在 init_cuda_graphs 中统一 capture ──
        # Do not capture cuda graph in `super().__init__()`
        # It will be captured later.
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True
        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        # req_to_token_pool 与 target worker 共享（同一块 paged KV 分配器）
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # ── Hot token map（EAGLE3 / speculative_token_map）────────────────────
        # Load hot token ids
        if self.speculative_algorithm.is_eagle3():
            # EAGLE3 模型内置 hot token map，外部指定会被忽略
            if server_args.speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None
        elif server_args.speculative_token_map is not None:
            # 加载外部 hot token map 文件，用于缩减 vocab 维度以加速 topk
            self.hot_token_id = load_token_map(server_args.speculative_token_map)
            # 将 hot_vocab_size 注入 json_model_override_args，draft 模型据此初始化
            server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None

        # ── Init draft worker（通过 super().__init__ 初始化 TpModelWorker）────
        # Init draft worker
        if server_args.enable_dp_attention and self.speculative_algorithm.is_eagle3():
            # EAGLE3 + DP attention：draft model 使用独立的 attention TP group
            ctx = draft_tp_context(get_attention_tp_group())
        else:
            ctx = empty_context()
        with ctx, speculative_moe_backend_context():
            super().__init__(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,  # FIXME
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                # 标记为 draft worker，影响 KV cache 分配和 model 加载逻辑
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                memory_pool_config=target_worker.model_runner.memory_pool_config,
                # 启用多层 EAGLE 模式（加载 speculative_num_steps 个 ModelRunner）
                is_multi_layer_eagle=True,
            )

        # ── 共享 embed + lm_head ──────────────────────────────────────────────
        # 从 target model 获取 embedding 层和 lm_head
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()

        if self.speculative_algorithm.is_eagle3():
            # EAGLE3：大多数情况 draft model 有自己的 lm_head
            # most cases EAGLE3 models don't share lm_head
            # but some models (e.g. nvidia/gpt-oss-120b-Eagle3) shares
            if (
                hasattr(self.draft_model_runner.model, "load_lm_head_from_target")
                and self.draft_model_runner.model.load_lm_head_from_target
            ):
                # 特殊 EAGLE3 模型：共享完整 embed + lm_head
                self.draft_model_runner.model.set_embed_and_head(embed, head)
            else:
                # 通常只共享 embed，lm_head 由 draft model 自己维护
                self.draft_model_runner.model.set_embed(embed)

            # grab hot token ids
            # 从 draft model 内部获取 hot_token_id（EAGLE3 模型自带）
            if self.draft_model_runner.model.hot_token_id is not None:
                self.hot_token_id = self.draft_model_runner.model.hot_token_id.to(
                    embed.device
                )

        else:
            # EAGLE / EAGLE2 / MTP：所有 draft step 共享同一 embed + lm_head
            if self.hot_token_id is not None:
                # hot token map 模式：裁剪 lm_head 到 hot_vocab_size 维度
                head = head.clone()
                self.hot_token_id = self.hot_token_id.to(head.device)
                head.data = head.data[self.hot_token_id]

            # Share the embedding and lm_head
            # 将所有 speculative_num_steps 个 draft step 的 embed/lm_head 指向同一份
            for i in range(self.speculative_num_steps):
                self.mtp_model_runner(i).model.set_embed_and_head(embed, head)

        # ── Init attention backend 和 CUDA graph ──────────────────────────────
        # Init attention backend and cuda graphs
        # 恢复 disable_cuda_graph 设置（之前临时关闭）
        for i in range(self.speculative_num_steps):
            self.mtp_model_runner(i).server_args.disable_cuda_graph = (
                backup_disable_cuda_graph
            )
        # draft_tp_context：DP attention 模式下切换 TP group
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        with self.draft_tp_context(
            self.mtp_model_runner(0).tp_group
        ), speculative_moe_backend_context():
            # 为每个 draft step 创建 attention backend
            self.init_attention_backend()
            # 为每个 draft step capture CUDA graph
            self.init_cuda_graphs()

        # ── 辅助标量张量 ──────────────────────────────────────────────────────
        # Some dummy tensors
        # 标量占位：记录每个 topk 路径新分配的 KV page 数
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        # 标量占位：记录 extend 长度
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

    def init_attention_backend(self):
        # 为每个 draft step 创建 DraftBackendFactory 并生成 draft extend attn backend
        # Create multi-step attn backends and cuda graph runners
        for step in range(self.speculative_num_steps):
            # DraftBackendFactory 根据 speculative_attention_mode 选择合适的 backend 实现
            draft_backend_factory = DraftBackendFactory(
                self.server_args,
                self.mtp_model_runner(step),
                self.topk,
                self.speculative_num_steps,
            )

            # Initialize draft extend attention backend (respects speculative_attention_mode setting)
            # 创建 draft extend backend（支持 CUDA graph replay 的特殊 attn backend）
            self.draft_extend_attn_backend_list.append(
                draft_backend_factory.create_draft_extend_backend()
            )

    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        # 初始化每个 draft step 的 CUDA graph runner 列表
        self.cuda_graph_runner_for_draft_extend_list = []

        # 若全局禁用 CUDA graph，直接返回（不 capture）
        if self.server_args.disable_cuda_graph:
            return

        # Capture extend
        # 为每个 draft step capture draft extend CUDA graph
        for step in range(self.speculative_num_steps):
            if self.draft_extend_attn_backend_list[step] and not _is_npu:
                # 记录 capture 开始时间和 GPU 可用内存（供日志输出）
                tic = time.perf_counter()
                before_mem = get_available_gpu_memory(self.device, self.gpu_id)
                logger.info(
                    f"Capture draft extend cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
                )
                # MultiLayerEagleDraftExtendCudaGraphRunner：capture 并持有该 step 的 CUDA graph
                self.cuda_graph_runner_for_draft_extend_list.append(
                    MultiLayerEagleDraftExtendCudaGraphRunner(self, step)
                )
                after_mem = get_available_gpu_memory(self.device, self.gpu_id)
                logger.info(
                    f"Capture draft extend cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
                )

    def mtp_model_runner(self, layer_id: int) -> ModelRunner:
        # 按 step 索引获取对应的 ModelRunner（model_runner_list 由 TpModelWorker 构建）
        return self.model_runner_list[layer_id]

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Run speculative decoding forward.

        NOTE: Many states of batch is modified as you go through. It is not guaranteed that
        the final output batch have the same state as the input.

        Args:
            batch: The batch to run forward. The state of the batch is modified as it runs.
        Returns:
            A tuple of the final logit output of the target model, next tokens accepted,
            the batch id (used for overlap schedule), and number of accepted tokens.
        """
        # ── Prefill（extend）路径 ─────────────────────────────────────────────
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            # 1. target model 做 full prefill，捕获所有 token 的 hidden states
            (
                logits_output,
                next_token_ids,
                seq_lens_cpu,
                can_run_cuda_graph,
            ) = self.forward_target_extend(batch)
            # 2. 使用 target hidden states 填充所有 draft step 的 KV cache
            with self.draft_tp_context(
                self.mtp_model_runner(0).tp_group
            ), speculative_moe_backend_context():
                self.forward_draft_extend(
                    batch, logits_output.hidden_states, next_token_ids, seq_lens_cpu
                )
            # prefill 路径不返回 accept 数（num_accepted_drafts=0）
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_drafts=0,
                can_run_cuda_graph=can_run_cuda_graph,
            )
        else:
            # ── Decode 路径：draft → verify → draft extend after decode ────────
            with self.draft_tp_context(
                self.mtp_model_runner(0).tp_group
            ), speculative_moe_backend_context():
                # draft：多步 topk 采样，构建候选 token 树，返回 EagleVerifyInput
                spec_info = self.draft(batch)
            # verify：target model 对 token 树做并行 verify，返回 accept/reject 结果
            logits_output, verify_output, model_worker_batch, can_run_cuda_graph = (
                self.verify(batch, spec_info)
            )

            with self.draft_tp_context(
                self.mtp_model_runner(0).tp_group
            ), speculative_moe_backend_context():
                # NOTE: We should use `check_forward_draft_extend_after_decode`
                # when DP attention is enabled, but it is slow. Skip it for now.
                # verify 后若有接受的 token，或 DP attention 模式下（全局同步），执行 draft extend
                if (
                    self.server_args.enable_dp_attention
                    or batch.spec_info.verified_id.shape[0] > 0
                ):
                    # decode is not finished
                    # 更新 draft KV cache，为下一轮 draft 生成做准备
                    self.forward_draft_extend_after_decode(batch)

            # 返回 verify 结果：logits、接受 token、接受总数
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=verify_output.verified_id,
                num_accepted_drafts=sum(verify_output.num_accepted_drafts_per_req_cpu),
                can_run_cuda_graph=can_run_cuda_graph,
            )

    def check_forward_draft_extend_after_decode(self, batch: ScheduleBatch):
        # 检查当前 batch 是否需要执行 draft extend after decode
        # 单卡：直接判断 verified_id 是否非空
        local_need_forward = batch.spec_info.verified_id.shape[0] > 0
        if not self.server_args.enable_dp_attention:
            return local_need_forward

        # DP attention 模式：需要全局 all_reduce 确保所有 DP rank 一致决策
        global_need_forward = torch.tensor(
            [
                (local_need_forward),
            ],
            dtype=torch.int64,
        )
        # 跨所有 TP rank 求和（任意 rank 需要则全部执行）
        torch.distributed.all_reduce(
            global_need_forward, group=get_tp_group().cpu_group
        )
        global_need_forward_cnt = global_need_forward[0].item()
        need_forward = global_need_forward_cnt > 0
        return need_forward

    def forward_target_extend(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, torch.Tensor, Optional[torch.Tensor], bool]:
        """Run the target extend.

        Args:
            batch: The batch to run. States could be modified.

        Returns:
            logits_output: The output of logits. It will contain the full hidden states.
            next_token_ids: Next token ids generated.
            seq_lens_cpu: CPU copy of sequence lengths for the draft prefill path.
            can_run_cuda_graph: Whether the target prefill ran with cuda graph.
        """
        # Forward with the target model and get hidden states.
        # We need the full hidden states to prefill the KV cache of the draft model.
        # 构造 ModelWorkerBatch，要求捕获所有 token 的 hidden states（FULL 模式）
        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        # return_hidden_states_before_norm=True：在 norm 前捕获 hidden states
        # 这样 draft model 使用的特征与 target model 内部一致
        model_worker_batch.return_hidden_states_before_norm = True
        # 调用 target worker 做 prefill forward
        batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
        logits_output, next_token_ids = (
            batch_result.logits_output,
            batch_result.next_token_ids,
        )
        # 返回 logits、采样 token、CPU 序列长度（用于 draft extend）、是否用了 CUDA graph
        return (
            logits_output,
            next_token_ids,
            model_worker_batch.seq_lens_cpu,
            batch_result.can_run_cuda_graph,
        )

    def _draft_preprocess_decode(self, batch: ScheduleBatch):
        from sglang.srt.speculative.eagle_worker import EAGLEWorker

        # FIXME: migrate multi-layer eagle worker to eagle worker
        # 暂时复用 EAGLEWorker 的 decode 预处理逻辑（待迁移）
        return EAGLEWorker._draft_preprocess_decode(self, batch)

    def _draft_preprocess_idle(self, batch: ScheduleBatch):
        from sglang.srt.speculative.eagle_worker import EAGLEWorker

        # FIXME: migrate multi-layer eagle worker to eagle worker
        # 暂时复用 EAGLEWorker 的 idle 预处理逻辑（待迁移）
        return EAGLEWorker._draft_preprocess_idle(self, batch)

    def draft(self, batch: ScheduleBatch):
        # ── 预处理：根据 forward_mode 选择 idle / decode 路径 ─────────────────
        # Parse args
        if batch.forward_mode.is_idle():
            # idle batch：无真实请求，仅做状态初始化
            self._draft_preprocess_idle(batch)
        else:
            # decode batch：准备 draft input（topk_p, topk_index, hidden_states）
            self._draft_preprocess_decode(batch)

        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        # ── 设置 draft 参数 ───────────────────────────────────────────────────
        # 只需要最后一个 token 的 hidden state（LAST 模式）
        spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        # 每个请求在 draft step 中生成的 token 数
        spec_info.num_tokens_per_req = self.topk
        spec_info.num_tokens_for_logprob_per_req = self.topk
        batch.return_hidden_states = False

        # ── 构造 ForwardBatch ────────────────────────────────────────────────
        # Get forward batch
        model_worker_batch = batch.get_model_worker_batch()
        assert model_worker_batch.capture_hidden_mode == CaptureHiddenMode.LAST
        # 用第 0 步的 ModelRunner 初始化 ForwardBatch（共享 KV cache 结构）
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.mtp_model_runner(0)
        )
        # 禁用 DP CUDA graph（draft 步骤不支持）
        forward_batch.can_run_dp_cuda_graph = False
        # 需要在 norm 前的 hidden states 用于下一步 draft
        forward_batch.return_hidden_states_before_norm = True

        # ── 解析上一轮 draft 的 topk 结果 ────────────────────────────────────
        # Parse args
        assert isinstance(spec_info, EagleDraftInput)
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        # 检测上一轮 topk_p 中的 NaN
        maybe_detect_nan(topk_p, "draft: NaN in initial topk_p from spec_info")

        # ── 初始化返回列表 ────────────────────────────────────────────────────
        # Return values
        score_list: List[torch.Tensor] = []    # 各节点的累积概率分数
        token_list: List[torch.Tensor] = []    # 各节点对应的 token ID
        parents_list: List[torch.Tensor] = []  # 各节点在树中的父节点索引

        # ── 多步 draft forward ───────────────────────────────────────────────
        # Forward multiple steps
        # select_top_k_tokens：从上一步的 topk_p/topk_index 中选出本步输入 input_ids
        # 同时返回 tree_info（得分、token、父节点信息）
        scores = None
        input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
            0, topk_p, topk_index, hidden_states, scores, self.topk
        )
        if self.speculative_num_steps == 1:
            # 单步：直接收集 tree_info
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])
        else:
            # 多步（MTP）：每步在 tree_info 的第 i 个位置取分数/token/父节点
            for i in range(self.speculative_num_steps):
                # 第 i 步的分数（三维张量的第 i 列，unsqueeze 恢复维度）
                score_list.append(tree_info[0][:, :, i].unsqueeze(-1))
                # 第 i 步 token 索引（unsqueeze 恢复维度）
                token_index = tree_info[1][:, i].unsqueeze(-1)
                token_list.append(token_index)
                if i == 0:
                    # 第 0 步：父节点由 tree_info 提供（完整树结构）
                    parents_list.append(tree_info[2])
                else:
                    # 后续步骤：父节点即为第 i 个节点（链式连接）
                    parents_list.append(
                        torch.full(
                            (tree_info[2].size(0), 1),
                            i,
                            dtype=torch.long,
                            device=self.device,
                        )
                    )

        # organize_draft_results：整理多步 topk 结果为树形结构
        #   parent_list：各节点父节点索引列表
        #   top_scores_index：按得分排序的路径索引
        #   draft_tokens：最终候选 token 列表
        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )

        # ── idle batch 直接返回空 verify input ───────────────────────────────
        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        # ── 构建 token 树（mask、position、retrieve_index）────────────────────
        # build_tree_kernel_efficient：
        #   tree_mask：注意力掩码（候选 token 只能看到其祖先节点）
        #   position：候选 token 的 position_id（基于 seq_len）
        #   retrieve_index：accept 后从树中提取已接受 token 的索引
        #   retrieve_next_token / retrieve_next_sibling：用于结构化输出和 bonus token 查找
        (
            tree_mask,
            position,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            spec_info.verified_id,
            parent_list,
            top_scores_index,
            draft_tokens,
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
        )

        # 返回 EagleVerifyInput：包含完整 token 树信息，供 target model verify 使用
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
            draft_token_num=self.server_args.speculative_num_draft_tokens,
            # FULL：verify 时需要所有 token 的 hidden states 用于 accept 后的 draft extend
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=forward_batch.seq_lens_sum,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
        )

    def clear_cache_pool(self):
        # KV cache 分配器与 target worker 共享，由 scheduler 统一清理
        # allocator and kv cache pool are shared with target worker
        pass

    def verify(self, batch: ScheduleBatch, spec_info: EagleVerifyInput):
        # ── 准备 verify：分配 KV cache slot，构建 attention 元数据 ────────────
        spec_info.prepare_for_verify(batch, self.page_size)
        batch.return_hidden_states = False
        # 切换 forward_mode 为 TARGET_VERIFY，使 attn backend 使用 verify 专用 mask
        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        # 将 verify_input 绑定到 batch 上（attn backend 根据此构建 tree mask）
        batch.spec_info = spec_info

        # 构造 ModelWorkerBatch（复用 spec_info 中缓存的 seq_lens_cpu 避免重复 CPU 同步）
        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=spec_info.seq_lens_cpu
        )
        # 确认 capture_hidden_mode 与 spec_info 中指定的一致（FULL）
        assert model_worker_batch.capture_hidden_mode == spec_info.capture_hidden_mode
        # 捕获 norm 前的 hidden states，用于后续 draft extend
        model_worker_batch.return_hidden_states_before_norm = True

        # ── 结构化输出（grammar）预处理：在 GPU forward 前预先在 CPU 构建 bitmask ──
        if batch.has_grammar:
            # 将 retrieve_next_token / retrieve_next_sibling / draft_token 搬到 CPU
            # 避免在 GPU forward 期间阻塞（CPU 计算与 GPU 并行）
            retrieve_next_token_cpu = spec_info.retrieve_next_token.cpu()
            retrieve_next_sibling_cpu = spec_info.retrieve_next_sibling.cpu()
            draft_tokens_cpu = spec_info.draft_token.view(
                spec_info.retrieve_next_token.shape
            ).cpu()

        # ── Forward：target model 对候选 token 树做并行推理 ───────────────────
        # Forward
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        # ── 结构化输出 logit mask（与 GPU forward 重叠的 CPU 计算）──────────
        vocab_mask = None
        if batch.has_grammar:
            # Generate the logit mask for structured output.
            # Overlap the CPU operations for bitmask generation with the forward pass.
            # 生成 token 树每个节点的 vocab logit mask（屏蔽不合法 token）
            vocab_mask = generate_token_bitmask(
                batch.reqs,
                spec_info,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )

            if vocab_mask is not None:
                assert spec_info.grammar is not None
                # 将 bitmask 搬到 GPU，供 logits 采样使用
                vocab_mask = vocab_mask.to(spec_info.retrieve_next_token.device)
                # NOTE (sk): otherwise, this vocab mask will be the one from the previous extend stage
                # and will be applied to produce wrong results
                # 清除之前 extend 阶段遗留的 vocab_mask，避免误用旧 mask
                batch.sampling_info.vocab_mask = None

        # ── 采样：accept/reject 决策 ─────────────────────────────────────────
        maybe_detect_nan(logits_output.next_token_logits, "verify: target model logits")

        # 保存 hidden states 到 spec_info，供 accept 后的 draft extend 使用
        spec_info.hidden_states = logits_output.hidden_states
        # spec_info.verify：执行 accept/reject 采样，更新 KV cache，返回 EagleVerifyOutput
        res: EagleVerifyOutput = spec_info.verify(
            batch,
            logits_output,
            self.token_to_kv_pool_allocator,
            self.page_size,
            vocab_mask,
        )

        # ── Post process：裁剪 logits 到已接受的 token 子集 ──────────────────
        # Post process based on verified outputs.
        # Pick indices that we care (accepted)
        # 只保留被接受 token 对应的 logits（用于 logprob 计算）
        logits_output.next_token_logits = logits_output.next_token_logits[
            res.accepted_indices
        ]
        # 同样裁剪 hidden states（用于后续 draft extend）
        logits_output.hidden_states = logits_output.hidden_states[res.accepted_indices]

        # ── Mamba SSM 状态更新（Hybrid Mamba 模型专用）───────────────────────
        if self.target_worker.model_runner.hybrid_gdn_config is not None:
            # accepted_length = 每条请求接受的 draft token 数 + 1（含 bonus token）
            accepted_length = (
                torch.tensor(
                    res.num_accepted_drafts_per_req_cpu,
                    device=logits_output.hidden_states.device,
                    dtype=torch.int64,
                )
                + 1
            )

            # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
            # res.accepted_indices.shape[0] > 0 skips DP attn idle batch
            # topk > 1 时，需要通过 token 树的 retrieve 索引计算每条请求接受的 SSM 状态偏移
            if spec_info.topk > 1 and res.accepted_indices.shape[0] > 0:
                # accepted_indices=[0,2,3,4,5,7,9,10,11], accepted_length=[4, 3, 2], cumulative_accepted_lengths=[4, 7, 9]
                # first_token_indices_per_req=prepend(0, accepted_indices[cumulative_accepted_lengths[:-1]]) = [0, 5, 10]
                # last_token_indices_per_req=accepted_indices[cumulative_accepted_lengths - 1] = [4, 9, 11] (last token ID of each req)
                # max_relative_indices_per_req = [4,4,1]; those are the per-req spec-decoding step offsets that contain the correct mamba caches
                # 累积接受长度，用于找到每条请求在 accepted_indices 中的边界
                cumulative_accepted_lengths = torch.cumsum(accepted_length, dim=0)
                # 每条请求第一个接受 token 在 accepted_indices 中的位置（前缀 0）
                req_start_positions = torch.cat(
                    [
                        torch.zeros(
                            1,
                            dtype=cumulative_accepted_lengths.dtype,
                            device=cumulative_accepted_lengths.device,
                        ),
                        cumulative_accepted_lengths[:-1],
                    ]
                )
                # 每条请求接受序列的第一个和最后一个 token 在 accepted_indices 中的全局索引
                first_token_indices_per_req = res.accepted_indices[req_start_positions]
                last_token_indices_per_req = res.accepted_indices[
                    cumulative_accepted_lengths - 1
                ]
                # max_relative_indices：每条请求接受的最后一个 token 相对于第一个的偏移
                # 即该请求对应的 Mamba SSM 缓存 step 偏移
                max_relative_indices_per_req = (
                    last_token_indices_per_req - first_token_indices_per_req
                )
            else:
                # topk == 1 时直接使用 accepted_length - 1 作为偏移
                max_relative_indices_per_req = accepted_length - 1
            # 用正确的 SSM 状态更新 Mamba 缓存（将 verify 路径的 SSM 状态提交）
            self.target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify(
                max_relative_indices_per_req, self.target_worker.model_runner.model
            )

        # ── Logprob 后处理 ────────────────────────────────────────────────────
        if batch.return_logprob:
            # 计算投机解码 v1 的 logprob（按接受路径汇总概率）
            add_output_logprobs_for_spec_v1(batch, res, logits_output)

        # ── 准备下一轮 draft 的批次状态 ──────────────────────────────────────
        # Prepare the batch for the next draft forwards.
        # 切换回 DECODE 模式，供 forward_draft_extend_after_decode 使用
        batch.forward_mode = (
            ForwardMode.DECODE if not batch.forward_mode.is_idle() else ForwardMode.IDLE
        )
        # res.draft_input：verify 后构建的 EagleDraftInput，包含 verified_id 和 hidden_states
        batch.spec_info = res.draft_input

        return logits_output, res, model_worker_batch, can_run_cuda_graph

    def forward_draft_extend(
        self,
        batch: ScheduleBatch,
        hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        """Run draft model extend. This API modifies the states of the batch.

        Args:
            batch: The batch to run.
            hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
        """
        # ── 构建 draft extend 的 EagleDraftInput ──────────────────────────────
        # 以 target model 输出的 hidden_states 和 next_token_ids 为起点
        # 填充所有 draft step 的 KV cache（prefill 路径）
        batch.spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            # verified_id：此处为 target prefill 产出的下一个 token（作为 draft 的起始）
            verified_id=next_token_ids,
            # prefill 路径每个请求只有 1 个 token 进入 draft extend
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )
        batch.return_hidden_states = False
        # 准备 draft extend 的 input_ids、position 等（在 batch.spec_info 上 in-place 操作）
        batch.spec_info.prepare_for_extend(batch)
        # 只需最后一个 token 的 hidden state 用于下一轮 draft
        batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        # 复用 target prefill 的 seq_lens_cpu（避免重复 CPU 同步）
        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=seq_lens_cpu
        )
        # 用第 0 步 ModelRunner 初始化 ForwardBatch
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.mtp_model_runner(0)
        )
        # 不需要 logprob（draft extend 只用于更新 draft KV cache）
        forward_batch.return_logprob = False
        # 捕获 norm 前的 hidden states 供后续 draft step 使用
        forward_batch.return_hidden_states_before_norm = True
        topk_p_list = []
        topk_index_list = []

        # ── 多步 draft extend forward ─────────────────────────────────────────
        for step in range(self.speculative_num_steps):
            # 每步 draft model forward，返回 logits
            logits_output = (
                self.mtp_model_runner(step).forward(forward_batch).logits_output
            )
            maybe_detect_nan(
                logits_output.next_token_logits,
                f"draft_extend_for_prefill step {step}",
            )
            # softmax → topk：获取本步的 topk 概率和 token 索引
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            topk_p_list.append(topk_p)
            topk_index_list.append(topk_index)
            # ── 滑动窗口：将 input_ids 向前移一步 ────────────────────────────
            # 将每条请求的 input_ids 窗口向右滑动一位：
            # [t1, t2, ..., tN] → [t2, ..., tN, topk_index[i]]
            # 模拟"已预测一步"的 KV cache 填充
            pt = 0
            if forward_batch.extend_seq_lens is not None:
                for i, extend_len in enumerate(forward_batch.extend_seq_lens):
                    input_ids = forward_batch.input_ids[pt : pt + extend_len]
                    forward_batch.input_ids[pt : pt + extend_len] = torch.cat(
                        (input_ids[1:], topk_index[i].reshape(1))
                    )
                    pt += extend_len

        # ── 将多步 topk 结果写入 spec_info（供下一轮 draft 使用）─────────────
        assert isinstance(forward_batch.spec_info, EagleDraftInput)
        assert forward_batch.spec_info is batch.spec_info
        # cat：将所有步的 topk_p 在 dim=1（topk 维）拼接，形成 [bs, num_steps*topk]
        forward_batch.spec_info.topk_p = torch.cat(topk_p_list, dim=1)
        forward_batch.spec_info.topk_index = torch.cat(topk_index_list, dim=1)

    def forward_draft_extend_after_decode(self, batch: ScheduleBatch):
        # verify 完成后执行 draft extend：更新 draft KV cache，为下一轮 draft 准备 topk 结果
        assert isinstance(batch.spec_info, EagleDraftInput)
        # ── 备份 batch 中会被 in-place 修改的字段 ──────────────────────────────
        # Backup fields that will be modified in-place
        seq_lens_backup = batch.seq_lens.clone()
        seq_lens_cpu_backup = batch.seq_lens_cpu.clone()
        req_pool_indices_backup = batch.req_pool_indices
        # 备份 accept 统计，防止 draft extend 过程中被覆盖
        num_accepted_drafts_backup = batch.spec_info.num_accepted_drafts
        num_accepted_tokens_backup = batch.spec_info.num_accepted_tokens
        return_logprob_backup = batch.return_logprob

        # 记录当前是否为 idle batch（无真实请求）
        input_is_idle = batch.forward_mode.is_idle()

        # ── 特殊处理：有请求但无 token 被接受（全拒绝）────────────────────────
        if not input_is_idle and batch.spec_info.verified_id.numel() == 0:
            # 全部 draft token 被拒绝，切换到 idle batch 以避免空 forward
            batch = batch.copy()
            batch.prepare_for_idle()
            # 构造空的 idle draft input（hidden states 全零，形状匹配 draft model）
            hidden_size = (
                self.model_config.hidden_size * 3
                if self.speculative_algorithm.is_eagle3()
                else self.model_config.hidden_size
            )
            batch.spec_info = EagleDraftInput.create_idle_input(
                device=self.device,
                hidden_size=hidden_size,
                dtype=self.model_config.dtype,
                topk=self.topk,
                capture_hidden_mode=CaptureHiddenMode.LAST,
            )

        # ── 设置 draft extend 参数 ────────────────────────────────────────────
        # num_tokens_per_req = speculative_num_steps + 1：
        #   verify 接受的 token（verified_id）+ num_steps 个 bonus 位置
        batch.spec_info.num_tokens_per_req = self.speculative_num_steps + 1
        # logprob 只需要最后一个 token
        batch.spec_info.num_tokens_for_logprob_per_req = 1
        # prepare_extend_after_decode：调整 seq_lens、input_ids、kv slot 等用于 draft extend
        batch.spec_info.prepare_extend_after_decode(
            batch,
            self.speculative_num_steps,
        )
        # 切换 forward_mode 为 DRAFT_EXTEND（attn backend 使用 extend 专用 mask）
        batch.forward_mode = (
            ForwardMode.DRAFT_EXTEND
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )

        batch.return_hidden_states = False
        model_worker_batch = batch.get_model_worker_batch()
        # draft extend 只需最后 token 的 hidden state
        assert model_worker_batch.capture_hidden_mode == CaptureHiddenMode.LAST
        # 用第 0 步 ModelRunner 初始化 ForwardBatch
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.mtp_model_runner(0)
        )
        # 捕获 norm 前 hidden states 供后续 draft step 使用
        forward_batch.return_hidden_states_before_norm = True
        # 更新 seq_lens_sum（用于 attention 计算的偏移基准）
        if forward_batch.seq_lens_cpu is not None:
            forward_batch.seq_lens_sum = forward_batch.seq_lens_cpu.sum().item()
        else:
            forward_batch.seq_lens_sum = batch.seq_lens.sum().item()
        topk_p_list = []
        topk_index_list = []

        # ── 多步 draft extend（decode 路径）──────────────────────────────────
        # Run
        for step in range(self.speculative_num_steps):
            # 尝试使用 CUDA graph replay 加速（形状匹配时 replay，否则走正常 forward）
            can_cuda_graph = len(
                self.cuda_graph_runner_for_draft_extend_list
            ) and self.cuda_graph_runner_for_draft_extend_list[step].can_run(
                forward_batch
            )
            if can_cuda_graph:
                # CUDA graph replay：直接重放预先 capture 的 graph
                logits_output = self.cuda_graph_runner_for_draft_extend_list[
                    step
                ].replay(forward_batch)
            else:
                # 正常 forward：禁用 DP CUDA graph，手动初始化 attn backend
                forward_batch.can_run_dp_cuda_graph = False
                if not forward_batch.forward_mode.is_idle():
                    # 初始化 attn backend 的 forward metadata（KV slot、mask 等）
                    self.mtp_model_runner(step).attn_backend.init_forward_metadata(
                        forward_batch
                    )
                # 跳过 attn backend 二次初始化（已在上方手动完成）
                logits_output = (
                    self.mtp_model_runner(step)
                    .forward(forward_batch, skip_attn_backend_init=True)
                    .logits_output
                )

            maybe_detect_nan(
                logits_output.next_token_logits,
                f"draft_extend_after_decode step {step} (cuda_graph={can_cuda_graph})",
            )
            # softmax → topk：获取本步 topk 候选
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            topk_p_list.append(topk_p)
            topk_index_list.append(topk_index)
            # ── 滑动窗口：input_ids 向右移一位 ───────────────────────────────
            pt = 0
            if forward_batch.extend_seq_lens is not None:
                for i, extend_len in enumerate(forward_batch.extend_seq_lens):
                    input_ids = forward_batch.input_ids[pt : pt + extend_len]
                    forward_batch.input_ids[pt : pt + extend_len] = torch.cat(
                        (input_ids[1:], topk_index[i].reshape(1))
                    )
                    pt += extend_len

        # ── 写入多步 topk 结果 ────────────────────────────────────────────────
        forward_batch.spec_info.topk_p = torch.cat(topk_p_list, dim=1)
        forward_batch.spec_info.topk_index = torch.cat(topk_index_list, dim=1)

        # ── 恢复备份字段 ──────────────────────────────────────────────────────
        # Restore backup.
        # This is because `seq_lens` can be modified in `prepare_extend_after_decode`
        # 恢复 forward_mode（供调用方使用正确状态）
        batch.forward_mode = (
            ForwardMode.DECODE if not input_is_idle else ForwardMode.IDLE
        )
        # 恢复 seq_lens（draft extend 中会 in-place 修改）
        batch.seq_lens = seq_lens_backup
        batch.seq_lens_cpu = seq_lens_cpu_backup
        # 恢复 req_pool_indices（draft extend 中可能修改）
        batch.req_pool_indices = req_pool_indices_backup
        # 恢复 accept 统计计数
        batch.spec_info.num_accepted_drafts = num_accepted_drafts_backup
        batch.spec_info.num_accepted_tokens = num_accepted_tokens_backup
        # 恢复 return_logprob 标志
        batch.return_logprob = return_logprob_backup
