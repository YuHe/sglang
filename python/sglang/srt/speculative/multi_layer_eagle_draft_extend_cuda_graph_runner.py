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

from __future__ import annotations

# Multi-Layer EAGLE 草稿 Extend 阶段 CUDA Graph 捕获与回放
# 支持链式 MTP 多步草稿：每一步对应一个 Runner，前步输出作为后步输入
import bisect
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional

import torch

from sglang.srt.layers.dp_attention import DpPaddingMode, set_dp_buffer_len
from sglang.srt.model_executor.cuda_graph_runner import (
    CUDA_GRAPH_CAPTURE_FAILED_MSG,
    CudaGraphRunner,
    DeepEPCudaGraphRunnerAdapter,
    LogitsProcessorOutput,
    get_batch_sizes_to_capture,
    get_global_graph_memory_pool,
    model_capture_mode,
    set_global_graph_memory_pool,
    set_is_extend_in_batch,
    set_torch_compile_config,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.input_buffers import ForwardInputBuffers
from sglang.srt.speculative.eagle_info import EagleDraftInput
# assign_new_state_triton：Triton 内核，将当前步的 topk 输出写入下一步缓冲区
from sglang.srt.speculative.multi_layer_eagle_utils import assign_new_state_triton
from sglang.srt.speculative.spec_utils import fast_topk
from sglang.srt.utils import (
    get_available_gpu_memory,
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_sync,
    require_mlp_tp_gather,
)

if TYPE_CHECKING:
    from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
        MultiLayerEagleDraftWorker,
    )


logger = logging.getLogger(__name__)


@dataclass
class MultiLayerEagleDraftExtendInputBuffers(ForwardInputBuffers):
    # Sliced from shared parent buffers
    # 以下字段从父级共享缓冲区按步次偏移切片，各步 Runner 独占其 offset 范围
    input_ids: torch.Tensor           # 当前步草稿 token ids，[num_tokens]
    out_cache_loc: torch.Tensor       # 当前步 KV slot 索引，[num_tokens]
    swa_out_cache_loc: torch.Tensor   # SWA（Sliding Window Attention）KV slot，[num_tokens]
    positions: torch.Tensor           # 位置索引，[num_tokens]
    # Shared from parent
    # 以下字段所有步共享同一块内存（不按步次切片）
    seq_lens: torch.Tensor            # prefix 长度，[bs]
    seq_lens_cpu: torch.Tensor        # CPU 副本，用于 attention 元数据
    req_pool_indices: torch.Tensor    # 请求在 req_to_token 池中的行号，[bs]
    num_accepted_drafts: torch.Tensor # 已接受的草稿 token 数（不含 bonus），[bs]
    num_accepted_tokens: torch.Tensor # 已接受 token 总数（含 bonus），[bs]
    # Per-step buffers
    # 以下字段每步独立分配
    extend_seq_lens: torch.Tensor     # 本步每请求的 extend 长度（= num_tokens_per_bs），[bs]
    extend_start_loc: torch.Tensor    # 本步每请求的 extend 起点（CSR 行偏移），[bs]
    mrope_positions: torch.Tensor     # Multi-rope 位置编码，[3, num_tokens]
    hidden_states: torch.Tensor       # 前序步骤输出的隐藏状态（草稿输入），[num_tokens, hidden]
    next_token_logits_buffer: torch.Tensor    # logits 输出缓冲区，[bs 或 num_tokens, vocab]
    global_num_tokens_gpu: Optional[torch.Tensor]             # DP gather 全局 token 数
    global_num_tokens_for_logprob_gpu: Optional[torch.Tensor] # DP gather logprob token 数


class MultiLayerEagleDraftExtendCudaGraphRunner:
    def __init__(self, eagle_worker: MultiLayerEagleDraftWorker, step: int):
        # Parse args
        # step：当前 MTP 层编号（0-based），决定 num_tokens_per_bs 和 model_runner
        self.step = step
        self.eagle_worker = eagle_worker
        # 每个 MTP 步对应一个独立的 draft model runner
        self.model_runner = model_runner = eagle_worker.mtp_model_runner(self.step)
        # MTP extend 阶段使用 DRAFT_EXTEND_V2 模式（输出全部 token 的 logits）
        self.forward_mode = ForwardMode.DRAFT_EXTEND_V2

        self.graphs = {}
        self.output_buffers = {}
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        # DP/TP gather 模式标志（决定 global_num_tokens 缓冲区的形状）
        self.require_gathered_buffer = require_gathered_buffer(model_runner.server_args)
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)
        self.require_mlp_sync = require_mlp_sync(model_runner.server_args)
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)
        self.tp_size = self.model_runner.tp_size
        self.dp_size = model_runner.server_args.dp_size
        self.enable_pdmux = model_runner.server_args.enable_pdmux
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps
        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
        )
        self.topk = model_runner.server_args.speculative_eagle_topk
        self.enable_profile_cuda_graph = (
            model_runner.server_args.enable_profile_cuda_graph
        )
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(model_runner)
        self.padded_static_len = -1
        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()

        # For Attention Backend
        # num_tokens_per_bs：每条请求在当前步需要处理的 token 数
        # = speculative_num_steps + 1（当前 token）+ step（前步累积的额外 token）
        self.num_tokens_per_bs = self.speculative_num_steps + 1 + step
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs

        # 初始化 attention backend 的 CUDA Graph 状态（预分配 KV 索引缓冲区）
        self.eagle_worker.draft_extend_attn_backend_list[
            self.step
        ].init_cuda_graph_state(self.max_bs, self.max_num_token)
        # seq_len_fill_value：padding 时用于填充 seq_lens 的占位值
        self.seq_len_fill_value = self.eagle_worker.draft_extend_attn_backend_list[
            self.step
        ].get_cuda_graph_seq_len_fill_value()

    def init_buffers_and_capture(
        self,
        cuda_graph_buffers,  # 由 MultiLayerEagleMultiStepDraftExtendCudaGraphRunner 分配的共享缓冲区
        offset,              # 当前步在共享 input_ids/out_cache_loc 等缓冲区中的起始偏移
        next_cuda_graph_runner,  # 下一步的 Runner（用于 assign_new_state_triton 传递输出）
    ):
        self.next_cuda_graph_runner = next_cuda_graph_runner
        seq_lens_cpu = cuda_graph_buffers["seq_lens_cpu"]
        # extend_seq_lens_cpu：CPU 端列表，捕获时每请求均为 num_tokens_per_bs
        self.extend_seq_lens_cpu = [self.num_tokens_per_bs] * self.max_bs

        if self.enable_torch_compile:
            set_torch_compile_config()

        # Graph inputs
        with torch.device(self.model_runner.device):
            # sliced buffers
            # slice according to max_num_token
            # 从共享缓冲区按偏移和最大 token 数切片，各步独占各自的缓冲区段
            input_ids = cuda_graph_buffers["input_ids"][
                offset : offset + self.max_num_token
            ]
            out_cache_loc = cuda_graph_buffers["out_cache_loc"][
                offset : offset + self.max_num_token
            ]
            swa_out_cache_loc = cuda_graph_buffers["swa_out_cache_loc"][
                offset : offset + self.max_num_token
            ]
            positions = cuda_graph_buffers["positions"][
                offset : offset + self.max_num_token
            ]

            # shared states
            # seq_lens/req_pool_indices 等所有步共享，不按偏移切片
            seq_lens = cuda_graph_buffers["seq_lens"]
            req_pool_indices = cuda_graph_buffers["req_pool_indices"]
            num_accepted_drafts = cuda_graph_buffers["num_accepted_drafts"]
            num_accepted_tokens = cuda_graph_buffers["num_accepted_tokens"]

            # extend_seq_lens：GPU 端固定为 num_tokens_per_bs，用于 attention 元数据
            extend_seq_lens = torch.full(
                (self.max_bs,),
                self.num_tokens_per_bs,
                dtype=torch.int32,
            )
            # extend_start_loc：每请求 extend 起点，步长 = num_tokens_per_bs
            extend_start_loc = torch.arange(
                0,
                self.max_bs * self.num_tokens_per_bs,
                step=self.num_tokens_per_bs,
                dtype=torch.int32,
            )

            mrope_positions = torch.zeros((3, self.max_num_token), dtype=torch.int64)

            # hidden_states：接收前步 MTP 层或目标模型的隐藏状态输出
            hidden_states = torch.zeros(
                (self.max_num_token, self.model_runner.model_config.hidden_size),
                dtype=self.model_runner.dtype,
            )

            if self.require_gathered_buffer:
                if self.require_mlp_tp_gather:
                    # MLP TP gather：global_num_tokens shape = [dp_size]
                    global_num_tokens_gpu = torch.zeros(
                        (self.dp_size,), dtype=torch.int32
                    )
                    global_num_tokens_for_logprob_gpu = torch.zeros(
                        (self.dp_size,), dtype=torch.int32
                    )
                else:
                    assert self.require_attn_tp_gather
                    # Attention TP gather：shape = [1]
                    global_num_tokens_gpu = torch.zeros((1,), dtype=torch.int32)
                    global_num_tokens_for_logprob_gpu = torch.zeros(
                        (1,), dtype=torch.int32
                    )
            else:
                global_num_tokens_gpu = None
                global_num_tokens_for_logprob_gpu = None

            # 按草稿模型的词汇表大小分配 logits 缓冲区
            if hasattr(
                self.model_runner.model_config.hf_config, "draft_vocab_size"
            ):  # llama_eagle
                vocab_size = self.model_runner.model_config.hf_config.draft_vocab_size
            elif hasattr(
                self.model_runner.model_config.hf_config, "hot_vocab_size"
            ):  # llama_eagle3
                vocab_size = self.model_runner.model_config.hf_config.hot_vocab_size
            else:
                vocab_size = self.model_runner.model_config.vocab_size

            # DRAFT_EXTEND_V2 模式：对所有 num_tokens 行输出 logits（非仅 bs 行）
            next_token_logits_buffer = torch.zeros(
                (
                    (
                        self.max_bs * self.num_tokens_per_bs
                        if self.forward_mode == ForwardMode.DRAFT_EXTEND_V2
                        else self.max_bs
                    ),
                    vocab_size,
                ),
                dtype=torch.float,
            )

        self.buffers = MultiLayerEagleDraftExtendInputBuffers(
            input_ids=input_ids,
            out_cache_loc=out_cache_loc,
            swa_out_cache_loc=swa_out_cache_loc,
            positions=positions,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            req_pool_indices=req_pool_indices,
            num_accepted_drafts=num_accepted_drafts,
            num_accepted_tokens=num_accepted_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_start_loc=extend_start_loc,
            mrope_positions=mrope_positions,
            hidden_states=hidden_states,
            next_token_logits_buffer=next_token_logits_buffer,
            global_num_tokens_gpu=global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob_gpu,
        )

        # Capture
        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture cuda graph failed: {e}\n{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )

    def can_run(self, forward_batch: ForwardBatch):
        # DP gather 路径：以 global_num_tokens_cpu 的最大值除以 num_tokens_per_bs 作为有效 bs
        if self.require_mlp_tp_gather:
            cuda_graph_bs = (
                max(forward_batch.global_num_tokens_cpu) // self.num_tokens_per_bs
                if self.model_runner.spec_algorithm.is_eagle()
                else max(forward_batch.global_num_tokens_cpu)
            )
        else:
            # 非 gather 路径：直接用 seq_lens 的元素数作为 bs
            cuda_graph_bs = forward_batch.seq_lens.numel()

        is_bs_supported = (
            cuda_graph_bs in self.graphs        # disable_padding：精确匹配
            if self.disable_padding
            else cuda_graph_bs <= self.max_bs   # 允许 padding：bs 不超过最大捕获 bs
        )

        # DP 同步：额外要求 forward_batch.can_run_dp_cuda_graph 为真
        if self.require_mlp_sync:
            is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph

        return is_bs_supported

    def _create_graph(self):
        return torch.cuda.CUDAGraph()

    def _capture_init(self, run_once_fn):
        # 捕获前执行 2 次 warmup，确保内核已 JIT 编译完成
        for _ in range(2):
            torch.cuda.synchronize()
            self.model_runner.tp_group.barrier()
            run_once_fn()

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        # 使用 torch.cuda.graph 上下文捕获 GPU 计算图
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            out = run_once_fn()
        return out

    def _replay(self, forward_batch: ForwardBatch):
        # 回放已捕获的 CUDA Graph（使用当前 bs 对应的 graph 对象）
        self.graphs[self.bs].replay()

    def capture(self):
        # 委托给 CudaGraphRunner 的通用 capture 流程
        CudaGraphRunner.capture(self)

    def get_forward_batch(self, bs: int) -> ForwardBatch:
        buffers = self.buffers
        num_tokens = bs * self.num_tokens_per_bs

        # Graph inputs
        # 按 bs 和 num_tokens 切片缓冲区，生成捕获用的 ForwardBatch
        input_ids = buffers.input_ids[:num_tokens]
        req_pool_indices = buffers.req_pool_indices[:bs]
        seq_lens = buffers.seq_lens[:bs]
        seq_lens_cpu = buffers.seq_lens_cpu[:bs]
        extend_seq_lens = buffers.extend_seq_lens[:bs]
        extend_seq_lens_cpu = self.extend_seq_lens_cpu[:bs]
        extend_start_loc = buffers.extend_start_loc[:bs]
        num_accepted_drafts = buffers.num_accepted_drafts[:bs]
        num_accepted_tokens = buffers.num_accepted_tokens[:bs]
        out_cache_loc = buffers.out_cache_loc[:num_tokens]
        positions = buffers.positions[:num_tokens]
        mrope_positions = buffers.mrope_positions[:, :num_tokens]
        hidden_states = buffers.hidden_states[:num_tokens]
        # DRAFT_EXTEND_V2 输出 num_tokens 行 logits，旧 DRAFT_EXTEND 仅输出 bs 行
        next_token_logits_buffer = buffers.next_token_logits_buffer[
            : bs if self.forward_mode == ForwardMode.DRAFT_EXTEND else num_tokens
        ]

        # 更新 DP gather 的 global_num_tokens 缓冲区
        if self.require_mlp_tp_gather:
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.dp_size,
                    dtype=torch.int32,
                    device=buffers.input_ids.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.dp_size,
                    dtype=torch.int32,
                    device=buffers.input_ids.device,
                )
            )
            global_dp_buffer_len = num_tokens * self.dp_size
        elif self.require_attn_tp_gather:
            # Attention TP gather：global_tokens = [num_tokens]，logprob = [bs]
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=buffers.input_ids.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [bs],
                    dtype=torch.int32,
                    device=buffers.input_ids.device,
                )
            )
            global_dp_buffer_len = num_tokens
        else:
            global_dp_buffer_len = None

        # 构建 EagleDraftInput spec_info，positions 设为 None（由 forward 内部计算）
        spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            num_accepted_drafts=num_accepted_drafts,
            num_accepted_tokens=num_accepted_tokens,
        )
        spec_info.positions = None

        # Forward batch
        forward_batch = ForwardBatch(
            forward_mode=self.forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            next_token_logits_buffer=next_token_logits_buffer,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            return_logprob=False,
            positions=positions,
            mrope_positions=mrope_positions,
            global_num_tokens_gpu=buffers.global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=buffers.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            attn_backend=self.eagle_worker.draft_extend_attn_backend_list[self.step],
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            padded_static_len=self.padded_static_len,
            # added args
            extend_start_loc=extend_start_loc,
            extend_num_tokens=self.num_tokens_per_bs * bs,
            num_token_non_padded_cpu=self.num_tokens_per_bs * bs,
            return_hidden_states_before_norm=True,
        )
        return forward_batch

    def capture_one_batch_size(self, bs: int, forward: Callable, stream_idx: int = 0):
        buffers = self.buffers
        graph = self._create_graph()
        stream = self.stream

        # DeepEP adapter：标记此次捕获包含 extend 序列
        self.deepep_adapter.capture(is_extend_in_batch=True)

        num_tokens = bs * self.num_tokens_per_bs
        forward_batch = self.get_forward_batch(bs)

        # 初始化 attention backend 的捕获元数据（固定形状参数）
        self.eagle_worker.draft_extend_attn_backend_list[
            self.step
        ].init_forward_metadata_capture_cuda_graph(
            bs=bs,
            num_tokens=num_tokens,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            encoder_lens=None,
            forward_mode=self.forward_mode,
            spec_info=forward_batch.spec_info,
        )

        # Run and capture
        def run_once():
            # Clean intermediate result cache for DP attention
            # 每次执行前重置 DP 本地分片缓存，避免 CUDA Graph 记录到错误状态
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(
                forward_batch.global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
            )
            set_is_extend_in_batch(False)

            # Backup two fields, which will be modified in-place in `draft_forward`.
            # 备份会被原地修改的字段，以便在 CUDA Graph 捕获中正确恢复
            output_cache_loc_backup = forward_batch.out_cache_loc
            hidden_states_backup = forward_batch.spec_info.hidden_states

            ret = self.model_runner.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
            )

            # Chain-style MTP: overwrite buffers.hidden_states with the draft model's
            # output (hidden_states_before_norm) so that assign_new_state_triton
            # propagates each MTP layer's own output to the next MTP layer,
            # rather than always feeding the target model's hidden states.
            # 链式 MTP：将本步草稿模型的 hidden_states_before_norm 写入缓冲区，
            # 使下一步 MTP 层接收当前草稿层的输出而非始终使用目标模型的隐藏状态
            if (
                self.eagle_worker.chain_mtp_hidden_states
                and ret.hidden_states is not None
            ):
                buffers.hidden_states[:num_tokens].copy_(ret.hidden_states[:num_tokens])

            # num_accepted_drafts is drafts-only; the last accepted draft sits at index
            # `num_accepted_drafts` within the (current_token + drafts) slot range.
            # select_index：从 logits 矩阵中为每条请求选取最后一个被接受的草稿 token 行
            select_index = (
                torch.arange(bs, device=self.model_runner.device)
                * (self.speculative_num_draft_tokens + self.step)
                + buffers.num_accepted_drafts[:bs]
                + self.step
            )

            # softmax 后取 topk，作为下一步草稿的候选 token
            probs = torch.softmax(ret.next_token_logits[select_index], dim=-1)
            ret.topk_p, ret.topk_index = fast_topk(probs, self.topk, dim=-1)

            if self.next_cuda_graph_runner is not None:
                next_buffers = self.next_cuda_graph_runner.buffers
                # rejected drafts = proposed drafts - accepted drafts.
                # speculative_num_draft_tokens includes the current-token slot, so -1.
                # padding_lens：当前步被拒绝的草稿数（用于 assign_new_state_triton 的偏移计算）
                padding_lens = (
                    self.speculative_num_draft_tokens - 1
                ) - buffers.num_accepted_drafts[:bs]
                # 将当前步的 topk 输出写入下一步 Runner 的 input_ids/hidden_states 缓冲区
                assign_new_state_triton(
                    ret.topk_index,
                    buffers.input_ids,
                    buffers.positions,
                    buffers.hidden_states,
                    buffers.out_cache_loc,
                    buffers.extend_seq_lens,
                    buffers.extend_start_loc,
                    next_buffers.input_ids,
                    next_buffers.positions,
                    next_buffers.hidden_states,
                    next_buffers.out_cache_loc,
                    next_buffers.extend_seq_lens,
                    next_buffers.extend_start_loc,
                    next_buffers.seq_lens,
                    padding_lens,
                    forward_batch.batch_size,
                    self.step,
                    forward_batch.req_pool_indices,
                    forward_batch.req_to_token_pool.req_to_token,
                    self.eagle_worker.req_to_hidden_states_pool,
                )
                # 同步更新下一步的 swa_out_cache_loc（full→swa slot 转换）
                next_buffers.swa_out_cache_loc.copy_(
                    self.model_runner.token_to_kv_pool.translate_loc_from_full_to_swa(
                        next_buffers.out_cache_loc
                    )
                )

            # 恢复被原地修改的字段（CUDA Graph 回放时这些字段由 replay() 正确填充）
            forward_batch.out_cache_loc = output_cache_loc_backup
            forward_batch.spec_info.hidden_states = hidden_states_backup
            return ret

        self._capture_init(run_once)

        out = self._capture_graph(
            graph, get_global_graph_memory_pool(), stream, run_once
        )

        set_global_graph_memory_pool(graph.pool())
        return graph, out

    def init_replay_state(
        self, forward_batch: ForwardBatch, bs: int, raw_bs: int, num_tokens: int
    ):
        buffers = self.buffers
        # Common inputs
        # 将真实输入数据复制到 CUDA Graph 的固定缓冲区，供回放使用
        buffers.input_ids[:num_tokens].copy_(forward_batch.input_ids)
        buffers.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)
        if forward_batch.extend_seq_lens is not None:
            buffers.extend_seq_lens[:raw_bs].copy_(forward_batch.extend_seq_lens)
            buffers.extend_start_loc[:raw_bs].copy_(forward_batch.extend_start_loc)
        buffers.out_cache_loc[:num_tokens].copy_(forward_batch.out_cache_loc)
        buffers.positions[:num_tokens].copy_(forward_batch.positions)
        # 仅在隐藏状态维度匹配时复制（多模态等场景可能维度不同）
        if (
            forward_batch.spec_info.hidden_states.shape[1]
            == buffers.hidden_states.shape[1]
        ):
            buffers.hidden_states[:num_tokens].copy_(
                forward_batch.spec_info.hidden_states
            )
        if forward_batch.spec_info.num_accepted_drafts is not None:
            buffers.num_accepted_drafts[:raw_bs].copy_(
                forward_batch.spec_info.num_accepted_drafts
            )
            buffers.num_accepted_tokens[:raw_bs].copy_(
                forward_batch.spec_info.num_accepted_tokens
            )
        buffers.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)

        # 若需要 padding（bs > raw_bs），先填充占位值再复制真实值
        if forward_batch.seq_lens_cpu is not None:
            if bs != raw_bs:
                buffers.seq_lens_cpu.fill_(self.seq_len_fill_value)
            buffers.seq_lens_cpu[:raw_bs].copy_(forward_batch.seq_lens_cpu)

        if forward_batch.extend_seq_lens_cpu is not None:
            self.extend_seq_lens_cpu[:raw_bs] = forward_batch.extend_seq_lens_cpu

    def replay(self, forward_batch: ForwardBatch, init_state: bool = True):
        assert forward_batch.out_cache_loc is not None
        self.deepep_adapter.replay()
        buffers = self.buffers

        # batch_size and num_seqs can be different in case there are finished examples
        # in the batch, which will not be counted as num_seqs
        raw_bs = forward_batch.batch_size
        num_tokens = raw_bs * self.num_tokens_per_bs
        # num_tokens = forward_batch.input_ids.shape[0]
        # 用 bisect 查找最小的满足条件的捕获 bs（向上取整以支持 padding）
        if self.require_mlp_tp_gather:
            max_batch_size = max(forward_batch.original_global_num_tokens_cpu)
            index = bisect.bisect_left(self.capture_bs, max_batch_size)
        else:
            index = bisect.bisect_left(self.capture_bs, raw_bs)

        bs = self.capture_bs[index]

        if init_state:
            self.init_replay_state(forward_batch, bs, raw_bs, num_tokens)

        # 填充 global_num_tokens（CUDA Graph 以 bs 为单位捕获，需用 padded bs 填充）
        if self.require_gathered_buffer:
            buffers.global_num_tokens_gpu.fill_(bs * self.num_tokens_per_bs)
            buffers.global_num_tokens_for_logprob_gpu.fill_(bs * self.num_tokens_per_bs)

        # 将缓冲区视图绑定到 forward_batch.spec_info，使 CUDA Graph 回放后结果直接可读
        forward_batch.spec_info.hidden_states = buffers.hidden_states[:num_tokens]
        forward_batch.spec_info.num_accepted_drafts = buffers.num_accepted_drafts[:bs]
        forward_batch.spec_info.num_accepted_tokens = buffers.num_accepted_tokens[:bs]
        forward_batch.spec_info.num_tokens_per_req = self.num_tokens_per_bs
        forward_batch.spec_info.num_tokens_for_logprob_per_req = 1
        forward_batch.spec_info.positions = buffers.positions[:num_tokens]
        forward_batch.spec_info.extend_seq_lens_tensor = buffers.extend_seq_lens[:bs]

        # 使用 padded bs 初始化 attention backend 的回放元数据
        self.eagle_worker.draft_extend_attn_backend_list[
            self.step
        ].init_forward_metadata_replay_cuda_graph(
            bs=bs,
            req_pool_indices=buffers.req_pool_indices,
            seq_lens=buffers.seq_lens,
            seq_lens_sum=forward_batch.seq_lens_sum
            + (bs - raw_bs) * self.seq_len_fill_value,
            encoder_lens=None,
            forward_mode=self.forward_mode,
            spec_info=forward_batch.spec_info,
            seq_lens_cpu=buffers.seq_lens_cpu,
        )

        # Replay
        self.raw_bs = raw_bs
        self.bs = bs
        self._replay(forward_batch)
        out = self.output_buffers[bs]

        if self.forward_mode == ForwardMode.DRAFT_EXTEND_V2:
            # DRAFT_EXTEND_V2: all tokens calculations whether accepted or not.
            # DRAFT_EXTEND_V2 输出所有 num_tokens 行，无需额外截断
            unpadding_bs = num_tokens
        elif bs != raw_bs:
            # 若 bs 做了 padding，需将 num_accepted_drafts/tokens 切回 raw_bs
            forward_batch.spec_info.num_accepted_drafts = buffers.num_accepted_drafts[
                :raw_bs
            ]
            forward_batch.spec_info.num_accepted_tokens = buffers.num_accepted_tokens[
                :raw_bs
            ]
            unpadding_bs = raw_bs
        else:
            unpadding_bs = None

        # 截断 logits/hidden_states 输出，去除 padding 行
        if unpadding_bs is not None:
            out_copy = out
            out = LogitsProcessorOutput(
                next_token_logits=out.next_token_logits[:unpadding_bs],
                hidden_states=out.hidden_states[:unpadding_bs],
            )
            out.topk_p = out_copy.topk_p[:raw_bs]
            out.topk_index = out_copy.topk_index[:raw_bs]
        return out


class MultiLayerEagleMultiStepDraftExtendCudaGraphRunner:
    def __init__(self, eagle_worker: MultiLayerEagleDraftWorker):
        # 顶层编排器：管理所有 MTP 步的 extend Runner，统一分配共享缓冲区并协调捕获顺序
        self.eagle_worker = eagle_worker
        self.device = eagle_worker.device
        self.gpu_id = eagle_worker.gpu_id
        self.speculative_num_steps = eagle_worker.speculative_num_steps
        self.draft_extend_attn_backend_list = (
            eagle_worker.draft_extend_attn_backend_list
        )

        self.runners = []
        self.cuda_graph_buffers = {}
        # seq_len_fill_value 和 max_bs 在 _init_and_capture 中更新
        self.seq_len_fill_value = 1
        self.max_bs = 1
        # offsets[step] 记录第 step 步在共享 input_ids 缓冲区中的起始偏移
        self.offsets = [0]

        self._init_and_capture()

    def _init_and_capture(self):
        if self.eagle_worker.server_args.disable_cuda_graph:
            # 禁用 CUDA Graph 时，所有步的 Runner 设为 None（回退到 eager 模式）
            self.runners = [None] * self.speculative_num_steps
            return

        self.runners: List[Optional[MultiLayerEagleDraftExtendCudaGraphRunner]] = []
        buffer_len_list: List[int] = []

        # 1. Capture loop
        # 第一遍：初始化各步 Runner，记录每步需要的缓冲区长度和偏移
        for step in range(self.speculative_num_steps):
            if self.draft_extend_attn_backend_list[step]:
                runner = MultiLayerEagleDraftExtendCudaGraphRunner(
                    self.eagle_worker, step
                )
                self.runners.append(runner)

                self.seq_len_fill_value = runner.seq_len_fill_value
                self.max_bs = runner.max_bs
                buffer_len_list.append(runner.max_num_token)
                # offsets 采用前缀和方式：offsets[step+1] = offsets[step] + step 的 max_num_token
                self.offsets.append(self.offsets[-1] + runner.max_num_token)
            else:
                self.runners.append(None)

        # 2. Allocate buffers
        # seq_lens_cpu 使用 CPU tensor，初始化为 seq_len_fill_value
        self.cuda_graph_buffers["seq_lens_cpu"] = torch.full(
            (self.max_bs,),
            self.seq_len_fill_value,
            dtype=torch.int32,
        )

        with torch.device(self.device):
            # Sliced buffers
            # 三类按步次切片的缓冲区：总长度为各步 max_num_token 之和
            self.cuda_graph_buffers["input_ids"] = torch.zeros(
                (self.offsets[-1],), dtype=torch.int64
            )
            # out_cache_loc 初始化为 1（非零，避免 KV 缓存地址越界）
            self.cuda_graph_buffers["out_cache_loc"] = torch.ones(
                (self.offsets[-1],), dtype=torch.int64
            )
            self.cuda_graph_buffers["swa_out_cache_loc"] = torch.ones(
                (self.offsets[-1],), dtype=torch.int64
            )
            self.cuda_graph_buffers["positions"] = torch.zeros(
                (self.offsets[-1],), dtype=torch.int64
            )

            # Shared states
            # 以下缓冲区所有步共享，不按步次切片
            self.cuda_graph_buffers["seq_lens"] = torch.full(
                (self.max_bs,),
                self.seq_len_fill_value,
                dtype=torch.int32,
            )
            self.cuda_graph_buffers["req_pool_indices"] = torch.zeros(
                (self.max_bs,), dtype=torch.int64
            )
            # num_accepted_drafts/tokens 初始化为 1（确保捕获时 select_index 合法）
            self.cuda_graph_buffers["num_accepted_drafts"] = torch.full(
                (self.max_bs,), 1, dtype=torch.int32
            )
            self.cuda_graph_buffers["num_accepted_tokens"] = torch.full(
                (self.max_bs,), 1, dtype=torch.int32
            )

        # 第二遍：逆序捕获（step N-1 先捕获），以便在 assign_new_state_triton 时
        # 下一步的 Runner 已经初始化完毕
        for step in range(self.speculative_num_steps - 1, -1, -1):
            if self.runners[step] is not None:
                tic = time.perf_counter()
                before_mem = get_available_gpu_memory(self.device, self.gpu_id)
                logger.info(
                    f"Capture draft extend cuda graph begin (step {step}). This can take up to several minutes. avail mem={before_mem:.2f} GB"
                )

                self.runners[step].init_buffers_and_capture(
                    self.cuda_graph_buffers,
                    self.offsets[step],
                    (
                        # 将下一步 Runner 传入，用于 assign_new_state_triton 跨步传递状态
                        self.runners[step + 1]
                        if step + 1 < self.speculative_num_steps
                        else None
                    ),
                )

                after_mem = get_available_gpu_memory(self.device, self.gpu_id)
                logger.info(
                    f"Capture draft extend cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
                )

    def reset_buffers(self, forward_batch, batch_result):
        # 每轮 forward 结束后重置共享缓冲区，准备下一轮捕获回放
        self.cuda_graph_buffers["input_ids"].zero_()
        self.cuda_graph_buffers["seq_lens"].fill_(self.seq_len_fill_value)
        self.cuda_graph_buffers["out_cache_loc"].zero_()
        self.cuda_graph_buffers["swa_out_cache_loc"].zero_()
        self.cuda_graph_buffers["positions"].zero_()
        # `batch_result.accept_lens` is drafts + bonus.
        # num_accepted_drafts = accept_lens - 1（去除 bonus token）
        bs = forward_batch.batch_size
        self.cuda_graph_buffers["num_accepted_drafts"][:bs].copy_(
            batch_result.accept_lens - 1
        )
        self.cuda_graph_buffers["num_accepted_tokens"][:bs].copy_(
            batch_result.accept_lens
        )

    def get_runner(self, step):
        # 获取指定步的 Runner
        return self.runners[step]

    def get_last_runner(self):
        # 获取最后一步的 Runner（如果存在）
        return self.runners[-1] if self.runners else None

    def can_run(self, forward_batch):
        # 以第一步 Runner 的 can_run 判断为准（所有步共享同一 bs 限制）
        return self.runners[0].can_run(forward_batch)
