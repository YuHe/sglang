from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.layers.dp_attention import DpPaddingMode, set_dp_buffer_len
from sglang.srt.model_executor.cuda_graph_runner import (
    CUDA_GRAPH_CAPTURE_FAILED_MSG,
    CudaGraphRunner,
    DeepEPCudaGraphRunnerAdapter,
    LogitsProcessorOutput,      # 从 cuda_graph_runner 重导出，供 replay 输出截取使用
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
# fast_topk: 高效 top-k 实现，比 torch.topk 更快（用于草稿 token 选取）
from sglang.srt.speculative.spec_utils import fast_topk
from sglang.srt.utils import (
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_sync,
    require_mlp_tp_gather,
)

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_worker import EAGLEWorker


@dataclass
class EagleDraftExtendInputBuffers(ForwardInputBuffers):
    # EAGLE draft extend 阶段的 CUDA Graph 固定输入缓冲区
    # extend 阶段：处理验证后剩余的 token（每个请求 speculative_num_steps + 1 个 token）
    input_ids: torch.Tensor                          # [max_num_token] 输入 token
    req_pool_indices: torch.Tensor                   # [max_bs] 请求索引
    out_cache_loc: torch.Tensor                      # [max_num_token] KV slot 位置
    positions: torch.Tensor                          # [max_num_token] 位置编码
    mrope_positions: torch.Tensor                    # [3, max_num_token] 多模态 RoPE
    hidden_states: torch.Tensor                      # [max_num_token, spec_hidden_size] 目标隐状态
    seq_lens: torch.Tensor                           # [max_bs] 序列长度
    seq_lens_cpu: torch.Tensor                       # [max_bs] CPU 序列长度
    extend_seq_lens: torch.Tensor                    # [max_bs] extend 步数（speculative_num_steps+1）
    num_accepted_drafts: torch.Tensor                # [max_bs] 本轮每请求接受的草稿 token 数
    num_accepted_tokens: torch.Tensor                # [max_bs] = num_accepted_drafts + 1（含 bonus）
    next_token_logits_buffer: torch.Tensor           # logits 输出缓冲区（V1: [bs, vocab], V2: [num_tokens, vocab]）
    global_num_tokens_gpu: Optional[torch.Tensor]
    global_num_tokens_for_logprob_gpu: Optional[torch.Tensor]


class EAGLEDraftExtendCudaGraphRunner:
    def __init__(
        self,
        eagle_worker: EAGLEWorker,
        *,
        draft_extend_attn_backend=None,
        speculative_num_steps: Optional[int] = None,
    ):
        # Parse args
        self.eagle_worker = eagle_worker
        if not hasattr(eagle_worker, "model_runner"):
            # V2: EagleDraftWorker（spec-v2 overlap scheduler）
            self.model_runner = model_runner = eagle_worker.draft_runner
            # DRAFT_EXTEND_V2：所有 token 都输出 logits（用于后续 topk 选取）
            self.forward_mode = ForwardMode.DRAFT_EXTEND_V2
        else:
            # V1: 标准 EAGLEWorker
            self.model_runner = model_runner = eagle_worker.model_runner
            # DRAFT_EXTEND：仅最后一个 token 输出 logits（草稿剪枝）
            self.forward_mode = ForwardMode.DRAFT_EXTEND

        self.graphs = {}
        self.output_buffers = {}
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.require_gathered_buffer = require_gathered_buffer(model_runner.server_args)
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)
        self.require_mlp_sync = require_mlp_sync(model_runner.server_args)
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)
        self.tp_size = self.model_runner.tp_size
        self.dp_size = self.model_runner.dp_size
        self.speculative_num_steps = (
            model_runner.server_args.speculative_num_steps
            if speculative_num_steps is None
            else speculative_num_steps
        )
        self.topk = model_runner.server_args.speculative_eagle_topk
        self.draft_extend_attn_backend = (
            draft_extend_attn_backend or eagle_worker.draft_extend_attn_backend
        )
        self.enable_profile_cuda_graph = (
            model_runner.server_args.enable_profile_cuda_graph
        )
        self.enable_pdmux = False
        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()

        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(model_runner)
        # padded_static_len: 静态 padding 长度（-1 表示未启用，由后端自动确定）
        self.padded_static_len = -1

        # extend 阶段每个请求的 token 数 = speculative_num_steps + 1（所有草稿步 + 1 个新 token）
        self.num_tokens_per_bs = self.speculative_num_steps + 1
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs

        # 初始化 draft extend 注意力后端的 CUDA Graph 状态
        self.draft_extend_attn_backend.init_cuda_graph_state(
            self.max_bs, self.max_num_token
        )
        self.seq_len_fill_value = (
            self.draft_extend_attn_backend.get_cuda_graph_seq_len_fill_value()
        )
        seq_lens_cpu = torch.full(
            (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
        )
        # extend_seq_lens_cpu 固定为 num_tokens_per_bs（每请求相同 extend 长度）
        self.extend_seq_lens_cpu = [self.num_tokens_per_bs] * self.max_bs

        if self.enable_torch_compile:
            set_torch_compile_config()

        # 预分配 CUDA Graph 输入缓冲区
        with torch.device(model_runner.device):
            input_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)
            req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int64)
            # out_cache_loc 初始化为 1（避免写入 slot 0 产生未定义行为）
            out_cache_loc = torch.ones(
                (self.max_num_token,), dtype=self._cache_loc_dtype()
            )
            positions = torch.zeros((self.max_num_token,), dtype=torch.int64)
            mrope_positions = torch.zeros((3, self.max_num_token), dtype=torch.int64)

            if (
                self.eagle_worker.speculative_algorithm.is_eagle3()
                and self.eagle_worker.eagle_use_aux_hidden_state
            ):
                # EAGLE3 + 辅助隐状态：hidden_size 为目标隐状态 * 3（三阶段特征拼接）
                hidden_states = torch.zeros(
                    (
                        self.max_num_token,
                        (
                            self.model_runner.model_config.hf_config.target_hidden_size
                            * 3
                            if hasattr(
                                self.model_runner.model_config.hf_config,
                                "target_hidden_size",
                            )
                            else self.model_runner.model_config.hidden_size * 3
                        ),
                    ),
                    dtype=self.model_runner.dtype,
                )
            else:
                # 标准 EAGLE：hidden_size = spec_hidden_size
                hidden_states = torch.zeros(
                    (
                        self.max_num_token,
                        self.model_runner.model_config.spec_hidden_size,
                    ),
                    dtype=self.model_runner.dtype,
                )
            # 重新获取目标模型注意力后端的 fill_value（可能与 extend 后端不同）
            self.seq_len_fill_value = (
                self.model_runner.attn_backend.get_cuda_graph_seq_len_fill_value()
            )
            seq_lens = torch.full(
                (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
            )
            # extend_seq_lens 和接受数统计初始化为 num_tokens_per_bs（全部接受的默认情况）
            extend_seq_lens = torch.full(
                (self.max_bs,), self.num_tokens_per_bs, dtype=torch.int32
            )
            num_accepted_drafts = torch.full(
                (self.max_bs,), self.num_tokens_per_bs, dtype=torch.int32
            )
            num_accepted_tokens = torch.full(
                (self.max_bs,), self.num_tokens_per_bs, dtype=torch.int32
            )

            if self.require_gathered_buffer:
                if self.require_mlp_tp_gather:
                    global_num_tokens_gpu = torch.zeros(
                        (self.dp_size,), dtype=torch.int32
                    )
                    global_num_tokens_for_logprob_gpu = torch.zeros(
                        (self.dp_size,), dtype=torch.int32
                    )
                else:
                    assert self.require_attn_tp_gather
                    global_num_tokens_gpu = torch.zeros((1,), dtype=torch.int32)
                    global_num_tokens_for_logprob_gpu = torch.zeros(
                        (1,), dtype=torch.int32
                    )
            else:
                global_num_tokens_gpu = None
                global_num_tokens_for_logprob_gpu = None

            # 草稿模型词表大小可能与目标模型不同（llama_eagle 使用 draft_vocab_size）
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

            # next_token_logits_buffer: V1 模式仅保存最后一个 token 的 logits（bs, vocab）
            # V2 模式保存所有 token 的 logits（num_tokens, vocab）
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

        self.buffers = EagleDraftExtendInputBuffers(
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            out_cache_loc=out_cache_loc,
            positions=positions,
            mrope_positions=mrope_positions,
            hidden_states=hidden_states,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            extend_seq_lens=extend_seq_lens,
            num_accepted_drafts=num_accepted_drafts,
            num_accepted_tokens=num_accepted_tokens,
            next_token_logits_buffer=next_token_logits_buffer,
            global_num_tokens_gpu=global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob_gpu,
        )
        self.buffers.share_buffers()

        # 捕获 CUDA Graph（在 model_capture_mode 上下文中执行）
        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture cuda graph failed: {e}\n{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )

    def can_run(self, forward_batch: ForwardBatch):
        # 判断当前 extend batch 是否可以使用 CUDA Graph replay
        if self.require_mlp_tp_gather:
            cuda_graph_bs = (
                max(forward_batch.global_num_tokens_cpu) // self.num_tokens_per_bs
                if self.model_runner.spec_algorithm.is_eagle()
                or self.model_runner.spec_algorithm.is_standalone()
                else max(forward_batch.global_num_tokens_cpu)
            )
        else:
            # extend 模式用 seq_lens.numel() 而非 batch_size（更精确）
            cuda_graph_bs = forward_batch.seq_lens.numel()

        is_bs_supported = (
            cuda_graph_bs in self.graphs
            if self.disable_padding
            else cuda_graph_bs <= self.max_bs
        )

        if self.require_mlp_sync:
            is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph

        return is_bs_supported

    def _create_graph(self):
        return torch.cuda.CUDAGraph()

    def _cache_loc_dtype(self):
        return torch.int64

    def _capture_init(self, run_once_fn):
        # 捕获前 warmup 两次
        for _ in range(2):
            torch.cuda.synchronize()
            self.model_runner.tp_group.barrier()
            run_once_fn()

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            out = run_once_fn()
        return out

    def _replay(self, forward_batch: ForwardBatch):
        self.graphs[self.bs].replay()

    def capture(self):
        CudaGraphRunner.capture(self)

    def capture_one_batch_size(self, bs: int, forward: Callable, stream_idx: int = 0):
        buffers = self.buffers
        graph = self._create_graph()
        stream = self.stream
        num_tokens = bs * self.num_tokens_per_bs

        # 切出当前 batch size 对应的缓冲区视图
        input_ids = buffers.input_ids[:num_tokens]
        req_pool_indices = buffers.req_pool_indices[:bs]
        seq_lens = buffers.seq_lens[:bs]
        seq_lens_cpu = buffers.seq_lens_cpu[:bs]
        extend_seq_lens = buffers.extend_seq_lens[:bs]
        extend_seq_lens_cpu = self.extend_seq_lens_cpu[:bs]
        out_cache_loc = buffers.out_cache_loc[:num_tokens]
        positions = buffers.positions[:num_tokens]
        mrope_positions = buffers.mrope_positions[:, :num_tokens]
        # hidden_states 每个 token 都有（extend 模式不同于 decode 的只有一个）
        hidden_states = buffers.hidden_states[:num_tokens]
        num_accepted_drafts = buffers.num_accepted_drafts[:bs]
        num_accepted_tokens = buffers.num_accepted_tokens[:bs]
        # V1 (DRAFT_EXTEND): pruned_states = bs (last token per seq)
        # V2 (DRAFT_EXTEND_V2): pruned_states = num_tokens (all tokens)
        next_token_logits_buffer = buffers.next_token_logits_buffer[
            : bs if self.forward_mode == ForwardMode.DRAFT_EXTEND else num_tokens
        ]

        # logprob 计算的 token 数量
        num_tokens_for_logprob = (
            num_tokens if self.forward_mode.is_draft_extend_v2() else bs
        )

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
                    [num_tokens_for_logprob] * self.dp_size,
                    dtype=torch.int32,
                    device=buffers.input_ids.device,
                )
            )
            global_dp_buffer_len = num_tokens * self.dp_size
        elif self.require_attn_tp_gather:
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=buffers.input_ids.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens_for_logprob],
                    dtype=torch.int32,
                    device=buffers.input_ids.device,
                )
            )
            global_dp_buffer_len = num_tokens
        else:
            global_dp_buffer_len = None

        # 构造草稿 extend 输入（hidden_states 对应各 token 的目标层特征）
        spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            num_accepted_drafts=num_accepted_drafts,
            num_accepted_tokens=num_accepted_tokens,
        )
        # positions 在 draft extend 中通过 spec_info 传递（非 forward_batch 顶层字段）
        spec_info.positions = None

        self.deepep_adapter.capture(is_extend_in_batch=True)

        # 构造 ForwardBatch（extend 模式：多 token per request）
        forward_batch = ForwardBatch(
            forward_mode=self.forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            next_token_logits_buffer=next_token_logits_buffer,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
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
            # LAST 模式：捕获最后一层的隐状态（草稿模型输入）
            capture_hidden_mode=CaptureHiddenMode.LAST,
            attn_backend=self.draft_extend_attn_backend,
            padded_static_len=self.padded_static_len,
        )

        # 初始化注意力后端的 extend CUDA Graph 捕获元数据
        self.draft_extend_attn_backend.init_forward_metadata_capture_cuda_graph(
            bs=bs,
            num_tokens=num_tokens,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            encoder_lens=None,
            forward_mode=self.forward_mode,
            spec_info=spec_info,
        )

        # 定义单次前向函数（draft extend 阶段直接调用模型 forward）
        def run_once():
            # 每次重置 DP 注意力的临时缓存
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(
                global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
            )
            set_is_extend_in_batch(False)

            # 备份会被就地修改的字段
            output_cache_loc_backup = forward_batch.out_cache_loc
            hidden_states_backup = forward_batch.spec_info.hidden_states

            ret = self.model_runner.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
            )
            # 对输出 logits 执行 softmax + topk，得到草稿 token 候选
            probs = torch.softmax(ret.next_token_logits, dim=-1)
            ret.topk_p, ret.topk_index = fast_topk(probs, self.topk, dim=-1)

            forward_batch.out_cache_loc = output_cache_loc_backup
            forward_batch.spec_info.hidden_states = hidden_states_backup
            return ret

        self._capture_init(run_once)

        out = self._capture_graph(
            graph, get_global_graph_memory_pool(), stream, run_once
        )

        set_global_graph_memory_pool(graph.pool())
        return graph, out

    def replay(self, forward_batch: ForwardBatch):
        assert forward_batch.out_cache_loc is not None
        self.deepep_adapter.replay()
        buffers = self.buffers

        # batch_size and num_seqs can be different in case there are finished examples
        # in the batch, which will not be counted as num_seqs
        # raw_bs: 当前实际 batch size（可能小于已捕获的 batch size）
        raw_bs = forward_batch.batch_size
        # num_tokens: 实际输入 token 数（raw_bs * num_tokens_per_bs 或更少）
        num_tokens = forward_batch.input_ids.shape[0]
        if self.require_mlp_tp_gather:
            max_num_tokens = max(forward_batch.global_num_tokens_cpu)
            max_batch_size = (
                max_num_tokens // self.num_tokens_per_bs
                if self.model_runner.spec_algorithm.is_eagle()
                else max_num_tokens
            )
            index = bisect.bisect_left(self.capture_bs, max_batch_size)
        else:
            index = bisect.bisect_left(self.capture_bs, raw_bs)

        bs = self.capture_bs[index]
        # 若 token 数与 bs * num_tokens_per_bs 不一致（有 padding 或不完整步）
        if bs * self.num_tokens_per_bs != num_tokens:
            buffers.seq_lens.fill_(self.seq_len_fill_value)
            buffers.out_cache_loc.zero_()
            buffers.positions.zero_()
            buffers.num_accepted_drafts.fill_(self.num_tokens_per_bs)
            buffers.num_accepted_tokens.fill_(self.num_tokens_per_bs)
            buffers.extend_seq_lens.fill_(self.num_tokens_per_bs)

        # 将真实 batch 数据拷贝到 CUDA Graph 输入缓冲区
        buffers.input_ids[:num_tokens].copy_(forward_batch.input_ids)
        buffers.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)
        if forward_batch.extend_seq_lens is not None:
            buffers.extend_seq_lens[:raw_bs].copy_(forward_batch.extend_seq_lens)
        else:
            buffers.extend_seq_lens[:raw_bs].fill_(self.num_tokens_per_bs)
        buffers.out_cache_loc[:num_tokens].copy_(forward_batch.out_cache_loc)
        buffers.positions[:num_tokens].copy_(forward_batch.positions)
        # 若 hidden_states 维度匹配（非 EAGLE3 aux 模式），直接拷贝
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

        # TODO(ch-wan): support num_token_non_padded
        if self.require_gathered_buffer:
            buffers.global_num_tokens_gpu.fill_(bs * self.num_tokens_per_bs)
            # V1: pruned_states = bs; V2: pruned_states = num_tokens
            if self.forward_mode.is_draft_extend_v2():
                buffers.global_num_tokens_for_logprob_gpu.fill_(
                    bs * self.num_tokens_per_bs
                )
            else:
                buffers.global_num_tokens_for_logprob_gpu.fill_(bs)

        if forward_batch.seq_lens_cpu is not None:
            if bs != raw_bs:
                buffers.seq_lens_cpu.fill_(self.seq_len_fill_value)
            buffers.seq_lens_cpu[:raw_bs].copy_(forward_batch.seq_lens_cpu)

        # 同步 extend_seq_lens_cpu 列表（供注意力后端 replay 使用）
        if forward_batch.extend_seq_lens_cpu is not None:
            self.extend_seq_lens_cpu[:raw_bs] = forward_batch.extend_seq_lens_cpu
        else:
            self.extend_seq_lens_cpu[:raw_bs] = [self.num_tokens_per_bs] * raw_bs
        if bs > raw_bs:
            # padding 部分填充默认 extend 长度
            self.extend_seq_lens_cpu[raw_bs:bs] = [self.num_tokens_per_bs] * (
                bs - raw_bs
            )
        forward_batch.spec_info.extend_seq_lens_cpu = list(
            self.extend_seq_lens_cpu[:bs]
        )
        forward_batch.spec_info.extend_seq_lens_tensor = buffers.extend_seq_lens[:bs]

        # 若有 padding，更新 spec_info 中的张量视图
        if bs != raw_bs:
            forward_batch.spec_info.positions = buffers.positions[:num_tokens]
            forward_batch.spec_info.num_accepted_drafts = buffers.num_accepted_drafts[
                :bs
            ]
            forward_batch.spec_info.num_accepted_tokens = buffers.num_accepted_tokens[
                :bs
            ]

        # 更新注意力后端的 extend replay 元数据
        self.draft_extend_attn_backend.init_forward_metadata_replay_cuda_graph(
            bs=bs,
            req_pool_indices=buffers.req_pool_indices,
            seq_lens=buffers.seq_lens,
            # seq_lens_sum 加上 padding 请求的虚拟序列长度
            seq_lens_sum=forward_batch.seq_lens_sum
            + (bs - raw_bs) * self.seq_len_fill_value,
            encoder_lens=None,
            forward_mode=self.forward_mode,
            spec_info=forward_batch.spec_info,
            seq_lens_cpu=buffers.seq_lens_cpu,
        )

        # 触发 CUDA Graph replay
        self.raw_bs = raw_bs
        self.bs = bs
        self._replay(forward_batch)
        out = self.output_buffers[bs]

        if self.forward_mode == ForwardMode.DRAFT_EXTEND_V2:
            # DRAFT_EXTEND_V2: all tokens calculations whether accepted or not.
            # V2 模式：截取到实际 token 数（去掉 padding token 的输出）
            unpadding_bs = num_tokens
        elif bs != raw_bs:
            # V1 + padding：恢复 spec_info 中的真实 bs 视图
            forward_batch.spec_info.num_accepted_drafts = buffers.num_accepted_drafts[
                :raw_bs
            ]
            forward_batch.spec_info.num_accepted_tokens = buffers.num_accepted_tokens[
                :raw_bs
            ]
            unpadding_bs = raw_bs
        else:
            unpadding_bs = None

        if unpadding_bs is not None:
            # 截取输出张量，去除 padding 部分
            out_copy = out
            out = LogitsProcessorOutput(
                next_token_logits=out.next_token_logits[:unpadding_bs],
                hidden_states=out.hidden_states[:unpadding_bs],
            )
            out.topk_p = out_copy.topk_p[:unpadding_bs]
            out.topk_index = out_copy.topk_index[:unpadding_bs]
        return out
