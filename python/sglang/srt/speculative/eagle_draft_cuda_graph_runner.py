from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch

# DP 注意力相关：填充模式枚举和 DP buffer 长度设置
from sglang.srt.layers.dp_attention import DpPaddingMode, set_dp_buffer_len
from sglang.srt.model_executor.cuda_graph_runner import (
    CUDA_GRAPH_CAPTURE_FAILED_MSG,     # CUDA Graph 捕获失败时的错误提示
    CudaGraphRunner,                   # 通用 CUDA Graph Runner 基类
    DeepEPCudaGraphRunnerAdapter,      # DeepEP MoE 的 CUDA Graph 适配器
    get_batch_sizes_to_capture,        # 获取需要捕获的 batch size 列表
    get_global_graph_memory_pool,      # 获取全局 CUDA Graph 显存池（跨 batch 共享）
    model_capture_mode,                # 上下文管理器：设置模型为捕获模式
    set_global_graph_memory_pool,      # 设置全局 CUDA Graph 显存池
    set_is_extend_in_batch,            # 设置当前 batch 是否包含 extend 阶段
    set_torch_compile_config,          # 配置 torch.compile 相关选项
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,   # 控制隐状态捕获模式（NULL/LAST/FULL）
    ForwardBatch,        # 模型前向所需的 batch 信息
    ForwardMode,         # 前向模式枚举（DECODE/EXTEND/TARGET_VERIFY 等）
)
from sglang.srt.model_executor.input_buffers import ForwardInputBuffers
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.utils import (
    require_attn_tp_gather,    # 检查是否需要 attention TP gather
    require_gathered_buffer,   # 检查是否需要 DP gathered buffer
    require_mlp_sync,          # 检查是否需要 MLP sync
    require_mlp_tp_gather,     # 检查是否需要 MLP TP gather
)

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_worker import EAGLEWorker


@dataclass
class EagleDraftInputBuffers(ForwardInputBuffers):
    # 草稿模型 CUDA Graph 的固定输入缓冲区，形状在捕获时确定，replay 时复用
    input_ids: torch.Tensor                          # [max_num_token] 草稿 token 输入
    req_pool_indices: torch.Tensor                   # [max_bs] 请求在 pool 中的索引
    out_cache_loc: torch.Tensor                      # [max_num_token * speculative_num_steps] KV slot 位置
    positions: torch.Tensor                          # [max_num_token] 位置编码
    mrope_positions: torch.Tensor                    # [3, max_num_token] 多模态 RoPE 位置
    seq_lens: torch.Tensor                           # [max_bs] 各请求序列长度
    seq_lens_cpu: torch.Tensor                       # [max_bs] CPU 上的序列长度（供调度器使用）
    extend_seq_lens: torch.Tensor                    # [max_bs] extend 阶段长度
    topk_p: torch.Tensor                             # [max_bs, topk] 草稿 token 的 top-k 概率
    topk_index: torch.Tensor                         # [max_bs, topk] 草稿 token 的 top-k 索引
    hidden_states: torch.Tensor                      # [max_bs, spec_hidden_size] 目标模型隐状态输入
    global_num_tokens_gpu: Optional[torch.Tensor]   # [dp_size 或 1] DP 模式下各 rank 的 token 数（GPU 张量）
    global_num_tokens_for_logprob_gpu: Optional[torch.Tensor]  # 同上，用于 logprob 计算


class EAGLEDraftCudaGraphRunner:
    def __init__(
        self,
        eagle_worker: EAGLEWorker,
        *,
        draft_attn_backend=None,
        speculative_num_steps: Optional[int] = None,
    ):
        # Parse args
        self.eagle_worker = eagle_worker
        if not hasattr(eagle_worker, "model_runner"):
            # V2: EagleDraftWorker（spec-v2 overlap scheduler 下的草稿 worker）
            self.model_runner = model_runner = eagle_worker.draft_runner
        else:
            # V1: EAGLEWorker 直接持有 model_runner
            self.model_runner = model_runner = eagle_worker.model_runner
        # graphs[bs]: 每个 batch size 对应的 CUDAGraph 对象
        self.graphs = {}
        # output_buffers[bs]: 每个 batch size 对应的输出缓冲区（草稿 token、scores 等）
        self.output_buffers = {}
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        # disable_padding: 不对 batch size 进行向上取整，仅捕获精确的 batch size
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.require_gathered_buffer = require_gathered_buffer(model_runner.server_args)
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)
        self.require_mlp_sync = require_mlp_sync(model_runner.server_args)
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)
        self.tp_size = self.model_runner.tp_size
        self.dp_size = self.model_runner.dp_size
        # 若未指定步数，从 server_args 读取
        self.speculative_num_steps = (
            model_runner.server_args.speculative_num_steps
            if speculative_num_steps is None
            else speculative_num_steps
        )
        # topk: EAGLE 每步生成的候选 token 数
        self.topk = model_runner.server_args.speculative_eagle_topk
        self.draft_attn_backend = draft_attn_backend or model_runner.draft_attn_backend
        self.enable_profile_cuda_graph = (
            model_runner.server_args.enable_profile_cuda_graph
        )
        self.enable_pdmux = False
        # DeepEP 适配器：处理 MoE expert-parallel 通信在 CUDA Graph 内的特殊行为
        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()

        # 获取需要捕获的 batch size 列表（compile_bs 用于 torch.compile 路径）
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(model_runner)

        # num_tokens_per_bs: EAGLE decode 每个请求对应的 token 数（= topk）
        self.num_tokens_per_bs = self.topk
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs

        # 初始化注意力后端的 CUDA Graph 状态（分配 KV 缓存元数据缓冲区等）
        self.draft_attn_backend.init_cuda_graph_state(self.max_bs, self.max_num_token)
        # 获取 CUDA Graph 捕获时 seq_lens 的填充值（各后端可能不同）
        self.seq_len_fill_value = self.draft_attn_backend.attn_backends[
            0
        ].get_cuda_graph_seq_len_fill_value()
        seq_lens_cpu = torch.full(
            (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
        )
        self.extend_seq_lens_cpu = [self.seq_len_fill_value] * self.max_bs

        if self.enable_torch_compile:
            set_torch_compile_config()

        # 预分配所有 CUDA Graph 输入缓冲区（在 GPU 上，形状固定为最大值）
        with torch.device(model_runner.device):
            input_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)
            req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int64)
            # out_cache_loc 需要容纳所有草稿步骤的 KV slot
            out_cache_loc = torch.zeros(
                (self.max_num_token * self.speculative_num_steps,),
                dtype=self._cache_loc_dtype(),
            )
            positions = torch.zeros((self.max_num_token,), dtype=torch.int64)
            mrope_positions = torch.zeros((3, self.max_num_token), dtype=torch.int64)
            seq_lens = torch.full(
                (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
            )
            # extend_seq_lens 初始化为 1（草稿 decode 每步扩展 1 个 token）
            extend_seq_lens = torch.ones((self.max_bs,), dtype=torch.int32)
            topk_p = torch.zeros((self.max_bs, self.topk), dtype=torch.float32)
            topk_index = torch.zeros((self.max_bs, self.topk), dtype=torch.int64)
            # hidden_states: 目标模型隐状态输入，形状 [max_bs, spec_hidden_size]
            hidden_states = torch.zeros(
                (self.max_bs, self.model_runner.model_config.spec_hidden_size),
                dtype=self.model_runner.dtype,
            )

            if self.require_gathered_buffer:
                if self.require_mlp_tp_gather:
                    # DP + MLP TP gather：每个 DP rank 各有一个 token 计数
                    global_num_tokens_gpu = torch.zeros(
                        (self.dp_size,), dtype=torch.int32
                    )
                    global_num_tokens_for_logprob_gpu = torch.zeros(
                        (self.dp_size,), dtype=torch.int32
                    )
                else:
                    assert self.require_attn_tp_gather
                    # Attention TP gather：仅需一个全局 token 计数
                    global_num_tokens_gpu = torch.zeros((1,), dtype=torch.int32)
                    global_num_tokens_for_logprob_gpu = torch.zeros(
                        (1,), dtype=torch.int32
                    )
            else:
                # 不需要 gathered buffer：置 None 以节省显存
                global_num_tokens_gpu = None
                global_num_tokens_for_logprob_gpu = None

        # 将所有缓冲区组织到 EagleDraftInputBuffers 数据类中
        self.buffers = EagleDraftInputBuffers(
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            out_cache_loc=out_cache_loc,
            positions=positions,
            mrope_positions=mrope_positions,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            extend_seq_lens=extend_seq_lens,
            topk_p=topk_p,
            topk_index=topk_index,
            hidden_states=hidden_states,
            global_num_tokens_gpu=global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob_gpu,
        )
        # share_buffers: 将缓冲区注册到模型，使模型内部张量与 buffer 共享同一内存地址
        self.buffers.share_buffers()

        # 捕获 CUDA Graph（在 model_capture_mode 上下文中执行）
        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture cuda graph failed: {e}\n{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )

    def _cache_loc_dtype(self):
        # KV 缓存 slot 索引使用 int64（确保大序列下不溢出）
        return torch.int64

    def can_run(self, forward_batch: ForwardBatch):
        # 判断当前 batch 是否可以使用 CUDA Graph replay
        if self.require_mlp_tp_gather:
            # DP + MLP TP gather 模式：用各 rank 最大 token 数换算 batch size
            cuda_graph_bs = (
                max(forward_batch.global_num_tokens_cpu) // self.num_tokens_per_bs
                if self.model_runner.spec_algorithm.is_eagle()
                or self.model_runner.spec_algorithm.is_standalone()
                else max(forward_batch.global_num_tokens_cpu)
            )
        else:
            # 非 DP 模式：直接用 batch_size
            cuda_graph_bs = forward_batch.batch_size

        # 检查 batch size 是否在已捕获的范围内
        is_bs_supported = (
            cuda_graph_bs in self.graphs
            if self.disable_padding
            else cuda_graph_bs <= self.max_bs
        )

        # DP 模式下还需检查 can_run_dp_cuda_graph 标志（各 rank batch 形状一致）
        if self.require_mlp_sync:
            is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph

        return is_bs_supported

    def _create_graph(self):
        # 创建一个新的 CUDA Graph 对象
        return torch.cuda.CUDAGraph()

    def _capture_init(self, run_once_fn):
        # 捕获前先 warmup 两次：确保 CUDA 内核被 JIT 编译并 GPU 状态稳定
        for _ in range(2):
            torch.cuda.synchronize()
            self.model_runner.tp_group.barrier()
            run_once_fn()

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        # 在指定 stream 中捕获 CUDA Graph（使用共享显存池避免重复分配）
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            out = run_once_fn()
        return out

    def _replay(self, forward_batch: ForwardBatch):
        # 触发指定 batch size 的 CUDA Graph replay
        self.graphs[self.bs].replay()

    def capture(self):
        # 委托给通用 CudaGraphRunner 的 capture 逻辑
        CudaGraphRunner.capture(self)

    def capture_one_batch_size(
        self, num_seqs: int, forward: Callable, stream_idx: int = 0
    ):
        buffers = self.buffers
        graph = self._create_graph()
        stream = self.stream
        num_tokens = num_seqs * self.num_tokens_per_bs

        # 切出当前 batch size 对应的缓冲区视图（不分配新显存）
        req_pool_indices = buffers.req_pool_indices[:num_seqs]
        seq_lens = buffers.seq_lens[:num_seqs]
        seq_lens_cpu = buffers.seq_lens_cpu[:num_seqs]
        extend_seq_lens = buffers.extend_seq_lens[:num_seqs]
        extend_seq_lens_cpu = self.extend_seq_lens_cpu[:num_seqs]
        # out_cache_loc 覆盖所有草稿步骤的 KV slot
        out_cache_loc = buffers.out_cache_loc[: num_tokens * self.speculative_num_steps]
        positions = buffers.positions[:num_tokens]
        mrope_positions = buffers.mrope_positions[:, :num_tokens]
        hidden_states = buffers.hidden_states[:num_seqs]
        topk_p = buffers.topk_p[:num_seqs]
        topk_index = buffers.topk_index[:num_seqs]

        if self.require_mlp_tp_gather:
            # 设置 DP 各 rank 的 token 数（所有 rank 相同）
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
            global_num_tokens = buffers.global_num_tokens_gpu
            global_dp_buffer_len = num_tokens * self.dp_size
            global_num_tokens_for_logprob = buffers.global_num_tokens_for_logprob_gpu
        elif self.require_attn_tp_gather:
            # attention TP gather 模式：仅一个全局 token 计数
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=buffers.input_ids.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=buffers.input_ids.device,
                )
            )
            global_num_tokens = buffers.global_num_tokens_gpu
            global_dp_buffer_len = num_tokens
            global_num_tokens_for_logprob = buffers.global_num_tokens_for_logprob_gpu
        else:
            # 不需要 gathered buffer
            global_num_tokens = None
            global_dp_buffer_len = None
            global_num_tokens_for_logprob = None

        # 构造草稿输入（topk_p/topk_index/hidden_states 来自上一步目标模型输出）
        spec_info = EagleDraftInput(
            topk_p=topk_p,
            topk_index=topk_index,
            hidden_states=hidden_states,
            # LAST 模式：只捕获最后一层的隐状态（草稿模型输入）
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )

        # 构造 ForwardBatch（DECODE 模式，草稿模型逐 token 生成）
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=num_seqs,
            input_ids=None,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            return_logprob=False,
            positions=positions,
            mrope_positions=mrope_positions,
            global_num_tokens_gpu=global_num_tokens,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=(
                spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
            ),
        )

        # 初始化注意力后端的 CUDA Graph 捕获元数据
        self.draft_attn_backend.init_forward_metadata_capture_cuda_graph(forward_batch)

        # 定义单次前向函数（捕获和 warmup 都调用此函数）
        def run_once():
            # 每次重置 DP 注意力的临时缓存
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(
                global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
            )
            set_is_extend_in_batch(False)

            # 备份两个会被 draft_forward 就地修改的字段
            output_cache_loc_backup = forward_batch.out_cache_loc
            hidden_states_backup = forward_batch.spec_info.hidden_states

            ret = self.eagle_worker.draft_forward(forward_batch)

            # 恢复备份（确保 CUDA Graph 捕获路径中这些指针不变）
            forward_batch.out_cache_loc = output_cache_loc_backup
            forward_batch.spec_info.hidden_states = hidden_states_backup
            return ret

        self.deepep_adapter.capture(is_extend_in_batch=False)

        # 先 warmup，再捕获
        self._capture_init(run_once)

        out = self._capture_graph(
            graph, get_global_graph_memory_pool(), stream, run_once
        )

        # 将当前图的显存池暴露给后续 batch size 共享（减少显存碎片）
        set_global_graph_memory_pool(graph.pool())
        return graph, out

    def _postprocess_output_to_raw_bs(self, out, raw_bs):
        # 当实际 bs < 捕获 bs（有 padding）时，截取有效部分
        parent_list, top_scores_index, draft_tokens = (t[:raw_bs] for t in out)
        return parent_list, top_scores_index, draft_tokens

    def replay(self, forward_batch: ForwardBatch):
        assert forward_batch.out_cache_loc is not None
        # DeepEP 适配器先 replay（MoE 通信特殊处理）
        self.deepep_adapter.replay()
        buffers = self.buffers

        raw_bs = forward_batch.batch_size
        raw_num_token = raw_bs * self.num_tokens_per_bs

        # 找到大于等于 raw_bs 的最小已捕获 batch size（用于 padding）
        if self.require_mlp_tp_gather:
            max_num_tokens = max(forward_batch.global_num_tokens_cpu)
            max_batch_size = (
                max_num_tokens // self.num_tokens_per_bs
                if self.model_runner.spec_algorithm.is_eagle()
                or self.model_runner.spec_algorithm.is_standalone()
                else max_num_tokens
            )
            index = bisect.bisect_left(self.capture_bs, max_batch_size)
        else:
            index = bisect.bisect_left(self.capture_bs, raw_bs)

        bs = self.capture_bs[index]
        # 若需要 padding（bs > raw_bs），先将缓冲区填充为安全值
        if bs != raw_bs:
            buffers.seq_lens.fill_(self.seq_len_fill_value)
            buffers.out_cache_loc.zero_()
            buffers.positions.zero_()

        num_tokens = bs * self.num_tokens_per_bs

        # 将真实 batch 数据拷贝到 CUDA Graph 输入缓冲区
        buffers.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)
        buffers.out_cache_loc[: raw_num_token * self.speculative_num_steps].copy_(
            forward_batch.out_cache_loc
        )
        buffers.positions[:raw_num_token].copy_(forward_batch.positions)
        buffers.topk_p[:raw_bs].copy_(forward_batch.spec_info.topk_p)
        buffers.topk_index[:raw_bs].copy_(forward_batch.spec_info.topk_index)
        buffers.hidden_states[:raw_bs].copy_(forward_batch.spec_info.hidden_states)
        buffers.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)

        # TODO(ch-wan): support num_token_non_padded
        if self.require_gathered_buffer:
            # 将 DP buffer 的 token 计数更新为 padding 后的大小
            buffers.global_num_tokens_gpu.fill_(bs * self.num_tokens_per_bs)
            buffers.global_num_tokens_for_logprob_gpu.fill_(bs * self.num_tokens_per_bs)

        # 若 batch size 改变（padding），更新 forward_batch 中引用的张量视图
        if bs != raw_bs:
            forward_batch.batch_size = bs
            forward_batch.seq_lens = buffers.seq_lens[:bs]
            forward_batch.req_pool_indices = buffers.req_pool_indices[:bs]
            forward_batch.positions = buffers.positions[:num_tokens]

        if forward_batch.seq_lens_cpu is not None:
            if bs != raw_bs:
                buffers.seq_lens_cpu.fill_(self.seq_len_fill_value)
            buffers.seq_lens_cpu[:raw_bs].copy_(forward_batch.seq_lens_cpu)
            forward_batch.seq_lens_cpu = buffers.seq_lens_cpu[:bs]

        # 更新注意力后端的 replay 元数据（KV 缓存索引等）
        self.draft_attn_backend.init_forward_metadata_replay_cuda_graph(
            forward_batch, bs
        )
        self.raw_bs = raw_bs
        self.bs = bs
        # TODO: The forward_batch.seq_len_sum might need to be updated to reflect the padding in the cuda graph

        # 触发 CUDA Graph replay，结果写入 output_buffers[bs]
        self._replay(forward_batch)
        out = self.output_buffers[bs]

        # 若有 padding，从输出中截取真实 batch 部分，并恢复 forward_batch 字段
        if bs != raw_bs:
            out = self._postprocess_output_to_raw_bs(out, raw_bs)
            forward_batch.batch_size = raw_bs
            forward_batch.positions = buffers.positions[:raw_num_token]
            forward_batch.seq_lens = buffers.seq_lens[:raw_bs]
            forward_batch.req_pool_indices = buffers.req_pool_indices[:raw_bs]
            if forward_batch.seq_lens_cpu is not None:
                forward_batch.seq_lens_cpu = buffers.seq_lens_cpu[:raw_bs]

        return out
