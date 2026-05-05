from __future__ import annotations

"""
Support attention backend for flashinfer MLA.
支持 FlashInfer MLA（多头潜在注意力）的注意力后端。
The flashinfer_mla_disable_ragged flag controls whether to use ragged prefill wrapper and defaults to be false.
When it's set to false, all wrappers are BatchMLAPaged wrapper.
When it's set to true, the backend uses BatchRagged and BatchMLAPaged wrapper for prefilling,
and uses BatchMLAPaged wrapper for decoding.
More details can be found in https://docs.flashinfer.ai/api/mla.html
"""

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable, Optional, Union

import torch

# 检测是否处于分段式 CUDA Graph 上下文中
from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
# 从 flashinfer_backend 导入用于填充 KV 索引的 Triton Kernel
from sglang.srt.layers.attention.flashinfer_backend import (
    create_flashinfer_kv_indices_triton,
)
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.utils import (
    is_flashinfer_available,
    is_sm100_supported,
    next_power_of_2,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.flashinfer_mla_backend import (
        FlashInferMlaAttnBackend,
    )
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

if envs.SGLANG_ENABLE_TORCH_COMPILE.get():
    # 启用 torch.compile 时，抑制 dynamo 错误以避免编译失败中断推理
    import logging

    torch._logging.set_logs(dynamo=logging.ERROR)
    torch._dynamo.config.suppress_errors = True

if is_flashinfer_available():
    # 导入 FlashInfer 的 MLA 分页注意力包装器和 Ragged 预填充包装器
    from flashinfer import (
        BatchMLAPagedAttentionWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    )


@dataclass
class DecodeMetadata:
    # 解码阶段使用的 MLA 分页注意力包装器
    decode_wrapper: BatchMLAPagedAttentionWrapper


@dataclass
class PrefillMetadata:
    # 预填充阶段使用的 MLA 分页注意力包装器
    prefill_wrapper: BatchMLAPagedAttentionWrapper
    # 是否使用 Ragged（不等长序列）模式的预填充
    use_ragged: bool


# 所有 FlashInfer 包装器共享此工作区缓冲区，避免重复分配
global_workspace_buffer = None


class FlashInferMhaChunkKVRunner:
    """MHA 分块 KV 缓存运行器，用于在 MLA 模型中对分块前缀 KV 缓存执行标准 MHA 注意力。"""

    def __init__(
        self, model_runner: ModelRunner, attn_backend: FlashInferMlaAttnBackend
    ):
        # 解析常量：从模型配置中获取注意力头数和各维度大小
        self.num_local_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim  # 无 RoPE 的 QK 头维度
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim  # 带 RoPE 的 QK 头维度
        self.v_head_dim = model_runner.model_config.v_head_dim              # V 的头维度
        self.data_type = model_runner.dtype
        self.q_data_type = model_runner.dtype

        # 从 attn_backend 中复用缓冲区和包装器，避免重复分配
        self.qo_indptr = attn_backend.qo_indptr
        self.kv_indptr = attn_backend.kv_indptr
        self.workspace_buffer = attn_backend.workspace_buffer
        self.fmha_backend = attn_backend.fmha_backend

        # 存放各分块对应的 ragged 包装器列表
        self.chunk_ragged_wrappers = []
        self.ragged_wrapper = attn_backend.prefill_wrapper_ragged

    def update_prefix_chunks(self, num_prefix_chunks: int):
        """按需扩展 chunk_ragged_wrappers 列表，确保每个前缀分块都有对应的包装器。"""
        while num_prefix_chunks > len(self.chunk_ragged_wrappers):
            # 创建新的 Ragged KV 缓存预填充包装器并追加到列表
            ragged_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                self.workspace_buffer, "NHD", backend=self.fmha_backend
            )
            self.chunk_ragged_wrappers.append(ragged_wrapper)

    def update_wrapper(
        self,
        forward_batch: ForwardBatch,
        disable_flashinfer_ragged: bool = False,
    ):
        """根据当前批次更新每个分块的包装器，设置 qo_indptr 和 kv_indptr 并调用 begin_forward。"""
        assert forward_batch.num_prefix_chunks is not None
        num_prefix_chunks = forward_batch.num_prefix_chunks
        # 确保有足够数量的分块包装器
        self.update_prefix_chunks(num_prefix_chunks)

        prefix_lens = forward_batch.extend_prefix_lens
        seq_lens = forward_batch.seq_lens

        bs = len(seq_lens)
        qo_indptr = self.qo_indptr
        # 新 token 数量 = seq_len - prefix_len，累积求和得到 qo_indptr
        qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)
        qo_indptr = qo_indptr[: bs + 1]

        for chunk_idx in range(forward_batch.num_prefix_chunks):
            # 对每个前缀分块执行 MHA，用于在 MLA 模型中处理分块前缀 KV 缓存
            assert forward_batch.prefix_chunk_idx is not None
            assert forward_batch.prefix_chunk_cu_seq_lens is not None
            assert forward_batch.prefix_chunk_max_seq_lens is not None

            # 该分块的 KV 累积序列长度
            kv_indptr = forward_batch.prefix_chunk_cu_seq_lens[chunk_idx]
            wrapper = self.chunk_ragged_wrappers[chunk_idx]
            wrapper.begin_forward(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr,
                num_qo_heads=self.num_local_heads,
                num_kv_heads=self.num_local_heads,
                head_dim_qk=self.qk_nope_head_dim + self.qk_rope_head_dim,  # 完整 QK 维度
                head_dim_vo=self.v_head_dim,
                q_data_type=self.q_data_type,
                causal=False,  # 前缀分块注意力无因果掩码
            )
        # 处理当前新 token 的 ragged 预填充
        if not disable_flashinfer_ragged:
            # one_shot 时需要单独的 kv_indptr（前缀已包含），否则使用 qo_indptr（新 token 自注意力）
            kv_indptr = (
                qo_indptr
                if not forward_batch.mha_one_shot
                else self.kv_indptr[: bs + 1]
            )
            self.ragged_wrapper.begin_forward(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr,
                num_qo_heads=self.num_local_heads,
                num_kv_heads=self.num_local_heads,
                head_dim_qk=self.qk_nope_head_dim + self.qk_rope_head_dim,
                head_dim_vo=self.v_head_dim,
                q_data_type=self.q_data_type,
                causal=True,  # 新 token 注意力需要因果掩码
            )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ):
        """执行 MHA 分块注意力：若正在处理前缀分块则使用 chunk_ragged_wrappers，否则使用 ragged_wrapper。"""
        logits_soft_cap = layer.logit_cap
        if forward_batch.attn_attend_prefix_cache:
            # 当前正在处理某个前缀分块的注意力
            chunk_idx = forward_batch.prefix_chunk_idx
            assert chunk_idx >= 0
            wrapper = self.chunk_ragged_wrappers[chunk_idx]
            # 返回注意力输出和 log-sum-exp（用于多分块结果合并）
            o = wrapper.forward_return_lse(
                q.view(-1, layer.tp_q_head_num, layer.head_dim),
                k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                v.view(-1, layer.tp_v_head_num, layer.v_head_dim).to(q.dtype),
                causal=False,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
            )
        else:
            # 使用 ragged 包装器处理新 token 的注意力
            # mha_return_lse 为 True 时需要返回 LSE 以便与分块结果合并
            forward = (
                self.ragged_wrapper.forward_return_lse
                if forward_batch.mha_return_lse
                else self.ragged_wrapper.forward
            )
            o = forward(
                q.view(-1, layer.tp_q_head_num, layer.head_dim),
                k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                v.view(-1, layer.tp_v_head_num, layer.v_head_dim).to(q.dtype),
                causal=True,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
            )
        return o


class FlashInferMLAAttnBackend(AttentionBackend):
    """FlashInfer MLA 注意力后端，支持分页 KV 缓存、Ragged 预填充和 CUDA Graph。"""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        q_indptr_decode_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # 解析常量：最大上下文长度、设备、是否跳过预填充阶段
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.skip_prefill = skip_prefill
        # 是否启用分块 KV 缓存：需要非 decode 专用节点且未禁用相关选项
        self.enable_chunk_kv = (
            not skip_prefill
            and get_global_server_args().disaggregation_mode != "decode"
            and not get_global_server_args().disable_chunked_prefix_cache
            and not get_global_server_args().flashinfer_mla_disable_ragged
        )
        self.page_size = model_runner.page_size

        # 分配工作区缓冲区：所有包装器共享单例缓冲区
        global global_workspace_buffer
        if global_workspace_buffer is None:
            # 与 flashinfer zero_init_global_workspace_buffer 不同，这里不做零初始化
            global_workspace_buffer = torch.empty(
                envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),
                dtype=torch.uint8,
                device=model_runner.device,
            )
        self.workspace_buffer = global_workspace_buffer

        max_bs = model_runner.req_to_token_pool.size
        # kv_indptr: 存储每个请求的 KV 令牌偏移量，形状 (max_bs+1,)
        if kv_indptr_buf is None:
            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
        else:
            self.kv_indptr = kv_indptr_buf

        if not self.skip_prefill:
            # qo_indptr: 存储每个请求的输出 token 偏移量，形状 (max_bs+1,)
            self.qo_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )

        # q_indptr_decode: 解码时每个请求恰好有 1 个 query token，故初始化为 0,1,2,...
        if q_indptr_decode_buf is None:
            self.q_indptr_decode = torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=model_runner.device
            )
        else:
            self.q_indptr_decode = q_indptr_decode_buf

        # SM100（Blackwell）支持时使用 cutlass 后端，否则自动选择
        if is_sm100_supported():
            self.fmha_backend = "cutlass"
        else:
            self.fmha_backend = "auto"

        # Ragged 预填充包装器：处理不等长序列（无分页）
        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend=self.fmha_backend
        )

        if not self.skip_prefill:
            # 分页预填充包装器：处理有前缀缓存的预填充
            self.prefill_wrapper_paged = BatchMLAPagedAttentionWrapper(
                self.workspace_buffer,
                backend="auto",
            )

            # FlashInfer MLA 后端使用 MLA 包装器执行目标验证（target verify）
            self.prefill_wrapper_verify = BatchMLAPagedAttentionWrapper(
                self.workspace_buffer,
                backend="auto",
            )

        # 解码阶段的 MLA 分页注意力包装器
        self.decode_wrapper = BatchMLAPagedAttentionWrapper(
            self.workspace_buffer, backend="auto"
        )

        # 创建索引更新器，负责在每次 forward 前填充 kv_indices 等元数据
        if not skip_prefill:
            self.indices_updater_prefill = FlashInferMLAIndicesUpdaterPrefill(
                model_runner, self
            )
            # 若启用分块 KV，则额外创建 MHA 分块运行器
            if self.enable_chunk_kv:
                self.mha_chunk_kv_cache = FlashInferMhaChunkKVRunner(model_runner, self)

        self.indices_updater_decode = FlashInferMLAIndicesUpdaterDecode(
            model_runner, self
        )

        # 其他元数据：当前 forward 的元数据对象，以及 CUDA Graph 捕获的元数据字典
        self.forward_metadata: Union[PrefillMetadata, DecodeMetadata] = None
        self.decode_cuda_graph_metadata = {}
        self.prefill_cuda_graph_metadata = {}  # 用于 target verify 的 CUDA Graph 元数据

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """根据 forward_mode 初始化本次 forward 所需的元数据（索引、包装器等）。"""
        if forward_batch.forward_mode.is_decode_or_idle():
            # 解码模式：更新解码包装器的索引并存储元数据
            self.indices_updater_decode.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                decode_wrapper=self.decode_wrapper,
                init_metadata_replay=False,
            )
            self.forward_metadata = DecodeMetadata(self.decode_wrapper)
        elif forward_batch.forward_mode.is_draft_extend():
            # 草稿扩展模式（EAGLE 投机解码）：使用分页预填充包装器，不使用 ragged
            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                prefix_lens=None,
                prefill_wrapper_paged=self.prefill_wrapper_paged,
                use_ragged=False,
                spec_info=forward_batch.spec_info,
            )
            self.forward_metadata = PrefillMetadata(self.prefill_wrapper_paged, False)
        elif forward_batch.forward_mode.is_target_verify():
            # 目标验证模式（EAGLE 投机解码验证阶段）：使用独立的 verify 包装器
            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                prefix_lens=None,
                prefill_wrapper_paged=self.prefill_wrapper_verify,
                use_ragged=False,
                spec_info=forward_batch.spec_info,
            )
            self.forward_metadata = PrefillMetadata(self.prefill_wrapper_verify, False)
        else:
            # 普通预填充模式：根据是否有前缀缓存决定使用 ragged 还是分页模式
            prefix_lens = forward_batch.extend_prefix_lens
            # 若所有请求都无前缀，则可使用效率更高的 ragged 模式
            extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
            use_ragged = (
                not get_global_server_args().flashinfer_mla_disable_ragged
                and extend_no_prefix
                # 分段式 CUDA Graph 需要使用分页预填充以兼容前缀缓存
                and not is_in_piecewise_cuda_graph()
            )

            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                prefix_lens,
                prefill_wrapper_paged=self.prefill_wrapper_paged,
                use_ragged=use_ragged,
            )
            self.forward_metadata = PrefillMetadata(
                self.prefill_wrapper_paged, use_ragged
            )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        """预分配 CUDA Graph 捕获所需的静态缓冲区。"""
        if kv_indices_buf is None:
            # 分配 KV 索引缓冲区：最大批次 × 最大上下文长度
            cuda_graph_kv_indices = torch.zeros(
                (max_bs * self.max_context_len,),
                dtype=torch.int32,
                device="cuda",
            )
        else:
            cuda_graph_kv_indices = kv_indices_buf

        self.cuda_graph_kv_indices = cuda_graph_kv_indices
        # qo_indptr 从 q_indptr_decode 克隆（CUDA Graph 期间不可改变形状）
        self.cuda_graph_qo_indptr = self.q_indptr_decode.clone()
        self.cuda_graph_kv_indptr = self.kv_indptr.clone()
        # kv_lens：每个请求的 KV 长度，初始值为 1（CUDA Graph 期间原地更新）
        self.cuda_graph_kv_lens = torch.ones(
            (max_bs,), dtype=torch.int32, device=self.device
        )

        # CPU 端副本用于 fast_mla_decode_plan，避免 GPU→CPU 同步
        self.cuda_graph_qo_indptr_cpu = self.cuda_graph_qo_indptr.to("cpu")
        self.cuda_graph_kv_indptr_cpu = self.cuda_graph_kv_indptr.to("cpu")
        # 快速解码参数字典，在 replay 时原地更新
        self.fast_decode_kwargs = {
            "qo_indptr_cpu": self.cuda_graph_qo_indptr_cpu,
            "kv_indptr_cpu": self.cuda_graph_kv_indptr_cpu,
            "kv_indices": self.cuda_graph_kv_indices,
        }

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        """CUDA Graph 捕获阶段：创建使用静态缓冲区的包装器并初始化元数据，按 bs 缓存。"""
        if forward_mode.is_decode_or_idle():
            # 解码模式：创建启用 CUDA Graph 的 decode 包装器，绑定静态缓冲区
            decode_wrapper = BatchMLAPagedAttentionWrapper(
                self.workspace_buffer,
                use_cuda_graph=True,
                qo_indptr=self.cuda_graph_qo_indptr[: num_tokens + 1],
                kv_indptr=self.cuda_graph_kv_indptr[: num_tokens + 1],
                kv_indices=self.cuda_graph_kv_indices,
                kv_len_arr=self.cuda_graph_kv_lens[:num_tokens],
                backend="auto",
            )

            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_decode.update(
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                decode_wrapper=decode_wrapper,
                init_metadata_replay=False,
                spec_info=spec_info,
            )
            # 将包装器按 bs 大小缓存，replay 时直接复用
            self.decode_cuda_graph_metadata[bs] = decode_wrapper
            self.forward_metadata = DecodeMetadata(decode_wrapper)
            # 替换 plan 函数为快速版本（跳过流同步）
            decode_wrapper.plan = partial(fast_mla_decode_plan, decode_wrapper)
        elif forward_mode.is_target_verify():
            # 目标验证模式：创建独立的 verify 包装器用于 CUDA Graph 捕获
            verify_wrapper = BatchMLAPagedAttentionWrapper(
                self.workspace_buffer,
                use_cuda_graph=True,
                qo_indptr=self.cuda_graph_qo_indptr[: bs + 1],
                kv_indptr=self.cuda_graph_kv_indptr[: bs + 1],
                kv_indices=self.cuda_graph_kv_indices,
                kv_len_arr=self.cuda_graph_kv_lens[:bs],
                backend="auto",
            )
            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_prefill.update(
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrapper_paged=verify_wrapper,
                use_ragged=False,
                spec_info=spec_info,
            )
            # 按 bs 缓存 verify 包装器
            self.prefill_cuda_graph_metadata[bs] = verify_wrapper
            self.forward_metadata = PrefillMetadata(verify_wrapper, False)
        elif forward_mode.is_draft_extend():
            # 草稿扩展模式：创建独立的 draft_extend 包装器用于 CUDA Graph 捕获
            draft_extend_wrapper = BatchMLAPagedAttentionWrapper(
                self.workspace_buffer,
                use_cuda_graph=True,
                qo_indptr=self.cuda_graph_qo_indptr[: bs + 1],
                kv_indptr=self.cuda_graph_kv_indptr[: bs + 1],
                kv_indices=self.cuda_graph_kv_indices,
                kv_len_arr=self.cuda_graph_kv_lens[:bs],
                backend="auto",
            )
            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_prefill.update(
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrapper_paged=draft_extend_wrapper,
                use_ragged=False,
                spec_info=spec_info,
            )
            self.prefill_cuda_graph_metadata[bs] = draft_extend_wrapper
            self.forward_metadata = PrefillMetadata(draft_extend_wrapper, False)
        else:
            raise ValueError(f"Invalid mode: {forward_mode=}")

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        """CUDA Graph 回放阶段：原地更新静态缓冲区中的 kv_indptr 等数据，不重新分配包装器。"""
        if forward_mode.is_decode_or_idle():
            assert seq_lens_cpu is not None
            # 直接在 CPU 上计算 kv_indptr，避免 GPU 同步
            kv_len_arr_cpu = seq_lens_cpu[:bs]
            self.cuda_graph_kv_indptr_cpu[1 : bs + 1] = torch.cumsum(
                kv_len_arr_cpu, dim=0
            )
            # 更新 fast_decode_kwargs 中的切片，使用最新的长度信息
            self.fast_decode_kwargs.update(
                {
                    "qo_indptr_cpu": self.cuda_graph_qo_indptr_cpu[: bs + 1],
                    "kv_indptr_cpu": self.cuda_graph_kv_indptr_cpu[: bs + 1],
                    "kv_len_arr_cpu": kv_len_arr_cpu,
                }
            )

            # 使用 init_metadata_replay=True 调用解码更新器，跳过 wrapper.plan 中的流同步
            self.indices_updater_decode.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_sum,
                decode_wrapper=self.decode_cuda_graph_metadata[bs],
                init_metadata_replay=True,
                spec_info=spec_info,
                **self.fast_decode_kwargs,
            )
        elif forward_mode.is_target_verify():
            # target verify 回放：更新预分配的 verify 包装器
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrapper_paged=self.prefill_cuda_graph_metadata[bs],
                use_ragged=False,
                spec_info=spec_info,
            )
        elif forward_mode.is_draft_extend():
            # draft extend 回放：更新预分配的 draft extend 包装器
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrapper_paged=self.prefill_cuda_graph_metadata[bs],
                use_ragged=False,
                spec_info=spec_info,
            )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

    def get_cuda_graph_seq_len_fill_value(self):
        # CUDA Graph 中序列长度的填充值为 1（解码时每个请求恰好有 1 个 query）
        return 1

    def init_mha_chunk_metadata(
        self, forward_batch: ForwardBatch, disable_flashinfer_ragged: bool = False
    ):
        """初始化分块 MHA 的包装器元数据（在每次 forward 前调用）。"""
        self.mha_chunk_kv_cache.update_wrapper(forward_batch, disable_flashinfer_ragged)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        """预填充阶段的前向传播，支持分块 MHA、Ragged 模式和 MLA 分页模式。"""
        if forward_batch.attn_attend_prefix_cache is not None and any(
            forward_batch.extend_prefix_lens_cpu
        ):  # 分块 MHA：存在前缀缓存时走 mha_chunk_kv_cache 路径
            assert self.enable_chunk_kv
            assert q_rope is None
            assert k_rope is None
            return self.mha_chunk_kv_cache.forward(q, k, v, layer, forward_batch)

        cache_loc = forward_batch.out_cache_loc
        logits_soft_cap = layer.logit_cap
        prefill_wrapper_paged = self.forward_metadata.prefill_wrapper

        # 写入 KV 缓存：根据是否有 k_rope 选择 MLA 格式或普通格式
        if save_kv_cache and k is not None:
            assert v is not None
            if save_kv_cache:
                if k_rope is not None:
                    # MLA 格式：分别存储 k（compressed latent）和 k_rope（rope 部分）
                    forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                        layer, cache_loc, k, k_rope
                    )
                else:
                    # 普通 MHA 格式
                    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
        # 若有分离的 q_rope，则 reshape 为 (num_tokens, num_heads, dim) 格式
        if q_rope is not None:
            q = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )

        if self.forward_metadata.use_ragged:
            # Ragged 预填充：将 q_nope 和 q_rope 拼接后使用 ragged 包装器
            if q_rope is not None:
                q = torch.cat([q, q_rope], dim=-1)
            qall = q.view(-1, layer.tp_q_head_num, layer.head_dim)
            if k_rope is not None:
                k = torch.cat([k, k_rope], dim=-1)
            o = self.prefill_wrapper_ragged.forward(
                qall,
                k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
                causal=True,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
            )
        else:
            # MLA 分页预填充：从 KV 缓冲区读取 k，分别传入 kv_nope 和 kv_rope 部分
            k_buf = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).to(
                q.dtype
            )
            if q_rope is None:
                # 若 q 未拆分，则手动拆分为 q_nope 和 q_rope
                qall = q.view(-1, layer.tp_q_head_num, layer.head_dim)
                q, q_rope = (
                    qall[:, :, : layer.v_head_dim],
                    qall[:, :, layer.v_head_dim :],
                )
            o = q.new_empty(q.shape)
            o = prefill_wrapper_paged.run(
                q,
                q_rope,
                k_buf[:, :, : layer.v_head_dim],    # kv_c（compressed latent）
                k_buf[:, :, layer.v_head_dim :],     # k_pe（positional encoding 部分）
                out=o,
            )

        # 将输出 reshape 为 (num_tokens, num_heads * v_head_dim)
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # MLA（多头潜在注意力）额外参数
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        """解码阶段的前向传播：写入 KV 缓存，分离 q_nope/q_rope，调用 MLA 解码核。"""
        decode_wrapper = self.forward_metadata.decode_wrapper
        cache_loc = forward_batch.out_cache_loc

        if k is not None:
            assert v is not None
            if save_kv_cache:
                if k_rope is not None:
                    # MLA 格式存储：k 为 compressed latent，k_rope 为 positional encoding 部分
                    forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope,
                    )
                else:
                    # 普通 MHA 格式存储
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        v,
                    )

        # 对 q 进行 reshape：拆分为 q_nope（latent 部分）和 q_rope（旋转位置编码部分）
        if q_rope is not None:
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
        else:
            reshaped_q = q.view(-1, layer.tp_q_head_num, layer.head_dim)
            q_nope = reshaped_q[:, :, : layer.v_head_dim]
            q_rope = reshaped_q[:, :, layer.v_head_dim :]

        # 从 KV 缓冲区读取 K，并转换为与 q 相同的 dtype
        k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).to(
            q.dtype
        )

        o = q_nope.new_empty(q_nope.shape)
        # 调用 FlashInfer MLA 解码核：分离传入 kv_c（nope 部分）和 k_pe（rope 部分）
        o = decode_wrapper.run(
            q_nope,
            q_rope,
            k_buffer[:, :, : layer.v_head_dim],   # kv_c：compressed latent
            k_buffer[:, :, layer.v_head_dim :],    # k_pe：positional encoding 部分
            out=o,
        )

        # 输出 reshape 为 (num_tokens, num_heads * v_head_dim)
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)


class FlashInferMLAIndicesUpdaterDecode:
    """解码阶段的 KV 索引更新器：负责在每次 forward 前填充 kv_indptr 和 kv_indices。"""

    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # 解析常量：本地头数（考虑张量并行分割）
        self.num_local_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank       # KV LoRA 秩（压缩维度）
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
        self.scaling = model_runner.model_config.scaling                  # 注意力缩放因子
        self.data_type = model_runner.dtype
        self.attn_backend = attn_backend

        # 从 attn_backend 中复用缓冲区，避免重复分配
        self.kv_indptr = attn_backend.kv_indptr
        self.req_to_token = model_runner.req_to_token_pool.req_to_token   # 请求到 token 的映射表
        self.q_indptr = attn_backend.q_indptr_decode

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        decode_wrapper: BatchMLAPagedAttentionWrapper,
        init_metadata_replay: bool = False,
        spec_info: Optional[SpecInput] = None,
        **fast_decode_kwargs,
    ):
        """委托给 call_begin_forward 执行索引填充和包装器初始化。"""
        decode_wrapper = decode_wrapper or self.decode_wrapper
        self.call_begin_forward(
            decode_wrapper,
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            self.q_indptr,
            self.kv_indptr,
            init_metadata_replay,
            spec_info,
            **fast_decode_kwargs,
        )

    def call_begin_forward(
        self,
        wrapper: BatchMLAPagedAttentionWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        q_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        init_metadata_replay: bool = False,
        spec_info: Optional[SpecInput] = None,
        **fast_decode_kwargs,
    ):
        """填充 kv_indices，并调用包装器的 plan 方法完成解码前的元数据初始化。"""
        bs = len(req_pool_indices)
        q_indptr = q_indptr[: bs + 1]
        kv_lens = paged_kernel_lens.to(torch.int32)
        sm_scale = self.scaling
        if spec_info is None:
            # 普通解码：通过累加序列长度得到 kv_indptr
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            # 若非 replay 模式，则新分配 kv_indices 缓冲区；replay 时复用已有缓冲区
            kv_indices = (
                torch.empty(paged_kernel_lens_sum, dtype=torch.int32, device="cuda")
                if not init_metadata_replay
                else fast_decode_kwargs["kv_indices"]
            )
            # 调用 Triton Kernel 将请求→token 映射展开为 kv_indices 平铺数组
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.shape[1],
            )
        else:
            # 投机解码：kv_indptr 和 kv_indices 由 spec_info 预先计算好
            kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices

        if not init_metadata_replay:
            # 首次捕获/普通 forward：调用完整 plan（含流同步）
            wrapper.plan(
                q_indptr,
                kv_indptr,
                kv_indices,
                kv_lens,
                self.num_local_heads,
                self.kv_lora_rank,       # head_dim_ckv：compressed KV 维度
                self.qk_rope_head_dim,   # head_dim_kpe：位置编码维度
                1,                       # page_size（MLA 中 KV 以 token 为单位存储）
                False,                   # causal=False（解码时每个 token 独立）
                sm_scale,
                self.data_type,
                self.data_type,
            )
        else:
            # CUDA Graph replay：使用 CPU 端的 indptr 调用快速 plan（跳过流同步）
            wrapper.plan(
                fast_decode_kwargs["qo_indptr_cpu"],
                fast_decode_kwargs["kv_indptr_cpu"],
                kv_indices,
                fast_decode_kwargs["kv_len_arr_cpu"],
                self.num_local_heads,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                1,
                False,
                sm_scale,
                self.data_type,
                self.data_type,
            )


class FlashInferMLAIndicesUpdaterPrefill:
    """预填充阶段的 KV 索引更新器：支持 ragged 模式和 MLA 分页模式。"""

    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # 解析常量
        self.num_local_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank       # KV LoRA 秩
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
        self.v_head_dim = model_runner.model_config.v_head_dim
        self.scaling = model_runner.model_config.scaling
        self.data_type = model_runner.dtype
        self.q_data_type = model_runner.dtype
        self.attn_backend = attn_backend

        # 从 attn_backend 中复用缓冲区
        self.kv_indptr = attn_backend.kv_indptr
        self.qo_indptr = attn_backend.qo_indptr
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.prefill_wrapper_ragged = attn_backend.prefill_wrapper_ragged

    def update(
        self,
        req_pool_indices: torch.Tnesor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefill_wrapper_paged: BatchMLAPagedAttentionWrapper,
        use_ragged: bool,
        spec_info: Optional[SpecInput] = None,
    ):
        """根据是否使用 ragged 模式确定分页 KV 长度，然后调用 call_begin_forward。"""
        if use_ragged:
            # ragged 模式：分页 KV 只包含前缀部分（新 token 直接用 ragged 计算）
            paged_kernel_lens = prefix_lens
            paged_kernel_lens_sum = paged_kernel_lens.sum().item()
        else:
            # 分页模式：分页 KV 包含全部序列长度
            paged_kernel_lens = seq_lens
            paged_kernel_lens_sum = seq_lens_sum

        self.call_begin_forward(
            self.prefill_wrapper_ragged,
            prefill_wrapper_paged,
            req_pool_indices,
            paged_kernel_lens,
            paged_kernel_lens_sum,
            seq_lens,
            prefix_lens,
            self.kv_indptr,
            self.qo_indptr,
            use_ragged,
            spec_info,
        )

    def call_begin_forward(
        self,
        wrapper_ragged: BatchPrefillWithRaggedKVCacheWrapper,
        wrapper_paged: BatchMLAPagedAttentionWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        kv_indptr: torch.Tensor,
        qo_indptr: torch.Tensor,
        use_ragged: bool,
        spec_info: Optional[SpecInput] = None,
    ):
        """填充 kv_indices 和 qo_indptr，初始化 ragged 或分页包装器的元数据。"""
        bs = len(seq_lens)
        sm_scale = self.scaling

        if spec_info is None:
            assert len(seq_lens) == len(req_pool_indices)
            # 计算 kv_indptr：各请求的 KV token 偏移量
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                paged_kernel_lens_sum,
                dtype=torch.int32,
                device=req_pool_indices.device,
            )
            # Triton Kernel 展开 req_to_token 为平铺的 kv_indices 数组
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.shape[1],
            )
            # qo_indptr：每个请求产出的新 token 数量（seq_len - prefix_len）
            qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
            custom_mask = None
        else:
            assert isinstance(spec_info, SpecInput)
            # TODO: 当 topk > 1 时需要支持自定义掩码
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    req_pool_indices,
                    paged_kernel_lens,
                    paged_kernel_lens_sum,
                    self.req_to_token,
                )
            )

        if use_ragged:
            # ragged 预填充：QO 和 KV 的 indptr 相同（每个 token 都是自己的 KV）
            wrapper_ragged.begin_forward(
                qo_indptr=qo_indptr,
                kv_indptr=qo_indptr,
                num_qo_heads=self.num_local_heads,
                num_kv_heads=self.num_local_heads,
                head_dim_qk=self.qk_nope_head_dim + self.qk_rope_head_dim,
                head_dim_vo=self.v_head_dim,
                q_data_type=self.q_data_type,
                causal=True,
            )
        else:
            # MLA 分页预填充：通过 kv_len_arr 传入每个请求的 KV 长度
            kv_len_arr = kv_indptr[1:] - kv_indptr[:-1]
            wrapper_paged.plan(
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_len_arr,
                self.num_local_heads,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                1,
                True,              # causal=True（预填充需要因果掩码）
                sm_scale,
                self.q_data_type,
                self.data_type,
            )


class FlashInferMLAMultiStepDraftBackend:
    """
    Wrap multiple flashinfer mla attention backends as one for multiple consecutive
    draft decoding steps.
    将多个 FlashInfer MLA 注意力后端封装为一个，用于多步连续草稿解码（EAGLE 投机解码）。
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        from sglang.srt.speculative.spec_utils import generate_draft_decode_kv_indices

        # 当前 FlashInfer MLA 仅支持 topk=1 的投机解码
        if topk > 1:
            raise ValueError(
                "Currently Flashinfer MLA only supports topk=1 for speculative decoding"
            )
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        # 用于生成各解码步骤的 KV 索引的 Triton Kernel
        self.generate_draft_decode_kv_indices = generate_draft_decode_kv_indices

        # 最大批次大小（考虑 topk 倍数）
        max_bs = model_runner.req_to_token_pool.size * self.topk
        # kv_indptr：形状 (num_steps, max_bs+1)，每步独立维护
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        # 解码 q_indptr：解码时每个序列恰好有 1 个 query，故初始化为 0,1,2,...
        self.q_indptr_decode = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=model_runner.device
        )

        # 为每步（除最后一步外）创建独立的 FlashInferMLAAttnBackend
        self.attn_backends = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                FlashInferMLAAttnBackend(
                    model_runner,
                    skip_prefill=True,                      # 草稿步骤只做解码
                    kv_indptr_buf=self.kv_indptr[i],        # 共享 kv_indptr 的第 i 行
                    q_indptr_decode_buf=self.q_indptr_decode,
                )
            )

        self.max_context_len = self.attn_backends[0].max_context_len

        # 缓存用于 generate_draft_decode_kv_indices 的常量参数
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]
        self.page_size = model_runner.server_args.page_size

    def common_template(
        self,
        forward_batch: ForwardBatch,
        kv_indices_buffer: torch.Tensor,
        call_fn: Callable,
    ):
        """通用模板：先用 Triton Kernel 生成各步的 KV 索引，再逐步调用 call_fn。"""
        num_seqs = forward_batch.batch_size
        bs = self.topk * num_seqs
        seq_lens_sum = forward_batch.seq_lens_sum

        # 批量生成所有解码步骤的 KV 索引：形状 (num_steps, seq_lens_sum*topk + bs*(step+1))
        self.generate_draft_decode_kv_indices[
            (self.speculative_num_steps, num_seqs, self.topk)
        ](
            forward_batch.req_pool_indices,
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.seq_lens,
            kv_indices_buffer,
            self.kv_indptr,
            forward_batch.positions,
            self.pool_len,
            kv_indices_buffer.shape[1],
            self.kv_indptr.shape[1],
            next_power_of_2(num_seqs),
            next_power_of_2(self.speculative_num_steps),
            next_power_of_2(bs),
            self.page_size,
        )

        assert forward_batch.spec_info is not None
        assert forward_batch.spec_info.is_draft_input()

        # 逐步更新 spec_info 中的 kv_indptr 和 kv_indices，然后执行 call_fn
        for i in range(self.speculative_num_steps - 1):
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]
            # 每步的 kv_indices 长度为：当前序列中已有 token 数 + 本步新生成 token 数
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][
                : seq_lens_sum * self.topk + bs * (i + 1)
            ]
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """普通（非 CUDA Graph）forward 前的元数据初始化。"""
        kv_indices = torch.zeros(
            (
                self.speculative_num_steps,
                forward_batch.batch_size * self.topk * self.max_context_len,
            ),
            dtype=torch.int32,
            device="cuda",
        )

        def call_fn(i, forward_batch):
            # 克隆 spec_info 中的张量，避免各步之间的引用混乱
            forward_batch.spec_info.kv_indptr = (
                forward_batch.spec_info.kv_indptr.clone()
            )
            forward_batch.spec_info.kv_indices = (
                forward_batch.spec_info.kv_indices.clone()
            )
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, kv_indices, call_fn)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """预分配 CUDA Graph 捕获所需的静态 KV 索引缓冲区。"""
        # 形状：(num_steps, max_bs * max_context_len)，各步共用一个大缓冲区的各行
        self.cuda_graph_kv_indices = torch.zeros(
            (self.speculative_num_steps, max_bs * self.max_context_len),
            dtype=torch.int32,
            device="cuda",
        )

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs, max_num_tokens, kv_indices_buf=self.cuda_graph_kv_indices[i]
            )

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        """CUDA Graph 捕获阶段：为各步初始化 CUDA Graph 包装器。"""
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        """CUDA Graph 回放阶段：原地更新各步的 KV 索引和 indptr。"""
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                seq_lens_sum=-1,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)
        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)


def fast_mla_decode_plan(
    self,
    qo_indptr_cpu: torch.Tensor,
    kv_indptr_cpu: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_len_arr_cpu: torch.Tensor,
    num_heads: int,
    head_dim_ckv: int,
    head_dim_kpe: int,
    page_size: int,
    causal: bool,
    sm_scale: float,
    q_data_type: torch.dtype,
    kv_data_type: torch.dtype,
) -> None:
    """BatchMLAPagedAttentionWrapper::plan 的快速版本，
    用于在 CUDA Graph 回放时跳过原始 plan 函数中的流同步，减少回放延迟。
    """
    # 保存解码参数供 run 时使用
    self._causal = causal
    self._page_size = page_size
    self._sm_scale = sm_scale

    try:
        # 调用底层 C++ 计划函数（无 use_profiler 参数的标准版本）
        self._cached_module.plan(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._pin_memory_int_workspace_buffer,
            qo_indptr_cpu,
            kv_indptr_cpu,
            kv_len_arr_cpu,
            num_heads,
            head_dim_ckv,
            causal,
        )
    except Exception as e:
        raise RuntimeError(f"Error in alternate MLA plan: {e}")
