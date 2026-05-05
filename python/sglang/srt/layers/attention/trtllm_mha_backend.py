from __future__ import annotations

"""
Support attention backend for TRTLLM MHA kernels from flashinfer.
支持使用 FlashInfer 提供的 TRT-LLM MHA（多头注意力）核的注意力后端。
The kernel supports sm100 only, with sliding window and attention sink features.
该核仅支持 SM100（Blackwell）架构，具备滑动窗口注意力和 Attention Sink 特性。
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
# 继承 FlashInfer 的 MHA 注意力后端和多步草稿后端
from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    FlashInferMultiStepDraftBackend,
)
# 融合 FP8 KV 缓存写入的 Triton Kernel
from sglang.srt.layers.attention.triton_ops.trtllm_fp8_kv_kernel import (
    fused_fp8_set_kv_buffer,
)
# 修正退化步幅（num_kv_heads=1 时的 TMA descriptor 问题）
from sglang.srt.layers.attention.utils import canonicalize_stride
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool, SWATokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_flashinfer_available
from sglang.srt.utils.common import is_sm90_supported, is_sm120_supported

logger = logging.getLogger(__name__)

if is_flashinfer_available():
    import flashinfer

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

# 常量：TRT-LLM MHA 工作区默认大小（MB），可通过环境变量覆盖
DEFAULT_WORKSPACE_SIZE_MB = 512

# 所有 TRT-LLM MHA 包装器共享的单例零初始化工作区缓冲区
global_zero_init_workspace_buffer = None


@dataclass
class TRTLLMMHAMetadata:
    # 批次中每个请求在 KV 缓存中已存储的序列长度（int32 格式）
    cache_seqlens_int32: torch.Tensor = None
    # 查询序列的最大长度
    max_seq_len_q: int = 1
    # KV 序列的最大长度
    max_seq_len_k: int = 0
    # 查询序列的累积长度（cu_seqlens_q[i+1] - cu_seqlens_q[i] 为第 i 个请求的 q 长度）
    cu_seqlens_q: torch.Tensor = None
    # KV 序列的累积长度
    cu_seqlens_k: torch.Tensor = None
    # 分页表：记录每个请求的 KV 缓存页编号，形状 (bs, max_num_pages)
    page_table: torch.Tensor = None
    # SWA 层专用分页表（将全池索引转换为 SWA 池索引后的页编号）
    swa_page_table: torch.Tensor = None


class TRTLLMHAAttnBackend(FlashInferAttnBackend):
    """TRT-LLM MHA 注意力后端，继承自 FlashInferAttnBackend，支持 SM100 专用核、SWA 和 FP8。"""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
        speculative_step_id: int = 0,
    ):
        # 在调用父类 __init__ 前捕获工作区大小，防止父类覆盖环境变量
        env_var = envs.SGLANG_FLASHINFER_WORKSPACE_SIZE
        workspace_size_bytes = (
            env_var.get()
            if env_var.is_set()
            else DEFAULT_WORKSPACE_SIZE_MB * 1024 * 1024
        )

        super().__init__(
            model_runner, skip_prefill, kv_indptr_buf, kv_last_page_len_buf
        )

        config = model_runner.model_config

        # MHA 专属维度参数
        self.max_context_len = model_runner.model_config.context_len
        self.hidden_size = config.hidden_size

        # 运行时参数
        self.data_type = model_runner.kv_cache_dtype   # KV 缓存数据类型（可能为 fp8）
        self.q_data_type = model_runner.dtype           # Q 的数据类型（模型精度）
        self.page_size = model_runner.page_size
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.device = model_runner.device

        # 工作区分配：所有包装器共享单例零初始化工作区缓冲区
        self.workspace_size = workspace_size_bytes
        global global_zero_init_workspace_buffer
        if global_zero_init_workspace_buffer is None:
            # 零初始化（与普通 FlashInfer workspace 不同，TRT-LLM 核要求零填充）
            global_zero_init_workspace_buffer = torch.zeros(
                self.workspace_size,
                dtype=torch.uint8,
                device=model_runner.device,
            )
        self.workspace_buffer = global_zero_init_workspace_buffer

        # CUDA Graph 捕获的解码元数据，按 bs 大小缓存
        self.decode_cuda_graph_metadata = {}

        # 投机解码参数（EAGLE）：当前仅支持 topk <= 1
        self.topk = model_runner.server_args.speculative_eagle_topk or 0
        self.speculative_step_id = speculative_step_id     # 当前草稿步骤编号（0-indexed）
        self.target_verify_metadata = {}

        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
        )

        # 滑动窗口注意力（SWA）混合模型支持
        # 对于混合 SWA 模型，KV 缓存被分成两个池（全量池和 SWA 池），各有独立索引空间
        # 我们为 SWA 层维护转换后的 page_table，使 TRT-LLM 核读取正确的池
        allocator = model_runner.token_to_kv_pool_allocator
        self.use_sliding_window_kv_pool = isinstance(
            allocator, SWATokenToKVPoolAllocator
        )
        self._swa_kv_pool: Optional[SWAKVPool] = (
            allocator.get_kvcache() if self.use_sliding_window_kv_pool else None
        )

        # 当前 forward 的元数据对象
        self.forward_metadata: Optional[TRTLLMMHAMetadata] = None

        # 初始化后端（XQA 或 TRT-LLM-GEN）
        # 不同后端对 q_type 和 out_type 的要求如下：
        # XQA: q_type 必须为 bf16
        #   KV bf16: q_type = bf16, out_type = model_runner.dtype
        #   KV fp8:  q_type = bf16, out_type = model_runner.dtype
        # TRT-LLM-GEN:
        #   KV bf16: q_type = bf16, out_type = model_runner.dtype
        #   KV fp8:  q_type = fp8,  out_type = model_runner.dtype
        self.is_xqa_impl = is_sm90_supported() or is_sm120_supported()

    def _maybe_translate_swa(
        self, token_indices: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """将全量池 token 索引转换为 SWA 池索引；非 SWA 模型返回 None。"""
        if not self.use_sliding_window_kv_pool:
            return None
        shape = token_indices.shape
        return self._swa_kv_pool.translate_loc_from_full_to_swa(
            token_indices.reshape(-1)
        ).reshape(shape)

    def _alloc_swa_page_table(
        self, max_bs: int, max_num_pages: int
    ) -> Optional[torch.Tensor]:
        """分配 SWA 专用分页表缓冲区；非 SWA 模型返回 None。"""
        if not self.use_sliding_window_kv_pool:
            return None
        return torch.zeros(max_bs, max_num_pages, dtype=torch.int32, device=self.device)

    def _copy_swa_page_table(
        self,
        metadata: TRTLLMMHAMetadata,
        page_indices: torch.Tensor,
        num_pages: int,
    ):
        """将 SWA 页索引转换并复制到 metadata.swa_page_table；非 SWA 模型为空操作。"""
        if metadata.swa_page_table is None:
            return
        swa_indices = self._maybe_translate_swa(page_indices)
        # 将 token 索引除以 page_size 得到页编号
        metadata.swa_page_table[:, :num_pages].copy_(swa_indices // self.page_size)

    def _get_layer_cache_loc(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """返回给定层在正确索引空间中的缓存位置（SWA 层使用 SWA 池索引）。"""
        if self.use_sliding_window_kv_pool:
            _, is_swa = self._swa_kv_pool.layers_mapping[layer.layer_id]
            if is_swa:
                if forward_batch.out_cache_loc_swa is not None:
                    return forward_batch.out_cache_loc_swa
                # 将全量池位置转换为 SWA 池位置
                return self._swa_kv_pool.translate_loc_from_full_to_swa(
                    forward_batch.out_cache_loc
                )
        return forward_batch.out_cache_loc

    def _bind_swa_page_table(
        self, metadata: TRTLLMMHAMetadata, source: dict, key: str, bs: int
    ):
        """将预分配的 SWA page_table 切片绑定到 metadata（用于 CUDA Graph 静态缓冲区）。"""
        buf = source.get(key)
        if buf is not None:
            metadata.swa_page_table = buf[:bs, :]

    def _get_layer_page_table(
        self, layer: RadixAttention, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        """返回给定层使用的分页表（SWA 层使用 swa_page_table，其他层使用全量 page_table）。"""
        swa_pt = self.forward_metadata.swa_page_table
        if swa_pt is not None:
            _, is_swa = self._swa_kv_pool.layers_mapping[layer.layer_id]
            if is_swa:
                return swa_pt
        return self.forward_metadata.page_table

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        """预分配 CUDA Graph 捕获所需的静态缓冲区（分页表、序列长度、cu_seqlens 等）。"""
        # 最大页数 = 向上取整(最大上下文长度 / 页大小)
        max_num_pages = (self.max_context_len + self.page_size - 1) // self.page_size
        self.decode_cuda_graph_metadata = {
            "cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "page_table": torch.zeros(
                max_bs,
                max_num_pages,
                dtype=torch.int32,
                device=self.device,
            ),
            "swa_page_table": self._alloc_swa_page_table(max_bs, max_num_pages),
            # strided_indices：用于按步长采样页起始 token 索引（[0, page_size, 2*page_size, ...]）
            "strided_indices": torch.arange(
                0, self.max_context_len, self.page_size, device=self.device
            ),
        }

        # 若配置了投机解码，则额外预分配草稿解码和目标验证的元数据缓冲区
        if (
            self.speculative_num_draft_tokens is not None
            and self.speculative_num_draft_tokens > 0
        ):
            # 草稿解码（draft decode）专用缓冲区：cu_seqlens_q 对应每步 1 个 query
            self.decode_cuda_graph_metadata["cu_seqlens_q"] = torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=self.device
            )
            self.decode_cuda_graph_metadata["cu_seqlens_k"] = torch.zeros(
                max_bs + 1, dtype=torch.int32, device=self.device
            )
            self.decode_cuda_graph_metadata["page_table_draft_decode"] = torch.zeros(
                max_bs,
                max_num_pages,
                dtype=torch.int32,
                device=self.device,
            )
            self.decode_cuda_graph_metadata["swa_page_table_draft_decode"] = (
                self._alloc_swa_page_table(max_bs, max_num_pages)
            )

            # 目标验证（target verify）元数据：每个请求产生 speculative_num_draft_tokens 个 query
            self.target_verify_metadata = {
                "cache_seqlens": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                # cu_seqlens_q：步长为 speculative_num_draft_tokens 的等差数列
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.speculative_num_draft_tokens + 1,
                    step=self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
                "page_table": torch.zeros(
                    max_bs,
                    max_num_pages,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "swa_page_table": self._alloc_swa_page_table(max_bs, max_num_pages),
                "strided_indices": torch.arange(
                    0, self.max_context_len, self.page_size, device=self.device
                ),
            }

            # 草稿扩展（draft extend）元数据：每个请求有可变数量的 query（accepted tokens）
            self.draft_extend_metadata = {
                "cache_seqlens": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                "cu_seqlens_q": torch.zeros(
                    max_bs + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
                "page_table": torch.zeros(
                    max_bs,
                    max_num_pages,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "swa_page_table": self._alloc_swa_page_table(max_bs, max_num_pages),
                "strided_indices": torch.arange(
                    0, self.max_context_len, self.page_size, device=self.device
                ),
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
        """CUDA Graph 捕获阶段：根据 forward_mode 初始化并缓存 TRTLLMMHAMetadata。"""
        metadata = TRTLLMMHAMetadata()
        device = seq_lens.device

        if forward_mode.is_decode_or_idle():
            if spec_info is not None:
                # 草稿解码（Draft Decode）：序列长度需加上当前草稿步骤偏移（speculative_step_id+1）
                # 当前仅支持 topk=1
                metadata.cache_seqlens_int32 = self.decode_cuda_graph_metadata[
                    "cache_seqlens"
                ][:bs]
                metadata.max_seq_len_k = seq_lens.max().item() + (
                    self.speculative_step_id + 1
                )
                # cu_seqlens_q：每个请求恰好 1 个 query，故 cu_seqlens_q = [0,1,2,...,bs]
                metadata.cu_seqlens_q = self.decode_cuda_graph_metadata["cu_seqlens_q"][
                    : bs + 1
                ]
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                metadata.page_table = self.decode_cuda_graph_metadata[
                    "page_table_draft_decode"
                ][:bs, :]
                self._bind_swa_page_table(
                    metadata,
                    self.decode_cuda_graph_metadata,
                    "swa_page_table_draft_decode",
                    bs,
                )
                # 按 bs 缓存，replay 时直接查找
                self.decode_cuda_graph_metadata[bs] = metadata
            else:
                # 普通解码（Normal Decode）
                metadata.cache_seqlens_int32 = seq_lens[:bs].to(torch.int32)
                batch_size = len(seq_lens)
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )

                # 预计算最大序列长度和 cu_seqlens_q
                metadata.max_seq_len_k = seq_lens.max().item()
                metadata.cu_seqlens_q = torch.arange(
                    0, batch_size + 1, dtype=torch.int32, device=device
                )
                # 绑定预分配的分页表切片
                metadata.page_table = self.decode_cuda_graph_metadata["page_table"][
                    :bs, :
                ]
                self._bind_swa_page_table(
                    metadata,
                    self.decode_cuda_graph_metadata,
                    "swa_page_table",
                    bs,
                )
                self.decode_cuda_graph_metadata[bs] = metadata
        elif forward_mode.is_target_verify():
            # 目标验证（Target Verify）：每个请求产生 speculative_num_draft_tokens 个 query
            # 当前仅支持 topk=1
            metadata.cache_seqlens_int32 = self.target_verify_metadata["cache_seqlens"][
                :bs
            ]
            # KV 长度 = 已有序列长度 + 新草稿 token 数
            metadata.cache_seqlens_int32.copy_(
                (seq_lens + self.speculative_num_draft_tokens)
            )

            # cu_seqlens_q：步长为 speculative_num_draft_tokens 的等差数列
            metadata.cu_seqlens_q = torch.arange(
                0,
                bs * self.speculative_num_draft_tokens + 1,
                self.speculative_num_draft_tokens,
                dtype=torch.int32,
                device=device,
            )

            metadata.cu_seqlens_k = self.target_verify_metadata["cu_seqlens_k"][
                : (bs + 1)
            ]

            metadata.max_seq_len_q = self.speculative_num_draft_tokens
            metadata.max_seq_len_k = (
                seq_lens.max().item() + self.speculative_num_draft_tokens
            )

            metadata.page_table = self.target_verify_metadata["page_table"][:bs, :]
            self._bind_swa_page_table(
                metadata,
                self.target_verify_metadata,
                "swa_page_table",
                bs,
            )

            self.target_verify_metadata[bs] = metadata
        elif forward_mode.is_draft_extend():
            # 草稿扩展（Draft Extend）：每个请求产生可变数量的 query（num_tokens/bs 个）
            metadata.cache_seqlens_int32 = self.draft_extend_metadata["cache_seqlens"][
                :bs
            ]
            metadata.cache_seqlens_int32.copy_(seq_lens)
            num_tokens_per_bs = num_tokens // bs
            # cu_seqlens_q：步长为 num_tokens_per_bs 的等差数列（固定大小）
            metadata.cu_seqlens_q = torch.arange(
                0,
                bs * num_tokens_per_bs + 1,
                num_tokens_per_bs,
                dtype=torch.int32,
                device=device,
            )

            metadata.cu_seqlens_k = self.draft_extend_metadata["cu_seqlens_k"][
                : (bs + 1)
            ]
            num_tokens_per_bs = num_tokens // bs
            metadata.max_seq_len_q = num_tokens_per_bs
            metadata.max_seq_len_k = seq_lens.max().item()

            metadata.page_table = self.draft_extend_metadata["page_table"][:bs, :]
            self._bind_swa_page_table(
                metadata,
                self.draft_extend_metadata,
                "swa_page_table",
                bs,
            )

            self.draft_extend_metadata[bs] = metadata
        self.forward_metadata = metadata

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
        """CUDA Graph 回放阶段：原地更新分页表、cu_seqlens 和 cache_seqlens，不重新分配缓冲区。"""
        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs]
        req_pool_indices = req_pool_indices[:bs]
        metadata = None
        if forward_mode.is_decode_or_idle():
            if spec_info is not None:
                # 草稿解码：KV 长度 = 当前序列长度 + 草稿步骤偏移
                metadata = self.decode_cuda_graph_metadata[bs]
                max_len = seq_lens_cpu.max().item()
                metadata.max_seq_len_k = max_len + self.speculative_step_id + 1

                max_seq_pages = (
                    metadata.max_seq_len_k + self.page_size - 1
                ) // self.page_size

                # 原地更新 cache_seqlens（GPU 张量）
                metadata.cache_seqlens_int32.copy_(
                    seq_lens + self.speculative_step_id + 1
                )
            else:
                # 普通解码：直接用当前序列长度
                metadata = self.decode_cuda_graph_metadata[bs]
                max_len = seq_lens_cpu.max().item()
                max_seq_pages = (max_len + self.page_size - 1) // self.page_size
                metadata.max_seq_len_k = max_len

                metadata.cache_seqlens_int32.copy_(seq_lens)

            # 更新 cu_seqlens_k（累积序列长度）
            metadata.cu_seqlens_k[1:].copy_(
                torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)
            )
            # 通过 strided_indices 从 req_to_token 中采样页起始 token，再除以 page_size 得到页编号
            page_indices = self.req_to_token[
                req_pool_indices[:, None],
                self.decode_cuda_graph_metadata["strided_indices"][:max_seq_pages][
                    None, :
                ],
            ]
            metadata.page_table[:, :max_seq_pages].copy_(page_indices // self.page_size)
            self._copy_swa_page_table(metadata, page_indices, max_seq_pages)
        elif forward_mode.is_target_verify():
            # 目标验证：KV 长度 = 序列长度 + 草稿 token 数
            metadata = self.target_verify_metadata[bs]
            metadata.cache_seqlens_int32.copy_(
                (seq_lens + self.speculative_num_draft_tokens)
            )

            metadata.max_seq_len_k = (
                seq_lens_cpu.max().item() + self.speculative_num_draft_tokens
            )
            max_len = seq_lens_cpu.max().item()
            metadata.cu_seqlens_k[1:].copy_(
                torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)
            )
            max_seq_pages = (
                metadata.max_seq_len_k + self.page_size - 1
            ) // self.page_size
            page_indices = self.req_to_token[
                req_pool_indices[:, None],
                self.decode_cuda_graph_metadata["strided_indices"][:max_seq_pages],
            ]
            metadata.page_table[:, :max_seq_pages].copy_(page_indices // self.page_size)
            self._copy_swa_page_table(metadata, page_indices, max_seq_pages)
            metadata.max_seq_len_q = self.speculative_num_draft_tokens
        elif forward_mode.is_draft_extend():
            # 草稿扩展：更新 cache_seqlens 和可变长度的 cu_seqlens_q
            metadata = self.draft_extend_metadata[bs]
            metadata.cache_seqlens_int32.copy_(seq_lens)

            metadata.max_seq_len_k = seq_lens_cpu.max().item()
            max_len = seq_lens_cpu.max().item()
            metadata.cu_seqlens_k[1:].copy_(
                torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)
            )
            # extend_lens：每个请求实际接受的 token 数
            extend_lens = spec_info.num_accepted_tokens[:bs]
            if spec_info.num_accepted_tokens_cpu:
                metadata.max_seq_len_q = max(spec_info.num_accepted_tokens_cpu)
            else:
                metadata.max_seq_len_q = 1

            metadata.cu_seqlens_q[1:].copy_(
                torch.cumsum(extend_lens, dim=0, dtype=torch.int32)
            )

            max_seq_pages = (
                metadata.max_seq_len_k + self.page_size - 1
            ) // self.page_size
            page_indices = self.req_to_token[
                req_pool_indices[:, None],
                self.draft_extend_metadata["strided_indices"][:max_seq_pages],
            ]
            metadata.page_table[:, :max_seq_pages].copy_(page_indices // self.page_size)
            self._copy_swa_page_table(metadata, page_indices, max_seq_pages)
        self.forward_metadata = metadata

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        """CUDA Graph 中序列长度的填充值（解码时每个请求的最小有效长度为 1）。"""
        return 1

    def _should_use_fused_fp8_path(self, save_kv_cache: bool, k: torch.Tensor) -> bool:
        """判断是否使用融合 FP8 KV 缓存写入路径（仅当 KV 数据类型为 fp8 时启用）。"""
        return save_kv_cache and k is not None and self.data_type == torch.float8_e4m3fn

    def _fused_fp8_set_kv_buffer(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        **kwargs,
    ):
        """融合 FP8 量化 + KV 缓存写入：将 k/v 量化为 fp8 并直接写入 KV 缓冲区。"""
        cache_loc = self._get_layer_cache_loc(layer, forward_batch)

        # 从 token_to_kv_pool 获取 K/V 缓冲区（已分配的页式存储）
        k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)

        fused_fp8_set_kv_buffer(
            k=k,
            v=v,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_loc=cache_loc,
            k_scale=layer.k_scale,  # 可能为 None（无 scale 时使用默认值）
            v_scale=layer.v_scale,  # 可能为 None
            page_size=self.page_size,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """根据 forward_mode 初始化本次 forward 所需的 TRTLLMMHAMetadata 元数据。"""

        metadata = TRTLLMMHAMetadata()
        seqlens_in_batch = forward_batch.seq_lens
        batch_size = forward_batch.batch_size
        device = seqlens_in_batch.device

        if forward_batch.forward_mode.is_decode_or_idle():
            if forward_batch.spec_info is not None:
                # 草稿解码：KV 长度 = 当前序列长度 + (草稿步骤编号 + 1) 个新生成 token
                # 当前仅支持 topk=1
                metadata.cache_seqlens_int32 = (
                    seqlens_in_batch + (self.speculative_step_id + 1)
                ).to(torch.int32)
                metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item() + (
                    self.speculative_step_id + 1
                )
                metadata.cu_seqlens_q = torch.arange(
                    0, batch_size + 1, dtype=torch.int32, device=device
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                # page_table：从 req_to_token 中取出最多 max_seq_len_k 列
                metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]
            else:
                # 普通解码
                metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
                metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
                metadata.cu_seqlens_q = torch.arange(
                    0, batch_size + 1, dtype=torch.int32, device=device
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
                )
                metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]
        elif forward_batch.forward_mode.is_target_verify():
            # 目标验证（仅支持 topk=1）：每个请求产生 speculative_num_draft_tokens 个 query
            metadata.cache_seqlens_int32 = (
                forward_batch.seq_lens + self.speculative_num_draft_tokens
            ).to(torch.int32)
            metadata.max_seq_len_q = self.speculative_num_draft_tokens
            metadata.max_seq_len_k = (
                forward_batch.seq_lens_cpu.max().item()
                + self.speculative_num_draft_tokens
            )
            metadata.cu_seqlens_q = torch.arange(
                0,
                batch_size * self.speculative_num_draft_tokens + 1,
                self.speculative_num_draft_tokens,
                dtype=torch.int32,
                device=device,
            )
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32),
                (1, 0),
            )
            metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.max_seq_len_k
            ]

        else:
            # 预填充（extend）模式
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )
            metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.max_seq_len_k
            ]

            if any(
                forward_batch.extend_prefix_lens_cpu
            ) or forward_batch.forward_mode.is_draft_extend(include_v2=True):
                # 存在前缀缓存或草稿扩展时：max_seq_len_q 和 cu_seqlens_q 需要单独计算
                extend_seq_lens = forward_batch.extend_seq_lens
                # 注意：在分段式 CUDA Graph warmup 中，extend_seq_lens_cpu 可能是 0-d tensor
                max_q = max(forward_batch.extend_seq_lens_cpu)
                metadata.max_seq_len_q = (
                    int(max_q.item()) if isinstance(max_q, torch.Tensor) else int(max_q)
                )
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
            else:
                # 无前缀缓存的普通预填充：q 和 k 的长度相同
                metadata.max_seq_len_q = metadata.max_seq_len_k
                metadata.cu_seqlens_q = metadata.cu_seqlens_k

        # 计算 SWA 分页表（非 SWA 模型返回 None）
        metadata.swa_page_table = self._maybe_translate_swa(metadata.page_table)

        # 将分页表转换为步幅格式（strided format）：每 page_size 个 token 取一个页起始索引
        if self.page_size > 1:
            self.strided_indices = torch.arange(
                0, metadata.page_table.shape[1], self.page_size, device=self.device
            )
            # token 索引 → 页编号：除以 page_size
            metadata.page_table = (
                metadata.page_table[:, self.strided_indices] // self.page_size
            )
            if metadata.swa_page_table is not None:
                metadata.swa_page_table = (
                    metadata.swa_page_table[:, self.strided_indices] // self.page_size
                )

        self.forward_metadata = metadata

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """使用 TRT-LLM MHA 核执行解码阶段前向传播。"""
        cache_loc = forward_batch.out_cache_loc

        # 判断是否走融合 FP8 路径（量化 + KV 缓存写入一步完成）
        use_fused_fp8_path = self._should_use_fused_fp8_path(save_kv_cache, k)

        if use_fused_fp8_path:
            # 融合 FP8 路径：量化并写入 KV 缓存，之后 k/v 设为 None（核直接读缓存）
            self._fused_fp8_set_kv_buffer(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
            )
            k = None
            v = None
        else:
            # 普通路径：调用标准 set_kv_buffer 写入 KV 缓存
            if save_kv_cache and k is not None:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        # XQA 后端要求 q_type 为 bf16；TRT-LLM-GEN 且 KV 为 fp8 时 q 转换为 fp8
        if self.data_type == torch.float8_e4m3fn and (not self.is_xqa_impl):
            q = q.to(torch.float8_e4m3fn)
        q = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
        k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        # KV 缓存 reshape：[num_pages, page_size, num_kv_heads, head_dim]
        #            → [num_pages, num_kv_heads, page_size, head_dim]
        # TRT-LLM 核要求 num_kv_heads 在 page_size 前
        k_cache = k_cache.view(
            -1, self.page_size, layer.tp_k_head_num, layer.head_dim
        ).permute(0, 2, 1, 3)
        v_cache = v_cache.view(
            -1, self.page_size, layer.tp_v_head_num, layer.head_dim
        ).permute(0, 2, 1, 3)

        # 当 num_kv_heads=1 时，permute 可能产生退化步幅，需要修正以避免 TMA 错误
        if layer.tp_k_head_num == 1:
            k_cache = canonicalize_stride(k_cache)
        if layer.tp_v_head_num == 1:
            v_cache = canonicalize_stride(v_cache)

        kv_cache = (k_cache, v_cache)

        # 计算 BMM1 缩放因子：q_scale * k_scale * attention_scaling
        q_scale = 1.0
        k_scale = (
            layer.k_scale_float
            if getattr(layer, "k_scale_float", None) is not None
            else 1.0
        )
        bmm1_scale = q_scale * k_scale * layer.scaling
        bmm2_scale = 1.0
        # attention_sink：注意力 sink（稳定 softmax 的额外分母项）
        attention_sink = kwargs.get("sinks", None)

        page_table = self._get_layer_page_table(layer, forward_batch)

        # 调用 TRT-LLM 解码核：输出形状为 [bs, acc_q_len, num_q_heads, head_dim]
        o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=kv_cache,
            workspace_buffer=self.workspace_buffer,
            block_tables=page_table,
            seq_lens=self.forward_metadata.cache_seqlens_int32,
            max_seq_len=self.max_context_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            window_left=layer.sliding_window_size,
            sinks=attention_sink,
            skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
            out_dtype=self.q_data_type,  # 输出数据类型与模型精度一致
        )

        # 展平输出为 (num_tokens, num_heads * head_dim)
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        **kwargs,
    ):
        """使用 TRT-LLM MHA 核执行预填充阶段前向传播（含 target_verify 和 draft_extend）。"""
        cache_loc = forward_batch.out_cache_loc

        # 判断是否走融合 FP8 路径
        use_fused_fp8_path = self._should_use_fused_fp8_path(save_kv_cache, k)

        if use_fused_fp8_path:
            # 融合 FP8 路径
            self._fused_fp8_set_kv_buffer(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
            )
            k = None
            v = None
        else:
            # 普通路径
            if save_kv_cache and k is not None:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        # 预填充时统一将 q 转换为 fp8（若 KV 数据类型为 fp8）
        if self.data_type == torch.float8_e4m3fn:
            q = q.to(torch.float8_e4m3fn)
        q = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
        # KV 缓存 reshape：[num_pages, page_size, num_kv_heads, head_dim]
        #               → [num_pages, num_kv_heads, page_size, head_dim]
        k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        k_cache = k_cache.view(
            -1, self.page_size, layer.tp_k_head_num, layer.head_dim
        ).permute(0, 2, 1, 3)
        v_cache = v_cache.view(
            -1, self.page_size, layer.tp_v_head_num, layer.head_dim
        ).permute(0, 2, 1, 3)

        # 修正 num_kv_heads=1 时的退化步幅
        if layer.tp_k_head_num == 1:
            k_cache = canonicalize_stride(k_cache)
        if layer.tp_v_head_num == 1:
            v_cache = canonicalize_stride(v_cache)

        kv_cache = (k_cache, v_cache)

        # attention_sink 参数（可选）
        attention_sink = kwargs.get("sinks", None)
        q_scale = 1.0
        k_scale = (
            layer.k_scale_float
            if getattr(layer, "k_scale_float", None) is not None
            else 1.0
        )
        bmm1_scale = q_scale * k_scale * layer.scaling
        bmm2_scale = 1.0

        page_table = self._get_layer_page_table(layer, forward_batch)

        if forward_batch.forward_mode.is_target_verify():
            # 目标验证：q_len_per_req = speculative_num_draft_tokens，使用解码核（每请求多 query）
            o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=self.workspace_buffer,
                block_tables=page_table,
                seq_lens=self.forward_metadata.cache_seqlens_int32,
                max_seq_len=self.max_context_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                window_left=layer.sliding_window_size,
                sinks=attention_sink,
                skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
                out_dtype=self.q_data_type,
                q_len_per_req=self.forward_metadata.max_seq_len_q,
            )
        else:
            # 普通预填充（含 draft_extend）：使用预填充核，支持不等长序列
            o = flashinfer.prefill.trtllm_batch_context_with_kv_cache(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=self.workspace_buffer,
                block_tables=page_table,
                seq_lens=self.forward_metadata.cache_seqlens_int32,
                max_q_len=self.forward_metadata.max_seq_len_q,
                max_kv_len=self.max_context_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                batch_size=self.forward_metadata.cu_seqlens_q.shape[0] - 1,
                cum_seq_lens_q=self.forward_metadata.cu_seqlens_q,
                cum_seq_lens_kv=self.forward_metadata.cu_seqlens_k,
                window_left=layer.sliding_window_size,
                sinks=attention_sink,
                skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR.get(),
                out_dtype=self.q_data_type,
            )

        # 展平输出为 (num_tokens, num_heads * head_dim)
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)


class TRTLLMHAAttnMultiStepDraftBackend(FlashInferMultiStepDraftBackend):
    """用于 EAGLE 投机解码的多步 TRT-LLM MHA 注意力后端。
    将多个 TRTLLMHAAttnBackend 实例组合，分别处理各草稿解码步骤。
    """

    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        super().__init__(model_runner, topk, speculative_num_steps)
        # 将父类创建的每个草稿步骤后端替换为 TRTLLMHAAttnBackend
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i] = TRTLLMHAAttnBackend(
                model_runner,
                skip_prefill=True,                    # 草稿步骤只做解码
                kv_indptr_buf=self.kv_indptr[i],      # 复用父类分配的 kv_indptr 行
                kv_last_page_len_buf=self.kv_last_page_len,
                speculative_step_id=i,                # 草稿步骤编号（影响 KV 长度偏移）
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """初始化所有草稿步骤的 forward 元数据。"""
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """为所有草稿步骤预分配 CUDA Graph 静态缓冲区。"""
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(
        self,
        forward_batch: ForwardBatch,
    ):
        """CUDA Graph 捕获阶段：为每个草稿步骤初始化并缓存元数据。"""
        assert forward_batch.spec_info is not None
        assert forward_batch.spec_info.is_draft_input()

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=forward_batch.encoder_lens,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        """CUDA Graph 回放阶段：原地更新每个草稿步骤的元数据。"""
        assert forward_batch.spec_info is not None
        assert forward_batch.spec_info.is_draft_input()

        for i in range(self.speculative_num_steps - 1):

            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                encoder_lens=forward_batch.encoder_lens,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
            )
