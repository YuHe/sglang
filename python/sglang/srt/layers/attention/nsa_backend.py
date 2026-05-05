"""NSA（Native Sparse Attention）后端实现。

本模块实现了 DeepSeek NSA 稀疏注意力机制，通过动态 top-k 稀疏索引
将每个 token 的注意力范围限制在最重要的 KV 块上，大幅降低长序列的计算量。
支持 FlashMLA、FA3、TileLang 等多种底层内核，以及 FP8 KV 缓存量化。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, TypeAlias

import torch

# NSA 模型配置辅助函数
from sglang.srt.configs.model_config import get_nsa_index_topk, is_deepseek_nsa
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
# NSA KV 缓存量化/反量化工具
from sglang.srt.layers.attention.nsa.dequant_k_cache import dequantize_k_cache_paged
from sglang.srt.layers.attention.nsa.nsa_backend_mtp_precompute import (
    NativeSparseAttnBackendMTPPrecomputeMixin,
    PrecomputedMetadata,
    compute_cu_seqlens,
)
# NSA 稀疏索引器基类
from sglang.srt.layers.attention.nsa.nsa_indexer import BaseIndexerMetadata
from sglang.srt.layers.attention.nsa.quant_k_cache import quantize_k_cache
# NSA 索引变换工具（page_size=1 与 real_page_size 互转）
from sglang.srt.layers.attention.nsa.transform_index import (
    transform_index_page_table_decode,
    transform_index_page_table_prefill,
)
# NSA CP（上下文并行）相关工具
from sglang.srt.layers.attention.nsa.utils import (
    can_nsa_prefill_cp_round_robin_split,
    compute_nsa_seqlens,
    is_nsa_enable_prefill_cp,
    nsa_cp_round_robin_split_data,
    nsa_cp_round_robin_split_q_seqs,
    pad_nsa_cache_seqlens,
)
# 注意力工具函数：concat_mla_absorb_q、FP8 量化/RoPE、seqlens 扩展
from sglang.srt.layers.attention.utils import (
    concat_mla_absorb_q_general,
    mla_quantize_and_rope_for_fp8,
    seqlens_expand_triton,
)
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_cuda, is_hip

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput


# 检测当前平台是否为 AMD ROCm（HIP）
_is_hip = is_hip()

if _is_hip:
    # AMD HIP 平台：导入 NSA Triton kernel 和 AITER 注意力内核
    from sglang.srt.layers.attention.nsa.triton_kernel import get_valid_kv_indices

    try:
        from aiter import (  # noqa: F401
            flash_attn_varlen_func,
            mha_batch_prefill_func,
            paged_attention_ragged,
        )
        from aiter.mla import mla_decode_fwd, mla_prefill_fwd  # noqa: F401
    except ImportError:
        print(
            "aiter is AMD specific kernel library. Please make sure aiter is installed on your AMD device."
        )
else:
    # CUDA 平台：导入 SGLang JIT FlashAttention 内核
    from sglang.jit_kernel.flash_attention import (
        flash_attn_varlen_func,
        flash_attn_with_kvcache,
    )


# Reuse this workspace buffer across all NSA backend instances
# 全局 workspace 缓冲区（跨 NSA 后端实例复用，减少显存分配）
global_workspace_buffer = None

# Control whether to use fused metadata copy kernel for cuda graph replay (default: enabled)
# Set SGLANG_USE_FUSED_METADATA_COPY=0 or false to disable
# 是否使用融合元数据复制内核加速 CUDA Graph 回放（AMD 平台不支持）
_USE_FUSED_METADATA_COPY = envs.SGLANG_USE_FUSED_METADATA_COPY.get() and not _is_hip


@dataclass(frozen=True)
class NSAFlashMLAMetadata:
    """FlashMLA 专用的注意力调度元数据。

    flashmla_metadata：FlashMLA 内核所需的分块调度信息。
    num_splits：每条序列的 KV 分块数（用于 softmax LSE 合并）。
    """

    flashmla_metadata: torch.Tensor  # FlashMLA 内核调度元数据张量
    num_splits: torch.Tensor  # 每条序列的 KV 分块数

    def slice(self, sli):
        """对 num_splits 做切片（flashmla_metadata 不变）。"""
        return NSAFlashMLAMetadata(
            flashmla_metadata=self.flashmla_metadata,
            num_splits=self.num_splits[sli],
        )

    def copy_(self, other: "NSAFlashMLAMetadata"):
        """就地复制 FlashMLA 元数据（用于 CUDA Graph 静态缓冲区更新）。"""
        self.flashmla_metadata.copy_(other.flashmla_metadata)
        self.num_splits.copy_(other.num_splits)


@dataclass(frozen=True)
class NSAMetadata:
    """NSA 稀疏注意力批次元数据，汇集所有内核所需的指针和形状信息。"""

    page_size: int  # KV 缓存的 page size（实际块大小）

    # Sequence lengths for the forward batch
    # cache_seqlens_int32：每条序列的 KV 缓存长度（int32，GPU 张量）
    cache_seqlens_int32: torch.Tensor
    # Maximum sequence length for query
    max_seq_len_q: int  # 当前批次中最大 Q 序列长度
    # Maximum sequence length for key
    max_seq_len_k: int  # 当前批次中最大 KV 序列长度
    # Cumulative sequence lengths for query
    cu_seqlens_q: torch.Tensor  # Q 序列的前缀和指针
    # Cumulative sequence lengths for key
    cu_seqlens_k: torch.Tensor  # KV 序列的前缀和指针
    # Page table, the index of KV Cache Tables/Blocks
    # this table is always with page_size = 1
    # page_table_1：page_size=1 格式的 KV 块索引表（逐 token 粒度）
    page_table_1: torch.Tensor

    # NOTE(dark): This will property be used in:
    # 1. dense decode/prefill, we use paged flash attention, need real_page_table
    # 2. sparse decode/prefill, indexer need real_page_table to compute the score
    # real_page_table：真实 page size 格式的 KV 块索引表（供 NSA 索引器使用）
    real_page_table: torch.Tensor

    # NSA metadata (nsa prefill are expanded)
    # nsa_cache_seqlens_int32：NSA 每条序列的 KV 长度，截断为 topk
    nsa_cache_seqlens_int32: torch.Tensor  # this seqlens is clipped to `topk`
    # nsa_cu_seqlens_q：NSA 预填充扩展后的 Q 前缀和，始终为 arange(0, n)
    nsa_cu_seqlens_q: torch.Tensor  # must be arange(0, len(nsa_cu_seqlens_k))
    # nsa_cu_seqlens_k：NSA KV 前缀和（由 nsa_cache_seqlens_int32 cumsum 得到）
    nsa_cu_seqlens_k: torch.Tensor  # cumsum of `nsa_cache_seqlens_int32`
    # nsa_extend_seq_lens_list：CPU 侧每条序列的 NSA 扩展长度列表
    nsa_extend_seq_lens_list: List[int]
    # nsa_seqlens_expanded：扩展后（未截断）的序列长度（用于 topk 变换）
    nsa_seqlens_expanded: torch.Tensor  # expanded, unclipped `seqlens`
    # nsa_max_seqlen_q：NSA Q 序列最大长度（解码时始终为 1）
    nsa_max_seqlen_q: Literal[1] = 1  # always 1 for decode, variable for extend

    # flashmla_metadata：FlashMLA 专用调度元数据（可选）
    flashmla_metadata: Optional[NSAFlashMLAMetadata] = None
    # DeepGEMM schedule metadata for paged MQA logits (decode/target_verify/draft_extend only).
    # Precomputed once per forward batch and reused across layers.
    # paged_mqa_schedule_metadata：分页 MQA logits 的 DeepGEMM 调度元数据（解码阶段）
    paged_mqa_schedule_metadata: Optional[torch.Tensor] = None
    # The sum of sequence lengths for key, prefill only
    seq_lens_sum: Optional[int] = None  # 预填充专用：KV 序列长度之和
    # The flattened 1D page table with shape (seq_lens_sum,), prefill only
    # this table is always with page_size = 1
    # page_table_1_flattened：预填充专用：扁平化的 page_size=1 page table
    page_table_1_flattened: Optional[torch.Tensor] = None
    # The offset of topk indices in ragged kv, prefill only
    # shape: (seq_lens_sum,)
    # topk_indices_offset：预填充专用：ragged KV 中每个 token 的 topk 索引偏移
    topk_indices_offset: Optional[torch.Tensor] = None

    # k_start and k_end in kv cache for each token.
    # indexer_k_start_end：索引器中每个 token 在 KV 缓存中的起止位置
    indexer_k_start_end: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    # seq lens for each batch.
    # indexer_seq_lens_cpu：索引器用的 CPU 侧序列长度
    indexer_seq_lens_cpu: Optional[torch.Tensor] = None
    # seq lens for each batch.
    # indexer_seq_lens：索引器用的 GPU 侧序列长度
    indexer_seq_lens: Optional[torch.Tensor] = None
    # batch index for each token.
    # token_to_batch_idx：每个 token 对应的批次索引
    token_to_batch_idx: Optional[torch.Tensor] = None


class TopkTransformMethod(IntEnum):
    # Transform topk indices to indices to the page table (page_size = 1)
    # PAGED：将 topk 索引转换为 page_size=1 格式的 page table 索引
    PAGED = auto()
    # Transform topk indices to indices to ragged kv (non-paged)
    # RAGGED：将 topk 索引转换为 ragged（非分页）KV 格式的索引
    RAGGED = auto()


@torch.compile
def _compiled_cat(tensors: list[torch.Tensor], dim: int = -1) -> torch.Tensor:
    # torch.compile 版本的 cat，提升融合性能
    return torch.cat(tensors, dim=dim)


def _cat(tensors: list[torch.Tensor], dim: int = -1) -> torch.Tensor:
    """
    Concatenate two tensors along the last dimension.
    Use this function to concatenate q_nope and q_rope or k_nope and k_rope.
    将 q_nope 与 q_rope（或 k_nope 与 k_rope）沿最后一维拼接，
    并通过 torch.compile 加速，同时标记 batch 维为动态维度。
    """
    assert len(tensors) == 2

    qk_nope, qk_rope = tensors
    assert qk_nope.ndim == 3 and qk_rope.ndim == 3

    # 标记 batch 维为动态维度，避免 torch.compile 对不同批量大小重编译
    torch._dynamo.mark_dynamic(qk_nope, 0)
    torch._dynamo.mark_dynamic(qk_rope, 0)

    return _compiled_cat([qk_nope, qk_rope], dim=dim)


@dataclass(frozen=True)
class NSAIndexerMetadata(BaseIndexerMetadata):
    """NSA 索引器元数据：封装 NSAMetadata 和 topk 变换方法。"""

    attn_metadata: NSAMetadata  # 包含完整 NSA 注意力元数据
    topk_transform_method: TopkTransformMethod  # topk 索引变换方法
    paged_mqa_schedule_metadata: Optional[torch.Tensor] = None  # MQA 调度元数据
    force_unfused_topk: bool = False  # 是否强制使用非融合 topk 内核

    def get_seqlens_int32(self) -> torch.Tensor:
        """返回 KV 缓存序列长度（int32 格式）。"""
        return self.attn_metadata.cache_seqlens_int32

    def get_page_table_64(self) -> torch.Tensor:
        """返回真实 page size 的 64-bit page table。"""
        return self.attn_metadata.real_page_table

    def get_page_table_1(self) -> torch.Tensor:
        """返回 page_size=1 格式的 page table（逐 token 粒度）。"""
        return self.attn_metadata.page_table_1

    def get_seqlens_expanded(self) -> torch.Tensor:
        """返回 NSA 扩展后（未截断）的序列长度。"""
        return self.attn_metadata.nsa_seqlens_expanded

    def get_cu_seqlens_k(self) -> torch.Tensor:
        """返回 KV 序列的前缀和指针。"""
        return self.attn_metadata.cu_seqlens_k

    def get_indexer_kvcache_range(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回索引器中每个 token 在 KV 缓存中的起止位置。"""
        return self.attn_metadata.indexer_k_start_end

    def get_indexer_seq_len(self) -> torch.Tensor:
        """返回 GPU 侧索引器序列长度。"""
        return self.attn_metadata.indexer_seq_lens

    def get_indexer_seq_len_cpu(self) -> torch.Tensor:
        """返回 CPU 侧索引器序列长度。"""
        return self.attn_metadata.indexer_seq_lens_cpu

    def get_nsa_extend_len_cpu(self) -> List[int]:
        """返回 CPU 侧 NSA 扩展序列长度列表。"""
        return self.attn_metadata.nsa_extend_seq_lens_list

    def get_token_to_batch_idx(self) -> torch.Tensor:
        """返回每个 token 对应的批次索引。"""
        return self.attn_metadata.token_to_batch_idx

    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
        ks: Optional[torch.Tensor] = None,
        cu_seqlens_q: torch.Tensor = None,
        ke_offset: torch.Tensor = None,
        batch_idx_list: List[int] = None,
        topk_indices_offset_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """将 logits 转换为 topk 稀疏索引。

        根据 topk_transform_method 选择 PAGED 或 RAGGED 两种融合内核，
        支持自定义 cu_seqlens_q 和 ke_offset 用于灵活的 topk 计算场景。
        """
        from sgl_kernel import (
            fast_topk_transform_fused,
            fast_topk_transform_ragged_fused,
            fast_topk_v2,
        )

        if topk_indices_offset_override is not None:
            # 使用外部传入的 topk indices offset（优先级最高）
            cu_topk_indices_offset = topk_indices_offset_override
            cu_seqlens_q_topk = None
        elif cu_seqlens_q is not None:
            # 根据 cu_seqlens_q 计算 topk 级别的前缀和
            cu_seqlens_q = cu_seqlens_q.to(torch.int32)
            cu_seqlens_q_topk = compute_cu_seqlens(cu_seqlens_q)
            cu_topk_indices_offset = torch.repeat_interleave(
                cu_seqlens_q_topk[:-1],
                cu_seqlens_q,
            )
        else:
            # 使用 NSA 元数据中预计算的 topk 前缀和和偏移
            cu_seqlens_q_topk = self.attn_metadata.cu_seqlens_q
            cu_topk_indices_offset = self.attn_metadata.topk_indices_offset
        if ke_offset is not None:
            # 自定义 KE 偏移（用于特定 batch 的 topk 计算）
            seq_lens_topk = ke_offset
        else:
            seq_lens_topk = self.get_seqlens_expanded()
        if batch_idx_list is not None:
            # 按 batch_idx_list 筛选 page_table_1 的子集
            page_table_size_1 = self.attn_metadata.page_table_1[batch_idx_list]
        else:
            page_table_size_1 = self.attn_metadata.page_table_1

        if not envs.SGLANG_NSA_FUSE_TOPK.get() or self.force_unfused_topk:
            # 非融合路径：直接调用 fast_topk_v2 计算 topk 索引
            return fast_topk_v2(logits, seq_lens_topk, topk, row_starts=ks)
        elif self.topk_transform_method == TopkTransformMethod.PAGED:
            # NOTE(dark): if fused, we return a transformed page table directly
            # PAGED 融合路径：返回已变换的 page table（跳过独立变换步骤）
            return fast_topk_transform_fused(
                score=logits,
                lengths=seq_lens_topk,
                page_table_size_1=page_table_size_1,
                cu_seqlens_q=cu_seqlens_q_topk,
                topk=topk,
                row_starts=ks,
            )
        elif self.topk_transform_method == TopkTransformMethod.RAGGED:
            if cu_topk_indices_offset is None:
                raise RuntimeError(
                    "RAGGED topk_transform requires topk_indices_offset; "
                    "expected extend-without-speculative metadata."
                )
            # RAGGED 融合路径：返回 ragged KV 格式的 topk 索引
            return fast_topk_transform_ragged_fused(
                score=logits,
                lengths=seq_lens_topk,
                topk_indices_offset=cu_topk_indices_offset,
                topk=topk,
                row_starts=ks,
            )
        else:
            assert False, f"Unsupported {self.topk_transform_method = }"


# NSA 底层实现类型别名：支持多种内核后端
_NSA_IMPL_T: TypeAlias = Literal[
    "flashmla_sparse", "flashmla_kv", "fa3", "tilelang", "trtllm"
]


class NativeSparseAttnBackend(
    NativeSparseAttnBackendMTPPrecomputeMixin, AttentionBackend
):
    """DeepSeek NSA（原生稀疏注意力）后端。

    混入 NativeSparseAttnBackendMTPPrecomputeMixin 以支持 MTP（Multi-Token Prediction）
    预计算优化。实现了 NSA 稀疏 topk 注意力的完整推理流程，包括：
    - 解码、预填充、目标验证、草稿扩展等多种前向模式
    - FlashMLA / FA3 / TileLang / TRT-LLM 等底层内核的统一分发
    - CUDA Graph 静态缓冲区分配与回放
    - FP8 KV 缓存量化支持
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        speculative_step_id=0,
        topk=0,
        speculative_num_steps=0,
    ):
        super().__init__()
        self.forward_metadata: NSAMetadata  # 当前批次的注意力元数据
        self.device = model_runner.device
        assert isinstance(model_runner.page_size, int)
        # real_page_size：KV 缓存的真实 page 大小（可能 > 1）
        self.real_page_size = model_runner.page_size
        # num_splits：确定性推理时强制使用 1 个 KV 分块（可重现性）
        self.num_splits = (
            1 if model_runner.server_args.enable_deterministic_inference else 0
        )
        # 验证模型为 DeepSeek NSA 架构
        self.use_nsa = is_deepseek_nsa(model_runner.model_config.hf_config)
        assert self.use_nsa, "NSA backend only supports DeepSeek NSA"
        # 是否以 FP8 格式存储 NSA KV 缓存
        self.nsa_kv_cache_store_fp8 = (
            model_runner.token_to_kv_pool.nsa_kv_cache_store_fp8
        )
        # NSA 稀疏索引的 topk 参数（每次注意力计算保留的最重要 KV 块数）
        self.nsa_index_topk = get_nsa_index_topk(model_runner.model_config.hf_config)
        self.max_context_len = model_runner.model_config.context_len
        # TP 后每个设备的 Q 头数
        self.num_q_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.kv_cache_dim = model_runner.token_to_kv_pool.kv_cache_dim
        # MLA 维度分解：nope（无位置编码）和 rope（旋转位置编码）部分
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim

        assert model_runner.req_to_token_pool is not None
        # req_to_token：请求池到 KV 物理位置的映射表
        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        self.use_mha: bool = False  # 是否使用 MHA（而非 MLA）
        # nsa_prefill_impl：预填充阶段使用的内核后端名称
        self.nsa_prefill_impl: _NSA_IMPL_T = (
            model_runner.server_args.nsa_prefill_backend
        )
        # nsa_decode_impl：解码阶段使用的内核后端名称
        self.nsa_decode_impl: _NSA_IMPL_T = model_runner.server_args.nsa_decode_backend
        # flashmla_kv_num_q_heads：FlashMLA KV 路径所需的对齐 Q 头数（64 或 128）
        if self.num_q_heads <= 64:
            self.flashmla_kv_num_q_heads = 64
        elif self.num_q_heads <= 128:
            self.flashmla_kv_num_q_heads = 128
        else:
            # Keep original head count if it exceeds current padded variants.
            # 超过 128 时保持原始头数（内核自动适配）
            self.flashmla_kv_num_q_heads = self.num_q_heads
        # enable_auto_select_prefill_impl：是否动态选择预填充内核
        self.enable_auto_select_prefill_impl = self.nsa_prefill_impl == "flashmla_auto"

        # _arange_buf：预分配的 int32 arange 缓冲区（避免频繁 GPU 分配）
        self._arange_buf = torch.arange(16384, device=self.device, dtype=torch.int32)

        if _is_hip:
            # AMD HIP 平台专用：预分配 KV 指针和索引缓冲区
            max_bs = model_runner.req_to_token_pool.size

            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )

            self.kv_indices = torch.zeros(
                max_bs * self.nsa_index_topk,
                dtype=torch.int32,
                device=self.device,
            )
            # Aiter mla_decode_fwd supports num_heads multiples of 16 in range [16, 128].
            # For models with fewer heads per GPU (e.g. GLM-5 64 heads / TP8 = 8), need to pad the heads to 16.
            # 当每 GPU Q 头数 < 16 时需要 padding 到 16（AITER 内核限制）
            self.need_pad_heads = self.num_q_heads < 16
            self.head_repeat_factor = (
                16 // self.num_q_heads if self.num_q_heads < 16 else 1
            )

        # Speculative decoding
        # 投机解码参数
        self.topk = model_runner.server_args.speculative_eagle_topk or 0
        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
        )
        self.speculative_step_id = speculative_step_id  # 当前草稿解码步骤编号

        self.device_capability = torch.cuda.get_device_capability()
        self.device_sm_major = self.device_capability[0]  # GPU SM 主版本号
        self.kv_cache_dtype = model_runner.kv_cache_dtype

        # Allocate global workspace buffer for TRT-LLM kernels (ragged attention on SM100/B200, or trtllm decode)
        # SM100（Blackwell）及以上或使用 TRT-LLM 解码内核时需要 workspace 缓冲区
        if self.device_sm_major >= 10 or self.nsa_decode_impl == "trtllm":
            global global_workspace_buffer
            if global_workspace_buffer is None:
                # 首次初始化时分配全局 workspace 缓冲区（后续实例复用）
                global_workspace_buffer = torch.empty(
                    envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),
                    dtype=torch.uint8,
                    device=model_runner.device,
                )
            self.workspace_buffer = global_workspace_buffer
        else:
            self.workspace_buffer = None

    def get_device_int32_arange(self, l: int) -> torch.Tensor:
        """返回长度为 l 的 GPU int32 arange（自动扩容）。"""
        if l > len(self._arange_buf):
            # 扩容到下一个 2 的幂次（减少频繁重分配）
            next_pow_of_2 = 1 << (l - 1).bit_length()
            self._arange_buf = torch.arange(
                next_pow_of_2, device=self.device, dtype=torch.int32
            )
        return self._arange_buf[:l]

    def _transform_table_1_to_real(self, page_table: torch.Tensor) -> torch.Tensor:
        """将 page_size=1 格式的 page table 转换为真实 page size 格式。

        通过步长索引取每个 page 的起始 token，再除以 page_size 得到 page 编号。
        """
        page_size = self.real_page_size
        if page_size == 1:
            # page_size=1 时无需转换
            return page_table
        max_seqlen_k = page_table.shape[1]
        # 按 page_size 步长取 page 起始索引
        strided_indices = torch.arange(
            0, max_seqlen_k, page_size, device=page_table.device, dtype=torch.int32
        )
        # 将 token 索引转换为 page 编号
        return page_table[:, strided_indices] // page_size

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """初始化 NSA 注意力后端的前向传播元数据。

        根据前向模式（解码、目标验证、草稿扩展、普通预填充）分别计算：
        - cache_seqlens_int32：KV 缓存序列长度
        - cu_seqlens_q / cu_seqlens_k：Q/KV 前缀和指针
        - page_table：KV 块索引表（page_size=1 格式）
        - seqlens_expanded：NSA topk 变换所需的扩展序列长度
        - real_page_table：真实 page size 格式的 KV 块索引表
        """
        batch_size = forward_batch.batch_size
        device = forward_batch.seq_lens.device

        if forward_batch.forward_mode.is_target_verify():
            # 目标验证模式：KV 序列包含草稿 token
            draft_token_num = self.speculative_num_draft_tokens
        else:
            draft_token_num = 0

        # cache_seqlens_int32：每条序列的 KV 缓存长度（含草稿 token）
        cache_seqlens_int32 = (forward_batch.seq_lens + draft_token_num).to(torch.int32)
        # cu_seqlens_k：KV 序列的前缀和指针
        cu_seqlens_k = compute_cu_seqlens(cache_seqlens_int32)
        assert forward_batch.seq_lens_cpu is not None
        # max_seqlen_k：当前批次最大 KV 序列长度（含草稿 token）
        max_seqlen_k = int(forward_batch.seq_lens_cpu.max().item() + draft_token_num)
        # [b, max_seqlen_k]
        # page_table：从请求池取当前批次的 KV 块索引表（page_size=1 格式）
        page_table = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :max_seqlen_k
        ]

        page_table_1_flattened = None  # 预填充专用的扁平化 page table
        topk_indices_offset = None  # RAGGED 格式的 topk 索引偏移

        # Centralized dispatch: decide all strategies for this batch
        # 集中分发：为当前批次决定预填充内核实现
        self.set_nsa_prefill_impl(forward_batch)
        # 根据前向模式选择解码或预填充实现
        nsa_impl_for_batch = (
            self.nsa_decode_impl
            if (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend(include_v2=True)
            )
            else self.nsa_prefill_impl
        )
        # 是否使用 FlashMLA KV 格式（非 MHA 且内核为 flashmla_kv）
        use_flashmla_kv = (not self.use_mha) and nsa_impl_for_batch == "flashmla_kv"
        # topk 变换方法：PAGED 或 RAGGED
        topk_transform_method = self.get_topk_transform_method(
            forward_batch.forward_mode
        )
        # Batch indices selected when cp enabled: After splitting multiple sequences,
        # a certain cp rank may not have some of these sequences.
        # We use bs_idx_cpu to mark which sequences are finally selected by the current cp rank,
        # a default value of None indicates that all sequences are selected.
        # bs_idx_cpu：CP（上下文并行）时当前 rank 选中的序列索引（None 表示全选）
        bs_idx_cpu = None
        # seq_len_cpu of selected sequences
        indexer_seq_lens_cpu = forward_batch.seq_lens_cpu
        indexer_seq_lens = forward_batch.seq_lens

        if forward_batch.forward_mode.is_decode_or_idle():
            # 解码模式：每条请求只有 1 个 Q token
            extend_seq_lens_cpu = [1] * batch_size
            max_seqlen_q = 1
            # cu_seqlens_q 为简单的 arange(0, batch_size+1)
            cu_seqlens_q = self.get_device_int32_arange(batch_size + 1)
            # seqlens_expanded：每个 Q token 对应完整 KV 序列长度
            seqlens_expanded = cache_seqlens_int32
        elif forward_batch.forward_mode.is_target_verify():
            # 目标验证模式：每条请求有 num_draft_tokens 个 Q token
            max_seqlen_q = 1
            cu_seqlens_q = torch.arange(
                0,
                batch_size * self.speculative_num_draft_tokens + 1,
                1,
                dtype=torch.int32,
                device=device,
            )
            extend_seq_lens_cpu = [self.speculative_num_draft_tokens] * batch_size
            forward_batch.extend_seq_lens_cpu = extend_seq_lens_cpu

            # 将 KV 序列长度扩展到每个草稿 token 都看到完整上下文
            seqlens_expanded = seqlens_expand_triton(
                torch.tensor(extend_seq_lens_cpu, dtype=torch.int32, device=device),
                cache_seqlens_int32,
                self.speculative_num_draft_tokens * batch_size,
                self.speculative_num_draft_tokens,
            )
            # 将 page_table 按草稿 token 数重复（每个草稿 token 看相同 KV 块）
            page_table = torch.repeat_interleave(
                page_table, repeats=self.speculative_num_draft_tokens, dim=0
            )
        elif forward_batch.forward_mode.is_draft_extend(include_v2=True):
            assert (
                forward_batch.extend_seq_lens_cpu is not None
                and forward_batch.extend_seq_lens is not None
                and forward_batch.extend_prefix_lens_cpu is not None
            ), "All of them must not be None"

            extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
            assert forward_batch.extend_seq_lens is not None

            max_seqlen_q = 1
            cu_seqlens_q = torch.arange(
                0,
                forward_batch.extend_num_tokens + 1,
                1,
                dtype=torch.int32,
                device=device,
            )

            # 将 KV 序列长度扩展到每个扩展 token 看到对应的 KV 上下文
            seqlens_expanded = seqlens_expand_triton(
                forward_batch.extend_seq_lens,
                cache_seqlens_int32,
                sum(extend_seq_lens_cpu),
                self.speculative_num_draft_tokens,
            )
            if forward_batch.forward_mode.is_draft_extend_v2():
                # DRAFT_EXTEND_V2: V2 worker pre-fills draft KV cache with ALL speculated
                # tokens upfront. All requests extend by the same fixed
                # (speculative_num_draft_tokens). Use scalar to avoid GPU sync.
                # EAGLE V2：所有请求扩展相同数量的草稿 token，用标量 repeats 避免 GPU 同步
                page_table = torch.repeat_interleave(
                    page_table, repeats=self.speculative_num_draft_tokens, dim=0
                )
            else:
                # DRAFT_EXTEND (v1): V1 worker extends by (num_accepted_drafts + 1) per request
                # after verification. Lengths vary per request based on how many tokens
                # were accepted.
                # EAGLE V1：每条请求扩展长度不同（实际接受的草稿 token 数）
                page_table = torch.repeat_interleave(
                    page_table, repeats=forward_batch.extend_seq_lens, dim=0
                )
        elif forward_batch.forward_mode.is_extend():
            assert (
                forward_batch.extend_seq_lens_cpu is not None
                and forward_batch.extend_seq_lens is not None
                and forward_batch.extend_prefix_lens_cpu is not None
            ), "All of them must not be None"
            extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
            assert forward_batch.extend_seq_lens is not None
            extend_seq_lens = forward_batch.extend_seq_lens

            # seqlens_expanded：对每个 Q token 生成从 kv_len-qo_len+1 到 kv_len 的序列长度
            seqlens_expanded = torch.cat(
                [
                    torch.arange(
                        kv_len - qo_len + 1,
                        kv_len + 1,
                        dtype=torch.int32,
                        device=device,
                    )
                    for qo_len, kv_len in zip(
                        forward_batch.extend_seq_lens_cpu,
                        forward_batch.seq_lens_cpu.tolist(),
                        strict=True,
                    )
                ]
            )

            if can_nsa_prefill_cp_round_robin_split(forward_batch):
                # CP round-robin 分割：将 Q/KV 序列分配到各 CP rank
                seqlens_expanded = nsa_cp_round_robin_split_data(seqlens_expanded)
                extend_seq_lens_cpu, extend_seq_lens, bs_idx_cpu, bs_idx = (
                    nsa_cp_round_robin_split_q_seqs(
                        extend_seq_lens_cpu, extend_seq_lens
                    )
                )
                # 根据 CP 分割结果更新序列长度、KV 前缀和、page table
                indexer_seq_lens_cpu = indexer_seq_lens_cpu[bs_idx_cpu]
                indexer_seq_lens = indexer_seq_lens[bs_idx]
                cache_seqlens_int32 = cache_seqlens_int32[bs_idx]
                cu_seqlens_k = compute_cu_seqlens(cache_seqlens_int32)
                max_seqlen_k = (
                    int(indexer_seq_lens_cpu.max().item() + draft_token_num)
                    if len(indexer_seq_lens_cpu) != 0
                    else 0
                )
                page_table = page_table[bs_idx, :max_seqlen_k]

            if (
                any(forward_batch.extend_prefix_lens_cpu)
                or forward_batch.forward_mode == ForwardMode.DRAFT_EXTEND
                or bs_idx_cpu is not None
            ):
                # 有前缀共享或 CP 分割时：动态计算 cu_seqlens_q
                max_seqlen_q = (
                    max(extend_seq_lens_cpu) if len(extend_seq_lens_cpu) != 0 else 1
                )
                cu_seqlens_q = compute_cu_seqlens(extend_seq_lens.to(torch.int32))
            else:
                # 无前缀：Q 和 KV 使用相同的 seqlens
                max_seqlen_q = max_seqlen_k
                cu_seqlens_q = cu_seqlens_k

            # Check if MHA FP8 dequantization is needed
            # 检查是否需要 MHA FP8 反量化（MHA 路径 + FP8 KV 缓存）
            mha_dequantize_needed = (
                self.use_mha
                and forward_batch.token_to_kv_pool.dtype == torch.float8_e4m3fn
            )
            forward_batch.using_mha_one_shot_fp8_dequant = mha_dequantize_needed

            # page_table_1_flattened is only used when prefix sharing is enabled:
            # page_table_1_flattened 仅在前缀共享时使用（RAGGED topk 或 MHA FP8 反量化）
            has_prefix_sharing = any(forward_batch.extend_prefix_lens_cpu)
            if has_prefix_sharing and (
                topk_transform_method == TopkTransformMethod.RAGGED
                or mha_dequantize_needed
            ):
                # 构建扁平化的 page table（将所有序列的 page 索引连接成 1D）
                page_table_1_flattened = torch.cat(
                    [
                        page_table[i, :kv_len]
                        for i, kv_len in enumerate(
                            indexer_seq_lens_cpu.tolist(),
                        )
                    ]
                )
                assert page_table_1_flattened.shape[0] == sum(
                    indexer_seq_lens_cpu
                ), f"{page_table_1_flattened.shape[0] = } must be the same as {sum(indexer_seq_lens_cpu) = }"

                # Validate indices when logical tokens exceed physical capacity
                # This is likely to be triggered by PP with high kv reuse & parallelism
                # 验证 page table 索引不超出 KV 缓存容量（PP + 高 KV 复用时容易触发）
                kv_cache_capacity = (
                    forward_batch.token_to_kv_pool.size
                    + forward_batch.token_to_kv_pool.page_size
                )
                if forward_batch.seq_lens_sum > kv_cache_capacity:
                    max_idx = page_table_1_flattened.max().item()
                    assert max_idx < kv_cache_capacity, (
                        f"Invalid page table index: max={max_idx}, "
                        f"kv_cache_capacity={kv_cache_capacity}"
                    )

            if topk_transform_method == TopkTransformMethod.RAGGED:
                # RAGGED 格式：计算每个 Q token 对应的 KV 起始偏移
                topk_indices_offset = torch.repeat_interleave(
                    cu_seqlens_k[:-1],
                    extend_seq_lens,
                )
        else:
            assert False, f"Unsupported {forward_batch.forward_mode = }"

        # 计算 prefill 中 indexer 的 K 起止位置及 token→batch 映射
        indexer_k_start_end, token_to_batch_idx = self._cal_indexer_k_start_end(
            forward_batch, bs_idx_cpu
        )
        # 1D, expanded seqlens (1D means cheap to compute, so always compute it)
        # 计算每个 Q token 对应的 NSA 缓存序列长度（剪裁到 topk 上限）
        nsa_cache_seqlens_int32 = compute_nsa_seqlens(
            original_seq_lens=seqlens_expanded,
            nsa_index_topk=self.nsa_index_topk,
        )
        # 对齐/填充 NSA 缓存序列长度，使其满足 kernel 对齐要求
        nsa_cache_seqlens_int32 = pad_nsa_cache_seqlens(
            forward_batch, nsa_cache_seqlens_int32
        )
        # 计算 NSA KV 的累积序列长度偏移（用于 FA3 / TileLang kernel）
        nsa_cu_seqlens_k = compute_cu_seqlens(nsa_cache_seqlens_int32)
        # NSA Q 侧 cu_seqlens：每个 token 占 1 行，用 arange 构造
        nsa_cu_seqlens_q = self.get_device_int32_arange(len(nsa_cu_seqlens_k))

        paged_mqa_schedule_metadata = None
        # DeepGEMM paged MQA logits path needs a schedule metadata tensor.
        # Compute it once per forward batch and reuse it across layers.
        # DeepGEMM SM90+ paged MQA 路径需要预先计算调度元数据
        if is_cuda() and (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend()
        ):
            try:
                import deep_gemm

                # NOTE: DeepGEMM paged path uses block_size=64.
                # decode 使用原始 cache_seqlens，verify/draft_extend 使用展开后的 seqlens
                seqlens_32 = (
                    seqlens_expanded
                    if (
                        forward_batch.forward_mode.is_target_verify()
                        or forward_batch.forward_mode.is_draft_extend()
                    )
                    else cache_seqlens_int32
                )
                # 获取 paged MQA logits 的调度元数据（block_size=64，使用全部 SM）
                paged_mqa_schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                    seqlens_32, 64, deep_gemm.get_num_sms()
                )
            except (ImportError, ModuleNotFoundError):
                paged_mqa_schedule_metadata = None

        # 组装本次 forward 的完整 NSAMetadata 对象，供各层共用
        metadata = NSAMetadata(
            page_size=self.real_page_size,
            cache_seqlens_int32=cache_seqlens_int32,
            max_seq_len_q=max_seqlen_q,
            max_seq_len_k=max_seqlen_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seq_lens_sum=forward_batch.seq_lens_sum,
            page_table_1=page_table,
            page_table_1_flattened=page_table_1_flattened,
            # 仅在 flashmla_kv 模式下计算 FlashMLA 元数据
            flashmla_metadata=(
                self._compute_flashmla_metadata(
                    cache_seqlens=nsa_cache_seqlens_int32,
                    seq_len_q=1,
                )
                if use_flashmla_kv
                else None
            ),
            paged_mqa_schedule_metadata=paged_mqa_schedule_metadata,
            nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
            nsa_cu_seqlens_q=nsa_cu_seqlens_q,
            nsa_cu_seqlens_k=nsa_cu_seqlens_k,
            nsa_seqlens_expanded=seqlens_expanded,
            nsa_extend_seq_lens_list=extend_seq_lens_cpu,
            # 将 page_size=1 的 page_table 转换为真实 page_size 的 page_table
            real_page_table=self._transform_table_1_to_real(page_table),
            nsa_max_seqlen_q=1,
            topk_indices_offset=topk_indices_offset,
            indexer_k_start_end=indexer_k_start_end,
            indexer_seq_lens_cpu=indexer_seq_lens_cpu,
            indexer_seq_lens=indexer_seq_lens,
            token_to_batch_idx=token_to_batch_idx,
        )
        # 将构造好的元数据存入实例，供后续 forward 方法取用
        self.forward_metadata = metadata

    def _cal_indexer_k_start_end(
        self,
        forward_batch: ForwardBatch,
        bs_idx: Optional[List[int]] = None,
    ):
        """计算 prefill indexer 中每个 Q token 对应的 K 起止位置。

        NSA prefill 需要知道每个 Q token 能"看到"的 KV 范围 [ks, ke)，
        即 causal mask 下该 token 对应的 KV 起始列（ks）和结束列（ke）。
        同时构造 token→batch 映射，用于聚合不同请求的 topk 得分。

        仅在 extend_without_speculative 模式下有效；其他模式返回 (None, None)。

        Args:
            forward_batch: 当前 forward 批次信息
            bs_idx: 仅处理选定的请求索引（Context Parallel 使用），None 表示全部

        Returns:
            ((ks, ke), token_to_batch_idx)：ks/ke 各为 [total_q_tokens] 的 int32 tensor；
            token_to_batch_idx 为每个 Q token 所属 batch 的索引。
        """
        if not forward_batch.forward_mode.is_extend_without_speculative():
            # 非 prefill 模式（decode / verify / draft_extend）无需此计算
            return None, None
        if forward_batch.batch_size == 0 or (bs_idx is not None and len(bs_idx) == 0):
            # 空批次：返回空张量
            empty_t = torch.empty(0, dtype=torch.int32, device=self.device)
            return (empty_t, empty_t), empty_t

        # Suppose there are two requests, with extend_seq_len = [3, 2]
        # and seq_lens = [10, 4]
        # The logits matrix looks like this, with * representing the valid logits
        # and - representing the invalid logits:
        # 示例：两个请求 extend_seq_len=[3,2]，seq_lens=[10,4]
        # logits 矩阵中 * 表示有效位置，- 表示被 mask 的位置
        #
        #  ********--|----
        #  *********-|----
        #  **********|----
        #  ----------|***-
        #  ----------|****
        #
        # ks = [0, 0, 0, 10, 10]
        # ke = [8, 9, 10, 13, 14]
        ks_list = []   # 每个 Q token 对应的 K 起始位置列表
        ke_list = []   # 每个 Q token 对应的 K 结束位置列表
        token_to_batch_idx = []   # 每个 Q token 所属的 batch 索引列表

        q_offset = 0   # 当前请求 Q token 在全局的起始偏移
        k_offset = 0   # 当前请求 K token 在全局的起始偏移

        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )
        for i in range(forward_batch.batch_size):
            # 获取当前请求的总序列长度和本次 extend 长度
            seq_len = forward_batch.seq_lens_cpu[i].item()
            assert isinstance(seq_len, int)
            extend_seq_len = forward_batch.extend_seq_lens_cpu[i]
            # 所有 Q token 的 K 起始位置均为 k_offset（当前请求 KV 的起点）
            ks = torch.full(
                (extend_seq_len,), k_offset, dtype=torch.int32, device=self.device
            )
            kv_len = seq_len
            if forward_batch.forward_mode.is_target_verify():
                # target_verify 模式：KV 还包含 draft token
                kv_len += self.speculative_num_draft_tokens
            # K 结束位置随 Q token 位置单调递增（causal mask）
            seq_lens_expanded = torch.arange(
                kv_len - extend_seq_len + 1,
                kv_len + 1,
                dtype=torch.int32,
                device=self.device,
            )
            ke = ks + seq_lens_expanded   # ke[j] = k_offset + (kv_len - extend_seq_len + 1 + j)
            ks_list.append(ks)
            ke_list.append(ke)

            # bi: The index within the selected batch bs_idx. Entries that were not selected are ignored.
            # bi: 在 bs_idx 中的索引（CP 分片时使用），未被选中的请求使用原始索引
            bi = bs_idx.index(i) if (bs_idx is not None and i in bs_idx) else i
            tb = torch.full(
                (extend_seq_len,), bi, dtype=torch.int32, device=self.device
            )
            token_to_batch_idx.append(tb)

            if bs_idx is None or i in bs_idx:  # skip batch not included in bs_idx
                # 仅统计被选中的请求的偏移
                q_offset += extend_seq_len
                k_offset += seq_len

        # 拼接所有请求的 ks/ke 及 token→batch 映射
        ks = torch.cat(ks_list, dim=0)
        ke = torch.cat(ke_list, dim=0)
        token_to_batch_idx = torch.cat(token_to_batch_idx, dim=0)
        if bs_idx is not None:
            # Context Parallel 模式：按 round-robin 方式重新排列数据
            assert can_nsa_prefill_cp_round_robin_split(forward_batch)
            ks = nsa_cp_round_robin_split_data(ks)
            ke = nsa_cp_round_robin_split_data(ke)
            token_to_batch_idx = nsa_cp_round_robin_split_data(token_to_batch_idx)
        return (ks, ke), token_to_batch_idx

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Initialize CUDA graph state for the attention backend.

        初始化 CUDA Graph 所需的静态缓冲区。
        CUDA Graph capture 阶段写入这些固定大小张量；replay 阶段只更新内容而不重新分配。

        Args:
            max_bs (int): Maximum batch size to support in CUDA graphs
            max_num_tokens (int): CUDA Graph 支持的最大 token 数（用于分配 page_table）

        This creates fixed-size tensors that will be reused during CUDA graph replay
        to avoid memory allocations.
        """
        self.decode_cuda_graph_metadata: Dict = {
            # 每个请求的 KV 缓存序列长度（decode 时等于 seq_len）
            "cache_seqlens": torch.ones(
                max_num_tokens, dtype=torch.int32, device=self.device
            ),
            # Q 侧累积序列长度：decode 时每请求 1 个 token，因此是 arange
            "cu_seqlens_q": torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=self.device
            ),
            # K 侧累积序列长度：初始化为 0，capture 时填入实际值
            "cu_seqlens_k": torch.zeros(
                max_bs + 1, dtype=torch.int32, device=self.device
            ),
            # fake page_table for sparse_prefill
            # Add extra columns for speculative draft tokens to avoid
            # overflow during target_verify when max_seqlen_k = seq_len + num_draft_tokens
            # 稀疏 prefill 用的假 page_table；额外列用于容纳 draft token
            "page_table": torch.zeros(
                max_num_tokens,
                self.max_context_len + (self.speculative_num_draft_tokens or 0),
                dtype=torch.int32,
                device=self.device,
            ),
            # 仅在 flashmla_kv 解码实现下预计算 FlashMLA 元数据
            "flashmla_metadata": (
                self._compute_flashmla_metadata(
                    cache_seqlens=torch.ones(
                        max_num_tokens, dtype=torch.int32, device=self.device
                    ),
                    seq_len_q=1,
                )
                if self.nsa_decode_impl == "flashmla_kv"
                else None
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
        self.set_nsa_prefill_impl(forward_batch=None)

        """Initialize forward metadata for capturing CUDA graph."""
        # ---- 普通 Decode 模式 ----
        if forward_mode.is_decode_or_idle():
            # Normal Decode
            # Get sequence information
            # 将序列长度转为 int32，作为 KV cache 序列长度
            cache_seqlens_int32 = seq_lens.to(torch.int32)
            cu_seqlens_k = compute_cu_seqlens(cache_seqlens_int32)

            # Use max context length for seq_len_k
            # 取静态 page_table 的前 bs 行（每行表示一个请求的 page 列表）
            page_table_1 = self.decode_cuda_graph_metadata["page_table"][:bs, :]
            max_seqlen_q = 1   # decode：每请求仅有 1 个 Q token
            max_seqlen_k = page_table_1.shape[1]   # K 最大长度 = max_context_len + draft_tokens

            # Precompute page table
            # Precompute cumulative sequence lengths

            # NOTE(dark): this is always arange, since we are decoding
            # decode 的 Q 侧 cu_seqlens 固定为 arange(0, bs+1)
            cu_seqlens_q = self.decode_cuda_graph_metadata["cu_seqlens_q"][: bs + 1]
            # 计算 NSA 剪裁后的 KV 序列长度
            nsa_cache_seqlens_int32 = compute_nsa_seqlens(
                cache_seqlens_int32, nsa_index_topk=self.nsa_index_topk
            )

            # decode 时 seqlens_expanded 与原始 cache_seqlens 相同
            seqlens_expanded = cache_seqlens_int32
            # 每个 token 的 extend_seq_len 均为 1（decode 每步只有 1 个新 token）
            nsa_extend_seq_lens_list = [1] * num_tokens
            if self.nsa_decode_impl == "flashmla_kv":
                # flashmla_kv 解码：切片静态 FlashMLA 元数据并更新为实际值
                flashmla_metadata = self.decode_cuda_graph_metadata[
                    "flashmla_metadata"
                ].slice(slice(0, num_tokens + 1))
                flashmla_metadata.copy_(
                    self._compute_flashmla_metadata(
                        cache_seqlens=nsa_cache_seqlens_int32,
                        seq_len_q=1,
                    )
                )
            else:
                flashmla_metadata = None
        # ---- target_verify 或 draft_extend 模式（投机解码）----
        elif forward_mode.is_target_verify() or forward_mode.is_draft_extend(
            include_v2=True
        ):
            # 在 seq_len 基础上加上 draft token 数量，得到实际 KV 长度
            cache_seqlens_int32 = (seq_lens + self.speculative_num_draft_tokens).to(
                torch.int32
            )
            cu_seqlens_k = compute_cu_seqlens(cache_seqlens_int32)
            max_seqlen_q = 1
            # 每个请求有 speculative_num_draft_tokens 个 Q token（draft token）
            page_table_1 = self.decode_cuda_graph_metadata["page_table"][
                : bs * self.speculative_num_draft_tokens, :
            ]
            max_seqlen_k = page_table_1.shape[1]

            # Q 侧 cu_seqlens：每请求有 speculative_num_draft_tokens 个 token
            cu_seqlens_q = torch.arange(
                0,
                bs * self.speculative_num_draft_tokens + 1,
                1,
                dtype=torch.int32,
                device=self.device,
            )

            # 每个请求的 extend_seq_len 均为 draft token 数量
            extend_seq_lens_cpu = [self.speculative_num_draft_tokens] * bs

            # 每请求的 kv_len = seq_len + speculative_num_draft_tokens
            seqlens_int32_cpu = [
                self.speculative_num_draft_tokens + kv_len
                for kv_len in seq_lens.tolist()
            ]
            # seqlens_expanded：每个 draft token 对应的 causal KV 长度（单调递增）
            seqlens_expanded = torch.cat(
                [
                    torch.arange(
                        kv_len - qo_len + 1,
                        kv_len + 1,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    for qo_len, kv_len in zip(
                        extend_seq_lens_cpu,
                        seqlens_int32_cpu,
                        strict=True,
                    )
                ]
            )
            # 基于展开后的 seqlens 计算 NSA 剪裁序列长度
            nsa_cache_seqlens_int32 = compute_nsa_seqlens(
                seqlens_expanded, nsa_index_topk=self.nsa_index_topk
            )
            # 所有 draft token 的 extend_seq_len 均为 1
            nsa_extend_seq_lens_list = [1] * bs * self.speculative_num_draft_tokens

            if self.nsa_decode_impl == "flashmla_kv":
                # 切片并更新 FlashMLA 元数据（bs * draft_tokens 个 Q token）
                flashmla_metadata = self.decode_cuda_graph_metadata[
                    "flashmla_metadata"
                ].slice(slice(0, bs * self.speculative_num_draft_tokens + 1))

                flashmla_metadata.copy_(
                    self._compute_flashmla_metadata(
                        cache_seqlens=nsa_cache_seqlens_int32,
                        seq_len_q=1,
                    )
                )
            else:
                flashmla_metadata = None

        # 计算 NSA KV 的累积序列长度（共用于 decode/verify/draft_extend）
        nsa_cu_seqlens_k = compute_cu_seqlens(nsa_cache_seqlens_int32)
        nsa_cu_seqlens_q = self.get_device_int32_arange(len(nsa_cu_seqlens_k))
        # 将 page_size=1 的 page_table 转换为真实 page_size
        real_page_table = self._transform_table_1_to_real(page_table_1)

        # DeepGEMM paged MQA 调度元数据（仅 CUDA 且非 HIP 时尝试）
        paged_mqa_schedule_metadata = None
        if is_cuda() and (
            forward_mode.is_decode_or_idle()
            or forward_mode.is_target_verify()
            or forward_mode.is_draft_extend()
        ):
            try:
                import deep_gemm

                # verify/draft_extend 使用展开后的 seqlens，decode 使用原始
                seqlens_32 = (
                    seqlens_expanded
                    if (
                        forward_mode.is_target_verify()
                        or forward_mode.is_draft_extend()
                    )
                    else cache_seqlens_int32
                )
                paged_mqa_schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                    seqlens_32, 64, deep_gemm.get_num_sms()
                )
            except (ImportError, ModuleNotFoundError):
                paged_mqa_schedule_metadata = None

        # 构造 capture 阶段的 NSAMetadata，存入 decode_cuda_graph_metadata[bs] 和 forward_metadata
        metadata = NSAMetadata(
            page_size=self.real_page_size,
            cache_seqlens_int32=cache_seqlens_int32,
            max_seq_len_q=max_seqlen_q,
            max_seq_len_k=max_seqlen_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            page_table_1=page_table_1,
            flashmla_metadata=flashmla_metadata,
            paged_mqa_schedule_metadata=paged_mqa_schedule_metadata,
            nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
            nsa_cu_seqlens_q=nsa_cu_seqlens_q,
            nsa_cu_seqlens_k=nsa_cu_seqlens_k,
            nsa_seqlens_expanded=seqlens_expanded,
            real_page_table=real_page_table,
            nsa_extend_seq_lens_list=nsa_extend_seq_lens_list,
        )
        # 以 bs 为 key 缓存 metadata，replay 时直接取用
        self.decode_cuda_graph_metadata[bs] = metadata
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
        out_cache_loc: Optional[torch.Tensor] = None,
    ):
        """Initialize forward metadata for replaying CUDA graph.

        CUDA Graph replay 阶段：只更新静态缓冲区的内容，不重新分配内存。
        从 decode_cuda_graph_metadata[bs] 取出 capture 阶段存储的 NSAMetadata，
        并用当前批次的真实 seq_lens / page_table / nsa_cache_seqlens 覆写其内容。
        """
        assert seq_lens_cpu is not None

        self.set_nsa_prefill_impl(forward_batch=None)

        # 截取当前批次大小对应的前 bs 个元素
        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs]
        req_pool_indices = req_pool_indices[:bs]

        # Normal Decode
        # 取出 capture 阶段已构造好的静态 NSAMetadata 对象
        metadata: NSAMetadata = self.decode_cuda_graph_metadata[bs]
        if forward_mode.is_decode_or_idle():
            # Normal Decode
            # 计算当前批次中最长的序列（用于确定 page_table 需要复制的列数）
            max_len = int(seq_lens_cpu.max().item())

            # 更新 KV 缓存序列长度（int32）
            cache_seqlens = seq_lens.to(torch.int32)
            metadata.cache_seqlens_int32.copy_(cache_seqlens)
            # 更新 K 侧累积序列长度
            metadata.cu_seqlens_k[1:].copy_(
                torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32)
            )
            # 从 req_to_token 表中读取每个请求的 page 索引并写入静态 page_table
            page_indices = self.req_to_token[req_pool_indices, :max_len]
            metadata.page_table_1[:, :max_len].copy_(page_indices)
            # 计算 NSA 剪裁序列长度并更新
            nsa_cache_seqlens = compute_nsa_seqlens(
                cache_seqlens, nsa_index_topk=self.nsa_index_topk
            )
            metadata.nsa_cache_seqlens_int32.copy_(nsa_cache_seqlens)
            # decode 时 seqlens_expanded 与 cache_seqlens 相同
            seqlens_expanded = cache_seqlens
        elif forward_mode.is_target_verify():
            # target_verify：KV 长度需加上 draft token 数
            max_seqlen_k = int(
                seq_lens_cpu.max().item() + self.speculative_num_draft_tokens
            )

            cache_seqlens = (seq_lens + self.speculative_num_draft_tokens).to(
                torch.int32
            )
            metadata.cache_seqlens_int32.copy_(cache_seqlens)
            metadata.cu_seqlens_k[1:].copy_(
                torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32)
            )
            # 读取 page 索引并按 draft_tokens 倍数重复（每个 draft token 共用相同 KV 页）
            page_indices = self.req_to_token[req_pool_indices, :max_seqlen_k]
            page_indices = torch.repeat_interleave(
                page_indices, repeats=self.speculative_num_draft_tokens, dim=0
            )
            metadata.page_table_1[:, :max_seqlen_k].copy_(page_indices)
            extend_seq_lens_cpu = [self.speculative_num_draft_tokens] * bs

            # 用 Triton kernel 展开 seqlens，生成每个 draft token 的 causal KV 长度
            seqlens_expanded = seqlens_expand_triton(
                torch.tensor(
                    extend_seq_lens_cpu, dtype=torch.int32, device=self.device
                ),
                cache_seqlens,
                self.speculative_num_draft_tokens * bs,
                self.speculative_num_draft_tokens,
            )
            metadata.nsa_seqlens_expanded.copy_(seqlens_expanded)
            nsa_cache_seqlens = compute_nsa_seqlens(
                seqlens_expanded, self.nsa_index_topk
            )
            metadata.nsa_cache_seqlens_int32.copy_(nsa_cache_seqlens)
        elif forward_mode.is_draft_extend(include_v2=True):
            # draft_extend：各请求接受的 token 数量可变（由 spec_info 提供）
            max_seqlen_k = int(seq_lens_cpu.max().item())
            cache_seqlens = seq_lens.to(torch.int32)
            metadata.cache_seqlens_int32.copy_(cache_seqlens)
            metadata.cu_seqlens_k[1:].copy_(
                torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32)
            )

            # 从 spec_info 中获取各请求实际接受的 token 数（可变长度）
            extend_seq_lens = spec_info.num_accepted_tokens[:bs]
            extend_seq_lens_cpu = extend_seq_lens.tolist()

            page_indices = self.req_to_token[req_pool_indices, :max_seqlen_k]
            # 按各请求的 extend_seq_lens 重复 page 索引行（可变重复次数）
            page_indices = torch.repeat_interleave(
                page_indices, repeats=extend_seq_lens, dim=0
            )
            metadata.page_table_1[: page_indices.shape[0], :max_seqlen_k].copy_(
                page_indices
            )

            # 展开 seqlens（总 token 数为 extend_seq_lens 之和）
            seqlens_expanded = seqlens_expand_triton(
                extend_seq_lens,
                cache_seqlens,
                sum(extend_seq_lens_cpu),
                self.speculative_num_draft_tokens,
            )
            metadata.nsa_seqlens_expanded[: seqlens_expanded.shape[0]].copy_(
                seqlens_expanded
            )
            nsa_cache_seqlens = compute_nsa_seqlens(
                seqlens_expanded, self.nsa_index_topk
            )
            metadata.nsa_cache_seqlens_int32[: seqlens_expanded.shape[0]].copy_(
                nsa_cache_seqlens
            )

        # Update DeepGEMM paged MQA schedule metadata outside the captured graph.
        # DeepGEMM 调度元数据在 CUDA Graph 外部更新（不纳入 graph 录制范围）
        if is_cuda() and (
            forward_mode.is_decode_or_idle()
            or forward_mode.is_target_verify()
            or forward_mode.is_draft_extend()
        ):
            try:
                import deep_gemm

                seqlens_32 = (
                    seqlens_expanded
                    if (
                        forward_mode.is_target_verify()
                        or forward_mode.is_draft_extend()
                    )
                    else metadata.cache_seqlens_int32
                )
                new_schedule = deep_gemm.get_paged_mqa_logits_metadata(
                    seqlens_32, 64, deep_gemm.get_num_sms()
                )
                if metadata.paged_mqa_schedule_metadata is None:
                    # 首次：直接赋值
                    metadata.paged_mqa_schedule_metadata = new_schedule
                else:
                    # 后续：复制内容以保持静态引用不变
                    metadata.paged_mqa_schedule_metadata.copy_(new_schedule)
            except (ImportError, ModuleNotFoundError):
                metadata.paged_mqa_schedule_metadata = None
        # 展开后 seqlens 的实际长度（可能小于静态缓冲区大小）
        seqlens_expanded_size = seqlens_expanded.shape[0]
        assert (
            metadata.nsa_cache_seqlens_int32 is not None
            and metadata.nsa_cu_seqlens_k is not None
            and self.nsa_index_topk is not None
        )

        # 更新 NSA KV 的累积序列长度（只更新有效部分）
        metadata.nsa_cu_seqlens_k[1 : 1 + seqlens_expanded_size].copy_(
            torch.cumsum(nsa_cache_seqlens, dim=0, dtype=torch.int32)
        )
        # NOTE(dark): (nsa-) cu_seqlens_q is always arange, no need to copy
        # cu_seqlens_q 固定为 arange，无需更新

        assert self.real_page_size == metadata.page_size
        if self.real_page_size > 1:
            # 将 page_size=1 的 page_table 转换为真实 page_size
            real_table = self._transform_table_1_to_real(page_indices)
            new_rows = real_table.shape[0]
            new_cols = real_table.shape[1]
            metadata.real_page_table[:new_rows, :new_cols].copy_(real_table)
        else:
            # page_size=1 时，real_page_table 与 page_table_1 是同一对象
            assert metadata.real_page_table is metadata.page_table_1

        if self.nsa_decode_impl == "flashmla_kv":
            # 切片并更新 FlashMLA 元数据（仅处理有效 token 范围）
            flashmla_metadata = metadata.flashmla_metadata.slice(
                slice(0, seqlens_expanded_size + 1)
            )
            flashmla_metadata.copy_(
                self._compute_flashmla_metadata(
                    cache_seqlens=nsa_cache_seqlens,
                    seq_len_q=1,
                )
            )

        # 更新 forward_metadata 指向当前批次的 metadata
        self.forward_metadata = metadata

    def init_forward_metadata_replay_cuda_graph_from_precomputed(
        self,
        bs: int,
        precomputed: PrecomputedMetadata,
        forward_mode: ForwardMode,
    ):
        """Fast path: copy precomputed metadata to this backend's metadata.

        快速路径：将预先计算好的 PrecomputedMetadata 复制到 CUDA Graph 的静态缓冲区。
        本函数只执行 copy 操作，不做任何计算，因此开销极低。
        若启用了 fused_metadata_copy_cuda JIT kernel，则通过单次 kernel 完成所有拷贝；
        否则 fallback 到逐字段的 tensor.copy_()。

        Args:
            bs: Batch size
            precomputed: Precomputed metadata to copy from
            forward_mode: Forward mode
        """
        self.set_nsa_prefill_impl(forward_batch=None)

        # 取出 capture 阶段存储的静态 metadata（以 bs 为 key）
        metadata = self.decode_cuda_graph_metadata[bs]

        # Track whether fused kernel succeeded
        # 标志位：是否成功使用了 fused kernel
        fused_kernel_succeeded = False

        # Use fused CUDA kernel for all copy operations
        # 如果启用了 fused 元数据拷贝 kernel，则尝试一次性完成所有拷贝
        if _USE_FUSED_METADATA_COPY:
            try:
                from sglang.jit_kernel.fused_metadata_copy import (
                    fused_metadata_copy_cuda,
                )

                # Map forward_mode to integer enum
                # 将 forward_mode 映射为整数枚举，传入 fused kernel
                if forward_mode.is_decode_or_idle():
                    mode_int = 0  # DECODE
                elif forward_mode.is_target_verify():
                    mode_int = 1  # TARGET_VERIFY
                elif forward_mode.is_draft_extend():
                    mode_int = 2  # DRAFT_EXTEND
                else:
                    raise ValueError(f"Unsupported forward_mode: {forward_mode}")

                # Prepare FlashMLA tensors if needed
                # 准备 FlashMLA 相关的 src/dst 张量（若不需要则保持 None）
                flashmla_num_splits_src = None
                flashmla_num_splits_dst = None
                flashmla_metadata_src = None
                flashmla_metadata_dst = None
                if precomputed.flashmla_metadata is not None:
                    flashmla_num_splits_src = precomputed.flashmla_metadata.num_splits
                    flashmla_num_splits_dst = metadata.flashmla_metadata.num_splits
                    flashmla_metadata_src = (
                        precomputed.flashmla_metadata.flashmla_metadata
                    )
                    flashmla_metadata_dst = metadata.flashmla_metadata.flashmla_metadata

                # Call fused kernel
                # 调用 fused CUDA kernel，一次性完成所有元数据张量的拷贝
                fused_metadata_copy_cuda(
                    # Source tensors
                    precomputed.cache_seqlens,
                    precomputed.cu_seqlens_k,
                    precomputed.page_indices,
                    precomputed.nsa_cache_seqlens,
                    precomputed.seqlens_expanded,
                    precomputed.nsa_cu_seqlens_k,
                    precomputed.real_page_table,
                    flashmla_num_splits_src,
                    flashmla_metadata_src,
                    # Destination tensors
                    metadata.cache_seqlens_int32,
                    metadata.cu_seqlens_k,
                    metadata.page_table_1,
                    metadata.nsa_cache_seqlens_int32,
                    metadata.nsa_seqlens_expanded,
                    metadata.nsa_cu_seqlens_k,
                    (
                        metadata.real_page_table
                        if precomputed.real_page_table is not None
                        else None
                    ),
                    flashmla_num_splits_dst,
                    flashmla_metadata_dst,
                    # Parameters
                    mode_int,
                    bs,
                    precomputed.max_len,
                    precomputed.max_seqlen_k,
                    precomputed.seqlens_expanded_size,
                )

                # Successfully used fused kernel
                fused_kernel_succeeded = True

            except ImportError:
                print(
                    "Warning: Fused metadata copy kernel not available, falling back to individual copies."
                )
            except Exception as e:
                print(
                    f"Warning: Fused metadata copy kernel failed with error: {e}, falling back to individual copies."
                )

        # Fallback to individual copy operations if fused kernel disabled or failed
        # fused kernel 不可用或失败时，逐字段执行 copy_()
        if not fused_kernel_succeeded:
            # Copy basic seqlens
            # 复制基础序列长度信息
            metadata.cache_seqlens_int32.copy_(precomputed.cache_seqlens)
            metadata.cu_seqlens_k[1:].copy_(precomputed.cu_seqlens_k[1:])

            # Mode-specific copy logic
            # 根据 forward_mode 选择对应字段的拷贝逻辑
            if forward_mode.is_decode_or_idle():
                # Decode mode
                # decode 模式：复制 page_indices 到 page_table_1
                metadata.page_table_1[:, : precomputed.max_len].copy_(
                    precomputed.page_indices
                )
                metadata.nsa_cache_seqlens_int32.copy_(precomputed.nsa_cache_seqlens)
                # seqlens_expanded is same as cache_seqlens (already copied)
                # decode 时 seqlens_expanded 与 cache_seqlens 相同，无需单独复制

            elif forward_mode.is_target_verify():
                # Target verify mode
                # target_verify 模式：page_table 按 max_seqlen_k 列数复制
                metadata.page_table_1[:, : precomputed.max_seqlen_k].copy_(
                    precomputed.page_indices
                )
                metadata.nsa_seqlens_expanded.copy_(precomputed.seqlens_expanded)
                metadata.nsa_cache_seqlens_int32.copy_(precomputed.nsa_cache_seqlens)

            elif forward_mode.is_draft_extend():
                # Draft extend mode
                # draft_extend 模式：行数可变（由 extend_seq_lens 决定）
                rows = precomputed.page_indices.shape[0]
                cols = precomputed.max_seqlen_k
                metadata.page_table_1[:rows, :cols].copy_(precomputed.page_indices)

                # 只复制有效范围内的 seqlens_expanded 和 nsa_cache_seqlens
                size = precomputed.seqlens_expanded_size
                metadata.nsa_seqlens_expanded[:size].copy_(precomputed.seqlens_expanded)
                metadata.nsa_cache_seqlens_int32[:size].copy_(
                    precomputed.nsa_cache_seqlens
                )

            # Copy NSA cu_seqlens
            # 复制 NSA KV 累积序列长度（只更新有效部分）
            size = precomputed.seqlens_expanded_size
            metadata.nsa_cu_seqlens_k[1 : 1 + size].copy_(
                precomputed.nsa_cu_seqlens_k[1 : 1 + size]
            )

            # Copy real page table
            # 复制真实 page_size 的 page_table（若存在）
            if precomputed.real_page_table is not None:
                rows, cols = precomputed.real_page_table.shape
                metadata.real_page_table[:rows, :cols].copy_(
                    precomputed.real_page_table
                )

            # Copy FlashMLA metadata in fallback path
            # 复制 FlashMLA 元数据（fallback 路径）
            if precomputed.flashmla_metadata is not None:
                size = precomputed.seqlens_expanded_size
                flashmla_metadata = metadata.flashmla_metadata.slice(slice(0, size + 1))
                flashmla_metadata.copy_(precomputed.flashmla_metadata)

        # 更新 forward_metadata 为当前批次的 metadata
        self.forward_metadata = metadata

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """NSA prefill/extend 阶段的前向计算。

        根据当前 forward_mode（target_verify / draft_extend / 普通 extend）
        选择对应的 nsa_impl（trtllm / tilelang / flashmla_sparse / flashmla_kv / fa3 / aiter），
        完成 KV 缓存写入和稀疏注意力计算。

        Args:
            q: Query 张量
            k/v: Key/Value 张量（MLA 路径下为 compressed KV）
            layer: 当前 RadixAttention 层（含 head_dim、scaling 等超参）
            forward_batch: 当前批次信息
            save_kv_cache: 是否将 KV 写入缓存
            q_rope/k_rope: RoPE 部分的 Q/K 张量（MLA 专用）
            topk_indices: NSA top-k 稀疏索引（[bs, topk]）
            cos_sin_cache/is_neox/llama_4_scaling: RoPE 相关参数（trtllm 路径使用）
        """

        causal = not layer.is_cross_attention
        metadata = self.forward_metadata
        assert causal, "NSA is causal only"

        # 根据 forward_mode 选择 decode（verify/draft）或 prefill 的 impl
        nsa_impl = (
            self.nsa_decode_impl
            if (
                forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend(include_v2=True)
            )
            else self.nsa_prefill_impl
        )

        # trtllm 路径（非 MHA）：直接分发到 _forward_trtllm
        if nsa_impl == "trtllm" and not self.use_mha:
            return self._forward_trtllm(
                q,
                k,
                v,
                layer,
                forward_batch,
                metadata.nsa_cache_seqlens_int32,
                save_kv_cache,
                q_rope,
                k_rope,
                topk_indices,
                cos_sin_cache,
                is_neox,
                llama_4_scaling,
                is_prefill=True,
            )

        # 写入 KV 缓存（MLA 路径：k 是 compressed KV，k_rope 是 RoPE 部分）
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                # 将 compressed KV 写入 token_to_kv_pool（MLA 专用接口）
                forward_batch.token_to_kv_pool.set_mla_kv_buffer(  # type: ignore
                    layer,
                    cache_loc,
                    k,
                    k_rope,
                )

        # Use MHA kernel if in MHA_ONE_SHOT mode
        # MHA_ONE_SHOT 模式：直接使用标准 MHA kernel（跳过稀疏逻辑）
        if self.use_mha:
            assert k is not None and v is not None
            assert q_rope is None, "MHA_ONE_SHOT path should not pass q_rope"
            assert (
                layer.tp_k_head_num == layer.tp_q_head_num > 1
            ), "MHA_ONE_SHOT requires dense multi-head config"
            return self._forward_standard_mha(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
                metadata=metadata,
            )

        # Do absorbed multi-latent attention (MLA path)
        # MLA 吸收路径：将 Q 分拆为 q_nope（content）和 q_rope（位置编码）部分
        assert q_rope is not None
        # 获取当前层的 KV 缓存（MLA compressed KV）
        kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)

        if q_rope is not None:
            # q 已拆分：q 为 nope 部分，q_rope 为 rope 部分
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
        else:
            # q 未拆分：从完整 q 中按维度切分 nope 和 rope 部分
            q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
            q_nope = q_all[:, :, : layer.v_head_dim]
            q_rope = q_all[:, :, layer.v_head_dim :]

        # Align topk_indices with q dimensions
        # This handles cases where q is padded (TP + partial DP attention)
        # 对齐 topk_indices 与实际 Q token 数（TP/DP padding 场景下 Q 可能被 pad）
        if topk_indices is not None:
            topk_indices = self._pad_topk_indices(topk_indices, q_nope.shape[0])

        # NOTE(dark): here, we use page size = 1
        # 获取 topk 变换方法（PAGED 或 RAGGED）
        topk_transform_method = self.get_topk_transform_method(
            forward_batch.forward_mode
        )
        if envs.SGLANG_NSA_FUSE_TOPK.get():
            # 融合 topk kernel 模式：page_table_1 直接使用原始 topk_indices
            page_table_1 = topk_indices
        else:
            if topk_transform_method == TopkTransformMethod.RAGGED:
                # RAGGED 格式：将 topk_indices 加上 topk_indices_offset 转为绝对索引
                topk_indices_offset = metadata.topk_indices_offset
                assert topk_indices_offset is not None
                mask = topk_indices != -1
                topk_indices_offset = (
                    topk_indices_offset.unsqueeze(1)
                    if topk_indices_offset.ndim == 1
                    else topk_indices_offset
                )
                # -1 位置（padding）保持 -1，其余位置加上全局偏移
                topk_indices = torch.where(
                    mask, topk_indices + topk_indices_offset, topk_indices
                )
            elif topk_transform_method == TopkTransformMethod.PAGED:
                # PAGED 格式：将 topk_indices 映射到 page_table_1（page_size=1）
                assert metadata.nsa_extend_seq_lens_list is not None
                page_table_1 = transform_index_page_table_prefill(
                    page_table=metadata.page_table_1,
                    topk_indices=topk_indices,
                    extend_lens_cpu=metadata.nsa_extend_seq_lens_list,
                    page_size=1,
                )

        # todo hisparse: to cover more backends
        # HiSparse 设备地址转换（实验性）
        if forward_batch.hisparse_coordinator is not None:
            page_table_1 = (
                forward_batch.token_to_kv_pool.translate_loc_to_hisparse_device(
                    page_table_1
                )
            )

        # 根据 nsa_impl 分发到对应的底层 kernel
        if nsa_impl == "tilelang":
            # TileLang kernel：拼接 q_nope 和 q_rope 后调用
            if q_rope is not None:
                q_all = concat_mla_absorb_q_general(q_nope, q_rope)
            return self._forward_tilelang(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
            )
        elif nsa_impl == "flashmla_sparse":
            # FlashMLA sparse kernel（稀疏 prefill 专用）
            if q_rope is not None:
                q_all = concat_mla_absorb_q_general(q_nope, q_rope)

            if topk_transform_method == TopkTransformMethod.RAGGED:
                # RAGGED 格式：需要将 KV cache 反量化或拼接
                if any(forward_batch.extend_prefix_lens_cpu):
                    # 有前缀时：从已有 KV cache 反量化
                    page_table_1_flattened = (
                        self.forward_metadata.page_table_1_flattened
                    )
                    assert page_table_1_flattened is not None
                    kv_cache = dequantize_k_cache_paged(
                        kv_cache, page_table_1_flattened
                    )
                else:
                    # 无前缀时：直接拼接当前 KV token
                    kv_cache = _cat([k, k_rope], dim=-1)
                page_table_1 = topk_indices

            return self._forward_flashmla_sparse(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
            )
        elif nsa_impl == "flashmla_kv":
            # FlashMLA KV kernel（paged KV，通用解码/prefill）
            if q_rope is not None:
                q_all = concat_mla_absorb_q_general(q_nope, q_rope)
            return self._forward_flashmla_kv(
                q_all=q_all,
                kv_cache=kv_cache,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
                # TODO optimize args
                layer=layer,
                metadata=metadata,
                page_table_1=page_table_1,
            )
        elif nsa_impl == "fa3":
            # FlashAttention3 paged kernel（ragged Q/K）
            return self._forward_fa3(
                q_rope=q_rope,
                kv_cache=kv_cache,
                v_head_dim=layer.v_head_dim,
                q_nope=q_nope,
                page_table=page_table_1,
                cache_seqlens=metadata.nsa_cache_seqlens_int32,
                cu_seqlens_q=metadata.nsa_cu_seqlens_q,
                cu_seqlens_k=metadata.nsa_cu_seqlens_k,
                max_seqlen_q=metadata.nsa_max_seqlen_q,
                sm_scale=layer.scaling,
                logit_cap=layer.logit_cap,
                page_size=1,
            )
        elif nsa_impl == "aiter":
            # AITER（AMD ROCm）extend kernel
            if q_rope is not None:
                q_all = torch.cat([q_nope, q_rope], dim=-1)
            return self._forward_aiter_extend(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                layer=layer,
            )
        else:
            raise ValueError(
                f"Unsupported {nsa_impl = } for forward_extend. Consider using an other attention backend."
            )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """NSA decode 阶段的前向计算。

        decode 阶段每个请求只有 1 个新 Q token，需要对已缓存的 KV 进行稀疏注意力。
        根据 nsa_decode_impl 选择对应的底层 kernel（trtllm / flashmla_sparse /
        flashmla_kv / tilelang / fa3 / aiter），并负责将新 KV 写入 KV 缓存。

        Args:
            q: Query 张量（decode：[bs, 1, head_dim]）
            k/v: 新生成的 Key/Value（MLA 路径为 compressed KV）
            layer: 当前 RadixAttention 层
            forward_batch: 当前批次信息
            save_kv_cache: 是否将新 KV 写入缓存
            q_rope/k_rope: RoPE 部分的 Q/K（MLA 专用）
            topk_indices: NSA top-k 稀疏索引（[bs, topk]）
            cos_sin_cache/is_neox/llama_4_scaling: RoPE 参数（trtllm 路径）
        """

        causal = not layer.is_cross_attention
        metadata = self.forward_metadata
        assert causal, "NSA is causal only"

        # trtllm 路径：直接分发（含 KV 写入和注意力计算）
        if self.nsa_decode_impl == "trtllm":
            return self._forward_trtllm(
                q,
                k,
                v,
                layer,
                forward_batch,
                metadata.cache_seqlens_int32,
                save_kv_cache,
                q_rope,
                k_rope,
                topk_indices,
                cos_sin_cache,
                is_neox,
                llama_4_scaling,
            )

        # 写入 KV 缓存（MLA 路径：k 为 compressed KV，k_rope 为 RoPE 部分）
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                forward_batch.token_to_kv_pool.set_mla_kv_buffer(  # type: ignore
                    layer,
                    cache_loc,
                    k,
                    k_rope,
                )

        # Do absorbed multi-latent attention
        # 获取已缓存的 compressed KV
        kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        # 将 Q 拆分为 q_nope（content）和 q_rope（position）
        if q_rope is not None:
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
        else:
            q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
            q_nope = q_all[:, :, : layer.v_head_dim]
            q_rope = q_all[:, :, layer.v_head_dim :]

        # Align topk_indices with q dimensions
        # 对齐 topk_indices 与实际 Q token 数（TP/DP padding 下 Q 可能被 pad）
        if topk_indices is not None:
            topk_indices = self._pad_topk_indices(topk_indices, q_nope.shape[0])

        # 将 topk_indices 转换为 page_table_1 格式（page_size=1）
        if forward_batch.hisparse_coordinator is not None:
            # HiSparse 模式：先换入选定的 KV 页，再得到设备地址
            page_table_1 = forward_batch.hisparse_coordinator.swap_in_selected_pages(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                topk_indices,
                layer.layer_id,
            )
        elif envs.SGLANG_NSA_FUSE_TOPK.get():
            # 融合 topk kernel：直接使用原始 topk_indices
            page_table_1 = topk_indices
        else:
            # 标准路径：通过 decode 变换将 topk_indices 映射到 page_table_1
            page_table_1 = transform_index_page_table_decode(
                page_table=metadata.page_table_1,
                topk_indices=topk_indices,
                page_size=1,
            )

        # 根据 nsa_decode_impl 分发到对应的底层 kernel
        if self.nsa_decode_impl == "flashmla_sparse":
            # FlashMLA sparse decode kernel
            if q_rope is not None:
                q_all = concat_mla_absorb_q_general(q_nope, q_rope)
            return self._forward_flashmla_sparse(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
            )
        elif self.nsa_decode_impl == "flashmla_kv":
            # FlashMLA KV paged decode kernel
            if q_rope is not None:
                q_all = concat_mla_absorb_q_general(q_nope, q_rope)
            return self._forward_flashmla_kv(
                q_all=q_all,
                kv_cache=kv_cache,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
                # TODO optimize args
                layer=layer,
                metadata=metadata,
                page_table_1=page_table_1,
            )
        elif self.nsa_decode_impl == "tilelang":
            # TileLang decode kernel
            if q_rope is not None:
                q_all = concat_mla_absorb_q_general(q_nope, q_rope)
            return self._forward_tilelang(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
            )
        elif self.nsa_decode_impl == "fa3":
            # FlashAttention3 paged decode kernel
            return self._forward_fa3(
                q_rope=q_rope,
                kv_cache=kv_cache,
                v_head_dim=layer.v_head_dim,
                q_nope=q_nope,
                page_table=page_table_1,
                cache_seqlens=metadata.nsa_cache_seqlens_int32,
                cu_seqlens_q=metadata.nsa_cu_seqlens_q,
                cu_seqlens_k=metadata.nsa_cu_seqlens_k,
                max_seqlen_q=metadata.nsa_max_seqlen_q,
                sm_scale=layer.scaling,
                logit_cap=layer.logit_cap,
                page_size=1,
            )
        elif self.nsa_decode_impl == "aiter":
            # AITER（AMD ROCm）decode kernel
            if q_rope is not None:
                q_all = torch.cat([q_nope, q_rope], dim=-1)
            return self._forward_aiter(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                layer=layer,
                metadata=metadata,
                bs=forward_batch.batch_size,
            )

        else:
            assert False, f"Unsupported {self.nsa_decode_impl = }"

    def _forward_fa3(
        self,
        q_rope: torch.Tensor,
        kv_cache: torch.Tensor,
        v_head_dim: int,
        q_nope: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        sm_scale: float,
        logit_cap: float,
        page_size: int,
    ) -> torch.Tensor:
        """调用 FlashAttention3 paged MLA kernel。

        将 kv_cache 拆分为 k_rope_cache（RoPE 部分）和 c_kv_cache（compressed KV）,
        并以 page_size 为块大小调用 flash_attn_with_kvcache。

        q_rope 作为标准 K 侧输入，q_nope 作为 V 侧等价输入（MLA 吸收变换）。
        """
        # 从 kv_cache 中分离 rope 部分（后 v_head_dim 列之后）和 compressed KV（前 v_head_dim 列）
        k_rope_cache = kv_cache[:, :, v_head_dim:]
        c_kv_cache = kv_cache[:, :, :v_head_dim]
        qk_rope_dim = k_rope_cache.shape[-1]
        # 重塑为 [num_pages, page_size, 1, dim] 形式，匹配 paged attention 接口
        k_rope_cache = k_rope_cache.view(-1, page_size, 1, qk_rope_dim)
        c_kv_cache = c_kv_cache.view(-1, page_size, 1, v_head_dim)
        # 调用 FlashAttention3 paged MLA kernel
        o = flash_attn_with_kvcache(
            q=q_rope,
            k_cache=k_rope_cache,
            v_cache=c_kv_cache,
            qv=q_nope,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=sm_scale,
            causal=True,
            softcap=logit_cap,
            return_softmax_lse=False,
            num_splits=self.num_splits,
        )
        return o  # type: ignore

    def _forward_flashmla_sparse(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        v_head_dim: int,
        page_table_1: torch.Tensor,
        sm_scale: float,
    ) -> torch.Tensor:
        """调用 FlashMLA sparse kernel（稀疏 prefill / decode）。

        FlashMLA sparse kernel 要求 num_heads 必须是 64（Hopper）或 128（Blackwell）的整数倍。
        使用 TP 时 num_heads 可能较小（如 256//8=32），此时需要 pad。
        """
        from sgl_kernel.flash_mla import flash_mla_sparse_fwd

        # FlashMLA sparse kernel requires num_heads to be a multiple of 64 (Hopper) or 128 (Blackwell)
        # When using TP, num_heads might be smaller (e.g., 256//8=32)
        num_tokens, num_heads, head_dim = q_all.shape

        # Determine required padding based on GPU architecture (use cached value)
        # 根据 GPU 架构确定所需的 head 对齐数（Blackwell=128, Hopper=64）
        required_padding = 128 if self.device_sm_major >= 10 else 64

        need_padding = num_heads % required_padding != 0

        if need_padding:
            assert required_padding % num_heads == 0, (
                f"num_heads {num_heads} cannot be padded to {required_padding}. "
                f"TP size may be too large for this model."
            )

            # Pad q to required size
            # 将 Q 补零 pad 到 required_padding 个 head
            q_padded = q_all.new_zeros((num_tokens, required_padding, head_dim))
            q_padded[:, :num_heads, :] = q_all
            q_input = q_padded
        else:
            q_input = q_all

        # indices shape must be (s_q, h_kv=1, topk), keep h_kv=1 unchanged
        # 添加 h_kv=1 维度（FlashMLA sparse 接口要求）
        indices_input = page_table_1.unsqueeze(1)

        o, _, _ = flash_mla_sparse_fwd(
            q=q_input,
            kv=kv_cache,
            indices=indices_input,
            sm_scale=sm_scale,
            d_v=v_head_dim,
        )

        # Trim output back to original num_heads if we padded
        # 去掉 pad 的 head，恢复原始 num_heads
        if need_padding:
            o = o[:, :num_heads, :]

        return o

    def _forward_flashmla_kv(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        v_head_dim: int,
        sm_scale: float,
        layer,
        metadata: NSAMetadata,
        page_table_1,
    ) -> torch.Tensor:
        """调用 FlashMLA KV paged kernel（支持稀疏 topk indices）。

        FlashMLA KV kernel 要求 Q head 数为 64 或 128 的整数倍；不满足时需 pad。
        KV cache 需要为 FP8 格式（若原始不是 FP8 则在线量化）。
        """
        from sgl_kernel.flash_mla import flash_mla_with_kvcache

        # 从 metadata 中取 NSA 剪裁后的 KV 序列长度
        cache_seqlens = metadata.nsa_cache_seqlens_int32
        assert metadata.flashmla_metadata is not None

        # TODO the 2nd dim is seq_len_q, need to be >1 when MTP
        # 重塑 Q 为 [bs, seq_len_q=1, num_heads, head_dim]
        q_all = q_all.view(-1, 1, layer.tp_q_head_num, layer.head_dim)
        num_q_heads = q_all.shape[2]
        target_q_heads = self.flashmla_kv_num_q_heads
        if target_q_heads != num_q_heads:
            # Pad q heads to match FlashMLA decode supported head-count variants.
            # 将 Q head 数 pad 到 FlashMLA 支持的数量
            q_input = q_all.new_zeros(
                q_all.shape[0], q_all.shape[1], target_q_heads, q_all.shape[3]
            )
            q_input[:, :, :num_q_heads, :] = q_all
        else:
            q_input = q_all

        # 将 KV cache 重塑为 [num_pages, page_size, 1, kv_cache_dim]
        kv_cache = kv_cache.view(-1, self.real_page_size, 1, self.kv_cache_dim)
        assert self.real_page_size == 64, "only page size 64 is supported"

        if not self.nsa_kv_cache_store_fp8:
            # inefficiently quantize the whole cache
            # 若 KV cache 未以 FP8 存储，在线量化（性能次优）
            kv_cache = quantize_k_cache(kv_cache)

        # 添加 h_kv=1 维度：indices shape 需为 (bs, 1, topk)
        indices = page_table_1.unsqueeze(1)
        assert (
            indices.shape[-1] == self.nsa_index_topk
        )  # requirement of FlashMLA decode kernel

        # 调用 FlashMLA KV paged kernel
        o, _ = flash_mla_with_kvcache(
            q=q_input,
            k_cache=kv_cache,
            cache_seqlens=cache_seqlens,
            head_dim_v=v_head_dim,
            tile_scheduler_metadata=metadata.flashmla_metadata.flashmla_metadata,
            num_splits=metadata.flashmla_metadata.num_splits,
            softmax_scale=sm_scale,
            indices=indices,
            # doc says it is not used, but if pass in None then error
            # 文档说不使用，但传 None 会报错，因此传一个空 block_table
            block_table=torch.empty(
                (q_all.shape[0], 0), dtype=torch.int32, device=q_all.device
            ),
            is_fp8_kvcache=True,
        )

        if target_q_heads != num_q_heads:
            # 去掉 pad 的 head，恢复原始 num_q_heads
            o = o[:, :, :num_q_heads, :]

        return o

    def _forward_standard_mha(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        metadata: NSAMetadata,
    ) -> torch.Tensor:
        """Standard MHA using FlashAttention varlen for MHA_ONE_SHOT mode.

        MHA_ONE_SHOT 模式下的标准多头注意力，使用 FlashAttention varlen 接口。
        SM100（Blackwell）使用 TRT-LLM ragged attention；SM90（Hopper）使用 FA3 varlen。
        """
        q = q.view(-1, layer.tp_q_head_num, layer.head_dim)
        k = k.view(-1, layer.tp_k_head_num, layer.head_dim)
        v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)

        # MHA_ONE_SHOT: k/v include all tokens (prefix + current)
        # MHA_ONE_SHOT：k/v 包含完整的 prefix + 当前 token
        cu_seqlens_q = metadata.cu_seqlens_q
        cu_seqlens_k = metadata.cu_seqlens_k
        max_seqlen_k = metadata.max_seq_len_k
        causal = True

        # Verify batch sizes match (length of cu_seqlens should be batch_size + 1)
        # 验证 Q/K 两侧的 cu_seqlens 长度匹配
        assert len(cu_seqlens_q) == len(cu_seqlens_k), (
            f"batch_size mismatch: cu_seqlens_q has {len(cu_seqlens_q)-1} requests, "
            f"cu_seqlens_k has {len(cu_seqlens_k)-1} requests"
        )

        # Use TRTLLm ragged attention for SM100 (Blackwell/B200) to avoid FA4 accuracy issues
        # SM100 使用 TRT-LLM ragged attention（规避 FA4 精度问题）
        if self.device_sm_major >= 10:
            import flashinfer

            seq_lens = metadata.cache_seqlens_int32
            return flashinfer.prefill.trtllm_ragged_attention_deepseek(
                query=q,
                key=k,
                value=v,
                workspace_buffer=self.workspace_buffer,
                seq_lens=seq_lens,
                max_q_len=metadata.max_seq_len_q,
                max_kv_len=max_seqlen_k,
                bmm1_scale=layer.scaling,
                bmm2_scale=1.0,
                o_sf_scale=1.0,
                batch_size=forward_batch.batch_size,
                window_left=-1,
                cum_seq_lens_q=cu_seqlens_q,
                cum_seq_lens_kv=cu_seqlens_k,
                enable_pdl=False,
                is_causal=causal,
                return_lse=False,
                skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR.get(),
            )

        # Use FA3 for SM90 (Hopper/H200)
        # SM90 使用 FlashAttention3 varlen kernel
        return flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=metadata.max_seq_len_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=layer.scaling,
            causal=causal,
        )

    def _forward_tilelang(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        v_head_dim: int,
        page_table_1: torch.Tensor,
        sm_scale: float,
    ) -> torch.Tensor:
        """调用 TileLang sparse MLA kernel。

        TileLang 是 NVIDIA GPU 上的高性能稀疏注意力 kernel，
        indices 维度需为 (s_q, 1, topk)。
        """
        from sglang.srt.layers.attention.nsa.tilelang_kernel import tilelang_sparse_fwd

        return tilelang_sparse_fwd(
            q=q_all,
            kv=kv_cache,
            indices=page_table_1.unsqueeze(1),   # 添加 h_kv=1 维度
            sm_scale=sm_scale,
            d_v=v_head_dim,
        )

    def _forward_aiter(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        page_table_1: torch.Tensor,
        layer: RadixAttention,
        metadata: NSAMetadata,
        bs: int,
    ) -> torch.Tensor:
        """调用 AITER MLA decode kernel（AMD ROCm，decode 阶段）。

        将 page_table_1 中的非 -1 索引提取为稀疏 kv_indices，
        再调用 mla_decode_fwd 完成稀疏注意力计算。
        若 num_heads 不满足对齐要求，使用 repeat_interleave pad 后截取结果。
        """
        # 将 q 重塑为 [num_tokens, num_heads * head_dim]
        q = q_all.reshape(-1, layer.tp_q_head_num * layer.head_dim)

        if layer.head_dim != layer.v_head_dim:
            # v_head_dim 不同时单独分配输出缓冲区
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if self.need_pad_heads:
            # 需要 pad heads：将 Q 重复 head_repeat_factor 倍后传入 kernel
            q_kernel = q.view(
                -1, layer.tp_q_head_num, layer.head_dim
            ).repeat_interleave(self.head_repeat_factor, dim=1)
            o_kernel = q.new_empty(
                (
                    q.shape[0],
                    layer.tp_q_head_num * self.head_repeat_factor,
                    layer.v_head_dim,
                )
            )
        else:
            q_kernel = q.view(-1, layer.tp_q_head_num, layer.head_dim)
            o_kernel = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        # 使用预分配的 kv_indptr 静态缓冲区
        kv_indptr = self.kv_indptr

        # 统计每行非 -1 的有效 topk 索引数量，构造 kv_indptr（前缀和）
        non_minus1_mask = page_table_1 != -1
        non_minus1_counts = non_minus1_mask.sum(dim=1)
        kv_indptr[1 : bs + 1] = torch.cumsum(non_minus1_counts, dim=0)

        # 从 page_table_1 中提取有效的 KV 索引（去掉 -1 填充位）
        kv_indices = self.kv_indices
        get_valid_kv_indices(page_table_1, kv_indptr, kv_indices, bs)

        # 调用 AITER MLA decode forward kernel
        mla_decode_fwd(
            q_kernel,
            kv_cache.view(-1, 1, 1, layer.head_dim),
            o_kernel,
            metadata.cu_seqlens_q,
            kv_indptr,
            kv_indices,
            metadata.cu_seqlens_q,
            metadata.max_seq_len_q,
            sm_scale=layer.scaling,
            logit_cap=layer.logit_cap,
        )

        if self.need_pad_heads:
            # 每 head_repeat_factor 个 head 取第一个，恢复原始 head 数
            o = o_kernel[:, :: self.head_repeat_factor, :]

        return o

    def _forward_aiter_extend(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        page_table_1: torch.Tensor,
        layer: RadixAttention,
    ) -> torch.Tensor:
        """调用 AITER MLA extend kernel（AMD ROCm，prefill/extend 阶段）。

        与 _forward_aiter 类似，但 kv_indptr 和 kv_indices 按 num_tokens 动态分配，
        且 cu_seqlens_q 为 arange（每个 token 的 seq_len_q=1）。
        """
        num_tokens = q_all.shape[0]
        q = q_all.reshape(-1, layer.tp_q_head_num * layer.head_dim)

        if layer.head_dim != layer.v_head_dim:
            o = q.new_empty((num_tokens, layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if self.need_pad_heads:
            q_kernel = q.view(
                -1, layer.tp_q_head_num, layer.head_dim
            ).repeat_interleave(self.head_repeat_factor, dim=1)
            o_kernel = q.new_empty(
                (
                    num_tokens,
                    layer.tp_q_head_num * self.head_repeat_factor,
                    layer.v_head_dim,
                )
            )
        else:
            q_kernel = q.view(-1, layer.tp_q_head_num, layer.head_dim)
            o_kernel = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        # 统计每行的有效 topk 索引数量，动态构造 kv_indptr
        non_minus1_mask = page_table_1 != -1
        non_minus1_counts = non_minus1_mask.sum(dim=1)

        kv_indptr = torch.zeros(num_tokens + 1, dtype=torch.int32, device=self.device)
        kv_indptr[1:] = torch.cumsum(non_minus1_counts, dim=0)

        # Allocate kv_indices with upper-bound size (num_tokens * topk)
        # 预分配 kv_indices（上界 = num_tokens * topk）
        topk = page_table_1.shape[1]
        kv_indices = torch.zeros(
            num_tokens * topk, dtype=torch.int32, device=self.device
        )

        # Use get_valid_kv_indices kernel to extract valid indices
        # 提取有效的 KV 索引
        get_valid_kv_indices(page_table_1, kv_indptr, kv_indices, num_tokens)

        # Build cu_seqlens_q for extend: each token is treated as seq_len_q=1
        # extend 模式：每个 token 独立处理，seq_len_q=1
        cu_seqlens_q = torch.arange(
            0, num_tokens + 1, dtype=torch.int32, device=self.device
        )
        # TODO support more forward_mode
        # 调用 AITER MLA decode forward（extend 路径：per-token seq_len_q=1）
        mla_decode_fwd(
            q_kernel,
            kv_cache.view(-1, 1, 1, layer.head_dim),
            o_kernel,
            cu_seqlens_q,
            kv_indptr,
            kv_indices,
            cu_seqlens_q,
            1,  # max_seq_len_q = 1 for per-token attention
            sm_scale=layer.scaling,
            logit_cap=layer.logit_cap,
        )

        if self.need_pad_heads:
            o = o_kernel[:, :: self.head_repeat_factor, :]

        return o

    def _forward_trtllm(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        seq_lens: torch.Tensor,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
        is_prefill: bool = False,
    ) -> torch.Tensor:
        """Forward using TRT-LLM sparse MLA kernel.

        TRT-LLM 稀疏 MLA kernel，同时支持 decode 和 prefill 阶段。
        FP8 路径下需要在线对 Q/K 进行量化和 RoPE 应用；BF16 路径直接合并 Q。
        topk_indices 会被转换为 page_table_1 后传入 kernel。
        """
        import flashinfer.decode

        metadata = self.forward_metadata

        # 若 q_rope 不为 None，说明后续需要合并 q_nope 和 q_rope
        merge_query = q_rope is not None
        if self.kv_cache_dtype == torch.float8_e4m3fn:
            # For FP8 path, we quantize the query and rope parts and merge them into a single tensor
            # Note: rope application in deepseek_v2.py:forward_absorb_prepare is skipped for FP8 decode path of this trtllm_mla backend
            # FP8 路径：对 Q/K 进行量化并应用 RoPE（FP8 decode 路径跳过了外部 RoPE）
            assert q_rope is not None, "For FP8 path q_rope should not be None."
            assert k_rope is not None, "For FP8 path k_rope should not be None."
            assert (
                cos_sin_cache is not None
            ), "For FP8 path cos_sin_cache should not be None."

            # 量化 Q/K 并应用 RoPE，返回合并后的 q 和 k/k_rope
            q, k, k_rope = mla_quantize_and_rope_for_fp8(
                q,
                q_rope,
                k.squeeze(1),
                k_rope.squeeze(1),
                forward_batch.positions,
                cos_sin_cache,
                is_neox,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
            )
            merge_query = False   # FP8 路径已完成合并，无需再 concat

            # Save KV cache if requested
        # 将新 KV 写入 KV 缓存
        if save_kv_cache:
            assert (
                k is not None and k_rope is not None
            ), "For populating trtllm_mla kv cache, both k_nope and k_rope should be not None."
            cache_loc = (
                forward_batch.out_cache_loc
                if not layer.is_cross_attention
                else forward_batch.encoder_out_cache_loc
            )
            forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                layer, cache_loc, k, k_rope
            )

        # 获取已缓存的全量 KV，重塑为 paged 格式
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        kv_cache = k_cache.view(-1, self.real_page_size, self.kv_cache_dim).unsqueeze(1)

        if merge_query:
            # BF16 路径：将 q_nope 和 q_rope 通过吸收变换合并为 q_all
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope_reshaped = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
            q_all = concat_mla_absorb_q_general(q_nope, q_rope_reshaped)
        else:
            # FP8 路径：q 已经是合并后的形式
            q_all = q.view(-1, layer.tp_q_head_num, layer.head_dim)

        # Align topk_indices with q dimensions
        # 对齐 topk_indices 行数（TP/DP padding 下 Q 行数可能比 topk_indices 多）
        if topk_indices is not None:
            topk_indices = self._pad_topk_indices(topk_indices, q.shape[0])

        # 将 topk_indices 转换为 page_table_1（paged 格式）
        if envs.SGLANG_NSA_FUSE_TOPK.get():
            # 融合 topk kernel 模式：直接使用 topk_indices
            page_table_1 = topk_indices
        elif is_prefill:
            # prefill 路径：通过 prefill 变换映射到 page_table_1
            page_table_1 = transform_index_page_table_prefill(
                page_table=metadata.page_table_1,
                topk_indices=topk_indices,
                extend_lens_cpu=metadata.nsa_extend_seq_lens_list,
                page_size=1,
            )
        else:
            # decode 路径：通过 decode 变换映射到 page_table_1
            page_table_1 = transform_index_page_table_decode(
                page_table=metadata.page_table_1,
                topk_indices=topk_indices,
                page_size=1,
            )

        # bmm1_scale = Q scale * K scale * attention scaling
        # 计算 BMM1 缩放因子（Q scale * K scale * softmax scale）
        q_scale = 1.0
        k_scale = (
            layer.k_scale_float
            if getattr(layer, "k_scale_float", None) is not None
            else 1.0
        )
        bmm1_scale = q_scale * k_scale * layer.scaling

        batch_size = page_table_1.shape[0]
        _, num_heads, head_dim = q_all.shape

        # 重塑为 TRT-LLM MLA kernel 所需的 4D 格式
        q = q_all.view(batch_size, 1, num_heads, head_dim)
        kv = kv_cache.view(-1, 1, self.real_page_size, self.kv_cache_dim)
        # 添加 h_kv=1 维度：block_tables shape = (bs, 1, topk)
        block_tables = page_table_1.unsqueeze(1)
        seq_lens = metadata.cache_seqlens_int32 if seq_lens is None else seq_lens

        # 调用 TRT-LLM paged MLA batch decode kernel
        out = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=q,
            kv_cache=kv,
            workspace_buffer=self.workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=metadata.max_seq_len_k,
            sparse_mla_top_k=self.nsa_index_topk,
            bmm1_scale=bmm1_scale,
            backend="trtllm-gen",
            skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
        )
        # Output: [batch, q_len=1, heads, v_dim] -> [batch, heads, v_dim]
        # 去掉 q_len=1 维度后返回
        return out.squeeze(1)

    def _pad_topk_indices(
        self, topk_indices: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        """将 topk_indices 的行数 pad 到 num_tokens（用 -1 填充不足部分）。

        TP + partial DP 场景下，Q 张量行数可能因为 padding 比 topk_indices 多，
        此时需要在 topk_indices 末尾追加 -1 行（表示无效 topk）。

        Args:
            topk_indices: 原始 topk 索引 [current_tokens, topk]
            num_tokens: 目标行数（通常等于 Q 张量的 token 数）

        Returns:
            pad 后的 topk_indices，形状 [num_tokens, topk]
        """
        current_tokens = topk_indices.shape[0]
        if current_tokens == num_tokens:
            # 行数已对齐，无需 pad
            return topk_indices

        assert current_tokens <= num_tokens, (
            f"topk_indices rows ({current_tokens}) > num_tokens ({num_tokens}); "
            "this indicates a mismatch between indexer output and q layout."
        )

        # 构造 -1 填充行并拼接
        pad_size = num_tokens - current_tokens
        padding = torch.full(
            (pad_size, topk_indices.shape[1]),
            -1,
            dtype=topk_indices.dtype,
            device=topk_indices.device,
        )
        return torch.cat([topk_indices, padding], dim=0)

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for sequence length in CUDA graph.

        CUDA Graph 静态缓冲区中 seq_len 的初始填充值。
        对于 NSA 后端，decode 阶段每个 token 序列长度最小为 1，因此填 1。
        """
        return 1

    def set_nsa_prefill_impl(self, forward_batch: Optional[ForwardBatch] = None):
        """
        Decide all attention prefill dispatch strategies for this batch.

        根据当前批次的 forward_mode、序列长度、dtype 和硬件架构，
        决定本次 prefill 使用 MHA_ONE_SHOT 还是 MLA 路径，
        以及选择具体的 MLA prefill 实现（flashmla_sparse 或 flashmla_kv）。
        """
        from sglang.srt.utils import get_device_sm, is_blackwell

        # Decide MHA vs MLA
        # 判断是否走 MHA_ONE_SHOT 路径
        if forward_batch and forward_batch.forward_mode.is_extend_without_speculative():
            # Check if sequence meets criteria for MHA_ONE_SHOT
            assert forward_batch.seq_lens_cpu is not None
            max_kv_len = forward_batch.seq_lens_cpu.max().item()
            sum_seq_lens = sum(forward_batch.seq_lens_cpu)
            device_sm = get_device_sm()

            # Requirements: H200/B200, short sequences, supported dtype, fits in chunk
            # MHA_ONE_SHOT 条件：SM90/SM100、短序列、bf16/fp8 dtype、满足 chunk 容量、未开 CP
            self.use_mha = (
                (
                    device_sm == 90 or (device_sm >= 100 and device_sm < 110)
                )  # SM90/SM100 only
                and max_kv_len
                <= envs.SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.get()  # Short enough for MHA
                and forward_batch.token_to_kv_pool.dtype
                in [torch.bfloat16, torch.float8_e4m3fn]
                and sum_seq_lens
                <= forward_batch.get_max_chunk_capacity()  # Fits in chunk
                and (not is_nsa_enable_prefill_cp())  # CP not enabled
                and (forward_batch.hisparse_coordinator is None)
            )
        else:
            self.use_mha = False  # Decode/verify always use MLA

        # Set MLA implementation only if not using MHA
        # 仅在非 MHA_ONE_SHOT 且启用自动选择时，决定 MLA prefill 实现
        if not self.use_mha and self.enable_auto_select_prefill_impl:
            if self.nsa_kv_cache_store_fp8:
                if (
                    is_blackwell()
                    and forward_batch is not None
                    and forward_batch.forward_mode == ForwardMode.EXTEND
                ):
                    total_kv_tokens = forward_batch.seq_lens_sum
                    total_q_tokens = forward_batch.extend_num_tokens
                    # Heuristic based on benchmarking flashmla_kv vs flashmla_sparse + dequantize_k_cache_paged
                    # Blackwell 启发式：短 KV 时用 flashmla_sparse（避免全局量化开销）
                    if total_kv_tokens < total_q_tokens * 512:
                        self.nsa_prefill_impl = "flashmla_sparse"
                        return
                # 其他情况（非 Blackwell 或非 EXTEND）使用 flashmla_kv
                self.nsa_prefill_impl = "flashmla_kv"
            else:
                # bf16 kv cache
                # BF16 KV cache：使用 flashmla_sparse（无需量化）
                self.nsa_prefill_impl = "flashmla_sparse"

    def get_topk_transform_method(
        self, forward_mode: Optional[ForwardMode] = None
    ) -> TopkTransformMethod:
        """
        SGLANG_NSA_FUSE_TOPK controls whether to fuse the topk transform into the topk kernel.
        This method is used to select the topk transform method which can be fused or unfused.

        决定 topk 变换方式（RAGGED 或 PAGED）：
        - RAGGED：适用于 FP8 + flashmla_sparse + 标准 EXTEND 模式（topk_indices 直接作为绝对 KV 偏移）
        - PAGED：其他所有情况（topk_indices 映射到 page_table_1）
        """
        if (
            # disable for MTP
            self.nsa_kv_cache_store_fp8
            and self.nsa_prefill_impl == "flashmla_sparse"
            and forward_mode == ForwardMode.EXTEND
        ):
            # FP8 + flashmla_sparse + EXTEND → RAGGED 格式
            topk_transform_method = TopkTransformMethod.RAGGED
        else:
            # 所有其他情况 → PAGED 格式
            topk_transform_method = TopkTransformMethod.PAGED
        return topk_transform_method

    def get_indexer_metadata(
        self, layer_id: int, forward_batch: ForwardBatch
    ) -> NSAIndexerMetadata:
        """构造当前批次的 NSAIndexerMetadata，供 topk 索引器使用。

        Args:
            layer_id: 当前层 ID（暂未使用，预留接口）
            forward_batch: 当前批次信息

        Returns:
            NSAIndexerMetadata 对象，包含 attn_metadata 和 topk_transform_method
        """
        # HiSparse 设备 decode 时强制使用 unfused topk
        force_unfused = (
            forward_batch.hisparse_coordinator is not None
            and forward_batch.forward_mode.is_decode_or_idle()
        )
        return NSAIndexerMetadata(
            attn_metadata=self.forward_metadata,
            topk_transform_method=self.get_topk_transform_method(
                forward_batch.forward_mode
            ),
            paged_mqa_schedule_metadata=self.forward_metadata.paged_mqa_schedule_metadata,
            force_unfused_topk=force_unfused,
        )

    def _compute_flashmla_metadata(self, cache_seqlens: torch.Tensor, seq_len_q: int):
        """计算 FlashMLA tile scheduler 元数据。

        调用 sgl_kernel.flash_mla.get_mla_metadata 获取 tile scheduler 元数据
        和 num_splits，用于 FlashMLA KV paged kernel 的调度。

        Args:
            cache_seqlens: KV 序列长度（NSA 剪裁后，int32）
            seq_len_q: Q 序列长度（通常为 1）

        Returns:
            NSAFlashMLAMetadata 封装对象
        """
        from sgl_kernel.flash_mla import get_mla_metadata

        num_heads_q = self.flashmla_kv_num_q_heads

        flashmla_metadata, num_splits = get_mla_metadata(
            cache_seqlens=cache_seqlens,
            # TODO doc says `num_q_tokens_per_q_seq * num_heads_q // num_heads_k`
            #      but the name looks like need seq_len_q?
            # seq_len_q * num_heads_q / num_heads_k（num_heads_k=1）
            num_q_tokens_per_head_k=seq_len_q * num_heads_q // 1,
            num_heads_k=1,
            num_heads_q=num_heads_q,
            is_fp8_kvcache=True,
            topk=self.nsa_index_topk,
        )

        return NSAFlashMLAMetadata(
            flashmla_metadata=flashmla_metadata,
            num_splits=num_splits,
        )


class NativeSparseAttnMultiStepBackend:
    """NSA 多步投机解码的后端包装器。

    管理多个 NativeSparseAttnBackend 实例，每个实例对应投机解码的一个步骤
    （共 speculative_num_steps - 1 个 draft 步骤）。
    每次 forward 时，所有步骤的元数据同步初始化，以支持批量 draft decode。
    """

    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        # 创建 speculative_num_steps - 1 个后端（最后一步由主后端处理）
        self.attn_backends = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                NativeSparseAttnBackend(
                    model_runner,
                    speculative_step_id=i,
                    topk=self.topk,
                    speculative_num_steps=self.speculative_num_steps,
                )
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # 为每个投机步骤的后端初始化 forward 元数据
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        # 为每个投机步骤的后端初始化 CUDA Graph 静态缓冲区
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        # CUDA Graph capture：为每个投机步骤构造 capture 阶段的 NSAMetadata
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        """CUDA Graph replay：更新所有投机步骤后端的 NSAMetadata 静态缓冲区。

        若启用了 SGLANG_NSA_ENABLE_MTP_PRECOMPUTE_METADATA，
        则只计算一次 PrecomputedMetadata，再通过 fused kernel 高效地
        并行复制到所有（最多 3 个）后端的静态缓冲区；
        超出 3 个时逐一复制。

        否则退化为逐后端独立 replay（性能较低）。
        """
        if envs.SGLANG_NSA_ENABLE_MTP_PRECOMPUTE_METADATA.get():
            # Precompute metadata once (shared across all backends)
            # 只计算一次元数据，所有后端共用同一份 PrecomputedMetadata
            precomputed = self.attn_backends[0]._precompute_replay_metadata(
                bs=bs,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

            # Use multi-backend fused copy when we have 3 or more backends
            # This is 3x faster than calling the single-backend copy 3 times
            # 后端数 >= 3 时，尝试使用 fused_metadata_copy_multi_cuda（3倍加速）
            if self.speculative_num_steps > 3:
                try:
                    from sglang.jit_kernel.fused_metadata_copy import (
                        fused_metadata_copy_multi_cuda,
                    )

                    # 取出前 3 个后端的静态 metadata
                    metadata0 = self.attn_backends[0].decode_cuda_graph_metadata[bs]
                    metadata1 = self.attn_backends[1].decode_cuda_graph_metadata[bs]
                    metadata2 = self.attn_backends[2].decode_cuda_graph_metadata[bs]

                    # Set nsa_prefill_impl for first 3 backends (required by the method)
                    # 初始化前 3 个后端的 nsa_prefill_impl（必须调用否则状态未初始化）
                    for i in range(3):
                        self.attn_backends[i].set_nsa_prefill_impl(forward_batch=None)

                    # Prepare FlashMLA tensors if needed
                    # 准备 FlashMLA 相关的 src/dst 张量（3 个后端各自独立）
                    flashmla_num_splits_src = None
                    flashmla_metadata_src = None
                    flashmla_num_splits_dst0 = None
                    flashmla_num_splits_dst1 = None
                    flashmla_num_splits_dst2 = None
                    flashmla_metadata_dst0 = None
                    flashmla_metadata_dst1 = None
                    flashmla_metadata_dst2 = None

                    if precomputed.flashmla_metadata is not None:
                        flashmla_num_splits_src = (
                            precomputed.flashmla_metadata.num_splits
                        )
                        flashmla_metadata_src = (
                            precomputed.flashmla_metadata.flashmla_metadata
                        )
                        flashmla_num_splits_dst0 = (
                            metadata0.flashmla_metadata.num_splits
                        )
                        flashmla_num_splits_dst1 = (
                            metadata1.flashmla_metadata.num_splits
                        )
                        flashmla_num_splits_dst2 = (
                            metadata2.flashmla_metadata.num_splits
                        )
                        flashmla_metadata_dst0 = (
                            metadata0.flashmla_metadata.flashmla_metadata
                        )
                        flashmla_metadata_dst1 = (
                            metadata1.flashmla_metadata.flashmla_metadata
                        )
                        flashmla_metadata_dst2 = (
                            metadata2.flashmla_metadata.flashmla_metadata
                        )

                    # Call the multi-backend fused kernel for first 3 backends
                    # 一次调用 fused kernel 同时更新前 3 个后端（3x 加速）
                    fused_metadata_copy_multi_cuda(
                        # Source tensors
                        precomputed.cache_seqlens,
                        precomputed.cu_seqlens_k,
                        precomputed.page_indices,
                        precomputed.nsa_cache_seqlens,
                        precomputed.nsa_cu_seqlens_k,
                        precomputed.real_page_table,
                        flashmla_num_splits_src,
                        flashmla_metadata_src,
                        # Destination tensors for backend 0
                        metadata0.cache_seqlens_int32,
                        metadata0.cu_seqlens_k,
                        metadata0.page_table_1,
                        metadata0.nsa_cache_seqlens_int32,
                        metadata0.nsa_cu_seqlens_k,
                        (
                            metadata0.real_page_table
                            if precomputed.real_page_table is not None
                            else None
                        ),
                        flashmla_num_splits_dst0,
                        flashmla_metadata_dst0,
                        # Destination tensors for backend 1
                        metadata1.cache_seqlens_int32,
                        metadata1.cu_seqlens_k,
                        metadata1.page_table_1,
                        metadata1.nsa_cache_seqlens_int32,
                        metadata1.nsa_cu_seqlens_k,
                        (
                            metadata1.real_page_table
                            if precomputed.real_page_table is not None
                            else None
                        ),
                        flashmla_num_splits_dst1,
                        flashmla_metadata_dst1,
                        # Destination tensors for backend 2
                        metadata2.cache_seqlens_int32,
                        metadata2.cu_seqlens_k,
                        metadata2.page_table_1,
                        metadata2.nsa_cache_seqlens_int32,
                        metadata2.nsa_cu_seqlens_k,
                        (
                            metadata2.real_page_table
                            if precomputed.real_page_table is not None
                            else None
                        ),
                        flashmla_num_splits_dst2,
                        flashmla_metadata_dst2,
                        # Parameters
                        bs,
                        precomputed.max_len,
                        precomputed.seqlens_expanded_size,
                    )

                    # Copy remaining backends one by one (if > 3 backends)
                    # 超出 3 个的后端逐一调用 precomputed 复制
                    for i in range(3, self.speculative_num_steps - 1):
                        self.attn_backends[
                            i
                        ].init_forward_metadata_replay_cuda_graph_from_precomputed(
                            bs=bs,
                            precomputed=precomputed,
                            forward_mode=ForwardMode.DECODE,
                        )
                except (ImportError, Exception) as e:
                    # Fallback to loop if multi-backend kernel not available or fails
                    # fused multi kernel 不可用或失败时退回逐一复制
                    if isinstance(e, ImportError):
                        print(
                            "Warning: Multi-backend fused metadata copy kernel not available, falling back to loop."
                        )
                    else:
                        print(
                            f"Warning: Multi-backend fused metadata copy kernel failed with error: {e}, falling back to loop."
                        )
                    for i in range(self.speculative_num_steps - 1):
                        self.attn_backends[
                            i
                        ].init_forward_metadata_replay_cuda_graph_from_precomputed(
                            bs=bs,
                            precomputed=precomputed,
                            forward_mode=ForwardMode.DECODE,
                        )
            else:
                # Less than 3 backends: copy to each backend individually
                # 不足 3 个后端时，逐一使用 precomputed 复制（无 fused 优化）
                for i in range(self.speculative_num_steps - 1):
                    self.attn_backends[
                        i
                    ].init_forward_metadata_replay_cuda_graph_from_precomputed(
                        bs=bs,
                        precomputed=precomputed,
                        forward_mode=ForwardMode.DECODE,
                    )
        else:
            # Fallback: compute metadata separately for each backend
            # 未启用预计算优化：为每个后端独立 replay（计算量为 N 倍）
            for i in range(self.speculative_num_steps - 1):
                self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                    bs=bs,
                    req_pool_indices=forward_batch.req_pool_indices,
                    seq_lens=forward_batch.seq_lens,
                    seq_lens_sum=forward_batch.seq_lens_sum,
                    encoder_lens=None,
                    forward_mode=ForwardMode.DECODE,
                    spec_info=forward_batch.spec_info,
                    seq_lens_cpu=forward_batch.seq_lens_cpu,
                    out_cache_loc=None,
                )
