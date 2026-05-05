from __future__ import annotations

"""
end to end attention solution with aiter kernels
使用 AITER（AMD Inference Toolkit Extensive Routines）kernel 的端到端注意力计算方案，
专为 AMD GPU（ROCm 平台）优化。
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

import torch
import triton

# 导入注意力后端基类
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
# 导入 KV 索引创建工具函数
from sglang.srt.layers.attention.utils import (
    create_flashinfer_kv_indices_triton,
    create_flashmla_kv_indices_triton,
)
# 导入数据并行注意力工具
from sglang.srt.layers.dp_attention import (
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_gfx95_supported

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

# 尝试导入 AMD 专属的 aiter 库
try:
    from aiter import (
        flash_attn_varlen_func,
        get_mla_metadata_info_v1,
        get_mla_metadata_v1,
        get_ps_metadata_info_v1,
        get_ps_metadata_v1,
        mha_batch_prefill_func,
        mla_prefill_ps_asm_fwd,
        mla_reduce_v1,
        paged_attention_ragged,
    )
    from aiter.mla import mla_decode_fwd, mla_prefill_fwd
    from aiter.ops.triton.attention.unified_attention import unified_attention
except ImportError:
    print(
        "aiter is AMD specific kernel library. Please make sure aiter is installed on your AMD device."
    )

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.utils import (
    launch_reshape_and_cache_flash,
    pad_sequence_with_mask,
)
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)

# 是否使用 AITER MLA persist 设计（适用于 fp8 KV cache）
# Use aiter mla persist design for fp8-kv cache
_use_mla_ps_kernel = get_bool_env_var("SGLANG_AITER_MLA_PERSIST", "True")

# 仅在 gfx95 架构上启用 fp8 prefill 注意力
# Use fp8 prefill only on gfx95
_use_fp8_prefill_attn = (
    get_bool_env_var("SGLANG_AITER_FP8_PREFILL_ATTN", "True") and is_gfx95_supported()
)

# Persist
# fast_mode=True if _use_mla_ps_kernel else False
# intra_batch_mode=False if _use_mla_ps_kernel else True

# fake non-ps, intra_batch_mode needs to be True for non-ps-mode
fast_mode = False
# intra_batch_mode 控制 MLA 内批次并行度
intra_batch_mode = True if _use_mla_ps_kernel else False


class WrapperDispatch(Enum):
    # 枚举：用于选择特殊注意力模式的分发类型
    SLIDING_WINDOW = auto()   # 滑动窗口注意力
    CROSS_ATTENTION = auto()  # 交叉注意力（encoder-decoder）


@dataclass
# AITER 后端前向传播所需的元数据
class ForwardMetadata:
    kv_indptr: torch.Tensor          # KV cache 的指针数组（前缀和），长度 bs+1
    kv_indices: torch.Tensor         # KV cache 的 token 索引
    qo_indptr: torch.Tensor          # query/output 的指针数组（前缀和）
    kv_last_page_len: torch.Tensor   # 每个请求最后一页的实际 token 数
    max_q_len: int                   # 最大 query 序列长度
    max_kv_len: Optional[int]        # 最大 KV 序列长度
    work_metadata: Optional[torch.Tensor] = None     # MLA 工作元数据缓冲区
    work_info_set: Optional[torch.Tensor] = None     # MLA 工作信息集
    work_indptr: Optional[torch.Tensor] = None       # MLA 工作指针
    reduce_indptr: Optional[torch.Tensor] = None     # MLA 归约指针
    reduce_final_map: Optional[torch.Tensor] = None  # MLA 最终归约映射
    reduce_partial_map: Optional[torch.Tensor] = None  # MLA 部分归约映射
    num_kv_splits: Optional[int] = None              # KV 分片数量
    run_graph: Optional[bool] = True                 # 是否在 CUDA 图中运行
    custom_mask: Optional[torch.Tensor] = None       # 自定义注意力掩码
    mask_indptr: Optional[torch.Tensor] = None       # 掩码指针数组
    max_extend_len: Optional[int] = None             # 最大扩展长度
    fp8_prefill_kv_indices: Optional[torch.Tensor] = None  # fp8 prefill 的 KV 索引
    swa_page_table: Optional[torch.Tensor] = None    # 滑动窗口注意力页表


# 全局工作区缓冲区（供 aiter kernel 使用）
global_workspace_buffer = None

# AMD ROCm 平台的注意力分片大小（每个分片 256 个 token）
_AITER_PARTITION_SIZE_ROCM = 256


class AiterAttnBackend(AttentionBackend):
    """AITER 注意力后端，专为 AMD GPU（ROCm/gfx）优化的注意力计算实现。"""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,       # 是否跳过 prefill 阶段
        kv_indptr_buf: Optional[torch.Tensor] = None,  # 可复用的 KV 指针缓冲区
        topk: int = 1,                    # 投机解码的 top-k 数量
    ):
        super().__init__()
        # 延迟导入，避免过早初始化 CUDA 上下文
        # Lazy import to avoid the initialization of cuda context
        from sglang.srt.layers.attention.triton_ops.extend_attention import (
            extend_attention_fwd,
        )

        self.input_dtype = model_runner.model_config.dtype

        self.page_size = model_runner.server_args.page_size

        # 禁用 torch.compile 以避免 extend_attention_fwd 编译问题
        self.extend_attention_fwd = torch.compiler.disable(extend_attention_fwd)

        self.device = model_runner.device
        self.is_multimodal = model_runner.model_config.is_multimodal
        # 投机解码草稿 token 数量
        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps
        self.topk = topk
        # 当前 TP 切片下的注意力头数
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.head_dim = model_runner.model_config.head_dim
        # 当前 TP 切片下的 KV 头数
        self.num_kv_head = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.kv_cache_dtype = model_runner.kv_cache_dtype

        # 请求到 token 的映射表
        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        # 是否使用 MLA（多头潜在注意力，DeepSeek 系列使用）
        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA

        # 根据模型类型获取 v_head_dim
        # Get v_head_dim based on model type
        if self.use_mla:
            # MLA 模型从配置中获取 v_head_dim
            # For MLA models, get v_head_dim from model config
            self.v_head_dim = model_runner.model_config.v_head_dim
        elif hasattr(model_runner.token_to_kv_pool, "get_v_head_dim"):
            # 混合模型（Mamba+attention, GDN, Kimi linear 等）layer_id=0 可能不是完整注意力层
            # For hybrid models (Mamba+attention, GDN, Kimi linear),
            # layer_id=0 may not be a full attention layer
            self.v_head_dim = model_runner.token_to_kv_pool.get_v_head_dim()
        else:
            self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[
                -1
            ]

        # Parse constants
        self.max_context_len = model_runner.model_config.context_len
        self.skip_prefill = skip_prefill

        max_bs = model_runner.req_to_token_pool.size

        # 初始化 KV 指针缓冲区
        if kv_indptr_buf is None:
            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
        else:
            self.kv_indptr = kv_indptr_buf

        # 每个请求最后一页的实际 token 数（初始化为 1，表示最后一页有 1 个 token）
        self.kv_last_page_len = torch.ones(
            (max_bs,), dtype=torch.int32, device=model_runner.device
        )
        # query/output 的指针数组（前缀和）
        self.qo_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=model_runner.device
        )
        # 掩码指针数组（用于投机解码的自定义掩码）
        self.mask_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int64, device=model_runner.device
        )
        # KV 索引暂存缓冲区（按需分配）
        self._kv_indices_scratch: Optional[torch.Tensor] = None

        # 为 prefill 阶段创建索引更新器
        # Create prefill indices updater
        if not skip_prefill:
            self.indices_updater_prefill = AiterIndicesUpdaterPrefill(
                model_runner, self
            )
            if self.use_mla:
                # MLA 专用的 prefill 索引更新器
                self.mla_indices_updater_prefill = AiterMlaIndicesUpdaterPrefill(
                    model_runner, self
                )

        # 滑动窗口注意力相关设置
        # sliding window attention
        self.use_sliding_window_kv_pool = (
            isinstance(model_runner.token_to_kv_pool, SWAKVPool)
            and model_runner.token_to_kv_pool.swa_layer_nums > 0
        )

        if self.use_sliding_window_kv_pool:
            self.token_to_kv_pool = model_runner.token_to_kv_pool
            # SWA 需要使用 Triton 统一注意力 kernel
            self.use_triton_unified_attention = True
        else:
            # 通过环境变量控制是否使用 Triton 统一注意力
            self.use_triton_unified_attention = get_bool_env_var(
                "SGLANG_USE_AITER_UNIFIED_ATTN"
            )

        # aiter kernel 相关初始化
        # aiter kernel related initialization
        # 计算最大分片数（基于 ROCm 分片大小）
        self.max_num_partitions = (
            self.max_context_len + _AITER_PARTITION_SIZE_ROCM - 1
        ) // _AITER_PARTITION_SIZE_ROCM

        # 每个 qo 元素的字节数（float32）
        nbyes_per_qo_elem = torch.finfo(torch.float32).bits // 8

        if not (self.use_mla or self.use_triton_unified_attention):
            # 为 aiter paged attention 分配工作区缓冲区
            self.workspace_buffer = torch.empty(
                (max_bs * self.num_head * self.max_num_partitions * self.head_dim)
                * nbyes_per_qo_elem
                + 2 * (max_bs * self.num_head * self.max_num_partitions) * 4,
                dtype=torch.uint8,
                device=self.device,
            )

        # 注意力缩放因子（1/sqrt(head_dim)）
        self.scale = float(1.0 / (self.head_dim**0.5))
        # KV cache 的量化缩放因子（默认为 1.0，无量化）
        self.k_scale = self.v_scale = torch.tensor([1.0], dtype=torch.float32).to(
            self.device
        )

        # 注意力 logit soft cap（0 表示不使用，Gemma2 等模型使用）
        self.logits_soft_cap = 0.0

        # 当前批次的前向传播元数据（在每次 forward 前初始化）
        self.forward_metadata: ForwardMetadata = None

        if self.use_mla:
            # 验证 MLA 支持的注意力头数（4、8 或 16 的倍数）
            _valid_heads = self.num_head in (4, 8) or (
                self.num_head % 16 == 0 and 16 <= self.num_head <= 128
            )
            assert _valid_heads, (
                f"Aiter MLA supports num_head of 4, 8, or multiples of 16 "
                f"in [16, 128].\n"
                f"Provided {self.num_head} number of heads.\n"
                "Try adjusting tensor_parallel_size value."
            )
            # 当头数小于 16 时，填充到 16 并使用重复因子
            self.num_head_padded = 16 if self.num_head < 16 else self.num_head
            self.head_repeat_factor = 16 // self.num_head if self.num_head < 16 else 1

            # 是否启用数据并行注意力
            self.enable_dp_attention = is_dp_attention_enabled()
            self.qo_indptr_ = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
            global _use_mla_ps_kernel, fast_mode, intra_batch_mode

            # current mla_decode_fwd only support fake-nps in self.num_head == 16
            # so all num_head size does not use qh16 kernel to simulate
            # it should not use fake-nps (fast_mode = False, intra_batch_mode = True)
            # it will cause gpu-fault or accuracy issue
            # num_head=32 或 128 时使用 fast_mode（非伪 NPS）
            if self.num_head == 32 or self.num_head == 128:
                fast_mode = True
                intra_batch_mode = False

            # current persist a16w16 mla_decode kernel does not support head_num = 128
            # need to fall back to non-persist
            # only use mla_ps_kernel when fp8 kv_cache
            # for non-fp8 kv_cache on tp8, use non-persist kernel to avoid performance degradation
            # head_num=16 (tp8 perf issue), head_num=128 (unsupported, like tp1 or --enable-dp-attention with tp8-dp8)
            # 非 fp8 KV cache 时，或头数为 16/128 时，回退到非持久化 kernel
            if (
                self.num_head_padded == 16 or self.num_head_padded == 128
            ) and self.kv_cache_dtype is not fp8_dtype:
                _use_mla_ps_kernel = False
                fast_mode = False
                intra_batch_mode = False

            # 最大 KV 分片数（PS kernel 为 32，非 PS 为 None）
            self.max_split_per_batch = 32 if _use_mla_ps_kernel else None

            if self.num_draft_tokens is None and _use_mla_ps_kernel:
                # 无草稿 token 时增加分片数以提高并行度
                self.max_split_per_batch = 64

            self.fix_max_split_per_batch = self.max_split_per_batch

    def make_mla_decode_meta_data_buffer(self, max_seqlen_qo, batch_size):
        """为 MLA 解码阶段分配工作元数据缓冲区。"""
        nhead = self.num_head_padded
        dtype = self.kv_cache_dtype

        if self.enable_dp_attention:
            # 数据并行时，根据当前 GPU 的 SM 数量动态调整分片数
            gpu = torch.cuda.current_device()
            device_properties = torch.cuda.get_device_properties(gpu)
            cu_num = device_properties.multi_processor_count
            self.max_split_per_batch = min(
                (cu_num + batch_size - 1) // batch_size, self.fix_max_split_per_batch
            )

        # 查询 MLA 元数据各缓冲区的尺寸和数据类型
        (
            (work_meta_data_size, work_meta_data_type),
            (work_indptr_size, work_indptr_type),
            (work_info_set_size, work_info_set_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = get_mla_metadata_info_v1(
            batch_size,
            max_seqlen_qo,
            nhead,
            dtype,
            dtype,
            is_sparse=False,
            fast_mode=fast_mode,
            num_kv_splits=self.max_split_per_batch,
            intra_batch_mode=intra_batch_mode,
        )

        # aiter implementation
        # the tensor's meaning please refer aiter/ops/attention.py
        # 分配各个工作缓冲区
        work_metadata = torch.empty(
            work_meta_data_size, dtype=work_meta_data_type, device="cuda"
        )
        work_indptr = torch.empty(
            work_indptr_size, dtype=work_indptr_type, device="cuda"
        )
        work_info_set = torch.empty(
            work_info_set_size,
            dtype=work_info_set_type,
            device="cuda",
        )
        reduce_indptr = torch.empty(
            reduce_indptr_size, dtype=reduce_indptr_type, device="cuda"
        )
        reduce_final_map = torch.empty(
            reduce_final_map_size, dtype=reduce_final_map_type, device="cuda"
        )
        reduce_partial_map = torch.empty(
            reduce_partial_map_size, dtype=reduce_partial_map_type, device="cuda"
        )

        return (
            work_metadata,
            work_indptr,
            work_info_set,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
        )

    def make_mla_meta_data(
        self,
        qo_indptr,
        kv_indptr,
        kv_last_page_len,
        work_metadata,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        max_q_len,
        fast_mode,
        max_split_per_batch,
        intra_batch_mode,
    ):
        """计算 MLA 解码的调度元数据，填充工作缓冲区。"""
        nhead_kv = 1  # MLA 模式下 KV 头数为 1（latent 向量）
        page_size = self.page_size
        dtype = self.kv_cache_dtype

        # 调用 aiter 的 MLA 元数据计算函数
        meta = get_mla_metadata_v1(
            qo_indptr,
            kv_indptr,
            kv_last_page_len,
            self.num_head_padded // nhead_kv,
            nhead_kv,
            False,
            work_metadata,
            work_info_set,
            work_indptr,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            kv_granularity=max(page_size, 16),  # KV 访问粒度，至少 16
            max_seqlen_qo=max_q_len,
            uni_seqlen_qo=max_q_len,
            fast_mode=fast_mode,
            max_split_per_batch=max_split_per_batch,
            intra_batch_mode=intra_batch_mode,
            dtype_q=dtype,
            dtype_kv=dtype,
        )

    def make_mla_prefill_ps_meta_data_buffer(
        self, batch_size: int, max_qlen: int, qlen_granularity: int
    ):
        """为 MLA prefill（PS 持久化模式）分配工作元数据缓冲区。"""
        (
            (work_meta_data_size, work_meta_data_type),
            (work_indptr_size, work_indptr_type),
            (work_info_size, work_info_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = get_ps_metadata_info_v1(
            batch_size=batch_size,
            num_head_k=self.num_kv_head,
            max_qlen=max_qlen,
            qlen_granularity=qlen_granularity,
        )

        device = self.device
        # 按照查询到的尺寸分配各缓冲区
        work_metadata_ptrs = torch.empty(
            work_meta_data_size, dtype=work_meta_data_type, device=device
        )
        work_indptr = torch.empty(
            work_indptr_size, dtype=work_indptr_type, device=device
        )
        work_info = torch.empty(work_info_size, dtype=work_info_type, device=device)
        reduce_indptr = torch.empty(
            reduce_indptr_size, dtype=reduce_indptr_type, device=device
        )
        reduce_final_map = torch.empty(
            reduce_final_map_size, dtype=reduce_final_map_type, device=device
        )
        reduce_partial_map = torch.empty(
            reduce_partial_map_size, dtype=reduce_partial_map_type, device=device
        )

        return (
            work_metadata_ptrs,
            work_indptr,
            work_info,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
        )

    def make_mla_prefill_ps_meta_data(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        seq_lens: torch.Tensor,
        work_metadata: torch.Tensor,
        work_indptr: torch.Tensor,
        work_info: torch.Tensor,
        reduce_indptr: torch.Tensor,
        reduce_final_map: torch.Tensor,
        reduce_partial_map: torch.Tensor,
        is_causal: bool = True,
    ):
        """计算 MLA prefill PS 模式的调度元数据。"""
        gqa_ratio = self.num_head // self.num_kv_head
        num_heads_k = self.num_kv_head
        tile_q = 256           # 每个 tile 处理的 query 数量
        qhead_granularity = gqa_ratio           # query 头粒度（GQA 比率）
        qlen_granularity = tile_q // qhead_granularity  # query 长度粒度
        kvlen_granularity = max(128, self.page_size)    # KV 长度粒度，至少 128
        block_size = self.page_size

        # 将输入张量转换到 CPU 以便调用 aiter 的 CPU 端函数
        qo_indptr_cpu = qo_indptr.to("cpu", dtype=torch.int32)
        kv_indptr_cpu = kv_indptr.to("cpu", dtype=torch.int32)
        seq_lens_cpu = seq_lens.to("cpu", dtype=torch.int32)

        get_ps_metadata_v1(
            qo_indptr_cpu,
            kv_indptr_cpu,
            seq_lens_cpu,
            gqa_ratio,
            num_heads_k,
            work_metadata,
            work_indptr,
            work_info,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            qhead_granularity=qhead_granularity,
            qlen_granularity=qlen_granularity,
            kvlen_granularity=kvlen_granularity,
            block_size=block_size,
            is_causal=is_causal,
        )

    # for page size > 1 useful conversion function
    def _transform_table_1_to_real(self, page_table: torch.Tensor) -> torch.Tensor:
        """将 page_size=1 格式的页表转换为实际分页后的页表（page_size>1 时有效）。"""
        page_size = self.page_size
        if page_size == 1:
            return page_table
        max_seqlen_k = page_table.shape[1]
        # 生成步长索引（每 page_size 个 token 取一个）
        strided_indices = torch.arange(
            0, max_seqlen_k, page_size, device=page_table.device, dtype=torch.int32
        )
        # 将 token 索引转换为页面索引（除以 page_size）
        return page_table[:, strided_indices] // page_size

    def _resolve_v2_num_draft_tokens(
        self,
        extend_seq_lens: Optional[torch.Tensor] = None,
        extend_seq_lens_cpu: Optional[list[int]] = None,
    ) -> int:
        """解析 DRAFT_EXTEND_V2 模式下每个请求固定的扩展长度。"""
        num_draft_tokens = self.num_draft_tokens
        if num_draft_tokens is None:
            if extend_seq_lens is not None and extend_seq_lens.numel() > 0:
                # 热路径中直接从张量获取，避免列表遍历
                # Avoid list scans in hot path when tensor lengths are already available.
                num_draft_tokens = int(extend_seq_lens[0].item())
            elif extend_seq_lens_cpu:
                num_draft_tokens = max(extend_seq_lens_cpu)
            else:
                raise ValueError(
                    "DRAFT_EXTEND_V2 requires speculative_num_draft_tokens or "
                    "non-empty extend_seq_lens/extend_seq_lens_cpu."
                )

        num_draft_tokens = int(num_draft_tokens)
        # 验证所有请求的扩展长度相同（DRAFT_EXTEND_V2 要求固定长度）
        if extend_seq_lens is not None and extend_seq_lens.numel() > 0:
            if not torch.all(extend_seq_lens == num_draft_tokens):
                raise ValueError(
                    "DRAFT_EXTEND_V2 expects fixed extend length per request; got "
                    f"extend_seq_lens={extend_seq_lens}, expected all == {num_draft_tokens}."
                )
        if extend_seq_lens_cpu and any(
            x != num_draft_tokens for x in extend_seq_lens_cpu
        ):
            raise ValueError(
                "DRAFT_EXTEND_V2 expects fixed extend length per request; got "
                f"{extend_seq_lens_cpu}, expected all == {num_draft_tokens}."
            )
        return num_draft_tokens

    def _get_kv_indices_scratch(
        self, required_tokens: int, device: torch.device
    ) -> torch.Tensor:
        """按需获取或重新分配 KV 索引暂存缓冲区（避免频繁内存分配）。"""
        if (
            self._kv_indices_scratch is None
            or self._kv_indices_scratch.device != device
            or self._kv_indices_scratch.numel() < required_tokens
        ):
            # 需要的 token 数超过现有缓冲区时，重新分配更大的缓冲区
            self._kv_indices_scratch = torch.empty(
                required_tokens, dtype=torch.int32, device=device
            )
        return self._kv_indices_scratch[:required_tokens]

    def _set_uniform_qo_indptr(
        self, bs: int, tokens_per_req: int, device: torch.device
    ) -> torch.Tensor:
        """设置均匀的 qo_indptr（每个请求有相同数量的 token）。"""
        qo_indptr = self.qo_indptr[: bs + 1]
        # 生成等步长的前缀和（步长为 tokens_per_req）
        qo_indptr[: bs + 1] = torch.arange(
            0,
            bs * tokens_per_req + 1,
            step=tokens_per_req,
            dtype=torch.int32,
            device=device,
        )
        return qo_indptr

    def _ensure_spec_v2_topk_supported(self):
        """确保 SPEC_V2 路径只在 topk<=1 时使用（当前限制）。"""
        if self.topk > 1:
            raise NotImplementedError(
                "AiterAttnBackend SPEC_V2 path currently supports topk <= 1 only. "
                f"Got topk={self.topk}."
            )

    def _mla_decode_fwd_with_head_pad(
        self,
        q: torch.Tensor,
        k_buffer_flat: torch.Tensor,
        layer,
        **kwargs,
    ):
        """调用 mla_decode_fwd 时处理 num_head<16 的头维度填充问题。

        Wrap mla_decode_fwd with head-dimension padding for num_head < 16.

        When head_repeat_factor > 1 (i.e. num_head is 4 or 8), q is
        repeat-interleaved to reach num_head_padded (16) before the kernel
        call, and the corresponding output columns are sliced back afterward.
        q / o must already be shaped (..., num_head, head_dim).
        """
        if self.head_repeat_factor > 1:
            # 将 query 按头维度重复到 num_head_padded（16）
            q_in = q.repeat_interleave(self.head_repeat_factor, dim=1)
            o = q.new_empty(
                (q.shape[0], self.num_head_padded, layer.v_head_dim),
                dtype=self.input_dtype,
            )
            mla_decode_fwd(q_in, k_buffer_flat, o, **kwargs)
            # 从填充后的输出中取回原始头的结果
            return o[:, :: self.head_repeat_factor, :]
        else:
            # 无需填充，直接调用 kernel
            o = q.new_empty(
                (q.shape[0], layer.tp_q_head_num, layer.v_head_dim),
                dtype=self.input_dtype,
            )
            mla_decode_fwd(q, k_buffer_flat, o, **kwargs)
            return o

    def mla_fp8_prefill_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
    ):
        """MLA fp8 prefill 注意力计算（使用 aiter PS ASM kernel）。"""
        total_q = q.shape[0]
        nhead = layer.tp_q_head_num
        v_head_dim = layer.v_head_dim

        # 确保 q/k/v 为 fp8 格式
        if q.dtype != fp8_dtype:
            q = q.to(fp8_dtype)
        if k.dtype != fp8_dtype:
            k = k.to(fp8_dtype)
        if v.dtype != fp8_dtype:
            v = v.to(fp8_dtype)
        # fp8 量化缩放因子（全 1，不进行额外缩放）
        one_scale = torch.ones((), dtype=torch.float32, device=q.device)

        tile_q = 256  # PS kernel 的 tile 大小
        reduce_indptr = self.forward_metadata.reduce_indptr
        reduce_final_map = self.forward_metadata.reduce_final_map
        reduce_partial_map = self.forward_metadata.reduce_partial_map

        # 分配部分结果缓冲区（PS kernel 会写入多个 tile 的中间结果）
        logits = torch.empty(
            (reduce_partial_map.size(0) * tile_q, nhead, v_head_dim),
            dtype=torch.float32,
            device=q.device,
        )
        attn_lse = torch.empty(
            (reduce_partial_map.size(0) * tile_q, nhead),
            dtype=torch.float32,
            device=q.device,
        )
        final_lse = torch.empty(
            (total_q, nhead),
            dtype=torch.float32,
            device=q.device,
        )
        output = q.new_empty(
            (total_q, nhead, v_head_dim),
            dtype=self.input_dtype,
        )

        # 调用 aiter PS ASM prefill 前向传播 kernel
        mla_prefill_ps_asm_fwd(
            q,
            k,
            v,
            self.forward_metadata.qo_indptr,
            self.forward_metadata.kv_indptr,
            self.forward_metadata.fp8_prefill_kv_indices,
            self.forward_metadata.work_indptr,
            self.forward_metadata.work_info_set,
            self.forward_metadata.max_q_len,
            layer.scaling,
            True,   # is_causal
            logits,
            attn_lse,
            output,
            one_scale,
            one_scale,
            one_scale,
        )
        # 使用归约 kernel 合并多个 tile 的结果
        mla_reduce_v1(
            logits,
            attn_lse,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            tile_q,
            output,
            final_lse,
        )
        return output

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """初始化 AITER 注意力后端的前向传播辅助变量（元数据）。"""
        # Init auxiliary variables for aiter attention backend.

        bs = forward_batch.batch_size
        kv_indptr = self.kv_indptr
        spec_info = forward_batch.spec_info
        qo_indptr = None
        kv_last_page_len = None
        max_q_len = None
        max_kv_len = None

        # MLA 解码工作缓冲区（初始化为 None）
        work_metadata = None
        work_indptr = None
        work_info_set = None
        reduce_indptr = None
        reduce_final_map = None
        reduce_partial_map = None

        num_kv_splits = None
        swa_page_table = None
        # 当前批次中最长 KV 序列长度
        max_kv_len = forward_batch.seq_lens_cpu.max().item()

        if forward_batch.forward_mode.is_decode_or_idle():
            # 解码阶段处理
            if spec_info is None or forward_batch.forward_mode.is_idle():
                # 普通解码或空闲：计算 KV 指针前缀和
                kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]

                if not self.use_triton_unified_attention:
                    # 使用 FlashInfer KV 索引格式（扁平化索引列表）
                    kv_indices = self._get_kv_indices_scratch(
                        forward_batch.seq_lens_sum, forward_batch.seq_lens.device
                    )
                    # 调用 Triton kernel 填充 KV 索引（每个 token 对应一个物理 KV 块位置）
                    create_flashinfer_kv_indices_triton[(bs,)](
                        self.req_to_token,
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        kv_indptr,
                        None,
                        kv_indices,
                        self.req_to_token.stride(0),
                    )
                else:
                    # 使用 FlashMLA 统一注意力，需要二维 page table（bs × max_kv_len）
                    max_q_len = 1
                    page_size = self.page_size
                    # 每条序列最多需要的 KV 块数
                    max_num_blocks_per_seq = (max_kv_len + page_size - 1) // page_size
                    kv_indices = torch.zeros(
                        bs, max_kv_len, dtype=torch.int32, device=self.device
                    )

                    # 调用 Triton kernel 填充 FlashMLA 格式的 KV 索引（二维 page table）
                    create_flashmla_kv_indices_triton[(bs,)](
                        self.req_to_token,
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        None,
                        kv_indices,
                        self.req_to_token.stride(0),
                        max_kv_len,
                        1,
                    )

                    if self.use_sliding_window_kv_pool:
                        # 滑动窗口 KV 池：将全局 page table 转换为 SWA（滑动窗口注意力）子池索引
                        swa_page_table = (
                            self.token_to_kv_pool.translate_loc_from_full_to_swa(
                                kv_indices
                            )
                        )
                        # 将 1-indexed page table 转换为真实的 0-indexed 块编号
                        kv_indices = self._transform_table_1_to_real(kv_indices)
                        swa_page_table = self._transform_table_1_to_real(swa_page_table)
                    elif self.page_size > 1:
                        # 仅做 1-indexed 到真实索引的转换
                        kv_indices = self._transform_table_1_to_real(kv_indices)

                    # qo_indptr: Q/O 序列的前缀和指针（每条序列末页长度的累积和）
                    qo_indptr = self.qo_indptr[: bs + 1]
                    qo_indptr[1 : bs + 1] = torch.cumsum(
                        self.kv_last_page_len[:bs], dim=0
                    )

            else:
                # 投机解码：复用 spec_info 中已预计算的 KV 指针和索引
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
                # 投机解码时 bs 由 kv_indptr 决定
                bs = kv_indptr.shape[0] - 1

            if self.use_mla:
                # MLA 解码：需要单独的 qo_indptr_（区别于非 MLA 路径）
                qo_indptr = self.qo_indptr_[: bs + 1]
                # qo_indptr 的前缀和由每条序列末页长度累积
                qo_indptr[1 : bs + 1] = torch.cumsum(self.kv_last_page_len[:bs], dim=0)
                kv_last_page_len = self.kv_last_page_len[:bs]
                max_q_len = 1  # 解码阶段每条请求只有 1 个 Q token

                if _use_mla_ps_kernel:
                    # 使用持久化 MLA PS 内核时，预先分配调度元数据缓冲区
                    (
                        work_metadata,
                        work_indptr,
                        work_info_set,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                    ) = self.make_mla_decode_meta_data_buffer(max_q_len, bs)

                    num_kv_splits = self.max_split_per_batch

                    # 计算 MLA 解码的调度元数据（分块策略、规约映射等）
                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        work_metadata,
                        work_info_set,
                        work_indptr,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                        max_q_len,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

            # 将所有解码阶段的元数据打包存入 forward_metadata
            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                max_kv_len,
                work_metadata=work_metadata,
                work_info_set=work_info_set,
                work_indptr=work_indptr,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                num_kv_splits=num_kv_splits,
                run_graph=False,
                swa_page_table=swa_page_table,
            )

        elif forward_batch.forward_mode.is_draft_extend_v2():
            # EAGLE V2: DRAFT_EXTEND_V2 模式——将所有草稿 token 扩展写入 KV 缓存
            self._ensure_spec_v2_topk_supported()
            if self.use_mla:
                device = forward_batch.seq_lens.device
                # 获取每步草稿 token 数量
                num_draft_tokens = self._resolve_v2_num_draft_tokens()
                # 构建均匀的 qo_indptr（每条请求有相同数量的 Q token）
                qo_indptr = self._set_uniform_qo_indptr(bs, num_draft_tokens, device)

                kv_indptr = self.kv_indptr[: bs + 1]
                kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)

                kv_indices = self._get_kv_indices_scratch(
                    forward_batch.seq_lens_sum, device
                )

                # 用 Triton kernel 填充 FlashInfer 格式的 KV 扁平索引
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )

                if _use_mla_ps_kernel:
                    # 使用持久化 MLA PS 内核，每条序列 Q 长度等于 num_draft_tokens
                    max_seqlen_qo = num_draft_tokens
                    (
                        work_metadata,
                        work_indptr,
                        work_info_set,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                    ) = self.make_mla_decode_meta_data_buffer(max_seqlen_qo, bs)

                    num_kv_splits = self.max_split_per_batch

                    # 计算 draft_extend_v2 的 MLA 调度元数据
                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        self.kv_last_page_len[:bs],
                        work_metadata,
                        work_info_set,
                        work_indptr,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                        max_seqlen_qo,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                # 将 draft_extend_v2 元数据打包
                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    self.kv_last_page_len[:bs],
                    num_draft_tokens,
                    forward_batch.seq_lens_cpu.max().item(),
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                    run_graph=False,
                )
            else:
                # 非 MLA 路径：调用预填充索引更新器处理草稿扩展
                self.indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    prefix_lens=None,
                    encoder_lens=forward_batch.encoder_lens,
                    spec_info=forward_batch.spec_info,
                )
                # 从预填充更新器取索引构建 ForwardMetadata
                self.forward_metadata = ForwardMetadata(
                    self.indices_updater_prefill.kv_indptr,
                    self.indices_updater_prefill.kv_indices,
                    None,
                    None,
                    self.indices_updater_prefill.max_q_len,
                    self.indices_updater_prefill.max_kv_len,
                )
        elif forward_batch.forward_mode.is_draft_extend():
            # EAGLE V1: DRAFT_EXTEND 模式——根据 spec_info.num_accepted_tokens 扩展草稿 KV
            if self.use_mla:
                # MLA 路径：通过 spec_info 生成预填充注意力参数
                kv_indices, kv_indptr, qo_indptr, custom_mask = (
                    spec_info.generate_attn_arg_prefill(
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        forward_batch.seq_lens_sum,
                        self.req_to_token,
                    )
                )

                if _use_mla_ps_kernel:
                    # 使用持久化 MLA PS 内核时计算 Q 序列最大长度
                    max_seqlen_qo = max(forward_batch.extend_seq_lens_cpu)
                    (
                        work_metadata,
                        work_indptr,
                        work_info_set,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                    ) = self.make_mla_decode_meta_data_buffer(max_seqlen_qo, bs)

                    num_kv_splits = self.max_split_per_batch

                    # 计算 draft_extend（EAGLE V1）的 MLA 调度元数据
                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        self.kv_last_page_len[:bs],
                        work_metadata,
                        work_info_set,
                        work_indptr,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                        max_seqlen_qo,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                # 打包 EAGLE V1 MLA draft_extend 元数据
                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    # self.mla_indices_updater_prefill.kv_last_page_len,
                    self.kv_last_page_len[:bs],
                    max(forward_batch.extend_seq_lens_cpu),
                    forward_batch.seq_lens_cpu.max().item(),
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                    run_graph=False,
                )
            else:
                # Non-MLA draft_extend: use triton extend kernel with causal masking
                kv_indices, kv_indptr, qo_indptr, custom_mask = (
                    spec_info.generate_attn_arg_prefill(
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        forward_batch.seq_lens_sum,
                        self.req_to_token,
                    )
                )
                # 非 MLA 路径将 kv_indices 转换为 int64（兼容 Triton kernel）
                kv_indices = kv_indices.to(torch.int64)
                # draft_max_extend_len：草稿中被接受 token 的最大扩展长度
                draft_max_extend_len = torch.max(spec_info.num_accepted_tokens).item()

                # 打包非 MLA EAGLE V1 元数据，包含 custom_mask 和 mask_indptr
                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    None,
                    draft_max_extend_len,
                    None,
                    custom_mask=custom_mask,
                    mask_indptr=None,
                    max_extend_len=draft_max_extend_len,
                )
        elif forward_batch.forward_mode.is_target_verify():
            # 目标验证阶段（EAGLE）：验证草稿 token 是否被目标模型接受
            if self.use_mla:
                draft_num = spec_info.draft_token_num
                # 目标验证时 KV 序列长度 = 已有序列长度 + 草稿 token 数
                kv_lens = forward_batch.seq_lens + draft_num
                kv_lens_sum = forward_batch.seq_lens_sum + draft_num * bs
                device = forward_batch.seq_lens.device

                # qo_indptr：每条请求有 draft_num 个 Q token（均匀步长）
                qo_indptr = self.qo_indptr[: bs + 1]
                qo_indptr[: bs + 1] = torch.arange(
                    0,
                    (1 + bs) * draft_num,
                    step=draft_num,
                    dtype=torch.int32,
                    device=device,
                )
                # kv_indptr：目标验证 KV 序列的前缀和指针
                kv_indptr = self.kv_indptr[: bs + 1]
                kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
                # 为目标验证分配 KV 扁平索引缓冲区
                kv_indices = self._get_kv_indices_scratch(
                    kv_lens_sum,
                    device,
                )
                # 用 Triton kernel 填充目标验证所需的 KV 索引
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    forward_batch.req_pool_indices,
                    kv_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )

                # if self.kv_cache_dtype == fp8_dtype:
                if _use_mla_ps_kernel:
                    # 使用持久化 MLA PS 内核：Q 序列长度 = draft_num
                    max_seqlen_qo = draft_num
                    (
                        work_metadata,
                        work_indptr,
                        work_info_set,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                    ) = self.make_mla_decode_meta_data_buffer(max_seqlen_qo, bs)

                    num_kv_splits = self.max_split_per_batch

                    # 计算目标验证阶段的 MLA 调度元数据
                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        self.kv_last_page_len[:bs],
                        work_metadata,
                        work_info_set,
                        work_indptr,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                        max_seqlen_qo,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    # self.mla_indices_updater_prefill.kv_last_page_len,
                    self.kv_last_page_len[:bs],
                    draft_num,
                    None,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                    run_graph=False,
                )
            else:
                # Non-MLA target_verify: use triton extend kernel with custom mask
                bs = len(forward_batch.req_pool_indices)
                draft_num = spec_info.draft_token_num

                # qo_indptr：每条请求有 draft_num 个 Q token，均匀步长
                qo_indptr = torch.arange(
                    0,
                    (1 + bs) * draft_num,
                    step=draft_num,
                    dtype=torch.int32,
                    device=self.device,
                )

                # kv_indptr：目标序列（不含草稿 token）的 KV 前缀和
                kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]

                # kv_indices：目标序列的扁平化 KV 物理索引
                kv_indices = torch.empty(
                    kv_indptr[-1], dtype=torch.int64, device=self.device
                )
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )

                # custom_mask：spec_info 中的自定义注意力掩码（草稿 token 可见性）
                custom_mask = spec_info.custom_mask
                # seq_mask_len：每条序列的掩码长度 = draft_num × (seq_len + draft_num)
                seq_mask_len = draft_num * (forward_batch.seq_lens + draft_num)
                mask_indptr = self.mask_indptr
                mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len[:bs], dim=0)
                mask_indptr = mask_indptr[: bs + 1]

                # 打包非 MLA 目标验证元数据
                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    None,
                    draft_num,
                    None,
                    custom_mask=custom_mask,
                    mask_indptr=mask_indptr,
                    max_extend_len=draft_num,
                )
        else:
            # 预填充（Extend）阶段：处理新 token 的 KV 扩展
            prefix_lens = forward_batch.extend_prefix_lens

            if self.is_multimodal:
                # 多模态场景下强制使用非无前缀路径（存在视觉 token 前缀）
                extend_no_prefix = False
            else:
                # 纯文本：检查是否所有序列的前缀长度均为零
                extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
            if self.use_mla:
                # MLA 预填充路径：通过专用更新器计算 KV/QO 指针和索引
                self.mla_indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    forward_batch.extend_seq_lens,
                    max(forward_batch.extend_seq_lens_cpu),
                    forward_batch.seq_lens_cpu.max().item(),
                    spec_info=None,
                )

                # 从更新器取 max_q_len、qo_indptr、kv_indptr
                max_q_len = self.mla_indices_updater_prefill.max_q_len
                qo_indptr = self.mla_indices_updater_prefill.qo_indptr
                kv_indptr = self.mla_indices_updater_prefill.kv_indptr

                # 初始化 PS 内核调度缓冲区为 None
                work_metadata = None
                work_indptr = None
                work_info_set = None
                reduce_indptr = None
                reduce_final_map = None
                reduce_partial_map = None
                fp8_prefill_kv_indices = None

                if _use_fp8_prefill_attn:
                    # FP8 预填充注意力：基于 tile 粒度计算调度元数据
                    tile_q = 256  # Q tile 大小（与内核实现对应）
                    # qlen_granularity：每个 tile 覆盖的 Q token 数（考虑 GQA 比例）
                    qlen_granularity = tile_q // (self.num_head // self.num_kv_head)
                    (
                        work_metadata,
                        work_indptr,
                        work_info_set,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                    ) = self.make_mla_prefill_ps_meta_data_buffer(
                        bs, max_q_len, qlen_granularity
                    )

                    # 计算 FP8 预填充的 MLA 调度元数据（因果掩码）
                    self.make_mla_prefill_ps_meta_data(
                        qo_indptr,
                        kv_indptr,
                        forward_batch.seq_lens,
                        work_metadata,
                        work_indptr,
                        work_info_set,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                        is_causal=True,
                    )

                    # fp8_prefill_kv_indices：连续整数索引（对应顺序排列的 KV token）
                    total_s = forward_batch.seq_lens_sum
                    fp8_prefill_kv_indices = torch.arange(
                        total_s, device=self.device, dtype=torch.int32
                    )

                # 打包 MLA 预填充元数据
                self.forward_metadata = ForwardMetadata(
                    self.mla_indices_updater_prefill.kv_indptr,
                    self.mla_indices_updater_prefill.kv_indices,
                    qo_indptr,
                    self.kv_last_page_len[:bs],
                    max_q_len,
                    self.mla_indices_updater_prefill.max_kv_len,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    fp8_prefill_kv_indices=fp8_prefill_kv_indices,
                )
            else:
                # 非 MLA 预填充：调用标准预填充索引更新器
                self.indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    prefix_lens,
                    encoder_lens=forward_batch.encoder_lens,
                    spec_info=None,
                )

                if self.use_sliding_window_kv_pool:
                    # 滑动窗口 KV 池：将全局索引转换为 SWA 子池索引
                    swa_page_table = (
                        self.token_to_kv_pool.translate_loc_from_full_to_swa(
                            self.indices_updater_prefill.kv_indices
                        )
                    )

                # 打包非 MLA 预填充元数据
                self.forward_metadata = ForwardMetadata(
                    self.indices_updater_prefill.kv_indptr,
                    self.indices_updater_prefill.kv_indices,
                    None,
                    None,
                    max(forward_batch.extend_seq_lens_cpu),
                    forward_batch.seq_lens_cpu.max().item(),
                    swa_page_table=swa_page_table,
                )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        """初始化 CUDA Graph 静态缓冲区（在图捕获前调用）。

        分配足够大的 kv_indptr、qo_indptr、mask_indptr 以覆盖最大批量大小，
        并预分配 kv_indices、custom_mask 等静态缓冲区供图回放时复用。
        """
        # PR #20978 pads max_bs beyond pool_size for higher cuda-graph
        # coverage. Reallocate indptr buffers so they fit the padded max_bs.
        # See: https://github.com/sgl-project/sglang/pull/20978
        if max_bs + 1 > self.kv_indptr.shape[0]:
            # 重新分配 kv_indptr 以适应更大的批量
            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=self.device
            )
            # 重新分配 qo_indptr（Q/O 前缀和指针）
            self.qo_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=self.device
            )
            # 重新分配 mask_indptr（自定义掩码前缀和指针）
            self.mask_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int64, device=self.device
            )
            if hasattr(self, "qo_indptr_"):
                # MLA 路径使用独立的 qo_indptr_，同步扩容
                self.qo_indptr_ = torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=self.device
                )

        # cuda_graph_kv_last_page_len：每条序列末页 KV 长度（初始值为 1）
        self.cuda_graph_kv_last_page_len = torch.ones(
            max_bs, dtype=torch.int32, device=self.device
        )
        if kv_indices_buf is None:
            # 按最大序列长度计算每条序列最多需要的 KV 块数
            max_num_blocks_per_seq = (
                self.max_context_len + self.page_size - 1
            ) // self.page_size
            # 分配扁平化 KV 索引缓冲区（bs × max_num_blocks_per_seq）
            self.cuda_graph_kv_indices = torch.zeros(
                (max_bs * max_num_blocks_per_seq),
                dtype=torch.int32,
                device=self.device,
            )
        else:
            # 使用外部传入的共享缓冲区（多后端复用场景）
            self.cuda_graph_kv_indices = kv_indices_buf

        if not self.skip_prefill:
            # 预填充阶段需要自定义掩码缓冲区（max_tokens × max_context_len）
            self.cuda_graph_custom_mask = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.uint8,
                device=self.device,
            )

        # if self.use_mla and (_use_mla_ps_kernel or self.kv_cache_dtype == fp8_dtype):
        if self.use_mla and _use_mla_ps_kernel:
            # MLA + 持久化 PS 内核：预分配 MLA 解码调度元数据缓冲区
            # for persistent mla_decode_fwd
            max_seqlen_qo = (
                1 if self.num_draft_tokens is None else self.num_draft_tokens
            )

            (
                self.work_metadata,
                self.work_indptr,
                self.work_info_set,
                self.reduce_indptr,
                self.reduce_final_map,
                self.reduce_partial_map,
            ) = self.make_mla_decode_meta_data_buffer(max_seqlen_qo, max_bs)

        else:
            # 非 MLA 或不使用 PS 内核：将调度元数据缓冲区置为 None
            self.work_metadata = None
            self.work_indptr = None
            self.work_info_set = None

            self.reduce_indptr = None
            self.reduce_final_map = None
            self.reduce_partial_map = None

        if self.use_sliding_window_kv_pool:
            # 滑动窗口 KV 池：额外分配 SWA page table 缓冲区（bs × max_blocks）
            max_num_blocks_per_seq = (
                self.max_context_len + self.page_size - 1
            ) // self.page_size
            self.cuda_graph_swa_page_table = torch.zeros(
                (max_bs, max_num_blocks_per_seq),
                dtype=torch.int32,
                device=self.device,
            )

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
        """捕获 CUDA Graph 时调用：用当前批次的真实数据填充静态缓冲区。

        与 init_forward_metadata_replay_cuda_graph 的区别在于：
        capture 阶段将实际张量值写入静态缓冲区，graph 捕获后地址固定不变；
        replay 阶段只需更新缓冲区内容（不改变指针），供图回放时直接使用。
        """

        num_kv_splits = None
        # num_kv_splits_indptr = None

        work_metadata = None
        work_info_set = None
        work_indptr = None

        reduce_indptr = None
        reduce_final_map = None
        reduce_partial_map = None

        swa_page_table = None

        # 从 CPU seq_lens 中取最大值作为 max_kv_len
        max_kv_len = torch.max(seq_lens).item()

        if forward_mode.is_decode_or_idle():
            qo_indptr = None
            kv_last_page_len = None
            max_q_len = None

            if spec_info is None:

                if not self.use_triton_unified_attention:
                    # FlashInfer 路径：用 kv_indptr 和扁平化 kv_indices 填充静态缓冲区
                    kv_indptr = self.kv_indptr
                    kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
                    kv_indptr = kv_indptr[: bs + 1]
                    # 复用预分配的 cuda_graph_kv_indices 静态缓冲区
                    kv_indices = self.cuda_graph_kv_indices
                    create_flashinfer_kv_indices_triton[(bs,)](
                        self.req_to_token,
                        req_pool_indices,
                        seq_lens,
                        kv_indptr,
                        None,
                        kv_indices,
                        self.req_to_token.stride(0),
                    )
                else:
                    # FlashMLA（统一注意力）路径：page table 格式
                    max_q_len = 1
                    # 计算每条序列最多需要的 page 数
                    max_num_blocks_per_seq = (
                        self.max_context_len + self.page_size - 1
                    ) // self.page_size
                    # 将静态缓冲区 reshape 为 (max_bs, max_num_blocks_per_seq)
                    kv_indices = self.cuda_graph_kv_indices.view(
                        -1, max_num_blocks_per_seq
                    )

                    # 从 req_to_token 中读取当前批次的 page 索引
                    page_indices = self.req_to_token[req_pool_indices[:bs], :max_kv_len]

                    if self.use_sliding_window_kv_pool:
                        # 滑动窗口：将全局 page indices 映射到 SWA 子池索引
                        swa_page_indices = (
                            self.token_to_kv_pool.translate_loc_from_full_to_swa(
                                page_indices
                            )
                        )
                        # 将 1-indexed 转换为真实块编号
                        page_indices = self._transform_table_1_to_real(page_indices)
                        swa_page_indices = self._transform_table_1_to_real(
                            swa_page_indices
                        )

                        new_rows = swa_page_indices.shape[0]
                        new_cols = swa_page_indices.shape[1]

                        # 将 page_indices 和 swa_page_indices 写入静态缓冲区
                        kv_indices[:new_rows, :new_cols].copy_(page_indices)
                        swa_page_table = self.cuda_graph_swa_page_table
                        swa_page_table[:new_rows, :new_cols].copy_(swa_page_indices)
                    elif self.page_size > 1:
                        # 仅做 1-indexed 转换
                        page_indices = self._transform_table_1_to_real(page_indices)
                        new_rows = page_indices.shape[0]
                        new_cols = page_indices.shape[1]
                        kv_indices[:new_rows, :new_cols].copy_(page_indices)

                    # qo_indptr 由末页长度的累积和计算
                    qo_indptr = self.qo_indptr[: bs + 1]
                    qo_indptr[1 : bs + 1] = torch.cumsum(
                        self.cuda_graph_kv_last_page_len[:bs], dim=0
                    )

                    kv_indptr = None
            else:
                # 投机解码（EAGLE）：直接使用 spec_info 的 KV 指针和索引
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices

            if self.use_mla:
                # MLA 路径：使用独立的 qo_indptr_（区别于非 MLA 路径）
                qo_indptr = self.qo_indptr_[: bs + 1]
                qo_indptr[1 : bs + 1] = torch.cumsum(
                    self.cuda_graph_kv_last_page_len[:bs], dim=0
                )
                kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
                max_q_len = 1

                if _use_mla_ps_kernel:
                    # 捕获阶段计算并写入 MLA 调度元数据（work_metadata 等）
                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        self.work_metadata,
                        self.work_info_set,
                        self.work_indptr,
                        self.reduce_indptr,
                        self.reduce_final_map,
                        self.reduce_partial_map,
                        max_q_len,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                    # 将静态缓冲区引用保存到局部变量，供 ForwardMetadata 使用
                    work_metadata = self.work_metadata
                    work_info_set = self.work_info_set
                    work_indptr = self.work_indptr

                    reduce_indptr = self.reduce_indptr
                    reduce_final_map = self.reduce_final_map
                    reduce_partial_map = self.reduce_partial_map

            # 打包解码阶段 CUDA Graph 捕获元数据
            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                max_kv_len,
                work_metadata=work_metadata,
                work_info_set=work_info_set,
                work_indptr=work_indptr,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                num_kv_splits=num_kv_splits,
                swa_page_table=swa_page_table,
            )

        elif forward_mode.is_target_verify():
            # 目标验证阶段（CUDA Graph 版）
            qo_indptr = self.qo_indptr[: bs + 1]
            # 每条请求有 num_draft_tokens 个 Q token，均匀步长
            qo_indptr[: bs + 1] = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            if self.use_mla:
                # MLA：KV 序列包含草稿 token
                kv_lens = seq_lens + self.num_draft_tokens
            else:
                # 非 MLA：KV 序列只含原始序列
                kv_lens = seq_lens
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
            # 复用静态 kv_indices 缓冲区
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                kv_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
            kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
            max_q_len = self.num_draft_tokens

            if self.use_mla:
                if _use_mla_ps_kernel:
                    # 计算并保存目标验证的 MLA 调度元数据
                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        self.work_metadata,
                        self.work_info_set,
                        self.work_indptr,
                        self.reduce_indptr,
                        self.reduce_final_map,
                        self.reduce_partial_map,
                        max_q_len,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                    work_metadata = self.work_metadata
                    work_info_set = self.work_info_set
                    work_indptr = self.work_indptr

                    reduce_indptr = self.reduce_indptr
                    reduce_final_map = self.reduce_final_map
                    reduce_partial_map = self.reduce_partial_map

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    max_kv_len,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                )
            else:
                # 非 MLA 目标验证：使用 custom_mask 和 mask_indptr
                custom_mask = self.cuda_graph_custom_mask
                # 将 spec_info 的自定义掩码复制到静态缓冲区前段
                custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask
                seq_mask_len = max_q_len * (seq_lens + max_q_len)
                mask_indptr = self.mask_indptr
                mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len[:bs], dim=0)
                mask_indptr = mask_indptr[: bs + 1]

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    max_kv_len,
                    custom_mask=custom_mask,
                    mask_indptr=mask_indptr,
                    max_extend_len=max_q_len,
                )
        elif forward_mode.is_draft_extend_v2():
            # EAGLE V2: Uses fixed num_draft_tokens per batch
            # EAGLE V2（CUDA Graph）：固定草稿 token 数
            self._ensure_spec_v2_topk_supported()
            num_tokens_per_bs = self._resolve_v2_num_draft_tokens()
            qo_indptr = self._set_uniform_qo_indptr(bs, num_tokens_per_bs, self.device)
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            # 复用静态 kv_indices 缓冲区
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
            kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
            max_q_len = num_tokens_per_bs

            if self.use_mla and _use_mla_ps_kernel:
                # MLA + PS 内核：计算 draft_extend_v2 的调度元数据
                num_kv_splits = self.max_split_per_batch

                self.make_mla_meta_data(
                    qo_indptr,
                    kv_indptr,
                    kv_last_page_len,
                    self.work_metadata,
                    self.work_info_set,
                    self.work_indptr,
                    self.reduce_indptr,
                    self.reduce_final_map,
                    self.reduce_partial_map,
                    max_q_len,
                    fast_mode=fast_mode,
                    max_split_per_batch=num_kv_splits,
                    intra_batch_mode=intra_batch_mode,
                )

                work_metadata = self.work_metadata
                work_info_set = self.work_info_set
                work_indptr = self.work_indptr

                reduce_indptr = self.reduce_indptr
                reduce_final_map = self.reduce_final_map
                reduce_partial_map = self.reduce_partial_map

            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                max_kv_len,
                work_metadata=work_metadata,
                work_info_set=work_info_set,
                work_indptr=work_indptr,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                num_kv_splits=num_kv_splits,
            )
        elif forward_mode.is_draft_extend():
            # EAGLE V1: Uses speculative_num_steps + 1
            # EAGLE V1（CUDA Graph）：Q 序列长度 = speculative_num_steps + 1
            num_tokens_per_bs = self.speculative_num_steps + 1
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                bs * num_tokens_per_bs + 1,
                step=num_tokens_per_bs,
                dtype=torch.int32,
                device=self.device,
            )
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            # 复用静态 kv_indices 缓冲区
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            if self.use_mla:
                kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
                max_q_len = num_tokens_per_bs

                if _use_mla_ps_kernel:
                    # 计算 EAGLE V1 MLA 调度元数据
                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        self.work_metadata,
                        self.work_info_set,
                        self.work_indptr,
                        self.reduce_indptr,
                        self.reduce_final_map,
                        self.reduce_partial_map,
                        max_q_len,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                    work_metadata = self.work_metadata
                    work_info_set = self.work_info_set
                    work_indptr = self.work_indptr

                    reduce_indptr = self.reduce_indptr
                    reduce_final_map = self.reduce_final_map
                    reduce_partial_map = self.reduce_partial_map

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    max_kv_len,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                )
            else:
                # Non-MLA draft_extend cuda graph: use triton extend kernel
                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    None,
                    num_tokens_per_bs,
                    None,
                    custom_mask=None,
                    mask_indptr=None,
                    max_extend_len=num_tokens_per_bs,
                )
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
        """CUDA Graph 回放阶段：更新静态缓冲区内容（不更改张量地址）。

        与 capture 阶段相同逻辑，但仅填充缓冲区值，
        因为 CUDA Graph 已捕获静态张量地址，回放时直接使用相同地址。
        """

        num_kv_splits = None
        # num_kv_splits_indptr = None

        work_metadata = None
        work_info_set = None
        work_indptr = None

        reduce_indptr = None
        reduce_final_map = None
        reduce_partial_map = None

        swa_page_table = None
        # 从 CPU 侧 seq_lens 计算最大 KV 序列长度
        max_kv_len = seq_lens_cpu.max().item()

        if forward_mode.is_decode_or_idle():
            qo_indptr = None
            kv_last_page_len = None
            max_q_len = None

            if spec_info is None:
                if not self.use_triton_unified_attention:
                    # FlashInfer 路径：更新 kv_indptr 并重新填充 kv_indices
                    kv_indptr = self.kv_indptr
                    kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
                    kv_indptr = kv_indptr[: bs + 1]
                    kv_indices = self.cuda_graph_kv_indices
                    create_flashinfer_kv_indices_triton[(bs,)](
                        self.req_to_token,
                        req_pool_indices,
                        seq_lens,
                        kv_indptr,
                        None,
                        kv_indices,
                        self.req_to_token.stride(0),
                    )
                else:
                    # FlashMLA 路径：更新二维 page table 静态缓冲区
                    max_q_len = 1
                    max_num_blocks_per_seq = (
                        self.max_context_len + self.page_size - 1
                    ) // self.page_size
                    kv_indices = self.cuda_graph_kv_indices.view(
                        -1, max_num_blocks_per_seq
                    )

                    # 从 req_to_token 读取当前批次的 page 索引
                    page_indices = self.req_to_token[req_pool_indices[:bs], :max_kv_len]

                    if self.use_sliding_window_kv_pool:
                        # 将全局 page 索引映射到 SWA 子池
                        swa_page_indices = (
                            self.token_to_kv_pool.translate_loc_from_full_to_swa(
                                page_indices
                            )
                        )

                        page_indices = self._transform_table_1_to_real(page_indices)
                        swa_page_indices = self._transform_table_1_to_real(
                            swa_page_indices
                        )

                        new_rows = swa_page_indices.shape[0]
                        new_cols = swa_page_indices.shape[1]

                        # 写入静态 kv_indices 和 swa_page_table 缓冲区
                        kv_indices[:new_rows, :new_cols].copy_(page_indices)
                        swa_page_table = self.cuda_graph_swa_page_table
                        swa_page_table[:new_rows, :new_cols].copy_(swa_page_indices)
                    elif self.page_size > 1:
                        page_indices = self._transform_table_1_to_real(page_indices)
                        new_rows = page_indices.shape[0]
                        new_cols = page_indices.shape[1]
                        kv_indices[:new_rows, :new_cols].copy_(page_indices)

                    # 更新 qo_indptr（末页长度累积和）
                    qo_indptr = self.qo_indptr[: bs + 1]
                    qo_indptr[1 : bs + 1] = torch.cumsum(
                        self.cuda_graph_kv_last_page_len[:bs], dim=0
                    )

                    kv_indptr = None
            else:
                # 投机解码：复用 spec_info 的 KV 指针/索引
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices

            if self.use_mla:
                # MLA 路径：更新 qo_indptr_ 和 MLA 调度元数据
                qo_indptr = self.qo_indptr_[: bs + 1]
                qo_indptr[1 : bs + 1] = torch.cumsum(
                    self.cuda_graph_kv_last_page_len[:bs], dim=0
                )
                kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
                max_q_len = 1

                if _use_mla_ps_kernel:
                    num_kv_splits = self.max_split_per_batch

                    # 重新计算 MLA 调度元数据（持久化 PS 内核）
                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        self.work_metadata,
                        self.work_info_set,
                        self.work_indptr,
                        self.reduce_indptr,
                        self.reduce_final_map,
                        self.reduce_partial_map,
                        max_q_len,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                    # 将静态缓冲区引用保存到局部变量
                    work_metadata = self.work_metadata
                    work_info_set = self.work_info_set
                    work_indptr = self.work_indptr

                    reduce_indptr = self.reduce_indptr
                    reduce_final_map = self.reduce_final_map
                    reduce_partial_map = self.reduce_partial_map

            # 用更新后的值重建 ForwardMetadata（静态缓冲区地址不变）
            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                max_kv_len,
                work_metadata=work_metadata,
                work_info_set=work_info_set,
                work_indptr=work_indptr,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                num_kv_splits=num_kv_splits,
                swa_page_table=swa_page_table,
                # num_kv_splits_indptr=num_kv_splits_indptr,
            )

        elif forward_mode.is_target_verify():
            # 目标验证阶段（CUDA Graph 回放）
            bs = len(req_pool_indices)
            qo_indptr = self.qo_indptr[: bs + 1]
            # Q 指针：每条请求有 num_draft_tokens 个 Q token
            qo_indptr[: bs + 1] = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            if self.use_mla:
                # MLA：KV 序列包含草稿 token
                kv_lens = seq_lens + self.num_draft_tokens
            else:
                kv_lens = seq_lens
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
            # 复用静态 kv_indices 缓冲区
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                kv_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
            kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
            max_q_len = self.num_draft_tokens

            if self.use_mla:
                if _use_mla_ps_kernel:
                    # 重新计算目标验证的 MLA 调度元数据
                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        self.work_metadata,
                        self.work_info_set,
                        self.work_indptr,
                        self.reduce_indptr,
                        self.reduce_final_map,
                        self.reduce_partial_map,
                        max_q_len,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                    work_metadata = self.work_metadata
                    work_info_set = self.work_info_set
                    work_indptr = self.work_indptr

                    reduce_indptr = self.reduce_indptr
                    reduce_final_map = self.reduce_final_map
                    reduce_partial_map = self.reduce_partial_map

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    max_kv_len,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                )
            else:
                # 非 MLA：使用 custom_mask + mask_indptr
                custom_mask = self.cuda_graph_custom_mask
                custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask
                seq_mask_len = max_q_len * (seq_lens + max_q_len)
                mask_indptr = self.mask_indptr[: bs + 1]
                mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len, dim=0)

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    max_kv_len,
                    custom_mask=custom_mask,
                    mask_indptr=mask_indptr,
                    max_extend_len=max_q_len,
                )
        elif forward_mode.is_draft_extend_v2():
            # EAGLE V2: Fixed num_draft_tokens per batch
            # EAGLE V2 回放：固定草稿 token 数，更新 KV/QO 指针
            self._ensure_spec_v2_topk_supported()
            # 只取当前 bs 的 seq_lens
            seq_lens = seq_lens[:bs]
            num_tokens_per_bs = self._resolve_v2_num_draft_tokens()
            # extend_lens：每条序列扩展长度均等于 num_tokens_per_bs
            extend_lens = torch.full(
                (bs,), num_tokens_per_bs, dtype=torch.int32, device=seq_lens.device
            )

            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[1 : bs + 1] = torch.cumsum(extend_lens, dim=0)
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            # 复用静态 kv_indices 缓冲区
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
            max_q_len = num_tokens_per_bs

            if self.use_mla and _use_mla_ps_kernel:
                # 重新计算 draft_extend_v2 的 MLA 调度元数据
                num_kv_splits = self.max_split_per_batch

                self.make_mla_meta_data(
                    qo_indptr,
                    kv_indptr,
                    kv_last_page_len,
                    self.work_metadata,
                    self.work_info_set,
                    self.work_indptr,
                    self.reduce_indptr,
                    self.reduce_final_map,
                    self.reduce_partial_map,
                    max_q_len,
                    fast_mode=fast_mode,
                    max_split_per_batch=num_kv_splits,
                    intra_batch_mode=intra_batch_mode,
                )

                work_metadata = self.work_metadata
                work_info_set = self.work_info_set
                work_indptr = self.work_indptr

                reduce_indptr = self.reduce_indptr
                reduce_final_map = self.reduce_final_map
                reduce_partial_map = self.reduce_partial_map

            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                max_kv_len,
                work_metadata=work_metadata,
                work_info_set=work_info_set,
                work_indptr=work_indptr,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                num_kv_splits=num_kv_splits,
            )
        elif forward_mode.is_draft_extend():
            # EAGLE V1: Uses spec_info.num_accepted_tokens
            # EAGLE V1 回放：根据每条请求实际接受 token 数构建 qo_indptr
            num_tokens_per_bs = self.speculative_num_steps + 1
            # 只取当前 bs 的 seq_lens
            seq_lens = seq_lens[:bs]
            # extend_lens 由 spec_info.num_accepted_tokens 决定（可变长度）
            extend_lens = spec_info.num_accepted_tokens[:bs]
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[1 : bs + 1] = torch.cumsum(extend_lens, dim=0)
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
            # max_q_len 使用上界（speculative_num_steps + 1）
            max_q_len = num_tokens_per_bs

            if self.use_mla and _use_mla_ps_kernel:
                # 重新计算 EAGLE V1 MLA 调度元数据
                num_kv_splits = self.max_split_per_batch

                self.make_mla_meta_data(
                    qo_indptr,
                    kv_indptr,
                    kv_last_page_len,
                    self.work_metadata,
                    self.work_info_set,
                    self.work_indptr,
                    self.reduce_indptr,
                    self.reduce_final_map,
                    self.reduce_partial_map,
                    max_q_len,
                    fast_mode=fast_mode,
                    max_split_per_batch=num_kv_splits,
                    intra_batch_mode=intra_batch_mode,
                )

                work_metadata = self.work_metadata
                work_info_set = self.work_info_set
                work_indptr = self.work_indptr

                reduce_indptr = self.reduce_indptr
                reduce_final_map = self.reduce_final_map
                reduce_partial_map = self.reduce_partial_map

            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                max_kv_len,
                work_metadata=work_metadata,
                work_info_set=work_info_set,
                work_indptr=work_indptr,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                num_kv_splits=num_kv_splits,
            )

        else:
            raise ValueError("Invalid forward mode")

    def get_cuda_graph_seq_len_fill_value(self):
        """返回 CUDA Graph 填充序列长度时使用的默认值。

        普通解码返回 1（每步生成 1 个 token），
        投机解码返回 num_draft_tokens（草稿长度）。
        """
        return 1 if self.num_draft_tokens is None else self.num_draft_tokens

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        # AITER verify path does not require post-draft buffer patching currently.
        # This override prevents overlap-plan stream mode from failing with the
        # base class NotImplementedError.
        # AITER 验证路径无需草稿后填充缓冲区，直接返回
        pass

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        """预填充（Extend）阶段的前向传播。

        处理 MLA 和非 MLA 两种路径，支持：
        1. 无前缀的纯自注意力（flash_attn_varlen_func）
        2. 有前缀的跨序列注意力（KV 索引寻址）
        3. MXFP4 权重的 FP8 prefill 融合内核
        4. 目标验证和草稿扩展（EAGLE 系列）
        """
        # 从当前层获取 logits soft cap 参数
        self.logits_soft_cap = layer.logit_cap

        # 确定 KV 缓存写入位置（交叉注意力使用 encoder_out_cache_loc）
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        # FP8 KV 缓存量化场景：获取 K/V 的反量化缩放因子
        k_descale = None
        v_descale = None
        if self.kv_cache_dtype == fp8_dtype:
            k_descale = layer.k_scale if layer.k_scale is not None else self.k_scale
            v_descale = layer.v_scale if layer.v_scale is not None else self.k_scale

        if k is not None:
            assert v is not None
            if save_kv_cache:
                # Only use SWA-specific kv cache write (reshape_and_cache_flash) when
                # both unified attention and sliding window kv pool are active.
                # Non-SWA models (e.g. Qwen3-VL) enabled via SGLANG_USE_AITER_UNIFIED_ATTN
                # use standard set_kv_buffer, as they lack SWA-specific attributes
                # like full_to_swa_index_mapping.
                if (
                    self.use_triton_unified_attention
                    and self.use_sliding_window_kv_pool
                ):
                    # SWA + 统一注意力：使用 reshape_and_cache_flash 写入 paged KV
                    token_to_kv_pool = forward_batch.token_to_kv_pool
                    k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                        layer.layer_id
                    )
                    # slot_mapping_swa：全局 slot 到 SWA 子池 slot 的映射
                    slot_mapping_swa = token_to_kv_pool.full_to_swa_index_mapping

                    # 调用 reshape_and_cache_flash 将 K/V 写入 paged KV cache
                    launch_reshape_and_cache_flash(
                        k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                        v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                        k_cache.view(
                            -1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
                        ),
                        v_cache.view(
                            -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                        ),
                        cache_loc,
                        (
                            slot_mapping_swa.long()
                            if layer.sliding_window_size > 0
                            else None
                        ),
                        k_scale=k_descale,
                        v_scale=v_descale,
                    )
                elif self.use_mla:
                    # MLA 路径：调用通用 set_kv_buffer（不需要缩放因子）
                    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
                else:
                    # 普通路径：set_kv_buffer 带 FP8 缩放因子
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, k_descale, v_descale
                    )

        if self.use_mla:
            # MLA 预填充：从 forward_metadata 中取元数据
            max_q_len = self.forward_metadata.max_q_len
            max_kv_len = self.forward_metadata.max_kv_len
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices
            qo_indptr = self.forward_metadata.qo_indptr
            # 从 KV 缓存池获取 K/V Buffer（压缩格式）
            K_Buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            V_Buffer = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
            # MLA 参数维度分解：kv_lora_rank、qk_rope_head_dim、qk_nope_head_dim
            kv_lora_rank = V_Buffer.shape[-1]
            qk_rope_head_dim = K_Buffer.shape[-1] - kv_lora_rank
            qk_nope_head_dim = k.shape[-1] - qk_rope_head_dim
            assert len(q.shape) == 3
            assert len(k.shape) == 3
            assert len(v.shape) == 3

            if (
                forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
                and not forward_batch.forward_mode.is_draft_extend_v2()
            ):
                # 普通预填充（非投机解码扩展）
                extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
                if kv_indices.shape[0] == 0 or extend_no_prefix:
                    # 无前缀：直接使用 Q/K/V 做 varlen 自注意力（不需要 KV 索引）
                    if _use_fp8_prefill_attn:
                        # FP8 prefill 路径
                        output = self.mla_fp8_prefill_attn(
                            q,
                            k,
                            v,
                            layer,
                        )
                    else:
                        # 标准 flash_attn_varlen_func：Q/K/V 共享同一 indptr（无前缀）
                        output = flash_attn_varlen_func(
                            q,
                            k,
                            v,
                            qo_indptr,
                            qo_indptr,
                            max_q_len,
                            max_q_len,
                            softmax_scale=layer.scaling,
                            causal=True,
                        )
                    return output
                elif layer.qk_head_dim != (kv_lora_rank + qk_rope_head_dim):
                    # 有前缀且 qk_head_dim 不等于 KV 缓存总维度：
                    # 需要从 KV 缓存中通过 kv_indices 取出 KV，解压后拼接 RoPE 分量
                    K_Buffer = torch.index_select(K_Buffer, 0, kv_indices)
                    # kvc：KV LoRA 压缩分量；k_pe：RoPE 编码分量
                    kvc, k_pe = torch.split(
                        K_Buffer, [kv_lora_rank, qk_rope_head_dim], dim=-1
                    )

                    if self.kv_cache_dtype == fp8_dtype:
                        # FP8 KV 缓存：将 kvc 和 k_pe 转换回 Q 的浮点精度
                        dtype = q.dtype

                        kvc = kvc.to(dtype)
                        k_pe = k_pe.to(dtype)

                    if (
                        _use_fp8_prefill_attn
                        and layer.kv_b_proj.weight.dtype == torch.uint8
                    ):
                        # MXFP4 weights + FP8 prefill: fuse GEMM, nope/v split, and k_pe cat
                        # into a single kernel (fused_gemm_afp4wfp4_split_cat) that writes k and v
                        # directly in FP8, avoiding a separate elementwise cast
                        # MXFP4 权重 + FP8 prefill：融合 GEMM、分割、拼接为单个内核
                        k, v = layer.kv_b_proj(
                            (
                                kvc.squeeze(1),
                                k_pe.expand(-1, layer.tp_k_head_num, -1),
                                qk_nope_head_dim,
                                layer.v_head_dim,
                                fp8_dtype,
                            )
                        )[0]
                    else:
                        # 标准路径：通过 kv_b_proj 还原 K/V，再拼接 RoPE 分量
                        kv = layer.kv_b_proj(kvc.contiguous())[0]

                        kv = kv.view(
                            -1, layer.tp_k_head_num, qk_nope_head_dim + layer.v_head_dim
                        )
                        # 将 kv 分割为 k_nope 和 v
                        k, v = torch.split(
                            kv, [qk_nope_head_dim, layer.v_head_dim], dim=-1
                        )
                        # 将 k_pe 广播并拼接到 k_nope 上，得到完整 K
                        k = torch.cat(
                            [
                                k,
                                torch.broadcast_to(
                                    k_pe,
                                    (k_pe.shape[0], layer.tp_k_head_num, k_pe.shape[2]),
                                ),
                            ],
                            dim=-1,
                        )

                    assert (
                        forward_batch.extend_prefix_lens.shape
                        == forward_batch.extend_seq_lens.shape
                    )

                    if _use_fp8_prefill_attn:
                        # FP8 prefill：调用融合内核
                        return self.mla_fp8_prefill_attn(q, k, v, layer)
                    else:
                        # 标准 varlen 注意力：Q 使用 qo_indptr，KV 使用 kv_indptr（含前缀）
                        return flash_attn_varlen_func(
                            q,
                            k,
                            v,
                            qo_indptr,
                            kv_indptr,
                            max_q_len,
                            max_kv_len,
                            softmax_scale=layer.scaling,
                            causal=True,
                        )

                else:
                    # qk_head_dim 与 KV 缓存维度相同：使用 mla_prefill_fwd 直接操作 KV Buffer
                    if layer.qk_head_dim != layer.v_head_dim:
                        # Q/V 维度不同：提前分配输出 tensor
                        o = q.new_empty(
                            (q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                        )
                    else:
                        o = torch.empty_like(q)

                    # 调用 MLA 预填充专用内核（使用 KV 索引寻址）
                    mla_prefill_fwd(
                        q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                        K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                        o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                        qo_indptr,
                        kv_indptr,
                        kv_indices,
                        self.forward_metadata.kv_last_page_len,
                        self.forward_metadata.max_q_len,
                        layer.scaling,
                        layer.logit_cap,
                    )
                    K_Buffer = K_Buffer.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
                    return o
            elif forward_batch.forward_mode.is_target_verify():
                # 目标验证阶段：使用 _mla_decode_fwd_with_head_pad 进行批量解码注意力
                work_metadata = self.forward_metadata.work_metadata
                work_indptr = self.forward_metadata.work_indptr
                work_info_set = self.forward_metadata.work_info_set

                reduce_indptr = self.forward_metadata.reduce_indptr
                reduce_final_map = self.forward_metadata.reduce_final_map
                reduce_partial_map = self.forward_metadata.reduce_partial_map

                num_kv_splits = self.forward_metadata.num_kv_splits

                o = self._mla_decode_fwd_with_head_pad(
                    q,
                    K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                    layer,
                    qo_indptr=self.forward_metadata.qo_indptr,
                    kv_indptr=self.forward_metadata.kv_indptr,
                    kv_indices=self.forward_metadata.kv_indices,
                    kv_last_page_lens=self.forward_metadata.kv_last_page_len,
                    max_seqlen_q=self.forward_metadata.max_q_len,
                    sm_scale=layer.scaling,
                    logit_cap=layer.logit_cap,
                    work_meta_data=work_metadata,
                    work_indptr=work_indptr,
                    work_info_set=work_info_set,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    q_scale=k_descale,
                    kv_scale=k_descale,
                    intra_batch_mode=intra_batch_mode,
                    num_kv_splits=num_kv_splits,
                )
                return o
            elif (
                forward_batch.forward_mode.is_draft_extend()
                or forward_batch.forward_mode.is_draft_extend_v2()
            ):
                # 草稿扩展阶段（EAGLE V1/V2）：使用变长 Q 序列解码注意力
                work_metadata = self.forward_metadata.work_metadata
                work_indptr = self.forward_metadata.work_indptr
                work_info_set = self.forward_metadata.work_info_set

                reduce_indptr = self.forward_metadata.reduce_indptr
                reduce_final_map = self.forward_metadata.reduce_final_map
                reduce_partial_map = self.forward_metadata.reduce_partial_map

                num_kv_splits = self.forward_metadata.num_kv_splits

                if self.forward_metadata.run_graph is not True:
                    # 非图模式：需要对 Q 做 padding 以对齐最大序列长度
                    bs, q_pad, q_mask = pad_sequence_with_mask(
                        q.view(q.shape[0], -1),
                        qo_indptr[:-1],
                        forward_batch.extend_seq_lens,
                        self.forward_metadata.max_q_len,
                    )
                    o = self._mla_decode_fwd_with_head_pad(
                        q_pad.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                        K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                        layer,
                        qo_indptr=self.forward_metadata.qo_indptr,
                        kv_indptr=self.forward_metadata.kv_indptr,
                        kv_indices=self.forward_metadata.kv_indices,
                        kv_last_page_lens=self.forward_metadata.kv_last_page_len,
                        max_seqlen_q=self.forward_metadata.max_q_len,
                        sm_scale=layer.scaling,
                        logit_cap=layer.logit_cap,
                        work_meta_data=work_metadata,
                        work_indptr=work_indptr,
                        work_info_set=work_info_set,
                        reduce_indptr=reduce_indptr,
                        reduce_final_map=reduce_final_map,
                        reduce_partial_map=reduce_partial_map,
                        q_scale=k_descale,
                        kv_scale=k_descale,
                        intra_batch_mode=intra_batch_mode,
                        num_kv_splits=num_kv_splits,
                    )

                    # 裁剪掉 padding 部分，只返回有效 Q token 的输出
                    total_valid_q = int(qo_indptr[-1].item())
                    return o[:total_valid_q]
                else:
                    # 图模式（CUDA Graph）：Q 已经是 padding 后的形状，直接调用
                    o = self._mla_decode_fwd_with_head_pad(
                        q,
                        K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                        layer,
                        qo_indptr=self.forward_metadata.qo_indptr,
                        kv_indptr=self.forward_metadata.kv_indptr,
                        kv_indices=self.forward_metadata.kv_indices,
                        kv_last_page_lens=self.forward_metadata.kv_last_page_len,
                        max_seqlen_q=self.forward_metadata.max_q_len,
                        sm_scale=layer.scaling,
                        logit_cap=layer.logit_cap,
                        work_meta_data=work_metadata,
                        work_indptr=work_indptr,
                        work_info_set=work_info_set,
                        reduce_indptr=reduce_indptr,
                        reduce_final_map=reduce_final_map,
                        reduce_partial_map=reduce_partial_map,
                        q_scale=k_descale,
                        kv_scale=k_descale,
                        intra_batch_mode=intra_batch_mode,
                        num_kv_splits=num_kv_splits,
                    )
                    return o
            else:
                raise ValueError(
                    f"Invalid forward mode for MLA prefill: {forward_batch.forward_mode=}"
                )
        else:
            # 非 MLA 预填充路径（标准 MHA/GQA）
            if (
                forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend()
            ):
                # Use triton extend kernel which supports custom masks and causal masking
                # 目标验证 / EAGLE V1 草稿扩展：使用支持自定义 mask 的 Triton extend kernel
                if layer.qk_head_dim != layer.v_head_dim:
                    # Q/V 维度不同时预分配输出
                    o = q.new_empty(
                        (q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                    )
                else:
                    o = torch.empty_like(q)

                # 调用 extend_attention_fwd（Triton 实现）支持 custom mask
                self.extend_attention_fwd(
                    q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    k.contiguous(),
                    v.contiguous(),
                    o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                    forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
                    forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
                    self.forward_metadata.qo_indptr,
                    self.forward_metadata.kv_indptr,
                    self.forward_metadata.kv_indices,
                    self.forward_metadata.custom_mask,
                    True,  # causal
                    self.forward_metadata.mask_indptr,
                    self.forward_metadata.max_extend_len,
                    1.0,  # k_scale
                    1.0,  # v_scale
                    layer.scaling,
                    logit_cap=layer.logit_cap,
                )
                return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

            # 普通预填充路径：使用 mha_batch_prefill_func（AITER 批量预填充内核）
            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )

            # bs0 = batch_size + 1（包含边界元素的 indptr 长度）
            bs0 = forward_batch.batch_size + 1

            # To keep the mha_batch_prefill_func function parameters
            # declare the necessary parameter and assign None as default value
            # 声明 q_descale 并赋默认值 None（维持 mha_batch_prefill_func 参数签名）
            q_descale = None

            # TODO kkhuang-amd need to remove it when mha_batch_prefill_func support fp8-kv
            if self.kv_cache_dtype == fp8_dtype:
                # FP8 KV 缓存时需要将 Q 也转为 fp8 并设置反量化缩放因子
                q = q.to(fp8_dtype)
                q_descale = layer.k_scale if layer.k_scale is not None else self.k_scale

            # 默认无滑动窗口（window_size = (-1, -1) 表示全局注意力）
            window_size = (-1, -1)
            page_table = self.forward_metadata.kv_indices

            if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
                # 滑动窗口注意力：设置窗口大小和 SWA page table
                window_size = (layer.sliding_window_size, -1)
                if self.forward_metadata.swa_page_table is not None:
                    page_table = self.forward_metadata.swa_page_table

            # 调用 AITER 的 mha_batch_prefill_func 批量预填充内核
            o = mha_batch_prefill_func(
                q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                k_cache,
                v_cache,
                self.qo_indptr[:bs0],
                self.forward_metadata.kv_indptr[:bs0],
                page_table,
                self.forward_metadata.max_q_len,
                self.forward_metadata.max_kv_len,
                causal=True,
                logits_soft_cap=self.logits_soft_cap,
                alibi_slopes=None,
                return_lse=False,
                return_attn_probs=False,
                window_size=window_size,
                sink_ptr=sinks,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
            )

            # The fp8bf16 aiter prefill kernel returns bf16 even when the
            # model computes in fp16. Cast back so the attention output keeps
            # the same dtype as the rest of the model activations.
            # FP8 prefill 内核可能输出 bf16，需要转回模型输入精度
            if o.dtype != self.input_dtype:
                o = o.to(self.input_dtype)

            return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        """解码阶段的前向传播。

        每步只有 1 个 Q token（或投机解码时为 num_draft_tokens），
        支持 MLA / 标准 MHA，以及 FlashMLA 统一注意力和分页注意力两种后端。
        """
        # 将 Q 展平为二维：(total_tokens, tp_q_head_num × qk_head_dim)
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        # FP8 KV 缓存场景：获取反量化缩放因子
        k_descale = None
        v_descale = None
        if self.kv_cache_dtype == fp8_dtype:
            k_descale = layer.k_scale if layer.k_scale is not None else self.k_scale
            v_descale = layer.v_scale if layer.v_scale is not None else self.k_scale

        if save_kv_cache:
            # Only use SWA-specific kv cache write (reshape_and_cache_flash) when
            # both unified attention and sliding window kv pool are active.
            # Non-SWA models (e.g. Qwen3-VL) enabled via SGLANG_USE_AITER_UNIFIED_ATTN
            # use standard set_kv_buffer, as they lack SWA-specific attributes
            # like full_to_swa_index_mapping.
            if self.use_triton_unified_attention and self.use_sliding_window_kv_pool:
                # SWA + 统一注意力：使用 reshape_and_cache_flash 写入分页 KV
                token_to_kv_pool = forward_batch.token_to_kv_pool
                k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                    layer.layer_id
                )
                slot_mapping_swa = token_to_kv_pool.full_to_swa_index_mapping

                launch_reshape_and_cache_flash(
                    k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                    v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                    k_cache.view(
                        -1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
                    ),
                    v_cache.view(
                        -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                    ),
                    forward_batch.out_cache_loc,
                    slot_mapping_swa.long() if layer.sliding_window_size > 0 else None,
                    k_scale=k_descale,
                    v_scale=v_descale,
                )
            elif self.use_triton_unified_attention and self.kv_cache_dtype == fp8_dtype:
                # [PATCH] FP8 non-SWA: use launch_reshape_and_cache_flash to
                # fuse bf16→fp8 cast + paged write in one Triton kernel,
                # eliminating separate float8_copy + store_kvcache overhead.
                # FP8 非 SWA 路径：使用单个 Triton kernel 融合 bf16→fp8 转换和分页写入
                token_to_kv_pool = forward_batch.token_to_kv_pool
                k_cache, v_cache = token_to_kv_pool.get_kv_buffer(layer.layer_id)
                launch_reshape_and_cache_flash(
                    k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                    v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                    k_cache.view(
                        -1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
                    ),
                    v_cache.view(
                        -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                    ),
                    forward_batch.out_cache_loc,
                )
            else:
                # 标准路径：直接调用 set_kv_buffer 写入 KV 缓存
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )

        if self.use_mla:
            # MLA 解码路径：从 KV 缓存获取压缩 K Buffer 进行解码注意力
            k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)

            # 从 forward_metadata 中提取 MLA PS 内核调度元数据
            work_metadata = self.forward_metadata.work_metadata
            work_indptr = self.forward_metadata.work_indptr
            work_info_set = self.forward_metadata.work_info_set

            reduce_indptr = self.forward_metadata.reduce_indptr
            reduce_final_map = self.forward_metadata.reduce_final_map
            reduce_partial_map = self.forward_metadata.reduce_partial_map

            num_kv_splits = self.forward_metadata.num_kv_splits

            # 调用 MLA 解码前向（带 head padding 对齐）
            o = self._mla_decode_fwd_with_head_pad(
                q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                k_buffer.view(-1, 1, 1, layer.qk_head_dim),
                layer,
                qo_indptr=self.forward_metadata.qo_indptr,
                kv_indptr=self.forward_metadata.kv_indptr,
                kv_indices=self.forward_metadata.kv_indices,
                kv_last_page_lens=self.forward_metadata.kv_last_page_len,
                max_seqlen_q=self.forward_metadata.max_q_len,
                sm_scale=layer.scaling,
                logit_cap=layer.logit_cap,
                work_meta_data=work_metadata,
                work_indptr=work_indptr,
                work_info_set=work_info_set,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                q_scale=k_descale,
                kv_scale=k_descale,
                intra_batch_mode=intra_batch_mode,
                num_kv_splits=num_kv_splits,
            )
        else:
            # 非 MLA 解码路径
            self.logits_soft_cap = layer.logit_cap

            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )

            # 预分配输出 tensor（与 Q 形状相同，使用模型输入精度）
            o = torch.empty_like(q, dtype=self.input_dtype)

            if self.use_triton_unified_attention:
                # FlashMLA 统一注意力（分页，支持 SWA）
                bs = forward_batch.batch_size
                window_size = (-1, -1)
                page_table = self.forward_metadata.kv_indices

                if (
                    layer.sliding_window_size is not None
                    and layer.sliding_window_size > -1
                ):
                    # 滑动窗口：设置非对称窗口大小（left = window_size - 1，right = 0）
                    window_size = (layer.sliding_window_size - 1, 0)
                    if self.forward_metadata.swa_page_table is not None:
                        page_table = self.forward_metadata.swa_page_table

                # max_kv_len：page table 中每条序列可覆盖的最大 KV token 数
                max_kv_len = page_table.shape[1] * self.page_size

                # 调用 unified_attention（FlashMLA 风格分页注意力内核）
                unified_attention(
                    q=q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    k=k_cache.view(
                        -1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
                    ),
                    v=v_cache.view(
                        -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                    ),
                    out=o.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    cu_seqlens_q=self.forward_metadata.qo_indptr,
                    seqused_k=forward_batch.seq_lens,
                    max_seqlen_q=self.forward_metadata.max_q_len,
                    max_seqlen_k=max_kv_len,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=window_size,
                    block_table=page_table,
                    softcap=0,
                    q_descale=None,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    sinks=sinks,
                )
            else:
                if self.kv_cache_dtype == fp8_dtype:
                    # FP8 KV 缓存：在分页注意力前转换为模型输入精度
                    k_cache = k_cache.to(self.input_dtype)
                    v_cache = v_cache.to(self.input_dtype)

                # 使用 paged_attention_ragged（AITER ROCm 分页注意力内核）
                paged_attention_ragged(
                    o.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    self.workspace_buffer,
                    q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    k_cache.view(-1, 1, layer.tp_k_head_num, layer.qk_head_dim),
                    v_cache.view(-1, 1, layer.tp_v_head_num, layer.v_head_dim),
                    self.scale,
                    self.forward_metadata.kv_indptr,
                    self.forward_metadata.kv_indices,
                    self.kv_last_page_len,
                    1,
                    self.max_num_partitions,
                    None,
                    "auto",
                    "NHD",
                    self.logits_soft_cap,
                    self.k_scale,
                    self.v_scale,
                    None,
                    _AITER_PARTITION_SIZE_ROCM,
                )

        return o


class AiterIndicesUpdaterPrefill:
    """预填充阶段（非 MLA）的 KV 索引更新器。

    负责在每个预填充批次中计算 kv_indptr、kv_indices、qo_indptr，
    供 AITER MHA 预填充内核（mha_batch_prefill_func）使用。
    """

    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # Parse Constants
        # 从 model_runner 解析注意力头数、head dim、数据类型等常量
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend

        # Buffers and wrappers
        # 共享 attn_backend 中预分配的 kv_indptr、kv_last_page_len、qo_indptr 缓冲区
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.qo_indptr = attn_backend.qo_indptr
        # req_to_token：请求池到 token 池的映射表
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        # 运行时将 update 绑定为 update_single_wrapper
        self.update = self.update_single_wrapper

        # 懒分配的 kv_indices 以及统计量
        self.kv_indices = None
        self.max_q_len = 0
        self.max_kv_len = 0

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
    ):
        """更新一个 wrapper（单后端）的 KV 索引。

        普通预填充路径：计算 kv_indptr、kv_indices（含 256-token 尾部 padding）以及 qo_indptr。
        投机解码路径：从 spec_info 生成预填充注意力参数。
        """
        kv_start_idx = None
        kv_indptr = self.kv_indptr
        qo_indptr = self.qo_indptr
        paged_kernel_lens = seq_lens
        paged_kernel_lens_sum = seq_lens_sum

        bs = len(req_pool_indices)
        if spec_info is None:
            # Normal extend
            # 普通预填充：计算 kv_indptr（KV 序列前缀和）
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]

            # (TODO: Kk) WA - CI test_moe_eval_accuracy_large.py
            # mha_batch_prefill reads 128 data to do computatoin
            # if real data is not long enough then original padding value 0 is used
            # but the 0 location will be made nan (noqa) in cuda graph capture mode
            # this will cause the output tensor value becomes nan
            # WA is to assure that last index of pool not changed
            # 分配 kv_indices，多分配 256 个元素防止 mha_batch_prefill 越界读取
            kv_indices = torch.empty(
                paged_kernel_lens_sum + 256,
                dtype=torch.int32,
                device=req_pool_indices.device,
            )
            # 用 Triton kernel 填充 kv_indices（扁平化 KV 物理索引）
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                kv_start_idx,
                kv_indices,
                self.req_to_token.shape[1],
            )

            # 将尾部 padding 位置的值设为合法索引（避免 nan 传播）
            token_num = kv_indptr[-1]
            kv_indices[token_num:] = kv_indices[0]

            # extend_lens：每条序列实际扩展的 token 数（seq_len - prefix_len）
            extend_lens = seq_lens - prefix_lens

            qo_indptr[1 : bs + 1] = torch.cumsum(extend_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
            custom_mask = None
        else:
            # 投机解码：通过 spec_info 生成预填充所需的 KV/QO 指针和 custom mask
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    req_pool_indices,
                    paged_kernel_lens,
                    paged_kernel_lens_sum,
                    self.req_to_token,
                )
            )

        # 将计算结果保存到实例变量供 forward_extend 使用
        self.kv_indices = kv_indices


class AiterMlaIndicesUpdaterPrefill:
    """MLA 预填充阶段的 KV 索引更新器。

    与 AiterIndicesUpdaterPrefill 类似，但专为 MLA（Multi-head Latent Attention）
    格式设计，额外维护 qo_indptr 和 kv_last_page_len，供 MLA 预填充内核使用。
    """

    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # Parse Constants
        # 保存 attn_backend 引用供 update_single_wrapper 使用
        self.attn_backend = attn_backend

        # Buffers and wrappers
        # req_to_token：请求池到 token 物理位置的映射表
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        # 运行时将 update 绑定为 update_single_wrapper
        self.update = self.update_single_wrapper

        # 懒分配的 KV/QO 指针和索引
        self.kv_indptr = None
        self.kv_indices = None
        self.qo_indptr = None
        self.kv_last_page_len = None
        self.max_q_len = 0
        self.max_kv_len = 0

    def update(
        self,
        req_pool_indices: torch.Tensor,
        kv_lens: torch.Tensor,
        kv_lens_sum: int,
        extend_lens: torch.Tensor,
        max_q_len: int,
        max_kv_len: int,
        spec_info: Optional[SpecInput],
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        kv_lens: torch.Tensor,
        kv_lens_sum: int,
        extend_lens: torch.Tensor,
        max_q_len: int,
        max_kv_len: int,
        spec_info: Optional[SpecInput],
    ):
        """更新 MLA 预填充的 KV/QO 索引。

        普通预填充：计算 kv_indptr（KV 前缀和）、kv_indices（扁平 KV 物理索引）
        和 qo_indptr（Q/O 前缀和）。
        投机解码：从 spec_info 生成对应的注意力参数。
        """
        bs = len(req_pool_indices)

        # 从 attn_backend 获取共享缓冲区
        kv_indptr = self.attn_backend.kv_indptr

        if spec_info is None:
            # Normal extend
            # 普通预填充：计算 KV 序列前缀和
            kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            # 为 KV 索引分配精确大小的缓冲区（MLA 不需要额外 padding）
            kv_indices = torch.empty(
                kv_lens_sum,
                dtype=torch.int32,
                device=req_pool_indices.device,
            )
            # 用 Triton kernel 填充 KV 扁平索引
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                kv_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            # qo_indptr：Q/O 序列前缀和（由 extend_lens 计算）
            qo_indptr = self.attn_backend.qo_indptr
            qo_indptr[1 : bs + 1] = torch.cumsum(extend_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
        else:
            # 投机解码：从 spec_info 生成预填充注意力参数
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    req_pool_indices,
                    kv_lens,
                    kv_lens_sum,
                    self.req_to_token,
                )
            )

        # 将结果保存到实例变量
        self.kv_indptr = kv_indptr
        self.kv_indices = kv_indices
        self.qo_indptr = qo_indptr
        self.max_q_len = max_q_len
        self.max_kv_len = max_kv_len


class AiterMultiStepDraftBackend:
    """将多个 AiterAttnBackend 实例组合为一个多步草稿解码后端。

    EAGLE 草稿解码在每个推理步骤中需要独立的注意力后端（每步 KV 索引不同），
    本类为 speculative_num_steps - 1 个步骤分别维护一个 AiterAttnBackend，
    并通过 common_template 统一调度 KV 索引计算和元数据更新。
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        from sglang.srt.speculative.spec_utils import generate_draft_decode_kv_indices

        self.topk = topk  # 每步保留的候选 token 数（EAGLE 的 top-k）
        self.speculative_num_steps = speculative_num_steps  # 草稿解码总步数
        self.generate_draft_decode_kv_indices = generate_draft_decode_kv_indices
        # 最大批量大小 = 请求池大小 × topk
        max_bs = model_runner.req_to_token_pool.size * self.topk
        # kv_indptr：为每一步草稿解码预分配 KV 前缀和指针 (speculative_num_steps, max_bs+1)
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        # 为每个草稿解码步骤创建独立的 AiterAttnBackend（跳过预填充）
        self.attn_backends = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                AiterAttnBackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                    topk=topk,
                )
            )
        self.max_context_len = self.attn_backends[0].max_context_len
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.device = model_runner.device
        # Cached variables for generate_draft_decode_kv_indices
        # 缓存 req_to_token 池的列宽（pool_len）和 page_size
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]
        self.page_size = model_runner.server_args.page_size

    def common_template(
        self, forward_batch: ForwardBatch, kv_indices_buffer: torch.Tensor, call_fn: int
    ):
        """多步草稿解码的公共调度模板。

        先通过 generate_draft_decode_kv_indices Triton kernel 并行计算所有步骤的 KV 索引，
        再依次调用 call_fn(i, forward_batch) 更新每个步骤的 forward_metadata。
        """
        num_seqs = forward_batch.batch_size
        # 扩展批量大小：实际参与解码的请求数 = topk × batch_size
        bs = self.topk * num_seqs
        seq_lens_sum = forward_batch.seq_lens_sum

        # 调用 Triton kernel 一次性计算所有步骤的 KV 索引和 kv_indptr
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
            triton.next_power_of_2(num_seqs),
            triton.next_power_of_2(self.speculative_num_steps),
            triton.next_power_of_2(bs),
            self.page_size,
        )

        # 对每个草稿解码步骤，将对应的 kv_indptr 和 kv_indices 写入 spec_info 并调用 call_fn
        for i in range(self.speculative_num_steps - 1):
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][
                : seq_lens_sum * self.topk + bs * (i + 1)
            ]
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """为多步草稿解码初始化每一步的前向元数据。

        为每一步分配独立的 kv_indices 缓冲区，并通过 clone 确保 spec_info 的
        kv_indptr 和 kv_indices 不被后续步骤覆盖。
        """
        # 为所有步骤预分配 KV 索引缓冲区 (speculative_num_steps, total_kv_tokens)
        kv_indices = torch.empty(
            (
                self.speculative_num_steps,
                forward_batch.batch_size * self.topk * self.max_context_len,
            ),
            dtype=torch.int32,
            device=self.device,
        )

        def call_fn(i, forward_batch):
            # 克隆 kv_indptr 和 kv_indices 防止 inplace 修改
            forward_batch.spec_info.kv_indptr = (
                forward_batch.spec_info.kv_indptr.clone()
            )
            forward_batch.spec_info.kv_indices = (
                forward_batch.spec_info.kv_indices.clone()
            )
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, kv_indices, call_fn)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """为多步草稿解码的 CUDA Graph 预分配静态 KV 索引缓冲区。"""
        # 预分配所有步骤共用的静态 KV 索引缓冲区
        self.cuda_graph_kv_indices = torch.zeros(
            (self.speculative_num_steps, max_num_tokens * self.max_context_len),
            dtype=torch.int32,
            device=self.device,
        )
        # 每个步骤的 attn_backend 共享对应切片的缓冲区
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs, max_num_tokens, kv_indices_buf=self.cuda_graph_kv_indices[i]
            )

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        """CUDA Graph 捕获阶段：初始化各步骤的静态元数据。"""
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
        """CUDA Graph 回放阶段：更新各步骤静态缓冲区内容。"""
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
