from __future__ import annotations

"""
Support different attention backends.
Now there are two backends: FlashInfer and Triton.
FlashInfer is faster and Triton is easier to customize.
Each backend supports two operators: extend (i.e. prefill with cached prefix) and decode.
"""

# 标准库导入
import logging
import os
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import torch

# SGLang 内部模块导入：调试、CUDA 图、DLLM 配置、环境变量
from sglang.kernel_api_logging import debug_kernel_api
from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.environ import envs
# 注意力后端基类和工具函数
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.utils import (
    get_int_env_var,
    is_flashinfer_available,
    is_sm100_supported,
    next_power_of_2,
)

# 类型检查时才导入，避免循环依赖
if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

# 模块级日志记录器
logger = logging.getLogger(__name__)

# 若启用 torch.compile，屏蔽 dynamo 警告并抑制错误
if envs.SGLANG_ENABLE_TORCH_COMPILE.get():
    torch._logging.set_logs(dynamo=logging.ERROR)
    torch._dynamo.config.suppress_errors = True


# 按需导入 FlashInfer 库，仅在可用时引入相关包装类
if is_flashinfer_available():
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,        # 分页 KV 缓存的批量 decode 包装器
        BatchPrefillWithPagedKVCacheWrapper,       # 分页 KV 缓存的批量 prefill 包装器
        BatchPrefillWithRaggedKVCacheWrapper,      # 不规则 KV 缓存的批量 prefill 包装器
        fast_decode_plan,                          # 快速 decode plan 函数，用于减少主机设备拷贝
    )
    from flashinfer.cascade import merge_state     # 合并多个注意力状态（用于级联注意力）


# 包装器分发类型枚举：滑动窗口 或 交叉注意力
class WrapperDispatch(Enum):
    SLIDING_WINDOW = auto()   # 滑动窗口注意力模式
    CROSS_ATTENTION = auto()  # 编码器-解码器交叉注意力模式


@dataclass
class MultiItemScoringParams:
    """多项目评分参数数据类：用于含分隔符的多项目序列的注意力计算。

    Parameters for multi-item scoring in attention computation.

    Used when processing sequences with multiple items separated by delimiters,
    where each item needs specific attention patterns that respect item boundaries.

    Attributes:
        prefix_len_ptr: A uint32 1D tensor indicating the prefix length of each prompt.
                       The tensor size is equal to the batch size.
        token_pos_in_items_ptr: A uint16 1D tensor indicating the token position of each item
                               starting from 0 (delimiter) for each item. For batch size > 1,
                               sequences are concatenated with zero padding to ensure same length.
        token_pos_in_items_len: Zero padding length for token_pos_in_items_ptr to handle
                               batch_size > 1 case. Defines the padded length for each sequence.
        max_item_len_ptr: A uint16 tensor containing the max token length of all items
                         for each prompt in the batch.

    """

    # 每个 prompt 的前缀长度指针（uint32 张量）
    prefix_len_ptr: Optional[torch.Tensor] = None
    # 每个 token 在所属 item 内的位置指针（uint16 张量）
    token_pos_in_items_ptr: Optional[torch.Tensor] = None
    # 批量处理时的零填充长度（用于对齐不同序列）
    token_pos_in_items_len: int = 0
    # 每个 prompt 中最长 item 的长度指针（uint16 张量）
    max_item_len_ptr: Optional[torch.Tensor] = None

    def is_enabled(self) -> bool:
        """检查多项目评分是否已启用（通过 prefix_len_ptr 是否为 None 判断）。"""
        return self.prefix_len_ptr is not None


@dataclass
class DecodeMetadata:
    # decode 阶段使用的 FlashInfer 包装器列表
    decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper]


@dataclass
class PrefillMetadata:
    # prefill 阶段使用的分页 KV 包装器列表
    prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper]
    # 是否使用不规则（ragged）KV 缓存路径
    use_ragged: bool
    # 当前批次是否完全没有前缀缓存（即全新序列）
    extend_no_prefix: bool
    # 多项目评分参数（可选）
    multi_item_params: Optional[MultiItemScoringParams] = None


# 全局工作空间缓冲区，跨所有 FlashInfer 包装器复用以节省显存
global_workspace_buffer = None

# 快速路径：覆盖 FlashInfer plan 函数中的 indptr，减少 host-to-device 拷贝开销
global_override_indptr_cpu = None


class FlashInferAttnBackend(AttentionBackend):
    """FlashInfer 注意力后端：封装 FlashInfer 高性能注意力 kernel，支持 prefill 和 decode。"""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,           # 是否跳过 prefill（投机解码的草稿步骤使用）
        kv_indptr_buf: Optional[torch.Tensor] = None,         # 外部传入的 KV 索引指针缓冲区
        kv_last_page_len_buf: Optional[torch.Tensor] = None,  # 外部传入的最后一页长度缓冲区
        init_new_workspace: bool = False,     # 是否为此后端独立分配新的工作空间
    ):
        super().__init__()
        # 默认使用 FA2（FlashAttention v2）作为 prefill 和 decode 后端
        self.prefill_backend = "fa2"
        self.decode_backend = "fa2"

        # Store multi-item scoring flag for efficient access
        self.enable_mis = model_runner.server_args.enable_mis

        # FIXME: remove dllm workarounds from flashinfer
        self.dllm_config = DllmConfig.from_server_args(model_runner.server_args)
        self.is_dllm_model = self.dllm_config is not None

        # Parse constants
        # 判断 decode 阶段是否应使用 Tensor Core（依赖 KV 数据类型和 head 数量比）
        self.decode_use_tensor_cores = should_use_tensor_core(
            kv_cache_dtype=model_runner.kv_cache_dtype,
            num_attention_heads=model_runner.model_config.num_attention_heads
            // get_attention_tp_size(),
            num_kv_heads=model_runner.model_config.get_num_kv_heads(
                get_attention_tp_size()
            ),
        )
        self.max_context_len = model_runner.model_config.context_len   # 最大上下文长度
        self.skip_prefill = skip_prefill
        self.is_multimodal = model_runner.model_config.is_multimodal   # 是否多模态模型
        # 滑动窗口与交叉注意力不能同时使用
        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        # 根据是否使用滑动窗口或编码器-解码器结构决定包装器数量和分发原因
        if model_runner.sliding_window_size is not None:
            self.num_wrappers = 2          # 需要两个包装器：一个滑动窗口，一个全局注意力
            self.dispatch_reason = WrapperDispatch.SLIDING_WINDOW
        elif model_runner.model_config.is_encoder_decoder:
            self.num_wrappers = 2          # 需要两个包装器：一个自注意力，一个交叉注意力
            self.dispatch_reason = WrapperDispatch.CROSS_ATTENTION
        else:
            self.num_wrappers = 1          # 标准情况只需一个包装器
            self.dispatch_reason = None

        # Qwen2/Qwen3 models require higher flashinfer workspace size
        # Qwen 系列模型需要更大的 FlashInfer 工作空间（512MB）
        if (
            "Qwen2ForCausalLM" in model_runner.model_config.hf_config.architectures
            or "Qwen3ForCausalLM" in model_runner.model_config.hf_config.architectures
            or "MiMoForCausalLM" in model_runner.model_config.hf_config.architectures
            or "Qwen3VLForConditionalGeneration"
            in model_runner.model_config.hf_config.architectures
            or "Qwen3VLMoeForConditionalGeneration"
            in model_runner.model_config.hf_config.architectures
        ):
            envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.set(512 * 1024 * 1024)

        # When deterministic inference is enabled, tensor cores should be used for decode
        # Also set split tile sizes for prefill and decode from environment variables, and disable kv split for cuda graph
        # More information can be found here: https://github.com/flashinfer-ai/flashinfer/pull/1675
        # 确定性推理模式：强制使用 Tensor Core 并设置 split tile 大小，禁用 CUDA 图 KV split
        self.enable_deterministic = (
            model_runner.server_args.enable_deterministic_inference
        )
        self.prefill_split_tile_size = None   # prefill 阶段的 tile 大小（确定性模式下从环境变量读取）
        self.decode_split_tile_size = None    # decode 阶段的 tile 大小
        self.disable_cuda_graph_kv_split = False  # 是否在 CUDA 图中禁用 KV split
        if self.enable_deterministic:
            self.decode_use_tensor_cores = True
            self.prefill_split_tile_size = get_int_env_var(
                "SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE", 4096
            )
            self.decode_split_tile_size = get_int_env_var(
                "SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE", 2048
            )
            self.disable_cuda_graph_kv_split = True
            # 确定性模式需要更大的工作空间（2GB）
            envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.set(2048 * 1024 * 1024)

        # Allocate buffers
        # 分配全局共享工作空间缓冲区（首次初始化时创建）
        global global_workspace_buffer
        if global_workspace_buffer is None:
            # different from flashinfer zero_init_global_workspace_buffer
            global_workspace_size = envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get()
            global_workspace_buffer = torch.empty(
                global_workspace_size,
                dtype=torch.uint8,
                device=model_runner.device,
            )
        if init_new_workspace:
            # 独立后端需要自己的工作空间缓冲区，不与全局共享
            self.workspace_buffer = torch.empty(
                envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),
                dtype=torch.uint8,
                device=model_runner.device,
            )
        else:
            self.workspace_buffer = global_workspace_buffer
        max_bs = model_runner.req_to_token_pool.size   # 最大批量大小
        if kv_indptr_buf is None:
            # 为每个包装器创建 KV 索引指针缓冲区（长度为 max_bs+1，int32）
            self.kv_indptr = [
                torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                )
                for _ in range(self.num_wrappers)
            ]
        else:
            assert self.num_wrappers == 1
            self.kv_indptr = [kv_indptr_buf]   # 使用外部传入的缓冲区

        if kv_last_page_len_buf is None:
            # 每个序列最后一页的有效长度，初始化为 1
            self.kv_last_page_len = torch.ones(
                (max_bs,), dtype=torch.int32, device=model_runner.device
            )
        else:
            assert self.num_wrappers == 1
            self.kv_last_page_len = kv_last_page_len_buf

        if not self.skip_prefill:
            # 为每个包装器创建 QO（query/output）索引指针缓冲区
            self.qo_indptr = [
                torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                )
                for _ in range(self.num_wrappers)
            ]

        fmha_backend = "auto"
        if is_sm100_supported():
            # Disable CUTLASS backend when piecewise cuda graph is enabled
            # due to TMA descriptor initialization issues on B200
            # B200 架构：分段 CUDA 图模式下禁用 CUTLASS 后端（TMA 初始化问题）
            if not model_runner.server_args.disable_piecewise_cuda_graph:
                logger.warning(
                    "CUTLASS backend is disabled when piecewise cuda graph is enabled "
                    "due to TMA descriptor initialization issues on B200. "
                    "Using auto backend instead for stability."
                )
            else:
                fmha_backend = "cutlass"  # B200 非分段图模式使用 CUTLASS 后端
        # 创建不规则 KV 缓存的 prefill 包装器（用于无前缀的全新序列）
        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend=fmha_backend
        )

        # Two wrappers: one for sliding window attention and one for full attention.
        # Using two wrappers is unnecessary in the current PR, but are prepared for future PRs
        # 分别为 prefill、验证阶段和 decode 阶段创建包装器列表
        self.prefill_wrappers_paged = []
        self.prefill_wrappers_verify = []
        self.decode_wrappers = []
        for _ in range(self.num_wrappers):
            if not skip_prefill:
                # 普通 prefill 的分页 KV 包装器
                self.prefill_wrappers_paged.append(
                    BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        backend=self.prefill_backend,
                    )
                )
                # 投机解码验证阶段的分页 KV 包装器
                self.prefill_wrappers_verify.append(
                    BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        backend=self.prefill_backend,
                    )
                )
            # decode 阶段分页 KV 包装器（可选择使用 Tensor Core）
            self.decode_wrappers.append(
                BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    backend=self.decode_backend,
                    use_tensor_cores=self.decode_use_tensor_cores,
                )
            )

        # Create indices updater
        # 创建 prefill 和 decode 两个阶段的 KV 索引更新器
        if not skip_prefill:
            self.indices_updater_prefill = FlashInferIndicesUpdaterPrefill(
                model_runner, self
            )  # for verify
        self.indices_updater_decode = FlashInferIndicesUpdaterDecode(model_runner, self)

        # Other metadata
        # 当前 forward 的元数据（PrefillMetadata 或 DecodeMetadata）
        self.forward_metadata: Union[PrefillMetadata, DecodeMetadata] = None

        # CUDA 图相关元数据字典（按批量大小索引）
        self.decode_cuda_graph_metadata = {}
        self.prefill_cuda_graph_metadata = {}  # For verify
        self.draft_extend_cuda_graph_metadata = {}  # For draft extend

    def _process_multi_item_scoring(
        self, forward_batch: ForwardBatch
    ) -> MultiItemScoringParams:
        """处理多项目评分所需的张量，供 FlashInfer 注意力机制使用。

        This method handles sequences containing multiple "items" separated by delimiter tokens,
        where each item needs specific attention patterns that respect item boundaries.

        The method produces four key tensors for FlashInfer:
        - prefix_len_ptr: uint32 tensor with prefix length for each prompt in batch
        - token_pos_in_items_ptr: uint16 tensor with token positions starting from 0 at delimiters
        - token_pos_in_items_len: padding length for batch processing
        - max_item_len_ptr: uint16 tensor with max item length for each prompt

        Args:
            forward_batch: The forward batch containing input sequences and delimiter info

        Returns:
            MultiItemScoringParams: The processed multi-item scoring parameters

        Examples:
            Following FlashInfer definition: for 3 items of length 3, 2, 4 respectively:
            token_pos_in_items_ptr = [0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0]

            Case 1: Single sequence
            Text: "What is the capital of France? <delim> London <delim> Paris <delim> Berlin <delim>"
            Tokens: [What, is, the, capital, of, France, ?, <delim>, London, <delim>, Paris, <delim>, Berlin, <delim>]
            Indices: [ 0,   1,  2,   3,      4,  5,     6,   7,     8,      9,     10,    11,    12,     13]
            - prefix_len_ptr: [7] (query length before first delimiter)
            - token_pos_in_items_ptr: [0, 1, 0, 1, 0, 1, 0] (delim=0, London=1, delim=0, Paris=1, delim=0, Berlin=1, delim=0)
            - token_pos_in_items_len: 7 (actual length)
            - max_item_len_ptr: [1] (max item length is 1 token - all options are single tokens)

            Case 2: Batch processing (batch_size=2)
            Sequence 1: 2 items of length 2, 1 → [0, 1, 2, 0, 1, 0] (6 elements)
            Sequence 2: 3 items of length 1, 3, 2 → [0, 1, 0, 1, 2, 3, 0, 1, 2, 0] (10 elements)
            After padding both to length 10:
            - token_pos_in_items_ptr: [0, 1, 2, 0, 1, 0, 0, 0, 0, 0,    0, 1, 0, 1, 2, 3, 0, 1, 2, 0]
            - token_pos_in_items_len: 10 (padded length for batch processing)
            - max_item_len_ptr: [2, 3] (max lengths per sequence)
        """

        # decode 阶段或未启用 MIS 时直接返回空参数
        if not self.enable_mis or forward_batch.forward_mode == ForwardMode.DECODE:
            return MultiItemScoringParams()

        # 获取预计算的分隔符位置索引
        precomputed_indices = forward_batch.multi_item_delimiter_indices
        if precomputed_indices is None:
            return MultiItemScoringParams()

        # 获取前缀缓存长度和 extend 序列长度（可能为 None）
        prefix_cache_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
        extend_seq_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        prefix_len_ptr, token_pos_in_items_ptr = [], []
        token_pos_in_items_len = 0
        device = forward_batch.input_ids.device

        # If no extend_seq_lens, treat whole batch as one sequence
        # 若无序列长度信息，将整个输入视为单一序列
        if extend_seq_lens is None or len(extend_seq_lens) <= 1:
            extend_seq_lens = [forward_batch.input_ids.size(0)]

        seq_start = 0
        for i, seq_len in enumerate(extend_seq_lens):
            seq_end = seq_start + seq_len
            delimiter_indices_cpu = precomputed_indices[i]
            # 若无分隔符，跳过此序列
            if len(delimiter_indices_cpu) == 0:
                seq_start = seq_end
                continue

            first_delim = delimiter_indices_cpu[0].item()  # CPU .item(), no GPU sync
            delimiter_indices = delimiter_indices_cpu.to(device, non_blocking=True)
            # 前缀长度 = 第一个分隔符位置 + 已缓存的前缀长度
            prefix_len = first_delim + (
                prefix_cache_lens[i] if prefix_cache_lens is not None else 0
            )
            prefix_len_ptr.append(prefix_len)

            # Compute relative positions within items using searchsorted (no GPU sync).
            #   suffix_range      = [0, 1, 2, 3, 4, ...]
            #   searchsorted      = bucket index for each position
            #   last_delim        = delimiter offset at start of current bucket
            #   pos_within_item   = suffix_range - last_delim
            # 计算从第一个分隔符起的后缀长度和相对位置
            suffix_len = seq_len - first_delim
            relative_positions = delimiter_indices - first_delim

            # 为每个后缀位置生成范围索引
            suffix_range = torch.arange(suffix_len, dtype=torch.int64, device=device)
            # searchsorted 找出每个位置属于哪个 item（桶索引）
            bucket_idx = torch.searchsorted(
                relative_positions, suffix_range, right=True
            )
            # 获取每个位置所在桶的起始分隔符偏移
            last_delim = relative_positions[torch.clamp(bucket_idx - 1, min=0)]
            # 计算每个 token 在其所属 item 内的位置（从 0 起，分隔符为 0）
            pos_within_item = suffix_range - last_delim

            token_pos_in_items_ptr.append(pos_within_item.to(torch.uint16))

            # 将 item 内位置用于更新位置编码（覆盖原始绝对位置）
            forward_batch.positions[seq_start + first_delim : seq_end] = (
                prefix_len + pos_within_item - 1
            )

            seq_start = seq_end

        # Pad token_pos_in_items_ptr for batch processing
        # 对批次中所有序列的 token_pos 进行零填充，对齐到最大长度
        if token_pos_in_items_ptr:
            token_pos_in_items_len = max(t.numel() for t in token_pos_in_items_ptr)
            token_pos_in_items_ptr = [
                torch.cat(
                    [
                        t,
                        torch.zeros(
                            token_pos_in_items_len - t.numel(),
                            dtype=torch.uint16,
                            device=device,
                        ),
                    ]
                )
                for t in token_pos_in_items_ptr
            ]

        # 若收集到的参数为空则返回空参数
        if not prefix_len_ptr or not token_pos_in_items_ptr:
            return MultiItemScoringParams()

        # 构造并返回最终多项目评分参数对象
        return MultiItemScoringParams(
            prefix_len_ptr=torch.tensor(
                prefix_len_ptr, dtype=torch.uint32, device=device
            ),
            token_pos_in_items_ptr=torch.cat(token_pos_in_items_ptr, dim=0),
            # 限制长度不超过 uint32 最大值
            token_pos_in_items_len=token_pos_in_items_len & 0xFFFFFFFF,
            # 每个序列中最长 item 的长度（uint16）
            max_item_len_ptr=torch.stack(
                [
                    t.to(torch.int32).max().to(torch.uint16)
                    for t in token_pos_in_items_ptr
                ],
                dim=0,
            ),
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """根据前向批次模式初始化 FlashInfer 所需的元数据（更新 KV 索引并准备包装器）。"""
        if forward_batch.forward_mode.is_decode_or_idle():
            # decode 模式：更新解码包装器的 KV 索引
            self.indices_updater_decode.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                decode_wrappers=self.decode_wrappers,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=forward_batch.spec_info,
                fixed_split_size=self.decode_split_tile_size,
                disable_split_kv=False,
            )
            self.forward_metadata = DecodeMetadata(self.decode_wrappers)
        elif forward_batch.forward_mode.is_draft_extend():
            # 草稿扩展模式（投机解码）：使用 prefill 包装器，不带前缀
            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.prefill_wrappers_paged,
                use_ragged=False,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=forward_batch.spec_info,
            )
            self.forward_metadata = PrefillMetadata(
                self.prefill_wrappers_paged, False, False
            )
        elif forward_batch.forward_mode.is_target_verify():
            # 目标验证模式（投机解码验证阶段）：使用验证专用包装器
            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.prefill_wrappers_verify,
                use_ragged=False,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=forward_batch.spec_info,
            )
            self.forward_metadata = PrefillMetadata(
                self.prefill_wrappers_verify, False, False
            )
        else:
            # 普通 prefill/extend 模式
            prefix_lens = forward_batch.extend_prefix_lens

            # Disable ragged wrapper and ensure prefix handling for multimodal and multi-item scoring
            if self.is_multimodal or self.enable_mis:
                # use_ragged = False: Multi-item scoring requires the paged wrapper because:
                # 1. Ragged wrapper doesn't support the specialized multi-item parameters
                #    (prefix_len_ptr, token_pos_in_items_ptr, etc.)
                # 2. Paged wrapper provides better control over attention masking needed
                #    for respecting item boundaries in multi-item sequences
                # 3. Custom masking logic conflicts with ragged wrapper's assumptions
                # 多模态或多项目评分场景必须禁用 ragged wrapper
                use_ragged = False
                extend_no_prefix = False
            else:
                # 非确定性且非分段 CUDA 图时才使用 ragged wrapper
                use_ragged = (
                    not self.enable_deterministic and not is_in_piecewise_cuda_graph()
                )
                # 判断本批次中是否所有序列都没有前缀缓存
                extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)

            # Process multi-item scoring in attention backend instead of ForwardBatch
            multi_item_params = MultiItemScoringParams()
            if self.enable_mis:
                # Use new backend-specific implementation
                # 启用 MIS 时计算多项目评分所需参数
                multi_item_params = self._process_multi_item_scoring(forward_batch)

            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                prefix_lens,
                prefill_wrappers=self.prefill_wrappers_paged,
                use_ragged=use_ragged,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=None,
                fixed_split_size=self.prefill_split_tile_size,
                multi_item_params=multi_item_params,
                cross_attention_custom_mask=forward_batch.cross_attention_custom_mask,
            )
            self.forward_metadata = PrefillMetadata(
                self.prefill_wrappers_paged,
                use_ragged,
                extend_no_prefix,
                multi_item_params,
            )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        """初始化 CUDA 图捕获所需的静态 KV 索引缓冲区。"""
        if kv_indices_buf is None:
            # 为 CUDA 图分配足够大的 KV 索引缓冲区
            cuda_graph_kv_indices = torch.zeros(
                (max_num_tokens * self.max_context_len,),
                dtype=torch.int32,
                device="cuda",
            )
        else:
            cuda_graph_kv_indices = kv_indices_buf

        # 为每个包装器创建独立的 KV 索引缓冲区（第一个复用，其余克隆）
        self.cuda_graph_kv_indices = [cuda_graph_kv_indices] + [
            cuda_graph_kv_indices.clone() for _ in range(self.num_wrappers - 1)
        ]

        # Ensure tensors are properly allocated
        # 触发一次写入以确保缓冲区真正被分配（避免 lazy allocation 问题）
        for i in range(self.num_wrappers):
            # Force allocation by performing a small operation
            if len(self.cuda_graph_kv_indices[i]) > 0:
                self.cuda_graph_kv_indices[i][0] = 0

        if not self.skip_prefill:
            # 为 prefill 的 CUDA 图准备自定义掩码和 QK/QO 索引指针缓冲区
            self.cuda_graph_custom_mask = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.uint8,
                device="cuda",
            )
            self.cuda_graph_qk_indptr = [x.clone() for x in self.kv_indptr]
            self.cuda_graph_qo_indptr = [x.clone() for x in self.kv_indptr]

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
        """在 CUDA 图捕获阶段初始化前向元数据（为指定批量大小创建专用包装器并记录到图元数据字典）。"""
        if forward_mode.is_decode_or_idle():
            # decode 阶段：为 CUDA 图创建带静态缓冲区的专用 decode 包装器
            decode_wrappers = []
            for i in range(self.num_wrappers):
                decode_wrappers.append(
                    BatchDecodeWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        backend=self.decode_backend,
                        use_cuda_graph=True,              # 启用 CUDA 图模式
                        use_tensor_cores=self.decode_use_tensor_cores,
                        paged_kv_indptr_buffer=self.kv_indptr[i][: num_tokens + 1],
                        paged_kv_indices_buffer=self.cuda_graph_kv_indices[i],
                        paged_kv_last_page_len_buffer=self.kv_last_page_len[
                            :num_tokens
                        ],
                    )
                )
            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_decode.update(
                req_pool_indices,
                seq_lens,
                seq_lens.cpu(),  # may add a little overhead in capture stage
                seq_lens_sum,
                decode_wrappers=decode_wrappers,
                encoder_lens=encoder_lens,
                spec_info=spec_info,
                fixed_split_size=None,
                disable_split_kv=self.disable_cuda_graph_kv_split,
            )
            # 保存到 CUDA 图元数据字典（按批量大小索引）
            self.decode_cuda_graph_metadata[bs] = decode_wrappers
            self.forward_metadata = DecodeMetadata(decode_wrappers)
            # 将 begin_forward 替换为 fast_decode_plan 以减少重放开销
            for i in range(self.num_wrappers):
                decode_wrappers[i].begin_forward = partial(
                    fast_decode_plan, decode_wrappers[i]
                )
        elif forward_mode.is_target_verify():
            # FlashInfer's prefill wrapper decides mask mode based on whether
            # `custom_mask_buf` is initialized (not whether a custom mask is provided).
            # For cases like DFLASH draft (ENCODER_ONLY / non-causal) we do NOT use a
            # custom mask, so we must avoid initializing `custom_mask_buf`, otherwise
            # FlashInfer will treat the (zero) buffer as a real mask and block attention.
            # 验证阶段：仅在有自定义掩码时初始化 custom_mask_buf
            use_custom_mask = (
                spec_info is not None
                and getattr(spec_info, "custom_mask", None) is not None
            )
            prefill_wrappers = []
            for i in range(self.num_wrappers):
                wrapper_kwargs = {}
                if use_custom_mask:
                    wrapper_kwargs = {
                        "custom_mask_buf": self.cuda_graph_custom_mask,
                        "mask_indptr_buf": self.cuda_graph_qk_indptr[i][: bs + 1],
                    }

                prefill_wrappers.append(
                    BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        use_cuda_graph=True,
                        backend=self.prefill_backend,
                        qo_indptr_buf=self.cuda_graph_qo_indptr[i][: bs + 1],
                        paged_kv_indptr_buf=self.kv_indptr[i][: bs + 1],
                        paged_kv_indices_buf=self.cuda_graph_kv_indices[i],
                        paged_kv_last_page_len_buf=self.kv_last_page_len[:bs],
                        **wrapper_kwargs,
                    )
                )
            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_prefill.update(
                req_pool_indices,
                seq_lens,
                seq_lens.cpu(),  # may add a little overhead in capture stage
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=prefill_wrappers,
                use_ragged=False,
                encoder_lens=encoder_lens,
                spec_info=spec_info,
            )
            self.prefill_cuda_graph_metadata[bs] = prefill_wrappers
            self.forward_metadata = PrefillMetadata(prefill_wrappers, False, False)
        elif forward_mode.is_draft_extend():
            # 草稿扩展模式的 CUDA 图捕获
            prefill_wrappers = []
            for i in range(self.num_wrappers):
                prefill_wrappers.append(
                    BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        backend=self.prefill_backend,
                        use_cuda_graph=True,
                        qo_indptr_buf=self.cuda_graph_qo_indptr[i][: bs + 1],
                        paged_kv_indptr_buf=self.kv_indptr[i][: bs + 1],
                        paged_kv_indices_buf=self.cuda_graph_kv_indices[i],
                        paged_kv_last_page_len_buf=self.kv_last_page_len[:bs],
                    )
                )

            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_prefill.update(
                req_pool_indices,
                seq_lens,
                seq_lens.cpu(),  # may add a little overhead in capture stage
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=prefill_wrappers,
                use_ragged=False,
                encoder_lens=encoder_lens,
                spec_info=spec_info,
            )
            self.prefill_cuda_graph_metadata[bs] = prefill_wrappers
            self.forward_metadata = PrefillMetadata(prefill_wrappers, False, False)
        elif forward_mode.is_dllm_extend():
            # DLLM extend 模式（分布式 LLM 扩展）
            prefill_wrappers = []
            for i in range(self.num_wrappers):
                prefill_wrappers.append(
                    BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        backend=self.prefill_backend,
                        use_cuda_graph=True,
                        qo_indptr_buf=self.cuda_graph_qo_indptr[i][: bs + 1],
                        paged_kv_indptr_buf=self.kv_indptr[i][: bs + 1],
                        paged_kv_indices_buf=self.cuda_graph_kv_indices[i],
                        paged_kv_last_page_len_buf=self.kv_last_page_len[:bs],
                    )
                )
            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_prefill.update(
                req_pool_indices,
                seq_lens,
                seq_lens.cpu(),  # may add a little overhead in capture stage
                seq_lens_sum,
                # DLLM 模式：前缀长度为总序列长度减去 block_size
                prefix_lens=seq_lens - self.dllm_config.block_size,
                prefill_wrappers=prefill_wrappers,
                use_ragged=True,
                encoder_lens=encoder_lens,
                spec_info=None,
            )
            self.prefill_cuda_graph_metadata[bs] = prefill_wrappers
            self.forward_metadata = PrefillMetadata(prefill_wrappers, True, False)
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
        """在 CUDA 图重放阶段更新 KV 索引（使用已捕获的静态包装器）。"""
        if forward_mode.is_decode_or_idle():
            # decode 模式：更新已捕获的 decode 包装器
            self.indices_updater_decode.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                decode_wrappers=self.decode_cuda_graph_metadata[bs],
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
                fixed_split_size=None,
                disable_split_kv=self.disable_cuda_graph_kv_split,
            )
        elif forward_mode.is_target_verify():
            # 验证模式：更新已捕获的验证 prefill 包装器
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.prefill_cuda_graph_metadata[bs],
                use_ragged=False,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
            )
        elif forward_mode.is_draft_extend():
            # 草稿扩展模式
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.prefill_cuda_graph_metadata[bs],
                use_ragged=False,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
            )
        elif forward_mode.is_dllm_extend():
            # DLLM extend 模式重放
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                prefix_lens=seq_lens - self.dllm_config.block_size,
                prefill_wrappers=self.prefill_cuda_graph_metadata[bs],
                use_ragged=True,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=None,
            )
        else:
            raise ValueError("Invalid forward mode")

    def get_cuda_graph_seq_len_fill_value(self):
        """返回 CUDA 图序列长度填充值（decode 阶段最小序列长度为 1）。"""
        return 1

    @debug_kernel_api
    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        """执行 prefill/extend 阶段的注意力前向计算（支持 ragged/paged 两种路径）。"""
        # 根据层索引选取对应包装器（滑动窗口/交叉注意力需要区分）
        prefill_wrapper_paged = self.forward_metadata.prefill_wrappers[
            self._get_wrapper_idx(layer)
        ]
        # 交叉注意力使用编码器的缓存位置，否则使用普通 KV 缓存位置
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        logits_soft_cap = layer.logit_cap  # 注意力 logit 软上限（防止数值过大）

        q = q.contiguous()  # 保证 q 内存连续
        if not self.forward_metadata.use_ragged:
            # 分页 KV 路径：先写入 KV 缓存，再调用 prefill 包装器
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )

            # 是否使用因果掩码：交叉注意力和纯编码器不需要因果掩码
            causal = (
                not layer.is_cross_attention
                and layer.attn_type != AttentionType.ENCODER_ONLY
            )
            o = prefill_wrapper_paged.forward(
                q.view(-1, layer.tp_q_head_num, layer.head_dim),
                forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                causal=causal,
                sm_scale=layer.scaling,
                # Disable sliding window attention for multi-item scoring:
                # - Sliding window could cut across item boundaries, breaking semantic coherence
                # - Multi-item sequences need full attention to properly handle delimiter tokens
                # - Specialized multi-item parameters (prefix_len_ptr, token_pos_in_items_ptr)
                #   provide more precise attention control than simple sliding windows
                # - Item-aware masking takes precedence over window-based masking
                # 启用多项目评分时禁用滑动窗口（-1 表示无限制），确保 item 边界不被截断
                window_left=(
                    layer.sliding_window_size
                    if not (
                        self.forward_metadata.multi_item_params
                        and self.forward_metadata.multi_item_params.is_enabled()
                    )
                    else -1
                ),
                logits_soft_cap=logits_soft_cap,
                # Must use _float to avoid device-to-host copy that breaks cuda graph capture.
                # 使用 float 标量以避免设备到主机拷贝（CUDA 图捕获兼容性）
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
            )
        else:
            # If `k`/`v` are not explicitly provided, fall back to the KV cache stored in
            # `forward_batch.token_to_kv_pool` for this layer. This enables attention over
            # previously cached context without re-materializing KV tensors (e.g., the
            # IQuestLoopCoder path uses token_to_kv_pool as the KV source).
            # 不规则 KV 路径：若未显式提供 k/v，从 KV 池中读取已缓存的值
            if k is None and v is None:
                k = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)[0]
                v = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)[1]
            causal = True
            if (
                layer.is_cross_attention
                or layer.attn_type == AttentionType.ENCODER_ONLY
            ):
                causal = False  # 交叉注意力和编码器不需要因果掩码
            if not self.is_dllm_model and layer.attn_type == AttentionType.ENCODER_ONLY:
                save_kv_cache = False  # 纯编码器层不需要保存 KV 缓存

            if self.forward_metadata.extend_no_prefix:
                # NOTE: FlashInfer currently has limitations with head_dim = 32 or other dimensions
                # The FlashInfer head_dim limitation itself is tracked here:
                # https://github.com/flashinfer-ai/flashinfer/issues/1048
                # 无前缀情况：直接使用 ragged wrapper 做 prefill
                o = self.prefill_wrapper_ragged.forward(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    k.view(-1, layer.tp_k_head_num, layer.head_dim),
                    v.view(-1, layer.tp_v_head_num, layer.head_dim),
                    causal=causal,
                    sm_scale=layer.scaling,
                    logits_soft_cap=logits_soft_cap,
                )

            else:
                # 有前缀情况：分别计算 extend 部分（ragged）和缓存前缀部分（paged），再合并状态
                o1, s1 = self.prefill_wrapper_ragged.forward_return_lse(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    k.view(-1, layer.tp_k_head_num, layer.head_dim),
                    v.view(-1, layer.tp_v_head_num, layer.head_dim),
                    causal=causal,
                    sm_scale=layer.scaling,
                    logits_soft_cap=logits_soft_cap,
                )
                # 计算对已缓存前缀的注意力（非因果，因为前缀已经在 KV 缓存中）
                o2, s2 = prefill_wrapper_paged.forward_return_lse(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                    causal=False,
                    sm_scale=layer.scaling,
                    logits_soft_cap=logits_soft_cap,
                )

                # 使用 FlashInfer 的 merge_state 合并两个注意力计算结果
                o, _ = merge_state(o1, s1, o2, s2)

            if save_kv_cache:
                # 将新计算的 K/V 写入 KV 缓存池
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        # 将输出从 (tokens, heads, head_dim) 重塑为 (tokens, heads*head_dim)
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    @debug_kernel_api
    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        """执行 decode 阶段的注意力前向计算（单步生成，使用分页 KV 缓存）。"""
        # 选取对应的 decode 包装器
        decode_wrapper = self.forward_metadata.decode_wrappers[
            self._get_wrapper_idx(layer)
        ]
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        if k is not None:
            assert v is not None
            if save_kv_cache:
                # 将当前步的 K/V 写入 KV 缓存池
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        # Call the wrapped function
        # 调用 FlashInfer decode wrapper 计算注意力输出
        o = decode_wrapper.forward(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
            # Must use _float to avoid device-to-host copy that breaks cuda graph capture.
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
        )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _get_wrapper_idx(self, layer: RadixAttention):
        """根据层属性选择对应的包装器索引（0 或 1）。"""
        if self.num_wrappers == 1:
            return 0  # 只有一个包装器时直接返回 0

        # 滑动窗口：window_size==-1 表示全局注意力，对应索引 1
        if self.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            return layer.sliding_window_size == -1
        # 交叉注意力：交叉注意力层对应索引 1
        if self.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            return layer.is_cross_attention

        raise ValueError(f"Unknown dispatch reason: {self.dispatch_reason}")


class FlashInferIndicesUpdaterDecode:
    """FlashInfer decode 阶段 KV 索引更新器：负责在每次 decode 前更新 KV 索引指针。"""

    def __init__(self, model_runner: ModelRunner, attn_backend: FlashInferAttnBackend):
        # Parse Constants
        # 计算张量并行后每个 worker 的 Q/KV head 数量
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.head_dim = model_runner.model_config.head_dim  # 每个 head 的维度
        self.data_type = model_runner.kv_cache_dtype       # KV 缓存数据类型
        self.q_data_type = model_runner.dtype              # Query 数据类型
        self.sliding_window_size = model_runner.sliding_window_size  # 滑动窗口大小
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr            # KV 索引指针缓冲区
        self.kv_last_page_len = attn_backend.kv_last_page_len  # 最后一页有效长度
        self.req_to_token = model_runner.req_to_token_pool.req_to_token  # 请求到 token 的映射表
        self.token_to_kv_pool_allocator = model_runner.token_to_kv_pool_allocator

        # Dispatch the update function
        # 根据注意力类型分发到对应的 update 实现
        if self.attn_backend.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            self.update = self.update_sliding_window
        elif self.attn_backend.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            self.update = self.update_cross_attention
        else:
            assert self.attn_backend.num_wrappers == 1
            self.update = self.update_single_wrapper

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        """单包装器 decode 更新：直接调用 call_begin_forward 更新 KV 索引。"""
        decode_wrappers = decode_wrappers or self.decode_wrappers
        self.call_begin_forward(
            decode_wrappers[0],
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            self.kv_indptr[0],
            None,         # kv_start_idx 为 None，从序列起始处开始
            spec_info,
            seq_lens_cpu,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
        )

    def update_sliding_window(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        """滑动窗口 decode 更新：包装器 0 使用窗口限制的 KV，包装器 1 使用全量 KV。"""
        assert self.sliding_window_size is not None
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # Sliding window attention
                # 限制 KV 长度不超过 sliding_window_size+1（当前 token 可见的窗口）
                paged_kernel_lens_tmp = torch.clamp(
                    seq_lens, max=self.sliding_window_size + 1
                )
                if seq_lens_cpu is not None:
                    seq_lens_cpu_tmp = torch.clamp(
                        seq_lens_cpu, max=self.sliding_window_size + 1
                    )
                    paged_kernel_lens_sum_tmp = seq_lens_cpu_tmp.sum().item()
                else:
                    paged_kernel_lens_sum_tmp = paged_kernel_lens_tmp.sum().item()
                # 计算 KV 起始索引（跳过窗口之外的旧 token）
                kv_start_idx_tmp = seq_lens - paged_kernel_lens_tmp
            else:
                # Full attention
                # 全量注意力使用完整 KV 序列
                paged_kernel_lens_tmp = seq_lens
                paged_kernel_lens_sum_tmp = seq_lens_sum
                seq_lens_cpu_tmp = seq_lens_cpu
                kv_start_idx_tmp = None

            # 判断是否使用滑动窗口专属 KV 池（SWATokenToKVPoolAllocator）
            use_sliding_window_kv_pool = wrapper_id == 0 and isinstance(
                self.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator
            )

            self.call_begin_forward(
                decode_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens_tmp,
                paged_kernel_lens_sum_tmp,
                self.kv_indptr[wrapper_id],
                kv_start_idx_tmp,
                spec_info,
                seq_lens_cpu=seq_lens_cpu_tmp,
                use_sliding_window_kv_pool=use_sliding_window_kv_pool,
            )

    def update_cross_attention(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        """交叉注意力 decode 更新：包装器 0 处理自注意力，包装器 1 处理对编码器 KV 的交叉注意力。"""
        # Cache encoder_lens on CPU to avoid GPU→CPU transfer per call
        # 提前将 encoder_lens 移到 CPU，避免每次调用都进行 GPU→CPU 传输
        encoder_lens_cpu = encoder_lens.cpu() if encoder_lens is not None else None
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # 自注意力：KV 从当前解码器序列位置（encoder_lens 之后）读取
                paged_kernel_lens = seq_lens
                kv_start_idx = encoder_lens    # 跳过编码器部分
                kv_lens_cpu = seq_lens_cpu
            else:
                # Cross-attention: attend to encoder tokens only
                # 交叉注意力：只关注编码器 token（KV 从 0 开始，长度为 encoder_lens）
                paged_kernel_lens = encoder_lens
                kv_start_idx = torch.zeros_like(encoder_lens)  # 从 0 开始
                seq_lens_sum = encoder_lens.sum().item()
                kv_lens_cpu = encoder_lens_cpu

            self.call_begin_forward(
                decode_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                seq_lens_sum,
                self.kv_indptr[wrapper_id],
                kv_start_idx,
                spec_info,
                seq_lens_cpu=kv_lens_cpu,
            )

    def call_begin_forward(
        self,
        wrapper: BatchDecodeWithPagedKVCacheWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        kv_indptr: torch.Tensor,
        kv_start_idx: torch.Tensor,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        use_sliding_window_kv_pool: bool = False,
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        """调用 FlashInfer decode wrapper 的 begin_forward，准备 KV 索引以供后续 forward 使用。"""
        if spec_info is None:
            bs = len(req_pool_indices)
            # 计算累积 KV 长度作为索引指针（indptr 格式）
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]

            if wrapper.is_cuda_graph_enabled:
                # Directly write to the cuda graph input buffer
                # CUDA 图模式：直接使用静态分配的 KV 索引缓冲区
                kv_indices = wrapper._paged_kv_indices_buf
            else:
                # 普通模式：动态分配 KV 索引缓冲区
                kv_indices = torch.empty(
                    paged_kernel_lens_sum, dtype=torch.int32, device="cuda"
                )

            # 使用 Triton kernel 填充 KV 索引（从 req_to_token 映射表中读取）
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                kv_start_idx,
                kv_indices,
                self.req_to_token.shape[1],
            )
        else:
            # 投机解码模式：直接使用 spec_info 中预计算的 KV 索引
            kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
            bs = kv_indptr.shape[0] - 1

        if use_sliding_window_kv_pool:
            # 将全量 KV 池地址转换为滑动窗口 KV 池地址
            kv_last_index = kv_indptr[-1]
            kv_indices[:kv_last_index] = (
                self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                    kv_indices[:kv_last_index]
                )
            )

        global global_override_indptr_cpu
        locally_override = False
        if seq_lens_cpu is not None and global_override_indptr_cpu is None:
            # 使用 CPU 端的 seq_lens 构建 indptr，避免 GPU→CPU 同步
            locally_override = True
            global_override_indptr_cpu = torch.empty_like(kv_indptr, device="cpu")
            global_override_indptr_cpu[0] = 0
            global_override_indptr_cpu[1 : bs + 1] = torch.cumsum(seq_lens_cpu, dim=0)

        # Check if this specific wrapper's begin_forward has been replaced with fast_decode_plan
        # by checking if it's a partial function with fast_decode_plan as the func
        # 检查该 wrapper 是否已替换为 fast_decode_plan（CUDA 图优化路径）
        wrapper_uses_fast_decode_plan = (
            hasattr(wrapper.begin_forward, "func")
            and wrapper.begin_forward.func == fast_decode_plan
        )

        if wrapper_uses_fast_decode_plan:
            # When begin_forward is replaced with fast_decode_plan, pass global_override_indptr_cpu
            # 使用 fast_decode_plan 时传入 CPU indptr 以减少 D2H 拷贝
            wrapper.begin_forward(
                kv_indptr,
                kv_indices,
                self.kv_last_page_len[:bs],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                1,                             # page_size = 1
                data_type=self.data_type,
                q_data_type=self.q_data_type,
                non_blocking=True,             # 异步传输
                fixed_split_size=fixed_split_size,
                disable_split_kv=(
                    disable_split_kv if disable_split_kv is not None else False
                ),
                global_override_indptr_cpu=global_override_indptr_cpu,
            )
        else:
            # When using original begin_forward, don't pass global_override_indptr_cpu
            # 标准 begin_forward 路径（不需要传 CPU indptr）
            wrapper.begin_forward(
                kv_indptr,
                kv_indices,
                self.kv_last_page_len[:bs],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                1,
                data_type=self.data_type,
                q_data_type=self.q_data_type,
                non_blocking=True,
                fixed_split_size=fixed_split_size,
                disable_split_kv=(
                    disable_split_kv if disable_split_kv is not None else False
                ),
            )

        # 若本次调用设置了 globally_override，调用结束后清除全局变量
        if locally_override:
            global_override_indptr_cpu = None


class FlashInferIndicesUpdaterPrefill:
    """FlashInfer prefill 阶段 KV 索引更新器：为 prefill/extend 准备 KV 索引和 QO 索引指针。"""

    def __init__(self, model_runner: ModelRunner, attn_backend: FlashInferAttnBackend):
        # Parse Constants
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
        self.kv_indptr = attn_backend.kv_indptr          # KV 索引指针列表（每个包装器一个）
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.qo_indptr = attn_backend.qo_indptr          # QO 索引指针（按 extend 长度累积）
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool_allocator = model_runner.token_to_kv_pool_allocator
        self.prefill_wrapper_ragged = attn_backend.prefill_wrapper_ragged  # 不规则 KV 包装器

        # Dispatch the update function
        if self.attn_backend.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            self.update = self.update_sliding_window
        elif self.attn_backend.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            self.update = self.update_cross_attention
        else:
            assert self.attn_backend.num_wrappers == 1
            self.update = self.update_single_wrapper

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        cross_attention_custom_mask: Optional[torch.Tensor] = None,
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        cross_attention_custom_mask: Optional[torch.Tensor] = None,
    ):
        """单包装器 prefill 更新：根据 use_ragged 决定分页 KV 长度。"""
        if use_ragged:
            # TODO: remove this device sync, we can use forward_batch.extend_prefix_lens_cpu
            # and forward_batch.extend_seq_lens_cpu
            # ragged 路径：只用前缀部分作为分页 KV 长度（extend 部分单独处理）
            paged_kernel_lens = prefix_lens
            paged_kernel_lens_sum = paged_kernel_lens.sum().item()
        else:
            # 非 ragged 路径：使用完整序列长度作为分页 KV 长度
            paged_kernel_lens = seq_lens
            paged_kernel_lens_sum = seq_lens_sum

        self.call_begin_forward(
            self.prefill_wrapper_ragged,
            prefill_wrappers[0],
            req_pool_indices,
            paged_kernel_lens,
            paged_kernel_lens_sum,
            seq_lens,
            prefix_lens,
            None,             # kv_start_idx 为 None
            self.kv_indptr[0],
            self.qo_indptr[0],
            use_ragged,
            spec_info,
            fixed_split_size=fixed_split_size,
            multi_item_params=multi_item_params,
        )

    def update_sliding_window(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        cross_attention_custom_mask: Optional[torch.Tensor] = None,
    ):
        """滑动窗口 prefill 更新：包装器 0 使用窗口限制的 KV，包装器 1 使用全量 KV。"""
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # window attention use paged only
                # 窗口注意力：KV 长度限制在 sliding_window_size + extend 部分长度内
                paged_kernel_lens = torch.minimum(
                    seq_lens,
                    torch.tensor(self.sliding_window_size) + seq_lens - prefix_lens,
                )
                paged_kernel_lens_sum = paged_kernel_lens.sum().item()
            else:
                # full attention
                paged_kernel_lens = seq_lens
                paged_kernel_lens_sum = seq_lens_sum

            # 窗口注意力的 KV 起始索引（跳过窗口之外的旧 token）
            kv_start_idx = seq_lens - paged_kernel_lens
            use_sliding_window_kv_pool = wrapper_id == 0 and isinstance(
                self.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator
            )

            self.call_begin_forward(
                self.prefill_wrapper_ragged,
                prefill_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                paged_kernel_lens_sum,
                seq_lens,
                prefix_lens,
                kv_start_idx,
                self.kv_indptr[wrapper_id],
                self.qo_indptr[wrapper_id],
                use_ragged,
                spec_info,
                use_sliding_window_kv_pool=use_sliding_window_kv_pool,
                multi_item_params=multi_item_params,
            )

    def update_cross_attention(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        cross_attention_custom_mask: Optional[torch.Tensor] = None,
    ):
        """交叉注意力 prefill 更新：包装器 0 用于自注意力，包装器 1 用于交叉注意力。"""
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # normal attention
                # 自注意力：从编码器之后的位置开始读取 KV
                paged_kernel_lens = seq_lens
                kv_start_idx = encoder_lens
                paged_kernel_lens_sum = seq_lens_sum
            else:
                # cross attention
                # 交叉注意力：KV 从 0 开始，长度为编码器序列长度
                paged_kernel_lens = encoder_lens
                kv_start_idx = torch.zeros_like(encoder_lens)
                paged_kernel_lens_sum = paged_kernel_lens.sum().item()

            self.call_begin_forward(
                self.prefill_wrapper_ragged,
                prefill_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                paged_kernel_lens_sum,
                seq_lens,
                prefix_lens,
                kv_start_idx,
                self.kv_indptr[wrapper_id],
                self.qo_indptr[wrapper_id],
                use_ragged,
                spec_info,
                multi_item_params=multi_item_params,
                # 只对交叉注意力（wrapper_id==1）传入自定义掩码
                cross_attention_custom_mask=(
                    cross_attention_custom_mask if wrapper_id == 1 else None
                ),
            )

    def call_begin_forward(
        self,
        wrapper_ragged: BatchPrefillWithRaggedKVCacheWrapper,
        wrapper_paged: BatchPrefillWithPagedKVCacheWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        kv_start_idx: torch.Tensor,
        kv_indptr: torch.Tensor,
        qo_indptr: torch.Tensor,
        use_ragged: bool,
        spec_info: Optional[SpecInput],
        use_sliding_window_kv_pool: bool = False,
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        cross_attention_custom_mask: Optional[torch.Tensor] = None,
    ):
        """调用 FlashInfer prefill wrapper 的 begin_forward，填充 KV 和 QO 索引并启动规划。"""
        bs = len(seq_lens)
        if spec_info is None:
            assert len(seq_lens) == len(req_pool_indices)
            # Normal extend
            # 计算累积的分页 KV 长度（indptr 格式）
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            # 额外分配 256 个元素的余量以防 Triton kernel 越界
            kv_indices = torch.empty(
                paged_kernel_lens_sum + 256,
                dtype=torch.int32,
                device=req_pool_indices.device,
            )
            # 使用 Triton kernel 填充 KV 索引
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                kv_start_idx,
                kv_indices,
                self.req_to_token.shape[1],
            )
            # 计算 QO 索引指针（每个请求的 extend 部分长度 = seq_len - prefix_len）
            qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]

            custom_mask = cross_attention_custom_mask
        else:
            # 投机解码模式：从 spec_info 生成 prefill 注意力参数
            assert isinstance(spec_info, SpecInput)
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    req_pool_indices,
                    paged_kernel_lens,
                    paged_kernel_lens_sum,
                    self.req_to_token,
                )
            )

        # extend part
        # 若使用 ragged wrapper，先调用其 begin_forward（处理 extend 部分）
        if use_ragged:
            wrapper_ragged.begin_forward(
                qo_indptr,
                qo_indptr,          # ragged KV 使用相同的 qo_indptr 作为 kv 指针
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                q_data_type=self.q_data_type,
            )

        if use_sliding_window_kv_pool:
            # 将 KV 索引从全量池地址转换为滑动窗口池地址
            kv_last_index = kv_indptr[-1]
            kv_indices[:kv_last_index] = (
                self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                    kv_indices[:kv_last_index]
                )
            )

        # cached part
        # Conditionally set multi-item parameters
        # 根据是否启用多项目评分选择注意力参数
        if multi_item_params is not None and multi_item_params.is_enabled():
            # Multi-item scoring is active - use specialized parameters and disable generic custom_mask
            # 启用多项目评分时使用专用参数，不使用通用自定义掩码
            use_custom_mask = None
            prefix_len_ptr = multi_item_params.prefix_len_ptr
            token_pos_in_items_ptr = multi_item_params.token_pos_in_items_ptr
            token_pos_in_items_len = multi_item_params.token_pos_in_items_len
            max_item_len_ptr = multi_item_params.max_item_len_ptr
        else:
            # No multi-item scoring - use standard parameters
            # 标准模式：使用普通自定义掩码（如有）
            use_custom_mask = custom_mask
            prefix_len_ptr = None
            token_pos_in_items_ptr = None
            token_pos_in_items_len = 0
            max_item_len_ptr = None

        # 调用分页 KV wrapper 的 begin_forward，完成注意力规划
        wrapper_paged.begin_forward(
            qo_indptr,
            kv_indptr,
            kv_indices,
            self.kv_last_page_len[:bs],
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,                          # page_size = 1
            q_data_type=self.q_data_type,
            kv_data_type=self.data_type,
            custom_mask=use_custom_mask,
            non_blocking=True,          # 异步传输
            fixed_split_size=fixed_split_size,
            # 多项目评分专用参数
            prefix_len_ptr=prefix_len_ptr,
            token_pos_in_items_ptr=token_pos_in_items_ptr,
            token_pos_in_items_len=token_pos_in_items_len,
            max_item_len_ptr=max_item_len_ptr,
        )


class FlashInferMultiStepDraftBackend:
    """
    多步草稿解码后端：将多个 FlashInfer 注意力后端封装为一个，
    支持连续多步投机解码（speculative decoding）的 draft 阶段。
    Wrap multiple flashinfer attention backends as one for multiple consecutive
    draft decoding steps.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,              # 每步保留的 top-k 候选数量
        speculative_num_steps: int,   # 投机解码的步数
    ):
        from sglang.srt.speculative.spec_utils import generate_draft_decode_kv_indices

        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.generate_draft_decode_kv_indices = generate_draft_decode_kv_indices  # Triton kernel
        self.page_size = model_runner.page_size

        # 最大批量大小 = 请求池大小 × topk
        max_bs = model_runner.req_to_token_pool.size * self.topk
        # 为每个投机解码步骤分配独立的 KV 索引指针缓冲区
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        # 最后一页长度，所有步骤和 batch 共用（初始化为 1）
        self.kv_last_page_len = torch.ones(
            (max_bs,), dtype=torch.int32, device=model_runner.device
        )
        # 为每个中间步骤（排除最后一步）创建独立的 FlashInfer 注意力后端
        self.attn_backends: List[FlashInferAttnBackend] = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                FlashInferAttnBackend(
                    model_runner,
                    skip_prefill=True,                  # 草稿步骤只需 decode
                    kv_indptr_buf=self.kv_indptr[i],    # 使用预分配的 indptr 缓冲区
                    kv_last_page_len_buf=self.kv_last_page_len,
                )
            )

        self.max_context_len = self.attn_backends[0].max_context_len

        # Cached variables for generate_draft_decode_kv_indices
        # 缓存请求到 token 映射表的列数（用于 Triton kernel 参数传递）
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]

    def common_template(
        self,
        forward_batch: ForwardBatch,
        kv_indices_buffer: torch.Tensor,
        call_fn: Callable,
    ):
        """公共模板方法：生成所有步骤的 KV 索引后，依次调用 call_fn 处理每一步。"""
        num_seqs = forward_batch.batch_size
        bs = self.topk * num_seqs   # 扩展后的批量大小
        seq_lens_sum = forward_batch.seq_lens_sum

        # 调用 Triton kernel 一次性生成所有步骤的 KV 索引
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

        # Copy the kv_indptr once to avoid multiple device-to-host copies in flashinfer's plan.
        # 一次性将所有步骤的 kv_indptr 拷贝到 CPU，避免多次 D2H 传输
        indptr_cpu_whole = self.kv_indptr[:, : bs + 1].cpu()
        global global_override_indptr_cpu

        # 逐步调用 call_fn，设置每步的 spec_info 和 global_override_indptr_cpu
        for i in range(self.speculative_num_steps - 1):
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][
                : seq_lens_sum * self.topk + bs * (i + 1)
            ]
            global_override_indptr_cpu = indptr_cpu_whole[i]
            call_fn(i, forward_batch)

        # 所有步骤完成后清除全局 override
        global_override_indptr_cpu = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """为非 CUDA 图模式初始化各步骤的前向元数据。"""
        kv_indices = torch.empty(
            (
                self.speculative_num_steps,
                forward_batch.batch_size * self.topk * self.max_context_len,
            ),
            dtype=torch.int32,
            device="cuda",
        )

        def call_fn(i, forward_batch):
            # 每步需要克隆 kv_indptr 和 kv_indices，避免后续步骤覆盖
            forward_batch.spec_info.kv_indptr = (
                forward_batch.spec_info.kv_indptr.clone()
            )
            forward_batch.spec_info.kv_indices = (
                forward_batch.spec_info.kv_indices.clone()
            )
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, kv_indices, call_fn)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """初始化 CUDA 图所需的静态 KV 索引缓冲区。"""
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
        """在 CUDA 图捕获阶段为所有步骤初始化前向元数据。"""
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
        """在 CUDA 图重放阶段为所有步骤更新前向元数据。"""
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                seq_lens_sum=-1,       # CUDA 图重放时 sum 不使用
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)


def should_use_tensor_core(
    kv_cache_dtype: torch.dtype,
    num_attention_heads: int,
    num_kv_heads: int,
) -> bool:
    """
    判断 decode 阶段注意力计算是否应使用 Tensor Core。

    Determine whether to use tensor cores for attention computation.

    Args:
        kv_cache_dtype: Data type of the KV cache
        num_attention_heads: Number of attention heads
        num_kv_heads: Number of key/value heads

    Returns:
        bool: Whether to use tensor cores
    """
    # Try to use environment variable first
    # 优先检查环境变量覆盖
    env_override = os.environ.get("SGLANG_FLASHINFER_USE_TENSOR_CORE")
    if env_override is not None:
        return env_override.lower() == "true"

    # Try to use _grouped_size_compiled_for_decode_kernels if available
    # This is for flashinfer <=0.1.6. Otherwise, there is an accuracy bug
    # 对于 FlashInfer <=0.1.6，使用内部函数检查 GQA 分组大小是否已编译支持
    try:
        from flashinfer.decode import _grouped_size_compiled_for_decode_kernels

        if not _grouped_size_compiled_for_decode_kernels(
            num_attention_heads,
            num_kv_heads,
        ):
            return True   # 未编译支持则使用 Tensor Core 路径（更通用）
        else:
            return False
    except (ImportError, AttributeError):
        pass

    # Calculate GQA group size
    # 计算 GQA 分组大小（Q head 数 / KV head 数）
    gqa_group_size = num_attention_heads // num_kv_heads

    # For Flashinfer, a GQA group size of at least 4 is needed to efficiently
    # use Tensor Cores, as it fuses the head group with the token dimension in MMA.
    # float8 类型总是使用 Tensor Core
    if kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return True
    elif kv_cache_dtype in (torch.float16, torch.half, torch.bfloat16):
        # fp16/bf16 类型：GQA 分组大小 >= 4 时才使用 Tensor Core（融合 MMA 更高效）
        return gqa_group_size >= 4
    else:
        return False
