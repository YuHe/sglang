from __future__ import annotations

"""
Support attention backend for TRTLLM MLA kernels from flashinfer.
基于 flashinfer 的 TensorRT-LLM MLA（Multi-head Latent Attention）注意力后端实现。
"""

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch
import triton
import triton.language as tl

from sglang.jit_kernel.fixup_zero_kv import fixup_zero_kv_rows
from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph
from sglang.srt.environ import envs
from sglang.srt.layers.attention.flashinfer_mla_backend import (
    FlashInferMLAAttnBackend,
    FlashInferMLAMultiStepDraftBackend,
)
from sglang.srt.layers.attention.utils import (
    concat_mla_absorb_q_general,
    create_flashmla_kv_indices_triton,
    get_num_page_per_block_flashmla,
    mla_quantize_and_rope_for_fp8,
)
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import is_flashinfer_available, is_float4_e2m1fn_x2

if is_flashinfer_available():
    import flashinfer

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

logger = logging.getLogger(__name__)

# Constants
DEFAULT_WORKSPACE_SIZE_MB = 150  # Memory workspace size in MB  工作区内存大小（MB）

# Block constraint from flashinfer requirements
# From flashinfer.decode._check_trtllm_gen_mla_shape:
#   block_num % (128 / block_size) == 0
# This imposes that the total number of blocks must be divisible by
# (128 / block_size). We capture the 128 constant here so we can
# compute the LCM with other padding constraints.
# TRT-LLM 要求总块数必须被 (128 / block_size) 整除，此常量用于计算 LCM 对齐
TRTLLM_BLOCK_CONSTRAINT = 128


@triton.jit
def pad_draft_extend_query_kernel(
    q_ptr,  # Input query tensor [total_seq_len, num_heads, head_dim]  输入 query 张量
    padded_q_ptr,  # Output padded query tensor [batch_size, max_seq_len, num_heads, head_dim]  填充后的输出
    seq_lens_q_ptr,  # Sequence lengths for each sequence [batch_size]  每个序列的实际长度
    cumsum_ptr,  # Cumulative sum of accept lengths [batch_size + 1]  长度前缀和（用于定位输入起始位置）
    batch_size,   # 批大小
    max_seq_len,  # 填充后的最大序列长度
    num_heads,    # 注意力头数
    head_dim,     # 每头的特征维度
    BLOCK_SIZE: tl.constexpr,  # 头和维度方向的处理块大小
):
    """Triton kernel for padding draft extended query tensor with parallelized head and dim processing.
    将变长的 draft extend query 张量填充为固定形状 (bs, max_seq_len, num_heads, head_dim)。
    """
    # Use 3D program IDs: (batch_seq, head_block, dim_block)
    # 三维 pid：batch_seq 轴处理 (batch_id, seq_pos) 对，另两维分别处理 head 和 dim 块
    batch_seq_pid = tl.program_id(0)
    head_pid = tl.program_id(1)
    dim_pid = tl.program_id(2)

    # 从一维 pid 解码出 batch_id 和序列内位置 seq_pos
    batch_id = batch_seq_pid // max_seq_len
    seq_pos = batch_seq_pid % max_seq_len

    if batch_id >= batch_size:
        return

    # Load accept length for this batch
    # 加载当前请求的实际序列长度，超出部分跳过（padding 区域不写入）
    seq_len = tl.load(seq_lens_q_ptr + batch_id)

    if seq_pos >= seq_len:
        return

    # Load cumulative sum to get start position in input tensor
    # 从前缀和数组获取当前请求在输入张量中的起始位置
    input_start = tl.load(cumsum_ptr + batch_id)
    input_pos = input_start + seq_pos

    # Calculate head and dim block ranges
    # 计算本块负责的 head 和 dim 范围
    head_start = head_pid * BLOCK_SIZE
    head_end = tl.minimum(head_start + BLOCK_SIZE, num_heads)
    head_mask = tl.arange(0, BLOCK_SIZE) < (head_end - head_start)

    dim_start = dim_pid * BLOCK_SIZE
    dim_end = tl.minimum(dim_start + BLOCK_SIZE, head_dim)
    dim_mask = tl.arange(0, BLOCK_SIZE) < (dim_end - dim_start)

    # Calculate input offset
    # 计算从打平输入张量读取的偏移地址
    input_offset = (
        input_pos * num_heads * head_dim
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * head_dim
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]
    )

    # Load data
    # 加载输入数据，超出边界的位置填 0
    data = tl.load(
        q_ptr + input_offset,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    # Calculate output offset
    # 计算写入填充输出张量的目标偏移地址
    output_offset = (
        batch_id * max_seq_len * num_heads * head_dim
        + seq_pos * num_heads * head_dim
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * head_dim
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]
    )

    # Store data
    # 写入目标位置
    tl.store(
        padded_q_ptr + output_offset,
        data,
        mask=head_mask[:, None] & dim_mask[None, :],
    )


@triton.jit
def unpad_draft_extend_output_kernel(
    raw_out_ptr,  # Input raw output tensor (batch_size, token_per_batch, tp_q_head_num, v_head_dim)  填充的原始输出
    output_ptr,  # Output tensor (-1, tp_q_head_num, v_head_dim)  去填充后的目标张量
    accept_length_ptr,  # Accept lengths for each sequence [batch_size]  每个请求的实际接受 token 数
    cumsum_ptr,  # Cumulative sum of accept lengths [batch_size + 1]  接受长度的前缀和
    batch_size,      # 批大小
    token_per_batch, # 填充维度的 token 数（= max_seq_len）
    tp_q_head_num,   # 张量并行下的 Query head 数
    v_head_dim,      # Value 的特征维度
    BLOCK_SIZE: tl.constexpr,  # 处理块大小
):
    """Triton kernel for unpadding draft extended output tensor with parallelized head and dim processing.
    将填充的输出 (bs, max_seq_len, num_heads, v_head_dim) 压缩回变长格式 (total_tokens, num_heads, v_head_dim)。
    """
    batch_seq_pid = tl.program_id(0)
    head_pid = tl.program_id(1)
    dim_pid = tl.program_id(2)

    # 从一维 pid 解码 batch_id 和序列内位置
    batch_id = batch_seq_pid // token_per_batch
    seq_pos = batch_seq_pid % token_per_batch

    if batch_id >= batch_size:
        return

    # Load accept length for this batch
    # 加载当前请求的实际接受 token 数
    accept_len = tl.load(accept_length_ptr + batch_id)

    # 超出实际接受长度的位置不需要写入输出
    if seq_pos >= accept_len:
        return

    # Load cumulative sum to get start position in output tensor
    # 从前缀和获取当前请求在输出张量中的起始位置
    output_start = tl.load(cumsum_ptr + batch_id)
    output_pos = output_start + seq_pos

    # Calculate head and dim block ranges
    head_start = head_pid * BLOCK_SIZE
    head_end = tl.minimum(head_start + BLOCK_SIZE, tp_q_head_num)
    head_mask = tl.arange(0, BLOCK_SIZE) < (head_end - head_start)

    dim_start = dim_pid * BLOCK_SIZE
    dim_end = tl.minimum(dim_start + BLOCK_SIZE, v_head_dim)
    dim_mask = tl.arange(0, BLOCK_SIZE) < (dim_end - dim_start)

    # Calculate input offset: (batch_id, seq_pos, head_id, dim_id)
    # 从填充的原始输出中读取数据
    input_offset = (
        batch_id * token_per_batch * tp_q_head_num * v_head_dim
        + seq_pos * tp_q_head_num * v_head_dim
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * v_head_dim
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]
    )

    # Load data
    data = tl.load(
        raw_out_ptr + input_offset,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    # 计算写入压缩输出的目标偏移
    output_offset = (
        output_pos * tp_q_head_num * v_head_dim
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * v_head_dim
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]
    )

    # Store data
    tl.store(
        output_ptr + output_offset,
        data,
        mask=head_mask[:, None] & dim_mask[None, :],
    )


def _quantize_fp8_qkv(q, k, v, layer):
    # 将 Q/K/V 量化为 FP8 格式，并返回量化后的张量及 k/v 的缩放系数
    q = q.to(torch.float8_e4m3fn)  # Q 直接转换为 FP8

    # 获取 K 的量化缩放系数，如无则为 1.0（等价于不缩放）
    k_scale = getattr(layer, "k_scale_float", None)
    if k_scale is None:
        k_scale = 1.0
    if k_scale != 1.0:
        # 使用 scaled_fp8_quant 对 K 进行有缩放的量化
        assert hasattr(layer, "k_scale"), "k_scale is not set"
        k_2d, _ = scaled_fp8_quant(
            k.reshape(-1, k.shape[-1]).contiguous(), layer.k_scale
        )
        k = k_2d.reshape(k.shape)
    else:
        # 无额外缩放时直接转换为 FP8
        k = k.to(torch.float8_e4m3fn)

    # 获取 V 的量化缩放系数
    v_scale = getattr(layer, "v_scale_float", None)
    if v_scale is None:
        v_scale = 1.0
    if v_scale != 1.0:
        # 使用 scaled_fp8_quant 对 V 进行有缩放的量化
        assert hasattr(layer, "v_scale"), "v_scale is not set"
        v_2d, _ = scaled_fp8_quant(
            v.reshape(-1, v.shape[-1]).contiguous(), layer.v_scale
        )
        v = v_2d.reshape(v.shape)
    else:
        v = v.to(torch.float8_e4m3fn)

    return q, k, v, k_scale, v_scale


# 全局共享的零初始化工作区缓冲（避免重复分配）
global_zero_init_workspace_buffer = None


@dataclass
class TRTLLMMLAPrefillMetadata:
    """Metadata for TRTLLM MLA prefill operations.
    存储 TRTLLM MLA prefill（首次填充）阶段所需的元数据。
    """
    max_seq_len: int                  # 本批次最大序列长度
    cum_seq_lens: torch.Tensor        # 累积序列长度（前缀和）
    seq_lens: torch.Tensor            # 每个请求的序列长度
    fallback_to_flashinfer_impl: bool = False  # 是否回退到 flashinfer 的实现


@dataclass
class TRTLLMMLADecodeMetadata:
    """Metadata for TRTLLM MLA decode operations.
    存储 TRTLLM MLA decode（续写解码）阶段所需的元数据。
    """
    block_kv_indices: Optional[torch.Tensor] = None  # 分块 KV 索引表
    max_seq_len_k: Optional[int] = None    # KV 序列最大长度
    max_seq_len_q: Optional[int] = None    # Query 序列最大长度（draft extend 用）
    sum_seq_lens_q: Optional[int] = None   # Query 序列长度之和
    cu_seqlens_q: Optional[torch.Tensor] = None  # Query 累积序列长度
    seq_lens_q: Optional[torch.Tensor] = None    # Query 各序列长度
    seq_lens_k: Optional[torch.Tensor] = None    # KV 各序列长度（target_verify/draft_extend 用）


class TRTLLMMLABackend(FlashInferMLAAttnBackend):
    """TRTLLM MLA attention kernel from flashinfer.
    基于 flashinfer 的 TRT-LLM MLA 注意力后端，继承自 FlashInferMLAAttnBackend。
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        q_indptr_decode_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            model_runner,
            skip_prefill,
            kv_indptr_buf,
            q_indptr_decode_buf,
        )

        config = model_runner.model_config

        # Model parameters  模型结构参数
        self.num_q_heads = config.num_attention_heads // get_attention_tp_size()
        self.num_kv_heads = config.get_num_kv_heads(get_attention_tp_size())
        self.num_local_heads = config.num_attention_heads // get_attention_tp_size()

        # MLA-specific dimensions  MLA 专属维度参数
        self.kv_lora_rank = config.kv_lora_rank           # KV LoRA 压缩维度
        self.qk_nope_head_dim = config.qk_nope_head_dim   # QK nope（无位置编码）部分维度
        self.qk_rope_head_dim = config.qk_rope_head_dim   # QK rope（位置编码）部分维度
        self.v_head_dim = config.v_head_dim               # Value 头维度
        self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim  # KV cache 总维度

        # Runtime parameters  运行时参数
        self.scaling = config.scaling        # 注意力分数缩放系数
        self.data_type = model_runner.kv_cache_dtype  # KV cache 数据类型
        self.q_data_type = model_runner.dtype         # Query 数据类型
        self.page_size = model_runner.page_size       # KV cache 分页大小
        self.req_to_token = model_runner.req_to_token_pool.req_to_token  # 请求到 token 的映射表

        # Workspace allocation  分配工作区缓冲（全局单例，避免重复分配）
        self.workspace_size = DEFAULT_WORKSPACE_SIZE_MB * 1024 * 1024
        global global_zero_init_workspace_buffer
        if global_zero_init_workspace_buffer is None:
            global_zero_init_workspace_buffer = torch.zeros(
                self.workspace_size,
                dtype=torch.uint8,
                device=model_runner.device,
            )
        self.workspace_buffer = global_zero_init_workspace_buffer

        # CUDA graph state  CUDA Graph 相关状态
        self.decode_cuda_graph_metadata = {}    # 按 bs 缓存的 decode 元数据
        self.decode_cuda_graph_kv_indices = None  # CUDA Graph 模式下的 KV 索引缓冲
        self.padded_q_buffer = None            # draft extend 填充 Q 的预分配缓冲
        self.unpad_output_buffer = None        # draft extend 去填充输出的预分配缓冲
        self.forward_prefill_metadata: Optional[TRTLLMMLAPrefillMetadata] = None
        self.forward_decode_metadata: Union[TRTLLMMLADecodeMetadata, None] = None

        # 是否禁用分块前缀缓存（影响 prefill 路径选择）
        self.disable_chunked_prefix_cache = (
            get_global_server_args().disable_chunked_prefix_cache
        )

        # 投机解码草稿 token 数量
        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens

    def _calc_padded_blocks(self, max_seq_len: int) -> int:
        """
        Calculate padded block count that satisfies both TRT-LLM and Triton constraints.
        计算满足 TRT-LLM 和 Triton 双重约束的填充块数。

        Args:
            max_seq_len: Maximum sequence length in tokens

        Returns:
            Number of blocks padded to satisfy all constraints
        """
        # 计算最少需要的块数（向上取整）
        blocks = triton.cdiv(max_seq_len, self.page_size)

        # Apply dual constraints (take LCM to satisfy both):
        # 1. TRT-LLM: block_num % (128 / page_size) == 0
        # 2. Triton: number of pages per block
        # 计算 TRT-LLM 和 Triton 各自的约束，取 LCM 同时满足两者
        trtllm_constraint = TRTLLM_BLOCK_CONSTRAINT // self.page_size
        triton_constraint = get_num_page_per_block_flashmla(self.page_size)
        constraint_lcm = math.lcm(trtllm_constraint, triton_constraint)

        if blocks % constraint_lcm != 0:
            blocks = triton.cdiv(blocks, constraint_lcm) * constraint_lcm
        return blocks

    def _create_block_kv_indices(
        self,
        batch_size: int,
        max_blocks: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create block KV indices tensor using Triton kernel.
        使用 Triton kernel 创建分块 KV 索引张量。

        Args:
            batch_size: Batch size
            max_blocks: Maximum number of blocks per sequence
            req_pool_indices: Request pool indices
            seq_lens: Sequence lengths
            device: Target device

        Returns:
            Block KV indices tensor
        """
        # 初始化为 -1（无效值），由 kernel 填入有效页索引
        block_kv_indices = torch.full(
            (batch_size, max_blocks), -1, dtype=torch.int32, device=device
        )

        # 使用 Triton kernel 将 token 索引转换为页索引
        create_flashmla_kv_indices_triton[(batch_size,)](
            self.req_to_token,
            req_pool_indices,
            seq_lens,
            None,
            block_kv_indices,
            self.req_to_token.stride(0),
            max_blocks,
            PAGED_SIZE=self.page_size,
        )

        return block_kv_indices

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        """Initialize CUDA graph state for TRTLLM MLA.
        初始化 TRTLLM MLA 的 CUDA Graph 状态，预分配所需缓冲区。
        """

        # 计算最大上下文长度对应的填充块数
        max_blocks_per_seq = self._calc_padded_blocks(self.max_context_len)

        # 预分配 KV 索引缓冲（全部初始化为 -1）
        self.decode_cuda_graph_kv_indices = torch.full(
            (max_bs, max_blocks_per_seq), -1, dtype=torch.int32, device=self.device
        )
        num_tokens_per_bs = max_num_tokens // max_bs

        if is_float4_e2m1fn_x2(self.data_type):
            # Buffer for padded query: (max_bs, max_draft_tokens, num_q_heads, v_head_dim)
            # FP4 数据类型需要特殊处理（uint8 存储）
            self.store_dtype = torch.uint8
            self.padded_q_buffer = torch.zeros(
                (max_bs, num_tokens_per_bs // 2, self.num_q_heads, self.kv_cache_dim),
                dtype=self.store_dtype,
                device=self.device,
            )

            # Buffer for unpadded output: (max_num_tokens, num_q_heads, v_head_dim)
            self.unpad_output_buffer = torch.zeros(
                (max_num_tokens // 2, self.num_q_heads, 512),
                dtype=self.store_dtype,
                device=self.device,
            )
        else:
            # Buffer for padded query: (max_bs, max_draft_tokens, num_q_heads, v_head_dim)
            # 普通数据类型的 draft extend 填充 Q 缓冲
            self.padded_q_buffer = torch.zeros(
                (max_bs, num_tokens_per_bs, self.num_q_heads, self.kv_cache_dim),
                dtype=self.data_type,
                device=self.device,
            )

            # Buffer for unpadded output: (max_num_tokens, num_q_heads, v_head_dim)
            # 输出去填充缓冲（最大宽度 512）
            self.unpad_output_buffer = torch.zeros(
                (max_num_tokens, self.num_q_heads, 512),
                dtype=self.data_type,
                device=self.device,
            )

        # 调用父类初始化其他公共 CUDA Graph 状态
        super().init_cuda_graph_state(max_bs, max_num_tokens, kv_indices_buf)

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
        """Initialize metadata for CUDA graph capture.
        为 CUDA Graph 捕获阶段初始化前向元数据。
        """

        # Delegate to parent for non-decode modes.
        # 非 decode/target_verify/draft_extend 模式委托给父类处理
        if (
            not forward_mode.is_decode_or_idle()
            and not forward_mode.is_target_verify()
            and not forward_mode.is_draft_extend(include_v2=True)
        ):
            return super().init_forward_metadata_capture_cuda_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            )

        metadata = TRTLLMMLADecodeMetadata()

        if forward_mode.is_target_verify():
            # target_verify：在序列长度中加上草稿 token 数，并保存 seq_lens_k
            seq_lens = seq_lens + self.num_draft_tokens
            metadata.seq_lens_k = torch.zeros(
                (bs,), dtype=torch.int32, device=seq_lens.device
            )
            metadata.seq_lens_k.copy_(seq_lens.to(dtype=torch.int32))
        elif forward_mode.is_draft_extend(include_v2=True):
            # draft_extend：处理变长 draft 序列，初始化 Q 序列长度和累积长度
            num_tokens_per_bs = num_tokens // bs
            metadata.max_seq_len_q = num_tokens_per_bs
            metadata.sum_seq_lens_q = num_tokens_per_bs * bs
            metadata.cu_seqlens_q = torch.arange(
                0,
                bs * num_tokens_per_bs + 1,
                num_tokens_per_bs,
                dtype=torch.int32,
                device=seq_lens.device,
            )
            metadata.seq_lens_q = torch.full(
                (bs,), num_tokens_per_bs, dtype=torch.int32, device=seq_lens.device
            )
            # NOTE(draft_extend seq_len handling):
            # forward_batch.seq_lens is the seq_lens of the prev_context + verified tokens.
            # To account for pad_draft_extend_query, we need seq_lens = prev_context + max_draft_tokens.
            # This will ensure queries align with kvs correctly when calling
            # flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla.
            # 调整 seq_lens 使 Q 与 KV 位置对齐（加上 max_draft_tokens）
            seq_lens = seq_lens - metadata.seq_lens_q + metadata.max_seq_len_q
            metadata.seq_lens_k = torch.zeros(
                (bs,), dtype=torch.int32, device=seq_lens.device
            )
            metadata.seq_lens_k.copy_(seq_lens.to(dtype=torch.int32))

        # Custom fast-path for decode/idle.
        # Capture with full width so future longer sequences are safe during replay
        # 捕获时使用最大上下文长度对应的块数，确保后续 replay 时不越界
        max_blocks_per_seq = self._calc_padded_blocks(self.max_context_len)
        block_kv_indices = self.decode_cuda_graph_kv_indices[:bs, :max_blocks_per_seq]

        # 调用 Triton kernel 填充 KV 索引
        create_flashmla_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices,
            seq_lens,
            None,
            block_kv_indices,
            self.req_to_token.stride(0),
            max_blocks_per_seq,
            PAGED_SIZE=self.page_size,
        )

        metadata.block_kv_indices = block_kv_indices
        metadata.max_seq_len_k = self.max_context_len

        # 将元数据按 bs 缓存，供 replay 时复用
        self.decode_cuda_graph_metadata[bs] = metadata
        self.forward_decode_metadata = metadata

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
        """Replay CUDA graph with new inputs.
        在 CUDA Graph replay 阶段更新元数据（使用新的序列长度）。
        """
        # Delegate to parent for non-decode modes.
        if (
            not forward_mode.is_decode_or_idle()
            and not forward_mode.is_target_verify()
            and not forward_mode.is_draft_extend(include_v2=True)
        ):
            return super().init_forward_metadata_replay_cuda_graph(
                bs,
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                encoder_lens,
                forward_mode,
                spec_info,
                seq_lens_cpu,
            )

        # 取出之前捕获时存储的元数据对象（原地修改）
        metadata = self.decode_cuda_graph_metadata[bs]

        if forward_mode.is_target_verify():
            # 更新 target_verify 模式的 KV 序列长度（加入草稿 token 数）
            seq_lens = seq_lens[:bs] + self.num_draft_tokens
            metadata.seq_lens_k.copy_(seq_lens.to(dtype=torch.int32))
            del seq_lens_sum  # not handle "num_draft_tokens" but we do not need it
        elif forward_mode.is_draft_extend(include_v2=True):
            # 更新 draft_extend 模式的各项序列长度元数据
            num_tokens_per_bs = self.num_draft_tokens
            metadata.max_seq_len_q = num_tokens_per_bs
            metadata.sum_seq_lens_q = num_tokens_per_bs * bs
            metadata.cu_seqlens_q[: bs + 1].copy_(
                torch.arange(
                    0,
                    bs * num_tokens_per_bs + 1,
                    step=num_tokens_per_bs,
                    dtype=torch.int32,
                    device=seq_lens.device,
                )
            )
            metadata.seq_lens_q[:bs].fill_(num_tokens_per_bs)
            # see NOTE(draft_extend seq_len handling)
            seq_lens = seq_lens[:bs] - metadata.seq_lens_q[:bs] + metadata.max_seq_len_q
            metadata.seq_lens_k.copy_(seq_lens.to(torch.int32))

        # Update block indices for new sequences.
        # 用新的请求序列长度重新填充 KV 索引
        create_flashmla_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices[:bs],
            seq_lens,
            None,
            metadata.block_kv_indices,
            self.req_to_token.stride(0),
            metadata.block_kv_indices.shape[1],
            PAGED_SIZE=self.page_size,
        )

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        """Get the fill value for sequence lengths in CUDA graph.
        返回 CUDA Graph 模式下序列长度的填充默认值（避免零长度序列）。
        """
        return 1

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize the metadata for a forward pass.
        根据前向传播模式（prefill/decode/target_verify/draft_extend）初始化对应的元数据。
        """
        # Delegate to parent for non-decode modes.
        if (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_target_verify()
            and not forward_batch.forward_mode.is_draft_extend(include_v2=True)
        ):
            # For extend batch with prefix length > 0, fallback to ragged kernel implemented in flashinfer MLA backend
            # when chunked prefix cache is disabled.
            # Also fallback to flashinfer MLA backend when in piecewise cuda graph, since it only supports MLA forward mode.
            # 当存在前缀且禁用分块前缀缓存时，或在分段 CUDA Graph 中，回退到 flashinfer MLA 实现
            has_prefix = any(forward_batch.extend_prefix_lens_cpu)
            fallback_to_flashinfer_impl = (
                self.disable_chunked_prefix_cache and has_prefix
            ) or is_in_piecewise_cuda_graph()
            if fallback_to_flashinfer_impl:
                super().init_forward_metadata(forward_batch)

            # 计算实际扩展长度（总序列长度 - 前缀长度）
            seq_lens = forward_batch.seq_lens - forward_batch.extend_prefix_lens
            # 构建 Q 的累积序列长度（prefill 路径需要）
            cum_seq_lens_q = torch.cat(
                (
                    torch.zeros(
                        1, dtype=torch.int32, device=forward_batch.seq_lens.device
                    ),
                    torch.cumsum(seq_lens, dim=0),
                )
            ).int()
            max_seq_len = max(forward_batch.extend_seq_lens_cpu)
            self.forward_prefill_metadata = TRTLLMMLAPrefillMetadata(
                max_seq_len,
                cum_seq_lens_q,
                seq_lens,
                fallback_to_flashinfer_impl,
            )
        elif (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend(include_v2=True)
        ):
            bs = forward_batch.batch_size
            self.forward_decode_metadata = TRTLLMMLADecodeMetadata()
            # This is necessary because the backend instance persists across forward passes,
            # and forward_prefill_metadata from a previous regular extend call could still be set.
            # target_verify/draft_extend 时清除之前的 prefill 元数据
            if (
                forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend(include_v2=True)
            ):
                self.forward_prefill_metadata = None
            # Get maximum sequence length.
            # 获取批次中最大 KV 序列长度
            if getattr(forward_batch, "seq_lens_cpu", None) is not None:
                max_seq = forward_batch.seq_lens_cpu.max().item()
            else:
                max_seq = forward_batch.seq_lens.max().item()

            seq_lens = forward_batch.seq_lens

            if forward_batch.forward_mode.is_target_verify():
                # target_verify：KV 序列长度加上草稿 token 数
                max_seq = max_seq + self.num_draft_tokens
                seq_lens = seq_lens + self.num_draft_tokens
                self.forward_decode_metadata.seq_lens_k = seq_lens.to(torch.int32)
            elif forward_batch.forward_mode.is_draft_extend(include_v2=True):
                # draft_extend：处理每个请求的变长 draft 序列
                sum_seq_lens_q = sum(forward_batch.extend_seq_lens_cpu)
                max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(
                        forward_batch.extend_seq_lens, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                # see NOTE(draft_extend seq_len handling)
                seq_lens = seq_lens - forward_batch.extend_seq_lens + max_seq_len_q

                self.forward_decode_metadata.max_seq_len_q = max_seq_len_q
                self.forward_decode_metadata.sum_seq_lens_q = sum_seq_lens_q
                self.forward_decode_metadata.cu_seqlens_q = cu_seqlens_q
                self.forward_decode_metadata.seq_lens_q = forward_batch.extend_seq_lens
                self.forward_decode_metadata.seq_lens_k = seq_lens.to(torch.int32)

            # 计算填充后的块数，创建 block KV 索引
            max_seqlen_pad = self._calc_padded_blocks(max_seq)
            block_kv_indices = self._create_block_kv_indices(
                bs,
                max_seqlen_pad,
                forward_batch.req_pool_indices,
                seq_lens,
                seq_lens.device,
            )

            self.forward_decode_metadata.block_kv_indices = block_kv_indices
            self.forward_decode_metadata.max_seq_len_k = int(max_seq)
            self.forward_decode_metadata.batch_size = bs

            # 将 decode 元数据附加到 forward_batch，供下游使用
            forward_batch.decode_trtllm_mla_metadata = self.forward_decode_metadata
        else:
            return super().init_forward_metadata(forward_batch)

    def init_mha_chunk_metadata(self, forward_batch: ForwardBatch):
        # 初始化 MHA chunk 元数据（禁用 flashinfer ragged，使用 TRTLLM 实现）
        super().init_mha_chunk_metadata(forward_batch, disable_flashinfer_ragged=True)

    def pad_draft_extend_query(
        self,
        q: torch.Tensor,
        padded_q: torch.Tensor,
        seq_lens_q: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
    ) -> torch.Tensor:
        """Pad draft extended query using Triton kernel.
        使用 Triton kernel 将变长 draft query 填充为固定形状。
        """
        batch_size = cu_seqlens_q.shape[0] - 1
        max_seq_len_q = padded_q.shape[1]
        num_heads = padded_q.shape[2]
        head_dim = padded_q.shape[3]

        # Launch Triton kernel with 3D grid for parallelized head and dim processing
        # 以 (batch_seq, head_blocks, dim_blocks) 三维 grid 并行处理
        BLOCK_SIZE = 64
        num_head_blocks = triton.cdiv(num_heads, BLOCK_SIZE)
        num_dim_blocks = triton.cdiv(head_dim, BLOCK_SIZE)
        grid = (batch_size * max_seq_len_q, num_head_blocks, num_dim_blocks)

        pad_draft_extend_query_kernel[grid](
            q_ptr=q,
            padded_q_ptr=padded_q,
            seq_lens_q_ptr=seq_lens_q,
            cumsum_ptr=cu_seqlens_q,
            batch_size=batch_size,
            max_seq_len=max_seq_len_q,
            num_heads=num_heads,
            head_dim=head_dim,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return padded_q

    def unpad_draft_extend_output(
        self,
        raw_out: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seq_lens_q: torch.Tensor,
        sum_seq_lens_q: int,
    ) -> torch.Tensor:
        """Unpad draft extended output using Triton kernel.
        使用 Triton kernel 将填充的输出张量压缩回变长格式。
        """
        # raw_out: (batch_size, token_per_batch, layer.tp_q_head_num, layer.v_head_dim)
        # 解析原始输出的形状
        batch_size = seq_lens_q.shape[0]
        token_per_batch = raw_out.shape[1]  # max_seq_len
        tp_q_head_num = raw_out.shape[2]  # num_heads
        v_head_dim = raw_out.shape[3]  # head_dim
        total_tokens = sum_seq_lens_q

        # Check if we're in CUDA graph mode (buffers are pre-allocated)
        if self.unpad_output_buffer is not None:
            # Use pre-allocated buffer for CUDA graph compatibility
            # CUDA Graph 模式：使用预分配缓冲（避免动态分配破坏 Graph）
            output = self.unpad_output_buffer[:total_tokens, :, :].to(
                dtype=raw_out.dtype
            )
        else:
            # Dynamic allocation for non-CUDA graph mode
            # 非 CUDA Graph 模式：动态分配输出张量
            output = torch.empty(
                (total_tokens, tp_q_head_num, v_head_dim),
                dtype=raw_out.dtype,
                device=raw_out.device,
            )

        # Launch Triton kernel with 3D grid for parallelized head and dim processing
        BLOCK_SIZE = 64
        num_head_blocks = triton.cdiv(tp_q_head_num, BLOCK_SIZE)
        num_dim_blocks = triton.cdiv(v_head_dim, BLOCK_SIZE)
        grid = (batch_size * token_per_batch, num_head_blocks, num_dim_blocks)

        unpad_draft_extend_output_kernel[grid](
            raw_out_ptr=raw_out,
            output_ptr=output,
            accept_length_ptr=seq_lens_q,
            cumsum_ptr=cu_seqlens_q,
            batch_size=batch_size,
            token_per_batch=token_per_batch,
            tp_q_head_num=tp_q_head_num,
            v_head_dim=v_head_dim,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output[:total_tokens, :, :]

    def forward_decode(
        self,
        q: torch.Tensor,  # q_nope  Query 的 nope（非位置编码）部分
        k: torch.Tensor,  # k_nope  Key 的 nope 部分
        v: torch.Tensor,  # not used in this backend  本后端不使用 V（MLA 直接用 KV cache）
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,  # Query 的 rope 部分
        k_rope: Optional[torch.Tensor] = None,  # Key 的 rope 部分
        cos_sin_cache: Optional[torch.Tensor] = None,  # RoPE cos/sin 缓存
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run forward for decode using TRTLLM MLA kernel.
        使用 TRT-LLM MLA kernel 执行解码阶段的注意力前向计算。
        """
        merge_query = q_rope is not None
        if self.data_type == torch.float8_e4m3fn:
            # For FP8 path, we quantize the query and rope parts and merge them into a single tensor
            # Note: rope application in deepseek_v2.py:forward_absorb_prepare is skipped for FP8 decode path of this trtllm_mla backend
            # FP8 路径：先做量化和 RoPE，将 q/k 合并为 FP8 格式
            assert all(
                x is not None for x in [q_rope, k_rope, cos_sin_cache]
            ), "For FP8 path and using flashinfer.rope.mla_rope_quantize we need all of q_rope, k_rope and cos_sin_cache to be not None."
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
            merge_query = False  # FP8 路径已在量化函数中完成合并

        # Save KV cache if requested
        # 将本次的 k/k_rope 写入 KV cache
        if save_kv_cache:
            assert (
                k is not None and k_rope is not None
            ), "For populating trtllm_mla kv cache, both k_nope and k_rope should be not None."
            forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                layer, forward_batch.out_cache_loc, k, k_rope
            )

        # Prepare query tensor inline
        if merge_query:
            # For FP16 path, we merge the query and rope parts into a single tensor
            # FP16/BF16 路径：手动拼接 q_nope 和 q_rope
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope_reshaped = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
            query = concat_mla_absorb_q_general(q_nope, q_rope_reshaped)
        else:
            # For FP8 path, we already have the query and rope parts merged because of the quantize_and_rope_for_fp8 function
            # FP8 路径：已经合并，直接 view 为 [bs, num_q_heads, head_dim]
            query = q.view(-1, layer.tp_q_head_num, layer.head_dim)

        # Apply llama 4 scaling if provided
        # 如果提供了 Llama4 特有的缩放系数，则在计算前应用
        if llama_4_scaling is not None:
            query = query.to(self.q_data_type) * llama_4_scaling
            query = query.to(self.data_type)

        # Ensure query has shape [bs, acc_q_len, num_q_heads, head_dim] when seq_len 1
        # TRT-LLM kernel 要求 query 为 4D 张量
        if query.dim() == 3:
            query = query.unsqueeze(1)

        # Prepare KV cache inline
        # 获取 KV cache 并整形为 TRT-LLM kernel 所需的 4D 格式
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(1)

        # Get metadata
        # 优先从 forward_batch 获取元数据，回退到实例变量
        metadata = (
            getattr(forward_batch, "decode_trtllm_mla_metadata", None)
            or self.forward_decode_metadata
        )

        # Ensure batch_size is sufficient, the batch size increase due to the padding from the forward batch
        # FIXME(@rainj-me), refactor the skip_attn_backend_init, init_forward_metadata for attn backends
        # and padding logic in prepare_mlp_sync_batch to avoid this
        # 如果由于 forward_batch 填充导致 batch_size 增大，重新初始化元数据
        batch_size = getattr(metadata, "batch_size", None)
        if batch_size is not None and batch_size < forward_batch.batch_size:
            self.init_forward_metadata(forward_batch)
            metadata = forward_batch.decode_trtllm_mla_metadata

        # Scale computation for TRTLLM MLA kernel BMM1 operation:
        # The final BMM1 scale is computed as: q_scale * k_scale * softmax_scale
        # Scale components:
        # - q_scale: Query scaling factor (set to 1.0 for both FP16/FP8 paths)
        # - k_scale: Key scaling factor from model checkpoint. Only applied when KV cache
        #   stores FP8-quantized values, to compensate for the quantization scaling.
        #   For BF16/FP16 KV cache, k_scale must be 1.0 since values are unscaled.
        # - softmax_scale: Attention softmax scaling = 1/sqrt(head_dim), pre-computed as layer.scaling
        # 计算 BMM1（Q*K^T）的联合缩放系数
        q_scale = 1.0
        if self.data_type == torch.float8_e4m3fn:
            # FP8 路径：从 layer 获取 k_scale（模型检查点中的量化缩放系数）
            k_scale = (
                layer.k_scale_float
                if getattr(layer, "k_scale_float", None) is not None
                else 1.0
            )
        else:
            # 非 FP8 路径：k_scale 必须为 1.0，否则警告
            if getattr(layer, "k_scale_float", None) is not None:
                logger.warning_once(
                    "Checkpoint has k_scale but KV cache dtype is not FP8. "
                    "Ignoring k_scale for BMM1 (k_scale=%.4f, kv_dtype=%s).",
                    layer.k_scale_float,
                    self.data_type,
                )
            k_scale = 1.0

        # 最终缩放系数 = q_scale * k_scale * softmax_scale
        bmm1_scale = q_scale * k_scale * layer.scaling

        # Call TRT-LLM kernel  调用 TRT-LLM decode 注意力 kernel
        raw_out = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=self.workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=metadata.block_kv_indices,
            seq_lens=forward_batch.seq_lens.to(torch.int32),
            max_seq_len=metadata.max_seq_len_k,
            bmm1_scale=bmm1_scale,
            skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
        )

        # Reshape output directly without slicing
        # 将输出 reshape 为 [total_tokens, num_q_heads * v_head_dim]
        output = raw_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)
        return output

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
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 如果需要回退到 flashinfer 实现（如有前缀且禁用分块前缀缓存），委托给父类
        if (
            self.forward_prefill_metadata is not None
            and self.forward_prefill_metadata.fallback_to_flashinfer_impl
        ):
            return super().forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, q_rope, k_rope
            )

        # TODO refactor to avoid code duplication
        merge_query = q_rope is not None
        if (
            self.data_type == torch.float8_e4m3fn
        ) and forward_batch.forward_mode.is_target_verify():
            # For FP8 path, we quantize the query and rope parts and merge them into a single tensor
            # Note: rope application in deepseek_v2.py:forward_absorb_prepare is skipped for FP8 decode path of this trtllm_mla backend
            # target_verify 的 FP8 路径：量化并合并 Q/K/rope
            assert all(
                x is not None for x in [q_rope, k_rope, cos_sin_cache]
            ), "For FP8 path and using flashinfer.rope.mla_rope_quantize we need all of q_rope, k_rope and cos_sin_cache to be not None."
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
            merge_query = False

        # Save KV cache if requested
        # 将 k/k_rope 写入 KV cache（prefill 阶段必须保存）
        if save_kv_cache:
            assert (
                k is not None and k_rope is not None
            ), "For populating trtllm_mla kv cache, both k_nope and k_rope should be not None."
            forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                layer, forward_batch.out_cache_loc, k, k_rope
            )

        # TODO refactor to avoid code duplication
        # Prepare query tensor inline
        if merge_query:
            # For FP16 path, we merge the query and rope parts into a single tensor
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope_reshaped = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
            q = concat_mla_absorb_q_general(q_nope, q_rope_reshaped)

        # 将 Q 整形为 [total_tokens, num_q_heads, head_dim]
        q = q.view(-1, layer.tp_q_head_num, layer.head_dim)

        # Apply llama 4 scaling if provided
        if llama_4_scaling is not None:
            q = q.to(self.q_data_type) * llama_4_scaling
            q = q.to(self.data_type)

        if (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend(include_v2=True)
        ):
            # target_verify / draft_extend：使用 decode kernel 处理多 token 的情况
            metadata = (
                getattr(forward_batch, "decode_trtllm_mla_metadata", None)
                or self.forward_decode_metadata
            )

            # Ensure batch_size is sufficient, the batch size increase due to the padding from the forward batch
            # FIXME(@rainj-me), refactor the skip_attn_backend_init, init_forward_metadata for attn backends
            # and padding logic in prepare_mlp_sync_batch to avoid this
            batch_size = getattr(metadata, "batch_size", None)
            if batch_size is not None and batch_size < forward_batch.batch_size:
                self.init_forward_metadata(forward_batch)
                metadata = forward_batch.decode_trtllm_mla_metadata

            # Ensure query has shape [bs, num_draft_tokens, num_q_heads, head_dim]
            bs = forward_batch.batch_size

            # 获取 KV cache 并整形为 TRT-LLM 所需的 4D 格式
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(1)

            # 计算 BMM1 缩放系数
            q_scale = 1.0
            if self.data_type == torch.float8_e4m3fn:
                k_scale = (
                    layer.k_scale_float
                    if getattr(layer, "k_scale_float", None) is not None
                    else 1.0
                )
            else:
                if getattr(layer, "k_scale_float", None) is not None:
                    logger.warning_once(
                        "Checkpoint has k_scale but KV cache dtype is not FP8. "
                        "Ignoring k_scale for BMM1 (k_scale=%.4f, kv_dtype=%s).",
                        layer.k_scale_float,
                        self.data_type,
                    )
                k_scale = 1.0
            q = q.to(self.data_type)

            bmm1_scale = q_scale * k_scale * layer.scaling
            if forward_batch.forward_mode.is_target_verify():
                # target_verify：所有序列草稿 token 数相同，直接 view
                max_seq_len = (
                    metadata.max_seq_len_k + forward_batch.spec_info.draft_token_num
                )
                # For target_verify, all sequences have the same number of draft tokens
                q = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)
                needs_unpad = False
            else:
                # draft_extend: handle varying num_accepted_drafts_per_req. If total_tokens % bs == 0,
                # we can directly reshape q; otherwise, pad to max_seq_len_q.
                # draft_extend：序列长度可能不均，需要判断是否填充
                total_tokens = q.shape[0]
                tokens_per_seq = total_tokens // bs if bs > 0 else 0
                can_direct_view = bs > 0 and (total_tokens % bs == 0)

                if can_direct_view:
                    # 长度均匀，可以直接 view
                    max_seq_len = metadata.max_seq_len_k + tokens_per_seq
                    q = q.view(bs, tokens_per_seq, layer.tp_q_head_num, layer.head_dim)
                    needs_unpad = False
                else:
                    # Varying lengths: pad q to (bs, max_seq_len_q, ...)
                    # 长度不均：填充到 max_seq_len_q
                    actual_seq_lens_q = forward_batch.extend_seq_lens
                    actual_max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                    max_seq_len = metadata.max_seq_len_k + actual_max_seq_len_q

                    actual_cu_seqlens_q = torch.nn.functional.pad(
                        torch.cumsum(actual_seq_lens_q, dim=0, dtype=torch.int32),
                        (1, 0),
                    )

                    # 使用预分配缓冲（CUDA Graph 兼容）或动态分配
                    if self.padded_q_buffer is not None:
                        padded_q = self.padded_q_buffer[
                            :bs, :actual_max_seq_len_q, :, :
                        ].to(dtype=q.dtype)
                        padded_q.zero_()
                    else:
                        padded_q = torch.zeros(
                            (
                                bs,
                                actual_max_seq_len_q,
                                layer.tp_q_head_num,
                                layer.head_dim,
                            ),
                            dtype=q.dtype,
                            device=q.device,
                        )

                    # 用 Triton kernel 填充 Q
                    q = self.pad_draft_extend_query(
                        q, padded_q, actual_seq_lens_q, actual_cu_seqlens_q
                    )
                    needs_unpad = True
                    unpad_seq_lens_q = actual_seq_lens_q
                    unpad_cu_seqlens_q = actual_cu_seqlens_q
                    unpad_sum_seq_lens_q = total_tokens

            assert kv_cache.dtype == self.data_type

            # 调用 TRT-LLM decode kernel（复用于 target_verify/draft_extend）
            raw_out = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=self.workspace_buffer,
                qk_nope_head_dim=self.qk_nope_head_dim,
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                block_tables=metadata.block_kv_indices,
                seq_lens=metadata.seq_lens_k,
                max_seq_len=max_seq_len,
                bmm1_scale=bmm1_scale,
                skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
            )

            if needs_unpad:
                # Unpad the output for draft_extend mode with varying lengths
                # Use the actual values computed during padding, not from metadata
                # draft_extend 模式（变长）：去填充，还原为打平格式
                output = self.unpad_draft_extend_output(
                    raw_out,
                    unpad_cu_seqlens_q,
                    unpad_seq_lens_q,
                    unpad_sum_seq_lens_q,
                )
                output = output.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            else:
                # 均匀长度：直接 view
                output = raw_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            return output

        # 常规 prefill 路径：将 k_rope 和 k 拼接，整形为 MHA 格式
        if k_rope is not None:
            k = torch.cat([k, k_rope], dim=-1)
        k = k.view(-1, layer.tp_k_head_num, layer.head_dim)
        v = v.view(-1, layer.tp_k_head_num, layer.v_head_dim)

        # 如果需要，对 Q/K/V 做 FP8 量化
        q_scale = k_scale = v_scale = 1.0
        if self.data_type == torch.float8_e4m3fn:
            q, k, v, k_scale, v_scale = _quantize_fp8_qkv(q, k, v, layer)

        # 构建 TRT-LLM ragged attention 的公共参数
        common_trtllm_args = {
            "query": q,
            "key": k,
            "value": v,
            "workspace_buffer": self.workspace_buffer,
            "batch_size": forward_batch.batch_size,
            "window_left": -1,
            "enable_pdl": False,
            "max_q_len": self.forward_prefill_metadata.max_seq_len,
            "bmm1_scale": q_scale * k_scale * layer.scaling,
            "bmm2_scale": v_scale,
            "cum_seq_lens_q": self.forward_prefill_metadata.cum_seq_lens,
            "skip_softmax_threshold_scale_factor": envs.SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR.get(),
        }

        # When chunked prefix cache is enabled, dispatch to different path for ragged attention.
        # 分块前缀缓存路径：调用 ragged attention 并处理 zero-KV 行修复
        if forward_batch.attn_attend_prefix_cache:
            # MHA for chunked prefix kv cache when running model with MLA
            assert forward_batch.prefix_chunk_idx is not None
            assert forward_batch.prefix_chunk_cu_seq_lens is not None
            assert q_rope is None
            assert k_rope is None
            chunk_idx = forward_batch.prefix_chunk_idx

            # 初始化输出为零（非 causal，部分 kv_len=0 的行需要保持为零）
            out = torch.zeros(
                q.shape[0],
                layer.tp_q_head_num,
                layer.v_head_dim,
                dtype=self.q_data_type,
                device=q.device,
            )
            result = flashinfer.prefill.trtllm_ragged_attention_deepseek(
                **common_trtllm_args,
                seq_lens=forward_batch.prefix_chunk_seq_lens[chunk_idx],
                max_kv_len=forward_batch.prefix_chunk_max_seq_lens[chunk_idx],
                o_sf_scale=-1.0,
                cum_seq_lens_kv=forward_batch.prefix_chunk_cu_seq_lens[chunk_idx],
                is_causal=False,
                return_lse=True,
                out=out,
            )

            # The TRT-LLM ragged attention cubin kernel does not correctly
            # handle rows with kv_len == 0: it leaves stale data in the
            # workspace softmaxStats buffer and may produce non-zero output
            # for those rows.  Fix up by forcing out=0 and lse=-inf for
            # zero-KV rows so that downstream merge_state ignores them.
            # Skip entirely when this chunk has no zero-KV rows (pure CPU
            # check, precomputed in prepare_chunked_prefix_cache_info).
            # 修复 kv_len=0 的行（TRT-LLM kernel 存在 bug，需要强制置零）
            if forward_batch.prefix_chunk_has_zero_kv[chunk_idx]:
                out_tensor, lse_tensor = result
                fixup_zero_kv_rows(
                    out_tensor,
                    lse_tensor,
                    forward_batch.prefix_chunk_seq_lens[chunk_idx],
                    self.forward_prefill_metadata.cum_seq_lens,
                    self.forward_prefill_metadata.max_seq_len,
                )

            return result
        else:
            # 常规因果 prefill 路径
            out = torch.zeros(
                q.shape[0],
                q.shape[1],
                v.shape[2],
                device=q.device,
                dtype=self.q_data_type,
            )
            return flashinfer.prefill.trtllm_ragged_attention_deepseek(
                **common_trtllm_args,
                seq_lens=self.forward_prefill_metadata.seq_lens,
                max_kv_len=self.forward_prefill_metadata.max_seq_len,
                o_sf_scale=1.0,
                cum_seq_lens_kv=self.forward_prefill_metadata.cum_seq_lens,
                is_causal=True,
                return_lse=forward_batch.mha_return_lse,
                out=out,
            )


class TRTLLMMLAMultiStepDraftBackend(FlashInferMLAMultiStepDraftBackend):
    """Multi-step draft backend for TRT-LLM MLA used by EAGLE.
    用于 EAGLE 投机解码的 TRT-LLM MLA 多步草稿后端。
    """

    def __init__(
        self, model_runner: "ModelRunner", topk: int, speculative_num_steps: int
    ):
        super().__init__(model_runner, topk, speculative_num_steps)

        # 将除最后一步外的所有步骤的注意力后端替换为 TRTLLMMLABackend
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i] = TRTLLMMLABackend(
                model_runner,
                skip_prefill=True,
                kv_indptr_buf=self.kv_indptr[i],
                q_indptr_decode_buf=self.q_indptr_decode,
            )
