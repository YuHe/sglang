import logging
from typing import Optional, Union

import torch
import triton
import triton.language as tl

# 导入注意力后端基类
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
# 导入 Mamba 卷积层的 PAD_SLOT_ID（填充槽 ID，标记无效位置）
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import PAD_SLOT_ID
# 导入 Mamba2 混合器模型
from sglang.srt.layers.attention.mamba.mamba import MambaMixer2
# 导入 Mamba2 前向元数据结构
from sglang.srt.layers.attention.mamba.mamba2_metadata import (
    ForwardMetadata,
    Mamba2Metadata,
)
# 导入融合 Mamba 状态散射内核（用于 MTP verify 后的状态更新）
from sglang.srt.layers.attention.mamba.mamba_state_scatter_triton import (
    fused_mamba_state_scatter_with_mask,
)
from sglang.srt.layers.radix_attention import RadixAttention
# 导入混合请求到 token 池（同时支持全注意力和 Mamba 状态）
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import get_global_server_args
# 导入 EAGLE 推测解码相关输入结构
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.utils import is_cpu

if not is_cpu():
    # 非 CPU 环境下导入 FLA（Flash Linear Attention）分块大小常量
    from sglang.srt.layers.attention.fla.chunk_delta_h import (
        CHUNK_SIZE as FLA_CHUNK_SIZE,
    )

logger = logging.getLogger(__name__)


# 追踪 Mamba 状态的 Triton JIT 内核（在 decode 后将状态复制到 prefix cache 槽）
@triton.jit
def track_mamba_state_if_needed_kernel(
    conv_states_ptr,
    ssm_states_ptr,
    cache_indices_ptr,
    mamba_track_mask_ptr,
    mamba_track_indices_ptr,
    conv_state_stride_0,  # stride for first dimension (batch/pool index)
    ssm_state_stride_0,  # stride for first dimension (batch/pool index)
    conv_state_numel_per_row: tl.constexpr,  # total elements per row
    ssm_state_numel_per_row: tl.constexpr,  # total elements per row
    BLOCK_SIZE: tl.constexpr,
):
    """
    Track conv_states and ssm_states rows based on track mask.

    This kernel replaces a Python loop that copies state tensors for mamba attention.
    For each batch element, if the track mask is True, it copies the entire row from
    the source index (cache_indices[i]) to the destination index (mamba_track_indices[i]).

    Grid: (batch_size,)
    Each block handles one batch element, using multiple threads to copy data in parallel.
    """
    # 每个 block 处理一个 batch 元素（program_id 对应 batch 索引）
    batch_idx = tl.program_id(0)

    # Load the copy mask for this batch element
    # 读取当前 batch 元素的追踪掩码
    track_mask = tl.load(mamba_track_mask_ptr + batch_idx)

    # Early exit if we don't need to track
    # 掩码为 False 时提前退出，无需复制状态
    if not track_mask:
        return

    # Load source and destination indices
    # 读取源槽和目标槽索引
    src_idx = tl.load(cache_indices_ptr + batch_idx)
    dst_idx = tl.load(mamba_track_indices_ptr + batch_idx)

    # Copy conv_states
    # Each thread handles BLOCK_SIZE elements
    # 并行复制 conv_states（每个线程处理 BLOCK_SIZE 个元素）
    for offset in range(0, conv_state_numel_per_row, BLOCK_SIZE):
        element_indices = offset + tl.arange(0, BLOCK_SIZE)
        mask = element_indices < conv_state_numel_per_row

        src_ptr = conv_states_ptr + src_idx * conv_state_stride_0 + element_indices
        dst_ptr = conv_states_ptr + dst_idx * conv_state_stride_0 + element_indices

        data = tl.load(src_ptr, mask=mask, other=0.0)
        tl.store(dst_ptr, data, mask=mask)

    # Copy ssm_states
    # 并行复制 ssm_states（结构同 conv_states 复制）
    for offset in range(0, ssm_state_numel_per_row, BLOCK_SIZE):
        element_indices = offset + tl.arange(0, BLOCK_SIZE)
        mask = element_indices < ssm_state_numel_per_row

        src_ptr = ssm_states_ptr + src_idx * ssm_state_stride_0 + element_indices
        dst_ptr = ssm_states_ptr + dst_idx * ssm_state_stride_0 + element_indices

        data = tl.load(src_ptr, mask=mask, other=0.0)
        tl.store(dst_ptr, data, mask=mask)


def track_mamba_states_if_needed(
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    mamba_track_mask: torch.Tensor,
    mamba_track_indices: torch.Tensor,
    batch_size: int,
):
    """
    Track mamba states using Triton kernel for better performance.

    Args:
        conv_states: Convolution states tensor [pool_size, ...]
        ssm_states: SSM states tensor [pool_size, ...]
        cache_indices: Source indices for each batch element [batch_size]
        mamba_track_mask: Boolean mask indicating which elements to track [batch_size]
        mamba_track_indices: Indices to track for each batch element [batch_size]
        batch_size: Number of batch elements
    """
    # 计算每行的元素数量
    conv_state_numel_per_row = conv_states[0].numel()
    ssm_state_numel_per_row = ssm_states[0].numel()

    # Choose BLOCK_SIZE based on the size of the data
    # 固定 BLOCK_SIZE=1024，适合大多数状态尺寸
    BLOCK_SIZE = 1024

    # Launch kernel with batch_size blocks
    # 以 batch_size 为 grid 启动内核，每个 block 对应一个序列
    grid = (batch_size,)
    track_mamba_state_if_needed_kernel[grid](
        conv_states,
        ssm_states,
        cache_indices,
        mamba_track_mask,
        mamba_track_indices,
        conv_states.stride(0),
        ssm_states.stride(0),
        conv_state_numel_per_row,
        ssm_state_numel_per_row,
        BLOCK_SIZE,
    )



class MambaAttnBackendBase(AttentionBackend):
    # Mamba 注意力后端基类：管理 conv_states/ssm_states 的 CUDA Graph 缓冲区和前向元数据
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.pad_slot_id = PAD_SLOT_ID  # 无效填充槽 ID
        self.device = model_runner.device
        self.topk = model_runner.server_args.speculative_eagle_topk or 0  # EAGLE topk 参数
        self.req_to_token_pool: HybridReqToTokenPool = model_runner.req_to_token_pool  # 混合请求池
        self.forward_metadata: ForwardMetadata = None  # 前向元数据（每次 forward 前更新）
        # 以下列表按 bs 大小预分配 CUDA Graph 所需的固定缓冲区
        self.state_indices_list = []
        self.query_start_loc_list = []
        self.retrieve_next_token_list = []
        self.retrieve_next_sibling_list = []
        self.retrieve_parent_token_list = []
        self.cached_cuda_graph_decode_query_start_loc: torch.Tensor = None  # decode 模式查询起始位置缓存
        self.cached_cuda_graph_verify_query_start_loc: torch.Tensor = None  # verify 模式查询起始位置缓存
        self.conv_states_shape: tuple[int, int] = None  # 卷积状态形状 (pool_size, conv_state_len)

    def _forward_metadata(self, forward_batch: ForwardBatch):
        # 根据前向模式构建 ForwardMetadata，支持 decode/extend/verify 等场景
        bs = forward_batch.batch_size

        # 初始化各类追踪辅助张量为 None
        retrieve_next_token = None
        retrieve_next_sibling = None
        retrieve_parent_token = None
        track_conv_indices = None
        track_ssm_h_src = None
        track_ssm_h_dst = None
        track_ssm_final_src = None
        track_ssm_final_dst = None

        # 获取当前 batch 每个序列对应的 Mamba 缓存槽索引
        mamba_cache_indices = self.req_to_token_pool.get_mamba_indices(
            forward_batch.req_pool_indices
        )

        if forward_batch.forward_mode.is_decode_or_idle():
            # decode/idle：query_start_loc 为 [0,1,2,...,bs]，每序列 1 个 token
            query_start_loc = torch.arange(
                0, bs + 1, dtype=torch.int32, device=self.device
            )
        elif forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
            if forward_batch.forward_mode.is_draft_extend_v2():
                # DRAFT_EXTEND_V2 模式：草稿模型只运行全注意力层，Mamba 层跳过，元数据为 None
                # HybridLinearAttnBackend.init_forward_metadata calls all sub-backends
                # unconditionally, but DRAFT_EXTEND_V2 only runs full-attn layers in
                # the draft model, so mamba metadata can be skipped.
                query_start_loc = None
            elif forward_batch.forward_mode.is_target_verify():
                # TARGET_VERIFY 模式：每条序列对应 draft_token_num 个 token
                query_start_loc = torch.arange(
                    0,
                    forward_batch.input_ids.shape[0] + 1,
                    step=forward_batch.spec_info.draft_token_num,
                    dtype=torch.int32,
                    device=forward_batch.input_ids.device,
                )

                if self.topk > 1:
                    # EAGLE 多分支：需要追踪 next_token 树结构
                    retrieve_next_token = forward_batch.spec_info.retrieve_next_token
                    retrieve_next_sibling = (
                        forward_batch.spec_info.retrieve_next_sibling
                    )
                    # retrieve_next_token is None during dummy run so skip tensor creation
                    # dummy run 时跳过父 token 张量创建
                    if retrieve_next_token is not None:
                        retrieve_parent_token = torch.empty_like(retrieve_next_token)
            else:
                # 标准 extend 模式：query_start_loc 基于 extend_start_loc 构建
                query_start_loc = torch.empty(
                    (bs + 1,), dtype=torch.int32, device=self.device
                )
                query_start_loc[:bs] = forward_batch.extend_start_loc
                query_start_loc[bs] = (
                    forward_batch.extend_start_loc[-1]
                    + forward_batch.extend_seq_lens[-1]
                )
                if (
                    forward_batch.mamba_track_mask is not None
                    and forward_batch.mamba_track_mask.any()
                ):
                    # 存在需要追踪的序列：计算卷积状态和 SSM 状态的源/目标索引
                    track_conv_indices = self._init_track_conv_indices(
                        query_start_loc, forward_batch
                    )

                    (
                        track_ssm_h_src,
                        track_ssm_h_dst,
                        track_ssm_final_src,
                        track_ssm_final_dst,
                    ) = self._init_track_ssm_indices(mamba_cache_indices, forward_batch)
        else:
            raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode=}")

        # 判断是否存在有效的 Mamba 追踪掩码
        has_mamba_track_mask = bool(
            forward_batch.mamba_track_mask is not None
            and forward_batch.mamba_track_mask.any()
        )

        return ForwardMetadata(
            query_start_loc=query_start_loc,
            mamba_cache_indices=mamba_cache_indices,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_parent_token=retrieve_parent_token,
            track_conv_indices=track_conv_indices,
            track_ssm_h_src=track_ssm_h_src,
            track_ssm_h_dst=track_ssm_h_dst,
            track_ssm_final_src=track_ssm_final_src,
            track_ssm_final_dst=track_ssm_final_dst,
            has_mamba_track_mask=has_mamba_track_mask,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # 直接调用 _forward_metadata 构建并存储前向元数据
        self.forward_metadata = self._forward_metadata(forward_batch)

    def _init_track_conv_indices(
        self, query_start_loc: torch.Tensor, forward_batch: ForwardBatch
    ):
        """
        Compute indices for extracting conv states from the input sequence during extend.

        In Mamba models, the conv layer maintains a sliding window of recent inputs.
        After processing a prefill chunk, we need to save the last `conv_state_len` tokens
        of the processed region for prefix caching.

        The key insight is that FLA (Flash Linear Attention) processes sequences in chunks
        of FLA_CHUNK_SIZE. We only track the conv state up to the last complete chunk boundary
        (aligned_len).

        start_indices is the starting token index of the conv state to track in this extend batch.
        indices include all pos to track in this extend batch, conv_state_len for each req that
        needs to be tracked (i.e. mamba_track_mask is True)

        Returns:
            indices: Tensor of shape [num_tracked_requests, conv_state_len] containing
                     flattened positions into the packed input tensor.
        """
        # 卷积状态窗口长度（即 conv_state 最后一维）
        conv_state_len = self.conv_states_shape[-1]

        # Calculate the end position of the last aligned chunk
        # 计算需要追踪的 token 长度（新 extend 部分，不含前缀）
        lens_to_track = (
            forward_batch.mamba_track_seqlens - forward_batch.extend_prefix_lens
        )
        mamba_cache_chunk_size = get_global_server_args().mamba_cache_chunk_size
        # 向下对齐到 chunk 边界，只追踪完整 chunk 内的状态
        aligned_len = (lens_to_track // mamba_cache_chunk_size) * mamba_cache_chunk_size
        start_indices = query_start_loc[:-1] + aligned_len - conv_state_len
        start_indices = start_indices[forward_batch.mamba_track_mask]

        # Create indices: [batch_size, conv_state_len]
        # 构建完整的 conv 状态索引：[追踪请求数, conv_state_len]
        indices = start_indices.unsqueeze(-1) + torch.arange(
            conv_state_len,
            device=self.device,
            dtype=start_indices.dtype,
        )

        # 确保索引不越界（clamp 到有效 token 范围）
        return indices.clamp(0, query_start_loc[-1] - 1)

    def _init_track_ssm_indices(
        self, mamba_cache_indices: torch.Tensor, forward_batch: ForwardBatch
    ):
        """
        Compute source and destination indices for tracking SSM states for prefix caching.

        After processing a prefill, we need to save the SSM recurrent state for prefix caching.
        The FLA kernel outputs intermediate hidden states `h` at each chunk boundary,
        plus a `last_recurrent_state` at the end of the chunked prefill size.

        The challenge is that sequences may or may not end on a chunk boundary:
          - Aligned case (len % FLA_CHUNK_SIZE == 0): In this case, FLA will store the to-cache
            state in the last_recurrent_state.
          - Unaligned case (len % FLA_CHUNK_SIZE != 0): The last_recurrent_state includes the
            unaligned position, but we only want state up to the last chunk boundary.
            We must extract from the intermediate `h` tensor at the appropriate chunk index.

        We compute the src and dst indices for all requests that need to be cached
        (i.e. mamba_track_mask is True) based on the rule above.

        For example:
        1. If chunked prefill length is < 64, then only final state has value. In this case we
           cache `final` state.
        2. if chunked prefill length == 64, then only final state has value. In this case we
           cache pos 64, from `final` state
        3. if chunked prefill length >64 and < 128, then both h and final state have value.
           We cache pos 64 from `h` state
        4. if chunked prefill length ==128, then both h and final state have value. We cache
           pos 128 from `final` state. Note `h` doesn't include the pos 128.

        Returns:
            track_ssm_h_src: Source indices into the packed `h` tensor (for unaligned seqs)
            track_ssm_h_dst: Destination cache slot indices (for unaligned seqs)
            track_ssm_final_src: Source indices into last_recurrent_state buffer (for aligned seqs)
            track_ssm_final_dst: Destination cache slot indices (for aligned seqs)
        """
        # Move to CPU to avoid kernel launches for masking operations
        # 所有索引计算在 CPU 完成，避免额外的 GPU 内核启动
        mamba_track_mask = forward_batch.mamba_track_mask.cpu()
        extend_seq_lens = forward_batch.extend_seq_lens.cpu()
        mamba_track_indices = forward_batch.mamba_track_indices.cpu()
        mamba_cache_indices = mamba_cache_indices.cpu()
        mamba_track_seqlens = forward_batch.mamba_track_seqlens.cpu()
        prefix_lens = forward_batch.extend_prefix_lens.cpu()

        # Calculate the number of hidden states per request
        # 每个序列 extend 部分对应的中间隐状态数（向上取整到 chunk 数）
        num_h_states = (extend_seq_lens - 1) // FLA_CHUNK_SIZE + 1

        # Calculate the starting offset for each sequence in the packed batch
        # 计算打包批次中每个序列的 h 状态起始偏移
        track_ssm_src_offset = torch.zeros_like(num_h_states)
        track_ssm_src_offset[1:] = torch.cumsum(num_h_states[:-1], dim=0)

        # Filter variables by track mask
        # 用追踪掩码过滤只需追踪的序列
        lens_to_track = mamba_track_seqlens - prefix_lens
        lens_masked = lens_to_track[mamba_track_mask]
        offset_masked = track_ssm_src_offset[mamba_track_mask]
        dst_masked = mamba_track_indices[mamba_track_mask]

        # Determine if the sequence ends at a chunk boundary
        # 判断各序列是否刚好对齐到 chunk 边界
        is_aligned = (lens_masked % FLA_CHUNK_SIZE) == 0

        # Case 1: Aligned. Use last_recurrent_state from ssm_states.
        # 对齐情况：直接从 ssm_states（final_state）复制
        track_ssm_final_src = mamba_cache_indices[mamba_track_mask][is_aligned]
        track_ssm_final_dst = dst_masked[is_aligned]

        # Case 2: Unaligned. Use intermediate state from h.
        # TODO: if support FLA_CHUNK_SIZE % page size != 0, then need to modify this
        # 非对齐情况：从中间 h 张量取对应 chunk 的隐状态
        not_aligned = ~is_aligned
        track_ssm_h_src = offset_masked[not_aligned] + (
            lens_masked[not_aligned] // FLA_CHUNK_SIZE
        )
        track_ssm_h_dst = dst_masked[not_aligned]

        # Move back to GPU
        # 将索引移回 GPU 以供后续内核使用
        return (
            track_ssm_h_src.to(self.device, non_blocking=True),
            track_ssm_h_dst.to(self.device, non_blocking=True),
            track_ssm_final_src.to(self.device, non_blocking=True),
            track_ssm_final_dst.to(self.device, non_blocking=True),
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        # CUDA Graph 捕获阶段：委托给 _capture_metadata 构建固定大小的元数据
        self.forward_metadata = self._capture_metadata(
            bs, req_pool_indices, forward_mode, spec_info
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        # CUDA Graph 回放阶段：委托给 _replay_metadata 更新动态内容
        self.forward_metadata = self._replay_metadata(
            bs, req_pool_indices, forward_mode, spec_info, seq_lens_cpu
        )

    def init_forward_metadata_capture_cpu_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        # CPU Graph 捕获阶段：与 CUDA Graph 捕获共用同一 _capture_metadata 逻辑
        self.forward_metadata = self._capture_metadata(
            bs, req_pool_indices, forward_mode, spec_info
        )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        # 预分配 CUDA Graph 所需的固定大小缓冲区（按 max_bs 构建列表）
        assert (
            max_num_tokens % max_bs == 0
        ), f"max_num_tokens={max_num_tokens} must be divisible by max_bs={max_bs}"
        draft_token_num = max_num_tokens // max_bs  # 每序列的推测 token 数
        for i in range(max_bs):
            # 状态槽索引，初始化为 PAD_SLOT_ID
            self.state_indices_list.append(
                torch.full(
                    (i + 1,), self.pad_slot_id, dtype=torch.int32, device=self.device
                )
            )
            # query 起始位置，初始全零
            self.query_start_loc_list.append(
                torch.zeros((i + 2,), dtype=torch.int32, device=self.device)
            )
            # EAGLE 树结构相关的 next_token 和 sibling 索引缓冲区
            self.retrieve_next_token_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
            self.retrieve_next_sibling_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
            self.retrieve_parent_token_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
        # 缓存 decode 和 verify 模式下 query_start_loc 的模板（步长分别为 1 和 draft_token_num）
        self.cached_cuda_graph_decode_query_start_loc = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=self.device
        )
        self.cached_cuda_graph_verify_query_start_loc = torch.arange(
            0,
            max_bs * draft_token_num + 1,
            step=draft_token_num,
            dtype=torch.int32,
            device=self.device,
        )

    def init_cpu_graph_state(self, max_bs: int, max_num_tokens: int):
        # CPU Graph 状态预分配（与 CUDA Graph 类似，但不需要 retrieve 相关缓冲区）
        assert (
            max_num_tokens % max_bs == 0
        ), f"max_num_tokens={max_num_tokens} must be divisible by max_bs={max_bs}"
        for i in range(max_bs):
            self.state_indices_list.append(
                torch.full(
                    (i + 1,), self.pad_slot_id, dtype=torch.int32, device=self.device
                )
            )
            self.query_start_loc_list.append(
                torch.empty((i + 2,), dtype=torch.int32, device=self.device)
            )
        self.cached_cuda_graph_decode_query_start_loc = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=self.device
        )

    def _capture_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        # CUDA/CPU Graph 捕获阶段：使用预分配缓冲区填充 query_start_loc 和 mamba_indices
        if forward_mode.is_decode_or_idle():
            # decode 模式：直接复制缓存的等差查询起始位置
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
            )
        elif forward_mode.is_target_verify():
            # verify 模式：查询起始位置步长为 draft_token_num
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
            )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")
        # 获取 mamba 缓存槽索引并写入预分配的 state_indices 缓冲区
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)

        # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
        # topk > 1 时需要附带 retrieve 树结构信息
        if forward_mode.is_target_verify() and self.topk > 1:
            # They are None during cuda graph capture so skip the copy_...
            # self.retrieve_next_token_list[bs - 1].copy_(spec_info.retrieve_next_token)
            # self.retrieve_next_sibling_list[bs - 1].copy_(spec_info.retrieve_next_sibling)
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                retrieve_next_token=self.retrieve_next_token_list[bs - 1],
                retrieve_next_sibling=self.retrieve_next_sibling_list[bs - 1],
                retrieve_parent_token=self.retrieve_parent_token_list[bs - 1],
            )
        else:
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
            )

    def _replay_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        # CUDA Graph 回放阶段：更新 mamba_indices 和 query_start_loc（处理 padding 请求）
        # 统计填充请求数（seq_len == fill_value 的位置）
        num_padding = torch.count_nonzero(
            seq_lens_cpu == self.get_cuda_graph_seq_len_fill_value()
        )
        # Make sure forward metadata is correctly handled for padding reqs
        # 将填充请求的 req_pool_indices 置为 0，mamba_indices 置为 -1（无效）
        req_pool_indices[bs - num_padding :] = 0
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        mamba_indices[bs - num_padding :] = -1
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)
        if forward_mode.is_decode_or_idle():
            if num_padding == 0:
                # 无填充：直接复制预缓存的 decode query_start_loc
                self.query_start_loc_list[bs - 1].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
                )
            else:
                # 有填充：有效部分从缓存取，填充部分用最后有效位置填充
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[: bs - num_padding]
                )
                self.query_start_loc_list[bs - 1][bs - num_padding :].fill_(
                    bs - num_padding
                )
        elif forward_mode.is_target_verify():
            if num_padding == 0:
                # 无填充：直接复制预缓存的 verify query_start_loc
                self.query_start_loc_list[bs - 1].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
                )
            else:
                # 有填充：有效部分从缓存取，填充部分用最后有效 token 数填充
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[: bs - num_padding]
                )
                self.query_start_loc_list[bs - 1][bs - num_padding :].fill_(
                    (bs - num_padding) * spec_info.draft_token_num
                )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

        # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
        # verify + topk > 1：需要更新 retrieve 树结构缓冲区
        if forward_mode.is_target_verify() and self.topk > 1:
            bs_without_pad = spec_info.retrieve_next_token.shape[0]
            self.retrieve_next_token_list[bs - 1][:bs_without_pad].copy_(
                spec_info.retrieve_next_token
            )
            self.retrieve_next_sibling_list[bs - 1][:bs_without_pad].copy_(
                spec_info.retrieve_next_sibling
            )
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                retrieve_next_token=self.retrieve_next_token_list[bs - 1],
                retrieve_next_sibling=self.retrieve_next_sibling_list[bs - 1],
                retrieve_parent_token=self.retrieve_parent_token_list[bs - 1],
            )
        else:
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
            )

    def get_cuda_graph_seq_len_fill_value(self):
        # Mamba 序列长度填充值为 1（不通过 seq_lens 索引 KV 缓存）
        return 1  # Mamba attn does not use seq lens to index kv cache

    def get_cpu_graph_seq_len_fill_value(self):
        # CPU Graph 同样使用 1 作为填充值
        return 1

    def _track_mamba_state_decode(
        self,
        forward_batch: ForwardBatch,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
    ):
        """
        Track and copy Mamba conv/SSM states during decode for prefix caching.

        During decode, each token update modifies conv_states and ssm_states in-place
        at positions indexed by cache_indices (the working slots). For prefix caching,
        we need to copy these updated states to persistent cache slots (mamba_track_indices)
        so they can be prefix cached.

        This delegates to `track_mamba_states_if_needed`, which performs:
            conv_states[mamba_track_indices[i]] = conv_states[cache_indices[i]]
            ssm_states[mamba_track_indices[i]] = ssm_states[cache_indices[i]]
        for all requests where mamba_track_mask[i] is True.
        """
        # 仅当存在追踪掩码时触发状态复制（避免无谓的内核启动）
        if forward_batch.mamba_track_mask is not None:
            track_mamba_states_if_needed(
                conv_states,
                ssm_states,
                cache_indices,
                forward_batch.mamba_track_mask,
                forward_batch.mamba_track_indices,
                forward_batch.batch_size,
            )

    def _track_mamba_state_extend(
        self,
        forward_batch: ForwardBatch,
        h: torch.Tensor,
        ssm_states: torch.Tensor,
        forward_metadata: ForwardMetadata,
    ):
        """
        Track and copy SSM states during extend for prefix caching.

        After the FLA chunked prefill kernel runs, we need to save the SSM recurrent
        state at the last chunk boundary so it can be reused for prefix caching.
        The source of the state depends on whether the sequence length is aligned
        to FLA_CHUNK_SIZE. See `_init_track_ssm_indices` for more details on how
        the source and destination indices are computed.

        Note: Conv state tracking for extend is handled separately via gather operations
        using indices computed by `_init_track_conv_indices`.
        """
        # 存在追踪掩码时才执行 SSM 状态复制
        if forward_metadata.has_mamba_track_mask:
            h = h.squeeze(0)

            # 非对齐序列：从中间 h 张量取特定 chunk 位置的隐状态
            if forward_metadata.track_ssm_h_src.numel() > 0:
                ssm_states[forward_metadata.track_ssm_h_dst] = h[
                    forward_metadata.track_ssm_h_src
                ].to(ssm_states.dtype, copy=False)
            # 对齐序列：从 ssm_states（final_state）自身复制到目标槽
            if forward_metadata.track_ssm_final_src.numel() > 0:
                ssm_states[forward_metadata.track_ssm_final_dst] = ssm_states[
                    forward_metadata.track_ssm_final_src
                ]



class Mamba2AttnBackend(MambaAttnBackendBase):
    """Attention backend wrapper for Mamba2Mixer kernels."""
    # Mamba2 注意力后端：在 MambaAttnBackendBase 基础上，增加 Mamba2Metadata 准备逻辑

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        config = model_runner.mamba2_config
        assert config is not None  # 必须提供 Mamba2 配置
        self.mamba_chunk_size = config.mamba_chunk_size  # FLA 分块大小

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # 构建 ForwardMetadata 并转换为 Mamba2Metadata（包含 decode/extend 双模式支持）
        metadata = self._forward_metadata(forward_batch)
        self.forward_metadata = Mamba2Metadata.prepare_mixed(
            metadata,
            self.mamba_chunk_size,
            forward_batch,
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        # CUDA Graph 捕获：构建 ForwardMetadata 后再包装为 Mamba2Metadata（decode/verify 模式）
        metadata = self._capture_metadata(bs, req_pool_indices, forward_mode, spec_info)
        draft_token_num = spec_info.draft_token_num if spec_info is not None else 1
        self.forward_metadata = Mamba2Metadata.prepare_decode(
            metadata,
            seq_lens,
            is_target_verify=forward_mode.is_target_verify(),
            draft_token_num=draft_token_num,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        # CUDA Graph 回放：重新生成 ForwardMetadata 并包装为 Mamba2Metadata
        metadata = self._replay_metadata(
            bs, req_pool_indices, forward_mode, spec_info, seq_lens_cpu
        )
        draft_token_num = spec_info.draft_token_num if spec_info is not None else 1
        self.forward_metadata = Mamba2Metadata.prepare_decode(
            metadata,
            seq_lens,
            is_target_verify=forward_mode.is_target_verify(),
            draft_token_num=draft_token_num,
        )

    def forward(
        self,
        mixer: MambaMixer2,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        layer_id: int,
        mup_vector: Optional[torch.Tensor] = None,
        use_triton_causal_conv: bool = False,
    ):
        # 直接调用 Mamba2Mixer 的 forward（混合 decode/extend），使用本层的 KV 缓存和元数据
        assert isinstance(self.forward_metadata, Mamba2Metadata)
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)  # 获取本层状态缓存
        return mixer.forward(
            hidden_states=hidden_states,
            output=output,
            layer_cache=layer_cache,
            metadata=self.forward_metadata,
            mup_vector=mup_vector,
            use_triton_causal_conv=use_triton_causal_conv,
        )

    def forward_decode(self, *args, **kwargs):
        # Mamba2 不走标准 decode 路径，直接通过 forward() 处理混合批次
        raise NotImplementedError(
            "Mamba2AttnBackend's forward is called directly instead of through HybridLinearAttnBackend, as it supports mixed prefill and decode"
        )

    def forward_extend(self, *args, **kwargs):
        # Mamba2 不走标准 extend 路径，直接通过 forward() 处理混合批次
        raise NotImplementedError(
            "Mamba2AttnBackend's forward is called directly instead of through HybridLinearAttnBackend, as it supports mixed prefill and decode"
        )


class HybridLinearAttnBackend(AttentionBackend):
    """Manages a full and linear attention backend"""
    # 混合线性注意力后端：组合全注意力后端和线性注意力（Mamba/GDN）后端，按层 ID 路由

    def __init__(
        self,
        full_attn_backend: AttentionBackend,
        linear_attn_backend: MambaAttnBackendBase,
        full_attn_layers: list[int],
    ):
        self.full_attn_layers = full_attn_layers  # 使用全注意力的层 ID 集合
        self.full_attn_backend = full_attn_backend  # 全注意力后端（如 FlashInfer/Triton）
        self.linear_attn_backend = linear_attn_backend  # 线性注意力后端（Mamba/GDN/KDA/Lightning）
        self.attn_backend_list = [full_attn_backend, linear_attn_backend]  # 统一迭代用列表

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # 同时初始化两个后端的前向元数据
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        # 同时初始化两个后端的 CUDA Graph 固定缓冲区
        for attn_backend in self.attn_backend_list:
            attn_backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def init_cpu_graph_state(self, max_bs: int, max_num_tokens: int):
        # 同时初始化两个后端的 CPU Graph 固定缓冲区
        for attn_backend in self.attn_backend_list:
            attn_backend.init_cpu_graph_state(max_bs, max_num_tokens)

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
        # 广播 CUDA Graph 捕获阶段的元数据初始化到所有子后端
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_capture_cuda_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            )

    def init_forward_metadata_capture_cpu_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        # 广播 CPU Graph 捕获阶段的元数据初始化到所有子后端
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_capture_cpu_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            )

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
        # 广播 CUDA Graph 回放阶段的元数据更新到所有子后端
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_replay_cuda_graph(
                bs,
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                encoder_lens,
                forward_mode,
                spec_info,
                seq_lens_cpu,
            )

    def get_cuda_graph_seq_len_fill_value(self):
        # 使用全注意力后端的序列长度填充值（由全注意力后端决定 KV 缓存索引方式）
        return self.full_attn_backend.get_cuda_graph_seq_len_fill_value()

    def get_cpu_graph_seq_len_fill_value(self):
        # 使用全注意力后端的 CPU Graph 序列长度填充值
        return self.full_attn_backend.get_cpu_graph_seq_len_fill_value()

    def forward_decode(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q: Optional[torch.Tensor] = None,  # For full attention
        k: Optional[torch.Tensor] = None,  # For full attention
        v: Optional[torch.Tensor] = None,  # For full attention
        mixed_qkv: Optional[torch.Tensor] = None,  # For linear attention
        a: Optional[torch.Tensor] = None,  # For GDN linear attention
        b: Optional[torch.Tensor] = None,  # For GDN linear attention
        **kwargs,
    ):
        # decode 阶段：根据层 ID 路由到全注意力或线性注意力后端
        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        if layer_id in self.full_attn_layers:
            # 全注意力层：使用 Q/K/V 调用全注意力后端
            return self.full_attn_backend.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        # Linear attention backend
        # 线性注意力层：使用 mixed_qkv 或 a/b 参数调用线性注意力后端
        return self.linear_attn_backend.forward_decode(
            q=q,
            k=k,
            v=v,
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            **kwargs,
        )

    def forward_extend(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q: Optional[torch.Tensor] = None,  # For full attention
        k: Optional[torch.Tensor] = None,  # For full attention
        v: Optional[torch.Tensor] = None,  # For full attention
        mixed_qkv: Optional[torch.Tensor] = None,  # For linear attention
        a: Optional[torch.Tensor] = None,  # For GDN linear attention
        b: Optional[torch.Tensor] = None,  # For GDN linear attention
        **kwargs,
    ):
        # extend/prefill 阶段：根据层 ID 路由到全注意力或线性注意力后端
        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        if layer_id in self.full_attn_layers:
            return self.full_attn_backend.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        # Linear attention backend
        return self.linear_attn_backend.forward_extend(
            q=q,
            k=k,
            v=v,
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            **kwargs,
        )

    def forward(
        self,
        q: Optional[torch.Tensor] = None,  # For full attention
        k: Optional[torch.Tensor] = None,  # For full attention
        v: Optional[torch.Tensor] = None,  # For full attention
        layer: RadixAttention = None,
        forward_batch: ForwardBatch = None,
        save_kv_cache: bool = True,
        mixed_qkv: Optional[torch.Tensor] = None,  # For linear attention
        a: Optional[torch.Tensor] = None,  # For linear attention
        b: Optional[torch.Tensor] = None,  # For linear attention
        **kwargs,
    ):
        # 统一 forward 入口：根据前向模式和层 ID 分发到对应子后端
        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        is_linear_attn = layer_id not in self.full_attn_layers  # 是否为线性注意力层

        if forward_batch.forward_mode.is_idle():
            # idle 模式：返回零输出张量（不执行真实计算）
            if is_linear_attn:
                return mixed_qkv.new_empty(
                    mixed_qkv.shape[0], layer.num_v_heads, layer.head_v_dim
                )
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
        elif forward_batch.forward_mode.is_decode():
            # decode 模式：委托给 forward_decode
            return self.forward_decode(
                layer,
                forward_batch,
                save_kv_cache,
                q,
                k,
                v,
                mixed_qkv,
                a,
                b,
                **kwargs,
            )
        else:
            # extend/prefill 模式：委托给 forward_extend
            return self.forward_extend(
                layer,
                forward_batch,
                save_kv_cache,
                q,
                k,
                v,
                mixed_qkv,
                a,
                b,
                **kwargs,
            )

    def update_mamba_state_after_mtp_verify(
        self,
        accepted_steps: torch.Tensor,
        mamba_track_indices: Optional[torch.Tensor],
        mamba_steps_to_track: Optional[torch.Tensor],
        model,
    ):
        """
        Update mamba states after MTP verify using fully fused Triton kernel.

        This replaces the original advanced indexing operations with a single fused
        gather-scatter kernel that also handles masking internally, avoiding:
        - index_elementwise_kernel from tensor[bool_mask]
        - index_select kernel launches
        - nonzero kernel launches
        """
        # MTP verify 完成后，根据接受步数更新 Mamba 状态（使用融合 Triton 内核）
        request_number = accepted_steps.shape[0]

        # 获取当前 batch 每个请求对应的 Mamba 缓存槽索引
        state_indices_tensor = (
            self.linear_attn_backend.forward_metadata.mamba_cache_indices[
                :request_number
            ]
        )

        # 获取所有层的 Mamba2 缓存（conv 状态、temporal 隐状态、中间状态）
        mamba_caches = (
            self.linear_attn_backend.req_to_token_pool.get_speculative_mamba2_params_all_layers()
        )

        conv_states = mamba_caches.conv[0]
        ssm_states = mamba_caches.temporal
        intermediate_state_cache = mamba_caches.intermediate_ssm
        intermediate_conv_window_cache = mamba_caches.intermediate_conv_window[0]

        # Use fully fused kernel that handles masking internally
        # This avoids separate nonzero() and index_select() calls
        # 使用融合内核将中间 SSM 状态按接受步数 scatter 到工作槽，避免多次 Python 级内核调用
        fused_mamba_state_scatter_with_mask(
            ssm_states,
            intermediate_state_cache,
            state_indices_tensor,
            accepted_steps,
        )
        # 同样处理卷积窗口状态
        fused_mamba_state_scatter_with_mask(
            conv_states,
            intermediate_conv_window_cache,
            state_indices_tensor,
            accepted_steps,
        )

        # Track indices used for tracking mamba states for prefix cache
        # 如果有需要追踪的请求（prefix cache），额外执行一次 scatter 到追踪槽
        if mamba_track_indices is not None:
            assert mamba_steps_to_track is not None
            # Use fully fused kernel for track scatter operations
            fused_mamba_state_scatter_with_mask(
                ssm_states,
                intermediate_state_cache,
                mamba_track_indices,
                mamba_steps_to_track,
            )
            fused_mamba_state_scatter_with_mask(
                conv_states,
                intermediate_conv_window_cache,
                mamba_track_indices,
                mamba_steps_to_track,
            )
