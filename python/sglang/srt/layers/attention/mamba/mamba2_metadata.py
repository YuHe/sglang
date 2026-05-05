# Copyright 2025 SGLang Team
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
# Adapted from https://github.com/vllm-project/vllm/blob/2c58742dff8613a3bd7496f2008ce927e18d38d1/vllm/model_executor/layers/mamba/mamba2_metadata.py

# Mamba2 前向传播元数据模块：负责计算 prefill/decode/mixed 场景下的辅助元数据

import math
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch


# 前向传播基础元数据数据类，持有请求的位置与缓存索引信息
@dataclass(kw_only=True)
class ForwardMetadata:
    # 累积序列起始位置，形状 (batch+1,)，用于变长 batch 的 token 定位
    query_start_loc: torch.Tensor
    # Mamba 缓存槽位索引，将 batch 维度映射到实际缓存行
    mamba_cache_indices: torch.Tensor
    # 用于树形解码（topk>1 eagle）时的 GDN 缓存索引
    mamba_cache_indices_gdn: Optional[torch.Tensor] = None
    # For topk > 1 eagle
    # Eagle 推测解码所需的 next-token 检索信息
    retrieve_next_token: Optional[torch.Tensor] = None
    retrieve_next_sibling: Optional[torch.Tensor] = None
    retrieve_parent_token: Optional[torch.Tensor] = None
    # For prefill radix cache
    # prefill 阶段 radix cache 所需的卷积状态追踪索引
    track_conv_indices: Optional[torch.Tensor] = None
    # SSM 隐藏状态的源/目标索引，用于 radix cache 状态复制
    track_ssm_h_src: Optional[torch.Tensor] = None
    track_ssm_h_dst: Optional[torch.Tensor] = None
    # SSM 最终状态的源/目标索引
    track_ssm_final_src: Optional[torch.Tensor] = None
    track_ssm_final_dst: Optional[torch.Tensor] = None

    # 是否处于目标验证阶段（推测解码用）
    is_target_verify: bool = False
    # 推测 draft token 数量
    draft_token_num: int = 1

    # 是否存在 Mamba track mask（用于过滤无效槽位）
    has_mamba_track_mask: bool = False
    mamba_track_mask_indices: Optional[torch.Tensor] = None
    conv_states_mask_indices: Optional[torch.Tensor] = None


# Mamba2 专用元数据，继承基础元数据，在整个 forward pass 的所有 mamba2 层中共享
@dataclass(kw_only=True)
class Mamba2Metadata(ForwardMetadata):
    """stable metadata across all mamba2 layers in the forward pass"""

    # prefill 序列数量（有新 token 需要计算的请求数）
    num_prefills: int
    # prefill 总 token 数
    num_prefill_tokens: int
    # decode 请求数量（每次仅生成 1 个 token 的请求数）
    num_decodes: int

    # mixed 请求（prefill+decode 混合批次）所需的额外元数据，frozen 保证不可变
    @dataclass(kw_only=True, frozen=True)
    class MixedMetadata:
        # 每条 prefill 序列是否具有初始 SSM 状态（即 prefix cache 命中）
        has_initial_states: torch.Tensor
        # 是否需要准备初始状态（避免后续 device sync）
        prep_initial_states: bool

        # Mamba 物理 chunk 大小（每个 chunk 含多少 token）
        chunk_size: int
        # 每个 token 所属的序列索引，形状 (1, num_prefill_tokens)
        seq_idx: torch.Tensor
        # 每个逻辑 chunk 对应的物理 chunk 索引
        chunk_indices: torch.Tensor
        # 每个逻辑 chunk 在物理 chunk 内的起始偏移
        chunk_offsets: torch.Tensor

        # 各 prefill 序列在 CPU 上的序列长度列表
        extend_seq_lens_cpu: list[int]

    # mixed_metadata 仅在 extend/mixed 请求时使用；pure decode 时为 None
    mixed_metadata: MixedMetadata | None = None
    """`mixed_metadata` is used for extend/mixed requests"""

    @staticmethod
    def _query_start_loc_to_chunk_indices_offsets(
        query_start_loc: torch.Tensor, chunk_size: int, total_seqlens: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        将累积序列位置张量转换为 Mamba chunk 的物理索引和偏移量。

        Args:
            query_start_loc (torch.Tensor): 1D tensor of cumulative sequence
                lengths, shape (num_seqs + 1,).
                The first element should be 0. Each entry represents the starting
                index of a sequence in the flattened token array.
            chunk_size (int): The size of each physical mamba chunk
                (number of tokens per chunk).
            total_seqlens (int): The total number of tokens in the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - chunk_indices (torch.Tensor): 1D tensor of indices
                    indicating the physical chunk for each logical chunk.
                - chunk_offsets (torch.Tensor): 1D tensor of offsets
                    indicating the starting index of each logical chunk within
                    its physical chunk.

        This function computes the chunk indices and offsets for the given
        query_start_loc and chunk_size. Both are tensors of integers with length N,
        where N is the number of logical (pseudo) chunks.
        A logical chunk is a sequence of tokens that are all part of the same
        sequence and are all in the same physical mamba chunk.
        In other words, a logical chunk changes every time we cross a sequence
        boundary or a physical mamba chunk boundary.
        Logical chunks are needed to handle batched requests with initial states
        (see _state_passing_fwd and _chunk_scan_fwd).
        The chunk_indices tensor contains the index of the physical chunk for each
        logical chunk.
        The chunk_offsets tensor contains the offset (AKA starting index) of the
        logical chunk in the physical chunk.

        Example:
        query_start_loc = [0, 5, 10]
        chunk_size = 8
        total_seqlens = 10
        -> chunk_indices = [0, 0, 1]
        -> chunk_offsets = [0, 5, 0]

        In this example, we have 2 sequences, each with 5 tokens. The physical
        chunk size is 8 tokens.
        We have three logical chunks:
        - the first logical chunk starts at token 0 in the first physical chunk
            and contains all 5 tokens from the first sequence
        - the second logical chunk starts at token 5 in the first physical chunk
            and contains first 3 tokens from the second sequence
        - the third logical chunk starts at token 0 in the second physical chunk
            and contains the remaining 2 tokens from the second sequence
        """

        # 去除首个 0 元素，得到各序列的结束位置
        cu_seqlens = query_start_loc[1:]  # remove prepended 0

        # 计算逻辑 chunk 总数 N：
        # 物理 chunk 数 + 因序列边界跨越 chunk 而增加的分裂数
        # outputs will have length expansion of chunks that do not divide
        # chunk_size
        N = (
            math.ceil(total_seqlens / chunk_size)
            + (cu_seqlens[:-1] % chunk_size > 0).sum()
        )
        # 初始化 chunk_indices 为 0..N-1（后续按序列边界调整）
        chunk_indices = torch.arange(N, dtype=torch.int, device=query_start_loc.device)
        # 初始化 chunk_offsets 全为 0
        chunk_offsets = torch.zeros(
            (N,), dtype=torch.int, device=query_start_loc.device
        )

        p = 0  # num of insertions
        # 遍历相邻序列对 (起始, 结束)
        for s, e in zip(cu_seqlens[:-1], cu_seqlens[1:]):

            # 若序列起始位置未对齐 chunk，则插入计数加 1
            # if does not divide chunk_size, then there is one chunk insertion
            p += s % chunk_size > 0

            # 计算逻辑 chunk 在输出数组中的范围 [_s, _e)
            # get the dimensions
            # - the + 1 for _e is to shift the boundary by one chunk
            # - this shifting is not needed if chunk_size divides e
            _s, _e = s // chunk_size + p, e // chunk_size + p + (e % chunk_size > 0)

            # 将该范围内的物理 chunk 索引减去插入偏移 p，还原真实物理索引
            # adjust indices and offsets
            chunk_indices[_s:_e] -= p
            # 记录本逻辑 chunk 在物理 chunk 内的起始偏移
            chunk_offsets[_s] = s % chunk_size

        return chunk_indices, chunk_offsets

    @staticmethod
    def prepare_decode(
        forward_metadata: ForwardMetadata,
        seq_lens: torch.Tensor,
        *,
        is_target_verify: bool,
        draft_token_num: int,
    ) -> "Mamba2Metadata":
        """纯 decode 路径（CUDA graph 捕获阶段），此时 num_prefills=0。"""
        """This path is run during CUDA graph capture, i.e. decode only, so `num_prefills` is 0"""
        return Mamba2Metadata(
            query_start_loc=forward_metadata.query_start_loc,
            mamba_cache_indices=forward_metadata.mamba_cache_indices,
            retrieve_next_token=forward_metadata.retrieve_next_token,
            retrieve_next_sibling=forward_metadata.retrieve_next_sibling,
            retrieve_parent_token=forward_metadata.retrieve_parent_token,
            num_decodes=len(seq_lens),
            num_prefills=0,
            num_prefill_tokens=0,
            is_target_verify=is_target_verify,
            draft_token_num=draft_token_num,
        )

    @classmethod
    def prepare_mixed(
        cls,
        forward_metadata: ForwardMetadata,
        chunk_size: int,
        forward_batch: ForwardBatch,
    ) -> "Mamba2Metadata":
        """mixed 路径（含 extend 请求），不能与 CUDA graph 一起使用。"""
        """This path cannot run with CUDA graph, as it contains extend requests."""
        # 若无 extend token，则退化为纯 decode 处理
        if forward_batch.extend_num_tokens is None:
            draft_token_num = (
                forward_batch.spec_info.draft_token_num
                if forward_batch.spec_info is not None
                else 1
            )
            return cls.prepare_decode(
                forward_metadata,
                forward_batch.seq_lens,
                is_target_verify=forward_batch.forward_mode.is_target_verify(),
                draft_token_num=draft_token_num,
            )
        # 统计 prefill/decode 数量及 token 数
        num_prefills = len(forward_batch.extend_seq_lens)
        num_prefill_tokens = forward_batch.extend_num_tokens
        num_decodes = len(forward_batch.seq_lens) - num_prefills
        # context_lens_tensor: 各序列在 prefill 前已存在的 prefix 长度
        context_lens_tensor = forward_batch.extend_prefix_lens
        assert context_lens_tensor is not None
        # precompute flag to avoid device syncs later
        # 预计算初始状态标志，避免后续在 GPU 上做 device sync
        has_initial_states = context_lens_tensor > 0
        prep_initial_states = torch.any(has_initial_states[:num_prefills]).item()

        # 截取 prefill 部分的累积序列位置（decode 序列不需要 chunk 计算）
        query_start_loc = forward_metadata.query_start_loc[: num_prefills + 1]
        # 构造每个 prefill token 所属序列的索引，形状 (1, num_prefill_tokens)
        seq_idx = torch.repeat_interleave(
            torch.arange(
                num_prefills, dtype=torch.int32, device=query_start_loc.device
            ),
            query_start_loc.diff(),
            output_size=num_prefill_tokens,
        )
        seq_idx.unsqueeze_(0)

        # 预先计算 chunked prefill 的元数据，在顶层 model forward 中算一次，复用于各 mamba 层
        # We compute metadata for chunked prefill once at the top level model
        # forward and reuse them in mamba layers. If not needed, they will be
        # ignored inside mamba kernels.
        chunk_offsets, chunk_indices = None, None
        if prep_initial_states:
            # 仅当存在初始状态时才需要 chunk 索引/偏移（无初始状态时 SSM 可顺序扫描）
            chunk_indices, chunk_offsets = (
                cls._query_start_loc_to_chunk_indices_offsets(
                    query_start_loc, chunk_size, num_prefill_tokens
                )
            )

        # 获取 draft token 数（推测解码用，默认 1）
        draft_token_num = (
            getattr(forward_batch.spec_info, "draft_token_num", 1)
            if forward_batch.spec_info is not None
            else 1
        )
        # 组装并返回完整的 Mamba2Metadata
        return Mamba2Metadata(
            query_start_loc=query_start_loc,
            mamba_cache_indices=forward_metadata.mamba_cache_indices,
            retrieve_next_token=forward_metadata.retrieve_next_token,
            retrieve_next_sibling=forward_metadata.retrieve_next_sibling,
            retrieve_parent_token=forward_metadata.retrieve_parent_token,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            is_target_verify=forward_batch.forward_mode.is_target_verify(),
            draft_token_num=draft_token_num,
            mixed_metadata=cls.MixedMetadata(
                has_initial_states=has_initial_states,
                prep_initial_states=prep_initial_states,
                chunk_size=chunk_size,
                seq_idx=seq_idx,
                chunk_indices=chunk_indices,
                chunk_offsets=chunk_offsets,
                extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            ),
        )
