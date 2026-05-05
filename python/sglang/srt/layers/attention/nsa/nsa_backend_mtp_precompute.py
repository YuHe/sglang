"""Multi-step precompute utilities for Native Sparse Attention backend.

This module provides optimization utilities for multi-step speculative decoding
by precomputing shared metadata once and copying it to multiple backend instances.
"""
# NSA MTP（Multi-Token Prediction / Speculative Decoding）元数据预计算模块
# 在多步推测解码场景中，将多个 backend 共用的元数据预计算一次，避免重复计算

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.nsa.utils import compute_nsa_seqlens

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.speculative.spec_info import SpecInput


@dataclass
class PrecomputedMetadata:
    """Precomputed metadata shared across multiple backend instances.

    Used for multi-step speculative decoding where multiple backends
    need identical metadata. Precomputing once and copying N times
    is much faster than computing N times.

    """
    # 预计算的共享元数据结构，供 decode/target_verify/draft_extend 三种模式使用

    # Basic seqlens
    # 基础序列长度：每个请求的 KV cache 长度和累计前缀和
    cache_seqlens: torch.Tensor  # int32, [bs]
    cu_seqlens_k: torch.Tensor  # int32, [bs+1]

    # Page table
    # 页表相关：稀疏注意力的 top-k block 物理地址和可选的转换版本
    page_indices: torch.Tensor  # int32, [bs, max_len] or [expanded_bs, max_len]
    real_page_table: Optional[torch.Tensor]  # int32, transformed version

    # NSA seqlens
    # NSA 稀疏注意力序列长度：扩展后（含推测 draft token）的序列长度及其前缀和
    seqlens_expanded: torch.Tensor  # int32, [expanded_size]
    nsa_cache_seqlens: torch.Tensor  # int32, [expanded_size]
    nsa_cu_seqlens_k: torch.Tensor  # int32, [expanded_size+1]
    seqlens_expanded_size: int

    # Dimensions
    # 尺寸参数：decode/draft_extend 使用 max_len，target_verify 使用 max_seqlen_k
    max_len: int  # for decode/draft_extend
    max_seqlen_k: int  # for target_verify

    # FlashMLA (optional)
    # FlashMLA 元数据（仅在 nsa_decode_impl == "flashmla_kv" 时使用）
    flashmla_metadata: Optional[torch.Tensor] = None


def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:
    """Compute cumulative sequence lengths with padding."""
    # 计算序列长度的前缀和（cumsum），并在开头补 0，得到 [0, s1, s1+s2, ...] 格式
    assert seqlens.dtype == torch.int32
    return torch.nn.functional.pad(
        torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)
    )


class NativeSparseAttnBackendMTPPrecomputeMixin:
    """Mixin class providing metadata precomputation for multi-step speculative decoding.

    This mixin provides the _precompute_replay_metadata method and its helpers,
    which are used to optimize CUDA graph replay in multi-step scenarios.
    """
    # NSA 后端 MTP 预计算 Mixin 类
    # 为多步推测解码提供元数据预计算能力，优化 CUDA graph 回放性能

    def _precompute_replay_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        forward_mode: "ForwardMode",
        spec_info: Optional["SpecInput"],
    ) -> PrecomputedMetadata:
        """Precompute all shared metadata for multi-step backends.

        This function extracts and computes all operations that are
        identical across different backend instances in multi-step
        speculative decoding.

        Args:
            bs: Batch size
            req_pool_indices: Request pool indices [bs]
            seq_lens: Sequence lengths [bs]
            seq_lens_cpu: Sequence lengths on CPU [bs]
            forward_mode: Forward mode (decode/target_verify/draft_extend)
            spec_info: Speculative decoding info (for draft_extend mode)

        Returns:
            PrecomputedMetadata containing all shared intermediate results
        """
        # Slice inputs to batch size
        # 截取到实际 batch size，丢弃预分配的多余空间
        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs]
        req_pool_indices = req_pool_indices[:bs]

        # Dispatch to mode-specific precomputation
        # 根据 forward_mode 分发到对应的预计算函数
        if forward_mode.is_decode_or_idle():
            return self._precompute_decode_mode(
                bs, req_pool_indices, seq_lens, seq_lens_cpu
            )
        elif forward_mode.is_target_verify():
            return self._precompute_target_verify_mode(
                bs, req_pool_indices, seq_lens, seq_lens_cpu
            )
        elif forward_mode.is_draft_extend():
            return self._precompute_draft_extend_mode(
                bs, req_pool_indices, seq_lens, seq_lens_cpu, spec_info
            )
        else:
            raise ValueError(f"Unsupported forward mode: {forward_mode}")

    def _precompute_decode_mode(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
    ) -> PrecomputedMetadata:
        """Precompute metadata for normal decode mode."""
        # decode 模式：每个请求生成 1 个新 token，KV cache 长度为当前序列长度
        max_len = int(seq_lens_cpu.max().item())  # 所有请求中最大的 KV 序列长度

        # Convert to int32 and compute cumsum
        # 计算 int32 序列长度及其前缀和
        cache_seqlens = seq_lens.to(torch.int32)
        cu_seqlens_k = compute_cu_seqlens(cache_seqlens)

        # Get page indices from cache
        # 从请求池页表中取出每个请求的 KV cache 物理地址（前 max_len 页）
        page_indices = self.req_to_token[req_pool_indices, :max_len].contiguous()

        # Compute NSA seqlens
        # 将 KV 长度截断到 nsa_index_topk，计算 NSA 稀疏注意力的有效长度
        nsa_cache_seqlens = compute_nsa_seqlens(
            cache_seqlens, nsa_index_topk=self.nsa_index_topk
        )
        seqlens_expanded = cache_seqlens  # decode 模式不扩展，expanded 即原始长度
        seqlens_expanded_size = seqlens_expanded.shape[0]

        # Compute NSA cumsum
        # 计算 NSA cache 序列长度的前缀和
        nsa_cu_seqlens_k = compute_cu_seqlens(nsa_cache_seqlens)

        # Transform page table if needed
        # 如果 real_page_size > 1，需要将 page_indices 转换为实际页表地址
        if self.real_page_size > 1:
            real_page_table = self._transform_table_1_to_real(page_indices)
        else:
            real_page_table = None  # Will use page_indices directly

        # Compute FlashMLA metadata if needed
        # 仅在使用 flashmla_kv 实现时预计算 FlashMLA 的 split 元数据
        flashmla_metadata = None
        if self.nsa_decode_impl == "flashmla_kv":
            flashmla_metadata = self._compute_flashmla_metadata(
                cache_seqlens=nsa_cache_seqlens,
                seq_len_q=1,  # decode 模式每个请求只有 1 个 query token
            )

        return PrecomputedMetadata(
            cache_seqlens=cache_seqlens,
            cu_seqlens_k=cu_seqlens_k,
            page_indices=page_indices,
            real_page_table=real_page_table,
            seqlens_expanded=seqlens_expanded,
            nsa_cache_seqlens=nsa_cache_seqlens,
            nsa_cu_seqlens_k=nsa_cu_seqlens_k,
            seqlens_expanded_size=seqlens_expanded_size,
            max_len=max_len,
            max_seqlen_k=max_len,
            flashmla_metadata=flashmla_metadata,
        )

    def _precompute_target_verify_mode(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
    ) -> PrecomputedMetadata:
        """Precompute metadata for target verify mode."""
        # target_verify 模式：目标模型验证多个 draft token，KV 长度包含 draft
        max_seqlen_k = int(
            seq_lens_cpu.max().item() + self.speculative_num_draft_tokens
        )

        # Cache seqlens with draft tokens
        # 每个请求的 KV 长度 = 原始长度 + draft token 数
        cache_seqlens = (seq_lens + self.speculative_num_draft_tokens).to(torch.int32)
        cu_seqlens_k = compute_cu_seqlens(cache_seqlens)

        # Page indices (repeated for each draft token)
        # 每行（请求）重复 speculative_num_draft_tokens 次，因为每个 draft token 共享同一行页表
        page_indices = self.req_to_token[req_pool_indices, :max_seqlen_k]
        page_indices = torch.repeat_interleave(
            page_indices, repeats=self.speculative_num_draft_tokens, dim=0
        ).contiguous()

        # Generate expanded seqlens
        # 为每个 draft token 生成对应的 KV 长度（从最早 draft 到最新）
        extend_seq_lens_cpu = [self.speculative_num_draft_tokens] * bs
        seqlens_int32_cpu = [
            self.speculative_num_draft_tokens + kv_len
            for kv_len in seq_lens_cpu.tolist()
        ]
        # 对每个请求，生成从 (kv_len - qo_len + 1) 到 kv_len 的等差序列
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

        # Compute NSA seqlens
        # 将 expanded 序列长度截断到 top-k，计算 NSA 稀疏注意力有效长度
        nsa_cache_seqlens = compute_nsa_seqlens(seqlens_expanded, self.nsa_index_topk)
        seqlens_expanded_size = seqlens_expanded.shape[0]

        # NSA cumsum
        nsa_cu_seqlens_k = compute_cu_seqlens(nsa_cache_seqlens)

        # Transform page table
        if self.real_page_size > 1:
            real_page_table = self._transform_table_1_to_real(page_indices)
        else:
            real_page_table = None

        # FlashMLA metadata
        flashmla_metadata = None
        if self.nsa_decode_impl == "flashmla_kv":
            flashmla_metadata = self._compute_flashmla_metadata(
                cache_seqlens=nsa_cache_seqlens,
                seq_len_q=1,
            )

        return PrecomputedMetadata(
            cache_seqlens=cache_seqlens,
            cu_seqlens_k=cu_seqlens_k,
            page_indices=page_indices,
            real_page_table=real_page_table,
            seqlens_expanded=seqlens_expanded,
            nsa_cache_seqlens=nsa_cache_seqlens,
            nsa_cu_seqlens_k=nsa_cu_seqlens_k,
            seqlens_expanded_size=seqlens_expanded_size,
            max_len=-1,  # Not used in this mode（target_verify 不需要 max_len）
            max_seqlen_k=max_seqlen_k,
            flashmla_metadata=flashmla_metadata,
        )

    def _precompute_draft_extend_mode(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        spec_info: "SpecInput",
    ) -> PrecomputedMetadata:
        """Precompute metadata for draft extend mode."""
        # draft_extend 模式：在推测解码的 draft 阶段，每个请求接受若干 token
        max_seqlen_k = int(seq_lens_cpu.max().item())

        # Cache seqlens
        cache_seqlens = seq_lens.to(torch.int32)
        cu_seqlens_k = compute_cu_seqlens(cache_seqlens)

        # Extend seqlens from spec_info: num_accepted_tokens already includes
        # the bonus token (drafts + 1).
        # 从 spec_info 获取每个请求实际被接受的 token 数（含 bonus token）
        extend_seq_lens = spec_info.num_accepted_tokens[:bs]
        extend_seq_lens_cpu = extend_seq_lens.tolist()

        # Page indices (repeated per accept length)
        # 将每个请求的页表行重复 accept_length 次，为每个被接受 token 提供完整 KV 索引
        page_indices = self.req_to_token[req_pool_indices, :max_seqlen_k]
        page_indices = torch.repeat_interleave(
            page_indices, repeats=extend_seq_lens, dim=0
        ).contiguous()

        # Generate expanded seqlens
        # 为每个被接受 token 生成对应的历史 KV 长度序列
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
                    seq_lens_cpu.tolist(),
                    strict=True,
                )
            ]
        )

        # Compute NSA seqlens
        # 截断到 top-k 并计算 NSA 稀疏注意力有效长度
        nsa_cache_seqlens = compute_nsa_seqlens(seqlens_expanded, self.nsa_index_topk)
        seqlens_expanded_size = seqlens_expanded.shape[0]

        # NSA cumsum
        nsa_cu_seqlens_k = compute_cu_seqlens(nsa_cache_seqlens)

        # Transform page table
        if self.real_page_size > 1:
            real_page_table = self._transform_table_1_to_real(page_indices)
        else:
            real_page_table = None

        # FlashMLA metadata
        flashmla_metadata = None
        if self.nsa_decode_impl == "flashmla_kv":
            flashmla_metadata = self._compute_flashmla_metadata(
                cache_seqlens=nsa_cache_seqlens,
                seq_len_q=1,
            )

        return PrecomputedMetadata(
            cache_seqlens=cache_seqlens,
            cu_seqlens_k=cu_seqlens_k,
            page_indices=page_indices,
            real_page_table=real_page_table,
            seqlens_expanded=seqlens_expanded,
            nsa_cache_seqlens=nsa_cache_seqlens,
            nsa_cu_seqlens_k=nsa_cu_seqlens_k,
            seqlens_expanded_size=seqlens_expanded_size,
            max_len=max_seqlen_k,   # draft_extend 模式复用 max_seqlen_k 作为 max_len
            max_seqlen_k=max_seqlen_k,
            flashmla_metadata=flashmla_metadata,
        )
