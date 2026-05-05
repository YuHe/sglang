"""
Verification utilities for NSA backend fused metadata copy operations.

This module contains verification code to ensure that fused metadata copy kernels
produce the same results as individual copy operations.
"""
# NSA MTP（Multi-Token Prediction）元数据融合拷贝验证模块
# 用于校验融合 kernel 的拷贝结果与逐个单独拷贝的结果是否完全一致

import torch


def verify_single_backend_fused_metadata_copy(
    metadata,
    precomputed,
    forward_mode,
    bs,
    flashmla_num_splits_src=None,
    flashmla_metadata_src=None,
    flashmla_num_splits_dst=None,
    flashmla_metadata_dst=None,
):
    """
    Verify that the fused metadata copy kernel produces the same results as individual copies.

    Args:
        metadata: The NSA metadata object containing destination tensors
        precomputed: The precomputed metadata containing source tensors
        forward_mode: The forward mode (decode, target_verify, or draft_extend)
        bs: Batch size
        flashmla_num_splits_src: Source FlashMLA num_splits tensor (optional)
        flashmla_metadata_src: Source FlashMLA metadata tensor (optional)
        flashmla_num_splits_dst: Destination FlashMLA num_splits tensor (optional)
        flashmla_metadata_dst: Destination FlashMLA metadata tensor (optional)

    Raises:
        RuntimeError: If verification fails (tensors don't match)
    """
    # 单 backend 融合拷贝验证：对比融合 kernel 结果与参考实现（逐个拷贝）
    # Clone destination tensors to preserve fused kernel results
    # 克隆融合 kernel 已写入的目标张量，用于后续比对
    fused_cache_seqlens = metadata.cache_seqlens_int32.clone()
    fused_cu_seqlens_k = metadata.cu_seqlens_k.clone()
    fused_page_table_1 = metadata.page_table_1.clone()
    fused_nsa_cache_seqlens = metadata.nsa_cache_seqlens_int32.clone()
    fused_nsa_seqlens_expanded = metadata.nsa_seqlens_expanded.clone()
    fused_nsa_cu_seqlens_k = metadata.nsa_cu_seqlens_k.clone()
    fused_real_page_table = (
        metadata.real_page_table.clone()
        if precomputed.real_page_table is not None
        else None
    )
    fused_flashmla_num_splits = None
    fused_flashmla_metadata = None
    if precomputed.flashmla_metadata is not None:
        # 克隆 FlashMLA 相关元数据（num_splits 和 metadata）
        fused_flashmla_num_splits = flashmla_num_splits_dst.clone()
        fused_flashmla_metadata = flashmla_metadata_dst.clone()

    # Create reference tensors (zeroed out)
    # 创建全零的参考张量，之后用逐个拷贝填充
    ref_cache_seqlens = torch.zeros_like(metadata.cache_seqlens_int32)
    ref_cu_seqlens_k = torch.zeros_like(metadata.cu_seqlens_k)
    ref_page_table_1 = torch.zeros_like(metadata.page_table_1)
    ref_nsa_cache_seqlens = torch.zeros_like(metadata.nsa_cache_seqlens_int32)
    ref_nsa_seqlens_expanded = torch.zeros_like(metadata.nsa_seqlens_expanded)
    ref_nsa_cu_seqlens_k = torch.zeros_like(metadata.nsa_cu_seqlens_k)
    ref_real_page_table = (
        torch.zeros_like(metadata.real_page_table)
        if precomputed.real_page_table is not None
        else None
    )
    ref_flashmla_num_splits = None
    ref_flashmla_metadata = None
    if precomputed.flashmla_metadata is not None:
        ref_flashmla_num_splits = torch.zeros_like(flashmla_num_splits_dst)
        ref_flashmla_metadata = torch.zeros_like(flashmla_metadata_dst)

    # Run individual copy operations (reference implementation)
    # 参考实现：逐个执行拷贝操作（与融合 kernel 等价的参考路径）
    ref_cache_seqlens.copy_(precomputed.cache_seqlens)
    ref_cu_seqlens_k[1:].copy_(precomputed.cu_seqlens_k[1:])  # 跳过 index 0（前缀和）

    if forward_mode.is_decode_or_idle():
        # Decode mode
        # decode 模式：拷贝页表和 NSA cache 序列长度
        ref_page_table_1[:, : precomputed.max_len].copy_(precomputed.page_indices)
        ref_nsa_cache_seqlens.copy_(precomputed.nsa_cache_seqlens)
    elif forward_mode.is_target_verify():
        # Target verify mode
        # target_verify 模式（MTP 目标序列验证）：还需拷贝 seqlens_expanded
        ref_page_table_1[:, : precomputed.max_seqlen_k].copy_(precomputed.page_indices)
        ref_nsa_seqlens_expanded.copy_(precomputed.seqlens_expanded)
        ref_nsa_cache_seqlens.copy_(precomputed.nsa_cache_seqlens)
    elif forward_mode.is_draft_extend():
        # Draft extend mode
        # draft_extend 模式（MTP draft 扩展）：按实际行/列大小拷贝
        rows = precomputed.page_indices.shape[0]
        cols = precomputed.max_seqlen_k
        ref_page_table_1[:rows, :cols].copy_(precomputed.page_indices)
        size = precomputed.seqlens_expanded_size
        ref_nsa_seqlens_expanded[:size].copy_(precomputed.seqlens_expanded)
        ref_nsa_cache_seqlens[:size].copy_(precomputed.nsa_cache_seqlens)

    # Copy NSA cu_seqlens
    # 拷贝 NSA 累计序列长度（前缀和），只拷贝有效区间 [1, 1+size]
    size = precomputed.seqlens_expanded_size
    ref_nsa_cu_seqlens_k[1 : 1 + size].copy_(precomputed.nsa_cu_seqlens_k[1 : 1 + size])

    # Copy real page table
    # 拷贝真实页表（稀疏注意力物理地址映射），只拷贝有效行/列
    if precomputed.real_page_table is not None:
        rows, cols = precomputed.real_page_table.shape
        ref_real_page_table[:rows, :cols].copy_(precomputed.real_page_table)

    # Copy FlashMLA metadata
    # 拷贝 FlashMLA 的 split 划分信息和 metadata
    if precomputed.flashmla_metadata is not None:
        size = precomputed.seqlens_expanded_size
        ref_flashmla_num_splits[: size + 1].copy_(flashmla_num_splits_src[: size + 1])
        ref_flashmla_metadata.copy_(flashmla_metadata_src)

    # Compare results and crash if inconsistent
    # 辅助函数：逐张量比对融合结果与参考结果，不一致时抛出详细错误信息
    def check_tensor_equal(name, fused, ref):
        if not torch.equal(fused, ref):
            max_diff = (fused.float() - ref.float()).abs().max().item()
            mismatched_elements = (fused != ref).sum().item()
            total_elements = fused.numel()
            raise RuntimeError(
                f"FUSED METADATA COPY VERIFICATION FAILED!\n"
                f"Tensor: {name}\n"
                f"Max difference: {max_diff}\n"
                f"Mismatched elements: {mismatched_elements}/{total_elements}\n"
                f"Fused shape: {fused.shape}, Ref shape: {ref.shape}\n"
                f"Forward mode: {forward_mode}, bs={bs}\n"
                f"The fused kernel produces different results than individual copies.\n"
                f"This indicates a bug in the fused metadata copy kernel."
            )

    # Verify all tensors (only compare the slices that were actually updated)
    # 验证基础张量：cache_seqlens 和 cu_seqlens_k
    check_tensor_equal("cache_seqlens", fused_cache_seqlens, ref_cache_seqlens)
    check_tensor_equal("cu_seqlens_k", fused_cu_seqlens_k, ref_cu_seqlens_k)

    # Compare page_table_1 only for the region that was actually updated
    # 仅比对实际被写入的 page_table_1 区域，避免未初始化区域干扰
    if forward_mode.is_decode_or_idle():
        check_tensor_equal(
            "page_table_1",
            fused_page_table_1[:, : precomputed.max_len],
            ref_page_table_1[:, : precomputed.max_len],
        )
    elif forward_mode.is_target_verify():
        check_tensor_equal(
            "page_table_1",
            fused_page_table_1[:, : precomputed.max_seqlen_k],
            ref_page_table_1[:, : precomputed.max_seqlen_k],
        )
    elif forward_mode.is_draft_extend():
        rows = precomputed.page_indices.shape[0]
        cols = precomputed.max_seqlen_k
        check_tensor_equal(
            "page_table_1",
            fused_page_table_1[:rows, :cols],
            ref_page_table_1[:rows, :cols],
        )

    # Compare nsa_cache_seqlens only for the region that was updated
    # 比对 NSA cache 序列长度的有效区域
    if forward_mode.is_decode_or_idle():
        check_tensor_equal(
            "nsa_cache_seqlens",
            fused_nsa_cache_seqlens,
            ref_nsa_cache_seqlens,
        )
    else:  # TARGET_VERIFY or DRAFT_EXTEND
        # target_verify/draft_extend 只比对前 size 个元素
        size = precomputed.seqlens_expanded_size
        check_tensor_equal(
            "nsa_cache_seqlens",
            fused_nsa_cache_seqlens[:size],
            ref_nsa_cache_seqlens[:size],
        )

    # Compare nsa_seqlens_expanded only for TARGET_VERIFY and DRAFT_EXTEND
    # MTP 场景下（非 decode）需要额外验证 seqlens_expanded
    if forward_mode.is_target_verify() or forward_mode.is_draft_extend():
        size = precomputed.seqlens_expanded_size
        check_tensor_equal(
            "nsa_seqlens_expanded",
            fused_nsa_seqlens_expanded[:size],
            ref_nsa_seqlens_expanded[:size],
        )

    # Compare nsa_cu_seqlens_k only for the region that was updated
    # 验证 NSA 累计序列长度的前缀和（有效区间）
    size = precomputed.seqlens_expanded_size
    check_tensor_equal(
        "nsa_cu_seqlens_k",
        fused_nsa_cu_seqlens_k[: 1 + size],
        ref_nsa_cu_seqlens_k[: 1 + size],
    )

    # 验证真实页表（如果存在）
    if precomputed.real_page_table is not None:
        rows, cols = precomputed.real_page_table.shape
        check_tensor_equal(
            "real_page_table",
            fused_real_page_table[:rows, :cols],
            ref_real_page_table[:rows, :cols],
        )

    # 验证 FlashMLA 元数据（如果存在）
    if precomputed.flashmla_metadata is not None:
        size = precomputed.seqlens_expanded_size
        check_tensor_equal(
            "flashmla_num_splits",
            fused_flashmla_num_splits[: size + 1],
            ref_flashmla_num_splits[: size + 1],
        )
        check_tensor_equal(
            "flashmla_metadata",
            fused_flashmla_metadata,
            ref_flashmla_metadata,
        )


def verify_multi_backend_fused_metadata_copy(
    metadata0,
    metadata1,
    metadata2,
    precomputed,
    bs,
    flashmla_num_splits_src=None,
    flashmla_metadata_src=None,
):
    """
    Verify that the multi-backend fused metadata copy kernel produces the same results
    as individual copies for all three backends.

    Args:
        metadata0: The NSA metadata object for backend 0
        metadata1: The NSA metadata object for backend 1
        metadata2: The NSA metadata object for backend 2
        precomputed: The precomputed metadata containing source tensors
        bs: Batch size
        flashmla_num_splits_src: Source FlashMLA num_splits tensor (optional)
        flashmla_metadata_src: Source FlashMLA metadata tensor (optional)

    Raises:
        RuntimeError: If verification fails (tensors don't match)
    """
    # 多 backend 融合拷贝验证：同时验证 3 个 backend 的元数据拷贝结果（仅 decode 模式）
    # Clone destination tensors to preserve fused kernel results
    # 依次克隆 3 个 backend 的目标张量，保存融合 kernel 的写入结果
    fused_results = []
    for idx, metadata in enumerate([metadata0, metadata1, metadata2]):
        fused_cache_seqlens = metadata.cache_seqlens_int32.clone()
        fused_cu_seqlens_k = metadata.cu_seqlens_k.clone()
        fused_page_table_1 = metadata.page_table_1.clone()
        fused_nsa_cache_seqlens = metadata.nsa_cache_seqlens_int32.clone()
        fused_nsa_cu_seqlens_k = metadata.nsa_cu_seqlens_k.clone()
        fused_real_page_table = (
            metadata.real_page_table.clone()
            if precomputed.real_page_table is not None
            else None
        )
        fused_flashmla_num_splits = None
        fused_flashmla_metadata = None
        if precomputed.flashmla_metadata is not None:
            # 克隆 FlashMLA 元数据（每个 backend 独立存储）
            fused_flashmla_num_splits = metadata.flashmla_metadata.num_splits.clone()
            fused_flashmla_metadata = (
                metadata.flashmla_metadata.flashmla_metadata.clone()
            )

        fused_results.append(
            {
                "cache_seqlens": fused_cache_seqlens,
                "cu_seqlens_k": fused_cu_seqlens_k,
                "page_table_1": fused_page_table_1,
                "nsa_cache_seqlens": fused_nsa_cache_seqlens,
                "nsa_cu_seqlens_k": fused_nsa_cu_seqlens_k,
                "real_page_table": fused_real_page_table,
                "flashmla_num_splits": fused_flashmla_num_splits,
                "flashmla_metadata": fused_flashmla_metadata,
            }
        )

    # Run individual copy operations for each backend (reference implementation)
    # 对每个 backend 分别执行参考实现（单独拷贝），生成比对基准
    ref_results = []
    for idx in range(3):
        metadata = [metadata0, metadata1, metadata2][idx]

        # Create reference tensors (zeroed out)
        # 为当前 backend 创建全零参考张量
        ref_cache_seqlens = torch.zeros_like(metadata.cache_seqlens_int32)
        ref_cu_seqlens_k = torch.zeros_like(metadata.cu_seqlens_k)
        ref_page_table_1 = torch.zeros_like(metadata.page_table_1)
        ref_nsa_cache_seqlens = torch.zeros_like(metadata.nsa_cache_seqlens_int32)
        ref_nsa_cu_seqlens_k = torch.zeros_like(metadata.nsa_cu_seqlens_k)
        ref_real_page_table = (
            torch.zeros_like(metadata.real_page_table)
            if precomputed.real_page_table is not None
            else None
        )
        ref_flashmla_num_splits = None
        ref_flashmla_metadata = None
        if precomputed.flashmla_metadata is not None:
            ref_flashmla_num_splits = torch.zeros_like(
                metadata.flashmla_metadata.num_splits
            )
            ref_flashmla_metadata = torch.zeros_like(
                metadata.flashmla_metadata.flashmla_metadata
            )

        # Copy operations (decode mode)
        # 参考拷贝（多 backend 仅支持 decode 模式）
        ref_cache_seqlens.copy_(precomputed.cache_seqlens)
        ref_cu_seqlens_k[1:].copy_(precomputed.cu_seqlens_k[1:])
        ref_page_table_1[:, : precomputed.max_len].copy_(precomputed.page_indices)
        ref_nsa_cache_seqlens.copy_(precomputed.nsa_cache_seqlens)

        # Copy NSA cu_seqlens
        # 拷贝 NSA 累计序列长度（前缀和）有效区间
        size = precomputed.seqlens_expanded_size
        ref_nsa_cu_seqlens_k[1 : 1 + size].copy_(
            precomputed.nsa_cu_seqlens_k[1 : 1 + size]
        )

        # Copy real page table
        # 拷贝真实页表（如果存在）
        if precomputed.real_page_table is not None:
            rows, cols = precomputed.real_page_table.shape
            ref_real_page_table[:rows, :cols].copy_(precomputed.real_page_table)

        # Copy FlashMLA metadata
        # 拷贝 FlashMLA split 和 metadata（如果存在）
        if precomputed.flashmla_metadata is not None:
            ref_flashmla_num_splits[: size + 1].copy_(
                flashmla_num_splits_src[: size + 1]
            )
            ref_flashmla_metadata.copy_(flashmla_metadata_src)

        ref_results.append(
            {
                "cache_seqlens": ref_cache_seqlens,
                "cu_seqlens_k": ref_cu_seqlens_k,
                "page_table_1": ref_page_table_1,
                "nsa_cache_seqlens": ref_nsa_cache_seqlens,
                "nsa_cu_seqlens_k": ref_nsa_cu_seqlens_k,
                "real_page_table": ref_real_page_table,
                "flashmla_num_splits": ref_flashmla_num_splits,
                "flashmla_metadata": ref_flashmla_metadata,
            }
        )

    # Compare results for all 3 backends
    # 辅助函数：带 backend 编号的张量比对，不一致时报告详细错误
    def check_tensor_equal(backend_idx, name, fused, ref):
        if not torch.equal(fused, ref):
            max_diff = (fused.float() - ref.float()).abs().max().item()
            mismatched_elements = (fused != ref).sum().item()
            total_elements = fused.numel()
            raise RuntimeError(
                f"MULTI-BACKEND FUSED METADATA COPY VERIFICATION FAILED!\n"
                f"Backend: {backend_idx}\n"
                f"Tensor: {name}\n"
                f"Max difference: {max_diff}\n"
                f"Mismatched elements: {mismatched_elements}/{total_elements}\n"
                f"Fused shape: {fused.shape}, Ref shape: {ref.shape}\n"
                f"Batch size: {bs}\n"
                f"The multi-backend fused kernel produces different results than individual copies.\n"
                f"This indicates a bug in the fused metadata copy kernel."
            )

    # Verify all tensors for all 3 backends (multi-backend is DECODE mode only)
    # 逐 backend 验证所有元数据张量（多 backend 仅支持 decode 模式）
    for idx in range(3):
        fused = fused_results[idx]
        ref = ref_results[idx]

        check_tensor_equal(
            idx,
            "cache_seqlens",
            fused["cache_seqlens"],
            ref["cache_seqlens"],
        )
        check_tensor_equal(
            idx,
            "cu_seqlens_k",
            fused["cu_seqlens_k"],
            ref["cu_seqlens_k"],
        )
        # Multi-backend is DECODE mode only, so compare only [:, :max_len]
        # decode 模式：只比对页表有效列
        check_tensor_equal(
            idx,
            "page_table_1",
            fused["page_table_1"][:, : precomputed.max_len],
            ref["page_table_1"][:, : precomputed.max_len],
        )
        check_tensor_equal(
            idx,
            "nsa_cache_seqlens",
            fused["nsa_cache_seqlens"],
            ref["nsa_cache_seqlens"],
        )
        # DECODE mode uses bs for nsa_cu_seqlens_k size
        # decode 模式下有效大小为 bs
        check_tensor_equal(
            idx,
            "nsa_cu_seqlens_k",
            fused["nsa_cu_seqlens_k"][: bs + 1],
            ref["nsa_cu_seqlens_k"][: bs + 1],
        )

        if precomputed.real_page_table is not None:
            rows, cols = precomputed.real_page_table.shape
            check_tensor_equal(
                idx,
                "real_page_table",
                fused["real_page_table"][:rows, :cols],
                ref["real_page_table"][:rows, :cols],
            )

        if precomputed.flashmla_metadata is not None:
            # DECODE mode uses bs + 1 for flashmla_num_splits
            # decode 模式：num_splits 有效区间为 [0, bs+1)
            check_tensor_equal(
                idx,
                "flashmla_num_splits",
                fused["flashmla_num_splits"][: bs + 1],
                ref["flashmla_num_splits"][: bs + 1],
            )
            check_tensor_equal(
                idx,
                "flashmla_metadata",
                fused["flashmla_metadata"],
                ref["flashmla_metadata"],
            )
