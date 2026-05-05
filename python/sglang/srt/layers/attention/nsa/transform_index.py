# NSA 稀疏 top-k 注意力的页表索引转换模块
# 将稀疏 top-k block 索引（逻辑位置）映射为实际 KV cache 页表中的物理地址
from typing import List, Optional

import torch
import triton
import triton.language as tl


def transform_index_page_table_prefill(**kwargs):
    # prefill 阶段的页表索引转换，当前默认使用参考实现（ref）
    return transform_index_page_table_prefill_ref(**kwargs)


def transform_index_page_table_decode(**kwargs):
    # decode 阶段的页表索引转换，当前默认使用参考实现（ref）
    return transform_index_page_table_decode_ref(**kwargs)


@triton.jit
def transform_index_page_table_decode_kernel(
    # Triton GPU kernel：将每个请求的 top-k 稀疏索引转换为实际页表地址
    page_table_ptr: torch.Tensor,    # 输入：页表，形状 [bs, max_seqlen_k]
    topk_indices_ptr: torch.Tensor,  # 输入：top-k 稀疏索引，形状 [bs, TOPK]
    result_ptr: torch.Tensor,        # 输出：转换后的物理地址，形状 [bs, TOPK]
    page_size: tl.constexpr,         # 页大小（目前仅支持 1）
    max_seqlen_k: tl.constexpr,      # KV 序列最大长度（页表列数）
):
    TOPK: tl.constexpr = 2048  # top-k 稀疏注意力中的 K 值上限，固定为 2048
    # 每个程序实例处理一个请求（batch item）
    req_id = tl.program_id(0)
    # 偏移到当前请求在页表和 top-k 索引中的起始位置
    page_table_ptr = page_table_ptr + req_id * max_seqlen_k
    topk_indices_ptr = topk_indices_ptr + req_id * TOPK
    result_ptr = result_ptr + req_id * TOPK

    offset = tl.arange(0, TOPK)  # topk should be 2048
    # 批量读取当前请求的所有 top-k 稀疏索引
    loaded_topk_indices = tl.load(topk_indices_ptr + offset)
    # mask 标记有效索引（-1 表示无效/填充位置）
    mask = loaded_topk_indices >= 0
    # 通过稀疏索引间接寻址，从页表中读取对应的物理 KV 地址
    loaded_kv_indices = tl.load(page_table_ptr + loaded_topk_indices, mask=mask)
    # 将有效映射结果写回，无效位置填 -1
    tl.store(result_ptr + offset, loaded_kv_indices, mask=mask)
    tl.store(result_ptr + offset, -1, mask=~mask)


def transform_index_page_table_decode_fast(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    result: Optional[torch.Tensor] = None,
    page_size: int = 1,
) -> torch.Tensor:
    """
    Transform the page table according to topk indices for sparse topk attention.
    Args:
        page_table: [qo_len, max_seqlen_k], the original page table
        topk_indices: [qo_len, topk], the topk indices for each query position
    Returns:
        transformed_page_table: [qo_len, topk], the transformed page table
        For out-of-bound indices in topk_indices, this should be filled with -1.
    """
    # decode 阶段 Triton 加速版：将 top-k 稀疏索引转换为物理 KV 地址
    assert page_size == 1  # 目前仅支持 page_size=1
    assert page_table.shape[0] == topk_indices.shape[0]
    assert topk_indices.shape[1] == 2048  # top-k 固定为 2048
    qo_len = topk_indices.shape[0]  # 查询序列总长（batch size）
    max_seqlen_k = page_table.shape[1]  # KV 序列最大长度
    if result is None:
        # 分配输出张量，与 topk_indices 形状相同
        result = torch.empty_like(topk_indices, dtype=torch.int32)
    # Launch triton kernel
    # 每个请求对应一个 Triton program，并行处理所有 batch
    grid = (qo_len,)
    transform_index_page_table_decode_kernel[grid](
        page_table,
        topk_indices,
        result,
        page_size,
        max_seqlen_k=max_seqlen_k,
    )
    return result


def transform_index_page_table_prefill_fast(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    extend_lens_cpu: List[int],
    page_size: int = 1,
) -> torch.Tensor:
    # prefill 阶段 Triton 加速版：逐序列调用 decode_fast 完成索引转换
    # TODO(baizhou): can be implemented with another triton kernel
    assert page_size == 1
    result = torch.empty_like(topk_indices, dtype=torch.int32)
    assert len(extend_lens_cpu) == page_table.shape[0]
    offset = 0
    for i, l in enumerate(extend_lens_cpu):
        # 对第 i 条序列，将其页表行扩展到 l 行后，与对应的 top-k 索引做转换
        transform_index_page_table_decode_fast(
            page_table[i].unsqueeze(0).expand(l, -1),
            topk_indices[offset : offset + l],
            result=result[offset : offset + l],
        )
        offset += l
    assert offset == topk_indices.shape[0]
    return result


def transform_index_page_table_decode_ref(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    result: Optional[torch.Tensor] = None,
    page_size: int = 1,
) -> torch.Tensor:
    # decode 阶段参考实现：使用 torch.gather 将 top-k 稀疏索引映射到物理 KV 地址
    assert page_size == 1
    assert page_table.shape[0] == topk_indices.shape[0]
    if result is None:
        result = torch.empty_like(topk_indices, dtype=torch.int32)
    assert result.shape == topk_indices.shape
    # torch.gather：按 topk_indices（clamp 到非负）从 page_table 的 dim=1 取值
    torch.gather(
        page_table.to(result.dtype),
        dim=1,
        index=topk_indices.clamp(min=0),
        out=result,
    )
    # 原始索引为 -1 的位置（无效 top-k 位）在结果中填 -1
    result[topk_indices < 0] = -1
    return result


def transform_index_page_table_prefill_ref(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    extend_lens_cpu: List[int],
    page_size: int = 1,
) -> torch.Tensor:
    # prefill 阶段参考实现：逐序列调用 decode_ref 完成索引转换
    assert page_size == 1
    result = torch.empty_like(topk_indices, dtype=torch.int32)
    assert len(extend_lens_cpu) == page_table.shape[0]
    offset = 0
    for i, l in enumerate(extend_lens_cpu):
        # 对第 i 条序列，将其页表行扩展到 l 行，与对应 top-k 索引做参考转换
        transform_index_page_table_decode_ref(
            page_table[i].unsqueeze(0).expand(l, -1),
            topk_indices[offset : offset + l],
            result=result[offset : offset + l],
        )
        offset += l
    assert offset == topk_indices.shape[0]
    return result


if __name__ == "__main__":
    # 正确性验证：对比参考实现与 Triton 加速实现的结果是否一致
    bs, topk, max_seqlen = 10, 2048, 3000
    page_table = torch.randint(0, 100, (bs, max_seqlen), device="cuda")
    # 构造带 -1 填充的 top-k 索引（模拟部分无效位置）
    topk_indices = torch.full((bs, topk), -1, device="cuda")
    topk_indices[:, :1600] = torch.arange(1600).unsqueeze(0).repeat(bs, 1)
    ref_result = transform_index_page_table_decode_ref(page_table, topk_indices)
    result = transform_index_page_table_decode_fast(page_table, topk_indices)
    # 验证两种实现结果完全一致
    assert torch.all(result == ref_result)
    print("Passed")
