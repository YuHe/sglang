"""
Fused Triton kernel for Mamba state scatter operations.

This kernel replaces the expensive advanced indexing operations in
`update_mamba_state_after_mtp_verify` with a single fused gather-scatter kernel,
avoiding multiple `index_elementwise_kernel` launches.
"""
# Mamba 状态散射融合 Triton kernel：用单个 GPU kernel 替代多次 advanced indexing 操作

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_mamba_state_scatter_with_mask_kernel(
    src_ptr,
    dst_ptr,
    # Raw index arrays (before index_select)
    # 原始目标索引数组（对应 state_indices_tensor），长度为 total_requests
    dst_indices_raw_ptr,  # [total_requests] - state_indices_tensor
    # 原始步骤索引数组（对应 accepted_steps 或 mamba_steps_to_track），长度为 total_requests
    step_indices_raw_ptr,  # [total_requests] - accepted_steps or mamba_steps_to_track
    # Total number of requests
    total_requests,
    # 每条缓存线（layer, cache_entry）包含的元素数（constexpr 供编译期优化）
    elem_per_entry: tl.constexpr,
    src_layer_stride,
    src_req_stride,
    src_step_stride,
    dst_layer_stride,
    dst_req_stride,
    src_req_size,
    src_step_size,
    dst_req_size,
    # 每个 block 处理的元素数（constexpr）
    BLOCK_SIZE: tl.constexpr,
):
    """
    融合 gather-scatter kernel，内置 mask 过滤无效请求。

    Fused gather-scatter kernel with built-in masking.

    This kernel fuses the index_select operations by:
    1. Iterating over all requests (pid_req from 0 to total_requests-1)
    2. Checking if step_indices_raw[pid_req] >= 0 (valid mask)
    3. If valid, performing the scatter:
       dst[l, dst_indices_raw[pid_req], :] = src[l, pid_req, step_indices_raw[pid_req], :]

    Grid: (total_requests, num_layers, ceil(elem_per_entry / BLOCK_SIZE))
    """
    # 获取当前 thread block 对应的请求/层/元素块索引
    pid_req = tl.program_id(0)
    pid_layer = tl.program_id(1).to(tl.int64)
    pid_block = tl.program_id(2).to(tl.int64)

    # Load step index to check validity (step >= 0 means valid)
    # 加载步骤索引，用于判断当前请求是否有效（step >= 0 为有效）
    step_idx = tl.load(step_indices_raw_ptr + pid_req).to(tl.int64)

    # Early exit if this request is not valid (step < 0)
    # 若步骤索引为负，该请求无效，直接退出
    if step_idx < 0:
        return

    # Load destination index
    # 加载目标缓存槽位索引
    dst_idx = tl.load(dst_indices_raw_ptr + pid_req).to(tl.int64)

    # Source index is just the request index itself
    # 源索引即为当前请求编号
    src_idx = pid_req

    # Bounds check to avoid illegal memory access
    # 越界检查，防止非法内存访问
    if not (
        (dst_idx >= 0)
        & (dst_idx < dst_req_size)
        & (src_idx >= 0)
        & (src_idx < src_req_size)
        & (step_idx < src_step_size)
    ):
        return

    # Compute base offsets
    # 计算源张量中的线性偏移（以元素为单位）
    src_offset = (
        pid_layer * src_layer_stride
        + src_idx * src_req_stride
        + step_idx * src_step_stride
    )
    # 计算目标张量中的线性偏移
    dst_offset = pid_layer * dst_layer_stride + dst_idx * dst_req_stride

    # Compute element range for this block
    # 计算本 block 负责的元素下标范围
    start = pid_block * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    # 过滤超出 elem_per_entry 的越界元素
    mask = offsets < elem_per_entry

    # Load from source and store to destination
    # 从 src 读取数据并写入 dst（GPU 融合 gather-scatter）
    data = tl.load(src_ptr + src_offset + offsets, mask=mask)
    tl.store(dst_ptr + dst_offset + offsets, data, mask=mask)


def fused_mamba_state_scatter_with_mask(
    dst: torch.Tensor,  # [num_layers, cache_size, *state_shape]
    src: torch.Tensor,  # [num_layers, spec_size, draft_tokens, *state_shape]
    dst_indices_raw: torch.Tensor,  # [total_requests] - raw indices (e.g., state_indices_tensor)
    step_indices_raw: torch.Tensor,  # [total_requests] - raw step indices (step >= 0 means valid)
):
    """
    融合 gather-scatter 操作的 Python 入口，含内置 mask 过滤，用于 Mamba 状态更新。

    Fully fused gather-scatter with built-in masking for mamba state updates.

    This function fuses the following operations into a single kernel:
    1. valid_mask = step_indices_raw >= 0
    2. valid_indices = valid_mask.nonzero()
    3. dst_indices = dst_indices_raw[valid_indices]  (index_select)
    4. step_indices = step_indices_raw[valid_indices]  (index_select)
    5. for each valid i: dst[:, dst_indices[i], :] = src[:, i, step_indices[i], :]

    Args:
        dst: Destination tensor [num_layers, cache_size, *state_shape]
        src: Source tensor [num_layers, spec_size, draft_tokens, *state_shape]
        dst_indices_raw: Raw destination indices for all requests [total_requests]
        step_indices_raw: Raw step indices; entry >= 0 means valid [total_requests]
    """
    # 请求数为 0 时直接返回
    total_requests = step_indices_raw.shape[0]
    if total_requests == 0:
        return

    # 检查 src 和 dst 必须在同一 CUDA 设备上
    if dst.device != src.device:
        raise ValueError(
            f"dst and src must be on the same device. {dst.device=} {src.device=}"
        )
    if not dst.is_cuda or not src.is_cuda:
        raise ValueError(
            "fused_mamba_state_scatter_with_mask only supports CUDA tensors."
        )
    # 检查张量维度合法性
    if dst.ndim < 2 or src.ndim < 3:
        raise ValueError(f"Unexpected tensor ranks: {dst.ndim=} {src.ndim=}")
    # 检查 num_layers 维度一致
    if dst.shape[0] != src.shape[0]:
        raise ValueError(
            f"Layer dimension mismatch: {dst.shape[0]=} vs {src.shape[0]=}"
        )
    # 检查状态形状（尾部维度）一致
    if dst.shape[2:] != src.shape[3:]:
        raise ValueError(
            f"Trailing dims mismatch: {dst.shape[2:]=} vs {src.shape[3:]=}"
        )
    # 索引张量必须为 1D
    if dst_indices_raw.ndim != 1 or step_indices_raw.ndim != 1:
        raise ValueError(
            f"indices must be 1D: {dst_indices_raw.shape=} {step_indices_raw.shape=}"
        )
    # 两个索引张量长度必须相同
    if dst_indices_raw.shape[0] != step_indices_raw.shape[0]:
        raise ValueError(
            f"indices length mismatch: {dst_indices_raw.shape[0]=} vs {step_indices_raw.shape[0]=}"
        )

    # 提取维度信息
    num_layers = dst.shape[0]
    src_req_size = src.shape[1]
    src_step_size = src.shape[2]
    dst_req_size = dst.shape[1]

    # Flatten trailing dimensions: number of elements per (layer, cache_line) entry.
    # 计算每条缓存线（除 layer 和 req 维度外）的元素总数
    elem_per_entry = dst.numel() // (dst.shape[0] * dst.shape[1])

    # Get strides (in elements, not bytes)
    # 获取 src/dst 各维度的步长（以元素数为单位）
    src_layer_stride = src.stride(0)
    src_req_stride = src.stride(1)
    src_step_stride = src.stride(2)
    dst_layer_stride = dst.stride(0)
    dst_req_stride = dst.stride(1)

    # Ensure indices are int32 and contiguous
    # 将索引转为 int32 并保证连续性（Triton kernel 要求）
    dst_indices_raw = dst_indices_raw.to(torch.int32).contiguous()
    step_indices_raw = step_indices_raw.to(torch.int32).contiguous()

    # Ensure tensors are contiguous
    # src/dst 必须连续
    if not dst.is_contiguous():
        raise ValueError("dst tensor must be contiguous")
    if not src.is_contiguous():
        raise ValueError("src tensor must be contiguous")

    # Block size for copying elements
    # 每个 Triton block 处理 1024 个元素
    BLOCK_SIZE = 1024

    # Grid over all requests - invalid ones will early-exit in the kernel
    # kernel grid：(总请求数, 层数, 元素块数)；无效请求在 kernel 内 early-exit
    grid = (total_requests, num_layers, triton.cdiv(elem_per_entry, BLOCK_SIZE))

    # 启动 Triton kernel
    _fused_mamba_state_scatter_with_mask_kernel[grid](
        src,
        dst,
        dst_indices_raw,
        step_indices_raw,
        total_requests,
        elem_per_entry,
        src_layer_stride,
        src_req_stride,
        src_step_stride,
        dst_layer_stride,
        dst_req_stride,
        src_req_size,
        src_step_size,
        dst_req_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
