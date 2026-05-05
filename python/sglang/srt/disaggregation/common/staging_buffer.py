"""
GPU Staging Buffer for heterogeneous TP KV cache transfer.

When prefill attn_tp_size != decode attn_tp_size, the per-token RDMA approach
generates O(tokens * layers) small RDMA requests. This module provides a staging
buffer mechanism that gathers scattered head slices into contiguous GPU memory,
enabling bulk RDMA transfers that reduce request count to O(layers) or O(1).

Usage:
    Activated by setting SGLANG_DISAGG_STAGING_BUFFER=1.
"""
# Staging Buffer 模块：解决 prefill/decode 异构 TP（attn_tp_size 不一致）时的 KV 传输问题
# 核心思路：将分散的 KV head 切片先 gather 到连续 GPU 缓冲区，再做少量大 RDMA，降低 RDMA 请求数量

from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# TODO(yangminl): remove torch fallback implementations once the Triton kernels
# have been validated in production across all configurations.
# 根据环境变量决定是否使用 Triton 融合 kernel（默认开启），或回退到 torch.gather 路径
_USE_TRITON_STAGING = not bool(os.environ.get("SGLANG_STAGING_USE_TORCH", ""))



# _fused_gather_to_staging_kernel：Triton JIT 融合 gather kernel
# 将 KV pool 中分散的 page token 数据按 page_indices 汇聚到连续的 staging 缓冲区
# grid: (2*num_layers, ceil(per_layer_elems/BLOCK_SIZE))，每个 program 处理一个 layer 的 K 或 V
@triton.jit
def _fused_gather_to_staging_kernel(
    layer_ptrs,         # int64 指针数组：[k_layer0, k_layer1, ..., v_layer0, v_layer1, ...]
    page_indices,       # [num_pages] 每个页面对应的 KV pool 中的页索引
    staging,            # 连续 staging 缓冲区（目标）
    num_tokens,         # 本次传输的 token 数量
    stride_pool_token,  # KV pool 中每个 token 的步长（head_num * head_dim）
    head_offset,        # 本 TP rank 负责的 head 在池中的起始偏移（head_start * head_dim）
    per_layer_elems,    # 每层的总元素数（num_tokens * num_heads * head_dim）
    ELEMS_PER_TOKEN: tl.constexpr,  # 每个 token 的元素数（num_heads * head_dim）
    PAGE_SIZE: tl.constexpr,        # 每页 token 数（paged KV 页大小）
    BLOCK_SIZE: tl.constexpr,       # 每个 CUDA block 处理的元素数
):
    # program_id(0) 是层索引（含 K/V，共 2*num_layers 维），program_id(1) 是元素分块索引
    layer_id = tl.program_id(0)
    block_id = tl.program_id(1)

    # 加载当前层的 KV pool 基址指针（转换为目标数据类型）
    layer_ptr = tl.load(layer_ptrs + layer_id).to(staging.dtype)

    # 计算本 block 处理的全局元素偏移范围
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < per_layer_elems

    # 将线性 offset 分解为 token 内索引和 token 索引
    t_idx = offsets // ELEMS_PER_TOKEN  # 第几个 token
    e_idx = offsets % ELEMS_PER_TOKEN   # token 内的第几个元素

    # 通过 page_indices 将逻辑 token 索引映射到 KV pool 中的物理位置
    page_id = t_idx // PAGE_SIZE        # 第几页
    intra_page = t_idx % PAGE_SIZE      # 页内 token 偏移
    page_val = tl.load(page_indices + page_id, mask=mask, other=0)  # 物理页号
    pool_token = page_val * PAGE_SIZE + intra_page  # KV pool 中的实际 token 位置

    # 从 KV pool 中读取数据（加上 head 起始偏移）
    src_offsets = (
        pool_token * stride_pool_token.to(tl.int64) + head_offset.to(tl.int64) + e_idx
    )
    vals = tl.load(layer_ptr + src_offsets, mask=mask)

    # 写入到 staging 缓冲区：按层顺序排列，layer_id * per_layer_elems 为层基址
    dst_offsets = tl.program_id(0).to(tl.int64) * per_layer_elems.to(tl.int64) + offsets
    tl.store(staging + dst_offsets, vals, mask=mask)


# _fused_scatter_from_staging_kernel：Triton JIT 融合 scatter kernel
# 将连续 staging 缓冲区中的数据写回 decode 侧 KV pool（支持多 writer/多 prefill TP rank）
# grid: (num_writers * 2*num_layers, ceil(per_layer_elems/BLOCK_SIZE))
@triton.jit
def _fused_scatter_from_staging_kernel(
    layer_ptrs,           # int64 指针数组：[k_layer0, ..., v_layer0, ...]
    page_indices,         # [num_pages] decode 侧目标页索引
    staging,              # 连续 staging 缓冲区（源）
    writer_head_offsets,  # 每个 writer 在目标 KV pool 中的 head 起始字节偏移
    num_tokens,           # token 总数
    stride_pool_token,    # 目标 KV pool 每 token 步长
    per_layer_elems,      # 每层元素数
    ELEMS_PER_TOKEN: tl.constexpr,  # 每 token 元素数
    PAGE_SIZE: tl.constexpr,        # 页大小
    NUM_LAYERS_X2: tl.constexpr,    # 2 * num_layers（K+V）
    BLOCK_SIZE: tl.constexpr,       # block 大小
):
    # 从 program_id(0) 拆分出 writer_id（哪个 prefill TP rank）和 layer_kv_id（K/V 层索引）
    prog_id = tl.program_id(0)
    block_id = tl.program_id(1)

    writer_id = prog_id // NUM_LAYERS_X2   # 第几个 prefill writer（TP rank）
    layer_kv_id = prog_id % NUM_LAYERS_X2  # 第几层（K 和 V 各占一半）

    # 加载该 writer/layer 组合的 KV pool 目标基址和 head 偏移
    layer_ptr = tl.load(layer_ptrs + layer_kv_id).to(staging.dtype)
    head_offset = tl.load(writer_head_offsets + writer_id)

    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < per_layer_elems

    # 将线性 offset 分解为 token 和元素索引
    t_idx = offsets // ELEMS_PER_TOKEN
    e_idx = offsets % ELEMS_PER_TOKEN

    # 将逻辑 token 映射到 decode 侧 KV pool 物理位置
    page_id = t_idx // PAGE_SIZE
    intra_page = t_idx % PAGE_SIZE
    page_val = tl.load(page_indices + page_id, mask=mask, other=0)
    pool_token = page_val * PAGE_SIZE + intra_page

    # 计算 staging 缓冲区中该 writer/layer 对应的源偏移
    # staging 布局：[writer_id][layer_kv_id][token*elems]
    per_rank_elems = per_layer_elems.to(tl.int64) * NUM_LAYERS_X2
    src_offsets = (
        writer_id.to(tl.int64) * per_rank_elems
        + layer_kv_id.to(tl.int64) * per_layer_elems.to(tl.int64)
        + offsets
    )
    vals = tl.load(staging + src_offsets, mask=mask)

    # 写入目标 KV pool（加上该 writer 对应的 head 起始偏移）
    dst_offsets = (
        pool_token * stride_pool_token.to(tl.int64) + head_offset.to(tl.int64) + e_idx
    )
    tl.store(layer_ptr + dst_offsets, vals, mask=mask)


# StagingBuffer：预分配的 GPU staging 缓冲区，用于 KV 批量传输
# 支持使用自定义内存池（如 Mooncake NVLink cuMemCreate 内存），以兼容 MNNVL 传输
class StagingBuffer:
    """Pre-allocated GPU staging buffer for bulk KV transfer.

    When a custom_mem_pool is provided (e.g., mooncake NVLink allocator),
    the buffer is allocated within that pool so it's compatible with
    NVLink/MNNVL transport (requires cuMemCreate-backed memory).
    """

    def __init__(
        self,
        size_bytes: int,   # 缓冲区总字节数
        device: str,       # CUDA 设备字符串，如 "cuda:0"
        gpu_id: int,       # GPU 设备编号
        custom_mem_pool=None,  # 可选：Mooncake/NVLink 自定义内存池
    ):
        self.size_bytes = size_bytes
        self.device = device
        self.gpu_id = gpu_id

        # 确保在目标 GPU 上分配内存
        torch.cuda.set_device(gpu_id)
        if custom_mem_pool is not None:
            # 使用自定义内存池（cuMemCreate backed）分配，兼容 NVLink/MNNVL RDMA
            with torch.cuda.use_mem_pool(custom_mem_pool):
                self.buffer = torch.empty(size_bytes, dtype=torch.uint8, device=device)
            alloc_method = "custom_mem_pool (cuMemCreate)"
        else:
            # 普通 cudaMalloc 路径（不支持 NVLink RDMA）
            self.buffer = torch.empty(size_bytes, dtype=torch.uint8, device=device)
            alloc_method = "cudaMalloc"
        # 记录缓冲区基地址指针，用于 RDMA 注册
        self.data_ptr = self.buffer.data_ptr()

        logger.info(
            f"StagingBuffer allocated: {size_bytes / (1024*1024):.1f} MB "
            f"on {device}, method={alloc_method}, ptr=0x{self.data_ptr:x}"
        )

    def get_ptr(self) -> int:
        # 返回 staging 缓冲区基地址指针（uint64）
        return self.data_ptr

    def get_size(self) -> int:
        # 返回缓冲区总字节数
        return self.size_bytes

    def fits(self, required_bytes: int) -> bool:
        # 判断 required_bytes 是否能放入缓冲区（单次传输的容量检查）
        return required_bytes <= self.size_bytes


# StagingAllocator：decode 侧动态 staging 环形缓冲区分配器（支持过度提交）
# 用大块预分配 GPU buffer 作为环形缓冲区，每次请求分配 (alloc_id, offset, round) 三元组
# 安全性由 watermark 机制保证：prefill 在 RDMA 写入前检查 watermark 确认目标区域已释放
class StagingAllocator:
    """Decode-side dynamic staging ring buffer allocator with overcommit.

    One large pre-allocated GPU buffer used as a ring buffer. Each request
    gets a (alloc_id, offset, round) triple based on its actual byte
    requirement. Allocation (assign) is overcommit — it always succeeds
    as long as the request fits in the buffer. Overlap safety is enforced
    on the prefill side before RDMA, using a watermark that tracks the
    oldest un-freed allocation.

    The watermark (round, tail_offset) is periodically sent to prefill.
    Prefill transfer workers wait before writing if their target region
    overlaps with not-yet-freed data from a previous round.
    """

    # 永久分配失败标记：chunk 大小超过整个环形缓冲区总容量
    ALLOC_OVERSIZED = -2

    def __init__(
        self,
        total_size_bytes: int,  # 环形缓冲区总字节数
        device: str,            # CUDA 设备字符串
        gpu_id: int,            # GPU 设备编号
        custom_mem_pool=None,   # 可选自定义内存池（NVLink 兼容）
    ):
        # 创建底层 StagingBuffer（实际 GPU 内存分配）
        self.buffer = StagingBuffer(total_size_bytes, device, gpu_id, custom_mem_pool)
        self.total_size = total_size_bytes
        self.base_ptr = self.buffer.data_ptr   # 缓冲区基地址
        # head：环形缓冲区当前写入头部位置（字节偏移）
        self.head = 0
        # round：每次 head 回绕到 0 时递增的轮次计数，用于 watermark 比较
        self.round = 0
        # allocations: alloc_id -> (offset, size, round) 活跃分配记录
        self.allocations: dict = {}  # alloc_id -> (offset, size, round)
        # alloc_order：按分配顺序记录的 alloc_id 列表，用于找最旧活跃分配（watermark 推进）
        self.alloc_order: List[int] = []
        self.next_alloc_id = 0   # 单调递增的分配 ID 生成器
        # watermark：当前最旧未释放分配的轮次和偏移，prefill 以此判断写入安全性
        self.watermark_round = 0
        self.watermark_tail = 0
        # 线程锁：保护所有分配状态的并发访问
        self.lock = threading.Lock()

        logger.info(
            f"StagingAllocator (ring+overcommit): "
            f"{total_size_bytes / (1024*1024):.1f} MB "
            f"on {device}, ptr=0x{self.base_ptr:x}"
        )

    def assign(self, required_bytes: int) -> Optional[Tuple[int, int, int]]:
        """Allocate a region. Returns (alloc_id, offset, round) or None."""
        # 分配 staging 区域：从环形缓冲区尾部分配，回绕时递增 round
        with self.lock:
            if required_bytes > self.total_size:
                # 超过整个环形缓冲区容量，永久失败
                return None

            space_at_end = self.total_size - self.head
            if required_bytes <= space_at_end:
                # 尾部空间足够，直接在 head 处分配
                offset = self.head
                self.head += required_bytes
            else:
                # 尾部空间不足，回绕到缓冲区头部重新分配
                self.round += 1
                offset = 0
                self.head = required_bytes

            # 分配新 ID 并记录 (offset, size, round)
            alloc_id = self.next_alloc_id
            self.next_alloc_id += 1
            self.allocations[alloc_id] = (offset, required_bytes, self.round)
            self.alloc_order.append(alloc_id)
            return (alloc_id, offset, self.round)

    def free(self, alloc_id: int):
        """Free an allocation and advance watermark past consecutive freed entries."""
        # 释放指定分配，并向前推进 watermark（跳过所有已释放的头部分配）
        with self.lock:
            if alloc_id not in self.allocations:
                return
            self.allocations.pop(alloc_id)

            # 从 alloc_order 头部移除所有已释放的条目（有序推进 watermark）
            while self.alloc_order and self.alloc_order[0] not in self.allocations:
                self.alloc_order.pop(0)

            if not self.allocations:
                # 所有分配已释放：watermark 指向当前 head（整块缓冲区安全）
                self.watermark_round = self.round
                self.watermark_tail = self.head
            elif self.alloc_order:
                # watermark 推进到最旧活跃分配的起始位置
                off, _, rnd = self.allocations[self.alloc_order[0]]
                self.watermark_round = rnd
                self.watermark_tail = off

    def get_watermark(self) -> Tuple[int, int]:
        """Return (round, tail_offset). Everything before this is safe to write."""
        # 返回当前 watermark (round, tail_offset)，prefill 用此判断目标区域是否可安全写入
        with self.lock:
            return (self.watermark_round, self.watermark_tail)

    def get_ptr(self, alloc_id: int) -> int:
        # 根据 alloc_id 计算该分配区域的绝对 GPU 地址指针
        offset, _, _ = self.allocations[alloc_id]
        return self.base_ptr + offset

    def get_offset(self, alloc_id: int) -> int:
        # 返回该分配在环形缓冲区中的字节偏移
        offset, _, _ = self.allocations[alloc_id]
        return offset

    def get_round(self, alloc_id: int) -> int:
        # 返回该分配对应的回绕轮次（用于 prefill 侧 watermark 比较）
        _, _, rnd = self.allocations[alloc_id]
        return rnd

    def get_base_ptr(self) -> int:
        # 返回环形缓冲区基地址（用于 RDMA 注册）
        return self.base_ptr

    def get_total_size(self) -> int:
        # 返回环形缓冲区总字节数
        return self.total_size


# gather_kv_head_slices：torch.gather 辅助函数，将分散的 KV pool token 按 head 切片汇聚到 staging
# 使用 out= 参数直接写入目标 tensor，避免临时 tensor 分配（减少 CUDA 缓存分配器开销）
def gather_kv_head_slices(
    kv_buffer_tensor: torch.Tensor,   # [pool_size, head_num, head_dim]，单层 KV 缓冲区
    gather_idx: torch.Tensor,         # [num_tokens, num_heads, head_dim] int64，预计算的 gather 索引
    head_start: int,                  # 本 TP rank 负责的起始 head 索引
    num_heads: int,                   # 需要 gather 的 head 数量
    staging_tensor: torch.Tensor,     # 输出 tensor：[num_tokens, num_heads, head_dim]
):
    """Gather KV head slices from scattered pages into contiguous staging buffer.

    Uses torch.gather(out=) to write directly into staging_tensor without
    allocating temporary tensors (avoids CUDA caching allocator stalls).

    Args:
        kv_buffer_tensor: [pool_size, head_num, head_dim], one layer.
        gather_idx: [num_tokens, num_heads, head_dim] int64, pre-computed
            token indices expanded for gather on dim=0.
        head_start: Starting head index for the slice.
        num_heads: Number of heads to gather.
        staging_tensor: Output tensor, shape [num_tokens, num_heads, head_dim].
    """
    # 先切出本 TP rank 负责的 head 范围，再 gather（in-place 写入 staging_tensor）
    src = kv_buffer_tensor[:, head_start : head_start + num_heads, :]
    torch.gather(src, 0, gather_idx, out=staging_tensor)


# scatter_kv_head_slices：将连续 staging 中的数据写回 decode 侧 KV pool
# 支持 page_size>1 的 paged KV（通过广播扩展页内偏移实现向量化 scatter）
def scatter_kv_head_slices(
    staging_tensor: torch.Tensor,    # 来自 staging 缓冲区的连续数据
    kv_buffer_tensor: torch.Tensor,  # 目标 KV 缓冲区：[pool_size, head_num, head_dim]
    page_indices: torch.Tensor,      # [num_pages] 目标页索引（int32/int64）
    head_start: int,                 # 目标起始 head 索引
    num_heads: int,                  # 要写入的 head 数量
    page_size: int = 1,              # 每页 token 数（paged KV 时 >1）
):
    """Scatter KV head slices from contiguous staging buffer to KV cache.

    Args:
        staging_tensor: Input tensor from staging buffer (contiguous packed data).
        kv_buffer_tensor: The KV buffer for one layer, shape [pool_size, head_num, head_dim].
        page_indices: [num_pages] int32/int64 tensor of page indices.
        head_start: Starting head index for the slice.
        num_heads: Number of heads to scatter.
        page_size: Number of tokens per page.
    """
    head_dim = kv_buffer_tensor.shape[-1]
    if page_size == 1:
        # page_size==1：token_indices 直接等于 page_indices
        num_tokens = page_indices.shape[0]
        data = staging_tensor.reshape(num_tokens, num_heads, head_dim)
        kv_buffer_tensor[page_indices, head_start : head_start + num_heads, :] = data
    else:
        # page_size>1：将页索引展开为 token 索引（page_idx * page_size + [0..page_size-1]）
        num_tokens = page_indices.shape[0] * page_size
        offsets = torch.arange(page_size, device=page_indices.device)
        token_indices = (page_indices.unsqueeze(1) * page_size + offsets).reshape(-1)
        data = staging_tensor.reshape(num_tokens, num_heads, head_dim)
        kv_buffer_tensor[token_indices, head_start : head_start + num_heads, :] = data


# _gather_all_layers_torch：torch.gather 路径的全层 gather 实现（回退路径）
# 逐层执行 gather，写入 staging 缓冲区；使用独立 _gather_stream 与默认流并行执行
def _gather_all_layers_torch(
    k_buffers: list,           # 所有层的 K 缓冲区列表
    v_buffers: list,           # 所有层的 V 缓冲区列表
    page_indices_np,           # numpy 页索引数组（将转换为 GPU tensor）
    staging_buffer: StagingBuffer,  # 目标 staging 缓冲区
    src_head_start: int,       # 源侧起始 head 索引（本 TP rank 的 head 范围起点）
    num_heads: int,            # 需要 gather 的 head 数量
    page_size: int,            # 每页 token 数
    gpu_id: int,               # GPU 设备编号
) -> int:
    """torch.gather path: zero per-layer allocation, one kernel per layer."""
    import numpy as np

    num_layers = len(k_buffers)
    head_dim = k_buffers[0].shape[-1]
    dtype_size = k_buffers[0].element_size()
    num_tokens = len(page_indices_np) * page_size
    # 每层单 K 或 V 的字节数（所有 gather 的 head × head_dim × dtype_size）
    per_layer_bytes = num_tokens * num_heads * head_dim * dtype_size

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    # 将 numpy 页索引转换为 GPU int64 tensor
    page_idx_tensor = torch.from_numpy(page_indices_np.astype(np.int64)).to(device)

    if page_size == 1:
        # 无页内偏移：直接用页索引作为 token 索引
        token_indices = page_idx_tensor
    else:
        # 展开为连续 token 索引序列
        offsets = torch.arange(page_size, device=device)
        token_indices = (page_idx_tensor.unsqueeze(1) * page_size + offsets).reshape(-1)

    # 预计算 gather 索引（广播到 [num_tokens, num_heads, head_dim]），避免循环内重复计算
    gather_idx = token_indices.view(-1, 1, 1).expand(num_tokens, num_heads, head_dim)

    # 创建独立 gather stream（如不存在），与默认流异步执行 gather
    if not hasattr(staging_buffer, "_gather_stream"):
        staging_buffer._gather_stream = torch.cuda.Stream(device=device)

    # 等待默认流上的前驱操作完成，确保 KV 数据已写入
    staging_buffer._gather_stream.wait_stream(
        torch.cuda.default_stream(torch.device(device))
    )

    staging_view = staging_buffer.buffer
    offset = 0
    with torch.cuda.stream(staging_buffer._gather_stream):
        # 先 gather 所有层的 K，再 gather 所有层的 V（K/V 交替排列在 staging 中）
        for layer_id in range(num_layers):
            dst = (
                staging_view[offset : offset + per_layer_bytes]
                .view(k_buffers[layer_id].dtype)
                .reshape(num_tokens, num_heads, head_dim)
            )
            gather_kv_head_slices(
                k_buffers[layer_id],
                gather_idx,
                src_head_start,
                num_heads,
                dst,
            )
            offset += per_layer_bytes
        for layer_id in range(num_layers):
            dst = (
                staging_view[offset : offset + per_layer_bytes]
                .view(v_buffers[layer_id].dtype)
                .reshape(num_tokens, num_heads, head_dim)
            )
            gather_kv_head_slices(
                v_buffers[layer_id],
                gather_idx,
                src_head_start,
                num_heads,
                dst,
            )
            offset += per_layer_bytes

    # 同步 gather stream，确保所有数据已写入 staging 再返回
    staging_buffer._gather_stream.synchronize()
    return offset


# _gather_all_layers_triton：Triton 融合 kernel 路径的全层 gather 实现（默认路径）
# 将所有层 K/V 的 head 切片在单次 kernel 中 gather 到 staging，减少 kernel launch 次数
def _gather_all_layers_triton(
    k_buffers: list,
    v_buffers: list,
    page_indices_np,
    staging_buffer: StagingBuffer,
    src_head_start: int,
    num_heads: int,
    page_size: int,
    gpu_id: int,
) -> int:
    """Triton fused kernel path: single kernel launch for all layers."""
    import numpy as np

    num_layers = len(k_buffers)
    head_dim = k_buffers[0].shape[-1]
    # total_heads：KV pool 中每个 token 的完整 head 数（用于计算 stride）
    total_heads = k_buffers[0].shape[1]
    dtype_size = k_buffers[0].element_size()
    num_tokens = len(page_indices_np) * page_size
    # 每 token 的元素数（仅 gather 的 head 范围）
    elems_per_token = num_heads * head_dim
    per_layer_elems = num_tokens * elems_per_token
    per_layer_bytes = per_layer_elems * dtype_size
    # K + V 两组，共 2*num_layers 层
    total_bytes = per_layer_bytes * num_layers * 2

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    page_idx_tensor = torch.from_numpy(page_indices_np.astype(np.int64)).to(device)

    # 将所有层的 K/V data_ptr 打包为 int64 tensor，传给 Triton kernel
    layer_ptrs = torch.tensor(
        [buf.data_ptr() for buf in k_buffers] + [buf.data_ptr() for buf in v_buffers],
        dtype=torch.int64,
        device=device,
    )
    # 选择与元素大小匹配的整数类型（bit-preserving 拷贝，不做类型转换）
    int_dtype_map = {1: torch.int8, 2: torch.int16, 4: torch.int32}
    int_dtype = int_dtype_map.get(dtype_size, torch.int16)
    staging_typed = staging_buffer.buffer[:total_bytes].view(int_dtype)

    if not hasattr(staging_buffer, "_gather_stream"):
        staging_buffer._gather_stream = torch.cuda.Stream(device=device)

    staging_buffer._gather_stream.wait_stream(
        torch.cuda.default_stream(torch.device(device))
    )

    # grid: (2*num_layers, ceil(per_layer_elems/BLOCK_SIZE))
    # 每个 program_id(0) 对应一个层（K/V 各一半）
    BLOCK_SIZE = 1024
    grid = (2 * num_layers, triton.cdiv(per_layer_elems, BLOCK_SIZE))

    with torch.cuda.stream(staging_buffer._gather_stream):
        _fused_gather_to_staging_kernel[grid](
            layer_ptrs,
            page_idx_tensor,
            staging_typed,
            num_tokens,
            total_heads * head_dim,  # stride_pool_token：每 token 的完整步长
            src_head_start * head_dim,  # head_offset：本 TP rank 的 head 起始字节偏移
            per_layer_elems,
            elems_per_token,
            page_size,
            BLOCK_SIZE,
        )

    # 同步 gather stream 确保数据全部写入 staging
    staging_buffer._gather_stream.synchronize()
    return total_bytes


# gather_all_layers_to_staging：统一入口，根据 _USE_TRITON_STAGING 分派 Triton 或 torch 路径
def gather_all_layers_to_staging(
    k_buffers: list,
    v_buffers: list,
    page_indices_np,
    staging_buffer: StagingBuffer,
    src_head_start: int,
    num_heads: int,
    page_size: int,
    gpu_id: int,
) -> int:
    """Gather all layers' K and V head slices into a staging buffer.

    Returns total bytes written.
    Dispatches to Triton fused kernel when available, falls back to torch.gather.
    """
    # 优先使用 Triton 融合 kernel（更高效），否则回退到逐层 torch.gather
    if _USE_TRITON_STAGING:
        return _gather_all_layers_triton(
            k_buffers,
            v_buffers,
            page_indices_np,
            staging_buffer,
            src_head_start,
            num_heads,
            page_size,
            gpu_id,
        )
    return _gather_all_layers_torch(
        k_buffers,
        v_buffers,
        page_indices_np,
        staging_buffer,
        src_head_start,
        num_heads,
        page_size,
        gpu_id,
    )


# _scatter_staging_to_kv_torch：torch 路径的 staging → KV pool scatter（回退路径）
# 支持 prefill_attn_tp_size > decode_attn_tp_size 时多 writer 的情况（每个 writer 负责不同 head 范围）
def _scatter_staging_to_kv_torch(
    staging_buffer_view: torch.Tensor,  # staging 缓冲区视图（uint8 字节序列）
    k_buffers: list,                    # decode 侧所有层 K 缓冲区列表
    v_buffers: list,                    # decode 侧所有层 V 缓冲区列表
    page_idx_tensor: torch.Tensor,      # [num_pages] decode 侧目标页索引（GPU tensor）
    page_size: int,                     # 每页 token 数
    prefill_attn_tp_size: int,          # prefill 侧 attention TP 并行度
    decode_attn_tp_size: int,           # decode 侧 attention TP 并行度
    dst_tp_rank: int,                   # 目标（decode）TP rank
    total_kv_heads: int,                # 全局 KV head 总数
) -> None:
    """torch path for scatter."""
    num_layers = len(k_buffers)
    head_dim = k_buffers[0].shape[-1]
    dtype_size = k_buffers[0].element_size()
    num_tokens = page_idx_tensor.shape[0] * page_size

    # 计算 writer 数：prefill TP > decode TP 时有多个 writer；否则只有 1 个
    if prefill_attn_tp_size > decode_attn_tp_size:
        num_writers = prefill_attn_tp_size // max(1, decode_attn_tp_size)
    else:
        num_writers = 1

    # 逐 writer 处理：每个 writer 贡献不同的 head 范围
    for writer_rank in range(num_writers):
        # 计算该 writer 对应的 head 切片参数（源起点、head数、目标起点）
        _, num_heads, dst_head_start, _ = compute_head_slice_params(
            prefill_attn_tp_size,
            decode_attn_tp_size,
            writer_rank,
            dst_tp_rank,
            total_kv_heads,
        )
        per_layer_bytes = num_tokens * num_heads * head_dim * dtype_size
        # staging 中每个 rank 的字节段大小（K + V 各 num_layers 层）
        per_rank_bytes = per_layer_bytes * num_layers * 2
        rank_base = writer_rank * per_rank_bytes

        # 先 scatter 所有层 K，再 scatter 所有层 V
        offset = rank_base
        for layer_id in range(num_layers):
            layer_data = (
                staging_buffer_view[offset : offset + per_layer_bytes]
                .view(k_buffers[layer_id].dtype)
                .reshape(num_tokens, num_heads, head_dim)
            )
            scatter_kv_head_slices(
                layer_data,
                k_buffers[layer_id],
                page_idx_tensor,
                dst_head_start,
                num_heads,
                page_size,
            )
            offset += per_layer_bytes
        for layer_id in range(num_layers):
            layer_data = (
                staging_buffer_view[offset : offset + per_layer_bytes]
                .view(v_buffers[layer_id].dtype)
                .reshape(num_tokens, num_heads, head_dim)
            )
            scatter_kv_head_slices(
                layer_data,
                v_buffers[layer_id],
                page_idx_tensor,
                dst_head_start,
                num_heads,
                page_size,
            )
            offset += per_layer_bytes


# _scatter_staging_to_kv_triton：Triton 融合 kernel 路径的 staging → KV pool scatter（默认路径）
# 一次 kernel launch 处理所有 writer × 所有层 × K/V，高效并行写回 KV pool
def _scatter_staging_to_kv_triton(
    staging_buffer_view: torch.Tensor,
    k_buffers: list,
    v_buffers: list,
    page_idx_tensor: torch.Tensor,
    page_size: int,
    prefill_attn_tp_size: int,
    decode_attn_tp_size: int,
    dst_tp_rank: int,
    total_kv_heads: int,
) -> None:
    """Triton fused kernel path for scatter."""
    num_layers = len(k_buffers)
    head_dim = k_buffers[0].shape[-1]
    # total_heads：KV pool 每 token 的完整 head 数（用于计算 stride_pool_token）
    total_heads = k_buffers[0].shape[1]
    dtype_size = k_buffers[0].element_size()
    num_tokens = page_idx_tensor.shape[0] * page_size
    device = page_idx_tensor.device

    # 计算 writer 数
    if prefill_attn_tp_size > decode_attn_tp_size:
        num_writers = prefill_attn_tp_size // max(1, decode_attn_tp_size)
    else:
        num_writers = 1

    # All writers share the same num_heads; only dst_head_start differs
    # 所有 writer 的 num_heads 相同，只有 dst_head_start 不同
    _, num_heads, _, _ = compute_head_slice_params(
        prefill_attn_tp_size,
        decode_attn_tp_size,
        0,
        dst_tp_rank,
        total_kv_heads,
    )
    elems_per_token = num_heads * head_dim
    per_layer_elems = num_tokens * elems_per_token

    # 打包所有层 K/V 数据指针
    layer_ptrs = torch.tensor(
        [buf.data_ptr() for buf in k_buffers] + [buf.data_ptr() for buf in v_buffers],
        dtype=torch.int64,
        device=device,
    )

    # 构建每个 writer 的目标 head 起始字节偏移列表（供 Triton kernel 查找）
    writer_head_offsets = torch.tensor(
        [
            compute_head_slice_params(
                prefill_attn_tp_size,
                decode_attn_tp_size,
                wr,
                dst_tp_rank,
                total_kv_heads,
            )[2]
            * head_dim
            for wr in range(num_writers)
        ],
        dtype=torch.int64,
        device=device,
    )

    # bit-preserving 整数类型映射，避免类型转换开销
    int_dtype_map = {1: torch.int8, 2: torch.int16, 4: torch.int32}
    int_dtype = int_dtype_map.get(dtype_size, torch.int16)
    total_staging_bytes = (
        num_tokens * elems_per_token * dtype_size * num_layers * 2 * num_writers
    )
    staging_typed = staging_buffer_view[:total_staging_bytes].view(int_dtype)

    # grid: (num_writers * 2*num_layers, ceil(per_layer_elems/BLOCK_SIZE))
    BLOCK_SIZE = 1024
    num_layers_x2 = 2 * num_layers
    grid = (num_writers * num_layers_x2, triton.cdiv(per_layer_elems, BLOCK_SIZE))

    _fused_scatter_from_staging_kernel[grid](
        layer_ptrs,
        page_idx_tensor,
        staging_typed,
        writer_head_offsets,
        num_tokens,
        total_heads * head_dim,  # stride_pool_token
        per_layer_elems,
        elems_per_token,
        page_size,
        num_layers_x2,
        BLOCK_SIZE,
    )


# scatter_staging_to_kv：统一 scatter 入口，根据 _USE_TRITON_STAGING 分派 Triton 或 torch 路径
def scatter_staging_to_kv(
    staging_buffer_view: torch.Tensor,
    k_buffers: list,
    v_buffers: list,
    page_idx_tensor: torch.Tensor,
    page_size: int,
    prefill_attn_tp_size: int,
    decode_attn_tp_size: int,
    dst_tp_rank: int,
    total_kv_heads: int,
) -> None:
    """Scatter data from a contiguous staging region into KV cache buffers."""
    # 优先使用 Triton 融合 kernel，否则回退到逐 writer × 逐层的 torch scatter
    if _USE_TRITON_STAGING:
        return _scatter_staging_to_kv_triton(
            staging_buffer_view,
            k_buffers,
            v_buffers,
            page_idx_tensor,
            page_size,
            prefill_attn_tp_size,
            decode_attn_tp_size,
            dst_tp_rank,
            total_kv_heads,
        )
    return _scatter_staging_to_kv_torch(
        staging_buffer_view,
        k_buffers,
        v_buffers,
        page_idx_tensor,
        page_size,
        prefill_attn_tp_size,
        decode_attn_tp_size,
        dst_tp_rank,
        total_kv_heads,
    )


# compute_head_slice_params：计算异构 TP 传输时的 head 切片参数
# 输入：prefill/decode 的 attn_tp_size 和 rank，输出：源/目标的起始 head 和 head 数量
# 两种情形：prefill TP > decode TP（多写 → 一读）或 prefill TP <= decode TP（一写 → 多读）
def compute_head_slice_params(
    src_attn_tp_size: int,   # prefill 侧 attention TP 并行度
    dst_attn_tp_size: int,   # decode 侧 attention TP 并行度
    src_tp_rank: int,        # prefill 侧当前 TP rank
    dst_tp_rank: int,        # decode 侧目标 TP rank
    total_kv_heads: int,     # 全局 KV head 总数
) -> Tuple[int, int, int, int]:
    """Compute head slicing parameters for heterogeneous TP transfer.

    Returns:
        (src_head_start, num_heads_to_send, dst_head_start, num_heads_to_send)
    """
    # 计算每个 rank 持有的 head 数量
    src_heads_per_rank = max(1, total_kv_heads // src_attn_tp_size)
    dst_heads_per_rank = max(1, total_kv_heads // dst_attn_tp_size)

    # 对 rank 进行模运算，处理 TP 间存在 head 复制的情况
    local_tp_rank = src_tp_rank % src_attn_tp_size
    dst_tp_rank_in_group = dst_tp_rank % dst_attn_tp_size

    if src_attn_tp_size > dst_attn_tp_size:
        # prefill TP > decode TP：多个 prefill rank 各持有部分 head，全部发送给同一 decode rank
        # src 端从 head 0 开始发送自己负责的所有 head
        src_head_start = 0
        num_heads_to_send = src_heads_per_rank
        # 处理 head 复制（当 TP > head 数时，多个 rank 持有同一 head）
        src_replication = max(1, src_attn_tp_size // total_kv_heads)
        unique_head_idx = local_tp_rank // src_replication
        # 目标 head 起点：在 decode rank 负责的 head 范围内，找到对应的写入偏移
        dst_head_start = (unique_head_idx * src_heads_per_rank) % dst_heads_per_rank
    else:
        # prefill TP <= decode TP：一个 prefill rank 持有多个 decode rank 需要的 head
        # 源端只发送目标 decode rank 对应的那一段 head
        src_head_start = (
            dst_tp_rank_in_group * dst_heads_per_rank
        ) % src_heads_per_rank
        num_heads_to_send = dst_heads_per_rank
        # 目标 head 起点为 0（decode rank 只接收自己负责的 head）
        dst_head_start = 0

    return src_head_start, num_heads_to_send, dst_head_start, num_heads_to_send


# compute_staging_layout：计算 staging 缓冲区中每个 writer 的字节布局
# 用于 decode 侧预分配 staging 空间，或 prefill 侧计算传输字节数
def compute_staging_layout(
    src_attn_tp_size: int,    # prefill 侧 attn TP 并行度
    dst_attn_tp_size: int,    # decode 侧 attn TP 并行度
    dst_tp_rank: int,         # 目标 decode TP rank
    total_kv_heads: int,      # 全局 KV head 总数
    num_tokens: int,          # 本次传输的 token 数量
    bytes_per_head_token: int,  # 每个 (head, token) 的字节数（head_dim * dtype_size）
    num_layers: int,          # 模型层数
) -> Tuple[int, List[int], int]:
    """Compute per-writer byte layout for a staging region.

    Returns:
        (num_writers, writer_bytes_list, total_bytes)
        where writer_bytes_list[i] = bytes for writer i covering all layers (K+V).
    """
    # 计算 writer 数（prefill TP > decode TP 时有多个 writer）
    if src_attn_tp_size > dst_attn_tp_size:
        num_writers = src_attn_tp_size // max(1, dst_attn_tp_size)
    else:
        num_writers = 1

    # 逐 writer 计算字节数（每 writer 的 head 数可能不同）
    writer_bytes = []
    for wr in range(num_writers):
        _, nh, _, _ = compute_head_slice_params(
            src_attn_tp_size,
            dst_attn_tp_size,
            wr,
            dst_tp_rank,
            total_kv_heads,
        )
        # 每个 writer：num_tokens × nh heads × bytes_per_head_token × num_layers × 2（K+V）
        writer_bytes.append(num_tokens * nh * bytes_per_head_token * num_layers * 2)
    return num_writers, writer_bytes, sum(writer_bytes)


# resolve_total_kv_heads：从 kv_args 元数据中解析全局 KV head 总数
# 优先使用 total_kv_head_num，回退到 kv_head_num * attn_tp_size
def resolve_total_kv_heads(
    kv_args,              # KVArgs 实例（含 KV 内存布局元数据）
    attn_tp_size: int,    # attention TP 并行度，用于从 per-rank head 数推算全局总数
) -> int:
    """Resolve the global total KV head count from kv_args metadata."""
    # 优先使用显式字段 total_kv_head_num
    total = getattr(kv_args, "total_kv_head_num", 0)
    if total > 0:
        return total
    # 回退：per-rank head 数 × attn_tp_size
    per_rank = getattr(kv_args, "kv_head_num", 0)
    if per_rank > 0:
        return per_rank * attn_tp_size
    raise ValueError(
        "Cannot resolve total_kv_heads: kv_args has neither total_kv_head_num "
        "nor kv_head_num. "
        "Ensure DecodePreallocQueue._init_kv_manager sets kv_args.kv_head_num."
    )
