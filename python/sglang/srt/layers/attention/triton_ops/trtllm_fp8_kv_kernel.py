"""
Fused FP8 quantization + paged KV cache write kernel for TRTLLM MHA backend.

This kernel fuses the following operations:
1. FP8 quantization of K and V tensors (from BF16/FP16 to FP8)
2. Per-token or per-page scale computation
3. Writing quantized K/V to paged KV cache layout

Performance benefits:
- Eliminates intermediate FP8 tensors in memory
- Reduces kernel launch overhead
- Better memory bandwidth utilization
"""

# 导入日志和类型提示模块
import logging
from typing import Optional

# 导入 PyTorch 和 Triton 相关库
import torch
import triton
import triton.language as tl

# 模块级日志对象
logger = logging.getLogger(__name__)


# Triton JIT：处理单个 K 或 V 张量的一个头块数据
# 将 BF16/FP16 数据量化为 FP8 并写入 paged KV 缓存
@triton.jit
def _process_kv_tensor(
    token_id,         # 当前处理的 token 索引
    head_block_id,    # 当前处理的头块索引（每块 BLOCK_HEAD 个头）
    page_id,          # 在 KV 缓存中对应的页索引
    page_offset,      # 在页内的偏移（token 在页内的位置）
    input_ptr,        # 输入张量指针（K 或 V）
    cache_ptr,        # 输出 KV 缓存指针
    inv_scale,        # 量化缩放因子的倒数（1/scale）
    use_provided_scale: tl.constexpr,  # 是否使用外部提供的缩放因子
    num_kv_heads: tl.constexpr,        # KV 头总数
    head_dim: tl.constexpr,            # 每个头的维度大小
    input_stride_token: tl.constexpr,  # 输入在 token 维度的步长
    input_stride_head: tl.constexpr,   # 输入在 head 维度的步长
    input_stride_dim: tl.constexpr,    # 输入在 dim 维度的步长
    cache_stride_page: tl.constexpr,   # 缓存在 page 维度的步长
    cache_stride_offset: tl.constexpr, # 缓存在 page_offset 维度的步长
    cache_stride_head: tl.constexpr,   # 缓存在 head 维度的步长
    cache_stride_dim: tl.constexpr,    # 缓存在 dim 维度的步长
    BLOCK_HEAD: tl.constexpr,          # 头维度的分块大小
    BLOCK_DIM: tl.constexpr,           # 头内维度的分块大小
):
    """Process a block of heads for a single K or V tensor."""
    # 计算当前头块的起始头索引
    head_idx = head_block_id * BLOCK_HEAD
    # 计算当前块内实际有效的头数（处理最后一块可能不足 BLOCK_HEAD 的情况）
    num_heads_in_block = min(BLOCK_HEAD, num_kv_heads - head_idx)

    # 遍历头维度（按 BLOCK_DIM 分块，减少寄存器压力）
    for dim_idx in range(0, head_dim, BLOCK_DIM):
        # 当前维度块内实际有效的维度数
        num_dims_in_block = min(BLOCK_DIM, head_dim - dim_idx)

        # 生成头和维度的索引范围
        head_offsets = head_idx + tl.arange(0, BLOCK_HEAD)
        dim_offsets = dim_idx + tl.arange(0, BLOCK_DIM)

        # 生成有效性掩码（过滤超出范围的头和维度）
        head_mask = head_offsets < (head_idx + num_heads_in_block)
        dim_mask = dim_offsets < (dim_idx + num_dims_in_block)

        # Load from input using 3D strides
        # 计算输入张量的 3D 内存地址偏移（token, head, dim）
        input_offsets = (
            token_id * input_stride_token
            + head_offsets[:, None] * input_stride_head
            + dim_offsets[None, :] * input_stride_dim
        )
        # 2D 掩码：同时过滤无效头和无效维度
        mask = head_mask[:, None] & dim_mask[None, :]

        # 从全局内存加载 BF16/FP16 数据块
        block = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)

        # Quantize to FP8
        # 若使用外部提供的缩放因子，先乘以 inv_scale 再转换为 FP8（E4M3 格式）
        if use_provided_scale:
            block_fp8 = (block * inv_scale).to(tl.float8e4nv)
        else:
            # 无缩放因子时直接类型转换（依赖 FP8 的动态范围）
            block_fp8 = block.to(tl.float8e4nv)

        # Write to cache at [page_id, page_offset, head, dim]
        # 计算 paged KV 缓存的 4D 内存地址偏移（page, offset, head, dim）
        cache_offsets = (
            page_id * cache_stride_page
            + page_offset * cache_stride_offset
            + head_offsets[:, None] * cache_stride_head
            + dim_offsets[None, :] * cache_stride_dim
        )

        # 将量化后的 FP8 数据写入 paged KV 缓存（按掩码写入）
        tl.store(cache_ptr + cache_offsets, block_fp8, mask=mask)


# Triton JIT：融合的 FP8 量化 + paged KV 缓存写入 Kernel
# Grid = (num_tokens, num_head_blocks, 2)，dim2=0 处理 K，dim2=1 处理 V
@triton.jit
def _fused_fp8_set_kv_buffer_kernel(
    # Input tensors (post-RoPE K and V in FP16/BF16)
    k_ptr,  # [num_tokens, num_kv_heads, head_dim] — 施加 RoPE 后的 Key（FP16/BF16）
    v_ptr,  # [num_tokens, num_kv_heads, head_dim] — Value 张量（FP16/BF16）
    # Output KV cache buffers (FP8 paged layout)
    k_cache_ptr,  # [total_slots, num_kv_heads, head_dim] — FP8 格式的 K paged 缓存
    v_cache_ptr,  # [total_slots, num_kv_heads, head_dim] — FP8 格式的 V paged 缓存
    # Cache location indices
    cache_loc_ptr,  # [num_tokens] — 每个 token 对应的缓存位置索引
    # Pointers to scalar inverse scales (computed on GPU in wrapper)
    inv_k_scale_ptr,  # K 量化缩放因子倒数的 GPU 标量指针（0-D tensor）
    inv_v_scale_ptr,  # V 量化缩放因子倒数的 GPU 标量指针（0-D tensor）
    use_provided_scale: tl.constexpr,  # 是否使用外部提供的缩放因子
    # Tensor dimensions
    num_kv_heads: tl.constexpr,  # KV 头总数
    head_dim: tl.constexpr,      # 每个头的维度大小
    page_size: tl.constexpr,     # 每个 page 包含的 token 数
    # Strides for K input [num_tokens, num_kv_heads, head_dim]
    k_stride_token: tl.constexpr,  # K 在 token 维度的步长
    k_stride_head: tl.constexpr,   # K 在 head 维度的步长
    k_stride_dim: tl.constexpr,    # K 在 dim 维度的步长
    # Strides for K cache [total_slots, num_kv_heads, head_dim] (logically paged)
    k_cache_stride_page: tl.constexpr,    # K 缓存在 page 维度的步长
    k_cache_stride_offset: tl.constexpr,  # K 缓存在 page_offset 维度的步长
    k_cache_stride_head: tl.constexpr,    # K 缓存在 head 维度的步长
    k_cache_stride_dim: tl.constexpr,     # K 缓存在 dim 维度的步长
    # Strides for V input [num_tokens, num_kv_heads, head_dim]
    v_stride_token: tl.constexpr,  # V 在 token 维度的步长
    v_stride_head: tl.constexpr,   # V 在 head 维度的步长
    v_stride_dim: tl.constexpr,    # V 在 dim 维度的步长
    # Strides for V cache [total_slots, num_kv_heads, head_dim] (logically paged)
    v_cache_stride_page: tl.constexpr,    # V 缓存在 page 维度的步长
    v_cache_stride_offset: tl.constexpr,  # V 缓存在 page_offset 维度的步长
    v_cache_stride_head: tl.constexpr,    # V 缓存在 head 维度的步长
    v_cache_stride_dim: tl.constexpr,     # V 缓存在 dim 维度的步长
    # Block sizes
    BLOCK_HEAD: tl.constexpr,  # Number of heads per block — 每个头块的头数
    BLOCK_DIM: tl.constexpr,  # Head dimension block size — 头内维度的分块大小
):
    """
    Fused FP8 quantization + paged KV cache write kernel.

    Each program processes one token-head_block-kv combination, quantizing and writing
    to the appropriate page in the KV cache.

    Grid: (num_tokens, num_head_blocks, 2) where dim2: 0=K, 1=V
    """
    # Get program IDs
    # 获取当前 token 索引
    token_id = tl.program_id(0)
    # 获取当前头块索引
    head_block_id = tl.program_id(1)
    # kv_idx=0 处理 K，kv_idx=1 处理 V（KV 并行处理）
    kv_idx = tl.program_id(2)  # 0 for K, 1 for V

    # Get cache location for this token
    # 加载当前 token 的 paged KV 缓存物理位置
    cache_loc = tl.load(cache_loc_ptr + token_id)

    # Compute page_id and offset within page
    # 将线性缓存位置分解为页索引和页内偏移
    page_id = cache_loc // page_size
    page_offset = cache_loc % page_size

    # Select K or V based on kv_idx
    if kv_idx == 0:
        # Process K tensor
        # 处理 K 张量：加载 inv_scale 并量化写入 K 缓存
        if use_provided_scale:
            inv_scale = tl.load(inv_k_scale_ptr)  # 从 GPU 内存加载 K 的量化缩放因子倒数
        else:
            inv_scale = 1.0
        _process_kv_tensor(
            token_id,
            head_block_id,
            page_id,
            page_offset,
            k_ptr,
            k_cache_ptr,
            inv_scale,
            use_provided_scale,
            num_kv_heads,
            head_dim,
            k_stride_token,
            k_stride_head,
            k_stride_dim,
            k_cache_stride_page,
            k_cache_stride_offset,
            k_cache_stride_head,
            k_cache_stride_dim,
            BLOCK_HEAD,
            BLOCK_DIM,
        )
    else:
        # Process V tensor
        # 处理 V 张量：加载 inv_scale 并量化写入 V 缓存
        if use_provided_scale:
            inv_scale = tl.load(inv_v_scale_ptr)  # 从 GPU 内存加载 V 的量化缩放因子倒数
        else:
            inv_scale = 1.0
        _process_kv_tensor(
            token_id,
            head_block_id,
            page_id,
            page_offset,
            v_ptr,
            v_cache_ptr,
            inv_scale,
            use_provided_scale,
            num_kv_heads,
            head_dim,
            v_stride_token,
            v_stride_head,
            v_stride_dim,
            v_cache_stride_page,
            v_cache_stride_offset,
            v_cache_stride_head,
            v_cache_stride_dim,
            BLOCK_HEAD,
            BLOCK_DIM,
        )


# Python 封装：融合 FP8 量化 + paged KV 缓存写入的主调用函数
def fused_fp8_set_kv_buffer(
    k: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim] or [num_tokens, num_kv_heads * head_dim]
    v: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim] or [num_tokens, num_kv_heads * head_dim]
    k_cache: torch.Tensor,  # [total_slots, num_kv_heads, head_dim] or [num_pages, page_size, num_kv_heads, head_dim]
    v_cache: torch.Tensor,  # [total_slots, num_kv_heads, head_dim] or [num_pages, page_size, num_kv_heads, head_dim]
    cache_loc: torch.Tensor,  # [num_tokens], dtype=int32 — 每个 token 的缓存位置
    k_scale: Optional[
        float
    ] = None,  # Scalar scale (matching original set_kv_buffer signature)
    v_scale: Optional[float] = None,
    page_size: int = 16,   # 每个 page 的 token 数（paged KV 的分页粒度）
    use_triton: bool = True,  # Whether to use Triton kernel (set to False to force naive fallback)
) -> None:
    """
    Python wrapper for the fused FP8 quantization + paged KV cache write kernel.

    This function replicates the exact behavior of the original set_kv_buffer but with
    a fused kernel that combines FP8 quantization and cache write.

    Args:
        k: Key tensor after RoPE, can be 2D or 3D
        v: Value tensor, can be 2D or 3D
        k_cache: Paged K cache buffer in FP8
        v_cache: Paged V cache buffer in FP8
        cache_loc: Cache location for each token, shape [num_tokens]
        k_scale: Optional scalar scale for K (matching original set_kv_buffer)
        v_scale: Optional scalar scale for V (matching original set_kv_buffer)
        page_size: Number of tokens per page
        use_triton: Whether to use optimized Triton kernel
    """
    # 获取 token 总数
    num_tokens = k.shape[0]

    # Step 1: Infer num_kv_heads and head_dim from cache shape
    # 步骤1：根据 KV 缓存形状推断头数和维度大小
    if k_cache.ndim == 3:
        # 3D cache layout: [total_slots, num_kv_heads, head_dim]
        # 3D 缓存格式：[总槽位数, KV头数, 头维度]
        total_slots, num_kv_heads, head_dim = k_cache.shape
        assert (
            total_slots % page_size == 0
        ), f"total_slots ({total_slots}) must be divisible by page_size ({page_size})"
        num_pages = total_slots // page_size
    elif k_cache.ndim == 4:
        # 4D cache layout: [num_pages, page_size, num_kv_heads, head_dim]
        # 4D 缓存格式：[页数, 每页token数, KV头数, 头维度]
        num_pages, ps, num_kv_heads, head_dim = k_cache.shape
        assert (
            ps == page_size
        ), f"page_size mismatch: cache has {ps}, expected {page_size}"
        total_slots = num_pages * page_size
    else:
        raise ValueError(f"Unsupported k_cache.ndim={k_cache.ndim}, expected 3 or 4")

    # Step 2: Validate k, v shapes and normalize
    # Store original 3D shape for Triton path
    # 步骤2：验证 K/V 形状并规范化为 2D 和 3D 两种视图
    k_3d = None
    v_3d = None

    if k.ndim == 3:
        # Input is [num_tokens, num_kv_heads, head_dim]
        # 输入已经是 3D 格式，直接使用
        assert (
            k.shape[1] == num_kv_heads
        ), f"num_kv_heads mismatch: k.shape[1]={k.shape[1]} vs cache={num_kv_heads}"
        assert (
            k.shape[2] == head_dim
        ), f"head_dim mismatch: k.shape[2]={k.shape[2]} vs cache={head_dim}"
        assert v.shape[1] == num_kv_heads and v.shape[2] == head_dim, "v shape mismatch"

        # Keep 3D for Triton kernel
        k_3d = k
        v_3d = v
        # Create 2D view for naive fallback (will be used only if use_triton=False)
        # 为朴素实现创建 2D 视图（将 head 和 dim 展平）
        k_2d = k.reshape(num_tokens, num_kv_heads * head_dim)
        v_2d = v.reshape(num_tokens, num_kv_heads * head_dim)
    elif k.ndim == 2:
        # Input is already [num_tokens, num_kv_heads * head_dim]
        # 输入是 2D 格式，需要重塑为 3D 供 Triton Kernel 使用
        assert (
            k.shape[1] == num_kv_heads * head_dim
        ), f"k.shape[1]={k.shape[1]} != {num_kv_heads * head_dim}"
        assert (
            v.shape[1] == num_kv_heads * head_dim
        ), f"v.shape[1]={v.shape[1]} != {num_kv_heads * head_dim}"

        # Create 3D view for Triton kernel
        k_3d = k.view(num_tokens, num_kv_heads, head_dim)
        v_3d = v.view(num_tokens, num_kv_heads, head_dim)
        # Keep 2D for naive
        k_2d = k
        v_2d = v
    else:
        raise ValueError(f"Unsupported k.ndim={k.ndim}, expected 2 or 3")

    # Step 3: Compute cache strides based on layout
    # 步骤3：根据缓存布局（3D 或 4D）计算各维度步长
    if k_cache.ndim == 3:
        # 3D cache: [total_slots, num_kv_heads, head_dim]
        # 3D 格式：slot 步长 * page_size = 页步长
        stride_slot = k_cache.stride(0)
        stride_head = k_cache.stride(1)
        stride_dim = k_cache.stride(2)

        k_cache_stride_page = stride_slot * page_size
        k_cache_stride_offset = stride_slot
        k_cache_stride_head = stride_head
        k_cache_stride_dim = stride_dim

        v_stride_slot = v_cache.stride(0)
        v_stride_head = v_cache.stride(1)
        v_stride_dim = v_cache.stride(2)

        v_cache_stride_page = v_stride_slot * page_size
        v_cache_stride_offset = v_stride_slot
        v_cache_stride_head = v_stride_head
        v_cache_stride_dim = v_stride_dim
    else:
        # 4D cache: [num_pages, page_size, num_kv_heads, head_dim]
        # 4D 格式：直接从张量步长中读取各维度步长
        k_cache_stride_page = k_cache.stride(0)
        k_cache_stride_offset = k_cache.stride(1)
        k_cache_stride_head = k_cache.stride(2)
        k_cache_stride_dim = k_cache.stride(3)

        v_cache_stride_page = v_cache.stride(0)
        v_cache_stride_offset = v_cache.stride(1)
        v_cache_stride_head = v_cache.stride(2)
        v_cache_stride_dim = v_cache.stride(3)

    # Decide whether to use provided scale
    # 判断是否使用外部提供的量化缩放因子（K 和 V 的 scale 必须同时提供）
    use_provided_scale = k_scale is not None and v_scale is not None

    if use_triton and num_tokens > 0:
        # Use optimized Triton kernel
        # Compute input strides for 3D k, v: [num_tokens, num_kv_heads, head_dim]
        # 计算 K/V 输入张量的 3D 步长（供 Triton Kernel 使用）
        k_stride_token = k_3d.stride(0)
        k_stride_head = k_3d.stride(1)
        k_stride_dim = k_3d.stride(2)

        v_stride_token = v_3d.stride(0)
        v_stride_head = v_3d.stride(1)
        v_stride_dim = v_3d.stride(2)

        # Block sizes for tiling (tunable)
        BLOCK_HEAD = min(num_kv_heads, 8)  # Process up to 8 heads at once — 每次处理最多 8 个头
        BLOCK_DIM = min(head_dim, 128)  # Process up to 128 dims at once — 每次处理最多 128 个维度

        # Compute number of head blocks
        # 计算头块数量（向上取整）
        num_head_blocks = (num_kv_heads + BLOCK_HEAD - 1) // BLOCK_HEAD

        # Grid: (num_tokens, num_head_blocks, 2)
        # - dim 0: tokens — 并行处理所有 token
        # - dim 1: head blocks — 并行处理头块
        # - dim 2: K/V (0=K, 1=V) — 并行处理 K 和 V
        grid = (num_tokens, num_head_blocks, 2)

        device = k_3d.device

        def _to_tensor_scale(scale):
            """Convert scale to 0-D CUDA tensor (accepts Python float or Tensor)."""
            if isinstance(scale, torch.Tensor):
                # 已是张量，仅转移到目标设备和数据类型
                return scale.to(device=device, dtype=torch.float32)
            else:
                # Python float / np scalar
                # Python 标量或 numpy 标量，转换为 0-D GPU 张量
                return torch.tensor(float(scale), device=device, dtype=torch.float32)

        # Compute inverse scales on GPU to avoid GPU→CPU sync in CUDA graph capture.
        # Previously we used float(k_scale) which triggers synchronization and fails
        # during CUDA graph capture with cudaErrorStreamCaptureUnsupported.
        # 在 GPU 上计算缩放因子的倒数，避免 GPU→CPU 同步（CUDA graph capture 会失败）
        if use_provided_scale:
            k_scale_tensor = _to_tensor_scale(k_scale)
            v_scale_tensor = _to_tensor_scale(v_scale)

            # Pure GPU scalar operation, safe for CUDA graph
            # 纯 GPU 标量运算，对 CUDA graph capture 安全
            inv_k_scale = (1.0 / k_scale_tensor).to(device=device, dtype=torch.float32)
            inv_v_scale = (1.0 / v_scale_tensor).to(device=device, dtype=torch.float32)

            inv_k_scale_ptr = inv_k_scale
            inv_v_scale_ptr = inv_v_scale
        else:
            # When use_provided_scale=False, kernel uses constant 1.0 for inv_scale.
            # Triton will optimize away the tl.load() calls via constant folding.
            # We pass dummy pointers (k_3d) which won't be accessed in the kernel.
            # This avoids creating new GPU tensors during CUDA graph capture.
            # 不使用缩放因子时，传入 dummy 指针（编译器会优化掉对应 tl.load 调用）
            inv_k_scale_ptr = k_3d
            inv_v_scale_ptr = k_3d

        # Launch Triton kernel
        # 启动融合的 FP8 量化 + paged KV 缓存写入 Triton Kernel
        _fused_fp8_set_kv_buffer_kernel[grid](
            k_3d,
            v_3d,
            k_cache,
            v_cache,
            cache_loc,
            inv_k_scale_ptr,
            inv_v_scale_ptr,
            use_provided_scale,
            num_kv_heads,
            head_dim,
            page_size,
            k_stride_token,
            k_stride_head,
            k_stride_dim,
            k_cache_stride_page,
            k_cache_stride_offset,
            k_cache_stride_head,
            k_cache_stride_dim,
            v_stride_token,
            v_stride_head,
            v_stride_dim,
            v_cache_stride_page,
            v_cache_stride_offset,
            v_cache_stride_head,
            v_cache_stride_dim,
            BLOCK_HEAD=BLOCK_HEAD,
            BLOCK_DIM=BLOCK_DIM,
        )
    else:
        # Fallback to naive implementation
        # 回退到朴素实现（当 use_triton=False 或无 token 时）
        _naive_fp8_set_kv_buffer(
            k_2d, v_2d, k_cache, v_cache, cache_loc, k_scale, v_scale, page_size
        )


# 朴素参考实现：逐 token 量化并写入 paged KV 缓存（无融合优化）
def _naive_fp8_set_kv_buffer(
    k: torch.Tensor,       # [num_tokens, num_kv_heads * head_dim] — 2D 格式的 Key
    v: torch.Tensor,       # [num_tokens, num_kv_heads * head_dim] — 2D 格式的 Value
    k_cache: torch.Tensor, # KV 缓存（3D 或 4D）
    v_cache: torch.Tensor,
    cache_loc: torch.Tensor,  # [num_tokens] — 每个 token 的缓存位置
    k_scale: Optional[float], # K 的量化缩放因子（可选）
    v_scale: Optional[float], # V 的量化缩放因子（可选）
    page_size: int,           # 每个 page 的 token 数
) -> None:
    """
    Naive fallback implementation that mimics the original set_kv_buffer logic.

    This directly replicates the behavior of MHATokenToKVPool.set_kv_buffer:
    1. Apply scale (if k.dtype != cache.dtype and scale is provided)
    2. Convert to FP8
    3. Write to cache at cache_loc

    Args:
        k: [num_tokens, num_kv_heads * head_dim], already reshaped to 2D
        v: [num_tokens, num_kv_heads * head_dim], already reshaped to 2D
        k_cache: [total_slots, num_kv_heads, head_dim] or [num_pages, page_size, num_kv_heads, head_dim]
        v_cache: Same shape as k_cache
        cache_loc: [num_tokens]
        k_scale: Optional scale for K
        v_scale: Optional scale for V
        page_size: Tokens per page
    """
    num_tokens = k.shape[0]

    # Infer dimensions from cache
    # 从缓存形状推断头数和维度
    if k_cache.ndim == 3:
        num_kv_heads = k_cache.shape[1]
        head_dim = k_cache.shape[2]
    elif k_cache.ndim == 4:
        num_kv_heads = k_cache.shape[2]
        head_dim = k_cache.shape[3]
    else:
        raise ValueError(f"Unsupported k_cache.ndim={k_cache.ndim}")

    # Determine target dtype and storage dtype
    # See: python/sglang/srt/mem_cache/memory_pool.py:445-449
    # 确定目标数据类型和存储数据类型
    store_dtype = k_cache.dtype
    if store_dtype == torch.uint8:
        # Cache is stored as uint8 for FP8 (due to index_put limitation)
        # PyTorch index_put 不支持 FP8，所以用 uint8 作为 FP8 的存储类型
        dtype = torch.float8_e4m3fn  # Logical dtype
    else:
        dtype = store_dtype  # Cache dtype is the logical dtype

    # Replicate the original set_kv_buffer behavior
    # See: python/sglang/srt/mem_cache/memory_pool.py:777-799
    # 复制原始 set_kv_buffer 的量化行为
    if k.dtype != dtype:
        # Need quantization - clone first to avoid modifying input
        # 需要量化：先克隆以避免修改原始输入
        k = k.clone()
        v = v.clone()

        # 应用缩放因子（in-place 除法）
        if k_scale is not None:
            k.div_(k_scale)  # In-place division
        if v_scale is not None:
            v.div_(v_scale)  # In-place division

        # 转换为目标 FP8 格式
        k = k.to(dtype)
        v = v.to(dtype)

    # View FP8 as uint8 if needed (for index_put compatibility)
    # 若存储格式为 uint8（FP8 的兼容存储），进行视图转换
    if store_dtype == torch.uint8 and dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
        k = k.view(torch.uint8)
        v = v.view(torch.uint8)

    # Reshape from [T, H*D] to [T, H, D]
    # 从 2D [T, H*D] 重塑为 3D [T, H, D]，匹配缓存布局
    k = k.view(num_tokens, num_kv_heads, head_dim)
    v = v.view(num_tokens, num_kv_heads, head_dim)

    # Write to cache using advanced indexing (same as original)
    # 使用高级索引将量化后的 KV 写入 paged 缓存
    if k_cache.ndim == 3:
        # 3D cache: [total_slots, H, D]
        # 3D 缓存：直接按 cache_loc 索引写入
        k_cache[cache_loc] = k
        v_cache[cache_loc] = v
    else:
        # 4D cache: [num_pages, page_size, H, D]
        # Decompose loc into page_id and page_offset (vectorized)
        # 4D 缓存：将线性位置分解为页索引和页内偏移（向量化操作）
        page_ids = cache_loc // page_size
        page_offsets = cache_loc % page_size
        k_cache[page_ids, page_offsets] = k
        v_cache[page_ids, page_offsets] = v
