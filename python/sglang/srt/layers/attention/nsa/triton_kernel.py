# NSA 稀疏注意力 Triton GPU kernel 模块
# 包含激活量化（FP8 block-wise）和有效 KV 索引提取两个核心 kernel
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


# Triton implementation
@triton.jit
def _act_quant_kernel(
    # 激活值分块量化 Triton kernel
    # 将输入 FP16/BF16 张量按 group_size 列分组量化为 FP8，并计算每组缩放因子
    X_ptr,              # 输入激活张量指针，形状 [M, N]
    Y_ptr,              # 输出量化张量指针，形状 [M, N]，dtype=float8_e4m3fn
    S_ptr,              # 输出缩放因子指针，形状 [M, N // group_size]
    M,                  # 行数（token 数）
    N,                  # 列数（特征维度，必须被 group_size 整除）
    group_size: tl.constexpr,  # 量化分组大小（每组独立计算 scale）
    round_scale: tl.constexpr, # 是否对 scale 做 round-to-power-of-2 近似
    BLOCK_M: tl.constexpr,     # 每个 block 处理的行数
    BLOCK_N: tl.constexpr,     # 每个 block 处理的列数（等于 group_size）
):
    """
    Triton kernel for activation quantization.

    Each block processes BLOCK_M rows and group_size columns.
    """
    # Get block IDs
    # 二维 grid：pid_m 对应行分块，pid_n 对应列（group）分块
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # FP8 constants
    # FP8 E4M3 格式的表示范围 [-448, 448]
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max  # 预计算倒数，用于 scale 计算

    # Calculate row and column offsets
    # 计算当前 block 在全局 tensor 中的起始行/列
    row_start = pid_m * BLOCK_M
    col_start = pid_n * group_size

    # Create offset arrays
    # 构建当前 block 内所有行/列的绝对索引
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)

    # Mask for valid rows and columns
    # 边界 mask：防止越界访问（M/N 不一定是 BLOCK 大小的整数倍）
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]

    # Load input data
    # 从显存加载当前 block 的激活值（越界位置填 0）
    x_ptrs = X_ptr + rows[:, None] * N + cols[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Compute absolute max along columns (group_size dimension) for each row
    # 计算每行（即每个 group）的绝对值最大值，用于确定量化 scale
    x_abs = tl.abs(x)
    amax = tl.max(x_abs, axis=1)  # Shape: (BLOCK_M,)

    # Clamp amax to avoid division by zero
    # 避免 amax=0 导致除零，设置最小值 1e-4
    amax = tl.maximum(amax, 1e-4)

    # Compute scale
    if round_scale:
        # Fast round scale using bit manipulation approximation
        # This is a simplified version - the exact bit manipulation is harder in Triton
        # Using log2 + ceil + pow2 as approximation
        # round_scale 模式：将 scale 近似到最近的 2 的幂次（减少量化误差累积）
        log_val = tl.log2(amax * fp8_max_inv)
        log_ceil = tl.ceil(log_val)        # 向上取整到整数幂次
        scale = tl.exp2(log_ceil)          # 还原为 2^ceil 的 scale
    else:
        # 直接使用 amax / fp8_max 作为 scale（标准 absmax 量化）
        scale = amax * fp8_max_inv

    # Quantize: y = clamp(x / scale, fp8_min, fp8_max)
    # 缩放后 clamp 到 FP8 表示范围，完成量化
    scale_broadcast = scale[:, None]  # 广播 scale 到与 x 相同的形状
    y = x / scale_broadcast
    y = tl.minimum(tl.maximum(y, fp8_min), fp8_max)  # clamp 到 [-448, 448]

    # Store quantized output
    # 将量化结果写回显存
    y_ptrs = Y_ptr + rows[:, None] * N + cols[None, :]
    tl.store(y_ptrs, y, mask=mask)

    # Store scales
    # 将每行（每 group）的缩放因子写回 scale 张量
    s_cols = pid_n  # scale 的列索引对应 group 编号
    s_ptrs = S_ptr + rows * (N // group_size) + s_cols
    s_mask = row_mask
    tl.store(s_ptrs, scale, mask=s_mask)


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization with Triton.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
        scale_fmt (Optional[str], optional): The format of the scale. Default is None.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    # 输入校验：必须连续存储且最后维度能被 block_size 整除
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"

    # Flatten all dims except last
    # 将所有前置维度合并为行，仅保留最后一维（特征维度）
    N = x.size(-1)
    x_flat = x.view(-1, N)
    M = x_flat.size(0)

    # Allocate output tensors
    # 分配 FP8 量化输出和 float32 缩放因子张量
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    y_flat = y.view(-1, N)
    s = x.new_empty(*x.size()[:-1], N // block_size, dtype=torch.float32)
    s_flat = s.view(-1, N // block_size)

    # Launch kernel
    # 二维 grid：行分块 × group 分块
    BLOCK_M = 32
    BLOCK_N = block_size
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, block_size))
    # scale_fmt 非空时开启 round_scale 模式
    round_scale = scale_fmt is not None

    _act_quant_kernel[grid](
        x_flat,
        y_flat,
        s_flat,
        M,
        N,
        group_size=block_size,
        round_scale=round_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        # round_scale 模式禁用流水线以保证正确性
        num_stages=0 if round_scale else 2,
    )

    return y, s


@triton.jit
def _get_valid_kv_indices_kernel(
    # 从稀疏 top-k 页表中提取有效（非-1）的 KV 物理索引，并紧凑排列
    page_table_ptr,  # [bs, topk] 稀疏注意力页表，-1 表示无效位置
    kv_indptr_ptr,  # [bs + 1] 每个 batch item 在输出中的起始偏移（前缀和）
    kv_indices_ptr,  # [bs * topk] 输出缓冲区，存放紧凑后的有效 KV 索引
    bs: tl.constexpr,    # batch size
    topk: tl.constexpr,  # 每个 batch item 的 top-k 容量
):
    """
    Extract valid indices (non -1) from page_table into kv_indices.
    Each program handles one batch.
    """
    # 每个 program 处理一个 batch item
    batch_id = tl.program_id(0)

    # Get the start position for this batch in kv_indices
    # 从前缀和数组读取当前 batch 在输出缓冲区的起始位置
    dst_start = tl.load(kv_indptr_ptr + batch_id)

    # Load all topk indices for this batch
    # 加载当前 batch 的全部 topk 个页表索引
    src_offset = batch_id * topk
    offsets = tl.arange(0, topk)
    indices = tl.load(page_table_ptr + src_offset + offsets)

    # Count valid indices and compact them
    # -1 表示无效（填充），mask 标记有效位置
    mask = indices != -1

    # Use prefix sum to compute destination positions for valid elements
    # For each position, count how many valid elements are before it
    # 对有效 mask 做前缀和，计算每个有效元素在紧凑输出中的相对偏移
    prefix_sum = tl.cumsum(mask.to(tl.int32), axis=0) - 1

    # Store valid indices to their compacted positions
    # 计算全局写入地址并将有效索引写入输出缓冲区（无效位置不写）
    dst_positions = dst_start + prefix_sum
    tl.store(kv_indices_ptr + dst_positions, indices, mask=mask)


def get_valid_kv_indices(
    page_table_1: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    bs: int,
):
    """
    Extract valid indices from page_table_1 into kv_indices buffer.

    Args:
        page_table_1: [bs, topk] page table with -1 as invalid
        kv_indptr: [bs + 1] cumulative count of valid indices per batch
        kv_indices: [bs * topk] pre-allocated output buffer
        bs: batch size
    """
    # 启动 bs 个 Triton program，每个负责一个 batch item 的有效索引提取
    topk = page_table_1.shape[1]
    grid = (bs,)
    _get_valid_kv_indices_kernel[grid](
        page_table_1,
        kv_indptr,
        kv_indices,
        bs,
        topk,
    )
