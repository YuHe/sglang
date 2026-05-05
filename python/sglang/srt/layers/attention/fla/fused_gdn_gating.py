# 融合门控衰减归一化（GDN Gating）模块：将 g 和 beta_output 的计算融合为单个 Triton kernel
from typing import Tuple

import torch
import triton
import triton.language as tl


# g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
# beta_output = b.sigmoid()
# Triton kernel：融合计算门控衰减值 g 和 sigmoid 门控输出 beta_output
@triton.jit
def fused_gdn_gating_kernel(
    g,           # 输出：门控衰减张量指针
    beta_output, # 输出：sigmoid 门控输出张量指针
    A_log,       # 输入：对数衰减参数 A_log
    a,           # 输入：时间步缩放参数 a
    b,           # 输入：门控参数 b（用于 sigmoid）
    dt_bias,     # 输入：时间步偏置
    seq_len,     # 序列长度（推理时固定为 1）
    stride_a,    # a 张量在 batch 维度的步长
    stride_b,    # b 张量在 batch 维度的步长
    NUM_HEADS: tl.constexpr,   # 注意力头总数（编译时常量）
    beta: tl.constexpr,        # softplus 的 beta 参数
    threshold: tl.constexpr,   # softplus 数值稳定性阈值
    BLK_HEADS: tl.constexpr,   # 每个 thread block 处理的头数
):
    # 三维 grid：(batch, seq_pos, head_block)
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # 计算当前 block 负责的 head 偏移范围
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    # 计算在展平张量中的全局偏移
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    # 加载各参数块（越界位置用 mask 屏蔽）
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + i_b * stride_a + head_off, mask=mask)
    blk_b = tl.load(b + i_b * stride_b + head_off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    # 计算 softplus(a + dt_bias)，使用阈值避免大值时的数值不稳定
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    # g = -exp(A_log) * softplus(a + dt_bias)，表示带衰减的门控值
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    # beta_output = sigmoid(b)，用于线性注意力门控
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(beta_output + off, blk_beta_output.to(b.dtype.element_ty), mask=mask)


# Python 封装函数：分配输出张量，配置 grid 并启动 fused_gdn_gating_kernel
def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 获取 batch 大小和 head 数量（推理时 seq_len 固定为 1）
    batch, num_heads = a.shape
    seq_len = 1
    stride_a = a.stride(0)
    stride_b = b.stride(0)
    # grid 划分：batch × seq_len × (头数/8) 个 block
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=torch.float32, device=b.device)
    # 启动 Triton kernel，每个 block 使用 1 个 warp
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        stride_a,
        stride_b,
        num_heads,
        beta,
        threshold,
        8,
        num_warps=1,
    )
    return g, beta_output
