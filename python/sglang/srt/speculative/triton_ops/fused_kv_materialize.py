# Copyright 2023-2024 SGLang Team
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
"""Fused Triton kernel for DFlash KV materialization.

Combines: KV projection (cuBLAS) + RMSNorm + RoPE (Triton), then pool-managed KV writes.
"""
# 融合 KV 物化（materialization）Triton 核函数：
# 将 KV 投影 + RMSNorm + RoPE 三个操作融合为单一 Triton kernel，
# 减少显存往返，提升 DFlash 投机解码的 KV 写入效率

from typing import Callable, List

import torch
import triton
import triton.language as tl


# Triton JIT 核函数：融合 RMSNorm(K) + RoPE(K) + KV 写出
# Grid: (total_ctx, num_kv_heads) — 每个 (context token, KV head) 对对应一个 program
@triton.jit
def _fused_norm_rope_kernel(
    kv_ptr,  # [total_ctx, kv_size * 2]  — 原始 KV 投影输出（K 在前，V 在后）
    k_norm_weight_ptr,  # [head_dim]      — K 的 RMSNorm 权重
    cos_sin_cache_ptr,  # [max_pos, rotary_dim] — 预计算的 cos/sin 表
    positions_ptr,  # [total_ctx]         — 每个 token 的位置编码索引
    k_out_ptr,  # [total_ctx, num_kv_heads, head_dim] — 归一化+旋转后的 K 输出
    v_out_ptr,  # [total_ctx, num_kv_heads, head_dim] — V 输出（不变换）
    kv_stride_ctx,         # kv 张量在 ctx 维度的 stride
    cos_sin_stride_pos,    # cos_sin_cache 在 pos 维度的 stride
    k_out_stride_ctx,      # k_out 在 ctx 维度的 stride
    k_out_stride_head,     # k_out 在 head 维度的 stride
    v_out_stride_ctx,
    v_out_stride_head,
    total_ctx,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    kv_size: tl.constexpr,     # = num_kv_heads * head_dim
    rotary_dim: tl.constexpr,  # RoPE 旋转的维度数（通常 < head_dim）
    half_rotary_dim: tl.constexpr,  # = rotary_dim // 2
    eps: tl.constexpr,         # RMSNorm 数值稳定性 epsilon
    BLOCK_HD: tl.constexpr,    # head_dim 的向上取整到 2 的幂次
):
    """Fused RMSNorm(K) + RoPE(K) materialization. Grid: (total_ctx, num_kv_heads)."""
    # 当前 program 处理的 context token 和 KV head 索引
    ctx_id = tl.program_id(0)
    head_id = tl.program_id(1)
    if ctx_id >= total_ctx:
        return

    # Load metadata
    # 加载当前 token 的位置编码索引（用于查 cos/sin 表）
    position = tl.load(positions_ptr + ctx_id)

    # Compute base pointers
    # 计算当前 (ctx, head) 在原始 KV 缓冲区中的基础指针
    kv_base = kv_ptr + ctx_id * kv_stride_ctx
    k_base = kv_base + head_id * head_dim           # K 的起始地址
    v_base = kv_base + kv_size + head_id * head_dim # V 的起始地址（K 之后）
    k_write = k_out_ptr + ctx_id * k_out_stride_ctx + head_id * k_out_stride_head
    v_write = v_out_ptr + ctx_id * v_out_stride_ctx + head_id * v_out_stride_head

    # Load K and V
    # 生成 head_dim 范围内的偏移量
    offs = tl.arange(0, BLOCK_HD)
    mask_hd = offs < head_dim         # 完整 head_dim 的掩码
    mask_half = offs < half_rotary_dim  # RoPE 旋转半维度的掩码

    # 读取原始 K（float32 精度）和 V
    k_raw = tl.load(k_base + offs, mask=mask_hd, other=0.0).to(tl.float32)
    v_raw = tl.load(v_base + offs, mask=mask_hd, other=0.0)

    # RMSNorm on K
    # 计算 K 的逆均方根（RMSNorm）并乘以可学习权重
    inv_rms = tl.rsqrt(tl.sum(k_raw * k_raw) / head_dim + eps)
    norm_w = tl.load(k_norm_weight_ptr + offs, mask=mask_hd, other=1.0).to(tl.float32)
    k_normed = k_raw * inv_rms * norm_w

    # RoPE (neox style): k_first, k_second -> rotated
    # 加载对应位置的 cos 和 sin 值（neox 风格：前半/后半分别旋转）
    cos_sin_base = cos_sin_cache_ptr + position * cos_sin_stride_pos
    cos_v = tl.load(cos_sin_base + offs, mask=mask_half, other=1.0).to(tl.float32)
    sin_v = tl.load(
        cos_sin_base + half_rotary_dim + offs, mask=mask_half, other=0.0
    ).to(tl.float32)

    # Extract first/second halves of K for rotation
    # 提取 K 的前半部分（已归一化）和后半部分（加载原始值再归一化）
    k_first = tl.where(mask_half, k_normed, 0.0)
    k_second_raw = tl.load(
        k_base + half_rotary_dim + offs, mask=mask_half, other=0.0
    ).to(tl.float32)
    norm_w_second = tl.load(
        k_norm_weight_ptr + half_rotary_dim + offs, mask=mask_half, other=1.0
    ).to(tl.float32)
    k_second = k_second_raw * inv_rms * norm_w_second

    # Apply rotation
    # 应用 neox 风格 RoPE 旋转：[k_first, k_second] → [k_first*cos - k_second*sin, ...]
    k_rot_first = k_first * cos_v - k_second * sin_v
    k_rot_second = k_second * cos_v + k_first * sin_v

    # Store V (no transform)
    # V 不需要 RMSNorm 或 RoPE，直接写出
    tl.store(v_write + offs, v_raw, mask=mask_hd)

    # Store K: rotated halves + pass-through
    # 写出旋转后的 K 前半部分
    tl.store(k_write + offs, k_rot_first.to(v_raw.dtype), mask=mask_half)
    # 写出旋转后的 K 后半部分
    tl.store(
        k_write + half_rotary_dim + offs, k_rot_second.to(v_raw.dtype), mask=mask_half
    )
    # head_dim 中超出 rotary_dim 的维度不旋转，直接写出归一化值
    mask_pass = (offs >= rotary_dim) & (offs < head_dim)
    tl.store(k_write + offs, k_normed.to(v_raw.dtype), mask=mask_pass)


def _fused_norm_rope(
    kv: torch.Tensor,  # [total_ctx, kv_size*2]
    k_norm_weight: torch.Tensor,  # [head_dim]
    cos_sin_cache: torch.Tensor,  # [max_pos, rotary_dim]
    positions: torch.Tensor,  # [total_ctx]
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused RMSNorm + RoPE materialization for a single layer."""
    total_ctx = kv.shape[0]
    # 处理空 batch（如 prefill 阶段 token 数为 0）
    if total_ctx == 0:
        empty = torch.empty(
            (0, num_kv_heads, head_dim), dtype=kv.dtype, device=kv.device
        )
        return empty, empty

    kv_size = num_kv_heads * head_dim
    # 校验 KV 投影输出的形状：第二维应为 kv_size * 2（K + V）
    if kv.shape[1] != kv_size * 2:
        raise ValueError(
            "Invalid fused KV projection shape: "
            f"got {tuple(kv.shape)}, expected second dim {kv_size * 2}."
        )
    # 校验 rotary_dim：必须为正偶数且不超过 head_dim
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2 != 0:
        raise ValueError(
            "Invalid fused KV rotary/head dim pair: "
            f"rotary_dim={rotary_dim}, head_dim={head_dim}."
        )

    half_rotary_dim = rotary_dim // 2
    # head_dim 向上取整到 2 的幂次，以满足 Triton BLOCK_HD 约束
    BLOCK_HD = triton.next_power_of_2(head_dim)

    # Ensure int64 for indexing
    # 确保位置编码索引为 int64，避免 Triton 内核中的索引溢出
    if positions.device != kv.device:
        positions = positions.to(device=kv.device, dtype=torch.int64)
    elif positions.dtype != torch.int64:
        positions = positions.to(torch.int64)

    # 分配 K 和 V 输出缓冲区
    k_out = torch.empty(
        (total_ctx, num_kv_heads, head_dim), dtype=kv.dtype, device=kv.device
    )
    v_out = torch.empty_like(k_out)

    # 启动 Triton 核函数，grid = (total_ctx, num_kv_heads)
    _fused_norm_rope_kernel[(total_ctx, num_kv_heads)](
        kv,
        k_norm_weight,
        cos_sin_cache,
        positions,
        k_out,
        v_out,
        kv.stride(0),
        cos_sin_cache.stride(0),
        k_out.stride(0),
        k_out.stride(1),
        v_out.stride(0),
        v_out.stride(1),
        total_ctx,
        num_kv_heads,
        head_dim,
        kv_size,
        rotary_dim,
        half_rotary_dim,
        eps,
        BLOCK_HD,
    )
    return k_out, v_out


class FusedKVMaterializeHelper:
    """Fused KV materialization helper using batched projection.

    Uses torch.einsum for batched KV projection across all layers,
    then a Triton kernel for fused RMSNorm + RoPE materialization per layer.
    """
    # 融合 KV 物化辅助类：
    # 1. 用 torch.einsum 对所有层批量执行 KV 投影（一次 cuBLAS 调用）
    # 2. 对每层调用 Triton 核函数融合 RMSNorm + RoPE
    # 3. 通过回调函数 write_layer_kv 将结果写入 KV 缓存池

    def __init__(
        self,
        layers: List,        # 模型所有层列表（用于提取 KV 权重）
        rotary_emb,          # RoPE 模块（提供 cos/sin 缓存）
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
    ):
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rotary_emb = rotary_emb
        self.n_layers = len(layers)
        self.device = device

        # 从 RoPE 模块提取旋转维度和风格（neox/standard）
        self.rotary_dim = int(getattr(rotary_emb, "rotary_dim", head_dim))
        self.is_neox_style = bool(getattr(rotary_emb, "is_neox_style", True))

        # 当前仅支持 neox 风格 RoPE
        if not self.is_neox_style:
            raise NotImplementedError("Only neox-style RoPE is supported.")
        if self.rotary_dim <= 0 or self.rotary_dim > self.head_dim:
            raise ValueError(
                "Invalid fused KV rotary/head dim pair: "
                f"rotary_dim={self.rotary_dim}, head_dim={self.head_dim}."
            )

        # Pre-extract and stack weights for batched projection.
        # 从每层提取 KV 投影权重、K 的 RMSNorm 权重及 epsilon
        kv_weights = []
        self.k_norm_weights = []
        self.eps_values = []

        for layer_id, layer in enumerate(layers):
            attn = layer.self_attn
            # 校验各层的 num_kv_heads 和 head_dim 一致性
            if int(attn.num_kv_heads) != self.num_kv_heads:
                raise ValueError(
                    "num_kv_heads mismatch across layers for fused KV path: "
                    f"expected {self.num_kv_heads}, got {int(attn.num_kv_heads)} at layer {layer_id}."
                )
            if int(attn.head_dim) != self.head_dim:
                raise ValueError(
                    "head_dim mismatch across layers for fused KV path: "
                    f"expected {self.head_dim}, got {int(attn.head_dim)} at layer {layer_id}."
                )
            # 校验各层 RoPE 配置一致性
            layer_rotary_dim = int(
                getattr(attn.rotary_emb, "rotary_dim", self.head_dim)
            )
            layer_is_neox = bool(getattr(attn.rotary_emb, "is_neox_style", True))
            if (
                layer_rotary_dim != self.rotary_dim
                or layer_is_neox != self.is_neox_style
            ):
                raise ValueError(
                    "RoPE config mismatch across layers for fused KV path: "
                    f"expected (rotary_dim={self.rotary_dim}, neox={self.is_neox_style}), "
                    f"got (rotary_dim={layer_rotary_dim}, neox={layer_is_neox}) at layer {layer_id}."
                )

            # Extract KV portion of QKV weight
            # 从 QKV 联合权重中切出 KV 部分：[2*kv_size, hidden_size]
            qkv_w = attn.qkv_proj.weight
            kv_weight = qkv_w[attn.q_size : attn.q_size + 2 * attn.kv_size]
            kv_weights.append(kv_weight)
            self.k_norm_weights.append(attn.k_norm.weight)
            self.eps_values.append(attn.k_norm.variance_epsilon)

        # Stack for batched einsum: [n_layers, kv_size*2, hidden_size]
        # 将所有层的 KV 权重堆叠，用于后续批量 einsum 一次计算所有层的 KV 投影
        self.batched_kv_weight = torch.stack(kv_weights)

    def materialize(
        self,
        ctx_hidden: torch.Tensor,  # [total_ctx, hidden_size] — 上下文隐状态
        positions: torch.Tensor,   # [total_ctx] — 位置编码索引
        write_layer_kv: Callable[[int, torch.Tensor, torch.Tensor], None],  # 写入 KV 的回调
    ) -> None:
        """Materialize KV cache for all layers using batched projection."""
        total_ctx = ctx_hidden.shape[0]
        # 空 batch 直接返回
        if total_ctx == 0:
            return

        # 确保 positions 为一维
        if positions.ndim != 1:
            positions = positions.reshape(-1)
        if positions.numel() != total_ctx:
            raise ValueError(
                "positions must match ctx_hidden token count for fused KV materialization: "
                f"positions={positions.numel()}, total_ctx={total_ctx}."
            )

        # 确保 cos/sin 缓存足够长（覆盖 max_position）
        max_position = int(positions.max().item())
        ensure_cos_sin_cache_length = getattr(
            self.rotary_emb, "_ensure_cos_sin_cache_length", None
        )
        if callable(ensure_cos_sin_cache_length):
            ensure_cos_sin_cache_length(max_position)

        cos_sin_cache = self.rotary_emb.cos_sin_cache
        if max_position >= int(cos_sin_cache.shape[0]):
            raise RuntimeError(
                "RoPE cos/sin cache is too short for fused KV materialization: "
                f"max_position={max_position}, cache_len={int(cos_sin_cache.shape[0])}."
            )
        # 如果 cos_sin_cache 在不同设备上，移至 ctx_hidden 所在设备
        if cos_sin_cache.device != ctx_hidden.device:
            cos_sin_cache = cos_sin_cache.to(ctx_hidden.device)

        # Batched KV projection: [n_layers, total_ctx, kv_size*2]
        # 一次性批量计算所有层的 KV 投影，比逐层调用 matmul 更高效
        kv_all = torch.einsum("th,loh->lto", ctx_hidden, self.batched_kv_weight)

        # Per-layer fused norm/RoPE/materialize, then delegate writes to the KV pool.
        # 对每层执行融合 RMSNorm + RoPE，然后通过回调写入 KV 缓存池
        for layer_id in range(self.n_layers):
            cache_k, cache_v = _fused_norm_rope(
                kv_all[layer_id],
                self.k_norm_weights[layer_id],
                cos_sin_cache,
                positions,
                self.num_kv_heads,
                self.head_dim,
                self.rotary_dim,
                self.eps_values[layer_id],
            )
            # 调用外部提供的写入函数，将 (cache_k, cache_v) 写入页表管理的 KV 缓存池
            write_layer_kv(layer_id, cache_k, cache_v)
