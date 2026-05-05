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
"""
Memory-efficient attention for prefill.
It supports page size = 1 and prefill with KV cache (i.e. extend).
"""

# 导入 PyTorch 和 Triton 相关库
import torch
import triton
import triton.language as tl

# 导入 prefill 阶段的上下文注意力（用于 redundant_attention 的朴素实现）
from sglang.srt.layers.attention.triton_ops.prefill_attention import (
    context_attention_fwd,
)
# 导入硬件检测工具
from sglang.srt.utils import is_cuda, is_hip

# 检测是否为 CUDA 环境，并获取 GPU 计算能力（用于选择分块大小）
_is_cuda = is_cuda()
if _is_cuda:
    CUDA_CAPABILITY = torch.cuda.get_device_capability()

# 检测是否为 HIP（ROCm/AMD）环境
_is_hip = is_hip()


def _get_block_sizes_for_extend_attention(Lq: int, Lv: int):
    """
    Get block sizes and configuration for extend attention kernels.

    Args:
        Lq: Query head dimension
        Lv: Value head dimension

    Returns:
        tuple: (BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps)
    """
    # Determine BLOCK_DMODEL and BLOCK_DPE based on head dimension
    # 根据头维度大小确定主维度分块和 PE 维度分块
    # MLA 特殊配置：576 = 512(NOPE) + 64(RoPE)
    if Lq == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lq == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    elif Lq == 192:
        BLOCK_DMODEL = 128
        BLOCK_DPE = 64
    else:
        # 普通情况：整体填充到 2 的幂次，无独立 PE 维度
        BLOCK_DMODEL = triton.next_power_of_2(Lq)
        BLOCK_DPE = 0

    # Value 维度填充到 2 的幂次
    BLOCK_DV = triton.next_power_of_2(Lv)

    # Determine BLOCK_M, BLOCK_N, and num_warps based on hardware
    # 根据硬件架构选择最优的 Query/KV 分块大小和 warp 数量
    if _is_hip:
        # AMD GPU 使用固定分块大小
        BLOCK_M, BLOCK_N = (64, 64)
        num_warps = 4
    else:
        if _is_cuda and CUDA_CAPABILITY[0] == 12:
            # sm120 workstation Blackwell architecture (RTX Pro 6000) has a much smaller shared memory size (100K)
            # Blackwell 工作站架构（RTX Pro 6000），共享内存较小（100K），使用较小分块
            if Lq <= 128:
                BLOCK_M, BLOCK_N = (64, 128)
            elif Lq <= 256:
                BLOCK_M, BLOCK_N = (64, 64)
            else:
                BLOCK_M, BLOCK_N = (32, 32)
        elif _is_cuda and CUDA_CAPABILITY[0] == 10:
            # Blackwell data-center architecture (GB200, B200, sm_100a)
            # sm_100a has different register constraints from Hopper; Hopper block sizes
            # cause PTX register exhaustion (>255 regs) for large head dims (Lq=512).
            # Blackwell 数据中心架构（GB200, B200），寄存器约束不同于 Hopper
            if Lq <= 256:
                BLOCK_M, BLOCK_N = (64, 64)
            else:
                BLOCK_M, BLOCK_N = (16, 64)
        elif _is_cuda and CUDA_CAPABILITY[0] >= 9:
            # Hopper architecture (H100, etc.)
            # Hopper 架构（H100 等），共享内存更大，可使用更大分块
            if Lq <= 256:
                BLOCK_M, BLOCK_N = (128, 64)
            else:
                BLOCK_M, BLOCK_N = (32, 64)
        elif _is_cuda and CUDA_CAPABILITY[0] >= 8:
            # Ampere architecture (A100, etc.)
            # sm86/sm89 has a much smaller shared memory size (100K) than sm80 (160K)
            # Ampere 架构（A100 等），sm86/sm89 共享内存（100K）小于 sm80（160K）
            if CUDA_CAPABILITY[1] == 9 or CUDA_CAPABILITY[1] == 6:
                # sm86（RTX 30 系列）和 sm89（RTX 40 系列）共享内存受限
                if Lq <= 128:
                    BLOCK_M, BLOCK_N = (64, 128)
                elif Lq <= 256:
                    BLOCK_M, BLOCK_N = (64, 64)
                else:
                    BLOCK_M, BLOCK_N = (32, 32)
            else:
                # sm80（A100）共享内存最大（160K），使用最大分块
                if Lq <= 128:
                    BLOCK_M, BLOCK_N = (128, 128)
                elif Lq <= 256:
                    BLOCK_M, BLOCK_N = (64, 64)
                else:
                    BLOCK_M, BLOCK_N = (32, 64)
        else:
            # Older architectures
            # 旧架构（Volta 等），使用保守分块大小
            BLOCK_M, BLOCK_N = (64, 64) if Lq <= 128 else (32, 32)

        # 根据头维度选择 warp 数量（小维度用 4 个 warp，大维度用 8 个 warp）
        num_warps = 4 if Lq <= 64 else 8

    return BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps


# Triton JIT：双曲正切函数（用于 logit cap 裁剪）
@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    # tanh(x) = 2 * sigmoid(2x) - 1
    return 2 * tl.sigmoid(2 * x) - 1


# Triton JIT：并行将 prefix 和 extend 的 KV 索引合并到统一缓冲区
# 每个 Kernel 实例处理一个序列（sequence-level 并行）
@triton.jit
def _copy_unified_indices_kernel(
    # Input buffers
    prefix_kv_indptr,    # prefix KV 的 CSR 指针 [batch+1]
    prefix_kv_indices,   # prefix KV 的物理页索引
    extend_start_loc,    # extend 部分在 extend_kv_indices 中的起始位置 [batch]
    extend_seq_lens,     # extend 部分的序列长度 [batch]
    extend_kv_indices,   # extend KV 的物理页索引
    unified_kv_indptr,   # 统一 KV 的 CSR 指针（已提前计算）[batch+1]
    # Output buffer
    unified_kv_indices,  # 输出：统一 KV 物理页索引（先 prefix 后 extend）
    # Size
    bs,                  # batch size
):
    """
    Triton kernel to copy indices to unified buffer (parallel per sequence).
    Each thread block processes one sequence with vectorized loads/stores.
    """
    # 每个线程块处理一个序列
    pid = tl.program_id(0)

    # 若 pid 超出 batch size，直接返回（边界保护）
    if pid >= bs:
        return

    # Load sequence info
    # 加载当前序列的 prefix 起止索引和 extend 信息
    prefix_start = tl.load(prefix_kv_indptr + pid)
    prefix_end = tl.load(prefix_kv_indptr + pid + 1)
    extend_start = tl.load(extend_start_loc + pid)
    extend_len = tl.load(extend_seq_lens + pid)

    # 计算 prefix 长度和统一缓冲区的起始位置
    prefix_len = prefix_end - prefix_start
    unified_start = tl.load(unified_kv_indptr + pid)

    # Copy indices in vectorized chunks
    # 每次处理 128 个索引（向量化复制）
    BLOCK_SIZE: tl.constexpr = 128

    # Process prefix indices
    # 将 prefix KV 索引批量复制到统一缓冲区
    for block_start in range(0, prefix_len, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < prefix_len

        src_idx = prefix_start + offs    # 源：prefix_kv_indices 中的索引
        dst_idx = unified_start + offs   # 目标：unified_kv_indices 中的位置

        # 加载 prefix 索引并写入统一缓冲区
        vals = tl.load(prefix_kv_indices + src_idx, mask=mask, other=0)
        tl.store(unified_kv_indices + dst_idx, vals, mask=mask)

    # Process extend indices
    # 将 extend KV 索引批量复制到统一缓冲区（紧接 prefix 之后）
    for block_start in range(0, extend_len, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < extend_len

        src_idx = extend_start + offs              # 源：extend_kv_indices 中的索引
        dst_idx = unified_start + prefix_len + offs  # 目标：紧接 prefix 之后

        # 加载 extend 索引并写入统一缓冲区
        vals = tl.load(extend_kv_indices + src_idx, mask=mask, other=0)
        tl.store(unified_kv_indices + dst_idx, vals, mask=mask)


# Python 封装：构建统一的 KV 索引数组（prefix + extend 合并）
def build_unified_kv_indices(
    prefix_kv_indptr: torch.Tensor,    # prefix KV 的 CSR 指针
    prefix_kv_indices: torch.Tensor,   # prefix KV 的物理页索引
    extend_start_loc: torch.Tensor,    # extend 部分在 extend_kv_indices 中的起始位置
    extend_seq_lens: torch.Tensor,     # extend 部分的序列长度
    extend_kv_indices: torch.Tensor,   # extend KV 的物理页索引
    bs: int,                           # batch size
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build unified KV indices efficiently:
    - Use PyTorch's optimized cumsum (NVIDIA CUB) for indptr
    - Use Triton kernel for parallel index copying

    Returns:
        (unified_kv_indptr, unified_kv_indices, prefix_lens)
    """
    device = prefix_kv_indptr.device

    # 计算每个序列的 prefix 长度
    prefix_lens = prefix_kv_indptr[1 : bs + 1] - prefix_kv_indptr[:bs]

    # Create unified_kv_indptr avoiding direct assignment (for CUDA graph compatibility)
    # 计算每个序列的统一 KV 长度（prefix + extend），使用 cumsum 构建 CSR 指针
    unified_lens = prefix_lens + extend_seq_lens[:bs]
    unified_kv_indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),  # 起始 0
            torch.cumsum(unified_lens, dim=0),                  # 累积和
        ]
    )

    # 预分配统一 KV 索引缓冲区（最大长度 = prefix + extend 之和）
    max_unified_len = len(prefix_kv_indices) + len(extend_kv_indices)

    unified_kv_indices = torch.empty(max_unified_len, dtype=torch.int64, device=device)

    # Launch Triton kernel for parallel index copying
    # 启动 Triton Kernel 并行复制索引（每个序列一个线程块）
    _copy_unified_indices_kernel[(bs,)](
        prefix_kv_indptr,
        prefix_kv_indices,
        extend_start_loc,
        extend_seq_lens,
        extend_kv_indices,
        unified_kv_indptr,
        unified_kv_indices,
        bs,
    )

    return unified_kv_indptr, unified_kv_indices, prefix_lens



# Triton JIT：extend 阶段注意力前向 Kernel（两阶段：先 prefix KV，再 extend KV）
# Grid = (batch_size, num_heads, ceil(max_len_extend / BLOCK_M))
@triton.jit
def _fwd_kernel(
    Q_Extend,      # extend 部分的 Query [总extend_token数, num_heads, Lq]
    K_Extend,      # extend 部分的 Key（新增 token 的 Key）
    V_Extend,      # extend 部分的 Value（新增 token 的 Value）
    O_Extend,      # 输出张量
    K_Buffer,      # KV 缓存中的 Key（prefix 部分）
    V_Buffer,      # KV 缓存中的 Value（prefix 部分）
    qo_indptr,     # Query/Output 的 CSR 指针（按序列分段）[batch+1]
    kv_indptr,     # prefix KV 的 CSR 指针 [batch+1]
    kv_indices,    # prefix KV 的物理页索引
    mask_ptr,      # 自定义注意力掩码指针（投机采样树注意力）
    mask_indptr,   # 自定义掩码的 CSR 指针 [batch+1]
    sink_ptr,      # 注意力 sink 的 lse 指针（StreamingLLM 等）
    window_kv_offset_ptr,  # 滑动窗口的 KV 偏移量 [batch]
    sm_scale,      # softmax 缩放因子
    k_scale,       # Key 量化缩放因子
    v_scale,       # Value 量化缩放因子
    kv_group_num,  # GQA 分组数
    stride_qbs,    # Q 在 token 维度的步长
    stride_qh,     # Q 在 head 维度的步长
    stride_kbs,    # K_Extend 在 token 维度的步长
    stride_kh,     # K_Extend 在 head 维度的步长
    stride_vbs,    # V_Extend 在 token 维度的步长
    stride_vh,     # V_Extend 在 head 维度的步长
    stride_obs,    # Output 在 token 维度的步长
    stride_oh,     # Output 在 head 维度的步长
    stride_buf_kbs,  # K_Buffer 在 token 维度的步长
    stride_buf_kh,   # K_Buffer 在 head 维度的步长
    stride_buf_vbs,  # V_Buffer 在 token 维度的步长
    stride_buf_vh,   # V_Buffer 在 head 维度的步长
    SLIDING_WINDOW_SIZE: tl.constexpr,      # 滑动窗口大小（-1=不使用）
    logit_cap: tl.constexpr,                # logit 裁剪上限
    xai_temperature_len: tl.constexpr,      # xAI 温度参数
    Lq: tl.constexpr,                       # Query 实际头维度
    Lv: tl.constexpr,                       # Value 实际头维度
    BLOCK_DMODEL: tl.constexpr,             # 主维度分块大小
    BLOCK_DPE: tl.constexpr,               # PE 维度分块大小（MLA 专用，0=不使用）
    BLOCK_DV: tl.constexpr,               # Value 维度分块大小
    BLOCK_M: tl.constexpr,               # Query 方向分块大小
    BLOCK_N: tl.constexpr,               # KV 方向分块大小
    USE_CUSTOM_MASK: tl.constexpr,       # 是否使用自定义注意力掩码
    IS_CAUSAL: tl.constexpr,             # 是否使用因果掩码
    SKIP_PREFIX_CUSTOM_MASK: tl.constexpr,  # 是否跳过 prefix 部分的自定义掩码
    STORE_TRANSPOSE: tl.constexpr,       # 是否以转置格式存储输出（HIP 需要）
    HAS_SINK: tl.constexpr,             # 是否包含注意力 sink
):
    # 获取序列索引、头索引和 Query 分块索引
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    # GQA：计算对应的 KV 头索引
    cur_kv_head = cur_head // kv_group_num

    # 加载当前序列的 extend 部分起始位置和长度
    cur_seq_extend_start_idx = tl.load(qo_indptr + cur_seq)
    cur_seq_len_extend = tl.load(qo_indptr + cur_seq + 1) - cur_seq_extend_start_idx
    # 加载当前序列的 prefix KV 起始位置
    cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)
    # prefix 长度 = KV 总长度（因为 kv_indptr 这里只包含 prefix）
    cur_seq_len_prefix = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx
    # 总序列长度 = prefix + extend
    cur_seq_len = cur_seq_len_prefix + cur_seq_len_extend

    # 若使用自定义掩码，加载掩码的起始位置
    if USE_CUSTOM_MASK:
        cur_seq_mask_start_idx = tl.load(mask_indptr + cur_seq)

    # For SWA, we should only load the mask in the sliding window
    # 滑动窗口注意力（SWA）：加载窗口 KV 偏移量
    window_kv_offset = 0
    if USE_CUSTOM_MASK and SLIDING_WINDOW_SIZE > 0:
        window_kv_offset = tl.load(window_kv_offset_ptr + cur_seq)

    # 初始化维度索引和掩码
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_m = tl.arange(0, BLOCK_M)
    # 有效 Query token 掩码（过滤超出 extend 长度的部分）
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend

    mask_d = offs_d < Lq   # Key/Query 主维度有效性掩码
    mask_dv = offs_dv < Lv  # Value 维度有效性掩码

    # xAI 温度缩放：按序列位置缩放注意力分数
    if xai_temperature_len > 0:
        offs_qidx = cur_seq_len_prefix + cur_block_m * BLOCK_M + offs_m
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        xai_temperature_reg = tl.where(
            offs_qidx > xai_temperature_len,
            tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale,
            1.0,
        )

    # 计算 Query 主维度的内存地址偏移
    offs_q = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    # 加载 Query 主维度（NOPE 或普通维度）
    q = tl.load(
        Q_Extend + offs_q, mask=(mask_m[:, None]) & (mask_d[None, :]), other=0.0
    )

    # 若有 PE 维度（MLA），加载 Q_PE
    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        offs_qpe = (
            (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
            * stride_qbs
            + cur_head * stride_qh
            + offs_dpe[None, :]
        )
        qpe = tl.load(Q_Extend + offs_qpe, mask=mask_m[:, None], other=0.0)

    # stage 1: compute scores with prefix
    # 阶段1：对 prefix KV 缓存计算注意力分数（prefix 与所有 extend Query 的交叉注意力）
    offs_n = tl.arange(0, BLOCK_N)

    # 初始化 Flash Attention 统计量
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)  # 加权 Value 累积器
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)             # 归一化因子
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # 最大 logit

    # 遍历 prefix KV 分块（prefix 中所有 token 的 KV）
    for start_n in range(0, cur_seq_len_prefix, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # prefix 范围内的有效性掩码
        mask_n = (start_n + offs_n) < cur_seq_len_prefix

        final_mask = mask_m[:, None] & mask_n[None, :]
        # 若使用自定义掩码（投机采样树注意力），加载并应用自定义掩码
        if USE_CUSTOM_MASK and not SKIP_PREFIX_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None])
                * (cur_seq_len + window_kv_offset)
                + window_kv_offset
                + start_n
                + offs_n[None, :],
                mask=(mask_m[:, None] & mask_n[None, :]),
                other=0,
            )
            final_mask &= custom_mask
        # 滑动窗口掩码：仅允许 Q 关注距离不超过 SLIDING_WINDOW_SIZE 的 K
        if SLIDING_WINDOW_SIZE > 0:
            # Add mask where q_id <= kv_id + sliding_window_size
            # q_id = prefix_len + cur_m, kv_id = cur_n
            window_mask = (
                cur_seq_len_prefix + cur_block_m * BLOCK_M + offs_m[:, None]
            ) <= (start_n + offs_n[None, :] + SLIDING_WINDOW_SIZE)
            final_mask &= window_mask

        # 检查是否可以跳过整个 tile（所有位置都被掩码屏蔽）
        SKIP_TILE = False
        if (USE_CUSTOM_MASK and not SKIP_PREFIX_CUSTOM_MASK) or SLIDING_WINDOW_SIZE > 0:
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            # 加载 prefix KV 的物理页索引
            offs_kv_loc = tl.load(
                kv_indices + cur_seq_kv_start_idx + start_n + offs_n,
                mask=mask_n,
                other=0,
            )

            # load k in transposed way
            # 加载 K（转置形式，用于矩阵乘法 Q*K^T）
            offs_buf_k = (
                offs_kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[:, None]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(mask_n[None, :]) & (mask_d[:, None]),
                other=0.0,
            )

            # 计算 Q * K^T（主维度矩阵乘法）
            qk = tl.dot(q.to(k.dtype), k)
            # 若有 PE 维度，追加 RoPE 部分的点积
            if BLOCK_DPE > 0:
                offs_kpe = (
                    offs_kv_loc[None, :] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_dpe[:, None]
                )
                kpe = tl.load(
                    K_Buffer + offs_kpe,
                    mask=mask_n[None, :],
                    other=0.0,
                )
                qk += tl.dot(qpe.to(kpe.dtype), kpe)
            # 应用缩放因子（含 k_scale 用于量化反量化）
            qk *= sm_scale * k_scale

            # logit 裁剪（可选）
            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            # xAI 温度缩放（可选）
            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            # 应用最终掩码（屏蔽无效位置）
            qk = tl.where(final_mask, qk, float("-inf"))

            # Flash Attention online softmax 更新
            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)

            re_scale = tl.exp(e_max - n_e_max)  # 旧累积的衰减因子
            p = tl.exp(qk - n_e_max[:, None])   # 当前块注意力概率
            deno = deno * re_scale + tl.sum(p, 1)

            # 加载 prefix V（从 KV 缓存读取）
            offs_buf_v = (
                offs_kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=mask_n[:, None] & mask_dv[None, :],
                other=0.0,
            )
            p = p.to(v.dtype)
            # 累积加权 Value（乘以 v_scale 进行量化反量化）
            acc = acc * re_scale[:, None] + tl.dot(p, v) * v_scale

            e_max = n_e_max

    # stage 2: compute the triangle part
    # 阶段2：对 extend 部分（因果三角形区域）计算注意力分数

    # 非因果模式遍历所有 extend token，因果模式只遍历到当前 Query 块末尾
    cur_block_m_end = (
        cur_seq_len_extend
        if not IS_CAUSAL
        else tl.minimum(cur_seq_len_extend, (cur_block_m + 1) * BLOCK_M)
    )
    # 遍历 extend KV 分块（extend 部分的自注意力）
    for start_n in range(0, cur_block_m_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_block_m_end

        final_mask = mask_m[:, None] & mask_n[None, :]
        # 应用自定义掩码（extend 部分）
        if USE_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None])
                * (cur_seq_len + window_kv_offset)
                + window_kv_offset
                + cur_seq_len_prefix
                + start_n
                + offs_n[None, :],
                mask=(mask_m[:, None] & mask_n[None, :]),
                other=0,
            )
            custom_mask &= mask_m[:, None] & mask_n[None, :]
            final_mask &= custom_mask
        elif IS_CAUSAL:
            # 因果掩码：每个 Query 位置只关注其之前（含自身）的 Key 位置
            mask_causual = (cur_block_m * BLOCK_M + offs_m[:, None]) >= (
                start_n + offs_n[None, :]
            )
            mask_causual &= mask_m[:, None] & mask_n[None, :]
            final_mask &= mask_causual
        else:
            # 非因果：屏蔽超出序列长度的 padding 位置
            mask_non_causal = mask_m[:, None] & mask_n[None, :]
            final_mask &= mask_non_causal

        # 滑动窗口掩码（extend 部分）
        if SLIDING_WINDOW_SIZE > 0:
            # Add mask where q_id <= kv_id + sliding_window_size
            window_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) <= (
                start_n + offs_n[None, :] + SLIDING_WINDOW_SIZE
            )
            final_mask &= window_mask

        # 检查是否可跳过整个 tile
        SKIP_TILE = False
        if USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            # load k in transposed way
            # 加载 extend 部分的 K（直接从 K_Extend 读取，转置形式）
            offs_k = (
                (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
                + cur_kv_head * stride_kh
                + offs_d[:, None]
            )
            k = tl.load(
                K_Extend + offs_k, mask=(mask_n[None, :]) & (mask_d[:, None]), other=0.0
            )

            # 计算 Q * K^T（矩阵乘法，float32 精度）
            qk = tl.dot(q, k, out_dtype=tl.float32)
            # 若有 PE 维度，追加 extend 部分的 RoPE 点积
            if BLOCK_DPE > 0:
                offs_kpe = (
                    (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
                    + cur_kv_head * stride_kh
                    + offs_dpe[:, None]
                )
                kpe = tl.load(
                    K_Extend + offs_kpe,
                    mask=mask_n[None, :],
                    other=0.0,
                )
                qk += tl.dot(qpe, kpe)

            # 应用缩放因子（extend 部分不需要额外的 k_scale）
            qk *= sm_scale

            # logit 裁剪（可选）
            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            # xAI 温度缩放（可选）
            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            # 应用最终掩码
            qk = tl.where(final_mask, qk, float("-inf"))

            # Flash Attention online softmax 更新
            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)

            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)

            # 加载 extend 部分的 V（直接从 V_Extend 读取）
            offs_v = (
                (cur_seq_extend_start_idx + start_n + offs_n[:, None]) * stride_vbs
                + cur_kv_head * stride_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Extend + offs_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0
            )
            p = p.to(v.dtype)
            # 累积加权 Value（extend 部分无 v_scale）
            acc = acc * re_scale[:, None] + tl.dot(p, v)

            e_max = n_e_max

    # 若有 sink token，将 sink 的贡献加入归一化因子
    if HAS_SINK:
        cur_sink = tl.load(sink_ptr + cur_head)
        deno += tl.exp(cur_sink - e_max)

    # 计算输出的内存地址偏移
    offs_o = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
    # 以转置格式存储（HIP/ROCm 需要）或正常格式存储
    if STORE_TRANSPOSE:
        tl.store(
            O_Extend + offs_o.T,
            (acc / deno[:, None]).T,
            mask=(mask_m[:, None] & mask_dv[None, :]).T,
        )
    else:
        # 将归一化后的注意力输出写入全局内存
        tl.store(
            O_Extend + offs_o,
            acc / deno[:, None],
            mask=mask_m[:, None] & mask_dv[None, :],
        )


# Python 封装：extend 阶段注意力前向（两阶段 prefix + extend）
def extend_attention_fwd(
    q_extend,     # extend 部分的 Query
    k_extend,     # extend 部分的 Key
    v_extend,     # extend 部分的 Value
    o_extend,     # 输出张量
    k_buffer,     # KV 缓存中的 prefix Key
    v_buffer,     # KV 缓存中的 prefix Value
    qo_indptr,    # Query/Output 的 CSR 指针
    kv_indptr,    # prefix KV 的 CSR 指针
    kv_indices,   # prefix KV 的物理页索引
    custom_mask,  # 自定义注意力掩码（可选）
    is_causal,    # 是否使用因果掩码
    mask_indptr,  # 自定义掩码的 CSR 指针
    max_len_extend,  # 最大 extend 序列长度
    k_scale,      # Key 量化缩放因子
    v_scale,      # Value 量化缩放因子
    sm_scale=None,           # softmax 缩放因子（默认 1/sqrt(Lq)）
    logit_cap=0.0,           # logit 裁剪上限
    skip_prefix_custom_mask=True,  # 是否跳过 prefix 部分的自定义掩码
    sliding_window_size=-1,  # 滑动窗口大小
    sinks=None,              # 注意力 sink
    window_kv_offsets=None,  # 滑动窗口的 KV 偏移量
    xai_temperature_len=-1,  # xAI 温度参数
):
    """
    q_extend, k_extend, v_extend, o_extend: contiguous tensors

    k_buffer, v_buffer: (prefix + extend) tensors in mem_manager
    """
    # 获取 Q/K/V 的头维度大小
    Lq, Lk, Lv = (
        q_extend.shape[-1],
        k_extend.shape[-1],
        v_extend.shape[-1],
    )

    # Get block sizes and configuration
    # 根据头维度和硬件获取最优分块配置
    BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps = (
        _get_block_sizes_for_extend_attention(Lq, Lv)
    )

    # 若未指定 sm_scale，使用默认值 1/sqrt(Lq)
    sm_scale = sm_scale or 1.0 / (Lq**0.5)
    batch_size, head_num = qo_indptr.shape[0] - 1, q_extend.shape[1]
    # 计算 GQA 分组数
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]

    USE_CUSTOM_MASK = custom_mask is not None
    # Skip custom mask for prefix part
    SKIP_PREFIX_CUSTOM_MASK = skip_prefix_custom_mask

    HAS_SINK = sinks is not None

    # grid = (batch_size, num_heads, ceil(max_len_extend / BLOCK_M))
    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))
    num_stages = 1

    # AMD ROCm 专属调优参数
    extra_kargs = {}
    if _is_hip:
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}

    # 启动 extend 注意力 Triton Kernel
    _fwd_kernel[grid](
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        mask_indptr,
        sinks,
        window_kv_offsets,
        sm_scale,
        k_scale,
        v_scale,
        kv_group_num,
        q_extend.stride(0),
        q_extend.stride(1),
        k_extend.stride(0),
        k_extend.stride(1),
        v_extend.stride(0),
        v_extend.stride(1),
        o_extend.stride(0),
        o_extend.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        SLIDING_WINDOW_SIZE=sliding_window_size,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        Lq=Lq,
        Lv=Lv,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        IS_CAUSAL=is_causal,
        SKIP_PREFIX_CUSTOM_MASK=SKIP_PREFIX_CUSTOM_MASK,
        HAS_SINK=HAS_SINK,
        STORE_TRANSPOSE=_is_hip,  # HIP 需要以转置格式存储
        num_warps=num_warps,
        num_stages=num_stages,
        **extra_kargs,
    )


# 朴素参考实现：将 extend 部分的 Q 插入到完整 KV 缓存中，调用 prefill 注意力
# 主要用于测试和验证
def redundant_attention(
    q_extend,          # extend 部分的 Query
    o_extend,          # 输出张量
    k_buffer,          # KV 缓存（包含所有 prefix + extend 的 Key）
    v_buffer,          # KV 缓存（包含所有 prefix + extend 的 Value）
    b_req_idx,         # 每个 batch 的请求索引
    b_start_loc,       # 每个序列在 KV 缓存中的起始位置
    b_seq_len,         # 每个序列的完整长度（prefix + extend）
    b_seq_len_prefix,  # 每个序列的 prefix 长度
    max_len_in_batch,  # batch 中最长序列的长度
):
    # 获取总 token 数和头数
    total_token_num = k_buffer.shape[0]
    B, H_Q, D = b_req_idx.shape[0], q_extend.shape[-2], q_extend.shape[-1]
    # 分配完整 Q 缓冲区（与 KV 缓存对齐）
    q_buffer = torch.empty(
        (total_token_num, H_Q, D), dtype=q_extend.dtype, device=q_extend.device
    )

    # 将 extend 部分的 Q 插入到 q_buffer 中对应位置（prefix 之后）
    pt = 0
    for i in range(B):
        cur_seq_len_extend = b_seq_len[i] - b_seq_len_prefix[i]
        pl, pr = b_start_loc[i] + b_seq_len_prefix[i], b_start_loc[i] + b_seq_len[i]
        q_buffer[pl:pr] = q_extend[pt : pt + cur_seq_len_extend]
        pt += cur_seq_len_extend

    # 调用 prefill 注意力（完整序列注意力）
    o_buffer = torch.empty_like(q_buffer)
    context_attention_fwd(
        q_buffer, k_buffer, v_buffer, o_buffer, b_start_loc, b_seq_len, max_len_in_batch
    )

    # 从 o_buffer 中提取 extend 部分的输出
    pt = 0
    for i in range(B):
        cur_seq_len_extend = b_seq_len[i] - b_seq_len_prefix[i]
        pl, pr = b_start_loc[i] + b_seq_len_prefix[i], b_start_loc[i] + b_seq_len[i]
        o_extend[pt : pt + cur_seq_len_extend] = o_buffer[pl:pr]
        pt += cur_seq_len_extend


# Triton JIT：统一单阶段 extend 注意力 Kernel（确定性推理版本）
# 与双阶段 _fwd_kernel 不同，本 Kernel 通过统一的 kv_indices 同时访问 prefix 和 extend KV
# 避免了分阶段合并的非确定性，适用于需要确定性输出的推理场景
@triton.jit
def _fwd_kernel_unified(
    Q,              # Query 张量 [num_tokens, num_heads, head_dim]
    O,              # 输出张量 [num_tokens, num_heads, head_dim]
    K_Buffer,       # Key 缓存（包含 prefix 和 extend 所有 KV）
    V_Buffer,       # Value 缓存（包含 prefix 和 extend 所有 KV）
    qo_indptr,      # Query/Output 序列指针（CSR 格式）[batch+1]
    kv_indptr,      # KV 序列指针（包含 prefix+extend 的统一索引）[batch+1]
    kv_indices,     # 统一 KV 物理页索引（prefix+extend 合并后）
    prefix_lens,    # 每个序列的 prefix 长度 [batch]
    mask_ptr,       # 自定义掩码数据（投机解码树注意力）
    mask_indptr,    # 自定义掩码序列指针 [batch+1]
    sink_ptr,       # Sink token 注意力 logit 值（每头一个）
    window_start_pos,  # 滑动窗口每个序列的起始绝对位置 [batch]
    sm_scale_withk, # 已融合 k_scale 的 softmax 缩放因子
    v_scale,        # Value 反量化缩放因子
    kv_group_num,   # GQA 分组数（每个 KV 头对应的 Q 头数）
    stride_qbs,     # Q 在 token 维度的步长
    stride_qh,      # Q 在 head 维度的步长
    stride_obs,     # O 在 token 维度的步长
    stride_oh,      # O 在 head 维度的步长
    stride_buf_kbs, # K_Buffer 在 token 维度的步长
    stride_buf_kh,  # K_Buffer 在 head 维度的步长
    stride_buf_vbs, # V_Buffer 在 token 维度的步长
    stride_buf_vh,  # V_Buffer 在 head 维度的步长
    SLIDING_WINDOW_SIZE: tl.constexpr,  # 滑动窗口大小（-1 表示无）
    logit_cap: tl.constexpr,            # logit 裁剪上限（0 表示不裁剪）
    xai_temperature_len: tl.constexpr,  # xAI 温度缩放长度阈值（-1 表示不启用）
    Lq: tl.constexpr,   # Q 实际头维度（NOPE+RoPE）
    Lv: tl.constexpr,   # V 实际头维度
    BLOCK_DMODEL: tl.constexpr,  # NOPE 部分头维度填充后大小
    BLOCK_DPE: tl.constexpr,     # RoPE 部分头维度（MLA 专用，非 MLA 为 0）
    BLOCK_DV: tl.constexpr,      # V 头维度填充后大小
    BLOCK_M: tl.constexpr,       # Query 分块大小
    BLOCK_N: tl.constexpr,       # KV 分块大小
    IS_CAUSAL: tl.constexpr,     # 是否启用因果掩码（extend 部分的下三角掩码）
    USE_CUSTOM_MASK: tl.constexpr,  # 是否使用自定义掩码（投机解码）
    HAS_SINK: tl.constexpr,         # 是否有 sink token（StreamingLLM）
):
    """
    Unified 1-stage kernel for deterministic extend attention.
    Both prefix and extend KV are accessed through the unified kv_indices.
    """
    # 获取当前 Kernel 对应的序列、注意力头、Query 分块索引
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    # 通过 GQA 映射计算对应的 KV 头索引
    cur_kv_head = cur_head // kv_group_num

    # Load sequence information
    # 加载当前序列的 Query 起始索引和长度
    cur_seq_q_start_idx = tl.load(qo_indptr + cur_seq)
    cur_seq_q_len = tl.load(qo_indptr + cur_seq + 1) - cur_seq_q_start_idx
    # 加载当前序列的统一 KV 起始索引和总长度（prefix+extend）
    cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)
    cur_seq_kv_len = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx
    # 加载当前序列的 prefix 长度（用于因果掩码边界判断）
    cur_seq_prefix_len = tl.load(prefix_lens + cur_seq)

    # Load window start position for sliding window attention
    # This is the absolute position of the first key in the window (0 if no sliding window)
    # 若启用滑动窗口，加载当前序列窗口的起始绝对位置
    cur_window_start = 0
    if SLIDING_WINDOW_SIZE > 0:
        cur_window_start = tl.load(window_start_pos + cur_seq)

    # Load custom mask start index if using custom mask (for speculative decoding)
    # 若使用自定义掩码，加载当前序列掩码数据的起始偏移
    if USE_CUSTOM_MASK:
        cur_seq_mask_start_idx = tl.load(mask_indptr + cur_seq)

    # 初始化维度偏移索引
    offs_d = tl.arange(0, BLOCK_DMODEL)   # NOPE 维度偏移
    offs_dv = tl.arange(0, BLOCK_DV)       # V 维度偏移
    offs_m = tl.arange(0, BLOCK_M)         # Query 分块内偏移
    # 有效 Query 位置掩码（过滤超出序列长度的 padding）
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_q_len
    # NOPE 维度有效性掩码（过滤 padding 维度）
    mask_d = offs_d < Lq
    # V 维度有效性掩码
    mask_dv = offs_dv < Lv

    # XAI temperature handling
    # xAI 温度缩放：根据 Q 的绝对位置计算温度系数（位置越靠后，温度越小）
    if xai_temperature_len > 0:
        # 当前 Q token 在完整序列中的绝对位置（prefix + extend 偏移）
        offs_qidx = cur_seq_prefix_len + cur_block_m * BLOCK_M + offs_m
        # 位置在 temperature_len 内时系数为 1，超出后按 temperature_len/pos 衰减
        xai_temperature_reg = tl.where(
            offs_qidx < xai_temperature_len,
            1.0,
            xai_temperature_len / (offs_qidx + 1.0),
        )

    # Load Q
    # 计算 Q（NOPE 部分）的内存偏移并加载
    offs_q = (
        (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(Q + offs_q, mask=(mask_m[:, None]) & (mask_d[None, :]), other=0.0)

    # 若启用 MLA（RoPE 部分维度 > 0），额外加载 Q 的 PE 部分（位置编码维度）
    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        offs_qpe = (
            (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
            + cur_head * stride_qh
            + offs_dpe[None, :]
        )
        qpe = tl.load(Q + offs_qpe, mask=mask_m[:, None], other=0.0)

    # Initialize accumulators
    # 初始化 Flash Attention online 统计量
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)       # 加权 Value 累积器
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)                  # softmax 归一化分母
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # 当前最大 logit

    # Unified loop: process all KV tokens (prefix + extend)
    # 统一循环：遍历所有 KV token（prefix 和 extend 合并在同一 kv_indices 中）
    for start_n in range(0, cur_seq_kv_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # 当前 KV 分块内有效 token 掩码
        mask_n = (start_n + offs_n) < cur_seq_kv_len

        # Compute mask
        # 基础掩码：Q 和 K 都在有效范围内
        final_mask = mask_m[:, None] & mask_n[None, :]

        # Apply custom mask if provided
        # 若有自定义掩码（投机解码树注意力），从全局掩码表中加载并叠加
        if USE_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None]) * cur_seq_kv_len
                + start_n
                + offs_n[None, :],
                mask=(mask_m[:, None] & mask_n[None, :]),
                other=0,
            )
            final_mask &= custom_mask

        # Apply causal mask for extend part
        # 因果掩码：仅对 extend 部分（KV index >= prefix_len）应用下三角掩码
        # prefix 部分（KV index < prefix_len）无因果限制，Q 可完全访问
        if IS_CAUSAL and not USE_CUSTOM_MASK:
            # Determine if current KV block is in extend region
            # Only apply causal mask when both Q and K are in extend region
            # Q 在 extend 中的相对位置
            q_idx = cur_block_m * BLOCK_M + offs_m[:, None]
            # K 在统一 KV 数组中的绝对位置
            k_idx_in_total = start_n + offs_n[None, :]

            # Causal mask: q_idx >= (k_idx - prefix_len) when k_idx >= prefix_len
            # For prefix region (k_idx < prefix_len), no causal mask
            # 判断 K 是否位于 extend 区域
            k_is_extend = k_idx_in_total >= cur_seq_prefix_len
            # K 在 extend 区域内的相对位置
            k_idx_in_extend = k_idx_in_total - cur_seq_prefix_len
            # 因果掩码：Q 的 extend 位置 >= K 的 extend 位置（下三角）
            causal_mask = tl.where(
                k_is_extend,
                q_idx >= k_idx_in_extend,
                True,  # No causal mask for prefix
            )
            final_mask &= causal_mask

        # 滑动窗口掩码：Q 只能关注绝对位置差在窗口大小内的 K
        if SLIDING_WINDOW_SIZE > 0:
            # Sliding window mask with correct absolute positions
            # Q absolute position: window_start + prefix_len + q_position_in_extend
            # Q 的绝对位置（窗口起始 + prefix 长度 + Q 在 extend 中的位置）
            q_abs_pos = (
                cur_window_start
                + cur_seq_prefix_len
                + cur_block_m * BLOCK_M
                + offs_m[:, None]
            )

            # K absolute position: window_start + k_index_in_unified_array
            # K 的绝对位置（窗口起始 + K 在统一数组中的位置）
            k_abs_pos = cur_window_start + start_n + offs_n[None, :]

            # Sliding window: query can attend to keys within window_size
            # 滑动窗口：Q 只关注满足 q_abs_pos <= k_abs_pos + window_size 的 K
            window_mask = q_abs_pos <= (k_abs_pos + SLIDING_WINDOW_SIZE)
            final_mask &= window_mask

        # Check if we can skip this tile
        # 若整块 tile 内所有位置都被掩码（full mask out），跳过本次迭代（优化性能）
        SKIP_TILE = False
        if USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            # Load KV indices
            # 通过统一 kv_indices 加载当前 KV 分块的物理页地址
            offs_kv_loc = tl.load(
                kv_indices + cur_seq_kv_start_idx + start_n + offs_n,
                mask=mask_n,
                other=0,
            )

            # Load K
            # 计算 K（NOPE 部分）的内存偏移（转置形式：[head_dim, BLOCK_N]）
            offs_buf_k = (
                offs_kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[:, None]
            )
            # 加载 K 分块（NOPE 部分）
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(mask_n[None, :]) & (mask_d[:, None]),
                other=0.0,
            )

            # Compute QK
            # 计算 NOPE 部分的注意力分数：qk = q @ k^T
            qk = tl.dot(q.to(k.dtype), k)
            # 若启用 MLA，额外加载 K 的 PE 部分并累加到注意力分数
            if BLOCK_DPE > 0:
                offs_kpe = (
                    offs_kv_loc[None, :] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_dpe[:, None]
                )
                # 加载 K 的 RoPE 部分（PE 维度）
                kpe = tl.load(
                    K_Buffer + offs_kpe,
                    mask=mask_n[None, :],
                    other=0.0,
                )
                # 累加 RoPE 部分分数：qk += qpe @ kpe^T
                qk += tl.dot(qpe.to(kpe.dtype), kpe)

            # 应用 softmax 缩放因子（已融合 k_scale）
            qk *= sm_scale_withk

            # 若设置 logit_cap，使用 tanh 裁剪注意力分数（防止极端值）
            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            # 应用 xAI 温度缩放（按位置调整注意力强度）
            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            # 应用最终复合掩码（无效位置设为 -inf）
            qk = tl.where(final_mask, qk, float("-inf"))

            # Online softmax
            # Flash Attention online softmax 更新：计算当前块最大值并更新累积统计量
            row_max = tl.max(qk, 1)
            # 防止全为 -inf 时出现 NaN，用 -1e20 替代 -inf
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)

            # 计算旧累积器的衰减因子和当前块的未归一化概率
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            # 更新归一化分母
            deno = deno * re_scale + tl.sum(p, 1)

            # Load V
            # 计算 V 的内存偏移并加载当前 KV 分块的 Value
            offs_buf_v = (
                offs_kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=mask_n[:, None] & mask_dv[None, :],
                other=0.0,
            )
            p = p.to(v.dtype)
            # 更新加权 Value 累积器：acc = acc * re_scale + p @ v
            acc = acc * re_scale[:, None] + tl.dot(p, v)

            # 更新全局最大值
            e_max = n_e_max

    # Handle sink tokens
    # StreamingLLM Sink token 处理：将 sink 的 logit 贡献加入归一化分母
    if HAS_SINK:
        cur_sink = tl.load(sink_ptr + cur_head)
        deno += tl.exp(cur_sink - e_max)

    # Store output
    # 计算输出的内存偏移，将归一化后的注意力输出（乘以 v_scale 反量化）写回
    offs_o = (
        (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
    tl.store(
        O + offs_o,
        acc / deno[:, None] * v_scale,  # 归一化并乘以 v_scale 还原量化精度
        mask=mask_m[:, None] & mask_dv[None, :],
    )


# Python 封装：统一单阶段 extend 注意力前向计算（确定性推理版本）
# 与 extend_attention_fwd 的双阶段实现不同，此版本通过统一 kv_indices 实现单阶段计算
# 适用于需要确定性输出的推理场景（如验证、对齐测试等）
def extend_attention_fwd_unified(
    q,              # Query 张量 [num_tokens, num_heads, head_dim]
    o,              # 输出张量 [num_tokens, num_heads, head_dim]
    k_buffer,       # Key 缓存（统一 prefix+extend KV）
    v_buffer,       # Value 缓存（统一 prefix+extend KV）
    k_scale,        # Key 量化缩放因子
    v_scale,        # Value 量化缩放因子
    qo_indptr,      # Query/Output 序列指针（CSR 格式）[batch+1]
    kv_indptr,      # 统一 KV 序列指针（prefix+extend 合并）[batch+1]
    kv_indices,     # 统一 KV 物理页索引
    prefix_lens,    # 每个序列的 prefix 长度 [batch]
    max_len_extend, # batch 中最大 extend 序列长度
    custom_mask=None,        # 自定义注意力掩码（投机解码树注意力）
    mask_indptr=None,        # 自定义掩码序列指针 [batch+1]
    sm_scale=None,           # softmax 缩放因子（默认 1/sqrt(Lq)）
    logit_cap=0.0,           # logit 裁剪上限（0 表示不裁剪）
    is_causal=True,          # 是否启用因果掩码
    sliding_window_size=-1,  # 滑动窗口大小（-1 表示无滑动窗口）
    sinks=None,              # Sink token logit 值（StreamingLLM）
    window_start_pos=None,   # 滑动窗口起始绝对位置 [batch]
    xai_temperature_len=-1,  # xAI 温度缩放长度阈值（-1 表示不启用）
):
    """
    Unified 1-stage extend attention for deterministic inference.

    Args:
        q: Query tensor [num_tokens, num_heads, head_dim]
        o: Output tensor [num_tokens, num_heads, head_dim]
        k_buffer: Key cache buffer
        v_buffer: Value cache buffer
        qo_indptr: Query offsets [batch_size + 1]
        kv_indptr: KV offsets [batch_size + 1] (includes both prefix and extend)
        kv_indices: Unified KV indices (both prefix and extend)
        prefix_lens: Prefix length for each sequence [batch_size]
        max_len_extend: Maximum extend length
        custom_mask: Custom attention mask (for speculative decoding tree attention)
        mask_indptr: Mask offsets [batch_size + 1]
        sm_scale: Softmax scale
        logit_cap: Logit capping value
        is_causal: Whether to apply causal mask
        sliding_window_size: Sliding window size (-1 for no sliding window)
        sinks: Sink tokens
        window_start_pos: Absolute position of first key in sliding window [batch_size]
                         (None if sliding window not used)
        xai_temperature_len: XAI temperature length
    """
    # 提取 Q 实际头维度和 V 实际头维度
    Lq, Lv = q.shape[-1], v_buffer.shape[-1]

    # Get block sizes and configuration
    # 根据头维度和硬件获取最优分块配置
    BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps = (
        _get_block_sizes_for_extend_attention(Lq, Lv)
    )

    # 若未指定 sm_scale，使用默认值 1/sqrt(Lq)
    sm_scale = sm_scale or 1.0 / (Lq**0.5)
    # 获取 batch 大小和头数
    batch_size, head_num = qo_indptr.shape[0] - 1, q.shape[1]
    # 计算 GQA 分组数
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    # 是否启用自定义掩码和 sink token
    USE_CUSTOM_MASK = custom_mask is not None
    HAS_SINK = sinks is not None

    # For sliding window attention, window_start_pos tracks the absolute position
    # of the first key in each sequence's window
    # 若未提供 window_start_pos 但启用了滑动窗口，默认所有序列窗口从位置 0 开始
    if sliding_window_size > 0 and window_start_pos is None:
        # If not provided, assume window starts at position 0
        window_start_pos = torch.zeros(batch_size, dtype=torch.int32, device=q.device)

    # 设置 Kernel grid：(batch, head, ceil(max_len_extend / BLOCK_M))
    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))
    num_stages = 1

    # AMD ROCm 专属调优参数
    extra_kargs = {}
    if _is_hip:
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}

    # 启动统一单阶段 extend 注意力 Triton Kernel
    _fwd_kernel_unified[grid](
        q,
        o,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        prefix_lens,
        custom_mask,
        mask_indptr,
        sinks,
        window_start_pos,
        sm_scale * k_scale,   # 将 k_scale 融合进 sm_scale，避免额外乘法
        v_scale,
        kv_group_num,
        q.stride(0),
        q.stride(1),
        o.stride(0),
        o.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        SLIDING_WINDOW_SIZE=sliding_window_size,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        Lq=Lq,
        Lv=Lv,
        IS_CAUSAL=is_causal,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        HAS_SINK=HAS_SINK,
        num_warps=num_warps,
        num_stages=num_stages,
        **extra_kargs,
    )
