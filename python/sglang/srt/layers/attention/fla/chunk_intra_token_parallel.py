# Adapted from flash-linear-attention project.
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Token-parallel implementation of KDA intra chunk kernel
# Token 并行版：每个 token 独立分配一个 thread block，减少填充浪费

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.op import exp2
from sglang.srt.layers.attention.fla.utils import autotune_cache_kwargs


# 启发式函数：自动判断是否为变长序列（cu_seqlens 非 None 时）
@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
# autotune：对 BH（每 block 处理的头数）和 num_warps 进行自动调优
@triton.autotune(
    configs=[
        triton.Config({"BH": BH}, num_warps=num_warps)
        for BH in [1, 2, 4, 8]
        for num_warps in [1, 2, 4, 8]
    ],
    key=["K", "H"],
    **autotune_cache_kwargs,
)
# Triton kernel：token 并行计算 KDA chunk 内注意力矩阵 Aqk 和 Akk
# Aqk[i, j] = scale * q_i · (k_j * exp2(g_i - g_j))  (j <= i，chunk 内)
# Akk[i, j] = beta_i * k_i · (k_j * exp2(g_i - g_j))  (j < i，严格因果)
@triton.jit(do_not_specialize=["T", "N"])
def chunk_kda_fwd_kernel_intra_token_parallel(
    q,           # query 张量指针 [B, T, H, K]
    k,           # key 张量指针 [B, T, H, K]
    g,           # 门控累积对数 gk [B, T, H, K]
    beta,        # beta 门控系数 [B, T, H]
    Aqk,         # 输出：q-k 注意力矩阵 [B, T, H, BT]
    Akk,         # 输出：k-k 对角块矩阵 [B, T, H, BC]
    scale,       # 注意力缩放因子
    cu_seqlens,  # 变长序列边界（IS_VARLEN 时使用）
    N,           # batch 数量（变长时为序列数）
    T,           # 序列长度
    H: tl.constexpr,   # 注意力头数
    K: tl.constexpr,   # 特征维度
    BT: tl.constexpr,  # chunk 大小
    BC: tl.constexpr,  # 子 chunk 大小（对角块）
    BK: tl.constexpr,  # K 维度分块大小（2 的幂次）
    BH: tl.constexpr,  # 每 block 处理的头数（autotune）
    IS_VARLEN: tl.constexpr,  # 是否变长序列
):
    # 二维 grid：(全局token编号, head分块)
    i_tg, i_hg = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        # 变长模式：用二分查找定位当前 token 属于哪条序列
        i_n = 0
        left, right = 0, N

        # Unrolled binary search (max B=2^32)
        # We can limit iterations based on expected max batch size if needed
        # 20 iterations covers B=1M, usually enough
        # 展开的二分查找：20 次迭代最多支持 100 万条序列
        for _ in range(20):
            if left < right:
                mid = (left + right) // 2
                if i_tg < tl.load(cu_seqlens + mid + 1).to(tl.int32):
                    right = mid
                else:
                    left = mid + 1
        i_n = left

        # 读取序列边界，计算序列内局部 token 编号
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        i_t = i_tg - bos
    else:
        # 固定长度模式：按 batch 线性映射
        bos = (i_tg // T) * T
        i_t = i_tg % T

    # 超出序列边界的 block 直接退出
    if i_t >= T:
        return

    # 计算当前 token 所在的 chunk 编号和子 chunk 编号
    i_c = i_t // BT       # chunk 编号
    i_s = (i_t % BT) // BC  # 子 chunk 编号
    i_tc = i_c * BT        # chunk 起始 token 位置
    i_ts = i_tc + i_s * BC  # 子 chunk 起始 token 位置

    # 将各指针偏移到当前序列起始位置
    q += bos * H * K
    k += bos * H * K
    g += bos * H * K
    Aqk += bos * H * BT
    Akk += bos * H * BC
    beta += bos * H

    # 计算当前 block 处理的头编号范围和 K 维掩码
    o_h = tl.arange(0, BH)
    o_k = tl.arange(0, BK)
    m_h = (i_hg * BH + o_h) < H
    m_k = o_k < K

    # 构造当前 token 的 q、k、g、beta block pointer
    p_q = tl.make_block_ptr(
        q + i_t * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0)
    )
    p_k = tl.make_block_ptr(
        k + i_t * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0)
    )
    p_g = tl.make_block_ptr(
        g + i_t * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0)
    )
    p_beta = tl.make_block_ptr(beta + i_t * H, (H,), (1,), (i_hg * BH,), (BH,), (0,))
    # [BH, BK]
    # 加载当前 token 的 q、k、g，并将 beta 缩放到 k 上
    b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    b_k = b_k * tl.load(p_beta, boundary_check=(0,)).to(tl.float32)[:, None]

    # 遍历子 chunk 内的历史 token（j <= i_t），计算 Aqk 和 Akk
    for j in range(i_ts, min(i_t + 1, min(T, i_ts + BC))):
        p_kj = tl.make_block_ptr(
            k + j * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0)
        )
        p_gj = tl.make_block_ptr(
            g + j * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0)
        )
        # [BH, BK]
        # 加载历史 token j 的 k 和 g
        b_kj = tl.load(p_kj, boundary_check=(0, 1)).to(tl.float32)
        b_gj = tl.load(p_gj, boundary_check=(0, 1)).to(tl.float32)

        # 计算门控差分衰减：k_j * exp2(g_i - g_j)（以 2 为底的指数门控）
        b_kgj = b_kj * exp2(b_g - b_gj)

        b_kgj = tl.where(m_k[None, :], b_kgj, 0.0)
        # [BH]
        # Aqk：q_i 与门控衰减后 k_j 的内积（含自身 j==i_t）
        b_Aqk = tl.sum(b_q * b_kgj, axis=1) * scale
        # Akk：beta*k_i 与门控衰减后 k_j 的内积（严格因果，j < i_t）
        b_Akk = tl.sum(b_k * b_kgj, axis=1) * tl.where(j < i_t, 1.0, 0.0)

        # 将结果写入 Aqk 和 Akk 对应位置
        tl.store(
            Aqk + i_t * H * BT + (i_hg * BH + o_h) * BT + j % BT,
            b_Aqk.to(Aqk.dtype.element_ty),
            mask=m_h,
        )
        tl.store(
            Akk + i_t * H * BC + (i_hg * BH + o_h) * BC + j - i_ts,
            b_Akk.to(Akk.dtype.element_ty),
            mask=m_h,
        )


# Python 封装：token 并行计算 KDA chunk 内注意力矩阵
def chunk_kda_fwd_intra_token_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    sub_chunk_size: int = 16,
) -> None:
    """
    Token-parallel implementation: each token gets its own thread block.
    Supports both fixed-length and variable-length sequences.
    Reduces wasted computation on padding.

    Writes directly to Aqk and Akk tensors (in-place).

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        gk: [B, T, H, K] cumsum of gates
        beta: [B, T, H]
        Aqk: [B, T, H, BT] output tensor to write to
        Akk: [B, T, H, BC] output tensor for diagonal blocks (fp32)
        scale: attention scale
        chunk_size: BT (default 64)
        sub_chunk_size: BC (default 16)
    """
    # 解析输入张量形状
    B, T, H, K = q.shape
    # 变长序列时 N 为序列数，固定长度时 N 为 batch 大小
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    BT = chunk_size
    BC = sub_chunk_size

    # grid = (B*T, H/BH)：每个 token 独立一个 block，头维度分块
    def grid(meta):
        return (B * T, triton.cdiv(H, meta["BH"]))

    # K 向上取 2 的幂次，满足 Triton 编译时常量要求
    BK = triton.next_power_of_2(K)

    chunk_kda_fwd_kernel_intra_token_parallel[grid](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        Aqk=Aqk,
        Akk=Akk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        N=N,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
    )
    return Aqk, Akk
