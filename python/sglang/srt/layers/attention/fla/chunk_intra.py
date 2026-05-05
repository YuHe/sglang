# Adapted from flash-linear-attention project.
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# 本模块实现 KDA（Key-gated Delta Attention）的 intra-chunk 前向计算
# 主要功能：
#   1. 计算带门控衰减的 chunk 内注意力矩阵 Aqk 和状态更新矩阵 Akk
#   2. 对 Akk 对角块进行前代换，得到 (I + Akk)^{-1}
#   3. 融合跨子块（inter-subchunk）的 Akk 计算和 solve_tril，减少 kernel 启动次数

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.chunk_intra_token_parallel import (
    chunk_kda_fwd_intra_token_parallel,
)
from sglang.srt.layers.attention.fla.index import (
    prepare_chunk_indices,
)
from sglang.srt.layers.attention.fla.op import exp2, gather
from sglang.srt.layers.attention.fla.utils import (
    autotune_cache_kwargs,
    is_gather_supported,
    is_tf32_supported,
)

# SM90（Hopper）架构支持 TF32，可加速 block-merge 矩阵乘法
if is_tf32_supported:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr("tf32")
else:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr("ieee")


################################################################################
# Fused inter + solve_tril kernel: compute off-diagonal Akk and solve in one pass
# 融合跨子块 Akk 计算 + solve_tril，一次 pass 完成 inter-subchunk 注意力矩阵和逆矩阵
################################################################################


@triton.heuristics(
    {
        # 启发式检测是否为变长序列模式
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps)
        for BK in [32, 64]
        for num_warps in [1, 2, 4]
    ],
    key=["H", "K", "BC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_kda_fwd_kernel_inter_solve_fused(
    q,          # query 张量指针 [B, T, H, K]
    k,          # key 张量指针 [B, T, H, K]
    g,          # 门控累积和指针 [B, T, H, K]（log2 空间）
    beta,       # beta 缩放系数指针 [B, T, H]
    Aqk,        # 输出：q-k 注意力矩阵指针 [B, T, H, BT]
    Akkd,       # 输入：对角块 (I+Akk)^{-1} 存储（fp32，来自 token_parallel）
    Akk,        # 输出：完整 (I+Akk)^{-1} 矩阵指针 [B, T, H, BT]
    scale,      # 注意力缩放因子（1/sqrt(K) 或类似）
    cu_seqlens, # 变长序列累积长度
    chunk_indices,  # chunk 索引映射
    T,          # 序列长度
    H: tl.constexpr,    # 注意力头数
    K: tl.constexpr,    # key 特征维度
    BT: tl.constexpr,   # chunk 大小（= 4 * BC）
    BC: tl.constexpr,   # 子块大小（固定为 16）
    BK: tl.constexpr,   # key 维度分块大小（autotune）
    IS_VARLEN: tl.constexpr,      # 是否为变长序列
    USE_SAFE_GATE: tl.constexpr,  # 是否使用 safe_gate 路径（已在 token_parallel 中求逆）
):
    """
    Fused kernel: compute inter-subchunk Akk + solve_tril in one pass.
    Prerequisite: token_parallel has already computed diagonal Akk blocks in Akkd.

    This kernel:
    1. Computes off-diagonal Aqk blocks -> writes to global
    2. Computes off-diagonal Akk blocks -> keeps in registers
    3. Loads diagonal Akk blocks from Akkd (fp32)
    4. Does forward substitution on diagonals
    5. Computes merged Akk_inv
    6. Writes Akk_inv to Akk
    """
    # 获取当前 chunk 索引和 batch*head 索引
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        # 变长序列：从 chunk_indices 解析当前样本和 chunk 偏移
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # 如果当前 chunk 超出序列长度，直接跳过
    if i_t * BT >= T:
        return

    # 4 个子块的起始 token 位置（BT = 4 * BC）
    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    # 调整指针到当前序列起始位置
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    Aqk += (bos * H + i_h) * BT
    Akk += (bos * H + i_h) * BT
    Akkd += (bos * H + i_h) * BC

    o_i = tl.arange(0, BC)
    # 各子块的有效 token 掩码
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    # 初始化跨子块的 Aqk（q-k 注意力）和 Akk（k-k 自注意力）矩阵块
    # 命名约定：b_Xij 表示子块 i 对子块 j 的 [BC, BC] 矩阵块
    b_Aqk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk10 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk21 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk32 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk32 = tl.zeros([BC, BC], dtype=tl.float32)

    ################################################################################
    # off-diagonal blocks
    # 计算所有跨子块的非对角 Aqk 和 Akk 矩阵块（按 K 维度分块累积点积）
    ################################################################################
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        # 加载子块 0 的 key 和 gate
        p_k0 = tl.make_block_ptr(
            k, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
        )
        p_g0 = tl.make_block_ptr(
            g, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
        )
        b_k0 = tl.load(p_k0, boundary_check=(0, 1)).to(tl.float32)
        b_g0 = tl.load(p_g0, boundary_check=(0, 1)).to(tl.float32)

        if i_tc1 < T:
            p_q1 = tl.make_block_ptr(
                q, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            p_k1 = tl.make_block_ptr(
                k, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            p_g1 = tl.make_block_ptr(
                g, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            # [BC, BK] 加载子块 1 的 q/k/g
            b_q1 = tl.load(p_q1, boundary_check=(0, 1)).to(tl.float32)
            b_k1 = tl.load(p_k1, boundary_check=(0, 1)).to(tl.float32)
            b_g1 = tl.load(p_g1, boundary_check=(0, 1)).to(tl.float32)
            # [BK] 加载子块 1 末尾（中点位置）的 gate 值，用于数值稳定归一化
            b_gn1 = tl.load(g + i_tc1 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
            # [BC, BK] 计算归一化后的门控衰减因子（exp2(g_row - g_norm)）
            b_gqn = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)
            # [BK, BC] 子块 0 的 key 带反向门控（exp2(g_norm - g_col)）并转置
            b_kgt = tl.trans(b_k0 * exp2(b_gn1[None, :] - b_g0))
            # [BC, BC] 累积跨子块注意力 Aqk(1,0) = q1*gq @ k0*gk^T
            b_Aqk10 += tl.dot(b_q1 * b_gqn, b_kgt)
            b_Akk10 += tl.dot(b_k1 * b_gqn, b_kgt)

            if i_tc2 < T:
                p_q2 = tl.make_block_ptr(
                    q, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                )
                p_k2 = tl.make_block_ptr(
                    k, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                )
                p_g2 = tl.make_block_ptr(
                    g, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                )
                # [BC, BK] 加载子块 2 的 q/k/g
                b_q2 = tl.load(p_q2, boundary_check=(0, 1)).to(tl.float32)
                b_k2 = tl.load(p_k2, boundary_check=(0, 1)).to(tl.float32)
                b_g2 = tl.load(p_g2, boundary_check=(0, 1)).to(tl.float32)
                # [BK] 子块 2 末尾 gate 归一化基准
                b_gn2 = tl.load(g + i_tc2 * H * K + o_k, mask=m_k, other=0).to(
                    tl.float32
                )
                # [BC, BK] 子块 2 的门控因子
                b_gqn2 = tl.where(m_tc2[:, None], exp2(b_g2 - b_gn2[None, :]), 0)
                b_qg2 = b_q2 * b_gqn2
                b_kg2 = b_k2 * b_gqn2
                # [BK, BC] 子块 0 带子块 2 归一化的反向门控 key
                b_kgt = tl.trans(b_k0 * exp2(b_gn2[None, :] - b_g0))
                b_Aqk20 += tl.dot(b_qg2, b_kgt)
                b_Akk20 += tl.dot(b_kg2, b_kgt)
                # [BC, BC] 子块 1 带子块 2 归一化的反向门控 key
                b_kgt = tl.trans(b_k1 * exp2(b_gn2[None, :] - b_g1))
                # [BC, BC] 累积 Aqk(2,1) 和 Akk(2,1)
                b_Aqk21 += tl.dot(b_qg2, b_kgt)
                b_Akk21 += tl.dot(b_kg2, b_kgt)

                if i_tc3 < T:
                    p_q3 = tl.make_block_ptr(
                        q, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                    )
                    p_k3 = tl.make_block_ptr(
                        k, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                    )
                    p_g3 = tl.make_block_ptr(
                        g, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                    )
                    # [BC, BK] 加载子块 3 的 q/k/g
                    b_q3 = tl.load(p_q3, boundary_check=(0, 1)).to(tl.float32)
                    b_k3 = tl.load(p_k3, boundary_check=(0, 1)).to(tl.float32)
                    b_g3 = tl.load(p_g3, boundary_check=(0, 1)).to(tl.float32)
                    # [BK] 子块 3 末尾 gate 归一化基准
                    b_gn3 = tl.load(g + i_tc3 * H * K + o_k, mask=m_k, other=0).to(
                        tl.float32
                    )
                    # [BC, BK] 子块 3 的门控因子
                    b_gqn3 = tl.where(m_tc3[:, None], exp2(b_g3 - b_gn3[None, :]), 0)
                    b_qg3 = b_q3 * b_gqn3
                    b_kg3 = b_k3 * b_gqn3
                    # [BK, BC] 子块 3 对子块 0 的反向门控 key
                    b_kgt = tl.trans(b_k0 * exp2(b_gn3[None, :] - b_g0))
                    # [BC, BC] 累积 Aqk(3,0) 和 Akk(3,0)
                    b_Aqk30 += tl.dot(b_qg3, b_kgt)
                    b_Akk30 += tl.dot(b_kg3, b_kgt)
                    # [BK, BC] 子块 3 对子块 1 的反向门控 key
                    b_kgt = tl.trans(b_k1 * exp2(b_gn3[None, :] - b_g1))
                    # [BC, BC] 累积 Aqk(3,1) 和 Akk(3,1)
                    b_Aqk31 += tl.dot(b_qg3, b_kgt)
                    b_Akk31 += tl.dot(b_kg3, b_kgt)
                    # [BK, BC] 子块 3 对子块 2 的反向门控 key
                    b_kgt = tl.trans(b_k2 * exp2(b_gn3[None, :] - b_g2))
                    # [BC, BC] 累积 Aqk(3,2) 和 Akk(3,2)
                    b_Aqk32 += tl.dot(b_qg3, b_kgt)
                    b_Akk32 += tl.dot(b_kg3, b_kgt)

    ################################################################################
    # save off-diagonal Aqk blocks and prepare Akk
    # 将跨子块 Aqk 写回全局内存，并对 Akk 进行 beta 缩放
    ################################################################################
    if i_tc1 < T:
        p_Aqk10 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0)
        )
        # 写回 Aqk(1,0)，乘以注意力缩放因子
        tl.store(
            p_Aqk10, (b_Aqk10 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )

        # 加载子块 1 的 beta，对 Akk(1,0) 进行行缩放
        p_b1 = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_tc1,), (BC,), (0,)
        )
        b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
        b_Akk10 = b_Akk10 * b_b1[:, None]
    if i_tc2 < T:
        p_Aqk20 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BC), (1, 0)
        )
        p_Aqk21 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc2, BC), (BC, BC), (1, 0)
        )
        tl.store(
            p_Aqk20, (b_Aqk20 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )
        tl.store(
            p_Aqk21, (b_Aqk21 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )

        # 加载子块 2 的 beta，对 Akk(2,0), Akk(2,1) 进行行缩放
        p_b2 = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_tc2,), (BC,), (0,)
        )
        b_b2 = tl.load(p_b2, boundary_check=(0,)).to(tl.float32)
        b_Akk20 = b_Akk20 * b_b2[:, None]
        b_Akk21 = b_Akk21 * b_b2[:, None]
    if i_tc3 < T:
        p_Aqk30 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BC), (1, 0)
        )
        p_Aqk31 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc3, BC), (BC, BC), (1, 0)
        )
        p_Aqk32 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0)
        )
        tl.store(
            p_Aqk30, (b_Aqk30 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )
        tl.store(
            p_Aqk31, (b_Aqk31 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )
        tl.store(
            p_Aqk32, (b_Aqk32 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )

        # 加载子块 3 的 beta，对 Akk(3,0), (3,1), (3,2) 进行行缩放
        p_b3 = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_tc3,), (BC,), (0,)
        )
        b_b3 = tl.load(p_b3, boundary_check=(0,)).to(tl.float32)
        b_Akk30 = b_Akk30 * b_b3[:, None]
        b_Akk31 = b_Akk31 * b_b3[:, None]
        b_Akk32 = b_Akk32 * b_b3[:, None]

    # 从 Akkd（fp32 精度对角块）加载 4 个 16x16 对角块逆矩阵
    p_Akk00 = tl.make_block_ptr(
        Akkd, (T, BC), (H * BC, 1), (i_tc0, 0), (BC, BC), (1, 0)
    )
    p_Akk11 = tl.make_block_ptr(
        Akkd, (T, BC), (H * BC, 1), (i_tc1, 0), (BC, BC), (1, 0)
    )
    p_Akk22 = tl.make_block_ptr(
        Akkd, (T, BC), (H * BC, 1), (i_tc2, 0), (BC, BC), (1, 0)
    )
    p_Akk33 = tl.make_block_ptr(
        Akkd, (T, BC), (H * BC, 1), (i_tc3, 0), (BC, BC), (1, 0)
    )
    b_Ai00 = tl.load(p_Akk00, boundary_check=(0, 1)).to(tl.float32)
    b_Ai11 = tl.load(p_Akk11, boundary_check=(0, 1)).to(tl.float32)
    b_Ai22 = tl.load(p_Akk22, boundary_check=(0, 1)).to(tl.float32)
    b_Ai33 = tl.load(p_Akk33, boundary_check=(0, 1)).to(tl.float32)

    ################################################################################
    # forward substitution on diagonals
    # 对各对角块进行前代换，计算 (I + Akk_diag)^{-1}
    # (USE_SAFE_GATE=False 时执行；True 时对角块已在 token_parallel 中完成求逆)
    ################################################################################

    if not USE_SAFE_GATE:
        # 构造严格下三角掩码和对角线掩码
        m_A = o_i[:, None] > o_i[None, :]
        m_I = o_i[:, None] == o_i[None, :]

        # 取各对角块的严格下三角部分（取负），准备前代换
        b_Ai00 = -tl.where(m_A, b_Ai00, 0)
        b_Ai11 = -tl.where(m_A, b_Ai11, 0)
        b_Ai22 = -tl.where(m_A, b_Ai22, 0)
        b_Ai33 = -tl.where(m_A, b_Ai33, 0)

        # 子块 0 的前代换（从 Akkd 中逐行加载原始数据）
        for i in range(2, min(BC, T - i_tc0)):
            b_a00 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a00 = tl.where(o_i < i, b_a00, 0.0)
            b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
            b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
        # 子块 1 的前代换（行索引偏移 BC）
        for i in range(BC + 2, min(2 * BC, T - i_tc0)):
            b_a11 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a11 = tl.where(o_i < i - BC, b_a11, 0.0)
            b_a11 += tl.sum(b_a11[:, None] * b_Ai11, 0)
            b_Ai11 = tl.where((o_i == i - BC)[:, None], b_a11, b_Ai11)
        # 子块 2 的前代换（行索引偏移 2*BC）
        for i in range(2 * BC + 2, min(3 * BC, T - i_tc0)):
            b_a22 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a22 = tl.where(o_i < i - 2 * BC, b_a22, 0.0)
            b_a22 += tl.sum(b_a22[:, None] * b_Ai22, 0)
            b_Ai22 = tl.where((o_i == i - 2 * BC)[:, None], b_a22, b_Ai22)
        # 子块 3 的前代换（行索引偏移 3*BC）
        for i in range(3 * BC + 2, min(4 * BC, T - i_tc0)):
            b_a33 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a33 = tl.where(o_i < i - 3 * BC, b_a33, 0.0)
            b_a33 += tl.sum(b_a33[:, None] * b_Ai33, 0)
            b_Ai33 = tl.where((o_i == i - 3 * BC)[:, None], b_a33, b_Ai33)

        # 前代换完成后加回单位矩阵（对角线置 1）
        b_Ai00 += m_I
        b_Ai11 += m_I
        b_Ai22 += m_I
        b_Ai33 += m_I

    ################################################################################
    # compute merged inverse using off-diagonals
    # 利用分块矩阵求逆公式，将 4 个对角块逆矩阵合并为完整的 (I+Akk)^{-1}
    ################################################################################

    # we used tf32 to maintain matrix inverse's precision whenever possible.
    # 一阶跨块逆：Ai_ij = -Ai_ii @ Akk_ij @ Ai_jj（相邻块）
    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_Akk10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai00,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22, b_Akk21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai11,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33, b_Akk32, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai22,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )

    # 二阶跨块逆：需利用一阶结果（间隔一个子块）
    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_Akk20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk31, b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk32, b_Ai21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    # 三阶跨块逆：需利用一阶和二阶结果（间隔两个子块）
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )

    ################################################################################
    # store full Akk_inv to Akk
    # 将完整的 (I+Akk)^{-1} 所有 10 个子块写回输出张量 Akk
    ################################################################################

    p_Akk00 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akk10 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_Akk11 = tl.make_block_ptr(
        Akk, (T, BT), (H * BT, 1), (i_tc1, BC), (BC, BC), (1, 0)
    )
    p_Akk20 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_Akk21 = tl.make_block_ptr(
        Akk, (T, BT), (H * BT, 1), (i_tc2, BC), (BC, BC), (1, 0)
    )
    p_Akk22 = tl.make_block_ptr(
        Akk, (T, BT), (H * BT, 1), (i_tc2, 2 * BC), (BC, BC), (1, 0)
    )
    p_Akk30 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
    p_Akk31 = tl.make_block_ptr(
        Akk, (T, BT), (H * BT, 1), (i_tc3, BC), (BC, BC), (1, 0)
    )
    p_Akk32 = tl.make_block_ptr(
        Akk, (T, BT), (H * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0)
    )
    p_Akk33 = tl.make_block_ptr(
        Akk, (T, BT), (H * BT, 1), (i_tc3, 3 * BC), (BC, BC), (1, 0)
    )

    tl.store(p_Akk00, b_Ai00.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk10, b_Ai10.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk11, b_Ai11.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk20, b_Ai20.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk21, b_Ai21.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk22, b_Ai22.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk30, b_Ai30.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk31, b_Ai31.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk32, b_Ai32.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk33, b_Ai33.to(Akk.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        # 启发式检测变长序列模式
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BT", "BC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_kda_fwd_kernel_intra_sub_chunk(
    q,          # query 张量指针
    k,          # key 张量指针
    g,          # gate 累积和指针（log2 空间，向量化）
    beta,       # beta 缩放系数指针
    Aqk,        # 输出：q-k 注意力矩阵（diagonal + off-diagonal）[B, T, H, BT]
    Akk,        # 输出：k-k 自注意力矩阵（对角块，fp32）[B, T, H, BC]
    scale,      # 注意力缩放因子
    cu_seqlens, # 变长序列累积长度
    chunk_indices,  # chunk 索引映射
    T,          # 序列长度
    H: tl.constexpr,    # 注意力头数
    K: tl.constexpr,    # key 特征维度
    BT: tl.constexpr,   # chunk 大小
    BC: tl.constexpr,   # 子块大小（固定 16）
    BK: tl.constexpr,   # key 分块大小
    IS_VARLEN: tl.constexpr,   # 是否变长序列
    USE_GATHER: tl.constexpr,  # 是否使用 tl.gather 取 gate 中间值（Triton 版本相关）
):
    # 三维 grid：(chunk 数, 子块内序号, batch*H)
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # 当前子块的起始 token 全局位置
    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    # 当前子块的 token 范围和有效掩码
    o_c = i_ti + tl.arange(0, BC)
    m_c = o_c < T

    # 调整各指针到当前序列起始
    q = q + (bos * H + i_h) * K
    k = k + (bos * H + i_h) * K
    g = g + (bos * H + i_h) * K
    beta = beta + bos * H + i_h
    Aqk = Aqk + (bos * H + i_h) * BT
    Akk = Akk + (bos * H + i_h) * BC

    # 加载当前子块的 q/k/g/beta 数据
    p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))

    p_beta = tl.make_block_ptr(beta, (T,), (H,), (i_ti,), (BC,), (0,))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_beta = tl.load(p_beta, boundary_check=(0,))

    if USE_GATHER:
        # 使用 tl.gather 获取子块中间 token 的 gate 值（数值稳定性基准）
        b_gn = gather(
            b_g, tl.full([1, BK], min(BC // 2, T - i_ti - 1), dtype=tl.int16), axis=0
        )
    else:
        # calculate offset 计算中间 token 的 gate 指针偏移
        p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * H * K + tl.arange(0, BK)
        b_gn = tl.load(p_gn, mask=tl.arange(0, BK) < K, other=0.0)
        b_gn = b_gn[None, :]

    # current block, keep numerical stability by subtracting the left boundary
    # less than 85 to avoid overflow in exp2
    # 以子块中间 gate 为基准做归一化，避免 exp2 溢出（上界约 85）
    b_gm = (b_g - b_gn).to(tl.float32)

    # 计算前向和反向门控因子
    b_gq = tl.where(m_c[:, None], exp2(b_gm), 0.0)
    b_gk = tl.where(m_c[:, None], exp2(-b_gm), 0.0)

    # k 带反向门控后转置，用于矩阵乘法
    b_kgt = tl.trans(b_k * b_gk)

    # 计算 Aqk 和 Akk：加注意力缩放/beta 缩放
    b_Aqk = tl.dot(b_q * b_gq, b_kgt) * scale
    b_Akk = tl.dot(b_k * b_gq, b_kgt) * b_beta[:, None]

    o_i = tl.arange(0, BC)
    # Aqk 保留下三角（含对角），Akk 只保留严格下三角
    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_Aqk = tl.where(m_Aqk, b_Aqk, 0.0)
    b_Akk = tl.where(m_Akk, b_Akk, 0.0)

    p_Aqk = tl.make_block_ptr(
        Aqk, (T, BT), (H * BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0)
    )
    p_Akk = tl.make_block_ptr(Akk, (T, BC), (H * BC, 1), (i_ti, 0), (BC, BC), (1, 0))
    # 写回 Aqk 对角块（含下三角）和 Akk 对角块（严格下三角）
    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk, b_Akk.to(Akk.dtype.element_ty), boundary_check=(0, 1))

    # 等待写入完成，确保后续前代换读取的是已更新的数据
    tl.debug_barrier()

    ################################################################################
    # forward substitution
    # 对当前对角子块进行前代换，计算 (I + Akk_diag)^{-1}
    ################################################################################

    b_Ai = -b_Akk
    for i in range(2, min(BC, T - i_ti)):
        b_a = -tl.load(Akk + (i_ti + i) * H * BC + o_i)
        b_a = tl.where(o_i < i, b_a, 0.0)
        b_a += tl.sum(b_a[:, None] * b_Ai, 0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
    # 加回单位矩阵，完成对角块逆矩阵计算
    b_Ai += m_I
    tl.store(p_Akk, b_Ai.to(Akk.dtype.element_ty), boundary_check=(0, 1))


# Python 封装：KDA intra-chunk 前向计算的统一入口
# 两步骤：1) 计算对角块 Akkd + Aqk（token_parallel/safe_gate 路径）
#         2) 融合跨子块 Akk + solve_tril，并调用 recompute_w_u_fwd
def chunk_kda_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor | None = None,   # 向量化 gate（KDA 特有）
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    safe_gate: bool = False,           # True：使用 safe_gate 路径（数值更稳定）
    disable_recompute: bool = False,   # True：禁用 q/k 的重计算（直接传入 q）
):
    B, T, H, K = k.shape
    BT = chunk_size
    BC = 16  # 子块固定大小
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)  # 每个 chunk 内的子块数（= BT / BC）

    # 初始化输出矩阵（零初始化保证未写入区域为 0）
    Aqk = torch.zeros(B, T, H, BT, device=k.device, dtype=k.dtype)
    # Akk must be zero-initialized - kernel only writes lower triangular
    # Akk 的上三角部分 kernel 不会写入，必须初始化为 0
    Akk = torch.zeros(B, T, H, BT, device=k.device, dtype=k.dtype)
    # Separate fp32 buffer for diagonal 16x16 blocks (for precision in solve_tril)
    # 单独分配 fp32 精度的对角块缓冲区，提高前代换的数值精度
    Akkd = torch.zeros(B, T, H, BC, device=k.device, dtype=torch.float32)

    # Step 1: Run token_parallel first to compute diagonal blocks into Akkd (fp32)
    # Step 1: compute diagonal blocks into Akk_diag (fp32)
    # 步骤1：计算各对角子块，写入 Akkd
    if safe_gate:
        # safe_gate 路径：使用更稳定的 intra_sub_chunk kernel
        grid = (NT, NC, B * H)
        BK = triton.next_power_of_2(K)
        chunk_kda_fwd_kernel_intra_sub_chunk[grid](
            q=q,
            k=k,
            g=gk,
            beta=beta,
            Aqk=Aqk,
            Akk=Akkd,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            K=K,
            BT=BT,
            BC=BC,
            BK=BK,
            USE_GATHER=is_gather_supported,
        )
    else:
        # 默认路径：使用 token_parallel kernel（更高并行度）
        Aqk, Akkd = chunk_kda_fwd_intra_token_parallel(
            q=q,
            k=k,
            gk=gk,
            beta=beta,
            Aqk=Aqk,
            Akk=Akkd,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=BT,
            sub_chunk_size=BC,
        )

    # Step 2: Fused inter + solve_tril (works for both fixed-len and varlen)
    # 步骤2：融合跨子块 Akk 计算 + solve_tril，将完整逆矩阵写入 Akk
    grid = (NT, B * H)
    chunk_kda_fwd_kernel_inter_solve_fused[grid](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        Aqk=Aqk,
        Akkd=Akkd,
        Akk=Akk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        USE_SAFE_GATE=safe_gate,
    )
    # 延迟导入避免循环依赖
    from sglang.srt.layers.attention.fla.kda import (
        recompute_w_u_fwd as kda_recompute_w_u_fwd,
    )

    # 步骤3：使用已求逆的 Akk 重计算 w（写权重）、u（更新值）、以及门控化 q/k
    w, u, qg, kg = kda_recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=Akk,
        q=q if disable_recompute else None,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return w, u, qg, kg, Aqk, Akk
