# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

# 从蚂蚁 PIA 项目 seg_la 实现复制而来
# Copied from https://code.alipay.com/pia/PainlessInferenceAcceleration/blob/v0.0.6/flood/flood/ops/seg_la.py

# 导入 dataclass（用于 SegLaMeta 元数据结构）
from dataclasses import dataclass
# 导入 Optional 类型注解
from typing import Optional

# 导入 PyTorch
import torch
# 导入 Triton JIT 编译器
import triton
# 导入 Triton 语言原语（tl.load/tl.store/tl.dot 等）
import triton.language as tl


# seg_la_fwd 的元数据参数结构（用于描述当前批次的分段线性注意力信息）
@dataclass
class SegLaMeta:
    batch_size: int  # batch size，当前批次的请求数量
    max_q_length: int  # max(seq_lens)，最长 Query 序列长度
    q_offsets: torch.Tensor  # [bs+1]，Query 累积起始位置（CSR 格式，query_start_locations）
    s_offsets: torch.Tensor  # [bs]，每个请求对应的 SSM 状态槽位索引（slot_ids）
    q_lengths: torch.Tensor  # [bs]，每个请求的 Query 长度
    s_scales: torch.Tensor  # [bs]，状态缩放标志：prefill=0（无历史状态），decode=1（有历史状态）
    s_offsets_stride: int = 0   # s_offsets 张量步幅（保留字段）
    q_offsets_stride: int = 0   # q_offsets 张量步幅（保留字段）
    s_scales_stride: int = 0    # s_scales 张量步幅（保留字段）
    decay_scales_stride: int = 0  # decay_scales 张量步幅（保留字段）
    mask: Optional[torch.Tensor] = None  # Currently not supported（投机解码掩码，暂不支持通用场景）


# 融合版 SegLA Kernel：支持 prefill（BLOCK>1）和 decode（BLOCK=1）两种模式
# 同时处理有/无历史状态（通过 s_scale 控制是否加载 SSM 状态）
@triton.jit
def seg_la_kernel(
    Q,              # Query 张量 [total_len, qo_heads, HEAD_DIM]
    K,              # Key 张量 [total_len, kv_heads, HEAD_DIM]
    V,              # Value 张量 [total_len, kv_heads, HEAD_DIM]
    S,              # SSM 状态池 [num_slots, kv_heads, HEAD_DIM, HEAD_DIM]
    Out,            # 输出张量 [total_len, qo_heads, HEAD_DIM]
    softmax_scale,  # 注意力缩放因子（通常 = HEAD_DIM^{-0.5}）
    stride_q,       # Q 序列维度步幅（token 步幅）
    stride_k,       # K 序列维度步幅
    stride_v,       # V 序列维度步幅
    stride_s,       # S 槽位维度步幅
    stride_o,       # Out 序列维度步幅
    s_offsets,      # [bs]，每个请求的 SSM 状态槽位索引
    q_offsets,      # [bs+1]，每个请求的 Query 累积起始位置
    q_lengths,      # [bs]，每个请求的 Query 长度
    s_scales,       # [bs]，是否使用历史状态（prefill=0, decode=1）
    decay_scales,   # [h]，每头的 ALiBi 衰减斜率
    HEAD_DIM: tl.constexpr,   # 头维度
    SPLIT_DIM: tl.constexpr,  # V 维度分块大小（按分块并行）
    BLOCK: tl.constexpr,      # 时序分块大小（decode=1, prefill>=1）
    EVEN: tl.constexpr,       # 序列长度是否恰好是 BLOCK 的整数倍
    DECOUPLE: tl.constexpr,   # 是否使用解耦模式（decouple 优化路径）
):
    bid = tl.program_id(0)  # 请求索引（批次维度）
    hid = tl.program_id(1)  # 注意力头索引
    sid = tl.program_id(2)  # V 维度分块索引

    # 加载当前请求的元信息：状态缩放标志、Query 长度、Query 起始位置、状态槽位
    s_scale = tl.load(s_scales + bid)     # prefill=0（无历史），decode=1（有历史）
    q_length = tl.load(q_lengths + bid)   # 当前请求的 Query token 数
    q_offset = tl.load(q_offsets + bid)   # 当前请求的 Query 全局起始位置
    s_offset = tl.load(s_offsets + bid)   # 当前请求的 SSM 状态槽位索引
    decay_scale = -tl.load(decay_scales + hid)  # ALiBi 衰减斜率（取负使衰减因子 < 1）

    # 初始化偏移数组：时序块偏移、头维度偏移、V 维度块偏移
    offs_b = tl.arange(0, BLOCK)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_s = tl.arange(0, SPLIT_DIM)

    # slot_id == -1 表示 padding 请求，直接跳过
    if s_offset == -1:
        return

    # 构建 Q、K 指针（BLOCK × HEAD_DIM 分块访问）
    q_ptrs = (
        Q
        + q_offset * stride_q        # 当前请求的起始 token
        + hid * HEAD_DIM             # 当前头的偏移
        + (offs_b[:, None] * stride_q + offs_d[None, :])  # [BLOCK, HEAD_DIM] 块指针
    )
    k_ptrs = (
        K
        + q_offset * stride_k
        + hid * HEAD_DIM
        + (offs_b[:, None] * stride_k + offs_d[None, :])
    )
    # V 指针：按 SPLIT_DIM 分块访问（sid 选择当前 V 维块）
    v_ptrs = (
        V
        + q_offset * stride_v
        + hid * HEAD_DIM
        + sid * SPLIT_DIM              # V 维度块起始偏移
        + (offs_b[:, None] * stride_v + offs_s[None, :])  # [BLOCK, SPLIT_DIM] 块指针
    )
    out_ptrs = (
        Out
        + q_offset * stride_o
        + hid * HEAD_DIM
        + sid * SPLIT_DIM
        + (offs_b[:, None] * stride_o + offs_s[None, :])
    )
    # 构建 SSM 状态指针（[HEAD_DIM, SPLIT_DIM] 块，按 KV 分块访问）
    s_ptrs = (
        S
        + s_offset * stride_s           # 当前请求的状态槽位偏移
        + hid * HEAD_DIM * HEAD_DIM     # 当前头的状态矩阵偏移 [HEAD_DIM, HEAD_DIM]
        + sid * SPLIT_DIM               # V 维度分块偏移
        + (offs_d[:, None] * HEAD_DIM + offs_s[None, :])  # [HEAD_DIM, SPLIT_DIM] 块指针
    )
    # 加载历史 SSM 状态（s_scale=0 时不加载，即新 prefill 的初始状态为 0）
    state = tl.load(s_ptrs, mask=s_scale > 0).to(tl.float32)

    if BLOCK > 1:
        # prefill 路径：按 BLOCK 分块处理序列
        for n in range(0, q_length, BLOCK):
            n = tl.multiple_of(n, BLOCK)

            if EVEN:
                # 序列长度为 BLOCK 整数倍，无需边界 mask
                q = tl.load(q_ptrs + n * stride_q).to(tl.float32)
                k = tl.trans(tl.load(k_ptrs + n * stride_k)).to(tl.float32)  # K 转置 [HEAD_DIM, BLOCK]
                v = tl.load(v_ptrs + n * stride_k).to(tl.float32)
            else:
                # 非整数倍，使用 mask 避免越界
                q = tl.load(
                    q_ptrs + n * stride_q,
                    mask=(n + offs_b)[:, None] < q_length,
                    other=0.0,
                ).to(tl.float32)
                k = tl.trans(
                    tl.load(
                        k_ptrs + n * stride_k,
                        mask=(n + offs_b)[:, None] < q_length,
                        other=0.0,
                    )
                ).to(tl.float32)
                v = tl.load(
                    v_ptrs + n * stride_k,
                    mask=(n + offs_b)[:, None] < q_length,
                    other=0.0,
                ).to(tl.float32)

            if DECOUPLE:
                # 解耦模式（decouple=True）：将 Q/K 乘以块内位置的倒数/正向衰减因子
                # 使得块内注意力等价于全局注意力（解耦 Q/K 的位置相关衰减）
                if EVEN:
                    b = BLOCK
                else:
                    b = min(BLOCK, q_length - n)  # 实际有效块长度
                b_offs = b - 1 - offs_b  # 块内反向偏移（最后位置偏移为 0）

                edb = tl.exp(decay_scale * b_offs)       # 正向衰减
                decays = tl.where(b_offs >= 0, edb, 0)   # 有效位置的衰减因子
                inv_decays = tl.where(b_offs >= 0, 1 / edb, 0)  # 倒数衰减

                q = q * inv_decays[:, None]  # Q 乘以倒数衰减（反向归一化）
                k = k * decays[None, :]      # K 乘以正向衰减（对应位置权重）
                qk = tl.dot(q, k) * softmax_scale  # 块内注意力分数
                # 因果掩码：仅保留下三角部分（当前和历史位置）
                qk = tl.where(offs_b[None, :] <= offs_b[:, None], qk, 0.0)
                o = tl.dot(qk, v)  # 块内注意力输出 [BLOCK, SPLIT_DIM]

                block_decay = tl.exp(decay_scale * b)      # 整块的衰减因子
                block_decay_plus = block_decay * softmax_scale
                # 加上历史状态的贡献：o += q @ state * block_decay
                o = tl.dot(q, state) * block_decay_plus + o

                # 更新 SSM 状态：state = state * block_decay + k @ v
                state = state * block_decay + tl.dot(k, v)
            else:
                # 非解耦模式（标准线性注意力）
                qk = tl.dot(q, k) * softmax_scale
                # 位置差衰减矩阵：decay[i, j] = exp(decay_scale * (i - j))
                decays = tl.exp(decay_scale * (offs_b[:, None] - offs_b[None, :]))
                decays = tl.where(offs_b[None, :] <= offs_b[:, None], decays, 0.0)
                qk *= decays  # 带位置衰减的注意力分数
                o = tl.dot(qk, v)

                # 历史状态贡献：o += (q * decay_arr) @ state
                decay_arr = tl.exp(decay_scale * (offs_b[:, None] + 1)) * softmax_scale
                o = tl.dot(q * decay_arr, state, acc=o)

                if EVEN:
                    b = BLOCK
                else:
                    b = min(BLOCK, q_length - n)
                b_offs = b - 1 - offs_b
                b_offs = tl.where(b_offs >= 0, b_offs, 10000)  # 无效位置使用大值（避免 exp 溢出）
                decays = tl.exp(decay_scale * b_offs)
                block_decay = tl.exp(decay_scale * b)
                # 更新 SSM 状态：state = state * block_decay + (k * decays) @ v
                state = state * block_decay + tl.dot(k * decays[None, :], v)

            # 写回当前块的输出
            if EVEN:
                tl.store(out_ptrs + n * stride_o, o.to(Out.dtype.element_ty))
            else:
                tl.store(
                    out_ptrs + n * stride_o,
                    o.to(Out.dtype.element_ty),
                    mask=(n + offs_b)[:, None] < q_length,
                )

        # 处理完所有 token 后写回更新后的 SSM 状态（供后续 decode 使用）
        tl.store(s_ptrs, state.to(S.dtype.element_ty))

    else:
        # decode 路径（BLOCK=1，单 token 更新）
        q = tl.trans(tl.load(q_ptrs)).to(tl.float32) * softmax_scale  # 1D query 转置
        k = tl.trans(tl.load(k_ptrs)).to(tl.float32)  # 1D key 转置
        v = tl.load(v_ptrs).to(tl.float32)
        # 更新 SSM 状态：S_t = decay * S_{t-1} + k * v
        state = state * tl.exp(decay_scale) + k * v

        # 计算输出：o = sum_k(q_k * S_t[k, :])
        o = tl.sum(q * state, axis=0, keep_dims=True)

        tl.store(out_ptrs, o.to(Out.dtype.element_ty))

        # 写回更新后的 SSM 状态
        tl.store(s_ptrs, state.to(S.dtype.element_ty))


# prefill 专用 Kernel：K/V 维度分块并行（K_SPLIT_DIM × V_SPLIT_DIM 分块）
# 比 seg_la_kernel 更高效（充分利用 SRAM 和并行度）
@triton.jit
def seg_la_p_kernel(
    Q,              # Query 张量 [total_len, qo_heads, HEAD_DIM]
    K,              # Key 张量 [total_len, kv_heads, HEAD_DIM]
    V,              # Value 张量 [total_len, kv_heads, HEAD_DIM]
    S,              # SSM 状态池 [num_slots, kv_heads, HEAD_DIM, HEAD_DIM]
    Out,            # 临时输出 [k_dim_block, total_len, qo_heads, HEAD_DIM]（KV 分块结果，后需 reduce）
    softmax_scale,  # 注意力缩放因子
    stride_q,       # Q token 步幅
    stride_k,       # K token 步幅
    stride_v,       # V token 步幅
    stride_s,       # S 槽位步幅
    stride_o,       # Out K 维块步幅
    s_offsets,      # [bs] SSM 状态槽位索引
    q_offsets,      # [bs+1] Query 累积起始位置
    q_lengths,      # [bs] Query 长度
    s_scales,       # [bs] 是否使用历史状态（0=新 prefill，1=有历史）
    decay_scales,   # [h] 每头 ALiBi 衰减斜率
    HEAD_DIM: tl.constexpr,     # 头维度
    K_SPLIT_DIM: tl.constexpr,  # K 维度分块大小（通常 32）
    V_SPLIT_DIM: tl.constexpr,  # V 维度分块大小（通常 32 或 64）
    BLOCK: tl.constexpr,        # 时序分块大小（通常 32）
    EVEN: tl.constexpr,         # 序列是否为 BLOCK 整数倍
):
    bid = tl.program_id(0)  # 请求索引
    hid = tl.program_id(1)  # 头索引
    kvid = tl.program_id(2)  # KV 分块联合索引（kid * (HEAD_DIM/V_SPLIT_DIM) + vid）
    N = HEAD_DIM // V_SPLIT_DIM  # V 维度分块数
    kid = kvid // N  # K 维度分块索引
    vid = kvid % N   # V 维度分块索引
    H = tl.num_programs(1)  # 总头数

    # 加载当前请求的元信息
    s_scale = tl.load(s_scales + bid)   # 是否使用历史状态
    q_length = tl.load(q_lengths + bid) # Query 长度
    q_offset = tl.load(q_offsets + bid) # Query 全局起始位置
    s_offset = tl.load(s_offsets + bid) # SSM 状态槽位
    decay_scale = -tl.load(decay_scales + hid)  # 衰减斜率

    # 初始化偏移数组
    offs_b = tl.arange(0, BLOCK)
    offs_k = tl.arange(0, K_SPLIT_DIM)
    offs_v = tl.arange(0, V_SPLIT_DIM)

    # slot_id == -1 表示 padding，跳过
    if s_offset == -1:
        return

    # 构建 Q、K 的 K 维分块指针（[BLOCK, K_SPLIT_DIM] 子块）
    q_ptrs = (
        Q
        + q_offset * stride_q
        + hid * HEAD_DIM
        + kid * K_SPLIT_DIM          # K 维度分块起始偏移
        + (offs_b[:, None] * stride_q + offs_k[None, :])
    )
    k_ptrs = (
        K
        + q_offset * stride_k
        + hid * HEAD_DIM
        + kid * K_SPLIT_DIM
        + (offs_b[:, None] * stride_k + offs_k[None, :])
    )
    # V 的 V 维分块指针（[BLOCK, V_SPLIT_DIM] 子块）
    v_ptrs = (
        V
        + q_offset * stride_v
        + hid * HEAD_DIM
        + vid * V_SPLIT_DIM
        + (offs_b[:, None] * stride_v + offs_v[None, :])
    )
    # Out 的布局：(num_k_dim_block, length, qo_heads, HEAD_DIM)
    # kid 为 K 维块索引，存入 Out 的第 kid 个 k_dim_block
    out_ptrs = (
        Out
        + kid * stride_o                # K 维块偏移（stride_o = total_len * qo_heads * HEAD_DIM）
        + q_offset * HEAD_DIM * H       # 当前请求的起始 token 偏移
        + hid * HEAD_DIM                # 头偏移
        + vid * V_SPLIT_DIM             # V 维块起始偏移
        + (offs_b[:, None] * H * HEAD_DIM + offs_v[None, :])  # [BLOCK, V_SPLIT_DIM] 访问
    )
    # SSM 状态指针（[K_SPLIT_DIM, V_SPLIT_DIM] 子块）
    s_ptrs = (
        S
        + s_offset * stride_s
        + hid * HEAD_DIM * HEAD_DIM
        + kid * HEAD_DIM * K_SPLIT_DIM  # K 维块偏移
        + vid * V_SPLIT_DIM             # V 维块起始偏移
        + (offs_k[:, None] * HEAD_DIM + offs_v[None, :])  # [K_SPLIT_DIM, V_SPLIT_DIM] 子块
    )
    # 加载历史 SSM 状态（s_scale=0 时初始状态为 0，即新 prefill 序列）
    state = tl.load(s_ptrs, mask=s_scale > 0).to(tl.float32)

    # 按 BLOCK 步长遍历当前请求的所有 token
    for n in range(0, q_length, BLOCK):
        n = tl.multiple_of(n, BLOCK)

        if EVEN:
            # 无边界 mask：整块加载
            q = tl.load(q_ptrs + n * stride_q).to(tl.float32)
            k = tl.trans(tl.load(k_ptrs + n * stride_k)).to(tl.float32)  # K 转置 [K_SPLIT_DIM, BLOCK]
            v = tl.load(v_ptrs + n * stride_v).to(tl.float32)
            b = BLOCK
            b_offs = b - 1 - offs_b  # 块内反向偏移
            decays = tl.exp(decay_scale * b_offs)     # 位置衰减因子
            inv_decays = 1 / decays                    # 倒数衰减
        else:
            # 有边界 mask：处理最后一块不足 BLOCK 的情况
            q = tl.load(
                q_ptrs + n * stride_q, mask=(n + offs_b)[:, None] < q_length, other=0.0
            ).to(tl.float32)
            k = tl.trans(
                tl.load(
                    k_ptrs + n * stride_k,
                    mask=(n + offs_b)[:, None] < q_length,
                    other=0.0,
                )
            ).to(tl.float32)
            v = tl.load(
                v_ptrs + n * stride_v, mask=(n + offs_b)[:, None] < q_length, other=0.0
            ).to(tl.float32)
            b = min(BLOCK, q_length - n)    # 实际有效块长度
            b_offs = b - 1 - offs_b          # 块内反向偏移
            block_decays = tl.exp(decay_scale * b_offs)
            decays = tl.where(b_offs >= 0, block_decays, 0)   # 有效位置的衰减
            inv_decays = tl.where(b_offs >= 0, 1 / block_decays, 0)  # 倒数衰减

        # 解耦线性注意力（decouple prefill 路径）：
        # Q/K 乘以位置衰减因子实现等效全局位置感知
        q = q * inv_decays[:, None]  # [BLOCK, K_SPLIT_DIM]
        k = k * decays[None, :]      # [K_SPLIT_DIM, BLOCK]
        qk = tl.dot(q, k) * softmax_scale  # 块内注意力分数 [BLOCK, BLOCK]
        # 因果掩码：仅保留下三角（当前和历史位置）
        qk = tl.where(offs_b[None, :] <= offs_b[:, None], qk, 0.0)
        o = tl.dot(qk, v)  # 块内注意力输出 [BLOCK, V_SPLIT_DIM]

        block_decay = tl.exp(decay_scale * b)   # 整块的衰减因子
        # 加入历史状态贡献：o += q @ state * block_decay * scale
        o = tl.dot(q, state) * block_decay * softmax_scale + o

        # 更新 SSM 状态：state = state * block_decay + k @ v
        state = state * block_decay + tl.dot(k, v)

        # 写回当前块的输出
        if EVEN:
            tl.store(out_ptrs + n * H * HEAD_DIM, o.to(Out.dtype.element_ty))
        else:
            tl.store(
                out_ptrs + n * H * HEAD_DIM,
                o.to(Out.dtype.element_ty),
                mask=(n + offs_b)[:, None] < q_length,
            )

    # 写回更新后的 SSM 状态（供后续 decode 使用）
    tl.store(s_ptrs, state.to(S.dtype.element_ty))


# 投机解码（speculative decoding）专用 Kernel：使用自定义 mask 矩阵处理树注意力
# 每个请求使用一个 [q_length, q_length] 的 attention mask（支持非顺序依赖）
@triton.jit
def seg_la_s_kernel(
    Q,              # Query 张量
    K,              # Key 张量
    V,              # Value 张量
    S,              # SSM 状态池
    Out,            # 临时输出（K 维分块格式）
    Mask,           # 注意力掩码 [bs, q_length, q_length]（投机解码的树注意力 mask）
    softmax_scale,  # 缩放因子
    stride_q,       # Q token 步幅
    stride_k,       # K token 步幅
    stride_v,       # V token 步幅
    stride_s,       # S 槽位步幅
    stride_o,       # Out K 维块步幅
    s_offsets,      # [bs] SSM 状态槽位索引
    q_offsets,      # [bs+1] Query 累积起始位置
    q_lengths,      # [bs] Query 长度
    s_scales,       # [bs] 是否使用历史状态
    decay_scales,   # [h] 每头衰减斜率
    HEAD_DIM: tl.constexpr,     # 头维度
    K_SPLIT_DIM: tl.constexpr,  # K 维度分块大小
    V_SPLIT_DIM: tl.constexpr,  # V 维度分块大小
    BLOCK: tl.constexpr,        # 时序块大小（投机序列长度对齐）
    EVEN: tl.constexpr,         # 是否整块对齐
):
    bid = tl.program_id(0)  # 请求索引
    hid = tl.program_id(1)  # 头索引
    kvid = tl.program_id(2)  # KV 分块联合索引
    N = HEAD_DIM // V_SPLIT_DIM
    kid = kvid // N   # K 维块索引
    vid = kvid % N    # V 维块索引
    H = tl.num_programs(1)  # 总头数

    # 加载当前请求的元信息（与 seg_la_p_kernel 相同）
    s_scale = tl.load(s_scales + bid)
    q_length = tl.load(q_lengths + bid)
    q_offset = tl.load(q_offsets + bid)
    s_offset = tl.load(s_offsets + bid)
    decay_scale = -tl.load(decay_scales + hid)

    offs_b = tl.arange(0, BLOCK)
    offs_k = tl.arange(0, K_SPLIT_DIM)
    offs_v = tl.arange(0, V_SPLIT_DIM)

    # slot_id == -1 表示 padding，跳过
    if s_offset == -1:
        return

    # 构建 Q、K、V 指针（与 seg_la_p_kernel 相同结构）
    q_ptrs = (
        Q
        + q_offset * stride_q
        + hid * HEAD_DIM
        + kid * K_SPLIT_DIM
        + (offs_b[:, None] * stride_q + offs_k[None, :])
    )
    k_ptrs = (
        K
        + q_offset * stride_k
        + hid * HEAD_DIM
        + kid * K_SPLIT_DIM
        + (offs_b[:, None] * stride_k + offs_k[None, :])
    )
    v_ptrs = (
        V
        + q_offset * stride_v
        + hid * HEAD_DIM
        + vid * V_SPLIT_DIM
        + (offs_b[:, None] * stride_v + offs_v[None, :])
    )
    # (num_dim_block, length, qo_heads, d)
    out_ptrs = (
        Out
        + kid * stride_o
        + q_offset * HEAD_DIM * H
        + hid * HEAD_DIM
        + vid * V_SPLIT_DIM
        + (offs_b[:, None] * H * HEAD_DIM + offs_v[None, :])
    )
    s_ptrs = (
        S
        + s_offset * stride_s
        + hid * HEAD_DIM * HEAD_DIM
        + kid * HEAD_DIM * K_SPLIT_DIM
        + vid * V_SPLIT_DIM
        + (offs_k[:, None] * HEAD_DIM + offs_v[None, :])
    )
    # 加载历史 SSM 状态（s_scale=0 时初始化为 0）
    state = tl.load(s_ptrs, mask=s_scale > 0).to(tl.float32)

    if EVEN:
        # 整块加载 Q、K、V（投机序列完整）
        q = tl.load(q_ptrs).to(tl.float32)
        k = tl.trans(tl.load(k_ptrs)).to(tl.float32)  # K 转置 [K_SPLIT_DIM, BLOCK]
        v = tl.load(v_ptrs).to(tl.float32)
        # 加载投机解码注意力 mask [BLOCK, BLOCK]（树注意力的依赖关系）
        mask = tl.load(
            Mask
            + bid * BLOCK * BLOCK
            + tl.arange(0, BLOCK)[:, None] * BLOCK
            + tl.arange(0, BLOCK)[None, :]
        ).to(tl.int32)
        # 通过 mask 计算每个 token 的位置（sum=token在树中的深度+1）
        positions = tl.sum(mask, 1) - 1  # 每行的 mask 和减 1 = 位置索引
        max_pos = tl.max(positions)       # 最大位置（用于统一衰减基准）
        b_offs = max_pos - positions      # 相对位置偏移（越大表示越靠前）
    else:
        # 有边界 mask：处理不足 BLOCK 的情况
        q = tl.load(q_ptrs, mask=offs_b[:, None] < q_length).to(tl.float32)
        k = tl.trans(tl.load(k_ptrs, mask=offs_b[:, None] < q_length)).to(tl.float32)
        v = tl.load(v_ptrs, mask=offs_b[:, None] < q_length).to(tl.float32)
        mask = tl.load(
            Mask
            + bid * q_length * q_length
            + tl.arange(0, BLOCK)[:, None] * q_length
            + tl.arange(0, BLOCK)[None, :],
            mask=(tl.arange(0, BLOCK)[:, None] < q_length)
            & (tl.arange(0, BLOCK)[None, :] < q_length),
        ).to(tl.int32)
        positions = tl.sum(mask, 1) - 1
        max_pos = tl.max(positions)
        b_offs = max_pos - positions

    # 计算位置衰减因子和倒数
    decays = tl.exp(decay_scale * b_offs)
    inv_decays = 1 / decays

    # 对 Q/K 施加位置衰减（解耦注意力）
    q = q * inv_decays[:, None]  # [BLOCK, K_SPLIT_DIM]
    k = k * decays[None, :]      # [K_SPLIT_DIM, BLOCK]
    qk = tl.dot(q, k) * softmax_scale  # 注意力分数 [BLOCK, BLOCK]
    # 使用 mask 替代因果掩码（投机解码的树形依赖关系）
    qk = qk * mask.to(tl.float32)
    o = tl.dot(qk, v)  # 注意力输出 [BLOCK, V_SPLIT_DIM]

    # 加入历史状态贡献：o += q @ state * (max_pos+1 的整体衰减)
    block_decay = tl.exp(decay_scale * (max_pos + 1))
    o = tl.dot(q, state) * block_decay * softmax_scale + o

    # 写回输出（不更新 SSM 状态，投机解码验证时禁止状态更新）
    if EVEN:
        tl.store(out_ptrs, o.to(Out.dtype.element_ty))
    else:
        tl.store(out_ptrs, o.to(Out.dtype.element_ty), mask=offs_b[:, None] < q_length)


# decode 专用 Kernel：单 token 解码，K/V 维度分块并行
# 每个 program 处理一个（batch, head, KV块）的组合
@triton.jit
def seg_la_d_kernel(
    Q,              # Query 张量 [bs, kv_heads, HEAD_DIM]（每请求 1 个 token）
    K,              # Key 张量 [bs, kv_heads, HEAD_DIM]
    V,              # Value 张量 [bs, kv_heads, HEAD_DIM]
    S,              # SSM 状态池 [num_slots, kv_heads, HEAD_DIM, HEAD_DIM]
    Out,            # 临时输出 [k_dim_block, bs, kv_heads, HEAD_DIM]
    softmax_scale,  # 缩放因子
    stride_q,       # Q batch 步幅（每 token 的步长）
    stride_k,       # K batch 步幅
    stride_v,       # V batch 步幅
    stride_s,       # S 槽位步幅
    stride_o,       # Out K 维块步幅
    s_offsets,      # [bs] SSM 状态槽位索引
    decay_scales,   # [h] 每头衰减斜率
    HEAD_DIM: tl.constexpr,     # 头维度
    K_SPLIT_DIM: tl.constexpr,  # K 维度分块大小
    V_SPLIT_DIM: tl.constexpr,  # V 维度分块大小
):
    bid = tl.program_id(0)  # 请求索引（batch 维）
    hid = tl.program_id(1)  # 头索引
    kvid = tl.program_id(2)  # KV 分块联合索引
    N = HEAD_DIM // V_SPLIT_DIM
    kid = kvid // N  # K 维块索引
    vid = kvid % N   # V 维块索引
    H = tl.num_programs(1)  # 总头数

    # 加载当前请求的 SSM 状态槽位
    s_offset = tl.load(s_offsets + bid)
    # slot_id == -1 表示 padding，跳过
    if s_offset == -1:
        return

    # 加载衰减斜率（取负）
    decay_scale = -tl.load(decay_scales + hid)

    # 初始化 K/V 维度偏移数组
    offs_k = tl.arange(0, K_SPLIT_DIM)
    offs_v = tl.arange(0, V_SPLIT_DIM)

    # Q/K/V 指针（单 token decode，无 BLOCK 维度）
    q_ptrs = Q + bid * stride_q + hid * HEAD_DIM + kid * K_SPLIT_DIM + (offs_k)
    k_ptrs = K + bid * stride_k + hid * HEAD_DIM + kid * K_SPLIT_DIM + (offs_k)
    v_ptrs = V + bid * stride_v + hid * HEAD_DIM + vid * V_SPLIT_DIM + (offs_v)
    # Out 布局：(num_k_dim_block, length, qo_heads, HEAD_DIM)
    out_ptrs = (
        Out
        + kid * stride_o         # K 维块偏移
        + bid * H * HEAD_DIM     # batch 偏移（每 token H * HEAD_DIM 个元素）
        + hid * HEAD_DIM         # 头偏移
        + vid * V_SPLIT_DIM      # V 维块起始偏移
        + (offs_v)               # 当前 V 维块内的索引
    )
    # SSM 状态指针（[K_SPLIT_DIM, V_SPLIT_DIM] 子块）
    s_ptrs = (
        S
        + s_offset * stride_s
        + hid * HEAD_DIM * HEAD_DIM
        + kid * HEAD_DIM * K_SPLIT_DIM
        + vid * V_SPLIT_DIM
        + (offs_k[:, None] * HEAD_DIM + offs_v[None, :])
    )
    # 加载历史 SSM 状态（decode 阶段始终有历史）
    state = tl.load(s_ptrs).to(tl.float32)

    # 加载单 token 的 K/K/Q 向量
    k = tl.load(k_ptrs).to(tl.float32)   # [K_SPLIT_DIM]
    v = tl.load(v_ptrs).to(tl.float32)   # [V_SPLIT_DIM]
    q = tl.load(q_ptrs).to(tl.float32) * softmax_scale  # [K_SPLIT_DIM]（缩放）

    # 更新 SSM 状态：S_t = decay * S_{t-1} + k_t * v_t^T（外积更新）
    state = state * tl.exp(decay_scale) + k[:, None] * v
    # 计算线性注意力输出：o = sum_k(q_k * S_t[k, :])
    o = tl.sum(q[:, None] * state, axis=0)

    # 写回输出和更新后的 SSM 状态
    tl.store(out_ptrs, o.to(Out.dtype.element_ty))
    tl.store(s_ptrs, state.to(S.dtype.element_ty))


# MTP（Multi-Token Prediction，多 token 预测/验证）专用 Kernel
# 用于投机解码的目标模型验证：并行处理每个请求的多个草稿 token
# 每 step 保存中间 SSM 状态到 CACHES（供并行验证使用）
@triton.jit
def seg_la_mtp_kernel(
    Q,              # Query 张量 [bs * step, kv_heads, HEAD_DIM]
    K,              # Key 张量 [bs * step, kv_heads, HEAD_DIM]
    V,              # Value 张量 [bs * step, kv_heads, HEAD_DIM]
    S,              # SSM 状态池（输入初始状态）
    CACHES,         # 中间 SSM 状态缓冲区 [bs, step, kv_heads, HEAD_DIM, HEAD_DIM]
    Out,            # 临时输出 [k_dim_block, bs * step, kv_heads, HEAD_DIM]
    softmax_scale,  # 缩放因子
    stride_q,       # Q token 步幅
    stride_k,       # K token 步幅
    stride_v,       # V token 步幅
    stride_s,       # S 槽位步幅
    stride_c,       # CACHES 步幅（每个 step 的状态大小）
    stride_o,       # Out K 维块步幅
    s_offsets,      # [bs] SSM 状态槽位索引
    cache_indices,  # [bs] 每个请求在 CACHES 中的起始索引
    decay_scales,   # [h] 每头衰减斜率
    step,           # 草稿 token 数（每请求的验证步数）
    HEAD_DIM: tl.constexpr,     # 头维度
    K_SPLIT_DIM: tl.constexpr,  # K 维度分块大小
    V_SPLIT_DIM: tl.constexpr,  # V 维度分块大小
):
    bid = tl.program_id(0)  # 请求索引
    hid = tl.program_id(1)  # 头索引
    kvid = tl.program_id(2)  # KV 分块联合索引
    N = HEAD_DIM // V_SPLIT_DIM
    kid = kvid // N   # K 维块索引
    vid = kvid % N    # V 维块索引
    H = tl.num_programs(1)  # 总头数

    # 加载 SSM 状态槽位和 cache 索引
    s_offset = tl.load(s_offsets + bid)
    if s_offset == -1:
        return  # padding 请求，跳过

    # 衰减因子（取 exp 处理）
    decay_scale = tl.exp(-tl.load(decay_scales + hid))

    offs_k = tl.arange(0, K_SPLIT_DIM)
    offs_v = tl.arange(0, V_SPLIT_DIM)

    # Q/K/V 指针：每次步长为 stride_q（每请求 step 个 token，bid * step 为起始）
    q_ptrs = Q + bid * step * stride_q + hid * HEAD_DIM + kid * K_SPLIT_DIM + (offs_k)
    k_ptrs = K + bid * step * stride_k + hid * HEAD_DIM + kid * K_SPLIT_DIM + (offs_k)
    v_ptrs = V + bid * step * stride_v + hid * HEAD_DIM + vid * V_SPLIT_DIM + (offs_v)
    # Out 布局：(num_k_dim_block, length, qo_heads, HEAD_DIM)
    out_ptrs = (
        Out
        + kid * stride_o
        + bid * step * H * HEAD_DIM  # 请求起始偏移（step 个 token 的间距）
        + hid * HEAD_DIM
        + vid * V_SPLIT_DIM
        + (offs_v)
    )
    # SSM 状态指针（初始状态从 S 池中读取）
    s_ptrs = (
        S
        + s_offset * stride_s
        + hid * HEAD_DIM * HEAD_DIM
        + kid * HEAD_DIM * K_SPLIT_DIM
        + vid * V_SPLIT_DIM
        + (offs_k[:, None] * HEAD_DIM + offs_v[None, :])
    )
    # 加载初始 SSM 状态（验证开始时的状态）
    state = tl.load(s_ptrs).to(tl.float32)
    # 加载当前请求在 CACHES 中的存储起始索引
    cache_indices = tl.load(cache_indices + bid)
    c_ptrs = (
        CACHES
        + cache_indices * stride_c   # 当前请求的 cache 起始偏移
        + hid * HEAD_DIM * HEAD_DIM
        + kid * HEAD_DIM * K_SPLIT_DIM
        + vid * V_SPLIT_DIM
        + (offs_k[:, None] * HEAD_DIM + offs_v[None, :])
    )

    # 遍历每个草稿 step：逐步更新 SSM 状态并保存中间状态
    for i in range(step):
        q = tl.load(q_ptrs).to(tl.float32) * softmax_scale
        k = tl.load(k_ptrs).to(tl.float32)
        v = tl.load(v_ptrs).to(tl.float32)

        # 更新 SSM 状态：S_t = decay * S_{t-1} + k_t * v_t^T
        state = state * decay_scale + k[:, None] * v
        # 计算当前 step 的输出：o = q^T @ S_t
        o = tl.sum(q[:, None] * state, axis=0)

        # 写回输出和中间 SSM 状态（保存每步状态用于并行验证）
        tl.store(out_ptrs, o.to(Out.dtype.element_ty))
        tl.store(c_ptrs, state.to(CACHES.dtype.element_ty))
        # 步进指针到下一个 token
        q_ptrs += stride_q
        k_ptrs += stride_k
        v_ptrs += stride_v
        out_ptrs += H * HEAD_DIM            # Out 移动一个 token 的距离
        c_ptrs += H * HEAD_DIM * HEAD_DIM   # CACHES 移动一个状态矩阵的距离


# K 维度 reduce Kernel：将 k_dim_block 个临时输出累加到最终输出
# (k_dim_block, length, qo_heads, HEAD_DIM) -> (length, qo_heads, HEAD_DIM)
@triton.jit
def seg_la_sum_kernel(T, O, DIM: tl.constexpr, NUM_BLOCK: tl.constexpr):
    pid = tl.program_id(0)   # 全局 token 索引（每个 program 处理一个 token）
    length = tl.num_programs(0)  # 总 token 数
    x = tl.zeros((DIM,), dtype=tl.float32)  # 累加器（DIM = qo_heads * HEAD_DIM）
    # 遍历 k_dim_block 个分块，累加每个分块对该 token 的贡献
    for i in range(NUM_BLOCK):
        x += tl.load(T + i * length * DIM + pid * DIM + tl.arange(0, DIM)).to(
            tl.float32
        )
    # 写回累加结果
    tl.store(O + pid * DIM + tl.arange(0, DIM), x)


def seg_la_fwd(
    q,               # Query 张量 [total_len, qo_heads, HEAD_DIM]
    k,               # Key 张量 [total_len, kv_heads, HEAD_DIM]
    v,               # Value 张量 [total_len, kv_heads, HEAD_DIM]
    s,               # SSM 状态池 [num_slots, kv_heads, HEAD_DIM, HEAD_DIM]
    decay_scales,    # 衰减斜率 [kv_heads]（每头的 ALiBi 斜率）
    meta,            # SegLaMeta 元数据（包含 batch/offset/lengths/scales 信息）
    caches=None,     # 可选：MTP 中间状态缓冲区（投机解码验证时传入）
    cache_indices=None,  # 可选：MTP 中间状态的 cache 起始索引 [bs]
    softmax_scale=None,  # 可选：注意力缩放因子（默认 HEAD_DIM^{-0.5}）
    decouple=False,      # 是否使用解耦模式（对 Q/K 施加反向位置衰减）
):
    # 分段线性注意力前向计算的 Python 入口函数
    length, qo_heads, HEAD_DIM = q.shape  # 总 token 数、头数、头维度
    _, kv_heads, _ = k.shape
    bs = meta.batch_size
    if softmax_scale is None:
        softmax_scale = HEAD_DIM ** (-0.5)  # 默认 scale = 1 / sqrt(d)

    # MAX_LENGTH = meta.max_q_length
    MAX_LENGTH = triton.cdiv(length, bs)  # 估计最大序列长度（用于判断 prefill/decode 路径）

    assert qo_heads == kv_heads, "seg_la does NOT support GQA currently"  # 目前不支持分组查询注意力

    if MAX_LENGTH > 1:
        # prefill 路径：K/V 维度分块并行（充分利用并行度）
        # BLOCK should <= 64 with decouple
        K_SPLIT_DIM = 32                           # K 维度分块大小
        V_SPLIT_DIM = 32 if bs <= 2 else 64        # V 维度分块大小（大 batch 用 64 提高并行度）

        num_warps = 2   # Triton warp 数
        num_stages = 3  # Triton pipeline stage 数

        k_dim_block = HEAD_DIM // K_SPLIT_DIM   # K 方向分块数
        v_dim_block = HEAD_DIM // V_SPLIT_DIM   # V 方向分块数
        # 临时缓冲区：存储各 K 维块的注意力输出（后续 reduce 求和）
        tmp = torch.empty(
            (k_dim_block, length, qo_heads, HEAD_DIM), device=q.device, dtype=q.dtype
        )
        grid = (bs, kv_heads, k_dim_block * v_dim_block)  # 计算网格（bs × heads × KV分块数）

        if caches is not None:
            # MTP 路径（投机解码目标验证）：使用 seg_la_mtp_kernel
            EVEN = False
            BLOCK = 32
            step = length // bs  # 每请求的草稿 token 数

            seg_la_mtp_kernel[grid](
                q,
                k,
                v,
                s,
                caches,
                tmp,
                softmax_scale,
                q.stride(0),
                k.stride(0),
                v.stride(0),
                s.stride(0),
                caches.stride(0),
                tmp.stride(0),
                meta.s_offsets,
                cache_indices,
                decay_scales,
                step,
                HEAD_DIM=HEAD_DIM,
                K_SPLIT_DIM=K_SPLIT_DIM,
                V_SPLIT_DIM=V_SPLIT_DIM,
                num_warps=num_warps,
                num_stages=num_stages,
            )

        elif meta.mask is not None:
            # 投机解码（树注意力）路径：使用 seg_la_s_kernel（自定义 mask）
            ms = meta.mask.size(-1)
            BLOCK = (ms + 15) // 16 * 16  # 对齐到 16 的倍数（优化 Triton 访问）
            EVEN = BLOCK == ms            # 是否完美对齐

            seg_la_s_kernel[grid](
                q,
                k,
                v,
                s,
                tmp,
                meta.mask,
                softmax_scale,
                q.stride(0),
                k.stride(0),
                v.stride(0),
                s.stride(0),
                tmp.stride(0),
                meta.s_offsets,
                meta.q_offsets,
                meta.q_lengths,
                meta.s_scales,
                decay_scales,
                HEAD_DIM=HEAD_DIM,
                K_SPLIT_DIM=K_SPLIT_DIM,
                V_SPLIT_DIM=V_SPLIT_DIM,
                BLOCK=BLOCK,
                EVEN=EVEN,
                num_warps=num_warps,
                num_stages=num_stages,
            )

        else:
            # 标准 prefill 路径：使用 seg_la_p_kernel
            BLOCK = 32
            EVEN = MAX_LENGTH % BLOCK == 0 if bs == 1 else False  # 单请求才尝试对齐优化

            seg_la_p_kernel[grid](
                q,
                k,
                v,
                s,
                tmp,
                softmax_scale,
                q.stride(0),
                k.stride(0),
                v.stride(0),
                s.stride(0),
                tmp.stride(0),
                meta.s_offsets,
                meta.q_offsets,
                meta.q_lengths,
                meta.s_scales,
                decay_scales,
                HEAD_DIM=HEAD_DIM,
                K_SPLIT_DIM=K_SPLIT_DIM,
                V_SPLIT_DIM=V_SPLIT_DIM,
                BLOCK=BLOCK,
                EVEN=EVEN,
                num_warps=num_warps,
                num_stages=num_stages,
            )

        # 将 k_dim_block 个分块的结果 reduce 求和得到最终输出
        if k_dim_block > 1:
            if length < 2048:
                # 短序列：直接调用 PyTorch sum（开销小）
                o = tmp.sum(0)
            else:
                # 长序列：使用 Triton seg_la_sum_kernel 并行 reduce（避免大 tensor copy）
                o = torch.empty(
                    (length, qo_heads, HEAD_DIM), device=q.device, dtype=q.dtype
                )
                seg_la_sum_kernel[(length,)](
                    tmp,
                    o,
                    DIM=qo_heads * HEAD_DIM,   # 每 token 的输出维度
                    NUM_BLOCK=k_dim_block,      # K 维分块数（reduce 的累加轮数）
                    num_warps=2,
                    num_stages=3,
                )
        else:
            # 只有一个 K 维块，直接取第 0 个块
            o = tmp[0]

    else:
        # decode 路径（MAX_LENGTH == 1，每请求仅 1 个 token）
        if bs <= 128:
            K_SPLIT_DIM = 128   # 小 batch 使用更大 K 分块（更高并行度）
            V_SPLIT_DIM = 32
            num_warps = 2
            num_stages = 2
        else:
            K_SPLIT_DIM = 128   # 大 batch 使用更大 V 分块（更高并行度）
            V_SPLIT_DIM = 64
            num_warps = 2
            num_stages = 3
        k_dim_block = HEAD_DIM // K_SPLIT_DIM
        v_dim_block = HEAD_DIM // V_SPLIT_DIM
        tmp = torch.empty(
            (k_dim_block, length, qo_heads, HEAD_DIM), device=q.device, dtype=q.dtype
        )
        grid = (bs, kv_heads, k_dim_block * v_dim_block)

        # 调用 decode 专用 Kernel
        seg_la_d_kernel[grid](
            q,
            k,
            v,
            s,
            tmp,
            softmax_scale,
            q.stride(0),
            k.stride(0),
            v.stride(0),
            s.stride(0),
            tmp.stride(0),
            meta.s_offsets,
            decay_scales,
            HEAD_DIM=HEAD_DIM,
            K_SPLIT_DIM=K_SPLIT_DIM,
            V_SPLIT_DIM=V_SPLIT_DIM,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        # Reduce K 维分块
        if k_dim_block > 1:
            o = tmp.sum(0)
        else:
            o = tmp[0]

    # if fallback:
    #     # prefill/decode with partitioning v only
    #     o = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    #     if MAX_LENGTH == 1:
    #         # decode
    #         BLOCK = 1
    #         EVEN = False
    #         SPLIT_DIM = 32
    #         num_warps = 8
    #         num_stages = 2
    #         num_dim_block = HEAD_DIM // SPLIT_DIM
    #         grid = (batch, kv_heads, num_dim_block)
    #     else:
    #         # prefill
    #         if decouple:
    #             BLOCK = 64
    #             SPLIT_DIM = 16
    #         else:
    #             BLOCK = HEAD_DIM
    #             SPLIT_DIM = 32
    #         # EVEN = all([x % BLOCK == 0 for x in meta.qls])
    #         EVEN = False
    #         num_warps = 8
    #         num_stages = 2
    #         # prop = torch.cuda.get_device_properties(q.device.index)
    #         # arch = prop.major * 10 + prop.minor
    #         # if arch not in (80, 90):
    #         #     num_stages = 1

    #         num_dim_block = HEAD_DIM // SPLIT_DIM
    #         grid = (batch, kv_heads, num_dim_block)

    #     seg_la_kernel[grid](
    #         q,
    #         k,
    #         v,
    #         s,
    #         o,
    #         softmax_scale,
    #         q.stride(0),
    #         k.stride(0),
    #         v.stride(0),
    #         s.stride(0),
    #         o.stride(0),
    #         meta.s_offsets,
    #         meta.q_offsets,
    #         meta.q_lengths,
    #         meta.s_scales,
    #         decay_scales,
    #         HEAD_DIM=HEAD_DIM,
    #         SPLIT_DIM=SPLIT_DIM,
    #         BLOCK=BLOCK,
    #         EVEN=EVEN,
    #         DECOUPLE=decouple,
    #         num_warps=num_warps,
    #         num_stages=num_stages
    #     )
    return o
