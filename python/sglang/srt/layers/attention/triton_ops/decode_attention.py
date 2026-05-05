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
Memory-efficient attention for decoding.
It supports page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage2.py

# 导入日志模块
import logging

# 导入 Triton 及其语言模块
import triton
import triton.language as tl

# 导入硬件检测工具，用于识别 AMD ROCm 环境
from sglang.srt.utils import is_hip

# 检测是否为 HIP 环境（AMD GPU），影响 Kernel 调优参数选择
_is_hip = is_hip()

# 模块级日志对象
logger = logging.getLogger(__name__)


# KV 序列分块的最小粒度，保证按此大小对齐以提升访存效率
_MIN_BLOCK_KV = 32


# Triton JIT：双曲正切函数（利用 sigmoid 近似，兼容 GPU 精度）
@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    # tanh(x) = 2 * sigmoid(2x) - 1
    return 2 * tl.sigmoid(2 * x) - 1



# Triton JIT：标准 MHA decode 阶段 Stage1 Kernel
# 每个 (batch, head, kv_split) 三元组对应一个 Kernel 实例
# 计算当前 KV split 的部分注意力输出和 log-sum-exp
@triton.jit
def _fwd_kernel_stage1(
    Q,               # Query 张量 [batch, num_heads, head_dim]
    K_Buffer,        # KV 缓存 Key [总token数, num_kv_heads, head_dim]
    V_Buffer,        # KV 缓存 Value [总token数, num_kv_heads, head_dim]
    sm_scale_withk,  # 已融合 k_scale 的 softmax 缩放因子
    kv_indptr,       # KV token 序列的 CSR 指针数组 [batch+1]
    kv_indices,      # KV token 的物理页索引 [总token数]
    Att_Out,         # 中间注意力输出 [batch, num_heads, max_kv_splits, head_dim]
    Att_Lse,         # 中间 log-sum-exp [batch, num_heads, max_kv_splits]
    num_kv_splits,   # 每个 batch 的实际 KV 分割数 [batch]
    stride_qbs,      # Q 在 batch 维度的步长
    stride_qh,       # Q 在 head 维度的步长
    stride_buf_kbs,  # K_Buffer 在 token 维度的步长
    stride_buf_kh,   # K_Buffer 在 head 维度的步长
    stride_buf_vbs,  # V_Buffer 在 token 维度的步长
    stride_buf_vh,   # V_Buffer 在 head 维度的步长
    stride_mid_ob,   # Att_Out 在 batch 维度的步长
    stride_mid_oh,   # Att_Out 在 head 维度的步长
    stride_mid_os,   # Att_Out 在 split 维度的步长
    kv_group_num: tl.constexpr,      # GQA 分组数（Q头数 / KV头数）
    BLOCK_DMODEL: tl.constexpr,      # 头维度的填充后大小（2 的幂次）
    BLOCK_DV: tl.constexpr,          # Value 维度的填充后大小（2 的幂次）
    BLOCK_N: tl.constexpr,           # KV token 维度的分块大小
    MIN_BLOCK_KV: tl.constexpr,      # KV 分块最小粒度（对齐保证）
    logit_cap: tl.constexpr,         # logit 裁剪上限（0=不裁剪）
    Lk: tl.constexpr,                # Key 的实际头维度
    Lv: tl.constexpr,                # Value 的实际头维度
    xai_temperature_len: tl.constexpr,  # xAI 温度缩放的序列长度参数（-1=不使用）
):
    # 获取当前 Kernel 的 batch、head、KV split 索引
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    # 通过 GQA 分组计算对应的 KV 头索引
    cur_kv_head = cur_head // kv_group_num

    # 生成头维度索引范围（Key 和 Value 分别使用不同掩码）
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk    # Key 维度有效性掩码
    mask_dv = offs_dv < Lv  # Value 维度有效性掩码

    # 加载当前 batch 的 KV 起始索引和序列长度
    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    # 加载当前 batch 实际使用的 KV split 数量
    kv_splits = tl.load(num_kv_splits + cur_batch)

    # xAI 温度缩放：按 log2 位置缩放注意力分数（用于长文本推理）
    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    # 计算 Query 的内存地址偏移
    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d

    # 计算当前 split 的起止范围（按 MIN_BLOCK_KV 对齐）
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    # 初始化 Flash Attention 的 online 统计量
    e_max = -float("inf")   # 当前最大 logit
    e_sum = 0.0              # 归一化因子
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)  # 加权 Value 累积器

    # 仅在当前 split 有效时执行计算
    if split_kv_end > split_kv_start:
        # 加载 Query 向量（屏蔽超出维度的部分）
        q = tl.load(Q + off_q, mask=mask_d, other=0.0)
        # 遍历当前 split 内的 KV 分块
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            # 通过 kv_indices 获取物理 KV 缓存页地址
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )
            # 计算 Key 的内存地址偏移
            offs_buf_k = (
                kv_loc[:, None] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[None, :]
            )
            # 加载 Key 分块（按掩码过滤无效位置）
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]),
                other=0.0,
            )
            # 计算点积注意力分数（向量内积）
            qk = tl.sum(q[None, :] * k, 1)
            # 乘以缩放因子（已融合 k_scale）
            qk *= sm_scale_withk

            # 若设置了 logit_cap，使用 tanh 裁剪（防止注意力分数过大）
            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            # xAI 温度缩放（可选）
            if xai_temperature_len > 0:
                qk *= xai_temperature_reg

            # 屏蔽超出 split 范围的无效位置
            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))

            # 计算 Value 的内存地址偏移
            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            # 加载 Value 分块
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            # Flash Attention online softmax 更新
            n_e_max = tl.maximum(tl.max(qk, 0), e_max)  # 更新最大值
            re_scale = tl.exp(e_max - n_e_max)           # 旧累积的衰减因子
            p = tl.exp(qk - n_e_max)                     # 当前块注意力概率
            acc *= re_scale                               # 衰减旧累积
            # 累积加权 Value：acc += p @ v（逐元素乘法 + 求和）
            acc += tl.sum(p[:, None] * v, 0)

            # 更新归一化因子和最大值
            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        # 计算中间注意力输出的存储偏移
        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv
        )

        # 写入归一化后的注意力输出（acc / e_sum）
        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum,
            mask=(mask_dv),
        )

        # 计算存储 lse 的偏移（以 Lv 为单位整除，映射到 lse 数组）
        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        # 写入 log-sum-exp 值（e_max + log(e_sum)）
        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
        )


# Python 封装：调用 _fwd_kernel_stage1 执行 MHA decode 阶段的 Stage1 计算
def _decode_att_m_fwd(
    q,              # Query [batch, num_heads, head_dim]
    k_buffer,       # KV 缓存 Key
    v_buffer,       # KV 缓存 Value
    att_out,        # 中间注意力输出缓冲区
    att_lse,        # 中间 lse 缓冲区
    kv_indptr,      # KV 序列 CSR 指针
    kv_indices,     # KV 物理页索引
    num_kv_splits,  # 各 batch 的 KV split 数 [batch]
    max_kv_splits,  # 最大 KV split 数（决定 grid 大小）
    sm_scale_withk, # 已融合 k_scale 的 softmax 缩放因子
    logit_cap,      # logit 裁剪上限
    xai_temperature_len=-1,  # xAI 温度参数
):
    # CUDA 使用 64，HIP 使用 8（规避 MI3xx SGPR 限制）
    BLOCK = 64
    # [TODO] work around SGPR limit on MI3xx
    if _is_hip:
        BLOCK = 8
    MAX_KV_SPLITS = max_kv_splits
    Lk = k_buffer.shape[-1]  # Key 头维度
    Lv = v_buffer.shape[-1]  # Value 头维度

    batch, head_num = q.shape[0], q.shape[1]

    # grid = (batch, num_heads, max_kv_splits)
    grid = (batch, head_num, MAX_KV_SPLITS)
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    # 根据 GQA 分组数选择 warp 数量
    if kv_group_num == 1:
        num_warps = 4
    else:
        num_warps = 2
        if _is_hip:
            num_warps = 1

    # 将维度填充到 2 的幂次（Triton 向量化要求）
    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)

    # 启动 Stage1 Triton Kernel
    _fwd_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale_withk,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        num_warps=num_warps,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
    )



# Triton JIT：GQA/MQA/MLA decode 阶段 Stage1 Kernel（分组多头版本）
# 相比 _fwd_kernel_stage1，每个 Kernel 实例处理 BLOCK_H 个 Q 头（提升 GPU 并行度）
@triton.jit
def _fwd_grouped_kernel_stage1(
    Q,               # Query [batch, q_head_num, head_dim]
    K_Buffer,        # KV 缓存 Key
    V_Buffer,        # KV 缓存 Value
    sm_scale_withk,  # 已融合 k_scale 的缩放因子
    kv_indptr,       # KV 序列 CSR 指针 [batch+1]
    kv_indices,      # KV 物理页索引
    Att_Out,         # 中间输出 [batch, q_head_num, max_kv_splits, head_dim]
    Att_Lse,         # 中间 lse
    num_kv_splits,   # 各 batch 的 KV split 数 [batch]
    stride_qbs,      # Q 在 batch 维度的步长
    stride_qh,       # Q 在 head 维度的步长
    stride_buf_kbs,  # K_Buffer 在 token 维度的步长
    stride_buf_kh,   # K_Buffer 在 head 维度的步长
    stride_buf_vbs,  # V_Buffer 在 token 维度的步长
    stride_buf_vh,   # V_Buffer 在 head 维度的步长
    stride_mid_ob,   # Att_Out 在 batch 维度的步长
    stride_mid_oh,   # Att_Out 在 head 维度的步长
    stride_mid_os,   # Att_Out 在 split 维度的步长
    kv_group_num: tl.constexpr,      # GQA 分组数
    q_head_num: tl.constexpr,        # Q 总头数
    BLOCK_DMODEL: tl.constexpr,      # 主 Key 维度分块大小（2 的幂次）
    BLOCK_DPE: tl.constexpr,         # 附加位置编码维度大小（MLA 专用，0=不使用）
    BLOCK_DV: tl.constexpr,          # Value 维度分块大小（2 的幂次）
    BLOCK_N: tl.constexpr,           # KV token 分块大小
    BLOCK_H: tl.constexpr,           # 每个 Kernel 实例处理的 Q 头数
    MIN_BLOCK_KV: tl.constexpr,      # KV 分块最小粒度
    logit_cap: tl.constexpr,         # logit 裁剪上限
    xai_temperature_len: tl.constexpr,  # xAI 温度参数
    Lk: tl.constexpr,                # Key 实际头维度
    Lv: tl.constexpr,                # Value 实际头维度
):
    # 获取 batch 索引、头块索引、KV split 索引
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    # 计算当前头块对应的 KV 头索引
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    # 计算当前头块实际有效的头数（处理边界情况）
    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    # 当前块负责的 Q 头索引范围
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    # 生成有效头掩码
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    # 生成维度索引和掩码
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk     # Key 主维度掩码（用于 NOPE 或普通 Key）
    mask_dv = offs_dv < Lv   # Value 维度掩码

    # 加载 KV 序列长度信息
    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    # xAI 温度缩放参数计算
    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    # Q 主维度（NOPE 部分）的内存地址偏移
    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]

    # 若有附加位置编码维度（BLOCK_DPE > 0，即 MLA 的 RoPE 维度）
    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        # Q_PE 部分的内存地址偏移（跟在主维度之后）
        off_qpe = (
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        )

    # 计算当前 split 的起止范围
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    # 初始化 Flash Attention 多头并行统计量
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")  # 各头的最大 logit
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)                  # 各头的归一化因子
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)         # 各头的 Value 累积

    if split_kv_end > split_kv_start:
        # 加载 Q 主维度向量
        q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)
        # 若有 PE 维度，加载 Q_PE
        if BLOCK_DPE > 0:
            qpe = tl.load(
                Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0
            )
        # 遍历当前 split 内的 KV 分块
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            # 获取物理页索引
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )
            # Key 主维度（NOPE 或普通）内存偏移（转置形式用于矩阵乘法）
            offs_buf_k = (
                kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[:, None]
            )
            # 加载 Key 主维度分块
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0.0,
            )
            # 矩阵乘法计算注意力分数（主维度部分）：[BLOCK_H, Lk] x [Lk, BLOCK_N]
            qk = tl.dot(q, k.to(q.dtype))
            # 若有 PE 维度（MLA），追加 RoPE 部分的贡献
            if BLOCK_DPE > 0:
                # 加载 K_PE（RoPE 位置编码维度的 Key）
                offs_buf_kpe = (
                    kv_loc[None, :] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_dpe[:, None]
                )
                kpe = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                    other=0.0,
                )
                # 累加 PE 维度的点积：qpe @ kpe
                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            # 乘以缩放因子
            qk *= sm_scale_withk

            # logit 裁剪（可选）
            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            # xAI 温度缩放（可选）
            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            # 屏蔽无效位置（超出序列长度或非有效头）
            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            # 计算 Value 的内存地址偏移
            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            # 加载 Value 分块
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            # Flash Attention 多头 online softmax 更新
            n_e_max = tl.maximum(tl.max(qk, 1), e_max)  # 各头更新最大值
            re_scale = tl.exp(e_max - n_e_max)           # 各头的衰减因子
            p = tl.exp(qk - n_e_max[:, None])            # 当前块注意力概率
            acc *= re_scale[:, None]                      # 衰减旧累积
            # 矩阵乘法累积：acc += p @ v，[BLOCK_H, BLOCK_N] x [BLOCK_N, BLOCK_DV]
            acc += tl.dot(p.to(v.dtype), v)

            # 更新各头的归一化因子和最大值
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        # 写入归一化的中间注意力输出
        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        # 写入各头的 log-sum-exp 值（e_max + log(e_sum)）
        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )



# Python 封装：调用 _fwd_grouped_kernel_stage1 执行 GQA/MQA/MLA 的 Stage1 计算
def _decode_grouped_att_m_fwd(
    q,              # Query
    k_buffer,       # KV 缓存 Key
    v_buffer,       # KV 缓存 Value
    att_out,        # 中间注意力输出缓冲区
    att_lse,        # 中间 lse 缓冲区
    kv_indptr,      # KV 序列 CSR 指针
    kv_indices,     # KV 物理页索引
    num_kv_splits,  # 各 batch 的 KV split 数
    max_kv_splits,  # 最大 KV split 数
    sm_scale_withk, # 缩放因子
    logit_cap,      # logit 裁剪上限
    xai_temperature_len=-1,  # xAI 温度参数
):
    BLOCK = 32
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    # [TODO] work around shmem limit on MI3xx
    # 针对 AMD MI3xx 的共享内存限制调整分块大小
    if _is_hip and Lk >= 576:
        BLOCK = 16

    # MLA 特殊维度配置：Lk=576 对应 DeepSeek V3 的 512+64 分割
    if Lk == 576:
        BLOCK_DMODEL = 512  # NOPE 维度
        BLOCK_DPE = 64      # RoPE 维度
    elif Lk == 288:
        BLOCK_DMODEL = 256  # NOPE 维度
        BLOCK_DPE = 32      # RoPE 维度
    else:
        # 普通 GQA/MQA：无独立 PE 维度，整体填充到 2 的幂次
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    # 每个 Kernel 实例处理 16 个 Q 头（固定值）
    BLOCK_H = 16
    MAX_KV_SPLITS = max_kv_splits
    # grid = (batch, head_blocks, max_kv_splits)
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        MAX_KV_SPLITS,
    )

    # AMD ROCm 专属 Kernel 调优参数
    extra_kargs = {}
    num_stages = 2
    if _is_hip:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
        num_stages = 1

    # 启动分组 Stage1 Triton Kernel
    _fwd_grouped_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale_withk,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        num_warps=4,
        num_stages=num_stages,
        Lk=Lk,
        Lv=Lv,
        **extra_kargs,
    )



# Triton JIT：decode 阶段 Stage2 Kernel — 合并各 KV split 的中间结果
# 对各 split 的注意力输出和 lse 进行加权合并（online log-sum-exp 归一化）
@triton.jit
def _fwd_kernel_stage2(
    Mid_O,       # Stage1 中间注意力输出 [batch, num_heads, max_kv_splits, head_dim]
    Mid_O_1,     # Stage1 中间 lse [batch, num_heads, max_kv_splits]
    O,           # 最终输出张量 [batch, num_heads, head_dim]
    v_scale,     # Value 量化缩放因子（用于 FP8/INT8 量化反量化）
    kv_indptr,   # KV 序列 CSR 指针
    num_kv_splits,  # 各 batch 的 KV split 数 [batch]
    sink_ptr,    # 注意力 sink 的 lse 指针（用于 StreamingLLM 等方法）
    stride_mid_ob,  # Mid_O 在 batch 维度的步长
    stride_mid_oh,  # Mid_O 在 head 维度的步长
    stride_mid_os,  # Mid_O 在 split 维度的步长
    stride_obs,     # O 在 batch 维度的步长
    stride_oh,      # O 在 head 维度的步长
    MAX_KV_SPLITS: tl.constexpr,  # 最大 KV split 数（编译期常量）
    MIN_BLOCK_KV: tl.constexpr,   # KV 分块最小粒度
    BLOCK_DV: tl.constexpr,       # Value 维度分块大小（2 的幂次）
    Lv: tl.constexpr,             # Value 实际维度大小
    HAS_SINK: tl.constexpr,       # 是否包含注意力 sink 项
):
    # 每个 Kernel 实例处理一个 (batch, head) 对
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    # 获取当前 batch 的序列长度
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(
        kv_indptr + cur_batch
    )
    # 获取当前 batch 实际使用的 KV split 数
    kv_splits = tl.load(num_kv_splits + cur_batch)

    # Value 维度的偏移和掩码
    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    # 初始化合并统计量
    e_sum = 0.0                                       # 归一化因子
    e_max = -float("inf")                             # 全局最大 lse
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)     # 加权输出累积器

    # 基础内存偏移（各 split 的中间结果从这里开始偏移）
    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh) // Lv
    # 计算每个 split 的 KV 长度（按 MIN_BLOCK_KV 对齐）
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )

    # 遍历所有 split，合并各 split 的中间注意力结果
    for split_kv_id in range(0, MAX_KV_SPLITS):
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        # 跳过空 split（序列不足时部分 split 为空）
        if split_kv_end > split_kv_start:
            # 加载当前 split 的中间注意力输出向量
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            # 加载当前 split 的 lse 值
            tlogic = tl.load(Mid_O_1 + offs_logic + split_kv_id * stride_mid_os // Lv)
            # 更新全局最大 lse（online 合并算法）
            n_e_max = tl.maximum(tlogic, e_max)

            # 计算旧累积的衰减因子
            old_scale = tl.exp(e_max - n_e_max)
            # 衰减旧累积
            acc *= old_scale
            # 计算当前 split 的归一化因子
            exp_logic = tl.exp(tlogic - n_e_max)
            # 加权累积当前 split 的输出
            acc += exp_logic * tv

            # 更新归一化因子和最大值
            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    # 若包含 sink 项（StreamingLLM 等方法），将 sink 的贡献加入归一化因子
    if HAS_SINK:
        cur_sink = tl.load(sink_ptr + cur_head)
        e_sum += tl.exp(cur_sink - e_max)

    # 将最终归一化输出写入输出张量（乘以 v_scale 反量化）
    tl.store(
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum * v_scale,
        mask=mask_d,
    )



# Python 封装：Stage2 — 合并各 KV split 的中间注意力结果，得到最终输出
def _decode_softmax_reducev_fwd(
    logits,          # Stage1 中间注意力输出（归一化后）
    lse,             # Stage1 中间 lse
    q,               # Query（仅用于获取 batch/head 形状）
    o,               # 最终输出张量
    v_scale,         # Value 量化缩放因子
    v_buffer,        # KV 缓存 Value（仅用于获取维度）
    kv_indptr,       # KV 序列 CSR 指针
    num_kv_splits,   # 各 batch 的 KV split 数
    max_kv_splits,   # 最大 KV split 数
    sinks=None,      # 注意力 sink 的 lse 值（可选）
):
    batch, head_num = q.shape[0], q.shape[1]
    Lv = v_buffer.shape[-1]
    # 将 Value 维度填充到 2 的幂次
    BLOCK_DV = triton.next_power_of_2(Lv)

    MAX_KV_SPLITS = max_kv_splits
    # 判断是否包含 sink 项
    HAS_SINK = sinks is not None

    # AMD ROCm 专属调优参数
    extra_kargs = {}
    if _is_hip:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}

    # grid = (batch, num_heads)，每个 (batch, head) 对由一个 Kernel 实例合并
    grid = (batch, head_num)
    # 启动 Stage2 Triton Kernel
    _fwd_kernel_stage2[grid](
        logits,
        lse,
        o,
        v_scale,
        kv_indptr,
        num_kv_splits,
        sinks,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        MAX_KV_SPLITS=MAX_KV_SPLITS,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        HAS_SINK=HAS_SINK,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )


# 标准 MHA decode 注意力前向（Stage1 + Stage2）
def decode_attention_fwd_normal(
    q,              # Query [batch, num_heads, head_dim]
    k_buffer,       # KV 缓存 Key
    v_buffer,       # KV 缓存 Value
    o,              # 最终输出
    kv_indptr,      # KV 序列 CSR 指针
    kv_indices,     # KV 物理页索引
    attn_logits,    # 中间注意力 logits 缓冲区
    attn_lse,       # 中间 lse 缓冲区
    num_kv_splits,  # 各 batch 的 KV split 数
    max_kv_splits,  # 最大 KV split 数
    sm_scale_withk, # 已融合 k_scale 的缩放因子
    v_scale,        # Value 量化缩放因子
    logit_cap=0.0,  # logit 裁剪上限
    sinks=None,     # 注意力 sink（可选）
    xai_temperature_len=-1,  # xAI 温度参数
):
    # Stage1：分 KV split 并行计算注意力分数和部分输出
    _decode_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        attn_lse,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        max_kv_splits,
        sm_scale_withk,
        logit_cap,
        xai_temperature_len,
    )
    # Stage2：合并各 split 的中间结果，得到最终注意力输出
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        v_scale,
        v_buffer,
        kv_indptr,
        num_kv_splits,
        max_kv_splits,
        sinks,
    )


# GQA/MQA/MLA decode 注意力前向（Stage1 + Stage2，分组多头版本）
def decode_attention_fwd_grouped(
    q,
    k_buffer,
    v_buffer,
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale_withk,
    v_scale,
    logit_cap=0.0,
    sinks=None,
    xai_temperature_len=-1,
):
    # Stage1：分组多头 + KV split 并行计算
    _decode_grouped_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        attn_lse,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        max_kv_splits,
        sm_scale_withk,
        logit_cap,
        xai_temperature_len,
    )
    # Stage2：合并各 split 的中间结果
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        v_scale,
        v_buffer,
        kv_indptr,
        num_kv_splits,
        max_kv_splits,
        sinks,
    )


# 统一入口函数：根据 kv_group_num 自动选择 MHA 或 GQA/MQA/MLA 实现路径
def decode_attention_fwd(
    q,
    k_buffer,
    v_buffer,
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale,        # 原始 softmax 缩放因子（未融合 k_scale）
    k_scale,         # Key 量化缩放因子
    v_scale,         # Value 量化缩放因子
    logit_cap=0.0,
    sinks=None,
    xai_temperature_len=-1,
):
    # 断言：KV split 维度与 attn_logits 对齐
    assert max_kv_splits == attn_logits.shape[2]
    assert q.shape[0] <= kv_indptr.shape[0] - 1
    assert q.shape[0] <= attn_logits.shape[0]

    # 计算 GQA 分组数（Q头数 / V头数）
    kv_group_num = q.shape[1] // v_buffer.shape[1]

    if kv_group_num == 1:
        # MHA（多头注意力）：Q 头数 = KV 头数，使用标准实现
        decode_attention_fwd_normal(
            q,
            k_buffer,
            v_buffer,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            sm_scale * k_scale,  # 融合 k_scale 到缩放因子
            v_scale,
            logit_cap=logit_cap,
            sinks=sinks,
            xai_temperature_len=xai_temperature_len,
        )
    else:
        # GQA/MQA/MLA：多个 Q 头共享 KV，使用分组多头实现
        decode_attention_fwd_grouped(
            q,
            k_buffer,
            v_buffer,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            sm_scale * k_scale,  # 融合 k_scale 到缩放因子
            v_scale,
            logit_cap=logit_cap,
            sinks=sinks,
            xai_temperature_len=xai_temperature_len,
        )
