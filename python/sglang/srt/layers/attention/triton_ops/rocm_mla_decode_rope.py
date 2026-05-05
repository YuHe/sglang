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

# 导入 Triton 和 Triton 语言模块
import triton
import triton.language as tl

# 从 decode_attention 模块导入第二阶段 softmax+reducev 函数
from sglang.srt.layers.attention.triton_ops.decode_attention import (
    _decode_softmax_reducev_fwd,
)


# 检测当前是否运行在 HIP（ROCm/AMD GPU）后端
def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


# 全局变量：标记是否为 HIP 环境，用于调整 Kernel 超参数
_is_hip = is_hip()


# Triton JIT：实现高效的双曲正切函数（通过 sigmoid 近似）
@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    # tanh(x) = 2 * sigmoid(2x) - 1，利用 sigmoid 计算避免原生 tanh 精度问题
    return 2 * tl.sigmoid(2 * x) - 1


# Triton JIT：MLA（Multi-head Latent Attention）解码阶段 Stage1 Kernel
# 融合了 RoPE（旋转位置编码）的 KV 注意力分数计算（分 KV split 并行）
@triton.jit
def _fwd_grouped_kernel_stage1_rope(
    Q,              # Query 张量 [batch, q_head_num, kv_lora_rank + qk_rope_head_dim]，包含 NOPE 和 PE 两部分
    K_Buffer,       # KV 缓存的 Key 部分 [总token数, kv_lora_rank + qk_rope_head_dim]
    V_buffer,       # KV 缓存的 Value 部分 [总token数, kv_lora_rank]
    cos_sin_cache,  # RoPE 旋转编码缓存 [max_seq_len, rotary_dim * 2]
    positions,      # 每个 batch 对应的序列位置 [batch]
    sm_scale,       # softmax 缩放因子
    kv_indptr,      # KV token 序列的指针数组（CSR 格式）[batch+1]
    kv_indices,     # KV token 的物理页索引 [总token数]
    Att_Out,        # 中间注意力输出 [batch, q_head_num, NUM_KV_SPLITS, kv_lora_rank+1]
    k_pe_t_out,     # 输出经过 RoPE 后的最后一个 token 的 k_pe
    stride_qb,      # Q 在 batch 维度的步长
    stride_qh,      # Q 在 head 维度的步长
    stride_buf_kbs, # K_Buffer 在 token 维度的步长
    stride_buf_vbs, # V_buffer 在 token 维度的步长
    stride_mid_ob,  # Att_Out 在 batch 维度的步长
    stride_mid_oh,  # Att_Out 在 head 维度的步长
    stride_mid_os,  # Att_Out 在 split 维度的步长
    stride_kpe_tokens_out_b,  # k_pe_t_out 在 batch 维度的步长
    stride_cos_sin_cache_s,   # cos_sin_cache 在序列位置维度的步长
    stride_positions_b,       # positions 在 batch 维度的步长
    rotary_dim: tl.constexpr,         # RoPE 旋转维度大小
    kv_lora_rank: tl.constexpr,       # KV LoRA 压缩维度（NOPE 部分维度大小）
    qk_rope_head_dim: tl.constexpr,   # RoPE 位置编码维度大小
    kv_group_num: tl.constexpr,       # GQA 分组数（每个 KV 头对应的 Q 头数）
    q_head_num: tl.constexpr,         # Q 的总头数
    BLOCK_C: tl.constexpr,            # NOPE 维度的分块大小（填充至 2 的幂次）
    BLOCK_R: tl.constexpr,            # RoPE 维度的分块大小
    BLOCK_N: tl.constexpr,            # KV token 维度的分块大小
    BLOCK_H: tl.constexpr,            # 头维度的分块大小（多头并行）
    NUM_KV_SPLITS: tl.constexpr,      # KV 序列的分割数（并行分段计算注意力）
    logit_cap: tl.constexpr,          # logit 裁剪值（防止注意力分数过大，tanh 裁剪）
    USE_ROPE: tl.constexpr,           # 是否融合 RoPE 计算
    IS_NEOX_STYLE: tl.constexpr,      # 是否使用 NeoX 风格的 RoPE（GPT-NeoX 格式）
):
    # 获取当前 Kernel 的 batch 索引（grid 第 0 维）
    cur_batch = tl.program_id(0)
    # 获取当前 Kernel 的头块索引（grid 第 1 维，每块处理 BLOCK_H 个头）
    cur_head_id = tl.program_id(1)
    # 获取当前 KV 分割块索引（grid 第 2 维）
    split_kv_id = tl.program_id(2)

    # 确定当前头块实际处理的有效头数（不超过 kv_group_num）
    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    # 计算当前块负责的头索引范围
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    # 生成有效头的掩码（过滤超出范围的头）
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    # NOPE 部分（压缩 KV）的维度偏移
    offs_c = tl.arange(0, BLOCK_C)
    # RoPE 部分（位置编码）的维度偏移（从 kv_lora_rank 偏移处开始）
    offs_qk_r = tl.arange(kv_lora_rank, kv_lora_rank + BLOCK_R)  # to get the k_pe

    # Q 的 RoPE 部分内存偏移（Q_PE 部分）
    off_q_pe = (
        cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_qk_r[None, :]
    )
    # Q 的 NOPE 部分内存偏移
    offs_q = cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_c[None, :]

    # NOPE 维度有效性掩码
    mask_c = offs_c < kv_lora_rank
    # RoPE 维度有效性掩码
    mask_qk_r = offs_qk_r < (kv_lora_rank + qk_rope_head_dim)

    # 加载当前 batch 的 KV 起始索引和序列长度
    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx

    # 加载 Q 的 NOPE 部分（压缩 KV 对应的查询向量）
    q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_c[None, :]), other=0.0)
    # 加载 Q 的 PE 部分（待施加 RoPE 的查询向量）
    q_pe = tl.load(
        Q + off_q_pe, mask=(mask_h[:, None]) & (mask_qk_r[None, :]), other=0.0
    )

    # 计算当前 KV split 的起止范围（均匀分割序列）
    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    # apply rotary embedding for q_pe, and k_pe (last token per batch of K_PE)
    # 标记当前 split 是否为最后一个分割块（需要对最后一个 token 的 k_pe 施加 RoPE）
    LAST_SPLIT = split_kv_end == cur_batch_seq_len
    # 初始化最后一个 token 的 k_pe（仅在 LAST_SPLIT 时填充）
    k_pe_last_token = tl.zeros([BLOCK_R], dtype=q.dtype)

    if USE_ROPE:
        if IS_NEOX_STYLE:
            # NeoX 风格 RoPE：将维度分为前后两半，前半旋转到后半位置
            # [BLOCK_ROTARY // 2, BLOCK_ROTARY // 2 + 1, BLOCK_ROTARY // 2 + 2, ..., 0, 1, 2, ..., BLOCK_ROTARY // 2 - 1, pass:]
            offs_qk_rot_r = kv_lora_rank + (
                (tl.arange(0, BLOCK_R) + (rotary_dim // 2)) % rotary_dim
            )
            # Which elements to flip
            # 前半维度需要取反（实现旋转）
            mask_rotate = tl.arange(0, BLOCK_R) < (rotary_dim // 2)
            # [0 , 1, 2, ..., rotary_dim // 2 - 1, 0 , 1, 2, ..., rotary_dim // 2 - 1]
            # 每个维度对应的旋转频率索引
            offs_rotary = tl.arange(0, BLOCK_R) % (rotary_dim // 2)
        else:
            # GPT-J 风格 RoPE：奇偶维度配对旋转（[1,0,3,2,5,4,...] 索引置换）
            # [1, 0, 3, 2, 5, 4, ..., BLOCK_R, BLOCK_R - 1]
            offs_qk_rot_r = (
                kv_lora_rank
                + (((tl.arange(0, BLOCK_R) + 1) % 2) * 2)
                - 1
                + tl.arange(0, BLOCK_R)
            )
            # 奇数索引位置需要取反
            mask_rotate = tl.arange(0, BLOCK_R) % 2 < 1
            # [0, 0, 1, 1, ..., rotary_dim // 2 - 1, rotary_dim // 2 - 1]
            # 每对维度共享同一个旋转频率
            offs_rotary = tl.arange(0, BLOCK_R) // 2

        # 若 qk_rope_head_dim 超过 rotary_dim，超出部分不做旋转（直接保留原索引）
        if qk_rope_head_dim > rotary_dim:
            offs_qk_rot_r = tl.where(
                tl.arange(0, BLOCK_R) < rotary_dim, offs_qk_rot_r, tl.arange(0, BLOCK_R)
            )
            offs_rotary = tl.where(
                tl.arange(0, BLOCK_R) < rotary_dim, offs_rotary, tl.arange(0, BLOCK_R)
            )

        # 只对实际旋转维度范围内的元素应用 cos/sin
        mask_rotary = tl.arange(0, BLOCK_R) < rotary_dim

        # 加载当前 batch 的序列位置（用于查询 cos/sin 缓存）
        pos = tl.load(positions + cur_batch * stride_positions_b)
        # 从 cos_sin_cache 中加载对应位置的余弦值
        cos = tl.load(
            cos_sin_cache + pos * stride_cos_sin_cache_s + offs_rotary,
            mask=mask_rotary,
            other=1.0,
        )
        # 从 cos_sin_cache 中加载对应位置的正弦值（偏移 rotary_dim//2 处）
        sin = tl.load(
            cos_sin_cache
            + pos * stride_cos_sin_cache_s
            + offs_rotary
            + rotary_dim // 2,
            mask_rotary,
            other=0.0,
        )

        # 计算旋转配对维度的内存地址
        off_q_pe_rot = (
            cur_batch * stride_qb
            + cur_head[:, None] * stride_qh
            + offs_qk_rot_r[None, :]
        )
        mask_qk_rot_r = offs_qk_rot_r < (kv_lora_rank + qk_rope_head_dim)

        # 0, 2, 4,.... 1, 3, 5...
        # 加载 Q_PE 的旋转配对元素（用于实现正交旋转）
        q_pe_rot = tl.load(
            Q + off_q_pe_rot,
            mask=(mask_h[:, None]) & (mask_qk_rot_r[None, :]),
            other=0.0,
        )
        # 对需要取反的维度取负（实现旋转矩阵乘法的一半）
        q_pe_rot = tl.where(mask_rotate[None, :], -q_pe_rot, q_pe_rot)

        # 应用 RoPE 旋转：q_pe = q_pe * cos + q_pe_rot * sin
        q_pe = q_pe * cos + q_pe_rot * sin

        # we only apply to the last token in the K_PE
        # 只在最后一个 split 中处理最后一个 token 的 k_pe RoPE（避免重复计算）
        if LAST_SPLIT:
            # debug assert
            if (cur_batch == 0 and cur_head == 0) and split_kv_id < NUM_KV_SPLITS - 1:
                tl.device_assert(False, "Only last split should compute k_pe")

            # 获取最后一个 KV token 的物理页索引
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + cur_batch_seq_len - 1
            )
            # 计算最后一个 token 的 k_pe 和旋转配对的内存偏移
            offs_buf_k_pe_last_token = kv_loc * stride_buf_kbs + offs_qk_r
            offs_buf_k_pe_rot_last_token = kv_loc * stride_buf_kbs + offs_qk_rot_r
            # 加载最后一个 token 的原始 k_pe
            k_pe_last_token = tl.load(K_Buffer + offs_buf_k_pe_last_token)

            # 加载旋转配对的 k_pe 并取反（同 Q_PE 处理方式一致）
            k_pe_rot_last_token = tl.load(K_Buffer + offs_buf_k_pe_rot_last_token)
            k_pe_rot_last_token = tl.where(
                mask_rotate, -k_pe_rot_last_token, k_pe_rot_last_token
            )

            # 对最后一个 token 的 k_pe 施加 RoPE
            k_pe_last_token = k_pe_last_token * cos + k_pe_rot_last_token * sin

    # 初始化 Flash Attention 的 online 统计量（分 head 并行）
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")  # 当前最大 logit
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)                  # 归一化因子累积
    acc = tl.zeros([BLOCK_H, BLOCK_C], dtype=tl.float32)           # 加权 Value 累积器

    # 若当前 split 有效（起止范围非空），则遍历 KV token 分块
    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            # 生成当前 KV 分块内的 token 索引
            offs_n = start_n + tl.arange(0, BLOCK_N)
            # 通过 kv_indices 获取物理 KV 缓存页地址
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )

            # 计算 K（NOPE 部分）和 K_PE（RoPE 部分）的内存偏移
            offs_buf_kv = kv_loc[None, :] * stride_buf_kbs + offs_c[:, None]
            offs_buf_k_pe = kv_loc[None, :] * stride_buf_kbs + offs_qk_r[:, None]

            # 加载 K_PE（RoPE 位置编码部分的 Key）
            k_pe = tl.load(
                K_Buffer + offs_buf_k_pe,
                mask=(offs_n[None, :] < split_kv_end) & (mask_qk_r[:, None]),
                other=0.0,
            )  # positional embedding part of keys

            # 若启用 RoPE 且处于最后一个 split，对最后一个 token 替换为 RoPE 后的 k_pe
            if (USE_ROPE and LAST_SPLIT) and start_n >= cur_batch_seq_len - BLOCK_N:
                k_pe = tl.where(
                    offs_n[None, :] != (split_kv_end - 1),
                    k_pe,
                    k_pe_last_token[:, None],
                )

            # (16, 64) x (64, 32)
            # dot product of rope parts
            # 计算 RoPE 部分的注意力分数：q_pe @ k_pe^T（位置感知部分）
            qk = tl.dot(q_pe, k_pe.to(q_pe.dtype))

            # 加载 KV（NOPE 共享 latent 张量，同时用于 Key 和 Value）
            kv = tl.load(
                K_Buffer + offs_buf_kv,
                mask=(offs_n[None, :] < split_kv_end) & (mask_c[:, None]),
                other=0.0,
            )  # the shared latent tensor for keys and values

            # (16, 512) x (512, 32)
            # dot product of nope parts
            # 累加 NOPE 部分的注意力分数：q @ kv^T（内容感知部分）
            qk += tl.dot(q, kv)

            # 乘以缩放因子（标准注意力缩放）
            qk *= sm_scale

            # 若设置了 logit_cap，使用 tanh 裁剪注意力分数（防止极端值）
            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            # 屏蔽无效位置（超出序列长度或非有效头）
            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            # 加载 Value（从 V_buffer 中读取 NOPE 维度的值）
            offs_buf_v = kv_loc[:, None] * stride_buf_vbs + offs_c[None, :]
            v = tl.load(
                V_buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_c[None, :]),
                other=0.0,
            )

            # Flash Attention online softmax 更新：更新最大值和归一化因子
            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)  # 旧累积器的衰减因子
            p = tl.exp(qk - n_e_max[:, None])   # 当前块的未归一化注意力概率
            # 对旧累积器进行 rescale
            acc *= re_scale[:, None]
            # (16, 32) x (32, 512)
            # 累积加权 Value：acc += p @ v
            acc += tl.dot(p.to(v.dtype), v)

            # 更新归一化因子和最大值
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        # 计算中间注意力输出的存储偏移
        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_c[None, :]
        )

        # 若启用 RoPE 且为最后一个 split，保存 RoPE 后的最后 token k_pe
        if USE_ROPE:
            if LAST_SPLIT:
                k_pe_last_token_ptrs = (
                    k_pe_t_out
                    + cur_batch * stride_kpe_tokens_out_b
                    + tl.arange(0, BLOCK_R)
                )
                # 将经过 RoPE 的 k_pe 写入输出缓冲区（供后续更新 KV cache 使用）
                tl.store(k_pe_last_token_ptrs, k_pe_last_token, mask=mask_qk_r)

        # 将归一化后的注意力输出写入中间缓冲区（acc / e_sum）
        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_c[None, :]),
        )

        # 计算存储 lse（log-sum-exp）的偏移（放在 kv_lora_rank 位置）
        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + kv_lora_rank
        )

        # 写入 log-sum-exp 值：e_max + log(e_sum)（用于 split 间的合并）
        tl.store(
            Att_Out + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


# TODO rope offset
# Python 封装：Stage1 — 分 KV split 并行计算带 RoPE 融合的 MLA 注意力分数
def _decode_grouped_att_m_fwd_rope(
    q,              # Query 张量 [batch, q_head_num, kv_lora_rank + qk_rope_head_dim]
    k_buffer,       # KV 缓存中的 Key 张量
    v_buffer,       # KV 缓存中的 Value 张量
    att_out,        # 中间注意力输出缓冲区
    k_pe_tokens_out,   # 输出 RoPE 后的最后一个 token k_pe
    kv_lora_rank,  # c — KV LoRA 压缩维度
    cos_sin_cache,  # RoPE cos/sin 缓存表
    positions,      # 序列位置
    rotary_dim,     # RoPE 旋转维度
    kv_indptr,      # KV token 序列指针（CSR 格式）
    kv_indices,     # KV token 物理页索引
    num_kv_splits,  # KV 序列分割数
    sm_scale,       # softmax 缩放因子
    logit_cap,      # logit 裁剪上限
    use_rope,       # 是否启用 RoPE 融合
    is_neox_style=True,  # RoPE 风格（True=NeoX，False=GPT-J）
):
    # 若启用 RoPE，必须提供 k_pe 输出缓冲区
    if use_rope:
        assert (
            k_pe_tokens_out is not None
        ), "We must output the k_pe tokens with rope applied if rope fusion enabled."

    # KV token 分块大小
    BLOCK = 32

    # # [TODO] work around shmem limit on MI3xx
    # if _is_hip and kv_lora_rank >= 576:
    #     BLOCK = 16

    # 计算 RoPE 维度大小（K_Buffer 总维度减去 NOPE 维度）
    qk_rope_head_dim = k_buffer.shape[-1] - kv_lora_rank
    # 获取 batch 大小和 Q 头数
    batch, head_num = kv_indptr.shape[0] - 1, q.shape[1]
    # 计算 GQA 分组数
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    # 填充 NOPE 维度和 RoPE 维度到 2 的幂次（Triton 向量化要求）
    BLOCK_C = triton.next_power_of_2(kv_lora_rank)
    BLOCK_R = triton.next_power_of_2(qk_rope_head_dim)

    # 每个头块处理 16 个头（固定值）
    BLOCK_H = 16
    NUM_KV_SPLITS = num_kv_splits
    # 设置 Kernel grid：(batch, head_blocks, num_kv_splits)
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        NUM_KV_SPLITS,
    )

    # 额外的 ROCm/HIP 特定 Kernel 调优参数
    extra_kargs = {}
    num_stages = 2
    if _is_hip:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        # AMD GPU 专属调优参数：wave 每 EU 数、矩阵指令非 K 维大小、kpack 参数
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
        num_stages = 1

    # 启动 Stage1 Triton Kernel（RoPE 融合版本）
    _fwd_grouped_kernel_stage1_rope[grid](
        q,
        k_buffer,
        v_buffer,
        cos_sin_cache,
        positions,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        k_pe_tokens_out,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        v_buffer.stride(0),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        k_pe_tokens_out.stride(0) if use_rope else 0,  # 仅 use_rope 时有效步长
        cos_sin_cache.stride(0) if use_rope else 0,     # 仅 use_rope 时有效步长
        positions.stride(0) if use_rope else 0,         # 仅 use_rope 时有效步长
        rotary_dim,
        kv_lora_rank,
        qk_rope_head_dim,
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_C=BLOCK_C,
        BLOCK_R=BLOCK_R,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        logit_cap=logit_cap,
        USE_ROPE=use_rope,
        IS_NEOX_STYLE=is_neox_style,
        num_warps=4,
        num_stages=num_stages,
        **extra_kargs
    )


# 完整的 MLA decode 注意力前向函数（含 RoPE 融合）
# Stage1：分 split 并行计算注意力分数（_decode_grouped_att_m_fwd_rope）
# Stage2：合并 split 结果，计算最终输出（_decode_softmax_reducev_fwd）
def decode_attention_fwd_grouped_rope(
    q,              # Query 张量
    k_buffer,       # KV 缓存 Key
    v_buffer,       # KV 缓存 Value
    o,              # 最终输出张量
    kv_indptr,      # KV 序列指针（CSR 格式）
    kv_indices,     # KV 物理页索引
    k_pe_tokens,    # 输出的 k_pe tokens（经过 RoPE 后）
    kv_lora_rank,   # KV LoRA 压缩维度
    rotary_dim,     # RoPE 旋转维度
    cos_sin_cache,  # RoPE 缓存表
    positions,      # 序列位置
    attn_logits,    # 中间注意力 logits 缓冲区
    num_kv_splits,  # KV 分割数
    sm_scale,       # softmax 缩放因子
    logit_cap=0.0,        # logit 裁剪上限（0 表示不裁剪）
    use_rope=False,       # 是否启用 RoPE 融合
    is_neox_style=False,  # RoPE 风格
):
    # Stage1：计算各 KV split 的部分注意力输出和 lse
    _decode_grouped_att_m_fwd_rope(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        k_pe_tokens,
        kv_lora_rank,
        cos_sin_cache,
        positions,
        rotary_dim,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        sm_scale,
        logit_cap,
        use_rope,
        is_neox_style,
    )
    # Stage2：对各 split 的中间输出进行 softmax 归一化和加权求和，得到最终注意力输出
    _decode_softmax_reducev_fwd(attn_logits, q, o, v_buffer, kv_indptr, num_kv_splits)
