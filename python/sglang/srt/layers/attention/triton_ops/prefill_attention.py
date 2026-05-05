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
It supporst page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/f2a54f0912293f683bf1d1695fd12c4098a5bf82/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py#L1
# 导入 PyTorch 和 Triton 相关库
import torch
import triton
import triton.language as tl

# 导入硬件检测工具，用于区分 CUDA 和 ROCm 环境
from sglang.srt.utils import is_cuda, is_hip

# 检测当前是否为 CUDA 或 HIP (ROCm) 环境
_is_cuda = is_cuda()
_is_hip = is_hip()

# 若在 CUDA 或 HIP 设备上，获取 GPU 计算能力（用于选择 block 大小）
if _is_cuda or _is_hip:
    CUDA_CAPABILITY = torch.cuda.get_device_capability()


# Triton JIT 编译的 prefill 阶段前向注意力 Kernel
# 实现 Flash Attention 算法：online softmax + 分块累积，内存高效
@triton.jit
def _fwd_kernel(
    Q,           # Query 张量指针 [总token数, num_heads, head_dim]
    K,           # Key 张量指针 [总token数, num_kv_heads, head_dim]
    V,           # Value 张量指针 [总token数, num_kv_heads, head_dim]
    sm_scale,    # softmax 缩放因子（通常为 1/sqrt(head_dim)）
    B_Start_Loc, # 每个 batch 的起始 token 偏移量 [batch_size]
    B_Seqlen,    # 每个 batch 的序列长度 [batch_size]
    Out,         # 输出张量指针 [总token数, num_heads, head_dim]
    stride_qbs,  # Q 在 token 维度的步长
    stride_qh,   # Q 在 head 维度的步长
    stride_kbs,  # K 在 token 维度的步长
    stride_kh,   # K 在 head 维度的步长
    stride_vbs,  # V 在 token 维度的步长
    stride_vh,   # V 在 head 维度的步长
    stride_obs,  # Out 在 token 维度的步长
    stride_oh,   # Out 在 head 维度的步长
    kv_group_num: tl.constexpr,  # GQA/MQA 中每个 KV 头对应的 Q 头数量
    BLOCK_M: tl.constexpr,       # Query 维度的分块大小（tile 大小）
    BLOCK_DMODEL: tl.constexpr,  # 头维度的填充后大小（2 的幂次）
    BLOCK_N: tl.constexpr,       # Key/Value 维度的分块大小
    IS_CAUSAL: tl.constexpr,     # 是否使用因果注意力掩码（下三角掩码）
    Lk: tl.constexpr,            # Key 的实际头维度大小
):
    # 获取当前 Kernel 对应的 batch 索引（grid 第 0 维）
    cur_batch = tl.program_id(0)
    # 获取当前 Kernel 对应的查询头索引（grid 第 1 维）
    cur_head = tl.program_id(1)
    # 获取当前 Kernel 处理的 Query 分块索引（grid 第 2 维）
    start_m = tl.program_id(2)

    # 通过 GQA/MQA 映射关系，计算当前 Q 头对应的 KV 头索引
    cur_kv_head = cur_head // kv_group_num

    # 加载当前 batch 的序列长度
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    # 加载当前 batch 在所有 token 中的起始位置偏移
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    # 计算当前 Query 分块的起始 token 位置
    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    # 初始化 Key/Value 分块内的偏移索引
    offs_n = tl.arange(0, BLOCK_N)
    # 初始化头维度偏移索引
    offs_d = tl.arange(0, BLOCK_DMODEL)
    # 计算当前 Query 分块内的绝对 token 偏移
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # 计算 Q 的内存地址偏移（行：token，列：head_dim）
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    # 计算 K 的内存地址偏移（列：token，行：head_dim）— 转置形式用于矩阵乘法
    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None]
    # 计算 V 的内存地址偏移（行：token，列：head_dim）
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :]

    # 生成头维度有效性掩码（屏蔽填充的维度）
    mask_d = offs_d < Lk

    # 从全局内存加载 Query 分块（超出序列长度或填充维度的位置置 0）
    q = tl.load(
        Q + off_q,
        mask=(offs_m[:, None] < cur_batch_seq_len) & (mask_d[None, :]),
        other=0.0,
    )

    # 初始化 K 和 V 的基础指针（后续循环中动态偏移）
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # initialize pointer to m and l
    # 初始化 Flash Attention 的 online 统计量：m_i（当前最大值）初始化为 -inf
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    # 初始化归一化因子 l_i（softmax 分母的累积）初始化为 0
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    # 初始化输出累积器 acc（加权 Value 之和）初始化为 0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # 若当前 Query 分块超出序列长度，则跳过整个循环（block_mask = 0）
    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)

    # 计算 Key/Value 迭代终止位置：非因果时遍历全部，因果时只到当前 Q 块末尾
    end_n = (
        cur_batch_seq_len
        if not IS_CAUSAL
        else tl.minimum((start_m + 1) * BLOCK_M, cur_batch_seq_len)
    )
    # 遍历所有 Key/Value 分块（外层 Flash Attention 循环）
    for start_n in range(0, block_mask * end_n, BLOCK_N):
        # 确保 start_n 对齐到 BLOCK_N 边界（提示编译器生成对齐访存指令）
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        # 加载 Key 分块（转置形式：[head_dim, BLOCK_N]），用于 Q*K^T 的矩阵乘法
        k = tl.load(
            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
            mask=((start_n + offs_n[None, :]) < cur_batch_seq_len) & (mask_d[:, None]),
            other=0.0,
        )
        # mask = tl.load(mask_ptrs + start_n, mask=start_n + offs_n < cur_batch_end_loc, other=0.0)

        # 初始化注意力分数矩阵 qk = [BLOCK_M, BLOCK_N]
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # GPU 矩阵乘法：qk += Q_block @ K_block^T，计算原始注意力分数
        qk += tl.dot(q, k)
        # 乘以缩放因子（防止点积过大导致 softmax 梯度消失）
        qk *= sm_scale

        # 因果掩码：仅允许每个 Query 位置关注其之前（含当前）的 Key 位置
        if IS_CAUSAL:
            qk += tl.where(
                (start_n + offs_n[None, :] < cur_batch_seq_len)
                & (offs_m[:, None] >= (start_n + offs_n[None, :])),
                0,
                float("-inf"),
            )
        else:
            # 非因果模式：仅屏蔽超出序列长度的 padding 位置
            qk += tl.where(
                (start_n + offs_n[None, :]) < cur_batch_seq_len, 0, float("-inf")
            )

        # -- compute m_ij, p, l_ij
        # Flash Attention 的 online softmax：计算当前块的最大值 m_ij
        m_ij = tl.max(qk, 1)
        # 计算当前块的未归一化概率（减去 m_ij 保证数值稳定）
        p = tl.exp(qk - m_ij[:, None])
        # 计算当前块的归一化因子 l_ij
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        # 更新全局最大值：m_i_new = max(m_i, m_ij)
        m_i_new = tl.maximum(m_i, m_ij)
        # 计算旧累积结果的衰减因子 alpha = exp(m_i - m_i_new)
        alpha = tl.exp(m_i - m_i_new)
        # 计算当前块概率的缩放因子 beta = exp(m_ij - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        # 更新归一化因子：l_i_new = alpha * l_i + beta * l_ij
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        # 归一化当前块概率（除以新的 l_i_new，并乘以 beta 进行尺度调整）
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        # 对旧累积器进行衰减（rescale 以适应新的最大值）
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # update acc
        # 加载 Value 分块 [BLOCK_N, head_dim]
        v = tl.load(
            v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=((start_n + offs_n[:, None]) < cur_batch_seq_len) & (mask_d[None, :]),
            other=0.0,
        )

        # 将概率矩阵转换为 Value 的数据类型（节省寄存器/SRAM）
        p = p.to(v.dtype)
        # GPU 矩阵乘法：acc += P_block @ V_block，累积加权 Value
        acc += tl.dot(p, v)
        # update m_i and l_i
        # 更新 online 统计量供下一次迭代使用
        l_i = l_i_new
        m_i = m_i_new
    # initialize pointers to output
    # 计算输出张量的内存地址偏移
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :]
    )
    out_ptrs = Out + off_o
    # 将最终累积结果写回全局内存（仅写有效 token 和头维度位置）
    tl.store(
        out_ptrs, acc, mask=(offs_m[:, None] < cur_batch_seq_len) & (mask_d[None, :])
    )


# Python 封装函数：调用 _fwd_kernel 执行 prefill 阶段的上下文注意力前向计算
def context_attention_fwd(
    q, k, v, o, b_start_loc, b_seq_len, max_input_len, is_causal=True, sm_scale=None
):
    """
    q, k, v: [b * s, head, head_dim]
    b_start_loc: [b]
    b_seq_len: [b]
    out: [b * s, head, head_dim]
    sm_scale: softmax scale, defaults to 1/sqrt(head_dim)
    """
    # 根据 GPU 计算能力选择分块大小：Ampere 以上用 128，否则用 64
    if (_is_cuda or _is_hip) and CUDA_CAPABILITY[0] > 8:
        BLOCK = 128
    else:
        BLOCK = 64

    # 提取 Q/K/V 的头维度大小
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]

    # 若未指定 sm_scale，默认使用 1/sqrt(head_dim)（标准注意力缩放）
    if sm_scale is None:
        sm_scale = 1.0 / (Lq**0.5)
    # 获取 batch 大小和 Q 的头数量
    batch, head = b_seq_len.shape[0], q.shape[1]
    # 计算 GQA/MQA 分组数（每个 KV 头对应的 Q 头数）
    kv_group_num = q.shape[1] // k.shape[1]

    # 设置 Kernel 的 grid 尺寸：(batch, head, ceil(max_input_len / BLOCK))
    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))
    # 根据头维度大小选择 warp 数量（影响并行粒度）
    num_warps = 4 if Lk <= 64 else 8

    # 启动 Triton Kernel 执行 prefill 注意力计算
    _fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        b_start_loc,
        b_seq_len,
        o,
        q.stride(0),    # Q 的 token 步长
        q.stride(1),    # Q 的 head 步长
        k.stride(0),    # K 的 token 步长
        k.stride(1),    # K 的 head 步长
        v.stride(0),    # V 的 token 步长
        v.stride(1),    # V 的 head 步长
        o.stride(0),    # Out 的 token 步长
        o.stride(1),    # Out 的 head 步长
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=triton.next_power_of_2(Lk),  # 头维度填充到 2 的幂次
        BLOCK_N=BLOCK,
        IS_CAUSAL=is_causal,
        num_warps=num_warps,
        num_stages=1,
        Lk=Lk,
    )
