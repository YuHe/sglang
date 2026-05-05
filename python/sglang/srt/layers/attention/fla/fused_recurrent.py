# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/fused_recurrent.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# 本模块实现 Gated Delta Rule 的融合递归前向计算
# 包含三个 Triton kernel：
#   1. fused_recurrent_gated_delta_rule_fwd_kernel：标准前向（prefill 用）
#   2. fused_recurrent_gated_delta_rule_packed_decode_kernel：packed decode（vllm 风格）
#   3. fused_recurrent_gated_delta_rule_update_fwd_kernel：带状态更新的 decode/target_verify 模式
# 递归公式：h_{t} = exp(g_t) * h_{t-1} + k_t^T * (beta_t * (v_t - h_{t-1} * k_t))
#           o_t = h_t @ q_t

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.op import exp
from sglang.srt.layers.attention.fla.utils import input_guard


# Triton kernel：Gated Delta Rule 标准前向，支持 GQA 和变长序列（prefill 场景）
@triton.jit(do_not_specialize=["T"])
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q,        # query 张量指针 [B, T, H, K]
    k,        # key 张量指针 [B, T, H, K]
    v,        # value 张量指针 [B, T, HV, V]
    g,        # gate 标量累积和指针 [B, T, HV]（GDN 模式）或向量 [B, T, H, K]（KDA 模式）
    beta,     # beta 缩放系数指针 [B, T, HV] 或 [B, T, HV, V]
    o,        # 输出张量指针
    h0,       # 初始隐状态指针 [N, HV, V, K]
    ht,       # 最终隐状态输出指针 [N, HV, V, K]
    cu_seqlens,  # 变长序列累积长度
    scale,    # 注意力缩放因子（1/sqrt(K)）
    T,        # 序列长度
    B: tl.constexpr,   # batch 大小
    H: tl.constexpr,   # query/key 头数
    HV: tl.constexpr,  # value 头数（GQA：HV >= H）
    K: tl.constexpr,   # key 特征维度
    V: tl.constexpr,   # value 特征维度
    BK: tl.constexpr,  # key 维度分块（= next_power_of_2(K)）
    BV: tl.constexpr,  # value 维度分块
    USE_INITIAL_STATE: tl.constexpr,      # 是否使用初始隐状态
    STORE_FINAL_STATE: tl.constexpr,      # 是否保存最终隐状态
    IS_BETA_HEADWISE: tl.constexpr,       # beta 是否为头维度向量（ndim == v.ndim）
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,  # 是否在 kernel 内对 q/k 做 L2 归一化
    IS_VARLEN: tl.constexpr,              # 是否为变长序列
    IS_KDA: tl.constexpr,                 # 是否为 KDA 模式（向量化 gate）
):
    # 三维 grid：(K分块, V分块, 序列数*HV)
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    # GQA 映射：多个 HV head 共享一个 H head
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        # 变长序列：从 cu_seqlens 获取当前序列起止位置
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T
    # 当前 K/V 分块的维度索引
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    # 初始化各张量指针，指向当前序列起始位置
    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        # 向量化 beta：每个 HV head 有 V 维 beta
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        # 标量 beta：每个 HV head 一个值
        p_beta = beta + bos * HV + i_hv
    if not IS_KDA:
        # GDN 模式：标量 gate（每个 HV head 一个值）
        p_g = g + bos * HV + i_hv
    else:
        # KDA 模式：向量 gate（每个 H head 有 K 维 gate）
        p_gk = g + (bos * H + i_h) * K + o_k

    # 输出指针：形状 [NK, B, T, HV, V] 展开后的偏移
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    # 构造 K/V 有效维度掩码
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    # 初始化隐状态 h 为零矩阵 [BV, BK]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        # 从初始状态张量加载当前 head 的 [BV, BK] 块
        p_h0 = h0 + i_nh * V * K + o_v[:, None] * K + o_k[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    # 时序递归循环：逐 token 更新隐状态
    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            # 可选：对 q/k 做 L2 归一化（防止数值溢出）
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale
        # [BV, BK] 对隐状态施加门控衰减
        if not IS_KDA:
            b_g = tl.load(p_g).to(tl.float32)
            # GDN：标量 gate，对整个 h 均匀衰减
            b_h *= exp(b_g)
        else:
            b_gk = tl.load(p_gk, mask=mask_k, other=0).to(tl.float32)
            # KDA：向量 gate，按 key 维度独立衰减
            b_h *= exp(b_gk[None, :])
        # [BV] delta rule 修正：v -= sum(h * k, dim=K)
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        # 对修正后的 v 施加 beta 缩放
        b_v *= b_beta
        # [BV, BK] 外积更新隐状态：h += v * k^T
        b_h += b_v[:, None] * b_k[None, :]
        # [BV] 计算输出：o = sum(h * q, dim=K)
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # 推进指针到下一个 token
        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        if not IS_KDA:
            p_g += HV
        else:
            p_gk += H * K
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)

    if STORE_FINAL_STATE:
        # 写回最终隐状态
        p_ht = ht + i_nh * V * K + o_v[:, None] * K + o_k[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


# Python 封装：标准 GDN 融合递归前向（prefill 场景，处理完整序列）
def fused_recurrent_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,          # gate 累积和
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,     # 初始隐状态 [N, HV, V, K]（可选）
    output_final_state: bool,        # 是否输出最终隐状态
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    # BK = next_power_of_2(K)，当前仅支持 NK=1（BK >= K）
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    # 输出张量：先分配 [NK, B, T, HV, V]，最后 squeeze(0)
    o = q.new_empty(NK, *v.shape)
    if output_final_state:
        final_state = q.new_empty(N, HV, V, K, dtype=torch.float32)
    else:
        final_state = None

    # grid = (K分块数, V分块数, 序列数*HV)
    grid = (NK, NV, N * HV)
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=final_state is not None,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_VARLEN=cu_seqlens is not None,
        IS_KDA=False,  # 标准 GDN 模式，使用标量 gate
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o, final_state


# Adapted from vllm project.
# Triton kernel：packed decode 模式的融合递归前向
# 输入 q/k/v 合并为 mixed_qkv（节省内存带宽），内部重建 delta rule 更新
@triton.jit
def fused_recurrent_gated_delta_rule_packed_decode_kernel(
    mixed_qkv,          # 打包的 q/k/v 张量 [B, H*K + H*K + HV*V]
    a,                  # 时步缩放参数 a [B, HV]（用于计算 dt）
    b,                  # beta 的 logit 参数 [B, HV]
    A_log,              # 对数衰减参数 [HV]（log(A)）
    dt_bias,            # dt 偏置 [HV]
    o,                  # 输出张量指针 [B, HV, V]
    h0,                 # 初始 SSM 隐状态 [pool_size, HV, V, K]
    ht,                 # 最终 SSM 隐状态（inplace 更新 h0 所在位置）
    ssm_state_indices,  # SSM 状态索引 [B]（负值跳过）
    scale,              # 注意力缩放因子
    stride_mixed_qkv_tok: tl.constexpr,    # mixed_qkv 的 token 步长
    stride_a_tok: tl.constexpr,            # a 的 token 步长
    stride_b_tok: tl.constexpr,            # b 的 token 步长
    stride_init_state_token: tl.constexpr, # 初始状态的 token 步长
    stride_final_state_token: tl.constexpr,# 最终状态的 token 步长
    stride_indices_seq: tl.constexpr,      # 索引数组步长
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,      # softplus 数值稳定阈值（>阈值时直接返回 x）
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr, # 是否对 q/k 做 L2 归一化
):
    # 二维 grid：(V分块, batch*HV)
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    # GQA 映射：多个 HV 头共享一个 H 头的 q/k
    i_h = i_hv // (HV // H)

    # 构造行列偏移索引和掩码
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    # 读取 SSM 状态索引；负值表示该序列无初始状态，输出零
    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
    p_o = o + (i_n * HV + i_hv) * V + o_v

    if state_idx < 0:
        # 无效状态：输出全零并提前退出
        zero = tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty)
        tl.store(p_o, zero, mask=mask_v)
        return

    # 从状态池加载初始隐状态 [BV, BK]
    p_h0 = h0 + state_idx * stride_init_state_token
    p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    # 从 mixed_qkv 中拆分 q/k/v（packed 布局：[q_part | k_part | v_part]）
    p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
    q_off = i_h * K + o_k              # q 在第一段
    k_off = (H * K) + i_h * K + o_k   # k 在第二段
    v_off = (2 * H * K) + i_hv * V + o_v  # v 在第三段
    b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(p_mixed + v_off, mask=mask_v, other=0).to(tl.float32)

    if USE_QK_L2NORM_IN_KERNEL:
        # L2 归一化 q 和 k
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    # 计算门控衰减 g = -exp(A_log) * softplus(a + dt_bias)
    a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    x = a_val + dt_bias_val
    # 数值稳定的 softplus：x > 阈值时直接返回 x（避免 exp 溢出）
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x
    # beta = sigmoid(b)（转回 float32 以保持精度）
    beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)

    # 标准 delta rule 更新：h = exp(g)*h + beta*(v - h*k) * k^T
    b_h *= exp(g_val)                  # [BV, BK] 施加衰减
    b_v -= tl.sum(b_h * b_k[None, :], 1)  # [BV] delta 修正
    b_v *= beta_val                    # [BV] beta 缩放
    b_h += b_v[:, None] * b_k[None, :]    # [BV, BK] 外积更新隐状态
    b_o = tl.sum(b_h * b_q[None, :], 1)   # [BV] 注意力输出
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

    # 将更新后的隐状态写回状态池（inplace 原位更新）
    p_ht = ht + state_idx * stride_final_state_token
    p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


# Python 封装：packed decode 模式的 GDN 融合递归前向
# 与标准接口不同：q/k/v 合并为 mixed_qkv；gate 由 a/A_log/dt_bias 动态计算
def fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv: torch.Tensor,  # 打包的 q/k/v [B, H*K+H*K+HV*V]
    a: torch.Tensor,          # 时步参数 [B, HV]
    b: torch.Tensor,          # beta logit [B, HV]
    A_log: torch.Tensor,      # log(A) [HV]，衰减参数
    dt_bias: torch.Tensor,    # dt 偏置 [HV]
    scale: float,             # 注意力缩放因子
    initial_state: torch.Tensor,  # 初始 SSM 状态 [pool_size, HV, V, K]
    out: torch.Tensor,            # 预分配输出 [B, 1, HV, V]
    ssm_state_indices: torch.Tensor,  # SSM 状态索引 [B]
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 输入校验：mixed_qkv 必须是 2D 且最后一维连续
    if mixed_qkv.ndim != 2:
        raise ValueError(
            f"`mixed_qkv` must be a 2D tensor (got ndim={mixed_qkv.ndim})."
        )
    if mixed_qkv.stride(-1) != 1:
        raise ValueError("`mixed_qkv` must be contiguous in the last dim.")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"`a` and `b` must be 2D tensors (got a.ndim={a.ndim}, b.ndim={b.ndim})."
        )
    if a.stride(-1) != 1 or b.stride(-1) != 1:
        raise ValueError("`a`/`b` must be contiguous in the last dim.")
    if A_log.ndim != 1 or dt_bias.ndim != 1:
        raise ValueError("`A_log`/`dt_bias` must be 1D tensors.")
    if A_log.stride(0) != 1 or dt_bias.stride(0) != 1:
        raise ValueError("`A_log`/`dt_bias` must be contiguous.")
    if ssm_state_indices.ndim != 1:
        raise ValueError(
            f"`ssm_state_indices` must be 1D for packed decode (got ndim={ssm_state_indices.ndim})."
        )
    if not out.is_contiguous():
        raise ValueError("`out` must be contiguous.")

    dev = mixed_qkv.device
    if any(
        t.device != dev
        for t in (a, b, A_log, dt_bias, initial_state, out, ssm_state_indices)
    ):
        raise ValueError("All inputs must be on the same device.")

    # 解析 batch 大小并验证各输入维度一致
    B = mixed_qkv.shape[0]
    if a.shape[0] != B or b.shape[0] != B:
        raise ValueError(
            "Mismatched batch sizes: "
            f"mixed_qkv.shape[0]={B}, a.shape[0]={a.shape[0]}, b.shape[0]={b.shape[0]}."
        )
    if ssm_state_indices.shape[0] != B:
        raise ValueError(
            f"`ssm_state_indices` must have shape [B] (got {tuple(ssm_state_indices.shape)}; expected ({B},))."
        )

    # 从 initial_state 推断 HV/V/K，验证 a/b 的最后一维与 HV 匹配
    if initial_state.ndim != 4:
        raise ValueError(
            f"`initial_state` must be a 4D tensor (got ndim={initial_state.ndim})."
        )
    if initial_state.stride(-1) != 1:
        raise ValueError("`initial_state` must be contiguous in the last dim.")
    HV, V, K = initial_state.shape[-3:]
    if a.shape[1] != HV or b.shape[1] != HV:
        raise ValueError(
            f"`a`/`b` must have shape [B, HV] with HV={HV} (got a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)})."
        )
    if A_log.numel() != HV or dt_bias.numel() != HV:
        raise ValueError(
            f"`A_log` and `dt_bias` must have {HV} elements (got A_log.numel()={A_log.numel()}, dt_bias.numel()={dt_bias.numel()})."
        )
    if out.shape != (B, 1, HV, V):
        raise ValueError(
            f"`out` must have shape {(B, 1, HV, V)} (got out.shape={tuple(out.shape)})."
        )

    # 计算 q_dim = H*K，推断 H；验证 GQA 约束 HV % H == 0
    qkv_dim = mixed_qkv.shape[1]
    qk_dim = qkv_dim - HV * V
    if qk_dim <= 0 or qk_dim % 2 != 0:
        raise ValueError(
            f"Invalid packed `mixed_qkv` last dim={qkv_dim} for HV={HV}, V={V}."
        )
    q_dim = qk_dim // 2
    if q_dim % K != 0:
        raise ValueError(f"Invalid packed Q size {q_dim}: must be divisible by K={K}.")
    H = q_dim // K
    if H <= 0 or HV % H != 0:
        raise ValueError(
            f"Invalid head config inferred from mixed_qkv: H={H}, HV={HV}."
        )

    # BK 对齐 2 的幂次；NK 必须为 1（packed decode 不支持 K 分块）
    BK = triton.next_power_of_2(K)
    if triton.cdiv(K, BK) != 1:
        raise ValueError(
            f"Packed decode kernel only supports NK=1 (got K={K}, BK={BK})."
        )
    # BV 最大 32，限制寄存器占用
    BV = min(triton.next_power_of_2(V), 32)
    num_stages = 3
    num_warps = 1

    # 预计算各张量步长（Triton kernel 需要 constexpr 步长）
    stride_mixed_qkv_tok = mixed_qkv.stride(0)
    stride_a_tok = a.stride(0)
    stride_b_tok = b.stride(0)
    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = initial_state.stride(0)
    stride_indices_seq = ssm_state_indices.stride(0)

    # grid = (V分块数, B*HV)；ht 指向同一 initial_state（inplace 更新 SSM 状态）
    NV = triton.cdiv(V, BV)
    grid = (NV, B * HV)
    fused_recurrent_gated_delta_rule_packed_decode_kernel[grid](
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        h0=initial_state,
        ht=initial_state,
        ssm_state_indices=ssm_state_indices,
        scale=scale,
        stride_mixed_qkv_tok=stride_mixed_qkv_tok,
        stride_a_tok=stride_a_tok,
        stride_b_tok=stride_b_tok,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    # 返回 (out, initial_state)：initial_state 已被 inplace 更新为最终隐状态
    return out, initial_state


# autograd Function 封装：使标准 GDN 融合递归前向支持 torch.autograd（仅前向）
class FusedRecurrentFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: Optional[torch.LongTensor] = None,
        use_qk_l2norm_in_kernel: bool = False,
    ):
        # 直接调用 Triton kernel 封装，返回 (o, final_state)
        o, final_state = fused_recurrent_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
        )

        return o, final_state

    @staticmethod
    @input_guard
    def backward(ctx, do, dht):
        # 反向传播尚未实现（dg 需要物化所有时间步隐状态，成本过高）
        raise NotImplementedError(
            "Backward pass is not implemented yet and we do not have plans to implement it "
            "because we haven't figured out how to compute dg without materializing the full "
            "hidden states for all time steps."
        )


# 公开 API：带门控的 delta rule 融合递归注意力，支持 GVA 和变长序列
def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, HV, V]`.
            GVA is applied if `HV > H`.
        g (torch.Tensor):
            g (decays) of shape `[B, T, HV]`.
        beta (torch.Tensor):
            betas of shape `[B, T, HV]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, HV, V, K]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, HV, V, K]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HV, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, HV, V, K]` if `output_final_state=True` else `None`.
    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, HV, K, V = 4, 2048, 4, 8, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, HV, V, device='cuda')
        >>> g = F.logsigmoid(torch.rand(B, T, HV, device='cuda'))
        >>> beta = torch.rand(B, T, HV, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, HV, V, K, device='cuda')
        >>> o, ht = fused_gated_recurrent_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g, beta))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = fused_gated_recurrent_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    if cu_seqlens is not None:
        # 变长序列校验：batch size 必须为 1，initial_state 数量须等于序列数
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    # 默认缩放因子 1/sqrt(K)（类 softmax attention）
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    # beta 默认全 1（无 delta rule 缩放）
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    # 通过 autograd Function 调用 Triton 前向 kernel
    o, final_state = FusedRecurrentFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
    )
    return o, final_state


# HAS_EAGLE_TREE_CUSTOM_ATTN_MASK is added to support eagle tree attention mask
# 以下注释说明 eagle tree 注意力掩码的树形结构：
# retrieve_parent_token_ptr: [N, NP2_T], retrieve_next_sibling_ptr: [N, NP2_T]
# e.g. for a sequence of length 4, the eagle tree attention structure is:
# retrieve_next_token=[1, 3, -1, -1] -> retrieve_next_token[i]: the 1st child token of token i
# retrieve_next_sibling=[-1, 2, -1, -1] -> retrieve_next_sibling[i]: the 1st tree sibling token of token i
# retrieve_parent_token=[n/a, 0, 0, 1] -> retrieve_parent_token[i]: the parent token of token i
# Tree:
#    0
#   / \
#  1   2
# /
# 3
# When calculating token 3's attention, it should attend to token 1 (parent) and token 0 (grand-parent)
# When calculating token 2's attention, it should attend to token 0 (parent)
# Triton kernel：带状态更新的融合递归前向，支持 decode/target_verify 两种模式
@triton.jit(do_not_specialize=["T"])
def fused_recurrent_gated_delta_rule_update_fwd_kernel(
    q,          # query 张量指针
    k,          # key 张量指针
    v,          # value 张量指针
    g,          # gate 标量累积和指针
    beta,       # beta 缩放系数指针
    o,          # 输出张量指针
    h0_source,  # 初始/最终隐状态存储张量指针
    h0_indices, # 状态索引（负值表示无效状态，跳过计算）
    cu_seqlens, # 变长序列累积长度
    scale,      # 注意力缩放因子
    intermediate_states_buffer,    # 中间状态缓冲区（target_verify 用）
    intermediate_state_indices,    # 中间状态缓冲区索引
    cache_steps,                   # 缓存步数
    retrieve_parent_token_ptr,     # 树形注意力父 token 索引（eagle tree 专用）
    stride_retrieve_parent_token_seq: tl.constexpr,    # 父 token 指针序列步长
    stride_retrieve_parent_token_token: tl.constexpr,  # 父 token 指针 token 步长
    T,          # 序列长度
    NP2_T: tl.constexpr,  # T 的下一个 2 的幂次（用于构造树形掩码数组）
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,              # 是否使用初始隐状态
    IS_BETA_HEADWISE: tl.constexpr,               # beta 是否为头维度向量
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,        # 是否对 q/k 做 L2 归一化
    IS_VARLEN: tl.constexpr,                      # 是否为变长序列
    DISABLE_STATE_UPDATE: tl.constexpr,           # 是否禁用最终状态写回（target_verify）
    DISABLE_OUTPUT_CALCULATION: tl.constexpr,     # 是否禁用输出计算（只更新状态）
    CACHE_INTERMEDIATE_STATES: tl.constexpr,      # 是否缓存中间状态（target_verify）
    HAS_EAGLE_TREE_CUSTOM_ATTN_MASK: tl.constexpr,  # 是否使用 eagle tree 注意力掩码
):
    # 三维 grid：(K分块, V分块, 序列数*HV)
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    # GQA 映射：多个 HV head 共享一个 H head
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    # 初始化各张量指针到当前序列起始位置
    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv
    # gate 指针（标量 gate，仅 GDN 模式）
    p_g = g + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
        # 预加载 eagle tree 父 token 索引数组 [NP2_T]
        token_indices = tl.arange(0, NP2_T)
        mask_retrieve = token_indices < T
        retrieve_parent_token_base = (
            retrieve_parent_token_ptr
            + (i_n * stride_retrieve_parent_token_seq)
            + token_indices * stride_retrieve_parent_token_token
        )
        parent_idx_tokens = tl.load(retrieve_parent_token_base, mask_retrieve)

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    # 初始化隐状态 [BV, BK]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        # Add bounds checking for idx 负索引表示无效，跳过加载
        if idx >= 0:  # Assuming negative indices are invalid
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_v[:, None] * K
                + o_k[None, :]
            )
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    # Prepare intermediate state cache variables if enabled
    # 准备中间状态缓存索引（target_verify 模式用于树形注意力）
    cache_idx = -1
    if CACHE_INTERMEDIATE_STATES:
        cache_idx = tl.load(intermediate_state_indices + i_n)

    step_idx = 0
    for _ in range(0, T):
        if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
            # step_idx = 0 should use the b_h from USE_INITIAL_STATE
            # 非第 0 步：从父 token 缓存加载隐状态（树形 attention）
            if step_idx != 0 and cache_idx >= 0:
                # when calculating current step's attention, load the state from the parent token
                parent_step_idx = tl.sum(
                    tl.where(token_indices == step_idx, parent_idx_tokens, 0)
                )
                step_offset = parent_step_idx * HV * K * V
                cache_ptr = (
                    intermediate_states_buffer
                    + cache_idx * cache_steps * HV * K * V
                    + step_offset
                    + i_hv * K * V
                    + o_v[:, None] * K
                    + o_k[None, :]
                )
                b_h = tl.load(cache_ptr, mask=mask_h, other=0).to(tl.float32)

        # 加载当前时间步输入
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale
        # [BK, BV] 施加门控衰减
        b_h *= exp(b_g)
        # [BV] delta rule 修正：v -= sum(h*k, dim=K)
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta
        # [BV, BK] 外积更新隐状态
        b_h += b_v[:, None] * b_k[None, :]
        # [BV] 计算注意力输出（可选禁用）
        if not DISABLE_OUTPUT_CALCULATION:
            b_o = tl.sum(b_h * b_q[None, :], 1)
            # core attn output
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # store intermediate states if enabled 缓存中间状态（target_verify）
        if CACHE_INTERMEDIATE_STATES:
            if cache_idx >= 0:
                # Compute cache pointer for this step
                step_offset = step_idx * HV * K * V
                cache_ptr = (
                    intermediate_states_buffer
                    + cache_idx * cache_steps * HV * K * V
                    + step_offset
                    + i_hv * K * V
                    + o_v[:, None] * K
                    + o_k[None, :]
                )
                tl.store(cache_ptr, b_h.to(cache_ptr.dtype.element_ty), mask=mask_h)

        step_idx += 1

        # 推进各指针
        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_g += HV
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)

    # Store final state back to h0_source with bounds checking
    # 将最终隐状态写回 h0_source（inplace 更新 SSM 状态）
    # ssm states
    if not DISABLE_STATE_UPDATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:  # Add bounds checking 负索引跳过
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_v[:, None] * K
                + o_k[None, :]
            )
            tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


# Python 封装：带状态索引的 GDN 融合递归前向（decode/target_verify 场景）
def fused_recurrent_gated_delta_rule_update_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state_source: torch.Tensor,   # 状态存储池 [pool_size, HV, K, V]
    initial_state_indices: torch.Tensor,   # 每条序列的状态索引（负值=无状态）
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    disable_state_update: bool = False,    # 禁用最终状态写回（target_verify）
    disable_output_calculation: bool = False,  # 禁用输出计算（仅更新状态）
    intermediate_states_buffer: Optional[torch.Tensor] = None,  # 中间状态缓冲
    intermediate_state_indices: Optional[torch.Tensor] = None,
    cache_steps: Optional[int] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,  # eagle tree 父 token 索引
) -> torch.Tensor:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    # BV 限制为 8，以减少寄存器压力（decode 场景 T 较短）
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    if disable_output_calculation:
        # When output calculation is disabled, allocate minimal tensor
        # 禁用输出时分配最小张量节省内存
        o = q.new_empty(NK, 1, 1, 1, 1)  # minimal allocation
    else:
        o = q.new_empty(NK, *v.shape)

    grid = (NK, NV, N * HV)

    # prepare retrieve next token buffer strides if provided
    # 准备 eagle tree 父 token 指针的步长（如果未提供则为 0）
    if retrieve_parent_token is not None:
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token = (
            retrieve_parent_token.stride(0),
            retrieve_parent_token.stride(1),
        )
    else:
        stride_retrieve_parent_token_seq = stride_retrieve_parent_token_token = 0

    NP2_T = triton.next_power_of_2(T)
    fused_recurrent_gated_delta_rule_update_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        scale=scale,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=0 if cache_steps is None else cache_steps,
        retrieve_parent_token_ptr=retrieve_parent_token,
        stride_retrieve_parent_token_seq=stride_retrieve_parent_token_seq,
        stride_retrieve_parent_token_token=stride_retrieve_parent_token_token,
        T=T,
        NP2_T=NP2_T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_INITIAL_STATE=initial_state_source is not None,
        IS_VARLEN=cu_seqlens is not None,
        CACHE_INTERMEDIATE_STATES=intermediate_states_buffer is not None,
        HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=retrieve_parent_token is not None,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        DISABLE_STATE_UPDATE=disable_state_update,
        DISABLE_OUTPUT_CALCULATION=disable_output_calculation,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o


# autograd Function 封装：带状态索引的 GDN 递归前向（decode/target_verify 场景）
class FusedRecurrentUpdateFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state_source: torch.Tensor,  # 状态池（inplace 更新）
        initial_state_indices: torch.Tensor, # 每条序列的状态索引
        cu_seqlens: Optional[torch.LongTensor] = None,
        use_qk_l2norm_in_kernel: bool = False,
        disable_state_update: bool = False,
        disable_output_calculation: bool = False,
        intermediate_states_buffer: Optional[torch.Tensor] = None,
        intermediate_state_indices: Optional[torch.Tensor] = None,
        cache_steps: Optional[int] = None,
        retrieve_parent_token: Optional[torch.Tensor] = None,
    ):
        # 调用 update_fwd 底层封装
        o = fused_recurrent_gated_delta_rule_update_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state_source=initial_state_source,
            initial_state_indices=initial_state_indices,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            disable_state_update=disable_state_update,
            disable_output_calculation=disable_output_calculation,
            intermediate_states_buffer=intermediate_states_buffer,
            intermediate_state_indices=intermediate_state_indices,
            cache_steps=cache_steps,
            retrieve_parent_token=retrieve_parent_token,
        )

        return o

    @staticmethod
    @input_guard
    def backward(ctx, do, dht):
        # 反向传播尚未实现（dg 需要物化所有时间步隐状态，成本过高）
        raise NotImplementedError(
            "Backward pass is not implemented yet and we do not have plans to implement it "
            "because we haven't figured out how to compute dg without materializing the full "
            "hidden states for all time steps."
        )


# 公开 API：decode/target_verify 模式下的带状态更新 GDN 融合递归注意力
def fused_recurrent_gated_delta_rule_update(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state_source: torch.Tensor = None,  # SSM 状态池（inplace 更新）
    initial_state_indices: torch.Tensor = None,  # 每条序列的状态索引
    cu_seqlens: Optional[torch.LongTensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    disable_state_update: bool = False,          # True：不写回最终状态（target_verify）
    disable_output_calculation: bool = False,    # True：仅更新状态，不计算输出
    intermediate_states_buffer: Optional[torch.Tensor] = None,  # 中间状态缓冲
    intermediate_state_indices: Optional[torch.Tensor] = None,
    cache_steps: Optional[int] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,  # eagle tree 父 token 索引
) -> torch.Tensor:
    if cu_seqlens is not None:
        # 变长序列校验：batch size 必须为 1，状态索引数量须匹配
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state_source is not None:
            if initial_state_indices.shape[0] != len(cu_seqlens) - 1:
                raise ValueError(
                    f"The number of initial states is expected to be equal to the number of input sequences, "
                    f"i.e., {len(cu_seqlens) - 1} rather than {initial_state_indices.shape[0]}."
                )
            if initial_state_indices.shape[0] != intermediate_state_indices.shape[0]:
                raise ValueError(
                    f"The number of intermediate state indices is expected to be equal to the number of input sequences, "
                    f"i.e., {initial_state_indices.shape[0]} != {intermediate_state_indices.shape[0]}."
                )
    # 默认缩放因子 1/sqrt(K)
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    # beta 默认全 1
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    # 通过 autograd Function 调用 Triton 前向 kernel
    o = FusedRecurrentUpdateFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state_source,
        initial_state_indices,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
        disable_state_update,
        disable_output_calculation,
        intermediate_states_buffer,
        intermediate_state_indices,
        cache_steps,
        retrieve_parent_token,
    )
    return o


# 别名：与 fused_recurrent_gated_delta_rule 保持命名兼容
fused_recurrent_gdn = fused_recurrent_gated_delta_rule
