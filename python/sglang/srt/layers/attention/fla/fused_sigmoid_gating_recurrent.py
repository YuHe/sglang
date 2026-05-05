from typing import Optional

import torch
import triton
import triton.language as tl

# Triton kernel：融合 sigmoid 门控计算与 delta rule 递归状态更新
# 流程：门控衰减 g -> beta 计算 -> delta rule 更新 h -> 输出 o = q @ h
@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,          # log(A) 门控参数指针，形状 [H]
    a,              # 动态时间步参数指针，形状 [T, HV] 或 [T, HV*K]
    dt_bias,        # 时间步偏置指针，形状 [HV] 或 [HV*K]
    softplus_beta,  # softplus 函数的缩放系数
    softplus_threshold,  # softplus 数值稳定性阈值
    q,              # query 张量指针
    k,              # key 张量指针
    v,              # value 张量指针
    b,              # beta 门控参数指针
    o,              # 输出张量指针
    h0_source,      # 初始隐状态存储指针
    h0_indices,     # 初始状态索引指针（负值表示无初始状态）
    cu_seqlens,     # 变长序列累积长度指针
    # Parameters for target_verify support (unused for decode)
    # 以下参数用于 target_verify 模式（decode 模式下未使用）
    intermediate_states_buffer,     # 中间状态缓存缓冲区
    intermediate_state_indices,     # 中间状态缓存索引
    cache_steps,                    # 缓存步数
    retrieve_parent_token_ptr,      # 树形注意力父 token 索引指针
    stride_retrieve_parent_token_seq: tl.constexpr,    # 父 token 指针序列步长
    stride_retrieve_parent_token_token: tl.constexpr,  # 父 token 指针 token 步长
    # ================================================
    scale,          # 注意力缩放因子（1/sqrt(K)）
    T,              # 序列长度
    stride_a,       # a 张量沿 token 轴的步长
    stride_q,       # q 张量沿 token 轴的步长
    stride_k,       # k 张量沿 token 轴的步长
    stride_v,       # v 张量沿 token 轴的步长
    stride_b,       # b 张量沿 token 轴的步长
    NP2_T: tl.constexpr,               # T 的下一个 2 的幂次（用于树形注意力掩码）
    B: tl.constexpr,                   # batch 大小
    H: tl.constexpr,                   # 注意力头数（q/k 维度）
    HV: tl.constexpr,                  # value 头数（支持 GQA）
    K: tl.constexpr,                   # key/query 特征维度
    V: tl.constexpr,                   # value 特征维度
    BK: tl.constexpr,                  # key 分块大小（2 的幂次，>=K）
    BV: tl.constexpr,                  # value 分块大小
    USE_INITIAL_STATE: tl.constexpr,   # 是否使用初始隐状态
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,  # 是否在 kernel 内对 q/k 做 L2 归一化
    IS_VARLEN: tl.constexpr,           # 是否为变长序列
    IS_KDA: tl.constexpr,              # 是否为 KDA 模式（按 key 维度门控）
    # Optional flags for target_verify support (default False for decode)
    DISABLE_STATE_UPDATE: tl.constexpr = False,          # 是否禁用状态写回
    CACHE_INTERMEDIATE_STATES: tl.constexpr = False,     # 是否缓存中间状态
    HAS_EAGLE_TREE_CUSTOM_ATTN_MASK: tl.constexpr = False,  # 是否使用树形注意力掩码
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    # 三维 grid：(K分块, V分块, batch*HV 索引)
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    # 计算 GQA 映射：多个 HV head 共享一个 H head
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        # 变长序列：从 cu_seqlens 获取当前序列的起止位置
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    # 计算当前 K/V 分块的维度索引
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    # 初始化各张量指针，指向当前序列起始位置
    p_q = q + bos * stride_q + i_h * K + o_k
    p_k = k + bos * stride_k + i_h * K + o_k
    p_v = v + bos * stride_v + i_hv * V + o_v
    p_b = b + bos * stride_b + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    # Gating computation pointers 门控计算指针
    p_A_log = A_log + i_hv
    if IS_KDA:
        # KDA 模式：a 和 dt_bias 按 key 维度展开
        p_a = a + bos * stride_a + i_hv * K + o_k
        p_dt_bias = dt_bias + i_hv * K + o_k
    else:
        # GDN 模式：a 和 dt_bias 为标量（每个 head 一个值）
        p_a = a + bos * stride_a + i_hv
        p_dt_bias = dt_bias + i_hv

    # 构造 K/V 维度的有效范围掩码
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    # 初始化隐状态 h 为零矩阵 [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        # 加载初始隐状态（如果索引有效）
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_v[None, :] * K
                + o_k[:, None]
            )
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    # Preload tree attention data if needed 预加载树形注意力的父 token 索引
    if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
        token_indices = tl.arange(0, NP2_T)
        mask_retrieve = token_indices < T
        retrieve_parent_token_base = (
            retrieve_parent_token_ptr
            + (i_n * stride_retrieve_parent_token_seq)
            + token_indices * stride_retrieve_parent_token_token
        )
        parent_idx_tokens = tl.load(
            retrieve_parent_token_base, mask=mask_retrieve, other=0
        )

    # Prepare intermediate state cache index if enabled 准备中间状态缓存索引
    cache_idx = -1
    if CACHE_INTERMEDIATE_STATES:
        cache_idx = tl.load(intermediate_state_indices + i_n)

    step_idx = 0
    for _ in range(0, T):
        # Tree attention: load parent's cached state 树形注意力：从父节点缓存加载状态
        if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
            # step_idx == 0 uses b_h from USE_INITIAL_STATE
            if step_idx != 0 and cache_idx >= 0:
                # 找到当前 token 的父 token 位置，并加载其中间状态
                parent_step_idx = tl.sum(
                    tl.where(token_indices == step_idx, parent_idx_tokens, 0)
                )
                step_offset = parent_step_idx * HV * K * V
                cache_ptr = (
                    intermediate_states_buffer
                    + cache_idx * cache_steps * HV * K * V
                    + step_offset
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                b_h = tl.load(cache_ptr, mask=mask_h, other=0).to(tl.float32)

        # Load inputs 加载当前时间步的 q, k, v, b 输入
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        # Compute sigmoid gating 计算 sigmoid 门控值
        # Load gating parameters 加载门控参数
        b_A_log = tl.load(p_A_log).to(tl.float32)
        if IS_KDA:
            b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
            b_dt_bias = tl.load(p_dt_bias, mask=mask_k, other=0).to(tl.float32)
        else:
            b_a = tl.load(p_a).to(tl.float32)
            b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

        # Compute g = -exp(A_log) * softplus(a + dt_bias) 计算门控衰减 g（对数空间）
        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        # Apply softplus with numerical stability 数值稳定的 softplus 计算
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x

        # Compute beta = sigmoid(b) 计算 beta 缩放系数（通过 sigmoid 激活）
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        # Apply L2 normalization if enabled 可选：对 q/k 做 L2 归一化
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))

        b_q = b_q * scale

        # Apply gating to hidden state: h *= exp(g) 对隐状态施加门控衰减
        if IS_KDA:
            b_h *= tl.exp(b_g[:, None])
        else:
            b_h *= tl.exp(b_g)

        # Delta rule: v -= sum(h * k, dim=0) delta rule 修正：减去历史状态对应的估计
        b_v -= tl.sum(b_h * b_k[:, None], 0)

        # Apply beta gating: v *= beta 对修正后的 v 施加 beta 缩放
        b_v *= b_beta

        # Update hidden state: h += k[:, None] * v[None, :] 外积更新隐状态
        b_h += b_k[:, None] * b_v[None, :]

        # Compute output: o = sum(h * q, dim=0) 计算输出：当前状态与 q 的内积
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Cache intermediate states if enabled 可选：缓存当前时间步的中间状态
        if CACHE_INTERMEDIATE_STATES:
            if cache_idx >= 0:
                step_offset = step_idx * HV * K * V
                cache_ptr = (
                    intermediate_states_buffer
                    + cache_idx * cache_steps * HV * K * V
                    + step_offset
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                tl.store(cache_ptr, b_h.to(cache_ptr.dtype.element_ty), mask=mask_h)

        step_idx += 1

        # Update pointers for next timestep 推进所有指针到下一个时间步
        p_q += stride_q
        p_k += stride_k
        p_v += stride_v
        p_b += stride_b
        p_o += HV * V
        p_a += stride_a

    # Store final state back to h0_source with bounds checking 将最终隐状态写回存储
    if not DISABLE_STATE_UPDATE:
        if USE_INITIAL_STATE:
            idx = tl.load(h0_indices + i_n)
            if idx >= 0:
                p_h0 = (
                    h0_source
                    + idx * HV * K * V
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


# Python 封装：融合 sigmoid 门控与 delta rule 递归更新的统一接口
def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,            # log(A) 门控参数
    a: torch.Tensor,                # 动态时间步参数
    dt_bias: torch.Tensor,          # 时间步偏置
    softplus_beta: float,           # softplus 缩放系数
    softplus_threshold: float,      # softplus 数值稳定性阈值
    q: torch.Tensor,                # query 张量
    k: torch.Tensor,                # key 张量
    v: torch.Tensor,                # value 张量
    b: torch.Tensor,                # beta 门控参数
    initial_state_source: torch.Tensor,    # 初始/最终隐状态存储张量
    initial_state_indices: torch.Tensor,   # 状态索引张量
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    is_kda: bool = False,
    # Optional parameters for target_verify support
    # 以下参数用于 target_verify 模式（推测解码验证）
    disable_state_update: bool = False,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    cache_steps: Optional[int] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
):
    """
    Fused triton implementation of sigmoid gating delta rule update.
    This function uses a single fused kernel that combines both sigmoid gating computation
    and the recurrent delta rule update for better performance.

    Supports both decode and target_verify modes:
    - decode: standard single-step update with state write-back
    - target_verify: multi-step with intermediate state caching, optional tree attention,
                     and optional state update disable
    """
    # 解析输入形状：B=batch, T=序列长度, H=q/k头数, K=key维度, V=value维度
    B, T, H, K, V = *k.shape, v.shape[-1]
    stride_q = q.stride()[1]
    stride_k = k.stride()[1]
    stride_v = v.stride()[1]
    stride_b = b.stride()[-2]
    # Both paths (KDA/GDN) advance p_a once per token, so use the token-axis stride.
    # For 2D a ([T, ...]) this is stride(0); for 3D a ([B, T, ...]) this is stride(1).
    # Using stride()[-2] covers GDN [T, HV] and KDA layouts ([T, HV*K] / [B, T, HV*K]).
    # 统一使用 stride()[-2] 获取 token 轴步长，兼容 GDN 和 KDA 两种布局
    stride_a = a.stride()[-2]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    # BK 为 K 的下一个 2 的幂次，BV 限制最大为 32
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    # 分配输出张量，形状 [NK, B, T, HV, V]
    o = q.new_empty(NK, *v.shape)

    # Prepare retrieve_parent_token strides 准备树形注意力的指针步长
    if retrieve_parent_token is not None:
        stride_retrieve_parent_token_seq = retrieve_parent_token.stride(0)
        stride_retrieve_parent_token_token = retrieve_parent_token.stride(1)
    else:
        stride_retrieve_parent_token_seq = 0
        stride_retrieve_parent_token_token = 0

    NP2_T = triton.next_power_of_2(T)

    # grid 为 (K分块数, V分块数, 序列数*HV)
    grid = (NK, NV, N * HV)

    fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q,
        k=k,
        v=v,
        b=b,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=0 if cache_steps is None else cache_steps,
        retrieve_parent_token_ptr=retrieve_parent_token,
        stride_retrieve_parent_token_seq=stride_retrieve_parent_token_seq,
        stride_retrieve_parent_token_token=stride_retrieve_parent_token_token,
        scale=scale,
        T=T,
        stride_a=stride_a,
        stride_q=stride_q,
        stride_k=stride_k,
        stride_v=stride_v,
        stride_b=stride_b,
        NP2_T=NP2_T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_INITIAL_STATE=initial_state_source is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_VARLEN=cu_seqlens is not None,
        IS_KDA=is_kda,
        DISABLE_STATE_UPDATE=disable_state_update,
        CACHE_INTERMEDIATE_STATES=intermediate_states_buffer is not None,
        HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=retrieve_parent_token is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    # 去掉 NK 维度（NK=1），恢复原始 v 形状
    o = o.squeeze(0)
    return o
