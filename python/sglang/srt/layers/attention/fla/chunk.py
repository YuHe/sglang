# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# 本模块实现 Gated Delta Rule 的分块（chunk）前向计算
# 流程：门控累积和 -> intra-chunk 计算(w/u/A) -> 跨 chunk 状态更新(h) -> 输出计算(o)

from typing import Optional

import torch
from einops import rearrange

from sglang.srt.layers.attention.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from sglang.srt.layers.attention.fla.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from sglang.srt.layers.attention.fla.chunk_o import chunk_fwd_o
from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
from sglang.srt.layers.attention.fla.index import (
    prepare_chunk_indices,
)
from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd
from sglang.srt.layers.attention.fla.utils import (
    SUPPRESS_LEVEL,
    autocast_custom_fwd,
    input_guard,
)

# 全局 chunk 大小，控制 FLA 分块粒度
CHUNK_SIZE = 64


# 分块门控 delta rule 前向计算核心流程
def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,          # query 张量 [B, T, H, K]
    k: torch.Tensor,          # key 张量 [B, T, H, K]
    v: torch.Tensor,          # value 张量 [B, T, H, V]
    g: torch.Tensor,          # 门控张量（对数空间）[B, T, H]
    beta: torch.Tensor,       # beta 缩放系数 [B, T, H]
    scale: float,             # 注意力缩放因子
    initial_state: torch.Tensor,           # 初始递归状态
    initial_state_indices: torch.Tensor,   # 初始状态索引
    cu_seqlens: Optional[torch.LongTensor] = None,  # 变长序列累积长度
    chunk_indices: torch.LongTensor | None = None,  # chunk 索引映射
):
    # 第一步：计算 chunk 内门控的局部前缀累积和（用于指数衰减）
    g = chunk_local_cumsum(
        g, chunk_size=CHUNK_SIZE, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )

    # 第二步：fused kkt + solve_tril + recompute_w_u
    # 计算 intra-chunk 的 w（写权重）、u（更新值）和 A（注意力矩阵）
    w, u, A = chunk_gated_delta_rule_fwd_intra(
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    # 第三步：利用 w/u 更新跨 chunk 的递归状态 h，并得到修正后的 v_new
    h, v_new = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    # 第四步：利用 q、k、v_new、h、g 计算最终输出 o
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    # 根据 SUPPRESS_LEVEL 决定返回中间变量的详细程度
    if SUPPRESS_LEVEL < 3:
        return g, o, A, None, h, None
    elif SUPPRESS_LEVEL >= 3:
        return g, o, A, w, h, v_new


# 自定义 autograd Function：封装门控 delta rule 的前向计算
class ChunkGatedDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        initial_state_indices: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor] = None,
        use_qk_l2norm_in_kernel: bool = False,
    ):
        # 保存原始 q/k（用于可能的反向传播）
        q_orig = q
        k_orig = k

        # 可选：在 kernel 内对 q/k 做 L2 归一化
        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)

        # 预计算变长序列的 chunk 索引映射
        chunk_indices = (
            prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
            if cu_seqlens is not None
            else None
        )
        # 执行完整的前向计算流程
        g, o, A, w, h, v_new = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        # 返回与输入相同精度的输出和最终状态 h
        return o.to(q.dtype), h


# 禁用 torch.compile 优化的公共接口函数
@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    initial_state_indices: torch.Tensor = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, V, K]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, V, K]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, V, K]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    # 验证输入数据类型一致性，且不支持 float32
    assert q.dtype == k.dtype == v.dtype
    assert (
        q.dtype != torch.float32
    ), "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    assert (
        len(beta.shape) == 3
    ), "beta must be of shape [B, T, H] if head_first=False, or [B, H, T] otherwise."

    if head_first:
        # head-first 格式已弃用，转换为 time-first 格式
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
        q, k, v, beta, g = map(
            lambda x: rearrange(x, "b h t ... -> b t h ..."), (q, k, v, beta, g)
        )
    # if not head_first and q.shape[1] < q.shape[2]:
    #     warnings.warn(
    #         f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
    #         "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
    #         "when head_first=False was specified. "
    #         "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
    #     )
    if cu_seqlens is not None:
        # 变长序列时验证 batch_size 必须为 1
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        # 验证初始状态数量与序列数一致
        if (
            initial_state_indices is not None
            and initial_state_indices.shape[0] != len(cu_seqlens) - 1
        ):
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state_indices.shape[0]}."
            )
    # 默认缩放因子为 1/sqrt(K)
    if scale is None:
        scale = k.shape[-1] ** -0.5
    # 调用 autograd Function 执行前向计算
    o, h = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        initial_state_indices,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
    )
    # 如需 head-first 输出，转换输出格式
    if head_first:
        o = rearrange(o, "b t h ... -> b h t ...")
    # 返回输出 o、None（无梯度流）和最终状态 h
    return o, None, h
