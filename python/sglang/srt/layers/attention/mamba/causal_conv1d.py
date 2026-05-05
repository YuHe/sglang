# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py

# 因果一维卷积接口模块：封装 sgl_kernel 和 Triton 两种后端实现

from typing import Optional

import torch

# 导入 Triton 后端的常量及函数
from .causal_conv1d_triton import PAD_SLOT_ID
from .causal_conv1d_triton import causal_conv1d_fn as _causal_conv1d_fn_triton
from .causal_conv1d_triton import causal_conv1d_update as _causal_conv1d_update_triton

try:
    # 尝试导入高性能 sgl_kernel 的因果卷积前向和更新接口
    from sgl_kernel import causal_conv1d_fwd
    from sgl_kernel import causal_conv1d_update as causal_conv1d_update_kernel

    # 验证 torch.ops 中存在对应算子注册
    torch.ops.sgl_kernel.causal_conv1d_update
    _HAS_SGL_KERNEL = True
except (ImportError, AttributeError):
    # sgl_kernel 不可用时回退到 Triton 实现
    _HAS_SGL_KERNEL = False


def _get_seq_lens_cpu(query_start_loc, x):
    # 从累积序列位置张量中计算每条序列的长度列表（在 CPU 上）
    if query_start_loc is not None:
        # 相邻元素差值即为每条序列的 token 数
        return (query_start_loc[1:] - query_start_loc[:-1]).cpu().tolist()
    # 无变长信息时，整个张量视为单条序列
    return [x.shape[-1]]


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    conv_states: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    **kwargs,
):
    """
    因果一维卷积前向函数（prefill 阶段），支持变长序列与卷积状态缓存。

    x: (batch, dim, seqlen) or (dim,cu_seq_len) for varlen
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    bias: (dim,)
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    cache_indices: (batch)  int32
        indicates the corresponding state index,
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
    conv_states: (...,dim,width - 1) itype
        updated inplace if provided
    activation: either None or "silu" or "swish"
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3


    out: (batch, dim, seqlen)
    """
    # 决定是否使用 Triton 后端：
    # 条件1：sgl_kernel 不可用；
    # 条件2：输入最后维非连续（stride != 1）且调用方已预计算 seq_lens_cpu。
    # Triton kernel 接受任意 stride，可避免 .contiguous() 的内存拷贝（大 prefill 批次可节省 >0.6ms/层）
    use_triton = not _HAS_SGL_KERNEL or (x.stride(-1) != 1 and "seq_lens_cpu" in kwargs)
    if use_triton:
        # 如果调用方未传入 seq_lens_cpu，则在此处计算
        if "seq_lens_cpu" not in kwargs:
            kwargs["seq_lens_cpu"] = _get_seq_lens_cpu(query_start_loc, x)
        # 调用 Triton 实现
        return _causal_conv1d_fn_triton(
            x,
            weight,
            bias,
            conv_states=conv_states,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            activation=activation,
            pad_slot_id=pad_slot_id,
            **kwargs,
        )
    # 校验激活函数类型
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    # 保证输入在最后一维上连续，sgl_kernel 要求此约束
    if x.stride(-1) != 1:
        x = x.contiguous()
    # bias 也需连续
    bias = bias.contiguous() if bias is not None else None

    # 调用 sgl_kernel 因果卷积前向，原地更新 conv_states 并将结果写回 x
    causal_conv1d_fwd(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        activation in ["silu", "swish"],  # 是否启用 silu/swish 激活
        pad_slot_id,
    )
    return x


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
):
    """
    因果一维卷积增量更新函数（decode 阶段），每次仅处理一个新 token。

    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state
        starting at the index
        @cache_seqlens % state_len.
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen)
    """
    # 若 sgl_kernel 不可用，则使用 Triton 实现
    use_triton = not _HAS_SGL_KERNEL
    if use_triton:
        return _causal_conv1d_update_triton(
            x,
            conv_state,
            weight,
            bias=bias,
            activation=activation,
            cache_seqlens=cache_seqlens,
            conv_state_indices=conv_state_indices,
            pad_slot_id=pad_slot_id,
        )
    # 校验激活函数类型
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError(
            f"activation must be None, silu, or swish, actual: {activation}"
        )
    # 将激活类型转为布尔值，传给 kernel
    activation_val = activation in ["silu", "swish"]
    # 若输入为 2D（无 seqlen 维），临时扩展最后一维以适配 kernel 接口
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    # 调用 sgl_kernel 卷积更新，原地修改 conv_state 环形缓冲区
    causal_conv1d_update_kernel(
        x,
        conv_state,
        weight,
        bias,
        activation_val,
        cache_seqlens,
        conv_state_indices,
        pad_slot_id,
    )
    # 还原临时扩展的维度
    if unsqueeze:
        x = x.squeeze(-1)
    return x
