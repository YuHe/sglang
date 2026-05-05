# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# 工具模块：用于 FLA（Flash Linear Attention）分块索引的预计算工具函数

import torch
import triton

from sglang.srt.layers.attention.fla.utils import tensor_cache


# 根据累积序列长度计算每条序列的实际长度
@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    # cu_seqlens 是前缀和形式的序列边界，相邻差即为各序列长度
    return cu_seqlens[1:] - cu_seqlens[:-1]


# 为变长序列批次中的每个 chunk（分块）生成 (序列索引, chunk内偏移) 二元组索引
@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    # 对每条序列按 chunk_size 向上取整，生成该序列内各 chunk 的局部编号
    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
        ]
    )
    # 第一列：序列全局编号（每次从 0 重置时递增）；第二列：chunk 在序列内的局部编号
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


# 计算每条序列的 chunk 数量前缀和，用于定位各序列第一个 chunk 在全局 chunk 表中的偏移
@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    # 先计算每条序列的 chunk 数量，再做累积求和，首位补 0
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)
