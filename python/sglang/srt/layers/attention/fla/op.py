# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/op.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# 本模块为 FLA Triton kernel 提供统一的数学基本算子入口（exp/log/gather/TMA 描述符）

import os

import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

from sglang.srt.layers.attention.fla.utils import is_gather_supported

# 根据环境变量 FLA_USE_FAST_OPS 决定使用高速近似函数还是标准 Triton 内置函数
if os.environ.get("FLA_USE_FAST_OPS", "0") == "1":
    # 启用快速近似版本：速度更快，精度略低
    exp = tldevice.fast_expf
    exp2 = tldevice.exp2
    log = tldevice.fast_logf
    log2 = tldevice.fast_log2f
else:
    # 默认使用标准 Triton 数学函数，精度更高
    exp = tl.exp
    exp2 = tl.math.exp2
    log = tl.log
    log2 = tl.log2


# 安全指数函数：仅对 x <= 0 的值计算 exp，其余置为 -inf，防止上溢
@triton.jit
def safe_exp(x):
    return exp(tl.where(x <= 0, x, float("-inf")))


# 根据当前 Triton 版本是否支持 tl.gather，提供兼容性适配
if not is_gather_supported:

    # tl.gather 不可用时的降级实现，返回 None 以满足编译器要求
    @triton.jit
    def gather(src, index, axis, _builder=None):
        """
        Gather operation that works when tl.gather is not supported.
        This is a fallback implementation that returns None.
        Just to make triton compiler happy.
        """
        return None

else:
    # 直接使用 Triton 原生 gather 操作
    gather = tl.gather


# 根据 Triton 版本选择正确的 TMA 张量描述符构造函数
if hasattr(triton.language, "_experimental_make_tensor_descriptor"):
    # Triton 3.3.x 使用实验性私有接口
    make_tensor_descriptor = triton.language._experimental_make_tensor_descriptor
elif hasattr(triton.language, "make_tensor_descriptor"):
    # Triton 3.4.x 及以上使用正式接口
    make_tensor_descriptor = triton.language.make_tensor_descriptor
else:
    """
    Fallback implementation when TMA is not supported.
    Returns None to indicate TMA descriptors are unavailable.
    Just make triton compiler happy.
    """

    # TMA 不可用时的降级实现，返回 None 占位
    @triton.jit
    def make_tensor_descriptor(
        base,
        shape,
        strides,
        block_shape,
        _builder=None,
    ):
        return None
