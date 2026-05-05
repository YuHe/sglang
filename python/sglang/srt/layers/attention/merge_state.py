# 类型提示导入
from typing import Optional, Tuple

import torch
# 导入 sgl_kernel 中的 CUDA 版 merge_state_v2 内核
from sgl_kernel import merge_state_v2

# 导入 Triton 版 merge_state 内核，用于不支持 CUDA 内核时的回退
from sglang.srt.layers.attention.triton_ops.merge_state import merge_state_triton
# 导入 is_cuda 工具函数，用于检测当前设备是否为 CUDA GPU
from sglang.srt.utils import is_cuda

# 在模块加载时一次性检测设备类型，避免重复判断
_is_cuda = is_cuda()


# Automatically fallback to the Triton kernel in some cases
# (e.g., for AMD GPUs, when the head dimension is not a multiple
# of 4 or 8, and in FP8 precision)
def _supported_dtypes(o: torch.Tensor) -> bool:
    # 判断张量数据类型是否为 CUDA 内核支持的类型（float32、float16、bfloat16）
    return o.dtype in [torch.float32, torch.half, torch.bfloat16]


def _supported_headdim(o: torch.Tensor) -> bool:
    # 取出 head 维度大小，张量形状为 [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    headdim = o.shape[2]  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    if o.dtype == torch.float32:
        # float32 时要求 head_dim 为 4 的倍数
        return headdim % 4 == 0
    # 其他精度要求 head_dim 为 8 的倍数
    return headdim % 8 == 0


def merge_state(
    prefix_output: torch.Tensor,   # 前缀部分的注意力输出
    prefix_lse: torch.Tensor,      # 前缀部分的 log-sum-exp 值
    suffix_output: torch.Tensor,   # 后缀部分的注意力输出
    suffix_lse: torch.Tensor,      # 后缀部分的 log-sum-exp 值
    output: Optional[torch.Tensor] = None,       # 可选的输出张量（原地写入）
    output_lse: Optional[torch.Tensor] = None,   # 可选的输出 lse 张量
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # 合并前缀和后缀两段注意力状态，通过 lse 进行数值稳定的加权融合
    if (
        _is_cuda                              # 当前设备为 CUDA
        and _supported_dtypes(prefix_output)  # 数据类型受 CUDA 内核支持
        and _supported_headdim(prefix_output) # head_dim 满足对齐要求
    ):
        # 满足条件时使用高效的 CUDA 融合内核
        return merge_state_v2(
            prefix_output, prefix_lse, suffix_output, suffix_lse, output, output_lse
        )
    else:
        # Fallback to Triton kernel
        # 条件不满足时回退到 Triton 内核（支持 AMD GPU 及其他特殊情形）
        return merge_state_triton(
            prefix_output, prefix_lse, suffix_output, suffix_lse, output, output_lse
        )
