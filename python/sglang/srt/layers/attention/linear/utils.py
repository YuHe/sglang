from __future__ import annotations

# 导入日志模块，用于运行时警告输出
import logging
# 导入枚举类，用于定义 Kernel 后端类型
from enum import Enum
# 类型检查专用导入（避免循环导入）
from typing import TYPE_CHECKING, Optional

# 导入 rank0 进程日志工具（分布式推理中只在 rank0 打印）
from sglang.srt.utils.common import rank0_log

if TYPE_CHECKING:
    # 仅用于类型注解，避免运行时循环导入
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


# 线性注意力 Kernel 后端枚举：支持 Triton、CuteDSL 和 FlashInfer 三种实现
class LinearAttnKernelBackend(Enum):
    TRITON = "triton"       # 基于 Triton JIT 的线性注意力 Kernel
    CUTEDSL = "cutedsl"     # 基于 CUTE DSL（NVIDIA CUTLASS）的高性能 Kernel
    FLASHINFER = "flashinfer"  # 基于 FlashInfer 库的线性注意力 Kernel

    def is_triton(self):
        # 判断是否为 Triton 后端
        return self == LinearAttnKernelBackend.TRITON

    def is_cutedsl(self):
        # 判断是否为 CuteDSL 后端
        return self == LinearAttnKernelBackend.CUTEDSL

    def is_flashinfer(self):
        # 判断是否为 FlashInfer 后端
        return self == LinearAttnKernelBackend.FLASHINFER


# 全局 decode/prefill 阶段的线性注意力 Kernel 后端（延迟初始化）
LINEAR_ATTN_DECODE_BACKEND: Optional[LinearAttnKernelBackend] = None
LINEAR_ATTN_PREFILL_BACKEND: Optional[LinearAttnKernelBackend] = None


def initialize_linear_attn_config(server_args: ServerArgs):
    # 从 server_args 中读取用户指定的线性注意力 Kernel 后端配置
    global LINEAR_ATTN_DECODE_BACKEND
    global LINEAR_ATTN_PREFILL_BACKEND

    # 读取基础后端配置（decode/prefill 单独配置优先，否则使用统一的 base 配置）
    base = server_args.linear_attn_backend
    decode = server_args.linear_attn_decode_backend or base
    prefill = server_args.linear_attn_prefill_backend or base

    # 将字符串配置转换为枚举类型（校验合法性）
    LINEAR_ATTN_DECODE_BACKEND = LinearAttnKernelBackend(decode)
    LINEAR_ATTN_PREFILL_BACKEND = LinearAttnKernelBackend(prefill)
    # 在 rank0 进程中打印配置信息
    rank0_log(
        f"Linear attention kernel backend: "
        f"decode={LINEAR_ATTN_DECODE_BACKEND.value}, "
        f"prefill={LINEAR_ATTN_PREFILL_BACKEND.value}"
    )


def get_linear_attn_decode_backend() -> LinearAttnKernelBackend:
    # 获取 decode 阶段的线性注意力 Kernel 后端
    global LINEAR_ATTN_DECODE_BACKEND
    if LINEAR_ATTN_DECODE_BACKEND is None:
        # 未初始化时发出警告并默认使用 Triton 后端
        logger.warning(
            "LINEAR_ATTN_DECODE_BACKEND is not initialized, using triton backend"
        )
        LINEAR_ATTN_DECODE_BACKEND = LinearAttnKernelBackend.TRITON
    return LINEAR_ATTN_DECODE_BACKEND


def get_linear_attn_prefill_backend() -> LinearAttnKernelBackend:
    # 获取 prefill 阶段的线性注意力 Kernel 后端
    global LINEAR_ATTN_PREFILL_BACKEND
    if LINEAR_ATTN_PREFILL_BACKEND is None:
        # 未初始化时发出警告并默认使用 Triton 后端
        logger.warning(
            "LINEAR_ATTN_PREFILL_BACKEND is not initialized, using triton backend"
        )
        LINEAR_ATTN_PREFILL_BACKEND = LinearAttnKernelBackend.TRITON
    return LINEAR_ATTN_PREFILL_BACKEND
