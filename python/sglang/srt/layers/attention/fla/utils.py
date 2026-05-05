# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/utils.py
# -*- coding: utf-8 -*-

# 本模块提供 FLA 的通用工具函数和环境检测，包括设备平台识别、张量缓存、autocast 适配等

import contextlib
import functools
import inspect
import logging
import os
import sys
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, Literal, Optional, Tuple

import torch
import triton
from packaging import version

from sglang.srt.utils.common import torch_release

logger = logging.getLogger(__name__)

# 编译器模式：优化 kernel 编译流程（通过环境变量控制）
COMPILER_MODE = os.getenv("FLA_COMPILER_MODE") == "1"
# CI 环境标识：允许放宽数值精度断言
FLA_CI_ENV = os.getenv("FLA_CI_ENV") == "1"
# 是否缓存 autotune 结果（默认开启，减少重复编译）
FLA_CACHE_RESULTS = os.getenv("FLA_CACHE_RESULTS", "1") == "1"


# 检查当前 Triton 版本是否支持 autotune 结果缓存
SUPPORTS_AUTOTUNE_CACHE = (
    "cache_results" in inspect.signature(triton.autotune).parameters
)

# 构造 autotune 缓存参数字典（兼容旧版 Triton）
autotune_cache_kwargs = (
    {"cache_results": FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}
)


@lru_cache(maxsize=1)
def check_environments():
    """
    Checks the current operating system, Triton version, and Python version,
    issuing warnings if they don't meet recommendations.
    This function's body only runs once due to lru_cache.
    """
    # Check Operating System 检查操作系统：Windows 不受官方支持
    if sys.platform == "win32":
        logger.warning(
            "Detected Windows operating system. Triton does not have an official Windows release, "
            "thus FLA will not be adapted for Windows, and any potential errors will not be fixed. "
            "Please consider using a Linux environment for compatibility."
        )

    triton_version = version.parse(triton.__version__)
    required_triton_version = version.parse("3.2.0")

    # 检查 Triton 版本是否满足最低要求 3.2.0
    if triton_version < required_triton_version:
        logger.warning(
            f"Current Triton version {triton_version} is below the recommended 3.2.0 version. "
            "Errors may occur and these issues will not be fixed. "
            "Please consider upgrading Triton."
        )

    # Check Python version 检查 Python 版本是否满足最低要求 3.11
    py_version = version.parse(f"{sys.version_info.major}.{sys.version_info.minor}")
    required_py_version = version.parse("3.11")

    if py_version < required_py_version:
        logger.warning(
            f"Current Python version {py_version} is below the recommended 3.11 version. "
            "It is recommended to upgrade to Python 3.11 or higher for the best experience."
        )

    return None


# 计算两个张量的最大绝对误差
def get_abs_err(x, y):
    return (x.detach() - y.detach()).flatten().abs().max().item()


# 计算两个张量的相对均方根误差
def get_err_ratio(x, y):
    err = (x.detach() - y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


# 断言两个张量接近，超出阈值时根据 CI 环境选择警告或报错
def assert_close(prefix, ref, tri, ratio, warning=False, err_atol=1e-6):
    abs_atol = get_abs_err(ref, tri)
    msg = f"{prefix} diff: {abs_atol:.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    logger.info(msg)
    error_rate = get_err_ratio(ref, tri)
    # 绝对误差极小时直接通过
    if abs_atol <= err_atol:
        return
    if warning or (FLA_CI_ENV and (error_rate < 0.01 or abs_atol <= 0.3)):
        if error_rate > ratio:
            import warnings

            warnings.warn(msg)
    else:
        assert error_rate < ratio, msg


# GDN 重计算抑制级别，控制前向传播保存的中间变量数量（0=最少，3=最多）
SUPPRESS_LEVEL = int(os.getenv("GDN_RECOMPUTE_SUPPRESS_LEVEL", "0"))


def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent results of a function with tensor inputs.
    This decorator will store the output of the decorated function for the most recent set of input tensors.
    The cache is limited to a fixed size (default is 4). When the cache is full, the oldest entry will be removed.
    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.
    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """

    # 维护固定大小的缓存列表，每条记录为 (args, kwargs, result)
    cache_entries: Tuple[Optional[Tuple], Optional[Dict], Any] = []
    cache_size = 4

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache_entries, cache_size
        # 遍历缓存条目，使用对象身份（is）比较张量参数
        for i, entry in enumerate(cache_entries):
            last_args, last_kwargs, last_result = entry
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args)) and all(
                    k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()
                ):
                    # 命中缓存：将此条目移至末尾（LRU 策略）并返回缓存结果
                    cache_entries = (
                        cache_entries[:i]
                        + cache_entries[i + 1 :]
                        + [(args, kwargs, last_result)]
                    )
                    return last_result

        # 未命中缓存：计算结果并写入缓存（超出大小则淘汰最旧条目）
        result = fn(*args, **kwargs)

        if len(cache_entries) >= cache_size:
            cache_entries = cache_entries[1:]
        cache_entries.append((args, kwargs, result))
        return result

    return wrapper


def input_guard(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # 将所有张量参数转为连续内存布局（contiguous）
        contiguous_args = (
            i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args
        )
        contiguous_kwargs = {
            k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
            for k, v in kwargs.items()
        }

        # 从参数中找到第一个张量，用于确定当前设备
        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        # 根据张量所在设备设置当前上下文，确保 kernel 在正确设备上运行
        if tensor is not None:
            ctx = custom_device_ctx(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


# contiguous 是 input_guard 的别名
contiguous = input_guard


def require_version(version, hint):
    """
    Perform a runtime check of the dependency versions, using the exact same syntax used by pip.
    """

    # 运行时版本检查装饰器，确保依赖版本满足要求
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(ctx, *args, **kwargs):
            from transformers.utils.versions import require_version

            require_version(version, hint)
            return fn(
                ctx,
                *(
                    i if not isinstance(i, torch.Tensor) else i.contiguous()
                    for i in args
                ),
                **{
                    k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
                    for k, v in kwargs.items()
                },
            )

        return wrapper

    return decorator


# gradient checkpointing 装饰器，节省显存
def checkpoint(fn):
    def wrapper(*args, **kwargs):
        return torch.utils.checkpoint.checkpoint(fn, *args, **kwargs)

    return wrapper


# CPU 设备警告（Triton 不支持 CPU 时触发）
def _cpu_device_warning():
    import warnings

    warnings.warn(
        ("Triton is not supported on current platform, roll back to CPU."), stacklevel=1
    )


# 获取指定设备的多处理器（SM）数量，用于计算最优 grid 大小
@lru_cache(maxsize=None)
def get_multiprocessor_count(tensor_idx: int = 0) -> int:
    try:
        return triton.runtime.driver.active.utils.get_device_properties(tensor_idx)[
            "multiprocessor_count"
        ]
    except BaseException:
        _cpu_device_warning()
        return -1


# 获取当前 Triton 后端设备类型（cuda/hip/xpu/cpu）
@lru_cache(maxsize=None)
def get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except BaseException:
        _cpu_device_warning()
        return "cpu"


# 检测 GPU 厂商平台（nvidia/amd/intel/musa）
@lru_cache(maxsize=None)
def _check_platform() -> Literal["nvidia", "amd", "intel", "musa"]:
    device = get_available_device()
    if device == "cuda":
        return "nvidia"
    elif device == "hip":
        return "amd"
    elif device == "xpu":
        return "intel"
    else:
        return device


# For AMD GPUs, the triton backend is 'hip', while for Nvidia GPUs, the triton backend is 'cuda'.
# However, the torch backend is 'cuda' for both Nvidia and AMD GPUs.
# Therefore, we need to check the triton backend to determine the actual GPU vendor.
# AMD hip 后端统一映射为 cuda（PyTorch 对两者使用相同 API）
device = get_available_device() if get_available_device() != "hip" else "cuda"
device_torch_lib = getattr(torch, device)
device_platform = _check_platform()

# 平台标志：用于条件编译和 kernel 选择
is_amd = device_platform == "amd"
is_intel = device_platform == "intel"
is_nvidia = device_platform == "nvidia"
# Intel Arc A 系列独立显卡标识
is_intel_alchemist = is_intel and "Intel(R) Arc(TM) A" in torch.xpu.get_device_name(0)
# Nvidia Hopper 架构（H 系列或计算能力 >= 9.0）
is_nvidia_hopper = is_nvidia and (
    "NVIDIA H" in torch.cuda.get_device_name(0)
    or torch.cuda.get_device_capability()[0] >= 9
)
# 是否启用 CUDA Graph 优化（通过环境变量控制）
use_cuda_graph = is_nvidia and os.environ.get("FLA_USE_CUDA_GRAPH", "0") == "1"

# Nvidia Ampere or newer, haven't check AMD and intel yet.
# Ampere 及以上架构支持 TF32 加速矩阵乘法
is_tf32_supported = is_nvidia and torch.cuda.get_device_capability(0)[0] >= 8
# 检查 Triton 是否支持 gather 操作
is_gather_supported = hasattr(triton.language, "gather")


# 获取所有 GPU 设备的最大共享内存大小（字节）
def get_all_max_shared_mem():
    try:
        return [
            triton.runtime.driver.active.utils.get_device_properties(i)[
                "max_shared_mem"
            ]
            for i in range(device_torch_lib.device_count())
        ]
    except BaseException:
        _cpu_device_warning()
        return [-1]


# GPU 架构的共享内存大小枚举（字节），用于判断 kernel 是否可用
class Backend(Enum):
    ADA = 101376  # RTX 4090
    AMPERE = 166912  # A100
    HOPPER = 232448  # H100
    DEFAULT = 102400  # Default

    @classmethod
    def get_shared_memory(cls, arch: str) -> int:
        # 根据架构名称返回对应的共享内存大小
        try:
            return cls[arch.upper()].value
        except KeyError:
            return cls.DEFAULT.value


# 检查指定设备是否有足够的共享内存以支持特定架构的 kernel
@lru_cache(maxsize=None)
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    try:
        device_shared_mem_list = get_all_max_shared_mem()
        max_shared_memory = device_shared_mem_list[tensor_idx]
        return max_shared_memory >= Backend.get_shared_memory(arch)
    except Exception:
        return False


if torch_release >= (2, 4):
    # PyTorch >= 2.4：使用新版 amp API，支持多后端设备
    device = "cuda" if device == "cpu" else device
    autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type=device)
    autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type=device)

    def custom_device_ctx(index: int):
        # 返回指定设备索引的上下文管理器
        return device_torch_lib.device(index)

else:
    # PyTorch < 2.4：仅支持 CUDA，使用旧版 amp API
    assert (
        device == "cuda"
    ), "Only cuda device is supported for PyTorch version < 2.4.0."
    autocast_custom_fwd = device_torch_lib.amp.custom_fwd
    autocast_custom_bwd = device_torch_lib.amp.custom_bwd

    def custom_device_ctx(index: int):
        return torch.cuda.device(index)


# 重新获取设备平台信息（覆盖之前的字符串赋值）
device_platform = get_available_device()
