# 导入抽象基类工具，用于定义线性注意力 Kernel 的统一接口
from abc import ABC, abstractmethod

# 导入 PyTorch，用于 Tensor 类型注解
import torch


class LinearAttnKernelBase(ABC):
    """Abstract base class for linear attention kernel implementations.

    Each concrete implementation wraps a specific kernel (Triton, CuTe DSL, etc.)
    and provides decode/extend/target_verify methods with a unified interface.
    """

    @abstractmethod
    def decode(
        self,
        q: torch.Tensor,        # Query 张量
        k: torch.Tensor,        # Key 张量
        v: torch.Tensor,        # Value 张量
        a: torch.Tensor,        # 线性注意力衰减门控因子 a（输入门）
        b: torch.Tensor,        # 线性注意力输入门控因子 b（遗忘门）
        *,
        A_log: torch.Tensor,         # 对数形式的 SSM 衰减矩阵 A_log
        dt_bias: torch.Tensor,       # 时间步长偏置 dt_bias（控制状态更新速率）
        ssm_states: torch.Tensor,    # SSM 状态缓存（线性注意力的隐状态）
        cache_indices: torch.Tensor, # 每个请求对应的缓存槽位索引
        query_start_loc: torch.Tensor,  # Query 在批次中的起始位置（CSR 格式）
        **kwargs,
    ) -> torch.Tensor: ...  # 返回注意力输出张量

    @abstractmethod
    def extend(
        self,
        q: torch.Tensor,    # Query 张量（prefill 阶段）
        k: torch.Tensor,    # Key 张量
        v: torch.Tensor,    # Value 张量
        g: torch.Tensor,    # 输入门控（sigmoid gating）
        beta: torch.Tensor, # 忘记门控（beta 因子）
        *,
        ssm_states: torch.Tensor,    # SSM 状态缓存（用于输入/输出历史状态）
        cache_indices: torch.Tensor, # 缓存槽位索引
        query_start_loc: torch.Tensor,  # Query 起始位置偏移
        **kwargs,
    ) -> tuple: ...  # 返回 (输出, 更新后的状态) 元组

    def target_verify(
        self,
        A_log: torch.Tensor,      # SSM 衰减矩阵对数
        dt_bias: torch.Tensor,    # 时间步长偏置
        q: torch.Tensor,          # Query 张量
        k: torch.Tensor,          # Key 张量
        v: torch.Tensor,          # Value 张量
        a: torch.Tensor,          # 输入门控 a
        b: torch.Tensor,          # 遗忘门控 b
        *,
        ssm_states: torch.Tensor,    # SSM 状态缓存
        cache_indices: torch.Tensor, # 缓存槽位索引
        query_start_loc: torch.Tensor,  # Query 起始位置
        **kwargs,
    ) -> torch.Tensor:
        # 默认不支持 target_verify（投机解码目标验证），子类可选择性实现
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support target_verify"
        )
