# 导入 Optional 类型，用于可选参数的类型注解
from typing import Optional

# 导入 PyTorch，用于 Tensor 类型注解
import torch

# 导入线性注意力 Kernel 抽象基类
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
# 导入 CPU 环境检测工具（CPU 不支持 Triton Kernel）
from sglang.srt.utils import is_cpu

# 仅在非 CPU 环境下导入 Triton Kernel（避免 CPU 环境报错）
if not is_cpu():
    # 导入融合 sigmoid 门控线性递归更新 Kernel（Triton 实现）
    from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )
    # 导入 KDA（Kimi Delta Attention）分块线性注意力前向函数（Triton 实现）
    from sglang.srt.layers.attention.fla.kda import chunk_kda


# Triton 实现的 KDA（Kimi Delta Attention）线性注意力 Kernel
# KDA 结合了 Delta Rule 状态更新和键值对归一化，是 Kimi 提出的线性注意力变体
class TritonKDAKernel(LinearAttnKernelBase):
    """Triton-based kernel for KDA (Kimi Delta Attention) linear attention."""

    def decode(
        self,
        q: torch.Tensor,        # Query 张量（decode 阶段，每个请求 1 个 token）
        k: torch.Tensor,        # Key 张量
        v: torch.Tensor,        # Value 张量
        a: torch.Tensor,        # 输入门控因子 a（控制新 KV 写入权重）
        b: torch.Tensor,        # 遗忘门控因子 b（控制历史状态保留比例）
        *,
        A_log: torch.Tensor,         # SSM 衰减矩阵的对数
        dt_bias: torch.Tensor,       # 时间步长偏置（softplus 激活后控制更新速率）
        ssm_states: torch.Tensor,    # SSM 隐状态缓存（线性注意力 KV 状态矩阵）
        cache_indices: torch.Tensor, # 每个请求对应的缓存槽位索引
        query_start_loc: torch.Tensor,  # Query 在批次中的累积起始位置
        **kwargs,
    ) -> torch.Tensor:
        # 调用 Triton 融合 Kernel：sigmoid 门控 + Delta Rule KDA 单步状态更新
        # is_kda=True 表示使用 KDA 模式（而非标准 GDN Delta Rule 更新）
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,    # 输入历史 KV 状态
            initial_state_indices=cache_indices, # 状态槽位索引
            cu_seqlens=query_start_loc,          # 累积序列长度（CSR 格式）
            use_qk_l2norm_in_kernel=True,        # Kernel 内进行 QK L2 归一化
            softplus_beta=1.0,                   # softplus beta 参数
            softplus_threshold=20.0,             # softplus 数值稳定阈值
            is_kda=True,                         # 启用 KDA 模式（区别于 GDN）
        )

    def extend(
        self,
        q: torch.Tensor,    # Query 张量（prefill/extend 阶段，多 token）
        k: torch.Tensor,    # Key 张量
        v: torch.Tensor,    # Value 张量
        g: torch.Tensor,    # 输入门控（sigmoid gating，替代 a/b 独立门控）
        beta: torch.Tensor, # 忘记门控 beta（Delta Rule 状态更新的衰减因子）
        *,
        ssm_states: torch.Tensor,    # SSM 隐状态缓存（输入初始状态，输出更新后状态）
        cache_indices: torch.Tensor, # 缓存槽位索引
        query_start_loc: torch.Tensor,   # Query 累积起始位置（CSR 格式）
        A_log: Optional[torch.Tensor] = None,      # 可选的 SSM 衰减矩阵对数
        dt_bias: Optional[torch.Tensor] = None,    # 可选的时间步长偏置
        lower_bound: Optional[float] = None,       # 可选的状态更新下界
        **kwargs,
    ) -> torch.Tensor:
        # 调用 chunk_kda：分块形式的 KDA 线性注意力前向计算（适用于长序列 prefill）
        # 分块计算可以在序列维度上分段处理，减少内存占用
        return chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=ssm_states,           # 输入初始状态
            initial_state_indices=cache_indices, # 状态槽位索引
            use_qk_l2norm_in_kernel=True,        # QK L2 归一化
            cu_seqlens=query_start_loc,          # 累积序列长度
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,             # 状态更新的数值下界约束
        )
