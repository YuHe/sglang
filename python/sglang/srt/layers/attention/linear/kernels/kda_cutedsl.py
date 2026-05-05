# 导入 PyTorch，用于 Tensor 类型注解
import torch

# 导入 CuteDSL 实现的融合 sigmoid 门控 KDA（Key-Decay-Attention）状态更新 Kernel
from sglang.jit_kernel.cutedsl_kda import cutedsl_fused_sigmoid_gating_kda_update
# 导入线性注意力 Kernel 抽象基类
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)


# CuteDSL 实现的 KDA decode Kernel（仅 CUDA 支持）
# KDA = Key-Decay-Attention，线性注意力的一种变体，通过 sigmoid 门控对 Key 进行衰减
class CuteDSLKDAKernel(LinearAttnKernelBase):
    """CuTe DSL kernel for KDA decode (CUDA only)."""

    def decode(
        self,
        q: torch.Tensor,        # Query 张量（decode 阶段，batch_size=1 per request）
        k: torch.Tensor,        # Key 张量
        v: torch.Tensor,        # Value 张量
        a: torch.Tensor,        # 输入门控因子 a
        b: torch.Tensor,        # 遗忘门控因子 b
        *,
        A_log: torch.Tensor,         # SSM 衰减矩阵的对数（用于计算状态衰减因子）
        dt_bias: torch.Tensor,       # 时间步长偏置（控制状态更新的粒度）
        ssm_states: torch.Tensor,    # SSM 隐状态缓存（线性注意力的 KV 状态）
        cache_indices: torch.Tensor, # 每个请求对应的缓存槽位索引
        query_start_loc: torch.Tensor,  # Query 在批次中的起始位置（CSR 格式）
        **kwargs,
    ) -> torch.Tensor:
        # 调用 CuteDSL 融合 Kernel：执行 sigmoid 门控 + KDA 状态更新
        # 内部融合了：softplus(dt + dt_bias)、sigmoid 门控、KV 状态原地更新、注意力输出计算
        return cutedsl_fused_sigmoid_gating_kda_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,    # 输入历史状态
            initial_state_indices=cache_indices, # 状态槽位索引
            cu_seqlens=query_start_loc,          # 累积序列长度（CSR 指针）
            use_qk_l2norm_in_kernel=True,        # 在 Kernel 内进行 QK L2 归一化
            softplus_beta=1.0,                   # softplus 的 beta 参数
            softplus_threshold=20.0,             # softplus 的数值稳定阈值
        )

    def extend(self, *args, **kwargs):
        # CuteDSL KDA Kernel 仅支持 decode 阶段，不支持 prefill extend
        raise NotImplementedError("CuteDSLKDAKernel only supports decode")

    def target_verify(self, *args, **kwargs):
        # CuteDSL KDA Kernel 不支持投机解码的目标验证
        raise NotImplementedError("CuteDSLKDAKernel only supports decode")
