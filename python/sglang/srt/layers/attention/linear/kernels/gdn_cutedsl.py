# 导入 PyTorch，用于 Tensor 类型注解
import torch

# 导入 CuteDSL 实现的融合 sigmoid 门控 Delta Rule 状态更新 Kernel
# GDN = Gated Delta Network（门控 Delta 网络），是 KDA 的一种变体
from sglang.jit_kernel.cutedsl_gdn import cutedsl_fused_sigmoid_gating_delta_rule_update
# 导入线性注意力 Kernel 抽象基类
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)


# CuteDSL 实现的 GDN decode Kernel（仅 CUDA 支持）
# GDN 使用 Delta Rule 更新隐状态：S_t = beta_t * S_{t-1} + k_t^T * v_t
class CuteDSLGDNKernel(LinearAttnKernelBase):
    """CuTe DSL kernel for GDN decode (CUDA only)."""

    def decode(
        self,
        q: torch.Tensor,        # Query 张量（decode 阶段）
        k: torch.Tensor,        # Key 张量
        v: torch.Tensor,        # Value 张量
        a: torch.Tensor,        # 输入门控因子 a（控制 KV 状态写入强度）
        b: torch.Tensor,        # 遗忘门控因子 b（控制历史状态保留比例）
        *,
        A_log: torch.Tensor,         # SSM 衰减矩阵的对数（用于状态衰减）
        dt_bias: torch.Tensor,       # 时间步长偏置（dt_bias 加到 dt 后经 softplus 激活）
        ssm_states: torch.Tensor,    # SSM 隐状态缓存（GDN 的 KV 状态矩阵）
        cache_indices: torch.Tensor, # 每个请求对应的缓存槽位索引
        query_start_loc: torch.Tensor,  # Query 在批次中的累积起始位置（CSR 格式）
        **kwargs,
    ) -> torch.Tensor:
        # 调用 CuteDSL 融合 Kernel：执行 sigmoid 门控 + Delta Rule 状态更新
        # Delta Rule 更新：S_t = sigmoid(b) * S_{t-1} + sigmoid(a) * k_t^T * v_t
        return cutedsl_fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,    # 输入历史 KV 状态
            initial_state_indices=cache_indices, # 状态槽位索引（批次内的请求映射）
            cu_seqlens=query_start_loc,          # 累积序列长度（CSR 格式指针）
            use_qk_l2norm_in_kernel=True,        # 在 Kernel 内对 QK 进行 L2 归一化
            softplus_beta=1.0,                   # softplus 的 beta 参数（控制激活平滑度）
            softplus_threshold=20.0,             # softplus 的数值稳定阈值（超过时线性化）
        )

    def extend(self, *args, **kwargs):
        # CuteDSL GDN Kernel 仅支持 decode 阶段，不支持 prefill extend
        raise NotImplementedError("CuteDSLGDNKernel only supports decode")

    def target_verify(self, *args, **kwargs):
        # CuteDSL GDN Kernel 不支持投机解码的目标验证
        raise NotImplementedError("CuteDSLGDNKernel only supports decode")
