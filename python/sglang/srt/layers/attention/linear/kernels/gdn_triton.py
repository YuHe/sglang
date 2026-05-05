# 导入 PyTorch，用于 Tensor 类型注解
import torch

# 导入线性注意力 Kernel 抽象基类
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
# 导入硬件环境检测工具（区分 CPU、NPU、CUDA/HIP）
from sglang.srt.utils import is_cpu, is_npu

# 仅在非 CPU 环境下导入 Triton/CUDA Kernel（CUDA/HIP 设备）
if not is_cpu():
    # 导入 GDN 分块线性注意力前向函数（Triton 实现，适用于 prefill 阶段长序列）
    from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
    # 导入 GDN 融合 packed decode Kernel（QKV 合并处理，减少 kernel launch 开销）
    from sglang.srt.layers.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )
    # 导入融合 sigmoid 门控 Delta Rule 递归更新 Kernel（decode 阶段单步更新）
    from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

# NPU（昇腾）环境：使用昇腾专属 Kernel 替换 CUDA 实现
if is_npu():
    from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu
    from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update_npu,
    )
    # 用 NPU 实现覆盖 CUDA 实现
    chunk_gated_delta_rule = chunk_gated_delta_rule_npu
    fused_sigmoid_gating_delta_rule_update = fused_sigmoid_gating_delta_rule_update_npu
elif is_cpu():
    # CPU 环境：使用 sgl_kernel 的 CPU 实现（C++ 扩展）
    from sgl_kernel.mamba import chunk_gated_delta_rule_cpu
    # 用 CPU 实现覆盖，确保后续代码可统一调用 chunk_gated_delta_rule
    chunk_gated_delta_rule = chunk_gated_delta_rule_cpu
    fused_sigmoid_gating_delta_rule_update = (
        torch.ops.sgl_kernel.fused_sigmoid_gating_delta_rule_update_cpu
    )


# Triton 实现的 GDN（Gated Delta Network）线性注意力 Kernel
# GDN 使用 Delta Rule 更新隐状态：S_t = sigmoid(b) * S_{t-1} + sigmoid(a) * k_t^T * v_t
class TritonGDNKernel(LinearAttnKernelBase):
    """Triton-based kernel for GDN (Gated Delta Network) linear attention."""

    # 是否支持 packed decode 快速路径（仅 CUDA/Triton 支持，NPU/CPU 不支持）
    supports_packed_decode: bool = not is_cpu() and not is_npu()

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,     # 打包的 QKV 投影输出 [B, qkv_dim]（conv1d 之后）
        a: torch.Tensor,             # 门控输入 a [B, HV]（输入门）
        b: torch.Tensor,             # 门控输入 b [B, HV]（遗忘门）
        *,
        A_log: torch.Tensor,         # 对数衰减参数 A_log [HV]
        dt_bias: torch.Tensor,       # 时间步长偏置 dt_bias [HV]
        scale: float,                # 注意力缩放因子（通常为 head_k_dim^{-0.5}）
        ssm_states: torch.Tensor,    # SSM 状态池 [num_slots, HV, V, K]
        cache_indices: torch.Tensor, # 每个请求的状态槽位索引 [B]
        num_v_heads: int,            # Value 头数（TP 分片后的实际头数）
        head_v_dim: int,             # 每个 Value 头的维度
        **kwargs,
    ) -> torch.Tensor:
        """Packed decode fast path: fuse QKV extraction + gating + recurrent
        update into a single Triton kernel, eliminating intermediate tensors
        and extra kernel launches.

        Args:
            mixed_qkv: [B, qkv_dim] packed projection output after conv1d.
            a, b: [B, HV] gating inputs.
            A_log: [HV] log-space decay parameter.
            dt_bias: [HV] time-step bias.
            scale: attention scale factor (typically head_k_dim ** -0.5).
            ssm_states: [num_slots, HV, V, K] full state pool.
            cache_indices: [B] per-request state slot indices.
            num_v_heads: number of value heads (after TP sharding).
            head_v_dim: dimension per value head.

        Returns:
            output tensor of shape [1, B, HV, V] matching the existing
            decode kernel output layout.
        """
        B = mixed_qkv.shape[0]
        # Packed kernel expects output shape [B, 1, HV, V]
        # 预分配输出张量，packed Kernel 直接写入（减少内存分配开销）
        out = mixed_qkv.new_empty(B, 1, num_v_heads, head_v_dim)

        # 调用 packed decode 融合 Kernel：一次 kernel launch 完成 QKV 拆分 + 门控 + 状态更新
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=ssm_states,
            out=out,
            ssm_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,  # 在 Kernel 内进行 QK L2 归一化
        )

        # Convert [B, 1, HV, V] → [1, B, HV, V] to match existing output
        # layout. transpose() returns a view — zero cost.
        # 转置输出形状以匹配 decode 路径的标准输出布局（零拷贝 view 操作）
        return out.transpose(0, 1)

    def decode(
        self,
        q: torch.Tensor,        # Query 张量（decode 阶段）
        k: torch.Tensor,        # Key 张量
        v: torch.Tensor,        # Value 张量
        a: torch.Tensor,        # 输入门控因子 a
        b: torch.Tensor,        # 遗忘门控因子 b
        *,
        A_log: torch.Tensor,         # 对数衰减参数
        dt_bias: torch.Tensor,       # 时间步长偏置
        ssm_states: torch.Tensor,    # SSM 隐状态缓存
        cache_indices: torch.Tensor, # 缓存槽位索引
        query_start_loc: torch.Tensor,  # Query 累积起始位置（CSR 格式）
        **kwargs,
    ) -> torch.Tensor:
        # 调用融合 sigmoid 门控 Delta Rule 单步更新 Kernel（GDN 标准 decode 路径）
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
            use_qk_l2norm_in_kernel=True,        # QK L2 归一化
            softplus_beta=1.0,                   # softplus 的 beta 参数
            softplus_threshold=20.0,             # softplus 的数值稳定阈值
        )

    def extend(
        self,
        q: torch.Tensor,    # Query 张量（prefill/extend 阶段，多 token）
        k: torch.Tensor,    # Key 张量
        v: torch.Tensor,    # Value 张量
        g: torch.Tensor,    # 输入门控（sigmoid gating factor）
        beta: torch.Tensor, # 忘记门控 beta（状态衰减因子）
        *,
        ssm_states: torch.Tensor,    # SSM 隐状态缓存（输入初始状态，输出更新后状态）
        cache_indices: torch.Tensor, # 缓存槽位索引
        query_start_loc: torch.Tensor,  # Query 累积起始位置
        **kwargs,
    ) -> tuple:
        # 默认使用完整状态池（CUDA/Triton 路径：通过 initial_state_indices 索引）
        recurrent_state = ssm_states
        recurrent_state_indices_args = {"initial_state_indices": cache_indices}
        # NPU/CPU 路径：不支持 indices 索引，需要先按索引提取状态
        if is_npu() or is_cpu():
            recurrent_state = ssm_states[cache_indices]  # 按索引提取对应状态
            recurrent_state_indices_args = {}            # 不传递 indices 参数
        # 调用 GDN 分块线性注意力前向计算
        return chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=recurrent_state,   # 输入初始状态
            cu_seqlens=query_start_loc,       # 累积序列长度
            head_first=False,                 # 张量布局：[B, T, H, D] 而非 [H, B, T, D]
            use_qk_l2norm_in_kernel=True,     # QK L2 归一化
            **recurrent_state_indices_args,
        )

    def target_verify(
        self,
        A_log: torch.Tensor,      # 对数衰减参数
        dt_bias: torch.Tensor,    # 时间步长偏置
        q: torch.Tensor,          # Query 张量（投机解码目标 token 的 Q）
        k: torch.Tensor,          # Key 张量
        v: torch.Tensor,          # Value 张量
        a: torch.Tensor,          # 输入门控因子 a
        b: torch.Tensor,          # 遗忘门控因子 b
        *,
        ssm_states: torch.Tensor,    # SSM 隐状态缓存
        cache_indices: torch.Tensor, # 缓存槽位索引
        query_start_loc: torch.Tensor,          # Query 累积起始位置
        intermediate_states_buffer: torch.Tensor,  # 中间状态缓冲区（用于并行验证）
        intermediate_state_indices: torch.Tensor,  # 中间状态的槽位索引
        cache_steps: int,                          # 投机解码的步数（待验证 token 数）
        retrieve_parent_token: torch.Tensor,       # 父 token 索引（树注意力用）
        **kwargs,
    ) -> torch.Tensor:
        # 调用融合 Kernel 执行目标验证（投机解码验证路径）
        # disable_state_update=True 表示不更新主 SSM 状态（仅计算输出用于验证）
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=False,                                     # GDN 模式（非 KDA）
            # target_verify specific parameters
            disable_state_update=True,                        # 禁止状态更新（验证模式）
            intermediate_states_buffer=intermediate_states_buffer,  # 中间状态缓冲区
            intermediate_state_indices=intermediate_state_indices,  # 中间状态索引
            cache_steps=cache_steps,                          # 投机步数
            retrieve_parent_token=retrieve_parent_token,      # 父 token 索引
        )
