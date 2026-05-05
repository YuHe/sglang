# 导入类型注解工具
from typing import Tuple, Union

# 导入 PyTorch
import torch

# 导入 Mamba 注意力后端基类（线性注意力的公共接口）
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
# 导入 Triton 实现的 KDA Kernel
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
# 导入线性注意力后端配置工具
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnKernelBackend,
    get_linear_attn_decode_backend,
    get_linear_attn_prefill_backend,
)
# 导入因果 Conv1D 函数（prefill 和 decode 两个版本）
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,      # prefill 阶段的 causal conv1d（多 token）
    causal_conv1d_update,  # decode 阶段的 causal conv1d（单 token 更新）
)
# 导入 RadixLinearAttention（线性注意力层模型接口）
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
# 导入硬件平台检测工具
from sglang.srt.utils import is_cpu, is_cuda, is_npu
# 导入 rank0 日志工具
from sglang.srt.utils.common import rank0_log

# KDA always uses the triton causal_conv1d_fn (no CUDA override).
# Only causal_conv1d_update needs platform-specific overrides for decode.
# KDA 的 causal_conv1d_fn 始终使用 Triton 实现（prefill 无平台特化）
# 但 decode 的 causal_conv1d_update 需要针对 NPU/CPU 进行平台特化
if is_npu():
    # NPU（昇腾）环境：使用昇腾专属 conv1d update 实现
    from sgl_kernel_npu.mamba.causal_conv1d import causal_conv1d_update_npu
    causal_conv1d_update = causal_conv1d_update_npu
elif is_cpu():
    # CPU 环境：使用 C++ 扩展的 CPU conv1d update 实现
    from sgl_kernel.mamba import causal_conv1d_update_cpu
    causal_conv1d_update = causal_conv1d_update_cpu

# 导入 ForwardBatch 和 ModelRunner，用于前向传播和模型运行时类型注解
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class KDAKernelDispatcher:
    """Dispatches KDA kernel calls to the appropriate backend per mode."""

    def __init__(
        self,
        decode_backend: LinearAttnKernelBackend,  # decode 阶段使用的 Kernel 后端
        prefill_backend: LinearAttnKernelBackend,  # prefill 阶段使用的 Kernel 后端
    ):
        # 创建 Triton KDA Kernel（可被 decode 和 extend 共用）
        triton_kernel = TritonKDAKernel()

        # 根据 decode 后端配置选择对应 Kernel
        if decode_backend.is_triton():
            self.decode_kernel = triton_kernel
        elif decode_backend.is_cutedsl():
            # CuteDSL 后端仅支持 CUDA 设备
            if not is_cuda():
                raise ValueError("KDA CuTe DSL backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.kda_cutedsl import (
                CuteDSLKDAKernel,
            )
            self.decode_kernel = CuteDSLKDAKernel()
        else:
            raise ValueError(
                f"Unsupported KDA decode backend: {decode_backend}. "
                "KDA currently only supports 'triton'."
            )

        # 根据 prefill 后端配置选择对应 Kernel（KDA 目前仅支持 Triton prefill）
        if prefill_backend.is_triton():
            self.extend_kernel = triton_kernel
        else:
            raise ValueError(
                f"Unsupported KDA prefill backend: {prefill_backend}. "
                "KDA currently only supports 'triton'."
            )

        # 打印最终选择的 Kernel 配置（仅 rank0 进程）
        rank0_log(
            f"KDA kernel dispatcher: decode={self.decode_kernel.__class__.__name__}, "
            f"extend={self.extend_kernel.__class__.__name__}"
        )

    def decode(
        self,
        q: torch.Tensor,        # Query 张量（decode 阶段）
        k: torch.Tensor,        # Key 张量
        v: torch.Tensor,        # Value 张量
        a: torch.Tensor,        # 输入门控因子 a
        b: torch.Tensor,        # 遗忘门控因子 b
        *,
        A_log: torch.Tensor,         # SSM 衰减矩阵对数
        dt_bias: torch.Tensor,       # 时间步长偏置
        ssm_states: torch.Tensor,    # SSM 隐状态缓存
        cache_indices: torch.Tensor, # 缓存槽位索引
        query_start_loc: torch.Tensor,  # Query 累积起始位置
        **kwargs,
    ) -> torch.Tensor:
        # 转发到 decode Kernel（根据后端配置分派到 Triton 或 CuteDSL）
        return self.decode_kernel.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def extend(
        self,
        q: torch.Tensor,    # Query 张量（prefill 阶段）
        k: torch.Tensor,    # Key 张量
        v: torch.Tensor,    # Value 张量
        g: torch.Tensor,    # 输入门控（sigmoid gating factor）
        beta: torch.Tensor, # 遗忘门控 beta
        *,
        ssm_states: torch.Tensor,    # SSM 隐状态缓存
        cache_indices: torch.Tensor, # 缓存槽位索引
        query_start_loc: torch.Tensor,  # Query 累积起始位置
        **kwargs,
    ) -> torch.Tensor:
        # 转发到 extend Kernel（KDA 始终使用 Triton 后端）
        return self.extend_kernel.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )


# KDA（Kimi Delta Attention）线性注意力后端实现
class KDAAttnBackend(MambaAttnBackendBase):
    """Attention backend for KDA (Kimi Delta Attention) linear attention."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        # 获取当前配置的 decode/prefill 后端类型
        decode_backend = get_linear_attn_decode_backend()
        prefill_backend = get_linear_attn_prefill_backend()
        # 创建 KDA Kernel 分派器（根据后端选择具体 Kernel）
        self.kernel_dispatcher = KDAKernelDispatcher(decode_backend, prefill_backend)

    def forward_decode(
        self,
        layer: RadixLinearAttention,              # 线性注意力层（含权重参数）
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],  # 混合 QKV 投影
        a: torch.Tensor,  # 输入门控因子 a
        b: torch.Tensor,  # 遗忘门控因子 b
        **kwargs,
    ):
        # 获取当前层的 conv 状态和 SSM 状态
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]     # conv1d 的状态缓存（保存历史输入）
        ssm_states = layer_cache.temporal     # SSM 的时序隐状态（线性注意力 KV 状态）
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        # decode 阶段：对混合 QKV 投影执行 causal conv1d 单步更新
        # conv_states.transpose(-1, -2) 调整维度以匹配 conv1d 接口
        qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states.transpose(-1, -2),
            layer.conv_weights,
            layer.bias,
            activation="silu",               # SiLU 激活函数
            conv_state_indices=cache_indices, # 按索引更新对应请求的 conv 状态
        )
        # 按 Q/K/V 维度分割 QKV 张量
        q, k, v = qkv.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)
        # reshape：[n, (h, d)] -> [1, n, h, d]（添加 batch 维度，展开头维度）
        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)  # n (h d) -> 1 n h d

        # 调用 KDA decode Kernel 计算注意力输出并更新 SSM 状态
        return self.kernel_dispatcher.decode(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )

    def forward_extend(
        self,
        layer: RadixLinearAttention,  # 线性注意力层
        forward_batch: ForwardBatch,  # 当前批次信息（包含 prefill/extend 数据）
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],  # 混合 QKV 投影
        a: torch.Tensor,  # 输入门控因子 a
        b: torch.Tensor,  # 遗忘门控因子 b
        **kwargs,
    ):
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        # 获取当前层的 conv 状态和 SSM 状态
        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0].transpose(-1, -2)  # 转置为 conv1d 接口格式
        ssm_states = mamba_cache_params.temporal  # SSM 隐状态

        # 判断每个 prefill 序列是否有历史 prefix（影响 conv1d 初始状态的使用）
        has_initial_state = forward_batch.extend_prefix_lens > 0

        # 按 Q/K/V 维度分割 conv 权重和状态
        splits = [layer.q_dim, layer.k_dim, layer.v_dim]
        q, k, v = mixed_qkv.transpose(0, 1).split(splits, dim=0)
        q_conv_weight, k_conv_weight, v_conv_weight = layer.conv_weights.split(
            splits, dim=0
        )
        q_conv_state, k_conv_state, v_conv_state = conv_states.split(splits, dim=-2)
        # 分割 bias（若有）
        if layer.bias is not None:
            q_bias, k_bias, v_bias = layer.bias.split(splits, dim=0)
        else:
            q_bias, k_bias, v_bias = None, None, None

        # prefill 阶段：对 Q/K/V 分别执行 causal conv1d（多 token 序列）
        q = causal_conv1d_fn(
            q,
            q_conv_weight,
            q_bias,
            activation="silu",
            conv_states=q_conv_state,               # conv1d 历史状态
            has_initial_state=has_initial_state,    # 是否使用 prefix 的历史状态
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,  # 序列长度（CPU 端）
        ).transpose(0, 1)
        k = causal_conv1d_fn(
            k,
            k_conv_weight,
            k_bias,
            activation="silu",
            conv_states=k_conv_state,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)
        v = causal_conv1d_fn(
            v,
            v_conv_weight,
            v_bias,
            activation="silu",
            conv_states=v_conv_state,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)

        # reshape：[n, (h, d)] -> [1, n, h, d]（添加 batch 维度，展开头维度）
        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)  # n (h d) -> 1 n h d

        # 调用 KDA extend Kernel 计算 prefill 阶段注意力输出（分块 chunk_kda）
        core_attn_out = self.kernel_dispatcher.extend(
            q=q,
            k=k,
            v=v,
            g=a,     # 输入门控（对应 chunk_kda 的 g 参数）
            beta=b,  # 遗忘门控（对应 chunk_kda 的 beta 参数）
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            lower_bound=getattr(layer, "lower_bound", None),  # 可选的数值下界约束
        )

        return core_attn_out
