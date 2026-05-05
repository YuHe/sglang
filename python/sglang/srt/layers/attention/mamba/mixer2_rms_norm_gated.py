# Mixer2 RMSNorm Gated 模块：为 Mamba2 mixer 提供门控 RMS 归一化层，支持张量并行
from typing import Union

import torch

# 导入张量并行通信原语
from sglang.srt.distributed.communication_op import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
# 导入张量并行状态查询工具
from sglang.srt.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
# 导入融合 RMSNorm+gate 的 CUDA kernel（FLA 实现）
from sglang.srt.layers.attention.fla.layernorm_gated import rms_norm_gated
from sglang.srt.layers.utils import MultiPlatformOp
from sglang.srt.model_loader.weight_utils import sharded_weight_loader
from sglang.srt.utils.common import set_weight_attrs


# 门控 RMSNorm 层：先用 SiLU gate 对输入加权，再做 RMSNorm，支持多种张量并行模式
class Mixer2RMSNormGated(MultiPlatformOp):
    def __init__(
        self,
        full_hidden_size: int,     # 全局（未分片）隐藏层维度
        full_n_groups: int,        # 全局归一化组数
        use_rms_norm: bool = True, # 是否启用 RMSNorm（关闭时只做 gate 乘积）
        eps: float = 1e-6,         # RMSNorm 数值稳定 epsilon
    ):
        super().__init__()
        # 获取张量并行大小和当前 rank
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.full_hidden_size = full_hidden_size
        # 每个归一化组的元素数
        self.group_size = full_hidden_size // full_n_groups
        # 当前 rank 负责的隐藏维度片段大小
        self.per_rank_hidden_size = full_hidden_size // self.tp_size
        # 全局组数
        self.n_groups = full_hidden_size // self.group_size

        self.variance_epsilon = eps
        self.use_rms_norm = use_rms_norm
        if self.use_rms_norm:
            # Register norm weight only if we're actually applying RMSNorm
            # 仅当启用 RMSNorm 时注册可学习的缩放权重（按 tp_rank 分片）
            self.weight = torch.nn.Parameter(torch.ones(self.per_rank_hidden_size))
            set_weight_attrs(self.weight, {"weight_loader": sharded_weight_loader(0)})
        else:
            # Avoid checkpoint mismatch by skipping unused parameter
            # 不启用 RMSNorm 时注册为 None，避免加载 checkpoint 时维度不匹配
            self.register_parameter("weight", None)
        assert (
            self.full_hidden_size % self.tp_size == 0
        ), "Tensor parallel world size must divide hidden size."

    def forward_native(
        self,
        x: torch.Tensor,   # 输入激活，形状 (..., per_rank_hidden_size)
        gate: torch.Tensor, # 门控张量，形状与 x 相同
    ):
        # Three tensor-parallel cases:
        #   1. n_groups is 1
        #      In this case we parallelize along the reduction dim.
        #      Each rank computes a local sum of squares followed by AllReduce
        #   2. tp_size divides n_groups
        #      Each rank only reduces within its local group(s).
        #      No collective ops necessary.
        #   3. The general case can be pretty complicated so we AllGather
        #      the input and then redundantly compute the RMSNorm.
        # 三种张量并行情况：
        #   1. n_groups==1：在归一化维度并行，需 AllReduce 聚合平方和
        #   2. tp_size 整除 n_groups：每 rank 独立归一化，无需通信
        #   3. 一般情况：AllGather 再冗余计算
        input_dtype = x.dtype
        # 先做 gate 乘积（SiLU 激活）：x = x * silu(gate)
        x = x * torch.nn.functional.silu(gate.to(torch.float32))
        if not self.use_rms_norm:
            # 不做 RMSNorm，直接返回 gate 加权结果
            return x.to(input_dtype)

        if self.n_groups == 1:
            if self.tp_size > 1:
                # Compute local sum and then reduce to obtain global sum
                # 计算本 rank 的局部平方和
                local_sums = x.pow(2).sum(dim=-1, keepdim=True)
                # AllReduce 聚合所有 rank 的平方和，得到全局平方和
                global_sums = tensor_model_parallel_all_reduce(local_sums)
                # Calculate the variance
                # 计算全局方差（除以总元素数）
                count = self.tp_size * x.shape[-1]
                variance = global_sums / count

            else:
                # 单卡时直接计算均值方差
                variance = x.pow(2).mean(-1, keepdim=True)
            # RMS 归一化：x = x / sqrt(variance + eps)
            x = x * torch.rsqrt(variance + self.variance_epsilon)
        else:
            # 判断是否需要 AllGather（组数无法被 tp_size 整除时需要冗余计算）
            redundant_tp: bool = self.n_groups % self.tp_size != 0
            if redundant_tp:
                # To handle the general case, redundantly apply the variance
                # AllGather 拼接所有 rank 的 x，用于后续冗余计算
                x = tensor_model_parallel_all_gather(x, -1)

            # 将最后一维重组为 (group_count, group_size) 以便分组归一化
            *prefix_dims, hidden_dim = x.shape
            group_count = hidden_dim // self.group_size
            x_grouped = x.view(*prefix_dims, group_count, self.group_size)
            # 计算每组内的方差
            variance = x_grouped.pow(2).mean(-1, keepdim=True)
            # 分组 RMS 归一化
            x_grouped = x_grouped * torch.rsqrt(variance + self.variance_epsilon)
            x = x_grouped.view(*prefix_dims, hidden_dim)

            if redundant_tp:
                # 截取当前 rank 负责的片段
                start = self.per_rank_hidden_size * self.tp_rank
                end = start + self.per_rank_hidden_size
                x = x[..., start:end]

        # 乘以可学习缩放权重并还原输入 dtype
        return self.weight * x.to(input_dtype)

    def forward_cuda(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # CUDA 路径：优先使用融合 kernel，否则回退到 forward_native
        input_dtype = x.dtype
        if not self.use_rms_norm:
            # Keep gate in float32 for numerical stability during silu
            # 不做归一化时直接返回 gate 乘积，gate 使用 float32 保证 silu 数值稳定
            return x * torch.nn.functional.silu(gate.to(torch.float32)).to(input_dtype)

        if ((self.n_groups % self.tp_size) != 0) or self.n_groups != 1:
            # 组数无法被 tp_size 整除，或组数不为 1，回退到 native 实现
            return self.forward_native(x, gate)

        # 使用 FLA 提供的融合 rms_norm_gated kernel（单组、能被 tp 整除）
        return rms_norm_gated(
            x=x,
            weight=self.weight.data,
            bias=None,
            z=gate,
            eps=self.variance_epsilon,
            norm_before_gate=False,  # gate 先作用于 x，再做 norm
            is_rms_norm=True,
        )
