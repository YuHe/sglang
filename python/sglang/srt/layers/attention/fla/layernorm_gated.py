# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/layernorm_gated.py
# Copyright (c) 2024, Tri Dao.
# Based on the Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
# For the backward pass, we keep weight_grad and bias_grad in registers and accumulate.
# This backward pass is faster for dimensions up to 8k, but after that it's much slower due to register spilling.
# The models we train have hidden dim up to 8k anyway (e.g. Llama 70B), so this is fine.

# 本模块实现带门控输出的融合 LayerNorm/RMSNorm Triton kernel
# 支持两种门控模式：norm_before_gate=True 时先归一化再门控（norm(x)*silu(z)），
# norm_before_gate=False 时先门控再归一化（norm(x*silu(z))）
# 同时支持 GroupNorm（按 group_size 分组归一化）

from functools import lru_cache

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange

from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    cdiv,
    cpu_has_amx_support,
    device_context,
    is_cpu,
    is_npu,
    next_power_of_2,
)

# 检测运行环境：NPU（昇腾）或 CPU AMX 加速
_is_npu = is_npu()
_use_cpu = is_cpu() and cpu_has_amx_support()

# Maximum rows per Triton block for layernorm gated kernel
# 每个 Triton block 处理的最大行数（控制 block 粒度上限）
MAX_ROWS_PER_BLOCK = 4


# 参考实现（PyTorch 版）：带可选门控的 RMSNorm，用于数值对比验证
def rms_norm_ref(
    x,
    weight,
    bias,
    z=None,          # 门控张量（可选）
    eps=1e-6,
    group_size=None,  # GroupNorm 分组大小（None 表示整个特征维度为一组）
    norm_before_gate=True,  # True：先归一化再门控；False：先门控再归一化
    upcast=True,     # 是否将输入转为 float32 计算
):
    dtype = x.dtype
    N = x.shape[-1]
    weight = weight.float()
    bias = bias.float() if bias is not None else None
    if upcast:
        # 上转型到 float32 提高计算精度
        x = x.float()
        z = z.float() if z is not None else z
    if z is not None and not norm_before_gate:
        # 先门控再归一化：x = x * silu(z)
        x = x * F.silu(z)
    if group_size is None:
        # 标准 RMSNorm：对整个特征维度计算 rms
        rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
        out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    else:
        # GroupNorm 模式：将特征维度分组，每组内独立归一化
        x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
        rstd = 1 / torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) + eps)
        out = rearrange(x_group * rstd, "... g d -> ... (g d)") * weight
        if bias is not None:
            out = out + bias
    if z is not None and norm_before_gate:
        # 先归一化再门控：out = norm(x) * silu(z)
        out *= F.silu(z)
    return out.to(dtype)


@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X,  # pointer to the input 输入张量指针
    Y,  # pointer to the output 输出张量指针
    W,  # pointer to the weights 权重指针
    B,  # pointer to the biases 偏置指针
    Z,  # pointer to the other branch 门控张量指针（可选）
    Mean,  # pointer to the mean 均值输出指针（LayerNorm 用）
    Rstd,  # pointer to the 1/std 标准差倒数输出指针
    stride_x_row,  # how much to increase the pointer when moving by 1 row 输入张量行步长
    stride_y_row,  # 输出张量行步长
    stride_z_row,  # 门控张量行步长
    M,  # number of rows in X 总行数
    N: tl.constexpr,  # number of columns in X 每组的特征维度
    eps,  # epsilon to avoid division by zero 防除零小量
    BLOCK_N: tl.constexpr,  # 特征维度分块大小（2 的幂次）
    ROWS_PER_BLOCK: tl.constexpr,  # 每个 block 处理的行数
    HAS_BIAS: tl.constexpr,   # 是否有偏置
    HAS_Z: tl.constexpr,      # 是否有门控输入 z
    NORM_BEFORE_GATE: tl.constexpr,  # 归一化顺序（True=先norm后gate）
    IS_RMS_NORM: tl.constexpr,  # True=RMSNorm，False=LayerNorm
    ACTIVATION: tl.constexpr,   # 门控激活函数类型：'swish'/'silu'/'sigmoid'
):
    # Map the program id to the starting row of X and Y it should compute.
    # 将 program_id 映射到当前 block 负责的起始行
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    # group 索引：支持 GroupNorm（多组特征分别归一化）
    group = tl.program_id(1)

    # Create 2D tile: [ROWS_PER_BLOCK, BLOCK_N]
    # 构造 2D tile 的行列索引
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)

    # Compute offsets for 2D tile
    # 计算行偏移（乘以行步长）和列偏移（加上当前 group 的起始列）
    row_offsets = rows[:, None] * stride_x_row
    col_offsets = cols[None, :] + group * N

    # Base pointers
    # 计算输入/输出基址指针
    X_base = X + row_offsets + col_offsets
    Y_base = Y + rows[:, None] * stride_y_row + col_offsets

    # Create mask for valid rows and columns
    # 构造有效行列掩码（处理序列末尾不满足 ROWS_PER_BLOCK 的情况）
    row_mask = rows[:, None] < M
    col_mask = cols[None, :] < N
    mask = row_mask & col_mask

    # Load input data with 2D tile
    # 加载输入数据块，越界位置填 0
    x = tl.load(X_base, mask=mask, other=0.0).to(tl.float32)

    if HAS_Z and not NORM_BEFORE_GATE:
        # 先门控模式：在归一化前对 x 施加门控激活
        Z_base = Z + rows[:, None] * stride_z_row + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            x *= z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            x *= tl.sigmoid(z)

    # Compute mean and variance per row (reduce along axis 1)
    # 沿列轴（axis=1）计算每行的均值和方差
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=1) / N  # Shape: [ROWS_PER_BLOCK] 每行均值
        # Store mean for each row 将均值写回，供反向传播使用
        mean_offsets = group * M + rows
        mean_mask = rows < M
        tl.store(Mean + mean_offsets, mean, mask=mean_mask)
        # Broadcast mean back to 2D for subtraction 广播均值并中心化
        xbar = tl.where(mask, x - mean[:, None], 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N  # Shape: [ROWS_PER_BLOCK] 方差
    else:
        # RMSNorm 无均值中心化，直接计算二次均值
        xbar = tl.where(mask, x, 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N  # Shape: [ROWS_PER_BLOCK]
        mean = 0.0  # Placeholder for RMS norm RMSNorm 无均值，占位符

    # 计算标准差的倒数（快速 rsqrt）
    rstd = tl.rsqrt(var + eps)  # Shape: [ROWS_PER_BLOCK]

    # Store rstd for each row 将标准差倒数写回，供反向传播使用
    rstd_offsets = group * M + rows
    rstd_mask = rows < M
    tl.store(Rstd + rstd_offsets, rstd, mask=rstd_mask)

    # Load weights and biases (broadcast across rows)
    # 加载可学习的缩放权重和偏置（对所有行广播）
    w_offsets = cols + group * N
    w_mask = cols < N
    w = tl.load(W + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

    if HAS_BIAS:
        b = tl.load(B + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

    # Normalize and apply linear transformation
    # 归一化并应用可学习变换：y = (x - mean) / std * w + b
    if not IS_RMS_NORM:
        x_hat = (x - mean[:, None]) * rstd[:, None]
    else:
        x_hat = x * rstd[:, None]

    y = x_hat * w[None, :] + b[None, :] if HAS_BIAS else x_hat * w[None, :]

    if HAS_Z and NORM_BEFORE_GATE:
        # 先归一化后门控：在归一化完成后施加门控激活
        Z_base = Z + rows[:, None] * stride_z_row + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            y *= z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            y *= tl.sigmoid(z)

    # Write output 将计算结果写回输出张量
    tl.store(Y_base, y, mask=mask)


@lru_cache
def _get_sm_count(device: torch.device) -> int:
    """Get and cache the SM count for a given device."""
    # 获取并缓存当前 GPU 设备的 SM（流式多处理器）数量
    props = torch.cuda.get_device_properties(device)
    return props.multi_processor_count


# 根据序列长度和 GPU SM 数量动态计算每个 block 处理的行数
def calc_rows_per_block(M: int, device: torch.device) -> int:
    # When piecewise cuda graph is enabled, use a constant value to avoid
    # torch.compile creating guards on the dynamic batch dimension.
    # 启用 piecewise CUDA graph 时，使用固定值以避免 torch.compile 为动态 batch 维度创建 guard
    try:
        if not get_global_server_args().disable_piecewise_cuda_graph:
            return MAX_ROWS_PER_BLOCK
    except ValueError:
        # Global server args not initialized (e.g., in unit tests)
        # 全局服务参数未初始化时（如单元测试），继续自适应计算
        pass
    sm_count = _get_sm_count(device)
    # 启发式：rows_per_block = next_power_of_2(ceil(M / (2 * sm_count)))，上限为 MAX_ROWS_PER_BLOCK
    rows_per_block = next_power_of_2(cdiv(M, 2 * sm_count))
    rows_per_block = min(rows_per_block, MAX_ROWS_PER_BLOCK)
    return rows_per_block


# Python 封装：执行带门控的 LayerNorm/RMSNorm 前向计算，调用 Triton kernel
def _layer_norm_fwd(
    x,
    weight,
    bias,
    eps,
    z=None,           # 门控张量（可选）
    out=None,         # 预分配输出张量（可选）
    group_size=None,  # GroupNorm 分组大小
    norm_before_gate=True,
    is_rms_norm=False,
    activation: str = "swish",
):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    # 计算分组数（GroupNorm 支持多组独立归一化）
    ngroups = N // group_size
    assert x.stride(-1) == 1
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    # allocate output 分配输出张量（若未预分配）
    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1
    # LayerNorm 需要存储均值，RMSNorm 不需要
    mean = (
        torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
        if not is_rms_norm
        else None
    )
    # 标准差倒数，供反向传播使用
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    # 限制分块大小不超过 64KB / element_size
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps 根据特征维度自适应选择 warp 数
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    # Calculate rows per block based on SM count 基于 SM 数量计算每 block 行数
    rows_per_block = calc_rows_per_block(M, x.device)
    # Update grid to use rows_per_block grid = (行 block 数, 分组数)
    grid = (cdiv(M, rows_per_block), ngroups)
    with device_context(x.device):
        _layer_norm_fwd_1pass_kernel[grid](
            x,
            out,
            weight,
            bias,
            z,
            mean,
            rstd,
            x.stride(0),
            out.stride(0),
            z.stride(0) if z is not None else 0,
            M,
            group_size,
            eps,
            BLOCK_N=BLOCK_N,
            ROWS_PER_BLOCK=rows_per_block,
            HAS_BIAS=bias is not None,
            HAS_Z=z is not None,
            NORM_BEFORE_GATE=norm_before_gate,
            IS_RMS_NORM=is_rms_norm,
            num_warps=num_warps,
            ACTIVATION=activation,
        )
    return out, mean, rstd


# NPU 环境下替换 _layer_norm_fwd 为 NPU 专用实现
if _is_npu:
    from sgl_kernel_npu.fla.layernorm_gated import layer_norm_fwd_npu as _layer_norm_fwd


# 函数式接口：带门控的 RMSNorm/LayerNorm，关键词参数版本
def rms_norm_gated(
    *,
    x,
    weight,
    bias,
    z=None,           # 门控张量 z（可选）
    eps=1e-6,
    group_size=None,  # GroupNorm 分组大小
    norm_before_gate=True,   # True：先归一化后门控
    is_rms_norm=False,       # True：RMSNorm，False：LayerNorm
    activation: str = "swish",  # 门控激活函数
):
    """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))"""

    x_shape_og = x.shape
    # reshape input data into 2D tensor 将多维输入展平为 2D [M, N]
    x = x.reshape(-1, x.shape[-1])
    if x.stride(-1) != 1:
        # 确保内存连续性（Triton kernel 要求最后一维步长为 1）
        x = x.contiguous()
    if z is not None:
        assert z.shape == x_shape_og
        z = z.reshape(-1, z.shape[-1])
        if z.stride(-1) != 1:
            z = z.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    if _is_npu:
        assert activation == "swish", "NPU only supports swish activation"
    # 调用 Triton/NPU 前向 kernel
    y, mean, rstd = _layer_norm_fwd(
        x,
        weight,
        bias,
        eps,
        z=z,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
        activation=activation,
    )
    # 恢复原始输入形状后返回
    return y.reshape(x_shape_og)


# autograd Function 封装：使 rms_norm_gated 支持自动微分（此处仅前向）
class LayerNormFn(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        z=None,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        is_rms_norm=False,
        activation: str = "swish",
    ):
        # 直接调用函数式接口（前向传播无需保存中间变量）
        return rms_norm_gated(
            x=x,
            weight=weight,
            bias=bias,
            eps=eps,
            z=z,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            is_rms_norm=is_rms_norm,
            activation=activation,
        )


# 函数式入口：通过 LayerNormFn.apply 调用带门控的层归一化
def layernorm_fn(
    x,
    weight,
    bias,
    z=None,
    eps=1e-6,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
    activation: str = "swish",
):
    return LayerNormFn.apply(
        x, weight, bias, z, eps, group_size, norm_before_gate, is_rms_norm, activation
    )


# nn.Module 封装：带门控的 LayerNorm 模块（含均值）
class LayerNorm(torch.nn.Module):

    def __init__(
        self,
        hidden_size,
        eps=1e-5,
        group_size=None,
        norm_before_gate=True,
        device=None,
        dtype=None,
    ):
        """If group_size is not None, we do GroupNorm with each group having group_size elements.
        group_size=None is equivalent to group_size=hidden_size (i.e. there's only 1 group).
        """
        # group_size=None 等价于整个特征维度为一组（标准 LayerNorm）

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        # 可学习的缩放权重和偏置
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.bias = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化：weight=1, bias=0（恒等变换）
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x, z=None):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))"""
        # 调用函数式接口，is_rms_norm=False 表示完整 LayerNorm（含均值）
        return layernorm_fn(
            x,
            self.weight,
            self.bias,
            z=z,
            group_size=self.group_size,
            eps=self.eps,
            norm_before_gate=self.norm_before_gate,
            is_rms_norm=False,
        )


# nn.Module 封装：带门控的 RMSNorm 模块（无偏置、无均值）
class RMSNorm(torch.nn.Module):

    def __init__(
        self,
        hidden_size,
        eps=1e-5,
        group_size=None,
        norm_before_gate=True,
        device=None,
        dtype=None,
        activation: str = "swish",
    ):
        """If group_size is not None, we do GroupNorm with each group having group_size elements.
        group_size=None is equivalent to group_size=hidden_size (i.e. there's only 1 group).
        """
        # group_size=None 等价于整个特征维度为一组（标准 RMSNorm）
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.activation = activation
        # 可学习缩放权重（RMSNorm 通常无偏置）
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化：weight=1（恒等缩放）
        torch.nn.init.ones_(self.weight)

    def forward(self, x, z=None):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))"""
        if _use_cpu:
            # CPU AMX 加速路径：使用 sgl_kernel 的融合 CPU 实现
            assert (
                self.norm_before_gate
                and self.group_size is None
                and self.activation == "swish"
            ), "CPU rmsnorm_gated currently only supports norm before gate without group size or activation other than swish"
            return torch.ops.sgl_kernel.fused_rmsnorm_gated_cpu(
                x, self.weight, z, self.eps
            )
        else:
            # GPU 路径：使用 Triton kernel，is_rms_norm=True 跳过均值计算
            return layernorm_fn(
                x,
                self.weight,
                self.bias,
                z=z,
                eps=self.eps,
                group_size=self.group_size,
                norm_before_gate=self.norm_before_gate,
                is_rms_norm=True,
                activation=self.activation,
            )
