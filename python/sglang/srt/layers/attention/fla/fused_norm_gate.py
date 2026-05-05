# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/fused_norm_gate.py
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# 本模块实现融合 LayerNorm/RMSNorm + 输出门控（swish/sigmoid）的 Triton kernel
# 用于 FLA 中门控归一化层的高效前向计算

import torch
import torch.nn as nn
import triton
import triton.language as tl

from sglang.srt.utils import (
    cdiv,
    cpu_has_amx_support,
    is_cpu,
    is_npu,
    next_power_of_2,
)

# 检测是否在 NPU 或 CPU AMX 加速环境下运行
_is_npu = is_npu()
_use_cpu = is_cpu() and cpu_has_amx_support()

# Maximum rows per Triton block for layernorm gated kernel
# 每个 Triton block 处理的最大行数（BT 上限）
MAX_ROWS_PER_BLOCK = 4


# Triton kernel（批量版）：使用 block pointer 处理 BT 行，适合 D<=512 的情况
@triton.jit
def layer_norm_gated_fwd_kernel(
    x,  # pointer to the input 输入张量指针
    g,  # pointer to the gate 门控张量指针
    y,  # pointer to the output 输出张量指针
    w,  # pointer to the weights 权重指针（可选）
    b,  # pointer to the biases 偏置指针（可选）
    residual,  # pointer to the residual 残差输入指针（可选）
    residual_out,  # pointer to the residual 残差输出指针（可选）
    mean,  # pointer to the mean 均值输出指针（LayerNorm 用）
    rstd,  # pointer to the 1/std 标准差倒数输出指针
    eps,  # epsilon to avoid division by zero 防除零小量
    T,  # number of rows in x 总行数
    D: tl.constexpr,   # number of columns in x 特征维度
    BT: tl.constexpr,  # 每个 block 处理的行数
    BD: tl.constexpr,  # 特征维度分块大小（2 的幂次）
    ACTIVATION: tl.constexpr,       # 激活函数类型：'swish'/'silu'/'sigmoid'
    IS_RMS_NORM: tl.constexpr,      # 是否使用 RMSNorm（无均值中心化）
    STORE_RESIDUAL_OUT: tl.constexpr,  # 是否保存残差输出
    HAS_RESIDUAL: tl.constexpr,     # 是否有残差输入
    HAS_WEIGHT: tl.constexpr,       # 是否有可学习权重
    HAS_BIAS: tl.constexpr,         # 是否有偏置
):
    # 每个程序实例处理第 i_t 个 BT 行块
    i_t = tl.program_id(0)

    o_d = tl.arange(0, BD)
    m_d = o_d < D

    # 加载输入数据块 [BT, BD]
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    if HAS_RESIDUAL:
        # 将残差加到输入上（pre-norm 结构）
        p_res = tl.make_block_ptr(
            residual, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0)
        )
        b_x += tl.load(p_res, boundary_check=(0, 1)).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        # 保存加残差后的 x（用于后续反向传播）
        p_res_out = tl.make_block_ptr(
            residual_out, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0)
        )
        tl.store(p_res_out, b_x.to(p_res_out.dtype.element_ty), boundary_check=(0, 1))
    if not IS_RMS_NORM:
        # LayerNorm：计算行均值并存储
        b_mean = tl.sum(b_x, axis=1) / D
        p_mean = tl.make_block_ptr(mean, (T,), (1,), (i_t * BT,), (BT,), (0,))
        tl.store(p_mean, b_mean.to(p_mean.dtype.element_ty), boundary_check=(0,))
        b_xbar = tl.where(m_d[None, :], b_x - b_mean[:, None], 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    else:
        # RMSNorm：无均值中心化，直接计算二次均值
        b_xbar = tl.where(m_d[None, :], b_x, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    # 计算标准差的倒数（归一化因子）
    b_rstd = 1 / tl.sqrt(b_var + eps)

    p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), boundary_check=(0,))

    # 加载可选的权重和偏置
    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_d, mask=m_d).to(tl.float32)
    # 归一化：(x - mean) / std，RMSNorm 则只除以 rms
    b_x_hat = (
        (b_x - b_mean[:, None]) * b_rstd[:, None]
        if not IS_RMS_NORM
        else b_x * b_rstd[:, None]
    )
    # 应用可学习缩放（weight）
    b_y = b_x_hat * b_w[None, :] if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b[None, :]

    # swish/sigmoid output gate 加载门控并应用激活函数
    p_g = tl.make_block_ptr(g, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        # swish/silu: y = y * g * sigmoid(g)
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == "sigmoid":
        # sigmoid gate: y = y * sigmoid(g)
        b_y = b_y * tl.sigmoid(b_g)

    # Write output 将结果写回输出张量
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


# Triton kernel（单行版）：每个程序实例处理一行，适合 D>512 的大特征维度
@triton.jit
def layer_norm_gated_fwd_kernel1(
    x,  # pointer to the input
    g,  # pointer to the gate
    y,  # pointer to the output
    w,  # pointer to the weights
    b,  # pointer to the biases
    residual,  # pointer to the residual
    residual_out,  # pointer to the residual
    mean,  # pointer to the mean
    rstd,  # pointer to the 1/std
    eps,  # epsilon to avoid division by zero
    D: tl.constexpr,   # number of columns in x
    BD: tl.constexpr,  # 特征维度分块（2 的幂次）
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # 每个程序实例处理第 i_t 行
    i_t = tl.program_id(0)
    # 调整指针到当前行的起始位置
    x += i_t * D
    y += i_t * D
    g += i_t * D
    if HAS_RESIDUAL:
        residual += i_t * D
    if STORE_RESIDUAL_OUT:
        residual_out += i_t * D

    o_d = tl.arange(0, BD)
    m_d = o_d < D
    # 加载当前行数据，越界位置填 0
    b_x = tl.load(x + o_d, mask=m_d, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        b_x += tl.load(residual + o_d, mask=m_d, other=0.0).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        tl.store(residual_out + o_d, b_x, mask=m_d)
    if not IS_RMS_NORM:
        # 计算行均值（LayerNorm 模式）
        b_mean = tl.sum(b_x, axis=0) / D
        tl.store(mean + i_t, b_mean)
        b_xbar = tl.where(m_d, b_x - b_mean, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    else:
        # RMSNorm 模式：无均值中心化
        b_xbar = tl.where(m_d, b_x, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)
    tl.store(rstd + i_t, b_rstd)

    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_d, mask=m_d).to(tl.float32)
    # 归一化并应用可学习参数
    b_x_hat = (b_x - b_mean) * b_rstd if not IS_RMS_NORM else b_x * b_rstd
    b_y = b_x_hat * b_w if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b

    # swish/sigmoid output gate 应用门控激活
    b_g = tl.load(g + o_d, mask=m_d, other=0.0).to(tl.float32)
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == "sigmoid":
        b_y = b_y * tl.sigmoid(b_g)

    # Write output 写回输出
    tl.store(y + o_d, b_y, mask=m_d)


# Python 封装：根据特征维度选择合适的 Triton kernel 执行融合归一化+门控前向计算
def layer_norm_gated_fwd(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = "swish",
    eps: float = 1e-5,
    residual: torch.Tensor = None,
    out_dtype: torch.dtype = None,
    residual_dtype: torch.dtype = None,
    is_rms_norm: bool = False,
):
    if residual is not None:
        residual_dtype = residual.dtype
    T, D = x.shape
    if residual is not None:
        assert residual.shape == (T, D)
    if weight is not None:
        assert weight.shape == (D,)
    if bias is not None:
        assert bias.shape == (D,)
    # allocate output 分配输出张量
    y = x if out_dtype is None else torch.empty_like(x, dtype=out_dtype)
    # 如果需要输出残差（类型转换或有残差输入），分配残差输出张量
    if residual is not None or (
        residual_dtype is not None and residual_dtype != x.dtype
    ):
        residual_out = torch.empty(T, D, device=x.device, dtype=residual_dtype)
    else:
        residual_out = None
    # LayerNorm 需要均值，RMSNorm 不需要
    mean = (
        torch.empty((T,), dtype=torch.float, device=x.device)
        if not is_rms_norm
        else None
    )
    rstd = torch.empty((T,), dtype=torch.float, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    # 限制分块大小不超过 64KB
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps 根据特征维度选择 kernel

    if D <= 512:
        # 小特征维度：批量处理 BT=32 行
        BT = 32
        layer_norm_gated_fwd_kernel[(cdiv(T, BT),)](
            x=x,
            g=g,
            y=y,
            w=weight,
            b=bias,
            residual=residual,
            residual_out=residual_out,
            mean=mean,
            rstd=rstd,
            eps=eps,
            T=T,
            D=D,
            BD=BD,
            BT=BT,
            ACTIVATION=activation,
            IS_RMS_NORM=is_rms_norm,
            STORE_RESIDUAL_OUT=residual_out is not None,
            HAS_RESIDUAL=residual is not None,
            HAS_WEIGHT=weight is not None,
            HAS_BIAS=bias is not None,
            num_warps=4,
        )
    else:
        # 大特征维度（>512）：逐行处理
        layer_norm_gated_fwd_kernel1[(T,)](
            x=x,
            g=g,
            y=y,
            w=weight,
            b=bias,
            residual=residual,
            residual_out=residual_out,
            mean=mean,
            rstd=rstd,
            eps=eps,
            D=D,
            BD=BD,
            ACTIVATION=activation,
            IS_RMS_NORM=is_rms_norm,
            STORE_RESIDUAL_OUT=residual_out is not None,
            HAS_RESIDUAL=residual is not None,
            HAS_WEIGHT=weight is not None,
            HAS_BIAS=bias is not None,
            num_warps=4,
        )
    # residual_out is None if residual is None and residual_dtype == input_dtype
    # 如果无需单独残差输出，直接返回 x 作为残差
    return y, mean, rstd, residual_out if residual_out is not None else x


# 自定义 autograd Function：封装融合归一化+门控的前向计算
class LayerNormGatedFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        g: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        activation: str,
        residual: torch.Tensor | None = None,
        eps: float = 1e-6,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
        is_rms_norm: bool = False,
    ):
        x_shape_og = x.shape
        g_shape_og = g.shape
        # reshape input data into 2D tensor 将输入展平为 2D [T, D]
        x = x.reshape(-1, x.shape[-1])
        g = g.reshape(-1, g.shape[-1])
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape(-1, residual.shape[-1])
        # 确定残差输出的数据类型
        residual_dtype = (
            residual.dtype
            if residual is not None
            else (torch.float if residual_in_fp32 else None)
        )
        # 执行 Triton kernel 前向计算
        y, mean, rstd, residual_out = layer_norm_gated_fwd(
            x=x,
            g=g,
            weight=weight,
            bias=bias,
            activation=activation,
            eps=eps,
            residual=residual,
            residual_dtype=residual_dtype,
            is_rms_norm=is_rms_norm,
        )
        # 保存反向传播所需的中间变量
        ctx.save_for_backward(residual_out, g, weight, bias, mean, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.g_shape_og = g_shape_og
        ctx.activation = activation
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        # 恢复原始形状
        y = y.reshape(x_shape_og)
        # prenorm 模式同时返回归一化输出和残差输出
        return y if not prenorm else (y, residual_out.reshape(x_shape_og))


# 函数式接口：RMSNorm + 门控激活
def rms_norm_gated(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = "swish",
    residual: torch.Tensor | None = None,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    eps: float = 1e-6,
):
    # 调用 autograd Function，is_rms_norm=True 表示使用 RMSNorm
    return LayerNormGatedFunction.apply(
        x,
        g,
        weight,
        bias,
        activation,
        residual,
        eps,
        prenorm,
        residual_in_fp32,
        True,
    )


# nn.Module 封装：带门控的融合 RMSNorm 模块
class FusedRMSNormGated(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        activation: str = "swish",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps
        self.activation = activation

        # 验证激活函数类型
        if self.activation not in ["swish", "silu", "sigmoid"]:
            raise ValueError(f"Unsupported activation: {self.activation}")

        if elementwise_affine:
            # 可学习缩放参数
            self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
        # 无偏置（RMSNorm 通常不使用偏置）
        self.register_parameter("bias", None)

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        residual: torch.Tensor | None = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> torch.Tensor:
        if _use_cpu:
            # CPU AMX 加速路径：使用 sgl_kernel 的 CPU 融合实现
            assert (
                self.activation == "silu"
            ), "CPU rmsnorm_gated currently only supports activation silu"
            return torch.ops.sgl_kernel.fused_rmsnorm_gated_cpu(
                x, self.weight, g, self.eps
            )
        else:
            # GPU 路径：使用 Triton kernel
            return rms_norm_gated(
                x,
                g,
                self.weight,
                self.bias,
                self.activation,
                residual=residual,
                eps=self.eps,
                prenorm=prenorm,
                residual_in_fp32=residual_in_fp32,
            )
