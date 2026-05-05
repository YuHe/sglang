from __future__ import annotations

# SSU（Selective State Update）后端分发模块：
# 提供统一 API，在 Triton 和 FlashInfer 两种 SSM 状态更新后端之间分发调用
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


# Mamba SSU 后端抽象基类：定义 selective_state_update 的统一接口
class MambaSSUBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name used for logging."""

    @abstractmethod
    def __call__(
        self,
        state: torch.Tensor,        # SSM 状态张量，(batch, nheads, dim, dstate)
        x: torch.Tensor,            # 当前 decode 步的输入 x
        dt: torch.Tensor,           # 时间步参数 ∆
        A: torch.Tensor,            # 状态转移矩阵对角线 A（负实数）
        B: torch.Tensor,            # SSM 输入矩阵 B
        C: torch.Tensor,            # SSM 输出矩阵 C
        D: torch.Tensor | None = None,       # 跳跃连接权重 D（可选）
        z: torch.Tensor | None = None,       # 门控分支 z（可选）
        dt_bias: torch.Tensor | None = None, # ∆ 偏置（可选）
        dt_softplus: bool = False,           # 是否对 ∆ 应用 softplus
        state_batch_indices: torch.Tensor | None = None,  # 连续批处理中状态的 batch 索引
        pad_slot_id: int = -1,               # 填充槽位 ID（用于跳过无效 batch 条目）
        out: torch.Tensor | None = None,     # 预分配的输出张量（原位更新）
        disable_state_update: bool = False,  # True 时不回写状态（投机解码验证用）
        intermediate_states_buffer: torch.Tensor | None = None,  # 中间状态缓存 buffer（EAGLE 用）
        cache_steps: int | None = None,      # buffer 中缓存的步数总量
        retrieve_parent_token: torch.Tensor | None = None,       # EAGLE tree attention 父 token 索引
        intermediate_state_indices: torch.Tensor | None = None,  # buffer 操作用的自定义索引
    ) -> None: ...


# Triton 后端实现：使用本地 Triton kernel 执行 selective_state_update
class TritonSSUBackend(MambaSSUBackend):
    """Triton-based selective-state-update backend."""

    def __init__(self) -> None:
        # 延迟导入 Triton 实现，避免不必要的 import 开销
        from sglang.srt.layers.attention.mamba.ops.mamba_ssm import (
            selective_state_update,
        )

        self._kernel = selective_state_update

    @property
    def name(self) -> str:
        return "triton"

    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        pad_slot_id: int = -1,
        out: torch.Tensor | None = None,
        disable_state_update: bool = False,
        intermediate_states_buffer: torch.Tensor | None = None,
        cache_steps: int | None = None,
        retrieve_parent_token: torch.Tensor | None = None,
        intermediate_state_indices: torch.Tensor | None = None,
    ) -> None:
        # 直接透传所有参数至 Triton kernel（所有参数均受支持）
        self._kernel(
            state,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            state_batch_indices=state_batch_indices,
            pad_slot_id=pad_slot_id,
            out=out,
            disable_state_update=disable_state_update,
            intermediate_states_buffer=intermediate_states_buffer,
            cache_steps=cache_steps,
            retrieve_parent_token=retrieve_parent_token,
            intermediate_state_indices=intermediate_state_indices,
        )


# FlashInfer 后端实现：使用 FlashInfer 库提供的 CUDA kernel 执行 selective_state_update
class FlashInferSSUBackend(MambaSSUBackend):
    """FlashInfer-based selective-state-update backend."""

    def __init__(self) -> None:
        # 延迟导入 FlashInfer，避免未安装时在 import 阶段报错
        from flashinfer.mamba import selective_state_update

        self._kernel = selective_state_update

    @property
    def name(self) -> str:
        return "flashinfer"

    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        pad_slot_id: int = -1,
        out: torch.Tensor | None = None,
        disable_state_update: bool = False,
        intermediate_states_buffer: torch.Tensor | None = None,
        cache_steps: int | None = None,
        retrieve_parent_token: torch.Tensor | None = None,
        intermediate_state_indices: torch.Tensor | None = None,
    ) -> None:
        # FlashInfer 不支持 EAGLE tree attention 的父 token 索引功能
        if retrieve_parent_token is not None:
            raise ValueError(
                "FlashInfer backend does not support retrieve_parent_token. "
                "Use --mamba-backend triton for EAGLE tree attention."
            )
        # FlashInfer expects cache_steps as an int (0 when unused).
        # FlashInfer 要求 cache_steps 为整数（不接受 None），未使用时传 0
        self._kernel(
            state,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            state_batch_indices=state_batch_indices,
            pad_slot_id=pad_slot_id,
            out=out,
            disable_state_update=disable_state_update,
            intermediate_states_buffer=intermediate_states_buffer,
            cache_steps=0 if cache_steps is None else cache_steps,  # None 转 0
            intermediate_state_indices=intermediate_state_indices,
        )


# 后端注册表：将后端名称映射到对应的类（可扩展新后端）
_BACKEND_REGISTRY: dict[str, type[MambaSSUBackend]] = {
    "triton": TritonSSUBackend,
    "flashinfer": FlashInferSSUBackend,
}

# 全局后端单例（在 initialize_mamba_selective_state_update_backend 中初始化）
_mamba_ssu_backend: MambaSSUBackend | None = None


# 初始化函数：根据 server_args 配置选择并实例化后端（在 scheduler 启动时调用一次）
def initialize_mamba_selective_state_update_backend(server_args: ServerArgs) -> None:
    """Instantiate the selective-state-update backend from server config.

    This should be called once during scheduler initialization.

    Args:
        server_args: Server arguments containing ``mamba_backend`` setting.

    Raises:
        ValueError: If the requested backend is unavailable or cannot be imported.
    """
    global _mamba_ssu_backend

    # 从 server_args 读取后端名称，默认使用 "triton"
    requested = server_args.mamba_backend or "triton"

    backend_cls = _BACKEND_REGISTRY.get(requested)
    if backend_cls is None:
        raise ValueError(
            f"Unknown mamba backend '{requested}'. "
            f"Available backends: {list(_BACKEND_REGISTRY.keys())}"
        )

    try:
        # 实例化后端（此时触发延迟 import，若依赖未安装则抛出 ImportError）
        _mamba_ssu_backend = backend_cls()
    except ImportError:
        raise ValueError(
            f"Mamba backend '{requested}' requested but its dependencies are not "
            f"available. Install the required package or use a different "
            f"--mamba-backend value."
        )

    logger.debug(
        "Mamba selective_state_update backend initialized: %s",
        _mamba_ssu_backend.name,
    )


# 公开 API：将 selective_state_update 调用分发到当前配置的后端
def selective_state_update(
    state: torch.Tensor,        # SSM 状态张量，(batch, nheads, dim, dstate)
    x: torch.Tensor,            # decode 步输入
    dt: torch.Tensor,           # 时间步 ∆
    A: torch.Tensor,            # 状态转移矩阵对角线 A
    B: torch.Tensor,            # SSM 输入矩阵 B
    C: torch.Tensor,            # SSM 输出矩阵 C
    D: torch.Tensor | None = None,       # 跳跃连接权重 D
    z: torch.Tensor | None = None,       # 门控分支 z
    dt_bias: torch.Tensor | None = None, # ∆ 偏置
    dt_softplus: bool = False,           # 是否对 ∆ 应用 softplus
    state_batch_indices: torch.Tensor | None = None,  # 连续批处理 batch 索引
    pad_slot_id: int = -1,               # 填充槽位 ID
    out: torch.Tensor | None = None,     # 预分配输出张量
    disable_state_update: bool = False,  # True 时不回写状态
    intermediate_states_buffer: torch.Tensor | None = None,  # 中间状态 buffer
    cache_steps: int | None = None,      # buffer 缓存步数
    retrieve_parent_token: torch.Tensor | None = None,       # EAGLE 父 token 索引
    intermediate_state_indices: torch.Tensor | None = None,  # buffer 操作自定义索引
) -> None:
    """Dispatch selective-state-update to the configured backend.

    This function provides a unified interface regardless of the underlying
    backend. Backend-specific argument adaptation is handled inside each
    :class:`MambaSSUBackend` subclass.

    Args:
        state: SSM state tensor (batch, nheads, dim, dstate)
        x: Input tensor
        dt: Delta time tensor
        A: A matrix
        B: B matrix
        C: C matrix
        D: Optional D vector
        z: Optional z tensor for gating
        dt_bias: Optional dt bias
        dt_softplus: Whether to apply softplus to dt
        state_batch_indices: Optional batch indices for state
        out: Preallocated output tensor (in-place updated)
        disable_state_update: If True, don't write back to state (for speculative verify)
        intermediate_states_buffer: Buffer to cache intermediate states
        cache_steps: Total number of steps in the buffer
        retrieve_parent_token: (batch, T) tensor of parent token indices for EAGLE tree attention
        intermediate_state_indices: (batch,) tensor of indices for intermediate_states_buffer operations.
            If provided, uses these indices instead of state_batch_indices for the buffer.
    """
    # 确保后端已初始化（必须先调用 initialize_mamba_selective_state_update_backend）
    assert _mamba_ssu_backend is not None, (
        "Mamba selective_state_update backend not initialized. "
        "Call initialize_mamba_selective_state_update_backend() first."
    )

    # 委托给当前激活的后端实例执行
    _mamba_ssu_backend(
        state,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        state_batch_indices=state_batch_indices,
        pad_slot_id=pad_slot_id,
        out=out,
        disable_state_update=disable_state_update,
        intermediate_states_buffer=intermediate_states_buffer,
        cache_steps=cache_steps,
        retrieve_parent_token=retrieve_parent_token,
        intermediate_state_indices=intermediate_state_indices,
    )
