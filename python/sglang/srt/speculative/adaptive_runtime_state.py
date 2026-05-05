import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

# 导入自适应投机解码参数类及配置加载函数
from sglang.srt.speculative.adaptive_spec_params import (
    AdaptiveSpeculativeParams,
    load_adaptive_config,
)

# 仅在类型检查阶段导入，避免循环依赖及运行时不必要的导入
if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
    from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
        EAGLEDraftExtendCudaGraphRunner,
    )

logger = logging.getLogger(__name__)


@dataclass
class SpecRuntimeState:
    """A complete set of runtime resources bound to a specific speculative
    decoding configuration.

    Each decode round runs three stages — draft, verify, extend — and every
    stage has shape-dependent resources (attention backends and CUDA graphs)
    that must match the current configuration.  Switching adaptive steps
    means swapping the entire state atomically.
    """
    # 投机解码运行时状态：封装一套完整的运行时资源，与特定投机步数绑定
    # 每次解码轮次包含三个阶段：草稿生成、验证、扩展
    # 切换自适应步数时需原子性地交换整个状态

    # -- Configuration (determines shapes for all stages) --
    # 当前配置的投机步数（决定所有阶段的张量形状）
    speculative_num_steps: int
    # 草稿 token 总数 = speculative_num_steps + 1（含初始验证 token）
    speculative_num_draft_tokens: int

    # -- Draft stage: draft model multi-step autoregressive generation --
    # 草稿阶段的注意力后端（草稿模型多步自回归生成）
    draft_attn_backend: "AttentionBackend | None"
    # 草稿阶段的 CUDA Graph runner（加速草稿模型推理）
    cuda_graph_runner: "EAGLEDraftCudaGraphRunner | None"

    # -- Verify stage: target model one-pass tree verification --
    # 验证阶段的注意力后端（目标模型一次性树验证）
    target_attn_backend: "AttentionBackend"
    # 验证阶段的 CUDA Graph runner（CPU Graph 或 GPU Graph）
    target_graph_runner: "CudaGraphRunner | CPUGraphRunner | None"

    # -- Extend stage: draft model KV cache catch-up after verify --
    # 扩展阶段的注意力后端（验证后草稿模型的 KV 缓存追赶）
    draft_extend_attn_backend: "AttentionBackend | None"
    # 扩展阶段的 CUDA Graph runner
    cuda_graph_runner_for_draft_extend: "EAGLEDraftExtendCudaGraphRunner | None"


# Worker 协议：定义支持自适应控制器所需的最小接口
class AdaptiveSpecWorker(Protocol):
    """Protocol that a worker must implement to use AdaptiveController."""

    # 当前投机步数，控制器据此进行状态查找和切换
    speculative_num_steps: int

    def build_adaptive_runtime_state(
        self, speculative_num_steps: int, speculative_num_draft_tokens: int
    ) -> SpecRuntimeState: ...

    def apply_runtime_state(self, state: SpecRuntimeState) -> None: ...


class AdaptiveController:
    """Facade that owns adaptive decision-making and runtime state switching.

    Works with any worker that implements ``AdaptiveSpecWorker`` protocol:
      - ``build_adaptive_runtime_state(steps, draft_tokens)`` → runtime state
      - ``apply_runtime_state(state)`` → apply it to the worker

    The worker only needs to:
      1. Call ``register()`` for the initial state, then ``init_states()``
         once during startup.
      2. Call ``on_verify_complete(num_accepted_drafts_per_req)`` after each decode verify.
    """
    # 自适应控制器：负责根据接受率 EMA 动态调整投机步数，并原子切换运行时状态

    def __init__(self, worker: AdaptiveSpecWorker, config_path: str | None = None):
        # 持有对 Worker 的引用，用于调用构建和应用状态的接口
        self.worker = worker
        # 加载自适应配置（步数候选集、EMA 阈值等）
        cfg = load_adaptive_config(config_path)
        # 初始化自适应参数对象，用于跟踪 EMA 并决定是否切换步数
        self.params = AdaptiveSpeculativeParams(
            initial_steps=worker.speculative_num_steps,
            config=cfg,
        )
        # 以投机步数为键，缓存各步数对应的运行时状态
        self._states: dict[int, SpecRuntimeState] = {}

    @property
    def candidate_steps(self) -> list[int]:
        # 返回所有候选投机步数列表（由配置决定）
        return self.params.candidate_steps

    def register(self, state: SpecRuntimeState, steps: int | None = None) -> None:
        """Register a pre-built runtime state.

        *steps* defaults to ``state.speculative_num_steps`` when not given.
        """
        # 将预构建的运行时状态注册到内部缓存，支持外部手动注册初始状态
        key = steps if steps is not None else state.speculative_num_steps
        self._states[key] = state

    def init_states(self) -> None:
        """Build and register runtime states for all candidate steps."""
        # 为所有候选步数构建并缓存对应的运行时状态（已存在则跳过）
        for steps in self.params.candidate_steps:
            if steps in self._states:
                continue
            state = self.worker.build_adaptive_runtime_state(
                speculative_num_steps=steps,
                # 草稿 token 数 = 步数 + 1
                speculative_num_draft_tokens=steps + 1,
            )
            self._states[steps] = state
        # 初始化完成后立即激活当前步数对应的状态
        self._activate(self.params.current_steps)

    def on_verify_complete(self, num_accepted_drafts_per_req: list[int]) -> None:
        """Feed verify results; switch runtime state if EMA warrants it."""
        # 每次验证完成后更新 EMA，如果 EMA 指示需要切换步数则触发状态切换
        if self.params.update(num_accepted_drafts_per_req):
            self._activate(self.params.current_steps)

    def _activate(self, speculative_num_steps: int) -> None:
        # 从缓存中取出对应步数的运行时状态
        state = self._states.get(speculative_num_steps)
        if state is None:
            raise ValueError(
                f"Missing adaptive runtime state for steps={speculative_num_steps}"
            )
        # 将选中的运行时状态应用到 Worker（原子切换注意力后端和 CUDA Graph）
        self.worker.apply_runtime_state(state)
