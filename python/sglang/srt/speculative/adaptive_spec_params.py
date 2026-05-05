"""Adaptive speculative decoding parameters.

Adjusts speculative_num_steps at runtime based on observed acceptance lengths.
"""
# 自适应投机解码参数模块：根据运行时观测到的 token 接受率，动态调整投机步数

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

# 仅在类型检查时导入，避免循环依赖
if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def adaptive_unsupported_reason(server_args: ServerArgs) -> str | None:
    """Return why adaptive spec cannot run under the given server args, or None if supported."""
    # 检查投机算法是否受支持（目前仅 EAGLE / EAGLE3 支持自适应调整）
    if server_args.speculative_algorithm not in ("EAGLE", "EAGLE3"):
        return (
            f"speculative_algorithm={server_args.speculative_algorithm} "
            "(only EAGLE/EAGLE3 are supported)"
        )
    # 仅支持 top-k=1 的情况（树搜索 top-k > 1 时自适应逻辑未实现）
    if server_args.speculative_eagle_topk != 1:
        return (
            f"speculative_eagle_topk={server_args.speculative_eagle_topk} "
            "(only topk=1 is supported)"
        )
    # DP Attention 模式下各 DP rank 的自适应决策未同步，暂不支持
    if server_args.enable_dp_attention:
        return (
            "enable_dp_attention=True is not supported "
            "(adaptive tier decisions are not synchronized across DP ranks)"
        )
    # overlap scheduler（spec v2）尚未实现自适应，仅 v1 支持
    if not server_args.disable_overlap_schedule:
        return (
            "the overlap scheduler (spec v2) is enabled "
            "(adaptive is only implemented for EAGLEWorker v1)"
        )
    # MultiLayerEagle 未实现自适应接口
    if server_args.enable_multi_layer_eagle:
        return (
            "enable_multi_layer_eagle=True is not supported "
            "(MultiLayerEagleWorker does not implement adaptive)"
        )
    # two batch overlap 模式下状态切换会丢弃 TboAttnBackend 包装
    if server_args.enable_two_batch_overlap:
        return (
            "enable_two_batch_overlap=True is not supported "
            "(adaptive state swap would discard the TboAttnBackend wrapper)"
        )
    # pdmux 模式下状态切换不更新 decode_attn_backend_group
    if server_args.enable_pdmux:
        return (
            "enable_pdmux=True is not supported "
            "(adaptive state swap does not update decode_attn_backend_group)"
        )
    return None


def load_adaptive_config(path: str | None) -> dict[str, object]:
    """Load adaptive speculative config from a JSON file.

    The file may contain any subset of the following keys:
        ema_alpha, update_interval, warmup_batches,
        down_hysteresis, up_hysteresis, candidate_steps

    Returns an empty dict when *path* is ``None``.
    """
    # 若未指定配置文件路径，返回空字典（使用默认超参数）
    if path is None:
        return {}
    # 从 JSON 文件加载自适应配置
    with open(path) as f:
        cfg = json.load(f)
    # 配置必须是字典格式
    if not isinstance(cfg, dict):
        raise ValueError(
            "speculative_adaptive_config must be a JSON object, "
            f"got {type(cfg).__name__}"
        )
    return cfg


class AdaptiveSpeculativeParams:
    """Tracks acceptance rate via EMA and adapts num_steps accordingly.

    The core idea: if drafts are consistently accepted, try more steps;
    if drafts are consistently rejected early, reduce steps to avoid waste.

    Formula: target_steps = clamp(round(ema_accept_len) + 1, min_steps, max_steps)
    - Probes one step beyond observed acceptance
    - EMA smoothing prevents oscillation
    - Only updates every `update_interval` batches for stability
    """
    # 通过 EMA（指数移动平均）跟踪接受率，根据 EMA 值在候选步数间切换
    # 核心思想：若草稿 token 持续被接受，则尝试更多步；若频繁被拒，则减少步数

    def __init__(
        self,
        initial_steps: int,
        config: dict[str, object] | None = None,
    ):
        cfg = config or {}
        # TODO: Wider range of candidate_steps (once lazy init is supported).
        # 候选投机步数列表（排序去重），默认 [1, 3, 7]
        self.candidate_steps = sorted(set(cfg.get("candidate_steps", [1, 3, 7])))
        assert (
            len(self.candidate_steps) >= 2
        ), "candidate_steps must have at least 2 distinct values"

        # 最小和最大候选步数，用于边界保护
        self.min_steps = self.candidate_steps[0]
        self.max_steps = self.candidate_steps[-1]
        # EMA 平滑系数：越大则对最新 batch 反应越快，越小则越平稳
        self.ema_alpha = cfg.get("ema_alpha", 0.2)
        # 每隔 update_interval 个 batch 才重新计算一次参数（避免频繁抖动）
        self.update_interval = cfg.get("update_interval", 5)
        # 预热阶段批次数：预热期间不进行参数调整，让 EMA 先稳定
        self.warmup_batches = cfg.get("warmup_batches", 10)
        # 下调迟滞：EMA 必须低于阈值 + down_hysteresis 才触发步数下调
        self.down_hysteresis = cfg.get("down_hysteresis", -0.25)
        # 上调迟滞：EMA 必须高于阈值 + up_hysteresis 才触发步数上调
        self.up_hysteresis = cfg.get("up_hysteresis", 0.0)

        # 从候选步数中选取与 initial_steps 最接近的值作为初始步数
        self.current_steps = min(
            self.candidate_steps,
            key=lambda step: (abs(step - initial_steps), -step),
        )

        # Initialize EMA at current steps - 1 (neutral starting point)
        # EMA 初始值设为 current_steps - 1，代表当前配置下期望的接受长度
        self.ema_accept_len = float(self.current_steps - 1)
        # 已处理的 batch 计数，用于控制预热和更新间隔
        self._batch_count = 0

        logger.info(
            f"AdaptiveSpeculativeParams initialized: "
            f"steps={self.current_steps}, candidate_steps={self.candidate_steps}"
        )

    def update(self, num_accepted_drafts_per_req: list[int]) -> bool:
        """Update EMA with observed accept lengths. Returns True if params changed.

        Args:
            num_accepted_drafts_per_req: Per-request accepted draft token counts from last verify.
        """
        # 若本 batch 无请求，跳过更新
        if not num_accepted_drafts_per_req:
            return False

        # 计算本 batch 的平均接受长度，用于更新 EMA
        batch_avg = sum(num_accepted_drafts_per_req) / len(num_accepted_drafts_per_req)
        # EMA 更新：新值 = (1 - alpha) * 旧EMA + alpha * 当前均值
        self.ema_accept_len = (
            1 - self.ema_alpha
        ) * self.ema_accept_len + self.ema_alpha * batch_avg

        self._batch_count += 1
        # 预热阶段不调整参数，等待 EMA 稳定
        if self._batch_count <= self.warmup_batches:
            return False

        # 仅每隔 update_interval 个 batch 才触发一次参数重计算
        if (self._batch_count - self.warmup_batches) % self.update_interval != 0:
            return False

        return self._recompute_params()

    def _recompute_params(self) -> bool:
        """Recompute steps from EMA. Returns True if params changed."""
        old_steps = self.current_steps
        current_idx = self.candidate_steps.index(old_steps)

        # TODO: Consider limiting step changes to avoid overshooting.
        # 向下扫描：若 EMA 低于下调阈值，则尝试减少步数
        while current_idx > 0:
            prev_step = self.candidate_steps[current_idx - 1]
            # 下调阈值 = 上一步数 - 0.5 + 迟滞量（迟滞使下调更保守）
            drop_threshold = prev_step - 0.5 + self.down_hysteresis
            if self.ema_accept_len <= drop_threshold:
                current_idx -= 1
            else:
                break

        # 向上扫描：若 EMA 高于上调阈值，则尝试增加步数
        while current_idx < len(self.candidate_steps) - 1:
            current_step = self.candidate_steps[current_idx]
            # 上调阈值 = 当前步数 - 0.5 + 迟滞量
            rise_threshold = current_step - 0.5 + self.up_hysteresis
            if self.ema_accept_len > rise_threshold:
                current_idx += 1
            else:
                break

        target = self.candidate_steps[current_idx]

        # 若步数发生变化，更新当前步数并记录日志
        if target != old_steps:
            self.current_steps = target
            logger.info(
                f"Adaptive spec params updated: steps {old_steps} -> {target} "
                f"(ema_accept_len={self.ema_accept_len:.2f})"
            )
            return True
        return False
