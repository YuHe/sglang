from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, IntEnum, auto
from typing import TYPE_CHECKING, List, Optional, Tuple, Type, Union

# 仅在类型检查阶段导入，避免循环依赖
if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
    from sglang.srt.speculative.ngram_worker import NGRAMWorker


# 投机解码算法枚举：标识当前使用的投机解码策略
class SpeculativeAlgorithm(Enum):
    """Enumeration of speculative decoding algorithms."""

    DFLASH = auto()     # DFlash: 基于草稿 Flash Attention 的投机解码
    EAGLE = auto()      # EAGLE: 基于轻量级草稿模型的树形投机解码
    EAGLE3 = auto()     # EAGLE3: EAGLE 的改进版本（三阶段训练）
    STANDALONE = auto() # STANDALONE: 独立草稿模型（不共享 embeddings/lm_head）
    NGRAM = auto()      # NGRAM: 基于 N-gram 匹配的无模型投机解码
    NONE = auto()       # NONE: 不使用投机解码（普通自回归）

    @classmethod
    def from_string(cls, name: Optional[str]) -> SpeculativeAlgorithm:
        # 从字符串名称解析枚举值（大小写不敏感）
        if name is None:
            return cls.NONE
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"Unknown speculative algorithm name: {name}")

    def is_none(self) -> bool:
        # 是否为非投机解码模式
        return self == SpeculativeAlgorithm.NONE

    def is_speculative(self) -> bool:
        # 是否启用了投机解码
        return self != SpeculativeAlgorithm.NONE

    def is_eagle(self) -> bool:
        # NOTE: EAGLE3 is a variant of EAGLE
        # EAGLE3 是 EAGLE 的变体，共享大部分逻辑
        return self == SpeculativeAlgorithm.EAGLE or self == SpeculativeAlgorithm.EAGLE3

    def is_eagle3(self) -> bool:
        return self == SpeculativeAlgorithm.EAGLE3

    def is_dflash(self) -> bool:
        return self == SpeculativeAlgorithm.DFLASH

    def is_standalone(self) -> bool:
        return self == SpeculativeAlgorithm.STANDALONE

    def is_ngram(self) -> bool:
        return self == SpeculativeAlgorithm.NGRAM

    def supports_spec_v2(self) -> bool:
        # 是否支持 spec v2（overlap scheduler）：目前仅 EAGLE 和 STANDALONE 支持
        return self.is_eagle() or self.is_standalone()

    def create_worker(
        self, server_args: ServerArgs
    ) -> Optional[Union[Type[BaseSpecWorker], Type[TpModelWorker], Type[NGRAMWorker]]]:
        # 根据算法类型和配置，返回对应的 Worker 类（由调用方实例化）
        assert (
            not self.is_none()
        ), "Cannot create worker for NONE speculative algorithm."

        # 是否启用 overlap scheduler（spec v2）
        enable_overlap = not server_args.disable_overlap_schedule

        if self.is_dflash():
            # DFlash 不支持 overlap scheduler
            if enable_overlap:
                raise ValueError(
                    "DFLASH does not support overlap scheduling (spec v2)."
                )
            from sglang.srt.speculative.dflash_worker import DFlashWorker

            return DFlashWorker

        if self.is_eagle() and server_args.enable_multi_layer_eagle:
            # FIXME: migrate to EagleWorker
            # 多层 EAGLE：支持 overlap scheduler 时使用 V2 版本
            if enable_overlap:
                from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
                    MultiLayerEagleWorkerV2,
                )

                return MultiLayerEagleWorkerV2

            from sglang.srt.speculative.multi_layer_eagle_worker import (
                MultiLayerEagleWorker,
            )

            return MultiLayerEagleWorker

        elif self.is_eagle():
            # 标准 EAGLE：支持 overlap scheduler 时使用 EAGLEWorkerV2
            if enable_overlap:
                from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

                return EAGLEWorkerV2

            from sglang.srt.speculative.eagle_worker import EAGLEWorker

            return EAGLEWorker
        elif self.is_standalone():
            # STANDALONE：支持 overlap scheduler 时使用 StandaloneWorkerV2
            if enable_overlap:
                from sglang.srt.speculative.standalone_worker_v2 import (
                    StandaloneWorkerV2,
                )

                return StandaloneWorkerV2

            from sglang.srt.speculative.standalone_worker import StandaloneWorker

            return StandaloneWorker
        elif self.is_ngram():
            # NGRAM 不支持 overlap scheduler
            if enable_overlap:
                raise ValueError(
                    f"Speculative algorithm {self.name} does not support overlap worker creation."
                )

            from sglang.srt.speculative.ngram_worker import NGRAMWorker

            return NGRAMWorker

        raise ValueError("Unreachable code path in create_worker.")


# 投机输入类型枚举：区分不同算法的草稿输入和验证输入
class SpecInputType(IntEnum):
    # NOTE: introduce this to distinguish the SpecInput types of multiple algorithms when asserting in attention backends.
    # If all algorithms can share the same datastrucutre of draft_input and verify_input, consider simplify it
    # 用于注意力后端中断言检查，区分不同算法的输入类型
    EAGLE_DRAFT = auto()    # EAGLE 草稿生成阶段的输入
    EAGLE_VERIFY = auto()   # EAGLE 验证阶段的输入
    DFLASH_DRAFT = auto()   # DFlash 草稿生成阶段的输入
    DFLASH_VERIFY = auto()  # DFlash 验证阶段的输入
    NGRAM_VERIFY = auto()   # NGRAM 验证阶段的输入（无草稿生成阶段）


# 投机输入抽象基类：所有草稿/验证输入类的公共接口
class SpecInput(ABC):
    def __init__(self, spec_input_type: SpecInputType):
        # 保存输入类型，便于后续断言和路由
        self.spec_input_type = spec_input_type

    def is_draft_input(self) -> bool:
        # FIXME: remove this function which is only used for assertion
        # or use another variable name like `draft_input` to substitute `spec_info`
        # 判断是否为草稿生成阶段的输入（EAGLE 或 DFlash 草稿）
        return self.spec_input_type in {
            SpecInputType.EAGLE_DRAFT,
            SpecInputType.DFLASH_DRAFT,
        }

    def is_verify_input(self) -> bool:
        # 判断是否为验证阶段的输入（EAGLE、DFlash 或 NGRAM 验证）
        return self.spec_input_type in {
            SpecInputType.EAGLE_VERIFY,
            SpecInputType.DFLASH_VERIFY,
            SpecInputType.NGRAM_VERIFY,
        }

    @abstractmethod
    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        # 返回 (c1, c2)：投机解码对 global_num_tokens 和 global_num_tokens_for_logprob 的倍数系数
        # c1 用于调整总 token 数（草稿 token 数 = seq_len * c1）
        # c2 用于调整 logprob 计算的 token 数
        pass

    def get_spec_adjusted_global_num_tokens(
        self, forward_batch: ModelWorkerBatch
    ) -> Tuple[List[int], List[int]]:
        # 根据投机解码的倍数系数，调整各 TP rank 的 global_num_tokens
        # 用于 TP 负载均衡和 logprob 计算的 token 数统计
        c1, c2 = self.get_spec_adjust_token_coefficient()
        global_num_tokens = [x * c1 for x in forward_batch.global_num_tokens]
        global_num_tokens_for_logprob = [
            x * c2 for x in forward_batch.global_num_tokens_for_logprob
        ]
        return global_num_tokens, global_num_tokens_for_logprob
