from __future__ import annotations

# 导入扩散语言模型配置
from sglang.srt.dllm.config import DllmConfig
# 导入前向批次信息模块
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
# 导入通用工具函数：向上对齐和固定内存检查
from sglang.srt.utils.common import ceil_align, is_pin_memory_available

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Store information about requests and batches.

The following is the flow of data structures for a batch:

ScheduleBatch -> ModelWorkerBatch -> ForwardBatch

- ScheduleBatch is managed by `scheduler.py::Scheduler`.
  It contains high-level scheduling data. Most of the data is on the CPU.
- ModelWorkerBatch is managed by `tp_worker.py::TpModelWorker`.
  It is a subset of `ScheduleBatch` that only contains data related to the model forward on GPU.
  It will be transformed from CPU scheduler to GPU model runner.
- ForwardBatch is managed by `model_runner.py::ModelRunner`.
  It contains low-level tensor data. Most of the data consists of GPU tensors.

TODO(lmzheng): ModelWorkerBatch seems a bit redundant and we consider removing it in the future.
"""

import copy
import dataclasses
import logging
import re
from concurrent.futures import Future
from enum import Enum, auto
from functools import lru_cache
from http import HTTPStatus
from itertools import chain
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.disaggregation.base import BaseKVSender
from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
    ScheduleBatchDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_rank
from sglang.srt.dllm.mixin.req import ReqDllmMixin
from sglang.srt.environ import envs
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.managers.embed_types import PositionalEmbeds
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchPrefixParams
from sglang.srt.mem_cache.common import (
    alloc_for_decode,
    alloc_for_extend,
    evict_from_tree_cache,
    release_kv_cache,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.observability.metrics_collector import (
    DPCooperationInfo,
    SchedulerMetricsCollector,
)
from sglang.srt.observability.req_time_stats import (
    APIServerReqTimeStats,
    DPControllerReqTimeStats,
    SchedulerReqTimeStats,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, get_global_server_args
from sglang.srt.utils import flatten_nested_list
from sglang.srt.utils.cuda_ipc_transport_utils import CudaIpcTensorTransportProxy

if TYPE_CHECKING:
    from typing import Any, Dict

    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from sglang.srt.observability.scheduler_metrics_mixin import PrefillStats
    from sglang.srt.session.session_controller import Session
    from sglang.srt.speculative.eagle_info import EagleDraftInput
    from sglang.srt.speculative.spec_info import SpecInput, SpeculativeAlgorithm

# 增量解码初始偏移量
INIT_INCREMENTAL_DETOKENIZATION_OFFSET = 5

# 多模态填充值基础偏移量，确保 pad_value 不与有效文本 token ID 冲突
# Constant used as the base offset for MM (multimodal) pad values.
# This ensures pad_values don't overlap with valid text token IDs.
MM_PAD_SHIFT_VALUE = 1_000_000

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def sanity_check_mm_pad_shift_value(vocab_size: int) -> None:
    # 检查词表大小是否超过 MM_PAD_SHIFT_VALUE，避免 pad_value 与有效 token ID 重叠
    if vocab_size > MM_PAD_SHIFT_VALUE:
        raise ValueError(
            f"Model vocab_size ({vocab_size}) exceeds MM_PAD_SHIFT_VALUE ({MM_PAD_SHIFT_VALUE}). "
            f"MM pad_values may overlap with valid token IDs. "
            f"Please increase MM_PAD_SHIFT_VALUE in schedule_batch.py."
        )


def _compute_pad_value(hash: int) -> int:
    """Compute pad value from hash."""
    # 根据哈希值计算多模态填充 token ID
    return MM_PAD_SHIFT_VALUE + (hash % (1 << 30))


# 请求完成原因基类
class BaseFinishReason:
    def __init__(self, is_error: bool = False):
        self.is_error = is_error

    def to_json(self):
        raise NotImplementedError()


# 匹配到停止 token 时的完成原因
class FINISH_MATCHED_TOKEN(BaseFinishReason):
    def __init__(self, matched: Union[int, List[int]]):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


# 匹配到停止字符串时的完成原因
class FINISH_MATCHED_STR(BaseFinishReason):
    def __init__(self, matched: str):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


# 匹配到停止正则时的完成原因
class FINISHED_MATCHED_REGEX(BaseFinishReason):
    def __init__(self, matched: str):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


# 达到最大长度限制时的完成原因
class FINISH_LENGTH(BaseFinishReason):
    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def to_json(self):
        return {
            "type": "length",  # to match OpenAI API's return value
            "length": self.length,
        }


# 请求被中止时的完成原因（is_error=True）
class FINISH_ABORT(BaseFinishReason):
    def __init__(self, message=None, status_code=None, err_type=None):
        super().__init__(is_error=True)
        self.message = message or "Aborted"
        self.status_code = status_code
        self.err_type = err_type

    def to_json(self):
        return {
            "type": "abort",
            "message": self.message,
            "status_code": self.status_code,
            "err_type": self.err_type,
        }


# 多模态输入的模态类型枚举
class Modality(Enum):
    IMAGE = auto()
    VIDEO = auto()
    AUDIO = auto()

    @staticmethod
    def from_str(modality_str: str):
        # 从字符串解析模态类型
        try:
            return Modality[modality_str.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid modality string: {modality_str}. Valid modalities are: {[m.name for m in Modality]}"
            )

    @staticmethod
    def all():
        # 返回所有模态类型列表
        return [Modality.IMAGE, Modality.VIDEO, Modality.AUDIO]


# 多模态输入格式枚举
class MultimodalInputFormat(Enum):
    NORMAL = auto()  # 普通格式
    PROCESSOR_OUTPUT = auto()  # 处理器原始输出格式
    PRECOMPUTED_EMBEDDING = auto()  # 预计算嵌入格式


@dataclasses.dataclass
class MultimodalDataItem:
    """
    One MultimodalDataItem represents a single multimodal input (one image, one video, or one audio).
    For example, if there are 3 images and 1 audio, there will be 4 MultimodalDataItems.

    Each item has its own hash and pad_value, enabling per-image RadixAttention caching.

    We put the common fields first and the model-specific fields in model_specific_data.
    """
    # 单个多模态输入数据项（一张图/一段视频/一段音频）

    modality: Modality
    hash: int = None  # 内容哈希，用于 RadixAttention 缓存命中
    pad_value: int = None  # 填充 token ID，确保不与文本 token 冲突
    offsets: Optional[list] = None  # 在 token 序列中的偏移位置

    format: MultimodalInputFormat = MultimodalInputFormat.NORMAL

    # the raw features returned by processor, e.g. pixel_values or audio_features
    feature: Union[torch.Tensor, np.ndarray] = None  # 处理器返回的原始特征张量
    # the precomputed embeddings, passed as final encoder embeddings
    # One and only one of the feature and precomputed_embeddings will be empty
    precomputed_embeddings: Optional[Union[torch.Tensor, np.ndarray]] = None  # 预计算的编码器嵌入

    # Model-specific data stored in a dictionary
    model_specific_data: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __getattr__(self, name: str):
        # 从 model_specific_data 中查找模型特定属性
        if (
            "model_specific_data" in self.__dict__
            and name in self.__dict__["model_specific_data"]
        ):
            return self.__dict__["model_specific_data"][name]
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    def __setitem__(self, key: str, value: Any):
        # 优先设置标准字段，否则存入 model_specific_data
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self.model_specific_data[key] = value

    def set(self, key: str, value: Any):
        # 设置字段值的便捷方法
        self.__setitem__(key, value)

    @staticmethod
    def is_empty_list(l):
        # 判断列表是否为空（None 或所有元素均为 None）
        if l is None:
            return True
        return len([item for item in flatten_nested_list(l) if item is not None]) == 0

    def set_pad_value(self):
        """
        Set the pad value after first hashing the data
        """
        # 已设置过 pad_value 则直接返回
        if self.pad_value is not None:
            return

        from sglang.srt.managers.mm_utils import hash_feature

        if envs.SGLANG_MM_SKIP_COMPUTE_HASH.get():
            import uuid

            # 跳过哈希计算时使用随机 UUID 作为哈希值
            self.hash = uuid.uuid4().int
            self.pad_value = _compute_pad_value(self.hash)
            return
        if self.hash is None:
            # 选择 feature 或 precomputed_embeddings 进行哈希
            if self.feature is not None:
                hashed_feature = self.feature
            else:
                hashed_feature = self.precomputed_embeddings
            self.hash = hash_feature(hashed_feature)
        assert self.hash is not None
        # 根据哈希计算 pad_value
        self.pad_value = _compute_pad_value(self.hash)

    def is_modality(self, modality: Modality) -> bool:
        # 判断是否为指定模态
        return self.modality == modality

    def is_audio(self):
        return self.modality == Modality.AUDIO

    def is_image(self):
        return self.modality == Modality.IMAGE

    def is_video(self):
        return self.modality == Modality.VIDEO

    def is_valid(self) -> bool:
        # 判断该多模态数据项是否有效（图像、视频或音频之一）
        return self.is_image() or self.is_video() or self.is_audio()

    def validate(self):
        ...
        # TODO

    def is_precomputed_embedding(self):
        # 判断是否为预计算嵌入格式
        return self.format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING

    @staticmethod
    def from_dict(obj: dict):
        # 从字典构造 MultimodalDataItem
        kwargs = dict(obj)
        modality = kwargs.pop("modality")
        if isinstance(modality, str):
            modality = Modality[modality]
        ret = MultimodalDataItem(modality=modality, **kwargs)
        ret.validate()
        return ret

    def reconstruct(self):
        # 将 CudaIpcTensorTransportProxy 还原为实际 GPU 张量
        if not isinstance(self.feature, CudaIpcTensorTransportProxy):
            return

        reconstruct_device = torch.cuda.current_device()
        if isinstance(self.feature, CudaIpcTensorTransportProxy):
            self.feature = self.feature.reconstruct_on_target_device(reconstruct_device)
        if isinstance(self.precomputed_embeddings, CudaIpcTensorTransportProxy):
            self.precomputed_embeddings = (
                self.precomputed_embeddings.reconstruct_on_target_device(
                    reconstruct_device
                )
            )
        for extra_key in self.model_specific_data:
            if isinstance(
                self.model_specific_data[extra_key], CudaIpcTensorTransportProxy
            ):
                extra_data = self.model_specific_data[
                    extra_key
                ].reconstruct_on_target_device(reconstruct_device)
                self.model_specific_data[extra_key] = extra_data


@dataclasses.dataclass
class MultimodalProcessorOutput:
    """Raw output from multimodal processors, before pad/hash computation.

    This is the typed replacement for the dict previously returned by
    ``BaseMultimodalProcessor.process_mm_data_async``.  Unlike
    ``MultimodalInputs``, items here do NOT carry pad_value or hash yet.
    """
    # 多模态处理器的原始输出，尚未计算 pad/hash

    mm_items: List[MultimodalDataItem]
    input_ids: Optional[List[int]] = None  # 可选的输入 token ID 序列

    # image（图像相关 token ID）
    im_token_id: Optional[int] = None
    im_start_id: Optional[int] = None
    im_end_id: Optional[int] = None
    slice_start_id: Optional[int] = None
    slice_end_id: Optional[int] = None

    # video（视频相关 token ID）
    video_token_id: Optional[int] = None

    # audio（音频相关 token ID）
    audio_token_id: Optional[int] = None
    audio_start_id: Optional[int] = None
    audio_end_id: Optional[int] = None

    # QWen2-VL related（QWen2-VL 多尺度旋转位置编码）
    mrope_positions: Optional[torch.Tensor] = None
    mrope_position_delta: Optional[torch.Tensor] = None

    # Moss-VL related（Moss-VL 相关字段）
    vision_position_ids: Optional[torch.Tensor] = None
    media_nums_per_sample: Optional[List[int]] = None
    visible_frame_counts: Optional[torch.Tensor] = None

    # for transformers-compatibility
    token_type_ids: Optional[torch.Tensor] = None

    @staticmethod
    def from_dict(d: dict) -> "MultimodalProcessorOutput":
        # 从字典构造 MultimodalProcessorOutput
        return MultimodalProcessorOutput(
            mm_items=d["mm_items"],
            input_ids=d.get("input_ids"),
            im_token_id=d.get("im_token_id"),
            im_start_id=d.get("im_start_id"),
            im_end_id=d.get("im_end_id"),
            slice_start_id=d.get("slice_start_id"),
            slice_end_id=d.get("slice_end_id"),
            video_token_id=d.get("video_token_id"),
            audio_token_id=d.get("audio_token_id"),
            audio_start_id=d.get("audio_start_id"),
            audio_end_id=d.get("audio_end_id"),
            mrope_positions=d.get("mrope_positions"),
            mrope_position_delta=d.get("mrope_position_delta"),
            vision_position_ids=d.get("vision_position_ids"),
            media_nums_per_sample=d.get("media_nums_per_sample"),
            visible_frame_counts=d.get("visible_frame_counts"),
        )


@dataclasses.dataclass
class MultimodalInputs:
    """The multimodal data related inputs."""
    # 多模态输入数据集合，包含所有多模态数据项和相关辅助字段

    # items of data
    mm_items: List[MultimodalDataItem]
    image_pad_len: Optional[list] = None  # 每张图像的填充长度列表
    num_image_tokens: Optional[int] = None  # 图像总 token 数

    # image（图像相关特殊 token ID）
    im_token_id: Optional[int] = None
    im_start_id: Optional[int] = None
    im_end_id: Optional[int] = None
    slice_start_id: Optional[int] = None
    slice_end_id: Optional[int] = None

    # video（视频相关特殊 token ID）
    video_token_id: Optional[int] = None

    # audio（音频相关特殊 token ID）
    audio_token_id: Optional[int] = None
    audio_start_id: Optional[int] = None
    audio_end_id: Optional[int] = None

    # QWen2-VL related（QWen2-VL 多尺度旋转位置编码）
    mrope_positions: Optional[torch.Tensor] = None
    mrope_position_delta: Optional[torch.Tensor] = None
    mrope_position_delta_repeated_cache: Optional[torch.Tensor] = None  # 缓存的重复位置增量

    # Moss-VL related（Moss-VL 相关字段）
    vision_position_ids: Optional[torch.Tensor] = None
    media_nums_per_sample: Optional[List[int]] = None
    visible_frame_counts: Optional[torch.Tensor] = None

    def release_features(self):
        """Release feature tensors to free GPU memory."""
        # 释放所有多模态数据项的 feature 张量，减少 GPU 内存占用
        for item in self.mm_items:
            item.feature = None

    @staticmethod
    def from_processor_output(obj: "MultimodalProcessorOutput"):
        # 从处理器输出构造 MultimodalInputs，同时计算 pad_value 和哈希
        mm_items = obj.mm_items
        for mm_item in mm_items:
            mm_item.reconstruct()

        ret = MultimodalInputs(
            mm_items=mm_items,
        )

        assert isinstance(ret.mm_items, list)
        # 过滤掉无效的多模态数据项
        ret.mm_items = [item for item in ret.mm_items if item.is_valid()]

        if envs.SGLANG_MM_BUFFER_SIZE_MB.get() > 0:
            # Multi-modal feature hashing optimization:
            # When SGLANG_MM_BUFFER_SIZE_MB > 0, we temporarily move feature tensors to GPU
            # for faster hash computation, while avoiding OOM issues.
            # 多模态特征哈希优化：将特征临时移到 GPU 加速哈希计算
            from sglang.srt.managers.mm_utils import (
                init_feature_buffer,
                is_feature_buffer_initialized,
                reset_buffer_offset,
                try_add_to_buffer,
            )

            device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            if not is_feature_buffer_initialized():
                init_feature_buffer(device)
            reset_buffer_offset()
            for item in ret.mm_items:
                if item.feature is not None:
                    if isinstance(item.feature, torch.Tensor):
                        item.feature = try_add_to_buffer(item.feature)

        # 为每个多模态数据项计算 pad_value
        for item in ret.mm_items:
            item.set_pad_value()

        if envs.SGLANG_MM_BUFFER_SIZE_MB.get() > 0:
            # 将特征移回 CPU（非阻塞传输）
            for item in ret.mm_items:
                if item.feature is not None:
                    item.feature = item.feature.to("cpu", non_blocking=True)

        # 从处理器输出中复制可选字段
        optional_args = [
            "mrope_positions",
            "mrope_position_delta",
            "im_token_id",
            "im_start_id",
            "im_end_id",
            "video_token_id",
            "slice_start_id",
            "slice_end_id",
            "audio_start_id",
            "audio_end_id",
            "audio_token_id",
            "vision_position_ids",
            "media_nums_per_sample",
            "visible_frame_counts",
        ]
        for arg in optional_args:
            val = getattr(obj, arg, None)
            if val is not None:
                setattr(ret, arg, val)

        return ret

    def contains_image_inputs(self) -> bool:
        # 判断是否包含图像输入
        return any(item.is_image() for item in self.mm_items)

    def contains_video_inputs(self) -> bool:
        # 判断是否包含视频输入
        return any(item.is_video() for item in self.mm_items)

    def contains_audio_inputs(self) -> bool:
        # 判断是否包含音频输入
        return any(item.is_audio() for item in self.mm_items)

    def contains_mm_input(self) -> bool:
        # 判断是否包含任意有效的多模态输入
        return any(True for item in self.mm_items if item.is_valid())

    def merge(self, other: MultimodalInputs):
        """
        merge image inputs when requests are being merged
        """
        # 合并多模态输入（用于请求合并场景）

        # args needed to be merged
        optional_args = [
            "mm_items",
            "image_pad_len",
        ]
        for arg in optional_args:
            self_arg = getattr(self, arg, None)
            if self_arg is not None:
                setattr(self, arg, self_arg + getattr(other, arg))

        # 合并 mrope_positions（沿序列维度拼接）
        mrope_positions = self.mrope_positions
        if mrope_positions is not None:
            if other.mrope_positions is None:
                self.mrope_positions = mrope_positions
            else:
                self.mrope_positions = torch.cat(
                    [self.mrope_positions, other.mrope_positions], dim=1
                )

        # 合并 mrope_position_delta（沿批次维度拼接）
        mrope_position_delta = self.mrope_position_delta
        if mrope_position_delta is not None:
            if other.mrope_position_delta is None:
                self.mrope_position_delta = mrope_position_delta
            else:
                self.mrope_position_delta = torch.cat(
                    [self.mrope_position_delta, other.mrope_position_delta], dim=0
                )

        for key, val in other.__dict__.items():
            if "_id" in key:
                # set token_ids（合并特殊 token ID，优先使用已有值）
                if getattr(self, key, None) is None:
                    setattr(self, key, getattr(other, key, None))
        # other args would be kept intact


# 请求类：存储单个推理请求的输入、输出和状态信息
class Req(ReqDllmMixin):
    """The input and output status of a request."""

    def __init__(
        self,
        rid: str,
        origin_input_text: str,
        origin_input_ids: List[int],
        sampling_params: SamplingParams,
        return_logprob: bool = False,
        top_logprobs_num: int = 0,
        dllm_config: Optional[DllmConfig] = None,
        token_ids_logprob: List[int] = None,
        stream: bool = False,
        origin_input_ids_unpadded: Optional[Tuple[int]] = None,
        lora_id: Optional[str] = None,
        input_embeds: Optional[List[List[float]]] = None,
        positional_embed_overrides: Optional[PositionalEmbeds] = None,
        token_type_ids: List[int] = None,
        session: Optional[Session] = None,
        custom_logit_processor: Optional[str] = None,
        require_reasoning: bool = False,
        return_hidden_states: bool = False,
        return_routed_experts: bool = False,
        eos_token_ids: Optional[Set[int]] = None,
        bootstrap_host: Optional[str] = None,
        bootstrap_port: Optional[int] = None,
        bootstrap_room: Optional[int] = None,
        disagg_mode: Optional[DisaggregationMode] = None,
        routed_dp_rank: Optional[int] = None,
        disagg_prefill_dp_rank: Optional[int] = None,
        vocab_size: Optional[int] = None,
        priority: Optional[int] = None,
        metrics_collector: Optional[SchedulerMetricsCollector] = None,
        extra_key: Optional[str] = None,
        routing_key: Optional[str] = None,
        dimensions: Optional[int] = None,
        http_worker_ipc: Optional[str] = None,
        time_stats: Optional[
            Union[APIServerReqTimeStats, DPControllerReqTimeStats]
        ] = None,
        return_pooled_hidden_states: bool = False,
        multi_item_delimiter_indices: Optional[List[int]] = None,
    ):
        # Input and output info（请求 ID、输入文本和 token）
        self.rid = rid
        self.origin_input_text = origin_input_text
        # 未填充前的原始输入 token（图像填充前）
        self.origin_input_ids_unpadded = (
            origin_input_ids_unpadded
            if origin_input_ids_unpadded
            else origin_input_ids  # Before image padding
        )
        self.origin_input_ids = origin_input_ids
        # Each decode stage's output ids（每轮解码的输出 token ID）
        self.output_ids = []
        # fill_ids = origin_input_ids + output_ids. Updated if chunked.（完整 token 序列）
        self.fill_ids = []
        self.session = session
        self.input_embeds = input_embeds
        self.positional_embed_overrides = positional_embed_overrides
        self.multi_item_delimiter_indices = multi_item_delimiter_indices

        # For req-level memory management（请求级 KV 缓存内存管理字段）
        self.kv_committed_len = 0  # 已提交到缓存树的 KV 长度
        self.kv_allocated_len = 0  # 实际分配的 KV 槽位长度
        self.kv_committed_freed = False  # 已提交部分是否已释放
        self.kv_overallocated_freed = False  # 超分配部分是否已释放

        # for corss-endoder model（交叉编码器模型所需的 token 类型 ID）
        self.token_type_ids = token_type_ids

        # The length of KV that have been removed in swa cache.
        # SWA KV cache eviction behavior differs by cache type:
        # - Radix cache: KV in range [cache_protected_len, swa_evicted_seqlen) is freed manually in
        #   `ScheduleBatch.maybe_evict_swa`; KV in range [0, cache_protected_len) is freed during radix cache eviction.
        # - Chunk cache: KV in range [0, swa_evicted_seqlen) is freed manually in `ScheduleBatch.maybe_evict_swa`.
        # 滑动窗口注意力缓存已驱逐的序列长度
        self.swa_evicted_seqlen = 0

        # The index of the extend / decode batch（在扩展批次/解码批次中的索引）
        self.extend_batch_idx = 0
        self.decode_batch_idx = 0

        # For multi-http worker（多 HTTP Worker 时的 IPC 地址）
        self.http_worker_ipc = http_worker_ipc

        # Require reasoning for the request（是否启用思维链推理模式）
        self.require_reasoning = require_reasoning

        # State indicating whether the reasoning phase has finished (only meaningful when require_reasoning is True)
        self._is_reasoning_over = False  # 推理阶段是否已完成
        self.reasoning_tokens = 0  # 推理阶段的 token 数

        # Sampling info（采样参数初始化）
        if isinstance(sampling_params.custom_params, dict):
            sampling_params = copy.copy(sampling_params)
            # 将请求对象本身注入到自定义参数中，供自定义 logit 处理器使用
            sampling_params.custom_params = sampling_params.custom_params | {
                "__req__": self
            }
        self.sampling_params = sampling_params
        self.custom_logit_processor = custom_logit_processor
        self.return_hidden_states = return_hidden_states

        # extra key for classifying the request (e.g. cache_salt)
        if lora_id is not None:
            # lora_id 拼接到 extra_key，用于 LoRA 缓存隔离
            extra_key = (
                extra_key or ""
            ) + lora_id  # lora_id is concatenated to the extra key

        self.extra_key = extra_key
        self.lora_id = lora_id
        self.routing_key = routing_key  # 用于路由键调度策略

        # Memory pool info（内存池索引）
        self.req_pool_idx: Optional[int] = None  # 请求在 ReqToTokenPool 中的索引
        self.mamba_pool_idx: Optional[torch.Tensor] = None  # shape (1)（Mamba SSM 状态池索引）
        self.mamba_ping_pong_track_buffer: Optional[torch.Tensor] = None  # shape (2)（Mamba 乒乓缓冲区）
        self.mamba_next_track_idx: Optional[int] = None  # 0 or 1（下一个写入的乒乓缓冲区索引）
        self.mamba_last_track_seqlen: Optional[int] = (
            None  # seq len of the last cached mamba state（最后缓存的 Mamba 状态的序列长度）
        )
        # the branching point seqlen to track mamba state. If set, given by prefix match,
        # it will be the tracked seqlen in the ping pong buffer for the right prefill pass.
        # 分支点序列长度，用于 Mamba 状态跟踪
        self.mamba_branching_seqlen: Optional[int] = None

        # Check finish（完成状态检查相关字段）
        self.tokenizer = None
        self.finished_reason: Optional[BaseFinishReason] = None
        # finished position (in output_ids), used when checking stop conditions with speculative decoding
        # 完成时在 output_ids 中的位置（用于投机解码的停止条件检查）
        self.finished_len = None
        # Whether this request has finished output（是否已完成输出）
        self.finished_output = None
        # If we want to abort the request in the middle of the event loop,
        # set to_finish instead of directly setting finished_reason.
        # Note: We should never set finished_reason in the middle, the req will get filtered and never respond
        # 中途中止时使用 to_finish，避免直接设置 finished_reason 导致响应丢失
        self.to_finish: Optional[BaseFinishReason] = None
        self.stream = stream
        self.eos_token_ids = eos_token_ids
        self.vocab_size = vocab_size
        self.priority = priority

        # For incremental decoding（增量解码位置追踪）
        # ----- | --------- read_ids -------|
        # ----- |   surr_ids  |
        # xxxxx | xxxxxxxxxxx | xxxxxxxxxxx |
        # ----- ^ ----------- ^ ----------- ^
        # ----- 1 ----------- 2 ----------- 3
        # 1: surr_offset
        # 2: read_offset
        # 3: last token
        self.surr_offset = None  # Surrounding offset to defeat the cleanup algorithm（环绕偏移量）
        self.read_offset = None  # 当前读取偏移量
        self.decoded_text = ""  # 已解码的文本

        # For multimodal inputs（多模态输入数据）
        self.multimodal_inputs: Optional[MultimodalInputs] = None

        # Prefix info（前缀缓存相关信息）
        # The indices to kv cache for the shared prefix.
        self.prefix_indices: torch.Tensor = torch.empty((0,), dtype=torch.int64)  # 共享前缀在 KV 缓存中的索引
        # Number of tokens to run prefill.
        self.extend_input_len = 0  # 需要执行 prefill 的 token 数
        # The relative logprob_start_len in an extend batch
        self.extend_logprob_start_len = 0  # 在扩展批次中 logprob 的起始偏移
        self.last_node: Any = None  # 在 Radix 树中的末端节点
        self.last_host_node: Any = None  # 在 host 缓存树中的末端节点
        self.host_hit_length = 0  # 在 host 缓存（L2）中命中的 token 数
        # Tokens loaded from storage backend (L3) during prefetch for this request
        self.storage_hit_length = 0  # 在存储后端（L3）中命中的 token 数
        # The node to lock until for swa radix tree lock ref
        self.swa_uuid_for_lock: Optional[int] = None  # SWA Radix 树加锁用的 UUID
        # The prefix length that is inserted into the tree cache
        self.cache_protected_len: int = 0  # 已插入树缓存的前缀长度

        # Whether or not if it is chunked. It increments whenever
        # it is chunked, and decrement whenever chunked request is
        # processed.
        self.is_chunked = 0  # 分块 prefill 计数器（非零表示正在分块中）

        # For retraction（回退相关状态）
        self.is_retracted = False  # 当前是否被回退
        # Indicates if the req has ever been retracted.
        self.retracted_stain = False  # 是否曾经被回退过

        # Incremental streamining（增量流式输出偏移）
        self.send_token_offset: int = 0
        self.send_decode_id_offset: int = 0
        # TODO (Byron): send_output_token_logprobs_offset and send_decode_id_offset can be different in disaggregation mode
        # because the decode server does not have the first output token logprobs
        self.send_output_token_logprobs_offset: int = 0

        # Logprobs (arguments)（对数概率计算参数）
        self.return_logprob = return_logprob
        # Start index to compute logprob from.
        self.logprob_start_len = 0  # 开始计算 logprob 的位置
        self.top_logprobs_num = top_logprobs_num
        self.token_ids_logprob = token_ids_logprob
        self.temp_scaled_logprobs = False  # 是否使用温度缩放 logprob
        self.top_p_normalized_logprobs = False  # 是否使用 top-p 归一化 logprob

        # Logprobs (return values)（对数概率返回值）
        # True means the input logprob has been already sent to detokenizer.
        self.input_logprob_sent: bool = False  # 输入 logprob 是否已发送给 detokenizer
        self.input_token_logprobs_val: Optional[List[float]] = None
        self.input_token_logprobs_idx: Optional[List[int]] = None
        self.input_top_logprobs_val: Optional[List[float]] = None
        self.input_top_logprobs_idx: Optional[List[int]] = None
        self.input_token_ids_logprobs_val: Optional[List[float]] = None
        self.input_token_ids_logprobs_idx: Optional[List[int]] = None
        # Temporary holder to store input_token_logprobs.
        self.input_token_logprobs: Optional[List[Tuple[int]]] = None
        self.temp_input_top_logprobs_val: Optional[List[torch.Tensor]] = None
        self.temp_input_top_logprobs_idx: Optional[List[int]] = None
        self.temp_input_token_ids_logprobs_val: Optional[List[float]] = None
        self.temp_input_token_ids_logprobs_idx: Optional[List[int]] = None

        if return_logprob:
            # shape: (bs, 1)（输出 token 的对数概率，每步一个）
            self.output_token_logprobs_val = []
            self.output_token_logprobs_idx = []
            # shape: (bs, k)（输出 token 的 top-k 对数概率）
            self.output_top_logprobs_val = []
            self.output_top_logprobs_idx = []
            # Can contain either lists or GPU tensors (delayed copy optimization for prefill-only scoring)
            # 可以包含列表或 GPU 张量（用于仅 prefill 评分的延迟复制优化）
            self.output_token_ids_logprobs_val: List[
                Union[List[float], torch.Tensor]
            ] = []
            self.output_token_ids_logprobs_idx = []
        else:
            # 不需要 logprob 时全部设为 None
            self.output_token_logprobs_val = self.output_token_logprobs_idx = (
                self.output_top_logprobs_val
            ) = self.output_top_logprobs_idx = self.output_token_ids_logprobs_val = (
                self.output_token_ids_logprobs_idx
            ) = None
        self.hidden_states: List[List[float]] = []
        self.hidden_states_tensor = None  # Note: use tensor instead of list to transfer hidden_states when PD + MTP
        self.output_topk_p = None
        self.output_topk_index = None

        # capture routed experts（记录路由的专家索引，用于 MoE 模型）
        self.return_routed_experts = return_routed_experts
        self.routed_experts: Optional[torch.Tensor] = (
            None  # cpu tensor: shape (seqlen, topk)
        )
        # Customized info（自定义扩展信息）
        self.customized_info: Optional[Dict[str, List[Any]]] = None

        # Embedding (return values)（嵌入向量输出）
        self.embedding = None

        # Constrained decoding（约束解码相关字段）
        self.grammar_key: Optional[Tuple[str, str]] = None
        self.grammar: Optional[Union[BaseGrammarObject, Future[BaseGrammarObject]]] = (
            None
        )
        self.grammar_wait_ct = 0  # 等待语法对象就绪的计数器

        # The number of cached tokens that were already cached in the KV cache
        self.cached_tokens = 0  # 已命中缓存的 token 总数
        self.already_computed = 0  # 已计算过的 token 数

        # Detailed breakdown of cached tokens by source (for HiCache)
        self.cached_tokens_device = 0  # Tokens from device cache (GPU)（来自 GPU 设备缓存的 token 数）
        self.cached_tokens_host = 0  # Tokens from host cache (CPU memory)（来自 CPU 内存缓存的 token 数）
        self.cached_tokens_storage = 0  # Tokens from L3 storage backend
        self._cache_breakdown_computed = (
            False  # Track if breakdown was already computed（是否已计算过缓存分类统计）
        )

        # Per-request count of verification forward passes.
        self.spec_verify_ct = 0  # 投机解码中验证前向传播的次数

        # Per-request count of accepted draft tokens (excludes the bonus token).
        self.spec_accepted_drafts = 0  # 投机解码中被接受的草稿 token 总数

        # Acceptance histogram for speculative decoding.
        # List index = number of accepted tokens in a step, List value = count of steps with that many accepted tokens.
        # Example: histogram[0] = 5 means 5 steps with 0 accepted tokens, histogram[3] = 10 means 10 steps with 3 accepted tokens.
        # 投机解码接受率直方图
        self.spec_acceptance_histogram: List[int] = []

        # The number of times this request has been retracted / preempted.
        self.retraction_count = 0  # 被回退/抢占的次数
        self.retraction_mb_id = None  # 回退时对应的 mini-batch ID

        # For observability（可观测性：指标收集和时间统计）
        self.metrics_collector = metrics_collector
        if time_stats is not None:
            self.time_stats = SchedulerReqTimeStats.new_from_obj(time_stats)
        else:
            self.time_stats = SchedulerReqTimeStats(disagg_mode=disagg_mode)
        self.time_stats.set_metrics_collector(metrics_collector)
        self.time_stats.set_scheduler_recv_time()
        self.has_log_time_stats: bool = False  # 是否已记录过时间统计日志

        # For disaggregation（解耦推理相关字段：预填充/解码分离）
        self.bootstrap_host: str = bootstrap_host
        self.bootstrap_port: Optional[int] = bootstrap_port
        self.bootstrap_room: Optional[int] = bootstrap_room
        self.disagg_kv_sender: Optional[BaseKVSender] = None

        self.routed_dp_rank: Optional[int] = routed_dp_rank
        self.disagg_prefill_dp_rank: Optional[int] = disagg_prefill_dp_rank

        # the start index of the sent kv cache
        # We want to send it chunk by chunk for chunked prefill.
        # After every chunk forward, we do the following:
        # kv_send(req.input_ids[req.start_send_idx:len(req.fill_ids)])
        # start_send_idx = len(req.fill_ids)
        # 已发送的 KV 缓存起始索引（用于分块 prefill 的 KV 传输）
        self.start_send_idx: int = 0

        # For overlap schedule, we delay the kv transfer until `process_batch_result_disagg_prefill` rather than `process_prefill_chunk` in non-overlap
        # This is because kv is not ready in `process_prefill_chunk`.
        # We use `tmp_end_idx` to store the end index of the kv cache to send.
        # 重叠调度模式下，延迟 KV 传输的临时终止索引
        self.tmp_end_idx: int = -1
        self.metadata_buffer_index: int = -1  # 元数据缓冲区索引

        # For Matryoshka embeddings（嵌入维度截断，用于 Matryoshka 嵌入）
        self.dimensions = dimensions

        # Whether to return pooled hidden states (pre-head transformer output)（是否返回池化的隐藏状态）
        self.return_pooled_hidden_states = return_pooled_hidden_states
        self.pooled_hidden_state = None

        # For diffusion LLM（扩散语言模型初始化）
        self.init_diffusion_llm(dllm_config)

        # For hisparse（HiSparse 稀疏注意力暂存标志）
        self.hisparse_staging = False

    @property
    def seqlen(self) -> int:
        """Get the current sequence length of the request."""
        return len(self.origin_input_ids) + len(self.output_ids)

    @property
    def is_prefill_only(self) -> bool:
        """Check if this request is prefill-only (no token generation needed)."""
        # NOTE: when spec is enabled, prefill_only optimizations are disabled

        spec_alg = get_global_server_args().speculative_algorithm
        return self.sampling_params.max_new_tokens == 0 and spec_alg is None

    @property
    def output_ids_through_stop(self) -> List[int]:
        """Get the output ids through the stop condition. Stop position is included."""
        if self.finished_len is not None:
            return self.output_ids[: self.finished_len]
        return self.output_ids

    def _cache_commit_len(self) -> int:
        # Report only the prompt prefix so thinking + answer fall into the
        # overallocated range and are reclaimed by release_kv_cache. #22373.
        if get_global_server_args().strip_thinking_cache and self.reasoning_tokens > 0:
            return min(self.kv_committed_len, len(self.origin_input_ids))
        return self.kv_committed_len

    def pop_committed_kv_cache(self) -> int:
        """Return the length of committed KV cache and mark them as freed."""
        assert (
            not self.kv_committed_freed
        ), f"Committed KV cache already freed ({self.kv_committed_len=})"
        self.kv_committed_freed = True
        return self._cache_commit_len()

    def pop_overallocated_kv_cache(self) -> Tuple[int, int]:
        """Return the range of over-allocated KV cache and mark them as freed."""

        # NOTE: This function is called when there is over-allocation of KV cache.
        # Over-allocation: we allocate more KV cache than the committed length.
        # e.g., speculative decoding may allocate more KV cache than actually used.
        assert (
            not self.kv_overallocated_freed
        ), f"Overallocated KV cache already freed, {self.kv_committed_len=}, {self.kv_allocated_len=}"
        self.kv_overallocated_freed = True
        return self._cache_commit_len(), self.kv_allocated_len

    def update_spec_acceptance_histogram(self, accepted_draft_tokens: int):
        """Update the speculative decoding acceptance histogram.

        Args:
            accepted_draft_tokens: Number of draft tokens accepted in this step.
        """
        if len(self.spec_acceptance_histogram) <= accepted_draft_tokens:
            self.spec_acceptance_histogram.extend(
                [0] * (accepted_draft_tokens - len(self.spec_acceptance_histogram) + 1)
            )
        self.spec_acceptance_histogram[accepted_draft_tokens] += 1

    def extend_image_inputs(self, image_inputs):
        # 追加多模态输入，若已有则合并
        if self.multimodal_inputs is None:
            self.multimodal_inputs = image_inputs
        else:
            self.multimodal_inputs.merge(image_inputs)

    def finished(self) -> bool:
        # Whether request reached finished condition（判断请求是否已完成）
        return self.finished_reason is not None

    def init_next_round_input(
        self,
        tree_cache: Optional[BasePrefixCache] = None,
        cow_mamba: Optional[bool] = None,
    ):
        # 初始化下一轮推理的输入：更新 fill_ids，并在 KV 缓存中查找最长前缀匹配
        if self.is_dllm():
            self._init_fill_ids_for_dllm()
            self.determine_dllm_phase()
        else:
            # 普通模式：fill_ids = 原始输入 + 已生成输出
            self.fill_ids = self.origin_input_ids + self.output_ids

        input_len = len(self.fill_ids)

        # Streaming sessions reuse committed KV from the session slot, so
        # custom logprob_start_len is not supported — override to -1.
        # 流式会话不支持自定义 logprob_start_len，强制覆盖为 -1
        if (
            self.session is not None
            and self.session.streaming
            and self.return_logprob
            and self.logprob_start_len >= 0
        ):
            logger.warning(
                "logprob_start_len=%d is not supported for streaming sessions "
                "and will be ignored (rid=%s). Only new-token logprobs are returned.",
                self.logprob_start_len,
                self.rid,
            )
            self.logprob_start_len = -1

        # NOTE: the matched length is at most 1 less than the input length to enable logprob computation
        # 最大前缀长度为 input_len - 1，确保最后一个 token 不被缓存以支持 logprob 计算
        max_prefix_len = input_len - 1
        if self.return_logprob and self.logprob_start_len >= 0:
            max_prefix_len = min(max_prefix_len, self.logprob_start_len)
        max_prefix_len = max(max_prefix_len, 0)
        token_ids = self.fill_ids[:max_prefix_len]

        # Disable prefix caching when embed overrides are present: same token IDs
        # with different override vectors must not share cached KV values.
        # 存在位置编码覆盖时禁用前缀缓存，避免不同向量共享缓存
        if self.positional_embed_overrides is not None:
            max_prefix_len = 0
            token_ids = []

        if tree_cache is not None:
            if cow_mamba is None:
                cow_mamba = tree_cache.supports_mamba()
            # 在 KV 缓存树中查找最长前缀匹配
            match_result = tree_cache.match_prefix(
                MatchPrefixParams(
                    key=RadixKey(token_ids=token_ids, extra_key=self.extra_key),
                    req=self,
                    cow_mamba=cow_mamba,
                )
            )
            # 更新前缀相关字段
            (
                self.prefix_indices,
                self.last_node,
                self.last_host_node,
                self.host_hit_length,
                self.mamba_branching_seqlen,
            ) = (
                match_result.device_indices,
                match_result.last_device_node,
                match_result.last_host_node,
                match_result.host_hit_length,
                match_result.mamba_branching_seqlen,
            )
            if match_result.cache_protected_len is not None:
                self.cache_protected_len = match_result.cache_protected_len
            else:
                self.cache_protected_len = len(self.prefix_indices)

            if self.is_dllm():
                self._update_block_offset_for_dllm()

        if (
            self.is_retracted
            and self.multimodal_inputs is not None
            and self.multimodal_inputs.mrope_positions is not None
        ):
            # 回退后重新计算 mrope 位置编码（扩展到包含已生成输出的长度）
            from sglang.srt.managers.mm_utils import (
                extend_mrope_positions_for_retracted_request,
            )

            self.multimodal_inputs.mrope_positions = (
                extend_mrope_positions_for_retracted_request(
                    self.multimodal_inputs.mrope_positions, len(self.output_ids)
                )
            )

        # 计算本轮需要执行 prefill 的 token 数
        self.set_extend_input_len(len(self.fill_ids) - len(self.prefix_indices))

    # Based on https://github.com/vllm-project/vllm/blob/7a64d24aad69e4d2548aa0bf528d9fe63428ab01/vllm/transformers_utils/detokenizer.py#L194-L313
    def init_incremental_detokenize(self):
        # 初始化增量解码，计算 surr_and_decode_ids 和 read_offset 的初始值
        first_iter = self.surr_offset is None or self.read_offset is None

        output_ids = self.output_ids_through_stop

        if first_iter:
            # 首次调用：设置 read_offset 和 surr_offset
            self.read_offset = len(self.origin_input_ids_unpadded)
            self.surr_offset = max(
                self.read_offset - INIT_INCREMENTAL_DETOKENIZATION_OFFSET, 0
            )
            self.surr_and_decode_ids = (
                self.origin_input_ids_unpadded[self.surr_offset :] + output_ids
            )
            self.cur_decode_ids_len = len(output_ids)
        else:
            # 后续调用：只追加新生成的 token
            self.surr_and_decode_ids.extend(output_ids[self.cur_decode_ids_len :])
            self.cur_decode_ids_len = len(output_ids)

        return self.surr_and_decode_ids, self.read_offset - self.surr_offset

    def tail_str(self) -> str:
        # Check stop strings and stop regex patterns together
        # 获取最近生成文本的尾部字符串，用于停止条件检查
        if (
            len(self.sampling_params.stop_strs) == 0
            and len(self.sampling_params.stop_regex_strs) == 0
        ):
            return ""

        max_len_tail_str = max(
            self.sampling_params.stop_str_max_len + 1,
            self.sampling_params.stop_regex_max_len + 1,
        )

        tail_len = min(max_len_tail_str, len(self.output_ids))
        return self.tokenizer.decode(self.output_ids[-tail_len:])

    def check_match_stop_str_prefix(self) -> bool:
        """
        Check if the suffix of tail_str overlaps with any stop_str prefix
        """
        # 检查当前尾部字符串是否与任意停止字符串存在前缀重叠（用于流式输出提前检测）
        if not self.sampling_params.stop_strs:
            return False

        tail_str = self.tail_str()

        # Early return if tail_str is empty
        if not tail_str:
            return False

        for stop_str in self.sampling_params.stop_strs:
            if not stop_str:
                continue
            # Check if stop_str is contained in tail_str (fastest check first)
            if stop_str in tail_str:
                return True

            # Check if tail_str suffix matches stop_str prefix
            # Only check if stop_str is not empty, it's for stream output
            min_len = min(len(tail_str), len(stop_str))
            for i in range(1, min_len + 1):
                if tail_str[-i:] == stop_str[:i]:
                    return True

        return False

    def _check_token_based_finish(self, new_accepted_tokens: List[int]) -> bool:
        # 检查新生成的 token 是否命中停止 token 集合（EOS 或自定义停止 token）
        if self.sampling_params.ignore_eos:
            return False

        # Check stop token ids
        matched_eos = False

        for i, token_id in enumerate(new_accepted_tokens):
            if self.sampling_params.stop_token_ids:
                matched_eos |= token_id in self.sampling_params.stop_token_ids
            if self.eos_token_ids:
                matched_eos |= token_id in self.eos_token_ids
            if self.tokenizer is not None:
                matched_eos |= token_id == self.tokenizer.eos_token_id
                if self.tokenizer.additional_stop_token_ids:
                    matched_eos |= token_id in self.tokenizer.additional_stop_token_ids
            if matched_eos:
                self.finished_reason = FINISH_MATCHED_TOKEN(matched=token_id)
                matched_pos = len(self.output_ids) - len(new_accepted_tokens) + i
                self.finished_len = matched_pos + 1
                return True

        return False

    def _check_str_based_finish(self):
        # 检查当前尾部字符串是否命中停止字符串或停止正则表达式
        if (
            len(self.sampling_params.stop_strs) > 0
            or len(self.sampling_params.stop_regex_strs) > 0
        ):
            tail_str = self.tail_str()

            # Check stop strings（检查停止字符串）
            if len(self.sampling_params.stop_strs) > 0:
                for stop_str in self.sampling_params.stop_strs:
                    if stop_str in tail_str or stop_str in self.decoded_text:
                        self.finished_reason = FINISH_MATCHED_STR(matched=stop_str)
                        return True

            # Check stop regex（检查停止正则表达式）
            if len(self.sampling_params.stop_regex_strs) > 0:
                for stop_regex_str in self.sampling_params.stop_regex_strs:
                    if re.search(stop_regex_str, tail_str):
                        self.finished_reason = FINISHED_MATCHED_REGEX(
                            matched=stop_regex_str
                        )
                        return True

        return False

    def _check_vocab_boundary_finish(self, new_accepted_tokens: List[int] = None):
        # 检查是否生成了超出词表范围的 token（NaN 或越界），若是则强制终止请求
        for i, token_id in enumerate(new_accepted_tokens):
            if token_id > self.vocab_size or token_id < 0:
                offset = len(self.output_ids) - len(new_accepted_tokens) + i
                if self.sampling_params.stop_token_ids:
                    # 将异常 token 替换为第一个停止 token
                    self.output_ids[offset] = next(
                        iter(self.sampling_params.stop_token_ids)
                    )
                if self.eos_token_ids:
                    self.output_ids[offset] = next(iter(self.eos_token_ids))
                self.finished_reason = FINISH_MATCHED_STR(matched="NaN happened")
                self.finished_len = offset + 1
                return True

        return False

    def check_finished(self, new_accepted_len: int = 1):
        # 综合检查请求是否已满足完成条件（长度限制、token 停止、字符串停止等）
        if self.finished():
            return

        # 优先处理显式中止标志
        if self.to_finish:
            self.finished_reason = self.to_finish
            self.to_finish = None
            return

        # 检查最大输出长度限制
        if len(self.output_ids) >= self.sampling_params.max_new_tokens:
            self.finished_reason = FINISH_LENGTH(
                length=self.sampling_params.max_new_tokens
            )
            self.finished_len = self.sampling_params.max_new_tokens
            return

        # 检查语法约束是否已终止
        if self.grammar is not None:
            if self.grammar.is_terminated():
                self.finished_reason = FINISH_MATCHED_TOKEN(matched=self.output_ids[-1])
                return

        # 获取最新接受的 token
        new_accepted_tokens = self.output_ids[-new_accepted_len:]

        # 依次检查 token 停止、词表越界和字符串停止条件
        if self._check_token_based_finish(new_accepted_tokens):
            return

        if self._check_vocab_boundary_finish(new_accepted_tokens):
            return

        if self._check_str_based_finish():
            return

    def reset_for_retract(self):
        # Increment retraction count before resetting other state. We should not reset this
        # since we are tracking the total number of retractions for each request.
        # 回退前先递增计数器，然后重置所有与当前推理状态相关的字段
        self.retraction_count += 1

        self.prefix_indices = torch.empty((0,), dtype=torch.int64)
        self.routed_experts = None
        self.last_node = None
        self.swa_uuid_for_lock = None
        self.extend_input_len = 0
        self.is_retracted = True
        self.retracted_stain = True
        self.input_token_logprobs = None
        self.temp_input_top_logprobs_val = None
        self.temp_input_top_logprobs_idx = None
        self.extend_logprob_start_len = 0
        self.is_chunked = 0
        self.mamba_pool_idx = None
        self.mamba_ping_pong_track_buffer = None
        self.mamba_next_track_idx = None
        self.mamba_last_track_seqlen = None
        self.mamba_branching_seqlen = None
        self.already_computed = 0
        self.kv_allocated_len = 0
        self.kv_committed_len = 0
        self.kv_committed_freed = False
        self.kv_overallocated_freed = False
        self.swa_evicted_seqlen = 0
        self.extend_batch_idx = 0
        self.decode_batch_idx = 0

        # When using input_embeds, we cannot easily mix the original input embeddings
        # with the newly generated output token IDs during re-prefill of retracted request.
        # output_ids will have no use, but will lead to wrong size cache indexes.
        # Therefore, we discard the generated output_ids and restart prefill and generation
        # to ensure shape consistency in KV cache.
        if self.input_embeds is not None:
            self.output_ids = []

    def offload_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):
        # 将请求的 KV 缓存（以及 Mamba 状态）卸载到 CPU 内存
        token_indices = req_to_token_pool.req_to_token[
            self.req_pool_idx, : self.seqlen - 1
        ]
        # Copies over both the kv cache and mamba state if available
        self.kv_cache_cpu = token_to_kv_pool_allocator.get_cpu_copy(
            token_indices, mamba_indices=self.mamba_pool_idx
        )

    def load_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):
        # 将 CPU 上暂存的 KV 缓存重新加载到 GPU
        token_indices = req_to_token_pool.req_to_token[
            self.req_pool_idx, : self.seqlen - 1
        ]
        # Loads both the kv cache and mamba state if exists
        token_to_kv_pool_allocator.load_cpu_copy(
            self.kv_cache_cpu, token_indices, mamba_indices=self.mamba_pool_idx
        )
        del self.kv_cache_cpu  # 释放 CPU 缓存副本

    def log_time_stats(self):
        # If overlap schedule, we schedule one decode batch ahead so this gets called twice.
        # 重叠调度时可能被调用两次，用 has_log_time_stats 防止重复记录
        if self.has_log_time_stats:
            return

        bootstrap_info = (
            f", bootstrap_room={self.bootstrap_room}"
            if self.bootstrap_room is not None
            else ""
        )
        prefix = f"Req Time Stats(rid={self.rid}{bootstrap_info}, input len={len(self.origin_input_ids)}, output len={len(self.output_ids)}, type={self.time_stats.disagg_mode_str()})"
        logger.info(f"{prefix}: {self.time_stats.convert_to_duration()}")
        self.has_log_time_stats = True

    def set_extend_input_len(self, extend_input_len: int):
        # Setting extend_input_len and computing the relative logprob_start_len in an extend batch
        #
        # Key variables:
        # - logprob_start_len: Absolute position in full sequence where logprob computation begins
        # - extend_logprob_start_len: Relative position within current extend batch where logprob computation begins
        # - extend_input_len: Number of tokens that need to be processed in this extend batch
        # 设置本轮 extend_input_len，并计算 logprob 在扩展批次内的相对起始位置
        self.extend_input_len = extend_input_len
        if self.logprob_start_len == -1:
            # -1 表示只计算最后一个输出 token 的 logprob
            logprob_start_len = len(self.fill_ids)
        else:
            # logprob_start_len should be at least the length of the prefix indices
            # 确保 logprob 起始位置不早于前缀缓存边界
            logprob_start_len = max(self.logprob_start_len, len(self.prefix_indices))
        self.extend_logprob_start_len = min(
            logprob_start_len - len(self.prefix_indices),
            self.extend_input_len,
        )

    def set_finish_with_abort(self, error_msg: str):
        # 将请求标记为中止状态，同时清理昂贵资源（多模态输入、语法对象）
        if get_tensor_model_parallel_rank() == 0:
            logger.error(f"{error_msg}, {self.rid=}")
        self.multimodal_inputs = None
        self.grammar = None
        self.origin_input_ids = [0]  # set it to one token to skip the long prefill（跳过长 prefill）
        self.return_logprob = False
        self.logprob_start_len = -1
        self.to_finish = FINISH_ABORT(
            error_msg, HTTPStatus.BAD_REQUEST, "BadRequestError"
        )

    def update_reasoning_tokens(self, token_id, think_end_id):
        # 更新推理阶段 token 计数，检测到思维结束 token 后标记推理阶段结束
        if self._is_reasoning_over:
            return

        if not isinstance(token_id, list):
            token_id = [token_id]

        try:
            # 找到思维结束 token 的位置，之前（含）的 token 都属于推理阶段
            end_pos = token_id.index(think_end_id)
            self.reasoning_tokens += end_pos + 1
            self._is_reasoning_over = True
        except ValueError:
            # 未出现思维结束 token，当前所有 token 都属于推理阶段
            self.reasoning_tokens += len(token_id)

    def __repr__(self):
        # 请求对象的字符串表示（用于调试和日志）
        return (
            f"Req(rid={self.rid}, "
            f"input_ids={self.origin_input_ids}, output_ids={self.output_ids}, "
            f"{self.grammar=}, "
            f"{self.sampling_params=})"
        )


@dataclasses.dataclass
class ScheduleBatch(ScheduleBatchDisaggregationDecodeMixin):
    """Store all information of a batch on the scheduler."""
    # 调度批次：存储调度器管理的一个批次的所有信息

    # Request, memory pool, and cache（请求列表、内存池和缓存）
    reqs: List[Req]
    req_to_token_pool: ReqToTokenPool = None  # 请求到 token 位置的映射池
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator = None  # KV 缓存分配器
    tree_cache: BasePrefixCache = None  # 前缀缓存树
    is_hybrid_swa: bool = False  # 是否为滑动窗口注意力混合模式

    # Batch configs（批次配置）
    model_config: ModelConfig = None
    forward_mode: ForwardMode = None  # 当前批次的前向传播模式（extend/decode/etc.）
    enable_overlap: bool = False  # 是否启用重叠调度
    # Tell whether the current running batch is full so that we can skip
    # the check of whether to prefill new requests.
    # This is an optimization to reduce the overhead of the prefill check.
    # 标记批次是否已满，用于跳过新请求的 prefill 准入检查（减少开销）
    batch_is_full: bool = False

    # For chunked prefill in PP（流水线并行中的分块 prefill 请求）
    chunked_req: Optional[Req] = None

    # Sampling info（采样批次信息）
    sampling_info: SamplingBatchInfo = None

    # Batched arguments to model runner（传入模型 runner 的批次张量）
    input_ids: torch.Tensor = None  # shape: [b], int64（输入 token ID）
    input_embeds: torch.Tensor = None  # shape: [b, hidden_size], float32（输入嵌入向量）
    # Token replacement embeddings and absolute positions (optional).
    replace_embeds: Optional[torch.Tensor] = None  # 替换嵌入向量（可选）
    replace_positions: Optional[torch.Tensor] = None  # 替换位置（可选）
    ne_token_table: torch.Tensor = None  # 非嵌入 token 表
    token_type_ids: torch.Tensor = None  # shape: [b], int64（token 类型 ID）
    req_pool_indices: torch.Tensor = None  # shape: [b], int64（请求在内存池中的索引）
    seq_lens: torch.Tensor = None  # shape: [b], int64（每个请求的序列长度）
    seq_lens_cpu: torch.Tensor = None  # shape: [b], int64（CPU 上的序列长度副本）
    # The output locations of the KV cache
    out_cache_loc: torch.Tensor = None  # shape: [b], int64（KV 缓存的输出位置）
    output_ids: torch.Tensor = None  # shape: [b], int64（本轮输出 token ID）

    # For hybrid GDN prefix cache（Mamba SSM 状态追踪相关张量）
    mamba_track_indices: torch.Tensor = None  # shape: [b], int64（Mamba 状态追踪索引）
    mamba_track_mask: torch.Tensor = None  # shape: [b], bool（Mamba 追踪掩码）
    mamba_track_seqlens: torch.Tensor = None  # shape: [b], int64（Mamba 追踪序列长度）

    # For multimodal inputs（多模态输入列表）
    multimodal_inputs: Optional[List] = None

    # The sum of all sequence lengths（所有序列长度之和）
    seq_lens_sum: int = None
    # The original sequence lengths, Qwen-1M related
    orig_seq_lens: torch.Tensor = None  # shape: [b], int32（原始序列长度，Qwen-1M 相关）

    # For DP attention（数据并行注意力相关字段）
    inner_idle_batch: Optional[ScheduleBatch] = None  # DP 内部空闲批次
    global_num_tokens: Optional[List[int]] = None  # 全局各 DP rank 的 token 数
    global_num_tokens_for_logprob: Optional[List[int]] = None
    is_extend_in_batch: bool = False  # 批次中是否有 extend 请求
    all_extend_in_batch: bool = False  # 批次中是否全部为 extend 请求
    can_run_dp_cuda_graph: bool = False  # 是否可运行 DP CUDA graph
    tbo_split_seq_index: Optional[int] = None  # TBO 分割序列索引
    global_forward_mode: Optional[ForwardMode] = None  # 全局前向传播模式

    # For processing logprobs（对数概率处理相关字段）
    return_logprob: bool = False  # 是否返回对数概率
    top_logprobs_nums: Optional[List[int]] = None  # 每个请求的 top-k logprob 数
    token_ids_logprobs: Optional[List[List[int]]] = None  # 指定计算 logprob 的 token ID

    # For logits and logprob post processing（logits 后处理标志）
    temp_scaled_logprobs: bool = False  # 是否使用温度缩放
    top_p_normalized_logprobs: bool = False  # 是否使用 top-p 归一化

    # For extend and mixed chunekd prefill（扩展批次和混合分块 prefill 相关）
    prefix_lens: List[int] = None  # 每个请求的前缀长度
    extend_lens: List[int] = None  # 每个请求本轮需扩展的长度
    extend_num_tokens: Optional[int] = None  # 本轮扩展的总 token 数
    decoding_reqs: List[Req] = None  # 解码请求列表（混合 prefill+decode 批次中）
    extend_logprob_start_lens: List[int] = None  # 扩展批次中各请求 logprob 的起始偏移
    # It comes empty list if logprob is not required.
    extend_input_logprob_token_ids: Optional[torch.Tensor] = None  # 用于计算输入 logprob 的 token ID

    # For encoder-decoder architectures（编码器-解码器架构相关）
    encoder_cached: Optional[List[bool]] = None  # 各请求的编码器输出是否已缓存
    encoder_lens: Optional[torch.Tensor] = None  # 编码器序列长度张量
    encoder_lens_cpu: Optional[List[int]] = None  # CPU 上的编码器序列长度
    encoder_out_cache_loc: Optional[torch.Tensor] = None  # 编码器输出缓存位置

    # For matryoshka embeddings（Matryoshka 嵌入维度截断）
    dimensions: Optional[list[int]] = None

    # Whether to return pooled hidden states (pre-head transformer output)（是否返回池化隐藏状态）
    return_pooled_hidden_states: bool = False

    # For split prefill（分割 prefill 相关字段）
    split_index: int = 0  # 当前分割索引
    split_prefill_finished: bool = False  # 分割 prefill 是否已完成
    split_forward_count: int = 1  # 分割前向传播次数
    split_forward_batch: ForwardBatch = None  # 分割前向批次
    seq_lens_cpu_cache: torch.Tensor = None  # 序列长度 CPU 缓存

    # Stream（是否有流式请求）
    has_stream: bool = False

    # Has grammar（是否有语法约束请求）
    has_grammar: bool = False

    # Device（计算设备）
    device: str = "cuda"

    # Speculative decoding（投机解码相关字段）
    spec_algorithm: SpeculativeAlgorithm = None
    # spec_info: Optional[SpecInput] = None
    spec_info: Optional[SpecInput] = None

    # Whether to return hidden states（是否返回隐藏状态）
    return_hidden_states: bool = False

    # Whether to return captured experts（是否返回路由专家信息）
    return_routed_experts: bool = False

    # Whether this batch is prefill-only (no token generation needed)（是否为仅 prefill 批次）
    is_prefill_only: bool = False

    # Multi-item scoring delimiter indices (set during prepare_for_extend)（多项评分分隔符索引）
    multi_item_delimiter_indices: Optional[List[torch.Tensor]] = None

    # hicache pointer for synchronizing data loading from CPU to GPU（HiCache 消费者索引）
    hicache_consumer_index: int = -1

    # Diffusion LLM（扩散语言模型配置）
    dllm_config: Optional[DllmConfig] = None

    # Metrics（指标收集字段）
    dp_cooperation_info: Optional[DPCooperationInfo] = None
    prefill_stats: Optional[PrefillStats] = None

    # HiSparse（稀疏注意力协调器）
    hisparse_coordinator: Optional[HiSparseCoordinator] = None

    @classmethod
    def init_new(
        cls,
        reqs: List[Req],
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tree_cache: BasePrefixCache,
        model_config: ModelConfig,
        enable_overlap: bool,
        spec_algorithm: SpeculativeAlgorithm,
        chunked_req: Optional[Req] = None,
        dllm_config: Optional[DllmConfig] = None,
    ):
        # 创建新的 ScheduleBatch，从请求列表中聚合批次级别的标志位
        return_logprob = any(req.return_logprob for req in reqs)

        is_hybrid_swa = False
        if isinstance(token_to_kv_pool_allocator, SWATokenToKVPoolAllocator):
            is_hybrid_swa = True

        return cls(
            reqs=reqs,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            is_hybrid_swa=is_hybrid_swa,
            model_config=model_config,
            enable_overlap=enable_overlap,
            return_logprob=return_logprob,
            has_stream=any(req.stream for req in reqs),
            has_grammar=any(req.grammar for req in reqs),
            device=req_to_token_pool.device,
            spec_algorithm=spec_algorithm,
            return_hidden_states=any(req.return_hidden_states for req in reqs),
            return_routed_experts=any(req.return_routed_experts for req in reqs),
            is_prefill_only=all(req.is_prefill_only for req in reqs),
            chunked_req=chunked_req,
            dllm_config=dllm_config,
        )

    def batch_size(self):
        # 返回批次中的请求数
        return len(self.reqs)

    def is_empty(self):
        # 判断批次是否为空
        return len(self.reqs) == 0

    def is_dllm(self):
        # 判断是否为扩散语言模型模式
        return self.dllm_config is not None

    def prepare_encoder_info_extend(self, input_ids: List[int], seq_lens: List[int]):
        # 为 encoder-decoder 模型准备编码器信息（如 cross-attention 的图像 token 数）
        _pin = is_pin_memory_available(self.device)
        self.encoder_lens_cpu = []  # 每个请求的编码器 token 长度（CPU 列表）
        self.encoder_cached = []    # 每个请求的编码器部分是否已缓存

        for req in self.reqs:
            im = req.multimodal_inputs
            if im is None or im.num_image_tokens is None:
                # No image input
                # 无图像输入，编码器长度为 0，标记为已缓存
                self.encoder_lens_cpu.append(0)
                self.encoder_cached.append(True)
            else:
                # 有图像 token，记录编码器长度
                self.encoder_lens_cpu.append(im.num_image_tokens)
                # 若处于 decode 阶段或前缀已覆盖图像 token，则标记为已缓存
                self.encoder_cached.append(
                    self.forward_mode.is_decode()
                    or len(req.prefix_indices) >= im.num_image_tokens
                )

        # 将编码器长度列表转为 GPU 张量
        self.encoder_lens = torch.tensor(
            self.encoder_lens_cpu, dtype=torch.int64, pin_memory=_pin
        ).to(self.device, non_blocking=True)

        # Strip encoder infos
        # 将 out_cache_loc 拆分为 encoder 部分和 decoder 部分
        pt = 0
        decoder_out_cache_loc = []
        encoder_out_cache_loc = []
        for i, req in enumerate(self.reqs):
            encoder_len = self.encoder_lens_cpu[i]
            seq_lens[i] -= encoder_len  # 从序列长度中去除编码器长度

            if len(req.prefix_indices) < encoder_len:
                # NOTE: the encoder part should be considered as a whole
                # 编码器部分未缓存，需要单独分配：prefix 必须为 0
                assert len(req.prefix_indices) == 0
                # 从 input_ids 中跳过编码器 token
                input_ids[i] = input_ids[i][encoder_len:]
                # 将 out_cache_loc 中编码器对应位置分配给 encoder_out_cache_loc
                encoder_out_cache_loc.append(self.out_cache_loc[pt : pt + encoder_len])
                decoder_out_cache_loc.append(
                    self.out_cache_loc[pt + encoder_len : pt + req.extend_input_len]
                )
                # 同步更新 extend_lens 和 extend_num_tokens
                self.extend_lens[i] -= encoder_len
                self.extend_num_tokens -= encoder_len
            else:
                # 编码器部分已缓存，decoder 直接对应完整 extend_input_len
                decoder_out_cache_loc.append(
                    self.out_cache_loc[pt : pt + req.extend_input_len]
                )
                # 前缀长度中去除编码器 token 数
                self.prefix_lens[i] -= encoder_len

            pt += req.extend_input_len  # 推进指针

        # Reassign
        # 重新构建 input_ids 张量（去掉编码器 token 后）
        self.input_ids = torch.tensor(
            sum(input_ids, []), dtype=torch.int64, pin_memory=_pin
        ).to(self.device, non_blocking=True)
        # 重新构建 seq_lens 张量（去掉编码器长度后）
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, pin_memory=_pin).to(
            self.device, non_blocking=True
        )
        self.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)

        # 重新构建 decoder 的 out_cache_loc
        if not decoder_out_cache_loc:
            self.out_cache_loc = torch.zeros(0, dtype=torch.int64).to(
                self.device, non_blocking=True
            )
        else:
            self.out_cache_loc = torch.cat(decoder_out_cache_loc)

        # 构建编码器对应的 out_cache_loc
        if not encoder_out_cache_loc:
            self.encoder_out_cache_loc = torch.zeros(0, dtype=torch.int64).to(
                self.device, non_blocking=True
            )
        else:
            self.encoder_out_cache_loc = torch.cat(encoder_out_cache_loc)

        # 验证 out_cache_loc 长度等于 extend_num_tokens
        assert (
            len(self.out_cache_loc) == self.extend_num_tokens
        ), f"Expected {len(self.out_cache_loc)}, got {self.extend_num_tokens}"

        # 如果需要 logprob，也要同步裁剪 extend_input_logprob_token_ids
        if self.extend_input_logprob_token_ids is not None:
            new_token_ids_parts = []
            offset = 0
            for i, req in enumerate(self.reqs):
                encoder_len = self.encoder_lens_cpu[i]
                old_start_len = self.extend_logprob_start_lens[i]
                old_contribution = req.extend_input_len - old_start_len

                if len(req.prefix_indices) < encoder_len:
                    # 编码器 token 未缓存时，从 logprob token id 中去掉编码器部分
                    tokens_to_strip = max(0, encoder_len - old_start_len)
                    new_token_ids_parts.append(
                        self.extend_input_logprob_token_ids[
                            offset + tokens_to_strip : offset + old_contribution
                        ]
                    )
                    # 更新 logprob 起始长度
                    self.extend_logprob_start_lens[i] = max(
                        0, old_start_len - encoder_len
                    )
                else:
                    # 编码器部分已缓存，logprob token id 直接保留
                    new_token_ids_parts.append(
                        self.extend_input_logprob_token_ids[
                            offset : offset + old_contribution
                        ]
                    )

                offset += old_contribution

            if new_token_ids_parts:
                self.extend_input_logprob_token_ids = torch.cat(new_token_ids_parts)
            else:
                self.extend_input_logprob_token_ids = None

        # 同步更新各请求的 extend_input_len 和 logprob 起始位置
        for i, req in enumerate(self.reqs):
            encoder_len = self.encoder_lens_cpu[i]
            if encoder_len == 0:
                continue  # 无编码器 token，跳过
            if len(req.prefix_indices) < encoder_len:
                # 编码器未缓存，extend_input_len 需减去编码器长度
                req.extend_input_len -= encoder_len
                req.extend_logprob_start_len = max(
                    0, req.extend_logprob_start_len - encoder_len
                )
            # logprob 起始位置不得低于编码器长度
            req.logprob_start_len = max(req.logprob_start_len, encoder_len)

    def prepare_for_extend(self):
        # 准备 prefill（extend）阶段所需的批次张量，包括 input_ids、seq_lens、缓存分配等
        self.forward_mode = ForwardMode.EXTEND

        if self.is_dllm():
            # For DLLM, we use a separate forward mode
            # 扩散语言模型使用专属 forward mode
            self.forward_mode = ForwardMode.DLLM_EXTEND

        # Init tensors
        # 初始化各 per-request 列表，用于构建批次张量
        reqs = self.reqs
        # 每个请求的 extend 部分 token ID（去掉前缀已缓存部分）
        input_ids = [r.fill_ids[len(r.prefix_indices) :] for r in reqs]
        # 本批次总的 extend token 数
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        # 每个请求的完整序列长度
        seq_lens = [len(r.fill_ids) for r in reqs]
        # 每个请求的原始序列长度（取 fill_ids 和 origin_input_ids 的较大值）
        orig_seq_lens = [max(len(r.fill_ids), len(r.origin_input_ids)) for r in reqs]
        # 每个请求的前缀缓存长度
        prefix_lens = [len(r.prefix_indices) for r in reqs]
        # 每个请求的 extend 输入长度
        extend_lens = [r.extend_input_len for r in reqs]

        # For matryoshka embeddings
        # 如果模型支持 matryoshka 嵌入且有请求指定了维度，收集各请求的 embedding 维度
        if self.model_config.is_matryoshka and any(
            r.dimensions is not None for r in reqs
        ):
            self.dimensions = [
                r.dimensions if r.dimensions else self.model_config.hidden_size
                for r in reqs
            ]

        # OR across the batch so ForwardBatch matches a single fused forward; requests
        # that did not ask for PHS still skip attaching it in the output processor.
        # 批次中只要有一个请求要求返回池化隐藏状态，整批都走该路径
        self.return_pooled_hidden_states = any(
            r.return_pooled_hidden_states for r in reqs
        )

        # 收集所有有 token_type_ids 的请求（用于 BERT 类模型）
        token_type_ids = [
            r.token_type_ids for r in reqs if r.token_type_ids is not None
        ]

        # 构建 input_ids 张量（展平拼接）
        _pin = is_pin_memory_available(self.device)
        input_ids_tensor = torch.tensor(
            list(chain.from_iterable(input_ids)), dtype=torch.int64, pin_memory=_pin
        ).to(self.device, non_blocking=True)
        # 构建 seq_lens 张量
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int64, pin_memory=_pin).to(
            self.device, non_blocking=True
        )
        seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        # 构建原始序列长度张量（int32 以节省显存）
        orig_seq_lens_tensor = torch.tensor(
            orig_seq_lens, dtype=torch.int32, pin_memory=_pin
        ).to(self.device, non_blocking=True)

        # 构建 token_type_ids 张量（如果有）
        token_type_ids_tensor = None
        if len(token_type_ids) > 0:
            token_type_ids_tensor = torch.tensor(
                sum(token_type_ids, []), dtype=torch.int64, pin_memory=_pin
            ).to(self.device, non_blocking=True)

        # Set batch fields needed by alloc_for_extend
        # 在调用 alloc_for_extend 前设置批次字段
        self.prefix_lens = prefix_lens
        self.extend_lens = extend_lens
        self.seq_lens = seq_lens_tensor
        self.seq_lens_cpu = seq_lens_cpu
        self.extend_num_tokens = extend_num_tokens

        # Allocate memory
        # 分配 KV cache 内存，返回 out_cache_loc 和请求池索引
        out_cache_loc, req_pool_indices_tensor, req_pool_indices = alloc_for_extend(
            self
        )

        # Set fields
        # 初始化各 per-request 辅助列表
        input_embeds = []
        all_replace_embeds: List[torch.Tensor] = []
        all_replace_positions: List[int] = []
        has_replace_embeds = False
        input_id_pointer = 0
        input_id_lens = [len(input_id) for input_id in input_ids]
        extend_input_logprob_token_ids = []
        multimodal_inputs = []
        mamba_track_mask_cpu = []
        mamba_track_indices_cpu = []
        mamba_track_seqlens_cpu = []

        for i, (req, seq_len, pre_len) in enumerate(zip(reqs, seq_lens, prefix_lens)):
            req.req_pool_idx = req_pool_indices[i]  # 设置请求在池中的索引
            assert seq_len - pre_len == req.extend_input_len

            req.extend_batch_idx += 1  # 统计该请求参与的 extend 批次次数

            # update req-level memory management fields
            # 更新请求级别的 KV 缓存已提交/已分配长度
            req.kv_committed_len = seq_len
            req.kv_allocated_len = seq_len

            # If input_embeds are available, store them
            # 如果请求有自定义输入嵌入，切片对应 extend 部分
            if req.input_embeds is not None:
                # Slice to match extend_input_len — PrefillAdder truncates
                # fill_ids/extend_input_len on chunk overflow but not input_embeds.
                input_embeds.extend(
                    req.input_embeds[pre_len : pre_len + req.extend_input_len]
                )

            if req.positional_embed_overrides is not None:
                # Override positions are absolute in the full sequence.
                # Convert to extend-tensor coordinates by subtracting pre_len,
                # then skip any that fall within the cached prefix.
                # 处理位置嵌入覆盖：绝对位置转换为 extend 坐标，跳过前缀范围
                embeds_to_add = []
                for embed_idx, pos in enumerate(
                    req.positional_embed_overrides.positions
                ):
                    extend_pos = pos - pre_len
                    if extend_pos < 0 or extend_pos >= req.extend_input_len:
                        continue  # Outside current extend chunk, skip
                    embeds_to_add.append((embed_idx, input_id_pointer + extend_pos))
                if embeds_to_add:
                    has_replace_embeds = True
                    indices, positions = zip(*embeds_to_add)
                    all_replace_embeds.append(
                        req.positional_embed_overrides.embeds[list(indices)]
                    )
                    all_replace_positions.extend(positions)
            input_id_pointer += input_id_lens[i]

            multimodal_inputs.append(req.multimodal_inputs)

            # Only calculate cached_tokens once. Once retracted, the 'retracted_stain'
            # flag will always True
            # 只在第一次非撤销状态下统计缓存命中 token 数
            if not req.retracted_stain:
                new_cached = pre_len - req.already_computed
                req.cached_tokens += new_cached

                # Calculate detailed breakdown of cached tokens by source (for HiCache)
                # Only compute once on FIRST chunk - subsequent chunks in chunked prefill
                # would incorrectly count previously computed tokens as cache hits.
                # 仅在第一次处理时计算缓存 token 的分层来源（device/host/storage）
                if not req._cache_breakdown_computed:
                    # At this point, prefix_indices has been extended with host data
                    # via init_load_back in schedule_policy, so:
                    # - len(prefix_indices) = device_original + host_loaded
                    # - host_hit_length = total tokens from host cache (including storage-prefetched)
                    # - storage_hit_length = tokens loaded from storage backend (L3 hits)
                    # - device_portion = len(prefix_indices) - host_hit_length
                    #
                    # Storage hits are now tracked via scheduler after prefetch completes.
                    # storage_hit_length is set by scheduler.pop_prefetch_loaded_tokens()
                    host_total = req.host_hit_length  # host 层命中的 token 总数
                    # Clamp storage to host_total to handle edge cases
                    storage_portion = min(host_total, req.storage_hit_length)  # storage 层命中数
                    host_portion = host_total - storage_portion  # 纯 host 层命中数
                    device_portion = max(0, len(req.prefix_indices) - host_total)  # device 层命中数

                    req.cached_tokens_device = device_portion
                    req.cached_tokens_host = host_portion
                    req.cached_tokens_storage = storage_portion
                    req._cache_breakdown_computed = True  # 标记已计算，避免重复统计

                req.already_computed = seq_len  # 更新已计算长度
            req.is_retracted = False  # 清除撤销标记

            # 如果启用了 Mamba 额外缓冲区，准备 Mamba radix cache v2 相关信息
            if get_global_server_args().enable_mamba_extra_buffer():
                self._mamba_radix_cache_v2_req_prepare_for_extend(
                    req,
                    mamba_track_mask_cpu,
                    mamba_track_indices_cpu,
                    mamba_track_seqlens_cpu,
                )

            if self.return_logprob:
                # Find input logprob token ids.
                # First, find a global index within origin_input_ids and slide it by 1
                # to compute input logprobs. It is because you need the next token
                # to compute input logprobs. E.g., (chunk size 2)
                #
                # input_logprobs = [1, 2, 3, 4]
                # fill_ids = [1, 2]
                # extend_input_logprob_token_id = [2, 3]
                #
                # Note that it can also overflow. In this case, we pad it with 0.
                # input_logprobs = [1, 2, 3, 4]
                # fill_ids = [3, 4]
                # extend_input_logprob_token_id = [4, 0]
                # 计算 input logprob token ids：使用"下一个 token"作为 logprob 目标
                global_start_idx, global_end_idx = (
                    len(req.prefix_indices),
                    len(req.fill_ids),
                )
                if req.logprob_start_len == -1:
                    # -1 表示从原始输入末尾开始计算 logprob
                    logprob_start_len = len(req.origin_input_ids)
                else:
                    logprob_start_len = req.logprob_start_len
                # Apply logprob_start_len
                # 跳过 logprob_start_len 之前的 token
                if global_start_idx < logprob_start_len:
                    global_start_idx = logprob_start_len

                # 取 origin_input_ids 中偏移 +1 的 token 作为 logprob 目标
                logprob_token_ids = req.origin_input_ids[
                    global_start_idx + 1 : global_end_idx + 1
                ]
                extend_input_logprob_token_ids.extend(logprob_token_ids)

                # We will need req.extend_input_len - req.extend_logprob_start_len number of
                # tokens, and logprob_token_ids is for input logprob, so pad the rest of them by 0.
                # 末尾不足部分用 0 填充（如 token 超出 origin_input_ids 范围时）
                extend_input_logprob_token_ids.extend(
                    [0]
                    * (
                        req.extend_input_len
                        - req.extend_logprob_start_len
                        - len(logprob_token_ids)
                    )
                )

        if self.return_logprob:
            extend_input_logprob_token_ids = torch.tensor(
                extend_input_logprob_token_ids
            )
            # Clamp placeholder or out-of-range token IDs (e.g., multimodal hashes)
            # so they stay within the vocab boundary before being sent to GPU.
            # 将超出词表范围的 token id（如多模态 hash）截断到合法范围
            extend_input_logprob_token_ids.clamp_(0, self.model_config.vocab_size - 1)
        else:
            extend_input_logprob_token_ids = None

        # 如果有位置嵌入覆盖，构建替换嵌入张量和位置张量
        if has_replace_embeds:
            replace_embeds_tensor = torch.cat(all_replace_embeds, dim=0).to(
                self.device, non_blocking=True
            )
            replace_positions_tensor = torch.tensor(
                all_replace_positions, dtype=torch.long, device=self.device
            )
        else:
            replace_embeds_tensor = None
            replace_positions_tensor = None

        # 将所有准备好的数据写入批次字段
        self.input_ids = input_ids_tensor
        self.req_pool_indices = req_pool_indices_tensor
        self.orig_seq_lens = orig_seq_lens_tensor
        self.out_cache_loc = out_cache_loc
        # 如果有自定义输入嵌入，转换为 GPU 张量；否则为 None
        self.input_embeds = (
            torch.tensor(input_embeds, pin_memory=_pin).to(
                self.device, non_blocking=True
            )
            if input_embeds
            else None
        )
        self.replace_embeds = replace_embeds_tensor
        self.replace_positions = replace_positions_tensor
        # 将多模态输入中的视觉位置 ID / 可见帧计数转移到 GPU
        for mm_input in multimodal_inputs:
            if mm_input is None:
                continue
            if isinstance(mm_input.vision_position_ids, torch.Tensor):
                mm_input.vision_position_ids = mm_input.vision_position_ids.to(
                    self.device, non_blocking=True
                )
            if isinstance(mm_input.visible_frame_counts, torch.Tensor):
                mm_input.visible_frame_counts = mm_input.visible_frame_counts.to(
                    self.device, non_blocking=True
                )
        self.multimodal_inputs = multimodal_inputs
        self.token_type_ids = token_type_ids_tensor
        self.seq_lens_sum = sum(seq_lens)  # 序列长度总和，用于统计

        # Pre-compute delimiter indices as CPU tensors for MIS.
        # When --enable-mis is on, every request in the batch is expected to
        # carry delimiter indices (the score endpoint always produces MIS-structured
        # requests). Consumers index this list without None-checking.
        # 如果启用了 Multi-Item Scoring（MIS），收集每个请求的分隔符索引
        if get_global_server_args().enable_mis and any(
            r.multi_item_delimiter_indices is not None for r in reqs
        ):
            assert all(
                r.multi_item_delimiter_indices is not None for r in reqs
            ), "MIS batch must have delimiter indices on every request"
            self.multi_item_delimiter_indices = [
                torch.tensor(r.multi_item_delimiter_indices, dtype=torch.int64)
                for r in reqs
            ]
        else:
            self.multi_item_delimiter_indices = None

        # 如果需要 logprob，收集 top_logprobs_num 和 token_ids_logprob
        if self.return_logprob:
            self.top_logprobs_nums = [r.top_logprobs_num for r in reqs]
            self.token_ids_logprobs = [r.token_ids_logprob for r in reqs]

        # 保存各请求的 logprob 起始长度和 logprob token id 列表
        self.extend_logprob_start_lens = [r.extend_logprob_start_len for r in reqs]
        self.extend_input_logprob_token_ids = extend_input_logprob_token_ids

        # 如果启用了 Mamba 额外缓冲区，将 track 信息转为 GPU 张量
        if get_global_server_args().enable_mamba_extra_buffer():
            self.mamba_track_indices = torch.tensor(
                mamba_track_indices_cpu,
                dtype=torch.int64,
                device=self.device,
            )
            self.mamba_track_mask = torch.tensor(
                mamba_track_mask_cpu,
                dtype=torch.bool,
                device=self.device,
            )
            self.mamba_track_seqlens = torch.tensor(
                mamba_track_seqlens_cpu,
                dtype=torch.int64,
                device=self.device,
            )

        # 如果是 encoder-decoder 模型，准备编码器信息
        if self.model_config.is_encoder_decoder:
            self.prepare_encoder_info_extend(input_ids, seq_lens)

        # Build sampling info
        # 构建采样信息（温度、top-p、top-k 等参数）
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    def _mamba_radix_cache_v2_req_prepare_for_extend(
        self,
        req: Req,
        mamba_track_mask_cpu: List[bool],
        mamba_track_indices_cpu: List[int],
        mamba_track_seqlens_cpu: List[int],
    ):
        # 为 Mamba Radix Cache v2 准备请求的 track 信息
        def _force_track_h(i: int) -> int:
            # 强制从 h（隐藏状态）而非 last_recurrent_state 检索 Mamba 状态
            assert i % FLA_CHUNK_SIZE == 0
            # There are 3 cases for mamba_track_seqlen passed to mamba_track_seqlens_cpu:
            # 1) aligned with FLA_CHUNK_SIZE-> retrieve from last_recurrent_state
            #    a) is the last position -> retrieve from last_recurrent_state
            #    b) is NOT the last position -> retrieve from h
            # 2) unaligned with FLA_CHUNK_SIZE -> retrieve from h
            # Currently, the math calculation only supports case 1a and 2. So for 1b, we need to add 1
            # to force the math calculation to retrieve the correct mamba state from h.
            return i + 1  # +1 使计算跳到"从 h 检索"的分支

        mamba_cache_chunk_size = get_global_server_args().mamba_cache_chunk_size
        # 只有 extend_input_len >= chunk_size 时才需要追踪 Mamba 状态
        mask = req.extend_input_len >= mamba_cache_chunk_size
        mamba_track_mask_cpu.append(mask)
        # 追踪当前激活的 ping-pong buffer 索引
        mamba_track_indices_cpu.append(
            req.mamba_ping_pong_track_buffer[req.mamba_next_track_idx].item()
        )
        mamba_track_seqlen = -1  # -1 表示无需追踪
        if mask:
            # mamba_track_seqlen is used to calculate the indices to track in
            # hybrid_linear_attn_backend's _init_track_ssm_indices. Due to the
            # fact that the ssm state between aligned and non-aligned are retrieved differently,
            # if 1) last pos and 2) is aligned, then retrieved from the last_recurrent_state,
            # otherwise retrieved from h (i.e. unaligned).
            # We need to pass the non-aligned seqlen to the calculation. Even though
            # we pass in mamba_track_seqlen, the actual tracked seqlen is mamba_last_track_seqlen.
            # 计算当前 extend 结束时的总序列长度
            mamba_track_seqlen = len(req.prefix_indices) + req.extend_input_len

            # mamba_track_seqlen_aligned/mamba_last_track_seqlen is actual tracked seqlen. Used to pass to
            # mamba radix cache to track which seqlen this mamba state should store at.
            # 对齐到 mamba_cache_chunk_size 的追踪序列长度
            mamba_track_seqlen_aligned = (
                len(req.prefix_indices)
                + (req.extend_input_len // mamba_cache_chunk_size)
                * mamba_cache_chunk_size
            )

            # mamba_track_fla_chunk_aligned is the aligned seqlen based on FLA_CHUNK_SIZE
            # If mamba_track_fla_chunk_aligned != mamba_track_seqlen_aligned, which can be true when
            # page_size > FLA_CHUNK_SIZE, we need to force the math calculation to retrieve the correct mamba state from h
            # by _force_track_h()
            # 基于 FLA_CHUNK_SIZE 的对齐序列长度
            mamba_track_fla_chunk_aligned = (
                len(req.prefix_indices)
                + (req.extend_input_len // FLA_CHUNK_SIZE) * FLA_CHUNK_SIZE
            )
            if mamba_track_fla_chunk_aligned != mamba_track_seqlen_aligned:
                # We want to track mamba_track_seqlen_aligned, and it's not the last position,
                # so we need to add 1 to the seqlen to retrieve the correct mamba state from h.
                # 两者不一致时，强制走"从 h 检索"路径
                mamba_track_seqlen = _force_track_h(mamba_track_seqlen_aligned)

            # 切换到 ping-pong 的另一个缓冲区索引
            req.mamba_next_track_idx = (
                self.req_to_token_pool.get_mamba_ping_pong_other_idx(
                    req.mamba_next_track_idx
                )
            )
            if req.mamba_branching_seqlen is not None:
                # track branching point in this forward if the branching point
                # is within the current extend batch.
                # 如果有分支点（用于 speculative decoding / beam search），检查是否在本批内
                branching_seqlen_aligned_mask = (
                    req.mamba_branching_seqlen - len(req.prefix_indices)
                ) % mamba_cache_chunk_size == 0
                if (
                    req.mamba_branching_seqlen > len(req.prefix_indices)
                    and req.mamba_branching_seqlen < mamba_track_seqlen
                    and branching_seqlen_aligned_mask
                ):
                    # We want to track mamba_track_seqlen_aligned, and it's not the last position,
                    # so we need to add 1 to the seqlen to retrieve the correct mamba state from h.
                    # See _force_track_h() for more details.
                    # 分支点在当前 extend 范围内，追踪分支点的 Mamba 状态
                    mamba_track_seqlen = _force_track_h(req.mamba_branching_seqlen)
                    mamba_track_seqlen_aligned = req.mamba_branching_seqlen
            req.mamba_last_track_seqlen = mamba_track_seqlen_aligned  # 保存最终追踪的对齐长度
        mamba_track_seqlens_cpu.append(mamba_track_seqlen)

    def prepare_for_split_prefill(self):
        # 准备 split prefill 模式：先执行 extend 准备，再切换为 SPLIT_PREFILL forward mode
        self.prepare_for_extend()
        # For split prefill, we need to set the forward mode to SPLIT_PREFILL
        self.forward_mode = ForwardMode.SPLIT_PREFILL

    def mix_with_running(self, running_batch: "ScheduleBatch"):
        # 将 prefill 批次与正在 decode 的批次混合，使用 MIXED forward mode
        self.forward_mode = ForwardMode.MIXED
        running_bs = running_batch.batch_size()

        # 对 running 批次的每个请求更新 fill_ids 和 extend_input_len
        for req in running_batch.reqs:
            req.fill_ids = req.origin_input_ids + req.output_ids
            req.set_extend_input_len(1)  # decode 阶段每次只处理 1 个 token

        # 拼接两批次的 input_ids 和 out_cache_loc
        input_ids = torch.cat([self.input_ids, running_batch.input_ids])
        out_cache_loc = torch.cat([self.out_cache_loc, running_batch.out_cache_loc])

        self.merge_batch(running_batch)
        self.input_ids = input_ids
        self.out_cache_loc = out_cache_loc

        # For overlap scheduler, the output_ids has one step delay
        # overlap 调度器中 output_ids 有 1 步延迟，delta 用于调整前缀长度
        delta = 0 if self.enable_overlap else -1

        # NOTE: prefix_indices is what has been cached, but we don't cache each decode step
        # 计算 running 批次中各请求的已缓存前缀长度
        self.prefix_lens.extend(
            [
                len(r.origin_input_ids) + len(r.output_ids) + delta
                for r in running_batch.reqs
            ]
        )
        self.extend_lens.extend([1] * running_bs)  # decode 阶段每次 extend 1 个 token
        self.extend_num_tokens += running_bs  # 累计 extend token 总数
        # TODO (lianmin): Revisit this. It should be seq_len - 1
        self.extend_logprob_start_lens.extend([0] * running_bs)  # logprob 起始长度置 0
        self.is_prefill_only = False  # 混合批次不再是纯 prefill

    def new_tokens_required_next_decode(
        self, selected_indices: Optional[List[int]] = None
    ):
        # 估算下一次 decode 步骤需要分配的新 KV cache token 数
        page_size = self.token_to_kv_pool_allocator.page_size
        requests = (
            self.reqs
            if selected_indices is None
            else [self.reqs[i] for i in selected_indices]
        )

        if self.spec_algorithm.is_none():
            # 普通 decode：只有跨页时才需要新分配（每 page_size tokens 分配一次）
            new_pages = sum(1 for r in requests if r.kv_committed_len % page_size == 0)
            return new_pages * page_size

        if self.is_spec_v2:
            # 投机解码 v2：使用紧凑估算方法
            return self._new_tokens_required_next_decode_spec_v2(requests, page_size)

        # 投机解码 v1：根据 num_steps、topk 和 draft_tokens 估算
        server_args = get_global_server_args()
        len_per_topk = server_args.speculative_num_steps or 1  # 每个 topk 分支的步数
        spec_topk = server_args.speculative_eagle_topk or 1    # EAGLE topk 宽度
        spec_tokens = server_args.speculative_num_draft_tokens  # draft token 数

        if page_size > 1 and spec_topk > 1:
            # last partial page and ceil alignment
            # 有 page size 且有 topk 时，需要额外对齐以覆盖最后不满页
            len_per_topk = ceil_align(len_per_topk + page_size, page_size)
            spec_tokens = ceil_align(spec_tokens, page_size)
        elif page_size > 1:
            # only page alignment
            # 只有 page size，直接对齐
            len_per_topk = ceil_align(len_per_topk, page_size)
            spec_tokens = ceil_align(spec_tokens, page_size)

        # 取 len_per_topk * spec_topk 和 spec_tokens 的最大值乘以请求数
        num_tokens = max(len_per_topk * spec_topk, spec_tokens) * len(requests)
        return num_tokens

    def _new_tokens_required_next_decode_spec_v2(self, requests, page_size):
        """Tight estimate matching eagle_info_v2.prepare_for_decode allocation."""
        # 与 eagle_info_v2.prepare_for_decode 分配逻辑对齐的紧凑估算
        from sglang.srt.managers.utils import get_alloc_len_per_decode

        alloc_len = get_alloc_len_per_decode()  # 每次 decode 预分配的 token 数
        total = 0
        for r in requests:
            # 计算该请求下一 decode 步还需额外分配的 token 数（超出已分配部分）
            x = max(0, r.kv_committed_len + 2 * alloc_len - r.kv_allocated_len)
            cur = r.kv_allocated_len
            nxt = cur + x
            # 按 page_size 对齐后的增量
            total += ceil_align(nxt, page_size) - ceil_align(cur, page_size)
        return total

    def check_decode_mem(self, selected_indices: Optional[List[int]] = None):
        # 检查是否有足够内存执行下一次 decode
        num_tokens = self.new_tokens_required_next_decode(selected_indices)
        # 先尝试从 radix tree 中驱逐缓存以腾出空间
        evict_from_tree_cache(self.tree_cache, num_tokens)
        return self.token_to_kv_pool_allocator.available_size() >= num_tokens

    def retract_all(self, server_args: ServerArgs):
        # 撤销批次中所有请求并返回被撤销的请求列表
        retracted_reqs = self.reqs
        for idx in range(len(self.reqs)):
            self.release_req(idx, len(self.reqs) - idx, server_args)

        self.filter_batch(retracted_reqs)
        return retracted_reqs

    def retract_decode(
        self, server_args: ServerArgs
    ) -> Tuple[List[Req], float, List[Req]]:
        """Retract the decoding requests when there is not enough memory."""
        # decode 内存不足时，按策略撤销部分请求以释放内存
        sorted_indices = list(range(len(self.reqs)))

        # TODO(lsyin): improve retraction policy for radix cache
        # For spec decoding, filter_batch API can only filter
        # requests from the back, so we can only retract from the back.
        # TODO(sang): Clean up finish path and support better retract
        # policy.
        if not server_args.speculative_algorithm:
            # 普通 decode：按输出最长、输入最短排序，优先撤销"最占内存"的请求
            sorted_indices.sort(
                key=lambda i: (
                    len(self.reqs[i].output_ids),
                    -len(self.reqs[i].origin_input_ids),
                ),
                reverse=True,
            )

        retracted_reqs = []
        first_iter = True
        while first_iter or (
            not self.check_decode_mem(selected_indices=sorted_indices)
        ):
            if len(sorted_indices) == 1:
                # Always keep at least one request
                # 至少保留一个请求，避免全部撤销
                break

            first_iter = False
            idx = sorted_indices.pop()  # 从尾部弹出（最应被撤销的请求）
            req = self.reqs[idx]
            retracted_reqs.append(req)
            # release memory and don't insert into the tree because we need the space instantly
            # 立即释放内存，不插入 radix tree（需要立即使用该空间）
            self.release_req(idx, len(sorted_indices), server_args)

        reqs_to_abort: List[Req] = []
        if len(sorted_indices) <= 1 and not self.check_decode_mem(
            selected_indices=sorted_indices
        ):
            # Even the last remaining request cannot fit in memory.
            # Instead of crashing the scheduler, gracefully abort it.
            # 即使最后一个请求也无法放入内存，优雅地 abort 而非 crash
            last_idx = sorted_indices.pop()
            last_req = self.reqs[last_idx]
            last_req.to_finish = FINISH_ABORT(
                "Out of memory even after retracting all other requests "
                "in the decode batch. Aborting the last request.",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            reqs_to_abort.append(last_req)
            self.release_req(last_idx, 0, server_args)
            logger.warning(
                "retract_decode: aborted last request %s due to OOM", last_req.rid
            )

        # 过滤批次，只保留未被撤销的请求
        self.filter_batch(keep_indices=sorted_indices)

        # Reqs in batch are filtered
        # 计算新的 decode 完成比例估算值，用于动态调整调度策略
        total_decoded_tokens = sum(len(r.output_ids) for r in self.reqs)
        total_max_new_tokens = sum(r.sampling_params.max_new_tokens for r in self.reqs)

        new_estimate_ratio = (
            total_decoded_tokens
            + envs.SGLANG_RETRACT_DECODE_STEPS.get() * len(self.reqs)
        ) / (
            total_max_new_tokens + 1
        )  # avoid zero division
        new_estimate_ratio = min(1.0, new_estimate_ratio)  # 上限为 1.0

        return retracted_reqs, new_estimate_ratio, reqs_to_abort

    def release_req(self, idx: int, remaing_req_count: int, server_args: ServerArgs):
        # 释放一个请求占用的资源并重置其状态，为内存回收做准备
        req = self.reqs[idx]

        # 如果启用了 HiSparse，通知 coordinator 撤销该请求
        if self.hisparse_coordinator is not None:
            self.hisparse_coordinator.retract_req(req)

        # 在 disaggregation decode 节点，将 KV cache 卸载回内存
        if server_args.disaggregation_mode == "decode":
            req.offload_kv_cache(
                self.req_to_token_pool, self.token_to_kv_pool_allocator
            )
        # TODO (csy): for preempted requests, we may want to insert into the tree
        # 释放该请求的 KV cache，不插入 radix tree（立即回收空间）
        release_kv_cache(req, self.tree_cache, is_insert=False)
        # NOTE(lsyin): we should use the newly evictable memory instantly.
        # 立即驱逐 tree cache 以利用刚释放的空间
        num_tokens = remaing_req_count * envs.SGLANG_RETRACT_DECODE_STEPS.get()
        evict_from_tree_cache(self.tree_cache, num_tokens)

        req.reset_for_retract()  # 重置请求状态，使其可重新入队

    def prepare_encoder_info_decode(self):
        # Reset the encoder cached status
        # decode 阶段所有请求的编码器部分均视为已缓存
        self.encoder_cached = [True] * len(self.reqs)

    def prepare_for_idle(self):
        # 准备空闲批次（无请求时），设置所有张量为空
        self.forward_mode = ForwardMode.IDLE
        self.input_ids = torch.empty(0, dtype=torch.int64, device=self.device)
        self.seq_lens = torch.empty(0, dtype=torch.int64, device=self.device)
        self.seq_lens_cpu = torch.empty(0, dtype=torch.int64)
        self.orig_seq_lens = torch.empty(0, dtype=torch.int32, device=self.device)
        self.out_cache_loc = torch.empty(0, dtype=torch.int64, device=self.device)
        self.req_pool_indices = torch.empty(0, dtype=torch.int64, device=self.device)
        self.seq_lens_sum = 0
        self.extend_num_tokens = 0
        # 构建空的采样信息
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    @property
    def is_spec_v2(self):
        # FIXME: finally deprecate is_spec_v2
        # 判断是否为 speculative decoding v2（overlap + spec 算法）
        ret = self.enable_overlap and not self.spec_algorithm.is_none()
        assert not ret or self.spec_algorithm.supports_spec_v2()
        return ret

    def prepare_for_decode(self):
        # 准备 decode 阶段所需的批次张量（input_ids 为上一步输出的 token）
        self.forward_mode = ForwardMode.DECODE
        bs = len(self.reqs)
        # Decode embeds the last output token via embed_tokens; clear the stale
        # prefill-time tensor so it doesn't leak into ForwardBatch.
        # 清除 prefill 阶段遗留的 input_embeds，decode 不使用自定义嵌入
        self.input_embeds = None

        # Clear context parallel metadata - CP is only for prefill, not decode
        # 清除 context parallel 元数据（仅 prefill 阶段使用）
        if hasattr(self, "attn_cp_metadata") and self.attn_cp_metadata is not None:
            self.attn_cp_metadata = None

        if self.is_spec_v2:
            # TODO(spec-v2): all spec v2 should go through this path
            # 投机解码 v2：由 EagleDraftInput 负责准备 decode 批次
            draft_input: EagleDraftInput = self.spec_info
            draft_input.prepare_for_decode(self)

        if not self.spec_algorithm.is_none():
            # if spec decoding is used, the decode batch is prepared inside
            # `forward_batch_speculative_generation` after running draft models.
            # 投机解码 v1：decode 批次在 draft model 运行后才会准备，此处直接返回
            return

        if self.sampling_info.penalizer_orchestrator.is_required:
            if self.enable_overlap:
                # TODO: this can be slow, optimize this.
                # overlap 模式下，从 output_ids 的延迟 token 中收集惩罚累计值
                delayed_output_ids = torch.tensor(
                    [
                        (
                            req.output_ids[-1]
                            if len(req.output_ids)
                            else req.origin_input_ids[-1]
                        )
                        for req in self.reqs
                    ],
                    dtype=torch.int64,
                    device=self.device,
                )
                self.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                    delayed_output_ids
                )
            else:
                # 非 overlap 模式：直接用当前 output_ids 更新惩罚累计值
                self.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                    self.output_ids.to(torch.int64)
                )

        # Update fields
        # 将上一步的 output_ids 作为本步的 input_ids
        self.input_ids = self.output_ids
        self.output_ids = None

        # encoder-decoder 模型需要更新编码器缓存状态
        if self.model_config.is_encoder_decoder:
            self.prepare_encoder_info_decode()

        # Allocate memory
        # 为本轮 decode 分配 KV cache（每个请求 1 个 token）
        self.out_cache_loc = alloc_for_decode(self, token_per_req=1)

        # Update req-level memory management fields
        # 更新各请求的 KV 已提交/已分配长度
        for req in self.reqs:
            req.decode_batch_idx += 1
            req.kv_committed_len += 1
            req.kv_allocated_len += 1

        # Update seq_lens after allocation
        # 更新序列长度（每次 decode 增加 1）
        if self.enable_overlap:
            # Do not use in-place operations in the overlap mode
            # overlap 模式不使用原地操作（避免异步冲突）
            self.seq_lens = self.seq_lens + 1
            self.seq_lens_cpu = self.seq_lens_cpu + 1
            self.orig_seq_lens = self.orig_seq_lens + 1
        else:
            # A faster in-place version
            # 非 overlap 模式使用更快的原地操作
            self.seq_lens.add_(1)
            self.seq_lens_cpu.add_(1)
            self.orig_seq_lens.add_(1)
        self.seq_lens_sum += bs  # 序列长度总和增加 bs

        if self.hisparse_coordinator is not None:
            # HiSparse 需要将最后一个 token 位置映射到对应的缓冲区
            self.hisparse_coordinator.map_last_loc_to_buffer(
                self.seq_lens,
                self.out_cache_loc,
                self.req_pool_indices,
                self.seq_lens_cpu,
            )

        if get_global_server_args().enable_mamba_extra_buffer():
            if len(self.reqs) == 0:
                # 无请求时初始化空的 track indices
                self.mamba_track_indices = torch.empty(
                    (0,), dtype=torch.int64, device=self.device
                )
            else:
                # already on device
                # 收集所有请求的 ping-pong buffer，通过 gather 获取当前激活的 buffer 索引
                all_buffers = torch.stack(
                    [req.mamba_ping_pong_track_buffer for req in self.reqs]
                )
                idx = (
                    torch.tensor(
                        [req.mamba_next_track_idx for req in self.reqs],
                        dtype=torch.int64,
                        pin_memory=True,
                    )
                    .unsqueeze(1)
                    .to(device=all_buffers.device, non_blocking=True)
                )
                self.mamba_track_indices = (
                    torch.gather(all_buffers, 1, idx).squeeze(1).to(torch.int64)
                )

            # async H2D
            # 根据序列长度和追踪间隔判断哪些请求需要追踪 Mamba 状态（异步 H2D 传输）
            self.mamba_track_mask = (
                (self.seq_lens_cpu % get_global_server_args().mamba_track_interval == 0)
                .pin_memory()
                .to(device=self.device, non_blocking=True)
            )

    def maybe_wait_verify_done(self):
        # 如果是 spec v2，等待验证阶段完成（CUDA event 同步）
        if self.is_spec_v2:
            draft_input: EagleDraftInput = self.spec_info
            if draft_input.verify_done is not None:
                draft_input.verify_done.synchronize()

    def filter_batch(
        self,
        chunked_req_to_exclude: Optional[Union[Req, List[Req]]] = None,
        keep_indices: Optional[List[int]] = None,
        # FIXME(lsyin): deprecate this API after spec v1 is deprecated
        v1_spec_info_filtered: Optional[bool] = False,
    ):
        # 过滤批次，移除已完成或需排除的请求
        # FIXME(lsyin): used here to get the correct seq_lens
        # The batch has been launched but we need it verified to get correct next batch info
        self.maybe_wait_verify_done()  # 确保 spec v2 验证完成

        if keep_indices is None:
            # 将 chunked_req_to_exclude 归一化为列表
            if isinstance(chunked_req_to_exclude, Req):
                chunked_req_to_exclude = [chunked_req_to_exclude]
            elif chunked_req_to_exclude is None:
                chunked_req_to_exclude = []
            # 保留未完成且不在排除列表中的请求
            keep_indices = [
                i
                for i in range(len(self.reqs))
                if not self.reqs[i].finished()
                and self.reqs[i] not in chunked_req_to_exclude
            ]

        if keep_indices is None or len(keep_indices) == 0:
            # Filter out all requests
            # 全部过滤，清空批次
            self.reqs = []
            return

        if len(keep_indices) == len(self.reqs):
            # No need to filter
            # 无需过滤，直接返回
            return

        # 构建 keep_indices 的 GPU 张量（用于张量索引）
        keep_indices_device = torch.tensor(
            keep_indices,
            dtype=torch.int64,
            pin_memory=is_pin_memory_available(self.device),
        ).to(self.device, non_blocking=True)

        # encoder-decoder 模型需要同步过滤编码器信息
        if self.model_config.is_encoder_decoder:
            self.encoder_lens = self.encoder_lens[keep_indices_device]
            self.encoder_lens_cpu = [self.encoder_lens_cpu[i] for i in keep_indices]

        # 过滤请求列表和各批次字段
        self.reqs = [self.reqs[i] for i in keep_indices]
        if self.multimodal_inputs is not None:
            self.multimodal_inputs = [self.multimodal_inputs[i] for i in keep_indices]
        self.req_pool_indices = self.req_pool_indices[keep_indices_device]
        self.seq_lens = self.seq_lens[keep_indices_device]
        self.seq_lens_cpu = self.seq_lens_cpu[keep_indices]
        self.orig_seq_lens = self.orig_seq_lens[keep_indices_device]
        self.out_cache_loc = None  # 过滤后 out_cache_loc 失效，置 None
        self.seq_lens_sum = self.seq_lens.sum().item()  # 重新计算总序列长度

        # 过滤 output_ids
        if self.output_ids is not None:
            self.output_ids = self.output_ids[keep_indices_device]

        # Mamba track 相关字段过滤后重置
        self.mamba_track_indices = None
        self.mamba_track_mask = None
        self.mamba_track_seqlens = None
        # 重新计算是否需要 logprob
        self.return_logprob = any(req.return_logprob for req in self.reqs)
        if self.return_logprob:
            self.top_logprobs_nums = [self.top_logprobs_nums[i] for i in keep_indices]
            self.token_ids_logprobs = [self.token_ids_logprobs[i] for i in keep_indices]
        else:
            self.top_logprobs_nums = None
            self.token_ids_logprobs = None

        # 重新计算流式输出和语法约束标志
        self.has_stream = any(req.stream for req in self.reqs)
        self.has_grammar = any(req.grammar for req in self.reqs)

        # 过滤采样信息
        self.sampling_info.filter_batch(keep_indices, keep_indices_device)
        # NOTE: spec_info filtered before batch filtering only happens in:
        # - Spec v1's verify phase
        # - Only for decode batch (running_batch)
        has_been_filtered = v1_spec_info_filtered and not self.is_spec_v2

        if self.spec_info:
            # 过滤投机解码信息
            self.spec_info.filter_batch(
                new_indices=keep_indices_device,
                has_been_filtered=has_been_filtered,
            )

    def merge_batch(self, other: "ScheduleBatch"):
        # In the regular scheduler path:
        # 1) self is always prefill, whose seq_lens is not a future
        # 2) other is always decode, which is finished in previous step
        # so verify_done is already synced and this is a no-op.
        # In disagg decode + overlap, merge_batch can be called before
        # filter_batch, so running_batch.seq_lens may still be a forward_stream
        # future. Synchronize here to avoid a cross-stream data race.
        # 将另一个批次合并到当前批次（通常是 prefill 与 decode 批次的合并）
        self.maybe_wait_verify_done()  # 确保 spec v2 验证完成后再合并

        # Penalizer orchestrator must be merged before Batch.reqs is merged. This is because
        # orchestrator.merge() depends on Batch.reqs during preparation of each penalizers, so it
        # needs to be called with pre-merged Batch.reqs.
        # 先合并惩罚器（依赖 Batch.reqs 的当前状态）
        self.sampling_info.merge_batch(other.sampling_info)

        # Encoder-decoder infos
        # 合并 encoder-decoder 相关信息
        if self.model_config.is_encoder_decoder:
            self.encoder_lens = torch.cat([self.encoder_lens, other.encoder_lens])
            self.encoder_lens_cpu.extend(other.encoder_lens_cpu)
        # 拼接请求池索引、序列长度等张量
        self.req_pool_indices = torch.cat(
            [self.req_pool_indices, other.req_pool_indices]
        )
        self.seq_lens = torch.cat([self.seq_lens, other.seq_lens])
        self.seq_lens_cpu = torch.cat([self.seq_lens_cpu, other.seq_lens_cpu])
        self.orig_seq_lens = torch.cat([self.orig_seq_lens, other.orig_seq_lens])
        self.out_cache_loc = None  # 合并后 out_cache_loc 失效
        self.seq_lens_sum += other.seq_lens_sum
        if self.output_ids is not None:
            self.output_ids = torch.cat([self.output_ids, other.output_ids])
        # Mamba track 字段合并后置 None，等待后续重新计算
        self.mamba_track_indices = None
        self.mamba_track_mask = None
        self.mamba_track_seqlens = None
        # 合并 logprob 相关字段（需考虑两个批次各自是否开启 logprob）
        if self.return_logprob and other.return_logprob:
            self.top_logprobs_nums.extend(other.top_logprobs_nums)
            self.token_ids_logprobs.extend(other.token_ids_logprobs)
        elif self.return_logprob:
            # self 有 logprob 但 other 没有，用 0 和 None 填充
            self.top_logprobs_nums.extend([0] * len(other.reqs))
            self.token_ids_logprobs.extend([None] * len(other.reqs))
        elif other.return_logprob:
            # other 有 logprob 但 self 没有，反向填充
            self.top_logprobs_nums = [0] * len(self.reqs) + other.top_logprobs_nums
            self.token_ids_logprobs = [None] * len(self.reqs) + other.token_ids_logprobs
        self.reqs.extend(other.reqs)  # 合并请求列表
        if self.multimodal_inputs is not None:
            self.multimodal_inputs.extend(other.multimodal_inputs)

        # 按 OR 合并各布尔标志
        self.return_logprob |= other.return_logprob
        self.has_stream |= other.has_stream
        self.has_grammar |= other.has_grammar
        self.return_hidden_states |= other.return_hidden_states
        # 混合批次不再是纯 prefill（需要两者都是 prefill 才算）
        self.is_prefill_only = self.is_prefill_only and other.is_prefill_only

        if self.spec_info:
            # 合并投机解码信息
            self.spec_info.merge_batch(other.spec_info)

    def get_model_worker_batch(
        self, seq_lens_cpu_cache: Optional[torch.Tensor] = None
    ) -> ModelWorkerBatch:
        # 将 ScheduleBatch 转换为 ModelWorkerBatch，传递给模型执行层
        if self.forward_mode.is_decode_or_idle():
            # decode 或 idle 阶段无 extend 信息
            extend_seq_lens = extend_prefix_lens = extend_logprob_start_lens = None
        else:
            # prefill/extend 阶段需传递 extend 相关信息
            extend_seq_lens = self.extend_lens
            extend_prefix_lens = self.prefix_lens
            extend_logprob_start_lens = self.extend_logprob_start_lens

        if self.sampling_info:
            if self.has_grammar:
                # 有语法约束时，将各请求的 grammar 对象列表传入
                self.sampling_info.grammars = [req.grammar for req in self.reqs]
            else:
                self.sampling_info.grammars = None

        # 优先使用外部传入的 seq_lens_cpu_cache（用于 overlap 调度）
        seq_lens_cpu = (
            seq_lens_cpu_cache if seq_lens_cpu_cache is not None else self.seq_lens_cpu
        )

        # 构建并返回 ModelWorkerBatch，包含所有模型执行所需字段
        return ModelWorkerBatch(
            forward_mode=self.forward_mode,
            input_ids=self.input_ids,
            req_pool_indices=self.req_pool_indices,
            seq_lens=self.seq_lens,
            orig_seq_lens=self.orig_seq_lens,
            out_cache_loc=self.out_cache_loc,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_sum=self.seq_lens_sum,
            return_logprob=self.return_logprob,
            top_logprobs_nums=self.top_logprobs_nums,
            token_ids_logprobs=self.token_ids_logprobs,
            global_num_tokens=self.global_num_tokens,
            global_num_tokens_for_logprob=self.global_num_tokens_for_logprob,
            is_extend_in_batch=self.is_extend_in_batch,
            all_extend_in_batch=self.all_extend_in_batch,
            can_run_dp_cuda_graph=self.can_run_dp_cuda_graph,
            tbo_split_seq_index=self.tbo_split_seq_index,
            global_forward_mode=self.global_forward_mode,
            extend_num_tokens=self.extend_num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_logprob_start_lens=extend_logprob_start_lens,
            multimodal_inputs=self.multimodal_inputs,
            encoder_cached=self.encoder_cached,
            encoder_lens=self.encoder_lens,
            encoder_lens_cpu=self.encoder_lens_cpu,
            encoder_out_cache_loc=self.encoder_out_cache_loc,
            lora_ids=[req.lora_id for req in self.reqs],
            sampling_info=self.sampling_info,
            input_embeds=self.input_embeds,
            replace_embeds=self.replace_embeds,
            replace_positions=self.replace_positions,
            ne_token_table=self.ne_token_table,
            token_type_ids=self.token_type_ids,
            spec_algorithm=self.spec_algorithm,
            spec_info=self.spec_info,
            hicache_consumer_index=self.hicache_consumer_index,
            # 根据是否需要返回隐藏状态确定 capture_hidden_mode
            capture_hidden_mode=(
                CaptureHiddenMode.FULL
                if self.return_hidden_states
                else (
                    getattr(
                        self.spec_info, "capture_hidden_mode", CaptureHiddenMode.NULL
                    )
                    if self.spec_info
                    else CaptureHiddenMode.NULL
                )
            ),
            extend_input_logprob_token_ids=self.extend_input_logprob_token_ids,
            is_prefill_only=self.is_prefill_only,
            multi_item_delimiter_indices=self.multi_item_delimiter_indices,
            dimensions=self.dimensions,
            return_pooled_hidden_states=self.return_pooled_hidden_states,
            dllm_block_offsets=[req.dllm_block_offset for req in self.reqs],
            dllm_config=self.dllm_config,
            reqs=self.reqs,
            has_grammar=self.has_grammar,
            mamba_track_indices=self.mamba_track_indices,
            mamba_track_mask=self.mamba_track_mask,
            mamba_track_seqlens=self.mamba_track_seqlens,
        )

    def copy(self):
        # Only contain fields that will be used by process_batch_result.
        # Shallow-copy the reqs list so that in-place mutations (filter_batch,
        # merge_batch) on the original don't corrupt this snapshot.
        # 创建批次的浅拷贝，仅包含 process_batch_result 所需字段
        return ScheduleBatch(
            reqs=self.reqs[:],  # 浅拷贝请求列表，避免原批次变更影响此快照
            req_to_token_pool=self.req_to_token_pool,
            req_pool_indices=self.req_pool_indices,
            model_config=self.model_config,
            forward_mode=self.forward_mode,
            out_cache_loc=self.out_cache_loc,
            return_logprob=self.return_logprob,
            decoding_reqs=self.decoding_reqs,
            spec_algorithm=self.spec_algorithm,
            global_num_tokens=self.global_num_tokens,
            global_num_tokens_for_logprob=self.global_num_tokens_for_logprob,
            can_run_dp_cuda_graph=self.can_run_dp_cuda_graph,
            all_extend_in_batch=self.all_extend_in_batch,
            is_extend_in_batch=self.is_extend_in_batch,
            is_prefill_only=self.is_prefill_only,
            seq_lens_cpu=self.seq_lens_cpu,
            enable_overlap=self.enable_overlap,
            mamba_track_indices=self.mamba_track_indices,
            mamba_track_mask=self.mamba_track_mask,
            mamba_track_seqlens=self.mamba_track_seqlens,
            dp_cooperation_info=self.dp_cooperation_info,
            prefill_stats=self.prefill_stats,
        )

    def maybe_evict_swa(self):
        # 如果 tree_cache 支持 SWA（Sliding Window Attention），按需驱逐超出滑动窗口的 KV
        if self.tree_cache.supports_swa():
            sliding_window_size = self.tree_cache.sliding_window_size
            server_args = get_global_server_args()

            # Eviction_interval: trade-off between SWA token waste and eviction overhead
            # 驱逐间隔：在 SWA token 浪费和驱逐开销之间取平衡
            page_size = self.tree_cache.page_size
            eviction_interval = max(
                page_size,
                int(
                    sliding_window_size
                    * envs.SGLANG_SWA_EVICTION_INTERVAL_MULTIPLIER.get()
                ),
            )
            # 对齐到 page_size
            eviction_interval = (eviction_interval // page_size) * page_size
            for idx, req in enumerate(self.reqs):
                if self.forward_mode.is_decode():
                    # We set evict_swa condition here with two reasons:
                    # 1. In overlap scheduler, we cannot evict swa when req.decode_batch_idx == 0 since the prev extend batch is still running.
                    # 2. Evict swa every eviction_interval tokens to reduce the overhead.
                    # decode 阶段：每 eviction_interval 步驱逐一次，避免驱逐太频繁
                    if req.decode_batch_idx % eviction_interval == 1:
                        self._evict_swa(req, req.seqlen - 1)
                elif self.forward_mode.is_extend() and self.tree_cache.is_chunk_cache():
                    # extend（chunked prefill）阶段
                    pre_len = self.prefix_lens[idx]
                    if self.enable_overlap:
                        # In chunked prefill case, when the second extend batch is scheduling, the first extend batch is still running, so we cannot evict swa tokens
                        # overlap 模式下，第一个 extend 批次还在运行时不能驱逐
                        if req.extend_batch_idx < 2:
                            continue
                        else:
                            # 减去 chunked_prefill_size 以对应上一个 extend 批次的前缀
                            pre_len = (
                                pre_len - server_args.chunked_prefill_size
                                if server_args.chunked_prefill_size > 0
                                else pre_len
                            )
                            self._evict_swa(req, pre_len)
                    else:
                        # 非 overlap 模式，直接驱逐
                        self._evict_swa(req, pre_len)

    def _evict_swa(self, req: Req, pre_len: int):
        # 实际执行 SWA KV cache 驱逐：释放超出滑动窗口且不在 radix tree 中的 token slot
        assert self.tree_cache.supports_swa(), "prefix cache must support swa"
        sliding_window_size = self.tree_cache.sliding_window_size

        # For swa radix cache, we need to evict the tokens that are not in the tree cache and also not in the sliding window
        # 确保 cache_protected_len 按 page_size 对齐
        assert (
            req.cache_protected_len % self.tree_cache.page_size == 0
        ), "cache_protected_len must be page aligned"
        # 驱逐起始点不得低于 cache_protected_len（受保护区域不可驱逐）
        req.swa_evicted_seqlen = max(req.swa_evicted_seqlen, req.cache_protected_len)

        # Subtract an extra page_size so the eviction frontier never reaches the
        # radix tree insert boundary (page_floor(seq_len)). This keeps at least one
        # page of non-evicted SWA KV for the tree to store as a non-tombstone node,
        # preserving cache reuse in multi-turn scenarios.
        # See also: _insert_helper case 3 in swa_radix_cache.py (defensive counterpart).
        # 计算新的驱逐边界：前缀长度 - 滑动窗口大小 - 额外一页（保留 radix tree 至少一页）
        new_swa_evicted_seqlen = max(
            req.swa_evicted_seqlen,
            pre_len - sliding_window_size - self.tree_cache.page_size,
        )

        if self.tree_cache.page_size > 1:
            # 向下对齐到 page_size
            new_swa_evicted_seqlen = (
                new_swa_evicted_seqlen // self.tree_cache.page_size
            ) * self.tree_cache.page_size

        if new_swa_evicted_seqlen > req.swa_evicted_seqlen:
            # 释放 [swa_evicted_seqlen, new_swa_evicted_seqlen) 范围内的 token slot
            free_slots = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, req.swa_evicted_seqlen : new_swa_evicted_seqlen
            ]
            self.token_to_kv_pool_allocator.free_swa(free_slots)
            req.swa_evicted_seqlen = new_swa_evicted_seqlen  # 更新已驱逐边界

    def __str__(self):
        # 返回批次的简短描述，包含 forward mode 和请求数
        return (
            f"ScheduleBatch(forward_mode={self.forward_mode.name if self.forward_mode else 'None'}, "
            f"#req={(len(self.reqs))})"
        )


@dataclasses.dataclass
class ModelWorkerBatch:
    # 传递给模型执行层的批次数据，由 ScheduleBatch.get_model_worker_batch() 构建
    # The forward mode
    # 本批次的前向模式（EXTEND/DECODE/MIXED/IDLE 等）
    forward_mode: ForwardMode
    # The input ids
    # 输入 token id 张量（GPU）
    input_ids: torch.Tensor
    # The indices of requests in the req_to_token_pool
    # 各请求在 req_to_token_pool 中的槽位索引
    req_pool_indices: torch.Tensor
    # The sequence length
    # 各请求当前序列长度（GPU）
    seq_lens: torch.Tensor
    # The indices of output tokens in the token_to_kv_pool_allocator
    # 本步输出 token 在 KV pool 中的分配位置
    out_cache_loc: torch.Tensor
    # The sequence length tensor on CPU
    # 序列长度的 CPU 副本（用于 Mamba/统计等 CPU 侧计算）
    seq_lens_cpu: Optional[torch.Tensor]
    seq_lens_sum: int  # 批次中所有序列长度之和

    # For logprob
    # 是否需要计算并返回 logprob
    return_logprob: bool
    # 各请求需要返回的 top-k logprob 数量
    top_logprobs_nums: Optional[List[int]]
    token_ids_logprobs: Optional[List[List[int]]]  # 各请求需要计算 logprob 的特定 token id

    # For DP attention
    # 数据并行（DP）注意力所需的全局 token 数
    global_num_tokens: Optional[List[int]]
    global_num_tokens_for_logprob: Optional[List[int]]
    is_extend_in_batch: bool     # 本批次中是否含有 extend 请求
    all_extend_in_batch: bool    # 本批次是否全为 extend 请求
    can_run_dp_cuda_graph: bool  # 是否可以使用 CUDA graph 加速 DP 推理
    tbo_split_seq_index: Optional[int]  # TBO（Two-Buffer Overlap）分割点
    global_forward_mode: Optional[ForwardMode]  # 全局 forward mode（DP 场景）

    # For extend
    # prefill/extend 阶段各请求的 extend token 数
    extend_num_tokens: Optional[int]
    extend_seq_lens: Optional[List[int]]         # 各请求的 extend 长度列表
    extend_prefix_lens: Optional[List[int]]      # 各请求的已缓存前缀长度列表
    extend_logprob_start_lens: Optional[List[int]]  # logprob 起始位置列表
    extend_input_logprob_token_ids: Optional[torch.Tensor]  # extend logprob 目标 token id

    # For multimodal
    # 多模态输入（图像/视频/音频）列表，每个请求对应一个 MultimodalInputs 或 None
    multimodal_inputs: Optional[List[MultimodalInputs]]

    # For encoder-decoder
    # encoder 部分是否已缓存的标志列表
    encoder_cached: Optional[List[bool]]
    encoder_lens: Optional[torch.Tensor]       # 编码器长度张量（GPU）
    encoder_lens_cpu: Optional[List[int]]      # 编码器长度列表（CPU）
    encoder_out_cache_loc: Optional[torch.Tensor]  # 编码器输出的 KV cache 位置

    # For LoRA
    lora_ids: Optional[List[str]]  # 各请求的 LoRA 适配器 ID

    # Sampling info
    # 采样参数信息（温度、top-p、top-k、惩罚等）
    sampling_info: SamplingBatchInfo

    # The original sequence lengths, Qwen-1M related
    # 原始序列长度（用于 Qwen-1M 等超长上下文模型）
    orig_seq_lens: Optional[torch.Tensor] = None

    # The input Embeds
    # 自定义输入嵌入（覆盖 embed_tokens 的输出）
    input_embeds: Optional[torch.Tensor] = None
    replace_embeds: Optional[torch.Tensor] = None    # 用于替换特定位置嵌入的嵌入张量
    replace_positions: Optional[torch.Tensor] = None  # 需要替换嵌入的位置索引

    # token table for ngram embedding
    # ngram embedding 的 token 查找表
    ne_token_table: Optional[torch.Tensor] = None

    # For corss-encoder model
    # cross-encoder 模型的 token 类型 ID（区分 query 和 passage）
    token_type_ids: Optional[torch.Tensor] = None

    # Speculative decoding
    # 投机解码算法类型
    spec_algorithm: SpeculativeAlgorithm = None

    spec_info: Optional[SpecInput] = None  # 投机解码的草稿/验证信息

    # If set, the output of the batch contains the hidden states of the run.
    # 是否捕获隐藏状态（FULL/NULL 等模式）
    capture_hidden_mode: CaptureHiddenMode = None
    hicache_consumer_index: int = -1  # HiCache 消费者索引（用于 L2/L3 缓存层）

    # For matryoshka embeddings
    # matryoshka 嵌入各请求的目标维度
    dimensions: Optional[list[int]] = None

    # Whether to return pooled hidden states (pre-head transformer output)
    # 是否返回池化后的隐藏状态（用于 embedding 模型）
    return_pooled_hidden_states: bool = False

    # Whether this batch is prefill-only (no token generation needed)
    # 是否为纯 prefill 批次（不需要生成 token）
    is_prefill_only: bool = False

    # Pre-computed delimiter indices for multi-item scoring (CPU tensors, one per request)
    # Multi-Item Scoring 的预计算分隔符索引列表（CPU 张量）
    multi_item_delimiter_indices: Optional[List[torch.Tensor]] = None

    # Diffusion LLM
    dllm_block_offsets: Optional[List[int]] = None  # 扩散语言模型的块偏移列表
    dllm_config: Optional[DllmConfig] = None         # 扩散语言模型配置

    # For constrained decoding
    # FIXME(lsyin): remove this after fully overlap grammar
    reqs: Optional[List[Req]] = None  # 请求列表（用于语法约束解码）
    has_grammar: bool = False          # 本批次是否有语法约束请求

    # For hidden states before normal
    # 是否返回 norm 前的隐藏状态（用于某些特殊分析场景）
    return_hidden_states_before_norm: bool = False

    # For mamba state tracking
    mamba_track_indices: Optional[torch.Tensor] = None  # shape: [b], int64
    mamba_track_mask: Optional[torch.Tensor] = None  # shape: [b], bool
    mamba_track_seqlens: Optional[torch.Tensor] = None  # shape: [b], int64
