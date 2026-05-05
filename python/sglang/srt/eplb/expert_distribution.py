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
# 专家分布统计模块：在每次前向传播中收集各专家的 token 处理量统计，
# 为 EPLB 再平衡提供负载数据，同时输出均衡度指标用于监控

from __future__ import annotations

import logging
import math
import time
from abc import ABC
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Type

import einops
import torch
import torch.distributed

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.observability.metrics_collector import ExpertDispatchCollector
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import Withable, get_device, get_int_env_var

if TYPE_CHECKING:
    from sglang.srt.eplb.expert_location import ExpertLocationMetadata

logger = logging.getLogger(__name__)

# --------------------------------------- Entrypoint -----------------------------------------

# 输出模式：file（保存到文件）或 object（直接返回 Python 对象）
_OutputMode = Literal["file", "object"]


@dataclass
class ExpertDistributionMetrics:
    # 专家分布均衡度指标（值越接近 1.0 表示负载越均衡）
    eplb_balancedness: torch.Tensor

    def copy_to_cpu(self):
        # 将均衡度张量移到 CPU，避免持有 GPU 内存
        self.eplb_balancedness = self.eplb_balancedness.to("cpu", non_blocking=True)


class ExpertDistributionRecorder(ABC):
    """Global expert distribution recording"""

    @staticmethod
    def init_new(
        server_args: ServerArgs,
        expert_location_metadata: ExpertLocationMetadata,
        rank: int,
    ):
        # 根据配置决定使用真实记录器还是空操作记录器
        if server_args.expert_distribution_recorder_mode is not None:
            assert (
                expert_location_metadata is not None
            ), "ExpertLocationMetadata is required for expert distribution recording. One possible"
            "reason is that you are using a model that does not support expert distribution"
            "recording. Try setting `get_model_config_for_expert_location` in your model."
            return _ExpertDistributionRecorderReal(
                server_args, expert_location_metadata, rank
            )
        else:
            # 未启用记录模式：返回 Noop 实现，所有调用均为空操作
            return _ExpertDistributionRecorderNoop()

    @contextmanager
    def with_current_layer(self, layer_idx):
        yield

    @contextmanager
    def with_debug_name(self, debug_name):
        yield

    @contextmanager
    def disable_this_region(self):
        yield

    @contextmanager
    def with_forward_pass(self, forward_pass_id: int, forward_batch: ForwardBatch):
        yield {}

    def on_select_experts(self, topk_ids: torch.Tensor):
        pass

    def on_deepep_dispatch_normal(
        self,
        local_physical_count_of_layer: List[int],
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
    ):
        pass

    def on_deepep_dispatch_low_latency(
        self, local_physical_count_of_layer: torch.Tensor
    ):
        pass

    def start_record(self):
        self._on_not_implemented()

    def stop_record(self):
        self._on_not_implemented()

    def dump_record(self, output_mode: _OutputMode = "file"):
        self._on_not_implemented()

    @property
    def recording(self):
        return False

    def _on_not_implemented(self):
        raise Exception(
            "Please set ServerArgs.expert_distribution_recorder_mode to use ExpertDistributionRecorder."
        )


class _ExpertDistributionRecorderNoop(ExpertDistributionRecorder):
    # 空操作实现：未启用记录时使用，所有接口均继承基类的空实现
    pass


class _ExpertDistributionRecorderReal(ExpertDistributionRecorder):
    # 真实记录器：内部维护 Gatherer（单次前向数据收集器）和 Accumulator（多次前向数据累积器）
    def __init__(
        self,
        server_args: ServerArgs,
        expert_location_metadata: ExpertLocationMetadata,
        rank: int,
    ):
        self._server_args = server_args
        self._expert_location_metadata = expert_location_metadata

        # 是否正在记录（由 start_record/stop_record 控制）
        self._recording = False
        # 全局禁用开关（用于在某些区域临时关闭记录）
        self._disable_all = False
        # 上下文变量：当前前向传播 ID、当前层 ID、当前调试名称
        self._current_forward_pass_id = Withable()
        self._current_layer_idx = Withable()
        self._current_debug_name = Withable()
        # 累积器：跨多次前向传播汇总统计数据
        self._accumulator = _Accumulator.init_new(
            server_args, expert_location_metadata, rank
        )
        # 单次前向传播数据收集器（可能有多个 key，用于 TBO 等场景）
        self._single_pass_gatherers = {
            k: _SinglePassGatherer.init_new(server_args, expert_location_metadata, rank)
            for k in self._accumulator.get_single_pass_gatherer_keys()
        }

        # 若启用了实时专家分布指标，自动启动记录
        if server_args.enable_expert_distribution_metrics:
            logger.info(
                "ExpertDistributionRecorder auto start record since enable_expert_distribution_metrics"
            )
            self.start_record()

    def with_current_layer(self, layer_idx):
        # 返回一个上下文管理器，在其内部 self._current_layer_idx.value == layer_idx
        return self._current_layer_idx.with_value(layer_idx)

    def with_debug_name(self, debug_name):
        # 返回一个上下文管理器，在其内部 self._current_debug_name.value == debug_name
        return self._current_debug_name.with_value(debug_name)

    @contextmanager
    def with_forward_pass(self, forward_pass_id: int, forward_batch: ForwardBatch):
        # 包裹整个前向传播：开始时初始化各 gatherer，结束时收集并累积统计数据
        outputs = {}
        with self._current_forward_pass_id.with_value(forward_pass_id):
            self._on_forward_pass_start(forward_batch)
            try:
                yield outputs
            finally:
                self._on_forward_pass_end(forward_pass_id, outputs)

    @contextmanager
    def disable_this_region(self):
        """Context manager to temporarily disable recording."""
        # 临时禁用记录（如在 warmup 或不需要统计的区域使用）
        previous_disable_all = self._disable_all
        self._disable_all = True
        try:
            yield
        finally:
            self._disable_all = previous_disable_all

    def _on_forward_pass_start(self, forward_batch: ForwardBatch):
        # 前向传播开始时：重置各 gatherer 并传入 batch 信息
        if not self._recording:
            return
        for gatherer_key, gatherer in self._single_pass_gatherers.items():
            gatherer.reset()
            gatherer.on_forward_pass_start(forward_batch)

    def _on_forward_pass_end(self, forward_pass_id: int, outputs: Dict[str, Any]):
        # 前向传播结束时：从各 gatherer 收集本次数据并追加到累积器
        if not self._recording:
            return
        for gatherer_key, gatherer in self._single_pass_gatherers.items():
            single_pass_data = gatherer.collect()
            self._accumulator.append(
                forward_pass_id, gatherer_key, single_pass_data, outputs
            )

    def on_select_experts(self, topk_ids: torch.Tensor):
        # 钩子：路由器选择专家后触发，将物理专家 ID 传递给当前层的 gatherer
        self._on_hook("on_select_experts", topk_ids=topk_ids)

    def on_deepep_dispatch_normal(
        self,
        local_physical_count_of_layer: List[int],
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
    ):
        # 钩子：DeepEP 普通模式 dispatch 完成后触发，收集本地物理专家计数
        self._on_hook(
            "on_deepep_dispatch_normal",
            local_physical_count_of_layer=local_physical_count_of_layer,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            num_tokens_per_expert=num_tokens_per_expert,
        )

    def on_deepep_dispatch_low_latency(
        self, local_physical_count_of_layer: torch.Tensor
    ):
        # 钩子：DeepEP 低延迟模式 dispatch 完成后触发
        self._on_hook(
            "on_deepep_dispatch_low_latency",
            local_physical_count_of_layer=local_physical_count_of_layer,
        )

    def _on_hook(self, hook_name: str, **kwargs):
        # 通用钩子分发：若全局禁用或非记录状态则跳过
        if self._disable_all:
            return
        if not (
            self._recording or torch.get_device_module().is_current_stream_capturing()
        ):
            return
        # 根据当前 debug_name 选择对应的 gatherer，并调用对应的钩子方法
        gatherer = self._single_pass_gatherers[
            self._accumulator.get_single_pass_gatherer_key(
                self._current_debug_name.value
            )
        ]
        getattr(gatherer, hook_name)(layer_idx=self._current_layer_idx.value, **kwargs)

    def _reset(self):
        """Reset the expert distribution recorder."""
        logger.info("Resetting ExpertDistributionRecorder...")
        assert (
            self._current_layer_idx.value is None
        ), f"{self._current_layer_idx.value=}"
        for gatherer in self._single_pass_gatherers.values():
            gatherer.reset()
        self._accumulator.reset()

    def start_record(self):
        """Start recording the expert distribution."""
        if self._recording:
            logger.warning(
                "SGLang server is already recording expert ids. Did you forget to dump the expert ids recorded so far by sending requests to the `/stop_expert_distribution_record` and `/dump_expert_distribution_record` endpoints?"
            )
        # 重置并开始记录
        self._reset()
        self._recording = True

    def stop_record(self):
        """Stop recording the expert distribution."""
        if not self._recording:
            logger.warning(
                "SGLang server has not been recording expert ids. Did you forget to start recording by sending request to the `/start_expert_distribution_record` endpoint?"
            )
        self._recording = False

    def dump_record(self, output_mode: _OutputMode = "file"):
        """Dump the expert distribution record and reset the recorder after dumping."""
        # 导出累积的统计数据，然后重置记录器（不停止记录）
        output = self._accumulator.dump(output_mode=output_mode)
        self._reset()
        return output

    @property
    def recording(self):
        return self._recording


# 全局记录器单例，默认为 Noop（空操作），由系统初始化时替换为真实记录器
_global_expert_distribution_recorder: Optional[ExpertDistributionRecorder] = (
    _ExpertDistributionRecorderNoop()
)


def get_global_expert_distribution_recorder():
    # 获取全局专家分布记录器实例
    return _global_expert_distribution_recorder


def set_global_expert_distribution_recorder(value):
    # 设置全局专家分布记录器（替换默认的 Noop）
    global _global_expert_distribution_recorder
    _global_expert_distribution_recorder = value


# --------------------------------------- SinglePassGatherer -----------------------------------------
# 单次前向传播数据收集器：负责在每层执行时记录专家的 token 处理量


class _SinglePassGatherer(ABC):
    @staticmethod
    def init_new(
        server_args: ServerArgs,
        expert_location_metadata: ExpertLocationMetadata,
        rank: int,
    ) -> "_SinglePassGatherer":
        # 根据记录模式和后端类型选择合适的 gatherer 实现
        if server_args.expert_distribution_recorder_mode == "per_token":
            # per_token 模式：记录每个 token 被路由到哪个专家的详细信息
            return _DetailSinglePassGatherer(
                server_args, expert_location_metadata, rank
            )

        if server_args.expert_distribution_recorder_mode == "stat_approx":
            # stat_approx 模式：使用 DeepEP 的近似统计（更低开销）
            if server_args.moe_a2a_backend != "none" and (
                server_args.deepep_mode == "normal"
            ):
                return _DeepepNormalSinglePassGatherer(expert_location_metadata, rank)
            else:
                raise NotImplementedError

        if server_args.moe_a2a_backend != "none":
            if server_args.deepep_mode == "normal":
                # DeepEP 普通模式：通过 select_experts 钩子收集物理专家计数
                return _SelectExpertsSinglePassGatherer(expert_location_metadata, rank)
            elif server_args.deepep_mode == "low_latency":
                # DeepEP 低延迟模式：通过 dispatch 钩子收集本地物理专家计数
                return _DeepepLowLatencySinglePassGatherer(
                    expert_location_metadata, rank
                )
            else:
                raise NotImplementedError

        # 默认：通过 select_experts 钩子收集全局物理专家计数
        return _SelectExpertsSinglePassGatherer(expert_location_metadata, rank)

    def __init__(self, expert_location_metadata: ExpertLocationMetadata, rank: int):
        self._expert_location_metadata = expert_location_metadata
        self._rank = rank

    def on_forward_pass_start(self, forward_batch: ForwardBatch):
        pass

    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor):
        pass

    def on_deepep_dispatch_normal(
        self,
        layer_idx: int,
        local_physical_count_of_layer: List[int],
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
    ):
        pass

    def on_deepep_dispatch_low_latency(
        self, layer_idx: int, local_physical_count_of_layer: torch.Tensor
    ):
        pass

    def reset(self):
        raise NotImplementedError

    def collect(self) -> Dict:
        raise NotImplementedError


class _DetailSinglePassGatherer(_SinglePassGatherer):
    # DeepSeek V3 has this value; should generalize later
    # per_token 模式的详细记录器：记录每个 token 被路由到哪个专家（精确但内存占用高）
    _TOP_K_NUM = 8

    def __init__(
        self,
        server_args: ServerArgs,
        expert_location_metadata: ExpertLocationMetadata,
        rank: int,
    ):
        super().__init__(expert_location_metadata, rank)
        # 当前前向传播的元数据（input_ids、positions 等）
        self._metadata: Optional[Dict[str, Any]] = None
        # 预分配张量：[num_layers, max_tokens, top_k]，存储每层每 token 的 top-k 专家 ID
        self._topk_ids_of_layer = torch.zeros(
            (
                expert_location_metadata.num_layers,
                # TODO determine the max number
                server_args.chunked_prefill_size * 8,
                self._TOP_K_NUM,
            ),
            dtype=torch.int32,
            device=server_args.device,
        )
        self._misc_objects: List[Dict[str, Any]] = []
        assert (
            not server_args.enable_two_batch_overlap
        ), "DetailSinglePassGatherer does not support TBO yet"
        # TODO assert shared experts fusion is disabled, o/w data is wrong

    def on_forward_pass_start(self, forward_batch: ForwardBatch):
        # 记录本次前向传播的基础信息（token IDs、位置、序列长度等）
        assert self._metadata is None
        self._metadata = dict(
            # TODO pr-chain
            # rids=forward_batch.rids,
            input_ids=forward_batch.input_ids.cpu().tolist(),
            positions=forward_batch.positions.cpu().tolist(),
            extend_seq_lens=forward_batch.extend_seq_lens_cpu,
            forward_mode=forward_batch.forward_mode.value,
        )

    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor):
        # 记录当前层每个 token 被路由到的物理专家 ID（写入预分配张量）
        self._topk_ids_of_layer[layer_idx, : topk_ids.shape[0], : topk_ids.shape[1]] = (
            topk_ids
        )

    def on_deepep_dispatch_normal(
        self,
        layer_idx: int,
        local_physical_count_of_layer: List[int],
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
    ):
        # 记录 DeepEP dispatch 的额外统计信息（如各 rank/RDMA rank 的 token 数分布）
        self._misc_objects.append(
            dict(
                layer_id=layer_idx,
                num_tokens_per_rank=num_tokens_per_rank.cpu().tolist(),
                num_tokens_per_rdma_rank=num_tokens_per_rdma_rank.cpu().tolist(),
                num_tokens_per_expert=num_tokens_per_expert.cpu().tolist(),
            )
        )

    def reset(self):
        # 重置所有状态（开始新一轮前向传播前调用）
        self._topk_ids_of_layer[...] = -1
        self._misc_objects.clear()
        self._metadata = None

    def collect(self) -> Dict:
        # 将本次前向传播的详细 topk_ids 转换为全局物理专家计数，并一起返回
        num_tokens = len(self._metadata["input_ids"])

        global_physical_count = _convert_per_token_to_global_physical_count(
            num_tokens,
            num_layers=self._expert_location_metadata.num_layers,
            num_physical_experts=self._expert_location_metadata.num_physical_experts,
            _topk_ids_of_layer=self._topk_ids_of_layer,
        )

        return dict(
            **self._metadata,
            topk_ids_of_layer=self._topk_ids_of_layer[:, :num_tokens, :].clone().cpu(),
            misc_objects=self._misc_objects,
            global_physical_count=global_physical_count,
        )


class _LayerBasedCpuSinglePassGatherer(_SinglePassGatherer):
    # CPU 端按层累积 gatherer 基类：每层数据以 Python list 形式存储，避免 GPU 内存占用
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 按层存储的对象字典：{layer_idx: [count_per_expert, ...]}
        self._objects_of_layer = {}

    def _on_layer_data(self, layer_idx: int, objects: List[int]):
        # 接收某层的数据，若已有则与原有数据逐元素相加（累积多次 dispatch）
        assert 0 <= layer_idx < self._expert_location_metadata.num_layers
        if layer_idx in self._objects_of_layer:
            self._objects_of_layer[layer_idx] = _list_sum(
                self._objects_of_layer[layer_idx], objects
            )
        else:
            self._objects_of_layer[layer_idx] = objects

    def reset(self):
        self._objects_of_layer.clear()

    def _collect_objects(self, pad_len: int) -> torch.Tensor:
        # 将各层数据整理为 [num_layers, pad_len] 的张量，缺失层用全零补充
        data = [
            self._objects_of_layer.get(layer_index) or ([0] * pad_len)
            for layer_index in range(self._expert_location_metadata.num_layers)
        ]
        return torch.tensor(data)


def _list_sum(a: List, b: List) -> List:
    # 两个等长列表逐元素相加
    return [x + y for x, y in zip(a, b, strict=True)]


class _LayerBasedGpuSinglePassGatherer(_SinglePassGatherer):
    # GPU 端按层累积 gatherer 基类：使用 GPU 张量存储，支持全局或本地物理专家计数
    def __init__(self, *args, enable_global_physical_experts: bool, **kwargs):
        super().__init__(*args, **kwargs)

        device = get_device()

        # 是否统计全局物理专家（True）或仅本地物理专家（False）
        self._enable_global_physical_experts = enable_global_physical_experts
        # 预分配数据张量：[num_layers, num_physical_experts] 或 [num_layers, num_local_experts]
        self._data = torch.zeros(
            (
                self._expert_location_metadata.num_layers,
                (
                    self._expert_location_metadata.num_physical_experts
                    if enable_global_physical_experts
                    else self._expert_location_metadata.num_local_physical_experts
                ),
            ),
            dtype=torch.int,
            device=device,
        )

    def reset(self):
        # 清零数据张量（每次前向传播开始时调用）
        self._data[...] = 0

    def collect(self) -> Dict:
        if self._enable_global_physical_experts:
            # 已是全局物理计数，直接返回
            global_physical_count = self._data
        else:
            # Can optimize if bottleneck
            # 本地计数转换为全局计数（在当前 rank 对应的偏移位置填入本地数据）
            global_physical_count = _convert_local_to_global_physical_count(
                self._data,
                rank=self._rank,
                num_local_physical_experts=self._expert_location_metadata.num_local_physical_experts,
                num_physical_experts=self._expert_location_metadata.num_physical_experts,
            )

        return dict(global_physical_count=global_physical_count)


class _SelectExpertsSinglePassGatherer(_LayerBasedGpuSinglePassGatherer):
    # 通过 select_experts 钩子统计全局物理专家计数（GPU scatter_add 实现）
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, enable_global_physical_experts=True)

    # can optimize (e.g. fuse / compile)
    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor):
        # 将 topk_ids 展平后，用 scatter_add 累加每个物理专家被选中的次数
        topk_ids = topk_ids.flatten()
        mask = topk_ids != -1
        self._data[layer_idx, :].scatter_add_(
            dim=0, index=topk_ids.masked_fill(~mask, 0).long(), src=mask.int()
        )


class _DeepepNormalSinglePassGatherer(_LayerBasedCpuSinglePassGatherer):
    # DeepEP 普通模式：通过 dispatch 钩子接收本地物理专家计数（CPU 侧近似统计）
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if torch.distributed.get_rank() == 0:
            logger.info(
                "DeepepNormalSinglePassGatherer gathers approximate statistics. "
                "If used with small batch size, consider using expert_distribution_recorder_mode=stat."
            )

    def on_deepep_dispatch_normal(
        self,
        layer_idx: int,
        local_physical_count_of_layer: List[int],
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
    ):
        # 接收 dispatch 后本地物理专家的 token 计数，按层累积
        assert isinstance(local_physical_count_of_layer, list)
        self._on_layer_data(layer_idx, local_physical_count_of_layer)

    def collect(self) -> Dict:
        # 将 CPU 端本地计数整理为张量，然后转换为全局物理计数
        local_physical_count = super()._collect_objects(
            pad_len=self._expert_location_metadata.num_local_physical_experts
        )
        global_physical_count = _convert_local_to_global_physical_count(
            local_physical_count,
            rank=self._rank,
            num_local_physical_experts=self._expert_location_metadata.num_local_physical_experts,
            num_physical_experts=self._expert_location_metadata.num_physical_experts,
        )
        return dict(global_physical_count=global_physical_count)


class _DeepepLowLatencySinglePassGatherer(_LayerBasedGpuSinglePassGatherer):
    # DeepEP 低延迟模式：通过 dispatch 钩子直接累积本地物理专家 GPU 张量计数
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, enable_global_physical_experts=False)

    def on_deepep_dispatch_low_latency(
        self, layer_idx: int, local_physical_count_of_layer: torch.Tensor
    ):
        # Most naive implementation, can optimize later
        # 直接累加本地物理专家计数（GPU 张量加法）
        self._data[layer_idx, :] += local_physical_count_of_layer


def _convert_per_token_to_global_physical_count(
    num_tokens: int,
    num_layers: int,
    num_physical_experts: int,
    _topk_ids_of_layer: torch.Tensor,  # [num_layers, max_tokens, top_k]
) -> torch.Tensor:
    # 将 per_token 的 topk_ids 统计转换为每层每个物理专家的 token 计数
    # 输出形状：[num_layers, num_physical_experts]
    topk_ids_layer_major = _topk_ids_of_layer[:, :num_tokens, :].reshape(num_layers, -1)
    mask = topk_ids_layer_major != -1  # 过滤 -1 填充项（无效路由）

    # 将 -1 替换为 0 以避免 index 越界，用 mask 控制 src（-1 处不计入计数）
    index = topk_ids_layer_major.masked_fill(~mask, 0).long()
    src = mask.int()

    ans = torch.zeros(
        (num_layers, num_physical_experts),
        dtype=_topk_ids_of_layer.dtype,
        device=_topk_ids_of_layer.device,
    )
    # scatter_add_：将每个物理专家被路由到的次数累加到对应位置
    ans.scatter_add_(dim=1, index=index, src=src)
    return ans


def _convert_local_to_global_physical_count(
    local_physical_count: torch.Tensor,  # [num_layers, num_local_physical_experts]
    rank: int,
    num_local_physical_experts: int,
    num_physical_experts: int,
) -> torch.Tensor:
    # 将本 rank 的本地专家计数嵌入到全局物理专家计数张量的对应位置
    dtype = local_physical_count.dtype
    device = local_physical_count.device
    num_layers, _ = local_physical_count.shape

    # 初始化全零张量，然后将本 rank 的计数填入对应的列范围
    ans = torch.zeros((num_layers, num_physical_experts), dtype=dtype, device=device)
    ans[
        :, num_local_physical_experts * rank : num_local_physical_experts * (rank + 1)
    ] = local_physical_count
    return ans


# --------------------------------------- Accumulator -----------------------------------------
# 累积器：跨多次前向传播聚合统计数据，为 EPLB 提供历史窗口内的负载统计

# 默认的 gatherer key（非 TBO 场景只有一个 gatherer）
_SINGLE_PASS_GATHERER_KEY_PRIMARY = "primary"


class _Accumulator(ABC):
    @staticmethod
    def init_new(
        server_args: ServerArgs,
        expert_location_metadata: ExpertLocationMetadata,
        rank: int,
    ) -> "_Accumulator":
        # 根据记录模式选择对应的 Accumulator 实现
        return _Accumulator.get_class(server_args)(
            server_args, expert_location_metadata, rank
        )

    @staticmethod
    def get_class(server_args: ServerArgs) -> Type["_Accumulator"]:
        # stat/stat_approx → 统计累积器；per_pass/per_token → 详细记录累积器
        return {
            "stat": _StatAccumulator,
            "stat_approx": _StatAccumulator,
            "per_pass": _DetailAccumulator,
            "per_token": _DetailAccumulator,
        }[server_args.expert_distribution_recorder_mode]

    def __init__(
        self,
        server_args: ServerArgs,
        expert_location_metadata: ExpertLocationMetadata,
        rank: int,
    ):
        self._server_args = server_args
        self._expert_location_metadata = expert_location_metadata
        self._rank = rank

    def get_single_pass_gatherer_keys(self):
        # 返回需要创建的 gatherer key 列表（默认只有 primary）
        return [_SINGLE_PASS_GATHERER_KEY_PRIMARY]

    def get_single_pass_gatherer_key(self, debug_name: Optional[str]):
        # 根据 debug_name 返回对应的 gatherer key（默认忽略 debug_name）
        return _SINGLE_PASS_GATHERER_KEY_PRIMARY

    def append(
        self,
        forward_pass_id: int,
        gatherer_key: str,
        single_pass_data: Dict,
        outputs: Dict[str, Any],
    ):
        pass

    def reset(self):
        pass

    def dump(self, output_mode: _OutputMode):
        pass


class _UtilizationRateAccumulatorMixin(_Accumulator):
    # Mixin：在 append 时同步计算并记录 GPU 利用率（均衡度）指标
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 是否启用实时指标（由 enable_expert_distribution_metrics 控制）
        self._enable = self._server_args.enable_expert_distribution_metrics

        if self._enable:
            # 多窗口大小历史记录（分别记录最近 10/100/1000 次前向传播的均衡度）
            self.window_sizes = [10, 100, 1000]
            self._history = _DequeCollection(maxlens=self.window_sizes)
            self._rank = torch.distributed.get_rank()
            self._expert_dispatch_collector = ExpertDispatchCollector(
                self._expert_location_metadata.ep_size
            )
            # 热力图采集计数器（按 interval 周期性采集）
            self._metric_heatmap_collection_counter = 0

    def append(
        self,
        forward_pass_id: int,
        gatherer_key: str,
        single_pass_data: Dict,
        outputs: Dict[str, Any],
    ):
        super().append(forward_pass_id, gatherer_key, single_pass_data, outputs)
        if self._enable:
            # 同步计算本次前向传播的 GPU 利用率并更新历史记录
            return self._append_utilization_rate(
                forward_pass_id, single_pass_data["global_physical_count"], outputs
            )

    def reset(self):
        super().reset()
        if self._enable:
            # 清空历史窗口（重新开始统计）
            self._history.clear()

    def _append_utilization_rate(
        self,
        forward_pass_id: int,
        single_pass_global_physical_count: torch.Tensor,  # [num_layers, num_physical_experts]
        outputs: Dict[str, Any],
    ):
        # 计算每个 GPU 的 token 总量（[num_layers, num_gpu]）
        gpu_physical_count = compute_gpu_physical_count(
            single_pass_global_physical_count,
            num_gpu=self._expert_location_metadata.ep_size,
        )
        gpu_physical_count = gpu_physical_count.to(self._server_args.device)
        # 跨所有 rank 汇总 GPU 物理计数（reduce 到 rank 0）
        torch.distributed.reduce(
            gpu_physical_count, dst=0, op=torch.distributed.ReduceOp.SUM
        )

        if self._rank == 0:
            # 采集热力图指标（Prometheus 直方图）
            self._handle_metric_eplb_heatmap(gpu_physical_count)

            # 计算均衡度 = 平均负载 / 最大负载（值越接近 1 越均衡）
            utilization_rate_gpu = torch.mean(
                compute_utilization_rate(gpu_physical_count)
            )
            if envs.SGLANG_ENABLE_EPLB_BALANCEDNESS_METRIC.get():
                print(f"hi {self._rank=} {utilization_rate_gpu=}")
                outputs["metrics"] = ExpertDistributionMetrics(
                    eplb_balancedness=utilization_rate_gpu,
                )
            else:
                # TODO maybe refactor this part to also avoid a `.item()` gpu->cpu sync
                utilization_rate_cpu = utilization_rate_gpu.item()
                self._history.append(utilization_rate_cpu)

                gpu_physical_count_sum = gpu_physical_count.sum().item()

                logger.info(
                    f"[Expert Balancedness] "
                    f"forward_pass_id={forward_pass_id} "
                    f"current_pass_balancedness={utilization_rate_cpu:.03f} "
                    f"{''.join(f'last_{size}_average_balancedness={value:.03f} ' for size, value in self._history.mean().items())} "
                    f"gpu_physical_count_sum={gpu_physical_count_sum}"
                    # f"current_pass_per_layer={[round(x, 2) for x in utilization_rate_tensor.cpu().tolist()]}"
                )

    # TODO refactor
    def _handle_metric_eplb_heatmap(self, gpu_physical_count: torch.Tensor):
        # sglang:eplb_gpu_physical_count metric is disabled if SGLANG_EPLB_HEATMAP_COLLECTION_INTERVAL <= 0
        # 按间隔周期性采集 GPU 物理专家分布热力图指标（写入 Prometheus 直方图）
        interval = get_int_env_var("SGLANG_EPLB_HEATMAP_COLLECTION_INTERVAL", 0)
        if interval > 0 and self._metric_heatmap_collection_counter % interval == 0:
            for layer_idx in range(self._expert_location_metadata.num_layers):
                count_of_layer = (
                    self._expert_dispatch_collector.eplb_gpu_physical_count.labels(
                        layer=str(layer_idx)
                    )
                )
                # Exclude the +Inf bucket.
                assert (
                    self._expert_location_metadata.ep_size
                    == len(count_of_layer._buckets) - 1
                ), f"{self._expert_location_metadata.ep_size=}, {len(count_of_layer._buckets)=}"
                for gpu_rank in range(self._expert_location_metadata.ep_size):
                    count = gpu_physical_count[layer_idx, gpu_rank]
                    if count > 0:
                        count_of_layer._sum.inc(count * gpu_rank)
                        count_of_layer._buckets[gpu_rank].inc(count)
        self._metric_heatmap_collection_counter += 1


class _DequeCollection:
    # 多窗口大小的 deque 集合：同时维护多个不同 maxlen 的滑动窗口
    def __init__(self, maxlens: List[int]):
        self._dequeues = [deque(maxlen=maxlen) for maxlen in maxlens]

    def append(self, value):
        # 向所有窗口追加相同的值
        for d in self._dequeues:
            d.append(value)

    def clear(self):
        # 清空所有窗口（reset 时调用）
        for d in self._dequeues:
            d.clear()

    def mean(self) -> Dict[int, float]:
        # 返回 {窗口大小: 窗口内均值} 的字典
        return {d.maxlen: sum(d) / len(d) for d in self._dequeues}


class _DetailAccumulator(_UtilizationRateAccumulatorMixin):
    # 详细累积器：保存每次前向传播的完整记录（per_pass/per_token 模式）
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 记录列表：每次前向传播的完整数据（包含 topk_ids、global_physical_count 等）
        self._records = []

    def get_single_pass_gatherer_keys(self):
        if False:  # TODO `server_args.enable_two_batch_overlap`
            return [_SINGLE_PASS_GATHERER_KEY_PRIMARY, "child_a", "child_b"]
        return super().get_single_pass_gatherer_keys()

    def get_single_pass_gatherer_key(self, debug_name: Optional[str]):
        if False:  # TODO `server_args.enable_two_batch_overlap`
            return debug_name or _SINGLE_PASS_GATHERER_KEY_PRIMARY
        return super().get_single_pass_gatherer_key(debug_name)

    def append(
        self,
        forward_pass_id: int,
        gatherer_key: str,
        single_pass_data: Dict,
        outputs: Dict[str, Any],
    ):
        super().append(forward_pass_id, gatherer_key, single_pass_data, outputs)

        # 将所有 GPU 张量转移到 CPU（避免长期占用 GPU 内存）
        def _process_object(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().clone()
            return obj

        single_pass_data_processed = {
            k: _process_object(v) for k, v in single_pass_data.items()
        }

        # 追加本次前向传播的完整记录
        self._records.append(
            dict(
                forward_pass_id=forward_pass_id,
                rank=self._rank,
                gatherer_key=gatherer_key,
                **single_pass_data_processed,
            )
        )

    def reset(self):
        super().reset()
        self._records.clear()

    def dump(self, output_mode: _OutputMode):
        # 只支持 file 模式：将所有记录保存到 .pt 文件
        assert output_mode == "file"
        output = dict(
            records=self._records,
            # NOTE: This may change during recording, so here we say it is the "last" one
            # 注意：录制过程中映射可能发生变化，此处保存的是最新的映射
            last_physical_to_logical_map=self._expert_location_metadata.physical_to_logical_map,
        )
        _dump_to_file(
            f"expert_distribution_recorder_{time.time()}_{self._rank}.pt", output
        )


class _StatAccumulator(_UtilizationRateAccumulatorMixin):
    # 统计累积器：使用循环缓冲区存储近期的物理专家计数，用于 EPLB 负载均衡决策
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 循环/无限缓冲区：存储最近 buffer_size 次前向传播的全局物理专家计数
        self._global_physical_count_of_buffered_step = _Buffer.init_new(
            item_shape=(
                self._expert_location_metadata.num_layers,
                # Cannot use local_physical_count to support select_experts
                # 使用全局物理计数（非本地）以兼容 select_experts 钩子场景
                self._expert_location_metadata.num_physical_experts,
            ),
            buffer_size=self._server_args.expert_distribution_recorder_buffer_size,
            dtype=torch.int32,
            device=self._server_args.device,
        )
        # 首次 dump 时清理 GPU 缓存
        self._first_dump = True

    def append(
        self,
        forward_pass_id: int,
        gatherer_key: str,
        single_pass_data: Dict,
        outputs: Dict[str, Any],
    ):
        super().append(forward_pass_id, gatherer_key, single_pass_data, outputs)
        # Can optimize if overhead here is large
        # 将本次全局物理计数追加到缓冲区（循环覆盖最旧的记录）
        self._global_physical_count_of_buffered_step.append(
            single_pass_data["global_physical_count"]
        )

    def reset(self):
        super().reset()
        self._global_physical_count_of_buffered_step.reset()

    def dump(self, output_mode: _OutputMode):
        # 将缓冲区内的全局物理计数转换为逻辑专家计数（用于 EPLB 算法输入）
        logical_count_of_buffered_step = _convert_global_physical_count_to_logical_count(
            self._global_physical_count_of_buffered_step.get_all(),
            num_layers=self._expert_location_metadata.num_layers,
            num_logical_experts=self._expert_location_metadata.num_logical_experts,
            physical_to_logical_map=self._expert_location_metadata.physical_to_logical_map,
        )

        if self._first_dump:
            self._first_dump = False
            # 首次 dump 前清理 GPU 缓存，释放转换前的临时内存
            torch.get_device_module().empty_cache()

        # 跨所有 rank 汇总逻辑专家计数（all_reduce SUM）
        torch.distributed.all_reduce(
            logical_count_of_buffered_step, op=torch.distributed.ReduceOp.SUM
        )

        output = dict(
            rank=self._rank,
            logical_count=logical_count_of_buffered_step,
            # 当前窗口内的平均 GPU 利用率（用于判断是否需要再平衡）
            average_utilization_rate_over_window=self._get_global_average_utilization_rate(),
        )

        if output_mode == "file":
            # 文件模式：只有 rank 0 写入文件
            if self._rank == 0:
                _dump_to_file(f"expert_distribution_recorder_{time.time()}.pt", output)
        elif output_mode == "object":
            # 对象模式：直接返回字典（供 EPLBManager 内存中使用）
            return output
        else:
            raise NotImplementedError

    def _get_global_average_utilization_rate(self):
        # 获取最大窗口内的全局平均 GPU 利用率（通过 broadcast 同步到所有 rank）
        if not self._enable or math.isclose(
            self._server_args.eplb_min_rebalancing_utilization_threshold, 1.0
        ):
            # 未启用指标或阈值为 1.0（始终触发再平衡）时，返回 None
            return None

        if self._rank == 0:
            utilization_mean_rates = self._history.mean()
            # 取最大窗口的均值（最平滑的估计）
            window_index = self.window_sizes[-1]
            average_utilization_rate_over_window = (
                utilization_mean_rates[window_index]
                if window_index in utilization_mean_rates
                else 0  # 历史数据不足时返回 0（触发再平衡）
            )

            avg_rate_tensor = torch.tensor(
                [average_utilization_rate_over_window],
                dtype=torch.float32,
                device="cuda",
            )
        else:
            # 非 rank 0 准备接收广播
            avg_rate_tensor = torch.empty(1, dtype=torch.float32, device="cuda")
        # 将 rank 0 的均值广播给所有 rank
        torch.distributed.broadcast(avg_rate_tensor, src=0)
        return avg_rate_tensor.item()


def _dump_to_file(name, data):
    # 将数据以 .pt 格式保存到指定目录（若目录不存在则自动创建）
    save_dir = Path(envs.SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR.get())
    path_output = save_dir / name
    logger.info(f"Write expert distribution to {path_output}")
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, str(path_output))


class _Buffer:
    # 缓冲区基类：提供 append/get_all/reset 接口，根据 buffer_size 选择循环或无限实现
    @staticmethod
    def init_new(item_shape: Tuple, buffer_size: int, dtype, device):
        # buffer_size < 0 表示无限缓冲区，否则使用固定大小的循环缓冲区
        if buffer_size < 0:
            return _InfiniteBuffer(item_shape, dtype=dtype, device=device)
        else:
            return _CircularBuffer(item_shape, buffer_size, dtype=dtype, device=device)

    def append(self, value: torch.Tensor):
        raise NotImplementedError

    def get_all(self) -> torch.Tensor:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class _CircularBuffer(_Buffer):
    # 循环缓冲区：固定大小，新数据覆盖最旧的数据（滑动窗口）
    def __init__(self, item_shape: Tuple, buffer_size: int, dtype, device):
        # 预分配 [buffer_size, *item_shape] 的张量
        self._buffer = torch.zeros(
            (buffer_size, *item_shape), dtype=dtype, device=device
        )
        # 当前写入位置（循环递增）
        self._curr_index = 0

    def append(self, value: torch.Tensor):
        # 写入当前位置，然后循环推进写指针
        self._buffer[self._curr_index] = value
        self._curr_index = (self._curr_index + 1) % len(self._buffer)

    def get_all(self) -> torch.Tensor:
        # 返回整个缓冲区（包含所有时间步的数据，含未使用的全零槽位）
        return self._buffer

    def reset(self):
        # 清零缓冲区并重置写指针
        self._buffer[...] = 0


class _InfiniteBuffer(_Buffer):
    # 无限缓冲区：动态扩容（每次满时容量翻倍），适用于 per_pass/per_token 详细记录模式
    def __init__(self, item_shape: Tuple, dtype, device):
        self._item_shape = item_shape
        # 初始容量 128，按需倍增
        self._buffer = torch.zeros((128, *item_shape), dtype=dtype, device=device)
        self._size = 0  # 当前有效数据数量

    def append(self, value: torch.Tensor):
        curr_buffer_size = len(self._buffer)
        dtype = self._buffer.dtype
        device = self._buffer.device

        # 缓冲区已满时，分配 2 倍大小的新缓冲区并拷贝旧数据
        if self._size == curr_buffer_size:
            new_buffer = torch.zeros(
                (2 * curr_buffer_size, *self._item_shape), dtype=dtype, device=device
            )
            new_buffer[:curr_buffer_size] = self._buffer
            self._buffer = new_buffer

        self._buffer[self._size] = value
        self._size += 1

    def get_all(self) -> torch.Tensor:
        # 只返回有效数据切片（跳过末尾的全零预留空间）
        return self._buffer[: self._size]

    def reset(self):
        # 清零缓冲区并重置有效数量
        self._buffer[...] = 0
        self._size = 0


def _convert_global_physical_count_to_logical_count(
    # (whatever, num_layers, num_physical_experts)
    global_physical_count: torch.Tensor,  # [dim_extra, num_layers, num_physical_experts]
    num_layers: int,
    num_logical_experts: int,
    physical_to_logical_map: torch.Tensor,  # [num_layers, num_physical_experts]，物理→逻辑映射
):
    # 将物理专家计数按 physical_to_logical_map 汇总为逻辑专家计数
    # 每个物理专家的计数被加到其对应的逻辑专家上（scatter_add_）
    dim_extra, _, _ = global_physical_count.shape
    dtype = global_physical_count.dtype
    device = global_physical_count.device
    logical_count = torch.zeros(
        (dim_extra, num_layers, num_logical_experts), dtype=dtype, device=device
    )
    logical_count.scatter_add_(
        dim=2,
        # 将 physical_to_logical_map 扩展到 [dim_extra, num_layers, num_physical_experts]
        index=physical_to_logical_map.unsqueeze(0)
        .expand(dim_extra, -1, -1)
        .to(torch.int64),
        src=global_physical_count,
    )
    return logical_count


def compute_gpu_physical_count(
    physical_count_of_whatever: torch.Tensor,  # (..., num_layer, num_physical_expert)
    num_gpu: int,
):
    """output: gpu_physical_count_of_batch (..., num_layer, num_gpu)"""
    # 将物理专家维度按 GPU 分块求和，得到每个 GPU 上的总 token 处理量
    return einops.reduce(
        physical_count_of_whatever,
        "... num_layer (num_gpu num_expert_per_gpu) -> ... num_layer num_gpu",
        "sum",
        num_gpu=num_gpu,
    )


def compute_utilization_rate(
    gpu_physical_count_of_batch: torch.Tensor,  # (..., num_layer, num_gpu)
):
    """output: utilization_rate (..., num_layer)"""
    # 均衡度 = 平均 GPU 负载 / 最大 GPU 负载（加小量 epsilon 防止除零）
    # 值越接近 1.0 表示各 GPU 负载越均衡
    gpu_physical_count_of_batch = gpu_physical_count_of_batch.float()
    max_gpu_physical_count = einops.reduce(
        gpu_physical_count_of_batch,
        "... num_layer num_gpu -> ... num_layer",
        "max",
    )
    avg_gpu_physical_count = einops.reduce(
        gpu_physical_count_of_batch,
        "... num_layer num_gpu -> ... num_layer",
        "mean",
    )
    return (avg_gpu_physical_count + 1e-5) / (max_gpu_physical_count + 1e-5)
