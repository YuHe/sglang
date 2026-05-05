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
# 专家位置元数据模块：管理物理专家与逻辑专家之间的映射关系
# 提供初始化、更新、查询接口，以及全局元数据单例的存取函数

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.distributed
import torch.nn.functional as F

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@dataclass
class ExpertLocationMetadata:
    """
    专家位置元数据：封装 EPLB 系统所需的全部专家映射张量。
    所有映射均有 GPU 版本和 CPU 版本，CPU 版本用于避免频繁的 GPU→CPU 同步。
    """
    physical_to_logical_map: torch.Tensor  # (layers, num_physical_experts) GPU 版，物理→逻辑映射
    physical_to_logical_map_cpu: torch.Tensor  # CPU 版，用于 P2P 通信时高效读取
    logical_to_all_physical_map: torch.Tensor  # (layers, num_logical_experts, X) GPU 版，逻辑→所有物理副本
    logical_to_all_physical_map_cpu: torch.Tensor  # CPU copy for performance
    logical_to_all_physical_map_num_valid: torch.Tensor  # (layers, num_logical_experts)，每个逻辑专家的有效副本数
    # (layers, num_logical_experts)：静态调度时，每个 rank 视角下每个逻辑专家对应的最优物理专家
    logical_to_rank_dispatch_physical_map: Optional[torch.Tensor]

    # -------------------------------- properties ------------------------------------

    @property
    def num_layers(self) -> int:
        # 模型中 MoE 层的总数
        return self.physical_to_logical_map.shape[0]

    @property
    def num_physical_experts(self) -> int:
        # 物理专家总数（含冗余副本）
        return self.physical_to_logical_map.shape[1]

    @property
    def num_local_physical_experts(self) -> int:
        # 每张 GPU 本地持有的物理专家数（= 物理专家总数 / EP 大小）
        ans, remainder = divmod(self.num_physical_experts, self.ep_size)
        assert remainder == 0
        return ans

    @property
    def num_logical_experts(self) -> int:
        # 逻辑专家总数（模型定义的专家数，不含冗余）
        return self.logical_to_all_physical_map.shape[1]

    @property
    def ep_size(self):
        # TODO change when EP size != world size
        # EP（专家并行）的进程数，当前与 world_size 相同
        return torch.distributed.get_world_size()

    def __post_init__(self):
        # 校验各映射张量的维度一致性
        num_layers_0, num_physical_experts_0 = self.physical_to_logical_map.shape
        num_layers_1, num_logical_experts_0, num_physical_experts_1 = (
            self.logical_to_all_physical_map.shape
        )
        num_layers_2, num_logical_experts_1 = (
            self.logical_to_all_physical_map_num_valid.shape
        )
        assert num_layers_0 == num_layers_1 == num_layers_2
        assert num_logical_experts_0 == num_logical_experts_1
        assert num_physical_experts_0 == num_physical_experts_1

    # -------------------------------- construction ------------------------------------

    @staticmethod
    def init_trivial(
        server_args: ServerArgs, model_config: ModelConfig, moe_ep_rank: int
    ):
        """Trivial location - logical expert i corresponds to physical expert i"""
        # 平凡初始化：物理专家 i 对应逻辑专家 i % num_logical_experts（循环分配）
        common = ExpertLocationMetadata._init_common(server_args, model_config)

        if common is None:
            return None

        num_physical_experts = common["num_physical_experts"]
        model_config_for_expert_location = common["model_config_for_expert_location"]
        num_layers = model_config_for_expert_location.num_layers
        num_logical_experts = model_config_for_expert_location.num_logical_experts

        # 构建平凡的物理→逻辑映射：物理索引循环取模逻辑专家数
        physical_to_logical_map = (
            torch.arange(0, num_physical_experts).repeat(num_layers, 1)
            % num_logical_experts
        )

        return ExpertLocationMetadata.init_by_mapping(
            server_args,
            model_config,
            physical_to_logical_map=physical_to_logical_map,
            moe_ep_rank=moe_ep_rank,
        )

    @staticmethod
    def init_by_mapping(
        server_args: ServerArgs,
        model_config: ModelConfig,
        physical_to_logical_map,     # 外部指定的物理→逻辑映射（可以是 list 或 Tensor）
        moe_ep_rank: int = None,
    ):
        # 将映射转为 GPU 张量
        if not isinstance(physical_to_logical_map, torch.Tensor):
            physical_to_logical_map = torch.tensor(physical_to_logical_map)
        physical_to_logical_map = physical_to_logical_map.to(server_args.device)

        common = ExpertLocationMetadata._init_common(server_args, model_config)

        if common is None:
            return None

        model_config_for_expert_location = common["model_config_for_expert_location"]
        # 根据物理→逻辑映射，计算逻辑→所有物理副本映射（考虑就近原则）
        logical_to_all_physical_map = _compute_logical_to_all_physical_map(
            server_args=server_args,
            physical_to_logical_map=physical_to_logical_map,
            num_logical_experts=model_config_for_expert_location.num_logical_experts,
            ep_size=common["ep_size"],
            moe_ep_rank=moe_ep_rank,
        )

        return ExpertLocationMetadata._init_raw(
            server_args=server_args,
            ep_size=common["ep_size"],
            physical_to_logical_map=physical_to_logical_map,
            logical_to_all_physical_map=logical_to_all_physical_map,
        )

    @staticmethod
    def init_by_eplb(
        server_args: ServerArgs, model_config: ModelConfig, logical_count: torch.Tensor
    ):
        # 基于专家负载统计（logical_count）运行 EPLB 算法，生成新的专家位置映射
        if not isinstance(logical_count, torch.Tensor):
            logical_count = torch.tensor(logical_count)
        # 若输入是 2D（单步），扩展为 3D（增加步维度）
        if len(logical_count.shape) == 2:
            logical_count = logical_count.unsqueeze(0)
        logical_count = logical_count.to(server_args.device)

        common = ExpertLocationMetadata._init_common(server_args, model_config)

        if common is None:
            return None

        model_config_for_expert_location = common["model_config_for_expert_location"]
        num_physical_experts = common["num_physical_experts"]
        num_groups = model_config_for_expert_location.num_groups
        num_nodes = server_args.nnodes

        from sglang.srt.eplb import eplb_algorithms

        # 调用 EPLB 算法计算新的物理→逻辑映射和逻辑→物理映射
        physical_to_logical_map, logical_to_all_physical_map, expert_count = (
            eplb_algorithms.rebalance_experts(
                tokens_per_expert=logical_count,
                num_physical_experts=num_physical_experts,
                num_local_physical_experts=num_physical_experts // common["ep_size"],
                num_groups=num_groups,
                num_nodes=num_nodes,
                algorithm=eplb_algorithms.compute_algorithm(
                    raw_algorithm=server_args.eplb_algorithm,
                    num_groups=num_groups,
                    num_nodes=num_nodes,
                ),
            )
        )

        return ExpertLocationMetadata._init_raw(
            server_args=server_args,
            ep_size=common["ep_size"],
            physical_to_logical_map=physical_to_logical_map.to(server_args.device),
            logical_to_all_physical_map=logical_to_all_physical_map.to(
                server_args.device
            ),
        )

    @staticmethod
    def _init_common(server_args: ServerArgs, model_config: ModelConfig):
        # 提取模型的专家相关配置（层数、逻辑专家数、分组数）
        model_config_for_expert_location = (
            ModelConfigForExpertLocation.from_model_config(model_config)
        )

        # 若模型不支持专家位置管理（非 MoE 模型），返回 None
        if model_config_for_expert_location is None:
            return None

        # 物理专家总数 = 逻辑专家数 + 冗余专家数（热专家副本）
        num_physical_experts = (
            model_config_for_expert_location.num_logical_experts
            + server_args.ep_num_redundant_experts
        )
        ep_size = server_args.ep_size
        assert num_physical_experts % ep_size == 0
        num_local_physical_experts = num_physical_experts // ep_size

        return dict(
            model_config_for_expert_location=model_config_for_expert_location,
            num_physical_experts=num_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
            ep_size=ep_size,
        )

    @staticmethod
    def _init_raw(
        server_args: ServerArgs,
        ep_size: int,
        physical_to_logical_map: torch.Tensor,
        logical_to_all_physical_map: torch.Tensor,
    ):
        _, num_physical_experts = physical_to_logical_map.shape

        # 将逻辑→物理映射右侧补零（-1 填充），使第三维度等于物理专家总数
        logical_to_all_physical_map_padded = F.pad(
            logical_to_all_physical_map,
            (0, num_physical_experts - logical_to_all_physical_map.shape[-1]),
            value=-1,
        )

        # 统计每个逻辑专家的有效物理副本数（非 -1 的数量）
        logical_to_all_physical_map_num_valid = torch.count_nonzero(
            logical_to_all_physical_map != -1, dim=-1
        )

        return ExpertLocationMetadata(
            physical_to_logical_map=physical_to_logical_map,
            physical_to_logical_map_cpu=physical_to_logical_map.cpu(),
            logical_to_all_physical_map=logical_to_all_physical_map_padded,
            logical_to_all_physical_map_cpu=logical_to_all_physical_map_padded.cpu(),
            logical_to_all_physical_map_num_valid=logical_to_all_physical_map_num_valid,
            # 静态调度时，预计算每个 rank 视角下每个逻辑专家的最优物理专家 ID
            logical_to_rank_dispatch_physical_map=(
                compute_logical_to_rank_dispatch_physical_map(
                    server_args=server_args,
                    logical_to_all_physical_map=logical_to_all_physical_map,
                    ep_size=ep_size,
                    num_physical_experts=num_physical_experts,
                    # TODO improve when we have real EP rank
                    ep_rank=torch.distributed.get_rank() % ep_size,
                )
                if server_args.ep_dispatch_algorithm == "static"
                else None
            ),
        )

    # -------------------------------- mutation ------------------------------------

    def update(
        self,
        other: "ExpertLocationMetadata",
        update_layer_ids: List[int],  # 只更新指定层的映射，其余层保持不变
    ):
        # 校验不可变属性（ep_size）必须一致
        for field in [
            "ep_size",
        ]:
            assert getattr(self, field) == getattr(other, field)

        # 对每个映射张量，用 update_layer_ids 对应的掩码做选择性更新
        for field in [
            "physical_to_logical_map",
            "physical_to_logical_map_cpu",
            "logical_to_all_physical_map",
            "logical_to_all_physical_map_cpu",
            "logical_to_all_physical_map_num_valid",
            "logical_to_rank_dispatch_physical_map",
        ]:
            other_field = getattr(other, field)
            self_field = getattr(self, field)
            assert (other_field is not None) == (self_field is not None)
            if self_field is not None:
                # 构建层维度的布尔掩码，True 表示该层需要更新
                mask_update = torch.tensor(
                    [i in update_layer_ids for i in range(self.num_layers)]
                )
                # 扩展掩码以匹配张量维度（在后续维度添加大小为 1 的维度）
                mask_update = mask_update.view(*([-1] + [1] * (self_field.dim() - 1)))
                mask_update = mask_update.to(self_field.device, non_blocking=True)
                # 就地更新：待更新层取 other 的值，其余层保持 self 的值
                self_field[...] = torch.where(mask_update, other_field, self_field)

    # -------------------------------- usage ------------------------------------

    def logical_to_all_physical(
        self,
        layer_id: int,
        logical_expert_id: int,
        require_global_experts: bool = False,
    ) -> List[int]:
        # Use CPU copy to avoid GPU→CPU sync on every call, which is expensive in update weights scenario
        # 优先使用 CPU 版本，避免 GPU→CPU 同步（在更新权重场景下代价较高）
        if require_global_experts:
            # require_global_experts=True：按全局顺序返回该逻辑专家的所有物理副本
            num_physical_experts = self.logical_to_all_physical_map_cpu[layer_id].shape[
                -1
            ]
            return list(
                range(logical_expert_id, num_physical_experts, self.num_logical_experts)
            )
        # 默认：返回映射表中该逻辑专家的所有有效物理副本（过滤掉 -1 占位符）
        return [
            physical_expert_id
            for physical_expert_id in self.logical_to_all_physical_map_cpu[
                layer_id, logical_expert_id
            ].tolist()
            if physical_expert_id != -1
        ]


# 全局专家位置元数据单例，系统启动时初始化一次，再平衡时就地更新
_global_expert_location_metadata: Optional[ExpertLocationMetadata] = None


def get_global_expert_location_metadata():
    # 获取全局专家位置元数据单例
    return _global_expert_location_metadata


def set_global_expert_location_metadata(value):
    # 初始化全局专家位置元数据（只允许设置一次）
    global _global_expert_location_metadata
    assert _global_expert_location_metadata is None
    _global_expert_location_metadata = value


def broadcast_global_expert_location_metadata(
    src_rank: int = 0, group: Optional[torch.distributed.ProcessGroup] = None
):
    """Broadcast the global ExpertLocationMetadata from src_rank to all ranks.

    This is used in Elastic EP rank recovery to ensure that all ranks (including
    newly recovered ones) share exactly the same expert location metadata.

    Note: The caller must ensure src_rank is a healthy rank. In recovery scenarios,
    this function is called after try_recover_ranks succeeds, at which point all
    ranks (including src_rank=0) have recovered and are ready.
    """
    metadata = get_global_expert_location_metadata()
    assert metadata is not None

    # Ensure device tensors are contiguous before broadcasting in-place
    # 广播前确保所有 GPU 张量连续存储，满足 torch.distributed.broadcast 的要求
    metadata.physical_to_logical_map = metadata.physical_to_logical_map.contiguous()
    metadata.logical_to_all_physical_map = (
        metadata.logical_to_all_physical_map.contiguous()
    )
    metadata.logical_to_all_physical_map_num_valid = (
        metadata.logical_to_all_physical_map_num_valid.contiguous()
    )
    if metadata.logical_to_rank_dispatch_physical_map is not None:
        metadata.logical_to_rank_dispatch_physical_map = (
            metadata.logical_to_rank_dispatch_physical_map.contiguous()
        )

    # 收集所有需要广播的 GPU 张量
    device_tensors = [
        metadata.physical_to_logical_map,
        metadata.logical_to_all_physical_map,
        metadata.logical_to_all_physical_map_num_valid,
    ]
    if metadata.logical_to_rank_dispatch_physical_map is not None:
        device_tensors.append(metadata.logical_to_rank_dispatch_physical_map)

    # 逐张量广播，从 src_rank 发送到所有 rank
    for tensor in device_tensors:
        torch.distributed.broadcast(tensor, src=src_rank, group=group)

    # After broadcasting device tensors, refresh corresponding CPU copies
    # 广播后同步更新 CPU 缓存副本
    metadata.physical_to_logical_map_cpu = metadata.physical_to_logical_map.cpu()
    metadata.logical_to_all_physical_map_cpu = (
        metadata.logical_to_all_physical_map.cpu()
    )



def _compute_logical_to_all_physical_map(
    server_args: ServerArgs,
    physical_to_logical_map: torch.Tensor,
    num_logical_experts: int,
    ep_size: int,
    moe_ep_rank: int,
):
    # This is rarely called, so we use for loops for maximum clarity
    # 此函数调用频率低，使用 for 循环以保证清晰度，无需追求极致性能

    num_layers, num_physical_experts = physical_to_logical_map.shape

    # 初始化嵌套列表：[层][逻辑专家] -> 物理专家 ID 列表
    logical_to_all_physical_map = [
        [[] for _ in range(num_logical_experts)] for _ in range(num_layers)
    ]

    # Find out the candidate physical experts for each logical expert on each layer
    # 遍历所有物理专家，将其物理 ID 添加到对应逻辑专家的候选列表中
    for layer_id in range(num_layers):
        for physical_expert_id in range(num_physical_experts):
            logical_expert_id = physical_to_logical_map[
                layer_id, physical_expert_id
            ].item()
            logical_to_all_physical_map[layer_id][logical_expert_id].append(
                physical_expert_id
            )

    # Replace by the physical expert on local GPU or node if possible
    # 若指定了 moe_ep_rank，则用最近的物理副本替换候选列表（优先本 GPU，其次同节点）
    if moe_ep_rank is not None:
        num_gpus_per_node = server_args.ep_size // server_args.nnodes
        num_local_gpu_physical_experts = num_physical_experts // ep_size
        num_local_node_physical_experts = (
            num_local_gpu_physical_experts * num_gpus_per_node
        )
        for layer_id in range(num_layers):
            for logical_expert_id in range(num_logical_experts):
                # Try to find the nearest physical expert
                # 寻找距离当前 rank 最近的物理副本（本 GPU > 同节点 > 跨节点）
                nearest_expert = _find_nearest_expert(
                    candidate_physical_expert_ids=logical_to_all_physical_map[layer_id][
                        logical_expert_id
                    ],
                    num_local_gpu_physical_experts=num_local_gpu_physical_experts,
                    moe_ep_rank=moe_ep_rank,
                    num_gpus_per_node=num_gpus_per_node,
                    num_local_node_physical_experts=num_local_node_physical_experts,
                )

                # Replace by the nearest physical expert
                # 若找到了就近副本，则将候选列表替换为单元素列表（只保留最近的）
                if nearest_expert != -1:
                    logical_to_all_physical_map[layer_id][logical_expert_id] = [
                        nearest_expert
                    ]

    # 对齐各逻辑专家的候选列表长度（用 -1 填充到最大长度）
    logical_to_all_physical_map = _pad_nested_array(
        logical_to_all_physical_map, pad_value=-1
    )

    return torch.tensor(
        logical_to_all_physical_map, device=physical_to_logical_map.device
    )


def _pad_nested_array(arr, pad_value):
    # 将嵌套列表中的每个内层列表填充到相同长度（用 pad_value 补齐）
    max_len = max(len(inner) for outer in arr for inner in outer)
    padded = [
        [inner + [pad_value] * (max_len - len(inner)) for inner in outer]
        for outer in arr
    ]
    return padded


# TODO optimize performance (rewrite and/or run in separate process with overlap)
def compute_logical_to_rank_dispatch_physical_map(
    server_args: ServerArgs,
    logical_to_all_physical_map: torch.Tensor,
    ep_size: int,
    num_physical_experts: int,
    ep_rank: int,
    seed: int = 42,
):
    # 为每个 rank 预计算逻辑→物理的静态调度映射，保证每个 rank 优先使用本地/邻近副本
    # 对于没有就近副本的情况，用带种子的随机均匀分配（保证负载均衡）
    r = random.Random(seed)

    device = logical_to_all_physical_map.device
    # 在 CPU 上计算，避免 GPU 同步开销
    logical_to_all_physical_map = logical_to_all_physical_map.cpu()

    num_local_gpu_physical_experts = num_physical_experts // ep_size
    num_gpus_per_node = server_args.ep_size // server_args.nnodes
    num_local_node_physical_experts = num_local_gpu_physical_experts * num_gpus_per_node
    num_layers, num_logical_experts, _ = logical_to_all_physical_map.shape
    dtype = logical_to_all_physical_map.dtype

    # result_list[rank][layer][logical_expert] = 该 rank 对该逻辑专家的首选物理专家 ID
    result_list = [
        [[-1] * num_logical_experts for _ in range(num_layers)] for _ in range(ep_size)
    ]

    for layer_id in range(num_layers):
        for logical_expert_id in range(num_logical_experts):
            # 获取该逻辑专家所有可用物理副本（过滤 -1）
            candidate_physical_expert_ids = _logical_to_all_physical_raw(
                logical_to_all_physical_map, layer_id, logical_expert_id
            )

            remaining_ranks = []
            for moe_ep_rank in range(ep_size):
                # 为每个 rank 寻找最近的物理副本
                val = _find_nearest_expert(
                    candidate_physical_expert_ids=candidate_physical_expert_ids,
                    num_local_gpu_physical_experts=num_local_gpu_physical_experts,
                    moe_ep_rank=moe_ep_rank,
                    num_gpus_per_node=num_gpus_per_node,
                    num_local_node_physical_experts=num_local_node_physical_experts,
                )

                result_list[moe_ep_rank][layer_id][logical_expert_id] = val
                # val == -1 表示该 rank 没有就近副本，需要公平随机分配
                if val == -1:
                    remaining_ranks.append(moe_ep_rank)

            if remaining_ranks:
                # 对没有就近副本的 rank，从候选物理副本中公平地随机分配
                choices = _fair_choices(
                    candidate_physical_expert_ids, k=len(remaining_ranks), r=r
                )
                for moe_ep_rank, choice in zip(remaining_ranks, choices, strict=True):
                    result_list[moe_ep_rank][layer_id][logical_expert_id] = choice

    logical_to_rank_dispatch_physical_map = torch.tensor(result_list, dtype=dtype)
    # 校验：所有 rank 的所有逻辑专家都已分配到有效物理副本
    assert torch.all(logical_to_rank_dispatch_physical_map != -1)

    # 只返回当前 ep_rank 视角的切片，形状为 [layers, num_logical_experts]
    return logical_to_rank_dispatch_physical_map[ep_rank, :, :].to(device)



def _logical_to_all_physical_raw(
    logical_to_all_physical_map, layer_id: int, logical_expert_id: int
) -> List[int]:
    # 从映射张量中读取某逻辑专家的所有有效物理副本 ID（过滤 -1 填充）
    return [
        physical_expert_id
        for physical_expert_id in logical_to_all_physical_map[
            layer_id, logical_expert_id
        ].tolist()
        if physical_expert_id != -1
    ]


def _compute_gpu_id_of_physical_expert(
    physical_expert_id: int, num_local_gpu_physical_experts: int
) -> int:
    # 根据物理专家 ID 计算其所在 GPU 的编号
    return physical_expert_id // num_local_gpu_physical_experts


def _compute_node_id_of_physical_expert(
    physical_expert_id: int, num_local_host_physical_experts: int
) -> int:
    # 根据物理专家 ID 计算其所在节点的编号
    return physical_expert_id // num_local_host_physical_experts


def _find_nearest_expert(
    candidate_physical_expert_ids: List[int],  # 某逻辑专家的所有候选物理副本 ID
    num_local_gpu_physical_experts: int,        # 每 GPU 本地专家数（用于计算 GPU ID）
    moe_ep_rank: int,                           # 当前 rank（视角）
    num_gpus_per_node: int,                     # 每节点 GPU 数（用于计算节点 ID）
    num_local_node_physical_experts: int,       # 每节点本地专家数（用于计算节点 ID）
) -> int:
    # 1. If only one candidate, return it directly
    # 只有一个候选副本时直接返回，无需比较
    if len(candidate_physical_expert_ids) == 1:
        return candidate_physical_expert_ids[0]

    # 2. Prefer same-GPU experts
    # 优先选择与当前 rank 同 GPU 的物理副本（通信代价最低）
    same_gpu_physical_expert_ids = [
        physical_expert_id
        for physical_expert_id in candidate_physical_expert_ids
        if _compute_gpu_id_of_physical_expert(
            physical_expert_id, num_local_gpu_physical_experts
        )
        == moe_ep_rank
    ]
    if len(same_gpu_physical_expert_ids) > 0:
        return same_gpu_physical_expert_ids[0]

    # 3. Otherwise, prefer same-node experts
    # 其次选择同节点内的物理副本（节点内 NVLink 比跨节点网络快）
    node_rank = moe_ep_rank // num_gpus_per_node
    same_node_physical_expert_ids = [
        physical_expert_id
        for physical_expert_id in candidate_physical_expert_ids
        if _compute_node_id_of_physical_expert(
            physical_expert_id, num_local_node_physical_experts
        )
        == node_rank
    ]
    if len(same_node_physical_expert_ids) > 0:
        return same_node_physical_expert_ids[0]

    # 4. At last, leave it as -1 to indicate not found.
    # 无同 GPU 也无同节点副本，返回 -1，由调用方处理（如随机分配）
    return -1


def _fair_choices(arr: List, k: int, r: random.Random) -> List:
    # 从 arr 中公平地随机选 k 个元素（循环复制后 shuffle，保证均匀分布）
    quotient, remainder = divmod(k, len(arr))
    ans = arr * quotient + r.sample(arr, k=remainder)
    r.shuffle(ans)
    return ans


@dataclass
class ModelConfigForExpertLocation:
    # 从 ModelConfig 中提取的专家位置相关配置：层数、逻辑专家数、分组数
    num_layers: int
    num_logical_experts: int
    num_groups: Optional[int] = None

    @staticmethod
    def from_model_config(model_config: ModelConfig):
        # 从模型类的静态方法中获取专家位置配置（若模型不支持则返回 None）
        from sglang.srt.model_loader import get_model_architecture

        model_class, _ = get_model_architecture(model_config)
        if hasattr(model_class, "get_model_config_for_expert_location"):
            return model_class.get_model_config_for_expert_location(
                model_config.hf_config
            )
        else:
            return None


def compute_initial_expert_location_metadata(
    server_args: ServerArgs,
    model_config: ModelConfig,
    moe_ep_rank: int,
) -> Optional[ExpertLocationMetadata]:
    # 根据 server_args.init_expert_location 决定初始化方式
    data = server_args.init_expert_location
    if data == "trivial":
        # "trivial" 模式：物理专家 i 直接映射到逻辑专家 i（不做负载均衡）
        return ExpertLocationMetadata.init_trivial(
            server_args, model_config, moe_ep_rank
        )

    # TODO unify with the utils function
    # 支持从 .pt 文件、.json 文件或内联 JSON 字符串加载初始映射
    if data.endswith(".pt"):
        data_dict = torch.load(data, weights_only=True)
    elif data.endswith(".json"):
        data_dict = json.loads(Path(data).read_text())
    else:
        data_dict = json.loads(data)

    if "physical_to_logical_map" in data_dict:
        # 提供了物理→逻辑映射，直接使用该映射初始化
        logger.info(
            "init_expert_location from init_by_mapping using ServerArgs.init_expert_location"
        )
        return ExpertLocationMetadata.init_by_mapping(
            server_args,
            model_config,
            **data_dict,
            moe_ep_rank=moe_ep_rank,
        )
    elif "logical_count" in data_dict:
        # 提供了历史 token 计数，运行 EPLB 算法计算初始映射
        logger.info(
            "init_expert_location from init_by_eplb using ServerArgs.init_expert_location"
        )
        return ExpertLocationMetadata.init_by_eplb(
            server_args, model_config, logical_count=data_dict["logical_count"]
        )
    else:
        raise NotImplementedError(
            f"Unknown init_expert_location format ({list(data_dict.keys())=})"
        )
