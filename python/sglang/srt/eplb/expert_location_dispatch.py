# Copyright 2023-2025 SGLang Team
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
# 专家位置调度模块：将路由器选出的逻辑专家 ID 映射为物理专家 ID
# 支持静态（每个逻辑专家固定分配到某 rank）和动态（随机负载均衡）两种调度策略

from dataclasses import dataclass
from typing import Literal, Optional

import torch

# 全局专家位置元数据：包含逻辑→物理映射的全局单例
from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
from sglang.srt.server_args import get_global_server_args


@dataclass
class ExpertLocationDispatchInfo:
    # 调度算法类型：static（静态）或 random/dynamic（动态随机）
    ep_dispatch_algorithm: Literal["static", "random"]
    # (num_logical_experts,)：静态调度时，每个逻辑专家对应的 rank 上的物理专家 ID
    partial_logical_to_rank_dispatch_physical_map: Optional[torch.Tensor]
    # (num_logical_experts, X)：动态调度时，每个逻辑专家所有可用物理副本的 ID 列表
    partial_logical_to_all_physical_map: torch.Tensor
    # (num_logical_experts,)：上面 all_physical_map 中每行的有效副本数（非 -1 的数量）
    partial_logical_to_all_physical_map_num_valid: torch.Tensor
    # 本层的物理专家总数
    num_physical_experts: int

    @classmethod
    def init_new(cls, layer_id: int):
        # 从全局配置和全局专家位置元数据中构建当前层的调度信息
        ep_dispatch_algorithm = get_global_server_args().ep_dispatch_algorithm
        expert_location_metadata = get_global_expert_location_metadata()
        assert expert_location_metadata is not None

        # 若未配置调度算法，返回 None（走默认路由，不做逻辑→物理映射）
        if ep_dispatch_algorithm is None:
            return None

        return cls(
            ep_dispatch_algorithm=ep_dispatch_algorithm,
            # 切片出当前层的静态调度映射（若不存在则为 None）
            partial_logical_to_rank_dispatch_physical_map=(
                expert_location_metadata.logical_to_rank_dispatch_physical_map[
                    layer_id, :
                ]
                if expert_location_metadata.logical_to_rank_dispatch_physical_map
                is not None
                else None
            ),
            # 切片出当前层的全量物理副本映射（用于动态调度）
            partial_logical_to_all_physical_map=expert_location_metadata.logical_to_all_physical_map[
                layer_id, :
            ],
            # 切片出当前层各逻辑专家的有效副本数
            partial_logical_to_all_physical_map_num_valid=expert_location_metadata.logical_to_all_physical_map_num_valid[
                layer_id, :
            ],
            num_physical_experts=expert_location_metadata.num_physical_experts,
        )


def transform_select_experts_inputs(
    router_logits: torch.Tensor,
    correction_bias: Optional[torch.Tensor],
    info: Optional[ExpertLocationDispatchInfo],
):
    # fake 模式：将 router_logits 随机化为 [5, 10] 均匀分布，并将 correction_bias 清零
    # 用于压力测试或调试场景，强制路由到随机专家
    if (info is not None) and (info.ep_dispatch_algorithm == "fake"):
        router_logits.uniform_(5, 10)
        if correction_bias is not None:
            correction_bias = torch.zeros_like(correction_bias)
    return router_logits, correction_bias


def topk_ids_logical_to_physical(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    # 将路由器输出的逻辑专家 ID 转换为实际物理专家 ID
    if info is None:
        # 无调度信息时直接返回原始逻辑专家 ID（不做映射）
        return topk_ids

    if info.ep_dispatch_algorithm == "static":
        # 静态调度：每个逻辑专家固定映射到特定 rank 的物理专家
        return _topk_ids_logical_to_physical_static(topk_ids, info)
    if info.ep_dispatch_algorithm in ["dynamic", "fake"]:
        # 动态/fake 调度：从各逻辑专家的多个物理副本中随机选一个
        return _topk_ids_logical_to_physical_dynamic(topk_ids, info)
    raise NotImplementedError(f"Unknown algorithm {info.ep_dispatch_algorithm}")


def _topk_ids_logical_to_physical_static(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    # 静态调度实现：直接用逻辑 ID 查表，返回对应 rank 上的物理专家 ID
    return info.partial_logical_to_rank_dispatch_physical_map[topk_ids]


def _topk_ids_logical_to_physical_dynamic(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    # 动态调度实现：对每个 token 被路由到的逻辑专家，随机选取其某个物理副本
    topk_ids_original_shape = topk_ids.shape
    device = topk_ids.device
    # 展平处理，方便一维索引操作
    topk_ids = topk_ids.flatten()

    # 在 [0, 65536) 范围内生成随机整数，取模有效副本数，得到随机选择的副本下标
    chosen_dispatch_index = (
        torch.randint(0, 65536, topk_ids.shape, dtype=torch.int32, device=device)
        % info.partial_logical_to_all_physical_map_num_valid[topk_ids]
    )
    # 用 (逻辑专家 ID, 随机副本下标) 二维索引取出实际物理专家 ID
    topk_ids = info.partial_logical_to_all_physical_map[topk_ids, chosen_dispatch_index]

    # 恢复原始形状后返回
    topk_ids = topk_ids.view(topk_ids_original_shape)
    return topk_ids
