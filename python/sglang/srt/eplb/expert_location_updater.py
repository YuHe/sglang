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
# 专家位置更新模块：在 EPLB 再平衡后，通过 P2P 通信将专家权重迁移到新的物理位置
# 核心流程：根据旧/新的物理→逻辑映射，计算需要发送/接收的专家权重，并执行通信
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import einops
import torch
import torch.distributed
from torch.distributed import P2POp

# 弹性 EP 状态管理器：用于判断哪些 rank 当前处于活跃状态
from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
# 专家位置元数据及全局实例获取接口
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    get_global_expert_location_metadata,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)


# 调试开关：若环境变量为真，则在每次更新前打印详细输入日志
_LOG_INPUT = get_bool_env_var("SGLANG_EXPERT_LOCATION_UPDATER_LOG_INPUT")


class ExpertLocationUpdater:
    # 专家位置更新器：封装单层/多层专家权重迁移的完整流程
    def __init__(self):
        # 首次执行标记，用于首次调用时清理 GPU 缓存
        self._first_execution = True

    def update(
        self,
        routed_experts_weights_of_layer: Dict[int, List[torch.Tensor]],  # 各层路由专家权重
        new_expert_location_metadata: ExpertLocationMetadata,             # 再平衡后的新专家位置元数据
        update_layer_ids: List[int],   # 本次需要更新的层 ID 列表
        nnodes: int,                   # 节点总数
        rank: int,                     # 当前进程的 rank
    ):
        """
        Update experts' physical location after EPLB.

        Returns a map of layer_id to expert_ids that are missing due to rank
        failures during fault conditions when elastic EP is enabled.
        """
        # 首次执行时清空 GPU 缓存，释放再平衡前的临时内存
        if self._first_execution:
            self._first_execution = False
            torch.get_device_module().empty_cache()

        # 获取当前（旧）专家位置元数据
        old_expert_location_metadata = get_global_expert_location_metadata()
        assert old_expert_location_metadata is not None

        # 执行专家权重迁移（通过 P2P 通信将专家从旧位置搬到新位置）
        missing_logical_experts_by_layers = _update_expert_weights(
            routed_experts_weights_of_layer=routed_experts_weights_of_layer,
            old_expert_location_metadata=old_expert_location_metadata,
            new_expert_location_metadata=new_expert_location_metadata,
            update_layer_ids=update_layer_ids,
            nnodes=nnodes,
            rank=rank,
        )
        # 将全局元数据就地更新为新的专家位置映射
        old_expert_location_metadata.update(
            new_expert_location_metadata,
            update_layer_ids=update_layer_ids,
        )

        # 返回因 rank 故障导致缺失的逻辑专家信息（弹性 EP 场景下使用）
        return missing_logical_experts_by_layers



def _update_expert_weights(**kwargs):
    # 根据环境变量决定是否启用金丝雀（canary）验证模式
    if get_bool_env_var("SGLANG_EXPERT_LOCATION_UPDATER_CANARY"):
        return _update_expert_weights_with_canary(**kwargs)
    else:
        return _update_expert_weights_raw(**kwargs)


# can add watchdog as well
def _update_expert_weights_with_canary(
    routed_experts_weights_of_layer: Dict[int, List[torch.Tensor]],
    old_expert_location_metadata: ExpertLocationMetadata,
    new_expert_location_metadata: ExpertLocationMetadata,
    update_layer_ids: List[int],
    nnodes: int,
    rank: int,
):
    # 金丝雀验证模式：在每层末尾附加一个金丝雀张量，更新后校验其内容是否符合预期
    # 若校验失败则说明专家迁移通信出现了数据错误
    num_local_physical_experts = old_expert_location_metadata.num_local_physical_experts

    def _get_canary_value(meta: ExpertLocationMetadata, layer_id: int):
        # 取当前 rank 负责的本地物理专家的逻辑专家 ID 切片作为金丝雀值
        return meta.physical_to_logical_map_cpu[
            layer_id,
            num_local_physical_experts * rank : num_local_physical_experts * (rank + 1),
        ]

    # 浅复制权重字典，避免修改原始数据结构
    routed_experts_weights_of_layer = {
        k: [x for x in v] for k, v in routed_experts_weights_of_layer.items()
    }
    for layer_id in update_layer_ids:
        # 将旧的金丝雀值（当前 rank 的 phy2log 切片）追加到权重列表末尾
        canary_tensor = (
            _get_canary_value(old_expert_location_metadata, layer_id)
            .clone()
            .to(device=get_global_server_args().device, non_blocking=True)
        )
        routed_experts_weights_of_layer[layer_id].append(canary_tensor)

    # 执行实际的专家权重迁移（金丝雀张量也会被正常迁移）
    missing_logical_experts_by_layers = _update_expert_weights_raw(
        routed_experts_weights_of_layer=routed_experts_weights_of_layer,
        old_expert_location_metadata=old_expert_location_metadata,
        new_expert_location_metadata=new_expert_location_metadata,
        update_layer_ids=update_layer_ids,
        nnodes=nnodes,
        rank=rank,
    )

    for layer_id in update_layer_ids:
        # can optimize speed if needed
        # 校验：更新后的金丝雀张量应等于新元数据中该 rank 的 phy2log 切片
        expect_value = _get_canary_value(new_expert_location_metadata, layer_id)
        actual_value = routed_experts_weights_of_layer[layer_id][-1].cpu()
        assert torch.all(expect_value == actual_value), (
            f"{expect_value=} {actual_value=} {layer_id=} "
            f"{old_expert_location_metadata.physical_to_logical_map_cpu.tolist()=} "
            f"{new_expert_location_metadata.physical_to_logical_map_cpu.tolist()=} "
        )

    return missing_logical_experts_by_layers



def _update_expert_weights_raw(
    routed_experts_weights_of_layer: Dict[int, List[torch.Tensor]],
    old_expert_location_metadata: ExpertLocationMetadata,
    new_expert_location_metadata: ExpertLocationMetadata,
    update_layer_ids: List[int],
    nnodes: int,
    rank: int,
):
    # 是否打印每层 P2P 通信量统计
    log_metrics = get_bool_env_var("SGLANG_EXPERT_LOCATION_UPDATER_LOG_METRICS")

    # 创建临时缓冲区（与第一层权重形状一致），用于接收来自其他 rank 的专家权重
    temp_buffers = create_temp_buffers(
        routed_experts_weights_of_layer[update_layer_ids[0]]
    )

    world_size = torch.distributed.get_world_size()
    num_local_physical_experts = old_expert_location_metadata.num_local_physical_experts
    # 每节点的 GPU 数（用于判断通信是节点内还是跨节点）
    num_gpu_per_node = world_size // nnodes

    # 收集各层中因 rank 故障导致缺失的逻辑专家
    missing_logical_experts_by_layers: Dict[int, List[int]] = {}

    for layer_id in update_layer_ids:
        missing_logical_experts_info: List[int] = []
        # 对单层执行专家权重迁移，结果直接写入 routed_experts_weights_of_layer
        update_expert_weights_single_layer(
            routed_experts_weights=routed_experts_weights_of_layer[layer_id],
            temp_buffers=temp_buffers,
            old_physical_to_logical_map=old_expert_location_metadata.physical_to_logical_map_cpu[
                layer_id
            ].tolist(),
            new_physical_to_logical_map=new_expert_location_metadata.physical_to_logical_map_cpu[
                layer_id
            ].tolist(),
            num_local_physical_experts=num_local_physical_experts,
            num_gpu_per_node=num_gpu_per_node,
            rank=rank,
            world_size=world_size,
            missing_logical_experts_info=missing_logical_experts_info,
            log_metrics=log_metrics,
        )
        if len(missing_logical_experts_info) > 0:
            missing_logical_experts_by_layers[layer_id] = missing_logical_experts_info
    return missing_logical_experts_by_layers



def create_temp_buffers(sample_tensors):
    # 根据样本张量列表分配等形状的临时缓冲区（用于 P2P irecv 接收目标）
    return [torch.empty_like(tensor) for tensor in sample_tensors]


def update_expert_weights_single_layer(
    routed_experts_weights: List[torch.Tensor],     # 当前 rank 本地的专家权重列表
    temp_buffers: List[torch.Tensor],               # 临时接收缓冲区
    old_physical_to_logical_map: List[int],  # (num_physical_Experts,)
    new_physical_to_logical_map: List[int],  # (num_physical_Experts,)
    num_local_physical_experts: int,         # 当前 rank 本地物理专家数
    num_gpu_per_node: int,                   # 每节点的 GPU 数
    rank: int,                               # 当前 rank
    world_size: Optional[int] = None,
    missing_logical_experts_info: Optional[List[int]] = None,  # 输出：缺失专家 ID 列表
    debug: bool = False,
    log_metrics: bool = False,
):
    # 校验：所有权重张量的第一维必须等于本地物理专家数
    assert all(
        tensor.shape[0] == num_local_physical_experts
        for tensor in routed_experts_weights
    ), f"{num_local_physical_experts=} {[x.shape for x in routed_experts_weights]=}"
    assert isinstance(old_physical_to_logical_map, list)
    assert isinstance(new_physical_to_logical_map, list)

    if _LOG_INPUT:
        logger.info(
            "update_expert_weights_single_layer "
            f"{[x.shape for x in routed_experts_weights]=} "
            f"{[x.shape for x in temp_buffers]=} "
            f"{old_physical_to_logical_map=} "
            f"{new_physical_to_logical_map=} "
            f"{num_local_physical_experts=} "
            f"{num_gpu_per_node=} "
            f"{rank=} "
            f"{world_size=} "
        )

    # debug 模式下收集操作日志，便于问题追踪
    output_logs = [] if debug else None

    num_physical_experts = len(old_physical_to_logical_map)
    num_tensors = len(routed_experts_weights)

    # 当前 rank 所属节点 ID
    self_node_id = rank // num_gpu_per_node

    # 当前 rank 负责的本地物理专家位置范围（全局物理索引）
    local_expert_location_range = (
        rank * num_local_physical_experts,
        (rank + 1) * num_local_physical_experts,
    )

    def _entrypoint():
        # List[Tuple[logical_expert_id, List[P2POp]]]
        # p2p_op_infos：每个条目描述一个逻辑专家的一组 P2P 发送或接收操作
        p2p_op_infos: List[Tuple[int, List[P2POp]]] = []
        # List[Tuple[temp_buffers_expert_location, routed_experts_weights_expert_location]]
        # buffer2weight_copy_infos：记录需要从临时缓冲区复制到权重张量的位置对
        buffer2weight_copy_infos: List[Tuple[int, int]] = []

        # 步骤1：为本 rank 所有目标物理槽位生成接收操作（irecv 或本地拷贝）
        _handle_recv(buffer2weight_copy_infos, p2p_op_infos)
        # 步骤2：为本 rank 当前持有的专家生成发送操作（isend）
        _create_isend_ops(p2p_op_infos)
        # 步骤3：过滤掉涉及离线 rank 的 P2P 操作（弹性 EP 场景）
        _filter_p2p_ops(p2p_op_infos)
        # 步骤4：批量执行所有 P2P 发送/接收操作并等待完成
        _execute_p2p_ops(p2p_op_infos)
        # 步骤5：将临时缓冲区中接收到的权重复制到正式权重张量
        _execute_buffer2weight_copies(buffer2weight_copy_infos)

        if log_metrics:
            _log_p2p_op_metrics(
                p2p_op_infos,
                world_size=world_size,
                num_gpu_per_node=num_gpu_per_node,
                self_node_id=self_node_id,
            )

        if debug:
            output_logs.append(f"{p2p_op_infos=}")
            output_logs.append(f"{buffer2weight_copy_infos=}")

    def _handle_recv(buffer2weight_copy_infos, p2p_op_infos):
        # 对本 rank 每个目标物理槽位，处理其接收逻辑
        for dst_expert_location in range(*local_expert_location_range):
            _handle_recv_of_dst_expert_location(
                dst_expert_location, buffer2weight_copy_infos, p2p_op_infos
            )

    def _handle_recv_of_dst_expert_location(
        dst_expert_location: int, buffer2weight_copy_infos, p2p_op_infos
    ):
        # 确定该目标物理槽位在新映射中对应的逻辑专家 ID
        logical_expert_id = new_physical_to_logical_map[dst_expert_location]

        # case 1: unchanged
        # 该槽位的逻辑专家未变，无需任何操作
        if old_physical_to_logical_map[dst_expert_location] == logical_expert_id:
            if debug:
                output_logs.append(
                    f"handle_recv_of_dst_expert_location {dst_expert_location=} case=unchanged"
                )
            return

        # case 2: same-gpu
        # 所需逻辑专家在本 rank 的其他物理槽位上已存在，直接本地拷贝到临时缓冲区
        for src_expert_location in range(*local_expert_location_range):
            if old_physical_to_logical_map[src_expert_location] == logical_expert_id:
                for i in range(num_tensors):
                    _get_tensor(temp_buffers, i, dst_expert_location).copy_(
                        _get_tensor(routed_experts_weights, i, src_expert_location)
                    )
                buffer2weight_copy_infos.append(
                    (dst_expert_location, dst_expert_location)
                )
                if debug:
                    output_logs.append(
                        f"handle_recv_of_dst_expert_location {dst_expert_location=} case=same-gpu {src_expert_location=}"
                    )
                return

        # case 3: free-rider
        # 本 rank 前面的某个槽位已经接收了相同逻辑专家（同一逻辑专家被本 rank 复制多份），
        # 直接从那个已接收的临时缓冲区复制，避免重复通信
        for src_expert_location in range(
            rank * num_local_physical_experts, dst_expert_location
        ):
            if new_physical_to_logical_map[src_expert_location] == logical_expert_id:
                buffer2weight_copy_infos.append(
                    (src_expert_location, dst_expert_location)
                )
                if debug:
                    output_logs.append(
                        f"handle_recv_of_dst_expert_location {dst_expert_location=} case=free-rider {src_expert_location=}"
                    )
                return

        # 计算该逻辑专家的通信信息（哪些 rank 发送、哪些 rank 接收）
        same_node_mapping, cross_node_mapping, need_comm_self_node_dst_ranks = (
            _compute_comm_info(logical_expert_id=logical_expert_id)
        )

        # case 4: same-node
        # 所需逻辑专家需要从同节点的另一个 rank 接收（节点内 NVLink 通信）
        if rank in need_comm_self_node_dst_ranks:
            chosen_src_rank = same_node_mapping.chunk_value_from_element_value(
                element_value=rank
            )
            _create_p2p_recv_and_buffer2weight_copy(
                buffer2weight_copy_infos,
                p2p_op_infos,
                src_rank=chosen_src_rank,
                logical_expert_id=logical_expert_id,
                dst_expert_location=dst_expert_location,
            )
            if debug:
                output_logs.append(
                    f"handle_recv_of_dst_expert_location {dst_expert_location=} case=same-node {chosen_src_rank=}"
                )
            return

        # case 5: cross-node
        # Future work: can optimize when there are multiple ranks in the same dst node that uses the same logical expert
        # 所需逻辑专家需要从跨节点的 rank 接收（跨节点网络通信，带宽较低）
        chosen_src_rank = cross_node_mapping.chunk_value_from_element_value(
            element_value=rank
        )
        _create_p2p_recv_and_buffer2weight_copy(
            buffer2weight_copy_infos,
            p2p_op_infos,
            src_rank=chosen_src_rank,
            logical_expert_id=logical_expert_id,
            dst_expert_location=dst_expert_location,
        )
        if debug:
            output_logs.append(
                f"handle_recv_of_dst_expert_location {dst_expert_location=} case=cross-node {chosen_src_rank=}"
            )
        return

    def _create_p2p_recv_and_buffer2weight_copy(
        buffer2weight_copy_infos,
        p2p_op_infos,
        *,
        logical_expert_id: int,
        src_rank: int,
        dst_expert_location: int,
    ):
        # 创建从 src_rank 接收专家权重的 irecv 操作，目标写入临时缓冲区
        p2p_op_infos.append(
            (
                logical_expert_id,
                [
                    P2POp(
                        op=torch.distributed.irecv,
                        tensor=_get_tensor(temp_buffers, i, dst_expert_location),
                        peer=src_rank,
                    )
                    for i in range(num_tensors)
                ],
            )
        )
        # 接收完成后，将临时缓冲区的数据复制到正式权重张量
        buffer2weight_copy_infos.append((dst_expert_location, dst_expert_location))

    def _create_isend_ops(p2p_op_infos):
        handled_logical_expert_ids = set()
        for src_expert_location in range(*local_expert_location_range):
            # 当前本地物理槽位在旧映射中对应的逻辑专家
            logical_expert_id = old_physical_to_logical_map[src_expert_location]

            # 同一逻辑专家在本 rank 可能有多个物理副本，只处理一次
            if logical_expert_id in handled_logical_expert_ids:
                continue
            handled_logical_expert_ids.add(logical_expert_id)

            _create_isend_ops_of_logical_expert_id(
                logical_expert_id, src_expert_location, p2p_op_infos
            )

    def _create_isend_ops_of_logical_expert_id(
        logical_expert_id, src_expert_location, p2p_op_infos
    ):
        same_node_mapping, cross_node_mapping, need_comm_self_node_dst_ranks = (
            _compute_comm_info(logical_expert_id=logical_expert_id)
        )

        # 计算需要向哪些节点内 rank 和跨节点 rank 发送该专家权重
        same_node_dst_ranks = same_node_mapping.element_values_from_chunk_value(
            chunk_value=rank
        )
        cross_node_dst_ranks = cross_node_mapping.element_values_from_chunk_value(
            chunk_value=rank
        )
        all_dst_ranks = same_node_dst_ranks + cross_node_dst_ranks

        if debug:
            output_logs.append(
                f"create_isend_ops_of_logical_expert_id {logical_expert_id=} {src_expert_location=} {same_node_dst_ranks=} {cross_node_dst_ranks=}"
            )

        # 为每个目标 rank 创建 isend 操作，发送本 rank 持有的该逻辑专家权重
        p2p_op_infos.append(
            (
                logical_expert_id,
                [
                    P2POp(
                        op=torch.distributed.isend,
                        tensor=_get_tensor(
                            routed_experts_weights, i, src_expert_location
                        ),
                        peer=dst_rank,
                    )
                    for dst_rank in all_dst_ranks
                    for i in range(num_tensors)
                ],
            )
        )

    def _compute_comm_info(logical_expert_id: int):
        # 找出旧映射中持有该逻辑专家的所有 rank（去重且保持顺序）
        all_src_ranks = _deduplicate_ordered(
            [
                x // num_local_physical_experts
                for x in range(num_physical_experts)
                if old_physical_to_logical_map[x] == logical_expert_id
            ]
        )
        all_src_nodes = [x // num_gpu_per_node for x in all_src_ranks]
        # 当前节点内持有该逻辑专家的 src rank 列表
        self_node_src_ranks = [
            x for x in all_src_ranks if x // num_gpu_per_node == self_node_id
        ]

        # 在新映射中需要该逻辑专家但旧映射中没有的 rank（需要接收的目标 rank）
        need_comm_dst_ranks = _deduplicate_ordered(
            [
                x // num_local_physical_experts
                for x in range(num_physical_experts)
                if new_physical_to_logical_map[x] == logical_expert_id
                and x // num_local_physical_experts not in all_src_ranks
            ]
        )
        # 节点内需要接收的 rank：只有节点内存在 src rank 时才从节点内获取（优先用 NVLink）
        need_comm_self_node_dst_ranks = (
            [x for x in need_comm_dst_ranks if x // num_gpu_per_node == self_node_id]
            if len(self_node_src_ranks) > 0
            else []
        )
        # 跨节点需要接收的 rank：整个节点都没有该专家，需要从别的节点获取
        need_comm_cross_node_dst_ranks = [
            x
            for x in need_comm_dst_ranks
            if (x // num_gpu_per_node) not in all_src_nodes
        ]

        # 构建节点内和跨节点的 src→dst rank 均衡分配映射工具
        same_node_mapping = _ChunkUtils(
            chunk_values=self_node_src_ranks,
            element_values=need_comm_self_node_dst_ranks,
        )

        cross_node_mapping = _ChunkUtils(
            chunk_values=all_src_ranks,
            element_values=need_comm_cross_node_dst_ranks,
        )

        return same_node_mapping, cross_node_mapping, need_comm_self_node_dst_ranks

    def _filter_p2p_ops(p2p_op_infos):
        elastic_ep_state = ElasticEPStateManager.instance()
        if elastic_ep_state is not None and missing_logical_experts_info is not None:
            # Filter out inactive P2P ops and record missing expert IDs in missing_logical_experts_info
            # 过滤掉涉及离线 rank 的 P2P 操作，并记录因此缺失的逻辑专家 ID
            is_active = elastic_ep_state.active_ranks_cpu
            for i, (logical_expert_id, ops) in enumerate(p2p_op_infos):
                has_isend = any(op.op == torch.distributed.isend for op in ops)
                has_irecv = any(op.op == torch.distributed.irecv for op in ops)
                assert not (has_isend and has_irecv), (
                    "Each p2p_op_infos entry is expected to contain only send "
                    "or only recv ops."
                )

                if has_isend:
                    # 发送方：只保留发往活跃 rank 的操作，离线 rank 的发送直接丢弃
                    p2p_op_infos[i] = (
                        logical_expert_id,
                        [op for op in ops if is_active[op.peer]],
                    )
                elif has_irecv:
                    # 接收方：若 src rank 离线，该逻辑专家数据将缺失，记录并清空操作列表
                    if any(not is_active[op.peer] for op in ops):
                        missing_logical_experts_info.append(logical_expert_id)
                        p2p_op_infos[i] = (logical_expert_id, [])

    def _execute_p2p_ops(p2p_op_infos):
        # 按逻辑专家 ID 排序以确保所有 rank 的操作顺序一致，避免死锁
        sorted_infos = sorted(p2p_op_infos, key=lambda info: info[0])
        p2p_ops = [op for _, ops in sorted_infos for op in ops]
        if len(p2p_ops) == 0:
            return

        # 批量提交所有 P2P 操作，然后等待全部完成
        reqs = torch.distributed.batch_isend_irecv(p2p_ops)
        for req in reqs:
            req.wait()

    def _execute_buffer2weight_copies(buffer2weight_copy_infos):
        # 将临时缓冲区中的接收数据复制回正式的权重张量
        for (
            temp_buffers_expert_location,
            routed_experts_weights_expert_location,
        ) in buffer2weight_copy_infos:
            for i in range(num_tensors):
                _get_tensor(
                    routed_experts_weights, i, routed_experts_weights_expert_location
                ).copy_(_get_tensor(temp_buffers, i, temp_buffers_expert_location))

    def _get_tensor(tensors, tensor_index: int, expert_location: int) -> torch.Tensor:
        # 通过全局物理专家位置索引取出对应的本地张量切片
        return tensors[tensor_index][_get_local_expert_location(expert_location)]

    def _get_local_expert_location(expert_location: int) -> int:
        # 将全局物理专家位置转为本 rank 内的局部索引（offset by rank * num_local_physical_experts）
        assert (
            local_expert_location_range[0]
            <= expert_location
            < local_expert_location_range[1]
        )
        return expert_location % num_local_physical_experts

    _entrypoint()

    return output_logs


class _ChunkUtils:
    """
    工具类：将 element_values（需要接收数据的目标 rank）均衡分配给 chunk_values（持有数据的 src rank）。
    实现 "尽量均匀分配负载" 的目标：每个 src rank 负责向大约相等数量的 dst rank 发送。
    """
    def __init__(self, *, chunk_values: List, element_values: List):
        # chunk_values：源 rank 列表（持有所需专家的 rank）
        self.chunk_values = chunk_values
        # element_values：目标 rank 列表（需要接收该专家的 rank）
        self.element_values = element_values

    def chunk_value_from_element_value(self, element_value):
        # 根据某个目标 rank 找到对应的源 rank（均衡分配后，该目标归属哪个 chunk）
        chunk_index = self._chunk_index_from_element_index(
            num_elements=len(self.element_values),
            num_chunks=len(self.chunk_values),
            element_index=self.element_values.index(element_value),
        )
        return self.chunk_values[chunk_index]

    def element_values_from_chunk_value(self, chunk_value) -> List:
        # 根据某个源 rank 找到它需要向哪些目标 rank 发送（该 chunk 对应的 element 切片）
        if len(self.element_values) == 0:
            return []
        element_slice = self._element_slice_from_chunk_index(
            num_elements=len(self.element_values),
            num_chunks=len(self.chunk_values),
            chunk_index=self.chunk_values.index(chunk_value),
        )
        return self.element_values[element_slice]

    @staticmethod
    def _chunk_index_from_element_index(
        num_elements: int, num_chunks: int, element_index: int
    ) -> int:
        # 将元素均衡分配到 chunk：前 num_long_chunks 个 chunk 各多分配一个元素
        short_chunk_size, num_long_chunks = divmod(num_elements, num_chunks)
        num_elements_for_long_chunks = num_long_chunks * (short_chunk_size + 1)
        if element_index < num_elements_for_long_chunks:
            # 该元素属于"长 chunk"区间
            return element_index // (short_chunk_size + 1)
        else:
            # 该元素属于"短 chunk"区间
            return (
                num_long_chunks
                + (element_index - num_elements_for_long_chunks) // short_chunk_size
            )

    @staticmethod
    def _element_slice_from_chunk_index(
        num_elements: int, num_chunks: int, chunk_index: int
    ) -> slice:
        # 计算第 chunk_index 个 chunk 对应的元素切片范围
        short_chunk_size, num_long_chunks = divmod(num_elements, num_chunks)
        # start 偏移量 = 前面所有 chunk 的累计元素数
        start = chunk_index * short_chunk_size + min(chunk_index, num_long_chunks)
        # 若当前 chunk 属于"长 chunk"则多一个元素
        end = start + short_chunk_size + int(chunk_index < num_long_chunks)
        return slice(start, end)


def _deduplicate_ordered(arr: List[int]):
    # 对已排序（或部分有序）的列表去除相邻重复元素，保持原始顺序
    output = []
    for item in arr:
        if len(output) == 0 or item != output[-1]:
            output.append(item)
    return output


def _log_p2p_op_metrics(
    p2p_op_infos: List[Tuple[int, List[P2POp]]],
    num_gpu_per_node: int,
    world_size: int,
    self_node_id: int,
):
    # 统计并打印每次专家迁移的 P2P 通信量（按 isend/irecv 分类，按 GPU 和节点汇总）
    text = ""
    all_ops = [op for _, ops in p2p_op_infos for op in ops]

    for direction, ops in _group_by(all_ops, _get_direction_from_op).items():
        # 统计每个 GPU 的 P2P 通信字节数
        nbytes_of_gpu = [0] * world_size
        for op in ops:
            nbytes_of_gpu[op.peer] += op.tensor.nbytes
        nbytes_of_gpu = torch.tensor(nbytes_of_gpu, dtype=torch.int64)

        # 将 GPU 粒度的字节数按节点聚合
        nbytes_of_node = einops.reduce(
            nbytes_of_gpu,
            "(num_nodes num_gpu_per_node) -> num_nodes",
            num_gpu_per_node=num_gpu_per_node,
            reduction="sum",
        )

        # 区分节点内通信量和跨节点通信量
        nbytes_curr_node = nbytes_of_node[self_node_id]
        nbytes_cross_node = torch.sum(nbytes_of_node) - nbytes_curr_node

        text += (
            f"{direction}_nbytes_of_gpu={nbytes_of_gpu.tolist()} "
            f"{direction}_nbytes_of_node={nbytes_of_node.tolist()} "
            f"{direction}_nbytes_curr_node={nbytes_curr_node.item()} "
            f"{direction}_nbytes_cross_node={nbytes_cross_node.item()} "
        )

    logger.info(f"[ExpertLocationUpdater] {text}")


def _get_direction_from_op(op: P2POp):
    # 根据 P2POp 的操作类型判断通信方向（isend 或 irecv）
    if op.op == torch.distributed.isend:
        return "isend"
    if op.op == torch.distributed.irecv:
        return "irecv"
    raise NotImplementedError


def _group_by(items, keyfunc):
    # 按 keyfunc 的返回值对 items 分组，返回 {key: [item, ...]} 字典
    ans = defaultdict(list)
    for item in items:
        ans[keyfunc(item)].append(item)
    return dict(ans)
