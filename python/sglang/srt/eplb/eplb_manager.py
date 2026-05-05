# EPLB 管理器模块：协调专家分布统计收集与负载均衡再平衡的调度逻辑
import logging
import time
from typing import TYPE_CHECKING, List

import torch.cuda

# 全局专家分布记录器：负责收集每次前向传播中各专家的 token 处理量
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
# 专家位置元数据：封装物理/逻辑专家映射关系，用于更新模型的专家路由
from sglang.srt.eplb.expert_location import ExpertLocationMetadata

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class EPLBManager:
    # EPLBManager 在每个 ModelRunner 上实例化，负责触发定期的专家再平衡
    def __init__(self, model_runner: "ModelRunner"):
        super().__init__()
        self._model_runner = model_runner
        self._server_args = model_runner.server_args
        # 每次再平衡时，分批更新的层数（None 表示一次性更新所有层）
        self._rebalance_layers_per_chunk = (
            self._server_args.eplb_rebalance_layers_per_chunk
        )
        # 触发再平衡的前向传播迭代间隔（每隔多少次前向传播执行一次再平衡）
        self._rebalance_num_iterations = self._server_args.eplb_rebalance_num_iterations

        # Otherwise, the circular buffer will contain stale data. If the case is needed, it can be implemented.
        # 校验：再平衡间隔必须 >= 统计窗口大小，否则循环缓冲区中会有过时数据
        assert (
            self._server_args.eplb_rebalance_num_iterations
            >= self._server_args.expert_distribution_recorder_buffer_size
        ), "eplb_rebalance_num_iterations must be greater than expert_distribution_recorder_buffer_size"

        # 若记录器尚未启动，则在此处启动专家分布统计
        if not get_global_expert_distribution_recorder().recording:
            get_global_expert_distribution_recorder().start_record()

        logger.info(
            f"[EPLBManager] system started, will rebalance per {self._rebalance_num_iterations} iterations."
        )

        # 初始化主协程生成器，用于在每次前向传播结束时推进调度逻辑
        self._main_generator = self._entrypoint()

    def on_forward_pass_end(self):
        # 每次前向传播结束后调用，推进内部生成器一步（计数或触发再平衡）
        next(self._main_generator)

    def reset_generator(self):
        # 重置生成器（例如在弹性 EP 状态变化后重新开始计数）
        self._main_generator = self._entrypoint()

    # can be more complex if needed
    def _entrypoint(self):
        # 无限循环：每经过 _rebalance_num_iterations 次前向传播后执行一次再平衡
        while True:
            for _ in range(self._rebalance_num_iterations):
                yield  # 等待下一次前向传播

            # 到达再平衡触发点，执行再平衡并 yield 等待分批更新完成
            yield from self.rebalance()

    def rebalance(self):
        logger.info("[EPLBManager] rebalance start")

        # 仅在不分批更新时启用计时（分批更新时中间有 yield 暂停，计时无意义）
        enable_timing = self._rebalance_layers_per_chunk is None

        if enable_timing:
            torch.get_device_module().synchronize()
            time_start = time.time()

        # 从全局记录器中导出统计结果：逻辑专家负载计数和窗口平均 GPU 利用率
        dump_record_output = get_global_expert_distribution_recorder().dump_record(
            output_mode="object"
        )
        logical_count = dump_record_output["logical_count"]
        average_utilization_rate_over_window = dump_record_output[
            "average_utilization_rate_over_window"
        ]

        # Check whether rebalancing is needed
        # 根据 GPU 利用率判断是否有必要执行再平衡（高负载下跳过以避免额外开销）
        if not self._check_rebalance_needed(average_utilization_rate_over_window):
            return

        # 基于最新的专家负载统计，计算新的物理/逻辑专家映射方案
        expert_location_metadata = ExpertLocationMetadata.init_by_eplb(
            self._server_args, self._model_runner.model_config, logical_count
        )

        # 将所有 MoE 层按 chunk 分批更新专家位置，减少单次更新对推理的阻塞时间
        update_layer_ids_chunks = self._compute_update_layer_ids_chunks()
        for chunk_index, update_layer_ids in enumerate(update_layer_ids_chunks):
            if len(update_layer_ids_chunks) > 1:
                yield  # 分批更新：每批之间插入 yield，让推理任务继续运行
            self._model_runner.update_expert_location(
                expert_location_metadata,
                update_layer_ids=update_layer_ids,
            )

        msg = f"[EPLBManager] rebalance end"
        if enable_timing:
            torch.get_device_module().synchronize()
            time_end = time.time()
            msg += f" time={time_end - time_start:.3f}s"
        logger.info(msg)

    def _check_rebalance_needed(self, average_utilization_rate_over_window):
        # 若窗口平均利用率不可用（数据不足），默认执行再平衡
        if average_utilization_rate_over_window is None:
            return True

        # 若当前 GPU 利用率已超过阈值，跳过本次再平衡（避免在高负载时增加额外延迟）
        if (
            average_utilization_rate_over_window
            > self._server_args.eplb_min_rebalancing_utilization_threshold
        ):
            logger.info(
                f"[EPLBManager] Skipped ep rebalancing: current GPU utilization {average_utilization_rate_over_window:.2f} > minimum rebalance threshold {self._server_args.eplb_min_rebalancing_utilization_threshold:.2f}"
            )
            return False

        return True

    def _compute_update_layer_ids_chunks(self) -> List[List[int]]:
        # 获取所有包含路由专家权重的层 ID，并按 chunk_size 分批
        all_layer_ids = sorted(
            list(self._model_runner.model.routed_experts_weights_of_layer.keys())
        )
        # chunk_size 为 None 时设为极大值，效果等同于一次性更新所有层
        chunk_size = self._rebalance_layers_per_chunk or 1000000
        return list(_chunk_list(all_layer_ids, chunk_size=chunk_size))


def _chunk_list(items: List, chunk_size):
    # 生成器：将列表 items 按 chunk_size 切分为多个子列表逐个 yield
    for start_index in range(0, len(items), chunk_size):
        yield items[start_index : start_index + chunk_size]
