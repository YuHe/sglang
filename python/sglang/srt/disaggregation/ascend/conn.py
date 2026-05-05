# 华为昇腾（Ascend）KV 传输连接模块：基于 MooncakeKV 框架适配昇腾 NPU 传输引擎
import concurrent.futures
import logging
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

# 导入昇腾专用传输引擎
from sglang.srt.disaggregation.ascend.transfer_engine import AscendTransferEngine
# 用于将 KV 索引按连续段分组，减少传输调用次数
from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous
# 继承 Mooncake 的 KV 管理/发送/接收/引导服务器实现
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVBootstrapServer,
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
)
from sglang.srt.utils.network import get_local_ip_auto

logger = logging.getLogger(__name__)


# 昇腾 KV 管理器：覆盖引擎初始化和缓冲区注册，适配昇腾 NPU
class AscendKVManager(MooncakeKVManager):
    def init_engine(self):
        # TransferEngine initialized on ascend.
        # 获取本机 IP 并创建 AscendTransferEngine，指定 NPU 设备 ID 和 PD 分离模式
        local_ip = get_local_ip_auto()
        self.engine = AscendTransferEngine(
            hostname=local_ip,
            npu_id=self.kv_args.gpu_id,
            disaggregation_mode=self.disaggregation_mode,
        )

    def register_buffer_to_engine(self):
        # 向昇腾传输引擎批量注册 KV 数据缓冲区的指针和长度
        self.engine.batch_register(self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens)
        # The Ascend backend optimize batch registration for small memory blocks.
        # 昇腾后端对小内存块的批量注册进行了优化，同时注册 aux 数据缓冲区
        self.engine.batch_register(
            self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
        )
        # Batch register state/extra pool data buffers
        # 若存在 state 数据缓冲区（如推测解码的 state），也一并注册
        if self.kv_args.state_data_ptrs and self.kv_args.state_data_lens:
            self.engine.batch_register(
                self.kv_args.state_data_ptrs, self.kv_args.state_data_lens
            )

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        # Group by indices
        # 将 prefill 侧和 decode 侧的 KV 索引按连续段分组，合并相邻块以减少传输次数
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )

        # 区分多流水线并行（PP>1）和单阶段两种情况，构建每层的源/目标指针和 item 长度列表
        if self.pp_size > 1:
            src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
                self.get_mha_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
            )

            # 先添加 K 缓存层参数，再添加 V 缓存层参数
            layers_params = [
                (
                    src_k_ptrs[layer_id],
                    dst_k_ptrs[layer_id],
                    self.kv_args.kv_item_lens[layer_id],
                )
                for layer_id in range(layers_current_pp_stage)
            ] + [
                (
                    src_v_ptrs[layer_id],
                    dst_v_ptrs[layer_id],
                    self.kv_args.kv_item_lens[layers_current_pp_stage + layer_id],
                )
                for layer_id in range(layers_current_pp_stage)
            ]
        else:
            # 单阶段：直接枚举所有层的源/目标指针
            num_layers = len(self.kv_args.kv_data_ptrs)
            layers_params = [
                (
                    self.kv_args.kv_data_ptrs[layer_id],
                    dst_kv_ptrs[layer_id],
                    self.kv_args.kv_item_lens[layer_id],
                )
                for layer_id in range(num_layers)
            ]

        def set_transfer_blocks(
            src_ptr: int, dst_ptr: int, item_len: int
        ) -> List[Tuple[int, int, int]]:
            # 为单层计算每个连续块的源地址、目标地址和传输长度
            transfer_blocks = []
            for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len
                length = item_len * len(prefill_index)
                transfer_blocks.append((src_addr, dst_addr, length))
            return transfer_blocks

        # Worker function for processing a single layer
        def process_layer(src_ptr: int, dst_ptr: int, item_len: int) -> int:
            # 处理单层 KV 数据传输，返回传输状态码
            transfer_blocks = set_transfer_blocks(src_ptr, dst_ptr, item_len)
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        # Worker function for processing all layers in a batch
        def process_layers(layers_params: List[Tuple[int, int, int]]) -> int:
            # 将所有层的传输块合并为一个批次，一次性提交给传输引擎（效率更高）
            transfer_blocks = []
            for src_ptr, dst_ptr, item_len in layers_params:
                transfer_blocks.extend(set_transfer_blocks(src_ptr, dst_ptr, item_len))
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        if self.enable_custom_mem_pool:
            # 启用自定义内存池时，使用线程池并行处理每层的 KV 传输
            futures = [
                executor.submit(
                    process_layer,
                    src_ptr,
                    dst_ptr,
                    item_len,
                )
                for (src_ptr, dst_ptr, item_len) in layers_params
            ]
            # 等待所有 future 完成，若有失败则取消其余任务
            for future in concurrent.futures.as_completed(futures):
                status = future.result()
                if status != 0:
                    for f in futures:
                        f.cancel()
                    return status
        else:
            # Combining all layers' params in one batch transfer is more efficient
            # compared to using multiple threads
            # 未使用自定义内存池时，批量合并所有层参数更高效
            return process_layers(layers_params)

        return 0


# 昇腾 KV 发送器：直接复用 Mooncake 实现，无需额外覆盖
class AscendKVSender(MooncakeKVSender):
    pass


# 昇腾 KV 接收器：直接复用 Mooncake 实现，无需额外覆盖
class AscendKVReceiver(MooncakeKVReceiver):
    pass


# 昇腾 KV 引导服务器：直接复用 Mooncake 实现，无需额外覆盖
class AscendKVBootstrapServer(MooncakeKVBootstrapServer):
    pass
