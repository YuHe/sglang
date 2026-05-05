# 华为昇腾（Ascend）KV 传输引擎模块：基于 memfabric_hybrid 实现 NPU 间 KV 缓存传输
import logging
import os
from typing import List

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
# 继承自 MooncakeTransferEngine，复用其接口规范
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    MooncakeTransferEngine,
)
from sglang.srt.utils.network import NetworkAddress

# 尝试导入昇腾 memfabric_hybrid 传输引擎库，失败时延迟报错
try:
    from memfabric_hybrid import TransferEngine

    import_error = None
except ImportError as e:
    import_error = e
    pass

logger = logging.getLogger(__name__)


# 昇腾传输引擎：封装 memfabric_hybrid.TransferEngine，支持 SDMA 和 Device RDMA 两种传输协议
class AscendTransferEngine(MooncakeTransferEngine):

    def __init__(
        self,
        hostname: str,
        npu_id: int,
        disaggregation_mode: DisaggregationMode,
    ):
        # 若导入失败则在此时抛出，提示用户安装 memfabric_hybrid
        if import_error is not None:
            logger.warning(
                "Please install memfabric_hybrid, for details, see docs/backend/pd_disaggregation.md"
            )
            raise import_error

        # 创建底层 TransferEngine 实例
        self.engine = TransferEngine()
        self.hostname = hostname
        self.npu_id = npu_id

        # Centralized storage address of the AscendTransferEngine
        # 从环境变量读取 memfabric 中心化存储地址（etcd/redis 等）
        self.store_url = os.getenv("ASCEND_MF_STORE_URL")
        # 根据 PD 分离模式设置本节点角色（Prefill 或 Decode）
        if disaggregation_mode == DisaggregationMode.PREFILL:
            self.role = "Prefill"
        elif disaggregation_mode == DisaggregationMode.DECODE:
            self.role = "Decode"
        else:
            logger.error(f"Unsupported DisaggregationMode: {disaggregation_mode}")
            raise ValueError(f"Unsupported DisaggregationMode: {disaggregation_mode}")
        # 生成 session_id：hostname:rpc_port 格式，用于引擎间握手
        self.session_id = NetworkAddress(
            self.hostname, self.engine.get_rpc_port()
        ).to_host_port_str()
        # 调用 initialize 完成传输引擎的实际初始化
        self.initialize()

    def initialize(self) -> None:
        # 初始化昇腾传输引擎：选择传输协议并完成引擎注册
        from sglang.srt.distributed.parallel_state import (
            get_world_group,
            get_world_size,
        )

        # 获取传输协议，默认 sdma，可选 device_rdma
        transfer_protocol = self._get_transfer_protocol()
        if transfer_protocol is None or transfer_protocol == "sdma":
            # SDMA：通过共享内存直接内存访问，适合节点内传输
            trans_op_type = TransferEngine.TransDataOpType.SDMA
        else:
            # Device RDMA：设备间 RDMA，适合跨节点高带宽传输
            trans_op_type = TransferEngine.TransDataOpType.DEVICE_RDMA
            """with device RDMA for PD transfer"""
            tmp_tensor = torch.zeros(1, device="npu")
            output_tensor_list = [
                torch.empty_like(tmp_tensor) for _ in range(get_world_size())
            ]
            # Initialize hccl in advance through all_gather to avoid conflicts with rdma initialization.
            # 提前通过 all_gather 初始化 HCCL，避免与 RDMA 初始化冲突
            torch.distributed.all_gather(
                output_tensor_list, tmp_tensor, group=get_world_group().device_group
            )
        """Initialize the ascend transfer instance."""
        # 调用底层 engine.initialize 完成传输实例初始化，失败则抛出异常
        ret_value = self.engine.initialize(
            self.store_url, self.session_id, self.role, self.npu_id, trans_op_type
        )
        if ret_value != 0:
            logger.error("Ascend Transfer Engine initialization failed.")
            raise RuntimeError("Ascend Transfer Engine initialization failed.")

    def batch_register(self, ptrs: List[int], lengths: List[int]):
        # 批量注册内存地址到传输引擎，使其可用于后续 KV 数据传输
        try:
            ret_value = self.engine.batch_register_memory(ptrs, lengths)
        except Exception:
            # Mark register as failed
            # 注册异常时标记为失败，不中断程序
            ret_value = -1
        if ret_value != 0:
            logger.debug(f"Ascend memory registration for ptr {ptrs} failed.")

    @staticmethod
    def _get_transfer_protocol():
        # 从环境变量 ASCEND_MF_TRANSFER_PROTOCOL 读取传输协议，仅接受 device_rdma 或 sdma
        protocol = os.getenv("ASCEND_MF_TRANSFER_PROTOCOL")
        allowed_protocols = {"device_rdma", "sdma"}
        if protocol and protocol.lower() in allowed_protocols:
            return protocol.lower()
        else:
            # 未指定或无效协议时使用默认协议（SDMA）
            logger.warning(
                "Invalid or no transfer protocol specified, using default protocol."
            )
            return None
