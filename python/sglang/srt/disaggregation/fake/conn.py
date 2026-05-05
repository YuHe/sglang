# fake 传输连接模块：用于 warmup 请求的虚假 KV 传输，不执行真实网络操作
import logging
from typing import List, Optional

import numpy as np
import numpy.typing as npt

# 导入 PD 分离基础抽象类和传输参数/状态定义
from sglang.srt.disaggregation.base.conn import (
    BaseKVManager,
    BaseKVReceiver,
    BaseKVSender,
    KVArgs,
    KVPoll,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


# For warmup reqs, we don't kv transfer, we use the fake manager, sender and receiver
# 用于 warmup 请求的虚假 KV 管理器，不执行真实的 bootstrap 注册
class FakeKVManager(BaseKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        # 调用父类初始化，传递 KV 参数、PD 分离模式和服务器参数
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)

    def register_to_bootstrap(self):
        # warmup 场景不需要向 bootstrap 服务器注册，直接空实现
        pass


# 用于 warmup 请求的虚假 KV 发送器，模拟发送流程但不传输真实数据
class FakeKVSender(BaseKVSender):
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        # 标记是否已完成虚假发送
        self.has_sent = False

    def poll(self) -> KVPoll:
        # 未发送时返回等待输入状态，模拟握手瞬间完成
        if self.has_sent is False:
            # Assume handshake completed instantly
            return KVPoll.WaitingForInput
        else:
            # Assume transfer completed instantly
            # 已发送则立即返回成功，模拟传输瞬间完成
            logger.debug("FakeKVSender poll success")
            return KVPoll.Success

    def init(
        self,
        kv_indices: list[int],
        aux_index: Optional[int] = None,
    ):
        # 记录初始化参数，不执行真实操作
        logger.debug(
            f"FakeKVSender init with kv_indices: {kv_indices}, aux_index: {aux_index}"
        )
        pass

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List[int]] = None,
    ):
        # 标记为已发送，记录日志，不执行真实 KV 数据传输
        self.has_sent = True
        logger.debug(
            f"FakeKVSender send with kv_indices: {kv_indices}, state_indices: {state_indices}"
        )

    def failure_exception(self):
        # 抛出虚假发送器异常（用于测试失败路径）
        raise Exception("Fake KVSender Exception")


# 用于 warmup 请求的虚假 KV 接收器，模拟接收流程但不实际等待数据
class FakeKVReceiver(BaseKVReceiver):
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        # bootstrap 是否完成的标志
        self.bootstrap_done = False
        # 是否已发送元数据的标志
        self.has_sent_metadata = False
        # 是否需要 staging 缓冲区（fake 模式不需要）
        self.require_staging: bool = False

    def poll(self) -> KVPoll:
        # 按照 bootstrap -> 等待元数据 -> 成功 的状态机顺序推进
        if not self.bootstrap_done:
            return KVPoll.Bootstrapping
        if not self.has_sent_metadata:
            return KVPoll.WaitingForInput
        logger.debug("FakeKVReceiver poll success")
        return KVPoll.Success

    def init(
        self,
        prefill_dp_rank: int,
    ):
        # 模拟 bootstrap 完成，标记为已完成
        self.bootstrap_done = True

    def send_metadata(
        self,
        kv_indices: list[int],
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
    ):
        # 标记元数据已发送，记录日志，不执行真实网络操作
        self.has_sent_metadata = True
        logger.debug(
            f"FakeKVReceiver send_metadata with kv_indices: {kv_indices}, aux_index: {aux_index}, state_indices: {state_indices}"
        )

    def failure_exception(self):
        # 抛出虚假接收器异常（用于测试失败路径）
        raise Exception("Fake KVReceiver Exception")
