"""
Mooncake 传输后端的 KV Cache 传输连接管理模块。
实现 PD 分离架构中基于 Mooncake 传输引擎（RDMA/NVLINK）的高性能 KV Cache 传输。

主要组件：
- MooncakeKVManager: Prefill/Decode 两侧的 KV 传输管理器，负责引擎初始化、缓冲区注册、
  ZMQ 握手线程和传输 worker 线程管理。
- MooncakeKVSender: Prefill 侧 KV 发送器，将 KV Cache 分块发送到 Decode 侧。
- MooncakeKVReceiver: Decode 侧 KV 接收器，管理与 Prefill 侧的握手和 KV 目标缓冲区注册。
- MooncakeKVBootstrapServer: 基于 CommonKVBootstrapServer 的 bootstrap 服务（直接继承）。
"""
from __future__ import annotations

import concurrent.futures
import ctypes
import dataclasses
import logging
import os
import struct
import threading
import time
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt

# KV 传输参数基类和传输状态枚举
from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
# 公共 KV 传输接口基类（Bootstrap 服务、管理器、发送器、接收器）
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
# FastQueue：无锁高性能队列；group_concurrent_contiguous：合并连续索引块
from sglang.srt.disaggregation.common.utils import (
    FastQueue,
    group_concurrent_contiguous,
)
# 检查 Mooncake 自定义内存池是否启用（NVLINK/INTRA_NODE_NVLINK 等）
from sglang.srt.disaggregation.mooncake.utils import (
    check_mooncake_custom_mem_pool_enabled,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    filter_kv_indices_for_cp_rank,  # CP（Context Parallelism）rank 过滤 KV 索引
)
# 获取全局 Mooncake 传输引擎单例
from sglang.srt.distributed.parallel_state import get_mooncake_transfer_engine
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.network import NetworkAddress

logger = logging.getLogger(__name__)


class KVTransferError(Exception):
    """KV 传输失败时抛出的异常，携带 bootstrap_room 和失败原因。"""
    def __init__(self, bootstrap_room: int, failure_reason: str):
        super().__init__(failure_reason)
        self.bootstrap_room = bootstrap_room
        self.failure_reason = failure_reason

    def __str__(self):
        return f"KVTransferError(bootstrap_room={self.bootstrap_room}): {self.failure_reason}"


# prefill
# Prefill 侧 KV 分块传输任务描述符
@dataclasses.dataclass
class TransferKVChunk:
    room: int                                     # bootstrap_room 唯一标识
    prefill_kv_indices: npt.NDArray[np.int32]    # 本 chunk 对应的 Prefill KV 索引
    index_slice: slice                            # 在完整 KV 序列中的切片范围
    is_last_chunk: bool                           # 是否为最后一个 chunk（触发 aux 元数据发送）
    prefill_aux_index: Optional[int]             # 元数据缓冲区槽位索引（仅最后 chunk 有效）
    state_indices: Optional[List[int]]           # Mamba/SWA/NSA 状态索引（仅最后 chunk 有效）


from sglang.srt.disaggregation.common.staging_handler import (
    DecodeStagingContext,
    PrefillStagingContext,
    StagingRegisterInfo,
    StagingTransferInfo,
)


# decode
# Decode 侧 KV 传输任务描述符：来自 Prefill 侧的 ZMQ 握手消息解析结果
@dataclasses.dataclass
class TransferInfo:
    room: int                                     # bootstrap_room 唯一标识
    endpoint: str                                 # Prefill 侧 IP 地址
    dst_port: int                                 # Prefill 侧接收状态的 ZMQ 端口
    mooncake_session_id: str                      # Mooncake session 唯一 ID（"ip:port"）
    dst_kv_indices: npt.NDArray[np.int32]        # Decode 侧 KV 缓冲区目标索引
    dst_aux_index: int                            # Decode 侧元数据缓冲区槽位索引
    dst_state_indices: List[int]                  # Decode 侧 Mamba/SWA/NSA 状态索引
    required_dst_info_num: int                    # 需要接收多少个 Prefill rank 的确认才算完成
    is_dummy: bool                                # True 表示该 Decode rank 不实际接收 KV（虚拟请求）
    # Note: always put the optional staging field at the final (it will be set through 'STAGING_RSP' pkg when needed)
    # staging 分配信息，由 STAGING_RSP 消息动态填充
    staging: Optional[StagingTransferInfo] = None

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        """从 ZMQ 多帧消息解析 TransferInfo（Prefill → Decode 握手消息格式）。"""
        if msg[4] == b"" and msg[5] == b"":
            is_dummy = True
            dst_kv_indices = np.array([], dtype=np.int32)
            dst_aux_index = None
            dst_state_indices = []
        else:
            dst_kv_indices = np.frombuffer(msg[4], dtype=np.int32)
            dst_aux_index = int(msg[5].decode("ascii"))
            if msg[6] == b"":
                dst_state_indices = []
            else:
                dst_state_indices = list(np.frombuffer(msg[6], dtype=np.int32))
            is_dummy = False
        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_kv_indices=dst_kv_indices,
            dst_aux_index=dst_aux_index,
            dst_state_indices=dst_state_indices,
            required_dst_info_num=int(msg[7].decode("ascii")),
            is_dummy=is_dummy,
        )


# decode
# Decode 侧 KV 参数注册信息：Decode 在握手时将自身缓冲区地址发送给 Prefill
@dataclasses.dataclass
class KVArgsRegisterInfo:
    room: str                                       # bootstrap_room（字符串格式）
    endpoint: str                                   # Decode 侧 IP
    dst_port: int                                   # Decode 侧 ZMQ 端口
    mooncake_session_id: str                        # Mooncake session ID
    dst_kv_ptrs: list[int]                         # Decode KV 缓冲区指针列表
    dst_aux_ptrs: list[int]                        # Decode 元数据缓冲区指针列表
    dst_state_data_ptrs: list[int]                 # Decode 状态缓冲区指针（Mamba/SWA/NSA）
    dst_tp_rank: int                               # Decode 侧 TP rank
    dst_attn_tp_size: int                          # Decode 侧 attention TP size
    dst_kv_item_len: int                           # Decode 侧每 token KV 数据字节数
    # for mamba state different tp slice transfer
    # Mamba 状态跨 TP 切片传输所需的维度信息
    dst_state_item_lens: list[int]                 # 每层 Mamba 状态 item 长度
    dst_state_dim_per_tensor: list[int]            # 每层 Mamba 状态可切片的维度大小
    # HiSparse: decode host pool stores KV at token granularity
    # HiSparse 模式：Decode host pool 按 token 粒度存储 KV，page_size=1
    enable_hisparse: bool = False
    # Note: always put the staging field at the final (since the staging field is optional and contains multiple inputs)
    # staging 注册信息（异构 TP staging buffer 的基址和总大小）
    staging: Optional[StagingRegisterInfo] = None

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        """从 ZMQ 多帧消息解析 KVArgsRegisterInfo（Decode → Prefill 注册消息格式）。"""
        return cls(
            room=str(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            # 指针数组：每个指针 8 字节（uint64），用 struct.unpack 解包
            dst_kv_ptrs=list(struct.unpack(f"{len(msg[4])//8}Q", msg[4])),
            dst_aux_ptrs=list(struct.unpack(f"{len(msg[5])//8}Q", msg[5])),
            dst_state_data_ptrs=list(struct.unpack(f"{len(msg[6])//8}Q", msg[6])),
            dst_tp_rank=int(msg[7].decode("ascii")),
            dst_attn_tp_size=int(msg[8].decode("ascii")),
            dst_kv_item_len=int(msg[9].decode("ascii")),
            dst_state_item_lens=(
                list(struct.unpack(f"{len(msg[10])//4}I", msg[10]))
                if len(msg) > 10 and len(msg[10]) > 0
                else []
            ),
            dst_state_dim_per_tensor=(
                list(struct.unpack(f"{len(msg[11])//4}I", msg[11]))
                if len(msg) > 11 and len(msg[11]) > 0
                else []
            ),
            enable_hisparse=(
                msg[12].decode("ascii") == "1" if len(msg) > 12 else False
            ),
            # Note: always put the staging field at the final
            staging=StagingRegisterInfo.from_zmq_fields(msg, 13),
        )


class AuxDataCodec:
    """辅助数据编解码器：处理元数据缓冲区（首 token、cached_tokens 等）的序列化与反序列化。"""

    @staticmethod
    def serialize_data_from_buffer(src_addr, data_length):
        """将内存缓冲区中的数据序列化为 bytes（用于 TCP 传输）。"""
        buffer = (ctypes.c_byte * data_length).from_address(src_addr)
        return bytes(buffer)

    @staticmethod
    def deserialize_data_to_buffer(kv_args, buffer_index, aux_index, data):
        """将接收到的 bytes 反序列化并写入目标内存缓冲区（Decode 侧元数据槽）。"""
        dst_aux_ptr = kv_args.aux_data_ptrs[buffer_index]
        item_len = kv_args.aux_item_lens[buffer_index]
        dst_addr = dst_aux_ptr + item_len * aux_index
        buffer = (ctypes.c_byte * len(data)).from_address(dst_addr)
        buffer[:] = data
        return


class MooncakeKVManager(CommonKVManager):
    """Mooncake KV 传输管理器：PD 分离架构中 Prefill 和 Decode 两侧的核心管理类。

    负责初始化 Mooncake 传输引擎、注册内存缓冲区、启动工作线程、
    管理传输任务队列与状态，以及处理 Staging Buffer（异构 TP 场景）。
    """

    # 辅助数据消息头标识符（用于 TCP 传输路径）
    AUX_DATA_HEADER = b"AUX_DATA"

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        # 调用父类初始化，完成公共字段（bootstrap socket、KVArgs 等）设置
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        # 初始化 Mooncake 传输引擎实例
        self.init_engine()
        # 将 KV/Aux/State 缓冲区注册到传输引擎
        self.register_buffer_to_engine()
        # 是否启用 Staging Buffer（用于异构 TP 的 KV 搬运）
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # Prefill 侧：启动 bootstrap 监听线程（接收 Decode 的预分配通知）
            self.start_prefill_thread()
            # 会话失败计数，用于检测死亡的 Decode 会话
            self.session_failures = defaultdict(int)
            # 已判定死亡的会话 ID 集合
            self.failed_sessions = set()
            # 保护 session_failures/failed_sessions 的互斥锁
            self.session_lock = threading.Lock()
            # Determine the number of threads to use for kv sender
            cpu_count = os.cpu_count()
            transfer_thread_pool_size = (
                envs.SGLANG_DISAGGREGATION_THREAD_POOL_SIZE.get()
            )
            # 未配置时按 CPU 数量自动推算，范围 [4, 12]
            if transfer_thread_pool_size is None:
                transfer_thread_pool_size = min(max(4, int(0.5 * cpu_count) // 8), 12)
            # 无锁传输任务队列数量（按目标 session 分片，提升并发度）
            transfer_queue_size = envs.SGLANG_DISAGGREGATION_QUEUE_SIZE.get()
            self.transfer_queues: List[FastQueue] = [
                FastQueue() for _ in range(transfer_queue_size)
            ]
            assert transfer_thread_pool_size >= transfer_queue_size, (
                f"The environment variable SGLANG_DISAGGREGATION_THREAD_POOL_SIZE={transfer_thread_pool_size} must be "
                f"greater than or equal to SGLANG_DISAGGREGATION_QUEUE_SIZE={transfer_queue_size}."
            )
            # 每个队列对应一个线程池，线程数平均分配
            self.executors = [
                concurrent.futures.ThreadPoolExecutor(
                    transfer_thread_pool_size // transfer_queue_size
                )
                for _ in range(transfer_queue_size)
            ]
            # 检查是否启用了 Mooncake 自定义内存池（如 NVLINK/INTRA_NODE）
            self.enable_custom_mem_pool, self.custom_mem_pool_type = (
                check_mooncake_custom_mem_pool_enabled()
            )
            # 创建 Prefill 侧 Staging 上下文（含 buffer 列表、水位线等）
            self._staging_ctx = PrefillStagingContext() if self.enable_staging else None
            if self.enable_staging:
                # 为每个传输队列初始化一个 Staging 缓冲区（注册到引擎）
                self._init_staging_buffers(len(self.transfer_queues))
            # 为每个（队列, 执行器）对启动一个后台传输工作线程
            for i, (queue, executor) in enumerate(
                zip(self.transfer_queues, self.executors)
            ):
                threading.Thread(
                    target=self.transfer_worker,
                    args=(
                        queue,
                        executor,
                        (
                            # 启用 Staging 时传入对应 buffer，否则传 None
                            self._staging_ctx.buffers[i]
                            if self.enable_staging and self._staging_ctx.buffers
                            else None
                        ),
                    ),
                    daemon=True,
                ).start()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            # Decode 侧：创建 Staging 上下文（含分配器、水位线订阅等）
            self._staging_ctx = DecodeStagingContext() if self.enable_staging else None
            if self.enable_staging:
                # 初始化 Decode 侧 Staging 分配器（ring buffer 管理）
                self._init_staging_allocator()
                # Staging 处理器，负责 scatter chunk 到真实 KV 槽
                self._staging_handler = None
                # 每个 room 各 chunk 的写入计数，用于等待所有 Prefill 写入完成
                self._chunk_writer_counts: dict = defaultdict(lambda: defaultdict(list))
            # 启动 Decode 侧 ZMQ 监听线程和心跳检测线程
            self.start_decode_thread()

    def init_engine(self):
        """初始化 Mooncake 传输引擎（RDMA/NVLINK 等底层通信后端）。"""
        self.engine = get_mooncake_transfer_engine()

    def register_buffer_to_engine(self):
        """将 KV/Aux/State 内存缓冲区批量注册到传输引擎，使其可作为 RDMA 源/目标。"""
        # Batch register KV data buffers
        if self.kv_args.kv_data_ptrs and self.kv_args.kv_data_lens:
            self.engine.batch_register(
                self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
            )

        # Batch register auxiliary data buffers
        if self.kv_args.aux_data_ptrs and self.kv_args.aux_data_lens:
            self.engine.batch_register(
                self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
            )

        # Batch register state/extra pool data buffers
        if self.kv_args.state_data_ptrs and self.kv_args.state_data_lens:
            self.engine.batch_register(
                self.kv_args.state_data_ptrs, self.kv_args.state_data_lens
            )

    # ------------------------------------------------------------------
    # Staging buffer methods (all delegate to staging_handler.py)
    # ------------------------------------------------------------------

    def register_staging_room_bootstrap(self, room, bootstrap_infos, receiver):
        """记录请求 room 对应的 bootstrap 信息和接收者，供后续 Staging 操作查询。"""
        self._staging_ctx.room_bootstrap[room] = bootstrap_infos
        self._staging_ctx.room_receivers[room] = receiver

    def set_kv_buffer_tensors(self, k_buffers: list, v_buffers: list, page_size: int):
        """保存 KV Buffer 张量引用（Prefill 侧的 K/V 层张量），用于 Staging 时聚合 head 数据。"""
        self.kv_buffer_tensors = {
            "k_buffers": k_buffers,
            "v_buffers": v_buffers,
            "page_size": page_size,
        }

    def _init_staging_buffers(self, count: int):
        """初始化 Prefill 侧 Staging 缓冲区（每个传输队列一个），并注册到引擎。"""
        from sglang.srt.disaggregation.common.staging_handler import (
            init_staging_buffers,
        )

        self._staging_ctx.buffers = init_staging_buffers(
            self.engine, self.kv_args, count
        )
        # kv_buffer_tensors 在后续 set_kv_buffer_tensors 调用时设置
        self.kv_buffer_tensors = None

    def _init_staging_allocator(self):
        """初始化 Decode 侧 Staging 分配器（ring buffer），用于为接收数据分配临时空间。"""
        from sglang.srt.disaggregation.common.staging_handler import (
            init_staging_allocator,
        )

        self._staging_ctx.allocator = init_staging_allocator(self.engine, self.kv_args)
        self.kv_buffer_tensors = None

    def _handle_staging_req(self, msg):
        """处理 Prefill 侧发来的 STAGING_REQ 消息，为其在 Decode 侧分配 Staging 空间并回复偏移。"""
        from sglang.srt.disaggregation.common.staging_handler import (
            handle_staging_req,
        )

        # 解析 room 和 session_id
        room = int(msg[1].decode("ascii"))
        session_id = msg[4].decode("ascii")
        handler = self._staging_handler
        assert (
            handler is not None
        ), "STAGING_REQ received before staging handler initialized"
        # 查找对应的 Decode 请求对象
        decode_req = handler._room_to_decode_req.get(room)
        if decode_req is None:
            logger.warning(
                "STAGING_REQ received for unregistered room=%s, skipping",
                room,
            )
            return
        # 获取 Prefill 侧 TP 尺寸（用于计算写入者数量）
        prefill_tp = decode_req.kv_receiver.prefill_info.attn_tp_size
        handle_staging_req(
            msg,
            self._staging_ctx.allocator,
            self.kv_args,
            self.attn_tp_size,
            prefill_tp,
            getattr(self, "kv_buffer_tensors", None),
            self._staging_ctx.room_receivers,
            self._staging_ctx.room_bootstrap,
        )

        # 若存在接收者，注册水位线订阅（用于接收 Decode 消耗进度反馈）
        receiver = self._staging_ctx.room_receivers.get(room)
        if receiver is not None:
            handler.register_wm_subscriber(receiver, session_id)

    def _is_watermark_ready(
        self, session_id: str, alloc_round: int, alloc_end: int
    ) -> bool:
        """检查 Decode 侧水位线是否已达到 alloc_end，即 Staging 空间已被消费足够多。"""
        from sglang.srt.disaggregation.common.staging_handler import (
            is_watermark_ready,
        )

        return is_watermark_ready(self._staging_ctx, session_id, alloc_round, alloc_end)

    def _try_create_staging_strategy(self, staging_buffer):
        """尝试创建 PrefillStagingStrategy 实例（需要 kv_buffer_tensors 已就绪）。"""
        if not self.enable_staging or self.kv_buffer_tensors is None:
            return None
        from sglang.srt.disaggregation.common.staging_handler import (
            PrefillStagingStrategy,
        )

        return PrefillStagingStrategy(self, staging_buffer)

    def _send_chunk_ready(self, req, chunk_idx, kv_chunk, prefill_unique_rank):
        """通知 Decode 侧一个非最后的 Staging chunk 的 RDMA 写入已完成。"""
        try:
            na = NetworkAddress(req.endpoint, req.dst_port)
            self._connect(
                na.to_tcp(),
                is_ipv6=na.is_ipv6,
            ).send_multipart(
                [
                    b"CHUNK_READY",
                    str(req.room).encode("ascii"),
                    str(chunk_idx).encode("ascii"),
                    # chunk 在全局 dst_kv_indices 中的起始 page 索引
                    str(kv_chunk.index_slice.start).encode("ascii"),
                    # 本 chunk 包含的 page 数量
                    str(len(kv_chunk.prefill_kv_indices)).encode("ascii"),
                    req.mooncake_session_id.encode("ascii"),
                    str(prefill_unique_rank).encode("ascii"),
                ]
            )
        except Exception:
            pass

    def _do_staging_transfer(
        self,
        staging_strategy,
        kv_chunk,
        req,
        target_info,
        chunked_dst_kv_indice,
        executor,
        queue,
        prefill_unique_rank,
    ):
        """执行一个 chunk 的 Staging 传输。返回 (ret, deferred)。

        流程：检查 Staging 空间是否就绪 → 执行 gather+RDMA →
        失败时 fallback 到 per-token slice 路径 → 非最后 chunk 发 CHUNK_READY。
        deferred=True 表示空间未就绪，调用者需将 chunk 重新入队并跳出。
        """
        _tp = self.attn_tp_rank
        # 检查 Staging 分配是否就绪（水位线是否充足、偏移是否已获取）
        ready, chunk_idx, c_offset, _, _ = staging_strategy.check_ready(
            req,
            kv_chunk.index_slice.start,
            len(kv_chunk.prefill_kv_indices),
        )
        if not ready:
            from sglang.srt.disaggregation.common.staging_buffer import StagingAllocator

            # 永久失败：chunk 超过了 ring buffer 总大小
            if c_offset == StagingAllocator.ALLOC_OVERSIZED:
                raise RuntimeError(
                    f"[Staging] Chunk staging allocation permanently failed: "
                    f"chunk exceeds ring buffer total size (room={kv_chunk.room}). "
                    f"Increase SGLANG_DISAGG_STAGING_POOL_SIZE_MB."
                )
            # 临时未就绪：重新入队，延迟处理
            queue.put(kv_chunk)
            return (-1, True)

        # 执行 Staging 传输（gather KV heads → 批量 RDMA 写入 Decode Staging 区）
        ret = staging_strategy.transfer(
            req.mooncake_session_id,
            kv_chunk.prefill_kv_indices,
            target_info.staging.base_ptr + c_offset,
            target_info.staging.total_size - c_offset,
            target_info,
        )
        if ret == -1:
            # Staging 传输失败，降级到 per-token slice 路径
            logger.warning(
                f"[Staging][tp{_tp}] Falling back to per-token slice path "
                f"(room={kv_chunk.room})"
            )
            ret = self.send_kvcache_slice(
                req.mooncake_session_id,
                kv_chunk.prefill_kv_indices,
                target_info.dst_kv_ptrs,
                chunked_dst_kv_indice,
                target_info.dst_tp_rank,
                target_info.dst_attn_tp_size,
                target_info.dst_kv_item_len,
                executor,
            )
        elif ret == 0 and not kv_chunk.is_last_chunk:
            # 非最后 chunk 传输成功，通知 Decode 侧可以开始 scatter
            self._send_chunk_ready(req, chunk_idx, kv_chunk, prefill_unique_rank)
        return (ret, False)

    def _prefetch_staging_reqs(self, room: int):
        """在 Prefill forward 前预请求 Staging 分配（提前通知 Decode 侧准备空间），减少等待延迟。"""
        if not self.enable_staging or self.kv_buffer_tensors is None:
            return

        # 检查此 room 是否有需要 Staging 的目标（Prefill TP ≠ Decode TP 才需要 Staging）
        room_infos = self.transfer_infos.get(room, {})
        needs_staging = any(
            not tinfo.is_dummy
            and self.decode_kv_args_table.get(tinfo.mooncake_session_id) is not None
            and self.decode_kv_args_table[tinfo.mooncake_session_id].dst_attn_tp_size
            != self.attn_tp_size
            for tinfo in room_infos.values()
        )
        if not needs_staging:
            return

        from sglang.srt.disaggregation.common.staging_handler import (
            prefetch_staging_reqs,
        )

        # 向 Decode 侧发送 STAGING_REQ 消息，预请求各 chunk 的 ring buffer 偏移
        prefetch_staging_reqs(
            room,
            self.transfer_infos,
            self.kv_buffer_tensors,
            self.server_args.chunked_prefill_size,
            self._staging_ctx.prefetch_requested,
            self._staging_ctx.prefetch_sockets,
        )

    def send_kvcache_staged(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_staging_ptr: int,
        dst_staging_size: int,
        dst_tp_rank: int,
        dst_attn_tp_size: int,
        dst_kv_item_len: int,
        staging_buffer=None,
    ) -> int:
        """通过 Staging Buffer 传输 KV Cache（异构 TP 路径）。

        流程：gather KV heads 到 Prefill Staging Buffer → 批量 RDMA 写入 Decode Staging 区。
        当 Prefill TP 与 Decode TP 不同时，需先在 Prefill 侧聚合 head 分片，
        再通过一次大块 RDMA 传输给 Decode 侧（比逐 token slice 更高效）。
        """
        from sglang.srt.disaggregation.common.staging_buffer import (
            compute_head_slice_params,
            compute_staging_layout,
            resolve_total_kv_heads,
        )

        # 需要 kv_buffer_tensors（K/V 层张量）和 staging_buffer 才能执行 Staging 路径
        if self.kv_buffer_tensors is None or staging_buffer is None:
            return -1

        k_buffers = self.kv_buffer_tensors["k_buffers"]
        v_buffers = self.kv_buffer_tensors["v_buffers"]
        page_size = self.kv_buffer_tensors["page_size"]
        num_layers = len(k_buffers)
        head_dim = k_buffers[0].shape[-1]
        dtype_size = k_buffers[0].element_size()

        # 计算总 KV head 数（跨所有 TP rank）
        total_kv_heads = resolve_total_kv_heads(self.kv_args, self.attn_tp_size)

        # 计算本 Prefill rank 在 group 内的编号
        local_tp_rank = self.kv_args.engine_rank % self.attn_tp_size
        # 计算本 rank 需要发送的 head 起始位置和数量
        src_head_start, num_heads_to_send, _, _ = compute_head_slice_params(
            self.attn_tp_size,
            dst_attn_tp_size,
            local_tp_rank,
            dst_tp_rank,
            total_kv_heads,
        )

        # 计算每 rank 需要传输的字节数
        num_tokens = len(prefill_kv_indices) * page_size
        per_layer_bytes = num_tokens * num_heads_to_send * head_dim * dtype_size
        per_rank_bytes = per_layer_bytes * num_layers * 2

        # 计算 Decode 侧 Staging 区域的布局（多 writer 时各自的写入范围）
        num_writers, writer_rank_bytes, total_staging_needed = compute_staging_layout(
            self.attn_tp_size,
            dst_attn_tp_size,
            dst_tp_rank,
            total_kv_heads,
            num_tokens,
            head_dim * dtype_size,
            num_layers,
        )
        # 计算本 rank 作为第几个 writer，以及在 Decode Staging 区的字节偏移
        writer_idx = local_tp_rank % num_writers if num_writers > 1 else 0
        rank_offset = sum(writer_rank_bytes[:writer_idx])

        # Prefill Staging Buffer 容量检查
        if not staging_buffer.fits(per_rank_bytes):
            logger.warning(
                f"Prefill staging too small for {per_rank_bytes} bytes, falling back"
            )
            return -1
        # Decode Staging 区容量检查
        if dst_staging_size < total_staging_needed:
            logger.warning(
                f"Decode staging too small: need {total_staging_needed} bytes "
                f"({num_writers if self.attn_tp_size > dst_attn_tp_size else 1} writers "
                f"x {per_rank_bytes} bytes/rank), have {dst_staging_size}, falling back"
            )
            return -1

        from sglang.srt.disaggregation.common.staging_buffer import (
            gather_all_layers_to_staging,
        )

        # 将所有层的 KV head 数据 gather 到 Prefill 侧 Staging Buffer（GPU → host）
        gather_all_layers_to_staging(
            k_buffers,
            v_buffers,
            prefill_kv_indices,
            staging_buffer,
            src_head_start,
            num_heads_to_send,
            page_size,
            self.kv_args.gpu_id,
        )

        # 计算在 Decode Staging 区的写入地址（加上本 rank 的偏移）
        dst_write_ptr = dst_staging_ptr + rank_offset
        # 一次性 RDMA 批量传输（Prefill Staging → Decode Staging）
        ret = self._transfer_data(
            mooncake_session_id,
            [(staging_buffer.get_ptr(), dst_write_ptr, per_rank_bytes)],
        )
        if ret != 0:
            raise RuntimeError(
                f"[Staging] Bulk RDMA transfer failed with ret={ret}. "
                f"src_ptr=0x{staging_buffer.get_ptr():x}, "
                f"dst_ptr=0x{dst_write_ptr:x}, size={per_rank_bytes}. "
                f"The decode staging buffer may not be properly registered."
            )
        return ret

    def _transfer_data(self, mooncake_session_id, transfer_blocks):
        """调用 Mooncake 引擎执行批量同步 RDMA 传输。"""
        if not transfer_blocks:
            return 0

        # 将 (src, dst, length) 三元组拆解成三个列表
        src_addrs, dst_addrs, lengths = zip(*transfer_blocks)
        return self.engine.batch_transfer_sync(
            mooncake_session_id, list(src_addrs), list(dst_addrs), list(lengths)
        )

    def _send_kvcache_generic(
        self,
        mooncake_session_id: str,
        src_data_ptrs: list[int],
        dst_data_ptrs: list[int],
        item_lens: list[int],
        prefill_data_indices: npt.NDArray[np.int32],
        dst_data_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
    ) -> int:
        """
        通用 KV Cache 传输方法，支持 MHA 和 MLA 两种架构，被 send_kvcache 和 maybe_send_extra 共用。

        流程：先对连续 page 分组优化传输块，再按 PP 分段选择当前 PP stage 的层，
        最后通过 Mooncake 引擎执行批量或并行 RDMA 传输。
        """
        # Group by indices for optimization
        # 将连续的 page 索引合并为大块，减少 RDMA 调用次数
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_data_indices, dst_data_indices
        )

        layers_params = None

        # Decode pp size should be equal to prefill pp size or 1
        if self.is_mla_backend:
            # MLA 架构：K 和 V 合并存储，只有一组指针
            src_kv_ptrs, dst_kv_ptrs, layers_current_pp_stage = (
                self.get_mla_kv_ptrs_with_pp(src_data_ptrs, dst_data_ptrs)
            )
            layers_params = [
                (
                    src_kv_ptrs[layer_id],
                    dst_kv_ptrs[layer_id],
                    item_lens[layer_id],
                )
                for layer_id in range(layers_current_pp_stage)
            ]
        else:
            # MHA 架构：K 和 V 分别存储，各有一组指针
            src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
                self.get_mha_kv_ptrs_with_pp(src_data_ptrs, dst_data_ptrs)
            )
            # item_lens structure: [k_layer0, k_layer1, ..., k_layerN, v_layer0, v_layer1, ..., v_layerN]
            # Use correct item lengths for K and V separately
            if layers_current_pp_stage > len(dst_k_ptrs):
                logger.error(
                    "Prefill transfer kvcache error, layers_current_pp_stage is out of range: "
                    f"layers_current_pp_stage={layers_current_pp_stage}, len(dst_k_ptrs)={len(dst_k_ptrs)}"
                )
                return -1
            # 先生成所有 K 层参数，再生成所有 V 层参数
            layers_params = [
                (
                    src_k_ptrs[layer_id],
                    dst_k_ptrs[layer_id],
                    item_lens[layer_id],  # K item length
                )
                for layer_id in range(layers_current_pp_stage)
            ] + [
                (
                    src_v_ptrs[layer_id],
                    dst_v_ptrs[layer_id],
                    item_lens[layers_current_pp_stage + layer_id],  # V item length
                )
                for layer_id in range(layers_current_pp_stage)
            ]
        assert layers_params is not None

        def set_transfer_blocks(
            src_ptr: int, dst_ptr: int, item_len: int
        ) -> List[Tuple[int, int, int]]:
            # 为单层生成 RDMA 传输块列表（连续 page 合并后的地址 + 长度）
            transfer_blocks = []
            for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len
                length = item_len * len(prefill_index)
                transfer_blocks.append((src_addr, dst_addr, length))
            return transfer_blocks

        # Worker function for processing a single layer
        def process_layer(src_ptr: int, dst_ptr: int, item_len: int) -> int:
            transfer_blocks = set_transfer_blocks(src_ptr, dst_ptr, item_len)
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        # Worker function for processing all layers in a batch
        def process_layers(layers_params: List[Tuple[int, int, int]]) -> int:
            # 将所有层的传输块合并为一次 batch_transfer_sync 调用（更高效）
            transfer_blocks = []
            for src_ptr, dst_ptr, item_len in layers_params:
                transfer_blocks.extend(set_transfer_blocks(src_ptr, dst_ptr, item_len))
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        if self.enable_custom_mem_pool:
            # 自定义内存池模式（如 NVLINK）：每层并行提交，逐个检查结果
            futures = [
                executor.submit(
                    process_layer,
                    src_ptr,
                    dst_ptr,
                    item_len,
                )
                for (src_ptr, dst_ptr, item_len) in layers_params
            ]
            for future in concurrent.futures.as_completed(futures):
                status = future.result()
                if status != 0:
                    for f in futures:
                        f.cancel()
                    return status
            return 0
        else:
            # Combining all layers' params in one batch transfer is more efficient
            # compared to using multiple threads
            # 默认路径：所有层合并为单次 batch 传输（减少 RDMA 调用开销）
            return process_layers(layers_params)

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        """标准 KV Cache 传输（同构 TP 路径）：使用 Prefill 和 Decode 的 KV 池直接传输。"""
        return self._send_kvcache_generic(
            mooncake_session_id=mooncake_session_id,
            src_data_ptrs=self.kv_args.kv_data_ptrs,
            dst_data_ptrs=dst_kv_ptrs,
            item_lens=self.kv_args.kv_item_lens,
            prefill_data_indices=prefill_kv_indices,
            dst_data_indices=dst_kv_indices,
            executor=executor,
        )

    def send_kvcache_hisparse(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        page_index_slice: slice,
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        """HiSparse 传输路径：Prefill page_size > Decode host page_size=1。

        将 Prefill 的 page 级索引展开为 token 级，与 Decode 的 token 级索引对齐后传输。
        用于 Direct-to-Host（RDMA 写入 host 内存池）场景。
        """
        page_size = self.kv_args.page_size
        # 将每个 page 的 item_len 除以 page_size 得到 token 级 item_len
        per_token_item_lens = [il // page_size for il in self.kv_args.kv_item_lens]

        # Expand page-level src indices to token-level
        # 将 page 索引 [p0, p1, ...] 展开为 token 索引 [p0*ps, p0*ps+1, ..., p1*ps, ...]
        base = np.repeat(prefill_kv_indices * page_size, page_size)
        offsets = np.tile(np.arange(page_size, dtype=np.int32), len(prefill_kv_indices))
        expanded_src = base + offsets

        # Expand page-level index_slice to token-level for dst
        # 根据 page_index_slice 计算对应的 token 范围
        token_start = page_index_slice.start * page_size
        token_end = min(page_index_slice.stop * page_size, len(dst_kv_indices))
        expanded_dst = dst_kv_indices[token_start:token_end]

        # Clip src to match dst length (last page may be partial)
        # 最后一页可能不完整，裁剪 src 使长度与 dst 对齐
        expanded_src = expanded_src[: len(expanded_dst)]

        logger.debug(
            f"Send KVCache for hisparse: {expanded_src.shape} -> {expanded_dst.shape}"
        )
        return self._send_kvcache_generic(
            mooncake_session_id=mooncake_session_id,
            src_data_ptrs=self.kv_args.kv_data_ptrs,
            dst_data_ptrs=dst_kv_ptrs,
            item_lens=per_token_item_lens,
            prefill_data_indices=expanded_src,
            dst_data_indices=expanded_dst,
            executor=executor,
        )

    def send_kvcache_slice(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        dst_tp_rank: int,
        dst_attn_tp_size: int,
        dst_kv_item_len: int,
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        """
        异构 TP 的 KV 切片传输（M-to-N TP 尺寸配置）：逐 token 内的 head 切片传输。

        当 Prefill TP ≠ Decode TP 且未使用 Staging Buffer 时采用此路径，
        对每个 token slot 内的 head 维度进行精确切片后发送。
        NOTE: 每个 token slot 都调用一次传输引擎，对长序列有性能开销（TTFT 增加）。
        """
        # Extract configuration
        # 本 rank 在 TP group 内的编号
        local_tp_rank_in_group = self.kv_args.engine_rank % self.attn_tp_size
        src_kv_item_len = self.kv_args.kv_item_lens[0]
        dst_tp_rank_in_group = dst_tp_rank % dst_attn_tp_size
        page_size = self.kv_args.page_size

        # Use total KV head count (not per-rank) for correct head distribution.
        # Per-rank kv_head_num is max(1, total//tp) which loses info when total < tp.
        # 使用全局 KV head 数（非 per-rank），避免 GQA 下信息丢失
        total_kv_heads = getattr(self.kv_args, "total_kv_head_num", 0)
        if total_kv_heads <= 0:
            total_kv_heads = self.kv_args.kv_head_num * self.attn_tp_size

        src_heads_per_rank = max(1, total_kv_heads // self.attn_tp_size)
        dst_heads_per_rank = max(1, total_kv_heads // dst_attn_tp_size)
        bytes_per_head_slice_to_send = (
            dst_kv_item_len // page_size // dst_heads_per_rank
        )

        # GQA replication: how many prefill ranks share the same KV head
        # GQA 复制因子：多少个 Prefill rank 持有相同的 KV head
        src_replication = max(1, self.attn_tp_size // total_kv_heads)

        # Determine slicing parameters based on TP configuration
        if self.attn_tp_size > dst_attn_tp_size:
            # Send KVCache from multiple prefill instances to 1 decode instance
            # 多 Prefill → 1 Decode：每个 Prefill rank 发送自己持有的 head 到 Decode 的对应偏移
            src_head_start_offset = 0
            num_heads_to_send = src_heads_per_rank
            unique_head_idx = local_tp_rank_in_group // src_replication
            dst_head_start_offset = (
                unique_head_idx * src_heads_per_rank
            ) % dst_heads_per_rank
        else:
            # Send KVCache from 1 prefill instance to multiple decode instances
            # 1 Prefill → 多 Decode：Prefill rank 对每个 Decode rank 发送不同的 head 切片
            src_head_start_offset = (
                dst_tp_rank_in_group * dst_heads_per_rank
            ) % src_heads_per_rank
            num_heads_to_send = dst_heads_per_rank
            dst_head_start_offset = 0

        src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
            self.get_mha_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
        )

        # Calculate precise byte offset and length for the sub-slice within the token
        # 计算每个 token slot 内 head 切片的字节偏移和长度
        src_head_slice_offset = src_head_start_offset * bytes_per_head_slice_to_send
        dst_head_slice_offset = dst_head_start_offset * bytes_per_head_slice_to_send
        heads_bytes_per_token_to_send = num_heads_to_send * bytes_per_head_slice_to_send

        # Sanity check: The data sub-slice to be sent should fit into the dst buffer.
        # This means heads_bytes_per_token_to_send <= (dst_kv_item_len // page_size)
        if heads_bytes_per_token_to_send > (dst_kv_item_len // page_size):
            logger.error(
                f"[{mooncake_session_id}] slice size ({heads_bytes_per_token_to_send}) exceeds "
                f"target token slot size ({dst_kv_item_len // page_size})"
            )
            return -1

        # 将 page 索引和 token 内偏移广播展开，得到所有 token slot 的地址
        prefill_page_indices = prefill_kv_indices.reshape(-1, 1).astype(np.int64)
        decode_page_indices = dst_kv_indices.reshape(-1, 1).astype(np.int64)
        tokens_per_page = np.arange(page_size, dtype=np.int64).reshape(1, -1)
        bytes_per_token_on_prefill = src_kv_item_len // page_size
        bytes_per_token_on_decode = dst_kv_item_len // page_size
        # src token slot 内 head 切片的字节偏移（相对于 token 起始）
        src_token_slot_offsets = (
            tokens_per_page * bytes_per_token_on_prefill + src_head_slice_offset
        )
        # dst token slot 内 head 切片的字节偏移
        dst_token_slot_offsets = (
            tokens_per_page * bytes_per_token_on_decode + dst_head_slice_offset
        )

        def process_layer_tp_aware(src_layer_ptr, dst_layer_ptr):
            # 对单层生成所有 token 的 (src_addr, dst_addr) 列表并批量传输
            src_page_base_addrs = src_layer_ptr + prefill_page_indices * src_kv_item_len
            dst_page_base_addrs = dst_layer_ptr + decode_page_indices * dst_kv_item_len
            src_slice_addrs = src_page_base_addrs + src_token_slot_offsets
            dst_slice_addrs = dst_page_base_addrs + dst_token_slot_offsets

            src_addr_list = src_slice_addrs.reshape(-1).tolist()
            if not src_addr_list:
                # Nothing to transfer for this layer.
                return 0
            dst_addr_list = dst_slice_addrs.reshape(-1).tolist()
            total_slices = len(src_addr_list)
            length_list = [heads_bytes_per_token_to_send] * total_slices
            return self.engine.batch_transfer_sync(
                mooncake_session_id, src_addr_list, dst_addr_list, length_list
            )

        # 并发处理所有 K 和 V 层
        futures = []
        for i in range(layers_current_pp_stage):
            futures.append(
                executor.submit(process_layer_tp_aware, src_k_ptrs[i], dst_k_ptrs[i])
            )
        for i in range(layers_current_pp_stage):
            futures.append(
                executor.submit(process_layer_tp_aware, src_v_ptrs[i], dst_v_ptrs[i])
            )

        for future in concurrent.futures.as_completed(futures):
            status = future.result()
            if status != 0:
                for f in futures:
                    f.cancel()
                return status

        return 0

    def send_aux(
        self,
        req: TransferInfo,
        prefill_aux_index: int,
        dst_aux_ptrs: list[int],
    ):
        """发送辅助元数据（首 token、cached_tokens 等）到 Decode 侧。

        默认使用 RDMA 传输；当使用 NVLINK 内存池或强制 TCP 模式时，退化为 TCP 传输。
        """
        # TODO(shangming): Fix me when nvlink_transport of Mooncake is bug-free
        if (
            self.enable_custom_mem_pool
            and self.custom_mem_pool_type in ("NVLINK", "INTRA_NODE_NVLINK")
        ) or envs.SGLANG_MOONCAKE_SEND_AUX_TCP.get():
            # NVLINK 模式或强制 TCP 标志：使用 TCP 回退路径
            return self.send_aux_tcp(req, prefill_aux_index, dst_aux_ptrs)

        transfer_blocks = []
        prefill_aux_ptrs = self.kv_args.aux_data_ptrs
        prefill_aux_item_lens = self.kv_args.aux_item_lens

        # 为每个辅助缓冲区构建传输块（Prefill aux 槽 → Decode aux 槽）
        for i, dst_aux_ptr in enumerate(dst_aux_ptrs):
            length = prefill_aux_item_lens[i]
            src_addr = prefill_aux_ptrs[i] + length * prefill_aux_index
            dst_addr = dst_aux_ptrs[i] + length * req.dst_aux_index
            transfer_blocks.append((src_addr, dst_addr, length))

        return self._transfer_data(req.mooncake_session_id, transfer_blocks)

    def send_aux_tcp(
        self,
        req: TransferInfo,
        prefill_aux_index: int,
        dst_aux_ptrs: list[int],
    ):
        """通过 TCP（ZMQ）发送辅助元数据（NVLINK 路径的回退实现）。"""
        prefill_aux_ptrs = self.kv_args.aux_data_ptrs
        prefill_aux_item_lens = self.kv_args.aux_item_lens

        # 逐个序列化并通过 ZMQ socket 发送每个辅助缓冲区的内容
        for i in range(len(prefill_aux_ptrs)):
            length = prefill_aux_item_lens[i]
            src_addr = prefill_aux_ptrs[i] + length * prefill_aux_index
            # 将内存地址对应的字节拷贝为 Python bytes
            data = AuxDataCodec.serialize_data_from_buffer(src_addr, length)

            self.send_aux_data_to_endpoint(
                remote=req.endpoint,
                dst_port=req.dst_port,
                room=req.room,
                buffer_index=i,
                aux_index=req.dst_aux_index,
                data=data,
            )

        return 0

    def send_aux_data_to_endpoint(
        self,
        remote: str,
        dst_port: int,
        room: int,
        buffer_index: int,
        aux_index: int,
        data: bytes,
    ):
        """将辅助数据通过 ZMQ TCP socket 发送到指定 Decode 端点。"""
        na = NetworkAddress(remote, dst_port)
        socket = self._connect(na.to_tcp(), is_ipv6=na.is_ipv6)

        # 消息格式：[AUX_DATA_HEADER, room, buffer_index, aux_index, 数据长度(4字节), 数据]
        socket.send_multipart(
            [
                MooncakeKVManager.AUX_DATA_HEADER,
                str(room).encode("ascii"),
                str(buffer_index).encode("ascii"),
                str(aux_index).encode("ascii"),
                struct.pack(">I", len(data)),
                data,
            ]
        )

    def _handle_aux_data(self, msg: List[bytes]):
        """处理 Decode 侧收到的 AUX_DATA 消息，将元数据写入对应内存槽。"""
        room = int(msg[1].decode("ascii"))
        buffer_index = int(msg[2].decode("ascii"))
        aux_index = int(msg[3].decode("ascii"))
        data_length = struct.unpack(">I", msg[4])[0]
        data = msg[5]

        # 校验数据长度一致性
        if len(data) != data_length:
            logger.error(f"AUX_DATA length mismatch for bootstrap_room {room}")
            return

        # 将序列化数据写入 Decode 侧对应的辅助内存缓冲区槽
        AuxDataCodec.deserialize_data_to_buffer(
            self.kv_args, buffer_index, aux_index, data
        )

        logger.debug(
            f"Received AUX_DATA for bootstrap_room {room} with length:{len(data)}"
        )

    def maybe_send_extra(
        self,
        req: TransferInfo,
        prefill_state_indices: list[int],
        dst_state_data_ptrs: list[int],
        executor: concurrent.futures.ThreadPoolExecutor,
        target_rank_registration_info: Optional[KVArgsRegisterInfo] = None,
    ):
        """根据 state_type 分发额外状态数据（Mamba SSM 状态 / SWA / NSA）的传输。"""
        state_type = getattr(self.kv_args, "state_type", "none")

        if state_type == "mamba":
            # Mamba 状态：检查是否需要 TP slice 传输（Prefill TP ≠ Decode TP）
            if (
                target_rank_registration_info is not None
                and self.attn_tp_size != target_rank_registration_info.dst_attn_tp_size
            ):
                # 异构 TP：按维度切片发送 Mamba 状态
                return self._send_mamba_state_slice(
                    req,
                    prefill_state_indices,
                    dst_state_data_ptrs,
                    target_rank_registration_info.dst_state_item_lens,
                    target_rank_registration_info.dst_state_dim_per_tensor,
                    target_rank_registration_info.dst_tp_rank,
                    target_rank_registration_info.dst_attn_tp_size,
                )
            else:
                # 同构 TP：直接传输整个 Mamba 状态
                return self._send_mamba_state(
                    req,
                    prefill_state_indices,
                    dst_state_data_ptrs,
                )
        elif state_type in ["swa", "nsa"]:
            # SWA and NSA hybrid models do not support different TP sizes yet
            # SWA/NSA 混合模型（非 MLA）暂不支持异构 TP
            if (
                target_rank_registration_info is not None
                and not self.is_mla_backend
                and self.attn_tp_size != target_rank_registration_info.dst_attn_tp_size
            ):
                raise RuntimeError(
                    f"PD Disaggregation does NOT support PD different TP sizes for non-MLA {state_type.upper()} hybrid models yet."
                )
            dst_state_indices = req.dst_state_indices
            # 对齐 prefill 和 dst 状态索引长度（较长的截断至较短的）
            if len(prefill_state_indices) > len(dst_state_indices):
                logger.warning(
                    f"len(prefill_state_indices) = {len(prefill_state_indices)}, len(dst_state_indices) = {len(dst_state_indices)}"
                )
                prefill_state_indices = prefill_state_indices[: len(dst_state_indices)]
            elif len(prefill_state_indices) < len(dst_state_indices):
                logger.warning(
                    f"len(prefill_state_indices) = {len(prefill_state_indices)}, len(dst_state_indices) = {len(dst_state_indices)}"
                )
                dst_state_indices = dst_state_indices[: len(prefill_state_indices)]
            # Reuse _send_kvcache_generic interface to send extra pool data
            # 复用通用 KV 传输接口发送额外 state pool 数据
            prefill_state_indices = np.array(prefill_state_indices, dtype=np.int32)
            dst_state_indices = np.array(dst_state_indices, dtype=np.int32)
            return self._send_kvcache_generic(
                mooncake_session_id=req.mooncake_session_id,
                src_data_ptrs=self.kv_args.state_data_ptrs,
                dst_data_ptrs=dst_state_data_ptrs,
                item_lens=self.kv_args.state_item_lens,
                prefill_data_indices=prefill_state_indices,
                dst_data_indices=dst_state_indices,
                executor=executor,
            )
        else:
            # 无额外状态（标准 transformer），直接返回 0
            return 0

    def _send_mamba_state(
        self,
        req: TransferInfo,
        prefill_mamba_index: list[int],
        dst_state_data_ptrs: list[int],
    ):
        """传输 Mamba SSM 状态（conv_state + temporal_state），同构 TP 路径。"""
        assert len(prefill_mamba_index) == 1, "Mamba should have single state index"

        transfer_blocks = []
        prefill_state_data_ptrs = self.kv_args.state_data_ptrs
        prefill_state_item_lens = self.kv_args.state_item_lens

        # 为每个状态张量（conv_state 和 temporal_state 各一个）构建传输块
        for i, dst_state_ptr in enumerate(dst_state_data_ptrs):
            length = prefill_state_item_lens[i]
            src_addr = prefill_state_data_ptrs[i] + length * int(prefill_mamba_index[0])
            dst_addr = dst_state_ptr + length * int(req.dst_state_indices[0])
            transfer_blocks.append((src_addr, dst_addr, length))

        return self._transfer_data(req.mooncake_session_id, transfer_blocks)

    def _send_mamba_state_slice(
        self,
        req: TransferInfo,
        prefill_mamba_index: list[int],
        dst_state_data_ptrs: list[int],
        dst_state_item_lens: list[int],
        dst_state_dim_per_tensor: list[int],
        dst_tp_rank: int,
        dst_attn_tp_size: int,
    ):
        """异构 TP 下的 Mamba 状态切片传输。

        Mamba 状态布局：
        - conv_state: [num_layers, size+1, conv_dim/tp, conv_kernel-1]
        - temporal_state: [num_layers, size+1, num_heads/tp, head_dim, state_size]
        第 3 维（conv_dim 或 num_heads）按 TP 切片，异构 TP 时需按目标 rank 重新对齐。
        """
        logger.warning_once(
            "Using Mamba state slice transfer for different TP sizes between prefill and decode. "
            f"Prefill attn_tp_size={self.attn_tp_size}, Decode attn_tp_size={dst_attn_tp_size}. "
            "Performance may be affected."
        )
        assert len(prefill_mamba_index) == 1, "Mamba should have single state index"

        transfer_blocks = []
        prefill_state_data_ptrs = self.kv_args.state_data_ptrs
        prefill_state_item_lens = self.kv_args.state_item_lens
        src_state_dim_per_tensor = getattr(self.kv_args, "state_dim_per_tensor", [])

        # If no dimension info available, fall back to regular transfer
        # 无维度信息时回退到整体传输
        if not src_state_dim_per_tensor or not dst_state_dim_per_tensor:
            return self._send_mamba_state(req, prefill_mamba_index, dst_state_data_ptrs)

        local_tp_rank_in_group = self.kv_args.engine_rank % self.attn_tp_size
        dst_tp_rank_in_group = dst_tp_rank % dst_attn_tp_size

        for i, dst_state_ptr in enumerate(dst_state_data_ptrs):
            src_item_len = prefill_state_item_lens[i]
            dst_item_len = dst_state_item_lens[i]
            src_dim = src_state_dim_per_tensor[i]
            dst_dim = dst_state_dim_per_tensor[i]

            # Calculate bytes per dimension slice
            # item_len = dim * trailing_dims_size, so trailing_dims_size = item_len / dim
            # 每个维度切片的字节数 = item_len / dim
            src_bytes_per_dim = src_item_len // src_dim
            dst_bytes_per_dim = dst_item_len // dst_dim

            # Determine slicing parameters based on TP configuration
            if self.attn_tp_size > dst_attn_tp_size:
                # Multiple prefill ranks send to 1 decode rank
                # Each prefill sends all its dims to the appropriate offset in decode
                # 多 Prefill → 1 Decode：每个 Prefill 把自己全部 dim 写到 Decode 的对应偏移
                src_dim_start = 0
                num_dims_to_send = src_dim
                writers_per_decode = self.attn_tp_size // dst_attn_tp_size
                local_writer_idx = local_tp_rank_in_group % writers_per_decode
                dst_dim_start = local_writer_idx * src_dim
            else:
                # 1 prefill rank sends to multiple decode ranks
                # Prefill sends a slice of its dims to each decode rank
                # 1 Prefill → 多 Decode：对每个目标 Decode rank 发送不同的 dim 切片
                src_dim_start = (dst_tp_rank_in_group * dst_dim) % src_dim
                num_dims_to_send = dst_dim
                dst_dim_start = 0

            # Calculate byte offsets
            # 计算字节偏移
            src_dim_offset = src_dim_start * src_bytes_per_dim
            dst_dim_offset = dst_dim_start * dst_bytes_per_dim
            bytes_to_send = num_dims_to_send * src_bytes_per_dim

            # Calculate addresses for this state tensor
            # 计算 src 和 dst 的精确内存地址（state 索引 × item_len + dim 偏移）
            src_addr = (
                prefill_state_data_ptrs[i]
                + src_item_len * int(prefill_mamba_index[0])
                + src_dim_offset
            )
            dst_addr = (
                dst_state_ptr
                + dst_item_len * int(req.dst_state_indices[0])
                + dst_dim_offset
            )

            transfer_blocks.append((src_addr, dst_addr, bytes_to_send))

        return self._transfer_data(req.mooncake_session_id, transfer_blocks)

    def sync_status_to_decode_endpoint(
        self, remote: str, dst_port: int, room: int, status: int, prefill_rank: int
    ):
        """通过 ZMQ TCP socket 向 Decode 端点同步本 room 的传输状态（Success/Failed）。"""
        na = NetworkAddress(remote, dst_port)
        self._connect(na.to_tcp(), is_ipv6=na.is_ipv6).send_multipart(
            [
                str(room).encode("ascii"),
                str(status).encode("ascii"),
                # prefill_rank 用于 Decode 侧统计已收到多少个 Prefill rank 的响应
                str(prefill_rank).encode("ascii"),
            ]
        )

    def transfer_worker(
        self,
        queue: FastQueue,
        executor: concurrent.futures.ThreadPoolExecutor,
        staging_buffer=None,
    ):
        """KV 传输工作线程主循环：从 FastQueue 取出 TransferKVChunk，执行 RDMA 传输并同步状态。

        每个线程绑定一个 queue 和 executor，同一 room 的请求路由到同一线程以确保顺序性。
        支持三种传输路径：同构 TP（send_kvcache）/ HiSparse / Staging / 异构 TP 切片。
        """
        staging_strategy = None

        while True:
            try:
                # 从无锁队列中取出待传输的 KV chunk 任务
                kv_chunk: TransferKVChunk = queue.get()
                # 延迟初始化 Staging 策略（需等待 kv_buffer_tensors 就绪）
                if (
                    self.enable_staging
                    and staging_strategy is None
                    and staging_buffer is not None
                ):
                    staging_strategy = self._try_create_staging_strategy(staging_buffer)
                # 获取此 room 关联的所有 TransferInfo（每个 Decode TP rank 一个）
                reqs_to_be_processed = (
                    self.transfer_infos[kv_chunk.room].values()
                    if kv_chunk.room in self.transfer_infos
                    else []
                )
                polls = []
                dst_ranks_infos = []
                # Unique id per prefill sender so decode's response set size matches expected_response_num.
                # 计算本 Prefill rank 的唯一全局编号（含 PP/CP 维度），用于 Decode 侧计数
                prefill_unique_rank = (
                    self.attn_tp_rank * (self.pp_size * self.attn_cp_size)
                    + self.pp_rank * self.attn_cp_size
                    + self.attn_cp_rank
                )
                # When staging transfer is not yet ready (watermark/allocation pending),
                # the chunk is re-enqueued and we break out of the req loop to retry later.
                # Staging 暂缓标志：若 Staging 空间未就绪则置 True，跳出 req 循环重试
                staging_deferred = False
                for req in reqs_to_be_processed:
                    if not req.is_dummy:
                        # Early exit if the request has failed
                        # 若会话已标记死亡，立即报告失败并通知 Decode 侧
                        with self.session_lock:
                            if req.mooncake_session_id in self.failed_sessions:
                                self.record_failure(
                                    kv_chunk.room,
                                    f"Decode instance could be dead, remote mooncake session {req.mooncake_session_id} is not alive",
                                )
                                self.update_status(kv_chunk.room, KVPoll.Failed)
                                self.sync_status_to_decode_endpoint(
                                    req.endpoint,
                                    req.dst_port,
                                    req.room,
                                    KVPoll.Failed,
                                    prefill_unique_rank,
                                )
                                break

                        # 提取本 chunk 在 dst_kv_indices 中对应的切片
                        chunked_dst_kv_indice = req.dst_kv_indices[kv_chunk.index_slice]

                        # NOTE: This is temporarily a workaround to deal with the case where the prefill_kv_indices
                        # is mismatched with the dst_kv_indices when page size > 1, this should never happen.
                        # 安全截断：防止 page_size > 1 时 src 比 dst 多的边界情况
                        if len(chunked_dst_kv_indice) < len(
                            kv_chunk.prefill_kv_indices
                        ):
                            logger.warning(
                                f"len(chunked_dst_kv_indice) = {len(chunked_dst_kv_indice)}, len(kv_chunk.prefill_kv_indices) = {len(kv_chunk.prefill_kv_indices)}"
                            )
                            kv_chunk.prefill_kv_indices = kv_chunk.prefill_kv_indices[
                                : len(chunked_dst_kv_indice)
                            ]

                        # 获取目标 Decode rank 的 KVArgs 注册信息（dst 指针 / TP 尺寸等）
                        target_rank_registration_info: KVArgsRegisterInfo = (
                            self.decode_kv_args_table[req.mooncake_session_id]
                        )
                        if self.is_mla_backend or (
                            self.attn_tp_size
                            == target_rank_registration_info.dst_attn_tp_size
                        ):
                            # 同构 TP（或 MLA）路径：直接传输
                            if target_rank_registration_info.enable_hisparse:
                                # HiSparse 路径：Decode 使用 host 内存池，page_size=1
                                ret = self.send_kvcache_hisparse(
                                    req.mooncake_session_id,
                                    kv_chunk.prefill_kv_indices,
                                    target_rank_registration_info.dst_kv_ptrs,
                                    req.dst_kv_indices,
                                    kv_chunk.index_slice,
                                    executor,
                                )
                            else:
                                # 标准同构 TP 路径
                                ret = self.send_kvcache(
                                    req.mooncake_session_id,
                                    kv_chunk.prefill_kv_indices,
                                    target_rank_registration_info.dst_kv_ptrs,
                                    chunked_dst_kv_indice,
                                    executor,
                                )
                        elif (
                            self.enable_staging
                            and staging_strategy is not None
                            and target_rank_registration_info.staging is not None
                        ):
                            # 异构 TP + Staging Buffer 路径：gather→bulk RDMA→scatter
                            ret, deferred = self._do_staging_transfer(
                                staging_strategy,
                                kv_chunk,
                                req,
                                target_rank_registration_info,
                                chunked_dst_kv_indice,
                                executor,
                                queue,
                                prefill_unique_rank,
                            )
                            if deferred:
                                staging_deferred = True
                                # Chunk re-enqueued; stop processing remaining reqs for this chunk
                                break
                        else:
                            # 异构 TP fallback 路径：逐 token head 切片传输（开销较大）
                            ret = self.send_kvcache_slice(
                                req.mooncake_session_id,
                                kv_chunk.prefill_kv_indices,
                                target_rank_registration_info.dst_kv_ptrs,
                                chunked_dst_kv_indice,
                                target_rank_registration_info.dst_tp_rank,
                                target_rank_registration_info.dst_attn_tp_size,
                                target_rank_registration_info.dst_kv_item_len,
                                executor,
                            )
                        if ret != 0:
                            # 传输失败：记录会话失败、更新状态、通知 Decode
                            with self.session_lock:
                                self.session_failures[req.mooncake_session_id] += 1
                                # Failures should never happen if the session is not dead, if the session fails once, mark it as failed
                                if self.session_failures[req.mooncake_session_id] >= 1:
                                    self.failed_sessions.add(req.mooncake_session_id)
                                    logger.error(
                                        f"Session {req.mooncake_session_id} failed."
                                    )
                            self.record_failure(
                                kv_chunk.room,
                                f"Failed to send kv chunk of {kv_chunk.room} to "
                                f"{NetworkAddress(req.endpoint, req.dst_port).to_host_port_str()}",
                            )
                            self.update_status(kv_chunk.room, KVPoll.Failed)
                            self.sync_status_to_decode_endpoint(
                                req.endpoint,
                                req.dst_port,
                                req.room,
                                KVPoll.Failed,
                                prefill_unique_rank,
                            )
                            break

                        if kv_chunk.is_last_chunk:
                            # 最后一个 chunk：发送额外状态（Mamba/SWA/NSA）和辅助数据
                            if kv_chunk.state_indices is not None:
                                self.maybe_send_extra(
                                    req,
                                    kv_chunk.state_indices,
                                    target_rank_registration_info.dst_state_data_ptrs,
                                    executor,
                                    target_rank_registration_info,
                                )
                            # Only the last chunk we need to send the aux data
                            # 最后 chunk：发送辅助数据（首 token 等元信息）
                            ret = self.send_aux(
                                req,
                                kv_chunk.prefill_aux_index,
                                target_rank_registration_info.dst_aux_ptrs,
                            )
                            polls.append(True if ret == 0 else False)
                            dst_ranks_infos.append(
                                (req.endpoint, req.dst_port, req.room)
                            )

                            # Only sync status when all the dst ranks have received the kvcache
                            # 等所有 Decode TP rank 都收到 KV Cache 后再同步最终状态
                            if len(polls) == req.required_dst_info_num:
                                status = KVPoll.Success if all(polls) else KVPoll.Failed
                                self.update_status(req.room, status)
                                for endpoint, dst_port, room in dst_ranks_infos:
                                    self.sync_status_to_decode_endpoint(
                                        endpoint,
                                        dst_port,
                                        room,
                                        status,
                                        prefill_unique_rank,
                                    )
                    else:
                        # Dummy request means the decode instance is not used, so its status can be marked as success directly
                        # Dummy request does not need to sync status to decode endpoint
                        # Dummy 请求（此 Decode rank 不参与实际计算），最后 chunk 直接标记成功
                        if kv_chunk.is_last_chunk and req.room in self.request_status:
                            self.update_status(req.room, KVPoll.Success)

                # Staging 暂缓时跳回循环继续等待（chunk 已重新入队）
                if staging_deferred:
                    continue

                # 所有传输完成或 room 已不存在时，清理 transfer_infos 释放内存
                if (
                    kv_chunk.room not in self.request_status
                    or self.check_status(kv_chunk.room) == KVPoll.Success
                ):
                    if kv_chunk.room in self.transfer_infos:
                        self.transfer_infos.pop(kv_chunk.room)

            except Exception as e:
                # NOTE(shangming): Remove this when we make sure the transfer thread is bug-free
                # 传输线程异常会导致整个 Prefill 实例不可用，因此直接抛出 RuntimeError
                raise RuntimeError(
                    f"Transfer thread failed because of {e}. Prefill instance with bootstrap_port={self.bootstrap_port} is dead."
                )

    def start_prefill_thread(self):
        """启动 Prefill 侧 bootstrap 监听线程，接收 Decode 的预分配通知和 Staging 消息。"""
        def bootstrap_thread():
            """此线程接收 Decode 侧发来的多种消息：
            - WATERMARK: Decode 消费进度反馈（Staging 水位线更新）
            - STAGING_RSP: Decode 回复的 Staging 空间偏移
            - KVArgs 注册（room=None）: Decode 注册自身 KV 缓冲区地址
            - 正常 bootstrap 消息: Decode 请求 Prefill 开始传输 KV
            """
            # This thread recvs pre-alloc notification from the decode engine
            # KVPoll.Bootstrapping -> KVPoll.WaitingForInput
            while True:
                waiting_req_bytes = self.server_socket.recv_multipart()
                room = waiting_req_bytes[0].decode("ascii")
                # Staging: decode reports consumption watermark back to prefill
                # Staging 水位线更新：Decode 报告已消费到哪个位置
                if room == "WATERMARK":
                    wm_round = int(waiting_req_bytes[1].decode("ascii"))
                    wm_tail = int(waiting_req_bytes[2].decode("ascii"))
                    wm_session = (
                        waiting_req_bytes[3].decode("ascii")
                        if len(waiting_req_bytes) > 3
                        else ""
                    )
                    # 更新水位线并唤醒等待中的传输线程
                    with self._staging_ctx.watermark_cv:
                        prev = self._staging_ctx.remote_watermarks.get(
                            wm_session, (0, 0)
                        )
                        if (wm_round, wm_tail) > prev:
                            self._staging_ctx.remote_watermarks[wm_session] = (
                                wm_round,
                                wm_tail,
                            )
                            self._staging_ctx.watermark_cv.notify_all()
                    continue
                # Staging: decode replies with allocated staging offset
                # Staging 分配回复：Decode 回复本次 chunk 的 ring buffer 偏移
                if room == "STAGING_RSP":
                    stg_room = int(waiting_req_bytes[1].decode("ascii"))
                    stg_chunk_idx = int(waiting_req_bytes[2].decode("ascii"))
                    stg_offset = int(waiting_req_bytes[3].decode("ascii"))
                    stg_round = int(waiting_req_bytes[4].decode("ascii"))
                    stg_end = int(waiting_req_bytes[5].decode("ascii"))
                    stg_session = waiting_req_bytes[6].decode("ascii")
                    room_infos = self.transfer_infos.get(stg_room, {})
                    tinfo = room_infos.get(stg_session)
                    if tinfo is not None:
                        if tinfo.staging is None:
                            tinfo.staging = StagingTransferInfo()
                        # 记录 Staging 分配结果（偏移 + round 计数，用于水位线对齐）
                        tinfo.staging.set_chunk(
                            stg_chunk_idx, stg_offset, stg_round, stg_end
                        )
                    else:
                        logger.warning(
                            "STAGING_RSP RECV but tinfo=None room=%s chunk=%d session=%s",
                            stg_room,
                            stg_chunk_idx,
                            stg_session,
                        )
                    continue
                mooncake_session_id = waiting_req_bytes[3].decode("ascii")
                if room == "None":
                    # KVArgs 注册消息（room="None"）：Decode 注册自身 KV 地址和 TP 信息
                    self.decode_kv_args_table[mooncake_session_id] = (
                        KVArgsRegisterInfo.from_zmq(waiting_req_bytes)
                    )
                    # 清除该会话的历史失败记录（重连后重置）
                    with self.session_lock:
                        if mooncake_session_id in self.failed_sessions:
                            self.failed_sessions.remove(mooncake_session_id)
                        if mooncake_session_id in self.session_failures:
                            del self.session_failures[mooncake_session_id]
                    logger.debug(
                        f"Register KVArgs from {mooncake_session_id} successfully"
                    )
                    continue
                else:
                    # 正常 bootstrap 消息：Decode 告知 Prefill 本次传输的目标 KV 索引
                    required_dst_info_num = int(waiting_req_bytes[7].decode("ascii"))
                    room = int(room)
                    if room not in self.transfer_infos:
                        self.transfer_infos[room] = {}

                    # 将 TransferInfo（包含 dst 地址、KV 索引等）存入 transfer_infos 表
                    self.transfer_infos[room][mooncake_session_id] = (
                        TransferInfo.from_zmq(waiting_req_bytes)
                    )
                    # NOTE: after bootstrapping we can mark the req as waiting for input
                    # 所有目标 Decode rank 都完成 bootstrap 后，状态更新为 WaitingForInput
                    if len(self.transfer_infos[room]) == required_dst_info_num:
                        self.update_status(room, KVPoll.WaitingForInput)

        threading.Thread(target=bootstrap_thread).start()

    def start_decode_thread(self):
        """启动 Decode 侧两个后台线程：
        - decode_thread: 接收 Prefill 侧 ZMQ 消息（状态同步、AUX_DATA、Staging 通知）
        - heartbeat_checker: 定期对所有 Prefill 实例做 HTTP 心跳检测
        """
        def decode_thread():
            """接收并处理来自 Prefill 侧的各类 ZMQ 消息。

            消息类型：
            - AUX_DATA: 辅助元数据（TCP 路径）
            - CHUNK_READY: 非最后 Staging chunk 写入完成通知
            - STAGING_REQ: Prefill 预请求 Staging 分配
            - 普通状态消息: Success/Failed，更新 bootstrap_room 状态
            """
            while True:
                msg = self.server_socket.recv_multipart()
                # 辅助数据（TCP 传输路径）
                if msg[0] == MooncakeKVManager.AUX_DATA_HEADER:
                    self._handle_aux_data(msg)
                    continue

                # Staging: prefill notifies a chunk written to staging buffer
                # Staging chunk 写入完成通知
                if msg[0] == b"CHUNK_READY":
                    room = int(msg[1].decode("ascii"))
                    chunk_idx = int(msg[2].decode("ascii"))
                    page_start = int(msg[3].decode("ascii"))
                    num_pages = int(msg[4].decode("ascii"))
                    session_id = msg[5].decode("ascii")
                    # 记录已到达的 writer 信息（等待所有 Prefill rank 写完才 scatter）
                    self._chunk_writer_counts[room][chunk_idx].append(
                        (page_start, num_pages, session_id)
                    )
                    handler = self._staging_handler
                    assert (
                        handler is not None
                    ), "CHUNK_READY received before staging handler initialized"
                    writers_arrived = len(self._chunk_writer_counts[room][chunk_idx])
                    decode_req = handler._room_to_decode_req.get(room)
                    if decode_req is None:
                        logger.warning(
                            "CHUNK_READY received for unregistered room=%s chunk=%d, skipping",
                            room,
                            chunk_idx,
                        )
                        continue
                    num_writers = handler.num_writers_for(decode_req)
                    # 所有 Prefill 写入者到齐后，提交 scatter（Staging → 真实 KV 槽）
                    if writers_arrived >= num_writers:
                        handler.submit_chunk_scatter(
                            room, chunk_idx, page_start, num_pages
                        )
                        del self._chunk_writer_counts[room][chunk_idx]
                    continue

                # Staging: prefill pre-requests staging allocation before forward
                # Staging 预分配请求：Prefill forward 前先申请 Decode 侧 Staging 空间
                if msg[0] == b"STAGING_REQ":
                    self._handle_staging_req(msg)
                    continue

                # 普通状态消息：(bootstrap_room, status, prefill_rank)
                bootstrap_room, status, prefill_rank = msg
                status = int(status.decode("ascii"))
                bootstrap_room = int(bootstrap_room.decode("ascii"))
                prefill_rank = int(prefill_rank.decode("ascii"))

                if status == KVPoll.Success:
                    if bootstrap_room in self.request_status:
                        # 记录已响应的 Prefill rank
                        self.prefill_response_tracker[bootstrap_room].add(prefill_rank)
                        expected_response_num = (
                            self.required_prefill_response_num_table[bootstrap_room]
                        )
                        arrived_response_num = len(
                            self.prefill_response_tracker[bootstrap_room]
                        )
                        # 所有 Prefill rank 均报告 Success 后，更新状态为 Success
                        if arrived_response_num == expected_response_num:
                            if self.enable_staging:
                                handler = self._staging_handler
                                if handler.is_staging_room(bootstrap_room):
                                    # 提交最后一批 Staging scatter
                                    handler.submit_last_scatter_async(bootstrap_room)
                                self._chunk_writer_counts.pop(bootstrap_room, None)
                            self.update_status(bootstrap_room, KVPoll.Success)
                elif status == KVPoll.Failed:
                    # Prefill 侧报告失败
                    self.record_failure(
                        bootstrap_room,
                        "Failed to get kvcache from prefill instance, it might be dead",
                    )
                    self.update_status(bootstrap_room, status)

        def heartbeat_checker():
            """定期对已知 Prefill 实例发送 HTTP 健康检查，检测节点故障并触发失败处理。"""
            while True:
                time.sleep(self.heartbeat_interval)
                with self.connection_lock:
                    addresses = list(self.prefill_info_table.keys())

                for bootstrap_addr in addresses:
                    session = None
                    try:
                        with self.session_pool_lock:
                            session = self.session_pool[bootstrap_addr]
                        response = session.get(
                            f"http://{bootstrap_addr}/health",
                            timeout=(2, 3),
                            headers={"Connection": "keep-alive"},
                        )
                        if response.status_code == 200:
                            # 心跳成功：重置失败计数，清理已完成的 room
                            self.heartbeat_failures[bootstrap_addr] = 0

                            current_rooms = self.addr_to_rooms_tracker[
                                bootstrap_addr
                            ].copy()

                            for bootstrap_room in current_rooms:
                                # Remove KVPoll.Success requests from the tracker
                                if bootstrap_room not in self.request_status:
                                    self.addr_to_rooms_tracker[bootstrap_addr].discard(
                                        bootstrap_room
                                    )
                        else:
                            # 心跳失败：尝试重连
                            logger.info(
                                f"Attempting to reconnect to {bootstrap_addr}..."
                            )
                            self.heartbeat_failures[bootstrap_addr] = (
                                self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                            )
                            with self.session_pool_lock:
                                if bootstrap_addr in self.session_pool:
                                    del self.session_pool[bootstrap_addr]
                    except Exception:
                        logger.info(f"Attempting to reconnect to {bootstrap_addr}...")
                        self.heartbeat_failures[bootstrap_addr] = (
                            self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                        )

                    # 超过最大失败次数时，触发节点故障处理
                    if (
                        self.heartbeat_failures.get(bootstrap_addr, 0)
                        >= self.max_failures
                    ):
                        self._handle_node_failure(bootstrap_addr)
                        with self.session_pool_lock:
                            if bootstrap_addr in self.session_pool:
                                del self.session_pool[bootstrap_addr]

        threading.Thread(target=decode_thread).start()
        threading.Thread(target=heartbeat_checker).start()

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last_chunk: bool,
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
    ):
        """将 KV 传输任务（TransferKVChunk）放入对应的传输队列。

        按目标 session 端口哈希分片，确保同一目标会话的任务路由到同一队列（顺序保证）。
        """
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        # 最后一个 chunk 必须提供 aux_index（辅助数据槽位）
        assert not is_last_chunk or (is_last_chunk and aux_index is not None)

        # 请求已失败时直接跳过
        if (
            bootstrap_room not in self.request_status
            or self.check_status(bootstrap_room) == KVPoll.Failed
        ):
            logger.debug(
                "Request with bootstrap_room=%s already failed", bootstrap_room
            )
            return

        if bootstrap_room not in self.transfer_infos:
            # This means that the current rank is a dummy rank for this request,
            # and it has already been marked as success, so there is no need to
            # add further chunks into the transfer queue.
            # 当前 rank 为 dummy（不参与实际传输），已标记成功，无需入队
            return

        # NOTE(shangming): sharding according to the dst_infos to make sure
        # requests with the same dst_sessions will be added into the same
        # queue, which enables early abort with failed sessions.
        # 按目标 session 端口之和哈希，将相同目标的请求路由到同一队列
        dst_infos = self.transfer_infos[bootstrap_room].keys()
        session_port_sum = sum(int(session.rsplit(":", 1)[1]) for session in dst_infos)
        shard_idx = session_port_sum % len(self.transfer_queues)

        # 将 TransferKVChunk 放入对应的无锁队列
        self.transfer_queues[shard_idx].put(
            TransferKVChunk(
                room=bootstrap_room,
                prefill_kv_indices=kv_indices,
                index_slice=index_slice,
                is_last_chunk=is_last_chunk,
                prefill_aux_index=aux_index,
                state_indices=state_indices,
            )
        )

    def get_session_id(self):
        """获取 Mooncake 传输引擎为本进程分配的唯一会话 ID（含 IP:port 标识）。"""
        return self.engine.get_session_id()

    def _handle_node_failure(self, failed_bootstrap_addr):
        """处理 Prefill 节点故障：清理连接、将该节点关联的未完成请求标记为失败。"""
        with self.connection_lock:
            # 清理与故障节点相关的所有 ZMQ 连接
            keys_to_remove = [
                k for k in self.connection_pool if k.startswith(failed_bootstrap_addr)
            ]
            for k in keys_to_remove:
                del self.connection_pool[k]

            # 收集可能受影响的 bootstrap_room 列表
            possible_affected_rooms = self.addr_to_rooms_tracker.get(
                failed_bootstrap_addr, []
            )
            self.prefill_info_table.pop(failed_bootstrap_addr, None)
            self.addr_to_rooms_tracker.pop(failed_bootstrap_addr, None)

        # Report the requests associated with the failed bootstrap addr and mark their status as KVPoll.Failed
        # 对所有未完成的关联请求报告失败
        affected_rooms = []
        for room in possible_affected_rooms:
            if (
                room in self.request_status
                and self.check_status(room) != KVPoll.Success
            ):
                self.record_failure(
                    room,
                    f"Losing connection with prefill instance (bootstrap_addr: {failed_bootstrap_addr})",
                )
                self.update_status(room, KVPoll.Failed)
                affected_rooms.append(room)
        logger.error(
            f"Losing connection with prefill instance (bootstrap_addr: {failed_bootstrap_addr}), {len(affected_rooms)} requests affected"
        )


class MooncakeKVSender(CommonKVSender):
    """Prefill 侧 KV 发送器。

    持有 bootstrap_room 标识，通过 add_transfer_request 将 KV chunks 提交到传输队列。
    支持 CP（Context Parallelism）rank 过滤和 bootstrap 超时检测。
    """

    def __init__(
        self,
        mgr: MooncakeKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank)
        # 最终状态缓存（Success/Failed），避免重复查询 manager
        self.conclude_state = None
        # 记录初始化时间，用于 bootstrap 超时检测
        self.init_time = time.time()

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List[int]] = None,
    ):
        """发送一个 KV chunk（可能是整个 prompt 或 chunked prefill 的一部分）。"""
        # 更新已发送的 KV indices 全局偏移
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)
        is_last_chunk = self.curr_idx == self.num_kv_indices

        # Special handling for cp
        # CP 场景：过滤掉本 CP rank 不负责的 KV indices
        if self.kv_mgr.enable_all_cp_ranks_for_transfer:
            kv_indices, index_slice = filter_kv_indices_for_cp_rank(
                self.kv_mgr,
                kv_indices,
                index_slice,
            )
        elif self.kv_mgr.is_dummy_cp_rank:
            # dummy CP rank（不参与实际计算）：非最后 chunk 直接跳过
            if not is_last_chunk:
                return
            else:
                # 最后 chunk 时直接标记成功（dummy rank 无需实际传输）
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Success)
                return

        if not is_last_chunk:
            # 非最后 chunk：不附带 aux_index
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                False,
            )
        else:
            # 最后 chunk：附带 aux_index 和 state_indices（触发辅助数据发送）
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                True,
                aux_index=self.aux_index,
                state_indices=state_indices,
            )

    def poll(self) -> KVPoll:
        """查询当前传输状态，支持 bootstrap 超时自动失败。"""
        if self.conclude_state is None:
            status = self.kv_mgr.check_status(self.bootstrap_room)
            if status in (KVPoll.Success, KVPoll.Failed):
                # 终态：缓存结果
                self.conclude_state = status
            elif status == KVPoll.Bootstrapping:
                # 检测 bootstrap 阶段是否超时
                if self.init_time is not None:
                    now = time.time()
                    elapsed = now - self.init_time
                    if elapsed >= self.kv_mgr.bootstrap_timeout:
                        logger.warning_once(
                            "Some requests timed out when bootstrapping, "
                            "which means prefill instances fail to receive the KV indices from the decode instance of this request. "
                            "If a greater mean TTFT is acceptable, you can 'export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600' (10 minutes) to relax the timeout condition. "
                        )
                        self.kv_mgr.record_failure(
                            self.bootstrap_room,
                            f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.Bootstrapping",
                        )
                        self.conclude_state = KVPoll.Failed
                        return KVPoll.Failed

            return status
        else:
            return self.conclude_state

    def clear(self) -> None:
        """清理 bootstrap_room 的状态记录（请求完成后调用）。"""
        if self.bootstrap_room in self.kv_mgr.request_status:
            self.kv_mgr.request_status.pop(self.bootstrap_room)

    def failure_exception(self):
        """强制标记失败并抛出 KVTransferError（供其他 rank 调用以传播故障）。"""
        # Explicitly set the status to failure since this request has failed in another rank
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "Failed due to an unknown reason from another rank"
            )
        raise KVTransferError(self.bootstrap_room, failure_reason)

    def abort(self):
        """中止传输（收到 AbortReq 时调用），将状态强制置为 Failed。"""
        self.kv_mgr.record_failure(
            self.bootstrap_room,
            "Aborted by AbortReq.",
        )
        # Explicitly set the status to failure since this request has been aborted
        self.conclude_state = KVPoll.Failed


class MooncakeKVReceiver(CommonKVReceiver):
    """Decode 侧 KV 接收器。

    负责向 Prefill 侧 bootstrap server 注册自身的 KV 缓冲区地址（_register_kv_args），
    以及通过 send_metadata 通知 Prefill 开始传输，并通过 poll 轮询传输状态。
    支持等待超时检测（WaitingForInput 阶段）。
    """

    def __init__(
        self,
        mgr: MooncakeKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        # 获取本进程的 Mooncake session ID（含 IP:port，用于 Prefill 建立 RDMA session）
        self.session_id = mgr.get_session_id()
        # 超时计时起点（send_metadata 调用后设置）
        self.init_time = None
        super().__init__(mgr, bootstrap_addr, bootstrap_room)

    def _register_kv_args(self):
        """向所有 Prefill bootstrap server 注册本 Decode rank 的 KV 缓冲区地址信息。

        发送内容包括：KV/Aux/State 内存指针、TP 信息、HiSparse 标志、Staging 信息。
        Prefill 收到后存入 decode_kv_args_table，传输时按 session_id 查找。
        """
        for bootstrap_info in self.bootstrap_infos:
            # 打包 KV 数据指针（uint64 列表 → bytes）
            packed_kv_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.kv_data_ptrs
            )
            # 打包辅助数据指针
            packed_aux_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
            )
            # 打包状态数据指针（Mamba/SWA/NSA）
            packed_state_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.state_data_ptrs
            )
            # Pack state_item_lens and state_dim_per_tensor for mamba state slice transfer
            # 打包状态 item 长度（用于 Mamba 异构 TP 切片）
            packed_state_item_lens = b"".join(
                struct.pack("I", item_len)
                for item_len in self.kv_mgr.kv_args.state_item_lens
            )
            state_dim_per_tensor = getattr(
                self.kv_mgr.kv_args, "state_dim_per_tensor", []
            )
            # 打包状态张量的 TP 切片维度
            packed_state_dim_per_tensor = b"".join(
                struct.pack("I", dim) for dim in state_dim_per_tensor
            )
            # Note(shangming): No need to add pp rank here since decode pp size should be equal to prefill pp size or 1
            # 本 rank 的 engine_rank 作为 TP rank（PP rank 无需单独发送）
            tp_rank = self.kv_mgr.kv_args.engine_rank
            kv_item_len = self.kv_mgr.kv_args.kv_item_lens[0]
            dst_tp_rank = str(tp_rank).encode("ascii")
            dst_attn_tp_size = str(self.kv_mgr.attn_tp_size).encode("ascii")
            dst_kv_item_len = str(kv_item_len).encode("ascii")
            # HiSparse 标志（Decode 使用 host 内存池时置 1）
            enable_hisparse = b"1" if self.kv_mgr.server_args.enable_hisparse else b"0"

            # 若启用 Staging 且分配器已就绪，附带 Staging 内存信息
            if (
                self.kv_mgr.enable_staging
                and self.kv_mgr._staging_ctx.allocator is not None
            ):
                _alloc = self.kv_mgr._staging_ctx.allocator
                packed_staging_base_ptr = struct.pack("Q", _alloc.get_base_ptr())
                staging_total_size_str = str(_alloc.get_total_size()).encode("ascii")
            else:
                packed_staging_base_ptr = b""
                staging_total_size_str = b""

            # 建立到 Prefill bootstrap server 的 ZMQ 连接并发送注册消息
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            with lock:
                sock.send_multipart(
                    [
                        "None".encode("ascii"),  # room="None" 表示注册消息
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.session_id.encode("ascii"),
                        packed_kv_data_ptrs,
                        packed_aux_data_ptrs,
                        packed_state_data_ptrs,
                        dst_tp_rank,
                        dst_attn_tp_size,
                        dst_kv_item_len,
                        packed_state_item_lens,
                        packed_state_dim_per_tensor,
                        enable_hisparse,
                        packed_staging_base_ptr,
                        staging_total_size_str,
                    ]
                )

    def init(
        self,
        prefill_dp_rank: int,
    ):
        """初始化接收器（调用父类 init，解析 Prefill DP 路由信息）。"""
        super().init(prefill_dp_rank)

    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
    ):
        """向所有 Prefill bootstrap server 发送本次请求的 KV 传输元数据。

        消息包含：bootstrap_room、本地 IP/port、session_id、dst KV 索引、辅助/状态索引。
        Prefill 收到后将其存入 transfer_infos 并触发传输流程。
        """
        if self.bootstrap_infos is None:
            # bootstrap_infos 为空说明无法获取 Prefill 并行信息，直接标记失败
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                f"Could not fetch prefill parallel info from bootstrap_addr: {self.bootstrap_addr}",
            )
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return

        # 若启用 Staging，注册 room 的 bootstrap 信息（供 Staging 分配器使用）
        if (
            self.kv_mgr.enable_staging
            and self.kv_mgr._staging_ctx.allocator is not None
        ):
            self.chunk_staging_infos = []
            self.kv_mgr.register_staging_room_bootstrap(
                self.bootstrap_room, self.bootstrap_infos, self
            )

        # 向每个 Prefill bootstrap server 发送元数据
        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            is_dummy = bootstrap_info["is_dummy"]

            with lock:
                sock.send_multipart(
                    [
                        str(self.bootstrap_room).encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.session_id.encode("ascii"),
                        # dummy rank 不发送实际数据（传空 bytes）
                        kv_indices.tobytes() if not is_dummy else b"",
                        str(aux_index).encode("ascii") if not is_dummy else b"",
                        (
                            np.array(
                                state_indices,
                                dtype=np.int32,
                            ).tobytes()
                            if not is_dummy and state_indices is not None
                            else b""
                        ),
                        str(self.required_dst_info_num).encode("ascii"),
                    ]
                )
        # 记录发送时间，用于 WaitingForInput 超时检测
        self.init_time = time.time()

    def poll(self) -> KVPoll:
        """查询接收状态，支持 WaitingForInput 阶段超时自动失败。"""
        if self.conclude_state is None:
            status = self.kv_mgr.check_status(self.bootstrap_room)
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status
            elif status == KVPoll.WaitingForInput:
                # 检测等待 KV 传输完成信号是否超时
                if self.init_time is not None:
                    now = time.time()
                    elapsed = now - self.init_time
                    if elapsed >= self.kv_mgr.waiting_timeout:
                        logger.warning_once(
                            "Some requests fail to receive KV Cache transfer done signal after bootstrapping. "
                            "If a greater mean TTFT is acceptable, you can 'export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600' (10 minutes) to relax the timeout condition. "
                        )
                        self.kv_mgr.record_failure(
                            self.bootstrap_room,
                            f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.WaitingForInput",
                        )
                        self.conclude_state = KVPoll.Failed
                        return KVPoll.Failed

            return status

        else:
            return self.conclude_state

    def clear(self) -> None:
        """清理 bootstrap_room 相关的所有状态记录（请求完成后调用）。"""
        if self.bootstrap_room in self.kv_mgr.request_status:
            self.kv_mgr.request_status.pop(self.bootstrap_room)

        # 清理预期响应数量记录
        if self.bootstrap_room in self.kv_mgr.required_prefill_response_num_table:
            self.kv_mgr.required_prefill_response_num_table.pop(self.bootstrap_room)

        # 清理已响应的 Prefill rank 记录
        if self.bootstrap_room in self.kv_mgr.prefill_response_tracker:
            self.kv_mgr.prefill_response_tracker.pop(self.bootstrap_room)

    def failure_exception(self):
        """强制标记失败并抛出 KVTransferError（供其他 rank 调用以传播故障）。"""
        # Explicitly set the status to failure since this request has failed in another rank
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "Failed due to an unknown reason from another rank"
            )
        raise KVTransferError(self.bootstrap_room, failure_reason)

    def abort(self):
        """中止接收（收到 AbortReq 时调用），将状态强制置为 Failed。"""
        self.kv_mgr.record_failure(
            self.bootstrap_room,
            "Aborted by AbortReq.",
        )
        # Explicitly set the status to failure since this request has been aborted
        self.conclude_state = KVPoll.Failed


class MooncakeKVBootstrapServer(CommonKVBootstrapServer):
    """Mooncake 传输后端的 bootstrap 服务器（直接使用 CommonKVBootstrapServer 实现）。"""
    pass
