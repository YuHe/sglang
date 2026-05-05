# mori/conn.py：基于 Mori RDMA 传输引擎的 PD 分离 KV 传输实现
# Mori 是一个基于 RDMA（InfiniBand/RoCE）的高性能数据传输库，支持 batch_write 批量 RDMA 写操作
# 本文件实现 MoriKVManager、MoriKVSender、MoriKVReceiver 和 MoriKVBootstrapServer
from __future__ import annotations

import ctypes
import dataclasses
import logging
import os
import struct
import threading
import time
from typing import Dict, List, Optional, Tuple

import msgspec
import numpy as np
import numpy.typing as npt
# Mori 传输状态枚举（InProgress / Success / Failed）
from mori.cpp import TransferStatus
from mori.io import (
    BackendType,       # 传输后端类型（RDMA）
    EngineDesc,        # 引擎描述符（包含连接信息）
    IOEngine,          # Mori IO 引擎主类
    IOEngineConfig,    # 引擎配置（host/port）
    MemoryDesc,        # 内存描述符（ptr/len/rdma_key）
    MemoryLocationType,  # 内存位置类型（GPU/CPU）
    PollCqMode,        # RDMA CQ 轮询模式（POLLING/INTERRUPT）
    RdmaBackendConfig, # RDMA 后端配置（QP数/batch_size/workers）
)

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    filter_kv_indices_for_cp_rank,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import get_int_env_var
from sglang.srt.utils.network import NetworkAddress, get_local_ip_auto

logger = logging.getLogger(__name__)
# MORI_GUARD：ZMQ 消息帧头标志，用于区分不同类型消息（防止误处理）
MORI_GUARD = b"MoriMsgGuard"


# _pack_mem_desc_list / _unpack_mem_desc_list：MemoryDesc 列表的 msgpack 序列化/反序列化
# 用于通过 ZMQ 传输 Mori 内存描述符（注册的 RDMA 内存区域元数据）
def _pack_mem_desc_list(mems: List[MemoryDesc]) -> bytes:
    if not mems:
        return b""
    packed_descs = [mem.pack() for mem in mems]
    return msgspec.msgpack.encode(packed_descs)


def _unpack_mem_desc_list(blob: bytes) -> List[MemoryDesc]:
    if not blob:
        return []
    desc_blobs = msgspec.msgpack.decode(blob)
    return [MemoryDesc.unpack(b) for b in desc_blobs]


# TransferInfo：decode 侧通过 ZMQ 发送给 prefill 侧的 KV 传输目标信息
# prefill 侧收到后，即可发起 RDMA batch_write 将 KV 数据写入 decode 侧内存
@dataclasses.dataclass
class TransferInfo:
    room: int                              # bootstrap 房间号（请求唯一标识）
    endpoint: str                          # decode 侧 IP 地址
    dst_port: int                          # decode 侧 ZMQ 状态回报端口
    engine_key: str                        # decode 侧 Mori 引擎 key（用于查找已注册 peer）
    dst_kv_indices: npt.NDArray[np.int32]  # decode 侧目标 KV page 索引
    dst_aux_index: int                     # decode 侧目标 aux cache 索引（-1 表示无）
    required_dst_info_num: int             # 需要收集的 dst info 数量（TP 不同时 >1）
    is_dummy: bool                         # 是否为哑请求（kv_indices 为空，无需实际传输）

    @classmethod
    def from_zmq(cls, payload: List[bytes]) -> TransferInfo:
        # 从 ZMQ 多帧消息中解析 TransferInfo
        room = int(payload[0].decode("ascii"))
        endpoint = payload[1].decode("ascii")
        dst_port = int(payload[2].decode("ascii"))
        engine_key = payload[3].decode("ascii")

        # 解析 KV 索引字节（可为空，表示哑请求）
        if payload[4]:
            dst_kv_indices = np.frombuffer(payload[4], dtype=np.int32)
        else:
            dst_kv_indices = np.array([], dtype=np.int32)

        # 解析 aux cache 索引（-1 表示无 aux 传输）
        if payload[5]:
            dst_aux_index = int(payload[5].decode("ascii"))
        else:
            dst_aux_index = -1

        # required_dst_info_num：decode TP > prefill TP 时需要多个 dst info
        required_dst_info_num = (
            int(payload[7].decode("ascii")) if len(payload) > 7 else 1
        )
        # kv_indices 和 aux_index 均为空时判定为哑请求
        is_dummy = dst_kv_indices.size == 0 and dst_aux_index < 0
        return cls(
            room=room,
            endpoint=endpoint,
            dst_port=dst_port,
            engine_key=engine_key,
            dst_kv_indices=dst_kv_indices,
            dst_aux_index=dst_aux_index,
            required_dst_info_num=required_dst_info_num,
            is_dummy=is_dummy,
        )


# KVArgsRegisterInfo：decode 侧通过 ZMQ 发送给 prefill 侧的 KV 内存注册信息
# prefill 侧收到后调用 register_remote_engine 建立 RDMA 连接，并缓存 decode 的内存描述符
@dataclasses.dataclass
class KVArgsRegisterInfo:
    endpoint: str                          # decode 侧 IP
    dst_port: int                          # decode 侧 ZMQ 端口
    engine_desc: EngineDesc                # decode Mori 引擎描述符（用于 RDMA 连接建立）
    dst_kv_mem_descs: List[MemoryDesc]     # decode 侧 KV 缓冲区 RDMA 描述符列表
    dst_aux_mem_descs: List[MemoryDesc]    # decode 侧 aux cache 描述符列表
    dst_state_mem_descs: List[MemoryDesc]  # decode 侧 state cache 描述符列表
    gpu_id: int                            # decode 侧 GPU id
    decode_tp_size: int                    # decode 侧 attention TP 并行度
    decode_tp_rank: int                    # decode 侧当前 TP rank
    dst_kv_item_len: int                   # decode 侧每个 KV page 的字节数

    @property
    def engine_key(self) -> str:
        # 引擎 key：唯一标识 decode 侧的 Mori 引擎
        return self.engine_desc.key

    @classmethod
    def from_zmq(cls, payload: List[bytes]) -> KVArgsRegisterInfo:
        # 从 ZMQ 多帧消息中解析 KVArgsRegisterInfo
        endpoint = payload[1].decode("ascii")
        dst_port = int(payload[2].decode("ascii"))
        engine_desc = EngineDesc.unpack(payload[3])
        dst_kv_mem_descs = _unpack_mem_desc_list(payload[4])
        dst_aux_mem_descs = _unpack_mem_desc_list(payload[5])
        dst_state_mem_descs = _unpack_mem_desc_list(payload[6])
        gpu_id = int(payload[7].decode("ascii"))
        decode_tp_size = int(payload[8].decode("ascii"))
        decode_tp_rank = int(payload[9].decode("ascii"))
        dst_kv_item_len = int(payload[10].decode("ascii"))
        return cls(
            endpoint=endpoint,
            dst_port=dst_port,
            engine_desc=engine_desc,
            dst_kv_mem_descs=dst_kv_mem_descs,
            dst_aux_mem_descs=dst_aux_mem_descs,
            dst_state_mem_descs=dst_state_mem_descs,
            gpu_id=gpu_id,
            decode_tp_size=decode_tp_size,
            decode_tp_rank=decode_tp_rank,
            dst_kv_item_len=dst_kv_item_len,
        )


# AuxDataCodec：aux cache 数据的序列化/反序列化
# Mori 使用 ZMQ TCP 传输 aux cache（非 RDMA），通过 ctypes 直接读写 GPU/CPU 内存
class AuxDataCodec:
    @staticmethod
    def serialize_data_from_buffer(src_addr, data_length):
        # 从内存地址 src_addr 读取 data_length 字节（ctypes 零拷贝读取）
        buffer = (ctypes.c_byte * data_length).from_address(src_addr)
        return bytes(buffer)

    @staticmethod
    def deserialize_data_to_buffer(kv_args, buffer_index, aux_index, data):
        # 将数据写入 decode 侧 aux cache 的指定 slot（ctypes 直接内存写入）
        dst_aux_ptr = kv_args.aux_data_ptrs[buffer_index]
        item_len = kv_args.aux_item_lens[buffer_index]
        dst_addr = dst_aux_ptr + item_len * aux_index
        buffer = (ctypes.c_byte * len(data)).from_address(dst_addr)
        buffer[:] = data
        return


# TPSliceConfig：异构 TP 传输时的 head 切片配置
# 当 prefill TP != decode TP 时，每次 RDMA 写只传输每页中对应的 head 字节切片
@dataclasses.dataclass
class TPSliceConfig:
    page_size: int               # 每页 token 数
    src_item_len: int            # prefill 侧每页字节数
    dst_item_len: int            # decode 侧每页字节数
    bytes_per_token_src: int     # prefill 侧每 token 字节数
    bytes_per_token_dst: int     # decode 侧每 token 字节数
    src_head_slice_offset: int   # 源侧 head 切片起始字节偏移（在每 token 内）
    dst_head_slice_offset: int   # 目标侧 head 切片起始字节偏移
    heads_bytes_per_token_to_send: int  # 每 token 需要传输的 head 字节数


# MoriKVManager：基于 Mori RDMA 的 KV 管理器
# 初始化 IOEngine，注册本地 KV/aux/state 内存，提供 RDMA batch_write 传输能力
# prefill 侧启动 bootstrap 线程监听 decode 发来的注册/传输消息
# decode 侧启动 decode 线程监听 prefill 发回的传输完成状态和 aux 数据
class MoriKVManager(CommonKVManager):
    AUX_DATA_HEADER = b"AUX_DATA"  # aux cache 数据消息头标志

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        # 初始化 Mori IOEngine（RDMA 传输引擎）
        self.engine = self._init_engine()
        # 获取本引擎的 EngineDesc（包含连接信息，用于向对端注册）
        self.engine_desc = self.engine.get_engine_desc()
        # kv/aux/state 内存描述符列表（注册 RDMA 内存区域后获得）
        self.kv_mem_descs: List[MemoryDesc] = []
        self.aux_mem_descs: List[MemoryDesc] = []
        self.state_mem_descs: List[MemoryDesc] = []
        # 传输锁：保护 transfer_infos 和状态更新的并发访问
        self.transfer_lock = threading.Lock()
        # 注册本地 KV/aux/state 缓冲区到 RDMA 引擎
        self._register_local_buffers()
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # prefill 侧：启动 bootstrap 监听线程（等待 decode 发来的注册/KV索引消息）
            self._start_bootstrap_thread()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            # decode 侧：维护 room -> bootstrap_addr 映射，启动 decode 状态监听线程
            self.room_to_bootstrap_addr: Dict[int, str] = {}
            self._start_decode_thread()

    def _init_engine(self) -> IOEngine:
        # 初始化 Mori IOEngine：配置 RDMA 设备、创建 RDMA 后端
        if self.kv_args.ib_device:
            # 通过环境变量指定 RDMA 设备（如 mlx5_0）
            os.environ["MORI_RDMA_DEVICES"] = self.kv_args.ib_device

        self.local_ip = get_local_ip_auto()
        config = IOEngineConfig(host=self.local_ip, port=0)  # port=0：自动绑定空闲端口

        # 引擎 key：唯一标识本引擎（包含模式/DP rank/TP rank/PID/IP）
        engine_key = (
            f"io-{self.disaggregation_mode.value}-"
            f"dp{self.system_dp_rank}-tp{self.attn_tp_rank}-"
            f"pid{os.getpid()}-{self.local_ip}"
        )

        engine = IOEngine(engine_key, config)
        # 使用 POLLING 模式（低延迟 busy-wait，适合 KV 传输场景）
        poll_mode = PollCqMode.POLLING

        # Number of RDMA Queue Pairs (QPs) used per transfer operation.
        # Higher values can increase parallelism and bandwidth utilization.
        # Default: 1
        # RDMA QP 数：更多 QP 可提高并行度和带宽利用率（默认 1）
        qp_per_transfer = get_int_env_var("SGLANG_MORI_QP_PER_TRANSFER", 1)

        # Number of RDMA work requests posted in a single batch to each QP.
        # Larger batch sizes reduce per-operation overhead and improve throughput
        # at the cost of higher latency. Use -1 for automatic sizing based on
        # the number of merged work requests and available endpoints.
        # Default: -1 (automatic)
        # 每个 QP 单次 post 的 WR 数量：-1 为自动（根据 merged WR 数和端点数计算）
        post_batch_size = get_int_env_var("SGLANG_MORI_POST_BATCH_SIZE", -1)

        # Number of worker threads in the RDMA executor thread pool.
        # Each worker handles RDMA operations on a separate CPU core (with affinity).
        # More workers can improve parallelism for large batch transfers across
        # multiple QPs, but excessive threads may cause contention.
        # Default: 1
        # RDMA 执行线程数：每个 worker 绑定一个 CPU core（亲和性），默认 1
        num_worker_threads = get_int_env_var("SGLANG_MORI_NUM_WORKERS", 1)

        # 创建 RDMA 后端并配置 QP/batch/workers/poll 模式
        rdma_cfg = RdmaBackendConfig(
            qp_per_transfer,
            post_batch_size,
            num_worker_threads,
            poll_mode,
            False,  # 不启用 inline data
        )
        engine.create_backend(BackendType.RDMA, rdma_cfg)
        actual_port = engine.get_engine_desc().port
        assert actual_port > 0, f"Failed to bind port for engine {engine_key}"
        logger.debug(
            "Initialized Mori IOEngine %s at %s:%s (qp_per_transfer=%s, workers=%s, poll_mode=%s)",
            engine_key,
            self.local_ip,
            actual_port,
            qp_per_transfer,
            num_worker_threads,
            poll_mode.name,
        )
        return engine

    def _register_local_buffers(self) -> None:
        # 将本地 KV/aux/state 内存注册到 Mori RDMA 引擎，获取 MemoryDesc（含 rdma_key）
        # KV 缓冲区（GPU 内存）：层级按 [K_layer0, ..., V_layer0, ...] 排列
        for ptr, length in zip(self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens):
            mem_desc = self.engine.register_memory(
                ptr,
                length,
                self.kv_args.gpu_id,
                MemoryLocationType.GPU,  # GPU HBM 内存
            )
            self.kv_mem_descs.append(mem_desc)
        # aux cache 缓冲区（CPU 内存，如 logprobs/hidden_states）
        for ptr, length in zip(self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens):
            desc = self.engine.register_memory(
                ptr,
                length,
                -1,                          # CPU 内存，gpu_id=-1
                MemoryLocationType.CPU,
            )
            self.aux_mem_descs.append(desc)
        # state cache 缓冲区（GPU 内存，如 MLA latent state）
        for ptr, length in zip(
            self.kv_args.state_data_ptrs, getattr(self.kv_args, "state_data_lens", [])
        ):
            desc = self.engine.register_memory(
                ptr,
                length,
                self.kv_args.gpu_id,
                MemoryLocationType.GPU,
            )
            self.state_mem_descs.append(desc)

    def _handle_register_message(self, payload: List[bytes]) -> None:
        # 处理 decode 侧发来的 KVArgs 注册消息（建立 RDMA 连接，缓存 decode 内存描述符）
        try:
            register_info = KVArgsRegisterInfo.from_zmq(payload)
            self._add_remote_peer(register_info)
        except Exception:
            logger.exception("Failed to register remote peer")

    def _handle_transfer_message(self, payload: List[bytes]) -> None:
        # 处理 decode 侧发来的 KV 索引传输消息（记录 TransferInfo，等待足够数量后推进状态）
        try:
            transfer_info = TransferInfo.from_zmq(payload)
            infos = self.transfer_infos.setdefault(transfer_info.room, {})
            # engine_key 作为 key，防止同一 room 的重复消息
            infos[transfer_info.engine_key] = transfer_info

            if len(infos) >= transfer_info.required_dst_info_num:
                # 收集到足够的 dst info 后，推进状态到 WaitingForInput（可以开始传输）
                logger.debug(
                    "Bootstrap room %s got enough transfer info (%s)",
                    transfer_info.room,
                    len(infos),
                )
                self.update_status(transfer_info.room, KVPoll.WaitingForInput)
        except Exception:
            logger.exception("Failed to parse transfer info message")

    def _validate_message(self, msg: List[bytes]) -> Optional[List[bytes]]:
        # 验证 ZMQ 消息头是否为 MORI_GUARD，过滤非法消息
        if not msg or msg[0] != MORI_GUARD:
            logger.warning("Received malformed bootstrap message")
            return None
        payload = msg[1:]
        if not payload:
            return None
        return payload

    def _start_bootstrap_thread(self) -> None:
        # prefill 侧启动 bootstrap 监听线程：处理来自 decode 的注册和传输消息
        def bootstrap_worker():
            while True:
                try:
                    msg = self.server_socket.recv_multipart()
                    payload = self._validate_message(msg)
                    if payload is None:
                        continue
                    room = payload[0].decode("ascii")

                    if room == "None":
                        # room 为 "None" 表示注册消息（KVArgs 注册）
                        self._handle_register_message(payload)
                    else:
                        # 否则为 KV 索引传输消息
                        self._handle_transfer_message(payload)
                except Exception:
                    logger.exception("Bootstrap worker failed")

        threading.Thread(target=bootstrap_worker, daemon=True).start()

    def _cleanup_room_tracking(self, bootstrap_room: int) -> None:
        # 传输完成或失败后，清理 room -> bootstrap_addr 映射和 addr_to_rooms_tracker
        bootstrap_addr = self.room_to_bootstrap_addr.pop(bootstrap_room, None)
        if bootstrap_addr is not None:
            rooms = self.addr_to_rooms_tracker.get(bootstrap_addr)
            if rooms is not None:
                rooms.discard(bootstrap_room)
                if not rooms:
                    self.addr_to_rooms_tracker.pop(bootstrap_addr, None)

    def _start_decode_thread(self) -> None:
        # decode 侧启动监听线程：处理 prefill 发回的 KVPoll 状态和 aux 数据
        def decode_worker():
            while True:
                try:
                    msg = self.server_socket.recv_multipart()
                    if msg and msg[0] == MoriKVManager.AUX_DATA_HEADER:
                        # aux 数据消息：反序列化并写入本地 aux cache
                        self._handle_aux_data(msg)
                        continue

                    # 验证消息头
                    if not msg or msg[0] != MORI_GUARD:
                        logger.warning(
                            "Received malformed status message on decode worker"
                        )
                        continue
                    payload = msg[1:]
                    if len(payload) < 3:
                        logger.warning("Incomplete status payload received")
                        continue
                    # 解析 prefill 发回的传输状态
                    bootstrap_room = int(payload[0].decode("ascii"))
                    status_code = int(payload[1].decode("ascii"))
                    # prefill_rank：attn_tp_rank * pp_size + pp_rank（用于追踪哪个 prefill rank 完成）
                    prefill_rank = int(payload[2].decode("ascii"))
                    failure_reason = (
                        payload[3].decode("utf-8")
                        if len(payload) > 3 and payload[3]
                        else None
                    )

                    if status_code == KVPoll.Success:
                        # 记录已完成的 prefill rank，当所有期望的 rank 完成后更新为 Success
                        tracker = self.prefill_response_tracker[bootstrap_room]
                        tracker.add(prefill_rank)
                        expected = self.required_prefill_response_num_table.get(
                            bootstrap_room, 1
                        )
                        if len(tracker) >= expected:
                            # 所有 prefill rank 均已完成，传输成功
                            self.prefill_response_tracker.pop(bootstrap_room, None)
                            self.update_status(bootstrap_room, KVPoll.Success)
                            self._cleanup_room_tracking(bootstrap_room)
                    elif status_code == KVPoll.Failed:
                        # 任意一个 prefill rank 失败，整个传输失败
                        if failure_reason:
                            self.record_failure(bootstrap_room, failure_reason)
                        self.prefill_response_tracker.pop(bootstrap_room, None)
                        self.update_status(bootstrap_room, KVPoll.Failed)
                        self._cleanup_room_tracking(bootstrap_room)
                    else:
                        logger.warning(
                            "Unknown status code %s received for room %s",
                            status_code,
                            bootstrap_room,
                        )
                except Exception:
                    logger.exception("Decode status worker failed")

        threading.Thread(target=decode_worker, daemon=True).start()

    def notify_decode_status(
        self,
        infos: List[TransferInfo],
        bootstrap_room: int,
        status: KVPoll,
        failure_reason: Optional[str] = None,
    ) -> None:
        # 向 decode 侧发送传输完成/失败状态（通过 ZMQ PUSH 发送到 decode 的 ZMQ PULL socket）
        if not infos:
            return
        payload = [
            MORI_GUARD,
            str(bootstrap_room).encode("ascii"),
            str(int(status)).encode("ascii"),
            # prefill_rank = attn_tp_rank * pp_size + pp_rank，decode 侧用于追踪
            str(self.attn_tp_rank * self.pp_size + self.pp_rank).encode("ascii"),
            failure_reason.encode("utf-8") if failure_reason else b"",
        ]
        for info in infos:
            try:
                na = NetworkAddress(info.endpoint, info.dst_port)
                socket = self._connect(na.to_tcp(), is_ipv6=na.is_ipv6)
                socket.send_multipart(payload)
            except Exception:
                logger.exception(
                    "Failed to sync status %s to decode endpoint %s:%s for room %s",
                    status,
                    info.endpoint,
                    info.dst_port,
                    bootstrap_room,
                )

    def _add_remote_peer(self, register_info: KVArgsRegisterInfo) -> None:
        # 向 Mori 引擎注册 decode 侧 peer，并缓存其 KVArgsRegisterInfo
        engine_key = register_info.engine_key
        if engine_key in self.decode_kv_args_table:
            # 已注册，避免重复（幂等操作）
            logger.debug("Remote peer %s already registered. Skipping.", engine_key)
            return
        # 建立到 decode 侧的 RDMA 连接（QP 握手）
        self.engine.register_remote_engine(register_info.engine_desc)
        # 缓存 decode 侧内存描述符，后续 send_kvcache 使用
        self.decode_kv_args_table[engine_key] = register_info
        logger.debug(
            "Registered decode peer %s (%s:%s)",
            engine_key,
            register_info.endpoint,
            register_info.dst_port,
        )

    def _get_mha_mem_desc_slices(
        self, dst_mem_descs: List[MemoryDesc]
    ) -> tuple[
        List[MemoryDesc], List[MemoryDesc], List[MemoryDesc], List[MemoryDesc], int
    ]:
        # 按 PP stage 从 MHA KV 描述符中切片：返回 (src_k, src_v, dst_k, dst_v, layers)
        src_descs = self.kv_mem_descs
        if not src_descs:
            raise RuntimeError("KV memory descriptors are empty on prefill side")

        num_local_layers = len(src_descs) // 2
        # 源侧：[K_layer0, ..., V_layer0, ...]
        src_k_descs = src_descs[:num_local_layers]
        src_v_descs = src_descs[num_local_layers:]

        start_layer = self.kv_args.prefill_start_layer
        end_layer = start_layer + num_local_layers
        dst_total_layers = len(dst_mem_descs) // 2
        if len(dst_mem_descs) < 2 or end_layer > dst_total_layers:
            raise ValueError(
                "Destination KV descriptors do not match prefill pp configuration"
            )
        # 目标侧：按 PP stage 层范围切片
        dst_k_descs = dst_mem_descs[start_layer:end_layer]
        dst_v_descs = dst_mem_descs[
            dst_total_layers + start_layer : dst_total_layers + end_layer
        ]
        return src_k_descs, src_v_descs, dst_k_descs, dst_v_descs, num_local_layers

    def _get_mla_mem_desc_slices(
        self, dst_mem_descs: List[MemoryDesc]
    ) -> tuple[List[MemoryDesc], List[MemoryDesc], int]:
        # MLA 模式：KV 无 K/V 分离，直接按 PP stage 切片
        src_descs = self.kv_mem_descs
        num_local_layers = len(src_descs)
        start_layer = self.kv_args.prefill_start_layer
        end_layer = start_layer + num_local_layers
        if end_layer > len(dst_mem_descs):
            raise ValueError(
                "Destination MLA KV descriptors do not match prefill pp configuration"
            )
        dst_slice = dst_mem_descs[start_layer:end_layer]
        return src_descs, dst_slice, num_local_layers

    def _issue_layer_transfers(
        self,
        src_desc: MemoryDesc,
        dst_desc: MemoryDesc,
        kv_item_len: int,
        src_groups: List[List[int]],
        dst_groups: List[List[int]],
    ) -> List[TransferStatus]:
        # 对一层发起 batch RDMA write：将 src_groups 中的页组写到 dst_groups 对应位置
        if not src_groups:
            return []
        # 将页索引转换为字节偏移（page_idx * kv_item_len = 页起始地址偏移）
        local_offsets = [int(src_group[0]) * kv_item_len for src_group in src_groups]
        remote_offsets = [int(dst_group[0]) * kv_item_len for dst_group in dst_groups]
        sizes = [len(src_group) * kv_item_len for src_group in src_groups]

        # 分配唯一 transfer_uid，用于追踪完成状态
        transfer_uid = self.engine.allocate_transfer_uid()

        # batch_write：批量 RDMA 写（减少 QP 轮次，提高吞吐）
        statuses = self.engine.batch_write(
            [src_desc],
            [local_offsets],
            [dst_desc],
            [remote_offsets],
            [sizes],
            [transfer_uid],
        )
        return statuses

    def _build_tp_slice_config(self, peer_info: KVArgsRegisterInfo) -> TPSliceConfig:
        # 构建异构 TP 传输的 head 切片配置（prefill TP != decode TP 时使用）
        page_size = self.kv_args.page_size

        src_item_len = self.kv_args.kv_item_lens[0]   # prefill 每页字节数
        dst_item_len = peer_info.dst_kv_item_len        # decode 每页字节数

        bytes_per_token_src = src_item_len // page_size  # prefill 每 token 字节数
        bytes_per_token_dst = dst_item_len // page_size  # decode 每 token 字节数

        prefill_tp_size = self.attn_tp_size
        decode_tp_size = peer_info.decode_tp_size

        # 计算 head 数量（每个 prefill rank 持有的 head 数，以及 decode 侧期望数）
        num_kv_heads = self.kv_args.kv_head_num
        src_heads_per_rank = num_kv_heads
        dst_heads_per_rank = num_kv_heads * prefill_tp_size // decode_tp_size
        if dst_heads_per_rank == 0:
            raise ValueError("Destination heads per rank evaluates to zero")

        # 每个 head 的字节数（以 decode token 为基准）
        bytes_per_head_slice = bytes_per_token_dst // dst_heads_per_rank
        if bytes_per_head_slice == 0:
            raise ValueError("Head slice size evaluates to zero")

        local_tp_rank = self.kv_args.engine_rank % prefill_tp_size
        dst_tp_rank = peer_info.decode_tp_rank % decode_tp_size

        if prefill_tp_size > decode_tp_size:
            # prefill TP > decode TP：prefill 每 rank 发送全部 head，目标 head 偏移不同
            src_head_start = 0
            num_heads_to_send = src_heads_per_rank
            dst_head_start = local_tp_rank * src_heads_per_rank
        else:
            # prefill TP <= decode TP：prefill 只发送 decode rank 对应的 head 切片
            src_head_start = (dst_tp_rank * dst_heads_per_rank) % src_heads_per_rank
            num_heads_to_send = dst_heads_per_rank
            dst_head_start = 0

        # 计算 head 切片在每 token 内的字节偏移
        src_head_slice_offset = src_head_start * bytes_per_head_slice
        dst_head_slice_offset = dst_head_start * bytes_per_head_slice
        heads_bytes_per_token = num_heads_to_send * bytes_per_head_slice

        if heads_bytes_per_token > bytes_per_token_dst:
            raise ValueError(
                "Slice size exceeds destination token capacity for TP slice transfer"
            )

        return TPSliceConfig(
            page_size=page_size,
            src_item_len=src_item_len,
            dst_item_len=dst_item_len,
            bytes_per_token_src=bytes_per_token_src,
            bytes_per_token_dst=bytes_per_token_dst,
            src_head_slice_offset=src_head_slice_offset,
            dst_head_slice_offset=dst_head_slice_offset,
            heads_bytes_per_token_to_send=heads_bytes_per_token,
        )

    def _issue_tp_slice_transfers(
        self,
        src_desc: MemoryDesc,
        dst_desc: MemoryDesc,
        kv_indices: npt.NDArray[np.int32],
        dst_indices: npt.NDArray[np.int32],
        tp_cfg: TPSliceConfig,
    ) -> List[TransferStatus]:
        # 异构 TP 传输：对每对 (src_page, dst_page) 内的每个 token 发起 head 切片 RDMA write
        if kv_indices.size == 0 or dst_indices.size == 0:
            return []

        limit = min(kv_indices.size, dst_indices.size)
        if not limit:
            return []

        src_pages = kv_indices[:limit].astype(np.int64)
        dst_pages = dst_indices[:limit].astype(np.int64)
        # 页内 token slot 偏移数组（[0, 1, ..., page_size-1]）
        token_slots = np.arange(tp_cfg.page_size, dtype=np.int64)

        # 每页的字节基地址
        src_page_bases = src_pages * tp_cfg.src_item_len
        dst_page_bases = dst_pages * tp_cfg.dst_item_len

        # 页内 token 字节偏移
        src_token_offsets = token_slots * tp_cfg.bytes_per_token_src
        dst_token_offsets = token_slots * tp_cfg.bytes_per_token_dst

        # 展开为每个 (page, token) 对的字节偏移列表（加上 head 切片偏移）
        local_offsets = (
            (
                src_page_bases[:, np.newaxis]
                + src_token_offsets
                + tp_cfg.src_head_slice_offset
            )
            .flatten()
            .tolist()
        )
        remote_offsets = (
            (
                dst_page_bases[:, np.newaxis]
                + dst_token_offsets
                + tp_cfg.dst_head_slice_offset
            )
            .flatten()
            .tolist()
        )

        # 每次 RDMA write 的字节数（每 token 中的 head 切片大小）
        num_transfers = limit * tp_cfg.page_size
        sizes = [tp_cfg.heads_bytes_per_token_to_send] * num_transfers

        if not local_offsets:
            return []

        transfer_uid = self.engine.allocate_transfer_uid()
        statuses = self.engine.batch_write(
            [src_desc],
            [local_offsets],
            [dst_desc],
            [remote_offsets],
            [sizes],
            [transfer_uid],
        )
        return statuses

    def send_kvcache(
        self,
        peer_info: KVArgsRegisterInfo,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_indices: npt.NDArray[np.int32],
    ) -> List[TransferStatus]:
        # 执行 KV 缓存 RDMA 传输（支持 MLA/MHA，同构/异构 TP）
        src_groups, dst_groups = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )
        statuses = []
        kv_item_len = self.kv_args.kv_item_lens[0]
        if self.is_mla_backend:
            # MLA 模式：KV 无 K/V 分离，直接逐层传输
            (
                src_descs,
                dst_descs,
                layers_current_pp_stage,
            ) = self._get_mla_mem_desc_slices(peer_info.dst_kv_mem_descs)
            for layer_id in range(layers_current_pp_stage):
                statuses.extend(
                    self._issue_layer_transfers(
                        src_descs[layer_id],
                        dst_descs[layer_id],
                        kv_item_len,
                        src_groups,
                        dst_groups,
                    )
                )
        else:
            # MHA 模式：K/V 分离传输
            tp_mismatch = peer_info.decode_tp_size != self.attn_tp_size
            (
                src_k_descs,
                src_v_descs,
                dst_k_descs,
                dst_v_descs,
                layers_current_pp_stage,
            ) = self._get_mha_mem_desc_slices(peer_info.dst_kv_mem_descs)

            if tp_mismatch:
                # 异构 TP：使用 head 切片传输（每页每 token 只传部分 head）
                tp_cfg = self._build_tp_slice_config(peer_info)
                for layer_id in range(layers_current_pp_stage):
                    statuses.extend(
                        self._issue_tp_slice_transfers(
                            src_k_descs[layer_id],
                            dst_k_descs[layer_id],
                            prefill_kv_indices,
                            dst_kv_indices,
                            tp_cfg,
                        )
                    )
                    statuses.extend(
                        self._issue_tp_slice_transfers(
                            src_v_descs[layer_id],
                            dst_v_descs[layer_id],
                            prefill_kv_indices,
                            dst_kv_indices,
                            tp_cfg,
                        )
                    )
            else:
                # 同构 TP：整页传输（按连续段 batch write）
                src_groups, dst_groups = group_concurrent_contiguous(
                    prefill_kv_indices, dst_kv_indices
                )
                for layer_id in range(layers_current_pp_stage):
                    statuses.extend(
                        self._issue_layer_transfers(
                            src_k_descs[layer_id],
                            dst_k_descs[layer_id],
                            kv_item_len,
                            src_groups,
                            dst_groups,
                        )
                    )
                    statuses.extend(
                        self._issue_layer_transfers(
                            src_v_descs[layer_id],
                            dst_v_descs[layer_id],
                            kv_item_len,
                            src_groups,
                            dst_groups,
                        )
                    )

        return statuses

    def send_aux(
        self,
        peer_info: KVArgsRegisterInfo,
        prefill_aux_index: int,
        dst_aux_index: int,
        room: int,
    ) -> List[TransferStatus]:
        # aux cache 传输入口：目前使用 TCP（ZMQ）发送，非 RDMA
        return self.send_aux_tcp(peer_info, prefill_aux_index, dst_aux_index, room)

    def send_aux_tcp(
        self,
        peer_info: KVArgsRegisterInfo,
        prefill_aux_index: int,
        dst_aux_index: int,
        room: int,
    ) -> List[TransferStatus]:
        # 通过 ZMQ TCP 发送 aux cache 数据（逐 buffer 序列化并推送）
        prefill_aux_ptrs = self.kv_args.aux_data_ptrs
        prefill_aux_item_lens = self.kv_args.aux_item_lens

        for i in range(len(prefill_aux_ptrs)):
            length = prefill_aux_item_lens[i]
            # 计算 src 地址：aux_ptr + item_len * index（第 i 个 buffer 的第 prefill_aux_index 个 slot）
            src_addr = prefill_aux_ptrs[i] + length * prefill_aux_index
            data = AuxDataCodec.serialize_data_from_buffer(src_addr, length)

            self.send_aux_data_to_endpoint(
                remote=peer_info.endpoint,
                dst_port=peer_info.dst_port,
                room=room,
                buffer_index=i,
                aux_index=dst_aux_index,
                data=data,
            )

        return []

    def send_aux_data_to_endpoint(
        self,
        remote: str,
        dst_port: int,
        room: int,
        buffer_index: int,
        aux_index: int,
        data: bytes,
    ):
        # 向 decode 侧 ZMQ PULL socket 发送 AUX_DATA 多帧消息
        # 消息格式：[AUX_DATA_HEADER, room, buffer_index, aux_index, data_len(4B big-endian), data]
        na = NetworkAddress(remote, dst_port)
        socket = self._connect(na.to_tcp(), is_ipv6=na.is_ipv6)

        socket.send_multipart(
            [
                MoriKVManager.AUX_DATA_HEADER,
                str(room).encode("ascii"),
                str(buffer_index).encode("ascii"),
                str(aux_index).encode("ascii"),
                struct.pack(">I", len(data)),  # 4 字节大端长度（用于接收侧校验）
                data,
            ]
        )

    def _handle_aux_data(self, msg: List[bytes]):
        """Handle AUX_DATA messages received by the decode thread."""
        # 解析并写入 aux cache 数据（校验长度后反序列化到目标内存地址）
        room = int(msg[1].decode("ascii"))
        buffer_index = int(msg[2].decode("ascii"))
        aux_index = int(msg[3].decode("ascii"))
        data_length = struct.unpack(">I", msg[4])[0]
        data = msg[5]

        if len(data) != data_length:
            logger.error(f"AUX_DATA length mismatch for bootstrap_room {room}")
            return

        AuxDataCodec.deserialize_data_to_buffer(
            self.kv_args, buffer_index, aux_index, data
        )

        logger.debug(
            f"Received AUX_DATA for bootstrap_room {room} with length:{len(data)}"
        )

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last: bool,
        aux_index: Optional[int] = None,
        state_indices: Optional[npt.NDArray[np.int32]] = None,
    ) -> Tuple[List[TransferStatus], Optional[List[TransferInfo]]]:
        # 发起 KV 传输请求：向所有目标 decode peer 发送 KV 缓存
        # 返回 (TransferStatus 列表, 最后一块的 TransferInfo 列表（用于发送完成通知）)
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        transfer_infos = self.transfer_infos.get(bootstrap_room)
        if not transfer_infos:
            raise RuntimeError(
                f"No transfer info found for bootstrap_room={bootstrap_room}"
            )
        result_statuses = []
        target_infos_snapshot: Optional[List[TransferInfo]] = None
        with self.transfer_lock:
            self.update_status(bootstrap_room, KVPoll.Transferring)
            for info in transfer_infos.values():
                # 查找该 decode peer 的 KVArgsRegisterInfo
                peer_info = self.decode_kv_args_table.get(info.engine_key)
                if not peer_info:
                    self.record_failure(
                        bootstrap_room,
                        f"Peer info missing for engine {info.engine_key}",
                    )
                    raise RuntimeError(
                        f"Missing decode peer info for {info.engine_key}"
                    )
                if not info.is_dummy:
                    # 非哑请求：发起实际 RDMA 传输
                    dst_indices_chunk = info.dst_kv_indices[index_slice]
                    statuses = self.send_kvcache(
                        peer_info, kv_indices, dst_indices_chunk
                    )
                    result_statuses.extend(statuses)
                if (
                    is_last
                    and aux_index is not None
                    and info.dst_aux_index >= 0
                    and self.pp_group.is_last_rank
                    # 只有最后一块且有 aux 索引时，在最后一个 PP rank 发送 aux 数据
                ):
                    result_statuses.extend(
                        self.send_aux(
                            peer_info, aux_index, info.dst_aux_index, bootstrap_room
                        )
                    )
            if is_last:
                # 最后一块：更新状态，记录 target_infos 用于发送完成通知
                self.update_status(bootstrap_room, KVPoll.Success)
                target_infos_snapshot = list(transfer_infos.values())
                # 清理已完成的 transfer_infos（释放内存）
                self.transfer_infos.pop(bootstrap_room, None)
        return result_statuses, target_infos_snapshot


# MoriKVSender：Mori RDMA KV 发送者（prefill 侧）
# 负责调用 MoriKVManager.add_transfer_request 发起 RDMA 传输，并轮询完成状态
class MoriKVSender(CommonKVSender):
    def __init__(
        self,
        mgr: MoriKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank)
        # transfer_statuses：收集所有 RDMA 操作的 TransferStatus，用于完成/错误检测
        self.transfer_statuses: List[TransferStatus] = []
        # pending_infos：最后一块传输完成时保存的 TransferInfo 列表（用于发送完成通知）
        self.pending_infos: Optional[List[TransferInfo]] = None
        self.sent_last_chunk = False      # 是否已发送最后一块 KV
        self.conclude_state: Optional[KVPoll] = None  # 最终状态（None 表示未结束）
        self.status_notified = False      # 是否已通知 decode 侧完成/失败
        self.init_time = time.time()      # 初始化时间，用于 bootstrap 超时计算

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List[int]] = None,
    ):
        # 发送一批 KV 索引对应的 KV 缓存（可能是多块之一）
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)
        is_last = self.curr_idx == self.num_kv_indices

        # Special handling for cp
        # CP 并行特殊处理：按 CP rank 过滤 KV 索引（每个 CP rank 只发送自己负责的页）
        if self.kv_mgr.enable_all_cp_ranks_for_transfer:
            kv_indices, index_slice = filter_kv_indices_for_cp_rank(
                self.kv_mgr,
                kv_indices,
                index_slice,
            )
        elif self.kv_mgr.is_dummy_cp_rank:
            # 哑 CP rank：直接跳过，最后一块时设置成功状态
            if not is_last:
                return
            else:
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Success)
                return
        statuses, infos = self.kv_mgr.add_transfer_request(
            self.bootstrap_room,
            kv_indices,
            index_slice,
            is_last,
            aux_index=self.aux_index if is_last else None,
        )
        self.transfer_statuses.extend(statuses)
        if infos is not None:
            # 最后一块：保存 pending_infos，标记 sent_last_chunk
            self.pending_infos = infos
            self.sent_last_chunk = True

    def poll(self) -> KVPoll:
        # 轮询传输状态：检查超时、RDMA 完成情况，必要时通知 decode
        if self.conclude_state is not None:
            return self.conclude_state

        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status == KVPoll.Bootstrapping:
            # 检查 bootstrap 超时（等待 decode 发来 KV 索引）
            elapsed = time.time() - self.init_time
            if elapsed >= self.kv_mgr.bootstrap_timeout:
                reason = (
                    f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s "
                    "waiting for decode handshake"
                )
                self.kv_mgr.record_failure(self.bootstrap_room, reason)
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                self._finalize_failure(reason)
                return KVPoll.Failed
            return status

        if status == KVPoll.Failed:
            self._finalize_failure()
            return KVPoll.Failed

        # 检查所有 RDMA 传输是否完成
        transfers_done = self._all_transfers_finished()
        if transfers_done:
            if self._has_transfer_error():
                # RDMA 传输出现错误
                reason = self._collect_failure_reason()
                self.kv_mgr.record_failure(self.bootstrap_room, reason)
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                self._finalize_failure(reason)
                return KVPoll.Failed
            # 所有传输成功：通知 decode 侧
            self._notify_decode(KVPoll.Success)
            self.conclude_state = KVPoll.Success
            return KVPoll.Success
        # 传输进行中：返回 Transferring（若 mgr 已标记 Success 则仍返回 Transferring，等待 RDMA 完成）
        return KVPoll.Transferring if status == KVPoll.Success else status

    def _all_transfers_finished(self) -> bool:
        # 检查所有 RDMA 操作是否完成（最后一块已发送且无进行中的操作）
        if not self.sent_last_chunk:
            return False
        if not self.transfer_statuses:
            return True
        return all(not status.InProgress() for status in self.transfer_statuses)

    def _has_transfer_error(self) -> bool:
        # 检查是否有任意 RDMA 操作失败
        return any(status.Failed() for status in self.transfer_statuses)

    def _collect_failure_reason(self) -> str:
        # 收集第一个失败的 RDMA 操作的错误消息
        for status in self.transfer_statuses:
            if status.Failed():
                return f"KV transfer failed: {status.Message()}"
        return "KV transfer failed due to unknown reason"

    def _notify_decode(
        self, status: KVPoll, failure_reason: Optional[str] = None
    ) -> None:
        # 向 decode 侧发送传输完成/失败通知（幂等，只通知一次）
        if self.status_notified:
            return
        if self.pending_infos:
            self.kv_mgr.notify_decode_status(
                self.pending_infos, self.bootstrap_room, status, failure_reason
            )
        self.status_notified = True

    def _finalize_failure(self, failure_reason: Optional[str] = None) -> None:
        # 处理失败：通知 decode 侧并设置 conclude_state 为 Failed
        if self.conclude_state == KVPoll.Failed:
            return
        if failure_reason is None:
            failure_reason = self.kv_mgr.failure_records.get(
                self.bootstrap_room, "KV transfer failed"
            )
        self._notify_decode(KVPoll.Failed, failure_reason)
        self.conclude_state = KVPoll.Failed

    def clear(self) -> None:
        # 清理本 bootstrap_room 的状态记录（传输完成或失败后释放）
        self.kv_mgr.request_status.pop(self.bootstrap_room, None)

    def failure_exception(self):
        # 抛出异常：确保通知 decode 失败，清理状态，再 raise RuntimeError
        if self.conclude_state is None:
            self._finalize_failure()
        self.clear()
        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "KV transfer failed"
            )
        raise RuntimeError(failure_reason)

    def abort(self):
        # abort：调用父类 abort，并通知 decode 侧失败
        super().abort()
        self._notify_decode(KVPoll.Failed, "Aborted by AbortReq.")


# MoriKVReceiver：Mori RDMA 传输的 decode 侧接收器
# 职责：向 prefill 注册 KV 内存描述符，发送 KV 索引元数据，轮询传输状态
class MoriKVReceiver(CommonKVReceiver):

    def __init__(
        self,
        mgr: MoriKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        # 调用父类构造，初始化 kv_mgr、bootstrap_addr、bootstrap_room 等公共字段
        super().__init__(mgr, bootstrap_addr, bootstrap_room)
        # init_time：记录 send_metadata 发出的时间，用于等待超时检测
        self.init_time: Optional[float] = None

    def init(
        self,
        prefill_dp_rank: int,
    ):
        # 调用父类 init：加载 rank 映射、检查 staging 需求、建立 bootstrap 连接
        super().init(prefill_dp_rank)
        if self.bootstrap_room is None:
            return
        # 记录 bootstrap_room -> bootstrap_addr 映射，供 prefill 侧通过 room 查找地址
        self.kv_mgr.room_to_bootstrap_addr[self.bootstrap_room] = self.bootstrap_addr

    def _register_kv_args(self):
        # 向所有关联的 prefill rank 发送 KV 内存注册消息（ZMQ 多帧）
        if self.bootstrap_infos is None:
            return
        # 序列化本地 RDMA engine 描述符（QP 连接信息）
        engine_desc_blob = self.kv_mgr.engine_desc.pack()
        # 序列化 KV/aux/state 内存描述符列表（供 prefill 侧注册远端内存）
        packed_kv_descs = _pack_mem_desc_list(self.kv_mgr.kv_mem_descs)
        packed_aux_descs = _pack_mem_desc_list(self.kv_mgr.aux_mem_descs)
        packed_state_descs = _pack_mem_desc_list(self.kv_mgr.state_mem_descs)
        # 编码 GPU id、decode TP 大小/rank、每个 KV 项字节数
        gpu_id = str(self.kv_mgr.kv_args.gpu_id).encode("ascii")
        decode_tp_size = str(self.kv_mgr.attn_tp_size).encode("ascii")
        decode_tp_rank = str(self.kv_mgr.kv_args.engine_rank).encode("ascii")
        kv_item_len = str(self.kv_mgr.kv_args.kv_item_lens[0]).encode("ascii")

        # 逐个 prefill rank 发送注册消息（room="None" 表示仅注册，不绑定具体请求）
        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            with lock:
                sock.send_multipart(
                    [
                        MORI_GUARD,
                        # room="None"：注册消息标志，与传输消息区分
                        "None".encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        engine_desc_blob,
                        packed_kv_descs,
                        packed_aux_descs,
                        packed_state_descs,
                        gpu_id,
                        decode_tp_size,
                        decode_tp_rank,
                        kv_item_len,
                    ]
                )

    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
    ):
        # 向 prefill 侧发送传输元数据：KV 内存索引、aux 索引、room 标识等
        if self.bootstrap_infos is None or self.bootstrap_room is None:
            return

        # 将 kv_indices 序列化为 int32 字节串（空时发空字节）
        kv_indices_bytes = (
            np.asarray(kv_indices, dtype=np.int32).tobytes() if kv_indices.size else b""
        )
        # aux_index 编码为 ASCII 字节（无时发空字节）
        aux_bytes = str(aux_index).encode("ascii") if aux_index is not None else b""
        # state_indices 当前未使用，预留字段
        state_bytes = b""

        # 逐个 prefill rank 发送传输请求消息
        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            # is_dummy：哑 prefill rank，发空 KV 索引和 aux（仅用于计数凑齐）
            is_dummy = bootstrap_info.get("is_dummy", False)
            with lock:
                sock.send_multipart(
                    [
                        MORI_GUARD,
                        # room：绑定此次传输的请求 ID
                        str(self.bootstrap_room).encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        # engine_key：prefill 用于查找 decode RDMA endpoint
                        self.kv_mgr.engine_desc.key.encode("ascii"),
                        kv_indices_bytes if not is_dummy else b"",
                        aux_bytes if not is_dummy else b"",
                        state_bytes,
                        # required_dst_info_num：prefill 需等待收到的 decode 消息数
                        str(self.required_dst_info_num).encode("ascii"),
                    ]
                )
        # 记录元数据发出时间，用于 waiting_timeout 检测
        self.init_time = time.time()

    def poll(self) -> KVPoll:
        # 轮询 decode 侧传输状态：直接查 kv_mgr 状态机，超时时标记失败
        if self.conclude_state is not None:
            return self.conclude_state

        status = self.kv_mgr.check_status(self.bootstrap_room)
        # 终态（Success/Failed）：缓存并返回
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
            return status

        # WaitingForInput 状态下检查超时（等待 prefill 完成 KV 传输）
        if status == KVPoll.WaitingForInput and self.init_time is not None:
            elapsed = time.time() - self.init_time
            if elapsed >= self.kv_mgr.waiting_timeout:
                reason = f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s waiting for KV transfer"
                self.kv_mgr.record_failure(self.bootstrap_room, reason)
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                self.conclude_state = KVPoll.Failed
                return KVPoll.Failed

        return status

    def clear(self) -> None:
        # 清理本请求在 kv_mgr 中的所有状态记录（传输完成或失败后调用）
        if self.bootstrap_room is None:
            return
        # 清除请求状态、prefill 响应计数表、响应 tracker 和 room 跟踪
        self.kv_mgr.request_status.pop(self.bootstrap_room, None)
        self.kv_mgr.required_prefill_response_num_table.pop(self.bootstrap_room, None)
        self.kv_mgr.prefill_response_tracker.pop(self.bootstrap_room, None)
        self.kv_mgr._cleanup_room_tracking(self.bootstrap_room)

    def failure_exception(self):
        # 失败处理：设置终态、清理状态、弹出失败原因并 raise RuntimeError
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()
        with self.kv_mgr.failure_lock:
            # 从 failure_records 弹出原因（不存在时返回通用错误消息）
            failure_reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "KV transfer failed"
            )
        raise RuntimeError(failure_reason)

    def abort(self):
        # abort：bootstrap_room 为空时直接返回；否则调用父类 abort 并将状态置为 Failed
        if self.bootstrap_room is None:
            return
        super().abort()
        # 将 kv_mgr 状态机中对应 room 的状态强制设为 Failed，停止后续传输
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
        self.clear()


# MoriKVBootstrapServer：Mori 专用 bootstrap 服务器，完全复用 CommonKVBootstrapServer
# CommonKVBootstrapServer 已提供 prefill 注册、decode 查询、房间管理等所有功能
class MoriKVBootstrapServer(CommonKVBootstrapServer):
    pass
