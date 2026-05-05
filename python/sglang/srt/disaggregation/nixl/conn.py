# NIXL（通用 RDMA 传输引擎）PD 分离 KV 缓存传输实现
# 依赖 ai-dynamo/nixl 库，通过 nixl_agent 提供后端无关的 RDMA/NVLink/共享内存传输
# 支持 MHA/MLA 架构、Mamba state、SWA/NSA hybrid 模型，以及异构 TP（prefill TP != decode TP）
from __future__ import annotations

import dataclasses
import logging
import struct
import threading
import time
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Set

import numpy as np
import numpy.typing as npt

# KVArgs：KV 缓存内存布局描述；KVPoll：传输状态枚举
from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
# group_concurrent_contiguous：将分散索引合并为连续块，减少 RDMA 操作数
from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    filter_kv_indices_for_cp_rank,
)
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# ZMQ 多帧消息头部守护字节，用于校验消息来源合法性
GUARD = "NixlMsgGuard".encode("ascii")


# TransferInfo：decode 侧发送给 prefill 的传输请求，包含目标内存索引和连接信息
@dataclasses.dataclass
class TransferInfo:
    """Contains indices for a transfer, sent by KVReceiver. Received by prefill bootstrap thread."""

    # bootstrap_room：唯一标识本次 prefill-decode 配对的请求 ID
    room: int
    # endpoint：decode 侧的 IP 地址
    endpoint: str
    # dst_port：decode 侧 ZMQ PULL socket 端口
    dst_port: int
    # agent_name：NIXL agent 名称，用于 RDMA 连接查找
    agent_name: str
    # dst_kv_indices：decode 侧 KV 内存池中分配的页索引（int32 数组）
    dst_kv_indices: npt.NDArray[np.int32]
    # dst_aux_index：decode 侧 aux 数据池中分配的索引
    dst_aux_index: int
    # required_dst_info_num：prefill 需收到的 decode 消息数（= TP 并行度）
    required_dst_info_num: int
    # dst_state_indices：decode 侧 state（如 Mamba）内存索引
    dst_state_indices: List[int]

    def is_dummy(self):
        # 哑传输：kv_indices 为空，表示该 decode rank 不需要接收 KV 数据
        return self.dst_kv_indices.size == 0

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        # 从 ZMQ 多帧消息解析 TransferInfo
        # Parse state_indices from msg[7] if present
        if len(msg) > 7 and msg[7] != b"":
            # 解析 state 索引（int32 字节序列 → list）
            dst_state_indices = list(np.frombuffer(msg[7], dtype=np.int32))
        else:
            dst_state_indices = []

        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            agent_name=msg[3].decode("ascii"),
            # int32 字节缓冲 → numpy 数组
            dst_kv_indices=np.frombuffer(msg[4], dtype=np.int32),
            dst_aux_index=int(msg[5].decode("ascii")),
            required_dst_info_num=int(msg[6].decode("ascii")),
            dst_state_indices=dst_state_indices,
        )


# KVArgsRegisterInfo：decode 侧向 prefill 注册的 KV/aux/state 内存基址信息（只需发一次）
@dataclasses.dataclass
class KVArgsRegisterInfo:
    """Contains base pointers and other info which only needs to be sent once by KVReceiver. Received by prefill bootstrap thread."""

    # room="None" 表示注册消息而非传输请求
    room: str
    # endpoint/dst_port：decode 侧 IP 和端口
    endpoint: str
    dst_port: int
    # agent_name：NIXL agent 名称（用于远端 peer 注册）
    agent_name: str
    # agent_metadata：NIXL agent 元数据（序列化的 QP/地址信息）
    agent_metadata: bytes
    # dst_kv_ptrs/dst_aux_ptrs/dst_state_data_ptrs：内存基址（uint64 列表）
    dst_kv_ptrs: list[int]
    dst_aux_ptrs: list[int]
    dst_state_data_ptrs: list[int]
    # gpu_id：decode 侧 GPU id
    gpu_id: int
    # decode_tp_size/decode_tp_rank：decode 侧 TP 并行配置（用于异构 TP 切片）
    decode_tp_size: int
    decode_tp_rank: int
    # dst_kv_item_len：decode 侧每个 KV 页的字节数
    dst_kv_item_len: int
    # dst_state_item_lens/dst_state_dim_per_tensor：Mamba state 维度信息
    dst_state_item_lens: list[int] = dataclasses.field(default_factory=list)
    dst_state_dim_per_tensor: list[int] = dataclasses.field(default_factory=list)

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        # 从 ZMQ 多帧消息解析 KVArgsRegisterInfo
        # Parse state_data_ptrs from msg[7] if present
        if len(msg) > 7 and msg[7] != b"":
            # 解析 state 数据指针（uint64 big-endian → list）
            dst_state_data_ptrs = list(struct.unpack(f"{len(msg[7]) // 8}Q", msg[7]))
        else:
            dst_state_data_ptrs = []

        # 解析 state item lens 和 state dim（可选，用于 Mamba state slice 传输）
        dst_state_item_lens = []
        dst_state_dim_per_tensor = []
        if len(msg) > 12 and len(msg[12]) > 0:
            dst_state_item_lens = list(struct.unpack(f"{len(msg[12]) // 4}I", msg[12]))
        if len(msg) > 13 and len(msg[13]) > 0:
            dst_state_dim_per_tensor = list(
                struct.unpack(f"{len(msg[13]) // 4}I", msg[13])
            )

        return cls(
            room=str(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            agent_name=msg[3].decode("ascii"),
            agent_metadata=msg[4],
            # 解析 KV/aux 基址（uint64 字节序列 → list）
            dst_kv_ptrs=list(struct.unpack(f"{len(msg[5]) // 8}Q", msg[5])),
            dst_aux_ptrs=list(struct.unpack(f"{len(msg[6]) // 8}Q", msg[6])),
            dst_state_data_ptrs=dst_state_data_ptrs,
            gpu_id=int(msg[8].decode("ascii")),
            decode_tp_size=int(msg[9].decode("ascii")),
            decode_tp_rank=int(msg[10].decode("ascii")),
            dst_kv_item_len=int(msg[11].decode("ascii")),
            dst_state_item_lens=dst_state_item_lens,
            dst_state_dim_per_tensor=dst_state_dim_per_tensor,
        )


# TransferStatus：decode 侧追踪 KV 传输进度的状态对象
# 使用多级 tracker（per-pp_rank 的 chunk set）确保所有 PP/TP rank 均完成
@dataclasses.dataclass
class TransferStatus:
    """Used by KV Receiver to know when a transfer is done."""

    # received_kvs_per_pp：已收到的 KV chunk 集合，按 PP rank 分组
    # KV chunks received per pp_rank: {pp_rank: set of chunk_ids}
    received_kvs_per_pp: Dict[int, Set[int]] = dataclasses.field(
        default_factory=lambda: defaultdict(set)
    )
    # expected_kvs_per_pp：各 PP rank 预期的 chunk 总数（最后一个 chunk 到达时设置）
    # Expected chunk count per pp_rank (set when is_last=True): {pp_rank: expected_count}
    expected_kvs_per_pp: Dict[int, int] = dataclasses.field(default_factory=dict)
    # num_pp_ranks_expected：参与传输的 PP rank 总数
    # Number of PP ranks expected to send data.
    num_pp_ranks_expected: Optional[int] = None
    # received_aux：是否已收到 aux 数据通知
    # Whether aux data has been received.
    received_aux: bool = False
    # received_state_per_pp：已收到 state 数据的 PP rank 集合（如 Mamba state）
    # PP ranks that have sent state data (state is layer-specific, each PP rank sends its portion).
    received_state_per_pp: Set[int] = dataclasses.field(default_factory=set)
    # expects_state：是否需要等待 state 数据（根据 state_type 设置）
    # Whether state data is expected (set based on state_type).
    expects_state: bool = False
    # is_failure：节点故障时标记传输失败
    # Mark as failed
    is_failure: bool = False

    def is_done(self):
        # 判断传输是否完成：失败时立即返回 True
        if self.is_failure:
            return True
        # 需要已设置 PP rank 数量且已收到 aux 数据
        if self.num_pp_ranks_expected is None or not self.received_aux:
            return False
        # 若需要 state 数据，检查所有 PP rank 均已发送
        # If state data is expected, check all PP ranks have sent it
        if (
            self.expects_state
            and len(self.received_state_per_pp) < self.num_pp_ranks_expected
        ):
            return False
        # 所有 PP rank 必须已上报预期 chunk 数
        # All PP ranks must have reported their expected count
        if len(self.expected_kvs_per_pp) < self.num_pp_ranks_expected:
            return False
        # 每个 PP rank 必须已收到全部 chunk
        # Each PP rank must have received all expected chunks
        for pp_rank, expected in self.expected_kvs_per_pp.items():
            if len(self.received_kvs_per_pp[pp_rank]) != expected:
                return False
        return True

    def is_failed(self):
        # 检查传输是否因故障被标记为失败
        return self.is_failure


# NixlKVManager：基于 NIXL 库的 KV 缓存传输管理器
# 继承 CommonKVManager，通过 nixl_agent 实现后端无关的 RDMA/NVLink/共享内存传输
class NixlKVManager(CommonKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        # 调用父类构造：初始化 TP/CP/DP/PP 并行配置、ZMQ socket、连接池等
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        try:
            # 动态导入 NIXL 库（需按官方文档安装）
            from nixl._api import nixl_agent, nixl_agent_config
        except ImportError as e:
            raise ImportError(
                "Please install NIXL by following the instructions at "
                "https://github.com/ai-dynamo/nixl/blob/main/README.md "
                "to run SGLang with NixlTransferEngine."
            ) from e

        # 从环境变量读取 NIXL 传输后端（如 UCX、GDR 等）
        backend = envs.SGLANG_DISAGGREGATION_NIXL_BACKEND.get()
        # prefill 侧启用多线程并发传输；decode 侧不需要
        agent_config = nixl_agent_config(
            backends=[backend],
            num_threads=(8 if disaggregation_mode == DisaggregationMode.PREFILL else 0),
        )
        # 使用 UUID 作为 agent 名称，避免多实例冲突
        self.agent = nixl_agent(str(uuid.uuid4()), agent_config)

        # 检查所请求的后端插件是否可用
        available_plugins = self.agent.get_plugin_list()
        if backend not in available_plugins:
            raise ValueError(
                f"NIXL backend '{backend}' not found. Available: {available_plugins}. "
                f"Please install the required NIXL plugin or choose from: {available_plugins}"
            )
        logger.info(f"NIXL KVManager initialized with backend: {backend}")

        # 将本地 KV/aux/state 内存注册到 NIXL agent（获取 RDMA 内存描述符）
        self.register_buffer_to_engine()

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # prefill 侧：启动 bootstrap 线程，等待 decode 侧发来 TransferInfo
            self._start_bootstrap_thread()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            # decode 侧：初始化 transfer_statuses 字典，并启动心跳检测线程
            self.transfer_statuses: Dict[int, TransferStatus] = defaultdict(
                TransferStatus
            )
            # 心跳检测：监控 prefill 节点存活，故障时标记受影响请求为 Failed
            self._start_heartbeat_checker_thread()
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )

    def _start_heartbeat_checker_thread(self):
        """
        Start the heartbeat checker thread for Decode worker.
        TODO (smor): unite nixl heartbeat checker with mooncake's.
        """
        # 心跳检测后台线程：定期向所有已连接 prefill 节点发 /health 请求
        def heartbeat_checker():
            while True:
                # 按 heartbeat_interval 间隔检查
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
                            # 心跳成功：重置失败计数
                            self.heartbeat_failures[bootstrap_addr] = 0

                        else:
                            logger.info(
                                f"Attempting to reconnect to {bootstrap_addr}..."
                            )
                            # 响应异常：递增失败计数，删除缓存 session
                            self.heartbeat_failures[bootstrap_addr] = (
                                self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                            )
                            with self.session_pool_lock:
                                if bootstrap_addr in self.session_pool:
                                    del self.session_pool[bootstrap_addr]
                    except Exception:
                        # 连接异常：递增失败计数
                        logger.info(f"Attempting to reconnect to {bootstrap_addr}...")
                        self.heartbeat_failures[bootstrap_addr] = (
                            self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                        )

                    # 失败次数超过阈值：触发节点故障处理
                    if (
                        self.heartbeat_failures.get(bootstrap_addr, 0)
                        >= self.max_failures
                    ):
                        self._handle_node_failure(bootstrap_addr)
                        with self.session_pool_lock:
                            if bootstrap_addr in self.session_pool:
                                del self.session_pool[bootstrap_addr]

        threading.Thread(target=heartbeat_checker, daemon=True).start()

    def _handle_node_failure(self, failed_bootstrap_addr):
        """Handle failure of a prefill node."""
        # 从连接池和 prefill 信息表中移除故障节点
        with self.connection_lock:
            keys_to_remove = [
                k for k in self.connection_pool if k.startswith(failed_bootstrap_addr)
            ]
            for k in keys_to_remove:
                del self.connection_pool[k]
            self.prefill_info_table.pop(failed_bootstrap_addr, None)

            # 查找受该节点影响的所有请求（通过 addr_to_rooms_tracker）
            possible_affected_rooms = self.addr_to_rooms_tracker.get(
                failed_bootstrap_addr, []
            )
            self.addr_to_rooms_tracker.pop(failed_bootstrap_addr, None)

        # Mark all pending transfers associated with the failed node as failed
        # 将所有未完成的关联请求标记为失败
        affected_rooms = []
        for room in possible_affected_rooms:
            if (
                room in self.transfer_statuses
                and not self.transfer_statuses[room].is_done()
            ):
                # 标记 TransferStatus.is_failure，is_done() 会立即返回 True
                self.transfer_statuses[room].is_failure = True
                affected_rooms.append(room)

        logger.error(
            f"Lost connection with prefill instance (bootstrap_addr: {failed_bootstrap_addr}), "
            f"{len(affected_rooms)} transfers affected"
        )
        # 将受影响的 room 状态机置为 Failed
        for room in possible_affected_rooms:
            logger.error(f"Let room {room} be failed due to prefill down")
            self.update_status(room, KVPoll.Failed)

    def register_buffer_to_engine(self):
        # 将本地 KV/aux/state GPU/CPU 内存注册到 NIXL agent
        # 注册后获得内存描述符（descs），用于后续 RDMA xfer_descs 构建
        kv_addrs = []
        for kv_data_ptr, kv_data_len in zip(
            self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
        ):
            # (ptr, len, gpu_id, "")：VRAM 类型注册
            kv_addrs.append((kv_data_ptr, kv_data_len, self.kv_args.gpu_id, ""))
        self.kv_descs = self.agent.register_memory(kv_addrs, "VRAM")
        logger.debug(f"Register kv tensors, len(kv_addr)= {len(kv_addrs)}")
        if not self.kv_descs:
            raise Exception("NIXL memory registration failed for kv tensors")
        # aux 数据存储在 CPU DRAM，gpu_id=0
        aux_addrs = []
        for aux_data_ptr, aux_data_len in zip(
            self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
        ):
            aux_addrs.append((aux_data_ptr, aux_data_len, 0, ""))
        self.aux_descs = self.agent.register_memory(aux_addrs, "DRAM")
        logger.debug(f"Register aux tensors, len(aux_addrs)= {len(aux_addrs)}")
        if not self.aux_descs:
            raise Exception("NIXL memory registration failed for aux tensors")

        # Register state/extra pool data buffers if present
        # 若存在 state 数据（Mamba/SWA/NSA），将其注册到 VRAM
        if self.kv_args.state_data_ptrs and self.kv_args.state_data_lens:
            state_addrs = []
            for state_data_ptr, state_data_len in zip(
                self.kv_args.state_data_ptrs, self.kv_args.state_data_lens
            ):
                state_addrs.append(
                    (state_data_ptr, state_data_len, self.kv_args.gpu_id, "")
                )
            self.state_descs = self.agent.register_memory(state_addrs, "VRAM")
            logger.debug(
                f"Register state tensors, len(state_addrs)= {len(state_addrs)}"
            )
            if not self.state_descs:
                raise Exception("NIXL memory registration failed for state tensors")

    def _add_remote_peer(self, decode_kv_args: KVArgsRegisterInfo):
        # 注册 decode 侧 NIXL agent 为远端 peer（幂等：已存在时跳过）
        agent_name = decode_kv_args.agent_name
        if agent_name in self.decode_kv_args_table:
            logger.info(f"Peer {agent_name} was already registered, ignoring.")
            return
        # 保存 decode KV 参数，供后续传输时查找目标内存地址
        self.decode_kv_args_table[agent_name] = decode_kv_args
        # 向 NIXL agent 注册远端 peer（建立 RDMA QP 等底层连接）
        self.agent.add_remote_agent(decode_kv_args.agent_metadata)

    def _send_kvcache_generic(
        self,
        peer_name: str,
        src_data_ptrs: list[int],
        dst_data_ptrs: list[int],
        item_lens: list[int],
        prefill_data_indices: npt.NDArray[np.int32],
        dst_data_indices: npt.NDArray[np.int32],
        dst_gpu_id: int,
        notif: str,
    ):
        """Generic KV cache transfer supporting both MHA and MLA architectures.
        Used by both send_kvcache and maybe_send_extra."""
        # 将分散的 page 索引分组为连续块，减少 RDMA 操作数量
        # group by indices
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_data_indices, dst_data_indices
        )

        logger.debug(f"sending kvcache to {peer_name} with notif {notif}")
        # 根据 MLA 或 MHA 架构获取各层 src/dst 指针
        # Make descs
        if self.is_mla_backend:
            # MLA：单个 K+V 融合缓冲区，直接获取每层的 src/dst 指针
            src_kv_ptrs, dst_kv_ptrs, layers_current_pp_stage = (
                self.get_mla_kv_ptrs_with_pp(src_data_ptrs, dst_data_ptrs)
            )
            # MLA 每层只有一个 (src, dst, item_len) 三元组
            layers_params = [
                (
                    src_kv_ptrs[layer_id],
                    dst_kv_ptrs[layer_id],
                    item_lens[layer_id],
                )
                for layer_id in range(layers_current_pp_stage)
            ]
        else:
            # MHA：K 和 V 分别存储，生成 K layers + V layers 共 2*layers 个三元组
            src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
                self.get_mha_kv_ptrs_with_pp(src_data_ptrs, dst_data_ptrs)
            )

            layers_params = [
                (
                    src_k_ptrs[layer_id],
                    dst_k_ptrs[layer_id],
                    item_lens[layer_id],
                )
                for layer_id in range(layers_current_pp_stage)
            ] + [
                (
                    src_v_ptrs[layer_id],
                    dst_v_ptrs[layer_id],
                    item_lens[layer_id],
                )
                for layer_id in range(layers_current_pp_stage)
            ]

        # 预分配 src/dst 地址和长度列表（每层一个 numpy 向量）
        src_addrs = []
        src_lens = []
        dst_addrs = []
        dst_lens = []

        # Precompute block starts/lengths to reduce Python-level loops.
        # 预计算各块的起始索引和长度（向量化，避免 Python 循环）
        prefill_starts = np.fromiter(
            (block[0] for block in prefill_kv_blocks), dtype=np.int64
        )
        dst_starts = np.fromiter((block[0] for block in dst_kv_blocks), dtype=np.int64)
        block_lens = np.fromiter(
            (len(block) for block in prefill_kv_blocks), dtype=np.int64
        )

        # 对每一层（K 或 V）计算实际字节地址
        for src_ptr, dst_ptr, item_len in layers_params:
            lengths = item_len * block_lens
            # src 地址 = 层基址 + 块起始索引 * 每项字节数
            src_addrs.append(src_ptr + prefill_starts * item_len)
            src_lens.append(lengths)
            dst_addrs.append(dst_ptr + dst_starts * item_len)
            dst_lens.append(lengths)

        def make_req_array(addr_chunks, len_chunks, gpu):
            # 将地址列表和长度列表合并为 NIXL 所需的 (addr, len, gpu_id) 矩阵
            if not addr_chunks:
                return np.empty((0, 3), dtype=np.int64)
            flat_addrs = np.concatenate(addr_chunks)
            flat_lens = np.concatenate(len_chunks)
            return np.column_stack(
                (
                    flat_addrs,
                    flat_lens,
                    np.full_like(flat_addrs, gpu),
                )
            )

        src_reqs = make_req_array(src_addrs, src_lens, self.kv_args.gpu_id)
        dst_reqs = make_req_array(dst_addrs, dst_lens, dst_gpu_id)

        logger.debug(
            f"len(src_addrs): before group: {len(prefill_data_indices)}, after group: {len(src_addrs)}"
        )
        # 通过 NIXL agent 获取传输描述符并发起 WRITE 操作
        src_descs = self.agent.get_xfer_descs(src_reqs, "VRAM")
        dst_descs = self.agent.get_xfer_descs(dst_reqs, "VRAM")
        # Transfer data
        # notif 编码为 ASCII 字节，decode 侧通过 get_new_notifs() 接收完成通知
        xfer_handle = self.agent.initialize_xfer(
            "WRITE",
            src_descs,
            dst_descs,
            peer_name,
            notif.encode("ascii"),  # type: ignore
        )
        if not xfer_handle:
            raise Exception("KVSender failed to create transfer")
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("KVSender failed to post transfer")
        # 返回 xfer_handle，调用方可通过 check_xfer_state() 轮询完成状态
        return xfer_handle

    def send_kvcache(
        self,
        peer_name: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        dst_gpu_id: int,
        notif: str,
    ):
        # 标准 KV 缓存传输（TP 相等或 MLA 架构）：直接调用 _send_kvcache_generic
        return self._send_kvcache_generic(
            peer_name=peer_name,
            src_data_ptrs=self.kv_args.kv_data_ptrs,
            dst_data_ptrs=dst_kv_ptrs,
            item_lens=self.kv_args.kv_item_lens,
            prefill_data_indices=prefill_kv_indices,
            dst_data_indices=dst_kv_indices,
            dst_gpu_id=dst_gpu_id,
            notif=notif,
        )

    def send_kvcache_slice(
        self,
        peer_name: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        dst_gpu_id: int,
        notif: str,
        prefill_tp_size: int,
        decode_tp_size: int,
        decode_tp_rank: int,
        dst_kv_item_len: int,
    ):
        # 异构 TP KV 切片传输：prefill 和 decode TP 大小不同时，按 head 维度切片发送
        # Get configuration from kv_args
        # 计算 prefill 和 decode 各自在 TP 组内的 rank
        local_tp_rank_in_group = self.kv_args.engine_rank % prefill_tp_size
        dst_tp_rank_in_group = decode_tp_rank % decode_tp_size

        # 每个 KV 页的字节数（prefill 侧）
        src_kv_item_len = self.kv_args.kv_item_lens[0]
        page_size = self.kv_args.page_size

        # Use total KV head count (not per-rank) for correct head distribution.
        # Per-rank kv_head_num is max(1, total//tp) which loses info when total < tp.
        # 使用总 KV head 数（非 per-rank）以正确处理 GQA 复制场景
        total_kv_heads = getattr(self.kv_args, "total_kv_head_num", 0)
        if total_kv_heads <= 0:
            total_kv_heads = self.kv_args.kv_head_num * prefill_tp_size

        # 每个 prefill/decode rank 拥有的 KV head 数（GQA 时 max(1, ...)）
        src_heads_per_rank = max(1, total_kv_heads // prefill_tp_size)
        dst_heads_per_rank = max(1, total_kv_heads // decode_tp_size)

        # 每个 token 中目标 head 切片的字节数
        bytes_per_head_slice_to_send = (
            dst_kv_item_len // page_size // dst_heads_per_rank
        )

        # GQA replication: how many prefill ranks share the same KV head
        # GQA 复制因子：多少个 prefill rank 共享同一 KV head
        src_replication = max(1, prefill_tp_size // total_kv_heads)

        # Determine which heads to send
        # 确定发送哪些 head（源/目标偏移）
        if prefill_tp_size > decode_tp_size:
            # Multiple prefill ranks to one decode rank
            # 多 prefill rank 汇聚到一个 decode rank：从 src head 起始发完整的 src 范围
            src_head_start_offset = 0
            num_heads_to_send = src_heads_per_rank
            unique_head_idx = local_tp_rank_in_group // src_replication
            dst_head_start_offset = (
                unique_head_idx * src_heads_per_rank
            ) % dst_heads_per_rank
        else:
            # Send KVCache from 1 prefill instance to multiple decode instances
            # 1 prefill rank 分散到多个 decode rank：从 dst head 切片偏移处取子集
            src_head_start_offset = (
                dst_tp_rank_in_group * dst_heads_per_rank
            ) % src_heads_per_rank
            num_heads_to_send = dst_heads_per_rank
            dst_head_start_offset = 0

        # 获取各层 K/V 指针（PP aware）
        src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
            self.get_mha_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
        )
        # Calculate precise byte offset and length for the sub-slice within the token
        # 计算 head 切片的精确字节偏移和长度
        src_head_slice_offset = src_head_start_offset * bytes_per_head_slice_to_send
        dst_head_slice_offset = dst_head_start_offset * bytes_per_head_slice_to_send
        heads_bytes_per_token_to_send = num_heads_to_send * bytes_per_head_slice_to_send

        # 构建 K 层和 V 层的 (src_ptr, dst_ptr) 对
        src_dst_ptr_pairs = [
            (
                src_k_ptrs[layer_id],
                dst_k_ptrs[layer_id],
            )
            for layer_id in range(layers_current_pp_stage)
        ] + [
            (
                src_v_ptrs[layer_id],
                dst_v_ptrs[layer_id],
            )
            for layer_id in range(layers_current_pp_stage)
        ]

        prefill_indices = np.asarray(prefill_kv_indices, dtype=np.int64)
        dst_indices = np.asarray(dst_kv_indices, dtype=np.int64)
        # 每个 token 在 prefill/decode 侧的字节数
        bytes_per_token_prefill = src_kv_item_len // page_size
        bytes_per_token_decode = dst_kv_item_len // page_size
        # page 内 token 偏移（0 到 page_size-1）
        token_offsets = np.arange(page_size, dtype=np.int64)

        src_addrs = []
        dst_addrs = []

        # 向量化计算每层每个 page 内所有 token 的精确 head 切片地址
        for src_ptr, dst_ptr in src_dst_ptr_pairs:
            # src 地址 = 层基址 + page_base + token_offset * bytes_per_token + head_slice_offset
            src_page_bases = src_ptr + prefill_indices * src_kv_item_len
            dst_page_bases = dst_ptr + dst_indices * dst_kv_item_len

            src_all = (
                src_page_bases[:, None]
                + token_offsets[None, :] * bytes_per_token_prefill
                + src_head_slice_offset
            ).ravel()
            dst_all = (
                dst_page_bases[:, None]
                + token_offsets[None, :] * bytes_per_token_decode
                + dst_head_slice_offset
            ).ravel()

            src_addrs.append(src_all)
            dst_addrs.append(dst_all)

        def make_req_array(addr_chunks, size, gpu):
            # 将地址列表合并为 (addr, size, gpu_id) 矩阵（每个 token/page 一行）
            if not addr_chunks:
                return np.empty((0, 3), dtype=np.int64)
            flat_addrs = np.concatenate(addr_chunks)
            return np.column_stack(
                (
                    flat_addrs,
                    np.full_like(flat_addrs, size),
                    np.full_like(flat_addrs, gpu),
                )
            )

        src_reqs = make_req_array(
            src_addrs, heads_bytes_per_token_to_send, self.kv_args.gpu_id
        )
        dst_reqs = make_req_array(dst_addrs, heads_bytes_per_token_to_send, dst_gpu_id)

        # Use NIXL agent for transfer
        # 通过 NIXL agent 发起 head 切片级别的 RDMA WRITE 操作
        src_descs = self.agent.get_xfer_descs(src_reqs, "VRAM")
        dst_descs = self.agent.get_xfer_descs(dst_reqs, "VRAM")

        xfer_handle = self.agent.initialize_xfer(
            "WRITE", src_descs, dst_descs, peer_name, notif.encode("ascii")
        )
        if not xfer_handle:
            raise Exception("Failed to create sliced KV transfer")

        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("Failed to post sliced KV transfer")

        return xfer_handle

    def send_aux(
        self,
        peer_name: str,
        prefill_aux_index: int,
        dst_aux_ptrs: list[int],
        dst_aux_index: int,
        notif: str,
    ):
        # 发送 aux 数据（first output token 元数据）到 decode 侧 CPU DRAM
        src_addrs = []
        dst_addrs = []

        prefill_aux_ptrs = self.kv_args.aux_data_ptrs
        prefill_aux_item_lens = self.kv_args.aux_item_lens

        # 遍历各 aux 缓冲区，计算精确的 src/dst 字节地址
        for i, _ in enumerate(dst_aux_ptrs):
            length = prefill_aux_item_lens[i]
            src_addr = prefill_aux_ptrs[i] + length * prefill_aux_index
            dst_addr = dst_aux_ptrs[i] + length * dst_aux_index
            # (addr, len, gpu_id=0)：DRAM 类型传输
            src_addrs.append((src_addr, length, 0))
            dst_addrs.append((dst_addr, length, 0))

        # 通过 NIXL agent 发起 DRAM WRITE 操作
        src_descs = self.agent.get_xfer_descs(src_addrs, "DRAM")
        dst_descs = self.agent.get_xfer_descs(dst_addrs, "DRAM")
        # Transfer data
        xfer_handle = self.agent.initialize_xfer(
            "WRITE",
            src_descs,
            dst_descs,
            peer_name,
            notif.encode("ascii"),  # type: ignore
        )
        if not xfer_handle:
            raise Exception("KVSender failed to create transfer")
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("KVSender failed to post transfer")
        return xfer_handle

    def _send_mamba_state(
        self,
        peer_name: str,
        prefill_state_indices: List[int],
        dst_state_data_ptrs: list[int],
        dst_state_indices: List[int],
        dst_gpu_id: int,
        notif: str,
    ):
        """Transfer Mamba states via RDMA."""
        # Mamba state 传输：state 是请求级别（单个索引），非 page 级别
        assert len(prefill_state_indices) == 1, "Mamba should have single state index"
        assert len(dst_state_indices) == len(
            prefill_state_indices
        ), "State indices count mismatch between Prefill and Decode"

        src_addrs = []
        dst_addrs = []

        prefill_state_data_ptrs = self.kv_args.state_data_ptrs
        prefill_state_item_lens = self.kv_args.state_item_lens

        # 计算每个 state tensor 的精确 src/dst 字节地址
        for i, dst_state_ptr in enumerate(dst_state_data_ptrs):
            length = prefill_state_item_lens[i]
            src_addr = prefill_state_data_ptrs[i] + length * int(
                prefill_state_indices[0]
            )
            dst_addr = dst_state_ptr + length * int(dst_state_indices[0])
            # state 存储在 VRAM（GPU），携带 gpu_id
            src_addrs.append((src_addr, length, self.kv_args.gpu_id))
            dst_addrs.append((dst_addr, length, dst_gpu_id))

        # 通过 NIXL agent 发起 VRAM WRITE 操作
        src_descs = self.agent.get_xfer_descs(src_addrs, "VRAM")
        dst_descs = self.agent.get_xfer_descs(dst_addrs, "VRAM")

        xfer_handle = self.agent.initialize_xfer(
            "WRITE",
            src_descs,
            dst_descs,
            peer_name,
            notif.encode("ascii"),
        )
        if not xfer_handle:
            raise Exception("Failed to create Mamba state transfer")
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("Failed to post Mamba state transfer")
        return xfer_handle

    def _send_mamba_state_slice(
        self,
        peer_name: str,
        prefill_state_indices: List[int],
        dst_state_data_ptrs: list[int],
        dst_state_indices: List[int],
        dst_gpu_id: int,
        notif: str,
        dst_state_item_lens: list[int],
        dst_state_dim_per_tensor: list[int],
        decode_tp_size: int,
        decode_tp_rank: int,
    ):
        """Transfer Mamba states with TP slice support via RDMA.

        When prefill and decode have different attn_tp_size, we slice the
        TP-sharded dimension (3rd dim) of conv_state and temporal_state
        accordingly, mirroring Mooncake's _send_mamba_state_slice.
        """
        # 异构 TP Mamba state 切片传输：按 TP sharded 维度切片（第 3 维）
        logger.warning_once(
            "Using Mamba state slice transfer for different TP sizes. "
            f"Prefill attn_tp_size={self.attn_tp_size}, "
            f"Decode attn_tp_size={decode_tp_size}."
        )
        assert len(prefill_state_indices) == 1, "Mamba should have single state index"

        prefill_state_data_ptrs = self.kv_args.state_data_ptrs
        prefill_state_item_lens = self.kv_args.state_item_lens
        # 获取 prefill 侧每个 tensor 的 TP-sharded 维度大小
        src_state_dim_per_tensor = getattr(self.kv_args, "state_dim_per_tensor", [])

        # 若无维度信息，降级为整体传输
        if not src_state_dim_per_tensor or not dst_state_dim_per_tensor:
            return self._send_mamba_state(
                peer_name,
                prefill_state_indices,
                dst_state_data_ptrs,
                dst_state_indices,
                dst_gpu_id,
                notif,
            )

        # 计算 TP 组内的本地 rank（prefill 和 decode 各自的）
        local_tp_rank_in_group = self.kv_args.engine_rank % self.attn_tp_size
        dst_tp_rank_in_group = decode_tp_rank % decode_tp_size

        src_addrs = []
        dst_addrs = []

        # 逐个 state tensor 计算切片范围
        for i, dst_state_ptr in enumerate(dst_state_data_ptrs):
            src_item_len = prefill_state_item_lens[i]
            dst_item_len = dst_state_item_lens[i]
            # src_dim/dst_dim：该 tensor 在 TP 分片维度上的大小
            src_dim = src_state_dim_per_tensor[i]
            dst_dim = dst_state_dim_per_tensor[i]

            # 每个分片维度元素的字节数
            src_bytes_per_dim = src_item_len // src_dim
            dst_bytes_per_dim = dst_item_len // dst_dim

            if self.attn_tp_size > decode_tp_size:
                # prefill 更多 rank：每个 prefill rank 向目标写一小块
                src_dim_start = 0
                num_dims_to_send = src_dim
                writers_per_decode = self.attn_tp_size // decode_tp_size
                local_writer_idx = local_tp_rank_in_group % writers_per_decode
                dst_dim_start = local_writer_idx * src_dim
            else:
                # decode 更多 rank：每个 prefill rank 只给特定 decode rank 发一切片
                src_dim_start = (dst_tp_rank_in_group * dst_dim) % src_dim
                num_dims_to_send = dst_dim
                dst_dim_start = 0

            # 计算切片的字节偏移和长度
            src_dim_offset = src_dim_start * src_bytes_per_dim
            dst_dim_offset = dst_dim_start * dst_bytes_per_dim
            bytes_to_send = num_dims_to_send * src_bytes_per_dim

            src_addr = (
                prefill_state_data_ptrs[i]
                + src_item_len * int(prefill_state_indices[0])
                + src_dim_offset
            )
            dst_addr = (
                dst_state_ptr
                + dst_item_len * int(dst_state_indices[0])
                + dst_dim_offset
            )
            src_addrs.append((src_addr, bytes_to_send, self.kv_args.gpu_id))
            dst_addrs.append((dst_addr, bytes_to_send, dst_gpu_id))

        # 通过 NIXL agent 发起切片级别的 VRAM WRITE 操作
        src_descs = self.agent.get_xfer_descs(src_addrs, "VRAM")
        dst_descs = self.agent.get_xfer_descs(dst_addrs, "VRAM")

        xfer_handle = self.agent.initialize_xfer(
            "WRITE",
            src_descs,
            dst_descs,
            peer_name,
            notif.encode("ascii"),
        )
        if not xfer_handle:
            raise Exception("Failed to create Mamba state slice transfer")
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("Failed to post Mamba state slice transfer")
        return xfer_handle

    def maybe_send_extra(
        self,
        peer_name: str,
        prefill_state_indices: List[int],
        dst_state_data_ptrs: list[int],
        dst_state_indices: List[int],
        dst_gpu_id: int,
        notif: str,
        decode_tp_size: int,
        decode_tp_rank: int = 0,
        dst_state_item_lens: list[int] | None = None,
        dst_state_dim_per_tensor: list[int] | None = None,
    ):
        """Send state or extra pool data with type-specific handling."""
        # 根据 state_type 分发到不同的传输函数
        state_type = getattr(self.kv_args, "state_type", "none")

        if state_type == "mamba":
            # Mamba 模型：需考虑异构 TP 切片
            if self.attn_tp_size != decode_tp_size:
                return self._send_mamba_state_slice(
                    peer_name,
                    prefill_state_indices,
                    dst_state_data_ptrs,
                    dst_state_indices,
                    dst_gpu_id,
                    notif,
                    dst_state_item_lens or [],
                    dst_state_dim_per_tensor or [],
                    decode_tp_size,
                    decode_tp_rank,
                )
            # TP 相等时直接整体传输
            return self._send_mamba_state(
                peer_name,
                prefill_state_indices,
                dst_state_data_ptrs,
                dst_state_indices,
                dst_gpu_id,
                notif,
            )
        elif state_type in ["swa", "nsa"]:
            # SWA/NSA hybrid 模型：state 数据复用 _send_kvcache_generic（page 级别）
            if not self.is_mla_backend and self.attn_tp_size != decode_tp_size:
                raise RuntimeError(
                    f"PD Disaggregation does NOT support PD different TP sizes for non-MLA {state_type.upper()} hybrid models yet."
                )
            if len(prefill_state_indices) != len(dst_state_indices):
                raise RuntimeError(
                    f"State index length mismatch: prefill={len(prefill_state_indices)}, "
                    f"dst={len(dst_state_indices)}"
                )
            return self._send_kvcache_generic(
                peer_name=peer_name,
                src_data_ptrs=self.kv_args.state_data_ptrs,
                dst_data_ptrs=dst_state_data_ptrs,
                item_lens=self.kv_args.state_item_lens,
                prefill_data_indices=np.array(prefill_state_indices, dtype=np.int32),
                dst_data_indices=np.array(dst_state_indices, dtype=np.int32),
                dst_gpu_id=dst_gpu_id,
                notif=notif,
            )
        else:
            # state_type="none"：无 state 数据，跳过
            if state_type != "none":
                raise RuntimeError(
                    f"PD Disaggregation via NIXL does NOT support {state_type} hybrid models yet."
                )
            return None

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last: bool,
        chunk_id: int,
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
    ):
        # prefill 侧核心方法：为所有关联 decode rank 发起 KV 传输并返回 xfer_handle 列表
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        # 最后一块时必须同时发送 aux 数据
        assert not is_last or (is_last and aux_index is not None)

        reqs_to_be_processed = self.transfer_infos[bootstrap_room].values()
        handles = []
        for req in reqs_to_be_processed:
            assert bootstrap_room == req.room
            # 哑传输：跳过，不实际发送 KV 数据
            if req.is_dummy():
                continue

            # 从 decode 侧 KV 索引中取出本 chunk 对应的切片
            chunked_dst_kv_indice = req.dst_kv_indices[index_slice]
            assert len(chunked_dst_kv_indice) == len(kv_indices)
            assert req.agent_name in self.decode_kv_args_table

            # notif 格式："{room}_kv_{chunk_id}_{is_last}_{pp_rank}"，decode 侧通过解析追踪进度
            notif = (
                f"{req.room}_kv_{chunk_id}_{int(is_last)}_{self.kv_args.engine_rank}"
            )
            decode_tp_size = self.decode_kv_args_table[req.agent_name].decode_tp_size

            # 选择传输方式：MLA 或 TP 相等 → 整块传输；否则 → head 切片传输
            if self.is_mla_backend or (decode_tp_size == self.attn_tp_size):
                kv_xfer_handle = self.send_kvcache(
                    req.agent_name,
                    kv_indices,
                    self.decode_kv_args_table[req.agent_name].dst_kv_ptrs,
                    chunked_dst_kv_indice,
                    self.decode_kv_args_table[req.agent_name].gpu_id,
                    notif,
                )
            else:
                kv_xfer_handle = self.send_kvcache_slice(
                    req.agent_name,
                    kv_indices,
                    self.decode_kv_args_table[req.agent_name].dst_kv_ptrs,
                    chunked_dst_kv_indice,
                    self.decode_kv_args_table[req.agent_name].gpu_id,
                    notif,
                    prefill_tp_size=self.attn_tp_size,
                    decode_tp_size=decode_tp_size,
                    decode_tp_rank=self.decode_kv_args_table[
                        req.agent_name
                    ].decode_tp_rank,
                    dst_kv_item_len=self.decode_kv_args_table[
                        req.agent_name
                    ].dst_kv_item_len,
                )

            handles.append(kv_xfer_handle)
            # Only the last chunk we need to send the aux data.
            # 最后一块：同时发送 state 数据（如有）和 aux 数据
            if is_last:
                if state_indices is not None:
                    dst_info = self.decode_kv_args_table[req.agent_name]
                    state_xfer_handle = self.maybe_send_extra(
                        req.agent_name,
                        state_indices,
                        dst_info.dst_state_data_ptrs,
                        req.dst_state_indices,
                        dst_info.gpu_id,
                        # state 通知格式："{room}_state_{pp_rank}"
                        f"{req.room}_state_{self.kv_args.engine_rank}",
                        decode_tp_size,
                        decode_tp_rank=dst_info.decode_tp_rank,
                        dst_state_item_lens=dst_info.dst_state_item_lens,
                        dst_state_dim_per_tensor=dst_info.dst_state_dim_per_tensor,
                    )
                    if state_xfer_handle is not None:
                        handles.append(state_xfer_handle)

                assert aux_index is not None
                # aux 通知格式："{room}_aux"
                aux_xfer_handle = self.send_aux(
                    req.agent_name,
                    aux_index,
                    self.decode_kv_args_table[req.agent_name].dst_aux_ptrs,
                    req.dst_aux_index,
                    f"{req.room}_aux",
                )
                handles.append(aux_xfer_handle)
        # 最后一块后删除 transfer_infos 中的 room 记录，释放内存
        if is_last:
            del self.transfer_infos[bootstrap_room]
        return handles

    def update_transfer_status(self):
        # 处理 NIXL 通知：解析完成通知字符串，更新 TransferStatus
        # Process notifications from received transfers.
        notif_map = self.agent.get_new_notifs()
        for peer_name, messages in notif_map.items():
            # We could also check that self.bootstrap_info['agent_name'] matches
            # the message sender. But the bootstrap room alone should be
            # sufficient to map the status.
            for msg in messages:
                # 通知格式："{room}_{type}[_{chunk_id}_{is_last}_{pp_rank}]"
                components = msg.decode("ascii").split("_", 4)
                room = int(components[0])
                if components[1] == "kv":
                    # KV 传输通知：记录已收到的 chunk
                    chunk_id = int(components[2])
                    is_last = bool(int(components[3]))
                    pp_rank = int(components[4]) if len(components) > 4 else 0
                    # Track received chunks per pp_rank
                    self.transfer_statuses[room].received_kvs_per_pp[pp_rank].add(
                        chunk_id
                    )
                    if is_last:
                        # Record expected chunk count for this pp_rank
                        # 最后一块到达：设置该 PP rank 的预期 chunk 总数 = chunk_id + 1
                        self.transfer_statuses[room].expected_kvs_per_pp[pp_rank] = (
                            chunk_id + 1
                        )
                        # Set num_pp_ranks_expected from table (or default to 1)
                        # 从 required_prefill_response_num_table 读取 PP rank 总数
                        if self.transfer_statuses[room].num_pp_ranks_expected is None:
                            self.transfer_statuses[room].num_pp_ranks_expected = (
                                self.required_prefill_response_num_table.get(room, 1)
                            )
                elif components[1] == "aux":
                    # aux 传输通知：标记已收到 aux 数据
                    self.transfer_statuses[room].received_aux = True
                elif components[1] == "state":
                    # state 传输通知：记录该 PP rank 的 state 数据已到达
                    pp_rank = int(components[2]) if len(components) > 2 else 0
                    self.transfer_statuses[room].received_state_per_pp.add(pp_rank)

    def check_transfer_done(self, room: int):
        # 检查指定 room 的传输是否完成（调用 TransferStatus.is_done()）
        if room not in self.transfer_statuses:
            return False
        return self.transfer_statuses[room].is_done()

    def _start_bootstrap_thread(self):
        # 启动 prefill 侧 bootstrap 线程：接收 decode 发来的注册/传输消息
        def bootstrap_thread():
            """This thread recvs transfer info from the decode engine"""
            while True:
                waiting_req_bytes = self.server_socket.recv_multipart()
                logger.debug(
                    f"Received multipart with total byte size {sum(len(x) for x in waiting_req_bytes)}"
                )
                # 校验消息头部守护字节，拒绝非法流量
                assert (
                    waiting_req_bytes[0] == GUARD
                ), f"First message should be {GUARD}. Foreign traffic?"
                waiting_req_bytes = waiting_req_bytes[1:]
                room = waiting_req_bytes[0].decode("ascii")
                agent_name = waiting_req_bytes[3].decode("ascii")
                if room == "None":
                    # room="None"：KV 内存注册消息，调用 _add_remote_peer 建立 RDMA 连接
                    # Register new peer and save KV base pointers.
                    self._add_remote_peer(
                        KVArgsRegisterInfo.from_zmq(waiting_req_bytes)
                    )
                    logger.debug(f"Register KVArgs from {agent_name} successfully")
                    continue
                # 正常传输请求：解析 TransferInfo 并检查是否已收到足够数量的 decode 消息
                room = int(room)
                if room not in self.transfer_infos:
                    self.transfer_infos[room] = {}
                self.transfer_infos[room][agent_name] = TransferInfo.from_zmq(
                    waiting_req_bytes
                )
                required_dst_info_num = self.transfer_infos[room][
                    agent_name
                ].required_dst_info_num
                logger.debug(f"got info {room=} {agent_name=} {required_dst_info_num=}")
                # 收到所有期望的 decode 消息后，将 room 状态推进到 WaitingForInput
                if len(self.transfer_infos[room]) == required_dst_info_num:
                    logger.debug(f"{room=} is bootstrapped")
                    self.update_status(room, KVPoll.WaitingForInput)

        threading.Thread(target=bootstrap_thread).start()


# NixlKVSender：NIXL 传输的 prefill 侧发送器
# 通过 NixlKVManager.add_transfer_request() 发起 RDMA WRITE，轮询 xfer_handle 完成状态
class NixlKVSender(CommonKVSender):
    def __init__(
        self,
        mgr: NixlKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        # 调用父类构造：初始化 kv_mgr、bootstrap_addr/room、curr_idx、num_kv_indices 等
        super().__init__(mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank)
        # xfer_handles：存储所有已发起的 NIXL xfer handle，用于轮询完成状态
        self.xfer_handles = []
        # has_sent：最后一块是否已发送
        self.has_sent = False
        # chunk_id：单调递增的 chunk 序号，用于 notif 字符串
        self.chunk_id = 0

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List[int]] = None,
    ):
        # 计算本批 KV 在全局索引中的切片范围
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)
        # 当 curr_idx == num_kv_indices 时为最后一块
        is_last = self.curr_idx == self.num_kv_indices

        # Special handling for cp
        # CP 并行特殊处理：按 CP rank 过滤 KV 索引
        if self.kv_mgr.enable_all_cp_ranks_for_transfer:
            kv_indices, index_slice = filter_kv_indices_for_cp_rank(
                self.kv_mgr,
                kv_indices,
                index_slice,
            )
        elif self.kv_mgr.is_dummy_cp_rank:
            # 哑 CP rank：非最后一块直接跳过；最后一块时标记 Success
            if not is_last:
                return
            else:
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Success)
                return

        # 通过 kv_mgr 发起传输，收集 xfer_handle
        new_xfer_handles = self.kv_mgr.add_transfer_request(
            self.bootstrap_room,
            kv_indices,
            index_slice,
            is_last,
            self.chunk_id,
            self.aux_index,
            state_indices,
        )
        self.xfer_handles.extend(new_xfer_handles)
        self.chunk_id += 1
        if is_last:
            # 最后一块：标记已发送完毕，删除 request_status 中的 room 记录
            self.has_sent = True
            del self.kv_mgr.request_status[self.bootstrap_room]

    def poll(self) -> KVPoll:
        # 轮询传输状态：未发送完时查状态机；已发送完时查所有 xfer_handle
        if not self.has_sent:
            return self.kv_mgr.check_status(self.bootstrap_room)
        # 检查所有 xfer_handle 是否均为 DONE
        states = [self.kv_mgr.agent.check_xfer_state(x) for x in self.xfer_handles]
        if all([x == "DONE" for x in states]):
            return KVPoll.Success  # type: ignore
        # 任意 handle 出错时抛出异常
        if any([x == "ERR" for x in states]):
            raise Exception("KVSender transfer encountered an error.")
        # 传输进行中：返回 WaitingForInput
        return KVPoll.WaitingForInput  # type: ignore

    def failure_exception(self):
        # 抛出 NIXL KVSender 异常（通用错误消息）
        raise RuntimeError("NIXL KVSender Exception")


# NixlKVReceiver：NIXL 传输的 decode 侧接收器
# 职责：向 prefill 注册 KV 内存基址，发送传输元数据，轮询 TransferStatus 等待传输完成
class NixlKVReceiver(CommonKVReceiver):
    def __init__(
        self,
        mgr: NixlKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        # started_transfer：是否已调用 send_metadata 发送传输请求
        self.started_transfer = False
        # 调用父类构造：初始化 kv_mgr、bootstrap_addr/room、conclude_state 等
        super().__init__(mgr, bootstrap_addr, bootstrap_room)
        # init_time：send_metadata 发出时间，用于 waiting_timeout 检测
        self.init_time = None

    def init(
        self,
        prefill_dp_rank: int,
    ):
        # 调用父类 init：加载 rank 映射、检查 staging 需求、建立 bootstrap 连接
        super().init(prefill_dp_rank)

    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
    ):
        # 向所有关联 prefill rank 发送传输元数据（ZMQ 多帧）
        if self.bootstrap_infos is None:
            logger.error(
                f"Could not fetch prefill parallel info from bootstrap_addr: {self.bootstrap_addr}",
            )
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return

        for bootstrap_info in self.bootstrap_infos:
            logger.debug(
                f"Fetched bootstrap info: {bootstrap_info} for engine rank: {self.kv_mgr.kv_args.engine_rank}"
            )
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            # is_dummy：哑 prefill rank，发空 KV 索引（仅用于计数）
            is_dummy = bootstrap_info["is_dummy"]
            logger.debug(
                f"Sending to prefill server with bootstrap room {self.bootstrap_room} {is_dummy=}"
            )
            with lock:
                sock.send_multipart(
                    [
                        GUARD,
                        # bootstrap_room：唯一标识本次 prefill-decode 配对
                        str(self.bootstrap_room).encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        # agent_name：NIXL agent 名称，prefill 侧用于查找远端 peer
                        self.kv_mgr.agent.name.encode("ascii"),
                        kv_indices.tobytes() if not is_dummy else b"",
                        str(aux_index).encode("ascii"),
                        # required_dst_info_num：prefill 需等待收到的 decode 消息数
                        str(self.required_dst_info_num).encode("ascii"),
                        (
                            np.array(state_indices, dtype=np.int32).tobytes()
                            if not is_dummy and state_indices is not None
                            else b""
                        ),
                    ]
                )

        # Mark that we expect state data if state_indices was provided
        # 若提供了 state_indices，在 TransferStatus 中标记需要等待 state 数据
        if state_indices is not None:
            self.kv_mgr.transfer_statuses[self.bootstrap_room].expects_state = True

        # 标记已发送传输请求，记录发出时间
        self.started_transfer = True
        self.init_time = time.time()

    def poll(self) -> KVPoll:
        # 轮询传输状态：检查超时和 TransferStatus.is_done()
        if self.conclude_state is not None:
            return self.conclude_state
        status = self.kv_mgr.check_status(self.bootstrap_room)
        # 终态（Success/Failed）：缓存并返回
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
            return status
        # 未发送传输请求时直接返回当前状态（仍在 Bootstrapping 阶段）
        if not self.started_transfer:
            return status

        now = time.time()
        elapsed = now - self.init_time

        # 等待超时检查
        if elapsed >= self.kv_mgr.waiting_timeout:
            logger.error(f"Request {self.bootstrap_room} waiting_timeout")
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.WaitingForInput",
            )
            self.conclude_state = KVPoll.Failed
            return KVPoll.Failed

        # 主动拉取 NIXL 传输完成通知并更新 TransferStatus
        self.kv_mgr.update_transfer_status()
        if self.kv_mgr.check_transfer_done(self.bootstrap_room):  # type: ignore
            # 传输完成：从 addr_to_rooms_tracker 中移除 room
            self.kv_mgr.addr_to_rooms_tracker[self.bootstrap_addr].discard(
                self.bootstrap_room
            )
            # Check if the transfer failed
            # 检查是否因节点故障被标记为失败
            if self.kv_mgr.transfer_statuses[self.bootstrap_room].is_failed():
                self.conclude_state = KVPoll.Failed
                logger.error(
                    f"Transfer for room {self.bootstrap_room} failed due to node failure"
                )
            else:
                self.conclude_state = KVPoll.Success
            # 清理 TransferStatus 记录
            del self.kv_mgr.transfer_statuses[self.bootstrap_room]
            return self.conclude_state  # type: ignore
        return KVPoll.WaitingForInput  # type: ignore

    def _register_kv_args(self):
        # 向所有关联 prefill rank 发送 KV 内存注册消息（ZMQ 多帧，room="None"）
        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            # 序列化 KV/aux/state 内存基址（uint64 big-endian）
            packed_kv_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.kv_data_ptrs
            )
            packed_aux_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
            )
            packed_state_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.state_data_ptrs
            )

            # 序列化 state item lens 和 state 维度信息（uint32）
            packed_state_item_lens = b"".join(
                struct.pack("I", item_len)
                for item_len in self.kv_mgr.kv_args.state_item_lens
            )
            state_dim_per_tensor = getattr(
                self.kv_mgr.kv_args, "state_dim_per_tensor", []
            )
            packed_state_dim_per_tensor = b"".join(
                struct.pack("I", dim) for dim in state_dim_per_tensor
            )

            with lock:
                sock.send_multipart(
                    [
                        GUARD,
                        # room="None"：注册消息，与传输请求区分
                        "None".encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        # agent_name 和 agent_metadata：供 prefill 侧调用 add_remote_agent
                        self.kv_mgr.agent.name.encode("ascii"),
                        self.kv_mgr.agent.get_agent_metadata(),
                        packed_kv_data_ptrs,
                        packed_aux_data_ptrs,
                        packed_state_data_ptrs,
                        str(self.kv_mgr.kv_args.gpu_id).encode("ascii"),
                        str(self.kv_mgr.attn_tp_size).encode("ascii"),
                        str(self.kv_mgr.kv_args.engine_rank).encode("ascii"),
                        str(self.kv_mgr.kv_args.kv_item_lens[0]).encode("ascii"),
                        packed_state_item_lens,
                        packed_state_dim_per_tensor,
                    ]
                )

    def failure_exception(self):
        # 抛出 NIXL KVReceiver 异常（通用错误消息）
        raise RuntimeError("NIXL KVReceiver Exception")


# NixlKVBootstrapServer：NIXL 专用 bootstrap 服务器，完全复用 CommonKVBootstrapServer
# CommonKVBootstrapServer 提供 prefill 注册、decode 查询、房间管理等所有功能
class NixlKVBootstrapServer(CommonKVBootstrapServer):
    pass
