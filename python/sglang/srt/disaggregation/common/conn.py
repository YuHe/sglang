# common/conn.py：PD 分离通用连接层，实现 KV Manager、Sender、Receiver 和 Bootstrap Server
# 支持多节点 TP/CP/PP/DP 组合下的 KV 缓存传输协调，通过 HTTP+ZMQ 完成节点发现和 KV 元数据交换
from __future__ import annotations

import asyncio
import dataclasses
import logging
import threading
import time
from collections import defaultdict
from functools import cache
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import numpy.typing as npt
import requests
import zmq
from aiohttp import web

from sglang.srt.disaggregation.base.conn import (
    BaseKVBootstrapServer,
    BaseKVManager,
    BaseKVReceiver,
    BaseKVSender,
    KVArgs,
    KVPoll,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed import get_pp_group
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    get_attention_cp_rank,
    get_attention_cp_size,
    get_attention_dp_rank,
    get_attention_dp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.network import (
    NetworkAddress,
    get_local_ip_auto,
    get_zmq_socket_on_host,
)

logger = logging.getLogger(__name__)


# PrefillServerInfo：从 bootstrap server 获取的 prefill 节点拓扑信息
# 包含 TP/CP/DP/PP 并行度，以及 decode 侧预计算的 rank 映射（由 try_ensure_parallel_info 填充）
@dataclasses.dataclass
class PrefillServerInfo:
    # Topology fields (fetched from bootstrap server)
    # 拓扑字段（从 bootstrap server 获取）
    attn_tp_size: int       # prefill 侧 attention TP 并行度
    attn_cp_size: int       # prefill 侧 context 并行度（CP）
    dp_size: int            # prefill 侧 DP 并行度
    pp_size: int            # prefill 侧 Pipeline 并行度
    page_size: Optional[int]          # KV 缓存页大小（token 数）
    kv_cache_dtype: Optional[str]     # KV 缓存数据类型
    follow_bootstrap_room: bool       # 是否使用 follow_bootstrap_room 负载均衡策略

    # Pre-computed rank mapping (set by try_ensure_parallel_info on decode side)
    # 预计算的 rank 映射（decode 侧 try_ensure_parallel_info 填充）
    target_tp_rank: Optional[int] = None          # 目标 prefill TP rank
    target_tp_ranks: Optional[List[int]] = None   # 目标 prefill TP rank 列表（多 rank 时）
    target_cp_ranks: Optional[List[int]] = None   # 目标 prefill CP rank 列表
    target_pp_ranks: Optional[List[int]] = None   # 目标 prefill PP rank 列表
    required_dst_info_num: Optional[int] = None           # decode 侧需要提供的目标信息数
    required_prefill_response_num: Optional[int] = None   # 需要等待的 prefill 响应数

    def __post_init__(self):
        # 类型强制转换，确保字段为正确类型
        self.attn_tp_size = int(self.attn_tp_size)
        self.attn_cp_size = int(self.attn_cp_size)
        self.dp_size = int(self.dp_size)
        self.pp_size = int(self.pp_size)
        self.page_size = int(self.page_size) if self.page_size is not None else None
        self.kv_cache_dtype = (
            str(self.kv_cache_dtype) if self.kv_cache_dtype is not None else None
        )
        self.follow_bootstrap_room = bool(self.follow_bootstrap_room)


# PrefillRankInfo：单个 prefill rank 的 IP 和 ZMQ 端口（从 bootstrap server 查询）
@dataclasses.dataclass
class PrefillRankInfo:
    rank_ip: str    # prefill rank 的 IP 地址
    rank_port: int  # prefill rank 的 ZMQ PULL socket 端口

    def __post_init__(self):
        self.rank_ip = str(self.rank_ip)
        self.rank_port = int(self.rank_port)


# CommonKVManager：通用 KV 管理器基类，维护传输状态和 bootstrap 连接
# 根据 disaggregation_mode 分别初始化 prefill 侧（注册到 bootstrap）或 decode 侧（发现 prefill）的状态
class CommonKVManager(BaseKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        self.kv_args = args
        self.is_mla_backend = is_mla_backend
        self.disaggregation_mode = disaggregation_mode
        self.server_args = server_args
        # for p/d multi node infer
        # bootstrap server 的地址和端口（用于 prefill 注册 / decode 发现）
        self.bootstrap_host = server_args.host
        self.bootstrap_port = server_args.disaggregation_bootstrap_port
        self.dist_init_addr = server_args.dist_init_addr
        # 获取当前节点的 attention TP/CP/DP 并行信息
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
        self.attn_cp_size = get_attention_cp_size()
        self.attn_cp_rank = get_attention_cp_rank()
        self.attn_dp_size = get_attention_dp_size()
        self.attn_dp_rank = get_attention_dp_rank()
        # system_dp_size：启用 dp_attention 时为 1（DP 折叠到 TP），否则为 dp_size
        self.system_dp_size = (
            1 if server_args.enable_dp_attention else server_args.dp_size
        )
        self.system_dp_rank = (
            self.kv_args.system_dp_rank if self.kv_args.system_dp_rank else 0
        )
        self.pp_size = server_args.pp_size
        self.pp_rank = self.kv_args.pp_rank
        # 获取本机 IP（自动选择合适的网卡）
        self.local_ip = get_local_ip_auto()
        # 是否允许所有 CP rank 参与 KV 传输（环境变量控制）
        self.enable_all_cp_ranks_for_transfer = (
            envs.SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER.get()
        )

        # bind zmq socket
        # 绑定 ZMQ PULL socket，用于接收来自 decode 侧的 KV 索引（prefill 模式）
        context = zmq.Context()
        self.rank_port, self.server_socket = get_zmq_socket_on_host(
            context, zmq.PULL, host=self.local_ip
        )
        logger.debug(f"kv manager bind to {self.local_ip}:{self.rank_port}")

        # bootstrap_room -> KVPoll 状态映射（WaitingForInput / Transferring / Success / Failed）
        self.request_status: Dict[int, KVPoll] = {}
        # bootstrap_room -> 失败原因字符串（用于错误报告）
        self.failure_records: Dict[int, str] = {}
        self.failure_lock = threading.Lock()

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # When SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER is True, all CP ranks
            # participate in KV transfer; Otherwise only CP rank 0 sends.
            # 非 CP rank 0 且未开启全量 CP 传输时，该 rank 为哑节点（不参与实际 KV 发送）
            self.is_dummy_cp_rank = (
                not self.enable_all_cp_ranks_for_transfer
                and self.attn_cp_size > 1
                and self.attn_cp_rank != 0
            )
            # 向 bootstrap server 注册本 prefill rank 的连接信息
            self.register_to_bootstrap()
            # transfer_infos：bootstrap_room -> 传输信息（KV indices、decode KVArgs 等）
            self.transfer_infos = {}
            # decode_kv_args_table：bootstrap_room -> decode 侧 KVArgs（用于配置 RDMA 目标）
            self.decode_kv_args_table = {}
            self.pp_group = get_pp_group()
            # If a timeout happens on the prefill side, it means prefill instances
            # fail to receive the KV indices from the decode instance of this request.
            # These timeout requests should be aborted to release the tree cache.
            # bootstrap 超时：prefill 未收到 decode 的 KV 索引，需要 abort 释放 tree cache
            self.bootstrap_timeout = envs.SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT.get()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            # enable_staging：是否开启 staging buffer 模式（异构 TP 传输）
            self.enable_staging: bool = False
            # connection_pool：bootstrap_key -> 已建立的连接信息列表（缓存避免重复查询）
            self.connection_pool: Dict[str, Dict[str, Union[str, int]]] = {}
            self.connection_lock = threading.Lock()
            # required_prefill_response_num_table：bootstrap_room -> 需要等待的 prefill 响应数
            self.required_prefill_response_num_table: Dict[int, int] = {}
            # prefill_info_table：bootstrap_addr -> PrefillServerInfo（缓存 prefill 拓扑信息）
            self.prefill_info_table: Dict[str, PrefillServerInfo] = {}
            # heartbeat_failures：bootstrap_addr -> 连续心跳失败次数
            self.heartbeat_failures: Dict[str, int] = {}
            # session_pool：使用 requests.Session 连接池，减少 HTTP 连接建立开销
            self.session_pool: Dict = defaultdict(requests.Session)
            self.session_pool_lock = threading.Lock()
            # addr_to_rooms_tracker：bootstrap_addr -> 使用该 prefill 地址的 bootstrap_room 集合
            self.addr_to_rooms_tracker: Dict[str, Set[int]] = defaultdict(set)
            # prefill_response_tracker：bootstrap_room -> 已收到响应的 prefill rank 集合
            self.prefill_response_tracker: Dict[int, Set[int]] = defaultdict(set)
            # Heartbeat interval should be at least 2 seconds
            # 心跳检测间隔（秒），最低 2 秒
            self.heartbeat_interval = max(
                envs.SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL.get(), 2.0
            )
            # Heartbeat failure should be at least 1
            # 连续心跳失败阈值（超过则认为 prefill 节点下线）
            self.max_failures = max(
                envs.SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE.get(), 1
            )
            # If a timeout happens on the decode side, it means decode instances
            # fail to receive the KV Cache transfer done signal after bootstrapping.
            # These timeout requests should be aborted to release the tree cache.
            # waiting 超时：decode 未收到 KV 传输完成信号，需要 abort
            self.waiting_timeout = envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.get()
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )

    def check_status(self, bootstrap_room: int) -> KVPoll:
        # 查询指定 bootstrap_room 的当前传输状态
        return self.request_status[bootstrap_room]

    def update_status(self, bootstrap_room: int, status: KVPoll):
        # 更新指定 bootstrap_room 的传输状态（Failed 优先，其他状态取 max 单调递增）
        if bootstrap_room not in self.request_status:
            self.request_status[bootstrap_room] = status
        else:
            if status == KVPoll.Failed:
                # 失败状态不可逆转
                self.request_status[bootstrap_room] = KVPoll.Failed
            else:
                # 状态只能向前推进（Bootstrapping < WaitingForInput < Transferring < Success）
                self.request_status[bootstrap_room] = max(
                    self.request_status[bootstrap_room], status
                )

    def record_failure(self, bootstrap_room: int, failure_reason: str):
        # 线程安全地记录失败原因（用于后续错误报告）
        with self.failure_lock:
            self.failure_records[bootstrap_room] = failure_reason

    def try_ensure_parallel_info(self, bootstrap_addr: str) -> bool:
        """Single non-blocking attempt to fetch and cache prefill parallel info.
        Returns True if info is available (cached or freshly fetched)."""
        # 非阻塞地获取 prefill 并行拓扑信息，缓存后返回 True；失败返回 False
        if bootstrap_addr in self.prefill_info_table:
            return True

        info: PrefillServerInfo = None
        try:
            # 通过 bootstrap server 的 /route?-1 参数获取 prefill 全局拓扑（不指定具体 rank）
            url = f"http://{bootstrap_addr}/route?prefill_dp_rank={-1}&prefill_cp_rank={-1}&target_tp_rank={-1}&target_pp_rank={-1}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                info = PrefillServerInfo(**data)
            else:
                logger.error(
                    f"Failed to get prefill server info: {response.status_code}, {response.text}"
                )
                return False
        except Exception as e:
            logger.error(f"Error fetching prefill server info from bootstrap: {e}")
            return False

        # Sanity checks
        # page_size 一致性校验（HiSparse 模式允许不同 page_size）
        if info.page_size is not None and info.page_size != self.kv_args.page_size:
            if self.server_args.enable_hisparse:
                # HiSparse: decode host pool page_size=1, prefill device pool page_size >= 1.
                # Transfer will use send_kvcache_hisparse with per-token item_lens.
                # HiSparse：decode 使用 page_size=1 的 host pool，prefill 使用 page_size>=1 的 device pool
                logger.info(
                    f"HiSparse PD transfer mode: prefill page_size={info.page_size}, "
                    f"decode host page_size={self.kv_args.page_size}"
                )
            else:
                raise RuntimeError(
                    f"Page size mismatch: prefill server has page_size={info.page_size}, "
                    f"but decode server has page_size={self.kv_args.page_size}. "
                    f"Both servers must use the same --page-size value."
                )

        # kv_cache_dtype 一致性校验
        if (
            info.kv_cache_dtype is not None
            and info.kv_cache_dtype != self.server_args.kv_cache_dtype
        ):
            raise RuntimeError(
                f"KV cache dtype mismatch: prefill server has kv_cache_dtype={info.kv_cache_dtype}, "
                f"but decode server has kv_cache_dtype={self.server_args.kv_cache_dtype}. "
                f"Both servers must use the same --kv-cache-dtype value."
            )

        # 计算并缓存 rank 映射关系
        self._resolve_rank_mapping(info)
        self.prefill_info_table[bootstrap_addr] = info
        logger.debug(f"Prefill parallel info for [{bootstrap_addr}]: {info}")
        return True

    def _resolve_rank_mapping(self, info: PrefillServerInfo) -> None:
        """Compute TP/CP/PP rank mapping and store on the PrefillServerInfo object.
        Deterministic for a given (bootstrap_addr, decode engine) pair."""
        # TP rank mapping
        # 计算 decode TP rank 对应的 prefill TP rank（处理 TP 数量不同的情形）
        if self.attn_tp_size == info.attn_tp_size:
            # TP 相同：直接一对一映射
            target_tp_rank = self.kv_args.engine_rank % self.attn_tp_size
            required_dst_info_num = 1
            required_prefill_response_num = 1
            target_tp_ranks = [target_tp_rank]
        elif self.attn_tp_size > info.attn_tp_size:
            # decode TP > prefill TP：多个 decode rank 对应同一个 prefill rank
            if not self.is_mla_backend:
                logger.warning_once(
                    "Performance is NOT guaranteed when using different TP sizes for non-MLA models. "
                )
            target_tp_rank = (self.kv_args.engine_rank % self.attn_tp_size) // (
                self.attn_tp_size // info.attn_tp_size
            )
            # 每个 prefill rank 对应多个 decode rank，需要发送给全部
            required_dst_info_num = self.attn_tp_size // info.attn_tp_size
            required_prefill_response_num = 1
            target_tp_ranks = [target_tp_rank]
        else:
            # decode TP < prefill TP：一个 decode rank 需要从多个 prefill rank 获取 KV
            if not self.is_mla_backend:
                logger.warning_once(
                    "Performance is NOT guaranteed when using different TP sizes for non-MLA models. "
                )
            # For non-MLA models, one decode rank needs to retrieve KVCache from multiple prefill ranks
            # 非 MLA：decode rank 需要从多个 prefill rank 拉取 KV（每个 prefill rank 持有部分 head）
            target_tp_ranks = list(
                range(
                    (self.kv_args.engine_rank % self.attn_tp_size)
                    * (info.attn_tp_size // self.attn_tp_size),
                    (self.kv_args.engine_rank % self.attn_tp_size + 1)
                    * (info.attn_tp_size // self.attn_tp_size),
                )
            )
            # For MLA models, we can retrieve KVCache from only one prefill rank, but we still need to maintain
            # multiple connections in the connection pool and have to send dummy requests to other prefill ranks,
            # or the KVPoll will never be set correctly
            # MLA：理论上只需一个 prefill rank，但需要向其他 rank 发送哑请求以保持状态机正确
            target_tp_rank = target_tp_ranks[0]
            required_dst_info_num = 1
            if self.is_mla_backend:
                required_prefill_response_num = 1
            else:
                # 非 MLA 需要等待所有 prefill rank 完成传输
                required_prefill_response_num = info.attn_tp_size // self.attn_tp_size

        # CP rank mapping — decode cp size should be equal to 1
        # decode 侧 CP 并行度必须为 1（CP 不在 decode 侧展开）
        assert self.attn_cp_size == 1, (
            f"Decode cp size ({self.attn_cp_size}) should be equal to 1",
        )
        if self.attn_cp_size == info.attn_cp_size:
            assert info.attn_cp_size == 1, (
                f"When prefill cp size is 1, attn cp size should be 1, but got {self.attn_cp_size}",
            )
            target_cp_ranks = [self.attn_cp_rank]
        else:
            # prefill 有多个 CP rank：默认只从 CP rank 0 获取（除非开启全量 CP 传输）
            target_cp_ranks = list(range(info.attn_cp_size))
            if not self.enable_all_cp_ranks_for_transfer:
                # Only retrieve from prefill CP rank 0 when not using all ranks
                target_cp_ranks = target_cp_ranks[:1]
                required_prefill_response_num *= 1
            else:
                # 开启全量 CP 传输时，需要等待所有 CP rank 完成
                required_prefill_response_num *= info.attn_cp_size // self.attn_cp_size

        # PP rank mapping — decode pp size should be equal to prefill pp size or 1
        # decode PP 并行度必须等于 prefill PP 或为 1
        assert self.pp_size == info.pp_size or self.pp_size == 1, (
            f"Decode pp size ({self.pp_size}) should be equal to prefill pp size ({info.pp_size}) or 1",
        )
        if info.pp_size == self.pp_size:
            # PP 相同：直接一对一映射
            target_pp_ranks = [self.pp_rank]
        else:
            # decode PP=1：需要从所有 prefill PP rank 获取 KV
            target_pp_ranks = list(range(info.pp_size))
            required_prefill_response_num *= info.pp_size // self.pp_size

        # 将计算结果存回 PrefillServerInfo
        info.target_tp_rank = target_tp_rank
        info.target_tp_ranks = target_tp_ranks
        info.target_cp_ranks = target_cp_ranks
        info.target_pp_ranks = target_pp_ranks
        info.required_dst_info_num = required_dst_info_num
        info.required_prefill_response_num = required_prefill_response_num

    def register_to_bootstrap(self):
        """Register prefill server info to bootstrap server via HTTP POST."""
        # 通过 HTTP PUT 将本 prefill rank 的连接信息注册到 bootstrap server
        if self.dist_init_addr:
            # Multi-node case: bootstrap server's host is dist_init_addr
            # 多节点：bootstrap server 地址来自 dist_init_addr（NCCL 初始化地址）
            host = NetworkAddress.parse(self.dist_init_addr).resolved().host
        else:
            # Single-node case: bootstrap server's host is the same as http server's host
            # 单节点：bootstrap server 与 HTTP server 在同一主机
            host = self.bootstrap_host

        bootstrap_na = NetworkAddress(host, self.bootstrap_port)
        url = f"{bootstrap_na.to_url()}/route"
        # 注册 payload 包含完整的并行配置和本 rank 的 ZMQ 连接端点
        payload = {
            "attn_tp_size": self.attn_tp_size,
            "attn_tp_rank": self.attn_tp_rank,
            "attn_cp_size": self.attn_cp_size,
            "attn_cp_rank": self.attn_cp_rank,
            "attn_dp_size": self.attn_dp_size,
            "attn_dp_rank": self.attn_dp_rank,
            "pp_size": self.pp_size,
            "pp_rank": self.pp_rank,
            "system_dp_size": self.system_dp_size,
            "system_dp_rank": self.system_dp_rank,
            "rank_ip": self.local_ip,
            "rank_port": self.rank_port,
            "page_size": self.kv_args.page_size,
            "kv_cache_dtype": self.server_args.kv_cache_dtype,
            "load_balance_method": self.server_args.load_balance_method,
        }

        try:
            response = requests.put(url, json=payload, timeout=5)
            if response.status_code == 200:
                logger.debug("Prefill successfully registered to bootstrap server.")
            else:
                logger.error(
                    f"Prefill instance failed to connect to bootstrap server: {response.status_code}, {response.text}"
                )
        except Exception as e:
            logger.error(
                f"Prefill instance failed to register to bootstrap server: {e}"
            )

    @cache
    def _connect(self, endpoint: str, is_ipv6: bool = False):
        # 创建到指定 endpoint 的 ZMQ PUSH socket（LRU 缓存，避免重复创建）
        socket = zmq.Context().socket(zmq.PUSH)
        if is_ipv6:
            socket.setsockopt(zmq.IPV6, 1)
        socket.connect(endpoint)
        return socket

    def get_mha_kv_ptrs_with_pp(
        self, src_kv_ptrs: List[int], dst_kv_ptrs: List[int]
    ) -> Tuple[List[int], List[int], List[int], List[int], int]:
        # 处理 PP>1 时 MHA KV 缓冲区指针的层范围切片
        # 返回：(src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage)
        start_layer = self.kv_args.prefill_start_layer
        num_kv_layers = len(src_kv_ptrs) // 2
        end_layer = start_layer + num_kv_layers
        dst_num_total_layers = len(dst_kv_ptrs) // 2
        # src_kv_ptrs 布局：[K_layer0, K_layer1, ..., V_layer0, V_layer1, ...]
        src_k_ptrs = src_kv_ptrs[:num_kv_layers]
        src_v_ptrs = src_kv_ptrs[num_kv_layers:]
        if num_kv_layers == dst_num_total_layers:
            # PP 对称：直接全量使用
            dst_k_ptrs = dst_kv_ptrs[:dst_num_total_layers]
            dst_v_ptrs = dst_kv_ptrs[dst_num_total_layers:]
        elif (
            num_kv_layers < dst_num_total_layers
            and dst_num_total_layers % num_kv_layers != 0
        ):
            # Case: Decode has draft model KV while Prefill is deployed without speculative decoding
            # dst_kv_ptrs layout: [K_main..., V_main..., draft_K..., draft_V...]
            # decode 有 speculative decoding draft 模型 KV，prefill 没有：只传 main model 部分
            multiplier_ratio = dst_num_total_layers // num_kv_layers
            dst_k_ptrs = dst_kv_ptrs[start_layer:end_layer]
            v_ptr_offset = num_kv_layers * multiplier_ratio
            dst_v_ptrs = dst_kv_ptrs[
                v_ptr_offset + start_layer : v_ptr_offset + end_layer
            ]
        else:
            # Decode pp size should be equal to prefill pp size or 1
            # 按 PP stage 层范围切片（prefill PP > decode PP 时）
            dst_k_ptrs = dst_kv_ptrs[start_layer:end_layer]
            dst_v_ptrs = dst_kv_ptrs[
                dst_num_total_layers + start_layer : dst_num_total_layers + end_layer
            ]
        layers_current_pp_stage = len(src_k_ptrs)
        return src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage

    def get_mla_kv_ptrs_with_pp(
        self, src_kv_ptrs: List[int], dst_kv_ptrs: List[int]
    ) -> Tuple[List[int], List[int], int]:
        # MLA 模式的 PP KV 指针切片（只有一组合并的 KV，无 K/V 分离）
        start_layer = self.kv_args.prefill_start_layer
        end_layer = start_layer + len(src_kv_ptrs)
        if len(src_kv_ptrs) == len(dst_kv_ptrs):
            # 层数相同：直接使用
            sliced_dst_kv_ptrs = dst_kv_ptrs
        else:
            # Decode pp size should be equal to prefill pp size or 1
            # 按层范围切片
            sliced_dst_kv_ptrs = dst_kv_ptrs[start_layer:end_layer]
        layers_current_pp_stage = len(src_kv_ptrs)
        return src_kv_ptrs, sliced_dst_kv_ptrs, layers_current_pp_stage


# CommonKVSender：通用 KV 发送者，负责 prefill 侧的 KV 传输协调
# 处理 CP 哑节点（非 CP rank 0）、DP 多节点场景下的 dp_rank 注册
class CommonKVSender(BaseKVSender):
    def __init__(
        self,
        mgr: CommonKVManager,
        bootstrap_addr: str,       # bootstrap server 地址（host:port）
        bootstrap_room: int,       # 本请求的唯一 bootstrap 房间号
        dest_tp_ranks: List[int],  # 目标 decode TP rank 列表
        pp_rank: int,              # 当前 PP rank
    ):
        self.kv_mgr = mgr
        self.bootstrap_room = bootstrap_room
        self.aux_index = None       # 辅助 KV 缓存索引（MLA aux cache）
        self.bootstrap_server_url = bootstrap_addr
        # inner state
        # curr_idx：当前发送进度（用于分批 send）
        self.curr_idx = 0
        if self.kv_mgr.is_dummy_cp_rank:
            # Non-authoritative CP ranks are dummy participants.
            # 非主 CP rank：直接设为 WaitingForInput 状态，不参与实际传输
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)
            return

        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Bootstrapping)
        if self.kv_mgr.server_args.dp_size > 1:
            # DP 多节点时需要注册本 prefill rank 的 dp_rank，decode 侧通过此找到正确节点
            if self.kv_mgr.server_args.load_balance_method != "follow_bootstrap_room":
                self._register_prefill_dp_rank()
            elif (
                self.kv_mgr.attn_dp_rank
                != self.bootstrap_room % self.kv_mgr.server_args.dp_size
            ):
                # follow_bootstrap_room was overridden by external routed_dp_rank
                # follow_bootstrap_room 策略被外部 routed_dp_rank 覆盖，检查是否允许混合路由
                if envs.SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK.get():
                    self._register_prefill_dp_rank()
                else:
                    self.kv_mgr.record_failure(
                        self.bootstrap_room,
                        f"follow_bootstrap_room conflict: dispatched to dp_rank "
                        f"{self.kv_mgr.attn_dp_rank} but bootstrap_room "
                        f"{self.bootstrap_room} implies dp_rank "
                        f"{self.bootstrap_room % self.kv_mgr.server_args.dp_size}. "
                        f"Set SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK=1 "
                        f"to allow mixed routing.",
                    )
                    self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                    return

    def _register_prefill_dp_rank(self):
        """Register this request's prefill dp_rank to the bootstrap server."""
        # 向 bootstrap server 注册本请求对应的 dp_rank，供 decode 侧查询
        url = f"http://{self.bootstrap_server_url}/register_dp_rank"
        payload = {
            "bootstrap_room": self.bootstrap_room,
            "dp_rank": self.kv_mgr.attn_dp_rank,
        }
        try:
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code != 200:
                logger.error(
                    f"Failed to register prefill dp_rank: {response.status_code}, {response.text}"
                )
        except Exception as e:
            logger.error(f"Failed to register prefill dp_rank: {e}")

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        # 初始化发送者状态：记录 KV 索引数量和辅助索引
        self.num_kv_indices = num_kv_indices
        self.aux_index = aux_index
        logger.debug(
            f"CommonKVSender init with num_kv_indices: {num_kv_indices} and aux_index: {aux_index}"
        )

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List[int]] = None,
    ):
        # 子类实现具体 KV 发送逻辑（Mooncake/NIXL/Mori 各自实现）
        pass

    def poll(self) -> KVPoll:
        # 子类实现 poll 状态查询
        pass

    def failure_exception(self):
        raise Exception("Fake KVReceiver Exception")

    def abort(self):
        # 主动 abort：记录失败原因，设置 conclude_state 为 Failed
        self.kv_mgr.record_failure(
            self.bootstrap_room,
            "Aborted by AbortReq.",
        )
        # Explicitly set the status to failure since this request has been aborted
        # 显式设置状态为 Failed，确保下游状态机感知到 abort
        self.conclude_state = KVPoll.Failed


# CommonKVReceiver：通用 KV 接收者，负责 decode 侧的 KV 接收协调
# 维护类级别的 ZMQ socket 缓存（_socket_cache），避免重复建立到同一 prefill 节点的连接
class CommonKVReceiver(BaseKVReceiver):
    # 类级别共享 ZMQ Context，所有 CommonKVReceiver 实例共用
    _ctx = zmq.Context()
    # endpoint -> ZMQ PUSH socket 的全局缓存（key: tcp://host:port）
    _socket_cache = {}
    # endpoint -> threading.Lock（每个 socket 一把锁，保证并发安全）
    _socket_locks = {}
    # 全局锁：保护 socket_cache 和 socket_locks 的创建操作
    _global_lock = threading.Lock()

    def __init__(
        self,
        mgr: CommonKVManager,
        bootstrap_addr: str,            # prefill bootstrap server 地址
        bootstrap_room: Optional[int] = None,  # 本请求的 bootstrap 房间号
    ):
        self.bootstrap_room = bootstrap_room
        self.bootstrap_addr = bootstrap_addr
        self.kv_mgr = mgr
        # conclude_state：传输最终状态（None 表示未结束）
        self.conclude_state: Optional[KVPoll] = None
        # require_staging：是否需要 staging buffer（prefill/decode TP 不同时为 True）
        self.require_staging: bool = False
        # 记录该 bootstrap_addr 下的所有 bootstrap_room，用于心跳监控批量处理
        self.kv_mgr.addr_to_rooms_tracker[self.bootstrap_addr].add(self.bootstrap_room)
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Bootstrapping)

    def init(self, prefill_dp_rank: int):
        # 初始化 receiver：从 prefill_info_table 中加载 rank 映射，并向所有目标 prefill rank 建立连接
        if self.bootstrap_addr not in self.kv_mgr.prefill_info_table:
            # prefill 节点已下线或未就绪，记录失败
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                f"Prefill server with bootstrap_addr: {self.bootstrap_addr} is healthy before, but now it is down. Request (bootstrap_room: {self.bootstrap_room}) has been marked as failed.",
            )
            self.conclude_state = KVPoll.Failed
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return

        # Read pre-computed rank mapping from prefill_info (computed in try_ensure_parallel_info)
        # 从缓存的 prefill_info 中读取预计算的 rank 映射
        self.prefill_info = self.kv_mgr.prefill_info_table[self.bootstrap_addr]
        self.target_tp_rank = self.prefill_info.target_tp_rank
        self.target_tp_ranks = self.prefill_info.target_tp_ranks
        self.target_cp_ranks = self.prefill_info.target_cp_ranks
        self.target_pp_ranks = self.prefill_info.target_pp_ranks
        self.required_dst_info_num = self.prefill_info.required_dst_info_num
        self.required_prefill_response_num = (
            self.prefill_info.required_prefill_response_num
        )

        # 记录本 bootstrap_room 需要等待的 prefill 响应总数
        self.kv_mgr.required_prefill_response_num_table[self.bootstrap_room] = (
            self.required_prefill_response_num
        )

        # 检查是否需要 staging buffer（异构 TP 时需要）
        if self.kv_mgr.enable_staging:
            self.require_staging = (
                self.prefill_info.attn_tp_size != 0
                and self.prefill_info.attn_tp_size != self.kv_mgr.attn_tp_size
            )

        self.prefill_dp_rank = prefill_dp_rank
        # 向所有目标 prefill rank 查询并建立 ZMQ 连接
        self._setup_bootstrap_infos()
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)

    def _setup_bootstrap_infos(self):
        # 遍历所有目标 CP rank，为每个 (bootstrap_addr, dp_rank, cp_rank, tp_rank) 组合建立连接
        all_bootstrap_infos = []
        # NOTE: key distinguished by bootstrap_addr, prefill_dp_rank, prefill_cp_rank, and target_tp_rank
        for target_cp_rank in self.target_cp_ranks:
            # connection_pool key：addr_dp_cp_tp 四元组，避免重复查询同一路径
            bootstrap_key = f"{self.bootstrap_addr}_{self.prefill_dp_rank}_{target_cp_rank}_{self.target_tp_rank}"

            if bootstrap_key not in self.kv_mgr.connection_pool:
                bootstrap_infos = []
                for target_tp_rank in self.target_tp_ranks:
                    # Enable higher PP ranks to be bootstrapped earlier to make PP PD requests bootstrap more robust
                    # 逆序遍历 PP rank（高 PP rank 优先 bootstrap，提高鲁棒性）
                    for target_pp_rank in reversed(self.target_pp_ranks):
                        bootstrap_info = self._get_bootstrap_info_from_server(
                            self.prefill_dp_rank,
                            target_cp_rank,
                            target_tp_rank,
                            target_pp_rank,
                        )
                        if bootstrap_info is not None:
                            if self.kv_mgr.is_mla_backend:
                                # For MLA: target_tp_rank is the selected real rank, others are dummy ranks
                                # MLA：只有 target_tp_rank 是实际需要的 rank，其他为哑 rank
                                bootstrap_info["is_dummy"] = not bool(
                                    target_tp_rank == self.target_tp_rank
                                    or self.target_tp_rank is None
                                )
                            else:
                                # For non-MLA: all target_tp_ranks are selected real ranks
                                # 非 MLA：所有 target_tp_ranks 都需要实际传输
                                bootstrap_info["is_dummy"] = False
                            logger.debug(
                                f"Fetched bootstrap info: {bootstrap_info} for DP {self.prefill_dp_rank} CP {target_cp_rank} TP {target_tp_rank} PP {target_pp_rank}"
                            )
                            bootstrap_infos.append(bootstrap_info)
                        else:
                            self.kv_mgr.record_failure(
                                self.bootstrap_room,
                                f"Could not fetch bootstrap info for: prefill_dp_rank: {self.prefill_dp_rank} prefill_cp_rank: {target_cp_rank} target_tp_rank: {target_tp_rank} and target_pp_rank {target_pp_rank}",
                            )
                            self.conclude_state = KVPoll.Failed
                            self.kv_mgr.update_status(
                                self.bootstrap_room, KVPoll.Failed
                            )
                            return

                self.bootstrap_infos = bootstrap_infos
                # 缓存到 connection_pool，后续相同路径的请求直接复用
                self.kv_mgr.connection_pool[bootstrap_key] = self.bootstrap_infos

                # Register kv_args only once to prefill KVManager according to the info fetched from the bootstrap server
                # 仅第一次建立连接时向 prefill 侧注册 kv_args（避免重复注册）
                self._register_kv_args()
            else:
                # 直接复用缓存的 bootstrap_infos
                self.bootstrap_infos = self.kv_mgr.connection_pool[bootstrap_key]

            assert len(self.bootstrap_infos) > 0
            all_bootstrap_infos.extend(self.bootstrap_infos)

        self.bootstrap_infos = all_bootstrap_infos

    def _get_bootstrap_info_from_server(
        self, prefill_dp_rank, prefill_cp_rank, target_tp_rank, target_pp_rank
    ):
        """Fetch the bootstrap info from the bootstrap server."""
        # 通过 HTTP GET 从 bootstrap server 获取指定 rank 组合的 prefill 连接信息
        try:
            url = f"http://{self.bootstrap_addr}/route?prefill_dp_rank={prefill_dp_rank}&prefill_cp_rank={prefill_cp_rank}&target_tp_rank={target_tp_rank}&target_pp_rank={target_pp_rank}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                bootstrap_info = response.json()
                return bootstrap_info
            else:
                logger.error(
                    f"Failed to get prefill server info: {response.status_code}, {response.text}"
                )
                return None
        except Exception as e:
            logger.error(f"Error fetching prefill info from bootstrap: {e}")
            return None

    @staticmethod
    def query_prefill_dp_ranks(
        bootstrap_addr: str, bootstrap_rooms: List[int]
    ) -> Dict[str, int]:
        """Batch query prefill dp_ranks for given bootstrap_rooms."""
        # 批量查询多个 bootstrap_room 对应的 prefill dp_rank（减少 HTTP 请求次数）
        try:
            url = f"http://{bootstrap_addr}/query_dp_ranks"
            response = requests.post(
                url,
                json={"bootstrap_rooms": bootstrap_rooms},
                timeout=5,
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(
                    f"Failed to query dp_ranks: {response.status_code}, {response.text}"
                )
                return {}
        except Exception as e:
            logger.error(f"Error querying dp_ranks from bootstrap: {e}")
            return {}

    @classmethod
    def _connect(cls, endpoint: str, is_ipv6: bool = False):
        # 类方法：线程安全地创建或复用到 endpoint 的 ZMQ PUSH socket
        with cls._global_lock:
            if endpoint not in cls._socket_cache:
                sock = cls._ctx.socket(zmq.PUSH)
                if is_ipv6:
                    sock.setsockopt(zmq.IPV6, 1)
                sock.connect(endpoint)
                # 缓存 socket 和对应的锁，供多线程安全使用
                cls._socket_cache[endpoint] = sock
                cls._socket_locks[endpoint] = threading.Lock()
            return cls._socket_cache[endpoint], cls._socket_locks[endpoint]

    @classmethod
    def _connect_to_bootstrap_server(cls, bootstrap_info: dict):
        # 根据 bootstrap_info 中的 IP/port 建立 ZMQ PUSH 连接
        ip_address = bootstrap_info["rank_ip"]
        port = bootstrap_info["rank_port"]
        na = NetworkAddress(ip_address, port)
        sock, lock = cls._connect(na.to_tcp(), is_ipv6=na.is_ipv6)
        return sock, lock

    def _register_kv_args(self):
        # 子类实现：向 prefill 侧发送 decode 的 kv_args（KV 内存布局信息）
        pass

    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
    ):
        # 子类实现：发送 KV 索引元数据到 prefill 侧（触发 KV 传输）
        raise NotImplementedError

    def failure_exception(self):
        raise Exception("Fake KVReceiver Exception")

    def abort(self):
        # 主动 abort：记录失败原因，设置状态为 Failed
        self.kv_mgr.record_failure(
            self.bootstrap_room,
            "Aborted by AbortReq.",
        )
        # Explicitly set the status to failure since this request has been aborted
        self.conclude_state = KVPoll.Failed


# CommonKVBootstrapServer：通用 KV Bootstrap HTTP 服务器
# 基于 aiohttp 实现，提供 /route（注册/查询）、/register_dp_rank、/query_dp_ranks、/health 接口
# prefill rank 启动时 PUT /route 注册自身，decode rank 通过 GET /route 查询 prefill 连接信息
class CommonKVBootstrapServer(BaseKVBootstrapServer):
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        # aiohttp Web 应用
        self.app = web.Application()
        # store：通用数据存储（子类可扩展）
        self.store = dict()
        self.lock = asyncio.Lock()
        self._setup_routes()
        # 以下字段由第一个注册的 prefill rank 填充（lazy init）
        self.pp_size = None
        self.attn_tp_size = None
        self.attn_cp_size = None
        self.dp_size = None
        self.page_size = None
        self.kv_cache_dtype: Optional[str] = None
        self.follow_bootstrap_room: Optional[bool] = None
        # prefill_port_table：dp_group -> cp_rank -> tp_rank -> pp_rank -> PrefillRankInfo
        # 四级嵌套字典，按并行维度索引 prefill rank 的 IP/port
        self.prefill_port_table: Dict[
            int, Dict[int, Dict[int, Dict[int, PrefillRankInfo]]]
        ] = {}
        # room_to_dp_rank：bootstrap_room -> {dp_rank, timestamp}，用于 DP 路由查询
        self.room_to_dp_rank: Dict[int, Dict[str, Union[int, float]]] = {}
        self._registered_count = 0   # 已注册的 prefill rank 数量
        # 过期条目清理间隔（秒），避免 room_to_dp_rank 无限增长
        self.entry_cleanup_interval = (
            envs.SGLANG_DISAGGREGATION_BOOTSTRAP_ENTRY_CLEANUP_INTERVAL.get()
        )

        # Start bootstrap server
        # 在独立 daemon 线程中运行 aiohttp 事件循环
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.run()

    def run(self):
        # 启动 bootstrap server 线程
        self.thread.start()

    def _is_ready(self) -> bool:
        # 判断所有 prefill rank 是否已注册完成（期望数 = dp * cp * tp * pp）
        if (
            self.attn_tp_size is None
            or self.attn_cp_size is None
            or self.pp_size is None
            or self.dp_size is None
        ):
            return False
        expected = self.dp_size * self.attn_cp_size * self.attn_tp_size * self.pp_size
        logger.debug(
            f"Expected {expected} prefill servers to be registered, {self._registered_count} registered so far"
        )
        return self._registered_count >= expected

    def _setup_routes(self):
        # 注册 HTTP 路由：/route（GET/PUT）、/register_dp_rank（POST）、/query_dp_ranks（POST）、/health（GET）
        self.app.router.add_route("*", "/route", self._handle_route)
        self.app.router.add_post("/register_dp_rank", self._handle_register_dp_rank)
        self.app.router.add_post("/query_dp_ranks", self._handle_query_dp_ranks)
        self.app.router.add_get("/health", self._handle_health_check)

    async def _handle_health_check(self, request):
        # 健康检查接口：始终返回 200 OK
        return web.Response(text="OK", status=200)

    async def _handle_route(self, request: web.Request):
        # /route 路由分发：PUT 用于注册，GET 用于查询
        method = request.method
        if method == "PUT":
            return await self._handle_route_put(request)
        elif method == "GET":
            return await self._handle_route_get(request)
        else:
            return web.Response(
                text="Method not allowed", status=405, content_type="application/json"
            )

    async def _handle_route_put(self, request: web.Request):
        # prefill rank 注册：解析 payload，更新 prefill_port_table
        data = await request.json()
        attn_tp_size = data["attn_tp_size"]
        attn_tp_rank = data["attn_tp_rank"]
        attn_cp_size = data["attn_cp_size"]
        attn_cp_rank = data["attn_cp_rank"]
        attn_dp_size = data["attn_dp_size"]
        attn_dp_rank = data["attn_dp_rank"]
        pp_size = data["pp_size"]
        pp_rank = data["pp_rank"]
        system_dp_size = data["system_dp_size"]
        system_dp_rank = data["system_dp_rank"]
        rank_ip = data["rank_ip"]
        rank_port = int(data["rank_port"])
        page_size = int(data["page_size"])
        kv_cache_dtype = data["kv_cache_dtype"]

        # 以第一个注册 rank 的值初始化全局拓扑参数（后续注册 rank 应与之一致）
        if self.attn_tp_size is None:
            self.attn_tp_size = attn_tp_size

        if self.attn_cp_size is None:
            self.attn_cp_size = attn_cp_size

        # dp_size：system DP 模式使用 system_dp_size，否则使用 attention dp_size
        if self.dp_size is None:
            self.dp_size = attn_dp_size if system_dp_size == 1 else system_dp_size

        if self.pp_size is None:
            self.pp_size = pp_size

        if self.page_size is None and page_size is not None:
            self.page_size = page_size

        if self.kv_cache_dtype is None and kv_cache_dtype is not None:
            self.kv_cache_dtype = kv_cache_dtype

        # 从 load_balance_method 推断 follow_bootstrap_room 策略
        if self.follow_bootstrap_room is None:
            load_balance_method = data.get(
                "load_balance_method", "follow_bootstrap_room"
            )
            self.follow_bootstrap_room = load_balance_method == "follow_bootstrap_room"

        # 选择 dp_group：system DP 时用 system_dp_rank，否则用 attn_dp_rank
        if system_dp_size == 1:
            dp_group = attn_dp_rank
        else:
            dp_group = system_dp_rank

        # Add lock to make sure thread-safe
        # 使用 asyncio.Lock 保证并发注册的线程安全性
        async with self.lock:
            dp_group_table = self.prefill_port_table.setdefault(dp_group, {})
            cp_group_table = dp_group_table.setdefault(attn_cp_rank, {})
            tp_group_table = cp_group_table.setdefault(attn_tp_rank, {})

            # 存储该 rank 的 IP/port 信息
            tp_group_table[pp_rank] = PrefillRankInfo(
                rank_ip=rank_ip,
                rank_port=rank_port,
            )

            self._registered_count += 1

        expected = self.dp_size * self.attn_cp_size * self.attn_tp_size * self.pp_size
        logger.debug(
            f"Register prefill bootstrap: DP{dp_group} CP{attn_cp_rank} TP{attn_tp_rank} PP{pp_rank} with rank_ip: {rank_ip} and rank_port: {rank_port}"
            f" ({self._registered_count}/{expected} registered)"
        )

        return web.Response(text="OK", status=200)

    async def _handle_route_get(self, request: web.Request):
        # decode rank 查询 prefill 连接信息：支持全量拓扑查询（-1 参数）和精确 rank 查询
        prefill_dp_rank = request.query.get("prefill_dp_rank")
        prefill_cp_rank = request.query.get("prefill_cp_rank")
        target_tp_rank = request.query.get("target_tp_rank")
        target_pp_rank = request.query.get("target_pp_rank")
        if (
            not prefill_dp_rank
            or not prefill_cp_rank
            or not target_tp_rank
            or not target_pp_rank
        ):
            return web.Response(text="Missing inputs for bootstrap server.", status=400)

        if (
            int(prefill_dp_rank) == -1
            and int(prefill_cp_rank) == -1
            and int(target_tp_rank) == -1
            and int(target_pp_rank) == -1
        ):
            # 全量查询（-1 参数）：返回 prefill 服务的整体拓扑信息（用于 try_ensure_parallel_info）
            if not self._is_ready():
                return web.Response(
                    text=f"Prefill server not fully registered yet"
                    f" ({self._registered_count} workers registered).",
                    status=503,
                )
            info = PrefillServerInfo(
                attn_tp_size=self.attn_tp_size,
                attn_cp_size=self.attn_cp_size,
                dp_size=self.dp_size,
                pp_size=self.pp_size,
                page_size=self.page_size,
                kv_cache_dtype=self.kv_cache_dtype,
                follow_bootstrap_room=(
                    self.follow_bootstrap_room
                    if self.follow_bootstrap_room is not None
                    else True
                ),
            )
            return web.json_response(dataclasses.asdict(info), status=200)

        if not self._is_ready():
            return web.Response(
                text=f"Prefill server not fully registered yet"
                f" ({self._registered_count} workers registered).",
                status=503,
            )

        # Find corresponding prefill info
        # 精确查询：按 dp_rank/cp_rank/tp_rank/pp_rank 四级索引查找 prefill rank 的 IP/port
        try:
            async with self.lock:
                bootstrap_info = self.prefill_port_table[int(prefill_dp_rank)][
                    int(prefill_cp_rank)
                ][int(target_tp_rank)][int(target_pp_rank)]
        except KeyError:
            return web.Response(
                text=f"Bootstrap info not found for dp_rank={prefill_dp_rank} cp_rank={prefill_cp_rank} "
                f"tp_rank={target_tp_rank} pp_rank={target_pp_rank}",
                status=404,
            )

        return web.json_response(dataclasses.asdict(bootstrap_info), status=200)

    async def _handle_register_dp_rank(self, request: web.Request):
        # 注册 bootstrap_room 对应的 prefill dp_rank（用于 DP 多节点路由）
        data = await request.json()
        bootstrap_room = int(data["bootstrap_room"])
        dp_rank = int(data["dp_rank"])
        async with self.lock:
            # 记录 dp_rank 和时间戳（时间戳用于过期清理）
            self.room_to_dp_rank[bootstrap_room] = {
                "dp_rank": dp_rank,
                "timestamp": time.time(),
            }
        logger.debug(f"Registered dp_rank={dp_rank} for {bootstrap_room=}")
        return web.Response(text="OK", status=200)

    async def _handle_query_dp_ranks(self, request: web.Request):
        # 批量查询多个 bootstrap_room 的 dp_rank（减少 HTTP 往返次数）
        data = await request.json()
        bootstrap_rooms = data["bootstrap_rooms"]
        result = {}
        async with self.lock:
            for room in bootstrap_rooms:
                room_int = int(room)
                if room_int in self.room_to_dp_rank:
                    result[str(room_int)] = self.room_to_dp_rank[room_int]["dp_rank"]
        return web.json_response(result, status=200)

    async def _cleanup_expired_entries(self):
        """Remove entries older than cleanup interval from room_to_dp_rank."""
        # 定期清理过期的 room_to_dp_rank 条目，防止内存泄漏
        while True:
            await asyncio.sleep(self.entry_cleanup_interval)
            current_time = time.time()
            async with self.lock:
                # 找出所有超过清理间隔的条目
                expired_keys = [
                    key
                    for key, value in self.room_to_dp_rank.items()
                    if current_time - value["timestamp"] > self.entry_cleanup_interval
                ]
                for key in expired_keys:
                    del self.room_to_dp_rank[key]
            if expired_keys:
                logger.debug(
                    f"Cleaned up {len(expired_keys)} expired entries from room_to_dp_rank"
                )

    def _run_server(self):
        # 在独立线程中运行 aiohttp 事件循环（daemon 线程，随主进程退出）
        try:
            # Event Loop
            # 创建新的事件循环（不共用主线程的事件循环）
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            # 启动后台定期清理任务
            self._loop.create_task(self._cleanup_expired_entries())

            # 仅在 DEBUG 级别时开启 aiohttp access log
            access_log = None
            if logging.getLogger(__name__).getEffectiveLevel() <= logging.DEBUG:
                access_log = self.app.logger

            self._runner = web.AppRunner(self.app, access_log=access_log)
            self._loop.run_until_complete(self._runner.setup())

            # 绑定 TCP 监听地址，开始接受 HTTP 请求
            site = web.TCPSite(self._runner, host=self.host, port=self.port)
            self._loop.run_until_complete(site.start())
            logger.info(
                f"CommonKVBootstrapServer started successfully on {self.host}:{self.port}"
            )
            # 永久运行事件循环直到 close() 被调用
            self._loop.run_forever()
        except Exception as e:
            logger.error(f"Server error: {str(e)}", exc_info=True)
        finally:
            # Cleanup
            # 清理 aiohttp runner 并关闭事件循环
            self._loop.run_until_complete(self._runner.cleanup())
            self._loop.close()

    def close(self):
        """Shutdown"""
        # 优雅关闭：停止事件循环，等待线程退出
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            logger.info("Stopping server loop...")

        if self.thread.is_alive():
            self.thread.join(timeout=2)
            logger.info("Server thread stopped")

    def poll(self) -> KVPoll: ...
