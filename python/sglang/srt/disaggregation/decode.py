"""
Decode 侧（PD 分离架构中的 Decode Server）请求生命周期管理模块。
请求依次经历 4 个阶段：PreallocQueue → TransferQueue → WaitingQueue → RunningBatch。

Life cycle of a request in the decode server

1. PreallocQueue:
    a. Initialize a receiver for each request
    b. The request handshakes first, and pre-allocate kv once there is available kv.
    c. Move the request to TransferQueue.

2. TransferQueue:
    a. Poll the receiver to check the transfer state
    b. If the transfer has finished, move the request to waiting queue

3. WaitingQueue:
    a. Use the requests in the queue to construct a PrebuiltExtendBatch
    b. Skip the prefill forward but only populate metadata

4. RunningBatch:
    a. Merge the resolved PrebuiltExtendBatch into running batch to run decoding
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.distributed import ProcessGroup

# Mamba2 SSM 状态缓存参数，支持混合 Mamba+Attention 架构
from sglang.srt.configs.mamba_utils import Mamba2CacheParams
# GPU 内存类型常量，用于内存分配标记
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
# KV 传输状态轮询枚举：Bootstrapping/WaitingForInput/Transferring/Success/Failed
from sglang.srt.disaggregation.base import KVPoll
# KV 传输管理器和接收器基类
from sglang.srt.disaggregation.common.conn import CommonKVManager, CommonKVReceiver
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,        # 伪 bootstrap 主机名，用于测试模式
    DisaggregationMode,         # PREFILL / DECODE 侧标识
    KVClassType,                # KV 类工厂枚举：MANAGER/RECEIVER/ARGS
    MetadataBuffers,            # 存储 Prefill 侧传回的首 token + 统计信息的共享缓冲区
    ReqToMetadataIdxAllocator,  # 元数据缓冲区索引分配器
    TransferBackend,            # 传输后端：mooncake/nixl/fake
    get_kv_class,               # 按后端和类型获取对应类
    is_mla_backend,             # 判断是否为 MLA（Multi-head Latent Attention）后端
    kv_to_page_indices,         # 将 token 索引转换为页级别索引
    poll_and_all_reduce,        # 跨 TP 各 rank all-reduce 轮询结果
    poll_and_all_reduce_with_staging,  # 带 staging buffer 的 all-reduce 轮询
    prepare_abort,              # 将请求标记为 FINISH_ABORT 状态
)
from sglang.srt.environ import envs
# 获取 Attention TP 并行度（可能小于全局 TP size）
from sglang.srt.layers.dp_attention import get_attention_tp_size
# 调度批次与完成原因枚举
from sglang.srt.managers.schedule_batch import FINISH_ABORT, ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,     # Mamba+Attention 混合 KV 池
    HybridReqToTokenPool,   # 混合模型请求到 token 池（含 Mamba 状态槽）
    KVCache,                # 标准 KV 缓存类型
    NSATokenToKVPool,       # Native Sparse Attention KV 池
    ReqToTokenPool,         # 标准请求-to-token 索引池
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool  # Sliding Window Attention KV 池
from sglang.srt.observability.req_time_stats import (
    set_schedule_time_batch,   # 记录批次调度时间戳
    set_time_batch,            # 批量设置请求时间戳
)
from sglang.srt.utils.network import NetworkAddress
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.server_args import ServerArgs

# 估算可分配 token 时裁剪 max_new_tokens 的上限，避免过于保守的内存预留
CLIP_MAX_NEW_TOKEN = envs.SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION.get()


def _is_fake_transfer(req: Req, server_args: ServerArgs) -> bool:
    """判断是否为测试用伪传输模式（不真实进行 KV 传输，用于单元测试）。"""
    return req.bootstrap_host == FAKE_BOOTSTRAP_HOST or (
        req.bootstrap_host is None
        and server_args.disaggregation_transfer_backend == "fake"
    )


def _bootstrap_addr(req: Req) -> str:
    # FIXME: make a property of a req
    # 将请求的 bootstrap_host:bootstrap_port 拼接为 "host:port" 字符串
    return NetworkAddress(req.bootstrap_host, req.bootstrap_port).to_host_port_str()


class DecodeReqToTokenPool:
    """
    Decode 侧扩展版请求-to-token 索引池。
    与标准 ReqToTokenPool 的关键区别：额外分配 pre_alloc_size 个槽位，
    使预分配（PreallocQueue）和传输中（TransferQueue）的请求不占用
    --max-running-requests 限额，从而不阻塞 Prefill 侧下发新请求。

    The difference of DecodeReqToTokenPool and ReqToTokenPool is that
    DecodeReqToTokenPool subscribes memory for pre-allocated requests.

    In ReqToTokenPool, if `--max-running-requests` is 8,
    #pre-allocated + #transfer + #running <= 8, but there are in fact more memory can carry pre-allocated requests.

    In DecodeReqToTokenPool, if `--max-running-requests` is 8,
    #running <= 8, #pre-allocated + #transfer <= pre_alloc_size, so we can use the free memory to pre-allocate requests to unblock prefill.
    """

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        pre_alloc_size: int,
    ):
        # 创建 TorchMemorySaverAdapter，启用时可将 KV 缓存 offload 到 CPU
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size               # 正常运行请求槽数（= --max-running-requests）
        self.max_context_len = max_context_len  # 每个请求最大上下文长度
        self.device = device
        self.pre_alloc_size = pre_alloc_size   # 额外预分配槽数，专用于 PreallocQueue/TransferQueue
        # req_to_token 矩阵：行=请求槽，列=各位置的 KV 索引
        with memory_saver_adapter.region(tag=GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_token = torch.zeros(
                (size + pre_alloc_size, max_context_len),
                dtype=torch.int32,
                device=device,
            )

        # free_slots: 可用槽位列表，包含普通 + 预分配槽
        self.free_slots = list(range(size + pre_alloc_size))

    def write(self, indices, values):
        """将 KV token 索引写入请求槽。"""
        self.req_to_token[indices] = values

    def available_size(self):
        """返回当前可用（空闲）槽数量。"""
        return len(self.free_slots)

    def alloc(self, reqs: List["Req"]) -> Optional[List[int]]:
        # Indices of reqs that already have a req_pool_idx and will reuse
        # their existing slot (e.g. chunked prefill continuing across chunks).
        # 找出已有 req_pool_idx 的请求（chunked prefill 续接场景，复用已有槽）
        reusing = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]
        assert (
            len(reusing) <= 1
        ), "only one chunked request may reuse req_pool_idx in a batch"
        assert all(
            reqs[i].is_chunked > 0 or reqs[i].kv_committed_len > 0 for i in reusing
        ), "reusing request must be chunked or have committed KV"

        # 计算实际需要新分配的槽数
        need_size = len(reqs) - len(reusing)
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        offset = 0
        for r in reqs:
            if r.req_pool_idx is None:
                r.req_pool_idx = select_index[offset]
                offset += 1
        return [r.req_pool_idx for r in reqs]

    def free(self, req: "Req"):
        """释放请求占用的槽位，归还到 free_slots。"""
        assert req.req_pool_idx is not None, "request must have req_pool_idx"
        self.free_slots.append(req.req_pool_idx)
        req.req_pool_idx = None

    def clear(self):
        """重置所有槽位为空闲状态（调度器重启或测试清场时使用）。"""
        self.free_slots = list(range(self.size + self.pre_alloc_size))


class HybridMambaDecodeReqToTokenPool(HybridReqToTokenPool):
    """
    Mamba+Attention 混合模型的 Decode 侧请求-to-token 池。
    在 DecodeReqToTokenPool 基础上额外管理 Mamba SSM 状态槽（含 ping-pong 缓冲区）。
    """

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        cache_params: "Mamba2CacheParams",
        mamba_layer_ids: List[int],
        speculative_num_draft_tokens: int,
        enable_mamba_extra_buffer: bool,
        pre_alloc_size: int,
        enable_overlap_schedule: bool,
        mamba_size: int = None,
        start_layer: int = None,
    ):
        # 调用 DecodeReqToTokenPool.__init__ 初始化基础请求槽（含 pre_alloc_size 扩展）
        DecodeReqToTokenPool.__init__(
            self,
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
            pre_alloc_size=pre_alloc_size,
        )

        # overlap 调度需要 2 个 ping-pong 缓冲区，单步只需 1 个
        self.mamba_ping_pong_track_buffer_size = 2 if enable_overlap_schedule else 1
        self.enable_mamba_extra_buffer = enable_mamba_extra_buffer
        self.enable_memory_saver = enable_memory_saver
        # Each request needs 1 main mamba slot + ping-pong slots when extra_buffer is enabled.
        # Cap the pool at max concurrent requests * slots_per_req to avoid allocating failed.
        # 每个请求需要：1 个主 Mamba 状态槽 + 可选 ping-pong 槽
        slots_per_req = 1 + (
            self.mamba_ping_pong_track_buffer_size if enable_mamba_extra_buffer else 0
        )
        max_slots_needed = (size + pre_alloc_size) * slots_per_req
        if mamba_size is not None:
            # 若用户指定 mamba_size 小于所需，自动扩大并打印 warning
            effective_mamba_size = max(mamba_size, max_slots_needed)
            if mamba_size < max_slots_needed:
                logger.warning(
                    "mamba_size (%d) is less than decode side's max_slots_needed (%d = %d reqs * %d slots/req), "
                    "raising effective_mamba_size to %d",
                    mamba_size,
                    max_slots_needed,
                    size + pre_alloc_size,
                    slots_per_req,
                    effective_mamba_size,
                )
        else:
            effective_mamba_size = max_slots_needed
        self.start_layer = start_layer if start_layer is not None else 0
        self.layer_transfer_counter = None
        # 初始化 Mamba 状态内存池（包括 spec decode 草稿槽）
        self._init_mamba_pool(
            size=effective_mamba_size,
            mamba_spec_state_size=size + pre_alloc_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
            enable_mamba_extra_buffer=self.enable_mamba_extra_buffer,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )

    def clear(self):
        """同时重置请求槽和 Mamba 状态池。"""
        self.free_slots = list(range(self.size + self.pre_alloc_size))
        self.mamba_pool.clear()


@dataclass
class DecodeRequest:
    """封装 Decode 侧单个请求的状态：原始请求、KV 接收器及元数据缓冲区索引。"""
    req: Req
    kv_receiver: CommonKVReceiver   # 负责与 Prefill 侧握手并接收 KV 数据
    waiting_for_input: bool = False  # True 表示握手完成，正在等待 Prefill 发送 KV
    metadata_buffer_index: int = -1  # 在 MetadataBuffers 中的槽位索引，-1 表示未分配

    @property
    def seqlen(self) -> int:
        """返回请求当前序列长度（prefill input ids 数量）。"""
        return self.req.seqlen


class DecodePreallocQueue:
    """
    预分配队列（阶段 1）：负责初始化 KV 接收器、握手并为请求预分配 KV 内存槽。
    握手成功后将请求移入 TransferQueue 等待真实 KV 数据到来。
    """

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        draft_token_to_kv_pool: Optional[KVCache],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        scheduler: Scheduler,
        transfer_queue: DecodeTransferQueue,
        tree_cache: BasePrefixCache,
        gloo_group: ProcessGroup,
        tp_rank: int,
        tp_size: int,
        dp_size: int,
        gpu_id: int,
        bootstrap_port: int,
        max_total_num_tokens: int,
        pp_rank: int,
        num_reserved_decode_tokens: int,
        transfer_backend: TransferBackend,
    ):
        # 基础内存池和分配器引用
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.token_to_kv_pool = token_to_kv_pool_allocator.get_kvcache()
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        # 判断是否为 MLA 后端（MLA 不使用 staging buffer）
        self.is_mla_backend = is_mla_backend(self.token_to_kv_pool)
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.scheduler = scheduler
        self.transfer_queue = transfer_queue
        self.tree_cache = tree_cache  # this is always a chunk cache
        # TP/DP 拓扑信息，用于计算目标 DP rank 和 RDMA 端点
        self.gloo_group = gloo_group
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.gpu_id = gpu_id
        self.bootstrap_port = bootstrap_port
        self.max_total_num_tokens = max_total_num_tokens
        self.pp_rank = pp_rank
        # 为每个运行中请求额外预留的 decode 步 token 槽，避免 OOM
        self.num_reserved_decode_tokens = num_reserved_decode_tokens
        self.transfer_backend = transfer_backend
        # Queue for requests pending pre-allocation
        # 主队列：等待握手 + 预分配 KV 的请求
        self.queue: List[DecodeRequest] = []
        # 因内存不足而被撤回的请求，等待内存恢复后重新调度
        self.retracted_queue: List[Req] = []
        # 因 Prefill DP 并行信息未就绪而暂存的请求
        self.pending_reqs: List[DecodeRequest] = []
        # 重试计数与时间戳，用于超时检测
        self._ensure_retry_count: Dict[str, int] = {}
        self._max_ensure_retries: int = 15  # scheduling cycles
        self._ensure_last_attempt_time: Dict[str, float] = {}
        self._ensure_retry_interval: float = 1.0  # seconds
        # staging buffer 异构 TP 时按序拼接 KV 分片，仅非 MLA 场景支持
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()
        if self.enable_staging and self.is_mla_backend:
            raise RuntimeError(
                "SGLANG_DISAGG_STAGING_BUFFER is designed for non-MLA models "
                "(e.g. GQA, MHA). MLA models should not set this flag."
            )
        # 初始化 CommonKVManager（含 RDMA/nixl 引擎、bootstrap 服务器等）
        self.kv_manager = self._init_kv_manager()
        if self.enable_staging:
            # 将 staging handler 注入 TransferQueue，以便后续 pop_transferred 使用
            self.transfer_queue._init_staging_handler(self.kv_manager)

        if self.scheduler.tp_worker.is_hybrid_swa:
            # FIXME: current SWA allocation allocate full kv cache size in prefill
            # SWA 模式下最大 token 数受限于 SWA 专属内存大小
            self.max_total_num_tokens = min(
                self.max_total_num_tokens,
                self.scheduler.tp_worker.model_runner.swa_max_total_num_tokens,
            )

    def _init_kv_manager(self) -> CommonKVManager:
        """
        初始化 CommonKVManager：配置 KV 缓冲区地址、元数据缓冲区、Mamba/SWA/NSA 状态缓冲区，
        并按传输后端（mooncake/nixl/fake）实例化对应的 KV 管理器。
        """
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()

        # Attention TP size 可能小于全局 TP size（MQA/GQA 下多 rank 共享 KV）
        attn_tp_size = get_attention_tp_size()
        kv_args.engine_rank = self.tp_rank % (attn_tp_size)

        kv_args.pp_rank = self.pp_rank
        kv_args.system_dp_rank = self.scheduler.dp_rank
        if self.scheduler.enable_hisparse:
            # Direct-to-host: register host pool pointers so P writes to D's host memory
            # HiSparse Direct-to-Host 路径：RDMA 直接写到 Host 内存池
            host_pool = self.scheduler.hisparse_coordinator.mem_pool_host
            kv_data_ptrs, kv_data_lens, kv_item_lens = (
                host_pool.get_contiguous_buf_infos()
            )
        else:
            # 标准路径：获取 GPU KV Cache 的连续内存块地址列表
            kv_data_ptrs, kv_data_lens, kv_item_lens = (
                self.token_to_kv_pool.get_contiguous_buf_infos()
            )
        if self.draft_token_to_kv_pool is not None:
            # We should also transfer draft model kv cache. The indices are
            # always shared with a target model.
            # Speculative Decoding 草稿模型 KV 缓冲区追加到主模型缓冲区列表
            draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = (
                self.draft_token_to_kv_pool.get_contiguous_buf_infos()
            )
            kv_data_ptrs += draft_kv_data_ptrs
            kv_data_lens += draft_kv_data_lens
            kv_item_lens += draft_kv_item_lens

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens
        # HiSparse Host pool has page_size=1; use it when hisparse is enabled
        # HiSparse 使用 page_size=1（每 token 独立寻址），普通模式用 KV Cache 自身 page size
        kv_args.page_size = (
            1 if self.scheduler.enable_hisparse else self.token_to_kv_pool.page_size
        )

        # 配置元数据缓冲区（存储首 token、cached_tokens 等 Prefill 侧反馈）
        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
            self.metadata_buffers.get_buf_infos()
        )

        if hasattr(self.token_to_kv_pool, "get_state_buf_infos"):
            # 模型有额外状态缓冲区（Mamba SSM 状态、SWA 窗口、NSA 稀疏 KV）
            state_data_ptrs, state_data_lens, state_item_lens = (
                self.token_to_kv_pool.get_state_buf_infos()
            )
            kv_args.state_data_ptrs = state_data_ptrs
            kv_args.state_data_lens = state_data_lens
            kv_args.state_item_lens = state_item_lens

            if isinstance(self.token_to_kv_pool, SWAKVPool):
                kv_args.state_type = "swa"     # Sliding Window Attention 专用状态
            elif isinstance(self.token_to_kv_pool, HybridLinearKVPool):
                kv_args.state_type = "mamba"   # Mamba SSM 状态
                # Get state dimension info for cross-TP slice transfer
                # 跨 TP rank 切片传输时需要知道每 rank 的 state 维度
                if hasattr(self.token_to_kv_pool, "get_state_dim_per_tensor"):
                    kv_args.state_dim_per_tensor = (
                        self.token_to_kv_pool.get_state_dim_per_tensor()
                    )
            elif isinstance(self.token_to_kv_pool, NSATokenToKVPool):
                kv_args.state_type = "nsa"     # Native Sparse Attention 专用状态
                if self.draft_token_to_kv_pool is not None and isinstance(
                    self.draft_token_to_kv_pool, NSATokenToKVPool
                ):
                    # NSA 草稿模型状态也需要一并传输
                    (
                        draft_state_data_ptrs,
                        draft_state_data_lens,
                        draft_state_item_lens,
                    ) = self.draft_token_to_kv_pool.get_state_buf_infos()
                    kv_args.state_data_ptrs += draft_state_data_ptrs
                    kv_args.state_data_lens += draft_state_data_lens
                    kv_args.state_item_lens += draft_state_item_lens

            else:
                kv_args.state_type = "none"
        else:
            # 标准 Attention 模型无额外状态缓冲区
            kv_args.state_data_ptrs = []
            kv_args.state_data_lens = []
            kv_args.state_item_lens = []
            kv_args.state_type = "none"

        # InfiniBand 网卡设备名（多网卡机器上可指定 RDMA 使用的网卡）
        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.gpu_id
        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(
            kv_args,
            DisaggregationMode.DECODE,
            self.scheduler.server_args,
            self.is_mla_backend,
        )
        # Staging buffer setup (only when heterogeneous TP staging is enabled)
        # staging buffer：异构 TP（Prefill TP ≠ Decode TP）时需要按 head 重排 KV
        if self.enable_staging and not self.is_mla_backend:
            kv_pool_for_heads = self.token_to_kv_pool
            if hasattr(kv_pool_for_heads, "full_kv_pool"):
                kv_pool_for_heads = kv_pool_for_heads.full_kv_pool
            per_rank_kv_heads = getattr(kv_pool_for_heads, "head_num", 0)
            if per_rank_kv_heads > 0:
                kv_args.kv_head_num = per_rank_kv_heads
                kv_args.total_kv_head_num = per_rank_kv_heads * attn_tp_size
            if hasattr(kv_manager, "set_kv_buffer_tensors"):
                kv_pool = kv_pool_for_heads
                if hasattr(kv_pool, "k_buffer") and hasattr(kv_pool, "v_buffer"):
                    # 将 K/V buffer tensor 直接传入 manager，用于零拷贝 staging 操作
                    kv_manager.set_kv_buffer_tensors(
                        kv_pool.k_buffer, kv_pool.v_buffer, kv_pool.page_size
                    )
        return kv_manager

    def add(self, req: Req, is_retracted: bool = False) -> None:
        """将新请求加入队列：超容量直接拒绝，被撤回请求放入 retracted_queue，否则创建接收器入队。"""
        if self._check_if_req_exceed_kv_capacity(req):
            return

        if is_retracted:
            # 被撤回的请求不需要重新握手，内存恢复后直接通过 resume_retracted_reqs 重分配
            req.retraction_mb_id = None
            self.retracted_queue.append(req)
        else:
            decode_req = self._create_receiver_and_enqueue(req)

            # NOTE: fake transfer does not need to resolve prefill dp rank in the pending queue
            # 测试用伪传输模式：直接以 dp_rank=0 初始化接收器，跳过握手
            if _is_fake_transfer(req, self.scheduler.server_args):
                decode_req.kv_receiver.init(0)
                return

            # Fast path: cache-only lookup, no network calls
            # 快速路径：从本地 prefill_info_table 缓存查到 dp_rank，无需网络请求
            prefill_dp_rank = self._resolve_prefill_dp_rank(req)
            logger.debug(f"prefill_dp_rank: {prefill_dp_rank}")
            if prefill_dp_rank is not None:
                decode_req.kv_receiver.init(prefill_dp_rank)
                return

            # 慢路径：需异步查询 Prefill 侧 DP 并行信息，放入 pending_reqs 等待
            self.pending_reqs.append(decode_req)

    def _resolve_prefill_dp_rank(self, req: Req) -> Optional[int]:
        """从本地缓存快速查询 Prefill 侧 DP rank（不发起网络请求）。返回 None 表示缓存未命中。"""
        prefill_info = self.kv_manager.prefill_info_table.get(_bootstrap_addr(req))
        # If None, it will go to the slow path and resolve prefill_info by _ensure_prefill_info then cache it
        if prefill_info is None:
            return None

        # 请求已显式指定目标 Prefill DP rank（调度器直接分配场景）
        if req.disagg_prefill_dp_rank is not None:
            return req.disagg_prefill_dp_rank

        # Prefill 无 DP 并行，rank 固定为 0
        if prefill_info.dp_size == 1:
            return 0

        if (
            prefill_info.follow_bootstrap_room
            and not envs.SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK.get()
        ):
            # bootstrap_room 对 dp_size 取模即为目标 rank（确定性哈希路由）
            return req.bootstrap_room % prefill_info.dp_size

        return None

    def _create_receiver_and_enqueue(self, req: Req) -> DecodeRequest:
        """为请求创建 KV 接收器并压入主队列，返回封装后的 DecodeRequest 对象。"""
        backend = (
            TransferBackend.FAKE
            if _is_fake_transfer(req, self.scheduler.server_args)
            else self.transfer_backend
        )
        kv_receiver_class = get_kv_class(backend, KVClassType.RECEIVER)

        kv_receiver = kv_receiver_class(
            mgr=self.kv_manager,
            bootstrap_addr=_bootstrap_addr(req),
            bootstrap_room=req.bootstrap_room,
        )

        decode_req = DecodeRequest(req=req, kv_receiver=kv_receiver)
        self.queue.append(decode_req)
        return decode_req

    def _check_if_req_exceed_kv_capacity(self, req: Req) -> bool:
        """若请求 input_ids 超过最大 token 容量，立即拒绝并输出错误响应；否则返回 False。"""
        if len(req.origin_input_ids) > self.max_total_num_tokens:
            message = f"Request {req.rid} exceeds the maximum number of tokens: {len(req.origin_input_ids)} > {self.max_total_num_tokens}"
            logger.error(message)
            prepare_abort(req, message, status_code=HTTPStatus.BAD_REQUEST)
            self.scheduler.stream_output([req], req.return_logprob)
            return True
        return False

    def extend(self, reqs: List[Req], is_retracted: bool = False) -> None:
        """批量调用 add()，逐个将请求加入队列。"""
        for req in reqs:
            self.add(req, is_retracted=is_retracted)

    def resume_retracted_reqs(
        self, rids_to_check: Optional[List[str]] = None
    ) -> List[Req]:
        """
        尝试恢复 retracted_queue 中因内存不足被撤回的请求。
        仅当可分配 token 空间充足时才重新预分配 KV 并返回给 waiting_queue。
        """
        # TODO refactor the scheduling part, reuse with the unified engine logic as much as possible

        # 计算当前不计入 retracted 请求的可分配 token 数
        resumed_reqs = []
        indices_to_remove = set()
        allocatable_tokens = self._allocatable_tokens(count_retracted=False)

        for i, req in enumerate(self.retracted_queue):
            if rids_to_check is not None and req.rid not in rids_to_check:
                continue

            # 请求槽满时停止，等待下一轮调度
            if self.req_to_token_pool.available_size() <= 0:
                break

            # 为撤回请求估算恢复所需 token 数（input + output + 保留 decode 步）
            required_tokens_for_request = (
                len(req.origin_input_ids)
                + len(req.output_ids)
                + self.num_reserved_decode_tokens
            )
            if required_tokens_for_request > allocatable_tokens:
                break

            resumed_reqs.append(req)
            indices_to_remove.add(i)
            req.is_retracted = False
            # 重新分配 KV 内存槽
            self._pre_alloc(req)
            allocatable_tokens -= required_tokens_for_request

            # load from cpu, release the cpu copy
            # 将 offload 到 CPU 的 KV 缓存重新加载回 GPU
            req.load_kv_cache(self.req_to_token_pool, self.token_to_kv_pool_allocator)

        self.retracted_queue = [
            entry
            for i, entry in enumerate(self.retracted_queue)
            if i not in indices_to_remove
        ]

        return resumed_reqs

    def _update_handshake_waiters(
        self, rids_to_check: Optional[List[str]] = None
    ) -> None:
        """
        对 queue 中所有未完成握手的请求轮询状态。
        Bootstrapping → 继续等待；WaitingForInput → 握手完成；Failed → 中止请求。
        """
        if not self.queue:
            return

        # 若全部请求已完成握手则直接跳过，避免无效轮询
        if all(decode_req.waiting_for_input for decode_req in self.queue):
            return

        # 跨 TP 所有 rank all-reduce 轮询结果，确保各 rank 状态一致
        polls = poll_and_all_reduce(
            [decode_req.kv_receiver for decode_req in self.queue], self.gloo_group
        )

        for i, (decode_req, poll) in enumerate(zip(self.queue, polls)):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            if poll == KVPoll.Bootstrapping:
                pass  # 握手仍在进行，本轮不处理
            elif poll == KVPoll.WaitingForInput:
                # 握手成功：标记为等待 KV 数据状态并记录时间戳
                decode_req.waiting_for_input = True
                decode_req.req.time_stats.set_bootstrap_done_time()
            elif poll == KVPoll.Failed:
                # 握手失败：记录错误并中止请求
                error_message = f"Decode handshake failed for request rank={self.tp_rank} {decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                try:
                    decode_req.kv_receiver.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.error(error_message)
                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                if self.scheduler.enable_metrics:
                    self.scheduler.metrics_collector.increment_bootstrap_failed_reqs()
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

    def _ensure_prefill_info(
        self, addr_to_reqs: Dict[str, List[DecodeRequest]]
    ) -> Tuple[Dict[str, List[DecodeRequest]], List[DecodeRequest]]:
        """
        非阻塞地为每个 bootstrap_addr 确保 Prefill 并行拓扑信息（DP/TP size 等）可用。
        超过重试间隔才会再次尝试，超过最大次数则中止相关请求。
        Returns (ready_addrs, remaining_reqs)：ready_addrs 中的请求可继续解析 dp_rank。
        """
        ready: Dict[str, List[DecodeRequest]] = {}
        remaining: List[DecodeRequest] = []

        now = time.monotonic()
        for bootstrap_addr, reqs in addr_to_reqs.items():
            # 距上次尝试时间不足 _ensure_retry_interval 秒则跳过（避免频繁 RPC）
            last_attempt = self._ensure_last_attempt_time.get(bootstrap_addr)
            if last_attempt is not None and (
                now - last_attempt < self._ensure_retry_interval
            ):
                remaining.extend(reqs)
                continue

            self._ensure_last_attempt_time[bootstrap_addr] = now

            if self.kv_manager.try_ensure_parallel_info(bootstrap_addr):
                # 获取成功：清理计数器，标记为 ready
                if bootstrap_addr in self._ensure_retry_count:
                    del self._ensure_retry_count[bootstrap_addr]
                if bootstrap_addr in self._ensure_last_attempt_time:
                    del self._ensure_last_attempt_time[bootstrap_addr]
                ready[bootstrap_addr] = reqs
                continue

            # 本次获取失败：累加重试计数
            count = self._ensure_retry_count.get(bootstrap_addr, 0) + 1
            self._ensure_retry_count[bootstrap_addr] = count

            if count >= self._max_ensure_retries:
                # 超过最大重试次数：放弃并中止所有相关请求
                error_msg = f"Could not fetch prefill parallel info from {bootstrap_addr} after {count} attempts"
                logger.error(error_msg)
                for decode_req in reqs:
                    decode_req.kv_receiver.abort()
                del self._ensure_retry_count[bootstrap_addr]
                del self._ensure_last_attempt_time[bootstrap_addr]
            else:
                # 继续等待，本轮暂不处理
                remaining.extend(reqs)

        return ready, remaining

    def _resolve_pending_reqs(self) -> None:
        """
        批量解析 pending_reqs 中每个请求的 Prefill DP rank，成功后调用 kv_receiver.init()。
        分两步：1) 确保 Prefill 并行拓扑信息就绪；2) 解析具体 dp_rank（本地缓存或网络查询）。
        """
        if not self.pending_reqs:
            return

        # 按 bootstrap_addr 分组，减少重复网络请求
        addr_to_reqs: Dict[str, List[DecodeRequest]] = {}
        for decode_req in self.pending_reqs:
            addr = _bootstrap_addr(decode_req.req)
            addr_to_reqs.setdefault(addr, []).append(decode_req)

        # Pass 1: ensure parallel info for each addr
        # 第一步：确保每个 Prefill 地址的并行拓扑信息已缓存
        ready_addrs, remaining = self._ensure_prefill_info(addr_to_reqs)

        resolved: List[Tuple[DecodeRequest, int]] = []
        for bootstrap_addr, decode_reqs in ready_addrs.items():
            need_query: List[DecodeRequest] = []
            for decode_req in decode_reqs:
                prefill_dp_rank = self._resolve_prefill_dp_rank(decode_req.req)
                if prefill_dp_rank is not None:
                    resolved.append((decode_req, prefill_dp_rank))
                else:
                    # 本地缓存无法确定 rank，需向 Prefill 服务发起 RPC 查询
                    need_query.append(decode_req)

            # Pass 2: resolve dp rank for addrs whose info is available
            # 第二步：通过 bootstrap 服务查询特定 room 的目标 DP rank
            if need_query:
                rooms = [decode_req.req.bootstrap_room for decode_req in need_query]
                room_to_rank = CommonKVReceiver.query_prefill_dp_ranks(
                    bootstrap_addr, rooms
                )
                for decode_req in need_query:
                    prefill_dp_rank = room_to_rank.get(
                        str(decode_req.req.bootstrap_room)
                    )
                    if prefill_dp_rank is not None:
                        resolved.append((decode_req, int(prefill_dp_rank)))
                    else:
                        # 查询返回中未包含该 room，继续等待
                        remaining.append(decode_req)

        # 更新 pending_reqs：仅保留仍未解析的请求
        self.pending_reqs = remaining

        for decode_req, prefill_dp_rank in resolved:
            # 解析成功：用确定的 dp_rank 初始化 KV 接收器，开始握手
            decode_req.kv_receiver.init(prefill_dp_rank)

    def pop_preallocated(
        self, rids_to_check: Optional[List[str]] = None
    ) -> Tuple[List[DecodeRequest], List[DecodeRequest]]:
        """
        从 PreallocQueue 中弹出已完成握手且内存预分配成功的请求，移入 TransferQueue。
        同时清除已中止的请求，返回 (preallocated_reqs, failed_reqs)。
        """
        # 先解析 pending_reqs（慢路径 DP rank 查询）并更新握手状态
        self._resolve_pending_reqs()
        self._update_handshake_waiters(rids_to_check)

        failed_reqs = []
        preallocated_reqs = []
        indices_to_remove = set()

        # We need to make sure that the sum of inflight tokens and allocatable tokens is greater than maximum input+output length of each inflight request
        # Otherwise it is possible for one request running decode out of memory, while all other requests are in the transfer queue that cannot be retracted.
        # 计算可被撤回的 token 数：当前 running batch 中各请求的 token 总和
        retractable_tokens = sum(
            len(r.origin_input_ids) + len(r.output_ids)
            for r in self.scheduler.running_batch.reqs
        )
        # 基于可撤回 token 估算当前可分配 token 数（含 retracted_queue 预留）
        allocatable_tokens = self._allocatable_tokens(
            retractable_tokens=retractable_tokens, count_retracted=True
        )
        # First, remove all failed requests from the queue
        # 第一步：清除所有已被标记为 FINISH_ABORT 的请求
        for i, decode_req in enumerate(self.queue):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue
            if isinstance(decode_req.req.finished_reason, FINISH_ABORT):
                self.scheduler.stream_output(
                    [decode_req.req], decode_req.req.return_logprob
                )
                failed_reqs.append(decode_req)
                indices_to_remove.add(i)

        # HiSparse physical constraint: max requests by device buffer capacity.
        # Each admitted req needs padded_buffer_size from hisparse device pool.
        # waiting_queue reqs already have device buffers (allocated in admit_request_direct),
        # only transfer_queue reqs are pending device buffer allocation.
        # HiSparse 设备缓冲区容量约束：可接纳请求数 = 可用设备缓冲区 / padded_buffer_size - transfer_queue 中已占用
        hisparse_req_budget = float("inf")
        if self.scheduler.enable_hisparse:
            hisparse_avail = (
                self.token_to_kv_pool_allocator.hisparse_attn_allocator.available_size()
            )
            hisparse_req_budget = max(
                0,
                hisparse_avail // self.scheduler.hisparse_coordinator.padded_buffer_size
                - len(self.transfer_queue.queue),
            )

        # Then, preallocate the remaining requests if possible
        # 第二步：逐个尝试为已完成握手（waiting_for_input=True）的请求预分配 KV 内存
        for i, decode_req in enumerate(self.queue):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            if i in indices_to_remove:
                continue

            # 只处理已完成握手、等待 Prefill 发送 KV 的请求
            if not decode_req.waiting_for_input:
                continue

            # 请求槽或元数据缓冲区不足时停止，等待下一轮
            if self.req_to_token_pool.available_size() <= 0:
                break

            if self.req_to_metadata_buffer_idx_allocator.available_size() <= 0:
                break

            if hisparse_req_budget <= 0:
                break

            # Memory estimation: don't add if the projected memory cannot be met
            # TODO: add new_token ratio
            # 内存估算：确保当前分配后，running batch 中最长请求仍能完成
            origin_input_len = len(decode_req.req.origin_input_ids)
            required_tokens_for_request = (
                origin_input_len + self.num_reserved_decode_tokens
            )

            if (
                max(
                    required_tokens_for_request,
                    origin_input_len
                    + min(
                        decode_req.req.sampling_params.max_new_tokens,
                        CLIP_MAX_NEW_TOKEN,
                    )
                    - retractable_tokens,
                )
                > allocatable_tokens
            ):
                break
            if required_tokens_for_request > allocatable_tokens:
                break

            # 扣除本次预分配消耗，更新剩余可分配量
            allocatable_tokens -= required_tokens_for_request
            hisparse_req_budget -= 1
            dst_kv_indices = self._pre_alloc(decode_req.req)

            origin_input_len = len(decode_req.req.origin_input_ids)
            if self.scheduler.enable_hisparse:
                # Must cast to int32 for ZMQ serialization — from_zmq reads np.int32.
                # HiSparse 直接写主机内存：取 host pool 索引，转为 int32
                kv_indices = (
                    dst_kv_indices[:origin_input_len].cpu().numpy().astype(np.int32)
                )
                page_size = 1  # host pool page_size
            else:
                # 标准路径：从 req_to_token 中读取分配的 KV 索引
                kv_indices_full = self.req_to_token_pool.req_to_token[
                    decode_req.req.req_pool_idx
                ][:origin_input_len]
                kv_indices = kv_indices_full.cpu().numpy()
                page_size = self.token_to_kv_pool_allocator.page_size

            # Prepare extra pool indices for hybrid models
            # 为混合模型准备附加状态索引（Mamba/SWA/NSA）
            if isinstance(self.token_to_kv_pool, HybridLinearKVPool):
                # Mamba hybrid model: single mamba state index
                # Mamba 混合模型：每个请求一个 SSM 状态索引
                state_indices = [
                    self.req_to_token_pool.req_index_to_mamba_index_mapping[
                        decode_req.req.req_pool_idx
                    ]
                    .cpu()
                    .numpy()
                ]
            elif isinstance(self.token_to_kv_pool, SWAKVPool):
                # SWA hybrid model: send decode-side SWA window indices
                # SWA 混合模型：只传输滑动窗口内的 KV 索引
                seq_len = len(decode_req.req.origin_input_ids)
                window_size = self.scheduler.sliding_window_size

                # 对窗口起始位置按页对齐
                window_start = max(0, seq_len - window_size)
                window_start = (window_start // page_size) * page_size
                window_kv_indices_full = self.req_to_token_pool.req_to_token[
                    decode_req.req.req_pool_idx, window_start:seq_len
                ]

                # Translate to SWA pool indices
                # 将全局 KV 索引转换为 SWA 专用池索引
                window_kv_indices_swa = (
                    self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                        window_kv_indices_full
                    )
                )
                state_indices = window_kv_indices_swa.cpu().numpy()
                state_indices = kv_to_page_indices(state_indices, page_size)
            elif isinstance(self.token_to_kv_pool, NSATokenToKVPool):
                # NSA 稀疏 Attention：使用设备级 page_size 计算页索引
                seq_len = len(decode_req.req.origin_input_ids)
                kv_indices_full = self.req_to_token_pool.req_to_token[
                    decode_req.req.req_pool_idx, :seq_len
                ]
                state_indices = kv_indices_full.cpu().numpy()
                # Indexer lives on device pool; always use device page_size
                device_page_size = self.token_to_kv_pool.page_size
                state_indices = kv_to_page_indices(state_indices, device_page_size)
            else:
                # 标准 Attention 模型无附加状态
                state_indices = None

            # 分配元数据缓冲区槽位（用于接收 Prefill 传回的首 token 等信息）
            decode_req.metadata_buffer_index = (
                self.req_to_metadata_buffer_idx_allocator.alloc()
            )
            assert decode_req.metadata_buffer_index is not None
            # 将 token 级 KV 索引转换为页级索引，发送给 Prefill 侧作为写目标
            page_indices = kv_to_page_indices(kv_indices, page_size)
            decode_req.kv_receiver.send_metadata(
                page_indices, decode_req.metadata_buffer_index, state_indices
            )
            if (
                self.transfer_queue.enable_staging
                and hasattr(decode_req.kv_receiver, "require_staging")
                and decode_req.kv_receiver.require_staging
            ):
                # 异构 TP staging 路径：注册到 staging_handler 以便按序拼接分片
                self.transfer_queue.staging_handler.register_decode_req(
                    decode_req.req.bootstrap_room, decode_req
                )
            preallocated_reqs.append(decode_req)
            indices_to_remove.add(i)
            # 记录进入 TransferQueue 的时间戳（用于延迟统计）
            decode_req.req.time_stats.set_decode_transfer_queue_entry_time()

        # 从队列中移除已预分配和已失败的请求
        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return preallocated_reqs, failed_reqs

    @property
    def num_tokens_pre_allocated(self):
        """返回当前 TransferQueue 中所有请求已预分配的 fill_ids token 总数。"""
        return sum(
            len(decode_req.req.fill_ids) for decode_req in self.transfer_queue.queue
        )

    def _allocatable_tokens(
        self, retractable_tokens: Optional[int] = None, count_retracted: bool = True
    ) -> int:
        """
        估算当前可安全分配给新预分配请求的 token 数。
        需确保：1) 每个正在运行的请求能完成（不被撤回后 OOM）；2) 为 retracted 请求预留空间。
        """
        # 若有可撤回 token，计算 running batch 中最大所需空间
        need_space_for_single_req = (
            max(
                [
                    min(x.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKEN)
                    + len(x.origin_input_ids)
                    - retractable_tokens
                    for x in self.scheduler.running_batch.reqs
                ]
            )
            if retractable_tokens is not None
            and len(self.scheduler.running_batch.reqs) > 0
            else 0
        )
        if self.scheduler.enable_hisparse:
            # HiSparse pre-alloc only allocates logical indices (alloc_logical_only),
            # so the logical pool is the binding constraint for admission control.
            # HiSparse 模式下，预分配只占逻辑索引，逻辑池是容量瓶颈
            available_size = (
                self.token_to_kv_pool_allocator.logical_attn_allocator.available_size()
            )
        else:
            available_size = self.token_to_kv_pool_allocator.available_size()
        # 可分配量 = 空闲量 - max(各队列预留 decode 步, 单请求最大 token 需求)
        allocatable_tokens = available_size - max(
            # preserve some space for future decode
            # 为所有 running+transfer+waiting 队列中的请求各预留若干 decode 步空间
            self.num_reserved_decode_tokens
            * (
                len(self.scheduler.running_batch.reqs)
                + len(self.transfer_queue.queue)
                + len(self.scheduler.waiting_queue)
            ),
            # make sure each request can finish if reach max_tokens with all other requests retracted
            need_space_for_single_req,
        )

        # Note: if the last prebuilt extend just finishes, and we enter `pop_preallocated` immediately in the next iteration
        #       the extend batch is not in any queue, so we need to explicitly add the tokens slots here
        # 若上一个 prebuilt extend batch 刚完成但尚未进入 running batch，需手动扣除其 decode 步预留
        if (
            self.scheduler.last_batch
            and self.scheduler.last_batch.forward_mode.is_prebuilt()
        ):
            allocatable_tokens -= self.num_reserved_decode_tokens * len(
                self.scheduler.last_batch.reqs
            )

        if count_retracted:
            # 为 retracted_queue 中被撤回的请求预留恢复所需的 token 空间
            allocatable_tokens -= sum(
                [
                    len(req.origin_input_ids)
                    + len(req.output_ids)
                    + self.num_reserved_decode_tokens
                    for req in self.retracted_queue
                ]
            )
        return allocatable_tokens

    def _pre_alloc(self, req: Req) -> torch.Tensor:
        """
        为请求预分配 req_to_token_pool 槽位和 KV Cache token 槽。
        返回用于 RDMA 写入目标的 KV 索引张量（HiSparse 路径返回 host pool 索引）。
        """
        req_pool_indices = self.req_to_token_pool.alloc([req])

        assert (
            req_pool_indices is not None
        ), "req_pool_indices is full! There is a bug in memory estimation."

        # Alloc all tokens for the prebuilt req (except for the reserved input token for decoding)
        # fill_len = input token 数 + 已生成 output token 数 - 1（最后一个 output token 留给 decode 步用）
        fill_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
        req.kv_allocated_len = fill_len
        req.kv_committed_len = fill_len

        if self.scheduler.enable_hisparse:
            # Direct-to-host path: only allocate logical indices (no hisparse
            # device indices) and allocate host indices for RDMA destination.
            # HiSparse：分配逻辑索引（不分 device 物理页）+ Host 池索引作为 RDMA 写目标
            coordinator = self.scheduler.hisparse_coordinator
            device = self.token_to_kv_pool_allocator.device
            kv_loc = self.token_to_kv_pool_allocator.alloc_logical_only(
                prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
                prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
                seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
                seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
                last_loc=torch.tensor([-1], dtype=torch.int64, device=device),
                extend_num_tokens=fill_len,
            )
            # Allocate host indices for the RDMA transfer target
            # 在 host 内存池中分配 RDMA 写目标槽
            host_indices = coordinator.mem_pool_host.alloc(fill_len)
            if host_indices is None:
                raise RuntimeError(
                    f"HiSparse host mem pool alloc failed for {fill_len} tokens "
                    f"in _pre_alloc (req {req.rid})"
                )
            host_indices = host_indices.to(device=coordinator.device)
            coordinator.req_to_host_pool[req.req_pool_idx, :fill_len] = host_indices
        elif self.token_to_kv_pool_allocator.page_size == 1:
            # 标准 token 级分配（page_size=1，每个 token 一个独立 KV 槽）
            kv_loc = self.token_to_kv_pool_allocator.alloc(fill_len)
        else:
            # 页级 KV 分配（paged attention 模式）
            device = self.token_to_kv_pool_allocator.device
            kv_loc = self.token_to_kv_pool_allocator.alloc_extend(
                prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
                prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
                seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
                seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
                last_loc=torch.tensor([-1], dtype=torch.int64, device=device),
                extend_num_tokens=fill_len,
            )

        assert (
            kv_loc is not None
        ), "KV cache is full! There is a bug in memory estimation."

        # 将 KV token 位置写入请求的 req_to_token 槽
        self.req_to_token_pool.write((req.req_pool_idx, slice(0, len(kv_loc))), kv_loc)

        # populate metadata
        # 填充 fill_ids：input + output（用于后续 prebuilt extend batch）
        req.fill_ids = req.origin_input_ids + req.output_ids
        req.set_extend_input_len(len(req.fill_ids))

        # Return the transfer destination indices:
        # HiSparse 返回 host 索引（RDMA 写目标），其余返回 GPU KV 索引
        if self.scheduler.enable_hisparse:
            return host_indices
        return kv_loc


class DecodeTransferQueue:
    """
    传输队列（阶段 2）：轮询 KV 接收器状态，KV 数据到达后将请求提升至 WaitingQueue。
    通过 all-reduce 确保所有 TP rank 对传输状态的判断一致。
    """

    def __init__(
        self,
        gloo_group: ProcessGroup,
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        tp_rank: int,
        metadata_buffers: MetadataBuffers,
        scheduler: Scheduler,
        tree_cache: BasePrefixCache,
    ):
        self.queue: List[DecodeRequest] = []
        self.gloo_group = gloo_group  # 用于跨 TP rank 同步轮询结果
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.tp_rank = tp_rank
        self.metadata_buffers = metadata_buffers  # 存储 Prefill 传回的首 token 等元数据
        self.scheduler = scheduler
        self.tree_cache = tree_cache
        self.spec_algorithm = scheduler.spec_algorithm  # speculative decoding 算法类型
        # staging buffer 开关：用于异构 TP 情况下的 KV 分片重排
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()
        self.staging_handler = None

    def add(self, decode_req: DecodeRequest) -> None:
        """将单个 DecodeRequest 加入传输队列。"""
        self.queue.append(decode_req)

    def extend(self, decode_reqs: List[DecodeRequest]) -> None:
        """批量加入传输队列；启用 staging 时同步注册到 staging_handler。"""
        self.queue.extend(decode_reqs)
        if self.enable_staging:
            for dr in decode_reqs:
                if (
                    hasattr(dr.kv_receiver, "require_staging")
                    and dr.kv_receiver.require_staging
                ):
                    # 异构 TP staging：按 bootstrap_room 注册，handler 负责按顺序拼接分片
                    self.staging_handler.register_decode_req(dr.req.bootstrap_room, dr)

    def _commit_transfer_to_req(self, decode_req: DecodeRequest) -> bool:
        """
        将元数据缓冲区中的 Prefill 侧结果写入请求对象。
        Returns:
            True  — 请求应从传输队列移除（成功提交或检测到数据损坏）
            False — 元数据尚未就绪，本轮跳过（保留在队列中继续轮询）
        """
        idx = decode_req.metadata_buffer_index
        # 从共享缓冲区读取 Prefill 传回的所有元数据字段
        (
            output_id,
            cached_tokens,
            output_token_logprobs_val,
            output_token_logprobs_idx,
            output_top_logprobs_val,
            output_top_logprobs_idx,
            output_topk_p,
            output_topk_index,
            output_hidden_states,
            output_bootstrap_room,
        ) = self.metadata_buffers.get_buf(idx)

        # Validate bootstrap_room to detect context corruption
        # 用 bootstrap_room 作为校验码，检测元数据缓冲区槽位碰撞
        actual_room = output_bootstrap_room[0].item()
        expected_room = (
            decode_req.req.bootstrap_room
            if decode_req.req.bootstrap_room is not None
            else 0
        )

        if _is_fake_transfer(decode_req.req, self.scheduler.server_args):
            pass
        elif actual_room == 0:
            # Case 1: Metadata not ready yet (actual_room == 0)
            # Keep request in queue and wait for next poll
            # 情形 1：元数据尚未写入（bootstrap_room 初始值为 0），等待下一轮
            return False
        elif actual_room != expected_room:
            # Case 2: Real corruption detected (mismatch)
            # Abort the request and remove from the queue
            # 情形 2：检测到真实数据损坏（room 不匹配），立即中止请求
            error_msg = (
                f"Context corruption detected: Request {decode_req.req.rid} "
                f"(bootstrap_room={expected_room}) received metadata from "
                f"bootstrap_room={actual_room}. "
                f"Metadata buffer index: {idx}. "
                f"This indicates metadata buffer index collision."
            )
            logger.error(error_msg)
            prepare_abort(
                decode_req.req,
                "Metadata corruption detected - bootstrap_room mismatch",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            decode_req.kv_receiver.clear()
            decode_req.kv_receiver = None
            return True

        # Case 3: Success - commit the transfer
        # 情形 3：成功 —— 将 Prefill 侧第一个 output token 及缓存统计写入请求
        decode_req.req.output_ids.append(output_id[0].item())
        decode_req.req.cached_tokens = cached_tokens[0].item()
        decode_req.req.cached_tokens_device = cached_tokens[1].item()
        decode_req.req.cached_tokens_host = cached_tokens[2].item()
        decode_req.req.cached_tokens_storage = cached_tokens[3].item()
        if not self.spec_algorithm.is_none():
            # Speculative decoding：写入草稿 top-k 概率和隐层状态
            decode_req.req.output_topk_p = output_topk_p
            decode_req.req.output_topk_index = output_topk_index
            decode_req.req.hidden_states_tensor = output_hidden_states

        if decode_req.req.return_logprob:
            # 写入 log probability 相关字段
            decode_req.req.output_token_logprobs_val.append(
                output_token_logprobs_val[0].item()
            )
            decode_req.req.output_token_logprobs_idx.append(
                output_token_logprobs_idx[0].item()
            )
            decode_req.req.output_top_logprobs_val.append(
                output_top_logprobs_val[: decode_req.req.top_logprobs_num].tolist()
            )
            decode_req.req.output_top_logprobs_idx.append(
                output_top_logprobs_idx[: decode_req.req.top_logprobs_num].tolist()
            )

        # 清理 KV 接收器并记录进入 WaitingQueue 的时间
        decode_req.kv_receiver.clear()
        decode_req.kv_receiver = None
        decode_req.req.time_stats.set_wait_queue_entry_time()
        return True

    def _poll_with_staging(self) -> list:
        """使用 staging handler 执行带分片重排的 all-reduce 轮询（异构 TP 场景）。"""
        return poll_and_all_reduce_with_staging(
            self.queue, self.staging_handler, self.gloo_group
        )

    def _init_staging_handler(self, kv_manager):
        """从 kv_manager 创建 staging handler，必须恰好调用一次。"""
        from sglang.srt.disaggregation.common.staging_handler import (
            DecodeStagingHandler,
        )

        self.staging_handler = DecodeStagingHandler.create(
            kv_manager, self.scheduler, self.tp_rank
        )
        kv_manager._staging_handler = self.staging_handler

    def pop_transferred(self, rids_to_check: Optional[List[str]] = None) -> List[Req]:
        """
        轮询传输状态，将 KV 已成功到达的请求提升至 WaitingQueue。
        处理三种情形：Failed（传输失败）、Success（提交元数据）、Bootstrapping/Transferring（继续等待）。
        """
        if not self.queue:
            return []

        # 按 staging 开关选择 all-reduce 轮询方式
        if self.enable_staging:
            polls = self._poll_with_staging()
        else:
            # 标准 all-reduce：确保所有 TP rank 对每个请求的轮询结果一致
            polls = poll_and_all_reduce(
                [dr.kv_receiver for dr in self.queue], self.gloo_group
            )

        transferred_reqs = []
        indices_to_remove = set()
        for i, (decode_req, poll) in enumerate(zip(self.queue, polls)):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            if poll == KVPoll.Failed:
                # KV 传输失败：记录错误、中止请求、释放 KV 缓存
                error_message = f"Decode transfer failed for request rank={self.tp_rank} {decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                try:
                    decode_req.kv_receiver.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.error(error_message)
                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                self.scheduler.stream_output(
                    [decode_req.req], decode_req.req.return_logprob
                )
                if self.scheduler.enable_hisparse:
                    self.scheduler.hisparse_coordinator.request_finished(decode_req.req)
                # release pre-allocated kv cache, but don't insert into the tree since it's failed
                # 释放预分配的 KV 缓存，但不插入前缀树（传输失败不应缓存）
                release_kv_cache(decode_req.req, self.tree_cache, is_insert=False)
                indices_to_remove.add(i)
                if self.scheduler.enable_metrics:
                    self.scheduler.metrics_collector.increment_transfer_failed_reqs()
                continue
            elif poll == KVPoll.Success:
                # KV 传输成功：提交元数据到请求对象
                should_remove = self._commit_transfer_to_req(decode_req)
                if should_remove:
                    indices_to_remove.add(i)
                    # Check if request was aborted due to corruption
                    # 检查是否因元数据损坏被中止
                    if isinstance(decode_req.req.finished_reason, FINISH_ABORT):
                        self.scheduler.stream_output(
                            [decode_req.req], decode_req.req.return_logprob
                        )
                        if self.scheduler.enable_hisparse:
                            self.scheduler.hisparse_coordinator.request_finished(
                                decode_req.req
                            )
                        release_kv_cache(
                            decode_req.req, self.tree_cache, is_insert=False
                        )
                        if self.scheduler.enable_metrics:
                            self.scheduler.metrics_collector.increment_transfer_failed_reqs()
                    else:
                        # 正常完成传输，加入 transferred_reqs 返回给 waiting_queue
                        transferred_reqs.append(decode_req.req)
            elif poll in [
                KVPoll.Bootstrapping,
                KVPoll.WaitingForInput,
                KVPoll.Transferring,
            ]:
                pass  # KV 传输仍在进行中，本轮跳过
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

        # 释放已移除请求的元数据缓冲区槽位
        for i in indices_to_remove:
            if self.enable_staging and self.staging_handler.is_staging_room(
                self.queue[i].req.bootstrap_room
            ):
                self.staging_handler.unregister_decode_req(
                    self.queue[i].req.bootstrap_room
                )
            idx = self.queue[i].metadata_buffer_index
            assert idx != -1
            self.req_to_metadata_buffer_idx_allocator.free(idx)

        # 从队列中移除已处理的请求
        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return transferred_reqs


class SchedulerDisaggregationDecodeMixin:
    """
    Decode 侧调度器 Mixin，提供 PD 分离架构下的事件循环和批次调度逻辑。
    包含 normal（串行）和 overlap（流水线并行）两种事件循环，以及
    prebuilt batch 构建、waiting queue 管理和 process_decode_queue 主流程。
    """

    @torch.no_grad()
    def event_loop_normal_disagg_decode(self: Scheduler):
        """Decode 侧串行事件循环：接收请求 → 处理 KV 队列 → 调度批次 → 运行推理。"""

        while True:
            # Receive requests
            # 接收来自 TokenizerManager 的新请求并加入预分配队列
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            # 推进 KV 传输队列：pop_preallocated + pop_transferred
            self.process_decode_queue()
            if self._engine_paused:
                continue

            # Get the next batch to run
            # 从 waiting_queue + running_batch 中构建下一个 decode 批次
            batch = self.get_next_disagg_decode_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # When the server is idle, do self-check and re-init some states
                # 空闲时执行自检和状态重置
                self.on_idle()

            # Update last_batch
            self.last_batch = batch

    @torch.no_grad()
    def event_loop_overlap_disagg_decode(self: Scheduler):
        """
        Decode 侧流水线事件循环：当前批次执行期间异步处理上一批次结果，
        提高 GPU 利用率（overlap compute with communication/post-processing）。
        """
        self.result_queue = deque()
        self.last_batch: Optional[ScheduleBatch] = None

        while True:
            # Receive requests
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            self.process_decode_queue()
            if self._engine_paused:
                continue

            # Get the next batch to run
            batch = self.get_next_disagg_decode_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            # 提交当前批次到 GPU 执行（异步，结果入队）
            if batch:
                batch_result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None

            # Process the last batch
            # 处理上一批次的结果（与当前批次 GPU 执行重叠）
            if self.last_batch:
                tmp_batch, tmp_result = self.result_queue.popleft()
                self.process_batch_result(tmp_batch, tmp_result)
            elif batch is None:
                self.on_idle()

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            # 采样依赖上一批次结果（如 grammar 约束），故在 process_batch_result 之后执行
            self.launch_batch_sample_if_needed(batch_result)

            # Update last_batch
            self.last_batch = batch

    def _run_batch_prebuilt(
        self: Scheduler, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        """
        执行 prebuilt batch（已完成 Prefill 的请求组成的 fake extend batch）。
        若存在 inner_idle_batch 则执行 idle batch，否则返回空结果。
        """
        if batch.inner_idle_batch is not None:
            idle_batch = batch.inner_idle_batch
            # Reset the inner idle batch to avoid reusing it.
            # 清空 inner_idle_batch，防止下轮重复使用
            batch.inner_idle_batch = None
            return self.run_batch(idle_batch)

        return GenerationBatchResult()

    def get_next_disagg_decode_batch_to_run(
        self: Scheduler,
    ) -> Optional[ScheduleBatch]:
        """
        处理 prebuilt batch（跳过 Prefill 直接填充元数据）并调度下一个 decode 批次。
        """
        # Process pending prebuilt batch: output processing + filter + merge
        # 处理待完成的 prebuilt batch：输出处理 → 过滤已结束请求 → 合并到 running batch
        new_prebuilt_batch = self.get_new_prebuilt_batch()
        if new_prebuilt_batch:
            assert self.chunked_req is None
            self.process_batch_result_prebuilt(new_prebuilt_batch)
            new_prebuilt_batch.filter_batch()
            if not new_prebuilt_batch.is_empty():
                if self.running_batch.is_empty():
                    # running_batch 为空，直接用 prebuilt batch 作为新的 running_batch
                    self.running_batch = new_prebuilt_batch
                    if self.enable_hisparse:
                        self.running_batch.hisparse_coordinator = (
                            self.hisparse_coordinator
                        )
                else:
                    # 将 prebuilt batch 中的请求合并到已有 running_batch
                    self.running_batch.merge_batch(new_prebuilt_batch)

        # Schedule decode batch
        if self.running_batch.is_empty():
            ret = None
        else:
            # 更新 running batch（检查停止条件、重试等）
            self.running_batch = self.update_running_batch(self.running_batch)
            ret = self.running_batch if not self.running_batch.is_empty() else None

        # MLP sync batch：DP attention 需要在各 rank 间同步 MLP 输出
        ret = self.maybe_prepare_mlp_sync_batch(ret)
        if ret:
            set_schedule_time_batch(ret)
        return ret

    def get_new_prebuilt_batch(self: Scheduler) -> Optional[ScheduleBatch]:
        """
        从 waiting_queue 中取出已完成 KV 传输的请求，构建 prebuilt extend batch
        （跳过 Prefill forward，直接填充 KV 元数据进入 Decode 阶段）。
        """
        if self.grammar_manager.has_waiting_grammars():
            # 优先处理 grammar 已就绪的请求
            ready_grammar_requests = self.grammar_manager.get_ready_grammar_requests()
            for req in ready_grammar_requests:
                self._add_request_to_queue(req)

        if len(self.waiting_queue) == 0:
            return None

        curr_batch_size = self.running_batch.batch_size()

        # 计算本轮最多可加入 running_batch 的新请求数
        batch_size = min(self.req_to_token_pool.size, self.max_running_requests)

        num_not_used_batch = batch_size - curr_batch_size

        # pop req from waiting queue
        # 从 waiting_queue 中取出不超过 num_not_used_batch 个请求
        can_run_list: List[Req] = []
        waiting_queue: List[Req] = []

        for i in range(len(self.waiting_queue)):
            req = self.waiting_queue[i]
            # we can only add at least `num_not_used_batch` new batch to the running queue
            if i < num_not_used_batch:
                can_run_list.append(req)
                # 初始化请求的 next round input（prefix cache 命中检查等）
                req.init_next_round_input(self.tree_cache)
            else:
                waiting_queue.append(req)

        self.waiting_queue = waiting_queue
        if len(can_run_list) == 0:
            return None

        # 批量记录进入 forward 阶段的时间戳
        set_time_batch(can_run_list, "set_forward_entry_time")

        # construct a schedule batch with those requests and mark as decode
        # 构建 ScheduleBatch，但 forward mode 标记为 prebuilt（跳过 Prefill 计算）
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )

        # construct fake completed prefill
        # 构建伪 Prefill 完成状态：填充 KV 元数据、设置 attention 掩码等
        new_batch.prepare_for_prebuilt()
        new_batch.process_prebuilt(self.server_args, self.future_map)

        return new_batch

    def process_decode_queue(self: Scheduler):
        """
        Decode 侧主流程：恢复被撤回请求 → pop_preallocated → pop_transferred → 更新 waiting_queue。
        按 polling_interval 控制轮询频率，避免过于频繁的跨节点状态查询。
        """
        if self.server_args.disaggregation_decode_enable_offload_kvcache:
            # KV Cache offload 到 CPU 时检查 offload 进度
            self.decode_offload_manager.check_offload_progress()

        # try to resume retracted requests if there are enough space for another `num_reserved_decode_tokens` decode steps
        # 尝试恢复因内存压力而被撤回的请求
        resumed_reqs = self.disagg_decode_prealloc_queue.resume_retracted_reqs()
        self.waiting_queue.extend(resumed_reqs)
        if len(self.disagg_decode_prealloc_queue.retracted_queue) > 0:
            # if there are still retracted requests, we do not allocate new requests
            # 仍有未恢复的撤回请求时，暂停接纳新预分配请求，优先保障已有请求
            return

        # 初始化轮询计数器
        if not hasattr(self, "polling_count"):
            self.polling_count = 0
            self.polling_interval = (
                self.server_args.disaggregation_decode_polling_interval
            )

        # 按 polling_interval 节流：每隔 polling_interval 个调度周期才执行一次 KV 队列轮询
        self.polling_count = (self.polling_count + 1) % self.polling_interval

        if self.polling_count % self.polling_interval == 0:
            # 将握手完成且内存已预分配的请求从 PreallocQueue 移入 TransferQueue
            req_conns, _ = self.disagg_decode_prealloc_queue.pop_preallocated()
            self.disagg_decode_transfer_queue.extend(req_conns)
            # 轮询 KV 传输状态，将已到达的请求提升至 WaitingQueue
            transferred_reqs = (
                self.disagg_decode_transfer_queue.pop_transferred()
            )  # the requests which kv has arrived
            if self.enable_hisparse:
                for req in transferred_reqs:
                    # Direct-to-host: KV data already in host pool, skip staging
                    # HiSparse Direct-to-Host：KV 已在 host 内存，直接 admit 无需 staging
                    self.hisparse_coordinator.admit_request_direct(req)
                self.waiting_queue.extend(transferred_reqs)
            else:
                self.waiting_queue.extend(transferred_reqs)
