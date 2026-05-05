"""
Prefill 侧（PD 分离架构中的 Prefill Server）请求生命周期管理模块。
Prefill 完成后，KV Cache 通过 RDMA/nixl 传输到 Decode Server，再由 Decode 侧进行后续 token 生成。

请求经历 3 个阶段：BootstrapQueue → WaitingQueue → InflightQueue。

Life cycle of a request in the prefill server

1. Bootstrap Queue
    a. Initialize a sender for each request
    b. Use the queue to store requests whose bootstrap (handshake and preallocation) has not finished
    c. Poll senders to check bootstrap state
    d. Once bootstrap is complete, move request to Waiting Queue

2. Waiting Queue
    a. Use PrefillAdder to pop requests
    b. Run forward
    c. Add the request to Inflight Queue

3. Inflight Queue
    a. Poll (non-blocking) the sender of the request
    b. Once the transfer has finished, return the request
"""

from __future__ import annotations

import logging
from collections import deque
from http import HTTPStatus
from typing import TYPE_CHECKING, List, Optional

import torch

# KV 传输状态轮询枚举：Bootstrapping/WaitingForInput/Transferring/Success/Failed
from sglang.srt.disaggregation.base import KVPoll
# KV 传输管理器基类（含 bootstrap 服务、transfer engine 等）
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,              # 伪 bootstrap 主机名，用于测试模式
    DisaggregationMode,               # PREFILL / DECODE 侧标识
    KVClassType,                      # KV 类工厂枚举
    MetadataBuffers,                  # Prefill → Decode 元数据共享缓冲区
    ReqToMetadataIdxAllocator,        # 元数据缓冲区索引分配器
    TransferBackend,                  # 传输后端：mooncake/nixl/fake
    get_kv_class,                     # 按后端和类型获取对应类
    is_mla_backend,                   # 判断是否为 MLA 后端
    kv_to_page_indices,               # token 索引 → 页级索引
    kv_to_page_num,                   # token 数 → 页数
    poll_and_all_reduce_attn_cp_tp_group,  # 跨 attn_cp 和 attn_tp 两组 all-reduce 轮询
    prepare_abort,                    # 将请求标记为 FINISH_ABORT
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,     # 请求中止完成原因
    FINISH_LENGTH,    # 达到最大长度完成原因（Prefill 完成后设置）
    Req,
    ScheduleBatch,
)
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, NSATokenToKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
# 批次调度时间戳工具
from sglang.srt.observability.req_time_stats import set_schedule_time_batch

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler
    from sglang.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)


def release_req_to_metadata_buffer(
    req: Req, allocator: ReqToMetadataIdxAllocator
) -> None:
    """
    安全释放 Prefill 侧为请求分配的元数据缓冲区槽位。
    仅当 metadata_buffer_index 有效（>= 0）时才实际释放，避免重复释放。

    Release the metadata buffer index allocated for a request in prefill disaggregation mode.

    This function safely releases the metadata buffer index if it was allocated.

    Args:
        req: The request object that may have a metadata_buffer_index allocated
        allocator: The ReqToMetadataIdxAllocator instance to free the index
    """
    if (
        hasattr(req, "metadata_buffer_index")
        and req.metadata_buffer_index is not None
        and req.metadata_buffer_index >= 0
    ):
        allocator.free(req.metadata_buffer_index)
        req.metadata_buffer_index = -1


class PrefillBootstrapQueue:
    """
    Bootstrap 队列（阶段 1）：管理 Prefill 侧 KV 发送器的握手过程。
    握手完成（WaitingForInput）后将请求移入 WaitingQueue，失败则中止请求。
    """

    def __init__(
        self,
        token_to_kv_pool: KVCache,
        draft_token_to_kv_pool: Optional[KVCache],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        tp_rank: int,
        tp_size: int,
        gpu_id: int,
        bootstrap_port: int,
        gloo_group: ProcessGroup,
        max_total_num_tokens: int,
        scheduler: Scheduler,
        pp_rank: int,
        pp_size: int,
        transfer_backend: TransferBackend,
    ):
        self.token_to_kv_pool = token_to_kv_pool
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        # MLA 后端无 head_num 属性，kv_args 配置有所不同
        self.is_mla_backend = is_mla_backend(token_to_kv_pool)
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        # TP/PP 拓扑信息
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.gpu_id = gpu_id
        self.bootstrap_port = bootstrap_port
        self.queue: List[Req] = []    # 正在握手的请求队列
        self.gloo_group = gloo_group  # 跨 TP rank 同步 gloo 进程组
        self.max_total_num_tokens = max_total_num_tokens
        self.scheduler = scheduler
        self.transfer_backend = transfer_backend
        # MLA 不支持 staging buffer
        if envs.SGLANG_DISAGG_STAGING_BUFFER.get() and self.is_mla_backend:
            raise RuntimeError(
                "SGLANG_DISAGG_STAGING_BUFFER is designed for non-MLA models "
                "(e.g. GQA, MHA). MLA models should not set this flag."
            )
        # 初始化 CommonKVManager（含 RDMA 引擎、bootstrap 服务等）
        self.kv_manager = self._init_kv_manager()

        if self.scheduler.tp_worker.is_hybrid_swa:
            # FIXME: current SWA allocation allocate full kv cache size in prefill
            # SWA 模式下最大 token 数受限于 SWA 专属内存
            self.max_total_num_tokens = min(
                self.max_total_num_tokens,
                self.scheduler.tp_worker.model_runner.swa_max_total_num_tokens,
            )

    def _init_kv_manager(self) -> CommonKVManager:
        """
        初始化 Prefill 侧 CommonKVManager：配置 KV 缓冲区、元数据缓冲区、
        Mamba/SWA/NSA 状态缓冲区，并实例化对应传输后端的 KV 管理器。
        """
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()
        # Prefill 侧直接用 tp_rank 作为 engine_rank（不像 Decode 侧需要对 attn_tp_size 取模）
        kv_args.engine_rank = self.tp_rank
        kv_args.pp_rank = self.pp_rank
        kv_args.system_dp_rank = self.scheduler.dp_rank
        # PP 分层传输：只传本 PP rank 负责的层的 KV 数据
        kv_args.prefill_start_layer = self.token_to_kv_pool.start_layer
        # 获取 KV Cache 连续内存块地址列表（供 RDMA engine 注册）
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            self.token_to_kv_pool.get_contiguous_buf_infos()
        )

        if self.draft_token_to_kv_pool is not None:
            # We should also transfer draft model kv cache. The indices are
            # always shared with a target model.
            # Spec Decode 草稿模型 KV 缓冲区追加到主模型列表
            draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = (
                self.draft_token_to_kv_pool.get_contiguous_buf_infos()
            )
            kv_data_ptrs += draft_kv_data_ptrs
            kv_data_lens += draft_kv_data_lens
            kv_item_lens += draft_kv_item_lens

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens
        if not self.is_mla_backend:
            # GQA/MHA 需要告知 transfer engine 每层的 KV head 数
            kv_args.kv_head_num = self.token_to_kv_pool.head_num
            kv_args.total_kv_head_num = (
                self.scheduler.model_config.get_total_num_kv_heads()
            )
        kv_args.page_size = self.token_to_kv_pool.page_size

        # 配置元数据缓冲区（Prefill 侧写入首 token 等，Decode 侧读取）
        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
            self.metadata_buffers.get_buf_infos()
        )
        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.gpu_id

        if hasattr(self.token_to_kv_pool, "get_state_buf_infos"):
            # 获取混合模型附加状态缓冲区（Mamba/SWA/NSA）
            state_data_ptrs, state_data_lens, state_item_lens = (
                self.token_to_kv_pool.get_state_buf_infos()
            )
            kv_args.state_data_ptrs = state_data_ptrs
            kv_args.state_data_lens = state_data_lens
            kv_args.state_item_lens = state_item_lens

            if isinstance(self.token_to_kv_pool, SWAKVPool):
                kv_args.state_type = "swa"
            elif isinstance(self.token_to_kv_pool, HybridLinearKVPool):
                kv_args.state_type = "mamba"
                # Get state dimension info for cross-TP slice transfer
                if hasattr(self.token_to_kv_pool, "get_state_dim_per_tensor"):
                    kv_args.state_dim_per_tensor = (
                        self.token_to_kv_pool.get_state_dim_per_tensor()
                    )
            elif isinstance(self.token_to_kv_pool, NSATokenToKVPool):
                kv_args.state_type = "nsa"
                if self.draft_token_to_kv_pool is not None and isinstance(
                    self.draft_token_to_kv_pool, NSATokenToKVPool
                ):
                    # NSA 草稿模型状态也需要传输
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
            # 标准 Attention 模型无额外状态
            kv_args.state_data_ptrs = []
            kv_args.state_data_lens = []
            kv_args.state_item_lens = []
            kv_args.state_type = "none"

        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(
            kv_args,
            DisaggregationMode.PREFILL,
            self.scheduler.server_args,
            self.is_mla_backend,
        )
        # Pass KV pool tensor refs to the manager for GPU gather (staging mode)
        # staging 模式（异构 TP）：将 K/V buffer tensor 传入 manager 用于 GPU gather
        if (
            envs.SGLANG_DISAGG_STAGING_BUFFER.get()
            and hasattr(kv_manager, "set_kv_buffer_tensors")
            and not self.is_mla_backend
        ):
            kv_pool = self.token_to_kv_pool
            if hasattr(kv_pool, "full_kv_pool"):
                kv_pool = kv_pool.full_kv_pool
            if hasattr(kv_pool, "k_buffer") and hasattr(kv_pool, "v_buffer"):
                kv_manager.set_kv_buffer_tensors(
                    kv_pool.k_buffer,
                    kv_pool.v_buffer,
                    kv_pool.page_size,
                )
        return kv_manager

    def add(self, req: Req, num_kv_heads: int) -> None:
        """
        为请求创建 KV 发送器并加入 Bootstrap 队列。
        超容量请求直接拒绝，测试用 FAKE_BOOTSTRAP_HOST 使用伪传输后端。
        """
        if self._check_if_req_exceed_kv_capacity(req):
            return

        # 根据 bootstrap_host 选择真实或伪传输后端
        backend = (
            TransferBackend.FAKE
            if req.bootstrap_host == FAKE_BOOTSTRAP_HOST
            else self.transfer_backend
        )
        kv_sender_class = get_kv_class(backend, KVClassType.SENDER)

        dest_tp_ranks = [self.tp_rank]

        # 创建 KV 发送器，向 Decode 侧 bootstrap 服务发起握手
        req.disagg_kv_sender = kv_sender_class(
            mgr=self.kv_manager,
            bootstrap_addr=f"{req.bootstrap_host}:{self.bootstrap_port}",
            bootstrap_room=req.bootstrap_room,
            dest_tp_ranks=dest_tp_ranks,
            pp_rank=self.pp_rank,
        )
        self._process_req(req)
        self.queue.append(req)

    def extend(self, reqs: List[Req], num_kv_heads: int) -> None:
        """批量调用 add()，将多个请求加入 Bootstrap 队列。"""
        for req in reqs:
            self.add(req, num_kv_heads)

    def _check_if_req_exceed_kv_capacity(self, req: Req) -> bool:
        """若请求 input_ids 超过最大 token 容量，拒绝并立即输出错误响应。"""
        if len(req.origin_input_ids) > self.max_total_num_tokens:
            message = f"Request {req.rid} exceeds the maximum number of tokens: {len(req.origin_input_ids)} > {self.max_total_num_tokens}"
            logger.error(message)
            req.time_stats.trace_ctx.abort(abort_info={"reason": message})
            prepare_abort(req, message, status_code=HTTPStatus.BAD_REQUEST)
            self.scheduler.stream_output([req], req.return_logprob)
            return True
        return False

    def _process_req(self, req: Req) -> None:
        """
        Set max_new_tokens = 1, so PrefillAdder memory estimation is accurate
        将 max_new_tokens 设为 1，确保 PrefillAdder 内存估算准确
        （Prefill 侧只需生成 1 个 token，后续由 Decode 侧继续生成）。
        """
        req.sampling_params.max_new_tokens = 1

    def pop_bootstrapped(
        self,
        return_failed_reqs: bool = False,
        rids_to_check: Optional[List[str]] = None,
    ) -> List[Req]:
        """
        轮询队列中所有请求的握手状态，弹出已完成握手的请求移入 WaitingQueue。
        Bootstrapping → 继续等待；WaitingForInput → 分配元数据槽，移入 waiting_queue；
        Failed → 中止请求，可选返回给 PP 上层 rank。

        return_failed_reqs: For PP, on rank 0, also return the failed reqs to notify the next rank
        rids_to_check: For PP, on rank > 0, check the rids from the previous rank has consensus with the current rank.
        """

        bootstrapped_reqs = []
        failed_reqs = []
        indices_to_remove = set()

        if len(self.queue) == 0:
            if return_failed_reqs is False:
                return []
            else:
                return [], []

        # 跨 attn_cp 和 attn_tp 两个进程组 all-reduce 轮询，确保所有 rank 状态一致
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in self.queue],
            self.scheduler.attn_cp_cpu_group,
            self.scheduler.attn_tp_cpu_group,
        )

        for i, (req, poll) in enumerate(zip(self.queue, polls)):
            if rids_to_check is not None:
                # if req not in reqs_info_to_check, skip
                # PP 模式：仅处理上层 rank 已确认的 rids
                if req.rid not in rids_to_check:
                    continue

            if poll == KVPoll.Bootstrapping:
                continue  # 握手仍在进行，本轮跳过
            elif poll == KVPoll.Failed:
                # 握手失败：中止请求并输出错误响应
                error_message = f"Prefill bootstrap failed for request rank={self.tp_rank} {req.rid=} {req.bootstrap_room=}"
                try:
                    req.disagg_kv_sender.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.error(error_message)
                req.time_stats.trace_ctx.abort(abort_info={"reason": error_message})
                prepare_abort(
                    req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
                )
                self.scheduler.stream_output([req], req.return_logprob)
                indices_to_remove.add(i)
                failed_reqs.append(req)
                if self.scheduler.enable_metrics:
                    self.scheduler.metrics_collector.increment_bootstrap_failed_reqs()
                if self.scheduler.enable_hicache_storage:
                    # to release prefetch events associated with the request
                    # HiCache：释放与该请求相关的预取事件
                    self.scheduler.tree_cache.release_aborted_request(req.rid)
                continue

            # KV.WaitingForInput - init here
            # 握手成功（WaitingForInput）：分配元数据缓冲区槽位并初始化发送器
            req.time_stats.set_bootstrap_done_time()
            num_kv_indices = len(req.origin_input_ids)
            # 元数据缓冲区槽位不足时停止，等待下一轮释放
            if self.req_to_metadata_buffer_idx_allocator.available_size() == 0:
                break

            req.metadata_buffer_index = (
                self.req_to_metadata_buffer_idx_allocator.alloc()
            )
            assert req.metadata_buffer_index is not None

            # 计算本请求需要的 KV 页数，用于初始化发送器（通知 Decode 侧分配多大接收缓冲区）
            num_pages = kv_to_page_num(num_kv_indices, self.token_to_kv_pool.page_size)
            req.disagg_kv_sender.init(num_pages, req.metadata_buffer_index)

            bootstrapped_reqs.append(req)
            indices_to_remove.add(i)
            req.time_stats.set_wait_queue_entry_time()

        # 从队列中移除已处理的请求
        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        if return_failed_reqs is False:
            return bootstrapped_reqs
        else:
            return bootstrapped_reqs, failed_reqs


class SchedulerDisaggregationPrefillMixin:
    """
    Prefill 侧调度器 Mixin，提供 PD 分离架构下 Prefill Server 的事件循环和核心调度逻辑。
    包含 normal 和 overlap 两种事件循环、Prefill 结果处理（KV 传输启动）、
    inflight 队列轮询，以及 chunked prefill 和 send_kv_chunk 分块发送逻辑。
    """

    def maybe_prefetch_staging_for_batch(self: Scheduler, batch: ScheduleBatch) -> None:
        """Pre-send STAGING_REQ so decode allocates staging during GPU forward.
        在 GPU forward 开始前预先向 Decode 侧发送 STAGING_REQ，
        使 Decode 侧能够在 Prefill 计算期间提前分配 staging buffer。
        """
        kv_mgr = self.disagg_prefill_bootstrap_queue.kv_manager
        prefetch = getattr(kv_mgr, "_prefetch_staging_reqs", None)
        if prefetch is None:
            return
        for req in batch.reqs:
            room = getattr(req, "bootstrap_room", None)
            if room is not None and room in kv_mgr.transfer_infos:
                prefetch(room)

    def get_next_disagg_prefill_batch_to_run(
        self: Scheduler,
    ) -> Optional[ScheduleBatch]:
        """
        调度下一个 Prefill batch：重置 batch_is_full 标志、处理 chunked req、
        获取新 batch 并可选添加 MLP sync batch。
        """
        # HACK (byronhsu): reset the batch_is_full flag because we never enter update_running_batch which resets it
        # Otherwise, it hangs under high concurrency
        # Prefill 侧不调用 update_running_batch，手动重置防止高并发时卡死
        self.running_batch.batch_is_full = False

        # 处理上一轮的 chunked req：缓存前缀树并发送 KV 分块
        self.process_prefill_chunk()

        # 从 waiting_queue 中取出请求构建新的 extend batch
        batch = self.get_new_batch_prefill()
        # DP attention 需要 MLP sync batch 同步各 rank 的 MLP 输出
        batch = self.maybe_prepare_mlp_sync_batch(batch)

        if batch:
            set_schedule_time_batch(batch)

        return batch

    @torch.no_grad()
    def event_loop_normal_disagg_prefill(self: Scheduler) -> None:
        """Prefill 侧串行事件循环：接收请求 → Bootstrap 轮询 → 调度批次 → 运行 Prefill → 处理 inflight 队列。"""
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()

        while True:
            # Receive requests
            # 接收 TokenizerManager 下发的新请求，加入 Bootstrap 队列
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            # 将握手完成的请求从 BootstrapQueue 移入 WaitingQueue
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )
            if self._engine_paused:
                continue

            # Get the next batch to run
            # 调度下一个 Prefill batch（含 chunked prefill 处理）
            batch = self.get_next_disagg_prefill_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                if self.enable_staging:
                    # staging 模式：在 GPU forward 前预发送 STAGING_REQ
                    self.maybe_prefetch_staging_for_batch(batch)
                result = self.run_batch(batch)
                # 处理 Prefill 结果：启动 KV 传输并将请求加入 inflight 队列
                self.process_batch_result(batch, result)
            else:
                self.on_idle()

            # 轮询 inflight 队列：将 KV 传输完成的请求响应给客户端
            self.process_disagg_prefill_inflight_queue()

            # Update last_batch
            self.last_batch = batch

    @torch.no_grad()
    def event_loop_overlap_disagg_prefill(self: Scheduler) -> None:
        """
        Prefill 侧流水线事件循环：当前批次 GPU 计算与上一批次结果处理重叠，
        提高 GPU 利用率。
        """
        self.result_queue = deque()
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()

        while True:
            # Receive requests
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )
            if self._engine_paused:
                continue

            # Get the next batch to run
            batch = self.get_next_disagg_prefill_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            # 提交当前批次 GPU 计算（异步），结果入 result_queue
            if batch:
                if self.enable_staging:
                    self.maybe_prefetch_staging_for_batch(batch)
                batch_result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None

            # Process the last batch
            # 处理上一批次结果（与当前批次 GPU 计算重叠）
            if self.last_batch:
                tmp_batch, tmp_result = self.result_queue.popleft()
                self.process_batch_result(tmp_batch, tmp_result)
            elif batch is None:
                # When the server is idle, do self-check and re-init some states
                self.on_idle()

            # 轮询 inflight 队列
            self.process_disagg_prefill_inflight_queue()

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            self.launch_batch_sample_if_needed(batch_result)

            # Update last_batch
            self.last_batch = batch

    def process_batch_result_disagg_prefill(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        """
        Prefill 批次结果处理：记录首 token、更新前缀树、启动 KV 传输、处理 logprobs 和 grammar。
        Transfer kv for prefill completed requests and add it into disagg_prefill_inflight_queue
        Adapted from process_batch_result_prefill
        """
        (
            logits_output,
            next_token_ids,
            extend_input_len_per_req,
            extend_logprob_start_len_per_req,
            copy_done,
        ) = (
            result.logits_output,
            result.next_token_ids,
            result.extend_input_len_per_req,
            result.extend_logprob_start_len_per_req,
            result.copy_done,
        )

        # 同步 GPU 到 CPU 的数据拷贝（logits 等）
        if copy_done is not None:
            copy_done.synchronize()
        if result.routed_experts_output is not None:
            result.routed_experts_output.finalize()
            result.routed_experts_output = None

        logprob_pt = 0
        # Transfer kv for prefill completed requests and add it into disagg_prefill_inflight_queue
        next_token_ids = result.next_token_ids.tolist()
        # 预先转换为 Python list，方便按请求逐个取值
        if batch.return_logprob:
            if logits_output.next_token_logprobs is not None:
                logits_output.next_token_logprobs = (
                    logits_output.next_token_logprobs.tolist()
                )
            if logits_output.input_token_logprobs is not None:
                logits_output.input_token_logprobs = tuple(
                    logits_output.input_token_logprobs.tolist()
                )

        for i, (req, next_token_id) in enumerate(
            zip(batch.reqs, next_token_ids, strict=True)
        ):
            if req.is_chunked <= 0:
                # 非 chunked req 或最后一个 chunk：Prefill 阶段完成
                req.time_stats.set_prefill_finished_time()

                # There is no output_ids for prefill
                # Prefill 只生成 1 个首 token，加入 output_ids
                req.output_ids.append(next_token_id)
                self.tree_cache.cache_unfinished_req(req)  # update the tree and lock
                # 加入 inflight 队列，等待 KV 传输完成后响应客户端
                self.disagg_prefill_inflight_queue.append(req)
                if self.spec_algorithm.is_eagle() and batch.spec_info is not None:
                    # EAGLE speculative decoding：保存草稿 top-k 概率和隐层状态
                    req.output_topk_p = batch.spec_info.topk_p[i]
                    req.output_topk_index = batch.spec_info.topk_index[i]
                    req.hidden_states_tensor = (
                        batch.spec_info.hidden_states[i].cpu().clone()
                    )
                else:
                    req.hidden_states_tensor = None
                if req.return_logprob:
                    assert extend_logprob_start_len_per_req is not None
                    assert extend_input_len_per_req is not None
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    num_input_logprobs = extend_input_len - extend_logprob_start_len
                    # 计算并附加 input/output log probabilities
                    self.add_logprob_return_values(
                        i,
                        req,
                        logprob_pt,
                        next_token_ids,
                        num_input_logprobs,
                        logits_output,
                    )
                    logprob_pt += num_input_logprobs
                # 启动最后一个 KV chunk 的传输（last_chunk=True 触发元数据写入）
                self.send_kv_chunk(req, last_chunk=True)
                req.time_stats.set_prefill_transfer_queue_entry_time()

                if req.grammar is not None:
                    # FIXME: this try-except block is for handling unexpected xgrammar issue.
                    try:
                        req.grammar.accept_token(next_token_id)
                    except ValueError as e:
                        # Grammar accept_token can raise ValueError if the token is not in the grammar.
                        # This can happen if the grammar is not set correctly or the token is invalid.
                        # grammar 约束 token 无效时中止请求
                        error_message = f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                        release_kv_cache(req, self.tree_cache)
                        prepare_abort(
                            req,
                            error_message,
                            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                        )
                    req.grammar.finished = req.finished()
            else:
                # being chunked reqs' prefill is not finished
                # chunked prefill 中间 chunk：递减计数器，不触发 KV 传输
                req.is_chunked -= 1

                if req.return_logprob:
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    if extend_logprob_start_len < extend_input_len:
                        # Update input logprobs.
                        # 非最后 chunk 的 input logprob 更新
                        num_input_logprobs = extend_input_len - extend_logprob_start_len
                        self.add_input_logprob_return_values(
                            i,
                            req,
                            logits_output,
                            logprob_pt,
                            num_input_logprobs,
                            last_prefill_chunk=False,
                        )
                        logprob_pt += num_input_logprobs

                if self.enable_overlap:
                    # overlap 模式：在 process_batch_result 中延迟发送中间 chunk
                    self.send_kv_chunk(req, last_chunk=False, end_idx=req.tmp_end_idx)
                req.time_stats.set_last_chunked_prefill_finish_time()

        can_run_cuda_graph = getattr(result, "can_run_cuda_graph", False)
        # 上报 Prefill 统计信息（token 数、CUDA graph 使用情况等）
        self.report_prefill_stats(
            prefill_stats=batch.prefill_stats,
            can_run_cuda_graph=can_run_cuda_graph,
            dp_cooperation_info=batch.dp_cooperation_info,
        )

    def process_disagg_prefill_inflight_queue(
        self: Scheduler, rids_to_check: Optional[List[str]] = None
    ) -> List[Req]:
        """
        轮询 inflight 队列中所有请求的 KV 传输状态。
        传输成功 → 解锁前缀树、设置 FINISH_LENGTH、响应客户端；
        传输失败 → 中止请求并响应客户端。
        rids_to_check: For PP, on rank > 0, check the rids from the previous rank has consensus with the current rank.
        """
        if len(self.disagg_prefill_inflight_queue) == 0:
            return []

        done_reqs = []

        # 跨 attn_cp 和 attn_tp 两个进程组 all-reduce 轮询 KV 传输状态
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in self.disagg_prefill_inflight_queue],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )

        undone_reqs: List[Req] = []
        # Check .poll() for the reqs in disagg_prefill_inflight_queue. If Success, respond to the client and remove it from the queue
        for req, poll in zip(self.disagg_prefill_inflight_queue, polls):

            if rids_to_check is not None:
                if req.rid not in rids_to_check:
                    undone_reqs.append(req)
                    continue

                # In PP mode, the previous rank may have reached a terminal
                # state (Success/Failed) while this rank's local poll is still
                # in a transient state due to clock skew or propagation delay.
                # Treat non-terminal states as undone instead of crashing.
                # PP 模式：上层 rank 已达到终止状态，本 rank 可能因时钟偏差仍处于中间状态
                if poll not in (
                    KVPoll.Success,
                    KVPoll.Failed,
                ):
                    logger.warning(
                        f"PP rank {self.pp_rank}: unexpected poll state {poll} for rid {req.rid} "
                        f"from consensus; treating as undone"
                    )
                    undone_reqs.append(req)
                    continue

            if poll in [KVPoll.WaitingForInput, KVPoll.Transferring]:
                # KV 传输仍在进行中，保留在 inflight 队列
                undone_reqs.append(req)
            elif poll == KVPoll.Success:  # transfer done
                # KV 传输完成：解锁前缀树（允许其他请求使用该 KV 缓存）
                release_kv_cache(req, self.tree_cache)  # unlock the tree
                req.finished_reason = FINISH_LENGTH(length=0)
                # FIXME: clean up req's data in transfer engine
                if hasattr(req.disagg_kv_sender, "clear"):
                    req.disagg_kv_sender.clear()
                done_reqs.append(req)
                req.time_stats.set_prefill_kv_transfer_finish_time()
            elif poll == KVPoll.Failed:
                # KV 传输失败：中止请求，解锁前缀树
                error_message = f"Prefill transfer failed for request rank={self.tp_rank} {req.rid=} {req.bootstrap_room=}"
                try:
                    req.disagg_kv_sender.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.warning(error_message)
                req.time_stats.trace_ctx.abort(abort_info={"reason": error_message})
                release_kv_cache(req, self.tree_cache)  # unlock the tree
                prepare_abort(
                    req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
                )
                done_reqs.append(req)
                if self.enable_metrics:
                    self.metrics_collector.increment_transfer_failed_reqs()
            else:
                # 非预期状态（如 Bootstrapping）：打印 warning 并保留在队列
                logger.warning(
                    f"Unexpected polling state {poll} for rid {req.rid} in inflight queue; "
                    f"treating as undone"
                )
                undone_reqs.append(req)

        # 记录所有完成请求的最终完成时间戳
        for req in done_reqs:
            req.time_stats.set_completion_time()

        # 计算 KV 传输速度和延迟统计所需的字节数
        page_size = self.token_to_kv_pool_allocator.page_size
        kv_item_lens = (
            self.disagg_prefill_bootstrap_queue.kv_manager.kv_args.kv_item_lens
        )
        bytes_per_page_all_layers = sum(kv_item_lens)

        for req in done_reqs:
            if isinstance(req.finished_reason, FINISH_ABORT):
                continue
            # 计算并上报 KV 传输延迟和吞吐量指标
            metrics = req.time_stats.compute_and_observe_kv_transfer_metrics(
                num_tokens=len(req.origin_input_ids),
                page_size=page_size,
                bytes_per_page_all_layers=bytes_per_page_all_layers,
            )
            if metrics:
                # Update last-value for REST API
                # 更新 REST API 可查询的最新 KV 传输指标
                if "latency_ms" in metrics:
                    self.kv_transfer_latency_ms = metrics["latency_ms"]
                if "speed_gb_s" in metrics:
                    self.kv_transfer_speed_gb_s = metrics["speed_gb_s"]

        # Stream requests which have finished transfer
        # 将已完成传输的请求响应给客户端（通过 TokenizerManager 流式输出）
        self.stream_output(
            done_reqs,
            any(req.return_logprob for req in done_reqs),
            None,
        )
        for req in done_reqs:
            req: Req

            # 释放元数据缓冲区槽位，供下一个请求使用
            release_req_to_metadata_buffer(
                req, self.req_to_metadata_buffer_idx_allocator
            )

        self.disagg_prefill_inflight_queue = undone_reqs

        return done_reqs

    def get_transferred_rids(self: Scheduler) -> List[str]:
        """
        Used by PP, get the transferred rids but **do not pop**
        PP 模式专用：查询 inflight 队列中已完成 KV 传输的请求 rid 列表，但不弹出（仅查询）。
        """
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in self.disagg_prefill_inflight_queue],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )

        transferred_rids: List[str] = []

        for req, poll in zip(self.disagg_prefill_inflight_queue, polls):
            # 传输成功或失败都视为已完成，上层 rank 需要收集这些 rid
            if poll == KVPoll.Success or poll == KVPoll.Failed:
                transferred_rids.append(req.rid)

        return transferred_rids

    def process_prefill_chunk(self: Scheduler) -> None:
        """
        处理上一轮的 chunked req：将其状态缓存到前缀树并发送对应的 KV chunk。
        同时从 last_batch 中过滤已结束的 chunked req，维护 running_batch 状态。
        """
        chunked_req_to_exclude = set()
        if self.chunked_req:
            chunked_req_to_exclude.add(self.chunked_req)
            # 缓存 chunked req 的前缀树节点（带 chunked=True 标记，不设置 finished 锁）
            self.tree_cache.cache_unfinished_req(self.chunked_req, chunked=True)
            if self.enable_overlap:
                # Delay KV transfer to process_batch_result_disagg_prefill when overlap is enabled to ensure results are resolved
                # overlap 模式：延迟 KV 发送到 process_batch_result 中，确保 logits 已解析
                self.chunked_req.tmp_end_idx = min(
                    len(self.chunked_req.fill_ids),
                    len(self.chunked_req.origin_input_ids),
                )
            else:
                # 非 overlap 模式：立即发送当前 chunk 的 KV 数据
                self.send_kv_chunk(self.chunked_req)
            self.running_batch.batch_is_full = False

        if self.last_batch and self.last_batch.forward_mode.is_extend():
            if self.last_batch.chunked_req:
                # In the context pipeline parallelism, after the last chunk, the current microbatch still track outdated chunked_req.
                # We need to discard it.
                # CP（Context Parallelism）场景：最后一个 chunk 后需要丢弃旧的 chunked_req 引用
                chunked_req_to_exclude.add(self.last_batch.chunked_req)

            last_bs = self.last_batch.batch_size()
            self.last_batch.filter_batch(
                chunked_req_to_exclude=list(chunked_req_to_exclude)
            )
            if self.last_batch.batch_size() < last_bs:
                self.running_batch.batch_is_full = False

    def send_kv_chunk(
        self: Scheduler,
        req: Req,
        last_chunk: bool = False,
        end_idx: Optional[int] = None,
    ) -> None:
        """
        将 Prefill 完成的 KV Cache 分块发送到 Decode 侧。
        非最后 chunk 时，对尾部不完整页延迟到下次发送；最后 chunk 时同时写入元数据缓冲区。
        """
        page_size = self.token_to_kv_pool_allocator.page_size
        start_idx = req.start_send_idx
        # 确定本次发送的结束位置（fill_ids 与 origin_input_ids 之间取较小值）
        end_idx = (
            end_idx
            if end_idx is not None
            else min(len(req.fill_ids), len(req.origin_input_ids))
        )

        if not last_chunk:
            # if not the last chunk and the last page is partial, delay the last partial page to the next send
            # 非最后 chunk：将末尾不完整页延迟到下次发送，确保按页对齐传输
            end_idx = end_idx - end_idx % page_size

        # 读取 [start_idx, end_idx) 范围的 KV token 索引
        kv_indices = (
            self.req_to_token_pool.req_to_token[req.req_pool_idx, start_idx:end_idx]
            .cpu()
            .numpy()
        )
        req.start_send_idx = end_idx
        state_indices = None
        if last_chunk:
            # 最后一个 chunk：写入元数据缓冲区（首 token、cached_tokens 等）供 Decode 侧读取
            self.disagg_metadata_buffers.set_buf(req)

            # Prepare extra pool indices for hybrid models
            # 为混合模型附加状态索引（Mamba/SWA/NSA）
            if isinstance(
                self.token_to_kv_pool_allocator.get_kvcache(), HybridLinearKVPool
            ):
                # Mamba hybrid model: send single mamba state index
                # Mamba 混合模型：发送 SSM 状态索引
                state_indices = [
                    self.req_to_token_pool.req_index_to_mamba_index_mapping[
                        req.req_pool_idx
                    ]
                    .cpu()
                    .numpy()
                ]
            elif isinstance(self.token_to_kv_pool_allocator.get_kvcache(), SWAKVPool):
                # SWA hybrid model: send last window KV indices
                # SWA 模式：只发送滑动窗口内的 KV 索引
                seq_len = len(req.fill_ids)
                window_size = self.sliding_window_size
                window_start = max(0, seq_len - window_size)
                window_start = (window_start // page_size) * page_size

                window_kv_indices_full = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, window_start:seq_len
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
            elif isinstance(
                self.token_to_kv_pool_allocator.get_kvcache(), NSATokenToKVPool
            ):
                # NSA 稀疏 Attention：发送完整序列的 KV 页索引
                seq_len = len(req.fill_ids)
                kv_indices_full = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, :seq_len
                ]
                state_indices = kv_indices_full.cpu().numpy()
                state_indices = kv_to_page_indices(state_indices, page_size)

        # 将 token 索引转换为页级索引
        page_indices = kv_to_page_indices(kv_indices, page_size)
        if len(page_indices) == 0:
            # 空页索引说明该 chunk 没有新数据需要发送（可能是纯 prefix cache 命中）
            logger.info(
                f"Skip sending kv chunk for request {req.rid=} {req.bootstrap_room=} because page_indices is empty"
            )
            return
        # 通过 KV 发送器将页索引和状态索引发送给 Decode 侧
        req.disagg_kv_sender.send(page_indices, state_indices)
