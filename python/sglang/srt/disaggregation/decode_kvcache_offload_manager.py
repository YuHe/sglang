# decode 侧 KV 缓存卸载管理器：将 decode 过程中增量生成的 KV 缓存从 GPU 卸载到 Host 内存并备份到存储
# 与 HiCache 控制器配合，实现层级化 KV 缓存管理（GPU -> Host -> Storage）
from __future__ import annotations

import json
import logging
import threading
import time
from typing import TYPE_CHECKING

import torch

from sglang.srt.disaggregation.kv_events import OffloadedState
from sglang.srt.environ import envs
# HiCache 控制器：负责 GPU <-> Host <-> Storage 三级 KV 缓存的异步读写
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
# Host 内存池：在 CPU 内存中存储卸载的 KV 缓存
from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


# decode 侧 KV 缓存卸载管理器：跟踪每个请求的卸载状态和备份进度
class DecodeKVCacheOffloadManager:
    """Manage decode-side KV cache offloading lifecycle and operations."""

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tp_group: torch.distributed.ProcessGroup,
        tree_cache: BasePrefixCache,
        server_args: ServerArgs,
    ) -> None:
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = server_args.page_size
        self.server_args = server_args
        # 请求计数器，用于生成唯一的 ack_id
        self.request_counter = 0
        self.tree_cache = tree_cache
        # 从环境变量读取卸载步长（stride），确保按 page_size 对齐
        env_stride = envs.SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE.get()
        if env_stride is None or env_stride <= 0:
            self.offload_stride = self.page_size
        else:
            self.offload_stride = max(
                self.page_size, (env_stride // self.page_size) * self.page_size
            )
        # 根据 KV 缓存类型（MHA 或 MLA）创建对应的 Host 内存池
        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        if isinstance(kv_cache, MHATokenToKVPool):
            self.decode_host_mem_pool = MHATokenToKVPoolHost(
                kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
            )
        elif isinstance(kv_cache, MLATokenToKVPool):
            # MLA（多头潜在注意力）专用 Host 内存池
            self.decode_host_mem_pool = MLATokenToKVPoolHost(
                kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
            )
        else:
            raise ValueError("Unsupported KV cache type for decode offload")

        self.tp_group = tp_group
        # TP 并行世界大小，用于跨 TP rank 同步卸载进度
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        # 解析 HiCache 存储后端额外配置（JSON 格式）
        hicache_storage_backend_extra_config = {}
        if server_args.hicache_storage_backend_extra_config:
            try:
                hicache_storage_backend_extra_config = json.loads(
                    server_args.hicache_storage_backend_extra_config
                )
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid hicache storage backend extra config JSON: {e}"
                )

        # 创建 HiCache 控制器，负责 device->host->storage 的异步 KV 缓存迁移
        self.cache_controller = HiCacheController(
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            mem_pool_host=self.decode_host_mem_pool,
            page_size=self.page_size,
            tp_group=tp_group,
            io_backend=server_args.hicache_io_backend,
            load_cache_event=threading.Event(),
            storage_backend=server_args.hicache_storage_backend,
            model_name=server_args.served_model_name,
            storage_backend_extra_config=hicache_storage_backend_extra_config,
        )

        # ongoing_offload: ack_id -> (req, host_indices, tokens, start_time, start, end)
        self.ongoing_offload = {}
        # ongoing_backup: ack_id -> (req_id, host_indices, start_time)
        self.ongoing_backup = {}
        # offloaded_state: req_id -> OffloadedState（记录已卸载的 prefill/inc 长度和最后 hash）
        self.offloaded_state = {}
        logger.info("Enable offload kv cache for decode side")

    def offload_kv_cache(self, req) -> bool:
        """Offload incremental KV cache for decode side."""
        # 将请求增量生成的 KV 缓存异步卸载到 Host 内存
        if self.cache_controller is None or self.decode_host_mem_pool is None:
            return False

        if req.req_pool_idx == -1 or len(req.output_ids) == 0:
            return False

        token_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx]
        if token_indices.dim() == 0 or token_indices.numel() == 0:
            return False

        # Prefill side offloads page-aligned origin_input_ids, decode side offloads the incremental part
        # 合并 origin_input_ids 和 output_ids（去掉最后一个尚未确认的 token）
        all_tokens = req.origin_input_ids + req.output_ids[:-1]
        # prefill 侧已卸载的对齐长度（按 page_size 取整）
        prefill_offloaded_len = (
            len(req.origin_input_ids) // self.page_size * self.page_size
        )
        # 获取或创建该请求的卸载状态追踪对象
        state = self.offloaded_state.get(req.rid)
        if state is None:
            # 首次卸载：计算 prefill 部分的 hash 链，作为增量 hash 的起点
            prefill_hashes = self._compute_prefix_hash(
                req.origin_input_ids[:prefill_offloaded_len]
            )
            last_prefill_hash = (
                prefill_hashes[-1] if prefill_offloaded_len > 0 else None
            )
            state = OffloadedState(
                prefill_len=prefill_offloaded_len,
                inc_len=0,
                last_hash=last_prefill_hash,
            )
            self.offloaded_state[req.rid] = state
        # 计算本次可卸载的新增对齐长度
        incremental_total = len(all_tokens) - state.prefill_len
        incremental_new = incremental_total - state.inc_len
        incremental_aligned_len = (
            incremental_new // self.offload_stride * self.offload_stride
        )

        if incremental_aligned_len == 0:
            # 不足一个 stride 的增量，暂不卸载
            return False

        # Extract incremental tokens and indices for the newly available chunk
        # 提取本次卸载的 token 序列和对应的 GPU 内存索引
        start = state.prefill_len + state.inc_len
        end = start + incremental_aligned_len
        incremental_tokens = all_tokens[start:end]
        incremental_indices = token_indices[start:end]

        # Early free prefill-offloaded GPU memory
        # 首次增量卸载时提前释放 prefill 部分占用的 GPU 显存
        if state.prefill_len > 0 and state.inc_len == 0:
            self.token_to_kv_pool_allocator.free(token_indices[: state.prefill_len])

        # Asynchronously offload incremental KV cache from device to host
        # 提交异步卸载任务，获取 ack_id 用于后续进度跟踪
        self.request_counter += 1
        ack_id = self.request_counter
        host_indices = self.cache_controller.write(
            device_indices=incremental_indices.long(),
            node_id=ack_id,
        )
        if host_indices is None:
            logger.error(f"Not enough host memory for request {req.rid}")
            return False

        # 记录进行中的卸载任务
        self.ongoing_offload[ack_id] = (
            req,
            host_indices,
            incremental_tokens,
            time.time(),
            start,
            end,
        )
        state.inc_len += incremental_aligned_len
        return True

    def check_offload_progress(self):
        """Check the progress of offload from device to host and backup from host to storage."""
        # 跨 TP rank 同步：取所有 rank 的最小完成数，确保一致性
        cc = self.cache_controller

        qsizes = torch.tensor(
            [
                len(cc.ack_write_queue),
                cc.ack_backup_queue.qsize(),
            ],
            dtype=torch.int,
        )
        if self.tp_world_size > 1:
            # 通过 all_reduce MIN 操作确保所有 TP rank 的进度同步
            torch.distributed.all_reduce(
                qsizes, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )

        n_write, n_backup = map(int, qsizes.tolist())
        # 分别检查 device->host 卸载进度和 host->storage 备份进度
        self._check_offload_progress(n_write)
        self._check_backup_progress(n_backup)

    def _check_offload_progress(self, finish_count):
        """Check the progress of offload from device to host."""
        # 处理 finish_count 个已完成的 device->host 卸载任务
        while finish_count > 0:
            _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
            # 等待 CUDA 异步拷贝完成
            finish_event.synchronize()
            for ack_id in ack_list:
                (
                    req,
                    host_indices,
                    incremental_tokens,
                    start_time,
                    start,
                    end,
                ) = self.ongoing_offload.pop(ack_id)

                if req.finished():
                    # 请求已完成，释放剩余 GPU KV 缓存
                    self._release_finished_req(req, start)
                else:
                    # 请求未完成，仅释放已卸载到 host 的 GPU 部分
                    kv_indices = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, start:end
                    ]
                    self.token_to_kv_pool_allocator.free(kv_indices)

                # 触发 host->storage 异步备份，获取最新 hash 值
                prior_hash = (
                    self.offloaded_state[req.rid].last_hash
                    if req.rid in self.offloaded_state
                    else None
                )
                last_hash = self._trigger_backup(
                    req, host_indices, incremental_tokens, start_time, prior_hash
                )
                if req.rid in self.offloaded_state:
                    self.offloaded_state[req.rid].last_hash = last_hash
            finish_count -= 1

    def _release_finished_req(self, req: Req, start_offset: int):
        # 释放已完成请求的剩余 KV 缓存（增量部分）以及过度分配的缓存槽
        kv_committed_len = req.pop_committed_kv_cache()
        start = start_offset
        end = kv_committed_len
        # Free the incremental part of the request (NSA-aware)
        kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, start:end]
        self.token_to_kv_pool_allocator.free(kv_indices)

        # Free over-allocated KV cache slots (e.g. from speculative decoding v2).
        # Without spec v2, start_p == end_p so this is a no-op.
        # 释放推测解码 v2 中过度分配的 KV 缓存槽（对齐到 page_size）
        start_p, end_p = req.pop_overallocated_kv_cache()
        if self.page_size > 1:
            start_p = ceil_align(start_p, self.page_size)
        if start_p < end_p:
            overalloc_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, start_p:end_p
            ]
            self.token_to_kv_pool_allocator.free(overalloc_indices)

        # 释放请求池槽位和前缀缓存保护计数
        self.req_to_token_pool.free(req)
        self.tree_cache.protected_size_ -= len(req.prefix_indices)
        # 清理卸载状态记录
        if req.rid in self.offloaded_state:
            del self.offloaded_state[req.rid]

    def _check_backup_progress(self, finish_count):
        """Check the progress of backup from host to storage."""
        # 处理 finish_count 个已完成的 host->storage 备份任务
        for _ in range(finish_count):
            storage_operation = self.cache_controller.ack_backup_queue.get()
            ack_id = storage_operation.id
            req_id, host_indices, start_time = self.ongoing_backup.pop(ack_id)

            # Release host memory
            # 备份完成后释放 host 内存池中的对应槽位
            self.decode_host_mem_pool.free(host_indices)

            logger.debug(
                f"Finished backup request {req_id}, free host memory, len:{len(host_indices)}, cost time:{time.time() - start_time:.2f} seconds."
            )

    def _trigger_backup(
        self, req, host_indices, incremental_tokens, start_time, prior_hash
    ):
        """Trigger async backup from host to storage."""
        # 计算增量 token 的前缀 hash，用于存储侧的 KV 缓存命中查找
        page_hashes = self._compute_prefix_hash(incremental_tokens, prior_hash)
        # 提交 host->storage 异步备份任务
        ack_id = self.cache_controller.write_storage(
            host_indices,
            incremental_tokens,
            hash_value=page_hashes,
        )
        self.ongoing_backup[ack_id] = (req.rid, host_indices, start_time)
        # 返回最后一个 page 的 hash，供下次增量卸载续接
        return page_hashes[-1] if len(page_hashes) > 0 else prior_hash

    def _compute_prefix_hash(self, tokens, prior_hash=""):
        # 按 page_size 步长逐页计算 token 序列的前缀 hash（链式 hash）
        page_hashes = []
        last_hash = prior_hash
        for offset in range(0, len(tokens), self.page_size):
            page_tokens = tokens[offset : offset + self.page_size]
            last_hash = self.cache_controller.get_hash_str(page_tokens, last_hash)
            page_hashes.append(last_hash)
        return page_hashes

    def finalize_release_on_finish(self, req: Req):
        """Free any remaining tail KV that was not offloaded due to non-aligned length."""
        # 请求完成时，释放因非对齐长度而未被卸载的尾部 KV 缓存
        if req.req_pool_idx == -1:
            return
        state = self.offloaded_state.get(req.rid)
        if state is None:
            # 无卸载记录时，计算理论上应卸载的 prefill 对齐长度
            prefill_len = len(req.origin_input_ids) // self.page_size * self.page_size
            inc_len = 0
        else:
            prefill_len = state.prefill_len
            inc_len = state.inc_len
        # If no incremental offload ever happened, the prefill-aligned part was never freed.
        # Free the prefill portion on request finish to avoid leaks.
        # 若从未发生增量卸载，此处补充释放 prefill 对齐部分的 GPU 显存，避免内存泄漏
        if prefill_len > 0 and inc_len == 0:
            token_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx]
            self.token_to_kv_pool_allocator.free(token_indices[:prefill_len])
            logger.info(
                f"Finalize release: freed prefill-aligned KV for req {req.rid}, len:{prefill_len}"
            )
        start_offset = prefill_len + inc_len
        # 释放剩余的尾部 KV 缓存（非对齐部分）
        self._release_finished_req(req, start_offset)
