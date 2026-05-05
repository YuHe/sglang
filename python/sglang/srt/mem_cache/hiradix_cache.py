from __future__ import annotations

# atexit：进程退出时自动注册清理函数；heapq：用于驱逐堆；json/os：配置文件解析
import atexit
import heapq
import json
import logging
import os
# threading：保护主机内存访问的并发控制；time：用于超时判断和时间戳
import threading
import time
from queue import Empty
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

# StorageMedium：区分 GPU/CPU 存储层级（用于 KV 事件）
from sglang.srt.disaggregation.kv_events import StorageMedium
# HiCacheController：管理 GPU↔CPU 的 DMA 传输；PrefetchOperation：L3 预取操作句柄
from sglang.srt.managers.cache_controller import HiCacheController, PrefetchOperation
# 基础前缀缓存参数/结果数据类，HiRadixCache 继承并扩展它们
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InitLoadBackParams,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
# hicache_storage：存储后端的池命中策略、池名、池传输描述
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
# HybridCacheController：支持多池（MHA+NSA）的混合控制器
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
# attach_hybrid_nsa_pool_to_hiradix_cache：为 NSA 模型动态挂载主机池
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    attach_hybrid_nsa_pool_to_hiradix_cache,
)
# GPU 端 KV 缓存池（MHA/MLA/NSA 三种架构）
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    NSATokenToKVPool,
)
# 主机端（CPU）KV 缓存池，用于 L2 层 offload
from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
)
# 继承 RadixCache 以及复用节点哈希计算工具
from sglang.srt.mem_cache.radix_cache import (
    RadixCache,
    RadixKey,
    TreeNode,
    compute_node_hash_values,
    split_node_hash_value,
)
# 存储层指标采集器（可选，仅在开启 metrics 时激活）
from sglang.srt.observability.metrics_collector import StorageMetricsCollector

# 类型检查期间导入，避免循环依赖
if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


# HiRadixCache：分层前缀缓存，在 RadixCache 基础上增加 CPU 主机内存（L2）和可选存储后端（L3）
class HiRadixCache(RadixCache):

    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        # 保存 metrics 开关（延迟到存储后端挂载后再决定是否启用）
        self._enable_metrics_flag = params.enable_metrics

        self.page_size = params.page_size
        self.kv_cache = params.token_to_kv_pool_allocator.get_kvcache()

        # 根据 GPU KV 缓存类型选择对应的 CPU 主机池（MHA/NSA/MLA 三种架构）
        if isinstance(self.kv_cache, MHATokenToKVPool):
            self.token_to_kv_pool_host = MHATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
        elif isinstance(self.kv_cache, NSATokenToKVPool):
            # NSA 的主机池在解析 extra_config 后由 attach_hybrid_nsa_pool_to_hiradix_cache 填入
            self.token_to_kv_pool_host = None
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            self.token_to_kv_pool_host = MLATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
        else:
            raise ValueError(
                "HiRadixCache only supports MHA, MLA, and NSA (DSA) models"
            )

        # TP/PP 分布式通信组，用于跨进程同步写穿/加载进度
        self.tp_group = params.tp_cache_group
        self.attn_cp_group = params.attn_cp_cache_group
        self.attn_tp_group = params.attn_tp_cache_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
        self.pp_rank = params.pp_rank
        self.pp_size = params.pp_size
        # 是否启用 L3 存储后端（nixl 等远程存储）
        self.enable_storage = server_args.hicache_storage_backend is not None
        self.enable_storage_metrics = self.enable_storage and params.enable_metrics
        self.extra_metric_labels = server_args.extra_metric_labels

        # 解析 L3 存储后端的扩展配置（预取阈值、超时参数等）
        (
            extra_config,
            prefetch_threshold,
            prefetch_timeout_base,
            prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys,
        ) = self._parse_storage_backend_extra_config(
            server_args.hicache_storage_backend_extra_config
        )
        # TODO: support more timeout check functions
        # 预取超时检查函数（目前仅支持线性超时策略）
        self.is_prefetch_timeout = self._prefetch_timeout_check_linear_func
        self.prefetch_stop_policy = server_args.hicache_storage_prefetch_policy

        # load_cache_event：通知调度器 KV 缓存加载已就绪
        self.load_cache_event = threading.Event()
        if isinstance(self.kv_cache, NSATokenToKVPool):
            # NSA 模型需要额外挂载混合池（包括 indexer pool）
            attach_hybrid_nsa_pool_to_hiradix_cache(
                self,
                params,
                server_args,
                extra_config=extra_config,
                prefetch_threshold=prefetch_threshold,
                enable_storage_metrics=self.enable_storage_metrics,
                load_cache_event=self.load_cache_event,
                attn_cp_group=self.attn_cp_group,
                attn_tp_group=self.attn_tp_group,
            )
        else:
            # 标准 MHA/MLA 路径：创建 HiCacheController 管理 GPU↔CPU DMA 传输
            self.cache_controller = HiCacheController(
                params.token_to_kv_pool_allocator,
                self.token_to_kv_pool_host,
                self.page_size,
                self.tp_group,
                load_cache_event=self.load_cache_event,
                attn_cp_group=self.attn_cp_group,
                attn_tp_group=self.attn_tp_group,
                write_policy=server_args.hicache_write_policy,
                io_backend=server_args.hicache_io_backend,
                storage_backend=server_args.hicache_storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=server_args.served_model_name,
                storage_backend_extra_config=extra_config,
                pp_rank=self.pp_rank,
                pp_size=self.pp_size,
                enable_storage_metrics=self.enable_storage_metrics,
            )
        # 应用运行时存储配置（预取阈值、超时、metrics 采集器）
        self._apply_storage_runtime_config(
            storage_backend=server_args.hicache_storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_base=prefetch_timeout_base,
            prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            enable_storage=self.enable_storage,
            enable_storage_metrics=self.enable_storage_metrics,
            extra_metric_labels=self.extra_metric_labels,
        )

        # ongoing_write_through：正在异步写入主机的节点 {node.id -> node}
        self.ongoing_write_through = {}
        # ongoing_load_back：正在从主机加载回 GPU 的节点段 {node.id -> node}
        self.ongoing_load_back = {}
        # ongoing_prefetch：正在从 L3 预取的请求 {req_id -> (last_host_node, prefetch_key, host_indices, operation)}
        self.ongoing_prefetch = {}
        # ongoing_backup：正在异步写入 L3 存储的节点 {operation_id -> node}
        self.ongoing_backup = {}
        # 按请求统计从存储加载的 token 数（L3 命中数），用于路由指标
        # key: request_id, value: number of tokens actually loaded from storage
        self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}
        # todo: dynamically adjust the threshold
        # write_through_threshold：命中次数达到该值才触发写穿到主机（write_through 时为 1，write_back 时为 2）
        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 2
        )
        # load_back_threshold：加载回 GPU 的最小 token 数，防止为极少量数据触发 DMA
        self.load_back_threshold = 10

        # 进程退出时自动注销存储后端（best-effort）
        atexit.register(self.shutdown)

        # 可驱逐主机叶节点集合（GPU 已驱逐、CPU 有备份且未被锁定的节点）
        self.evictable_host_leaves = set()

        # 调用父类 RadixCache.__init__ 完成根节点和驱逐策略的初始化
        super().__init__(params=params)

    def _all_reduce_attn_groups(self, tensor: torch.Tensor, op):
        # 优先在 attn_cp / attn_tp 组内做 all_reduce，保证注意力并行一致性
        # 若无专用注意力并行组则回退到全局 tp_group
        reduced = False
        for group in (self.attn_cp_group, self.attn_tp_group):
            if group is not None and torch.distributed.get_world_size(group=group) > 1:
                torch.distributed.all_reduce(tensor, op=op, group=group)
                reduced = True
        if not reduced and self.tp_world_size > 1:
            torch.distributed.all_reduce(tensor, op=op, group=self.tp_group)

    def _barrier_attn_groups(self):
        # 在注意力并行组内执行 barrier，确保所有 TP worker 到达同一检查点
        waited = False
        for group in (self.attn_cp_group, self.attn_tp_group):
            if group is not None and torch.distributed.get_world_size(group=group) > 1:
                torch.distributed.barrier(group=group)
                waited = True
        if not waited and self.tp_world_size > 1:
            torch.distributed.barrier(group=self.tp_group)

    def shutdown(self):
        """Best-effort auto-detach of storage backend on process shutdown.

        This keeps startup and runtime behavior consistent: if a backend was attached
        (either via CLI args or via admin API), we attempt to detach it on exit.
        """
        try:
            # 进程退出时尝试优雅关闭存储后端，失败则仅记录日志不崩溃
            if self.enable_storage:
                self.detach_storage_backend()
        except Exception:
            logger.exception("Failed to detach storage backend on process shutdown.")

    def _apply_storage_runtime_config(
        self,
        *,
        storage_backend: Optional[str],
        prefetch_threshold: int,
        prefetch_timeout_base: float,
        prefetch_timeout_per_ki_token: float,
        hicache_storage_pass_prefix_keys: bool,
        enable_storage: bool,
        enable_storage_metrics: bool,
        extra_metric_labels: Optional[Dict[str, str]],
    ) -> None:
        # 将每 1024 token 的超时换算为每 page 的超时（与 page_size 成正比）
        prefetch_timeout_per_page = (
            self.page_size / 1024 * prefetch_timeout_per_ki_token
        )

        # 更新运行时存储配置字段
        self.enable_storage = enable_storage
        self.prefetch_threshold = prefetch_threshold
        self.prefetch_timeout_base = prefetch_timeout_base
        self.prefetch_timeout_per_page = prefetch_timeout_per_page
        self.hicache_storage_pass_prefix_keys = hicache_storage_pass_prefix_keys
        self.enable_storage_metrics = enable_storage_metrics

        if self.enable_storage_metrics:
            # 收集 TP/CP/PP rank 信息用于 metrics 标签
            attn_cp_rank, attn_cp_size = (
                self.cache_controller.get_attn_cp_rank_and_size()
            )
            labels = {
                "storage_backend": storage_backend,
                "tp_rank": self.cache_controller.tp_rank,
                "dp_rank": self.cache_controller.dp_rank,
                "pp_rank": self.cache_controller.pp_rank,
                "pp_size": self.cache_controller.pp_size,
                "attn_cp_rank": attn_cp_rank,
                "attn_cp_size": attn_cp_size,
            }
            if extra_metric_labels:
                labels.update(extra_metric_labels)
            # 如果采集器不存在则新建；标签 key 不变时更新 labels；否则保留旧采集器避免重复注册
            existing_collector = getattr(self, "storage_metrics_collector", None)
            if existing_collector is None:
                self.storage_metrics_collector = StorageMetricsCollector(labels=labels)
            elif set(existing_collector.labels.keys()) == set(labels.keys()):
                existing_collector.labels = labels
            else:
                logger.warning(
                    "Storage metrics labels changed (%s -> %s). Keep existing labels to "
                    "avoid duplicate metric registration.",
                    sorted(existing_collector.labels.keys()),
                    sorted(labels.keys()),
                )

    def attach_storage_backend(
        self,
        storage_backend: str,
        storage_backend_extra_config_json: Optional[str] = None,
        served_model_name: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Attach (enable) storage backend at runtime.

        This will start storage threads inside `HiCacheController` and enable
        prefetch/backup paths. Caller must ensure there are no running/queued
        requests to avoid races.
        """
        # 先验证参数合法性（无副作用），避免校验失败后留下部分状态
        if hicache_storage_prefetch_policy is not None:
            allowed = ["best_effort", "wait_complete", "timeout"]
            if hicache_storage_prefetch_policy not in allowed:
                return (
                    False,
                    f"Invalid hicache_storage_prefetch_policy: {hicache_storage_prefetch_policy!r}. "
                    f"Expected one of {allowed}.",
                )

        if hicache_write_policy is not None:
            allowed = ["write_back", "write_through", "write_through_selective"]
            if hicache_write_policy not in allowed:
                return (
                    False,
                    f"Invalid hicache_write_policy: {hicache_write_policy!r}. "
                    f"Expected one of {allowed}.",
                )

        # 幂等处理：若已启用同一后端则只更新策略；若后端不同则拒绝
        if self.enable_storage:
            current_backend = self.cache_controller.storage_backend_type

            if current_backend == storage_backend:
                # 相同后端：允许热更新预取/写回策略
                if hicache_storage_prefetch_policy is not None:
                    self.prefetch_stop_policy = hicache_storage_prefetch_policy
                    logger.info(
                        f"Set hicache_storage_prefetch_policy to {hicache_storage_prefetch_policy}"
                    )
                if hicache_write_policy is not None:
                    self.cache_controller.write_policy = hicache_write_policy
                    self.write_through_threshold = (
                        1 if hicache_write_policy == "write_through" else 2
                    )
                    logger.info(f"Set hicache_write_policy to {hicache_write_policy}")
                return (
                    True,
                    "HiCache storage backend already enabled with same backend; policies updated.",
                )

            # 不同后端：必须先 detach 再 attach
            return (
                False,
                f"HiCache storage backend is already enabled with backend '{current_backend}'. "
                f"Cannot attach different backend '{storage_backend}'. Detach first.",
            )

        # 未启用时：先更新策略，再挂载后端（让存储线程观察到最新策略）
        if hicache_storage_prefetch_policy is not None:
            self.prefetch_stop_policy = hicache_storage_prefetch_policy
            logger.info(
                f"Set hicache_storage_prefetch_policy to {hicache_storage_prefetch_policy}"
            )

        if hicache_write_policy is not None:
            self.cache_controller.write_policy = hicache_write_policy
            self.write_through_threshold = (
                1 if hicache_write_policy == "write_through" else 2
            )
            logger.info(f"Set hicache_write_policy to {hicache_write_policy}")

        logger.info(f"Attaching HiCache storage backend: {storage_backend}")
        try:
            (
                extra_config,
                prefetch_threshold,
                prefetch_timeout_base,
                prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys,
            ) = self._parse_storage_backend_extra_config(
                storage_backend_extra_config_json
            )
        except Exception as e:
            logger.exception(f"Failed to parse storage_backend_extra_config_json: {e}")
            return (
                False,
                f"Failed to parse storage_backend_extra_config_json '{storage_backend_extra_config_json}': {e}",
            )

        try:
            self.cache_controller.attach_storage_backend(
                storage_backend=storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=served_model_name,
                storage_backend_extra_config=extra_config,
                **self._get_hybrid_storage_attach_kwargs(),
            )
        except Exception as e:
            logger.exception(
                f"Failed to attach storage backend '{storage_backend}': {e}"
            )
            return False, f"Failed to attach storage backend '{storage_backend}': {e}"

        self._apply_storage_runtime_config(
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_base=prefetch_timeout_base,
            prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            enable_storage=True,
            enable_storage_metrics=self._enable_metrics_flag,
            extra_metric_labels=self.extra_metric_labels,
        )
        return True, "Attached HiCache storage backend successfully."

    def detach_storage_backend(self) -> tuple[bool, str]:
        """Detach (disable) storage backend at runtime.

        Caller must ensure there are no running/queued requests to avoid races.
        """
        try:
            # 先排空控制队列再停止存储线程，防止 ack/release 无法匹配已清理的节点
            # IMPORTANT: this must happen before we clear `ongoing_*`, otherwise acks/releases
            # cannot be matched to nodes and may leak host pages / locks.
            self._drain_storage_control_queues_local()
            # 幂等 detach：即使 enable_storage 已为 False 也执行，清理可能遗留的状态
            # Idempotent detach: always ask controller to best-effort cleanup, even if
            # `self.enable_storage` is already False (may be leftover state from a
            # previous partial detach).
            self.cache_controller.detach_storage_backend()
        except Exception as e:
            logger.exception("Failed to detach storage backend.")
            # Do NOT crash the server for admin operations. Return failure with detail.
            return False, f"Failed to detach HiCache storage backend: {e}"

        # 再次排队以清理在控制器停止前产生的残余消息
        self._drain_storage_control_queues_local()
        # 强制释放遗留的预取/备份操作占用的主机内存和锁引用
        self._force_release_pending_storage_ops()

        self.enable_storage = False
        self.enable_storage_metrics = False
        return True, "Detached HiCache storage backend successfully."

    def _force_release_pending_storage_ops(self):
        """Force release any leftover pending prefetch/backup bookkeeping.

        This is a safety net for detach/shutdown paths. It assumes storage threads
        have been stopped already (via controller.detach), so no concurrent access
        to these structures should happen.
        """
        cc = self.cache_controller

        # 强制释放未完成的预取操作：归还预分配的主机内存页，并解除匹配前缀节点的保护
        try:
            for req_id, info in list(self.ongoing_prefetch.items()):
                try:
                    last_host_node, token_ids, host_indices, _operation = info
                except Exception:
                    # Unexpected shape; just drop it.
                    self.ongoing_prefetch.pop(req_id, None)
                    continue

                try:
                    if host_indices is not None:
                        cc.mem_pool_host.free(host_indices)
                except Exception:
                    logger.exception(
                        "Failed to free host indices for prefetch %s", req_id
                    )

                try:
                    last_host_node.release_host()
                except Exception:
                    logger.exception(
                        "Failed to release host protection for prefetch %s", req_id
                    )

                try:
                    cc.prefetch_tokens_occupied -= len(token_ids)
                    if cc.prefetch_tokens_occupied < 0:
                        cc.prefetch_tokens_occupied = 0
                except Exception:
                    pass

                self.ongoing_prefetch.pop(req_id, None)
        except Exception:
            logger.exception("Force release pending prefetch ops failed.")

        # 强制释放未完成的备份操作：解除节点在主机侧的保护锁
        try:
            for ack_id, node in list(self.ongoing_backup.items()):
                try:
                    node.release_host()
                except Exception:
                    logger.exception(
                        "Failed to release host protection for backup op %s", ack_id
                    )
                self.ongoing_backup.pop(ack_id, None)
        except Exception:
            logger.exception("Force release pending backup ops failed.")

    def _drain_storage_control_queues_local(self):
        """Drain storage control queues without TP synchronization.

        This is intended for shutdown/detach paths where we want to make best-effort
        cleanup even if queue sizes temporarily differ across ranks.
        """
        # 传入 None 表示不限制每个队列的排空数量，一次性全部消费
        self._drain_storage_control_queues_impl(
            n_revoke=None,
            n_backup=None,
            n_release=None,
            log_metrics=False,
        )

    def _drain_storage_control_queues_impl(
        self,
        n_revoke: Optional[int],
        n_backup: Optional[int],
        n_release: Optional[int],
        log_metrics: bool,
    ):
        cc = self.cache_controller

        # 通用限量排空生成器：limit=None 时消费到队列为空
        def _drain_queue(q, limit: Optional[int]):
            drained = 0
            while limit is None or drained < limit:
                try:
                    item = q.get_nowait()
                except Empty:
                    break
                drained += 1
                yield item

        def _drain_revoke():
            # 撤销预取：释放主机保护并更新占用计数
            for req_id in _drain_queue(cc.prefetch_revoke_queue, n_revoke):
                info = self.ongoing_prefetch.pop(req_id, None)
                if info is not None:
                    last_host_node, token_ids, _, _ = info
                    last_host_node.release_host()
                    cc.prefetch_tokens_occupied -= len(token_ids)
                    if cc.prefetch_tokens_occupied < 0:
                        cc.prefetch_tokens_occupied = 0

        def _drain_backup():
            # 确认备份完成：解除节点主机保护，并可选记录备份 token 数指标
            for operation in _drain_queue(cc.ack_backup_queue, n_backup):
                ack_id = operation.id
                entry = self.ongoing_backup.pop(ack_id, None)
                if entry is not None:
                    entry.release_host()
                if log_metrics and self.enable_storage_metrics:
                    self.storage_metrics_collector.log_backuped_tokens(
                        operation.completed_tokens
                    )

        def _drain_release():
            # 批量归还主机内存：合并多个 release 请求一次 free，减少内存碎片
            host_indices_list = []
            for host_indices in _drain_queue(cc.host_mem_release_queue, n_release):
                host_indices_list.append(host_indices)
            if host_indices_list:
                host_indices = torch.cat(host_indices_list, dim=0)
                cc.mem_pool_host.free(host_indices)

        _drain_revoke()
        _drain_backup()
        _drain_release()

    def _parse_storage_backend_extra_config(
        self, storage_backend_extra_config: Optional[str]
    ):
        """
        Parse storage backend extra config JSON and extract specific parameters.

        Args:
            storage_backend_extra_config: JSON string containing extra configuration

        Returns:
            tuple: (extra_config_dict, prefetch_threshold, prefetch_timeout_base, prefetch_timeout_per_ki_token, hicache_storage_pass_prefix_keys)
        """
        # 解析扩展配置：支持内联 JSON 字符串或 @file_path（json/toml/yaml 文件）
        extra_config = {}
        if storage_backend_extra_config:
            try:
                if storage_backend_extra_config.startswith("@"):
                    # 从 @<路径> 指定的配置文件读取，根据扩展名选择解析器
                    path = storage_backend_extra_config[1:]
                    ext = os.path.splitext(path)[1].lower()
                    with open(path, "rb" if ext == ".toml" else "r") as f:
                        if ext == ".json":
                            extra_config = json.load(f)
                        elif ext == ".toml":
                            import tomllib

                            extra_config = tomllib.load(f)
                        elif ext in (".yaml", ".yml"):
                            import yaml

                            extra_config = yaml.safe_load(f)
                        else:
                            raise ValueError(
                                f"Unsupported config file {path} (config format: {ext})"
                            )
                else:
                    # 直接解析 JSON 字符串
                    extra_config = json.loads(storage_backend_extra_config)
            except Exception as e:
                logger.error(f"Invalid backend extra config JSON: {e}")
                raise e

        # 从 extra_config 中 pop 已知参数，剩余部分传给底层存储后端
        prefetch_threshold = extra_config.pop("prefetch_threshold", 256)  # tokens
        prefetch_timeout_base = extra_config.pop("prefetch_timeout_base", 1)  # seconds
        prefetch_timeout_per_ki_token = extra_config.pop(
            "prefetch_timeout_per_ki_token", 0.25
        )  # seconds per 1024 tokens
        # 是否将前缀键传给存储后端（用于位置感知哈希匹配）
        hicache_storage_pass_prefix_keys = extra_config.pop(
            "hicache_storage_pass_prefix_keys", False
        )

        # 类型校验，确保配置值符合预期类型
        if not isinstance(prefetch_threshold, int):
            raise ValueError(
                f"prefetch_threshold must be int, got {type(prefetch_threshold).__name__}"
            )
        if not isinstance(prefetch_timeout_base, (int, float)):
            raise ValueError(
                f"prefetch_timeout_base must be number, got {type(prefetch_timeout_base).__name__}"
            )
        if not isinstance(prefetch_timeout_per_ki_token, (int, float)):
            raise ValueError(
                f"prefetch_timeout_per_ki_token must be number, got {type(prefetch_timeout_per_ki_token).__name__}"
            )
        if not isinstance(hicache_storage_pass_prefix_keys, bool):
            raise ValueError(
                "hicache_storage_pass_prefix_keys must be bool, got "
                f"{type(hicache_storage_pass_prefix_keys).__name__}"
            )

        # 统一将超时时间转为 float，避免整数除法精度问题
        return (
            extra_config,
            prefetch_threshold,
            float(prefetch_timeout_base),
            float(prefetch_timeout_per_ki_token),
            hicache_storage_pass_prefix_keys,
        )

    def reset(self):
        # 重置全局节点计数器，清空控制器和主机内存池
        TreeNode.counter = 0
        self.cache_controller.reset()
        self.token_to_kv_pool_host.clear()
        # 清空请求级 token 追踪字典和可驱逐主机叶节点集合
        self.prefetch_loaded_tokens_by_reqid.clear()
        self.evictable_host_leaves.clear()
        super().reset()

    def get_height(self, node: TreeNode):
        # 返回节点到根的深度（用于调试和统计）
        height = 0
        while node != self.root_node:
            node = node.parent
            height += 1
        return height

    def _get_extra_pools(self) -> dict:
        # NSA 混合控制器需要额外传入 INDEXER 池参数（ALL_PAGES 命中策略）
        if not isinstance(self.cache_controller, HybridCacheController):
            return {}
        if isinstance(self.kv_cache, NSATokenToKVPool):
            pool = PoolTransfer(
                name=PoolName.INDEXER,
                hit_policy=PoolHitPolicy.ALL_PAGES,
            )
            return {"extra_pools": [pool]}
        else:
            return {}

    def _get_hybrid_storage_attach_kwargs(self) -> dict:
        """Extra kwargs for attach_storage_backend when controller is HybridCacheController."""
        # 混合控制器需要传入主机池列表，标准控制器返回空字典
        if isinstance(self.cache_controller, HybridCacheController):
            return {"host_pools": self.cache_controller.mem_pool_host.entries}
        return {}

    def clear_storage_backend(self) -> bool:
        # 清空 L3 存储后端（目前仅 nixl 类后端支持 clear 操作）
        if self.enable_storage:
            try:
                # Check if the storage backend has a clear method (for nixl backends)
                if hasattr(self.cache_controller.storage_backend, "clear"):
                    self.cache_controller.storage_backend.clear()
                    logger.info(
                        "Hierarchical cache storage backend cleared successfully!"
                    )
                    return True
                else:
                    logger.warning(
                        f"Storage backend {type(self.cache_controller.storage_backend).__name__} does not support clear operation."
                    )
                    return False
            except Exception as e:
                logger.error(f"Failed to clear hierarchical cache storage backend: {e}")
                return False
        else:
            logger.warning("Hierarchical cache storage backend is not enabled.")
            return False

    def write_backup(self, node: TreeNode, write_back=False) -> int:
        # 写穿不变式：已备份节点必须从根节点连续覆盖（不允许有间隙）
        # 若父节点尚未备份则跳过，等待父节点先完成写穿
        # Backup invariant (for write-through mode): backed-up nodes must form a
        # contiguous prefix from root — no gaps.  Skip if parent isn't backed
        # up yet;
        if not write_back and (
            node.parent != self.root_node and not node.parent.backuped
        ):
            return 0

        # 启动异步 DMA 将 GPU 端 KV 块拷贝到主机内存
        host_indices = self.cache_controller.write(
            device_indices=node.value,
            node_id=node.id,
            **self._get_extra_pools(),
        )
        if host_indices is None:
            # 主机内存不足时先驱逐再重试
            self.evict_host(len(node.value))
            host_indices = self.cache_controller.write(
                device_indices=node.value,
                node_id=node.id,
                **self._get_extra_pools(),
            )
        if host_indices is not None:
            node.host_value = host_indices.clone()
            assert len(node.host_value) > 0
            # 将节点加入 ongoing_write_through，等待 DMA 确认后更新 backuped 状态
            self.ongoing_write_through[node.id] = node
            if not write_back:
                # write_back 模式下调用方自己管理锁引用，这里只在 write_through 时加锁
                self.inc_lock_ref(node)
            # Note: store(CPU) event is deferred to writing_check() after the
            # async DMA transfer is confirmed complete.
            # 注意：CPU 端的 BlockStored 事件推迟到 writing_check() 中 DMA 确认后再发出
        else:
            return 0

        return len(host_indices)

    def write_backup_storage(self, node: TreeNode):
        # 将已写入主机内存的 KV 块进一步异步备份到 L3 存储后端
        prefix_keys = (
            node.get_prefix_hash_values(node.parent)
            if self.hicache_storage_pass_prefix_keys
            else None
        )

        # 提交异步存储写请求并记录到 ongoing_backup
        operation_id = self.cache_controller.write_storage(
            node.host_value,
            node.key,
            node.hash_value,
            prefix_keys,
            **self._get_extra_pools(),
        )
        self.ongoing_backup[operation_id] = node
        # 保护主机内存，防止在存储写入完成前被驱逐
        node.protect_host()

    def _inc_hit_count(self, node: TreeNode, chunked=False):
        # skip the hit count update for chunked requests
        # write_back 策略下由驱逐逻辑决定写入时机，无需追踪命中次数
        if self.cache_controller.write_policy == "write_back" or chunked:
            return
        node.hit_count += 1

        if not node.backuped:
            if node.hit_count >= self.write_through_threshold:
                # 命中次数达到阈值且节点尚未备份，触发写穿到主机
                self.write_backup(node)

    def writing_check(self, write_back=False):
        if write_back:
            # write_back 模式：阻塞等待所有写穿操作完成（同步语义）
            while len(self.ongoing_write_through) > 0:
                for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
                    finish_event.synchronize()
                    for ack_id in ack_list:
                        backuped_node = self.ongoing_write_through.pop(ack_id)
                        # DMA confirmed -- block is now on host.
                        # DMA 确认完成 —— 发出 CPU 端 BlockStored 事件
                        self._record_store_event(
                            backuped_node, medium=StorageMedium.CPU
                        )
                        if self.enable_storage:
                            # DMA 写入主机后，进一步触发 L3 存储写入
                            self.write_backup_storage(backuped_node)
                self.cache_controller.ack_write_queue.clear()
                assert len(self.ongoing_write_through) == 0
            return

        # NOTE: all ranks has the same ongoing_write_through, can skip sync if empty
        # 若队列为空则跳过跨 TP worker 同步（各 rank 状态一致）
        if len(self.ongoing_write_through) == 0:
            return

        # 轮询 ACK 队列，统计本 rank 已完成的写穿数
        finish_count = 0
        for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
            if not finish_event.query():
                break
            finish_count += 1
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        # 跨 CP/TP 参与者取最小值，确保所有 rank 以相同顺序推进缓存状态
        self._all_reduce_attn_groups(queue_size, torch.distributed.ReduceOp.MIN)

        finish_count = int(queue_size.item())
        while finish_count > 0:
            _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                backuped_node = self.ongoing_write_through.pop(ack_id)
                # DMA confirmed -- block is now on host.
                # DMA 传输确认：发出 CPU 存储事件，解锁节点，并可选触发 L3 存储
                self._record_store_event(backuped_node, medium=StorageMedium.CPU)
                self.dec_lock_ref(backuped_node)
                if self.enable_storage:
                    self.write_backup_storage(backuped_node)
            finish_count -= 1

    def loading_check(self):
        # 检查 ACK 加载队列，处理已完成的 CPU→GPU 加载任务
        finish_count = 0
        for _, finish_event, ack_list in self.cache_controller.ack_load_queue:
            if not finish_event.query():
                # the KV cache loading is still ongoing
                # KV 缓存加载尚未完成，后续事件也无需检查（保证有序）
                break
            finish_count += 1
            # no need to sync across TP workers as batch forwarding is synced
            # 批次前向传播已在外部同步，此处无需再跨 TP worker 同步
            for ack_id in ack_list:
                end_node = self.ongoing_load_back.pop(ack_id)
                self.dec_lock_ref(end_node)

        # ACK until all events are processed
        # 从头部弹出已处理的事件，避免二次处理
        del self.cache_controller.ack_load_queue[:finish_count]

    def evictable_size(self):
        # 返回当前可驱逐的 GPU token 数
        return self.evictable_size_

    def _to_radix_key(self, token_ids: List[int]) -> RadixKey:
        """Convert raw token_ids to a RadixKey; must be list (not tuple) for paged match."""
        return RadixKey(token_ids=list(token_ids))

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                # 从可驱逐大小转移到受保护大小
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            # 同步更新 GPU 侧和主机侧的叶节点集合
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            node = node.parent
        return IncLockRefResult(delta=delta)

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                # lock_ref 降至 0：从受保护大小转移回可驱逐大小
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            if node.parent is None:
                assert (
                    node is self.root_node
                ), f"This request holds the node from another tree"
            node = node.parent
        return DecLockRefResult(delta=delta)

    def _update_host_leaf_status(self, node: TreeNode):
        # 如果节点未被 GPU 驱逐或仍被锁定，则不应出现在主机可驱逐集合中
        if not node.evicted or node.lock_ref > 0:
            if node in self.evictable_host_leaves:
                self.evictable_host_leaves.remove(node)
            return

        # 若有任意子节点在主机上有备份，说明该节点不是主机叶节点（有后继依赖）
        for child in node.children.values():
            if child.backuped:
                if node in self.evictable_host_leaves:
                    self.evictable_host_leaves.remove(node)
                return

        # 该节点已驱逐、未被锁定且无备份子节点 —— 可作为主机叶节点被驱逐
        if node not in self.evictable_host_leaves:
            self.evictable_host_leaves.add(node)

    def evict(self, params: EvictParams) -> EvictResult:
        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        # 从当前可驱逐叶节点构建最小堆，优先驱逐优先级最低的节点
        leaves = list(self.evictable_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        write_back_nodes = []
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            if x.lock_ref > 0:
                continue

            if not x.backuped:
                if self.cache_controller.write_policy == "write_back":
                    # write_back 策略：驱逐前先将数据同步写入主机
                    written = self.write_backup(x, write_back=True)
                    num_evicted += written
                    if written > 0:
                        write_back_nodes.append(x)
                else:
                    # 无备份且非 write_back：直接驱逐（数据丢失）
                    num_evicted += self._evict_regular(x)
            else:
                # 已在主机有备份：只释放 GPU 侧内存
                num_evicted += self._evict_backuped(x)

            # 检查父节点：若所有子节点都已驱逐，则父节点晋升为可驱逐叶节点
            for child in x.parent.children.values():
                if child in write_back_nodes:
                    continue
                if not child.evicted:
                    break
            else:
                # all children are evicted or no children
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

        if self.cache_controller.write_policy == "write_back":
            # 等待所有 write_back 写入主机完成，再驱逐对应 GPU 内存
            self.writing_check(write_back=True)
            for node in write_back_nodes:
                assert node.backuped
                self._evict_backuped(node)

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def _evict_backuped(self, node: TreeNode):
        # GPU -> CPU 降级：块从设备移到主机
        # 发出 remove(GPU) 事件，让下游路由停止将此块视为设备本地
        # The matching store(CPU) was emitted when write_backup() copied to host.
        self._record_remove_event(node, medium=StorageMedium.GPU)
        num_evicted = self.cache_controller.evict_device(node.value)
        assert num_evicted > 0
        self.evictable_size_ -= num_evicted
        node.value = None  # GPU 侧引用清除，节点进入 evicted 状态
        self._update_leaf_status(node)
        self._update_host_leaf_status(node)
        # update leaf status for the parent because the node is evicted
        # 子节点被驱逐后，父节点可能成为新叶节点
        self._update_leaf_status(node.parent)
        return num_evicted

    def _evict_regular(self, node: TreeNode):
        # evict a node not initiated write to host -- emit BlockRemoved
        # 直接驱逐无主机备份的节点：数据永久丢失，发出 BlockRemoved 事件
        assert len(node.children) == 0, f"non-leaf, {node.id=}"

        self._record_remove_event(node)
        self.cache_controller.mem_pool_device_allocator.free(node.value)
        num_evicted = len(node.value)
        self._delete_leaf(node)
        return num_evicted

    def evict_host(self, num_tokens: int):
        # 驱逐主机侧 KV 块（L2 → 丢弃），仅驱逐 GPU 已释放且无主机引用的节点
        leaves = list(self.evictable_host_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)
            if x == self.root_node:
                break
            # only evict the host value of evicted nodes
            # 只驱逐 GPU 侧已释放的节点的主机内存
            if not x.evicted:
                continue

            if x.host_ref_counter > 0:
                continue

            # 节点完全删除（GPU + CPU 均释放）：发出 remove(CPU) 事件
            self._record_remove_event(x, medium=StorageMedium.CPU)
            num_evicted += self.cache_controller.evict_host(x.host_value)

            # 从父节点的 children 中移除并更新叶节点状态
            key = x.key.child_key(self.page_size)
            v = x.parent.children.pop(key, None)
            assert v == x, f"parent does not have child key, {key}"
            if x in self.evictable_host_leaves:
                self.evictable_host_leaves.remove(x)
            self._update_host_leaf_status(x.parent)

            # 若父节点也已全部 evicted，则晋升为主机叶节点候选
            if len(x.parent.children) == 0 and x.parent.evicted:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

    def load_back(
        self, node: TreeNode, mem_quota: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        # 从主机内存将 KV 块加载回 GPU（L2→L1 提升），支持内存配额限制

        start_time = time.perf_counter()
        last_hit_node = node
        nodes_to_load = []
        # 向上收集所有连续 evicted 节点（它们在 GPU 上已无数据）
        while node.evicted:
            assert (
                node.backuped
            ), "No backup available on evicted nodes, should not happen"
            nodes_to_load.insert(0, node)
            node = node.parent
        else:
            ancester_node = node

        # 锁住祖先节点，防止在加载期间被驱逐
        result = self.inc_lock_ref(ancester_node)
        delta = result.delta

        # 要么全部加载回 GPU，要么完全不加载（避免部分加载导致树状态不一致）
        host_indices = torch.cat([n.host_value for n in nodes_to_load])
        if len(host_indices) < self.load_back_threshold or (
            len(host_indices) > mem_quota + delta if mem_quota is not None else False
        ):
            # 数据量太少或超出内存配额，跳过加载
            self.dec_lock_ref(ancester_node)
            return None

        device_indices = self.cache_controller.load(
            host_indices=host_indices,
            node_id=last_hit_node.id,
            **self._get_extra_pools(),
        )
        if device_indices is None:
            # GPU 内存不足时先驱逐再重试
            self.evict(EvictParams(num_tokens=len(host_indices)))
            device_indices = self.cache_controller.load(
                host_indices=host_indices,
                node_id=last_hit_node.id,
                **self._get_extra_pools(),
            )
        self.dec_lock_ref(ancester_node)
        if device_indices is None:
            # no sufficient GPU memory to load back KV caches
            logger.warning(
                "load_back: FAILED to load %d tokens for node %d "
                "even after eviction (evictable_size=%d)",
                len(host_indices),
                last_hit_node.id,
                self.evictable_size_,
            )
            return None

        # 将加载回的 GPU 索引分配给各节点，并发出 BlockStored(GPU) 事件
        self.ongoing_load_back[last_hit_node.id] = last_hit_node
        offset = 0
        for node in nodes_to_load:
            node.value = device_indices[offset : offset + len(node.host_value)].clone()
            offset += len(node.host_value)
            # Block promoted from host to GPU -- emit store(GPU) so downstream
            # indexers see it as device-local again.
            self._record_store_event(node, medium=StorageMedium.GPU)
        self.evictable_size_ += len(device_indices)
        # 加载完成后锁住末端节点，防止加载结果在使用前被驱逐
        self.inc_lock_ref(last_hit_node)

        if self.metrics_collector is not None:
            self.metrics_collector.observe_load_back_duration(
                time.perf_counter() - start_time
            )
            self.metrics_collector.increment_load_back_num_tokens(len(device_indices))

        return device_indices

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ):
        # 根据末端主机节点触发 load_back；若节点已在 GPU 上则直接返回空 tensor
        last_node = params.last_host_node
        mem_quota = params.mem_quota
        if last_node.evicted:
            loading_values = self.load_back(last_node, mem_quota)
            if loading_values is not None:
                logger.debug(
                    f"loading back {len(loading_values)} tokens for node {last_node.id}"
                )
                return loading_values, last_node

            # load_back 失败（配额不足或数据量太小）：向上找到未驱逐的祖先节点
            while last_node.evicted:
                last_node = last_node.parent

        # 节点未被驱逐或加载失败时，返回空 GPU 索引和当前最深有效节点
        return (
            torch.empty((0,), dtype=torch.int64, device=self.device),
            last_node,
        )

    def ready_to_load_host_cache(self) -> int:
        """
        Notify the cache controller to start the KV cache loading.
        Return the consumer index for the schedule batch manager to track.
        """
        # 通知控制器开始执行主机→GPU 加载，返回消费者索引供调度器追踪
        return self.cache_controller.start_loading()

    def flush_write_through_acks(self) -> None:
        # 强制处理所有待确认的写穿 ACK（通常在推理步骤结束后调用）
        self.writing_check()

    def check_hicache_events(self):
        # 统一检查写穿、加载和存储控制队列（每个推理步骤调用一次）
        self.writing_check()
        self.loading_check()
        if self.enable_storage:
            self.drain_storage_control_queues()
        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_storage_metrics(
                self.cache_controller.storage_backend.get_stats()
            )

    def drain_storage_control_queues(self):
        """
        Combine prefetch revoke, backup ack, and host mem release checks
        to minimize TP synchronization and Python overhead.
        """
        cc = self.cache_controller

        # 收集三个队列的待处理数量，通过 all_reduce(MIN) 确保各 TP rank 处理相同数量
        qsizes = torch.tensor(
            [
                cc.prefetch_revoke_queue.qsize(),
                cc.ack_backup_queue.qsize(),
                cc.host_mem_release_queue.qsize(),
            ],
            dtype=torch.int,
        )
        self._all_reduce_attn_groups(qsizes, torch.distributed.ReduceOp.MIN)

        n_revoke, n_backup, n_release = map(int, qsizes.tolist())
        # 以同步的最小数量处理各队列，避免不同 rank 状态不一致
        self._drain_storage_control_queues_impl(
            n_revoke=n_revoke,
            n_backup=n_backup,
            n_release=n_release,
            log_metrics=True,
        )

    # Timeout is linearly increasing with the number of pages
    # 预取超时时间随页数线性增长
    def _prefetch_timeout_check_linear_func(self, operation: PrefetchOperation):
        # If hash_value has not been computed in timeout_base seconds, terminate it.
        # 若从预取启动到现在超过了 base + 页数×per_page 秒，则认为超时
        return (
            time.monotonic() - operation.start_time
            > self.prefetch_timeout_base
            + len(operation.hash_value) * self.prefetch_timeout_per_page
        )

    def can_terminate_prefetch(self, operation: PrefetchOperation):
        # 根据预取停止策略决定是否可以终止预取操作
        can_terminate = True

        if self.prefetch_stop_policy == "best_effort":
            # 尽力策略：直接允许终止
            return can_terminate

        if len(operation.hash_value) == 0:
            completed = False
        else:
            # 判断是否所有 token 都已完成预取
            completed = (
                operation.completed_tokens == len(operation.hash_value) * self.page_size
            )

        if self.prefetch_stop_policy == "wait_complete":
            # 等待完成策略：必须全部完成才能终止
            can_terminate = completed
        elif self.prefetch_stop_policy == "timeout":
            # 超时策略：完成或超时均可终止
            can_terminate = completed or self.is_prefetch_timeout(operation)
        else:
            # unknown prefetch stop policy, just return True
            return True

        operation_terminated = operation.is_terminated()
        # 将本地结果广播给所有 TP worker，取最大值确保一致性
        states = torch.tensor(
            [1 - int(can_terminate), int(operation_terminated)],
            dtype=torch.int,
        )
        self._all_reduce_attn_groups(states, torch.distributed.ReduceOp.MAX)
        can_terminate = states[0].item() == 0
        operation_terminated = states[1].item() == 1
        # the operation should be terminated if it is already terminated on any TP worker
        # or it meets the termination condition on all TP workers
        # 任一 TP worker 已标记终止，或所有 worker 均满足终止条件
        can_terminate = can_terminate or operation_terminated
        return can_terminate

    def check_prefetch_progress(self, req_id: str) -> bool:
        if req_id not in self.ongoing_prefetch:
            # there is no ongoing prefetch for this request or it has been revoked
            # 无正在进行的预取（未启动或已被撤销）
            return True

        # todo: more policies for prefetch progress such as timeout
        # the current policy is to prefetch with best effort and terminate when queuing is over
        last_host_node, prefetch_key, host_indices, operation = self.ongoing_prefetch[
            req_id
        ]

        if operation.host_indices is None:
            # prefetch has not been issued due to insufficient host memory
            # 预取因主机内存不足未能提交
            return True

        if not self.can_terminate_prefetch(operation):
            # 预取尚未满足终止条件，本轮调度等待
            return False

        completed_tokens, hash_value = self.cache_controller.terminate_prefetch(
            operation
        )
        logger.debug(f"Prefetch {req_id} completed with {completed_tokens} tokens")

        min_completed_tokens = completed_tokens
        # 各 TP rank 同步已完成的 token 数，取最小值保证树状态一致
        completed_tokens_tensor = torch.tensor(min_completed_tokens, dtype=torch.int)
        self._all_reduce_attn_groups(
            completed_tokens_tensor, torch.distributed.ReduceOp.MIN
        )
        min_completed_tokens = completed_tokens_tensor.item()
        # 截取实际写入主机的 key/indices/hash，插入主机侧前缀树
        fetched_key = prefetch_key[:min_completed_tokens]
        written_indices = host_indices[:min_completed_tokens]
        matched_length = self._insert_helper_host(
            last_host_node,
            fetched_key,
            written_indices,
            hash_value[: min_completed_tokens // self.page_size],
        )

        # 归还已匹配（树中已有）的主机页，异步释放超出匹配的已写入页
        self.cache_controller.mem_pool_host.free(host_indices[:matched_length])
        self.cache_controller.append_host_mem_release(
            host_indices[min_completed_tokens:completed_tokens]
        )
        last_host_node.release_host()
        del self.ongoing_prefetch[req_id]
        self.cache_controller.prefetch_tokens_occupied -= len(prefetch_key)

        # Track tokens actually loaded from storage for this request (L3 hits)
        # 统计实际从 L3 加载的 token 数（扣除已在 L2 匹配的部分）
        loaded_from_storage = min_completed_tokens - matched_length
        self.prefetch_loaded_tokens_by_reqid[req_id] = loaded_from_storage

        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_prefetched_tokens(loaded_from_storage)

        return True

    def terminate_prefetch(self, req_id: str):
        # 强制标记预取为已终止（调度取消时调用）
        if req_id not in self.ongoing_prefetch:
            return

        _, _, _, operation = self.ongoing_prefetch[req_id]
        if operation.host_indices is None:
            return
        operation.mark_terminate()

    def pop_prefetch_loaded_tokens(self, req_id: str) -> int:
        """
        Pop and return the number of tokens loaded from storage for a request.
        Returns 0 if no prefetch was done or was revoked.
        This should be called after check_prefetch_progress() returns True.
        """
        # 弹出并返回该请求从 L3 存储加载的 token 数（用于路由指标上报）
        return self.prefetch_loaded_tokens_by_reqid.pop(req_id, 0)

    def match_prefix(self, params: MatchPrefixParams):
        empty_value = torch.empty((0,), dtype=torch.int64, device=self.device)

        def empty_match_result():
            return MatchResult(
                device_indices=empty_value,
                last_device_node=self.root_node,
                last_host_node=self.root_node,
                host_hit_length=0,
            )

        if self.disable:
            return empty_match_result()

        key = params.key
        # EAGLE 推测解码时转换为 bigram 视图；截断到页对齐长度
        key, _ = key.maybe_to_bigram_view(self.is_eagle)
        key = key.page_aligned(self.page_size)
        if len(key) == 0:
            return empty_match_result()

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = empty_value

        # 统计主机命中长度（GPU 已驱逐但 CPU 有备份的连续节点）
        host_hit_length = 0
        last_host_node = last_node
        while last_node.evicted:
            host_hit_length += len(last_node.host_value)
            last_node = last_node.parent
        # last_host_node：最深的已备份节点（L2 命中的末端）
        while not last_host_node.backuped:
            last_host_node = last_host_node.parent

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_host_node,
            host_hit_length=host_hit_length,
        )

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node: TreeNode,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        # 构造预取键并对齐到页大小
        prefetch_key = RadixKey(
            new_input_tokens,
            extra_key=last_host_node.key.extra_key,
            is_bigram=self.is_eagle,
        )
        # align the number of fetching tokens to the page size
        prefetch_key = prefetch_key.page_aligned(self.page_size)
        prefetch_length = len(prefetch_key)
        # 未启用存储、预取长度不足阈值或受速率限制时跳过
        if (
            not self.enable_storage
            or prefetch_length < self.prefetch_threshold
            or self.cache_controller.prefetch_rate_limited()
        ):
            return

        # 锁住匹配前缀节点，防止在预取期间被驱逐
        last_host_node.protect_host()
        # 为预取数据预分配主机内存，不足时先驱逐主机内存再重试
        host_indices = self.cache_controller.mem_pool_host.alloc(prefetch_length)
        if host_indices is None:
            self.evict_host(prefetch_length)
            host_indices = self.cache_controller.mem_pool_host.alloc(prefetch_length)
        if host_indices is None:
            # 仍然不足时尝试缩减预取量到可用大小
            avaliable_size = self.cache_controller.mem_pool_host.available_size()
            prefetch_length = avaliable_size - (avaliable_size % self.page_size)
            if prefetch_length >= self.prefetch_threshold:
                new_input_tokens = new_input_tokens[:prefetch_length]
                host_indices = self.cache_controller.mem_pool_host.alloc(
                    prefetch_length
                )
            else:
                last_host_node.release_host()
                # no sufficient host memory for prefetch
                return
        # 提交异步预取操作并记录到 ongoing_prefetch
        operation = self.cache_controller.prefetch(
            req_id,
            host_indices,
            prefetch_key,
            last_hash,
            prefix_keys,
            **self._get_extra_pools(),
        )
        self.ongoing_prefetch[req_id] = (
            last_host_node,
            prefetch_key,
            host_indices,
            operation,
        )
        self.cache_controller.prefetch_tokens_occupied += len(prefetch_key)

    def _insert_helper_host(
        self, node: TreeNode, key: RadixKey, host_value, hash_value
    ):
        # 将从 L3 预取的 KV 数据插入主机侧前缀树（仅更新 host_value，不涉及 GPU）
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = key.child_key(self.page_size)

        matched_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = node.key.match(key, page_size=self.page_size)
            # 更新 key/host_value/hash_value 指针，跳过已匹配部分
            key = key[prefix_len:]
            host_value = host_value[prefix_len:]
            hash_value = hash_value[prefix_len // self.page_size :]
            matched_length += prefix_len

            if prefix_len < len(node.key):
                # 部分匹配：分裂节点后指向新前缀节点
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = key.child_key(self.page_size)

        if len(key):
            # 还有未匹配的键：创建新节点存储剩余主机 KV 数据
            new_node = TreeNode(priority=node.priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = None  # GPU 侧尚未分配
            new_node.host_value = host_value.clone()
            new_node.hash_value = hash_value
            node.children[child_key] = new_node
            self._update_host_leaf_status(new_node)
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            # Publish the newly materialized host suffix immediately so downstream
            # cache indexers can resolve descendants that extend this L2-only prefix.
            # 立即发布 CPU 侧 BlockStored 事件，让下游路由感知到新的 L2 前缀
            self._record_store_event(new_node, medium=StorageMedium.CPU)

        # 返回已匹配（可以立即释放主机页）的 token 长度
        return matched_length

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        # 在前缀树中查找最长匹配前缀，返回匹配的 GPU 索引列表和末端节点
        node.last_access_time = time.monotonic()
        child_key = key.child_key(self.page_size)
        value = []

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = child.key.match(key, page_size=self.page_size)
            if prefix_len < len(child.key):
                # 部分匹配：分裂节点后收集前缀部分的 GPU 索引
                new_node = self._split_node(child.key, child, prefix_len)
                if not new_node.evicted:
                    value.append(new_node.value)
                node = new_node
                break
            else:
                # 完全匹配：收集该节点的 GPU 索引并继续向下匹配
                if not child.evicted:
                    value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = key.child_key(self.page_size)

        return value, node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # child node split into new_node -> child
        # 分裂节点：创建新前缀节点 new_node，child 变为其子节点
        new_node = TreeNode(priority=child.priority)
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.hit_count = child.hit_count

        # split value and host value if exists
        # 分别分裂 GPU 侧和主机侧的 value（若已驱逐则 value=None）
        if child.evicted:
            new_node.value = None
        else:
            new_node.value = child.value[:split_len].clone()
            child.value = child.value[split_len:].clone()
        if child.backuped:
            # 若主机端有备份，同样按分裂点拆分主机 value
            new_node.host_value = child.host_value[:split_len].clone()
            child.host_value = child.host_value[split_len:].clone()

        # 按分裂点拆分 SHA256 哈希值列表
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[key.child_key(self.page_size)] = new_node

        return new_node

    def insert(self, params: InsertParams) -> InsertResult:
        key = params.key
        value = params.value
        chunked = params.chunked
        priority = params.priority

        if priority is None:
            priority = 0

        # EAGLE 推测解码时转换为 bigram 视图；截断到页对齐长度
        key, value = key.maybe_to_bigram_view(self.is_eagle, value)
        key = key.page_aligned(self.page_size)
        if value is not None:
            value = value[: len(key)]

        if len(key) == 0:
            return InsertResult(prefix_len=0)

        node = self.root_node
        child_key = key.child_key(self.page_size)
        total_prefix_length = 0

        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            # 优先级取节点历史最大值（确保高优先级请求不会被低优先级覆盖）
            node.priority = max(node.priority, priority)
            prefix_len = node.key.match(key, page_size=self.page_size)

            if prefix_len == len(node.key):
                if node.evicted:
                    # change the reference if the node is evicted
                    # 节点已被驱逐（KV 缓存重新计算场景）：恢复 GPU 侧 value
                    node.value = value[:prefix_len].clone()
                    self.evictable_size_ += len(node.value)
                    self._update_leaf_status(node)
                    self._update_host_leaf_status(node)
                    # update parent status as a new leaf is added into device
                    self._update_leaf_status(node.parent)
                else:
                    # 完全命中：更新命中计数（可能触发写穿）
                    self._inc_hit_count(node, chunked)
                    total_prefix_length += prefix_len
            else:
                # partial match, split the node
                # 部分匹配：分裂节点，前缀节点继承更新后的优先级
                new_node = self._split_node(node.key, node, prefix_len)
                # shared-prefix node should also reflect max priority
                new_node.priority = max(new_node.priority, priority)
                if new_node.evicted:
                    new_node.value = value[:prefix_len].clone()
                    self.evictable_size_ += len(new_node.value)
                    self._update_leaf_status(new_node)
                    self._update_host_leaf_status(new_node)
                    # update parent status as a new leaf is added into device
                    self._update_leaf_status(new_node.parent)
                else:
                    self._inc_hit_count(new_node, chunked)
                    total_prefix_length += prefix_len
                node = new_node

            key = key[prefix_len:]
            value = value[prefix_len:]

            if len(key):
                child_key = key.child_key(self.page_size)

        if len(key):
            # 剩余未匹配键：创建新叶节点存储新 KV 数据
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)

            # Compute hash_value if storage or kv events are enabled
            # 若启用存储或 KV 事件，则计算该节点的 SHA256 哈希链
            if self.enable_storage or self.enable_kv_cache_events:
                new_node.hash_value = compute_node_hash_values(new_node, self.page_size)

            # Emit BlockStored so the router indexes this block.
            # 发出 BlockStored 让路由层感知到新块
            self._record_store_event(new_node)

            if self.cache_controller.write_policy != "write_back":
                # write_back 策略下不在 insert 时触发写穿
                self._inc_hit_count(new_node, chunked)
        return InsertResult(prefix_len=total_prefix_length)

    def release_aborted_request(self, rid: str):
        # Clean up storage hit tracking for aborted request
        # 清理已取消请求的 L3 命中追踪
        self.prefetch_loaded_tokens_by_reqid.pop(rid, None)

        if rid not in self.ongoing_prefetch:
            return

        last_host_node, prefetch_key, host_indices, operation = self.ongoing_prefetch[
            rid
        ]
        if operation.host_indices is None:
            return

        # 等待预取完成后释放主机内存（barrier 确保各 TP rank 同步）
        completed_tokens, _ = self.cache_controller.terminate_prefetch(operation)
        self._barrier_attn_groups()
        last_host_node.release_host()
        del self.ongoing_prefetch[rid]
        # 异步释放已写入主机的 token 占用的内存
        self.cache_controller.append_host_mem_release(host_indices[:completed_tokens])
        self.cache_controller.prefetch_tokens_occupied -= len(prefetch_key)
