from __future__ import annotations

import logging

from sglang.srt.managers.prefill_delayer import PrefillDelayerSinglePassExecutor
from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams
from sglang.srt.utils import get_bool_env_var

# 从环境变量读取是否开启路由键策略的调试日志
_ROUTING_KEY_POLICY_DEBUG_LOG = get_bool_env_var("SGLANG_ROUTING_KEY_POLICY_DEBUG_LOG")
# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Request scheduler policy"""

import os
import random
from collections import Counter, defaultdict
from contextlib import contextmanager
from enum import Enum, auto
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union

import torch

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.attention.nsa.utils import is_nsa_prefill_cp_in_seq_split
from sglang.srt.layers.utils.cp_utils import is_prefill_context_parallel_enabled
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

# 对 max_new_tokens 进行截断估算，防止服务器过于保守地预留 KV 空间
# 该截断仅影响调度器的估算，不影响实际停止条件
CLIP_MAX_NEW_TOKENS = int(
    os.environ.get("SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION", "4096")
)

# 批内前缀缓存检查阈值：当请求的已匹配前缀长度不超过此值时，才进行批内前缀缓存检查
# 设置为 -1 表示禁用批内前缀缓存
IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD = int(
    os.environ.get("IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD", "32")
)

# 批内前缀缓存去优先化阈值：当批内已匹配前缀长度超过此值时，调度器降低该请求的优先级
IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD = int(
    os.environ.get("IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD", "32")
)


# ignore_eos 模式下，为边界情况预留的 token 数量
IGNORE_EOS_RESERVE_TOKENS = 1


# 感知树缓存的调度策略枚举
class CacheAwarePolicy(Enum):
    """Scheduling policies that are aware of the tree cache."""

    LPM = "lpm"  # longest prefix match（最长前缀匹配）
    DFS_WEIGHT = "dfs-weight"  # depth-first search weighting（深度优先搜索权重）


# 不感知树缓存的调度策略枚举
class CacheAgnosticPolicy(Enum):
    """Scheduling policies that are not aware of the tree cache."""

    FCFS = "fcfs"  # first come first serve（先来先服务）
    LOF = "lof"  # longest output first（最长输出优先）
    RANDOM = "random"  # 随机调度
    ROUTING_KEY = "routing-key"  # prioritize by routing key frequency in running batch（按路由键频率优先）


# 调度策略管理类，负责根据配置选择并执行调度算法
class SchedulePolicy:
    Policy = Union[CacheAwarePolicy, CacheAgnosticPolicy]

    def __init__(
        self,
        policy: str,
        tree_cache: BasePrefixCache,
        enable_hierarchical_cache: bool,
        enable_priority_scheduling: bool,
        schedule_low_priority_values_first: bool,
    ):
        # 校验并根据树缓存情况调整策略
        self.policy = self._validate_and_adjust_policy(policy, tree_cache)
        self.tree_cache = tree_cache
        self.enable_hierarchical_cache = enable_hierarchical_cache
        self.enable_priority_scheduling = enable_priority_scheduling
        self.schedule_low_priority_values_first = schedule_low_priority_values_first
        # priority_sign 决定优先级数值越小是否越优先
        self.priority_sign = 1 if schedule_low_priority_values_first else -1

        # 用于批内前缀缓存检查的模拟 Radix 树
        self.waiting_queue_radix_tree = RadixCache.create_simulated()

    def calc_priority(
        self, waiting_queue: List[Req], running_batch: Optional[ScheduleBatch] = None
    ) -> bool:
        # FCFS 策略下，可选按优先级+时间戳排序，不需要计算前缀匹配
        if self.policy == CacheAgnosticPolicy.FCFS:
            if self.enable_priority_scheduling:
                SchedulePolicy._sort_by_priority_and_fcfs(
                    waiting_queue, self.priority_sign
                )
            return False

        # 动态判断当前应使用的有效策略（队列过长时降级为 FCFS）
        policy = self._determine_active_policy(waiting_queue)

        prefix_computed = False
        if isinstance(policy, CacheAwarePolicy):
            # 感知缓存策略：计算前缀匹配结果，然后按策略排序
            prefix_computed = True
            temporary_deprioritized = self._compute_prefix_matches(
                waiting_queue, policy
            )
            if policy == CacheAwarePolicy.LPM:
                # 最长前缀匹配策略：匹配越长的请求越优先
                SchedulePolicy._sort_by_longest_prefix(
                    waiting_queue, temporary_deprioritized
                )
            elif policy == CacheAwarePolicy.DFS_WEIGHT:
                # 深度优先搜索权重策略：按树节点权重排序
                SchedulePolicy._sort_by_dfs_weight(waiting_queue, self.tree_cache)
            else:
                raise ValueError(f"Unknown CacheAware Policy: {policy=}")
        else:
            if policy == CacheAgnosticPolicy.FCFS:
                pass  # FCFS 不需要额外排序
            elif policy == CacheAgnosticPolicy.LOF:
                # 最长输出优先策略
                SchedulePolicy._sort_by_longest_output(
                    waiting_queue,
                    self.enable_priority_scheduling,
                    self.priority_sign,
                )
            elif policy == CacheAgnosticPolicy.RANDOM:
                # 随机策略
                SchedulePolicy._sort_randomly(waiting_queue)
            elif policy == CacheAgnosticPolicy.ROUTING_KEY:
                # 路由键策略：优先调度与当前运行批次路由键匹配的请求
                if running_batch is not None:
                    SchedulePolicy._sort_by_routing_key(waiting_queue, running_batch)
            else:
                raise ValueError(f"Unknown CacheAgnostic Policy: {policy=}")
        return prefix_computed

    def _determine_active_policy(self, waiting_queue: List[Req]) -> Policy:
        # 当等待队列超过 128 个请求时，LPM 策略的开销太高，自动降级为 FCFS
        if self.policy == CacheAwarePolicy.LPM and len(waiting_queue) > 128:
            # Turn off the expensive prefix matching and sorting when the #queue is large.
            return CacheAgnosticPolicy.FCFS
        return self.policy

    def _validate_and_adjust_policy(
        self, policy: str, tree_cache: BasePrefixCache
    ) -> Policy:
        """
        Validates the policy and adjusts it if necessary based on tree cache settings.
        """
        try:
            policy_enum = CacheAwarePolicy(policy)
            if getattr(tree_cache, "disable", True):
                # 若树缓存被禁用，感知缓存策略无意义，降级为 FCFS
                return CacheAgnosticPolicy.FCFS
            return policy_enum
        except ValueError:
            try:
                return CacheAgnosticPolicy(policy)
            except ValueError:
                raise ValueError(f"Unknown schedule_policy: {policy=}")

    def _compute_prefix_matches(
        self, waiting_queue: List[Req], policy: CacheAwarePolicy
    ) -> Set[int]:
        """
        Computes and caches the matching prefixes for requests in the waiting queue,
            and handles in-batch prefix caching logic.
        """
        # 存放临时被降低优先级的请求 ID 集合
        temporary_deprioritized: Set[int] = set()
        # 重置批内前缀匹配用的模拟 Radix 树
        self.waiting_queue_radix_tree.reset()

        for r in waiting_queue:
            # 构造完整 token 序列（输入 + 已生成输出）用于前缀匹配
            prefix_ids = r.origin_input_ids + r.output_ids
            extra_key = r.extra_key
            # NOTE: the prefix_indices must always be aligned with last_node
            # 在主 KV 缓存树中查找最长匹配前缀
            match_result = self.tree_cache.match_prefix(
                MatchPrefixParams(
                    key=RadixKey(token_ids=prefix_ids, extra_key=extra_key)
                )
            )
            # 更新请求的前缀索引、末端节点等缓存相关字段
            (
                r.prefix_indices,
                r.last_node,
                r.last_host_node,
                r.host_hit_length,
            ) = (
                match_result.device_indices,
                match_result.last_device_node,
                match_result.last_host_node,
                match_result.host_hit_length,
            )

            # NOTE(sang): This logic is for in-batch prefix caching;
            # 批内前缀缓存逻辑：若请求在主缓存的匹配不足阈值，则在等待队列内再次检查
            # 对共享相同前缀的多个请求，只调度一个，以提高缓存命中率
            if len(r.prefix_indices) <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:
                match_result = self.waiting_queue_radix_tree.match_prefix(
                    MatchPrefixParams(
                        key=RadixKey(token_ids=prefix_ids, extra_key=extra_key)
                    )
                )
                in_batch_matching_prefixes = match_result.device_indices
                if (
                    len(in_batch_matching_prefixes)
                    >= IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD
                ):
                    # 批内已有足够长的共享前缀，将该请求临时降低优先级
                    temporary_deprioritized.add(r.rid)
                else:
                    # 将该请求的前缀插入批内 Radix 树，供后续请求复用
                    self.waiting_queue_radix_tree.insert(
                        InsertParams(
                            key=RadixKey(token_ids=prefix_ids, extra_key=extra_key),
                            value=torch.empty(len(prefix_ids), dtype=torch.bool),
                        )
                    )
        return temporary_deprioritized

    @staticmethod
    def _sort_by_longest_prefix(
        waiting_queue: List[Req], temporary_deprioritized: Set[int]
    ) -> None:
        """Sorts the waiting queue based on the longest prefix match."""
        # 被临时降低优先级的请求排到最后，其余按前缀长度降序
        waiting_queue.sort(
            key=lambda r: (
                -len(r.prefix_indices)
                if r.rid not in temporary_deprioritized
                else float("inf")
            )
        )

    @staticmethod
    def _sort_by_dfs_weight(
        waiting_queue: List[Req], tree_cache: BasePrefixCache
    ) -> None:
        """Sorts the waiting queue based on a depth-first search weighting."""
        # 建立末端节点到请求列表的映射
        last_node_to_reqs = defaultdict(list)
        for req in waiting_queue:
            last_node_to_reqs[req.last_node].append(req)

        # 初始化每个节点的权重（该节点下直接等待的请求数）
        node_to_weight = defaultdict(int)
        for node in last_node_to_reqs:
            node_to_weight[node] = len(last_node_to_reqs[node])
        # 递归累加子树权重
        SchedulePolicy._calc_weight(tree_cache.root_node, node_to_weight)

        # 按 DFS 权重重新填充等待队列
        waiting_queue.clear()
        SchedulePolicy._get_dfs_priority(
            tree_cache.root_node,
            node_to_weight,
            last_node_to_reqs,
            waiting_queue,
        )

    @staticmethod
    def _sort_by_longest_output(
        waiting_queue: List[Req],
        enable_priority_scheduling: bool,
        priority_sign: int,
    ) -> None:
        """Sorts the waiting queue based on the longest output (max_new_tokens). If using priority scheduling, sort by priority first."""
        if enable_priority_scheduling:
            # 先按优先级排序，再按 max_new_tokens 降序
            waiting_queue.sort(
                key=lambda x: (
                    x.priority * priority_sign,
                    -x.sampling_params.max_new_tokens,
                )
            )
        else:
            # 仅按 max_new_tokens 降序排列
            waiting_queue.sort(key=lambda x: -x.sampling_params.max_new_tokens)

    @staticmethod
    def _sort_randomly(waiting_queue: List[Req]) -> None:
        """Shuffles the waiting queue randomly."""
        # 随机打乱等待队列顺序
        random.shuffle(waiting_queue)

    @staticmethod
    def _sort_by_priority_and_fcfs(
        waiting_queue: List[Req], priority_sign: int
    ) -> None:
        """Sorts the waiting queue based on the request priority then received titmestamp."""
        # 先按优先级，再按进入等待队列的时间戳（先来先服务）
        waiting_queue.sort(
            key=lambda x: (
                x.priority * priority_sign,
                x.time_stats.wait_queue_entry_time,
            )
        )

    @staticmethod
    def _sort_by_routing_key(
        waiting_queue: List[Req], running_batch: ScheduleBatch
    ) -> None:
        """Sorts waiting queue by routing key frequency in running batch."""
        # 统计当前运行批次中各路由键的出现频率
        routing_key_counts = Counter(
            r.routing_key for r in running_batch.reqs if r.routing_key
        )

        if _ROUTING_KEY_POLICY_DEBUG_LOG:
            waiting_keys_before = [r.routing_key for r in waiting_queue]
            logger.info(
                f"routing_key_counts={dict(routing_key_counts)}, "
                f"waiting_keys_before={waiting_keys_before}"
            )

        # 若运行批次中没有路由键，则不进行排序
        if not routing_key_counts:
            return

        def sort_key(req: Req):
            key = req.routing_key
            if key and key in routing_key_counts:
                # 路由键在运行批次中存在：优先调度，频率越高越靠前
                count = routing_key_counts[key]
                return (0, -count, key)
            else:
                # 路由键不在运行批次中：排到后面
                return (1, 0, key or "")

        waiting_queue.sort(key=sort_key)

        if _ROUTING_KEY_POLICY_DEBUG_LOG:
            waiting_keys_after = [r.routing_key for r in waiting_queue]
            logger.info(f"waiting_keys_after={waiting_keys_after}")

    @staticmethod
    def _calc_weight(cur_node: TreeNode, node_to_weight: Dict[TreeNode, int]) -> None:
        # 递归计算每个节点的累计权重（子树中所有等待请求数之和）
        for child in cur_node.children.values():
            SchedulePolicy._calc_weight(child, node_to_weight)
            # 将子节点的权重累加到父节点
            node_to_weight[cur_node] += node_to_weight[child]

    @staticmethod
    def _get_dfs_priority(
        cur_node: TreeNode,
        node_to_priority: Dict[TreeNode, int],
        last_node_to_reqs: Dict[TreeNode, List[Req]],
        q: List,
    ) -> None:
        # 按子节点权重降序递归访问，实现深度优先搜索
        children = [child for child in cur_node.children.values()]
        children.sort(key=lambda x: -node_to_priority[x])
        for child in children:
            SchedulePolicy._get_dfs_priority(
                child, node_to_priority, last_node_to_reqs, q
            )
        # 将当前节点上的请求追加到结果队列
        q.extend(last_node_to_reqs[cur_node])


# 添加请求到可运行列表时的返回状态枚举
class AddReqResult(Enum):
    CONTINUE = auto()  # Continue to add requests（继续添加请求）
    NO_TOKEN = auto()  # No token left（token 空间不足）
    OTHER = auto()  # Other reasons to stop adding requests（其他原因停止）


# Prefill 阶段请求准入控制器：管理 token 预算、分块 prefill、优先级抢占等逻辑
class PrefillAdder:
    def __init__(
        self,
        page_size: int,
        tree_cache: BasePrefixCache,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        running_batch: ScheduleBatch,
        new_token_ratio: float,
        rem_input_tokens: int,
        rem_chunk_tokens: Optional[int],
        num_mixed_decode_tokens: int = 0,
        priority_scheduling_preemption_threshold: int = 0,
        max_prefill_bs: int = 0,
        max_running_requests: Optional[int] = None,
        prefill_max_requests: Optional[int] = None,
        prefill_delayer_single_pass: Optional[PrefillDelayerSinglePassExecutor] = None,
        dllm_config: Optional[DllmConfig] = None,
    ):
        self.page_size = page_size
        self.tree_cache = tree_cache
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.running_batch = running_batch
        self.new_token_ratio = new_token_ratio
        # 减去混合解码 token 占用，得到可用的 prefill 输入 token 预算
        self.rem_input_tokens = rem_input_tokens - num_mixed_decode_tokens
        self.rem_chunk_tokens = rem_chunk_tokens
        self.dllm_config = dllm_config

        # 若使用扩散语言模型配置，初始化 dllm 相关元数据
        if self.dllm_config is not None:
            self._init_dllm_meta(dllm_config)

        if self.rem_chunk_tokens is not None:
            # 减去混合解码 token 占用
            self.rem_chunk_tokens -= num_mixed_decode_tokens
        # 记录已被占用的总 token 偏移量（初始值为混合解码 token 数）
        self.rem_total_token_offset = num_mixed_decode_tokens
        self.cur_rem_token_offset = num_mixed_decode_tokens

        # 可运行请求列表、抢占列表、分块请求、日志统计
        self.req_states = None
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None
        self.log_hit_tokens = 0
        # TODO(lsyin): report the real input tokens excluding page alignment
        self.log_input_tokens = 0

        if running_batch is not None:
            # 累加当前运行批次中所有请求的预估剩余 token 消耗
            self.rem_total_token_offset += sum(
                [
                    self._get_running_request_total_token_offset(r)
                    for r in running_batch.reqs
                ]
            )

        # 检测是否为滑动窗口注意力（SWA）混合模式
        self.is_hybrid_swa = isinstance(
            self.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator
        )
        # 检测是否为混合 SSM 缓存模式（如 Mamba）
        self.is_hybrid_ssm_cache = self.tree_cache.supports_mamba()

        self.rem_swa_token_offset = 0

        self.priority_scheduling_preemption_threshold = (
            priority_scheduling_preemption_threshold
        )
        # 检测是否开启 NSA prefill CP 序列分割
        self.nsa_prefill_cp_in_seq_split = is_nsa_prefill_cp_in_seq_split()
        self.max_running_requests = max_running_requests
        # 检测是否开启 prefill 上下文并行
        self.prefill_context_parallel_enabled = is_prefill_context_parallel_enabled()
        self.prefill_max_requests = prefill_max_requests
        self.prefill_delayer_single_pass = prefill_delayer_single_pass
        self.max_prefill_bs = max_prefill_bs

    def _init_dllm_meta(self, dllm_config: DllmConfig):
        # 初始化扩散语言模型调度所需的 token 块大小和剩余 token 预算
        self.dllm_block_size = dllm_config.block_size
        max_running_reqs = dllm_config.max_running_requests

        # dllm 模式下的 token 预算 = 最大并发请求数 × 块大小
        self.rem_dllm_tokens = max_running_reqs * self.dllm_block_size

    def _get_running_request_total_token_offset(self, req: Req) -> int:
        # 估算运行中请求还需要消耗的 token 数（考虑 CLIP_MAX_NEW_TOKENS 截断）
        return (
            min(
                (req.sampling_params.max_new_tokens - len(req.output_ids)),
                CLIP_MAX_NEW_TOKENS,
            )
            * self.new_token_ratio
        )

    @property
    def rem_total_tokens(self):
        # 计算总剩余可用 token 数（可用空间 + 可驱逐缓存 - 已预留偏移量）
        if self.is_hybrid_swa:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.full_available_size()
                + self.tree_cache.full_evictable_size()
            )
        elif self.is_hybrid_ssm_cache:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.available_size()
                + self.tree_cache.full_evictable_size()
            )
        else:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.available_size()
                + self.tree_cache.evictable_size()
            )
        return available_and_evictable - self.rem_total_token_offset

    @property
    def rem_swa_tokens(self):
        # 计算滑动窗口注意力（SWA）池的剩余 token 数
        return (
            self.token_to_kv_pool_allocator.swa_available_size()
            + self.tree_cache.swa_evictable_size()
            - self.rem_swa_token_offset
        )

    @property
    def cur_rem_tokens(self):
        # 计算当前批次剩余可用 token 数（不含已预分配给当前 prefill 批次的 token）
        if self.is_hybrid_swa:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.full_available_size()
                + self.tree_cache.full_evictable_size()
            )
        elif self.is_hybrid_ssm_cache:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.available_size()
                + self.tree_cache.full_evictable_size()
            )
        else:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.available_size()
                + self.tree_cache.evictable_size()
            )

        return available_and_evictable - self.cur_rem_token_offset

    def _swa_budget_for_req(self, extend_input_len: int) -> int:
        """SWA pool budget per request. Only valid when is_hybrid_swa is True.

        With chunked prefill + overlap scheduler, the peak SWA occupancy is:
          chunk N (running, not yet in tree) + sliding window (locked in tree)
          + chunk N+1 (new allocation)
        Since chunk N and locked tokens are already excluded from
        swa_available + swa_evictable, the budget only needs to cover the
        chunk N+1 allocation. We floor at sliding_window_size to reserve
        room for the decode phase.
        """
        # 分块 prefill 时，实际分配量取 extend_input_len 与 rem_chunk_tokens 的较小值
        if self.rem_chunk_tokens is not None:
            alloc = min(extend_input_len, self.rem_chunk_tokens)
        else:
            alloc = extend_input_len
        # 预算至少为 sliding_window_size，再加一页对齐开销
        return max(alloc, self.tree_cache.sliding_window_size) + self.page_size

    def ceil_paged_tokens(self, tokens: int) -> int:
        # 向上取整到页大小的整数倍（页对齐）
        return -(-tokens // self.page_size) * self.page_size

    def budget_state(self):
        # 检查总 token 预算和当前批次 token 预算是否耗尽
        no_token = self.rem_total_tokens <= 0 or self.cur_rem_tokens <= 0
        if not no_token and self.is_hybrid_swa:
            # SWA 模式下还需检查 SWA 池剩余量
            no_token = self.rem_swa_tokens <= 0
        if no_token:
            return AddReqResult.NO_TOKEN

        # 检查输入 token 预算（非 token 空间不足，而是 chunk 级别限制）
        if self.rem_input_tokens <= 0:
            return AddReqResult.OTHER

        if self.dllm_config is not None:
            if self.rem_dllm_tokens <= 0:
                return AddReqResult.OTHER
        else:
            if self.rem_chunk_tokens is not None and self.rem_chunk_tokens <= 0:
                return AddReqResult.OTHER

        return AddReqResult.CONTINUE

    def _update_prefill_budget(
        self, prefix_len: int, extend_input_len: int, max_new_tokens: int
    ):
        # TODO(lsyin): check this workaround logic, which only ensures the prefill will not out of memory, and may be too conservative
        # 对 extend_input_len 进行页对齐
        extend_input_len = self.ceil_paged_tokens(extend_input_len)

        # alloc_extend reserves an extra page_size per request to make sure the budget doesn't over-commit
        # 每个请求预留一页作为页对齐开销，防止超出预算
        page_overhead = self.page_size
        # 更新总 token 偏移量（输入 + 预估输出 + 页开销）
        self.rem_total_token_offset += extend_input_len + max_new_tokens + page_overhead
        # 更新当前批次 token 偏移量（输入 + 页开销）
        self.cur_rem_token_offset += extend_input_len + page_overhead
        # 消耗输入 token 预算
        self.rem_input_tokens -= extend_input_len

        if self.is_hybrid_swa:
            # SWA 模式下同步更新 SWA token 偏移量
            self.rem_swa_token_offset += self._swa_budget_for_req(extend_input_len)

        if self.dllm_config is not None:
            # dllm 模式消耗扩散语言模型 token 预算
            self.rem_dllm_tokens -= extend_input_len
        elif self.rem_chunk_tokens is not None:
            # 分块 prefill 模式消耗 chunk token 预算
            self.rem_chunk_tokens -= extend_input_len

        # 累计命中缓存的 token 数和实际输入 token 数（用于日志）
        self.log_hit_tokens += prefix_len
        self.log_input_tokens += extend_input_len

    def _get_dllm_remain_tokens(self) -> int:
        # 获取扩散语言模型模式下当前可分配的最大 token 数
        _rem_tokens = min(
            self.rem_dllm_tokens,
            self.dllm_block_size,
            int(self.rem_total_tokens),
        )
        if _rem_tokens <= 0:
            # 若计算结果为负，则退回到 dllm 剩余 token 数
            _rem_tokens = self.rem_dllm_tokens

        return _rem_tokens

    def _add_dllm_req(self, req: Req, prefix_len: int):
        # FIXME: consider the case when rem_dllm_tokens < dllm_block_size,
        # the diffusion unmask process may have some problems
        # 确保至少分配一页，并对齐到页大小
        trunc_len = (
            min(self.rem_dllm_tokens, self.dllm_block_size)
            // self.page_size
            * self.page_size
        )

        # 截断请求的输入长度并更新 fill_ids
        req.extend_input_len = trunc_len
        req.fill_ids = req.fill_ids[: prefix_len + trunc_len]

        # 将请求加入可运行列表并更新预算
        self.can_run_list.append(req)

        self._update_prefill_budget(prefix_len, trunc_len, 0)

    def _req_inc_lock_ref(self, req: Req):
        # 增加请求末端节点的锁引用计数，防止缓存被驱逐
        result = self.tree_cache.inc_lock_ref(req.last_node)
        if self.is_hybrid_swa:
            req.swa_uuid_for_lock = result.swa_uuid_for_lock

    def add_dllm_staging_req(self, req: Req):
        # 将扩散语言模型暂存请求加入可运行列表
        assert self.dllm_config is not None
        _rem_tokens = self._get_dllm_remain_tokens()

        if _rem_tokens <= 0:
            return AddReqResult.NO_TOKEN

        # Truncate input length to available tokens and update request metadata
        # 根据可用 token 数截断输入长度
        truncated = req.extend_input_len > _rem_tokens
        req.extend_input_len = min(req.extend_input_len, _rem_tokens)
        req.fill_ids = req.fill_ids[: len(req.prefix_indices) + req.extend_input_len]
        self.can_run_list.append(req)

        # Update budget: reserve max_new_tokens only if not truncated
        # 若未被截断则预留最大输出 token 预算，否则置 0
        max_new_tokens = (
            min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
            if not truncated
            else 0
        )
        self._update_prefill_budget(0, req.extend_input_len, max_new_tokens)

        # Return based on remaining token availability
        return (
            AddReqResult.NO_TOKEN
            if self._get_dllm_remain_tokens() <= 0
            else AddReqResult.CONTINUE
        )

    def add_chunked_req(self, req: Req):
        # 处理已在进行中的分块 prefill 请求，追加剩余 chunk
        if self.dllm_config is not None:
            _rem_tokens = self._get_dllm_remain_tokens()
        else:
            _rem_tokens = min(self.rem_chunk_tokens, int(self.rem_total_tokens))
            if self.is_hybrid_swa:
                # alloc_extend needs extend_num_tokens + page_size per request,
                # so reserve one page here to avoid OOM
                # SWA 模式下预留一页以防止 OOM
                _rem_tokens = min(
                    _rem_tokens, int(self.rem_swa_tokens) - self.page_size
                )
            # The chunked_req must be added to the list; otherwise, it will cause a memory leak.
            # Therefore, in certain cases where _rem_tokens <= 0, it should be replaced with rem_chunk_tokens.
            if _rem_tokens <= 0:
                if self.is_hybrid_swa:
                    return req
                _rem_tokens = self.rem_chunk_tokens

        # 如果当前 chunk 超出可用 token，进行截断
        truncated = req.extend_input_len > _rem_tokens
        req.set_extend_input_len(min(req.extend_input_len, _rem_tokens))
        req.fill_ids = req.fill_ids[: len(req.prefix_indices) + req.extend_input_len]
        self.can_run_list.append(req)
        self._update_prefill_budget(
            0,
            req.extend_input_len,
            (
                # 最后一个 chunk 时预留输出 token 预算，中间 chunk 置 0
                min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
                if not truncated
                else 0
            ),
        )

        # Return if chunked prefill not finished
        # 若被截断则返回请求对象（表示还有 chunk 未完成），否则返回 None
        return req if truncated else None

    @contextmanager
    def _lock_node(self, last_node: TreeNode):
        # 上下文管理器：在操作期间对 Radix 树节点加锁，退出时解锁
        try:
            result = self.tree_cache.inc_lock_ref(last_node)
            if self.tree_cache.supports_swa() and self.tree_cache.is_tree_cache():
                swa_uuid_for_lock = result.swa_uuid_for_lock
            yield None
        finally:
            if self.tree_cache.supports_swa() and self.tree_cache.is_tree_cache():
                self.tree_cache.dec_lock_ref(
                    last_node, DecLockRefParams(swa_uuid_for_lock=swa_uuid_for_lock)
                )
            else:
                self.tree_cache.dec_lock_ref(last_node)

    def add_one_req_ignore_eos(self, req: Req):
        # 检查当前请求的页对齐输入 token 是否超出可用空间
        paged_input = self.ceil_paged_tokens(req.extend_input_len)
        if paged_input > min(self.cur_rem_tokens, self.rem_total_tokens):
            return AddReqResult.NO_TOKEN
        if self.is_hybrid_swa:
            # SWA 模式下还需检查滑动窗口注意力池的剩余量
            if self._swa_budget_for_req(req.extend_input_len) > self.rem_swa_tokens:
                return AddReqResult.NO_TOKEN

        def add_req_state(r, insert_sort=False):
            # 根据是否 ignore_eos 决定 new_token_ratio
            new_token_ratio = (
                1.0 if r.sampling_params.ignore_eos else self.new_token_ratio
            )
            # 估算剩余需生成的 token 数
            tokens_left = r.sampling_params.max_new_tokens * new_token_ratio - len(
                r.output_ids
            )
            # 当前请求已占用的总 token 数
            tokens_occupied = len(r.origin_input_ids) + len(r.output_ids)

            if tokens_left <= 0:
                return

            if not insert_sort:
                # 直接追加到状态列表
                self.req_states.append((tokens_left, tokens_occupied))
            else:
                # 插入排序：按 tokens_left 升序维护有序列表
                i = 0
                for i in range(len(self.req_states)):
                    if tokens_left <= self.req_states[i][0]:
                        break
                self.req_states.insert(i, (tokens_left, tokens_occupied))

        if self.req_states is None:
            # 首次调用：初始化状态列表，包含当前、运行批次和已可运行请求
            self.req_states = []
            add_req_state(req)
            if self.running_batch is not None:
                for r in self.running_batch.reqs:
                    add_req_state(r)
            for r in self.can_run_list:
                add_req_state(r)
            # 按剩余 token 升序排列
            self.req_states.sort(key=lambda x: x[0])
        else:
            # 后续调用：插入排序方式添加新请求
            add_req_state(req, insert_sort=True)

        if not self.is_hybrid_swa:
            # Skip this logic for swa. The SWA has different memory management, and
            # this mechanism is underestimating the memory usage.
            # 非 SWA 模式：验证所有请求的 token 消耗不会导致 OOM
            cur_rem_tokens = self.cur_rem_tokens - self.ceil_paged_tokens(
                req.extend_input_len
            )
            tokens_freed = 0
            for i, (tokens_left, tokens_occupied) in enumerate(self.req_states):
                # tokens_left gives a reservative calculation as the last token is not stored
                # 当前轮次的剩余批次槽位数
                bs = len(self.req_states) - i
                # 计算保守的最小可用 token 余量
                min_free_tokens = cur_rem_tokens + tokens_freed - tokens_left * bs
                # reserve tokens for corner cases
                if min_free_tokens <= IGNORE_EOS_RESERVE_TOKENS * bs:
                    return AddReqResult.NO_TOKEN
                tokens_freed += tokens_occupied

        if self.dllm_config is not None:
            if self.rem_dllm_tokens <= 0:
                return AddReqResult.OTHER

            self._add_dllm_req(req, 0)
        elif (
            self.rem_chunk_tokens is None  # chunked prefill is disabled（未启用分块 prefill）
            or req.extend_input_len <= self.rem_chunk_tokens  # it is the last chunk（最后一个 chunk）
        ):
            # Non-chunked prefill（非分块或最后一个 chunk，一次性处理完整输入）
            self.can_run_list.append(req)
            self._update_prefill_budget(
                0,
                req.extend_input_len,
                min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS),
            )
        else:
            if self.rem_chunk_tokens <= 0:
                return AddReqResult.OTHER

            # Chunked prefill（分块 prefill：截断到当前 chunk 大小）
            trunc_len = self.rem_chunk_tokens

            req.set_extend_input_len(trunc_len)
            req.fill_ids = req.fill_ids[:trunc_len]
            self.can_run_list.append(req)
            self.new_chunked_req = req
            self._update_prefill_budget(0, trunc_len, 0)

        return self.budget_state()

    def add_one_req(
        self, req: Req, has_chunked_req: bool, truncation_align_size: Optional[int]
    ):
        # 若配置了 prefill 延迟器，检查是否允许当前 prefill
        if (self.prefill_delayer_single_pass is not None) and (
            not self.prefill_delayer_single_pass.negotiate_should_allow_prefill(
                local_prefillable=True,
                running_batch=self.running_batch.batch_size(),
                max_prefill_bs=self.max_prefill_bs,
                max_running_requests=self.max_running_requests,
            )
        ):
            return AddReqResult.OTHER
        # TODO support cp with multiple requests
        # Enabling context parallelism currently presents precision issues;
        # therefore, the prefill-batch setting is temporarily set to 1.
        # NSA prefill CP 或 prefill 上下文并行时，批次大小限制为 1
        if (
            self.nsa_prefill_cp_in_seq_split or self.prefill_context_parallel_enabled
        ) and len(self.can_run_list) >= 1:
            return AddReqResult.OTHER

        # 检查是否超出 prefill 最大请求数限制
        if (x := self.prefill_max_requests) is not None and len(self.can_run_list) >= x:
            return AddReqResult.OTHER

        # ignore_eos 且树缓存被禁用时，走特殊的准入逻辑
        if req.sampling_params.ignore_eos and getattr(self.tree_cache, "disable", True):
            return self.add_one_req_ignore_eos(req)

        # Reserve page_size for page-alignment overhead. The paged allocator
        # may consume up to one extra page per request (see alloc_extend), and
        # _update_prefill_budget already accounts for this in the deduction.
        # Without this, admission is more optimistic than the actual budget
        # deduction, allowing over-admission when the pool is nearly full.
        # 计算该请求需要的总 token 数（输入 + 预估输出 + 页对齐开销）
        max_new = min(
            max(req.sampling_params.max_new_tokens - len(req.output_ids), 0),
            CLIP_MAX_NEW_TOKENS,
        )
        total_tokens = req.extend_input_len + max_new + self.page_size

        # adjusting the input_tokens based on host_hit_length and page_size
        # 计算实际需要加载的输入 token 数（减去 host 命中的缓存）
        real_input_tokens = req.extend_input_len - req.host_hit_length
        real_input_tokens = self.ceil_paged_tokens(real_input_tokens)
        prefix_len = len(req.prefix_indices)

        # 若总 token 需求超出剩余空间，拒绝该请求
        if total_tokens >= self.rem_total_tokens:
            return AddReqResult.NO_TOKEN

        if self.is_hybrid_swa:
            # SWA 模式下检查滑动窗口注意力池剩余量
            swa_needed = self._swa_budget_for_req(req.extend_input_len)
            if swa_needed >= self.rem_swa_tokens:
                return AddReqResult.NO_TOKEN

        # 若输入 token 超出预算且已有请求在列表中，停止添加
        if real_input_tokens >= self.rem_input_tokens and len(self.can_run_list) != 0:
            return AddReqResult.OTHER

        with self._lock_node(req.last_node):
            # self.rem_total_tokens may decrease after the lock acquisition
            # 加锁后再次检查（锁操作可能影响缓存驱逐空间）
            if total_tokens >= self.rem_total_tokens:
                return AddReqResult.NO_TOKEN

            if self.is_hybrid_swa:
                swa_needed = self._swa_budget_for_req(req.extend_input_len)
                if swa_needed >= self.rem_swa_tokens:
                    return AddReqResult.NO_TOKEN

            if req.host_hit_length > 0:
                # 若命中 host 缓存，从 host 加载 KV，更新前缀索引
                new_indices, req.last_node = self.tree_cache.init_load_back(
                    InitLoadBackParams(
                        last_host_node=req.last_host_node,
                        host_hit_length=req.host_hit_length,
                        req=req,
                    )
                )
                req.prefix_indices = torch.cat([req.prefix_indices, new_indices])
                req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
                prefix_len = len(req.prefix_indices)
                req.cache_protected_len = prefix_len

            # 计算页对齐后的实际输入 token 数
            input_tokens = self.ceil_paged_tokens(req.extend_input_len)

            if input_tokens >= self.rem_input_tokens and len(self.can_run_list) != 0:
                return AddReqResult.OTHER

            if self.dllm_config is not None:
                if self.rem_dllm_tokens <= 0:
                    return AddReqResult.OTHER

                assert (
                    truncation_align_size is None
                ), "truncation_align_size is not supported for dllm prefill"

                # dllm 模式：添加扩散语言模型请求并增加锁引用
                self._add_dllm_req(req, prefix_len)
                self._req_inc_lock_ref(req)
            elif self.rem_chunk_tokens is None or input_tokens <= self.rem_chunk_tokens:
                # Non-chunked prefill（非分块或最后一个 chunk）
                self.can_run_list.append(req)

                self._req_inc_lock_ref(req)
                self._update_prefill_budget(
                    prefix_len,
                    input_tokens,
                    min(
                        req.sampling_params.max_new_tokens,
                        CLIP_MAX_NEW_TOKENS,
                    ),
                )
            else:
                # Make sure at least one page is available
                # 截断到 chunk 预算并页对齐
                trunc_len = self.rem_chunk_tokens // self.page_size * self.page_size

                if trunc_len <= 0:
                    return AddReqResult.OTHER

                # When truncation align size is set, we want to assert that the prefill prefix length is multiple of truncation align size
                # A typical use case is when deterministic inference is enabled with flashinfer attention backend,
                # we need the prefill prefix length to be multiple of attention split size
                # 若要求截断对齐大小（如 FlashInfer 确定性推理），进一步对齐截断长度
                if truncation_align_size is not None:
                    if trunc_len < truncation_align_size:
                        return AddReqResult.OTHER
                    else:
                        trunc_len = truncation_align_size * (
                            trunc_len // truncation_align_size
                        )

                # Chunked prefill（分块 prefill：设置分块后的输入长度和 fill_ids）
                req.set_extend_input_len(trunc_len)
                req.fill_ids = req.fill_ids[: len(req.prefix_indices) + trunc_len]

                self.can_run_list.append(req)
                self.new_chunked_req = req

                self._req_inc_lock_ref(req)
                self._update_prefill_budget(prefix_len, trunc_len, 0)

        return self.budget_state()

    def preempt_to_schedule(self, req: Req, server_args: ServerArgs) -> bool:
        """
        Preempt running requests to serve the new request if the priority threshold is met and token count sum is verified.
        Returns True if preemption was committed, and the new request can be scheduled.
        """
        # Iterate running requests to find preemptible requests
        # 确定优先级排序方向（值越小优先级越高 or 越低）
        priority_sign = 1 if server_args.schedule_low_priority_values_first else -1

        # NOTE: A request finishes in two phases:
        #   1) check_finished + release_kv_cache  (in process_batch_result)
        #   2) filter out of batch                (in get_next_batch_to_run / update_running_batch)
        # Preemption runs between these two phases (inside get_new_batch_prefill),
        # so running_batch may still contain requests whose KV cache is already freed.
        # We must skip them here to avoid a double-free on release_req.
        # 过滤掉已完成（KV 已释放）和已在抢占列表中的请求
        valid_running_reqs = (
            r
            for r in self.running_batch.reqs
            if r not in self.preempt_list and not r.finished()
        )

        # 按优先级（反向）和进入时间（降序）排序，找到最应被抢占的请求
        sorted_valid_running_reqs = sorted(
            valid_running_reqs,
            key=lambda x: (
                x.priority * (-priority_sign),
                -x.time_stats.wait_queue_entry_time,
            ),
        )

        preemptible_reqs = []
        # 计算需要释放的最少 token 数量
        min_tokens_to_remove = (
            req.extend_input_len
            + min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
            - self.rem_total_tokens
        )
        for running_req in sorted_valid_running_reqs:
            # Priority difference needs to meet the threshold to be preemptible.
            # 优先级差值需超过阈值才可被抢占
            priority_diff = (req.priority - running_req.priority) * (-priority_sign)

            if priority_diff > self.priority_scheduling_preemption_threshold:
                preemptible_reqs.append(running_req)
                min_tokens_to_remove -= self._get_running_request_total_token_offset(
                    running_req
                )
                if min_tokens_to_remove <= 0:
                    # 已收集足够释放量，停止遍历
                    break
            else:
                # 后续请求优先级差不满足，停止遍历
                break

        # Check max token count limit can be met
        # 若没有可抢占请求或释放量仍不足，返回 False
        if len(preemptible_reqs) == 0 or min_tokens_to_remove > 0:
            return False

        # Preempt running requests. Release allocated resources for immediate usage.
        # 执行抢占：释放被抢占请求的 KV 资源，从运行批次中移除
        preemptible_reqs = set(preemptible_reqs)
        keep_indices = []
        release_counter = 0
        for i, running_req in enumerate(self.running_batch.reqs):
            if running_req in preemptible_reqs:
                # 归还被抢占请求占用的 token 预算
                self.rem_total_token_offset -= (
                    self._get_running_request_total_token_offset(running_req)
                )
                release_counter += 1
                self.running_batch.release_req(
                    i, len(self.running_batch.reqs) - release_counter, server_args
                )
            else:
                keep_indices.append(i)
        # 过滤运行批次，保留未被抢占的请求
        self.running_batch.filter_batch(keep_indices=keep_indices)
        # 记录已抢占的请求，避免后续重复抢占
        self.preempt_list.extend(preemptible_reqs)
        return True
