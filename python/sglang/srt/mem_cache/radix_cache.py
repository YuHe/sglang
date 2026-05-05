from __future__ import annotations

# 导入缓存初始化参数类
from sglang.srt.mem_cache.cache_init_params import CacheInitParams

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the KV cache.
"""
# 本模块实现了基于前缀树（Radix Tree）的 KV 缓存管理，支持多种驱逐策略

import hashlib  # 用于计算 SHA256 哈希值，实现位置感知的页面标识
import heapq    # 用于维护驱逐堆，高效获取最低优先级节点
import logging  # 日志模块
import sys      # 用于获取最大整数值（根节点优先级设置）
import time     # 用于记录访问时间和创建时间
from collections import defaultdict  # 用于树节点的子节点字典（支持默认值）
from functools import lru_cache      # 用于缓存前缀哈希值计算结果
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Tuple, Union

import torch  # PyTorch，用于管理 KV 缓存的张量数据

# 创建本模块的日志记录器
logger = logging.getLogger(__name__)

# 导入 KV 缓存事件类型，用于记录缓存块的存储/删除/清空事件
from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    StorageMedium,
)
# 导入前缀缓存基类和相关参数/结果数据类
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
# 导入所有支持的驱逐策略类
from sglang.srt.mem_cache.evict_policy import (
    EvictionStrategy,
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
    SLRUStrategy,
)
# 导入字符串哈希到 int64 的工具函数
from sglang.srt.mem_cache.utils import hash_str_to_int64

# 仅在类型检查时导入 Req，避免循环导入
if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


# RadixKey：前缀树中的键类型，支持普通 token 序列和 bigram（双字节对）视图
class RadixKey:
    """is_bigram=True: token_ids holds raw tokens (N+1 for N bigrams); slices share one boundary token."""

    __slots__ = ("token_ids", "extra_key", "is_bigram")

    def __init__(
        self,
        token_ids: List[int],
        extra_key: Optional[str] = None,
        is_bigram: bool = False,
    ):
        # token ids sequence (raw ints in both modes)
        # 存储原始 token id 序列（无论是否为 bigram 模式）
        self.token_ids = token_ids
        # extra key (e.g. lora_id, cache_salt)
        # 额外键，用于命名空间隔离（如 LoRA id、缓存盐值）
        self.extra_key = extra_key
        # bigram view over token_ids: length = max(0, len(token_ids) - 1)
        # 是否启用 bigram 视图：N 个原始 token 对应 N-1 个 bigram 单元
        self.is_bigram = is_bigram

    def __len__(self) -> int:
        # bigram 模式下逻辑长度比原始 token 数少 1
        if self.is_bigram:
            n = len(self.token_ids)
            return n - 1 if n > 0 else 0
        return len(self.token_ids)

    def __iter__(self) -> Iterator:
        # bigram 模式：逐个生成相邻 token 对 (t[i], t[i+1])
        if self.is_bigram:
            t = self.token_ids
            for i in range(len(t) - 1):
                yield (t[i], t[i + 1])
        else:
            # 普通模式：直接迭代 token id
            yield from self.token_ids

    def __getitem__(self, idx: Union[int, slice]) -> "RadixKey":
        # Normalize int -> 1-element slice so the rest handles one shape.
        # 将整数索引规范化为单元素切片，统一处理逻辑
        if isinstance(idx, int):
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError(f"RadixKey index out of range: {idx}")
            idx = slice(idx, idx + 1)
        start, stop, step = idx.indices(len(self))
        if step != 1:
            raise ValueError("RadixKey slice step must be 1")

        if self.is_bigram:
            # bigrams [start, stop) span raw tokens [start, stop + 1);
            # empty slice -> empty raw tokens (not a dangling boundary token).
            # bigram 切片 [start, stop) 对应原始 token [start, stop+1)；空切片返回空列表
            raw = self.token_ids[start : stop + 1] if stop > start else []
            return RadixKey(raw, self.extra_key, is_bigram=True)
        return RadixKey(self.token_ids[start:stop], self.extra_key)

    def __repr__(self) -> str:
        # 只展示前 10 个 token 的预览信息
        preview = self.token_ids[:10]
        return f"RadixKey(extra_key={self.extra_key!r}, token_ids={preview}{'...' if len(self.token_ids) > 10 else ''}, is_bigram={self.is_bigram})"

    def page_aligned(self, page_size: int) -> "RadixKey":
        # 将逻辑长度对齐到 page_size 的整数倍，截去末尾不完整的页
        if page_size == 1:
            return self
        aligned_len = len(self) // page_size * page_size
        return self[:aligned_len]

    def maybe_to_bigram_view(
        self,
        is_eagle: bool,
        value: Optional[torch.Tensor] = None,
    ) -> Tuple["RadixKey", Optional[torch.Tensor]]:
        # O(1): flip the bigram flag instead of materializing a tuple list.
        # value is paired with raw tokens and gets truncated to the bigram count.
        # O(1) 操作：仅翻转 bigram 标志，不重建 tuple 列表；value 对应 bigram 数量截断
        if is_eagle and not self.is_bigram:
            self.is_bigram = True
            if value is not None:
                value = value[: len(self)]
        return self, value

    def _check_compatible(self, other: "RadixKey") -> None:
        # 校验两个 RadixKey 的 extra_key 是否匹配，不匹配则抛出异常
        if self.extra_key != other.extra_key:
            raise ValueError(
                f"RadixKey operations require matching extra_key, but got "
                f"{self.extra_key=} != {other.extra_key=}"
            )

    def match(self, other: "RadixKey", page_size: int = 1) -> int:
        """Logical-unit prefix length shared with ``other``. Result is rounded down to ``page_size``."""
        # 计算两个 RadixKey 的公共前缀长度（以逻辑单元计），结果向下对齐到 page_size
        self._check_compatible(other)
        t0, t1 = self.token_ids, other.token_ids

        if self.is_bigram:
            # Walk raw tokens; L matching tokens imply L-1 matching bigrams.
            # 遍历原始 token：L 个匹配的 token 意味着 L-1 个匹配的 bigram
            i = 0
            for a, b in zip(t0, t1):
                if a != b:
                    break
                i += 1
            matched = max(0, min(i - 1, len(self), len(other)))
            return (matched // page_size) * page_size if page_size > 1 else matched

        if page_size == 1:
            # 逐个比较 token，找到第一个不同位置
            i = 0
            for a, b in zip(t0, t1):
                if a != b:
                    break
                i += 1
            return i

        # 按页大小逐块比较，对齐匹配
        min_len = min(len(self), len(other))
        i = 0
        while i < min_len:
            if t0[i : i + page_size] != t1[i : i + page_size]:
                break
            i += page_size
        return i

    def child_key(self, page_size: int = 1):
        """Hashable dict-key for the first ``page_size`` logical units, namespaced by ``extra_key``."""
        # 生成可哈希的字典键，用于在父节点的 children 字典中定位子节点
        t = self.token_ids
        if self.is_bigram:
            if page_size == 1:
                plain = (t[0], t[1])  # bigram 模式下第一个单元为 (t[0], t[1])
            else:
                plain = tuple((t[j], t[j + 1]) for j in range(page_size))
        else:
            plain = t[0] if page_size == 1 else tuple(t[:page_size])
        # 若有 extra_key 则将其纳入键中以实现命名空间隔离
        return plain if self.extra_key is None else (self.extra_key, plain)

    def hash_page(self, start: int, end: int, prior_hash: Optional[str] = None) -> str:
        """SHA256 for logical units [start, end); bigram mode feeds overlapping (t_i, t_{i+1}) byte pairs."""
        # 计算逻辑单元 [start, end) 的 SHA256 哈希，支持链式哈希（使用前一页哈希作为初始值）
        hasher = hashlib.sha256()
        if prior_hash:
            # 将前一页哈希作为初始状态，实现位置感知的链式哈希
            hasher.update(bytes.fromhex(prior_hash))
        t = self.token_ids
        if self.is_bigram:
            # bigram 模式：将每对相邻 token 的字节写入哈希器
            for j in range(start, end):
                hasher.update(t[j].to_bytes(4, byteorder="little", signed=False))
                hasher.update(t[j + 1].to_bytes(4, byteorder="little", signed=False))
        else:
            # 普通模式：将每个 token 的 4 字节小端表示写入哈希器
            for j in range(start, end):
                hasher.update(t[j].to_bytes(4, byteorder="little", signed=False))
        return hasher.hexdigest()


# TreeNode：前缀树的节点类，存储 KV 缓存数据及元数据
class TreeNode:

    counter = 0  # 全局节点计数器，用于分配唯一节点 ID

    def __init__(self, id: Optional[int] = None, priority: int = 0):
        self.children = defaultdict(TreeNode)  # 子节点字典，键为 child_key
        self.parent: TreeNode = None           # 父节点引用
        self.key: RadixKey = None              # 该节点存储的 token 序列键
        self.value: Optional[torch.Tensor] = None  # GPU 上的 KV 缓存索引张量
        self.lock_ref = 0                      # 引用计数锁，>0 时节点受保护不被驱逐
        self.last_access_time = time.monotonic()  # 最后访问时间，用于 LRU/MRU 策略
        self.creation_time = time.monotonic()     # 创建时间，用于 FIFO/FILO 策略

        self.hit_count = 0  # 命中次数，用于 LFU/SLRU 策略
        # indicating the node is locked to protect from eviction
        # incremented when the node is referenced by a storage operation
        # Host 引用计数，保护 host_value 不被驱逐，存储操作引用时递增
        self.host_ref_counter = 0
        # store the host indices of KV cache
        # 存储 Host（CPU）端的 KV 缓存索引，用于分级缓存
        self.host_value: Optional[torch.Tensor] = None
        # store hash values of each pages
        # 存储每个页面的哈希值列表，用于 KV 缓存事件记录和位置感知标识
        self.hash_value: Optional[List[str]] = None
        # priority for priority-aware eviction
        # 优先级数值，用于优先级感知驱逐策略
        self.priority = priority

        # 分配唯一 ID：若未指定则使用全局计数器
        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        # 节点已被驱逐（GPU 缓存值为空）
        return self.value is None

    @property
    def backuped(self):
        # 节点已备份到 Host 端（host_value 不为空）
        return self.host_value is not None

    def protect_host(self):
        """Protect the host value from eviction."""
        # 增加 host 引用计数，防止 host_value 被驱逐
        self.host_ref_counter += 1

    def release_host(self):
        """Release the host value, allowing it to be evicted."""
        # 减少 host 引用计数；若计数已为 0 则抛出异常
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            raise RuntimeError("Host reference counter is already zero.")

    def get_last_hash_value(self) -> Optional[str]:
        """Returns the hash value of the last page in this node."""
        # 返回该节点最后一个页面的哈希值，若无哈希值则返回 None
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    @lru_cache(maxsize=1)
    def get_prefix_hash_values(self, node: TreeNode) -> List[str]:
        # 递归收集从根节点到当前节点的所有哈希值列表（使用 lru_cache 缓存结果）
        if node is None or node.hash_value is None:
            return []

        return node.get_prefix_hash_values(node.parent) + node.hash_value

    def __lt__(self, other: "TreeNode"):
        # 定义节点比较顺序（按最后访问时间），用于堆操作
        return self.last_access_time < other.last_access_time


def compute_node_hash_values(node: "TreeNode", page_size: int) -> List[str]:
    """Compute SHA256-based hash values for position-aware identification."""
    # 为节点的每个页面计算位置感知的 SHA256 哈希值，用于 KV 缓存事件标识
    hash_values = []

    # 获取父节点最后一个页面的哈希值，作为当前节点第一页的链式哈希初始值
    parent_hash = None
    if node.parent is not None and node.parent.hash_value is not None:
        if len(node.parent.key) > 0 and len(node.parent.hash_value) > 0:
            parent_hash = node.parent.hash_value[-1]

    # 按页大小逐页计算哈希，前一页哈希作为后一页的输入实现链式依赖
    logical_len = len(node.key)
    for start in range(0, logical_len, page_size):
        end = min(start + page_size, logical_len)
        if end <= start:
            continue
        hash_val = node.key.hash_page(start, end, parent_hash)
        hash_values.append(hash_val)
        parent_hash = hash_val  # 当前页哈希作为下一页的前置哈希
    return hash_values


def split_node_hash_value(
    child_hash_value: Optional[List[str]], split_len: int, page_size: int
) -> tuple[Optional[List[str]], Optional[List[str]]]:
    """Split hash_value between parent and child nodes during node splitting.

    Args:
        child_hash_value: The hash_value list from the child node being split
        split_len: The length at which to split (in tokens)
        page_size: The page size for calculating number of pages

    Returns:
        Tuple of (new_node_hash_value, updated_child_hash_value)
    """
    # 节点分裂时将哈希值列表分配给新创建的父节点和保留的子节点
    if child_hash_value is None:
        return None, None

    # 计算分裂点对应的页面数
    if page_size == 1:
        split_pages = split_len
    else:
        split_pages = split_len // page_size

    # 前 split_pages 个哈希值归新节点（父），剩余归子节点
    new_node_hash = child_hash_value[:split_pages]
    child_hash = child_hash_value[split_pages:]

    return new_node_hash, child_hash


# RadixCache：基于前缀树的 KV 缓存管理器，实现高效前缀复用和多策略驱逐
class RadixCache(BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        self.disable = params.disable  # 是否禁用缓存（禁用时跳过所有缓存操作）
        self.req_to_token_pool = params.req_to_token_pool  # 请求到 token 的映射池
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator  # KV 缓存分配器
        self.page_size = params.page_size  # 页大小，决定缓存分配的最小粒度
        self.enable_kv_cache_events = params.enable_kv_cache_events  # 是否启用 KV 缓存事件记录
        self.is_eagle = params.is_eagle  # 是否使用 EAGLE 模式（bigram 键）
        self.disable_finished_insert = params.disable_finished_insert  # 是否禁用完成请求的插入（确定性模式）
        self.eviction_policy = params.eviction_policy.lower()  # 驱逐策略名称（小写）

        self.kv_event_queue = []  # KV 缓存事件队列，存储待发送的事件

        # 若启用指标收集则初始化指标收集器
        if params.enable_metrics:
            self.init_metrics_collector()

        # 根据 KV 池分配器确定设备（GPU/CPU）
        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        # 根据配置的驱逐策略名称实例化对应策略对象
        if self.eviction_policy == "lru":
            self.eviction_strategy: EvictionStrategy = LRUStrategy()
        elif self.eviction_policy == "lfu":
            self.eviction_strategy: EvictionStrategy = LFUStrategy()
        elif self.eviction_policy == "fifo":
            self.eviction_strategy: EvictionStrategy = FIFOStrategy()
        elif self.eviction_policy == "mru":
            self.eviction_strategy: EvictionStrategy = MRUStrategy()
        elif self.eviction_policy == "filo":
            self.eviction_strategy: EvictionStrategy = FILOStrategy()
        elif self.eviction_policy == "priority":
            self.eviction_strategy: EvictionStrategy = PriorityStrategy()
        elif self.eviction_policy == "slru":
            self.eviction_strategy: EvictionStrategy = SLRUStrategy()

        else:
            raise ValueError(
                f"Unknown eviction policy: {self.eviction_policy}. Supported policies: 'lru', 'lfu', 'fifo', 'mru', 'filo', 'priority', 'slru'."
            )

        self.evictable_leaves = set()  # 当前可驱逐的叶节点集合（无锁且未被驱逐）
        self.reset()  # 初始化/重置前缀树状态

    @classmethod
    def create_simulated(
        self,
        disable: bool = False,
        mock_allocator: Optional[Any] = None,
        page_size: int = 1,
        enable_kv_cache_events: bool = False,
    ) -> RadixCache:
        """Init a radix cache without memory pools for simulation purpose."""
        # 创建不依赖内存池的模拟 RadixCache，用于单元测试和仿真
        params = CacheInitParams(
            disable=disable,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=page_size,
            enable_kv_cache_events=enable_kv_cache_events,
        )
        return RadixCache(params)

    ##### Public API #####

    def reset(self):
        # Initialize root with minimum priority so any real priority overrides it
        # 初始化根节点（使用最小优先级，确保任何实际节点优先级都高于它）
        self.root_node = TreeNode(priority=-sys.maxsize)
        self.root_node.key = RadixKey(token_ids=[], extra_key=None)  # 根节点键为空序列
        self.root_node.value = []        # 根节点无 KV 缓存值
        self.root_node.host_value = []   # 根节点无 Host 缓存值
        self.root_node.lock_ref = 1      # 根节点始终锁定，防止被驱逐
        self.root_node.hash_value = []   # 根节点哈希值为空
        self.evictable_size_ = 0         # 可驱逐的缓存大小（token 数）
        self.protected_size_ = 0         # 受保护的缓存大小（token 数）
        self.evictable_leaves.clear()    # 清空可驱逐叶节点集合
        self._record_all_cleared_event() # 记录缓存全部清空事件

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Find the longest cached prefix of ``key`` in the radix tree.

        The logical namespace for prefix matching is determined by both the
        token id sequence and the optional ``extra_key`` carried by ``RadixKey``.
        Entries that share identical leading token ids but have *different*
        ``extra_key`` values are intentionally kept disjoint and never share
        prefix nodes. This is useful to:

        * Isolate KV cache lines for different LoRA / adapter IDs.
        * Separate requests that intentionally should not share state (e.g.,
          different sampling salt, cache version, or retrieval augmentation
          context) by supplying a distinct ``extra_key``.

        Args:
            params (MatchPrefixParams): Parameters containing the lookup key
                with a list of token ids and an optional ``extra_key`` namespace tag.
                If ``page_size > 1`` the length is internally truncated to a multiple
                of ``page_size`` before matching. Passing an empty key returns an
                empty result with the root as the last node.

        Returns:
            MatchResult: ``device_indices`` is a 1-D ``torch.int64`` tensor of
            the concatenated KV cache indices corresponding to the longest
            cached prefix (may be length 0). ``last_device_node`` and
            ``last_host_node`` (currently the same) are the tree node objects
            representing the terminal node of the matched prefix. This method
            may mutate internal structure by splitting an existing node if the
            match ends inside a stored segment.

        Internal updates:
            * Refreshes access metadata (timestamps) used by the
                configured eviction strategy.
            * If the lookup ends inside a stored segment the node is split once
                to expose a precise boundary; this structural refinement improves
                subsequent match efficiency and does not duplicate data.
        """
        # 获取键，并可选地转为 bigram 视图（EAGLE 模式）
        key = params.key
        key, _ = key.maybe_to_bigram_view(self.is_eagle)

        def empty_match_result():
            # 构造空匹配结果（无命中），设备索引为长度 0 的张量
            return MatchResult(
                device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
            )

        # 缓存禁用或键为空时直接返回空结果
        if self.disable or len(key) == 0:
            return empty_match_result()

        # 对齐到页大小，截去不完整的末尾页
        key = key.page_aligned(self.page_size)

        if len(key) == 0:
            return empty_match_result()

        # 在前缀树中递归查找最长匹配前缀
        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)  # 拼接所有命中节点的 KV 缓存索引
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)
        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
        )

    def insert(self, params: InsertParams) -> InsertResult:
        # 缓存禁用时直接返回前缀长度为 0
        if self.disable:
            return InsertResult(prefix_len=0)

        key = params.key
        value = params.value
        priority = params.priority
        chunked = params.chunked

        # 转为 bigram 视图并对齐到页大小
        key, value = key.maybe_to_bigram_view(self.is_eagle, value)
        key = key.page_aligned(self.page_size)
        if value is not None:
            value = value[: len(key)]  # 截断 value 使其与对齐后的 key 长度一致
        else:
            # Debug/test fallback: use token ids themselves as values.
            # 调试/测试回退：将 token id 本身用作 value（实际运行中不应触发）
            value = torch.tensor(key.token_ids[: len(key)], dtype=torch.int64)

        # 递归插入前缀树，返回已缓存的前缀长度
        prefix_len = self._insert_helper(self.root_node, key, value, priority, chunked)
        return InsertResult(prefix_len=prefix_len)

    def cache_finished_req(self, req: Req, is_insert: bool = True):
        """Cache request when it finishes."""
        # In deterministic mode, disable finished request insertion to radix cache
        # 确定性模式下禁用已完成请求的插入，保证输出确定性
        if self.disable_finished_insert:
            is_insert = False

        # 获取已提交的 KV 缓存长度（弹出并返回）
        kv_committed_len = req.pop_committed_kv_cache()
        if self.disable:
            # 缓存禁用时直接释放 KV 缓存索引
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            return

        # 构建完整的 token id 序列（输入 + 输出）
        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # 构造 RadixKey 并对齐到页大小
        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        key_len = len(radix_key)
        values = kv_indices[:key_len].to(dtype=torch.int64, copy=True)

        # Radix Cache takes one ref in memory pool
        # 将已完成请求的 KV 缓存插入前缀树
        if is_insert:
            priority = getattr(req, "priority", 0) or 0
            result = self.insert(
                InsertParams(key=radix_key, value=values, priority=priority)
            )
            new_prefix_len = result.prefix_len
            # Free the duplicates that were already in the tree
            # 释放树中已存在的重复 KV 缓存条目（避免内存泄漏）
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : new_prefix_len]
            )
        else:
            # 不插入时直接释放超出保护范围的 KV 缓存
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : key_len]
            )

        # free the unaligned tail
        # 释放页对齐后多余的末尾 token KV 缓存
        self.token_to_kv_pool_allocator.free(kv_indices[key_len:])

        # Remove req slot release the cache lock
        # 释放请求的缓存锁引用，允许节点被驱逐
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked=False):
        """Cache request when it is unfinished."""
        # 缓存未完成（流式/分块）请求的 KV 数据
        if self.disable:
            return

        # 获取当前已填充的 token id 和对应 KV 索引
        token_ids = req.fill_ids
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # 构造页对齐的 RadixKey
        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        values = kv_indices[: len(radix_key)].to(dtype=torch.int64, copy=True)

        # Radix Cache takes one ref in memory pool
        # 插入当前分块的 KV 缓存到前缀树
        result = self.insert(
            InsertParams(
                key=radix_key,
                value=values,
                chunked=chunked,
                priority=getattr(req, "priority", 0) or 0,
            )
        )
        new_prefix_len = result.prefix_len

        # 释放已插入树中的重复 KV 缓存
        self.token_to_kv_pool_allocator.free(
            kv_indices[req.cache_protected_len : new_prefix_len]
        )

        # The prefix indices could be updated, reuse it
        # 重新匹配前缀以获取最新的 KV 索引（插入可能更新了树结构）
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices, new_last_node = (
            match_result.device_indices,
            match_result.last_device_node,
        )
        assert len(new_indices) == len(
            radix_key
        ), f"{len(new_indices)=}, {len(radix_key)=}"

        # 将新的 KV 索引写回请求的 token 池（保护范围之后的部分）
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        # The cache_protected_len is not always equal to len(req.prefix_indices)
        # since for page_size > 1, the partial part is added to req.prefix_indices, but that part of kv indices is not added to the tree.
        # It should be freed in the next cache_unfinished_req and final cache_finished_req to avoid memory leak.
        # So we introduce this `cache_protected_len` field to make sure the partial part can be freed correctly.
        # 更新受保护长度：对于 page_size > 1，部分页不进树但需要记录，以便后续正确释放
        req.cache_protected_len = len(new_indices)

        # 释放旧锁引用，获取新节点的锁引用
        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        # - page_size != 1: there is a partial page at the end, keep the full kv_indices
        # - eagle case: bigram keys will only cache len - 1 kv indices
        # 更新 prefix_indices：若 new_indices 短于 kv_indices（部分页情况），补充末尾部分
        if len(new_indices) < len(kv_indices):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices

        req.last_node = new_last_node  # 更新请求当前指向的最后节点

    def pretty_print(self):
        # 打印前缀树的可读格式，用于调试
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        # 返回前缀树中所有节点的总 token 数
        return self._total_size_helper()

    def evict(self, params: EvictParams) -> EvictResult:
        # 缓存禁用时直接返回空结果
        if self.disable:
            return EvictResult()

        start_time = time.perf_counter()
        num_tokens = params.num_tokens  # 需要驱逐的 token 数量目标
        # 从可驱逐叶节点集合构建最小堆（按驱逐策略优先级排序）
        leaves = list(self.evictable_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)  # 建堆，O(n)

        num_evicted = 0
        # 持续从堆中弹出最低优先级节点直到驱逐足够的 token
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            # 释放该节点的 GPU KV 缓存内存
            self.token_to_kv_pool_allocator.free(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)  # 从树中删除叶节点

            # 若父节点因子节点删除而变为叶节点且无锁，将其加入驱逐堆
            if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

            self._record_remove_event(x)  # 记录块删除事件

        # 更新驱逐统计指标
        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        # 增加节点及其所有祖先的锁引用计数，将其从可驱逐状态转为受保护状态
        if self.disable:
            return IncLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                # 首次锁定：从可驱逐大小移出，加入受保护大小
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            self._update_leaf_status(node)  # 更新节点是否在可驱逐叶节点集合中
            node = node.parent
        return IncLockRefResult(delta=delta)

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        # 减少节点及其所有祖先的锁引用计数，可能使其重新变为可驱逐状态
        if self.disable:
            return DecLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                # 最后一个锁释放：从受保护大小移出，加入可驱逐大小
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            if node.parent is None:
                assert (
                    node is self.root_node
                ), f"This request holds the node from another tree"
            node = node.parent
        return DecLockRefResult(delta=delta)

    def evictable_size(self):
        # 返回当前可驱逐的缓存总大小（token 数）
        return self.evictable_size_

    def protected_size(self):
        # protected size refers to the size of the cache that is locked
        # 返回当前受保护（锁定）的缓存大小（token 数）
        return self.protected_size_

    def all_values_flatten(self):
        # 收集前缀树中所有节点的 value 并拼接为一维张量（用于调试/检查）
        values = []

        def _dfs_helper(node: TreeNode):
            # 深度优先遍历所有子节点，收集其 value
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values)

    ##### Internal Helper Functions #####

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        # 记录访问时间用于驱逐策略（LRU 等）
        access_time = time.monotonic()
        node.last_access_time = access_time

        # 计算键的第一个逻辑单元对应的子节点字典键
        child_key = key.child_key(self.page_size)

        value = []
        # 沿前缀树向下匹配，直到找不到匹配的子节点
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = access_time  # 更新命中节点的访问时间
            prefix_len = child.key.match(key, page_size=self.page_size)
            if prefix_len < len(child.key):
                # 部分匹配：在匹配点分裂节点，返回新节点
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break
            else:
                # 完全匹配当前节点，继续向下匹配剩余键
                value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = key.child_key(self.page_size)

        return value, node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # new_node -> child
        # New node inherits child's priority (represents shared prefix)
        # 在 split_len 处分裂节点：创建新父节点，child 成为其子节点
        new_node = TreeNode(priority=child.priority)
        new_node.hit_count = child.hit_count  # 继承命中次数
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref  # 继承锁引用计数
        new_node.key = child.key[:split_len]    # 新节点持有前半段键
        new_node.value = child.value[:split_len].clone()  # 克隆前半段 value
        child.parent = new_node
        child.key = child.key[split_len:]        # child 持有后半段键
        child.value = child.value[split_len:].clone()  # 克隆后半段 value
        new_node.parent.children[key.child_key(self.page_size)] = new_node

        # Split hash_value if it was already computed, otherwise leave as None
        # 分裂哈希值列表：若已计算则按页数分配给新旧节点，否则保持 None
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )

        return new_node

    def _inc_hit_count(self, node: TreeNode, chunked: bool = False):
        # Skip the hit count update for chunked requests to avoid self-referencing
        # inflation where a chunked request increments hit_count on nodes it created
        # in previous chunks.
        # 跳过分块请求的命中计数更新，避免分块请求对自己创建的节点进行自引用膨胀
        if chunked:
            return
        node.hit_count += 1

    def _insert_helper(
        self,
        node: TreeNode,
        key: RadixKey,
        value,
        priority: int = 0,
        chunked: bool = False,
    ):
        # Convert None priority to 0
        # 将 None 优先级转换为 0，避免比较错误
        if priority is None:
            priority = 0
        access_time = time.monotonic()
        node.last_access_time = access_time
        # Update priority along the path (take max to propagate higher priority)
        # 沿路径传播更高优先级（取最大值）
        node.priority = max(node.priority, priority)
        if len(key) == 0:
            return 0

        child_key = key.child_key(self.page_size)

        total_prefix_length = 0
        # 沿前缀树向下遍历，跳过已存在的前缀部分
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = access_time
            prefix_len = node.key.match(key, page_size=self.page_size)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                # 部分匹配：分裂节点后继续
                new_node = self._split_node(node.key, node, prefix_len)
                new_node.priority = max(new_node.priority, priority)
                self._inc_hit_count(new_node, chunked)
                node = new_node
            else:
                # 完全匹配当前节点，更新优先级和命中计数
                node.priority = max(node.priority, priority)
                self._inc_hit_count(node, chunked)
            if len(key):
                child_key = key.child_key(self.page_size)

        if len(key):
            # 插入新叶节点：存储剩余的 key 和 value
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()  # 克隆 value 避免外部修改影响缓存
            self._inc_hit_count(new_node, chunked)
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)  # 更新可驱逐大小
            self._update_leaf_status(node)    # 父节点可能不再是叶节点
            self._update_leaf_status(new_node)  # 新节点可能是叶节点
            # Hash will be computed lazily during event emission
            # 哈希值在事件发送时按需计算（惰性计算）
            self._record_store_event(new_node)
        return total_prefix_length

    def _print_helper(self, node: TreeNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        # 使用迭代栈（非递归）打印前缀树，避免深度过大时栈溢出
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current_node.key),
                current_node.key.token_ids[:10],
                f"r={current_node.lock_ref}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                # 验证父节点的子节点键与子节点的 child_key 一致
                assert key == child.key.child_key(
                    self.page_size
                ), f"{key=}, {child.key.child_key(self.page_size)=}"

    def _delete_leaf(self, node):
        # 从父节点的 children 字典中移除该叶节点
        key = node.key.child_key(self.page_size)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        # 更新可驱逐大小统计
        self.evictable_size_ -= len(node.key)
        if node in self.evictable_leaves:
            self.evictable_leaves.remove(node)
        self._update_leaf_status(node.parent)  # 父节点可能变为新的叶节点

    def _update_leaf_status(self, node: TreeNode):
        # 更新节点在可驱逐叶节点集合中的状态
        if node.evicted or node.lock_ref > 0:
            # 节点已被驱逐或被锁定，从可驱逐叶节点集合中移除
            if node in self.evictable_leaves:
                self.evictable_leaves.remove(node)
            return

        # 若有任何未被驱逐的子节点，则该节点不是叶节点
        for child in node.children.values():
            if not child.evicted:
                if node in self.evictable_leaves:
                    self.evictable_leaves.remove(node)
                return

        # 所有子节点均已驱逐或无子节点：该节点是叶节点，加入可驱逐集合
        if node not in self.evictable_leaves:
            self.evictable_leaves.add(node)

    def _total_size_helper(self):
        # 迭代遍历前缀树，统计所有未被驱逐节点的 value 总长度
        total_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            for child in current_node.children.values():
                if child.evicted:
                    continue  # 跳过已被驱逐的子节点
                stack.append(child)
        return total_size

    def _record_store_event(self, node: TreeNode, medium=None):
        # One BlockStored per ``page_size`` chunk.
        # ``medium`` defaults to StorageMedium.GPU but callers may override
        # for lower-tier insertions (e.g. StorageMedium.CPU for host/L2 cache).
        # 为节点的每个页面生成 BlockStored 事件，每个 page_size 大小的块一个事件
        if self.enable_kv_cache_events:
            if medium is None:
                medium = StorageMedium.GPU

            # Compute hash_value lazily if not already set
            # 惰性计算哈希值：若尚未计算则立即计算
            if node.hash_value is None:
                node.hash_value = compute_node_hash_values(node, self.page_size)

            # Get parent's last hash value for first page
            # 获取父节点最后一页的哈希值，作为当前节点第一页的父块哈希
            parent_block_hash = None
            if node.parent is not None and node.parent != self.root_node:
                if (
                    node.parent.hash_value is not None
                    and len(node.parent.hash_value) > 0
                ):
                    parent_block_hash = hash_str_to_int64(node.parent.hash_value[-1])

            page_index = 0
            logical_len = len(node.key)
            is_bigram = node.key.is_bigram
            raw = node.key.token_ids
            # 按页大小遍历节点的所有逻辑单元，为每页生成一个存储事件
            for start in range(0, logical_len, self.page_size):
                end = min(start + self.page_size, logical_len)
                if end <= start:
                    continue
                # Preserve historical event payload: bigram pages expose tuples.
                # 保持历史事件格式：bigram 模式下页面 token 以元组形式表示
                if is_bigram:
                    page_tokens = [(raw[j], raw[j + 1]) for j in range(start, end)]
                else:
                    page_tokens = raw[start:end]

                # 将哈希字符串转换为 int64 作为块哈希标识
                block_hash = hash_str_to_int64(node.hash_value[page_index])

                self.kv_event_queue.append(
                    BlockStored(
                        block_hashes=[block_hash],
                        parent_block_hash=parent_block_hash,
                        token_ids=page_tokens,
                        block_size=len(page_tokens),
                        lora_id=None,
                        medium=medium,
                    )
                )

                parent_block_hash = block_hash  # 当前页哈希作为下一页的父哈希
                page_index += 1

    def _record_remove_event(self, node: TreeNode, medium=None):
        # One BlockRemoved per chunk.
        # ``medium`` defaults to StorageMedium.GPU but callers may override for
        # lower-tier removals (e.g. StorageMedium.CPU when evicting from host).
        # 为节点的每个页面生成 BlockRemoved 事件
        if self.enable_kv_cache_events:
            if medium is None:
                medium = StorageMedium.GPU

            # Compute hash_value lazily if not already set (must match what was stored)
            # 惰性计算哈希值：必须与存储时的哈希值一致
            if node.hash_value is None:
                node.hash_value = compute_node_hash_values(node, self.page_size)

            page_index = 0
            logical_len = len(node.key)
            # 遍历所有页，生成对应的删除事件
            for start in range(0, logical_len, self.page_size):
                end = min(start + self.page_size, logical_len)
                if end <= start:
                    continue

                block_hash = hash_str_to_int64(node.hash_value[page_index])

                self.kv_event_queue.append(
                    BlockRemoved(block_hashes=[block_hash], medium=medium)
                )

                page_index += 1

    def _record_all_cleared_event(self):
        # 发送全部缓存清空事件（如初始化/重置时）
        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

    def take_events(self):
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        # 原子性地取出所有待发送事件并清空队列，返回事件列表
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []  # 重置队列
        return events


if __name__ == "__main__":
    # 创建模拟（不依赖实际内存池）的 RadixCache 实例进行测试
    tree = RadixCache.create_simulated()

    # Example token id sequences (as lists of ints)
    # 测试插入：相同前缀 [1,2,3] 应该只存储一次
    tree.insert(InsertParams(key=RadixKey(token_ids=[1, 2, 3], extra_key=None)))
    tree.insert(InsertParams(key=RadixKey(token_ids=[1, 2, 3], extra_key=None)))
    tree.insert(InsertParams(key=RadixKey(token_ids=[1, 2, 4, 5], extra_key=None)))
    tree.insert(
        InsertParams(key=RadixKey(token_ids=[1, 2, 4, 5, 6, 7], extra_key=None))
    )
    tree.insert(
        InsertParams(key=RadixKey(token_ids=[8, 9, 10, 11, 12], extra_key=None))
    )
    tree.pretty_print()  # 打印前缀树结构

    # 测试前缀匹配：查找 [1,2,3,13,14] 的最长缓存前缀
    print(
        tree.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=[1, 2, 3, 13, 14], extra_key=None))
        )
    )
