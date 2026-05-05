from __future__ import annotations

# 导入 ABC 抽象基类和 abstractmethod 装饰器，用于定义驱逐策略接口
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, Union

# 仅在类型检查时导入 TreeNode，避免循环导入
if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode


# 驱逐策略抽象基类，所有具体策略都必须实现 get_priority 方法
class EvictionStrategy(ABC):
    @abstractmethod
    def get_priority(self, node: "TreeNode") -> Union[float, Tuple]:
        # 返回节点的驱逐优先级，值越小越先被驱逐
        pass


# LRU（最近最少使用）策略：优先驱逐最久未访问的节点
class LRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        # 以最后访问时间作为优先级，时间越早值越小，越先被驱逐
        return node.last_access_time


# LFU（最少频率使用）策略：优先驱逐访问次数最少的节点，次数相同则驱逐最久未访问的
class LFUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        # 以 (访问次数, 最后访问时间) 作为优先级，次数越少越先驱逐
        return (node.hit_count, node.last_access_time)


# FIFO（先进先出）策略：优先驱逐最早创建的节点
class FIFOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        # 以创建时间作为优先级，创建越早的节点越先被驱逐
        return node.creation_time


# MRU（最近最多使用）策略：优先驱逐最近访问的节点（与 LRU 相反）
class MRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        # 取负值使最近访问的节点优先级最低（最先被驱逐）
        return -node.last_access_time


# FILO（后进先出）策略：优先驱逐最新创建的节点（与 FIFO 相反）
class FILOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        # 取负值使最新创建的节点优先级最低（最先被驱逐）
        return -node.creation_time


class PriorityStrategy(EvictionStrategy):
    """Priority-aware eviction: lower priority values evicted first, then LRU within same priority."""
    # 优先级感知驱逐策略：优先级数值越低越先被驱逐，同级别内按 LRU 驱逐

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        # Return (priority, last_access_time) so lower priority nodes are evicted first
        # 返回 (优先级, 最后访问时间)，低优先级节点先被驱逐
        return (node.priority, node.last_access_time)


# SLRU（分段 LRU）策略：将节点分为试用段和保护段，试用段中的节点优先被驱逐
class SLRUStrategy(EvictionStrategy):
    def __init__(self, protected_threshold: int = 2):
        # 设置保护阈值，访问次数达到该阈值的节点进入保护段
        self.protected_threshold = protected_threshold

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        # Priority Logic:
        # Smaller value = Evicted earlier.
        #
        # Segment 0 (Probationary): hit_count < threshold
        # Segment 1 (Protected): hit_count >= threshold
        #
        # Tuple comparison: (segment, last_access_time)
        # Nodes in segment 0 will always be evicted before segment 1.
        # Inside the same segment, older nodes (smaller time) are evicted first.
        # 优先级逻辑：
        # 较小的值 = 更早被驱逐
        # 段 0（试用段）：hit_count < 阈值
        # 段 1（保护段）：hit_count >= 阈值
        # 试用段节点总是在保护段节点之前被驱逐
        # 同段内，访问时间较早的节点先被驱逐

        # 根据访问次数判断节点是否属于保护段（1）或试用段（0）
        is_protected = 1 if node.hit_count >= self.protected_threshold else 0
        return (is_protected, node.last_access_time)
