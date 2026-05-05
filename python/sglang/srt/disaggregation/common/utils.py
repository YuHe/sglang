# 线程和数据结构工具模块，用于 PD 分离场景下的并发安全队列和 KV 缓存索引分组
import threading
from collections import deque
from typing import List, Tuple

import numpy as np
import numpy.typing as npt


# 基于 deque + Condition 实现的线程安全快速队列，用于跨线程传递 KV 传输任务
class FastQueue:
    def __init__(self):
        # 内部双端队列，用于存储待处理的元素
        self._buf = deque()
        # 条件变量，用于阻塞/唤醒等待线程
        self._cond = threading.Condition()

    def put(self, item):
        # 向队列中放入元素，并唤醒一个等待中的消费者线程
        with self._cond:
            self._buf.append(item)
            # wake up a thread of wait()
            self._cond.notify()

    def get(self):
        # 从队列中取出元素，若队列为空则阻塞直到有新元素到来
        with self._cond:
            # if queue is empty  ,block until is notified()
            while not self._buf:
                self._cond.wait()
            return self._buf.popleft()


def group_concurrent_contiguous(
    src_indices: npt.NDArray[np.int32], dst_indices: npt.NDArray[np.int32]
) -> Tuple[List[npt.NDArray[np.int32]], List[npt.NDArray[np.int32]]]:
    """Vectorised NumPy implementation."""
    # 将 KV 缓存的源/目标索引按连续段分组，便于批量传输时减少传输次数
    # 若索引数组为空，直接返回空列表
    if src_indices.size == 0:
        return [], []

    # 找出连续性断裂点：源或目标索引差值不为 1 的位置即为段边界
    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    # 按断裂点将源索引和目标索引分别切分为连续段列表
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    # 将 numpy 数组段转换为 Python list，方便后续序列化/传输
    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups
