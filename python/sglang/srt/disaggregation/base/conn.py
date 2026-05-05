# PD 分离传输层基础抽象定义：KV 参数结构、传输状态枚举及 Manager/Sender/Receiver/Bootstrap 抽象基类
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import numpy.typing as npt

from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.disaggregation.utils import DisaggregationMode


# KV 传输参数数据类：描述一个节点上 KV 缓存、aux 缓存和 state 缓存的内存布局信息
class KVArgs:
    engine_rank: int          # 传输引擎在 TP 组内的排名
    kv_data_ptrs: List[int]   # 每层 KV 缓存的内存指针列表
    kv_data_lens: List[int]   # 每层 KV 缓存的总字节长度
    kv_item_lens: List[int]   # 每个 token/page 的 KV 缓存字节长度
    aux_data_ptrs: List[int]  # 辅助数据（如 MLA 压缩 KV）的内存指针
    aux_data_lens: List[int]  # 辅助数据的总字节长度
    aux_item_lens: List[int]  # 每个 token/page 辅助数据的字节长度
    state_data_ptrs: List[int]  # 状态数据（Mamba/SWA 等）的内存指针
    state_data_lens: List[int]  # 状态数据的总字节长度
    state_item_lens: List[int]  # 每个 token/page 状态数据的字节长度
    state_type: str  # "none", "mamba", "swa"  # 状态类型：none/mamba/swa
    # for mamba state different tp slice transfer
    # Mamba 状态在不同 TP rank 间切分时每个张量的切片维度
    state_dim_per_tensor: List[int]  # dimension to slice for each state tensor
    ib_device: str           # InfiniBand 设备名称（用于 RDMA 传输）
    ib_traffic_class: str    # IB 流量类别（QoS 配置）
    gpu_id: int              # GPU/NPU 设备 ID
    kv_head_num: int         # 本节点的 KV head 数量（TP 切分后）
    total_kv_head_num: int   # 模型原始 KV head 总数
    page_size: int           # KV 缓存 page 大小（token 数）
    # for pp prefill
    # 流水线并行相关：PP rank 和本阶段起始层编号
    pp_rank: int
    prefill_start_layer: int
    # for system dp
    # 系统级数据并行的 rank，用于路由和负载均衡
    system_dp_rank: int


# KV 传输状态枚举：描述一次 KV 传输的生命周期阶段
class KVPoll:
    Failed = 0           # 传输失败
    Bootstrapping = 1    # 正在进行 bootstrap 握手（交换地址/元数据）
    WaitingForInput = 2  # 握手完成，等待调用方提供 KV 索引等输入
    Transferring = 3     # 正在传输 KV 数据
    Success = 4          # 传输成功完成


# KV 管理器抽象基类：负责初始化传输引擎并向 bootstrap 服务器注册本节点信息
class BaseKVManager(ABC):
    """Base class for managing transfer states"""

    @abstractmethod
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ): ...

    @abstractmethod
    def register_to_bootstrap(self):
        """Register prefill server info to the bootstrap server."""
        # 将 prefill 节点的传输端点信息注册到 bootstrap 服务器，供 decode 端发现
        ...


# KV 发送器抽象基类：prefill 侧主动将 KV 缓存推送到 decode 侧
class BaseKVSender(ABC):

    @abstractmethod
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ): ...

    @abstractmethod
    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        """
        Set req's index metadata locally or notify the decoder server about the kv indices length and aux index.
        """
        # 在本地保存请求的 KV 索引数量和 aux 索引，或通知 decode 端准备接收
        ...

    @abstractmethod
    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List[int]] = None,
    ):
        """
        Send the kv cache at the given kv indices and the extra cache/state at the given indices to the decoder server.
        """
        # 将指定 KV 索引对应的缓存数据发送到 decode 节点，可选附带 state 索引
        ...

    @abstractmethod
    def poll(self) -> KVPoll:
        """
        Check the status of the kv cache transfer.
        """
        # 查询当前 KV 传输的状态，返回 KVPoll 枚举值
        ...

    @abstractmethod
    def failure_exception(self):
        """
        Raise an exception if the kv cache transfer fails.
        """
        # 传输失败时抛出异常，供上层捕获处理
        ...


# KV 接收器抽象基类：decode 侧等待并接收来自 prefill 侧的 KV 缓存
class BaseKVReceiver(ABC):

    @abstractmethod
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ): ...

    @abstractmethod
    def init(
        self,
        prefill_dp_rank: int,
    ):
        """
        Resolve bootstrap metadata and mark the receiver ready for transfer metadata.
        """
        # 解析 bootstrap 元数据（获取 prefill 端地址），将接收器标记为就绪
        ...

    @abstractmethod
    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
    ):
        """
        Notify the prefill server about the kv indices, aux index, and state_indices.
        """
        # 向 prefill 端发送本次请求的目标 KV 索引、aux 索引和 state 索引（元数据交换）
        ...

    @abstractmethod
    def poll(self) -> KVPoll:
        """
        Check the status of the kv cache transfer.
        """
        # 查询接收侧 KV 传输状态
        ...

    @abstractmethod
    def failure_exception(self):
        """
        Raise an exception if the kv cache transfer fails.
        """
        # 传输失败时抛出异常
        ...

    def clear(self):
        """
        Clear any internal states.
        """
        # 清除接收器内部状态（如 staging 缓冲区），默认空实现
        pass

    def abort(self):
        """
        Abort the current transfer.
        """
        # 中止当前传输，默认空实现
        pass


# KV 引导服务器抽象基类：负责在 prefill 和 decode 间协调连接建立
class BaseKVBootstrapServer(ABC):
    @abstractmethod
    def __init__(self, host: str, port: int): ...
