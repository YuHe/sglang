"""
Copyright 2025 SGLang Team
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
KV caching events
"""
# KV 缓存事件模块：定义 KV 缓存块存储/删除/清空等事件及 ZMQ 发布者，用于 PD 分离场景中的事件监控

import atexit
import enum
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from itertools import count
from queue import Queue
from typing import Any, Callable, Optional, Union

# msgspec：高性能序列化库，用于 KV 事件的 msgpack 编解码
import msgspec
import zmq
# pydantic 用于 KVEventsConfig 的配置验证
from pydantic import BaseModel

logger = logging.getLogger(__name__)



# EventBatch：将多个事件打包为一个批次，包含时间戳和可选的 DP rank 标注
class EventBatch(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
):
    ts: float          # 批次时间戳（Unix 秒）
    events: list[Any]  # 批次中的事件列表
    attn_dp_rank: Optional[int] = None  # 注意力 DP rank（多 DP 时区分来源）


# KVCacheEvent：所有 KV 缓存事件的基类，使用 msgspec tag 实现多态序列化
class KVCacheEvent(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
    tag=True,
):
    """Base class for all KV cache-related events"""


# StorageMedium：KV 缓存的存储层级枚举（GPU HBM -> Host Pinned -> SSD -> 远程池）
class StorageMedium(str, enum.Enum):
    """Storage tier for KV cache events."""

    GPU = "GPU"  # L1: device HBM
    CPU = "CPU_PINNED"  # L2: host pinned memory
    DISK = "DISK"  # L3: SSD / NVMe
    EXTERNAL = "EXTERNAL"  # L4: shared / remote pool (e.g. Mooncake)


# OffloadedState：记录单个请求在 HiCache 中已卸载的 KV 缓存状态
class OffloadedState:
    """
    OffloadedState represents the state of a KV cache block offloaded to the hicache.

    - prefill_len (int): The length of the prefill part of the KV cache block.
    - inc_len (int): The length of the incremental part of the KV cache block.
    - last_hash (Optional[str]): The hash of the last token in the KV cache block.
    """

    def __init__(
        self, prefill_len: int, inc_len: int = 0, last_hash: Optional[str] = None
    ):
        # prefill_len: prefill 阶段已对齐卸载的 token 数
        self.prefill_len = prefill_len
        # inc_len: decode 阶段已增量卸载的 token 数
        self.inc_len = inc_len
        # last_hash: 最后一个已卸载 page 的 hash，用于续接增量 hash 链
        self.last_hash = last_hash


# BlockStored：KV 缓存块存储事件，记录块 hash、父 hash、token ids 等信息
class BlockStored(KVCacheEvent):
    block_hashes: list[int]              # 存储的块 hash 列表
    parent_block_hash: Optional[int]     # 父块 hash（前缀树结构）
    token_ids: list[int]                 # 块对应的 token id 列表
    block_size: int                      # 块大小（token 数）
    lora_id: Optional[int]              # LoRA adapter id（若有）
    medium: Optional[str] = None        # 存储介质（GPU/CPU/DISK/EXTERNAL）


# BlockRemoved：KV 缓存块删除事件
class BlockRemoved(KVCacheEvent):
    block_hashes: list[int]    # 被删除的块 hash 列表
    medium: Optional[str] = None  # 删除时所在的存储介质


# AllBlocksCleared：所有 KV 缓存块清空事件（如服务重启时触发）
class AllBlocksCleared(KVCacheEvent):
    pass


# KVEventBatch：专用于 KV 缓存事件的批次，限定事件类型为 BlockStored/BlockRemoved/AllBlocksCleared
class KVEventBatch(EventBatch):
    events: list[Union[BlockStored, BlockRemoved, AllBlocksCleared]]


# EventPublisher：事件发布者抽象基类，支持 DP 注意力并行场景下的事件分发
class EventPublisher(ABC):
    """
    Lightweight publisher for EventBatch batches with
    support for DP attention.

    In DP attention - each rank has its own Scheduler and
    KV cache instance in order to avoid duplicate events
    and ensure proper event attribution. In our implementation

    - Each DP rank has its own EventPublisher
    - Publishers annotate events with the dp rank
    - This allows consumers to distinguish events from different DP ranks
    """

    def __init__(self, attn_dp_rank: int = 0):
        # 记录所属的 attention DP rank
        self._attn_dp_rank = attn_dp_rank

    @abstractmethod
    def publish(self, events: EventBatch) -> None:
        """Emit events in order.

        Implementations should guarantee at-least-once delivery and
        monotonic ordering (e.g., via sequence numbers).
        """
        # 发布事件批次，实现须保证顺序性和至少一次投递

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the publisher."""
        # 关闭发布者，释放资源


# NullEventPublisher：空操作发布者，默认禁用时使用
class NullEventPublisher(EventPublisher):
    """No-op implementation (default when disabled)."""

    def publish(self, events) -> None:
        return

    def shutdown(self) -> None:
        return


# ZmqEventPublisher：基于 ZMQ PUB/ROUTER 的可靠事件发布者
# 内置内存回放缓冲区，支持订阅者请求历史事件；使用独立后台线程异步发布
class ZmqEventPublisher(EventPublisher):
    """Reliable PUB/ROUTER publisher with an in-memory replay buffer.

    Spawns a separate thread to handle publishing from a queue.

    Parameters
    ----------
    endpoint:
        PUB address. Use ``tcp://*:5557`` to bind or ``tcp://host:5557`` to
        connect.
    replay_endpoint:
        Optional ROUTER address for replay requests. When given, subscribers can
        request missed batches by sending the starting sequence number as an
        8-byte big-endian integer.
    buffer_steps:
        Number of past batches to keep for replay.
    hwm:
        ZeroMQ high-water-mark for PUB socket.
    max_queue_size:
        Maximum number of events to buffer in memory.
    topic:
        Topic to publish events to.
    """

    # 关闭时最长等待时间（秒）
    SHUTDOWN_TIMEOUT: float = 1.0
    # 回放结束标记：-1 的 8 字节大端编码
    END_SEQ = (-1).to_bytes(8, "big", signed=True)

    def __init__(
        self,
        attn_dp_rank: int,
        endpoint: str = "tcp://*:5557",
        replay_endpoint: Optional[str] = None,
        buffer_steps: int = 10_000,
        hwm: int = 100_000,
        max_queue_size: int = 100_000,
        topic: str = "",
    ) -> None:
        # Storage
        super().__init__(attn_dp_rank)
        # 内存队列：生产者（publish）和消费者（后台线程）之间的缓冲
        self._event_queue = Queue[Optional[EventBatch]](maxsize=max_queue_size)
        # 回放缓冲区：存储最近 buffer_steps 个批次的 (序号, 序列化数据)
        self._buffer = deque[tuple[int, bytes]](maxlen=buffer_steps)

        # ZMQ sockets
        self._ctx = zmq.Context.instance()
        self._pub: Optional[zmq.Socket] = None      # PUB socket：广播事件
        self._replay: Optional[zmq.Socket] = None   # ROUTER socket：处理回放请求
        self._dp_rank = attn_dp_rank
        # 根据 DP rank 偏移端口，避免多 DP rank 冲突
        self._endpoint = self.offset_endpoint_port(endpoint, self._dp_rank)
        self._replay_endpoint = self.offset_endpoint_port(
            replay_endpoint, self._dp_rank
        )
        self._hwm = hwm
        self._socket_setup()

        # Payload
        # 单调递增序号生成器，用于事件排序和回放定位
        self._seq_gen = count()
        self._topic_bytes = topic.encode("utf-8")

        # Thread
        self._running = True
        logger.info("Starting ZMQ publisher thread")

        # 启动后台发布线程（daemon，随主线程退出）
        self._thread = threading.Thread(
            target=self._publisher_thread, daemon=True, name="zmq-publisher"
        )
        self._thread.start()

        # 注册 atexit 钩子，确保进程退出时优雅关闭
        atexit.register(self.shutdown)

    def publish(self, events: EventBatch) -> None:
        # 将事件批次放入队列，由后台线程异步发布
        if not self._running:
            raise RuntimeError("Publisher is closed")
        if events.attn_dp_rank is None:
            events.attn_dp_rank = self._dp_rank
        self._event_queue.put(events)

    def shutdown(self) -> None:
        """Stop the publisher thread and clean up resources."""
        # 发送 None 哨兵值通知后台线程退出，并等待队列清空
        self._running = False
        self._event_queue.put_nowait(None)

        start = time.time()
        pending_items = True
        # 等待队列清空或超时
        while pending_items and (time.time() - start < self.SHUTDOWN_TIMEOUT):
            pending_items = not self._event_queue.empty()
            if pending_items:
                time.sleep(0.1)

        if pending_items:
            logger.warning(
                "Warning: Queue still has %s items after %s seconds timeout",
                self._event_queue.qsize(),
                self.SHUTDOWN_TIMEOUT,
            )

        if self._thread.is_alive():
            self._thread.join(timeout=self.SHUTDOWN_TIMEOUT)

        # Clean up ZMQ resources
        # 清理 ZMQ socket 资源（linger=0 立即关闭不等待未发消息）
        try:
            if self._pub is not None:
                self._pub.close(linger=0)
            if self._replay is not None:
                self._replay.close(linger=0)
        finally:
            pass  # Do not terminate context; other sockets may use it

    def _socket_setup(self) -> None:
        """Initialize sockets
        https://pyzmq.readthedocs.io/en/v19.0.0/morethanbindings.html#thread-safety
        """
        # 创建 PUB socket 并根据地址类型决定 bind 或 connect
        if self._pub is None:
            self._pub = self._ctx.socket(zmq.PUB)
            self._pub.set_hwm(self._hwm)
            # Heuristic: bind if wildcard / * present, else connect.
            # bind stable, connect volatile convention
            # 含通配符/IPC/inproc 时 bind，否则 connect
            if (
                "*" in self._endpoint
                or "::" in self._endpoint
                or self._endpoint.startswith("ipc://")
                or self._endpoint.startswith("inproc://")
            ):
                logger.debug(
                    f"ZmqEventPublisher socket publisher_endpoint bind to {self._endpoint}"
                )
                self._pub.bind(self._endpoint)
            else:
                self._pub.connect(self._endpoint)

        # Set up replay socket: use ROUTER
        # 1) handles multiple REQ clients (identities)
        # 2) lets us send back one request → many replies (streamed events)
        # 3) works in our non‑blocking poll loop alongside PUB
        # 回放 socket 使用 ROUTER 模式，支持多客户端身份识别和流式回放
        if self._replay_endpoint is not None:
            self._replay = self._ctx.socket(zmq.ROUTER)
            logger.debug(
                f"ZmqEventPublisher socket replay_endpoint bind to {self._replay_endpoint}"
            )
            self._replay.bind(self._replay_endpoint)

    def _publisher_thread(self) -> None:
        """Background thread that processes the event queue."""
        # 后台线程：持续从队列取事件、序列化后通过 PUB socket 发布
        self._pack = msgspec.msgpack.Encoder()

        assert self._pub is not None  # narrows type for mypy

        while self._running or self._event_queue.qsize() > 0:
            # --- replay (non-critical) ---------------------------------
            # 优先处理回放请求（非关键路径）
            if self._replay is not None and self._replay.poll(0):
                try:
                    self._service_replay()
                except Exception as e:
                    logger.exception("Error in replay: %s", e)

            # --- main queue (critical) ---------------------------------
            # 从队列取事件，超时 0.1s 后继续轮询
            try:
                event = self._event_queue.get(timeout=0.1)
                if event is None:
                    break  # Sentinel received, exit thread
            except queue.Empty:
                continue

            try:
                # 生成递增序号，序列化事件并多帧发布 [topic, seq_bytes, payload]
                seq = next(self._seq_gen)

                payload = self._pack.encode(event)
                seq_bytes = seq.to_bytes(8, "big")
                self._pub.send_multipart((self._topic_bytes, seq_bytes, payload))

                # 保存到回放缓冲区
                self._buffer.append((seq, payload))
                self._event_queue.task_done()

            except Exception as e:
                # Publishing failed;  back-off a bit to avoid a tight error loop
                # 发布失败时短暂休眠，避免紧密错误循环
                logger.exception("Error in publisher thread: %s", e)
                time.sleep(0.1)

    def _service_replay(self) -> None:
        """If a replay request is waiting, send buffered batches."""
        # 处理客户端的回放请求：从 start_seq 开始流式发送历史批次
        assert self._replay is not None  # narrows type for mypy

        frame = self._replay.recv_multipart()
        if len(frame) != 3:
            logger.warning("Invalid replay request: %s", frame)
            return
        client_id, _, start_seq_bytes = frame
        start_seq = int.from_bytes(start_seq_bytes, "big")

        for seq, buf in self._buffer:
            if seq >= start_seq:
                # [identity, empty_delim, seq_bytes, payload]
                # (identity, empty_delim) are stripped off by the router
                # receiving payload is (seq_bytes, payload)
                # 将历史批次发送给请求客户端（ROUTER 模式需携带 identity 帧）
                self._replay.send_multipart(
                    (client_id, b"", seq.to_bytes(8, "big"), buf)
                )
        # Send end of sequence marker
        # receiving payload is (-1, b""")
        # 发送结束标记，通知客户端回放完成
        self._replay.send_multipart((client_id, b"", self.END_SEQ, b""))

    @staticmethod
    def offset_endpoint_port(
        endpoint: Optional[str], data_parallel_rank: int
    ) -> Optional[str]:
        """Helper function to offset the port in an endpoint by
            the data parallel rank.

        Args:
            endpoint: The endpoint string
                (e.g., "tcp://*:5557" or "inproc://cache")
            data_parallel_rank: The data parallel rank to offset by

        Returns:
            The endpoint with the port offset by data_parallel_rank
                or suffix appended
        """
        # Do nothing if input is None or data_parallel_rank is 0
        # 根据 DP rank 偏移端口号（inproc 则追加后缀，tcp 则加端口偏移量）
        if not endpoint or data_parallel_rank == 0:
            return endpoint

        if "inproc" in endpoint:
            # inproc 地址追加 _dp{rank} 后缀
            return f"{endpoint}_dp{data_parallel_rank}"
        if "tcp" in endpoint:
            if endpoint and ":" in endpoint:
                # Get everything after the last colon (the port)
                # 解析端口并加上 DP rank 偏移
                last_colon_idx = endpoint.rfind(":")
                base_addr = endpoint[:last_colon_idx]
                base_port = int(endpoint[last_colon_idx + 1 :])
                new_port = base_port + data_parallel_rank
                return f"{base_addr}:{new_port}"
            return endpoint
        raise ValueError("Invalid endpoint: must contain 'inproc' or 'tcp'")


# KVEventsConfig：KV 事件发布配置，通过 pydantic BaseModel 验证
class KVEventsConfig(BaseModel):
    """Configuration for KV event publishing."""

    publisher: str = "null"
    """The publisher to use for publishing kv events. Can be "null", "zmq".
    """
    # 发布者类型：null（禁用）或 zmq

    endpoint: str = "tcp://*:5557"
    """The zmq endpoint to use for publishing kv events.
    """
    # ZMQ PUB socket 地址

    replay_endpoint: Optional[str] = None
    """The zmq endpoint to use for replaying kv events.
    """
    # ZMQ ROUTER socket 地址（用于历史事件回放），为 None 时禁用回放

    buffer_steps: int = 10_000
    """The number of steps to cache for replay endpoint. Will only save
    events from the last N steps for the replay endpoint.
    """
    # 回放缓冲区保留的历史批次数量

    hwm: int = 100_000
    """The zmq high water mark for the event publisher. After queueing N events,
    events will start dropping if the consumer is not keeping up.
    """
    # ZMQ 高水位标记：队列积压超过此值时开始丢弃事件

    max_queue_size: int = 100_000
    """The maximum number of events to queue while waiting for publishing.
    """
    # 内存队列最大容量

    topic: str = ""
    """The topic to use for the event publisher. Consumers can subscribe to
    this topic to receive events.
    """
    # ZMQ PUB topic，订阅者可按 topic 过滤事件

    @classmethod
    def from_cli(cls, cli_value: str) -> "KVEventsConfig":
        """Parse the CLI value for the event publisher config."""
        # 从 JSON 字符串解析配置（用于命令行参数传入）
        return KVEventsConfig.model_validate_json(cli_value)


# EventPublisherFactory：事件发布者工厂，支持注册自定义发布者类型
class EventPublisherFactory:
    # 注册表：名称 -> 构造函数
    _registry: dict[str, Callable[..., EventPublisher]] = {
        "null": NullEventPublisher,
        "zmq": ZmqEventPublisher,
    }

    @classmethod
    def register_publisher(cls, name: str, ctor: Callable[..., EventPublisher]) -> None:
        # 注册自定义发布者类型，名称重复时抛出 KeyError
        if name in cls._registry:
            raise KeyError(f"publisher '{name}' already registered")
        cls._registry[name] = ctor

    @classmethod
    def create(cls, config: Optional[str], attn_dp_rank: int = 0) -> EventPublisher:
        """Create publisher from a config mapping."""
        # 根据配置字符串创建对应的发布者实例
        if not config:
            return NullEventPublisher()
        config = KVEventsConfig.from_cli(config)
        config_dict = config.model_dump()

        kind = config_dict.pop("publisher", "null")
        try:
            constructor = cls._registry[kind]
        except KeyError as exc:
            raise ValueError(f"Unknown event publisher '{kind}'") from exc
        return constructor(attn_dp_rank=attn_dp_rank, **config_dict)
