# 编码接收器模块：在 PD 分离架构的 Prefill 端负责接收 embedding 数据
# 负责向编码服务器发送编码请求，并通过 ZMQ 或 Mooncake 接收返回的 embedding
# 支持 HTTP 和 gRPC 两种传输模式，支持多模态（图像/视频/音频）多编码器负载均衡
import asyncio
import itertools
import logging
import pickle
import random
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from enum import IntEnum
from http import HTTPStatus
from typing import TYPE_CHECKING, Dict, List, Optional

import aiohttp
import numpy as np
import torch
import zmq
import zmq.asyncio
from transformers import PretrainedConfig

from sglang.srt.distributed.parallel_state import (
    GroupCoordinator,
    get_mooncake_transfer_engine,
)
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import GenerateReqInput, TokenizedGenerateReqInput
from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
from sglang.srt.managers.schedule_batch import Modality, Req
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import ImageData
from sglang.srt.utils.hf_transformers_utils import get_processor
from sglang.srt.utils.network import (
    NetworkAddress,
    get_local_ip_auto,
    get_zmq_socket_on_host,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


def _grpc_target(url: str) -> str:
    # 将 gRPC URL 转换为 gRPC 目标地址（去除 grpc:// 前缀）
    if url.startswith("grpc://"):
        return url[len("grpc://") :]
    if url.startswith("grpcs://"):
        raise ValueError("grpcs:// is not supported; use grpc://")
    return url


def _normalize_embedding_ports(embedding_port):
    # 统一 embedding_port 为列表格式（None -> []，标量 -> [标量]）
    if embedding_port is None:
        return []
    if isinstance(embedding_port, list):
        return embedding_port
    return [embedding_port]


def _grpc_scheduler_receive_url(target, req_id, receive_url, receive_count):
    # 通过 gRPC 向编码服务器注册调度器接收 URL（同步调用，运行在线程中）
    import grpc
    from smg_grpc_proto import sglang_encoder_pb2, sglang_encoder_pb2_grpc

    timeout_secs = envs.SGLANG_ENCODER_GRPC_TIMEOUT_SECS.get()
    channel = grpc.insecure_channel(target)
    stub = sglang_encoder_pb2_grpc.SglangEncoderStub(channel)
    try:
        stub.SchedulerReceiveUrl(
            sglang_encoder_pb2.SchedulerReceiveUrlRequest(
                req_id=req_id,
                receive_url=receive_url,
                receive_count=receive_count,
            ),
            timeout=timeout_secs,
        )
    finally:
        channel.close()


def _grpc_encode_request(target, encode_request):
    # 通过 gRPC 向编码服务器发送编码请求，返回编码响应（同步调用）
    import grpc
    from smg_grpc_proto import sglang_encoder_pb2, sglang_encoder_pb2_grpc

    timeout_secs = envs.SGLANG_ENCODER_GRPC_TIMEOUT_SECS.get()
    channel = grpc.insecure_channel(target)
    stub = sglang_encoder_pb2_grpc.SglangEncoderStub(channel)
    try:
        response = stub.Encode(
            sglang_encoder_pb2.EncodeRequest(
                mm_items=encode_request["mm_items"],
                req_id=encode_request["req_id"],
                num_parts=encode_request["num_parts"],
                part_idx=encode_request["part_idx"],
                prefill_host=encode_request["prefill_host"],
                embedding_port=_normalize_embedding_ports(
                    encode_request["embedding_port"]
                ),
            ),
            timeout=timeout_secs,
        )
        return response
    finally:
        channel.close()


def _grpc_send_request(target, request_json):
    # 通过 gRPC 向编码服务器触发 Mooncake RDMA 传输（发送 session_id 和 buffer_address）
    import grpc
    from smg_grpc_proto import sglang_encoder_pb2, sglang_encoder_pb2_grpc

    timeout_secs = envs.SGLANG_ENCODER_GRPC_TIMEOUT_SECS.get()
    channel = grpc.insecure_channel(target)
    stub = sglang_encoder_pb2_grpc.SglangEncoderStub(channel)
    try:
        stub.Send(
            sglang_encoder_pb2.SendRequest(
                req_id=request_json["req_id"],
                prefill_host=request_json["prefill_host"],
                embedding_port=request_json["embedding_port"],
                session_id=request_json["session_id"],
                buffer_address=request_json["buffer_address"],
            ),
            timeout=timeout_secs,
        )
    finally:
        channel.close()


class EmbeddingData:
    # 编码服务器返回的 embedding 数据载体：包含 embedding 张量、网格维度、模态信息和错误状态
    def __init__(
        self,
        req_id,
        num_parts,
        part_idx,
        grid_dim,
        modality,
        embedding=None,
        embedding_shape=None,
        error_msg=None,
        error_code=None,
        **kwargs,
    ):
        self.req_id = req_id
        # num_parts：该请求总共分成几个部分（多编码器并行时 > 1）
        self.num_parts = num_parts
        # part_idx：当前部分的索引（0-based）
        self.part_idx = part_idx
        # grid_dim：patch 网格维度（T, H, W），用于恢复 embedding 的空间结构
        self.grid_dim = grid_dim
        self.modality = modality
        self.embedding = embedding
        self.send_time = None
        self.dtype = embedding.dtype if embedding is not None else None
        if embedding_shape is not None:
            self.shape = embedding_shape
        else:
            self.shape = list(embedding.shape) if embedding is not None else None
        self.error_msg = error_msg
        self.error_code = error_code
        # Store additional metadata (e.g., video_timestamps for qwen3_vl)
        # 存储额外元数据（如视频时间戳），通过 kwargs 动态注入
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_grid(self):
        """Get the grid dimension of the embedding, used for image/video/audio.
        获取 embedding 的网格维度（patch grid），图像/视频/音频通用"""
        return self.grid_dim

    def get_embedding(self):
        # 返回原始 embedding 张量
        return self.embedding

    def __repr__(self):
        return f"EmbeddingData(req_id={self.req_id}, num_parts={self.num_parts}, part_idx={self.part_idx}) error_msg={self.error_msg}"

    def copy_without_embedding(self):
        # 创建不含 embedding 数据的副本（用于 ZMQ 元数据帧发送）
        new_data = EmbeddingData(
            req_id=self.req_id,
            num_parts=self.num_parts,
            part_idx=self.part_idx,
            grid_dim=self.grid_dim,
            modality=self.modality,
            embedding=None,
            embedding_shape=self.shape,
            error_msg=self.error_msg,
            error_code=self.error_code,
        )
        for key, value in self.__dict__.items():
            if key.startswith("_") or key == "embedding":
                continue
            setattr(new_data, key, value)
        return new_data


# Modality -> (list attr name, whether to flatten grid for that list)
# 模态 -> (列表属性名, 是否展平 grid 张量)，用于 MultiModalEmbeddingData 中按模态存储 grid
_MODALITY_GRID_ATTRS = {
    Modality.IMAGE: ("img_grid_thw", False),
    Modality.VIDEO: ("video_grid_thw", False),
    Modality.AUDIO: ("audio_feature_lens", True),
}
# 视频模态的辅助元数据属性名（时间戳相关）
_VIDEO_META_ATTRS = ("video_timestamps", "second_per_grid_ts")


def _cat_grid(dims, flatten_items=False):
    """Concatenate non-None grid entries; supports tensor/ndarray/list inputs.
    将多个 grid 张量（来自不同 part）拼接为单一张量，忽略 None 项"""

    def _to_tensor(g):
        # 统一将 numpy 数组和列表转换为 torch.Tensor
        if isinstance(g, torch.Tensor):
            return g.cpu() if g.is_cuda else g
        if isinstance(g, np.ndarray):
            return torch.from_numpy(g)
        return torch.as_tensor(g)

    valid = []
    for g in dims:
        if g is None:
            continue
        t = _to_tensor(g)
        if flatten_items:
            t = t.flatten()
        elif t.ndim == 0:
            # Keep cat semantics stable for scalar-like metadata.
            # 标量张量需要先 unsqueeze 才能 cat
            t = t.unsqueeze(0)
        valid.append(t)

    return torch.cat(valid, dim=0) if valid else None


class MultiModalEmbeddingData(EmbeddingData):
    # 多部分 embedding 聚合容器：当一个请求被分配到多个编码器时，
    # 用此类收集所有 part 的 embedding，待 ready 后合并传给多模态处理器
    def __init__(
        self,
        part_idx,
        num_parts,
        req_id,
        grid_dim,
        modality,
        embedding,
        embedding_shape,
        **kwargs,
    ):
        super().__init__(
            req_id,
            num_parts,
            part_idx,
            grid_dim,
            modality,
            embedding,
            embedding_shape,
            **kwargs,
        )
        # 各 part 的图像/视频/音频 grid 列表（未到达的 part 为 None）
        self.img_grid_thw = [None] * num_parts
        self.video_grid_thw = [None] * num_parts
        self.audio_feature_lens = [None] * num_parts
        # 各 part 的模态类型列表
        self.modality_list = [
            modality if part_idx == i else None for i in range(num_parts)
        ]
        # 标记各 part 是否已接收完成
        self.ready_list = [i == part_idx for i in range(num_parts)]
        # 各 part 的 embedding 张量列表
        self.embedding_list = [
            embedding if i == part_idx else None for i in range(num_parts)
        ]
        # 各 part 的 embedding 形状列表
        self.embedding_shape_list = [
            embedding_shape if i == part_idx else None for i in range(num_parts)
        ]
        # 视频时间戳和每格时间（各 part 独立存储）
        self.video_timestamps = [None] * num_parts
        self.second_per_grid_ts = [None] * num_parts

        # 初始化首个 part 的 grid
        self._set_part_grid(part_idx, modality, self.get_grid())
        if modality == Modality.VIDEO:
            self._set_video_meta_for_part(part_idx, kwargs)

    def _set_part_grid(self, part_idx, modality, grid):
        """Set the grid for one part according to modality (IMAGE/VIDEO/AUDIO).
        将指定 part 的 grid 维度写入对应模态的列表中"""
        spec = _MODALITY_GRID_ATTRS.get(modality)
        if spec is None:
            raise ValueError(f"Invalid modality: {modality}")
        attr_name, flatten = spec
        value = grid.flatten() if flatten else grid
        getattr(self, attr_name)[part_idx] = value

    def _set_video_meta_for_part(self, part_idx, source):
        """Copy video_timestamps and second_per_grid_ts from source (dict or object).
        从字典或对象中提取视频元数据并写入对应 part 的位置"""
        for attr_name in _VIDEO_META_ATTRS:
            val = (
                source.get(attr_name)
                if isinstance(source, dict)
                else getattr(source, attr_name, None)
            )
            if val is not None:
                getattr(self, attr_name)[part_idx] = val

    @classmethod
    def from_embedding_data(cls, embedding_data: EmbeddingData):
        """Create MultiModalEmbeddingData from an EmbeddingData instance.
        从单个 EmbeddingData 创建 MultiModalEmbeddingData（首个 part 到达时调用）"""
        # Only forward known optional attrs (e.g. video metadata) so they land on the instance
        # 仅转发已知的可选属性（如视频元数据）
        extra = {}
        for attr in _VIDEO_META_ATTRS:
            val = getattr(embedding_data, attr, None)
            if val is not None:
                extra[attr] = val
        mm_data = cls(
            part_idx=embedding_data.part_idx,
            num_parts=embedding_data.num_parts,
            req_id=embedding_data.req_id,
            grid_dim=embedding_data.grid_dim,
            modality=embedding_data.modality,
            embedding=embedding_data.embedding,
            embedding_shape=embedding_data.shape,
            **extra,
        )
        mm_data.send_time = embedding_data.send_time
        return mm_data

    def __repr__(self):
        return f"MultiModalEmbeddingData(req_id={self.req_id}, num_parts={self.num_parts}, part_idx={self.part_idx}, modality={self.modality})"

    def get_embedding(self, is_concat=False):
        # 返回 embedding 列表或按模态拼接的 embedding 字典
        if is_concat:
            # 按模态分组，将同模态的 embedding 在 GPU 上拼接后转回 CPU
            groups = defaultdict(list)
            for i, e in enumerate(self.embedding_list):
                if e is not None:
                    groups[self.modality_list[i]].append(e.cuda())
            return {
                mod: torch.concat(tensors).to("cpu", non_blocking=True)
                for mod, tensors in groups.items()
            }
        return self.embedding_list

    @property
    def ready(self):
        # 当所有 part 都已接收时返回 True
        return sum(self.ready_list) == self.num_parts

    def get_mm_extra_meta(self):
        """Build kwargs for mm_processor.get_mm_data() from grid and optional video meta.
        构建传给 mm_processor.get_mm_data() 的额外元数据（grid 和视频时间戳等）"""
        kwargs = {
            "img_grid_thw": _cat_grid(self.img_grid_thw),
            "video_grid_thw": _cat_grid(self.video_grid_thw),
            "audio_feature_lens": _cat_grid(
                self.audio_feature_lens, flatten_items=True
            ),
        }
        # 合并各 part 的视频时间戳等元数据
        for attr in _VIDEO_META_ATTRS:
            lst = getattr(self, attr, None)
            if not lst:
                continue
            valid = [a for a in lst if a is not None]
            if valid:
                kwargs[attr] = list(itertools.chain(*valid))
        return kwargs

    def add(self, embedding_data: EmbeddingData):
        # 将新到达的 part embedding 合并到当前聚合对象中
        if self.req_id != embedding_data.req_id:
            logger.warning(
                f"Dropping embedding data with mismatched req_id: "
                f"expected {self.req_id}, got {embedding_data.req_id}"
            )
            return False
        assert not self.ready_list[embedding_data.part_idx]
        pid = embedding_data.part_idx
        # 标记该 part 已就绪并填充 embedding/grid 数据
        self.ready_list[pid] = True
        self.modality_list[pid] = embedding_data.modality
        self.embedding_list[pid] = embedding_data.get_embedding()
        self.embedding_shape_list[pid] = embedding_data.shape
        self._set_part_grid(pid, embedding_data.modality, embedding_data.get_grid())
        if embedding_data.modality == Modality.VIDEO:
            self._set_video_meta_for_part(pid, embedding_data)


# 等待图像请求的状态枚举
class WaitingImageRequestStatus(IntEnum):
    FAIL = -1
    PENDING = 0
    SUCCESS = 1
    TIMEOUT = -2


def create_part_req_id(original_req_id: str, part_idx: int) -> str:
    """Create a unique part request ID by appending part index suffix.
    为分片请求创建唯一 ID，格式为 {original_req_id}_local_part_{part_idx}"""
    return f"{original_req_id}_local_part_{part_idx}"


def extract_original_req_id(part_req_id: str) -> str:
    """Extract the original request ID from a part request ID.
    从分片请求 ID 中提取原始请求 ID"""
    if "_local_part_" in part_req_id:
        return part_req_id.rsplit("_local_part_", 1)[0]
    return part_req_id


def calculate_modality_num_parts(modalities, num_items_assigned):
    """
    Calculate total number of parts and number of parts per modality.
    计算各模态的分片数和总分片数

    Args:
        modalities: List of modalities in order
        num_items_assigned: Dictionary mapping modality to list of assignment counts per encoder

    Returns:
        Tuple of (total_num_parts, modality_num_parts_dict)
        - total_num_parts: Total number of parts across all modalities
        - modality_num_parts: Dictionary mapping modality to number of parts for that modality
    """
    total_num_parts = 0
    modality_num_parts = {}
    for modality in modalities:
        num_items_assigned_modality = num_items_assigned.get(modality)
        # 有分配项的编码器才算一个 part
        num_parts = sum(1 for x in num_items_assigned_modality if x != 0)
        modality_num_parts[modality] = num_parts
        total_num_parts += num_parts
    return total_num_parts, modality_num_parts


# For zmq_to_scheduler
# zmq_to_scheduler 模式下使用的等待请求对象：负责在后台线程中等待 embedding 到达
class WaitingImageRequest:
    def __init__(
        self,
        rid: str,
        recv_req: TokenizedGenerateReqInput,
        mm_processor,
        encoder_urls,
        host_name,
        receive_count,
    ):
        self.rid = rid
        self.recv_req = recv_req
        self.mm_inputs = None
        self.error = None
        self.thread = None
        self.mm_processor = mm_processor
        self.encoder_urls = encoder_urls
        self.host_name = host_name
        # receive_count：期望接收的 embedding 分片数（等于 TP 大小）
        self.receive_count = receive_count
        self.num_items_assigned = recv_req.num_items_assigned
        # 在本地随机端口创建 ZMQ PULL socket，等待编码器推送 embedding
        self.embedding_port, self.recv_socket = get_zmq_socket_on_host(
            zmq.Context(), zmq.PULL, host=host_name
        )
        logger.info(f"Waiting for input {self.embedding_port = }")
        self.recv_embedding_data = None
        # ok=1 pending=0 fail=-1
        self.status = WaitingImageRequestStatus.PENDING
        self.error_msg = None
        self.error_code = None
        self.start_time = time.time()

    def send_encode_request(self):
        # 向编码服务器注册本机 ZMQ 接收地址，触发编码器将 embedding 推送过来
        async def _send_single_request(session, url, payload):
            try:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    return await response.text()
            except Exception as e:
                logger.error(f"Failed to send request to {url}: {e}")
                raise

        async def send_embedding_port(req_id, receive_count, host_name, embedding_port):
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=1800)
            ) as session:
                tasks = []
                logger.info(f"{self.num_items_assigned = } ")

                # Calculate part_idx_offset similar to encode() method
                # 按模态顺序计算 part_idx 偏移量，与 encode() 方法保持一致
                modalities = list(self.num_items_assigned.keys())
                _, modality_num_parts = calculate_modality_num_parts(
                    modalities, self.num_items_assigned
                )

                part_idx_offset = 0
                for modality in modalities:
                    assigned_nums = self.num_items_assigned[modality]
                    num_parts = modality_num_parts[modality]
                    cum_idx = 0
                    for idx, assigned_num in enumerate(assigned_nums):
                        if assigned_num == 0:
                            continue
                        part_idx = part_idx_offset + cum_idx
                        part_req_id = create_part_req_id(req_id, part_idx)
                        encoder_url = self.encoder_urls[idx]
                        target_url = f"{encoder_url}/scheduler_receive_url"
                        payload = {
                            "req_id": part_req_id,  # use part_req_id to match encode request
                            "receive_count": receive_count,
                            "receive_url": NetworkAddress(
                                host_name, embedding_port
                            ).to_host_port_str(),
                            "modality": modality.name,
                        }
                        logger.info(
                            f"Preparing to send to {target_url} with part_req_id={part_req_id}"
                        )
                        task = _send_single_request(session, target_url, payload)
                        tasks.append(task)
                        cum_idx += 1
                    part_idx_offset += num_parts

                if not tasks:
                    logger.info("No tasks to send.")
                    return
                logger.info(f"Concurrently sending {len(tasks)} requests...")
                # 并发发送所有注册请求
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Request {i} failed: {result}")
                    else:
                        logger.debug(f"Request {i} succeeded.")

        asyncio.run(
            send_embedding_port(
                self.recv_req.rid,
                self.receive_count,
                self.host_name,
                self.embedding_port,
            )
        )

    def _try_recv_mm_data(self):
        # 非阻塞轮询 ZMQ socket，接收 embedding 分片并聚合
        if self.status != WaitingImageRequestStatus.PENDING:
            return
        while self.recv_embedding_data is None or not self.recv_embedding_data.ready:
            try:
                parts = self.recv_socket.recv_multipart(flags=zmq.NOBLOCK, copy=False)
            except zmq.Again:
                # No data available yet, wait a bit and retry
                # 暂无数据，本次轮询结束，等待下次调用
                return
            recv_obj: EmbeddingData = pickle.loads(parts[0])
            if getattr(recv_obj, "error_msg", None) is not None:
                logger.warning(
                    f"Received error signal from encoder for {self.rid}: {recv_obj.error_msg} {recv_obj.error_code = }"
                )
                self.error_msg = recv_obj.error_msg
                self.error_code = recv_obj.error_code
                self.status = WaitingImageRequestStatus.FAIL
                self.recv_socket.close()
                return

            # Extract original req_id from part_req_id and drop stale payloads
            # that may arrive on a reused ZMQ port after a prior request aborted.
            # 从分片 ID 中恢复原始 req_id，丢弃 ZMQ 端口复用导致的过期消息
            original_req_id = extract_original_req_id(recv_obj.req_id)
            if original_req_id != self.recv_req.rid:
                logger.warning(
                    f"Dropping stale embedding data: expected rid={self.recv_req.rid}, "
                    f"got rid={recv_obj.req_id} (likely from ZMQ port reuse)"
                )
                continue
            recv_obj.req_id = original_req_id

            # 从 ZMQ 消息的第二帧解析 embedding 张量（零拷贝接收后 clone）
            buffer = parts[1].buffer if hasattr(parts[1], "buffer") else parts[1]
            recv_obj.embedding = (
                torch.frombuffer(buffer, dtype=recv_obj.dtype)
                .reshape(recv_obj.shape)
                .clone()
            )

            # 首个 part 到达时创建聚合对象，后续 part 调用 add 合并
            if self.recv_embedding_data is None:
                self.recv_embedding_data = MultiModalEmbeddingData.from_embedding_data(
                    recv_obj
                )
            else:
                self.recv_embedding_data.add(recv_obj)

        # 所有 part 已就绪，调用 mm_processor 生成最终 mm_inputs
        recv_embedding = self.recv_embedding_data.get_embedding(is_concat=True)
        mm_inputs = self.mm_processor.get_mm_data(
            self.recv_req.input_text,
            recv_embedding,
            **self.recv_embedding_data.get_mm_extra_meta(),
        )
        self.recv_req.mm_inputs = mm_inputs
        self.recv_req.input_ids = mm_inputs.input_ids
        self.status = WaitingImageRequestStatus.SUCCESS
        self.recv_socket.close()


class WaitingImageRequestGrpc(WaitingImageRequest):
    # gRPC 模式下的等待请求对象：使用 gRPC 协议注册接收 URL
    def send_encode_request(self):
        async def send_embedding_port(req_id, receive_count, host_name, embedding_port):
            tasks = []
            # gRPC image-only: flatten modality dict to flat list
            # gRPC 模式当前仅支持图像模态
            assigned = list(self.num_items_assigned.values())[0]
            logger.info(f"num_items_assigned={assigned}")

            for idx, assigned_num in enumerate(assigned):
                if assigned_num == 0:
                    continue
                encoder_url = self.encoder_urls[idx]
                receive_url = f"{host_name}:{embedding_port}"
                target_url = f"{encoder_url}/SchedulerReceiveUrl"
                logger.info(f"Preparing to send to {target_url}")
                # 在线程中异步调用 gRPC 同步方法
                tasks.append(
                    asyncio.to_thread(
                        _grpc_scheduler_receive_url,
                        _grpc_target(encoder_url),
                        req_id,
                        receive_url,
                        receive_count,
                    )
                )

            if not tasks:
                logger.info("No tasks to send.")
                return
            logger.info(f"Concurrently sending {len(tasks)} requests...")
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Request {i} failed: {result}")
                else:
                    logger.debug(f"Request {i} succeeded.")

        asyncio.run(
            send_embedding_port(
                self.recv_req.rid,
                self.receive_count,
                self.host_name,
                self.embedding_port,
            )
        )


def _determine_tensor_transport_mode(server_args):
    # 根据是否跨节点部署决定 tensor 传输模式：单节点用 cuda_ipc，多节点用默认 CPU 传输
    is_cross_node = server_args.dist_init_addr

    if is_cross_node:
        # Fallback to default CPU transport for multi-node
        return "default"
    else:
        return "cuda_ipc"


class MMReceiverBase(ABC):
    # 多模态 embedding 接收器基类：管理编码请求的发送、embedding 的接收和多模态处理
    # 在 PD 分离的 Prefill 端初始化，支持 Mooncake RDMA 和 ZMQ 两种传输后端
    def __init__(
        self,
        server_args: ServerArgs,
        dtype: Optional[torch.dtype] = None,
        hf_config: Optional[PretrainedConfig] = None,
        pp_rank: Optional[int] = None,
        tp_rank: Optional[int] = None,
        tp_group: Optional[GroupCoordinator] = None,
        scheduler: Optional["Scheduler"] = None,
    ):
        # 异步 ZMQ 上下文，用于接收 embedding
        self.context = zmq.asyncio.Context(20)
        self.encoder_transfer_backend = server_args.encoder_transfer_backend
        self.encode_urls = server_args.encoder_urls
        self.host = get_local_ip_auto(server_args.host)
        if self.encoder_transfer_backend == "mooncake":
            # Mooncake 传输模式：初始化 RDMA 传输引擎，预分配 embedding 接收缓冲区
            self.dtype = dtype
            self.embeddings_engine = get_mooncake_transfer_engine()
            if self.embeddings_engine is None:
                from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
                    init_mooncake_transfer_engine,
                )

                self.embeddings_engine = init_mooncake_transfer_engine(
                    hostname=self.host,
                    ib_device=(
                        server_args.disaggregation_ib_device
                        or server_args.mooncake_ib_device
                    ),
                )
            # req_id -> 预分配的 embedding 缓冲区 tensor
            self.embeddings_buffer = dict()
        elif self.encoder_transfer_backend == "zmq_to_scheduler":
            # zmq_to_scheduler 模式：调度器侧负责协调 embedding 传输
            self.pp_rank = pp_rank
            self.tp_rank = tp_rank
            self.tp_size = server_args.tp_size
            self.tp_group = tp_group
            self.nnodes = server_args.nnodes
            self.hostname = get_local_ip_auto()
            # 等待 embedding 到达的请求队列
            self.waiting_list: List[WaitingImageRequest] = []
            self.scheduler = scheduler
            self.wait_timeout = envs.SGLANG_ENCODER_RECV_TIMEOUT.get()
            if hf_config is not None:
                # 确定 tensor 传输模式（单节点 cuda_ipc / 多节点 default）
                transport_mode = _determine_tensor_transport_mode(server_args)
                import_processors("sglang.srt.multimodal.processors")
                _processor = None
                try:
                    _processor = get_processor(
                        server_args.tokenizer_path,
                        tokenizer_mode=server_args.tokenizer_mode,
                        trust_remote_code=server_args.trust_remote_code,
                        revision=server_args.revision,
                        use_fast=not server_args.disable_fast_image_processor,
                        tokenizer_backend=server_args.tokenizer_backend,
                    )
                except ValueError as e:
                    error_message = str(e)
                    if "does not have a slow version" in error_message:
                        logger.info(
                            f"Processor {server_args.tokenizer_path} does not have a slow version. Automatically use fast version"
                        )
                        # 不支持慢速版本时自动降级为快速版本
                        _processor = get_processor(
                            server_args.tokenizer_path,
                            tokenizer_mode=server_args.tokenizer_mode,
                            trust_remote_code=server_args.trust_remote_code,
                            revision=server_args.revision,
                            use_fast=True,
                            tokenizer_backend=server_args.tokenizer_backend,
                        )
                    else:
                        raise e

                # Skip mm_pool if not adaptive dispatch to encoder
                # 若未启用自适应分发，跳过多模态内存池初始化
                enable_adaptive_dispatch_to_encoder = (
                    server_args.enable_adaptive_dispatch_to_encoder
                )
                self.mm_processor = get_mm_processor(
                    hf_config,
                    server_args,
                    _processor,
                    transport_mode,
                    model_config=(
                        getattr(self.scheduler, "model_config", None)
                        if self.scheduler is not None
                        else None
                    ),
                    skip_mm_pool=not enable_adaptive_dispatch_to_encoder,
                )

    @abstractmethod
    def process_waiting_requests(self, recv_reqs):
        # 处理等待 embedding 的请求队列（子类实现）
        pass

    async def recv_mm_data(
        self, request_obj, mm_processor, prompt, need_wait_for_mm_inputs=True
    ):
        # zmq_to_tokenizer 模式下异步接收 embedding 的主入口
        req_id = None
        try:
            if len(self.encode_urls) == 0 or not need_wait_for_mm_inputs:
                return None
            req_id = uuid.uuid4().hex
            # 在随机端口创建 ZMQ PULL socket 接收编码器推送的 embedding
            embedding_port, recv_socket = get_zmq_socket_on_host(
                self.context, zmq.PULL, host=self.host
            )
            mm_data = self._extract_url_data(request_obj)
            # 异步发送编码请求，不阻塞当前协程
            asyncio.create_task(
                self.encode(req_id, mm_data, embedding_port, "encode", "send")
            )
            return await asyncio.wait_for(
                self._recv_mm_data(req_id, recv_socket, mm_processor, prompt),
                timeout=20,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Embedding recv timeout for request {req_id}")
            if req_id is not None:
                self._cleanup_mooncake_buffer(req_id)
            return None

    def _cleanup_mooncake_buffer(self, req_id):
        # 清理 Mooncake RDMA 缓冲区（注销内存注册）
        if self.encoder_transfer_backend != "mooncake":
            return
        if not hasattr(self, "embeddings_buffer"):
            return
        embeddings = self.embeddings_buffer.pop(req_id, None)
        if embeddings is None:
            return
        try:
            self.embeddings_engine.deregister(embeddings.data_ptr())
        except Exception:
            logger.exception(
                "mooncake: failed to deregister buffer for req_id=%s", req_id
            )

    async def _recv_mm_data(self, req_id, recv_socket, mm_processor, prompt):
        # 异步接收所有 embedding 分片，聚合后调用 mm_processor 生成 mm_inputs
        if req_id is None:
            return None

        recv_embedding = None

        recv_embedding_data: MultiModalEmbeddingData = None

        try:
            while recv_embedding_data is None or not recv_embedding_data.ready:
                # 异步等待下一条多帧 ZMQ 消息
                parts = await recv_socket.recv_multipart(copy=False)
                if not parts:
                    continue
                recv_obj: EmbeddingData = pickle.loads(parts[0])
                if getattr(recv_obj, "error_msg", None) is not None:
                    logger.warning(
                        f"Encoder error for req_id={req_id}: {recv_obj.error_msg} "
                        f"error_code={getattr(recv_obj, 'error_code', None)}"
                    )
                    self._cleanup_mooncake_buffer(req_id)
                    return None
                logger.debug("recv_obj=%s", recv_obj)
                # Extract original req_id from part_req_id
                # 还原原始 req_id 用于聚合
                part_req_id = recv_obj.req_id
                original_req_id = extract_original_req_id(part_req_id)
                # Update recv_obj.req_id to original for aggregation
                recv_obj.req_id = original_req_id
                if self.encoder_transfer_backend == "zmq_to_tokenizer":
                    if len(parts) < 2:
                        logger.error(
                            "zmq_to_tokenizer expected 2-part message, got %d parts",
                            len(parts),
                        )
                        return None
                    buffer = (
                        parts[1].buffer if hasattr(parts[1], "buffer") else parts[1]
                    )
                    # Clone so we don't depend on ZMQ buffer after next recv.
                    # 从 ZMQ 数据帧解析 embedding 张量并 clone，避免 ZMQ 缓冲区被复用
                    recv_obj.embedding = (
                        torch.frombuffer(buffer, dtype=recv_obj.dtype)
                        .reshape(recv_obj.shape)
                        .clone()
                    )
                if recv_embedding_data is None:
                    recv_embedding_data = MultiModalEmbeddingData.from_embedding_data(
                        recv_obj
                    )
                else:
                    recv_embedding_data.add(recv_obj)

            if self.encoder_transfer_backend == "mooncake":
                # Mooncake 模式：embedding 已通过 RDMA 写入预分配缓冲区，直接从缓冲区切片
                if req_id not in self.embeddings_buffer:
                    logger.error(
                        "mooncake: embeddings_buffer missing req_id=%s", req_id
                    )
                    return None
                raw_buffer = self.embeddings_buffer.pop(req_id)
                # 注销 RDMA 内存注册
                self.embeddings_engine.deregister(raw_buffer.data_ptr())
                byte_offset = 0
                for i in range(recv_embedding_data.num_parts):
                    shape = recv_embedding_data.embedding_shape_list[i]
                    if shape is None:
                        continue
                    # 按各 part 的形状从连续缓冲区中切片出 embedding
                    part_bytes = (
                        shape[0]
                        * shape[1]
                        * torch.tensor([], dtype=self.dtype).element_size()
                    )
                    recv_embedding_data.embedding_list[i] = (
                        raw_buffer[byte_offset : byte_offset + part_bytes]
                        .view(self.dtype)
                        .reshape(shape)
                    )
                    byte_offset += part_bytes

            # 将所有 part 的 embedding 按模态拼接
            recv_embedding = recv_embedding_data.get_embedding(is_concat=True)

            # 调用多模态处理器生成最终的 mm_inputs
            mm_inputs = mm_processor.get_mm_data(
                prompt,
                recv_embedding,
                **recv_embedding_data.get_mm_extra_meta(),
            )
            return mm_inputs
        finally:
            recv_socket.close()

    def send_encode_request(self, obj):
        # 向编码服务器发送编码请求的公开接口（委托给 _send_encode_request）
        self._send_encode_request(obj)

    def _send_encode_request(self, obj):
        # 提取多模态数据并在后台线程中触发编码流程
        mm_data = self._extract_url_data(obj)
        if obj.rid is None:
            obj.rid = uuid.uuid4().hex
        if mm_data and self.encode_urls:
            logger.info(f"Processing {len(mm_data)} mm items for request {obj.rid}")
            obj.need_wait_for_mm_inputs = True

            # 按模态和编码器数量分配多媒体项
            num_items_assigned = self._assign_items_by_modality(
                mm_data, len(self.encode_urls)
            )
            obj.num_items_assigned = num_items_assigned
            # 在后台守护线程中运行 encode，不阻塞调度循环
            encode_thread = threading.Thread(
                target=self._run_encode_in_thread,
                args=(
                    obj.rid,
                    mm_data,
                    "encode",
                    num_items_assigned,
                    None,
                ),
                daemon=True,
            )
            encode_thread.start()

    # For zmq_to_scheduler
    # zmq_to_scheduler 模式的请求处理：将需要等待 embedding 的请求放入等待队列
    def _process_waiting_requests(self, recv_reqs, waiting_cls):
        new_recv_reqs = []
        for recv_req in recv_reqs:
            if (
                isinstance(recv_req, TokenizedGenerateReqInput)
                and recv_req.need_wait_for_mm_inputs is True
            ):
                # 创建等待对象并向编码服务器注册接收 URL
                waiting_req = waiting_cls(
                    rid=recv_req.rid,
                    recv_req=recv_req,
                    mm_processor=self.mm_processor,
                    encoder_urls=self.encode_urls,
                    host_name=self.hostname,
                    receive_count=self.tp_size,
                )
                waiting_req.send_encode_request()
                self.waiting_list.append(waiting_req)
            else:
                new_recv_reqs.append(recv_req)

        if len(self.waiting_list) == 0:
            return new_recv_reqs, []

        current_time = time.time()
        local_status = []
        for waiting_req in self.waiting_list:
            # 非阻塞轮询各等待请求的 embedding 接收状态
            waiting_req._try_recv_mm_data()
            if current_time - waiting_req.start_time > self.wait_timeout:
                waiting_req.status = WaitingImageRequestStatus.TIMEOUT
            local_status.append(waiting_req.status)

        # 使用 all_reduce(MIN) 跨 TP rank 同步状态，确保所有 rank 对每个请求的判断一致
        local_status = torch.tensor(local_status, device="cpu", dtype=torch.int32)

        torch.distributed.all_reduce(
            local_status,
            op=torch.distributed.ReduceOp.MIN,
            group=self.tp_group.cpu_group,
        )

        new_waiting = []
        abort_reqs = []
        for i, waiting_req in enumerate(self.waiting_list):
            status_value = local_status[i].item()
            if status_value == WaitingImageRequestStatus.SUCCESS:
                # embedding 接收成功，将请求加入正常处理队列
                new_recv_reqs.append(waiting_req.recv_req)
            elif status_value == WaitingImageRequestStatus.FAIL:
                logger.error(
                    f"Waiting request {waiting_req.rid} failed: {waiting_req.error_msg} {waiting_req.error_code = }"
                )
                abort_reqs.append(
                    (
                        self.create_req(waiting_req.recv_req),
                        waiting_req.error_msg,
                        waiting_req.error_code,
                    )
                )
            elif status_value == WaitingImageRequestStatus.TIMEOUT:
                logger.error(
                    f"Timed out waiting for image embeddings for request {waiting_req.rid}"
                )
                abort_reqs.append(
                    (
                        self.create_req(waiting_req.recv_req),
                        f"Timeout waiting for image embedding after {self.wait_timeout}s",
                        HTTPStatus.REQUEST_TIMEOUT,
                    )
                )
            else:  # status_value == WaitingImageRequestStatus.PENDING
                new_waiting.append(waiting_req)

        self.waiting_list = new_waiting
        return new_recv_reqs, abort_reqs

    def _run_encode_in_thread(
        self, req_id, mm_data, endpoint_encode, num_items_assigned, embedding_port
    ):
        # 在线程中运行异步 encode 方法（asyncio.run 创建新事件循环）
        try:
            asyncio.run(
                self.encode(
                    req_id=req_id,
                    mm_data=mm_data,
                    embedding_port=embedding_port,
                    endpoint_encode=endpoint_encode,
                    endpoint_send=None,
                    num_items_assigned=num_items_assigned,
                )
            )
        except Exception as e:
            logger.error(f"Encode failed for request {req_id}: {e}", exc_info=True)

    def create_req(self, recv_req: TokenizedGenerateReqInput):
        # 从 TokenizedGenerateReqInput 创建 Req 对象，用于中止失败请求
        req = Req(
            recv_req.rid,
            recv_req.input_text,
            recv_req.input_ids,
            recv_req.sampling_params,
            return_logprob=recv_req.return_logprob,
            top_logprobs_num=recv_req.top_logprobs_num,
            token_ids_logprob=recv_req.token_ids_logprob,
            stream=recv_req.stream,
            lora_id=recv_req.lora_id,
            input_embeds=recv_req.input_embeds,
            custom_logit_processor=recv_req.custom_logit_processor,
            require_reasoning=recv_req.require_reasoning,
            return_hidden_states=recv_req.return_hidden_states,
            return_routed_experts=recv_req.return_routed_experts,
            eos_token_ids=self.scheduler.model_config.hf_eos_token_id,
            bootstrap_host=recv_req.bootstrap_host,
            bootstrap_port=recv_req.bootstrap_port,
            bootstrap_room=recv_req.bootstrap_room,
            disagg_mode=self.scheduler.disaggregation_mode,
            routed_dp_rank=recv_req.routed_dp_rank,
            disagg_prefill_dp_rank=recv_req.disagg_prefill_dp_rank,
            vocab_size=self.scheduler.model_config.vocab_size,
            priority=recv_req.priority,
            metrics_collector=(
                self.scheduler.metrics_collector
                if self.scheduler.enable_metrics
                else None
            ),
            http_worker_ipc=recv_req.http_worker_ipc,
            dllm_config=self.scheduler.dllm_config,
        )
        req.tokenizer = self.scheduler.tokenizer
        return req

    async def allocate_embedding_buffer(self, req_id, total_bytes):
        # 为 Mooncake RDMA 传输预分配接收缓冲区，注册内存后返回物理地址
        embeddings = torch.empty(total_bytes, dtype=torch.uint8)
        self.embeddings_engine.register(
            embeddings.data_ptr(),
            embeddings.nbytes,
        )
        self.embeddings_buffer[req_id] = embeddings
        return embeddings.data_ptr()

    def _assign_items_by_modality(
        self, mm_data, encoder_num, random_shuffle=True
    ) -> Dict:
        """
        Assign multimodal items across encoders by modality with cross-modality load balancing.
        按模态将多模态数据项分配给多个编码器，支持跨模态负载均衡

        Args:
            mm_data: List of multimodal data items, each with a "modality" key
            encoder_num: Number of encoders
            random_shuffle: Whether to shuffle the encoder indices

        Returns:
            Dictionary mapping modality to list of assignment counts per encoder
            Format: {modality: [count_for_encoder_0, count_for_encoder_1, ...]}
        """
        encode_idx = list(range(encoder_num))
        if random_shuffle:
            # 随机打乱编码器顺序，实现负载均衡
            random.shuffle(encode_idx)
        # Get unique modalities with order preserved
        # 保序提取唯一模态列表
        modalities = list(dict.fromkeys(mm_item.get("modality") for mm_item in mm_data))
        # Use OrderedDict to explicitly maintain modality order
        # 使用 OrderedDict 维护模态处理顺序
        num_items_assigned = OrderedDict()
        current_offset = 0

        for modality in modalities:
            mm_data_modality = [
                mm_item for mm_item in mm_data if mm_item.get("modality") == modality
            ]
            num_items = len(mm_data_modality)
            if num_items == 0:
                continue

            base = num_items // len(encode_idx)
            remainder = num_items % len(encode_idx)
            # Rotate assignments based on current_offset to balance load across modalities
            # 使用 current_offset 轮转分配，跨模态均衡各编码器负载
            assignments = [0] * len(encode_idx)
            for i in range(len(encode_idx)):
                # keep shuffle order when assigning items to encoders
                pos_in_shuffled = (current_offset + i) % len(encode_idx)
                actual_encoder_idx = encode_idx[pos_in_shuffled]
                assignments[actual_encoder_idx] = base + (1 if i < remainder else 0)
            num_items_assigned[modality] = assignments
            current_offset = (current_offset + remainder) % len(encode_idx)

        return num_items_assigned

    def _extract_url_data(self, request_obj) -> List[Dict]:
        # 从请求对象中提取各模态的多媒体 URL，返回 [{url, modality}, ...] 列表
        def flatten_mm_items(items):
            # 递归展平嵌套列表
            if not isinstance(items, list):
                return [items]

            flat = []
            for item in items:
                if isinstance(item, (list, tuple)):
                    flat.extend(flatten_mm_items(list(item)))
                else:
                    flat.append(item)
            return flat

        def to_raw_url(mm_item):
            # 统一提取多媒体项的 URL（支持 ImageData 对象和字典格式）
            if isinstance(mm_item, ImageData):
                return mm_item.url
            if isinstance(mm_item, dict):
                # tolerate {"url": ...} shaped payloads
                return mm_item.get("url", mm_item)
            return mm_item

        mm_data = []
        for attr, modality in [
            ("image_data", Modality.IMAGE),
            ("video_data", Modality.VIDEO),
            ("audio_data", Modality.AUDIO),
        ]:
            mm_items = getattr(request_obj, attr, None)
            if mm_items:
                mm_items = flatten_mm_items(mm_items)
                for mm_item in mm_items:
                    mm_data.append(
                        {
                            "url": to_raw_url(mm_item),
                            "modality": modality,
                        }
                    )
        return mm_data


class MMReceiverHTTP(MMReceiverBase):
    # HTTP 传输模式的多模态 embedding 接收器（使用 aiohttp 与编码服务器通信）
    def __init__(
        self,
        server_args: ServerArgs,
        dtype: Optional[torch.dtype] = None,
        hf_config: Optional[PretrainedConfig] = None,
        pp_rank: Optional[int] = None,
        tp_rank: Optional[int] = None,
        tp_group: Optional[GroupCoordinator] = None,
        scheduler: Optional["Scheduler"] = None,
    ):
        super().__init__(
            server_args,
            dtype=dtype,
            hf_config=hf_config,
            pp_rank=pp_rank,
            tp_rank=tp_rank,
            tp_group=tp_group,
            scheduler=scheduler,
        )

    # For zmq_to_scheduler
    def process_waiting_requests(self, recv_reqs):
        # 调用父类实现，使用 HTTP 版等待请求类
        return self._process_waiting_requests(recv_reqs, WaitingImageRequest)

    async def encode(
        self,
        req_id,
        mm_data,
        embedding_port,
        endpoint_encode,
        endpoint_send,
        num_items_assigned=None,
    ):
        # 异步向各编码服务器发送编码请求，聚合响应后触发 Mooncake RDMA 传输（若需要）
        if len(mm_data) == 0:
            return

        # get unique modalities with order preserved
        # 提取唯一模态列表（保序），用于分片索引计算
        modalities = [mm_item.get("modality") for mm_item in mm_data]
        modalities = list(dict.fromkeys(modalities))
        encode_requests = []

        if num_items_assigned is None:
            num_items_assigned = self._assign_items_by_modality(
                mm_data, len(self.encode_urls)
            )

        # Calculate total num_parts across all modalities
        # 计算所有模态的总分片数
        total_num_parts, modality_num_parts = calculate_modality_num_parts(
            modalities, num_items_assigned
        )

        part_idx_offset = 0
        for modality in modalities:
            num_items_assigned_modality = num_items_assigned.get(modality)
            mm_data_modality = [
                mm_item for mm_item in mm_data if mm_item.get("modality") == modality
            ]

            num_parts = modality_num_parts[modality]
            cum_num_items = 0
            cum_idx = 0
            for idx, assigned_num in enumerate(num_items_assigned_modality):
                if assigned_num == 0:
                    continue
                part_idx = part_idx_offset + cum_idx
                part_req_id = create_part_req_id(req_id, part_idx)
                # 构建每个分片的编码请求（包含分配的媒体项 URL 和分片信息）
                encode_requests.append(
                    {
                        "encoder_idx": idx,
                        "mm_items": [
                            mm_item.get("url")
                            for mm_item in mm_data_modality[
                                cum_num_items : cum_num_items + assigned_num
                            ]
                        ],
                        "num_parts": total_num_parts,
                        "part_idx": part_idx,
                        "req_id": part_req_id,  # use part_req_id to avoid key collision
                        "modality": modality.name,  # convert enum to string for json serialization
                        "prefill_host": self.host,
                        "embedding_port": embedding_port,
                    }
                )
                cum_idx += 1
                cum_num_items += assigned_num
            part_idx_offset += num_parts

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=1800
            )  # Add timeout for request reliability
        ) as session:
            # Send encode requests
            # 并发向各编码服务器发送编码请求
            tasks = [
                session.post(
                    f"{self.encode_urls[encode_request['encoder_idx']]}/{endpoint_encode}",
                    json=encode_request,
                )
                for encode_request in encode_requests
            ]

            responses = await asyncio.gather(*tasks)
            for response in responses:
                if response.status != 200:
                    try:
                        err_data = await response.json()
                        msg = err_data.get("message", "Unknown encoder error")
                    except:
                        msg = await response.text()

                    logger.error(f"Encoder returned error {response.status}: {msg}")
                    return
            response_json_list_unsort = [
                await response.json() for response in responses
            ]

            # zmq backend: return is None
            # ZMQ 传输后端：编码服务器不返回 JSON 数据
            if None in response_json_list_unsort:
                return

            # mooncake backend: send bootstrap info
            # Mooncake 传输后端：收集各分片的 embedding 大小，分配 RDMA 缓冲区并触发传输

            embedding_size_list_sort = [None for _ in range(total_num_parts)]
            response_json_list_sort = [None for _ in range(total_num_parts)]
            # 按 part_idx 排序响应（各编码器可能乱序返回）
            for response_json in response_json_list_unsort:
                idx = response_json["part_idx"]
                embedding_size_list_sort[idx] = response_json["embedding_size"]
                response_json_list_sort[idx] = response_json

            total_embedding_bytes = sum(
                s for s in embedding_size_list_sort if s is not None
            )
            offset = 0
            metadata_tasks = []
            # 预分配连续 RDMA 缓冲区（所有分片共用一块大 buffer）
            buffer_address = await self.allocate_embedding_buffer(
                req_id,
                total_embedding_bytes,
            )
            for idx in range(len(tasks)):
                response_json = response_json_list_sort[idx]
                buffer_address_adjust = offset + buffer_address
                # 为每个分片提供 RDMA 会话 ID 和缓冲区偏移地址
                response_json.update(
                    {
                        "session_id": self.embeddings_engine.session_id,
                        "buffer_address": buffer_address_adjust,
                    }
                )
                metadata_tasks.append(
                    session.post(
                        f"{self.encode_urls[response_json['encoder_idx']]}/{endpoint_send}",
                        json=response_json,
                    )
                )
                offset += embedding_size_list_sort[idx]
            # 并发触发各编码服务器的 RDMA 传输
            await asyncio.gather(*metadata_tasks)


class MMReceiverGrpc(MMReceiverBase):
    # gRPC 传输模式的多模态 embedding 接收器（使用 gRPC 与编码服务器通信）
    def __init__(
        self,
        server_args: ServerArgs,
        dtype: Optional[torch.dtype] = None,
        hf_config: Optional[PretrainedConfig] = None,
        pp_rank: Optional[int] = None,
        tp_rank: Optional[int] = None,
        tp_group: Optional[GroupCoordinator] = None,
        scheduler: Optional["Scheduler"] = None,
    ):
        super().__init__(
            server_args,
            dtype=dtype,
            hf_config=hf_config,
            pp_rank=pp_rank,
            tp_rank=tp_rank,
            tp_group=tp_group,
            scheduler=scheduler,
        )

    def build_and_send_encode_request(self, image_urls, rid):
        # 构建并发送 gRPC 编码请求（仅支持图像模态）
        encode_req = GenerateReqInput(
            image_data=[ImageData(url=url) for url in image_urls],
            rid=rid,
        )
        self.send_encode_request(encode_req)
        return encode_req

    # For zmq_to_scheduler
    def process_waiting_requests(self, recv_reqs):
        # 调用父类实现，使用 gRPC 版等待请求类
        return self._process_waiting_requests(recv_reqs, WaitingImageRequestGrpc)

    async def encode(
        self,
        req_id,
        mm_data,
        embedding_port,
        endpoint_encode,
        endpoint_send,
        num_items_assigned=None,
    ):
        # gRPC 编码请求：当前仅支持图像模态，在线程池中并发调用同步 gRPC 方法
        if not mm_data:
            return

        # gRPC currently only supports image; flatten new dict formats to simple lists
        # gRPC 模式仅支持图像模态，检测到其他模态时报错
        if mm_data and isinstance(mm_data[0], dict):
            non_image = [
                item.get("modality")
                for item in mm_data
                if item.get("modality") != Modality.IMAGE
            ]
            if non_image:
                raise NotImplementedError(
                    f"gRPC encode only supports IMAGE modality, got: {non_image}"
                )
            img_data = [item.get("url") for item in mm_data]
        else:
            img_data = mm_data
        if isinstance(num_items_assigned, dict):
            num_items_assigned = list(num_items_assigned.values())[0]

        encode_requests = []
        if num_items_assigned is None:
            # 随机分配图像到编码器（round-robin 变体）
            encode_idx = list(range(len(self.encode_urls)))
            random.shuffle(encode_idx)
            num_items_assigned = [
                (idx + len(img_data)) // len(self.encode_urls) for idx in encode_idx
            ]
        num_parts = sum(1 for x in num_items_assigned if x != 0)
        cum_num_items = 0
        cum_idx = 0
        for idx, assigned_num in enumerate(num_items_assigned):
            if assigned_num == 0:
                continue
            start = cum_num_items
            end = cum_num_items + assigned_num
            encode_requests.append(
                {
                    "encoder_idx": idx,
                    "mm_items": img_data[start:end],
                    "num_parts": num_parts,
                    "part_idx": cum_idx,
                    "req_id": req_id,
                    "prefill_host": self.host,
                    "embedding_port": embedding_port,
                }
            )
            cum_idx += 1
            cum_num_items += assigned_num

        # 在线程中并发调用各编码服务器的 gRPC Encode 方法
        grpc_tasks = [
            asyncio.to_thread(
                _grpc_encode_request,
                _grpc_target(self.encode_urls[encode_request["encoder_idx"]]),
                encode_request,
            )
            for encode_request in encode_requests
        ]
        grpc_responses = await asyncio.gather(*grpc_tasks)
        response_json_unsorted = []
        for encode_request, response in zip(encode_requests, grpc_responses):
            if self.encoder_transfer_backend == "zmq_to_scheduler":
                # zmq_to_scheduler 模式下 gRPC 不返回 embedding 元数据
                response_json_unsorted.append(None)
                continue
            response_json_unsorted.append(
                {
                    "req_id": encode_request["req_id"],
                    "prefill_host": encode_request["prefill_host"],
                    "embedding_port": encode_request["embedding_port"],
                    "encoder_idx": encode_request["encoder_idx"],
                    "part_idx": encode_request["part_idx"],
                    "embedding_size": response.embedding_size,
                    "embedding_len": response.embedding_len,
                    "embedding_dim": response.embedding_dim,
                }
            )

        if None in response_json_unsorted:
            return

        # 按 part_idx 排序响应，准备 RDMA 缓冲区分配
        embedding_size_by_part = [None for _ in range(num_parts)]
        response_json_sorted = [None for _ in range(num_parts)]
        for response_json in response_json_unsorted:
            idx = response_json["part_idx"]
            embedding_size_by_part[idx] = response_json["embedding_size"]
            response_json_sorted[idx] = response_json

        total_embedding_bytes = sum(s for s in embedding_size_by_part if s is not None)
        offset = 0
        # 分配 RDMA 缓冲区并通知各编码服务器开始传输
        buffer_address = await self.allocate_embedding_buffer(
            req_id,
            total_embedding_bytes,
        )
        grpc_metadata_tasks = []
        for response_json in response_json_sorted:
            response_json.update(
                {
                    "session_id": self.embeddings_engine.session_id,
                    "buffer_address": offset + buffer_address,
                }
            )
            grpc_metadata_tasks.append(
                asyncio.to_thread(
                    _grpc_send_request,
                    _grpc_target(self.encode_urls[response_json["encoder_idx"]]),
                    response_json,
                )
            )
            offset += embedding_size_by_part[response_json["part_idx"]]

        if grpc_metadata_tasks:
            await asyncio.gather(*grpc_metadata_tasks)


def _validate_transport_mode(transport_mode: str, encoder_urls):
    # 校验传输模式与编码器 URL 前缀是否匹配（grpc 模式需要 grpc:// URL，http 模式需要 http:// URL）
    if transport_mode == "grpc":
        invalid_prefix = "http://"
        error_msg = (
            "EPD MMReceiver: grpc mode requires grpc:// encoder URLs. "
            "Set SGLANG_ENCODER_MM_RECEIVER_MODE=http for http:// URLs."
        )
    elif transport_mode == "http":
        invalid_prefix = "grpc://"
        error_msg = (
            "EPD MMReceiver: http mode requires http:// encoder URLs. "
            "Set SGLANG_ENCODER_MM_RECEIVER_MODE=grpc for grpc:// URLs."
        )
    else:
        return

    if any(url.startswith(invalid_prefix) for url in encoder_urls):
        raise ValueError(error_msg)


# 传输模式到接收器类的映射表
_MM_RECEIVER_BY_MODE = {
    "grpc": MMReceiverGrpc,
    "http": MMReceiverHTTP,
}


def create_mm_receiver(
    server_args: ServerArgs,
    dtype: Optional[torch.dtype] = None,
    hf_config: Optional[PretrainedConfig] = None,
    pp_rank: Optional[int] = None,
    tp_rank: Optional[int] = None,
    tp_group: Optional[GroupCoordinator] = None,
    scheduler: Optional["Scheduler"] = None,
    transport_mode: Optional[str] = None,
):
    # 工厂函数：根据传输模式创建对应的多模态 embedding 接收器实例
    if transport_mode is None:
        transport_mode = envs.SGLANG_ENCODER_MM_RECEIVER_MODE.get()
        logger.debug(f"MMReceiver transport_mode from env: {transport_mode}")

    _validate_transport_mode(transport_mode, server_args.encoder_urls)
    logger.info(f"EPD MMReceiver: using transport_mode={transport_mode}")

    receiver_cls = _MM_RECEIVER_BY_MODE.get(transport_mode)
    if receiver_cls is None:
        raise ValueError(f"Unsupported transport_mode: {transport_mode}")
    return receiver_cls(
        server_args,
        dtype=dtype,
        hf_config=hf_config,
        pp_rank=pp_rank,
        tp_rank=tp_rank,
        tp_group=tp_group,
        scheduler=scheduler,
    )
