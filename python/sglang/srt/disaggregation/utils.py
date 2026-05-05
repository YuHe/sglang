# PD 分离通用工具模块：包含分离模式枚举、KV 传输后端选择、元数据缓冲区、KV 页索引计算等核心工具
from __future__ import annotations

import os
import random
from collections import deque
from contextlib import nullcontext
from enum import Enum
from typing import TYPE_CHECKING, Literal, Optional, Tuple, Type, overload

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.utils import is_npu

if TYPE_CHECKING:
    from sglang.srt.disaggregation.base.conn import KVArgs
    from sglang.srt.disaggregation.common.conn import (
        CommonKVBootstrapServer,
        CommonKVManager,
        CommonKVReceiver,
        CommonKVSender,
    )
    from sglang.srt.managers.schedule_batch import Req

#########################
# Constants & Enums
#########################
# 用于测试 bootstrap 连接的虚假 IP 地址（无效地址，标记 warmup 请求）
FAKE_BOOTSTRAP_HOST = "2.2.2.2"


# DisaggregationMode：PD 分离模式枚举
# NULL: 非分离模式（unified 推理）
# PREFILL: 仅执行 prefill 阶段
# DECODE: 仅执行 decode 阶段
class DisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"

    @staticmethod
    def to_engine_type(mode: str) -> str:
        # 将分离模式字符串转换为引擎类型字符串
        if mode == DisaggregationMode.PREFILL.value:
            return "prefill"
        elif mode == DisaggregationMode.DECODE.value:
            return "decode"
        return "unified"


#########################
# Synchronization
#########################

# env var for testing failure, convert to float explicitly
# 故障注入概率，用于测试失败路径（通过环境变量 DISAGGREGATION_TEST_FAILURE_PROB 设置）
FAILURE_PROB = float(os.getenv("DISAGGREGATION_TEST_FAILURE_PROB", 0))


def poll_and_all_reduce(pollers, gloo_group: dist.ProcessGroup):
    # 轮询所有 KV 传输对象的状态，通过 gloo all_reduce MIN 操作同步各 TP rank 的最小状态
    # at a certain prob, the poll is failed to simulate failure
    # 按概率随机注入失败状态，用于测试失败处理路径
    if FAILURE_PROB > 0:
        from sglang.srt.disaggregation.base import KVPoll

        polls = [
            int(KVPoll.Failed) if random.random() < FAILURE_PROB else int(poller.poll())
            for poller in pollers
        ]
    else:
        polls = [int(poller.poll()) for poller in pollers]
    # 使用 MIN all_reduce：只有所有 TP rank 都达到某状态，该状态才生效
    tensor_to_reduce = torch.tensor(polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.MIN, group=gloo_group)
    return tensor_to_reduce.tolist()


def poll_and_all_reduce_attn_cp_tp_group(
    pollers,
    attn_cp_cpu_group: dist.ProcessGroup,
    attn_tp_cpu_group: dist.ProcessGroup,
):
    # First sync across attn-tp ranks so all TP participants for a given (dp, cp)
    # shard observe the same status transitions.
    # 先在 TP 维度同步，再在 CP 维度同步，确保 TPxCP 全组状态一致
    polls = poll_and_all_reduce(pollers, attn_tp_cpu_group)

    # Then sync across attn-cp ranks, so all TPxCP participants in one DP shard
    # converge to the same global status.
    tensor_to_reduce = torch.tensor(polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(
        tensor_to_reduce,
        op=dist.ReduceOp.MIN,
        group=attn_cp_cpu_group,
    )
    return tensor_to_reduce.tolist()


def poll_and_all_reduce_with_staging(
    decode_reqs, staging_handler, gloo_group: dist.ProcessGroup
):
    """Staging-aware polling: advance scatter, demote incomplete transfers, all_reduce."""
    # staging 感知轮询：先推进 staging scatter，再对不完整传输降级状态，最后 all_reduce
    from sglang.srt.disaggregation.base import KVPoll

    # 先推进所有需要 staging 且未完成的请求的 scatter 操作
    for decode_req in decode_reqs:
        if decode_req.kv_receiver.require_staging and not staging_handler.is_done(
            decode_req
        ):
            staging_handler.advance_scatter(decode_req)

    raw_polls = [int(dr.kv_receiver.poll()) for dr in decode_reqs]
    # 对已完成接收但 staging scatter 未完成的请求，将状态从 Success 降级为 Transferring
    for i, decode_req in enumerate(decode_reqs):
        if raw_polls[i] == int(KVPoll.Success):
            if decode_req.kv_receiver.require_staging and not staging_handler.is_done(
                decode_req
            ):
                raw_polls[i] = int(KVPoll.Transferring)
    poll_tensor = torch.tensor(raw_polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(poll_tensor, op=dist.ReduceOp.MIN, group=gloo_group)
    return poll_tensor.tolist()


#########################
# Metadata Buffers
#########################


# ReqToMetadataIdxAllocator：请求元数据缓冲区的索引分配器（类似 token pool 分配器）
class ReqToMetadataIdxAllocator:
    """A memory pool that maps a request to its first output token location."""

    def __init__(
        self,
        size: int,
    ):
        self.size = size
        # 使用 deque 存储空闲槽位索引，支持 O(1) 分配和释放
        self.free_slots = deque(list(range(size)))

    def available_size(self):
        # 返回当前可用的空闲槽位数量
        return len(self.free_slots)

    def alloc(self) -> Optional[int]:
        # 分配一个元数据缓冲区槽位，无空闲时返回 None
        if len(self.free_slots) == 0:
            return None

        return self.free_slots.popleft()

    def free(self, free_index: int):
        # 归还槽位
        self.free_slots.append(free_index)


# MetadataBuffers：存储 prefill 端第一个输出 token 及其相关元数据的张量缓冲区
# 通过 RDMA 传输到 decode 端，供 decode 端直接使用
class MetadataBuffers:
    def __init__(
        self,
        size: int,
        hidden_size: int,
        hidden_states_dtype: torch.dtype,
        max_top_logprobs_num: int = 128,
        custom_mem_pool: torch.cuda.MemPool = None,
    ):
        self.custom_mem_pool = custom_mem_pool
        bootstrap_room_dtype = torch.uint64
        device = "cpu"
        if is_npu():
            # For ascend backend, output tokens are placed in the NPU and will be transferred by D2D channel.
            # 昇腾后端：输出 token 存放在 NPU 上，通过 D2D 通道传输
            device = "npu"
            # TODO: Fix me when npu backend supports torch.uint64
            bootstrap_room_dtype = torch.int64
        elif self.custom_mem_pool:
            # TODO(shangming): Fix me (use 'cuda') when nvlink_transport of Mooncake is bug-free
            # 使用 Mooncake 自定义内存池时暂用 CPU（NVLink 传输尚有 bug）
            device = "cpu"
        elif envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get() == "INTRA_NODE_NVLINK":
            device = "cpu"
        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else nullcontext()
        ):
            # TODO: abort top_logprobs_num > 128 in PD

            # We transfer the metadata of first output token to decode
            # The minimal size for RDMA is 64Bytes, so we pad it to > 64Bytes
            # 每个槽位最小 64 字节（RDMA 传输最小单元），因此各缓冲区扩充到 ≥16 个元素
            self.output_ids = torch.zeros((size, 16), dtype=torch.int32, device=device)
            # cached_tokens[0]: total, [1]: device, [2]: host, [3]: storage
            self.cached_tokens = torch.zeros(
                (size, 16), dtype=torch.int32, device=device
            )
            # 输出 token 的对数概率值和索引（logprob 功能）
            self.output_token_logprobs_val = torch.zeros(
                (size, 16), dtype=torch.float32, device=device
            )
            self.output_token_logprobs_idx = torch.zeros(
                (size, 16), dtype=torch.int32, device=device
            )
            # top-k logprob：最多 max_top_logprobs_num 个
            self.output_top_logprobs_val = torch.zeros(
                (size, max_top_logprobs_num), dtype=torch.float32, device=device
            )
            self.output_top_logprobs_idx = torch.zeros(
                (size, max_top_logprobs_num), dtype=torch.int32, device=device
            )
            # For PD + spec decode
            # 推测解码（EAGLE）所需的 top-k 概率和索引
            self.output_topk_p = torch.zeros(
                (size, 16), dtype=torch.float32, device=device
            )
            self.output_topk_index = torch.zeros(
                (size, 16), dtype=torch.int64, device=device
            )
            # EAGLE 的 hidden states，传递给 decode 端的 draft 网络
            self.output_hidden_states = torch.zeros(
                (size, hidden_size), dtype=hidden_states_dtype, device=device
            )
            # Request validation: store bootstrap_room to detect metadata corruption
            # 存储 bootstrap_room 用于 decode 端验证元数据完整性
            self.bootstrap_room = torch.zeros(
                (size, 8), dtype=bootstrap_room_dtype, device=device
            )

    def get_buf_infos(self):
        # 返回所有缓冲区的 (指针列表, 总字节长度列表, 单项字节长度列表)，用于 RDMA 注册
        ptrs = [
            self.output_ids.data_ptr(),
            self.cached_tokens.data_ptr(),
            self.output_token_logprobs_val.data_ptr(),
            self.output_token_logprobs_idx.data_ptr(),
            self.output_top_logprobs_val.data_ptr(),
            self.output_top_logprobs_idx.data_ptr(),
            self.output_topk_p.data_ptr(),
            self.output_topk_index.data_ptr(),
            self.output_hidden_states.data_ptr(),
            self.bootstrap_room.data_ptr(),
        ]
        data_lens = [
            self.output_ids.nbytes,
            self.cached_tokens.nbytes,
            self.output_token_logprobs_val.nbytes,
            self.output_token_logprobs_idx.nbytes,
            self.output_top_logprobs_val.nbytes,
            self.output_top_logprobs_idx.nbytes,
            self.output_topk_p.nbytes,
            self.output_topk_index.nbytes,
            self.output_hidden_states.nbytes,
            self.bootstrap_room.nbytes,
        ]
        item_lens = [
            self.output_ids[0].nbytes,
            self.cached_tokens[0].nbytes,
            self.output_token_logprobs_val[0].nbytes,
            self.output_token_logprobs_idx[0].nbytes,
            self.output_top_logprobs_val[0].nbytes,
            self.output_top_logprobs_idx[0].nbytes,
            self.output_topk_p[0].nbytes,
            self.output_topk_index[0].nbytes,
            self.output_hidden_states[0].nbytes,
            self.bootstrap_room[0].nbytes,
        ]
        return ptrs, data_lens, item_lens

    def get_buf(self, idx: int):
        # 按槽位索引返回该请求所有元数据缓冲区的张量视图元组
        return (
            self.output_ids[idx],
            self.cached_tokens[idx],
            self.output_token_logprobs_val[idx],
            self.output_token_logprobs_idx[idx],
            self.output_top_logprobs_val[idx],
            self.output_top_logprobs_idx[idx],
            self.output_topk_p[idx],
            self.output_topk_index[idx],
            self.output_hidden_states[idx],
            self.bootstrap_room[idx],
        )

    def set_buf(self, req: Req):
        # 将请求的输出元数据写入对应槽位，供后续 RDMA 传输到 decode 端
        self.output_ids[req.metadata_buffer_index][0] = req.output_ids[0]
        # cached_tokens: 分别记录 total/device/host/storage 层级的缓存 token 数
        self.cached_tokens[req.metadata_buffer_index][0] = req.cached_tokens
        self.cached_tokens[req.metadata_buffer_index][1] = req.cached_tokens_device
        self.cached_tokens[req.metadata_buffer_index][2] = req.cached_tokens_host
        self.cached_tokens[req.metadata_buffer_index][3] = req.cached_tokens_storage
        if req.return_logprob:
            if req.output_token_logprobs_val:  # not none or empty list
                self.output_token_logprobs_val[req.metadata_buffer_index][0] = (
                    req.output_token_logprobs_val[0]
                )
            if req.output_token_logprobs_idx:  # not none or empty list
                self.output_token_logprobs_idx[req.metadata_buffer_index][0] = (
                    req.output_token_logprobs_idx[0]
                )

            if req.output_top_logprobs_val:  # not none or empty list
                self.output_top_logprobs_val[req.metadata_buffer_index][
                    : len(req.output_top_logprobs_val[0])
                ] = torch.tensor(
                    req.output_top_logprobs_val[0], dtype=torch.float32, device="cpu"
                )
            if req.output_top_logprobs_idx:  # not none or empty list
                self.output_top_logprobs_idx[req.metadata_buffer_index][
                    : len(req.output_top_logprobs_idx[0])
                ] = torch.tensor(
                    req.output_top_logprobs_idx[0], dtype=torch.int32, device="cpu"
                )
        # For PD + spec decode
        # 推测解码（EAGLE）所需数据：topk 概率/索引和 hidden states
        if req.hidden_states_tensor is not None:
            # speculative_eagle_topk should not be greater than 16 currently
            topk = req.output_topk_p.size(0)

            self.output_topk_p[req.metadata_buffer_index, :topk].copy_(
                req.output_topk_p
            )
            self.output_topk_index[req.metadata_buffer_index, :topk].copy_(
                req.output_topk_index
            )
            self.output_hidden_states[req.metadata_buffer_index].copy_(
                req.hidden_states_tensor
            )
        # Store bootstrap_room for validation on decode side
        # 写入 bootstrap_room 供 decode 端验证元数据来源正确性
        self.bootstrap_room[req.metadata_buffer_index, 0] = (
            req.bootstrap_room if req.bootstrap_room is not None else 0
        )


#########################
# Transfer Backend
#########################


# TransferBackend：支持的 KV 传输后端枚举
# MOONCAKE: 基于 Mooncake 传输引擎（GPU RDMA/NVLink）
# MORI: 基于 Mori 传输引擎
# NIXL: 基于 NIXL 传输引擎（通用 RDMA）
# ASCEND: 昇腾 NPU 传输引擎
# FAKE: 用于 warmup/测试的虚假后端
class TransferBackend(Enum):
    MOONCAKE = "mooncake"
    MORI = "mori"
    NIXL = "nixl"
    ASCEND = "ascend"
    FAKE = "fake"


# KVClassType：KV 传输相关类的类型枚举，用于 get_kv_class 工厂函数的参数
class KVClassType(Enum):
    KVARGS = "kvargs"             # KV 传输参数类
    MANAGER = "manager"           # KV 管理器类
    SENDER = "sender"             # KV 发送器类
    RECEIVER = "receiver"         # KV 接收器类
    BOOTSTRAP_SERVER = "bootstrap_server"  # KV 引导服务器类


@overload
def get_kv_class(
    transfer_backend: TransferBackend, class_type: Literal[KVClassType.KVARGS]
) -> Type[KVArgs]: ...
@overload
def get_kv_class(
    transfer_backend: TransferBackend, class_type: Literal[KVClassType.MANAGER]
) -> Type[CommonKVManager]: ...
@overload
def get_kv_class(
    transfer_backend: TransferBackend, class_type: Literal[KVClassType.SENDER]
) -> Type[CommonKVSender]: ...
@overload
def get_kv_class(
    transfer_backend: TransferBackend, class_type: Literal[KVClassType.RECEIVER]
) -> Type[CommonKVReceiver]: ...
@overload
def get_kv_class(
    transfer_backend: TransferBackend, class_type: Literal[KVClassType.BOOTSTRAP_SERVER]
) -> Type[CommonKVBootstrapServer]: ...


def get_kv_class(
    transfer_backend: TransferBackend, class_type: KVClassType
) -> Optional[Type]:
    # 根据传输后端和类型枚举返回对应的 KV 类，实现传输后端的插件化
    from sglang.srt.disaggregation.fake import FakeKVReceiver, FakeKVSender

    if transfer_backend == TransferBackend.MOONCAKE:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.mooncake import (
            MooncakeKVBootstrapServer,
            MooncakeKVManager,
            MooncakeKVReceiver,
            MooncakeKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: MooncakeKVManager,
            KVClassType.SENDER: MooncakeKVSender,
            KVClassType.RECEIVER: (MooncakeKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: MooncakeKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.MORI:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.mori import (
            MoriKVBootstrapServer,
            MoriKVManager,
            MoriKVReceiver,
            MoriKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: MoriKVManager,
            KVClassType.SENDER: MoriKVSender,
            KVClassType.RECEIVER: (MoriKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: MoriKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.ASCEND:
        # 昇腾后端：使用 AscendKV 系列类
        from sglang.srt.disaggregation.ascend import (
            AscendKVBootstrapServer,
            AscendKVManager,
            AscendKVReceiver,
            AscendKVSender,
        )
        from sglang.srt.disaggregation.base import KVArgs

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: AscendKVManager,
            KVClassType.SENDER: AscendKVSender,
            KVClassType.RECEIVER: (AscendKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: AscendKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.NIXL:
        # NIXL 后端：通用 RDMA 传输，适合多种网络设备
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.nixl import (
            NixlKVBootstrapServer,
            NixlKVManager,
            NixlKVReceiver,
            NixlKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: NixlKVManager,
            KVClassType.SENDER: NixlKVSender,
            KVClassType.RECEIVER: (NixlKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: NixlKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.FAKE:
        # FAKE 后端：warmup 或测试用，不进行真实传输
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.fake import (
            FakeKVManager,
            FakeKVReceiver,
            FakeKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: FakeKVManager,
            KVClassType.SENDER: FakeKVSender,
            KVClassType.RECEIVER: (FakeKVReceiver),
        }
        return class_mapping.get(class_type)

    raise ValueError(f"Unsupported transfer backend: {transfer_backend}")


#########################
# KV Pages
#########################


def kv_to_page_indices(kv_indices: np.ndarray, page_size: int):
    # 1. The page is guaranteed to be full except the last page.
    # 2. page index = kv_index // page_size
    # The return vector is kv_indices[::page_size] // page_size
    # 将 token 级别的 KV 索引转换为 page 级别的索引
    # 每 page_size 个 KV 索引对应一个 page，取步长采样后除以 page_size
    if page_size == 1:  # shortcut
        return kv_indices

    return kv_indices[::page_size] // page_size


def kv_to_page_num(num_kv_indices: int, page_size: int):
    # ceil(num_kv_indices / page_size)
    # 向上取整计算所需 page 数
    return (num_kv_indices + page_size - 1) // page_size


def page_indices_to_cp_rank_page_indices(
    page_indices: np.ndarray,
    total_pages: int,
    cp_rank: int,
    cp_size: int,
) -> np.ndarray:
    """
    Filter page_indices (which are *global* page ids in the KV pool) to those
    belonging to the given CP rank for this request.

    For a single request, its pages occupy a contiguous global range
    [first_page, first_page + total_pages). We first compute the local
    split [0, total_pages) across cp_size ranks, then shift that local
    range by first_page back into the global page id space and take
    the intersection with page_indices.

    Returns:
        Subset of page_indices that fall in this rank's global
        [start_page, end_page) slice for the given CP rank.
    """
    # 将全局 page 索引过滤为属于当前 CP rank 的子集
    # 用于 CP（Context Parallelism）场景下各 rank 只处理自己负责的 page 范围
    if cp_size <= 1:
        return page_indices

    if page_indices.size == 0:
        return np.asarray(page_indices)

    first_page = int(page_indices.min())
    base = total_pages // cp_size
    rem = total_pages % cp_size

    # 计算当前 CP rank 的本地 page 起止范围（余数分配给前 rem 个 rank）
    if rem == 0:
        local_start = cp_rank * base
        local_end = local_start + base
    else:
        local_start = cp_rank * base + min(cp_rank, rem)
        n_pages = base + (1 if cp_rank < rem else 0)
        local_end = local_start + n_pages

    # Map back to global page ids.
    # 将本地范围映射回全局 page id 空间
    start_page = first_page + local_start
    end_page = first_page + local_end

    mask = (page_indices >= start_page) & (page_indices < end_page)
    return np.asarray(page_indices)[mask]


def filter_kv_indices_for_cp_rank(
    kv_mgr: CommonKVManager, kv_indices: np.ndarray, index_slice: slice
) -> Tuple[np.ndarray, slice]:
    """Filters kv_indices and index_slice for the current CP rank."""
    # 过滤 KV 索引，仅保留当前 CP rank 负责的部分，并相应调整 index_slice
    total_pages = len(kv_indices)
    cp_rank = kv_mgr.attn_cp_rank
    cp_size = kv_mgr.attn_cp_size

    rank_page_indices = page_indices_to_cp_rank_page_indices(
        page_indices=kv_indices,
        total_pages=total_pages,
        cp_rank=cp_rank,
        cp_size=cp_size,
    )

    if rank_page_indices.size == 0:
        # 当前 rank 无负责的 page，返回空索引和空 slice
        new_kv_indices = kv_indices[:0]
        new_index_slice = slice(index_slice.start, index_slice.start)
    else:
        mask = np.isin(kv_indices, rank_page_indices)
        if not mask.any():
            new_kv_indices = kv_indices[:0]
            new_index_slice = slice(index_slice.start, index_slice.start)
        else:
            # 找到连续匹配范围的首尾位置
            first_pos = int(mask.argmax())
            last_pos = len(mask) - int(mask[::-1].argmax())

            new_kv_indices = kv_indices[first_pos:last_pos]
            new_index_slice = slice(
                index_slice.start + first_pos,
                index_slice.start + last_pos,
            )
    return new_kv_indices, new_index_slice


#########################
# Misc
#########################


def is_mla_backend(target_kv_pool) -> bool:
    # 判断目标 KV 池是否为 MLA（多头潜在注意力）模式
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    return isinstance(target_kv_pool, MLATokenToKVPool)


def prepare_abort(req: Req, error_message: str, status_code=None):
    # 为失败请求设置 FINISH_ABORT 状态，并清空 logprob 字段
    from sglang.srt.managers.schedule_batch import FINISH_ABORT

    # populate finish metadata and stream output
    req.finished_reason = FINISH_ABORT(error_message, status_code)

    if req.return_logprob:
        # 清空所有 logprob 字段，避免向客户端返回脏数据
        req.input_token_logprobs_val = []
        req.input_token_logprobs_idx = []
        req.input_top_logprobs_val = []
        req.input_top_logprobs_idx = []
        req.input_token_ids_logprobs_val = []
        req.input_token_ids_logprobs_idx = []
