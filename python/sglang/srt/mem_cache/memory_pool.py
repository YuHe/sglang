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

from __future__ import annotations

"""
Memory pool.

SGLang has two levels of memory pool.
ReqToTokenPool maps a request to its token locations.
TokenToKVPoolAllocator manages the indices to kv cache data.
KVCache actually holds the physical kv cache.
"""
# 内存池模块：SGLang 使用两级内存池管理 KV 缓存。
# ReqToTokenPool: 请求 -> token 位置映射
# TokenToKVPoolAllocator: 管理 KV 缓存的索引分配
# KVCache: 存储实际的 KV 缓存数据

import abc         # 抽象基类支持
import dataclasses  # 数据类工具
import logging     # 日志模块
from contextlib import contextmanager, nullcontext  # 上下文管理器
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np  # 数值计算
import torch        # PyTorch
import triton       # Triton JIT 内核编译器
import triton.language as tl  # Triton 内核语言

from sglang.jit_kernel.kvcache import can_use_store_cache, store_cache  # JIT 内核：高效写入 KV 缓存
from sglang.srt.configs.mamba_utils import BaseLinearStateParams  # Mamba 状态参数配置
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE  # KV 缓存的 GPU 内存类型标识
from sglang.srt.environ import envs  # 环境变量配置
from sglang.srt.layers.attention.nsa import index_buf_accessor  # NSA 注意力的索引缓冲访问器
from sglang.srt.layers.attention.nsa.quant_k_cache import (
    quantize_k_cache,
    quantize_k_cache_separate,
)  # FP8 量化 K 缓存工具函数
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype, is_fp8_fnuz  # FP8 数据类型和检测函数
from sglang.srt.layers.radix_attention import RadixAttention  # Radix Attention 层
from sglang.srt.mem_cache.utils import (
    get_mla_kv_buffer_triton,
    maybe_init_custom_mem_pool,
    set_mla_kv_buffer_triton,
    set_mla_kv_buffer_triton_fp8_quant,
    set_mla_kv_scale_buffer_triton,
)  # MLA KV 缓冲工具函数（Triton 加速）
from sglang.srt.platforms import current_platform  # 当前平台（CUDA/ROCm/CPU 等）
from sglang.srt.utils import (
    cpu_has_amx_support,
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
    next_power_of_2,
)  # 平台检测和数学工具
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter  # 内存节省适配器

# 仅在类型检查时导入，避免循环导入
if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter
    from sglang.srt.managers.schedule_batch import Req


logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024  # 1 GB 的字节数，用于内存计算
# 运行时平台标志，避免在热路径中重复调用
_is_cuda = is_cuda()
_is_npu = is_npu()
_is_cpu = is_cpu()
_cpu_has_amx_support = cpu_has_amx_support()
_is_hip = is_hip()
_is_fp8_fnuz = is_fp8_fnuz()


def get_tensor_size_bytes(t: Union[torch.Tensor, List[torch.Tensor]]):
    # 递归计算张量（或张量列表）的总字节数
    if isinstance(t, list):
        return sum(get_tensor_size_bytes(x) for x in t)
    return np.prod(t.shape) * t.dtype.itemsize


def _set_kv_buffer_impl(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    row_dim: int,  # head_num * head_dim
    store_dtype: torch.dtype,
    device_module: Any,
    alt_stream: Optional[torch.cuda.Stream] = None,
    same_kv_dim: bool = True,
) -> None:
    # 计算每行字节数，用于判断是否可以使用 JIT 优化内核
    row_bytes = row_dim * store_dtype.itemsize
    # 在 CUDA/HIP 且 K/V 维度相同时，使用高效的 store_cache JIT 内核
    if (_is_cuda or _is_hip) and same_kv_dim and can_use_store_cache(row_bytes):
        return store_cache(
            k.view(-1, row_dim),
            v.view(-1, row_dim),
            k_cache.view(-1, row_dim),
            v_cache.view(-1, row_dim),
            indices,
            row_bytes=row_bytes,
        )

    from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

    if get_is_capture_mode() and alt_stream is not None:
        # CUDA Graph 捕获模式：使用交替流并行写入 K 和 V，减少串行等待
        current_stream = device_module.current_stream()
        alt_stream.wait_stream(current_stream)
        k_cache[indices] = k
        with device_module.stream(alt_stream):
            v_cache[indices] = v
        current_stream.wait_stream(alt_stream)
    else:  # fallback to naive implementation
        # 回退到朴素实现：顺序写入 K 和 V 缓存
        k_cache[indices] = k
        v_cache[indices] = v


# ReqToTokenPool：请求到 token 位置的映射内存池，记录每个请求使用的 KV 缓存槽位
class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
    ):
        # 创建内存节省适配器（可选）
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size                     # 最大并发请求数
        self.max_context_len = max_context_len  # 每个请求的最大上下文长度
        self.device = device
        # 分配请求 -> token 映射矩阵：shape=(size, max_context_len)，int32 类型
        with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_token = torch.zeros(
                (size, max_context_len), dtype=torch.int32, device=device
            )
        self.free_slots = list(range(size))  # 空闲槽位列表

    def write(self, indices, values):
        # 将 values 写入指定请求槽位的对应 token 位置
        self.req_to_token[indices] = values

    def available_size(self):
        # 返回当前可用（空闲）的请求槽位数
        return len(self.free_slots)

    def alloc(self, reqs: list[Req]) -> Optional[List[int]]:
        # Indices of reqs that already have a req_pool_idx and will reuse
        # their existing slot (e.g. chunked prefill continuing across chunks).
        # 找出已有 req_pool_idx 的请求（复用已有槽位，如分块预填充跨块复用）
        reusing = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]
        # NOTE: this check is relaxed temporarily
        # https://github.com/sgl-project/sglang/pull/20476
        # if not any(r.is_dllm() for r in reqs):
        #     assert (
        #         sum(1 for i in reusing if reqs[i].is_chunked > 0) <= 1
        #     ), "only one chunked request may reuse req_pool_idx in a batch"
        # 确保复用槽位的请求是分块请求或已有提交的 KV 缓存
        assert all(
            reqs[i].is_chunked > 0 or reqs[i].kv_committed_len > 0 for i in reusing
        ), "reusing request must be chunked or have committed KV"

        # 计算需要新分配的槽位数
        need_size = len(reqs) - len(reusing)
        if need_size > len(self.free_slots):
            return None  # 空闲槽位不足，返回 None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        offset = 0
        # 为没有 req_pool_idx 的请求分配新槽位
        for r in reqs:
            if r.req_pool_idx is None:
                r.req_pool_idx = select_index[offset]
                offset += 1
        return [r.req_pool_idx for r in reqs]

    def free(self, req: Req):
        # 释放请求占用的槽位，将其归还到空闲列表
        assert req.req_pool_idx is not None, "request must have req_pool_idx"
        self.free_slots.append(req.req_pool_idx)
        req.req_pool_idx = None

    def clear(self):
        # 重置所有槽位为空闲状态
        self.free_slots = list(range(self.size))


# MambaPool：Mamba 模型状态缓存池，管理卷积状态和时序状态的 GPU 内存
class MambaPool:
    @dataclass(frozen=True, kw_only=True)
    class State:
        # Mamba 状态数据类：卷积状态（列表）和时序状态（单张量）
        conv: List[torch.Tensor]
        temporal: torch.Tensor

        def at_layer_idx(self, layer: int):
            # 返回指定层的状态切片
            kwargs = {}
            # Use fields instead of vars to avoid torch.compile graph break
            # 使用 fields 而非 vars 避免 torch.compile 图断点
            for f in fields(self):
                name = f.name
                v = getattr(self, name)
                if name in ("conv", "intermediate_conv_window"):
                    kwargs[name] = [conv[layer] for conv in v]
                else:
                    kwargs[name] = v[layer]

            return type(self)(**kwargs)

        def mem_usage_bytes(self):
            # 计算该状态所占的总内存字节数
            return sum(
                get_tensor_size_bytes(getattr(self, f.name))
                for f in dataclasses.fields(self)
            )

    @dataclass(frozen=True, kw_only=True)
    class SpeculativeState(State):
        # 推测解码专用状态：额外包含中间 SSM 状态和中间卷积窗口
        intermediate_ssm: torch.Tensor
        intermediate_conv_window: List[torch.Tensor]

    def __init__(
        self,
        *,
        size: int,
        spec_state_size: int,
        cache_params: BaseLinearStateParams,
        mamba_layer_ids: List[int],
        device: str,
        enable_memory_saver: bool = False,
        speculative_num_draft_tokens: Optional[int] = None,
    ):
        # 提取卷积状态形状、时序状态形状和数据类型
        conv_state_shape = cache_params.shape.conv
        temporal_state_shape = cache_params.shape.temporal
        conv_dtype = cache_params.dtype.conv
        ssm_dtype = cache_params.dtype.temporal
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        num_mamba_layers = len(mamba_layer_ids)

        self.size = size
        self.device = device

        # for disagg with nvlink
        # 为 NVLink 分布式推理初始化自定义内存池
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE), (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.enable_custom_mem_pool
            else nullcontext()
        ):
            # 为每种卷积状态形状分配 GPU 张量：shape=(num_layers, size+1, *conv_shape)
            # 多出的第 0 槽位用于接收 padded token 的虚拟输出
            conv_state = [
                torch.zeros(
                    size=(num_mamba_layers, size + 1) + conv_shape,
                    dtype=conv_dtype,
                    device=device,
                )
                for conv_shape in conv_state_shape
            ]

            if _is_npu:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    _init_npu_conv_state,
                )
                # NPU 特殊初始化：使用 NPU 专用内核初始化卷积状态
                conv_state = _init_npu_conv_state(
                    conv_state[0], conv_state_shape, speculative_num_draft_tokens
                )

            if _is_cpu and _cpu_has_amx_support:
                from sglang.srt.layers.amx_utils import _init_amx_conv_state

                # CPU uses a different layout of conv_state for kernel optimization
                # AMX 支持时使用不同的内存布局以优化 CPU 内核
                conv_state = _init_amx_conv_state(conv_state)

            # 分配时序状态张量：shape=(num_layers, size+1, *temporal_shape)
            temporal_state = torch.zeros(
                size=(num_mamba_layers, size + 1) + temporal_state_shape,
                dtype=ssm_dtype,
                device=device,
            )
            if speculative_num_draft_tokens is not None:
                # Cache intermediate SSM states per draft token during target verify
                # Shape: [num_layers, size + 1, speculative_num_draft_tokens, HV, K, V]
                # 推测解码验证阶段：缓存每个草稿 token 的中间 SSM 状态
                intermediate_ssm_state_cache = torch.zeros(
                    size=(
                        num_mamba_layers,
                        spec_state_size + 1,
                        speculative_num_draft_tokens,
                        temporal_state_shape[0],
                        temporal_state_shape[1],
                        temporal_state_shape[2],
                    ),
                    dtype=ssm_dtype,
                    device="cuda",
                )
                # Cache intermediate conv windows (last K-1 inputs) per draft token during target verify
                # Shape: [num_layers, size + 1, speculative_num_draft_tokens, dim, K-1]
                # 缓存每个草稿 token 的中间卷积窗口（最后 K-1 个输入）
                intermediate_conv_window_cache = [
                    torch.zeros(
                        size=(
                            num_mamba_layers,
                            spec_state_size + 1,
                            speculative_num_draft_tokens,
                            conv_shape[0],
                            conv_shape[1],
                        ),
                        dtype=conv_dtype,
                        device="cuda",
                    )
                    for conv_shape in conv_state_shape
                ]
                # 使用推测解码状态类型存储缓存
                self.mamba_cache = self.SpeculativeState(
                    conv=conv_state,
                    temporal=temporal_state,
                    intermediate_ssm=intermediate_ssm_state_cache,
                    intermediate_conv_window=intermediate_conv_window_cache,
                )
                logger.info(
                    f"Mamba Cache is allocated. "
                    f"max_mamba_cache_size: {size}, "
                    f"conv_state size: {get_tensor_size_bytes(conv_state) / GB:.2f}GB, "
                    f"ssm_state size: {get_tensor_size_bytes(temporal_state) / GB:.2f}GB "
                    f"intermediate_ssm_state_cache size: {get_tensor_size_bytes(intermediate_ssm_state_cache) / GB:.2f}GB "
                    f"intermediate_conv_window_cache size: {get_tensor_size_bytes(intermediate_conv_window_cache) / GB:.2f}GB "
                )
            else:
                # 非推测解码：使用标准状态类型
                self.mamba_cache = self.State(conv=conv_state, temporal=temporal_state)
                logger.info(
                    f"Mamba Cache is allocated. "
                    f"max_mamba_cache_size: {size}, "
                    f"conv_state size: {get_tensor_size_bytes(conv_state) / GB:.2f}GB, "
                    f"ssm_state size: {get_tensor_size_bytes(temporal_state) / GB:.2f}GB "
                )
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            # 槽位 0 用于 padded token 的虚拟输出，实际可用槽位从 1 开始
            self.free_slots = torch.arange(
                1, self.size + 1, dtype=torch.int64, device=self.device
            )
            self.mem_usage = self.mamba_cache.mem_usage_bytes() / GB  # 总内存占用（GB）
            self.num_mamba_layers = num_mamba_layers

    def get_speculative_mamba2_params_all_layers(self) -> SpeculativeState:
        # 获取所有层的推测解码 Mamba 状态（需确保当前使用 SpeculativeState）
        assert isinstance(self.mamba_cache, self.SpeculativeState)
        return self.mamba_cache

    def mamba2_layer_cache(self, layer_id: int):
        # 返回指定层的 Mamba 状态切片
        return self.mamba_cache.at_layer_idx(layer_id)

    def available_size(self):
        # 返回当前可用的 Mamba 缓存槽位数
        return len(self.free_slots)

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        # 分配指定数量的 Mamba 缓存槽位，返回槽位索引张量
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        # clear at alloc time — expand a scalar GPU zero to the right shape, no CPU-GPU sync
        # 分配时清零：将 GPU 标量 0 扩展到正确形状，避免 CPU-GPU 同步
        for i in range(len(self.mamba_cache.conv)):
            t = self.mamba_cache.conv[i]
            z = torch.zeros(1, dtype=t.dtype, device=t.device).expand(
                t.shape[0], need_size, *t.shape[2:]
            )
            t[:, select_index] = z
        t = self.mamba_cache.temporal
        z = torch.zeros(1, dtype=t.dtype, device=t.device).expand(
            t.shape[0], need_size, *t.shape[2:]
        )
        t[:, select_index] = z

        return select_index

    def free(self, free_index: torch.Tensor):
        # 释放指定槽位，将其追加回空闲列表
        if free_index.numel() == 0:
            return
        self.free_slots = torch.cat((self.free_slots, free_index))

    def clear(self):
        # 重置所有槽位为空闲（从 1 开始，0 号槽位保留为 padded token 虚拟输出）
        self.free_slots = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )

    def copy_from(self, src_index: torch.Tensor, dst_index: torch.Tensor):
        # 将 src_index 对应的 Mamba 状态复制到 dst_index（用于请求分叉/投机解码）
        for i in range(len(self.mamba_cache.conv)):
            self.mamba_cache.conv[i][:, dst_index] = self.mamba_cache.conv[i][
                :, src_index
            ]
        self.mamba_cache.temporal[:, dst_index] = self.mamba_cache.temporal[
            :, src_index
        ]
        return

    def fork_from(self, src_index: torch.Tensor) -> Optional[torch.Tensor]:
        # 分配一个新槽位并复制 src_index 的状态（用于束搜索/请求分叉）
        dst_index = self.alloc(1)
        if dst_index is None:
            return None
        self.copy_from(src_index, dst_index)
        return dst_index

    def get_cpu_copy(self, indices):
        # 同步 GPU 并将指定索引的 Mamba 状态异步复制到 CPU
        torch.cuda.synchronize()
        conv_cpu = [
            conv[:, indices].to("cpu", non_blocking=True)
            for conv in self.mamba_cache.conv
        ]
        temporal_cpu = self.mamba_cache.temporal[:, indices].to(
            "cpu", non_blocking=True
        )
        torch.cuda.synchronize()
        return conv_cpu, temporal_cpu

    def load_cpu_copy(self, mamba_cache_cpu, indices):
        # 将 CPU 上的 Mamba 状态异步加载回 GPU 的指定槽位
        conv_cpu, temporal_cpu = mamba_cache_cpu
        torch.cuda.synchronize()
        for i, conv in enumerate(self.mamba_cache.conv):
            conv[:, indices] = conv_cpu[i].to(conv.device, non_blocking=True)
        self.mamba_cache.temporal[:, indices] = temporal_cpu.to(
            self.mamba_cache.temporal.device, non_blocking=True
        )
        torch.cuda.synchronize()

    def get_contiguous_buf_infos(self):
        """
        Get buffer info for RDMA registration.
        Only returns conv and temporal state buffers, excluding intermediate buffers
        used for speculative decoding (intermediate_ssm, intermediate_conv_window).
        """
        # 获取 RDMA 注册所需的连续缓冲区信息（排除推测解码专用的中间缓冲区）
        state_tensors = []
        for field in vars(self.mamba_cache):
            # Skip intermediate buffers used only for speculative decoding
            # These buffers have different size (spec_state_size + 1) and should not be transferred
            # 跳过推测解码专用的中间缓冲区（尺寸不同，不参与传输）
            if field in ("intermediate_ssm", "intermediate_conv_window"):
                continue
            value = getattr(self.mamba_cache, field)
            if isinstance(value, list):
                state_tensors.extend(value)
            else:
                state_tensors.append(value)
        data_ptrs, data_lens, item_lens = [], [], []

        # 为每个状态张量收集各层的数据指针、字节长度和单元素字节长度
        for _, state_tensor in enumerate(state_tensors):
            data_ptrs += [
                state_tensor[i].data_ptr() for i in range(self.num_mamba_layers)
            ]
            data_lens += [state_tensor[i].nbytes for i in range(self.num_mamba_layers)]
            item_lens += [
                state_tensor[i][0].nbytes for i in range(self.num_mamba_layers)
            ]
        return data_ptrs, data_lens, item_lens

    def get_state_dim_per_tensor(self):
        """Get the sliceable dimension size for each state tensor.

        For mamba state, the layout is:
        - conv_state: [num_layers, size+1, conv_dim/tp, conv_kernel-1]
        - temporal_state: [num_layers, size+1, num_heads/tp, head_dim, state_size]

        The 3rd dimension (index 2) is the one that gets sliced by TP.
        Returns the size of this dimension for each tensor (repeated for each layer).
        """
        # 获取每个状态张量中被 TP（张量并行）切片的维度大小（第 3 维，索引 2）
        state_tensors = []
        for field in vars(self.mamba_cache):
            value = getattr(self.mamba_cache, field)
            if isinstance(value, list):
                state_tensors.extend(value)
            else:
                state_tensors.append(value)

        dim_per_tensor = []
        for state_tensor in state_tensors:
            # state_tensor shape: [num_layers, size+1, sliceable_dim, ...]
            # The sliceable dimension is at index 2 (after num_layers and size)
            # 可切片维度位于索引 2（num_layers 和 size 之后）
            sliceable_dim = state_tensor.shape[2]
            # Repeat for each layer since we have per-layer data_ptrs
            # 每层都需要记录，因为我们有逐层的数据指针
            dim_per_tensor += [sliceable_dim] * self.num_mamba_layers
        return dim_per_tensor


# HybridReqToTokenPool：混合模型（Mamba + 全注意力）请求 token 池，继承 ReqToTokenPool
class HybridReqToTokenPool(ReqToTokenPool):
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        *,
        size: int,
        mamba_size: int,
        mamba_spec_state_size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        cache_params: BaseLinearStateParams,
        mamba_layer_ids: List[int],
        enable_mamba_extra_buffer: bool,
        speculative_num_draft_tokens: int = None,
        enable_overlap_schedule: bool = True,
        start_layer: Optional[int] = None,
    ):
        super().__init__(
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
        )

        # 乒乓缓冲区大小：流水线调度时使用 2，否则使用 1
        self.mamba_ping_pong_track_buffer_size = 2 if enable_overlap_schedule else 1
        self.enable_mamba_extra_buffer = enable_mamba_extra_buffer
        self.enable_memory_saver = enable_memory_saver
        self.start_layer = start_layer if start_layer is not None else 0
        self.layer_transfer_counter = None
        # 初始化 Mamba 状态缓存池
        self._init_mamba_pool(
            size=mamba_size,
            mamba_spec_state_size=mamba_spec_state_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
            enable_mamba_extra_buffer=enable_mamba_extra_buffer,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )

    def _init_mamba_pool(
        self,
        size: int,
        mamba_spec_state_size: int,
        cache_params: BaseLinearStateParams,
        mamba_layer_ids: List[int],
        device: str,
        enable_mamba_extra_buffer: bool,
        speculative_num_draft_tokens: int = None,
    ):
        # 创建 MambaPool 实例
        self.mamba_pool = MambaPool(
            size=size,
            spec_state_size=mamba_spec_state_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
            enable_memory_saver=self.enable_memory_saver,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )
        # 建立全局 layer_id -> Mamba 层本地索引的映射
        self.mamba_map = {layer_id: i for i, layer_id in enumerate(mamba_layer_ids)}

        self.device = device
        # 请求索引 -> Mamba 索引的映射张量
        self.req_index_to_mamba_index_mapping: torch.Tensor = torch.zeros(
            size, dtype=torch.int32, device=self.device
        )
        if enable_mamba_extra_buffer:
            # 额外的乒乓缓冲区索引映射，用于流水线调度时追踪当前写入的缓冲区
            self.req_index_to_mamba_ping_pong_track_buffer_mapping: torch.Tensor = (
                torch.zeros(
                    (size, self.mamba_ping_pong_track_buffer_size),
                    dtype=torch.int32,
                    device=self.device,
                )
            )

    def register_layer_transfer_counter(
        self, layer_transfer_counter: "LayerDoneCounter"
    ):
        # 注册层传输计数器，用于分层传输时等待指定层完成
        self.layer_transfer_counter = layer_transfer_counter

    # For chunk prefill req, we do not need to allocate mamba cache,
    # We could use allocated mamba cache instead.
    # 对于分块预填充请求，复用已分配的 Mamba 缓存而非重新分配
    def alloc(self, reqs: List["Req"]) -> Optional[List[int]]:
        select_index = super().alloc(reqs)
        if select_index is None:
            return None

        mamba_indices: list[torch.Tensor] = []
        mamba_ping_pong_track_buffers: list[torch.Tensor] = []
        for req in reqs:
            mid = None
            if req.mamba_pool_idx is not None:  # for radix cache
                # 已有 Mamba 缓存索引（Radix Cache 已预分配）则复用
                mid = req.mamba_pool_idx
            else:
                # 新请求：从 MambaPool 中分配 1 个槽位
                mid = self.mamba_pool.alloc(1)
                assert (
                    mid is not None
                ), f"Not enough space for mamba cache, try to increase --mamba-full-memory-ratio or --max-mamba-cache-size. {mid=}, {self.mamba_pool.size=}, {self.mamba_pool.available_size()=}, {len(reqs)=}"
                mid = mid[0]
                req.mamba_pool_idx = mid
            mamba_indices.append(mid)
            if self.enable_mamba_extra_buffer:
                if req.mamba_ping_pong_track_buffer is None:
                    # 分配乒乓缓冲区（用于流水线调度的 in-flight 状态追踪）
                    req.mamba_ping_pong_track_buffer = self.mamba_pool.alloc(
                        self.mamba_ping_pong_track_buffer_size
                    )
                    assert (
                        req.mamba_ping_pong_track_buffer is not None
                    ), "Not enough space for mamba ping pong idx, try to increase --mamba-full-memory-ratio."
                    req.mamba_next_track_idx = 0
                mamba_ping_pong_track_buffers.append(req.mamba_ping_pong_track_buffer)
        assert len(select_index) == len(
            mamba_indices
        ), f"Not enough space for mamba cache, try to increase --mamba-full-memory-ratio or --max-mamba-cache-size."
        if self.enable_mamba_extra_buffer:
            assert len(select_index) == len(
                mamba_ping_pong_track_buffers
            ), f"Not enough space for mamba ping pong idx, try to increase --mamba-full-memory-ratio."
        # 更新请求索引到 Mamba 索引的映射
        mamba_index_tensor = torch.stack(mamba_indices).to(dtype=torch.int32)
        self.req_index_to_mamba_index_mapping[select_index] = mamba_index_tensor
        if self.enable_mamba_extra_buffer:
            ping_pong_tensor = torch.stack(mamba_ping_pong_track_buffers).to(
                dtype=torch.int32
            )
            self.req_index_to_mamba_ping_pong_track_buffer_mapping[select_index] = (
                ping_pong_tensor
            )
        return select_index

    def get_mamba_indices(self, req_indices: torch.Tensor) -> torch.Tensor:
        # 根据请求索引查询对应的 Mamba 缓存索引
        return self.req_index_to_mamba_index_mapping[req_indices]

    def mamba2_layer_cache(self, layer_id: int):
        # 等待指定层传输完成后返回对应层的 Mamba 缓存
        assert layer_id in self.mamba_map
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.mamba_pool.mamba2_layer_cache(self.mamba_map[layer_id])

    def get_speculative_mamba2_params_all_layers(self) -> MambaPool.SpeculativeState:
        # 获取所有层的推测解码 Mamba 参数
        return self.mamba_pool.get_speculative_mamba2_params_all_layers()

    def get_mamba_ping_pong_other_idx(self, mamba_next_track_idx: int) -> int:
        # 返回乒乓缓冲区的另一个索引（双缓冲时返回 1-idx，单缓冲时返回 0）
        if self.mamba_ping_pong_track_buffer_size == 2:
            return 1 - mamba_next_track_idx
        else:
            return mamba_next_track_idx

    def free_mamba_cache(
        self, req: "Req", mamba_ping_pong_track_buffer_to_keep: Optional[int] = None
    ):
        # 释放请求的 Mamba 缓存，可选保留乒乓缓冲区中的一个槽
        mamba_index = req.mamba_pool_idx
        assert mamba_index is not None, "double free? mamba_index is None"
        self.mamba_pool.free(mamba_index.unsqueeze(0))
        req.mamba_pool_idx = None

        if self.enable_mamba_extra_buffer:
            mamba_ping_pong_track_buffer_to_free = (
                self.req_index_to_mamba_ping_pong_track_buffer_mapping[req.req_pool_idx]
            )
            if mamba_ping_pong_track_buffer_to_keep is not None:
                assert mamba_ping_pong_track_buffer_to_keep in [
                    0,
                    1,
                ], f"mamba_ping_pong_track_buffer_to_keep must be 0 or 1, {mamba_ping_pong_track_buffer_to_keep=}"
                # Avoid Python-list advanced indexing on a device tensor.
                # The ping-pong buffer size is either 2 (normal) or 1 (spec decode).
                # 避免在设备张量上使用 Python 列表高级索引；乒乓缓冲区大小为 2 或 1
                if self.mamba_ping_pong_track_buffer_size == 2:
                    # 双缓冲时：释放另一个槽，保留指定槽
                    idx_to_free = 1 - mamba_ping_pong_track_buffer_to_keep
                    mamba_ping_pong_track_buffer_to_free = (
                        mamba_ping_pong_track_buffer_to_free[
                            idx_to_free : idx_to_free + 1
                        ]
                    )
                else:
                    assert self.mamba_ping_pong_track_buffer_size == 1, (
                        f"Unexpected mamba_ping_pong_track_buffer_size="
                        f"{self.mamba_ping_pong_track_buffer_size}"
                    )
                    assert mamba_ping_pong_track_buffer_to_keep == 0, (
                        "mamba_ping_pong_track_buffer_to_keep must be 0 when "
                        "mamba_ping_pong_track_buffer_size is 1"
                    )
                    # Keep the only slot, so free nothing.
                    # 单缓冲时保留唯一槽，不释放任何内容
                    mamba_ping_pong_track_buffer_to_free = (
                        mamba_ping_pong_track_buffer_to_free[0:0]
                    )
            self.mamba_pool.free(mamba_ping_pong_track_buffer_to_free)

    def clear(self):
        # 清空所有缓存（重置 token 池、Mamba 池和映射张量）
        logger.info("Reset HybridReqToTokenPool")
        super().clear()
        self.mamba_pool.clear()
        self.req_index_to_mamba_index_mapping.zero_()
        if self.enable_mamba_extra_buffer:
            self.req_index_to_mamba_ping_pong_track_buffer_mapping.zero_()


# KVCache：KV 缓存抽象基类，定义所有 KV 缓存实现必须支持的接口
class KVCache(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        self.size = size           # 缓存总槽位数（不含 padded 槽）
        self.page_size = page_size  # 每页包含的 token 数
        self.dtype = dtype          # KV 缓存的计算数据类型
        self.device = device
        # FP8 类型需要以 uint8 存储（因为 Tensor.index_put 不支持 float8）
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype  # 其他类型直接使用原始数据类型
        self.layer_num = layer_num
        self.start_layer = start_layer or 0   # 该 KV 缓存负责的起始层（TP 分片时使用）
        self.end_layer = end_layer or layer_num - 1  # 该 KV 缓存负责的终止层
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        self.mem_usage = 0  # 实际内存使用量（GB），子类初始化后更新

        # used for chunked cpu-offloading
        # 分块 CPU 卸载时每批次处理的 token 数（避免单次传输过大）
        self.cpu_offloading_chunk_size = 8192

        # default state for optional layer-wise transfer control
        # 层级传输控制器（用于分布式场景中等待特定层完成传输）
        self.layer_transfer_counter = None

        # for disagg with nvlink
        # 为 NVLink 分布式推理初始化自定义 CUDA 内存池
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

    def _finalize_allocation_log(self, num_tokens: int):
        """Common logging and mem_usage computation for KV cache allocation.
        Supports both tuple (K, V) size returns and single KV size returns.
        """
        # 统一记录 KV 缓存分配日志并计算内存使用量
        kv_size_bytes = self.get_kv_size_bytes()
        if isinstance(kv_size_bytes, tuple):
            k_size, v_size = kv_size_bytes
            k_size_GB = k_size / GB
            v_size_GB = v_size / GB
            logger.info(
                f"KV Cache is allocated. #tokens: {num_tokens}, K size: {k_size_GB:.2f} GB, V size: {v_size_GB:.2f} GB"
            )
            self.mem_usage = k_size_GB + v_size_GB
        else:
            kv_size_GB = kv_size_bytes / GB
            logger.info(
                f"KV Cache is allocated. #tokens: {num_tokens}, KV size: {kv_size_GB:.2f} GB"
            )
            self.mem_usage = kv_size_GB

    @abc.abstractmethod
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        # 返回指定层的 K 缓存张量（含同步等待）
        raise NotImplementedError()

    @abc.abstractmethod
    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        # 返回指定层的 V 缓存张量（含同步等待）
        raise NotImplementedError()

    @abc.abstractmethod
    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 返回指定层的 (K, V) 缓存张量元组
        raise NotImplementedError()

    @abc.abstractmethod
    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        # 将 K/V 数据写入指定层的缓存槽位
        raise NotImplementedError()

    def register_layer_transfer_counter(self, layer_transfer_counter: LayerDoneCounter):
        # 注册层传输计数器，用于分层传输时的同步等待
        self.layer_transfer_counter = layer_transfer_counter

    def get_cpu_copy(self, indices, mamba_indices=None):
        # 将指定槽位的 KV 缓存复制到 CPU（基类不实现，由子类覆盖）
        raise NotImplementedError()

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        # 将 CPU 端的 KV 缓存加载回 GPU（基类不实现，由子类覆盖）
        raise NotImplementedError()

    def maybe_get_custom_mem_pool(self):
        # 返回自定义 CUDA 内存池（用于 NVLink 分布式推理）
        return self.custom_mem_pool


# MHATokenToKVPool：多头注意力（MHA）的 KV 缓存实现，分别存储 K 和 V 张量
class MHATokenToKVPool(KVCache):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        v_head_dim: Optional[int] = None,        # V 头维度（不同于 K 时指定）
        swa_head_num: Optional[int] = None,       # 滑动窗口注意力的头数
        swa_head_dim: Optional[int] = None,       # 滑动窗口注意力的头维度
        swa_v_head_dim: Optional[int] = None,     # 滑动窗口注意力的 V 头维度
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,           # 是否启用交替流（减少 K/V 写入串行等待）
        enable_kv_cache_copy: bool = False,       # 是否启用 Triton 内核加速的 KV 缓存复制
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        # 优先使用滑动窗口注意力配置（若未指定则回退到全注意力配置）
        self.head_num = swa_head_num if swa_head_num is not None else head_num
        self.head_dim = swa_head_dim if swa_head_dim is not None else head_dim
        self.v_head_dim = (
            swa_v_head_dim
            if swa_v_head_dim is not None
            else v_head_dim if v_head_dim is not None else head_dim
        )

        self._create_buffers()  # 分配 GPU K/V 缓冲区

        self.device_module = torch.get_device_module(self.device)

        # 仅在 CUDA 类设备上创建交替流
        _use_alt_stream = _is_cuda or current_platform.is_cuda_alike()
        self.alt_stream = (
            self.device_module.Stream()
            if _use_alt_stream and enable_alt_stream
            else None
        )

        # 初始化 KV 复制 Triton 内核（可选，用于 DLLM 等场景）
        if enable_kv_cache_copy:
            self._init_kv_copy_and_warmup()
        else:
            self._kv_copy_config = None

        self._finalize_allocation_log(size)

        # for store_cache JIT kernel
        # store_cache JIT 内核所需的行维度（head_num * head_dim）
        self.row_dim = self.head_num * self.head_dim
        self.same_kv_dim = self.head_dim == self.v_head_dim  # K/V 头维度是否相同

    def _init_kv_copy_and_warmup(self):
        # Heuristics for KV copy tiling
        # KV 复制分块启发式参数：根据 stride 大小选择合适的 tile 大小和 warp 数
        _KV_COPY_STRIDE_THRESHOLD_LARGE = 8192
        _KV_COPY_STRIDE_THRESHOLD_MEDIUM = 4096
        _KV_COPY_TILE_SIZE_LARGE = 512
        _KV_COPY_TILE_SIZE_MEDIUM = 256
        _KV_COPY_TILE_SIZE_SMALL = 128
        _KV_COPY_NUM_WARPS_LARGE_TILE = 8
        _KV_COPY_NUM_WARPS_SMALL_TILE = 4

        # 每行字节数决定选用哪个 tile 大小
        stride_bytes = int(self.data_strides[0].item())
        if stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_LARGE:
            bytes_per_tile = _KV_COPY_TILE_SIZE_LARGE
        elif stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_MEDIUM:
            bytes_per_tile = _KV_COPY_TILE_SIZE_MEDIUM
        else:
            bytes_per_tile = _KV_COPY_TILE_SIZE_SMALL

        # Calculate num_locs_upper to avoid large Triton specialization (e.g. 8192)
        # 计算 num_locs_upper 以避免触发 Triton 过大的特化（如 8192 个 token）
        chunk_upper = 128 if bytes_per_tile >= _KV_COPY_TILE_SIZE_LARGE else 256

        # 保存 KV 复制配置供后续 move_kv_cache 使用
        self._kv_copy_config = {
            "bytes_per_tile": bytes_per_tile,
            "byte_tiles": (stride_bytes + bytes_per_tile - 1) // bytes_per_tile,
            "num_warps": (
                _KV_COPY_NUM_WARPS_SMALL_TILE
                if bytes_per_tile <= _KV_COPY_TILE_SIZE_MEDIUM
                else _KV_COPY_NUM_WARPS_LARGE_TILE
            ),
            "num_locs_upper": chunk_upper,
        }

        # 预热（warmup）：使用虚拟数据触发 Triton 内核 JIT 编译
        dummy_loc = torch.zeros(chunk_upper, dtype=torch.int64, device=self.device)
        grid = (self.data_ptrs.numel(), self._kv_copy_config["byte_tiles"])

        copy_all_layer_kv_cache_tiled[grid](
            self.data_ptrs,
            self.data_strides,
            dummy_loc,
            dummy_loc,
            1,
            chunk_upper,
            BYTES_PER_TILE=self._kv_copy_config["bytes_per_tile"],
            num_warps=self._kv_copy_config["num_warps"],
            num_stages=2,
        )

    def _create_buffers(self):
        # 在内存节省区域和可选自定义内存池中分配 K/V 缓冲区
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # [size, head_num, head_dim] for each layer
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                # 为每层分配 K 缓冲：shape=(size+page_size, head_num, head_dim)
                # +page_size 是因为槽位 0 保留为 padded token 虚拟输出
                self.k_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, self.head_num, self.head_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                # 为每层分配 V 缓冲：形状与 K 类似，但使用 v_head_dim
                self.v_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, self.head_num, self.v_head_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

        # 预先收集所有层的 K/V 数据指针，供 Triton 内核批量处理
        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        # 合并 K 和 V 数据指针（K 在前，V 在后）
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        # 每行字节数（head_num * head_dim * itemsize），供 Triton 内核计算偏移量
        self.data_strides = torch.tensor(
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in self.k_buffer + self.v_buffer
            ],
            device=self.device,
        )

    def _clear_buffers(self):
        # 释放 K/V 缓冲区引用（用于内存节省模式下的显式释放）
        del self.k_buffer
        del self.v_buffer

    def get_kv_size_bytes(self):
        # 计算 K 和 V 缓冲区的总字节数
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        k_size_bytes = 0
        for k_cache in self.k_buffer:
            k_size_bytes += get_tensor_size_bytes(k_cache)
        v_size_bytes = 0
        for v_cache in self.v_buffer:
            v_size_bytes += get_tensor_size_bytes(v_cache)
        return k_size_bytes, v_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        # 获取所有层 K/V 缓冲区的数据指针、字节长度和每项字节长度（用于 RDMA 注册）
        # layer_num x [seq_len, head_num, head_dim]
        # layer_num x [page_num, page_size, head_num, head_dim]
        kv_data_ptrs = [
            self._get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_data_lens = [
            self._get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_item_lens = [
            self._get_key_buffer(i)[0].nbytes * self.page_size
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i)[0].nbytes * self.page_size
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_cpu_copy(self, indices, mamba_indices=None):
        # 同步 GPU 并将指定槽位的 KV 缓存分块异步复制到 CPU
        torch.cuda.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu = self.k_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                v_cpu = self.v_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append([k_cpu, v_cpu])
        torch.cuda.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        # 将 CPU 端的 KV 缓存分块异步加载回 GPU 的指定槽位
        torch.cuda.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu, v_cpu = (
                    kv_cache_cpu[layer_id][i // chunk_size][0],
                    kv_cache_cpu[layer_id][i // chunk_size][1],
                )
                assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                k_chunk = k_cpu.to(self.k_buffer[0].device, non_blocking=True)
                v_chunk = v_cpu.to(self.v_buffer[0].device, non_blocking=True)
                self.k_buffer[layer_id][chunk_indices] = k_chunk
                self.v_buffer[layer_id][chunk_indices] = v_chunk
        torch.cuda.synchronize()

    def _get_key_buffer(self, layer_id: int):
        # for internal use of referencing
        # 内部使用：返回指定层的 K 缓冲区（FP8 时转换视图类型）
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.k_buffer[layer_id - self.start_layer]

    def get_key_buffer(self, layer_id: int):
        # note: get_key_buffer is hooked with synchronization for layer-wise KV cache loading
        # it is supposed to be used only by attention backend not for information purpose
        # same applies to get_value_buffer and get_kv_buffer
        # 注意：此方法带同步钩子，只应由注意力后端调用，不用于信息查询
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_key_buffer(layer_id)

    def _get_value_buffer(self, layer_id: int):
        # for internal use of referencing
        # 内部使用：返回指定层的 V 缓冲区（FP8 时转换视图类型）
        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.v_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        # 等待层传输完成后返回指定层的 V 缓冲区
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        # 返回指定层的 (K, V) 缓冲区元组
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        # 确定目标层 ID（支持 layer_id_override 用于混合注意力层映射）
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            # 若数据类型不匹配且有缩放因子，先反量化（除以 scale）
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            # 将数据类型转换为目标类型
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            # FP8 存储时，将计算类型视图转换为存储类型（uint8）
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        # 调用底层实现写入缓冲区（可能使用 JIT 内核或交替流优化）
        _set_kv_buffer_impl(
            cache_k,
            cache_v,
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
            loc,
            row_dim=self.row_dim,
            store_dtype=self.store_dtype,
            device_module=self.device_module,
            alt_stream=self.alt_stream,
            same_kv_dim=self.same_kv_dim,
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        # 将 src_loc 的 KV 缓存移动到 tgt_loc（用于 DLLM 等动态内存管理场景）
        if envs.SGLANG_NATIVE_MOVE_KV_CACHE.get():
            # 使用原生 Python 实现（逐层复制）
            move_kv_cache_native(self.k_buffer, self.v_buffer, tgt_loc, src_loc)
            return

        N = tgt_loc.numel()
        if N == 0:
            return

        assert (
            self._kv_copy_config is not None
        ), "KV copy not initialized. Set enable_kv_cache_copy=True in __init__"

        cfg = self._kv_copy_config
        cap = int(cfg.get("num_locs_upper", 256))
        grid = (self.data_ptrs.numel(), cfg["byte_tiles"])

        if N <= cap:
            # 小批量：直接使用 Triton 内核一次完成
            upper = next_power_of_2(N)
            copy_all_layer_kv_cache_tiled[grid](
                self.data_ptrs,
                self.data_strides,
                tgt_loc,
                src_loc,
                N,
                upper,
                BYTES_PER_TILE=cfg["bytes_per_tile"],
                num_warps=cfg["num_warps"],
                num_stages=2,
            )
            return

        # Huge N: chunk, but each chunk's upper is still pow2(<= cap)
        # 大批量：分块处理，每块大小不超过 cap，使用 2 的幂对齐
        for start in range(0, N, cap):
            end = min(start + cap, N)
            chunk_len = end - start
            upper = next_power_of_2(chunk_len)
            copy_all_layer_kv_cache_tiled[grid](
                self.data_ptrs,
                self.data_strides,
                tgt_loc[start:end],
                src_loc[start:end],
                chunk_len,
                upper,
                BYTES_PER_TILE=cfg["bytes_per_tile"],
                num_warps=cfg["num_warps"],
                num_stages=2,
            )


# MHATokenToKVPoolFP4：支持 FP4 量化的 MHA KV 缓存，额外存储量化缩放因子
class MHATokenToKVPoolFP4(MHATokenToKVPool):

    def _create_buffers(self):
        # 在内存节省区域中分配 FP4 量化的 K/V 缓冲区及对应缩放因子缓冲区
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # [size, head_num, head_dim] for each layer
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                m = self.size + self.page_size  # 总槽位数（含 padded 槽）
                n = self.head_num              # 头数
                k = self.head_dim              # 头维度

                scale_block_size = 16  # FP4 缩放因子的块大小（每 16 个元素共享一个缩放因子）
                self.store_dtype = torch.uint8  # FP4 数据以 uint8 格式存储
                # K 缓冲：head_dim 压缩为 head_dim//2（每 2 个 FP4 打包为 1 个 uint8）
                self.k_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                # V 缓冲：与 K 缓冲形状相同
                self.v_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

                # K 量化缩放因子缓冲：每 scale_block_size 个元素一个缩放因子
                self.k_scale_buffer = [
                    torch.zeros(
                        (m, (n * k) // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                # V 量化缩放因子缓冲：形状与 K 缩放因子相同
                self.v_scale_buffer = [
                    torch.zeros(
                        (m, (n * k) // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        # 释放 K/V 缓冲区和缩放因子缓冲区
        del self.k_buffer
        del self.v_buffer
        del self.k_scale_buffer
        del self.v_scale_buffer

    def _get_key_buffer(self, layer_id: int):
        # for internal use of referencing
        # FP4 模式：需要先反量化得到计算精度的 K 缓冲
        if self.store_dtype != self.dtype:
            cache_k_nope_fp4 = self.k_buffer[layer_id - self.start_layer].view(
                torch.uint8
            )
            cache_k_nope_fp4_sf = self.k_scale_buffer[layer_id - self.start_layer]

            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            # 批量反量化：FP4 数据 + 缩放因子 -> 全精度张量
            cache_k_nope_fp4_dequant = KVFP4QuantizeUtil.batched_dequantize(
                cache_k_nope_fp4, cache_k_nope_fp4_sf
            )
            return cache_k_nope_fp4_dequant
        return self.k_buffer[layer_id - self.start_layer]

    def _get_value_buffer(self, layer_id: int):
        # for internal use of referencing
        # FP4 模式：需要先反量化得到计算精度的 V 缓冲
        if self.store_dtype != self.dtype:
            cache_v_nope_fp4 = self.v_buffer[layer_id - self.start_layer].view(
                torch.uint8
            )
            cache_v_nope_fp4_sf = self.v_scale_buffer[layer_id - self.start_layer]

            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            cache_v_nope_fp4_dequant = KVFP4QuantizeUtil.batched_dequantize(
                cache_v_nope_fp4, cache_v_nope_fp4_sf
            )
            return cache_v_nope_fp4_dequant
        return self.v_buffer[layer_id - self.start_layer]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            # 先反量化（若有 scale），再进行 FP4 量化以适配 FP4 存储
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)

            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            # 批量量化：全精度 -> FP4 数据 + 缩放因子
            cache_k, cache_k_fp4_sf = KVFP4QuantizeUtil.batched_quantize(cache_k)
            cache_v, cache_v_fp4_sf = KVFP4QuantizeUtil.batched_quantize(cache_v)

        if self.store_dtype != self.dtype:
            # 将量化数据转换为存储数据类型视图
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

            cache_k_fp4_sf = cache_k_fp4_sf.view(self.store_dtype)
            cache_v_fp4_sf = cache_v_fp4_sf.view(self.store_dtype)

        if get_is_capture_mode() and self.alt_stream is not None:
            # Overlap the copy of K and V cache for small batch size
            # CUDA Graph 捕获模式：使用交替流并行写入 K 和 V 数据及缩放因子
            current_stream = self.device_module.current_stream()
            self.alt_stream.wait_stream(current_stream)
            self.k_buffer[layer_id - self.start_layer][loc] = cache_k

            self.k_scale_buffer[layer_id - self.start_layer][loc] = cache_k_fp4_sf
            with self.device_module.stream(self.alt_stream):
                self.v_buffer[layer_id - self.start_layer][loc] = cache_v

                self.v_scale_buffer[layer_id - self.start_layer][loc] = cache_v_fp4_sf
            current_stream.wait_stream(self.alt_stream)
        else:
            # 非 Graph 模式：顺序写入 K/V 数据和缩放因子
            self.k_buffer[layer_id - self.start_layer][loc] = cache_k
            self.v_buffer[layer_id - self.start_layer][loc] = cache_v

            self.k_scale_buffer[layer_id - self.start_layer][loc] = cache_k_fp4_sf
            self.v_scale_buffer[layer_id - self.start_layer][loc] = cache_v_fp4_sf


class HybridLinearKVPool(KVCache):
    """KV cache with separate pools for full and linear attention layers."""
    # 混合线性 KV 缓存：为全注意力层和线性注意力（Mamba）层分别维护独立的 KV 池

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        page_size: int,
        head_num: int,
        head_dim: int,
        full_attention_layer_ids: List[int],  # 全注意力层的全局层 ID 列表
        enable_kvcache_transpose: bool,
        device: str,
        mamba_pool: MambaPool,  # 已初始化的 Mamba 状态缓存池
        enable_memory_saver: bool = False,
        # TODO: refactor mla related args
        use_mla: bool = False,          # 是否使用 MLA（多层注意力，DeepSeek 模型）
        kv_lora_rank: int = None,
        qk_rope_head_dim: int = None,
        start_layer: Optional[int] = None,
    ):
        self.size = size
        self.dtype = dtype
        self.device = device
        self.full_layer_nums = len(full_attention_layer_ids)  # 全注意力层总数
        self.page_size = page_size
        self.start_layer = start_layer if start_layer is not None else 0
        self.layer_transfer_counter = None
        self.head_num = head_num
        self.head_dim = head_dim
        self.mamba_pool = mamba_pool  # 引用外部传入的 Mamba 池
        # TODO MHATransposedTokenToKVPool if enable_kvcache_transpose is True
        assert not enable_kvcache_transpose  # 暂不支持转置 KV 缓存布局
        self.use_mla = use_mla
        if not use_mla:
            # 非 MLA 模式：使用 MHA KV 池（或平台特定实现）
            TokenToKVPoolClass = MHATokenToKVPool

            if current_platform.is_out_of_tree():
                TokenToKVPoolClass = current_platform.get_mha_kv_pool_cls()
            elif _is_npu:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMHATokenToKVPool,
                )

                TokenToKVPoolClass = NPUMHATokenToKVPool

            self.full_kv_pool = TokenToKVPoolClass(
                size=size,
                page_size=self.page_size,
                dtype=dtype,
                head_num=head_num,
                head_dim=head_dim,
                layer_num=self.full_layer_nums,
                device=device,
                enable_memory_saver=enable_memory_saver,
            )
        else:
            # MLA 模式：使用 MLA KV 池（或平台特定实现）
            TokenToKVPoolClass = MLATokenToKVPool

            if current_platform.is_out_of_tree():
                TokenToKVPoolClass = current_platform.get_mla_kv_pool_cls()
            elif _is_npu:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMLATokenToKVPool,
                )

                TokenToKVPoolClass = NPUMLATokenToKVPool

            self.full_kv_pool = TokenToKVPoolClass(
                size=size,
                page_size=self.page_size,
                dtype=dtype,
                layer_num=self.full_layer_nums,
                device=device,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                enable_memory_saver=enable_memory_saver,
            )
        # 建立全局 layer_id -> 全注意力层本地索引的映射
        self.full_attention_layer_id_mapping = {
            id: i for i, id in enumerate(full_attention_layer_ids)
        }
        # 计算内存使用量
        if use_mla:
            self.mem_usage = self.get_kv_size_bytes() / GB
        else:
            k_size, v_size = self.get_kv_size_bytes()
            self.mem_usage = (k_size + v_size) / GB

    def get_kv_size_bytes(self):
        # 返回底层全注意力 KV 池的字节大小
        return self.full_kv_pool.get_kv_size_bytes()

    def get_contiguous_buf_infos(self):
        # 返回底层全注意力 KV 池的连续缓冲区信息（用于 RDMA）
        return self.full_kv_pool.get_contiguous_buf_infos()

    def get_state_buf_infos(self):
        # 返回 Mamba 状态池的缓冲区信息（用于 RDMA）
        mamba_data_ptrs, mamba_data_lens, mamba_item_lens = (
            self.mamba_pool.get_contiguous_buf_infos()
        )
        return mamba_data_ptrs, mamba_data_lens, mamba_item_lens

    def get_state_dim_per_tensor(self):
        """Get the sliceable dimension size for each mamba state tensor."""
        # 获取每个 Mamba 状态张量中被 TP 切片的维度大小
        return self.mamba_pool.get_state_dim_per_tensor()

    def maybe_get_custom_mem_pool(self):
        # 返回底层全注意力 KV 池的自定义内存池
        return self.full_kv_pool.maybe_get_custom_mem_pool()

    def _transfer_full_attention_id(self, layer_id: int):
        # 将全局 layer_id 转换为全注意力层的本地索引
        if layer_id not in self.full_attention_layer_id_mapping:
            raise ValueError(
                f"{layer_id=} not in full attention layers: {self.full_attention_layer_id_mapping.keys()}"
            )
        return self.full_attention_layer_id_mapping[layer_id]

    def register_layer_transfer_counter(
        self, layer_transfer_counter: "LayerDoneCounter"
    ):
        # 注册层传输计数器（在 HybridLinearPool 层处理同步，底层池无需额外等待）
        self.layer_transfer_counter = layer_transfer_counter
        # The layer-wise wait logic is executed at the Hybrid LinearPool level;
        # no additional wait is needed in the full_kv_pool
        self.full_kv_pool.register_layer_transfer_counter(None)

    def _wait_for_layer(self, layer_id: int):
        # 等待指定层的传输完成（若有层传输计数器）
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

    def get_key_buffer(self, layer_id: int):
        # 等待层传输完成后，将全局 layer_id 转为本地索引并获取 K 缓冲
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_key_buffer(layer_id)

    def get_value_buffer(self, layer_id: int):
        # 等待层传输完成后，将全局 layer_id 转为本地索引并获取 V 缓冲
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        # 等待层传输完成后，返回指定全注意力层的 (K, V) 缓冲
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_kv_buffer(layer_id)

    @contextmanager
    def _transfer_id_context(self, layer: RadixAttention):
        # 临时覆盖 layer.layer_id 为本地索引，用于需要 layer 对象的底层调用

        @contextmanager
        def _patch_layer_id(layer):
            # 保存并替换 layer_id，退出时恢复原始值
            original_layer_id = layer.layer_id
            layer.layer_id = self._transfer_full_attention_id(layer.layer_id)
            try:
                yield
            finally:
                layer.layer_id = original_layer_id

        with _patch_layer_id(layer):
            yield

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ):
        # 将全局 layer_id 转换为本地索引后，写入底层全注意力 KV 池
        layer_id = self._transfer_full_attention_id(layer.layer_id)
        if not self.use_mla:
            # 非 MLA 模式：直接传递本地 layer_id
            self.full_kv_pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id,
            )
        else:
            # MLA 模式：需要传递 layer 对象（因为底层读取 layer 属性），使用上下文临时替换 ID
            with self._transfer_id_context(layer):
                self.full_kv_pool.set_kv_buffer(
                    layer,
                    loc,
                    cache_k,
                    cache_v,
                )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        # 委托底层全注意力 KV 池执行缓存移动
        self.full_kv_pool.move_kv_cache(tgt_loc, src_loc)

    def get_cpu_copy(self, indices, mamba_indices=None):
        # 同时获取 KV 缓存和 Mamba 状态的 CPU 副本
        kv_cpu = self.full_kv_pool.get_cpu_copy(indices)
        mamba_cpu = (
            self.mamba_pool.get_cpu_copy(mamba_indices)
            if mamba_indices is not None
            else None
        )
        return kv_cpu, mamba_cpu

    def load_cpu_copy(self, cache_cpu, indices, mamba_indices=None):
        # 将 CPU 副本分别加载回 KV 缓存和 Mamba 状态池
        kv_cpu, mamba_cpu = cache_cpu
        self.full_kv_pool.load_cpu_copy(kv_cpu, indices)
        if mamba_cpu is not None and mamba_indices is not None:
            self.mamba_pool.load_cpu_copy(mamba_cpu, mamba_indices)

    def get_v_head_dim(self):
        # 返回 V 缓冲区的头维度（从第 0 层的形状中推断）
        return self.full_kv_pool.get_value_buffer(0).shape[-1]

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        # MLA 模式：写入 no-positional-encoding K 和 rotary-positional-encoding K 到底层池
        assert self.use_mla, "set_mla_kv_buffer called when use_mla is False"
        with self._transfer_id_context(layer):
            self.full_kv_pool.set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        # MLA 模式：从底层池读取 no-pe K 和 rope K
        assert self.use_mla, "get_mla_kv_buffer called when use_mla is False"
        with self._transfer_id_context(layer):
            return self.full_kv_pool.get_mla_kv_buffer(layer, loc, dst_dtype)


# MLATokenToKVPool：多层注意力（MLA）的 KV 缓存，将 K_nope 和 K_rope 拼接存储在单一缓冲区
class MLATokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,       # KV LoRA 压缩维度（K_nope 的维度）
        qk_rope_head_dim: int,   # 旋转位置编码的头维度（K_rope 的维度）
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        use_nsa: bool = False,   # 是否使用 NSA（分级稀疏注意力）
        override_kv_cache_dim: Optional[int] = None,  # NSA FP8 存储时覆盖缓存维度
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.use_nsa = use_nsa
        # NSA FP8 存储条件：使用 NSA 且数据类型为 FP8 且有覆盖维度
        self.nsa_kv_cache_store_fp8 = (
            use_nsa
            and dtype == torch.float8_e4m3fn
            and override_kv_cache_dim is not None
        )
        # When override_kv_cache_dim is provided with nsa model, we assume the
        # override kv cache dim is correct and use it directly.
        # NSA FP8 模式：直接使用覆盖维度；否则使用 kv_lora_rank + qk_rope_head_dim
        self.kv_cache_dim = (
            override_kv_cache_dim
            if self.nsa_kv_cache_store_fp8
            else (kv_lora_rank + qk_rope_head_dim)
        )

        self._create_buffers()  # 分配 KV 缓冲区

        # 预先收集所有层的数据指针供 Triton 内核使用
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        if not use_nsa:
            # NSA will allocate indexer KV cache later and then log the total size
            # 非 NSA 模式立即记录分配日志；NSA 模式等索引缓冲分配完后再记录
            self._finalize_allocation_log(size)

    def _create_buffers(self):
        # 分配 MLA KV 缓冲区：shape=(size+page_size, 1, kv_cache_dim)
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                # 槽位 0 保留为 padded token 虚拟输出；head_num=1（MLA 压缩所有头为 1 个向量）
                self.kv_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, 1, self.kv_cache_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        # 释放 KV 缓冲区引用
        del self.kv_buffer

    def get_kv_size_bytes(self):
        # 计算所有层 KV 缓冲区的总字节数
        assert hasattr(self, "kv_buffer")
        kv_size_bytes = 0
        for kv_cache in self.kv_buffer:
            kv_size_bytes += get_tensor_size_bytes(kv_cache)
        return kv_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        # MLA has only one kv_buffer, so only the information of this buffer needs to be returned.
        # MLA 只有一个统一 kv_buffer（K_nope + K_rope 合并存储）
        kv_data_ptrs = [self.kv_buffer[i].data_ptr() for i in range(self.layer_num)]
        kv_data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
        kv_item_lens = [
            self.kv_buffer[i][0].nbytes * self.page_size for i in range(self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_key_buffer(self, layer_id: int):
        # 等待层传输完成后返回完整 KV 缓冲（MLA 中 K 和 V 共享同一缓冲）
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer].view(self.dtype)

        return self.kv_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        # MLA 中 V 等同于 K_nope 部分（前 kv_lora_rank 个维度）
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer][
                ..., : self.kv_lora_rank
            ].view(self.dtype)
        return self.kv_buffer[layer_id - self.start_layer][..., : self.kv_lora_rank]

    def get_kv_buffer(self, layer_id: int):
        # 返回 (K, V) 元组，V 是 K_nope 子集
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        # 将 K 张量（已包含 K_nope 和 K_rope）写入指定槽位
        layer_id = layer.layer_id
        assert not self.nsa_kv_cache_store_fp8  # NSA FP8 模式不走此路径
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)

        if self.store_dtype != self.dtype:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k.view(
                self.store_dtype
            )
        else:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        # 将分离的 K_nope 和 K_rope 拼接后写入 KV 缓冲区（支持多种量化模式）
        layer_id = layer.layer_id

        if _is_hip and self.use_nsa and self.dtype == fp8_dtype:
            # HIP FP8 path uses raw MLA KV layout (nope + rope) without per-block scales.
            # Fuse BF16/FP16 -> FP8 cast with paged KV write.
            # HIP FP8 路径：原始 MLA KV 布局，融合类型转换和分页写入
            set_mla_kv_buffer_triton_fp8_quant(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope,
                cache_k_rope,
                fp8_dtype,
            )
        elif self.nsa_kv_cache_store_fp8:
            # OPTIMIZATION: Quantize k_nope and k_rope separately to avoid concat overhead
            # This also enables reuse of set_mla_kv_buffer_triton two-tensor write path
            # quantize_k_cache_separate returns (nope_part, rope_part) as uint8 bytes
            # 优化：分别量化 K_nope 和 K_rope，避免拼接开销；返回 uint8 字节格式
            cache_k_nope_fp8, cache_k_rope_fp8 = quantize_k_cache_separate(
                cache_k_nope, cache_k_rope
            )

            # Reuse existing two-tensor write kernel (works with FP8 byte layout)
            # cache_k_nope_fp8: (num_tokens, 1, 528) uint8 [nope_fp8(512) | scales(16)]
            # cache_k_rope_fp8: (num_tokens, 1, 128) uint8 [rope_bf16_bytes(128)]
            # 复用双张量写入内核（与 FP8 字节布局兼容）
            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope_fp8,
                cache_k_rope_fp8,
            )
        else:
            # 标准路径：将 K_nope 和 K_rope 类型转换后使用 Triton 内核写入
            if cache_k_nope.dtype != self.dtype:
                cache_k_nope = cache_k_nope.to(self.dtype)
                cache_k_rope = cache_k_rope.to(self.dtype)
            if self.store_dtype != self.dtype:
                cache_k_nope = cache_k_nope.view(self.store_dtype)
                cache_k_rope = cache_k_rope.view(self.store_dtype)

            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope,
                cache_k_rope,
            )

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        # get k nope and k rope from the kv buffer, and optionally cast them to dst_dtype.
        # 从 KV 缓冲区中拆分出 K_nope 和 K_rope，可选转换到目标数据类型
        layer_id = layer.layer_id
        kv_buffer = self.get_key_buffer(layer_id)
        dst_dtype = dst_dtype or self.dtype
        # 分配输出张量
        cache_k_nope = torch.empty(
            (loc.shape[0], 1, self.kv_lora_rank),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        cache_k_rope = torch.empty(
            (loc.shape[0], 1, self.qk_rope_head_dim),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        # 使用 Triton 内核从合并缓冲区中分离 K_nope 和 K_rope
        get_mla_kv_buffer_triton(kv_buffer, loc, cache_k_nope, cache_k_rope)
        return cache_k_nope, cache_k_rope

    def get_cpu_copy(self, indices, mamba_indices=None):
        # 同步 GPU 并将指定槽位的 MLA KV 缓存分块异步复制到 CPU
        torch.cuda.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = self.kv_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append(kv_cpu)
        torch.cuda.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        # 将 CPU 端的 MLA KV 缓存分块异步加载回 GPU
        torch.cuda.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = kv_cache_cpu[layer_id][i // chunk_size]
                assert kv_cpu.shape[0] == len(chunk_indices)
                kv_chunk = kv_cpu.to(self.kv_buffer[0].device, non_blocking=True)
                self.kv_buffer[layer_id][chunk_indices] = kv_chunk
        torch.cuda.synchronize()


# MLATokenToKVPoolFP4：支持 FP4 量化的 MLA KV 缓存，额外存储量化缩放因子
class MLATokenToKVPoolFP4(MLATokenToKVPool):

    def _create_buffers(self):
        # 分配 FP4 量化的 MLA KV 缓冲区及对应缩放因子缓冲区
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                m = self.size + self.page_size  # 总槽位数
                n = 1  # head_num（MLA 压缩为 1 个头）
                k = self.kv_cache_dim  # head_dim（kv_lora_rank + qk_rope_head_dim）

                scale_block_size = 16  # FP4 缩放因子块大小
                self.store_dtype = torch.uint8  # FP4 数据以 uint8 格式存储

                # KV 缓冲：head_dim 压缩为 head_dim//2
                self.kv_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

                # KV 量化缩放因子缓冲：每 scale_block_size 个元素一个缩放因子
                self.kv_scale_buffer = [
                    torch.zeros(
                        (m, k // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        # 释放 KV 缓冲区和缩放因子缓冲区
        del self.kv_buffer
        del self.kv_scale_buffer

    def get_key_buffer(self, layer_id: int):
        # FP4 模式：反量化后返回全精度 KV 缓冲
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            cache_k_nope_fp4 = self.kv_buffer[layer_id - self.start_layer].view(
                torch.uint8
            )
            cache_k_nope_fp4_sf = self.kv_scale_buffer[layer_id - self.start_layer]

            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            # 使用缩放因子反量化 FP4 数据
            cache_k_nope_fp4_dequant = KVFP4QuantizeUtil.batched_dequantize(
                cache_k_nope_fp4, cache_k_nope_fp4_sf
            )
            return cache_k_nope_fp4_dequant

        return self.kv_buffer[layer_id - self.start_layer]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        # FP4 模式：量化后写入 KV 缓冲区和缩放因子缓冲区
        layer_id = layer.layer_id
        assert not self.nsa_kv_cache_store_fp8  # NSA FP8 模式不走此路径
        if cache_k.dtype != self.dtype:
            from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

            # 量化：全精度 -> FP4 数据 + 缩放因子
            cache_k_fp4, cache_k_fp4_sf = KVFP4QuantizeUtil.batched_quantize(cache_k)

        if self.store_dtype != self.dtype:
            # 将 FP4 数据和缩放因子写入对应缓冲区
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k_fp4.view(
                self.store_dtype
            )
            self.kv_scale_buffer[layer_id - self.start_layer][loc] = (
                cache_k_fp4_sf.view(self.store_dtype)
            )
        else:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        # FP4 模式：对 K_nope 和 K_rope 分别量化后写入缓冲区
        layer_id = layer.layer_id

        if self.nsa_kv_cache_store_fp8:
            # original cache_k: (num_tokens, num_heads 1, hidden 576); we unsqueeze the page_size=1 dim here
            # TODO no need to cat
            # NSA FP8 路径：拼接后整体量化
            cache_k = torch.cat([cache_k_nope, cache_k_rope], dim=-1)
            cache_k = quantize_k_cache(cache_k.unsqueeze(1)).squeeze(1)
            cache_k = cache_k.view(self.store_dtype)
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k
        else:
            if cache_k_nope.dtype != self.dtype:
                from sglang.srt.layers.quantization.kvfp4_tensor import (
                    KVFP4QuantizeUtil,
                )

                # 分别量化 K_nope 和 K_rope
                cache_k_nope_fp4, cache_k_nope_fp4_sf = (
                    KVFP4QuantizeUtil.batched_quantize(cache_k_nope)
                )
                cache_k_rope_fp4, cache_k_rope_fp4_sf = (
                    KVFP4QuantizeUtil.batched_quantize(cache_k_rope)
                )

            if self.store_dtype != self.dtype:
                cache_k_nope = cache_k_nope.view(self.store_dtype)
                cache_k_rope = cache_k_rope.view(self.store_dtype)

            # 使用 Triton 内核写入 FP4 数据
            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope_fp4,
                cache_k_rope_fp4,
            )
            # 使用 Triton 内核写入缩放因子
            set_mla_kv_scale_buffer_triton(
                self.kv_scale_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope_fp4_sf,
                cache_k_rope_fp4_sf,
            )


# NSATokenToKVPool：分级稀疏注意力（NSA）的 KV 缓存，在 MLA 基础上额外维护索引 K 缓冲
class NSATokenToKVPool(MLATokenToKVPool):
    quant_block_size = 128  # FP8 量化块大小（每 128 个元素共享一个缩放因子）
    index_k_with_scale_buffer_dtype = torch.uint8  # 索引 K 缓冲的存储类型
    rope_storage_dtype = torch.bfloat16  # rope 部分始终以 bfloat16 存储

    def __init__(
        self,
        size: int,
        page_size: int,
        kv_lora_rank: int,
        dtype: torch.dtype,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        index_head_dim: int,   # 索引 K 的头维度（NSA 中为 128）
        enable_memory_saver: bool,
        kv_cache_dim: int,     # KV 缓存的实际维度（NSA 可能与 kv_lora_rank+rope_dim 不同）
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        index_buf_size: Optional[int] = None,  # 索引缓冲大小（默认与 size 相同）
    ):
        # 若 kv_cache_dim 与默认值不同则传入覆盖维度
        override_dim = (
            kv_cache_dim if kv_cache_dim != kv_lora_rank + qk_rope_head_dim else None
        )

        super().__init__(
            size,
            page_size,
            dtype,
            kv_lora_rank,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
            use_nsa=True,
            override_kv_cache_dim=override_dim,
        )
        # self.index_k_dtype = torch.float8_e4m3fn
        # self.index_k_scale_dtype = torch.float32
        self.index_head_dim = index_head_dim  # 索引 K 的头维度（固定为 128）
        if index_buf_size is None:
            index_buf_size = size
        # num head == 1 and head dim == 128 for index_k in NSA
        assert index_head_dim == 128  # NSA 中索引 K 头维度固定为 128

        # HIP 平台：page_size=1；CUDA 平台：page_size=64
        if _is_hip:
            assert self.page_size == 1
        else:
            assert self.page_size == 64
        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else nullcontext()
        ):
            # 为每层分配索引 K（含缩放因子）缓冲区
            # 布局：每页存 (fp8_data | fp32_scales)，合并为 uint8 连续存储
            self.index_k_with_scale_buffer = [
                torch.zeros(
                    # Layout:
                    #     ref: test_attention.py :: kv_cache_cast_to_fp8
                    #     shape: (num_pages, page_size 64 * head_dim 128 + page_size 64 * fp32_nbytes 4)
                    #     data: for page i,
                    #         * buf[i, :page_size * head_dim] for fp8 data
                    #         * buf[i, page_size * head_dim:].view(float32) for scale
                    # 每页：前 page_size*head_dim 字节为 FP8 数据，后续字节为 FP32 缩放因子
                    (
                        (index_buf_size + page_size + 1) // self.page_size,
                        self.page_size
                        * (
                            index_head_dim + index_head_dim // self.quant_block_size * 4
                        ),
                    ),
                    dtype=self.index_k_with_scale_buffer_dtype,
                    device=device,
                )
                for _ in range(layer_num)
            ]
        # 记录完整分配日志（包含索引缓冲大小）
        self._finalize_allocation_log(size)

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        # 等待层传输完成后返回指定层的索引 K 缓冲（含缩放因子）
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.index_k_with_scale_buffer[layer_id - self.start_layer]

    def get_index_k_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        # 从分页索引缓冲中读取连续的索引 K 数据
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        return index_buf_accessor.GetK.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def get_index_k_scale_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        # 从分页索引缓冲中读取连续的索引 K 缩放因子
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        return index_buf_accessor.GetS.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ):
        """
        Fused method to get both index K and scale data in a single call using Triton.
        More efficient than calling get_index_k_continuous and get_index_k_scale_continuous separately.

        :param layer_id: Layer index
        :param seq_len: Sequence length
        :param page_indices: Page indices tensor
        :return: tuple of (k_fp8, k_scale) where
                 k_fp8: (seq_len, index_head_dim), uint8
                 k_scale: (seq_len, 4), uint8
        """
        # 融合方法：一次 Triton 调用同时获取索引 K 和缩放因子（比分两次调用更高效）
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        return index_buf_accessor.GetKAndS.execute(
            self,
            buf,
            page_indices=page_indices,
            seq_len_tensor=seq_len_tensor,
            seq_len_sum=seq_len_sum,
            max_seq_len=max_seq_len,
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        # 将索引 K 数据和缩放因子写入指定槽位的索引缓冲区
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        index_buf_accessor.SetKAndS.execute(
            pool=self, buf=buf, loc=loc, index_k=index_k, index_k_scale=index_k_scale
        )

    def get_state_buf_infos(self):
        # 返回索引 K 缓冲区的数据指针、字节长度和单元素字节长度（用于 RDMA）
        data_ptrs = [
            self.index_k_with_scale_buffer[i].data_ptr() for i in range(self.layer_num)
        ]
        data_lens = [
            self.index_k_with_scale_buffer[i].nbytes for i in range(self.layer_num)
        ]
        item_lens = [
            self.index_k_with_scale_buffer[i][0].nbytes for i in range(self.layer_num)
        ]
        return data_ptrs, data_lens, item_lens

    def get_kv_size_bytes(self):
        # 总 KV 缓存字节数 = 父类 KV 缓冲 + 索引 K 缓冲
        kv_size_bytes = super().get_kv_size_bytes()
        for index_k_cache in self.index_k_with_scale_buffer:
            kv_size_bytes += get_tensor_size_bytes(index_k_cache)
        return kv_size_bytes


def move_kv_cache_native(
    k_buffer: List[torch.Tensor],
    v_buffer: List[torch.Tensor],
    tgt_loc: torch.Tensor,
    src_loc: torch.Tensor,
):
    # 原生 Python 实现的 KV 缓存移动：逐层将 src_loc 的数据复制到 tgt_loc
    if tgt_loc.numel() == 0:
        return  # 无需移动时直接返回

    tgt_loc_flat = tgt_loc.view(-1).long()  # 展平并转换为 int64，用于张量索引
    src_loc_flat = src_loc.view(-1).long()
    for k_cache, v_cache in zip(k_buffer, v_buffer):
        # 逐层复制：使用高级索引进行原地复制
        k_cache[tgt_loc_flat] = k_cache[src_loc_flat]
        v_cache[tgt_loc_flat] = v_cache[src_loc_flat]


@triton.jit
def copy_all_layer_kv_cache_tiled(
    data_ptrs,
    strides,
    tgt_loc_ptr,
    src_loc_ptr,
    num_locs,
    num_locs_upper: tl.constexpr,
    BYTES_PER_TILE: tl.constexpr,
):
    """2D tiled kernel. Safe for in-place copy."""
    # 2D 分块 Triton 内核：同时处理所有层（bid 维度）和所有字节分块（tid 维度）
    # 安全支持原地复制（src 和 tgt 可以是同一缓冲区的不同位置）
    bid = tl.program_id(0)  # 块 ID：对应某一层的 K 或 V 缓冲
    tid = tl.program_id(1)  # 线程 ID：对应字节分块偏移

    stride = tl.load(strides + bid)      # 该缓冲区每行的字节数
    base_ptr = tl.load(data_ptrs + bid)  # 该缓冲区的基地址
    base_ptr = tl.cast(base_ptr, tl.pointer_type(tl.uint8))  # 转换为字节指针

    # 计算当前字节分块的偏移范围
    byte_off = tid * BYTES_PER_TILE + tl.arange(0, BYTES_PER_TILE)
    mask_byte = byte_off < stride  # 超出行边界的字节跳过
    tl.multiple_of(byte_off, 16)   # 提示对齐，优化内存访问

    # 加载所有需要处理的位置索引（使用 mask 处理末尾对齐填充）
    loc_idx = tl.arange(0, num_locs_upper)
    mask_loc = loc_idx < num_locs

    src = tl.load(src_loc_ptr + loc_idx, mask=mask_loc, other=0)  # 源位置索引
    tgt = tl.load(tgt_loc_ptr + loc_idx, mask=mask_loc, other=0)  # 目标位置索引

    # 计算每个位置每个字节分块的源/目标地址
    src_ptr = base_ptr + src[:, None] * stride + byte_off[None, :]
    tgt_ptr = base_ptr + tgt[:, None] * stride + byte_off[None, :]

    # 组合位置掩码和字节掩码，批量复制
    mask = mask_loc[:, None] & mask_byte[None, :]
    vals = tl.load(src_ptr, mask=mask)
    tl.store(tgt_ptr, vals, mask=mask)
