# 允许在类型注解中使用前向引用（PEP 563）
from __future__ import annotations

# 抽象基类支持
from abc import ABC, abstractmethod
# 类型检查时导入（避免循环依赖）
from typing import TYPE_CHECKING, Optional

import torch

# 内核 API 调试装饰器，用于记录调用信息
from sglang.kernel_api_logging import debug_kernel_api
# NPU 设备检测工具
from sglang.srt.utils.common import is_npu

if TYPE_CHECKING:
    # 仅在类型检查阶段导入，避免运行时循环引用
    from sglang.srt.layers.attention.nsa.nsa_indexer import BaseIndexerMetadata
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.speculative.spec_info import SpecInput


class AttentionBackend(ABC):
    """The base class of attention backends"""
    # 所有注意力后端实现的抽象基类，定义统一的前向接口规范

    @abstractmethod
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        # 初始化一次前向传播所需的元数据（子类必须实现）
        raise NotImplementedError()

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Init the global shared states for cuda graph."""
        # 初始化 CUDA Graph 捕获所需的全局共享状态缓冲区
        raise NotImplementedError()

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        """Init the metadata for a forward pass for capturing a cuda graph."""
        # 在 CUDA Graph 捕获阶段初始化前向元数据，用于录制计算图
        raise NotImplementedError()

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        """Init the metadata for a forward pass for replaying a cuda graph."""
        # 在 CUDA Graph 回放阶段初始化前向元数据，更新动态参数
        raise NotImplementedError()

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for padded seq lens. Typically, it is 0 or 1."""
        # 返回 CUDA Graph 中序列长度张量的填充值（通常为 0 或 1）
        raise NotImplementedError()

    def get_verify_buffers_to_fill_after_draft(self):
        """
        Return buffers of verify attention kernels that needs to be filled after draft.

        Typically, these are tree mask and position buffers.
        """
        # 返回 draft 阶段之后需要填充的 verify 注意力缓冲区（树形掩码和位置编码缓冲）
        return [None, None]

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        """
        Update the buffers returned by get_verify_fill_after_draft_buffers if needed.

        Here, we need to redo the computation of all metadata of the attention backend
        that depends on tree mask and position buffers.
        """
        # 在 draft 完成后更新 verify 阶段所需的注意力后端元数据（树形掩码与位置缓冲相关）
        raise NotImplementedError()

    @debug_kernel_api  # 装饰器：记录内核 API 调用信息，方便调试
    def forward(
        self,
        q: torch.Tensor,   # Query 张量
        k: torch.Tensor,   # Key 张量
        v: torch.Tensor,   # Value 张量
        layer: RadixAttention,          # 当前注意力层对象
        forward_batch: ForwardBatch,    # 当前批次的前向信息
        save_kv_cache: bool = True,     # 是否将 KV 写入缓存
        **kwargs,
    ):
        """Run forward on an attention layer."""
        # 根据当前前向模式（idle/decode/mixed/extend）分发到对应的子方法
        if forward_batch.forward_mode.is_idle():
            # 空闲模式：直接返回全零输出，不做实际计算
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
        elif forward_batch.forward_mode.is_decode():
            # 解码模式：逐 token 生成，每步只处理一个新 token
            return self.forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        elif forward_batch.forward_mode.is_mixed() and is_npu():
            # NPU 上的混合模式（同时含 prefill 和 decode 请求）
            return self.forward_mixed(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        else:
            # extend 模式：处理 prefill 或续写序列
            return self.forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        """Run a forward for decode."""
        # 解码阶段前向（子类需实现）：逐步生成 token 时的注意力计算
        raise NotImplementedError()

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        """Run a forward for extend."""
        # extend/prefill 阶段前向（子类需实现）：处理输入上下文的注意力计算
        raise NotImplementedError()

    def forward_mixed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        """Run a forward for mix."""
        # 混合模式前向（子类需实现）：同批次包含 prefill 和 decode 请求时使用
        raise NotImplementedError()

    def support_triton(self):
        """Check if the current backend supports triton."""
        # 检查该后端是否支持 Triton 内核（默认返回 True）
        return True

    def get_indexer_metadata(
        self,
        layer_id: int,
        forward_batch: ForwardBatch,
    ) -> Optional[BaseIndexerMetadata]:
        """Get the indexer metadata. None means don't support indexer."""
        # 获取 NSA 索引器元数据；返回 None 表示当前后端不支持索引器
        return None
