# 类型提示：Optional 用于可选参数
from typing import Optional

import torch

# 导入注意力后端基类
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
# 导入 NSA 索引器基础元数据类型
from sglang.srt.layers.attention.nsa.nsa_indexer import BaseIndexerMetadata
# 导入 RadixAttention 层
from sglang.srt.layers.radix_attention import RadixAttention
# 导入前向批次信息与前向模式枚举
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
# 导入模型运行器
from sglang.srt.model_executor.model_runner import ModelRunner
# 导入推测解码输入信息
from sglang.srt.speculative.spec_info import SpecInput


class HybridAttnBackend(AttentionBackend):
    """Support different backends for prefill and decode."""
    # 混合注意力后端：prefill 和 decode 阶段可使用不同的后端实现，灵活支持多种加速策略

    def __init__(
        self,
        model_runner: ModelRunner,
        prefill_backend: AttentionBackend,  # 用于 prefill/extend 阶段的后端
        decode_backend: AttentionBackend,   # 用于 decode 阶段的后端
    ):
        self.model_runner = model_runner
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend
        # 记录 KV 缓存数据类型
        self.data_type = model_runner.kv_cache_dtype

    def _select_backend(self, forward_mode: ForwardMode) -> AttentionBackend:
        """
        Select the appropriate attention backend based on the forward mode.

        Args:
            forward_mode: The current forward mode indicating the operation type

        Returns:
            The selected attention backend (prefill or decode)

        Note:
            - decode_or_idle: Always uses decode backend
            - target_verify or draft_extend: Uses decode backend if speculative_attention_mode is "decode", otherwise prefill backend
            - prefill: Always uses prefill backend
        """
        # 根据前向模式选择合适的后端：decode/idle → decode 后端；prefill → prefill 后端
        if forward_mode.is_decode_or_idle():
            # decode 或 idle 模式始终使用 decode 后端
            return self.decode_backend
        elif forward_mode.is_target_verify() or forward_mode.is_draft_extend():
            # 推测解码验证/draft-extend 阶段，根据 speculative_attention_mode 参数选择
            return (
                self.decode_backend
                if self.model_runner.server_args.speculative_attention_mode == "decode"
                else self.prefill_backend
            )
        else:
            # prefill/extend 模式使用 prefill 后端
            return self.prefill_backend

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # 根据当前批次的前向模式选择后端，并初始化其元数据
        backend = self._select_backend(forward_batch.forward_mode)
        backend.init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        # decode 后端始终需要初始化 CUDA Graph 状态
        self.decode_backend.init_cuda_graph_state(max_bs, max_num_tokens)
        if (
            self.model_runner.server_args.speculative_algorithm is not None
            and self.model_runner.server_args.speculative_attention_mode == "prefill"
        ):
            # When speculative decoding is enabled, we need to initialize the backend
            # that will be used for target_verify.
            # 推测解码且 target_verify 使用 prefill 后端时，也需初始化 prefill 后端的 CUDA Graph 状态
            self.prefill_backend.init_cuda_graph_state(max_bs, max_num_tokens)

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
        # CUDA Graph 捕获阶段：将调用委托给根据 forward_mode 选取的后端
        backend = self._select_backend(forward_mode)
        backend.init_forward_metadata_capture_cuda_graph(
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            encoder_lens,
            forward_mode,
            spec_info,
        )

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
        # CUDA Graph 回放阶段：委托给对应后端更新动态元数据
        backend = self._select_backend(forward_mode)
        backend.init_forward_metadata_replay_cuda_graph(
            bs,
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            encoder_lens,
            forward_mode,
            spec_info,
            seq_lens_cpu,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        # 使用 decode 后端的填充值（混合模式下以 decode 后端为准）
        return self.decode_backend.get_cuda_graph_seq_len_fill_value()

    def forward(
        self,
        q: Optional[torch.Tensor] = None,  # For full attention
        k: Optional[torch.Tensor] = None,  # For full attention
        v: Optional[torch.Tensor] = None,  # For full attention
        layer: Optional[RadixAttention] = None,
        forward_batch: Optional[ForwardBatch] = None,
        save_kv_cache: bool = True,
        *,
        mixed_qkv: Optional[torch.Tensor] = None,  # For linear attention
        a: Optional[torch.Tensor] = None,  # For linear attention
        b: Optional[torch.Tensor] = None,  # For linear attention
        **kwargs,
    ):
        """Forward method that supports both regular attention (q, k, v) and linear attention (mixed_qkv, a, b)."""
        # 统一前向入口：同时支持标准注意力（q/k/v）和线性注意力（mixed_qkv/a/b）两种调用方式
        backend = self._select_backend(forward_batch.forward_mode)
        if mixed_qkv is not None:
            # 线性注意力路径：使用 mixed_qkv、a、b 参数
            return backend.forward(
                layer=layer,
                forward_batch=forward_batch,
                save_kv_cache=save_kv_cache,
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                **kwargs,
            )
        # 标准注意力路径：使用 q、k、v 参数
        return backend.forward(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        # decode 阶段：直接委托给 decode 后端
        return self.decode_backend.forward_decode(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        # extend/prefill 阶段：根据 forward_mode 选取合适后端（推测解码时可能用 decode 后端）
        backend = self._select_backend(forward_batch.forward_mode)
        return backend.forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def get_indexer_metadata(
        self, layer_id: int, forward_batch: ForwardBatch
    ) -> Optional[BaseIndexerMetadata]:
        # 获取 NSA 索引器元数据，委托给当前前向模式对应的后端
        backend = self._select_backend(forward_batch.forward_mode)
        return backend.get_indexer_metadata(layer_id, forward_batch)

    def forward(
        self,
        q: torch.Tensor = None,
        k: torch.Tensor = None,
        v: torch.Tensor = None,
        layer: RadixAttention = None,
        forward_batch: ForwardBatch = None,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Delegate forward to the appropriate backend based on forward mode."""
        # 根据 forward_mode 将前向计算委托给 prefill 或 decode 后端
        backend = self._select_backend(forward_batch.forward_mode)
        return backend.forward(
            q=q,
            k=k,
            v=v,
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            **kwargs,
        )
