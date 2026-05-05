# 导入日志模块
import logging
# 导入数学工具（用于计算 ALiBi 衰减斜率）
import math
# 导入类型注解
from typing import Optional, Union

# 导入 PyTorch
import torch

# 导入 Mamba 注意力后端基类
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
# 导入 Lightning 线性注意力 Kernel 和 decode 前向函数
from sglang.srt.layers.attention.linear.lightning_attn import (
    BailingLinearKernel,         # 线性注意力 Kernel 封装（含 prefix 前向）
    linear_decode_forward_triton, # decode 阶段线性注意力前向函数
)
# 导入 Bailing 线性注意力元数据（含 prefill/decode 信息）
from sglang.srt.layers.attention.linear.linear_metadata import BailingLinearMetadata
# 导入 SegLA（Segment Linear Attention）元数据和前向函数
from sglang.srt.layers.attention.linear.seg_la import SegLaMeta, seg_la_fwd
# 导入 RadixAttention 层接口
from sglang.srt.layers.radix_attention import RadixAttention
# 导入前向批次信息和前向模式枚举
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
# 导入 ModelRunner
from sglang.srt.model_executor.model_runner import ModelRunner
# 导入投机解码信息（EAGLE 算法的草稿/验证输入）
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput

logger = logging.getLogger(__name__)


class LightningAttentionBackend(MambaAttnBackendBase):
    """
    Note about the init:
    - If no spec decoding
        - FlashAttentionBackend will be init once when the server starts.
    - If spec decoding
        - FlashAttentionBackend will be init once for the target worker
        - FlashAttentionMultiStepBackend will be once for the draft worker
            - It will spawn num_steps FlashAttentionBackend for the draft worker

    Note about CUDA Graph:
    - We only support CUDA Graph for Decode (Normal Decode and Draft Decode) and Target Verify.
    - We don't support CUDA Graph for Extend and Draft Extend.
    - When server init, init_cuda_graph_state will be called first and then init_cuda_graph_capture will be called.
    - For each forward batch, init_replay_cuda_graph will be called first and then replay the graph.
    """

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)

        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        # extra metadata for handling speculative decoding topk > 1, extended draft decode and verify
        # 最大上下文长度（用于位置编码等限制）
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        # decode CUDA Graph 的元数据缓存（按 batch size 索引）
        self.decode_cuda_graph_metadata = {}
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.kv_cache_dtype_str = model_runner.server_args.kv_cache_dtype
        # 线性注意力的分块大小（从模型配置读取，默认 256）
        self.BLOCK = (
            model_runner.model_config.block
            if hasattr(model_runner.model_config, "block")
            else 256
        )
        # 构建 ALiBi（Attention with Linear Biases）衰减斜率张量
        total_num_heads = model_runner.model_config.hf_config.num_attention_heads
        num_hidden_layers = model_runner.model_config.hf_config.num_hidden_layers
        self.tp_slope = LightningAttentionBackend._build_slope_tensor(
            total_num_heads, num_hidden_layers, self.device
        )
        # 从模型配置中读取线性注意力后端类型（"minimax" 或 "seg_la"）
        self.linear_backend = getattr(
            model_runner.model_config.hf_config, "linear_backend", "seg_la"
        )
        logger.info(
            f"linear_backend for linear attention in hybrid_linear_backend: {self.linear_backend}"
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # 初始化混合（prefill+decode）批次的前向元数据
        metadata = self._forward_metadata(forward_batch)
        self.forward_metadata = BailingLinearMetadata.prepare_mixed(
            metadata.query_start_loc,
            metadata.mamba_cache_indices,
            forward_batch,
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        # CUDA Graph 捕获时初始化前向元数据（仅 decode 模式）
        metadata = self._capture_metadata(bs, req_pool_indices, forward_mode, spec_info)
        self.forward_metadata = BailingLinearMetadata.prepare_decode(
            metadata.query_start_loc, metadata.mamba_cache_indices, bs, seq_lens
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        # CUDA Graph 回放时初始化前向元数据（仅 decode 模式）
        metadata = self._replay_metadata(
            bs, req_pool_indices, forward_mode, spec_info, seq_lens_cpu
        )
        self.forward_metadata = BailingLinearMetadata.prepare_decode(
            metadata.query_start_loc, metadata.mamba_cache_indices, bs, seq_lens
        )

    @staticmethod
    def _build_slope_tensor(
        n_attention_heads: int, num_hidden_layers: int, device="cuda"
    ):
        # 构建 ALiBi 衰减斜率张量
        # ALiBi 为每个注意力头分配不同的指数衰减斜率，用于相对位置偏置
        def get_slopes(n):
            # 递归计算 n 个头的斜率（处理非 2 的幂次头数）
            def get_slopes_power_of_2(n):
                # 对 2 的幂次头数，生成从 2^{-(2^{-(log2n-3)})} 开始的等比数列
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                # n 是 2 的幂次，直接计算
                return get_slopes_power_of_2(n)
            else:
                # n 不是 2 的幂次，用最近的 2 的幂次补齐
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
                )

        # 生成所有头的斜率张量 [n_heads, 1, 1]
        slopes = torch.tensor(
            get_slopes(n_attention_heads), dtype=torch.float32
        ).reshape(n_attention_heads, 1, 1)
        from sglang.srt.layers.dp_attention import (
            get_attention_tp_rank,
            get_attention_tp_size,
        )

        # 按 TP（Tensor Parallel）分片：每个 TP rank 只处理部分头
        tp_heads = n_attention_heads // get_attention_tp_size()
        tp_rank = get_attention_tp_rank()
        # 每层使用不同的斜率缩放因子（随层深度线性衰减）
        if num_hidden_layers <= 1:
            slope_rate_list = [slopes * (1 + 1e-5)]
        else:
            slope_rate_list = [
                slopes * (1 - layer_id / (num_hidden_layers - 1) + 1e-5)
                for layer_id in range(num_hidden_layers)
            ]

        # 提取当前 TP rank 对应的斜率切片并移到目标设备
        tp_slope = [
            slope_rate_list[layer_id][tp_rank * tp_heads : (tp_rank + 1) * tp_heads]
            .contiguous()
            .to(device)
            for layer_id in range(num_hidden_layers)
        ]

        return tp_slope

    def _prefill_and_mix_infer(
        self,
        q,              # Query 张量 [total_tokens, num_heads, head_dim]
        k,              # Key 张量
        v,              # Value 张量
        kv_cache,       # KV 缓存（线性注意力的 SSM 状态池）
        state_indices_tensor,  # 每个请求的状态槽位索引
        forward_batch,  # 当前批次信息
        layer,          # 当前注意力层
        metadata,       # 前向元数据（含 prefill/decode 数量）
    ):
        # minimax 后端的混合推理：逐序列处理 prefill，然后处理所有 decode
        hidden = []
        for _prefill_idx in range(metadata.num_prefills):
            if _prefill_idx >= forward_batch.extend_start_loc.shape[0]:
                break
            if _prefill_idx >= state_indices_tensor.shape[0]:
                break

            # 获取当前 prefill 序列的起始和结束 token 位置
            _start = forward_batch.extend_start_loc[_prefill_idx]

            if _prefill_idx + 1 < forward_batch.extend_start_loc.shape[0]:
                _end = forward_batch.extend_start_loc[_prefill_idx + 1]
            else:
                # 最后一个 prefill 序列，边界由 extend_seq_lens 或总长度决定
                if (
                    forward_batch.extend_seq_lens is not None
                    and _prefill_idx < forward_batch.extend_seq_lens.shape[0]
                    and metadata.num_decodes > 0
                ):
                    seq_len = forward_batch.extend_seq_lens[_prefill_idx]
                    _end = _start + seq_len
                else:
                    _end = q.shape[0]

            slot_id = state_indices_tensor[_prefill_idx]
            # 提取当前序列的 QKV 并转置为 [heads, seq, dim] 格式
            qs = q[_start:_end].transpose(0, 1).contiguous()
            ks = k[_start:_end].transpose(0, 1).contiguous()
            vs = v[_start:_end].transpose(0, 1).contiguous()
            # 获取当前请求的 KV 缓存切片
            slice_layer_cache = kv_cache[slot_id, ...]
            # 调用 Bailing 线性 Kernel 的 prefix 前向（处理有历史状态的 prefill）
            out_slice = BailingLinearKernel.jit_linear_forward_prefix(
                qs,
                ks,
                vs,
                slice_layer_cache,
                self.tp_slope[layer.layer_id],  # 当前层的 ALiBi 斜率
                self.BLOCK,
                layer_idx=layer.layer_id,
            )
            hidden.append(out_slice.contiguous())
        # 处理 decode 请求（若有）
        if metadata.num_decodes > 0:
            hidden.append(
                self._decode_infer(
                    q, k, v, kv_cache, state_indices_tensor, metadata, layer
                )
            )

        if not hidden:
            return torch.empty((0, q.size(-1)), device=q.device, dtype=q.dtype)

        # 合并所有序列的输出
        hidden = torch.concat(hidden, dim=0).contiguous()
        return hidden

    def _decode_infer(self, q, k, v, kv_cache, state_indices_tensor, metadata, layer):
        # minimax 后端的 decode 推理（逐 token 线性注意力更新）
        num_prefill_tokens = metadata.num_prefill_tokens
        num_prefills = metadata.num_prefills
        # 跳过 prefill 部分，只处理 decode token（每个请求 1 个 token）
        q = q[num_prefill_tokens:].unsqueeze(2).contiguous()  # [bs, heads, 1, dim]
        k = k[num_prefill_tokens:].unsqueeze(2).contiguous()
        v = v[num_prefill_tokens:].unsqueeze(2).contiguous()
        slot_id = state_indices_tensor[num_prefills:]  # decode 请求的状态槽位索引

        assert slot_id.shape[0] == q.shape[0], (
            f"slot_id length {slot_id.shape[0]} does not match decode batch size {q.shape[0]}. "
            "This indicates a bug in the upstream logic that should be investigated."
        )
        # 调用 Triton 实现的线性 decode 前向（同时更新 SSM 状态）
        hidden = linear_decode_forward_triton(
            q, k, v, kv_cache, self.tp_slope[layer.layer_id], slot_id, 32
        )
        return hidden

    def _linear_attention_entry(
        self,
        q,              # Query 张量
        k,              # Key 张量
        v,              # Value 张量
        kv_cache,       # KV 缓存（SSM 状态池）
        state_indices_tensor,  # 状态槽位索引
        metadata,       # 前向元数据
        layer,          # 当前注意力层
        mask=None,           # 可选的自定义掩码
        temp_cache=None,     # 可选的临时缓存（投机解码用）
        intermediate_state_indices=None,  # 中间状态索引（投机解码用）
    ):
        # seg_la 后端：使用 SegLA（Segment Linear Attention）算法
        q_offsets = metadata.query_start_loc

        # 构建 SegLA 元数据（包含批次信息和掩码）
        seg_meta = SegLaMeta(
            batch_size=metadata.batch_size,
            q_offsets=metadata.query_start_loc,  # Query 的 CSR 格式偏移
            s_offsets=state_indices_tensor,       # 状态槽位索引
            q_lengths=q_offsets.diff(),           # 每个序列的 Query 长度
            s_scales=metadata.has_initial_states, # 是否有历史状态
            max_q_length=None,
            mask=mask,
        )
        # 调用 SegLA 前向函数（分段线性注意力，支持混合 prefill+decode）
        hidden = seg_la_fwd(
            q=q,
            k=k,
            v=v,
            s=kv_cache,                           # SSM 状态（输入/输出）
            decay_scales=self.tp_slope[layer.layer_id],  # ALiBi 衰减斜率
            meta=seg_meta,
            caches=temp_cache,                    # 临时缓存（投机解码中间状态）
            cache_indices=intermediate_state_indices,  # 中间状态索引
            decouple=True,                        # 解耦模式（分离 prefill 和 decode）
        )
        return hidden

    def forward_extend(
        self,
        q: torch.Tensor,  # Query 张量
        k: torch.Tensor,  # Key 张量
        v: torch.Tensor,  # Value 张量
        layer: RadixAttention,  # 注意力层
        forward_batch: ForwardBatch,  # 当前批次信息
        save_kv_cache=True,
        **kwargs,
    ):
        q_rope = kwargs["q_rope"] if "q_rope" in kwargs else None
        k_rope = kwargs["k_rope"] if "k_rope" in kwargs else None
        layer_id = layer.layer_id if layer else kwargs["layer_id"]

        metadata = self.forward_metadata

        # 若使用 KV cache 量化，将 Q 转换为对应数据类型
        if self.kv_cache_dtype_str != "auto" and layer.k_scale is not None:
            q = q.to(self.kv_cache_dtype)

        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices
        # 获取当前层的 SSM 状态（线性注意力的 KV 状态矩阵）
        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        ssm_states = mamba_cache_params.temporal
        if self.linear_backend == "minimax":
            # minimax 后端：混合 prefill+decode 推理
            o = self._prefill_and_mix_infer(
                q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                k,
                v,
                ssm_states,
                cache_indices,
                forward_batch,
                layer,
                metadata,
            )
        elif self.linear_backend == "seg_la":
            # seg_la 后端：SegLA 分段线性注意力
            # 投机解码目标验证时需要保存中间状态（用于并行验证）
            intermediate_state_indices = (
                torch.arange(
                    cache_indices.shape[0],
                    dtype=torch.int32,
                    device=cache_indices.device,
                )
                if forward_batch.forward_mode.is_target_verify()
                else None
            )
            o = self._linear_attention_entry(
                q,
                k,
                v,
                ssm_states,
                cache_indices,
                metadata,
                layer,
                temp_cache=(
                    mamba_cache_params.intermediate_ssm  # 中间状态缓冲区（仅验证模式）
                    if forward_batch.forward_mode.is_target_verify()
                    else None
                ),
                intermediate_state_indices=intermediate_state_indices,
            )
        else:
            raise ValueError(
                f"linear backend: {self.linear_backend} is not support for now"
            )
        # 将输出 reshape 为 [total_tokens, head_num * head_dim]
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,  # Query 张量
        k: torch.Tensor,  # Key 张量
        v: torch.Tensor,  # Value 张量
        layer: RadixAttention,  # 注意力层
        forward_batch: ForwardBatch,  # 当前批次信息
        save_kv_cache=True,
        **kwargs,
    ) -> torch.Tensor:
        q_rope = kwargs["q_rope"] if "q_rope" in kwargs else None
        k_rope = kwargs["k_rope"] if "k_rope" in kwargs else None
        layer_id = layer.layer_id if layer else kwargs["layer_id"]

        # Use precomputed metadata across all layers
        # 使用预计算的前向元数据（所有层共用）
        metadata = self.forward_metadata

        # 若使用 KV cache 量化，将 Q 转换为对应数据类型
        if self.kv_cache_dtype_str != "auto":
            q = q.to(self.kv_cache_dtype)

        # Do linear attention
        # 执行线性注意力前向计算
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices
        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        ssm_states = mamba_cache_params.temporal  # SSM 隐状态
        if self.linear_backend == "minimax":
            # minimax 后端的纯 decode 路径
            o = self._decode_infer(q, k, v, ssm_states, cache_indices, metadata, layer)
        elif self.linear_backend == "seg_la":
            # seg_la 后端的 decode 路径（通过通用入口函数）
            o = self._linear_attention_entry(
                q, k, v, ssm_states, cache_indices, metadata, layer
            )
        else:
            raise ValueError(
                f"linear backend: {self.linear_backend} is not support for now"
            )
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)
