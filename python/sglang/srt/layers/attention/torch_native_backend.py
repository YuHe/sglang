# 允许注解中使用前向引用
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
# 导入 PyTorch 原生缩放点积注意力算子
from torch.nn.functional import scaled_dot_product_attention

# 导入注意力后端基类
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
# 导入注意力类型枚举（如交叉注意力、编码器注意力等）
from sglang.srt.layers.radix_attention import AttentionType
# 导入前向批次信息
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    # 仅类型检查时导入，避免循环依赖
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


class TorchNativeAttnBackend(AttentionBackend):
    # 基于 PyTorch 原生 SDPA 的注意力后端，不依赖 FlashAttention/Triton 等特殊内核
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None  # 无需额外元数据
        self.device = model_runner.device  # 记录运行设备

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        # 原生后端无需额外初始化元数据
        pass

    def _run_sdpa_forward_extend(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        """Run the extend forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            extend_prefix_lens: [num_seqs]
            extend_seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """
        # extend/prefill 阶段：逐序列调用 PyTorch 原生 SDPA，结合 PagedKV 缓存实现注意力计算

        # 断言序列数量一致性
        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        # 将 token 维度从第 0 轴移至倒数第 2 轴，以满足 SDPA 的输入格式要求
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.
            # 逐序列处理（性能待优化，目前按序列循环）

            extend_seq_len_q = extend_seq_lens[seq_idx]    # 当前序列 extend 部分的 token 数
            prefill_seq_len_q = extend_prefix_lens[seq_idx]  # 当前序列已有前缀的 token 数

            seq_len_kv = seq_lens[seq_idx]  # KV 缓存中该序列的总 token 数
            end_q = start_q + extend_seq_len_q
            end_kv = start_kv + seq_len_kv

            # 提取当前序列对应的 Query（仅 extend 部分）
            per_req_query = query[:, start_q:end_q, :]
            # 创建一个包含前缀占位的冗余 Query 张量（前缀部分为零）
            per_req_query_redudant = torch.empty(
                (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
                dtype=per_req_query.dtype,
                device=per_req_query.device,
            )

            # 将实际 Query 填入冗余张量的非前缀部分（前缀位置保持零以屏蔽）
            per_req_query_redudant[:, prefill_seq_len_q:, :] = per_req_query

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            # 通过 req_to_token 映射，从 KV 缓存中取出该序列的 Key 和 Value
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                # scaled_dot_product_attention() expects query, key, and value to have the same dtype
                # 确保 Q/K/V 数据类型一致，KV 缓存可能为 fp8 等低精度类型
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

            # 调用 SDPA 计算注意力，结果形状为 [num_heads, seq_len_kv, head_size]
            per_req_out_redudant = (
                scaled_dot_product_attention(
                    per_req_query_redudant.unsqueeze(0),  # 添加 batch 维度
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)                           # 去除 batch 维度
                .movedim(query.dim() - 2, 0)          # 将 head 维度移回第 0 轴
            )
            # 仅保留 extend 部分的输出（丢弃前缀位置的冗余输出）
            output[start_q:end_q, :, :] = per_req_out_redudant[prefill_seq_len_q:, :, :]
            start_q, start_kv = end_q, end_kv
        return output

    def _run_sdpa_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        """Run the decode forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """
        # decode 阶段：逐序列调用 PyTorch 原生 SDPA，每步只有 1 个新 Query token

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.
            # 逐序列处理（性能待优化）

            seq_len_q = 1  # decode 阶段每次只有 1 个新 token
            seq_len_kv = seq_lens[seq_idx]  # KV 缓存中该序列的总 token 数
            end_q = start_q + seq_len_q
            end_kv = start_kv + seq_len_kv

            # 提取当前序列的单个 Query token
            per_req_query = query[:, start_q:end_q, :]

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            # 通过 req_to_token 映射从 KV 缓存中取 Key/Value
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                # scaled_dot_product_attention() expects query, key, and value to have the same dtype
                # 确保 Q/K/V 数据类型一致
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

            # 调用 SDPA 完成单步注意力计算
            per_req_out = (
                scaled_dot_product_attention(
                    per_req_query.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out
            start_q, start_kv = end_q, end_kv

        return output

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # prefill/extend 阶段前向：先写 KV 缓存，再调用 SDPA extend 逻辑
        if layer.qk_head_dim != layer.v_head_dim:
            # QK 和 V head_dim 不同时，分配独立输出缓冲区
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if layer.is_cross_attention:
            # 交叉注意力使用编码器输出的缓存位置
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        if save_kv_cache:
            # 将当前步骤的 K、V 写入 KV 缓存池
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        # 判断是否使用 GQA（分组查询注意力）：Q 头数多于 K 头数时启用
        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        causal = True  # 默认为因果注意力（decoder）
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            # 交叉注意力和纯编码器注意力不需要因果掩码
            causal = False

        self._run_sdpa_forward_extend(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.extend_prefix_lens,
            forward_batch.extend_seq_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=causal,
        )
        return o

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        # torch.compile 中 rotary_emb 可能产生 3D 输出，此处强制 reshape 为 2D
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            # QK 和 V head_dim 不同时，分配独立输出缓冲区
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if layer.is_cross_attention:
            # 交叉注意力使用编码器输出缓存位置
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        if save_kv_cache:
            # 写入当前 token 的 KV 到缓存
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        # 判断是否启用 GQA
        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        self._run_sdpa_forward_decode(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=False,  # decode 阶段不使用因果掩码（已通过 seq_len_q=1 隐式处理）
        )

        return o

    def support_triton(self):
        # 原生 SDPA 后端不依赖 Triton 内核
        return False
