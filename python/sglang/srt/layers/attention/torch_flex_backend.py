# 允许注解中使用前向引用
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
# 导入 PyTorch Flex Attention：支持自定义块级掩码的新一代注意力算子
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

# 导入注意力后端基类
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
# 导入注意力类型枚举
from sglang.srt.layers.radix_attention import AttentionType
# 导入前向批次信息
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    # 仅类型检查时导入，避免循环依赖
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


class TorchFlexAttnBackend(AttentionBackend):
    # 基于 PyTorch Flex Attention 的注意力后端，支持动态块级掩码（适合研究和可扩展性）
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device
        # 使用 torch.compile 编译 flex_attention 以提升执行效率，dynamic=True 支持动态形状
        self.flex_attention = torch.compile(flex_attention, dynamic=True)
        # 增大 dynamo 缓存上限，避免因形状变化导致频繁重新编译
        torch._dynamo.config.cache_size_limit = 1024
        torch._dynamo.config.accumulated_cache_size_limit = 1024

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        # TODO: find a more elegant way to save memory
        # Currently maintain the same memory as torch_native_backend
        # 清理 GPU 显存碎片（Flex Attention 需要额外显存来存储块掩码）
        torch.cuda.empty_cache()

        # Provide two block_mask Lists per seq_idx for lower latency, later will support per layer level mask generation
        # 为每条序列预生成块掩码列表，降低前向延迟（当前为批次级别，未来计划支持层级别）
        self.extend_block_masks = []
        self.decode_block_masks = []

        if forward_batch.forward_mode.is_extend():
            # prefill/extend 阶段：为每条序列生成因果注意力块掩码
            for seq_idx in range(forward_batch.seq_lens.shape[0]):
                seq_len_kv = forward_batch.seq_lens[seq_idx]
                seq_len_q = seq_len_kv  # extend 阶段 Q 长度等于完整序列长度
                self.extend_block_masks.append(
                    create_block_mask(
                        self._causal_mask,
                        None,
                        None,
                        seq_len_q,
                        seq_len_kv,
                        device=self.device,
                        _compile=False,
                    )
                )

        elif forward_batch.forward_mode.is_decode():
            # decode 阶段：为每条序列生成解码掩码（Q 长度为 1）
            for seq_idx in range(forward_batch.seq_lens.shape[0]):
                seq_len_q = 1  # decode 阶段每步只有 1 个新 token
                seq_len_kv = forward_batch.seq_lens[seq_idx]

                self.decode_block_masks.append(
                    create_block_mask(
                        self._decode_mask,
                        None,
                        None,
                        seq_len_q,
                        seq_len_kv,
                        device=self.device,
                        _compile=False,
                    )
                )

    def _causal_mask(self, b, h, q_idx, kv_idx):
        # 因果掩码：Q 位置 >= KV 位置时可见（下三角掩码）
        return q_idx >= kv_idx

    def _decode_mask(self, b, h, q_idx, kv_idx):
        # decode 掩码：Q 位置 <= KV 位置时可见（decode 阶段 Q=1，访问全部历史 KV）
        return q_idx <= kv_idx

    def _run_flex_forward_extend(
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
        """Run the extend forward by using torch flex attention op.

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
        # extend/prefill 阶段：使用 Flex Attention 逐序列计算注意力

        # 断言序列数量与前缀/扩展长度数组大小一致
        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0

        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.
            # 逐序列循环（性能待优化）
            extend_seq_len_q = extend_seq_lens[seq_idx]    # 当前序列 extend 部分的 Q 长度
            prefill_seq_len_q = extend_prefix_lens[seq_idx]  # 已缓存的前缀长度

            seq_len_kv = seq_lens[seq_idx]  # KV 缓存中该序列的总 token 数
            end_q = start_q + extend_seq_len_q
            end_kv = start_kv + seq_len_kv

            # 提取当前序列的 Query（仅 extend 部分）
            per_req_query = query[:, start_q:end_q, :]
            # 创建与完整 KV 序列等长的冗余 Query 张量（前缀位置填零）
            per_req_query_redundant = torch.empty(
                (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
                dtype=per_req_query.dtype,
                device=per_req_query.device,
            )

            # 将实际 Query 填入冗余张量的非前缀部分
            per_req_query_redundant[:, prefill_seq_len_q:, :] = per_req_query

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            # 通过 req_to_token 映射从 KV 缓存中取 Key/Value
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            if not causal:
                # 当前实现仅支持因果注意力
                raise NotImplementedError("Non-causal mode is not yet implemented.")

            # 调用编译后的 Flex Attention，使用预生成的因果块掩码
            per_req_out_redundant = (
                self.flex_attention(
                    per_req_query_redundant.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    block_mask=self.extend_block_masks[seq_idx],  # 使用对应序列的预生成掩码
                    scale=scaling,
                    enable_gqa=enable_gqa,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            # 仅保留 extend 部分的输出（丢弃前缀冗余部分）
            output[start_q:end_q, :, :] = per_req_out_redundant[
                prefill_seq_len_q:, :, :
            ]
            start_q, start_kv = end_q, end_kv
        return output

    def _run_flex_forward_decode(
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
        """Run the decode forward by using torch flex attention op.

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
        # decode 阶段：使用 Flex Attention 逐序列进行单步注意力计算

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.

            seq_len_q = 1  # decode 每步只处理 1 个新 token
            seq_len_kv = seq_lens[seq_idx]  # 完整 KV 序列长度
            end_q = start_q + seq_len_q
            end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            # 从 KV 缓存中取出该序列的 Key/Value
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            # 调用编译后的 Flex Attention，使用预生成的 decode 块掩码
            per_req_out = (
                self.flex_attention(
                    per_req_query.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    block_mask=self.decode_block_masks[seq_idx],
                    scale=scaling,
                    enable_gqa=enable_gqa,
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
        # prefill/extend 阶段前向：先写 KV 缓存，再调用 Flex Attention
        if layer.qk_head_dim != layer.v_head_dim:
            # QK 和 V head_dim 不同时，分配独立输出缓冲区
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if save_kv_cache:
            # 将当前步骤的 K、V 写入 KV 缓存
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        # 判断是否启用 GQA
        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        causal = True  # 默认因果注意力
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            # Flex Attention 后端暂不支持非因果注意力
            raise NotImplementedError(
                "TorchFlexAttnBackend does not support non-causal attention for now."
            )

        self._run_flex_forward_extend(
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
        # 修复 torch.compile 下 rotary_emb 输出可能为 3D 的问题
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if save_kv_cache:
            # 写入当前 token 的 KV 到缓存
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        # 判断是否启用 GQA
        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num
        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        self._run_flex_forward_decode(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=False,  # decode 阶段通过 decode_mask 控制，causal 参数不使用
        )

        return o

    def support_triton(self):
        # Flex Attention 后端不依赖 Triton 内核
        return False
