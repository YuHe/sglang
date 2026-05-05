# 允许在注解中使用前向引用
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

# 导入注意力后端基类
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
# 导入前向批次信息
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    # 仅类型检查时导入，避免循环依赖
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


class IntelAMXAttnBackend(AttentionBackend):
    # Intel AMX（Advanced Matrix Extensions）注意力后端，专为 CPU 上的高效矩阵运算优化
    def __init__(self, model_runner: ModelRunner):
        import sgl_kernel  # noqa: F401  # 导入 sgl_kernel 以注册 CPU 注意力算子

        super().__init__()
        self.forward_metadata = None  # 前向元数据，在 init_forward_metadata 中初始化
        self.device = model_runner.device  # 记录运行设备（通常为 CPU）

        # 计算张量并行切分后的本地注意力头数
        self.num_head = (
            model_runner.model_config.num_attention_heads // model_runner.tp_size
        )

        # [NB]: `layer_id` set to 0 for qwen3-next models, as not all attn layers require kv pool
        # using "full_attention_layer_id_mapping" to map which layer needs kv pool
        layer_id = 0  # 默认使用第 0 层
        if hasattr(model_runner.token_to_kv_pool, "full_attention_layer_id_mapping"):
            # 若模型有完整注意力层映射（如 qwen3-next），取第一个需要 KV 池的层 ID
            layer_id = [*model_runner.token_to_kv_pool.full_attention_layer_id_mapping][
                0
            ]
        # 通过 value buffer 的最后一维获取 v_head_dim
        self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(
            layer_id
        ).shape[-1]
        # 绑定 CPU 解码和 extend 注意力算子
        self.decode_attention_fwd = torch.ops.sgl_kernel.decode_attention_cpu
        self.extend_attention_fwd = torch.ops.sgl_kernel.extend_attention_cpu

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        # 初始化前向元数据：分配注意力 logits 缓冲区，并计算最大 extend 长度

        bs = forward_batch.batch_size
        # 分配 attn_logits 缓冲区：[bs, num_head, num_kv_splits, v_head_dim+1]
        attn_logits = torch.zeros(
            (
                bs,
                self.num_head,
                8,  # self.num_kv_splits,  # KV 分片数（当前固定为 8）
                self.v_head_dim + 1,  # +1 用于存储 lse（log-sum-exp）
            ),
            dtype=torch.float32,
            device=self.device,
        )
        if forward_batch.forward_mode.is_decode_or_idle():
            # decode/idle 模式不需要 extend 长度
            max_extend_len = None
        else:
            # prefill/extend 模式：计算批次中最长的 extend 序列长度
            max_extend_len = torch.max(forward_batch.extend_seq_lens).item()
        self.forward_metadata = (attn_logits, max_extend_len)

    def get_cpu_graph_seq_len_fill_value(self):
        # CPU Graph 中序列长度的填充值为 1（避免除零等问题）
        return 1

    def init_forward_metadata_capture_cpu_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens,
        forward_mode,
        spec_info,
    ):
        # CPU Graph 捕获阶段：分配固定形状的 attn_logits 缓冲，extend 长度设为 None
        attn_logits = torch.zeros(
            (
                bs,
                self.num_head,
                8,  # self.num_kv_splits,
                self.v_head_dim + 1,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        max_extend_len = None
        self.forward_metadata = (attn_logits, max_extend_len)

    def init_cpu_graph_state(self, max_bs: int, max_num_tokens: int):
        # CPU Graph 全局状态初始化（当前无需额外操作）
        pass

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # prefill/extend 阶段：调用 CPU extend_attention 内核完成注意力计算
        if layer.qk_head_dim != layer.v_head_dim:
            # QK 和 V 的 head_dim 不一致时，需单独分配输出缓冲区
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if save_kv_cache:
            # 将当前步骤的 K、V 写入 KV 缓存池
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        _, max_extend_len = self.forward_metadata  # 取出最大 extend 长度

        # 调用 CPU extend 注意力算子（含 PagedAttention 逻辑）
        self.extend_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k,
            v,
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.extend_seq_lens,
            forward_batch.extend_start_loc,
            max_extend_len,
            layer.scaling,       # 注意力缩放因子（1/sqrt(head_dim)）
            layer.logit_cap,     # logit 截断上限（防止数值溢出）
        )
        return o

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # decode 阶段：调用 CPU decode_attention 内核完成逐 token 注意力计算
        attn_logits, _ = self.forward_metadata  # 取出注意力 logits 缓冲区

        # 将 Q 重塑为 [num_tokens, num_heads * head_dim] 的二维张量
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            # QK 和 V head_dim 不同时，分配独立输出缓冲区
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        # 调用 CPU decode 注意力算子（使用 PagedKV 缓存）
        self.decode_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            k,
            v,
            forward_batch.out_cache_loc,     # 当前步 token 对应的 KV 缓存槽位
            attn_logits,                     # 中间 logits 缓冲区（含 lse）
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            layer.scaling,
            layer.logit_cap,
        )

        return o

    def support_triton(self):
        # Intel AMX 后端仅支持 CPU 算子，不支持 Triton（GPU）内核
        return False
