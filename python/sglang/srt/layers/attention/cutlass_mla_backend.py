from __future__ import annotations

"""
Support attention backend for Cutlass MLA.

"""
# Cutlass MLA（Multi-head Latent Attention）注意力后端支持模块

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch
import triton

# 导入 FlashInfer MLA 后端作为父类（prefill 阶段沿用 FlashInfer 实现）
from sglang.srt.layers.attention.flashinfer_mla_backend import FlashInferMLAAttnBackend
# 导入用于构建 FlashMLA KV 索引的 Triton 内核
from sglang.srt.layers.attention.utils import create_flashmla_kv_indices_triton
# 导入注意力张量并行尺寸获取函数
from sglang.srt.layers.dp_attention import get_attention_tp_size
# 导入前向批次信息和前向模式枚举
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
# 导入 CUDA 设备检测工具
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    # 仅类型检查时导入，避免运行时循环引用
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

# 一次性检测是否为 CUDA 设备
_is_cuda = is_cuda()
if _is_cuda:
    # 仅在 CUDA 设备上导入 Cutlass MLA 解码内核
    from sgl_kernel import cutlass_mla_decode, cutlass_mla_get_workspace_size


# Cutlass MLA only supports pagesize=128
# Cutlass MLA 内核固定页大小为 128 tokens
PAGE_SIZE = 128


@dataclass
class CutlassMLADecodeMetadata:
    # Cutlass MLA decode 阶段所需的元数据：显存工作区和块级 KV 索引
    workspace: Optional[torch.Tensor] = None         # GPU 工作区缓冲区
    block_kv_indices: Optional[torch.Tensor] = None  # 每条序列对应的 KV 块索引表

    def __init__(
        self,
        workspace: Optional[torch.Tensor] = None,
        block_kv_indices: Optional[torch.Tensor] = None,
    ):
        self.workspace = workspace
        self.block_kv_indices = block_kv_indices


class CutlassMLABackend(FlashInferMLAAttnBackend):
    """Cutlass attention kernels."""
    # Cutlass MLA 注意力后端：decode 阶段使用 Cutlass 高效内核，prefill 阶段沿用 FlashInfer

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
    ):
        # 初始化父类 FlashInfer MLA 后端
        super().__init__(
            model_runner, skip_prefill, kv_indptr_buf, kv_last_page_len_buf
        )

        # 计算张量并行切分后的本地 Q 头数
        self.num_q_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        # 获取 KV 头数（考虑 GQA 等配置）
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.req_to_token = model_runner.req_to_token_pool.req_to_token  # 请求到 token 的映射表
        # 本地 attention head 数（与 num_q_heads 相同）
        self.num_local_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.forward_metadata: Union[CutlassMLADecodeMetadata] = None  # 前向元数据
        # MLA 特有参数
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank       # KV LoRA 秩
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim  # 非 RoPE 部分的 QK 维度
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim  # RoPE 部分的 QK 维度
        self.v_head_dim = model_runner.model_config.v_head_dim             # Value head 维度
        self.scaling = model_runner.model_config.scaling                   # 注意力缩放因子
        self.data_type = model_runner.kv_cache_dtype                       # KV 缓存数据类型
        self.q_data_type = model_runner.dtype                              # Query 数据类型
        # KV 缓存维度 = kv_lora_rank + qk_rope_head_dim（压缩的 KV + RoPE 部分）
        self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # 初始化前向元数据：decode 阶段构建 Cutlass KV 索引，其余阶段委托给父类

        bs = forward_batch.batch_size
        spec_info = forward_batch.spec_info
        if forward_batch.forward_mode.is_decode_or_idle():
            if spec_info is None:
                # 标准 decode：计算按页对齐的最大序列长度
                max_seqlen_pad = triton.cdiv(
                    forward_batch.seq_lens_cpu.max().item(), PAGE_SIZE
                )
                # 初始化块级 KV 索引表，填充值 -1 表示无效块
                block_kv_indices = torch.full(
                    (bs, max_seqlen_pad),
                    -1,
                    dtype=torch.int32,
                    device=forward_batch.seq_lens.device,
                )
                # 使用 Triton 内核批量填充 KV 块索引
                create_flashmla_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    None,
                    block_kv_indices,
                    self.req_to_token.stride(0),
                    max_seqlen_pad,
                    PAGED_SIZE=PAGE_SIZE,
                )
                # 计算并分配 Cutlass MLA 所需的 GPU 工作区
                workspace_size = cutlass_mla_get_workspace_size(
                    max_seqlen_pad * PAGE_SIZE, bs, num_kv_splits=1
                )
                workspace = torch.empty(
                    workspace_size, device="cuda", dtype=torch.uint8
                )
                self.forward_metadata = CutlassMLADecodeMetadata(
                    workspace,
                    block_kv_indices,
                )
            else:
                # 推测解码场景：使用父类 FlashInfer 初始化
                super().init_forward_metadata(forward_batch)
        else:
            # prefill/extend 阶段：使用父类 FlashInfer 初始化
            super().init_forward_metadata(forward_batch)

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        block_kv_indices: Optional[torch.Tensor] = None,
    ):
        # 初始化 CUDA Graph 所需的固定缓冲区：KV 索引和工作区
        if block_kv_indices is None:
            # 默认创建最大尺寸的 KV 索引缓冲区（填充为 1 以满足 Cutlass 的初始化要求）
            cuda_graph_kv_indices = torch.full(
                (max_bs, (self.max_context_len + PAGE_SIZE) // PAGE_SIZE),
                1,
                dtype=torch.int32,
                device="cuda",
            )
        else:
            cuda_graph_kv_indices = block_kv_indices

        # 根据最大 KV 块数计算工作区大小
        workspace_size = cutlass_mla_get_workspace_size(
            cuda_graph_kv_indices.shape[1] * PAGE_SIZE, max_bs, num_kv_splits=1
        )
        # 预分配 CUDA Graph 专用工作区（避免 Graph 捕获后的动态内存分配）
        self.cuda_graph_mla_workspace = torch.empty(
            workspace_size, device="cuda", dtype=torch.uint8
        )
        self.cuda_graph_kv_indices = cuda_graph_kv_indices

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
        # CUDA Graph 捕获阶段：decode 模式填充 KV 索引，其余委托给父类
        if forward_mode.is_decode_or_idle():
            if spec_info is None:
                max_seqlen_pad = self.cuda_graph_kv_indices.shape[1]  # 使用预分配缓冲区的列数

                # 填充预分配的 CUDA Graph KV 索引缓冲区
                create_flashmla_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices,
                    seq_lens,
                    None,
                    self.cuda_graph_kv_indices,
                    self.req_to_token.stride(0),
                    self.cuda_graph_kv_indices.stride(0),
                    PAGED_SIZE=PAGE_SIZE,
                )
                # 绑定 CUDA Graph 固定缓冲区（切片到当前 bs）
                self.forward_metadata = CutlassMLADecodeMetadata(
                    self.cuda_graph_mla_workspace,
                    self.cuda_graph_kv_indices[:bs, :max_seqlen_pad],
                )
        else:
            super().init_forward_metadata_capture_cuda_graph(
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
        # CUDA Graph 回放阶段：更新 KV 索引缓冲区中的动态内容
        if forward_mode.is_decode_or_idle():
            assert seq_lens_cpu is not None  # 回放时必须提供 CPU 序列长度
            seq_lens = seq_lens[:bs]  # 截取当前批次大小的序列长度

            # 重新填充 KV 块索引（seq_lens 已变化）
            create_flashmla_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices[:bs],
                seq_lens,
                None,
                self.cuda_graph_kv_indices,
                self.req_to_token.stride(0),
                self.cuda_graph_kv_indices.stride(0),
                PAGED_SIZE=PAGE_SIZE,
            )
        else:
            super().init_forward_metadata_replay_cuda_graph(
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
        # CUDA Graph 中序列长度的填充值为 1
        return 1

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,   # RoPE 部分的 Query（MLA 专用）
        k_rope: Optional[torch.Tensor] = None,   # RoPE 部分的 Key（MLA 专用）
    ):
        # decode 阶段：使用 Cutlass MLA 高效内核完成注意力计算
        cache_loc = forward_batch.out_cache_loc

        if k is not None:
            assert v is not None
            if save_kv_cache:
                if k_rope is not None:
                    # MLA 模式：分别存储压缩 KV 和 RoPE Key
                    forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope,
                    )
                else:
                    # 标准 KV 存储
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        v,
                    )

        # Reshape inputs
        # 拆分 Q 为 nope（非 RoPE）和 rope（RoPE）两部分
        if q_rope is not None:
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
        else:
            # Q 已融合：前 v_head_dim 为 nope 部分，后续为 rope 部分
            reshaped_q = q.view(-1, layer.tp_q_head_num, layer.head_dim)
            q_nope = reshaped_q[:, :, : layer.v_head_dim]
            q_rope = reshaped_q[:, :, layer.v_head_dim :]

        # 将 Q 转换为模型精度（避免 fp8 等低精度 KV 缓存影响计算）
        q_nope = q_nope.to(self.q_data_type)
        q_rope = q_rope.to(self.q_data_type)

        # 获取完整 KV 缓存（包含压缩 C 和 RoPE k_pe 部分）
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)

        # 调用 Cutlass MLA decode 内核
        o = cutlass_mla_decode(
            q_nope=q_nope,
            q_pe=q_rope,
            kv_c_and_k_pe_cache=k_cache.view(-1, PAGE_SIZE, self.kv_cache_dim),  # 将 KV 缓存重排为页格式
            seq_lens=forward_batch.seq_lens.to(torch.int32),
            page_table=self.forward_metadata.block_kv_indices,  # 块级 KV 索引表
            workspace=self.forward_metadata.workspace,           # GPU 工作区
            sm_scale=layer.scaling,                              # 注意力缩放因子
            num_kv_splits=1,
        )

        # 将输出从 [bs, num_heads, v_head_dim] 展平为 [bs, num_heads * v_head_dim]
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)
