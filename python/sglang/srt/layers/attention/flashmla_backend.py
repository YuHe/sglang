"""
Support attention backend for FlashMLA.
"""
# FlashMLA（Flash Multi-head Latent Attention）注意力后端支持模块

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

import torch
import triton
# 导入 FlashMLA 内核：flash_mla_with_kvcache 执行注意力计算，get_mla_metadata 计算调度元数据
from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata

# 导入 FlashInfer MLA 后端作为父类（prefill 阶段沿用 FlashInfer 实现）
from sglang.srt.layers.attention.flashinfer_mla_backend import FlashInferMLAAttnBackend
# 导入用于构建 FlashMLA KV 索引的 Triton 内核
from sglang.srt.layers.attention.utils import create_flashmla_kv_indices_triton
# 导入注意力张量并行尺寸获取函数
from sglang.srt.layers.dp_attention import get_attention_tp_size
# 导入 FP8 量化内核
from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant
# 导入前向批次信息和前向模式枚举
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    # 仅类型检查时导入，避免循环依赖
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput


# FlashMLA 固定页大小为 64 tokens（与 Cutlass MLA 的 128 不同）
PAGE_SIZE = 64


@dataclass
class FlashMLADecodeMetadata:
    # FlashMLA decode 阶段所需的元数据：调度元数据、SM 分片数和块级 KV 索引
    flashmla_metadata: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  # tile 调度信息
    num_splits: Optional[torch.Tensor] = None       # 每条序列的 SM 分片数
    block_kv_indices: Optional[torch.Tensor] = None # 每条序列对应的 KV 块索引表

    def __init__(
        self,
        flashmla_metadata: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_splits: Optional[torch.Tensor] = None,
        block_kv_indices: Optional[torch.Tensor] = None,
    ):
        self.flashmla_metadata = flashmla_metadata
        self.num_splits = num_splits
        self.block_kv_indices = block_kv_indices


class FlashMLABackend(FlashInferMLAAttnBackend):
    # FlashMLA 注意力后端：decode/verify 阶段使用 FlashMLA 高效内核，prefill 阶段沿用 FlashInfer
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
        self.req_to_token = model_runner.req_to_token_pool.req_to_token  # 请求到 token 的映射表
        # 本地 attention head 数（与 num_q_heads 相同）
        self.num_local_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.forward_metadata: Union[FlashMLADecodeMetadata] = None  # 前向元数据
        # MLA 特有参数
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank       # KV LoRA 秩（压缩维度）
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim  # 非 RoPE 部分维度
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim  # RoPE 部分维度
        self.v_head_dim = model_runner.model_config.v_head_dim             # Value head 维度
        self.scaling = model_runner.model_config.scaling                   # 注意力缩放因子
        self.data_type = model_runner.kv_cache_dtype                       # KV 缓存数据类型
        self.q_data_type = model_runner.dtype                              # Query 数据类型
        # KV 缓存维度 = kv_lora_rank + qk_rope_head_dim
        self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim
        # 检测是否使用 FP8 KV 缓存（需要特殊量化处理）
        self.is_fp8_kvcache = self.data_type in {
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        }

        # 推测解码草稿步数
        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens

        # CUDA Graph 专用缓冲区（在 init_cuda_graph_state 中初始化）
        self.cuda_graph_kv_indices = None
        self.cuda_graph_mla_metadata = None
        self.cuda_graph_num_splits = None
        self.cuda_graph_mla_metadata_view = None
        self.cuda_graph_num_splits_view = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # 初始化前向元数据：根据前向模式选择 decode/verify/prefill 分支
        bs = forward_batch.batch_size
        if forward_batch.forward_mode.is_decode_or_idle():
            # 标准 decode：计算 KV 块索引和 MLA 调度元数据
            max_seqlen_pad = triton.cdiv(
                forward_batch.seq_lens_cpu.max().item(), PAGE_SIZE
            )
            # 初始化块级 KV 索引表（-1 表示无效块）
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
            )
            # 获取 FlashMLA tile 调度元数据和每序列的 SM 分片数
            mla_metadata, num_splits = get_mla_metadata(
                forward_batch.seq_lens.to(torch.int32),
                self.num_q_heads,
                1,
                is_fp8_kvcache=self.is_fp8_kvcache,
            )
            self.forward_metadata = FlashMLADecodeMetadata(
                mla_metadata,
                num_splits,
                block_kv_indices,
            )
        elif forward_batch.forward_mode.is_target_verify():
            # 推测解码 verify 阶段：序列长度需加上草稿 token 数
            seq_lens_cpu = forward_batch.seq_lens_cpu + self.num_draft_tokens
            seq_lens = forward_batch.seq_lens + self.num_draft_tokens

            max_seqlen_pad = triton.cdiv(seq_lens_cpu.max().item(), PAGE_SIZE)
            block_kv_indices = torch.full(
                (bs, max_seqlen_pad),
                -1,
                dtype=torch.int32,
                device=seq_lens.device,
            )
            # 使用包含草稿 token 的序列长度填充 KV 索引
            create_flashmla_kv_indices_triton[(bs,)](
                self.req_to_token,
                forward_batch.req_pool_indices,
                seq_lens,
                None,
                block_kv_indices,
                self.req_to_token.stride(0),
                max_seqlen_pad,
            )
            # verify 阶段 Q 头数为 num_draft_tokens * num_q_heads（多个草稿 token 同时 verify）
            mla_metadata, num_splits = get_mla_metadata(
                seq_lens.to(torch.int32),
                self.num_draft_tokens * self.num_q_heads,
                1,
                is_fp8_kvcache=self.is_fp8_kvcache,
            )
            self.forward_metadata = FlashMLADecodeMetadata(
                mla_metadata,
                num_splits,
                block_kv_indices,
            )
        else:
            # prefill/extend 阶段：使用父类 FlashInfer 初始化
            super().init_forward_metadata(forward_batch)

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        block_kv_indices: Optional[torch.Tensor] = None,
    ):
        # 初始化 CUDA Graph 所需的固定缓冲区：KV 索引、MLA 元数据和 SM 分片数
        if block_kv_indices is None:
            # 默认创建最大尺寸的 KV 索引缓冲区
            self.cuda_graph_kv_indices = torch.full(
                (max_bs, (self.max_context_len + PAGE_SIZE) // PAGE_SIZE),
                1,
                dtype=torch.int32,
                device="cuda",
            )
        else:
            self.cuda_graph_kv_indices = block_kv_indices

        # 获取当前设备的 SM（流式多处理器）数量，作为最大分片数上限
        device_props = torch.cuda.get_device_properties(self.req_to_token.device)
        max_num_sm_parts = device_props.multi_processor_count

        # 预分配 MLA tile 调度元数据缓冲区 [max_num_sm_parts, 8]
        self.cuda_graph_mla_metadata = torch.empty(
            (max_num_sm_parts, 8),
            dtype=torch.int32,
            device="cuda",
        )
        # 预分配 SM 分片数缓冲区（长度为 max_bs + 1，用于前缀和格式）
        self.cuda_graph_num_splits = torch.empty(
            max_bs + 1,
            dtype=torch.int32,
            device="cuda",
        )

        # 视图缓冲区在捕获阶段确定具体切片范围
        self.cuda_graph_mla_metadata_view = None
        self.cuda_graph_num_splits_view = None

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
        # CUDA Graph 捕获阶段：填充 KV 索引并将 MLA 元数据写入预分配缓冲区
        if forward_mode.is_decode_or_idle():
            max_seqlen_pad = triton.cdiv(seq_lens.max().item(), PAGE_SIZE)

            # 填充预分配的 KV 索引缓冲区
            create_flashmla_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                None,
                self.cuda_graph_kv_indices,
                self.req_to_token.stride(0),
                self.cuda_graph_kv_indices.stride(0),
            )
            num_q_heads = self.num_q_heads

            # 计算 MLA 调度元数据
            mla_metadata, num_splits = get_mla_metadata(
                seq_lens.to(torch.int32),
                num_q_heads,
                1,
                is_fp8_kvcache=self.is_fp8_kvcache,
            )

            actual_num_sm_parts = mla_metadata.shape[0]
            # 确保实际 SM 分片数不超过预分配的最大值
            assert actual_num_sm_parts <= self.cuda_graph_mla_metadata.shape[0], (
                f"num_sm_parts {actual_num_sm_parts} exceeds preallocated max "
                f"{self.cuda_graph_mla_metadata.shape[0]}"
            )

            # 将元数据拷贝到预分配缓冲区
            self.cuda_graph_mla_metadata[:actual_num_sm_parts].copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)

            # 创建对应实际大小的视图（CUDA Graph 捕获时固定这个切片范围）
            self.cuda_graph_mla_metadata_view = self.cuda_graph_mla_metadata[
                :actual_num_sm_parts
            ]
            self.cuda_graph_num_splits_view = self.cuda_graph_num_splits[: bs + 1]

            self.forward_metadata = FlashMLADecodeMetadata(
                self.cuda_graph_mla_metadata_view,
                self.cuda_graph_num_splits_view,
                self.cuda_graph_kv_indices[:bs, :max_seqlen_pad],
            )

        elif forward_mode.is_target_verify():
            # verify 阶段：序列长度加上草稿 token 数
            seq_lens = seq_lens + self.num_draft_tokens
            max_seqlen_pad = triton.cdiv(seq_lens.max().item(), PAGE_SIZE)

            create_flashmla_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                None,
                self.cuda_graph_kv_indices,
                self.req_to_token.stride(0),
                self.cuda_graph_kv_indices.stride(0),
            )

            mla_metadata, num_splits = get_mla_metadata(
                seq_lens.to(torch.int32),
                self.num_draft_tokens * self.num_q_heads,
                1,
                is_fp8_kvcache=self.is_fp8_kvcache,
            )

            actual_num_sm_parts = mla_metadata.shape[0]
            assert actual_num_sm_parts <= self.cuda_graph_mla_metadata.shape[0]

            self.cuda_graph_mla_metadata[:actual_num_sm_parts].copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)

            self.cuda_graph_mla_metadata_view = self.cuda_graph_mla_metadata[
                :actual_num_sm_parts
            ]
            self.cuda_graph_num_splits_view = self.cuda_graph_num_splits[: bs + 1]

            self.forward_metadata = FlashMLADecodeMetadata(
                self.cuda_graph_mla_metadata_view,
                self.cuda_graph_num_splits_view,
                self.cuda_graph_kv_indices[:bs, :max_seqlen_pad],
            )
        else:
            # prefill 阶段委托给父类
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
        # CUDA Graph 回放阶段：更新 KV 索引和 MLA 调度元数据（seq_lens 已动态变化）
        if forward_mode.is_decode_or_idle():
            assert seq_lens_cpu is not None  # 回放时必须提供 CPU 序列长度
            seq_lens = seq_lens[:bs]
            seq_lens_cpu = seq_lens_cpu[:bs]
            max_seqlen_pad = triton.cdiv(seq_lens_cpu.max().item(), PAGE_SIZE)

            # 重新填充 KV 块索引（每次 decode 步 seq_lens 变化）
            create_flashmla_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices[:bs],
                seq_lens,
                None,
                self.cuda_graph_kv_indices,
                self.req_to_token.stride(0),
                self.cuda_graph_kv_indices.stride(0),
            )
            num_q_heads = self.num_q_heads

            # 重新计算 MLA 调度元数据
            mla_metadata, num_splits = get_mla_metadata(
                seq_lens.to(torch.int32),
                num_q_heads,
                1,
                is_fp8_kvcache=self.is_fp8_kvcache,
            )

            actual_num_sm_parts = mla_metadata.shape[0]

            if actual_num_sm_parts != self.cuda_graph_mla_metadata_view.shape[0]:
                # 若 SM 分片数发生变化（如 batch size 变化），更新视图并发出警告
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"num_sm_parts mismatch in CUDA Graph replay: "
                    f"capture={self.cuda_graph_mla_metadata_view.shape[0]}, "
                    f"replay={actual_num_sm_parts}. "
                    f"This may indicate batch size changed between capture and replay."
                )
                self.cuda_graph_mla_metadata_view = self.cuda_graph_mla_metadata[
                    :actual_num_sm_parts
                ]
                self.cuda_graph_num_splits_view = self.cuda_graph_num_splits[: bs + 1]

            # 将新的元数据写入预分配缓冲区（原地更新，CUDA Graph 录制的地址不变）
            self.cuda_graph_mla_metadata[:actual_num_sm_parts].copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)

            # 更新 forward_metadata 的引用（视图指向同一内存，无需重新分配）
            self.forward_metadata.mla_metadata = self.cuda_graph_mla_metadata_view
            self.forward_metadata.num_splits = self.cuda_graph_num_splits_view
            self.forward_metadata.block_kv_indices = self.cuda_graph_kv_indices[
                :bs, :max_seqlen_pad
            ]

        elif forward_mode.is_target_verify():
            # verify 阶段回放：序列长度加上草稿 token 数
            seq_lens = seq_lens[:bs] + self.num_draft_tokens
            seq_lens_cpu = seq_lens_cpu[:bs] + self.num_draft_tokens
            max_seqlen_pad = triton.cdiv(seq_lens_cpu.max().item(), PAGE_SIZE)

            # 重新填充 KV 索引（包含草稿 token 的序列长度）
            create_flashmla_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices[:bs],
                seq_lens,
                None,
                self.cuda_graph_kv_indices,
                self.req_to_token.stride(0),
                self.cuda_graph_kv_indices.stride(0),
            )

            # 重新计算 verify 阶段 MLA 调度元数据
            mla_metadata, num_splits = get_mla_metadata(
                seq_lens.to(torch.int32),
                self.num_draft_tokens * self.num_q_heads,
                1,
                is_fp8_kvcache=self.is_fp8_kvcache,
            )

            actual_num_sm_parts = mla_metadata.shape[0]

            # 若 SM 分片数变化，更新视图
            if actual_num_sm_parts != self.cuda_graph_mla_metadata_view.shape[0]:
                self.cuda_graph_mla_metadata_view = self.cuda_graph_mla_metadata[
                    :actual_num_sm_parts
                ]
                self.cuda_graph_num_splits_view = self.cuda_graph_num_splits[: bs + 1]

            # 原地更新预分配缓冲区
            self.cuda_graph_mla_metadata[:actual_num_sm_parts].copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)

            # 更新 forward_metadata 引用
            self.forward_metadata.mla_metadata = self.cuda_graph_mla_metadata_view
            self.forward_metadata.num_splits = self.cuda_graph_num_splits_view
            self.forward_metadata.block_kv_indices = self.cuda_graph_kv_indices[
                :bs, :max_seqlen_pad
            ]
        else:
            # prefill 阶段委托给父类
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
        # CUDA Graph 序列长度填充值为 1
        return 1

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        # decode 阶段：使用 FlashMLA 内核完成注意力计算（支持 FP8 KV 缓存）
        cache_loc = forward_batch.out_cache_loc

        if k is not None:
            assert v is not None
            if save_kv_cache:
                # 将当前步的 KV 写入缓存
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer,
                    cache_loc,
                    k,
                    v,
                )
        bs = forward_batch.batch_size
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)

        # 将 Q 重塑为 [bs, num_tokens_per_seq, num_heads, head_dim] 格式
        reshape_q = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)
        if self.is_fp8_kvcache:
            # FP8 KV 缓存路径：需要对 Q 进行 FP8 量化
            if layer.k_scale is not None:
                q_scale = layer.k_scale
                descale_q = layer.k_scale.reshape(1)
                descale_k = layer.k_scale.reshape(1)
            else:
                # 无量化缩放因子时使用 1.0（无损量化）
                q_scale = torch.ones((1,), dtype=torch.float32, device=reshape_q.device)
                descale_q = torch.ones(
                    (1,), dtype=torch.float32, device=reshape_q.device
                )
                descale_k = torch.ones(
                    (1,), dtype=torch.float32, device=reshape_q.device
                )

            # 对 Q 进行 FP8 量化（先展平到 2D，量化后再 reshape 回原形状）
            q_shape = reshape_q.shape
            reshape_q_2d = reshape_q.reshape(-1, q_shape[-1])
            reshape_q_fp8_2d, _ = scaled_fp8_quant(reshape_q_2d, q_scale)
            reshape_q_fp8 = reshape_q_fp8_2d.reshape(q_shape)
            # 调用 FlashMLA 内核（FP8 路径，携带 descale 因子）
            o, _ = flash_mla_with_kvcache(
                q=reshape_q_fp8,
                k_cache=k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim),
                block_table=self.forward_metadata.block_kv_indices[:bs],
                cache_seqlens=forward_batch.seq_lens.to(torch.int32),
                head_dim_v=self.kv_lora_rank,    # V 维度等于 kv_lora_rank
                tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                num_splits=self.forward_metadata.num_splits,
                softmax_scale=layer.scaling,
                causal=True,
                descale_q=descale_q,
                descale_k=descale_k,
            )

            return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)
        else:
            # 标准精度路径：直接调用 FlashMLA 内核
            o, _ = flash_mla_with_kvcache(
                q=reshape_q,
                k_cache=k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim),
                block_table=self.forward_metadata.block_kv_indices[:bs],
                cache_seqlens=forward_batch.seq_lens.to(torch.int32),
                head_dim_v=self.kv_lora_rank,
                tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                num_splits=self.forward_metadata.num_splits,
                softmax_scale=layer.scaling,
                causal=True,
            )

            return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        # extend/prefill 阶段：标准 extend 和 draft_extend 委托给父类 FlashInfer
        if (
            forward_batch.forward_mode == ForwardMode.EXTEND
            or forward_batch.forward_mode == ForwardMode.DRAFT_EXTEND
        ):
            return super().forward_extend(q, k, v, layer, forward_batch, save_kv_cache)
        else:
            # target_verify 阶段（多草稿 token）：使用 FlashMLA 内核处理
            cache_loc = forward_batch.out_cache_loc

            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

            bs = forward_batch.batch_size
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)

            reshape_q = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)
            if self.is_fp8_kvcache:
                if layer.k_scale is not None:
                    q_scale = layer.k_scale
                    descale_q = layer.k_scale.reshape(1)
                    descale_k = layer.k_scale.reshape(1)
                else:
                    q_scale = torch.ones(
                        (1,), dtype=torch.float32, device=reshape_q.device
                    )
                    descale_q = torch.ones(
                        (1,), dtype=torch.float32, device=reshape_q.device
                    )
                    descale_k = torch.ones(
                        (1,), dtype=torch.float32, device=reshape_q.device
                    )

                q_shape = reshape_q.shape
                reshape_q_2d = reshape_q.reshape(-1, q_shape[-1])
                reshape_q_fp8_2d, _ = scaled_fp8_quant(reshape_q_2d, q_scale)
                reshape_q_fp8 = reshape_q_fp8_2d.reshape(q_shape)
                # verify 阶段序列长度需加上草稿 token 数
                o, _ = flash_mla_with_kvcache(
                    q=reshape_q_fp8,
                    k_cache=k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim),
                    block_table=self.forward_metadata.block_kv_indices[:bs],
                    cache_seqlens=forward_batch.seq_lens.to(torch.int32)
                    + self.num_draft_tokens,
                    head_dim_v=self.kv_lora_rank,
                    tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                    num_splits=self.forward_metadata.num_splits,
                    softmax_scale=layer.scaling,
                    causal=True,
                    descale_q=descale_q,
                    descale_k=descale_k,
                )
            else:
                o, _ = flash_mla_with_kvcache(
                    q=reshape_q,
                    k_cache=k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim),
                    block_table=self.forward_metadata.block_kv_indices[:bs],
                    cache_seqlens=forward_batch.seq_lens.to(torch.int32)
                    + self.num_draft_tokens,
                    head_dim_v=self.kv_lora_rank,
                    tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                    num_splits=self.forward_metadata.num_splits,
                    softmax_scale=layer.scaling,
                    causal=True,
                )
            return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)


class FlashMLAMultiStepDraftBackend:
    # FlashMLA 多步推测解码草稿后端：管理多个步骤的 FlashMLA 后端实例
    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,                    # 每步保留的 top-k 候选数（当前仅支持 1）
        speculative_num_steps: int,   # 推测解码总步数
    ):
        if topk > 1:
            raise ValueError(
                "Currently FlashMLA only supports topk=1 for speculative decoding"
            )
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        max_bs = model_runner.req_to_token_pool.size * self.topk
        # 为每个推测解码步骤预分配独立的 kv_indptr 缓冲区
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )

        # 为每个草稿步骤（除最后一步外）创建独立的 FlashMLA 后端
        self.attn_backends = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                FlashMLABackend(
                    model_runner,
                    skip_prefill=True,            # 草稿步骤不需要 prefill 逻辑
                    kv_indptr_buf=self.kv_indptr[i],
                    kv_last_page_len_buf=None,
                )
            )

    def common_template(
        self,
        forward_batch: ForwardBatch,
        call_fn: Callable,
    ):
        # 通用模板：对所有草稿步骤执行 call_fn
        assert forward_batch.spec_info is not None

        for i in range(self.speculative_num_steps - 1):
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # 初始化所有草稿步骤的前向元数据
        def call_fn(i, forward_batch):
            assert forward_batch.spec_info is not None
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, call_fn)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        # 初始化所有草稿步骤后端的 CUDA Graph 状态
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs, max_num_tokens, block_kv_indices=None
            )

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        # CUDA Graph 捕获阶段：所有草稿步骤均使用 DECODE 模式捕获
        def call_fn(i, forward_batch):
            # EAGLE draft worker uses DECODE mode for draft steps
            # EAGLE 草稿 worker 使用 DECODE 模式处理每个草稿步骤
            from sglang.srt.model_executor.forward_batch_info import ForwardMode

            # Create a dummy forward_mode for draft step
            # 为草稿步骤创建 DECODE 前向模式
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

        self.common_template(forward_batch, call_fn)

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        # CUDA Graph 回放阶段：更新所有草稿步骤的元数据
        def call_fn(i, forward_batch):
            from sglang.srt.model_executor.forward_batch_info import ForwardMode

            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                seq_lens_sum=-1,          # 草稿步骤不需要 seq_lens_sum
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
            )

        self.common_template(forward_batch, call_fn)
