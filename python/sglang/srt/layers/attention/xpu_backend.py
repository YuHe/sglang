from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
# 从 FlashAttention 后端导入元数据类和辅助函数
from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionMetadata,
    make_local_attention_virtual_batches,
    merge_state_v2_wrapper,
    prepare_swa_spec_page_table_triton,
)
from sglang.srt.managers.schedule_batch import get_global_server_args
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import get_device_core_count

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

# XPU 专用 FlashMLA 解码核（sgl_kernel 为英特尔 XPU 提供的算子库）
from sgl_kernel import flash_mla_decode, flash_mla_get_workspace_size, merge_state_v2
from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache


class XPUAttentionBackend(AttentionBackend):
    """XPU FlashAttention backend, currently based on FlashAttentionBackend, will be refactored later.
    XPU FlashAttention 后端，当前基于 FlashAttentionBackend，后续将重构。

    TODO:
    - Prefill and Decode disaggregation, currently only chunked prefill is supported
    - Speculative Decoding support
    - XPU Graph support, see https://github.com/pytorch/pytorch/issues/162143
    - MLA Prefill support
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        speculative_step_id=0,
        topk=0,
        speculative_num_steps=0,
    ):
        super().__init__()

        # 不允许同时启用滑动窗口注意力和交叉注意力（编码器-解码器模型）
        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        self.forward_metadata: FlashAttentionMetadata = None
        # 用于投机解码 topk>1 时扩展草稿解码/验证的额外元数据
        self.forward_metadata_spec_decode_expand: FlashAttentionMetadata = None
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.decode_cuda_graph_metadata = {}
        self.target_verify_metadata = {}
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.kv_cache_dtype_str = model_runner.server_args.kv_cache_dtype  # 字符串形式，用于判断是否为 fp8
        self.page_size = model_runner.page_size
        # 是否使用 MLA（多头潜在注意力，DeepSeek 架构）
        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA
        self.skip_prefill = skip_prefill
        # 是否为混合 SWA 模型（部分层使用滑动窗口）
        self.is_hybrid_swa = model_runner.is_hybrid_swa
        if self.is_hybrid_swa:
            # 全量池索引 → SWA 池索引的映射表
            self.full_to_swa_index_mapping = (
                model_runner.token_to_kv_pool.full_to_swa_index_mapping
            )
        self.topk = model_runner.server_args.speculative_eagle_topk or 0
        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
        )
        self.speculative_step_id = speculative_step_id  # 当前草稿步骤编号

        # 局部注意力（Local Attention）设置：Llama4 等模型使用分块注意力
        self.attention_chunk_size = (
            model_runner.attention_chunk_size
            if hasattr(model_runner, "attention_chunk_size")
            else None
        )

        # 滑动窗口大小（不同层可能不同，仅用于 SWA 元数据准备）
        # 每层是否启用 SWA 由 layer.sliding_window_size 决定
        self.sliding_window_size = model_runner.sliding_window_size
        self.has_swa = (
            self.sliding_window_size is not None and self.sliding_window_size > -1
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """初始化 forward 元数据，供该批次所有层复用（避免重复计算）。"""
        metadata = FlashAttentionMetadata()
        seqlens_in_batch = forward_batch.seq_lens
        batch_size = forward_batch.batch_size
        device = seqlens_in_batch.device

        if forward_batch.forward_mode.is_decode_or_idle():
            # 草稿解码（Draft Decode）分支
            if forward_batch.spec_info is not None:
                assert (
                    False
                ), "XPUAttentionBackend doesn't support speculative decoding yet, please use --attention-backend triton instead."
                # 以下代码暂未生效（assert False 之后），保留供后续实现参考
                if self.topk <= 1:
                    # topk=1：每个请求多产生 speculative_step_id+1 个 token
                    metadata.cache_seqlens_int32 = (
                        seqlens_in_batch + (self.speculative_step_id + 1)
                    ).to(torch.int32)
                    metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item() + (
                        self.speculative_step_id + 1
                    )
                    metadata.cu_seqlens_q = torch.arange(
                        0, batch_size + 1, dtype=torch.int32, device=device
                    )
                    metadata.cu_seqlens_k = torch.nn.functional.pad(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
                    metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, : metadata.max_seq_len_k
                    ]
                else:
                    # topk>1：每个请求有 topk 个并行候选，使用级联注意力（cascade attention）
                    metadata.cache_seqlens_int32 = (seqlens_in_batch).to(torch.int32)
                    metadata.max_seq_len_q = self.topk
                    metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
                    metadata.cu_seqlens_q = torch.arange(
                        0,
                        batch_size * self.topk + 1,
                        step=self.topk,
                        dtype=torch.int32,
                        device=device,
                    )
                    metadata.cu_seqlens_k = torch.nn.functional.pad(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
                    metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, : metadata.max_seq_len_k
                    ]

                    # 扩展元数据：记录每个草稿 token 在已生成位置的 KV 信息
                    metadata_expand = FlashAttentionMetadata()
                    decode_length = self.speculative_step_id + 1
                    metadata_expand.cache_seqlens_int32 = torch.full(
                        (seqlens_in_batch.numel() * self.topk,),
                        decode_length,
                        device=device,
                        dtype=torch.int32,
                    )
                    metadata_expand.max_seq_len_q = 1
                    metadata_expand.cu_seqlens_q = torch.arange(
                        0,
                        metadata_expand.cache_seqlens_int32.numel() + 1,
                        dtype=torch.int32,
                        device=device,
                    )
                    metadata_expand.cu_seqlens_k = torch.arange(
                        0,
                        metadata_expand.cache_seqlens_int32.numel() * decode_length + 1,
                        step=decode_length,
                        dtype=torch.int32,
                        device=device,
                    )
                    # out_cache_loc 形状: [bs, num_steps, topk] → [bs*topk, num_steps]
                    cache_loc = forward_batch.out_cache_loc.view(
                        -1, self.speculative_num_steps
                    )
                    # 只取前 decode_length 步的缓存位置作为扩展分页表
                    metadata_expand.page_table = (
                        cache_loc[:, :decode_length].contiguous().to(torch.int32)
                    )
                    self.forward_metadata_spec_decode_expand = metadata_expand
            else:
                # 普通解码（Normal Decode）
                metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
                metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
                metadata.cu_seqlens_q = torch.arange(
                    0, batch_size + 1, dtype=torch.int32, device=device
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
                )
                metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]
            # TODO: Llama 4 EAGLE 场景需要测试局部注意力
            self._init_local_attn_metadata(forward_batch, metadata, device)
        elif forward_batch.forward_mode.is_target_verify():
            if self.topk <= 1:
                # topk=1 目标验证：KV 长度 = 序列长度 + 草稿 token 数
                metadata.cache_seqlens_int32 = (
                    forward_batch.seq_lens + self.speculative_num_draft_tokens
                ).to(torch.int32)
                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                metadata.max_seq_len_k = (
                    forward_batch.seq_lens_cpu.max().item()
                    + self.speculative_num_draft_tokens
                )
                metadata.cu_seqlens_q = torch.arange(
                    0,
                    batch_size * self.speculative_num_draft_tokens + 1,
                    self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=device,
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]

                self._init_local_attn_metadata(forward_batch, metadata, device)
            else:
                # topk>1 目标验证：需要级联注意力（公共前缀 + 新草稿 token 分开计算）
                metadata.cache_seqlens_int32 = forward_batch.seq_lens.to(torch.int32)
                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
                metadata.cu_seqlens_q = torch.arange(
                    0,
                    batch_size * self.speculative_num_draft_tokens + 1,
                    step=self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=device,
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]

                # 扩展元数据：处理每个草稿 token 对其他草稿 token 的注意力（因果掩码）
                metadata_expand = FlashAttentionMetadata()

                metadata_expand.max_seq_len_q = 1
                metadata_expand.cu_seqlens_q = torch.arange(
                    0,
                    forward_batch.seq_lens.numel() * self.speculative_num_draft_tokens
                    + 1,
                    dtype=torch.int32,
                    device=device,
                )

                # 构建扩展分页表（处理 topk>1 时的自注意力掩码）
                # offsets: 草稿 token 在序列末尾的偏移量 [0, 1, ..., speculative_num_draft_tokens-1]
                offsets = torch.arange(
                    self.speculative_num_draft_tokens, device=device
                ).unsqueeze(
                    0
                )  # 形状: (1, speculative_num_draft_tokens)
                # cols: 每个请求的草稿 token 在 req_to_token 中的列索引
                cols = offsets.expand(
                    forward_batch.seq_lens.numel(), -1
                ) + forward_batch.seq_lens.unsqueeze(1)
                # cum_len: 用于从自定义掩码中提取每个草稿 token 的因果掩码行
                cum_len = torch.nn.functional.pad(
                    torch.cumsum(
                        (
                            forward_batch.seq_lens + self.speculative_num_draft_tokens
                        ).repeat_interleave(self.speculative_num_draft_tokens),
                        dim=0,
                    ),
                    (1, 0),
                )[:-1]
                mask_extraction_indices = (
                    cols.repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                    + cum_len[:, None]
                ).view(1, -1)
                # mask: 形状 (bsz * draft_num, draft_num)，每行表示一个草稿 token 可见的其他草稿 token
                mask = forward_batch.spec_info.custom_mask[
                    mask_extraction_indices
                ].view(
                    -1, self.speculative_num_draft_tokens
                )

                # 基于掩码对分页表列进行排序（将有效列移到前面，无效列移到后面）
                # 未掩码的示例：[[8, 9, 10], [8, 9, 10], [8, 9, 10]]  掩码（整数格式）: [[1,0,0],[1,1,0],[1,0,1]]
                # 带填充的掩码：[[8, 0, 0], [8, 9, 0], [8, 0, 10]]  无填充掩码：[[8, 9, 10], [8, 9, 10], [8, 10, 9]]
                # cache_seqlens_int32 = [1, 2, 2]，所以多余列会被忽略
                col_indices = offsets.expand(
                    mask.shape[0], self.speculative_num_draft_tokens
                )
                # keys: 有效位置保持原索引，无效位置加偏移使其排在后面
                keys = torch.where(
                    mask, col_indices, col_indices + self.speculative_num_draft_tokens
                )
                _, sort_order = torch.sort(keys, dim=1)
                non_masked_page_table = (
                    forward_batch.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, :
                    ]
                    .gather(1, cols)
                    .repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                )  # 形状: (bsz, draft_num)
                # 按排序顺序重排分页表，将有效 KV 移到每行前面
                metadata_expand.page_table = non_masked_page_table.gather(1, sort_order)
                metadata_expand.cache_seqlens_int32 = mask.sum(dim=1).to(torch.int32)
                metadata_expand.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata_expand.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                self.forward_metadata_spec_decode_expand = metadata_expand

                # 若有 SWA，则初始化 SWA 专用的投机解码元数据
                if self.has_swa:
                    self._init_sliding_window_attn_spec_metadata(
                        metadata, metadata_expand
                    )

        elif forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed():
            # 预填充（extend）模式
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )
            metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.max_seq_len_k
            ]

            if (
                any(forward_batch.extend_prefix_lens_cpu)
                or forward_batch.forward_mode == ForwardMode.DRAFT_EXTEND
            ):
                # 有前缀缓存或草稿扩展：max_seq_len_q 取新 token 的最大长度
                extend_seq_lens = forward_batch.extend_seq_lens
                metadata.max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
            else:
                # 无前缀缓存：q 和 k 长度相同
                metadata.max_seq_len_q = metadata.max_seq_len_k
                metadata.cu_seqlens_q = metadata.cu_seqlens_k

            # 若为普通预填充（EXTEND），初始化局部注意力元数据
            if forward_batch.forward_mode == ForwardMode.EXTEND:
                self._init_local_attn_metadata(forward_batch, metadata, device)

        # 编码器-解码器交叉注意力的元数据
        if forward_batch.encoder_lens is not None:
            assert (
                forward_batch.encoder_lens.numel() == 1
            ), "Only encoder size 1 is supported for now"

            metadata.encoder_lens_int32 = forward_batch.encoder_lens.to(torch.int32)
            metadata.encoder_cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(metadata.encoder_lens_int32, dim=0, dtype=torch.int32),
                (1, 0),
            )
            metadata.encoder_max_seq_len_k = metadata.encoder_lens_int32.max().item()
            metadata.encoder_page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.encoder_max_seq_len_k
            ]

            # 目前只支持 encoder_lens.numel()==1：解码器 page_table 从编码器之后开始
            metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices,
                metadata.encoder_max_seq_len_k : (
                    metadata.encoder_max_seq_len_k + metadata.max_seq_len_k
                ),
            ]

        if self.use_mla:
            # MLA 模式：计算 flash_mla_decode 所需的工作区大小并按需分配
            workspace_size = flash_mla_get_workspace_size(
                self.max_context_len,
                batch_size,
                sm_count=get_device_core_count(),
                num_kv_splits=-1,  # 自动选择最优分割数
            )
            if (
                not hasattr(self, "workspace")
                or self.workspace.numel() < workspace_size
            ):
                self.workspace = torch.empty(
                    workspace_size, device=self.device, dtype=torch.uint8
                )

        # 将分页表转换为步幅格式（FA3 API 要求）：每 page_size 个 token 取一个页起始索引
        if self.page_size > 1:
            self.strided_indices = torch.arange(
                0, metadata.page_table.shape[1], self.page_size, device=self.device
            )
            # token 索引 → 页编号
            metadata.page_table = (
                metadata.page_table[:, self.strided_indices] // self.page_size
            )

        self.forward_metadata = metadata

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # MLA 额外参数：分离的 Q/K 旋转位置编码
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ):
        """预填充阶段前向传播：写入 KV 缓存，根据模式选择注意力核。"""
        if k is not None:
            assert v is not None
            if save_kv_cache:
                # 获取缓存位置：交叉注意力使用编码器位置，自注意力使用解码器位置
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                if not self.use_mla:
                    # 普通 MHA：存储 k/v（可带 fp8 scale）
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )
                else:
                    # MLA：只存储压缩后的 k（latent）和 k_rope（位置编码部分）
                    forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope,
                    )

        # 复用预计算的元数据，避免重复计算
        metadata = self.forward_metadata

        # 计算窗口大小（两侧包含式）
        # 注意：model.get_attention_sliding_window_size() 已经 -1，这里不再减
        is_hybrid_swa = (
            layer.sliding_window_size is not None and layer.sliding_window_size > -1
        )
        window_size = (layer.sliding_window_size, 0) if is_hybrid_swa else (-1, -1)

        # 当前不支持 FP8 KV 缓存（预填充阶段）
        k_descale, v_descale = None, None
        causal = not layer.is_cross_attention

        # 判断是否使用局部注意力（Llama4 等模型的分块注意力）
        use_local_attn = (
            self.attention_chunk_size is not None
            and metadata.local_attn_metadata is not None
            and (hasattr(layer, "use_irope") and layer.use_irope)
        )

        # 目标验证（topk>1）使用级联注意力（prefix + new tokens 分两阶段）
        # 滑动窗口不使用级联注意力（FA3 不支持批量 window sizes；公共前缀重复计算代价小）
        use_cascade_attn = (
            forward_batch.forward_mode.is_target_verify()
            and self.topk > 1
            and not is_hybrid_swa
        )

        # 兼容旧版 FA3 接口：将新字段放入条件 kwargs
        kwargs = {}
        if sinks is not None:
            kwargs["sinks"] = sinks

        # 根据是否使用局部注意力选择合适的分页表和序列长度信息
        if use_local_attn:
            local_metadata = metadata.local_attn_metadata
            page_table = local_metadata.local_block_table
            cu_seqlens_q = local_metadata.local_query_start_loc
            cache_seqlens = local_metadata.local_seqused_k
            max_seqlen_q = local_metadata.local_max_query_len
        elif is_hybrid_swa and metadata.swa_spec_metadata is not None:
            # SWA 投机解码：使用合并后的 SWA 专用元数据
            swa_spec_metadata = metadata.swa_spec_metadata
            page_table = swa_spec_metadata.page_table
            cu_seqlens_q = swa_spec_metadata.cu_seqlens_q
            cache_seqlens = swa_spec_metadata.cache_seqlens_int32
            max_seqlen_q = swa_spec_metadata.max_seq_len_q
            cu_seqlens_k = swa_spec_metadata.cu_seqlens_k
        else:
            page_table = metadata.page_table
            cu_seqlens_q = metadata.cu_seqlens_q
            cache_seqlens = metadata.cache_seqlens_int32
            max_seqlen_q = metadata.max_seq_len_q
            cu_seqlens_k = metadata.cu_seqlens_k

        # 预填充阶段使用 Flash Attention
        if not self.use_mla:
            # 标准 MHA 预填充
            key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )
            # reshape 为 [num_pages, page_size, num_kv_heads, head_dim]
            key_cache = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            value_cache = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.head_dim
            )
            if layer.is_cross_attention:
                # 交叉注意力使用编码器分页表
                page_table = metadata.encoder_page_table
                cache_seqlens = metadata.encoder_lens_int32
                cu_seqlens_k = metadata.encoder_cu_seqlens_k
                window_size = (-1, -1)

            result = flash_attn_with_kvcache(
                q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                k_cache=key_cache,
                v_cache=value_cache,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
                max_seqlen_q=max_seqlen_q,
                softmax_scale=layer.scaling,
                causal=False if use_cascade_attn else causal,
                window_size=window_size,
                softcap=layer.logit_cap,
                k_descale=k_descale,
                v_descale=v_descale,
                return_softmax_lse=use_cascade_attn,
                **kwargs,
            )

            if use_cascade_attn:
                # 级联注意力：合并公共前缀注意力和新草稿 token 注意力
                o, softmax_lse, *rest = result
                o_expand, softmax_lse_expand, *rest_expand = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=self.forward_metadata_spec_decode_expand.page_table,
                    cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                    cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                    cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                    max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=True,
                    **kwargs,
                )
                # 使用 online softmax 合并两部分注意力输出（FlashInfer merge_state_v2）
                o, _ = merge_state_v2_wrapper(
                    o,
                    softmax_lse.T.contiguous(),
                    o_expand,
                    softmax_lse_expand.T.contiguous(),
                )
            else:
                o = result
        else:
            if (
                forward_batch.attn_attend_prefix_cache is not None
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
            ):
                # MLA + 分块前缀缓存：使用 varlen（变长）注意力
                if forward_batch.attn_attend_prefix_cache:
                    assert not get_global_server_args().disable_chunked_prefix_cache
                    # 对分块前缀 KV 缓存执行 MHA（MLA 模型使用）
                    assert forward_batch.prefix_chunk_idx is not None
                    assert forward_batch.prefix_chunk_cu_seq_lens is not None
                    assert forward_batch.prefix_chunk_max_seq_lens is not None

                    chunk_idx = forward_batch.prefix_chunk_idx
                    assert chunk_idx >= 0

                    assert forward_batch.mha_return_lse
                    # 对当前分块执行非因果注意力（前缀不需要因果掩码）
                    output = flash_attn_varlen_func(
                        q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                        v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
                        cu_seqlens_q=metadata.cu_seqlens_q,
                        cu_seqlens_k=forward_batch.prefix_chunk_cu_seq_lens[chunk_idx],
                        max_seqlen_q=metadata.max_seq_len_q,
                        max_seqlen_k=forward_batch.prefix_chunk_max_seq_lens[chunk_idx],
                        softmax_scale=layer.scaling,
                        causal=False,
                        return_softmax_lse=True,
                    )
                else:
                    # 新 token 对自身的因果注意力（不访问前缀缓存）
                    output = flash_attn_varlen_func(
                        q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                        v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
                        cu_seqlens_q=metadata.cu_seqlens_q,
                        cu_seqlens_k=metadata.cu_seqlens_q,
                        max_seqlen_q=metadata.max_seq_len_q,
                        max_seqlen_k=metadata.max_seq_len_q,
                        softmax_scale=layer.scaling,
                        causal=True,
                        return_softmax_lse=forward_batch.mha_return_lse,
                    )
                if forward_batch.mha_return_lse:
                    output, lse, *rest = output
                    lse = torch.transpose(lse, 0, 1).contiguous()
                    return output, lse
                return output
            else:
                # MLA 吸收式注意力（absorbed MLA）：K 以 (c_kv, k_rope) 分开存储
                kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(
                    layer.layer_id
                ).to(q.dtype)
                # 从统一 KV 缓冲区中分离 k_rope 和 c_kv（compressed KV）
                k_rope = kv_cache[:, :, layer.v_head_dim :]
                c_kv = kv_cache[:, :, : layer.v_head_dim]
                k_rope_cache = k_rope.view(
                    -1,
                    self.page_size,
                    layer.tp_k_head_num,
                    layer.head_dim - layer.v_head_dim,
                )
                c_kv_cache = c_kv.view(
                    -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                )
                # 拆分 q 为 q_nope（latent 部分）和 q_rope（位置编码部分）
                if q_rope is not None:
                    q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                    q_rope = q_rope.view(
                        -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                    )
                else:
                    q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                    q_nope = q_all[:, :, : layer.v_head_dim]
                    q_rope = q_all[:, :, layer.v_head_dim :]

                # 调用 XPU FlashAttention + MLA 核（qv=q_nope 表示 V 侧的 latent）
                result = flash_attn_with_kvcache(
                    q=q_rope,
                    k_cache=k_rope_cache,
                    v_cache=c_kv_cache,
                    qv=q_nope,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=False if use_cascade_attn else causal,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=use_cascade_attn,
                )
                if use_cascade_attn:
                    # 级联注意力：合并两阶段结果
                    o, softmax_lse, *rest = result
                    o_expand, softmax_lse_expand, *rest_expand = (
                        flash_attn_with_kvcache(
                            q=q_rope,
                            k_cache=k_rope_cache,
                            v_cache=c_kv_cache,
                            qv=q_nope,
                            page_table=self.forward_metadata_spec_decode_expand.page_table,
                            cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                            cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                            cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                            max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                            softmax_scale=layer.scaling,
                            causal=False,
                            window_size=window_size,
                            softcap=layer.logit_cap,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            return_softmax_lse=True,
                        )
                    )
                    o, _ = merge_state_v2_wrapper(
                        o,
                        softmax_lse.T.contiguous(),
                        o_expand,
                        softmax_lse_expand.T.contiguous(),
                    )
                else:
                    o = result

        # 展平输出：(num_tokens, num_heads * v_head_dim)
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # MLA 额外参数
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """解码阶段前向传播：写入 KV 缓存，调用 XPU Flash Attention 核。"""
        if k is not None:
            assert v is not None
            if save_kv_cache:
                # 交叉注意力使用编码器缓存位置
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                if not self.use_mla:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )
                else:
                    # MLA：k 为 compressed latent，k_rope 从 k 末尾切出或单独传入
                    k_rope_val = (
                        k_rope if k_rope is not None else k[:, :, layer.v_head_dim :]
                    )
                    forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope_val,
                    )

        # 复用预计算的元数据
        metadata = self.forward_metadata
        local_attn_metadata = getattr(metadata, "local_attn_metadata", None)
        use_local_attn = (
            self.attention_chunk_size is not None
            and local_attn_metadata is not None
            and (hasattr(layer, "use_irope") and layer.use_irope)
        )

        # 投机解码（topk>1）时启用级联注意力：
        # 1. DRAFT_DECODE：topk>1 时使用级联注意力
        # 2. IDLE：无 spec_info，不使用级联注意力
        use_cascade_attn = forward_batch.spec_info is not None and self.topk > 1

        # 计算窗口大小（两侧包含式）
        window_size = (
            (layer.sliding_window_size, 0)
            if layer.sliding_window_size is not None and layer.sliding_window_size > -1
            else (-1, -1)
        )
        causal = not layer.is_cross_attention

        # 兼容旧版 FA3 接口
        kwargs = {}
        if sinks is not None:
            kwargs["sinks"] = sinks

        k_descale, v_descale = None, None
        # FP8 KV 缓存缩放：仅当 fp8 明确启用、有 scale 参数且 head_dim<=256 时生效
        if self.kv_cache_dtype_str != "auto" and layer.head_dim <= 256:
            if layer.k_scale is not None:
                descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
                k_descale = layer.k_scale.expand(descale_shape)
                v_descale = layer.v_scale.expand(descale_shape)
            # 将 q 转换为 KV 缓存数据类型（fp8）
            q = q.to(self.kv_cache_dtype)
            q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
            k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None
        if not self.use_mla:
            # 标准 MHA 解码

            key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )
            # reshape 为 [num_pages, page_size, num_kv_heads, head_dim]
            key_cache = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            value_cache = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.head_dim
            )

            if layer.is_cross_attention:
                # 交叉注意力：使用编码器分页表，不使用因果掩码
                o = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=metadata.encoder_page_table,
                    cache_seqlens=metadata.encoder_lens_int32,
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    cu_seqlens_k_new=metadata.encoder_cu_seqlens_k,
                    max_seqlen_q=1,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=(-1, -1),
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    **kwargs,
                )
            elif use_local_attn:
                # 局部注意力（分块自注意力）
                o = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=local_attn_metadata.local_block_table,
                    cache_seqlens=local_attn_metadata.local_seqused_k,
                    cu_seqlens_q=local_attn_metadata.local_query_start_loc,
                    cu_seqlens_k_new=None,
                    max_seqlen_q=local_attn_metadata.local_max_query_len,
                    softmax_scale=layer.scaling,
                    causal=True,
                    window_size=(-1, -1),
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    **kwargs,
                )
            else:
                # 普通解码自注意力
                page_table = metadata.page_table
                cache_seqlens = metadata.cache_seqlens_int32
                cu_seqlens_k = metadata.cu_seqlens_k
                max_seqlen_q = metadata.max_seq_len_q
                q_reshaped = q.contiguous().view(
                    -1, layer.tp_q_head_num, layer.head_dim
                )

                # 默认：单 token 自注意力
                result = flash_attn_with_kvcache(
                    q=q_reshaped,
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=False if use_cascade_attn else causal,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=use_cascade_attn,
                    **kwargs,
                )
                if use_cascade_attn:
                    # 级联注意力：合并公共前缀注意力和草稿新 token 注意力
                    o, softmax_lse, *rest = result
                    o_expand, softmax_lse_expand, *rest_expand = (
                        flash_attn_with_kvcache(
                            q=q_reshaped,
                            k_cache=key_cache,
                            v_cache=value_cache,
                            page_table=self.forward_metadata_spec_decode_expand.page_table,
                            cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                            cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                            cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                            max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                            softmax_scale=layer.scaling,
                            causal=False,
                            window_size=window_size,
                            softcap=layer.logit_cap,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            return_softmax_lse=True,
                            **kwargs,
                        )
                    )
                    # 使用 sgl_kernel 的 merge_state_v2 合并（非 FlashInfer 版本）
                    o, _ = merge_state_v2(
                        o,
                        softmax_lse.T.contiguous(),
                        o_expand,
                        softmax_lse_expand.T.contiguous(),
                    )
                else:
                    o = result
        else:
            # MLA 吸收式解码（absorbed MLA decode）
            kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).to(
                q.dtype
            )
            assert not use_cascade_attn, "Cascade attention is not supported with MLA"

            # 拆分 q 为 q_nope 和 q_rope
            if q_rope is not None:
                q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                q_rope = q_rope.view(
                    -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                )
            else:
                q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                q_nope = q_all[:, :, : layer.v_head_dim]
                q_rope = q_all[:, :, layer.v_head_dim :]

            # 调用 XPU flash_mla_decode 核：接受 q_nope, q_rope, kv_cache, workspace
            o = flash_mla_decode(
                q_nope,
                q_rope,
                kv_cache.view(-1, self.page_size, layer.head_dim),
                metadata.cache_seqlens_int32,
                metadata.page_table,
                self.workspace,
                layer.scaling,
            )

        # 展平输出
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def get_cuda_graph_seq_len_fill_value(self):
        """CUDA Graph 中序列长度的填充值（解码时最小有效长度为 1）。"""
        return 1

    def _init_local_attn_metadata(
        self, forwardbatch: ForwardBatch, metadata: FlashAttentionMetadata, device
    ):
        """初始化局部注意力元数据（分块自注意力），若未启用分块则设为 None。"""
        if self.attention_chunk_size is None:
            metadata.local_attn_metadata = None
            return

        cu_seqlens_q = metadata.cu_seqlens_q
        cache_seqlens_int32 = metadata.cache_seqlens_int32
        # 混合 SWA 模型需要将全量池分页表转换为 SWA 池分页表
        if self.is_hybrid_swa:
            page_table = self.full_to_swa_index_mapping[metadata.page_table].to(
                torch.int32
            )
        else:
            page_table = metadata.page_table
        # 任一必要输入为 None 时跳过局部注意力初始化
        if cu_seqlens_q is None or cache_seqlens_int32 is None or page_table is None:
            metadata.local_attn_metadata = None
            return

        # 将张量转换到 CPU 供 numpy 处理（make_local_attention_virtual_batches 使用 numpy）
        cu_seqlens_q_np = cu_seqlens_q.cpu().numpy()
        seq_lens_np = cache_seqlens_int32.cpu().numpy()
        # 生成局部注意力的虚拟批次：将每个序列按 attention_chunk_size 切块
        (
            seqlens_q_local_np,
            cu_seqlens_q_local_np,
            seqlens_k_local_np,
            block_table_local,
        ) = make_local_attention_virtual_batches(
            self.attention_chunk_size,
            cu_seqlens_q_np,
            seq_lens_np,
            page_table,
            self.page_size,
        )

        # 构建局部注意力专用元数据对象
        local_metadata = FlashAttentionMetadata.LocalAttentionMetadata(
            local_query_start_loc=torch.from_numpy(cu_seqlens_q_local_np).to(device),
            local_seqused_k=torch.from_numpy(seqlens_k_local_np).to(device),
            local_block_table=block_table_local.to(device),
            local_max_query_len=int(seqlens_q_local_np.max()),
            local_max_seq_len=int(seqlens_k_local_np.max()),
        )
        metadata.local_attn_metadata = local_metadata

    def _init_sliding_window_attn_spec_metadata(
        self,
        metadata: FlashAttentionMetadata,
        metadata_expand: FlashAttentionMetadata,
        metadata_swa: Optional[FlashAttentionMetadata] = None,
    ):
        """初始化滑动窗口注意力的投机解码元数据，合并公共前缀和新草稿 token 的 KV。"""
        # TODO: 当 page_size > 1 时需支持 SWA spec
        assert (
            self.page_size == 1
        ), "FlashAttention backend doesn't support topk > 1 speculative decoding with page size > 1 sliding window attention"

        # 合并序列长度：公共前缀长度（重复 speculative_num_draft_tokens 次）+ 新草稿 token 的长度
        cache_seqlens_int32 = (
            metadata.cache_seqlens_int32.repeat_interleave(
                self.speculative_num_draft_tokens
            )
            + metadata_expand.cache_seqlens_int32
        )
        cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(cache_seqlens_int32, dim=0, dtype=torch.int32), (1, 0)
        )
        bs = cache_seqlens_int32.shape[0]
        # 分配合并后的分页表（宽度 = 公共前缀页数 + 新草稿 token 页数）
        page_table = (
            metadata.page_table.new_zeros(
                (bs, metadata.max_seq_len_k + metadata_expand.page_table.shape[1])
            )
            if metadata_swa is None
            else metadata_swa.page_table
        )

        # 调用 Triton Kernel 将公共前缀和新草稿 token 的页索引合并到 page_table
        prepare_swa_spec_page_table_triton(
            page_table,
            metadata.page_table,
            metadata_expand.page_table,
            metadata.cache_seqlens_int32,
            metadata_expand.cache_seqlens_int32,
            self.speculative_num_draft_tokens,
        )

        if metadata_swa is None:
            # 首次初始化：创建 SWA 专用元数据
            metadata_swa = FlashAttentionMetadata()
            metadata_swa.max_seq_len_q = 1
            metadata_swa.cu_seqlens_q = metadata_expand.cu_seqlens_q
            metadata_swa.cache_seqlens_int32 = cache_seqlens_int32
            metadata_swa.cu_seqlens_k = cu_seqlens_k
            metadata_swa.page_table = page_table
        else:
            # CUDA Graph replay：原地更新已有元数据中的可变字段
            metadata_swa.cache_seqlens_int32.copy_(cache_seqlens_int32)
            metadata_swa.cu_seqlens_k.copy_(cu_seqlens_k)

        # 将 SWA 投机解码元数据挂载到主元数据
        metadata.swa_spec_metadata = metadata_swa
