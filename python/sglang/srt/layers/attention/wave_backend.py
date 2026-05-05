# 允许注解中使用前向引用
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

# 导入注意力后端基类
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
# 导入 FlashInfer 风格的 KV 索引构建 Triton 内核
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
# 导入注意力张量并行尺寸获取函数
from sglang.srt.layers.dp_attention import get_attention_tp_size
# 导入前向批次信息和前向模式枚举
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
# 导入环境变量读取和设备 SM 核心数获取工具
from sglang.srt.utils import get_bool_env_var, get_device_core_count

if TYPE_CHECKING:
    # 仅类型检查时导入，避免循环依赖
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

logger = logging.getLogger(__name__)


@triton.jit
def get_num_kv_splits_triton(
    num_kv_splits_ptr,   # 输出：每条序列的 KV 分片数
    seq_lens_ptr,        # 输入：各序列的长度
    num_seq,             # 序列总数
    num_group,           # 每序列的 token 组数（通常为 1，verify 阶段为 num_draft_tokens）
    num_head,            # 注意力 Q 头数
    num_kv_head,         # KV 头数（GQA 时小于 num_head）
    max_kv_splits,       # KV 分片数的上限
    device_core_count,   # GPU SM 核心数
    MAX_NUM_SEQ: tl.constexpr,  # 编译时常量：序列数量的 2 次幂上界
):
    # TODO: this method is tunable, we need more online serving data to tune it
    # 根据序列长度分布动态确定 KV 分片数，平衡 SM 利用率和内存访问效率
    offs_seq = tl.arange(0, MAX_NUM_SEQ)
    mask_seq = offs_seq < num_seq

    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=0)
    max_seq_len = tl.max(seq_lens)
    # 重新加载时用 max_seq_len 填充越界位置，确保 min 计算正确
    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=max_seq_len)
    min_seq_len = tl.min(seq_lens)
    # 若最大最小序列长度差异不超过 25%，视为均匀分布，使用 max 代替 min
    if max_seq_len * 8 < min_seq_len * 10:
        min_seq_len = max_seq_len
    # 方案 1：基于 max/min 序列长度比值限制分片数
    max_kv_splits_1 = tl.minimum(tl.cdiv(max_seq_len, min_seq_len), max_kv_splits)
    kv_chunk_size_1 = tl.cdiv(max_seq_len, max_kv_splits_1)

    # NOTE: this is a hack to let num_kv_split grows up with seqlen gradually
    # 方案 2：根据序列长度对数增长调整有效 SM 核心数，确保分片数随序列长度平滑增长
    ext_seq_len = tl.cast(max_seq_len, tl.float32) / 64.0
    ext_device_core_count = tl.cast(
        device_core_count * tl.maximum(tl.log2(ext_seq_len), 1.0), tl.int32
    )
    block_h, num_kv_group = 16, num_head // num_kv_head
    if num_kv_group == 1:
        # MHA/GQA 中 num_kv_group=1 的路径：token_grid = 序列数 * 组数 * 头数
        token_grid = num_seq * num_group * num_head
    else:
        # from triton_ops/decode_attention.py:_decode_grouped_att_m_fwd
        # GQA 路径：block_h 限制每 tile 处理的 Q 头数
        block_h = tl.minimum(block_h, num_kv_group)
        token_grid = num_seq * num_group * tl.cdiv(num_head, block_h)
    max_kv_splits_2 = tl.minimum(
        tl.cdiv(ext_device_core_count, token_grid), max_kv_splits
    )
    kv_chunk_size_2 = tl.cdiv(max_seq_len, max_kv_splits_2)

    # 取两种方案的较大分片数（保守策略，确保不欠分）
    num_kv_splits = tl.maximum(
        tl.cdiv(seq_lens, kv_chunk_size_1), tl.cdiv(seq_lens, kv_chunk_size_2)
    )

    # 将每条序列的分片数写出到输出缓冲区（按 num_group 步长展开）
    offs_token = offs_seq * num_group
    mask_token = offs_token < num_seq * num_group
    for i in range(0, num_group):
        tl.store(num_kv_splits_ptr + i + offs_token, num_kv_splits, mask=mask_token)


@dataclass
class ForwardMetadata:
    # Wave 注意力后端的前向元数据，汇总 decode/extend 所需的所有辅助张量
    attn_logits: torch.Tensor    # decode 中间 logits 缓冲区
    attn_lse: torch.Tensor       # decode 中间 log-sum-exp 缓冲区
    max_extend_len: int          # extend 阶段最大序列长度
    num_kv_splits: torch.Tensor  # 每条序列的 KV 分片数
    kv_indptr: torch.Tensor      # KV 索引的前缀指针（CSR 格式）
    kv_indices: torch.Tensor     # KV 缓存物理块索引
    qo_indptr: torch.Tensor      # Q/O 索引的前缀指针（verify/extend 阶段）
    custom_mask: torch.Tensor    # 自定义注意力掩码（verify 阶段）
    mask_indptr: torch.Tensor    # 掩码索引的前缀指针


class WaveAttnBackend(AttentionBackend):
    # Wave 注意力后端：使用 Wave 语言编写的 decode/extend 内核，支持持续批处理优化
    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
    ):
        # Lazy import to avoid the initialization of cuda context
        # 延迟导入以避免过早初始化 CUDA 上下文
        from sglang.srt.layers.attention.wave_ops.decode_attention import (
            decode_attention_fwd,
        )
        from sglang.srt.layers.attention.wave_ops.extend_attention import (
            extend_attention_wave,
        )

        super().__init__()

        # Set unique cache dir for each process to avoid cache write races
        # 为每个 TP rank 设置独立的 Wave 内核缓存目录，避免并发写入冲突
        import wave_lang.kernel.wave.cache as cache

        base_cache_dir = cache.CACHE_BASE_DIR
        new_dir = base_cache_dir / f"worker_{model_runner.tp_rank}"
        logger.info(f"Setting Wave cache dir: {new_dir}")
        cache.CACHE_BASE_DIR = new_dir

        # 绑定 decode 和 extend 注意力内核
        self.decode_attention_fwd = decode_attention_fwd
        self.extend_attention_fwd = extend_attention_wave

        self.skip_prefill = skip_prefill  # 是否跳过 prefill 相关初始化

        max_bs = model_runner.req_to_token_pool.size  # 最大批次大小

        if kv_indptr_buf is None:
            # 分配 KV 前缀指针缓冲区（长度 max_bs + 1，CSR 格式）
            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
        else:
            self.kv_indptr = kv_indptr_buf

        self.req_to_token = model_runner.req_to_token_pool.req_to_token  # 请求到 token 的映射

        if not self.skip_prefill:
            # prefill/extend 阶段：分配 Q/O 前缀指针和掩码前缀指针
            self.qo_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )

            self.mask_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int64, device=model_runner.device
            )

        # 推测解码草稿 token 数
        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens

        # Q 头数和 KV 头数（考虑张量并行和 GQA）
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_head = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )

        # 是否使用静态 KV 分片（关闭动态调优，固定为 max_kv_splits）
        self.static_kv_splits = get_bool_env_var(
            "SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS", "false"
        )
        self.max_kv_splits = model_runner.server_args.triton_attention_num_kv_splits
        # Value head 维度
        self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[-1]

        self.forward_metadata: ForwardMetadata = None  # 前向元数据，每步更新

        self.max_context_len = model_runner.model_config.context_len  # 最大上下文长度

        self.device = model_runner.device
        # GPU SM 核心数（用于动态 KV 分片调优）
        self.device_core_count = get_device_core_count(model_runner.gpu_id)

    def get_num_kv_splits(
        self,
        num_kv_splits: torch.Tensor,
        seq_lens: torch.Tensor,
    ):
        # 动态计算每条序列的最优 KV 分片数（平衡 SM 利用率和内存带宽）
        num_token, num_seq = num_kv_splits.shape[0], seq_lens.shape[0]
        num_group = num_token // num_seq

        assert (
            num_group * num_seq == num_token
        ), f"num_seq({num_seq}), num_token({num_token}), something goes wrong!"

        if self.static_kv_splits or self.device_core_count <= 0:
            # 静态模式或无法获取核心数时，直接填充最大分片数
            num_kv_splits.fill_(self.max_kv_splits)
            return

        # 对 num_seq 向上取 2 的幂，作为 Triton 内核的 constexpr 参数
        if num_seq < 256:
            SCHEDULE_SEQ = 256
        else:
            SCHEDULE_SEQ = triton.next_power_of_2(num_seq)

        # 调用 Triton 内核计算每条序列的最优分片数
        get_num_kv_splits_triton[(1,)](
            num_kv_splits,
            seq_lens,
            num_seq,
            num_group,
            self.num_head,
            self.num_kv_head,
            self.max_kv_splits,
            self.device_core_count,
            MAX_NUM_SEQ=SCHEDULE_SEQ,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init auxiliary variables for wave attention backend."""
        # 初始化前向元数据：根据前向模式分别处理 decode/verify/draft_extend/extend 四种情况

        bs = forward_batch.batch_size
        kv_indptr = self.kv_indptr
        spec_info = forward_batch.spec_info

        if forward_batch.forward_mode.is_decode_or_idle():
            if spec_info is None:
                # 标准 decode：计算 KV 前缀指针和物理块索引
                kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                kv_indices = torch.empty(
                    forward_batch.seq_lens_sum, dtype=torch.int32, device=self.device
                )
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
            else:
                # 推测解码：直接使用 spec_info 中预计算的 KV 索引
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
                bs = kv_indptr.shape[0] - 1

            from sglang.srt.layers.attention.wave_ops.decode_attention import (
                decode_attention_intermediate_arrays_shapes,
            )

            # 计算 decode 中间缓冲区的形状
            attn_logits_shape, attn_logits_max_shape = (
                decode_attention_intermediate_arrays_shapes(
                    bs, self.v_head_dim, self.num_head, self.max_kv_splits
                )
            )
            # 分配 decode 中间缓冲区（logits 和 lse）
            attn_logits = torch.empty(
                attn_logits_shape,
                dtype=torch.float32,
                device=self.device,
            )
            attn_lse = torch.empty(
                attn_logits_max_shape,
                dtype=torch.float32,
                device=self.device,
            )
            num_kv_splits = torch.empty((bs,), dtype=torch.int32, device=self.device)

            # 动态计算 KV 分片数
            self.get_num_kv_splits(num_kv_splits, forward_batch.seq_lens)

            qo_indptr = None
            custom_mask = None
            mask_indptr = None
            max_extend_len = None
        elif forward_batch.forward_mode.is_target_verify():
            # verify 阶段：需要 qo_indptr（Q 按草稿 token 分组）和自定义掩码
            bs = len(forward_batch.req_pool_indices)
            # qo_indptr 按 num_draft_tokens 步长等差分布
            qo_indptr = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            # Different with flashinfer kv_indptr and kv_indices construction
            # 构建 KV 前缀指针（包含所有历史 KV，含草稿部分）
            kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                kv_indptr[-1], dtype=torch.int32, device=self.device
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            # 构建 verify 阶段的自定义掩码前缀指针
            custom_mask = spec_info.custom_mask
            seq_mask_len = self.num_draft_tokens * (
                forward_batch.seq_lens + self.num_draft_tokens
            )
            mask_indptr = self.mask_indptr
            mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len[:bs], dim=0)
            mask_indptr = mask_indptr[: bs + 1]
            max_extend_len = self.num_draft_tokens  # verify 阶段的 extend 长度等于草稿 token 数
            num_kv_splits = None
            attn_logits = None
            attn_lse = None
        elif forward_batch.forward_mode.is_draft_extend():
            # draft_extend 阶段：通过 spec_info 生成 prefill 格式的注意力参数
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    None,
                    self.req_to_token,
                )
            )
            mask_indptr = None
            # TODO(FIXME): This will trigger an invalid Eagle tree when using
            # `max(spec_info.num_accepted_tokens_cpu)`.
            # It might have been forgotten to update somewhere.
            # 使用实际接受的 token 数作为 max_extend_len
            max_extend_len = torch.max(spec_info.num_accepted_tokens).item()
            num_kv_splits = None
            attn_logits = None
            attn_lse = None
        else:
            # 标准 extend/prefill 阶段：构建前缀 KV 索引和 Q/O 前缀指针
            kv_indptr[1 : bs + 1] = torch.cumsum(
                forward_batch.extend_prefix_lens, dim=0
            )
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                forward_batch.extend_prefix_lens.sum().item(),
                dtype=torch.int32,
                device=self.device,
            )
            # 构建 extend 前缀部分的 KV 物理索引
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.extend_prefix_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            qo_indptr = self.qo_indptr
            qo_indptr[1 : bs + 1] = torch.cumsum(forward_batch.extend_seq_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
            custom_mask = None
            mask_indptr = None
            attn_logits = None
            attn_lse = None
            max_extend_len = torch.max(forward_batch.extend_seq_lens).item()
            num_kv_splits = None

        # 将所有元数据打包到 ForwardMetadata 数据类
        self.forward_metadata = ForwardMetadata(
            attn_logits,
            attn_lse,
            max_extend_len,
            num_kv_splits,
            kv_indptr,
            kv_indices,
            qo_indptr,
            custom_mask,
            mask_indptr,
        )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        # 初始化 CUDA Graph 所需的固定缓冲区
        from sglang.srt.layers.attention.wave_ops.decode_attention import (
            decode_attention_intermediate_arrays_shapes,
        )

        # 计算最大批次下的 decode 中间缓冲区形状
        attn_logits_shape, attn_logits_max_shape = (
            decode_attention_intermediate_arrays_shapes(
                max_bs, self.v_head_dim, self.num_head, self.max_kv_splits
            )
        )
        # 预分配固定大小的 decode 中间缓冲区（CUDA Graph 捕获后不可动态分配）
        self.cuda_graph_attn_logits = torch.zeros(
            attn_logits_shape,
            dtype=torch.float32,
            device=self.device,
        )
        self.cuda_graph_attn_lse = torch.zeros(
            attn_logits_max_shape,
            dtype=torch.float32,
            device=self.device,
        )
        # 预分配 KV 分片数缓冲区（初始值为 max_kv_splits）
        self.cuda_graph_num_kv_splits = torch.full(
            (max_bs,), self.max_kv_splits, dtype=torch.int32, device=self.device
        )
        if kv_indices_buf is None:
            # 预分配 KV 索引缓冲区（最大尺寸 = max_bs * max_context_len）
            self.cuda_graph_kv_indices = torch.zeros(
                (max_bs * self.max_context_len),
                dtype=torch.int32,
                device=self.device,
            )
        else:
            self.cuda_graph_kv_indices = kv_indices_buf

        if not self.skip_prefill:
            # verify 阶段需要预分配自定义掩码缓冲区
            self.cuda_graph_custom_mask = torch.zeros(
                (max_bs * self.max_context_len),
                dtype=torch.uint8,
                device=self.device,
            )

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
        # CUDA Graph 捕获阶段：填充所有必要缓冲区并构建 ForwardMetadata
        assert encoder_lens is None, "Not supported"

        if forward_mode.is_decode_or_idle():
            if spec_info is None:
                # 标准 decode：计算 KV 前缀指针并填充预分配的 KV 索引缓冲区
                kv_indptr = self.kv_indptr
                kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                kv_indices = self.cuda_graph_kv_indices  # 使用预分配缓冲区（地址固定）
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices,
                    seq_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
            else:
                # 推测解码：使用 spec_info 中的 KV 索引
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices

            attn_logits = self.cuda_graph_attn_logits   # 使用预分配 logits 缓冲区
            attn_lse = self.cuda_graph_attn_lse          # 使用预分配 lse 缓冲区
            max_extend_len = None
            num_kv_splits = self.cuda_graph_num_kv_splits  # 使用预分配分片数缓冲区
            qo_indptr = None
            custom_mask = None
            mask_indptr = None
        elif forward_mode.is_target_verify():
            # verify 阶段：构建 qo_indptr、KV 索引和掩码
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            custom_mask = self.cuda_graph_custom_mask
            seq_mask_len = self.num_draft_tokens * (seq_lens + self.num_draft_tokens)
            mask_indptr = self.mask_indptr[: bs + 1]
            mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len, dim=0)
            max_extend_len = self.num_draft_tokens
            num_kv_splits = None
            attn_logits = None
            attn_lse = None
        else:
            # 其他前向模式在 CUDA Graph 捕获阶段不支持
            raise ValueError(
                f"Invalid forward mode: {forward_mode=} for CUDA Graph capture."
            )

        self.forward_metadata = ForwardMetadata(
            attn_logits,
            attn_lse,
            max_extend_len,
            num_kv_splits,
            kv_indptr,
            kv_indices,
            qo_indptr,
            custom_mask,
            mask_indptr,
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
        # NOTE: encoder_lens expected to be zeros or None
        # CUDA Graph 回放阶段：原地更新动态变化的 KV 索引等缓冲区
        if forward_mode.is_decode_or_idle():
            # Update kv_indptr, kv_indices
            # 更新 KV 前缀指针和物理索引（seq_lens 每步变化）
            kv_indptr = self.kv_indptr
            kv_indices = self.cuda_graph_kv_indices
            num_kv_splits = self.cuda_graph_num_kv_splits
            if spec_info is None:
                kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens[:bs], dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices[:bs],
                    seq_lens[:bs],
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
                num_token = bs
            else:
                # 推测解码：直接将 spec_info 的索引写入预分配缓冲区
                kv_indptr[: spec_info.kv_indptr.shape[0]] = spec_info.kv_indptr
                kv_indices[: spec_info.kv_indices.shape[0]] = spec_info.kv_indices
                num_token = spec_info.kv_indptr.shape[0] - 1
            # 重新计算 KV 分片数（seq_lens 变化后需重新调优）
            self.get_num_kv_splits(num_kv_splits[:num_token], seq_lens[:bs])
        elif forward_mode.is_target_verify():
            # Update qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr
            # 更新 verify 阶段的所有动态缓冲区
            bs = len(req_pool_indices)
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
            # 将 spec_info 中的自定义掩码写入预分配缓冲区
            custom_mask = self.cuda_graph_custom_mask
            custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask
            seq_mask_len = self.num_draft_tokens * (seq_lens + self.num_draft_tokens)
            mask_indptr = self.mask_indptr[: bs + 1]
            mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len, dim=0)
        else:
            raise ValueError(
                f"Invalid forward mode: {forward_mode=} for CUDA Graph replay."
            )

    def get_cuda_graph_seq_len_fill_value(self):
        # CUDA Graph 序列长度填充值为 1
        return 1

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # TODO: reuse the buffer across layers
        # extend/prefill 阶段：调用 Wave extend 内核完成注意力计算
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

        max_extend_len = self.forward_metadata.max_extend_len
        computed_max_ext_seq_len = torch.max(forward_batch.extend_seq_lens)
        if computed_max_ext_seq_len != max_extend_len:
            # CUDA Graph 场景：实际 extend 长度可能小于捕获时的最大长度，需要对齐
            assert len(forward_batch.extend_seq_lens) == 1
            forward_batch.extend_seq_lens[0] = max_extend_len
            forward_batch.seq_lens = max_extend_len

        # 调用 Wave extend 内核（支持 prefill、verify 和 draft_extend 三种 extend 子模式）
        self.extend_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k.contiguous(),
            v.contiguous(),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            self.forward_metadata.qo_indptr,
            self.forward_metadata.kv_indptr,
            self.forward_metadata.kv_indices,
            self.forward_metadata.custom_mask,   # None 表示标准因果掩码
            self.forward_metadata.mask_indptr,   # None 表示无自定义掩码前缀指针
            self.forward_metadata.max_extend_len,
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            is_causal=True,
            layer_scaling=layer.scaling,
            logit_cap=layer.logit_cap,
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
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        # 修复 torch.compile 下 rotary_emb 产生 3D 输出的问题
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        # TODO: reuse the buffer across layers
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if save_kv_cache:
            # 写入当前 token 的 KV 到缓存
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        # 调用 Wave decode 内核（支持 KV 分片并行和 GQA）
        self.decode_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            self.forward_metadata.kv_indptr,     # KV 前缀指针
            self.forward_metadata.kv_indices,    # KV 物理块索引
            self.forward_metadata.attn_logits,   # 中间 logits 缓冲区
            self.forward_metadata.attn_lse,      # 中间 lse 缓冲区
            self.forward_metadata.num_kv_splits, # 每条序列的 KV 分片数
            self.max_kv_splits,
            layer.scaling,
            layer.logit_cap,
        )
        return o
