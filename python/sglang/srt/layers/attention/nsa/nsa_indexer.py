# NSA Indexer 模块：负责 NSA 稀疏注意力的 top-k block 索引计算
# 支持 CUDA（DeepGEMM）、HIP（aiter/ROCm）、NPU（Ascend）三大平台
# 核心功能：Q/K 投影、FP8 量化、paged/ragged MQA logits、top-k 稀疏索引选择
from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from einops import rearrange

# 融合存储 K-cache 的 JIT 内核（CUDA 专用，page_size=64）
from sglang.jit_kernel.fused_store_index_cache import (
    can_use_nsa_fused_store,
    fused_store_index_k_cache,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import attn_tp_all_gather_into_tensor
from sglang.srt.layers.layernorm import LayerNorm
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype, is_fp8_fnuz
from sglang.srt.layers.utils import MultiPlatformOp
from sglang.srt.utils import (
    add_prefix,
    ceil_align,
    get_bool_env_var,
    is_cuda,
    is_gfx95_supported,
    is_hip,
    is_npu,
)

global _use_multi_stream
# 平台检测标志：决定使用哪条代码路径
_is_cuda = is_cuda()   # NVIDIA CUDA 平台
_is_hip = is_hip()     # AMD ROCm/HIP 平台
_is_npu = is_npu()     # 华为昇腾 NPU 平台
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip  # 是否使用 aiter（ROCm gfx95 优化）
_is_fp8_fnuz = is_fp8_fnuz()          # 是否使用 FP8 FNUZ 格式（AMD 特有）
_is_gfx95_supported = is_gfx95_supported()  # 是否支持 gfx95 架构（AMD MI300X 等）
if _is_cuda:
    try:
        import deep_gemm   # NVIDIA DeepGEMM：高效 FP8 矩阵乘法库
    except ImportError as e:
        deep_gemm = e      # 导入失败时保留异常对象，运行时检查

if _use_aiter:
    # aiter 提供 ROCm 平台的 indexer K-cache 融合量化+存储 kernel
    from aiter.ops.cache import indexer_k_quant_and_cache

if is_npu():
    # 华为昇腾 NPU 专用算子和工具
    import torch_npu
    from sglang.srt.hardware_backend.npu.utils import get_indexer_weight_stream

# 分布式并行工具：context parallel（CP）rank/world_size 查询
from sglang.srt.distributed import (
    get_attn_context_model_parallel_rank,
    get_attn_context_model_parallel_world_size,
)
from sglang.srt.distributed.parallel_state import get_pp_group
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.attention.nsa.utils import (
    is_nsa_enable_prefill_cp,       # 是否开启 prefill context parallel
    is_nsa_prefill_cp_in_seq_split, # CP 是否按序列维度切分
)
from sglang.srt.layers.communicator import ScatterMode
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.layers.utils.cp_utils import cp_all_gather_rerange_output
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args

_use_ag_after_qlora = envs.SGLANG_USE_AG_AFTER_QLORA.get()  # 是否在 qLoRA 后做 all-gather
if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool


# CUDA 平台双流优化阈值：超过 1024 token 时启用双流并行（HIP 不启用）
DUAL_STREAM_TOKEN_THRESHOLD = 1024 if _is_cuda else 0


class BaseIndexerMetadata(ABC):
    """NSA Indexer 元数据抽象基类：定义各平台共用的接口，由具体 backend 实现。"""

    @abstractmethod
    def get_seqlens_int32(self) -> torch.Tensor:
        """
        Return: (batch_size,) int32 tensor
        """
        # 返回每个请求的 KV 序列长度（int32），用于 paged MQA logits 计算

    @abstractmethod
    def get_page_table_64(self) -> torch.Tensor:
        """
        Return: (batch_size, num_blocks) int32, page table.
                The page size of the table is 64.
        """
        # 返回 page_size=64 的分页表（CUDA 用）：每行为一个请求的物理 KV block 索引

    @abstractmethod
    def get_page_table_1(self) -> torch.Tensor:
        """
        Return: (batch_size, num_blocks) int32, page table.
                The page size of the table is 1.
        """
        # 返回 page_size=1 的分页表（HIP 用）：每行为一个 token 的物理 KV slot 索引

    @abstractmethod
    def get_seqlens_expanded(self) -> torch.Tensor:
        """
        Return: (sum_extend_seq_len,) int32 tensor
        """
        # 返回 extend 场景下展开后的序列长度张量（用于 ragged MQA logits）

    def get_indexer_kvcache_range(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return: (tokens, ), (tokens, ) int32, k_start and k_end in kv cache(token,xxx) for each token.
        """
        # 返回每个 token 在 KV cache 中对应的起始和终止位置（用于 ragged 路径边界计算）

    def get_indexer_seq_len_cpu(self) -> torch.Tensor:
        """
        Return: seq lens for each batch.
        """
        # 返回 CPU 上的每个请求序列长度（避免 GPU-CPU 同步）

    def get_indexer_seq_len(self) -> torch.Tensor:
        """
        Return: seq lens for each batch.
        """
        # 返回 GPU 上的每个请求序列长度张量

    def get_nsa_extend_len_cpu(self) -> List[int]:
        """
        Return: extend seq lens for each batch.
        """
        # 返回 extend 阶段每个请求实际扩展的 token 数（Python list，用于 q_offset 计算）

    def get_token_to_batch_idx(self) -> torch.Tensor:
        """
        Return: batch idx for each token.
        """
        # 返回每个 token 对应的 batch 序号（用于 chunk 分块 topk 转换时的 batch 映射）

    @abstractmethod
    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        """
        Perform topk selection on the logits and possibly transform the result.

        NOTE that attention backend may override this function to do some
        transformation, which means the result of this topk_transform may not
        be the topk indices of the input logits.

        Return: Anything, since it will be passed to the attention backend
                for further processing on sparse attention computation.
                Don't assume it is the topk indices of the input logits.
        """
        # 对 logits 执行 top-k 选取，并可能做进一步变换（如 block 坐标转换）
        # 返回值将被传递给 attention backend 用于稀疏注意力计算，不保证是原始 logits 的 topk 索引


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """对激活值做 Hadamard 旋转变换，增强 FP8 量化的数值分布均匀性。"""
    assert x.dtype == torch.bfloat16
    # from sgl_kernel import hadamard_transform
    if _is_hip:
        # HIP 平台使用 fast_hadamard_transform 库
        from fast_hadamard_transform import hadamard_transform
    else:
        # CUDA 平台使用 SGLang JIT 版 Hadamard kernel
        from sglang.jit_kernel.hadamard import hadamard_transform

    hidden_size = x.size(-1)
    assert (
        hidden_size & (hidden_size - 1)
    ) == 0, "Hidden size must be a power of 2 for Hadamard transform."
    # 缩放因子 = 1/sqrt(hidden_size)，保持 L2 范数不变
    return hadamard_transform(x, scale=hidden_size**-0.5)


class Indexer(MultiPlatformOp):
    """NSA 稀疏注意力 Indexer：负责计算每个 token 应关注哪些 KV block（top-k 稀疏索引）。

    包含以下子模块：
      - wq_b: Q LoRA 升维投影（q_lora → query）
      - wk: K 投影（hidden → key）
      - weights_proj: 每 head 门控权重投影（hidden → n_heads float32）
      - k_norm: K 的 LayerNorm（稳定数值）
      - rotary_emb: RoPE 旋转位置编码
    """

    def __init__(
        self,
        hidden_size: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        index_topk: int,
        q_lora_rank: int,
        max_position_embeddings: int,
        rope_theta: float,
        layer_id: int,
        scale_fmt: Optional[str],
        block_size: int = 128,          # FP8 量化 block 大小（K cache 每 block 128 token）
        rope_scaling: Optional[Dict[str, Any]] = None,
        is_neox_style: bool = True,
        prefix: str = "",
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,  # 双流优化用辅助 CUDA stream
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = index_n_heads         # indexer 使用的 Q head 数（通常 64）
        self.head_dim = index_head_dim       # 每个 head 的维度（rope + nope 合计，通常 128）
        self.rope_head_dim = rope_head_dim   # RoPE 部分维度（通常 64）
        self.index_topk = index_topk         # 选取的稀疏 block 数（top-k）
        self.q_lora_rank = q_lora_rank       # Q LoRA 低秩维度
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()  # 是否开启 prefill CP 并行
        if self.nsa_enable_prefill_cp:
            # 获取 context parallel 的 world_size 和 rank
            self.cp_size = get_attn_context_model_parallel_world_size()
            self.cp_rank = get_attn_context_model_parallel_rank()
        else:
            self.cp_size = None
            self.cp_rank = None
        if _is_cuda:
            # 获取 GPU SM 总数，用于 DeepGEMM 调度优化
            self.sm_count = deep_gemm.get_num_sms()
            # 半数 SM（向上对齐到 8），用于双流时各占一半 SM
            self.half_device_sm_count = ceil_align(self.sm_count // 2, 8)
            pp_size = get_global_server_args().pp_size
            # PP 非最后 rank 需要同时 recv，占用 1 个 SM
            self.logits_with_pp_recv = pp_size > 1 and not get_pp_group().is_last_rank
        else:
            self.logits_with_pp_recv = False

        # Q 投影：q_lora_rank → n_heads * head_dim（复制权重，不做张量并行）
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
        )

        # K 投影：hidden_size → head_dim（MQA 风格，单 head）
        self.wk = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wk", prefix),
        )
        # 门控权重投影：hidden_size → n_heads（float32 输出，不做量化）
        self.weights_proj = ReplicatedLinear(
            self.hidden_size,
            self.n_heads,
            bias=False,
            params_dtype=torch.bfloat16,
            prefix=add_prefix("weights_proj", prefix),
        )
        # K-cache LayerNorm：在 HIP 上用 BF16，CUDA 上用 FP32
        self.k_norm = LayerNorm(
            self.head_dim, dtype=torch.bfloat16 if _use_aiter else torch.float32
        )
        # RoPE 旋转位置编码（仅作用于 rope_head_dim 维度）
        self.rotary_emb = get_rope_wrapper(
            rope_head_dim,
            rotary_dim=rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,  # type: ignore
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
            device=get_global_server_args().device,
        )
        self.block_size = block_size     # FP8 量化 tile 大小（128）
        self.scale_fmt = scale_fmt       # FP8 scale 格式（e4m3fn 或 fnuz）
        self.softmax_scale = self.head_dim**-0.5  # 注意力缩放因子 = 1/sqrt(head_dim)

    @contextlib.contextmanager
    def _with_real_sm_count(self):
        """PP 并行场景下动态减少 DeepGEMM 可用 SM 数，避免与 PP recv 操作争抢 SM。

        Pipeline Parallelism 非最后 rank 在执行前向时，同时有一个 recv 操作占用 1 个 SM，
        因此需要将 DeepGEMM 的调度 SM 数减 1。
        """
        # When pipeline parallelism is enabled, each PP rank initiates a recv operation after the _pp_launch_batch
        # request to receive the PP proxy tensor or output from the previous stage, occupying one SM resource.
        # Model execution runs in parallel with the recv operation, so the SMs available to the indexer must be reduced
        # by 1. Currently, the last rank starts the send result + recv request only after waiting for execution results.
        if self.logits_with_pp_recv:
            pp_recv_sm_count = 1  # PP recv 操作占用 1 个 SM
            with deep_gemm_wrapper.configure_deep_gemm_num_sms(
                self.sm_count - pp_recv_sm_count  # 减去 recv 占用的 SM 数
            ):
                yield
        else:
            yield  # 无 PP recv，直接使用全部 SM

    def _weights_proj_bf16_in_fp32_out(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ) -> torch.Tensor:
        """计算门控权重投影，输出 float32（或 BF16）。

        ROCm gfx95 aiter 路径：从 fused_rms_fp8_group_quant 的 (fp8, scale, bf16) 3-tuple
        中直接取出 BF16 激活值，避免 FP8 反量化开销。
        CUDA DeepGEMM 路径：使用 bf16bf16f32 GEMM，输出 float32。
        """
        # aiter (ROCm gfx95): extract the passthrough bf16 tensor from the
        # 3-tuple (fp8, scale, bf16) produced by fused_rms_fp8_group_quant,
        # avoiding an expensive FP8-to-bf16 dequantization.
        if _use_aiter and _is_gfx95_supported and isinstance(x, tuple) and len(x) == 3:
            # 取第 3 个元素（BF16 passthrough）
            x = x[2]
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            # CUDA DeepGEMM 路径：BF16×BF16→FP32 矩阵乘
            weight = self.weights_proj.weight
            out = torch.empty(
                (x.shape[0], weight.shape[0]),
                dtype=torch.float32,
                device=x.device,
            )
            deep_gemm_wrapper.gemm_nt_bf16bf16f32(x, weight, out)
            return out

        # 标准路径：调用 weights_proj 线性层
        weights, _ = self.weights_proj(x)
        if _is_hip:
            # HIP 平台返回 BF16（q_scale 乘法会自动提升回 fp32）
            return weights
        return weights.float()  # CUDA 平台升精度到 float32

    @torch.compile(dynamic=True)
    def _project_and_scale_head_gates(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ):
        """计算 head 门控权重并缩放（decode 双流路径专用）。

        输出 = weights_proj(x) * (1/sqrt(n_heads))
        用于 decode 模式下提前计算门控权重，与 K-cache 存储并行执行。
        """
        weights = self._weights_proj_bf16_in_fp32_out(x)
        weights = weights * self.n_heads**-0.5  # 缩放因子 = 1/sqrt(n_heads)
        return weights

    @torch.compile(dynamic=True)
    def _get_logits_head_gate(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, ...]], q_scale: torch.Tensor
    ):
        """计算带 q_scale 融合的门控权重（extend 路径）。

        输出 = weights_proj(x) * (1/sqrt(n_heads)) * q_scale * softmax_scale
        将门控权重与 Q 的 FP8 量化缩放因子及注意力缩放一次性融合，减少计算步骤。
        """
        weights = self._weights_proj_bf16_in_fp32_out(x)
        weights = weights * self.n_heads**-0.5              # 门控缩放
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale  # 融合 FP8 scale 和 softmax scale
        return weights

    def _get_q_k_bf16(
        self,
        q_lora: torch.Tensor,   # Q LoRA 低秩激活 [tokens, q_lora_rank]
        x: torch.Tensor,         # 隐状态（可能是 tuple）[tokens, hidden_size]
        positions: torch.Tensor, # token 位置 id（用于 RoPE）
        enable_dual_stream: bool, # 是否启用双流并行（Q 和 K 分别在两个 stream 上计算）
        forward_batch: ForwardBatch,
    ):
        """计算 BF16 格式的 Q 和 K，并应用 RoPE + Hadamard 旋转。

        双流模式：Q 在主 stream 上计算，K 在 alt_stream 上并行计算，
        利用 GPU 双引擎并发提升吞吐量。
        CP 模式：K 计算完后立即做 all-gather，聚合所有 CP rank 的 K。
        """
        if enable_dual_stream:
            # 双流并行：主 stream 先等待 alt_stream 的历史操作完成
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)

            # 主 stream 上计算 Q（仅使用半数 SM，给 K 计算留 SM）
            with deep_gemm_wrapper.configure_deep_gemm_num_sms(
                self.half_device_sm_count
            ):
                query, _ = self.wq_b(q_lora)  # [tokens, n_heads * head_dim]
                query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
                # 分离 rope 部分和 nope 部分
                q_rope, _ = torch.split(
                    query,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )
            with torch.cuda.stream(self.alt_stream):
                # alt_stream 上计算 K（TODO: 是否也应限制半数 SM？）
                key, _ = self.wk(x)
                key = self.k_norm(key)  # LayerNorm

                k_rope, _ = torch.split(
                    key,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )

            # 等待 alt_stream 完成 K 计算
            current_stream.wait_stream(self.alt_stream)
        else:
            # 单流路径：顺序计算 Q 和 K
            query, _ = self.wq_b(q_lora)
            query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
            q_rope, _ = torch.split(
                query, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
            )
            key, _ = self.wk(x)
            key = self.k_norm(key)
            k_rope, _ = torch.split(
                key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
            )

        # 对 Q 和 K 的 rope 部分应用旋转位置编码（RoPE）
        q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)

        # 将 RoPE 结果写回 Q/K 的 rope 子区间
        self._update_rope_guarded(query[..., : self.rope_head_dim], q_rope)
        self._update_rope_guarded(key[..., : self.rope_head_dim], k_rope)

        if enable_dual_stream:
            # 双流模式：Q 和 K 分别在主/辅 stream 上并行做 Hadamard 旋转
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            query = rotate_activation(query)  # 主 stream 处理 Q

            with torch.cuda.stream(self.alt_stream):
                key = rotate_activation(key)  # alt_stream 处理 K
            current_stream.wait_stream(self.alt_stream)
        elif (
            self.alt_stream is not None
            and forward_batch.attn_cp_metadata is not None
            and self.nsa_enable_prefill_cp
        ):
            # CP 模式（双 stream + context parallel）：K 在辅 stream 上 Hadamard 后立即 all-gather
            key = rotate_activation(key)
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            query = rotate_activation(query)  # 主 stream 处理 Q

            with torch.cuda.stream(self.alt_stream):
                # alt_stream 上对 K 做 all-gather，将所有 CP rank 的 K 聚合
                key = cp_all_gather_rerange_output(
                    key.contiguous(),
                    self.cp_size,
                    forward_batch,
                    torch.cuda.current_stream(),
                )
            current_stream.wait_stream(self.alt_stream)
            return query, key  # CP 路径：K 已包含所有 rank 数据，直接返回
        else:
            # 单流标准路径：Q 和 K 顺序做 Hadamard 旋转
            query = rotate_activation(query)
            key = rotate_activation(key)

        # allgather+rerrange
        # 标准 CP 路径（无双流）：对 K 做 all-gather 聚合各 CP rank 数据
        if forward_batch.attn_cp_metadata is not None and self.nsa_enable_prefill_cp:
            key = cp_all_gather_rerange_output(
                key.contiguous(),
                self.cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )
        return query, key

    def _get_k_bf16(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        enable_dual_stream: bool,
    ):
        """仅计算 K（跳过 Q），用于不需要 top-k logits 的快速存 K-cache 路径。

        例如：序列长度 <= index_topk 时，直接选取所有 block，无需计算 logits。
        """
        # Compute only key, skip query
        key, _ = self.wk(x)            # hidden → key [tokens, head_dim]
        key = self.k_norm(key)         # LayerNorm 稳定 K 数值
        k_rope, _ = torch.split(
            key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )

        # 用 k_rope 作为占位的 q_rope（rotary_emb 仅用 k_rope 的输出）
        _, k_rope = self.rotary_emb(positions, k_rope, k_rope)
        # 将 RoPE 结果写回 key 的 rope 子区间
        self._update_rope_guarded(key[..., : self.rope_head_dim], k_rope)
        # 对 K 做 Hadamard 旋转（与存入 FP8 K-cache 前的激活旋转保持一致）
        key = rotate_activation(key)

        return key

    @staticmethod
    def _update_rope_guarded(dst: torch.Tensor, src: torch.Tensor) -> None:
        """安全地将 RoPE 旋转结果写回源张量，防止 AMD 就地 RoPE kernel 的自别名问题。

        AMD 某些 in-place RoPE 实现中，src 和 dst 指向相同内存时写回会造成数据损坏，
        检查到自别名时跳过写回操作。
        """
        # On AMD with in-place RoPE kernels, self-aliasing can occur;
        # skip write-back when src/dst tensors point to a single memory.
        if src.data_ptr() == dst.data_ptr():
            return  # 源与目标相同，无需拷贝
        dst.copy_(src)  # 将 RoPE 旋转后的 rope 部分写回 Q/K 张量

    def _get_topk_paged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,     # FP8 量化的 Q [tokens, n_heads, head_dim_with_sf]
        weights: torch.Tensor,   # 融合了 scale 的门控权重 [tokens, n_heads, 1]
        metadata: BaseIndexerMetadata,
    ) -> torch.Tensor:
        """使用分页 KV cache 计算 MQA logits，并返回 top-k block 索引。

        decode/target_verify/draft_extend 模式专用路径：
        - CUDA：调用 DeepGEMM fp8_paged_mqa_logits（page_size=64）
        - HIP：调用 aiter deepgemm_fp8_paged_mqa_logits（page_size=1）
        """
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)

        page_size = forward_batch.token_to_kv_pool.page_size
        # NOTE(dark): blocksize = 64 is hardcoded in deep_gemm
        if _is_hip:
            assert page_size == 1, "only support page size 1"
            block_tables = metadata.get_page_table_1()   # HIP: page_size=1 的分页表
        else:
            assert page_size == 64, "only support page size 64"
            # NOTE(dark): this support extend/decode/decode+graph
            block_tables = metadata.get_page_table_64()  # CUDA: page_size=64 的分页表

        # 最大序列长度 = 分页表列数 × 每页 token 数
        max_seq_len = block_tables.shape[1] * page_size
        # 从 KV pool 获取 FP8 格式的 K-cache（带 scale，每 token 共 132 字节）
        kv_cache_fp8 = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=layer_id
        )

        blocksize = page_size
        # MTP 场景（target_verify/draft_extend）使用展开后的序列长度
        if (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend(include_v2=True)
        ):
            seqlens_32 = metadata.get_seqlens_expanded()  # MTP：展开后的序列长度
        else:
            seqlens_32 = metadata.get_seqlens_int32()     # 标准 decode：原始序列长度
        # Reuse pre-computed schedule metadata if available (from init_forward_metadata),
        # otherwise fall back to computing it here.
        # CUDA graph 回放时复用预计算的 DeepGEMM 调度元数据（避免重复计算）
        schedule_metadata = getattr(metadata, "paged_mqa_schedule_metadata", None)
        if _is_cuda:
            if schedule_metadata is None:
                # 实时计算 paged MQA 的 DeepGEMM 调度参数
                schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                    seqlens_32, blocksize, self.sm_count
                )

        assert len(q_fp8.shape) == 3
        # 插入 next_n 维度（decode 时恒为 1），使形状为 [bs, 1, n_heads, head_dim_with_sf]
        q_fp8 = q_fp8.unsqueeze(1)  # the next_n dim is 1 now
        assert len(kv_cache_fp8.shape) == 2
        # HIP page_size=1，CUDA page_size=64
        block_kv = 1 if _is_hip else 64
        num_heads_kv = 1       # MQA 风格：K/V 只有 1 个 head
        head_dim_with_sf = 132  # FP8 格式 head_dim：128 + 4 字节 scale
        if _is_hip:
            # HIP：将 K-cache 重塑为 (total_tokens, 1, 1, 132)
            kv_cache_fp8 = kv_cache_fp8.view(
                -1, block_kv, num_heads_kv, head_dim_with_sf
            )
        else:
            # CUDA：将 K-cache 重塑为 (num_pages, 64, 1, 132)
            kv_cache_fp8 = kv_cache_fp8.view(
                kv_cache_fp8.shape[0], block_kv, num_heads_kv, head_dim_with_sf
            )
        assert len(weights.shape) == 3
        weights = weights.squeeze(2)  # 去掉末尾的 1 维，变为 [tokens, n_heads]

        # When attn_tp_size > 1 or in the MAX_LEN padding mode, padding may exist in the hidden states,
        # and it is necessary to extract the actual q length.
        # 计算实际有效的 Q token 数（去掉 padding）
        q_offset = sum(metadata.get_nsa_extend_len_cpu())
        if _is_hip:
            # HIP 路径：调用 aiter 的 FP8 paged MQA logits kernel
            from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits

            batch_size, next_n, heads, _ = q_fp8.shape
            # 初始化 logits 矩阵为 -inf（后续只填充有效位置）
            logits = torch.full(
                (batch_size * next_n, max_seq_len),
                float("-inf"),
                device=q_fp8.device,
                dtype=torch.float32,
            )
            deepgemm_fp8_paged_mqa_logits(
                q_fp8,
                kv_cache_fp8,
                weights,
                logits,
                seqlens_32,
                block_tables,
                max_seq_len,
                Preshuffle=False,
                KVBlockSize=block_kv,
            )
        else:
            # CUDA 路径：调用 DeepGEMM fp8_paged_mqa_logits
            logits = deep_gemm.fp8_paged_mqa_logits(
                q_fp8[:q_offset],      # 仅取有效 Q token
                kv_cache_fp8,
                weights[:q_offset],    # 对应的门控权重
                seqlens_32,
                block_tables,
                schedule_metadata,
                max_seq_len,
                clean_logits=False,    # logits 清零由 topk_transform 处理
            )

        # NOTE(dark): logits should be cleaned in topk_transform
        # 对 logits 执行 top-k 选取（并可能做坐标变换）
        topk_result = metadata.topk_transform(logits, self.index_topk)
        # Restore possible padding exist in the hidden states.
        # 如果存在 padding，补充 -1（无效索引）
        if not _is_hip and q_offset < q_fp8.shape[0]:
            pad_len = q_fp8.shape[0] - q_offset
            padding = torch.full(
                (pad_len, topk_result.shape[1]),
                -1,
                dtype=topk_result.dtype,
                device=topk_result.device,
            )
            topk_result = torch.cat([topk_result, padding], dim=0)
        return topk_result

    def _should_chunk_mqa_logits(
        self, num_q: int, num_k: int, device: torch.device
    ) -> Tuple[bool, int]:
        """检测是否需要分块计算 MQA logits（防止 OOM）。

        当 logits 矩阵超过显存阈值时，启用分块（chunk）路径，每次只计算一部分 Q token。
        Return: (need_chunk, free_mem)
        """
        # Quick static check for normal batches
        # 快速静态检查：8M 元素以内（约 32MB float32）不需要分块
        if num_q * num_k < 8_000_000:  # 8M elements ≈ 32MB logits
            return False, 0

        # 查询当前 GPU 可用显存和总显存
        free_mem, total_mem = torch.cuda.mem_get_info(device)
        bytes_per_elem = 4  # float32 每元素 4 字节
        logits_bytes = num_q * num_k * bytes_per_elem

        # Logits should not exceed 50% of free memory or 30% of total memory
        # logits 不应超过 50% 可用显存或 30% 总显存，否则触发分块
        need_chunk = (logits_bytes * 2 > free_mem) or (logits_bytes > total_mem * 0.3)
        return need_chunk, free_mem

    def _get_topk_ragged(
        self,
        enable_dual_stream: bool,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,     # FP8 量化的 Q [tokens, n_heads, 132]
        weights: torch.Tensor,   # 门控权重 [tokens, n_heads, 1]
        metadata: BaseIndexerMetadata,
    ) -> torch.Tensor:
        """使用 ragged（不规则长度）布局计算 MQA logits，返回 top-k block 索引。

        prefill/extend 模式专用路径：每个序列的 KV cache 长度不同，需要 ragged 格式处理。
        支持 OOM 保护的分块（chunk）路径，防止超长序列导致显存溢出。
        """
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)

        assert forward_batch.forward_mode.is_extend_without_speculative()

        page_size = forward_batch.token_to_kv_pool.page_size
        if _is_hip:
            assert page_size == 1, "only support page size 1"
        else:
            assert page_size == 64, "only support page size 64"

        assert len(weights.shape) == 3
        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )
        weights = weights.squeeze(-1)  # 去掉末尾 1 维：[tokens, n_heads]

        # 根据平台选取对应 page_size 的分页表
        if _is_hip:
            block_tables = metadata.get_page_table_1()   # HIP: page_size=1
        else:
            block_tables = metadata.get_page_table_64()  # CUDA: page_size=64

        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )

        batch_size = len(block_tables)
        token_nums, _, _ = q_fp8.shape
        device = q_fp8.device

        # 初始化 topk 结果为 -1（表示无效索引）
        topk_result = torch.full(
            (token_nums, self.index_topk), -1, device=device, dtype=torch.int32
        )
        if batch_size == 0:
            return topk_result

        # 获取每个 token 在 KV cache 中的起始(ks)和终止位置(ke)
        ks, ke = metadata.get_indexer_kvcache_range()

        indexer_seq_lens_cpu = metadata.get_indexer_seq_len_cpu()
        seq_len_sum = torch.sum(indexer_seq_lens_cpu).item()    # 所有序列长度之和
        max_seq_len = torch.max(indexer_seq_lens_cpu).item()    # 最大序列长度
        # 从 KV pool 获取 FP8 格式的 K（含 scale），形状为 (seq_len_sum, head_dim_fp8)
        k_fp8, k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_buffer(
            layer_id,
            metadata.get_indexer_seq_len(),
            block_tables,
            seq_len_sum,
            max_seq_len,
        )
        # 转换 FP8 dtype（FNUZ 格式用于 AMD，标准 E4M3 用于 NVIDIA）
        if _is_fp8_fnuz:
            k_fp8 = k_fp8.view(torch.float8_e4m3fnuz)
        else:
            k_fp8 = k_fp8.view(torch.float8_e4m3fn)

        k_scale = k_scale.view(torch.float32).squeeze(-1)  # K 的 absmax scale [seq_len_sum]
        kv_fp8 = (k_fp8, k_scale)

        # Check if we need to chunk to avoid OOM
        # 获取展开后的序列长度（每个 Q token 对应的历史 KV 长度）
        seq_lens_expanded = metadata.get_seqlens_expanded()
        token_to_batch_idx = metadata.get_token_to_batch_idx()  # 每个 token 的 batch 序号
        q_offset = ks.shape[0]       # 有效 Q token 数（去掉 padding）
        k_offset = k_fp8.shape[0]    # K-cache token 总数
        need_chunk, free_mem = self._should_chunk_mqa_logits(q_offset, k_offset, device)

        if not need_chunk:
            # 非分块路径：一次性计算所有 logits
            assert q_fp8[:q_offset].shape[0] != 0
            with self._with_real_sm_count():
                if _is_hip:
                    # HIP 路径：aiter fp8_mqa_logits
                    from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits

                    kv, scale = kv_fp8
                    logits = fp8_mqa_logits(
                        q_fp8[:q_offset], kv, scale, weights[:q_offset], ks, ke
                    )
                else:
                    # CUDA 路径：DeepGEMM fp8_mqa_logits（ragged 格式）
                    logits = deep_gemm.fp8_mqa_logits(
                        q_fp8[:q_offset],
                        kv_fp8,
                        weights[:q_offset],
                        ks,
                        ke,
                        clean_logits=False,
                    )
            assert logits.shape[0] == len(seq_lens_expanded)
            assert logits.shape[1] == k_offset

            # topk 变换并存储结果
            raw_topk_result = metadata.topk_transform(logits, self.index_topk, ks=ks)
            topk_result[:q_offset] = raw_topk_result
            return topk_result

        # Chunk path（分块路径：防止 OOM）
        bytes_per_elem = 4  # float32
        bytes_per_row = k_offset * bytes_per_elem
        # Reserve 50% of free memory for logits（使用 50% 可用显存）
        max_rows = max(1, int((free_mem * 0.5) // max(bytes_per_row, 1)))
        max_rows = min(max_rows, q_offset)  # 不超过实际 Q token 数

        global_topk_offset = metadata.attn_metadata.topk_indices_offset

        assert (
            seq_lens_expanded.shape[0] == q_offset
        ), f"seq_lens_expanded length mismatch: {seq_lens_expanded.shape[0]} != {q_offset}"
        if global_topk_offset is not None:
            assert (
                global_topk_offset.shape[0] >= q_offset
            ), f"topk_indices_offset too short: {global_topk_offset.shape[0]} < {q_offset}"

        start = 0
        while start < q_offset:
            end = min(start + max_rows, q_offset)  # 当前 chunk 的 token 范围

            with self._with_real_sm_count():
                if _is_hip:
                    from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits

                    kv, scale = kv_fp8
                    logits_chunk = fp8_mqa_logits(
                        q_fp8[start:end],
                        kv,
                        scale,
                        weights[start:end],
                        ks[start:end],
                        ke[start:end],
                    )
                else:
                    logits_chunk = deep_gemm.fp8_mqa_logits(
                        q_fp8[start:end],
                        kv_fp8,
                        weights[start:end],
                        ks[start:end],
                        ke[start:end],
                        clean_logits=False,
                    )

            lengths_chunk = seq_lens_expanded[start:end]

            # RAGGED: use global offset; PAGED: construct local cu_seqlens_q per chunk
            # RAGGED 路径：使用全局 topk offset；PAGED 路径：构造局部 cu_seqlens_q
            if global_topk_offset is not None:
                # RAGGED path
                topk_offset_chunk = global_topk_offset[start:end]
                cu_seqlens_q_chunk = None
                batch_idx_chunk = None
            else:
                # PAGED path: treat each token as a length-1 sequence
                # PAGED 路径：将每个 token 视为长度 1 的独立序列
                topk_offset_chunk = None
                B_chunk = logits_chunk.shape[0]
                # 为 chunk 内每个 token 构造全 1 的 cu_seqlens_q（每个 token q_len=1）
                cu_seqlens_q_chunk = torch.ones(
                    B_chunk, dtype=torch.int32, device=device
                )
                batch_idx_chunk = token_to_batch_idx[start:end]  # 对应的 batch 序号

            # 对当前 chunk 的 logits 执行 top-k 变换
            raw_topk_chunk = metadata.topk_transform(
                logits_chunk,
                self.index_topk,
                ks=ks[start:end],
                cu_seqlens_q=cu_seqlens_q_chunk,
                ke_offset=lengths_chunk,
                batch_idx_list=batch_idx_chunk,
                topk_indices_offset_override=topk_offset_chunk,
            )
            topk_result[start:end] = raw_topk_chunk
            start = end  # 移动到下一个 chunk

        return topk_result

    def _forward_cuda_k_only(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        act_quant,              # FP8 量化函数（用于 fallback 路径）
        enable_dual_stream: bool,
        metadata: BaseIndexerMetadata,
        return_indices: bool = True,
    ) -> Optional[torch.Tensor]:
        """仅计算并存储 K-cache，跳过 Q/logits 计算（快速路径）。

        当序列长度 <= index_topk 时，无需计算 logits，直接将所有 block 纳入稀疏索引，
        使用 dummy logits 触发 topk kernel 的快速路径（返回连续 block 编号）。
        """
        assert forward_batch.forward_mode.is_extend_without_speculative()
        x_meta = x[0] if isinstance(x, tuple) else x

        # Fast path: only compute and store k cache, skip all q and weights ops
        # 只计算 K（跳过 Q 投影和门控权重投影）
        key = self._get_k_bf16(x, positions, enable_dual_stream)

        # 确保 out_cache_loc 内存连续（fused store kernel 要求）
        if not forward_batch.out_cache_loc.is_contiguous():
            forward_batch.out_cache_loc = forward_batch.out_cache_loc.contiguous()

        # 量化并存储 K 到 KV pool 的 index K buffer
        self._store_index_k_cache(
            forward_batch=forward_batch,
            layer_id=layer_id,
            key=key,
            act_quant=act_quant,
        )

        # MHA doesn't need topk_indices
        if not return_indices:
            return None  # MHA 无需 top-k 索引，直接返回 None

        # MLA: use dummy logits with topk kernel's fast path to generate indices
        # When length <= 2048, naive_topk_cuda directly generates [0,1,...,length-1,-1,...]
        # MLA 路径：构造全零 dummy logits，触发 topk kernel 的 naive 快速路径
        seq_lens_expanded = metadata.get_seqlens_expanded()
        dummy_logits = torch.zeros(
            seq_lens_expanded.shape[0],
            self.index_topk,
            dtype=torch.float32,
            device=x_meta.device,
        )
        # topk_transform 对 dummy logits 执行快速路径，生成 [0,1,...,length-1,-1,...] 的连续索引
        return metadata.topk_transform(dummy_logits, self.index_topk)

    def _get_topk_ragged_with_cp(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,       # FP8 Q，只包含当前 CP rank 负责的序列段
        weights: torch.Tensor,     # 门控权重，对应 q_fp8
        metadata: BaseIndexerMetadata,
        kv_len: int,               # 当前 CP 段的 KV 长度
        actual_seq_q: int,         # 当前 CP 段的 Q token 数
        cp_index: List[Tuple[int, int, int]] = None,  # 多 batch CP 索引（batch_idx, start, end）
    ) -> torch.Tensor:
        """Context Parallel（CP）模式下的 ragged MQA logits 计算。

        CP 按序列维度切分时，每个 CP rank 只处理部分序列，需要从对应 KV cache 范围
        拉取数据并计算 logits，之后做 top-k 选取。
        支持多 batch（cp_index 非空）和单 batch（cp_index=None）两种情况。
        """
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)

        page_size = forward_batch.token_to_kv_pool.page_size
        assert page_size == 64, "only support page size 64"
        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)  # [tokens, n_heads]
        # 用于拼接多 batch 数据的列表
        k_fp8_list = []
        k_scale_list = []
        ks_list = []
        ke_offset_list = []
        offset = 0
        actual_seq_q_list = []
        batch_idx_list = []

        block_tables = metadata.get_page_table_64()  # page_size=64 的分页表

        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )
        if cp_index is not None:
            # TODO Multi-batch support has accuracy issues
            # 多 batch CP 路径：逐 batch 构造 KV range 并拼接
            for batch_idx, start_seq_position, end_seq_position in cp_index:
                # 计算该请求的 pre-fill token 数（已有 KV 长度）
                pre_chunk_offset = (
                    forward_batch.seq_lens_cpu[batch_idx].item()
                    - forward_batch.extend_seq_lens_cpu[batch_idx]
                )
                # 将 CP 内的相对位置转换为绝对 KV cache 位置
                start_seq_position += pre_chunk_offset
                end_seq_position += pre_chunk_offset
                if offset == 0 and batch_idx != 0:
                    offset += forward_batch.extend_seq_lens_cpu[batch_idx - 1]
                # 获取该 batch 从 0 到 end_seq_position 的 K cache（连续格式）
                k_fp8 = forward_batch.token_to_kv_pool.get_index_k_continuous(
                    layer_id,
                    end_seq_position,
                    block_tables[batch_idx],
                )
                k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_continuous(
                    layer_id,
                    end_seq_position,
                    block_tables[batch_idx],
                )

                extend_seq_len = end_seq_position - start_seq_position
                # ks：该 CP 段内所有 Q token 的 K 起始位置（uniform）
                ks = torch.full(
                    (extend_seq_len,), offset, dtype=torch.int32, device="cuda"
                )
                k_fp8_list.append(k_fp8)
                k_scale_list.append(k_scale)
                ks_list.append(ks)
                # ke_offset：每个 Q token 对应的 K 终止位置（等差数列）
                ke_offset = torch.arange(
                    start_seq_position + 1,
                    end_seq_position + 1,
                    dtype=torch.int32,
                    device="cuda",
                )
                ke_offset_list.append(ke_offset)
                actual_seq_q = torch.tensor(
                    [extend_seq_len], dtype=torch.int32, device="cuda"
                )
                actual_seq_q_list.append(actual_seq_q)
                batch_idx_list.append(batch_idx)

            # 拼接所有 batch 的 K、ks、ke_offset
            k_fp8 = torch.cat(k_fp8_list, dim=0).view(torch.float8_e4m3fn)
            k_scale = torch.cat(k_scale_list, dim=0).view(torch.float32).squeeze(-1)
            kv_fp8 = (k_fp8, k_scale)
            ks = torch.cat(ks_list, dim=0)
            ke_offset = torch.cat(ke_offset_list, dim=0)
            ke = ks + ke_offset  # 绝对 K 终止位置
            actual_seq_q = torch.cat(actual_seq_q_list, dim=0)
            with self._with_real_sm_count():
                # 计算 ragged MQA logits
                logits = deep_gemm.fp8_mqa_logits(
                    q_fp8,
                    kv_fp8,
                    weights,
                    ks,
                    ke,
                    clean_logits=False,
                )
            # top-k 变换（multi-batch 路径需要 batch_idx_list）
            topk_result = metadata.topk_transform(
                logits,
                self.index_topk,
                ks=ks,
                cu_seqlens_q=actual_seq_q,
                ke_offset=ke_offset,
                batch_idx_list=batch_idx_list,
            )
        else:
            # 单 batch CP 路径
            kv_len = (
                forward_batch.seq_lens_cpu[0].item()
                - forward_batch.extend_seq_lens_cpu[0]
                + kv_len  # 加上当前 CP 段的 KV 长度
            )
            # 获取该 batch 到 kv_len 的连续 K cache
            k_fp8 = forward_batch.token_to_kv_pool.get_index_k_continuous(
                layer_id,
                kv_len,
                block_tables[0],
            )
            k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_continuous(
                layer_id,
                kv_len,
                block_tables[0],
            )

            k_fp8 = k_fp8.view(torch.float8_e4m3fn)
            k_scale = k_scale.view(torch.float32).squeeze(-1)
            kv_fp8 = (k_fp8, k_scale)
            # 所有 Q token 的 K 起始位置均为 0（偏移量 offset=0）
            ks = torch.full((actual_seq_q,), offset, dtype=torch.int32, device="cuda")
            # ke_offset：从 kv_len-actual_seq_q+1 到 kv_len 的等差序列
            ke_offset = torch.arange(
                (kv_len - actual_seq_q) + 1,
                kv_len + 1,
                dtype=torch.int32,
                device="cuda",
            )
            ke = ks + ke_offset

            with self._with_real_sm_count():
                logits = deep_gemm.fp8_mqa_logits(
                    q_fp8,
                    kv_fp8,
                    weights,
                    ks,
                    ke,
                    clean_logits=False,
                )
            # actual_seq_q 转为 GPU tensor（topk_transform 需要）
            actual_seq_q = torch.tensor([actual_seq_q], dtype=torch.int32).to(
                device="cuda", non_blocking=True
            )
            topk_result = metadata.topk_transform(
                logits,
                self.index_topk,
                ks=ks,
                cu_seqlens_q=actual_seq_q,
                ke_offset=ke_offset,
            )

        return topk_result

    def forward_indexer(
        self,
        q_fp8: torch.Tensor,       # FP8 量化的 Q
        weights: torch.Tensor,     # 门控权重
        forward_batch: ForwardBatch,
        topk: int,
        layer_id: int,
    ) -> Optional[torch.Tensor]:
        """NPU 兼容的逐序列 indexer 前向（基于 fp8_index TileLang kernel）。

        不使用 DeepGEMM，而是逐个序列调用 fp8_index 计算 logits，
        再取 top-k block 索引，适用于 NPU 或无 DeepGEMM 的场景。
        """
        if not _is_npu:
            # CUDA/HIP 路径：导入 TileLang fp8_index kernel
            from sglang.srt.layers.attention.nsa.tilelang_kernel import fp8_index

        page_size = forward_batch.token_to_kv_pool.page_size
        assert page_size == 64, "only support page size 64"

        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)

        # logits = deep_gemm.fp8_mqa_logits(q_fp8, kv_fp8, weights, ks, ke)
        # 逐序列拼接 K cache 数据
        k_fp8_list = []
        k_scale_list = []

        topk_indices_list = []

        # 构造 page_size=64 的分页表（步长为 64）
        block_tables = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :
        ]
        strided_indices = torch.arange(
            0, block_tables.shape[-1], page_size, device="cuda"
        )
        # 将 token 级 page_table 转换为 block 级（除以 page_size）
        block_tables = block_tables[:, strided_indices] // page_size

        q_len_start = 0

        for i in range(forward_batch.batch_size):
            seq_len = forward_batch.seq_lens[i].item()  # 当前请求的 KV 序列长度
            q_len = (
                forward_batch.extend_seq_lens_cpu[i]    # extend 模式：取扩展 token 数
                if forward_batch.forward_mode.is_extend()
                else 1                                   # decode 模式：每请求 1 个 Q token
            )
            q_len_end = q_len_start + q_len

            # 取当前请求对应的 Q 切片，并增加 batch 维度
            q_fp8_partial = q_fp8[q_len_start:q_len_end]
            q_fp8_partial = q_fp8_partial.unsqueeze(0).contiguous()  # [1, q_len, n_heads, 132]

            weights_partial = weights[q_len_start:q_len_end]
            weights_partial = weights_partial.squeeze(-1).unsqueeze(0).contiguous()

            # 获取当前请求的连续 K cache（FP8）
            k_fp8 = forward_batch.token_to_kv_pool.get_index_k_continuous(
                layer_id,
                seq_len,
                block_tables[i],
            )
            k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_continuous(
                layer_id,
                seq_len,
                block_tables[i],
            )

            # 转换类型并增加 batch 维度
            k_fp8 = k_fp8.view(torch.float8_e4m3fn).unsqueeze(0).contiguous()
            k_scale = k_scale.view(torch.float32).squeeze(-1).unsqueeze(0).contiguous()

            # 调用 TileLang fp8_index 计算该请求的 Q·K 稀疏 logits
            index_score = fp8_index(
                q_fp8_partial,
                weights_partial,
                k_fp8,
                k_scale,
            )
            end_pos = seq_len
            # 取 top-k block 索引（不超过序列长度）
            topk_indices = index_score.topk(min(topk, end_pos), dim=-1)[1].squeeze(0)

            # 将 topk_indices 长度补齐到 2048 的整数倍（pad 为 -1）
            pad_len = ceil_align(topk_indices.shape[-1], 2048) - topk_indices.shape[-1]
            topk_indices = torch.nn.functional.pad(
                topk_indices, (0, pad_len), "constant", -1
            )

            topk_indices_list.append(topk_indices)

            q_len_start = q_len_end  # 更新 Q offset

        # 拼接所有请求的 top-k 索引
        topk_indices = torch.cat(topk_indices_list, dim=0)
        return topk_indices

    def _store_index_k_cache(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        key: torch.Tensor,
        *,
        act_quant=None,  # fallback only
    ) -> None:
        """将 NSA indexer 的 K cache 量化并存入 KV pool。

        优先路径（按顺序尝试）：
        1. JIT fused store（CUDA, page_size=64, 非 FNUZ）：最高效，一步完成量化+写 cache
        2. AITER fused quant+store（HIP, page_size=1）：ROCm 优化路径
        3. Fallback：act_quant 量化后手动写 KV pool（通用路径）

        Preferred: fused_store_index_k_cache(key, cache, out_cache_loc, page_size)
        Fallback : act_quant(key) + token_to_kv_pool.set_index_k_scale_buffer(...)
        """

        # Fast path: JIT fused store (CUDA, page_size=64, non-fnuz)
        # 优先路径 1：CUDA JIT 融合 store（最优）
        if (
            _is_cuda
            and (not _is_fp8_fnuz)
            and can_use_nsa_fused_store(
                key.dtype,
                forward_batch.out_cache_loc.dtype,
                forward_batch.token_to_kv_pool.page_size,
            )
        ):
            # NOTE: wrapper already normalizes shape/contiguity and asserts dtypes.
            # 获取 FP8+scale 格式的 K-cache buffer（形状 [num_pages, 132]）
            buf = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(
                layer_id=layer_id
            )
            # 一次性量化并写入 paged K-cache
            fused_store_index_k_cache(
                key,
                buf,
                forward_batch.out_cache_loc,
                forward_batch.token_to_kv_pool.page_size,
            )
            return

        # Fast path: AITER fused quant + cache store (HIP, page_size=1)
        # 优先路径 2：HIP aiter 融合量化+存储
        if _use_aiter:
            buf = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(
                layer_id=layer_id
            )
            # Reshape from (num_pages, 132) uint8 to (num_pages, 1, 132) fp8
            # to match kernel's (num_blocks, block_size, head_dim + scale_bytes) layout
            # 将 buf 重塑为 (num_pages, 1, 132) 的 FP8 格式，适配 aiter kernel 要求
            kv_cache = buf.unsqueeze(1).view(fp8_dtype)
            out_loc = forward_batch.out_cache_loc
            if not out_loc.is_contiguous():
                out_loc = out_loc.contiguous()
            # aiter 融合 kernel：量化 key 并直接写入 kv_cache（paged layout）
            indexer_k_quant_and_cache(
                key, kv_cache, out_loc, self.block_size, self.scale_fmt
            )
            return

        # Fallback: original path（通用回退路径）
        assert act_quant is not None
        # 1. 手动调用 act_quant 对 K 进行 FP8 量化
        k_fp8, k_scale = act_quant(key, self.block_size, self.scale_fmt)

        out_loc = forward_batch.out_cache_loc
        if not out_loc.is_contiguous():
            out_loc = out_loc.contiguous()

        # 2. 将量化后的 K 和 scale 写入 KV pool 的 index K buffer
        forward_batch.token_to_kv_pool.set_index_k_scale_buffer(
            layer_id=layer_id,
            loc=out_loc,
            index_k=k_fp8,
            index_k_scale=k_scale,
        )

    def forward_cuda(
        self,
        x: torch.Tensor,           # 隐状态（可能是 FP8 tuple）
        q_lora: torch.Tensor,      # Q LoRA 低秩激活
        positions: torch.Tensor,   # token 位置 id
        forward_batch: ForwardBatch,
        layer_id: int,
        return_indices: bool = True,
    ) -> Optional[torch.Tensor]:
        """NSA Indexer 的 CUDA/HIP 前向主入口。

        根据 forward_mode 分发到不同路径：
        - decode/target_verify/draft_extend：使用 paged MQA logits（_get_topk_paged）
        - extend（prefill）：使用 ragged MQA logits（_get_topk_ragged）
        - CP 模式 extend：使用 _get_topk_ragged_with_cp
        - K only 快速路径：当 KV 长度 <= topk 时跳过 logits 计算
        """
        if _is_hip:
            # HIP 平台使用 TileLang act_quant kernel
            from sglang.srt.layers.attention.nsa.tilelang_kernel import act_quant
        elif not _is_npu:
            # CUDA 平台使用 Triton act_quant kernel
            from sglang.srt.layers.attention.nsa.triton_kernel import act_quant

        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)

        # When upstream uses fused FP8 RMSNorm+quant, activations may be passed as
        # a tuple like (x_fp8, x_scale[, y]). Use `x_meta` for shape/device queries.
        # 当上游使用融合 FP8 RMSNorm+量化时，x 可能是 tuple，取第一个元素用于形状查询
        x_meta = x[0] if isinstance(x, tuple) else x

        # 获取当前 layer/batch 对应的 indexer 元数据
        metadata = forward_batch.attn_backend.get_indexer_metadata(
            layer_id, forward_batch
        )

        # 判断是否启用双流并行：仅在 CUDA graph 捕获模式且 token 数不超过阈值时启用
        enable_dual_stream = (
            self.alt_stream is not None
            and get_is_capture_mode()
            and q_lora.shape[0] > 0
            and q_lora.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
        )

        # skip NSA if attention backend choose to skip this batch
        # attention backend 返回 None 时跳过该 batch 的 NSA 计算
        if metadata is None:
            return None

        # Determine if should skip topk based on sequence length
        # We can only skip the logits computation if cuda graph is not involved
        # 判断是否可以跳过 logits 计算（序列长度 <= topk 时无需排序选取）
        skip_logits_computation = False
        if forward_batch.forward_mode.is_extend_without_speculative():
            if forward_batch.seq_lens_cpu is not None:
                max_kv_len = forward_batch.seq_lens_cpu.max().item()
                skip_logits_computation = max_kv_len <= self.index_topk  # 所有 KV 都在 topk 内

        # Optimization: fast path when skipping topk computation
        # 优化路径：跳过 logits，仅存 K-cache 并生成连续 block 索引
        if skip_logits_computation and (not self.nsa_enable_prefill_cp):
            return self._forward_cuda_k_only(
                x,
                positions,
                forward_batch,
                layer_id,
                act_quant,
                enable_dual_stream,
                metadata,
                return_indices,
            )

        if enable_dual_stream and forward_batch.forward_mode.is_decode_or_idle():
            # 双流 decode 路径：门控权重和 K-cache 存储并行
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            # 主 stream 计算门控权重（使用半数 SM）
            weights = self._project_and_scale_head_gates(x)
            # 计算 Q/K（双流并行）
            query, key = self._get_q_k_bf16(
                q_lora, x, positions, enable_dual_stream, forward_batch=forward_batch
            )
            # 对 Q 做 FP8 量化
            q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
            with torch.cuda.stream(self.alt_stream):
                # alt_stream 上存 K-cache（与主 stream 计算 Q logits 并行）
                self._store_index_k_cache(
                    forward_batch=forward_batch,
                    layer_id=layer_id,
                    key=key,
                    act_quant=act_quant,
                )
            current_stream.wait_stream(self.alt_stream)
            # 将 q_scale 融合到门控权重
            weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        else:
            # 非双流 decode 或 extend 路径
            query, key = self._get_q_k_bf16(
                q_lora, x, positions, enable_dual_stream, forward_batch=forward_batch
            )

            if enable_dual_stream:
                # 双流 extend 路径：Q 量化和 K-cache 存储并行
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)

                q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
                with torch.cuda.stream(self.alt_stream):
                    self._store_index_k_cache(
                        forward_batch=forward_batch,
                        layer_id=layer_id,
                        key=key,
                        act_quant=act_quant,
                    )
                current_stream.wait_stream(self.alt_stream)
            else:
                # 单流路径：顺序执行 Q 量化和 K-cache 存储
                q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
                self._store_index_k_cache(
                    forward_batch=forward_batch,
                    layer_id=layer_id,
                    key=key,
                    act_quant=act_quant,
                )

            # aiter (ROCm gfx95): the 3-tuple (fp8, scale, bf16) from
            # fused_rms_fp8_group_quant is passed directly to _get_logits_head_gate,
            # which extracts the bf16 tensor via _weights_proj_bf16_in_fp32_out,
            # completely skipping the FP8 dequantization path below.
            # 处理 x 的 tuple 格式，以正确计算门控权重
            if (
                _use_aiter
                and _is_gfx95_supported
                and isinstance(x, tuple)
                and len(x) == 3
            ):
                # aiter gfx95：直接传入 3-tuple，内部提取 BF16 passthrough
                x_for_gate = x
            elif isinstance(x, tuple):
                # 通用 tuple 路径（x_fp8, x_scale[, y]）：手动反量化为 BF16
                assert len(x) in (
                    2,
                    3,
                ), "For tuple input, only (x, x_s) or (x, x_s, y) formats are accepted"
                x_q, x_s = x[0], x[1]
                if (
                    x_s is not None
                    and x_q.dim() == 2
                    and x_s.dim() == 2
                    and x_q.shape[0] == x_s.shape[0]
                ):
                    m, n = x_q.shape
                    ng = x_s.shape[1]
                    if ng > 0 and n % ng == 0:
                        group = n // ng
                        # 分组反量化：x_q * x_s（按 group 广播）→ BF16
                        x_for_gate = (
                            x_q.to(torch.float32)
                            .view(m, ng, group)
                            .mul_(x_s.to(torch.float32).unsqueeze(-1))
                            .view(m, n)
                            .to(torch.bfloat16)
                        )
                    else:
                        x_for_gate = x_q.to(torch.bfloat16)
                else:
                    x_for_gate = x_q.to(torch.bfloat16)
            else:
                x_for_gate = x  # 标准 BF16 张量，直接使用

            # 计算融合了 q_scale 的门控权重
            weights = self._get_logits_head_gate(x_for_gate, q_scale)

        if _is_cuda or _is_hip:
            assert forward_batch.seq_lens_cpu is not None
            if len(forward_batch.seq_lens_cpu) == 0:
                # this seems b/c max-pad, no worries?
                # seq_lens 为空（MAX_LEN padding 场景），返回全 -1 的 topk 结果
                return torch.full(
                    (x_meta.shape[0], self.index_topk),
                    -1,
                    dtype=torch.int,
                    device=x_meta.device,
                )

            if (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend(include_v2=True)
            ):
                # decode/MTP 推测解码路径：使用 paged MQA logits
                topk_result = self._get_topk_paged(
                    forward_batch, layer_id, q_fp8, weights, metadata
                )
            else:
                # extend/prefill 路径：判断是否使用 CP 分割
                if (
                    forward_batch.attn_cp_metadata is not None
                    and is_nsa_prefill_cp_in_seq_split()
                ):
                    # CP 序列维度切分路径：将 Q 和 weights 各自切成 prev/next 两半
                    kv_len_prev = forward_batch.attn_cp_metadata.kv_len_prev
                    kv_len_next = forward_batch.attn_cp_metadata.kv_len_next
                    actual_seq_q_prev = forward_batch.attn_cp_metadata.actual_seq_q_prev
                    actual_seq_q_next = forward_batch.attn_cp_metadata.actual_seq_q_next

                    # TODO support mutil-batch
                    # cp_batch_seq_index_prev = forward_batch.attn_cp_metadata["cp_batch_seq_index_prev"]
                    # cp_batch_seq_index_next = forward_batch.attn_cp_metadata["cp_batch_seq_index_next"]
                    # TODO prev, next, combined into a single call
                    # 按序列前半/后半切分 Q 和 weights
                    q_fp8_prev, q_fp8_next = torch.split(
                        q_fp8, (q_fp8.shape[0] + 1) // 2, dim=0
                    )
                    weights_prev, weights_next = torch.split(
                        weights, (weights.shape[0] + 1) // 2, dim=0
                    )
                    # 分别计算 prev 和 next 的 top-k 索引
                    topk_result_prev = self._get_topk_ragged_with_cp(
                        forward_batch,
                        layer_id,
                        q_fp8_prev,
                        weights_prev,
                        metadata,
                        kv_len_prev,
                        actual_seq_q_prev,
                    )

                    topk_result_next = self._get_topk_ragged_with_cp(
                        forward_batch,
                        layer_id,
                        q_fp8_next,
                        weights_next,
                        metadata,
                        kv_len_next,
                        actual_seq_q_next,
                    )
                    # 拼接 prev 和 next 的结果
                    return torch.cat([topk_result_prev, topk_result_next], dim=0)
                else:
                    # 标准 ragged extend 路径（无 CP 或非序列维度 CP）
                    topk_result = self._get_topk_ragged(
                        enable_dual_stream,
                        forward_batch,
                        layer_id,
                        q_fp8,
                        weights,
                        metadata,
                    )
        else:
            # 非 CUDA/HIP 路径（如测试环境）：使用逐序列 forward_indexer
            topk_result = self.forward_indexer(
                q_fp8.contiguous(),
                weights,
                forward_batch,
                topk=self.index_topk,
                layer_id=layer_id,
            )
        return topk_result

    def forward_npu(
        self,
        x: torch.Tensor,                          # 隐状态 [bs, hidden_size]
        q_lora: torch.Tensor,                     # Q LoRA 低秩激活
        positions: torch.Tensor,                  # token 位置 id
        forward_batch: ForwardBatch,
        layer_id: int,
        layer_scatter_modes=None,                 # TP/AG scatter 模式
        dynamic_scale: torch.Tensor = None,       # 动态量化 scale（可选）
    ) -> torch.Tensor:
        """华为昇腾 NPU 平台的 Indexer 前向。

        使用 torch_npu.npu_rotary_mul 做 RoPE，
        通过 torch_npu.npu_lightning_indexer 做稀疏注意力索引计算。
        支持 neox_style 和非 neox_style RoPE，支持 CP 平衡索引。
        """
        # 获取实际 KV 序列长度（int 或 tensor 格式）
        if forward_batch.attn_backend.forward_metadata.seq_lens_cpu_int is None:
            actual_seq_lengths_kv = forward_batch.attn_backend.forward_metadata.seq_lens
        else:
            actual_seq_lengths_kv = (
                forward_batch.attn_backend.forward_metadata.seq_lens_cpu_int
            )
        # 判断是否为 prefill 模式（非推测解码场景）
        is_prefill = (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_target_verify()
            and not forward_batch.forward_mode.is_draft_extend()
        )

        bs = q_lora.shape[0]  # batch size（token 数）

        if self.rotary_emb.is_neox_style:
            # neox 风格 RoPE：前半为 cos，后半为 sin，需要 repeat 拼接
            if not hasattr(forward_batch, "npu_indexer_sin_cos_cache"):
                # 首次调用：从 rotary_emb cache 查表并缓存
                cos_sin = self.rotary_emb.cos_sin_cache[positions]
                cos, sin = cos_sin.chunk(2, dim=-1)
                # repeat 拼接使 cos/sin 维度翻倍，匹配 rope_head_dim
                cos = cos.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
                sin = sin.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
                forward_batch.npu_indexer_sin_cos_cache = (sin, cos)
            else:
                # 复用缓存的 sin/cos（避免重复查表）
                sin, cos = forward_batch.npu_indexer_sin_cos_cache

            if self.alt_stream is not None:
                # 多 stream 路径：Q 投影和 RoPE 在辅 stream 上并行
                self.alt_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(self.alt_stream):
                    # 传入动态 scale 时与 q_lora 打包
                    q_lora = (
                        (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
                    )
                    q = self.wq_b(q_lora)[
                        0
                    ]  # [bs, 1536] @ [1536, 64 * 128] = [bs, 64 * 128]
                    wq_b_event = self.alt_stream.record_event()  # 记录 wq_b 完成事件
                    q = q.view(bs, self.n_heads, self.head_dim)  # [bs, 64, 128]
                    # 分离 rope 和 nope 部分
                    q_pe, q_nope = torch.split(
                        q,
                        [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                        dim=-1,
                    )  # [bs, 64, 64 + 64]
                    # 增加 head_k 维度用于 npu_rotary_mul
                    q_pe = q_pe.view(bs, self.n_heads, 1, self.rope_head_dim)
                    # 应用 NPU 旋转位置编码（neox 风格）
                    q_pe = torch_npu.npu_rotary_mul(q_pe, cos, sin).view(
                        bs, self.n_heads, self.rope_head_dim
                    )  # [bs, n, d]
                    # 将 rope 结果与 nope 拼接回完整 query
                    q = torch.cat([q_pe, q_nope], dim=-1)
                    q.record_stream(self.alt_stream)
                    q_rope_event = self.alt_stream.record_event()  # 记录 rope 完成事件
            else:
                # 单 stream 路径
                q_lora = (
                    (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
                )
                q = self.wq_b(q_lora)[
                    0
                ]  # [bs, 1536] @ [1536, 64 * 128] = [bs, 64 * 128]
                q = q.view(bs, self.n_heads, self.head_dim)  # [bs, 64, 128]
                q_pe, q_nope = torch.split(
                    q,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )  # [bs, 64, 64 + 64]
                q_pe = q_pe.view(bs, self.n_heads, 1, self.rope_head_dim)
                q_pe = torch_npu.npu_rotary_mul(q_pe, cos, sin).view(
                    bs, self.n_heads, self.rope_head_dim
                )  # [bs, n, d]
                q = torch.cat([q_pe, q_nope], dim=-1)

            # 门控权重投影：x → weights [bs, n_heads]
            if envs.SGLANG_NPU_USE_MULTI_STREAM.get():
                # 多 stream 路径：weights 投影在独立的 indexer_weight_stream 上并行
                indexer_weight_stream = get_indexer_weight_stream()
                indexer_weight_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(indexer_weight_stream):
                    x = x.view(-1, self.hidden_size)
                    weights = self.weights_proj(x.float())[0].to(torch.bfloat16)
                    weights.record_stream(indexer_weight_stream)
                    weights_event = indexer_weight_stream.record_event()
            else:
                x = x.view(-1, self.hidden_size)
                weights = self.weights_proj(x.float())[0].to(torch.bfloat16)

            # K 投影和 RoPE
            k_proj = self.wk(x)[0]  # [b, s, 7168] @ [7168, 128] = [b, s, 128]
            k = self.k_norm(k_proj)
            # TP AG 模式：对 K 做 all-gather 以获取完整 TP 维度数据
            if (
                _use_ag_after_qlora
                and layer_scatter_modes.layer_input_mode == ScatterMode.SCATTERED
                and layer_scatter_modes.attn_mode == ScatterMode.TP_ATTN_FULL
            ):
                k = scattered_to_tp_attn_full(k, forward_batch)
            # 分离 K 的 rope 和 nope 部分
            k_pe, k_nope = torch.split(
                k,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )  # [bs, 64 + 64]

            # 对 K 的 rope 部分应用 NPU 旋转位置编码
            k_pe = k_pe.view(-1, 1, 1, self.rope_head_dim)
            k_pe = torch.ops.npu.npu_rotary_mul(k_pe, cos, sin).view(
                bs, 1, self.rope_head_dim
            )  # [bs, 1, d]
            # 拼接 K 的 rope 和 nope 部分
            k = torch.cat([k_pe, k_nope.unsqueeze(1)], dim=-1)  # [bs, 1, 128]

        else:
            # 非 neox_style RoPE 路径（使用 rotary_emb 标准接口）
            if envs.SGLANG_NPU_USE_MULTI_STREAM.get():
                # 多 stream 路径：weights 在独立 stream 上并行投影
                indexer_weight_stream = get_indexer_weight_stream()
                indexer_weight_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(indexer_weight_stream):
                    x = x.view(-1, self.hidden_size)
                    weights = self.weights_proj(x.float())[0].to(torch.bfloat16)
                    weights.record_stream(indexer_weight_stream)
                    weights_event = indexer_weight_stream.record_event()
            else:
                x = x.view(-1, self.hidden_size)
                weights = self.weights_proj(x.float())[0].to(torch.bfloat16)

            q_lora = (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
            q = self.wq_b(q_lora)[0]  # [bs, 1536] @ [1536, 64 * 128] = [bs, 64 * 128]
            q = q.view(bs, self.n_heads, self.head_dim)  # [bs, 64, 128]
            # 分离 Q 的 rope 和 nope 部分
            q_pe, q_nope = torch.split(
                q,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )  # [bs, 64, 64 + 64]

            k_proj = self.wk(x)[0]  # [b, s, 7168] @ [7168, 128] = [b, s, 128]
            k = self.k_norm(k_proj)
            # 分离 K 的 rope 和 nope 部分
            k_pe, k_nope = torch.split(
                k,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )  # [bs, 64 + 64]

            k_pe = k_pe.unsqueeze(1)  # [bs, 1, rope_head_dim]

            # 第一层时初始化 rotary_emb 的 sin_cos 查找表
            if layer_id == 0:
                self.rotary_emb.sin_cos_cache = (
                    self.rotary_emb.cos_sin_cache.index_select(0, positions)
                )

            # 使用标准 rotary_emb 接口对 Q 和 K 应用 RoPE
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
            k_pe = k_pe.squeeze(1)  # 去掉 head_k 维度
            # 拼接 rope 和 nope 部分
            q = torch.cat([q_pe, q_nope], dim=-1)
            k = torch.cat([k_pe, k_nope], dim=-1)

        # CP 模式：对 K 做 all-gather（聚合各 CP rank 的 K 数据）
        if (
            is_prefill
            and self.nsa_enable_prefill_cp
            and forward_batch.attn_cp_metadata is not None
        ):
            k = cp_all_gather_rerange_output(
                k.contiguous().view(-1, self.head_dim),
                self.cp_size,
                forward_batch,
                torch.npu.current_stream(),
            )

        # 将当前 step 的 K 存入 KV pool 的 index K buffer
        forward_batch.token_to_kv_pool.set_index_k_buffer(
            layer_id, forward_batch.out_cache_loc, k
        )
        if is_prefill:
            # Prefill 模式：设置 actual_seq_lengths_q（Q 序列长度）
            if (
                self.nsa_enable_prefill_cp
                and forward_batch.attn_cp_metadata is not None
            ):
                # CP 路径：从 cp_metadata 获取 prev/next 两段的 Q 长度
                forward_batch.attn_backend.forward_metadata.actual_seq_lengths_q = (
                    forward_batch.attn_cp_metadata.actual_seq_q_prev_tensor,
                    forward_batch.attn_cp_metadata.actual_seq_q_next_tensor,
                )
                if sum(forward_batch.extend_prefix_lens_cpu) > 0:
                    # 有 prefix 共享时，KV 长度需加上 prefix 长度
                    total_kv_len_prev_tensor = (
                        forward_batch.attn_cp_metadata.kv_len_prev_tensor
                        + forward_batch.extend_prefix_lens.squeeze()
                    )
                    total_kv_len_next_tensor = (
                        forward_batch.attn_cp_metadata.kv_len_next_tensor
                        + forward_batch.extend_prefix_lens.squeeze()
                    )
                    forward_batch.attn_backend.forward_metadata.actual_seq_lengths_kv = (
                        total_kv_len_prev_tensor,
                        total_kv_len_next_tensor,
                    )
                else:
                    # 无 prefix 时直接使用 cp_metadata 中的 KV 长度
                    forward_batch.attn_backend.forward_metadata.actual_seq_lengths_kv = (
                        forward_batch.attn_cp_metadata.kv_len_prev_tensor,
                        forward_batch.attn_cp_metadata.kv_len_next_tensor,
                    )
                actual_seq_lengths_q = (
                    forward_batch.attn_backend.forward_metadata.actual_seq_lengths_q
                )
                actual_seq_lengths_kv = (
                    forward_batch.attn_backend.forward_metadata.actual_seq_lengths_kv
                )
            else:
                # 标准 prefill：直接使用 seq_lens 和 extend 累积长度
                actual_seq_lengths_kv = forward_batch.seq_lens
                actual_seq_lengths_q = forward_batch.extend_seq_lens.cumsum(dim=0)
        else:
            # Decode/MTP 模式：计算 actual_seq_lengths_q
            if forward_batch.attn_backend.forward_metadata.actual_seq_lengths_q is None:
                if (
                    forward_batch.forward_mode.is_draft_extend_v2()
                    or forward_batch.forward_mode.is_target_verify()
                    or forward_batch.forward_mode.is_draft_extend()
                ):
                    # MTP 推测解码模式：每次步进 num_draft_tokens 个 token
                    num_draft_tokens = (
                        forward_batch.attn_backend.speculative_num_draft_tokens
                    )
                    actual_seq_lengths_q = torch.arange(
                        num_draft_tokens,
                        num_draft_tokens + bs,
                        num_draft_tokens,
                        dtype=torch.int32,
                        device=k.device,
                    )
                else:
                    # 标准 decode：每个请求固定 1 个 Q token，q_len 累积为 [1, 2, ..., bs]
                    actual_seq_lengths_q = torch.tensor(
                        [1 + i * 1 for i in range(bs)],
                        dtype=torch.int32,
                        device=k.device,
                    )
            else:
                # 复用 backend 预计算的 actual_seq_lengths_q
                actual_seq_lengths_q = (
                    forward_batch.attn_backend.forward_metadata.actual_seq_lengths_q
                )

        # 获取历史 K-cache（完整 buffer，用于 indexer 注意力计算）
        past_key_states = forward_batch.token_to_kv_pool.get_index_k_buffer(layer_id)

        # 等待 alt_stream（Q RoPE）和 weights_event（门控权重投影）完成
        if self.rotary_emb.is_neox_style and self.alt_stream is not None:
            torch.npu.current_stream().wait_event(q_rope_event)
        if envs.SGLANG_NPU_USE_MULTI_STREAM.get():
            torch.npu.current_stream().wait_event(weights_event)
        # TP AG 模式：对 weights 做 all-gather
        if (
            _use_ag_after_qlora
            and layer_scatter_modes.layer_input_mode == ScatterMode.SCATTERED
            and layer_scatter_modes.attn_mode == ScatterMode.TP_ATTN_FULL
        ):
            weights = scattered_to_tp_attn_full(weights, forward_batch)
        block_table = forward_batch.attn_backend.forward_metadata.block_tables
        if (
            is_prefill
            and self.nsa_enable_prefill_cp
            and forward_batch.attn_cp_metadata is not None
        ):
            # CP prefill 路径：截取对应 CP rank 的 block_table 并做 CP 平衡 indexer
            block_table = block_table[: actual_seq_lengths_q[0].numel()]
            topk_indices = self.do_npu_cp_balance_indexer(
                q.view(-1, self.n_heads, self.head_dim),
                past_key_states,
                weights,
                actual_seq_lengths_q,
                actual_seq_lengths_kv,
                block_table,
            )
            return topk_indices
        else:
            # 标准路径：截取 prefill 时的 block_table
            block_table = (
                block_table[: actual_seq_lengths_q.size()[0]]
                if is_prefill
                else block_table
            )

            # 调用 npu_lightning_indexer 计算稀疏注意力 top-k 索引
            topk_indices = torch_npu.npu_lightning_indexer(
                query=q.view(-1, self.n_heads, self.head_dim),
                key=past_key_states,
                weights=weights,
                actual_seq_lengths_query=actual_seq_lengths_q.to(torch.int32),
                actual_seq_lengths_key=actual_seq_lengths_kv.to(k.device).to(
                    torch.int32
                ),
                block_table=block_table,
                layout_query="TND",         # Q 布局：[Token, n_heads, head_dim]
                layout_key="PA_BSND",       # K 布局：分页（PA）格式
                sparse_count=self.index_topk,  # top-k block 数
                sparse_mode=3,              # 稀疏模式 3（block-level sparse）
            )
            return topk_indices[0]

    def do_npu_cp_balance_indexer(
        self,
        q,                  # Q 张量 [tokens, n_heads, head_dim]
        past_key_states,    # 历史 K-cache
        indexer_weights,    # 门控权重 [tokens, n_heads]
        actual_seq_lengths_q,   # (prev_tensor, next_tensor)：Q 序列长度对
        actual_seq_lengths_kv,  # (prev_tensor, next_tensor)：KV 序列长度对
        block_table,        # 分页表
    ):
        """NPU Context Parallel 平衡索引：将 Q 切为 prev/next 两半，分别调用 npu_lightning_indexer。

        CP 按序列切分时，将序列的前半段（prev）和后半段（next）各自独立计算 top-k 索引，
        确保每个 CP rank 的计算负载均衡。
        """
        # 按 token 数的一半切分 Q 和 weights
        q_prev, q_next = torch.split(q, (q.size(0) + 1) // 2, dim=0)
        weights_prev, weights_next = None, None
        if indexer_weights is not None:
            weights_prev, weights_next = torch.split(
                indexer_weights, (indexer_weights.size(0) + 1) // 2, dim=0
            )
            # 确保 weights 内存连续
            weights_prev = weights_prev.contiguous().view(-1, weights_prev.shape[-1])
            weights_next = weights_next.contiguous().view(-1, weights_next.shape[-1])

        # 解包 prev/next 的序列长度
        actual_seq_lengths_q_prev, actual_seq_lengths_q_next = actual_seq_lengths_q
        actual_seq_lengths_kv_prev, actual_seq_lengths_kv_next = actual_seq_lengths_kv

        # 计算序列前半段的稀疏索引
        topk_indices_prev = torch_npu.npu_lightning_indexer(
            query=q_prev,
            key=past_key_states,
            weights=weights_prev,
            actual_seq_lengths_query=actual_seq_lengths_q_prev.to(
                device=q.device, dtype=torch.int32
            ),
            actual_seq_lengths_key=actual_seq_lengths_kv_prev.to(
                device=q.device, dtype=torch.int32
            ),
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk,
            sparse_mode=3,
        )
        # 计算序列后半段的稀疏索引
        topk_indices_next = torch_npu.npu_lightning_indexer(
            query=q_next,
            key=past_key_states,
            weights=weights_next,
            actual_seq_lengths_query=actual_seq_lengths_q_next.to(
                device=q.device, dtype=torch.int32
            ),
            actual_seq_lengths_key=actual_seq_lengths_kv_next.to(
                device=q.device, dtype=torch.int32
            ),
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk,
            sparse_mode=3,
        )
        # 返回 (prev_indices, next_indices) tuple
        return topk_indices_prev[0], topk_indices_next[0]


def scattered_to_tp_attn_full(
    hidden_states: torch.Tensor,
    forward_batch,
) -> torch.Tensor:
    """TP 注意力模式下的 all-gather：将散布的局部 hidden_states 聚合为完整张量。

    当 layer_scatter_mode=SCATTERED 且 attn_mode=TP_ATTN_FULL 时，每个 TP rank
    只持有部分 token 的 hidden_states，需要 all-gather 获取所有 token 的完整数据。
    """
    # 分配目标张量（全 token 数，与当前 rank 相同的 hidden_dim）
    hidden_states, local_hidden_states = (
        torch.empty(
            (forward_batch.input_ids.shape[0], hidden_states.shape[1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        ),
        hidden_states,
    )
    # 执行 TP all-gather（将各 rank 的局部 hidden_states 聚合到完整张量）
    attn_tp_all_gather_into_tensor(hidden_states, local_hidden_states.contiguous())
    return hidden_states
