from __future__ import annotations

import dataclasses
import functools
import math
import warnings
from functools import lru_cache, partial
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from sglang.jit_kernel.norm import can_use_fused_inplace_qknorm as can_use_jit_qk_norm
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.utils import (
    get_bool_env_var,
    get_device_capability,
    is_blackwell_supported,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    is_xpu,
    print_info_once,
)
from sglang.srt.utils.multi_stream_utils import (
    maybe_execute_in_parallel,
    with_multi_stream,
)

# 检测当前运行的硬件平台类型
_is_cuda = is_cuda()
_is_musa = is_musa()
_is_npu = is_npu()
_is_hip = is_hip()
_is_xpu = is_xpu()

if _is_cuda:
    # CUDA 环境下导入 cuDNN batch prefill 和 FlashAttention varlen 函数
    from flashinfer.prefill import cudnn_batch_prefill_with_kv_cache

    from sglang.jit_kernel.flash_attention import (
        flash_attn_varlen_func,
    )

if _is_musa:
    # MUSA 平台（摩尔线程）导入 flash attention varlen 接口
    from flash_attn_interface import flash_attn_varlen_func

if _is_npu:
    # 昇腾 NPU 平台导入相关库
    import torch_npu

from sglang.srt.distributed import (
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
)
from sglang.srt.distributed import utils as dist_utils
from sglang.srt.layers.attention.triton_ops.prefill_attention import (
    context_attention_fwd,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.rotary_embedding import apply_rotary_pos_emb
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, get_bool_env_var

# 是否启用 aiter（AMD ROCm 专用加速 kernel）
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

# 旋转位置编码实现类映射表
ROTARY_EMBED_CLASSES = {
    "normal": apply_rotary_pos_emb,
}

# === Vision Encoder === #
# FlashInfer 工作区缓冲大小（128MB）
FLASHINFER_WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024

# Batch buckets for cuDNN graph caching - graphs are cached per bucket size
# This avoids creating a new graph for each unique batch size at runtime
# cuDNN 图缓存的 batch size 分桶列表，避免对每个 batch size 重新建图
BATCH_BUCKETS = [8, 16, 32, 64]

# Bucketized max seqlens to reduce cuDNN recompilation frequency while
# preserving a tighter upper bound than a single fixed max seqlen.
# FlashInfer max_seqlen 分桶列表，降低 cuDNN 重编译频率
FLASHINFER_MAX_SEQLEN_BUCKETS = [
    4 * 1024,
    8 * 1024,
    16 * 1024,
    32 * 1024,
    64 * 1024,
    128 * 1024,
]


@dataclasses.dataclass
class SingletonCache:
    # 单例缓存数据类，用于懒初始化并缓存计算结果
    data: Any = None

    def set_data(self, value: Any) -> None:
        # 设置缓存数据
        self.data = value

    def get_data(self) -> Optional[Any]:
        # 获取缓存数据
        return self.data

    def empty(self) -> bool:
        # 判断缓存是否为空
        return self.get_data() is None


# TODO: requires real seqlens from images
@functools.lru_cache(maxsize=128)
def _get_cu_seqlens_for_shape(batch_size: int, seqlen: int, device) -> torch.Tensor:
    """
    Generates cumulative sequence lengths (cu_seqlens) for a given batch_size, seqlen, and device.
    Caches the result based on these parameters.
    为给定批大小和序列长度生成累积序列长度（cu_seqlens），结果按参数缓存。
    """
    # 生成 [0, seqlen, 2*seqlen, ..., batch_size*seqlen] 的累积序列长度
    cu_seqlens = torch.arange(
        0,
        (batch_size + 1) * seqlen,
        step=seqlen,
        dtype=torch.int32,
        device=device,
    )
    return cu_seqlens


def resolve_seqlens(
    cu_seqlens: torch.Tensor | SingletonCache | None,
    bsz: int,
    seq_len: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    # 统一处理 cu_seqlens 的三种输入类型：None / SingletonCache / torch.Tensor
    if cu_seqlens is None:
        # 未提供时，按 bsz * seq_len 均匀生成
        resolved_seqlens = _get_cu_seqlens_for_shape(bsz, seq_len, device=device)
    elif isinstance(cu_seqlens, SingletonCache):
        # SingletonCache：懒初始化，首次访问时计算并缓存
        if cu_seqlens.empty():
            cu_seqlens.set_data(_get_cu_seqlens_for_shape(bsz, seq_len, device=device))
        resolved_seqlens = cu_seqlens.get_data()
    else:
        # 直接使用传入的张量
        resolved_seqlens = cu_seqlens
    assert isinstance(
        resolved_seqlens, torch.Tensor
    ), "cu_seqlens must be a torch.Tensor"
    return resolved_seqlens


class VisionSdpaAttention(nn.Module):
    r"""
    Scaled Dot Product Attention inner product
    使用 PyTorch 内置 SDPA（缩放点积注意力）实现的视觉注意力模块
    """

    def __init__(
        self,
        head_dim: int,       # 每个注意力头的特征维度
        num_heads: int,      # Query 的头数
        num_kv_heads: int,   # KV 的头数（GQA 时 < num_heads）
        dropout: float = 0.0,
        flatten_batch: bool = False,  # 是否将 batch 维展平（适用于 cu_seqlens 打包格式）
        softmax_in_single_precision: bool = False,  # 是否在单精度下计算 softmax
        softmax_scale: float | None = None,  # 注意力分数缩放因子，None 时自动计算
        **kwargs,
    ):
        super().__init__()
        self.head_size = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.flatten_batch = flatten_batch
        self.softmax_in_single_precision = softmax_in_single_precision
        self.dropout = dropout
        # 未提供 softmax_scale 时，使用标准缩放 1/sqrt(head_dim)
        self.scale = (
            softmax_scale
            if softmax_scale is not None
            else 1.0 / math.sqrt(self.head_size)
        )

    @staticmethod
    @lru_cache(maxsize=128)
    def _generate_mask_cache(
        s: int, flatten_batch: bool, cu_seqlens: tuple
    ) -> torch.BoolTensor:
        """
        Generate a boolean attention mask with caching mechanism.
        Args:
            s: sequence length
            flatten_batch: whether to flatten batch dimension
            cu_seqlens: tuple of cumulative sequence lengths
        Returns:
            attention mask tensor of shape [b, 1, s, s] or [1, s, s]
        生成布尔注意力掩码（带缓存机制），避免重复计算。
        """
        if flatten_batch:
            # flatten_batch 模式：将不同图片的注意力区域用 True 标记，其余为 False
            mask = torch.zeros([1, s, s], dtype=torch.bool)
            for i in range(1, len(cu_seqlens)):
                start = cu_seqlens[i - 1]
                end = cu_seqlens[i]
                mask[..., start:end, start:end] = True
        else:
            # 正常 batch 模式：对每个序列独立生成矩形掩码
            # [1, 1, 1, s]
            row_indices = torch.arange(s).view(1, 1, 1, s)
            # [1, 1, s, 1]
            col_indices = torch.arange(s).view(1, 1, s, 1)
            # [b, 1, 1, 1]
            seq_lens = torch.tensor(
                [end - start for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])],
            ).view(-1, 1, 1, 1)

            # 有效位置：row 和 col 均在当前序列长度内
            mask = (row_indices < seq_lens) & (col_indices < seq_lens)

        return mask

    def generate_patch_attention_mask(
        self,
        s: int,
        cu_seqlens: Optional[torch.Tensor],
        flatten_batch: bool = False,
    ) -> Optional[torch.Tensor]:
        r"""
        Creates a non-causal 4D mask of shape `(b, 1, s, s)` or `(1, 1, s, s)`.
        创建非因果的 4D 注意力掩码（视觉 patch 注意力专用）。
        Args:
            s: sequence length
            cu_seqlens: cumulative sequence lengths tensor. If not, returns an empty mask
            flatten_batch: whether to flatten batch dimension
        Returns:
            attention mask tensor or None
        """
        if cu_seqlens is None:
            return None

        # 将 tensor 转为 tuple 以便 lru_cache 缓存
        cu_seqlens_tuple = tuple(cu_seqlens.cpu().tolist())

        return self._generate_mask_cache(s, flatten_batch, cu_seqlens_tuple)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bsz: int,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        # flatten_batch 模式下 bsz 必须为 1（所有序列拼接在一起）
        if self.flatten_batch:
            assert bsz == 1, "flatten_batch is True, bsz must be 1"

        assert q.dim() == 3, q.shape

        # 计算单个序列的长度
        s = q.shape[0] // bsz

        # [b, 1, s, s]  若未提供 attention_mask 则自动生成
        if attention_mask is None:
            attention_mask = self.generate_patch_attention_mask(
                s, cu_seqlens, flatten_batch=self.flatten_batch
            )

        if attention_mask is None:
            if self.softmax_in_single_precision:
                raise RuntimeError("Empty attention mask")
        else:
            # 将掩码移到与 q 相同的设备
            attention_mask = attention_mask.to(device=q.device)

        # 将 (b*s, h, d) 重排为 (b, h, s, d) 以满足 SDPA 接口要求
        q, k, v = [rearrange(x, "(b s) h d -> b h s d", b=bsz) for x in [q, k, v]]

        if self.softmax_in_single_precision:
            # 单精度 softmax 路径：手动计算 attention（不使用 SDPA）
            k = rearrange(k, "b h s d -> b h d s")
            attn_weights = torch.matmul(q, k) * self.scale
            del k
            # masking
            # 将无效位置用极小值填充（近似 -inf），使 softmax 后趋近于 0
            attention_mask = (~attention_mask) * torch.finfo(q.dtype).min
            attn_weights = attn_weights + attention_mask
            del attention_mask
            # full-precision
            # 在 float32 精度下计算 softmax，再转回原始 dtype
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(q.dtype)
            attn_weights = nn.functional.dropout(
                attn_weights, p=self.dropout, training=False
            )
            output = torch.matmul(attn_weights, v)
            del attn_weights, v
        else:
            # SDPA
            # [b, h, s, head_size]  使用 PyTorch 内置高效 SDPA kernel
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=self.dropout,
                is_causal=False,
                scale=self.scale,
            )

        # [b, h, s, head_size] --> [b * s, h, head_size]  恢复到打平的 token 维度
        output = rearrange(output, "b h s d -> (b s) h d")

        return output


class VisionTritonAttention(nn.Module):
    """
    Triton-implemented attention without a causal mask
    使用 Triton kernel 实现的非因果注意力（视觉编码器专用）
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()
        # 如果使用数据并行则 tp_size=1，否则使用张量并行大小
        use_data_parallel = (
            kwargs["use_data_parallel"] if "use_data_parallel" in kwargs else False
        )
        self.tp_size = 1 if use_data_parallel else get_attention_tp_size()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | SingletonCache | None,
        bsz: int,
        seq_len: int,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
            softmax_scale: override softmax scale (default 1/sqrt(head_dim))
        Returns:
             [b * s, h, head_size]
        """
        if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():
            # CUDA Graph 模式：使用预分配的输出工作区，cu_seqlens 为列表格式
            if "output_ws" not in kwargs:
                raise RuntimeError("output_ws should be prepared for cuda-graph mode")

            if not isinstance(cu_seqlens, list):
                raise RuntimeError("cuda-graph mode cu_seqlens should be a list")

            output = kwargs["output_ws"]
            # cu_seqlens[0]=indptr, [1]=seq_lens, [2]=max_seqlen
            context_attention_fwd(
                q,
                k,
                v,
                output,
                cu_seqlens[0],
                cu_seqlens[1],
                cu_seqlens[2],
                is_causal=False,
                sm_scale=softmax_scale,
            )
        else:
            # 普通模式：解析 cu_seqlens 并动态分配输出
            cu_seqlens = resolve_seqlens(cu_seqlens, bsz, seq_len, device=q.device)

            # [b * s, head, head_size]  分配与 q 相同形状和 dtype 的输出张量
            output = torch.empty_like(q)

            # 计算每个序列的实际长度及最大长度
            seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            max_seqlen = seq_lens.max().item()
            context_attention_fwd(
                q,
                k,
                v,
                output,
                cu_seqlens.to(q.device),
                seq_lens.to(q.device),
                max_seqlen,
                is_causal=False,
                sm_scale=softmax_scale,
            )

        return output


class VisionFlash3Attention(nn.Module):
    # 使用 FlashAttention v3 的视觉注意力模块（仅支持 CUDA 或 MUSA）
    def __init__(
        self,
        **kwargs,
    ):
        if not (_is_cuda or _is_musa):
            raise Exception("VisionFlash3Attention is only available for cuda or musa")
        super().__init__()
        use_data_parallel = (
            kwargs["use_data_parallel"] if "use_data_parallel" in kwargs else False
        )
        self.tp_size = 1 if use_data_parallel else get_attention_tp_size()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | SingletonCache | None,
        bsz: int,
        seq_len: int,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        # 获取滑动窗口大小和 sink token 辅助张量
        window_size = kwargs.get("window_size", (-1, -1))
        s_aux = kwargs.get("s_aux", None)

        if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():
            # CUDA Graph 模式：cu_seqlens 为 [indptr, max_seqlen] 格式
            max_seqlen = cu_seqlens[1]
            fa_kwargs = dict(
                cu_seqlens_q=cu_seqlens[0],
                cu_seqlens_k=cu_seqlens[0],
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=softmax_scale,
                window_size=window_size,
            )
            if s_aux is not None:
                fa_kwargs["sinks"] = s_aux
            output = flash_attn_varlen_func(q, k, v, **fa_kwargs)
        else:
            # 普通模式：解析 cu_seqlens 并动态计算 max_seqlen
            cu_seqlens = resolve_seqlens(cu_seqlens, bsz, seq_len, device=q.device)
            cu_seqlens = cu_seqlens.to(dtype=torch.int32).to(q.device)
            seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            max_seqlen = seq_lens.max().item()

            fa_kwargs = dict(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=softmax_scale,
                window_size=window_size,
            )
            if s_aux is not None:
                fa_kwargs["sinks"] = s_aux
            output = flash_attn_varlen_func(q, k, v, **fa_kwargs)

        return output


class VisionFlash4Attention(nn.Module):
    # 使用 FlashAttention v4 的视觉注意力模块（仅支持 CUDA，适用于 Blackwell GPU）
    def __init__(
        self,
        **kwargs,
    ):
        if not _is_cuda:
            raise Exception("VisionFlash4Attention is only available for cuda")
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | SingletonCache | None,
        bsz: int,
        seq_len: int,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        # 解析 cu_seqlens（None / SingletonCache 均需处理）
        if cu_seqlens is None:
            cu_seqlens = _get_cu_seqlens_for_shape(bsz, seq_len, device=q.device)
        elif isinstance(cu_seqlens, SingletonCache):
            if cu_seqlens.empty():
                cu_seqlens.set_data(
                    _get_cu_seqlens_for_shape(bsz, seq_len, device=q.device)
                )
            cu_seqlens = cu_seqlens.get_data()

        # 转为 int32 并计算每序列长度和最大长度
        cu_seqlens = cu_seqlens.to(dtype=torch.int32).to(q.device)
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = seq_lens.max().item()

        # 调用 FlashAttention v4（ver=4 参数）
        output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale,
            ver=4,
        )

        return output


class VisionFlashInferAttention(nn.Module):
    # 使用 FlashInfer cuDNN batch prefill 的视觉注意力模块（仅支持 CUDA）
    def __init__(
        self,
        **kwargs,
    ):
        if not _is_cuda:
            raise Exception("VisionFlashInferAttention is only available for cuda")
        super().__init__()
        # 工作区缓冲（128MB），用于 cuDNN 内部临时存储
        self.workspace_buffer = (
            kwargs["workspace_buffer"] if "workspace_buffer" in kwargs else None
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | SingletonCache | None,
        bsz: int,
        seq_len: int,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        # 校验必须提供 sequence_lengths 和 max_seqlen
        if "sequence_lengths" not in kwargs:
            raise RuntimeError(
                "sequence_lengths should be prepared for vision flashinfer_cudnn attention backend"
            )
        if "max_seqlen" not in kwargs:
            raise RuntimeError(
                "max_seqlen should be prepared for vision flashinfer_cudnn attention backend"
            )

        sequence_lengths = kwargs["sequence_lengths"]  # (B_padded,) or (B_padded,1,1,1)
        max_seqlen = kwargs["max_seqlen"]

        # max_seqlen must be python int
        # max_seqlen 必须转为 Python int（cuDNN API 要求）
        if isinstance(max_seqlen, torch.Tensor):
            if max_seqlen.is_cuda:
                max_seqlen = int(max_seqlen.detach().cpu().item())
            else:
                max_seqlen = int(max_seqlen.item())
        else:
            max_seqlen = int(max_seqlen)

        # flatten if caller gives (b, s, h, d)
        # 如果输入为 4D (b, s, h, d)，将其展平为 (b*s, h, d)
        is_reshaped = q.dim() == 4
        if is_reshaped:
            reshape_batch_size = q.shape[0]
            q, k, v = (rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])

        if not isinstance(cu_seqlens, torch.Tensor):
            raise RuntimeError(
                "flashinfer_cudnn expects packed indptrs as a torch.Tensor"
            )

        # sequence_lengths -> (B,)
        if not isinstance(sequence_lengths, torch.Tensor):
            raise RuntimeError("sequence_lengths must be a torch.Tensor")
        seq_lens_1d = sequence_lengths.view(-1).to(device=q.device, dtype=torch.int32)
        B = int(seq_lens_1d.numel())

        # cu_seqlens contains packed *element indptrs*:
        # [qk_indptr(B+1), v_indptr(B+1), o_indptr(B+1)] => total 3*(B+1)
        # cu_seqlens 是打包的元素级 indptr，包含 qk/v/output 三段
        cu_seqlens_1d = cu_seqlens.view(-1).to(device=q.device, dtype=torch.int32)
        expected = 3 * (B + 1)
        if int(cu_seqlens_1d.numel()) != expected:
            raise RuntimeError(
                f"packed indptr numel mismatch: got {cu_seqlens_1d.numel()}, expected {expected} (= 3*(B+1))"
            )

        # 将打包的 indptr 分成三段，分别对应 QK/V/Output 的偏移
        split = B + 1
        indptr_qk = cu_seqlens_1d[:split].view(split, 1, 1, 1)
        indptr_v = cu_seqlens_1d[split : 2 * split].view(split, 1, 1, 1)
        indptr_o = cu_seqlens_1d[2 * split :].view(split, 1, 1, 1)

        # cuDNN style: (B,1,1,1)
        # 将序列长度整形为 cuDNN 要求的 (B,1,1,1) 格式
        seq_lens_4d = seq_lens_1d.view(B, 1, 1, 1)

        # indptr are in ELEMENT offsets (not token offsets)
        # 每个 token 的元素数量 = heads * head_dim
        token_width_q = int(q.shape[1] * q.shape[2])  # heads * head_dim on this rank
        total_elems_q = int(q.numel())

        # check each real sequence fits
        # (skip padded tail where seq_len==0)
        # 验证每个真实序列的数据范围不超出 q 张量边界
        start_elems = indptr_qk.view(-1)[:-1]  # (B,)
        end_elems = start_elems + seq_lens_1d * token_width_q
        if (end_elems > total_elems_q).any():
            raise RuntimeError("offset + len out of bounds; packed indptr is wrong")

        _, _, head_size = q.shape
        scale = softmax_scale if softmax_scale is not None else head_size**-0.5

        # 调用 cuDNN batch prefill 计算非因果注意力
        output, _ = cudnn_batch_prefill_with_kv_cache(
            q,
            k,
            v,
            scale,
            self.workspace_buffer,
            max_token_per_sequence=max_seqlen,
            max_sequence_kv=max_seqlen,
            actual_seq_lens_q=seq_lens_4d,
            actual_seq_lens_kv=seq_lens_4d,
            causal=False,
            return_lse=True,
            batch_offsets_q=indptr_qk,
            batch_offsets_k=indptr_qk,
            batch_offsets_v=indptr_v,
            batch_offsets_o=indptr_o,
            is_cuda_graph_compatible=True,
        )

        # 如果输入被展平，则还原为 (b, s, h, d) 格式
        if is_reshaped:
            output = rearrange(output, "(b s) h d -> b s h d", b=reshape_batch_size)

        return output


class VisionAiterAttention(nn.Module):
    # 使用 aiter 库（AMD ROCm 专用）的视觉注意力模块
    def __init__(
        self,
        **kwargs,
    ):
        if not _is_hip:
            raise Exception("aiter_attn is only available for AMD")
        try:
            from aiter import flash_attn_varlen_func as aiter_flash_attn_varlen_func
        except ImportError as e:
            raise ImportError(
                "aiter is AMD specific kernel library. Please make sure aiter is installed on your AMD device."
            ) from e

        # 保存 aiter 提供的 varlen flash attention 函数
        self.flash_attn_varlen_func = aiter_flash_attn_varlen_func
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | SingletonCache | None,
        bsz: int,
        seq_len: int,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        # 解析 cu_seqlens 并计算序列长度统计
        cu_seqlens = resolve_seqlens(cu_seqlens, bsz, seq_len, device=q.device)

        cu_seqlens = cu_seqlens.to(dtype=torch.int32).to(q.device)
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = seq_lens.max().item()

        # 调用 aiter 的非因果 varlen flash attention
        return self.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale,
        )


class VisionAscendAttention(nn.Module):
    # 使用昇腾 NPU flash attention 的视觉注意力模块
    def __init__(
        self,
        **kwargs,
    ):
        if not _is_npu:
            raise Exception("VisionAscendAttention is only available for ascend npu")
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | SingletonCache | None,
        bsz: int,
        seq_len: int,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():
            # NPU Graph 模式：使用预分配的输出工作区
            if "output_ws" not in kwargs:
                raise RuntimeError("output_ws should be prepared for npu-graph mode")
            output = kwargs["output_ws"]
            seq_len_arg = cu_seqlens
        else:
            # 普通模式：在 CPU 上解析 cu_seqlens（NPU 不支持直接计算差分）
            cu_seqlens = resolve_seqlens(cu_seqlens, bsz, seq_len, device="cpu")
            seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            if seq_lens.is_npu:
                seq_lens = seq_lens.to("cpu")
            output = torch.empty_like(q)
            seq_len_arg = seq_lens.to(torch.int32)

        _, num_heads, head_size = q.shape
        num_kv_heads = k.shape[1]

        # 计算注意力缩放系数（未提供时使用默认值）
        scale_value = softmax_scale if softmax_scale is not None else head_size**-0.5

        # 调用昇腾 NPU 专用的 flash attention unpad 接口
        torch_npu._npu_flash_attention_unpad(
            query=q,
            key=k,
            value=v,
            seq_len=seq_len_arg,
            scale_value=scale_value,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            out=output,
        )
        return output


# 注意力后端名称到实现类的映射表，供 VisionAttention 统一选择
QKV_BACKEND_IMPL = {
    "triton_attn": VisionTritonAttention,
    "sdpa": VisionSdpaAttention,
    "fa3": VisionFlash3Attention,
    "fa4": VisionFlash4Attention,
    "flashinfer_cudnn": VisionFlashInferAttention,
    "ascend_attn": VisionAscendAttention,
    "aiter_attn": VisionAiterAttention,
}


class VisionAttention(nn.Module):
    r"""
        Multi-headed attention without any cache, mostly used for multimodal transformers.
        无 KV Cache 的多头注意力模块，主要用于多模态视觉编码器。

    Args:
        use_qkv_parallel (bool, optional): If True, use QKV-parallel attention.
        softmax_in_single_precision (bool, default to False):
            if ``True``, the softmax will be performed in single-precision
            Otherwise, it will be performed in half-precision

    """

    def __init__(
        self,
        embed_dim: int,          # 输入嵌入维度
        num_heads: int,          # Query 注意力头数
        projection_size: int,    # 投影层输出大小
        use_qkv_parallel: bool,  # 是否使用 QKV 并行线性层（GQA/MQA 优化）
        num_kv_heads: Optional[int] = None,   # KV 头数（None 时与 num_heads 相同）
        head_dim: Optional[int] = None,       # 每头的特征维度（None 时从 embed_dim 推导）
        qkv_backend: Optional[str] = None,    # 强制指定注意力后端（None 时自动选择）
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        dropout: float = 0.0,
        softmax_in_single_precision: bool = False,
        softmax_scale: Optional[float] = None,
        flatten_batch: bool = False,
        prefix: str = "",
        proj_bias: bool = True,
        num_dummy_heads: int = 0,  # 填充头数量，用于 TP 对齐
        qkv_bias: bool = True,
        qk_normalization: bool = False,            # 是否对 Q/K 做全局 RMSNorm（InternVL 用）
        qk_normalization_by_head_size: bool = False,  # 是否按 head_size 做 RMSNorm（GLM-OCR 用）
        layer_norm_eps: float = 1e-06,
        customized_position_embedding_applier: Callable[
            [torch.Tensor, torch.Tensor, Any, Any], Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # 自定义位置编码应用函数（可选）
        use_data_parallel: bool = False,     # 是否使用数据并行（替代张量并行）
        use_dp_attention_reduce: bool = False,
        aux_stream: Optional[torch.cuda.Stream] = None,  # 用于 QK Norm 并行计算的辅助 CUDA 流
        workspace_buffer: Optional[torch.Tensor] = None,  # cuDNN 工作区缓冲
        use_sink: bool = False,              # 是否使用 sink tokens（滑动窗口注意力辅助）
        window_size: Tuple[int, int] = (-1, -1),  # 滑动窗口注意力范围（-1 表示全局注意力）
        **kwargs,
    ):
        super().__init__()
        # 兼容旧版 head_size 参数名（已废弃，改用 head_dim）
        if head_dim is None and "head_size" in kwargs:
            head_dim = kwargs.pop("head_size")
            warnings.warn(
                "VisionAttention(head_size=...) is deprecated; use head_dim=...",
                DeprecationWarning,
                stacklevel=2,
            )
        # 数据并行时 tp_size=1，否则使用张量并行大小
        self.tp_size = 1 if use_data_parallel else get_attention_tp_size()
        self.tp_rank = 0 if use_data_parallel else get_attention_tp_rank()
        self.dropout = dropout
        # 如果未指定 num_kv_heads，则与 num_heads 相同（MHA 模式）
        num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_size = head_dim if head_dim is not None else embed_dim // num_heads
        # 每个注意力头在投影层后的隐藏大小
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads
        )
        # 每个 TP rank 负责的 Query head 数（含 dummy heads）
        self.num_attention_heads_per_partition = dist_utils.divide(
            num_dummy_heads + num_heads, self.tp_size
        )
        # 每个 TP rank 负责的 KV head 数（含 dummy heads）
        self.num_attention_kv_heads_per_partition = dist_utils.divide(
            num_dummy_heads + num_kv_heads, self.tp_size
        )

        # 当前 TP rank 的 Q 和 KV 总特征维度
        self.q_size = self.num_attention_heads_per_partition * self.head_size
        self.kv_size = self.num_attention_kv_heads_per_partition * self.head_size

        self.qk_normalization = qk_normalization
        self.qk_normalization_by_head_size = qk_normalization_by_head_size

        # Additional dummy heads are used to enable TP for common GPU counts.
        # dummy heads 用于使总 head 数可以整除 tp_size
        self.dummy_dim = (num_dummy_heads + num_heads) * self.head_size

        if self.qk_normalization:
            # InternVL 风格：对全局 QK 做 RMSNorm
            self.q_norm, self.k_norm = self._init_qk_norm(
                self.dummy_dim, layer_norm_eps, embed_dim
            )

        elif self.qk_normalization_by_head_size:
            # GLM-OCR 风格：对每个 head 的特征维度做 RMSNorm
            self.q_norm, self.k_norm = self._init_qk_norm(
                self.head_size, layer_norm_eps
            )

        # Select attention backend via a unified method
        # 通过统一方法选择最优注意力后端
        _passed_backend = qkv_backend
        qkv_backend = self._determine_attention_backend(_passed_backend)
        if (
            get_global_server_args().mm_attention_backend is None
            and _passed_backend is None
        ):
            print_info_once(f"Multimodal attention backend not set. Use {qkv_backend}.")
        print_info_once(f"Using {qkv_backend} as multimodal attention backend.")

        self.customized_position_embedding_applier = (
            customized_position_embedding_applier
        )
        self.softmax_scale = softmax_scale
        # 实例化选定的注意力后端
        self.qkv_backend = QKV_BACKEND_IMPL[qkv_backend](
            head_dim=self.head_size,
            num_heads=self.num_attention_heads_per_partition,
            num_kv_heads=self.num_attention_kv_heads_per_partition,
            dropout=dropout,
            flatten_batch=flatten_batch,
            softmax_in_single_precision=softmax_in_single_precision,
            softmax_scale=softmax_scale,
            use_data_parallel=use_data_parallel,
            workspace_buffer=workspace_buffer,
        )

        self.use_qkv_parallel = use_qkv_parallel
        if use_qkv_parallel:
            # QKV 并行：使用专门的 QKVParallelLinear（支持 GQA/MQA）
            self.qkv_proj = QKVParallelLinear(
                hidden_size=embed_dim,
                head_size=self.head_size,
                total_num_heads=num_dummy_heads + num_heads,
                total_num_kv_heads=num_dummy_heads + num_kv_heads,
                bias=qkv_bias,
                quant_config=quant_config,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                prefix=add_prefix("qkv_proj", prefix),
            )
        else:
            # 非 QKV 并行：使用列并行线性层，输出 3 * dummy_dim
            self.qkv_proj = ColumnParallelLinear(
                input_size=embed_dim,
                output_size=3 * self.dummy_dim,
                bias=qkv_bias,
                quant_config=quant_config,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                prefix=add_prefix("qkv_proj", prefix),
            )
        # 输出投影层（行并行线性，用于 all-reduce）
        self.proj = RowParallelLinear(
            input_size=self.dummy_dim,
            output_size=embed_dim,
            bias=proj_bias,
            quant_config=quant_config,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            prefix=add_prefix("proj", prefix),
            use_dp_attention_reduce=use_dp_attention_reduce,
        )

        self.workspace_buffer = workspace_buffer
        self.aux_stream = aux_stream
        # 多流并行（Q/K Norm）使用的 CUDA 事件对
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()] if aux_stream else []

        self.window_size = window_size
        if use_sink:
            # Allocate the full (unsharded) sink tensor for weight loading;
            # only the local TP slice is used in forward.
            # 分配完整 sink 权重用于加载，前向时只使用当前 TP rank 对应的切片
            self.sinks = nn.Parameter(
                torch.empty(
                    self.num_attention_heads_per_partition * self.tp_size,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
        else:
            self.sinks = None

    def _init_qk_norm(
        self, norm_dim: int, eps: float, var_hidden_size: Optional[int] = None
    ):
        # 初始化 QK Norm 的 RMSNorm 层，支持 RL 训练时的特殊参数
        norm_kwargs = (
            dict(
                weight_dtype=torch.float32,
                cast_x_before_out_mul=True,
            )
            if get_global_server_args().rl_on_policy_target is not None
            else {}
        )
        q_norm = RMSNorm(
            norm_dim,
            eps=eps,
            var_hidden_size=var_hidden_size,
            **norm_kwargs,
        )
        k_norm = RMSNorm(
            norm_dim,
            eps=eps,
            var_hidden_size=var_hidden_size,
            **norm_kwargs,
        )
        return q_norm, k_norm

    def _determine_attention_backend(self, passed_backend: Optional[str]) -> str:
        """Decide the multimodal attention backend string.
        决定多模态注意力后端字符串。

        Priority: server args override > constructor arg > platform default.
        优先级：服务器参数覆盖 > 构造函数参数 > 平台默认值。

        Platform defaults:
        - CUDA (Hopper SM90): "fa3"
        - CUDA (Blackwell SM100): "fa4"
        - CUDA (other): "triton_attn"
        - Non-CUDA: "sdpa"
        """
        # 最高优先级：全局服务器参数中指定的 mm_attention_backend
        override_backend = get_global_server_args().mm_attention_backend
        if override_backend is not None:
            backend = override_backend
        elif passed_backend is not None:
            # 次优先级：构造函数传入的后端名称
            backend = passed_backend
        elif is_cuda():
            # CUDA 平台按 GPU 架构自动选择
            major, minor = get_device_capability()
            if major == 9:
                backend = "fa3"   # Hopper (H100/H200)
            elif major == 10:
                backend = "fa4"   # Blackwell
            else:
                backend = "triton_attn"  # 其他 CUDA GPU 使用 Triton
        elif _is_musa:
            # 摩尔线程 MUSA 平台
            if get_device_capability() >= (3, 1):
                backend = "fa3"
            else:
                backend = "triton_attn"
        elif _is_hip:
            # AMD ROCm 平台，如满足条件且启用 aiter 则使用 aiter
            if get_device_capability() >= (9, 4) and _use_aiter:
                backend = "aiter_attn"
            else:
                backend = "triton_attn"
        elif _is_xpu:
            # Intel XPU 平台使用 Triton
            backend = "triton_attn"
        else:
            # 其他平台（CPU 等）回退到 SDPA
            backend = "sdpa"
        # Blackwell GPU 不支持 fa3
        if backend == "fa3" and is_blackwell_supported():
            raise ValueError("The 'fa3' backend is not supported on Blackwell GPUs")

        return backend

    def _apply_qk_norm_head_size(self, q: torch.Tensor, k: torch.Tensor):
        """apply qk norm for GLM-OCR vit attn
        按 head_size 维度对 Q/K 做 RMSNorm（GLM-OCR ViT 注意力专用）
        """
        # 将 Q/K 重塑为 (total_tokens * num_heads, head_size)，逐 head 归一化
        q_by_head = q.reshape(-1, self.head_size)
        q_by_head = self.q_norm(q_by_head)
        k_by_head = k.reshape(-1, self.head_size)
        k_by_head = self.k_norm(k_by_head)
        q = q_by_head.view(q.shape)
        k = k_by_head.view(k.shape)
        return q, k

    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor):
        """apply qk norm for internvl vit attn
        对 InternVL ViT 注意力的 Q/K 做全局 RMSNorm（支持 TP all-gather）
        """

        def q_l2norm():
            # 展平 Q 的 head 维度后做 all-gather（张量并行时跨 rank 聚合）
            q_ = q.flatten(1, 2)
            if self.tp_size > 1:
                q_ = tensor_model_parallel_all_gather(q_.contiguous())
            q_ = self.q_norm(q_)
            if self.tp_size > 1:
                # 归一化后重新按 tp_rank 切分
                splitter = partial(
                    split_tensor_along_last_dim, num_partitions=self.tp_size
                )
                q_ = splitter(q_)[self.tp_rank]
            q_ = q_.unflatten(-1, (-1, self.head_size))
            return q_

        def k_l2norm():
            # K 与 Q 同理，展平→all-gather→归一化→切分
            k_ = k.flatten(1, 2)
            if self.tp_size > 1:
                k_ = tensor_model_parallel_all_gather(k_.contiguous())
            k_ = self.k_norm(k_)
            if self.tp_size > 1:
                splitter = partial(
                    split_tensor_along_last_dim, num_partitions=self.tp_size
                )
                k_ = splitter(k_)[self.tp_rank]
            k_ = k_.unflatten(-1, (-1, self.head_size))
            return k_

        # 使用辅助流并行执行 Q 和 K 的归一化，减少串行等待
        with with_multi_stream(True):
            q, k = maybe_execute_in_parallel(
                q_l2norm,
                k_l2norm,
                self.ln_events,
                self.aux_stream,
            )
        return q, k

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        full_attn: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            x: [b, s, embed_dim]  输入特征张量
            cu_seqlens: [b]       累积序列长度（变长序列时使用）
        Returns:
             [s, b, head * head_size]  or [b, s, embed_dim]
        """
        # 处理 2D 输入（单序列场景），扩充 batch 维
        if x.dim() == 2:
            x = x.unsqueeze(0)
        assert x.dim() == 3, x.shape
        if (
            get_global_server_args().rl_on_policy_target is not None
            and position_embeddings is not None
        ):
            # RL on-policy 训练时确保 position_embeddings 为 tuple，并转换到 x 的 dtype
            assert isinstance(position_embeddings, tuple), (
                "expected position_embeddings to be a tuple of two tensors,\n"
                f"but got {type(position_embeddings)}, change if needed"
            )
            position_embeddings = tuple(p.to(x.dtype) for p in position_embeddings)
        x_shape = x.shape
        bsz, s, _ = x_shape
        head = self.num_attention_heads_per_partition
        kv_head = self.num_attention_kv_heads_per_partition

        # 从 kwargs 中提取可选的预分配输出工作区、max_seqlen 和序列长度
        attn_output_ws = kwargs["output_ws"] if "output_ws" in kwargs else None
        max_seqlen = kwargs["max_seqlen"] if "max_seqlen" in kwargs else None
        sequence_lengths = (
            kwargs["sequence_lengths"] if "sequence_lengths" in kwargs else None
        )
        if self.use_qkv_parallel:
            # QKV 并行路径：直接用 QKVParallelLinear 生成 Q/K/V
            # [b, s, embed_dim] --> [b, s, embed_dim]
            qkv, _ = self.qkv_proj(x)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            # [b, s, embed_dim] --> [b * s, head, head_size]
            q = q.reshape(bsz * s, head, -1).contiguous()
            k = k.reshape(bsz * s, kv_head, -1).contiguous()
            v = v.reshape(bsz * s, kv_head, -1).contiguous()
            if self.qk_normalization_by_head_size:
                q, k = self._apply_qk_norm_head_size(q, k)
        else:
            # 非 QKV 并行路径：使用 ColumnParallelLinear
            # [b, s, embed_dim] --> [s, b, embed_dim]
            x = rearrange(x, "b s ... -> s b ...")
            # [s, b, embed_dim] --> [s, b, head * 3 * head_size]
            qkv, _ = self.qkv_proj(x)

            # [s, b, head, head_dim_sum]  重排以便切分 Q/K/V
            new_x_shape = qkv.size()[:-1] + (
                head,
                self.q_size + 2 * self.kv_size,
            )
            qkv = qkv.view(*new_x_shape)

            # [s, b, head, 3 * head_size] --> 3 [s, b, head, head_size]
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            # [s, b, head, head_size] --> [b, s, head, head_size]
            q, k, v = [
                rearrange(x, "s b ... -> b s ...").contiguous() for x in (q, k, v)
            ]

            if self.qk_normalization_by_head_size:
                q, k = self._apply_qk_norm_head_size(q, k)

        cos = None
        sin = None

        if position_embeddings is not None:
            if self.customized_position_embedding_applier is not None:
                # 使用自定义位置编码应用器（如 2D RoPE、ALiBi 等）
                q, k = self.customized_position_embedding_applier(
                    q, k, position_embeddings, x_shape
                )
            else:
                # 标准 RoPE：从 position_embeddings 提取 cos/sin
                cos, sin = position_embeddings
        elif rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            # 直接传入的 cos/sin 向量
            cos = rotary_pos_emb_cos
            sin = rotary_pos_emb_sin

        if cos is not None and sin is not None:
            original_q_shape = q.shape
            original_k_shape = k.shape

            # [total_tokens, head, head_size] for q / [total_tokens, kv_head, head_size] for k
            # 展平为 token 维以应用 RoPE
            q = q.view(-1, head, self.head_size)
            k = k.view(-1, kv_head, self.head_size)

            # 如果 cos 只提供了半维度，则复制为全维度（适用于某些 RoPE 实现）
            if cos.size(-1) * 2 == self.head_size:
                cos = torch.cat([cos, cos], dim=-1)
                sin = torch.cat([sin, sin], dim=-1)

            # 应用旋转位置编码
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            q = q.view(original_q_shape)
            k = k.view(original_k_shape)

        if q.dim() == 4:
            # [b, s, head, head_size] --> [b * s, head, head_size]
            q = rearrange(q, "b s ... -> (b s) ...")
        if k.dim() == 4:
            # [b, s, head, head_size] --> [b * s, head, head_size]
            k = rearrange(k, "b s ... -> (b s) ...")
        if v.dim() == 4:
            # [b, s, head, head_size] --> [b * s, head, head_size]
            v = rearrange(v, "b s ... -> (b s) ...")

        assert q.dim() == 3, q.dim()
        assert k.dim() == 3, k.dim()
        assert v.dim() == 3, v.dim()

        # internvl
        # InternVL 风格：对 Q/K 做全局 RMSNorm（与 head_size 归一化互斥）
        if self.qk_normalization and not self.qk_normalization_by_head_size:
            # jit kernel
            if can_use_jit_qk_norm(self.head_size, q.dtype):

                # q: [tokens, head, head_size]  ->  [tokens, embed_dim]
                # 使用 JIT 融合 kernel 加速 QK Norm
                head_dim_for_norm = head * self.head_size

                q, k = apply_qk_norm(
                    q=q,
                    k=k,
                    q_norm=self.q_norm,
                    k_norm=self.k_norm,
                    head_dim=head_dim_for_norm,
                    alt_stream=self.aux_stream,
                )

            else:
                # 回退到 Python 实现的 QK Norm（支持 all-gather）
                q, k = self._apply_qk_norm(q, k)

        # 确定是否使用滑动窗口注意力
        if full_attn or self.sinks is None:
            effective_window_size = (-1, -1)   # 全局注意力
            s_aux = None
        else:
            effective_window_size = self.window_size  # 滑动窗口
            # 取当前 TP rank 对应的 sink tokens 切片
            q_head_start = self.tp_rank * self.num_attention_heads_per_partition
            q_head_end = (self.tp_rank + 1) * self.num_attention_heads_per_partition
            s_aux = self.sinks[q_head_start:q_head_end]

        # 调用选定的注意力后端执行注意力计算
        output = self.qkv_backend.forward(
            q=q,
            k=k,
            v=v,
            bsz=bsz,
            seq_len=s,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            sequence_lengths=sequence_lengths,
            max_seqlen=max_seqlen,
            output_ws=attn_output_ws,
            softmax_scale=self.softmax_scale,
            window_size=effective_window_size,
            s_aux=s_aux,
        )

        assert output.dim() == 3, output.shape

        if self.use_qkv_parallel:
            # [b * s, h, head_size] --> [b, s, h * head_size]  QKV 并行路径的输出重排
            output = rearrange(output, "(b s) ... h d -> b s ... (h d)", b=bsz)

            # [b, s, h * head_size] --> [b, s, h * head_size]  输出投影
            output, _ = self.proj(output)
        else:
            # [b * s, h, head_size] --> [s, b, h * head_size]  非 QKV 并行路径输出重排
            context_layer = rearrange(
                output, "(b s) h d -> s b (h d)", b=bsz, s=s
            ).contiguous()

            # [s, b, h * head_size] --> [s, b, h * head_size]  输出投影
            output, _ = self.proj(context_layer)

            # [s, b, h * head_size] --> [b, s, h * head_size]  恢复 batch-first 格式
            output = output.view(bsz, s, -1)

        return output
