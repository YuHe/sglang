# Mamba2 混合器核心模块：实现 Mamba2 SSM 的完整前向计算，含 TP 分片、prefill/decode 两路处理
import logging
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn

# 导入 Mamba2 缓存形状参数和 TP extra groups 工具
from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    extra_groups_for_head_shards,
)
# 导入张量并行工具
from sglang.srt.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.attention.mamba.mamba2_metadata import Mamba2Metadata
from sglang.srt.layers.attention.mamba.mixer2_rms_norm_gated import Mixer2RMSNormGated
# 导入 SSM 主要算子：chunk scan 和 selective state update
from sglang.srt.layers.attention.mamba.ops import (
    mamba_chunk_scan_combined,
    selective_state_update,
)
# 导入线性层（支持张量并行）
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_loader.weight_utils import (
    composed_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.utils import (
    is_cpu,
    is_cuda,
    is_npu,
    set_weight_attrs,
)

# 按后端加载因果卷积接口
if is_cuda():
    from sglang.srt.layers.attention.mamba.causal_conv1d import (
        causal_conv1d_fn,
        causal_conv1d_update,
    )
    from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
        causal_conv1d_fn as causal_conv1d_fn_triton,
    )
    from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
        causal_conv1d_update as causal_conv1d_update_triton,
    )
elif is_npu():
    # NPU 后端使用 sgl_kernel_npu 实现
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_fn_npu as causal_conv1d_fn,
    )
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_update_npu as causal_conv1d_update,
    )

# 权重加载器函数类型别名
LoaderFunction = Callable[[torch.Tensor, torch.Tensor], None]

logger = logging.getLogger(__name__)


def mamba_v2_sharded_weight_loader(
    shard_spec: List[Tuple[int, int, float]],
    tp_size: int,
    tp_rank: int,
) -> LoaderFunction:
    """为 Mamba v2 创建自定义权重加载器，确保 x/B/C 投影按正确方式分片，
    并将与 head shard 关联的 group 维度一起分配。
    Create a weight loader for mamba v2. This ensures that the projections
    are correctly sharded so that they can be split into x, B, C. It also
    ensures that all the groups corresponding to a head shard is placed
    together with it.
    """

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:

        # - track boundary of (sharded) param, and loaded_weight, respectively
        # 分别追踪目标 param 和源 loaded_weight 的写入/读取边界
        boundary, loaded_boundary = 0, 0

        # Calculate padding size for CPU when TP odd size
        # CPU 路径：当 TP 奇数分片导致权重无法整除时，需要 zero-padding 补齐
        if is_cpu():
            full_dim_sum = 0
            full_dim_list = []
            weight_full_dim_list = []
            for full_dim, _, _ in shard_spec:
                full_dim_sum = full_dim_sum + full_dim
                full_dim_list.append(full_dim)
            for full_dim in full_dim_list:
                weight_full_dim_list.append(
                    int(full_dim / full_dim_sum * loaded_weight.size(0))
                )
            assert sum(weight_full_dim_list) == loaded_weight.size(
                0
            ), f"Padding the loaded weight failed due to sizes are not divisible cleanly from {weight_full_dim_list} to {loaded_weight.size(0)}"
            if loaded_weight.size(0) < full_dim_sum and tp_rank == 0:
                logger.warning(
                    f"[ZERO-PADDING] Loaded_weight.dim(0) size:{loaded_weight.size(0)} is padding to {full_dim_sum}"
                    f", where original sizes of {weight_full_dim_list} will be updated to {full_dim_list}",
                )

        # - iterate over the shard specs
        # 遍历每个分片规格 (full_dim, extra, duplicate_groups)
        for full_dim, extra, duplicate_groups in shard_spec:
            # - full dim is the model dim (before TP).
            # - extra > 0, means there is expected overall increase
            #   of dimensions. This is so because of replication.
            # - ratio is used map the tp_rank to the actual shard
            #   rank. This is useful when there is replication of
            #   groups to accompany head shards.
            # full_dim: 模型完整维度（TP 前）；extra: 复制引起的额外维度；
            # duplicate_groups: 是否需要 group 复制（n_groups==1 时为 True）

            # - size of the loaded shard
            # 每个 rank 对应的分片大小
            shard_size = full_dim // tp_size

            # - compute the rank into the loaded shard.
            # - if there is replication, different TP shards will
            #   take from the same rank.
            # NOTE: currently we only support duplication
            # in the case where num_groups == 1
            # 若 duplicate_groups 为 True，所有 rank 均从 rank=0 加载（复制模式）
            rank = 0 if duplicate_groups else tp_rank

            # - leftmost boundary index into loaded weight.
            # 计算从 loaded_weight 中读取的起始位置
            loaded_skip = rank * shard_size
            loaded_start_idx = loaded_boundary + loaded_skip

            # - take these many dims from the loaded weight.
            # 实际需要从 loaded_weight 中取的维度数（防止越界）
            take = min(shard_size, full_dim - extra - loaded_skip)

            # CPU logic of padding size for qwen3-next
            # TODO : make this common for all mamba.
            # CPU 模式：对尺寸不足的权重进行 zero-padding 以匹配目标形状
            if is_cpu() and (loaded_weight.size(0) < full_dim_sum):
                import copy

                loaded_weight_ = copy.deepcopy(loaded_weight)
                q, k, v = torch.split(
                    loaded_weight_,
                    weight_full_dim_list,
                    dim=0,
                )
                pad_qk = torch.zeros(
                    full_dim_list[0] - weight_full_dim_list[0],
                    loaded_weight.size(1),
                    loaded_weight.size(2),
                ).to(loaded_weight.dtype)
                pad_v = torch.zeros(
                    full_dim_list[2] - weight_full_dim_list[2],
                    loaded_weight.size(1),
                    loaded_weight.size(2),
                ).to(loaded_weight.dtype)
                q = torch.cat((q, pad_qk), dim=0)
                k = torch.cat((k, pad_qk), dim=0)
                v = torch.cat((v, pad_v), dim=0)
                loaded_weight_qk = torch.cat((q, k), dim=0)
                loaded_weight = torch.cat((loaded_weight_qk, v), dim=0)

            # - always shard on dim 0
            # - the ignore is for a mundane mypy error as it does not
            #   seem to handle slices well.
            # https://github.com/python/mypy/issues/2410
            # 将 loaded_weight 对应片段写入 param，沿 dim 0 分片
            param.data[
                boundary : (boundary + take), ...  # type: ignore[misc]
            ] = loaded_weight[
                loaded_start_idx : (loaded_start_idx + take)  # type: ignore[misc]
            ]  # type: ignore[misc]

            # move indexing boundaries
            # 更新两侧的边界指针
            boundary += shard_size
            loaded_boundary += full_dim - extra

    return loader


class MambaMixer2(torch.nn.Module):
    """
    Mamba2 混合器：计算 SSM 状态空间参数 ∆、A、B、C、D，并输出上下文化状态。
    - A、D 是输入无关的（固定转移矩阵/跳连系数）
    - ∆（时间步长）、B（输入投影矩阵）、C（输出投影矩阵）是输入相关的（selective）

    Compute ∆, A, B, C, and D the state space parameters and compute
    the `contextualized_states`. A, D are input independent
    (see Mamba paper [1] Section 3.5.2 "Interpretation of A"
    for why A isn't selective) ∆, B, C are input-dependent
    (this is a key difference between Mamba and the linear time
    invariant S4, and is why Mamba is called
    **selective** state spaces)
    """

    def __init__(
        self,
        cache_params: Mamba2CacheParams,  # 缓存形状参数（heads/dim/state_size/conv_kernel）
        hidden_size: int,                  # 模型隐藏层维度
        use_conv_bias: bool,               # 是否使用卷积偏置
        use_bias: bool,                    # 是否使用线性投影偏置
        n_groups: int = 1,                 # SSM 组数（B/C 各有 n_groups 个）
        rms_norm_eps: float = 1e-5,        # RMSNorm epsilon
        activation: str = "silu",          # 激活函数
        use_rms_norm: bool = True,         # 是否使用 RMSNorm
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        # For TP, the sharding plan is as follows:
        # - for the conv modules, since
        #   conv_dim = intermediate_size * 2 * n_groups * ssm_state_size,
        #   we shard intermediate_size and n_groups
        # - since intermediate_size = n_heads * head_dim, sharding on
        #   intermediate_size is achieved by sharding on n_heads.
        # - IF, world_size divides groups, then sharding
        #   (n_groups / world_size, n_heads / world_size)
        #   also maintains the invariant n_heads % n_groups == 0
        # - HOWEVER IF, world_size DOES NOT divide groups, then we need
        #   to allocate extra space in the shard, such that groups
        #   may be replicated to follow the head shard.
        # - NOTE: currently for the world size DOES NOT divide groups
        #   case, we only support the case when n_groups == 1
        # TP 分片方案：
        #   - 沿 n_heads 维分片 intermediate_size
        #   - 若 tp_size 整除 n_groups：正常分片 B/C group
        #   - 若不整除（n_groups==1 时）：复制 group 以配合 head shard
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        # SSM head 数量和每 head 维度
        self.num_heads = num_heads = cache_params.shape.num_heads
        self.head_dim = cache_params.shape.head_dim

        assert (
            num_heads % self.tp_size == 0
        ), "Tensor parallel world size must divide num heads."

        assert (n_groups % self.tp_size) == 0 or n_groups == 1, (
            "If tensor parallel world size does not divide num_groups, "
            "then num_groups must equal 1."
        )

        assert (
            (n_groups % self.tp_size == 0) or self.tp_size == 1 or quant_config is None
        ), (
            "Tensor parallel currently supported for quantized models only "
            "if tensor parallel world size divides num groups."
        )

        # SSM 状态维度 d_state（即 B/C 的最后一维大小）
        self.ssm_state_size = cache_params.shape.ssm_state_size
        self.activation = activation

        conv_kernel_size = cache_params.shape.conv_kernel
        self.intermediate_size = intermediate_size = (
            cache_params.shape.intermediate_size
        )
        self.n_groups = n_groups
        if n_groups % self.tp_size != 0:
            # - for TP we shard conv_dim by sharding on n_groups,
            # - but if n_groups cannot divide tp_size, we need to
            #   extend some extra groups
            # n_groups 不能被 tp_size 整除时，扩展 extra groups 以满足 head shard 对齐
            groups = extra_groups_for_head_shards(n_groups, self.tp_size)
            self.n_groups = n_groups + groups
        # B/C 合并后的 group*state 维度
        self.groups_ssm_state_size = self.n_groups * self.ssm_state_size
        self.conv_dim = cache_params.shape.conv_dim

        if n_groups % self.tp_size == 0:
            # 标准情况：n_groups 可被 tp_size 整除，使用 MergedColumnParallelLinear
            self.conv1d = MergedColumnParallelLinear(
                input_size=conv_kernel_size,
                output_sizes=[
                    intermediate_size,
                    self.groups_ssm_state_size,
                    self.groups_ssm_state_size,
                ],
                bias=use_conv_bias,
                quant_config=None,
                prefix=f"{prefix}.conv1d",
            )

            # in_proj 输出 [gate, x, B, C, dt]，分别对应 Mamba2 各分量
            self.in_proj = MergedColumnParallelLinear(
                input_size=hidden_size,
                output_sizes=[
                    intermediate_size,
                    intermediate_size,
                    self.groups_ssm_state_size,
                    self.groups_ssm_state_size,
                    self.num_heads,
                ],
                bias=use_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj",
            )
        else:
            # This is the n_groups == 1 case,
            # where we need to duplicate groups if TP>1.
            # n_groups==1 特殊情况：无法分片 group，需要复制

            self.conv1d = ColumnParallelLinear(
                input_size=conv_kernel_size,
                output_size=self.conv_dim,
                bias=use_conv_bias,
                quant_config=None,
                prefix=f"{prefix}.conv1d",
            )

            self.in_proj = ColumnParallelLinear(
                input_size=hidden_size,
                output_size=intermediate_size + self.conv_dim + self.num_heads,
                bias=use_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj",
            )

            # - because in_proj is a concatenation of 3 weights, we
            #   need to interleave them before sharding
            # - use the custom weight loader mamba_v2_sharded_weight_loader
            #   for conv1d.bias, covn1d.weight and in_proj.weight
            # - need to set these settings, to assign the groups
            #   to the head shards
            # 自定义分片加载器配置：各分量的 (全局维度, extra dims, duplicate_groups)
            group_shard_settings = (
                self.groups_ssm_state_size,  # expected model size
                (self.n_groups - n_groups) * self.ssm_state_size,  # extra dims assigned
                n_groups == 1,  # if there was only one group
            )
            intermediate_settings = (intermediate_size, 0, False)
            head_settings = (self.num_heads, 0, False)

            # - the weight already has a "weight_loader" attribute
            #   which set_weight_attrs will raise if we do not
            #   delete before trying to override it
            # - ditto for the other two weights below
            # 覆盖默认 weight_loader 前需先删除已有属性
            delattr(self.conv1d.bias, "weight_loader")
            set_weight_attrs(
                self.conv1d.bias,
                {
                    "weight_loader": mamba_v2_sharded_weight_loader(
                        [
                            intermediate_settings,
                            group_shard_settings,
                            group_shard_settings,
                        ],
                        self.tp_size,
                        self.tp_rank,
                    )
                },
            )

            delattr(self.conv1d.weight, "weight_loader")
            set_weight_attrs(
                self.conv1d.weight,
                {
                    "weight_loader": mamba_v2_sharded_weight_loader(
                        [
                            intermediate_settings,
                            group_shard_settings,
                            group_shard_settings,
                        ],
                        self.tp_size,
                        self.tp_rank,
                    )
                },
            )

            if quant_config is None:
                # - quant layers do not have a weight loader
                # 非量化模式下为 in_proj.weight 设置自定义加载器
                delattr(self.in_proj.weight, "weight_loader")
                set_weight_attrs(
                    self.in_proj.weight,
                    {
                        "weight_loader": mamba_v2_sharded_weight_loader(
                            [
                                intermediate_settings,  # for gate
                                intermediate_settings,
                                group_shard_settings,
                                group_shard_settings,
                                head_settings,  # for dt
                            ],
                            self.tp_size,
                            self.tp_rank,
                        )
                    },
                )

        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `MergedColumnParallelLinear`,
        # and `set_weight_attrs` doesn't allow to override it
        # conv1d 的 weight 需要额外的 channel 维度以符合 causal_conv1d 接口
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # - these are TPed by heads to reduce the size of the
        #   temporal shape
        # A 矩阵（SSM 状态转移矩阵对角元素的对数），按 head 维度 TP 分片
        # 加载时通过 composed_weight_loader 转换：A_log → -exp(A_log)
        self.A = nn.Parameter(
            torch.empty(
                divide(num_heads, self.tp_size),
                dtype=torch.float32,
            )
        )
        # D 矩阵（跳连系数，每个 head 一个标量），按 head 维度 TP 分片
        self.D = nn.Parameter(torch.ones(num_heads // self.tp_size))
        # dt_bias（时间步长偏置），按 head 维度 TP 分片
        self.dt_bias = nn.Parameter(torch.ones(num_heads // self.tp_size))
        self.use_rms_norm = use_rms_norm

        set_weight_attrs(self.D, {"weight_loader": sharded_weight_loader(0)})
        # A 的加载器：先分片，再做 -exp() 转换（从 log 域转回负实数域）
        a_weight_loader = composed_weight_loader(
            sharded_weight_loader(0), lambda x: -torch.exp(x.float())
        )
        set_weight_attrs(self.A, {"weight_loader": a_weight_loader})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        # 输出投影：将 SSM 输出从 intermediate_size 投影回 hidden_size
        self.out_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=use_bias,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        # 门控 RMSNorm 层：对 SSM 输出应用 gate 乘积并归一化
        self.norm = Mixer2RMSNormGated(
            intermediate_size, n_groups, self.use_rms_norm, eps=rms_norm_eps
        )

        self.prefix = prefix

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,        # 输入 hidden states，形状 (num_tokens, hidden_size)
        output: torch.Tensor,               # 预分配的输出 tensor，原地写入
        layer_cache: MambaPool.State,       # 当前层的 conv state 和 SSM state 缓存
        metadata: Mamba2Metadata,           # 本次 forward 的批次元数据
        mup_vector: Optional[torch.Tensor] = None,  # muP 缩放向量（可选）
        use_triton_causal_conv: bool = False,        # 是否使用 Triton causal conv（推测解码时必须为 True）
    ):
        # metadata contains metadata necessary for the mamba2 triton
        # kernels to operate in continuous batching and in chunked prefill
        # modes; they are computed at top-level model forward since they
        # stay the same and reused for all mamba layers in the same iteration
        # 元数据在顶层 model forward 中预计算并在所有 mamba 层间复用
        state_indices_tensor = metadata.mamba_cache_indices
        # 取出卷积状态缓存 (batch_cache_size, d_model, conv_kernel-1)
        conv_state = layer_cache.conv[0]
        # 取出 SSM 隐藏状态缓存 (batch_cache_size, nheads, headdim, dstate)
        ssm_state = layer_cache.temporal

        query_start_loc = metadata.query_start_loc

        # 1. Gated MLP's linear projection
        # 步骤1：线性投影，将 hidden_states 映射到 [gate, x_B_C, dt] 拼接空间
        projected_states, _ = self.in_proj(hidden_states)

        # 若使用 muP（Maximal Update Parameterization），对投影结果缩放
        if mup_vector is not None:
            projected_states = projected_states * mup_vector

        # 将 projected_states 拆分为：gate（门控）、hidden_states_B_C（x+B+C）、dt（时间步长）
        gate, hidden_states_B_C, dt = torch.split(
            projected_states,
            [
                self.intermediate_size // self.tp_size,
                self.conv_dim // self.tp_size,
                self.num_heads // self.tp_size,
            ],
            dim=-1,
        )
        # 将 conv1d 的权重形状从 (out, 1, kernel) 压缩为 (out, kernel) 以传给 causal_conv1d
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        # - get hidden_states, B and C after depthwise convolution.
        # 定义将卷积后的 hidden_states_B_C 拆分为 x、B（输入矩阵）、C（输出矩阵）的 lambda
        split_hidden_states_B_C_fn = lambda hidden_states_B_C: torch.split(
            hidden_states_B_C,
            [
                self.intermediate_size // self.tp_size,
                self.groups_ssm_state_size // self.tp_size,
                self.groups_ssm_state_size // self.tp_size,
            ],
            dim=-1,
        )

        # 统计 prefill/decode 请求和 token 数量
        num_prefills = metadata.num_prefills  # request count
        num_decodes = metadata.num_decodes  # token count (=request)
        num_decode_tokens = (
            num_decodes * metadata.draft_token_num
            if metadata.is_target_verify
            else num_decodes
        )
        num_prefill_tokens = metadata.num_prefill_tokens  # token count
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        num_actual_tokens = num_prefill_tokens + num_decode_tokens
        assert num_actual_tokens == projected_states.shape[0]

        # NOTE: V0 put prefill before decode
        # Separate prefill and decode by splitting varlen input
        # Split along token dimension
        # 沿 token 维度将 prefill 和 decode tokens 拆分（V0 约定：prefill tokens 在前）
        hidden_states_B_C_p, hidden_states_B_C_d = torch.split(
            hidden_states_B_C,
            [num_prefill_tokens, num_decode_tokens],
            dim=0,
        )
        dt_p, dt_d = torch.split(
            dt,
            [num_prefill_tokens, num_decode_tokens],
            dim=0,
        )
        # Split along batch dimension
        # 沿 batch 维度拆分 prefill/decode 的缓存索引
        state_indices_tensor_p, state_indices_tensor_d = torch.split(
            state_indices_tensor,
            [num_prefills, num_decodes],
            dim=0,
        )
        # prefill 的累积序列位置（decode 无需此信息）
        query_start_loc_p = query_start_loc[: num_prefills + 1] if has_prefill else None

        # Preallocate output tensor to avoid memcpy cost for merging prefill
        # and decode outputs
        # 预分配 SSM 输出 tensor，避免 prefill/decode 输出拼接时的内存拷贝
        preallocated_ssm_out = torch.empty(
            [
                projected_states.shape[0],
                (self.num_heads * self.head_dim) // self.tp_size,
            ],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        # 将预分配 tensor 同样按 prefill/decode token 数量拆分视图
        preallocated_ssm_out_p, preallocated_ssm_out_d = torch.split(
            preallocated_ssm_out,
            [num_prefill_tokens, num_decode_tokens],
            dim=0,
        )

        # Process prefill requests
        # === Prefill 路径 ===
        if has_prefill:
            mixed_metadata = metadata.mixed_metadata
            assert mixed_metadata is not None
            # 2. Convolution sequence transformation
            # - "cache_indices" updates the conv_state cache in positions
            #   pointed to by "state_indices_tensor"
            # 步骤2（prefill）：因果卷积序列变换，同时更新 conv_state 缓存
            has_initial_states_p = mixed_metadata.has_initial_states
            prep_initial_states = mixed_metadata.prep_initial_states
            cache_indices = state_indices_tensor_p
            # causal_conv1d 期望输入形状为 (dim, seqlen)，需转置
            x = hidden_states_B_C_p.transpose(
                0, 1
            )  # this is the form that causal-conv see
            # 选择 sgl_kernel 或 Triton 实现
            ccfn = (
                causal_conv1d_fn
                if not use_triton_causal_conv
                else causal_conv1d_fn_triton
            )
            hidden_states_B_C_p = ccfn(
                x,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_states_p,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc_p,
                seq_lens_cpu=mixed_metadata.extend_seq_lens_cpu,
            ).transpose(0, 1)[:num_prefill_tokens]  # 转置回 (token, dim) 并截取实际 token 数

            # 将卷积后的 hidden_states_B_C 拆分为 x（特征）、B（SSM 输入矩阵）、C（SSM 输出矩阵）
            hidden_states_p, B_p, C_p = split_hidden_states_B_C_fn(hidden_states_B_C_p)

            # 3. State Space Model sequence transformation
            # 步骤3（prefill）：SSM 序列变换
            # 若存在 prefix cache 命中的初始状态，从 ssm_state 缓存中加载
            initial_states = None
            if has_initial_states_p is not None and prep_initial_states:
                initial_states = torch.where(
                    has_initial_states_p[:, None, None, None],
                    ssm_state[state_indices_tensor_p],
                    0,
                )

            # NOTE: final output is an in-place update of out tensor
            # mamba_chunk_scan_combined 执行 chunked SSM 扫描，返回每个 prefill 序列的最终状态
            varlen_state = mamba_chunk_scan_combined(
                hidden_states_p.view(
                    1, num_prefill_tokens, self.num_heads // self.tp_size, self.head_dim
                ),
                dt_p.unsqueeze(0),
                self.A,          # SSM 状态转移矩阵（对角，负实数）
                B_p.view(1, num_prefill_tokens, self.n_groups // self.tp_size, -1),  # 输入矩阵 B
                C_p.view(1, num_prefill_tokens, self.n_groups // self.tp_size, -1),  # 输出矩阵 C
                chunk_size=mixed_metadata.chunk_size,
                D=self.D,        # 跳连系数 D（残差项）
                z=None,
                dt_bias=self.dt_bias,  # 时间步长偏置
                seq_idx=mixed_metadata.seq_idx,
                chunk_indices=mixed_metadata.chunk_indices,
                chunk_offsets=mixed_metadata.chunk_offsets,
                cu_seqlens=query_start_loc_p,
                initial_states=initial_states,
                return_varlen_states=True,
                return_final_states=False,
                dt_softplus=True,
                dt_limit=(0.0, float("inf")),
                out=preallocated_ssm_out_p.view(
                    1, num_prefill_tokens, -1, self.head_dim
                ),
                state_dtype=ssm_state.dtype,
            )

            # update ssm states
            # - varlen state is a (num_prefills, nheads, headdim, dstate) tensor
            # 将 prefill 扫描后的最终 SSM 状态写回缓存（供后续 decode 步骤使用）
            ssm_state[state_indices_tensor_p] = varlen_state

        # Process decode requests
        # === Decode 路径 ===
        if has_decode:
            is_target_verify = metadata.is_target_verify

            # 2. Convolution sequence transformation
            # 步骤2（decode）：增量式因果卷积更新
            if is_target_verify:
                # 推测解码验证模式：需要保存中间卷积状态以支持 tree-structure 回溯
                assert (
                    use_triton_causal_conv
                ), "Speculative decoding requires use_triton_causal_conv=True for intermediate state support"
                assert isinstance(
                    layer_cache, MambaPool.SpeculativeState
                ), "layer_cache must be SpeculativeState for speculative decoding"
                draft_token_num = metadata.draft_token_num
                self.intermediate_state_indices = torch.arange(
                    num_decodes, dtype=torch.int32, device=state_indices_tensor_d.device
                )

                # Reshape for batch processing
                # 将 (num_decodes*draft_token_num, dim) 重塑为 (num_decodes, dim, draft_token_num)
                hidden_states_B_C_d_reshaped = hidden_states_B_C_d.view(
                    num_decodes, draft_token_num, -1
                ).transpose(1, 2)

                # 调用 Triton causal_conv1d_update，保存中间卷积窗口状态
                hidden_states_B_C_d_processed = causal_conv1d_update_triton(
                    hidden_states_B_C_d_reshaped,
                    conv_state,
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                    conv_state_indices=state_indices_tensor_d[:num_decodes],
                    intermediate_conv_window=layer_cache.intermediate_conv_window[0],
                    intermediate_state_indices=self.intermediate_state_indices,
                    retrieve_next_token=metadata.retrieve_next_token,
                    retrieve_next_sibling=metadata.retrieve_next_sibling,
                    retrieve_parent_token=metadata.retrieve_parent_token,
                )
                # 还原为 (num_decode_tokens, dim)
                hidden_states_B_C_d = hidden_states_B_C_d_processed.transpose(
                    1, 2
                ).view(num_decode_tokens, -1)
            else:
                # 普通 decode：每请求一次更新卷积状态
                ccu = (
                    causal_conv1d_update
                    if not use_triton_causal_conv
                    else causal_conv1d_update_triton
                )
                hidden_states_B_C_d = ccu(
                    hidden_states_B_C_d,
                    conv_state,
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                    conv_state_indices=state_indices_tensor_d,
                )

            # 将 decode 路径的卷积输出拆分为 x、B、C
            hidden_states_d, B_d, C_d = split_hidden_states_B_C_fn(hidden_states_B_C_d)

            # 3. State Space Model sequence transformation
            # 步骤3（decode）：single-step SSM 状态更新
            n_groups = self.n_groups // self.tp_size
            # 将 A 扩展为 (nheads, head_dim, dstate) 以匹配 selective_state_update 接口
            A_d = (
                self.A[:, None, ...][:, :, None]
                .expand(-1, self.head_dim, self.ssm_state_size)
                .to(dtype=torch.float32)
            )
            # 扩展 dt、dt_bias、D 到 head_dim 维度
            dt_d = dt_d[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
            D_d = self.D[:, None, ...].expand(-1, self.head_dim)
            # 将 B/C 重组为 (batch, n_groups, dstate) 形状
            B_d = B_d.view(-1, n_groups, B_d.shape[1] // n_groups)
            C_d = C_d.view(-1, n_groups, C_d.shape[1] // n_groups)
            hidden_states_d = hidden_states_d.view(
                -1, self.num_heads // self.tp_size, self.head_dim
            )

            if is_target_verify:
                # 推测解码验证：多步 selective_state_update，保存中间 SSM 状态
                selective_state_update(
                    ssm_state,
                    hidden_states_d.view(
                        num_decodes,
                        draft_token_num,
                        self.num_heads // self.tp_size,
                        self.head_dim,
                    ),
                    dt_d.view(
                        num_decodes,
                        draft_token_num,
                        self.num_heads // self.tp_size,
                        self.head_dim,
                    ),
                    A_d,
                    B_d.view(num_decodes, draft_token_num, n_groups, -1),
                    C_d.view(num_decodes, draft_token_num, n_groups, -1),
                    D_d,
                    z=None,
                    dt_bias=dt_bias,
                    dt_softplus=True,
                    state_batch_indices=state_indices_tensor_d[:num_decodes],
                    out=preallocated_ssm_out_d.view(
                        num_decodes,
                        draft_token_num,
                        self.num_heads // self.tp_size,
                        self.head_dim,
                    ),
                    disable_state_update=True,
                    intermediate_states_buffer=layer_cache.intermediate_ssm,
                    cache_steps=draft_token_num,
                    retrieve_parent_token=metadata.retrieve_parent_token,
                    intermediate_state_indices=self.intermediate_state_indices,
                )
            else:
                # 普通 decode：单步 selective_state_update，直接更新 ssm_state 缓存
                selective_state_update(
                    ssm_state,
                    hidden_states_d,
                    dt_d,
                    A_d,
                    B_d,
                    C_d,
                    D_d,
                    z=None,
                    dt_bias=dt_bias,
                    dt_softplus=True,
                    state_batch_indices=state_indices_tensor_d,
                    out=preallocated_ssm_out_d.view(num_decodes, -1, self.head_dim),
                )

        # 4. gated MLP
        # GatedRMSNorm internally applying SiLU to the gate
        # SiLU is applied internally before normalization, unlike standard
        # norm usage
        # 步骤4：门控 RMSNorm（内部先对 gate 做 SiLU，再归一化 SSM 输出）
        hidden_states = self.norm(preallocated_ssm_out, gate[:num_actual_tokens])

        # 5. Final linear projection
        # 步骤5：输出线性投影，将 SSM 结果映射回 hidden_size
        output[:num_actual_tokens], _ = self.out_proj(hidden_states)

    @property
    def mamba_type(self) -> str:
        # 返回当前模块的 Mamba 类型标识
        return "mamba2"
