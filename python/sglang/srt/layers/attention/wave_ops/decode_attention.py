"""
Memory-efficient attention for decoding.
It supports page size = 1.
"""
# Wave 框架（wave_lang）实现的 decode 阶段注意力计算
# 使用分页 decode 注意力（paged decode attention），两阶段计算：
#   Phase 0: 计算 QK^T 注意力分数和 logit max（softmax 归一化准备）
#   Phase 1: 计算加权 V（softmax 归一化 + reduce）

# 导入 functools.lru_cache（缓存编译后的 Wave Kernel）
import functools
# 导入日志模块
import logging

# 导入 Wave 语言全局符号（运行时常量和宏）
from wave_lang.kernel.lang.global_symbols import *
# 导入 Wave 编译选项和 wave_compile 函数
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
# 导入矩阵乘法操作类型（GenericDot 为通用点积，MMAType 为 AMD MFMA 类型）
from wave_lang.kernel.wave.constraints import GenericDot, MMAOperand, MMAType
# 导入分页 decode 注意力 Kernel 生成函数和辅助工具
from wave_lang.kernel.wave.templates.paged_decode_attention import (
    get_paged_decode_attention_kernels,         # 获取两阶段 Kernel
    get_paged_decode_intermediate_arrays_shapes, # 获取中间结果数组形状（logits/max）
    paged_decode_attention_shape,               # decode 注意力形状描述符
)
# 导入默认调度超参数
from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params
# 导入默认运行配置
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config

logger = logging.getLogger(__name__)
# 导入 os，用于读取 WAVE_DUMP_MLIR 调试环境变量
import os

# 环境变量控制：是否将生成的 MLIR 代码转储到文件（调试用）
dump_generated_mlir = int(os.environ.get("WAVE_DUMP_MLIR", 0))


@functools.lru_cache(maxsize=4096)
def get_wave_kernel(
    shape: paged_decode_attention_shape,  # decode 注意力形状（含 block_size）
    max_kv_splits,    # KV 分割数（控制 KV cache 并行读取的分块数量）
    input_dtype,      # 输入数据类型（fp16/bf16）
    output_dtype,     # 输出数据类型
    logit_cap,        # Logit 截断值（避免 softmax 溢出）
):
    # 检查是否为 MHA（Multi-Head Attention，num_query_heads == num_kv_heads）
    mha = (shape.num_query_heads // shape.num_kv_heads) == 1

    # 根据 MHA/GQA 选择不同的 MFMA 变体
    if mha:
        # MHA 使用 GenericDot（通用向量点积）：沿 M 维度，向量宽度 4 或 64
        # 适合 decode 阶段的单头注意力（非矩阵形状）
        mfma_variant = (
            GenericDot(along_dim=MMAOperand.M, k_vec_size=4, k_mult=1),    # Phase 0 QK dot
            GenericDot(along_dim=MMAOperand.M, k_vec_size=1, k_mult=64),   # Phase 1 SV dot
        )
    else:
        # GQA 使用标准 MFMA 矩阵乘法（适合多 query head 对应少 kv head 的情况）
        mfma_variant = (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16)

    # 获取两阶段 decode 注意力 Kernel 和对应的编译参数
    (
        phase_0,           # Phase 0 Kernel（计算 QK^T 和 logit max）
        phase_1,           # Phase 1 Kernel（计算加权 V 并 reduce）
        hyperparams_0,     # Phase 0 超参数（Tile 大小等）
        hyperparams_1,     # Phase 1 超参数
        dynamic_symbols_0, # Phase 0 动态符号（运行时决定的 KV 序列长度等）
        dynamic_symbols_1, # Phase 1 动态符号
    ) = get_paged_decode_attention_kernels(
        shape,
        mfma_variant,
        max_kv_splits,     # KV 分割数（影响 Phase 0 的并行度）
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        logit_cap=logit_cap,
    )
    # 合并默认调度超参数
    hyperparams_0.update(get_default_scheduling_params())
    hyperparams_1.update(get_default_scheduling_params())

    # 编译 Phase 0 Kernel（QK 注意力分数计算）
    options = WaveCompileOptions(
        subs=hyperparams_0,
        canonicalize=True,
        run_bench=False,
        use_buffer_ops=True,            # Phase 0 使用 buffer 操作（优化 KV cache 访问）
        waves_per_eu=2,                 # 每个 CU 的 Wave 数
        dynamic_symbols=dynamic_symbols_0,
        wave_runtime=True,
    )
    options = set_default_run_config(options)
    phase_0 = wave_compile(options, phase_0)

    # 编译 Phase 1 Kernel（SoftMax + 加权 V reduce）
    options = WaveCompileOptions(
        subs=hyperparams_1,
        canonicalize=True,
        run_bench=False,
        use_buffer_ops=False,           # Phase 1 不使用 buffer 操作
        waves_per_eu=4,                 # Phase 1 使用更多 Wave（reduce 阶段）
        dynamic_symbols=dynamic_symbols_1,
        wave_runtime=True,
    )
    options = set_default_run_config(options)
    phase_1 = wave_compile(options, phase_1)

    return phase_0, phase_1


def decode_attention_intermediate_arrays_shapes(
    num_seqs, head_size_kv, num_query_heads, max_kv_splits
):
    # 计算中间结果数组（attn_logits 和 attn_logits_max）的形状
    # 这些数组用于存储 Phase 0 的分片 attention 分数和 softmax max 值
    # Not all fields are used, but we need to pass them to the function
    # 构建形状描述符（仅需 num_seqs, num_query_heads, head_size_kv 字段）
    shape = paged_decode_attention_shape(
        num_query_heads=num_query_heads,
        num_kv_heads=0,          # 不使用（传 0 占位）
        head_size=0,             # 不使用（传 0 占位）
        head_size_kv=head_size_kv,
        block_size=0,            # 不使用（传 0 占位）
        num_seqs=num_seqs,
    )
    # 返回 (attn_logits_shape, attn_logits_max_shape)
    return get_paged_decode_intermediate_arrays_shapes(shape, max_kv_splits)


def decode_attention_wave(
    q,               # Query 张量 [num_seqs, num_query_heads, head_size]（每请求 1 个 token）
    k_buffer,        # KV 缓存中的 Key [num_cached_tokens, num_kv_heads, head_size]
    v_buffer,        # KV 缓存中的 Value [num_cached_tokens, num_kv_heads, head_size_kv]
    o,               # 输出张量 [num_seqs, num_query_heads, head_size_kv]
    b_req_idx,       # 每个请求对应的 KV cache 池索引 [num_seqs]
    req_to_token,    # 请求索引到 token 位置的映射表（paged KV cache 地址转换）
    attn_logits,     # 中间结果：分片注意力分数 [num_seqs, num_query_heads, max_kv_splits, head_size_kv]
    attn_logits_max, # 中间结果：每分片的 softmax max 值 [num_seqs, num_query_heads, max_kv_splits]
    num_kv_splits,   # 实际使用的 KV 分割数（动态决定，<= max_kv_splits）
    max_kv_splits,   # 最大 KV 分割数（编译时固定）
    sm_scale,        # Softmax 缩放因子（通常 = 1 / sqrt(head_size)）
    logit_cap,       # Logit 截断值（0 表示不截断）
):
    # Wave 框架 decode 注意力的两阶段前向计算
    num_seqs, num_query_heads, head_size = q.shape
    _, num_kv_heads, _ = k_buffer.shape
    _, _, head_size_kv = v_buffer.shape
    block_size = 32  # KV cache 分页大小（page size = 32 tokens per block）
    # 构建 decode 注意力形状描述符
    shape = paged_decode_attention_shape(
        num_query_heads,
        num_kv_heads,
        head_size,
        head_size_kv,
        block_size,
        num_seqs,
    )

    # 获取（或从 cache 加载）编译好的两阶段 Kernel
    phase_0, phase_1 = get_wave_kernel(
        shape, max_kv_splits, q.dtype, o.dtype, logit_cap
    )

    # Phase 0：计算 Q @ K^T 注意力分数，并找到每分片的 logit max（用于数值稳定的 softmax）
    mb_qk = phase_0(
        q,               # Query
        k_buffer,        # KV 缓存 Key
        v_buffer,        # KV 缓存 Value（Phase 0 可能预先 load 供 Phase 1 使用）
        b_req_idx,       # 请求索引
        req_to_token,    # 请求到 token 的映射
        attn_logits,     # 输出：分片 attention logits
        attn_logits_max, # 输出：分片 softmax max 值
    )
    # 调试模式：转储 Phase 0 的 MLIR
    if dump_generated_mlir:
        filename = f"wave_decode_attention_phase0_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(mb_qk.module_op.get_asm())

    # Phase 1：对各分片的 attention logits 做 online softmax + 加权 V reduce（最终输出）
    mb_sv = phase_1(attn_logits, attn_logits_max, b_req_idx, o)
    # 调试模式：转储 Phase 1 的 MLIR
    if dump_generated_mlir:
        filename = f"wave_decode_attention_phase1_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(mb_sv.module_op.get_asm())


def decode_attention_fwd(
    q,               # Query 张量
    k_buffer,        # KV 缓存 Key
    v_buffer,        # KV 缓存 Value
    o,               # 输出张量
    b_req_idx,       # 请求索引
    req_to_token,    # 请求到 token 的映射
    attn_logits,     # 中间结果：分片注意力分数
    attn_logits_max, # 中间结果：分片 softmax max
    num_kv_splits,   # 实际 KV 分割数
    max_kv_splits,   # 最大 KV 分割数
    sm_scale,        # Softmax 缩放因子
    logit_cap=0.0,   # Logit 截断值（默认 0 = 不截断）
):
    # decode 注意力前向计算的公共接口（直接转发给 decode_attention_wave）
    decode_attention_wave(
        q,
        k_buffer,
        v_buffer,
        o,
        b_req_idx,
        req_to_token,
        attn_logits,
        attn_logits_max,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
        logit_cap,
    )
