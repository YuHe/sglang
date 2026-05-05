"""
Memory-efficient attention for prefill.
It support page size = 1.
"""
# Wave 框架实现的 extend（有 prefix KV cache）注意力计算
# 支持带历史 KV 缓存（radix cache/paged cache）的 prefill/extend 阶段

# 导入 functools.lru_cache（用于缓存编译后的 Wave Kernel，避免重复编译）
import functools
# 导入 os，用于读取 WAVE_DUMP_MLIR 调试环境变量
import os

# 导入 PyTorch（用于 dtype 类型注解）
import torch
# 导入 Wave 语言全局符号
from wave_lang.kernel.lang.global_symbols import *
# 导入 Wave 编译选项和 wave_compile 函数
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
# 导入 MFMA 矩阵乘法变体（AMD GPU 矩阵乘法指令类型）
from wave_lang.kernel.wave.constraints import MMAType
# 导入调度类型（NONE 表示不使用 Wave 自动调度）
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
# 导入注意力形状描述符
from wave_lang.kernel.wave.templates.attention_common import AttentionShape
# 导入 extend 注意力 Kernel 生成函数（支持访问 prefix KV cache）
from wave_lang.kernel.wave.templates.extend_attention import get_extend_attention_kernel
# 导入默认调度超参数
from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params
# 导入默认运行配置
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config

# 环境变量控制：是否转储生成的 MLIR 代码（调试用）
dump_generated_mlir = int(os.environ.get("WAVE_DUMP_MLIR", 0))


@functools.lru_cache
def get_wave_kernel(
    shape: AttentionShape,         # 注意力形状描述符
    q_shape: tuple[int],           # Query 张量形状（含总 token 数、头数、头维度）
    k_shape: tuple[int],           # Key 张量形状（extend 部分）
    v_shape: tuple[int],           # Value 张量形状（extend 部分）
    k_cache_shape: tuple[int],     # KV 缓存中 Key 的形状（prefix 历史 KV）
    v_cache_shape: tuple[int],     # KV 缓存中 Value 的形状
    o_shape: tuple[int],           # 输出张量形状
    input_dtype: torch.dtype,      # 输入数据类型（fp16/bf16）
    output_dtype: torch.dtype,     # 输出数据类型
    size_dtype: torch.dtype,       # 索引/长度张量类型（int32）
    is_causal: bool,               # 是否使用因果掩码（下三角注意力）
    logit_cap: float,              # Logit 截断值（避免数值溢出）
    layer_scaling: float,          # 层缩放因子（用于注意力权重缩放）
):
    # lru_cache 确保相同形状配置只编译一次 Kernel（Kernel 编译耗时较长）
    assert shape.num_query_heads % shape.num_kv_heads == 0  # GQA 要求整除

    # 选择 MFMA 变体：QK 乘法用 F32_16x16x32_K8_F16，SV 乘法用 F32_16x16x16_F16
    # K8 表示 K 维度每步处理 8 个元素（更高的 MFMA 效率）
    mfma_variant = (MMAType.F32_16x16x32_K8_F16, MMAType.F32_16x16x16_F16)
    (
        extend_attention,    # Wave 注意力函数（等待编译）
        hyperparams,         # 编译时超参数（Tile 大小等）
        dynamic_symbols,     # 动态符号列表（seq_len 等运行时决定的值）
    ) = get_extend_attention_kernel(
        shape,
        mfma_variant,
        q_shape,
        k_shape,
        v_shape,
        k_cache_shape,
        v_cache_shape,
        o_shape,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        size_dtype=size_dtype,
        is_causal=is_causal,
        layer_scaling=layer_scaling,
        logit_cap=logit_cap,
    )

    # 合并默认调度超参数（Tile 大小等）
    hyperparams.update(get_default_scheduling_params())
    # 配置编译选项（extend 注意力不使用 Wave 自动调度，禁用调度 barrier）
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=False,
        schedule=SchedulingType.NONE,   # 不使用自动调度
        use_scheduling_barriers=False,  # 不使用调度 barrier
        dynamic_symbols=dynamic_symbols, # 动态符号列表（运行时绑定）
        use_buffer_ops=True,            # 使用 buffer 操作（提高内存访问效率）
        waves_per_eu=2,                 # 每个 CU 的 Wave 数（控制并行度）
        denorm_fp_math_f32="preserve-sign",  # 保留 FP32 次正规数的符号位
        wave_runtime=True,              # 使用 Wave 运行时（优化执行流程）
    )
    options = set_default_run_config(options)  # 设置 AMD GPU 运行时配置
    extend_attention = wave_compile(options, extend_attention)  # 编译 Kernel

    return extend_attention


def extend_attention_wave(
    q_extend,       # 新增（extend）部分的 Query 张量 [total_new_tokens, num_heads, head_dim]
    k_extend,       # 新增部分的 Key 张量
    v_extend,       # 新增部分的 Value 张量
    k_buffer,       # KV 缓存中的 Key（历史 prefix tokens）[num_cached_tokens, num_kv_heads, head_dim]
    v_buffer,       # KV 缓存中的 Value
    qo_indptr,      # Query 和输出的 CSR 格式起始位置 [num_seqs + 1]
    kv_indptr,      # KV（prefix+extend）的 CSR 格式起始位置 [num_seqs + 1]
    kv_indices,     # KV 索引：每个 KV 位置对应 KV 缓存中的 token 索引
    custom_mask,    # 自定义注意力 mask（可选，通常为 None）
    mask_indptr,    # 自定义 mask 的 CSR 起始位置（可选）
    max_seq_len,    # 批次中最长的完整序列长度（prefix + extend）
    output,         # 输出张量 [total_new_tokens, num_heads, head_dim_v]（in-place 写入）
    is_causal=True,     # 是否使用因果掩码
    layer_scaling=None, # 层缩放因子（不同层可以有不同缩放）
    logit_cap=0,        # Logit 截断值（0 表示不截断）
):
    # Wave 框架 extend 注意力的 Python 包装函数
    # 构建注意力形状描述符
    shape = AttentionShape(
        num_query_heads=q_extend.shape[1],  # Query 头数
        num_kv_heads=k_extend.shape[1],     # KV 头数（GQA 时 < num_query_heads）
        head_size=q_extend.shape[2],        # Query/Key 头维度
        head_size_kv=k_extend.shape[2],     # Value 头维度
        num_seqs=kv_indptr.shape[0] - 1,   # 批次中的序列数（从 indptr 推断）
        max_seq_len=max_seq_len,            # 最大完整序列长度
    )

    # 获取（或从 cache 中加载）编译好的 Wave extend 注意力 Kernel
    extend_attention = get_wave_kernel(
        shape,
        q_extend.shape,
        k_extend.shape,
        v_extend.shape,
        k_buffer.shape,
        v_buffer.shape,
        output.shape,
        input_dtype=q_extend.dtype,
        output_dtype=output.dtype,
        size_dtype=qo_indptr.dtype,
        is_causal=is_causal,
        layer_scaling=layer_scaling,
        logit_cap=logit_cap,
    )

    # 调用编译后的 extend 注意力 Kernel
    mb = extend_attention(
        q_extend,     # 新增部分的 Q
        k_extend,     # 新增部分的 K
        v_extend,     # 新增部分的 V
        k_buffer,     # 历史 KV 缓存 K
        v_buffer,     # 历史 KV 缓存 V
        qo_indptr,    # Query/Output 的 CSR 偏移
        kv_indptr,    # KV 的 CSR 偏移
        kv_indices,   # KV 缓存索引（用于 paged 访问）
        max_seq_len,  # 动态符号：最大序列长度（运行时绑定）
        output,       # 输出张量
    )

    # 调试模式：将生成的 MLIR 代码保存到文件
    if dump_generated_mlir:
        shape_list = [
            q_extend.shape[0],
            q_extend.shape[1],
            k_extend.shape[1],
            q_extend.shape[2],
            k_extend.shape[2],
        ]
        filename = f"wave_prefill_attention_{'x'.join(map(str, shape_list))}.mlir"
        with open(filename, "w") as f:
            f.write(mb.module_op.get_asm())  # 写入 MLIR 文本格式
