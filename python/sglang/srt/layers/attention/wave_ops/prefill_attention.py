"""
Memory-efficient attention for prefill.
It support page size = 1.
"""
# Wave 框架（wave_lang）实现的 prefill 注意力计算
# 使用 MLIR/IREE 后端在 AMD GPU 上生成高效的注意力 Kernel

# 导入数学工具（计算 log2e 和 dk_sqrt）
import math
# 导入 os，用于读取环境变量（WAVE_DUMP_MLIR 调试开关）
import os

# 导入 Wave 语言全局符号（MLIR/Wave 运行时需要的全局常量和宏）
from wave_lang.kernel.lang.global_symbols import *
# 导入 Wave 编译选项和 wave_compile 函数（将 Python Wave 函数编译为 MLIR Kernel）
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
# 导入 MFMA（矩阵乘法累加）变体类型（AMD GPU 矩阵计算单元配置）
from wave_lang.kernel.wave.constraints import MMAType
# 导入注意力形状描述符（封装 batch/head/seq 等维度）
from wave_lang.kernel.wave.templates.attention_common import AttentionShape
# 导入 prefill 注意力 Kernel 生成函数（根据形状生成对应的 Wave Kernel）
from wave_lang.kernel.wave.templates.prefill_attention import (
    get_prefill_attention_kernel,
)
# 导入默认调度参数（Wave Kernel 的调度超参数）
from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params
# 导入默认运行配置（设置 AMD GPU 相关的运行时配置）
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config

# 环境变量控制：是否将编译后的 MLIR 代码转储到文件（调试用）
dump_generated_mlir = int(os.environ.get("WAVE_DUMP_MLIR", 0))


def prefill_attention_wave(
    q, k, v, o, b_start_loc, b_seq_len, max_seq_len, is_causal=True
):
    # Wave 框架实现的 prefill 注意力前向计算
    # 适用于 AMD GPU（Instinct 系列），使用 MFMA 矩阵乘法加速

    # 构建注意力形状描述符（包含所有维度信息）
    shape = AttentionShape(
        num_query_heads=q.shape[1],      # Query 头数（包含 GQA 的多头）
        num_kv_heads=k.shape[1],         # KV 头数（GQA 时 < num_query_heads）
        head_size=q.shape[2],            # Query/Key 头维度
        head_size_kv=k.shape[2],         # Value 头维度（通常与 head_size 相等）
        num_seqs=b_seq_len.shape[0],     # 批次中的序列数
        max_seq_len=max_seq_len,         # 最大序列长度（用于内存分配和 Kernel 配置）
        total_seq_len=q.shape[0],        # 总 token 数（所有序列长度之和）
    )

    assert shape.num_query_heads % shape.num_kv_heads == 0  # GQA 要求整除

    output_shape = (shape.total_seq_len, shape.num_query_heads, shape.head_size_kv)
    # 选择 MFMA 变体：F32_16x16x16_F16 = 16×16 矩阵乘法，FP16 输入，FP32 累加
    mfma_variant = (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16)
    # 根据形状生成对应的 Wave prefill Kernel 和超参数
    prefill, hyperparams = get_prefill_attention_kernel(
        shape,
        mfma_variant,
        q.shape,
        k.shape,
        v.shape,
        output_shape,
        input_dtype=q.dtype,        # 输入数据类型（fp16/bf16）
        output_dtype=o.dtype,       # 输出数据类型
        size_dtype=b_seq_len.dtype, # 序列长度张量类型（int32）
    )

    # 合并默认调度超参数（Wave Kernel 的 tile 大小等）
    hyperparams.update(get_default_scheduling_params())

    # log2(e)：用于将 exp 转换为 exp2（AMD GPU 原生支持 exp2 指令）
    log2e = 1.44269504089
    # 注意力缩放因子：1 / sqrt(head_size)
    dk_sqrt = math.sqrt(1.0 / shape.head_size)

    # 配置 Wave 编译选项
    options = WaveCompileOptions(
        subs=hyperparams,               # 超参数替换（Tile 大小等）
        canonicalize=True,              # 启用 MLIR canonicalize pass（优化代码）
        run_bench=False,                # 不运行 benchmark 测试
        use_scheduling_barriers=False,  # 不使用调度 barrier（影响性能）
    )
    options = set_default_run_config(options)  # 设置 AMD GPU 运行时默认配置
    prefill = wave_compile(options, prefill)    # 将 Wave Kernel 编译为可执行代码

    # 调用编译后的 prefill Kernel（Q 预先乘以缩放因子和 log2e）
    mb = prefill(
        q * dk_sqrt * log2e,  # 缩放后的 Q（避免 Kernel 内部额外乘法）
        k,
        v,
        b_start_loc,          # 每个序列在 token 维度的起始位置
        b_seq_len,            # 每个序列的长度
        o,                    # 输出张量（in-place 写入）
    )
    # 调试模式：将生成的 MLIR 代码保存到文件（供检查和调试）
    if dump_generated_mlir:
        shape_list = [q.shape[0], q.shape[1], k.shape[1], q.shape[2], k.shape[2]]
        filename = f"wave_prefill_attention_{'x'.join(map(str, shape_list))}.mlir"
        with open(filename, "w") as f:
            f.write(mb.module_op.get_asm())  # 写入 MLIR 文本格式
