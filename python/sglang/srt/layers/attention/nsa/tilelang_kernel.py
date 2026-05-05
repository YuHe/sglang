# NSA TileLang GPU Kernel 模块
# 使用 TileLang DSL 编写的 NSA（Native Sparse Attention）稀疏注意力 GPU kernel
# 包含：FP8 激活量化、稀疏索引打分、稀疏 MHA prefill、稀疏 MLA decode（partial+combine）
from functools import lru_cache
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.utils import is_gfx95_supported, is_hip

# 设置 TileLang 日志级别，避免过多编译期输出
tilelang.set_log_level("WARNING")

# TileLang pass 配置：禁用 warp 特化、TMA 下降、fast math，确保正确性
pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,   # 禁用 warp 特化优化（兼容性）
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,          # 禁用 TMA 下降（兼容性）
}
# TL_DISABLE_FAST_MATH has deprecated in v0.1.7.post1 tilelang
# 旧版用 TL_DISABLE_FAST_MATH，新版用 TL_ENABLE_FAST_MATH=False
if hasattr(tilelang.PassConfigKey, "TL_DISABLE_FAST_MATH"):
    pass_configs[tilelang.PassConfigKey.TL_DISABLE_FAST_MATH] = True
elif hasattr(tilelang.PassConfigKey, "TL_ENABLE_FAST_MATH"):
    pass_configs[tilelang.PassConfigKey.TL_ENABLE_FAST_MATH] = False

# 检测硬件平台：HIP=ROCm/AMD，gfx95=高端 AMD GPU，fp8_fnuz=AMD FP8 变体
_is_hip = is_hip()
_is_gfx95_supported = is_gfx95_supported()
_is_fp8_fnuz = is_fp8_fnuz()

# 数据类型字符串常量（TileLang kernel 参数使用字符串类型名）
BF16 = "bfloat16"
FP8 = "float8_e4m3fnuz" if _is_fp8_fnuz else "float8_e4m3"  # AMD 使用 fnuz 变体
FP32 = "float32"


def fast_log2_ceil(x):
    # 快速向上取整的 log2（整数位操作实现）：从 float32 IEEE 754 位字段提取指数
    bits_x = T.reinterpret("uint32", x)         # 将 float32 重新解释为 uint32，便于位操作
    exp_x = (bits_x >> 23) & 0xFF               # 提取 8 位指数字段（偏置指数）
    man_bits = bits_x & ((1 << 23) - 1)         # 提取尾数字段（23 位），判断是否为整数幂次
    # 指数 - 127 = 真实指数；如果尾数非零则向上取整 +1
    return T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))


def fast_pow2(x):
    # 快速计算 2^x（整数 x），通过构造 float32 指数字段实现
    bits_x = (x + 127) << 23  # 加上偏置 127，移位到 IEEE 754 指数位位置
    return T.reinterpret("float32", bits_x)      # 将构造的位模式重新解释为 float32


def fast_round_scale(amax, fp8_max_inv):
    # 将 absmax scale 舍入到 2 的幂次，使量化 scale 更规整，提高 FP8 数值稳定性
    # amax * fp8_max_inv = amax / fp8_max，再向上舍入到最近的 2^k
    return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))


@lru_cache(maxsize=8)
def _pick_inner_iter(seq: int, ni: int, cu: int, block_per_cu: int) -> int:
    """
    Pick the largest valid inner_iter (power-of-two divisor of ni) that keeps
    enough work per CU (seq * ni / inner_iter / cu >= block_per_cu), so we avoid
    under-utilization while minimizing the number of partial groups.
    """
    # 根据 CU 数量和每 CU 所需最低 block 数，选择最大合法的 inner_iter（需整除 ni 且为 2 的幂次）
    # inner_iter 越大，每个 GPU block 处理的 KV tile 越多，partial group 数越少
    max_it = int(seq * ni / (cu * block_per_cu))  # 每 CU work 满足条件时允许的最大 inner_iter
    it = ni
    while it >= 2:
        if it <= max_it and ni % it == 0:  # 找到能整除 ni 且不超过 max_it 的最大值
            return it
        it //= 2
    return 1  # 默认 inner_iter=1（不合并 KV tile）


@tilelang.jit(pass_configs=pass_configs)
def act_quant_kernel(
    N, in_dtype=BF16, out_dtype=FP8, scale_dtype=FP32, round_scale=False
):
    # TileLang FP8 激活量化 kernel 工厂函数：对输入 [M, N] BF16 张量按行分组进行 absmax FP8 量化
    # 输出 Y（FP8）和 S（float32 scale），每 group_size=128 列共享一个 scale
    M = T.symbolic("M")
    fp8_min = -224.0 if _is_fp8_fnuz else -448.0  # FP8 最小值（fnuz 与标准不同）
    fp8_max = 224.0 if _is_fp8_fnuz else 448.0    # FP8 最大值
    fp8_max_inv = 1 / fp8_max                      # 提前计算倒数，避免 kernel 内除法
    num_stages = 0 if round_scale else 2           # round_scale 时不流水，普通量化流水 2 级
    blk_m = 32                                     # 每个 CUDA block 处理 32 行
    group_size = 128                               # 每 128 列共享一个量化 scale

    @T.prim_func
    def act_quant_kernel_(
        X: T.Tensor[(M, N), in_dtype],                          # 输入：BF16 [M, N]
        Y: T.Tensor[(M, N), out_dtype],                         # 输出：FP8 [M, N]
        S: T.Tensor[(M, T.ceildiv(N, group_size)), scale_dtype], # 输出：float32 scale [M, N/128]
    ):
        # 二维 kernel grid：(M/32, N/128)，每个 block 处理一个 32×128 tile
        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (
            pid_m,
            pid_n,
        ):
            x_shared = T.alloc_shared((blk_m, group_size), in_dtype)   # 共享内存：输入 tile
            x_local = T.alloc_fragment((blk_m, group_size), in_dtype)  # 线程寄存器：输入
            amax_local = T.alloc_fragment((blk_m,), scale_dtype)       # 每行 absmax 值
            s_local = T.alloc_fragment((blk_m,), scale_dtype)          # 每行量化 scale
            y_local = T.alloc_fragment((blk_m, group_size), out_dtype) # FP8 量化值
            y_shared = T.alloc_shared((blk_m, group_size), out_dtype)  # 共享内存：输出 tile

            for _ in T.Pipelined(1, num_stages=num_stages):
                T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)  # 全局→共享内存
                T.copy(x_shared, x_local)                                # 共享内存→寄存器
                T.reduce_absmax(x_local, amax_local, dim=1)              # 按行计算 absmax
                for i in T.Parallel(blk_m):
                    amax_local[i] = T.max(amax_local[i], 1e-4)          # 防止 scale 为 0（eps 保护）
                    if round_scale:
                        # 可选：将 scale 舍入到 2 的幂次（提高乘法精度）
                        s_local[i] = fast_round_scale(amax_local[i], fp8_max_inv)
                    else:
                        s_local[i] = amax_local[i] * fp8_max_inv        # scale = amax / fp8_max
                for i, j in T.Parallel(blk_m, group_size):
                    y_local[i, j] = T.clamp(
                        x_local[i, j] / s_local[i], fp8_min, fp8_max   # 量化：x/scale clamp 到 FP8 范围
                    )
                for i in T.Parallel(blk_m):
                    S[pid_m * blk_m + i, pid_n] = s_local[i]            # 写入 scale 输出
                T.copy(y_local, y_shared)                                # 寄存器→共享内存
                T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])  # 共享内存→全局

    return act_quant_kernel_


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
        scale_fmt (Optional[str], optional): The format of the scale. Default is None.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    # 公共入口：调用 TileLang kernel 对输入张量进行 block-wise FP8 量化
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"
    N = x.size(-1)  # 最后一维大小，必须被 block_size 整除
    if _is_fp8_fnuz:
        y = torch.empty_like(x, dtype=torch.float8_e4m3fnuz)   # AMD FP8 类型
    else:
        y = torch.empty_like(x, dtype=torch.float8_e4m3fn)     # NVIDIA FP8 类型
    s = x.new_empty(*x.size()[:-1], N // block_size, dtype=torch.float32)  # scale 张量 [M, N/128]
    kernel = act_quant_kernel(N, round_scale=scale_fmt is not None)  # 根据 scale_fmt 决定是否 round
    kernel(x.view(-1, N), y.view(-1, N), s.view(-1, N // block_size))  # 展平为 2D 调用 kernel
    return y, s


@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def fp8_index_kernel(h: int, d: int, clear_accum=True):
    # TileLang FP8 稀疏索引打分 kernel 工厂：计算 Q·K^T 的稀疏分数，用于 NSA top-k block 选择
    # 流程：FP8 Q·K^T → ReLU 门控 → 乘 Q scale → 行求和 → 乘 K scale → 输出稀疏打分
    b = T.symbolic("b")   # batch 维度（符号化）
    m = T.symbolic("m")   # Q 序列长度（符号化）
    n = T.symbolic("n")   # K 序列长度（符号化）

    blk_n1 = 512   # 外层 n 分块大小（每次处理 512 个 K）
    blk_n2 = 128   # 内层流水分块大小（pipeline 每步 128 个 K）

    @T.prim_func
    def fp8_index_kernel_(
        q: T.Tensor[(b, m, h, d), FP8],     # FP8 量化的 Q：[b, m, h, d]
        q_s: T.Tensor[(b, m, h), FP32],     # Q 的每-head scale：[b, m, h]
        k: T.Tensor[(b, n, d), FP8],        # FP8 量化的 K（展平 head）：[b, n, d]
        k_s: T.Tensor[(b, n), FP32],        # K 的每-token scale（e8m0 格式）：[b, n]
        o: T.Tensor[(b, m, n), FP32],       # 输出稀疏打分：[b, m, n]
    ) -> None:
        # 三维 kernel grid：(b, m, ceil(n/blk_n1))，每个 block 处理一段 K
        with T.Kernel(b, m, T.ceildiv(n, blk_n1)) as (i_b, i_m, i1_n):
            q_smem = T.alloc_shared((h, d), FP8)   # 加载当前 Q token 的所有 head FP8 数据
            T.copy(q[i_b, i_m, 0, 0], q_smem)

            q_s_frag = T.alloc_fragment(h, FP32)   # Q scale：每个 head 一个 float32 scale
            T.copy(q_s[i_b, i_m, 0], q_s_frag)

            # 内层流水：每步加载 blk_n2=128 个 K 计算得分
            for i2_n in T.Pipelined(blk_n1 // blk_n2, num_stages=2):
                k_smem = T.alloc_shared((blk_n2, d), FP8)   # 当前 K 块（128 个 K）
                T.copy(k[i_b, i1_n * blk_n1 + i2_n * blk_n2, 0], k_smem)

                k_s_frag = T.alloc_fragment(blk_n2, FP32)   # K scale：每个 K token 一个 scale
                T.copy(k_s[i_b, i1_n * blk_n1 + i2_n * blk_n2], k_s_frag)

                logits = T.alloc_fragment((blk_n2, h), FP32)  # Q·K^T logit：[blk_n2, h]
                if not clear_accum:
                    T.fill(logits, 0)
                # K^T · Q^T = logits^T：FP8 GEMM，计算 128 个 K 与当前 Q 所有 head 的点积
                T.gemm(
                    k_smem,
                    q_smem,
                    logits,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=clear_accum,
                )

                for i_h, i3_n in T.Parallel(h, blk_n2):
                    # ReLU 门控：抑制负分数；乘 Q scale 还原量化误差
                    logits[i3_n, i_h] = T.max(logits[i3_n, i_h], 0) * q_s_frag[i_h]

                logits_sum = T.alloc_fragment(blk_n2, FP32)  # 对所有 head 求和（稀疏打分）
                T.reduce_sum(logits, logits_sum, dim=1)

                for i3_n in T.Parallel(blk_n2):
                    logits_sum[i3_n] *= k_s_frag[i3_n]  # 乘 K scale 得到最终稀疏得分

                T.copy(logits_sum, o[i_b, i_m, i1_n * blk_n1 + i2_n * blk_n2])  # 写入输出

    return fp8_index_kernel_


def fp8_index(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
) -> torch.Tensor:
    """
    Perform index score using FP8 precision.

    Args:
        q (torch.Tensor): The Q tensor, must be contiguous.
        q_s (torch.Tensor): The scaling factor for Q (float), must be contiguous.
        k (torch.Tensor): The K tensor, must be contiguous.
        k_s (torch.Tensor): The scaling factor for K (e8m0 here), must be contiguous.

        fp8 q @ fp8 k -> fp32 logits
        relu(fp32 logits) * q_s (weights) -> fp32 logits
        fp32 logits -> fp32 logits_sum
        fp32 logits_sum * k_s (e8m0) -> fp32 index_score
    """
    # HIP（AMD）平台使用 clear_accum=False（累加模式），NVIDIA 使用 clear_accum=True（清零模式）
    if _is_hip:
        return fp8_index_kernel(q.shape[2], q.shape[3], False)(q, q_s, k, k_s)
    else:
        return fp8_index_kernel(q.shape[2], q.shape[3])(q, q_s, k, k_s)


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_attention_fwd_kernel_v1(
    num_heads,
    dim,
    tail_dim,
    topk,
    *,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_I=64,
    num_stages=2,
    threads=256,
):
    # V1 版本：稀疏 MHA prefill kernel（无 warp 特化）
    # 对每个 Q token，通过 top-k 索引选取 KV，计算在线 softmax + 加权聚合输出
    assert dim == tilelang.math.next_power_of_2(
        dim
    ), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(
        tail_dim
    ), f"haven't check padding correctness yet, dim={tail_dim}"
    assert is_causal == True, "non-casual is not supported"
    assert (
        topk % block_I == 0
    ), "otherwise will load some index=0 thus causing wrong kv to be loaded"
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504  # log2(e)：将 exp 转为 exp2
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    # 符号化维度：支持动态 batch/seq_len
    batch = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")
    seq_len_kv = T.symbolic("seq_len_kv")

    head_kv = num_heads // kv_group    # 每个 KV head 对应多少个 Q head（GQA）
    q_shape = [batch, seq_len, num_heads, dim + tail_dim]      # Q: [b, s, h, d+d_tail]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]   # KV: [b, s_kv, g, d+d_tail]
    o_shape = [batch, seq_len, num_heads, dim]                  # O: [b, s, h, d]（不含 rope 部分）
    indices_shape = [batch, seq_len, kv_group, topk]           # top-k 索引: [b, s, g, topk]
    indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"

    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)  # pad 到 2 的幂次且 ≥16
    if padded_H != H:
        assert kv_group == 1  # GQA 时 head 数对齐需要 kv_group=1
    BI = block_I          # 每个 KV block 的大小（默认 64）
    NI = tilelang.cdiv(topk, block_I)  # top-k block 数
    D = dim
    D_tail = tail_dim

    # 超过 64 头时拆分为多个 64-head block 处理
    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64   # 需要复制的 block 数
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64  # 每个 kernel block 处理的 head 数

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),         # type: ignore
        KV: T.Tensor(kv_shape, dtype),        # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore  # top-k KV 物理索引
        Output: T.Tensor(o_shape, dtype),     # type: ignore
    ):
        # 三维 kernel grid：(seq_len * REPLICATE_H, batch, kv_group)
        with T.Kernel(seq_len * REPLICATE_H, batch, kv_group, threads=threads) as (
            bx,
            by,
            bz,
        ):
            Q_shared = T.alloc_shared([H_per_block, D], dtype)          # Q main part 共享内存
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype) # Q rope part 共享内存
            KV_shared = T.alloc_shared([BI, D], dtype)                  # KV main part 共享内存
            K_tail_shared = T.alloc_shared([BI, D_tail], dtype)         # K rope part 共享内存
            O_shared = T.alloc_shared([H_per_block, D], dtype)          # 输出共享内存
            mask = T.alloc_fragment([BI], "bool")                       # 有效 KV 掩码（index≥0）

            acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)     # 输出累加器（float32）
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)    # 注意力分数累加器
            S_shared = T.alloc_shared([H_per_block, BI], dtype)         # softmax 分数共享内存
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)       # 在线 softmax 归一化因子
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)     # 当前 block 的归一化因子
            alpha = T.alloc_fragment([H_per_block], accum_dtype)        # 历史分数衰减因子
            m_i = T.alloc_fragment([H_per_block], accum_dtype)          # 当前最大分数（在线 softmax）
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)     # 上一步最大分数

            T.fill(acc_o, 0)           # 初始化输出累加器为 0
            T.fill(sumexp, 0)          # 初始化归一化因子为 0
            T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan（避免 nan）

            b_i, g_i = by, bz
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)   # 当前 Q 序列位置
            q_i = s_i
            max_kv_i = q_i  # 因果掩码：只关注 ≤ 当前 Q 位置的 KV

            # 计算当前 block 处理的 head 范围 [H0, H1)
            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)    # 加载 Q main part
            T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared) # 加载 Q tail（rope）

            # 流水线遍历所有 top-k KV block
            for i_i in T.Pipelined(NI, num_stages=num_stages):

                for bi_i in T.Parallel(BI):
                    # 检查当前索引是否有效（≥0），生成掩码
                    mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i] >= 0

                for bi_i, d_i in T.Parallel(BI, D):
                    # 按 top-k 索引从 KV cache 中 gather K main part
                    KV_shared[bi_i, d_i] = KV[
                        b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, d_i
                    ]
                for bi_i, d_i in T.Parallel(BI, D_tail):
                    # 按 top-k 索引从 KV cache 中 gather K tail（rope）
                    K_tail_shared[bi_i, d_i] = KV[
                        b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, D + d_i
                    ]

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    # 无效 KV（index<0）赋值 -inf，确保 softmax 后权重为 0
                    acc_s[h_i, bi_i] = T.if_then_else(
                        mask[bi_i], 0, -T.infinity(acc_s.dtype)
                    )
                # Q·K^T GEMM（main part）
                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                # Q·K^T GEMM（tail/rope part），结果累加到 acc_s
                T.gemm(
                    Q_tail_shared,
                    K_tail_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.copy(m_i, m_i_prev)                          # 保存上一轮最大值
                T.reduce_max(acc_s, m_i, dim=1, clear=False)   # 更新全局最大值
                for h_i in T.Parallel(H_per_block):
                    # 计算历史分数的衰减因子：exp2((prev_max - new_max) * scale)
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    # 在线 softmax：exp2(score * scale - max * scale)
                    acc_s[h_i, bi_i] = T.exp2(
                        acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                    )
                T.reduce_sum(acc_s, sumexp_i, dim=1)  # is this a accumulate operator?
                for h_i in T.Parallel(H_per_block):
                    # 更新归一化因子：sumexp = prev * alpha + new
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D):
                    # 用 alpha 衰减历史输出累加器
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                T.copy(acc_s, S_shared)                                          # softmax 分数→共享内存
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)  # S·V 加权聚合

            # Rescale（归一化）
            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] /= sumexp[h_i]  # 除以归一化因子得到注意力输出
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale  # 计算 log-sum-exp（供外部使用）

            T.copy(acc_o, O_shared)                      # 输出→共享内存
            T.copy(acc_o, Output[b_i, s_i, H0:H1, :])   # 共享内存→全局输出

    return main


@tilelang.jit(
    out_idx=[-1],
    compile_flags=[
        "-O3",
        "-Wno-deprecated-declarations",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--ptxas-options=-v,--register-usage-level=10",
        "-DNDEBUG",
    ],
)  # type: ignore
def sparse_attention_fwd_kernel_v2(
    num_heads: int,
    dim: int,
    tail_dim: int,
    topk: int,
    *,
    kv_group: int = 1,
    sm_scale: Optional[float] = None,
    block_I: int = 64,
):
    # V2 版本：warp 特化稀疏 MHA prefill kernel（NVIDIA CUDA 专用）
    # 384 线程分 3 组（tx<128 消费者L，128≤tx<256 消费者R，tx≥256 生产者）
    # 生产者：异步加载 KV 数据；消费者L/R：并行处理 Q·K GEMM + S·V GEMM 的左右半部分
    assert dim == tilelang.math.next_power_of_2(
        dim
    ), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(
        tail_dim
    ), f"haven't check padding correctness yet, dim={tail_dim}"
    assert (
        topk % block_I == 0
    ), "otherwise will load some index=0 thus causing wrong kv to be loaded"
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)
    threads = 384   # 384 线程：3 个 warp 组（每组 128 线程）

    # 动态符号维度
    batch = T.symbolic("batch")
    qo_len = T.symbolic("seq_len")
    num_pages = T.symbolic("num_pages")

    q_shape = [batch, qo_len, num_heads, dim + tail_dim]    # Q: [b, qo, h, d+dt]
    kv_shape = [batch, num_pages, kv_group, dim + tail_dim] # KV: [b, pages, g, d+dt]（分页存储）
    o_shape = [batch, qo_len, num_heads, dim]               # O: [b, qo, h, d]
    indices_shape = [batch, qo_len, kv_group, topk]        # top-k 索引: [b, qo, g, topk]

    indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"

    H = num_heads
    padded_H = max(tilelang.math.next_power_of_2(num_heads), 16)  # head 数向上 pad 到 2 的幂次
    if padded_H != H:
        assert kv_group == 1
    BI = block_I          # KV block 大小（默认 64）
    NI = tilelang.cdiv(topk, block_I)  # top-k block 数
    assert NI % 2 == 0, "NI should be a multiple of 2"  # 双缓冲要求 NI 为偶数
    D = dim
    D_tail = tail_dim
    # 超过 64 头时拆分为多个 64-head block
    if num_heads > 64:
        assert num_heads % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = num_heads // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64  # 每 kernel block 处理的 head 数

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),         # type: ignore
        KV: T.Tensor(kv_shape, dtype),        # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),     # type: ignore
    ):
        """
        Q: [b, qo_len, H, D + D_tail] (bfloat16)
        KV: [b, num_pages, kv_group, D + D_tail] (bfloat16)
        Indices: [b, qo_len, kv_group, topk] (int32)
        """

        # 三维 kernel grid：(qo_len * REPLICATE_H, batch, 1)
        with T.Kernel(qo_len * REPLICATE_H, batch, 1, threads=threads) as (bx, by, bz):  # type: ignore
            # 双缓冲 KV 共享内存（_l=左半维度，_r=右半维度），交替使用 _0 和 _1
            Q_shared_l = T.alloc_shared([H_per_block, D // 2], dtype)       # Q 左半 main
            Q_shared_r = T.alloc_shared([H_per_block, D // 2], dtype)       # Q 右半 main
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)    # Q tail（rope）
            KV_shared_0_l = T.alloc_shared([BI, D // 2], dtype)             # KV buffer0 左半
            KV_shared_0_r = T.alloc_shared([BI, D // 2], dtype)             # KV buffer0 右半
            KV_shared_1_l = T.alloc_shared([BI, D // 2], dtype)             # KV buffer1 左半
            KV_shared_1_r = T.alloc_shared([BI, D // 2], dtype)             # KV buffer1 右半
            K_tail_shared_0 = T.alloc_shared([BI, D_tail], dtype)           # K tail buffer0
            K_tail_shared_1 = T.alloc_shared([BI, D_tail], dtype)           # K tail buffer1
            O_shared_l = Q_shared_l  # 复用 Q 共享内存存储输出（节省 SMEM）
            O_shared_r = Q_shared_r
            is_kv_valid_0 = T.alloc_shared([BI], "bool", scope="shared")  # buffer0 有效 KV 掩码
            is_kv_valid_1 = T.alloc_shared([BI], "bool", scope="shared")  # buffer1 有效 KV 掩码

            # 消费者寄存器：L 组负责输出左半，R 组负责输出右半
            acc_o_l = T.alloc_fragment([H_per_block, D // 2], accum_dtype)  # 消费者L 输出累加器
            acc_o_r = T.alloc_fragment([H_per_block, D // 2], accum_dtype)  # 消费者R 输出累加器
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)        # 注意力分数累加器
            S_shared = T.alloc_shared([H_per_block, BI], dtype)             # softmax 分数共享内存
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)           # 在线 softmax 归一化因子
            sum_exp_shared = T.alloc_shared([H_per_block], accum_dtype)     # 归一化因子共享（消费者R读取）
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)         # 当前 block 归一化因子
            alpha_shared = T.alloc_shared([H_per_block], accum_dtype, scope="shared")  # alpha 共享（消费者R读取）
            alpha_local = T.alloc_fragment([H_per_block], accum_dtype)      # 消费者L 计算的衰减因子
            m_i = T.alloc_fragment([H_per_block], accum_dtype)              # 在线最大分数
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)         # 上一步最大分数
            indices_local = T.alloc_local([1], indices_dtype)               # 生产者本地 KV 索引
            indices_tmp = T.alloc_local([1], indices_dtype)                 # 生产者临时索引（用于 valid 检测）

            # 同步屏障：协调消费者/生产者之间的数据就绪和缓冲区释放
            bar_q = T.alloc_barrier(arrive_count=384)              # 等待所有线程加载 Q
            bar_k_0_ready = T.alloc_barrier(arrive_count=128)      # buffer0 KV 数据就绪（生产者→消费者L）
            bar_k_1_ready = T.alloc_barrier(arrive_count=128)      # buffer1 KV 数据就绪（生产者→消费者L）
            bar_k_0_free = T.alloc_barrier(arrive_count=256)       # buffer0 已用完（消费者→生产者）
            bar_k_1_free = T.alloc_barrier(arrive_count=256)       # buffer1 已用完（消费者→生产者）
            bar_sScale_and_sS_ready = T.alloc_barrier(arrive_count=256)   # S 和 alpha 就绪（消费者L→消费者R）
            bar_sScale_and_sS_free = T.alloc_barrier(arrive_count=256)    # S 和 alpha 已读完（消费者R→消费者L）

            bar_0_128 = T.alloc_barrier(arrive_count=128)          # 消费者L 内部同步
            bar_1_128 = T.alloc_barrier(arrive_count=128)          # 消费者R 内部同步
            bar_2_128 = T.alloc_barrier(arrive_count=128)          # 生产者内部同步
            bar_final = T.alloc_barrier(arrive_count=128)          # 最终归一化同步（L→R）

            b_i, g_i = by, bz
            s_i = bx if REPLICATE_H == 1 else bx // REPLICATE_H   # 当前 Q token 位置

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            tx = T.get_thread_binding()  # 获取线程 ID：0-383

            # 所有线程（0-383）共同加载 Q 到共享内存
            T.copy(Q[b_i, s_i, H0:H1, 0 : D // 2], Q_shared_l)    # Q 左半
            T.copy(Q[b_i, s_i, H0:H1, D // 2 : D], Q_shared_r)    # Q 右半
            T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)          # Q tail（rope）
            T.barrier_arrive(bar_q)  # 通知 Q 加载完成

            if tx < 128:
                # ========== 消费者 L（tx 0-127）：处理输出左半 D//2 维度 ==========
                T.set_max_nreg(240, 1)  # 最大化寄存器分配（提升 GEMM 性能）
                T.fill(sumexp, 0)
                T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan
                T.fill(acc_o_l, 0)
                T.barrier_wait(bar_q, 0)  # 等待 Q 加载完成

                # 每次处理 2 个 KV block（双缓冲：buffer0 + buffer1），共 NI/2 次迭代
                for i_i in T.serial(T.ceildiv(NI, 2)):
                    # Buffer 0：等待生产者加载 buffer0 完成
                    # with sync_at(bar_0_128, 0):
                    T.barrier_wait(bar_k_0_ready[0], (i_i & 1))  # 等待 buffer0 数据就绪（奇偶轮次交替）
                    T.barrier_arrive(bar_0_128)
                    T.barrier_wait(bar_0_128, 0)                  # 内部同步确保所有消费者L线程就绪

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        # 无效 KV 位置设为 -inf（掩码处理）
                        acc_s[h_i, bi_i] = T.if_then_else(
                            is_kv_valid_0[bi_i], 0, -T.infinity(acc_s.dtype)
                        )
                    # Q·K^T GEMM（左半 main、右半 main、tail/rope），wg_wait=-1 表示异步 wgmma
                    T.gemm(
                        Q_shared_l, KV_shared_0_l, acc_s, transpose_B=True, wg_wait=-1
                    )
                    T.gemm(
                        Q_shared_r, KV_shared_0_r, acc_s, transpose_B=True, wg_wait=-1
                    )
                    T.gemm(
                        Q_tail_shared,
                        K_tail_shared_0,
                        acc_s,
                        transpose_B=True,
                        wg_wait=-1,
                    )

                    T.wait_wgmma(0)  # 等待异步 GEMM 完成（收集所有分散 wgmma 的结果）

                    if i_i != 0:
                        # 通知消费者R：上一步的 S/alpha 已不再需要（可覆盖）
                        T.barrier_arrive(bar_sScale_and_sS_free)
                        T.barrier_wait(bar_sScale_and_sS_free, ((i_i * 2) & 1) ^ 1)

                    T.copy(m_i, m_i_prev)                          # 保存上轮最大值
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)   # 更新全局最大值（在线 softmax）
                    for h_i in T.Parallel(H_per_block):
                        # 历史分数衰减因子：exp2((prev_max - new_max) * scale)
                        alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        # 在线 softmax：exp2(score*scale - max*scale)
                        acc_s[h_i, bi_i] = T.exp2(
                            acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                        )
                    T.reduce_sum(
                        acc_s, sumexp_i, dim=1
                    )  # is this a accumulate operator?
                    for h_i in T.Parallel(H_per_block):
                        # 更新归一化因子：sumexp = prev * alpha + new
                        sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                    for h_i, d_i in T.Parallel(H_per_block, D // 2):
                        acc_o_l[h_i, d_i] *= alpha_local[h_i]    # 用 alpha 衰减历史输出
                    T.copy(alpha_local, alpha_shared)              # 将 alpha 写入共享内存（消费者R 使用）

                    T.copy(acc_s, S_shared)                        # softmax 分数→共享内存（消费者R 使用）
                    T.gemm(S_shared, KV_shared_0_l, acc_o_l)       # S·V 左半加权聚合

                    T.barrier_arrive(bar_sScale_and_sS_ready)      # 通知消费者R：S/alpha 已就绪
                    T.barrier_arrive(bar_k_0_free[0])              # 通知生产者：buffer0 已用完

                    # Buffer 1：等待生产者加载 buffer1 完成
                    T.barrier_wait(bar_k_1_ready[0], (i_i & 1))   # 等待 buffer1 数据就绪
                    T.barrier_arrive(bar_0_128)
                    T.barrier_wait(bar_0_128, 1)

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(
                            is_kv_valid_1[bi_i], 0, -T.infinity(acc_s.dtype)
                        )
                    T.gemm(
                        Q_shared_l, KV_shared_1_l, acc_s, transpose_B=True, wg_wait=-1
                    )
                    T.gemm(
                        Q_shared_r, KV_shared_1_r, acc_s, transpose_B=True, wg_wait=-1
                    )
                    T.gemm(
                        Q_tail_shared,
                        K_tail_shared_1,
                        acc_s,
                        transpose_B=True,
                        wg_wait=-1,
                    )

                    T.wait_wgmma(0)

                    T.barrier_arrive(bar_sScale_and_sS_free)
                    T.barrier_wait(bar_sScale_and_sS_free, ((i_i * 2 + 1) & 1) ^ 1)

                    T.copy(m_i, m_i_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for h_i in T.Parallel(H_per_block):
                        alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.exp2(
                            acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                        )
                    T.reduce_sum(
                        acc_s, sumexp_i, dim=1
                    )  # is this a accumulate operator?
                    for h_i in T.Parallel(H_per_block):
                        sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                    for h_i, d_i in T.Parallel(H_per_block, D // 2):
                        acc_o_l[h_i, d_i] *= alpha_local[h_i]
                    T.copy(alpha_local, alpha_shared)

                    T.copy(acc_s, S_shared)
                    T.gemm(S_shared, KV_shared_1_l, acc_o_l)

                    T.barrier_arrive(bar_sScale_and_sS_ready)  # 通知消费者R：S/alpha 已就绪
                    T.barrier_arrive(bar_k_1_free[0])          # 通知生产者：buffer1 已用完

                # Rescale（归一化输出）
                for h_i in T.Parallel(H_per_block):
                    sum_exp_shared[h_i] = sumexp[h_i]  # 将归一化因子写入共享内存（消费者R 读取）
                T.barrier_arrive(bar_final)                    # 通知消费者R 可以做最终归一化
                for h_i, d_i in T.Parallel(H_per_block, D // 2):
                    acc_o_l[h_i, d_i] /= sumexp[h_i]          # 归一化输出左半
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale  # log-sum-exp
                T.copy(acc_o_l, O_shared_l)                    # 结果→共享内存
                T.copy(O_shared_l, Output[b_i, s_i, H0:H1, 0 : D // 2])  # 共享内存→全局输出
            elif tx >= 128 and tx < 256:
                # ========== 消费者 R（tx 128-255）：处理输出右半 D//2 维度 ==========
                # T.set_max_nreg(168, 1)
                T.fill(acc_o_r, 0)
                for i_i in T.serial(T.ceildiv(NI, 2)):
                    # Buffer 0：等待消费者L 计算好 S/alpha
                    T.barrier_arrive(bar_sScale_and_sS_ready)
                    T.barrier_wait(bar_sScale_and_sS_ready, ((i_i * 2) & 1))  # 等待 S/alpha 就绪
                    T.barrier_arrive(bar_1_128)
                    T.barrier_wait(bar_1_128, 0)
                    for h_i, d_i in T.Parallel(H_per_block, D // 2):
                        acc_o_r[h_i, d_i] *= alpha_shared[h_i]  # 用共享内存中的 alpha 衰减历史输出
                    T.gemm(S_shared, KV_shared_0_r, acc_o_r)    # S·V 右半加权聚合
                    T.barrier_arrive(bar_k_0_free[0])            # 通知生产者：buffer0 右半用完
                    T.barrier_arrive(bar_sScale_and_sS_free)     # 通知消费者L：S/alpha 可被覆盖

                    # Buffer 1：等待消费者L 计算好 S/alpha
                    T.barrier_arrive(bar_sScale_and_sS_ready)
                    T.barrier_wait(bar_sScale_and_sS_ready, ((i_i * 2 + 1) & 1))
                    T.barrier_arrive(bar_1_128)
                    T.barrier_wait(bar_1_128, 1)
                    for h_i, d_i in T.Parallel(H_per_block, D // 2):
                        acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                    T.gemm(S_shared, KV_shared_1_r, acc_o_r)    # S·V 右半加权聚合
                    T.barrier_arrive(bar_k_1_free[0])            # 通知生产者：buffer1 右半用完
                    if i_i != T.ceildiv(NI, 2) - 1:
                        T.barrier_arrive(bar_sScale_and_sS_free)

                # Rescale
                T.barrier_wait(bar_final, 0)                   # 等待消费者L 写入 sum_exp_shared
                for h_i, d_i in T.Parallel(H_per_block, D // 2):
                    acc_o_r[h_i, d_i] /= sum_exp_shared[h_i]  # 归一化输出右半

                T.copy(acc_o_r, O_shared_r)                    # 结果→共享内存
                T.copy(O_shared_r, Output[b_i, s_i, H0:H1, D // 2 : D])  # 共享内存→全局输出
            elif tx >= 256:
                # ========== 生产者（tx 256-383）：异步加载 KV 数据到双缓冲共享内存 ==========
                T.set_max_nreg(80, 0)   # 减少生产者寄存器分配（释放寄存器给消费者）
                indices_local[0] = 0    # 初始化本地索引（用于 invalid KV 时填充 page 0 数据）
                for i_i in T.serial(T.ceildiv(NI, 2)):
                    # Buffer 0：等待消费者用完 buffer0（上一轮）
                    T.barrier_wait(bar_k_0_free[0], ((i_i & 1) ^ 1))
                    T.barrier_arrive(bar_2_128)
                    T.barrier_wait(bar_2_128, 0)  # 等待所有生产者线程就绪，开始加载 buffer0

                    for r in T.serial(4):
                        # 每个生产者线程负责 buffer 中的 1/8 行（128线程/8=16行，r×16+offset）
                        indices_tmp[0] = Indices[
                            b_i, s_i, g_i, (i_i * 2) * BI + r * 16 + (tx - 256) // 8
                        ]
                        is_kv_valid_0[r * 16 + (tx - 256) // 8] = indices_tmp[0] >= 0  # 判断是否有效 KV
                        if is_kv_valid_0[r * 16 + (tx - 256) // 8]:
                            indices_local[0] = indices_tmp[0]  # 更新本地有效索引

                        with T.attr("default", "async_scope", 1):  # type: ignore  # 异步加载
                            for u in T.serial(4):
                                for v in T.vectorized(8):
                                    # 异步加载 KV main part 左半（前 D//2 列）到 buffer0
                                    KV_shared_0_l[
                                        r * 16 + (tx - 256) // 8,
                                        64 * u + (tx - 256) % 8 * 8 + v,
                                    ] = KV[
                                        b_i,
                                        indices_local[0],
                                        g_i,
                                        64 * u + (tx - 256) % 8 * 8 + v,
                                    ]
                                    # 异步加载 KV main part 右半（后 D//2 列）到 buffer0
                                    KV_shared_0_r[
                                        r * 16 + (tx - 256) // 8,
                                        64 * u + (tx - 256) % 8 * 8 + v,
                                    ] = KV[
                                        b_i,
                                        indices_local[0],
                                        g_i,
                                        D // 2 + 64 * u + (tx - 256) % 8 * 8 + v,
                                    ]
                        with T.attr("default", "async_scope", 1):  # type: ignore  # 异步加载 K tail
                            for v in T.vectorized(8):
                                # 异步加载 K tail（rope）到 buffer0
                                K_tail_shared_0[
                                    r * 16 + (tx - 256) // 8, (tx - 256) % 8 * 8 + v
                                ] = KV[
                                    b_i,
                                    indices_local[0],
                                    g_i,
                                    D + (tx - 256) % 8 * 8 + v,
                                ]

                    T.cp_async_barrier_noinc(bar_k_0_ready[0])  # 通知消费者L：buffer0 数据就绪

                    # Buffer 1：等待消费者用完 buffer1（上一轮）
                    T.barrier_wait(bar_k_1_free[0], ((i_i & 1) ^ 1))
                    T.barrier_arrive(bar_2_128)
                    T.barrier_wait(bar_2_128, 1)

                    for r in T.serial(4):
                        indices_tmp[0] = Indices[
                            b_i, s_i, g_i, (i_i * 2 + 1) * BI + r * 16 + (tx - 256) // 8
                        ]
                        is_kv_valid_1[r * 16 + (tx - 256) // 8] = indices_tmp[0] >= 0
                        if is_kv_valid_1[r * 16 + (tx - 256) // 8]:
                            indices_local[0] = indices_tmp[0]

                        with T.attr("default", "async_scope", 1):  # type: ignore
                            for u in T.serial(4):
                                for v in T.vectorized(8):
                                    # 异步加载 KV 左半到 buffer1
                                    KV_shared_1_l[
                                        r * 16 + (tx - 256) // 8,
                                        64 * u + (tx - 256) % 8 * 8 + v,
                                    ] = KV[
                                        b_i,
                                        indices_local[0],
                                        g_i,
                                        64 * u + (tx - 256) % 8 * 8 + v,
                                    ]
                                    # 异步加载 KV 右半到 buffer1
                                    KV_shared_1_r[
                                        r * 16 + (tx - 256) // 8,
                                        64 * u + (tx - 256) % 8 * 8 + v,
                                    ] = KV[
                                        b_i,
                                        indices_local[0],
                                        g_i,
                                        D // 2 + 64 * u + (tx - 256) % 8 * 8 + v,
                                    ]
                        with T.attr("default", "async_scope", 1):  # type: ignore
                            for v in T.vectorized(8):
                                # 异步加载 K tail 到 buffer1
                                K_tail_shared_1[
                                    r * 16 + (tx - 256) // 8, (tx - 256) % 8 * 8 + v
                                ] = KV[
                                    b_i,
                                    indices_local[0],
                                    g_i,
                                    D + (tx - 256) % 8 * 8 + v,
                                ]

                    T.cp_async_barrier_noinc(bar_k_1_ready[0])  # 通知消费者L：buffer1 数据就绪

    return main


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_mla_fwd_decode_partial(
    heads,
    dim,
    tail_dim,
    topk,
    *,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_I=64,
    inner_iter=1,
    num_stages=1,
    threads=256,
):
    """
    grid: (seq_len * REPLICATE_H, top_k / block_I / inner_iter)
    Each GPU block processes `inner_iter` consecutive KV tiles and writes one (partial_o, partial_lse) entry.
    """
    # BF16 稀疏 MLA decode partial attention kernel（AMD HIP 专用）
    # 将 topk KV 分成若干组，每组（group）对应一个 GPU block；
    # 每个 block 处理 inner_iter 个连续 KV tile，写出 partial_o 和 partial_lse（log-sum-exp）
    # 最终由 sparse_mla_fwd_decode_combine 合并各组的部分结果

    assert is_causal == True, "non-causal is not supported"
    assert kv_group == 1
    assert topk % block_I == 0
    assert topk % (block_I * inner_iter) == 0, (
        f"topk ({topk}) must be divisible by block_I * inner_iter = "
        f"{block_I} * {inner_iter}"
    )

    # log2(e) = 1.44269504
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504  # 默认 scale
    else:
        sm_scale = sm_scale * 1.44269504  # 乘以 log2(e) 将 exp 转换为 exp2

    # 批大小固定为 1（调用方通过 unsqueeze 添加 batch 维度）
    batch = 1
    seq_len = T.dynamic("seq_len")       # Q 序列长度（动态）
    seq_len_kv = T.dynamic("seq_len_kv") # KV 序列长度（动态）

    head_kv = heads // kv_group          # 每 KV head 对应多少 Q head
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)  # head 数 pad 到 2 的幂次
    REPLICATE_H = (head_kv // 64) if head_kv > 64 else 1  # 超过 64 头时拆分
    H_per_block = padded_H if REPLICATE_H == 1 else 64   # 每 kernel block 处理的 head 数
    N_GROUPS = topk // (block_I * inner_iter)             # 总 group 数（partial 结果数量）
    BI = block_I  # KV block 大小
    D = dim       # main 维度
    D_tail = tail_dim  # tail/rope 维度

    q_shape = [batch, seq_len, heads, dim + tail_dim]              # Q: [1, s, h, d+dt]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]       # KV: [1, s_kv, 1, d+dt]
    indices_shape = [batch, seq_len, kv_group, topk]               # 索引: [1, s, 1, topk]
    partial_o_shape = [batch, seq_len, N_GROUPS, heads, dim]       # 部分输出: [1, s, n_g, h, d]
    partial_lse_shape = [batch, seq_len, N_GROUPS, heads]          # 部分 LSE: [1, s, n_g, h]
    indices_dtype = T.int32
    dtype = T.bfloat16
    accum_dtype = T.float32

    _q_in_shared = inner_iter == 1  # inner_iter==1 时 Q 放共享内存（否则寄存器）

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(kv_shape, dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Partial_O: T.Tensor(partial_o_shape, dtype),
        Partial_Lse: T.Tensor(partial_lse_shape, accum_dtype),
    ):
        # 二维 kernel grid：(seq_len * REPLICATE_H, N_GROUPS)
        with T.Kernel(seq_len * REPLICATE_H, N_GROUPS, threads=threads) as (bx, by):
            if _q_in_shared:
                Q_buf = T.alloc_shared([H_per_block, D], dtype)       # Q main 共享内存
                Q_tail_buf = T.alloc_shared([H_per_block, D_tail], dtype)  # Q tail 共享内存
            else:
                Q_buf = T.alloc_fragment([H_per_block, D], dtype)     # Q main 寄存器片段
                Q_tail_buf = T.alloc_fragment([H_per_block, D_tail], dtype)

            KV_shared = T.alloc_shared([BI, D], dtype)                # KV main 共享内存
            K_tail_shared = T.alloc_shared([BI, D_tail], dtype)       # K tail 共享内存
            S_shared = T.alloc_shared([H_per_block, BI], dtype)       # softmax 分数共享内存
            mask = T.alloc_fragment([BI], T.bool)                     # 有效 KV 掩码

            acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)   # 输出累加器（float32）
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)  # 注意力分数累加器
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)     # 在线 softmax 归一化因子
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)   # 当前 block 归一化因子
            alpha = T.alloc_fragment([H_per_block], accum_dtype)      # 历史分数衰减因子
            m_i = T.alloc_fragment([H_per_block], accum_dtype)        # 当前最大分数
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)   # 上一步最大分数

            T.fill(acc_o, 0)            # 初始化输出累加器
            T.fill(sumexp, 0)           # 初始化归一化因子
            T.fill(m_i, -(2**30))       # 初始化最大分数（避免 nan）

            b_i, g_i = 0, 0
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)  # Q token 序列位置
            group_i = by                                             # 当前处理的 KV group 编号
            H0 = 0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64  # head 起始偏移
            H1 = H0 + H_per_block

            T.copy(Q[b_i, s_i, H0:H1, :D], Q_buf)       # 加载 Q main part
            T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_buf)  # 加载 Q tail（rope）

            # 流水线遍历当前 group 的所有 KV tile（inner_iter 个 tile）
            for k_i in T.Pipelined(inner_iter, num_stages=num_stages):
                topk_block_i = group_i * inner_iter + k_i  # 全局 KV block 编号

                for bi_i in T.Parallel(BI):
                    # 判断当前 KV 索引是否有效（≥0），生成掩码
                    mask[bi_i] = Indices[b_i, s_i, g_i, topk_block_i * BI + bi_i] >= 0
                for bi_i, d_i in T.Parallel(BI, D):
                    idx = Indices[b_i, s_i, g_i, topk_block_i * BI + bi_i]
                    # 按 top-k 索引 gather K main part（无效时 fallback 到 index 0）
                    KV_shared[bi_i, d_i] = KV[
                        b_i, T.if_then_else(idx >= 0, idx, 0), g_i, d_i
                    ]
                for bi_i, d_i in T.Parallel(BI, D_tail):
                    idx = Indices[b_i, s_i, g_i, topk_block_i * BI + bi_i]
                    # 按 top-k 索引 gather K tail（rope）
                    K_tail_shared[bi_i, d_i] = KV[
                        b_i, T.if_then_else(idx >= 0, idx, 0), g_i, D + d_i
                    ]

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    # 无效 KV 设为 -inf（softmax 后权重为 0）
                    acc_s[h_i, bi_i] = T.if_then_else(
                        mask[bi_i], 0, -T.infinity(acc_s.dtype)
                    )

                # Q·K^T GEMM（main + tail 两部分，结果累加到 acc_s）
                T.gemm(
                    Q_buf,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.gemm(
                    Q_tail_buf,
                    K_tail_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                T.copy(m_i, m_i_prev)                          # 保存上轮最大值
                T.reduce_max(acc_s, m_i, dim=1, clear=False)   # 更新全局最大值
                for h_i in T.Parallel(H_per_block):
                    # 计算历史分数衰减因子：exp2((prev_max - new_max) * scale)
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    # 在线 softmax：exp2(score * scale - max * scale)
                    acc_s[h_i, bi_i] = T.exp2(
                        acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                    )
                T.reduce_sum(acc_s, sumexp_i, dim=1)            # 当前 block 归一化因子
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]  # 更新累计归一化因子
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] *= alpha[h_i]              # 衰减历史输出

                T.copy(acc_s, S_shared)                                         # softmax 分数→共享内存
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)  # S·V 加权聚合

            # sumexp==0 (all masked), divide by 1 to get 0 and avoid nan
            # 全掩码情况：归一化因子为 0 时除以 1（避免 NaN），输出也为 0
            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] = acc_o[h_i, d_i] / T.if_then_else(
                    sumexp[h_i] == 0.0, 1.0, sumexp[h_i]
                )
            # sumexp==0 (all masked), use large negative so combine ignores this split
            # 全掩码时 LSE 设为极小值，combine 阶段会忽略此 split
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.if_then_else(
                    sumexp[h_i] == 0.0,
                    -(2**30),  # 大负数确保 combine 阶段 exp2(lse - lse_max) ≈ 0
                    T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale,  # log-sum-exp
                )

            # 写出当前 group 的部分输出和 LSE
            T.copy(acc_o, Partial_O[b_i, s_i, group_i, H0:H1, :])
            T.copy(sumexp, Partial_Lse[b_i, s_i, group_i, H0:H1])

    return main


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_mla_fwd_decode_combine(
    heads,
    dim,
    topk,
    head_per_block,
    *,
    block_I=64,
    threads=256,
):
    """
    grid: (seq_len * REPLICATE_H). batch=1, kv_group=1.
    Each block does one tile of heads (e.g. 4 or 8 for decode).
    """
    # BF16 稀疏 MLA decode combine kernel（AMD HIP 专用）
    # 将 sparse_mla_fwd_decode_partial 输出的 NI 个部分结果合并为最终输出
    # 合并算法：log-sum-exp 归一化：scale_k = exp2(lse_k - lse_max - log2(sum_exp))
    # acc_o = sum_k(scale_k * partial_o_k)

    assert heads % head_per_block == 0, f"head_per_block must divide heads"

    batch = 1
    seq_len = T.dynamic("seq_len")

    NI = topk // block_I           # partial 结果数量（每 block_I 个 KV 生成一个 partial）
    H_per_block = head_per_block   # 每个 kernel block 处理的 head 数（decode 通常 4 或 8）
    REPLICATE_H = heads // H_per_block  # 总 head 数 / 每 block head 数 = 需要多少个 block

    # 形状定义
    partial_o_shape = [batch, seq_len, NI, heads, dim]     # 部分输出: [1, s, NI, h, d]
    partial_lse_shape = [batch, seq_len, NI, heads]        # 部分 LSE: [1, s, NI, h]
    o_shape = [batch, seq_len, heads, dim]                 # 最终输出: [1, s, h, d]
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Partial_O: T.Tensor(partial_o_shape, dtype),
        Partial_Lse: T.Tensor(partial_lse_shape, accum_dtype),
        Output: T.Tensor(o_shape, dtype),
    ):
        # 一维 kernel grid：(seq_len * REPLICATE_H)，每个 block 处理一个 token 的一组 head
        with T.Kernel(seq_len * REPLICATE_H, threads=threads) as (bx,):
            shared_lse = T.alloc_shared([NI, H_per_block], accum_dtype)  # 所有 NI 个 partial LSE 缓冲

            lse_max = T.alloc_fragment([H_per_block], accum_dtype)   # 各 head 的最大 LSE
            lse_sum = T.alloc_fragment([H_per_block], accum_dtype)   # 各 head 的 exp(lse) 之和
            scale = T.alloc_fragment([H_per_block, NI], accum_dtype) # 各 head 各 partial 的归一化系数
            acc_o = T.alloc_fragment([H_per_block, dim], accum_dtype) # 最终输出累加器

            b_i = 0
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)   # Q token 序列位置
            H0 = 0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * H_per_block  # head 起始偏移
            H1 = H0 + H_per_block

            # 1. 加载所有 NI 个 partial LSE 到共享内存
            for k in T.serial(NI):
                T.copy(Partial_Lse[b_i, s_i, k, H0:H1], shared_lse[k, :])

            # 2. 计算各 head 的 lse 最大值（用于数值稳定）
            T.fill(lse_max, -(2**30))
            for k in T.serial(NI):
                for h_i in T.Parallel(H_per_block):
                    lse_max[h_i] = T.max(lse_max[h_i], shared_lse[k, h_i])  # 逐 partial 取最大

            # 3. 计算 lse_sum = sum_k(exp2(lse_k - lse_max))
            T.fill(lse_sum, 0)
            for k in T.serial(NI):
                for h_i in T.Parallel(H_per_block):
                    lse_sum[h_i] = lse_sum[h_i] + T.exp2(
                        shared_lse[k, h_i] - lse_max[h_i]  # 稳定计算 exp（减去最大值）
                    )

            # 4. 计算每个 partial 的归一化系数：scale_k = exp2(lse_k - lse_max - log2(lse_sum))
            for k in T.serial(NI):
                for h_i in T.Parallel(H_per_block):
                    scale[h_i, k] = T.exp2(
                        shared_lse[k, h_i] - lse_max[h_i] - T.log2(lse_sum[h_i])
                    )

            # 5. 加权合并所有 partial 输出：acc_o = sum_k(scale_k * partial_o_k)
            T.fill(acc_o, 0)
            for k in T.serial(NI):
                for h_i, d_i in T.Parallel(H_per_block, dim):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] + scale[h_i, k] * Partial_O[
                        b_i, s_i, k, H0 + h_i, d_i
                    ].astype(accum_dtype)  # 将 partial_o（BF16）转 float32 再加权累加

            T.copy(acc_o, Output[b_i, s_i, H0:H1, :])  # 写出最终合并结果到全局内存

    return main


@tilelang.jit(out_idx=[-2, -1], pass_configs=pass_configs)
def sparse_mla_fwd_decode_partial_fp8(
    num_heads: int,
    d_v: int,
    d_tail: int,
    topk: int,
    *,
    sm_scale=None,
    block_I=64,
    inner_iter=1,
    threads=256,
):
    # FP8 版稀疏 MLA decode partial attention kernel（AMD HIP 专用，与 BF16 版配套）
    # 将 K=512 的主维度拆分为 4×128 tile（减少 MFMA 累加依赖链，提升性能）
    # softmax 分数在 [0,1] 范围内，乘 fp8_max_val 后 cast 到 FP8（充分利用 FP8 动态范围）
    assert d_v == 512, f"only support d_v=512"
    assert (
        topk % block_I == 0
    ), "otherwise will load some index=0 thus causing wrong kv to be loaded"

    # Softmax scores are in [0, 1]. We scale by fp8_max_val before FP8 cast
    # to better utilize FP8 dynamic range, then apply the inverse scale after GEMM.
    # This is numerically safe because softmax output is bounded by 1.
    # softmax 分数 clamp 到 FP8 范围后，GEMM 结果再乘 s_scale_const 还原
    fp8_dtype = "float8_e4m3fnuz" if _is_fp8_fnuz else "float8_e4m3fn"
    fp8_max_val = 240.0 if _is_fp8_fnuz else 448.0  # FP8 最大值（fnuz=240, e4m3fn=448）
    s_inv_scale_const = fp8_max_val          # softmax 分数乘此值后 cast FP8（放大到 FP8 范围）
    s_scale_const = 1.0 / fp8_max_val       # GEMM 后乘此值还原（除以放大因子）

    BI = block_I                             # KV block 大小（默认 64）
    group_size = 128                         # K=512 的分块大小（4 个 128-tile）
    dim_quant_fp8 = d_v + d_tail             # FP8 KV 总维度（d_v + d_tail）
    rope_offset_fp8 = d_v                    # rope 部分在 KV 中的偏移（d_v=512 之后）
    n_groups = topk // (BI * inner_iter)     # 总 group 数（每组输出一个 partial）

    if sm_scale is None:
        sm_scale = (1.0 / (d_v + d_tail)) ** 0.5 * 1.44269504  # 默认 scale * log2(e)
    else:
        sm_scale = sm_scale * 1.44269504     # 乘 log2(e) 将 exp 转 exp2

    h_per_block = 16                         # 每个 kernel block 固定处理 16 个 head
    # Match bf16 partial behavior: keep fixed 16-head tiles and use
    # sliced T.copy on H0:H1 for tail handling.
    assert (
        num_heads <= h_per_block or num_heads % h_per_block == 0
    ), "num_heads must be <=16 or divisible by 16"
    head_blocks_per_seq = (num_heads + h_per_block - 1) // h_per_block  # 处理全部 head 需要多少个 block

    batch = 1       # 固定 batch=1（调用方通过 unsqueeze 添加）
    kv_group = 1    # 固定 kv_group=1
    seq_len = T.symbolic("seq_len")     # Q 序列长度（符号化）
    num_pages = T.symbolic("num_pages") # KV 页表总页数（符号化）

    # 形状定义（FP8 版本 KV 为 [b, pages, group, d_v+d_tail]）
    q_fp8_shape = [batch, seq_len, num_heads, d_v + d_tail]
    kv_fp8_shape = [batch, num_pages, kv_group, dim_quant_fp8]
    idx_shape = [batch, seq_len, kv_group, topk]
    partial_o_shape = [batch, seq_len, n_groups, num_heads, d_v]  # 部分输出（不含 rope）
    partial_lse_shape = [batch, seq_len, n_groups, num_heads]

    accum_dtype = T.float32
    dtype_bf16 = T.bfloat16

    @T.prim_func
    def main(
        q_fp8: T.Tensor(q_fp8_shape, fp8_dtype),              # FP8 Q：[1, s, h, d_v+d_tail]
        kv_fp8: T.Tensor(kv_fp8_shape, fp8_dtype),            # FP8 KV（分页）：[1, pages, 1, d_v+d_tail]
        indices: T.Tensor(idx_shape, T.int32),                 # top-k 页表索引：[1, s, 1, topk]
        partial_o: T.Tensor(partial_o_shape, dtype_bf16),      # 输出部分 O：[1, s, n_g, h, d_v]
        partial_lse: T.Tensor(partial_lse_shape, accum_dtype), # 输出部分 LSE：[1, s, n_g, h]
    ):
        # 二维 kernel grid：(seq_len * head_blocks_per_seq, n_groups)
        with T.Kernel(seq_len * head_blocks_per_seq, n_groups, threads=threads) as (
            bx,
            by,
        ):
            b_i, g_i = 0, 0
            s_i = bx // head_blocks_per_seq    # Q token 序列位置
            group_i = by                        # 当前 KV group 编号
            H0 = (bx % head_blocks_per_seq) * h_per_block  # head 起始偏移
            H1 = H0 + h_per_block

            # We intentionally split the K=512 GEMM into 4x128 tiles.
            # Although this adds extra intermediate memory traffic,
            # it shortens the MFMA accumulation dependency chain and improves performance.
            # 将 K=512 主维度拆成 4×128 tile：4 个 Q tile 和 4 个 KV tile 的共享内存
            q_tile0 = T.alloc_shared([h_per_block, group_size], fp8_dtype)  # Q tile 0（第 1 个 128 列）
            q_tile1 = T.alloc_shared([h_per_block, group_size], fp8_dtype)  # Q tile 1
            q_tile2 = T.alloc_shared([h_per_block, group_size], fp8_dtype)  # Q tile 2
            q_tile3 = T.alloc_shared([h_per_block, group_size], fp8_dtype)  # Q tile 3
            kv_tile0 = T.alloc_shared([BI, group_size], fp8_dtype)          # KV tile 0
            kv_tile1 = T.alloc_shared([BI, group_size], fp8_dtype)          # KV tile 1
            kv_tile2 = T.alloc_shared([BI, group_size], fp8_dtype)          # KV tile 2
            kv_tile3 = T.alloc_shared([BI, group_size], fp8_dtype)          # KV tile 3
            q_tail_buf = T.alloc_shared([h_per_block, d_tail], fp8_dtype)   # Q tail（rope）共享内存
            k_tail_shared = T.alloc_shared([BI, d_tail], fp8_dtype)         # K tail（rope）共享内存
            s_fp8_shared = T.alloc_shared([h_per_block, BI], fp8_dtype)     # softmax 分数 FP8（clamp 后）
            page_idx_shared = T.alloc_shared([BI], T.int32)                 # 当前 KV block 的物理页索引

            mask = T.alloc_fragment([BI], T.bool)                           # 有效 KV 掩码
            acc_s = T.alloc_fragment([h_per_block, BI], accum_dtype)        # 注意力分数累加器（4 tile 合并）
            acc_tile = T.alloc_fragment([h_per_block, BI], accum_dtype)     # 临时 tile GEMM 结果
            sv_tile = T.alloc_fragment([h_per_block, group_size], accum_dtype)  # S·V 部分结果
            sumexp = T.alloc_fragment([h_per_block], accum_dtype)           # 在线 softmax 归一化因子
            sumexp_i = T.alloc_fragment([h_per_block], accum_dtype)         # 当前 block 归一化因子
            alpha = T.alloc_fragment([h_per_block], accum_dtype)            # 历史分数衰减因子
            m_i = T.alloc_fragment([h_per_block], accum_dtype)              # 当前最大分数
            m_i_prev = T.alloc_fragment([h_per_block], accum_dtype)         # 上一步最大分数
            inv_denom = T.alloc_fragment([h_per_block], accum_dtype)        # 归一化因子的倒数

            # 4 个输出 tile（对应 d_v=512 的 4×128 列）
            acc_o_tile0 = T.alloc_fragment([h_per_block, group_size], accum_dtype)
            acc_o_tile1 = T.alloc_fragment([h_per_block, group_size], accum_dtype)
            acc_o_tile2 = T.alloc_fragment([h_per_block, group_size], accum_dtype)
            acc_o_tile3 = T.alloc_fragment([h_per_block, group_size], accum_dtype)

            T.fill(acc_o_tile0, 0)     # 初始化输出累加器
            T.fill(acc_o_tile1, 0)
            T.fill(acc_o_tile2, 0)
            T.fill(acc_o_tile3, 0)
            T.fill(sumexp, 0)          # 初始化归一化因子
            T.fill(m_i, -(2**30))      # 初始化最大分数（避免 nan）

            T.copy(q_fp8[b_i, s_i, H0:H1, d_v:], q_tail_buf)                        # 加载 Q tail（rope）
            T.copy(q_fp8[b_i, s_i, H0:H1, 0 * group_size : 1 * group_size], q_tile0) # 加载 Q 第 1 个 128 列 tile
            T.copy(q_fp8[b_i, s_i, H0:H1, 1 * group_size : 2 * group_size], q_tile1) # 加载 Q 第 2 个 tile
            T.copy(q_fp8[b_i, s_i, H0:H1, 2 * group_size : 3 * group_size], q_tile2) # 加载 Q 第 3 个 tile
            T.copy(q_fp8[b_i, s_i, H0:H1, 3 * group_size : 4 * group_size], q_tile3) # 加载 Q 第 4 个 tile

            for k_i in T.serial(inner_iter):
                topk_block_i = group_i * inner_iter + k_i  # 全局 top-k block 编号

                for bi_i in T.Parallel(BI):
                    idx = indices[b_i, s_i, g_i, topk_block_i * BI + bi_i]
                    valid = idx >= 0
                    page_idx_shared[bi_i] = T.if_then_else(valid, idx, 0)  # 有效时用真实页索引，否则 0
                    mask[bi_i] = valid                                       # 记录有效性掩码

                for bi_i, j in T.Parallel(BI, group_size):
                    page = page_idx_shared[bi_i]
                    # 按页索引 gather FP8 KV 的 4 个 128-tile
                    kv_tile0[bi_i, j] = kv_fp8[b_i, page, g_i, 0 * group_size + j]
                    kv_tile1[bi_i, j] = kv_fp8[b_i, page, g_i, 1 * group_size + j]
                    kv_tile2[bi_i, j] = kv_fp8[b_i, page, g_i, 2 * group_size + j]
                    kv_tile3[bi_i, j] = kv_fp8[b_i, page, g_i, 3 * group_size + j]

                for bi_i, j in T.Parallel(BI, d_tail):
                    page = page_idx_shared[bi_i]
                    k_tail_shared[bi_i, j] = kv_fp8[b_i, page, g_i, rope_offset_fp8 + j]  # 加载 K tail

                for h_i, bi_i in T.Parallel(h_per_block, BI):
                    # 无效 KV 设为 -inf
                    acc_s[h_i, bi_i] = T.if_then_else(
                        mask[bi_i], 0, -T.infinity(acc_s.dtype)
                    )

                # Q·K^T：分 4 个 128-tile GEMM，结果累加到 acc_s
                T.gemm(q_tile0, kv_tile0, acc_s, transpose_B=True, clear_accum=False)   # tile0 不清零（累加）
                T.gemm(q_tile1, kv_tile1, acc_tile, transpose_B=True, clear_accum=True)
                for h_i, bi_i in T.Parallel(h_per_block, BI):
                    acc_s[h_i, bi_i] += acc_tile[h_i, bi_i]   # 累加 tile1 结果
                T.gemm(q_tile2, kv_tile2, acc_tile, transpose_B=True, clear_accum=True)
                for h_i, bi_i in T.Parallel(h_per_block, BI):
                    acc_s[h_i, bi_i] += acc_tile[h_i, bi_i]   # 累加 tile2 结果
                T.gemm(q_tile3, kv_tile3, acc_tile, transpose_B=True, clear_accum=True)
                for h_i, bi_i in T.Parallel(h_per_block, BI):
                    acc_s[h_i, bi_i] += acc_tile[h_i, bi_i]   # 累加 tile3 结果
                # 对 tail 部分单独 GEMM（使用 FullCol policy）
                T.gemm(
                    q_tail_buf,
                    k_tail_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                T.copy(m_i, m_i_prev)                          # 保存上轮最大值
                T.reduce_max(acc_s, m_i, dim=1, clear=False)   # 更新全局最大值（在线 softmax）
                for h_i in T.Parallel(h_per_block):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)  # 衰减因子
                for h_i, bi_i in T.Parallel(h_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(
                        acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale  # 在线 softmax
                    )
                T.reduce_sum(acc_s, sumexp_i, dim=1)  # 当前 block 归一化因子
                for h_i in T.Parallel(h_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]  # 更新累计因子
                for h_i, j in T.Parallel(h_per_block, group_size):
                    # 用 alpha 衰减所有 4 个输出 tile 的历史值
                    acc_o_tile0[h_i, j] = acc_o_tile0[h_i, j] * alpha[h_i]
                    acc_o_tile1[h_i, j] = acc_o_tile1[h_i, j] * alpha[h_i]
                    acc_o_tile2[h_i, j] = acc_o_tile2[h_i, j] * alpha[h_i]
                    acc_o_tile3[h_i, j] = acc_o_tile3[h_i, j] * alpha[h_i]

                for h_i, bi_i in T.Parallel(h_per_block, BI):
                    # 将 float32 softmax 分数 × fp8_max_val 后 clamp 到 FP8 范围，转 FP8
                    s_fp8_shared[h_i, bi_i] = T.clamp(
                        acc_s[h_i, bi_i] * s_inv_scale_const,
                        -fp8_max_val,
                        fp8_max_val,
                    )
                # S·V GEMM：FP8 分数与 FP8 KV tile 相乘，结果乘 s_scale_const 还原
                T.gemm(s_fp8_shared, kv_tile0, sv_tile, clear_accum=True)
                for h_i, j in T.Parallel(h_per_block, group_size):
                    acc_o_tile0[h_i, j] = (
                        acc_o_tile0[h_i, j] + sv_tile[h_i, j] * s_scale_const  # 缩放还原并累加
                    )

                T.gemm(s_fp8_shared, kv_tile1, sv_tile, clear_accum=True)
                for h_i, j in T.Parallel(h_per_block, group_size):
                    acc_o_tile1[h_i, j] = (
                        acc_o_tile1[h_i, j] + sv_tile[h_i, j] * s_scale_const
                    )

                T.gemm(s_fp8_shared, kv_tile2, sv_tile, clear_accum=True)
                for h_i, j in T.Parallel(h_per_block, group_size):
                    acc_o_tile2[h_i, j] = (
                        acc_o_tile2[h_i, j] + sv_tile[h_i, j] * s_scale_const
                    )

                T.gemm(s_fp8_shared, kv_tile3, sv_tile, clear_accum=True)
                for h_i, j in T.Parallel(h_per_block, group_size):
                    acc_o_tile3[h_i, j] = (
                        acc_o_tile3[h_i, j] + sv_tile[h_i, j] * s_scale_const
                    )

            # 计算归一化因子的倒数（sumexp==0 时 fallback 为 1 避免除零）
            for h_i in T.Parallel(h_per_block):
                denom = T.if_then_else(sumexp[h_i] == 0.0, 1.0, sumexp[h_i])
                inv_denom[h_i] = 1.0 / denom
            # 归一化所有 4 个输出 tile
            for h_i, j in T.Parallel(h_per_block, group_size):
                acc_o_tile0[h_i, j] = acc_o_tile0[h_i, j] * inv_denom[h_i]
                acc_o_tile1[h_i, j] = acc_o_tile1[h_i, j] * inv_denom[h_i]
                acc_o_tile2[h_i, j] = acc_o_tile2[h_i, j] * inv_denom[h_i]
                acc_o_tile3[h_i, j] = acc_o_tile3[h_i, j] * inv_denom[h_i]

            # 计算 log-sum-exp（全掩码时置为极小值）
            for h_i in T.Parallel(h_per_block):
                sumexp[h_i] = T.if_then_else(
                    sumexp[h_i] == 0.0,
                    -(2**30),  # 全掩码时极小值，combine 阶段忽略此 split
                    T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale,
                )

            # 写出 4 个输出 tile 到 partial_o（按 128 列分段写）
            T.copy(
                acc_o_tile0,
                partial_o[b_i, s_i, group_i, H0:H1, 0 * group_size : 1 * group_size],
            )
            T.copy(
                acc_o_tile1,
                partial_o[b_i, s_i, group_i, H0:H1, 1 * group_size : 2 * group_size],
            )
            T.copy(
                acc_o_tile2,
                partial_o[b_i, s_i, group_i, H0:H1, 2 * group_size : 3 * group_size],
            )
            T.copy(
                acc_o_tile3,
                partial_o[b_i, s_i, group_i, H0:H1, 3 * group_size : 4 * group_size],
            )

            T.copy(sumexp, partial_lse[b_i, s_i, group_i, H0:H1])  # 写出 LSE

    return main


def tilelang_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> torch.Tensor:
    # 稀疏 MLA 前向传播的公共入口：根据硬件平台（HIP/NVIDIA）选择对应 kernel 路径
    # HIP（AMD）：使用 partial+combine 两阶段（decode 模式，支持 FP8）
    # NVIDIA：使用 warp 特化 v2 kernel（prefill/通用模式）
    assert q.dim() == 3 and kv.dim() == 3 and indices.dim() == 3
    num_heads = q.shape[1]
    dim = q.shape[2]
    tail_dim = dim - d_v         # rope 维度 = 总维度 - main 维度
    topk = indices.shape[-1]
    assert topk == 2048          # 当前仅支持 top-k=2048

    if _is_hip:
        # AMD/ROCm：根据 KV 数据类型决定使用 FP8 还是 BF16 kernel
        is_fp8_kv = kv.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        if is_fp8_kv:
            # FP8 路径：Q 需要与 KV 同 FP8 类型
            if q.dtype != kv.dtype:
                q = q.to(kv.dtype)
            if _is_gfx95_supported:
                block_I, threads, block_per_cu, cu = 64, 256, 2, 256   # 高端 AMD GPU 配置
            else:
                block_I, threads, block_per_cu, cu = 64, 256, 1, 304   # 低端 AMD GPU 配置
            ni = topk // block_I  # top-k block 数
            # 自动选择最优 inner_iter（避免 CU 利用率不足）
            inner_iter = _pick_inner_iter(q.shape[0], ni, cu, block_per_cu)
            kernel_partial = sparse_mla_fwd_decode_partial_fp8(
                num_heads,
                d_v,
                tail_dim,
                topk,
                sm_scale=sm_scale,
                block_I=block_I,
                inner_iter=inner_iter,
                threads=threads,
            )
        else:
            # BF16 路径
            if _is_gfx95_supported:
                block_I, threads, block_per_cu, cu = 64, 256, 2, 256
            else:
                block_I, threads, block_per_cu, cu = 32, 128, 1, 304
            ni = topk // block_I
            inner_iter = _pick_inner_iter(q.shape[0], ni, cu, block_per_cu)
            kernel_partial = sparse_mla_fwd_decode_partial(
                num_heads,
                d_v,
                tail_dim,
                topk,
                sm_scale=sm_scale,
                block_I=block_I,
                inner_iter=inner_iter,
                threads=threads,
            )
        # 执行 partial kernel（添加 batch=1 维度）
        partial_o_batched, partial_lse_batched = kernel_partial(
            q.unsqueeze(0), kv.unsqueeze(0), indices.unsqueeze(0)
        )
        n_groups = ni // inner_iter  # partial 组数
        # 执行 combine kernel 合并所有 partial 结果
        kernel_combine = sparse_mla_fwd_decode_combine(
            num_heads,
            d_v,
            n_groups * block_I,  # topk（通过 n_groups * block_I 计算）
            head_per_block=4,
            block_I=block_I,
            threads=threads,
        )
        out = kernel_combine(partial_o_batched, partial_lse_batched)
    else:
        # NVIDIA：使用 warp 特化 v2 kernel（不分 partial+combine）
        kernel = sparse_attention_fwd_kernel_v2(
            num_heads, d_v, tail_dim, topk, sm_scale=sm_scale
        )
        out = kernel(q.unsqueeze(0), kv.unsqueeze(0), indices.unsqueeze(0))  # type: ignore
    return out
