# 改编自 vLLM 项目的线性注意力实现
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/linear_attn.py

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 导入 PyTorch
import torch
# 导入 Triton JIT 编译器（GPU Kernel 编译）
import triton
# 导入 Triton 语言原语（向量化内存访问、矩阵乘法等）
import triton.language as tl
# 导入 einops 的 rearrange（张量形状重排工具）
from einops import rearrange


@triton.jit
def _fwd_diag_kernel(
    Q,         # Query 张量（全序列）
    K,         # Key 张量（全序列）
    V,         # Value 张量（全序列）
    Out,       # 输出张量（写入对角块注意力结果）
    S,         # ALiBi 衰减斜率张量，每头一个标量 [h]
    b: tl.constexpr,      # batch 大小
    h: tl.constexpr,      # 头数
    n,                    # 序列长度
    d: tl.constexpr,      # QK 头维度
    e: tl.constexpr,      # V 头维度
    BLOCK: tl.constexpr,  # 大块大小（通常 256）
    NUM_BLOCK,            # 总块数 = ceil(n / BLOCK)
    CBLOCK: tl.constexpr, # 子块大小（通常 32）
):
    # 对角块 Kernel：计算同一 BLOCK 内 Q 与 K 的注意力（因果掩码 + ALiBi 衰减）
    # 每个 program 处理 [off_block*BLOCK+off_cblock*CBLOCK : +CBLOCK) 行的 Q
    off = tl.program_id(0)
    off_bh = off // NUM_BLOCK  # batch-head 索引
    off_block = off % NUM_BLOCK  # 当前序列块索引
    off_cblock = tl.program_id(1)  # 当前子块索引（CBLOCK 粒度）

    off_h = off_bh % h  # 当前头索引（从 batch-head 拆分）

    # 计算当前 batch-head 的基础偏移（以元素为单位）
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e

    # 计算当前大块（BLOCK）的起始偏移
    block_offset = off_block * BLOCK
    qk_block_offset = block_offset * d
    v_block_offset = block_offset * e
    o_block_offset = block_offset * e

    # 计算当前子块（CBLOCK）的起始偏移
    cblock_offset = off_cblock * CBLOCK
    q_cblock_offset = cblock_offset * d
    o_cblock_offset = cblock_offset * e

    # 计算 Q、K_转置、V、O 的 Triton 指针（二维块指针）
    Q_block_ptr = (
        Q
        + qk_offset
        + qk_block_offset
        + q_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * d  # Q 行偏移 [CBLOCK, 1]
        + tl.arange(0, d)[None, :]            # Q 列偏移 [1, d]
    )
    K_trans_block_ptr = (
        K
        + qk_offset
        + qk_block_offset
        + tl.arange(0, CBLOCK)[None, :] * d  # K 列偏移（转置后为行）[1, CBLOCK]
        + tl.arange(0, d)[:, None]            # K 行偏移（转置后为列）[d, 1]
    )
    V_block_ptr = (
        V
        + v_offset
        + v_block_offset
        + tl.arange(0, CBLOCK)[:, None] * e  # V 行偏移 [CBLOCK, 1]
        + tl.arange(0, e)[None, :]            # V 列偏移 [1, e]
    )
    O_block_ptr = (
        Out
        + o_offset
        + o_block_offset
        + o_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * e  # O 行偏移 [CBLOCK, 1]
        + tl.arange(0, e)[None, :]            # O 列偏移 [1, e]
    )

    # 加载当前头的 ALiBi 衰减斜率标量 s
    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    i = off_cblock
    q_index = tl.arange(0, CBLOCK) + i * CBLOCK  # 当前子块 Q 的全局行索引

    # 加载当前子块 Q，超出序列长度的位置填 0
    q = tl.load(Q_block_ptr, mask=block_offset + q_index[:, None] < n, other=0.0).to(
        tl.float32
    )

    # 初始化输出累加器 [CBLOCK, e]（对角块部分结果）
    qkv = tl.zeros([CBLOCK, e], dtype=tl.float32)

    # 遍历当前子块之前的所有 KV 子块（含自身），实现因果注意力
    for j in range(i + 1):
        kv_index = tl.arange(0, CBLOCK) + j * CBLOCK  # 当前 KV 子块的全局行索引
        diff = q_index[:, None] - kv_index[None, :]    # Q 与 K 的相对位置差 [CBLOCK, CBLOCK]
        s_index = s * diff                              # ALiBi 偏置 = slope * 距离
        # 因果掩码：diff < 0 时（未来位置）设为 -inf，否则取负值（距离越远衰减越大）
        s_index = tl.where(diff >= 0, -s_index, float("-inf"))
        decay = tl.exp(s_index)  # 指数衰减矩阵 [CBLOCK, CBLOCK]

        # 加载 K 转置块（KV 子块 j 的 K，维度 [d, CBLOCK]）
        k_trans = tl.load(
            K_trans_block_ptr,
            mask=block_offset + kv_index[None, :] < n,
            other=0.0,
        ).to(tl.float32)
        # 加载 V 块（KV 子块 j 的 V，维度 [CBLOCK, e]）
        v = tl.load(
            V_block_ptr,
            mask=block_offset + kv_index[:, None] < n,
            other=0.0,
        ).to(tl.float32)

        # 计算带衰减的注意力分数 QK^T * decay，维度 [CBLOCK, CBLOCK]
        qk = tl.dot(q, k_trans) * decay

        # 加权值累加：qkv += QK * V，维度 [CBLOCK, e]
        qkv += tl.dot(qk, v)

        # 移动到下一个 KV 子块
        K_trans_block_ptr += CBLOCK * d
        V_block_ptr += CBLOCK * e

    # 将对角块注意力结果写回 Out
    tl.store(
        O_block_ptr,
        qkv.to(O_block_ptr.dtype.element_ty),
        mask=block_offset + q_index[:, None] < n,
    )


@triton.jit
def _fwd_kv_parallel(
    K,          # Key 张量（全序列）
    V,          # Value 张量（全序列）
    K_decay,    # Key 的 ALiBi 衰减因子 [h, BLOCK]（每头每位置的指数衰减）
    KV,         # 输出：每块的 KV 外积累积值 [b*h, NUM_BLOCK, d, e]
    b: tl.constexpr,          # batch 大小
    h: tl.constexpr,          # 头数
    n,                        # 序列长度
    d: tl.constexpr,          # QK 头维度
    e: tl.constexpr,          # V 头维度
    BLOCK: tl.constexpr,      # 大块大小
    NUM_BLOCK,                # 总块数
    D_FBLOCK: tl.constexpr,   # 特征维度子块大小（D 方向分块）
    E_FBLOCK: tl.constexpr,   # 特征维度子块大小（E 方向分块）
    NUM_FBLOCK: tl.constexpr, # 特征分块数量
    CBLOCK: tl.constexpr,     # 子块大小（时序方向）
    NUM_CBLOCK: tl.constexpr, # 子块总数 = BLOCK // CBLOCK
):
    # 并行 KV 外积 Kernel：每个 program 独立计算一个（batch-head, block）对应的 KV 累积外积
    # KV[bh, block] = sum_{t in block} decay(t) * k_t^T * v_t
    off_bh = tl.program_id(0)  # batch-head 索引
    off_block = tl.program_id(1)  # 当前序列块索引

    off_h = off_bh % h  # 当前头索引

    block_offset = off_block * BLOCK  # 当前块的序列起始位置

    # 计算 K、V、KV 在当前块的偏移量
    k_block_offset = block_offset * d
    v_block_offset = block_offset * e
    kv_block_offset = off_block * d * e

    # 计算当前 batch-head 的基础偏移
    k_offset = off_bh * n * d
    v_offset = off_bh * n * e
    kv_offset = off_bh * NUM_BLOCK * d * e

    # 构建 K_转置、V、KV 的 Triton 块指针（D_FBLOCK × CBLOCK 等分块访问）
    K_trans_block_ptr = (
        K
        + k_offset
        + k_block_offset
        + tl.arange(0, CBLOCK)[None, :] * d   # K 行偏移（转置后的列方向）[1, CBLOCK]
        + tl.arange(0, D_FBLOCK)[:, None]      # K 列偏移（转置后的行方向）[D_FBLOCK, 1]
    )
    V_block_ptr = (
        V
        + v_offset
        + v_block_offset
        + tl.arange(0, CBLOCK)[:, None] * e   # V 行偏移 [CBLOCK, 1]
        + tl.arange(0, E_FBLOCK)[None, :]      # V 列偏移 [1, E_FBLOCK]
    )
    KV_block_ptr = (
        KV
        + kv_offset
        + kv_block_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e  # KV 行偏移（K 维方向）[D_FBLOCK, 1]
        + tl.arange(0, E_FBLOCK)[None, :]       # KV 列偏移（V 维方向）[1, E_FBLOCK]
    )

    # 加载当前头的 K 衰减因子指针（k_decay 按块对齐，维度 [h, BLOCK]）
    k_decay_ptr = K_decay + off_h * BLOCK + tl.arange(0, CBLOCK)[None, :]

    kv_index = tl.arange(0, CBLOCK)

    # 初始化 KV 外积累加器 [D_FBLOCK, E_FBLOCK]
    kv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)

    # 处理最后一块可能不满 BLOCK 的情况（边界对齐）
    if off_block == NUM_BLOCK - 1:
        split_n = n - (NUM_BLOCK - 1) * BLOCK  # 最后一块的实际长度
    else:
        split_n = BLOCK
    left_shift = tl.cdiv(split_n, CBLOCK) * CBLOCK - split_n  # 左填充量（用于边界对齐）
    num_blocks = min(tl.cdiv(split_n, CBLOCK), NUM_CBLOCK)    # 实际处理的子块数量
    k_decay_ptr += (NUM_CBLOCK - num_blocks) * CBLOCK          # 对齐到实际起始位置

    # 遍历当前块的所有子块，累积加权 KV 外积
    for j in range(num_blocks):
        left_bound = (1 - j) * left_shift  # 当前子块的有效起始位置
        # 加载 K 转置（考虑边界左移）
        k_trans = tl.load(
            K_trans_block_ptr - left_shift * d,
            mask=kv_index[None, :] >= left_bound,
            other=0.0,
        )
        # 加载 V（考虑边界左移）
        v = tl.load(
            V_block_ptr - left_shift * e,
            mask=kv_index[:, None] >= left_bound,
            other=0.0,
        )

        # 加载当前子块的 K 衰减因子，计算加权 KV 外积并累加
        k_decay = tl.load(k_decay_ptr)
        kv += tl.dot(k_trans * k_decay, v)  # 衰减 K × V = 加权外积 [D_FBLOCK, E_FBLOCK]

        # 移动到下一个子块
        K_trans_block_ptr += CBLOCK * d
        V_block_ptr += CBLOCK * e
        k_decay_ptr += CBLOCK

    # 写入当前块的 KV 外积累积结果
    tl.store(KV_block_ptr, kv.to(KV_block_ptr.dtype.element_ty))


@triton.jit
def _fwd_kv_reduce(
    S,           # ALiBi 衰减斜率 [h]（每头一个标量）
    KV,          # 输入/输出：各块的 KV 外积 [b*h, NUM_BLOCK, d, e]（in-place 更新）
    KV_HISTORY,  # 输入/输出：上一轮末尾的 KV 历史状态 [b*h, d, e]（in-place 更新）
    b: tl.constexpr,          # batch 大小
    h: tl.constexpr,          # 头数
    n,                        # 序列长度
    d: tl.constexpr,          # QK 头维度
    e: tl.constexpr,          # V 头维度
    BLOCK: tl.constexpr,      # 大块大小
    NUM_BLOCK,                # 总块数
    D_FBLOCK: tl.constexpr,   # D 方向特征子块大小
    E_FBLOCK: tl.constexpr,   # E 方向特征子块大小
):
    # KV 跨块 reduce Kernel：以 prefix-scan 方式将历史 KV 状态传播到每块的 KV 外积前缀
    # 公式：KV_pre[i] = decay^block_size * KV_pre[i-1] + KV_cur[i]
    # 同时更新 KV_HISTORY 为最终的累积状态（供 non-diagonal kernel 使用）
    off_bh = tl.program_id(0)  # batch-head 索引
    off_h = off_bh % h  # 当前头索引

    kv_offset = off_bh * NUM_BLOCK * d * e  # 当前 batch-head 的 KV 起始偏移

    # 构建 KV 块指针（D_FBLOCK × E_FBLOCK 子块）
    KV_block_ptr = (
        KV
        + kv_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e  # KV 行偏移 [D_FBLOCK, 1]
        + tl.arange(0, E_FBLOCK)[None, :]       # KV 列偏移 [1, E_FBLOCK]
    )

    # 加载当前头的 ALiBi 衰减斜率 s
    s_ptrs = S + off_h
    s = tl.load(s_ptrs)

    # 构建 KV 历史状态指针（上轮末尾状态 [D_FBLOCK, E_FBLOCK]）
    kv_history_offset = off_bh * d * e
    KV_HISTORY_block_ptr = (
        KV_HISTORY
        + kv_history_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e  # 历史 KV 行偏移
        + tl.arange(0, E_FBLOCK)[None, :]       # 历史 KV 列偏移
    )

    # 加载上一轮的 KV 历史状态（float32 保证累积精度）
    kv_pre = tl.load(KV_HISTORY_block_ptr).to(tl.float32)

    # 按块顺序进行前缀扫描（prefix scan）：将历史状态传播到每块
    for i in range(NUM_BLOCK):
        block_size = min(n - i * BLOCK, BLOCK)  # 当前块的实际有效长度
        # 计算当前块的整体衰减因子（e^{-s * block_size}）
        block_decay = tl.exp(-s.to(tl.float32) * block_size)

        # 加载当前块的 KV 外积（由 _fwd_kv_parallel 计算得到）
        kv_cur = tl.load(KV_block_ptr).to(tl.float32)
        # 将 kv_pre（历史状态）写回到当前块位置（供 non-diagonal kernel 使用）
        tl.store(KV_block_ptr, kv_pre.to(KV_block_ptr.dtype.element_ty))

        # 更新历史状态：kv_pre = decay * kv_pre + kv_cur（递推公式）
        kv_pre = block_decay * kv_pre + kv_cur
        KV_block_ptr += d * e  # 移动到下一块

    # 将最终累积状态写回 KV_HISTORY（供下次调用或后续 decode 使用）
    tl.store(KV_HISTORY_block_ptr, kv_pre)


@triton.jit
def _fwd_none_diag_kernel(
    Q,    # Query 张量（全序列）
    Out,  # 输出张量（in-place 累加：对角块结果 + 非对角块结果）
    S,    # ALiBi 衰减斜率 [h]
    KV,   # 各块的 KV 前缀累积值（由 _fwd_kv_reduce 计算）[b*h, NUM_BLOCK, d, e]
    b: tl.constexpr,          # batch 大小
    h: tl.constexpr,          # 头数
    n,                        # 序列长度
    d: tl.constexpr,          # QK 头维度
    e: tl.constexpr,          # V 头维度
    BLOCK: tl.constexpr,      # 大块大小
    NUM_BLOCK,                # 总块数
    E_FBLOCK: tl.constexpr,   # E 方向特征子块大小
    CBLOCK: tl.constexpr,     # 时序方向子块大小
    NUM_CBLOCK: tl.constexpr, # 子块总数 = BLOCK // CBLOCK
):
    # 非对角块 Kernel：计算 Q 与历史 KV 状态（当前块之前所有 KV 的累积）的注意力
    # 公式：O_ndiag[t] = q_t * KV_prefix[block(t)] * exp(-s * (t mod BLOCK))
    # 注：KV_prefix[block] = sum_{past blocks} decay * k^T v（由 _fwd_kv_reduce 填充）
    off_bh = tl.program_id(0)  # batch-head 索引
    off_h = off_bh % h  # 当前头索引

    off_nc = tl.program_id(1)
    off_n = off_nc // NUM_CBLOCK  # 当前大块索引
    off_c = off_nc % NUM_CBLOCK  # 当前子块索引
    off_e = tl.program_id(2)  # E 方向特征块索引

    n_offset = off_n * BLOCK     # 大块的序列起始位置
    c_offset = off_c * CBLOCK    # 子块的序列偏移
    e_offset = off_e * E_FBLOCK  # E 特征维度起始偏移
    block_offset = n_offset + c_offset  # 当前子块的全局序列起始位置

    # 计算 Q、O、KV 的偏移量
    q_offset = off_bh * n * d + (n_offset + c_offset) * d
    o_offset = off_bh * n * e + (n_offset + c_offset) * e + e_offset
    kv_offset = off_bh * NUM_BLOCK * d * e + off_n * d * e + e_offset

    # 构建 Q、O、KV 的 Triton 块指针
    Q_block_ptr = (
        Q + q_offset + tl.arange(0, CBLOCK)[:, None] * d + tl.arange(0, d)[None, :]
    )
    O_block_ptr = (
        Out
        + o_offset
        + tl.arange(0, CBLOCK)[:, None] * e  # O 行偏移 [CBLOCK, 1]
        + tl.arange(0, E_FBLOCK)[None, :]     # O 列偏移（E 特征分块）[1, E_FBLOCK]
    )
    KV_block_ptr = (
        KV + kv_offset + tl.arange(0, d)[:, None] * e + tl.arange(0, E_FBLOCK)[None, :]
    )

    # 加载当前头的 ALiBi 衰减斜率 s
    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    c_array = tl.arange(0, CBLOCK)

    # 加载当前块（off_n）的 KV 历史前缀（由 _fwd_kv_reduce 写入）[d, E_FBLOCK]
    kv = tl.load(KV_block_ptr).to(tl.float32)
    q_index = block_offset + tl.arange(0, CBLOCK)  # 当前子块的全局行索引

    # 加载当前子块的 Q，超出序列长度填 0 [CBLOCK, d]
    q = tl.load(Q_block_ptr, mask=q_index[:, None] < n, other=0.0).to(tl.float32)

    # 计算块内位置衰减因子：q_decay[t] = exp(-s * t)，t 为 t 在当前块中的相对位置
    q_decay = tl.exp(-s.to(tl.float32) * (off_c * CBLOCK + c_array[:, None]))

    # 非对角块注意力：Q × KV_前缀 × 位置衰减 [CBLOCK, E_FBLOCK]
    qkv_none_diag = tl.dot(q, kv) * q_decay

    # 加载已有的对角块注意力输出（由 _fwd_diag_kernel 写入）[CBLOCK, E_FBLOCK]
    qkv_diag = tl.load(O_block_ptr, mask=q_index[:, None] < n, other=0.0).to(tl.float32)

    # 合并对角块 + 非对角块注意力结果
    qkv = qkv_diag + qkv_none_diag

    # 写回合并后的输出
    tl.store(
        O_block_ptr, qkv.to(O_block_ptr.dtype.element_ty), mask=q_index[:, None] < n
    )


class _attention(torch.autograd.Function):
    # Lightning Attention 自动求导函数封装（仅实现前向，无反向传播）

    @staticmethod
    def forward(ctx, q, k, v, s, kv_history):
        # Forward pass of the lightning attention algorithm
        # 分三阶段计算线性注意力：对角块 → KV 前缀扫描 → 非对角块
        q = q.contiguous()  # 确保内存连续（Triton Kernel 要求）
        k = k.contiguous()
        v = v.contiguous()
        s = s.contiguous()

        # 检查 CUDA 计算能力，Lightning Attention 要求 SM80+（Ampere）
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            raise RuntimeError(
                "Flash attention currently only supported",
                "for compute capability >= 80",
            )

        # 获取输入维度 [batch, heads, seq_len, head_dim]
        b, h, n, d = q.shape
        e = v.shape[-1]

        # 初始化输出张量 [b, h, n, e]
        o = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)

        # 设置大块和子块大小（对角块阶段）
        BLOCK = 256
        NUM_BLOCK = triton.cdiv(n, BLOCK)

        CBLOCK = 32
        NUM_CBLOCK = BLOCK // CBLOCK
        assert BLOCK % CBLOCK == 0, "BLOCK must be a multiple of CBLOCK"

        # 预计算 K 的衰减因子 k_decay[h, BLOCK] = exp(-s * (BLOCK - position))
        # 使得 KV 外积按位置距离加权衰减
        array = torch.arange(0, BLOCK, device=q.device) + 1
        k_decay = torch.exp(-s * (BLOCK - array.reshape(1, -1)))

        # 阶段 1：计算所有对角块（每块内 Q 与 K 的因果注意力）
        grid = (b * h * NUM_BLOCK, NUM_CBLOCK)
        _fwd_diag_kernel[grid](
            q,
            k,
            v,
            o,
            s,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            CBLOCK=CBLOCK,
        )

        # 设置特征维度分块参数（KV 并行和非对角块阶段使用更大 CBLOCK=64）
        NUM_FBLOCK = 1
        D_FBLOCK = d // NUM_FBLOCK
        assert d % NUM_FBLOCK == 0
        E_FBLOCK = e // NUM_FBLOCK
        assert e % NUM_FBLOCK == 0

        CBLOCK = 64
        NUM_CBLOCK = BLOCK // CBLOCK
        assert BLOCK % CBLOCK == 0, "BLOCK must be a multiple of CBLOCK"

        # 阶段 2：并行计算每块的 KV 外积（_fwd_kv_parallel）
        kv = torch.empty((b, h, NUM_BLOCK, d, e), dtype=torch.float32, device=q.device)
        grid = (b * h, NUM_BLOCK)
        _fwd_kv_parallel[grid](
            k,
            v,
            k_decay,
            kv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            D_FBLOCK=D_FBLOCK,
            E_FBLOCK=E_FBLOCK,
            NUM_FBLOCK=NUM_FBLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
        )

        # 阶段 3：前缀扫描（prefix scan），传播历史 KV 状态到每块并更新 kv_history
        grid = (b * h, NUM_FBLOCK)
        _fwd_kv_reduce[grid](
            s,
            kv,
            kv_history,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            D_FBLOCK=D_FBLOCK,
            E_FBLOCK=E_FBLOCK,
        )

        # 阶段 4：计算非对角块（Q 与历史 KV 前缀的注意力）并叠加到输出
        grid = (b * h, NUM_BLOCK * NUM_CBLOCK)
        _fwd_none_diag_kernel[grid](
            q,
            o,
            s,
            kv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            E_FBLOCK=E_FBLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
        )

        # 保存中间张量供反向传播（若需要）
        ctx.save_for_backward(q, k, v, s, kv)
        ctx.BLOCK = BLOCK

        # 返回输出和更新后的 KV 历史（最后一块 + kv_history 拼接）
        return o, torch.cat([kv, kv_history.unsqueeze(2)], dim=2)


# 将 _attention.apply 封装为可直接调用的函数（类 Flash Attention 接口）
lightning_attention_ = _attention.apply


def lightning_attention(q, k, v, ed, block_size=256, kv_history=None):
    """
    Apply lightning attention algorithm
    to compute attention efficiently.

    Args:
        q: Query tensor of shape [batch, heads, seq_len, dim]
        k: Key tensor of shape [batch, heads, seq_len, dim]
        v: Value tensor of shape [batch, heads, seq_len, dim_v]
        ed: Decay rate tensor of shape [heads]
        block_size: Size of blocks for block-sparse attention
        kv_history: Optional key-value history from previous computations

    Returns:
        output: Attention output
        kv: Updated key-value history
    """
    # Lightning Attention Python 包装函数：将长序列分块处理，累积注意力输出
    d = q.shape[-1]
    e = v.shape[-1]

    # ed 可能是 1D 标量列表，reshape 为 [1, h, 1, 1] 以广播到 [b, h, n, d]
    if ed.dim() == 1:
        ed = ed.view(1, -1, 1, 1)

    # 将 Q 头维度按 m 分块，避免单次 Triton Kernel 处理过大的矩阵
    m = 128 if d >= 128 else 64
    assert d % m == 0, f"Dimension d ({d}) must be divisible by m ({m})"
    arr = [m * i for i in range(d // m + 1)]
    if arr[-1] != d:
        arr.append(d)
    n = len(arr)
    output = 0  # 累积输出（各子块结果相加）

    # 初始化或克隆 KV 历史状态（首次调用时创建全零状态）
    if kv_history is None:
        kv_history = torch.zeros(
            (q.shape[0], q.shape[1], d, e), dtype=torch.float32, device=q.device
        )
    else:
        kv_history = kv_history.clone().contiguous()  # 克隆避免修改原始状态

    # 遍历 Q 的每个维度子块，调用 lightning_attention_ 并累积输出
    for i in range(n - 1):
        s = arr[i]
        e = arr[i + 1]
        q1 = q[..., s:e]  # 当前子块的 Q 切片 [b, h, n, m]
        k1 = k[..., s:e]  # 当前子块的 K 切片 [b, h, n, m]
        o, kv = lightning_attention_(q1, k1, v, ed, kv_history)
        output = output + o  # 线性累加（线性注意力的可加性）
    return output, kv


@triton.jit
def _linear_attn_decode_kernel(
    q_ptr,          # Query 指针 [B, H, 1, D]
    k_ptr,          # Key 指针 [B, H, 1, D]
    v_ptr,          # Value 指针 [B, H, 1, D]
    kv_cache_ptr,   # KV 缓存状态指针 [pool, H, D, D]（线性注意力的 SSM 状态矩阵）
    slope_rate,     # ALiBi 衰减斜率 [H]（每头一个标量）
    slot_idx,       # 每个请求对应的缓存槽位索引 [B]
    output_ptr,     # 输出指针 [B, H, 1, D]
    D: tl.constexpr,           # 头维度
    qkv_b_stride,              # QKV batch 维度步幅
    qkv_h_stride,              # QKV head 维度步幅
    cache_b_stride,            # KV 缓存 batch 维度步幅（槽位步幅）
    cache_h_stride,            # KV 缓存 head 维度步幅
    cache_d0_stride,           # KV 缓存第一维（K 维）步幅
    cache_d1_stride,           # KV 缓存第二维（V 维）步幅
    BLOCK_SIZE: tl.constexpr,  # 每次处理的 V 维度块大小（用于并行化 V 维）
):
    """
    Kernel for linear attention decoding with KV cache.

    This kernel computes attention for a single token using the KV cache.
    """
    pid_b = tl.program_id(0)  # batch 索引（当前请求）
    pid_h = tl.program_id(1)  # head 索引
    pid_d = tl.program_id(2)  # V 维度块索引

    # 从 slot_idx 加载当前请求对应的 KV 缓存槽位
    slot_id = tl.load(slot_idx + pid_b)

    # slot_id == -1 表示 padding 请求，直接跳过
    if slot_id == -1:
        return

    batch_id = pid_b
    head_id = pid_h

    # 加载当前头的 ALiBi 衰减斜率标量 ratio
    ratio = tl.load(slope_rate + pid_h)

    # 计算 Q/K 和 V 的维度偏移（D 维和 V 维块偏移）
    qk_d_offsets = tl.arange(0, D)
    v_d_offsets = tl.arange(0, BLOCK_SIZE) + pid_d * BLOCK_SIZE  # V 维按块并行
    cache_d_offsets = (
        qk_d_offsets[:, None] * cache_d0_stride + v_d_offsets[None, :] * cache_d1_stride
    )  # 缓存矩阵的二维偏移 [D, BLOCK_SIZE]

    # 计算 Q、K、V 的 batch-head 基础偏移（步幅计算）
    q_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride
    k_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride
    v_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride

    # 计算 KV 缓存的 slot-head 基础偏移
    cache_offset = slot_id * cache_b_stride + head_id * cache_h_stride

    # 构建加载掩码（避免越界访问）
    qk_mask = qk_d_offsets < D
    v_mask = v_d_offsets < D

    # 加载 Q、K、V 向量（单 token decode，形状退化为 1D 向量）
    q = tl.load(q_ptr + q_offset + qk_d_offsets, mask=qk_mask, other=0.0)
    k = tl.load(k_ptr + k_offset + qk_d_offsets, mask=qk_mask, other=0.0)
    v = tl.load(v_ptr + v_offset + v_d_offsets, mask=v_mask, other=0.0)

    # 计算当前 token 的 KV 外积（k^T * v，维度 [D, BLOCK_SIZE]）
    kv_outer = k[:, None] * v[None, :]
    kv_mask = qk_mask[:, None] & v_mask[None, :]

    # 将衰减斜率转换为衰减因子 e^{-ratio}（ALiBi 的指数衰减）
    ratio = tl.exp(-ratio)
    kv_ptr = kv_cache_ptr + cache_offset + cache_d_offsets
    kv_cache_old = tl.load(kv_ptr, mask=kv_mask, other=0.0)  # 加载历史 KV 状态
    # 更新 KV 状态：S_t = decay * S_{t-1} + k_t^T * v_t（线性注意力递推公式）
    kv_outer = kv_outer + ratio * kv_cache_old

    # 计算线性注意力输出：o = sum_d(q_d * S_t[d, :])，即 q^T @ S_t
    output = q[:, None].to(tl.float32) * kv_outer
    output = tl.sum(output, axis=0)  # 对 K 维求和，得到 [BLOCK_SIZE] 输出

    # 写回更新后的 KV 缓存状态（in-place 更新）
    tl.store(kv_ptr, kv_outer, mask=kv_mask)
    # 写回 attention 输出
    tl.store(output_ptr + q_offset + v_d_offsets, output, mask=v_mask)


def linear_decode_forward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_caches: torch.Tensor,
    slope_rate: torch.Tensor,
    slot_idx: torch.Tensor,
    BLOCK_SIZE: int = 32,
) -> torch.Tensor:
    """
    Perform linear attention decoding using Triton kernels.

    Args:
        q: Query tensor of shape [B, H, 1, D]
        k: Key tensor of shape [B, H, 1, D]
        v: Value tensor of shape [B, H, 1, D]
        kv_caches: Key-value cache tensor
        slope_rate: Decay rate tensor
        slot_idx: Slot indices for batches
        BLOCK_SIZE: Size of blocks for processing

    Returns:
        output: Attention output tensor
    """
    # 线性注意力 decode 的 Python 包装函数：调度 Triton Kernel 并做张量预处理
    B, H, _, D = q.shape
    assert k.shape == (B, H, 1, D)
    assert v.shape == (B, H, 1, D)

    # 预分配输出张量（与 q 形状相同 [B, H, 1, D]）
    output = torch.empty_like(q)

    # 网格维度：(batch, head, D // BLOCK_SIZE)，V 维按块并行
    grid = (B, H, D // BLOCK_SIZE)

    # 计算各张量步幅（供 Triton Kernel 使用）
    qkv_b_stride = q.stride(0)
    qkv_h_stride = q.stride(1)

    cache_b_stride = kv_caches.stride(0)  # KV 缓存 slot 维度步幅
    cache_h_stride = kv_caches.stride(1)  # KV 缓存 head 维度步幅
    cache_d0_stride = kv_caches.stride(2) # KV 缓存 K 维步幅
    cache_d1_stride = kv_caches.stride(3) # KV 缓存 V 维步幅

    # 启动 Triton Kernel（并行 decode 单步线性注意力更新）
    _linear_attn_decode_kernel[grid](
        q,
        k,
        v,
        kv_caches,
        slope_rate,
        slot_idx,
        output,
        D,
        qkv_b_stride,
        qkv_h_stride,
        cache_b_stride,
        cache_h_stride,
        cache_d0_stride,
        cache_d1_stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # 将输出从 [B, H, 1, D] reshape 为 [B, H*D]（合并头维度）
    output = rearrange(output, "b h n d -> b n (h d)")
    return output.squeeze(1).contiguous()  # squeeze 掉 seq_len=1 维度，返回 [B, H*D]


class BailingLinearKernel:
    """
    Linear attention kernel implementation for Bailing models.

    This class is adapted from MiniMaxText01LinearKernel in vllm:
    https://github.com/vllm-project/vllm/blob/a9138e85b14047e06300685b48e3485b995425fb/vllm/model_executor/models/minimax_text_01.py#L289

    The implementation maintains the same functionality while being renamed to
    match our Bailing model naming convention.
    """
    # Bailing 线性注意力 Kernel 封装：对应 vLLM 中的 MiniMaxText01LinearKernel
    # 主要方法：jit_linear_forward_prefix（含历史状态的 prefill 前向）

    @staticmethod
    def jit_linear_forward_prefix(
        q: torch.Tensor,          # Query 张量 [h, seq, d]（heads 优先格式）
        k: torch.Tensor,          # Key 张量 [h, seq, d]
        v: torch.Tensor,          # Value 张量 [h, seq, d]
        kv_caches: torch.Tensor,  # KV 缓存状态（单请求的 SSM 状态 [h, d, e]）
        slope_rate: torch.Tensor, # ALiBi 衰减斜率 [h, 1, 1]
        block_size: int,          # 分块大小（lightning_attention 的 block_size 参数）
        layer_idx: int = None,    # 层索引（调试用，无实际计算意义）
        **kwargs,
    ) -> torch.Tensor:
        # 将斜率转换为 float32 保证精度
        slope_rate = slope_rate.to(torch.float32)
        should_pad_dim = q.dim() == 3  # 如果输入是 3D（无 batch 维度），需要添加 batch=1
        if should_pad_dim:
            q = q.unsqueeze(0)  # [h, seq, d] -> [1, h, seq, d]
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
        b, h, n, d = q.shape
        e = d
        # 将 KV 缓存 reshape 为 [1, h, d, e]（batch=1 的 KV 历史状态）
        kv_history = kv_caches.reshape(1, h, d, e).contiguous()
        # 调用 Lightning Attention 前向（使用已有历史状态进行有状态推理）
        output, kv_history = lightning_attention(
            q, k, v, slope_rate, block_size=block_size, kv_history=kv_history
        )
        # 将更新后的 KV 历史写回 kv_caches（in-place 更新，供后续 decode 使用）
        kv_caches.copy_(kv_history[:, :, -1, :, :].reshape(h, d, e))
        assert output.shape[0] == 1, "batch size must be 1"
        # 将输出从 [1, h, n, d] reshape 为 [n, h*d]（标准注意力输出格式）
        return output.squeeze(0).transpose(0, 1).reshape([n, h * d]).contiguous()
