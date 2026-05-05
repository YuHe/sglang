import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_cuda

# FlashMLA 创建 KV 块时使用的块大小（token 数量）
_FLASHMLA_CREATE_KV_BLOCK_SIZE = 4096
# 将块大小封装为 Triton 编译期常量，用于 jit kernel 内部引用
FLASHMLA_CREATE_KV_BLOCK_SIZE_TRITON = tl.constexpr(_FLASHMLA_CREATE_KV_BLOCK_SIZE)

# 检测当前运行环境是否为 CUDA，用于条件导入
_is_cuda = is_cuda()

if _is_cuda:
    # 仅在 CUDA 环境下导入融合的 MLA Q 拼接算子
    from sgl_kernel import concat_mla_absorb_q


@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]  每个请求对应的 token 索引表
    req_pool_indices_ptr,  # 请求在池中的索引
    page_kernel_lens_ptr,  # 每个请求的 KV 长度
    kv_indptr,  # KV 索引的起始偏移数组
    kv_start_idx,  # 可选：每个请求的 KV 起始位置（用于增量解码）
    kv_indices_ptr,  # 输出：KV 索引数组
    req_to_token_ptr_stride: tl.constexpr,  # req_to_token_ptr 的行步长
):
    # 每个程序实例处理一个 batch 中的单个请求，块大小为 512
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)  # 当前程序处理的请求索引

    # find the req pool idx, this is for batch to token
    # 获取当前请求在请求池中的实际索引
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    # 获取当前请求的 KV 索引写入偏移量
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        # 如果提供了起始位置，则从该位置开始读取（支持增量/Chunked Prefill）
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    # kv_end 等于起始位置加上本次需要处理的 KV 长度
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    # 计算循环次数，每次处理 BLOCK_SIZE 个 token
    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for i in range(num_loop):
        # index into req_to_token_ptr needs to be int64
        # 计算当前块内每个 token 相对于 kv_start 的偏移，需要 int64 防止溢出
        offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
        # 只处理实际有效的 token，超出范围的用 mask 屏蔽
        mask = offset < kv_end - kv_start
        # 从 req_to_token_ptr 读取全局 token 索引
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + offset,
            mask=mask,
        )
        # 将读取到的 token 索引写入 kv_indices_ptr
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)


def get_num_page_per_block_flashmla(page_size: int = 64) -> int:
    # 计算每个大块（_FLASHMLA_CREATE_KV_BLOCK_SIZE token）包含多少个分页
    num_page_per_block = _FLASHMLA_CREATE_KV_BLOCK_SIZE // page_size
    return num_page_per_block


@triton.jit
def create_flashmla_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]  请求到 token 的映射表
    req_pool_indices_ptr,  # 请求在池中的索引
    page_kernel_lens_ptr,  # 每个请求的 KV 长度（token 数）
    kv_start_idx,  # 可选：增量解码时的起始 token 位置
    kv_indices_ptr,  # 输出：按页索引写入的目标数组
    req_to_token_ptr_stride: tl.constexpr,  # req_to_token_ptr 的行步长
    kv_indices_ptr_stride: tl.constexpr,  # kv_indices_ptr 的行步长（每个请求）
    PAGED_SIZE: tl.constexpr = 64,  # FlashMLA 的分页大小（默认 64 token/页）
):
    # 每个大块包含的分页数，编译期计算
    NUM_PAGE_PER_BLOCK: tl.constexpr = (
        FLASHMLA_CREATE_KV_BLOCK_SIZE_TRITON // PAGED_SIZE
    )
    pid = tl.program_id(axis=0)  # 当前处理的请求编号

    # find the req pool idx, this is for batch to token
    # 获取当前请求在请求池中的实际索引
    req_pool_index = tl.load(req_pool_indices_ptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        # 支持增量 prefill：从指定位置开始
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start

    # kv_end 为本请求需要处理的最后 token 位置
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    # 总页数（向上取整）
    num_paged = tl.cdiv(kv_end - kv_start, PAGED_SIZE)
    # 大块循环次数（每大块包含 FLASHMLA_CREATE_KV_BLOCK_SIZE 个 token）
    num_pages_loop = tl.cdiv(kv_end - kv_start, FLASHMLA_CREATE_KV_BLOCK_SIZE_TRITON)

    for i in range(num_pages_loop):
        # index into req_to_token_ptr needs to be int64
        # 计算当前大块内每页的起始 token 偏移（步长为 PAGED_SIZE），需要 int64
        paged_offset = (
            tl.arange(0, NUM_PAGE_PER_BLOCK).to(tl.int64) + i * NUM_PAGE_PER_BLOCK
        ) * PAGED_SIZE
        # 输出数组中当前大块的页索引偏移
        paged_offset_out = tl.arange(0, NUM_PAGE_PER_BLOCK) + i * NUM_PAGE_PER_BLOCK

        # 有效数据掩码（按 token 级和页级分别控制）
        mask = paged_offset < num_paged * PAGED_SIZE
        mask_out = paged_offset_out < num_paged

        # 读取每页首个 token 的全局索引
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + paged_offset,
            mask=mask,
        )
        # 将 token 索引转换为页索引（整除 PAGED_SIZE）后写入输出
        tl.store(
            kv_indices_ptr + pid * kv_indices_ptr_stride + paged_offset_out,
            data // PAGED_SIZE,
            mask=mask_out,
        )


@triton.jit
def concat_and_cast_mha_k_kernel(
    k_ptr,       # 输出 K 张量指针，形状 [num_tokens, num_heads, nope_dim+rope_dim]
    k_nope_ptr,  # 输入：K 的 nope（非位置编码）部分，形状 [num_tokens, num_heads, nope_dim]
    k_rope_ptr,  # 输入：K 的 rope（位置编码）部分，形状 [num_tokens, 1, rope_dim]（共享）
    head_cnt: tl.constexpr,   # 注意力头数
    k_stride0: tl.constexpr,  # k_ptr 的 token 维步长
    k_stride1: tl.constexpr,  # k_ptr 的 head 维步长
    nope_stride0: tl.constexpr,  # k_nope_ptr 的 token 维步长
    nope_stride1: tl.constexpr,  # k_nope_ptr 的 head 维步长
    rope_stride0: tl.constexpr,  # k_rope_ptr 的 token 维步长（head=1，共享）
    nope_dim: tl.constexpr,   # nope 部分的特征维度
    rope_dim: tl.constexpr,   # rope 部分的特征维度
):
    # 每个程序处理一个 token（pid_loc 为 token 索引）
    pid_loc = tl.program_id(0)
    # 所有头的索引范围 [0, head_cnt)
    head_range = tl.arange(0, head_cnt)

    # 计算输出 k 中当前 token 所有头的基地址
    k_head_ptr = k_ptr + pid_loc * k_stride0 + head_range[:, None] * k_stride1

    # 拷贝 nope 部分：从 k_nope_ptr 读取并写入 k_ptr 的前 nope_dim 维
    nope_offs = tl.arange(0, nope_dim)

    src_nope_ptr = (
        k_nope_ptr
        + pid_loc * nope_stride0
        + head_range[:, None] * nope_stride1
        + nope_offs[None, :]
    )
    dst_nope_ptr = k_head_ptr + nope_offs[None, :]

    src_nope = tl.load(src_nope_ptr)
    tl.store(dst_nope_ptr, src_nope)

    # 拷贝 rope 部分：从 k_rope_ptr 读取（只有 1 个 head，广播到所有头）并写入 k_ptr 的后 rope_dim 维
    rope_offs = tl.arange(0, rope_dim)
    src_rope_ptr = k_rope_ptr + pid_loc * rope_stride0 + rope_offs[None, :]
    dst_rope_ptr = k_head_ptr + nope_dim + rope_offs[None, :]
    src_rope = tl.load(src_rope_ptr)
    tl.store(dst_rope_ptr, src_rope)


def concat_and_cast_mha_k_triton(
    k: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
):
    # The source data type will be implicitly converted to the target data type.
    # 校验三个张量均为 3D，维度不符时抛出详细错误信息
    assert (
        len(k.shape) == 3 and len(k_nope.shape) == 3 and len(k_rope.shape) == 3
    ), f"shape should be 3d, but got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"
    # 校验 token 维度一致
    assert (
        k.shape[0] == k_nope.shape[0] and k.shape[0] == k_rope.shape[0]
    ), f"invalid shape, got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"
    # k 和 k_nope 的 head 数相同；k_rope 的 head 数为 1（共享 rope）
    assert (
        k.shape[1] == k_nope.shape[1] and 1 == k_rope.shape[1]
    ), f"invalid shape, got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"
    # 输出 k 的最后维度 = nope_dim + rope_dim
    assert (
        k.shape[-1] == k_nope.shape[-1] + k_rope.shape[-1]
    ), f"invalid shape, got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"

    # 提取 nope 和 rope 的特征维度
    nope_dim = k_nope.shape[-1]
    rope_dim = k_rope.shape[-1]
    # 每个 token 对应一个程序实例
    grid = (k.shape[0],)

    # 启动 Triton kernel，将 k_nope 和 k_rope 拼接写入 k
    concat_and_cast_mha_k_kernel[grid](
        k,
        k_nope,
        k_rope,
        k.shape[1],
        k.stride(0),
        k.stride(1),
        k_nope.stride(0),
        k_nope.stride(1),
        k_rope.stride(0),
        nope_dim,
        rope_dim,
    )


@triton.jit
def pad_sequence_with_mask_kernel(
    input_ptr,  # (total_tokens, hidden)  打平后的所有序列 token 嵌入
    offsets_ptr,  # (B,)  每条序列在 input 中的起始偏移
    lengths_ptr,  # (B,)  每条序列的实际长度
    output_ptr,  # (B, max_len, hidden)  填充后的输出张量
    mask_ptr,  # (B, max_len)  注意力掩码（True=有效 token）
    max_len,    # 批次中最长序列的长度
    hidden_dim,  # 嵌入隐藏维度
    BLOCK_M: tl.constexpr,  # seq block  序列方向的分块大小
    BLOCK_D: tl.constexpr,  # hidden block  隐藏维度方向的分块大小
):
    b = tl.program_id(0)  # batch index  当前处理的 batch 序号
    m = tl.program_id(1)  # seq block index  当前处理的序列块编号

    # 加载当前序列的起始偏移和实际长度
    offset = tl.load(offsets_ptr + b)
    length = tl.load(lengths_ptr + b)

    # 当前块内的序列位置范围
    seq_ids = m * BLOCK_M + tl.arange(0, BLOCK_M)
    hid_ids = tl.arange(0, BLOCK_D)

    # seq_mask: 序列位置不超过 max_len；valid_token: 位置在真实长度内
    seq_mask = seq_ids < max_len
    valid_token = seq_ids < length

    # input index
    # 从打平的 input 中计算有效 token 的地址
    in_token = offset + seq_ids
    in_ptr = input_ptr + in_token[:, None] * hidden_dim + hid_ids[None, :]

    # output index
    # 计算目标 (B, max_len, hidden) 中的目标地址
    out_ptr = (
        output_ptr
        + b * max_len * hidden_dim
        + seq_ids[:, None] * hidden_dim
        + hid_ids[None, :]
    )

    # 对有效 token 加载嵌入，无效位置填 0（padding）
    values = tl.load(
        in_ptr,
        mask=valid_token[:, None] & (hid_ids[None, :] < hidden_dim),
        other=0.0,
    )

    # 将嵌入写入填充后的输出张量
    tl.store(
        out_ptr,
        values,
        mask=seq_mask[:, None] & (hid_ids[None, :] < hidden_dim),
    )

    # attention mask
    # 仅由第 0 个 hidden 块来写注意力掩码，避免重复写入
    if tl.program_id(2) == 0:
        mask_out_ptr = mask_ptr + b * max_len + seq_ids
        tl.store(mask_out_ptr, valid_token, mask=seq_mask)


def pad_sequence_with_mask(
    input_emb,  # (total_tokens, hidden)  打平的嵌入输入
    offsets,  # (B,)  每条序列在 input_emb 中的起始索引
    lengths,  # (B,)  每条序列的真实长度
    max_len,  # 本批次最大序列长度
):
    # 获取批大小和嵌入维度
    B = offsets.shape[0]
    hidden_dim = input_emb.shape[1]

    # 分配填充后的输出张量，初始化为 0
    output = torch.zeros(
        (B, max_len, hidden_dim),
        device=input_emb.device,
        dtype=input_emb.dtype,
    )
    # 分配注意力掩码张量（展平为一维，方便 Triton kernel 写入）
    attn_mask = torch.empty(
        (B * max_len),
        device=input_emb.device,
        dtype=torch.bool,
    )

    # 将块大小向上取到 2 的幂次，保证 Triton 向量化效率
    BLOCK_D = triton.next_power_of_2(hidden_dim)
    BLOCK_M = triton.next_power_of_2(max_len)

    # grid: (批大小, 序列块数, 1)
    grid = (
        B,
        triton.cdiv(max_len, BLOCK_M),
        1,
    )

    # 启动 Triton kernel 执行序列填充与掩码生成
    pad_sequence_with_mask_kernel[grid](
        input_emb,
        offsets,
        lengths,
        output,
        attn_mask,
        max_len,
        hidden_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
    )

    # 返回批大小、填充后嵌入及注意力掩码
    return B, output, attn_mask


@triton.jit
def seqlens_expand_kernel(
    extend_seq_lens_ptr,  # [N]  每个请求本次新增的 token 数（qo_len）
    seq_lens_ptr,  # [N]  每个请求目前已有的总 KV 长度
    offsets_ptr,  # [N+1]  每个请求输出的起始偏移（cumsum 前缀和）
    output_ptr,  # [sum(extend_seq_lens)]  输出：每个 query token 对应的 KV 序列长度
    N,  # 请求数量
    BLOCK: tl.constexpr,  # 每次处理的最大 qo_len（2 的幂次）
):
    pid = tl.program_id(0)  # 当前处理的请求编号

    # 超出范围的程序直接退出
    if pid >= N:
        return

    # qo_len：本次新增的 token 数；kv_len：目前的总 KV 长度
    qo_len = tl.load(extend_seq_lens_ptr + pid)
    kv_len = tl.load(seq_lens_ptr + pid)

    # 第一个新增 token 对应的 KV 长度（从 kv_len - qo_len + 1 开始递增）
    start = kv_len - qo_len + 1
    # 当前请求的输出起始偏移
    out_offset = tl.load(offsets_ptr + pid)

    offs = tl.arange(0, BLOCK)
    # 只写入 qo_len 个有效位置
    mask = offs < qo_len

    # 依次生成 [start, start+1, ..., start+qo_len-1]
    values = start + offs
    tl.store(output_ptr + out_offset + offs, values, mask=mask)


def seqlens_expand_triton(
    extend_seq_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    total_len: int,
    max_q_len: int,
):
    """
    extend_seq_lens: [N], int32, CUDA  每个请求新增的 token 数
    seq_lens:        [N], int32, CUDA  每个请求当前的总 KV 长度
    """
    # 确保输入张量在 CUDA 设备上
    assert extend_seq_lens.is_cuda
    assert seq_lens.is_cuda

    N = extend_seq_lens.numel()

    # 构建前缀和偏移数组：offsets[i] 为第 i 个请求的输出起始位置
    offsets = torch.zeros(N + 1, device=extend_seq_lens.device, dtype=torch.int32)
    offsets[1:] = torch.cumsum(extend_seq_lens, dim=0)
    # 分配输出张量，存储每个 query token 对应的 KV 序列长度
    output = torch.empty(total_len, device=extend_seq_lens.device, dtype=torch.int32)

    # 块大小向上对齐到 2 的幂，每个请求一个程序实例
    BLOCK = triton.next_power_of_2(max_q_len)
    grid = (N,)

    # 启动 Triton kernel 展开序列长度
    seqlens_expand_kernel[grid](
        extend_seq_lens,
        seq_lens,
        offsets,
        output,
        N,
        BLOCK=BLOCK,
    )

    return output


# 当 num_kv_heads=1 时，张量会出现退化步长（degenerate strides），例如：
# - shape: [num_pages, 1, 64, 128]
# - stride: [8192, 128, 128, 1]
# 这会导致 flashinfer（trtllm-mha backend）的 TMA 描述符验证失败。
#
# See: https://github.com/flashinfer-ai/flashinfer/issues/2232
def canonicalize_stride(tensor: torch.Tensor) -> torch.Tensor:
    """
    Adjust degenerate strides for a tensor, make it canonical.
    修正退化步长，使张量步长与形状严格对应（标准化）。
    """
    sizes = tensor.size()
    strides = tensor.stride()
    ndim = tensor.dim()

    # 检查是否存在退化步长：某维度大小为 1 且步长与下一维相同
    need_fix = any(
        sizes[i] == 1 and strides[i] == strides[i + 1] for i in range(ndim - 1)
    )

    if not need_fix:
        return tensor

    # canonicalize the stride
    # Example:
    # - shape: [num_pages, 1, 64, 128]
    # - stride: [8192, 128, 128, 1] (wrong!)
    # Gives new stride: [8192, 8192, 128 ,1] (correct!)
    # 从最后一维开始向前重新计算标准步长（乘积累积）
    new_strides = [0] * ndim
    new_strides[-1] = 1
    for i in range(ndim - 2, -1, -1):
        new_strides[i] = new_strides[i + 1] * sizes[i + 1]

    # 使用 as_strided 应用新步长（不拷贝数据）
    return tensor.as_strided(sizes, new_strides)


def mla_quantize_and_rope_for_fp8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    pos_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import flashinfer.rope

    """Quantize and apply RoPE for FP8 attention path.

        This function handles the FP8 quantization and RoPE application for MLA attention.
        It takes separate query/key nope and rope components, applies RoPE to the rope parts,
        quantizes all components to FP8, and merges the query components into a single tensor.
        本函数用于 MLA 注意力路径的 FP8 量化和旋转位置编码（RoPE）应用。

        Args:
            q_nope: Query no-position-encoding component [seq_len, num_heads, kv_lora_rank]
                - expected dtype: torch.bfloat16
            q_rope: Query RoPE component [seq_len, num_heads, qk_rope_head_dim]
                - expected dtype: torch.bfloat16
            k_nope: Key no-position-encoding component [seq_len, num_heads, kv_lora_rank]
                - expected dtype: torch.bfloat16
            k_rope: Key RoPE component [seq_len, num_heads, qk_rope_head_dim]
                - expected dtype: torch.bfloat16
            pos_ids: Position indices for each token
                - expected dtype: torch.int64 or torch.int32
            cos_sin_cache: Precomputed cosine/sine cache for RoPE
                - expected dtype: matches q_/k_ input dtype (torch.bfloat16)
            is_neox: Whether to use NeoX-style RoPE (interleaved) or GPT-style (half rotation)
            kv_lora_rank: Dimension of the no-position-encoding component
            qk_rope_head_dim: Dimension of the RoPE component

        Returns:
            tuple: (merged_q_out, k_nope_out, k_rope_out) quantized to FP8
                - merged_q_out: [seq_len, num_heads, kv_lora_rank + qk_rope_head_dim], dtype=torch.float8_e4m3fn
                - k_nope_out:   [seq_len, num_heads, kv_lora_rank], dtype=torch.float8_e4m3fn
                - k_rope_out:   [seq_len, num_heads, qk_rope_head_dim], dtype=torch.float8_e4m3fn
        """
    # 目标量化数据类型：FP8 E4M3 格式
    attn_dtype = torch.float8_e4m3fn
    q_len, num_heads = q_rope.shape[0], q_rope.shape[1]

    # Allocate output tensors with FP8 dtype
    # Query output will contain merged nope + rope components
    # 分配合并后的 query 输出（nope + rope 拼接），数据类型为 FP8
    q_out = q_rope.new_empty(
        q_len,
        num_heads,
        kv_lora_rank + qk_rope_head_dim,
        dtype=attn_dtype,
    )

    # Key outputs maintain original shapes but with FP8 dtype
    # key rope 和 nope 分别量化为 FP8，形状不变
    k_rope_out = k_rope.new_empty(k_rope.shape, dtype=attn_dtype)
    k_nope_out = k_nope.new_empty(k_nope.shape, dtype=attn_dtype)

    # Apply RoPE and quantize all components in a single fused kernel call
    # This kernel handles:
    # 1. RoPE application to q_rope and k_rope using cos_sin_cache and positions
    # 2. Quantization of all components to FP8 format
    # 3. Output placement into pre-allocated tensors
    # 调用 flashinfer 融合 kernel：一次性完成 RoPE + FP8 量化
    flashinfer.rope.mla_rope_quantize_fp8(
        q_rope=q_rope,
        k_rope=k_rope,
        q_nope=q_nope,
        k_nope=k_nope,
        cos_sin_cache=cos_sin_cache,
        pos_ids=pos_ids,
        is_neox=is_neox,
        quantize_dtype=attn_dtype,
        # Output tensor slicing: q_out contains [nope_part, rope_part]
        q_rope_out=q_out[..., kv_lora_rank:],  # RoPE part goes to end  rope 部分写到末尾
        k_rope_out=k_rope_out,
        q_nope_out=q_out[..., :kv_lora_rank],  # Nope part goes to beginning  nope 部分写到前面
        k_nope_out=k_nope_out,
        # Quantization scales (set to 1.0 for no additional scaling)
        # 量化缩放系数均为 1.0（不做额外缩放）
        quant_scale_q=1.0,
        quant_scale_kv=1.0,
    )

    # 返回量化后的合并 query、key nope 和 key rope
    return q_out, k_nope_out, k_rope_out


def concat_mla_absorb_q_general(q_nope, q_rope):
    # 优先使用融合 CUDA kernel（仅在 CUDA 且特定维度下可用）
    if _is_cuda and q_nope.shape[-1] == 512 and q_rope.shape[-1] == 64:
        # 使用 sgl_kernel 提供的高性能融合拼接算子
        return concat_mla_absorb_q(q_nope, q_rope)
    else:
        # 通用回退路径：在最后一维拼接 q_nope 和 q_rope
        return torch.cat([q_nope, q_rope], dim=-1)


@triton.jit
def reshape_and_cache_flash(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    swa_slot_mapping_ptr,
    k_scale_ptr,
    v_scale_ptr,
    block_stride,
    key_stride,
    value_stride,
    num_heads,
    head_size,
    block_size,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_SWA: tl.constexpr,
    USE_SCALE: tl.constexpr,
):
    """
    Triton kernel for reshaping per-token K/V tensors into paged KV cache layout.

    Source layout:
        key/value: [num_tokens, num_heads, head_size]

    Target cache layout:
        cache: [num_blocks, block_size, num_heads, head_size]

    Each Triton program instance handles:
        - one token (program_id(0))
        - one block of heads (program_id(1))

    Features:
        - optional SWA slot remapping
        - optional FP8 scale dequantization before cache write

    Args:
        key_ptr: Pointer to source key tensor.
        value_ptr: Pointer to source value tensor.
        key_cache_ptr: Pointer to destination key cache tensor.
        value_cache_ptr: Pointer to destination value cache tensor.
        slot_mapping_ptr: Maps token -> cache slot.
        swa_slot_mapping_ptr: Optional second-stage slot remap for SWA mode.
        k_scale_ptr: Optional key scaling factor pointer.
        v_scale_ptr: Optional value scaling factor pointer.
        block_stride: Stride between cache blocks.
        key_stride: Stride between source key tokens.
        value_stride: Stride between source value tokens.
        num_heads: Number of attention heads.
        head_size: Hidden dimension per head.
        block_size: Number of slots per cache block.
        HEAD_BLOCK: Number of heads processed per program.
        BLOCK_D: Vectorized dimension size (power-of-2 padded).
        HAS_SWA: Enable SWA remapping.
        USE_SCALE: Enable scale division before storing.
    """

    # ----------------------------------
    # program ids
    # pid0 = token
    # pid1 = head block
    # ----------------------------------
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)

    # ----------------------------------
    # slot mapping
    # ----------------------------------
    slot_idx = tl.load(slot_mapping_ptr + token_idx)

    if HAS_SWA:
        slot_idx = tl.load(swa_slot_mapping_ptr + slot_idx)

    if slot_idx < 0:
        return

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    # ----------------------------------
    # head range
    # ----------------------------------
    head_idx = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)

    head_mask = head_idx < num_heads

    dim_idx = tl.arange(0, BLOCK_D)

    # shape = [HEAD_BLOCK, BLOCK_D]
    offs = head_idx[:, None] * head_size + dim_idx[None, :]

    mask = head_mask[:, None] & (dim_idx[None, :] < head_size)

    # ----------------------------------
    # source load
    # ----------------------------------
    src_key = token_idx * key_stride + offs
    src_value = token_idx * value_stride + offs

    k = tl.load(key_ptr + src_key, mask=mask)
    v = tl.load(value_ptr + src_value, mask=mask)

    # ----------------------------------
    # optional scale
    # ----------------------------------
    if USE_SCALE:
        k_scale = tl.load(k_scale_ptr)
        v_scale = tl.load(v_scale_ptr)

        k = k / k_scale
        v = v / v_scale

    # ----------------------------------
    # target layout
    # [block_idx, block_offset, head, dim]
    # ----------------------------------
    tgt = block_idx * block_stride + block_offset * num_heads * head_size + offs

    tl.store(key_cache_ptr + tgt, k, mask=mask)
    tl.store(value_cache_ptr + tgt, v, mask=mask)


def launch_reshape_and_cache_flash(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    swa_slot_mapping=None,
    k_scale=None,
    v_scale=None,
):
    """
    Launch wrapper for reshape_and_cache_flash Triton kernel.

    This wrapper prepares launch configuration and dispatches the Triton kernel
    that writes token-major K/V tensors into paged KV cache layout.

    Args:
        key: Source key tensor [num_tokens, num_heads, head_size]
        value: Source value tensor [num_tokens, num_heads, head_size]
        key_cache: Destination key cache [num_blocks, block_size, num_heads, head_size]
        value_cache: Destination value cache [num_blocks, block_size, num_heads, head_size]
        slot_mapping: Token-to-cache slot mapping
        swa_slot_mapping: Optional SWA remapping table
        k_scale: Optional key scaling factor
        v_scale: Optional value scaling factor
    """

    num_tokens = key.shape[0]
    num_heads = key.shape[1]
    head_size = key.shape[2]

    HEAD_BLOCK = 4

    BLOCK_D = triton.next_power_of_2(head_size)

    grid = (
        num_tokens,
        triton.cdiv(num_heads, HEAD_BLOCK),
    )

    reshape_and_cache_flash[grid](
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        swa_slot_mapping,
        k_scale if k_scale is not None else key,
        v_scale if v_scale is not None else key,
        key_cache.stride(0),
        key.stride(0),
        value.stride(0),
        num_heads,
        head_size,
        key_cache.shape[1],
        HEAD_BLOCK=HEAD_BLOCK,
        BLOCK_D=BLOCK_D,
        HAS_SWA=(swa_slot_mapping is not None),
        USE_SCALE=(k_scale is not None),
    )


@triton.jit
def _get_gptj_rotated_x(
    x,
    x_rotated_mask,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
):
    # GPT-J rotary layout:
    # Pair adjacent dimensions and apply:
    # [x0, x1, x2, x3] -> [-x1, x0, -x3, x2]
    # GPT-J 旋转布局：相邻维度两两配对，奇数位取负

    # Apply sign inversion on odd positions.
    # 对偶数位置保持正值，奇数位置取负（x_rotated_mask=True 为偶数位）
    x_rotated = tl.where(x_rotated_mask, x, -x)
    # Reshape into (D/2, 2) pairs.
    # 重塑为 (D/2, 2) 的配对形式
    x_rotated = tl.reshape(x_rotated, (BLOCK_D_HALF, 2))
    # Swap each pair.
    # 交换每对中的两个元素
    x_rotated = tl.flip(x_rotated, 1)
    # Flatten back to original shape.
    # 重新展平回 (D,) 形状
    x_rotated = tl.reshape(x_rotated, (BLOCK_D,))
    return x_rotated


@triton.jit
def _get_neox_rotated_x(
    x,
    x_rotated_mask,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
):
    # GPT-NeoX rotary layout:
    # Split head dimension into two halves:
    # [x0, x1, x2, x3] -> [-x2, -x3, x0, x1]
    # GPT-NeoX 旋转布局：前半取负旋转，后半正常

    # Keep first half positive, second half negative.
    # 前半段（mask=True）保持原值，后半段取负
    x_rotated = tl.where(x_rotated_mask, x, -x)
    # Reshape into (2, D/2).
    # 重塑为 (2, D/2) 形式
    x_rotated = tl.reshape(x_rotated, (2, BLOCK_D_HALF))
    # Reverse each half.
    # 翻转每个半段
    x_rotated = tl.flip(x_rotated, 1)
    # Flatten and reverse full vector.
    # 展平后再整体翻转，得到 NeoX 旋转结果
    x_rotated = tl.reshape(x_rotated, (BLOCK_D,))
    x_rotated = tl.flip(x_rotated, 0)
    return x_rotated


@triton.jit
def _unit_rope(
    x_ptrs,       # 指向待旋转向量的指针
    cos,          # 当前位置的余弦值（形状与 x_pe 相同）
    sin,          # 当前位置的正弦值
    d_pe_offs,    # 维度偏移索引数组 [0, BLOCK_D_pe)
    IS_NEOX: tl.constexpr,         # 是否使用 NeoX 旋转布局
    BLOCK_D_pe: tl.constexpr,      # RoPE 部分的维度大小
    BLOCK_D_HALF_pe: tl.constexpr, # RoPE 维度的一半
):
    # Load one full attention head vector.
    # 加载整个注意力头向量（RoPE 部分）
    x_pe = tl.load(x_ptrs)

    # Stage 1: Build rotated vector according to rotary layout.
    # 阶段 1：根据旋转布局构建旋转向量
    if IS_NEOX:
        # NeoX 布局：前半 mask 为 True
        x_rotated_mask = d_pe_offs < BLOCK_D_HALF_pe
        x_pe_rotated = _get_neox_rotated_x(
            x_pe, x_rotated_mask, BLOCK_D_pe, BLOCK_D_HALF_pe
        )
    else:
        # GPT-J 布局：偶数位 mask 为 True
        x_rotated_mask = d_pe_offs % 2 == 0
        x_pe_rotated = _get_gptj_rotated_x(
            x_pe, x_rotated_mask, BLOCK_D_pe, BLOCK_D_HALF_pe
        )

    # Stage 2: Apply RoPE transform:
    # x' = x*cos + rotate(x)*sin
    # 阶段 2：应用旋转位置编码公式：x' = x*cos + rotate(x)*sin
    x_pe = x_pe * cos + x_pe_rotated * sin

    return x_pe


@triton.jit
def _load_cos_sin(
    cos_sin_ptr,  # cos/sin 缓存指针（合并存储：前半为 cos，后半为 sin）
    pos,          # 当前 token 的位置 ID
    d_cos_offs,   # 频率维度偏移
    stride_t,     # 位置维度步长
    stride_d,     # 频率维度步长
    freq_dim,     # 频率维度大小（用于区分 cos 和 sin 的偏移）
):
    # 计算当前位置在 cos_sin 缓存中的基地址
    base = pos * stride_t
    # 加载 cos 值（前半部分）
    cos = tl.load(cos_sin_ptr + base + d_cos_offs * stride_d)
    # 加载 sin 值（后半部分，偏移 freq_dim）
    sin = tl.load(cos_sin_ptr + base + (d_cos_offs + freq_dim) * stride_d)
    return cos, sin


@triton.jit
def _fused_qk_rope_reshape_and_cache_kernel(
    q_ptr,       # Query 张量指针 [T, QH, D]
    k_ptr,       # Key 张量指针 [T_slot, KH, D]
    v_ptr,       # Value 张量指针 [T_slot, KH, D]
    pos_ptr,     # 每个 token 的位置 ID
    cos_sin_ptr, # RoPE cos/sin 缓存
    offs_ptr,    # 可选：每个 token 的位置偏移（chunked prefill 用）
    key_cache_ptr,    # KV cache 中的 Key 存储
    value_cache_ptr,  # KV cache 中的 Value 存储
    slot_mapping_ptr,     # token -> cache slot 映射
    swa_slot_mapping_ptr, # 可选：SWA 二次 slot 重映射
    q_out_ptr,   # 输出：RoPE 后的 Query
    k_out_ptr,   # 输出：RoPE 后的 Key
    zeros_out_ptr,  # 可选：与 q_out 相同形状的全零输出
    T,           # decode token 数量
    T_slot,      # 需要更新 cache 的 token 数量（>= T）
    q_stride_t,   # q 的 token 步长
    q_stride_h,   # q 的 head 步长
    q_stride_d,   # q 的 dim 步长
    k_stride_t,   # k 的 token 步长
    k_stride_h,   # k 的 head 步长
    k_stride_d,   # k 的 dim 步长
    v_stride_t,   # v 的 token 步长
    v_stride_h,   # v 的 head 步长
    v_stride_d,   # v 的 dim 步长
    cos_sin_stride_t,  # cos_sin 的 token（位置）步长
    cos_sin_stride_d,  # cos_sin 的 dim 步长
    q_out_stride_t,  # q_out 的 token 步长
    q_out_stride_h,  # q_out 的 head 步长
    q_out_stride_d,  # q_out 的 dim 步长
    k_out_stride_t,  # k_out 的 token 步长
    k_out_stride_h,  # k_out 的 head 步长
    k_out_stride_d,  # k_out 的 dim 步长
    key_cache_stride_t,   # key_cache 的块/token 步长
    key_cache_stride_h,   # key_cache 的 head 步长
    key_cache_stride_d,   # key_cache 的 dim 步长
    key_cache_stride_b,   # key_cache 的 block_offset 步长
    key_cache_stride_x,   # key_cache 的 X 分组步长（非 flash layout）
    value_cache_stride_t,   # value_cache 的块/token 步长
    value_cache_stride_h,   # value_cache 的 head 步长
    value_cache_stride_d,   # value_cache 的 dim 步长
    value_cache_stride_b,   # value_cache 的 block_offset 步长
    value_cache_stride_slot_chunk,  # value_cache shuffle 布局中的 slot_chunk 步长
    value_cache_stride_x,   # value_cache shuffle 布局中的 X 步长
    zeros_out_stride_t,  # zeros_out 的 token 步长
    zeros_out_stride_h,  # zeros_out 的 head 步长
    zeros_out_stride_d,  # zeros_out 的 dim 步长
    k_scale_ptr,  # 可选：Key 量化缩放系数
    v_scale_ptr,  # 可选：Value 量化缩放系数
    QH_PER_KH: tl.constexpr,  # 每个 KV head 对应的 Q head 数
    QH: tl.constexpr,          # Query head 总数
    KH: tl.constexpr,          # KV head 总数
    REUSE_FREQS_FRONT_PART: tl.constexpr,  # cos/sin 是否只存了前半段频率（共享）
    IS_NEOX: tl.constexpr,     # 是否使用 NeoX RoPE 布局
    BLOCK_D_pe: tl.constexpr,  # RoPE 部分的维度大小（即 D）
    BLOCK_D_HALF_pe: tl.constexpr,  # RoPE 维度的一半
    BLOCK_SIZE: tl.constexpr,  # KV cache 的 block_size（每块 token 数）
    X_SIZE: tl.constexpr,      # 非 flash layout 时的 X 分组大小
    FLASH_LAYOUT: tl.constexpr,  # 是否使用 flash attention 布局的 cache
    VALUE_SHUFFLE_LAYOUT: tl.constexpr = False,  # value cache 是否使用 shuffle 布局
    HAVE_POS: tl.constexpr = False,     # 是否有额外位置偏移（offs_ptr）
    HAVE_K_SCALE: tl.constexpr = False, # 是否有 K 量化缩放
    HAVE_V_SCALE: tl.constexpr = False, # 是否有 V 量化缩放
    HAVE_ZEROS: tl.constexpr = False,   # 是否输出零张量
    HAS_SWA: tl.constexpr = False,     # 是否启用 SWA slot 重映射
):
    # ============================================================
    # Stage 0: Static stride assumptions for Triton compiler
    #
    # These assumptions help Triton optimize pointer arithmetic and
    # simplify generated address calculations.
    # ============================================================

    tl.assume(q_stride_t >= 0)
    tl.assume(q_stride_h >= 0)
    tl.assume(q_stride_d >= 0)
    tl.assume(k_stride_t >= 0)
    tl.assume(k_stride_h >= 0)
    tl.assume(k_stride_d >= 0)
    tl.assume(v_stride_t >= 0)
    tl.assume(v_stride_h >= 0)
    tl.assume(v_stride_d >= 0)
    tl.assume(cos_sin_stride_t >= 0)
    tl.assume(cos_sin_stride_d >= 0)
    tl.assume(q_out_stride_t >= 0)
    tl.assume(q_out_stride_h >= 0)
    tl.assume(q_out_stride_d >= 0)
    tl.assume(k_out_stride_t >= 0)
    tl.assume(k_out_stride_h >= 0)
    tl.assume(k_out_stride_d >= 0)
    tl.assume(key_cache_stride_t >= 0)
    tl.assume(key_cache_stride_h >= 0)
    tl.assume(key_cache_stride_d >= 0)
    tl.assume(key_cache_stride_b >= 0)
    tl.assume(key_cache_stride_x >= 0)
    tl.assume(value_cache_stride_t >= 0)
    tl.assume(value_cache_stride_h >= 0)
    tl.assume(value_cache_stride_d >= 0)
    tl.assume(value_cache_stride_b >= 0)
    tl.assume(value_cache_stride_slot_chunk >= 0)
    tl.assume(value_cache_stride_x >= 0)
    tl.assume(zeros_out_stride_t >= 0)
    tl.assume(zeros_out_stride_h >= 0)
    tl.assume(zeros_out_stride_d >= 0)

    # ============================================================
    # Stage 1: Program instance mapping
    #
    # Each program handles:
    #   - one (token, q_head) for Q path
    #   - selected KV ownership for cache write path
    #
    # pid layout:
    #   [0, T*QH)            -> decode Q path
    #   [T*QH, extra KV)     -> KV-only path
    # ============================================================

    pid = tl.program_id(0)
    tl.assume(pid >= 0)

    d_pe_offs = tl.arange(0, BLOCK_D_pe).to(tl.int64)

    # ============================================================
    # Stage 2: Main decode path (Q always active)
    # ============================================================

    if pid < T * QH:
        pid_t = pid // QH
        pid_hq = pid % QH

        # --------------------------------------------------------
        # Stage 2.1: Compute rotary frequency offsets
        #
        # RoPE frequencies may be stored as:
        #   D/2 frequencies (shared front-half)
        #   D frequencies (full explicit)
        # --------------------------------------------------------

        if REUSE_FREQS_FRONT_PART:
            if IS_NEOX:
                d_cos_offs = d_pe_offs
                d_cos_offs = tl.where(
                    (d_cos_offs >= BLOCK_D_HALF_pe) & (d_cos_offs < BLOCK_D_pe),
                    d_cos_offs - BLOCK_D_HALF_pe,
                    d_cos_offs,
                ).to(d_cos_offs.dtype)
                # d_cos_mask = d_cos_offs < BLOCK_D_pe
            else:
                d_cos_offs = d_pe_offs // 2
                # d_cos_mask = d_cos_offs < BLOCK_D_HALF_pe
        else:
            d_cos_offs = d_pe_offs
            # d_cos_mask = d_cos_offs < BLOCK_D_pe

        # --------------------------------------------------------
        # Stage 2.2: Load token position and optional offset
        #
        # offs_ptr is used by chunked prefill / sliding-window decode.
        # --------------------------------------------------------
        pos = tl.load(pos_ptr + pid_t)
        if HAVE_POS:
            offset = tl.load(offs_ptr + pid_t)
            pos = pos + offset

        # --------------------------------------------------------
        # Stage 2.3: Load cosine / sine table
        # --------------------------------------------------------
        # cos_offs = pos * cos_stride_t + d_cos_offs * cos_stride_d
        # cos = tl.load(cos_ptr + cos_offs)
        # sin = tl.load(sin_ptr + cos_offs)

        freq_dim = BLOCK_D_HALF_pe if REUSE_FREQS_FRONT_PART else BLOCK_D_pe

        cos, sin = _load_cos_sin(
            cos_sin_ptr,
            pos,
            d_cos_offs,
            cos_sin_stride_t,
            cos_sin_stride_d,
            freq_dim,
        )

        # --------------------------------------------------------
        # Stage 2.4: Apply RoPE to Q
        # --------------------------------------------------------
        q_ptrs = (
            q_ptr + pid_t * q_stride_t + pid_hq * q_stride_h + d_pe_offs * q_stride_d
        )
        q_pe = _unit_rope(
            q_ptrs,
            cos,
            sin,
            d_pe_offs,
            IS_NEOX,
            BLOCK_D_pe,
            BLOCK_D_HALF_pe,
        )

        # Store rotated Q output.
        q_out_ptrs = (
            q_out_ptr
            + pid_t * q_out_stride_t
            + pid_hq * q_out_stride_h
            + d_pe_offs * q_out_stride_d
        )
        tl.store(q_out_ptrs, q_pe.to(q_out_ptr.dtype.element_ty))

        if HAVE_ZEROS:
            z = tl.zeros((BLOCK_D_pe,), dtype=zeros_out_ptr.dtype.element_ty)
            zeros_out_ptrs = (
                zeros_out_ptr
                + pid_t * zeros_out_stride_t
                + pid_hq * zeros_out_stride_h
                + d_pe_offs * zeros_out_stride_d
            )
            tl.store(zeros_out_ptrs, z)

        # ========================================================
        # Stage 3: KV ownership path
        #
        # Only one Q group leader writes KV:
        #   pid_hq % QH_PER_KH == 0
        #
        # This prevents duplicated KV cache writes.
        # ========================================================

        if pid_hq % QH_PER_KH == 0:
            # ----------------------------------------------------
            # Stage 3.1: Resolve cache slot
            # ----------------------------------------------------
            pid_slot = tl.load(slot_mapping_ptr + pid_t).to(tl.int64)
            if HAS_SWA:
                pid_slot = tl.load(swa_slot_mapping_ptr + pid_slot)

            # ------------------------------------------------
            # Stage 3.2: Apply RoPE to K
            # ------------------------------------------------
            if pid_slot >= 0:
                pid_t_slot = pid_slot // BLOCK_SIZE
                pid_b = pid_slot % BLOCK_SIZE
                pid_hk = pid_hq // QH_PER_KH
                if HAVE_K_SCALE:
                    k_scale = tl.load(k_scale_ptr)
                else:
                    k_scale = 1
                k_ptrs = (
                    k_ptr
                    + pid_t * k_stride_t
                    + pid_hk * k_stride_h
                    + d_pe_offs * k_stride_d
                )
                k_pe = _unit_rope(
                    k_ptrs,
                    cos,
                    sin,
                    d_pe_offs,
                    IS_NEOX,
                    BLOCK_D_pe,
                    BLOCK_D_HALF_pe,
                )

                k_out_ptrs = (
                    k_out_ptr
                    + pid_t * k_out_stride_t
                    + pid_hk * k_out_stride_h
                    + d_pe_offs * k_out_stride_d
                )
                tl.store(k_out_ptrs, k_pe.to(k_out_ptr.dtype.element_ty))

                # ------------------------------------------------
                # Stage 3.3: Optional fp8 scaling before cache
                # ------------------------------------------------

                k_scale_rcprl = 1 / k_scale
                k_pe = k_pe * k_scale_rcprl

                # ------------------------------------------------
                # Stage 3.4: Write K cache
                #
                # Two layouts supported:
                #   FLASH_LAYOUT
                #   paged KV layout
                # ------------------------------------------------

                if FLASH_LAYOUT:
                    k_out_ptrs = (
                        key_cache_ptr
                        + pid_t_slot * key_cache_stride_t
                        + pid_b * key_cache_stride_b
                        + pid_hk * key_cache_stride_h
                        + d_pe_offs * key_cache_stride_d
                    )
                else:
                    k_pe = tl.reshape(k_pe, (BLOCK_D_pe // X_SIZE, X_SIZE))
                    dx_offs = tl.arange(0, BLOCK_D_pe // X_SIZE).to(tl.int64)
                    x_offs = tl.arange(0, X_SIZE).to(tl.int64)
                    k_out_ptrs = (
                        key_cache_ptr
                        + pid_t_slot * key_cache_stride_t
                        + pid_hk * key_cache_stride_h
                        + dx_offs[:, None] * key_cache_stride_d
                        + pid_b * key_cache_stride_b
                        + x_offs[None, :] * key_cache_stride_x
                    )

                tl.store(k_out_ptrs, k_pe.to(key_cache_ptr.dtype.element_ty))

                # ------------------------------------------------
                # Stage 3.5: Write V cache
                #
                # Supports:
                #   normal layout
                #   shuffle layout
                # ------------------------------------------------

                v_ptrs = (
                    v_ptr
                    + pid_t * v_stride_t
                    + pid_hk * v_stride_h
                    + d_pe_offs * v_stride_d
                )
                if HAVE_V_SCALE:
                    v_scale = tl.load(v_scale_ptr)
                else:
                    v_scale = 1
                v_scale_rcprl = 1 / v_scale
                v = tl.load(v_ptrs) * v_scale_rcprl
                if VALUE_SHUFFLE_LAYOUT:
                    slot_chunk = pid_b // X_SIZE
                    x_off = pid_b % X_SIZE
                    v_out_ptrs = (
                        value_cache_ptr
                        + pid_t_slot * value_cache_stride_t
                        + pid_hk * value_cache_stride_h
                        + slot_chunk * value_cache_stride_slot_chunk
                        + d_pe_offs.to(tl.int64) * value_cache_stride_d
                        + x_off * value_cache_stride_x
                    )
                else:
                    v_out_ptrs = (
                        value_cache_ptr
                        + pid_t_slot * value_cache_stride_t
                        + pid_hk * value_cache_stride_h
                        + d_pe_offs.to(tl.int64) * value_cache_stride_d
                        + pid_b * value_cache_stride_b
                    )
                tl.store(v_out_ptrs, v.to(value_cache_ptr.dtype.element_ty))
    # ============================================================
    # Stage 4: Extra KV-only path
    #
    # Handles tokens that only require cache update:
    #   T_slot > T
    #
    # No Q / no RoPE on Q branch.
    # ============================================================
    else:
        pid = pid - T * QH + T * KH
        if pid < T_slot * KH:
            pid_t = pid // KH
            pid_hk = pid % KH
            pid_slot = tl.load(slot_mapping_ptr + pid_t).to(tl.int64)
            if HAS_SWA:
                pid_slot = tl.load(swa_slot_mapping_ptr + pid_slot)

            if pid_slot >= 0:
                pid_t_slot = pid_slot // BLOCK_SIZE
                pid_b = pid_slot % BLOCK_SIZE
                if HAVE_K_SCALE:
                    k_scale = tl.load(k_scale_ptr)
                else:
                    k_scale = 1
                k_ptrs = (
                    k_ptr
                    + pid_t * k_stride_t
                    + pid_hk * k_stride_h
                    + d_pe_offs * k_stride_d
                )

                k_pe = tl.load(k_ptrs)

                k_out_ptrs = (
                    k_out_ptr
                    + pid_t * k_out_stride_t
                    + pid_hk * k_out_stride_h
                    + d_pe_offs * k_out_stride_d
                )
                tl.store(k_out_ptrs, k_pe.to(k_out_ptr.dtype.element_ty))

                k_scale_rcprl = 1 / k_scale
                k_pe = k_pe * k_scale_rcprl

                if FLASH_LAYOUT:
                    k_out_ptrs = (
                        key_cache_ptr
                        + pid_t_slot * key_cache_stride_t
                        + d_pe_offs * key_cache_stride_d
                        + pid_b * key_cache_stride_b
                        + pid_hk * key_cache_stride_h
                    )
                else:
                    k_pe = tl.reshape(k_pe, (BLOCK_D_pe // X_SIZE, X_SIZE))
                    dx_offs = tl.arange(0, BLOCK_D_pe // X_SIZE).to(tl.int64)
                    x_offs = tl.arange(0, X_SIZE).to(tl.int64)
                    k_out_ptrs = (
                        key_cache_ptr
                        + pid_t_slot * key_cache_stride_t
                        + pid_hk * key_cache_stride_h
                        + dx_offs[:, None] * key_cache_stride_d
                        + pid_b * key_cache_stride_b
                        + x_offs[None, :] * key_cache_stride_x
                    )
                tl.store(k_out_ptrs, k_pe.to(key_cache_ptr.dtype.element_ty))

                v_ptrs = (
                    v_ptr
                    + pid_t * v_stride_t
                    + pid_hk * v_stride_h
                    + d_pe_offs * v_stride_d
                )
                if HAVE_V_SCALE:
                    v_scale = tl.load(v_scale_ptr)
                else:
                    v_scale = 1
                v_scale_rcprl = 1 / v_scale
                v = tl.load(v_ptrs) * v_scale_rcprl
                if VALUE_SHUFFLE_LAYOUT:
                    slot_chunk = pid_b // X_SIZE
                    x_off = pid_b % X_SIZE
                    v_out_ptrs = (
                        value_cache_ptr
                        + pid_t_slot * value_cache_stride_t
                        + pid_hk * value_cache_stride_h
                        + slot_chunk * value_cache_stride_slot_chunk
                        + d_pe_offs * value_cache_stride_d
                        + x_off * value_cache_stride_x
                    )
                else:
                    v_out_ptrs = (
                        value_cache_ptr
                        + pid_t_slot * value_cache_stride_t
                        + pid_hk * value_cache_stride_h
                        + d_pe_offs * value_cache_stride_d
                        + pid_b * value_cache_stride_b
                    )
                tl.store(v_out_ptrs, v.to(value_cache_ptr.dtype.element_ty))


def fused_qk_rope_reshape_and_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    pos: torch.Tensor,
    cos_sin: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    is_neox: bool,
    flash_layout: bool,
    apply_scale: bool = True,
    offs: torch.Tensor = None,
    q_out: torch.Tensor = None,
    k_out: torch.Tensor = None,
    output_zeros: bool = True,
    zeros_out: torch.Tensor = None,
    swa_slot_mapping=None,
):
    """
    Perform RoPE on q and k and along the last dimension and copy k and v in to key_cache and value_cache inplace
    对 q 和 k 应用 RoPE 位置编码，并将 k/v 写入 KV cache（原地操作）

    Key parameters:
    - q: shape (T, QH, D).
    - k: shape (T_slot, KH, D).
    - v: shape (T_slot, KH, D).
    - if flash_layout:
    -     key_cache: shape (T_cache, block_size, KH, D).
    -     value_cache: shape (T_cache, block_size, KH, D).
    - else:
    -     key_cache: shape (T_cache, KH, D // x, block_size, x).
    -     value_cache: shape (T_cache, KH, D, block_size).
    - slot_mapping: shape (T_slot, ).

    T is the number of decode tokens, T_cahce * block_size is the max number of tokens of kv_cache
    QH must be multiple of KH

    Returns:
    - q_out: same shape as input q.
    - k_out: same shape as input k.
    - key_cache: same shape as input key_cache (inplace).
    - value_cache: same shape as input value_cache (inplace).
    - zeros_out: same shape as input q.
    """

    # 提取 q/k/v 的形状信息
    t, qh, d = q.shape
    tk, kh, dk = k.shape
    tv, vh, dv = v.shape
    # 根据 cache 布局分别解析 cache 形状
    if flash_layout:
        t_cache, block_size, kh_cache, dk_cache = key_cache.shape
        t_cache_v, block_size_v, vh_cache, dv_cache = value_cache.shape
        value_shuffle_layout = False
    else:
        t_cache, kh_cache, dkx_cache, block_size, x_cache = key_cache.shape
        if value_cache.ndim == 5:
            # value_cache shuffle: (num_blocks, num_kv_heads, block_size // x, head_size, x)
            # value cache 使用 shuffle 布局（用于某些优化）
            t_cache_v, vh_cache, slot_chunk_v, dv_cache, x_v = value_cache.shape
            value_shuffle_layout = True
            block_size_v = slot_chunk_v * x_v
            assert block_size_v == block_size and x_v == x_cache, (
                f"value_cache shuffle (T,KH,block_size//x,D,x) must match key: "
                f"{block_size_v=} {block_size=} {x_v=} {x_cache=}"
            )
        else:
            # 标准 value cache 布局
            t_cache_v, vh_cache, dv_cache, block_size_v = value_cache.shape
            value_shuffle_layout = False
    (t_slot,) = slot_mapping.shape

    # 校验 token 数量一致，且 slot_mapping 大小不超过 k/v 的 token 数
    assert (
        t == tk == tv and t_slot <= tk
    ), f"Number of tokens should be identical for q, kand v. The number of tokens of slot_mapping should no more than that of q, k and v, {t=} {tk=} {tv=} {t_slot=}"
    # 校验 key/value cache 的 block_size 一致
    assert (
        block_size == block_size_v
    ), f"block size should be identical for key_cache, and value_cache {block_size} {block_size_v}"
    # 校验 KV head 数在各张量间一致
    assert (
        kh == vh == kh_cache == vh_cache
    ), "KV head should be identical for k, v, key_cache, and value_cache"
    # 校验 key/value cache 的块数一致
    assert (
        t_cache == t_cache_v
    ), "Number of tokens should be identical for key_cache, and value_cache"
    # 校验特征维度在各张量间一致
    if flash_layout:
        assert (
            d == dk == dv == dk_cache == dv_cache
        ), "D dimension should be identical for q, k, and v"
    else:
        assert (
            d == dk == dv == dkx_cache * x_cache == dv_cache
        ), "D dimension should be identical for q, k, and v"
        assert x_cache == triton.next_power_of_2(x_cache), "x_size should be power of 2"

    # D 和 block_size 必须是 2 的幂，Triton 向量化要求
    assert d == triton.next_power_of_2(d), "D dimension should be power of 2"
    assert block_size == triton.next_power_of_2(
        block_size
    ), "block_size should be power of 2"
    assert qh % kh == 0, "Q heads must be multiple of H heads"
    # 判断 cos_sin 缓存是否只存了前半段频率（可复用到后半段）
    d_freq = cos_sin.shape[-1] // 2
    assert (d_freq == d // 2) or (
        d_freq == d
    ), "cos/sin last dim should be the same or half of the qk last dim"
    reuse_freqs_front_part = d_freq == d // 2

    # 若未预先分配输出张量，则在此分配
    if q_out is None:
        q_out = torch.empty((t, qh, d), dtype=q.dtype, device=q.device)

    if k_out is None:
        k_out = torch.empty((tk, kh, dk), dtype=k.dtype, device=q.device)

    # 处理 zeros_out：若已传入则验证形状，否则按需分配
    if zeros_out is not None:
        tz, qhz, dz = zeros_out.shape
        assert (
            t == tz and qh == qhz and d == dz
        ), f"q and zeros shape mismatch {q.shape=} {zeros_out.shape=}"
        output_zeros = True
    elif output_zeros:
        zeros_out = torch.empty((t, qh, d), dtype=q.dtype, device=q.device)
    else:
        zeros_out = None

    # 计算总程序数：decode token*QH + 额外的 KV-only token*KH
    n_pid = t * qh + (t_slot - t) * kh if t_slot >= t else t * qh
    grid = (n_pid, 1, 1)
    # 启动融合 Triton kernel
    _fused_qk_rope_reshape_and_cache_kernel[grid](
        q,
        k,
        v,
        pos,
        cos_sin,
        offs,
        key_cache,
        value_cache,
        slot_mapping,
        swa_slot_mapping,
        q_out,
        k_out,
        zeros_out,
        t,
        t_slot,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        cos_sin.stride(0),
        cos_sin.stride(-1),
        *q_out.stride(),
        *k_out.stride(),
        # key_cache 步长根据 flash/paged 布局传入不同维度顺序
        key_cache.stride(0) if not flash_layout else key_cache.stride(0),
        key_cache.stride(1) if not flash_layout else key_cache.stride(2),
        key_cache.stride(2) if not flash_layout else key_cache.stride(3),
        key_cache.stride(3) if not flash_layout else key_cache.stride(1),
        key_cache.stride(4) if not flash_layout else 0,
        # value_cache 步长根据布局类型传入
        value_cache.stride(0) if not flash_layout else value_cache.stride(0),
        value_cache.stride(1) if not flash_layout else value_cache.stride(2),
        (
            value_cache.stride(3)
            if (not flash_layout and value_shuffle_layout)
            else (value_cache.stride(2) if not flash_layout else value_cache.stride(3))
        ),
        (
            0
            if (not flash_layout and value_shuffle_layout)
            else (value_cache.stride(3) if not flash_layout else value_cache.stride(1))
        ),
        value_cache.stride(2) if (not flash_layout and value_shuffle_layout) else 0,
        value_cache.stride(4) if (not flash_layout and value_shuffle_layout) else 0,
        # zeros_out 步长（无则传 0）
        zeros_out.stride(0) if zeros_out is not None else 0,
        zeros_out.stride(1) if zeros_out is not None else 0,
        zeros_out.stride(2) if zeros_out is not None else 0,
        k_scale_ptr=k_scale,
        v_scale_ptr=v_scale,
        QH_PER_KH=qh // kh,
        QH=qh,
        KH=kh,
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
        IS_NEOX=is_neox,
        BLOCK_D_pe=d,
        BLOCK_D_HALF_pe=d // 2,
        BLOCK_SIZE=block_size,
        X_SIZE=x_cache if not flash_layout else 0,
        FLASH_LAYOUT=flash_layout,
        VALUE_SHUFFLE_LAYOUT=value_shuffle_layout,
        HAVE_POS=(offs is not None),
        HAVE_K_SCALE=(k_scale is not None and apply_scale),
        HAVE_V_SCALE=(v_scale is not None and apply_scale),
        HAVE_ZEROS=output_zeros,
        HAS_SWA=(swa_slot_mapping is not None),
        num_warps=1,
    )

    # 根据是否有 zeros_out 返回不同数量的输出张量
    if zeros_out is not None:
        return q_out.view(-1, qh * d), k_out, key_cache, value_cache, zeros_out
    return q_out.view(-1, qh * d), k_out, key_cache, value_cache
