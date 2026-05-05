# NSA K-cache 反量化模块
# 将 FP8 量化存储的 K-cache 还原为 BF16，支持分块缩放（nope 部分）和直接拷贝（rope 部分）
import torch
import triton
import triton.language as tl


def dequantize_k_cache(quant_k_cache):
    # 公共入口：使用快速 Triton kernel 进行 K-cache 反量化
    return _dequantize_k_cache_fast_wrapped(quant_k_cache)


def _dequantize_k_cache_ref(
    quant_k_cache: torch.Tensor,  # (num_blocks, block_size, 1, bytes_per_token)
    dv: int = 512,
    tile_size: int = 128,
    d: int = 576,
) -> torch.Tensor:
    """
    De-quantize the k-cache
    """
    # 参考实现（纯 PyTorch）：按 tile 逐块反量化 nope 部分，直接拷贝 rope 部分
    assert dv % tile_size == 0
    original_ndim = quant_k_cache.ndim
    if original_ndim == 3:
        # set block_size = 1
        # 3D 输入视为 block_size=1，统一处理为 4D
        quant_k_cache = quant_k_cache.unsqueeze(1)
    num_tiles = dv // tile_size  # nope 部分分成多少个 tile（每个 tile 独立 scale）
    num_blocks, block_size, h_k, _ = quant_k_cache.shape
    assert h_k == 1
    result = torch.empty(
        (num_blocks, block_size, d), dtype=torch.bfloat16, device=quant_k_cache.device
    )

    # 将 quant_k_cache 展平为 [num_blocks, block_size, total_bytes]
    quant_k_cache = quant_k_cache.view(num_blocks, block_size, -1)

    # 解析量化存储布局：[nope_int8 | nope_scales_fp32 | rope_bf16]
    input_nope = quant_k_cache[..., :dv]                                        # FP8 量化的 nope 部分
    input_scale = quant_k_cache[..., dv : dv + num_tiles * 4].view(torch.float32)  # 每 tile 的 float32 缩放因子
    input_rope = quant_k_cache[..., dv + num_tiles * 4 :].view(torch.bfloat16)   # rope 部分（未量化，直接 BF16）
    # rope 部分直接拷贝到输出的高维位置
    result[..., dv:] = input_rope

    for tile_idx in range(0, num_tiles):
        # 取出当前 tile 的 FP8 量化值，转 float32 后乘以对应 scale，完成反量化
        cur_nope = input_nope[
            ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
        ].to(torch.float32)
        cur_scales = input_scale[..., tile_idx].unsqueeze(-1)  # 广播到 tile_size 维度
        result[..., tile_idx * tile_size : (tile_idx + 1) * tile_size] = (
            cur_nope * cur_scales
        )

    if original_ndim == 3:
        # 恢复原始 3D 形状 [num_blocks, 1, d]
        return result.view(num_blocks, 1, -1)
    else:
        # 恢复原始 4D 形状 [num_blocks, block_size, 1, d]
        return result.view(num_blocks, block_size, 1, -1)


def _dequantize_k_cache_fast_wrapped(
    quant_k_cache: torch.Tensor,
    dv: int = 512,
    tile_size: int = 128,
) -> torch.Tensor:
    # 快速 Triton kernel 的包装函数：处理输入形状归一化和输出形状恢复
    original_ndim = quant_k_cache.ndim
    if original_ndim == 3:
        # set block_size = 1
        quant_k_cache = quant_k_cache.unsqueeze(1)
    num_blocks, block_size, _, dim_quant = quant_k_cache.shape
    # 固定配置：dv=512, dim_quant=656, tile_size=128
    assert dv == 512
    assert dim_quant == 656
    assert tile_size == 128
    # 将所有 blocks/tokens 合并为 2D，方便 kernel 处理
    quant_k_cache = quant_k_cache.view((-1, dim_quant))

    output = _dequantize_k_cache_fast(quant_k_cache)

    if original_ndim == 3:
        return output.view(num_blocks, 1, -1)
    else:
        return output.view(num_blocks, block_size, 1, -1)


def _dequantize_k_cache_fast(quant_k_cache, group_size: int = 128):
    # 调用 Triton kernel 对 2D 量化 K-cache [num_tokens, dim_quant] 进行反量化
    num_tokens, dim_quant = quant_k_cache.shape

    assert quant_k_cache.dtype == torch.float8_e4m3fn
    dim_nope = 512   # nope 维度（量化部分）
    dim_rope = 64    # rope 维度（未量化，直接 BF16 存储）
    num_tiles = dim_nope // group_size  # nope 部分的 tile 数 = 512 // 128 = 4
    assert dim_quant == 656  # 512(nope_fp8) + 4*4(scales_fp32) + 64*2(rope_bf16)/2 = 512+16+128=656

    output = torch.empty(
        (num_tokens, dim_nope + dim_rope),
        dtype=torch.bfloat16,
        device=quant_k_cache.device,
    )

    # 每个 token 的 block 数：ceil((512+64)/128) = 5（4 个 nope tile + 1 个 rope block）
    num_blocks_per_token = triton.cdiv(dim_nope + dim_rope, group_size)
    assert num_blocks_per_token == 5

    assert dim_nope % group_size == 0

    # 解析量化存储布局
    input_nope_q = quant_k_cache[:, :dim_nope]                                 # FP8 量化 nope
    input_nope_s = quant_k_cache[:, dim_nope : dim_nope + num_tiles * 4].view(
        torch.float32
    )                                                                           # float32 缩放因子
    input_rope = quant_k_cache[:, dim_nope + num_tiles * 4 :].view(torch.bfloat16)  # BF16 rope

    # 二维 grid：(num_tokens, num_blocks_per_token)
    _dequantize_k_cache_fast_kernel[(num_tokens, num_blocks_per_token)](
        output,
        input_nope_q,
        input_nope_s,
        input_rope,
        output.stride(0),
        input_nope_q.stride(0),
        input_nope_s.stride(0),
        input_rope.stride(0),
        NUM_NOPE_BLOCKS=num_tiles,
        GROUP_SIZE=group_size,
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
    )

    return output


@triton.jit
def _dequantize_k_cache_fast_kernel(
    # Triton kernel：逐 token、逐 block 并行反量化 K-cache
    output_ptr,
    input_nope_q_ptr,    # FP8 量化 nope 数据指针
    input_nope_s_ptr,    # float32 缩放因子指针
    input_rope_ptr,      # BF16 rope 数据指针
    output_stride_0: int,
    input_nope_q_stride_0: int,
    input_nope_s_stride_0: int,
    input_rope_stride_0: int,
    NUM_NOPE_BLOCKS: tl.constexpr,  # nope tile 数量（= dim_nope / GROUP_SIZE）
    GROUP_SIZE: tl.constexpr,       # 每个 tile/block 的列数
    DIM_NOPE: tl.constexpr,         # nope 维度总长
    DIM_ROPE: tl.constexpr,         # rope 维度总长
):
    # program_id(0)=token_id，program_id(1)=block_id（前 NUM_NOPE_BLOCKS 个处理 nope，最后一个处理 rope）
    token_id = tl.program_id(0)
    raw_block_id = tl.program_id(1)

    if raw_block_id < NUM_NOPE_BLOCKS:
        # a. dequant nope
        # 处理 nope 部分：FP8 量化值 × 对应 scale → BF16
        effective_block_id = raw_block_id

        # 计算当前 tile 在 nope 维度的列偏移
        offs_q = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs_q < DIM_NOPE
        # 加载当前 token、当前 tile 的 FP8 量化值
        ptr_q = input_nope_q_ptr + token_id * input_nope_q_stride_0 + offs_q
        # 加载当前 token、当前 tile 的 float32 缩放因子（标量）
        ptr_s = input_nope_s_ptr + token_id * input_nope_s_stride_0 + effective_block_id

        y_q = tl.load(ptr_q, mask=mask, other=0.0).to(tl.float32)  # FP8 → float32
        y_s = tl.load(ptr_s)                                         # 加载 scale

        # 反量化：乘以 scale 后转回 BF16
        y = (y_q * y_s).to(output_ptr.dtype.element_ty)

        # 写入输出张量的 nope 部分
        dst_ptr = output_ptr + token_id * output_stride_0 + offs_q
        tl.store(dst_ptr, y, mask=mask)
    else:
        # b. copy rope
        # 处理 rope 部分：直接从 BF16 rope 拷贝到输出高维位置
        effective_block_id = raw_block_id - NUM_NOPE_BLOCKS

        offs = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_ROPE

        # 源地址：rope 数据（BF16）
        src_ptr = input_rope_ptr + token_id * input_rope_stride_0 + offs
        # 目标地址：输出中 nope 之后的 rope 区域
        dst_ptr = output_ptr + token_id * output_stride_0 + DIM_NOPE + offs

        data = tl.load(src_ptr, mask=mask).to(tl.bfloat16)
        tl.store(dst_ptr, data, mask=mask)


def dequantize_k_cache_paged(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """
    De-quantize the k-cache with paged layout
    Args:
        quant_k_cache: [total_num_tokens, 1, dim_quant] or [num_blocks, block_size, 1, dim_quant], the quantized k-cache in paged layout
        page_table_1_flattened: [num_tokens], the flattened page_table_1 with the page indices in each requests concatenated together
    Returns:
        output: [num_tokens, 1, dim_nope + dim_rope], the de-quantized k-cache
    """
    # 分页布局下的 K-cache 反量化：通过 page_table_1 间接寻址，支持前缀共享
    dim_quant = quant_k_cache.shape[-1]
    assert (
        dim_quant == 656
    ), f"dim_quant: {dim_quant} != 656 detected in dequantize_k_cache_paged"
    # 将所有 blocks 展平为 [total_tokens, dim_quant]
    quant_k_cache = quant_k_cache.view((-1, dim_quant))

    # num_tokens can exceed kv_cache_size due to prefix sharing (multiple seqs share same KV slots)
    # Index bounds validated in nsa_backend.init_forward_metadata
    # 输出 token 数由 page_table_1 决定（可能有多个序列共享同一 KV slot）
    num_tokens = page_table_1_flattened.shape[0]
    assert quant_k_cache.dtype == torch.float8_e4m3fn
    dim_nope = 512
    dim_rope = 64
    num_tiles = dim_nope // group_size  # 512 // 128 = 4

    output = torch.empty(
        (num_tokens, 1, dim_nope + dim_rope),
        dtype=torch.bfloat16,
        device=quant_k_cache.device,
    )

    # cdiv(512 + 64, 128) = 5
    # 每个 token 对应 5 个 block（4 个 nope tile + 1 个 rope block）
    num_blocks_per_token = triton.cdiv(dim_nope + dim_rope, group_size)
    assert num_blocks_per_token == 5

    assert dim_nope % group_size == 0

    # 解析量化存储布局
    input_nope_q = quant_k_cache[:, :dim_nope]
    # [:, 512:512+4*4] = [:, 512:528]
    # float32 缩放因子（4 个 tile × 4 字节 = 16 字节）
    input_nope_s = quant_k_cache[:, dim_nope : dim_nope + num_tiles * 4].view(
        torch.float32
    )
    # [:, 528:]
    # rope 部分：剩余字节解释为 BF16
    input_rope = quant_k_cache[:, dim_nope + num_tiles * 4 :].view(torch.bfloat16)

    # 二维 grid：(num_tokens, num_blocks_per_token)，每个 kernel 处理一个 token 的一个 block
    _dequantize_k_cache_paged_kernel[(num_tokens, num_blocks_per_token)](
        output,
        input_nope_q,
        input_nope_s,
        input_rope,
        page_table_1_flattened,
        output.stride(0),
        input_nope_q.stride(0),
        input_nope_s.stride(0),
        input_rope.stride(0),
        NUM_NOPE_BLOCKS=num_tiles,
        GROUP_SIZE=group_size,
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
    )

    return output


@triton.jit
def _dequantize_k_cache_paged_kernel(
    # 分页版 K-cache 反量化 Triton kernel
    # 与 _dequantize_k_cache_fast_kernel 的区别：通过 page_table_1 间接寻址源数据
    output_ptr,
    input_nope_q_ptr,
    input_nope_s_ptr,
    input_rope_ptr,
    page_table_1_ptr,        # 分页表：token_id → 物理 KV slot 索引
    output_stride_0: int,
    input_nope_q_stride_0: int,
    input_nope_s_stride_0: int,
    input_rope_stride_0: int,
    NUM_NOPE_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
):
    # token_id 为逻辑 token 编号，token_id_paged 为对应的物理 KV cache 地址
    token_id = tl.program_id(0)
    token_id_paged = tl.load(page_table_1_ptr + token_id).to(tl.int32)  # 通过页表转换为物理地址
    raw_block_id = tl.program_id(1)

    if raw_block_id < NUM_NOPE_BLOCKS:
        # a. dequant nope
        # 使用物理地址 token_id_paged 读取量化数据，写入逻辑地址 token_id
        effective_block_id = raw_block_id

        offs_q = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs_q < DIM_NOPE
        # 读取使用物理地址（paged），写入使用逻辑地址
        ptr_q = input_nope_q_ptr + token_id_paged * input_nope_q_stride_0 + offs_q
        ptr_s = (
            input_nope_s_ptr
            + token_id_paged * input_nope_s_stride_0
            + effective_block_id
        )

        y_q = tl.load(ptr_q, mask=mask, other=0.0).to(tl.float32)
        y_s = tl.load(ptr_s)

        # 反量化：FP8 × scale → BF16
        y = (y_q * y_s).to(output_ptr.dtype.element_ty)

        dst_ptr = output_ptr + token_id * output_stride_0 + offs_q
        tl.store(dst_ptr, y, mask=mask)
    else:
        # b. copy rope
        # rope 部分：使用物理地址读取，写入逻辑地址的 nope 之后位置
        effective_block_id = raw_block_id - NUM_NOPE_BLOCKS

        offs = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_ROPE

        src_ptr = input_rope_ptr + token_id_paged * input_rope_stride_0 + offs
        dst_ptr = output_ptr + token_id * output_stride_0 + DIM_NOPE + offs

        data = tl.load(src_ptr, mask=mask).to(tl.bfloat16)
        tl.store(dst_ptr, data, mask=mask)


if __name__ == "__main__":
    # 单元测试请参见 quant_k_cache.py
    raise Exception("UT is in quant_k_cache.py")
