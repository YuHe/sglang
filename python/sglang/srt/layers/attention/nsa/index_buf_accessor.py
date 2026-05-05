# NSA 稀疏索引 KV buffer 访问器模块
# 负责从分页（paged）的 index_k buffer 中高效读写 FP8 量化 K 数据和 float32 scale
# Buffer 布局：每页 = page_size 个 token × (128B FP8 K + 4B FP32 scale)
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.utils import is_hip

# 检测硬件平台：HIP（AMD ROCm）和 FP8 FNUZ 格式（AMD）
_is_hip = is_hip()
_is_fp8_fnuz = is_fp8_fnuz()

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool

"""
k: data, 128 item per token, fp8
s: scale, 1 item per token, fp32
"""


class GetK:
    # 从分页 buffer 读取 FP8 量化 K 数据的封装类
    @classmethod
    def execute(cls, *args, **kwargs):
        # 默认使用 Triton 加速实现
        return cls.triton(*args, **kwargs)

    @classmethod
    def slow(
        cls, pool: "NSATokenToKVPool", buf, seq_len: int, page_indices: torch.Tensor
    ):
        # 参考实现（慢速）：逐页循环读取 K 数据
        num_pages = (seq_len + pool.page_size - 1) // pool.page_size
        seq_len_ = num_pages * pool.page_size  # 对齐到页边界的长度
        index_k_fp8 = torch.empty(
            (seq_len_, pool.index_head_dim),
            dtype=torch.uint8,
            device=pool.device,
        )
        for i in range(num_pages):
            page_index = page_indices[i]
            # 从当前页读取 page_size 个 token 的 K 数据
            index_k_fp8[i * pool.page_size : (i + 1) * pool.page_size] = buf[
                page_index
            ][: pool.page_size * pool.index_head_dim].view(-1, pool.index_head_dim)

        return index_k_fp8[:seq_len]  # 截断到实际序列长度

    @classmethod
    def torch_fast(
        cls, pool: "NSATokenToKVPool", buf, seq_len: int, page_indices: torch.Tensor
    ):
        """
        :param page_indices: (num_pages,), int32
        :return: (seq_len, index_head_dim), uint8
        """

        # can handle per 128B instead of per element
        # 利用 torch.gather 批量读取，每次处理 128B 块而非逐元素

        # page_indices: (num_pages,), element := a page index
        buf_numel_per_page = buf.shape[1]  # 每页的字节数

        num_k_bytes_per_page = pool.page_size * pool.index_head_dim  # 每页 K 数据字节数
        num_k_bytes_per_token = pool.index_head_dim  # 每个 token 的 K 字节数

        # buf: (num_pages, page_size 64 * head_dim 128 + page_size 64 * fp32_nbytes 4), uint8
        # flat_buf: (whatever,), uint8
        flat_buf = buf.flatten()  # 展平为 1D 便于索引

        # flat_indices: (num_pages, num_k_bytes_per_page), int32, element := an index into flat_buf that we want to access
        # 为每页的每个字节计算在 flat_buf 中的全局偏移
        flat_indices = (page_indices * buf_numel_per_page)[:, None] + torch.arange(
            num_k_bytes_per_page, dtype=torch.int32, device="cuda"
        )[None, :]
        flat_indices = flat_indices.flatten()[: seq_len * num_k_bytes_per_token]

        out = flat_buf[flat_indices]  # 一次性 gather 所有 K 数据
        return out.view(-1, 128)  # 恢复为 (seq_len, 128) 形状

    @classmethod
    def triton(
        cls, pool: "NSATokenToKVPool", buf, seq_len: int, page_indices: torch.Tensor
    ):
        """
        Triton implementation for gathering K data from paged buffer.
        :param page_indices: (num_pages,), int32/int64
        :return: (seq_len, index_head_dim), uint8
        """
        # 调用 Triton kernel 并行读取所有 token 的 K 数据
        return _get_k_triton(
            buf=buf,
            page_indices=page_indices,
            seq_len=seq_len,
            page_size=pool.page_size,
            index_head_dim=pool.index_head_dim,
        )


class GetS:
    # 从分页 buffer 读取 float32 scale 数据的封装类
    @classmethod
    def execute(cls, *args, **kwargs):
        return cls.triton(*args, **kwargs)

    @classmethod
    def slow(
        cls, pool: "NSATokenToKVPool", buf, seq_len: int, page_indices: torch.Tensor
    ):
        # 参考实现（慢速）：逐页循环读取 scale 数据（存储在 K 数据之后）
        num_pages = (seq_len + pool.page_size - 1) // pool.page_size
        seq_len_ = num_pages * pool.page_size
        assert pool.index_head_dim // pool.quant_block_size == 1
        index_k_scale_fp8 = torch.empty(
            (seq_len_, 4),  # 每个 token 1 个 float32 scale = 4 字节
            dtype=torch.uint8,
            device=pool.device,
        )
        for i in range(num_pages):
            page_index = page_indices[i]
            # scale 存储在 K 数据之后，偏移量为 page_size * index_head_dim
            index_k_scale_fp8[i * pool.page_size : (i + 1) * pool.page_size] = buf[
                page_index
            ][pool.page_size * pool.index_head_dim :].view(-1, 4)
        return index_k_scale_fp8[:seq_len]

    @classmethod
    def torch_fast(
        cls, pool: "NSATokenToKVPool", buf, seq_len: int, page_indices: torch.Tensor
    ):
        """
        :param page_indices: (num_pages,), int32
        :return: (seq_len, index_head_dim // quant_block_size), uint8
        """
        buf_numel_per_page = buf.shape[1]

        # scale 区域在每页的 K 数据之后
        num_s_bytes_per_page = buf.shape[1] - pool.page_size * pool.index_head_dim
        num_s_bytes_per_token = pool.index_head_dim // pool.quant_block_size * 4  # 每个 token 的 scale 字节数
        s_offset_in_page = pool.page_size * pool.index_head_dim  # scale 在页内的起始偏移

        flat_buf = buf.flatten()
        # 为每页的每个 scale 字节计算全局偏移
        flat_indices = (
            (page_indices * buf_numel_per_page)[:, None]
            + torch.arange(num_s_bytes_per_page, dtype=torch.int32, device="cuda")[
                None, :
            ]
            + s_offset_in_page
        )
        flat_indices = flat_indices.flatten()[: seq_len * num_s_bytes_per_token]

        out = flat_buf[flat_indices]
        return out.view(-1, 4)  # 恢复为 (seq_len, 4) 形状（每个 token 1 个 fp32 scale）

    @classmethod
    def triton(
        cls, pool: "NSATokenToKVPool", buf, seq_len: int, page_indices: torch.Tensor
    ):
        """
        Triton implementation for gathering S (scale) data from paged buffer.
        :param page_indices: (num_pages,), int32/int64
        :return: (seq_len, 4), uint8
        """
        return _get_s_triton(
            buf=buf,
            page_indices=page_indices,
            seq_len=seq_len,
            page_size=pool.page_size,
            index_head_dim=pool.index_head_dim,
        )


class GetKAndS:
    # 融合读取 K 和 scale 数据的封装类（单次 kernel 调用，比分开调用更高效）
    @classmethod
    def execute(cls, *args, **kwargs):
        return cls.triton(*args, **kwargs)

    @classmethod
    def triton(
        cls,
        pool: "NSATokenToKVPool",
        buf: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_tensor: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ):
        """
        Triton implementation for gathering both K and S data from paged buffer in a single call.
        :param page_indices: (num_pages,), int32/int64
        :param seq_len_tensor: (num_pages,), int32/int64
        :param seq_len_sum: sum of all sequence len, int32
        :param max_seq_len: max of all sequence len, int32
        :return: tuple of (k_fp8, k_scale) where
                 k_fp8: (seq_len, index_head_dim), uint8
                 k_scale: (seq_len, 4), uint8
        """
        # 单次 Triton kernel 同时读取 K 数据和 scale，减少显存访问次数
        return _get_k_and_s_triton(
            buf=buf,
            page_indices=page_indices,
            seq_lens=seq_len_tensor,
            seq_len_sum=seq_len_sum,
            max_seq_len=max_seq_len,
            page_size=pool.page_size,
            index_head_dim=pool.index_head_dim,
        )


class SetK:
    # 向分页 buffer 写入 FP8 量化 K 数据的封装类
    @classmethod
    def execute(cls, *args, buf, **kwargs):
        return cls.torch_fast(*args, **kwargs, buf=buf)

    @classmethod
    def slow(
        cls,
        pool: "NSATokenToKVPool",
        buf: torch.Tensor,
        loc: torch.Tensor,
        index_k: torch.Tensor,
    ):
        # 参考实现（慢速）：逐 token 循环写入 K 数据
        for i in range(len(loc)):
            # 根据 token 全局索引计算页号和页内偏移
            page_index = loc[i] // pool.page_size
            offset = loc[i] % pool.page_size
            buf[
                page_index,
                offset * pool.index_head_dim : (offset + 1) * pool.index_head_dim,
            ] = index_k[i].view(torch.uint8)

    @classmethod
    def torch_fast(
        cls,
        pool: "NSATokenToKVPool",
        buf: torch.Tensor,
        loc: torch.Tensor,
        index_k: torch.Tensor,
    ):
        # 快速 PyTorch 实现：批量计算写入地址，一次性写入所有 K 数据
        (num_tokens_to_write,) = loc.shape
        buf_numel_per_page = buf.shape[1]
        num_k_bytes_per_token = pool.index_head_dim

        # loc: (num_tokens_to_write,), int32, element := the token index to write to
        # 将 token 全局索引分解为页号和页内偏移
        loc_page_index = loc // pool.page_size
        loc_token_offset_in_page = loc % pool.page_size

        flat_buf = buf.flatten()
        # 计算每个 token 的每个字节在 flat_buf 中的全局写入地址
        flat_indices = (
            (loc_page_index * buf_numel_per_page)[:, None]
            + (loc_token_offset_in_page * num_k_bytes_per_token)[:, None]
            + torch.arange(num_k_bytes_per_token, dtype=torch.int32, device="cuda")[
                None, :
            ]
        )
        num_k_bytes_total = num_tokens_to_write * num_k_bytes_per_token
        flat_indices = flat_indices.flatten()[:num_k_bytes_total]
        # 批量写入，将 index_k 视为 uint8 并展平
        flat_buf[flat_indices] = index_k.view(torch.uint8).flatten()


class SetS:
    # 向分页 buffer 写入 float32 scale 数据的封装类
    @classmethod
    def execute(cls, *args, buf, **kwargs):
        return cls.torch_fast(*args, **kwargs, buf=buf)

    @classmethod
    def slow(
        cls,
        pool: "NSATokenToKVPool",
        buf: torch.Tensor,
        loc: torch.Tensor,
        index_k_scale: torch.Tensor,
    ):
        # 参考实现（慢速）：逐 token 写入 scale（存储在 K 数据之后）
        for i in range(len(loc)):
            page_index = loc[i] // pool.page_size
            offset = loc[i] % pool.page_size
            start = pool.page_size * pool.index_head_dim  # scale 区域起始偏移
            buf[page_index, start + offset * 4 : start + (offset + 1) * 4] = (
                index_k_scale[i].view(torch.uint8)
            )

    @classmethod
    def torch_fast(
        cls,
        pool: "NSATokenToKVPool",
        buf: torch.Tensor,
        loc: torch.Tensor,
        index_k_scale: torch.Tensor,
    ):
        # 快速 PyTorch 实现：批量计算 scale 写入地址（在 K 数据之后）
        (num_tokens_to_write,) = loc.shape
        buf_numel_per_page = buf.shape[1]
        num_s_bytes_per_token = 4  # float32 = 4 字节
        s_offset_in_page = pool.page_size * pool.index_head_dim  # scale 区域起始

        # loc: (num_tokens_to_write,), int32, element := the token index to write to
        loc_page_index = loc // pool.page_size
        loc_token_offset_in_page = loc % pool.page_size

        flat_buf = buf.flatten()
        # 计算每个 token 的 scale 在 flat_buf 中的写入地址
        flat_indices = (
            (loc_page_index * buf_numel_per_page)[:, None]
            + s_offset_in_page  # 跳过 K 数据区域
            + (loc_token_offset_in_page * num_s_bytes_per_token)[:, None]
            + torch.arange(num_s_bytes_per_token, dtype=torch.int32, device="cuda")[
                None, :
            ]
        )
        number_s_bytes_total = num_tokens_to_write * num_s_bytes_per_token
        flat_indices = flat_indices.flatten()[:number_s_bytes_total]
        flat_buf[flat_indices] = index_k_scale.view(torch.uint8).flatten()


class SetKAndS:
    # 融合写入 K 数据和 scale 的封装类（单次 Triton kernel，比分开调用更高效）
    @classmethod
    def execute(cls, *args, buf, **kwargs):
        if 0:
            # print("SetK, SetS comparison test")
            # 调试模式：对比两种实现的结果
            buf_cloned = buf.clone()
            cls.vanilla(*args, **kwargs, buf=buf)
            cls.triton(*args, **kwargs, buf=buf_cloned)

            def _clear_token_0(target):
                target[0, :128] = target[0, 64 * 128 : 64 * 128 + 4] = 0

            _clear_token_0(buf)
            _clear_token_0(buf_cloned)

            assert torch.all(
                buf == buf_cloned
            ), f"{buf=} {buf_cloned=} {kwargs['loc'].to_list()=}"
            return

        # 正常路径：使用 Triton 融合 kernel
        cls.triton(*args, **kwargs, buf=buf)

    @classmethod
    def vanilla(cls, pool, buf, loc, index_k, index_k_scale):
        # 非融合参考实现：分别调用 SetK 和 SetS
        SetK.execute(pool=pool, buf=buf, loc=loc, index_k=index_k)
        SetS.execute(pool=pool, buf=buf, loc=loc, index_k_scale=index_k_scale)

    @classmethod
    def triton(cls, pool, buf, loc, index_k, index_k_scale):
        # 融合 Triton 实现：单次 kernel 同时写入 K 数据和 scale
        _set_k_and_s_triton(
            buf=buf,
            loc=loc,
            index_k=index_k,
            index_k_scale=index_k_scale,
            page_size=pool.page_size,
        )


def _set_k_and_s_triton(
    buf: torch.Tensor,
    loc: torch.Tensor,
    index_k: torch.Tensor,
    index_k_scale: torch.Tensor,
    page_size: int,
):
    """
    :param buf: (num_pages, page_size 64 * (128B data + 4B scale)), uint8
    :param loc: (num_tokens_to_write,), int, element := the token index to write to
    :param index_k: (num_tokens_to_write, 128 elem), fp8
    :param index_k_scale: (num_tokens_to_write, 1 elem), fp32
    :return:
    """
    # 融合 K 和 scale 写入：校验参数、转换视图后启动 Triton kernel
    num_pages, buf_numel_per_page = buf.shape
    (num_tokens_to_write,) = loc.shape
    num_tokens_to_write_, index_head_dim = index_k.shape

    # Handle both 1D (num_tokens,) and 2D (num_tokens, 1) shapes for index_k_scale
    # 支持 1D 和 2D 两种 scale 张量形状
    if index_k_scale.ndim == 1:
        num_tokens_to_write__ = index_k_scale.shape[0]
        scale_dim = 1
    elif index_k_scale.ndim == 2:
        num_tokens_to_write__, scale_dim = index_k_scale.shape
    else:
        raise ValueError(
            f"index_k_scale must be 1D or 2D, got shape {index_k_scale.shape}"
        )
    # HIP（AMD）和 CUDA 的 page_size 不同：HIP=1, CUDA=64
    if _is_hip:
        assert buf_numel_per_page == 1 * (128 + 4)
    else:
        assert buf_numel_per_page == 64 * (128 + 4)
    assert num_tokens_to_write == num_tokens_to_write_ == num_tokens_to_write__
    assert index_head_dim == 128  # FP8 K 数据固定 128 维
    assert scale_dim == 1         # 每个 token 1 个 scale
    if _is_hip:
        assert page_size == 1
    else:
        assert page_size == 64

    # 数据类型和内存连续性校验
    assert buf.dtype == torch.uint8
    assert loc.dtype == torch.int64, f"{loc.dtype=}"  # can be int32
    if _is_fp8_fnuz:
        assert index_k.dtype == torch.float8_e4m3fnuz
    else:
        assert index_k.dtype == torch.float8_e4m3fn
    assert index_k_scale.dtype == torch.float32

    assert buf.is_contiguous()
    assert loc.is_contiguous()
    assert index_k.is_contiguous()
    assert index_k_scale.is_contiguous()

    # 为 buf 创建 FP8 和 float32 两种视图，分别用于写 K 和 scale
    if _is_fp8_fnuz:
        buf_fp8 = buf.view(torch.float8_e4m3fnuz)
    else:
        buf_fp8 = buf.view(torch.float8_e4m3fn)
    buf_fp32 = buf.view(torch.float32)  # float32 视图用于写 scale（//4 地址空间）

    # 每个 token 对应一个 Triton program
    _set_k_and_s_triton_kernel[(num_tokens_to_write,)](
        buf_fp8,
        buf_fp32,
        loc,
        index_k,
        index_k_scale,
        index_k.stride(0),
        PAGE_SIZE=page_size,
        BUF_NUMEL_PER_PAGE=buf_numel_per_page,
        NUM_K_ELEMS_PER_TOKEN=index_head_dim,
        S_OFFSET_NBYTES_IN_PAGE=page_size * index_head_dim,  # scale 在页内的字节偏移
    )


@triton.jit
def _set_k_and_s_triton_kernel(
    # 融合写入 K（FP8）和 scale（float32）的 Triton kernel
    buf_fp8_ptr,          # FP8 视图的 buffer 指针（用于写 K 数据）
    buf_fp32_ptr,         # float32 视图的 buffer 指针（用于写 scale）
    loc_ptr,              # token 全局索引指针
    index_k_ptr,          # FP8 K 数据指针 [num_tokens, 128]
    index_k_scale_ptr,    # float32 scale 指针 [num_tokens]
    index_k_ptr_stride_0, # index_k 的行步长（字节数）
    PAGE_SIZE: tl.constexpr,             # 每页的 token 数
    BUF_NUMEL_PER_PAGE: tl.constexpr,   # 每页的总字节数（FP8 视图）
    NUM_K_ELEMS_PER_TOKEN: tl.constexpr, # 每个 token 的 K 元素数（= 128）
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr, # scale 区域在页内的字节偏移
):
    token_id = tl.program_id(0)  # 每个 program 处理一个 token

    # 从 loc 数组读取当前 token 的全局索引
    loc = tl.load(loc_ptr + token_id)

    in_k_offsets = token_id * index_k_ptr_stride_0 + tl.arange(0, NUM_K_ELEMS_PER_TOKEN)

    # no need for `mask`, since we read 128B for k and 4B for scale, both pow of 2
    # 128 和 4 都是 2 的幂次，无需 mask
    k = tl.load(index_k_ptr + in_k_offsets)       # 加载 128 个 FP8 K 值
    k_scale = tl.load(index_k_scale_ptr + token_id)  # 加载 1 个 float32 scale

    # 将 token 全局索引分解为页号和页内 token 偏移
    loc_page_index = loc // PAGE_SIZE
    loc_token_offset_in_page = loc % PAGE_SIZE

    # 计算 K 数据写入地址（FP8 视图）
    out_k_offsets = (
        loc_page_index * BUF_NUMEL_PER_PAGE
        + loc_token_offset_in_page * NUM_K_ELEMS_PER_TOKEN
        + tl.arange(0, NUM_K_ELEMS_PER_TOKEN)
    )

    # "//4" b/c it is fp32 instead of uint8
    # scale 地址使用 float32 视图（地址空间缩小 4 倍），需要除以 4
    out_s_offset = (
        loc_page_index * BUF_NUMEL_PER_PAGE // 4
        + S_OFFSET_NBYTES_IN_PAGE // 4
        + loc_token_offset_in_page
    )

    # 写入 K 数据（FP8）和 scale（float32）
    tl.store(buf_fp8_ptr + out_k_offsets, k)
    tl.store(buf_fp32_ptr + out_s_offset, k_scale)


def _get_k_triton(
    buf: torch.Tensor,
    page_indices: torch.Tensor,
    seq_len: int,
    page_size: int,
    index_head_dim: int,
):
    """
    Gather K (key) data from paged buffer using Triton.

    :param buf: (num_pages, page_size * 128 + page_size * 4), uint8
    :param page_indices: (num_pages,), int32/int64
    :param seq_len: int, number of tokens to gather
    :param page_size: int, typically 64
    :param index_head_dim: int, typically 128
    :return: (seq_len, index_head_dim), uint8
    """
    # 分配输出张量并启动 Triton kernel（每个 token 一个 program）
    num_pages, buf_numel_per_page = buf.shape

    # Allocate output
    out = torch.empty((seq_len, index_head_dim), dtype=torch.uint8, device=buf.device)

    # Launch kernel with one thread per token
    grid = (seq_len,)
    _get_k_triton_kernel[grid](
        buf,
        page_indices,
        out,
        seq_len,
        page_size,
        buf_numel_per_page,
        index_head_dim,
        BLOCK_SIZE=128,  # 每个 token 读取 128 字节
    )

    return out


@triton.jit
def _get_k_triton_kernel(
    # 从分页 buffer 读取 K 数据的 Triton kernel
    buf_ptr,
    page_indices_ptr,
    out_ptr,
    seq_len: tl.constexpr,
    page_size: tl.constexpr,
    buf_numel_per_page: tl.constexpr,
    index_head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program handles one token (seq_len tokens total).
    Loads 128 bytes from the appropriate page.
    """
    token_id = tl.program_id(0)

    # Calculate which page and offset within page
    # 根据 token 全局 ID 计算对应的页号和页内偏移
    page_idx = token_id // page_size
    token_offset_in_page = token_id % page_size

    # Load the page index from page_indices
    # 从 page_indices 读取当前 page 的物理地址
    page_index = tl.load(page_indices_ptr + page_idx)

    # Calculate source offset in buf
    # buf[page_index, token_offset_in_page * index_head_dim : ...]
    # 计算 K 数据在 buf 中的起始偏移
    src_base_offset = (
        page_index * buf_numel_per_page + token_offset_in_page * index_head_dim
    )

    # Load 128 bytes (index_head_dim elements)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < index_head_dim
    data = tl.load(buf_ptr + src_base_offset + offsets, mask=mask)

    # Store to output
    dst_offset = token_id * index_head_dim
    tl.store(out_ptr + dst_offset + offsets, data, mask=mask)


def _get_s_triton(
    buf: torch.Tensor,
    page_indices: torch.Tensor,
    seq_len: int,
    page_size: int,
    index_head_dim: int,
):
    """
    Gather S (scale) data from paged buffer using Triton.

    :param buf: (num_pages, page_size * 128 + page_size * 4), uint8
    :param page_indices: (num_pages,), int32/int64
    :param seq_len: int, number of tokens to gather
    :param page_size: int, typically 64
    :param index_head_dim: int, typically 128
    :return: (seq_len, 4), uint8 (representing fp32 scale)
    """
    num_pages, buf_numel_per_page = buf.shape
    s_offset_in_page = page_size * index_head_dim  # Scales start after K data（scale 区域起始字节偏移）

    # Allocate output
    out = torch.empty((seq_len, 4), dtype=torch.uint8, device=buf.device)

    # Launch kernel with one thread per token
    grid = (seq_len,)
    _get_s_triton_kernel[grid](
        buf,
        page_indices,
        out,
        seq_len,
        page_size,
        buf_numel_per_page,
        s_offset_in_page,
    )

    return out


@triton.jit
def _get_s_triton_kernel(
    # 从分页 buffer 读取 scale 数据的 Triton kernel
    buf_ptr,
    page_indices_ptr,
    out_ptr,
    seq_len: tl.constexpr,
    page_size: tl.constexpr,
    buf_numel_per_page: tl.constexpr,
    s_offset_in_page: tl.constexpr,  # scale 在页内的起始字节偏移
):
    """
    Each program handles one token (seq_len tokens total).
    Loads 4 bytes (fp32 scale) from the appropriate page.
    """
    token_id = tl.program_id(0)

    # Calculate which page and offset within page
    page_idx = token_id // page_size
    token_offset_in_page = token_id % page_size

    # Load the page index from page_indices
    page_index = tl.load(page_indices_ptr + page_idx)

    # Calculate source offset in buf
    # Scales are stored after K data: page_size * index_head_dim offset
    # buf[page_index, s_offset_in_page + token_offset_in_page * 4 : ...]
    # scale 在页内的地址 = page 基址 + scale 区域偏移 + token 在 scale 区域的偏移
    src_base_offset = (
        page_index * buf_numel_per_page + s_offset_in_page + token_offset_in_page * 4
    )

    # Load 4 bytes (fp32 scale)
    offsets = tl.arange(0, 4)
    data = tl.load(buf_ptr + src_base_offset + offsets)

    # Store to output
    dst_offset = token_id * 4
    tl.store(out_ptr + dst_offset + offsets, data)


def _get_k_and_s_triton(
    buf: torch.Tensor,
    page_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_len_sum: int,
    max_seq_len: int,
    page_size: int,
    index_head_dim: int,
):
    """
    Fused gather of both K (key) and S (scale) data from paged buffer using Triton.
    This is more efficient than calling GetK and GetS separately.

    :param buf: (num_pages, page_size * 128 + page_size * 4), uint8
    :param page_indices: (num_pages,), int32/int64
    :param seq_lens: tensor of sequence lens, int64
    :param seq_len_sum: sum of all sequence len, int32
    :param max_seq_len: max of sequence len, int32
    :param page_size: int, typically 64
    :param index_head_dim: int, typically 128
    :return: tuple of (k_out, s_out) where
             k_out: (seq_len, index_head_dim), uint8
             s_out: (seq_len, 4), uint8
    """
    # 融合读取 K 和 scale：分配输出张量，计算 grid，启动 Triton kernel
    # Allocate outputs
    k_out = torch.empty(
        (seq_len_sum, index_head_dim), dtype=torch.uint8, device=buf.device
    )
    s_out = torch.empty((seq_len_sum, 4), dtype=torch.uint8, device=buf.device)

    _, buf_numel_per_page = buf.shape
    _, page_indice_batch_offset = page_indices.shape
    s_offset_in_page = page_size * index_head_dim  # scale 在页内的起始字节偏移

    # Launch kernel with one thread per token
    BLOCK_SIZE = 256    # 每个 block 处理的 token 数（沿 token 维度分块）
    BLOCK_SIZE_K = 128  # 每个线程处理的 K 维度大小

    # 计算三维 grid：(seq_num, token_blocks, k_blocks)
    num_token_blocks = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_k_threads = (index_head_dim + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K

    seq_num = seq_lens.shape[0]
    grid = (seq_num, num_token_blocks, num_k_threads)
    # 计算 seq_num 的最小 2 的幂次方（用于 Triton 的前缀和计算）
    seq_num_pow2 = 1
    while seq_num_pow2 < seq_num:
        seq_num_pow2 *= 2

    _get_k_and_s_triton_kernel[grid](
        buf_ptr=buf,
        page_indices_ptr=page_indices,
        k_out_ptr=k_out,
        s_out_ptr=s_out,
        seq_len_ptr=seq_lens,
        seq_len_num_pow=seq_num_pow2,
        page_size=page_size,
        buf_numel_per_page=buf_numel_per_page,
        index_head_dim=index_head_dim,
        s_offset_in_page=s_offset_in_page,
        page_indice_batch_offset=page_indice_batch_offset,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return k_out, s_out


@triton.jit
def _get_k_and_s_triton_kernel(
    # 融合读取 K 和 scale 的 Triton kernel
    # 三维 grid：(batch_id, token_block_id, k_block_id)
    buf_ptr,
    page_indices_ptr,
    k_out_ptr,
    s_out_ptr,
    seq_len_ptr,
    seq_len_num_pow: tl.constexpr,   # seq_num 的最小 2 幂（用于前缀和）
    page_size: tl.constexpr,
    buf_numel_per_page: tl.constexpr,
    index_head_dim: tl.constexpr,
    s_offset_in_page: tl.constexpr,
    page_indice_batch_offset: tl.constexpr,  # 每个 batch 在 page_indices 中的步长
    BLOCK_SIZE: tl.constexpr,    # token 维度分块大小
    BLOCK_SIZE_K: tl.constexpr,  # K 维度分块大小
):
    """
    Fused kernel that gathers both K and S data in a single pass.
    Each program handles one token (seq_len tokens total).
    Loads 128 bytes (K) + 4 bytes (S) from the appropriate page.
    """
    # 三维 program ID：batch 维度、token 块维度、K 维度
    batch_id = tl.program_id(0)
    block_token_start = tl.program_id(1) * BLOCK_SIZE
    thread_idx = tl.program_id(2)

    # Define the token range within the block and the K dimension range handled by the thread.
    # 定义当前 block 内的 token 偏移和 K 维度偏移
    token_ids_in_block = tl.arange(0, BLOCK_SIZE)
    token_ids = block_token_start + token_ids_in_block
    k_offsets = thread_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    # 加载当前 batch 的序列长度，生成有效 token mask
    seq_len = tl.load(seq_len_ptr + batch_id)
    token_valid_mask = token_ids < seq_len

    # 计算当前 batch 在输出中的 token 起始偏移（对前序所有 batch 的序列长度求和）
    pre_batch_idx = tl.arange(0, seq_len_num_pow)
    mask_pre_batch_idx = pre_batch_idx < batch_id
    prev_seq_lens = tl.load(seq_len_ptr + pre_batch_idx, mask=mask_pre_batch_idx)
    batch_token_offset = tl.sum(prev_seq_lens)  # 当前 batch 在输出中的起始 token 索引

    # Batch calculate the page index and in-page offset of each token.
    # 批量计算每个 token 所在的页号和页内偏移
    page_idx = token_ids // page_size
    token_offset_in_page = token_ids % page_size
    page_indices_base = batch_id * page_indice_batch_offset
    page_idx_valid_mask = page_idx < page_indice_batch_offset
    # 从 page_indices 表中读取每个 token 对应的物理页地址
    page_index = tl.load(
        page_indices_ptr + page_idx + page_indices_base,
        mask=token_valid_mask & page_idx_valid_mask,
    )

    # ===== Load K data =====
    # The address calculation logic for K: page_index * total number of elements in a single page + K offset of the token within the page.
    # K 数据地址 = 页基址 + token 在页内的 K 偏移
    k_src_token_offset = token_offset_in_page * index_head_dim
    k_src_base_offset = page_index * buf_numel_per_page + k_src_token_offset

    # 广播计算每个 (token, k_dim) 的加载地址
    k_load_addr = buf_ptr + k_src_base_offset[:, None] + k_offsets[None, :]
    k_dim_mask = k_offsets[None, :] < index_head_dim
    k_mask = token_valid_mask[:, None] & k_dim_mask

    k_data = tl.load(k_load_addr, mask=k_mask, other=0)

    # Store K to output
    # 写入 K 数据到输出张量（按全局 token 索引排列）
    k_dst_token_offset = batch_token_offset + token_ids
    k_dst_base_offset = k_dst_token_offset * index_head_dim
    k_store_addr = k_out_ptr + k_dst_base_offset[:, None] + k_offsets[None, :]
    tl.store(k_store_addr, k_data, mask=k_mask)

    # ===== Load S data =====
    # The address calculation logic for S: page_index * total number of elements in a single page + starting offset of S within the page + offset of token within S in the page
    # scale 地址 = 页基址 + scale 区域偏移 + token 在 scale 区域内的偏移（每个 token 4 字节）
    s_src_token_offset = s_offset_in_page + token_offset_in_page * 4
    s_src_base_offset = page_index * buf_numel_per_page + s_src_token_offset

    # 每个 scale 4 字节
    s_offsets = tl.arange(0, 4)
    s_load_addr = buf_ptr + s_src_base_offset[:, None] + s_offsets[None, :]
    s_mask = token_valid_mask[:, None] & (s_offsets[None, :] < 4)
    s_data = tl.load(s_load_addr, mask=s_mask, other=0)

    # Store S to output
    # 写入 scale 数据到输出张量
    s_dst_token_offset = batch_token_offset + token_ids
    s_dst_base_offset = s_dst_token_offset * 4
    s_store_addr = s_out_ptr + s_dst_base_offset[:, None] + s_offsets[None, :]
    tl.store(s_store_addr, s_data, mask=s_mask)
