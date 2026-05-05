# NSA（Native Sparse Attention）上下文并行工具函数模块
# 提供 cp（context parallel）分片、序列长度计算、填充等辅助函数
from typing import TYPE_CHECKING, List, Tuple, Union

import torch
import triton
import triton.language as tl

# 导入数据并行注意力相关工具：rank/size 查询和填充模式
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    get_attention_cp_rank,
    get_attention_cp_size,
    get_attention_dp_rank,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils.common import ceil_align, ceil_div

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def compute_nsa_seqlens(original_seq_lens, nsa_index_topk: int):
    # 将原始序列长度截断到 top-k 索引数量，避免超出稀疏索引容量
    return original_seq_lens.clamp(max=nsa_index_topk)


def is_nsa_enable_prefill_cp():
    # 查询全局配置，判断是否开启 NSA prefill 阶段的 context parallel
    return get_global_server_args().enable_nsa_prefill_context_parallel


def is_nsa_prefill_cp_in_seq_split():
    # 判断 NSA prefill cp 是否采用 in-seq-split 模式（序列内连续切分）
    return (
        is_nsa_enable_prefill_cp()
        and get_global_server_args().nsa_prefill_cp_mode == "in-seq-split"
    )


def is_nsa_prefill_cp_round_robin_split():
    # 判断 NSA prefill cp 是否采用 round-robin-split 模式（轮询切分）
    return (
        is_nsa_enable_prefill_cp()
        and get_global_server_args().nsa_prefill_cp_mode == "round-robin-split"
    )


def can_nsa_prefill_cp_round_robin_split(forward_batch: "ForwardBatch"):
    # 判断当前 forward batch 是否满足轮询切分条件
    if not forward_batch.forward_mode.is_context_parallel_extend():
        # 非 context parallel extend 模式，不可分片
        return False
    cp_size = get_attention_cp_size()
    seq_len = sum(forward_batch.extend_seq_lens_cpu)
    # 序列必须非空且长度大于等于 cp_size，同时 cp_size > 1 才有意义
    return (
        is_nsa_prefill_cp_round_robin_split()
        and seq_len > 0
        and seq_len >= cp_size
        and cp_size > 1
    )


def nsa_cp_round_robin_split_data(input_: Union[torch.Tensor, List]):
    """
    # for round-robin-split, split the tokens evenly according to the rule of token_idx % cp_size.
    |   +-----------before split------------+|
    | token0, token1, token2, token3, token4, token5, token6, token7, ...
    |
    |   +--------------result-------------------+
    | dp_atten_tp0: token0, token4, token8, token12, token16, ... |
    | dp_atten_tp1: token1, token5, token9, token13, token17, ... |
    | dp_atten_tp2: token2, token6, token10, token14, token18, ... |
    | dp_atten_tp3: token3, token7, token11, token15, token19, ... |
    |   +-------------------------+
    """
    # 按 token_idx % cp_size == cp_rank 规则，将数据轮询分配到当前 rank
    cp_size = get_attention_cp_size()
    cp_rank = get_attention_cp_rank()
    if isinstance(input_, (tuple, list)):
        # 对列表/元组，直接取步长为 cp_size 的切片
        indices = range(cp_rank, len(input_), cp_size)
        return input_[indices]

    tokens = len(input_)
    if tokens % cp_size != 0:
        # token 数不能整除 cp_size，需要按实际分配量取索引
        cur_len = tokens // cp_size + (tokens % cp_size > cp_rank)
        if cur_len == 0:
            # 当前 rank 分到 0 个 token，返回空张量
            return input_.new_empty(0, *input_.shape[1:])
        indices = torch.arange(cp_rank, tokens, cp_size, device=input_.device)
        return input_[indices]

    # for torch device tensor
    # token 数可整除 cp_size，reshape 后直接按 rank 列取值（contiguous 保证内存连续）
    return input_.view(-1, cp_size, *input_.shape[1:])[:, cp_rank].contiguous()


def cal_padded_tokens(forward_batch: "ForwardBatch"):
    # Consistent with the padding calculation logic in ForwardBatch.prepare_mlp_sync_batch,
    # calculate the actual token length after padding when attn_tp_size > 1 or in the MAX_LEN padding mode.
    # 复制全局 token 数列表，按 attn_cp_size 对齐每个 DP rank 的 token 数
    global_num_tokens = forward_batch.global_num_tokens_cpu.copy()
    sync_group_size = len(global_num_tokens)
    attn_cp_size = get_attention_cp_size()
    for i in range(sync_group_size):
        # 向上取整对齐，保证每个 rank 的 token 数是 cp_size 的倍数
        global_num_tokens[i] = ceil_align(global_num_tokens[i], attn_cp_size)
    # 根据是否有 extend batch 及各 rank token 数决定填充模式
    dp_padding_mode = DpPaddingMode.get_dp_padding_mode(
        forward_batch.is_extend_in_batch, global_num_tokens
    )
    if dp_padding_mode.is_max_len():
        # MAX_LEN 模式：取所有 rank 中最大的 token 数作为统一长度
        tokens = max(global_num_tokens)
    elif len(global_num_tokens) > 1:
        # 多 DP rank：取当前 rank 自己的 token 数
        tokens = global_num_tokens[get_attention_dp_rank()]
    else:
        tokens = global_num_tokens[0]
    if can_nsa_prefill_cp_round_robin_split(forward_batch):
        # 轮询切分时，每个 cp rank 只处理 tokens / cp_size 个 token
        tokens = ceil_div(tokens, attn_cp_size)
    return tokens


def pad_nsa_cache_seqlens(forward_batch: "ForwardBatch", nsa_cache_seqlens):
    # 对 NSA KV cache 序列长度向量进行填充，使其与 padded token 数对齐
    attn_cp_size = get_attention_cp_size()
    needs_cp_pad = attn_cp_size > 1 and can_nsa_prefill_cp_round_robin_split(
        forward_batch
    )
    # 存在多 DP rank 时也需要填充
    needs_dp_pad = forward_batch.global_num_tokens_cpu is not None
    if not needs_cp_pad and not needs_dp_pad:
        # 无需填充，直接返回原始序列长度向量
        return nsa_cache_seqlens
    tokens = cal_padded_tokens(forward_batch)
    pad_len = tokens - nsa_cache_seqlens.shape[0]
    if pad_len > 0:
        # 在末尾追加零填充，使长度与 padded tokens 对齐
        nsa_cache_seqlens = torch.cat(
            [
                nsa_cache_seqlens,
                nsa_cache_seqlens.new_zeros(pad_len, *nsa_cache_seqlens.shape[1:]),
            ]
        )
    return nsa_cache_seqlens


def can_nsa_cp_split(seq_len: int, cp_size: int, use_nsa: bool, forward_batch):
    # 判断当前序列是否可以进行 NSA context parallel 切分
    if is_nsa_prefill_cp_round_robin_split():
        # 轮询模式：seq_len 必须能被 cp_size 整除
        cur_cp_seq_len = seq_len // cp_size
        assert (
            seq_len % cp_size == 0
        ), f"seq_len {seq_len} is not divisible by cp_size {cp_size} when nsa_prefill_cp_mode is round-robin-split"
    else:
        # TODO current just support prefill batch=1 and len(input_ids) > self.cp_size * 2
        # Note: (self.cp_size * 2) To achieve load balancing for seq computation,
        # the seq data needs to be divided and recombined at twice the size of cp_size.
        # in-seq-split 模式：每个 rank 处理 seq_len // (2*cp_size) 个 token
        cur_cp_seq_len = seq_len // (cp_size * 2)
    if (
        cur_cp_seq_len != 0
        and cp_size > 1
        and use_nsa
        and forward_batch.forward_mode.is_context_parallel_extend()
        and is_nsa_enable_prefill_cp()
        and sum(forward_batch.extend_seq_lens_cpu) >= cp_size
    ):
        # 满足所有条件：分片有效长度非零、启用了 NSA、当前为 cp extend 模式
        return True
    else:
        return False


@triton.jit
def nsa_cp_round_robin_split_q_seqs_kernel(
    # Triton GPU kernel：轮询切分 Q 序列长度，计算当前 rank 分到的 token 数
    in_seqs_ptr,   # 输入：每个序列的原始长度指针
    out_seqs_ptr,  # 输出：当前 rank 分到的序列长度指针
    bs_idx_ptr,    # 输出：当前 rank 有非零长度的 batch 索引指针
    tokens: tl.constexpr,    # batch 大小（序列总数）
    cp_size: tl.constexpr,   # context parallel 总 rank 数
    cp_rank: tl.constexpr,   # 当前 rank 编号
):
    extra_seq = 0  # 跨序列边界的余量 token 数（上个序列未被整除的余数）
    bs_idx = 0     # 已写入的有效 batch 索引计数
    for bs in range(tokens):
        # 从显存读取当前序列长度
        cur_len = tl.load(in_seqs_ptr + bs)
        # 加上上一序列的余量，实现跨序列的连续轮询
        cur_len += extra_seq
        # 计算当前 rank 应分到的 token 数（若 cur_len % cp_size > cp_rank 则多分一个）
        cur_seq = cur_len // cp_size + (cur_len % cp_size > cp_rank)
        if cur_seq > 0:
            # 只记录有非零分配的序列
            tl.store(bs_idx_ptr + bs_idx, bs)
            tl.store(out_seqs_ptr + bs_idx, cur_seq)
            bs_idx += 1
        # 更新余量：总长度减去当前 rank 处理量的 cp_size 倍
        extra_seq = cur_len - cur_seq * cp_size


def nsa_cp_round_robin_split_q_seqs_cpu(extend_seqs):
    # CPU 版本：轮询切分序列长度，返回当前 rank 的非零长度列表及对应 batch 索引
    cp_size = get_attention_cp_size()
    cp_rank = get_attention_cp_rank()
    extra_seq = 0  # 跨序列累计余量
    q_seqs = []
    for bs, cur_len in enumerate(extend_seqs):
        # 加上余量后计算当前 rank 分到的 token 数
        cur_len += extra_seq
        cur_seq = cur_len // cp_size + int(cur_len % cp_size > cp_rank)
        q_seqs.append(cur_seq)
        extra_seq = cur_len - cur_seq * cp_size
    # 过滤掉长度为 0 的序列，返回有效 batch 索引
    bs_idx = list([i for i, x in enumerate(q_seqs) if x > 0])
    q_seqs = [q_len for q_len in q_seqs if q_len > 0]
    return q_seqs, bs_idx


def nsa_cp_round_robin_split_q_seqs(
    extend_seqs_cpu, extend_seqs
) -> Tuple[List, torch.Tensor, List, torch.Tensor]:
    """
    round-robin-split distributes tokens across ranks based on token_idx % cp_size.

    Return:
    ret_q_lens_cpu(List) and ret_q_lens(torch.Tensor): the partitioned length (excluding zeros) on the current cp rank
        for each sequence after distribution across cp ranks.
    bs_idx_cpu(List) and bs_idx(torch.Tensor): marks which sequences are ultimately selected,
        i.e., those with a partitioned length greater than zero.
    """
    # 同时在 CPU 和 GPU 上计算轮询切分结果，返回两种格式供不同场景使用
    cp_size = get_attention_cp_size()
    cp_rank = get_attention_cp_rank()
    # len(ret_q_lens_cpu) == len(bs_idx_cpu)
    # 先在 CPU 上计算，得到 CPU 列表格式的结果
    ret_q_lens_cpu, bs_idx_cpu = nsa_cp_round_robin_split_q_seqs_cpu(extend_seqs_cpu)
    # 在 GPU 上分配输出张量
    ret_q_lens = torch.empty(
        (len(bs_idx_cpu),), device=extend_seqs.device, dtype=extend_seqs.dtype
    )
    bs_idx = torch.empty(
        (len(bs_idx_cpu),), device=extend_seqs.device, dtype=torch.int32
    )
    # 启动单线程 Triton kernel 在 GPU 上执行相同切分逻辑
    grid = (1,)
    nsa_cp_round_robin_split_q_seqs_kernel[grid](
        extend_seqs, ret_q_lens, bs_idx, len(extend_seqs), cp_size, cp_rank
    )
    return ret_q_lens_cpu, ret_q_lens, bs_idx_cpu, bs_idx


def nsa_use_prefill_cp(forward_batch, nsa_enable_prefill_cp=None):
    # 判断当前 forward batch 是否应使用 NSA prefill context parallel
    if nsa_enable_prefill_cp is None:
        nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
    if (
        forward_batch.attn_cp_metadata is not None
        and nsa_enable_prefill_cp
        and forward_batch.forward_mode.is_context_parallel_extend()
    ):
        # 已有 cp 元数据、全局配置开启、且处于 cp extend 模式
        return True
    else:
        return False
