# 导入数据类装饰器，用于定义结构化元数据
from dataclasses import dataclass

# 导入 PyTorch，用于 Tensor 类型注解
import torch

# 导入 ForwardMetadata 基类（来自 Mamba2 模块，提供公共 forward 元数据字段）
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
# 导入 ForwardBatch，封装当前批次的请求信息（prefill/decode 混合）
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


# Bailing 线性注意力前向传播所需的元数据数据类
# 继承 ForwardMetadata，扩展了线性注意力特有的字段
@dataclass(kw_only=True)
class BailingLinearMetadata(ForwardMetadata):
    num_prefills: int           # 当前批次中 prefill 请求的数量
    num_prefill_tokens: int     # prefill 请求的总 token 数
    num_decodes: int            # 当前批次中 decode 请求的数量
    batch_size: int             # 总批次大小（prefill + decode）
    has_initial_states: torch.Tensor  # 布尔张量：每个序列是否有历史 KV 状态（prefix > 0）
    q_lengths: torch.Tensor           # 每个序列的 Query 长度（从 query_start_loc 差分得到）

    @staticmethod
    def prepare_decode(
        query_start_loc: torch.Tensor,   # Query 的起始位置偏移（CSR 格式）
        mamba_cache_indices: torch.Tensor,  # Mamba 缓存槽位索引
        bs: int,                          # batch size（decode 请求数）
        seq_lens: torch.Tensor,           # 每个 decode 请求的序列长度
    ) -> "BailingLinearMetadata":
        """This path is run during CUDA graph capture, i.e. decode only, so `num_prefills` is 0"""
        # CUDA graph 捕获路径：仅 decode 模式，所有请求均有历史状态（has_initial_states 全 1）
        return BailingLinearMetadata(
            batch_size=bs,
            query_start_loc=query_start_loc,
            mamba_cache_indices=mamba_cache_indices,
            num_decodes=seq_lens.shape[0],  # decode 请求数等于 seq_lens 的元素数
            num_prefills=0,                  # CUDA graph 模式无 prefill 请求
            num_prefill_tokens=0,
            has_initial_states=torch.ones_like(seq_lens),  # decode 均有历史状态
            q_lengths=query_start_loc.diff(),  # 通过差分计算每个序列的 Query 长度
        )

    @classmethod
    def prepare_mixed(
        cls,
        query_start_loc: torch.Tensor,    # Query 的起始位置偏移
        mamba_cache_indices: torch.Tensor, # Mamba 缓存槽位索引
        forward_batch: ForwardBatch,       # 包含混合 prefill+decode 批次信息的对象
    ) -> "BailingLinearMetadata":
        """This path cannot run with CUDA graph, as it contains extend requests."""
        # 若无 extend 请求，退化为纯 decode 路径
        if forward_batch.extend_num_tokens is None:
            return cls.prepare_decode(
                query_start_loc=query_start_loc,
                mamba_cache_indices=mamba_cache_indices,
                bs=forward_batch.batch_size,
                seq_lens=forward_batch.seq_lens,
            )
        # 提取混合批次中的 prefill 和 decode 信息
        num_prefills = len(forward_batch.extend_seq_lens)      # prefill 请求数
        num_prefill_tokens = forward_batch.extend_num_tokens    # prefill 总 token 数
        num_decodes = len(forward_batch.seq_lens) - num_prefills  # decode 请求数
        # 获取每个 prefill 序列的 prefix 长度（prefix > 0 表示有历史状态）
        context_lens_tensor = forward_batch.extend_prefix_lens
        assert context_lens_tensor is not None
        # 布尔掩码：prefix 长度大于 0 的序列有历史状态
        has_initial_states = context_lens_tensor > 0

        # 仅保留 prefill 请求的 query_start_loc（slice 到 num_prefills+1）
        query_start_loc = query_start_loc[: num_prefills + 1]

        # 构建并返回混合模式的元数据
        return BailingLinearMetadata(
            batch_size=forward_batch.batch_size,
            query_start_loc=query_start_loc,
            mamba_cache_indices=mamba_cache_indices,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            has_initial_states=has_initial_states,
            q_lengths=query_start_loc.diff(),  # 每个 prefill 序列的 Query 长度
        )
