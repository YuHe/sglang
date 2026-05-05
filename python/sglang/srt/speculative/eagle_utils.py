import math
from enum import IntEnum
from typing import List, Optional

import torch

# 导入硬件平台检测工具，用于区分 CUDA / HIP / MUSA / NPU 环境
from sglang.srt.utils import is_cuda, is_hip, is_musa, is_npu

# 在模块加载时一次性检测硬件平台，避免重复调用
_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_musa = is_musa()

# 仅在 CUDA / HIP / MUSA 设备上导入高效树构建内核
if _is_cuda or _is_hip or _is_musa:
    from sgl_kernel import (
        build_tree_kernel_efficient as sgl_build_tree_kernel_efficient,
    )


def organize_draft_results(
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    num_draft_token: int,
):
    # 将多层草稿分数拼接为 [batch, total_candidates] 形状并展平，便于 topk 筛选
    score_list = torch.cat(score_list, dim=1).flatten(1)
    # 将多层草稿 token 拼接为 [batch, total_candidates] 形状
    ss_token_list = torch.cat(token_list, dim=1)
    # 从所有候选中选出得分最高的 (num_draft_token-1) 个草稿 token
    top_scores = torch.topk(score_list, num_draft_token - 1, dim=-1)
    top_scores_index = top_scores.indices
    # 对选出的索引排序，保证草稿 token 按树遍历顺序排列
    top_scores_index = torch.sort(top_scores_index).values
    # 根据排序后的索引从候选序列中取出对应的草稿 token
    draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

    # 拼接父节点列表（去掉最后一层，因为最后一层无子节点需要父指针）
    if len(parents_list) > 1:
        parent_list = torch.cat(parents_list[:-1], dim=1)
    else:
        # 只有一层时，父节点列表为空张量
        batch_size = parents_list[0].shape[0]
        parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

    return parent_list, top_scores_index, draft_tokens


# 树掩码模式枚举：控制 Attention 树掩码的计算与存储方式
class TreeMaskMode(IntEnum):
    FULL_MASK = 0                # 完整掩码：记录每个草稿 token 对所有历史 token 的注意力模式
    QLEN_ONLY = 1                # 仅记录 Q 长度维度的掩码（节省显存）
    QLEN_ONLY_BITPACKING = 2     # 仅 Q 长度维度 + 位压缩（进一步节省显存）


def build_tree_kernel_efficient(
    verified_id: torch.Tensor,
    parent_list: List[torch.Tensor],
    top_scores_index: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_sum: int,
    topk: int,
    spec_steps: int,
    num_verify_tokens: int,
    tree_mask_mode: TreeMaskMode = TreeMaskMode.FULL_MASK,
    tree_mask_buf: Optional[torch.Tensor] = None,
    position_buf: Optional[torch.Tensor] = None,
):
    # 将已验证的 token ID 拼接到草稿序列头部，构成完整的验证输入序列
    draft_tokens = torch.cat((verified_id.unsqueeze(1), draft_tokens), dim=1).flatten()

    # seq_lens_sum == sum(seq_lens); seq_lens: sequence length without draft tokens
    bs = seq_lens.numel()
    device = seq_lens.device
    # e.g. for bs=1, tree_mask: num_draft_token, seq_lens_sum + num_draft_token (flattened)
    # where each row indicates the attending pattern of each draft token
    # if use_partial_packed_tree_mask is True, tree_mask: num_draft_token (flattened, packed)
    if tree_mask_buf is not None:
        # 使用预分配的缓冲区，根据掩码模式进行初始化，避免重复分配显存
        tree_mask = tree_mask_buf
        if tree_mask_mode == TreeMaskMode.QLEN_ONLY:
            tree_mask.fill_(True)
        elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
            tree_mask.fill_(0)
        elif tree_mask_mode == TreeMaskMode.FULL_MASK:
            tree_mask.fill_(True)
        else:
            raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY:
        # 仅 Q 长度维度的布尔掩码，形状为 [bs * num_verify_tokens * num_verify_tokens]
        tree_mask = torch.full(
            (num_verify_tokens * bs * num_verify_tokens,),
            True,
            dtype=torch.bool,
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
        # 位压缩模式：根据 num_verify_tokens 自动选择 uint8/uint16/uint32 数据类型
        packed_dtypes = [torch.uint8, torch.uint16, torch.uint32]
        packed_dtype_idx = int(math.ceil(math.log2((num_verify_tokens + 7) // 8)))
        tree_mask = torch.zeros(
            (num_verify_tokens * bs,),
            dtype=packed_dtypes[packed_dtype_idx],
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.FULL_MASK:
        # 完整掩码：覆盖历史 token 与所有草稿 token 的注意力关系
        tree_mask = torch.full(
            (
                seq_lens_sum * num_verify_tokens
                + num_verify_tokens * num_verify_tokens * bs,
            ),
            True,
            device=device,
        )
    else:
        raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")

    # TODO: make them torch.empty and fuse them into `sgl_build_tree_kernel`
    # retrieve_buf 存储树遍历所需的三个辅助数组，初始化为 -1 表示"无效"
    retrieve_buf = torch.full(
        (3, bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    # retrieve_index: 每个草稿 token 在展平序列中的索引
    # retrieve_next_token: 同层下一个兄弟节点的 token
    # retrieve_next_sibling: 同层下一个兄弟节点的位置
    retrieve_index, retrieve_next_token, retrieve_next_sibling = retrieve_buf
    # position: where each token belongs to
    # e.g. if depth of each draft token is [0, 1, 1, 2] and the prompt length is 7
    # then, positions = [7, 8, 8, 9]
    if position_buf is not None:
        # 使用预分配的位置缓冲区，减少显存分配开销
        positions = position_buf
    else:
        positions = torch.empty(
            (bs * num_verify_tokens,), device=device, dtype=torch.long
        )

    # 根据硬件平台调用对应的树构建内核
    if _is_npu:
        torch.ops.npu.build_tree_kernel_efficient(
            parent_list.to(dtype=torch.int64),
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    else:
        # CUDA / HIP / MUSA 路径：调用 sgl_kernel 的高效树构建内核
        sgl_build_tree_kernel_efficient(
            parent_list,
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    return (
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        draft_tokens,
    )


def verify_tree_greedy_func(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
    topk: int = -1,
):
    # 贪心验证树结构中的草稿 token：逐层比对草稿与目标模型的预测，确定可接受的最长前缀
    if _is_cuda or _is_hip or _is_musa:
        from sgl_kernel import verify_tree_greedy

        # 调用 CUDA/HIP/MUSA 内核进行树验证，结果就地写入可变参数
        verify_tree_greedy(
            predicts=predicts,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=accept_token_num,  # mutable
            candidates=candidates,
            # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            target_predict=target_predict,
        )

    elif _is_npu:
        # NPU 路径：使用专用的 NPU 验证内核
        from sgl_kernel_npu.sample.verify_tree_greedy import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            target_predict=target_predict,
        )
    # 返回更新后的预测结果、接受索引和每个请求接受的 token 数量
    return predicts, accept_index, accept_token_num
