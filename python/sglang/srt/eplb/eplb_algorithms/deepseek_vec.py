# This file is copied from https://github.com/deepseek-ai/EPLB/blob/main/eplb.py since that one is not a pypi package
# DeepSeek 向量化 EPLB 算法：保留每步（step）维度的 token 分布信息，
# 支持以 chunk 为单位分配冗余专家，并在 GPU 间做二次负载均衡
from typing import Optional, Tuple

import torch


def pack_groups(tokens_per_group: torch.Tensor, num_nodes: int) -> torch.Tensor:
    # 将专家 group 按 token 总量均衡地分配到各节点
    # tokens_per_group: [num_layers, num_groups]
    num_layers, num_groups = tokens_per_group.shape
    assert num_groups % num_nodes == 0
    # 每个节点分到的 group 数
    groups_per_rank = num_groups // num_nodes

    # 按每层各 group 的 token 总量降序排列，贪心分配时负载最重的 group 优先处理
    indices = tokens_per_group.float().sort(-1, descending=True).indices.cpu()
    # ret[layer, group] = 该 group 在分配后的节点局部线性编号
    ret = torch.full_like(
        tokens_per_group, fill_value=-1, dtype=torch.int64, device="cpu"
    )
    for layer in range(num_layers):
        # node_tokens[r]：第 r 个节点当前的总 token 负载（用于贪心最优选择）
        node_tokens = [0] * num_nodes
        # node_groups[r]：第 r 个节点已分配的 group 数
        node_groups = [0] * num_nodes
        for group in indices[layer]:
            # 优先选择未满（已分配 group 数 < groups_per_rank）且当前 token 最少的节点
            def key_func(rank: int) -> int:
                if node_groups[rank] >= groups_per_rank:
                    # 节点已满，设置最低优先级
                    return 1, 0
                else:
                    return 0, node_tokens[rank]

            rank = min(range(num_nodes), key=key_func)
            assert node_groups[rank] < groups_per_rank
            # 节点局部编号 = rank * groups_per_rank + 节点内位置
            ret[layer, group] = rank * groups_per_rank + node_groups[rank]
            node_tokens[rank] += tokens_per_group[layer, group]
            node_groups[rank] += 1
    return ret


def make_redundant_experts_chunkwise(
    tokens_per_expert: torch.Tensor,       # [num_steps, num_moe_layers, num_logical_experts]
    num_physical_experts: int,             # 物理专家总数（含冗余副本）
    num_local_physical_experts: int,       # 每张 GPU 本地的物理专家数
    num_physical_experts_per_chunk: int,   # 每个 chunk 包含的物理专家数（分层时 = 每节点物理专家数）
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 提取维度信息
    num_steps, num_moe_layers, num_logical_experts = tokens_per_expert.shape
    # 冗余副本数 = 物理专家总数 - 逻辑专家数
    num_redundancy_experts = num_physical_experts - num_logical_experts

    # 初始化物理→逻辑映射（每个物理专家槽位对应哪个逻辑专家）
    physical_to_logical_map = torch.empty(
        num_moe_layers,
        num_physical_experts,
        dtype=torch.int,
        device=tokens_per_expert.device,
    )
    # 初始化逻辑→物理映射（每个逻辑专家对应哪些物理副本），默认 -1 表示无效槽位
    logical_to_physical_map = torch.full(
        (num_moe_layers, num_logical_experts, num_redundancy_experts + 1),
        -1,
        dtype=torch.int,
        device=tokens_per_expert.device,
    )
    # 初始化每个逻辑专家的副本计数，初始为 1（原始副本）
    logical_count = torch.ones(
        num_moe_layers,
        num_logical_experts,
        dtype=torch.int,
        device=tokens_per_expert.device,
    )

    # 计算 chunk 相关维度
    assert num_physical_experts % num_physical_experts_per_chunk == 0
    num_chunks = num_physical_experts // num_physical_experts_per_chunk
    assert num_logical_experts % num_chunks == 0
    # 每个 chunk 对应的逻辑专家数
    num_logical_experts_per_group = num_logical_experts // num_chunks
    assert num_redundancy_experts % num_chunks == 0
    # 每个 chunk 对应的冗余副本数
    num_redundancy_experts_per_group = num_redundancy_experts // num_chunks

    # 预计算常用 arange 索引张量，避免循环中重复创建
    arange_num_moe_layers_num_groups = torch.arange(
        num_moe_layers * num_chunks, dtype=torch.int, device=tokens_per_expert.device
    )
    arange_num_logical_experts = torch.arange(
        num_logical_experts, dtype=torch.int, device=tokens_per_expert.device
    )
    arange_num_logical_experts_per_group = torch.arange(
        num_logical_experts_per_group, dtype=torch.int, device=tokens_per_expert.device
    )
    arange_num_groups = torch.arange(
        num_chunks, dtype=torch.int, device=tokens_per_expert.device
    )
    # 初始化每个 chunk 前半段（逻辑专家原始槽位）的物理→逻辑映射
    physical_to_logical_map.view(
        num_moe_layers, num_chunks, num_physical_experts_per_chunk
    )[:, :, :num_logical_experts_per_group] = arange_num_logical_experts.view(
        num_chunks, num_logical_experts_per_group
    )
    # 初始化逻辑→物理映射的第 0 个副本槽位（原始副本的物理位置）
    logical_to_physical_map[:, :, 0] = (
        arange_num_logical_experts_per_group.expand(
            num_chunks, num_logical_experts_per_group
        )
        + arange_num_groups[:, None] * num_physical_experts_per_chunk
    ).view(num_logical_experts)

    # 加入极小扰动，保证同一层内各逻辑专家的 score 值互不相同，避免 argmax 随机打破平局
    tokens_per_expert_all_diff = tokens_per_expert + arange_num_logical_experts * 1e-4
    # 逐个冗余槽位分配热专家（贪心策略：选择分配后能使所有 chunk 最大负载下降最多的逻辑专家）
    for i in range(num_redundancy_experts_per_group):
        # score：当前每副本平均负载（越高越需要再复制）
        score = (
            tokens_per_expert_all_diff / logical_count
        )  # NOTE: Values in score must be different from each other
        # score1：复制后每副本的预期负载（用于在 argmin 前替换当前 max 位置的分数）
        score1 = tokens_per_expert / (logical_count + 1)
        score = score.view(
            num_steps, num_moe_layers, num_chunks, num_logical_experts_per_group
        )
        score1 = score1.view_as(score)
        # 找到每个 chunk 内 score 最大的逻辑专家（候选热专家）
        values, indices = score.max(-1, keepdim=True)
        values = values.expand_as(score).contiguous()
        # 用 score1 替换 values 中候选热专家的位置，再对更新后的 values 求新 max
        score.scatter_(-1, indices, score1.gather(-1, indices))
        values.scatter_(-1, indices, score.max(-1, keepdim=True).values)
        # 跨所有 step 求和后，选出使最大 chunk 负载最小的逻辑专家
        redundancy_indices = values.sum(0).argmin(-1)
        # 将冗余副本槽位的 phy2log 指向选出的热专家
        physical_to_logical_map.view(
            num_moe_layers, num_chunks, num_physical_experts_per_chunk
        )[:, :, num_logical_experts_per_group + i] = (
            redundancy_indices + arange_num_groups * num_logical_experts_per_group
        )
        # 读取被选热专家当前的副本计数（用于确定在 log2phy 中写入的槽位位置）
        redundancy_count = (
            logical_count.view(
                num_moe_layers * num_chunks, num_logical_experts_per_group
            )
            .gather(-1, redundancy_indices.view(num_moe_layers * num_chunks, 1))
            .squeeze(1)
        )
        # 计算本轮新增物理副本的全局物理索引
        physical_redundancy_indices = (
            (
                arange_num_groups * num_physical_experts_per_chunk
                + num_logical_experts_per_group
                + i
            )
            .expand(num_moe_layers, num_chunks)
            .flatten()
        )
        # 将物理副本索引写入逻辑→物理映射的对应位置
        logical_to_physical_map.view(
            num_moe_layers * num_chunks,
            num_logical_experts_per_group,
            num_redundancy_experts + 1,
        )[
            arange_num_moe_layers_num_groups,
            redundancy_indices.view(num_moe_layers * num_chunks),
            redundancy_count,
        ] = physical_redundancy_indices
        # 更新被选热专家的副本计数 +1
        logical_count.view(num_moe_layers * num_chunks, num_logical_experts_per_group)[
            arange_num_moe_layers_num_groups,
            redundancy_indices.view(num_moe_layers * num_chunks),
        ] += 1

    if num_local_physical_experts > 1:
        # Load-balancing between GPUs
        # 当每 GPU 本地专家数 > 1 时，做 GPU 间二次负载均衡：按副本平均负载排序并重排物理槽位
        physical_to_logical_map_int64 = physical_to_logical_map.to(torch.int64)
        # 读取每个物理副本对应逻辑专家的副本数
        counts = logical_count.gather(-1, physical_to_logical_map_int64)
        # 计算每个物理副本的预期负载（各 step 汇总后 / 副本数）
        score = tokens_per_expert.sum(0).gather(-1, physical_to_logical_map_int64)
        score = score / counts
        # 在每个 chunk 内按负载从高到低排序，得到重排索引
        score = score.view(num_moe_layers, num_chunks, num_physical_experts_per_chunk)
        indices = score.argsort(-1, descending=True)
        # 将 chunk 内相对索引转为全局物理专家索引
        indices += torch.arange(
            0,
            num_physical_experts,
            num_physical_experts_per_chunk,
            dtype=indices.dtype,
            device=indices.device,
        )[None, :, None]

        # 将 chunk 内的物理专家按 GPU 本地分组，交替翻转奇数组顺序以实现蛇形分配
        assert num_physical_experts_per_chunk % num_local_physical_experts == 0
        num_local_groups = num_physical_experts_per_chunk // num_local_physical_experts
        indices = indices.view(
            num_moe_layers, num_chunks, num_local_physical_experts, num_local_groups
        )
        # 奇数 GPU 组内顺序翻转（蛇形排列），使相邻 GPU 的负载更均衡
        indices[:, :, 1::2, :] = indices[:, :, 1::2, :].flip(-1)
        # 转置后展平为最终的物理专家重排索引
        indices = indices.transpose(2, 3)
        indices = indices.reshape(num_moe_layers, num_physical_experts)
        # 按新顺序重排 physical_to_logical_map
        physical_to_logical_map = physical_to_logical_map.gather(-1, indices)
        # 同步更新 logical_to_physical_map：将旧物理索引替换为重排后的新索引
        mask = logical_to_physical_map == -1
        logical_to_physical_map[mask] = 0  # 临时填 0 避免 gather 越界
        logical_to_physical_map = (
            indices.argsort(-1)
            .gather(
                -1, logical_to_physical_map.view(num_moe_layers, -1).to(torch.int64)
            )
            .view_as(logical_to_physical_map)
            .to(torch.int)
        )
        # 恢复无效槽位的 -1 标记
        logical_to_physical_map[mask] = -1

    return physical_to_logical_map, logical_to_physical_map, logical_count


def decode_rebalance_experts(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int,
):
    # Decode 场景：不分层，将所有物理专家视为单个 chunk 进行全局冗余分配
    return make_redundant_experts_chunkwise(
        tokens_per_expert,
        num_physical_experts,
        num_local_physical_experts,
        num_physical_experts,  # 单 chunk = 全部物理专家
    )


def prefill_rebalance_experts(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int,
    num_groups: int,
    num_nodes: int,
):
    # Prefill 场景：先将专家 group 按 token 量均衡分配到节点（节点间负载均衡），
    # 再在节点内用 chunk 方式分配冗余副本（节点内负载均衡）
    tokens_per_expert = tokens_per_expert.float().cpu()

    num_steps, _, num_logical_experts = tokens_per_expert.shape
    assert num_logical_experts % num_groups == 0
    # 每个 group 内的逻辑专家数
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0, f"{num_groups=} {num_nodes=}"

    # 计算各 group 在所有 step 上的累计 token 量，用于节点间均衡
    tokens_per_group = tokens_per_expert.sum(0).unflatten(-1, (num_groups, -1)).sum(-1)
    group_perm = pack_groups(
        tokens_per_group, num_nodes
    )  # [num_moe_layers, num_groups] => [num_moe_layers, num_nodes]

    # log2mlog [layers, #logexp] -> [layers, #logexp]
    # 构建逻辑专家 → 节点局部重排索引（mlog）的映射
    log2mlog = (
        (group_perm * group_size).unsqueeze(-1)
        + torch.arange(group_size, dtype=torch.int64, device=group_perm.device)
    ).flatten(-2)

    # mlog2log [layers, #logexp] -> [layers, #logexp], inverse of log2mlog
    # mlog2log 是 log2mlog 的逆映射（节点局部索引 → 全局逻辑专家索引）
    mlog2log = torch.empty_like(log2mlog)
    arange = torch.arange(
        num_logical_experts, dtype=torch.int64, device=mlog2log.device
    )
    mlog2log.scatter_(1, log2mlog, arange.expand(log2mlog.size(0), -1))

    # tokens_per_mlog[i][j][k] = tokens_per_expert[i][j][mlog2log[j][k]]
    # 将 token 分布按节点局部顺序重排
    tokens_per_mlog = tokens_per_expert.gather(
        2, mlog2log.unsqueeze(0).expand(num_steps, -1, -1)
    )

    # 在节点局部视图上，以每节点物理专家数为 chunk 进行冗余副本分配
    phy2mlog, mlog2phy, mlog_count = make_redundant_experts_chunkwise(
        tokens_per_mlog,
        num_physical_experts,
        num_local_physical_experts,
        num_physical_experts // num_nodes,  # 每个 chunk = 每个节点的物理专家数
    )

    # phy2log[i][j] = mlog2log[i][phy2mlog[i][j]]
    # 将节点局部的 phy2mlog 映射转换回全局逻辑专家索引
    phy2log = mlog2log.gather(1, phy2mlog.to(torch.int64))

    # mlog2phy: [num_moe_layers, num_logical_experts, ...]
    # log2phy[i][j][k] = mlog2phy[i][log2mlog[i][j]][k]
    # 将节点局部的 log2phy 映射按 log2mlog 重排回全局逻辑专家顺序
    log2phy = mlog2phy.gather(
        1, log2mlog.unsqueeze(-1).expand(-1, -1, mlog2phy.size(-1)).to(torch.int64)
    )

    # log_count[i][j] = mlog_count[i][log2mlog[i][j]]
    # 将节点局部副本计数映射回全局逻辑专家顺序
    log_count = mlog_count.gather(1, log2mlog)
    return phy2log, log2phy, log_count


def rebalance_experts(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int,
    num_groups: Optional[int],   # 专家分组数（prefill 分层时使用）
    num_nodes: int,
    enable_hierarchical: bool,   # True=prefill 分层策略，False=decode 全局策略
):
    # 根据是否启用分层，选择 prefill 或 decode 再平衡策略
    if enable_hierarchical:
        return prefill_rebalance_experts(
            tokens_per_expert=tokens_per_expert,
            num_physical_experts=num_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
            num_groups=num_groups,
            num_nodes=num_nodes,
        )
    else:
        # decode 模式：不需要 num_groups 和 num_nodes，全局均衡
        return decode_rebalance_experts(
            tokens_per_expert=tokens_per_expert,
            num_physical_experts=num_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
        )
