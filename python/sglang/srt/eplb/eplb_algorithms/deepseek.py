# This file is copied from https://github.com/deepseek-ai/EPLB/blob/main/eplb.py since that one is not a pypi package
# DeepSeek EPLB 核心算法：负载均衡装箱、专家复制与分层再平衡
from typing import Tuple

import torch


def balanced_packing(
    weight: torch.Tensor, num_packs: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly n/m objects and the weights of all packs
    are as balanced as possible.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    # 提取层数和分组数，并校验分组数可被目标 pack 数整除
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    # 每个 pack 需要容纳的 group 数量
    groups_per_pack = num_groups // num_packs

    # 特殊情况：每个 pack 只有一个 group，直接顺序分配，无需排序
    if groups_per_pack == 1:
        pack_index = torch.arange(
            weight.size(-1), dtype=torch.int64, device=weight.device
        ).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    # 按权重降序排列各 group，得到贪心分配的访问顺序（负载最高的 group 优先分配）
    indices_list = weight.float().sort(-1, descending=True).indices.tolist()
    weight_list = weight.tolist()
    # 初始化结果列表，-1 表示尚未分配
    pack_index_list = [[-1] * num_groups for _ in range(num_layers)]
    rank_in_pack_list = [[-1] * num_groups for _ in range(num_layers)]
    for i in range(num_layers):
        # pack_weights[j] 记录第 j 个 pack 当前的总权重，用于贪心最优选择
        pack_weights = [0] * num_packs
        # pack_items[j] 记录第 j 个 pack 已分配的 group 数
        pack_items = [0] * num_packs
        # 按权重从大到小遍历 group，将每个 group 分配到当前总权重最小的未满 pack
        for group in indices_list[i]:
            pack = min(
                (j for j in range(num_packs) if pack_items[j] < groups_per_pack),
                key=pack_weights.__getitem__,
            )
            assert pack_items[pack] < groups_per_pack
            # 记录该 group 被分配到哪个 pack 及其在 pack 内的位置（rank）
            pack_index_list[i][group] = pack
            rank_in_pack_list[i][group] = pack_items[pack]
            pack_weights[pack] += weight_list[i][group]
            pack_items[pack] += 1
    # 将 Python 列表转回张量，保持在 CPU 上
    pack_index = torch.tensor(pack_index_list, dtype=torch.int64, device="cpu")
    rank_in_pack = torch.tensor(rank_in_pack_list, dtype=torch.int64, device="cpu")
    return pack_index, rank_in_pack


def replicate_experts(
    weight: torch.Tensor, num_phy: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas, such that the maximum load of all replicas is minimized.

    Parameters:
        weight: [X, num_log]
        num_phy: total number of experts after replication

    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], the replica rank
        logcnt: [X, num_log], number of replicas for each logical expert
    """
    # 提取批次数（层数）和逻辑专家数
    n, num_log = weight.shape
    # 冗余副本数 = 物理专家数 - 逻辑专家数（需复制的热专家数量）
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    # 初始化 phy2log：前 num_log 个物理专家与逻辑专家一一对应
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(n, 1)
    # rank：每个物理专家在其对应逻辑专家所有副本中的序号（0 表示原始）
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    # logcnt：每个逻辑专家当前拥有的副本数，初始为 1
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)
    # 贪心地为冗余槽位选择最需要复制的逻辑专家（按 weight/副本数 的最大值）
    for i in range(num_log, num_phy):
        # 选择当前每副本平均负载最高的逻辑专家作为热专家进行复制
        redundant_indices = (weight / logcnt).max(dim=-1).indices
        # 将该物理槽位指向选中的逻辑专家
        phy2log[:, i] = redundant_indices
        # 新副本的 rank = 当前副本数（即插入前的副本计数）
        rank[:, i] = logcnt[arangen, redundant_indices]
        # 更新副本计数
        logcnt[arangen, redundant_indices] += 1
    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,              # 专家分组数；设为 1 时退化为全局均衡
    num_nodes: int,               # 节点数；设为 1 时退化为全局均衡
    num_gpus: int,
):
    """
    Parameters:
        weight: [num_moe_layers, num_logical_experts]
        num_physical_experts: number of physical experts after replication
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [num_moe_layers, num_physical_experts]
        logical_to_physical_map: [num_moe_layers, num_logical_experts, X]
        logical_count: [num_moe_layers, num_logical_experts]
    """
    # 提取层数和逻辑专家总数，并校验各维度整除关系
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    # 每个 group 内的逻辑专家数
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    # 每个节点分配到的 group 数
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    # 每张 GPU 持有的物理专家数
    phy_experts_per_gpu = num_physical_experts // num_gpus

    # 辅助函数：计算排列的逆排列（将 perm[i]=j 变为 inv[j]=i）
    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv.scatter_(
            1,
            perm,
            torch.arange(perm.size(1), dtype=torch.int64, device=perm.device).expand(
                perm.shape
            ),
        )
        return inv

    # Step 1: pack groups to nodes
    # 计算每个 group 的总 token 数（在 group_size 维度上求和），用于节点间负载均衡
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    # 将各 group 均衡地打包到各节点，得到每个 group 归属的节点编号及在节点内的排名
    group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group, num_nodes)
    # 构建 log2mlog 映射：逻辑专家索引 → 重排后的节点局部逻辑专家索引（mlog）
    # mlog 是将专家按节点分组后的线性编号
    log2mlog = (
        (
            (group_pack_index * groups_per_node + group_rank_in_pack) * group_size
        ).unsqueeze(-1)
        + torch.arange(group_size, dtype=torch.int64, device=group_pack_index.device)
    ).flatten(-2)
    # mlog2log 是 log2mlog 的逆映射（节点局部索引 → 全局逻辑专家索引）
    mlog2log = inverse(log2mlog)

    # Step 2: construct redundant experts within nodes
    # [num_layers * num_nodes, num_logical_experts // num_nodes]
    # 将权重按节点局部顺序重排，然后展平为 (num_layers * num_nodes, per_node_experts) 形状
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes
    )
    # 在每个节点内部，对逻辑专家进行副本分配（热专家复制策略）
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes
    )

    # Step 3: pack physical_experts to GPUs
    # [num_layers * num_nodes, num_physical_experts // num_nodes]
    # 计算每个物理副本的实际负载（原始权重 / 副本数）
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    # 将节点内的物理副本均衡装箱到各 GPU
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy, num_gpus // num_nodes)
    # phy2pphy：物理副本索引 → 打包后的 GPU 本地物理专家索引
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    # pphy2phy：逆映射（GPU 本地索引 → 原物理副本全局索引）
    pphy2phy = inverse(phy2pphy)

    # 将打包后的物理专家索引映射回节点局部逻辑专家索引
    pphy2mlog = phy2mlog.gather(
        -1, pphy2phy
    )  # [num_layers * num_nodes, num_log_per_nodes]
    # 加上节点偏移量，将节点局部 mlog 索引转为全局 mlog 索引
    pphy2mlog = (
        pphy2mlog.view(num_layers, num_nodes, -1)
        + torch.arange(
            0,
            num_logical_experts,
            num_logical_experts // num_nodes,
            device=group_pack_index.device,
        ).view(1, -1, 1)
    ).flatten(-2)
    # pphy2log：最终物理专家到逻辑专家的映射（经过节点打包和全局映射）
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    # pphyrank：每个物理专家副本的 rank（在同一逻辑专家的所有副本中的序号）
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    # logcnt：每个逻辑专家的副本数（从节点局部视角映射回全局逻辑专家顺序）
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
    return pphy2log, pphyrank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
    enable_hierarchical: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point for expert-parallelism load balancer.

    Parameters:
        weight: [layers, num_logical_experts], the load statistics for all logical experts
        num_replicas: number of physical experts, must be a multiple of `num_gpus`
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [layers, num_replicas], the expert index of each replica
        logical_to_physical_map: [layers, num_logical_experts, X], the replica indices for each expert
        expert_count: [layers, num_logical_experts], number of physical replicas for each logical expert
    """

    # 提取维度，并将权重转为 float32 CPU 张量
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    if enable_hierarchical:
        # use hierarchical load-balance policy
        # 启用分层：先节点间均衡 groups，再节点内复制热专家，再在 GPU 间装箱
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus
        )
    else:
        # use global load-balance policy
        # 全局均衡：不分节点和 group，直接在所有 GPU 上统一分配（传入 num_groups=1, num_nodes=1）
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus
        )
    # 所有逻辑专家中的最大副本数，决定 log2phy 的第三维度大小
    maxlogcnt = logcnt.max().item()
    # 初始化逻辑→物理映射表，默认 -1 表示该槽位未使用
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    # scatter_ 填充：将物理副本的全局索引写入 log2phy 的对应位置
    # 目标位置 = phy2log[i] * maxlogcnt + phyrank[i]（逻辑专家展平后的偏移）
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(num_replicas, dtype=torch.int64, device=log2phy.device).expand(
            num_layers, -1
        ),
    )
    # 返回三张映射表：物理→逻辑、逻辑→物理、每逻辑专家副本数
    return phy2log, log2phy, logcnt


__all__ = ["rebalance_experts"]
