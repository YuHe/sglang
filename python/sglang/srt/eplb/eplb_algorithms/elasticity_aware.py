# 弹性感知专家负载均衡算法模块
# 在 DeepSeek 分层算法基础上，额外处理弹性 EP（部分 rank 离线）场景
from typing import Tuple

import torch

# 复用 DeepSeek 分层负载均衡的核心实现
from sglang.srt.eplb.eplb_algorithms.deepseek import rebalance_experts_hierarchical


def rebalance_experts(
    weight: torch.Tensor,          # [layers, num_logical_experts]，每层各逻辑专家的负载统计权重
    num_replicas: int,             # 物理专家（副本）总数，须为 num_gpus 的整数倍
    num_groups: int,               # 专家分组数（DeepSeek MoE 的 group_topk 分组）
    num_nodes: int,                # 服务器节点数（节点内 NVLink 带宽更高）
    num_gpus: int,                 # GPU 总数，须为 num_nodes 的整数倍
    enable_hierarchical: bool,     # 是否启用分层负载均衡（节点间 + 节点内两级）
    active_ranks: torch.Tensor,    # 布尔张量，标记哪些 rank 当前处于活跃（在线）状态
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

    # 提取层数和逻辑专家数，并将权重转为 float32 CPU 张量（算法在 CPU 上运行）
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    # 统计当前活跃 rank 数量（弹性场景下可能少于 num_gpus）
    num_active_ranks = active_ranks.sum().item()
    # 每张 GPU 本地持有的专家数（= 总物理专家数 / GPU 总数）
    num_local_experts = num_replicas // num_gpus
    if num_active_ranks < num_gpus:
        # Must fall back to global load-balance policy
        # and fix some params
        # 有 rank 离线：退化为全局均衡，只在活跃 rank 上分配专家副本
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight,
            num_local_experts * num_active_ranks,  # 只分配活跃 rank 所拥有的副本槽位
            1,              # 不分组（全局均衡）
            1,              # 视为单节点
            num_active_ranks,
        )
    elif enable_hierarchical:
        # use hierarchical load-balance policy
        # 所有 rank 均在线且启用分层：使用完整的节点间+节点内两级分层策略
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus
        )
    else:
        # use global load-balance policy
        # 所有 rank 在线但不分层：使用全局均衡（num_groups=1, num_nodes=1）
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus
        )
    # 最大副本数（任意逻辑专家最多被复制多少份），用于 log2phy 张量的第三维度
    maxlogcnt = logcnt.max().item()
    # 初始化逻辑专家到物理副本的映射表，默认填 -1 表示无效
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    # 利用 scatter_ 将物理副本索引写入 log2phy：
    # phy2log * maxlogcnt + phyrank 计算每个物理副本在展平后的 log2phy 中的目标位置
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(
            num_local_experts * num_active_ranks,  # 物理副本的全局索引（0 ~ N-1）
            dtype=torch.int64,
            device=log2phy.device,
        ).expand(num_layers, -1),
    )
    if num_active_ranks < num_gpus:
        # 弹性场景后处理：将活跃 rank 的 phy2log 切片按 rank 顺序重新插入，
        # 离线 rank 插入全零占位切片，保证 phy2log 与完整 GPU 编号对齐
        phy2log_slices = list(
            phy2log.view(num_layers, num_active_ranks, -1).unbind(dim=1)
        )
        active_ranks_list = active_ranks.tolist()
        for idx, active_rank in enumerate(active_ranks_list):
            if not active_rank:
                # 当前 rank 离线：插入零张量占位，表示该 GPU 上无有效专家分配
                phy2log_slices.insert(idx, torch.zeros_like(phy2log_slices[0]))
                # 同步更新 log2phy：将原先指向 idx 之后物理副本的索引向后偏移 num_local_experts
                log2phy = torch.where(
                    log2phy >= idx * num_local_experts,
                    log2phy + num_local_experts,
                    log2phy,
                )
        # 重新拼合各 rank 切片，恢复为 [layers, num_gpus * num_local_experts] 的完整形状
        phy2log = torch.stack(phy2log_slices, dim=1).contiguous().view(num_layers, -1)
    # 返回：物理→逻辑映射、逻辑→物理映射、每个逻辑专家的副本数
    return phy2log, log2phy, logcnt
