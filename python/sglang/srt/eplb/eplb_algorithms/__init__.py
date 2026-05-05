# EPLB（专家并行负载均衡）算法包的入口模块
# 提供算法枚举、专家再平衡接口和算法自动选择逻辑
from enum import Enum, auto
from typing import Optional

import torch

# 导入三种具体的专家再平衡算法实现
from sglang.srt.eplb.eplb_algorithms import deepseek, deepseek_vec, elasticity_aware


# EPLB 算法枚举：列出所有支持的负载均衡算法及其分层变体
class EplbAlgorithm(Enum):
    # DeepSeek 原版算法（基于 token 聚合权重进行专家副本分配）
    deepseek = auto()
    # DeepSeek 分层算法（先在节点间分配，再在节点内 GPU 间分配）
    deepseek_hierarchical = auto()
    # DeepSeek 向量化算法（保留每个 rank 维度的 token 分布信息）
    deepseek_vec = auto()
    # DeepSeek 向量化分层算法
    deepseek_vec_hierarchical = auto()
    # 弹性感知算法（考虑弹性 EP 中部分节点离线的情况）
    elasticity_aware = auto()
    # 弹性感知分层算法
    elasticity_aware_hierarchical = auto()
    # TODO may have more algorithm later


def rebalance_experts(
    tokens_per_expert: torch.Tensor,       # 每个物理专家在各 rank 上处理的 token 数，形状 [num_ranks, num_logical_experts]
    num_physical_experts: int,              # 物理专家总数（包含热专家的冗余副本）
    num_local_physical_experts: int,        # 每张 GPU 本地拥有的物理专家数
    num_groups: Optional[int],              # 专家分组数（DeepSeek MoE 中的分组概念）
    num_nodes: int,                         # 参与计算的节点（机器）数
    algorithm: EplbAlgorithm,              # 指定使用哪种负载均衡算法
):
    # --- DeepSeek 原版 / 分层算法分支 ---
    if algorithm in [EplbAlgorithm.deepseek, EplbAlgorithm.deepseek_hierarchical]:
        # 对所有 rank 的 token 数求和，得到每个逻辑专家的总负载权重
        return deepseek.rebalance_experts(
            weight=tokens_per_expert.sum(dim=0),
            num_replicas=num_physical_experts,
            num_groups=num_groups,
            num_nodes=num_nodes,
            # 总 GPU 数 = 物理专家总数 / 每 GPU 本地专家数
            num_gpus=num_physical_experts // num_local_physical_experts,
            enable_hierarchical=algorithm == EplbAlgorithm.deepseek_hierarchical,
        )

    # --- DeepSeek 向量化 / 向量化分层算法分支 ---
    if algorithm in [
        EplbAlgorithm.deepseek_vec,
        EplbAlgorithm.deepseek_vec_hierarchical,
    ]:
        # 向量化版本保留 per-rank 的 token 分布，能做更细粒度的负载均衡
        return deepseek_vec.rebalance_experts(
            tokens_per_expert=tokens_per_expert,
            num_physical_experts=num_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
            num_groups=num_groups,
            num_nodes=num_nodes,
            enable_hierarchical=algorithm == EplbAlgorithm.deepseek_vec_hierarchical,
        )

    # --- 弹性感知 / 弹性感知分层算法分支 ---
    if algorithm in [
        EplbAlgorithm.elasticity_aware,
        EplbAlgorithm.elasticity_aware_hierarchical,
    ]:
        # 延迟导入弹性 EP 状态管理器，避免循环依赖
        from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager

        return elasticity_aware.rebalance_experts(
            weight=tokens_per_expert.sum(dim=0),
            num_replicas=num_physical_experts,
            num_groups=num_groups,
            num_nodes=num_nodes,
            num_gpus=num_physical_experts // num_local_physical_experts,
            enable_hierarchical=(
                algorithm == EplbAlgorithm.elasticity_aware_hierarchical
            ),
            # 获取当前活跃的 rank 列表：优先从状态管理器实例获取，否则用健康状态默认值
            active_ranks=(
                ElasticEPStateManager.instance().active_ranks
                if ElasticEPStateManager.instance() is not None
                else ElasticEPStateManager.healthy_rank_state()
            ),
        )

    # 不支持的算法类型，抛出未实现异常
    raise NotImplementedError


def compute_algorithm(
    raw_algorithm: str,           # 用户配置的算法名称字符串，或 "auto" 表示自动选择
    num_groups: Optional[int],    # 专家分组数
    num_nodes: int,               # 节点数
) -> EplbAlgorithm:
    # 若用户明确指定了算法名称，则直接按名称查找枚举值
    if raw_algorithm != "auto":
        return EplbAlgorithm[raw_algorithm]

    # TODO test on real scenarios and know which ones perform better
    # 自动选择：当专家分组数能被节点数整除时，启用分层算法（节点内外两级负载均衡）
    if (num_groups is not None) and (num_groups % num_nodes == 0):
        return EplbAlgorithm.deepseek_hierarchical
    else:
        # 默认回退到 DeepSeek 基础算法
        return EplbAlgorithm.deepseek
