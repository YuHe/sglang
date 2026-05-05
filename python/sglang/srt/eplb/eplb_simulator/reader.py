# EPLB 模拟器数据读取模块
# 负责从 ExpertDistributionRecorder 生成的 .pt 文件中还原专家分布统计数据
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

# 导入物理专家计数转逻辑专家计数的工具函数
from sglang.srt.eplb.expert_distribution import (
    _convert_global_physical_count_to_logical_count,
)

# 将内部函数以公开别名暴露，供外部模块直接调用
convert_global_physical_count_to_logical_count = (
    _convert_global_physical_count_to_logical_count
)


def read_mode_per_pass(dir_data: Path):
    """Read data from ExpertDistributionRecorder when recorded with mode `per_pass`"""

    # gpc := global_physical_count（全局物理专家 token 处理计数）
    # 以 forward_pass_id -> rank -> gpc_tensor 的嵌套字典结构暂存数据
    gpc_of_forward_pass_and_rank = defaultdict(lambda: defaultdict())
    # 遍历目录下所有 .pt 文件（每个文件对应一批记录数据包）
    for path in tqdm(list(dir_data.glob("*.pt"))):
        data_pack = torch.load(path, weights_only=True)
        # 保存最后一次记录的物理专家到逻辑专家映射表
        last_physical_to_logical_map = data_pack["last_physical_to_logical_map"]
        # 遍历数据包中的每条前向传播记录
        for record in data_pack["records"]:
            forward_pass_id = record["forward_pass_id"]
            rank = record["rank"]
            # 校验：同一 forward_pass_id + rank 不应出现重复记录
            assert (
                gpc_of_forward_pass_and_rank[forward_pass_id].get(rank) is None
            ), f"Duplicated {forward_pass_id=} {rank=}"
            # 存储该 rank 在该前向传播中每个物理专家的 token 计数向量
            gpc_of_forward_pass_and_rank[forward_pass_id][rank] = record[
                "global_physical_count"
            ]

    # 对 forward_pass_id 排序，确保后续处理顺序一致
    forward_pass_ids = sorted(gpc_of_forward_pass_and_rank.keys())
    print(f"Make {forward_pass_ids=} into array")

    items = []
    # 按 forward_pass_id 顺序处理每次前向传播的多 rank 数据
    for forward_pass_id, gpc_of_rank in sorted(gpc_of_forward_pass_and_rank.items()):
        # 将各 rank 的 gpc 张量堆叠后在 rank 维度求和，得到全局物理专家负载
        gpc_of_rank_tensor = torch.stack(
            [gpc for rank, gpc in sorted(gpc_of_rank.items())]
        ).sum(dim=0)
        items.append(gpc_of_rank_tensor)

    # 将所有前向传播的全局物理计数堆叠为二维张量 [num_passes, num_physical_experts]
    gpc_of_forward_pass = torch.stack(items)
    print(f"{gpc_of_forward_pass.shape=}")

    # 返回包含全局物理计数、物理到逻辑映射及 pass ID 列表的字典
    return dict(
        global_physical_count_of_forward_pass=gpc_of_forward_pass,
        last_physical_to_logical_map=last_physical_to_logical_map,
        forward_pass_ids=forward_pass_ids,
    )
