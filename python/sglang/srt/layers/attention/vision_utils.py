"""Utility functions for vision attention layers."""
# 视觉注意力层工具函数模块

import torch

# 导入张量并行注意力尺寸获取函数
from sglang.srt.layers.dp_attention import get_attention_tp_size


def update_vit_attn_dummy_heads_config(config):
    """Update HF config to ensure vision attention num_attention_heads is divisible by tp_size"""
    # 更新视觉编码器配置，确保注意力头数能被张量并行尺寸整除，不足时补充虚拟头
    tp_size = get_attention_tp_size()  # 获取当前张量并行尺寸
    # 兼容两种字段名：num_heads 或 num_attention_heads
    num_heads = getattr(
        config.vision_config,
        "num_heads",
        getattr(config.vision_config, "num_attention_heads", None),
    )
    # 计算每个注意力头的维度
    head_dim = config.vision_config.hidden_size // num_heads
    num_dummy_heads = 0  # 初始化虚拟头数量为 0

    if num_heads % tp_size != 0:
        # 当头数不能被 tp_size 整除时，计算需要填充的虚拟头数
        num_dummy_heads = ((num_heads + tp_size - 1) // tp_size) * tp_size - num_heads

    # 将 head_dim 和 num_dummy_heads 写回配置，供后续权重填充使用
    setattr(config.vision_config, "head_dim", head_dim)
    setattr(config.vision_config, "num_dummy_heads", num_dummy_heads)


def pad_vit_attn_dummy_heads(config, name: str, loaded_weight: torch.Tensor):
    """Pad attention qkv weights for dummy heads"""
    # 对视觉注意力的 QKV 权重进行虚拟头填充，使权重形状与扩展后的头数对齐
    num_dummy_heads = config.vision_config.num_dummy_heads  # 需要填充的虚拟头数
    if num_dummy_heads == 0:
        # 无需填充时直接返回原始权重
        return loaded_weight
    head_dim = config.vision_config.head_dim  # 每头维度

    if "attn.qkv_proj" in name:
        # 融合 QKV 权重：先按 3 等分拆出 Q、K、V
        wq, wk, wv = loaded_weight.chunk(3, dim=0)
        if name.endswith(".weight"):
            dummy_shape = [num_dummy_heads, head_dim, wq.shape[-1]]  # 线性层权重填充形状
        elif name.endswith(".bias"):
            dummy_shape = [num_dummy_heads, head_dim]  # bias 填充形状
        else:
            raise RuntimeError(f"Unsupported weight with name={name}")
        # 将原始权重重排为 [num_heads, head_dim, ...] 后拼接零填充虚拟头，再展平
        pad_func = lambda x: torch.cat(
            [x.unflatten(0, (-1, head_dim)), x.new_zeros(dummy_shape)], dim=0
        ).flatten(0, 1)
        wq, wk, wv = pad_func(wq), pad_func(wk), pad_func(wv)
        # 将填充后的 Q、K、V 重新拼接为融合权重
        loaded_weight = torch.cat([wq, wk, wv], dim=0)
    elif any([_ in name for _ in ["attn.q_proj", "attn.k_proj", "attn.v_proj"]]):
        # 独立的 Q/K/V 投影权重填充
        if name.endswith(".weight"):
            dummy_shape = [num_dummy_heads, head_dim, loaded_weight.shape[-1]]
        elif name.endswith(".bias"):
            dummy_shape = [num_dummy_heads, head_dim]
        else:
            raise RuntimeError(f"Unsupported weight with name={name}")
        # 创建全零虚拟头权重并拼接到原权重之后
        padded_weight = loaded_weight.new_zeros(dummy_shape)
        loaded_weight = torch.cat(
            [loaded_weight.unflatten(0, (-1, head_dim)), padded_weight], dim=0
        ).flatten(0, 1)
    elif "attn.proj.weight" in name:
        # 输出投影权重：在输入维度（列方向）填充虚拟头对应的零列
        padded_weight = loaded_weight.new_zeros(
            loaded_weight.shape[0], head_dim * num_dummy_heads
        )
        loaded_weight = torch.cat([loaded_weight, padded_weight], dim=-1)
    elif "attn.q_norm.weight" in name or "attn.k_norm.weight" in name:
        # Q/K 归一化权重：在末尾拼接虚拟头的零权重
        padded_weight = loaded_weight.new_zeros(head_dim * num_dummy_heads)
        loaded_weight = torch.cat([loaded_weight, padded_weight], dim=0)
    return loaded_weight
