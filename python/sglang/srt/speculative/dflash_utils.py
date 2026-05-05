from __future__ import annotations

# DFlash（Drafting with Flash）辅助工具模块：配置解析、KV 预算、接受长度计算等
from dataclasses import dataclass
from numbers import Integral
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F

# 未量化线性层方法，用于判断 DFlash 是否可直接切片 QKV 权重
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
# 平台检测：is_cuda/is_musa 决定 sgl_kernel 的导入路径
from sglang.srt.utils import is_cuda, is_musa

# DFlash mask token 的默认字符串表示（用于推理时填充 mask 位置）
DEFAULT_DFLASH_MASK_TOKEN = "<|MASK|>"

# 非贪心采样验证核是否可用的全局标志（仅 CUDA/MUSA 且成功导入时为 True）
_DFLASH_SAMPLING_VERIFY_AVAILABLE = False
# 全局缓存：按 (device_index, draft_token_num) 键存放链式验证所需的 GPU 缓冲区
_DFLASH_CHAIN_VERIFY_BUFFERS: dict[tuple[Optional[int], int], dict[str, Any]] = {}
# 不需要自定义 causal mask 的 attention 后端名称集合（这些后端内置掩码生成）
_DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS = frozenset(
    {
        "FlashInferAttnBackend",
        "FlashInferMLAAttnBackend",
        "FlashAttentionBackend",
        "TRTLLMHAAttnBackend",
        "TRTLLMMLABackend",
    }
)


# 仅 CUDA/MUSA 平台才尝试导入 sgl_kernel 中的采样核
if is_cuda() or is_musa():
    try:
        from sgl_kernel import (
            top_k_renorm_prob,          # top-k 重归一化概率
            top_p_renorm_prob,          # top-p 重归一化概率
            tree_speculative_sampling_target_only,  # 仅目标模型的树形推测采样
        )

        # 成功导入则标记非贪心验证可用
        _DFLASH_SAMPLING_VERIFY_AVAILABLE = True
    except Exception:
        # 导入失败时降级为 None，后续调用会通过 _DFLASH_SAMPLING_VERIFY_AVAILABLE 检查
        top_k_renorm_prob = None
        top_p_renorm_prob = None
        tree_speculative_sampling_target_only = None
else:
    # 非 CUDA/MUSA 平台不支持这些核
    top_k_renorm_prob = None
    top_p_renorm_prob = None
    tree_speculative_sampling_target_only = None


def is_dflash_sampling_verify_available() -> bool:
    # 供外部查询非贪心采样验证核是否已成功加载
    return _DFLASH_SAMPLING_VERIFY_AVAILABLE


def scale_kv_cell_size_per_token_for_dflash(
    *,
    target_cell_size_per_token: int,      # 目标模型每个 token 占用的 KV 字节数
    target_num_layers: int,               # 目标模型 transformer 层数
    draft_num_layers: int,                # DFlash 草稿模型层数
    draft_cell_size_per_token: Optional[int] = None,  # 草稿模型每 token KV 字节数（可选）
) -> int:
    """Compute bytes/token budget for combined target+draft KV pools (DFLASH).

    DFLASH runs a separate draft runner with its own KV pool. The target runner's
    token capacity must fit both pools in aggregate.

    Returns:
        Approximate per-token bytes for (target KV + draft KV), expressed as a
        scaled version of `target_cell_size_per_token`, unless an explicit
        `draft_cell_size_per_token` is provided (in which case we sum them).
    """
    if target_cell_size_per_token <= 0:
        raise ValueError(
            "target_cell_size_per_token must be positive, "
            f"got {target_cell_size_per_token}."
        )

    # 如果明确给定了草稿 KV 大小，直接相加作为联合预算
    if draft_cell_size_per_token is not None:
        draft_cell_size_per_token = int(draft_cell_size_per_token)
        if draft_cell_size_per_token <= 0:
            raise ValueError(
                "draft_cell_size_per_token must be positive when provided, "
                f"got {draft_cell_size_per_token}."
            )
        return int(target_cell_size_per_token) + int(draft_cell_size_per_token)

    # 层数无效时直接返回目标模型 KV 大小（不缩放）
    if target_num_layers <= 0 or draft_num_layers <= 0:
        return int(target_cell_size_per_token)

    # 根据层数比例估算联合 KV 预算：(target + draft) / target 向上取整
    total_layers = int(target_num_layers) + int(draft_num_layers)
    return (
        int(target_cell_size_per_token) * int(total_layers) + int(target_num_layers) - 1
    ) // int(target_num_layers)


def resolve_dflash_verify_mask_policy(attn_backend: Any) -> tuple[str, bool]:
    # 通过最多 4 层 full_attn_backend 解引用，获取底层 attention 后端对象
    backend = attn_backend
    for _ in range(4):
        full_backend = getattr(backend, "full_attn_backend", None)
        if full_backend is None:
            break
        backend = full_backend
    # 取最终后端的类名，用于与免 mask 集合对比
    backend_name = type(backend).__name__
    # 返回 (后端名称, 是否需要自定义 causal mask)
    return backend_name, (backend_name not in _DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS)


def _get_or_create_chain_verify_buffers(
    *,
    bs: int,                   # 当前批大小
    draft_token_num: int,      # 每请求的草稿 token 数（即 block_size）
    device: torch.device,      # 目标 GPU 设备
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    # 缓存键：(device_index, draft_token_num)，不同 bs 共用同一批缓冲区（按需扩容）
    key = (device.index, int(draft_token_num))
    cached = _DFLASH_CHAIN_VERIFY_BUFFERS.get(key)
    cap_bs = 0 if cached is None else int(cached["cap_bs"])
    # 如果已缓存容量不足，重新分配（容量加倍以摊销分配开销）
    if cap_bs < bs:
        new_cap = max(int(bs), cap_bs * 2 if cap_bs > 0 else int(bs))
        # retrieve_index[i, j] = i * draft_token_num + j，线性链无树分叉，直接顺序索引
        retrieve_index = torch.arange(
            new_cap * draft_token_num, dtype=torch.int64, device=device
        ).view(new_cap, draft_token_num)
        # retrieve_next_token[i, j] = j+1，末尾置 -1 表示链终止
        row_next = torch.arange(
            1, draft_token_num + 1, dtype=torch.int64, device=device
        )
        row_next[-1] = -1
        retrieve_next_token = row_next.unsqueeze(0).expand(new_cap, -1).clone()
        # retrieve_next_sibling 全置 -1，线性链无兄弟节点
        retrieve_next_sibling = torch.full(
            (new_cap, draft_token_num), -1, dtype=torch.int64, device=device
        )
        # predicts：被采样/验证后最终选出的 token id，展平存储
        predicts = torch.empty(
            (new_cap * draft_token_num,), dtype=torch.int32, device=device
        )
        # accept_index[i, k] 表示第 i 条请求接受长度为 k 时对应的 predicts 索引
        accept_index = torch.empty(
            (new_cap, draft_token_num), dtype=torch.int32, device=device
        )
        # accept_token_num[i] 存放第 i 条请求最终被接受的草稿 token 数
        accept_token_num = torch.empty((new_cap,), dtype=torch.int32, device=device)
        cached = {
            "cap_bs": int(new_cap),
            "retrieve_index": retrieve_index,
            "retrieve_next_token": retrieve_next_token,
            "retrieve_next_sibling": retrieve_next_sibling,
            "predicts": predicts,
            "accept_index": accept_index,
            "accept_token_num": accept_token_num,
        }
        # 写回全局缓存
        _DFLASH_CHAIN_VERIFY_BUFFERS[key] = cached

    assert cached is not None
    # 按实际 bs 切片返回，避免越界访问
    retrieve_index = cached["retrieve_index"][:bs]
    retrieve_next_token = cached["retrieve_next_token"][:bs]
    retrieve_next_sibling = cached["retrieve_next_sibling"][:bs]
    predicts = cached["predicts"][: bs * draft_token_num]
    accept_index = cached["accept_index"][:bs]
    accept_token_num = cached["accept_token_num"][:bs]
    return (
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        predicts,
        accept_index,
        accept_token_num,
    )


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> List[int]:
    """Select target layer indices used to build DFlash context features.

    Args:
        num_target_layers: Number of transformer layers in the runtime target model.
        num_draft_layers: Number of layers in the DFlash draft model.

    Returns:
        A list of 0-based target layer indices of length `num_draft_layers`.

    Notes:
        - DFlash uses hidden states after each selected target layer (HF-style).
        - SGLang captures "before layer i", so the model hook will typically add +1
          when mapping to capture points.
    """
    if num_target_layers <= 0:
        raise ValueError(
            f"num_target_layers must be positive, got {num_target_layers}."
        )
    if num_draft_layers <= 0:
        raise ValueError(f"num_draft_layers must be positive, got {num_draft_layers}.")

    # 草稿只有 1 层时，取目标模型中间层的隐藏状态
    if num_draft_layers == 1:
        return [num_target_layers // 2]

    # 均匀分布：在 [1, num_target_layers-3] 区间内按草稿层数等间距采样目标层
    start = 1
    end = num_target_layers - 3
    if end < start:
        raise ValueError(
            "DFlash layer selection requires num_target_layers >= 4. "
            f"Got num_target_layers={num_target_layers}."
        )

    span = end - start
    # 对每个草稿层 i，用插值公式确定对应的目标层索引（四舍五入取整）
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    # 统一访问接口：dict 用 .get()，对象用 getattr()
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _get_text_config(config: Any) -> Any:
    # 从多模态或嵌套 HF config 中提取文本子配置（VLM 通常将语言模型配置嵌套在 text_config）
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get("text_config", config)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return text_config
    # 尝试通过 get_text_config() 方法解析（部分 HF config 类提供此接口）
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            resolved = get_text_config()
            if resolved is not None:
                return resolved
        except TypeError:
            pass
    return config


def _get_dflash_config(config: Any) -> dict:
    # 从 HF config 中提取 dflash_config 子字典；不存在时返回空字典
    if isinstance(config, dict):
        cfg = config.get("dflash_config", None)
    else:
        cfg = getattr(config, "dflash_config", None)
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg

    # 非 dict 类型（如自定义 config 对象）尝试转换为 dict
    try:
        return dict(cfg)
    except Exception:
        return {}


def _parse_optional_int(
    value: Any,
    *,
    field_name: str,                      # 字段名称，用于错误信息
    min_value: Optional[int] = None,      # 最小合法值（None 表示不限）
) -> Optional[int]:
    # None 透传：允许配置字段缺省
    if value is None:
        return None
    try:
        parsed = int(value)
    except Exception as e:
        raise ValueError(f"Invalid {field_name}={value!r}.") from e
    # 检查最小值约束（min_value=1 时错误信息改为 "must be positive"）
    if min_value is not None and parsed < int(min_value):
        comparator = "positive" if int(min_value) == 1 else f">= {int(min_value)}"
        raise ValueError(f"{field_name} must be {comparator}, got {parsed}.")
    return parsed


@dataclass(frozen=True)
# DFlash 草稿模型解析后的配置数据类（不可变，构建后只读）
class DFlashDraftConfig:
    num_hidden_layers: Optional[int]      # 草稿模型隐藏层数（从 text_config 解析）
    num_target_layers: Optional[int]      # 目标模型层数（用于 target_layer_ids 推断）
    block_size: Optional[int]             # DFlash 每次草稿推理的 block（sequence）长度
    target_layer_ids: Optional[List[int]] # 显式指定的目标层索引列表（None 则自动推断）
    mask_token: str                       # mask 位置填充 token 字符串
    mask_token_id: Optional[int]          # mask token 对应的词汇表 ID（None 则运行时查找）

    def require_num_layers(self) -> int:
        # 强制要求 num_hidden_layers 存在，缺失时抛出明确错误
        if self.num_hidden_layers is None:
            raise ValueError(
                "DFLASH requires draft num_hidden_layers in config. "
                "Got config without num_hidden_layers."
            )
        return int(self.num_hidden_layers)

    def resolve_block_size(self, *, default: Optional[int] = None) -> Optional[int]:
        # 返回配置中的 block_size，未设置时使用调用方提供的默认值
        return self.block_size if self.block_size is not None else default

    def resolve_target_layer_ids(
        self,
        *,
        target_num_layers: int,
        draft_num_layers: Optional[int] = None,  # 未指定时从 num_hidden_layers 推断
    ) -> List[int]:
        target_num_layers = int(target_num_layers)
        if target_num_layers <= 0:
            raise ValueError(
                f"target_num_layers must be positive, got {target_num_layers}."
            )

        # 未显式指定目标层时，调用 build_target_layer_ids 均匀自动选层
        if self.target_layer_ids is None:
            if draft_num_layers is None:
                draft_num_layers = self.require_num_layers()
            return build_target_layer_ids(target_num_layers, int(draft_num_layers))

        # 显式指定时，校验每个 layer_id 在合法范围内
        resolved = list(self.target_layer_ids)
        if len(resolved) <= 0:
            raise ValueError(
                "DFLASH dflash_config.target_layer_ids must be non-empty. "
                f"Got len(target_layer_ids)={len(resolved)}."
            )
        for idx, val in enumerate(resolved):
            if val < 0 or val >= target_num_layers:
                raise ValueError(
                    "DFLASH target_layer_ids contains an out-of-range layer id. "
                    f"target_layer_ids[{idx}]={val}, target_num_layers={target_num_layers}."
                )
        return resolved


def parse_dflash_draft_config(*, draft_hf_config: Any) -> DFlashDraftConfig:
    """Parse and validate DFLASH draft config fields from HF config/dict."""
    # 提取 dflash_config 子字典和文本子配置
    dflash_cfg = _get_dflash_config(draft_hf_config)
    draft_text_config = _get_text_config(draft_hf_config)

    # 从文本子配置中解析草稿模型层数（必须为正整数）
    num_hidden_layers = _parse_optional_int(
        _cfg_get(draft_text_config, "num_hidden_layers", None),
        field_name="DFLASH draft num_hidden_layers",
        min_value=1,
    )
    # num_target_layers 优先从 dflash_config 子字典获取，其次从顶层配置回退
    raw_num_target_layers = dflash_cfg.get(
        "num_target_layers",
        _cfg_get(draft_hf_config, "num_target_layers", None),
    )
    num_target_layers = _parse_optional_int(
        raw_num_target_layers,
        field_name="DFLASH draft num_target_layers",
        min_value=1,
    )

    # block_size 兼容旧版 checkpoint（支持顶层和 dflash_config 子字典两种存放位置）
    raw_block_size = dflash_cfg.get(
        "block_size",
        _cfg_get(draft_hf_config, "block_size", None),
    )
    block_size = _parse_optional_int(
        raw_block_size,
        field_name="DFLASH block_size",
        min_value=1,
    )

    # target_layer_ids 同样支持两处来源，None 表示运行时自动推断
    layer_ids = dflash_cfg.get(
        "target_layer_ids",
        _cfg_get(draft_hf_config, "target_layer_ids", None),
    )
    parsed_target_layer_ids: Optional[List[int]]
    if layer_ids is None:
        # 未指定则延迟到 resolve_target_layer_ids() 时自动计算
        parsed_target_layer_ids = None
    else:
        # 校验类型为列表/元组，且非空，每个元素转 int
        if not isinstance(layer_ids, (list, tuple)):
            raise ValueError(
                "DFLASH dflash_config.target_layer_ids must be a list of ints, "
                f"got type={type(layer_ids).__name__}."
            )
        parsed_target_layer_ids = [int(x) for x in layer_ids]
        if len(parsed_target_layer_ids) <= 0:
            raise ValueError(
                "DFLASH dflash_config.target_layer_ids must be non-empty. "
                f"Got len(target_layer_ids)={len(parsed_target_layer_ids)}."
            )

    # mask_token：缺省使用全局默认值，必须为非空字符串
    mask_token = dflash_cfg.get("mask_token", None)
    if mask_token is None:
        mask_token = DEFAULT_DFLASH_MASK_TOKEN
    if not isinstance(mask_token, str) or not mask_token:
        raise ValueError(
            "DFLASH dflash_config.mask_token must be a non-empty string, "
            f"got {mask_token!r}."
        )

    # mask_token_id：可选，若提供必须为非负整数
    mask_token_id = dflash_cfg.get("mask_token_id", None)
    if mask_token_id is not None:
        if not isinstance(mask_token_id, Integral) or isinstance(mask_token_id, bool):
            raise ValueError(
                "DFLASH dflash_config.mask_token_id must be an integer, "
                f"got {mask_token_id!r} (type={type(mask_token_id).__name__})."
            )
        mask_token_id = int(mask_token_id)
        if mask_token_id < 0:
            raise ValueError(
                "DFLASH dflash_config.mask_token_id must be non-negative, "
                f"got {mask_token_id}."
            )

    # 构建并返回不可变配置对象
    return DFlashDraftConfig(
        num_hidden_layers=num_hidden_layers,
        num_target_layers=num_target_layers,
        block_size=block_size,
        target_layer_ids=parsed_target_layer_ids,
        mask_token=mask_token,
        mask_token_id=mask_token_id,
    )


def can_dflash_slice_qkv_weight(qkv_proj: Any) -> Tuple[bool, str]:
    """Validate whether DFlash can slice KV weights from a fused QKV linear layer."""
    # 量化层无法直接切片权重矩阵，DFlash 要求原始 float 权重
    quant_method = getattr(qkv_proj, "quant_method", None)
    if not isinstance(quant_method, UnquantizedLinearMethod):
        return (
            False,
            "quantized qkv_proj is not supported for this path "
            f"(quant_method={type(quant_method).__name__})",
        )
    # 权重张量必须存在（某些特殊配置下可能缺失）
    if not hasattr(qkv_proj, "weight"):
        return False, "qkv weight tensor is missing"
    return True, ""


def can_dflash_use_fused_qkv_proj(qkv_proj: Any) -> Tuple[bool, str]:
    """Validate whether a QKV layer is eligible for DFlash fused KV materialization."""
    # 先检查基础权重切片条件
    eligible, reason = can_dflash_slice_qkv_weight(qkv_proj)
    if not eligible:
        return False, reason
    # fused KV 路径不支持 bias（需要纯矩阵乘法才能高效融合）
    if getattr(qkv_proj, "bias", None) is not None:
        return False, "qkv bias is not supported for fused KV path"
    return True, ""


def compute_dflash_accept_len_and_bonus(
    *,
    candidates: torch.Tensor,       # [bs, block_size]，草稿 token 候选（含当前 token）
    target_predict: torch.Tensor,   # [bs, block_size]，目标模型各位置的 argmax 预测
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute DFlash accept lengths and bonus tokens (greedy verify rule).

    Args:
        candidates: Token ids proposed by the DFlash draft, including the current token.
            Shape: [bs, block_size]. candidates[:, 0] is the current token.
        target_predict: Token ids predicted by the target model for each position in the block.
            Shape: [bs, block_size]. target_predict[:, t] corresponds to argmax at position t.

    Returns:
        accept_len: int32 tensor [bs], number of accepted *draft* tokens (excluding current token and bonus token).
        bonus: int64 tensor [bs], the target-predicted token at index accept_len (the "bonus" token to append).

    Notes:
        Matches the reference implementation rule:
          accept while candidates[:, 1:] == target_predict[:, :-1] consecutively.
    """
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got shape={tuple(candidates.shape)}")
    if target_predict.shape != candidates.shape:
        raise ValueError(
            "target_predict must have the same shape as candidates. "
            f"candidates.shape={tuple(candidates.shape)}, target_predict.shape={tuple(target_predict.shape)}"
        )

    bs, block_size = candidates.shape
    if bs <= 0:
        raise ValueError(f"batch size must be positive, got {bs}.")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    # 对齐比较：candidates[:, 1:] 与 target_predict[:, :-1] 逐位对比
    # cumprod 实现"连续匹配"语义：一旦某位不匹配，后续全变 0
    matches = candidates[:, 1:] == target_predict[:, :-1]
    accept_len = matches.to(torch.int32).cumprod(dim=1).sum(dim=1)
    # bonus token 是目标模型在接受位置后预测的下一个 token
    bonus = target_predict[torch.arange(bs, device=target_predict.device), accept_len]
    return accept_len, bonus.to(torch.int64)


def compute_dflash_sampling_accept_len_and_bonus(
    *,
    candidates: torch.Tensor,           # [bs, draft_token_num]，草稿提议的 token（含当前 token）
    next_token_logits: torch.Tensor,    # [bs * draft_token_num, vocab_size]，目标模型每位置 logits
    sampling_info: Any,                 # 采样参数（temperature、top_k、top_p 等）
    threshold_single: Optional[float] = None,    # 单步接受阈值（None 则从 server_args 读取）
    threshold_acc: Optional[float] = None,       # 累积接受阈值（None 则从 server_args 读取）
    uniform_samples: Optional[torch.Tensor] = None,                  # [bs, draft_token_num]，预分配随机数
    uniform_samples_for_final_sampling: Optional[torch.Tensor] = None,  # [bs]，最终 bonus 采样随机数
    use_sparse_topk: bool = True,       # 是否使用稀疏 top-k 路径（节省显存带宽）
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute DFlash accept lengths and bonus tokens for non-greedy sampling.

    This is a chain-specialized variant of speculative target-only verification:
      - DFlash proposals are linear (topk == 1), so each verify level has at most one candidate.
      - When a candidate is rejected at a level, the final token is sampled from
        `relu(q - p)` where `p` has only the rejected candidate mass.
    """
    # 非贪心验证核不可用时快速失败
    if not _DFLASH_SAMPLING_VERIFY_AVAILABLE:
        raise RuntimeError(
            "DFLASH non-greedy verification is unavailable on this build/device."
        )
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got shape={tuple(candidates.shape)}")
    if next_token_logits.ndim != 2:
        raise ValueError(
            "next_token_logits must be 2D, "
            f"got shape={tuple(next_token_logits.shape)}."
        )

    bs, draft_token_num = candidates.shape
    if bs <= 0:
        raise ValueError(f"batch size must be positive, got {bs}.")
    if draft_token_num <= 0:
        raise ValueError(f"draft_token_num must be positive, got {draft_token_num}.")
    # logits 的行数必须等于 bs * draft_token_num（目标模型为每个草稿位置单独输出）
    if next_token_logits.shape[0] != bs * draft_token_num:
        raise ValueError(
            "next_token_logits row count mismatch. "
            f"Expected {bs * draft_token_num}, got {next_token_logits.shape[0]}."
        )
    if candidates.device != next_token_logits.device:
        raise ValueError(
            "candidates and next_token_logits must be on the same device, "
            f"got {candidates.device} and {next_token_logits.device}."
        )

    # 从 server_args 延迟加载接受阈值（避免模块导入时强依赖）
    if threshold_single is None:
        from sglang.srt.server_args import get_global_server_args

        threshold_single = get_global_server_args().speculative_accept_threshold_single
    if threshold_acc is None:
        from sglang.srt.server_args import get_global_server_args

        threshold_acc = get_global_server_args().speculative_accept_threshold_acc
    threshold_single = float(threshold_single)
    # threshold_acc 用作除数，加 1e-9 防止零除
    threshold_acc = max(float(threshold_acc), 1e-9)

    device = next_token_logits.device

    # 若未提供随机数则在 GPU 上现场采样（支持外部注入以便复现）
    if uniform_samples is None:
        uniform_samples = torch.rand(
            (bs, draft_token_num), dtype=torch.float32, device=device
        )
    else:
        if uniform_samples.shape != (bs, draft_token_num):
            raise ValueError(
                "uniform_samples shape mismatch. "
                f"Expected {(bs, draft_token_num)}, got {tuple(uniform_samples.shape)}."
            )
        uniform_samples = uniform_samples.to(device=device, dtype=torch.float32)

    # 用于在被拒绝位置后 bonus 采样的额外随机数
    if uniform_samples_for_final_sampling is None:
        uniform_samples_for_final_sampling = torch.rand(
            (bs,), dtype=torch.float32, device=device
        )
    else:
        if uniform_samples_for_final_sampling.shape != (bs,):
            raise ValueError(
                "uniform_samples_for_final_sampling shape mismatch. "
                f"Expected {(bs,)}, got {tuple(uniform_samples_for_final_sampling.shape)}."
            )
        uniform_samples_for_final_sampling = uniform_samples_for_final_sampling.to(
            device=device,
            dtype=torch.float32,
        )

    need_top_k = bool(getattr(sampling_info, "need_top_k_sampling", True))
    need_top_p = bool(getattr(sampling_info, "need_top_p_sampling", False))
    # 将 temperature 按草稿长度展开，与 logits 行一一对应
    expanded_temperature = torch.repeat_interleave(
        sampling_info.temperatures, draft_token_num, dim=0
    )
    scaled_logits = next_token_logits / expanded_temperature
    sparse_topk_applied = False

    # 稀疏 top-k 路径：先 topk 再 softmax，节省全词汇表 softmax 开销
    if use_sparse_topk and need_top_k:
        repeated_top_ks = torch.repeat_interleave(
            sampling_info.top_ks, draft_token_num, dim=0
        ).to(dtype=torch.int64)
        vocab_size = int(scaled_logits.shape[-1])
        repeated_top_ks.clamp_(min=1, max=vocab_size)
        max_top_k = int(repeated_top_ks.max().item())

        # Sparse exact path for top-k/top-p (top-k-first semantics), then scatter to dense.
        if 0 < max_top_k < vocab_size:
            # 批量 topk：每行取最大的 max_top_k 个 logits
            topk_logits, topk_indices = torch.topk(scaled_logits, k=max_top_k, dim=-1)
            # 若各请求 top_k 不相同，用 mask 填 -inf 使超出部分无效
            if not torch.all(repeated_top_ks == max_top_k):
                ranks = torch.arange(max_top_k, device=device, dtype=torch.int64)[
                    None, :
                ]
                valid = ranks < repeated_top_ks.unsqueeze(1)
                topk_logits = topk_logits.masked_fill(~valid, float("-inf"))

            topk_probs = F.softmax(topk_logits, dim=-1)
            # 在稀疏 top-k 子集上应用 top-p 重归一化
            if need_top_p:
                repeated_top_ps = torch.repeat_interleave(
                    sampling_info.top_ps, draft_token_num, dim=0
                )
                topk_probs = top_p_renorm_prob(topk_probs, repeated_top_ps)

            # 将稀疏概率散射回完整词汇表大小（用于 tree_speculative_sampling）
            target_probs = torch.zeros_like(scaled_logits, dtype=topk_probs.dtype)
            target_probs.scatter_(1, topk_indices, topk_probs)
            sparse_topk_applied = True

    # 密集路径：全量 softmax，按需应用 top-k/top-p 重归一化
    if not sparse_topk_applied:
        target_probs = F.softmax(scaled_logits, dim=-1)
        if need_top_k:
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(sampling_info.top_ks, draft_token_num, dim=0),
            )
        if need_top_p:
            target_probs = top_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(sampling_info.top_ps, draft_token_num, dim=0),
            )
    # 重塑为 [bs, draft_token_num, vocab_size] 供 tree_speculative_sampling 使用
    target_probs = target_probs.view(bs, draft_token_num, -1).contiguous()
    # DFlash 链式验证中草稿分布置零（仅使用目标分布的 target-only 模式）
    draft_probs = torch.zeros_like(target_probs)

    # 获取或创建链式验证所需的持久化 GPU 缓冲区
    (
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        predicts,
        accept_index,
        accept_token_num,
    ) = _get_or_create_chain_verify_buffers(
        bs=bs,
        draft_token_num=draft_token_num,
        device=device,
    )
    # candidates 转 int64 以匹配 sgl_kernel 算子签名
    candidates_i64 = (
        candidates if candidates.dtype == torch.int64 else candidates.to(torch.int64)
    )
    # 调用树形推测采样（此处退化为链式，retrieve_next_sibling 全为 -1）
    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates_i64,
        # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
        retrive_index=retrieve_index,
        retrive_next_token=retrieve_next_token,
        retrive_next_sibling=retrieve_next_sibling,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=True,
    )

    # accept_token_num 即接受的草稿 token 数（不含 bonus）
    accept_len = accept_token_num
    row_ids = torch.arange(bs, dtype=torch.long, device=device)
    # 从 accept_index 查找 bonus token 在 predicts 中的位置
    accept_pos = accept_index[row_ids, accept_len.to(torch.long)].to(torch.long)
    bonus = predicts[accept_pos].to(torch.int64)
    return accept_len, bonus
