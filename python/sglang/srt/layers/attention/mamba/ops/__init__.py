# Mamba ops 子包公共接口：导出 SSM 相关核心算子
# PAD_SLOT_ID: 填充槽位标识符（用于连续批处理中跳过无效序列）
from .mamba_ssm import PAD_SLOT_ID
# mamba_chunk_scan_combined: Mamba2 chunked SSM 扫描（prefill 阶段核心算子）
from .ssd_combined import mamba_chunk_scan_combined
# selective_state_update: 选择性 SSM 状态更新（decode 阶段核心算子）
# initialize_mamba_selective_state_update_backend: 初始化 SSM 更新后端（Triton/CUDA 切换）
from .ssu_dispatch import (
    initialize_mamba_selective_state_update_backend,
    selective_state_update,
)

__all__ = [
    "PAD_SLOT_ID",
    "selective_state_update",
    "mamba_chunk_scan_combined",
    "initialize_mamba_selective_state_update_backend",
]
