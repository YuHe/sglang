# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Mooncake-specific utilities for custom memory pool management."""

# Mooncake 传输引擎自定义内存池工具：支持 NVLINK/BAREX/INTRA_NODE_NVLINK 等池类型
import logging
from typing import Any, Optional, Tuple

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Mooncake 支持的自定义内存池类型常量列表
# NVLINK: 基于 NVLink 高速互联的内存池；BAREX: 裸机 RDMA 内存池；INTRA_NODE_NVLINK: 节点内 NVLink
# Global constants for custom memory pool types
SUPPORTED_MOONCAKE_CUSTOM_MEM_POOL_TYPES = ["NVLINK", "BAREX", "INTRA_NODE_NVLINK"]


def init_mooncake_custom_mem_pool(
    device: str,
) -> Tuple[bool, Optional[Any], Optional[str]]:
    """
    Initialize custom memory pool based on environment variable.

    Args:
        device: The device to allocate memory on

    Returns:
        Tuple of (enable_custom_mem_pool, custom_mem_pool, custom_mem_pool_type)
    """
    # 根据环境变量检查是否启用自定义内存池，并获取池类型
    enable_custom_mem_pool, custom_mem_pool_type = (
        check_mooncake_custom_mem_pool_enabled()
    )

    # 默认内存池对象为 None，按需初始化
    custom_mem_pool = None

    if enable_custom_mem_pool:
        try:
            # TODO(shangming): abstract custom allocator class for more backends
            # 根据池类型选择对应的 Mooncake 分配器
            if custom_mem_pool_type == "NVLINK":
                # NVLink 分配器：利用 NVLink 带宽在 GPU 间高速共享内存
                from mooncake.allocator import NVLinkAllocator

                allocator = NVLinkAllocator.get_allocator(device)
            elif custom_mem_pool_type == "BAREX":
                # Barex 分配器：裸机 RDMA 模式，直接操作物理内存
                from mooncake.allocator import BarexAllocator

                allocator = BarexAllocator.get_allocator(device)
            elif custom_mem_pool_type == "INTRA_NODE_NVLINK":
                # 节点内 NVLink 模式暂不支持自定义内存池，直接返回禁用
                return False, None, None
            else:
                # This should not happen due to the enable_custom_mem_pool check above
                raise ValueError(
                    f"Unsupported custom mem pool type: {custom_mem_pool_type}"
                )

            # 将 Mooncake 分配器包装为 PyTorch MemPool 对象，供 CUDA 使用
            custom_mem_pool = torch.cuda.MemPool(allocator.allocator())
            logger.debug(
                f"Initialized custom memory pool: {custom_mem_pool_type} on device {device}"
            )
        except ImportError as e:
            # 若 mooncake 分配器库未安装，回退到默认内存池
            logger.warning(
                f"Failed to import mooncake allocator for {custom_mem_pool_type}: {e}. "
                f"Falling back to default memory pool."
            )
            enable_custom_mem_pool = False
            custom_mem_pool = None
            custom_mem_pool_type = None
        except Exception as e:
            # 其他初始化异常，同样回退到默认内存池
            logger.error(
                f"Failed to initialize custom memory pool {custom_mem_pool_type}: {e}. "
                f"Falling back to default memory pool."
            )
            enable_custom_mem_pool = False
            custom_mem_pool = None
            custom_mem_pool_type = None
    else:
        # 未启用自定义内存池，返回全部为 None/False
        return False, None, None

    return enable_custom_mem_pool, custom_mem_pool, custom_mem_pool_type


def check_mooncake_custom_mem_pool_enabled() -> Tuple[bool, Optional[str]]:
    """
    Check if custom memory pool is enabled without importing allocators.

    Returns:
        Tuple of (enable_custom_mem_pool, custom_mem_pool_type)
    """
    # 从环境变量读取自定义内存池类型配置
    custom_mem_pool_type = envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get()

    if custom_mem_pool_type is not None:
        # Handle boolean True as NVLINK
        # 兼容旧版配置：若设置为字符串 "true"，映射为 NVLINK 类型
        if custom_mem_pool_type.lower() == "true":
            custom_mem_pool_type = "NVLINK"
        # 判断池类型是否在支持列表中
        enable_custom_mem_pool = (
            custom_mem_pool_type in SUPPORTED_MOONCAKE_CUSTOM_MEM_POOL_TYPES
        )
    else:
        # 未配置环境变量，禁用自定义内存池
        enable_custom_mem_pool = False
        custom_mem_pool_type = None

    return enable_custom_mem_pool, custom_mem_pool_type
