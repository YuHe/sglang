# Copyright 2023-2024 SGLang Team
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
"""ModelRunner runs the forward passes of the models."""
# 本模块实现 ModelRunner，负责模型权重加载、CUDA Graph 构建、forward pass 执行及 TP/PP 分布式推理管理

from __future__ import annotations

# 标准库导入
import contextlib
import datetime
import gc
import inspect
import logging
import os
import socket
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

# PyTorch 核心库
import torch
import torch.distributed as dist
from torch import nn

from sglang.jit_kernel.ngram_embedding import update_token_table
# 导入各类模型配置，支持多种混合架构模型
from sglang.srt.configs import (
    BailingHybridConfig,
    FalconH1Config,
    GraniteMoeHybridConfig,
    JetNemotronConfig,
    JetVLMConfig,
    KimiLinearConfig,
    Lfm2Config,
    Lfm2MoeConfig,
    Lfm2VlConfig,
    NemotronH_Nano_VL_V2_Config,
    NemotronHConfig,
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3NextConfig,
)
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.linear_attn_model_registry import get_linear_attn_config
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.configs.model_config import AttentionArch, ModelConfig, ModelImpl
from sglang.srt.configs.update_config import adjust_config_with_unaligned_cpu_tp
from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
# 调试工具：支持 tensor dump 和前向钩子
from sglang.srt.debug_utils.dumper import dumper
from sglang.srt.debug_utils.tensor_dump_forward_hook import (
    register_forward_hook_for_model,
)
# 分布式通信相关：TP/PP 组初始化与 AllReduce 设置
from sglang.srt.distributed import (
    get_default_distributed_backend,
    get_pp_group,
    get_tp_group,
    get_world_group,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
    set_mscclpp_all_reduce,
    set_torch_symm_mem_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.distributed.parallel_state import monkey_patch_vllm_parallel_state
# 弹性 EP（Expert Parallelism）状态管理，支持专家节点动态伸缩
from sglang.srt.elastic_ep.elastic_ep import (
    ElasticEPStateManager,
    join_process_groups,
    try_recover_ranks,
)
from sglang.srt.elastic_ep.expert_backup_client import ExpertBackupClient
from sglang.srt.environ import envs
# EPLB（专家负载均衡）管理器及专家分布统计工具
from sglang.srt.eplb.eplb_manager import EPLBManager
from sglang.srt.eplb.expert_distribution import (
    ExpertDistributionMetrics,
    ExpertDistributionRecorder,
    get_global_expert_distribution_recorder,
    set_global_expert_distribution_recorder,
)
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    broadcast_global_expert_location_metadata,
    compute_initial_expert_location_metadata,
    get_global_expert_location_metadata,
    set_global_expert_location_metadata,
)
from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner
from sglang.srt.layers import deep_gemm_wrapper
# 注意力层工具：后端注册表、NSA、TBO、DP Attention 支持
from sglang.srt.layers.attention.attention_registry import (
    ATTENTION_BACKENDS,
    attn_backend_wrapper,
)
from sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    get_attention_tp_group,
    get_attention_tp_size,
    initialize_dp_attention,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
# MoE 路由专家捕获器，用于记录推理时路由到的专家
from sglang.srt.layers.moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    RoutedExpertsOutput,
    get_global_experts_capturer,
    set_global_experts_capturer,
)
from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype
from sglang.srt.layers.sampler import create_sampler
from sglang.srt.layers.torchao_utils import apply_torchao_config_to_model
# LoRA 适配器管理器及 LoRA 注册表引用
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.managers.schedule_batch import sanity_check_mm_pad_shift_value
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.breakable_cuda_graph_runner import (
    BreakableCudaGraphRunner,
)
from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
from sglang.srt.model_executor.cuda_graph_runner import (
    CudaGraphRunner,
    DecodeInputBuffers,
    set_torch_compile_config,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_executor.hook_manager import register_forward_hooks
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.srt.model_executor.piecewise_cuda_graph_runner import (
    PiecewiseCudaGraphRunner,
)
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
# 模型加载工具：权重加载器、远程实例权重传输引擎
from sglang.srt.model_loader.loader import DefaultModelLoader, get_model_loader
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    RemoteInstanceWeightLoaderBackend,
    register_memory_region,
    trigger_init_weights_send_group_for_remote_instance_request,
)
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.platforms import current_platform
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    MultiprocessingSerializer,
    broadcast_pyobj,
    cpu_has_amx_support,
    dynamic_import,
    empty_context,
    enable_show_time_cost,
    get_available_gpu_memory,
    get_bool_env_var,
    get_cpu_ids_by_node,
    init_custom_process_group,
    is_hip,
    is_host_cpu_arm64,
    is_npu,
    log_info_on_rank0,
    monkey_patch_p2p_access_check,
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_tp_gather,
    reserve_rope_cache_for_long_sequences,
    set_cuda_arch,
    slow_rank_detector,
)
from sglang.srt.utils.network import NetworkAddress, get_local_ip_auto
from sglang.srt.utils.nvtx_pytorch_hooks import PytHooks
from sglang.srt.utils.offloader import (
    create_offloader_from_server_args,
    get_offloader,
    set_offloader,
)
from sglang.srt.utils.patch_torch import (
    monkey_patch_torch_reductions,
    register_sgl_tp_rank,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils.weight_checker import WeightChecker
# 权重同步：扁平化 tensor 桶，用于在线权重更新
from sglang.srt.weight_sync.tensor_bucket import (
    FlattenedTensorBucket,
    FlattenedTensorMetadata,
)

# 检测当前平台是否为 HIP（AMD ROCm）、NPU（昇腾）、是否支持 AMX 指令集及 ARM64 架构
_is_hip = is_hip()
_is_npu = is_npu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu_arm64 = is_host_cpu_arm64()
# 仅在 HIP 平台且环境变量启用时使用 aiter 后端
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

# NPU 后端（昇腾）初始化
if _is_npu:
    from sglang.srt.hardware_backend.npu.utils import init_npu_backend

    init_npu_backend()
elif current_platform.is_out_of_tree():
    # 第三方平台（非 GPU/NPU）初始化后端
    current_platform.init_backend()

# 支持 MLA（Multi-head Latent Attention）的注意力后端列表
MLA_ATTENTION_BACKENDS = [
    "aiter",
    "flashinfer",
    "fa3",
    "fa4",
    "triton",
    "flashmla",
    "cutlass_mla",
    "trtllm_mla",
    "ascend",
    "nsa",
    "intel_xpu",
]

# 支持分块前缀缓存（Chunked Prefix Cache）的注意力后端列表
CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS = [
    "flashinfer",
    "fa3",
    "fa4",
    "flashmla",
    "cutlass_mla",
    "trtllm_mla",
]

# torch dtype 到 KV Cache 字符串表示的映射，用于选择合适的量化格式
TORCH_DTYPE_TO_KV_CACHE_STR = {
    torch.float8_e4m3fn: "fp8_e4m3",
    torch.float8_e4m3fnuz: "fp8_e4m3",
    torch.float8_e5m2: "fp8_e5m2",
    torch.bfloat16: "bf16",
}


def add_mla_attention_backend(backend_name):
    # 动态注册新的 MLA 注意力后端
    if backend_name not in MLA_ATTENTION_BACKENDS:
        MLA_ATTENTION_BACKENDS.append(backend_name)
        logger.info(f"Added {backend_name} to MLA_ATTENTION_BACKENDS.")


def add_chunked_prefix_cache_attention_backend(backend_name):
    # 动态注册支持分块前缀缓存的注意力后端
    if backend_name not in CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS:
        CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS.append(backend_name)
        logger.info(
            f"Added {backend_name} to CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS."
        )


# Detect stragger ranks in model loading
# 检测模型加载时的滞后 rank，超时阈值设为 480 秒（预留后处理时间）
UNBALANCED_MODEL_LOADING_TIMEOUT_S = 480  # leave more time for post data processing


logger = logging.getLogger(__name__)


def resolve_language_model(model: nn.Module) -> nn.Module:
    # 从多模态/复合模型中提取纯语言模型部分
    model_cls_name = model.__class__.__name__
    if model_cls_name == "Qwen3OmniMoeForConditionalGeneration":
        # Qwen3Omni 模型的语言模型在 thinker.model 中
        return model.thinker.model
    return model.model


class RankZeroFilter(logging.Filter):
    """Filter that only allows INFO level logs from rank 0, but allows all other levels from any rank."""
    # 日志过滤器：INFO 级别日志仅允许 rank 0 输出，其他级别所有 rank 均可输出

    def __init__(self, is_rank_zero):
        super().__init__()
        self.is_rank_zero = is_rank_zero  # 标记当前进程是否为 rank 0

    def filter(self, record):
        # INFO 级别仅在 rank 0 时放行
        if record.levelno == logging.INFO:
            return self.is_rank_zero
        return True


@dataclass
class ModelRunnerOutput:
    # ModelRunner 前向传播的输出结构
    logits_output: Union[LogitsProcessorOutput, PPProxyTensors]  # logits 或流水线代理张量
    can_run_graph: bool  # 当前批次是否可运行 CUDA Graph
    expert_distribution_metrics: Optional[ExpertDistributionMetrics] = None  # MoE 专家分布统计
    routed_experts_output: Optional[RoutedExpertsOutput] = None  # 路由专家输出信息


class ModelRunner(ModelRunnerKVCacheMixin):
    """ModelRunner runs the forward passes of the models."""
    # 主模型执行器，继承 KV Cache 管理混入类，负责整个推理生命周期

    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,  # 静态权重占用显存比例
        gpu_id: int,
        tp_rank: int,   # Tensor Parallelism rank
        tp_size: int,   # Tensor Parallelism 总进程数
        moe_ep_rank: int,  # MoE Expert Parallelism rank
        moe_ep_size: int,  # MoE Expert Parallelism 总进程数
        pp_rank: int,   # Pipeline Parallelism rank
        pp_size: int,   # Pipeline Parallelism 总阶段数
        nccl_port: int,
        server_args: ServerArgs,
        dp_rank: Optional[int] = None,   # Data Parallelism rank（可选）
        attn_cp_rank: Optional[int] = None,  # Attention Context Parallelism rank（可选）
        moe_dp_rank: Optional[int] = None,   # MoE Data Parallelism rank（可选）
        is_draft_worker: bool = False,        # 是否为投机解码草稿模型 worker
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
        memory_pool_config: Optional[MemoryPoolConfig] = None,
        draft_model_idx: Optional[int] = None,
    ):
        # Parse args
        # 保存各并行维度参数及基础配置
        self.mem_fraction_static = mem_fraction_static
        self.device = server_args.device
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.moe_ep_rank = moe_ep_rank
        self.moe_ep_size = moe_ep_size
        # 如果启用 DP Attention，则使用 dp_size；否则为 1
        self.dp_size = server_args.dp_size if server_args.enable_dp_attention else 1
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.attn_cp_rank = attn_cp_rank
        self.attn_cp_size = server_args.attn_cp_size
        self.moe_dp_rank = moe_dp_rank
        self.moe_dp_size = server_args.moe_dp_size
        self.model_config = model_config
        self.dist_port = nccl_port
        self.server_args = server_args
        self.is_draft_worker = is_draft_worker
        self.memory_pool_config = memory_pool_config
        self.is_generation = model_config.is_generation  # 是否为生成式模型（区别于 embedding 模型）
        self.is_multimodal = model_config.is_multimodal  # 是否为多模态模型
        self.is_multimodal_chunked_prefill_supported = (
            model_config.is_multimodal_chunked_prefill_supported
        )
        # 解析投机解码算法（如 EAGLE、DFlash 等）
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.page_size = server_args.page_size  # KV Cache 分页大小
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.is_hybrid_swa = model_config.is_hybrid_swa  # 是否使用混合滑动窗口注意力
        self.is_hybrid_swa_compress = model_config.is_hybrid_swa_compress
        self.use_mla_backend = self.model_config.attention_arch == AttentionArch.MLA  # 是否使用 MLA 注意力后端
        self.attention_chunk_size = model_config.attention_chunk_size
        self.forward_pass_id = 0  # 前向传播计数器，用于调试追踪
        self.init_new_workspace = False
        self.draft_model_idx = draft_model_idx
        self.enable_hisparse = server_args.enable_hisparse  # 是否启用 HiSparse 稀疏注意力

        # 远程实例权重加载相关状态初始化
        self.remote_instance_transfer_engine = None
        self.remote_instance_transfer_engine_session_id = ""
        self.remote_instance_transfer_engine_weight_info = None

        # msprobe 精度调试工具初始化
        self.msprobe_debugger = None
        if server_args.msprobe_dump_config is not None:
            self.init_msprobe()

        # auxiliary hidden capture mode. TODO: expose this to server args?
        # EAGLE3 和 DFlash 投机解码算法需要捕获辅助隐状态
        self.eagle_use_aux_hidden_state = False
        self.dflash_use_aux_hidden_state = False
        self.dflash_target_layer_ids = None
        self.dflash_draft_num_layers = None
        if self.spec_algorithm.is_eagle3() and not self.is_draft_worker:
            # load draft config
            # 加载草稿模型配置以确定辅助隐状态所在层
            draft_model_config = self._build_model_config(
                server_args,
                model_path=(server_args.speculative_draft_model_path),
                model_revision=server_args.speculative_draft_model_revision,
                is_draft_model=True,
            )
            self.eagle_use_aux_hidden_state = True

            try:
                # get the aux layer from draft model config
                # 从草稿模型配置中读取 EAGLE 辅助隐状态层 ID
                eagle_config = getattr(
                    draft_model_config.hf_config, "eagle_config", None
                )
                self.eagle_use_aux_hidden_state = eagle_config.get(
                    "use_aux_hidden_state", True
                )
                self.eagle_aux_hidden_state_layer_ids = eagle_config[
                    "eagle_aux_hidden_state_layer_ids"
                ]
            except:
                # if there is no aux layer, set to None
                # 若草稿模型无辅助层配置，则设为 None
                self.eagle_aux_hidden_state_layer_ids = None

        if self.spec_algorithm.is_dflash() and not self.is_draft_worker:
            from sglang.srt.speculative.dflash_utils import (
                parse_dflash_draft_config,
            )

            # Select target layers to capture for building DFlash context features.
            # 解析 DFlash 草稿配置，确定需要捕获的目标层 ID
            draft_model_config = self._build_model_config(
                server_args,
                model_path=(server_args.speculative_draft_model_path),
                model_revision=server_args.speculative_draft_model_revision,
                is_draft_model=True,
            )
            dflash_draft_config = parse_dflash_draft_config(
                draft_hf_config=draft_model_config.hf_config
            )
            draft_num_layers = dflash_draft_config.require_num_layers()
            trained_target_layers = dflash_draft_config.num_target_layers

            # 获取目标模型（主模型）的隐藏层数
            target_num_layers = getattr(
                self.model_config.hf_text_config, "num_hidden_layers", None
            )
            if target_num_layers is None:
                raise ValueError(
                    "DFLASH requires target num_hidden_layers in config. "
                    f"Got target={target_num_layers}."
                )
            target_num_layers = int(target_num_layers)

            if (
                trained_target_layers is not None
                and trained_target_layers != target_num_layers
            ):
                # 若草稿配置中的目标层数与实际模型不符，发出警告并按运行时模型决定
                logger.warning(
                    "DFLASH draft config num_target_layers=%s differs from runtime target num_hidden_layers=%s; "
                    "selecting capture layers based on the runtime target model.",
                    trained_target_layers,
                    target_num_layers,
                )

            self.dflash_use_aux_hidden_state = True
            self.dflash_draft_num_layers = int(draft_num_layers)
            # 计算 DFlash 需要捕获的具体层 ID 列表
            self.dflash_target_layer_ids = dflash_draft_config.resolve_target_layer_ids(
                target_num_layers=int(target_num_layers),
                draft_num_layers=int(draft_num_layers),
            )

        # Apply the rank zero filter to logger
        # 如果启用时间开销展示，则激活全局计时
        if server_args.show_time_cost:
            enable_show_time_cost()

        # Model-specific adjustment
        # 对特定模型进行参数调整（如修改某些配置以适配特殊架构）
        self.model_specific_adjustment()

        # Set the global server_args in the scheduler process
        # 将 server_args 同步到全局状态，供各子模块读取
        set_global_server_args_for_scheduler(server_args)
        global_server_args = get_global_server_args()

        # FIXME: hacky set `use_mla_backend`
        # 临时方案：将 MLA 后端标志注入全局配置
        global_server_args.use_mla_backend = self.use_mla_backend

        # Init OpenMP threads binding for CPU
        # CPU 模式下初始化 OpenMP 线程绑定（NUMA 亲和性）
        if self.device == "cpu":
            self.init_threads_binding()

        # Get available memory before model loading
        # 在加载模型前初始化 torch 分布式环境并获取当前可用显存量
        pre_model_load_memory = self.init_torch_distributed()

        # Initialize MooncakeTransferEngine
        # 初始化 Mooncake 传输引擎（用于跨机器 KV Cache 迁移）
        self.init_shared_mooncake_transfer_engine()

        # Init forward stream for overlap schedule
        # 创建独立 CUDA Stream，用于 overlap 调度（计算与通信重叠）
        self.forward_stream = torch.get_device_module(self.device).Stream()

        # CPU offload
        # 初始化 CPU offload 策略（将部分权重或激活卸载到 CPU 内存）
        set_offloader(create_offloader_from_server_args(server_args, dp_rank=dp_rank))

        self._weight_checker = WeightChecker(model_runner=self)

        if envs.SGLANG_DETECT_SLOW_RANK.get():
            # 检测并标记慢速 rank（用于诊断分布式推理中的性能瓶颈）
            slow_rank_detector.execute()

        # Init mindspore running environment when model impl is "mindspore"
        # 若模型实现为 MindSpore（昇腾），则初始化对应分布式通信环境
        self.init_mindspore_runner()

        # Update deep gemm configure
        # 更新 DeepGEMM JIT 编译配置（针对特定 GPU 优化的矩阵乘法内核）
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            deep_gemm_wrapper.update_deep_gemm_config(gpu_id, server_args)

        # For hisparse (must be set before initialize() so CUDA graph capture can see it)
        # HiSparse 稀疏注意力协调器，必须在 initialize() 前设置
        self.hisparse_coordinator = None

        # Initialize the model runner
        # 执行核心初始化：加载模型权重、分配 KV Cache、构建 CUDA Graph 等
        self.initialize(pre_model_load_memory)
        self.check_quantized_moe_compatibility()

        if (
            self.server_args.elastic_ep_backend is not None
            and self.server_args.elastic_ep_rejoin
        ):
            # 弹性 EP 重加入流程：重新加入进程组并广播专家位置元数据
            join_process_groups()
            broadcast_global_expert_location_metadata(
                src_rank=self._get_healthy_expert_location_src_rank(
                    invoked_in_elastic_ep_rejoin_path=True
                )
            )
            ElasticEPStateManager.instance().reset()

        if self.is_multimodal:
            # 多模态模型：校验 mm_pad_shift 的词表越界问题
            sanity_check_mm_pad_shift_value(self.model_config.vocab_size)

        # Temporary cached values
        # 检查模型 forward 签名是否支持 PP 流水线代理张量
        self.support_pp = (
            "pp_proxy_tensors" in inspect.signature(self.model.forward).parameters
        )

        if self.pp_size > 1:
            assert (
                self.support_pp
            ), "Pipeline Parallel is not compatible with this model."

        # For weight updates
        # 在线权重更新通信组（用于训练-推理融合场景）
        self._model_update_group = {}
        self._weights_send_group = {}

    def _build_model_config(
        self, server_args, model_path=None, model_revision=None, is_draft_model=False
    ):
        # 从 server_args 构建 ModelConfig，支持指定不同路径和版本（用于草稿模型）
        return ModelConfig.from_server_args(
            server_args,
            model_path=model_path,
            model_revision=model_revision,
            is_draft_model=is_draft_model,
        )

    def init_msprobe(self):
        # Init the msprobe
        # 初始化昇腾 msprobe 精度调试工具，用于 tensor 数据 dump
        try:
            from msprobe.pytorch import PrecisionDebugger, seed_all
        except ImportError:
            logger.warning(
                "Please install msprobe for tensor data dump: pip install mindstudio-probe --pre, "
                "see https://gitcode.com/Ascend/msprobe for details."
            )
            return
        seed_all(mode=True)  # 设置随机种子以保证结果可复现
        self.msprobe_debugger = PrecisionDebugger(
            config_path=self.server_args.msprobe_dump_config
        )

    def init_mindspore_runner(self):
        # Init the mindspore runner
        # for now, there is only some communication initialization work
        # 为 MindSpore 实现初始化分布式通信（仅在 NPU 上运行）
        if self.server_args.model_impl.lower() == ModelImpl.MINDSPORE and _is_npu:
            from sglang.srt.model_executor.mindspore_runner import init_ms_distributed

            init_ms_distributed(
                world_size=self.tp_size * self.pp_size,  # 总进程数 = TP × PP
                rank=self.tp_size * self.pp_rank + self.tp_rank,  # 当前进程全局 rank
                local_rank=self.gpu_id,
                server_args=self.server_args,
                port=self.dist_port,
            )

    def initialize(self, pre_model_load_memory: float):
        # ModelRunner 核心初始化入口：按顺序完成内存管理、专家位置、模型加载、KV Cache 分配等
        server_args = self.server_args

        # 创建内存节省适配器（可选，用于节省显存）
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.server_args.enable_memory_saver
        )

        if self.server_args.remote_instance_weight_loader_use_transfer_engine():
            # 若使用远程传输引擎加载权重，则先初始化传输引擎
            self.remote_instance_init_transfer_engine()

        if not self.is_draft_worker:
            # 非草稿 worker 需要初始化全局专家位置元数据（MoE 路由依赖）
            set_global_expert_location_metadata(
                compute_initial_expert_location_metadata(
                    server_args=server_args,
                    model_config=self.model_config,
                    moe_ep_rank=self.moe_ep_rank,
                )
            )
            if self.tp_rank == 0 and envs.SGLANG_LOG_EXPERT_LOCATION_METADATA.get():
                # rank 0 打印初始专家位置元数据（调试用）
                logger.info(
                    f"Initial expert_location_metadata: {get_global_expert_location_metadata()}"
                )

            # 初始化专家分布记录器（用于统计各 expert 的负载）
            set_global_expert_distribution_recorder(
                ExpertDistributionRecorder.init_new(
                    server_args,
                    get_global_expert_location_metadata(),
                    rank=self.tp_rank,
                )
            )

        # Expert parallelism
        # 初始化 EPLB（专家负载均衡）管理器，仅非草稿 worker 且启用时创建
        self.eplb_manager = (
            EPLBManager(self)
            if self.server_args.enable_eplb and (not self.is_draft_worker)
            else None
        )
        self.expert_location_updater = ExpertLocationUpdater()  # 专家位置在线更新器

        (
            ElasticEPStateManager.init(self.server_args)
            if self.server_args.elastic_ep_backend
            else None
        )
        # Load the model
        # 创建采样器并加载模型权重（核心步骤）
        self.sampler = create_sampler()
        self.load_model()

        # Load the expert backup client
        # 若启用弹性专家备份，则初始化备份客户端
        self.expert_backup_client = (
            ExpertBackupClient(self.server_args, self)
            if (
                self.server_args.enable_elastic_expert_backup
                and self.server_args.elastic_ep_backend is not None
            )
            else None
        )

        if (
            self.server_args.remote_instance_weight_loader_use_transfer_engine()
            and self.remote_instance_transfer_engine is not None
            and self.remote_instance_transfer_engine_weight_info is None
        ):
            # Register memory and upstream the transfer engine info to the bootstrap server
            # 注册模型权重内存区域并上报传输引擎元信息到 bootstrap server
            self.remote_instance_transfer_engine_weight_info = register_memory_region(
                self.model, self.remote_instance_transfer_engine
            )
            self._register_to_engine_info_bootstrap()

        # For MTP models like DeepSeek-V3 or GLM-4.5, the MTP layer(s) are used separately as draft
        # models for speculative decoding. In those cases, `num_nextn_predict_layers` is used to
        # determine the number of layers.
        # 处理 MTP（Multi-Token Prediction）模型的层数计算逻辑
        model_has_mtp_layers = self.model_config.num_nextn_predict_layers is not None
        model_num_layers = (
            self.model_config.num_nextn_predict_layers
            if self.is_draft_worker and model_has_mtp_layers
            else max(
                self.model_config.num_hidden_layers,
                self.model_config.num_attention_layers,
            )
        )
        # 特殊架构：MiMoV2MTP 和 Step3p5MTP 强制为 1 层
        if self.model_config.hf_config.architectures[0] == "MiMoV2MTP":
            model_num_layers = 1
        elif self.model_config.hf_config.architectures[0] == "Step3p5MTP":
            model_num_layers = 1
        # 获取 PP 分片的起始/结束层（流水线并行时每个 stage 只持有部分层）
        self.start_layer = getattr(self.model, "start_layer", 0)
        self.end_layer = getattr(self.model, "end_layer", model_num_layers)
        self.num_effective_layers = self.end_layer - self.start_layer

        # For LoopCoder models, each loop has its own layer_id, so we need to multiply by loop_num
        # LoopCoder 模型每个循环体有独立 layer_id，需乘以循环次数
        loop_num = getattr(self.model_config.hf_config, "loop_num", 1)
        if loop_num > 1:
            self.num_effective_layers = self.num_effective_layers * loop_num

        assert (
            (not model_has_mtp_layers)
            or (self.spec_algorithm.is_none())
            or (
                (not self.spec_algorithm.is_none())
                and (self.num_effective_layers == model_num_layers)
            )
        ), "PP is not compatible with MTP models."

        # Consider PP, so use start_layer and end_layer.
        # 根据 PP 分片范围过滤出本 stage 持有的全量注意力层和滑动窗口注意力层 ID
        full_attention_layer_ids = [
            layer_idx
            for layer_idx in range(self.start_layer, self.end_layer + 1)
            if hasattr(self.model_config, "full_attention_layer_ids")
            and layer_idx in self.model_config.full_attention_layer_ids
        ]
        swa_attention_layer_ids = [
            layer_idx
            for layer_idx in range(self.start_layer, self.end_layer + 1)
            if hasattr(self.model_config, "swa_attention_layer_ids")
            and layer_idx in self.model_config.swa_attention_layer_ids
        ]
        # Update back to model_config.
        # 将过滤后的层 ID 写回 model_config
        self.model_config.swa_attention_layer_ids = swa_attention_layer_ids
        self.model_config.full_attention_layer_ids = full_attention_layer_ids

        # Apply torchao quantization
        # 应用 torchao 量化（若模型加载时尚未应用）
        torchao_applied = getattr(self.model, "torchao_applied", False)
        # In layered loading, torchao may have been applied
        if not torchao_applied:
            apply_torchao_config_to_model(
                self.model, get_global_server_args().torchao_config
            )

        # Apply torch TP if the model supports it
        # 若模型支持 torch 原生张量并行，则应用
        supports_torch_tp = getattr(self.model, "supports_torch_tp", False)
        if self.tp_size > 1 and supports_torch_tp:
            self.apply_torch_tp()

        # Init lora
        # 初始化 LoRA 适配器管理器，并预分配 CUDA Graph 所需的 MoE 中间缓冲区
        if server_args.enable_lora:
            self.init_lora_manager()
            if not server_args.disable_cuda_graph:
                # Phase 1 of LoRA CUDA graph init: pre-allocate large MoE
                # intermediate buffers before init_memory_pool() so memory
                # profiling accounts for them.  Phase 2 (dense LoRA batch
                # metadata) is handled in CudaGraphRunner.__init__() via
                # lora_manager.init_cuda_graph_batch_info().
                self._init_lora_cuda_graph_moe_buffers()

        # Enable batch invariant mode
        # 启用批次不变模式，确保相同输入产生确定性输出（用于精度对齐）
        if server_args.enable_deterministic_inference:
            from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

            enable_batch_invariant_mode()

        # Deduce KV cache dtype
        # 推断 KV Cache 的数据类型（如 bf16/fp8 等）
        self.configure_kv_cache_dtype()

        # Init memory pool and attention backends
        # 初始化显存池并配置注意力后端（flashinfer/triton 等）
        self.init_memory_pool(pre_model_load_memory)

        # Init ngram embedding token table
        # 初始化 ngram embedding 的 token 查表（用于投机解码）
        self.maybe_init_ngram_embedding()

        # Init hisparse coordinator (must happen before CUDA graph capture)
        # 初始化 HiSparse 稀疏注意力协调器（必须在 CUDA Graph 捕获前完成）
        if self.enable_hisparse:
            from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
            from sglang.srt.mem_cache.sparsity import parse_hisparse_config

            hisparse_cfg = parse_hisparse_config(self.server_args)
            self.hisparse_coordinator = HiSparseCoordinator(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                top_k=hisparse_cfg.top_k,
                device_buffer_size=hisparse_cfg.device_buffer_size,
                device=self.device,
                tp_group=(
                    self.attention_tp_group.cpu_group
                    if self.server_args.enable_dp_attention
                    else self.tp_group.cpu_group
                ),
                host_to_device_ratio=hisparse_cfg.host_to_device_ratio,
            )

        # Init routed experts capturer
        # 初始化 MoE 路由专家捕获器（用于统计和分析专家路由情况）
        self.init_routed_experts_capturer()

        # TODO: Refactor device-specific init branches into platform interface (separate PR).
        # Must be called BEFORE init_device_graphs() so CUDA graph capture
        # runs with aux hidden state capture enabled.
        # 初始化辅助隐状态捕获（投机解码所需，必须在图捕获前调用）
        self.init_aux_hidden_state_capture()

        if self.device == "cuda" or self.device == "musa":
            # CUDA/MUSA 设备：依次初始化 cuBLAS、注意力后端、kernel 预热、AllReduce workspace 和设备图
            self.init_cublas()
            self.init_attention_backend()
            self.kernel_warmup()
            self._pre_initialize_flashinfer_allreduce_workspace()
            self.init_device_graphs()
        elif self.device in ["npu", "cpu"]:
            # NPU/CPU 设备：仅初始化注意力后端和设备图
            self.init_attention_backend()
            self.init_device_graphs()
        elif current_platform.is_out_of_tree():
            # 第三方平台：初始化注意力后端，若平台支持 CUDA Graph 则构建图
            self.init_attention_backend()
            if current_platform.support_cuda_graph():
                self.init_device_graphs()
            else:
                self.graph_runner = None
                self.graph_mem_usage = 0
        else:
            self.graph_runner = None
            self.graph_mem_usage = 0
            self.init_attention_backend()

        if server_args.forward_hooks:
            # 注册用户自定义前向钩子（用于监控/调试）
            register_forward_hooks(self.model, server_args.forward_hooks)

        # Initialize piecewise CUDA graph
        # 初始化分段 CUDA Graph（将 prefill 和 decode 分段捕获以支持更大批次）
        self.init_piecewise_cuda_graphs()

        self.prealloc_symmetric_memory_pool()  # 预分配对称内存池（用于 AllReduce）

    def init_routed_experts_capturer(self):
        # 初始化 MoE 路由专家捕获器：收集推理时每个 token 被路由到的专家信息
        if not self.server_args.disable_shared_experts_fusion and hasattr(
            self.model, "num_fused_shared_experts"
        ):
            # 若融合共享专家，获取融合数量
            num_fused_shared_experts = self.model.num_fused_shared_experts
        else:
            num_fused_shared_experts = 0

        # 创建并设置全局捕获器实例
        set_global_experts_capturer(
            RoutedExpertsCapturer.create(
                enable=get_global_server_args().enable_return_routed_experts,
                model_config=self.model_config,
                num_fused_shared_experts=num_fused_shared_experts,
                num_tokens=self.max_total_num_tokens + self.page_size,
                max_running_requests=self.max_running_requests,
                device=self.device,
            )
        )

    def init_aux_hidden_state_capture(self):
        """Configure auxiliary hidden state capture for speculative decoding.

        Must be called before CUDA graph capture so the captured graphs
        include aux hidden state output paths.
        """
        # 配置辅助隐状态捕获：EAGLE3 和 DFlash 投机解码需要捕获中间隐状态
        if self.eagle_use_aux_hidden_state:
            # 通知模型需要捕获哪些层的辅助隐状态（EAGLE3 专用）
            self.model.set_eagle3_layers_to_capture(
                self.eagle_aux_hidden_state_layer_ids
            )
        if self.dflash_use_aux_hidden_state:
            if not hasattr(self.model, "set_dflash_layers_to_capture"):
                raise ValueError(
                    f"Model {self.model.__class__.__name__} does not implement "
                    "set_dflash_layers_to_capture, which is required for DFLASH."
                )
            # 通知模型需要捕获哪些目标层的隐状态（DFlash 专用）
            self.model.set_dflash_layers_to_capture(self.dflash_target_layer_ids)

    def remote_instance_init_transfer_engine(self):
        # 初始化 Mooncake TransferEngine 用于远程实例间的 RDMA 权重传输
        try:
            from mooncake.engine import TransferEngine
        except ImportError as e:
            logger.warning(
                "Please install mooncake for using remote instance transfer engine: pip install mooncake"
            )
            return
        self.remote_instance_transfer_engine = TransferEngine()
        local_ip = get_local_ip_auto()  # 自动检测本机 IP
        self.remote_instance_transfer_engine.initialize(
            local_ip, "P2PHANDSHAKE", "rdma", envs.MOONCAKE_DEVICE.get()
        )
        # 将本地 IP 和 RPC 端口组合为会话 ID 字符串
        self.remote_instance_transfer_engine_session_id = NetworkAddress(
            local_ip, self.remote_instance_transfer_engine.get_rpc_port()
        ).to_host_port_str()

    def _register_to_engine_info_bootstrap(self):
        """Register transfer engine info with the EngineInfoBootstrapServer via HTTP PUT.

        The bootstrap server runs on node_rank==0. For multi-node setups, the
        host is derived from dist_init_addr. For single-node, use 127.0.0.1.
        """
        # 将传输引擎信息通过 HTTP PUT 上报到 bootstrap server，以便远程实例发现本节点
        import requests as http_requests

        if self.server_args.dist_init_addr:
            # Multi-node: bootstrap server is on the head node (node_rank==0).
            # Derive host from dist_init_addr (shared across all nodes).
            # 多节点场景：从 dist_init_addr 解析 bootstrap server 地址
            bootstrap_host = (
                NetworkAddress.parse(self.server_args.dist_init_addr).resolved().host
            )
        else:
            bootstrap_host = "127.0.0.1"  # 单节点直接使用本地回环地址

        bootstrap_port = self.server_args.engine_info_bootstrap_port
        bootstrap_na = NetworkAddress(bootstrap_host, bootstrap_port)
        url = f"{bootstrap_na.to_url()}/register_transfer_engine_info"

        # 构造包含 tp_rank 和传输引擎会话信息的注册 payload
        payload = {
            "tp_rank": self.tp_rank,
            "transfer_engine_info": {
                "session_id": self.remote_instance_transfer_engine_session_id,
                "weights_info_dict": self.remote_instance_transfer_engine_weight_info,
            },
        }

        try:
            resp = http_requests.put(url, json=payload, timeout=5)
            if resp.status_code == 200:
                logger.info(
                    f"Registered transfer engine info for tp_rank={self.tp_rank} "
                    f"with bootstrap server at {bootstrap_na}"
                )
            else:
                logger.error(
                    f"Failed to register transfer engine info for tp_rank={self.tp_rank}: "
                    f"{resp.status_code}, {resp.text}"
                )
        except Exception as e:
            logger.error(
                f"Failed to register transfer engine info for tp_rank={self.tp_rank}: {e}"
            )

    def _publish_modelexpress_metadata(self):
        """Publish metadata to ModelExpress server (seed mode).

        Supports two transport backends:
        - transfer_engine: publishes TransferEngine session_id (Mooncake)
        - nixl: creates NIXL agent, registers tensors, publishes nixl_metadata
        """
        # 将模型权重元数据发布到 ModelExpress 服务器，支持 transfer_engine 和 nixl 两种传输后端
        try:
            from modelexpress import p2p_pb2
            from modelexpress.client import MxClient
        except ImportError as exc:
            raise ImportError(
                "ModelExpress support requires the 'modelexpress' package. "
                "Install it with: pip install modelexpress"
            ) from exc

        model_name = (
            self.server_args.modelexpress_model_name or self.server_args.model_path
        )
        mx_url = self.server_args.modelexpress_url
        transport = self.server_args.modelexpress_transport

        # Build SourceIdentity for this instance
        # 构建本节点的身份信息（模型名、并行配置、量化类型等）
        identity = p2p_pb2.SourceIdentity(
            model_name=model_name,
            backend_framework=p2p_pb2.BACKEND_FRAMEWORK_SGLANG,
            tensor_parallel_size=self.server_args.tp_size,
            pipeline_parallel_size=self.server_args.pp_size,
            expert_parallel_size=self.server_args.ep_size,
            dtype=self.server_args.dtype or "",
            quantization=self.server_args.quantization or "",
        )

        if transport == "nixl":
            # 使用 NIXL RDMA 传输构建 worker 元数据
            worker, tensor_count = self._build_nixl_worker_metadata(p2p_pb2)
        else:
            # 使用 Mooncake TransferEngine 传输构建 worker 元数据
            worker, tensor_count = self._build_transfer_engine_worker_metadata(p2p_pb2)
            if worker is None:
                return

        # Generate a unique worker_id for this running instance
        # 为当前运行实例生成唯一 worker_id
        worker_id = str(uuid.uuid4())

        mx_client = MxClient(server_url=mx_url)
        try:
            logger.info(
                "ModelExpress source [%s]: publishing metadata for model=%s, "
                "tp_rank=%d, %d tensors, worker_id=%s",
                transport,
                model_name,
                self.tp_rank,
                tensor_count,
                worker_id,
            )
            # 发布元数据并更新状态为 READY
            mx_source_id = mx_client.publish_metadata(identity, worker, worker_id)
            mx_client.update_status(
                mx_source_id=mx_source_id,
                worker_id=worker_id,
                worker_rank=self.tp_rank,
                status=p2p_pb2.SOURCE_STATUS_READY,
            )
            logger.info(
                "ModelExpress source: published ready for model=%s, "
                "tp_rank=%d, mx_source_id=%s",
                model_name,
                self.tp_rank,
                mx_source_id,
            )
        finally:
            mx_client.close()

    def _build_transfer_engine_worker_metadata(self, p2p_pb2):
        """Build WorkerMetadata using TransferEngine session_id."""
        # 构建基于 Mooncake TransferEngine 的 worker 元数据
        session_id = self.remote_instance_transfer_engine_session_id
        weight_info = self.remote_instance_transfer_engine_weight_info

        if not session_id or weight_info is None:
            logger.warning(
                "ModelExpress source: skipping publish -- "
                "TransferEngine not initialized or no weight info"
            )
            return None, 0

        # 遍历权重信息，构建每个 tensor 的描述符（地址、大小、设备 ID）
        tensors = []
        for name, (addr, numel, element_size) in weight_info.items():
            tensors.append(
                p2p_pb2.TensorDescriptor(
                    name=name,
                    addr=addr,
                    size=numel * element_size,
                    device_id=self.gpu_id,
                )
            )

        worker = p2p_pb2.WorkerMetadata(
            worker_rank=self.tp_rank,
            transfer_engine_session_id=session_id,
            tensors=tensors,
        )
        return worker, len(tensors)

    def _build_nixl_worker_metadata(self, p2p_pb2):
        """Build WorkerMetadata using NIXL agent for RDMA transfers."""
        # 构建基于 NIXL RDMA 的 worker 元数据（用于高性能跨节点权重传输）
        from modelexpress.nixl_transfer import NixlTransferManager

        agent_name = f"sglang-seed-rank{self.tp_rank}-{uuid.uuid4().hex[:8]}"
        nixl_mgr = NixlTransferManager(agent_name, self.gpu_id)
        nixl_mgr.initialize()

        # Collect model tensors for NIXL registration
        # 收集模型参数 tensor 用于 NIXL 注册
        model_tensors = {}
        for name, param in self.model.named_parameters():
            t = param.data
            if t.is_contiguous():
                model_tensors[name] = t  # 连续 tensor 直接注册
            else:
                # Non-contiguous tensors: register underlying storage as byte view
                # 非连续 tensor：注册底层存储的字节视图
                sv = torch.empty(0, dtype=torch.uint8, device=t.device).set_(
                    t.untyped_storage()
                )
                if sv.data_ptr() not in {v.data_ptr() for v in model_tensors.values()}:
                    model_tensors[f"{name}.__storage"] = sv

        nixl_metadata = nixl_mgr.register_tensors(model_tensors)

        # Build tensor descriptors from registered tensors
        # 从已注册的 tensor 构建 protobuf 描述符
        tensors = []
        for td in nixl_mgr.tensor_descriptors:
            tensors.append(
                p2p_pb2.TensorDescriptor(
                    name=td.name,
                    addr=td.addr,
                    size=td.size,
                    device_id=td.device_id,
                    dtype=td.dtype,
                )
            )

        worker = p2p_pb2.WorkerMetadata(
            worker_rank=self.tp_rank,
            nixl_metadata=nixl_metadata,
            tensors=tensors,
        )

        # Keep reference alive so NIXL agent isn't garbage collected
        # 保持 nixl_mgr 引用，防止被 GC 回收导致 NIXL agent 失效
        self._nixl_manager = nixl_mgr

        return worker, len(tensors)

    def model_specific_adjustment(self):
        # 针对特定模型类型进行参数自动调整
        server_args = self.server_args

        if self.is_multimodal:
            if not self.is_multimodal_chunked_prefill_supported:
                # 多模态模型不支持分块 prefill 时自动关闭
                server_args.chunked_prefill_size = -1
                logger.info(
                    f"Automatically turn off --chunked-prefill-size as it is not supported for "
                    f"{self.model_config.hf_config.model_type}"
                )

        if (
            not self.use_mla_backend
            or server_args.attention_backend
            not in CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS
        ):
            # 非 MLA 后端或不支持分块前缀缓存的后端，强制关闭此功能
            server_args.disable_chunked_prefix_cache = True

        if not server_args.disable_chunked_prefix_cache:
            log_info_on_rank0(logger, "Chunked prefix cache is turned on.")

    def check_quantized_moe_compatibility(self):
        # 检查量化 MoE 模型与当前 TP/EP 配置的兼容性
        if (
            quantization_config := getattr(
                self.model_config.hf_config, "quantization_config", None
            )
        ) is not None and (
            weight_block_size := quantization_config.get("weight_block_size", None)
        ) is not None:
            weight_block_size_n = weight_block_size[0]

            # tp_size 必须能被 ep_size 整除
            if self.tp_size % self.moe_ep_size != 0:
                raise ValueError(
                    f"tp_size {self.tp_size} must be divisible by ep_size {self.moe_ep_size}"
                )
            # 每个 EP rank 分配的 TP 大小
            moe_tp_size = self.tp_size // self.moe_ep_size // self.moe_dp_size

            moe_intermediate_size = getattr(
                self.model_config.hf_text_config, "moe_intermediate_size", None
            )
            if moe_intermediate_size is None:
                return

            # 检查 MoE 中间维度能否被 moe_tp_size 整除
            if moe_intermediate_size % moe_tp_size != 0:
                raise ValueError(
                    f"moe_intermediate_size {moe_intermediate_size} must be divisible by moe_tp_size ({moe_tp_size}) which is tp_size ({self.tp_size}) divided by moe_ep_size ({self.moe_ep_size})."
                )

            # 检查量化块大小与 TP 分片后维度的对齐关系（aiter 后端例外）
            if (
                moe_intermediate_size // moe_tp_size
            ) % weight_block_size_n != 0 and not _use_aiter:
                raise ValueError(
                    f"For quantized MoE models, please make sure ({moe_intermediate_size=} / {moe_tp_size=}) % {weight_block_size_n=} == 0 "
                    f"where moe_tp_size is equal to tp_size ({self.tp_size}) divided by ep_size ({self.moe_ep_size}). "
                    f"You can fix this by setting arguments `--tp` and `--ep` correctly."
                )

    def init_torch_distributed(self):
        # 初始化 PyTorch 分布式环境，设置通信后端、进程组和模型并行配置
        tic = time.perf_counter()
        logger.info("Init torch distributed begin.")

        try:
            # 设置当前进程使用的 GPU 设备
            torch.get_device_module(self.device).set_device(self.gpu_id)
        except Exception:
            logger.warning(
                f"Context: {self.device=} {self.gpu_id=} {os.environ.get('CUDA_VISIBLE_DEVICES')=} {self.tp_rank=} {self.tp_size=}"
            )
            raise

        # 根据设备类型选择分布式通信后端（nccl/gloo/mooncake 等）
        backend = get_default_distributed_backend(self.device)
        if self.device == "cuda" and self.server_args.elastic_ep_backend == "mooncake":
            # 使用 Mooncake 弹性 EP 后端时切换为 mooncake 通信
            backend = "mooncake"
            if self.server_args.mooncake_ib_device:
                mooncake_ib_device = self.server_args.mooncake_ib_device.split(",")
                try:
                    from mooncake import ep as mooncake_ep

                    mooncake_ep.set_device_filter(mooncake_ib_device)
                except:
                    pass  # A warning will be raised in `init_distributed_environment`

        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        if not self.server_args.enable_p2p_check:
            # 跳过 P2P 访问检查以加速初始化（可能牺牲安全性）
            monkey_patch_p2p_access_check()

        # Allow external orchestrators (e.g. trainpi) to override the distributed
        # init method.  When set to "env://", torch uses MASTER_ADDR/MASTER_PORT
        # env-vars and an externally-created TCPStore, completely avoiding port
        # conflicts with intra-host collocation.
        # 支持通过环境变量覆盖分布式初始化方法（便于与外部训练框架集成）
        dist_init_method_override = envs.SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE.get()
        if dist_init_method_override:
            dist_init_method = dist_init_method_override
        elif self.server_args.dist_init_addr:
            na = NetworkAddress.parse(self.server_args.dist_init_addr)
            dist_init_method = na.to_tcp()
        else:
            # 默认使用 host:port 方式进行分布式初始化
            dist_init_method = NetworkAddress(
                self.server_args.host or "127.0.0.1", self.dist_port
            ).to_tcp()
        # 配置 AllReduce 策略
        set_custom_all_reduce(not self.server_args.disable_custom_all_reduce)
        set_mscclpp_all_reduce(self.server_args.enable_mscclpp)
        set_torch_symm_mem_all_reduce(self.server_args.enable_torch_symm_mem)

        if not self.is_draft_worker:
            if self.device == "cpu":
                if _is_cpu_amx_available or _is_cpu_arm64:
                    # Bind OpenMP threads to CPU cores
                    # 绑定 OpenMP 线程到指定 CPU 核（提升 CPU 推理性能）
                    torch.ops.sgl_kernel.init_cpu_threads_env(self.local_omp_cpuid)

                    # Set local size to hint SGLang to use shared memory based AllReduce
                    # 设置本地进程数，启用基于共享内存的 AllReduce
                    os.environ["LOCAL_SIZE"] = str(self.tp_size)
                    torch.ops.sgl_kernel.initialize(self.tp_size, self.tp_rank)

                    @torch.library.register_fake("sgl_kernel::shm_allgather")
                    def _(data, dim):
                        # 注册 allgather 的 fake 实现（用于 torch.compile 推导形状）
                        return torch.cat([data] * self.tp_size, dim=dim)

                else:
                    logger.warning(
                        "init_cpu_threads_env and shared memory based AllReduce is disabled, only intel amx backend and arm64 are supported"
                    )

            # Only initialize the distributed environment on the target model worker.
            # 初始化全局分布式环境（主模型 worker 才需要，草稿 worker 复用已有进程组）
            init_distributed_environment(
                backend=backend,
                world_size=self.tp_size * self.pp_size,  # 全局 world size = TP × PP
                rank=self.tp_size * self.pp_rank + self.tp_rank,  # 全局 rank
                local_rank=self.gpu_id,
                distributed_init_method=dist_init_method,
                timeout=self.server_args.dist_timeout,
                moe_a2a_backend=self.server_args.moe_a2a_backend,
                recovered_rank=self.server_args.elastic_ep_rejoin,
            )
            # 初始化各维度模型并行进程组（TP/PP/EP/DP 等）
            initialize_model_parallel(
                tensor_model_parallel_size=self.tp_size,
                attention_data_parallel_size=self.dp_size,
                pipeline_model_parallel_size=self.pp_size,
                expert_model_parallel_size=self.moe_ep_size,
                attention_context_model_parallel_size=self.attn_cp_size,
                moe_data_model_parallel_size=self.moe_dp_size,
                duplicate_tp_group=self.server_args.enable_pdmux,
                enable_symm_mem=self.server_args.enable_symm_mem,
                recovered_rank=self.server_args.elastic_ep_rejoin,
            )
            # 初始化 DP Attention 相关配置
            initialize_dp_attention(
                server_args=self.server_args,
                model_config=self.model_config,
            )
            if is_npu():
                register_sgl_tp_rank(self.gpu_id)

            # Pre-warm NCCL/RCCL to eliminate cold-start latency in first request
            # Controlled by --pre-warm-nccl flag (default: enabled on AMD GPUs)
            # 预热 NCCL/RCCL 通信，消除第一次请求的冷启动延迟
            if self.server_args.pre_warm_nccl and (
                self.tp_size > 1 or self.pp_size > 1 or self.moe_ep_size > 1
            ):
                warmup_start = time.perf_counter()
                tp_group_handle = get_tp_group().device_group

                # Single warmup all_reduce to initialize NCCL/RCCL communicator
                # 执行一次 all_reduce 以初始化 NCCL/RCCL 通信器
                warmup_tensor = torch.zeros(1, device=torch.cuda.current_device())
                dist.all_reduce(warmup_tensor, group=tp_group_handle)
                torch.cuda.synchronize()

                warmup_elapsed = time.perf_counter() - warmup_start
                logger.info(
                    f"NCCL/RCCL warmup completed in {warmup_elapsed:.3f}s "
                    f"(tp_size={self.tp_size}, pp_size={self.pp_size}, ep_size={self.moe_ep_size})"
                )

        # 获取加载模型前的显存量（用于后续计算模型权重占用）
        pre_model_load_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=get_world_group().world_size > 1,
            cpu_group=get_world_group().cpu_group,
        )
        # 保存常用进程组引用
        self.tp_group = get_tp_group()
        self.pp_group = get_pp_group()
        self.attention_tp_group = get_attention_tp_group()

        # Check memory for tensor parallelism
        # 检测各 GPU 可用显存是否均衡（不均衡可能导致 OOM）
        local_gpu_memory = get_available_gpu_memory(self.device, self.gpu_id)
        if self.tp_size > 1 and not self.is_draft_worker:
            if pre_model_load_memory < local_gpu_memory * 0.9:
                msg = "The memory capacity is unbalanced. Some GPUs may be occupied by other processes. "
                msg += f"{pre_model_load_memory=}, {local_gpu_memory=}, {local_gpu_memory * 0.9=}"
                if envs.SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK.get():
                    raise RuntimeError(msg)
                else:
                    logger.warning(msg)

        logger.info(
            f"Init torch distributed ends. elapsed={time.perf_counter() - tic:.2f} s, "
            f"mem usage={(before_avail_memory - local_gpu_memory):.2f} GB"
        )
        return pre_model_load_memory

    def init_shared_mooncake_transfer_engine(self):
        """
        Need MooncakeTransferEngine when:
        1) PD disaggregation uses mooncake for KV transfer (prefill/decode)
        2) HiCache uses mooncake storage backend
        3) Encoder disaggregation uses mooncake
        """
        # 判断是否需要初始化 Mooncake 传输引擎（多种场景下的 KV/权重传输）
        use_mooncake_te = (
            (
                # PD 分离模式（Prefill-Decode 分离）使用 Mooncake 传输 KV
                self.server_args.disaggregation_mode != "null"
                and self.server_args.disaggregation_transfer_backend == "mooncake"
            )
            or (
                # HiCache 分层缓存使用 Mooncake 存储后端且复用传输引擎
                self.server_args.enable_hierarchical_cache
                and self.server_args.hicache_storage_backend == "mooncake"
                and envs.SGLANG_HICACHE_MOONCAKE_REUSE_TE.get()
            )
            or (
                # Encoder 分离模式使用 Mooncake 传输
                self.server_args.encoder_only
                and self.server_args.encoder_transfer_backend == "mooncake"
            )
            or (
                self.server_args.language_only
                and self.server_args.encoder_transfer_backend == "mooncake"
            )
            or (
                # 弹性专家备份场景
                self.server_args.enable_elastic_expert_backup
                and self.server_args.elastic_ep_backend is not None
            )
        )

        if use_mooncake_te:
            from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
                init_mooncake_transfer_engine,
            )

            # 初始化 Mooncake 传输引擎（绑定到本机 IP 和 GPU）
            init_mooncake_transfer_engine(
                hostname=get_local_ip_auto(),
                gpu_id=self.gpu_id,
                ib_device=(
                    self.server_args.disaggregation_ib_device
                    or self.server_args.mooncake_ib_device
                ),
            )

    def load_model(self):
        # 模型权重加载主流程：检测显卡能力、配置量化、实际加载权重
        tic_total = time.perf_counter()
        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Load weight begin. avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        # This can reduce thread conflicts and speed up weight loading.
        # 限制 CPU 线程数为 1，减少线程竞争从而加速权重加载
        if self.device != "cpu":
            torch.set_num_threads(1)
        if self.device == "cuda":
            if torch.cuda.get_device_capability()[0] < 8:
                # sm80 以下（如 T4/V100）不支持 bfloat16，强制使用 float16
                logger.info(
                    "Compute capability below sm80. Use float16 due to lack of bfloat16 support."
                )
                self.server_args.dtype = "float16"
                self.model_config.dtype = torch.float16
                if torch.cuda.get_device_capability()[1] < 5:
                    raise RuntimeError("SGLang only supports sm75 and above.")

        set_cuda_arch()  # 根据 GPU 架构设置编译标志

        # Prepare the model config
        # 构建 ModelOpt（TensorRT-LLM 量化工具）配置
        from sglang.srt.configs.modelopt_config import ModelOptConfig

        modelopt_config = ModelOptConfig(
            quant=self.server_args.modelopt_quant,
            checkpoint_restore_path=self.server_args.modelopt_checkpoint_restore_path,
            checkpoint_save_path=self.server_args.modelopt_checkpoint_save_path,
            export_path=self.server_args.modelopt_export_path,
            quantize_and_serve=self.server_args.quantize_and_serve,
        )

        # 构建加载配置，包含格式、下载目录、远程加载参数和 ModelExpress 参数等
        self.load_config = LoadConfig(
            load_format=self.server_args.load_format,
            download_dir=self.server_args.download_dir,
            model_loader_extra_config=self.server_args.model_loader_extra_config,
            tp_rank=self.tp_rank,
            remote_instance_weight_loader_seed_instance_ip=self.server_args.remote_instance_weight_loader_seed_instance_ip,
            remote_instance_weight_loader_seed_instance_service_port=self.server_args.remote_instance_weight_loader_seed_instance_service_port,
            remote_instance_weight_loader_send_weights_group_ports=self.server_args.remote_instance_weight_loader_send_weights_group_ports,
            remote_instance_weight_loader_backend=self.server_args.remote_instance_weight_loader_backend,
            remote_instance_weight_loader_transfer_engine=self.remote_instance_transfer_engine,
            modelexpress_url=self.server_args.modelexpress_url,
            modelexpress_model_name=self.server_args.modelexpress_model_name
            or self.server_args.model_path,
            modelexpress_tp_size=self.server_args.tp_size,
            modelexpress_pp_size=self.server_args.pp_size,
            modelexpress_ep_size=self.server_args.ep_size,
            modelexpress_dtype=self.server_args.dtype,
            modelexpress_quantization=self.server_args.quantization or "",
            modelexpress_transport=self.server_args.modelexpress_transport,
            modelopt_config=modelopt_config,
            rl_quant_profile=self.server_args.rl_quant_profile,
            draft_model_idx=self.draft_model_idx,
        )
        if self.device == "cpu":
            # CPU 推理时调整不对齐的 TP 配置
            self.model_config = adjust_config_with_unaligned_cpu_tp(
                self.model_config, self.load_config, self.tp_size
            )

        if (
            self.server_args.load_format == LoadFormat.REMOTE_INSTANCE
            and self.server_args.remote_instance_weight_loader_backend
            == RemoteInstanceWeightLoaderBackend.NCCL
        ):
            if self.tp_rank == 0:
                # rank 0 在独立线程中发送初始化权重传输组的请求
                instance_ip = NetworkAddress.resolve_host(socket.gethostname())
                t = threading.Thread(
                    target=trigger_init_weights_send_group_for_remote_instance_request,
                    args=(
                        self.server_args.remote_instance_weight_loader_seed_instance_ip,
                        self.server_args.remote_instance_weight_loader_seed_instance_service_port,
                        self.server_args.remote_instance_weight_loader_send_weights_group_ports,
                        instance_ip,
                    ),
                )
                t.start()

        # Load the model
        # Remove monkey_patch when linear.py quant remove dependencies with vllm
        # 临时 patch vllm 并行状态（兼容量化层中对 vllm 的依赖）
        monkey_patch_vllm_parallel_state()

        # 判断是否需要 CPU 备份权重（用于在线更新或草稿模型）
        enable_cpu_backup = self.server_args.enable_weights_cpu_backup or (
            self.is_draft_worker and self.server_args.enable_draft_weights_cpu_backup
        )
        # 在内存节省适配器的权重区域内加载模型
        with self.memory_saver_adapter.region(
            GPU_MEMORY_TYPE_WEIGHTS,
            enable_cpu_backup=enable_cpu_backup,
        ):
            self.loader = get_model_loader(
                load_config=self.load_config,
                model_config=self.model_config,
            )
            # 实际执行模型权重加载（最耗时步骤）
            self.model = self.loader.load_model(
                model_config=self.model_config,
                device_config=DeviceConfig(self.device, self.gpu_id),
            )
            if hasattr(self.loader, "remote_instance_transfer_engine_weight_info"):
                # 若加载器已包含权重信息，同步到 runner
                self.remote_instance_transfer_engine_weight_info = (
                    self.loader.remote_instance_transfer_engine_weight_info
                )
        # Cache needs to be cleared after loading model weights (in the self.loader.load_model function).
        # To avoid conflict with memory_saver_adapter.region, empty_cache operation is now moved here.
        # 加载完成后清空缓存（NPU 需要显式调用）
        if _is_npu:
            torch.npu.empty_cache()
        monkey_patch_vllm_parallel_state(reverse=True)  # 恢复 vllm 并行状态

        # Publish metadata to ModelExpress if running as seed source
        # 若本实例为 ModelExpress seed 源，发布权重元数据
        if self.server_args.modelexpress_source:
            # Seed loads via DefaultModelLoader (load_format=auto), which doesn't
            # call register_memory_region(). Do it here so weight_info is populated.
            if (
                self.remote_instance_transfer_engine_weight_info is None
                and self.remote_instance_transfer_engine is not None
            ):
                from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
                    register_memory_region,
                )

                self.remote_instance_transfer_engine_weight_info = (
                    register_memory_region(
                        self.model, self.remote_instance_transfer_engine
                    )
                )
            self._publish_modelexpress_metadata()

        get_offloader().post_init()  # CPU offload 后处理初始化

        # Register model for layerwise NVTX profiling if enabled
        # 若启用逐层 NVTX 性能分析，为模型注册 PyTorch 钩子
        if self.server_args.enable_layerwise_nvtx_marker:
            self.pyt_hooks = PytHooks()
            self.pyt_hooks.register_hooks(self.model, module_prefix="model")

        if self.server_args.kv_cache_dtype == "fp8_e4m3":
            if self.server_args.quantization_param_path is not None:
                if callable(getattr(self.model, "load_kv_cache_scales", None)):
                    # 加载 FP8 KV Cache 的量化缩放因子
                    self.model.load_kv_cache_scales(
                        self.server_args.quantization_param_path
                    )
                    logger.info(
                        "Loaded KV cache scaling factors from %s",
                        self.server_args.quantization_param_path,
                    )
                else:
                    raise RuntimeError(
                        "Using FP8 KV cache and scaling factors provided but "
                        "model %s does not support loading scaling factors.",
                        self.model.__class__,
                    )
            else:
                # 使用默认缩放因子 1.0（可能降低精度）
                logger.warning(
                    "Using FP8 KV cache but no scaling factors "
                    "provided. Defaulting to scaling factors of 1.0. "
                    "This may lead to less accurate results!"
                )

        # Parse other args
        # 解析滑动窗口大小（SWA 或 chunked attention 场景）
        self.sliding_window_size = None
        if hasattr(self.model, "get_attention_sliding_window_size"):
            self.sliding_window_size = self.model.get_attention_sliding_window_size()
        elif (
            self.model_config.is_hybrid_swa
            and self.model_config.sliding_window_size is not None
        ):
            # sliding window field in model config may have different meaning for different kinds of models (e.g., dllm), here we only consider the sliding window in SWA model
            # 混合 SWA 模型专用：仅取 SWA 模型的滑动窗口大小
            self.sliding_window_size = self.model_config.sliding_window_size
        elif self.model_config.attention_chunk_size is not None:
            # 对于分块注意力，将 attention_chunk_size 作为滑动窗口大小
            self.sliding_window_size = self.model_config.attention_chunk_size
            logger.info(
                f"Setting sliding_window_size to be attention_chunk_size: {self.sliding_window_size}"
            )

        self.dtype = self.model_config.dtype  # 保存模型计算精度

        after_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        self.weight_load_mem_usage = before_avail_memory - after_avail_memory  # 权重占用显存量
        # Get quantization config from ModelConfig
        # This handles both config.json (standard) and hf_quant_config.json (ModelOpt)
        # 获取量化配置日志字符串（兼容标准和 ModelOpt 两种格式）
        quant_str = self.model_config.get_quantization_config_log_str()

        logger.info(
            f"Load weight end. "
            f"elapsed={time.perf_counter() - tic_total:.2f} s, "
            f"type={type(self.model).__name__}, "
            f"{quant_str + ', ' if quant_str else ''}"
            f"avail mem={after_avail_memory:.2f} GB, "
            f"mem usage={self.weight_load_mem_usage:.2f} GB."
        )
        if self.server_args.debug_tensor_dump_output_folder is not None:
            # 注册 tensor dump 前向钩子（用于调试中间激活）
            dump_folder = self.server_args.debug_tensor_dump_output_folder
            if self.spec_algorithm.is_eagle():
                role = "draft" if self.is_draft_worker else "target"
                dump_folder = os.path.join(dump_folder, role)
            register_forward_hook_for_model(
                self.model,
                dump_folder,
                self.server_args.debug_tensor_dump_layers,
                self.tp_size,
                self.tp_rank,
                self.pp_rank,
            )

        if dumper.may_enable:
            # 应用 dumper 源代码 patch 并注册非侵入式 dump 钩子
            dumper.apply_source_patches()
            dumper.register_non_intrusive_dumper(self.model)

        # Pre-expand RoPE cache before CUDA Graph capture
        # 预扩展 RoPE 位置编码缓存（避免长序列在 CUDA Graph 捕获后触发缓存重建）
        reserve_rope_cache_for_long_sequences(
            self.model,
            self.server_args,
            self.model_config,
            logger,
        )

        if self.server_args.elastic_ep_backend == "mooncake":
            # Mooncake does not support `monitored_barrier`
            # Mooncake 后端不支持 monitored_barrier，使用普通 barrier
            dist.barrier(group=get_tp_group().cpu_group)
        else:
            # Handle the case where some ranks do not finish loading.
            # 使用带超时的 monitored_barrier 检测慢速/异常 rank
            try:
                dist.monitored_barrier(
                    group=get_tp_group().cpu_group,
                    timeout=datetime.timedelta(
                        seconds=UNBALANCED_MODEL_LOADING_TIMEOUT_S
                    ),
                    wait_all_ranks=True,
                )
            except RuntimeError:
                raise ValueError(
                    f"TP rank {self.tp_rank} could finish the model loading, but there are other ranks that didn't finish loading. It is likely due to unexpected failures (e.g., OOM) or a slow node."
                ) from None

    def update_expert_location(
        self,
        new_expert_location_metadata: ExpertLocationMetadata,
        update_layer_ids: List[int],
    ):
        # 更新 MoE 专家位置元数据，并补充加载 P2P 传输过程中缺失的专家权重
        p2p_missing_logical_experts = self.expert_location_updater.update(
            self.model.routed_experts_weights_of_layer,
            new_expert_location_metadata,
            update_layer_ids=update_layer_ids,
            nnodes=self.server_args.nnodes,
            rank=self.tp_rank,
        )

        if len(p2p_missing_logical_experts) > 0:
            # Load the missing expert weights from disk
            # 有专家权重缺失，需从磁盘或 DRAM 备份中补充加载
            if callable(getattr(self.model, "generate_weight_name_filter", None)):
                # Filter and load only missing expert weights
                # 生成只过滤缺失专家权重的名称过滤器
                weight_name_filter = self.model.generate_weight_name_filter(
                    p2p_missing_logical_experts
                )
            else:
                # Do a full reload from disk/DRAM
                # 模型不支持精细过滤，执行全量权重重载
                logger.info(
                    "[Elastic EP] Model does not implement generate_weight_name_filter. "
                    "Performing full weight reload."
                )
                weight_name_filter = None

            if (
                self.expert_backup_client is not None
                and self.expert_backup_client.use_backup
            ):
                # Load the missing weights from the DRAM backup
                # 优先从 DRAM 备份中加载（速度更快）
                self.expert_backup_client.update_weights(weight_name_filter)
            else:
                # Load the missing weights from disk
                # 从磁盘加载缺失权重
                self.update_weights_from_disk(
                    get_global_server_args().model_path,
                    get_global_server_args().load_format,
                    weight_name_filter=weight_name_filter,
                )

    def maybe_recover_ep_ranks(self):
        # TODO(perf): `active_ranks.all()` on a CUDA tensor triggers host-device
        # synchronization, and this function is on the forward-path.
        # This check only runs when `--elastic-ep-backend` is enabled, so the
        # synchronization overhead does not propagate to other configs.
        # Leave for future optimization of the elastic EP path.
        # 检查是否有 EP rank 失活，若有则尝试恢复（弹性 EP 容错机制）
        if self.tp_group.active_ranks.all() and self.tp_group.active_ranks_cpu.all():
            return  # 所有 rank 均活跃，无需恢复

        tp_active_ranks = self.tp_group.active_ranks.detach().cpu().numpy()
        tp_active_ranks_cpu = self.tp_group.active_ranks_cpu.detach().numpy()
        tp_active_ranks &= tp_active_ranks_cpu  # 取 GPU 和 CPU 活跃状态的交集
        # NOTE: `ranks_to_recover` uses indices in `tp_group`. For the current
        # Mooncake elastic EP implementation we assume `--pp-size=1`, so the
        # tp-group index is the same as the global rank index.
        # 找出需要恢复的 rank 列表
        ranks_to_recover = [
            i for i in range(len(tp_active_ranks)) if not tp_active_ranks[i]
        ]

        # try_recover_ranks polls peer state via Mooncake EP backend.
        # Mooncake's internal semantics guarantee that all ranks observe
        # consistent peer readiness state, so collective operations below
        # are safe even though polling appears local.
        # 调用 Mooncake 恢复接口，成功后重置相关状态
        if ranks_to_recover and try_recover_ranks(ranks_to_recover):
            self.forward_pass_id = 0  # 重置 forward pass 计数器
            self.eplb_manager.reset_generator()
            # 广播新的专家位置元数据（从健康 rank 广播）
            broadcast_global_expert_location_metadata(
                src_rank=self._get_healthy_expert_location_src_rank(
                    invoked_in_elastic_ep_rejoin_path=False
                )
            )
            ElasticEPStateManager.instance().reset()

            # 广播随机种子以同步各 rank 状态
            broadcast_pyobj(
                [self.server_args.random_seed],
                get_world_group().rank,
                get_world_group().cpu_group,
                src=get_world_group().ranks[0],
            )
            logger.info(f"recover ranks {ranks_to_recover} done")

    def _get_healthy_expert_location_src_rank(
        self, invoked_in_elastic_ep_rejoin_path: bool
    ) -> int:
        # 查找第一个非 rejoin rank，作为广播专家位置元数据的源 rank
        world_group = get_world_group()
        # NOTE: do not key off `self.server_args.elastic_ep_rejoin` here.
        # A rank that was started as a rejoin rank may later act as a healthy
        # rank in a subsequent recovery cycle.
        local_rejoin_flag = bool(invoked_in_elastic_ep_rejoin_path)
        gathered_rejoin_flags = world_group.all_gather_object(local_rejoin_flag)

        for rank_in_group, is_rejoin_rank in enumerate(gathered_rejoin_flags):
            if not is_rejoin_rank:
                return world_group.ranks[rank_in_group]

        raise RuntimeError(
            "No healthy rank found for broadcasting expert location metadata. "
            "All ranks are marked as elastic_ep_rejoin."
        )

    def update_weights_from_disk(
        self,
        model_path: str,
        load_format: str,
        weight_name_filter: Optional[Callable[[str], bool]] = None,
        recapture_cuda_graph: bool = False,
    ) -> tuple[bool, str]:
        """Update engine weights in-place from the disk."""
        # 从磁盘在线更新模型权重（支持 RLHF 等场景下的权重热更新）
        logger.info(
            f"Update engine weights online from disk begin. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id, empty_cache=False):.2f} GB"
        )

        target_device = torch.device(self.device)
        self.model_config.model_path = model_path
        load_config = LoadConfig(load_format=load_format)

        # Only support DefaultModelLoader for now
        # 目前仅支持 DefaultModelLoader 进行在线权重更新
        loader = get_model_loader(load_config, self.model_config)
        if not isinstance(loader, DefaultModelLoader):
            message = f"Failed to get model loader: {loader}."
            return False, message

        def get_weight_iter(config):
            # 获取权重迭代器，支持按名称过滤（仅加载部分权重）
            iter = loader._get_weights_iterator(
                DefaultModelLoader.Source.init_new(config, self.model)
            )
            if weight_name_filter is not None:
                iter = (
                    (name, weight) for name, weight in iter if weight_name_filter(name)
                )

            return iter

        def model_load_weights(model, iter):
            # 将迭代器中的权重加载并后处理到目标设备
            loader.load_weights_and_postprocess(model, iter, target_device)
            return model

        with set_default_torch_dtype(self.model_config.dtype):
            try:
                iter = get_weight_iter(self.model_config)
            except Exception as e:
                message = f"Failed to get weights iterator: {e}."
                return False, message
            try:
                model = model_load_weights(self.model, iter)
            except Exception as e:
                # 权重更新失败：回滚到原始权重
                message = (
                    f"Failed to update weights: {e}.\nRolling back to original weights."
                )
                del iter
                gc.collect()
                iter = get_weight_iter(self.model_config)
                self.model = model_load_weights(self.model, iter)
                return False, message

        self.model = model
        self.server_args.model_path = model_path
        self.server_args.load_format = load_format
        self.load_config = load_config

        # 若需要重新捕获 CUDA Graph（权重更新后图可能失效）
        if recapture_cuda_graph and (
            self.device == "cuda"
            or self.device == "musa"
            or (
                current_platform.is_out_of_tree()
                and current_platform.support_cuda_graph()
            )
        ):
            self.init_device_graphs()

        logger.info("Update weights end.")
        return True, "Succeeded to update model weights."

    def init_weights_send_group_for_remote_instance(
        self,
        master_address,
        ports,
        group_rank,
        world_size,
        group_name,
        backend="nccl",
    ):
        # 为远程实例（如 decode 节点）初始化权重发送通信组
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        ports_list = ports.split(",")
        assert (
            len(ports_list) == self.tp_size
        ), f"Expected {self.tp_size} ports, but got {len(ports_list)} ports."
        # 每个 tp_rank 使用独立的端口以避免冲突
        group_port = ports_list[self.tp_rank]
        group_name = f"{group_name}_{group_port}_{self.tp_rank}"

        logger.info(
            f"init custom process group: tp_rank={self.tp_rank}, gpu_id={self.gpu_id}, master_address={master_address}, master_port={group_port}, "
            f"group_rank={group_rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
        )

        torch.cuda.empty_cache()
        success = False
        message = ""
        try:
            na = NetworkAddress(master_address, group_port)
            # 创建用于权重传输的自定义进程组
            self._weights_send_group[group_name] = init_custom_process_group(
                backend=backend,
                init_method=na.to_tcp(),
                world_size=world_size,
                rank=group_rank,
                group_name=group_name,
                device_id=torch.device("cuda", self.gpu_id),
            )
            dist.barrier(group=self._weights_send_group[group_name])
            success = True
            message = f"Succeeded to init group through {na.to_host_port_str()} group."
        except Exception as e:
            message = f"Failed to init group: {e}."
            logger.error(message)

        torch.cuda.empty_cache()
        return success, message

    def send_weights_to_remote_instance(
        self,
        master_address,
        ports,
        group_name,
    ):
        # 通过已初始化的通信组将本 rank 权重广播到远程实例
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        ports_list = ports.split(",")
        assert (
            len(ports_list) == self.tp_size
        ), f"Expected {self.tp_size} ports, but got {len(ports_list)} ports."
        group_port = ports_list[self.tp_rank]
        group_name = f"{group_name}_{group_port}_{self.tp_rank}"

        if self._weights_send_group[group_name] is not None:
            send_group = self._weights_send_group[group_name]
        else:
            message = f"Group {group_name} not in _weights_send_group list. Please call `init_weights_send_group_for_remote_instance` first."
            logger.error(message)
            return False, message

        torch.cuda.empty_cache()
        success = False
        na = NetworkAddress(master_address, group_port)
        message = ""
        try:
            # 遍历所有模型参数，通过 NCCL broadcast 发送给 rank 0 以外的进程
            for _, weights in self.model.named_parameters():
                torch.distributed.broadcast(
                    weights,
                    src=0,
                    group=send_group,
                )
            success = True
            message = f"Succeeded to send weights through {na.to_host_port_str()} {group_name}."
        except Exception as e:
            message = f"Failed to send weights: {e}."
            logger.error(message)

        # destroy the process group after sending weights
        # 发送完成后销毁临时通信组，释放资源
        del self._weights_send_group[group_name]
        torch.distributed.distributed_c10d.destroy_process_group(send_group)
        torch.cuda.empty_cache()
        return success, message

    def init_weights_update_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
    ):
        """Initialize the Torch process group for model parameter updates.

        `_model_update_group` is used in the RLHF workflow, where rank
        0 is the actor model in the training engine, and the other ranks are
        the inference engine, which is used for rollout.

        In the RLHF workflow, the training engine updates the model
        weights/parameters online, and broadcasts them to the inference
        engine through the `_model_update_group` process group.
        """
        # 初始化 RLHF 在线权重更新通信组，训练引擎（rank 0）将更新后的参数广播给推理引擎
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        rank = rank_offset + self.tp_rank  # 当前 rank 在更新组中的位置

        logger.info(
            f"init custom process group: master_address={master_address}, master_port={master_port}, "
            f"rank_offset={rank_offset}, rank={rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
        )

        try:
            na = NetworkAddress(master_address, master_port)
            # 创建权重更新专用进程组
            self._model_update_group[group_name] = init_custom_process_group(
                backend=backend,
                init_method=na.to_tcp(),
                world_size=world_size,
                rank=rank,
                group_name=group_name,
            )
            return True, "Succeeded to initialize custom process group."
        except Exception as e:
            message = f"Failed to initialize custom process group: {e}."
            logger.error(message)
            return False, message

    def destroy_weights_update_group(self, group_name):
        # 销毁权重更新通信组，释放分布式资源
        try:
            if group_name in self._model_update_group:
                pg = self._model_update_group.pop(group_name)
                torch.distributed.destroy_process_group(pg)
                return True, "Succeeded to destroy custom process group."
            else:
                return False, "The group to be destroyed does not exist."
        except Exception as e:
            message = f"Failed to destroy custom process group: {e}."
            logger.error(message)
            return False, message

    def update_weights_from_distributed(
        self,
        names,
        dtypes,
        shapes,
        group_name,
        load_format: Optional[str] = None,
    ):
        """
        Update specific parameter in the model weights online
        through `_model_update_group` process group.

        Args:
            name: the name of the parameter to be updated.
            dtype: the data type of the parameter to be updated.
            shape: the shape of the parameter to be updated.
        """
        # 通过分布式通信组接收并更新模型指定参数

        assert group_name in self._model_update_group, (
            f"Group {group_name} not in {list(self._model_update_group.keys())}. "
            "Please call `init_weights_update_group` first."
        )

        if load_format == "flattened_bucket":
            # 使用扁平化桶格式批量更新权重（减少通信次数）
            return self._update_bucketed_weights_from_distributed(
                names, dtypes, shapes, group_name
            )
        try:
            weights = []
            handles = []
            # 为每个参数分配空缓冲区，并发起异步广播
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                weight = torch.empty(shape, dtype=target_dtype, device=self.device)
                handles.append(
                    torch.distributed.broadcast(
                        weight,
                        src=0,
                        group=self._model_update_group[group_name],
                        async_op=True,  # 异步广播，稍后等待完成
                    )
                )
                weights.append((name, weight))
            # 等待所有广播完成
            for handle in handles:
                handle.wait()

            self.model.load_weights(weights)
            return True, "Succeeded to update parameter online."

        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

    def _update_bucketed_weights_from_distributed(
        self, names, dtypes, shapes, group_name
    ):
        # 使用扁平化桶方式批量接收权重：将多个 tensor 合并为一个扁平 tensor 传输
        try:
            named_tensors = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                named_tensors.append(
                    (name, torch.empty(shape, dtype=target_dtype, device=self.device))
                )
            bucket = FlattenedTensorBucket(named_tensors=named_tensors)
            flattened_tensor = bucket.get_flattened_tensor()
            # 一次性广播整个扁平 tensor
            torch.distributed.broadcast(
                flattened_tensor,
                src=0,
                group=self._model_update_group[group_name],
            )
            # 从扁平 tensor 还原各个参数 tensor
            reconstructed_tensors = bucket.reconstruct_tensors()
            self.model.load_weights(reconstructed_tensors)
            return True, f"Succeeded to update parameter online."
        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, Union[torch.Tensor, "LocalSerializedTensor"]]],
        load_format: Optional[str] = None,
    ):
        # 从本地 tensor 列表直接更新模型权重（无需分布式通信）
        monkey_patch_torch_reductions()
        if load_format == "flattened_bucket":
            # Handle flattened bucket format
            # 处理扁平化桶格式
            return self._update_weights_from_flattened_bucket(
                flattened_tensor_bucket_dict=named_tensors
            )

        # We need to get device after patch otherwise the device would be wrong
        # patch 后获取设备，确保 tensor 被放到正确设备
        self.device_module = torch.get_device_module(self.device)
        infered_device = self.device_module.current_device()

        # 解包（反序列化）tensor 并赋值到正确设备
        named_tensors = [
            (name, _unwrap_tensor(tensor, tp_rank=self.tp_rank, device=infered_device))
            for name, tensor in named_tensors
        ]
        if load_format == "direct":
            # 直接覆盖（不经过模型自定义 load_weights 逻辑）
            _model_load_weights_direct(self.model, named_tensors)
        elif load_format in self.server_args.custom_weight_loader:
            # 使用自定义权重加载器
            custom_loader = dynamic_import(load_format)
            custom_loader(self.model, named_tensors)
        elif load_format is None:
            self.model.load_weights(named_tensors)
        else:
            raise NotImplementedError(f"Unknown load_format={load_format}")
        return True, "Success"

    def _update_weights_from_flattened_bucket(
        self,
        flattened_tensor_bucket_dict,
    ):
        """Handle flattened bucket format for weight updates"""
        # 从扁平化桶字典中解包并加载权重
        flattened_tensor = flattened_tensor_bucket_dict["flattened_tensor"]
        metadata = flattened_tensor_bucket_dict["metadata"]

        # Convert metadata dict to our format
        # 将元数据转换为内部格式
        converted_metadata = []
        for meta in metadata:
            converted_meta = FlattenedTensorMetadata(
                name=meta.name,
                shape=meta.shape,
                dtype=meta.dtype,
                start_idx=meta.start_idx,
                end_idx=meta.end_idx,
                numel=meta.numel,
            )
            converted_metadata.append(converted_meta)

        # Create bucket and reconstruct tensors
        # 创建桶对象并从扁平 tensor 重建各参数
        bucket = FlattenedTensorBucket(
            flattened_tensor=flattened_tensor, metadata=converted_metadata
        )
        reconstructed_tensors = bucket.reconstruct_tensors()

        # Load the reconstructed tensors using the standard method
        # 使用标准接口加载重建后的权重
        self.model.load_weights(reconstructed_tensors)

        return True, "Success"

    def get_weights_by_name(
        self, name: str, truncate_size: int = 100
    ) -> Optional[torch.Tensor]:
        """Get the weights of the parameter by its name. Similar to `get_parameter` in Hugging Face.

        Only used for unit test with an unoptimized performance.
        For optimized performance, please use torch.save and torch.load.
        """
        # 按参数名获取模型权重（仅用于单测，性能未优化）
        # TODO: (chenyang) Add support for Qwen models.
        try:
            return self.model.get_weights_by_name(
                name, truncate_size, tp_size=self.tp_size
            )
        except Exception as e:
            logger.error(f"Error when getting parameter {name}: {e}")
            return None

    def init_lora_manager(self):
        # 初始化 LoRA 管理器，管理所有 LoRA 适配器的生命周期
        self.lora_manager = LoRAManager(
            base_model=self.model,
            base_hf_config=self.model_config.hf_config,
            max_loras_per_batch=self.server_args.max_loras_per_batch,
            load_config=self.load_config,
            dtype=self.dtype,
            server_args=self.server_args,
            lora_backend=self.server_args.lora_backend,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            max_lora_rank=self.server_args.max_lora_rank,
            target_modules=self.server_args.lora_target_modules,
            lora_paths=self.server_args.lora_paths,
        )

    def _init_lora_cuda_graph_moe_buffers(self):
        """Phase 1 of LoRA CUDA graph init: pre-allocate MoE intermediate buffers.

        Must be called before init_memory_pool() so that memory profiling
        sees the reduced available memory and sizes KV cache correctly.
        All MoE LoRA layers share one set of buffers (managed by the
        lora_backend) since they execute sequentially during forward.

        Phase 2 (dense LoRA batch metadata) is handled later in
        CudaGraphRunner.__init__() via lora_manager.init_cuda_graph_batch_info(),
        because it needs capture-time parameters (max_bs, num_tokens_per_bs)
        that are only available at that stage.
        """
        # LoRA CUDA Graph 初始化第一阶段：预分配 MoE 中间缓冲区
        from sglang.srt.lora.layers import FusedMoEWithLoRA

        max_bs = self.server_args.cuda_graph_max_bs
        max_loras = self.server_args.max_loras_per_batch
        # 只需为第一个 FusedMoEWithLoRA 模块分配（所有层共享一组缓冲区）
        for module in self.model.modules():
            if isinstance(module, FusedMoEWithLoRA):
                self.lora_manager.init_cuda_graph_moe_buffers(
                    max_bs, max_loras, self.dtype, module
                )
                logger.info(
                    f"Pre-allocated shared MoE LoRA CUDA graph buffers "
                    f"(max_bs={max_bs}, max_loras={max_loras})"
                )
                break

    def load_lora_adapter(self, lora_ref: LoRARef):
        """Load a new lora adapter from disk or huggingface."""
        # 从磁盘或 HuggingFace 动态加载 LoRA 适配器

        logger.info(
            f"LoRA adapter loading starts: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        result = self.lora_manager.load_lora_adapter(lora_ref)

        logger.info(
            f"LoRA adapter loading completes: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        return result

    def load_lora_adapter_from_tensors(
        self, lora_ref: LoRARef, tensors, config_dict, added_tokens_config=None
    ):
        # 从已加载的 tensor 字典直接注入 LoRA 适配器权重（无需磁盘 IO）
        logger.info(f"LoRA adapter loading from tensors starts: {lora_ref}.")
        result = self.lora_manager.load_lora_adapter_from_tensors(
            lora_ref, tensors, config_dict, added_tokens_config
        )
        logger.info(f"LoRA adapter loading from tensors completes: {lora_ref}.")
        return result

    def unload_lora_adapter(self, lora_ref: LoRARef):
        """Unload a lora adapter that was previously loaded during initialization or dynamic loading."""
        # 卸载指定 LoRA 适配器，释放显存

        logger.info(
            f"LoRA adapter unloading starts: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        result = self.lora_manager.unload_lora_adapter(lora_ref)

        logger.info(
            f"LoRA adapter unloading completes: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        return result

    @property
    def qwen3_next_config(self):
        # 返回 Qwen3Next 配置（若当前模型为 Qwen3Next 架构）
        config = self.model_config.hf_config
        if isinstance(config, Qwen3NextConfig):
            return config
        return None

    @property
    def hybrid_lightning_config(self):
        # 返回 BailingHybrid（混合架构）配置
        config = self.model_config.hf_config
        if isinstance(config, BailingHybridConfig):
            return config
        return None

    @property
    def hybrid_gdn_config(self):
        # 返回支持 GDN（门控扩散网络）的混合架构配置
        config = self.model_config.hf_config.get_text_config()
        if isinstance(
            config,
            Qwen3NextConfig
            | Qwen3_5Config
            | Qwen3_5MoeConfig
            | JetNemotronConfig
            | JetVLMConfig,
        ):
            return config
        return None

    @property
    def mamba2_config(self):
        # 返回包含 Mamba2 层的混合模型配置
        config = self.model_config.hf_config
        if isinstance(config, NemotronHConfig) and self.is_draft_worker:
            # NemotronH MTP draft models have no Mamba layers (pattern like "*E")
            # so they shouldn't use HybridLinearAttnBackend
            # NemotronH MTP 草稿模型可能无 Mamba 层，需检查模式字符串
            pattern = getattr(config, "mtp_hybrid_override_pattern", None)
            if pattern is not None and "M" not in pattern:
                return None
        if isinstance(
            config,
            FalconH1Config
            | NemotronHConfig
            | Lfm2Config
            | Lfm2MoeConfig
            | Lfm2VlConfig,
        ):
            return config
        if isinstance(config, NemotronH_Nano_VL_V2_Config):
            return config.llm_config

        if isinstance(config, GraniteMoeHybridConfig):
            has_mamba = any(
                layer_type == "mamba"
                for layer_type in getattr(config, "layer_types", [])
            )
            if not has_mamba:
                return None
            else:
                return config

        return None

    @property
    def max_token_pool_size(self):
        """Return the max token pool size considering hybrid swa settings."""
        # 混合 SWA 模式下使用全量最大 token 数，否则使用普通最大 token 数
        if self.is_hybrid_swa:
            return self.full_max_total_num_tokens
        else:
            return self.max_total_num_tokens

    @property
    def kimi_linear_config(self):
        # 返回 KimiLinear（线性注意力）架构配置
        config = self.model_config.hf_config
        if isinstance(config, KimiLinearConfig):
            return config
        return None

    def _get_linear_attn_registry_result(self):
        # 缓存线性注意力注册表查询结果（避免重复查询）
        if not hasattr(self, "_linear_attn_registry_cache"):
            self._linear_attn_registry_cache = get_linear_attn_config(
                self.model_config.hf_config
            )
        return self._linear_attn_registry_cache

    @property
    def linear_attn_model_spec(self):
        # 返回线性注意力模型规格（第一个匹配项）
        result = self._get_linear_attn_registry_result()
        return result[0] if result else None

    @property
    def mambaish_config(self):
        # 返回类 Mamba 架构配置（Mamba2/GDN/Kimi/Lightning/线性注意力等均属此类）
        existing = (
            self.mamba2_config
            or self.hybrid_gdn_config
            or self.kimi_linear_config
            or self.hybrid_lightning_config
        )
        if existing:
            return existing
        result = self._get_linear_attn_registry_result()
        return result[1] if result else None

    def configure_kv_cache_dtype(self):
        # 根据配置和硬件能力推断 KV Cache 的实际数据类型
        if self.server_args.kv_cache_dtype == "auto":
            # 自动模式：检查模型量化配置中是否指定了 FP8 KV Cache
            quant_config = getattr(self.model, "quant_config", None)
            kv_cache_quant_algo = getattr(quant_config, "kv_cache_quant_algo", None)
            if (
                isinstance(kv_cache_quant_algo, str)
                and kv_cache_quant_algo.upper() == "FP8"
            ):
                if _is_hip:
                    # AMD GPU 使用 HIP 专用 FP8 类型
                    self.kv_cache_dtype = fp8_dtype
                    self.server_args.kv_cache_dtype = TORCH_DTYPE_TO_KV_CACHE_STR[
                        self.kv_cache_dtype
                    ]
                else:
                    # NVIDIA GPU 使用 float8_e4m3fn
                    self.kv_cache_dtype = torch.float8_e4m3fn
                    self.server_args.kv_cache_dtype = TORCH_DTYPE_TO_KV_CACHE_STR[
                        self.kv_cache_dtype
                    ]
            else:
                # 无量化配置则 KV Cache 与模型精度相同
                self.kv_cache_dtype = self.dtype
        elif self.server_args.kv_cache_dtype == "fp8_e5m2":
            if _is_hip:  # Using natively supported format
                self.kv_cache_dtype = fp8_dtype
            else:
                self.kv_cache_dtype = torch.float8_e5m2
        elif self.server_args.kv_cache_dtype == "fp8_e4m3":
            if _is_hip:  # Using natively supported format
                self.kv_cache_dtype = fp8_dtype
            else:
                self.kv_cache_dtype = torch.float8_e4m3fn
        elif self.server_args.kv_cache_dtype in ("bf16", "bfloat16"):
            self.kv_cache_dtype = torch.bfloat16
        elif self.server_args.kv_cache_dtype == "fp4_e2m1":
            if hasattr(torch, "float4_e2m1fn_x2"):
                self.kv_cache_dtype = torch.float4_e2m1fn_x2
                logger.warning(f"FP4 (E2M1) KV Cache might lead to a accuracy drop!")
            else:
                # 当前 torch 版本不支持 FP4，回退到 auto 模式
                logger.warning(
                    f"--kv-cache-dtype falls back to 'auto' because this torch version does not support torch.float4_e2m1fn_x2"
                )
                self.kv_cache_dtype = self.dtype
        else:
            raise ValueError(
                f"Unsupported kv_cache_dtype: {self.server_args.kv_cache_dtype}."
            )

        log_info_on_rank0(logger, f"Using KV cache dtype: {self.kv_cache_dtype}")

    def init_cublas(self):
        """We need to run a small matmul to init cublas. Otherwise, it will raise some errors later."""
        # 通过一次小矩阵乘法预初始化 cuBLAS，避免首次推理时的延迟
        dtype = torch.float16
        device = "cuda"
        a = torch.ones((16, 16), dtype=dtype, device=device)
        b = torch.ones((16, 16), dtype=dtype, device=device)
        c = a @ b
        return c

    def init_attention_backend(self):
        """Init attention kernel backend."""
        # 初始化注意力计算后端（根据配置选择 pdmux/tbo/普通模式）
        if self.server_args.enable_pdmux:
            # pdmux 模式：为不同 SM 组分配独立的注意力后端
            self.attn_backend = self._get_attention_backend(init_new_workspace=True)
            self.decode_attn_backend_group = []
            for _ in range(self.server_args.sm_group_num):
                self.decode_attn_backend_group.append(self._get_attention_backend())
            self.decode_attn_backend = self.decode_attn_backend_group[0]
        elif self.server_args.enable_two_batch_overlap and not self.is_draft_worker:
            # TBO（Two-Batch Overlap）模式：交叉执行两个批次以隐藏延迟
            self.attn_backend = TboAttnBackend.init_new(self._get_attention_backend)
        else:
            self.attn_backend = self._get_attention_backend()

    def _get_attention_backend(self, init_new_workspace: bool = False):
        """Init attention kernel backend."""
        # 根据配置选择具体的注意力后端（支持草稿模型覆盖、混合后端等）
        draft_attn_backend = self.server_args.speculative_draft_attention_backend
        if self.is_draft_worker and draft_attn_backend:
            # 草稿模型可使用独立的注意力后端
            logger.warning(
                f"Overriding draft attention backend to {draft_attn_backend}."
            )
            return self._get_attention_backend_from_str(
                draft_attn_backend,
                init_new_workspace=init_new_workspace,
            )

        # 获取 prefill 和 decode 阶段各自的后端字符串
        (
            self.prefill_attention_backend_str,
            self.decode_attention_backend_str,
        ) = self.server_args.get_attention_backends()

        if self.decode_attention_backend_str != self.prefill_attention_backend_str:
            # prefill 和 decode 使用不同后端时，创建混合后端
            from sglang.srt.layers.attention.hybrid_attn_backend import (
                HybridAttnBackend,
            )

            attn_backend = HybridAttnBackend(
                self,
                decode_backend=self._get_attention_backend_from_str(
                    self.decode_attention_backend_str,
                    init_new_workspace=init_new_workspace,
                ),
                prefill_backend=self._get_attention_backend_from_str(
                    self.prefill_attention_backend_str,
                    init_new_workspace=init_new_workspace,
                ),
            )
            logger.info(
                f"Using hybrid attention backend for decode and prefill: "
                f"decode_backend={self.decode_attention_backend_str}, "
                f"prefill_backend={self.prefill_attention_backend_str}."
            )
            logger.warning(
                "Warning: Attention backend specified by --attention-backend or default backend might be overridden."
                "The feature of hybrid attention backend is experimental and unstable. Please raise an issue if you encounter any problem."
            )
        else:
            # prefill 和 decode 使用相同后端
            attn_backend = self._get_attention_backend_from_str(
                self.server_args.attention_backend,
                init_new_workspace=init_new_workspace,
            )

        # 同步后端信息到全局配置
        (
            get_global_server_args().prefill_attention_backend,
            get_global_server_args().decode_attention_backend,
        ) = (self.prefill_attention_backend_str, self.decode_attention_backend_str)
        return attn_backend

    def _get_attention_backend_from_str(
        self, backend_str: str, init_new_workspace: bool = False
    ):
        # 根据后端名字符串实例化注意力后端，并用 wrapper 包装
        if backend_str not in ATTENTION_BACKENDS:
            raise ValueError(f"Invalid attention backend: {backend_str}")
        self.init_new_workspace = init_new_workspace
        full_attention_backend = ATTENTION_BACKENDS[backend_str](self)
        return attn_backend_wrapper(self, full_attention_backend)

    def kernel_warmup(self):
        """
        Warmup and tune kernels before cuda graph capture.
        Currently only doing FlashInfer autotune.
        """
        # CUDA Graph 捕获前的 kernel 预热和自动调优
        if self.device != "cuda":
            return

        if self._should_run_flashinfer_autotune():
            self._flashinfer_autotune()

    def _pre_initialize_flashinfer_allreduce_workspace(self):
        """Pre-initialize flashinfer allreduce fusion workspaces.

        Must run before CUDA graph capture to avoid collective operations
        (broadcasts, barriers) inside the graph capture context, which can
        deadlock with custom_all_reduce.register_graph_buffers.
        """
        # 预初始化 FlashInfer AllReduce 融合所需的 workspace，避免在图捕获阶段触发集合通信
        if not self.server_args.enable_flashinfer_allreduce_fusion:
            return

        from sglang.srt.layers.communicator import FUSE_ALLREDUCE_MAX_BATCH_SIZE
        from sglang.srt.layers.flashinfer_comm_fusion import (
            pre_initialize_workspaces,
        )

        pre_initialize_workspaces(
            max_token_num=FUSE_ALLREDUCE_MAX_BATCH_SIZE,
            hidden_dim=self.model_config.hidden_size,
            dtype=self.dtype,
        )

    def _should_run_flashinfer_autotune(self) -> bool:
        """Check if flashinfer autotune should be run."""
        # 判断是否需要运行 FlashInfer MoE 自动调优
        if self.server_args.disable_flashinfer_autotune:
            return False

        # CuteDSL v1 (cutedsl runner + deepep a2a) bypasses MoeRunner and must not
        # be autotuned -- its _dummy_run would dispatch more tokens per rank than
        # SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK, tripping a DeepEP assert.
        # Read server_args directly to avoid depending on initialize_moe_config()
        # having already populated the MoE backend globals.
        # CuteDSL + DeepEP 组合不支持自动调优
        if (
            self.server_args.moe_runner_backend == "flashinfer_cutedsl"
            and self.server_args.moe_a2a_backend == "deepep"
        ):
            return False

        backend_str = self.server_args.moe_runner_backend

        # TODO smor- support other cases for flashinfer autotune, such as, mamba backend
        # 仅部分 FlashInfer 后端支持自动调优
        if backend_str not in [
            "flashinfer_trtllm",
            # TODO: Enable for flashinfer_trtllm_routed once https://github.com/flashinfer-ai/flashinfer/issues/2749 is fixed.
            # "flashinfer_trtllm_routed",
            "flashinfer_mxfp4",
            "flashinfer_cutedsl",
            # TODO: flashinfer_cutlass will cause some flashinfer compilation errors. To be fixed.
            # "flashinfer_cutlass",
        ]:
            return False

        major, _ = torch.cuda.get_device_capability()
        if major < 9:
            # 自动调优仅在 sm90+ (Hopper) 架构上有意义
            return False

        if self.spec_algorithm.is_speculative():
            # 投机解码时仅对目标模型 worker 运行调优
            return not self.is_draft_worker

        return True

    def _flashinfer_autotune(self):
        """Run flashinfer autotune."""
        # 在独立 CUDA stream 上运行 FlashInfer MoE 自动调优（避免干扰默认流）
        from flashinfer.autotuner import autotune

        logger.info("Running FlashInfer autotune...")

        # Run warmup on the non-default stream to avoid NCCL 2.29+ cudaMemcpyBatchAsync
        # calls on default stream (unsupported by CUDA) when --enable-symm-mem is used.
        # 在 forward_stream 上运行，避免对称内存场景下的 NCCL 冲突
        self.forward_stream.wait_stream(torch.cuda.current_stream())
        with torch.get_device_module(self.device).stream(self.forward_stream):
            with torch.inference_mode(), autotune():
                self._dummy_run(
                    batch_size=self.req_to_token_pool.size, run_ctx=autotune()
                )
        torch.cuda.current_stream().wait_stream(self.forward_stream)
        logger.info("FlashInfer autotune completed.")

    def _dummy_run(self, batch_size: int, run_ctx=None):
        """Run a dummy forward pass for warmup/profiling."""
        # 构造虚拟批次并执行一次 forward pass，用于 kernel 预热或显存 profiling
        if self.is_generation:
            capture_forward_mode = ForwardMode.DECODE
        else:
            capture_forward_mode = ForwardMode.EXTEND
        capture_hidden_mode = CaptureHiddenMode.NULL
        num_tokens_per_bs = 1
        if self.spec_algorithm.is_speculative():
            if self.is_draft_worker:
                if not self.spec_algorithm.is_dflash():
                    raise RuntimeError("This should not happen")
            # 投机解码目标模型验证模式：每个批次包含多个草稿 token
            capture_forward_mode = ForwardMode.TARGET_VERIFY
            num_tokens_per_bs = self.server_args.speculative_num_draft_tokens

        if self.server_args.enable_return_hidden_states:
            capture_hidden_mode = CaptureHiddenMode.FULL

        num_tokens = batch_size * num_tokens_per_bs

        # 对齐 num_tokens 到 attn_tp_size 的整数倍（DP Attention 要求）
        if require_gathered_buffer(self.server_args):
            attn_tp_size = get_attention_tp_size()
            if attn_tp_size > 1 and num_tokens % attn_tp_size != 0:
                num_tokens = num_tokens // attn_tp_size * attn_tp_size
                batch_size = num_tokens // num_tokens_per_bs

        seq_len_fill_value = self.attn_backend.get_cuda_graph_seq_len_fill_value()

        if self.server_args.enable_torch_compile:
            set_torch_compile_config()
            should_disable_torch_compile = not getattr(
                self.model, "_can_torch_compile", True
            )
            if should_disable_torch_compile:
                log_info_on_rank0(
                    logger,
                    "Transformers backend model reports it is not torch.compile "
                    "compatible (e.g. dynamic rope scaling). Disabling torch.compile.",
                )
                self.server_args.enable_torch_compile = False

        # NOTE: aux hidden state capture (eagle3/dflash) is already
        # configured by init_aux_hidden_state_capture() in initialize().
        # 辅助隐状态捕获已在 init_aux_hidden_state_capture() 中配置，无需重复设置

        require_mlp_tp_gather_ = require_mlp_tp_gather(self.server_args)
        if require_gathered_buffer(self.server_args):
            assert require_mlp_tp_gather_ or require_attn_tp_gather(self.server_args)

        # 创建 decode 阶段所需的输入缓冲区
        buffers: DecodeInputBuffers = DecodeInputBuffers.create(
            device=self.device,
            max_bs=batch_size,
            max_num_token=num_tokens,
            hidden_size=self.model_config.hidden_size,
            vocab_size=self.model_config.vocab_size,
            dtype=self.model_config.dtype,
            dp_size=self.server_args.dp_size,
            pp_size=self.server_args.pp_size,
            is_encoder_decoder=self.model_config.is_encoder_decoder,
            require_mlp_tp_gather=require_mlp_tp_gather_,
            seq_len_fill_value=seq_len_fill_value,
            encoder_len_fill_value=(
                getattr(self.model_config.hf_config, "max_source_positions", 0)
                if self.model_config.is_encoder_decoder
                else 0
            ),
            num_tokens_per_bs=num_tokens_per_bs,
            cache_loc_dtype=torch.int64,
            enable_mamba_track=False,
        )
        buffers.num_token_non_padded[...] = num_tokens

        # For extend mode
        # 非生成模型（embedding）使用 extend 模式，构造前缀和序列长度信息
        if not self.is_generation:
            extend_prefix_lens_cpu = [0] * batch_size
            extend_seq_lens_cpu = [seq_len_fill_value] * batch_size
            extend_num_tokens = num_tokens
            extend_seq_lens = torch.full(
                (batch_size,), seq_len_fill_value, dtype=torch.int32, device=self.device
            )
            extend_prefix_lens = torch.zeros(
                (batch_size,), dtype=torch.int32, device=self.device
            )
            extend_start_loc = torch.arange(
                0, num_tokens, num_tokens_per_bs, dtype=torch.int32, device=self.device
            )
        else:
            # 生成模型的 decode 阶段不需要 extend 相关信息
            extend_prefix_lens_cpu = None
            extend_seq_lens_cpu = None
            extend_num_tokens = None
            extend_seq_lens = None
            extend_prefix_lens = None
            extend_start_loc = None

        if self.server_args.pp_size > 1:
            # 流水线并行：构造跨 stage 传递的代理 tensor（截取前 num_tokens 个 token）
            pp_proxy_tensors = PPProxyTensors(
                {k: v[:num_tokens] for k, v in buffers.pp_proxy_tensors.items()}
            )

        if require_mlp_tp_gather_:
            # MLP TP gather 模式：每个 DP rank 有 num_tokens 个 token
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.server_args.dp_size,
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.server_args.dp_size,
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            global_dp_buffer_len = num_tokens * self.server_args.dp_size
        elif require_attn_tp_gather(self.server_args):
            # Attention TP gather 模式：整体 token 数为 num_tokens
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            global_dp_buffer_len = num_tokens
        else:
            global_dp_buffer_len = None

        def get_spec_info():
            # 构造投机解码验证所需的辅助信息（EAGLE/DFlash/Ngram 各有不同格式）
            spec_info = None
            if self.spec_algorithm.is_eagle() or self.spec_algorithm.is_standalone():
                from sglang.srt.speculative.eagle_info import EagleVerifyInput

                if self.is_draft_worker:
                    raise RuntimeError("This should not happen.")
                else:
                    # EAGLE 验证输入：虚拟填充（None），用于图捕获时的形状推断
                    spec_info = EagleVerifyInput(
                        draft_token=None,
                        custom_mask=buffers.custom_mask,
                        positions=None,
                        retrieve_index=None,
                        retrieve_next_token=None,
                        retrieve_next_sibling=None,
                        retrieve_cum_len=None,
                        spec_steps=self.server_args.speculative_num_steps,
                        topk=self.server_args.speculative_eagle_topk,
                        draft_token_num=self.server_args.speculative_num_draft_tokens,
                        capture_hidden_mode=CaptureHiddenMode.FULL,
                        seq_lens_sum=None,
                        seq_lens_cpu=None,
                    )
            elif self.spec_algorithm.is_dflash():
                from sglang.srt.speculative.dflash_info import DFlashVerifyInput

                # Dummy warmup only needs shape metadata; avoid forcing custom-mask mode.
                # DFlash 虚拟预热仅需形状信息，不强制 custom_mask 模式
                spec_info = DFlashVerifyInput(
                    draft_token=None,
                    positions=None,
                    draft_token_num=self.server_args.speculative_num_draft_tokens,
                    custom_mask=None,
                    capture_hidden_mode=(
                        CaptureHiddenMode.NULL
                        if self.is_draft_worker
                        else CaptureHiddenMode.FULL
                    ),
                )

            elif self.spec_algorithm.is_ngram():
                from sglang.srt.speculative.ngram_info import NgramVerifyInput

                spec_info = NgramVerifyInput(
                    draft_token=None,
                    tree_mask=buffers.custom_mask,
                    positions=None,
                    retrieve_index=None,
                    retrieve_next_token=None,
                    retrieve_next_sibling=None,
                    draft_token_num=num_tokens_per_bs,
                )
                spec_info.capture_hidden_mode = CaptureHiddenMode.NULL

            return spec_info

        spec_info = get_spec_info()
        if capture_hidden_mode != CaptureHiddenMode.FULL:
            # 如果未强制 FULL 模式，则从 spec_info 中推断（或默认 NULL）
            capture_hidden_mode = (
                spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
            )

        if self.server_args.enable_lora:
            lora_ids = [None] * batch_size  # LoRA 虚拟 ID（预热时所有请求不使用 LoRA）
        else:
            lora_ids = None

        # 构造完整的 ForwardBatch 对象，包含所有前向推理所需信息
        forward_batch = ForwardBatch(
            forward_mode=capture_forward_mode,
            batch_size=batch_size,
            input_ids=buffers.input_ids,
            req_pool_indices=buffers.req_pool_indices,
            seq_lens=buffers.seq_lens,
            seq_lens_cpu=buffers.seq_lens_cpu,
            next_token_logits_buffer=buffers.next_token_logits_buffer,
            orig_seq_lens=buffers.seq_lens,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool=self.token_to_kv_pool,
            attn_backend=self.attn_backend,
            out_cache_loc=buffers.out_cache_loc,
            seq_lens_sum=buffers.seq_lens.sum().item(),
            encoder_lens=buffers.encoder_lens,
            return_logprob=False,
            positions=buffers.positions,
            extend_num_tokens=extend_num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_start_loc=extend_start_loc,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            global_num_tokens_gpu=buffers.global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=buffers.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            mrope_positions=buffers.mrope_positions,
            spec_algorithm=self.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=capture_hidden_mode,
            num_token_non_padded=buffers.num_token_non_padded,
            global_forward_mode=capture_forward_mode,
            lora_ids=lora_ids,
        )

        if lora_ids is not None:
            self.lora_manager.prepare_lora_batch(forward_batch)

        self.attn_backend.init_forward_metadata(forward_batch)

        def run_once():
            # 执行一次 forward pass（图捕获时重复调用此函数）
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(
                global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
            )
            set_is_extend_in_batch(False)

            kwargs = {}
            if (
                self.server_args.pp_size > 1
                and "pp_proxy_tensors"
                in inspect.signature(self.model.forward).parameters
            ):
                # PP 模式：传递上一 stage 的代理 tensor
                kwargs["pp_proxy_tensors"] = PPProxyTensors(
                    {k: v.clone() for k, v in pp_proxy_tensors.tensors.items()}
                )
            if not self.is_generation:
                kwargs["get_embedding"] = True  # embedding 模型需要返回隐状态

            logits_output_or_pp_proxy_tensors = self.model.forward(
                buffers.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )
            return logits_output_or_pp_proxy_tensors

        torch.get_device_module(self.device).synchronize()
        self.tp_group.barrier()  # 确保所有 TP rank 同步后再执行预热
        with torch.inference_mode(), run_ctx or empty_context():
            run_once()

    def maybe_init_ngram_embedding(self):
        # 初始化 ngram embedding 的 token 查表（投机解码 ngram 策略专用）
        self.use_ngram_embedding = self.model_config.use_ngram_embedding
        if self.use_ngram_embedding:
            from sglang.srt.layers.n_gram_embedding import NgramEmbedding

            # 为每个请求预分配全上下文长度的 token 表
            self.token_table = torch.empty(
                self.req_to_token_pool.size,
                self.model_config.context_len,
                dtype=torch.int32,
                device=self.device,
            )
            chunked_prefill_size = self.server_args.chunked_prefill_size
            assert (
                chunked_prefill_size is not None and chunked_prefill_size > 0
            ), "Ngram embedding requires chunked prefill to be enabled (chunked_prefill_size > 0)"
            # 为每个 NgramEmbedding 模块初始化缓冲区
            for module in self.model.modules():
                if isinstance(module, NgramEmbedding):
                    module.init_buffers(
                        self.max_running_requests, chunked_prefill_size, self.device
                    )

    def maybe_update_ngram_token_table(
        self,
        next_token_ids: torch.Tensor,
        forward_batch: "ForwardBatch",
    ):
        """Update the ngram embedding token table after sampling."""
        # 采样后更新 ngram embedding token 表（将新生成的 token 记录到表中）
        ngram_embedding_info = forward_batch.ngram_embedding_info
        if ngram_embedding_info is None:
            return
        # 更新每个请求的输出列起始位置为当前序列长度
        ngram_embedding_info.out_column_starts[: forward_batch.batch_size] = (
            forward_batch.seq_lens
        )
        ngram_embedding_info.out_req_lens[: forward_batch.batch_size] = 1
        update_token_table(
            ne_token_table=ngram_embedding_info.token_table,
            tokens=next_token_ids.to(torch.int32),
            row_indices=forward_batch.req_pool_indices,
            column_starts=ngram_embedding_info.out_column_starts,
            req_lens=torch.ones_like(ngram_embedding_info.out_column_starts),
            ignore_tokens=None,
        )

    def init_device_graphs(self):
        """Capture device graphs."""
        # 捕获设备图（CUDA/NPU/CPU Graph），加速 decode 阶段的重复 forward
        self.graph_runner = None
        self.graph_mem_usage = 0

        if not self.is_generation:
            # TODO: Currently, cuda graph only captures decode steps, which only exists for generation models
            # 非生成模型（embedding）不需要图捕获
            return

        if self.server_args.model_impl.lower() == ModelImpl.MINDSPORE:
            # MindSpore 后端不使用 CUDA Graph
            return

        if self.device != "cpu" and self.server_args.disable_cuda_graph:
            return

        if self.device == "cpu" and not self.server_args.enable_torch_compile:
            # CPU 推理只在启用 torch.compile 时才使用图优化
            return

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        graph_backend = defaultdict(
            lambda: f"{current_platform.device_name} graph",
            {
                "cuda": "cuda graph",
                "musa": "cuda graph",
                "cpu": "cpu graph",
                "npu": "npu graph",
            },
        )
        logger.info(
            f"Capture {graph_backend[self.device]} begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
        )
        if current_platform.is_out_of_tree():
            # 第三方平台使用平台提供的 GraphRunner 类
            GraphRunnerCls = current_platform.get_graph_runner_cls()
            self.graph_runner = GraphRunnerCls(self)
        else:
            # 根据设备类型选择对应的图运行器
            graph_runners = defaultdict(
                lambda: CudaGraphRunner,
                {
                    "cpu": CPUGraphRunner,
                    "npu": NPUGraphRunner,
                },
            )
            self.graph_runner = graph_runners[self.device](self)

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        self.graph_mem_usage = before_mem - after_mem  # 图捕获占用的显存
        logger.info(
            f"Capture {graph_backend[self.device]} end. Time elapsed: {time.perf_counter() - tic:.2f} s. "
            f"mem usage={self.graph_mem_usage:.2f} GB. avail mem={after_mem:.2f} GB."
        )

    def init_piecewise_cuda_graphs(self):
        """Initialize piecewise CUDA graph runner."""
        # 初始化分段 CUDA Graph：将模型的各层分块捕获，支持更灵活的批次大小
        self.piecewise_cuda_graph_runner = None

        if self.server_args.disable_piecewise_cuda_graph:
            logger.info(
                "Disable piecewise CUDA graph because --disable-piecewise-cuda-graph is set"
            )
            return

        # Draft models use decode CUDA graphs, not PCG
        # 草稿模型使用 decode 图，不使用分段图
        if self.is_draft_worker:
            return

        # Disable piecewise CUDA graph for non-language models
        # 非语言模型（无 model 属性）不支持分段图
        if not hasattr(self.model, "model"):
            logger.warning(
                "Disable piecewise CUDA graph because the model is not a language model"
            )
            return

        # Disable piecewise CUDA graph for non capture size
        if not self.server_args.piecewise_cuda_graph_tokens:
            logger.warning(
                "Disable piecewise CUDA graph because the capture size is not set"
            )
            return

        # Collect attention layers and moe layers from the model
        # 从模型中收集注意力层和 MoE 层，用于逐层图捕获
        self.model.model = resolve_language_model(self.model)
        language_model = getattr(self.model, "language_model", self.model)

        # Some draft models (e.g. eagle3) don't have a standard 'layers' attribute
        # 部分模型没有标准的 layers 属性，跳过
        if not hasattr(language_model.model, "layers"):
            logger.warning(
                "Disable piecewise CUDA graph because the model does not have a 'layers' attribute"
            )
            return

        self.attention_layers = []
        self.moe_layers = []
        self.moe_fusions = []
        # 遍历所有 transformer 层，提取注意力层和 MoE 层对象
        for layer in language_model.model.layers:
            attn_layer = None
            if hasattr(layer, "self_attn"):
                if hasattr(layer.self_attn, "attn"):
                    attn_layer = layer.self_attn.attn
                elif hasattr(layer.self_attn, "attn_mqa"):
                    # For DeepSeek model
                    attn_layer = layer.self_attn.attn_mqa
            # For hybrid model
            elif hasattr(layer, "attn"):
                attn_layer = layer.attn
            elif hasattr(layer, "linear_attn"):
                if hasattr(layer.linear_attn, "attn"):
                    attn_layer = layer.linear_attn.attn
                else:
                    attn_layer = layer.linear_attn
            # For InternVL model
            elif hasattr(layer, "attention"):
                if hasattr(layer.attention, "attn"):
                    attn_layer = layer.attention.attn
            # For NemotronH and similar hybrid models using 'mixer' attribute
            elif hasattr(layer, "mixer"):
                if hasattr(layer.mixer, "attn"):
                    attn_layer = layer.mixer.attn
                elif hasattr(layer, "_forward_mamba"):
                    # Mamba layer with split op support - store the layer itself
                    attn_layer = layer

            if attn_layer is not None:
                self.attention_layers.append(attn_layer)
            elif hasattr(layer, "mixer"):
                self.attention_layers.append(None)  # Mamba 层占位

            moe_block = None
            moe_fusion = None
            # 多种模型结构中 MoE 层的访问方式各不相同
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                moe_block = layer.mlp.experts
                moe_fusion = layer.mlp
            if hasattr(layer, "block_sparse_moe") and hasattr(
                layer.block_sparse_moe, "experts"
            ):
                moe_block = layer.block_sparse_moe.experts
                moe_fusion = layer.block_sparse_moe
            if hasattr(layer, "moe") and hasattr(layer.moe, "experts"):
                moe_block = layer.moe.experts
                moe_fusion = layer.moe
            # For NemotronH MoE layers using 'mixer' attribute
            if hasattr(layer, "mixer") and hasattr(layer.mixer, "experts"):
                moe_block = layer.mixer.experts
                moe_fusion = layer.mixer
            self.moe_layers.append(moe_block)
            self.moe_fusions.append(moe_fusion)

        if len(self.attention_layers) < self.model_config.num_hidden_layers:
            # TODO(yuwei): support Non-Standard GQA
            # 注意力层数量不足，说明存在非标准 GQA，跳过分段图
            log_info_on_rank0(
                logger,
                "Disable piecewise CUDA graph because some layers do not apply Standard GQA",
            )
            return

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture piecewise CUDA graph begin. avail mem={before_mem:.2f} GB"
        )

        if self.server_args.enable_breakable_cuda_graph:
            # Experimental feature
            # 实验性功能：可中断的 CUDA Graph，支持动态打断图执行
            self.piecewise_cuda_graph_runner = BreakableCudaGraphRunner(self)
        else:
            # 标准分段 CUDA Graph Runner
            self.piecewise_cuda_graph_runner = PiecewiseCudaGraphRunner(self)

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        mem_usage = before_mem - after_mem
        logger.info(
            f"Capture piecewise CUDA graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. "
            f"mem usage={mem_usage:.2f} GB. avail mem={after_mem:.2f} GB."
        )

    def init_threads_binding(self):
        # 初始化 CPU 推理的 OpenMP 线程绑定（NUMA 亲和性配置）
        omp_cpuids = os.environ.get("SGLANG_CPU_OMP_THREADS_BIND", "all")
        cpu_ids_by_node = get_cpu_ids_by_node()
        n_numa_node = len(cpu_ids_by_node)
        if omp_cpuids == "all":
            # 默认模式：每个 TP rank 绑定到一个 NUMA 节点的所有核
            assert self.tp_size <= n_numa_node, (
                f"SGLANG_CPU_OMP_THREADS_BIND is not set, in this case, "
                f"tp_size {self.tp_size} should be smaller than or equal to number of numa node on the machine {n_numa_node}. "
                f"If you need tp_size to be larger than number of numa node, please set the CPU cores for each tp rank via SGLANG_CPU_OMP_THREADS_BIND explicitly. "
                f"For example, on a machine with 2 numa nodes, where core 0-31 are on numa node 0 and core 32-63 are on numa node 1, "
                f"it is suggested to use -tp 2 and bind tp rank 0 to core 0-31 and tp rank 1 to core 32-63. "
                f"This is the default behavior if SGLANG_CPU_OMP_THREADS_BIND is not set and it is the same as setting SGLANG_CPU_OMP_THREADS_BIND=0-31|32-63. "
                f"If you do need tp_size to be larger than the number of numa nodes, you could set SGLANG_CPU_OMP_THREADS_BIND explicitly for example SGLANG_CPU_OMP_THREADS_BIND=0-15|16-31|32-47|48-63 and run with -tp 4. "
                f"If you don't want each tp rank to use all the cores on one numa node, you could set for example SGLANG_CPU_OMP_THREADS_BIND=0-15|32-47 and run with -tp 2."
            )
            if self.tp_size < n_numa_node:
                logger.warning(
                    f"Detected the current machine has {n_numa_node} numa nodes available, but tp_size is set to {self.tp_size}, so only {self.tp_size} numa nodes are used."
                )
            self.local_omp_cpuid = cpu_ids_by_node[self.tp_rank]
        else:
            # 手动模式：从环境变量中读取每个 TP rank 对应的 CPU ID 范围（用 | 分隔）
            threads_bind_list = omp_cpuids.split("|")
            assert self.tp_size == len(threads_bind_list), (
                f"SGLANG_CPU_OMP_THREADS_BIND setting must be aligned with TP size parameter ({self.tp_size}). "
                f"Please double check your settings."
            )
            self.local_omp_cpuid = threads_bind_list[self.tp_rank]
            if self.tp_size > n_numa_node:
                # TP 大于 NUMA 节点数时无法预测每个 rank 的可用内存，建议手动设置 max-total-tokens
                logger.warning(
                    f"TP size ({self.tp_size})is larger than numa node number ({n_numa_node}), "
                    f"in this case the available memory amount of each rank cannot be determined in prior. "
                    f"Please set proper `--max-total-tokens` to avoid the out-of-memory error."
                )

    def apply_torch_tp(self):
        # 应用 PyTorch 原生张量并行（通过 device_mesh 实现）
        logger.info(f"Enabling torch tensor parallelism on {self.tp_size} devices.")
        from sglang.srt.layers.model_parallel import tensor_parallel

        device_mesh = torch.distributed.init_device_mesh(self.device, (self.tp_size,))
        tensor_parallel(self.model, device_mesh)

    def update_decode_attn_backend(self, stream_idx: int):
        # 切换当前使用的 decode 注意力后端（pdmux 多 SM 流模式下使用）
        self.decode_attn_backend = self.decode_attn_backend_group[stream_idx]

    def forward_decode(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors=None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        # 执行 decode 阶段前向传播（逐 token 生成）
        if not skip_attn_backend_init:
            if hasattr(self.model, "prepare_forward_batch"):
                # Prepare model-specific attention metadata before planning,
                # e.g. Moss-VL's prefill cross-attention custom mask.
                # 特殊模型（如 Moss-VL）需要在规划前准备 attention metadata
                self.model.prepare_forward_batch(forward_batch)
            if self.server_args.enable_pdmux:
                # pdmux 模式使用独立的 decode 注意力后端
                self.decode_attn_backend.init_forward_metadata(forward_batch)
                forward_batch.attn_backend = self.decode_attn_backend
            else:
                self.attn_backend.init_forward_metadata(forward_batch)
        # FIXME: add pp_proxy_tensors arg to all models
        kwargs = {}
        if self.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors
        return self.model.forward(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
            **kwargs,
        )

    def forward_extend(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors=None,
    ) -> Tuple[
        Union[LogitsProcessorOutput, PPProxyTensors, EmbeddingPoolerOutput], bool
    ]:
        # 执行 prefill/extend 阶段前向传播（处理新 token 并填充 KV Cache）
        kwargs = {}
        if self.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors
        if forward_batch.input_embeds is not None:
            # 使用预计算的 embedding（如多模态输入）
            kwargs["input_embeds"] = forward_batch.input_embeds.bfloat16()
        if (
            forward_batch.replace_embeds is not None
            and forward_batch.replace_positions is not None
        ):
            # Token embedding overrides: get base embeddings, scatter replacements
            # Token embedding 替换：先获取基础 embedding，再scatter替换指定位置
            if "input_embeds" not in kwargs:
                embed_layer = self.model.get_input_embeddings()
                kwargs["input_embeds"] = embed_layer(forward_batch.input_ids)
            kwargs["input_embeds"][forward_batch.replace_positions] = (
                forward_batch.replace_embeds.to(kwargs["input_embeds"].dtype)
            )
        if not self.is_generation:
            kwargs["get_embedding"] = True  # embedding 模型返回隐状态而非 logits

        # 检查是否可以使用分段 CUDA Graph（更大批次的优化路径）
        can_run_graph = (
            self.piecewise_cuda_graph_runner is not None
            and self.piecewise_cuda_graph_runner.can_run(forward_batch)
        )

        if can_run_graph:
            # 使用分段图 replay（无需重新初始化 attention metadata）
            return (
                self.piecewise_cuda_graph_runner.replay(forward_batch, **kwargs),
                can_run_graph,
            )

        if not skip_attn_backend_init:
            if hasattr(self.model, "prepare_forward_batch"):
                # Prepare model-specific attention metadata before planning,
                # e.g. Moss-VL's prefill cross-attention custom mask.
                self.model.prepare_forward_batch(forward_batch)
            self.attn_backend.init_forward_metadata(forward_batch)

        return (
            self.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            ),
            can_run_graph,
        )

    def forward_idle(
        self, forward_batch: ForwardBatch, pp_proxy_tensors=None
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        # In DP Attention, IDLE batches are padded (batch_size > 0) for MLP sync.
        # in this case, we need to reinit the forward metadata, otherwise the stale
        # metadata causes batch_size mismatch in attention kernel(e.g. NSA Indexer).
        # 空闲批次（DP Attention padding 用）：重新初始化 forward metadata 避免维度不匹配
        if forward_batch.batch_size > 0:
            self.attn_backend.init_forward_metadata(forward_batch)

        kwargs = {}
        if self.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors
        return self.model.forward(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
            **kwargs,
        )

    def forward_split_prefill(
        self,
        forward_batch: ForwardBatch,
        reinit_attn_backend: bool = False,
        forward_count: int = 1,
    ) -> LogitsProcessorOutput:
        # 分段 prefill：每次处理 forward_count 层（用于超长序列 prefill 的内存管理）
        if forward_batch.split_index == 0 or reinit_attn_backend:
            # 首段或需要重新初始化时才调用 init_forward_metadata
            self.attn_backend.init_forward_metadata(forward_batch)
        next_split_index = min(
            forward_batch.split_index + forward_count,
            self.model_config.num_hidden_layers,
        )
        ret = self.model.forward_split_prefill(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
            (forward_batch.split_index, next_split_index),  # 当前分段的层范围
        )
        forward_batch.split_index = next_split_index  # 更新分段游标
        return ret

    def forward(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        reinit_attn_backend: bool = False,
        split_forward_count: int = 1,
    ) -> ModelRunnerOutput:
        # ModelRunner 主前向入口：调度到具体的 forward 方法，并处理 EPLB/调试逻辑
        self.forward_pass_id += 1  # 递增前向传播计数器

        if self.msprobe_debugger is not None:
            # 启动 msprobe 精度调试器
            rank_id = (
                self.gpu_id if self.dp_size is not None and self.dp_size > 1 else None
            )
            self.msprobe_debugger.start(model=self.model, rank_id=rank_id)

        # 构造性能分析 span（仅在 profiler 启用时）
        step_span_ctx = (
            torch.profiler.record_function(_build_step_span_name(forward_batch))
            if torch.autograd._profiler_enabled()
            else contextlib.nullcontext()
        )
        with (
            step_span_ctx,
            # 记录本次 forward 的专家分布（用于 EPLB 负载均衡决策）
            get_global_expert_distribution_recorder().with_forward_pass(
                self.forward_pass_id,
                forward_batch,
            ) as recorder_outputs,
        ):
            output = self._forward_raw(
                forward_batch,
                skip_attn_backend_init,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )
            elastic_ep_state = ElasticEPStateManager.instance()
            if (
                elastic_ep_state is not None
                and not elastic_ep_state.is_active_equal_last()
            ):
                # 检测到弹性 EP rank 活跃状态变化，触发 EPLB 再均衡
                elastic_ep_state.snapshot_active_to_last()
                elastic_ep_state.sync_active_to_cpu()
                logging.info("EPLB due to rank faults")
                gen = self.eplb_manager.rebalance()
                while True:
                    try:
                        next(gen)
                    except StopIteration:
                        break
                # 再均衡后重新执行 forward
                output = self._forward_raw(
                    forward_batch,
                    skip_attn_backend_init,
                    pp_proxy_tensors,
                    reinit_attn_backend,
                    split_forward_count,
                )
        output.expert_distribution_metrics = recorder_outputs.get("metrics")

        # 获取路由专家输出（可选地不等待 CPU 拷贝完成，用于 overlap 调度）
        no_copy_to_cpu = not self.server_args.disable_overlap_schedule
        output.routed_experts_output = get_global_experts_capturer().on_forward_end(
            forward_batch=forward_batch,
            can_run_graph=output.can_run_graph,
            cuda_graph_batch=getattr(self.graph_runner, "bs", None),
            no_copy_to_cpu=no_copy_to_cpu,
        )

        if self.eplb_manager is not None:
            self.eplb_manager.on_forward_pass_end()  # 通知 EPLB 管理器本次 forward 结束

        if dumper.may_enable:
            dumper.step()  # 触发 tensor dump 步进

        if self.msprobe_debugger is not None:
            self.msprobe_debugger.stop()
            self.msprobe_debugger.step()  # 停止并推进 msprobe 采集步

        if self.server_args.elastic_ep_backend is not None:
            self.maybe_recover_ep_ranks()  # 尝试恢复失活的 EP rank

        return output

    def _forward_raw(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool,
        pp_proxy_tensors: Optional[PPProxyTensors],
        reinit_attn_backend: bool = False,
        split_forward_count: int = 1,
    ) -> ModelRunnerOutput:
        # 底层 forward 分发：判断是否走 CUDA Graph 路径，否则根据 forward_mode 路由
        mode_check = (
            forward_batch.forward_mode.is_cpu_graph
            if self.device == "cpu"
            else forward_batch.forward_mode.is_cuda_graph
        )
        # 判断当前批次是否可以使用图加速
        can_run_graph = bool(
            mode_check()
            and self.graph_runner
            and self.graph_runner.can_run(forward_batch)
        )

        if (
            self.hisparse_coordinator is not None
            and forward_batch.forward_mode.is_decode()
        ):
            # HiSparse 模式：等待上一批次的稀疏注意力备份计算完成
            self.hisparse_coordinator.wait_for_pending_backup()

        if can_run_graph:
            # 使用 CUDA Graph replay（避免 Python 层调度开销）
            ret = self.graph_runner.replay(
                forward_batch,
                skip_attn_backend_init=skip_attn_backend_init,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)

        # For MLP sync
        # 准备 MLP TP gather 或 Attention TP scatter 所需的同步数据
        if forward_batch.global_num_tokens_cpu is not None:
            forward_batch.prepare_mlp_sync_batch(self)
        else:
            forward_batch.prepare_attn_tp_scatter_input(self)

        # Normalize num_token_non_padded to be local to this attention TP rank if needed.
        # 将 num_token_non_padded 调整为当前 attention TP rank 的本地值
        if (
            forward_batch.num_token_non_padded is not None
            and forward_batch.global_num_tokens_gpu is not None
            and require_gathered_buffer(self.server_args)
            and not is_nsa_enable_prefill_cp()
        ):
            forward_batch.adjust_num_token_non_padded_for_attn_tp(
                server_args=self.server_args,
            )

        # Use precomputed SWA cache location
        # 若有预计算的 SWA KV Cache 位置，则设置到 token pool 中
        if forward_batch.out_cache_loc_swa is not None:
            self.token_to_kv_pool.set_swa_loc(forward_batch.out_cache_loc_swa)

        forward_batch.hisparse_coordinator = self.hisparse_coordinator
        if self.hisparse_coordinator is not None:
            self.hisparse_coordinator.num_real_reqs.fill_(forward_batch.batch_size)

        # 根据 forward_mode 分发到对应的 forward 方法
        if forward_batch.forward_mode.is_decode():
            ret = self.forward_decode(
                forward_batch,
                skip_attn_backend_init=skip_attn_backend_init,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        elif forward_batch.forward_mode.is_split_prefill():
            ret = self.forward_split_prefill(
                forward_batch,
                reinit_attn_backend=reinit_attn_backend,
                forward_count=split_forward_count,
            )
        elif forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
            ret, can_run_graph = self.forward_extend(
                forward_batch,
                skip_attn_backend_init=skip_attn_backend_init,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        elif forward_batch.forward_mode.is_idle():
            ret = self.forward_idle(forward_batch, pp_proxy_tensors=pp_proxy_tensors)
        else:
            raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode}")

        if (
            forward_batch.global_num_tokens_cpu is not None
            and self.pp_group.is_last_rank
        ):
            # PP 最后阶段：对 MLP sync 批次做后处理（汇总各 DP rank 的结果）
            forward_batch.post_forward_mlp_sync_batch(ret)

        return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)

    def _preprocess_logits(
        self, logits_output: LogitsProcessorOutput, sampling_info: SamplingBatchInfo
    ):
        # 对 logits 做采样前预处理：更新正则词表掩码并应用 logits bias
        # NOTE: In overlap mode, the function update_regex_vocab_mask (in sample)
        #       was executed after we processed last batch's results.
        # 注：overlap 模式下，update_regex_vocab_mask 已在上一批次结果处理后调用

        # Calculate logits bias and apply it to next_token_logits.
        # 更新结构化输出（grammar/regex）所需的词表掩码
        sampling_info.update_regex_vocab_mask()
        # 将用户指定的 logits_bias 叠加到下一 token 的 logit 向量上
        sampling_info.apply_logits_bias(logits_output.next_token_logits)

        # Release the vocab_mask GPU tensor immediately after it has been applied
        # to the logits. In overlap scheduling, the sampling_info (and its
        # vocab_mask) can be kept alive by the delay_sample_func closure and
        # batch_record_buf until the next iteration, causing a steady VRAM leak
        # when structured output (grammar) is used.
        # 立即释放 vocab_mask GPU 张量，防止 overlap 调度下因闭包引用导致的 VRAM 泄漏
        sampling_info.vocab_mask = None

    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Sample and compute logprobs and update logits_output.
        对 logits_output 执行采样、计算 logprobs，并更新 logits_output。

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output

        Returns:
            A list of next_token_ids
        """
        # For duplex models with multiple output streams.
        # 双工模型（如多输出流模型）：递归处理每个输出流，然后沿最后一维 stack
        if isinstance(logits_output, tuple):
            return torch.stack(
                [self.sample(values, forward_batch) for values in logits_output],
                axis=-1,
            )

        # 先对 logits 做预处理（词表掩码 + logits bias）
        self._preprocess_logits(logits_output, forward_batch.sampling_info)
        # Sample the next tokens
        # 调用 sampler 执行 top-p/top-k/temperature 等采样策略
        next_token_ids = self.sampler(
            logits_output,
            forward_batch.sampling_info,
            forward_batch.return_logprob,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
            # For prefill, we only use the position of the last token.
            # decode 阶段使用当前位置；prefill 阶段使用各序列最后一个 token 的位置
            (
                forward_batch.positions
                if forward_batch.forward_mode.is_decode()
                else forward_batch.seq_lens - 1
            ),
        )
        # 若启用 ngram 投机解码，将新采样的 token 更新到 ngram token 表中
        self.maybe_update_ngram_token_table(next_token_ids, forward_batch)
        return next_token_ids

    def compute_logprobs_only(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> None:
        """
        Compute token_ids_logprobs without performing sampling.
        仅计算 token_ids_logprobs，不执行采样操作。

        Optimized path for prefill-only requests that need token_ids_logprobs but don't
        require next token generation. Skips expensive sampling operations
        while still providing requested probability information.
        适用于仅需 logprobs 而无需生成下一 token 的 prefill-only 请求，跳过采样以节省开销。

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output
        """
        # 若批次中没有需要 logprobs 的请求，直接返回
        if not forward_batch.token_ids_logprobs:
            return

        # Preprocess logits (same as in sample method)
        # 与 sample 方法相同的 logits 预处理步骤
        self._preprocess_logits(logits_output, forward_batch.sampling_info)

        # Delegate to sampler for logprob-only computation
        # This populates logits_output with requested token probabilities
        # 委托 sampler 仅计算 logprobs，填充 logits_output 中的概率信息
        self.sampler.compute_logprobs_only(
            logits_output,
            forward_batch.sampling_info,
            forward_batch.return_logprob,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
        )

    @property
    def model_is_mrope(self) -> bool:
        """Detect if the model has "mrope" rope_scaling type.
        mrope requires keep "rope_deltas" between prompt and decoding phases.
        检测模型是否使用 mrope（多模态 RoPE）位置编码。
        mrope 需要在 prefill 和 decode 阶段之间保持 rope_deltas。"""
        # 优先读取 rope_parameters，否则回退到 rope_scaling 字典
        rope_scaling = getattr(
            self.model_config.hf_text_config, "rope_parameters", None
        ) or getattr(self.model_config.hf_text_config, "rope_scaling", {})
        if rope_scaling is None:
            return False
        # 只要 rope_scaling 中包含 mrope_section 键，即视为 mrope 模型
        is_mrope_enabled = "mrope_section" in rope_scaling
        return is_mrope_enabled

    def save_remote_model(self, url: str):
        # 将模型权重保存到远程存储（如 S3/HDFS）
        from sglang.srt.model_loader.loader import RemoteModelLoader

        logger.info(f"Saving model to {url}")
        RemoteModelLoader.save_model(self.model, self.model_config.model_path, url)

    def save_sharded_model(
        self, path: str, pattern: Optional[str] = None, max_size: Optional[int] = None
    ):
        # 将模型权重按分片方式保存到本地目录，支持文件名 pattern 和单文件大小上限
        from sglang.srt.model_loader.loader import ShardedStateLoader

        logger.info(
            f"Save sharded model to {path} with pattern {pattern} and max_size {max_size}"
        )
        ShardedStateLoader.save_model(self.model, path, pattern, max_size)

    def check_weights(self, action: str):
        # 触发权重检查器执行指定动作（如校验、打印摘要等）
        self._weight_checker.handle(action=action)

    def update_weights_from_ipc(self, recv_req):
        """Update weights from IPC for checkpoint-engine integration.
        通过 IPC（进程间通信）从 checkpoint-engine 更新模型权重。"""
        try:
            from sglang.srt.checkpoint_engine.checkpoint_engine_worker import (
                SGLangCheckpointEngineWorkerExtensionImpl,
            )

            # Create a worker extension that integrates with SGLang's model
            # 创建与 SGLang 模型集成的 checkpoint worker 扩展
            worker = SGLangCheckpointEngineWorkerExtensionImpl(self)
            worker.update_weights_from_ipc(recv_req.zmq_handles)
            return True, "IPC weight update completed successfully"
        except ImportError as e:
            return False, f"IPC weight update failed: ImportError {e}"
        except Exception as e:
            logger.error(f"IPC weight update failed: {e}")
            return False, str(e)

    def prealloc_symmetric_memory_pool(self):
        # 预分配对称内存池（symmetric memory），限制内存碎片化。
        # PyTorch 内存池在 OOM 时不会自动碎片整理，提前分配大块连续内存可减少碎片。
        # PyTorch mempools never de-fragment memory in OOM scenarios, so we need to pre-allocate a large chunk of memory to limit fragmentation.
        if (
            self.is_draft_worker          # draft worker 不参与预分配
            or not self.server_args.enable_symm_mem  # 未启用对称内存则跳过
            or envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.get() <= 0  # 配置大小为 0 则跳过
        ):
            return

        # Memory allocation is tied to a cuda stream, use the forward stream
        # 内存分配需绑定到 CUDA stream，使用 forward_stream 保证与推理流同步
        with torch.get_device_module(self.device).stream(self.forward_stream):
            logger.info(
                f"Pre-allocating symmetric memory pool with {envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.get()} GiB"
            )
            # 在 TP group 的对称内存上下文中分配大块 uint8 张量
            with use_symmetric_memory(get_tp_group()):
                torch.empty(
                    (envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.get() * 1024 * 1024 * 1024,),
                    dtype=torch.uint8,
                    device=self.device,
                )


def _model_load_weights_direct(model, named_tensors: List[Tuple[str, torch.Tensor]]):
    """直接将给定张量列表加载到模型参数中，绕过文件 IO，用于分布式权重更新场景。"""
    # 构建参数名 -> 参数张量的映射，便于按名称查找
    params_dict = dict(model.named_parameters())
    for name, tensor in named_tensors:
        # 使用 default_weight_loader 将源张量复制到对应参数（支持量化/分片权重）
        default_weight_loader(params_dict[name], tensor)


def _unwrap_tensor(tensor, tp_rank, device):
    """解包可能序列化的张量：若为 LocalSerializedTensor，则按 tp_rank 反序列化，然后移动到目标设备。"""
    if isinstance(tensor, LocalSerializedTensor):
        # 多进程序列化张量：按当前 TP rank 取出对应分片
        tensor = tensor.get(tp_rank)
    return tensor.to(device)


def _build_step_span_name(forward_batch: ForwardBatch) -> str:
    """Build a profile-trace span name for one forward step.
    为单次 forward 步骤构建性能追踪 span 名称，供 Chrome trace 使用。

    Format:
      step[decode bs=N]                   — decode-only batch
      step[prefill bs=N toks=T]           — extend-only (prefill) batch
      step[mixed bs=N ext=T dec=D]        — extend+decode mixed batch
      step[idle]                          — idle/padding step
      step[<MODE> bs=N]                   — other modes (target-verify, etc.)

    Used by ModelRunner.forward to wrap each step in a torch.profile
    record_function so Chrome traces show labeled step boundaries.
    供 ModelRunner.forward 用 torch.profiler.record_function 包裹每个步骤，
    使 Chrome 追踪中显示带标签的步骤边界。
    """
    mode = forward_batch.forward_mode
    bs = forward_batch.batch_size
    # 空闲步骤（无实际请求的填充步）
    if mode.is_idle():
        return "step[idle]"
    # 纯 decode 批次：仅包含续写 token 的请求
    if mode.is_decode():
        return f"step[decode bs={bs}]"
    # extend（prefill）批次：可能包含纯 prefill 或 prefill+decode 混合
    if mode.is_extend():
        ext_toks = forward_batch.extend_num_tokens or 0
        # 统计 extend 序列数（若 extend_seq_lens 存在则取其 shape[0]，否则用 batch_size）
        ext_seqs = (
            forward_batch.extend_seq_lens.shape[0]
            if forward_batch.extend_seq_lens is not None
            else bs
        )
        dec_seqs = bs - ext_seqs
        # 若有 decode 序列，则为混合批次
        if dec_seqs > 0:
            return f"step[mixed bs={bs} ext={ext_toks} dec={dec_seqs}]"
        return f"step[prefill bs={bs} toks={ext_toks}]"
    # 其他模式（target-verify 等）统一格式
    return f"step[{mode.name} bs={bs}]"


@dataclass
class LocalSerializedTensor:
    """torch.Tensor that gets serialized by MultiprocessingSerializer (which only serializes a pointer and not the data).
    The i-th element in the list corresponds to i-th rank's GPU.
    通过 MultiprocessingSerializer 序列化的张量（只序列化指针而非数据本身）。
    列表中第 i 个元素对应第 i 个 TP rank 的 GPU 上的张量分片。
    """

    values: List[bytes]  # 每个 rank 对应的序列化字节串列表

    def get(self, rank: int):
        # 按 rank 反序列化并返回对应的张量
        return MultiprocessingSerializer.deserialize(self.values[rank])
