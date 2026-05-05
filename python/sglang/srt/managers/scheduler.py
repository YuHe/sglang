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
"""A scheduler that manages a tensor parallel GPU worker."""

# -------- 标准库导入 --------
import faulthandler  # faulthandler — 崩溃时打印 C 层调用栈
import logging  # logging — 标准日志模块
import os  # os — 操作系统接口（读取环境变量等）
import signal  # signal — 进程信号处理
import sys  # sys — Python 解释器接口（sys.maxsize 等）
import time  # time — 时间工具（sleep、perf_counter）
from collections import deque  # deque — 双端队列，用作 result_queue 等
from contextlib import nullcontext  # nullcontext — 空上下文管理器（MPS 平台占位）
from dataclasses import dataclass  # dataclass — 数据类装饰器
from http import HTTPStatus  # HTTPStatus — HTTP 状态码枚举
from typing import Any, Deque, Dict, List, Optional, Tuple, Union  # typing — 类型注解

from sglang.srt.utils.common import suppress_noisy_warnings  # suppress_noisy_warnings — 屏蔽第三方库的噪声警告

suppress_noisy_warnings()  # 在引入其他第三方库之前先抑制噪声警告

# -------- 第三方库导入 --------
import psutil  # psutil — 进程/系统资源信息（获取父进程等）
import setproctitle  # setproctitle — 设置进程标题（ps 命令可见）
import torch  # torch — PyTorch 核心
import torch.distributed  # torch.distributed — 分布式通信原语
import zmq  # zmq — ZeroMQ 消息队列（Scheduler 与其他管理器的 IPC）
from torch.cuda import Stream as CudaStream  # CudaStream — CUDA 流类型
from torch.distributed import barrier  # barrier — 分布式同步屏障

# -------- SGLang 内部模块导入 --------
from sglang.jit_kernel.ngram_embedding import update_token_table  # update_token_table — Ngram Embedding token 表更新 JIT kernel
from sglang.srt.configs.model_config import ModelConfig, ModelImpl  # ModelConfig — 模型配置；ModelImpl — 模型实现枚举
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX  # HEALTH_CHECK_RID_PREFIX — 健康检查请求 ID 前缀
from sglang.srt.constrained.grammar_manager import GrammarManager  # GrammarManager — 约束生成（JSON/正则）语法管理器
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,  # DecodePreallocQueue — 解码端预分配 KV Cache 的队列
    DecodeTransferQueue,  # DecodeTransferQueue — 解码端接收 KV Cache 传输的队列
    SchedulerDisaggregationDecodeMixin,  # SchedulerDisaggregationDecodeMixin — 解耦模式解码侧 Mixin
)
from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,  # DecodeKVCacheOffloadManager — 解码端 KV Cache 卸载到 CPU 的管理器
)
from sglang.srt.disaggregation.encode_receiver import create_mm_receiver  # create_mm_receiver — EPD 模式多模态编码结果接收器工厂
from sglang.srt.disaggregation.prefill import (
    PrefillBootstrapQueue,  # PrefillBootstrapQueue — 预填充端 Bootstrap 握手队列
    SchedulerDisaggregationPrefillMixin,  # SchedulerDisaggregationPrefillMixin — 解耦模式预填充侧 Mixin
    release_req_to_metadata_buffer,  # release_req_to_metadata_buffer — 释放元数据缓冲区
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,  # DisaggregationMode — 解耦模式枚举（NULL/PREFILL/DECODE）
    MetadataBuffers,  # MetadataBuffers — 预填充/解码间传递元数据的缓冲区
    ReqToMetadataIdxAllocator,  # ReqToMetadataIdxAllocator — 请求到元数据缓冲区索引的分配器
    TransferBackend,  # TransferBackend — KV Cache 传输后端枚举
    prepare_abort,  # prepare_abort — 将请求标记为 ABORT 状态
)
from sglang.srt.distributed import get_pp_group, get_world_group  # get_pp_group — 流水线并行进程组；get_world_group — 全局进程组
from sglang.srt.distributed.parallel_state import get_tp_group  # get_tp_group — 张量并行进程组
from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin  # SchedulerDllmMixin — 扩散 LLM 调度器 Mixin
from sglang.srt.environ import envs  # envs — SGLang 环境变量集中管理对象
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder  # get_global_expert_distribution_recorder — MoE 专家分布统计全局记录器
from sglang.srt.layers.attention.mamba.ops import (
    initialize_mamba_selective_state_update_backend,  # initialize_mamba_selective_state_update_backend — 初始化 Mamba SSM 选择性状态更新后端
)
from sglang.srt.layers.dp_attention import (
    compute_dp_attention_world_info,  # compute_dp_attention_world_info — 计算 DP Attention 分布信息
    get_attention_cp_group,  # get_attention_cp_group — 获取 Attention Context Parallel 进程组
    get_attention_tp_group,  # get_attention_tp_group — 获取 Attention TP 进程组
)
from sglang.srt.layers.moe import initialize_moe_config  # initialize_moe_config — 初始化 MoE 全局配置
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config  # initialize_fp4_gemm_config — 初始化 FP4 GEMM 配置
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config  # initialize_fp8_gemm_config — 初始化 FP8 GEMM 配置
from sglang.srt.lora.lora_overlap_loader import LoRAOverlapLoader  # LoRAOverlapLoader — 与计算重叠的 LoRA 权重加载器
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator  # HiSparseCoordinator — HiSparse 预填充-解码过渡协调器
# io_struct — 所有 IPC 请求/响应的数据结构定义
from sglang.srt.managers.io_struct import (
    AbortReq,  # AbortReq — 中止请求
    ActiveRanksOutput,  # ActiveRanksOutput — 活跃 DP rank 输出
    AddExternalCorpusReqInput,  # AddExternalCorpusReqInput — 添加外部语料库请求
    AddExternalCorpusReqOutput,  # AddExternalCorpusReqOutput — 添加外部语料库响应
    AttachHiCacheStorageReqInput,  # AttachHiCacheStorageReqInput — 挂载 HiCache 存储后端请求
    AttachHiCacheStorageReqOutput,  # AttachHiCacheStorageReqOutput — 挂载 HiCache 存储后端响应
    BaseBatchReq,  # BaseBatchReq — 批量请求基类
    BaseReq,  # BaseReq — 单个请求基类
    BatchTokenizedEmbeddingReqInput,  # BatchTokenizedEmbeddingReqInput — 批量嵌入请求
    BatchTokenizedGenerateReqInput,  # BatchTokenizedGenerateReqInput — 批量生成请求
    CheckWeightsReqInput,  # CheckWeightsReqInput — 权重校验请求
    ClearHiCacheReqInput,  # ClearHiCacheReqInput — 清除 HiCache 请求
    ClearHiCacheReqOutput,  # ClearHiCacheReqOutput — 清除 HiCache 响应
    CloseSessionReqInput,  # CloseSessionReqInput — 关闭会话请求
    ContinueGenerationReqInput,  # ContinueGenerationReqInput — 恢复生成请求
    DestroyWeightsUpdateGroupReqInput,  # DestroyWeightsUpdateGroupReqInput — 销毁权重更新组请求
    DetachHiCacheStorageReqInput,  # DetachHiCacheStorageReqInput — 卸载 HiCache 存储后端请求
    DetachHiCacheStorageReqOutput,  # DetachHiCacheStorageReqOutput — 卸载 HiCache 存储后端响应
    DumperControlReqInput,  # DumperControlReqInput — 调试转储控制请求
    DumperControlReqOutput,  # DumperControlReqOutput — 调试转储控制响应
    ExpertDistributionReq,  # ExpertDistributionReq — MoE 专家分布统计请求
    ExpertDistributionReqOutput,  # ExpertDistributionReqOutput — MoE 专家分布统计响应
    ExpertDistributionReqType,  # ExpertDistributionReqType — 专家分布统计操作类型枚举
    FlushCacheReqInput,  # FlushCacheReqInput — 清空缓存请求
    FlushCacheReqOutput,  # FlushCacheReqOutput — 清空缓存响应
    FreezeGCReq,  # FreezeGCReq — 冻结 GC 请求
    GetInternalStateReq,  # GetInternalStateReq — 获取内部状态请求
    GetInternalStateReqOutput,  # GetInternalStateReqOutput — 获取内部状态响应
    GetLoadsReqInput,  # GetLoadsReqInput — 获取负载信息请求
    GetWeightsByNameReqInput,  # GetWeightsByNameReqInput — 按名称获取权重请求
    HealthCheckOutput,  # HealthCheckOutput — 健康检查响应
    InitWeightsSendGroupForRemoteInstanceReqInput,  # InitWeightsSendGroupForRemoteInstanceReqInput — 初始化权重发送组请求
    InitWeightsSendGroupForRemoteInstanceReqOutput,  # InitWeightsSendGroupForRemoteInstanceReqOutput — 初始化权重发送组响应
    InitWeightsUpdateGroupReqInput,  # InitWeightsUpdateGroupReqInput — 初始化权重更新组请求
    ListExternalCorporaReqInput,  # ListExternalCorporaReqInput — 列举外部语料库请求
    ListExternalCorporaReqOutput,  # ListExternalCorporaReqOutput — 列举外部语料库响应
    LoadLoRAAdapterFromTensorsReqInput,  # LoadLoRAAdapterFromTensorsReqInput — 从张量加载 LoRA 适配器请求
    LoadLoRAAdapterFromTensorsReqOutput,  # LoadLoRAAdapterFromTensorsReqOutput — 从张量加载 LoRA 适配器响应
    LoadLoRAAdapterReqInput,  # LoadLoRAAdapterReqInput — 加载 LoRA 适配器请求
    LoadLoRAAdapterReqOutput,  # LoadLoRAAdapterReqOutput — 加载 LoRA 适配器响应
    OpenSessionReqInput,  # OpenSessionReqInput — 打开会话请求
    PauseGenerationReqInput,  # PauseGenerationReqInput — 暂停生成请求
    ProfileReq,  # ProfileReq — 性能分析请求
    ReleaseMemoryOccupationReqInput,  # ReleaseMemoryOccupationReqInput — 释放内存占用请求
    RemoveExternalCorpusReqInput,  # RemoveExternalCorpusReqInput — 移除外部语料库请求
    RemoveExternalCorpusReqOutput,  # RemoveExternalCorpusReqOutput — 移除外部语料库响应
    ResumeMemoryOccupationReqInput,  # ResumeMemoryOccupationReqInput — 恢复内存占用请求
    RpcReqInput,  # RpcReqInput — RPC 调用请求
    RpcReqOutput,  # RpcReqOutput — RPC 调用响应
    SendWeightsToRemoteInstanceReqInput,  # SendWeightsToRemoteInstanceReqInput — 发送权重到远程实例请求
    SendWeightsToRemoteInstanceReqOutput,  # SendWeightsToRemoteInstanceReqOutput — 发送权重到远程实例响应
    SetInternalStateReq,  # SetInternalStateReq — 设置内部状态请求
    SetInternalStateReqOutput,  # SetInternalStateReqOutput — 设置内部状态响应
    SlowDownReqInput,  # SlowDownReqInput — 人工减速请求（调试用）
    SlowDownReqOutput,  # SlowDownReqOutput — 人工减速响应
    TokenizedEmbeddingReqInput,  # TokenizedEmbeddingReqInput — 已 tokenize 的嵌入请求
    TokenizedGenerateReqInput,  # TokenizedGenerateReqInput — 已 tokenize 的生成请求
    UnloadLoRAAdapterReqInput,  # UnloadLoRAAdapterReqInput — 卸载 LoRA 适配器请求
    UnloadLoRAAdapterReqOutput,  # UnloadLoRAAdapterReqOutput — 卸载 LoRA 适配器响应
    UpdateWeightFromDiskReqInput,  # UpdateWeightFromDiskReqInput — 从磁盘更新权重请求
    UpdateWeightsFromDistributedReqInput,  # UpdateWeightsFromDistributedReqInput — 从分布式更新权重请求
    UpdateWeightsFromIPCReqInput,  # UpdateWeightsFromIPCReqInput — 从 IPC 更新权重请求
    UpdateWeightsFromTensorReqInput,  # UpdateWeightsFromTensorReqInput — 从张量更新权重请求
)
from sglang.srt.managers.mm_utils import (
    has_shm_features,  # has_shm_features — 检查请求列表是否含共享内存多模态特征
    init_mm_embedding_cache,  # init_mm_embedding_cache — 初始化多模态嵌入缓存
    unwrap_shm_features,  # unwrap_shm_features — 从共享内存指针展开实际多模态数据
)
from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors  # get_mm_processor — 获取多模态处理器；import_processors — 动态导入处理器
from sglang.srt.managers.overlap_utils import FutureMap  # FutureMap — overlap 调度中存储异步 GPU 结果的映射表
from sglang.srt.managers.prefill_delayer import (
    PrefillDelayer,  # PrefillDelayer — 延迟调度 prefill 以平衡负载的组件
    PrefillDelayerSinglePassExecutor,  # PrefillDelayerSinglePassExecutor — 单次循环中执行 PrefillDelayer 决策的执行器
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,  # FINISH_ABORT — 请求被中止时的完成原因
    ModelWorkerBatch,  # ModelWorkerBatch — 发给 ModelWorker 的批次数据结构
    MultimodalInputs,  # MultimodalInputs — 多模态输入（图片/视频等）的封装
    Req,  # Req — 单个推理请求对象
    ScheduleBatch,  # ScheduleBatch — 调度器管理的一个批次
)
from sglang.srt.managers.schedule_policy import (
    AddReqResult,  # AddReqResult — 请求加入预填充批次的结果枚举
    PrefillAdder,  # PrefillAdder — 负责将请求逐条填入预填充批次的辅助类
    SchedulePolicy,  # SchedulePolicy — 调度策略（LPM/LFP/DFS/随机等）
)
from sglang.srt.managers.scheduler_dp_attn_mixin import SchedulerDPAttnMixin  # SchedulerDPAttnMixin — DP Attention 模式调度器 Mixin
from sglang.srt.managers.scheduler_input_blocker import SchedulerInputBlocker  # SchedulerInputBlocker — 协同批量生成时阻断输入的组件
from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,  # SchedulerOutputProcessorMixin — 批次输出处理（流式输出、token 追加等）Mixin
)
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin  # SchedulerPPMixin — 流水线并行事件循环 Mixin
from sglang.srt.managers.scheduler_profiler_mixin import SchedulerProfilerMixin  # SchedulerProfilerMixin — 性能分析 Mixin
from sglang.srt.managers.scheduler_recv_skipper import SchedulerRecvSkipper  # SchedulerRecvSkipper — overlap 模式下按需跳过收包的组件
from sglang.srt.managers.scheduler_runtime_checker_mixin import (
    SchedulerRuntimeCheckerMixin,  # SchedulerRuntimeCheckerMixin — 运行时健康检查 Mixin
    create_scheduler_watchdog,  # create_scheduler_watchdog — 创建调度器看门狗线程
)
from sglang.srt.managers.scheduler_update_weights_mixin import (
    SchedulerUpdateWeightsMixin,  # SchedulerUpdateWeightsMixin — 在线权重更新 Mixin
)
from sglang.srt.managers.utils import GenerationBatchResult, validate_input_length  # GenerationBatchResult — 生成批次结果；validate_input_length — 校验输入长度
from sglang.srt.mem_cache.cache_init_params import CacheInitParams  # CacheInitParams — 缓存初始化参数
from sglang.srt.mem_cache.common import release_kv_cache  # release_kv_cache — 释放请求的 KV Cache
from sglang.srt.mem_cache.radix_cache import RadixCache  # RadixCache — 基于 Radix 树的前缀共享 KV Cache
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors  # ForwardMode — 前向传播模式枚举；PPProxyTensors — PP 代理张量
from sglang.srt.model_loader.utils import get_resolved_model_impl  # get_resolved_model_impl — 获取解析后的模型实现类型
from sglang.srt.multiplex.multiplexing_mixin import SchedulerMultiplexMixin  # SchedulerMultiplexMixin — PD 多路复用调度器 Mixin
from sglang.srt.observability.req_time_stats import (
    real_time,  # real_time — 获取当前时间（用于空闲检测）
    set_schedule_time_batch,  # set_schedule_time_batch — 记录批次调度时间戳
    set_time_batch,  # set_time_batch — 为批次中的所有请求设置时间戳
)
from sglang.srt.observability.scheduler_metrics_mixin import (
    RECORD_STEP_TIME,  # RECORD_STEP_TIME — 是否记录步进时间的全局开关
    PrefillStats,  # PrefillStats — 预填充统计数据类
    SchedulerMetricsMixin,  # SchedulerMetricsMixin — Prometheus 指标收集 Mixin
)
from sglang.srt.observability.trace import process_tracing_init, trace_set_thread_info  # process_tracing_init — 初始化 OTLP 追踪；trace_set_thread_info — 设置线程追踪信息
from sglang.srt.parser.reasoning_parser import ReasoningParser  # ReasoningParser — 推理链解析器（think token 检测）
from sglang.srt.plugins import load_plugins  # load_plugins — 加载用户插件
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo  # SamplingBatchInfo — 采样参数批量封装
from sglang.srt.server_args import PortArgs, ServerArgs, get_global_server_args  # PortArgs — 端口配置；ServerArgs — 服务器参数；get_global_server_args — 获取全局服务器参数
from sglang.srt.session.session_controller import SessionController  # SessionController — 有状态会话（持续对话）控制器
from sglang.srt.session.streaming_session import StreamingSession  # StreamingSession — 流式会话包装器
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # SpeculativeAlgorithm — 推测解码算法枚举（EAGLE/Ngram/DFLASH等）
from sglang.srt.utils import (
    DynamicGradMode,  # DynamicGradMode — 动态控制 torch.no_grad 的上下文装饰器
    broadcast_pyobj,  # broadcast_pyobj — 跨进程广播 Python 对象（gloo）
    configure_gc_logger,  # configure_gc_logger — 配置 GC 日志记录
    configure_logger,  # configure_logger — 配置进程日志格式
    freeze_gc,  # freeze_gc — 冻结 Python GC（减少延迟抖动）
    get_available_gpu_memory,  # get_available_gpu_memory — 获取可用 GPU 显存大小
    get_bool_env_var,  # get_bool_env_var — 读取布尔类型环境变量
    get_int_env_var,  # get_int_env_var — 读取整型环境变量
    is_mps,  # is_mps — 判断是否在 Apple MPS 设备上运行
    kill_itself_when_parent_died,  # kill_itself_when_parent_died — 父进程死亡时自杀
    point_to_point_pyobj,  # point_to_point_pyobj — PP 流水线中点对点传送 Python 对象
    require_mlp_sync,  # require_mlp_sync — 判断是否需要 DP+MoE 的 MLP 同步
    set_gpu_proc_affinity,  # set_gpu_proc_affinity — 设置进程 CPU 亲和性
    set_random_seed,  # set_random_seed — 设置随机种子（复现性）
    suppress_other_loggers,  # suppress_other_loggers — 屏蔽非 sglang 模块日志
)
from sglang.srt.utils.common import is_npu  # is_npu — 判断是否在华为 NPU 上运行
from sglang.srt.utils.hf_transformers_utils import (
    get_processor,  # get_processor — 获取 HuggingFace 多模态 processor
    get_tokenizer,  # get_tokenizer — 获取 HuggingFace tokenizer
    get_tokenizer_from_processor,  # get_tokenizer_from_processor — 从 processor 中提取 tokenizer
)
from sglang.srt.utils.network import get_zmq_socket  # get_zmq_socket — 创建 ZMQ socket 的工具函数
from sglang.srt.utils.numa_utils import get_numa_node_if_available, numa_bind_to_node  # get_numa_node_if_available — 获取可用 NUMA 节点；numa_bind_to_node — 绑定进程到 NUMA 节点
from sglang.srt.utils.tensor_bridge import use_mlx  # use_mlx — 判断是否使用 Apple MLX 后端
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter  # TorchMemorySaverAdapter — GPU 内存节省器适配器（权重卸载等）
from sglang.utils import TypeBasedDispatcher, get_exception_traceback  # TypeBasedDispatcher — 基于类型的请求分发器；get_exception_traceback — 格式化异常堆栈

# -------- 硬件平台适配：MPS（Apple Metal）vs CUDA --------
if is_mps():
    CudaStreamContext = nullcontext  # MPS 上用空上下文代替 CUDA 流上下文
    from sglang.srt.hardware_backend.mlx.scheduler_mixin import SchedulerMlxOverlapMixin  # MLX overlap 调度 Mixin（Apple 平台专用）
else:
    from torch.cuda import StreamContext as CudaStreamContext  # CUDA 平台的流上下文

    class SchedulerMlxOverlapMixin:  # 非 MPS 平台提供空实现，避免继承报错
        pass


# -------- 模块级全局变量 --------
logger = logging.getLogger(__name__)  # 获取当前模块的 logger

# Test retract decode for debugging purposes
TEST_RETRACT = envs.SGLANG_TEST_RETRACT.get()  # 是否开启测试用的强制 retract（调试专用）
TEST_RETRACT_INTERVAL = envs.SGLANG_TEST_RETRACT_INTERVAL.get()  # 触发测试 retract 的 forward 间隔步数
TEST_RETRACT_NO_PREFILL_BS = envs.SGLANG_TEST_RETRACT_NO_PREFILL_BS.get()  # 测试 retract 时禁止 prefill 的 batch size 阈值

_is_npu = is_npu()  # 全局标记：当前是否在华为 NPU 上运行


@dataclass
class EmbeddingBatchResult:
    """Result from an embedding/classification forward pass.

    Attributes:
        embeddings: Model output — pooled embeddings or classification logits.
        pooled_hidden_states: Raw hidden states before the task head.  Present
            only when the batch contained ``return_pooled_hidden_states=True``
            requests.  Tensor (uniform shapes) or list of tensors (MIS).
        copy_done: CUDA event recorded after the async CPU copy completes.
    """

    embeddings: torch.Tensor  # 模型输出的嵌入向量（池化嵌入或分类 logits）
    pooled_hidden_states: Optional[torch.Tensor] = None  # 任务头之前的原始隐藏状态（仅 return_pooled_hidden_states=True 时存在）
    copy_done: Optional[torch.cuda.Event] = None  # 异步 CPU 拷贝完成时记录的 CUDA 事件

    # -------- 异步 CPU 拷贝方法 --------
    def copy_to_cpu(self):
        """Copy embeddings and pooled hidden states to CPU for overlap scheduling."""
        if isinstance(self.embeddings, torch.Tensor):
            self.copy_done = torch.get_device_module(self.embeddings.device).Event()  # 创建设备事件用于同步
            self.embeddings = self.embeddings.to("cpu", non_blocking=True)  # 非阻塞异步拷贝到 CPU
        else:
            assert isinstance(self.embeddings, list)  # embeddings 也可能是 Tensor 列表（MIS 场景）
            if len(self.embeddings) == 0:
                return  # 空列表直接返回

            self.copy_done = torch.get_device_module(self.embeddings[0].device).Event()  # 取第一个张量的设备创建事件
            self.embeddings = [
                emb.to("cpu", non_blocking=True) for emb in self.embeddings  # 逐个非阻塞拷贝
            ]

        if self.pooled_hidden_states is not None:
            if isinstance(self.pooled_hidden_states, list):
                self.pooled_hidden_states = [
                    t.to("cpu", non_blocking=True) for t in self.pooled_hidden_states  # 列表形式的隐藏状态逐个拷贝
                ]
            else:
                self.pooled_hidden_states = self.pooled_hidden_states.to(
                    "cpu", non_blocking=True  # 单张量形式的隐藏状态拷贝
                )

        self.copy_done.record()  # 记录事件，标志所有异步拷贝操作已提交


# -------- DFLASH 推测解码请求校验 --------
def validate_dflash_request(req: Req) -> Optional[str]:
    # 校验请求是否与 DFLASH 推测解码兼容，返回 None 表示合法，否则返回错误信息字符串
    if req.return_logprob:
        return "DFLASH speculative decoding does not support return_logprob yet."  # DFLASH 暂不支持 logprob 返回

    if (
        req.sampling_params.json_schema is not None  # JSON Schema 约束生成
        or req.sampling_params.regex is not None  # 正则表达式约束生成
        or req.sampling_params.ebnf is not None  # EBNF 文法约束生成
        or req.sampling_params.structural_tag is not None  # 结构化标签约束生成
    ):
        return (
            "DFLASH speculative decoding does not support "
            "grammar-constrained decoding yet."  # DFLASH 暂不支持任何文法约束解码
        )

    return None  # 请求合法


# -------- Scheduler 主类 --------
# 通过多重继承组合各功能 Mixin：输出处理、权重更新、性能分析、指标收集、
# PD 解耦（预填充/解码）、多路复用、运行时检查、流水线并行、DP Attention、
# 扩散 LLM、Apple MLX overlap 等
class Scheduler(
    SchedulerOutputProcessorMixin,
    SchedulerUpdateWeightsMixin,
    SchedulerProfilerMixin,
    SchedulerMetricsMixin,
    SchedulerDisaggregationDecodeMixin,
    SchedulerDisaggregationPrefillMixin,
    SchedulerMultiplexMixin,
    SchedulerRuntimeCheckerMixin,
    SchedulerPPMixin,
    SchedulerDPAttnMixin,
    SchedulerDllmMixin,
    SchedulerMlxOverlapMixin,
):
    """A scheduler that manages a tensor parallel GPU worker."""

    # -------- 初始化方法 --------
    def __init__(
        self,
        server_args: ServerArgs,  # 服务器全局参数
        port_args: PortArgs,  # IPC 端口参数
        gpu_id: int,  # 物理 GPU 编号
        tp_rank: int,  # 张量并行 rank
        moe_ep_rank: int,  # MoE 专家并行 rank
        pp_rank: int,  # 流水线并行 rank
        attn_cp_rank: int,  # Attention Context Parallel rank
        moe_dp_rank: int,  # MoE 数据并行 rank
        dp_rank: Optional[int],  # 数据并行 rank（可为 None）
    ):
        self.is_initializing = True  # 初始化标志，初始化完成后设为 False
        self.init_soft_watchdog(server_args)  # 启动软超时看门狗（初始化阶段）

        # -------- 解析并存储各类并行/功能开关参数 --------
        self.server_args = server_args  # 保存服务器参数引用
        self.tp_rank = tp_rank  # 张量并行 rank
        self.moe_ep_rank = moe_ep_rank  # MoE 专家并行 rank
        self.pp_rank = pp_rank  # 流水线并行 rank
        self.attn_cp_rank = attn_cp_rank  # Attention Context Parallel rank
        self.attn_cp_size = server_args.attn_cp_size  # Attention CP 并行度
        self.moe_dp_rank = moe_dp_rank  # MoE 数据并行 rank
        self.moe_dp_size = server_args.moe_dp_size  # MoE 数据并行度
        self.dp_rank = dp_rank  # 数据并行 rank
        self.tp_size = server_args.tp_size  # 张量并行度
        self.moe_ep_size = server_args.ep_size  # MoE 专家并行度
        self.pp_size = server_args.pp_size  # 流水线并行度
        self.dp_size = server_args.dp_size  # 数据并行度
        self.nccl_port = port_args.nccl_port  # NCCL 通信端口
        self.schedule_policy = server_args.schedule_policy  # 调度策略名称
        self.enable_priority_scheduling = server_args.enable_priority_scheduling  # 是否启用优先级调度
        self.abort_on_priority_when_disabled = (
            server_args.abort_on_priority_when_disabled  # 优先级调度未启用时收到带优先级请求是否 abort
        )
        self.schedule_low_priority_values_first = (
            server_args.schedule_low_priority_values_first  # True=数值小的优先级值先调度
        )
        self.priority_scheduling_preemption_threshold = (
            server_args.priority_scheduling_preemption_threshold  # 优先级抢占的阈值
        )
        self.enable_lora = server_args.enable_lora  # 是否启用 LoRA
        self.enable_lora_overlap_loading = server_args.enable_lora_overlap_loading  # 是否与计算重叠加载 LoRA
        self.max_loras_per_batch = server_args.max_loras_per_batch  # 每批次最多并发 LoRA 适配器数
        self.enable_overlap = not server_args.disable_overlap_schedule and not use_mlx()  # CUDA 平台 overlap 调度开关
        self.enable_overlap_mlx = not server_args.disable_overlap_schedule and use_mlx()  # MLX 平台 overlap 调度开关
        self.enable_pdmux = server_args.enable_pdmux  # 是否启用 PD 多路复用
        self.skip_tokenizer_init = server_args.skip_tokenizer_init  # 是否跳过 tokenizer 初始化
        self.stream_interval = server_args.stream_interval  # 流式输出的 token 间隔
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm  # 推测解码算法类型
        )
        self.gpu_id = gpu_id  # 物理 GPU ID
        self.page_size = server_args.page_size  # KV Cache 页大小（token 数）
        self.enable_hierarchical_cache = server_args.enable_hierarchical_cache  # 是否启用分层 KV Cache（GPU+CPU+SSD）
        self.enable_hicache_storage = server_args.hicache_storage_backend is not None  # HiCache 存储后端是否启用
        self.max_recv_per_poll = envs.SGLANG_SCHEDULER_MAX_RECV_PER_POLL.get()  # 单轮最多接收的请求数上限
        self.enable_hisparse = server_args.enable_hisparse  # 是否启用 HiSparse 注意力
        self.hisparse_coordinator: Optional[HiSparseCoordinator] = None  # HiSparse 协调器（延迟到缓存初始化后赋值）

        # -------- 分布式 Rank 信息计算 --------
        # Distributed rank info
        self.attn_tp_rank, self.attn_tp_size, self.attn_dp_rank = (
            compute_dp_attention_world_info(  # 计算 DP Attention 下 rank/size 信息
                server_args.enable_dp_attention,
                self.tp_rank,
                self.tp_size,
                self.dp_size,
                self.attn_cp_size,
            )
        )

        # -------- 各子组件依次初始化（顺序很重要，互有依赖）--------
        # Init model configs
        self.init_model_config()  # 初始化模型配置（hf_config、vocab_size 等）

        # Init metrics stats
        self.init_metrics(tp_rank, pp_rank, dp_rank)  # 初始化 Prometheus 等指标收集器

        # Init inter-process communication
        self.init_ipc_channels(port_args)  # 初始化 ZMQ IPC 通道（收发请求/结果）

        # Init PD-multiplexing context
        if self.enable_pdmux:
            self.init_pdmux()  # 初始化 PD 多路复用上下文

        # Init tokenizer
        self.init_tokenizer()  # 初始化 tokenizer 和多模态 processor

        # Init moe config and GEMM config (FP8 GEMM, etc.)
        self.init_moe_gemm_config()  # 初始化 MoE 配置及 FP8/FP4 GEMM 配置

        # Init mamba backend
        self.init_mamba_backend()  # 初始化 Mamba SSM 后端（如使用 Mamba 架构）

        # Launch a model worker and draft model worker if using speculative decoding
        self.init_model_worker()  # 初始化 TP worker（及推测解码的 draft worker）

        if (t := envs.SGLANG_TEST_STUCK_SCHEDULER_INIT.get()) > 0:
            time.sleep(t)  # 测试用：人为卡住初始化流程（模拟超时场景）

        # Init cache and memory pool
        self.init_cache_with_memory_pool()  # 初始化 KV Cache 和内存池（根据配置选择具体实现）

        # Register draft KV pool (when spec + HiCache co-enabled).
        self._maybe_register_hicache_draft()  # 如果同时启用了投机解码和 HiCache，注册草稿 KV 池

        # Init running status
        self.init_running_status()  # 初始化运行状态（等待队列、running_batch 等）

        # Init chunked prefill
        self.init_chunked_prefill()  # 初始化分块预填充（chunked prefill）相关参数

        # Init diffusion LLM
        self.init_diffusion_llm()  # 初始化扩散 LLM（dLLM）相关配置（非扩散模型为空操作）

        # Init schedule policy and new token estimation
        self.init_schedule_policy()  # 初始化调度策略和新 token 比例估算参数

        # Init watchdog, memory saver, input blocker and recv skipper
        self.init_watch_dog_memory_saver_input_blocker()  # 初始化看门狗、内存节省器、输入阻断器和收包跳过器

        # Init profiler
        self.init_profiler()  # 初始化性能分析器（torch profiler 集成）

        # Init prefill-decodedisaggregation
        self.init_disaggregation()  # 初始化 PD 解耦相关队列和缓冲区

        # Init overlap schedule
        self.init_overlap()  # 初始化 overlap 调度（CUDA 流、FutureMap 等）

        # Init Ngram Embedding
        self.maybe_init_ngram_embedding()  # 如果使用 Ngram Embedding，初始化 token 表

        # Init prefill kv split size when deterministic inference is enabled with various attention backends
        self.init_deterministic_inference_config()  # 初始化确定性推理的预填充 KV 分块对齐大小

        # Init request dispatcher
        self.init_request_dispatcher()  # 初始化基于类型的请求分发器（TypeBasedDispatcher）

        # Init LoRA overlap loader
        if self.enable_lora_overlap_loading:
            self.lora_overlap_loader = LoRAOverlapLoader(
                self.tp_worker.model_runner.lora_manager  # 将 LoRA 管理器传入，实现计算与加载重叠
            )

        # Init the grammar backend for constrained generation
        self.grammar_manager = GrammarManager(self)  # 初始化文法管理器（JSON Schema / 正则 / EBNF 约束生成）

        self.is_initializing = False  # 标记初始化完成

    # -------- 模型配置初始化 --------
    def init_model_config(self):
        # 从 server_args 中构建 ModelConfig（含 hf_config、dtype、context_len 等）
        self.model_config = ModelConfig.from_server_args(self.server_args)
        if _is_npu:
            # make sure the page size is not larger than block_size and chunked_prefill_size on NPU backend
            # the npu backend request the defined page size to be no larger than block_size and chunked_prefill_size
            from sglang.srt.dllm.config import DllmConfig  # NPU 平台延迟导入扩散 LLM 配置

            self.dllm_config = (  # For diffusion LLM
                DllmConfig.from_server_args(self.server_args)  # 构建扩散 LLM 配置
                if self.server_args.dllm_algorithm is not None
                else None
            )
            if self.dllm_config:
                if self.dllm_config.block_size < self.page_size:  # dLLM block_size 必须 >= page_size
                    logger.warning(
                        "WARNING: "
                        f"The page size {self.page_size} should not be larger than dllm block size {self.dllm_config.block_size}."
                        f"Page size now falls back to {self.dllm_config.block_size}"
                    )
                    self.page_size = self.dllm_config.block_size  # 强制回退 page_size 到 block_size

    # -------- IPC 通道初始化 --------
    def init_ipc_channels(self, port_args: PortArgs):
        # 初始化与 TokenizerManager、DetokenizerManager、RPC 及指标收集的 ZMQ 通道
        context = zmq.Context(2)  # 创建 ZMQ 上下文，2 个 IO 线程
        self.idle_sleeper = None  # 空闲睡眠器（sleep_on_idle 模式下使用）

        # 仅 PP rank 0 且 attn_tp rank 0 且 attn_cp rank 0 的 rank 负责接收/发送请求
        if self.pp_rank == 0 and self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                context, zmq.PULL, port_args.scheduler_input_ipc_name, False  # 从 TokenizerManager 拉取已 tokenize 的请求
            )
            self.recv_from_rpc = get_zmq_socket(
                context, zmq.DEALER, port_args.rpc_ipc_name, False  # 从 RPC 接口接收控制命令
            )

            send_to_tokenizer = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_ipc_name, False  # 发送结果/响应给 TokenizerManager
            )
            if self.server_args.skip_tokenizer_init:
                # Directly send to the TokenizerManager
                send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.tokenizer_ipc_name, False  # 跳过 detokenizer 时直接发往 TokenizerManager
                )
            else:
                # Send to the DetokenizerManager
                send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.detokenizer_ipc_name, False  # 正常模式下发往 DetokenizerManager（解码 token → 文本）
                )

            self.send_to_tokenizer = SenderWrapper(send_to_tokenizer)  # 包装为 SenderWrapper（支持 http_worker_ipc 字段传递）
            self.send_to_detokenizer = SenderWrapper(send_to_detokenizer)  # 包装为 SenderWrapper

            if self.server_args.sleep_on_idle:
                self.idle_sleeper = IdleSleeper(
                    [
                        self.recv_from_tokenizer,
                        self.recv_from_rpc,  # 在这两个 socket 上 poll，空闲时降低 CPU 使用率
                    ]
                )
        else:
            # 非 leader rank：不直接接收/发送，设为 None 或空包装
            self.recv_from_tokenizer = None
            self.recv_from_rpc = None
            self.send_to_tokenizer = SenderWrapper(None)
            self.send_to_detokenizer = SenderWrapper(None)

        if self.current_scheduler_metrics_enabled:
            self.send_metrics_from_scheduler = get_zmq_socket(
                context, zmq.PUSH, port_args.metrics_ipc_name, False  # 发送调度器指标数据的 PUSH socket
            )

    # -------- Tokenizer 初始化 --------
    def init_tokenizer(self):
        # 根据是否多模态、是否跳过 tokenizer，加载相应的 tokenizer 或 processor
        server_args = self.server_args
        self.is_generation = self.model_config.is_generation  # 是否是生成模型（False 表示嵌入/分类模型）

        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None  # 跳过 tokenizer 初始化（由外部提供 token ids）
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    server_args.tokenizer_path,  # 多模态模型需要 processor（包含图像预处理）
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                    use_fast=not server_args.disable_fast_image_processor,
                    tokenizer_backend=server_args.tokenizer_backend,
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)  # 从 processor 中提取文本 tokenizer
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,  # 纯文本模型直接加载 tokenizer
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                    tokenizer_backend=server_args.tokenizer_backend,
                )

        # Load multimodal processor for M-RoPE fallback computation.
        self._mm_processor = None  # M-RoPE 位置计算的多模态 processor（gRPC 预处理路径的备用）
        if self.model_config.is_multimodal and self.processor is not None:
            try:
                import_processors("sglang.srt.multimodal.processors")  # 动态注册所有多模态 processor 实现
                self._mm_processor = get_mm_processor(
                    self.model_config.hf_config,
                    server_args,
                    self.processor,
                    "default",
                    skip_mm_pool=True,  # 调度器端不需要 mm pool（只需计算位置）
                )
            except Exception:
                logger.warning(
                    "Failed to load multimodal processor in scheduler; "
                    "M-RoPE fallback will not be available."
                )

        # Set reasoning_parser and think_end_id if --reasoning_parser is enabled
        if self.server_args.reasoning_parser and self.tokenizer:
            reasoning_parser = ReasoningParser(
                model_type=self.server_args.reasoning_parser, stream_reasoning=False  # 创建推理链解析器（不流式）
            )
            self.model_config.think_end_id = self.tokenizer.encode(
                reasoning_parser.detector.think_end_token, add_special_tokens=False  # 将 think 结束 token 转为 id，用于推理链截断
            )[0]

    # -------- Mamba 后端初始化 --------
    def init_mamba_backend(self) -> None:
        # 初始化 Mamba selective state update 后端（使用 Mamba 架构时调用）
        initialize_mamba_selective_state_update_backend(self.server_args)

    # -------- MoE 与 GEMM 配置初始化 --------
    def init_moe_gemm_config(self):
        # For the MM models, check the text_config for MoE settings
        config_to_check = getattr(
            self.model_config.hf_config, "text_config", self.model_config.hf_config  # 多模态模型从 text_config 读取 MoE 设置
        )

        if hasattr(config_to_check, "num_experts_per_tok"):
            initialize_moe_config(self.server_args)  # 初始化 MoE 全局配置（ep_size、topk 等）

        # Initialize GEMM-related configuration for FP8 and FP4 backends.
        initialize_fp8_gemm_config(self.server_args)  # 初始化 FP8 量化 GEMM 配置
        initialize_fp4_gemm_config(self.server_args)  # 初始化 FP4 量化 GEMM 配置

        # This must be called after initialize_moe_config
        self.require_mlp_sync = require_mlp_sync(self.server_args)  # 判断是否需要 DP+MoE 的 MLP 集体同步

    # -------- TP Model Worker 初始化 --------
    def init_tp_model_worker(self):
        # 构建 TP worker 的参数字典，选择 MLX 或 CUDA 实现
        worker_kwargs = dict(
            server_args=self.server_args,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            moe_ep_rank=self.moe_ep_rank,
            pp_rank=self.pp_rank,
            attn_cp_rank=self.attn_cp_rank,
            moe_dp_rank=self.moe_dp_rank,
            dp_rank=self.dp_rank,
            nccl_port=self.nccl_port,
        )

        # FIXME: move tp worker's init logic outside of the scheduler.
        if use_mlx():
            from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker  # Apple MLX 后端

            self.tp_worker = MlxTpModelWorker(**worker_kwargs)  # 使用 MLX TP worker
        else:
            from sglang.srt.managers.tp_worker import TpModelWorker  # CUDA 后端

            self.tp_worker = TpModelWorker(**worker_kwargs)  # 使用 CUDA TP worker

    # -------- 草稿 Worker 初始化（推测解码） --------
    def maybe_init_draft_worker(self):
        # 若不使用推测解码，跳过草稿 worker 初始化
        if self.spec_algorithm.is_none():
            self.draft_worker = None
            self.external_corpus_manager = None
            return

        # Launch a draft worker for speculative decoding
        draft_worker_kwargs = dict(
            server_args=self.server_args,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            moe_ep_rank=self.moe_ep_rank,
            nccl_port=self.nccl_port,
            target_worker=self.tp_worker,
            dp_rank=self.dp_rank,
            attn_cp_rank=self.attn_cp_rank,
            moe_dp_rank=self.moe_dp_rank,
        )

        if self.server_args.speculative_draft_load_format is not None:
            self.server_args.load_format = (
                self.server_args.speculative_draft_load_format  # 覆盖权重加载格式（草稿模型可能使用不同格式）
            )
            logger.info(
                f"Using draft model load_format: '{self.server_args.speculative_draft_load_format}'"
            )

        DraftWorkerClass = self.spec_algorithm.create_worker(self.server_args)  # 按算法类型（EAGLE/Ngram等）选择草稿 Worker 类
        self.draft_worker = DraftWorkerClass(**draft_worker_kwargs)  # 实例化草稿 Worker

        if self.spec_algorithm.is_ngram():
            from sglang.srt.speculative.external_corpus_manager import (
                ExternalCorpusManager,  # Ngram 推测解码的外部语料库管理器
            )

            self.external_corpus_manager = ExternalCorpusManager(
                self.draft_worker,
                self.send_to_tokenizer.send_output,  # 加载完成时通知 TokenizerManager
            )
        else:
            self.external_corpus_manager = None  # 非 Ngram 算法无需外部语料库

    # -------- 完整 Model Worker 初始化（TP + 草稿） --------
    def init_model_worker(self):
        self.init_tp_model_worker()  # 初始化主模型 TP worker
        self.maybe_init_draft_worker()  # 初始化草稿 worker（推测解码）

        # Dispatch the model worker
        if self.spec_algorithm.is_none():
            self.model_worker = self.tp_worker  # 无推测解码：主 worker 即 model worker
        else:
            self.model_worker = self.draft_worker  # 推测解码：草稿 worker 封装了主 worker

        # Get token and memory info from the model worker
        (
            self.max_total_num_tokens,  # KV Cache 总 token 容量
            self.max_prefill_tokens,  # 单次 prefill 最大 token 数
            self.max_running_requests,  # 同时运行的最大请求数
            self.max_queued_requests,  # 最大排队请求数（None 表示不限）
            self.max_req_len,  # 单个请求最大总长度（输入+输出）
            self.max_req_input_len,  # 单个请求最大输入长度
            self.random_seed,  # 随机种子
            self.device,  # 设备字符串（"cuda:N" 或 "cpu"）
            self.forward_stream,  # 前向传播 CUDA 流
            _,  # 保留字段（未使用）
            _,  # 保留字段（未使用）
            _,  # 保留字段（未使用）
        ) = self.tp_worker.get_worker_info()
        if not get_global_server_args().pp_max_micro_batch_size:
            get_global_server_args().pp_max_micro_batch_size = max(
                self.max_running_requests // self.pp_size, 1  # 每个 PP 阶段的最大 micro-batch size
            )

        self.tp_group = get_tp_group()  # 张量并行进程组
        self.tp_cpu_group = self.tp_group.cpu_group  # CPU（gloo）组，用于广播 Python 对象
        self.attn_tp_group = get_attention_tp_group()  # Attention TP 进程组
        self.attn_tp_cpu_group = self.attn_tp_group.cpu_group  # Attention TP CPU 组
        self.attn_cp_group = get_attention_cp_group()  # Attention Context Parallel 进程组
        self.attn_cp_cpu_group = self.attn_cp_group.cpu_group  # Attention CP CPU 组
        self.pp_group = get_pp_group()  # 流水线并行进程组
        self.world_group = get_world_group()  # 全局进程组

        # NOTE: dp_tp_* are request/data-plane coordination groups (not tensor collectives).
        # When DP attention is enabled, scope to the attention-TP group; otherwise use
        # the base TP group. Entry rank is the local rank 0 in that group.
        # Use the CPU (gloo) group to broadcast VLM Python objects and avoid CUDA
        # stream/device coupling (#11910).
        self.dp_tp_group = (
            self.attn_tp_group  # DP Attention 开启：用 attn_tp_group 协调请求广播
            if self.server_args.enable_dp_attention
            else self.tp_group  # 普通模式：用完整 tp_group
        )
        self.dp_tp_cpu_group = self.dp_tp_group.cpu_group  # 对应的 CPU（gloo）组

        self.pad_input_ids_func = self.tp_worker.get_pad_input_ids_func()  # 获取图像 token 扩展填充函数（模型专用）
        set_random_seed(self.random_seed)  # 设置全局随机种子保证可复现性

        # Print debug info
        if self.tp_rank == 0:
            avail_mem = get_available_gpu_memory(
                self.device, self.gpu_id, empty_cache=False
            )
            logger.info(
                f"max_total_num_tokens={self.max_total_num_tokens}, "
                f"chunked_prefill_size={self.server_args.chunked_prefill_size}, "
                f"max_prefill_tokens={self.max_prefill_tokens}, "
                f"max_running_requests={self.max_running_requests}, "
                f"context_len={self.model_config.context_len}, "
                f"{'available_cpu_mem' if self.device == 'cpu' else 'available_gpu_mem'}={avail_mem:.2f} GB"
            )

        if self.enable_metrics and hasattr(self, "metrics_collector"):
            self.metrics_collector.emit_cache_config_info(
                self.page_size, self.max_total_num_tokens // self.page_size
            )

    def init_cache_with_memory_pool(self):
        server_args = self.server_args
        uses_transformers_backend = (
            get_resolved_model_impl(self.model_config) == ModelImpl.TRANSFORMERS
        )

        # -------- 混合内存池检测（SWA / SSM / Mamba 混合架构）--------
        self.is_hybrid_swa = self.tp_worker.is_hybrid_swa  # 是否使用滑动窗口注意力（Hybrid SWA）
        _spec = self.tp_worker.model_runner.linear_attn_model_spec  # 线性注意力模型规格（如 Mamba 系列）
        _registry_needs_mamba = (
            _spec.uses_mamba_radix_cache if _spec is not None else False  # 该模型规格是否需要 Mamba 专用 RadixCache
        )
        self.is_hybrid_ssm = (
            self.tp_worker.model_runner.hybrid_gdn_config is not None  # GDN（Gated Dense Network）混合配置
            or self.tp_worker.model_runner.mamba2_config is not None   # Mamba2 配置
            or _registry_needs_mamba                                   # 或线性注意力规格要求 Mamba RadixCache
        )

        self.sliding_window_size = None  # 滑动窗口大小，非 SWA 模型时为 None
        if self.is_hybrid_swa:
            self.sliding_window_size = self.tp_worker.sliding_window_size  # 从 worker 读取实际窗口大小
            self.full_tokens_per_layer, self.swa_tokens_per_layer = (
                self.tp_worker.get_tokens_per_layer_info()  # 每层 full/swa token 容量（用于 KV 分配）
            )

        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            self.tp_worker.get_memory_pool()  # 获取请求→token 映射池 和 token→KV 分配器
        )

        # -------- Radix Cache 启用决策 --------
        self.disable_radix_cache = server_args.disable_radix_cache or (
            self.model_config.is_multimodal and uses_transformers_backend
            # 多模态 + Transformers 后端时强制禁用 RadixCache，避免前缀缓存错位
        )
        if self.disable_radix_cache and not server_args.disable_radix_cache:
            logger.warning(
                "Radix cache is disabled for multimodal models with the "
                "Transformers backend to avoid multimodal prefix-cache mismatches."
            )

        effective_chunked_prefill_size = server_args.chunked_prefill_size  # 生效的 chunked prefill 大小
        if self.model_config.is_multimodal and uses_transformers_backend:
            effective_chunked_prefill_size = None  # 多模态+Transformers 后端不支持 chunked prefill

        # -------- 构造 CacheInitParams，传给各类 tree_cache 实现 --------
        params = CacheInitParams(
            disable=self.disable_radix_cache,                      # 是否禁用 RadixCache
            req_to_token_pool=self.req_to_token_pool,              # 请求→token 池
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,  # KV 分配器
            page_size=self.page_size,                              # KV 页大小
            is_eagle=self.spec_algorithm.is_eagle(),               # 是否为 EAGLE 推测解码
            tp_cache_group=(
                self.attn_tp_cpu_group                             # DP Attention 时用 attn_tp 组
                if self.server_args.enable_dp_attention
                else self.tp_cpu_group                             # 普通模式用标准 tp 组
            ),
            attn_cp_cache_group=self.attn_cp_cpu_group,            # Attention Context Parallel 组
            attn_tp_cache_group=self.attn_tp_cpu_group,            # Attention TP 组
            eviction_policy=server_args.radix_eviction_policy,     # KV 驱逐策略（LRU 等）
            enable_metrics=self.enable_metrics,                    # 是否收集缓存命中率指标
            enable_kv_cache_events=self.enable_kv_cache_events,    # 是否发布 KV 缓存事件
            enable_mamba_extra_buffer=server_args.enable_mamba_extra_buffer(),  # Mamba 额外缓存 buffer
            pp_rank=self.pp_rank,                                  # 流水线并行 rank
            pp_size=self.pp_size,                                  # 流水线并行总数
            chunked_prefill_size=effective_chunked_prefill_size,   # chunked prefill 大小
            sliding_window_size=self.sliding_window_size,          # 滑动窗口大小
        )

        # -------- 根据特性选择 tree_cache 实现 --------
        if effective_chunked_prefill_size is not None and self.disable_radix_cache:
            # chunked prefill + 无 RadixCache：使用 ChunkCache
            if not self.is_hybrid_swa:
                from sglang.srt.mem_cache.chunk_cache import ChunkCache

                self.tree_cache = ChunkCache(params)  # 普通模型的 ChunkCache
            else:
                from sglang.srt.mem_cache.chunk_cache import SWAChunkCache

                self.tree_cache = SWAChunkCache(params)  # SWA 模型的 ChunkCache
        else:
            if envs.SGLANG_EXPERIMENTAL_CPP_RADIX_TREE.get():
                # lazy import to avoid JIT overhead
                from sglang.srt.mem_cache.radix_cache_cpp import RadixCacheCpp

                logger.info("Using experimental C++ radix tree implementation.")
                self.tree_cache = RadixCacheCpp(params=params, server_args=server_args)  # 实验性 C++ Radix Tree
            elif self.enable_hierarchical_cache:
                # HiCache：分层 KV 缓存（GPU → Host Memory → L3 Storage）
                if self.is_hybrid_ssm:
                    from sglang.srt.mem_cache.hi_mamba_radix_cache import (
                        HiMambaRadixCache,
                    )

                    self.tree_cache = HiMambaRadixCache(
                        params=params, server_args=server_args
                    )  # 混合 SSM 的 HiCache
                else:
                    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

                    self.tree_cache = HiRadixCache(
                        params=params, server_args=server_args
                    )  # 标准 HiRadixCache
                self.tp_worker.register_hicache_layer_transfer_counter(
                    self.tree_cache.cache_controller.layer_done_counter
                    # 注册层迁移计数器，让 TP worker 感知 HiCache 层传输进度
                )
            elif envs.SGLANG_ENABLE_UNIFIED_RADIX_TREE.get():
                # 统一 RadixTree：同时管理 FULL + SWA 或 FULL + MAMBA 两种组件
                from sglang.srt.mem_cache.unified_cache_components import (
                    ComponentType,
                )
                from sglang.srt.mem_cache.unified_radix_cache import (
                    UnifiedRadixCache,
                )

                tree_components = [ComponentType.FULL]  # 必须有 FULL 组件
                if self.is_hybrid_swa or self.is_hybrid_ssm:
                    tree_components.append(
                        ComponentType.SWA if self.is_hybrid_swa else ComponentType.MAMBA
                        # 根据模型类型追加 SWA 或 MAMBA 组件
                    )
                params.tree_components = tuple(tree_components)
                self.tree_cache = UnifiedRadixCache(params)  # 统一 RadixCache
            elif self.is_hybrid_swa:
                from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache

                self.tree_cache = SWARadixCache(params=params)  # 纯 SWA 模型的 RadixCache
            elif self.is_hybrid_ssm:
                from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache

                self.tree_cache = MambaRadixCache(params)  # 纯 Mamba/SSM 模型的 RadixCache
            elif server_args.enable_lmcache:
                # LMCache：外部 KV 缓存存储后端（支持 Redis 等）
                from sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache import (
                    LMCRadixCache,
                )

                self.tree_cache = LMCRadixCache(
                    params=params,
                    model_config=self.model_config,
                    tp_size=self.tp_size,
                    rank=self.tp_rank,
                    tp_group=self.tp_group,
                )  # LMCache 集成的 RadixCache
            else:
                self.tree_cache = RadixCache(params)  # 默认：标准 RadixCache

        # 如果启用流式会话，需要用 StreamingSession 包装 tree_cache
        if (
            server_args.enable_streaming_session
            and not self.tree_cache.supports_streaming_session()
        ):
            self.tree_cache = StreamingSession(self.tree_cache)  # 包装支持流式会话

        if self.enable_hisparse:
            # Coordinator was created inside ModelRunner.initialize() before CUDA graph capture
            self.hisparse_coordinator = self.tp_worker.model_runner.hisparse_coordinator  # HiSparse 注意力协调器
            self.hisparse_coordinator.set_decode_producer_stream(self.forward_stream)  # 绑定解码生产者 CUDA 流

        # 解码端 KV Cache offload manager（解码节点 + 开启 offload 时使用）
        if (
            server_args.disaggregation_mode == "decode"
            and server_args.disaggregation_decode_enable_offload_kvcache
        ):
            self.decode_offload_manager = DecodeKVCacheOffloadManager(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                tp_group=params.tp_cache_group,
                tree_cache=self.tree_cache,
                server_args=self.server_args,
            )  # 解码端 KV Cache offload 管理器（将 KV 卸载到 CPU/SSD）
        else:
            self.decode_offload_manager = None  # 不开启 offload

        embedding_cache_size = envs.SGLANG_VLM_CACHE_SIZE_MB.get()  # 多模态嵌入缓存大小（MB）
        init_mm_embedding_cache(embedding_cache_size * 1024 * 1024)  # 初始化多模态嵌入缓存

    # -------- 辅助方法：获取 draft worker 的 KV 池 --------
    # 返回 (draft_token_to_kv_pool, draft_model_config)；无 draft 时返回 (None, None)
    def _get_draft_kv_pool(self):
        """Return (draft_token_to_kv_pool, draft_model_config) for the current
        draft worker, or (None, None) when no draft KV pool is available."""
        if self.draft_worker is None or self.spec_algorithm.is_ngram():
            return None, None  # Ngram 投机不使用 KV Pool，直接跳过

        if self.spec_algorithm.supports_spec_v2() and self.enable_overlap:
            # Spec V2 + overlap 模式：从 draft_runner 获取 KV Pool
            if self.server_args.enable_multi_layer_eagle:
                # 多层 EAGLE：使用第 0 层的 draft_runner
                draft_runner = self.draft_worker.draft_worker.draft_runner_list[0]
            else:
                draft_runner = self.draft_worker.draft_worker.draft_runner
            return draft_runner.token_to_kv_pool, draft_runner.model_config

        # 普通 spec v1 路径：直接从 model_runner 获取
        return (
            self.draft_worker.model_runner.token_to_kv_pool,
            self.draft_worker.model_config,
        )

    # -------- 辅助方法：将 draft KV 池注册到 HiCache 控制器 --------
    def _maybe_register_hicache_draft(self) -> None:
        """Register draft KV pool with HiCacheController for piggyback L2/L3 ops."""
        if not self.enable_hierarchical_cache:
            return  # 未启用层次缓存，跳过

        draft_kv_pool, _ = self._get_draft_kv_pool()
        if draft_kv_pool is None:
            return  # 无 draft KV Pool，跳过

        from sglang.srt.mem_cache.memory_pool import (
            HybridLinearKVPool,  # 混合线性 KV 池（如 Mamba 混合架构）
            MHATokenToKVPool,    # 多头注意力 KV Pool
            MLATokenToKVPool,    # MLA（Multi-head Latent Attention）KV Pool
        )
        from sglang.srt.mem_cache.memory_pool_host import (
            MHATokenToKVPoolHost,  # MHA KV Pool 的主机端（CPU）镜像
            MLATokenToKVPoolHost,  # MLA KV Pool 的主机端（CPU）镜像
        )

        pool = draft_kv_pool
        if isinstance(pool, HybridLinearKVPool):
            # 混合池：取其中的全精度 KV 子池，忽略线性注意力部分
            pool = pool.full_kv_pool

        # 为 draft 创建 host pool，slot 数量与 target host pool 保持 1:1，
        # 确保 target 和 draft 的 host 侧索引一一对应
        primary = self.tree_cache.cache_controller.mem_pool_host  # target 的主机 Pool
        kw = dict(
            host_to_device_ratio=primary.size / pool.size,  # 主机侧与设备侧 slot 比例
            host_size=0,                                     # 初始 host size 为 0（延迟分配）
            page_size=self.page_size,                        # 分页大小
            layout=self.server_args.hicache_mem_layout,      # 内存布局（交错/分离等）
        )
        if isinstance(pool, MHATokenToKVPool):
            draft_host_pool = MHATokenToKVPoolHost(pool, **kw)   # 创建 MHA host pool
        elif isinstance(pool, MLATokenToKVPool):
            draft_host_pool = MLATokenToKVPoolHost(pool, **kw)   # 创建 MLA host pool
        else:
            logger.warning(
                "Draft pool type %s not supported for HiCache, skipping.",
                type(pool).__name__,
            )
            return  # 不支持的 pool 类型，跳过注册

        # 将 draft KV pool 及其主机镜像注册到 HiCache 控制器
        self.tree_cache.cache_controller.set_draft_kv_pool(pool, draft_host_pool)

    # -------- 初始化运行时状态变量 --------
    def init_running_status(self):
        self.waiting_queue: List[Req] = []                    # 等待调度的请求队列（先进先出/优先队列）
        # The running decoding batch for continuous batching
        self.running_batch: ScheduleBatch = ScheduleBatch(reqs=[], batch_is_full=False)
        # 当前正在 decode 的持续批次（continuous batching 核心数据结构）
        # The current forward batch
        self.cur_batch: Optional[ScheduleBatch] = None        # 本轮正在执行的批次
        # The last forward batch
        self.last_batch: Optional[ScheduleBatch] = None       # 上一轮执行的批次（用于 overlap/merge）
        self.forward_ct = 0                                   # 已执行的 forward 轮次计数
        self.return_health_check_ipcs: Deque[Optional[str]] = deque()  # 待回复的健康检查 IPC 地址队列
        self._pending_flush: Optional[Tuple[FlushCacheReqInput, float]] = None  # 延迟 flush cache 请求及截止时间
        self.num_retracted_reqs: int = 0                      # 本轮被回收（retract）的请求数
        self.num_paused_reqs: int = 0                         # 被暂停的请求数
        self.session_controller = SessionController(self.tree_cache)  # 会话控制器（管理多轮对话 session）
        self.forward_sleep_time = None                        # 可选的 forward 人为延迟（调试/测速用）
        self._engine_paused = False                           # 引擎是否处于暂停状态（pause_generation）

    # -------- 初始化分块 prefill（Chunked Prefill）及混合块配置 --------
    def init_chunked_prefill(self):
        self.chunked_prefill_size = self.server_args.chunked_prefill_size  # 每块最大 token 数（None 表示禁用）
        uses_transformers_backend = (
            get_resolved_model_impl(self.model_config) == ModelImpl.TRANSFORMERS
        )  # 判断是否使用 Transformers 后端（非 FlashInfer/Triton）
        if (
            self.chunked_prefill_size is not None
            and self.chunked_prefill_size > 0
            and self.model_config.is_multimodal
            and uses_transformers_backend
        ):
            # 多模态 + Transformers 后端：禁用 chunked prefill，
            # 避免把一个多模态 token 拆分到两个 chunk 中导致形状不匹配
            logger.warning(
                "Chunked prefill is disabled for multimodal models with the "
                "Transformers backend to avoid partial multimodal chunk mismatches."
            )
            self.chunked_prefill_size = None
        elif self.chunked_prefill_size is not None and self.chunked_prefill_size <= 0:
            # chunked_prefill_size <= 0 等价于禁用
            self.chunked_prefill_size = None
        self.chunked_req = None  # 当前正在被分块的请求（跨批次续传）
        self.is_mixed_chunk = (
            self.chunked_prefill_size is not None
            and self.server_args.enable_mixed_chunk
        )  # 混合块模式：prefill chunk + decode 请求在同一批次中执行

        # Init the dynamic chunking predictor for PP
        self.enable_dynamic_chunking = (
            self.server_args.enable_dynamic_chunking and self.pp_size > 1
        )  # 动态块大小预测器（仅 PP > 1 时有意义）
        if self.enable_dynamic_chunking:
            try:
                self.profile_and_init_predictor()  # 对 prefill 延迟做 profiling，拟合预测模型
            except Exception as e:
                logger.warning(
                    f"[PP Dynamic Chunk] Failed to profile prefill latency: {e}. "
                    "Dynamic chunking will be disabled."
                )
                self.enable_dynamic_chunking = False  # profiling 失败则降级为固定块大小

    # -------- 初始化调度策略及 token 比例估算 --------
    def init_schedule_policy(self):
        # Init schedule policy and new token estimation
        self.policy = SchedulePolicy(
            self.schedule_policy,              # 调度策略名称（如 "lpm"、"random"、"fcfs"）
            self.tree_cache,                   # RadixCache / HiCache（用于 cache-aware 排序）
            self.enable_hierarchical_cache,    # 是否启用层次缓存（影响优先级计算）
            self.enable_priority_scheduling,   # 是否启用优先级调度
            self.schedule_low_priority_values_first,  # 小数值 = 高优先级（数值越低越先调度）
        )
        self.prefill_delayer: Optional[PrefillDelayer] = None  # 预填充延迟器（多 DP 场景平衡负载）
        self.max_prefill_bs: int = 0  # 历史最大 prefill batch size（用于预测下一轮可用空间）
        if self.server_args.enable_prefill_delayer:
            if self.server_args.disaggregation_mode == "decode":
                # decode 节点不做 prefill，PrefillDelayer 无意义
                logger.info(
                    "Ignoring --enable-prefill-delayer on decode engine "
                    "(no prefill scheduling path; delayer would be a no-op)."
                )
            else:
                # 创建 PrefillDelayer：监控 KV Cache 使用率，
                # 当使用率低于 watermark 时主动延迟 prefill，
                # 让其他 DP 节点有机会接收新请求，提高全局吞吐
                self.prefill_delayer = PrefillDelayer(
                    dp_size=self.dp_size,
                    attn_tp_size=self.attn_tp_size,
                    cpu_group=self.tp_cpu_group,
                    server_args=self.server_args,
                    metrics_collector=(
                        self.metrics_collector if self.enable_metrics else None
                    ),
                    max_delay_passes=self.server_args.prefill_delayer_max_delay_passes,  # 最大延迟轮次
                    token_usage_low_watermark=self.server_args.prefill_delayer_token_usage_low_watermark,
                    device=(
                        self.tp_group.device                         # 非 overlap：直接用 GPU tensor
                        if self.server_args.disable_overlap_schedule
                        else "cpu"                                    # overlap：用 CPU tensor 减少同步开销
                    ),
                )

        # NOTE: preemption is enabled by default for priority scheduling.
        self.enable_priority_preemption = (
            self.enable_priority_scheduling
            and not self.server_args.disable_priority_preemption
        )  # 启用优先级调度时，允许抢占低优先级请求

        # -------- 新 token 比例自适应调节（动态内存保守系数）--------
        # new_token_ratio 估算每个正在运行的请求还会产生多少 token，
        # 过大 → KV Cache 容易耗尽导致 retract；过小 → 内存浪费
        self.init_new_token_ratio = min(
            envs.SGLANG_INIT_NEW_TOKEN_RATIO.get()
            * self.server_args.schedule_conservativeness,  # 保守系数（用户可调）
            1.0,
        )  # 初始 new_token_ratio，上限为 1.0
        self.min_new_token_ratio = min(
            self.init_new_token_ratio * envs.SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR.get(),
            1.0,
        )  # 最小 new_token_ratio（decode 顺利时逐渐下降到此值）
        self.new_token_ratio_decay = (
            self.init_new_token_ratio - self.min_new_token_ratio
        ) / envs.SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS.get()
        # 每轮 decode 成功后递减的步长（线性衰减，retract 后重置回 init 值）
        self.new_token_ratio = self.init_new_token_ratio  # 当前生效的 new_token_ratio

    # -------- 初始化软性看门狗（超时警告，不强制终止进程）--------
    def init_soft_watchdog(self, server_args: ServerArgs):
        if (x := server_args.soft_watchdog_timeout) is not None:
            # soft=True：超时仅打印警告，不 kill 进程（适合调试慢请求）
            self.soft_watchdog = create_scheduler_watchdog(
                self, watchdog_timeout=x, soft=True
            )

    # -------- 初始化硬性看门狗、内存节省器及输入阻断器 --------
    def init_watch_dog_memory_saver_input_blocker(self):
        # Start watchdog thread
        # 硬性看门狗：超时则强制终止调度进程（防止挂死）
        self.watchdog = create_scheduler_watchdog(
            self, watchdog_timeout=self.server_args.watchdog_timeout
        )

        # Init memory saver, profiler and metric stats
        # TorchMemorySaverAdapter：在内存紧张时自动 offload 张量到 CPU
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.server_args.enable_memory_saver
        )
        self.offload_tags = set()  # 已被 offload 的张量标签集合

        # Init recv skipper and input blocker
        # SchedulerRecvSkipper：overlap 模式下，按策略跳过部分 recv，降低调度开销
        self.recv_skipper = SchedulerRecvSkipper.maybe_create(self.server_args)
        # SchedulerInputBlocker：co-located batch gen 模式下，非 leader rank 屏蔽输入
        self.input_blocker = (
            SchedulerInputBlocker(noop=self.attn_tp_rank != 0)
            if get_bool_env_var("SGLANG_ENABLE_COLOCATED_BATCH_GEN")
            else None
        )

        # Configure GC logger
        if envs.SGLANG_LOG_GC.get():
            configure_gc_logger()  # 记录 Python GC 事件（用于排查 GC 暂停导致的延迟尖刺）

    # -------- 初始化 PD（Prefill-Decode）解耦所需的队列与缓冲区 --------
    def init_disaggregation(self):
        self.disaggregation_mode = DisaggregationMode(
            self.server_args.disaggregation_mode
        )  # 解耦模式：NULL / PREFILL / DECODE
        self.transfer_backend = TransferBackend(
            self.server_args.disaggregation_transfer_backend
        )  # KV 传输后端（RDMA / ZMQ / FAKE 等）

        # todo: should we fix this when enabling mtp or it doesn't matter since we only enable mtp in decode node thus we don't transfer draft kvs between P and D?
        draft_token_to_kv_pool, model_config = self._get_draft_kv_pool()
        # 获取 draft KV pool（若无 draft，则为 None；仅用于 EAGLE 场景的 hidden state 传输）

        if (
            self.disaggregation_mode == DisaggregationMode.DECODE
        ):  # *2 for the headroom.
            # -------- DECODE 节点：分配接收 KV Cache 所需的各种队列 --------
            buffer_size = (self.req_to_token_pool.size) * 2  # metadata buffer 大小（留 2 倍余量）
            self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
                buffer_size
            )  # 为每个请求分配 metadata buffer 下标（RDMA 写入用）
            self.disagg_metadata_buffers = MetadataBuffers(
                buffer_size,
                hidden_size=(
                    model_config.spec_hidden_size   # EAGLE：传输 draft hidden state
                    if self.spec_algorithm.is_eagle()
                    else 16  # minimal padding size for RDMA（非 EAGLE：最小 RDMA 对齐 padding）
                ),
                hidden_states_dtype=(
                    model_config.dtype              # EAGLE：与 draft 模型精度一致
                    if self.spec_algorithm.is_eagle()
                    else torch.float32              # 非 EAGLE：float32（仅占位）
                ),
                custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
            )  # 用于在 P→D KV 传输中承载 metadata（请求 ID、seq len 等）的 pinned 内存池

            # The decode requests polling kv cache
            self.disagg_decode_transfer_queue = DecodeTransferQueue(
                gloo_group=self.attn_tp_cpu_group,  # CPU gloo 通信组（用于 TP 间同步）
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                tp_rank=self.tp_rank,
                metadata_buffers=self.disagg_metadata_buffers,
                scheduler=self,
                tree_cache=self.tree_cache,
            )  # 管理正在等待 KV Cache 传输完成的请求队列（轮询传输进度）

            # The decode requests pending for pre-allocation
            self.disagg_decode_prealloc_queue = DecodePreallocQueue(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                draft_token_to_kv_pool=draft_token_to_kv_pool,  # EAGLE draft KV pool（可为 None）
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                metadata_buffers=self.disagg_metadata_buffers,
                scheduler=self,
                transfer_queue=self.disagg_decode_transfer_queue,
                tree_cache=self.tree_cache,
                gloo_group=self.attn_tp_cpu_group,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                dp_size=self.server_args.dp_size,
                gpu_id=self.gpu_id,
                bootstrap_port=self.server_args.disaggregation_bootstrap_port,  # bootstrap 握手端口
                max_total_num_tokens=self.max_total_num_tokens,
                pp_rank=self.pp_rank,
                num_reserved_decode_tokens=self.server_args.num_reserved_decode_tokens,  # 预留 decode token 槽
                transfer_backend=self.transfer_backend,
            )  # 管理等待 KV Cache 预分配的请求队列

        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            # -------- PREFILL 节点：分配发送 KV Cache 所需的队列 --------
            # *2 for the headroom.
            buffer_size = self.max_running_requests * 2  # prefill 节点并发量更小，以 max_running_requests 为基准
            self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
                buffer_size
            )
            self.disagg_metadata_buffers = MetadataBuffers(
                buffer_size,
                hidden_size=(
                    model_config.spec_hidden_size
                    if self.spec_algorithm.is_eagle()
                    or self.spec_algorithm.is_standalone()
                    else 16  # minimal padding size for RDMA
                ),
                hidden_states_dtype=(
                    model_config.dtype                          # EAGLE / standalone：与 draft 模型精度一致
                    if self.spec_algorithm.is_eagle()
                    or self.spec_algorithm.is_standalone()
                    else torch.float32                          # 否则 float32（仅占位）
                ),
                custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
            )  # prefill 节点 metadata buffer（包含待传输 KV 的 hidden state）

            # BootstrapQueue：与 decode 节点握手，协商 KV 传输通道
            self.disagg_prefill_bootstrap_queue = PrefillBootstrapQueue(
                token_to_kv_pool=self.token_to_kv_pool_allocator.get_kvcache(),   # 主 KV Pool
                draft_token_to_kv_pool=draft_token_to_kv_pool,                   # draft KV Pool（EAGLE）
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                metadata_buffers=self.disagg_metadata_buffers,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                gpu_id=self.gpu_id,
                bootstrap_port=self.server_args.disaggregation_bootstrap_port,   # bootstrap 握手监听端口
                gloo_group=self.attn_tp_cpu_group,
                max_total_num_tokens=self.max_total_num_tokens,
                scheduler=self,
                pp_rank=self.pp_rank,
                pp_size=self.pp_size,
                transfer_backend=self.transfer_backend,
            )
            # The prefill requests that are in the middle of kv sending
            self.disagg_prefill_inflight_queue: List[Req] = []  # 正在进行 KV 传输（in-flight）的 prefill 请求

        # Init mm receiver for EPD disaggregation mode
        # EPD（Encoder-Prefill-Decode）解耦：语言模型节点通过 ZMQ 接收多模态 encoder 输出
        if (
            self.server_args.language_only
            and self.server_args.encoder_transfer_backend == "zmq_to_scheduler"
        ):
            self.mm_receiver = create_mm_receiver(
                self.server_args,
                hf_config=self.model_config.hf_config,
                pp_rank=self.pp_rank,
                tp_rank=self.tp_rank,
                tp_group=self.tp_group,
                scheduler=self,
            )  # 接收来自 encoder 节点的多模态特征（图像/视频嵌入）

    # -------- 初始化 Overlap 调度所需的 CUDA 流和 FutureMap --------
    def init_overlap(self):
        self.device_module = torch.get_device_module(self.device)  # 获取设备特定模块（cuda/mps/cpu）

        if use_mlx():
            # MLX overlap scheduling uses mx.async_eval / mx.eval for
            # synchronisation so no CUDA/MPS streams or FutureMap needed.
            # Apple MLX：overlap 通过 mx.async_eval 实现，无需 CUDA stream / FutureMap
            self.future_map = None
            # Empty result_queue is needed because idle-check references it
            # when enable_overlap is True.
            self.result_queue: Deque = deque()  # idle 检查会引用此队列，需要初始化为空
            return

        # 创建 forward stream 上下文（GPU 模型 forward 在此 stream 上运行）
        self.forward_stream_ctx: CudaStreamContext = self.device_module.stream(
            self.forward_stream
        )
        # copy stream：用于将 GPU 结果异步拷贝到 CPU（与 forward stream 并行）
        self.copy_stream: CudaStream = self.device_module.Stream()
        self.copy_stream_ctx: CudaStreamContext = self.device_module.stream(
            self.copy_stream
        )

        if not self.enable_overlap:
            self.future_map = None  # 非 overlap 模式：不需要 FutureMap
            return

        # FutureMap：管理 GPU → CPU 异步拷贝的"未来结果"（future）句柄
        # overlap 模式下，GPU 还在 forward，CPU 已经开始准备下一批请求
        # FutureMap 让下一轮调度能安全地读取上一轮的 GPU 输出（next_token_ids）
        self.future_map = FutureMap(
            self.max_running_requests,       # 最大并发请求数（决定 future slot 数量）
            self.chunked_prefill_size,       # chunked prefill 大小（影响 future 生命周期）
            self.model_config.context_len,   # 最大上下文长度
            self.device,
            self.spec_algorithm,
        )
        self.batch_record_buf = [None] * 2   # 双缓冲：防止 overlap 期间 GPU tensor 被 GC 提前释放
        self.batch_record_ct = 0             # 双缓冲当前写入位置（0 或 1）

    # -------- 初始化 Ngram 嵌入投机解码 --------
    def maybe_init_ngram_embedding(self):
        self.use_ngram_embedding = self.tp_worker.model_config.use_ngram_embedding  # 是否使用 ngram 嵌入
        if self.use_ngram_embedding:
            self.token_table = self.tp_worker.model_runner.token_table  # 全词表 token 嵌入表
            hf_config = self.tp_worker.model_config.hf_config
            self.ngram_embedding_n = hf_config.ngram_embedding_n  # ngram 的 n（上下文窗口）
            self.ngram_embedding_k = hf_config.ngram_embedding_k  # ngram 的 k（候选数量）

    # -------- 在 forward 前为 Ngram 嵌入模型填充 token 表 --------
    def _maybe_prepare_ngram_embedding(
        self, batch: Optional[ScheduleBatch]
    ) -> Optional[ScheduleBatch]:
        """Fill the token table for ngram embedding before a forward pass."""
        if batch is None or not self.use_ngram_embedding:
            return batch  # 非 ngram 嵌入模型，直接返回
        batch.ne_token_table = self.token_table  # 挂载 token 表引用
        if batch.forward_mode == ForwardMode.EXTEND:
            # EXTEND（prefill）时，需要更新 token 表中每条请求的 n-gram 上下文
            all_tokens = []       # 所有请求的拼接 token 序列
            column_starts = []    # 每条请求在 token_table 中的起始列（用于写入位置）
            request_lengths = []  # 每条请求实际写入的 token 数量
            for req in batch.reqs:
                start = len(req.prefix_indices)              # KV 命中长度（已缓存，不需重新填）
                end = start + req.extend_input_len           # 本次 extend 的结束位置
                fill_ids = req.origin_input_ids + req.output_ids
                if start == 0:
                    # 无前缀命中：从头开始写入
                    tokens = fill_ids[start:end]
                    column_starts.append(0)
                elif start < self.ngram_embedding_n:
                    # 前缀长度不足 n：从 0 开始写入（保证有足够上下文）
                    tokens = fill_ids[0:end]
                    column_starts.append(0)
                else:
                    # Prepend n-1 tokens before prefix_len for n-gram context
                    # 前缀足够长：在 start 前补 n-1 个 token 作为 n-gram 上下文
                    tokens = fill_ids[start - self.ngram_embedding_n + 1 : end]
                    column_starts.append(start - self.ngram_embedding_n + 1)
                all_tokens.extend(tokens)
                request_lengths.append(len(tokens))
            dtype = self.token_table.dtype
            device = self.token_table.device
            # 批量更新 token 表（GPU kernel，避免逐条写入的开销）
            update_token_table(
                ne_token_table=self.token_table,
                tokens=torch.tensor(all_tokens, dtype=dtype, device=device),
                row_indices=batch.req_pool_indices,   # 各请求在 token_table 中的行索引
                column_starts=torch.tensor(
                    column_starts, dtype=torch.int32, device=device
                ),
                req_lens=torch.tensor(
                    request_lengths, dtype=torch.int32, device=device
                ),
                ignore_tokens=None,
            )
        return batch

    # -------- 初始化确定性推理截断对齐大小 --------
    def init_deterministic_inference_config(self):
        """Initialize deterministic inference configuration for different attention backends."""
        if not self.server_args.enable_deterministic_inference:
            self.truncation_align_size = None  # 未启用确定性推理：不截断
            return

        # 不同 attention 后端使用不同的对齐大小环境变量
        backend_sizes = {
            "flashinfer": ("SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE", 4096),
            "triton": ("SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE", 4096),
        }
        env_var, default_size = backend_sizes.get(
            self.server_args.attention_backend, (None, None)
        )
        # truncation_align_size：将 prefill token 数截断对齐到该大小（保证不同 batch 结果一致）
        self.truncation_align_size = (
            get_int_env_var(env_var, default_size) if env_var else None
        )

    # -------- 初始化请求类型分发器（TypeBasedDispatcher） --------
    # 根据请求对象类型，路由到对应的 handle_* 方法
    def init_request_dispatcher(self):
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.handle_generate_request),           # 单条生成请求
                (TokenizedEmbeddingReqInput, self.handle_embedding_request),         # 单条嵌入请求
                (BatchTokenizedGenerateReqInput, self.handle_batch_generate_request), # 批量生成请求

                (BatchTokenizedEmbeddingReqInput, self.handle_batch_embedding_request),  # 批量嵌入请求
                (FlushCacheReqInput, self.flush_cache_wrapped),                         # 清空 KV Cache
                (ClearHiCacheReqInput, self.clear_hicache_storage_wrapped),             # 清空层次缓存存储
                (AttachHiCacheStorageReqInput, self.attach_hicache_storage_wrapped),    # 挂载层次缓存存储后端
                (DetachHiCacheStorageReqInput, self.detach_hicache_storage_wrapped),    # 卸载层次缓存存储后端
                (AbortReq, self.abort_request),                                         # 中止指定请求
                (OpenSessionReqInput, self.open_session),                               # 打开多轮对话 session
                (CloseSessionReqInput, self.close_session),                             # 关闭多轮对话 session
                (UpdateWeightFromDiskReqInput, self.update_weights_from_disk),          # 从磁盘加载新权重
                (InitWeightsUpdateGroupReqInput, self.init_weights_update_group),       # 初始化权重更新通信组
                (DestroyWeightsUpdateGroupReqInput, self.destroy_weights_update_group), # 销毁权重更新通信组
                (
                    InitWeightsSendGroupForRemoteInstanceReqInput,
                    self.init_weights_send_group_for_remote_instance,  # 初始化跨实例权重发送组
                ),
                (
                    SendWeightsToRemoteInstanceReqInput,
                    self.send_weights_to_remote_instance,               # 将权重发送到远端实例
                ),
                (
                    UpdateWeightsFromDistributedReqInput,
                    self.update_weights_from_distributed,               # 从分布式源更新权重（RLHF 在线训练）
                ),
                (UpdateWeightsFromTensorReqInput, self.update_weights_from_tensor),     # 从 tensor 更新权重
                (UpdateWeightsFromIPCReqInput, self.update_weights_from_ipc),           # 从 IPC 共享内存更新权重
                (GetWeightsByNameReqInput, self.get_weights_by_name),                   # 按名称获取权重 tensor
                (ReleaseMemoryOccupationReqInput, self.release_memory_occupation),      # 释放显存占用（休眠模式）
                (ResumeMemoryOccupationReqInput, self.resume_memory_occupation),        # 恢复显存占用
                (CheckWeightsReqInput, self.check_weights),                             # 校验权重正确性
                (SlowDownReqInput, self.slow_down),                                     # 人为减速（调试/测速）
                (ProfileReq, self.profile),                                             # 启动/停止性能 profiling
                (FreezeGCReq, self.handle_freeze_gc),                                   # 冻结 Python GC
                (GetInternalStateReq, self.get_internal_state),                         # 获取调度器内部状态
                (SetInternalStateReq, self.set_internal_state),                         # 设置调度器内部状态
                (RpcReqInput, self.handle_rpc_request),                                 # 通用 RPC 请求
                (ExpertDistributionReq, self.expert_distribution_handle),               # MoE 专家分布记录
                (LoadLoRAAdapterReqInput, self.load_lora_adapter),                      # 加载 LoRA 适配器
                (
                    LoadLoRAAdapterFromTensorsReqInput,
                    self.load_lora_adapter_from_tensors,                               # 从 tensor 加载 LoRA
                ),
                (UnloadLoRAAdapterReqInput, self.unload_lora_adapter),                  # 卸载 LoRA 适配器
                (GetLoadsReqInput, self.get_loads),                                     # 获取负载信息
                (PauseGenerationReqInput, self.pause_generation),                       # 暂停生成
                (ContinueGenerationReqInput, self.continue_generation),                 # 恢复生成
                (DumperControlReqInput, self.handle_dumper_control),                    # 调试 dumper 控制
                (AddExternalCorpusReqInput, self.add_external_corpus),                  # 添加外部语料（ngram 投机）
                (
                    RemoveExternalCorpusReqInput,
                    self.remove_external_corpus,                                        # 删除外部语料
                ),
                (
                    ListExternalCorporaReqInput,
                    self.list_external_corpora,                                         # 列出外部语料
                ),
            ]
        )

    # -------- 检查 running_batch 中超时运行的请求，将其标记为 abort --------
    def _abort_on_running_timeout(self):
        # NOTE: this should be called before a batch is launched,
        # as current spec-v1 still filters batch inside verify stage.
        # 注意：必须在 batch 启动前调用，spec-v1 仍在 verify 阶段内过滤 batch
        timeout_s = envs.SGLANG_REQ_RUNNING_TIMEOUT.get()  # 运行超时阈值（秒），0 表示禁用
        if timeout_s <= 0:
            return  # 超时检查未启用
        if self.running_batch.is_empty():
            return  # running_batch 为空，无需检查

        deadline = time.perf_counter() - timeout_s  # 超时截止时间点（早于此时进入 forward 的请求超时）
        for req in self.running_batch.reqs:
            if not req.finished() and 0 < req.time_stats.forward_entry_time < deadline:
                # 请求仍在运行且首次进入 forward 的时间早于截止时间 → 超时，标记为 abort
                req.to_finish = FINISH_ABORT(
                    "Request running timeout reached.", HTTPStatus.SERVICE_UNAVAILABLE
                )

    # -------- 返回调度器初始化信息（供 TokenizerManager 握手验证）--------
    def get_init_info(self) -> Dict[str, Any]:
        """Return scheduler initialization info for handshake.

        This method provides the initialization info needed by the tokenizer manager
        and other components to verify the scheduler is ready.
        """
        result_dict = {
            "status": "ready",                              # 调度器已就绪
            "max_total_num_tokens": self.max_total_num_tokens,  # KV Cache 总 token 容量
            "max_req_input_len": self.max_req_input_len,    # 单条请求最大输入长度
        }

        return result_dict

    # -------- 启动事件循环入口（设置 schedule_stream 并分发到具体 event_loop）--------
    def run_event_loop(self) -> None:
        """Run the scheduler's event loop.

        Sets up the schedule stream and dispatches to the appropriate event loop.
        The event loop blocks until shutdown.
        """
        if use_mlx():
            # MLX overlap uses mx.async_eval for CPU/GPU overlap,
            # not PyTorch MPS streams.
            # Apple MLX：直接分发，无需 CUDA stream
            dispatch_event_loop(self)
            return

        # 创建调度流（schedule_stream），优先级为 0（高于默认 forward stream）
        self.schedule_stream = self.device_module.Stream(priority=0)
        if self.device == "cpu":
            self.schedule_stream.synchronize = lambda: None  # No-op for CPU（CPU 无 stream 同步）
        with self.device_module.StreamContext(self.schedule_stream):
            dispatch_event_loop(self)  # 在 schedule_stream 上下文中运行事件循环（阻塞直到关闭）

    @DynamicGradMode()
    def event_loop_normal(self):
        """A normal scheduler loop."""
        # 【学习注释 ③】Scheduler 子进程的主循环（非 overlap 版本）
        # 每次迭代 = 一个批次：收请求 → 调度 → GPU 执行 → 处理结果
        # 全程在同一个 CPU 线程里顺序执行，GPU 执行期间 CPU 阻塞等待
        while True:
            # Receive requests
            recv_reqs = self.recv_requests()
            # ↑ 见 recv_requests()：非阻塞拉取所有新到达的请求
            self.process_input_requests(recv_reqs)
            # ↑ 把收到的请求解析成 Req 对象，放入 self.waiting_queue
            if self._engine_paused:
                self.cancel_bubble_timer()
                continue

            # Get the next batch to run
            batch = self.get_next_batch_to_run()
            # ↑ 核心调度决策：从 waiting_queue + running_batch 凑一批
            #   返回 ScheduleBatch 或 None（无任务时）
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                result = self.run_batch(batch)
                # ↑ 转换数据结构 → 送 GPU → 等待结果（同步阻塞）
                self.process_batch_result(batch, result)
                # ↑ 处理输出：追加 token、检查 finish、更新 RadixCache、发给 Detokenizer
            else:
                # When the server is idle, do self-check and re-init some states.
                self.on_idle()
                # ↑ 无任务：做 GC 清理、指标上报、RadixCache 内存检查等

            # Update last_batch
            self.last_batch = batch
            # ↑ 保存本轮批次供下一轮使用
            #   get_next_batch_to_run 需要知道上轮是 EXTEND 还是 DECODE
            #   以决定是否把刚完成 prefill 的请求合并进 running_batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.self_check_during_busy()

    @DynamicGradMode()
    # -------- Overlap 调度事件循环：CPU 调度与 GPU 计算流水重叠 --------
    def event_loop_overlap(self):
        """A scheduler loop that overlaps the CPU processing and GPU computation."""
        # 核心思路：
        #   第 N 轮：GPU 执行 batch_N，同时 CPU 处理 batch_{N-1} 的输出（process_batch_result）
        #   这样 CPU 处理开销被 GPU 计算时间"遮蔽"，从而提升吞吐量
        #
        #   result_queue 最多存 1 个元素：(上一批 ScheduleBatch 快照, 上一批 batch_result)
        #   batch_result 里的 next_token_ids 是 GPU tensor，通过 FutureMap 延迟拷贝到 CPU
        self.result_queue: Deque[
            Tuple[ScheduleBatch, Union[GenerationBatchResult, EmbeddingBatchResult]]
        ] = deque()

        def pop_and_process():
            # Process the results of the last batch
            # 取出上一批结果并处理（追加 token、更新 RadixCache、发给 Detokenizer）
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        while True:
            # Receive requests
            recv_reqs = self.recv_requests()     # 非阻塞从 ZMQ 拉取新请求
            self.process_input_requests(recv_reqs)  # 解析并放入 waiting_queue
            if self._engine_paused:
                continue  # 引擎已暂停：跳过调度，等待 continue_generation

            # Get the next batch to run
            batch = self.get_next_batch_to_run()  # 调度决策：返回本轮要跑的批次
            self.cur_batch = batch
            disable_overlap_for_batch = self.is_disable_overlap_for_batch(batch)
            # 连续两批 prefill 时，禁用 overlap 以改善 TTFT；spec+grammar 场景也禁用

            # If we do not need to overlap the current batch with the last batch,
            # we can process the last batch immediately.
            # 当前批次禁用 overlap：先同步处理上一批结果，再启动当前批次
            if disable_overlap_for_batch:
                pop_and_process()

            # Launch the current batch
            if batch:
                batch_result = self.run_batch(batch)  # GPU forward（在 forward_stream 上异步执行）
                self.result_queue.append((batch.copy(), batch_result))
                # batch.copy()：快照当前批次状态，防止 overlap 调度修改原始数据
            else:
                batch_result = None
                self.cancel_bubble_timer()  # 无任务：取消气泡计时器

            # Process the last batch
            # 若 overlap 未被禁用，在 GPU 跑当前批次的同时处理上一批结果
            if self.last_batch:
                if not disable_overlap_for_batch:
                    pop_and_process()  # CPU 处理上一批 ↔ GPU 执行当前批（真正的 overlap）
            elif batch is None:
                # When the server is idle, do self-check and re-init some states
                self.on_idle()  # 完全空闲：做 GC 清理、指标上报等

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            # 对当前批次的采样（需要上一批 grammar 状态），必须在上一批处理完后才能执行
            if self.is_generation:
                self.launch_batch_sample_if_needed(batch_result)

            # Update last_batch
            self.last_batch = batch  # 保存本轮批次，供下一轮 overlap 使用

            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.self_check_during_busy()  # 严格内存检查（调试用）

    # -------- 判断当前批次是否需要禁用 overlap --------
    def is_disable_overlap_for_batch(self, batch: ScheduleBatch) -> bool:
        # For two consecutive prefill batches, we disable overlap to improve the TTFT of the first batch.
        # This might slightly hurt the throughput, so we use an environment variable to control it.
        # In DP attention mode, use the globally synchronized is_extend_in_batch
        # so all DP ranks make the same overlap decision (avoiding deadlock).
        # In non-DP mode, use the local forward_mode directly.
        # DP Attention 模式：使用全局同步的 is_extend_in_batch，确保所有 DP rank 做出相同决策
        if self.require_mlp_sync:
            is_extend = lambda b: b and b.is_extend_in_batch  # DP 模式：使用全局同步的 extend 标志
        else:
            is_extend = lambda b: b and b.forward_mode.is_extend()  # 非 DP 模式：直接用本地 forward_mode

        batch_is_extend = is_extend(batch)            # 当前批次是否为 EXTEND（prefill）
        last_batch_is_extend = is_extend(self.last_batch)  # 上一批次是否为 EXTEND

        # 连续两批都是 prefill：禁用 overlap（环境变量控制，可牺牲一点吞吐换更低 TTFT）
        disable_overlap_for_batch = (
            envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.get()
            and batch_is_extend
            and last_batch_is_extend
        )

        # We do not support overlap + spec + grammar yet,
        # so we need to turn off overlap for this batch.
        # TODO(lsyin): support overlap + spec + grammar
        # spec v2 + 结构化输出（grammar）在 decode 时需要同步上一批 grammar 状态，
        # 无法 overlap，必须顺序处理
        need_grammar_sync = (
            batch
            and batch.is_spec_v2
            and batch.has_grammar
            and batch.forward_mode.is_decode()
            and len(self.result_queue) > 0  # result_queue 有待处理结果时才需要同步
        )

        return disable_overlap_for_batch or need_grammar_sync

    def recv_limit_reached(self, num_recv_reqs: int) -> bool:
        if self.max_recv_per_poll < 0:
            return False
        # 当已接收请求数达到限制时返回 True
        return num_recv_reqs >= self.max_recv_per_poll

    def recv_requests(
        self,
    ) -> List[Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput, Any]]:
        """Receive results at tp_rank = 0 and broadcast it to all other TP ranks."""
        # 【学习注释 ②】非阻塞地从 ZMQ 队列取出所有待处理请求
        # 这是"批量"形成的关键：并发的 HTTP 请求都排在同一个 ZMQ 队列里
        # 一次性清空队列，本轮循环就自然拿到了多条并发请求

        if self.recv_skipper is not None:
            last_forward_mode = (
                self.last_batch.forward_mode if self.last_batch is not None else None
            )
            if not self.recv_skipper.handle(last_forward_mode):
                return []
            # overlap 模式下按需跳过收包，减少调度开销

        if self.pp_rank == 0:
            if self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
                recv_reqs = []

                while True:
                    try:
                        if self.recv_limit_reached(len(recv_reqs)):
                            break
                            # ↑ 防止单轮收包过多导致调度延迟（有上限）
                        recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                        # ↑ zmq.NOBLOCK：非阻塞！有消息就取，没消息立刻抛 ZMQError
                        #   取出的对象是 TokenizedGenerateReqInput（已完成 tokenize）
                    except zmq.ZMQError:
                        break  # ↑ 队列空了，退出循环
                    recv_reqs.append(recv_req)

                while True:
                    try:
                        if self.recv_limit_reached(len(recv_reqs)):
                            break
                        recv_rpc = self.recv_from_rpc.recv_pyobj(zmq.NOBLOCK)
                        # 接收来自 RPC 的额外请求
                    except zmq.ZMQError:
                        break  # RPC 队列已空
                    recv_reqs.append(recv_rpc)
            else:
                recv_reqs = None
        else:
            # Pipeline Parallel 非首 stage：从前一个 stage 点对点接收
            if self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
                dp_offset = self.attn_dp_rank * self.attn_tp_size
                recv_reqs = point_to_point_pyobj(
                    [],
                    self.pp_rank * self.tp_size + dp_offset,
                    self.world_group.cpu_group,
                    (self.pp_rank - 1) * self.tp_size + dp_offset,
                    self.pp_rank * self.tp_size + dp_offset,
                )
            else:
                recv_reqs = None  # 非 leader rank 不直接参与通信

        if self.input_blocker is not None:
            # 输入阻断器：用于可控地暂停或过滤请求
            recv_reqs = self.input_blocker.handle(recv_reqs)

        if self.server_args.enable_dp_attention:
            # DP Attention 模式：拆分工作请求与控制请求，分别广播
            if self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
                work_reqs, control_reqs = self._split_work_and_control_reqs(recv_reqs)
            else:
                work_reqs = None
                control_reqs = None

            # 在 attn_tp_group 内广播工作请求
            if self.attn_tp_size != 1:
                work_reqs = broadcast_pyobj(
                    work_reqs,
                    self.attn_tp_group.rank,
                    self.attn_tp_cpu_group,
                    src=self.attn_tp_group.ranks[0],
                )

            # 在 attn_cp_group 内广播工作请求
            if self.attn_cp_size != 1:
                work_reqs = broadcast_pyobj(
                    work_reqs,
                    self.attn_cp_group.rank,
                    self.attn_cp_cpu_group,
                    src=self.attn_cp_group.ranks[0],
                )

            # 控制请求广播策略：
            # 启用 local_control_broadcast 时，仅在 attn_tp + attn_cp 组内广播，
            # 避免全量 tp_group 的昂贵 gloo 同步。
            _local_ctrl = self.server_args.enable_dp_attention_local_control_broadcast
            if _local_ctrl:
                if self.attn_tp_size != 1:
                    control_reqs = broadcast_pyobj(
                        control_reqs,
                        self.attn_tp_group.rank,
                        self.attn_tp_cpu_group,
                        src=self.attn_tp_group.ranks[0],
                    )
                if self.attn_cp_size != 1:
                    control_reqs = broadcast_pyobj(
                        control_reqs,
                        self.attn_cp_group.rank,
                        self.attn_cp_cpu_group,
                        src=self.attn_cp_group.ranks[0],
                    )
            elif self.tp_size != 1:
                # 未启用 local control 时，在完整 tp_group 内广播控制请求
                control_reqs = broadcast_pyobj(
                    control_reqs,
                    self.tp_group.rank,
                    self.tp_cpu_group,
                    src=self.tp_group.ranks[0],
                )
            recv_reqs = work_reqs + control_reqs
        elif self.tp_size != 1:
            # 非 DP Attention 模式：在完整 tp_group 内广播所有请求
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.tp_group.rank,
                self.tp_cpu_group,
                src=self.tp_group.ranks[0],
            )

        # EPD-disaggregation 模式下处理多模态请求
        if (
            self.pp_rank == 0
            and self.server_args.language_only
            and self.server_args.encoder_transfer_backend == "zmq_to_scheduler"
        ):
            recv_reqs, abort_reqs = self.mm_receiver.process_waiting_requests(recv_reqs)
            # 处理需要中止的请求，返回错误并输出流结果
            for req, error_msg, error_code in abort_reqs:
                status_code = (
                    HTTPStatus.BAD_REQUEST
                    if error_code == 400
                    else HTTPStatus.INTERNAL_SERVER_ERROR
                )
                prepare_abort(req, error_msg, status_code=status_code)
                self.stream_output([req], req.return_logprob)

        # 所有广播完成后解包共享内存数据，
        # 使得广播过程中只序列化 ShmPointerMMData 元数据而非完整张量。
        if recv_reqs:
            # 非 DP Attention 路径需要 barrier：
            # source rank 在 broadcast_pyobj 后立即返回，而其他 rank 仍在
            # pickle.loads -> __setstate__ -> shm_open 过程中。
            # 若无 barrier，source 可能先调用 materialize/shm_unlink，
            # 导致其他 rank 无法打开共享内存段。
            # DP Attention 路径不需要 barrier：
            # control_reqs 在 tp_cpu_group 上的广播（step 3）是集体操作，
            # 会强制所有 rank 先完成前面 work_reqs 的反序列化（含 shm_open）。
            if (
                not self.server_args.enable_dp_attention
                and self.tp_size > 1
                and self.model_config.is_multimodal
                and has_shm_features(recv_reqs)
            ):
                barrier(group=self.tp_cpu_group)
            for req in recv_reqs:
                unwrap_shm_features(req)

        return recv_reqs

    # -------- 将收到的请求拆分为工作请求（generate/embedding）和控制请求 --------
    # DP Attention 模式下，工作请求和控制请求走不同的广播路径
    def _split_work_and_control_reqs(self, recv_reqs: List):
        # 工作请求：生成/嵌入类（需要调度和 GPU 执行）
        work_reqs = [
            req
            for req in recv_reqs
            if isinstance(
                req,
                (
                    TokenizedGenerateReqInput,
                    TokenizedEmbeddingReqInput,
                    BatchTokenizedGenerateReqInput,
                    BatchTokenizedEmbeddingReqInput,
                ),
            )
        ]
        # 控制请求：除工作请求外的所有请求（flush、abort、update_weights 等管理命令）
        control_reqs = [
            req
            for req in recv_reqs
            if not isinstance(
                req,
                (
                    TokenizedGenerateReqInput,
                    TokenizedEmbeddingReqInput,
                    BatchTokenizedGenerateReqInput,
                    BatchTokenizedEmbeddingReqInput,
                ),
            )
        ]
        return work_reqs, control_reqs

    # -------- 处理一批输入请求：分发给对应 handle_* 方法 --------
    def process_input_requests(self, recv_reqs: List):
        now = time.monotonic()
        self.session_controller.maybe_reap(now)  # 清理过期 session
        for recv_req in recv_reqs:
            # Skip health check when server is busy — ongoing requests already carry health info.
            # 服务繁忙时跳过健康检查（延迟到有空闲时再回复，避免阻塞生成请求）
            if is_health_check_generate_req(recv_req) and not self.is_fully_idle(
                for_health_check=True
            ):
                self.return_health_check_ipcs.append(
                    getattr(recv_req, "http_worker_ipc", None)
                )
                continue

            output = self._request_dispatcher(recv_req)  # 根据类型路由到对应 handle_* 方法
            if output is not None:
                if not isinstance(output, RpcReqOutput):
                    # 普通输出：发回给 TokenizerManager（再路由到对应 HTTP worker）
                    self.send_to_tokenizer.send_output(output, recv_req)
                else:
                    # RPC 输出：发回给 RPC 调用方
                    if self.recv_from_rpc is not None:
                        self.recv_from_rpc.send_pyobj(output)

        self._check_pending_flush()  # 检查是否有待执行的延迟 flush_cache
        if self.external_corpus_manager is not None:
            self.external_corpus_manager.check_pending_load()  # 检查外部语料库是否加载完成

    # -------- 初始化请求的最大新 token 数（考虑上下文长度限制）--------
    def init_req_max_new_tokens(self, req):
        req.sampling_params.max_new_tokens = min(
            (
                req.sampling_params.max_new_tokens  # 用户设置的 max_new_tokens
                if req.sampling_params.max_new_tokens is not None
                else 1 << 30  # 未设置：使用极大值（约 10 亿）
            ),
            self.max_req_len - len(req.origin_input_ids) - 1,  # 不超过上下文长度限制
        )

    # -------- 在主（entry）rank 上处理多模态输入并广播给其他 TP rank --------
    # 避免每个 TP rank 都重复执行 CPU 密集型的 image processor，降低 CUDA kernel 启动延迟
    def _process_and_broadcast_mm_inputs(
        self,
        raw_mm_inputs,
    ):
        """Materialize MultimodalInputs once on the entry rank and broadcast to others.

        Entry rank:
        - constructs MultimodalInputs.from_processor_output() once
        - broadcasts to other ranks in self.cpu_group (if world_size > 1)

        Non-entry ranks:
        - receive the object via broadcast (if world_size > 1)
        - otherwise (single-rank / no group) fall back to local from_processor_output

        Returns:
            MultimodalInputs | None
        """
        if raw_mm_inputs is None:
            return None

        group_world_size = 1
        try:
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and self.dp_tp_cpu_group is not None
            ):
                group_world_size = torch.distributed.get_world_size(
                    group=self.dp_tp_cpu_group
                )
        except Exception as e:
            logger.warning(
                f"Failed to get world size in mm_inputs handling with {e}, fallback to 1."
            )

        # In case tp size > 1, all the Scheduler TP ranks runs the duplicated computing
        # process in CPU which occupies the main thread CPU cycle. This computing logic
        # merely needs to be run on TP0 and be broadcast to other TP ranks.
        # Since the Scheduler is single-threaded, any large CPU cost will impact
        # handling of other messages. For example, CPU hits 99.9% can significantly
        # increase the CUDA kernel launch time.
        if self.dp_tp_group.rank_in_group == 0:
            # Only the entry rank materializes once from dict.
            image_inputs = MultimodalInputs.from_processor_output(raw_mm_inputs)
            # Broadcast to other TP ranks (use src=0 within the group).
            if group_world_size > 1:
                obj_list = [image_inputs]
                torch.distributed.broadcast_object_list(
                    obj_list,
                    src=self.dp_tp_group.first_rank,
                    group=self.dp_tp_cpu_group,
                )
                image_inputs = obj_list[0]
        else:
            # Non-entry ranks: receive if group size > 1; otherwise materialize locally.
            if group_world_size > 1:
                obj_list = [None]
                torch.distributed.broadcast_object_list(
                    obj_list,
                    src=self.dp_tp_group.first_rank,
                    group=self.dp_tp_cpu_group,
                )
                image_inputs = obj_list[0]
            else:
                image_inputs = MultimodalInputs.from_processor_output(raw_mm_inputs)

        return image_inputs

    # -------- 获取多模态输入（可选择是否广播以节省 CPU 开销）--------
    def _get_multimodal_inputs(self, mm_inputs_dict):
        if self.server_args.enable_broadcast_mm_inputs_process:
            # 只在 entry rank 处理，再广播到所有 TP rank（节省 CPU）
            return self._process_and_broadcast_mm_inputs(mm_inputs_dict)
        else:
            # 每个 rank 独立处理（兼容旧行为）
            return MultimodalInputs.from_processor_output(mm_inputs_dict)

    # -------- 补全 M-RoPE（多模态旋转位置编码）位置信息 --------
    def _maybe_compute_mrope_positions(self, req) -> None:
        """Compute M-RoPE positions when they are missing (e.g. gRPC preprocessed path)."""
        if self._mm_processor is None:
            return  # 模型不使用 M-RoPE
        mm = req.multimodal_inputs
        if mm is None or mm.mrope_positions is not None:
            return  # 无多模态输入，或位置信息已经计算过

        # 计算 M-RoPE 位置（如 Qwen-VL 的 3D-RoPE：time/height/width 三维位置）
        mrope_positions, mrope_position_delta = (
            self._mm_processor.compute_mrope_positions(
                req.origin_input_ids, mm.mm_items
            )
        )
        if mrope_positions is not None:
            mm.mrope_positions = mrope_positions                   # 各 token 的 3D RoPE 位置
            mm.mrope_position_delta = mrope_position_delta         # 相对位置偏移（用于增量解码）

    # -------- 请求完成后释放多模态特征缓存（节省显存）--------
    def _maybe_clear_mm_inputs(self, batch: ScheduleBatch) -> None:
        for req in batch.reqs:
            if not req.finished() or not (mm_inputs := req.multimodal_inputs):
                continue  # 未完成或无多模态输入，跳过
            # For session requests, keep mm_inputs for the next request
            if req.session:
                continue  # session 请求：保留多模态输入供后续轮次复用
            # For non-session requests, clear features and mm_inputs
            mm_inputs.release_features()  # 释放 GPU 上的特征张量（如 visual tokens）
            req.multimodal_inputs = None  # 置空，允许 GC 回收

    # -------- 处理单条文本生成请求 --------
    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
    ):
        # Route: normal request / session request / session-not-found
        # 路由分三种：普通请求 / session 续接请求 / session 不存在（直接 abort）
        session_id = (
            recv_req.session_params.id if recv_req.session_params is not None else None
        )

        if session_id is None:
            # Normal non-session request（普通非 session 请求）
            if recv_req.input_embeds is not None:
                # Generate fake input_ids based on the length of input_embeds
                # 使用自定义 embedding（如 input_embeds 直接注入）：生成等长的假 input_ids
                seq_length = len(recv_req.input_embeds)
                fake_input_ids = [1] * seq_length  # 占位符 ID（实际不会被模型 embedding 层使用）
                recv_req.input_ids = fake_input_ids

            if recv_req.bootstrap_port is None:
                # Use default bootstrap port
                recv_req.bootstrap_port = self.server_args.disaggregation_bootstrap_port

            # 构造调度器内部的 Req 对象（从 TokenizedGenerateReqInput 转换）
            req = Req(
                recv_req.rid,                              # 请求 ID
                recv_req.input_text,                       # 原始文本（用于日志/debug）
                recv_req.input_ids,                        # 已 tokenize 的 token ID 列表
                recv_req.sampling_params,                  # 采样参数（temperature, top_p 等）
                return_logprob=recv_req.return_logprob,    # 是否返回 logprob
                top_logprobs_num=recv_req.top_logprobs_num,  # 返回 top-k logprob 的 k 值
                token_ids_logprob=recv_req.token_ids_logprob,  # 指定 token 的 logprob
                stream=recv_req.stream,                    # 是否流式返回
                lora_id=recv_req.lora_id,                  # LoRA 适配器 ID（None 表示基础模型）
                input_embeds=recv_req.input_embeds,        # 自定义输入嵌入（可选）
                positional_embed_overrides=recv_req.positional_embed_overrides,  # 覆盖位置嵌入
                token_type_ids=recv_req.token_type_ids,    # token 类型 ID（BERT 风格，可选）
                custom_logit_processor=recv_req.custom_logit_processor,  # 自定义 logit 处理器
                require_reasoning=recv_req.require_reasoning,  # 是否要求 reasoning（思维链）
                return_hidden_states=recv_req.return_hidden_states,  # 是否返回隐藏状态
                return_routed_experts=recv_req.return_routed_experts,  # 是否返回 MoE 路由专家信息
                eos_token_ids=self.model_config.hf_eos_token_id,  # EOS token 集合
                bootstrap_host=recv_req.bootstrap_host,    # PD 解耦 bootstrap 主机地址
                bootstrap_port=recv_req.bootstrap_port,    # PD 解耦 bootstrap 端口
                bootstrap_room=recv_req.bootstrap_room,    # PD 解耦 bootstrap room ID（唯一标识一次 P→D 传输）
                disagg_mode=self.disaggregation_mode,      # 当前节点的解耦模式
                routed_dp_rank=recv_req.routed_dp_rank,    # 路由到的 DP rank（负载均衡用）
                disagg_prefill_dp_rank=recv_req.disagg_prefill_dp_rank,  # prefill 端 DP rank
                vocab_size=self.model_config.vocab_size,   # 词表大小（用于校验）
                priority=recv_req.priority,                # 请求优先级（优先级调度用）
                metrics_collector=(
                    self.metrics_collector if self.enable_metrics else None  # 指标收集器
                ),
                routing_key=recv_req.routing_key,          # 路由键（用于 prefix-aware 路由）
                extra_key=recv_req.extra_key,              # 额外路由键
                http_worker_ipc=recv_req.http_worker_ipc,  # HTTP worker IPC 地址（多 worker 场景）
                dllm_config=self.dllm_config,              # dLLM（分布式 LLM）配置
                time_stats=recv_req.time_stats,            # 请求全链路时间统计
                multi_item_delimiter_indices=recv_req.multi_item_delimiter_indices,  # 多item分隔符位置
            )
            req.tokenizer = self.tokenizer  # 绑定 tokenizer（用于 streaming decode 时 detokenize）

            if self.disaggregation_mode != DisaggregationMode.NULL:
                # Invalid request for disaggregated mode
                # PD 解耦模式：请求必须携带 bootstrap_room（标识 P→D 传输通道）
                if (
                    recv_req.bootstrap_room is None
                    and self.transfer_backend != TransferBackend.FAKE  # FAKE 后端（测试用）可跳过
                ):
                    error_msg = (
                        f"Invalid request: Disaggregated request received without "
                        f"bootstrap room id. {req.rid=}"
                    )
                    logger.error(error_msg)
                    recv_req.time_stats.trace_ctx.abort(
                        abort_info={"reason": error_msg}
                    )
                    prepare_abort(req, error_msg, status_code=HTTPStatus.BAD_REQUEST)
                    self.stream_output([req], req.return_logprob)
                    return

        elif (
            session_id in self.session_controller
            and not self.session_controller.get(session_id).close_on_finish
        ):
            # Session exists and is not closing: create request from session
            # session 存在且未关闭：从 session 历史上下文创建请求（多轮对话续接）
            session = self.session_controller.get(session_id)
            req = session.create_req(
                recv_req,
                self.tokenizer,
                self.model_config.vocab_size,
                eos_token_ids=self.model_config.hf_eos_token_id,
            )
            # TODO: set trace context
            if self.enable_metrics:
                req.time_stats.set_metrics_collector(self.metrics_collector)
            if isinstance(req.finished_reason, FINISH_ABORT):
                # session.create_req 检测到错误（如超长）：直接 abort
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return

        else:
            # Session not found, or session is closing（session 不存在或已请求关闭）
            if session_id in self.session_controller:
                error_msg = (
                    f"Invalid request: close was requested for session {session_id}"
                )
            else:
                error_msg = f"Invalid request: session id {session_id} does not exist"
            # 构造一个仅用于报错的 Req，立即 abort
            req = Req(
                recv_req.rid,
                recv_req.input_text,
                recv_req.input_ids,
                recv_req.sampling_params,
                vocab_size=self.model_config.vocab_size,
                http_worker_ipc=recv_req.http_worker_ipc,
            )
            req.tokenizer = self.tokenizer
            req.set_finish_with_abort(error_msg)   # 标记为 abort 状态
            self.init_req_max_new_tokens(req)
            self._add_request_to_queue(req)         # 加入队列（会立即被 process_batch_result 清理并返回错误）
            return

        if self.spec_algorithm.is_dflash():
            # dflash 投机解码：校验请求合法性（如 token 格式要求）
            error_msg = validate_dflash_request(req)
            if error_msg is not None:
                req.set_finish_with_abort(error_msg)
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return

        # Handle multimodal inputs（处理多模态输入：图像/视频等）
        if recv_req.mm_inputs is not None:
            image_inputs = self._get_multimodal_inputs(recv_req.mm_inputs)

            SessionController.adjust_mm_offsets(recv_req, req, image_inputs)
            # 调整 session 续接时的多模态 token 偏移（保证多轮对话中图像 token 位置正确）

            # The following steps are already fast, execute locally on each rank.
            # Expand a single image token into multiple dummy tokens for receiving image embeddings.
            # The pad function is model-specific and can be None for some backends.
            # 将单个图像占位 token 扩展为多个假 token（为图像嵌入预留位置）
            if self.pad_input_ids_func:
                req.origin_input_ids = self.pad_input_ids_func(
                    req.origin_input_ids, image_inputs
                )
            req.extend_image_inputs(image_inputs)       # 把多模态特征挂载到 req
            self._maybe_compute_mrope_positions(req)    # 计算 M-RoPE 3D 位置

            if len(req.origin_input_ids) >= self.max_req_input_len:
                # 扩展后超过最大输入长度：abort
                req.set_finish_with_abort(
                    error_msg=(
                        "Multimodal prompt is too long after expanding multimodal tokens. "
                        f"After expanding {len(req.origin_input_ids_unpadded)=} => {len(req.origin_input_ids)} >= {self.max_req_input_len}."
                    )
                )
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return

        # initialize before returning（在 return 前必须先设置 max_new_tokens）
        self.init_req_max_new_tokens(req)

        # Validate prompt length（校验 prompt 长度，超限则 abort 或自动截断）
        error_msg = validate_input_length(
            req,
            self.max_req_input_len,
            self.server_args.allow_auto_truncate,
        )
        if error_msg:
            req.set_finish_with_abort(error_msg)
            self._add_request_to_queue(req)
            return

        if not recv_req.return_logprob and recv_req.logprob_start_len != -1:
            # When return_logprob is False, logprob_start_len should be ignored
            # 不需要 logprob 时，忽略 logprob_start_len 设置
            recv_req.logprob_start_len = -1

        if recv_req.logprob_start_len == -1:
            if recv_req.return_logprob and recv_req.token_ids_logprob is None:
                # If logprob is required but neither token_ids_logprob nor logprob_start_len is
                # set, return the logprobs for output tokens by default
                # 需要 logprob 但未指定起始位置：默认从 output token 开始（跳过 input 的 logprob）
                req.logprob_start_len = len(req.origin_input_ids)
            elif req.is_prefill_only:
                # For prefill-only requests with logprob_start_len == -1, set logprob_start_len
                # beyond input sequence to skip input logprob computation entirely
                # prefill-only 请求：跳过所有 logprob 计算（仅做 KV 填充）
                req.logprob_start_len = len(req.origin_input_ids)
            else:
                # If return_logprob is False, only the last token requires logprob computation
                # 不需要 logprob：-1 表示只计算最后一个 token 的 logprob（采样用）
                req.logprob_start_len = -1
        else:
            req.logprob_start_len = recv_req.logprob_start_len  # 使用用户指定的起始位置

        if req.logprob_start_len > len(req.origin_input_ids):
            # logprob_start_len 超过了 input token 数量：非法参数，abort
            error_msg = f"{req.logprob_start_len=} is higher than the number of input tokens {len(req.origin_input_ids)=}. Please use a smaller logprob_start_len."
            req.logprob_start_len = -1
            req.set_finish_with_abort(error_msg)
            self._add_request_to_queue(req)
            return

        # 处理语法约束（JSON Schema / regex / EBNF）：
        # 如果有 grammar，把请求放入 grammar_queue（等待 grammar 编译完成）
        # grammar 编译完后再移入 waiting_queue
        added_to_grammar_queue = self.grammar_manager.process_req_with_grammar(req)
        if not added_to_grammar_queue:
            self._add_request_to_queue(req)  # 无 grammar 约束：直接加入等待队列

    # -------- 处理批量文本生成请求（逐条调用 handle_generate_request）--------
    def handle_batch_generate_request(
        self,
        recv_req: BatchTokenizedGenerateReqInput,
    ):
        """Handle optimized batch generate request."""
        logger.debug(f"Processing batch generate request with {len(recv_req)} requests")

        # Process each request in the batch
        for tokenized_req in recv_req:
            self.handle_generate_request(tokenized_req)  # 复用单条处理逻辑

    # -------- 从 HiCache 存储（L3）预取 KV Cache（异步预热）--------
    def _prefetch_kvcache(self, req: Req):
        if self.enable_hicache_storage:
            req.init_next_round_input(self.tree_cache, cow_mamba=False)
            last_host_node = req.last_host_node  # 最后一个主机侧 RadixTree 节点
            if last_host_node.backuped or last_host_node is self.tree_cache.root_node:
                last_hash = last_host_node.get_last_hash_value()
                matched_len = len(req.prefix_indices) + req.host_hit_length  # 已命中的前缀长度
                new_input_tokens = req.fill_ids[matched_len:]  # 未命中的 token（需要从 L3 加载）

                prefix_keys = (
                    last_host_node.get_prefix_hash_values(last_host_node.parent)
                    if self.tree_cache.hicache_storage_pass_prefix_keys  # 是否传递前缀 key
                    else None
                )
                # 异步触发 L3 → GPU 的 KV Cache 预取（后台线程执行，不阻塞调度循环）
                self.tree_cache.prefetch_from_storage(
                    req.rid,
                    last_host_node,
                    new_input_tokens,
                    last_hash,
                    prefix_keys,
                )

    # -------- 将请求加入对应的等待队列（根据解耦模式路由）--------
    def _add_request_to_queue(self, req: Req, is_retracted: bool = False):
        if self.disaggregation_mode == DisaggregationMode.NULL:
            # 普通模式：校验优先级 → 检查队列上限 → 预取 KV → 加入 waiting_queue
            if not self._set_or_validate_priority(req):
                return  # 优先级校验失败（如服务器不支持优先级调度），已 abort
            if self._abort_on_queued_limit(req):
                return  # 队列已满，已 abort（或踢出低优先级请求）
            self._prefetch_kvcache(req)                     # 异步预取 HiCache L3 KV
            self.waiting_queue.append(req)
            req.time_stats.set_wait_queue_entry_time()      # 记录入队时间（用于超时检查）
        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            # Prefill 节点：加入 bootstrap 队列（等待与 decode 节点握手）
            self._prefetch_kvcache(req)
            self.disagg_prefill_bootstrap_queue.add(
                req, self.model_config.num_key_value_heads  # KV head 数量（用于 RDMA buffer 计算）
            )
            req.time_stats.set_prefill_bootstrap_queue_entry_time()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            # Decode 节点：加入预分配队列（等待 KV Cache 预分配完成）
            self.disagg_decode_prealloc_queue.add(req, is_retracted=is_retracted)
            if not is_retracted:
                req.time_stats.set_decode_prealloc_queue_entry_time()  # 首次入队：记录时间
            else:
                req.time_stats.set_retract_time()  # 被 retract 后重入队：记录 retract 时间
        else:
            raise ValueError(f"Invalid {self.disaggregation_mode=}")

    # -------- 设置或校验请求的优先级 --------
    def _set_or_validate_priority(self, req: Req) -> bool:
        """Set the default priority value, or abort the request based on the priority scheduling mode."""
        if self.enable_priority_scheduling and req.priority is None:
            # 启用优先级调度但请求未设置优先级：赋默认优先级（最低优先级）
            if self.schedule_low_priority_values_first:
                req.priority = sys.maxsize          # 小值高优先：默认最低优先级 = 最大值
            else:
                req.priority = -sys.maxsize - 1     # 大值高优先：默认最低优先级 = 最小值
        elif (
            not self.enable_priority_scheduling
            and req.priority is not None
            and self.abort_on_priority_when_disabled  # 服务器禁用优先级时，拒绝带优先级的请求
        ):
            # 服务器不支持优先级调度，但请求携带了 priority → abort
            abort_req = AbortReq(
                finished_reason={
                    "type": "abort",
                    "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                    "message": "Using priority is disabled for this server. Please send a new request without a priority.",
                },
                rid=req.rid,
            )
            req.time_stats.trace_ctx.abort(abort_info=abort_req.finished_reason)
            self.send_to_tokenizer.send_output(abort_req, req)
            return False
        return True

    # -------- 队列满时中止最低优先级请求（或直接拒绝新请求）--------
    def _abort_on_queued_limit(self, recv_req: Req) -> bool:
        """Abort an incoming or existing request if the waiting queue is full. Returns True if the incoming request is aborted."""
        if (
            self.max_queued_requests is None                             # 未设置队列上限
            or len(self.waiting_queue) + 1 <= self.max_queued_requests  # 队列未满
        ):
            return False

        # Reject the incoming request by default.
        req_to_abort = recv_req  # 默认 abort 新到达的请求
        message = "The request queue is full."
        if self.enable_priority_scheduling:
            # With priority scheduling, consider aboritng an existing request based on the priority.
            # direction = 1  => smaller number = higher priority; -1 => larger number = higher priority.
            # max(...) + (direction * priority, queue_time_start) picks the least-preferred request.
            # Tie: later queue_time_start (newer) is evicted first. Preempt only if strictly better.
            # 优先级调度：找出队列中最低优先级的请求，若新请求优先级更高则踢出旧请求
            direction = 1 if self.schedule_low_priority_values_first else -1  # 数值方向：小值高优先时 direction=1
            key_fn = lambda item: (
                direction * item[1].priority,           # 主排序：优先级（值越大越低优先）
                item[1].time_stats.wait_queue_entry_time,  # 次排序：入队时间（越晚越先被踢出）
            )
            idx, candidate_req = max(enumerate(self.waiting_queue), key=key_fn)
            # 新请求的优先级是否严格高于队列中最低优先级的请求？
            abort_existing_req = (
                direction * recv_req.priority < direction * candidate_req.priority
            )
            if abort_existing_req:
                # 踢出队列中最低优先级的请求，为新请求腾出位置
                if self.enable_hicache_storage:
                    # Release prefetch events associated with the request
                    self.tree_cache.release_aborted_request(candidate_req.rid)  # 释放预取事件
                elif self.enable_hierarchical_cache:
                    self.tree_cache.terminate_prefetch(candidate_req.rid)  # 终止预取
                self.waiting_queue.pop(idx)
                req_to_abort = candidate_req           # abort 的是被踢出的旧请求
                message = "The request is aborted by a higher priority request."

        # 发送 abort 通知给 TokenizerManager
        self.send_to_tokenizer.send_output(
            AbortReq(
                finished_reason={
                    "type": "abort",
                    "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                    "message": message,
                },
                rid=req_to_abort.rid,
            ),
            req_to_abort,
        )
        req_to_abort.time_stats.trace_ctx.abort(abort_info={"reason": message})
        return req_to_abort.rid == recv_req.rid  # True 表示新请求本身被 abort（队列未踢旧请求）

    # -------- 清理等待超时的请求 --------
    def _abort_on_waiting_timeout(self):
        if (timeout_s := envs.SGLANG_REQ_WAITING_TIMEOUT.get()) <= 0:
            return  # 等待超时检查未启用

        deleted_reqs = set()
        deadline = time.perf_counter() - timeout_s  # 超过此时刻入队的请求视为超时
        for req in self.waiting_queue:
            entry_time = req.time_stats.wait_queue_entry_time
            if 0 < entry_time < deadline:
                if self.enable_hicache_storage:
                    # Release prefetch events associated with the request
                    self.tree_cache.release_aborted_request(req.rid)  # 释放 L3 预取事件
                # 发送超时 abort 通知
                self.send_to_tokenizer.send_output(
                    AbortReq(
                        finished_reason={
                            "type": "abort",
                            "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                            "message": "Request waiting timeout reached.",
                        },
                        rid=req.rid,
                    ),
                    req,
                )
                deleted_reqs.add(req)

        if deleted_reqs:
            # 从 waiting_queue 中移除所有超时请求
            self.waiting_queue = [
                req for req in self.waiting_queue if req not in deleted_reqs
            ]

    # -------- 处理单条嵌入（embedding）请求 --------
    def handle_embedding_request(
        self,
        recv_req: TokenizedEmbeddingReqInput,
    ):
        # 构造内部 Req 对象（嵌入请求比生成请求字段更少）
        req = Req(
            recv_req.rid,
            recv_req.input_text,
            recv_req.input_ids,
            recv_req.sampling_params,
            positional_embed_overrides=recv_req.positional_embed_overrides,  # 覆盖位置编码
            token_type_ids=recv_req.token_type_ids,                          # BERT token 类型 ID
            routed_dp_rank=recv_req.routed_dp_rank,                          # 路由到的 DP rank
            priority=recv_req.priority,                                      # 请求优先级
            dimensions=recv_req.dimensions,                                  # 输出嵌入维度（降维用）
            lora_id=recv_req.lora_id,                                        # LoRA 适配器 ID
            http_worker_ipc=recv_req.http_worker_ipc,
            time_stats=recv_req.time_stats,
            return_pooled_hidden_states=recv_req.return_pooled_hidden_states,  # 是否返回 pooled 隐藏状态
            multi_item_delimiter_indices=recv_req.multi_item_delimiter_indices,
        )
        req.tokenizer = self.tokenizer

        # Handle multimodal inputs（多模态嵌入：如图文混合的 embedding 模型）
        if recv_req.image_inputs is not None:
            image_inputs = self._get_multimodal_inputs(recv_req.image_inputs)
            # Expand a single image token into multiple dummy tokens for receiving image embeddings
            # The `pad_input_ids_func` is model-specific and may be None for
            # embedding models or models not requiring special padding.
            # If None, `req.origin_input_ids` is expected to be correctly populated already.
            if self.pad_input_ids_func:
                req.origin_input_ids = self.pad_input_ids_func(
                    req.origin_input_ids, image_inputs
                )

            req.extend_image_inputs(image_inputs)
            self._maybe_compute_mrope_positions(req)  # 计算多模态 RoPE 位置（如适用）

            if len(req.origin_input_ids) >= self.max_req_input_len:
                req.set_finish_with_abort(
                    error_msg=(
                        "Multimodal prompt is too long after expanding multimodal tokens. "
                        f"After expanding {len(req.origin_input_ids_unpadded)=} => {len(req.origin_input_ids)} >= {self.max_req_input_len}."
                    )
                )
                self._add_request_to_queue(req)
                return

        # Validate prompts length（校验 prompt 长度）
        error_msg = validate_input_length(
            req,
            self.max_req_input_len,
            self.server_args.allow_auto_truncate,
        )
        if error_msg:
            self._add_request_to_queue(req)  # 超长则 abort（req 中已含错误信息）
            return

        # Copy more attributes
        req.logprob_start_len = -1  # 嵌入请求不需要 logprob
        self._add_request_to_queue(req)

    # -------- 处理批量嵌入请求（逐条调用 handle_embedding_request）--------
    def handle_batch_embedding_request(
        self,
        recv_req: BatchTokenizedEmbeddingReqInput,
    ):
        """Handle optimized batch embedding request."""
        logger.debug(
            f"Processing batch embedding request with {len(recv_req)} requests"
        )

        # Process each request in the batch
        for tokenized_req in recv_req:
            self.handle_embedding_request(tokenized_req)

    # -------- 将未完成的分块请求保存到 RadixCache（等待下一块继续）--------
    def stash_chunked_request(self, req: Req):
        self.tree_cache.cache_unfinished_req(req, chunked=True)  # 保存前缀到 RadixCache（chunked=True 标记为未完成）

    # -------- 为 HiSparse 请求构建 decode 批次（从 staging 阶段过渡到 decode 阶段）--------
    def _build_hisparse_decode_batch(self, reqs):
        """Build a ScheduleBatch for hisparse requests transitioning from staging to decode."""
        device = self.device

        # 初始化一个新的 ScheduleBatch（不经过 waiting_queue 的常规路径）
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=self.enable_overlap,
            spec_algorithm=self.spec_algorithm,
        )

        batch.req_pool_indices = torch.tensor(
            [r.req_pool_idx for r in reqs], dtype=torch.int64, device=device
        )
        seq_lens = [len(r.origin_input_ids) + len(r.output_ids) - 1 for r in reqs]
        batch.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=device)
        batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        batch.seq_lens_sum = sum(seq_lens)
        # output_ids = last generated token, used as input_ids by prepare_for_decode
        batch.output_ids = torch.tensor(
            [r.output_ids[-1] for r in reqs], dtype=torch.int64, device=device
        )

        # Set logprob fields if any request needs them
        if batch.return_logprob:
            batch.top_logprobs_nums = [r.top_logprobs_num for r in reqs]
            batch.token_ids_logprobs = [list(r.origin_input_ids) for r in reqs]

        # Build sampling info from scratch for these requests
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch, self.model_config.vocab_size
        )
        # todo hisparse, maybe other info to contain for the new batch
        return batch

    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
        # 【学习注释 ④】核心调度决策函数，决定本轮 GPU 跑哪些请求
        # 逻辑：上轮 prefill 完的请求先合并进 running_batch，再看要不要新 prefill
        # 返回值：ScheduleBatch（forward_mode=EXTEND/DECODE/MIXED）或 None
        self._abort_on_waiting_timeout()
        self._abort_on_running_timeout()
        # ↑ 超时的请求直接 abort，防止一条慢请求占用资源
        if self.dllm_config is not None:
            self.dllm_manager.filter_finished_reqs()

        # Merge the prefill batch into the running batch
        chunked_req_to_exclude = set()

        if self.dllm_config is not None and self.dllm_manager.any_staging_reqs():
            chunked_req_to_exclude.update(self.dllm_manager.staging_queue)
            for req in self.dllm_manager.staging_queue:
                self.stash_chunked_request(req)

        if self.chunked_req is not None:
            # Move the chunked request out of the batch so that we can merge
            # only finished requests to running_batch.
            chunked_req_to_exclude.add(self.chunked_req)
            self.stash_chunked_request(self.chunked_req)

        # HiSparse has its own prefill-to-decode transition; skip last_batch merge.
        if self.enable_hisparse:
            ready_reqs = self.hisparse_coordinator.collect_ready_reqs()
            if len(ready_reqs) > 0:
                new_batch = self._build_hisparse_decode_batch(ready_reqs)
                if self.running_batch.is_empty():
                    self.running_batch = new_batch
                else:
                    self.running_batch.merge_batch(new_batch)
                self.running_batch.hisparse_coordinator = self.hisparse_coordinator
            # Reset batch_is_full so the scheduler can schedule more prefills.
            self.running_batch.batch_is_full = False

        if (
            not self.enable_hisparse
            and self.last_batch
            and self.last_batch.forward_mode.is_extend()
        ):
            # ↑ 上一轮是 EXTEND（prefill）：把完成 prefill 的请求合并进 running_batch
            #   这些请求下一轮就会进入 DECODE 阶段（每步生成 1 个 token）
            if self.last_batch.chunked_req is not None:
                # In the context pipeline parallelism, after the last chunk, the current microbatch still track outdated chunked_req.
                # We need to discard it.
                chunked_req_to_exclude.add(self.last_batch.chunked_req)

            if self.dllm_config is not None and self.last_batch.reqs:
                chunked_req_to_exclude.update(self.last_batch.reqs)

            # Filter batch
            last_bs = self.last_batch.batch_size()
            self.last_batch.filter_batch(
                chunked_req_to_exclude=list(chunked_req_to_exclude)
            )
            if self.last_batch.batch_size() < last_bs:
                self.running_batch.batch_is_full = False

            # Merge the new batch into the running batch.
            if not self.last_batch.is_empty():
                if self.running_batch.is_empty():
                    self.running_batch = self.last_batch
                else:
                    # Merge running_batch with prefill batch
                    self.running_batch.merge_batch(self.last_batch)

        # For prefill-only batch, filter out finished requests since they
        # won't go through the decode step. This keeps running_batch accurate
        # for load reporting (num_running_reqs via /v1/loads).
        # Runs outside the last_batch block so stale requests are cleaned
        # even when no new batches arrive (e.g. traffic stops).
        if self.running_batch.is_prefill_only:
            self.running_batch.filter_batch()
            if self.running_batch.is_empty():
                self.running_batch.batch_is_full = False

        if self.dllm_config is not None:
            new_batch = self.get_new_batch_dllm()
        else:
            new_batch = self.get_new_batch_prefill()
            # ↑ 尝试从 waiting_queue 凑新的 prefill 批次
            #   → _get_new_batch_prefill_raw()

        need_mlp_sync = self.require_mlp_sync
        if (
            need_mlp_sync
            and not self.spec_algorithm.is_none()
            and not self.server_args.speculative_skip_dp_mlp_sync
        ):
            # NOTE: This branch makes sure prefill and decode batches will not be mixed when spec and dp-attn is enabled.
            # Before merging the new batch into running batch:
            # 1. All new batches are none -> need_mlp_sync remains true (sync is needed for decode batch).
            # 2. All new batches are some (prefill / idle) -> we do not need prepare mlp sync one more time.
            new_batch = self.maybe_prepare_mlp_sync_batch(new_batch)
            need_mlp_sync = new_batch is None

        if new_batch is not None:
            # Run prefill first if possible
            ret = new_batch
            # ↑ 有新 prefill 批次：优先做 prefill（EXTEND 或 MIXED）
        else:
            # Run decode (skip for prefill-only batches)
            if (
                not self.running_batch.is_empty()
                and not self.running_batch.is_prefill_only
            ):
                self.running_batch = self.update_running_batch(self.running_batch)
                ret = self.running_batch if not self.running_batch.is_empty() else None
                # ↑ 没有新 prefill：跑 running_batch 里的 decode 请求
            else:
                ret = None
                # ↑ 没有任何请求可跑 → 返回 None → event_loop 调用 on_idle()

        # Handle DP attention and log stats
        ret = self.maybe_prepare_mlp_sync_batch(ret, need_sync=need_mlp_sync)

        # Handle ngram embedding
        ret = self._maybe_prepare_ngram_embedding(ret)

        if ret:
            set_schedule_time_batch(ret)

        return ret

    def get_num_allocatable_reqs(self, running_bs):
        res = get_global_server_args().pp_max_micro_batch_size - running_bs
        if self.pp_size > 1:
            res = min(res, self.req_to_token_pool.available_size())
        return res

    def get_new_batch_prefill(self) -> Optional[ScheduleBatch]:
        prefill_delayer_single_pass = None
        if self.prefill_delayer:
            # Get max usage across all pools for prefill delay decision
            max_pool_usage = self.get_pool_stats().get_max_pool_usage()
            prefill_delayer_single_pass = PrefillDelayerSinglePassExecutor(
                self.prefill_delayer, token_usage=max_pool_usage
            )

        ret = self._get_new_batch_prefill_raw(
            prefill_delayer_single_pass=prefill_delayer_single_pass
        )

        if self.prefill_delayer:
            prefill_delayer_single_pass.finalize(actual_prefill=ret is not None)

        return ret

    def _get_new_batch_prefill_raw(
        self, prefill_delayer_single_pass: Optional[PrefillDelayerSinglePassExecutor]
    ) -> Optional[ScheduleBatch]:
        # 【学习注释 ④-续】从 waiting_queue 选出哪些请求做 prefill
        # 核心约束：总 token 数 ≤ max_prefill_tokens，KV Cache 空间够用
        # Check if the grammar is ready in the grammar queue
        if self.grammar_manager.has_waiting_grammars():
            ready_grammar_requests = self.grammar_manager.get_ready_grammar_requests()
            for req in ready_grammar_requests:
                self._add_request_to_queue(req)

        if self.enable_hierarchical_cache:
            self.tree_cache.check_hicache_events()

        if self.enable_priority_preemption:
            # Reset batch_is_full to try preemption with a prefill adder.
            self.running_batch.batch_is_full = False

        if (
            self.running_batch.batch_is_full or len(self.waiting_queue) == 0
        ) and self.chunked_req is None:
            return None
            # ↑ 快速返回：batch 已满 或 等待队列空，不需要新 prefill

        running_bs = len(self.running_batch.reqs)

        # Ignore the check if self.chunked_req is not None.
        # In the non-PP case, when self.chunked_req is not None, num_allocatable_reqs should always be greater than 0,
        # as the space for the chunked requests has just been released.
        # In PP case, chunked requests (or dllm requests) can start in one microbatch and end in another microbatch, so the max_running_requests per microbatch should not be strict.
        # Instead, we should always allow chunked requests to be added, otherwise, there will be a memory leak.
        if (
            self.get_num_allocatable_reqs(running_bs) <= 0
            and self.chunked_req is None
            and not self.enable_priority_preemption
        ):
            self.running_batch.batch_is_full = True
            return None

        # Get priority queue
        self.policy.calc_priority(self.waiting_queue, self.running_batch)
        # ↑ cache-aware 排序：优先调度能复用更多 RadixCache 前缀的请求
        #   相同系统 prompt 的请求会聚在一起，让 KV 共享最大化

        if TEST_RETRACT and running_bs > TEST_RETRACT_NO_PREFILL_BS:
            # If we are testing retraction and the running batch size exceeds
            # TEST_RETRACT_NO_PREFILL_BS, we skip the prefill to keep the requests
            # in the waiting queue.
            return None

        # Determine chunked_prefill_size for this batch
        chunked_prefill_size = self.chunked_prefill_size
        if self.chunked_req is not None and self.enable_dynamic_chunking:
            history_len = len(self.chunked_req.prefix_indices)
            dynamic_size = self.predict_next_chunk_size(history_len)
            if dynamic_size is not None:
                chunked_prefill_size = dynamic_size

        # Prefill policy
        adder = PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            self.running_batch,
            self.new_token_ratio,
            self.max_prefill_tokens,
            chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
            self.priority_scheduling_preemption_threshold,
            max_prefill_bs=self.max_prefill_bs,
            max_running_requests=self.max_running_requests,
            prefill_max_requests=self.server_args.prefill_max_requests,
            prefill_delayer_single_pass=prefill_delayer_single_pass,
            dllm_config=self.dllm_config,
        )
        # ↑ PrefillAdder 逐条"试填"请求：
        #   add_one_req(req) 内部会：
        #     1. tree_cache.match_prefix(req.token_ids) → 查 RadixCache 命中了多少
        #     2. 估算需要新分配的 KV Cache 页数
        #     3. 空间够 → alloc KV → 加入批次
        #     4. 空间不够 or token 预算超 → 停止

        if self.chunked_req is not None:
            self.chunked_req.init_next_round_input()
            self.chunked_req = adder.add_chunked_req(self.chunked_req)

        if self.enable_lora:
            running_loras = {req.lora_id for req in self.running_batch.reqs}

        # Get requests from the waiting queue to a new prefill batch
        for req in self.waiting_queue:
            if self.enable_lora and req.lora_id not in running_loras:
                if self.enable_lora_overlap_loading:
                    # For overlapping loading of LoRA weights with computation, we will load each adapter one at a time,
                    # as opposed to loading them in one batch
                    res = self.lora_overlap_loader.try_overlap_load_lora(
                        req.lora_id, running_loras
                    )
                    if not res:
                        continue
                else:
                    new_lora_set = {req.lora_id} | running_loras
                    if not self.tp_worker.model_runner.lora_manager.validate_lora_batch(
                        new_lora_set
                    ):
                        continue

            running_bs = len(self.running_batch.reqs)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                self.running_batch.batch_is_full = True
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                # In prefill mode, prealloc queue and transfer queue can also take memory,
                # so we need to check if the available size for the actual available size.
                if len(adder.can_run_list) >= self.req_to_token_pool.available_size():
                    self.running_batch.batch_is_full = True

            if self.running_batch.batch_is_full:
                if (
                    not self.enable_priority_preemption
                    or not adder.preempt_to_schedule(req, self.server_args)
                ):
                    break

            if self.enable_hicache_storage:
                prefetch_done = self.tree_cache.check_prefetch_progress(req.rid)
                if not prefetch_done:
                    # skip staging requests that are ongoing prefetch
                    continue
                # Pop the number of tokens loaded from storage (L3 hits)
                req.storage_hit_length = self.tree_cache.pop_prefetch_loaded_tokens(
                    req.rid
                )

            req.init_next_round_input(self.tree_cache)
            res = adder.add_one_req(
                req,
                has_chunked_req=(self.chunked_req is not None),
                truncation_align_size=self.truncation_align_size,
            )

            if self.enable_lora:
                running_loras.add(req.lora_id)

            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    if self.enable_hierarchical_cache:
                        # Set batch_is_full after making sure there are requests that can be served
                        self.running_batch.batch_is_full = len(
                            adder.can_run_list
                        ) > 0 or (not self.running_batch.is_empty())
                    else:
                        self.running_batch.batch_is_full = True
                # revert matched mamba idx to avoid memory leak, if req is not added.
                # Only free if the slot was freshly allocated in this batch (not
                # pre-existing from a session). Session-held slots have their own
                # lifecycle and freeing them here causes double-free.
                added = len(adder.can_run_list) > 0 and req is adder.can_run_list[-1]
                if (
                    not added
                    and req.mamba_pool_idx is not None
                    and not getattr(req, "session", None)
                ):
                    self.tree_cache.req_to_token_pool.mamba_pool.free(
                        req.mamba_pool_idx.unsqueeze(-1)
                    )
                    req.mamba_pool_idx = None
                break

        # Update waiting queue
        can_run_list: List[Req] = adder.can_run_list
        if len(can_run_list) == 0:
            return None

        can_run_set = set(can_run_list)
        self.waiting_queue = [x for x in self.waiting_queue if x not in can_run_set]
        if adder.preempt_list:
            for req in adder.preempt_list:
                self._add_request_to_queue(req)

        if adder.new_chunked_req is not None:
            # Update chunked prefill
            assert self.chunked_req is None
            self.chunked_req = adder.new_chunked_req

        if self.chunked_req is not None:
            self.chunked_req.is_chunked += 1

        # Record for logging prefill stats after forward
        self.adder = adder
        self.can_run_list = can_run_list
        self.running_bs = len(self.running_batch.reqs)

        set_time_batch(can_run_list, "set_forward_entry_time")

        # Create a new batch
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            chunked_req=self.chunked_req,
        )
        self.max_prefill_bs = max(self.max_prefill_bs, len(can_run_list))
        if self.enable_hierarchical_cache:
            # todo (zhiqiang): disable cuda graph execution if hicache loading triggered
            new_batch.hicache_consumer_index = (
                self.tree_cache.ready_to_load_host_cache()
            )

        new_batch.prepare_for_extend()

        # Record prefill stats for logging after forward.
        new_batch.prefill_stats = PrefillStats.from_adder(
            adder,
            self.running_batch.reqs,
            self.enable_priority_scheduling,
            num_pending_tokens=self._get_num_pending_tokens(
                chunk_deduct=(
                    self.chunked_req.extend_input_len
                    if self.chunked_req is not None
                    else 0
                )
            ),
        )

        # Mixed-style chunked prefill
        if (
            self.is_mixed_chunk
            and not self.running_batch.is_empty()
            and not (new_batch.return_logprob or self.running_batch.return_logprob)
            # mix_with_running cats input_ids but not input_embeds — shapes would mismatch
            and new_batch.input_embeds is None
        ):
            # TODO (lianmin): support return_logprob + mixed chunked prefill
            self.running_batch.filter_batch(v1_spec_info_filtered=True)
            if not self.running_batch.is_empty():
                self.running_batch.prepare_for_decode()
                new_batch.mix_with_running(self.running_batch)
                new_batch.decoding_reqs = self.running_batch.reqs
            self.running_batch = ScheduleBatch(
                reqs=[], batch_is_full=self.running_batch.batch_is_full
            )
        else:
            new_batch.decoding_reqs = None

        return new_batch

    # -------- 更新正在 decode 的 running_batch（过滤完成请求，处理 KV 溢出）--------
    def update_running_batch(self, batch: ScheduleBatch) -> Optional[ScheduleBatch]:
        """Update the current running decoding batch."""
        initial_bs = batch.batch_size()  # 记录初始 batch 大小（用于检测是否有请求被过滤）

        # 过滤已完成的请求（finished）和 spec v1 已处理的请求
        batch.filter_batch(v1_spec_info_filtered=True)
        if batch.is_empty():
            batch.batch_is_full = False  # batch 已空，允许下次调度新请求
            return batch

        # Eagerly release lock_ref on completed write-through nodes so they
        # become evictable, improving batch scheduling headroom.
        # 提前释放已完成 write-through 节点的锁引用，使其可被 evict，增大调度空间
        if self.enable_hierarchical_cache:
            self.tree_cache.flush_write_through_acks()

        # Check if decode out of memory（检查 KV Cache 是否溢出）
        if (kv_full_retract_flag := not batch.check_decode_mem()) or (
            TEST_RETRACT and self.forward_ct % TEST_RETRACT_INTERVAL == 0  # 测试 retract 模式
        ):
            # KV Cache 不足：触发 retract 流程
            # retract = 把部分 decode 请求踢回 waiting_queue，释放 KV Cache 空间
            old_available_tokens = self.token_to_kv_pool_allocator.available_size()
            old_ratio = self.new_token_ratio
            mamba_pool = getattr(self.tree_cache.req_to_token_pool, "mamba_pool", None)
            old_mamba_available = (
                mamba_pool.available_size() if mamba_pool is not None else None
            )
            # batch.retract_decode：选出要被踢走的请求，释放 KV Cache，并返回新的 token ratio
            retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode(
                self.server_args
            )
            new_available_tokens = self.token_to_kv_pool_allocator.available_size()
            new_token_gained = new_available_tokens - old_available_tokens  # 回收的 token 数
            mamba_num_gained = (
                mamba_pool.available_size() - old_mamba_available
                if mamba_pool is not None
                else None
            )

            self.num_retracted_reqs = len(retracted_reqs)
            if self.enable_metrics and len(retracted_reqs) > 0:
                # 记录 retract 指标（retracted 请求数、输入/输出 token 数）
                self.metrics_collector.increment_retracted_reqs(
                    num_retracted_reqs=len(retracted_reqs),
                    num_retracted_input_tokens=sum(
                        len(r.origin_input_ids) for r in retracted_reqs
                    ),
                    num_retracted_output_tokens=sum(
                        len(r.output_ids) for r in retracted_reqs
                    ),
                )
            self.new_token_ratio = new_token_ratio  # 更新 token ratio（保守系数上升，防止再次溢出）
            # 向 TokenizerManager 发送需要直接 abort（无法 retract）的请求的错误
            for req in reqs_to_abort:
                abort_reason: FINISH_ABORT = req.to_finish
                self.send_to_tokenizer.send_output(
                    AbortReq(
                        finished_reason=abort_reason.to_json(),
                        rid=req.rid,
                    ),
                    req,
                )

            msg_prefix = (
                "KV cache pool is full. Retract requests. "
                if kv_full_retract_flag
                else "Testing retraction. "
            )
            msg_details = f"#retracted_reqs: {len(retracted_reqs)}, #new_tokens_gained: {new_token_gained}"
            if mamba_num_gained is not None:
                msg_details += f", #mamba_num_gained: {mamba_num_gained}"
            if kv_full_retract_flag:
                msg_details += (
                    f", #new_token_ratio: {old_ratio:.4f} -> {new_token_ratio:.4f}"
                )
            logger.warning(msg_prefix + msg_details)

            # 将被 retract 的请求重新加入 waiting_queue（标记 is_retracted=True）
            for req in retracted_reqs:
                self._add_request_to_queue(req, is_retracted=True)
        else:
            # decode 成功：逐步降低 new_token_ratio（朝最小值线性衰减）
            self.new_token_ratio = max(
                self.new_token_ratio - self.new_token_ratio_decay,
                self.min_new_token_ratio,
            )

        if batch.batch_size() < initial_bs:
            batch.batch_is_full = False  # 有请求被过滤，重置 batch_is_full

        if batch.is_empty():
            return batch  # 所有请求都被 retract 或 abort，返回空 batch

        # Update batch tensors（准备 decode 所需的 GPU tensor，如 seq_lens、position_ids）
        batch.prepare_for_decode()
        return batch

    # -------- 记录当前 batch 到双缓冲（防止 overlap 模式下 GPU tensor 被 GC 提前释放）--------
    def record_batch_in_overlap(self, model_worker_batch: ModelWorkerBatch):
        # FIXME(lsyin): hacky way to keep a reference to avoid GPU tensors being freed by torch GC
        # NOTE: More Reliable: record all tensors into the forward stream
        # NOTE: - for all future tensors, we shall always read from future map
        #       - for all non-future tensors (produced only by schedule stream),
        #       we shall keep its reference not being release during all the forwarding pass
        # 使用双缓冲（ping-pong）：始终保留最近 2 批次的 ModelWorkerBatch 引用，
        # 防止 forward_stream 还在使用 GPU tensor 时 Python 端将其 GC 掉
        self.batch_record_ct = (self.batch_record_ct + 1) % 2
        self.batch_record_buf[self.batch_record_ct] = model_worker_batch

    def run_batch(
        self,
        batch: ScheduleBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[GenerationBatchResult, EmbeddingBatchResult]:
        """Run a batch."""
        # 【学习注释 ⑤】把 ScheduleBatch 转换并送到 GPU 执行，返回结果
        # 数据变换链：ScheduleBatch → ModelWorkerBatch → (ModelRunner) → ForwardBatch
        self.forward_ct += 1

        # Whether to run the profiler
        self._profile_batch_predicate(batch)
        if self.forward_sleep_time is not None:
            logger.info(f"Scheduler.run_batch sleep {self.forward_sleep_time}s")
            time.sleep(self.forward_sleep_time)

        # Capture prefill start time for EXTEND mode
        if batch.forward_mode == ForwardMode.EXTEND:
            set_time_batch(batch.reqs, "set_prefill_run_batch_start_time")

        # Place holder handling for pd-disagg decode event loop
        if batch.forward_mode.is_prebuilt():
            return self._run_batch_prebuilt(batch)
            # ↑ PD 解耦场景：KV Cache 已从 Prefill 实例传输过来，直接进 decode

        # Run forward
        if self.is_generation:
            if self.spec_algorithm.is_none() or self.enable_overlap:
                # In most cases, we use the model worker batch to run the forward.
                worker_batch_or_batch = batch.get_model_worker_batch()
                # ↑ ScheduleBatch → ModelWorkerBatch
                #   把 List[Req] 里的字段提取并打包成 tensor
                #   例：[req.fill_ids] → input_ids tensor
                #       [req.kv_committed_len] → extend_prefix_lens
            else:
                # In speculative decoding v1 (non-overlap) case, we use the batch directly.
                # TODO(lsyin): delete this branch after unifying the abstraction.
                worker_batch_or_batch = batch

            if self.enable_overlap:
                model_worker_batch = worker_batch_or_batch
                self.record_batch_in_overlap(model_worker_batch)

                # Sampling info will be modified during forward, so we store a copy.
                model_worker_batch.sampling_info = (
                    model_worker_batch.sampling_info.copy_for_forward()
                )
                bs = len(model_worker_batch.seq_lens)
                future_indices = self.future_map.alloc_future_indices(bs)

                with self.forward_stream_ctx, self.record_bubble_metrics(batch):
                    self.forward_stream.wait_stream(self.schedule_stream)
                    self.future_map.resolve_future(model_worker_batch)
                    with self.record_forward_metrics(batch):
                        batch_result = self.model_worker.forward_batch_generation(
                            model_worker_batch
                            # here pp is not compatible with overlap
                        )
                    # FIXME(lsyin): maybe move this to forward_batch_generation
                    batch_result.copy_done = self.device_module.Event()
                    if batch_result.delay_sample_func is None:
                        self.future_map.store_to_map(future_indices, batch_result)
                        batch_result.copy_to_cpu(return_logprob=batch.return_logprob)
                    else:
                        batch_result.future_indices = future_indices

                # FIXME(lsyin): move this assignment elsewhere
                future_indices_or_next_token_ids = -future_indices.indices

                if batch.is_spec_v2:
                    # FIXME(lsyin): tmp code for spec v2
                    # We only keep future indices for next draft input

                    batch.spec_info = batch_result.next_draft_input
                    batch.spec_info.future_indices = future_indices

                    # batch.spec_info = EagleDraftInput(
                    #     future_indices=future_indices,
                    #     verify_done=batch_result.next_draft_input.verify_done,
                    # )

                    # The future value, usually for next batch preparation
                    # Current implementation strictly synchronizes the seq_lens
                    batch.seq_lens = batch_result.next_draft_input.new_seq_lens
            elif self.enable_pdmux and batch.forward_mode.is_split_prefill():
                batch_result = self.tp_worker.forward_batch_split_prefill(batch)
                future_indices_or_next_token_ids = batch_result.next_token_ids
            else:
                kwargs = (
                    {"pp_proxy_tensors": pp_proxy_tensors}
                    if self.spec_algorithm.is_none()
                    else {}
                )
                with self.record_forward_metrics(batch):
                    batch_result = self.model_worker.forward_batch_generation(
                        worker_batch_or_batch, **kwargs
                    )
                    # ↑ TpModelWorker.forward_batch_generation()
                    #     → ModelRunner.forward_batch_generation()
                    #       → 构造 ForwardBatch（全 GPU tensor）
                    #       → attention_backend.init_forward_metadata()
                    #       → model.forward()  真正执行 transformer
                    #       → logits_processor + sampler → next_token_ids
                    # 返回 GenerationBatchResult：
                    #   next_token_ids: GPU tensor shape [batch_size]
                    #   logits_output: 含 logprobs 等附加信息
                future_indices_or_next_token_ids = batch_result.next_token_ids
                self.update_cache_from_scheduler(batch, batch_result)

            # NOTE: future_indices_or_next_token_ids is used in ScheduleBatch,
            #       which can probably be replaced by future_indices later [TODO(lsyin)].
            #       we shall still keep the original outputs, e.g. next_token_ids
            #       in the GenerationBatchOutput for processing after copy_done.
            batch.output_ids = future_indices_or_next_token_ids

            # These 2 values are needed for processing the output, but the values can be
            # modified by overlap schedule. So we have to copy them here so that
            # we can use the correct values in output processing.
            if batch.return_logprob:
                batch_result.extend_input_len_per_req = [
                    req.extend_input_len for req in batch.reqs
                ]
                batch_result.extend_logprob_start_len_per_req = [
                    req.extend_logprob_start_len for req in batch.reqs
                ]
            else:
                batch_result.extend_input_len_per_req = None
                batch_result.extend_logprob_start_len_per_req = None

            ret = batch_result
        else:  # embedding or reward model
            model_worker_batch = batch.get_model_worker_batch()

            if self.enable_overlap:
                self.record_batch_in_overlap(model_worker_batch)
                with self.forward_stream_ctx, self.record_bubble_metrics(batch):
                    self.forward_stream.wait_stream(self.schedule_stream)
                    pooler_output = self.tp_worker.forward_batch_embedding(
                        model_worker_batch
                    )
                    ret = EmbeddingBatchResult(
                        embeddings=pooler_output.embeddings,
                        pooled_hidden_states=pooler_output.pooled_hidden_states,
                    )
                    ret.copy_to_cpu()
            else:
                pooler_output = self.tp_worker.forward_batch_embedding(
                    model_worker_batch
                )
                ret = EmbeddingBatchResult(
                    embeddings=pooler_output.embeddings,
                    pooled_hidden_states=pooler_output.pooled_hidden_states,
                )

        # Capture prefill end time for EXTEND mode
        if batch.forward_mode == ForwardMode.EXTEND:
            set_time_batch(batch.reqs, "set_prefill_run_batch_end_time")

        if (
            self.server_args.enable_dp_attention
            and self.server_args.elastic_ep_backend is not None
        ):
            # Get the tensors indicating rank activeness
            tp_active_ranks = self.tp_group.active_ranks.detach().cpu().numpy()
            tp_active_ranks_cpu = self.tp_group.active_ranks_cpu.detach().numpy()
            tp_active_ranks &= tp_active_ranks_cpu
            dp_active_ranks = tp_active_ranks.reshape(self.dp_size, -1).prod(axis=1)
            self.send_to_tokenizer.send_output(
                ActiveRanksOutput(status=dp_active_ranks.tolist())
            )

        return ret

    def launch_batch_sample_if_needed(
        self, batch_result: GenerationBatchResult
    ) -> Union[GenerationBatchResult]:
        # TODO(lsyin): make the delayed sample a default behavior after
        # unifying the forward_batch_generation interface (related to spec V2).
        if batch_result is None or batch_result.delay_sample_func is None:
            return

        with self.forward_stream_ctx:
            self.forward_stream.wait_stream(self.schedule_stream)
            _batch_result = batch_result.delay_sample_func()
            assert _batch_result is batch_result
            self.future_map.store_to_map(batch_result.future_indices, batch_result)
            batch_result.copy_to_cpu(return_logprob=self.cur_batch.return_logprob)

        # Release the closure and large GPU tensors that are no longer needed.
        # The delay_sample_func closure captures forward_batch (which holds
        # sampling_info with vocab_mask) and logits_output (which holds
        # next_token_logits). Without clearing these, they stay alive via
        # batch_result in result_queue and batch_record_buf until the next
        # iteration, causing a steady VRAM leak with structured output.
        batch_result.delay_sample_func = None
        if batch_result.logits_output is not None:
            batch_result.logits_output.next_token_logits = None

    # -------- 根据批次 forward_mode 分发结果处理 --------
    def process_batch_result(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        # 【学习注释 ⑥】根据 forward_mode 分发到对应的结果处理函数
        # DECODE → process_batch_result_decode（每步 1 token）
        # EXTEND → process_batch_result_prefill（prefill 产出第 1 个 token）
        if batch.forward_mode.is_decode():
            self.process_batch_result_decode(batch, result)    # 正常 decode 步
        elif batch.forward_mode.is_extend():
            if batch.is_dllm():
                self.process_batch_result_dllm(batch, result)  # dLLM（分布式 LLM）prefill
            elif self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.process_batch_result_disagg_prefill(batch, result)  # PD 解耦 prefill
            else:
                self.process_batch_result_prefill(batch, result)  # 普通 prefill
        elif batch.forward_mode.is_prebuilt():
            self.process_batch_result_prebuilt(batch)           # PD 解耦 decode（KV 已预传输）
        elif batch.forward_mode.is_idle():
            self.process_batch_result_idle(batch, result)       # idle 占位批次

        self.log_batch_result_stats(batch, result)     # 记录吞吐、延迟等指标
        self._maybe_clear_mm_inputs(batch)             # 释放已完成请求的多模态特征缓存
        self.maybe_send_health_check_signal()          # 发送积压的健康检查响应

    # -------- 发送积压的健康检查信号 --------
    def maybe_send_health_check_signal(self):
        if self.return_health_check_ipcs:
            # Return some signal for the health check.
            # This is used to prevent the health check signal being blocked by long context prefill.
            # However, one minor issue is that this code path does not check the status of detokenizer manager.
            # 注意：此处未检查 detokenizer 管理器状态，存在轻微误报
            self.send_to_tokenizer.send_output(
                HealthCheckOutput(
                    http_worker_ipc=self.return_health_check_ipcs.popleft()  # 弹出最早的健康检查请求
                )
            )

    # -------- 检查并执行延迟的 flush_cache 请求 --------
    def _check_pending_flush(self):
        if self._pending_flush is None:
            return  # 无待处理的 flush

        pending_req, deadline = self._pending_flush

        if self.is_fully_idle():
            # 服务器已空闲：执行 flush，并回复请求方
            success = self.flush_cache()
            self._pending_flush = None
            self.send_to_tokenizer.send_output(
                FlushCacheReqOutput(success=success), pending_req
            )
            return

        if time.monotonic() >= deadline:
            # 超过等待截止时间：放弃 flush，回复失败
            logging.warning(
                "Deferred flush_cache timed out while waiting for idle state."
            )
            self._pending_flush = None
            self.send_to_tokenizer.send_output(
                FlushCacheReqOutput(
                    success=False, message="Timed out waiting for idle state."
                ),
                pending_req,
            )

    # -------- 添加/删除/列出 ngram 投机解码的外部语料库 --------
    def add_external_corpus(
        self, recv_req: AddExternalCorpusReqInput
    ) -> Optional[AddExternalCorpusReqOutput]:
        if self.external_corpus_manager is None:
            return AddExternalCorpusReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        return self.external_corpus_manager.add(recv_req)  # 添加语料库

    def remove_external_corpus(
        self, recv_req: RemoveExternalCorpusReqInput
    ) -> RemoveExternalCorpusReqOutput:
        if self.external_corpus_manager is None:
            return RemoveExternalCorpusReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        return self.external_corpus_manager.remove(recv_req)  # 删除语料库

    def list_external_corpora(
        self, recv_req: ListExternalCorporaReqInput
    ) -> ListExternalCorporaReqOutput:
        if self.external_corpus_manager is None:
            return ListExternalCorporaReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        return self.external_corpus_manager.list(recv_req)  # 列出所有语料库

    # -------- 执行或延迟 flush_cache 请求 --------
    def flush_cache_wrapped(
        self, recv_req: FlushCacheReqInput
    ) -> Optional[FlushCacheReqOutput]:
        if self._pending_flush is not None:
            # 已有待处理的 flush：拒绝重复请求
            return FlushCacheReqOutput(
                success=False,
                message="Another flush_cache is already in progress.",
            )

        timeout_s = float(recv_req.timeout_s or 0.0)
        if timeout_s <= 0.0:
            # 无超时：立即尝试 flush（若不空闲则返回失败）
            return FlushCacheReqOutput(success=self.flush_cache())

        if self.is_fully_idle():
            # 服务器已空闲：立即 flush
            return FlushCacheReqOutput(success=self.flush_cache())

        # 服务器繁忙但有超时：延迟执行（等待空闲或超时）
        self._pending_flush = (recv_req, time.monotonic() + timeout_s)
        return None  # None 表示异步处理，不立即返回响应

    # -------- 清空层次缓存存储后端（L3 KV 存储）--------
    def clear_hicache_storage_wrapped(self, recv_req: ClearHiCacheReqInput):
        if self.enable_hierarchical_cache:
            self.tree_cache.clear_storage_backend()  # 清空 L3 存储（如磁盘 KV 数据库）
            logger.info("Hierarchical cache cleared successfully!")
            if_success = True
        else:
            logging.warning("Hierarchical cache is not enabled.")
            if_success = False
        return ClearHiCacheReqOutput(success=if_success)

    # -------- 检查服务器是否完全空闲（所有队列和批次均为空）--------
    def is_fully_idle(self, for_health_check=False) -> bool:
        # Health check piggybacks on running requests in process_output.
        # Only running_batch + waiting_queue guarantee active GPU processing;
        # disagg queues (bootstrap/prealloc/transfer) may have items without
        # any request actually running on GPU — e.g. stuck handshake, full
        # KV cache, or stalled transfer — so they can't carry health info.
        # Batch running status
        idle = (
            self.running_batch.is_empty()
            and self.chunked_req is None
            and not self.dllm_manager.any_staging_reqs()
            and (self.last_batch is None or self.last_batch.is_empty())
            and (self.cur_batch is None or self.cur_batch.is_empty())
            and (not self.enable_overlap or len(self.result_queue) == 0)
            and (self.pp_size == 1 or all(x.is_empty() for x in self.running_mbs))
        )

        # Waiting queues: waiting + bootstrapping + preallocation + kv transfer (decode)
        idle &= len(self.waiting_queue) == 0

        if not for_health_check:
            # Grammar queue and prefill inflight queue may not produce batch
            # results instantly, but they still indicate the server is not idle.
            idle &= len(self.grammar_manager.grammar_queue) == 0
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                idle &= len(self.disagg_prefill_inflight_queue) == 0
                idle &= len(self.disagg_prefill_bootstrap_queue.queue) == 0

            if self.disaggregation_mode == DisaggregationMode.DECODE:
                idle &= len(self.disagg_decode_prealloc_queue.queue) == 0
                idle &= len(self.disagg_decode_transfer_queue.queue) == 0
                if self.decode_offload_manager is not None:
                    idle &= len(self.decode_offload_manager.ongoing_offload) == 0

            # HiSparse: staging requests transitioning prefill -> decode
            if self.enable_hisparse:
                idle &= not self.hisparse_coordinator.has_ongoing_staging()

            # HiCache: in-flight async ops (GPU↔Host↔L3) must drain before
            # destructive operations like attach/detach/flush_cache.
            if self.enable_hierarchical_cache:
                tc = self.tree_cache
                idle &= len(tc.ongoing_write_through) == 0
                idle &= len(tc.ongoing_load_back) == 0
                if tc.enable_storage:
                    idle &= len(tc.ongoing_prefetch) == 0
                    idle &= len(tc.ongoing_backup) == 0

        return idle

    def attach_hicache_storage_wrapped(
        self, recv_req: AttachHiCacheStorageReqInput
    ) -> AttachHiCacheStorageReqOutput:
        if not self.enable_hierarchical_cache:
            return AttachHiCacheStorageReqOutput(
                success=False, message="Hierarchical cache is not enabled."
            )

        if not self.is_fully_idle():
            return AttachHiCacheStorageReqOutput(
                success=False,
                message=(
                    "Reject attach: scheduler is not idle. "
                    f"#queue-req={len(self.waiting_queue)} "
                    f"#running-req={len(self.running_batch.reqs)}"
                ),
            )

        if not hasattr(self.tree_cache, "attach_storage_backend"):
            return AttachHiCacheStorageReqOutput(
                success=False,
                message="Current tree_cache implementation does not support dynamic attach.",
            )

        try:
            ok, msg = self.tree_cache.attach_storage_backend(
                storage_backend=recv_req.hicache_storage_backend,
                storage_backend_extra_config_json=recv_req.hicache_storage_backend_extra_config_json,
                served_model_name=self.server_args.served_model_name,
                hicache_storage_prefetch_policy=recv_req.hicache_storage_prefetch_policy,
                hicache_write_policy=recv_req.hicache_write_policy,
            )
        except Exception as e:
            logger.exception("Attach HiCache storage backend failed with exception.")
            return AttachHiCacheStorageReqOutput(success=False, message=str(e))
        if ok:
            self.enable_hicache_storage = True
            self.server_args.hicache_storage_backend = recv_req.hicache_storage_backend
            if recv_req.hicache_storage_backend_extra_config_json is not None:
                self.server_args.hicache_storage_backend_extra_config = (
                    recv_req.hicache_storage_backend_extra_config_json
                )
            if recv_req.hicache_storage_prefetch_policy is not None:
                self.server_args.hicache_storage_prefetch_policy = (
                    recv_req.hicache_storage_prefetch_policy
                )
            if recv_req.hicache_write_policy is not None:
                self.server_args.hicache_write_policy = recv_req.hicache_write_policy
            logger.info(
                f"Attached HiCache storage backend: {recv_req.hicache_storage_backend}"
            )
        return AttachHiCacheStorageReqOutput(success=ok, message=msg)

    def detach_hicache_storage_wrapped(
        self, recv_req: DetachHiCacheStorageReqInput
    ) -> DetachHiCacheStorageReqOutput:
        if not self.enable_hierarchical_cache:
            return DetachHiCacheStorageReqOutput(
                success=False, message="Hierarchical cache is not enabled."
            )

        if not self.is_fully_idle():
            return DetachHiCacheStorageReqOutput(
                success=False,
                message=(
                    "Reject detach: scheduler is not idle. "
                    f"#queue-req={len(self.waiting_queue)} "
                    f"#running-req={len(self.running_batch.reqs)}"
                ),
            )

        if not hasattr(self.tree_cache, "detach_storage_backend"):
            return DetachHiCacheStorageReqOutput(
                success=False,
                message="Current tree_cache implementation does not support dynamic detach.",
            )

        # Idempotent detach: even if scheduler thinks storage is disabled, we still
        # attempt best-effort cleanup in tree_cache (it may have leftover state).
        try:
            ok, msg = self.tree_cache.detach_storage_backend()
        except Exception as e:
            logger.exception("Detach HiCache storage backend failed with exception.")
            return DetachHiCacheStorageReqOutput(success=False, message=str(e))

        if ok or (not self.enable_hicache_storage):
            # Treat "already disabled / nothing to do" as success for idempotence.
            self.enable_hicache_storage = False
            self.server_args.hicache_storage_backend = None
            self.server_args.hicache_storage_backend_extra_config = None
            logger.info("Detached HiCache storage backend.")
            return DetachHiCacheStorageReqOutput(
                success=True, message=msg or "HiCache storage backend is detached."
            )

        return DetachHiCacheStorageReqOutput(success=False, message=msg)

    # -------- 清空内存池和 KV Cache（仅在完全空闲时执行）--------
    def flush_cache(self, empty_cache: bool = True):
        """Flush the memory pool and cache."""
        if self.is_fully_idle():
            self.cur_batch = None   # 清空当前批次引用
            self.last_batch = None  # 清空上一批次引用
            self.tree_cache.reset()                      # 重置 RadixCache（清空所有节点）
            self.req_to_token_pool.clear()               # 清空请求→token 映射池
            self.token_to_kv_pool_allocator.clear()      # 清空 token → KV Cache 分配器
            self.grammar_manager.clear()                 # 清空 grammar 约束状态
            self.reset_metrics()                         # 重置指标统计

            if self.draft_worker:
                self.draft_worker.clear_cache_pool()     # 清空 draft worker 的 KV Cache 池

            if empty_cache:
                torch.cuda.empty_cache()
            logger.info("Cache flushed successfully!")
            success = True
        else:
            logging.warning(
                f"Cache not flushed because there are pending requests. "
                f"#queue-req: {len(self.waiting_queue)}, "
                f"#running-req: {len(self.running_batch.reqs)}"
            )
            success = False
        return success

    # -------- 获取调度器内部状态（用于监控/调试）--------
    def get_internal_state(self, recv_req: GetInternalStateReq):
        ret = dict(vars(get_global_server_args()))  # vars returns a ref to obj.__dict__（全局 server_args 的字典副本）
        ret["last_gen_throughput"] = self.last_gen_throughput  # 最新生成吞吐量（token/s）
        ret["memory_usage"] = {
            "weight": round(self.tp_worker.model_runner.weight_load_mem_usage, 2),  # 权重显存用量（GB）
            "kvcache": round(
                self.token_to_kv_pool_allocator.get_kvcache().mem_usage, 2
            ),  # KV Cache 显存用量（GB）
            "token_capacity": int(self.max_total_num_tokens),  # KV Cache 总 token 容量
            "graph": round(self.tp_worker.model_runner.graph_mem_usage, 2),  # CUDA Graph 显存用量（GB）
        }
        ret["effective_max_running_requests_per_dp"] = self.max_running_requests  # 每 DP rank 最大并发请求数

        if not self.spec_algorithm.is_none() and self.spec_total_num_forward_ct > 0:
            # 投机解码统计：平均接受长度（accepted tokens / forward 轮次）
            ret["avg_spec_accept_length"] = (
                self.spec_total_num_accepted_tokens / self.spec_total_num_forward_ct
            )

        if RECORD_STEP_TIME:
            ret["step_time_dict"] = self.step_time_dict  # 每步时间统计（调试用）

        # This field is not serializable.
        ret.pop("model_config", None)  # 移除不可序列化的 model_config 字段

        return GetInternalStateReqOutput(internal_state=ret)

    # -------- 动态更新服务器内部参数（仅允许白名单字段）--------
    def set_internal_state(self, recv_req: SetInternalStateReq):
        server_args_dict = recv_req.server_args  # 待更新的参数字典
        # 只允许更新以下字段（其他字段改动风险较高，需重启生效）
        args_allow_update = set(
            [
                "pp_max_micro_batch_size",               # PP 每个 micro batch 的最大请求数
                "speculative_accept_threshold_single",   # 投机解码单步接受阈值
                "speculative_accept_threshold_acc",      # 投机解码累积接受阈值
            ]
        )

        if_success = True
        for k, v in server_args_dict.items():
            if k not in args_allow_update:
                logging.warning(f"Updating {k} is not supported.")
                if_success = False
                break
            elif k == "pp_max_micro_batch_size" and (
                v > self.max_running_requests // self.pp_size or v < 1  # 超出合法范围
            ):
                logging.warning(
                    f"Updating {k} to {v} is rejected because it is out of the valid range [1, {self.max_running_requests // self.pp_size}]."
                )
                if_success = False
                break

        if if_success:
            if not self.spec_algorithm.is_none() and self.spec_total_num_forward_ct > 0:
                # 更新前记录当前投机解码接受率（便于对比）
                avg_spec_accept_length = (
                    self.spec_total_num_accepted_tokens / self.spec_total_num_forward_ct
                )
                logger.info(f"{avg_spec_accept_length=}")
            # 重置投机解码统计计数器
            self.spec_total_num_accepted_tokens = self.spec_total_num_forward_ct = 0
            # 应用参数更新
            for k, v in server_args_dict.items():
                setattr(get_global_server_args(), k, v)
            logger.info(f"Global server args updated! {get_global_server_args()=}")
        return SetInternalStateReqOutput(
            updated=True,
            server_args=vars(get_global_server_args()),
        )

    # -------- 通用 RPC 请求处理（通过方法名反射调用 Scheduler 方法）--------
    def handle_rpc_request(self, recv_req: RpcReqInput):
        # Handle RPC requests
        logger.info(
            f"handle_rpc_request: {recv_req.method}, param: {recv_req.parameters}"
        )

        success = True
        exec = None
        try:
            func = getattr(self, recv_req.method)  # 反射获取方法
            if recv_req.parameters is not None:
                func(**recv_req.parameters)  # 带参数调用
            else:
                func()                       # 无参数调用
        except Exception as e:
            success = False
            exec = e
            logger.error(f"Failed to call rpc {recv_req.method}: {str(e)}")

        barrier()  # 等待所有 TP rank 完成（防止 rank 间状态不一致）
        return RpcReqOutput(success, "" if not exec else str(exec))

    # -------- 中止指定请求（支持三种中止策略）--------
    def abort_request(self, recv_req: AbortReq):
        # todo hisparse, release resources for abort requests in hisparse coordinator
        # Delete requests in the waiting queue
        to_del = []
        for i, req in enumerate(self.waiting_queue):
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                to_del.append(i)

        # Sort in reverse order to avoid index issues when deleting
        for i in reversed(to_del):
            # Abort method 1: directly pop from the queue
            # This only works for requests that have not started anything.
            # We still need to send something back to TokenizerManager to clean up the state.
            # 中止方式一：直接从 waiting_queue 弹出（最高效，仅适用于未开始处理的请求）
            req = self.waiting_queue.pop(i)
            if self.enable_hicache_storage:
                # to release prefetch events associated with the request
                self.tree_cache.release_aborted_request(req.rid)
            self.send_to_tokenizer.send_output(AbortReq(rid=req.rid), req)
            # For disaggregation decode mode, the request in the waiting queue has KV cache allocated.
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                if self.enable_hisparse:
                    self.hisparse_coordinator.request_finished(req)
                release_kv_cache(req, self.tree_cache)
            # For disaggregation prefill mode, free the metadata buffer index
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                release_req_to_metadata_buffer(
                    req, self.req_to_metadata_buffer_idx_allocator
                )

            # For mamba radix cache
            if (
                req.mamba_pool_idx is not None
                and self.disaggregation_mode != DisaggregationMode.DECODE
            ):
                release_kv_cache(req, self.tree_cache, is_insert=False)
            logger.debug(f"Abort queued request. {req.rid=}")

        # Delete the requests in the grammar queue
        # Abort method 2: call `set_finish_with_abort`
        # The request will still run one prefill forward pass.
        # In this case, we change the input_ids to be only one token to make this prefill cheap.
        # 中止方式二：将 grammar 队列中的请求标记为 abort
        # 请求仍会执行一次 prefill（input_ids 被截短为 1 个 token，成本极低）
        self.grammar_manager.abort_requests(recv_req)

        # Delete requests not in the waiting queue when PD disaggregation is enabled
        # PD 解耦模式：还需要清理各类子队列中的请求
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # Abort requests that have not yet been bootstrapped
            # 中止尚未完成与 decode 节点握手的请求
            for req in self.disagg_prefill_bootstrap_queue.queue:
                if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort bootstrap queue request. {req.rid=}")
                    if hasattr(req.disagg_kv_sender, "abort"):
                        req.disagg_kv_sender.abort()  # 通知 KV 发送方取消传输

            # Abort in-flight requests
            # 中止正在传输 KV Cache 的请求（in-flight = KV 传输进行中）
            for req in self.disagg_prefill_inflight_queue:
                if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort inflight queue request. {req.rid=}")
                    if hasattr(req.disagg_kv_sender, "abort"):
                        req.disagg_kv_sender.abort()

        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            # Abort requests that have not yet finished preallocation
            # 中止等待 KV Cache 预分配的请求
            for decode_req in self.disagg_decode_prealloc_queue.queue:
                if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort prealloc queue request. {decode_req.req.rid=}")
                    decode_req.kv_receiver.abort()  # 通知 KV 接收方取消

            # Abort requests waiting for kvcache to release tree cache
            # 中止等待 KV 传输完成的请求
            for decode_req in self.disagg_decode_transfer_queue.queue:
                if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort transfer queue request. {decode_req.req.rid=}")
                    decode_req.kv_receiver.abort()

            # Abort requests already retracted to CPU cache（中止已被 retract 到 CPU 缓存的请求）
            if self.disagg_decode_prealloc_queue.retracted_queue:
                remaining_retracted = []
                for decode_req in self.disagg_decode_prealloc_queue.retracted_queue:
                    if recv_req.abort_all or decode_req.rid.startswith(recv_req.rid):
                        assert hasattr(decode_req, "kv_cache_cpu")
                        del decode_req.kv_cache_cpu  # 释放 CPU 上的 KV Cache
                        self.send_to_tokenizer.send_output(
                            AbortReq(rid=decode_req.rid), decode_req
                        )
                    else:
                        remaining_retracted.append(decode_req)
                self.disagg_decode_prealloc_queue.retracted_queue = remaining_retracted

        # Delete requests in the running batch
        # 在 running_batch 和 cur_batch 中查找目标请求
        if self.cur_batch is self.running_batch or self.cur_batch is None:
            reqs = self.running_batch.reqs
        else:
            reqs = self.running_batch.reqs + self.cur_batch.reqs

        for req in reqs:
            if not req.finished() and (
                recv_req.abort_all or req.rid.startswith(recv_req.rid)
            ):
                # Abort method 3: set `to_finish`
                # The request will still run one decode forward pass.
                # Then we reuse all existing code to clean up the KV cache allocation.
                # 中止方式三：设置 to_finish = FINISH_ABORT，在下一次 decode 后清理 KV Cache
                # 这是最安全的方式：复用现有的 process_batch_result 清理逻辑
                logger.debug(f"Abort running request. {req.rid=}")
                req.to_finish = FINISH_ABORT()

    # -------- 引擎暂停接口（由子类实现，基类直接抛异常）--------
    def _pause_engine(self) -> Tuple[List[Req], int]:
        raise NotImplementedError()

    # -------- 暂停生成（两种模式：in_place 就地暂停 / retract 回收请求）--------
    def pause_generation(self, recv_req: PauseGenerationReqInput):
        self._engine_paused = True  # 设置暂停标志，event_loop 检测到后停止调度

        if recv_req.mode == "in_place":
            # In-place pause: just set the flag and return immediately.
            # All scheduler state (running_batch, last_batch, chunked_req,
            # result_queue) is left untouched. On resume, the normal event
            # loop (get_next_batch_to_run) handles last_batch merge,
            # chunked_req cleanup, and overlap result processing through
            # the standard code paths. This avoids duplicating batch
            # manipulation logic and the accounting bugs that come with it.
            return

        if self.enable_overlap and self.last_batch:
            # Process the results of the last batch
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        if self.last_batch and self.last_batch.forward_mode.is_extend():
            chunked_req_to_exclude = set()
            self.last_batch.filter_batch(
                chunked_req_to_exclude=list(chunked_req_to_exclude)
            )
            # Skip merge for disagg prefill: completed prefill requests are
            # already in disagg_prefill_inflight_queue. Merging them into
            # running_batch leaks them, since the prefill event loop never
            # calls update_running_batch to clean them up.
            if (
                not self.last_batch.is_empty()
                and self.disaggregation_mode != DisaggregationMode.PREFILL
            ):
                if self.running_batch.is_empty():
                    self.running_batch = self.last_batch
                else:
                    self.running_batch.merge_batch(self.last_batch)

        self.last_batch = None
        self.cur_batch = None

        if recv_req.mode == "retract" and not self.running_batch.is_empty():
            self.running_batch.filter_batch(v1_spec_info_filtered=True)
            if len(self.running_batch.reqs) != 0:
                retracted_reqs = self.running_batch.retract_all(self.server_args)
                for req in retracted_reqs:
                    self._add_request_to_queue(req)

            self.running_batch.batch_is_full = False
            self.chunked_req = None

    # -------- 恢复生成（清除暂停标志）--------
    def continue_generation(self, recv_req: ContinueGenerationReqInput):
        self._engine_paused = False  # 清除暂停标志，event_loop 下一轮将恢复调度

    # -------- LoRA 适配器管理（加载/卸载）--------
    def load_lora_adapter(
        self, recv_req: LoadLoRAAdapterReqInput
    ) -> LoadLoRAAdapterReqOutput:
        """In-place loading a new lora adapter from disk or huggingface."""
        # 委托给 TP worker 执行（需在所有 TP rank 同步执行）
        result = self.tp_worker.load_lora_adapter(recv_req)
        return result

    def load_lora_adapter_from_tensors(
        self, recv_req: LoadLoRAAdapterFromTensorsReqInput
    ) -> LoadLoRAAdapterFromTensorsReqOutput:
        """In-place loading a new lora adapter from serialized tensors."""
        # 从序列化 tensor 加载 LoRA（用于在线 fine-tune 后直接注入权重）
        result = self.tp_worker.load_lora_adapter_from_tensors(recv_req)
        return result

    def unload_lora_adapter(
        self, recv_req: UnloadLoRAAdapterReqInput
    ) -> UnloadLoRAAdapterReqOutput:
        """Unload the lora adapter."""
        result = self.tp_worker.unload_lora_adapter(recv_req)
        return result

    # -------- 跨实例权重传输（RLHF 在线更新场景）--------
    def init_weights_send_group_for_remote_instance(
        self, recv_req: InitWeightsSendGroupForRemoteInstanceReqInput
    ):
        """Init the seed and client instance communication group."""
        success, message = self.tp_worker.init_weights_send_group_for_remote_instance(
            recv_req
        )
        return InitWeightsSendGroupForRemoteInstanceReqOutput(success, message)

    def send_weights_to_remote_instance(
        self, recv_req: SendWeightsToRemoteInstanceReqInput
    ):
        """Send the seed instance weights to the destination instance."""
        success, message = self.tp_worker.send_weights_to_remote_instance(recv_req)
        return SendWeightsToRemoteInstanceReqOutput(success, message)

    # -------- 调试辅助：人为减速 forward（用于复现时序相关 bug）--------
    def slow_down(self, recv_req: SlowDownReqInput):
        t = recv_req.forward_sleep_time
        if t is not None and t <= 0:
            t = None  # <= 0 视为取消减速
        self.forward_sleep_time = t  # 设置每次 forward 前的等待时间（秒）
        return SlowDownReqOutput()

    # -------- MoE 专家分布记录控制（开始/停止/导出）--------
    def expert_distribution_handle(self, recv_req: ExpertDistributionReq):
        action = recv_req.action
        if action == ExpertDistributionReqType.START_RECORD:
            get_global_expert_distribution_recorder().start_record()   # 开始记录专家路由分布
        elif action == ExpertDistributionReqType.STOP_RECORD:
            get_global_expert_distribution_recorder().stop_record()    # 停止记录
        elif action == ExpertDistributionReqType.DUMP_RECORD:
            get_global_expert_distribution_recorder().dump_record()    # 导出记录到文件
        else:
            raise ValueError(f"Unrecognized ExpertDistributionReq value: {recv_req=}")
        return ExpertDistributionReqOutput()

    # -------- 会话管理（多轮对话 session）--------
    def open_session(self, recv_req: OpenSessionReqInput):
        output = self.session_controller.open(recv_req)  # 在 SessionController 中注册新 session
        # 只有主 rank（pp=0, tp=0, attn_cp=0）才向客户端返回 session ID
        if self.pp_rank == 0 and self.tp_rank == 0 and self.attn_cp_rank == 0:
            return output
        return None  # 其他 rank 不返回（避免重复响应）

    def close_session(self, recv_req: CloseSessionReqInput):
        self.session_controller.close(recv_req)  # 关闭 session（标记为 close_on_finish）

    # -------- 空闲时主动 sleep（节省 CPU / 降功耗）--------
    def maybe_sleep_on_idle(self):
        if self.idle_sleeper is not None:
            self.idle_sleeper.maybe_sleep()  # 用 zmq Poller 等待 1s，避免空转

    # -------- 冻结 GC 请求处理（减少 GC 引发的延迟尖刺）--------
    def handle_freeze_gc(self, recv_req: FreezeGCReq):
        """Handle freeze_gc request: freeze scheduler's GC and forward to detokenizer."""
        freeze_gc("Scheduler")                             # 冻结 Scheduler 进程的 GC
        self.send_to_detokenizer.send_output(recv_req, recv_req)  # 转发给 Detokenizer 也冻结
        return None

    # -------- 调试 dumper 控制（动态开启/关闭 tensor dump）--------
    def handle_dumper_control(self, recv_req: DumperControlReqInput):
        from sglang.srt.debug_utils.dumper import dumper  # 延迟导入调试模块

        try:
            response: list = []
            if (
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0  # 只在 rank 0 执行（避免重复）
            ):
                response = dumper._http_manager.handle_request(
                    method=recv_req.method, body=recv_req.body
                )
            self.send_to_tokenizer.send_output(
                DumperControlReqOutput(success=True, response=response), recv_req
            )
        except Exception as e:
            print(f"[Scheduler] handle_dumper_control error: {e}", flush=True)
            self.send_to_tokenizer.send_output(
                DumperControlReqOutput(success=False, response=[], error=str(e)),
                recv_req,
            )

    # placeholder for override
    # 占位方法：供子类（如 dLLM Scheduler）覆盖，在 forward 完成后更新 LMCache 等外部缓存
    def update_cache_from_scheduler(
        self, schedule_batch: ScheduleBatch, batch_result: GenerationBatchResult
    ):
        pass


# -------- IdleSleeper：空闲时降低 CPU 占用的辅助类 --------
class IdleSleeper:
    """
    In setups which have long inactivity periods it is desirable to reduce
    system power consumption when sglang does nothing. This would lead not only
    to power savings, but also to more CPU thermal headroom when a request
    eventually comes. This is important in cases when multiple GPUs are connected
    as each GPU would otherwise pin one thread at 100% CPU usage.

    The simplest solution is to use zmq.Poller on all sockets that may receive
    data that needs handling immediately.
    """
    # 原理：用 zmq.Poller.poll(1000ms) 替代忙等，
    # 空闲期间让出 CPU 时间片，适合长时间无请求的低负载场景

    def __init__(self, sockets):
        self.poller = zmq.Poller()
        self.last_empty_time = real_time()
        for s in sockets:
            self.poller.register(s, zmq.POLLIN)  # 监听所有输入 socket

        self.empty_cache_interval = envs.SGLANG_EMPTY_CACHE_INTERVAL.get()  # 定期清空 CUDA cache 的间隔（秒）

    def maybe_sleep(self):
        self.poller.poll(1000)  # 最多等待 1 秒（有数据时立即返回）
        if (
            self.empty_cache_interval > 0
            and real_time() - self.last_empty_time > self.empty_cache_interval
        ):
            # 超过清空间隔：调用 cuda.empty_cache() 释放碎片化显存
            self.last_empty_time = real_time()
            torch.cuda.empty_cache()


# -------- 辅助函数：判断是否为健康检查请求 --------
def is_health_check_generate_req(recv_req):
    rid = getattr(recv_req, "rid", None)
    # 健康检查请求的 rid 以特殊前缀开头（避免走完整的 GPU forward）
    return rid is not None and rid.startswith(HEALTH_CHECK_RID_PREFIX)


# -------- 辅助函数：判断是否为工作请求（generate/embedding，而非控制命令）--------
def is_work_request(recv_req):
    return isinstance(
        recv_req,
        (
            TokenizedGenerateReqInput,
            TokenizedEmbeddingReqInput,
            BatchTokenizedGenerateReqInput,
            BatchTokenizedEmbeddingReqInput,
        ),
    )


# -------- SenderWrapper：封装 ZMQ socket 的发送操作，自动处理 multi-http worker 的 IPC 路由 --------
class SenderWrapper:
    def __init__(self, socket: zmq.Socket):
        self.socket = socket  # ZMQ send socket（可为 None，此时所有发送操作为 no-op）

    def send_output(
        self,
        output: Union[BaseReq, BaseBatchReq],
        recv_obj: Optional[Union[BaseReq, BaseBatchReq]] = None,
    ):
        if self.socket is None:
            return  # socket 未初始化（如 pp_rank != 0 的 stage），跳过发送

        if (
            isinstance(recv_obj, BaseReq)
            and recv_obj.http_worker_ipc is not None
            and output.http_worker_ipc is None
        ):
            # handle communicator reqs for multi-http worker case
            # 多 HTTP worker 场景：将原始请求的 IPC 地址复制到输出对象，
            # 确保响应能路由回发出请求的那个 HTTP worker
            output.http_worker_ipc = recv_obj.http_worker_ipc

        self.socket.send_pyobj(output)  # 序列化并发送（pickle）


# -------- dispatch_event_loop：根据解耦模式和并行配置选择合适的事件循环 --------
def dispatch_event_loop(scheduler: Scheduler):
    # Dispatch to the appropriate event loop based on the disaggregation mode
    # 事件循环选择矩阵：
    #   NULL + pdmux          → event_loop_pdmux（prefill/decode 复用模式）
    #   NULL + PP > 1         → event_loop_pp（Pipeline Parallel）
    #   NULL + MLX overlap    → event_loop_overlap_mlx（Apple MLX 后端）
    #   NULL + overlap        → event_loop_overlap（CUDA stream overlap）
    #   NULL                  → event_loop_normal（最简单的顺序循环）
    #   PREFILL + PP          → event_loop_pp_disagg_prefill
    #   PREFILL + overlap     → event_loop_overlap_disagg_prefill
    #   PREFILL               → event_loop_normal_disagg_prefill
    #   DECODE + PP           → event_loop_pp_disagg_decode
    #   DECODE + overlap      → event_loop_overlap_disagg_decode
    #   DECODE                → event_loop_normal_disagg_decode
    server_args = scheduler.server_args
    disaggregation_mode: DisaggregationMode = scheduler.disaggregation_mode
    if disaggregation_mode == DisaggregationMode.NULL:
        if scheduler.enable_pdmux:
            scheduler.event_loop_pdmux()          # prefill/decode 同节点复用
        elif server_args.pp_size > 1:
            scheduler.event_loop_pp()              # Pipeline Parallel 事件循环
        elif scheduler.enable_overlap_mlx:
            scheduler.event_loop_overlap_mlx()     # Apple MLX overlap 事件循环
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap()         # CUDA stream overlap 事件循环
        else:
            scheduler.event_loop_normal()          # 普通顺序事件循环
    elif disaggregation_mode == DisaggregationMode.PREFILL:
        if server_args.pp_size > 1:
            scheduler.event_loop_pp_disagg_prefill()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap_disagg_prefill()
        else:
            scheduler.event_loop_normal_disagg_prefill()
    elif disaggregation_mode == DisaggregationMode.DECODE:
        if server_args.pp_size > 1:
            scheduler.event_loop_pp_disagg_decode()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap_disagg_decode()
        else:
            scheduler.event_loop_normal_disagg_decode()


# -------- configure_scheduler_process：配置调度器子进程（日志、进程名、CPU 亲和性等）--------
def configure_scheduler_process(
    server_args: ServerArgs,
    gpu_id: int,
    tp_rank: int,
    attn_cp_rank: int,
    moe_dp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
) -> Optional[int]:
    """Configure scheduler worker: logging, process title, etc.

    Returns:
        dp_rank
    """
    kill_itself_when_parent_died()  # 注册父进程死亡信号处理（防止孤儿进程）

    # Generate the logger prefix
    # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
    # 路由器场景：通过环境变量传递 dp_rank（适配动态扩缩容）
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        dp_rank = int(os.environ["SGLANG_DP_RANK"])

    # 构建日志前缀（如 " DP0 PP1 TP2"），用于区分不同并行维度的进程日志
    prefix = ""
    if dp_rank is not None:
        prefix += f" DP{dp_rank}"
    if server_args.pp_size > 1:
        prefix += f" PP{pp_rank}"
    if server_args.attn_cp_size > 1:
        prefix += f" ATTN_CP{attn_cp_rank}"
    if server_args.moe_dp_size > 1:
        prefix += f" MOE_DP{moe_dp_rank}"
    if server_args.tp_size > 1:
        prefix += f" TP{tp_rank}"
    if server_args.ep_size > 1:
        prefix += f" EP{moe_ep_rank}"

    # Config the process
    setproctitle.setproctitle(f"sglang::scheduler{prefix.replace(' ', '_')}")  # 设置进程名（ps 可见）
    faulthandler.enable()  # 启用 C 级别 traceback（SIGSEGV 时打印调用栈）

    # Configure the logger
    configure_logger(server_args, prefix=prefix)  # 初始化带前缀的日志器
    suppress_other_loggers()                       # 屏蔽第三方库的冗余日志

    # Set cpu affinity to this gpu process
    if envs.SGLANG_SET_CPU_AFFINITY.get():
        # 将进程绑定到与 GPU 最近的 CPU 核心（减少跨 NUMA 访问延迟）
        set_gpu_proc_affinity(
            server_args.pp_size, server_args.tp_size, server_args.nnodes, gpu_id
        )
    if not envs.SGLANG_NUMA_BIND_V2.get():
        numa_node = get_numa_node_if_available(server_args, gpu_id)
        if numa_node is not None:
            numa_bind_to_node(numa_node)  # 绑定到对应 NUMA 节点（减少内存访问延迟）

    return dp_rank


# -------- run_scheduler_process：调度器子进程入口函数（由 multiprocessing.Process 调用）--------
def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    attn_cp_rank: int,
    moe_dp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,  # 与父进程通信的 Pipe 写端（用于发送初始化完成信号）
):
    # Load plugins so hooks can override Scheduler and its dependencies.
    load_plugins()  # 加载插件（允许用户通过 hook 覆盖 Scheduler 行为）
    dp_rank = configure_scheduler_process(
        server_args,
        gpu_id,
        tp_rank,
        attn_cp_rank,
        moe_dp_rank,
        moe_ep_rank,
        pp_rank,
        dp_rank,
    )
    parent_process = psutil.Process().parent()  # 记录父进程（用于异常时发送 SIGQUIT）

    # Set up tracing
    if server_args.enable_trace:
        # 初始化 OpenTelemetry 追踪（发送到 OTLP endpoint）
        process_tracing_init(server_args.otlp_traces_endpoint, "sglang")
        thread_label = "Scheduler"
        if server_args.disaggregation_mode == "prefill":
            thread_label = "Prefill Scheduler"
        elif server_args.disaggregation_mode == "decode":
            thread_label = "Decode Scheduler"
        trace_set_thread_info(thread_label, tp_rank, dp_rank, pp_rank)  # 设置线程级追踪信息

    # Create a scheduler and run the event loop
    try:
        scheduler = Scheduler(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            attn_cp_rank,
            moe_dp_rank,
            dp_rank,
        )

        # Send initialization info back to the parent process
        # 发送初始化完成信号（含 max_total_num_tokens、max_req_input_len 等）
        pipe_writer.send(scheduler.get_init_info())

        # Run the event loop (blocks until shutdown)
        scheduler.run_event_loop()  # 阻塞，直到进程被终止

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)  # 通知父进程（launcher）终止整个服务器
