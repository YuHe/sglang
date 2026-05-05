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
"""TokenizerManager is a process that tokenizes the text."""

# ——— 标准库 ———
import asyncio  # 标准库 — 异步IO事件循环
import copy  # 标准库 — 深/浅拷贝对象
import dataclasses  # 标准库 — 数据类装饰器
import json  # 标准库 — JSON序列化/反序列化
import logging  # 标准库 — 日志记录
import os  # 标准库 — 操作系统接口（环境变量等）
import pickle  # 标准库 — Python对象序列化
import signal  # 标准库 — 进程信号处理
import socket  # 标准库 — 网络套接字
import sys  # 标准库 — Python运行时系统接口
import threading  # 标准库 — 多线程支持
from collections import deque  # 标准库 — 双端队列，用于崩溃转储缓冲
from contextlib import nullcontext  # 标准库 — 空上下文管理器（占位用）
from datetime import datetime  # 标准库 — 日期时间处理
from enum import Enum  # 标准库 — 枚举类型
from http import HTTPStatus  # 标准库 — HTTP状态码常量
from typing import Any, Awaitable, Dict, List, Optional, Tuple, Union  # 标准库 — 类型注解

# ——— 第三方库 ———
import fastapi  # fastapi — Web框架，用于HTTP请求对象类型注解
import pybase64  # pybase64 — 高性能Base64编解码（用于图像数据）
import torch  # torch — PyTorch深度学习框架
import uvloop  # uvloop — 高性能异步IO事件循环（替代asyncio默认循环）
import zmq  # zmq — ZeroMQ进程间通信库
import zmq.asyncio  # zmq.asyncio — ZMQ的异步IO适配层
from fastapi import BackgroundTasks  # fastapi — 后台任务注入类

# ——— SGLang 内部模块 ———
from sglang.srt.configs.model_config import ModelConfig  # 模型配置类（context_len/dtype等）
from sglang.srt.disaggregation.encode_receiver import create_mm_receiver  # 编码器分离模式的多模态接收器
from sglang.srt.disaggregation.utils import DisaggregationMode  # PD解耦模式枚举（prefill/decode分离）
from sglang.srt.environ import envs  # 全局环境变量读取接口
from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry  # LoRA适配器引用与注册表
from sglang.srt.managers.async_dynamic_batch_tokenizer import AsyncDynamicbatchTokenizer  # 异步动态批次分词器
from sglang.srt.managers.disagg_service import start_disagg_service  # 启动PD解耦引导服务
from sglang.srt.managers.embed_types import PositionalEmbeds  # 位置编码类型定义
from sglang.srt.managers.io_struct import (  # IO数据结构 — 各种请求/响应数据类
    AbortReq,  # 中止请求
    ActiveRanksOutput,  # 活跃rank输出
    BatchEmbeddingOutput,  # 批次嵌入输出
    BatchStrOutput,  # 批次字符串输出
    BatchTokenIDOutput,  # 批次Token ID输出
    BatchTokenizedEmbeddingReqInput,  # 已分词的批次嵌入请求
    BatchTokenizedGenerateReqInput,  # 已分词的批次生成请求
    ConfigureLoggingReq,  # 配置日志请求
    ContinueGenerationReqInput,  # 继续生成请求（用于分步生成）
    EmbeddingReqInput,  # 嵌入向量请求输入
    FreezeGCReq,  # 冻结GC请求
    GenerateReqInput,  # 文本生成请求输入
    HealthCheckOutput,  # 健康检查输出
    LoadLoRAAdapterReqInput,  # 加载LoRA适配器请求
    OpenSessionReqOutput,  # 打开会话输出
    PauseGenerationReqInput,  # 暂停生成请求
    SessionParams,  # 会话参数
    TokenizedEmbeddingReqInput,  # 已分词的嵌入请求
    TokenizedGenerateReqInput,  # 已分词的生成请求
    UpdateWeightFromDiskReqInput,  # 从磁盘更新权重请求
    UpdateWeightFromDiskReqOutput,  # 从磁盘更新权重输出
    WatchLoadUpdateReq,  # 观察加载更新请求
)
from sglang.srt.managers.mm_utils import TensorTransportMode, wrap_shm_features  # 多模态张量传输模式与共享内存包装
from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors  # 多模态处理器工厂与导入
from sglang.srt.managers.schedule_batch import MultimodalDataItem  # 多模态数据条目（图像/音频等）
from sglang.srt.managers.scheduler import is_health_check_generate_req  # 判断是否为健康检查生成请求
from sglang.srt.managers.scheduler_input_blocker import input_blocker_guard_region  # 调度器输入阻塞器守卫区
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin  # 分词器控制混入类（暂停/继续等控制接口）
from sglang.srt.managers.tokenizer_manager_score_mixin import (  # 分词器打分混入类（rerank/score接口）
    TokenizerManagerScoreMixin,
)
from sglang.srt.observability.cpu_monitor import start_cpu_monitor_thread  # 启动CPU监控后台线程
from sglang.srt.observability.metrics_collector import TokenizerMetricsCollector  # 分词器指标收集器（Prometheus）
from sglang.srt.observability.req_time_stats import (  # 请求时间统计工具
    APIServerReqTimeStats,  # API服务器请求时间统计数据类
    convert_time_to_realtime,  # 转换时间为实时时间戳
    real_time,  # 获取当前实时时间戳
    set_time_batch,  # 批量设置时间戳
)
from sglang.srt.observability.request_metrics_exporter import (  # 请求指标导出管理器
    RequestMetricsExporterManager,
)
from sglang.srt.observability.trace import SpanAttributes, extract_trace_headers  # 分布式追踪span属性与header提取
from sglang.srt.sampling.sampling_params import SamplingParams  # 采样参数（temperature/top_p等）
from sglang.srt.server_args import (  # 服务器参数
    PortArgs,  # 端口参数（各IPC通道地址）
    ServerArgs,  # 完整服务器启动参数
    set_global_server_args_for_tokenizer,  # 将ServerArgs设置为全局单例（供分词器模块访问）
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # 推测解码算法枚举（eagle等）
from sglang.srt.utils import (  # 通用工具函数
    configure_gc_warning,  # 配置GC耗时警告阈值
    freeze_gc,  # 冻结垃圾回收
    get_bool_env_var,  # 读取布尔型环境变量
    get_or_create_event_loop,  # 获取或创建asyncio事件循环
    kill_process_tree,  # 递归杀死进程树
)
from sglang.srt.utils.aio_rwlock import RWLock  # 异步读写锁（用于模型权重更新）
from sglang.srt.utils.hf_transformers_utils import (  # HuggingFace transformers工具
    get_processor,  # 加载多模态处理器
    get_tokenizer,  # 加载分词器
    get_tokenizer_from_processor,  # 从处理器中提取分词器
)
from sglang.srt.utils.network import get_zmq_socket  # 创建ZMQ socket（含绑定/连接逻辑）
from sglang.srt.utils.request_logger import RequestLogger  # 请求日志记录器
from sglang.srt.utils.watchdog import Watchdog  # 看门狗（检测事件循环卡死）
from sglang.utils import TypeBasedDispatcher, get_exception_traceback  # 基于类型的消息分发器 + 异常堆栈获取

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())  # 将asyncio默认事件循环替换为uvloop（C实现，性能更高）

_REQUEST_STATE_WAIT_TIMEOUT = envs.SGLANG_REQUEST_STATE_WAIT_TIMEOUT.get()  # 请求状态等待超时时间（秒），从环境变量读取

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

# 流式增量输出时需要按偏移量切片的元数据字段名元组
# 这些字段在每次增量响应中只返回新增部分而非全量
_INCREMENTAL_STREAMING_META_INFO_KEYS = (
    "output_token_logprobs",  # 输出token的对数概率列表
    "output_top_logprobs",  # 输出token的top-k对数概率列表
    "output_token_ids_logprobs",  # 输出token ID的对数概率列表
)


@dataclasses.dataclass
# 单条请求的完整运行时状态容器，贯穿请求从接收到返回的整个生命周期
class ReqState:
    """Store the state a request."""

    out_list: List[Dict[Any, Any]]  # 输出缓冲列表，每条增量响应字典依次追加
    finished: bool  # 请求是否已完成（收到finish_reason后置True）
    event: asyncio.Event  # 异步通知事件，新输出到达时由handle_loop协程set()
    obj: Union[GenerateReqInput, EmbeddingReqInput]  # 原始请求对象（含prompt/参数）

    # For performance metrics
    time_stats: APIServerReqTimeStats  # 请求各阶段时间戳统计（TTFT/E2E延迟等）
    last_completion_tokens: int = 1  # 上次指标采样时的已完成token数（用于增量计算吞吐量）
    ttft_observed: bool = False  # 是否已记录首token延迟（TTFT），避免重复记录

    # For streaming output
    last_output_offset: int = 0  # 流式输出已发送的token数偏移量（用于增量截取meta_info）

    # Accumulate text lazily so incremental streaming can emit the incoming
    # delta directly without rebuilding the full output prefix.
    text: str = ""  # 已合并的文本输出（懒惰合并，减少字符串反复拼接开销）
    text_chunks: List[str] = dataclasses.field(default_factory=list)  # 待合并的文本块缓冲列表

    # 追加文本块到缓冲（非空才追加，避免空字符串污染列表）
    def append_text(self, chunk: str):
        if chunk:
            self.text_chunks.append(chunk)  # 暂存新增chunk，延迟触发合并

    # 获取当前完整文本（懒惰合并所有缓冲块）
    def get_text(self) -> str:
        if self.text_chunks:
            self.text += "".join(self.text_chunks)  # 一次性拼接所有缓冲块，比逐次+=高效
            self.text_chunks.clear()  # 清空缓冲，避免下次重复计算
        return self.text  # 返回最新的完整文本字符串

    # 生成崩溃转储输出摘要（仅包含有内容的字段）
    def get_crash_dump_output(self) -> Dict[Any, Any]:
        out = {}
        if self.text or self.text_chunks:
            out["text"] = self.get_text()  # 触发懒惰合并并写入文本
        if self.output_ids:
            out["output_ids"] = self.output_ids.copy()  # 写入已生成的token ID列表副本
        return out

    # For incremental state update.
    # TODO(lianmin): do not initialize some lists if not needed.
    output_ids: List[int] = dataclasses.field(default_factory=list)  # 已生成的输出token ID序列（逐步追加）
    input_token_logprobs_val: List[float] = dataclasses.field(default_factory=list)  # 输入token对数概率值
    input_token_logprobs_idx: List[int] = dataclasses.field(default_factory=list)  # 输入token对数概率对应token ID
    output_token_logprobs_val: List[float] = dataclasses.field(default_factory=list)  # 输出token对数概率值
    output_token_logprobs_idx: List[int] = dataclasses.field(default_factory=list)  # 输出token对数概率对应token ID
    input_top_logprobs_val: List[List[float]] = dataclasses.field(default_factory=list)  # 输入每位置top-k对数概率值
    input_top_logprobs_idx: List[List[int]] = dataclasses.field(default_factory=list)  # 输入每位置top-k token ID
    output_top_logprobs_val: List[List[float]] = dataclasses.field(default_factory=list)  # 输出每位置top-k对数概率值
    output_top_logprobs_idx: List[List[int]] = dataclasses.field(default_factory=list)  # 输出每位置top-k token ID
    input_token_ids_logprobs_val: List = dataclasses.field(default_factory=list)  # 输入token ID级对数概率值
    input_token_ids_logprobs_idx: List = dataclasses.field(default_factory=list)  # 输入token ID级对数概率索引
    output_token_ids_logprobs_val: List = dataclasses.field(default_factory=list)  # 输出token ID级对数概率值
    output_token_ids_logprobs_idx: List = dataclasses.field(default_factory=list)  # 输出token ID级对数概率索引

    # For detokenized logprobs
    input_token_logprobs: List[Any] = dataclasses.field(default_factory=list)  # 反分词后的输入token对数概率列表
    output_token_logprobs: List[Any] = dataclasses.field(default_factory=list)  # 反分词后的输出token对数概率列表
    input_top_logprobs: List[Any] = dataclasses.field(default_factory=list)  # 反分词后的输入top-k对数概率列表
    output_top_logprobs: List[Any] = dataclasses.field(default_factory=list)  # 反分词后的输出top-k对数概率列表
    input_token_ids_logprobs: List[Any] = dataclasses.field(default_factory=list)  # 反分词后的输入token ID级对数概率
    output_token_ids_logprobs: List[Any] = dataclasses.field(default_factory=list)  # 反分词后的输出token ID级对数概率


# 将流式增量输出的元数据按当前偏移量切片，使其只包含新增部分
def _slice_streaming_output_meta_info(
    meta_info: Dict[Any, Any],
    last_output_offset: int,
) -> None:
    """Align output-side metadata with the current incremental streaming chunk."""
    for key in meta_info.keys() & set(_INCREMENTAL_STREAMING_META_INFO_KEYS):  # 仅处理需要增量切片的字段
        meta_info[key] = meta_info[key][last_output_offset:]  # 从上次偏移量开始截取，丢弃已发送的前缀


# 输入文本格式枚举，用于分词前判断如何处理输入
class InputFormat(Enum):
    """Input format types for tokenization handling."""

    SINGLE_STRING = 1  # 单条字符串，如 "Hello world"
    BATCH_STRINGS = 2  # 字符串列表，如 ["Hello", "World"]
    CROSS_ENCODER_PAIRS = 3  # 交叉编码器对，如 [["query", "document"]]


# 分词器管理器主类：继承控制混入（暂停/继续/健康检查）和打分混入（rerank/score），负责分词、请求路由和响应聚合
class TokenizerManager(TokenizerControlMixin, TokenizerManagerScoreMixin):
    """TokenizerManager is a process that tokenizes the text."""

    # 构造函数：按依赖顺序逐步初始化各功能子模块
    def __init__(
        self,
        server_args: ServerArgs,  # 完整服务器启动参数（模型路径/端口/特性开关等）
        port_args: PortArgs,  # IPC通道的端口/Socket路径参数
    ):
        # Parse args
        self.server_args = server_args  # 保存服务器参数供各初始化方法使用
        self.enable_metrics = server_args.enable_metrics  # 是否启用Prometheus指标收集
        self.preferred_sampling_params = server_args.preferred_sampling_params  # 用户指定的采样参数默认覆盖值
        self.crash_dump_folder = server_args.crash_dump_folder  # 崩溃时输出转储文件的目录
        set_global_server_args_for_tokenizer(server_args)  # 注册全局单例，让分词器工具函数无需传参即可访问

        # Init model config
        self.init_model_config()  # 初始化模型配置（上下文长度/图像token/推测解码保留槽位等）

        # Initialize tokenizer and multimodalprocessor
        self.init_tokenizer_and_processor()  # 初始化HuggingFace分词器和多模态图像/音频处理器

        # Init inter-process communication
        self.init_ipc_channels(port_args)  # 创建ZMQ PUSH/PULL socket，连接Scheduler和Detokenizer进程

        # Init running status
        self.init_running_status()  # 初始化请求状态字典、服务器健康状态、会话表等运行时变量

        # Init logging and dumping
        self.init_request_logging_and_dumping()  # 初始化请求日志器和崩溃/慢请求转储缓冲区

        # Init weight update
        self.init_weight_update()  # 初始化在线权重更新的读写锁和暂停/恢复条件变量

        # Init LoRA status
        self.init_lora()  # 初始化LoRA注册表、序列化更新锁和适配器引用缓存字典

        # Init PD disaggregation and encoder disaggregation
        self.init_disaggregation()  # 初始化Prefill-Decode解耦模式和编码器解耦的多模态接收器

        # Init metric collector and watchdog
        self.init_metric_collector_watchdog()  # 初始化Prometheus指标收集器、CPU监控线程和事件循环看门狗

        # Init request dispatcher
        self.init_request_dispatcher()  # 初始化基于消息类型的分发器（将不同响应类型路由到对应处理函数）

    # 初始化模型配置：读取模型元数据（上下文长度/图像token/推测解码参数等）
    def init_model_config(self):
        server_args = self.server_args  # 局部引用，减少self查找开销
        model_config_class = getattr(self, "model_config_class", ModelConfig)  # 允许子类覆盖配置类

        # Read model args
        self.model_path = server_args.model_path  # 模型权重路径或HuggingFace模型名
        self.served_model_name = server_args.served_model_name  # 对外暴露的模型名（用于API响应）
        self.model_config = model_config_class.from_server_args(server_args)  # 解析HF config.json，构建模型配置
        self.is_generation = self.model_config.is_generation  # 是否为生成模型（True）还是嵌入模型（False）
        self.context_len = self.model_config.context_len  # 模型最大上下文长度（token数）
        self.image_token_id = self.model_config.image_token_id  # 图像占位符token的ID（多模态模型）
        self.max_req_input_len = None  # 最大请求输入长度，由engine.py在后续步骤中设置
        self.enable_priority_scheduling = server_args.enable_priority_scheduling  # 是否启用优先级调度
        self.default_priority_value = server_args.default_priority_value  # 默认请求优先级数值
        speculative_algorithm = SpeculativeAlgorithm.from_string(  # 解析推测解码算法枚举
            server_args.speculative_algorithm  # 字符串形式的算法名（eagle/medusa/none等）
        )
        if speculative_algorithm.is_eagle():  # Eagle算法需要预留额外token槽位存放草稿token
            # In the current eagle implementation, we store the draft tokens in the output token slots,
            # so we need to reserve the space for the draft tokens.
            # Eagle实现将草稿token存储在输出slot中，因此需要预留足够槽位
            self.num_reserved_tokens = max(
                server_args.speculative_eagle_topk * server_args.speculative_num_steps,  # topk候选 × 推测步数
                server_args.speculative_num_draft_tokens,  # 或直接指定的草稿token数
            )
        else:
            self.num_reserved_tokens = 0  # 非Eagle算法无需预留额外token槽位
        self.validate_total_tokens = True  # 是否在tokenize时校验总token数不超过限制

    # 初始化分词器和多模态处理器：根据模型类型（多模态/纯文本）分别加载对应组件
    def init_tokenizer_and_processor(self):
        server_args = self.server_args  # 局部引用，便于后续多次访问

        # Initialize tokenizer and processor
        if self.model_config.is_multimodal:  # 多模态模型（含图像/音频等）的初始化路径
            import_processors("sglang.srt.multimodal.processors")  # 动态导入内置多模态处理器注册表
            if mm_process_pkg := envs.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE.get():  # 若设置了外部处理器包
                import_processors(mm_process_pkg, overwrite=True)  # 导入并覆盖内置处理器（用于自定义模型）
            _processor = _get_processor_wrapper(server_args)  # 加载HuggingFace AutoProcessor（含tokenizer+图像处理）
            transport_mode = _determine_tensor_transport_mode(self.server_args)  # 决定张量传输模式（共享内存/ZMQ等）

            # We want to parallelize the image pre-processing so we create an executor for it
            # We create mm_processor for any skip_tokenizer_init to make sure we still encode
            # images even with skip_tokenizer_init=False.
            # 即使skip_tokenizer_init也要创建mm_processor，确保图像编码始终可用
            self.mm_processor = get_mm_processor(  # 创建多模态处理器（含图像预处理线程池）
                self.model_config.hf_config,  # HuggingFace模型配置（含图像分辨率等参数）
                server_args,  # 服务器参数（含处理器配置）
                _processor,  # HuggingFace处理器实例
                transport_mode,  # 张量传输模式（影响图像特征如何传给Scheduler）
                model_config=self.model_config,  # 模型配置（类型信息等）
            )

            if server_args.skip_tokenizer_init:  # 跳过分词器初始化模式（推断时由外部提供token ID）
                self.tokenizer = self.processor = None  # 不加载分词器和处理器，节省内存
            else:
                self.processor = _processor  # 保存HuggingFace处理器引用
                self.tokenizer = get_tokenizer_from_processor(self.processor)  # 从处理器中提取分词器组件
                os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用HuggingFace tokenizer内部并行，避免与进程池冲突
        else:  # 纯文本模型的初始化路径
            self.mm_processor = self.processor = None  # 无多模态处理器

            if server_args.skip_tokenizer_init:  # 跳过分词器初始化
                self.tokenizer = None  # 不加载分词器（由外部提供token ID）
            else:
                self.tokenizer = get_tokenizer(  # 加载标准HuggingFace分词器
                    server_args.tokenizer_path,  # 分词器路径（可与模型路径不同）
                    tokenizer_mode=server_args.tokenizer_mode,  # 分词器模式（auto/slow/mistral等）
                    trust_remote_code=server_args.trust_remote_code,  # 是否信任远程自定义代码
                    revision=server_args.revision,  # 模型版本/分支/commit hash
                    tokenizer_backend=server_args.tokenizer_backend,  # 后端实现（huggingface/sentencepiece等）
                )

        # Initialize async dynamic batch tokenizer if enabled (common for both multimodal and non-multimodal)
        # 若启用动态批次分词器（异步攒批提升分词吞吐），则创建实例；否则置None
        if (
            server_args.enable_dynamic_batch_tokenizer  # 功能开关
            and not server_args.skip_tokenizer_init  # 跳过初始化时无需创建
        ):
            self.async_dynamic_batch_tokenizer = AsyncDynamicbatchTokenizer(  # 创建异步动态批次分词器实例
                self.tokenizer,  # 底层分词器实例
                max_batch_size=server_args.dynamic_batch_tokenizer_batch_size,  # 最大攒批大小
                batch_wait_timeout_s=server_args.dynamic_batch_tokenizer_batch_timeout,  # 攒批等待超时（秒）
            )
        else:
            self.async_dynamic_batch_tokenizer = None  # 不使用动态批次分词，每次单独分词

    # 初始化ZMQ进程间通信通道：建立与Detokenizer（接收）和Scheduler（发送）的socket连接
    def init_ipc_channels(self, port_args: PortArgs):
        context = zmq.asyncio.Context(2)  # 创建ZMQ异步上下文，2个IO线程
        self.recv_from_detokenizer = get_zmq_socket(
            context, zmq.PULL, port_args.tokenizer_ipc_name, True  # PULL socket，接收Detokenizer发来的解码结果
        )
        if self.server_args.tokenizer_worker_num == 1:  # 单分词器worker模式（常规部署）
            self.send_to_scheduler = get_zmq_socket(  # 创建PUSH socket直接发给Scheduler
                context, zmq.PUSH, port_args.scheduler_input_ipc_name, True  # PUSH socket，直接发送给Scheduler
            )
        else:  # 多分词器worker模式（高并发HTTP服务）
            from sglang.srt.managers.multi_tokenizer_mixin import SenderWrapper

            # Use tokenizer_worker_ipc_name in multi-tokenizer mode
            # 多worker模式使用专用IPC地址，通过路由器转发到具体的tokenizer worker
            send_to_scheduler = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_worker_ipc_name, False  # PUSH到多worker路由器
            )

            # Make sure that each request carries the tokenizer_ipc_name for response routing
            # 包装器确保每条请求携带来源tokenizer_ipc_name，以便Detokenizer能正确路由响应
            self.send_to_scheduler = SenderWrapper(port_args, send_to_scheduler)

    # 初始化运行时状态变量：请求状态字典、服务器健康状态、会话表和子进程看门狗
    def init_running_status(self):
        # Request states
        self.rid_to_state: Dict[str, ReqState] = {}  # 请求ID→ReqState映射表，存储所有活跃请求的状态
        self.event_loop = None  # asyncio事件循环引用，在handle_loop_wrapper中赋值
        self.asyncio_tasks = set()  # 持有所有活跃asyncio Task引用，防止被GC提前回收

        # Health check
        self.server_status = ServerStatus.Starting  # 服务器当前状态枚举（Starting/Ready/Exiting）
        self.gracefully_exit = False  # 收到优雅退出信号后置True，等待所有请求完成后退出
        self.last_receive_tstamp = real_time()  # 最后一次收到消息的时间戳（供看门狗检测进程卡死）

        # Session
        self.session_futures = {}  # session_id → asyncio事件，用于多轮对话的会话状态同步

        # Subprocess liveness watchdog — set by Engine or http_server after construction
        self._subprocess_watchdog = None  # 子进程存活看门狗，由Engine或http_server在构建完成后注入

    # 初始化请求日志记录和转储功能：配置日志器、崩溃转储缓冲和性能指标导出器
    def init_request_logging_and_dumping(self):
        # TODO: Refactor and organize the log export code.
        # Request logging
        self.request_logger = RequestLogger(  # 请求日志记录器（记录输入/输出内容）
            log_requests=self.server_args.log_requests,  # 是否启用请求日志
            log_requests_level=self.server_args.log_requests_level,  # 日志详细级别
            log_requests_format=self.server_args.log_requests_format,  # 日志输出格式
            log_requests_target=self.server_args.log_requests_target,  # 日志输出目标（文件/stdout等）
        )

        # Dumping
        self.dump_requests_folder = ""  # 请求转储输出目录，默认为空表示不转储
        self.dump_requests_threshold = 1000  # 转储触发阈值（积累N条后写入文件）
        self.dump_request_list: List[Tuple] = []  # 待转储的请求列表缓冲
        self.crash_dump_request_list: deque[Tuple] = deque()  # 崩溃转储的请求双端队列（滑动窗口缓存最近请求）
        self.crash_dump_performed = False  # 崩溃转储是否已执行标志，确保只转储一次
        self.straggler_request_list: List[Tuple] = []  # 慢请求（straggler）记录列表

        # Initialize performance metrics loggers with proper skip names
        _, obj_skip_names, out_skip_names = self.request_logger.metadata  # 从日志器获取需要跳过的字段名
        self.request_metrics_exporter_manager = RequestMetricsExporterManager(  # 请求指标导出管理器
            self.server_args, obj_skip_names, out_skip_names  # 传入跳过字段名，避免记录敏感/冗余字段
        )

    # 初始化在线权重更新相关状态：权重加载标志、读写锁和暂停/恢复机制
    def init_weight_update(self):
        # Initial weights status
        self.initial_weights_loaded = True  # 初始权重是否已加载完成的标志
        if self.server_args.checkpoint_engine_wait_weights_before_ready:
            self.initial_weights_loaded = False  # checkpoint引擎模式：启动后等待外部注入权重才标记就绪

        # Weight updates
        # The event to notify the weight sync is finished.
        self.model_update_lock = RWLock()  # 异步读写锁：推理时持读锁，权重更新时持写锁（互斥）
        self.model_update_result: Optional[Awaitable[UpdateWeightFromDiskReqOutput]] = (
            None  # 存储权重更新操作的awaitable结果，更新完成后设置
        )
        self.is_pause = False  # 是否处于暂停状态（在线权重更新期间暂停接受新请求）
        self.is_pause_cond = asyncio.Condition()  # 与is_pause配合的条件变量，用于协程等待/唤醒

    # 初始化LoRA适配器管理：注册表、序列化更新锁和适配器引用缓存
    def init_lora(self):
        # LoRA
        # Initialize the `LoRARegistry` with initial LoRA adapter paths provided in `server_args`.
        # The registry dynamically updates as adapters are loaded / unloaded during runtime. It
        # serves as the source of truth for available adapters and maps user-friendly LoRA names
        # to internally used unique LoRA IDs.
        # 使用启动参数中的初始LoRA路径列表创建注册表，注册表是适配器状态的唯一来源
        self.lora_registry = LoRARegistry(self.server_args.lora_paths)
        # Lock to serialize LoRA update operations.
        # Please note that, unlike `model_update_lock`, this does not block inference, allowing
        # LoRA updates and inference to overlap.
        # 注意：与model_update_lock不同，此锁不阻塞推理，LoRA更新和推理可以并发进行
        self.lora_update_lock = asyncio.Lock()  # 序列化LoRA加载/卸载操作，防止并发修改注册表
        # A cache for mapping the lora_name for LoRA adapters that have been loaded at any
        # point to their latest LoRARef objects, so that they can be
        # dynamically loaded if needed for inference
        # 缓存曾经加载过的适配器名→LoRARef映射，用于推理时按名称动态解析适配器ID
        self.lora_ref_cache: Dict[str, LoRARef] = {}
        if self.server_args.lora_paths is not None:  # 若启动时指定了初始LoRA适配器
            for lora_ref in self.server_args.lora_paths:
                self.lora_ref_cache[lora_ref.lora_name] = lora_ref  # 预填充缓存，加速首次推理时的名称解析

    # 初始化PD解耦（Prefill-Decode分离）和编码器解耦模式相关组件
    def init_disaggregation(self):
        # PD Disaggregation
        self.disaggregation_mode = DisaggregationMode(
            self.server_args.disaggregation_mode  # 解耦模式字符串（none/prefill/decode）
        )
        self.bootstrap_server = start_disagg_service(self.server_args)  # 启动PD解耦引导服务（用于节点间协调）
        # Single-source counter for auto-assigning fake bootstrap_room.
        self.fake_bootstrap_room_counter = 0  # 自动分配虚拟bootstrap_room ID的单调递增计数器

        # Encoder Disaggregation
        if self.server_args.language_only:  # 仅语言模式：多模态特征由独立编码器节点提供
            self.mm_receiver = create_mm_receiver(  # 创建多模态特征接收器（通过ZMQ接收编码器节点发来的图像特征）
                self.server_args,  # 服务器参数（含接收器配置）
                dtype=self.model_config.dtype,  # 特征张量的数据类型与模型一致
            )

    # 初始化Prometheus指标收集器（含标签配置）和事件循环软看门狗
    def init_metric_collector_watchdog(self):
        # Metrics
        if self.enable_metrics:  # 仅在启用指标时创建收集器（避免无效开销）
            engine_type = DisaggregationMode.to_engine_type(
                self.server_args.disaggregation_mode  # 将解耦模式转换为engine_type标签值（prefill/decode/none）
            )

            labels = {
                "model_name": self.server_args.served_model_name,  # Prometheus标签：模型名称
                "engine_type": engine_type,  # Prometheus标签：引擎类型（用于区分PD节点）
            }
            if self.enable_priority_scheduling:
                labels["priority"] = ""  # 启用优先级调度时添加priority维度标签
            if self.server_args.tokenizer_metrics_allowed_custom_labels:
                for label in self.server_args.tokenizer_metrics_allowed_custom_labels:
                    labels[label] = ""  # 添加用户自定义的Prometheus标签维度
            if self.server_args.extra_metric_labels:
                labels.update(self.server_args.extra_metric_labels)  # 合并额外的固定标签键值对
            self.metrics_collector = TokenizerMetricsCollector(  # 分词器指标收集器（TTFT/E2E延迟/吞吐量等）
                server_args=self.server_args,
                labels=labels,
                bucket_time_to_first_token=self.server_args.bucket_time_to_first_token,  # TTFT直方图桶边界
                bucket_e2e_request_latency=self.server_args.bucket_e2e_request_latency,  # E2E延迟直方图桶边界
                bucket_inter_token_latency=self.server_args.bucket_inter_token_latency,  # 逐token延迟直方图桶边界
            )

            start_cpu_monitor_thread("tokenizer")  # 启动CPU使用率监控后台线程（定期采样并暴露给Prometheus）

        if self.server_args.gc_warning_threshold_secs > 0.0:
            configure_gc_warning(self.server_args.gc_warning_threshold_secs)  # 配置GC耗时超阈值警告
        self.soft_watchdog = Watchdog.create(
            debug_name="TokenizerManager",  # 看门狗标识名（用于日志中区分不同组件）
            watchdog_timeout=self.server_args.soft_watchdog_timeout,  # 看门狗超时时间（秒）
            soft=True,  # 软看门狗：超时后仅告警而不强制终止进程
            test_stuck_time=envs.SGLANG_TEST_STUCK_TOKENIZER.get(),  # 测试用：模拟卡死时间（正常为None）
        )

    # 初始化消息类型分发器：将Detokenizer发来的不同类型响应路由到对应处理函数
    def init_request_dispatcher(self):
        self._result_dispatcher = TypeBasedDispatcher(  # 基于Python类型的消息分发器（类似switch-case）
            [
                (AbortReq, self._handle_abort_req),  # 中止请求：清理rid_to_state并通知等待协程
                (OpenSessionReqOutput, self._handle_open_session_req_output),  # 会话打开响应：解锁等待的会话future
                (
                    UpdateWeightFromDiskReqOutput,
                    self._handle_update_weights_from_disk_req_output,  # 权重更新完成响应：设置更新结果并释放写锁
                ),
                (FreezeGCReq, lambda x: None),  # 冻结GC请求：直接忽略（控制流已在其他地方处理）
                # For handling case when scheduler skips detokenizer and forwards back to the tokenizer manager, we ignore it.
                # 调度器绕过detokenizer直接回传healthcheck时忽略（不影响业务逻辑）
                (HealthCheckOutput, lambda x: None),  # 健康检查输出：忽略
                (ActiveRanksOutput, self.update_active_ranks),  # 活跃rank列表更新（DP模式下动态调整路由）
            ]
        )
        self.init_communicators(self.server_args)  # 初始化控制混入中的通信通道（健康检查/控制指令等）

        self.sampling_params_class = SamplingParams  # 采样参数类引用（允许子类替换为自定义实现）
        self.signal_handler_class = SignalHandler  # 信号处理器类引用（允许子类替换）

    # 生成请求全链路入口（async generator）：分词→发送→等待响应→流式yield输出
    async def generate_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],  # 生成请求或嵌入请求输入对象
        request: Optional[fastapi.Request] = None,  # FastAPI原始请求对象（用于客户端断连检测）
    ):
        # 全链路入口：HTTP handler 调用此 async generator
        # 它是一个 async generator 函数（含 yield），每收到一个新 token 就 yield 一次
        # FastAPI 把每次 yield 的内容写入 SSE 流，实现流式输出
        self.auto_create_handle_loop()  # 确保handle_loop协程已在事件循环中运行（懒初始化，只创建一次）

        # Normalize the request
        obj.normalize_batch_and_arguments()  # 标准化请求：展开batch、填充默认参数
        self._set_default_priority(obj)  # 若请求未设置优先级，使用服务器默认优先级值

        if isinstance(obj, GenerateReqInput) and obj.routed_dp_rank is not None:
            dp_size = self.server_args.dp_size  # 数据并行总数
            if dp_size <= 1 and obj.routed_dp_rank == 0:
                logger.warning(  # dp_size=1时routed_dp_rank无意义，记录警告但不报错
                    f"routed_dp_rank={obj.routed_dp_rank} is ignored because dp_size={dp_size}"
                )
            elif obj.routed_dp_rank < 0 or obj.routed_dp_rank >= dp_size:
                raise ValueError(  # routed_dp_rank越界，直接报错
                    f"routed_dp_rank={obj.routed_dp_rank} out of range [0, {dp_size})"
                )

        self._init_req_state(obj, request)  # 为此请求创建ReqState，存入rid_to_state[rid]映射表
        # ReqState包含：asyncio.Event（新输出通知）、out_list（输出缓冲）、time_stats等
        # rid是请求的唯一ID（UUID字符串），全链路用它关联请求
        if self.server_args.language_only:
            self._handle_epd_disaggregation_encode_request(obj)  # 编码器解耦模式：转发多模态特征
        if self.server_args.tokenizer_worker_num > 1:
            self._attach_multi_http_worker_info(obj)  # 多worker模式：附加来源worker信息供响应路由

        # Log the request
        self.request_logger.log_received_request(obj, self.tokenizer, request)  # 记录收到的请求（按配置级别）

        async with self.is_pause_cond:
            await self.is_pause_cond.wait_for(lambda: not self.is_pause)  # 若engine暂停（在线权重更新中），此处挂起等待恢复

        async with self.model_update_lock.reader_lock:  # 持读锁：允许并发推理，但阻塞权重更新写锁
            await self._validate_and_resolve_lora(obj)  # 验证LoRA名称合法性并解析为内部LoRA ID

            # Tokenize the request and send it to the scheduler
            if obj.is_single:  # 单条请求（非batch）
                tokenized_obj = await self._tokenize_one_request(obj)  # 调用分词器：text→token_ids；多模态时提取图像特征
                self._send_one_request(tokenized_obj)  # 通过ZMQ PUSH发给Scheduler（pickle序列化，非阻塞）
                async for response in self._wait_one_response(obj, request):
                    yield response  # 挂起等待Detokenizer回传结果，每次有新token即yield给HTTP层
            else:  # batch请求
                async for response in self._handle_batch_request(obj, request):
                    yield response  # 展开为多条单请求并行处理，汇总结果yield

    # 检测输入文本的格式类型，用于后续分词策略选择
    def _detect_input_format(
        self, texts: Union[str, List[str]], is_cross_encoder: bool
    ) -> InputFormat:
        """Detect the format of input texts for proper tokenization handling.

        Returns:
            - InputFormat.SINGLE_STRING: Regular single text like "Hello world"
            - InputFormat.BATCH_STRINGS: Regular batch like ["Hello", "World"]
            - InputFormat.CROSS_ENCODER_PAIRS: Cross-encoder pairs like [["query", "document"]]
        """
        if isinstance(texts, str):
            return InputFormat.SINGLE_STRING  # 纯字符串 → 单条文本格式

        if (
            is_cross_encoder  # 交叉编码器模式
            and len(texts) > 0  # 非空列表
            and isinstance(texts[0], list)  # 第一个元素是列表（文本对）
            and len(texts[0]) == 2  # 每个文本对恰好包含两个元素（query和document）
        ):
            return InputFormat.CROSS_ENCODER_PAIRS  # [[query,doc],...] 格式

        return InputFormat.BATCH_STRINGS  # 默认：[text1,text2,...] 批量文本格式

    def _prepare_tokenizer_input(
        self, texts: Union[str, List[str]], input_format: InputFormat
    ) -> Union[List[str], List[List[str]]]:
        """根据检测到的输入格式，将文本准备为分词器所需的输入形式。"""
        if input_format == InputFormat.SINGLE_STRING:
            return [texts]  # 单字符串包装为列表，供批量处理
        elif input_format == InputFormat.CROSS_ENCODER_PAIRS:
            return texts  # 交叉编码器对已是正确格式：[["query", "doc"]]
        else:  # BATCH_STRINGS
            return texts  # 批量字符串已是正确格式：["text1", "text2"]

    # 根据输入格式从分词器批量输出中提取单条或批量结果
    def _extract_tokenizer_results(
        self,
        input_ids: List[List[int]],  # 分词器输出的二维token ID列表
        token_type_ids: Optional[List[List[int]]],  # 分词器输出的token_type_ids（交叉编码器使用）
        input_format: InputFormat,  # 输入格式枚举（决定如何提取结果）
        original_batch_size: int,  # 原始batch大小（用于判断是否为单条）
    ) -> Union[
        Tuple[List[int], Optional[List[int]]],  # 单条结果：一维token ID + 可选token_type_ids
        Tuple[List[List[int]], Optional[List[List[int]]]],  # 批量结果：二维token ID + 可选token_type_ids
    ]:
        """根据输入格式从分词器输出中提取结果。"""

        # 对单条输入（单字符串或单个交叉编码器对），取第一条结果（解包列表）
        if (
            input_format in [InputFormat.SINGLE_STRING, InputFormat.CROSS_ENCODER_PAIRS]
            and original_batch_size == 1  # 只有原始batch_size=1时才解包，避免误解包多条batch
        ):
            single_input_ids = input_ids[0] if input_ids else []  # 取第一条token ID序列（解包外层列表）
            single_token_type_ids = token_type_ids[0] if token_type_ids else None  # 取对应token_type_ids
            return single_input_ids, single_token_type_ids

        # 真正的批量输入：直接原样返回二维结果
        return input_ids, token_type_ids

    async def _tokenize_texts(
        self, texts: Union[str, List[str]], is_cross_encoder: bool = False
    ) -> Union[
        Tuple[List[int], Optional[List[int]]],
        Tuple[List[List[int]], Optional[List[List[int]]]],
    ]:
        """
        Tokenize text(s) using the appropriate tokenizer strategy.

        This method handles multiple input formats and chooses between async dynamic
        batch tokenizer (for single texts only) and regular tokenizer.

        Args:
            texts: Text input in various formats:

                   Regular cases:
                   - Single string: "How are you?"
                   - Batch of strings: ["Hello", "World", "How are you?"]

                   Cross-encoder cases (sentence pairs for similarity/ranking):
                   - Single pair: [["query text", "document text"]]
                   - Multiple pairs: [["q1", "d1"], ["q2", "d2"], ["q3", "d3"]]

            is_cross_encoder: Whether to return token_type_ids for cross-encoder models.
                             Enables proper handling of sentence pairs with segment IDs.

        Returns:
            Single input cases:
                Tuple[List[int], Optional[List[int]]]: (input_ids, token_type_ids)
                Example: ([101, 2129, 102], [0, 0, 0]) for single text
                Example: ([101, 2129, 102, 4068, 102], [0, 0, 0, 1, 1]) for cross-encoder pair

            Batch input cases:
                Tuple[List[List[int]], Optional[List[List[int]]]]: (batch_input_ids, batch_token_type_ids)
                Example: ([[101, 2129, 102], [101, 4068, 102]], None) for regular batch

            Note: token_type_ids is None unless is_cross_encoder=True.
        """
        if not texts or self.tokenizer is None:
            raise ValueError("texts cannot be empty and tokenizer must be initialized")  # 校验输入合法性

        # 第一步：检测输入格式，准备分词器输入
        input_format = self._detect_input_format(texts, is_cross_encoder)  # 判断输入格式（单串/批量/交叉编码器对）
        tokenizer_input = self._prepare_tokenizer_input(texts, input_format)  # 转换为分词器期望的格式
        original_batch_size = len(texts) if not isinstance(texts, str) else 1  # 记录原始批大小，用于后续结果提取

        # 第二步：构建分词器关键字参数
        tokenizer_kwargs = (
            {"return_token_type_ids": is_cross_encoder} if is_cross_encoder else {}  # 交叉编码器需要 token_type_ids
        )

        # 第三步：选择分词策略（异步动态批分词器 vs 普通分词器）
        use_async_tokenizer = (
            self.async_dynamic_batch_tokenizer is not None  # 异步分词器已初始化
            and input_format == InputFormat.SINGLE_STRING  # 仅支持单字符串格式
        )

        if use_async_tokenizer:
            logger.debug("Using async dynamic batch tokenizer for single text")  # 使用异步动态批分词器
            result = await self.async_dynamic_batch_tokenizer.encode(
                tokenizer_input[0], **tokenizer_kwargs  # 传入第一条文本
            )
            # 转换为批量格式以保持一致性
            input_ids = [result["input_ids"]]  # 包装为列表
            token_type_ids = (
                [result["token_type_ids"]]
                if is_cross_encoder and result.get("token_type_ids")  # 仅交叉编码器时返回
                else None
            )
        else:
            logger.debug(f"Using regular tokenizer for {len(tokenizer_input)} inputs")  # 使用普通分词器
            encoded = self.tokenizer(tokenizer_input, **tokenizer_kwargs)  # 批量分词
            input_ids = encoded["input_ids"]  # 提取 token id 列表
            token_type_ids = encoded.get("token_type_ids") if is_cross_encoder else None  # 交叉编码器时提取 token_type_ids

        # 第四步：根据输入格式提取最终结果
        return self._extract_tokenizer_results(
            input_ids, token_type_ids, input_format, original_batch_size
        )

    # 对单条请求进行分词处理，根据请求类型决定分词策略
    async def _tokenize_one_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
    ):
        """对单条请求进行分词。"""
        # 优先使用外部传入的 embedding 或 token IDs，避免重复 tokenize
        input_embeds = None  # 输入嵌入向量，默认为空
        input_text = obj.text  # 原始文本输入
        token_type_ids = None  # 段落 id，仅交叉编码器使用
        is_cross_encoder_request = (
            isinstance(obj, EmbeddingReqInput) and obj.is_cross_encoder_request  # 判断是否为交叉编码器请求
        )
        if obj.input_embeds is not None:
            # 使用预计算嵌入向量：需要禁用基数缓存
            if not self.server_args.disable_radix_cache:
                raise ValueError(
                    "input_embeds is provided while disable_radix_cache is False. "
                    "Please add `--disable-radix-cache` when you launch the server "
                    "if you want to use input_embeds as inputs."
                )
            input_embeds = obj.input_embeds  # 赋値嵌入向量
            input_ids = obj.input_ids  # 使用配套的 token id
        elif obj.input_ids is not None:
            input_ids = obj.input_ids  # 直接使用预分词的 token id，跳过分词
        else:
            # 需要对文本进行分词
            if self.tokenizer is None:
                raise ValueError(
                    "The engine initialized with skip_tokenizer_init=True cannot "
                    "accept text prompts. Please provide input_ids or re-initialize "
                    "the engine with skip_tokenizer_init=False."
                )

            # 多模态纯音频请求（如 Whisper），multimodal processor 后续会注入 input_ids
            if not input_text and self.mm_processor and obj.contains_mm_input():
                input_ids = []  # 音频类请求暂时留空，后续由 mm_processor 填充
            else:
                input_ids, token_type_ids = await self._tokenize_texts(
                    input_text, is_cross_encoder_request  # 调用统一分词方法
                )

        contains_mm_input = obj.contains_mm_input()  # 检测是否包含多模态输入
        is_mossvl = (
            "MossVLForConditionalGeneration"
            in self.model_config.hf_config.architectures  # 检测是否为 MossVL 模型
        )
        # 多模态输入或 MossVL 模型需要调用 multimodal processor
        should_run_mm_processor = self.mm_processor is not None and (
            contains_mm_input or is_mossvl  # mm_processor 已初始化且存在多模态输入或 MossVL
        )

        if should_run_mm_processor:
            # 确保 image/video/audio 数据为 list 格式，便于统一处理
            if obj.image_data is not None and not isinstance(obj.image_data, list):
                obj.image_data = [obj.image_data]  # 将单张图片包装为列表
            if obj.video_data is not None and not isinstance(obj.video_data, list):
                obj.video_data = [obj.video_data]  # 将单个视频包装为列表
            if obj.audio_data is not None and not isinstance(obj.audio_data, list):
                obj.audio_data = [obj.audio_data]  # 将单条音频包装为列表
            if contains_mm_input:
                self._validate_mm_limits(obj)  # 校验多模态输入数量限制

            mm_inputs = None  # 初始化多模态输入结果

            # language_only 模式下通过 encoder 传输多模态数据时，从 receiver 获取
            if (
                not self.server_args.language_only
                or self.server_args.encoder_transfer_backend
                in ["zmq_to_tokenizer", "mooncake"]  # 支持从 receiver 获取的后端类型
            ):
                if self.server_args.language_only:
                    # language_only 模式：通过 zmq/mooncake 接收多模态数据
                    mm_inputs = await self.mm_receiver.recv_mm_data(
                        request_obj=obj,
                        mm_processor=self.mm_processor,
                        prompt=(input_text or input_ids),
                        need_wait_for_mm_inputs=obj.need_wait_for_mm_inputs,
                    )
                # 普通模式直接本地处理多模态数据
                if mm_inputs is None:
                    mm_inputs = await self.mm_processor.process_mm_data_async(
                        image_data=obj.image_data,
                        audio_data=obj.audio_data,
                        input_text=(input_text or input_ids),
                        request_obj=obj,
                        max_req_input_len=self.max_req_input_len,  # 限制最大输入长度
                    )
            elif (
                self.server_args.language_only
                and self.server_args.encoder_transfer_backend == "zmq_to_scheduler"
                and not obj.need_wait_for_mm_inputs  # 非等待类请求
            ):
                # zmq_to_scheduler 模式下非等待类请求，退化为本地处理
                mm_inputs = await self.mm_processor.process_mm_data_async(
                    image_data=obj.image_data,
                    audio_data=obj.audio_data,
                    input_text=(input_text or input_ids),
                    request_obj=obj,
                    max_req_input_len=self.max_req_input_len,
                )

            # 更新 input_ids 和 token_type_ids（可能由 mm_processor 重写）
            if mm_inputs and mm_inputs.input_ids is not None:
                input_ids = mm_inputs.input_ids  # mm_processor 重写了 input_ids
            if mm_inputs and mm_inputs.token_type_ids is not None:
                token_type_ids = mm_inputs.token_type_ids  # mm_processor 重写了 token_type_ids
                if not isinstance(token_type_ids, list):
                    token_type_ids = token_type_ids.flatten().tolist()  # 转换为平序列表
            # 预计算 hash 时需要设置 pad_value
            if (
                envs.SGLANG_MM_PRECOMPUTE_HASH.get()
                and mm_inputs
                and mm_inputs.mm_items
            ):
                for item in mm_inputs.mm_items:
                    if isinstance(item, MultimodalDataItem):
                        item.set_pad_value()  # 为各多模态 item 设置 pad_value
        else:
            mm_inputs = None  # 无多模态输入

        self._validate_one_request(obj, input_ids)  # 输入长度等校验
        return self._create_tokenized_object(
            obj, input_text, input_ids, input_embeds, mm_inputs, token_type_ids  # 构建分词后请求对象
        )

    # 验证单条请求的 token 数量小于模型上下文长度
    def _validate_one_request(
        self, obj: Union[GenerateReqInput, EmbeddingReqInput], input_ids: List[int]
    ) -> None:
        """校验输入 token 数量和请求生成数量不超过模型上下文长度。"""
        # FIXME: unify the length validation logic with the one in the scheduler.
        _max_req_len = self.context_len  # 最大请求长度等于上下文长度
        # 计算输入 token 数量，包含预留 token
        input_token_num = len(input_ids) if input_ids is not None else 0  # 输入序列的 token 数
        input_token_num += self.num_reserved_tokens  # 加上预留的 token 数

        # 验证输入长度是否超过上下文长度
        if input_token_num >= self.context_len:
            if self.server_args.allow_auto_truncate:
                # 开启自动截断时，截断超长输入
                logger.warning(
                    f"The input ({input_token_num} tokens) is longer than the "
                    f"model's context length ({self.context_len} tokens). "
                    "Truncating the input."
                )
                del input_ids[_max_req_len:]  # 截断超出部分
                input_token_num = len(input_ids)  # 更新数量
            else:
                raise ValueError(
                    f"The input ({input_token_num} tokens) is longer than the "
                    f"model's context length ({self.context_len} tokens)."
                )

        # 验证总token数（输入 + max_new_tokens）是否超过限制
        max_new_tokens = obj.sampling_params.get("max_new_tokens")
        if (
            self.validate_total_tokens
            and max_new_tokens is not None
            and (max_new_tokens + input_token_num) > _max_req_len
        ):
            if self.server_args.allow_auto_truncate:
                # 自动截断max_new_tokens以适应上下文长度
                logger.warning(
                    f"Requested token count ({input_token_num} input + {max_new_tokens} new) "
                    f"exceeds the model's context length ({self.context_len} tokens). "
                    "Truncating max_new_tokens."
                )
                obj.sampling_params["max_new_tokens"] = max(
                    0, _max_req_len - input_token_num
                )
            else:
                total_tokens = max_new_tokens + input_token_num
                error_msg = (
                    f"Requested token count exceeds the model's maximum context length "
                    f"of {self.context_len} tokens. You requested a total of {total_tokens} "
                    f"tokens: {input_token_num} tokens from the input messages and "
                    f"{max_new_tokens} tokens for the completion. Please reduce the number "
                    f"of tokens in the input messages or the completion to fit within the limit."
                )
                raise ValueError(error_msg)

        # 验证embedding请求：生成模型不支持embedding请求
        if isinstance(obj, EmbeddingReqInput) and self.is_generation:
            raise ValueError(
                "This model does not appear to be an embedding model by default. "
                "Please add `--is-embedding` when launching the server or try another model."
            )

        # 验证Matryoshka嵌入维度参数
        if isinstance(obj, EmbeddingReqInput):
            self._validate_for_matryoshka_dim(obj)

        # 验证自定义logit processor和hidden states相关配置
        if isinstance(obj, GenerateReqInput):
            # 检查hidden states返回是否已启用
            if (
                obj.return_hidden_states
                and not self.server_args.enable_return_hidden_states
            ):
                raise ValueError(
                    "The server is not configured to return the hidden states. "
                    "Please set `--enable-return-hidden-states` to enable this feature."
                )
            # 检查自定义logit processor是否已启用
            if (
                obj.custom_logit_processor
                and not self.server_args.enable_custom_logit_processor
            ):
                raise ValueError(
                    "The server is not configured to enable custom logit processor. "
                    "Please set `--enable-custom-logit-processor` to enable this feature."
                )

    # 校验各模态输入数量不超过每请求限制
    def _validate_mm_limits(
        self, obj: Union[GenerateReqInput, EmbeddingReqInput]
    ) -> None:
        if not self.server_args.limit_mm_data_per_request:  # 未配置限制则直接返回
            return

        for modality, limit in self.server_args.limit_mm_data_per_request.items():  # 逐个模态进行校验
            data = getattr(obj, f"{modality}_data", None)  # 获取该模态的数据列表
            if data:
                count = len(data) if isinstance(data, list) else 1  # 计算该模态输入数量
                if count > limit:
                    raise ValueError(
                        f"{modality.capitalize()} count {count} exceeds limit {limit} per request."
                    )

    # 校验 Matryoshka 嵌入维度是否合法
    def _validate_for_matryoshka_dim(self, obj: EmbeddingReqInput) -> None:
        """若请求设置了 dimensions 字段，校验 Matryoshka 维度是否合法。"""
        if obj.dimensions is None:
            return  # 未指定维度，跳过校验

        if not self.model_config.is_matryoshka:
            raise ValueError(
                f"Model '{self.model_config.model_path}' does not support matryoshka representation, "
                f"changing output dimensions will lead to poor results."
            )

        if obj.dimensions < 1:
            raise ValueError("Requested dimensions must be greater than 0")  # 维度必须大于 0

        if (
            self.model_config.matryoshka_dimensions  # 模型有限定支持的维度列表
            and obj.dimensions not in self.model_config.matryoshka_dimensions  # 请求维度不在支持列表内
        ):
            raise ValueError(
                f"Model '{self.model_config.model_path}' only supports {self.model_config.matryoshka_dimensions} matryoshka dimensions, "
                f"using other output dimensions will lead to poor results."
            )

        if obj.dimensions > self.model_config.hidden_size:
            raise ValueError(
                f"Provided dimensions are greater than max embedding dimension: {self.model_config.hidden_size}"  # 超过最大嵌入维度
            )

    # 校验 token id 是否都在词表范围内
    def _validate_input_ids_in_vocab(
        self, input_ids: Union[List[int], List[List[int]]], vocab_size: int
    ) -> None:
        # 区分单条序列与批次序列
        if isinstance(input_ids[0], list):
            # 批次序列：逐条校验
            for seq in input_ids:
                # 检查是否存在超出词表范围的 token id
                if any(id >= vocab_size for id in seq):
                    raise ValueError(
                        f"The input_ids {seq} contains values greater than the vocab size ({vocab_size})."
                    )
        else:
            # 单条序列：直接校验 token id 范围
            if any(id >= vocab_size for id in input_ids):
                raise ValueError(
                    f"The input_ids {input_ids} contains values greater than the vocab size ({vocab_size})."
                )

    # 根据公共参数构建分词后的请求对象
    def _create_tokenized_object(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        input_text: str,
        input_ids: List[int],
        input_embeds: Optional[Union[List[float], None]] = None,
        mm_inputs=None,
        token_type_ids: Optional[List[int]] = None,
    ) -> Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput]:
        """根据公共参数构建分词后的请求对象。"""
        # 合并采样参数：显式传入的参数优先级高于预设默认值
        if self.preferred_sampling_params:
            sampling_kwargs = {**self.preferred_sampling_params, **obj.sampling_params}  # 请求参数覆盖预设
        else:
            sampling_kwargs = obj.sampling_params  # 直接使用请求参数
        # 构造采样参数对象，并做归一化和词表范围校验
        sampling_params = self.sampling_params_class(**sampling_kwargs)  # 创建采样参数对象
        sampling_params.normalize(self.tokenizer)  # 归一化（如：将字符串 stop 转为 token id）
        sampling_params.verify(self.model_config.vocab_size)  # 校验参数在词表范围内

        # 根据请求类型分别构建分词后的请求对象
        if isinstance(obj, GenerateReqInput):
            # 若存在 session 参数则解析为 SessionParams 对象
            session_params = (
                SessionParams(**obj.session_params) if obj.session_params else None  # 会话参数封装
            )

            # fake 后端下若未指定 bootstrap_room，自动分配递增计数器
            bootstrap_room = obj.bootstrap_room  # 获取原始 bootstrap_room
            if (
                bootstrap_room is None
                and self.server_args.disaggregation_transfer_backend == "fake"  # 仅 fake 后端需要自动分配
            ):
                bootstrap_room = self.fake_bootstrap_room_counter  # 使用当前计数器值
                self.fake_bootstrap_room_counter += 1  # 递增计数器

            # 构建生成请求的分词对象，透传所有字段
            tokenized_obj = TokenizedGenerateReqInput(
                input_text,
                input_ids,
                mm_inputs,
                sampling_params,
                obj.return_logprob,
                obj.logprob_start_len,
                obj.top_logprobs_num,
                obj.token_ids_logprob,
                obj.stream,
                rid=obj.rid,
                http_worker_ipc=obj.http_worker_ipc,
                bootstrap_host=obj.bootstrap_host,
                bootstrap_port=obj.bootstrap_port,
                bootstrap_room=bootstrap_room,
                lora_id=obj.lora_id,
                input_embeds=input_embeds,
                positional_embed_overrides=obj.positional_embed_overrides,
                session_params=session_params,
                custom_logit_processor=obj.custom_logit_processor,
                require_reasoning=obj.require_reasoning,
                return_hidden_states=obj.return_hidden_states,
                return_routed_experts=obj.return_routed_experts,
                routed_dp_rank=obj.routed_dp_rank,
                disagg_prefill_dp_rank=obj.disagg_prefill_dp_rank,
                priority=obj.priority,
                extra_key=obj.extra_key,
                routing_key=obj.routing_key,
                token_type_ids=token_type_ids,
                need_wait_for_mm_inputs=obj.need_wait_for_mm_inputs,
                num_items_assigned=obj.num_items_assigned,
                multi_item_delimiter_indices=obj.multi_item_delimiter_indices,
            )
        elif isinstance(obj, EmbeddingReqInput):
            # 若未预先解析位置嵌入覆盖，则根据 input_ids 和指定 token_id 进行解析
            positional_embed_overrides = obj.positional_embed_overrides  # 先取请求中的值
            if (
                positional_embed_overrides is None  # 未指定位置嵌入覆盖
                and obj.embed_overrides is not None  # 但有嵌入覆盖数据
                and obj.embed_override_token_id is not None  # 且有 placeholder token id
            ):
                positional_embed_overrides = self._resolve_embed_overrides(
                    input_ids, obj.embed_override_token_id, obj.embed_overrides  # 根据占位符解析位置
                )

            # 构建嵌入请求的分词对象
            tokenized_obj = TokenizedEmbeddingReqInput(
                input_text,
                input_ids,
                mm_inputs,
                token_type_ids,
                sampling_params,
                positional_embed_overrides=positional_embed_overrides,  # 位置嵌入覆盖
                rid=obj.rid,
                priority=obj.priority,
                dimensions=obj.dimensions,  # Matryoshka 输出维度
                lora_id=obj.lora_id,
                http_worker_ipc=obj.http_worker_ipc,
                return_pooled_hidden_states=obj.return_pooled_hidden_states,  # 是否返回池化 hidden states
                multi_item_delimiter_indices=obj.multi_item_delimiter_indices,
            )

        # 将时间统计对象挂载到分词结果，并记录分词完成时间
        tokenized_obj.time_stats = self.rid_to_state[obj.rid].time_stats  # 挂载时间统计
        self.rid_to_state[obj.rid].time_stats.set_tokenize_finish_time()  # 记录分词完成时刻

        return tokenized_obj  # 返回分词后的请求对象

    # 静态工具方法：解析 input_ids 中的 placeholder 位置，创建 PositionalEmbeds
    @staticmethod
    def _resolve_embed_overrides(
        input_ids: List[int],
        token_id: int,
        embeds: List[torch.Tensor],
    ) -> PositionalEmbeds:
        """解析 input_ids 中占位符位置并创建 PositionalEmbeds。

        扫描 input_ids 中所有等于 token_id 的位置，与 embeds 中的张量一一对应。
        """
        positions = [idx for idx, tok in enumerate(input_ids) if tok == token_id]  # 找出所有占位符位置
        if len(positions) != len(embeds):
            raise ValueError(
                f"input contains {len(positions)} occurrences of "
                f"embed_override_token_id={token_id}, "
                f"but embed_overrides has {len(embeds)} entries."  # 占位符数量与嵌入数量不匹配
            )
        return PositionalEmbeds(embeds=embeds, positions=positions)  # 返回位置-嵌入对

    # 批量分词并处理请求，仅支持纯文本输入
    async def _batch_tokenize_and_process(
        self, batch_size: int, obj: Union[GenerateReqInput, EmbeddingReqInput]
    ) -> List[Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput]]:
        """批量分词并处理文本输入（仅支持纯文本输入）。"""
        logger.debug(f"Starting batch tokenization for {batch_size} text requests")  # 记录开始批量分词

        # 若批次中无文本输入则无需分词，直接逐条处理
        if not self._batch_has_text(batch_size, obj):
            # 所有请求均已有 input_ids，无需分词
            return [await self._tokenize_one_request(obj[i]) for i in range(batch_size)]

        self._validate_batch_tokenization_constraints(batch_size, obj)  # 校验批量分词约束条件

        # 收集所有请求及对应文本
        requests = [obj[i] for i in range(batch_size)]  # 展开批次为请求列表
        texts = [req.text for req in requests]  # 提取每条请求的文本

        # 检查是否存在交叉编码器请求
        is_cross_encoder_request = any(
            isinstance(req, EmbeddingReqInput) and req.is_cross_encoder_request
            for req in requests  # 任意一条为交叉编码器请求则整批按此处理
        )

        # 使用统一方法对所有文本进行批量分词
        input_ids_list, token_type_ids_list = await self._tokenize_texts(
            texts, is_cross_encoder_request  # 批量分词入口
        )

        # 逐条处理分词结果，构建分词后的请求对象
        tokenized_objs = []
        for i, req in enumerate(requests):
            self._validate_one_request(obj[i], input_ids_list[i])  # 校验每条 token 长度
            token_type_ids = (
                token_type_ids_list[i] if token_type_ids_list is not None else None  # 提取 token_type_ids
            )
            tokenized_objs.append(
                self._create_tokenized_object(
                    req, req.text, input_ids_list[i], None, None, token_type_ids  # 构建分词对象
                )
            )
        logger.debug(f"Completed batch processing for {batch_size} requests")  # 记录批量处理完成
        return tokenized_objs

    # 校验批量分词的约束条件（不支持多模态/预分词/预嵌入）
    def _validate_batch_tokenization_constraints(
        self, batch_size: int, obj: Union[GenerateReqInput, EmbeddingReqInput]
    ) -> None:
        """校验批量分词约束条件，确保输入类型兼容批量分词。"""
        for i in range(batch_size):
            if self.is_generation and obj[i].contains_mm_input():
                # 多模态输入不支持批量分词
                raise ValueError(
                    "For multimodal input processing do not set `enable_tokenizer_batch_encode`."
                )
            if obj[i].input_ids is not None:
                # 已有 input_ids 时无需批量分词
                raise ValueError(
                    "Batch tokenization is not needed for pre-tokenized input_ids. Do not set `enable_tokenizer_batch_encode`."
                )
            if obj[i].input_embeds is not None:
                # 已有 input_embeds 时无需批量分词
                raise ValueError(
                    "Batch tokenization is not needed for input_embeds. Do not set `enable_tokenizer_batch_encode`."
                )

    # 检查批次中是否存在文本输入（文本或多模态）
    def _batch_has_text(
        self, batch_size: int, obj: Union[GenerateReqInput, EmbeddingReqInput]
    ) -> bool:
        """检查批次中是否存在文本输入。"""
        for i in range(batch_size):
            if obj[i].text:
                return True  # 存在文本输入
            elif self.is_generation and obj[i].contains_mm_input():
                return True  # 生成模型包含多模态输入也视为需要分词

        return False  # 所有请求均无需分词

    # 判断是否应使用批量分词模式
    def _should_use_batch_tokenization(self, batch_size, requests) -> bool:
        """判断是否应使用批量分词模式。

        当前策略：
        - 遵守服务器显式标志 enable_tokenizer_batch_encode。
        - 或者，当所有请求均无文本/多模态输入（使用预分词 id 或预嵌入）时，批量化请求。
        - 注意：批量分词暂不支持 DP attention，所有请求均会路由到第一个 rank。
        """
        return batch_size > 0 and (
            self.server_args.enable_tokenizer_batch_encode  # 显式开启批量分词
            or (
                (not self.server_args.enable_dp_attention)  # 未启用 DP attention
                and (not self._batch_has_text(batch_size, requests))  # 且批次无文本输入
            )
        )

    # 将单条分词请求发送给调度器
    def _send_one_request(
        self,
        tokenized_obj: Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput],
    ):
        tokenized_obj.time_stats.set_api_server_dispatch_time()  # 记录开始分发时刻
        tokenized_obj = wrap_shm_features(tokenized_obj)  # 将大型张量移入共享内存
        self.send_to_scheduler.send_pyobj(tokenized_obj)  # 通过 ZMQ 发送给调度器
        tokenized_obj.time_stats.set_api_server_dispatch_finish_time()  # 记录分发完成时刻

    # 将批量分词请求作为单个批次请求发送给调度器
    def _send_batch_request(
        self,
        tokenized_objs: List[
            Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput]
        ],
    ):
        """将批量分词请求以单个批次请求形式发送给调度器。"""
        if isinstance(tokenized_objs[0], TokenizedGenerateReqInput):
            batch_req = BatchTokenizedGenerateReqInput(batch=tokenized_objs)  # 生成请求批次
        else:
            batch_req = BatchTokenizedEmbeddingReqInput(batch=tokenized_objs)  # 嵌入请求批次

        set_time_batch(tokenized_objs, "set_api_server_dispatch_time")  # 记录批次开始分发时刻
        self.send_to_scheduler.send_pyobj(batch_req)  # 发送批次给调度器
        set_time_batch(tokenized_objs, "set_api_server_dispatch_finish_time")  # 记录批次分发完成时刻

    # 将多个增量流式输出块合并为单个输出块
    def _coalesce_streaming_chunks(
        self,
        out_list: list,
        rid: str,
    ) -> dict:
        """将多个增量流式块合并为单个输出块。

        text 和 output_ids 是增量 delta，需拼接；
        其余字段（meta_info 等）取自最后一个块。
        """
        if len(out_list) >= 20:
            # 积压过多块时警告，可能导致 P99 ITL 升高
            logger.warning(
                "Streaming backlog: rid=%s, coalescing %d queued chunks into one. "
                "This may inflate P99 ITL for affected requests.",
                rid,
                len(out_list),
            )
        out = dict(out_list[-1])  # 以最后一个块为基础
        if "output_ids" in out:
            out["output_ids"] = [id for chunk in out_list for id in chunk["output_ids"]]  # 拼接所有 output_ids
        if "text" in out:
            out["text"] = "".join(chunk["text"] for chunk in out_list)  # 拼接所有文本片段
        if "meta_info" in out:
            meta_info_list = [chunk["meta_info"] for chunk in out_list]  # 收集所有 meta_info
            meta_info = dict(meta_info_list[-1])  # 以最后一个 meta_info 为基础
            for key in _INCREMENTAL_STREAMING_META_INFO_KEYS:
                if any(key in m for m in meta_info_list):
                    meta_info[key] = [
                        item for m in meta_info_list for item in m.get(key, [])  # 拼接增量字段
                    ]
            out["meta_info"] = meta_info  # 更新合并后的 meta_info
        return out  # 返回合并后的输出块

    # 处理调度器返回的 abort/error 完成原因
    async def _handle_abort_finish_reason(
        self,
        out: dict,
        state: ReqState,
        is_stream: bool,
    ) -> Optional[dict]:
        """处理调度器返回的 abort/error 完成原因。

        流式 abort 时返回 out 供调用方 yield；非流式 abort 时抛出异常。
        正常流程时返回 None。
        """
        finish_reason = out["meta_info"]["finish_reason"]  # 提取完成原因字典

        if (
            finish_reason.get("type") == "abort"
            and finish_reason.get("status_code") == HTTPStatus.BAD_REQUEST  # 客户端错误：400
        ):
            if not is_stream:
                raise ValueError(finish_reason["message"])  # 非流式：抛出 ValueError
            return out  # 流式：返回 out，让上层 yield 给客户端

        if finish_reason.get("type") == "abort" and finish_reason.get(
            "status_code"
        ) in (
            HTTPStatus.SERVICE_UNAVAILABLE,  # 503
            HTTPStatus.INTERNAL_SERVER_ERROR,  # 500
        ):
            # 删除状态记录，防止重复发送 abort 请求，同时清理 aborted 请求状态
            if state.obj.rid in self.rid_to_state:
                del self.rid_to_state[state.obj.rid]

            # 标记 LoRA 请求已完成，释放 LoRA 资源
            if self.server_args.enable_lora and state.obj.lora_path:
                await self.lora_registry.release(state.obj.lora_id)
            if not is_stream:
                raise fastapi.HTTPException(
                    status_code=finish_reason["status_code"],
                    detail=finish_reason["message"],  # 非流式：抛出 HTTP 异常
                )
            return out  # 流式：返回 out

        return None  # 正常流程，无需特殊处理

    # 等待单条请求的响应，并以生成器方式逐步 yield 结果
    async def _wait_one_response(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        """等待单条请求的响应并逐步 yield 输出。"""
        state = self.rid_to_state[obj.rid]  # 获取请求状态
        # 并非所有请求类型都有 stream 字段（如 EmbeddingReqInput），默认非流式
        is_stream = getattr(obj, "stream", False)  # 判断是否为流式请求
        while True:
            try:
                # 等待调度器通过 event 通知有新输出
                await asyncio.wait_for(
                    state.event.wait(), timeout=_REQUEST_STATE_WAIT_TIMEOUT  # 带超时等待
                )
            except asyncio.TimeoutError:
                # 超时时检查客户端是否已断开
                if (
                    request is not None
                    and not obj.background
                    and await request.is_disconnected()
                ):
                    # 客户端断开（非流式，等待队列中）：终止请求
                    self.abort_request(obj.rid)
                    # 抛出异常以终止整个调用栈和 asyncio 任务
                    raise ValueError(
                        f"Request is disconnected from the client side (type 1). Abort request {obj.rid=}"
                    )
                continue  # 未断开则继续等待

            # 原子性地取走所有待输出块
            out_list = state.out_list  # 取出输出列表
            state.out_list = []  # 清空，防止重复处理
            finished = state.finished  # 记录是否已完成
            state.event.clear()  # 清除 event，准备下次等待

            # 增量流式模式下每个块是 delta，合并多个积压块以防止丢失 token id
            incremental_stream = (
                is_stream and self.server_args.incremental_streaming_output  # 是否开启增量流式
            )
            if incremental_stream and len(out_list) > 1:
                out = self._coalesce_streaming_chunks(out_list, obj.rid)  # 合并多个积压块
            else:
                out = out_list[-1]  # 取最后一个（通常只有一个）

            # 非增量流式：延迟解析文本，避免 O(n²) 的字符串拼接开销
            # _handle_batch_output 对中间块将 "text" 设为 None
            if (
                is_stream
                and not incremental_stream
                and "text" in out
                and out["text"] is None  # 中间块文本为 None，需在此处解析
            ):
                out["text"] = state.get_text()  # 从 state 中获取完整文本

            if finished:
                # 请求完成：记录响应发送时刻（在日志和指标记录前）
                if not state.time_stats.response_sent_to_client_time:
                    state.time_stats.set_response_sent_to_client_time()  # 记录首次响应时刻
                    out["meta_info"][
                        "response_sent_to_client_ts"
                    ] = state.time_stats.get_response_sent_to_client_realtime()  # 写入响应时间戳
                self.request_logger.log_finished_request(
                    obj,
                    out,
                    request=request,  # 记录请求完成日志
                )

                if self.request_metrics_exporter_manager.exporter_enabled():
                    # 异步写入请求级别指标
                    asyncio.create_task(
                        self.request_metrics_exporter_manager.write_record(obj, out)
                    )

                # 检查是否为调度器产生的 abort/error 完成原因
                if isinstance(out["meta_info"].get("finish_reason"), dict):
                    abort_out = await self._handle_abort_finish_reason(
                        out, state, is_stream  # 处理 abort 原因
                    )
                    if abort_out is not None:
                        yield abort_out  # 流式 abort：yield 给客户端
                        break

                yield out  # yield 最终结果
                break  # 退出循环

            if is_stream:
                # 流式模式：每个中间块都立即 yield 给客户端
                if not state.time_stats.response_sent_to_client_time:
                    state.time_stats.set_response_sent_to_client_time()  # 记录首次响应时刻
                    out["meta_info"][
                        "response_sent_to_client_ts"
                    ] = state.time_stats.get_response_sent_to_client_realtime()  # 写入响应时间戳
                yield out  # yield 流式中间块
            else:
                # 非流式模式：检查客户端是否已断开（running 状态）
                if (
                    request is not None
                    and not obj.background
                    and await request.is_disconnected()
                ):
                    # 客户端断开（非流式，运行中）：终止请求
                    self.abort_request(obj.rid)
                    # 抛出异常终止整个调用栈和 asyncio 任务
                    raise ValueError(
                        f"Request is disconnected from the client side (type 3). Abort request {obj.rid=}"
                    )

    # 处理批量请求（batch_size > 1），根据是否使用并行采样选择不同路径
    async def _handle_batch_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        batch_size = obj.batch_size  # 本批次的请求数量

        generators = []  # 存储每个请求对应的异步生成器
        rids = []  # 存储每个请求的 rid，用于流式输出时的索引映射
        if getattr(obj, "parallel_sample_num", 1) == 1:  # 无并行采样，走普通批处理路径
            if self._should_use_batch_tokenization(batch_size, obj):  # 判断是否启用批量 tokenization
                tokenized_objs = await self._batch_tokenize_and_process(batch_size, obj)  # 批量 tokenize
                self._send_batch_request(tokenized_objs)  # 一次性将整批请求发给调度器

                # Set up generators for each request in the batch
                for i in range(batch_size):  # 为批次中每个请求建立等待生成器
                    tmp_obj = obj[i]  # 取第 i 个子请求对象
                    generators.append(self._wait_one_response(tmp_obj, request))  # 添加等待协程
                    rids.append(tmp_obj.rid)  # 记录对应的 rid
            else:
                # Sequential tokenization and processing
                with (
                    input_blocker_guard_region(send_to_scheduler=self.send_to_scheduler)  # 共置批量生成时阻塞输入
                    if get_bool_env_var("SGLANG_ENABLE_COLOCATED_BATCH_GEN")  # 检查是否启用共置批量生成
                    else nullcontext()  # 否则使用空上下文管理器
                ):
                    for i in range(batch_size):  # 逐个 tokenize 并发送请求
                        tmp_obj = obj[i]  # 取第 i 个子请求
                        tokenized_obj = await self._tokenize_one_request(tmp_obj)  # 单个请求 tokenize
                        self._send_one_request(tokenized_obj)  # 发送给调度器
                        generators.append(self._wait_one_response(tmp_obj, request))  # 添加等待生成器
                        rids.append(tmp_obj.rid)  # 记录 rid
        else:
            # FIXME: When using batch and parallel_sample_num together, the perf is not optimal.
            if batch_size > 128:  # 批量并行采样超过 128 时发出性能警告
                logger.warning(
                    "Sending a single large batch with parallel sampling (n > 1) has not been well optimized. "
                    "The performance might be better if you just duplicate the requests n times or use "
                    "many threads to send them one by one with parallel sampling (n > 1)."
                )

            # Tokenize all requests
            objs = [obj[i] for i in range(batch_size)]  # 展开批次为独立的子请求列表
            tokenized_objs = await asyncio.gather(
                *(self._tokenize_one_request(obj) for obj in objs)  # 并发 tokenize 所有子请求
            )

            # Cache the common prefix for parallel sampling
            for i in range(batch_size):  # 为每个请求预填充公共前缀（max_new_tokens=0）
                tmp_obj = copy.copy(objs[i])  # 浅拷贝请求对象，避免修改原始对象
                tokenized_obj = copy.copy(tokenized_objs[i])  # 浅拷贝 tokenized 对象
                tokenized_obj.rid = tmp_obj.regenerate_rid()  # 生成新 rid 用于前缀缓存请求
                tokenized_obj.sampling_params = copy.copy(tokenized_obj.sampling_params)  # 拷贝采样参数
                tokenized_obj.sampling_params.max_new_tokens = 0  # 设为 0，只做 prefill 不生成 token
                tokenized_obj.stream = False  # 前缀预填充不使用流式输出
                self._init_req_state(tmp_obj)  # 初始化请求状态
                self._send_one_request(tokenized_obj)  # 发送前缀预填充请求
                await self._wait_one_response(tmp_obj, request).__anext__()  # 等待前缀预填充完成

            # Expand requests, assign new rids for them, and send them
            for i in range(batch_size):  # 对每个原始请求展开 parallel_sample_num 份
                for _ in range(obj.parallel_sample_num):  # 每份并行采样
                    tmp_obj = copy.copy(objs[i])  # 拷贝子请求对象
                    tokenized_obj = copy.copy(tokenized_objs[i])  # 拷贝 tokenized 对象
                    tokenized_obj.rid = tmp_obj.regenerate_rid()  # 生成新的唯一 rid
                    self._init_req_state(tmp_obj)  # 初始化该份请求的状态
                    tokenized_obj.time_stats = self.rid_to_state[tmp_obj.rid].time_stats  # 关联时间统计对象
                    self._send_one_request(tokenized_obj)  # 发送给调度器
                    generators.append(self._wait_one_response(tmp_obj, request))  # 添加等待生成器
                    rids.append(tmp_obj.rid)  # 记录 rid

                self.rid_to_state[objs[i].rid].time_stats.set_finished_time()  # 标记原始请求完成计时
                del self.rid_to_state[objs[i].rid]  # 删除原始请求状态（已展开为多份）

        # Wait for all requests
        is_stream = hasattr(obj, "stream") and obj.stream  # 判断是否为流式请求
        if not is_stream:  # 非流式：等待所有请求完成后一次性返回
            outputs = await asyncio.gather(*(gen.__anext__() for gen in generators))  # 并发等待所有生成器
            yield outputs  # 以列表形式一次性 yield 所有输出
        else:  # 流式：交错 yield 各请求的输出（先到先出）
            rid_to_index = {rid: i for i, rid in enumerate(rids)}  # 建立 rid -> 原始批次索引的映射
            task_map = {asyncio.create_task(gen.__anext__()): gen for gen in generators}  # 为每个生成器创建异步任务
            while task_map:  # 循环直到所有生成器耗尽
                done, _ = await asyncio.wait(
                    task_map.keys(), return_when=asyncio.FIRST_COMPLETED  # 等待任意一个任务完成
                )

                for task in done:  # 处理本轮完成的所有任务
                    gen = task_map.pop(task)  # 移除已完成任务，取出对应生成器
                    try:
                        result = task.result()  # 获取该任务的输出结果
                        result["index"] = rid_to_index[result["meta_info"]["id"]]  # 附加原始批次索引
                        yield result  # yield 单条流式输出
                        new_task = asyncio.create_task(gen.__anext__())  # 继续从该生成器取下一个结果
                        task_map[new_task] = gen  # 将新任务重新加入 task_map
                    except StopAsyncIteration:  # 生成器已耗尽，不重新加入
                        pass

    # 中止指定请求或所有请求：向调度器发送 AbortReq 消息
    def abort_request(self, rid: str = "", abort_all: bool = False):
        if not abort_all and rid not in self.rid_to_state:  # 单个请求不存在时直接返回
            return
        req = AbortReq(rid=rid, abort_all=abort_all)  # 构造中止请求消息
        self.send_to_scheduler.send_pyobj(req)  # 发送给调度器
        if self.enable_metrics:  # 启用指标收集时记录中止请求数
            # TODO: also use custom_labels from the request
            self.metrics_collector.observe_one_aborted_request(
                self.metrics_collector.labels
            )

    # 暂停生成：设置暂停标志并通知调度器，或在 abort 模式下等待所有请求完成
    async def pause_generation(self, obj: PauseGenerationReqInput):
        async with self.is_pause_cond:  # 获取暂停条件变量锁
            self.is_pause = True  # 标记为暂停状态
            if obj.mode != "abort":  # 非 abort 模式：通知调度器暂停生成
                await self.send_to_scheduler.send_pyobj(obj)
            else:
                # we are using the model_update_lock to check if there is still on-going requests.
                while True:  # abort 模式：循环中止所有请求直到没有进行中的请求
                    # TODO: maybe make it async instead of fire-and-forget
                    self.abort_request(abort_all=True)  # 中止所有请求
                    is_locked = await self.model_update_lock.is_locked()  # 检查是否还有进行中的请求
                    if not is_locked:  # 锁未被持有说明无进行中请求，退出循环
                        break
                    await asyncio.sleep(1.0)  # 等待 1 秒后重试

    # 恢复生成：清除暂停标志，通知调度器继续，并唤醒所有等待协程
    async def continue_generation(self, obj: ContinueGenerationReqInput):
        async with self.is_pause_cond:  # 获取暂停条件变量锁
            self.is_pause = False  # 清除暂停标志
            await self.send_to_scheduler.send_pyobj(obj)  # 通知调度器继续生成
            self.is_pause_cond.notify_all()  # 唤醒所有等待暂停条件的协程

    # 从磁盘加载新权重并更新模型，支持优雅暂停和版本追踪
    async def update_weights_from_disk(
        self,
        obj: UpdateWeightFromDiskReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()  # 确保事件循环已启动

        # default the load format to the server_args
        if obj.load_format is None:  # 若未指定加载格式，使用服务器默认格式
            obj.load_format = self.server_args.load_format
        logger.info("Start update_weights. Load format=%s", obj.load_format)

        if obj.abort_all_requests:  # 若需要，先中止所有进行中的请求
            self.abort_request(abort_all=True)

        # Immediately update the weights if the engine is in paused state
        async with self.is_pause_cond:  # 读取当前暂停状态
            is_paused = self.is_pause

        lock_context = (
            self.model_update_lock.writer_lock if not is_paused else nullcontext()  # 非暂停时需要写锁；暂停时无需加锁
        )
        async with lock_context:  # 加锁期间阻止新请求进入
            success, message, num_paused_requests = (
                await self._wait_for_model_update_from_disk(obj)  # 等待调度器完成权重更新
            )

        if success and obj.weight_version is not None:  # 更新成功且提供了版本号时更新版本信息
            self._update_weight_version_if_provided(obj.weight_version)
            message += f" Weight version updated to {obj.weight_version}."

        return success, message, num_paused_requests  # 返回成功标志、消息和暂停请求数

    # 更新本地存储的模型路径和加载格式信息
    def _update_model_path_info(self, model_path: str, load_format: str):
        self.served_model_name = model_path  # 更新对外提供的模型名称
        self.server_args.model_path = model_path  # 更新服务器参数中的模型路径
        self.server_args.load_format = load_format  # 更新服务器参数中的加载格式
        self.model_path = model_path  # 更新实例变量中的模型路径

    # 向调度器发送权重更新请求并等待结果，支持单副本和数据并行两种模式
    async def _wait_for_model_update_from_disk(
        self, obj: UpdateWeightFromDiskReqInput
    ) -> Tuple[bool, str]:
        self.send_to_scheduler.send_pyobj(obj)  # 向调度器发送权重更新请求
        self.model_update_result = asyncio.Future()  # 创建 Future 等待调度器回调
        if self.server_args.dp_size == 1:  # 单数据并行副本
            result = await self.model_update_result  # 等待调度器返回结果
            if result.success:  # 更新成功时同步模型路径信息
                self._update_model_path_info(obj.model_path, obj.load_format)
            return result.success, result.message, result.num_paused_requests
        else:  # self.server_args.dp_size > 1，多数据并行副本
            self.model_update_tmp = []  # 初始化临时结果列表，收集所有副本的结果
            result = await self.model_update_result  # 等待所有副本结果汇总后返回

            all_success = all([r.success for r in result])  # 所有副本均成功才算整体成功
            if all_success is True:  # 全部成功时更新模型路径信息
                self._update_model_path_info(obj.model_path, obj.load_format)
            all_message = [r.message for r in result]  # 收集所有副本的消息
            all_message = " | ".join(all_message)  # 用 | 拼接所有消息
            all_paused_requests = [r.num_paused_requests for r in result]  # 收集所有副本的暂停请求数
            return all_success, all_message, all_paused_requests

    # 动态配置日志级别、格式及请求转储相关参数
    def configure_logging(self, obj: ConfigureLoggingReq):
        self.request_logger.configure(  # 更新请求日志记录器的配置
            log_requests=obj.log_requests,  # 是否启用请求日志
            log_requests_level=obj.log_requests_level,  # 日志级别
            log_requests_format=obj.log_requests_format,  # 日志格式
        )
        if obj.dump_requests_folder is not None:  # 若指定了转储目录则更新
            self.dump_requests_folder = obj.dump_requests_folder
        if obj.dump_requests_threshold is not None:  # 若指定了转储阈值则更新
            self.dump_requests_threshold = obj.dump_requests_threshold
        if obj.crash_dump_folder is not None:  # 若指定了崩溃转储目录则更新
            self.crash_dump_folder = obj.crash_dump_folder
        logging.info(f"Config logging: {obj=}")  # 记录配置变更日志

    async def freeze_gc(self):
        """Send a freeze_gc message to the scheduler first, then freeze locally."""
        self.send_to_scheduler.send_pyobj(FreezeGCReq())  # 先通知调度器冻结 GC
        freeze_gc("Tokenizer Manager")  # 再冻结本进程的 Python GC
        return None

    # 创建后台中止任务：客户端断开连接时延迟 2 秒后中止对应请求
    def create_abort_task(self, obj: GenerateReqInput):
        # Abort the request if the client is disconnected.
        async def abort_request():
            await asyncio.sleep(2)  # 等待 2 秒，给客户端重连机会
            if obj.is_single:  # 单个请求
                self.abort_request(obj.rid)
            else:  # 批量请求，逐个中止
                for rid in obj.rid:
                    self.abort_request(rid)

        background_tasks = BackgroundTasks()  # 创建 FastAPI 后台任务容器
        background_tasks.add_task(abort_request)  # 注册中止协程为后台任务
        return background_tasks  # 返回后台任务对象，由 FastAPI 在响应后执行

    # 懒加载方式启动 asyncio 事件循环和核心协程任务（幂等，可多次调用）
    def auto_create_handle_loop(self):
        if self.event_loop is not None:  # 已初始化则直接返回，避免重复创建
            return

        # Create and start the handle_loop task
        loop = get_or_create_event_loop()  # 获取或创建 asyncio 事件循环
        self.asyncio_tasks.add(
            loop.create_task(print_exception_wrapper(self.handle_loop))  # 启动主消息接收循环任务
        )
        self.event_loop = loop  # 记录事件循环引用

        # We only add signal handler when the tokenizer manager is in the main thread
        # due to the CPython limitation.
        if threading.current_thread() is threading.main_thread():  # CPython 限制：只能在主线程注册信号处理器
            signal_handler = self.signal_handler_class(self)  # 创建信号处理器实例
            loop.add_signal_handler(signal.SIGTERM, signal_handler.sigterm_handler)  # 注册 SIGTERM 优雅退出处理器
            # Update the signal handler for the process. It overrides the sigquit handler in the launch phase.
            loop.add_signal_handler(
                signal.SIGQUIT, signal_handler.running_phase_sigquit_handler  # 注册运行阶段的 SIGQUIT 处理器（覆盖启动阶段的）
            )

        self.asyncio_tasks.add(
            loop.create_task(print_exception_wrapper(self.sigterm_watchdog))  # 启动 SIGTERM 看门狗任务，负责优雅排干请求
        )

    async def handle_loop(self):
        """The event loop that handles requests"""
        # 【学习注释 ⑨】主进程中持续运行的 asyncio 协程
        # 负责接收 DetokenizerManager 发来的已解码文字，唤醒对应的 HTTP 等待协程
        while True:
            with self.soft_watchdog.disable():
                recv_obj = await self.recv_from_detokenizer.recv_pyobj()
                # ↑ 异步等待 ZMQ 消息，不阻塞其他协程
                #   收到的是 BatchStrOutput（含多条请求的文字增量）
            if isinstance(
                recv_obj,
                (BatchStrOutput, BatchEmbeddingOutput, BatchTokenIDOutput),
            ):
                await self._handle_batch_output(recv_obj)
                # ↑ 遍历批次里每条请求，找到对应 ReqState，唤醒等待协程
            else:
                self._result_dispatcher(recv_obj)
                # ↑ 处理其他控制消息（flush cache 响应、weight update 响应等）
            self.last_receive_tstamp = real_time()
            self.soft_watchdog.feed()

    async def _handle_batch_output(
        self,
        recv_obj: Union[
            BatchStrOutput,
            BatchEmbeddingOutput,
            BatchTokenIDOutput,
        ],
    ):
        # 【学习注释 ⑨-续】将批次输出分发给各自等待的 HTTP 协程
        pending_notify: dict[str, ReqState] = {}
        batch_notify_size = self.server_args.batch_notify_size
        for i, rid in enumerate(recv_obj.rids):
            # ↑ recv_obj.rids 是本批次所有请求的 rid 列表
            state = self.rid_to_state.get(rid, None)
            # ↑ 用 rid 找到对应的 ReqState（在 generate_request 里创建的）
            if state is None:
                logger.error(
                    f"Received output for {rid=} but the state was deleted in TokenizerManager."
                )
                continue

            # Build meta_info and return value
            meta_info = {  # 构造元信息字典，包含请求 id、完成原因、token 计数等
                "id": rid,  # 请求 id
                "finish_reason": recv_obj.finished_reasons[i],  # 完成原因（None 表示未完成）
                "prompt_tokens": recv_obj.prompt_tokens[i],  # 输入 prompt 的 token 数
                "weight_version": self.server_args.weight_version,  # 当前权重版本号
                "total_retractions": recv_obj.retraction_counts[i],  # 推测解码的回退次数
            }

            if self.enable_metrics:  # 启用指标收集时附加调度器侧的时间统计
                if recv_obj.time_stats is not None:
                    scheduler_time_stats = recv_obj.time_stats[i]  # 取第 i 个请求的调度器时间统计
                    meta_info.update(scheduler_time_stats.convert_to_output_meta_info())  # 合并到 meta_info

            if getattr(state.obj, "return_logprob", False):  # 若请求要求返回 logprob
                self.convert_logprob_style(  # 将 logprob 原始数据转换并写入 meta_info
                    meta_info,
                    state,
                    state.obj.top_logprobs_num,  # 需要返回的 top-k logprob 数量
                    state.obj.token_ids_logprob,  # 指定 token id 的 logprob
                    state.obj.return_text_in_logprobs  # 是否在 logprob 中包含 token 文本
                    and not self.server_args.skip_tokenizer_init,
                    recv_obj,
                    i,
                )

            if not isinstance(recv_obj, BatchEmbeddingOutput):  # embedding 请求无生成 token，跳过此块
                meta_info.update(
                    {
                        "reasoning_tokens": recv_obj.reasoning_tokens[i],  # 推理 token 数（思维链）
                        "completion_tokens": recv_obj.completion_tokens[i],  # 生成 token 数
                        "cached_tokens": recv_obj.cached_tokens[i],  # KV 缓存命中的 token 数
                    }
                )
                # Add detailed cache breakdown if available
                if (
                    hasattr(recv_obj, "cached_tokens_details")
                    and recv_obj.cached_tokens_details
                ):
                    meta_info["cached_tokens_details"] = recv_obj.cached_tokens_details[  # 缓存命中详细分类
                        i
                    ]

            if getattr(recv_obj, "output_hidden_states", None):  # 若有隐藏状态则附加到 meta_info
                meta_info["hidden_states"] = recv_obj.output_hidden_states[i]
            if getattr(recv_obj, "routed_experts", None):  # 若有 MoE 路由专家信息则以 base64 编码附加
                routed_experts_tensor = recv_obj.routed_experts[i]
                if routed_experts_tensor is not None:
                    meta_info["routed_experts"] = pybase64.b64encode(
                        routed_experts_tensor.numpy().tobytes()
                    ).decode("utf-8")  # 将 tensor 序列化为 base64 字符串
            if getattr(recv_obj, "customized_info", None):  # 自定义扩展信息，按键附加到 meta_info
                for k, v in recv_obj.customized_info.items():
                    meta_info[k] = v[i]
            if getattr(recv_obj, "dp_ranks", None):  # 若有数据并行 rank 信息则附加
                meta_info["dp_rank"] = recv_obj.dp_ranks[i]

            state.finished = recv_obj.finished_reasons[i] is not None  # finished_reason 非 None 表示请求已完成
            if isinstance(recv_obj, BatchStrOutput):  # 文本生成输出分支
                # Not all request types have `stream` (e.g., EmbeddingReqInput). Default to non-streaming.
                is_stream = getattr(state.obj, "stream", False)  # 判断该请求是否使用流式输出
                incremental = (
                    self.server_args.incremental_streaming_output and is_stream  # 增量流式：每次只返回新增 token
                )
                delta_text = recv_obj.output_strs[i]  # 本次新增的文本片段
                delta_output_ids = recv_obj.output_ids[i]  # 本次新增的 token id 列表
                output_offset = state.last_output_offset  # 上次输出的 token 偏移量（用于增量切片）
                state.append_text(delta_text)  # 将新增文本追加到状态中
                state.output_ids.extend(delta_output_ids)  # 将新增 token id 追加到状态中

                if is_stream:  # 流式输出路径
                    if incremental:  # 增量流式：只返回本次新增部分
                        output_token_ids = delta_output_ids
                        _slice_streaming_output_meta_info(meta_info, output_offset)  # 对 meta_info 中的 logprob 等做增量切片
                        state.last_output_offset = len(state.output_ids)  # 更新输出偏移
                        out_dict = {
                            "text": delta_text,  # 仅返回增量文本
                            "output_ids": output_token_ids,
                            "meta_info": meta_info,
                        }
                    elif state.finished:  # 非增量流式 + 请求已完成：返回完整文本
                        out_dict = {
                            "text": state.get_text(),  # 返回累积的完整文本
                            "output_ids": state.output_ids.copy(),
                            "meta_info": meta_info,
                        }
                    else:
                        # Non-incremental intermediate: pass reference (no
                        # copy) and defer text to _wait_one_response to avoid
                        # O(n) per-step cost that compounds to O(n^2).
                        out_dict = {
                            "text": None,  # text=None，由 _wait_one_response 负责在消费时拼接，避免 O(n^2) 开销
                            "output_ids": state.output_ids,  # 传引用，不复制
                            "meta_info": meta_info,
                        }
                elif state.finished:  # 非流式 + 已完成：返回完整输出
                    out_dict = {
                        "text": state.get_text(),
                        "output_ids": state.output_ids.copy(),
                        "meta_info": meta_info,
                    }
                else:  # 非流式且未完成：不输出中间结果
                    out_dict = None
            elif isinstance(recv_obj, BatchTokenIDOutput):  # token id 输出分支（skip_tokenizer_init 模式）
                is_stream = getattr(state.obj, "stream", False)  # 判断是否流式
                incremental = (
                    self.server_args.incremental_streaming_output and is_stream  # 是否增量流式
                )
                delta_output_ids = recv_obj.output_ids[i]  # 本次新增的 token id
                output_offset = state.last_output_offset  # 上次输出偏移
                state.output_ids.extend(delta_output_ids)  # 追加到状态

                if is_stream:  # 流式路径
                    if incremental:  # 增量流式：只返回本次新增 token id
                        output_token_ids = delta_output_ids
                        _slice_streaming_output_meta_info(meta_info, output_offset)
                        state.last_output_offset = len(state.output_ids)
                        out_dict = {
                            "output_ids": output_token_ids,  # 仅返回增量 token id
                            "meta_info": meta_info,
                        }
                    elif state.finished:  # 非增量流式 + 已完成
                        out_dict = {
                            "output_ids": state.output_ids.copy(),  # 返回完整 token id 列表
                            "meta_info": meta_info,
                        }
                    else:  # 非增量流式中间步骤
                        out_dict = {
                            "output_ids": state.output_ids,  # 传引用（未完成时逐步累积）
                            "meta_info": meta_info,
                        }
                elif state.finished:  # 非流式 + 已完成
                    out_dict = {
                        "output_ids": state.output_ids.copy(),
                        "meta_info": meta_info,
                    }
                else:  # 非流式且未完成
                    out_dict = None
            else:  # embedding 输出分支
                assert isinstance(recv_obj, BatchEmbeddingOutput)
                out_dict = {
                    "embedding": recv_obj.embeddings[i],  # embedding 向量
                    "meta_info": meta_info,
                }
                if (
                    recv_obj.pooled_hidden_states is not None
                    and recv_obj.pooled_hidden_states[i] is not None
                ):
                    out_dict["pooled_hidden_state"] = recv_obj.pooled_hidden_states[i]  # 附加 pooled 隐藏状态

            # Set first_token_time on the first output batch.
            # This is the single write point for first_token_time.
            if state.time_stats.first_token_time == 0.0:  # 首次收到 token 输出时记录 first_token 时间
                state.time_stats.set_first_token_time()

            if state.finished:  # 请求已完成时做收尾处理
                if state.time_stats.trace_ctx.tracing_enable:  # 若启用了分布式追踪
                    state.time_stats.trace_ctx.trace_set_root_attrs(
                        self.convert_to_span_attrs(state, recv_obj, i)  # 将请求属性写入 trace span
                    )
                state.time_stats.set_finished_time()  # 记录完成时间戳
                meta_info["e2e_latency"] = state.time_stats.get_e2e_latency()  # 计算端到端延迟

                if self.server_args.speculative_algorithm:  # 推测解码时计算接受率等指标
                    self._calculate_spec_decoding_metrics(meta_info, recv_obj, i)
                if self.enable_metrics:  # 启用指标收集时更新完整延迟统计
                    scheduler_time_stats = (
                        recv_obj.time_stats[i]
                        if recv_obj.time_stats is not None
                        else None
                    )
                    completion_tokens = (
                        recv_obj.completion_tokens[i]  # 生成的 token 数（embedding 请求为 0）
                        if not isinstance(recv_obj, BatchEmbeddingOutput)
                        else 0
                    )
                    meta_info.update(
                        state.time_stats.convert_to_output_meta_info(  # 将完整时间统计合并到 meta_info
                            scheduler_time_stats, completion_tokens
                        )
                    )

                del self.rid_to_state[rid]  # 请求完成，释放状态（避免内存泄漏）

                # Mark ongoing LoRA request as finished.
                if self.server_args.enable_lora and state.obj.lora_path:  # 启用 LoRA 时释放对应 adapter 的引用计数
                    asyncio.create_task(self.lora_registry.release(state.obj.lora_id))

            if out_dict is not None:  # 有输出内容时通知等待协程
                state.out_list.append(out_dict)  # 将输出加入状态的输出队列
                pending_notify[rid] = state  # 标记该 rid 需要通知

                if len(pending_notify) >= batch_notify_size:  # 达到批量通知阈值时批量 set event
                    for s in pending_notify.values():
                        s.event.set()  # 唤醒等待该请求输出的协程
                    pending_notify = {}  # 清空待通知列表
                    await asyncio.sleep(0)  # 让出 CPU，允许被唤醒的协程运行

            if self.enable_metrics and state.obj.log_metrics:  # 收集 Prometheus 指标
                self.collect_metrics(state, recv_obj, i)
            if self.dump_requests_folder and state.finished and state.obj.log_metrics:  # 转储请求到文件
                self.dump_requests(state, out_dict)
            if self.crash_dump_folder and state.finished and state.obj.log_metrics:  # 记录请求到崩溃转储缓冲区
                self.record_request_for_crash_dump(state, out_dict)

        # handle_loop awaits next recv immediately
        for s in pending_notify.values():  # 批次结束后通知剩余未达阈值的请求
            s.event.set()

        # When skip_tokenizer_init is enabled, tokensizer_manager receives
        # BatchTokenIDOutput.
        if (
            self.server_args.dp_size > 1  # 数据并行模式下需要汇报各副本负载
            and isinstance(recv_obj, (BatchStrOutput, BatchTokenIDOutput))
            and recv_obj.load is not None
        ):
            load_update_req = WatchLoadUpdateReq(loads=[recv_obj.load])  # 构造负载更新请求
            self.send_to_scheduler.send_pyobj(load_update_req)  # 向调度器更新负载信息

    # 将状态中累积的 logprob 原始数据（val/idx）转换为可读格式并写入 meta_info
    def add_logprob_to_meta_info(
        self,
        meta_info: dict,
        state: ReqState,
        top_logprobs_num: int,
        token_ids_logprob: List[int],
        return_text_in_logprobs: bool,
    ):
        # 1. Handle regular logprobs
        if len(state.input_token_logprobs_val) > len(state.input_token_logprobs):  # 有未解码的输入 logprob
            state.input_token_logprobs.extend(
                self.detokenize_logprob_tokens(
                    state.input_token_logprobs_val[len(state.input_token_logprobs) :],  # 取未解码的新增部分
                    state.input_token_logprobs_idx[len(state.input_token_logprobs) :],
                    return_text_in_logprobs,  # 是否将 token id 解码为文本
                )
            )

        if len(state.output_token_logprobs_val) > len(state.output_token_logprobs):  # 有未解码的输出 logprob
            state.output_token_logprobs.extend(
                self.detokenize_logprob_tokens(
                    state.output_token_logprobs_val[len(state.output_token_logprobs) :],
                    state.output_token_logprobs_idx[len(state.output_token_logprobs) :],
                    return_text_in_logprobs,
                )
            )

        meta_info["input_token_logprobs"] = state.input_token_logprobs  # 写入输入 token logprob
        meta_info["output_token_logprobs"] = state.output_token_logprobs  # 写入输出 token logprob
        meta_info["output_token_logprobs_length"] = len(state.output_token_logprobs)  # 输出 logprob 长度

        # 2. Handle top logprobs
        if top_logprobs_num > 0:  # 需要返回每个位置 top-k 候选 logprob
            if len(state.input_top_logprobs_val) > len(state.input_top_logprobs):  # 有未解码的输入 top logprob
                state.input_top_logprobs.extend(
                    self.detokenize_top_logprobs_tokens(
                        state.input_top_logprobs_val[len(state.input_top_logprobs) :],
                        state.input_top_logprobs_idx[len(state.input_top_logprobs) :],
                        return_text_in_logprobs,
                    )
                )
            if len(state.output_top_logprobs_val) > len(state.output_top_logprobs):  # 有未解码的输出 top logprob
                state.output_top_logprobs.extend(
                    self.detokenize_top_logprobs_tokens(
                        state.output_top_logprobs_val[len(state.output_top_logprobs) :],
                        state.output_top_logprobs_idx[len(state.output_top_logprobs) :],
                        return_text_in_logprobs,
                    )
                )

            meta_info["input_top_logprobs"] = state.input_top_logprobs  # 写入输入 top logprob
            meta_info["output_top_logprobs"] = state.output_top_logprobs  # 写入输出 top logprob

        # 3. Handle token_ids_logprob
        if token_ids_logprob is not None:  # 需要返回指定 token id 的 logprob
            if len(state.input_token_ids_logprobs_val) > len(
                state.input_token_ids_logprobs
            ):  # 有未解码的输入 token_ids logprob
                state.input_token_ids_logprobs.extend(
                    self.detokenize_top_logprobs_tokens(
                        state.input_token_ids_logprobs_val[
                            len(state.input_token_ids_logprobs) :
                        ],
                        state.input_token_ids_logprobs_idx[
                            len(state.input_token_ids_logprobs) :
                        ],
                        return_text_in_logprobs,
                    )
                )
            if len(state.output_token_ids_logprobs_val) > len(
                state.output_token_ids_logprobs
            ):  # 有未解码的输出 token_ids logprob
                state.output_token_ids_logprobs.extend(
                    self.detokenize_top_logprobs_tokens(
                        state.output_token_ids_logprobs_val[
                            len(state.output_token_ids_logprobs) :
                        ],
                        state.output_token_ids_logprobs_idx[
                            len(state.output_token_ids_logprobs) :
                        ],
                        return_text_in_logprobs,
                    )
                )

            meta_info["input_token_ids_logprobs"] = state.input_token_ids_logprobs  # 写入指定 token 的输入 logprob
            meta_info["output_token_ids_logprobs"] = state.output_token_ids_logprobs  # 写入指定 token 的输出 logprob

    # 将批次输出中的 logprob 原始数据累积到 state 中，再转换写入 meta_info
    def convert_logprob_style(
        self,
        meta_info: dict,
        state: ReqState,
        top_logprobs_num: int,
        token_ids_logprob: List[int],
        return_text_in_logprobs: bool,
        recv_obj: BatchStrOutput,
        recv_obj_index: int,
    ):
        if recv_obj.input_token_logprobs_val is None:  # 若批次输出不含 logprob 则直接返回
            return

        if (
            len(recv_obj.input_token_logprobs_val) > 0
            and recv_obj.input_token_logprobs_val[recv_obj_index] is not None
        ):  # 有输入 logprob 数据时追加到 state
            state.input_token_logprobs_val.extend(
                recv_obj.input_token_logprobs_val[recv_obj_index]  # 输入 token logprob 值
            )
            state.input_token_logprobs_idx.extend(
                recv_obj.input_token_logprobs_idx[recv_obj_index]  # 输入 token logprob 对应的 token id
            )
        state.output_token_logprobs_val.extend(
            recv_obj.output_token_logprobs_val[recv_obj_index]  # 输出 token logprob 值
        )
        state.output_token_logprobs_idx.extend(
            recv_obj.output_token_logprobs_idx[recv_obj_index]  # 输出 token logprob 对应的 token id
        )

        if top_logprobs_num > 0:  # 追加 top-k logprob 数据
            if len(recv_obj.input_top_logprobs_val) > 0:  # 有输入 top logprob
                state.input_top_logprobs_val.extend(
                    recv_obj.input_top_logprobs_val[recv_obj_index]
                )
                state.input_top_logprobs_idx.extend(
                    recv_obj.input_top_logprobs_idx[recv_obj_index]
                )
            state.output_top_logprobs_val.extend(
                recv_obj.output_top_logprobs_val[recv_obj_index]  # 输出 top logprob 值
            )
            state.output_top_logprobs_idx.extend(
                recv_obj.output_top_logprobs_idx[recv_obj_index]  # 输出 top logprob token id
            )

        if token_ids_logprob is not None:  # 追加指定 token id 的 logprob 数据
            if len(recv_obj.input_token_ids_logprobs_val) > 0:
                state.input_token_ids_logprobs_val.extend(
                    recv_obj.input_token_ids_logprobs_val[recv_obj_index]
                )
                state.input_token_ids_logprobs_idx.extend(
                    recv_obj.input_token_ids_logprobs_idx[recv_obj_index]
                )
            state.output_token_ids_logprobs_val.extend(
                recv_obj.output_token_ids_logprobs_val[recv_obj_index]
            )
            state.output_token_ids_logprobs_idx.extend(
                recv_obj.output_token_ids_logprobs_idx[recv_obj_index]
            )

        self.add_logprob_to_meta_info(  # 将累积的 logprob 原始数据转换并写入 meta_info
            meta_info,
            state,
            state.obj.top_logprobs_num,
            state.obj.token_ids_logprob,
            return_text_in_logprobs,
        )

    # 将一组 (logprob, token_id) 对解码为 (logprob, token_id, text) 三元组列表
    def detokenize_logprob_tokens(
        self,
        token_logprobs_val: List[float],
        token_logprobs_idx: List[int],
        decode_to_text: bool,
    ):
        if not decode_to_text:  # 不需要文本时，text 字段置 None
            return [
                (logprob, token_id, None)
                for logprob, token_id in zip(token_logprobs_val, token_logprobs_idx)
            ]
        else:
            assert self.tokenizer is not None
            # In transformers v5, batch_decode([1, 2, 3]) concatenates all tokens
            # into one string. Wrap each ID in its own list so they decode separately.
            token_texts = self.tokenizer.batch_decode(
                [[idx] for idx in token_logprobs_idx]  # 每个 id 单独包装，避免 v5 把所有 token 拼成一个字符串
            )
            return list(zip(token_logprobs_val, token_logprobs_idx, token_texts))  # 打包为三元组列表

    # 对多个位置的 top-k 候选逐位置调用 detokenize_logprob_tokens
    def detokenize_top_logprobs_tokens(
        self,
        token_logprobs_val: List[float],
        token_logprobs_idx: List[int],
        decode_to_text: bool,
    ):
        # TODO: The current implementation only batches the detokenization for top-k tokens per single position.
        # We should batch all top-k tokens in all positions.
        ret = []
        for i in range(len(token_logprobs_val)):  # 遍历每个位置
            if token_logprobs_val[i]:  # 该位置有 top-k 候选数据
                ret.append(
                    self.detokenize_logprob_tokens(
                        token_logprobs_val[i], token_logprobs_idx[i], decode_to_text  # 解码该位置的 top-k
                    )
                )
            else:
                ret.append(None)  # 该位置无数据（如 prompt 位置无输出 logprob）
        return ret  # 返回按位置排列的 top-k logprob 列表

    # 计算推测解码指标：接受率、平均接受长度和接受直方图，写入 meta_info
    def _calculate_spec_decoding_metrics(
        self,
        meta_info: Dict[str, Any],
        recv_obj: Union[
            BatchStrOutput,
            BatchEmbeddingOutput,
            BatchTokenIDOutput,
        ],
        i: int,
    ) -> None:
        """Calculate speculative decoding metrics, such as acceptance rate and acceptance length metrics."""
        if (
            hasattr(recv_obj, "spec_verify_ct")
            and recv_obj.spec_verify_ct[i] > 0  # 至少经历了一次验证步骤
            and hasattr(recv_obj, "spec_accepted_drafts")
            and len(recv_obj.spec_accepted_drafts) > i
        ):
            # Total number of proposed draft tokens per request.
            all_drafts = recv_obj.spec_verify_ct[i] * (
                self.server_args.speculative_num_draft_tokens - 1  # 每轮验证的草稿 token 数（不含 bonus token）
            )
            accepted_drafts = recv_obj.spec_accepted_drafts[i]  # 被接受的草稿 token 总数

            # Calculate per-request acceptance rate and average acceptance length.
            if all_drafts > 0:
                # accept_rate: accepted_drafts / total_proposed_drafts (strict count, no bonus).
                meta_info["spec_accept_rate"] = accepted_drafts / all_drafts  # 草稿接受率（严格计数）
                # accept_length: completion_tokens / verify_ct (includes bonus token).
                meta_info["spec_accept_length"] = (
                    recv_obj.completion_tokens[i] / recv_obj.spec_verify_ct[i]  # 平均每轮验证接受的 token 数（含 bonus）
                )

                meta_info["spec_accepted_drafts"] = accepted_drafts  # 被接受的草稿 token 总数
                meta_info["spec_proposed_drafts"] = all_drafts  # 提出的草稿 token 总数
                meta_info["spec_verify_ct"] = recv_obj.spec_verify_ct[i]  # 验证轮数

            # Acceptance histogram: tracks how many decoding steps accepted a certain number of draft tokens.
            if (
                recv_obj.spec_acceptance_histogram
                and len(recv_obj.spec_acceptance_histogram) > i
                and recv_obj.spec_acceptance_histogram[i]
            ):
                meta_info["spec_accept_histogram"] = recv_obj.spec_acceptance_histogram[  # 接受数量的分布直方图
                    i
                ]

    # 判断请求是否使用了结构化输出（grammar）约束
    def _request_has_grammar(self, obj: GenerateReqInput) -> bool:
        return (
            obj.sampling_params.get("json_schema", None)  # JSON Schema 约束
            or obj.sampling_params.get("regex", None)  # 正则表达式约束
            or obj.sampling_params.get("ebnf", None)  # EBNF 文法约束
            or obj.sampling_params.get("structural_tag", None)  # 结构化标签约束
        )

    # 收集 Prometheus 指标：TTFT、ITL（inter-token latency）和请求完成统计
    def collect_metrics(self, state: ReqState, recv_obj: BatchStrOutput, i: int):
        completion_tokens = (
            recv_obj.completion_tokens[i]  # 当前已生成的 token 数
            if getattr(recv_obj, "completion_tokens", None)
            else 0
        )

        custom_labels = getattr(state.obj, "custom_labels", None)  # 请求携带的自定义标签
        labels = dict(self.metrics_collector.labels)  # 复制基础标签字典
        if custom_labels:  # 合并请求级别的自定义标签
            labels.update(custom_labels)
        if self.enable_priority_scheduling:  # 启用优先级调度时将优先级附加到标签
            priority = getattr(state.obj, "priority", None)
            if priority is not None:
                labels["priority"] = str(priority)
        if (
            not state.ttft_observed  # 首次观测到输出（TTFT 事件）
            and self.disaggregation_mode != DisaggregationMode.PREFILL  # prefill 解耦模式下不记录 TTFT
        ):
            state.ttft_observed = True  # 标记 TTFT 已记录
            state.last_completion_tokens = completion_tokens
            self.metrics_collector.observe_time_to_first_token(
                labels, state.time_stats.get_first_token_latency()  # 记录首 token 延迟
            )
        else:
            num_new_tokens = completion_tokens - state.last_completion_tokens  # 本次新增的 token 数
            if num_new_tokens:
                self.metrics_collector.observe_inter_token_latency(
                    labels,
                    state.time_stats.get_interval(),  # 上次到本次的时间间隔
                    num_new_tokens,  # 新增 token 数
                )
                state.time_stats.set_last_time()  # 更新上次记录时间
                state.last_completion_tokens = completion_tokens  # 更新已记录的 token 数

        if state.finished:  # 请求完成时记录完整请求的指标
            retraction_count = (
                recv_obj.retraction_counts[i]  # 推测解码回退次数
                if getattr(recv_obj, "retraction_counts", None)
                and i < len(recv_obj.retraction_counts)
                else 0
            )

            # Get detailed cache breakdown if available
            cached_tokens_details = None
            if (
                hasattr(recv_obj, "cached_tokens_details")
                and recv_obj.cached_tokens_details
            ):
                cached_tokens_details = recv_obj.cached_tokens_details[i]  # 缓存命中详细分类

            self.metrics_collector.observe_one_finished_request(  # 记录完成请求的各项统计
                labels,
                recv_obj.prompt_tokens[i],  # 输入 prompt token 数
                completion_tokens,  # 生成 token 数
                recv_obj.cached_tokens[i],  # 缓存命中 token 数
                state.time_stats.get_e2e_latency(),  # 端到端延迟
                self._request_has_grammar(state.obj),  # 是否使用了结构化约束
                retraction_count,  # 推测解码回退次数
                cached_tokens_details,  # 缓存详细分类
            )

    # 将完成的请求追加到转储队列，达到阈值后批量写入 pickle 文件
    def dump_requests(self, state: ReqState, out_dict: dict):
        self.dump_request_list.append(
            (
                state.obj,  # 原始请求对象
                out_dict,  # 请求的输出字典
                convert_time_to_realtime(state.time_stats.created_time),  # 请求创建时间（真实时间）
                convert_time_to_realtime(state.time_stats.finished_time),  # 请求完成时间（真实时间）
            )
        )

        if len(self.dump_request_list) >= self.dump_requests_threshold:  # 达到阈值时写入文件
            filename = os.path.join(
                self.dump_requests_folder,
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pkl",  # 以时间戳命名文件
            )
            self._dump_data_to_file(
                data_list=self.dump_request_list,
                filename=filename,
                log_message=f"Dump {len(self.dump_request_list)} requests to {filename}",
            )
            self.dump_request_list = []  # 清空队列

    # 将完成的请求记录到滑动窗口缓冲区中（保留最近 5 分钟），供崩溃时转储
    def record_request_for_crash_dump(self, state: ReqState, out_dict: dict):
        current_time = real_time()  # 获取当前真实时间
        self.crash_dump_request_list.append(
            (
                state.obj,  # 原始请求对象
                out_dict,  # 请求输出
                convert_time_to_realtime(state.time_stats.created_time),  # 创建时间
                current_time,  # 完成时间（用于滑动窗口淘汰）
            )
        )
        # Remove requests older than 5 minutes based on finish time
        while (
            self.crash_dump_request_list
            and current_time - self.crash_dump_request_list[0][3] >= 300  # 超过 5 分钟的记录弹出
        ):
            self.crash_dump_request_list.popleft()

    # 将请求列表异步写入 pickle 文件（在后台线程中执行，不阻塞事件循环）
    def _dump_data_to_file(
        self, data_list: List[Tuple], filename: str, log_message: str
    ):
        logger.info(log_message)
        to_dump_with_server_args = {
            "server_args": self.server_args,  # 包含服务器配置
            "requests": data_list.copy(),  # 复制一份请求列表（避免后续修改影响写入）
        }

        def background_task():
            os.makedirs(os.path.dirname(filename), exist_ok=True)  # 确保目录存在
            with open(filename, "wb") as f:
                pickle.dump(to_dump_with_server_args, f)  # 序列化为 pickle 文件

        asyncio.create_task(asyncio.to_thread(background_task))  # 在线程池中异步执行，避免阻塞事件循环

    # 崩溃前将最近的请求（已完成和进行中）转储到 pickle 文件，便于事后调试
    def dump_requests_before_crash(
        self, hostname: str = os.getenv("HOSTNAME", socket.gethostname())  # 默认取主机名用于构建目录
    ):
        if not self.crash_dump_folder:  # 未配置崩溃转储目录时跳过
            return

        if self.crash_dump_performed:  # 避免重复转储（如 SIGTERM+SIGQUIT 同时触发）
            logger.info(
                "SIGTERM/SIGQUIT/Exception triggered, but crash dump already performed, skipping."
            )
            return
        else:
            self.crash_dump_performed = True  # 标记已执行，防止重复

        logger.error(f"Dumping requests before crash. {self.crash_dump_folder=}")

        # Add finished requests from crash_dump_request_list
        data_to_dump = []
        if self.crash_dump_request_list:  # 将滑动窗口中的已完成请求加入转储列表
            data_to_dump.extend(self.crash_dump_request_list)

        # Add unfinished requests from rid_to_state
        unfinished_requests = []
        for rid, state in self.rid_to_state.items():  # 遍历所有进行中的请求
            if not state.finished:  # 只处理未完成的请求
                state.time_stats.set_finished_time()  # 强制记录结束时间（崩溃时刻）
                unfinished_requests.append(
                    (
                        state.obj,  # 原始请求对象
                        (
                            state.out_list[-1]  # 最后一次输出（如果有）
                            if state.out_list
                            else state.get_crash_dump_output()  # 否则生成一个空输出占位符
                        ),
                        convert_time_to_realtime(state.time_stats.created_time),
                        convert_time_to_realtime(state.time_stats.finished_time),
                    )
                )
        if unfinished_requests:
            data_to_dump.extend(unfinished_requests)  # 合并未完成请求

        if not data_to_dump:  # 无数据时不创建文件
            return

        # Create a file
        filename = os.path.join(
            self.crash_dump_folder,
            hostname,  # 以主机名为子目录，便于多节点区分
            f'crash_dump_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pkl',  # 以时间戳命名
        )
        os.makedirs(os.path.dirname(filename), exist_ok=True)  # 确保目录存在

        # Write the data to the file
        data_to_dump_with_server_args = {
            "server_args": self.server_args,  # Include server_args in the dump
            "requests": data_to_dump,  # 请求数据列表
            "launch_command": " ".join(sys.argv),  # 记录启动命令，便于复现
        }
        with open(filename, "wb") as f:
            pickle.dump(data_to_dump_with_server_args, f)  # 同步写入（崩溃前无法用异步）
        logger.error(
            f"Dumped {len(self.crash_dump_request_list)} finished and {len(unfinished_requests)} unfinished requests before crash to {filename}"
        )
        return filename  # 返回转储文件路径

    # SIGTERM 看门狗：等待 gracefully_exit 标志，然后排干剩余请求后退出
    async def sigterm_watchdog(self):
        while not self.gracefully_exit:  # 循环等待 SIGTERM 信号触发
            await asyncio.sleep(5)

        # Drain requests
        while True:  # 进入排干循环，等待所有请求完成
            remain_num_req = len(self.rid_to_state)  # 当前进行中的请求数量
            remaining_rids = list(self.rid_to_state.keys())  # 记录剩余请求 rid，用于日志

            if self.server_status == ServerStatus.UnHealthy:  # 健康检查失败时强制退出
                # if health check failed, we should exit immediately
                logger.error(
                    "Signal SIGTERM received while health check failed. Force exiting."
                )
                self.dump_requests_before_crash()  # 转储请求
                self.force_exit_handler()  # 执行自定义退出逻辑
                break

            elif get_bool_env_var("SGL_FORCE_SHUTDOWN"):  # 强制关闭标志时立即退出
                # if force shutdown flag set, exit immediately
                logger.error(
                    "Signal SIGTERM received while force shutdown flag set. Force exiting."
                )
                self.force_exit_handler()
                break

            logger.info(
                f"Gracefully exiting... Remaining number of requests {remain_num_req}. Remaining requests {remaining_rids=}."
            )
            if remain_num_req > 0:  # 还有未完成请求，继续等待
                await asyncio.sleep(5)
            else:  # 所有请求已完成，执行最终转储并退出
                self.dump_requests_before_crash()
                break

        kill_process_tree(os.getpid(), include_parent=True)  # 杀死进程树（包括父进程）
        sys.exit(0)  # 正常退出

    def force_exit_handler(self):
        """Put some custom force exit logic here."""
        pass  # 子类可覆盖此方法添加自定义强制退出逻辑

    # 处理来自调度器的请求中止回调：构造中止输出并唤醒等待协程
    def _handle_abort_req(self, recv_obj: AbortReq):
        if is_health_check_generate_req(recv_obj):  # 健康检查请求不需要处理
            return
        state = self.rid_to_state[recv_obj.rid]  # 获取请求状态
        state.finished = True  # 标记为已完成
        state.time_stats.set_finished_time()  # 记录完成时间

        abort_message = recv_obj.abort_message or "Abort in waiting queue"  # 中止消息
        finish_reason = {
            "type": "abort",
            "message": abort_message,
        }
        if recv_obj.finished_reason:  # 若调度器提供了更详细的完成原因则覆盖
            finish_reason = recv_obj.finished_reason
        meta_info = {
            "id": recv_obj.rid,  # 请求 id
            "finish_reason": finish_reason,  # 完成原因（中止）
            "weight_version": self.server_args.weight_version,
            "e2e_latency": state.time_stats.get_e2e_latency(),  # 端到端延迟
        }
        is_stream = getattr(state.obj, "stream", False)  # 判断是否为流式请求
        if getattr(state.obj, "return_logprob", False):  # 若需要返回 logprob 则填充
            self.add_logprob_to_meta_info(
                meta_info,
                state,
                state.obj.top_logprobs_num,
                state.obj.token_ids_logprob,
                state.obj.return_text_in_logprobs
                and not self.server_args.skip_tokenizer_init,
            )

        output_ids = state.output_ids  # 已生成的 token id 列表
        meta_info["completion_tokens"] = len(output_ids)  # 记录已生成 token 数
        if is_stream:  # 流式请求：只返回最后一个 token id（避免重复）
            output_ids = [output_ids[-1]] if len(output_ids) > 0 else []
        out = {
            "text": state.get_text(),  # 已生成的文本
            "output_ids": output_ids,
            "meta_info": meta_info,
        }
        state.out_list.append(out)  # 将中止输出加入输出队列
        state.event.set()  # 唤醒等待该请求输出的协程

    # 将活跃 rank 信息转发给调度器（用于数据并行负载均衡）
    def update_active_ranks(self, ranks: ActiveRanksOutput):
        self.send_to_scheduler.send_pyobj(ranks)  # 直接转发给调度器

    # 处理 open_session 请求的回调：将 session_id 设置到对应的 Future 上
    def _handle_open_session_req_output(self, recv_obj):
        future = self.session_futures.get(recv_obj.session_id)  # 找到对应的等待 Future
        if future is None:  # Future 已被清理（超时或请求取消）
            logger.warning(
                "Open session response arrived after waiter cleanup: %s",
                recv_obj.session_id,
            )
            return
        if not future.done():  # 避免重复设置 Future 结果（如重复响应）
            future.set_result(recv_obj.session_id if recv_obj.success else None)  # 成功则返回 session_id，失败返回 None

    # 处理磁盘权重更新的回调：单副本直接设置结果，多副本收集所有结果后汇总
    def _handle_update_weights_from_disk_req_output(self, recv_obj):
        if self.server_args.dp_size == 1:  # 单数据并行副本：直接设置 Future 结果
            self.model_update_result.set_result(recv_obj)
        else:  # self.server_args.dp_size > 1，多副本：收集所有副本结果后再汇总
            self.model_update_tmp.append(recv_obj)  # 追加当前副本的结果
            # set future if the all results are received
            if len(self.model_update_tmp) == self.server_args.dp_size:  # 所有副本结果到齐
                self.model_update_result.set_result(self.model_update_tmp)  # 汇总后设置 Future

    # 校验 LoRA 请求：若未启用 LoRA 则抛异常；否则解析 LoRA 路径并加载
    async def _validate_and_resolve_lora(
        self, obj: Union[GenerateReqInput, EmbeddingReqInput]
    ) -> None:
        if not obj.lora_path:  # 请求未指定 LoRA adapter，直接返回
            return

        if not self.server_args.enable_lora:  # 服务器未启用 LoRA，抛出清晰的错误提示
            first_adapter = (
                obj.lora_path  # 单个 adapter 路径
                if isinstance(obj.lora_path, str)
                else next((a for a in obj.lora_path if a), None)  # 批量请求取第一个非空 adapter
            )

            raise ValueError(
                f"LoRA adapter '{first_adapter}' was requested, but LoRA is not enabled. "
                "Please launch the server with --enable-lora flag and preload adapters "
                "using --lora-paths or /load_lora_adapter endpoint."
            )

        await self._resolve_lora_path(obj)  # 解析并（必要时重新加载）LoRA adapter

    # 解析 LoRA 路径：校验数量限制，重新加载被淘汰的 adapter，获取 lora_id
    async def _resolve_lora_path(self, obj: Union[GenerateReqInput, EmbeddingReqInput]):
        if isinstance(obj.lora_path, str):  # 单个 adapter 路径
            unique_lora_paths = set([obj.lora_path])
        else:  # 批量请求中可能包含多个不同的 adapter 路径
            unique_lora_paths = set(obj.lora_path)

        if (
            self.server_args.max_loaded_loras is not None
            and len(unique_lora_paths) > self.server_args.max_loaded_loras  # 超出最大加载 adapter 数量限制
        ):
            raise ValueError(
                f"Received request with {len(unique_lora_paths)} unique loras requested "
                f"but max loaded loras is {self.server_args.max_loaded_loras}"
            )

        # Reload all existing LoRA adapters that have been dynamically unloaded
        unregistered_loras = await self.lora_registry.get_unregistered_loras(
            unique_lora_paths  # 查询哪些 adapter 已被动态卸载
        )
        for lora_path in unregistered_loras:  # 逐个重新加载被卸载的 adapter
            if lora_path is None:  # 跳过 None（批量请求中的空槽位）
                continue

            if lora_path not in self.lora_ref_cache:  # 从未加载过的 adapter 无法隐式重载
                raise ValueError(
                    f"Got LoRA adapter that has never been loaded: {lora_path}\n"
                    f"All loaded adapters: {self.lora_ref_cache.keys()}."
                )

            logger.info(f"Reloading evicted adapter: {lora_path}")
            new_lora_ref = self.lora_ref_cache[lora_path]  # 从缓存中取出 adapter 元数据
            load_result = await self.load_lora_adapter(
                LoadLoRAAdapterReqInput(
                    lora_name=new_lora_ref.lora_name,
                    lora_path=new_lora_ref.lora_path,
                    pinned=new_lora_ref.pinned,
                )
            )
            if (
                not load_result.success
                and "already loaded" not in load_result.error_message  # "already loaded" 说明并发加载成功，非真正失败
            ):
                raise ValueError(
                    f"Failed to implicitly load LoRA adapter {lora_path}: {load_result.error_message}"
                )

        # Look up the LoRA ID from the registry and start tracking ongoing LoRA requests.
        obj.lora_id = await self.lora_registry.acquire(obj.lora_path)  # 获取 lora_id 并递增引用计数
        # Propagate lora_id to any sub-objects already cached by __getitem__.
        for i, sub_obj in obj.__dict__.get("_sub_obj_cache", {}).items():  # 将 lora_id 同步到已缓存的子对象
            sub_obj.lora_id = (
                obj.lora_id[i] if isinstance(obj.lora_id, list) else obj.lora_id  # 批量请求中每个子对象对应各自的 lora_id
            )

    # 初始化请求状态：创建 ReqState 并注册到 rid_to_state，处理 trace context
    def _init_req_state(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        created_time = obj.received_time  # 请求到达时间（用于计时）

        external_trace_header = None  # 初始化分布式追踪上下文头
        if self.server_args.enable_trace:  # 启用追踪时提取 trace context
            if obj.external_trace_header:
                # When the request comes from the rust grpc server or Engine there isn't a
                # real request object but we still need to propagate the trace context from
                # the trace context that is explicitly passed in
                external_trace_header = obj.external_trace_header  # 直接使用请求中传递的 trace header
            elif request:  # 从 HTTP 请求头中提取 trace context
                external_trace_header = extract_trace_headers(request.headers)
                obj.external_trace_header = external_trace_header  # 回写到请求对象供后续使用

        # Normalize single/batch into a uniform list of (rid, sub_obj, bootstrap_room)
        if not hasattr(obj, "is_single") or obj.is_single:  # 单个请求
            items = [(obj.rid, obj, getattr(obj, "bootstrap_room", None))]
        else:  # 批量请求：展开为 (rid, sub_obj, bootstrap_room) 三元组列表
            items = [
                (
                    obj.rid[i],  # 第 i 个请求的 rid
                    obj[i],  # 第 i 个子请求对象
                    (
                        obj.bootstrap_room[i]  # 第 i 个请求的 bootstrap_room（分离式 prefill 用）
                        if hasattr(obj, "bootstrap_room") and obj.bootstrap_room
                        else None
                    ),
                )
                for i in range(len(obj.rid))
            ]

        for rid, sub_obj, bootstrap_room in items:  # 为每个（子）请求创建状态
            if rid in self.rid_to_state:  # 检测重复 rid，防止状态覆盖
                raise ValueError(f"Duplicate request ID detected: {rid}")
            time_stats = APIServerReqTimeStats(disagg_mode=self.disaggregation_mode)  # 创建时间统计对象
            state = ReqState([], False, asyncio.Event(), sub_obj, time_stats)  # 创建请求状态
            self.rid_to_state[rid] = state  # 注册到全局状态字典
            if self.server_args.enable_trace:  # 启用追踪时初始化 trace context
                time_stats.init_trace_ctx(rid, bootstrap_room, external_trace_header)
            time_stats.set_created_time(created_time)  # 记录请求创建时间

    def _should_dispatch_to_encoder(
        self, obj: Union[GenerateReqInput, EmbeddingReqInput]
    ) -> bool:
        """Check if the request should be dispatched to encoder for processing.

        Returns True if the request should be dispatched to encoder (multiple multimodal items),
        False if it should be processed locally (single multimodal item or no multimodal items).

        Args:
            obj: The request input object

        Returns:
            bool: True if should dispatch to encoder, False otherwise
        """
        if obj.batch_size > 1:  # 批量请求暂不支持 EPD 编码器分发
            logger.warning(
                "Batch request (batch_size=%d) is not supported in EPD disaggregation mode; skipping encoder dispatch.",
                obj.batch_size,
            )
            return False
        if not isinstance(obj, GenerateReqInput) or not obj.contains_mm_input():  # 非生成请求或无多模态输入时跳过
            return False

        # Count image / video / audio items for dispatch threshold
        def _count_mm_items(data):  # 辅助函数：统计列表或单个多模态项的数量
            return (
                len(data) if isinstance(data, list) else (1 if data is not None else 0)
            )

        total_mm_items = (
            _count_mm_items(getattr(obj, "image_data", None))  # 图片数量
            + _count_mm_items(getattr(obj, "video_data", None))  # 视频数量
            + _count_mm_items(getattr(obj, "audio_data", None))  # 音频数量
        )
        return total_mm_items >= envs.SGLANG_ENCODER_DISPATCH_MIN_ITEMS.get()  # 超过阈值才分发给编码器

    # 处理 EPD 分离模式下的多模态编码请求：决定是否分发给远端编码器
    def _handle_epd_disaggregation_encode_request(
        self, obj: Union[GenerateReqInput, EmbeddingReqInput]
    ):
        """Handle EPD-disaggregation mode encoding request."""
        if isinstance(obj, GenerateReqInput) and obj.contains_mm_input():  # 只处理含多模态输入的生成请求
            # dispatch to encoder by default
            should_dispatch = True  # 默认分发给编码器
            if self.server_args.enable_adaptive_dispatch_to_encoder:  # 启用自适应分发时动态决策
                should_dispatch = self._should_dispatch_to_encoder(obj)

            # Set need_wait_for_mm_inputs flag based on whether we dispatch to encoder
            # This flag will be used in _tokenize_one_request to determine processing path
            if should_dispatch:
                obj.need_wait_for_mm_inputs = True  # 标记需要等待编码器返回多模态特征
                if self.server_args.encoder_transfer_backend == "zmq_to_scheduler":  # ZMQ 传输模式下发送编码请求
                    self.mm_receiver.send_encode_request(obj)
            else:
                obj.need_wait_for_mm_inputs = False  # 本地处理，无需等待编码器

    # 将请求的各项属性转换为 OpenTelemetry Span 属性字典（用于分布式追踪）
    def convert_to_span_attrs(
        self,
        state: ReqState,
        recv_obj: Union[
            BatchStrOutput,
            BatchEmbeddingOutput,
            BatchTokenIDOutput,
        ],
        i: int,
    ) -> Dict[str, Any]:
        """Convert attributes to span attributes."""
        span_attrs = {}  # 初始化 span 属性字典

        if not self.server_args.enable_trace:  # 未启用追踪时返回空字典
            return span_attrs

        # Token usage attributes
        if not isinstance(recv_obj, BatchEmbeddingOutput):  # embedding 无生成 token 数
            span_attrs[SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS] = (
                recv_obj.completion_tokens[i]  # 生成 token 数
            )
        span_attrs[SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS] = recv_obj.prompt_tokens[
            i  # 输入 prompt token 数
        ]
        span_attrs[SpanAttributes.GEN_AI_USAGE_CACHED_TOKENS] = recv_obj.cached_tokens[
            i  # KV 缓存命中 token 数
        ]

        # Request identifiers
        span_attrs[SpanAttributes.GEN_AI_REQUEST_ID] = (
            str(state.obj.rid) if state.obj.rid else None  # 请求 id
        )

        # Sampling parameters
        sampling_params = state.obj.sampling_params or {}  # 采样参数字典

        if max_new_tokens := sampling_params.get("max_new_tokens"):  # 最大生成 token 数
            span_attrs[SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS] = max_new_tokens

        if top_p := sampling_params.get("top_p"):  # nucleus sampling 阈值
            span_attrs[SpanAttributes.GEN_AI_REQUEST_TOP_P] = top_p

        if temperature := sampling_params.get("temperature"):  # 采样温度
            span_attrs[SpanAttributes.GEN_AI_REQUEST_TEMPERATURE] = temperature

        if top_k := sampling_params.get("top_k"):  # top-k 采样
            span_attrs[SpanAttributes.GEN_AI_REQUEST_TOP_K] = top_k

        if n := sampling_params.get("n"):  # 并行采样数量
            span_attrs[SpanAttributes.GEN_AI_REQUEST_N] = n

        # Response attributes
        span_attrs[SpanAttributes.GEN_AI_RESPONSE_MODEL] = self.served_model_name  # 模型名称

        finish_reason = (
            recv_obj.finished_reasons[i].get("type")  # 完成原因类型（stop/length/abort 等）
            if recv_obj.finished_reasons[i]
            else None
        )
        if finish_reason:
            span_attrs[SpanAttributes.GEN_AI_RESPONSE_FINISH_REASONS] = json.dumps(
                [finish_reason]  # 序列化为 JSON 列表
            )

        # Latency attributes
        span_attrs.update(state.time_stats.convert_to_gen_ai_span_attrs())  # 合并延迟统计属性

        return span_attrs  # 返回完整的 span 属性字典

    # 若启用优先级调度且请求未设置优先级，则填充默认优先级值
    def _set_default_priority(self, obj: Union[GenerateReqInput, EmbeddingReqInput]):
        """Set the default priority value."""
        if (
            self.enable_priority_scheduling  # 启用优先级调度
            and obj.priority is None  # 请求未设置优先级
            and self.default_priority_value is not None  # 有配置默认值
        ):
            obj.priority = self.default_priority_value  # 设置默认优先级


# 服务器状态枚举：用于健康检查和优雅退出判断
class ServerStatus(Enum):
    Up = "Up"  # 正常运行
    Starting = "Starting"  # 启动中
    UnHealthy = "UnHealthy"  # 健康检查失败


# 异步协程的异常包装器：捕获异常后打印堆栈、转储请求并终止进程
async def print_exception_wrapper(func):
    """
    Sometimes an asyncio function does not print exception.
    We do another wrapper to handle the exception.
    """
    try:
        await func()  # 执行被包装的异步协程
    except Exception:
        traceback = get_exception_traceback()  # 获取格式化的异常堆栈
        logger.error(f"TokenizerManager hit an exception: {traceback}")
        if hasattr(func, "__self__") and isinstance(func.__self__, TokenizerManager):
            func.__self__.dump_requests_before_crash()  # 崩溃前转储未完成请求
        kill_process_tree(os.getpid(), include_parent=True)  # 杀死整个进程树
        sys.exit(1)


# 加载 processor（tokenizer + 图像处理器）的包装函数，自动降级为快速版本
def _get_processor_wrapper(server_args):
    try:
        processor = get_processor(
            server_args.tokenizer_path,  # tokenizer/processor 路径
            tokenizer_mode=server_args.tokenizer_mode,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            use_fast=not server_args.disable_fast_image_processor,  # 默认使用快速图像处理器
            tokenizer_backend=server_args.tokenizer_backend,
        )
    except ValueError as e:
        error_message = str(e)
        if "does not have a slow version" in error_message:  # 没有慢速版本时自动切换到快速版本
            logger.info(
                f"Processor {server_args.tokenizer_path} does not have a slow version. Automatically use fast version"
            )
            processor = get_processor(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                use_fast=True,  # 强制使用快速版本
                tokenizer_backend=server_args.tokenizer_backend,
            )
        else:
            raise e  # 其他错误直接抛出
    return processor  # 返回加载好的 processor


# 根据是否跨节点决定 tensor 传输模式
def _determine_tensor_transport_mode(server_args: ServerArgs) -> TensorTransportMode:
    is_cross_node = server_args.dist_init_addr  # 有 dist_init_addr 表示跨节点部署

    if is_cross_node:
        # Fallback to default CPU transport for multi-node
        return "default"  # 多节点场景使用 CPU 传输（不支持 CUDA IPC）
    else:
        return "cuda_ipc"  # 单节点场景使用 CUDA IPC 高效传输


# 信号处理器：负责响应 SIGTERM（优雅退出）和 SIGQUIT（子进程崩溃）信号
class SignalHandler:
    def __init__(self, tokenizer_manager: TokenizerManager):
        self.tokenizer_manager = tokenizer_manager  # 持有 TokenizerManager 引用

    # 处理 SIGTERM：设置优雅退出标志，由 sigterm_watchdog 协程负责排干请求
    def sigterm_handler(self, signum=None, frame=None):
        logger.warning(
            f"SIGTERM received. {signum=} {frame=}. Draining requests and shutting down..."
        )
        self.tokenizer_manager.gracefully_exit = True  # 触发 sigterm_watchdog 开始排干请求

    # 处理 SIGQUIT：通常表示子进程崩溃，立即转储请求并终止进程树
    def running_phase_sigquit_handler(self, signum=None, frame=None):
        logger.error(
            f"SIGQUIT received. {signum=}, {frame=}. It usually means one child failed."
        )
        # Stop subprocess watchdog before killing processes to prevent false-positive
        # crash detection during normal shutdown
        if self.tokenizer_manager._subprocess_watchdog is not None:
            self.tokenizer_manager._subprocess_watchdog.stop()  # 先停止子进程看门狗，避免误报
        self.tokenizer_manager.dump_requests_before_crash()  # 转储请求到文件
        kill_process_tree(os.getpid())  # 终止整个进程树


# 请求中止处理逻辑说明：下表列出了各种场景下如何正确中止请求
# Note: request abort handling logic
# We should handle all of the following cases correctly.
#
# | entrypoint | is_streaming | status          | abort engine    | cancel asyncio task   | rid_to_state                |
# | ---------- | ------------ | --------------- | --------------- | --------------------- | --------------------------- |
# | http       | yes          | validation      | background task | fast api              | del in _handle_abort_req    |
# | http       | yes          | waiting queue   | background task | fast api              | del in _handle_abort_req    |
# | http       | yes          | running         | background task | fast api              | del in _handle_batch_output |
# | http       | no           | validation      | http exception  | http exception        | del in _handle_abort_req    |
# | http       | no           | waiting queue   | type 1          | type 1 exception      | del in _handle_abort_req    |
# | http       | no           | running         | type 3          | type 3 exception      | del in _handle_batch_output |
#
