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
"""
The entry point of inference server. (SRT = SGLang Runtime)

This file implements python APIs for the inference engine.
"""
# 推理服务器（SRT = SGLang Runtime）的入口点，实现推理引擎的 Python API

from __future__ import annotations  # 启用 PEP 563 延迟求值注解，支持前向引用类型提示

# -------- 标准库导入 --------
import asyncio          # 异步 I/O 框架 — 用于管理协程和事件循环
import atexit           # 退出钩子注册 — 程序退出时自动调用 shutdown
import dataclasses      # 数据类装饰器 — 用于定义 SchedulerInitResult 等结构体
import logging          # 日志系统 — 记录引擎启动/运行日志
import multiprocessing as mp  # 多进程模块 — 用于启动调度器/解码器子进程
import os               # 操作系统接口 — 读写环境变量、进程 PID 等
import random           # 随机数生成 — 用于生成唯一的运行 ID
import signal           # 信号处理 — 注册 SIGQUIT 处理器以捕获子进程崩溃
import threading        # 线程模块 — 修复 threading bug 并检测是否在主线程
import time             # 时间模块 — 生成带时间戳的运行 ID
from typing import (    # 类型注解工具集
    Any,                # 任意类型
    AsyncIterator,      # 异步迭代器
    Callable,           # 可调用对象类型
    Dict,               # 字典类型
    Iterator,           # 同步迭代器
    List,               # 列表类型
    Optional,           # 可选类型
    Tuple,              # 元组类型
    Union,              # 联合类型
)

# -------- 修复 Python threading 模块已知 Bug --------
# 防止 threading._register_atexit 在子进程退出时引发异常
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

# -------- 第三方库导入 --------
import torch    # PyTorch — 深度学习框架，用于张量操作和权重管理
import uvloop   # 基于 libuv 的高性能事件循环 — 替换默认 asyncio 事件循环
import zmq      # ZeroMQ — 高性能进程间通信库，用于主进程与子进程消息传递

# -------- SGLang 内部模块导入 --------
from sglang.srt.elastic_ep.expert_backup_manager import run_expert_backup_manager  # 弹性专家备份管理器 — MoE 专家容错
from sglang.srt.entrypoints.engine_info_bootstrap_server import (
    EngineInfoBootstrapServer,  # 引擎信息自举服务器 — 在多节点传输引擎时分发引擎信息
)
from sglang.srt.entrypoints.engine_score_mixin import EngineScoreMixin  # 打分混入类 — 提供 score/async_score 接口
from sglang.srt.entrypoints.EngineBase import EngineBase                # 引擎基类 — 定义引擎公共接口
from sglang.srt.managers.data_parallel_controller import (
    SCHEDULER_PIDS_ARG,                       # DP 控制器传回子调度器 PID 所用的参数键名
    run_data_parallel_controller_process,     # 数据并行控制器进程入口函数
)
from sglang.srt.managers.detokenizer_manager import run_detokenizer_process  # 解码器管理器进程入口函数
from sglang.srt.managers.io_struct import (
    CloseSessionReqInput,                      # 关闭会话请求结构
    DestroyWeightsUpdateGroupReqInput,         # 销毁权重更新组请求结构
    EmbeddingReqInput,                         # 嵌入/编码请求结构
    GenerateReqInput,                          # 文本生成请求结构
    GetWeightsByNameReqInput,                  # 按名称获取权重请求结构
    InitWeightsUpdateGroupReqInput,            # 初始化权重更新组请求结构
    LoadLoRAAdapterFromTensorsReqInput,        # 从张量加载 LoRA 适配器请求结构
    LoadLoRAAdapterReqInput,                   # 从路径加载 LoRA 适配器请求结构
    MultimodalDataInputFormat,                 # 多模态数据输入格式类型别名
    OpenSessionReqInput,                       # 打开会话请求结构
    ReleaseMemoryOccupationReqInput,           # 释放内存占用请求结构
    ResumeMemoryOccupationReqInput,            # 恢复内存占用请求结构
    RpcReqInput,                               # RPC 调用请求结构
    RpcReqOutput,                              # RPC 调用返回结构
    UnloadLoRAAdapterReqInput,                 # 卸载 LoRA 适配器请求结构
    UpdateWeightFromDiskReqInput,              # 从磁盘更新权重请求结构
    UpdateWeightsFromDistributedReqInput,      # 从分布式源更新权重请求结构
    UpdateWeightsFromIPCReqInput,              # 通过 IPC 更新权重请求结构
    UpdateWeightsFromTensorReqInput,           # 从张量更新权重请求结构
)
from sglang.srt.managers.multi_tokenizer_mixin import MultiTokenizerRouter  # 多分词器路由器 — 支持多分词器并发
from sglang.srt.managers.scheduler import run_scheduler_process              # 调度器进程入口函数
from sglang.srt.managers.template_manager import TemplateManager             # 模板管理器 — 管理聊天/完成模板
from sglang.srt.managers.tokenizer_manager import TokenizerManager           # 分词器管理器 — 主进程中的请求分词与调度
from sglang.srt.observability.trace import process_tracing_init, trace_set_thread_info  # 链路追踪初始化与线程标注
from sglang.srt.plugins import load_plugins                                  # 插件加载器 — 在引擎启动前注册插件钩子
from sglang.srt.server_args import PortArgs, ServerArgs                      # 服务端口参数和服务器配置参数
from sglang.srt.utils import (
    MultiprocessingSerializer,          # 多进程序列化工具 — 跨进程传递 Python 对象
    assert_pkg_version,                 # 依赖版本断言 — 检查 flashinfer/sglang-kernel 版本
    configure_logger,                   # 日志配置函数
    get_bool_env_var,                   # 读取布尔类型环境变量
    is_cuda,                            # 判断当前是否为 CUDA 环境
    kill_process_tree,                  # 杀死进程树（含子进程）
    launch_dummy_health_check_server,   # 启动虚拟健康检查 HTTP 服务 — 供非零节点使用
    maybe_reindex_device_id,            # 可能重新映射 GPU device id（多卡场景）
    numa_utils,                         # NUMA 亲和性配置工具
    set_prometheus_multiproc_dir,       # 设置 Prometheus 多进程目录
    set_ulimit,                         # 设置系统文件描述符上限
)
from sglang.srt.utils.network import get_zmq_socket, is_port_available  # ZMQ socket 创建工具 及 端口可用性检测
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter  # PyTorch 内存节省适配器
from sglang.srt.utils.watchdog import SubprocessWatchdog  # 子进程存活看门狗 — 检测子进程崩溃
from sglang.version import __version__  # SGLang 版本号 — 用于 get_server_info 返回

# -------- 模块级初始化 --------
logger = logging.getLogger(__name__)                     # 获取当前模块专属 logger
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())  # 将默认事件循环策略替换为高性能 uvloop

_is_cuda = is_cuda()  # 全局缓存：当前运行环境是否为 CUDA（避免重复检测）


# -------- SchedulerInitResult：调度器启动结果数据类 --------
@dataclasses.dataclass
class SchedulerInitResult:
    """Result from launching schedulers."""
    # 调度器启动结果，汇总所有调度器进程的信息与回调

    scheduler_infos: List[Dict[str, Any]]  # 每个调度器进程通过管道返回的初始化信息列表
    all_child_pids: List[int] = dataclasses.field(default_factory=list)  # 所有子进程 PID（用于进程树管理）
    wait_for_ready: Callable[[], None] = lambda: None       # 等待所有调度器就绪的回调函数
    wait_for_completion: Callable[[], None] = lambda: None  # 等待所有调度器进程退出的回调函数
    engine_info_bootstrap_server: Optional[Any] = None      # 引擎信息自举服务器实例（可选）


# -------- init_tokenizer_manager：初始化分词器管理器及模板管理器 --------
def init_tokenizer_manager(
    server_args: ServerArgs,
    port_args: PortArgs,
    TokenizerManagerClass: Optional[TokenizerManager] = None,
) -> Tuple[TokenizerManager, TemplateManager]:
    # 如未指定分词器管理器类，则使用默认 TokenizerManager
    TokenizerManagerClass = TokenizerManagerClass or TokenizerManager
    tokenizer_manager = TokenizerManagerClass(server_args, port_args)  # 实例化分词器管理器

    # 初始化聊天/完成模板
    template_manager = TemplateManager()  # 创建模板管理器实例
    template_manager.initialize_templates(
        tokenizer_manager=tokenizer_manager,          # 传入分词器管理器以获取 tokenizer
        model_path=server_args.model_path,            # 模型路径（用于自动检测模板）
        chat_template=server_args.chat_template,      # 用户指定的聊天模板
        completion_template=server_args.completion_template,  # 用户指定的补全模板
    )

    return tokenizer_manager, template_manager  # 返回初始化完成的两个管理器


# -------- Engine：推理引擎主类 --------
class Engine(EngineScoreMixin, EngineBase):
    """
    The entry point to the inference engine.

    - The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.

    Note:
    1. The HTTP server, Engine, and TokenizerManager all run in the main process.
    2. Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.
    """
    # 推理引擎由三个组件构成：
    #   1. TokenizerManager（主进程）：将请求分词后发送给调度器
    #   2. Scheduler（子进程）：接收分词后的请求，批量调度并前向推理，将输出 token 发给解码器
    #   3. DetokenizerManager（子进程）：将 token 解码为文本后返回给 TokenizerManager
    # 进程间通信通过 ZMQ IPC（每个进程使用不同端口）完成

    # -------- 类级可覆盖属性（供子类定制启动行为） --------
    server_args_class: ServerArgs = ServerArgs                              # 服务器参数类（子类可替换为自定义版本）
    init_tokenizer_manager_func: Callable = staticmethod(init_tokenizer_manager)  # 分词器管理器初始化函数
    run_scheduler_process_func: Callable = staticmethod(run_scheduler_process)    # 调度器进程启动函数
    run_detokenizer_process_func: Callable = staticmethod(run_detokenizer_process)  # 解码器进程启动函数

    # -------- __init__：引擎构造函数 --------
    def __init__(self, **kwargs):
        """
        The arguments of this function is the same as `sglang/srt/server_args.py::ServerArgs`.
        Please refer to `ServerArgs` for the documentation.
        """
        # 参数与 ServerArgs 完全相同，请参考 ServerArgs 的文档说明

        # 在构造 ServerArgs 之前确保插件已加载，
        # 使 ServerArgs.__post_init__ 中的钩子能正确触发
        load_plugins()

        # -------- 解析 server_args --------
        if "server_args" in kwargs:
            # 直接从 kwargs 中提取已有的 server_args 对象
            server_args = kwargs["server_args"]
        else:
            # 从 kwargs 中构造 server_args
            if "log_level" not in kwargs:
                # 默认不打印日志，避免 API 调用时产生大量输出
                kwargs["log_level"] = "error"
            server_args = self.server_args_class(**kwargs)  # 构建服务器参数实例
        self.server_args = server_args          # 保存服务器参数
        logger.info(f"{server_args=}")          # 记录服务器参数信息

        # 提前初始化 tokenizer_manager 为 None，
        # 防止 shutdown() 中的 atexit 回调触发 AttributeError
        self.tokenizer_manager = None

        # 注册程序退出时的自动清理钩子
        atexit.register(self.shutdown)

        # -------- 启动所有子进程 --------
        (
            tokenizer_manager,
            template_manager,
            port_args,
            scheduler_init_result,
            subprocess_watchdog,
        ) = self._launch_subprocesses(
            server_args=server_args,
            init_tokenizer_manager_func=self.init_tokenizer_manager_func,
            run_scheduler_process_func=self.run_scheduler_process_func,
            run_detokenizer_process_func=self.run_detokenizer_process_func,
        )
        self.tokenizer_manager = tokenizer_manager          # 保存分词器管理器引用
        self.template_manager = template_manager            # 保存模板管理器引用
        self._scheduler_init_result = scheduler_init_result  # 保存调度器初始化结果
        if tokenizer_manager is not None:
            tokenizer_manager._subprocess_watchdog = subprocess_watchdog  # 将看门狗绑定到分词器管理器
        self.port_args = port_args  # 保存端口参数

        # 若引擎信息自举服务器已启动，则获取传输引擎信息
        if scheduler_init_result.engine_info_bootstrap_server is not None:
            self.remote_instance_transfer_engine_info = (
                scheduler_init_result.engine_info_bootstrap_server.transfer_engine_info
            )

        # -------- 初始化 ZMQ RPC 套接字 --------
        context = zmq.Context(2)  # 创建 ZMQ 上下文，I/O 线程数为 2
        if self.server_args.node_rank == 0:
            # 仅主节点（rank 0）需要 RPC 发送套接字
            self.send_to_rpc = get_zmq_socket(
                context, zmq.DEALER, self.port_args.rpc_ipc_name, True
            )
        else:
            self.send_to_rpc = None  # 非主节点不使用 RPC 套接字

        # -------- 初始化链路追踪（可选） --------
        if server_args.enable_trace:
            process_tracing_init(server_args.otlp_traces_endpoint, "sglang")  # 初始化 OTLP 追踪导出
            thread_label = "Tokenizer"  # 默认线程标签
            if server_args.disaggregation_mode == "prefill":
                thread_label = "Prefill Tokenizer"   # 预填充解耦模式标签
            elif server_args.disaggregation_mode == "decode":
                thread_label = "Decode Tokenizer"    # 解码解耦模式标签
            trace_set_thread_info(thread_label)  # 设置当前线程的追踪标签

        # -------- 获取或创建事件循环 --------
        try:
            self.loop = asyncio.get_running_loop()   # 尝试获取当前已运行的事件循环
        except RuntimeError:
            self.loop = asyncio.new_event_loop()     # 若不存在则创建新事件循环
            asyncio.set_event_loop(self.loop)        # 将新循环设为当前线程的默认循环

    # -------- get_all_child_pids：获取所有子进程 PID --------
    def get_all_child_pids(self) -> List[int]:
        """Returns a list of all child process PIDs."""
        # 返回所有调度器及解码器子进程的 PID 列表
        return self._scheduler_init_result.all_child_pids

    # -------- _resolve_routed_dp_rank：解析并验证数据并行路由 rank --------
    def _resolve_routed_dp_rank(
        self,
        routed_dp_rank: Optional[int],
        data_parallel_rank: Optional[int],
    ) -> Optional[int]:
        if data_parallel_rank is not None:
            import warnings  # 仅在需要时导入 warnings 模块

            # 向用户发出弃用警告：data_parallel_rank 已被 routed_dp_rank 替代
            warnings.warn(
                "'data_parallel_rank' is deprecated, use 'routed_dp_rank' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            if routed_dp_rank is None:
                routed_dp_rank = data_parallel_rank  # 用旧参数值填充新参数

        if routed_dp_rank is not None:
            dp_size = self.server_args.dp_size  # 获取数据并行度
            if dp_size <= 1 and routed_dp_rank == 0:
                # dp_size=1 时 rank=0 无意义，忽略并警告
                logger.warning(
                    f"routed_dp_rank={routed_dp_rank} is ignored because dp_size={dp_size}"
                )
                return None
            if routed_dp_rank < 0 or routed_dp_rank >= dp_size:
                # rank 超出合法范围，抛出异常
                raise ValueError(
                    f"routed_dp_rank={routed_dp_rank} out of range [0, {dp_size})"
                )

        logger.debug(f"routed_dp_rank: {routed_dp_rank}")  # 调试日志：记录最终使用的 dp rank
        return routed_dp_rank  # 返回解析后的 dp rank


    # -------- generate：同步文本生成接口 --------
    def generate(
        self,
        # 输入提示词，支持单条字符串或批量列表
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,  # 采样参数（temperature、top_p 等）
        # 输入 token id，与 prompt 二选一
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        # 图像输入，可为图像实例、文件路径、URL 或 base64 字符串；
        # 支持：单图、图像列表（一请求一图）、图像列表的列表（一请求多图）、
        #       HuggingFace processor 输出（dict 含 format='processor_output'）、
        #       预计算嵌入（dict 含 format='precomputed_embedding'）
        image_data: Optional[MultimodalDataInputFormat] = None,
        audio_data: Optional[MultimodalDataInputFormat] = None,   # 音频输入（同图像格式规范）
        video_data: Optional[MultimodalDataInputFormat] = None,   # 视频输入（同图像格式规范）
        return_logprob: Optional[Union[List[bool], bool]] = False,  # 是否返回对数概率
        logprob_start_len: Optional[Union[List[int], int]] = None,  # 对数概率起始 token 位置
        top_logprobs_num: Optional[Union[List[int], int]] = None,   # 每步返回 top-k 对数概率数量
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,  # 指定 token 的对数概率
        lora_path: Optional[List[Optional[str]]] = None,            # LoRA 适配器路径列表
        custom_logit_processor: Optional[Union[List[str], str]] = None,  # 自定义 logit 处理器
        return_hidden_states: bool = False,    # 是否返回隐状态
        return_routed_experts: bool = False,   # 是否返回 MoE 路由专家信息
        stream: bool = False,                  # 是否流式返回（逐 token 输出）
        bootstrap_host: Optional[Union[List[str], str]] = None,   # 传输引擎自举主机地址
        bootstrap_port: Optional[Union[List[int], int]] = None,   # 传输引擎自举端口
        bootstrap_room: Optional[Union[List[int], int]] = None,   # 传输引擎房间号
        routed_dp_rank: Optional[int] = None,          # 指定路由到的数据并行 rank
        disagg_prefill_dp_rank: Optional[int] = None,  # 解耦预填充场景下的 DP rank
        # 已弃用：请使用 routed_dp_rank 替代
        data_parallel_rank: Optional[int] = None,
        external_trace_header: Optional[Dict] = None,  # 外部追踪上下文头（分布式追踪）
        rid: Optional[Union[List[str], str]] = None,    # 请求 ID（用于去重/追踪）
        session_params: Optional[Dict] = None,          # 会话参数（多轮对话场景）
        priority: Optional[int] = None,                 # 请求优先级
    ) -> Union[Dict, Iterator[Dict]]:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::GenerateReqInput`.
        Please refer to `GenerateReqInput` for the documentation.
        """
        # 解析并验证 data_parallel_rank（旧参数）与 routed_dp_rank 的关系
        routed_dp_rank = self._resolve_routed_dp_rank(
            routed_dp_rank, data_parallel_rank
        )

        # 构造生成请求对象
        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            audio_data=audio_data,
            video_data=video_data,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            lora_path=lora_path,
            custom_logit_processor=custom_logit_processor,
            return_hidden_states=return_hidden_states,
            return_routed_experts=return_routed_experts,
            stream=stream,
            bootstrap_host=bootstrap_host,
            bootstrap_port=bootstrap_port,
            bootstrap_room=bootstrap_room,
            routed_dp_rank=routed_dp_rank,
            disagg_prefill_dp_rank=disagg_prefill_dp_rank,
            external_trace_header=external_trace_header,
            rid=rid,
            session_params=session_params,
            priority=priority,
        )
        generator = self.tokenizer_manager.generate_request(obj, None)  # 提交请求到分词器管理器，获得异步生成器

        if stream:
            # 流式模式：将异步生成器包装为同步迭代器
            def generator_wrapper():
                while True:
                    try:
                        chunk = self.loop.run_until_complete(generator.__anext__())  # 逐个取出 token 块
                        yield chunk  # 逐步产出给调用方
                    except StopAsyncIteration:
                        break  # 生成完毕，退出循环

            return generator_wrapper()  # 返回同步生成器
        else:
            # 非流式模式：等待完整结果后一次性返回
            ret = self.loop.run_until_complete(generator.__anext__())
            return ret

    # -------- async_generate：异步文本生成接口 --------
    async def async_generate(
        self,
        # 输入提示词，支持单条字符串或批量列表
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,  # 采样参数
        # 输入 token id，与 prompt 二选一
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        # 图像输入（格式说明同 generate）
        image_data: Optional[MultimodalDataInputFormat] = None,
        audio_data: Optional[MultimodalDataInputFormat] = None,   # 音频输入
        video_data: Optional[MultimodalDataInputFormat] = None,   # 视频输入
        return_logprob: Optional[Union[List[bool], bool]] = False,  # 是否返回对数概率
        logprob_start_len: Optional[Union[List[int], int]] = None,  # 对数概率起始位置
        top_logprobs_num: Optional[Union[List[int], int]] = None,   # top-k 对数概率数量
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,  # 指定 token 的对数概率
        lora_path: Optional[List[Optional[str]]] = None,            # LoRA 适配器路径
        custom_logit_processor: Optional[Union[List[str], str]] = None,  # 自定义 logit 处理器
        return_hidden_states: bool = False,    # 是否返回隐状态
        return_routed_experts: bool = False,   # 是否返回路由专家信息
        stream: bool = False,                  # 是否流式返回
        bootstrap_host: Optional[Union[List[str], str]] = None,   # 传输引擎自举主机
        bootstrap_port: Optional[Union[List[int], int]] = None,   # 传输引擎自举端口
        bootstrap_room: Optional[Union[List[int], int]] = None,   # 传输引擎房间号
        routed_dp_rank: Optional[int] = None,          # 路由 DP rank
        disagg_prefill_dp_rank: Optional[int] = None,  # 解耦预填充 DP rank
        # 已弃用：请使用 routed_dp_rank 替代
        data_parallel_rank: Optional[int] = None,
        external_trace_header: Optional[Dict] = None,  # 外部追踪头
        rid: Optional[Union[List[str], str]] = None,    # 请求 ID
        session_params: Optional[Dict] = None,          # 会话参数
        priority: Optional[int] = None,                 # 请求优先级
    ) -> Union[Dict, AsyncIterator[Dict]]:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::GenerateReqInput`.
        Please refer to `GenerateReqInput` for the documentation.
        """
        # 解析 dp rank（兼容旧参数 data_parallel_rank）
        routed_dp_rank = self._resolve_routed_dp_rank(
            routed_dp_rank, data_parallel_rank
        )

        # 构造生成请求对象
        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            audio_data=audio_data,
            video_data=video_data,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            lora_path=lora_path,
            return_hidden_states=return_hidden_states,
            return_routed_experts=return_routed_experts,
            stream=stream,
            custom_logit_processor=custom_logit_processor,
            bootstrap_host=bootstrap_host,
            bootstrap_port=bootstrap_port,
            bootstrap_room=bootstrap_room,
            routed_dp_rank=routed_dp_rank,
            disagg_prefill_dp_rank=disagg_prefill_dp_rank,
            external_trace_header=external_trace_header,
            rid=rid,
            session_params=session_params,
            priority=priority,
        )
        generator = self.tokenizer_manager.generate_request(obj, None)  # 提交请求，获得异步生成器

        if stream is True:
            return generator          # 流式模式：直接返回异步生成器供调用方 async for 消费
        else:
            return await generator.__anext__()  # 非流式模式：await 取出完整结果后返回

    # -------- encode：同步嵌入编码接口 --------
    def encode(
        self,
        prompt: Union[str, List[str], List[Dict], List[List[Dict]]],  # 输入文本（支持多种格式）
        image_data: Optional[MultimodalDataInputFormat] = None,         # 图像输入
        audio_data: Optional[MultimodalDataInputFormat] = None,         # 音频输入
        video_data: Optional[MultimodalDataInputFormat] = None,         # 视频输入
        dimensions: Optional[int] = None,                               # 输出嵌入维度（部分模型支持）
        lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None,  # LoRA 适配器路径
        embed_override_token_id: Optional[int] = None,                  # 覆盖嵌入的 token id
        embed_overrides: Optional[List[List[torch.Tensor]]] = None,     # 自定义嵌入覆盖张量列表
        external_trace_header: Optional[Dict] = None,                   # 外部追踪头
        rid: Optional[Union[List[str], str]] = None,                    # 请求 ID
    ) -> Dict:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::EmbeddingReqInput`.
        Please refer to `EmbeddingReqInput` for the documentation.
        """
        # 构造嵌入请求对象
        obj = EmbeddingReqInput(
            text=prompt,
            image_data=image_data,
            audio_data=audio_data,
            video_data=video_data,
            dimensions=dimensions,
            lora_path=lora_path,
            embed_override_token_id=embed_override_token_id,
            embed_overrides=embed_overrides,
            external_trace_header=external_trace_header,
            rid=rid,
        )
        generator = self.tokenizer_manager.generate_request(obj, None)  # 提交嵌入请求
        ret = self.loop.run_until_complete(generator.__anext__())        # 同步等待嵌入结果
        return ret  # 返回嵌入向量字典

    # -------- async_encode：异步嵌入编码接口 --------
    async def async_encode(
        self,
        prompt: Union[str, List[str], List[Dict], List[List[Dict]]],  # 输入文本
        image_data: Optional[MultimodalDataInputFormat] = None,         # 图像输入
        audio_data: Optional[MultimodalDataInputFormat] = None,         # 音频输入
        video_data: Optional[MultimodalDataInputFormat] = None,         # 视频输入
        dimensions: Optional[int] = None,                               # 嵌入维度
        lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None,  # LoRA 路径
        embed_override_token_id: Optional[int] = None,                  # 覆盖嵌入 token id
        embed_overrides: Optional[List[List[torch.Tensor]]] = None,     # 自定义嵌入覆盖
        external_trace_header: Optional[Dict] = None,                   # 外部追踪头
        rid: Optional[Union[List[str], str]] = None,                    # 请求 ID
    ) -> Dict:
        """
        Asynchronous version of encode method.

        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::EmbeddingReqInput`.
        Please refer to `EmbeddingReqInput` for the documentation.
        """
        # encode 的异步版本，参数与 encode 相同
        # 构造嵌入请求对象
        obj = EmbeddingReqInput(
            text=prompt,
            image_data=image_data,
            audio_data=audio_data,
            video_data=video_data,
            dimensions=dimensions,
            lora_path=lora_path,
            embed_override_token_id=embed_override_token_id,
            embed_overrides=embed_overrides,
            external_trace_header=external_trace_header,
            rid=rid,
        )
        generator = self.tokenizer_manager.generate_request(obj, None)  # 提交嵌入请求
        return await generator.__anext__()  # 异步等待并返回嵌入结果

    # -------- rerank：交叉编码器重排序接口 --------
    def rerank(
        self,
        prompt: Union[List[List[str]]],  # 输入文本对列表（[query, passage] 格式）
    ) -> Dict:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::EmbeddingReqInput`.
        Please refer to `EmbeddingReqInput` for the documentation.
        """
        # 构造交叉编码器重排序请求（标记为 cross_encoder 请求）
        obj = EmbeddingReqInput(text=prompt, is_cross_encoder_request=True)
        generator = self.tokenizer_manager.generate_request(obj, None)  # 提交重排序请求
        ret = self.loop.run_until_complete(generator.__anext__())        # 同步等待重排序结果
        return ret  # 返回相关性分数字典

    # -------- _launch_scheduler_processes：类方法，启动调度器子进程 --------
    @classmethod
    def _launch_scheduler_processes(
        cls,
        server_args: ServerArgs,
        port_args: PortArgs,
        run_scheduler_process_func: Callable,
    ) -> Tuple[SchedulerInitResult, Optional[List]]:
        """Launch scheduler processes using multiprocessing.
        Override in subclasses for different backends (e.g. Ray).

        Returns:
            Tuple of (SchedulerInitResult, scheduler_procs).
            scheduler_procs is None for RayEngine (uses Ray actors instead).
        """
        # 使用 multiprocessing 启动调度器进程；子类可覆盖此方法以支持 Ray 等后端
        # 返回 (SchedulerInitResult, scheduler_procs)；RayEngine 中 scheduler_procs 为 None
        scheduler_procs = []  # 收集所有调度器进程对象

        if server_args.dp_size == 1:
            # -------- 单数据并行：直接启动张量并行调度器进程 --------
            memory_saver_adapter = TorchMemorySaverAdapter.create(
                enable=server_args.enable_memory_saver  # 根据配置决定是否启用内存节省模式
            )
            scheduler_pipe_readers = []  # 收集每个调度器的管道读取端

            # 计算当前节点负责的 PP rank 范围和 TP rank 范围
            pp_rank_range, tp_rank_range, pp_size_per_node, tp_size_per_node = (
                _calculate_rank_ranges(
                    server_args.nnodes,    # 总节点数
                    server_args.pp_size,   # 流水线并行度
                    server_args.tp_size,   # 张量并行度
                    server_args.node_rank, # 当前节点 rank
                )
            )

            # 遍历 PP rank 和 TP rank 的组合，为每个组合启动一个调度器进程
            for pp_rank in pp_rank_range:
                for tp_rank in tp_rank_range:
                    reader, writer = mp.Pipe(duplex=False)  # 创建单向管道，用于调度器->主进程通信
                    # 计算该 (pp_rank, tp_rank) 对应的 GPU id
                    gpu_id = (
                        server_args.base_gpu_id
                        + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                        + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                    )
                    # 计算注意力 CP rank、MoE DP rank 和 MoE EP rank
                    attn_cp_rank, moe_dp_rank, moe_ep_rank = _compute_parallelism_ranks(
                        server_args, tp_rank
                    )

                    with maybe_reindex_device_id(gpu_id) as gpu_id:  # 可能重映射 GPU id
                        proc = mp.Process(
                            target=run_scheduler_process_func,  # 调度器进程入口函数
                            args=(
                                server_args,
                                port_args,
                                gpu_id,        # 分配的 GPU id
                                tp_rank,       # 张量并行 rank
                                attn_cp_rank,  # 注意力上下文并行 rank
                                moe_dp_rank,   # MoE 数据并行 rank
                                moe_ep_rank,   # MoE 专家并行 rank
                                pp_rank,       # 流水线并行 rank
                                None,          # dp_rank（单 DP 时为 None）
                                writer,        # 管道写入端（子进程用于发送就绪信号）
                            ),
                        )
                        # 在内存节省和 NUMA 亲和性配置下启动子进程
                        with memory_saver_adapter.configure_subprocess(), numa_utils.configure_subprocess(
                            server_args, gpu_id
                        ):
                            proc.start()  # 启动调度器子进程

                    scheduler_procs.append(proc)            # 记录进程对象
                    scheduler_pipe_readers.append(reader)   # 记录管道读取端
        else:
            # -------- 多数据并行：启动 DP 控制器进程（由其再启动各 DP 分片的调度器） --------
            reader, writer = mp.Pipe(duplex=False)   # 创建单向管道
            scheduler_pipe_readers = [reader]         # DP 控制器只有一个管道读取端
            proc = mp.Process(
                target=run_data_parallel_controller_process,  # DP 控制器进程入口
                kwargs=dict(
                    server_args=server_args,
                    port_args=port_args,
                    pipe_writer=writer,                           # 就绪信号写入端
                    run_scheduler_process_func=run_scheduler_process_func,  # 传入调度器启动函数
                ),
            )
            proc.start()                    # 启动 DP 控制器进程
            scheduler_procs.append(proc)    # 记录进程对象

        # -------- 收集所有子进程 PID（用于进程树管理） --------
        all_child_pids = [proc.pid for proc in scheduler_procs]
        scheduler_infos = []  # 初始化调度器信息列表（待就绪后填充）

        # 等待所有调度器就绪的回调函数
        def wait_for_ready():
            infos = _wait_for_scheduler_ready(scheduler_pipe_readers, scheduler_procs)
            scheduler_infos.extend(infos)  # 将就绪信息存入外层列表
            # dp_size > 1 时，从 DP 控制器收集子调度器的 PID
            if server_args.dp_size > 1:
                for info in infos:
                    if SCHEDULER_PIDS_ARG in info:
                        all_child_pids.extend(info[SCHEDULER_PIDS_ARG])  # 追加子调度器 PID

        # 等待所有调度器进程退出的回调函数
        def wait_for_completion():
            for proc in scheduler_procs:
                proc.join()  # 阻塞等待进程退出
                logger.error(
                    f"Scheduler or DataParallelController {proc.pid} "
                    f"terminated with {proc.exitcode}"  # 记录非正常退出的错误日志
                )

        return (
            SchedulerInitResult(
                scheduler_infos=scheduler_infos,             # 调度器初始化信息列表
                all_child_pids=all_child_pids,               # 所有子进程 PID
                wait_for_ready=wait_for_ready,               # 就绪等待回调
                wait_for_completion=wait_for_completion,     # 完成等待回调
            ),
            scheduler_procs,  # 调度器进程对象列表
        )

    # -------- _launch_subprocesses：类方法，启动所有子进程并返回各管理器 --------
    @classmethod
    def _launch_subprocesses(
        cls,
        server_args: ServerArgs,
        init_tokenizer_manager_func: Callable,
        run_scheduler_process_func: Callable,
        run_detokenizer_process_func: Callable,
        port_args: Optional[PortArgs] = None,
    ) -> Tuple[
        TokenizerManager,
        TemplateManager,
        PortArgs,
        SchedulerInitResult,
        Optional[SubprocessWatchdog],
    ]:
        """Launch the TokenizerManager in the main process, the Scheduler in a subprocess, and the DetokenizerManager in another subprocess.

        Returns:
            Tuple of (tokenizer_manager, template_manager, port_args, scheduler_init_result, subprocess_watchdog).
        """
        # 在主进程中启动 TokenizerManager，在子进程中启动 Scheduler 和 DetokenizerManager
        # 返回：(tokenizer_manager, template_manager, port_args, scheduler_init_result, subprocess_watchdog)

        # -------- 全局环境配置 --------
        configure_logger(server_args)         # 根据服务器参数配置日志级别和格式
        _set_envs_and_config(server_args)     # 设置 CUDA/NCCL 等全局环境变量

        # 确保插件已加载（可能已由 Engine.__init__ 或 CLI 入口加载）
        load_plugins()

        server_args.check_server_args()  # 校验服务器参数（互斥选项检查等）
        _set_gc(server_args)             # 按需配置 Python GC 阈值

        # -------- 端口分配 --------
        if port_args is None:
            port_args = PortArgs.init_new(server_args)  # 自动分配所有 IPC 端口
        logger.info(f"{server_args=}")  # 记录最终服务器参数

        # -------- 启动引擎信息自举服务器（可选，用于多节点传输引擎场景） --------
        engine_info_bootstrap_server = None
        if (
            server_args.remote_instance_weight_loader_start_seed_via_transfer_engine
            and server_args.node_rank == 0  # 仅主节点启动自举服务器
        ):
            bootstrap_port = server_args.engine_info_bootstrap_port  # 获取自举服务器端口
            if not is_port_available(bootstrap_port):
                # 端口被占用时抛出明确错误（多实例场景需各实例使用不同端口）
                raise RuntimeError(
                    f"engine_info_bootstrap_port {bootstrap_port} is already in use. "
                    f"When running multiple instances on the same node, each instance must use a "
                    f"different --engine-info-bootstrap-port."
                )
            engine_info_bootstrap_server = EngineInfoBootstrapServer(
                host=server_args.host, port=bootstrap_port  # 绑定主机和端口
            )

        # -------- 启动调度器子进程 --------
        scheduler_init_result, scheduler_procs = cls._launch_scheduler_processes(
            server_args, port_args, run_scheduler_process_func
        )
        # 将自举服务器实例绑定到调度器初始化结果
        scheduler_init_result.engine_info_bootstrap_server = (
            engine_info_bootstrap_server
        )

        # -------- 弹性专家备份（可选，MoE 容错） --------
        if (
            server_args.enable_elastic_expert_backup
            and server_args.elastic_ep_backend is not None
        ):
            run_expert_backup_manager(server_args, port_args)  # 启动专家备份管理器

        # -------- 多节点非零 rank 节点：无需运行分词器/解码器，直接等待 --------
        if server_args.node_rank >= 1:
            # 非零 rank 节点只需等待调度器就绪，不参与分词和解码
            scheduler_init_result.wait_for_ready()

            if os.getenv("SGLANG_BLOCK_NONZERO_RANK_CHILDREN") == "0":
                # 以 Python API 方式使用时，不阻塞非零 rank 节点
                return (
                    None,   # tokenizer_manager（非零节点无需）
                    None,   # template_manager（非零节点无需）
                    port_args,
                    scheduler_init_result,
                    None,   # subprocess_watchdog（非零节点无需）
                )

            # 启动虚拟健康检查服务，供外部探活
            launch_dummy_health_check_server(
                server_args.host, server_args.port, server_args.enable_metrics
            )

            # 阻塞直至调度器进程全部退出（服务器停止后返回）
            scheduler_init_result.wait_for_completion()
            return (
                None,   # tokenizer_manager
                None,   # template_manager
                port_args,
                scheduler_init_result,
                None,   # subprocess_watchdog
            )

        # -------- 启动解码器子进程（仅主节点 rank 0 执行） --------
        detoken_proc = mp.Process(
            target=run_detokenizer_process_func,  # 解码器进程入口函数
            args=(
                server_args,
                port_args,
            ),
        )
        detoken_proc.start()  # 启动解码器子进程
        scheduler_init_result.all_child_pids.append(detoken_proc.pid)  # 记录解码器 PID

        # -------- 初始化分词器管理器（解码器进程启动后，引擎信息自举服务器在此初始化） --------
        if server_args.tokenizer_worker_num == 1:
            # 单分词器模式
            tokenizer_manager, template_manager = init_tokenizer_manager_func(
                server_args, port_args
            )
        else:
            # 多分词器路由模式
            tokenizer_manager = MultiTokenizerRouter(server_args, port_args)  # 使用多分词器路由器
            template_manager = None  # 多分词器模式暂不使用模板管理器

        # -------- 等待模型加载完毕 --------
        scheduler_init_result.wait_for_ready()

        # 将调度器返回的最大请求输入长度同步到分词器管理器
        tokenizer_manager.max_req_input_len = scheduler_init_result.scheduler_infos[0][
            "max_req_input_len"
        ]

        # -------- 启动子进程存活看门狗 --------
        # 注意：RayEngine 中 scheduler_procs=None（使用 Ray Actor，而非 mp.Process）
        processes = list(scheduler_procs or [])                           # 调度器进程列表
        names = [f"scheduler_{i}" for i in range(len(processes))]        # 调度器进程名称列表
        processes.append(detoken_proc)                                    # 追加解码器进程
        names.append("detokenizer")                                       # 追加解码器名称
        subprocess_watchdog = SubprocessWatchdog(
            processes=processes, process_names=names                      # 创建看门狗，监控所有子进程
        )
        subprocess_watchdog.start()  # 启动看门狗线程

        # 返回所有已初始化的管理器和配置
        return (
            tokenizer_manager,       # 分词器管理器
            template_manager,        # 模板管理器
            port_args,               # 端口参数
            scheduler_init_result,   # 调度器初始化结果
            subprocess_watchdog,     # 子进程看门狗
        )

    # -------- shutdown：关闭引擎并等待子进程释放 GPU --------
    def shutdown(self):
        """Shutdown the engine; block until the scheduler subprocess releases
        its GPU context so the caller can immediately reallocate on the same
        device."""
        # 关闭引擎，阻塞直到调度器子进程释放 GPU 上下文（便于调用方立即重分配同一设备）
        if (
            self.tokenizer_manager is not None
            and self.tokenizer_manager._subprocess_watchdog is not None
        ):
            self.tokenizer_manager._subprocess_watchdog.stop()  # 先停止看门狗，避免误报
        kill_process_tree(os.getpid(), include_parent=False, wait_timeout=60)  # 杀死所有子进程（不含自身），等待最多 60 秒

    # -------- 上下文管理器支持（with 语句） --------
    def __enter__(self):
        return self  # 进入 with 块时返回引擎自身

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()  # 退出 with 块时自动关闭引擎
        return False     # 不抑制异常，让异常继续传播

    # -------- flush_cache：清空 KV 缓存 --------
    def flush_cache(self):
        return self.loop.run_until_complete(self.tokenizer_manager.flush_cache())  # 同步等待缓存清空完成

    # -------- open_session：开启多轮对话会话 --------
    def open_session(
        self,
        capacity_of_str_len: int,          # 会话允许的最大字符串长度容量
        session_id: Optional[str] = None,  # 可选的会话 ID；若不提供则自动生成 UUID
        streaming: bool = False,           # 是否使用低开销的实时流式路径（仅追加模式）
        timeout: Optional[float] = None,   # 会话超时时间（秒）；超时后自动关闭
    ) -> str:
        """Open a session for multi-turn conversation with shared context.

        Args:
            capacity_of_str_len: Maximum string length capacity for the session.
            session_id: Optional session ID. If not provided, a UUID will be generated.
            streaming: Use low-overhead path for realtime streaming (append-only mode).
            timeout: If set, the session is automatically closed after being inactive
                for this many seconds. Inactivity is measured from session open or the
                most recent request submission.

        Returns:
            The session ID (either the provided one or a newly generated UUID).
        """
        # 构造开启会话请求对象
        obj = OpenSessionReqInput(
            capacity_of_str_len=capacity_of_str_len,
            session_id=session_id,
            streaming=streaming,
            timeout=timeout,
        )
        return self.loop.run_until_complete(
            self.tokenizer_manager.open_session(obj, None)  # 同步等待会话创建，返回会话 ID
        )

    # -------- close_session：关闭会话并释放资源 --------
    def close_session(self, session_id: str) -> None:
        """Close a session and release its resources.

        Args:
            session_id: The session ID to close.
        """
        obj = CloseSessionReqInput(session_id=session_id)  # 构造关闭会话请求
        self.loop.run_until_complete(self.tokenizer_manager.close_session(obj, None))  # 同步等待会话关闭

    # -------- start_profile：启动性能分析 --------
    def start_profile(self, **kwargs):
        self.loop.run_until_complete(self.tokenizer_manager.start_profile(**kwargs))  # 同步启动分析器

    # -------- stop_profile：停止性能分析 --------
    def stop_profile(self):
        self.loop.run_until_complete(self.tokenizer_manager.stop_profile())  # 同步停止分析器

    # -------- start_expert_distribution_record：开始记录专家分布（MoE 场景） --------
    def start_expert_distribution_record(self):
        self.loop.run_until_complete(
            self.tokenizer_manager.start_expert_distribution_record()  # 同步启动专家分布记录
        )

    # -------- stop_expert_distribution_record：停止专家分布记录 --------
    def stop_expert_distribution_record(self):
        self.loop.run_until_complete(
            self.tokenizer_manager.stop_expert_distribution_record()  # 同步停止专家分布记录
        )

    # -------- dump_expert_distribution_record：导出专家分布记录 --------
    def dump_expert_distribution_record(self):
        self.loop.run_until_complete(
            self.tokenizer_manager.dump_expert_distribution_record()  # 同步导出专家分布统计数据
        )

    # -------- get_server_info：获取服务器状态与版本信息 --------
    def get_server_info(self):
        internal_states = self.loop.run_until_complete(
            self.tokenizer_manager.get_internal_state()  # 获取分词器管理器内部状态
        )
        return {
            **dataclasses.asdict(self.tokenizer_manager.server_args),  # 展开服务器参数字典
            **self._scheduler_init_result.scheduler_infos[0],          # 展开第一个调度器信息
            "internal_states": internal_states,                        # 内部状态快照
            "version": __version__,                                    # SGLang 版本号
        }

    # -------- init_weights_update_group：初始化权重更新通信组（RLHF/在线训练场景） --------
    def init_weights_update_group(
        self,
        master_address: str,   # 分布式通信主节点地址
        master_port: int,      # 分布式通信主节点端口
        rank_offset: int,      # 当前引擎在全局 rank 中的偏移量
        world_size: int,       # 通信组的总进程数
        group_name: str,       # 通信组名称（用于标识/隔离不同更新组）
        backend: str = "nccl", # 通信后端，默认使用 NCCL
    ):
        """Initialize parameter update group."""
        # 初始化权重更新通信组（用于 RLHF 等在线训练场景的权重同步）
        obj = InitWeightsUpdateGroupReqInput(
            master_address=master_address,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
        )
        return self.loop.run_until_complete(
            self.tokenizer_manager.init_weights_update_group(obj, None)  # 同步等待通信组初始化
        )

    # -------- destroy_weights_update_group：销毁权重更新通信组 --------
    def destroy_weights_update_group(
        self,
        group_name: str,  # 要销毁的通信组名称
    ):
        """Destroy parameter update group."""
        # 销毁指定名称的权重更新通信组，释放相关资源
        obj = DestroyWeightsUpdateGroupReqInput(
            group_name=group_name,
        )
        return self.loop.run_until_complete(
            self.tokenizer_manager.destroy_weights_update_group(obj, None)  # 同步等待通信组销毁
        )

    # -------- update_weights_from_distributed：从分布式源更新权重 --------
    def update_weights_from_distributed(
        self,
        names: list[str],              # 要更新的权重参数名列表
        dtypes: list[str],             # 各参数对应的数据类型列表
        shapes: list[list[int]],       # 各参数对应的形状列表
        group_name: str = "weight_update_group",  # 使用的通信组名称
        flush_cache: bool = True,       # 更新后是否清空 KV 缓存
        load_format: Optional[str] = None,        # 加载格式（如 'flattened_bucket'）
    ):
        """Update weights from distributed source."""
        # 通过分布式通信组将新权重广播到推理引擎（用于 RLHF 训练后热更新）
        obj = UpdateWeightsFromDistributedReqInput(
            names=names,
            dtypes=dtypes,
            shapes=shapes,
            group_name=group_name,
            flush_cache=flush_cache,
            load_format=load_format,
        )
        return self.loop.run_until_complete(
            self.tokenizer_manager.update_weights_from_distributed(obj, None)  # 同步等待权重更新
        )

    # -------- update_weights_from_tensor：从张量更新权重 --------
    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]],  # (参数名, 张量) 元组列表
        load_format: Optional[str] = None,               # 加载格式
        flush_cache: bool = True,                        # 更新后是否清空 KV 缓存
    ):
        """Update weights from distributed source. If there are going to be more updates, set `flush_cache` to be false
        to avoid duplicated cache cleaning operation."""
        # 从本地张量直接更新模型权重；若后续还有更多更新，将 flush_cache=False 以避免重复清缓存
        if load_format == "flattened_bucket":
            # flattened_bucket 格式：张量已经序列化，直接使用
            serialized_named_tensors = named_tensors
        else:
            # 其他格式：对每个 TP rank 分别序列化张量（多进程间传递）
            serialized_named_tensors = [
                MultiprocessingSerializer.serialize(named_tensors)
                for _ in range(self.server_args.tp_size)  # 每个 TP rank 需要一份序列化副本
            ]
        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=serialized_named_tensors,
            load_format=load_format,
            flush_cache=flush_cache,
        )
        return self.loop.run_until_complete(
            self.tokenizer_manager.update_weights_from_tensor(obj, None)  # 同步等待权重更新完成
        )

    # -------- update_weights_from_disk：从磁盘热加载模型权重 --------
    def update_weights_from_disk(
        self,
        model_path: str,                    # 新权重的磁盘路径
        load_format: Optional[str] = None,  # 加载格式（None 表示自动检测）
    ):
        """Update the weights from disk inplace without re-launching the engine.

        This method allows updating the model weights from disk without restarting
        the engine. It can be used to load a different model or update weights with
        new training.
        """
        # 不重启引擎，直接从磁盘热加载模型权重（可用于加载新训练的 checkpoint）
        obj = UpdateWeightFromDiskReqInput(
            model_path=model_path,
            load_format=load_format,
        )

        return self.loop.run_until_complete(
            self.tokenizer_manager.update_weights_from_disk(obj, None)  # 同步等待磁盘权重加载完成
        )

    # -------- update_weights_from_ipc：通过 IPC（ZMQ）更新权重 --------
    def update_weights_from_ipc(
        self,
        zmq_handles: Dict[str, str],  # ZMQ 内存句柄字典（参数名 -> IPC handle 字符串）
        flush_cache: bool = True,      # 更新后是否清空 KV 缓存
    ):
        """Update weights from IPC for checkpoint-engine integration."""
        # 通过 ZMQ IPC 内存句柄直接更新权重（用于 checkpoint-engine 集成场景）
        obj = UpdateWeightsFromIPCReqInput(
            zmq_handles=zmq_handles,
            flush_cache=flush_cache,
        )
        return self.loop.run_until_complete(
            self.tokenizer_manager.update_weights_from_ipc(obj, None)  # 同步等待 IPC 权重更新完成
        )

    # -------- get_weights_by_name：按参数名获取权重 --------
    def get_weights_by_name(self, name: str, truncate_size: int = 100):
        """Get weights by parameter name."""
        # 按参数名获取模型权重（主要用于调试，默认截断到前 100 个元素）
        obj = GetWeightsByNameReqInput(name=name, truncate_size=truncate_size)
        return self.loop.run_until_complete(
            self.tokenizer_manager.get_weights_by_name(obj, None)  # 同步等待权重查询结果
        )

    # -------- load_lora_adapter_from_tensors：从张量加载 LoRA 适配器 --------
    def load_lora_adapter_from_tensors(
        self,
        lora_name: str,              # LoRA 适配器名称（用于标识/后续卸载）
        tensors,                     # LoRA 张量数据（格式取决于 load_format）
        config_dict: Dict,           # LoRA 配置字典（rank、alpha 等超参数）
        load_format: Optional[str] = None,  # 加载格式（None 或 'flattened_bucket'）
    ):
        if load_format == "flattened_bucket":
            # flattened_bucket 格式：张量已序列化，直接使用
            serialized_tensors = tensors
        else:
            # 其他格式：将张量序列化为字符串（跨进程传输）
            serialized_tensors = MultiprocessingSerializer.serialize(
                tensors, output_str=True
            )
        lora_req = LoadLoRAAdapterFromTensorsReqInput(
            lora_name=lora_name,
            config_dict=config_dict,
            serialized_tensors=serialized_tensors,
            load_format=load_format,
        )
        return self.loop.run_until_complete(
            self.tokenizer_manager.load_lora_adapter_from_tensors(lora_req, None)  # 同步等待 LoRA 加载完成
        )

    # -------- load_lora_adapter：从路径加载 LoRA 适配器 --------
    def load_lora_adapter(self, lora_name: str, lora_path: str, pinned: bool = False):
        """Load a new LoRA adapter without re-launching the engine."""
        # 不重启引擎，从磁盘路径动态加载 LoRA 适配器
        obj = LoadLoRAAdapterReqInput(
            lora_name=lora_name,   # LoRA 适配器名称
            lora_path=lora_path,   # LoRA 权重磁盘路径
            pinned=pinned,         # 是否固定到内存（防止被 LRU 驱逐）
        )

        return self.loop.run_until_complete(
            self.tokenizer_manager.load_lora_adapter(obj, None)  # 同步等待 LoRA 加载完成
        )

    # -------- unload_lora_adapter：卸载 LoRA 适配器 --------
    def unload_lora_adapter(self, lora_name: str):
        """Unload a LoRA adapter without re-launching the engine."""
        # 不重启引擎，动态卸载指定名称的 LoRA 适配器并释放内存
        obj = UnloadLoRAAdapterReqInput(lora_name=lora_name)  # 构造卸载请求

        return self.loop.run_until_complete(
            self.tokenizer_manager.unload_lora_adapter(obj, None)  # 同步等待卸载完成
        )

    # -------- async_load_lora_adapter：异步加载 LoRA 适配器 --------
    async def async_load_lora_adapter(
        self, lora_name: str, lora_path: str, pinned: bool = False
    ):
        """
        Asynchronous version of load_lora_adapter.

        See load_lora_adapter() for detailed documentation.
        """
        # load_lora_adapter 的异步版本，参数含义相同
        obj = LoadLoRAAdapterReqInput(
            lora_name=lora_name,
            lora_path=lora_path,
            pinned=pinned,
        )

        return await self.tokenizer_manager.load_lora_adapter(obj, None)  # 异步等待 LoRA 加载完成

    # -------- async_unload_lora_adapter：异步卸载 LoRA 适配器 --------
    async def async_unload_lora_adapter(self, lora_name: str):
        """
        Asynchronous version of unload_lora_adapter.

        See unload_lora_adapter() for detailed documentation.
        """
        # unload_lora_adapter 的异步版本
        obj = UnloadLoRAAdapterReqInput(lora_name=lora_name)  # 构造卸载请求

        return await self.tokenizer_manager.unload_lora_adapter(obj, None)  # 异步等待卸载完成

    # -------- release_memory_occupation：释放内存占用（如 GPU 显存） --------
    def release_memory_occupation(self, tags: Optional[List[str]] = None):
        obj = ReleaseMemoryOccupationReqInput(tags=tags)  # 构造释放内存请求（可按标签过滤）
        return self.loop.run_until_complete(
            self.tokenizer_manager.release_memory_occupation(obj, None)  # 同步等待内存释放
        )

    # -------- resume_memory_occupation：恢复内存占用 --------
    def resume_memory_occupation(self, tags: Optional[List[str]] = None):
        obj = ResumeMemoryOccupationReqInput(tags=tags)  # 构造恢复内存请求
        return self.loop.run_until_complete(
            self.tokenizer_manager.resume_memory_occupation(obj, None)  # 同步等待内存恢复
        )

    # -------- freeze_gc：冻结 Python GC，降低长尾延迟 --------
    def freeze_gc(self):
        """
        To maintain a high performance server with low latency, we want to reduce the
        stalls caused by the garbage collector scanning through a large number of objects.

        It is usually helpful to start the server and warm it up with real requests to
        initialize many of the long-lived objects that do not need to be garbage collected.

        After sufficient warmup, we can call this function to freeze the garbage collector
        so that all objects created before this point are considered out of scope for garbage
        collection.
        """
        # 服务器充分预热后调用此方法，冻结 GC 扫描范围，将预热期间创建的对象标记为无需回收，
        # 从而减少 GC 扫描大量对象导致的延迟抖动
        self.loop.run_until_complete(self.tokenizer_manager.freeze_gc())  # 同步等待 GC 冻结完成

    """
    Execute an RPC call on all scheduler processes.
    """
    # 向所有调度器进程广播 RPC 调用

    # -------- collective_rpc：向所有调度器进程发送 RPC 调用 --------
    def collective_rpc(self, method: str, **kwargs):
        obj = RpcReqInput(method=method, parameters=kwargs)  # 构造 RPC 请求（方法名 + 参数）
        self.send_to_rpc.send_pyobj(obj)                     # 通过 ZMQ DEALER 套接字发送请求
        recv_req = self.send_to_rpc.recv_pyobj(zmq.BLOCKY)   # 阻塞等待 RPC 响应
        assert isinstance(recv_req, RpcReqOutput)            # 断言响应类型正确
        assert recv_req.success, recv_req.message            # 断言 RPC 调用成功，否则抛出错误信息

    # -------- save_remote_model：通过 RPC 保存远程模型 --------
    def save_remote_model(self, **kwargs):
        self.collective_rpc("save_remote_model", **kwargs)  # 广播 save_remote_model RPC 到所有调度器

    # -------- save_sharded_model：通过 RPC 保存分片模型 --------
    def save_sharded_model(self, **kwargs):
        self.collective_rpc("save_sharded_model", **kwargs)  # 广播 save_sharded_model RPC 到所有调度器

    # score() 和 async_score() 方法由 EngineScoreMixin 混入类提供


# -------- _set_envs_and_config：设置全局环境变量与配置 --------
def _set_envs_and_config(server_args: ServerArgs):
    # -------- NCCL 环境变量配置 --------
    if "NCCL_CUMEM_ENABLE" not in os.environ or server_args.enable_symm_mem:
        # 仅在未手动设置时才写入；若启用对称内存则强制开启 CUMEM
        os.environ["NCCL_CUMEM_ENABLE"] = str(int(server_args.enable_symm_mem))
    if (
        "NCCL_NVLS_ENABLE" not in os.environ
        or server_args.enable_nccl_nvls
        or server_args.enable_symm_mem
    ):
        # 仅在未手动设置时才写入；启用 NVLS 或对称内存时强制开启
        os.environ["NCCL_NVLS_ENABLE"] = str(
            int(server_args.enable_nccl_nvls or server_args.enable_symm_mem)
        )
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "8"    # 限制每个 CUDA 设备的最大并发连接数，提升 NCCL 通信效率
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"          # 使用 CUDA 模块按需加载，减少启动时间

    if os.environ.get("TRTLLM_ENABLE_PDL", "1") != "0":
        # 启用 TensorRT-LLM PDL 内核（MoE、量化等各类算子均依赖此变量）
        os.environ["TRTLLM_ENABLE_PDL"] = "1"

    if os.environ.get("CUTE_DSL_LOG_LEVEL") is None:
        # 未手动设置时默认使用 WARNING 级别日志（30），避免产生过多日志
        os.environ["CUTE_DSL_LOG_LEVEL"] = "30"

    if os.environ.get("CUTE_DSL_LOG_TO_CONSOLE") is None:
        # 必须同时启用控制台日志，否则日志级别设置不生效
        os.environ["CUTE_DSL_LOG_TO_CONSOLE"] = "1"

    # 生成唯一运行 ID（时间戳 + 随机数），用于区分不同次引擎启动
    os.environ["SGLANG_RUN_ID"] = (
        f"sglang-run-{time.time()}-{random.randint(0, 100000000)}"
    )

    # -------- Prometheus 指标配置 --------
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()  # 设置 Prometheus 多进程模式工作目录

    # -------- 系统资源限制 --------
    set_ulimit()  # 提升文件描述符上限，防止高并发时报 "Too many open files"

    # -------- 依赖版本检查 --------
    if not get_bool_env_var("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK"):
        if server_args.attention_backend == "flashinfer":
            # 使用 flashinfer 后端时检查版本兼容性
            assert_pkg_version(
                "flashinfer_python",
                "0.6.8.post1",  # 最低要求版本
                "Please uninstall the old version and "
                "reinstall the latest version by following the instructions "
                "at https://docs.flashinfer.ai/installation.html.",
            )
        if _is_cuda:
            # CUDA 环境下检查 sglang-kernel 版本
            assert_pkg_version(
                "sglang-kernel",
                "0.4.1.post1",  # 最低要求版本
                "Please reinstall the latest version with `pip install sglang-kernel --force-reinstall`",
            )

    # -------- 信号处理器注册（仅限主线程） --------
    if threading.current_thread() is threading.main_thread():
        if server_args.custom_sigquit_handler is None:
            # 注册默认 SIGQUIT 处理器：
            # 子进程发生错误时会向主进程发送 SIGQUIT，
            # 主进程收到后清理整个进程树
            def launch_phase_sigquit_handler(signum, frame):
                logger.error(
                    "Received sigquit from a child process. It usually means the child failed."
                )
                kill_process_tree(os.getpid())  # 杀死整个进程树（含自身）

            signal.signal(signal.SIGQUIT, launch_phase_sigquit_handler)  # 注册 SIGQUIT 处理器
        else:
            # 用户提供了自定义 SIGQUIT 处理器（如崩溃转储），直接使用
            logger.error(
                f"Using custom SIGQUIT handler: {server_args.custom_sigquit_handler}"
            )
            signal.signal(signal.SIGQUIT, server_args.custom_sigquit_handler)  # 注册自定义处理器
    else:
        # 非主线程中无法注册信号处理器，记录警告
        logger.warning(
            "Signal handler is not added because the engine is not in the "
            "main thread. This disables the SIGQUIT handler for cleaning up "
            "the process tree when a child process fails."
        )

    # 设置多进程启动方式为 spawn（安全、跨平台）
    mp.set_start_method("spawn", force=True)


# -------- _set_gc：配置 Python 垃圾回收阈值 --------
def _set_gc(server_args: ServerArgs):
    if gc_threshold := server_args.gc_threshold:  # 若用户指定了 GC 阈值
        import gc  # 仅在需要时导入 gc 模块

        gc.set_threshold(*gc_threshold)  # 按用户配置设置 GC 三代阈值


# -------- _scheduler_died_error：构造调度器进程死亡的详细错误信息 --------
def _scheduler_died_error(rank: int, proc) -> RuntimeError:
    """Build a descriptive error for a scheduler process that died during init."""
    # 构造包含 rank、退出码及 OOM 排查建议的 RuntimeError
    proc.join(timeout=10)  # 等待进程退出，最多 10 秒
    return RuntimeError(
        f"Rank {rank} scheduler died during initialization "
        f"(exit code: {proc.exitcode}). "
        f"If exit code is -9 (SIGKILL), a common cause is the OS OOM killer. "
        f"Run `dmesg -T | grep -i oom` to check."  # 提示用户通过 dmesg 检查 OOM
    )


# -------- _wait_for_scheduler_ready：等待所有调度器就绪并返回初始化信息 --------
def _wait_for_scheduler_ready(
    scheduler_pipe_readers: List,  # 每个调度器对应的管道读取端列表
    scheduler_procs: List,         # 调度器进程对象列表（用于存活检测）
) -> List[Dict]:
    """Wait for the model to finish loading and return scheduler infos.

    Uses poll() with timeout instead of blocking recv(), so that child process
    death (e.g. OOM SIGKILL) is detected promptly instead of hanging forever.
    """
    # 使用带超时的 poll() 替代阻塞 recv()，以便快速检测子进程死亡（如 OOM SIGKILL）
    scheduler_infos = []  # 收集所有调度器的就绪信息
    for i in range(len(scheduler_pipe_readers)):
        while True:
            if scheduler_pipe_readers[i].poll(timeout=5.0):  # 每 5 秒轮询一次管道
                try:
                    data = scheduler_pipe_readers[i].recv()  # 从管道读取就绪数据
                except EOFError:
                    # 管道提前关闭，说明调度器进程已死亡
                    raise _scheduler_died_error(i, scheduler_procs[i])
                if data["status"] != "ready":
                    # 收到非就绪状态，说明初始化失败
                    raise RuntimeError(
                        "Initialization failed. Please see the error messages above."
                    )
                scheduler_infos.append(data)  # 记录就绪信息
                break  # 该调度器已就绪，跳出内层循环

            # poll 超时：检查所有调度器进程是否还存活
            for j in range(len(scheduler_procs)):
                if not scheduler_procs[j].is_alive():
                    raise _scheduler_died_error(j, scheduler_procs[j])  # 进程已死亡，抛出错误

    return scheduler_infos  # 返回所有调度器的初始化信息列表


# -------- _calculate_rank_ranges：计算节点负责的 PP/TP rank 范围 --------
def _calculate_rank_ranges(
    nnodes: int,     # 总节点数
    pp_size: int,    # 流水线并行度
    tp_size: int,    # 张量并行度
    node_rank: int,  # 当前节点 rank
) -> Tuple[range, range, int, int]:
    """Calculate pp_rank_range and tp_rank_range for a given node.

    Args:
        nnodes: Total number of nodes.
        pp_size: Pipeline parallel size.
        tp_size: Tensor parallel size.
        node_rank: The rank of the node to compute ranges for.

    Returns:
        A tuple of (pp_rank_range, tp_rank_range, pp_size_per_node, tp_size_per_node):
        - pp_rank_range: range of pipeline-parallel ranks assigned to this node.
        - tp_rank_range: range of tensor-parallel ranks assigned to this node.
        - pp_size_per_node: number of PP ranks per node.
        - tp_size_per_node: number of TP ranks per node.
    """
    # 计算每个节点负责的 PP rank 数量（至少为 1，避免节点多于 PP rank 的情况）
    pp_size_per_node = max(pp_size // nnodes, 1)
    # 计算每个 PP rank 占用的节点数
    nnodes_per_pp_rank = max(nnodes // pp_size, 1)
    # 计算当前节点的 PP rank 范围
    pp_rank_range = range(
        pp_size_per_node * (node_rank // nnodes_per_pp_rank),       # 起始 PP rank
        pp_size_per_node * (node_rank // nnodes_per_pp_rank + 1),   # 结束 PP rank（不含）
    )

    # 每个 TP 组使用的节点数等于每个 PP rank 占用的节点数
    nnodes_per_tp_group = nnodes_per_pp_rank
    # 每个节点负责的 TP rank 数量
    tp_size_per_node = tp_size // nnodes_per_tp_group
    # 计算当前节点的 TP rank 范围（在 TP 组内取模以循环分配）
    tp_rank_range = range(
        tp_size_per_node * (node_rank % nnodes_per_tp_group),       # 起始 TP rank
        tp_size_per_node * (node_rank % nnodes_per_tp_group + 1),   # 结束 TP rank（不含）
    )

    return pp_rank_range, tp_rank_range, pp_size_per_node, tp_size_per_node


# -------- _compute_parallelism_ranks：根据 TP rank 推导其他并行维度的 rank --------
def _compute_parallelism_ranks(
    server_args: ServerArgs,  # 服务器参数（包含各并行度配置）
    tp_rank: int,             # 当前进程的全局张量并行 rank
) -> Tuple[int, int, int]:
    """Compute attention-CP, MoE-DP, and MoE-EP ranks for a TP rank."""
    # 计算注意力 CP rank、MoE DP rank 和 MoE EP rank
    attn_dp_size = server_args.dp_size if server_args.enable_dp_attention else 1  # 注意力 DP 大小

    # 并行层次结构（从最外层到最内层）：
    # - 注意力：Global(TP) -> DP -> ATTN_CP -> ATTN_TP（最内层）
    # - MoE：Global(TP) -> MOE_DP -> EP -> MOE_TP（最内层）
    attn_tp_size = server_args.tp_size // attn_dp_size // server_args.attn_cp_size  # 注意力 TP 组大小
    attn_cp_rank = (tp_rank // attn_tp_size) % server_args.attn_cp_size             # 注意力 CP rank（在 CP 组内循环）
    moe_dp_rank = tp_rank // (server_args.tp_size // server_args.moe_dp_size)       # MoE DP rank（按 DP 分组）
    moe_ep_rank = (
        tp_rank
        % (server_args.tp_size // server_args.moe_dp_size)                         # 在 DP 组内的局部 rank
        // (server_args.tp_size // server_args.moe_dp_size // server_args.ep_size)  # 除以 EP 组大小得到 EP rank
    )
    return attn_cp_rank, moe_dp_rank, moe_ep_rank  # 返回三个并行维度的 rank
