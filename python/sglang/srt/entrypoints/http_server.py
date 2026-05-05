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

This file implements HTTP APIs for the inference engine via fastapi.
"""
# -------- 标准库 import --------
import asyncio          # 异步事件循环 — 用于协程调度
import dataclasses      # 数据类装饰器 — 用于定义轻量结构体
import logging          # 日志模块 — 记录服务器运行信息
import os               # 操作系统接口 — 读取环境变量、进程操作
import tempfile         # 临时文件 — 在多 tokenizer 模式下生成 IPC 地址
import threading        # 多线程 — 用于后台 warmup 线程
import time             # 时间工具 — 用于超时判断和时间戳
from contextlib import asynccontextmanager  # 异步上下文管理器 — 用于 FastAPI lifespan
from http import HTTPStatus                 # HTTP 状态码枚举 — 构造错误响应
from typing import (
    Any,            # 任意类型
    AsyncGenerator, # 异步生成器类型
    AsyncIterator,  # 异步迭代器类型
    Callable,       # 可调用对象类型
    Dict,           # 字典类型
    List,           # 列表类型
    Optional,       # 可选类型（可为 None）
    Union,          # 联合类型
)

# -------- 修复 Python threading 的一个已知 bug --------
# 将 _register_atexit 替换为空函数，避免多进程场景下 atexit 钩子引发的问题
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

# -------- 第三方库 import --------
import numpy as np      # 数值计算库 — 用于构造 warmup 输入数据
import requests         # HTTP 客户端 — 用于服务器内部 warmup 请求
import uvicorn          # ASGI 服务器 — 运行 FastAPI 应用
import uvloop           # 高性能事件循环 — 替换默认 asyncio 事件循环
from fastapi import (
    Depends,        # 依赖注入 — 用于请求前置校验
    FastAPI,        # Web 框架核心类
    File,           # 文件上传参数声明
    Form,           # 表单参数声明
    HTTPException,  # HTTP 异常类
    Query,          # 查询参数声明
    Request,        # 原始请求对象
    UploadFile,     # 上传文件对象
)
from fastapi.exceptions import RequestValidationError   # 请求体校验异常 — 用于自定义 400 响应
from fastapi.middleware.cors import CORSMiddleware       # CORS 中间件 — 允许跨域请求
from fastapi.responses import ORJSONResponse, Response, StreamingResponse  # 响应类型 — 普通/流式响应

# -------- sglang 内部模块 import --------
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX  # 健康检查请求 ID 前缀常量
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST, DisaggregationMode  # PD 分离模式相关工具
from sglang.srt.entrypoints.anthropic.protocol import (
    AnthropicCountTokensRequest,    # Anthropic Token 计数请求协议
    AnthropicMessagesRequest,       # Anthropic Messages API 请求协议
)
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing  # Anthropic 兼容服务处理器
from sglang.srt.entrypoints.engine import (
    Engine,                     # 推理引擎主类
    init_tokenizer_manager,     # 初始化 Tokenizer 管理器函数
    run_detokenizer_process,    # 启动 Detokenizer 子进程函数
    run_scheduler_process,      # 启动 Scheduler 子进程函数
)
from sglang.srt.entrypoints.ollama.protocol import (
    OllamaChatRequest,      # Ollama 聊天请求协议
    OllamaGenerateRequest,  # Ollama 生成请求协议
    OllamaShowRequest,      # Ollama 模型信息请求协议
)
from sglang.srt.entrypoints.ollama.serving import OllamaServing  # Ollama 兼容服务处理器
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,  # OpenAI 聊天补全请求
    ClassifyRequest,        # 分类请求
    CompletionRequest,      # 文本补全请求
    DetokenizeRequest,      # 反 tokenize 请求
    EmbeddingRequest,       # Embedding 请求
    ErrorResponse,          # 错误响应结构
    ModelCard,              # 模型信息卡片
    ModelList,              # 模型列表
    ResponsesRequest,       # Responses API 请求
    ScoringRequest,         # 评分请求
    TokenizeRequest,        # Tokenize 请求
    V1RerankReqInput,       # Rerank 请求
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat            # OpenAI 聊天服务处理器
from sglang.srt.entrypoints.openai.serving_classify import OpenAIServingClassify    # OpenAI 分类服务处理器
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion  # OpenAI 补全服务处理器
from sglang.srt.entrypoints.openai.serving_embedding import OpenAIServingEmbedding  # OpenAI Embedding 服务处理器
from sglang.srt.entrypoints.openai.serving_rerank import OpenAIServingRerank        # OpenAI Rerank 服务处理器
from sglang.srt.entrypoints.openai.serving_score import OpenAIServingScore          # OpenAI 评分服务处理器
from sglang.srt.entrypoints.openai.serving_tokenize import (
    OpenAIServingDetokenize,  # OpenAI 反 tokenize 服务处理器
    OpenAIServingTokenize,    # OpenAI tokenize 服务处理器
)
from sglang.srt.entrypoints.openai.serving_transcription import (
    OpenAIServingTranscription,  # OpenAI 语音转录服务处理器
)
from sglang.srt.entrypoints.warmup import execute_warmups  # 执行自定义 warmup 逻辑
from sglang.srt.environ import envs                        # 环境变量统一管理对象
from sglang.srt.function_call.function_call_parser import FunctionCallParser  # 函数调用解析器
from sglang.srt.managers.io_struct import (
    AbortReq,                                       # 中止请求结构
    AttachHiCacheStorageReqInput,                   # 附加 HiCache 存储后端请求
    CheckWeightsReqInput,                           # 权重校验请求
    CloseSessionReqInput,                           # 关闭会话请求
    ConfigureLoggingReq,                            # 日志配置请求
    ContinueGenerationReqInput,                     # 继续生成请求
    DestroyWeightsUpdateGroupReqInput,              # 销毁权重更新组请求
    DumperControlReqInput,                          # Dumper 控制请求
    EmbeddingReqInput,                              # Embedding 推理请求
    GenerateReqInput,                               # 文本生成推理请求
    GetWeightsByNameReqInput,                       # 按名称获取权重请求
    InitWeightsSendGroupForRemoteInstanceReqInput,  # 初始化远程实例权重发送组请求
    InitWeightsUpdateGroupReqInput,                 # 初始化权重更新组请求
    LoadLoRAAdapterFromTensorsReqInput,             # 从张量加载 LoRA 适配器请求
    LoadLoRAAdapterReqInput,                        # 加载 LoRA 适配器请求
    OpenSessionReqInput,                            # 打开会话请求
    ParseFunctionCallReq,                           # 解析函数调用请求
    PauseGenerationReqInput,                        # 暂停生成请求
    ProfileReqInput,                                # 性能分析请求
    ReleaseMemoryOccupationReqInput,                # 释放 GPU 内存占用请求
    ResumeMemoryOccupationReqInput,                 # 恢复 GPU 内存占用请求
    SendWeightsToRemoteInstanceReqInput,            # 发送权重到远程实例请求
    SeparateReasoningReqInput,                      # 分离推理文本请求
    SetInternalStateReq,                            # 设置内部状态请求
    SlowDownReqInput,                               # 人为降速请求（测试用）
    UnloadLoRAAdapterReqInput,                      # 卸载 LoRA 适配器请求
    UpdateWeightFromDiskReqInput,                   # 从磁盘更新权重请求
    UpdateWeightsFromDistributedReqInput,           # 从分布式更新权重请求
    UpdateWeightsFromIPCReqInput,                   # 通过 IPC 更新权重请求
    UpdateWeightsFromTensorReqInput,                # 从张量更新权重请求
    UpdateWeightVersionReqInput,                    # 更新权重版本请求
    VertexGenerateReqInput,                         # Vertex AI 生成请求
)
from sglang.srt.managers.multi_tokenizer_mixin import (
    MultiTokenizerRouter,               # 多 tokenizer 路由器
    TokenizerWorker,                    # tokenizer 工作进程类
    get_main_process_id,                # 获取主进程 PID
    monkey_patch_uvicorn_multiprocessing,  # 补丁 uvicorn 以支持多进程
    read_from_shared_memory,            # 从共享内存读取配置
    write_data_for_multi_tokenizer,     # 将配置写入共享内存
)
from sglang.srt.managers.template_manager import TemplateManager          # 聊天模板管理器
from sglang.srt.managers.tokenizer_manager import ServerStatus, TokenizerManager  # Tokenizer 管理器及服务器状态枚举
from sglang.srt.observability.func_timer import enable_func_timer          # 启用函数计时（可观测性）
from sglang.srt.observability.trace import (
    process_tracing_init,       # 初始化进程级链路追踪
    set_global_trace_level,     # 设置全局追踪级别
    trace_set_thread_info,      # 设置线程追踪信息
)
from sglang.srt.parser.reasoning_parser import ReasoningParser  # 推理文本解析器
from sglang.srt.server_args import PortArgs, ServerArgs         # 端口参数与服务器参数数据类
from sglang.srt.utils import (
    add_prometheus_middleware,                  # 添加 Prometheus 监控中间件
    add_prometheus_track_response_middleware,   # 添加 Prometheus 响应跟踪中间件
    delete_directory,                           # 删除目录工具
    get_bool_env_var,                           # 读取布尔类型环境变量
    kill_process_tree,                          # 杀死进程树
    set_uvicorn_logging_configs,                # 配置 uvicorn 日志格式
)
from sglang.srt.utils.auth import AuthLevel, app_has_admin_force_endpoints, auth_level  # 鉴权级别与装饰器
from sglang.srt.utils.json_response import (
    SGLangORJSONResponse,   # SGLang 自定义 ORJSON 响应类
    dumps_json,             # 序列化为 JSON bytes
    orjson_response,        # 构造 ORJSON 响应对象
)
from sglang.srt.utils.watchdog import SubprocessWatchdog  # 子进程守护/监控工具
from sglang.utils import get_exception_traceback           # 获取异常堆栈字符串
from sglang.version import __version__                     # SGLang 版本号

logger = logging.getLogger(__name__)            # 获取当前模块的日志记录器
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())  # 将事件循环策略替换为 uvloop 以提升性能

# -------- 全局常量 --------
# 健康检查超时秒数，默认 20 秒，可通过环境变量覆盖
HEALTH_CHECK_TIMEOUT = int(os.getenv("SGLANG_HEALTH_CHECK_TIMEOUT", 20))
# 等待权重就绪的超时秒数，默认 120 秒
WAIT_WEIGHTS_READY_TIMEOUT = int(os.getenv("SGLANG_WAIT_WEIGHTS_READY_TIMEOUT", 120))


# -------- 全局状态数据类 --------
# 存储 HTTP 服务运行期间的全局单例状态
@dataclasses.dataclass
class _GlobalState:
    tokenizer_manager: Union[TokenizerManager, MultiTokenizerRouter, TokenizerWorker]  # tokenizer 管理器（支持多种模式）
    template_manager: TemplateManager   # 聊天模板管理器
    scheduler_info: Dict                # scheduler 初始化信息字典


_global_state: Optional[_GlobalState] = None  # 全局状态单例，初始为 None


# 设置全局状态单例
def set_global_state(global_state: _GlobalState):
    global _global_state          # 声明使用全局变量
    _global_state = global_state  # 将传入的状态对象赋值给全局变量


# 获取全局状态单例
def get_global_state() -> _GlobalState:
    return _global_state  # 返回当前全局状态对象


# -------- Granian 工作进程初始化（HTTP/2 模式） --------
# 从共享内存读取配置并初始化当前 Granian worker 进程的 tokenizer 管理器
async def _init_granian_worker() -> ServerArgs:
    main_pid = get_main_process_id()  # 获取主进程 PID，用于定位共享内存 key
    port_args, server_args, scheduler_info = read_from_shared_memory(
        f"multi_tokenizer_args_{main_pid}"  # 从以主进程 PID 命名的共享内存块读取初始化参数
    )

    tokenizer_manager = TokenizerManager(server_args, port_args)  # 创建 tokenizer 管理器实例
    template_manager = TemplateManager()                           # 创建聊天模板管理器
    template_manager.initialize_templates(                         # 加载模型对应的聊天/补全模板
        tokenizer_manager=tokenizer_manager,
        model_path=server_args.model_path,
        chat_template=server_args.chat_template,
        completion_template=server_args.completion_template,
    )
    tokenizer_manager.max_req_input_len = scheduler_info["max_req_input_len"]  # 从 scheduler 信息中获取最大输入长度

    set_global_state(                  # 将初始化结果写入全局状态
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            template_manager=template_manager,
            scheduler_info=scheduler_info,
        )
    )
    return server_args  # 返回服务器参数供后续使用


# -------- 多 tokenizer 工作进程初始化 --------
async def init_multi_tokenizer() -> ServerArgs:
    """
    Initialization function for multi-process tokenizer mode.
    It read args information from shm and inits tokenizer manager for current process.
    """
    # 从共享内存读取主进程写入的初始化参数
    main_pid = get_main_process_id()  # 获取主进程 PID
    port_args, server_args, scheduler_info = read_from_shared_memory(
        f"multi_tokenizer_args_{main_pid}"  # 以主进程 PID 命名的共享内存 key
    )
    server_args: ServerArgs   # 服务器参数类型注解
    port_args: PortArgs       # 端口参数类型注解

    # 多 tokenizer 模式不支持 API 密钥鉴权，断言确保未配置
    assert (
        server_args.api_key is None
    ), "API key is not supported in multi-tokenizer mode"

    # 为当前 worker 进程生成独立的 IPC（进程间通信）地址
    port_args.tokenizer_ipc_name = (
        f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"  # 基于临时文件路径生成唯一 IPC 地址
    )
    logger.info(
        f"Start multi-tokenizer worker process {os.getpid()}, "
        f"ipc_name={port_args.tokenizer_ipc_name}"  # 记录当前进程 PID 和 IPC 地址
    )

    # 创建 TokenizerWorker（多进程模式下的 tokenizer 工作单元）
    tokenizer_manager = TokenizerWorker(server_args, port_args)
    template_manager = TemplateManager()          # 创建聊天模板管理器
    template_manager.initialize_templates(        # 加载聊天/补全模板
        tokenizer_manager=tokenizer_manager,
        model_path=server_args.model_path,
        chat_template=server_args.chat_template,
        completion_template=server_args.completion_template,
    )

    tokenizer_manager.max_req_input_len = scheduler_info["max_req_input_len"]  # 设置最大输入长度

    set_global_state(          # 将初始化结果写入全局状态
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            template_manager=template_manager,
            scheduler_info=scheduler_info,
        )
    )

    return server_args  # 返回服务器参数


# -------- FastAPI 应用生命周期管理 --------
# lifespan 在 FastAPI 启动/关闭时执行，负责各服务处理器的初始化
@asynccontextmanager
async def lifespan(fast_api_app: FastAPI):
    # 根据运行模式选择初始化路径：单 tokenizer / Granian worker / 多 tokenizer worker
    if getattr(fast_api_app, "is_single_tokenizer_mode", False):
        # 单 tokenizer 模式：直接从 app 属性读取已初始化的参数
        server_args = fast_api_app.server_args
        warmup_thread_kwargs = fast_api_app.warmup_thread_kwargs
        thread_label = "Tokenizer"
    elif envs.SGLANG_GRANIAN_PARENT_PID.get() is not None:
        # Granian HTTP/2 模式：从共享内存初始化 worker
        server_args = await _init_granian_worker()
        warmup_thread_kwargs = dict(server_args=server_args)
        thread_label = "Tokenizer"
    else:
        # 多 tokenizer uvicorn worker 模式：从共享内存读取参数并初始化
        server_args = await init_multi_tokenizer()
        warmup_thread_kwargs = dict(server_args=server_args)
        thread_label = f"MultiTokenizer-{_global_state.tokenizer_manager.worker_id}"  # 标签包含 worker id

    # 若启用了 Prometheus 指标，添加中间件并开启函数计时
    if server_args.enable_metrics:
        add_prometheus_middleware(app)  # 注册 Prometheus HTTP 指标采集中间件
        enable_func_timer()             # 开启关键函数的耗时统计

    # 若启用了链路追踪，初始化 OpenTelemetry 并设置线程标签
    if server_args.enable_trace:
        process_tracing_init(server_args.otlp_traces_endpoint, "sglang")  # 初始化 OTLP 链路追踪
        if server_args.disaggregation_mode == "prefill":
            thread_label = "Prefill" + thread_label   # PD 分离 prefill 模式添加前缀
        elif server_args.disaggregation_mode == "decode":
            thread_label = "Decode" + thread_label    # PD 分离 decode 模式添加前缀
        trace_set_thread_info(thread_label)            # 将当前线程标签写入追踪上下文

    # -------- 初始化各 OpenAI 兼容服务处理器 --------
    fast_api_app.state.openai_serving_completion = OpenAIServingCompletion(
        _global_state.tokenizer_manager, _global_state.template_manager  # 文本补全服务
    )
    fast_api_app.state.openai_serving_chat = OpenAIServingChat(
        _global_state.tokenizer_manager, _global_state.template_manager  # 聊天补全服务
    )
    fast_api_app.state.openai_serving_embedding = OpenAIServingEmbedding(
        _global_state.tokenizer_manager, _global_state.template_manager  # Embedding 服务
    )
    fast_api_app.state.openai_serving_classify = OpenAIServingClassify(
        _global_state.tokenizer_manager, _global_state.template_manager  # 分类服务
    )
    fast_api_app.state.openai_serving_score = OpenAIServingScore(
        _global_state.tokenizer_manager  # 评分服务（无需模板管理器）
    )
    fast_api_app.state.openai_serving_rerank = OpenAIServingRerank(
        _global_state.tokenizer_manager, _global_state.template_manager  # Rerank 服务
    )
    fast_api_app.state.openai_serving_tokenize = OpenAIServingTokenize(
        _global_state.tokenizer_manager  # tokenize 服务
    )
    fast_api_app.state.openai_serving_detokenize = OpenAIServingDetokenize(
        _global_state.tokenizer_manager  # 反 tokenize 服务
    )
    fast_api_app.state.openai_serving_transcription = OpenAIServingTranscription(
        _global_state.tokenizer_manager  # 语音转录服务
    )

    # 初始化 Ollama 兼容服务处理器
    fast_api_app.state.ollama_serving = OllamaServing(_global_state.tokenizer_manager)

    # 初始化 Anthropic 兼容服务处理器（复用 OpenAI Chat serving）
    fast_api_app.state.anthropic_serving = AnthropicServing(
        fast_api_app.state.openai_serving_chat  # 基于 OpenAI Chat 服务构建 Anthropic 适配层
    )

    # -------- 启动工具服务器（可选） --------
    tool_server = None
    if server_args.tool_server == "demo":
        from sglang.srt.entrypoints.openai.tool_server import DemoToolServer

        tool_server = DemoToolServer()  # 使用演示用工具服务器
    elif server_args.tool_server:
        from sglang.srt.entrypoints.openai.tool_server import MCPToolServer

        tool_server = MCPToolServer()                               # 使用 MCP 协议工具服务器
        await tool_server.add_tool_server(server_args.tool_server)  # 注册配置的工具服务地址

    # 初始化 OpenAI Responses API 服务处理器（较新，可能不可用故捕获异常）
    try:
        from sglang.srt.entrypoints.openai.serving_responses import (
            OpenAIServingResponses,
        )

        fast_api_app.state.openai_serving_responses = OpenAIServingResponses(
            _global_state.tokenizer_manager,
            _global_state.template_manager,
            enable_prompt_tokens_details=True,  # 启用 prompt token 详情返回
            tool_server=tool_server,            # 绑定工具服务器
        )
    except Exception:
        traceback = get_exception_traceback()   # 获取异常堆栈
        logger.warning(f"Can not initialize OpenAIServingResponses, error: {traceback}")

    # 执行用户配置的自定义 warmup 列表（逗号分隔）
    if server_args.warmups is not None:
        await execute_warmups(
            server_args.disaggregation_mode,         # 传入当前分离模式
            server_args.warmups.split(","),           # 将逗号分隔的 warmup 名拆分为列表
            _global_state.tokenizer_manager,
        )
        logger.info("Warmup ended")  # 自定义 warmup 完成日志

    # 在后台线程中执行通用 warmup（发送测试请求确认服务可用）
    warmup_thread = threading.Thread(
        target=_wait_and_warmup,         # warmup 目标函数
        kwargs=warmup_thread_kwargs,     # 传入服务器参数等关键字参数
    )
    warmup_thread.start()  # 启动后台 warmup 线程

    # HTTP 服务器运行期间 —— yield 之后 FastAPI 接受请求
    try:
        yield  # 挂起此处，等待 FastAPI 应用正常运行直至关闭
    finally:
        warmup_thread.join()  # 服务器关闭时等待 warmup 线程结束


# -------- FastAPI 应用实例 --------
# 创建 FastAPI 主应用，绑定 lifespan 钩子；根据环境变量决定是否暴露 OpenAPI 文档
app = FastAPI(
    lifespan=lifespan,
    openapi_url=None if get_bool_env_var("DISABLE_OPENAPI_DOC") else "/openapi.json",  # 可通过环境变量禁用 API 文档
)
# 添加跨域中间件，允许所有来源、方法和请求头
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 允许所有域名跨域访问
    allow_credentials=True,     # 允许携带凭证（如 Cookie）
    allow_methods=["*"],        # 允许所有 HTTP 方法
    allow_headers=["*"],        # 允许所有请求头
)

# -------- 挂载子路由 --------
from sglang.srt.entrypoints.v1_loads import router as v1_loads_router  # v1/loads 路由模块

app.include_router(v1_loads_router)  # 将 loads 路由注册到主应用


# -------- 全局异常处理器：HTTPException --------
# 将 FastAPI 抛出的 HTTPException 统一格式化为 OpenAI 风格的错误响应
@app.exception_handler(HTTPException)
async def validation_exception_handler(request: Request, exc: HTTPException):
    """Enrich HTTP exception with status code and other details.

    For /v1/responses, emit OpenAI-style nested error envelope:
    {"error": {"message": "...", "type": "...", "param": null, "code": <status>}}
    """
    # /v1/responses 端点使用嵌套 error 结构（与其他端点略有不同）
    if request.url.path.startswith("/v1/responses"):
        nested_error = {
            "message": exc.detail,                          # HTTP 异常详细信息
            "type": HTTPStatus(exc.status_code).phrase,     # 状态码对应短语
            "param": None,                                   # 无特定参数
            "code": exc.status_code,                         # 状态码整数值
        }
        return ORJSONResponse(
            content={"error": nested_error}, status_code=exc.status_code
        )

    # 其他端点使用 OpenAI ErrorResponse 结构
    error = ErrorResponse(
        object="error",
        message=exc.detail,               # 错误详细信息
        type=str(exc.status_code),        # 类型字段使用状态码字符串
        code=exc.status_code,             # 数字状态码
    )
    return ORJSONResponse(content=error.model_dump(), status_code=exc.status_code)


# -------- 全局异常处理器：请求体校验错误 --------
# 将 FastAPI 默认 422 校验错误覆盖为 400，并统一格式化
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Override FastAPI's default 422 validation error with 400.

    For /v1/responses, emit OpenAI-style nested error envelope; for other endpoints keep legacy format.
    """
    exc_str = str(exc)          # 异常字符串表示
    errors_str = str(exc.errors())  # Pydantic 校验错误详情

    # 若两者不同则合并，提供更完整的错误信息
    if errors_str and errors_str != exc_str:
        message = f"{exc_str} {errors_str}"
    else:
        message = exc_str

    if request.url.path.startswith("/v1/responses"):
        # /v1/responses 专用嵌套 error 格式（注意字段名与其他端点不同）
        nested_error = {
            "message": message,
            "type": HTTPStatus.BAD_REQUEST.phrase,   # "Bad Request"
            "param": None,
            "code": HTTPStatus.BAD_REQUEST.value,    # 400
        }
        return ORJSONResponse(status_code=400, content={"error": nested_error})

    # 其他端点使用标准 ErrorResponse 格式
    err = ErrorResponse(
        message=message,
        type=HTTPStatus.BAD_REQUEST.phrase,  # "Bad Request"
        code=HTTPStatus.BAD_REQUEST.value,   # 400
    )

    return ORJSONResponse(
        status_code=400,
        content=err.model_dump(),
    )


# -------- 请求 Content-Type 校验依赖 --------
# 用作 Depends()，确保请求头中 Content-Type 为 application/json
async def validate_json_request(raw_request: Request):
    """Validate that the request content-type is application/json."""
    content_type = raw_request.headers.get("content-type", "").lower()  # 读取请求头 content-type（转小写）
    media_type = content_type.split(";", maxsplit=1)[0]                  # 提取媒体类型部分（忽略参数如 charset）
    if media_type != "application/json":
        raise RequestValidationError(  # 非 JSON 则抛出校验异常，触发上方 400 处理器
            errors=[
                {
                    "loc": ["header", "content-type"],
                    "msg": "Unsupported Media Type: Only 'application/json' is allowed",
                    "type": "value_error",
                }
            ]
        )


##### 原生 API 端点 #####


# -------- 健康检查端点 --------
# 同时注册 /health 和 /health_generate 两个路由
@app.get("/health")
@app.get("/health_generate")
async def health_generate(request: Request) -> Response:
    """
    Check the health of the inference server by sending a special request to generate one token.

    If the server is running something, this request will be ignored, so it creates zero overhead.
    If the server is not running anything, this request will be run, so we know whether the server is healthy.
    """
    # 若服务正在优雅关闭，返回 503
    if _global_state.tokenizer_manager.gracefully_exit:
        logger.info("Health check request received during shutdown. Returning 503.")
        return Response(status_code=503)

    # 服务器仍在启动阶段，返回 503
    if _global_state.tokenizer_manager.server_status == ServerStatus.Starting:
        return Response(status_code=503)

    # /health 端点且未启用生成健康检查时，直接返回 200（零开销）
    if (
        not envs.SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION.get()
        and request.url.path == "/health"
    ):
        return Response(status_code=200)

    sampling_params = {"max_new_tokens": 1, "temperature": 0.0}  # 健康检查仅生成 1 个 token
    rid = f"{HEALTH_CHECK_RID_PREFIX}_{time.time()}"              # 生成唯一请求 ID

    # 根据模型类型（生成型 / 嵌入型）构造不同请求
    if _global_state.tokenizer_manager.is_generation:
        gri = GenerateReqInput(
            rid=rid,
            input_ids=[0],                  # 使用 token id 0 作为最小输入
            sampling_params=sampling_params,
            log_metrics=False,              # 健康检查不记录指标
        )
        if (
            _global_state.tokenizer_manager.server_args.disaggregation_mode
            != DisaggregationMode.NULL.value
        ):
            gri.bootstrap_host = FAKE_BOOTSTRAP_HOST  # PD 分离模式需要设置 bootstrap_host
            gri.bootstrap_room = 0
    else:
        gri = EmbeddingReqInput(
            rid=rid, input_ids=[0], sampling_params=sampling_params, log_metrics=False
        )

    # 异步消费生成结果（只取第一个响应即可）
    async def gen():
        async for _ in _global_state.tokenizer_manager.generate_request(gri, request):
            break

    task = asyncio.create_task(gen())  # 以异步 task 方式提交请求

    # 轮询等待 detokenizer 响应，超时前收到任意响应则认为服务健康
    tic = time.time()
    while time.time() < tic + HEALTH_CHECK_TIMEOUT:
        await asyncio.sleep(1)                                                     # 每秒检查一次
        if _global_state.tokenizer_manager.last_receive_tstamp > tic:             # 收到新响应
            task.cancel()                                                           # 取消健康检查 task
            _global_state.tokenizer_manager.rid_to_state.pop(rid, None)            # 清理请求状态
            _global_state.tokenizer_manager.server_status = ServerStatus.Up        # 标记服务正常
            return Response(status_code=200)

    # 超时未收到响应，标记服务不健康
    task.cancel()
    tic_time = time.strftime("%H:%M:%S", time.localtime(tic))
    last_receive_time = time.strftime(
        "%H:%M:%S", time.localtime(_global_state.tokenizer_manager.last_receive_tstamp)
    )
    logger.error(
        f"Health check failed. Server couldn't get a response from detokenizer for last "
        f"{HEALTH_CHECK_TIMEOUT} seconds. tic start time: {tic_time}. "
        f"last_heartbeat time: {last_receive_time}"
    )
    _global_state.tokenizer_manager.rid_to_state.pop(rid, None)                   # 清理请求状态
    _global_state.tokenizer_manager.server_status = ServerStatus.UnHealthy        # 标记服务不健康
    return Response(status_code=503)


# -------- 模型信息端点（废弃旧路径） --------
@app.get("/get_model_info")
async def get_model_info():
    """Get the model information (deprecated - use /model_info instead)."""
    logger.warning(
        "Endpoint '/get_model_info' is deprecated and will be removed in a future version. "
        "Please use '/model_info' instead."
    )
    return await model_info()  # 转发到新的 /model_info 端点


# -------- 模型信息端点 --------
@app.get("/model_info")
async def model_info():
    """Get the model information."""
    model_config = _global_state.tokenizer_manager.model_config  # 获取模型配置对象
    result = {
        "model_path": _global_state.tokenizer_manager.model_path,                               # 模型路径
        "tokenizer_path": _global_state.tokenizer_manager.server_args.tokenizer_path,           # tokenizer 路径
        "is_generation": _global_state.tokenizer_manager.is_generation,                         # 是否为生成型模型
        "preferred_sampling_params": _global_state.tokenizer_manager.server_args.preferred_sampling_params,  # 偏好采样参数
        "weight_version": _global_state.tokenizer_manager.server_args.weight_version,           # 权重版本号
        "has_image_understanding": model_config.is_image_understandable_model,                  # 是否支持图像理解
        "has_audio_understanding": model_config.is_audio_understandable_model,                  # 是否支持音频理解
        "model_type": getattr(model_config.hf_config, "model_type", None),                      # HF 模型类型字符串
        "architectures": getattr(model_config.hf_config, "architectures", None),                # 模型架构列表
        "weight_version": _global_state.tokenizer_manager.server_args.weight_version,           # 权重版本（重复字段，保留原始）
        # "hf_config": model_config.hf_config.to_dict(),  # 完整 HF 配置（已注释）
    }
    return result


# -------- 权重版本端点（已废弃） --------
@app.get("/get_weight_version")
@app.get("/weight_version")
async def weight_version():
    """Get the current weight version."""
    # 两个旧端点均已废弃，请使用 /model_info
    raise HTTPException(
        status_code=404,
        detail="Endpoint '/get_weight_version' or '/weight_version' is deprecated. Please use '/model_info' instead.",
    )


# -------- 服务器信息端点（废弃旧路径） --------
@app.get("/get_server_info")
async def get_server_info():
    """Get the server information (deprecated - use /server_info instead)."""
    logger.warning(
        "Endpoint '/get_server_info' is deprecated and will be removed in a future version. "
        "Please use '/server_info' instead."
    )
    return await server_info()  # 转发到新的 /server_info 端点


# -------- 服务器信息端点 --------
@app.get("/server_info")
async def server_info():
    """Get the server information."""
    # 获取每个数据并行（DP）分片的内部状态列表
    internal_states: List[Dict[Any, Any]] = (
        await _global_state.tokenizer_manager.get_internal_state()
    )

    # 合并 server_args（不含 model_config，因其不可序列化）、scheduler_info 及版本信息
    return {
        **dataclasses.asdict(_global_state.tokenizer_manager.server_args),  # 服务器参数字典
        **_global_state.scheduler_info,                                      # scheduler 初始化信息
        "internal_states": internal_states,                                  # 各 DP 分片内部状态
        "version": __version__,                                              # SGLang 版本号
    }


# -------- 负载查询端点（废弃旧路径） --------
# 兼容历史客户端，将旧 /get_load 数据结构映射到新 /v1/loads 结果
@app.get("/get_load")
async def get_load():
    """Get load metrics (deprecated - use /v1/loads instead).

    Legacy shim backed by /v1/loads. Projects GetLoadsReqOutput down to the
    historical field shape (dp_rank, num_reqs, num_waiting_reqs, num_tokens,
    num_pending_tokens, ts_tic) so existing clients keep working.
    """
    logger.warning(
        "Endpoint '/get_load' is deprecated and will be removed in a future version. "
        "Please use '/v1/loads' instead."
    )
    load_results = await _global_state.tokenizer_manager.get_loads(include=["core"])  # 仅获取 core 指标
    ts = time.perf_counter()  # 获取当前时间戳（高精度）
    return [
        {
            "dp_rank": r.dp_rank,                                              # 数据并行分片编号
            "num_reqs": r.num_running_reqs + r.num_waiting_reqs,               # 总请求数 = 运行中 + 等待中
            "num_waiting_reqs": r.num_waiting_reqs,                            # 等待中请求数
            "num_tokens": r.num_total_tokens,                                  # 总 token 槽数
            "num_pending_tokens": r.num_total_tokens - r.num_used_tokens,      # 空闲 token 槽数
            "ts_tic": ts,                                                       # 采样时间戳
        }
        for r in load_results
    ]


# -------- 内部状态设置端点 --------
# 示例用法：curl -s -X POST http://localhost:30000/set_internal_state -H "Content-Type: application/json" -d '{"server_args": {"pp_max_micro_batch_size": 8}}'
@app.api_route("/set_internal_state", methods=["POST", "PUT"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)  # 可选管理员鉴权
async def set_internal_state(obj: SetInternalStateReq, request: Request):
    res = await _global_state.tokenizer_manager.set_internal_state(obj)  # 透传至 tokenizer_manager 处理
    return res


# -------- Dumper 控制端点（仅在 DUMPER_SERVER_PORT=reuse 时注册） --------
# 避免直接导入 dumper.py 以减少不必要依赖
if os.environ.get("DUMPER_SERVER_PORT") == "reuse":

    @app.api_route("/dumper/{method}", methods=["POST"])
    @auth_level(AuthLevel.ADMIN_OPTIONAL)  # 可选管理员鉴权
    async def _dumper_control_handler(method: str, request: Request):
        body_bytes = await request.body()                          # 读取请求体原始字节
        body = await request.json() if body_bytes else {}          # 有请求体则解析 JSON，否则为空字典
        obj = DumperControlReqInput(method=method, body=body)      # 构造 Dumper 控制请求对象
        results = await _global_state.tokenizer_manager.dumper_control(obj)  # 发送至 tokenizer_manager
        if any(not r.success for r in results):                    # 若任意分片失败则返回 400
            errors = [r.error for r in results if not r.success]
            return ORJSONResponse(status_code=400, content={"error": errors})
        return [x for result in results for x in result.response]  # 展开并返回所有分片的响应列表


# -------- 原生文本生成端点 --------
# FastAPI 会自动将 JSON 请求体反序列化为 GenerateReqInput 对象
@app.api_route(
    "/generate",
    methods=["POST", "PUT"],
    response_class=SGLangORJSONResponse,  # 使用 SGLang 自定义 ORJSON 响应类
)
async def generate_request(obj: GenerateReqInput, request: Request):
    """Handle a generate request."""
    if obj.stream:
        # 流式响应：异步迭代生成结果并以 SSE 格式发送
        async def stream_results() -> AsyncIterator[bytes]:
            try:
                async for out in _global_state.tokenizer_manager.generate_request(
                    obj, request
                ):
                    yield b"data: " + dumps_json(out) + b"\n\n"  # 每条结果以 SSE data: 格式发送
            except ValueError as e:
                out = {"error": {"message": str(e)}}
                logger.error(f"[http_server] Error: {e}")
                yield b"data: " + dumps_json(out) + b"\n\n"      # 错误信息也通过 SSE 发送
            yield b"data: [DONE]\n\n"                             # 流结束标志

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",                                            # SSE 媒体类型
            background=_global_state.tokenizer_manager.create_abort_task(obj),        # 连接断开时自动中止任务
        )
    else:
        # 非流式响应：直接获取第一个（也是唯一一个）结果
        try:
            ret = await _global_state.tokenizer_manager.generate_request(
                obj, request
            ).__anext__()  # 取异步生成器的第一个值
            return orjson_response(ret)
        except ValueError as e:
            logger.error(f"[http_server] Error: {e}")
            return _create_error_response(e)


# -------- 原生 Embedding 编码端点 --------
@app.api_route("/encode", methods=["POST", "PUT"])
async def encode_request(obj: EmbeddingReqInput, request: Request):
    """Handle an embedding request."""
    try:
        ret = await _global_state.tokenizer_manager.generate_request(
            obj, request
        ).__anext__()  # 获取第一个（唯一）嵌入结果
        return ret
    except ValueError as e:
        return _create_error_response(e)


# -------- 原生分类/奖励模型端点 --------
@app.api_route("/classify", methods=["POST", "PUT"])
async def classify_request(obj: EmbeddingReqInput, request: Request):
    """Handle a reward model request. Now the arguments and return values are the same as embedding models."""
    try:
        ret = await _global_state.tokenizer_manager.generate_request(
            obj, request
        ).__anext__()  # 获取分类结果（与 embedding 接口共用）
        return ret
    except ValueError as e:
        return _create_error_response(e)


# -------- Radix Cache 清空端点 --------
@app.api_route("/flush_cache", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def flush_cache(timeout: float = Query(0.0, ge=0.0)):  # timeout 查询参数，最小为 0
    """Flush the radix cache."""
    ret = await _global_state.tokenizer_manager.flush_cache(timeout_s=timeout)  # 调用清空缓存接口
    if ret.success:
        content = (
            "Cache flushed.\nPlease check backend logs for more details. "
            "(When there are running or waiting requests, the operation will not be performed.)\n"
        )
    else:
        content = ret.message or "Flush cache failed.\n"  # 失败时返回错误信息
    return Response(
        content=content,
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )


# -------- 外部语料库管理端点（用于 ngram 推测解码） --------
@app.post("/add_external_corpus")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def add_external_corpus(request: Request):
    """Add an external corpus for ngram speculative decoding."""
    from sglang.srt.managers.io_struct import AddExternalCorpusReqInput

    try:
        obj = AddExternalCorpusReqInput(**(await request.json()))  # 解析请求体为语料请求对象
    except TypeError as e:
        return ORJSONResponse(
            {"success": False, "message": str(e)},
            status_code=HTTPStatus.BAD_REQUEST,
        )
    result = await _global_state.tokenizer_manager.add_external_corpus(obj)  # 添加语料库
    return ORJSONResponse(
        {
            "success": result.success,                           # 是否成功
            "corpus_id": result.corpus_id,                       # 语料库 ID
            "message": result.message,                           # 结果消息
            "loaded_token_count": result.loaded_token_count,     # 已加载的 token 数量
        },
        status_code=200 if result.success else HTTPStatus.BAD_REQUEST,
    )


# -------- 移除外部语料库端点 --------
@app.post("/remove_external_corpus")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def remove_external_corpus(request: Request):
    """Remove an external corpus by ID."""
    body = await request.json()
    corpus_id = body.get("corpus_id")       # 从请求体获取 corpus_id
    if not corpus_id:
        return ORJSONResponse(
            {"success": False, "message": "corpus_id is required."},
            status_code=HTTPStatus.BAD_REQUEST,
        )
    result = await _global_state.tokenizer_manager.remove_external_corpus(corpus_id)  # 按 ID 删除语料库
    return ORJSONResponse(
        {"success": result.success, "message": result.message},
        status_code=200 if result.success else HTTPStatus.BAD_REQUEST,
    )


# -------- 列出所有外部语料库端点 --------
@app.get("/list_external_corpora")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def list_external_corpora():
    """List all active external corpora."""
    result = await _global_state.tokenizer_manager.list_external_corpora()  # 获取所有语料库列表
    return ORJSONResponse(
        {
            "success": result.success,
            "corpus_token_counts": result.corpus_token_counts,  # 各语料库 token 数量统计
            "message": result.message,
        },
        status_code=200 if result.success else HTTPStatus.BAD_REQUEST,
    )


# -------- HiCache 存储后端管理端点 --------
# 清空 HiCache 存储后端（废弃旧路径）
@app.api_route("/clear_hicache_storage_backend", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def clear_hicache_storage_backend_deprecated():
    """Deprecated: use POST /hicache/storage-backend/clear."""
    ret = await _global_state.tokenizer_manager.clear_hicache_storage()  # 调用清空接口
    return Response(
        content=(
            "Deprecated endpoint. Use POST /hicache/storage-backend/clear.\n"
            "Hierarchical cache storage backend cleared.\n"
        ),
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )


# 示例用法：curl -s -X POST http://127.0.0.1:30000/clear_hicache_storage_backend
# -------- 清空 HiCache 存储后端（新路径） --------
@app.api_route("/hicache/storage-backend/clear", methods=["POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def clear_hicache_storage_backend():
    """Clear the hierarchical cache storage backend."""
    ret = await _global_state.tokenizer_manager.clear_hicache_storage()  # 清空分层缓存存储
    return Response(
        content="Hierarchical cache storage backend cleared.\n",
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )


# 示例用法：
# curl -s -X PUT http://127.0.0.1:30000/hicache/storage-backend \
#  -H 'Content-Type: application/json' \
#   -d '{
#     "hicache_storage_backend": "file",
#     "hicache_storage_backend_extra_config_json": "{}",
#     "hicache_storage_prefetch_policy": "timeout",
#     "hicache_write_policy": "write_through"
#   }'
# -------- 挂载（启用）HiCache 存储后端端点 --------
@app.api_route("/hicache/storage-backend", methods=["PUT"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def attach_hicache_storage_backend(obj: AttachHiCacheStorageReqInput):
    """Attach (enable) HiCache storage backend at runtime.

    Only allowed when there are NO running / queued requests.
    """
    if not _global_state.tokenizer_manager.server_args.admin_api_key:
        return _admin_api_key_missing_response()  # 必须配置 admin_api_key 才能操作

    ret = await _global_state.tokenizer_manager.attach_hicache_storage(
        hicache_storage_backend=obj.hicache_storage_backend,                                    # 存储后端类型
        hicache_storage_backend_extra_config_json=obj.hicache_storage_backend_extra_config_json,  # 额外配置 JSON
        hicache_storage_prefetch_policy=obj.hicache_storage_prefetch_policy,                    # 预取策略
        hicache_write_policy=obj.hicache_write_policy,                                          # 写入策略
    )
    msg = getattr(ret, "message", "")  # 获取附加消息（若有）
    return Response(
        content=(
            (
                "HiCache storage backend attached.\n"
                if ret.success
                else "Failed to attach HiCache storage backend.\n"
            )
            + (msg + "\n" if msg else "")  # 追加详细消息
        ),
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )


# 示例用法：curl -s -X DELETE http://127.0.0.1:30000/hicache/storage-backend
# -------- 卸载（禁用）HiCache 存储后端端点 --------
@app.api_route("/hicache/storage-backend", methods=["DELETE"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def detach_hicache_storage_backend():
    """Detach (disable) HiCache storage backend at runtime.

    Only allowed when there are NO running / queued requests.
    """
    if not _global_state.tokenizer_manager.server_args.admin_api_key:
        return _admin_api_key_missing_response()  # 需要 admin_api_key

    ret = await _global_state.tokenizer_manager.detach_hicache_storage()  # 卸载存储后端
    msg = getattr(ret, "message", "")  # 获取附加消息
    return Response(
        content=(
            (
                "HiCache storage backend detached.\n"
                if ret.success
                else "Failed to detach HiCache storage backend.\n"
            )
            + (msg + "\n" if msg else "")
        ),
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )


# 示例用法：curl -s http://127.0.0.1:30000/hicache/storage-backend
# -------- 查询 HiCache 存储后端状态端点 --------
@app.get("/hicache/storage-backend")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def hicache_storage_backend_status():
    """Get current HiCache storage backend status (tokenizer-side view)."""
    if not _global_state.tokenizer_manager.server_args.admin_api_key:
        return _admin_api_key_missing_response()

    return {
        "hicache_storage_backend": _global_state.tokenizer_manager.server_args.hicache_storage_backend,                          # 存储后端类型
        "hicache_storage_backend_extra_config": _global_state.tokenizer_manager.server_args.hicache_storage_backend_extra_config,  # 额外配置
        "hicache_storage_prefetch_policy": _global_state.tokenizer_manager.server_args.hicache_storage_prefetch_policy,            # 预取策略
        "hicache_write_policy": _global_state.tokenizer_manager.server_args.hicache_write_policy,                                  # 写入策略
    }


# -------- 性能分析控制端点 --------
@app.api_route("/start_profile", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def start_profile_async(obj: Optional[ProfileReqInput] = None):
    """Start profiling."""
    if obj is None:
        obj = ProfileReqInput()  # 使用默认参数创建分析请求

    await _global_state.tokenizer_manager.start_profile(  # 向 scheduler 发送开始分析命令
        output_dir=obj.output_dir,                 # 分析结果输出目录
        start_step=obj.start_step,                 # 开始采样的步骤号
        num_steps=obj.num_steps,                   # 采样步骤总数
        activities=obj.activities,                 # 采样活动类型（CPU/CUDA）
        with_stack=obj.with_stack,                 # 是否记录调用栈
        record_shapes=obj.record_shapes,           # 是否记录张量形状
        profile_by_stage=obj.profile_by_stage,     # 是否按阶段分析
        merge_profiles=obj.merge_profiles,         # 是否合并多个分析结果
        profile_prefix=obj.profile_prefix,         # 输出文件名前缀
        profile_stages=obj.profile_stages,         # 需要分析的阶段列表
    )
    return Response(
        content="Start profiling.\n",
        status_code=200,
    )


@app.api_route("/stop_profile", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def stop_profile_async():
    """Stop profiling."""
    await _global_state.tokenizer_manager.stop_profile()  # 停止性能分析并保存结果
    return Response(
        content="Stop profiling. This will take some time.\n",
        status_code=200,
    )


# -------- 链路追踪级别设置端点 --------
@app.api_route("/set_trace_level", methods=["GET", "POST"])
def set_trace_level(level: int = Query(..., ge=0)):  # level 为必填查询参数，最小为 0
    set_global_trace_level(level)  # 更新全局追踪级别

    return Response(
        content="success",
        status_code=200,
    )


# -------- 冻结 GC 端点 --------
@app.api_route("/freeze_gc", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def freeze_gc_async():
    """
    See engine.freeze_gc for more details.
    """
    await _global_state.tokenizer_manager.freeze_gc()  # 冻结 Python 垃圾回收（减少 GC 抖动）
    return Response(
        content="Garbage collection frozen.\n",
        status_code=200,
    )


# -------- 专家分布记录端点（MoE 模型调试用） --------
@app.api_route("/start_expert_distribution_record", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def start_expert_distribution_record_async():
    """Start recording the expert distribution. Clear the previous record if any."""
    await _global_state.tokenizer_manager.start_expert_distribution_record()  # 开始记录专家分布并清除旧记录
    return Response(
        content="Start recording the expert distribution.\n",
        status_code=200,
    )


@app.api_route("/stop_expert_distribution_record", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def stop_expert_distribution_record_async():
    """Stop recording the expert distribution."""
    await _global_state.tokenizer_manager.stop_expert_distribution_record()  # 停止专家分布记录
    return Response(
        content="Stop recording the expert distribution.\n",
        status_code=200,
    )


@app.api_route("/dump_expert_distribution_record", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def dump_expert_distribution_record_async():
    """Dump expert distribution record."""
    await _global_state.tokenizer_manager.dump_expert_distribution_record()  # 将专家分布记录输出到文件
    return Response(
        content="Dump expert distribution record.\n",
        status_code=200,
    )


# -------- 权重热更新端点 --------
@app.post("/update_weights_from_disk")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def update_weights_from_disk(obj: UpdateWeightFromDiskReqInput, request: Request):
    """Update the weights from disk inplace without re-launching the server."""
    success, message, num_paused_requests = (
        await _global_state.tokenizer_manager.update_weights_from_disk(obj, request)  # 从磁盘加载新权重
    )

    content = {
        "success": success,                        # 是否更新成功
        "message": message,                        # 结果消息
        "num_paused_requests": num_paused_requests,  # 更新期间被暂停的请求数
    }
    if success:
        return ORJSONResponse(
            content,
            status_code=HTTPStatus.OK,
        )
    else:
        return ORJSONResponse(
            content,
            status_code=HTTPStatus.BAD_REQUEST,
        )


# -------- 远程实例权重发送组初始化端点 --------
@app.post("/init_weights_send_group_for_remote_instance")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def init_weights_send_group_for_remote_instance(
    obj: InitWeightsSendGroupForRemoteInstanceReqInput, request: Request
):
    success, message = (
        await _global_state.tokenizer_manager.init_weights_send_group_for_remote_instance(
            obj, request
        )  # 初始化向远程实例发送权重所需的通信组
    )
    content = {"success": success, "message": message}
    if success:
        return ORJSONResponse(content, status_code=200)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


# -------- 将权重发送到远程实例端点 --------
@app.post("/send_weights_to_remote_instance")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def send_weights_to_remote_instance(
    obj: SendWeightsToRemoteInstanceReqInput, request: Request
):
    success, message = (
        await _global_state.tokenizer_manager.send_weights_to_remote_instance(
            obj, request
        )  # 将当前实例权重推送至远程实例
    )
    content = {"success": success, "message": message}
    if success:
        return ORJSONResponse(content, status_code=200)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


# -------- 获取远程实例传输引擎信息端点（废弃旧路径） --------
@app.get("/get_remote_instance_transfer_engine_info")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def get_remote_instance_transfer_engine_info(rank: int = None):
    """Get the server information (deprecated - use /remote_instance_transfer_engine_info instead)."""
    logger.warning(
        "Endpoint '/get_remote_instance_transfer_engine_info' is deprecated and will be removed in a future version. "
        "Please use '/remote_instance_transfer_engine_info' instead."
    )
    return await remote_instance_transfer_engine_info(rank=rank)  # 转发到新端点


# -------- 获取远程实例传输引擎信息端点 --------
@app.get("/remote_instance_transfer_engine_info")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def remote_instance_transfer_engine_info(rank: int = None):
    if rank is None or rank < 0:  # 参数校验：rank 必须为非负整数
        return ORJSONResponse(
            {"error": {"message": "Missing or invalid rank parameter"}},
            status_code=HTTPStatus.BAD_REQUEST,
        )

    server_args = _global_state.tokenizer_manager.server_args
    try:
        # 从 bootstrap 服务器获取指定 rank 的传输引擎信息
        resp = requests.get(
            f"{server_args.engine_info_bootstrap_url}/get_transfer_engine_info",
            params={"rank": rank},
            timeout=5,  # 5 秒超时
        )
        if resp.status_code == 200:
            return resp.json()
    except (requests.exceptions.RequestException, ValueError) as e:
        logger.warning(f"Failed to get transfer engine info for rank {rank}: {e}")

    return ORJSONResponse(
        {"error": {"message": f"Failed to get transfer engine info for rank {rank}"}},
        status_code=HTTPStatus.BAD_REQUEST,
    )


# -------- 权重更新通信组管理端点 --------
@app.post("/init_weights_update_group")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def init_weights_update_group(
    obj: InitWeightsUpdateGroupReqInput, request: Request
):
    """Initialize the parameter update group."""
    success, message = await _global_state.tokenizer_manager.init_weights_update_group(
        obj, request
    )  # 初始化用于参数同步的通信组（RLHF 训练场景）
    content = {"success": success, "message": message}
    if success:
        return ORJSONResponse(content, status_code=200)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


@app.post("/destroy_weights_update_group")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def destroy_weights_update_group(
    obj: DestroyWeightsUpdateGroupReqInput, request: Request
):
    """Destroy the parameter update group."""
    success, message = (
        await _global_state.tokenizer_manager.destroy_weights_update_group(obj, request)  # 销毁权重更新通信组
    )
    content = {"success": success, "message": message}
    return ORJSONResponse(
        content, status_code=200 if success else HTTPStatus.BAD_REQUEST
    )


# -------- 从张量热更新权重端点 --------
@app.post("/update_weights_from_tensor")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def update_weights_from_tensor(
    obj: UpdateWeightsFromTensorReqInput, request: Request
):
    """Update the weights from tensor inplace without re-launching the server.
    Notes:
    1. Ensure that the model is on the correct device (e.g., GPU) before calling this endpoint. If the model is moved to the CPU unexpectedly, it may cause performance issues or runtime errors.
    2. HTTP will transmit only the metadata of the tensor, while the tensor itself will be directly copied to the model.
    3. Any binary data in the named tensors should be base64 encoded.
    """
    success, message = await _global_state.tokenizer_manager.update_weights_from_tensor(
        obj, request
    )  # 通过 IPC 共享内存直接覆写模型张量（HTTP 只传元数据）

    content = {"success": success, "message": message}
    return ORJSONResponse(
        content, status_code=200 if success else HTTPStatus.BAD_REQUEST
    )


# -------- 从分布式在线更新权重端点 --------
@app.post("/update_weights_from_distributed")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def update_weights_from_distributed(
    obj: UpdateWeightsFromDistributedReqInput, request: Request
):
    """Update model parameter from distributed online."""
    success, message = (
        await _global_state.tokenizer_manager.update_weights_from_distributed(
            obj, request
        )  # 通过分布式通信（如 NCCL）从训练端同步权重到推理端
    )

    content = {"success": success, "message": message}
    if success:
        return ORJSONResponse(content, status_code=200)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


# -------- 通过 IPC 更新权重端点（checkpoint-engine 集成） --------
@app.post("/update_weights_from_ipc")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def update_weights_from_ipc(obj: UpdateWeightsFromIPCReqInput, request: Request):
    """Update the weights from IPC (Inter-Process Communication) for checkpoint-engine integration."""
    success, message = await _global_state.tokenizer_manager.update_weights_from_ipc(
        obj, request
    )  # 通过 IPC 共享内存从 checkpoint 引擎加载权重

    content = {"success": success, "message": message}
    if success:
        if _global_state.tokenizer_manager.initial_weights_loaded is False:
            _global_state.tokenizer_manager.initial_weights_loaded = True  # 标记初始权重已加载完成
        return ORJSONResponse(content)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


# -------- 更新权重版本号端点 --------
@app.post("/update_weight_version")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def update_weight_version(obj: UpdateWeightVersionReqInput, request: Request):
    """Update the weight version. This operation requires no active requests."""
    if obj.abort_all_requests:
        _global_state.tokenizer_manager.abort_request(abort_all=True)  # 若请求要求，先中止所有正在进行的请求

    # 使用简单直接的方式更新权重版本号（无需复杂的锁机制，因为版本号不影响模型权重）
    try:
        # 更新 server_args 中的权重版本（唯一数据来源）
        _global_state.tokenizer_manager.server_args.weight_version = obj.new_version

        return ORJSONResponse(
            {
                "success": True,
                "message": f"Weight version updated to {obj.new_version}",  # 版本更新成功消息
                "new_version": obj.new_version,                               # 新版本号
            },
            status_code=HTTPStatus.OK,
        )
    except Exception as e:
        return ORJSONResponse(
            {
                "success": False,
                "message": f"Failed to update weight version: {str(e)}",  # 版本更新失败消息
            },
            status_code=HTTPStatus.BAD_REQUEST,
        )


# -------- 按名称获取模型权重端点 --------
@app.api_route("/get_weights_by_name", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def get_weights_by_name(obj: GetWeightsByNameReqInput, request: Request):
    """Get model parameter by name."""
    try:
        ret = await _global_state.tokenizer_manager.get_weights_by_name(obj, request)  # 从模型获取指定名称的权重张量
        if ret is None:
            return _create_error_response("Get parameter by name failed")
        else:
            return ORJSONResponse(ret, status_code=200)
    except Exception as e:
        return _create_error_response(e)


# -------- GPU 内存占用管理端点 --------
@app.api_route("/release_memory_occupation", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def release_memory_occupation(
    obj: ReleaseMemoryOccupationReqInput, request: Request
):
    """Release GPU memory occupation temporarily."""
    try:
        await _global_state.tokenizer_manager.release_memory_occupation(obj, request)  # 临时释放 GPU 内存（如 RLHF 训练阶段）
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/resume_memory_occupation", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def resume_memory_occupation(
    obj: ResumeMemoryOccupationReqInput, request: Request
):
    """Resume GPU memory occupation."""
    try:
        await _global_state.tokenizer_manager.resume_memory_occupation(obj, request)  # 恢复 GPU 内存占用
    except Exception as e:
        return _create_error_response(e)


# -------- 权重校验端点 --------
@app.post("/weights_checker")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def check_weights(obj: CheckWeightsReqInput, request: Request):
    success, message = await _global_state.tokenizer_manager.check_weights(obj, request)  # 验证模型权重完整性
    return ORJSONResponse(
        {"success": success, "message": message},
        status_code=200 if success else HTTPStatus.BAD_REQUEST,
    )


# -------- 人为降速端点（测试用） --------
@app.api_route("/slow_down", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def slow_down(obj: SlowDownReqInput, request: Request):
    """Slow down the system deliberately. Only for testing. Example scenario:
    when we want to test performance of D in large-scale PD disaggregation and have no enough nodes for P,
    we can use this to slow down D to let it have enough running sequences, and then disable slowdown
    to let it run in full batch size.
    """
    try:
        await _global_state.tokenizer_manager.slow_down(obj, request)  # 向 scheduler 发送降速指令
    except Exception as e:
        return _create_error_response(e)


# -------- LoRA 适配器管理端点 --------
@app.api_route("/load_lora_adapter", methods=["POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def load_lora_adapter(obj: LoadLoRAAdapterReqInput, request: Request):
    """Load a new LoRA adapter without re-launching the server."""
    result = await _global_state.tokenizer_manager.load_lora_adapter(obj, request)  # 动态加载 LoRA 适配器

    if result.success:
        return ORJSONResponse(
            result,
            status_code=HTTPStatus.OK,
        )
    else:
        return ORJSONResponse(
            result,
            status_code=HTTPStatus.BAD_REQUEST,
        )


@app.api_route("/load_lora_adapter_from_tensors", methods=["POST"])
async def load_lora_adapter_from_tensors(
    obj: LoadLoRAAdapterFromTensorsReqInput, request: Request
):
    """Load a new LoRA adapter from tensors without re-launching the server."""
    result = await _global_state.tokenizer_manager.load_lora_adapter_from_tensors(
        obj, request
    )  # 从张量数据直接加载 LoRA 适配器（无需写磁盘）

    if result.success:
        return ORJSONResponse(result, status_code=HTTPStatus.OK)
    else:
        return ORJSONResponse(result, status_code=HTTPStatus.BAD_REQUEST)


@app.api_route("/unload_lora_adapter", methods=["POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def unload_lora_adapter(obj: UnloadLoRAAdapterReqInput, request: Request):
    """Load a new LoRA adapter without re-launching the server."""
    result = await _global_state.tokenizer_manager.unload_lora_adapter(obj, request)  # 动态卸载已加载的 LoRA 适配器

    if result.success:
        return ORJSONResponse(
            result,
            status_code=HTTPStatus.OK,
        )
    else:
        return ORJSONResponse(
            result,
            status_code=HTTPStatus.BAD_REQUEST,
        )


# -------- 会话管理端点 --------
@app.api_route("/open_session", methods=["GET", "POST"])
async def open_session(obj: OpenSessionReqInput, request: Request):
    """Open a session, and return its unique session id."""
    try:
        session_id = await _global_state.tokenizer_manager.open_session(obj, request)  # 开启新会话并返回唯一 ID
        if session_id is None:
            raise Exception(
                "Failed to open the session. Check if a session with the same id is still open."
            )
        return session_id
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/close_session", methods=["GET", "POST"])
async def close_session(obj: CloseSessionReqInput, request: Request):
    """Close the session."""
    try:
        await _global_state.tokenizer_manager.close_session(obj, request)  # 关闭指定会话并释放资源
        return Response(status_code=200)
    except Exception as e:
        return _create_error_response(e)


# -------- 日志配置端点 --------
@app.api_route("/configure_logging", methods=["GET", "POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def configure_logging(obj: ConfigureLoggingReq, request: Request):
    """Configure the request logging options."""
    _global_state.tokenizer_manager.configure_logging(obj)  # 动态修改请求日志配置（无需重启）
    return Response(status_code=200)


# -------- 中止请求端点 --------
@app.post("/abort_request")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def abort_request(obj: AbortReq, request: Request):
    """Abort a request."""
    try:
        _global_state.tokenizer_manager.abort_request(
            rid=obj.rid, abort_all=obj.abort_all  # 按 rid 中止单个请求或中止所有请求
        )
        return Response(status_code=200)
    except Exception as e:
        return _create_error_response(e)


# -------- 函数调用解析端点 --------
@app.post("/parse_function_call")
async def parse_function_call_request(obj: ParseFunctionCallReq, request: Request):
    """
    A native API endpoint to parse function calls from a text.
    """
    # 1) 根据请求体中指定的工具和解析器类型初始化解析器
    parser = FunctionCallParser(tools=obj.tools, tool_call_parser=obj.tool_call_parser)

    # 2) 调用非流式解析方法，一次性解析全部函数调用
    normal_text, calls = parser.parse_non_stream(obj.text)

    # 3) 构造响应：包含普通文本和解析出的函数调用列表
    response_data = {
        "normal_text": normal_text,
        "calls": [
            call.model_dump() for call in calls
        ],  # 将 pydantic 对象转换为字典
    }

    return ORJSONResponse(content=response_data, status_code=200)


# -------- 推理文本分离端点 --------
@app.post("/separate_reasoning")
async def separate_reasoning_request(obj: SeparateReasoningReqInput, request: Request):
    """
    A native API endpoint to separate reasoning from a text.
    """
    # 1) 根据请求体中的模型类型初始化推理解析器
    parser = ReasoningParser(model_type=obj.reasoning_parser, request=request)

    # 2) 调用非流式解析方法，分离推理过程与普通文本
    reasoning_text, normal_text = parser.parse_non_stream(obj.text)

    # 3) 构造响应数据
    response_data = {
        "reasoning_text": reasoning_text,  # 推理过程文本（如 <think>...</think> 内容）
        "text": normal_text,               # 去除推理部分后的正常文本
    }

    return ORJSONResponse(content=response_data, status_code=200)


# -------- 生成暂停/继续端点 --------
@app.post("/pause_generation")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def pause_generation(obj: PauseGenerationReqInput, request: Request):
    """Pause generation."""
    await _global_state.tokenizer_manager.pause_generation(obj)  # 暂停所有正在进行的生成任务
    return ORJSONResponse(
        content={"message": "Generation paused successfully.", "status": "ok"},
        status_code=200,
    )


@app.post("/continue_generation")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def continue_generation(obj: ContinueGenerationReqInput, request: Request):
    """Continue generation."""
    await _global_state.tokenizer_manager.continue_generation(obj)  # 恢复被暂停的生成任务
    return ORJSONResponse(
        content={"message": "Generation continued successfully.", "status": "ok"},
        status_code=200,
    )


##### OpenAI 兼容 API 端点 #####


# -------- 文本补全端点 --------
@app.post("/v1/completions", dependencies=[Depends(validate_json_request)])
async def openai_v1_completions(request: CompletionRequest, raw_request: Request):
    """OpenAI-compatible text completion endpoint."""
    return await raw_request.app.state.openai_serving_completion.handle_request(
        request, raw_request  # 转发至 OpenAI Completion 服务处理器
    )


# -------- 聊天补全端点 --------
@app.post("/v1/chat/completions", dependencies=[Depends(validate_json_request)])
async def openai_v1_chat_completions(
    request: ChatCompletionRequest, raw_request: Request
):
    """OpenAI-compatible chat completion endpoint."""
    return await raw_request.app.state.openai_serving_chat.handle_request(
        request, raw_request  # 转发至 OpenAI Chat 服务处理器
    )


# -------- Embedding 端点 --------
@app.post(
    "/v1/embeddings",
    response_class=ORJSONResponse,
    dependencies=[Depends(validate_json_request)],
)
async def openai_v1_embeddings(request: EmbeddingRequest, raw_request: Request):
    """OpenAI-compatible embeddings endpoint."""
    return await raw_request.app.state.openai_serving_embedding.handle_request(
        request, raw_request  # 转发至 OpenAI Embedding 服务处理器
    )


# -------- 分类端点 --------
@app.post(
    "/v1/classify",
    response_class=ORJSONResponse,
    dependencies=[Depends(validate_json_request)],
)
async def openai_v1_classify(request: ClassifyRequest, raw_request: Request):
    """OpenAI-compatible classification endpoint."""
    return await raw_request.app.state.openai_serving_classify.handle_request(
        request, raw_request  # 转发至 OpenAI Classify 服务处理器
    )


# -------- Tokenize 端点（双路径） --------
@app.post(
    "/v1/tokenize",
    response_class=ORJSONResponse,
    dependencies=[Depends(validate_json_request)],
)
@app.post(
    "/tokenize",                        # 同时注册旧路径（不在 schema 中显示）
    response_class=ORJSONResponse,
    dependencies=[Depends(validate_json_request)],
    include_in_schema=False,
)
async def openai_v1_tokenize(request: TokenizeRequest, raw_request: Request):
    """OpenAI-compatible tokenization endpoint."""
    return await raw_request.app.state.openai_serving_tokenize.handle_request(
        request, raw_request
    )


# -------- Detokenize 端点（双路径） --------
@app.post(
    "/v1/detokenize",
    response_class=ORJSONResponse,
    dependencies=[Depends(validate_json_request)],
)
@app.post(
    "/detokenize",                      # 同时注册旧路径（不在 schema 中显示）
    response_class=ORJSONResponse,
    dependencies=[Depends(validate_json_request)],
    include_in_schema=False,
)
async def openai_v1_detokenize(request: DetokenizeRequest, raw_request: Request):
    """OpenAI-compatible detokenization endpoint."""
    return await raw_request.app.state.openai_serving_detokenize.handle_request(
        request, raw_request
    )


# -------- 语音转录端点 --------
@app.post("/v1/audio/transcriptions")
async def openai_v1_audio_transcriptions(
    raw_request: Request,
    file: UploadFile = File(...),                             # 上传的音频文件
    model: str = Form(default="default"),                    # 模型名称
    language: Optional[str] = Form(default=None),            # 源语言（可选）
    response_format: str = Form(default="json"),             # 响应格式
    temperature: float = Form(default=0.0),                  # 采样温度
    stream: bool = Form(default=False),                      # 是否流式返回
    timestamp_granularities: Optional[List[str]] = Form(
        default=None, alias="timestamp_granularities[]"      # 时间戳粒度列表
    ),
):
    """OpenAI-compatible audio transcription endpoint."""
    # 校验响应格式是否在支持范围内
    if response_format not in ["json", "text", "verbose_json"]:
        return ORJSONResponse(
            content={
                "error": {
                    "message": "Only 'json', 'text', and 'verbose_json' formats supported"
                }
            },
            status_code=400,
        )

    audio_data = await file.read()  # 读取上传的音频文件字节数据

    return (
        await raw_request.app.state.openai_serving_transcription.create_transcription(
            audio_data=audio_data,
            model=model,
            language=language,
            response_format=response_format,
            temperature=temperature,
            stream=stream,
            timestamp_granularities=timestamp_granularities,
            raw_request=raw_request,
        )
    )


# -------- 模型列表端点 --------
@app.get("/v1/models", response_class=ORJSONResponse)
async def available_models():
    """Show available models. OpenAI-compatible endpoint."""
    served_model_names = [_global_state.tokenizer_manager.served_model_name]  # 获取当前服务的模型名列表
    model_cards = []

    # 添加基础模型卡片
    for served_model_name in served_model_names:
        model_cards.append(
            ModelCard(
                id=served_model_name,                                                          # 模型 ID
                root=served_model_name,                                                        # 根模型名
                max_model_len=_global_state.tokenizer_manager.model_config.context_len,        # 最大上下文长度
            )
        )

    # 若启用了 LoRA，追加所有已注册的 LoRA 适配器卡片
    if _global_state.tokenizer_manager.server_args.enable_lora:
        lora_registry = _global_state.tokenizer_manager.lora_registry
        for _, lora_ref in lora_registry.get_all_adapters().items():
            model_cards.append(
                ModelCard(
                    id=lora_ref.lora_name,            # LoRA 适配器名称
                    root=lora_ref.lora_path,           # LoRA 适配器路径
                    parent=served_model_names[0],      # 父模型名
                    max_model_len=None,                # LoRA 适配器不限制长度
                )
            )

    return ModelList(data=model_cards)


# -------- 单个模型信息查询端点 --------
@app.get("/v1/models/{model:path}", response_class=ORJSONResponse)
async def retrieve_model(model: str):
    """Retrieves a model instance, providing basic information about the model."""
    served_model_names = [_global_state.tokenizer_manager.served_model_name]

    if model not in served_model_names:
        return ORJSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": f"The model '{model}' does not exist",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )

    return ModelCard(
        id=model,
        root=model,
        max_model_len=_global_state.tokenizer_manager.model_config.context_len,
    )


# -------- 评分端点 --------
@app.post("/v1/score", dependencies=[Depends(validate_json_request)])
async def v1_score_request(request: ScoringRequest, raw_request: Request):
    """Endpoint for the scoring API. Supports CausalLM (logprob-based) and SequenceClassification (class logit-based) models. See Engine.score() for documentation."""
    return await raw_request.app.state.openai_serving_score.handle_request(
        request, raw_request  # 转发至评分服务处理器
    )


# -------- Responses API 端点 --------
@app.post("/v1/responses", dependencies=[Depends(validate_json_request)])
async def v1_responses_request(request: dict, raw_request: Request):
    """Endpoint for the responses API with reasoning support."""

    request_obj = ResponsesRequest(**request)  # 将字典反序列化为 ResponsesRequest 对象
    result = await raw_request.app.state.openai_serving_responses.create_responses(
        request_obj, raw_request
    )

    # 处理流式响应（AsyncGenerator）
    if isinstance(result, AsyncGenerator):
        return StreamingResponse(
            result,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},  # SSE 必要响应头
        )

    return result


@app.get("/v1/responses/{response_id}")
async def v1_retrieve_responses(response_id: str, raw_request: Request):
    """Retrieve a response by ID."""
    return await raw_request.app.state.openai_serving_responses.retrieve_responses(
        response_id  # 根据响应 ID 查询已存储的响应
    )


@app.post("/v1/responses/{response_id}/cancel")
async def v1_cancel_responses(response_id: str, raw_request: Request):
    """Cancel a background response."""
    return await raw_request.app.state.openai_serving_responses.cancel_responses(
        response_id  # 取消后台正在进行的响应任务
    )


# -------- Rerank 端点 --------
@app.api_route(
    "/v1/rerank", methods=["POST", "PUT"], dependencies=[Depends(validate_json_request)]
)
async def v1_rerank_request(request: V1RerankReqInput, raw_request: Request):
    """Endpoint for reranking documents based on query relevance."""
    return await raw_request.app.state.openai_serving_rerank.handle_request(
        request, raw_request  # 转发至 Rerank 服务处理器
    )


##### Ollama 兼容 API 端点 #####

# 读取 Ollama 根路由环境变量，若设置则使用自定义路径，否则使用默认 /
_ollama_root_route = os.environ.get("SGLANG_OLLAMA_ROOT_ROUTE")
if _ollama_root_route is not None:

    @app.get(_ollama_root_route)
    @app.head(_ollama_root_route)
    async def ollama_root():
        """Ollama-compatible root endpoint."""
        return "Ollama is running"  # 兼容 Ollama 客户端的根路径检测

else:

    @app.get("/")
    @app.head("/")
    async def sglang_root():
        """Default root endpoint."""
        return "SGLang is running"  # 默认根路径返回服务状态


# -------- Ollama 聊天端点 --------
@app.post(os.environ.get("SGLANG_OLLAMA_CHAT_ROUTE", "/api/chat"))
async def ollama_chat(request: OllamaChatRequest, raw_request: Request):
    """Ollama-compatible chat endpoint."""
    return await raw_request.app.state.ollama_serving.handle_chat(request, raw_request)


# -------- Ollama 生成端点 --------
@app.post(os.environ.get("SGLANG_OLLAMA_GENERATE_ROUTE", "/api/generate"))
async def ollama_generate(request: OllamaGenerateRequest, raw_request: Request):
    """Ollama-compatible generate endpoint."""
    return await raw_request.app.state.ollama_serving.handle_generate(
        request, raw_request
    )


# -------- Ollama 模型列表端点 --------
@app.get(os.environ.get("SGLANG_OLLAMA_TAGS_ROUTE", "/api/tags"))
async def ollama_tags(raw_request: Request):
    """Ollama-compatible list models endpoint."""
    return raw_request.app.state.ollama_serving.get_tags()


# -------- Ollama 模型信息端点 --------
@app.post(os.environ.get("SGLANG_OLLAMA_SHOW_ROUTE", "/api/show"))
async def ollama_show(request: OllamaShowRequest, raw_request: Request):
    """Ollama-compatible show model info endpoint."""
    return raw_request.app.state.ollama_serving.get_show(request.model)


##### Anthropic 兼容 API 端点 #####


# -------- Anthropic Messages 端点 --------
@app.post("/v1/messages", dependencies=[Depends(validate_json_request)])
async def anthropic_v1_messages(
    request: AnthropicMessagesRequest, raw_request: Request
):
    """Anthropic-compatible Messages API endpoint."""
    return await raw_request.app.state.anthropic_serving.handle_messages(
        request, raw_request  # 转发至 Anthropic Messages 服务处理器
    )


# -------- Anthropic Token 计数端点 --------
@app.post("/v1/messages/count_tokens", dependencies=[Depends(validate_json_request)])
async def anthropic_v1_count_tokens(
    request: AnthropicCountTokensRequest, raw_request: Request
):
    """Anthropic-compatible token counting endpoint."""
    return await raw_request.app.state.anthropic_serving.handle_count_tokens(
        request, raw_request  # 转发至 Anthropic Token 计数处理器
    )


## SageMaker API
# -------- SageMaker 健康检查端点 --------
@app.get("/ping")
async def sagemaker_health() -> Response:
    """Check the health of the http server."""
    return Response(status_code=200)  # SageMaker 平台要求 /ping 端点返回 200 表示服务就绪


# -------- SageMaker 推理端点 --------
@app.post("/invocations")
async def sagemaker_chat_completions(
    request: ChatCompletionRequest, raw_request: Request
):
    """OpenAI-compatible chat completion endpoint."""
    return await raw_request.app.state.openai_serving_chat.handle_request(
        request, raw_request  # 复用 OpenAI Chat 服务处理器响应 SageMaker 调用
    )


## Vertex AI API
# -------- Vertex AI 生成端点 --------
@app.post(os.environ.get("AIP_PREDICT_ROUTE", "/vertex_generate"))
async def vertex_generate(vertex_req: VertexGenerateReqInput, raw_request: Request):
    if not vertex_req.instances:
        return []                 # Vertex AI 请求中无实例则直接返回空列表
    inputs = {}
    # 从每个实例中提取 text / input_ids / input_embeds 中的第一个可用字段
    for input_key in ("text", "input_ids", "input_embeds"):
        if vertex_req.instances[0].get(input_key):
            inputs[input_key] = [
                instance.get(input_key) for instance in vertex_req.instances
            ]
            break
    # 收集所有实例中不为 None 的图像数据
    image_data = [
        instance.get("image_data")
        for instance in vertex_req.instances
        if instance.get("image_data") is not None
    ] or None
    # 构造生成请求对象，合并输入和采样参数
    req = GenerateReqInput(
        **inputs,
        image_data=image_data,
        **(vertex_req.parameters or {}),  # 采样参数来自请求的 parameters 字段
    )
    ret = await generate_request(req, raw_request)  # 复用 /generate 端点逻辑
    if isinstance(ret, Response):
        return ret
    return ORJSONResponse({"predictions": ret})     # 包装为 Vertex AI 格式的 predictions 字段


# -------- 辅助函数：构造通用错误响应 --------
# 将异常信息包装为 400 Bad Request JSON 响应
def _create_error_response(e):
    return ORJSONResponse(
        {"error": {"message": str(e)}}, status_code=HTTPStatus.BAD_REQUEST
    )


# FIXME：理论上某些管理端点应配置 ADMIN_FORCE，但目前这样做会导致所有端点都经过
# add_api_key_middleware（即使未配置 api-key 或 admin-api-key）。
#
# 目前通过显式检查 admin_api_key 参数来模拟 ADMIN_FORCE 行为。
# 待鉴权机制重构后，应切换为直接使用 ADMIN_FORCE 装饰器。
# -------- 辅助函数：缺少管理员密钥响应 --------
def _admin_api_key_missing_response(
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
) -> ORJSONResponse:
    return ORJSONResponse(
        content={
            "error": (
                "This endpoint requires admin API key, but this server was started "
                "without one (admin-api-key). Restart with --admin-api-key to enable."
            )
        },
        status_code=status_code,
    )


# -------- 最小 PNG 图片（warmup 用） --------
# 用于视觉语言模型 warmup 的 32x32 黑色 PNG 图片（base64 编码）
# GLM4v 等模型要求输入图片至少 32x32 像素
MINIMUM_PNG_PICTURE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAbUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGBcua9HOR7Y6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=="


# -------- 服务器 warmup 核心逻辑 --------
# 等待服务器启动后发送测试请求，确认推理引擎正常工作
def _execute_server_warmup(server_args: ServerArgs):
    headers = {}
    url = server_args.url()  # 获取服务器完整 URL
    if server_args.api_key:
        headers["Authorization"] = f"Bearer {server_args.api_key}"  # 若配置了 API key 则设置鉴权头

    ssl_verify = server_args.ssl_verify()  # 获取 SSL 证书验证配置

    # 轮询等待服务器启动（最多 120 秒）
    success = False
    for _ in range(120):
        time.sleep(1)  # 每秒检查一次
        try:
            res = requests.get(
                url + "/model_info", timeout=5, headers=headers, verify=ssl_verify
            )
            assert res.status_code == 200, f"{res=}, {res.text=}"
            success = True
            break
        except (AssertionError, requests.exceptions.RequestException):
            last_traceback = get_exception_traceback()  # 记录最后一次异常堆栈
            pass

    if not success:
        logger.error(f"Initialization failed. warmup error: {last_traceback}")
        kill_process_tree(os.getpid())  # 初始化失败则终止整个进程树
        return success

    model_info = res.json()  # 解析模型信息响应

    # 根据模型类型和配置构造 warmup 请求
    is_vlm = bool(model_info.get("has_image_understanding", False))  # 是否为视觉语言模型
    if model_info["is_generation"]:
        if is_vlm and not server_args.skip_tokenizer_init:
            request_name = "/v1/chat/completions"  # VLM 使用聊天补全格式（支持图像输入）
        else:
            request_name = "/generate"             # 普通生成模型使用 /generate
    else:
        request_name = "/encode"                   # 嵌入模型使用 /encode
    max_new_tokens = 8 if model_info["is_generation"] else 1  # 生成模型产 8 个 token，嵌入模型产 1 个
    json_data = {
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        },
    }
    if server_args.skip_tokenizer_init:
        # 跳过 tokenizer 初始化时，直接使用 token id 列表
        json_data["input_ids"] = [[10, 11, 12] for _ in range(server_args.dp_size)]
        # TODO: 解决 embedding 模型对长度为 1 的列表报错的问题
        if server_args.dp_size == 1:
            json_data["input_ids"] = json_data["input_ids"][0]
    elif (
        is_vlm
        and server_args.disaggregation_mode == "null"
        and model_info["is_generation"]
    ):
        # TODO: ChatCompletionRequest 不包含 PD 分离所需的 bootstrap 信息，暂时禁用图像 warmup
        # 仅对生成模型使用聊天补全格式
        json_data = {
            "model": _global_state.tokenizer_manager.served_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{MINIMUM_PNG_PICTURE_BASE64}"  # 使用最小图片
                            },
                        },
                        {
                            "type": "text",
                            "text": "Describe the image.",  # 简单的图像描述提示词
                        },
                    ],
                }
            ],
            "max_tokens": max_new_tokens,
            "stream": False,
            "temperature": 0.0,
        }
    else:
        # 使用固定文本作为 warmup 输入（批量复制 dp_size 份）
        json_data["text"] = ["The capital city of France is"] * server_args.dp_size
        # TODO: 解决 embedding 模型对长度为 1 的列表报错的问题
        if server_args.dp_size == 1:
            json_data["text"] = json_data["text"][0]

    # 若配置了调试张量导出，从文件加载输入 id 并跳过生成
    if server_args.debug_tensor_dump_input_file:
        json_data.pop("text", None)
        json_data["input_ids"] = np.load(
            server_args.debug_tensor_dump_input_file
        ).tolist()
        json_data["sampling_params"]["max_new_tokens"] = 0  # 不实际生成，只做前向传播

    # 发送 warmup 请求并等待完成
    warmup_timeout = envs.SGLANG_WARMUP_TIMEOUT.get()
    try:
        if server_args.disaggregation_mode == "null":
            # 标准模式 warmup
            res = requests.post(
                url + request_name,
                json=json_data,
                headers=headers,
                timeout=warmup_timeout if warmup_timeout > 0 else 600,  # 默认 10 分钟超时
                verify=ssl_verify,
            )
            assert res.status_code == 200, f"{res.text}"
            _global_state.tokenizer_manager.server_status = ServerStatus.Up  # 标记服务就绪

        else:
            # PD 分离模式 warmup
            logger.info(f"Start of pd disaggregation warmup ...")
            json_data = {
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 8,
                    "ignore_eos": True,       # 忽略 EOS 确保生成满足指定长度
                },
                "bootstrap_host": [FAKE_BOOTSTRAP_HOST] * server_args.dp_size,  # 使用假 bootstrap 地址
                # 确保 prefill warmup 期间每个 dp rank 使用唯一的 bootstrap_room
                "bootstrap_room": [
                    i * (2**63 // server_args.dp_size) + (i % server_args.tp_size)
                    for i in range(server_args.dp_size)
                ],
                "input_ids": [[10, 11, 12, 13]] * server_args.dp_size,  # 4 个 token 的测试输入
            }
            res = requests.post(
                url + request_name,
                json=json_data,
                headers=headers,
                timeout=(
                    warmup_timeout if warmup_timeout > 0 else 1800
                ),  # PD 分离 warmup 超时更长（DeepGEMM 预缓存耗时）
                verify=ssl_verify,
            )
            if res.status_code == 200:
                logger.info(
                    f"End of prefill disaggregation mode warmup with status {res.status_code}, resp: {res.json()}"
                )
                _global_state.tokenizer_manager.server_status = ServerStatus.Up    # 标记服务就绪
            else:
                logger.info(
                    "Prefill disaggregation mode warm Up Failed, status code: {}".format(
                        res.status_code
                    )
                )
                _global_state.tokenizer_manager.server_status = ServerStatus.UnHealthy  # 标记服务不健康

    except Exception:
        last_traceback = get_exception_traceback()
        logger.error(f"Initialization failed. warmup error: {last_traceback}")
        kill_process_tree(os.getpid())  # warmup 失败则终止整个进程树
        return False

    # 调试打印（已注释）
    # logger.info(f"warmup request returns: {res.json()=}")
    return success


# -------- 等待并执行 warmup --------
# 可选等待权重就绪，然后调用 warmup 函数确认服务就绪
def _wait_and_warmup(
    server_args: ServerArgs,
    launch_callback: Optional[Callable[[], None]] = None,       # 服务就绪后的回调函数（可选）
    execute_warmup_func: Callable = _execute_server_warmup,     # warmup 执行函数（可替换为自定义实现）
):
    if server_args.checkpoint_engine_wait_weights_before_ready:
        _wait_weights_ready()  # 若配置了 checkpoint 引擎，需先等待权重加载完毕

    # 执行服务 warmup（发送测试请求）
    if not server_args.skip_server_warmup:
        if not execute_warmup_func(server_args):
            return  # warmup 失败则直接返回
    else:
        _global_state.tokenizer_manager.server_status = ServerStatus.Up  # 跳过 warmup 直接标记就绪

    # 服务已就绪，打印日志
    logger.info("The server is fired up and ready to roll!")

    if server_args.delete_ckpt_after_loading:
        delete_directory(server_args.model_path)  # 若配置了加载后删除 checkpoint，则清理模型目录

    if server_args.debug_tensor_dump_input_file:
        kill_process_tree(os.getpid())  # 调试张量导出模式下完成后终止进程

    if launch_callback is not None:
        launch_callback()  # 执行启动完成回调（如通知父进程服务就绪）


# -------- 等待权重就绪 --------
def _wait_weights_ready():
    """Wait for weights to be ready within the specified timeout."""
    timeout = WAIT_WEIGHTS_READY_TIMEOUT  # 从全局常量获取超时秒数
    start_time = time.time()

    for _ in range(timeout):
        if _global_state.tokenizer_manager.initial_weights_loaded:  # 检查权重是否已加载
            logger.info(
                f"Weights are ready after {time.time() - start_time:.2f} seconds"
            )
            return
        time.sleep(1)  # 每秒轮询一次

    # 超时仍未就绪，记录错误日志
    logger.error(
        f"Weights are not ready after waiting {timeout} seconds. "
        f"Consider increasing SGLANG_WAIT_WEIGHTS_READY_TIMEOUT environment variable. "
        f"Current status: initial_weights_loaded={_global_state.tokenizer_manager.initial_weights_loaded}"
    )


# -------- 关闭主进程 ZMQ 套接字（Granian 模式专用） --------
def _close_main_process_sockets():
    """Close the main process's ZMQ sockets before spawning Granian workers.

    Granian workers create their own TokenizerManager with fresh ZMQ sockets.
    The main process must release its sockets first to avoid binding conflicts
    on the same IPC addresses.
    """
    if _global_state is None or _global_state.tokenizer_manager is None:
        return  # 若全局状态未初始化则跳过
    tm = _global_state.tokenizer_manager
    # 遍历需要关闭的套接字属性名
    for attr in ("recv_from_detokenizer", "send_to_scheduler"):
        sock = getattr(tm, attr, None)
        if sock is None:
            continue
        inner = getattr(sock, "socket", None)   # 尝试获取内部套接字
        if inner is not None:
            inner.close()                        # 关闭内部 ZMQ 套接字
        elif hasattr(sock, "close"):
            sock.close()                         # 直接关闭套接字
        setattr(tm, attr, None)                  # 清空属性引用，防止重复关闭


# -------- 启动 Granian HTTP/2 服务器 --------
def _run_granian_server(server_args: ServerArgs):
    """Launch Granian with HTTP/2 support"""
    from granian import Granian                                        # Granian ASGI 服务器
    from granian.constants import HTTPModes, Interfaces, Loops         # Granian 配置常量

    granian_kwargs = dict(
        target="sglang.srt.entrypoints.http_server:app",               # ASGI 应用路径
        address=server_args.host,                                       # 监听地址
        port=server_args.port,                                          # 监听端口
        interface=Interfaces.ASGI,                                      # 使用 ASGI 接口
        http=HTTPModes.auto,                                            # 自动选择 HTTP 版本（支持 HTTP/2）
        loop=Loops.uvloop,                                              # 使用 uvloop 事件循环
        log_level=server_args.log_level_http or server_args.log_level or "info",  # 日志级别
        workers=1,                                                      # worker 进程数
    )

    ssl_enabled = server_args.ssl_certfile and server_args.ssl_keyfile  # 判断是否启用 SSL
    if ssl_enabled:
        granian_kwargs["ssl_cert"] = server_args.ssl_certfile  # SSL 证书路径
        granian_kwargs["ssl_key"] = server_args.ssl_keyfile    # SSL 私钥路径

    server = Granian(**granian_kwargs)  # 创建 Granian 服务器实例
    server.serve()                      # 启动服务器（阻塞）


# -------- HTTP 服务器配置与启动 --------
# 设置全局状态、配置中间件、启动 uvicorn（由 launch_server 调用）
def _setup_and_run_http_server(
    server_args: ServerArgs,
    tokenizer_manager,                                        # tokenizer 管理器实例
    template_manager,                                         # 模板管理器实例
    port_args: PortArgs,                                      # 端口参数
    scheduler_infos: List[Dict],                              # scheduler 初始化信息列表
    subprocess_watchdog: Optional[SubprocessWatchdog],        # 子进程守护器（可选）
    execute_warmup_func: Callable = _execute_server_warmup,   # warmup 执行函数
    launch_callback: Optional[Callable[[], None]] = None,     # 启动完成回调
):
    """Set up global state, configure middleware, and run uvicorn.

    Called by launch_server after subprocesses have been launched.
    """
    # 将全局状态写入单例
    set_global_state(
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            template_manager=template_manager,
            scheduler_info=scheduler_infos[0],  # 使用第一个 scheduler 的信息
        )
    )

    # 将子进程守护器存储到 tokenizer_manager（SIGQUIT 处理器的唯一数据来源）
    if tokenizer_manager is not None:
        tokenizer_manager._subprocess_watchdog = subprocess_watchdog

    if server_args.enable_metrics:
        add_prometheus_track_response_middleware(app)  # 注册 Prometheus 响应追踪中间件

    # -------- HTTP/2 模式（Granian） --------
    if server_args.enable_http2:
        # 复用多 tokenizer 共享内存机制传递初始化参数给 Granian worker 进程
        multi_tokenizer_args_shm = write_data_for_multi_tokenizer(
            port_args, server_args, scheduler_infos[0]
        )
        try:
            if server_args.ssl_certfile:
                logger.info(
                    f"SSL enabled: certfile={server_args.ssl_certfile}, "
                    f"keyfile={server_args.ssl_keyfile}"
                )
            logger.info(
                f"Starting Granian HTTP/2 server on "
                f"{server_args.host}:{server_args.port}"
            )
            # 通过环境变量将主进程 PID 传递给 Granian worker，让其找到共享内存
            envs.SGLANG_GRANIAN_PARENT_PID.set(os.getpid())
            _close_main_process_sockets()   # 关闭主进程套接字，避免与 worker 冲突
            _run_granian_server(server_args)
        finally:
            if multi_tokenizer_args_shm is not None:
                multi_tokenizer_args_shm.unlink()  # 清理共享内存
        return

    # -------- 单/多 tokenizer uvicorn 模式 --------
    # 将额外参数通过 app 属性传递给 lifespan 函数进行初始化
    if server_args.tokenizer_worker_num == 1:
        # 单 tokenizer 模式：通过 app 属性直接传递参数
        app.is_single_tokenizer_mode = True
        app.server_args = server_args
        app.warmup_thread_kwargs = dict(
            server_args=server_args,
            launch_callback=launch_callback,
            execute_warmup_func=execute_warmup_func,
        )

        # 添加 API 密钥鉴权中间件（仅单 tokenizer 模式支持）
        #
        # 向后兼容说明：
        # - 仅 api_key：行为与旧版一致（所有端点需要 api_key）
        # - 无密钥：旧版无限制；但 ADMIN_FORCE 端点在无 admin_api_key 时仍应拒绝
        if (
            server_args.api_key
            or server_args.admin_api_key
            or app_has_admin_force_endpoints(app)
        ):
            from sglang.srt.utils.auth import add_api_key_middleware

            add_api_key_middleware(
                app,
                api_key=server_args.api_key,               # 普通 API 密钥
                admin_api_key=server_args.admin_api_key,   # 管理员 API 密钥
            )
    else:
        # 多 tokenizer 模式：将参数写入共享内存供其他 worker 读取
        app.is_single_tokenizer_mode = False
        multi_tokenizer_args_shm = write_data_for_multi_tokenizer(
            port_args, server_args, scheduler_infos[0]
        )

    try:
        # 配置 uvicorn 日志格式
        set_uvicorn_logging_configs(server_args)

        if server_args.ssl_certfile:
            logger.info(
                f"SSL enabled: certfile={server_args.ssl_certfile}, "
                f"keyfile={server_args.ssl_keyfile}"
            )

        # 启动 HTTP 服务器（阻塞）
        if server_args.tokenizer_worker_num == 1:
            if server_args.enable_ssl_refresh:
                # SSL 自动刷新模式：使用 Config/Server API 获取 SSLContext 句柄
                config = uvicorn.Config(
                    app,
                    host=server_args.host,
                    port=server_args.port,
                    root_path=server_args.fastapi_root_path,
                    log_level=server_args.log_level_http or server_args.log_level,
                    timeout_keep_alive=envs.SGLANG_TIMEOUT_KEEP_ALIVE.get(),
                    loop="uvloop",
                    ssl_keyfile=server_args.ssl_keyfile,
                    ssl_certfile=server_args.ssl_certfile,
                    ssl_ca_certs=server_args.ssl_ca_certs,
                    ssl_keyfile_password=server_args.ssl_keyfile_password,
                )
                config.load()  # 创建 SSLContext 实例

                from sglang.srt.entrypoints.ssl_utils import SSLCertRefresher

                server = uvicorn.Server(config)

                # 异步协程：同时运行 uvicorn 服务器和 SSL 证书刷新器
                async def _run_with_ssl_refresh():
                    refresher = SSLCertRefresher(
                        config.ssl,
                        server_args.ssl_keyfile,
                        server_args.ssl_certfile,
                        server_args.ssl_ca_certs,
                    )
                    logger.info("SSL certificate auto-refresh enabled.")
                    try:
                        await server.serve()  # 启动 uvicorn 服务器
                    finally:
                        refresher.stop()       # 服务器停止时关闭刷新器

                import asyncio

                asyncio.run(_run_with_ssl_refresh())
            else:
                # 默认单 tokenizer 模式：直接运行 uvicorn
                uvicorn.run(
                    app,
                    host=server_args.host,
                    port=server_args.port,
                    root_path=server_args.fastapi_root_path,
                    log_level=server_args.log_level_http or server_args.log_level,
                    timeout_keep_alive=envs.SGLANG_TIMEOUT_KEEP_ALIVE.get(),
                    loop="uvloop",
                    ssl_keyfile=server_args.ssl_keyfile,
                    ssl_certfile=server_args.ssl_certfile,
                    ssl_ca_certs=server_args.ssl_ca_certs,
                    ssl_keyfile_password=server_args.ssl_keyfile_password,
                )
        else:
            # 多 tokenizer/多 HTTP worker 模式
            from uvicorn.config import LOGGING_CONFIG

            # 为本模块注册独立的日志处理器
            LOGGING_CONFIG["loggers"]["sglang.srt.entrypoints.http_server"] = {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False,
            }
            monkey_patch_uvicorn_multiprocessing()  # 打补丁以支持多进程模式

            if server_args.enable_ssl_refresh:
                logger.warning(
                    "--enable-ssl-refresh is not supported with multiple "
                    "tokenizer workers (--tokenizer-worker-num > 1). "
                    "SSL refresh will be disabled."
                )

            # 多进程 uvicorn（通过字符串指定 app 路径）
            uvicorn.run(
                "sglang.srt.entrypoints.http_server:app",
                host=server_args.host,
                port=server_args.port,
                root_path=server_args.fastapi_root_path,
                log_level=server_args.log_level_http or server_args.log_level,
                timeout_keep_alive=envs.SGLANG_TIMEOUT_KEEP_ALIVE.get(),
                loop="uvloop",
                workers=server_args.tokenizer_worker_num,  # 启动多个 worker 进程
                ssl_keyfile=server_args.ssl_keyfile,
                ssl_certfile=server_args.ssl_certfile,
                ssl_ca_certs=server_args.ssl_ca_certs,
                ssl_keyfile_password=server_args.ssl_keyfile_password,
            )
    finally:
        if server_args.tokenizer_worker_num > 1:
            if multi_tokenizer_args_shm is not None:
                multi_tokenizer_args_shm.unlink()             # 清理共享内存
            if _global_state is not None:
                _global_state.tokenizer_manager.socket_mapping.clear_all_sockets()  # 清理所有 ZMQ 套接字


# -------- 服务器启动入口函数 --------
def launch_server(
    server_args: ServerArgs,
    init_tokenizer_manager_func: Callable = init_tokenizer_manager,           # 初始化 tokenizer 管理器函数（可替换）
    run_scheduler_process_func: Callable = run_scheduler_process,             # 启动 scheduler 子进程函数（可替换）
    run_detokenizer_process_func: Callable = run_detokenizer_process,         # 启动 detokenizer 子进程函数（可替换）
    execute_warmup_func: Callable = _execute_server_warmup,                   # warmup 执行函数（可替换）
    launch_callback: Optional[Callable[[], None]] = None,                     # 服务就绪回调（可选）
):
    """
    Launch SRT (SGLang Runtime) Server.

    The SRT server consists of an HTTP server and an SRT engine.

    - HTTP server: A FastAPI server that routes requests to the engine.
    - The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.

    Note:
    1. The HTTP server, Engine, and TokenizerManager all run in the main process.
    2. Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.
    """
    # 启动引擎所有子进程（scheduler、detokenizer），并获取初始化结果
    (
        tokenizer_manager,
        template_manager,
        port_args,
        scheduler_init_result,
        subprocess_watchdog,
    ) = Engine._launch_subprocesses(
        server_args=server_args,
        init_tokenizer_manager_func=init_tokenizer_manager_func,
        run_scheduler_process_func=run_scheduler_process_func,
        run_detokenizer_process_func=run_detokenizer_process_func,
    )

    # 配置全局状态并启动 HTTP 服务器（阻塞直至服务器关闭）
    _setup_and_run_http_server(
        server_args,
        tokenizer_manager,
        template_manager,
        port_args,
        scheduler_init_result.scheduler_infos,  # 传入所有 scheduler 分片的初始化信息
        subprocess_watchdog,
        execute_warmup_func=execute_warmup_func,
        launch_callback=launch_callback,
    )
