# 导入标准库
import atexit
import json
import multiprocessing
import time
import warnings
from typing import Dict, List, Optional, Union

# 异步 HTTP 客户端（用于 async_generate）
import aiohttp
# 同步 HTTP 客户端（用于 generate）
import requests

# SGLang 全局配置（skip_special_tokens 等输出控制选项）
from sglang.global_config import global_config
# 导入后端基类
from sglang.lang.backend.base_backend import BaseBackend
# 导入聊天模板工厂函数
from sglang.lang.chat_template import get_chat_template, get_chat_template_by_model_path
# 导入选择决策和采样方法
from sglang.lang.choices import ChoicesDecision, ChoicesSamplingMethod
# 导入流式执行器
from sglang.lang.interpreter import StreamExecutor
# 导入正则约束常量和采样参数 IR 节点
from sglang.lang.ir import (
    REGEX_BOOL,    # 匹配 true/false 的正则
    REGEX_FLOAT,   # 匹配浮点数的正则
    REGEX_INT,     # 匹配整数的正则
    REGEX_STR,     # 匹配字符串的正则
    SglSamplingParams,
)
# 工具函数：带认证的 HTTP 请求封装
from sglang.utils import http_request



# RuntimeEndpoint：连接本地或远程 SGLang Runtime HTTP 服务的后端实现
class RuntimeEndpoint(BaseBackend):
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        verify: Optional[str] = None,
        chat_template_name: Optional[str] = None,
    ):
        super().__init__()
        # Runtime 后端支持 KV cache concatenate_and_append 操作
        self.support_concate_and_append = True

        # 保存服务器基础 URL、API Key 和 SSL 验证参数
        self.base_url = base_url
        self.api_key = api_key
        self.verify = verify

        # 启动时查询服务器模型信息（包括模型路径、最大序列长度等）
        res = http_request(
            self.base_url + "/get_model_info",
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)
        self.model_info = res.json()

        # 根据参数或模型路径自动选择聊天模板
        if chat_template_name:
            self.chat_template = get_chat_template(chat_template_name)
        else:
            self.chat_template = get_chat_template_by_model_path(
                self.model_info["model_path"]
            )

    def get_model_name(self):
        # 返回服务器加载的模型路径
        return self.model_info["model_path"]

    def flush_cache(self):
        # 发送 POST 请求清空服务器端 KV cache
        res = http_request(
            self.base_url + "/flush_cache",
            api_key=self.api_key,
            verify=self.verify,
            method="POST",
        )
        self._assert_success(res)

    def get_server_info(self):
        # 查询服务器运行时信息（内存用量、队列状态等）
        res = http_request(
            self.base_url + "/server_info",
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)
        return res.json()

    def get_chat_template(self):
        # 返回当前后端使用的聊天模板
        return self.chat_template

    def cache_prefix(self, prefix_str: str):
        # 通过发送 max_new_tokens=0 的请求将前缀字符串填入 KV cache
        res = http_request(
            self.base_url + "/generate",
            json={"text": prefix_str, "sampling_params": {"max_new_tokens": 0}},
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)

    def start_profile(self):
        # 启动服务器端性能分析（profiling）
        res = http_request(
            self.base_url + "/start_profile",
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)

    def stop_profile(self):
        # 停止服务器端性能分析
        res = http_request(
            self.base_url + "/stop_profile",
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)

    def commit_lazy_operations(self, s: StreamExecutor):
        # 提交延迟操作：发送 max_new_tokens=0 请求将当前文本预填充到 KV cache
        data = {"text": s.text_, "sampling_params": {"max_new_tokens": 0}}
        self._add_images(s, data)
        res = http_request(
            self.base_url + "/generate",
            json=data,
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)

    def fill_image(self, s: StreamExecutor):
        # 将图像数据预填充到服务器端 KV cache（多模态场景）
        data = {"text": s.text_, "sampling_params": {"max_new_tokens": 0}}
        self._add_images(s, data)
        res = http_request(
            self.base_url + "/generate",
            json=data,
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)

    def _handle_dtype_to_regex(self, sampling_params: SglSamplingParams):
        # 将 dtype 约束转换为对应的正则表达式（由 Runtime 服务器执行约束生成）
        if sampling_params.dtype is None:
            return

        # 确保 stop 参数为列表格式以便追加
        if sampling_params.stop == ():
            sampling_params.stop = []

        dtype_regex = None
        if sampling_params.dtype in ["int", int]:
            # 整数约束：使用整数正则，并添加空格/换行作为停止符
            dtype_regex = REGEX_INT
            sampling_params.stop.extend([" ", "\n"])
        elif sampling_params.dtype in ["float", float]:
            # 浮点数约束：使用浮点正则，添加停止符
            dtype_regex = REGEX_FLOAT
            sampling_params.stop.extend([" ", "\n"])
        elif sampling_params.dtype in ["str", str]:
            # 字符串约束：使用字符串正则
            dtype_regex = REGEX_STR
        elif sampling_params.dtype in ["bool", bool]:
            # 布尔值约束：使用布尔正则
            dtype_regex = REGEX_BOOL
        else:
            raise RuntimeError(f"Invalid dtype: {sampling_params.dtype}")

        # 若 dtype 和 regex 同时设置，发出警告并以 dtype 优先
        if dtype_regex is not None and sampling_params.regex is not None:
            warnings.warn(
                f"Both dtype and regex are set. Only dtype will be used. dtype: {sampling_params.dtype}, regex: {sampling_params.regex}"
            )

        # 将 dtype 转换结果写入 regex 字段
        sampling_params.regex = dtype_regex

    def generate(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        # 将 dtype 约束转换为正则（若有）
        self._handle_dtype_to_regex(sampling_params)
        # 构建请求体，合并全局输出配置和采样参数
        data = {
            "text": s.text_,
            "sampling_params": {
                "skip_special_tokens": global_config.skip_special_tokens_in_output,
                "spaces_between_special_tokens": global_config.spaces_between_special_tokens_in_out,
                # 将 SglSamplingParams 转换为 SRT 服务器支持的参数格式
                **sampling_params.to_srt_kwargs(),
            },
        }

        # 若设置了 logprob 相关参数，追加到顶层请求体
        for item in [
            "return_logprob",
            "logprob_start_len",
            "top_logprobs_num",
            "return_text_in_logprobs",
        ]:
            value = getattr(sampling_params, item, None)
            if value is not None:
                data[item] = value

        # 若有图像数据，添加到请求体
        self._add_images(s, data)

        # 发起同步 HTTP 请求
        res = http_request(
            self.base_url + "/generate",
            json=data,
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)

        # 解析响应，返回 (生成文本, 元信息)
        obj = res.json()
        comp = obj["text"]
        return comp, obj["meta_info"]

    def generate_stream(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        # 同 generate：先处理 dtype 约束
        self._handle_dtype_to_regex(sampling_params)

        # 构建流式请求体
        data = {
            "text": s.text_,
            "sampling_params": {
                "skip_special_tokens": global_config.skip_special_tokens_in_output,
                "spaces_between_special_tokens": global_config.spaces_between_special_tokens_in_out,
                **sampling_params.to_srt_kwargs(),
            },
        }

        # 追加 logprob 相关参数
        for item in [
            "return_logprob",
            "logprob_start_len",
            "top_logprobs_num",
            "return_text_in_logprobs",
        ]:
            value = getattr(sampling_params, item, None)
            if value is not None:
                data[item] = value

        # 开启流式传输
        data["stream"] = True
        self._add_images(s, data)

        # 发起流式 HTTP 请求
        res = http_request(
            self.base_url + "/generate",
            json=data,
            stream=True,
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)
        # pos 跟踪累计文本长度，用于提取每个 chunk 的增量文本
        pos = 0

        # 解析 SSE（Server-Sent Events）格式响应流
        for chunk in res.iter_lines(decode_unicode=False):
            chunk = chunk.decode("utf-8")
            if chunk and chunk.startswith("data:"):
                # 流结束标志
                if chunk == "data: [DONE]":
                    break
                data = json.loads(chunk[5:].strip("\n"))
                # 提取增量文本（当前 chunk 相对于上一个位置的新增部分）
                chunk_text = data["text"][pos:]
                meta_info = data["meta_info"]
                pos += len(chunk_text)
                yield chunk_text, meta_info

    def select(
        self,
        s: StreamExecutor,
        choices: List[str],
        temperature: float,
        choices_method: ChoicesSamplingMethod,
    ) -> ChoicesDecision:
        # select 仅支持贪婪解码（temperature=0）
        assert temperature <= 1e-5

        # Cache common prefix
        # 第一步：发送公共前缀预填充请求，获取 prompt token 数量（用于 token healing）
        data = {"text": s.text_, "sampling_params": {"max_new_tokens": 0}}
        obj = self._generate_http_request(s, data)
        prompt_len = obj["meta_info"]["prompt_tokens"]
        # 设置 logprob_start_len（往前留 2 个 token 以应对 token healing）
        logprob_start_len = max(prompt_len - 2, 0)  # For token healing

        # Compute logprob
        # 第二步：对所有候选字符串批量计算 input logprob
        data = {
            "text": [s.text_ + c for c in choices],
            "sampling_params": {
                "max_new_tokens": 0,
                "temperature": 0,
            },
            "return_logprob": True,
            "return_text_in_logprobs": True,
            "logprob_start_len": logprob_start_len,
        }
        obj = self._generate_http_request(s, data)

        # 从各候选结果中提取 input/output token logprobs
        input_token_logprobs = [r["meta_info"]["input_token_logprobs"] for r in obj]
        output_token_logprobs = [r["meta_info"]["output_token_logprobs"] for r in obj]
        # 计算每个候选的归一化 prompt logprob（用于排序比较）
        normalized_prompt_logprobs = [
            compute_normalized_prompt_logprobs(r["meta_info"]["input_token_logprobs"])
            for r in obj
        ]

        # Remove extra token if no token healing occurred
        # 若发生 token healing，需要去除多余的 token（还原到候选真正的起始位置）
        for i in range(len(input_token_logprobs)):
            healed_token_str = input_token_logprobs[i][0][-1]
            if s.text_.endswith(healed_token_str):
                healed_token_logprob = input_token_logprobs[i][0][0]
                # 从归一化 logprob 中减去 healed token 的贡献
                normalized_prompt_logprobs[i] = (
                    normalized_prompt_logprobs[i] * len(input_token_logprobs[i])
                    - healed_token_logprob
                ) / (len(input_token_logprobs[i]) - 1)
                input_token_logprobs[i] = input_token_logprobs[i][1:]

        # Compute unconditional logprobs if required
        # 若选择方法需要无条件 logprob（如 PMI 方法），额外计算
        if choices_method.requires_unconditional_logprobs:
            input_ids = [[el[1] for el in subl] for subl in input_token_logprobs]
            data = {
                "input_ids": input_ids,
                "sampling_params": {"max_new_tokens": 0},
                "return_logprob": True,
            }
            obj = self._generate_http_request(s, data)
            unconditional_token_logprobs = [
                r["meta_info"]["input_token_logprobs"] for r in obj
            ]
        else:
            unconditional_token_logprobs = None

        # 调用选择方法（如 token_length_normalized / greedy / pmi 等）计算最终决策
        return choices_method(
            choices=choices,
            normalized_prompt_logprobs=normalized_prompt_logprobs,
            input_token_logprobs=input_token_logprobs,
            output_token_logprobs=output_token_logprobs,
            unconditional_token_logprobs=unconditional_token_logprobs,
        )

    def concatenate_and_append(self, src_rids: List[str], dst_rid: str):
        # 将多个源请求的 KV cache 拼接并追加到目标请求（用于 fork/join 并行模式）
        res = http_request(
            self.base_url + "/concate_and_append_request",
            json={"src_rids": src_rids, "dst_rid": dst_rid},
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)

    def _generate_http_request(self, s: StreamExecutor, data):
        # 内部辅助方法：添加图像数据后发起 generate 请求并返回解析后的 JSON
        self._add_images(s, data)
        res = http_request(
            self.base_url + "/generate",
            json=data,
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)
        return res.json()

    def _add_images(self, s: StreamExecutor, data):
        # 若执行器中包含图像，将第一张图像的 base64 数据添加到请求体
        if s.images_:
            assert len(s.images_) == 1, "Only support one image."
            data["image_data"] = s.images_[0][1]

    def _assert_success(self, res):
        # 检查 HTTP 响应状态码，非 200 时提取错误信息并抛出异常
        if res.status_code != 200:
            try:
                content = res.json()
            except json.JSONDecodeError:
                content = res.text
            raise RuntimeError(content)


def compute_normalized_prompt_logprobs(input_logprobs):
    # 计算 input token logprob 的均值（归一化以消除序列长度偏差）
    values = [x[0] for x in input_logprobs if x[0]]
    return sum(values) / len(values)


class Runtime:
    """
    A wrapper for the HTTP server.
    This is used for launching the server in a python program without
    using the command line interface.

    It is mainly used for the frontend language.
    You should use the Engine class if you want to do normal offline processing without the frontend language.
    """

    def __init__(
        self,
        log_level: str = "error",
        launch_timeout: float = 300.0,
        *args,
        **kwargs,
    ):
        """See the arguments in server_args.py::ServerArgs

        Args:
            log_level: Log level for the server.
            timeout: Timeout in seconds for waiting for the server to start.
            *args: Additional arguments passed to ServerArgs.
            **kwargs: Additional keyword arguments passed to ServerArgs.
        """
        # We delay the import of any `sglang.srt` components in `sglang.lang`, so users can run
        # client code without installing SRT server and its dependency if they want.
        # 延迟导入 SRT 服务器组件，避免仅使用客户端时引入不必要的依赖
        from sglang.srt.entrypoints.http_server import launch_server
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.utils.network import is_port_available

        # 构建服务器参数对象
        self.server_args = ServerArgs(*args, log_level=log_level, **kwargs)

        # Pre-allocate ports
        # 从指定端口开始向上查找第一个可用端口
        for port in range(self.server_args.port, 40000):
            if is_port_available(port):
                break
        self.server_args.port = port

        # 构建服务器 URL 和生成端点 URL
        self.url = self.server_args.url()
        self.generate_url = self.url + "/generate"

        # NOTE: We store pid instead of proc to fix some issues during __delete__
        # 存储子进程 PID（而非 proc 对象）以避免 __del__ 时的问题
        self.pid = None

        # 使用 "spawn" 方式创建子进程，避免 fork 带来的 CUDA 上下文问题
        ctx = multiprocessing.get_context("spawn")
        proc = ctx.Process(
            target=launch_server,
            args=(self.server_args,),
        )
        proc.start()
        self.pid = proc.pid

        # Before python program terminates, call shutdown implicitly. Therefore, users don't have to explicitly call .shutdown()
        # 注册 atexit 回调，确保 Python 退出时自动关闭服务器进程
        atexit.register(self.shutdown)

        # Wait for server to be ready by polling /health_generate
        # 轮询 /health_generate 等待服务器就绪
        start_time = time.time()
        with requests.Session() as session:
            while time.time() - start_time < launch_timeout:
                try:
                    response = session.get(f"{self.url}/health_generate")
                    if response.status_code == 200:
                        break
                except requests.RequestException:
                    pass

                # 若子进程已退出则立即终止等待并报错
                if not proc.is_alive():
                    self.shutdown()
                    raise RuntimeError(
                        "Initialization failed. Please see the error messages above."
                    )

                time.sleep(2)
            else:
                # 超过 launch_timeout 仍未就绪则抛出超时错误
                self.shutdown()
                raise TimeoutError("Server failed to start within the timeout period.")

        # 创建 RuntimeEndpoint 实例供 SGLang 前端语言使用
        self.endpoint = RuntimeEndpoint(self.url)

    def shutdown(self):
        # 杀掉服务器进程及其所有子进程
        from sglang.srt.utils import kill_process_tree

        if self.pid is not None:
            kill_process_tree(self.pid)
            self.pid = None

    def start_profile(self):
        # 代理到 endpoint 的 start_profile
        self.endpoint.start_profile()

    def stop_profile(self):
        # 代理到 endpoint 的 stop_profile
        self.endpoint.stop_profile()

    def cache_prefix(self, prefix: str):
        # 代理到 endpoint 的 cache_prefix
        self.endpoint.cache_prefix(prefix)

    def get_tokenizer(self):
        # 加载服务器模型对应的分词器（用于外部 tokenize 操作）
        from sglang.srt.utils.hf_transformers_utils import get_tokenizer

        return get_tokenizer(
            self.server_args.tokenizer_path,
            tokenizer_mode=self.server_args.tokenizer_mode,
            trust_remote_code=self.server_args.trust_remote_code,
            revision=self.server_args.revision,
        )

    async def async_generate(
        self,
        prompt: str,
        sampling_params: Optional[Dict] = None,
    ):
        # 异步流式生成：使用 aiohttp 发起 SSE 流式请求并逐块产出文本
        if self.server_args.skip_tokenizer_init:
            # 跳过分词时直接传 input_ids
            json_data = {
                "input_ids": prompt,
                "sampling_params": sampling_params,
                "stream": True,
            }
        else:
            json_data = {
                "text": prompt,
                "sampling_params": sampling_params,
                "stream": True,
            }
        pos = 0

        # 设置 3 小时超时（适应长序列生成）
        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            async with session.post(self.generate_url, json=json_data) as response:
                async for chunk, _ in response.content.iter_chunks():
                    chunk = chunk.decode("utf-8")
                    if chunk and chunk.startswith("data:"):
                        # 流结束标志
                        if chunk == "data: [DONE]\n\n":
                            break
                        data = json.loads(chunk[5:].strip("\n"))
                        if "text" in data:
                            # 提取增量文本
                            cur = data["text"][pos:]
                            if cur:
                                yield cur
                            pos += len(cur)
                        else:
                            # 返回原始数据（如错误信息）
                            yield data

    # add_request 是 async_generate 的别名，兼容旧接口
    add_request = async_generate

    def generate(
        self,
        prompt: Union[str, List[str]],
        sampling_params: Optional[Dict] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
    ):
        # 同步批量生成：发起 POST 请求并以 JSON 字符串形式返回结果
        json_data = {
            "text": prompt,
            "sampling_params": sampling_params,
            "return_logprob": return_logprob,
            "logprob_start_len": logprob_start_len,
            "top_logprobs_num": top_logprobs_num,
            "lora_path": lora_path,
        }
        # 多 prompt 时校验 lora_path 数量与 prompt 数量一致
        assert not isinstance(lora_path, list) or len(lora_path) == len(prompt)
        response = requests.post(
            self.url + "/generate",
            json=json_data,
        )
        # 返回序列化的 JSON 字符串
        return json.dumps(response.json())

    def encode(
        self,
        prompt: Union[str, List[str], List[Dict], List[List[Dict]]],
    ):
        # 将 prompt 发送到 /encode 端点进行嵌入编码（embedding）
        json_data = {"text": prompt}
        response = requests.post(self.url + "/encode", json=json_data)
        return json.dumps(response.json())

    async def get_server_info(self):
        # 异步查询服务器信息（内存、队列状态等）
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.url}/server_info") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_data = await response.json()
                    raise RuntimeError(
                        f"Failed to get server info. {error_data['error']['message']}"
                    )

    def __del__(self):
        # 对象被垃圾回收时自动关闭服务器进程
        self.shutdown()
