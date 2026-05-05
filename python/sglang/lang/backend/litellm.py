# 导入类型注解
from typing import Mapping, Optional

# 导入后端基类
from sglang.lang.backend.base_backend import BaseBackend
# 通过模型路径自动推断聊天模板
from sglang.lang.chat_template import get_chat_template_by_model_path
# 导入流式执行器
from sglang.lang.interpreter import StreamExecutor
# 导入采样参数 IR 节点
from sglang.lang.ir import SglSamplingParams

# 尝试导入 litellm 统一调用库；若未安装则捕获异常，并预设重试次数
try:
    import litellm
except ImportError as e:
    litellm = e
    litellm.num_retries = 1


# LiteLLM 后端：通过 litellm 统一接口调用多种 LLM 提供商（OpenAI/Anthropic/Cohere 等）
class LiteLLM(BaseBackend):
    def __init__(
        self,
        model_name,
        chat_template=None,
        api_key=None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = 600,
        # 默认使用 litellm 全局配置的重试次数
        max_retries: Optional[int] = litellm.num_retries,
        default_headers: Optional[Mapping[str, str]] = None,
    ):
        super().__init__()

        # 若导入失败则在此时抛出 ImportError
        if isinstance(litellm, Exception):
            raise litellm

        # 保存模型名称（litellm 格式，如 "openai/gpt-4o"）
        self.model_name = model_name

        # 使用传入模板或根据模型名称自动推断聊天模板
        self.chat_template = chat_template or get_chat_template_by_model_path(
            model_name
        )

        # 将连接参数统一存储，供 generate/generate_stream 调用时展开
        self.client_params = {
            "api_key": api_key,
            "organization": organization,
            "base_url": base_url,
            "timeout": timeout,
            "max_retries": max_retries,
            "default_headers": default_headers,
        }

    def get_chat_template(self):
        # 返回当前后端使用的聊天模板
        return self.chat_template

    def generate(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        # 优先使用结构化 messages，否则将纯文本包装为 user 消息
        if s.messages_:
            messages = s.messages_
        else:
            messages = [{"role": "user", "content": s.text_}]

        # 调用 litellm.completion 发起同步请求，合并连接参数和采样参数
        ret = litellm.completion(
            model=self.model_name,
            messages=messages,
            **self.client_params,
            # 将 SglSamplingParams 转换为 litellm 兼容参数
            **sampling_params.to_litellm_kwargs(),
        )
        # 取第一个 choice 的 message content 作为生成结果
        comp = ret.choices[0].message.content

        # 返回 (生成文本, 元信息字典)
        return comp, {}

    def generate_stream(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        # 同 generate：优先使用结构化 messages
        if s.messages_:
            messages = s.messages_
        else:
            messages = [{"role": "user", "content": s.text_}]

        # 开启 stream=True 发起流式请求
        ret = litellm.completion(
            model=self.model_name,
            messages=messages,
            stream=True,
            **self.client_params,
            **sampling_params.to_litellm_kwargs(),
        )
        # 遍历流式 chunk，提取 delta.content 逐 token 产出
        for chunk in ret:
            text = chunk.choices[0].delta.content
            # 过滤 None（流结束时 delta.content 为 None）
            if text is not None:
                yield text, {}
