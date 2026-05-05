# 导入标准库
import os
import warnings

# 导入后端基类
from sglang.lang.backend.base_backend import BaseBackend
# 导入聊天模板
from sglang.lang.chat_template import get_chat_template
# 导入流式执行器
from sglang.lang.interpreter import StreamExecutor
# 导入采样参数 IR 节点
from sglang.lang.ir import SglSamplingParams

# 尝试导入 Google Cloud Vertex AI SDK；若未安装则将异常保存到 GenerativeModel
try:
    import vertexai
    from vertexai.preview.generative_models import (
        GenerationConfig,   # 生成参数配置类
        GenerativeModel,    # Gemini 等生成式模型类
        Image,              # 图像数据类（用于多模态输入）
    )
except ImportError as e:
    GenerativeModel = e


# Google Cloud Vertex AI 后端，支持 Gemini 系列多模态模型
class VertexAI(BaseBackend):
    def __init__(self, model_name, safety_settings=None):
        super().__init__()

        # 若导入失败则在此时抛出 ImportError
        if isinstance(GenerativeModel, Exception):
            raise GenerativeModel

        # 从环境变量读取 GCP 项目 ID（必填）
        project_id = os.environ["GCP_PROJECT_ID"]
        # 从环境变量读取 GCP 区域（可选，默认为 Vertex AI 默认区域）
        location = os.environ.get("GCP_LOCATION")
        # 初始化 Vertex AI 客户端
        vertexai.init(project=project_id, location=location)

        # 保存模型名称（如 "gemini-1.0-pro-vision"）
        self.model_name = model_name
        # 使用默认聊天模板（Vertex AI 消息格式在内部转换）
        self.chat_template = get_chat_template("default")
        # 安全设置（可选，用于控制有害内容过滤级别）
        self.safety_settings = safety_settings

    def get_chat_template(self):
        # 返回当前使用的聊天模板
        return self.chat_template

    def generate(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        # 多轮对话模式：将 messages 转换为 Vertex AI 格式
        if s.messages_:
            prompt = self.messages_to_vertexai_input(s.messages_)
        else:
            # single-turn
            # 单轮模式：若有图像则构建多模态输入，否则直接使用文本
            prompt = (
                self.text_to_vertexai_input(s.text_, s.cur_images)
                if s.cur_images
                else s.text_
            )
        # 创建模型实例并发起同步生成请求
        ret = GenerativeModel(self.model_name).generate_content(
            prompt,
            # 将 SglSamplingParams 转换为 Vertex AI GenerationConfig
            generation_config=GenerationConfig(**sampling_params.to_vertexai_kwargs()),
            safety_settings=self.safety_settings,
        )

        # 取生成结果文本
        comp = ret.text

        # 返回 (生成文本, 元信息字典)
        return comp, {}

    def generate_stream(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        # 同 generate：优先处理多轮消息，否则构建单轮输入
        if s.messages_:
            prompt = self.messages_to_vertexai_input(s.messages_)
        else:
            # single-turn
            prompt = (
                self.text_to_vertexai_input(s.text_, s.cur_images)
                if s.cur_images
                else s.text_
            )
        # 开启 stream=True 发起流式生成请求，返回生成器
        generator = GenerativeModel(self.model_name).generate_content(
            prompt,
            stream=True,
            generation_config=GenerationConfig(**sampling_params.to_vertexai_kwargs()),
            safety_settings=self.safety_settings,
        )
        # 逐块产出文本片段，元信息字典为空
        for ret in generator:
            yield ret.text, {}

    def text_to_vertexai_input(self, text, images):
        # 构建多模态输入列表（文本 + 图像交替排列）
        input = []
        # split with image token
        # 按图像占位符切分文本，与图像列表交替插入
        text_segs = text.split(self.chat_template.image_token)
        for image_path, image_base64_data in images:
            text_seg = text_segs.pop(0)
            # 非空文本段直接追加
            if text_seg != "":
                input.append(text_seg)
            # 将 base64 数据转为 Vertex AI Image 对象
            input.append(Image.from_bytes(image_base64_data))
        # 追加最后一段文本（图像之后的文本）
        text_seg = text_segs.pop(0)
        if text_seg != "":
            input.append(text_seg)
        return input

    def messages_to_vertexai_input(self, messages):
        # 将 OpenAI 格式的 messages 列表转换为 Vertex AI 格式
        vertexai_message = []
        # from openai message format to vertexai message format
        for msg in messages:
            # 提取消息文本（content 可能是字符串或包含 text/image 的列表）
            if isinstance(msg["content"], str):
                text = msg["content"]
            else:
                # 多模态消息取第一个元素的文本
                text = msg["content"][0]["text"]

            if msg["role"] == "system":
                # Vertex AI 不支持原生 system 消息，转为 user/model 交互模拟
                warnings.warn("Warning: system prompt is not supported in VertexAI.")
                vertexai_message.append(
                    {
                        "role": "user",
                        "parts": [{"text": "System prompt: " + text}],
                    }
                )
                vertexai_message.append(
                    {
                        "role": "model",
                        "parts": [{"text": "Understood."}],
                    }
                )
                continue
            # user 角色直接映射
            if msg["role"] == "user":
                vertexai_msg = {
                    "role": "user",
                    "parts": [{"text": text}],
                }
            elif msg["role"] == "assistant":
                # assistant 在 Vertex AI 中对应 "model" 角色
                vertexai_msg = {
                    "role": "model",
                    "parts": [{"text": text}],
                }

            # images
            # 处理消息中的图像（content 列表第一项后的元素均为图像）
            if isinstance(msg["content"], list) and len(msg["content"]) > 1:
                for image in msg["content"][1:]:
                    assert image["type"] == "image_url"
                    # 提取 base64 数据并以 inline_data 格式添加到 parts
                    vertexai_msg["parts"].append(
                        {
                            "inline_data": {
                                # data URI 格式 "data:image/jpeg;base64,<data>"，取逗号后部分
                                "data": image["image_url"]["url"].split(",")[1],
                                "mime_type": "image/jpeg",
                            }
                        }
                    )

            vertexai_message.append(vertexai_msg)
        return vertexai_message
