# 导入后端基类
from sglang.lang.backend.base_backend import BaseBackend
# 导入聊天模板获取函数
from sglang.lang.chat_template import get_chat_template
# 导入流式执行器
from sglang.lang.interpreter import StreamExecutor
# 导入采样参数 IR 节点
from sglang.lang.ir import SglSamplingParams

# 尝试导入 anthropic SDK，若未安装则将异常保存供运行时抛出
try:
    import anthropic
except ImportError as e:
    anthropic = e


# Anthropic Claude 系列模型的后端实现
class Anthropic(BaseBackend):
    def __init__(self, model_name, *args, **kwargs):
        super().__init__()

        # 若导入失败则在此时抛出 ImportError
        if isinstance(anthropic, Exception):
            raise anthropic

        # 保存模型名称
        self.model_name = model_name
        # Claude 使用专属聊天模板（system 消息格式与 OpenAI 不同）
        self.chat_template = get_chat_template("claude")
        # 创建 Anthropic 官方客户端实例
        self.client = anthropic.Anthropic(*args, **kwargs)

    def get_chat_template(self):
        # 返回 Claude 聊天模板
        return self.chat_template

    def generate(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        # 优先使用结构化 messages 列表，否则将纯文本包装为 user 消息
        if s.messages_:
            messages = s.messages_
        else:
            messages = [{"role": "user", "content": s.text_}]

        # Anthropic API 要求 system 消息作为独立字段传递，从消息列表中提取
        if messages and messages[0]["role"] == "system":
            system = messages.pop(0)["content"]
        else:
            system = ""

        # 调用 Anthropic messages.create 接口发起同步请求
        ret = self.client.messages.create(
            model=self.model_name,
            system=system,
            messages=messages,
            # 将 SglSamplingParams 转换为 Anthropic API 所需参数格式
            **sampling_params.to_anthropic_kwargs(),
        )
        # 取第一个 content block 的文本作为生成结果
        comp = ret.content[0].text

        # 返回 (生成文本, 元信息字典)
        return comp, {}

    def generate_stream(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        # 同 generate：优先使用结构化 messages，否则包装为 user 消息
        if s.messages_:
            messages = s.messages_
        else:
            messages = [{"role": "user", "content": s.text_}]

        # 同样提取 system 消息作为独立字段
        if messages and messages[0]["role"] == "system":
            system = messages.pop(0)["content"]
        else:
            system = ""

        # 使用 Anthropic streaming 上下文管理器发起流式请求
        with self.client.messages.stream(
            model=self.model_name,
            system=system,
            messages=messages,
            **sampling_params.to_anthropic_kwargs(),
        ) as stream:
            # 逐 token 产出文本片段，元信息字典为空
            for text in stream.text_stream:
                yield text, {}
