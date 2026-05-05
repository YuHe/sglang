# 正则表达式工具（用于模型路径匹配）
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, List, Tuple


# 聊天模板样式枚举：决定角色前/后缀的拼接方式
class ChatTemplateStyle(Enum):
    PLAIN = auto()   # 普通拼接：直接使用 role_prefix_and_suffix 中定义的值
    LLAMA2 = auto()  # LLaMA-2 特殊格式：第一条 user 消息需要特殊处理（system 嵌入其中）


# 聊天模板数据类：描述模型对话格式的完整配置
@dataclass
class ChatTemplate:
    name: str                                           # 模板名称（注册键）
    default_system_prompt: str                          # 默认 system 提示（None 表示无默认）
    role_prefix_and_suffix: Dict[str, Tuple[str, str]]  # 各角色的（前缀, 后缀）映射
    stop_str: List[str] = ()                            # 生成终止字符串列表
    image_token: str = "<image>"                        # 图像占位符 token
    audio_token: str = "<audio>"                        # 音频占位符 token
    style: ChatTemplateStyle = ChatTemplateStyle.PLAIN  # 模板拼接样式

    def get_prefix_and_suffix(
        self, role: str, hist_messages: List[Dict]
    ) -> Tuple[str, str]:
        # 根据角色和历史消息获取该角色的前缀和后缀
        prefix, suffix = self.role_prefix_and_suffix.get(role, ("", ""))

        if self.style == ChatTemplateStyle.LLAMA2:
            # LLaMA-2 特殊处理：第一条 system 消息需要嵌套在第一个 [INST] 中
            if role == "system" and not hist_messages:
                user_prefix, _ = self.role_prefix_and_suffix.get("user", ("", ""))
                system_prefix, system_suffix = self.role_prefix_and_suffix.get(
                    "system", ("", "")
                )
                # system 前缀 = user 前缀 + system 前缀（合并到第一个 [INST] 块内）
                return (user_prefix + system_prefix, system_suffix)
            elif (
                role == "user"
                and len(hist_messages) == 1
                and hist_messages[0]["content"] is not None
            ):
                # 第一条 user 消息（紧跟 system 消息后）：去掉重复的 [INST] 前缀
                return ("", suffix)

        return prefix, suffix

    def get_prompt(self, messages: List[Dict]) -> str:
        # 将完整的消息列表渲染为单个字符串 prompt
        prompt = ""
        for i, message in enumerate(messages):
            role, content = message["role"], message["content"]
            # system 角色 content 为 None 时使用默认 system prompt
            if role == "system" and content is None:
                content = self.default_system_prompt
                if content is None:
                    continue

            # 获取该角色在当前位置的前/后缀并拼接
            prefix, suffix = self.get_prefix_and_suffix(role, messages[:i])
            prompt += f"{prefix}{content}{suffix}"
        return prompt


# 全局聊天模板注册表：name → ChatTemplate 映射
chat_template_registry: Dict[str, ChatTemplate] = {}
# 全局模型路径匹配函数注册表：按注册顺序尝试匹配
matching_function_registry: List[Callable] = []


def register_chat_template(template):
    # 向注册表中添加聊天模板（以 name 为键）
    chat_template_registry[template.name] = template


def register_chat_template_matching_function(func):
    # 注册模型路径匹配函数（装饰器用法）
    matching_function_registry.append(func)


def get_chat_template(name):
    # 按名称获取聊天模板（不存在时抛出 KeyError）
    return chat_template_registry[name]


def get_chat_template_by_model_path(model_path):
    # 通过模型路径自动匹配聊天模板（依次尝试所有匹配函数）
    for matching_func in matching_function_registry:
        template_name = matching_func(model_path)
        if template_name is not None:
            return get_chat_template(template_name)
    # 所有匹配函数均未命中时返回默认模板
    return get_chat_template("default")


# ==================== 聊天模板注册 ====================

# 通用默认模板（适用于无特定格式要求的模型）
register_chat_template(
    ChatTemplate(
        name="default",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": ("SYSTEM:", "\n"),
            "user": ("USER:", "\n"),
            "assistant": ("ASSISTANT:", "\n"),
        },
    )
)

# Anthropic Claude 系列对话格式
register_chat_template(
    ChatTemplate(
        name="claude",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": ("", ""),
            "user": ("\n\nHuman: ", ""),
            "assistant": ("\n\nAssistant:", ""),
        },
    )
)

# ChatML 格式（<|im_start|>/<|im_end|>，适用于 Qwen/TinyLlama 等）
register_chat_template(
    ChatTemplate(
        name="chatml",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": ("<|im_start|>system\n", "<|im_end|>\n"),
            "user": ("<|im_start|>user\n", "<|im_end|>\n"),
            "assistant": ("<|im_start|>assistant\n", "<|im_end|>\n"),
        },
        style=ChatTemplateStyle.PLAIN,
        stop_str=("<|im_end|>",),
    )
)

# ChatML + LLaVA（带默认 system prompt 和图像 token 的多模态变体）
register_chat_template(
    ChatTemplate(
        name="chatml-llava",
        default_system_prompt="You are a helpful assistant.",
        role_prefix_and_suffix={
            "system": ("<|im_start|>system\n", "<|im_end|>\n"),
            "user": ("<|im_start|>user\n", "<|im_end|>\n"),
            "assistant": ("<|im_start|>assistant\n", "<|im_end|>\n"),
        },
        style=ChatTemplateStyle.PLAIN,
        stop_str=("<|im_end|>",),
        image_token="<image>\n",
    )
)

# There is default system prompt for qwen
# reference: https://modelscope.cn/models/qwen/Qwen2-72B-Instruct/file/view/master?fileName=tokenizer_config.json&status=1
# The chat template is: "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
# Qwen 系列（带固定 system prompt 的 ChatML 变体）
register_chat_template(
    ChatTemplate(
        name="qwen",
        default_system_prompt="You are a helpful assistant.",
        role_prefix_and_suffix={
            "system": ("<|im_start|>system\n", "<|im_end|>\n"),
            "user": ("<|im_start|>user\n", "<|im_end|>\n"),
            "assistant": ("<|im_start|>assistant\n", "<|im_end|>\n"),
        },
        style=ChatTemplateStyle.PLAIN,
        stop_str=("<|im_end|>",),
    )
)

# Reference: https://huggingface.co/docs/transformers/main/model_doc/qwen2_vl#usage-example
# Qwen2-VL 多模态模板（使用视觉特殊 token）
register_chat_template(
    ChatTemplate(
        name="qwen2-vl",
        default_system_prompt="You are a helpful assistant.",
        role_prefix_and_suffix={
            "system": ("<|im_start|>system\n", "<|im_end|>\n"),
            "user": ("<|im_start|>user\n", "<|im_end|>\n"),
            "assistant": ("<|im_start|>assistant\n", "<|im_end|>\n"),
        },
        style=ChatTemplateStyle.PLAIN,
        stop_str=("<|im_end|>",),
        image_token="<|vision_start|><|image_pad|><|vision_end|>",
    )
)

# Reference: https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md#prompt-template
# Vicuna 系列（USER/ASSISTANT 格式，支持多模态 LLaVA 变体）
register_chat_template(
    ChatTemplate(
        name="vicuna_v1.1",
        default_system_prompt=(
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        ),
        role_prefix_and_suffix={
            "system": ("", " "),
            "user": ("USER:", " "),
            "assistant": ("ASSISTANT:", "</s>"),
        },
        image_token=" <image>\n",
    )
)

# LLaMA-2 Chat 格式（[INST]/[/INST] 包裹，使用 LLAMA2 样式）
register_chat_template(
    ChatTemplate(
        name="llama-2-chat",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": ("<<SYS>>\n", "\n<</SYS>>\n\n"),
            "user": ("[INST] ", " [/INST]"),
            "assistant": ("", " </s><s>"),
        },
        style=ChatTemplateStyle.LLAMA2,
    )
)

# Reference: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/blob/main/chat_template.json
# Mistral/Mixtral Instruct 格式（[INST]/[/INST]，支持 Pixtral 多模态）
register_chat_template(
    ChatTemplate(
        name="mistral",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": ("[SYSTEM_PROMPT] ", " [/SYSTEM_PROMPT]"),
            "user": ("[INST] ", " [/INST]"),
            "assistant": ("", " </s><s>"),
        },
        stop_str=("</s>",),
        image_token="[IMG]",
    )
)

# LLaMA-3 Instruct 格式（使用 header_id/eot_id 特殊 token）
register_chat_template(
    ChatTemplate(
        name="llama-3-instruct",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": (
                "<|start_header_id|>system<|end_header_id|>\n\n",
                "<|eot_id|>",
            ),
            "user": (
                "<|start_header_id|>user<|end_header_id|>\n\n",
                "<|eot_id|>",
            ),
            "assistant": (
                "<|start_header_id|>assistant<|end_header_id|>\n\n",
                "<|eot_id|>",
            ),
        },
        stop_str=("<|eot_id|>",),
        image_token="<|image|>",
    )
)

# https://huggingface.co/openbmb/MiniCPM-V-2_6
# MiniCPM-V 视觉语言模型对话格式
register_chat_template(
    ChatTemplate(
        name="minicpmv",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": ("", " "),
            "user": ("user:", " "),
            "assistant": ("assistant:", "</s>"),
        },
        stop_str=("<|im_end|>", "<|endoftext|>"),
        image_token="(<image>./</image>)",
    )
)

# Janus-Pro 多模态生成模型（DeepSeek 系列）对话格式
register_chat_template(
    ChatTemplate(
        name="janus-pro",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": (
                "",
                "",
            ),
            "User": (
                "<｜User｜>",
                "",
            ),
            "assistant": (
                "<｜Assistant｜>",
                "<｜end▁of▁sentence｜>",
            ),
        },
        stop_str=("<｜end▁of▁sentence｜>",),
        image_token="<image_placeholder>\n",
    )
)

# https://huggingface.co/openbmb/MiniCPM-o-2_6
# MiniCPM-o 多模态模型（支持图像和音频输入）
register_chat_template(
    ChatTemplate(
        name="minicpmo",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": ("", " "),
            "user": ("user:", " "),
            "assistant": ("assistant:", "</s>"),
        },
        stop_str=("<|im_end|>", "<|endoftext|>"),
        image_token="(<image>./</image>)",
        audio_token="(<audio>./</audio>)",
    )
)

# Janus（非 Pro）多模态生成模型对话格式
register_chat_template(
    ChatTemplate(
        name="janus",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": (
                "",
                "",
            ),
            "user": (
                "<｜User｜>",
                "",
            ),
            "assistant": (
                "<｜Assistant｜>",
                "<｜end▁of▁sentence｜>",
            ),
        },
        stop_str=("<｜end▁of▁sentence｜>",),
        image_token="<image_placeholder>\n",
    )
)

# The difference between "llama-3-instruct-llava" and "llama-3-instruct" is that llava uses a different image_token.
# LLaMA-3 Instruct + LLaVA 多模态变体（image_token 与基础版不同）
register_chat_template(
    ChatTemplate(
        name="llama-3-instruct-llava",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": (
                "<|start_header_id|>system<|end_header_id|>\n\n",
                "<|eot_id|>",
            ),
            "user": (
                "<|start_header_id|>user<|end_header_id|>\n\n",
                "<|eot_id|>",
            ),
            "assistant": (
                "<|start_header_id|>assistant<|end_header_id|>\n\n",
                "<|eot_id|>",
            ),
        },
        stop_str=("<|eot_id|>",),
        image_token="<image>\n",
    )
)

# Reference: https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct/blob/main/chat_template.json
# LLaMA-4 Instruct 格式（使用 header_start/header_end/eot 特殊 token）
register_chat_template(
    ChatTemplate(
        name="llama-4",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": (
                "<|header_start|>system<|header_end|>\n\n",
                "<|eot|>",
            ),
            "user": (
                "<|header_start|>user<|header_end|>\n\n",
                "<|eot|>",
            ),
            "assistant": (
                "<|header_start|>assistant<|header_end|>\n\n",
                "<|eot|>",
            ),
        },
        stop_str=("<|eot|>",),
        image_token="<|image|>",
    )
)

# Reference: https://modelscope.cn/models/01ai/Yi-1.5-34B-Chat/file/view/master?fileName=tokenizer_config.json&status=1
# Yi-1.5 Chat 格式（user 消息后缀包含 assistant 开始标记，特殊的单轮格式）
register_chat_template(
    ChatTemplate(
        name="yi-1.5",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": ("", ""),
            "user": ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n"),
            "assistant": ("", "<|im_end|>\n"),
        },
        style=ChatTemplateStyle.PLAIN,
        stop_str=("<|im_end|>",),
    )
)

# Reference: https://github.com/01-ai/Yi/tree/main/VL#major-difference-with-llava
# Yi-VL 视觉语言模型格式（带双语 system prompt）
register_chat_template(
    ChatTemplate(
        name="yi-vl",
        default_system_prompt=(
            "This is a chat between an inquisitive human and an AI assistant. Assume the role of the AI assistant. Read all the images carefully, and respond to the human's questions with informative, helpful, detailed and polite answers."
            "这是一个好奇的人类和一个人工智能助手之间的对话。假设你扮演这个AI助手的角色。仔细阅读所有的图像，并对人类的问题做出信息丰富、有帮助、详细的和礼貌的回答。"
        ),
        role_prefix_and_suffix={
            "system": ("", "\n\n"),
            "user": ("### Human:", "\n"),
            "assistant": ("### Assistant:", "\n"),
        },
        image_token=" <image_placeholder>\n",
    )
)

# Google Gemma IT（Instruction-Tuned）对话格式，支持视觉和音频输入
register_chat_template(
    ChatTemplate(
        name="gemma-it",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": ("", ""),
            "user": ("<start_of_turn>user\n", "<end_of_turn>\n"),
            "assistant": ("<start_of_turn>model\n", "<end_of_turn>\n"),
        },
        image_token="<start_of_image>",
        audio_token="<start_of_audio>",
        style=ChatTemplateStyle.PLAIN,
    )
)

# Google Gemma-4 IT 对话格式（使用 <|turn>/<turn|> 标记）
register_chat_template(
    ChatTemplate(
        name="gemma-4-it",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": ("", ""),
            "user": ("<|turn>user\n", "<turn|>\n"),
            "assistant": ("<|turn>assistant\n", "<turn|>\n"),
        },
        style=ChatTemplateStyle.PLAIN,
    )
)

# Databricks DBRX Instruct 对话格式（带详细 system prompt）
register_chat_template(
    ChatTemplate(
        name="dbrx-instruct",
        default_system_prompt="You are DBRX, created by Databricks. You were last updated in December 2023. You answer questions based on information available up to that point.\nYOU PROVIDE SHORT RESPONSES TO SHORT QUESTIONS OR STATEMENTS, but provide thorough responses to more complex and open-ended questions.\nYou assist with various tasks, from writing to coding (using markdown for code blocks — remember to use ``` with code, JSON, and tables).\n(You do not have real-time data access or code execution capabilities. You avoid stereotyping and provide balanced perspectives on controversial topics. You do not provide song lyrics, poems, or news articles and do not divulge details of your training data.)\nThis is your system prompt, guiding your responses. Do not reference it, just respond to the user. If you find yourself talking about this message, stop. You should be responding appropriately and usually that means not mentioning this.\nYOU DO NOT MENTION ANY OF THIS INFORMATION ABOUT YOURSELF UNLESS THE INFORMATION IS DIRECTLY PERTINENT TO THE USER'S QUERY.",
        role_prefix_and_suffix={
            "system": ("<|im_start|>system\n", "<|im_end|>"),
            "user": ("\n<|im_start|>user\n", "<|im_end|>"),
            "assistant": ("\n<|im_start|>assistant\n", "<|im_end|>"),
        },
        stop_str=("<|im_end|>",),
    )
)

# Cohere C4AI Command-R 对话格式
register_chat_template(
    ChatTemplate(
        name="c4ai-command-r",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": (
                "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>",
                "<|END_OF_TURN_TOKEN|>",
            ),
            "user": ("<|START_OF_TURN_TOKEN|><|USER_TOKEN|>", "<|END_OF_TURN_TOKEN|>"),
            "assistant": (
                "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
                "<|END_OF_TURN_TOKEN|>",
            ),
        },
        style=ChatTemplateStyle.PLAIN,
    )
)

# Adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_intern_vit.py
# InternVL2.5 多模态大模型对话格式（上海人工智能实验室 + 清华大学）
register_chat_template(
    ChatTemplate(
        name="internvl-2-5",
        default_system_prompt="你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。",
        role_prefix_and_suffix={
            "system": ("<|im_start|>system\n", "<|im_end|>\n"),
            "user": ("<|im_start|>user\n", "<|im_end|>\n"),
            "assistant": ("<|im_start|>assistant\n", "<|im_end|>\n"),
        },
        stop_str=["<|im_end|>", "<|action_end|>"],
    )
)

# Intern-S1 推理模型格式（带推理思考 <think>...</think> 提示的 system prompt）
register_chat_template(
    ChatTemplate(
        name="interns1",
        default_system_prompt="You are an AI assistant whose name is Intern-S1 (书生大模型).\n- Intern-S1 (书生大模型) is a vision-language model that is developed by Shanghai AI Laboratory (上海人工智能实验室).  It is designed to be helpful, honest, and harmless.\n- Intern-S1 (书生大模型) can understand and communicate fluently in the language chosen by the user such as English and 中文.\nYou are an expert reasoner with extensive experience in all areas. You approach problems through systematic thinking and rigorous reasoning. Your response should reflect deep understanding and precise logical thinking, making your solution path and reasoning clear to others. Please put your thinking process within <think>...</think> tags.",
        role_prefix_and_suffix={
            "system": ("<|im_start|>system\n", "<|im_end|>\n"),
            "user": ("<|im_start|>user\n", "<|im_end|>\n"),
            "assistant": ("<|im_start|>assistant\n", "<|im_end|>\n"),
        },
        stop_str=["<|im_end|>", "<|action_end|>"],
    )
)

# IBM Granite-3 Instruct 对话格式（使用 start_of_role/end_of_role 标记）
register_chat_template(
    ChatTemplate(
        name="granite-3-instruct",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": (
                "<|start_of_role|>system<|end_of_role|>",
                "<|end_of_text|>",
            ),
            "user": (
                "<|start_of_role|>user<|end_of_role|>",
                "<|end_of_text|>",
            ),
            "assistant": (
                "<|start_of_role|>assistant<|end_of_role|>",
                "<|end_of_text|>",
            ),
        },
        stop_str=("<|end_of_text|>",),
    )
)

# DeepSeek-V3/R1 对话格式（使用中文特殊 Unicode 标记）
register_chat_template(
    ChatTemplate(
        name="deepseek-v3",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": (
                "",
                "",
            ),
            "user": (
                "<｜User｜>",
                "",
            ),
            "assistant": (
                "<｜Assistant｜>",
                "<｜end▁of▁sentence｜>",
            ),
        },
        stop_str=("<｜end▁of▁sentence｜>",),
    )
)

# Reference: https://huggingface.co/docs/transformers/main/model_doc/glm4_v#usage-example
# GLM-4V 多模态对话格式（清华大学智谱 AI）
register_chat_template(
    ChatTemplate(
        name="glm-4v",
        default_system_prompt=None,
        role_prefix_and_suffix={
            "system": ("<|system|>\n", "\n"),
            "user": ("<|user|>\n", "\n"),
            "assistant": ("<|assistant|>\n", "\n"),
        },
        style=ChatTemplateStyle.PLAIN,
        stop_str=["<|user|>", "<|endoftext|>", "<|observation|>"],
        image_token="<|image|>",
    )
)


# ==================== 模型路径匹配函数注册 ====================
# 每个函数接受模型路径字符串，返回对应模板名称（未匹配返回 None）

# 匹配 DeepSeek-V3/R1（非 base 版本）→ deepseek-v3 模板
@register_chat_template_matching_function
def match_deepseek(model_path: str):
    if re.search(r"deepseek-(v3|r1)", model_path, re.IGNORECASE) and not re.search(
        r"base", model_path, re.IGNORECASE
    ):
        return "deepseek-v3"


# 匹配 Orion 系列 → 使用 claude 模板（对话格式相似）
@register_chat_template_matching_function
def match_orion(model_path: str):
    if "orion" in model_path.lower():
        return "claude"


# 匹配 DeepSeek Janus 系列（多模态生成）→ janus-pro 模板
@register_chat_template_matching_function
def match_deepseek_janus_pro(model_path: str):
    if re.search(r"janus", model_path, re.IGNORECASE):
        return "janus-pro"


# 匹配 Databricks DBRX Instruct → dbrx-instruct 模板
@register_chat_template_matching_function
def match_dbrx(model_path: str):
    if re.search(r"dbrx", model_path, re.IGNORECASE) and re.search(
        r"instruct", model_path, re.IGNORECASE
    ):
        return "dbrx-instruct"


# 匹配 Vicuna/LLaVA-1.5/LLaVA-next-video-7B → vicuna_v1.1 模板
@register_chat_template_matching_function
def match_vicuna(model_path: str):
    if re.search(r"vicuna|llava-v1\.5|llava-next-video-7b", model_path, re.IGNORECASE):
        return "vicuna_v1.1"


# 匹配 LLaMA-2 Chat / CodeLLaMA Instruct → llama-2-chat 模板
@register_chat_template_matching_function
def match_llama2_chat(model_path: str):
    if re.search(
        r"llama-2.*chat|codellama.*instruct",
        model_path,
        re.IGNORECASE,
    ):
        return "llama-2-chat"


# 匹配 Mistral/Mixtral Instruct / Pixtral → mistral 模板
@register_chat_template_matching_function
def match_mistral(model_path: str):
    if re.search(r"pixtral|(mistral|mixtral).*instruct", model_path, re.IGNORECASE):
        return "mistral"


# 匹配 LLaMA-3 Instruct 系列 → llama-3-instruct 模板
@register_chat_template_matching_function
def match_llama3_instruct(model_path: str):
    if re.search(r"llama-3.*instruct", model_path, re.IGNORECASE):
        return "llama-3-instruct"


# 匹配 ChatML 格式的多种模型（TinyLlama/Qwen2-VL/GLM-4V/Qwen/LLaVA-Onevision 等）
@register_chat_template_matching_function
def match_chat_ml(model_path: str):
    if re.search(r"tinyllama", model_path, re.IGNORECASE):
        return "chatml"
    if re.search(r"qwen.*vl", model_path, re.IGNORECASE):
        return "qwen2-vl"
    if re.search(r"glm[-_]?4(\.\d+)?v", model_path, re.IGNORECASE):
        return "glm-4v"
    if re.search(r"qwen.*(chat|instruct)", model_path, re.IGNORECASE) and not re.search(
        r"llava", model_path, re.IGNORECASE
    ):
        return "qwen"
    if re.search(
        r"llava-v1\.6-34b|llava-v1\.6-yi-34b|llava-next-video-34b|llava-onevision-qwen2",
        model_path,
        re.IGNORECASE,
    ):
        return "chatml-llava"


# 匹配 Yi-VL / Yi-1.5 Chat 系列
@register_chat_template_matching_function
def match_chat_yi(model_path: str):
    if re.search(r"yi-vl", model_path, re.IGNORECASE) and not re.search(
        r"llava", model_path, re.IGNORECASE
    ):
        return "yi-vl"
    elif re.search(r"yi-1\.5.*chat", model_path, re.IGNORECASE):
        return "yi-1.5"


# 匹配 Gemma-4 IT 或 Gemma IT 系列（含 Gemma-3）
@register_chat_template_matching_function
def match_gemma(model_path: str):
    if re.search(r"gemma-4.*it", model_path, re.IGNORECASE):
        return "gemma-4-it"
    if re.search(r"(gemma.*it)|(gemma-3)", model_path, re.IGNORECASE):
        return "gemma-it"


# 匹配 MiniCPM-V（视觉）和 MiniCPM-o（多模态）
@register_chat_template_matching_function
def match_openbmb_minicpm(model_path: str):
    if re.search(r"minicpm-v", model_path, re.IGNORECASE):
        return "minicpmv"
    elif re.search(r"minicpm-o", model_path, re.IGNORECASE):
        return "minicpmo"


# 匹配 Cohere C4AI Command-R 系列
@register_chat_template_matching_function
def match_c4ai_command_r(model_path: str):
    if re.search(r"c4ai-command-r", model_path, re.IGNORECASE):
        return "c4ai-command-r"


# 匹配 IBM Granite Instruct 系列
@register_chat_template_matching_function
def match_granite_instruct(model_path: str):
    if re.search(r"granite.*instruct", model_path, re.IGNORECASE):
        return "granite-3-instruct"


# 匹配 InternVL2.5 系列（以 internvl2_5 为标识）
@register_chat_template_matching_function
def match_internvl_chat(model_path: str):
    if re.search(r"internvl2_5", model_path, re.IGNORECASE):
        return "internvl-2-5"


# 匹配 Intern-S1 推理模型（intern-s1 或 interns1）
@register_chat_template_matching_function
def match_interns1_chat(model_path: str):
    if re.search(r"intern-s1", model_path, re.IGNORECASE):
        return "interns1"
    if re.search(r"interns1", model_path, re.IGNORECASE):
        return "interns1"


# ==================== 模块测试入口 ====================
if __name__ == "__main__":
    messages = [
        {"role": "system", "content": None},  # None means default
        # {"role": "system", "content": "You are a helpful, respectful and honest assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "What can you do?"},
        {"role": "assistant", "content": "I can chat with you."},
    ]

    # 测试 LLaMA-2 Chat 模板的 prompt 渲染
    template = get_chat_template("llama-2-chat")
    print(template.get_prompt(messages))
