# 导入标准库
import dataclasses
import logging
import time
import warnings
from typing import List, Optional, Union

# 数值计算（用于 argmax 选择最优候选）
import numpy as np

# 导入后端基类
from sglang.lang.backend.base_backend import BaseBackend
# 导入聊天模板相关类型和工厂函数
from sglang.lang.chat_template import ChatTemplate, get_chat_template_by_model_path
# 导入选择决策和采样方法类型
from sglang.lang.choices import ChoicesDecision, ChoicesSamplingMethod
# 导入流式执行器
from sglang.lang.interpreter import StreamExecutor
# 导入采样参数 IR 节点
from sglang.lang.ir import SglSamplingParams

# 尝试导入 openai 和 tiktoken；若未安装则将异常存储供运行时抛出
try:
    import openai
    import tiktoken
except ImportError as e:
    openai = tiktoken = e


# 模块级日志记录器
logger = logging.getLogger(__name__)


def create_logit_bias_int(tokenizer):
    """Get logit bias for integer numbers."""
    # 收集所有代表数字（0-9）及空格的 token ID
    int_token_ids = []

    tokens = tokenizer._mergeable_ranks
    for token, token_id in tokens.items():
        s = tokenizer.decode([token_id])
        # 只保留纯数字 token 和空格 token
        if all([c.isdigit() for c in s]) or s in [" "]:
            int_token_ids.append(token_id)
            # OpenAI API logit_bias 最多支持 300 个 token
            if len(int_token_ids) >= 300:  # OpenAI API limit
                break
    special_tokens = tokenizer._special_tokens
    # 对筛选出的 token 设置 logit bias +100（强制生成整数）
    mask = {t: 100 for t in int_token_ids[:299]}
    # 同时允许 EOS token 以便模型能够结束生成
    mask[special_tokens["<|endoftext|>"]] = 100
    return mask


# 仅支持 Completion 接口（非 Chat）的 instruct 模型名称列表
INSTRUCT_MODEL_NAMES = [
    "gpt-3.5-turbo-instruct",
]


# 用于记录 API token 消耗的数据类
@dataclasses.dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int

    def reset(self):
        # 重置 token 计数
        self.prompt_tokens = self.completion_tokens = 0


# OpenAI 后端：支持 Chat Completion 和 Completion（instruct）两种接口模式
class OpenAI(BaseBackend):
    def __init__(
        self,
        model_name: str,
        is_chat_model: Optional[bool] = None,    # 手动指定是否为 chat 模型
        chat_template: Optional[ChatTemplate] = None,
        is_azure: bool = False,                  # 是否使用 Azure OpenAI 服务
        *args,
        **kwargs,
    ):
        super().__init__()

        # 若导入失败则在此时抛出 ImportError
        if isinstance(openai, Exception):
            raise openai

        # 根据是否 Azure 选择不同的客户端类
        if is_azure:
            self.client = openai.AzureOpenAI(*args, **kwargs)
        else:
            self.client = openai.OpenAI(*args, **kwargs)

        self.model_name = model_name
        # 加载对应模型的 tiktoken 分词器（用于 select 和 logit_bias 构建）
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # 模型未注册时回退到 cl100k_base（GPT-4/3.5-turbo 默认编码）
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        # 预建整数 logit_bias 掩码，供 dtype=int 约束生成时使用
        self.logit_bias_int = create_logit_bias_int(self.tokenizer)

        # 确定聊天模板（手动指定优先，否则按模型名称自动推断）
        self.chat_template = chat_template or get_chat_template_by_model_path(
            model_name
        )

        # 自动判断是否为 chat 模型（instruct 模型使用 Completion 接口）
        if is_chat_model is not None:
            self.is_chat_model = is_chat_model
        else:
            if model_name in INSTRUCT_MODEL_NAMES:
                self.is_chat_model = False
            else:
                self.is_chat_model = True

        # 记录 assistant 角色前缀，用于检测生成位置是否合法
        self.chat_prefix = self.chat_template.role_prefix_and_suffix["assistant"][0]

        # Usage
        # 初始化 token 用量统计对象
        self.token_usage = TokenUsage(0, 0)

        # API speculative execution
        # TODO(ying): This does not support multi-threading (run_batch)
        # 投机执行（speculative execution）相关状态：一次 API 调用填充多个变量
        self.spec_kwargs = {}      # 合并后的采样参数
        self.spec_format = []      # 记录各变量的文本段和停止词
        self.spec_max_num_tries = 3  # 最大重试次数（模式匹配失败时重试）

    def get_chat_template(self):
        # 返回当前后端使用的聊天模板
        return self.chat_template

    def _prepare_spec_execution(
        self,
        sampling_params: SglSamplingParams,
        num_api_spec_tokens: int,
        spec_var_name: str,
    ):
        # 设置或校验投机执行的 max_tokens 预算
        if "max_tokens" not in self.spec_kwargs:
            self.spec_kwargs["max_tokens"] = num_api_spec_tokens
        else:
            assert self.spec_kwargs["max_tokens"] == num_api_spec_tokens

        params = sampling_params.to_openai_kwargs()
        for key, value in params.items():
            # stop 参数由各变量独立管理，不纳入全局 spec_kwargs
            if key in ["stop"]:
                continue
            if key in ["max_tokens"]:
                # max_tokens 由投机 token 数覆盖，忽略用户设置
                warnings.warn(
                    "The parameter max_tokens will be overwritten by speculated number of tokens."
                )
                continue
            # 其余参数合并到全局 spec_kwargs，并校验一致性
            if key not in self.spec_kwargs:
                self.spec_kwargs[key] = value
            else:
                assert (
                    value == self.spec_kwargs[key]
                ), "sampling parameters should be consistent if turn on api speculative execution."
        # 将当前变量的格式信息（停止词、变量名）追加到 spec_format
        self.spec_format.append(
            {"text": "", "stop": params["stop"], "name": spec_var_name}
        )
        # 占位返回，真正结果在 role_end_generate 中填充
        return "", {}

    def generate(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
        spec_var_name: str = None,
    ):
        # dtype=None 表示普通文本生成（无类型约束）
        if sampling_params.dtype is None:
            if self.is_chat_model:
                if s.num_api_spec_tokens is None:
                    # 非投机模式：要求当前文本以 assistant 前缀结尾
                    if not s.text_.endswith(self.chat_prefix):
                        raise RuntimeError(
                            "This use case is not supported if api speculative execution is off. "
                            "For OpenAI chat models, sgl.gen must be right after sgl.assistant. "
                            "Example of adding api speculative execution: @function(num_api_spec_tokens=128)."
                        )
                    # 使用结构化 messages 作为 prompt
                    prompt = s.messages_
                else:
                    # 投机模式：暂不发请求，记录格式后返回
                    return self._prepare_spec_execution(
                        sampling_params, s.num_api_spec_tokens, spec_var_name
                    )
            else:
                # Completion 模式直接使用文本 prompt
                prompt = s.text_

            kwargs = sampling_params.to_openai_kwargs()
            # o1/o3 模型使用 max_completion_tokens，其余模型使用 max_tokens
            if (
                self.model_name.startswith("o1")
                or self.model_name.startswith("o3")
                or "o1" in self.model_name
            ):
                kwargs.pop("max_tokens", None)
            else:
                kwargs.pop("max_completion_tokens", None)

            # 调用统一的 openai_completion 函数发起请求
            comp = openai_completion(
                client=self.client,
                token_usage=self.token_usage,
                is_chat=self.is_chat_model,
                model=self.model_name,
                prompt=prompt,
                **kwargs,
            )
            # Keep the returned list (or string) as is.
        elif sampling_params.dtype in [str, "str", "string"]:
            # 字符串约束：在 prompt 末尾加引号，强制生成带引号的字符串
            assert (
                not self.is_chat_model
            ), "constrained type not supported on chat model"
            kwargs = sampling_params.to_openai_kwargs()
            kwargs.pop("stop")
            comp = openai_completion(
                client=self.client,
                token_usage=self.token_usage,
                is_chat=self.is_chat_model,
                model=self.model_name,
                prompt=s.text_ + '"',
                stop='"',
                **kwargs,
            )
            # Wrap each element in quotes if we have a list.
            # 将生成结果包裹回引号（恢复字符串格式）
            if isinstance(comp, list):
                comp = ['"' + x + '"' for x in comp]
            else:
                comp = '"' + comp + '"'
        elif sampling_params.dtype in [int, "int"]:
            # 整数约束：使用 logit_bias 强制只生成数字 token
            assert (
                not self.is_chat_model
            ), "constrained type not supported on chat model"
            kwargs = sampling_params.to_openai_kwargs()
            kwargs.pop("stop")
            comp = openai_completion(
                client=self.client,
                token_usage=self.token_usage,
                is_chat=self.is_chat_model,
                model=self.model_name,
                prompt=s.text_,
                logit_bias=self.logit_bias_int,
                stop=[" "],
                **kwargs,
            )
            # Leave as a list if that's what is returned.
        else:
            raise ValueError(f"Unknown dtype: {sampling_params.dtype}")

        # 返回 (生成文本, 元信息字典)
        return comp, {}

    def spec_fill(self, value: str):
        # 在 spec_format 中插入固定文本段（非生成变量，如角色前缀）
        assert self.is_chat_model
        self.spec_format.append({"text": value, "stop": None, "name": None})

    def spec_pattern_match(self, comp):
        # 尝试将投机生成的完整文本按 spec_format 格式拆分回各变量
        for i, term in enumerate(self.spec_format):
            text = term["text"]
            if text != "":
                # 固定文本段：comp 必须以该段开头
                if comp.startswith(text):
                    comp = comp[len(text) :]
                else:
                    return False
            else:
                # 变量段：找到对应的 stop 标记，提取其前的内容
                pos = comp.find(term["stop"])
                if pos != -1:
                    term["text"] = comp[:pos]
                    comp = comp[pos:]
                else:
                    # 最后一个变量允许消耗剩余所有文本
                    if i == len(self.spec_format) - 1:
                        term["text"] = comp
                    else:
                        return False
        return True

    def role_end_generate(
        self,
        s: StreamExecutor,
    ):
        # 在 assistant 角色结束时触发投机执行：发送一次 API 请求填充所有变量
        if s.num_api_spec_tokens is None or not s.text_.endswith(self.chat_prefix):
            return

        comp = ""
        # 若有未填充的变量，则发起实际 API 调用
        if not all(x["name"] is None for x in self.spec_format):
            # TODO(ying): throw errors or warnings
            for i in range(self.spec_max_num_tries):
                comp = openai_completion(
                    client=self.client,
                    token_usage=self.token_usage,
                    is_chat=self.is_chat_model,
                    model=self.model_name,
                    prompt=s.messages_,
                    **self.spec_kwargs,
                )
                # Use a string for pattern matching.
                # 批量生成时取第一个结果进行模式匹配
                comp_for_match = comp[0] if isinstance(comp, list) else comp
                if self.spec_pattern_match(comp_for_match):
                    break

        # 将各变量填充结果写回执行器状态
        for term in self.spec_format:
            s.text_ += term["text"]
            name = term["name"]
            if name is not None:
                # 设置变量值并触发等待该变量的事件
                s.variables[name] = term["text"]
                s.meta_info[name] = {}
                s.variable_event[name].set()

        # 重置投机执行状态，准备下一轮
        self.spec_kwargs = {}
        self.spec_format = []

    def generate_stream(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        # 流式生成仅支持 dtype=None（无类型约束）
        if sampling_params.dtype is None:
            if self.is_chat_model:
                # 同 generate：要求当前文本以 assistant 前缀结尾
                if not s.text_.endswith(self.chat_prefix):
                    raise RuntimeError(
                        "This use case is not supported. "
                        "For OpenAI chat models, sgl.gen must be right after sgl.assistant"
                    )
                prompt = s.messages_
            else:
                prompt = s.text_

            kwargs = sampling_params.to_openai_kwargs()
            # 调用流式完成函数，返回生成器
            generator = openai_completion_stream(
                client=self.client,
                token_usage=self.token_usage,
                is_chat=self.is_chat_model,
                model=self.model_name,
                prompt=prompt,
                **kwargs,
            )
            return generator
        else:
            raise ValueError(f"Unknown dtype: {sampling_params.dtype}")

    def select(
        self,
        s: StreamExecutor,
        choices: List[str],
        temperature: float,
        choices_method: ChoicesSamplingMethod,
    ) -> ChoicesDecision:
        """Note: `choices_method` is not used by the OpenAI backend."""
        # select 操作仅支持 Completion 接口（非 chat 模型）
        if self.is_chat_model:
            raise NotImplementedError(
                "select/choices is not supported for chat models. "
                "Please try to use a non-chat model such as gpt-3.5-turbo-instruct"
            )

        n_choices = len(choices)
        # 将各候选字符串编码为 token ID 序列
        token_ids = [self.tokenizer.encode(x) for x in choices]
        # 初始化各候选的得分
        scores = [0] * n_choices
        # valid[i] 表示第 i 个候选是否仍然有效（未被排除）
        valid = [len(x) > 0 for x in token_ids]
        # 将当前 prompt 文本编码为 token ID 序列
        prompt_tokens = self.tokenizer.encode(s.text_)

        # 按最长候选的长度逐 step 进行 token 级别选择
        max_len = max([len(x) for x in token_ids])
        for step in range(max_len):
            # Build logit bias
            # 为仍有效的候选的当前 step token 设置 logit_bias +100
            logit_bias = {}
            for i in range(n_choices):
                if valid[i]:
                    logit_bias[token_ids[i][step]] = 100

            # Call API
            # 发起单 token 生成请求，通过 logit_bias 引导模型选择
            ret = self.client.completions.create(
                model=self.model_name,
                prompt=prompt_tokens,
                logit_bias=logit_bias,
                max_tokens=1,
                temperature=temperature,
            )
            ret_str = ret.choices[0].text
            # 解码返回的 token
            ret_token = self.tokenizer.encode(ret_str)[0]
            # 更新 token 用量统计
            self.token_usage.prompt_tokens += ret.usage.prompt_tokens
            self.token_usage.completion_tokens = ret.usage.completion_tokens

            # TODO:
            # 1. return logits as the scores
            # 2. compute logits of the full choice
            # 3. consider chunk-based decoding

            # Update valid
            # 根据生成 token 更新各候选的有效性和得分
            hit = False
            for i in range(n_choices):
                if valid[i]:
                    # 候选已到最后一个 token，下一步标记为无效
                    if step == len(token_ids[i]) - 1:
                        valid[i] = False

                    if ret_token == token_ids[i][step]:
                        # 命中：得分 +1
                        scores[i] += 1
                        hit = True
                    else:
                        # 未命中：排除该候选
                        valid[i] = False
            assert hit

            # 若有效候选已收敛到 1 个以内则提前停止
            if np.sum(valid) <= 1:
                break

            # 将生成的 token 追加到 prompt，进行下一步
            prompt_tokens.append(ret_token)

        # 返回得分最高的候选及相关元信息
        return ChoicesDecision(
            decision=choices[np.argmax(scores)],
            meta_info={"scores": scores},
        )


def openai_completion(
    client, token_usage, is_chat=None, retries=3, prompt=None, **kwargs
) -> Union[str, List[str]]:
    # if "ebnf" is in kwargs, warn and remove
    # EBNF 约束暂不支持 OpenAI 接口，发出警告并移除
    if "ebnf" in kwargs:
        warnings.warn("EBNF is not officially supported by OpenAI endpoints. Ignoring.")
        del kwargs["ebnf"]

    # 带重试的 API 调用循环
    for attempt in range(retries):
        try:
            if is_chat:
                # Chat Completion 接口：移除 stop=None 以避免 API 报错
                if "stop" in kwargs and kwargs["stop"] is None:
                    kwargs.pop("stop")
                ret = client.chat.completions.create(messages=prompt, **kwargs)
                # 单候选返回字符串，多候选（n>1）返回列表
                if len(ret.choices) == 1:
                    comp = ret.choices[0].message.content
                else:
                    comp = [c.message.content for c in ret.choices]
            else:
                # Completion 接口：prompt 为列表时返回列表
                ret = client.completions.create(prompt=prompt, **kwargs)
                if isinstance(prompt, (list, tuple)):
                    comp = [c.text for c in ret.choices]
                else:
                    comp = ret.choices[0].text
                    if len(ret.choices) > 1:
                        comp = [c.text for c in ret.choices]

            # 累加 token 用量
            token_usage.prompt_tokens += ret.usage.prompt_tokens
            token_usage.completion_tokens += ret.usage.completion_tokens
            break
        except (openai.APIError, openai.APIConnectionError, openai.RateLimitError) as e:
            # API 错误或限流：等待 5 秒后重试
            logger.error(f"OpenAI Error: {e}. Waiting 5 seconds...")
            time.sleep(5)
            if attempt == retries - 1:
                raise e
        except Exception as e:
            # 其他未知异常直接抛出
            logger.error(f"RuntimeError {e}.")
            raise e

    return comp


def openai_completion_stream(
    client, token_usage, is_chat=None, retries=3, prompt=None, **kwargs
):
    # if "ebnf" is in kwargs, warn and remove
    # EBNF 约束不支持 OpenAI 接口，移除并警告
    if "ebnf" in kwargs:
        warnings.warn("EBNF is not officially supported by OpenAI endpoints. Ignoring.")
        del kwargs["ebnf"]

    # 带重试的流式 API 调用循环
    for attempt in range(retries):
        try:
            if is_chat:
                if "stop" in kwargs and kwargs["stop"] is None:
                    kwargs.pop("stop")
                # Chat 流式请求，开启 include_usage 以获取最终 token 用量
                generator = client.chat.completions.create(
                    messages=prompt,
                    stream=True,
                    stream_options={"include_usage": True},
                    **kwargs,
                )
                for ret in generator:
                    # 跳过无 choices 的心跳包（最后一条 usage 包）
                    if len(ret.choices) == 0:
                        continue
                    try:
                        content = ret.choices[0].delta.content
                    except IndexError:
                        content = None
                    # 产出文本片段（None 时替换为空字符串）
                    yield content or "", {}
            else:
                # Completion 流式请求
                generator = client.completions.create(
                    prompt=prompt,
                    stream=True,
                    stream_options={"include_usage": True},
                    **kwargs,
                )
                for ret in generator:
                    if len(ret.choices) == 0:
                        continue
                    content = ret.choices[0].text
                    yield content or "", {}

            # 流结束后累加 token 用量（ret 为最后一条包含 usage 的响应）
            token_usage.prompt_tokens += ret.usage.prompt_tokens
            token_usage.completion_tokens += ret.usage.completion_tokens
            break
        except (openai.APIError, openai.APIConnectionError, openai.RateLimitError) as e:
            # 限流或连接错误：等待 5 秒后重试
            logger.error(f"OpenAI Error: {e}. Waiting 5 seconds...")
            time.sleep(5)
            if attempt == retries - 1:
                raise e
        except Exception as e:
            logger.error(f"RuntimeError {e}.")
            raise e
