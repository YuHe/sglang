"""The intermediate representation."""
# SGLang 前端语言的中间表示（IR）节点定义
# 所有 sgl.gen / sgl.select / sgl.system 等操作均被编译为这里定义的 IR 节点

import dataclasses
import inspect
import warnings
from typing import List, Optional, Union

# 全局配置（default_backend 等）
from sglang.global_config import global_config
# 候选选择策略类型
from sglang.lang.choices import ChoicesSamplingMethod

# 约束生成用正则表达式常量
REGEX_INT = r"[-+]?[0-9]+[ \n]*"       # 匹配整数（含符号和尾部空白）
REGEX_FLOAT = r"[-+]?[0-9]*\.?[0-9]+[ \n]*"  # 匹配浮点数
REGEX_BOOL = r"(True|False)"            # 匹配布尔值
REGEX_STR = r"\"[\w\d\s]*\""  # bugs with regex r"\".*\"" in interegular pkg
# 匹配带引号的字符串（注意 interegular 包中 ".*" 模式有 bug）


# 采样参数 IR 节点：描述一次 LLM 调用的完整采样配置
@dataclasses.dataclass
class SglSamplingParams:
    max_new_tokens: int = 128           # 最大生成 token 数
    min_new_tokens: int = 0             # 最少生成 token 数
    n: int = 1                          # 并行采样数量（n>1 时返回列表）
    stop: Union[str, List[str]] = ()    # 停止字符串（遇到时终止生成）
    stop_token_ids: Optional[List[int]] = ()  # 停止 token ID 列表
    stop_regex: Optional[Union[str, List[str]]] = ()  # 停止正则（SRT 特有）
    temperature: float = 1.0            # 采样温度（0 为贪婪）
    top_p: float = 1.0                  # nucleus sampling 阈值
    top_k: int = -1  # -1 means disable  # top-k 采样（-1 表示禁用）
    min_p: float = 0.0                  # min-p 采样阈值
    frequency_penalty: float = 0.0     # 词频惩罚（重复词降权）
    presence_penalty: float = 0.0      # 存在惩罚（出现过的词降权）
    ignore_eos: bool = False            # 是否忽略 EOS token（强制生成到 max_new_tokens）
    return_logprob: Optional[bool] = None          # 是否返回 logprob
    logprob_start_len: Optional[int] = (None,)     # logprob 起始位置（token 索引）
    top_logprobs_num: Optional[int] = (None,)      # 返回 top-N logprob 数量
    return_text_in_logprobs: Optional[bool] = (None,)  # logprob 中是否包含对应文本
    json_schema: Optional[str] = None              # JSON Schema 约束（SRT 特有）

    # for constrained generation, not included in to_xxx_kwargs
    # 以下字段用于约束生成，不传递给后端 API（由后端内部处理）
    dtype: Optional[str] = None    # 类型约束（int/float/str/bool）
    regex: Optional[str] = None    # 正则约束（与 dtype 互斥）

    def clone(self):
        # 深复制（不含 dtype/regex，这两个由 SglGen 节点单独管理）
        return SglSamplingParams(
            self.max_new_tokens,
            self.min_new_tokens,
            self.n,
            self.stop,
            self.stop_token_ids,
            self.stop_regex,
            self.temperature,
            self.top_p,
            self.top_k,
            self.min_p,
            self.frequency_penalty,
            self.presence_penalty,
            self.ignore_eos,
            self.return_logprob,
            self.logprob_start_len,
            self.top_logprobs_num,
            self.return_text_in_logprobs,
            self.json_schema,
        )

    def to_openai_kwargs(self):
        # OpenAI does not support top_k, so we drop it here
        # 转换为 OpenAI API 参数格式（不含 top_k，OpenAI 不支持）
        if self.regex is not None:
            warnings.warn("Regular expression is not supported in the OpenAI backend.")
        return {
            "max_tokens": self.max_new_tokens,
            "max_completion_tokens": self.max_new_tokens,  # o1/o3 系列使用此参数
            "n": self.n,
            "stop": self.stop or None,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }

    def to_vertexai_kwargs(self):
        # 转换为 Vertex AI GenerationConfig 参数格式
        if self.regex is not None:
            warnings.warn(
                "Regular expression is not supported in the VertexAI backend."
            )
        return {
            "candidate_count": 1,                              # Vertex AI 仅支持 1 个候选
            "max_output_tokens": self.max_new_tokens,
            "stop_sequences": self.stop,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k if self.top_k > 0 else None,  # top_k=-1 时传 None
        }

    def to_anthropic_kwargs(self):
        # Anthropic does not support frequency_penalty or presence_penalty, so we drop it here
        # 转换为 Anthropic API 参数格式（不含 frequency/presence_penalty）
        if self.regex is not None:
            warnings.warn(
                "Regular expression is not supported in the Anthropic backend."
            )
        return {
            "max_tokens": self.max_new_tokens,
            # Anthropic 的 stop_sequences 必须为列表格式
            "stop_sequences": (
                self.stop if isinstance(self.stop, (list, tuple)) else [self.stop]
            ),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }

    def to_litellm_kwargs(self):
        # 转换为 LiteLLM 参数格式（兼容 OpenAI 风格）
        if self.regex is not None:
            warnings.warn("Regular expression is not supported in the LiteLLM backend.")
        return {
            "max_tokens": self.max_new_tokens,
            "stop": self.stop or None,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }

    def to_srt_kwargs(self):
        # 转换为 SGLang Runtime（SRT）服务器参数格式（最完整，支持所有特性）
        return {
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": self.min_new_tokens,
            "n": self.n,
            "stop": self.stop,
            "stop_token_ids": self.stop_token_ids,
            "stop_regex": self.stop_regex,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "ignore_eos": self.ignore_eos,
            "regex": self.regex,           # 正则约束（SRT 原生支持）
            "json_schema": self.json_schema,  # JSON Schema 约束
        }


# SglFunction：将 Python 函数包装为可执行的 SGLang 程序节点
class SglFunction:
    def __init__(self, func, num_api_spec_tokens=None, bind_arguments=None):
        self.func = func
        # num_api_spec_tokens：OpenAI 投机执行的预留 token 预算
        self.num_api_spec_tokens = num_api_spec_tokens
        # bind_arguments：通过 bind() 绑定的固定参数
        self.bind_arguments = bind_arguments or {}
        # pin_prefix_rid：前缀缓存的请求 ID（首次 run 时填充）
        self.pin_prefix_rid = None

        # Parse arguments
        # 解析函数签名，提取参数名和默认值
        argspec = inspect.getfullargspec(func)
        assert argspec.args[0] == "s", 'The first argument must be "s"'
        # 去掉第一个参数 "s"（执行器），剩余为用户参数
        self.arg_names = argspec.args[1:]
        self.arg_defaults = argspec.defaults if argspec.defaults is not None else []

    def bind(self, **kwargs):
        # 绑定部分参数，返回新的 SglFunction（类似 functools.partial）
        assert all(key in self.arg_names for key in kwargs)

        new_bind_dict = {**self.bind_arguments, **kwargs}
        return SglFunction(self.func, bind_arguments=new_bind_dict)

    def run(
        self,
        *args,
        max_new_tokens: int = 128,
        n: int = 1,
        stop: Optional[Union[str, List[str]]] = None,
        stop_token_ids: Optional[List[int]] = None,
        stop_regex: Optional[Union[str, List[str]]] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        ignore_eos: bool = False,
        return_logprob: Optional[bool] = None,
        logprob_start_len: Optional[int] = None,
        top_logprobs_num: Optional[int] = None,
        return_text_in_logprobs: Optional[bool] = None,
        stream: bool = False,
        backend=None,
        use_thread: bool = True,
        **kwargs,
    ):
        # 单次执行 SGLang 程序，返回 ProgramState 或流式生成器
        from sglang.lang.interpreter import run_program

        # avoid using [] as the default arg: https://nikos7am.com/posts/mutable-default-arguments/
        # 避免可变默认参数问题（Python 函数默认参数共享同一对象）
        if stop is None:
            stop = []
        if stop_token_ids is None:
            stop_token_ids = []
        if stop_regex is None:
            stop_regex = []

        # 构建默认采样参数（调用者可在 SglGen 节点中覆盖部分参数）
        default_sampling_para = SglSamplingParams(
            max_new_tokens=max_new_tokens,
            n=n,
            stop=stop,
            stop_token_ids=stop_token_ids,
            stop_regex=stop_regex,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            ignore_eos=ignore_eos,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            return_text_in_logprobs=return_text_in_logprobs,
        )
        # 未显式指定后端时使用全局默认后端
        backend = backend or global_config.default_backend
        return run_program(
            self,
            backend,
            args,
            kwargs,
            default_sampling_para,
            stream,
            use_thread=use_thread,
        )

    def run_batch(
        self,
        batch_kwargs,
        *,
        max_new_tokens: int = 128,
        n: int = 1,
        stop: Optional[Union[str, List[str]]] = None,
        stop_token_ids: Optional[List[int]] = None,
        stop_regex: Optional[Union[str, List[str]]] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        ignore_eos: bool = False,
        return_logprob: Optional[bool] = None,
        logprob_start_len: Optional[int] = None,
        top_logprobs_num: Optional[int] = None,
        return_text_in_logprobs: Optional[bool] = None,
        backend=None,
        num_threads: Union[str, int] = "auto",
        progress_bar: bool = False,
        generator_style: bool = False,
    ):
        # 批量执行 SGLang 程序，返回 ProgramState 列表（或生成器）
        from sglang.lang.interpreter import run_program_batch

        # 同 run：避免可变默认参数问题
        if stop is None:
            stop = []
        if stop_token_ids is None:
            stop_token_ids = []
        if stop_regex is None:
            stop_regex = []

        assert isinstance(batch_kwargs, (list, tuple))
        # 空批次直接返回空列表
        if len(batch_kwargs) == 0:
            return []
        if not isinstance(batch_kwargs[0], dict):
            num_programs = len(batch_kwargs)
            # change the list of argument values to dict of arg_name -> arg_value
            # 将位置参数列表转换为 {arg_name: value} 字典格式
            batch_kwargs = [
                {self.arg_names[i]: v for i, v in enumerate(arg_values)}
                for arg_values in batch_kwargs
                if isinstance(arg_values, (list, tuple))
                and len(self.arg_names) - len(self.arg_defaults)
                <= len(arg_values)
                <= len(self.arg_names)
            ]
            # Ensure to raise an exception if the number of arguments mismatch
            # 若转换后数量不匹配，说明有参数数量不对的调用
            if len(batch_kwargs) != num_programs:
                raise Exception("Given arguments mismatch the SGL function signature")

        # 构建批次的默认采样参数
        default_sampling_para = SglSamplingParams(
            max_new_tokens=max_new_tokens,
            n=n,
            stop=stop,
            stop_token_ids=stop_token_ids,
            stop_regex=stop_regex,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            ignore_eos=ignore_eos,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            return_text_in_logprobs=return_text_in_logprobs,
        )
        backend = backend or global_config.default_backend
        return run_program_batch(
            self,
            backend,
            batch_kwargs,
            default_sampling_para,
            num_threads,
            progress_bar,
            generator_style=generator_style,
        )

    def trace(self, *, backend=None, **kwargs):
        # 追踪程序结构，生成 IR 节点树（不实际执行生成）
        from sglang.lang.tracer import trace_program

        backend = backend or global_config.default_backend
        return trace_program(self, kwargs, backend)

    def cache(self, backend=None):
        # 预热前缀缓存（将程序前缀填充到 KV cache）
        from sglang.lang.interpreter import cache_program

        backend = backend or global_config.default_backend
        return cache_program(self, backend)

    def __call__(self, *args, **kwargs):
        # 函数调用入口：根据是否在 TracingScope 中选择 run 或 trace
        from sglang.lang.tracer import TracingScope

        tracing_scope = TracingScope.get_current_scope()
        if tracing_scope is None:
            # 正常执行模式
            return self.run(*args, **kwargs)
        else:
            # 追踪模式（被另一个 SGLang 函数调用时）
            kwargs["backend"] = tracing_scope.tracer_state.backend
            return self.trace(*args, **kwargs)


# 所有 IR 表达式节点的基类
class SglExpr:
    # 全局节点计数器，用于生成唯一 node_id
    node_ct = 0

    def __init__(self):
        self.node_id = SglExpr.node_ct
        # prev_node：链式表示中的前驱节点（构成 DAG）
        self.prev_node = None
        # pid：所属程序的唯一 ID（追踪时设置）
        self.pid = None
        SglExpr.node_ct += 1

    def __add__(self, other):
        # 支持 expr1 + expr2 语法，生成 SglExprList
        if isinstance(other, str):
            other = SglConstantText(other)
        assert isinstance(other, SglExpr)

        return self.concatenate_ir(self, other)

    def __radd__(self, other):
        # 支持 "string" + expr 语法
        if isinstance(other, str):
            other = SglConstantText(other)
        assert isinstance(other, SglExpr), f"{other}"

        return self.concatenate_ir(other, self)

    def concatenate_ir(self, a, b):
        # 将两个 IR 节点拼接为 SglExprList（扁平化处理嵌套列表）
        if isinstance(a, SglExprList):
            if isinstance(b, SglExprList):
                return SglExprList(a.expr_list + b.expr_list)
            else:
                return SglExprList(a.expr_list + [b])
        elif isinstance(b, SglExprList):
            return SglExprList([a] + b.expr_list)

        return SglExprList([a, b])

    def print_graph_dfs(self):
        # 以 DFS 顺序打印 IR 节点图（调试用）
        ret = [""]
        visited = set()

        def dfs_print(x):
            if x is None or x in visited:
                return
            visited.add(x)

            # Print dependency
            # 先递归打印前驱节点
            if x.prev_node is not None:
                dfs_print(x.prev_node)

            if isinstance(x, SglExprList):
                for y in x.expr_list:
                    dfs_print(y)
            # elif isinstance(x, SglRole):
            #    dfs_print(x.expr)
            elif isinstance(x, SglVariable):
                dfs_print(x.source)

            # Print the node itself
            # 打印当前节点（fork 相关节点单独处理）
            if isinstance(x, (SglFork, SglGetForkItem)):
                ret[0] += f"%{x.node_id} = {x}\n"
            else:
                if x.prev_node is not None:
                    ret[0] += (
                        f"%{x.node_id} = %{x.prev_node.node_id} + " + str(x) + "\n"
                    )
                else:
                    ret[0] += f"%{x.node_id} = " + str(x) + "\n"

        dfs_print(self)
        return ret[0]


# 表达式列表节点：将多个 IR 节点组合为一个序列
class SglExprList(SglExpr):
    def __init__(self, expr_list: List[SglExpr]):
        super().__init__()
        self.expr_list = expr_list

    def __repr__(self):
        return f"ExprList({self.expr_list})"


# 函数参数节点：在追踪时表示尚未绑定实际值的参数
class SglArgument(SglExpr):
    def __init__(self, name: str, value: str):
        super().__init__()
        self.name = name
        self.value = value

    def __repr__(self):
        return f"Argument(name={self.name}, value={repr(self.value)})"

    def __len__(self):
        return len(self.value)

    def __getitem__(self, i):
        return self.value[i]

    def __int__(self):
        return self.value

    def __bool__(self):
        return self.value

    def __format__(self, *args):
        # 禁止在 f-string 中使用 SglArgument（追踪时无法展开）
        raise TypeError(
            "Cannot put argument inside a f-string. "
            "This is not compatible with the tracer. "
        )


# 图像节点：表示一个图像输入（path 可以是文件路径、URL 或 base64）
class SglImage(SglExpr):
    def __init__(self, path: str):
        self.path = path

    def __repr__(self) -> str:
        return f"SglImage({self.path})"


# 视频节点：表示一个视频输入（path 为文件路径，num_frames 为采样帧数）
class SglVideo(SglExpr):
    def __init__(self, path: str, num_frames: int):
        self.path = path
        self.num_frames = num_frames

    def __repr__(self) -> str:
        return f"SglVideo({self.path}, {self.num_frames})"


# 文本生成节点：对应 sgl.gen() 调用，携带所有采样参数
class SglGen(SglExpr):
    def __init__(
        self,
        name: Optional[str] = None,          # 变量名（结果存储键）
        max_new_tokens: Optional[int] = None,
        min_new_tokens: Optional[int] = None,
        n: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stop_token_ids: Optional[List[int]] = None,
        stop_regex: Optional[Union[str, List[str]]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        ignore_eos: Optional[bool] = None,
        return_logprob: Optional[bool] = None,
        logprob_start_len: Optional[int] = None,
        top_logprobs_num: Optional[int] = None,
        return_text_in_logprobs: Optional[bool] = None,
        dtype: Optional[type] = None,         # 类型约束（int/str/float/bool）
        regex: Optional[str] = None,          # 正则约束
        json_schema: Optional[str] = None,    # JSON Schema 约束
    ):
        """Call the model to generate. See the meaning of the arguments in docs/backend/sampling_params.md"""
        super().__init__()
        self.name = name
        # 将所有参数封装为 SglSamplingParams 节点（None 值保留，由执行器合并默认值）
        self.sampling_params = SglSamplingParams(
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            n=n,
            stop=stop,
            stop_regex=stop_regex,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            ignore_eos=ignore_eos,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            return_text_in_logprobs=return_text_in_logprobs,
            dtype=dtype,
            regex=regex,
            json_schema=json_schema,
        )

    def __repr__(self):
        return f"Gen('{self.name}')"


# 常量文本节点：表示程序中的固定字符串填充
class SglConstantText(SglExpr):
    def __init__(self, value: str):
        super().__init__()
        self.value = value

    def __repr__(self):
        return f"Constant({repr(self.value)})"


# 角色开始节点：表示 Chat 模式下某个角色（system/user/assistant）的开始
class SglRoleBegin(SglExpr):
    def __init__(self, role: str):
        super().__init__()
        self.role = role

    def __repr__(self):
        return f"RoleBegin({self.role})"


# 角色结束节点：表示 Chat 模式下某个角色的结束
class SglRoleEnd(SglExpr):
    def __init__(self, role: str):
        super().__init__()
        self.role = role

    def __repr__(self):
        return f"RoleEnd({self.role})"


# 候选选择节点：从 choices 列表中选择最优候选（受限生成）
class SglSelect(SglExpr):

    def __init__(
        self,
        name: str,
        choices: List[str],
        temperature: float,
        choices_method: ChoicesSamplingMethod,
    ):
        super().__init__()
        self.name = name
        self.choices = choices
        self.temperature = temperature
        # 选择策略（如 token_length_normalized/greedy_token_selection）
        self.choices_method = choices_method

    def __repr__(self):
        return f"Select({self.name}, choices={self.choices}, choices_method={self.choices_method})"


# Fork 节点：将当前执行状态分叉为多个并行分支
class SglFork(SglExpr):
    def __init__(self, number: int, position_ids_offset=None):
        super().__init__()
        self.number = number                              # fork 数量
        self.position_ids_offset = position_ids_offset   # 位置 ID 偏移（可选）

    def __repr__(self):
        return (
            f"Fork(%{self.prev_node.node_id}, number={self.number}, "
            f"position_ids_offset={self.position_ids_offset})"
        )


# GetForkItem 节点：取 fork 中第 index 个分支的结果
class SglGetForkItem(SglExpr):
    def __init__(self, index: int):
        super().__init__()
        self.index = index

    def __repr__(self):
        return f"GetForkItem(%{self.prev_node.node_id}, index={self.index})"


# 变量节点：持有一个命名变量，source 指向生成该变量的 SglGen/SglSelect 节点
class SglVariable(SglExpr):
    def __init__(self, name: str, source):
        super().__init__()
        self.name = name
        self.source = source

    def __repr__(self):
        return f"Variable('{self.name}', source=%{self.source.node_id})"


# 变量作用域开始节点：标记一段代码块的变量作用域起始（暂未广泛使用）
class SglVarScopeBegin(SglExpr):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def __repr__(self):
        return f"VarScopeBegin('{self.name}')"


# 变量作用域结束节点：标记作用域结束，并将范围内生成的内容绑定为变量
class SglVarScopeEnd(SglExpr):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def __repr__(self):
        return f"VarScopeEnd('{self.name}')"


# 拼接并追加节点：将多个执行状态的 KV cache 拼接后追加到目标（Runtime 特有）
class SglConcateAndAppend(SglExpr):
    def __init__(self, states):
        super().__init__()
        self.states = states

    def __repr__(self):
        return f"ConcatenateAndAppend('{self.states}')"


# 提交延迟操作节点：强制将当前未提交的填充发送给后端（触发 KV cache 预填充）
class SglCommitLazy(SglExpr):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "CommitLazy()"


# 分离推理节点：将推理过程（思维链）从最终答案中分离出来并存储到独立变量
class SglSeparateReasoning(SglExpr):
    def __init__(self, model_type: str, expr: SglExpr):
        super().__init__()
        # 模型类型（如 "deepseek-r1"），用于选择解析策略
        self.model_type = model_type

        self.expr = expr
        self.name = None
        # 从关联表达式中提取变量名，并为推理内容生成独立变量名
        self._process_expr(expr)

    def process_name_for_reasoning(self, name):
        # 推理内容变量名为原变量名 + "_reasoning_content" 后缀
        if not name:
            raise ValueError("name must be provided")
        return f"{name}_reasoning_content"

    def _process_expr(self, expr):
        # 递归遍历关联表达式，为 SglGen/SglSelect 节点生成推理变量名
        if isinstance(expr, SglGen):
            self.name = self.process_name_for_reasoning(expr.name)
        elif isinstance(expr, SglSelect):
            self.name = self.process_name_for_reasoning(expr.name)
        elif isinstance(expr, SglExprList):
            for x in expr.expr_list:
                self._process_expr(x)

    def __repr__(self):
        return f"SeparateReasoning(model_type={self.model_type}, name={self.name})"
