"""Public APIs of the language."""
# SGLang 前端语言的公开 API：提供 function/gen/select/role 等用户可直接调用的构建块

import re
from typing import Callable, List, Optional, Union

# 全局配置（default_backend 等）
from sglang.global_config import global_config
# 后端基类
from sglang.lang.backend.base_backend import BaseBackend
# 候选选择方法和默认策略
from sglang.lang.choices import ChoicesSamplingMethod, token_length_normalized
# 导入所有对外暴露的 IR 节点
from sglang.lang.ir import (
    SglExpr,              # 所有 IR 节点的基类
    SglExprList,          # 表达式列表（用于 role 包装等）
    SglFunction,          # 函数程序节点
    SglGen,               # 文本生成节点
    SglImage,             # 图像节点
    SglRoleBegin,         # 角色开始节点
    SglRoleEnd,           # 角色结束节点
    SglSelect,            # 候选选择节点
    SglSeparateReasoning, # 分离推理节点（思维链分离）
    SglVideo,             # 视频节点
)


def function(
    func: Optional[Callable] = None, num_api_spec_tokens: Optional[int] = None
):
    # @sgl.function 装饰器：将 Python 函数包装为 SglFunction 程序节点
    # 支持带参数和不带参数两种装饰器用法
    if func:
        # 直接装饰（无括号）：@sgl.function
        return SglFunction(func, num_api_spec_tokens=num_api_spec_tokens)

    def decorator(func):
        # 带参数装饰（有括号）：@sgl.function(num_api_spec_tokens=128)
        return SglFunction(func, num_api_spec_tokens=num_api_spec_tokens)

    return decorator


def Runtime(*args, **kwargs):
    # Avoid importing unnecessary dependency
    # 延迟导入 Runtime，避免未安装 SRT 服务器时的导入错误
    from sglang.lang.backend.runtime_endpoint import Runtime

    return Runtime(*args, **kwargs)


def Engine(*args, **kwargs):
    # Avoid importing unnecessary dependency
    # 延迟导入 Engine（离线批处理引擎），避免不必要的依赖
    from sglang.srt.entrypoints.engine import Engine

    return Engine(*args, **kwargs)


def set_default_backend(backend: BaseBackend):
    # 设置全局默认后端（后续调用 run/run_batch 时无需显式传入 backend）
    global_config.default_backend = backend


def flush_cache(backend: Optional[BaseBackend] = None):
    # 清空后端 KV cache（不指定后端时使用全局默认后端）
    backend = backend or global_config.default_backend
    if backend is None:
        return False

    # If backend is Runtime
    # 若 backend 是 Runtime 包装器，取其内部 endpoint
    if hasattr(backend, "endpoint"):
        backend = backend.endpoint
    return backend.flush_cache()


def get_server_info(backend: Optional[BaseBackend] = None):
    # 获取后端服务器信息（内存、队列状态等）
    backend = backend or global_config.default_backend
    if backend is None:
        return None

    # If backend is Runtime
    # 若是 Runtime 包装器则取 endpoint
    if hasattr(backend, "endpoint"):
        backend = backend.endpoint
    return backend.get_server_info()


def gen(
    name: Optional[str] = None,
    max_tokens: Optional[int] = None,
    min_tokens: Optional[int] = None,
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
    dtype: Optional[Union[type, str]] = None,
    choices: Optional[List[str]] = None,
    choices_method: Optional[ChoicesSamplingMethod] = None,
    regex: Optional[str] = None,
    json_schema: Optional[str] = None,
):
    """Call the model to generate. See the meaning of the arguments in docs/backend/sampling_params.md"""

    # 若指定了 choices 参数则创建 SglSelect 节点（受限生成）
    if choices:
        return SglSelect(
            name,
            choices,
            # temperature 未指定时 select 使用 0.0（贪婪）
            0.0 if temperature is None else temperature,
            # choices_method 未指定时使用 token 长度归一化策略
            token_length_normalized if choices_method is None else choices_method,
        )

    # check regex is valid
    # 验证 regex 合法性（防止运行时才报错）
    if regex is not None:
        try:
            re.compile(regex)
        except re.error as e:
            raise e

    # 创建并返回 SglGen 节点，携带所有采样参数
    return SglGen(
        name,
        max_tokens,
        min_tokens,
        n,
        stop,
        stop_token_ids,
        stop_regex,
        temperature,
        top_p,
        top_k,
        min_p,
        frequency_penalty,
        presence_penalty,
        ignore_eos,
        return_logprob,
        logprob_start_len,
        top_logprobs_num,
        return_text_in_logprobs,
        dtype,
        regex,
        json_schema,
    )


def gen_int(
    name: Optional[str] = None,
    max_tokens: Optional[int] = None,
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
):
    # gen_int：强制生成整数的快捷函数（dtype=int）
    return SglGen(
        name,
        max_tokens,
        None,   # min_tokens 不适用
        n,
        stop,
        stop_token_ids,
        stop_regex,
        temperature,
        top_p,
        top_k,
        min_p,
        frequency_penalty,
        presence_penalty,
        ignore_eos,
        return_logprob,
        logprob_start_len,
        top_logprobs_num,
        return_text_in_logprobs,
        int,    # dtype=int：强制整数约束
        None,   # regex=None，由 dtype 自动生成
    )


def gen_string(
    name: Optional[str] = None,
    max_tokens: Optional[int] = None,
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
):
    # gen_string：强制生成带引号字符串的快捷函数（dtype=str）
    return SglGen(
        name,
        max_tokens,
        None,   # min_tokens 不适用
        n,
        stop,
        stop_token_ids,
        stop_regex,
        temperature,
        top_p,
        top_k,
        min_p,
        frequency_penalty,
        presence_penalty,
        ignore_eos,
        return_logprob,
        logprob_start_len,
        top_logprobs_num,
        return_text_in_logprobs,
        str,    # dtype=str：强制字符串约束
        None,
    )


def image(expr: SglExpr):
    # 将表达式包装为 SglImage 节点（多模态图像输入）
    return SglImage(expr)


def video(path: str, num_frames: int):
    # 创建 SglVideo 节点（多模态视频输入，指定路径和帧数）
    return SglVideo(path, num_frames)


def select(
    name: Optional[str] = None,
    choices: Optional[List[str]] = None,
    temperature: float = 0.0,
    choices_method: ChoicesSamplingMethod = token_length_normalized,
):
    # 创建 SglSelect 节点：从 choices 中选择最优候选
    assert choices is not None
    return SglSelect(name, choices, temperature, choices_method)


def _role_common(name: str, expr: Optional[SglExpr] = None):
    # 内部辅助函数：将可选内容包装在角色开始/结束节点之间
    if expr is None:
        # 无内容时返回空角色对（用于 with 语句块）
        return SglExprList([SglRoleBegin(name), SglRoleEnd(name)])
    else:
        # 有内容时将内容嵌入角色包装中
        return SglExprList([SglRoleBegin(name), expr, SglRoleEnd(name)])


def system(expr: Optional[SglExpr] = None):
    # 创建 system 角色块（包含可选内容）
    return _role_common("system", expr)


def user(expr: Optional[SglExpr] = None):
    # 创建 user 角色块（包含可选内容）
    return _role_common("user", expr)


def assistant(expr: Optional[SglExpr] = None):
    # 创建 assistant 角色块（包含可选内容）
    return _role_common("assistant", expr)


def system_begin():
    # system 角色开始标记（与 system_end() 配合使用，用于 with 块）
    return SglRoleBegin("system")


def system_end():
    # system 角色结束标记
    return SglRoleEnd("system")


def user_begin():
    # user 角色开始标记
    return SglRoleBegin("user")


def user_end():
    # user 角色结束标记
    return SglRoleEnd("user")


def assistant_begin():
    # assistant 角色开始标记
    return SglRoleBegin("assistant")


def assistant_end():
    # assistant 角色结束标记
    return SglRoleEnd("assistant")


def separate_reasoning(
    expr: Optional[SglExpr] = None, model_type: Optional[str] = None
):
    # 创建推理分离节点：将思维链（reasoning）与最终答案分离输出
    # 通常用于 o1/DeepSeek-R1 等带推理过程的模型
    return SglExprList([expr, SglSeparateReasoning(model_type, expr=expr)])
