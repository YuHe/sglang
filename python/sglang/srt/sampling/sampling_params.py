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
"""Sampling parameters for text generation."""
# 本模块定义文本生成时使用的采样参数，包括温度、top-p、stop 条件等

import logging
from typing import Any, Dict, List, Optional, Union

# sre_parse 在 Python 3.11+ 已被标记为废弃，优先从 re._parser 导入
try:
    import re._parser as sre_parse  # Python >= 3.11
except ImportError:
    import sre_parse  # Python < 3.11

# 判断浮点参数"几乎为零"的阈值，用于 temperature ≈ 0 → greedy 的特殊处理
_SAMPLING_EPS = 1e-6
# top_k 取值为此常量时表示"全词表"，即不做 top-k 截断（1<<30 ≈ 10亿）
TOP_K_ALL = 1 << 30

logger = logging.getLogger(__name__)


class SamplingParams:
    """
    The sampling parameters.

    See docs/backend/sampling_params.md or
    https://docs.sglang.io/backend/sampling_params.html
    for the documentation.
    """

    def __init__(
        self,
        max_new_tokens: int = 128,            # 最多生成的 token 数量
        stop: Optional[Union[str, List[str]]] = None,  # 遇到这些字符串时停止生成
        stop_token_ids: Optional[List[int]] = None,    # 遇到这些 token id 时停止生成
        stop_regex: Optional[Union[str, List[str]]] = None,  # 遇到匹配该正则的文本时停止
        temperature: float = 1.0,             # 采样温度；越高越随机，越低越确定
        top_p: float = 1.0,                   # nucleus sampling：只从累积概率 ≤ top_p 的词中采样
        top_k: int = -1,                      # 只从概率最高的 top_k 个词中采样；-1 表示不限制
        min_p: float = 0.0,                   # 最低采样概率阈值（相对最大概率的比例）
        frequency_penalty: float = 0.0,       # 频率惩罚：对已出现词按出现次数降低 logit
        presence_penalty: float = 0.0,        # 存在惩罚：对已出现过的词固定降低 logit
        repetition_penalty: float = 1.0,      # 重复惩罚因子：>1 抑制重复，<1 鼓励重复
        min_new_tokens: int = 0,              # 至少生成这么多 token，忽略中途的 stop 条件
        n: int = 1,                           # 每个请求并行生成的候选序列数量
        json_schema: Optional[str] = None,    # 结构化输出：限定输出符合该 JSON Schema
        regex: Optional[str] = None,          # 结构化输出：限定输出匹配该正则表达式
        ebnf: Optional[str] = None,           # 结构化输出：限定输出符合该 EBNF 语法
        structural_tag: Optional[str] = None, # 结构化输出标签（与 json_schema/regex/ebnf 配合）
        ignore_eos: bool = False,             # 是否忽略 EOS token（继续生成直到 max_new_tokens）
        skip_special_tokens: bool = True,     # 解码时是否跳过特殊 token（如 <s>、</s>）
        spaces_between_special_tokens: bool = True,  # 特殊 token 之间是否插入空格
        no_stop_trim: bool = False,           # 若为 True，不从输出末尾裁剪匹配到的 stop 字符串
        custom_params: Optional[Dict[str, Any]] = None,  # 用户自定义扩展参数（透传给插件等）
        stream_interval: Optional[int] = None,           # 流式输出时每隔多少 token 推送一次
        logit_bias: Optional[Dict[str, float]] = None,   # token_id → 偏置值，直接加到对应 logit 上
        sampling_seed: Optional[int] = None,             # 随机采样的种子，保证可复现
    ) -> None:
        # 对非 Optional 参数：若调用方（如 /generate 接口）传入 null，则视为使用默认值，
        # 避免后续 verify() 报错
        self.max_new_tokens = max_new_tokens

        # stop_strs：停止字符串列表，直接存储（可能是 None、str 或 List[str]）
        self.stop_strs = stop

        # stop_token_ids：过滤掉 None 值并转为整数集合；若为空则置 None
        if stop_token_ids:
            filtered = {int(t) for t in stop_token_ids if t is not None}
            self.stop_token_ids = filtered or None
        else:
            self.stop_token_ids = None

        # stop_regex_strs：停止正则列表，直接存储
        self.stop_regex_strs = stop_regex

        # 各采样超参，若外部传入 None 则使用硬编码默认值
        self.temperature = temperature if temperature is not None else 1.0
        self.top_p = top_p if top_p is not None else 1.0
        self.top_k = top_k if top_k is not None else -1
        self.min_p = min_p if min_p is not None else 0.0
        self.frequency_penalty = (
            frequency_penalty if frequency_penalty is not None else 0.0
        )
        self.presence_penalty = (
            presence_penalty if presence_penalty is not None else 0.0
        )
        self.repetition_penalty = (
            repetition_penalty if repetition_penalty is not None else 1.0
        )
        self.min_new_tokens = min_new_tokens if min_new_tokens is not None else 0

        # 结构化输出参数
        self.regex = regex
        self.n = n if n is not None else 1
        self.json_schema = json_schema
        self.ebnf = ebnf
        self.structural_tag = structural_tag

        # 解码行为控制
        self.ignore_eos = ignore_eos if ignore_eos is not None else False
        self.skip_special_tokens = (
            skip_special_tokens if skip_special_tokens is not None else True
        )
        self.spaces_between_special_tokens = (
            spaces_between_special_tokens
            if spaces_between_special_tokens is not None
            else True
        )
        self.no_stop_trim = no_stop_trim if no_stop_trim is not None else False

        # 其余杂项参数
        self.custom_params = custom_params
        self.stream_interval = stream_interval
        self.logit_bias = logit_bias
        self.sampling_seed = sampling_seed

        # ---- 特殊情况后处理 ----
        if 0 <= self.temperature < _SAMPLING_EPS:
            # temperature ≈ 0 等价于 greedy decoding：
            # 将 temperature 重置为 1.0（避免除零），同时设 top_k=1 强制取最高概率词
            self.temperature = 1.0
            self.top_k = 1
        if self.top_k == -1:
            # top_k == -1 是"不限制"的语义，内部统一替换为全词表大小常量
            self.top_k = TOP_K_ALL  # whole vocabulary

    def verify(self, vocab_size):
        """校验采样参数的合法性，非法时抛出 ValueError"""

        # temperature 不能为负
        if self.temperature < 0.0:
            raise ValueError(
                f"temperature must be non-negative, got {self.temperature}."
            )
        # top_p 必须在 (0, 1] 范围内
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        # min_p 必须在 [0, 1] 范围内
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}.")
        # top_k 必须 ≥ 1 或等于 -1（-1 在 __init__ 中已被替换，此处作为防御性检查）
        if self.top_k < 1 or self.top_k == -1:
            raise ValueError(
                f"top_k must be -1 (disable) or at least 1, got {self.top_k}."
            )
        # frequency_penalty 合法范围 [-2, 2]
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError(
                "frequency_penalty must be in [-2, 2], got "
                f"{self.frequency_penalty}."
            )
        # presence_penalty 合法范围 [-2, 2]
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError(
                "presence_penalty must be in [-2, 2], got " f"{self.presence_penalty}."
            )
        # repetition_penalty 合法范围 [0, 2]
        if not 0.0 <= self.repetition_penalty <= 2.0:
            raise ValueError(
                "repetition_penalty must be in [0, 2], got "
                f"{self.repetition_penalty}."
            )
        # min_new_tokens 不能为负
        if not 0 <= self.min_new_tokens:
            raise ValueError(
                f"min_new_tokens must be in [0, max_new_tokens], got "
                f"{self.min_new_tokens}."
            )
        # max_new_tokens 若非 None，必须 ≥ 0 且 ≥ min_new_tokens
        if self.max_new_tokens is not None:
            if self.max_new_tokens < 0:
                raise ValueError(
                    f"max_new_tokens must be at least 0, got {self.max_new_tokens}."
                )
            if not self.min_new_tokens <= self.max_new_tokens:
                raise ValueError(
                    f"min_new_tokens must be in [0, max_new_tokens({self.max_new_tokens})], got "
                    f"{self.min_new_tokens}."
                )
        # logit_bias 的 key（token_id）必须在词表范围内
        if self.logit_bias is not None:
            for token_id in self.logit_bias:
                if not 0 <= int(token_id) < vocab_size:
                    raise ValueError(
                        f"logit_bias must has keys in [0, {vocab_size - 1}], got "
                        f"{token_id}."
                    )

        # json_schema / regex / ebnf 三者互斥，最多只能设置一个
        grammars = [
            self.json_schema,
            self.regex,
            self.ebnf,
        ]  # since mutually exclusive, only one can be set
        if sum(x is not None for x in grammars) > 1:
            raise ValueError("Only one of regex, json_schema, or ebnf can be set.")

    def normalize(self, tokenizer):
        """对停止条件做归一化处理，计算各 stop 匹配所需的最大缓冲 token 数"""

        # ---- 处理 stop 字符串 ----
        if self.stop_strs is None:
            # 没有 stop 字符串时，初始化为空列表，最大长度为 0
            self.stop_strs = []
            self.stop_str_max_len = 0
        else:
            # 若传入的是单个字符串，统一转为列表
            if isinstance(self.stop_strs, str):
                self.stop_strs = [self.stop_strs]

            stop_str_max_len = 0
            for stop_str in self.stop_strs:
                if tokenizer is not None:
                    # 用 tokenizer 编码 stop 字符串，得到对应 token 数量
                    stop_str_ids = tokenizer.encode(stop_str, add_special_tokens=False)
                    stop_str_max_len = max(stop_str_max_len, len(stop_str_ids))
                else:
                    # 无 tokenizer 时，用字符数作为保守估计
                    stop_str_max_len = max(stop_str_max_len, len(stop_str))
            # 记录所有 stop 字符串中最长的 token 数，用于确定解码时需要缓冲多少 token
            self.stop_str_max_len = stop_str_max_len

        # ---- 处理 stop 正则表达式 ----
        if self.stop_regex_strs is None:
            # 没有 stop 正则时，初始化为空列表，最大长度为 0
            self.stop_regex_strs = []
            self.stop_regex_max_len = 0
        else:
            # 若传入的是单个正则字符串，统一转为列表
            if isinstance(self.stop_regex_strs, str):
                self.stop_regex_strs = [self.stop_regex_strs]

            stop_regex_max_len = 0
            for stop_regex in self.stop_regex_strs:
                # 通过正则分析估算该正则最多能匹配多少个字符/token
                stop_regex_max_len = max(
                    stop_regex_max_len, get_max_seq_length(stop_regex)
                )

            # 记录所有 stop 正则中最大的匹配长度上界
            self.stop_regex_max_len = stop_regex_max_len


# 给定一个正则表达式字符串，返回它能匹配的最大字符/token 数的严格上界。
# 注意：最坏情况下，一个需要缓冲的字符对应一个 token。
def get_max_seq_length(regex_str: str):
    # 先用 sre_parse 将正则解析为 SubPattern AST，再递归计算最大长度
    return _max_length_from_subpattern(sre_parse.parse(regex_str))


# 正则匹配长度无上界时使用该常量（如 .* 或 .+）
MAX_LEN = 2**30


def _max_length_from_subpattern(subpattern: sre_parse.SubPattern):
    """
    递归遍历 sre_parse 解析出的 SubPattern AST，
    计算该子模式能匹配的最大字符数上界。
    """
    total = 0
    for token, value in subpattern:
        if token in {
            sre_parse.LITERAL,  # 单个字面字符，如 'a'，贡献长度 1
            sre_parse.IN,       # 字符集，如 [abc]，匹配其中一个字符，贡献长度 1
            sre_parse.ANY,      # 通配符 "."，匹配任意一个字符，贡献长度 1
        }:
            total += 1
        elif token == sre_parse.SUBPATTERN:
            # 捕获组，如 (a\d+)，解析结构为：
            # (SUBPATTERN, (group_id, add_flags, del_flags, inner_subpattern))
            _, _, _, inner_subpattern = value
            # 递归计算内部子模式的最大长度
            total += _max_length_from_subpattern(inner_subpattern)
        elif token == sre_parse.BRANCH:
            # 分支选择，如 a|bc，取所有分支中最长的一个
            _, branches = value
            total += max(_max_length_from_subpattern(branch) for branch in branches)
        elif token in {sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT}:
            # 量词，如 a{2,5} 或 a*（贪婪/非贪婪）
            # 解析结构为 (min_repeat, max_repeat, inner_subpattern)
            _, max_num_repeat, inner_subpattern = value
            if max_num_repeat == sre_parse.MAXREPEAT:
                # 无上界量词（如 *、+），最大长度视为无穷大，用 MAX_LEN 代替
                total += MAX_LEN
            else:
                # 有上界量词（如 {2,5}），最大长度 = 上界 × 子模式最大长度
                total += max_num_repeat * _max_length_from_subpattern(inner_subpattern)
        elif token == sre_parse.AT:
            # 零宽断言，如 ^、$、\b，不消耗任何字符，贡献长度 0
            total += 0
        else:
            # 遇到未处理的 token 类型，保守地返回 MAX_LEN
            logger.warning(f"Got unhandled regex token: {token}")
            total += MAX_LEN

    return total
