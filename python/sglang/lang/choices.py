# 导入抽象基类和抽象方法装饰器
from abc import ABC, abstractmethod
# 导入数据类装饰器
from dataclasses import dataclass
# 导入类型注解
from typing import Any, Dict, List, Optional

# 数值计算（argmax、均值等）
import numpy as np


# 选择操作的决策结果数据类：包含最终选中的字符串和元信息
@dataclass
class ChoicesDecision:
    decision: str                            # 最终选中的候选字符串
    meta_info: Optional[Dict[str, Any]] = None  # 附加元信息（logprob 等调试信息）


# 候选选择方法的抽象基类，所有具体选择策略均继承自此类
class ChoicesSamplingMethod(ABC):

    @property
    def requires_unconditional_logprobs(self) -> bool:
        # 默认不需要无条件 logprob（PMI 类方法需要重写为 True）
        return False

    @abstractmethod
    def __call__(
        self,
        *,
        choices: List[str],                                        # 候选字符串列表
        normalized_prompt_logprobs: List[float],                   # 各候选的归一化 logprob
        input_token_logprobs: List[List[Any]],                     # 各候选 input token 的 logprob 列表
        output_token_logprobs: List[List[Any]],                    # 各候选 output token 的 logprob 列表
        unconditional_token_logprobs: Optional[List[List[Any]]] = None,  # 无条件 logprob（可选）
    ) -> ChoicesDecision: ...


# 策略一：按 token 长度归一化的 logprob 选择最优候选
class TokenLengthNormalized(ChoicesSamplingMethod):

    def __call__(
        self,
        *,
        choices: List[str],
        normalized_prompt_logprobs: List[float],
        input_token_logprobs: List[List[Any]],
        output_token_logprobs: List[List[Any]],
        unconditional_token_logprobs: Optional[List[List[Any]]] = None,
    ) -> ChoicesDecision:
        """Select the option with the highest token length normalized prompt logprob."""
        # 选择归一化 logprob 最高的候选
        best_choice = choices[np.argmax(normalized_prompt_logprobs)]
        # 将详细 logprob 信息打包为元信息返回
        meta_info = {
            "normalized_prompt_logprobs": normalized_prompt_logprobs,
            "input_token_logprobs": input_token_logprobs,
            "output_token_logprobs": output_token_logprobs,
        }
        return ChoicesDecision(decision=best_choice, meta_info=meta_info)


# 全局单例：token 长度归一化选择方法
token_length_normalized = TokenLengthNormalized()


# 策略二：逐 token 贪婪选择（处理候选之间存在公共前缀的情况）
class GreedyTokenSelection(ChoicesSamplingMethod):

    def __call__(
        self,
        *,
        choices: List[str],
        normalized_prompt_logprobs: List[float],
        input_token_logprobs: List[List[Any]],
        output_token_logprobs: List[List[Any]],
        unconditional_token_logprobs: Optional[List[List[Any]]] = None,
    ) -> ChoicesDecision:
        """Select the option based on greedy logprob selection. For overlapping options
        where one option is a subset of a longer option, extend the shorter option using
        its average logprob for comparison against the longer option."""

        num_options = len(choices)
        # 获取所有候选中 token 数量最多的长度
        max_tokens = max(len(option) for option in input_token_logprobs)
        # 构建 logprob 矩阵（num_options × max_tokens），短候选用均值填充
        logprob_matrix = self._build_logprob_matrix(
            input_token_logprobs, max_tokens, num_options
        )
        # 执行逐列贪婪筛选
        remaining = self._greedy_selection(logprob_matrix, num_options, max_tokens)

        # 取最终剩余候选中的第一个（贪婪唯一最优）
        best_choice = choices[remaining[0]]
        meta_info = {
            "normalized_prompt_logprobs": normalized_prompt_logprobs,
            "input_token_logprobs": input_token_logprobs,
            "output_token_logprobs": output_token_logprobs,
            "greedy_logprob_matrix": logprob_matrix.tolist(),
        }
        return ChoicesDecision(decision=best_choice, meta_info=meta_info)

    def _build_logprob_matrix(self, input_token_logprobs, max_tokens, num_options):
        # 初始化全零矩阵（num_options × max_tokens）
        logprob_matrix = np.zeros((num_options, max_tokens))
        for i, option in enumerate(input_token_logprobs):
            # 提取各 token 的 logprob（每个元素为 (logprob, token_id, ...) 元组）
            actual_logprobs = [token[0] for token in option]
            # 计算均值，用于填充短候选的空余列
            avg_logprob = np.mean(actual_logprobs)
            logprob_matrix[i, : len(option)] = actual_logprobs
            # 短候选用均值补齐（等效于假设后续 token 与均值持平）
            if len(option) < max_tokens:
                logprob_matrix[i, len(option) :] = avg_logprob
        return logprob_matrix

    def _greedy_selection(self, logprob_matrix, num_options, max_tokens):
        # 初始所有候选均有效
        remaining = np.arange(num_options)
        # 逐列（token step）比较，保留当前列 logprob 最高的候选
        for j in range(max_tokens):
            max_logprob = np.max(logprob_matrix[remaining, j])
            remaining = remaining[logprob_matrix[remaining, j] == max_logprob]
            # 若唯一候选则提前退出
            if len(remaining) == 1:
                break
        return remaining


# 全局单例：贪婪 token 选择方法
greedy_token_selection = GreedyTokenSelection()


# 策略三：用无条件 logprob 归一化的 PMI 方法（Pointwise Mutual Information）
class UnconditionalLikelihoodNormalized(ChoicesSamplingMethod):

    @property
    def requires_unconditional_logprobs(self) -> bool:
        # PMI 方法需要无条件 logprob（即不考虑 prompt 时候选的 logprob）
        return True

    def __call__(
        self,
        *,
        choices: List[str],
        normalized_prompt_logprobs: List[float],
        input_token_logprobs: List[List[Any]],
        output_token_logprobs: List[List[Any]],
        unconditional_token_logprobs: Optional[List[List[Any]]] = None,
    ) -> ChoicesDecision:
        """Select the option with the highest average token logprob once normalized by
        the unconditional token logprobs.

        The first unconditional token logprob is assumed to be None. If so, it is
        replaced with 0 for the purposes of normalization."""

        if unconditional_token_logprobs is None:
            raise ValueError(
                "Unconditional token logprobs are required for this method."
            )

        # 计算条件 logprob 与无条件 logprob 的差值均值（PMI 近似）
        normalized_unconditional_prompt_logprobs = self._normalize_logprobs(
            input_token_logprobs, unconditional_token_logprobs
        )

        # 选择 PMI 得分最高的候选
        best_choice = choices[np.argmax(normalized_unconditional_prompt_logprobs)]
        meta_info = {
            "normalized_prompt_logprobs": normalized_prompt_logprobs,
            "input_token_logprobs": input_token_logprobs,
            "output_token_logprobs": output_token_logprobs,
            "unconditional_token_logprobs": unconditional_token_logprobs,
            "normalized_unconditional_prompt_logprobs": normalized_unconditional_prompt_logprobs,
        }
        return ChoicesDecision(decision=best_choice, meta_info=meta_info)

    def _normalize_logprobs(self, input_token_logprobs, unconditional_token_logprobs):
        # 对每个候选计算条件 logprob 均值减无条件 logprob 均值
        normalized_unconditional_prompt_logprobs = []
        for inputs, unconditionals in zip(
            input_token_logprobs, unconditional_token_logprobs
        ):
            # 提取条件 logprob 数组
            inputs_logprobs = np.array([token[0] for token in inputs])
            # 提取无条件 logprob 数组
            unconditionals_logprobs = np.array([token[0] for token in unconditionals])
            # 第一个无条件 logprob 可能为 None（token healing 头部），替换为 0
            unconditionals_logprobs[0] = unconditionals_logprobs[0] or 0
            # 计算 PMI：条件 logprob - 无条件 logprob 的均值
            normalized_unconditional_prompt_logprobs.append(
                float(np.mean(inputs_logprobs - unconditionals_logprobs))
            )
        return normalized_unconditional_prompt_logprobs


# 全局单例：无条件归一化选择方法（PMI）
unconditional_likelihood_normalized = UnconditionalLikelihoodNormalized()
