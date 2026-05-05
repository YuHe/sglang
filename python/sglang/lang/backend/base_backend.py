# 引入类型注解支持
from typing import List, Optional, Union

# 导入聊天模板工具函数
from sglang.lang.chat_template import get_chat_template
# 导入选择决策和采样方法类型
from sglang.lang.choices import ChoicesDecision, ChoicesSamplingMethod
# 导入流式执行器（用于程序执行上下文）
from sglang.lang.interpreter import StreamExecutor
# 导入采样参数 IR 节点
from sglang.lang.ir import SglSamplingParams


# 所有后端的抽象基类，定义了统一的接口规范
class BaseBackend:
    def __init__(self) -> None:
        # 默认不支持 concatenate_and_append 操作（仅 runtime 后端支持）
        self.support_concate_and_append = False
        # 初始化默认聊天模板
        self.chat_template = get_chat_template("default")

    def get_model_name(self):
        # 子类必须实现：返回当前后端使用的模型名称
        raise NotImplementedError()

    def get_chat_template(self):
        # 返回当前后端使用的聊天模板
        return self.chat_template

    def cache_prefix(self, prefix_str: str):
        # 缓存前缀字符串（用于 KV cache 预填充优化，默认空实现）
        pass

    def uncache_prefix(self, rid: str):
        # 根据请求 ID 取消前缀缓存（默认空实现）
        pass

    def end_request(self, rid: Union[str, List[str]]):
        # 结束一个或多个请求（释放资源，默认空实现）
        pass

    def begin_program(self, s: StreamExecutor):
        # 程序开始时的钩子，可用于初始化会话状态（默认空实现）
        pass

    def end_program(self, s: Union[StreamExecutor, List[StreamExecutor]]):
        # 程序结束时的钩子，可用于清理会话资源（默认空实现）
        pass

    def commit_lazy_operations(self, s: StreamExecutor):
        # 提交延迟操作（如批量填充），将挂起的操作发送给后端（默认空实现）
        pass

    def fork_program(
        self,
        src: StreamExecutor,
        dst: List[StreamExecutor],
        position_ids_offset: Optional[List[int]] = None,
    ):
        # 将一个执行器 fork 为多个并行分支（用于并行采样/选择，默认空实现）
        pass

    def fill_image(self, s: StreamExecutor):
        # 向后端填充图像数据（多模态场景，默认空实现）
        pass

    def generate(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        # 子类必须实现：同步生成文本（一次性返回完整结果）
        raise NotImplementedError()

    def generate_stream(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        # 子类必须实现：流式生成文本（逐 token 返回）
        raise NotImplementedError()

    def select(
        self,
        s: StreamExecutor,
        choices: List[str],
        temperature: float,
        choices_method: Optional[ChoicesSamplingMethod] = None,
    ) -> ChoicesDecision:
        # 子类必须实现：从候选字符串列表中选择最优项（用于受限生成）
        raise NotImplementedError()

    def concatenate_and_append(self, src_rids: List[str], dst_rid: str):
        # 子类必须实现：将多个请求的 KV cache 拼接并追加到目标请求（仅 runtime 支持）
        raise NotImplementedError()

    def shutdown(self):
        # 关闭后端，释放所有资源（默认空实现）
        pass

    def flush_cache(self):
        # 清空后端 KV cache（默认空实现）
        pass

    def get_server_info(self):
        # 获取后端服务器信息（默认空实现）
        pass
