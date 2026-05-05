from __future__ import annotations

# 导入抽象基类和抽象方法装饰器，用于定义投机解码 Worker 的接口规范
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

# 仅在类型检查阶段导入 TpModelWorker，避免循环依赖
if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker


# 草稿 Worker 抽象基类：定义投机解码中草稿模型的公共接口
class BaseDraftWorker(ABC):
    # 生成草稿 token（prefill 阶段），子类必须实现
    @abstractmethod
    def draft():
        pass

    # 在已有上下文基础上扩展草稿 token（decode 阶段），子类必须实现
    @abstractmethod
    def draft_extend():
        pass


# 投机解码 Worker 抽象基类：封装目标模型与草稿模型的协作接口
class BaseSpecWorker(ABC):
    # 目标模型 Worker（验证草稿 token 的主模型）的属性访问器
    @property
    @abstractmethod
    def target_worker(self) -> TpModelWorker:
        pass

    # 草稿模型 Worker（生成候选 token 的小模型）的属性访问器
    @property
    @abstractmethod
    def draft_worker(self) -> BaseDraftWorker:
        pass

    # 清理 KV 缓存池，释放投机解码过程中占用的显存资源
    @abstractmethod
    def clear_cache_pool(self):
        # TODO: move this abstract method to BaseTpWorker and call through self.model_runner
        pass
