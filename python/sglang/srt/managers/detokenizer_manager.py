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
"""DetokenizerManager is a process that detokenizes the token ids."""
# 模块文档：DetokenizerManager 是一个独立子进程，负责将 token id 序列反向解码为文本字符串
# 数据流：Scheduler → (ZMQ PUSH/PULL) → DetokenizerManager → (ZMQ PUSH/PULL) → TokenizerManager

# ----------------------------------------------------------------
# 标准库导入
# ----------------------------------------------------------------
import dataclasses  # dataclasses — 提供 @dataclass 装饰器，简化数据类定义
import logging      # logging — 运行时日志记录
import os           # os — 操作系统接口，读取环境变量
import signal       # signal — 信号处理，异常时向父进程发送 SIGQUIT
from collections import OrderedDict, defaultdict  # OrderedDict — 有序字典（LRU 驱逐基础）；defaultdict — 带默认值的字典
from typing import Dict, List, Optional, Tuple, Union  # 类型注解工具

# ----------------------------------------------------------------
# 第三方库导入
# ----------------------------------------------------------------
import psutil        # psutil — 进程/系统信息，获取父进程句柄
import setproctitle  # setproctitle — 设置进程名称，便于 top/htop 识别
import zmq           # zmq — ZeroMQ，进程间通信

# ----------------------------------------------------------------
# SGLang 内部模块导入
# ----------------------------------------------------------------
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX  # 健康检查请求 ID 前缀常量
from sglang.srt.environ import envs                       # 环境变量配置中心
from sglang.srt.managers.io_struct import (
    BatchEmbeddingOutput,   # Embedding 模型批次输出（无需 detokenize，直接透传）
    BatchStrOutput,         # 字符串批次输出，发回 TokenizerManager
    BatchTokenIDOutput,     # Token ID 批次输出，来自 Scheduler
    FreezeGCReq,            # 冻结 GC 请求
)
from sglang.srt.managers.multi_tokenizer_mixin import MultiHttpWorkerDetokenizerMixin  # 多 HTTP worker 支持 mixin
from sglang.srt.observability.cpu_monitor import start_cpu_monitor_thread  # CPU 使用率监控线程
from sglang.srt.server_args import PortArgs, ServerArgs  # 服务器启动参数 / 端口参数
from sglang.srt.utils import (
    configure_logger,            # 配置 logger 格式和级别
    freeze_gc,                   # 冻结 Python GC（减少 STW 停顿）
    kill_itself_when_parent_died,  # 父进程退出时自动终止本进程
)
from sglang.srt.utils.hf_transformers_utils import get_tokenizer   # 加载 HuggingFace tokenizer
from sglang.srt.utils.network import get_zmq_socket                # 封装 ZMQ socket 创建
from sglang.srt.utils.watchdog import Watchdog                     # 软看门狗，检测进程卡死
from sglang.utils import (
    TypeBasedDispatcher,      # 按消息类型自动路由到对应处理函数的分发器
    find_printable_text,      # 从字节序列中提取可打印的 UTF-8 子串（处理流式不完整字符）
    get_exception_traceback,  # 获取格式化异常堆栈字符串
)

# ----------------------------------------------------------------
# 模块级 logger
# ----------------------------------------------------------------
logger = logging.getLogger(__name__)

# Maximum number of request states that detokenizer can hold. When exceeded,
# oldest request states will be evicted. Default: 65536 (1<<16).
# For more details, see: https://github.com/sgl-project/sglang/issues/2812
# Use power of 2 values for better memory allocation.
# DetokenizerManager 最多保留的请求状态数量；超过时驱逐最旧记录，防止内存无限增长
DETOKENIZER_MAX_STATES = int(os.environ.get("SGLANG_DETOKENIZER_MAX_STATES", 1 << 16))


# ----------------------------------------------------------------
# DecodeStatus — 单个请求的增量解码状态
# ----------------------------------------------------------------
@dataclasses.dataclass
class DecodeStatus:
    """Store the status of incremental decoding."""
    # 用于增量解码：每次只解码新增的 token，拼接到历史文本上

    decoded_text: str       # 已确认输出的文本（不含待确认尾部）
    decode_ids: List[int]   # 累积的所有 token id 列表
    surr_offset: int        # "surrogate" 起始偏移：处理 UTF-8 多字节字符时的回溯点
    read_offset: int        # 已读（可安全输出）的 token 偏移
    # Offset that's sent to tokenizer for incremental update.
    sent_offset: int = 0    # 已发送给上游的文本长度偏移（实现增量推送）


# ----------------------------------------------------------------
# DetokenizerManager — 主类
# ----------------------------------------------------------------
# DetokenizerManager 继承 MultiHttpWorkerDetokenizerMixin，
# 支持多 HTTP worker 场景下的 detokenize 分发
class DetokenizerManager(MultiHttpWorkerDetokenizerMixin):
    """DetokenizerManager is a process that detokenizes the token ids."""

    def __init__(
        self,
        server_args: ServerArgs,  # 全局服务器启动参数
        port_args: PortArgs,      # 各 ZMQ 通道端口/路径参数
    ):
        # 初始化 ZMQ 收发通道
        self.init_ipc_channels(port_args)

        # 初始化 tokenizer（用于将 token id 解码为文本）
        self.init_tokenizer(server_args)

        # 初始化运行状态（decode_status 字典、watchdog 等）
        self.init_running_status(server_args)

        # 初始化消息类型分发器
        self.init_request_dispatcher()

    # ------------------------------------------------------------------
    # init_ipc_channels — 建立与 Scheduler / TokenizerManager 的 ZMQ 通道
    # ------------------------------------------------------------------
    def init_ipc_channels(self, port_args: PortArgs):
        context = zmq.Context(2)  # 创建 ZMQ Context，2 个 I/O 线程
        self.recv_from_scheduler = get_zmq_socket(
            context, zmq.PULL, port_args.detokenizer_ipc_name, True
            # PULL socket，绑定（bind=True）到 detokenizer_ipc_name，接收 Scheduler 的 BatchTokenIDOutput
        )
        self.send_to_tokenizer = get_zmq_socket(
            context, zmq.PUSH, port_args.tokenizer_ipc_name, False
            # PUSH socket，连接（bind=False）到 tokenizer_ipc_name，把 BatchStrOutput 发回主进程
        )

    # ------------------------------------------------------------------
    # init_tokenizer — 加载 tokenizer 实例
    # ------------------------------------------------------------------
    def init_tokenizer(self, server_args: ServerArgs):
        if server_args.skip_tokenizer_init:
            self.tokenizer = None  # 跳过 tokenizer 初始化（纯 token id 透传模式）
        else:
            self.tokenizer = get_tokenizer(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                tokenizer_backend=server_args.tokenizer_backend,
            )  # 加载 HuggingFace 兼容的 tokenizer

    # ------------------------------------------------------------------
    # init_running_status — 初始化运行时状态
    # ------------------------------------------------------------------
    def init_running_status(self, server_args: ServerArgs):
        self.decode_status = LimitedCapacityDict(capacity=DETOKENIZER_MAX_STATES)
        # decode_status: rid → DecodeStatus，记录每个请求的增量解码进度
        # 使用 LimitedCapacityDict 防止内存无限增长

        self.disable_tokenizer_batch_decode = server_args.disable_tokenizer_batch_decode
        # 是否禁用 batch_decode（某些特殊 tokenizer 如 gpt-oss 需要逐条解码）

        self.is_tool_call_parser_gpt_oss = server_args.tool_call_parser == "gpt-oss"
        # 是否使用 gpt-oss 工具调用解析器（影响 stop token 剪裁逻辑）

        self.soft_watchdog = Watchdog.create(
            debug_name="DetokenizerManager",
            watchdog_timeout=server_args.soft_watchdog_timeout,
            soft=True,                                          # 软看门狗：超时只打印告警，不强制终止
            test_stuck_time=envs.SGLANG_TEST_STUCK_DETOKENIZER.get(),  # 测试用：人为制造卡死场景
        )

        if server_args.enable_metrics:
            start_cpu_monitor_thread("detokenizer")  # 启动 CPU 监控后台线程，上报 detokenizer CPU 使用率

    # ------------------------------------------------------------------
    # init_request_dispatcher — 注册消息类型 → 处理函数的映射
    # ------------------------------------------------------------------
    def init_request_dispatcher(self):
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (BatchEmbeddingOutput, self.handle_batch_embedding_out),  # Embedding 输出 → 直接透传
                (BatchTokenIDOutput, self.handle_batch_token_id_out),     # Token ID 输出 → detokenize
                (FreezeGCReq, self.handle_freeze_gc_req),                 # GC 冻结请求 → 调用 freeze_gc
            ]
        )

    # ------------------------------------------------------------------
    # event_loop — 主事件循环（单 HTTP worker 模式）
    # ------------------------------------------------------------------
    def event_loop(self):
        """The event loop that handles requests"""
        # DetokenizerManager 子进程的主循环
        # 职责：从 Scheduler 收 token_ids → detokenize → 发给 TokenizerManager
        while True:
            with self.soft_watchdog.disable():
                recv_obj = self.recv_from_scheduler.recv_pyobj()
                # 阻塞等待 Scheduler 发来的 BatchTokenIDOutput（或其他类型消息）
            output = self._request_dispatcher(recv_obj)
            # 按类型分发处理；BatchTokenIDOutput → handle_batch_token_id_out() 完成增量 decode
            if output is not None:
                self.send_to_tokenizer.send_pyobj(output)
                # 把 BatchStrOutput（含各请求的增量文字）发回主进程的 TokenizerManager
            self.soft_watchdog.feed()  # 喂狗，重置看门狗计时器

    # ------------------------------------------------------------------
    # trim_matched_stop — 裁剪命中的 stop string 或 stop token
    # ------------------------------------------------------------------
    def trim_matched_stop(
        self, output: Union[str, List[int]], finished_reason: Dict, no_stop_trim: bool
    ):
        if no_stop_trim or not finished_reason:
            return output  # 不裁剪：用户明确禁用或请求未结束

        matched = finished_reason.get("matched", None)
        if not matched:
            return output  # 没有命中任何 stop，直接返回

        # TODO(lmzheng): handle the case where multiple stop strs are hit

        # Trim stop str（文本模式裁剪）
        if isinstance(matched, str) and isinstance(output, str):
            pos = output.find(matched)            # 找到 stop 字符串在输出中的位置
            return output[:pos] if pos != -1 else output  # 截掉 stop 字符串及其后内容

        # Trim stop token（token id 模式裁剪）
        if isinstance(matched, int) and isinstance(output, list):
            # 200012 <|call|> is the tool call token and one of eos tokens for gpt-oss model
            if output[-1] == 200012 and self.is_tool_call_parser_gpt_oss:
                return output  # gpt-oss 工具调用 token 保留，不裁剪
            assert len(output) > 0
            # NOTE: We can always assume the last token is the matched stop token
            return output[:-1]  # 去掉最后一个 stop token
        return output

    # ------------------------------------------------------------------
    # handle_batch_embedding_out — 处理 Embedding 输出（直接透传）
    # ------------------------------------------------------------------
    def handle_batch_embedding_out(self, recv_obj: BatchEmbeddingOutput):
        # If it is embedding model, no detokenization is needed.
        return recv_obj  # Embedding 结果不需要 detokenize，原样返回给 TokenizerManager

    # ------------------------------------------------------------------
    # _grouped_batch_decode — 按 (skip_special_tokens, spaces_between_special_tokens) 分组批量解码
    # ------------------------------------------------------------------
    def _grouped_batch_decode(
        self,
        ids_list: List[List[int]],  # 待解码的 token id 列表（每个元素对应一个请求）
        skip_list: List[bool],      # 每个请求是否跳过特殊 token
        space_list: List[bool],     # 每个请求是否在特殊 token 间插入空格
    ) -> List[str]:
        """Batch decode with grouping by (skip_special_tokens, spaces_between_special_tokens)."""

        # fast path：所有请求参数相同时，直接一次性 batch_decode
        first_skip, first_space = skip_list[0], space_list[0]
        if all(
            s == first_skip and sp == first_space
            for s, sp in zip(skip_list, space_list)
        ):
            return self.tokenizer.batch_decode(
                ids_list,
                skip_special_tokens=first_skip,
                spaces_between_special_tokens=first_space,
            )

        # Group indices by (skip, space) tuple
        # 参数不同时，按参数组合分组，各组分别调用 batch_decode
        groups: Dict[Tuple[bool, bool], List[int]]
        groups = defaultdict(list)
        for idx, (skip, space) in enumerate(zip(skip_list, space_list)):
            groups[(skip, space)].append(idx)  # 将请求下标按参数组合分组

        # Decode each group and collect results
        results: List[str] = [""] * len(ids_list)  # 预分配结果列表
        for (skip, space), indices in groups.items():
            decoded = self.tokenizer.batch_decode(
                [ids_list[idx] for idx in indices],  # 取出该组的 token id 列表
                skip_special_tokens=skip,
                spaces_between_special_tokens=space,
            )
            for idx, text in zip(indices, decoded):
                results[idx] = text  # 将解码结果写回对应位置

        return results

    # ------------------------------------------------------------------
    # _decode_batch_token_id_output — 核心增量解码逻辑
    # ------------------------------------------------------------------
    def _decode_batch_token_id_output(self, recv_obj: BatchTokenIDOutput):
        bs = len(recv_obj.rids)  # 本批次请求数量

        # ---- 第一步：初始化或更新 decode_status ----
        read_ids, surr_ids = [], []
        for i in range(bs):
            rid = recv_obj.rids[i]  # 请求 ID
            if rid not in self.decode_status:
                # 新请求：创建 DecodeStatus 并初始化
                s = DecodeStatus(
                    decoded_text=recv_obj.decoded_texts[i],  # 来自 prefill 阶段已解码的前缀文本
                    decode_ids=recv_obj.decode_ids[i],       # 初始 token id 列表
                    surr_offset=0,
                    read_offset=recv_obj.read_offsets[i],    # 初始已读偏移
                )
                self.decode_status[rid] = s
            else:
                # 已有请求：追加新产生的 token ids
                s = self.decode_status[rid]
                s.decode_ids.extend(recv_obj.decode_ids[i])

            # read_ids：从 surr_offset 到末尾，可能已裁掉 stop token
            read_ids.append(
                self.trim_matched_stop(
                    s.decode_ids[s.surr_offset :],
                    recv_obj.finished_reasons[i],
                    recv_obj.no_stop_trim[i],
                )
            )
            # surr_ids：surr_offset 到 read_offset，用于回溯处理多字节字符
            surr_ids.append(s.decode_ids[s.surr_offset : s.read_offset])

        # ---- 第二步：将 token id 批量解码为文本 ----
        if not self.disable_tokenizer_batch_decode:
            # 使用 batch_decode（效率更高）
            surr_texts = self._grouped_batch_decode(
                surr_ids,
                recv_obj.skip_special_tokens,
                recv_obj.spaces_between_special_tokens,
            )
            read_texts = self._grouped_batch_decode(
                read_ids,
                recv_obj.skip_special_tokens,
                recv_obj.spaces_between_special_tokens,
            )
        else:
            # Do not use batch decode to prevent some detokenization edge cases (e.g., gpt-oss).
            # 逐条解码，避免 batch_decode 在特殊 tokenizer 下产生边界错误
            surr_texts = [
                self.tokenizer.decode(
                    surr, skip_special_tokens=skip, spaces_between_special_tokens=space
                )
                for surr, skip, space in zip(
                    surr_ids,
                    recv_obj.skip_special_tokens,
                    recv_obj.spaces_between_special_tokens,
                )
            ]
            read_texts = [
                self.tokenizer.decode(
                    read, skip_special_tokens=skip, spaces_between_special_tokens=space
                )
                for read, skip, space in zip(
                    read_ids,
                    recv_obj.skip_special_tokens,
                    recv_obj.spaces_between_special_tokens,
                )
            ]

        # ---- 第三步：计算增量文本并更新状态 ----
        output_strs = []
        for i in range(bs):
            rid = recv_obj.rids[i]
            try:
                s = self.decode_status[rid]
            except KeyError:
                # 状态被 LRU 驱逐，抛出友好错误，提示用户调大 DETOKENIZER_MAX_STATES
                raise RuntimeError(
                    f"Decode status not found for request {rid}. "
                    "It may be due to the request being evicted from the decode status due to memory pressure. "
                    "Please increase the maximum number of requests by setting "
                    "the SGLANG_DETOKENIZER_MAX_STATES environment variable to a bigger value than the default value. "
                    f"The current value is {DETOKENIZER_MAX_STATES}. "
                    "For more details, see: https://github.com/sgl-project/sglang/issues/2812"
                )
            new_text = read_texts[i][len(surr_texts[i]) :]
            # new_text：read_text 去掉 surr_text 前缀，即本轮新增的文本片段

            if recv_obj.finished_reasons[i] is None:
                # 请求仍在生成中（streaming 场景）
                if new_text and not new_text.endswith("�"):
                    # 新文本合法（无 UTF-8 不完整尾部）：提交到 decoded_text，更新偏移
                    s.decoded_text += new_text
                    s.surr_offset = s.read_offset
                    s.read_offset = len(s.decode_ids)
                    new_text = ""  # 已提交，清空临时 new_text
                else:
                    # 末尾有不可打印字符（多字节字符被截断）：等待更多 token 再输出
                    new_text = find_printable_text(new_text)
            else:
                # 请求已结束：清理 decode_status 释放内存
                if rid in self.decode_status:
                    del self.decode_status[rid]

            output_str = self.trim_matched_stop(
                s.decoded_text + new_text,  # 完整输出文本 = 已提交部分 + 本轮新增部分
                recv_obj.finished_reasons[i],
                recv_obj.no_stop_trim[i],
            )

            # Incrementally send text.
            # 只发送上次发送之后新增的部分（incremental streaming）
            incremental_output = output_str[s.sent_offset :]
            s.sent_offset = len(output_str)  # 更新已发送偏移
            output_strs.append(incremental_output)

        return output_strs

    # ------------------------------------------------------------------
    # handle_batch_token_id_out — 处理来自 Scheduler 的 BatchTokenIDOutput
    # ------------------------------------------------------------------
    def handle_batch_token_id_out(self, recv_obj: BatchTokenIDOutput):
        # If handling idle batch, set output_strs to [].
        output_strs = (
            self._decode_batch_token_id_output(recv_obj)
            if len(recv_obj.rids) > 0  # 非空批次才执行解码
            else []
        )
        # 封装成 BatchStrOutput，携带所有元信息字段（logprobs / spec 统计 / token 数量等）
        return BatchStrOutput(
            rids=recv_obj.rids,
            http_worker_ipcs=recv_obj.http_worker_ipcs,
            finished_reasons=recv_obj.finished_reasons,
            output_strs=output_strs,                         # 各请求本轮增量文本
            output_ids=recv_obj.output_ids,
            prompt_tokens=recv_obj.prompt_tokens,
            reasoning_tokens=recv_obj.reasoning_tokens,
            completion_tokens=recv_obj.completion_tokens,
            cached_tokens=recv_obj.cached_tokens,
            cached_tokens_details=recv_obj.cached_tokens_details,
            spec_verify_ct=recv_obj.spec_verify_ct,
            spec_accepted_drafts=recv_obj.spec_accepted_drafts,
            spec_acceptance_histogram=recv_obj.spec_acceptance_histogram,
            input_token_logprobs_val=recv_obj.input_token_logprobs_val,
            input_token_logprobs_idx=recv_obj.input_token_logprobs_idx,
            output_token_logprobs_val=recv_obj.output_token_logprobs_val,
            output_token_logprobs_idx=recv_obj.output_token_logprobs_idx,
            input_top_logprobs_val=recv_obj.input_top_logprobs_val,
            input_top_logprobs_idx=recv_obj.input_top_logprobs_idx,
            output_top_logprobs_val=recv_obj.output_top_logprobs_val,
            output_top_logprobs_idx=recv_obj.output_top_logprobs_idx,
            input_token_ids_logprobs_val=recv_obj.input_token_ids_logprobs_val,
            input_token_ids_logprobs_idx=recv_obj.input_token_ids_logprobs_idx,
            output_token_ids_logprobs_val=recv_obj.output_token_ids_logprobs_val,
            output_token_ids_logprobs_idx=recv_obj.output_token_ids_logprobs_idx,
            output_token_entropy_val=recv_obj.output_token_entropy_val,
            output_hidden_states=recv_obj.output_hidden_states,
            routed_experts=recv_obj.routed_experts,
            customized_info=recv_obj.customized_info,
            placeholder_tokens_idx=None,   # detokenizer 不处理 placeholder token
            placeholder_tokens_val=None,
            retraction_counts=recv_obj.retraction_counts,
            token_steps=recv_obj.token_steps,
            load=recv_obj.load,
            dp_ranks=recv_obj.dp_ranks,
            time_stats=recv_obj.time_stats,
        )

    # ------------------------------------------------------------------
    # handle_freeze_gc_req — 处理 GC 冻结请求
    # ------------------------------------------------------------------
    def handle_freeze_gc_req(self, recv_req: FreezeGCReq):
        freeze_gc("Detokenizer Manager")  # 冻结 Python 垃圾回收，减少推理延迟抖动
        return None  # 无需向上游回复


# ----------------------------------------------------------------
# is_health_check_request — 判断是否为健康检查请求
# ----------------------------------------------------------------
def is_health_check_request(rid: Optional[str]) -> bool:
    # 健康检查请求的 rid 以特定前缀开头，需要特殊处理（不计入统计等）
    return isinstance(rid, str) and rid.startswith(HEALTH_CHECK_RID_PREFIX)


# ----------------------------------------------------------------
# LimitedCapacityDict — 固定容量的有序字典（LRU 驱逐）
# ----------------------------------------------------------------
class LimitedCapacityDict(OrderedDict):
    """固定容量有序字典：超过容量时自动驱逐最旧（最早插入）的条目。"""

    def __init__(self, capacity: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capacity = capacity  # 最大容量

    def __setitem__(self, key, value):
        if len(self) >= self.capacity:
            # Remove the oldest element (first item in the dict)
            self.popitem(last=False)  # last=False 驱逐最旧（FIFO 顺序）
        # Set the new item
        super().__setitem__(key, value)  # 写入新条目


# ----------------------------------------------------------------
# run_detokenizer_process — 子进程入口函数
# ----------------------------------------------------------------
def run_detokenizer_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    detokenizer_manager_class=DetokenizerManager,  # 支持依赖注入以便测试/扩展
):
    kill_itself_when_parent_died()                 # 父进程退出时自动终止，防止僵尸进程
    setproctitle.setproctitle("sglang::detokenizer")  # 设置进程名，便于 ps/top 识别
    configure_logger(server_args)                  # 配置日志格式和输出级别
    parent_process = psutil.Process().parent()     # 获取父进程句柄，异常时用于发送信号

    manager = None
    try:
        manager = detokenizer_manager_class(server_args, port_args)  # 实例化并初始化 DetokenizerManager
        if server_args.tokenizer_worker_num == 1:
            manager.event_loop()              # 单 HTTP worker 模式：简单事件循环
        else:
            manager.multi_http_worker_event_loop()  # 多 HTTP worker 模式：支持并行分发
    except Exception:
        traceback = get_exception_traceback()  # 捕获完整堆栈信息
        logger.error(f"DetokenizerManager hit an exception: {traceback}")
        if manager is not None:
            manager.maybe_clear_socket_mapping()  # 清理 socket 映射，避免资源泄漏
        parent_process.send_signal(signal.SIGQUIT)  # 通知父进程（Engine）本进程异常退出
