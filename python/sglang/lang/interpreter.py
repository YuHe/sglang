"""The interpreter that executes SGL programs"""
# SGL 程序的解释器：负责调度 StreamExecutor 后台线程执行 IR 节点，支持流式/非流式/批量执行

import asyncio
# contextvars：用于在子线程中复制父线程的上下文变量（Python 3.7+ 标准库）
import contextvars
import copy
import multiprocessing
import queue
import threading
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

# tqdm：进度条库，用于批量执行时显示进度
import tqdm

# 全局配置（verbosity、enable_precache_with_tracing、enable_parallel_encoding 等）
from sglang.global_config import global_config
# 导入所有需要执行的 IR 节点类型
from sglang.lang.ir import (
    SglCommitLazy,           # 提交懒执行操作节点
    SglConcateAndAppend,     # 拼接并追加 KV cache 节点
    SglConstantText,         # 常量文本节点
    SglExpr,                 # IR 节点基类
    SglExprList,             # IR 节点列表
    SglGen,                  # 文本生成节点
    SglImage,                # 图像节点
    SglRoleBegin,            # 角色开始节点
    SglRoleEnd,              # 角色结束节点
    SglSelect,               # 候选选择节点
    SglSeparateReasoning,    # 推理分离节点
    SglVariable,             # 变量引用节点
    SglVarScopeBegin,        # 变量作用域开始节点
    SglVarScopeEnd,          # 变量作用域结束节点
    SglVideo,                # 视频节点
)
# 图像/视频编码工具和异常堆栈获取工具
from sglang.utils import (
    encode_image_base64,
    encode_video_base64,
    get_exception_traceback,
)


def run_internal(state, program, func_args, func_kwargs, sync):
    # 在当前线程中实际调用用户定义的 SGL 程序函数，并在结束时通知 StreamExecutor
    try:
        # 调用用户函数（state 作为第一个参数，即 SGL 程序状态）
        state.ret_value = program.func(state, *func_args, **func_kwargs)
    except Exception as e:
        raise e
    finally:
        # 无论是否出错，都要发送结束信号给 StreamExecutor 后台线程
        state.stream_executor.end()

    if sync:
        # sync=True 时等待所有队列任务执行完毕
        state.stream_executor.sync()

    if global_config.verbosity >= 2:
        # 调试模式：打印最终生成文本
        print(state.text())


def run_program(
    program,
    backend,
    func_args,
    func_kwargs,
    default_sampling_para,
    stream,
    sync=False,
    use_thread=True,
):
    # 单个 SGL 程序的运行入口：创建 StreamExecutor 并执行用户函数
    if hasattr(backend, "endpoint"):
        # 若 backend 是 Runtime 包装器，取其内部 endpoint
        backend = backend.endpoint
    assert backend is not None, "Please specify a backend"
    # 将 bind_arguments（通过 bind() 预设的参数）合并到 func_kwargs
    func_kwargs.update(program.bind_arguments)
    # 创建后台执行器（负责队列化执行所有 IR 节点）
    stream_executor = StreamExecutor(
        backend,
        func_kwargs,
        default_sampling_para,
        chat_template=None,
        stream=stream,
        num_api_spec_tokens=program.num_api_spec_tokens,
        use_thread=use_thread,
    )
    # 将 StreamExecutor 包装为用户可见的 ProgramState
    state = ProgramState(stream_executor)

    if stream:
        # 流式模式：在子线程中异步执行，立即返回 state（用户可通过 text_iter 迭代结果）
        t = threading.Thread(
            target=run_internal, args=(state, program, func_args, func_kwargs, sync)
        )
        t.start()
        return state
    else:
        # 非流式模式：在当前线程阻塞执行，完成后返回 state
        run_internal(state, program, func_args, func_kwargs, sync)
        return state


def run_program_batch(
    program,
    backend,
    batch_arguments,
    default_sampling_para,
    num_threads,
    progress_bar,
    generator_style=False,
):
    # 批量执行 SGL 程序：支持多线程并发执行，可选前缀预缓存和生成器模式
    if hasattr(backend, "endpoint"):
        # 若 backend 是 Runtime 包装器，取其内部 endpoint
        backend = backend.endpoint

    # Pre-cache the common prefix for a batch. The prefix is extracted by tracing the program.
    # 若启用追踪预缓存且批次大于 1，先将公共前缀缓存到后端 KV cache
    if global_config.enable_precache_with_tracing and len(batch_arguments) > 1:
        cache_program(program, backend)

    # Run all programs
    # 自动计算线程数：max(96, CPU核数×16)，同时不超过批次大小
    if num_threads == "auto":
        num_threads = max(96, multiprocessing.cpu_count() * 16)
    num_threads = min(num_threads, len(batch_arguments))

    if generator_style:
        # 生成器模式：边执行边 yield 结果，避免全部等待完成
        return _run_program_batch_generator(
            program,
            backend,
            batch_arguments,
            default_sampling_para,
            num_threads,
            progress_bar,
        )

    # Original code path when generator_style=False
    # 单线程执行：逐个顺序运行（用于调试或小批次）
    if num_threads == 1:
        rets = []
        if progress_bar:
            for arguments in tqdm.tqdm(batch_arguments):
                rets.append(
                    run_program(
                        program,
                        backend,
                        (),
                        arguments,
                        default_sampling_para,
                        False,
                        True,
                    )
                )
        else:
            for arguments in batch_arguments:
                rets.append(
                    run_program(
                        program,
                        backend,
                        (),
                        arguments,
                        default_sampling_para,
                        False,
                        True,
                    )
                )
    else:
        # 多线程执行：使用 ThreadPoolExecutor 并发提交所有任务
        if progress_bar:
            pbar = tqdm.tqdm(total=len(batch_arguments))

        with ThreadPoolExecutor(num_threads) as executor:
            futures = []
            for arguments in batch_arguments:
                futures.append(
                    executor.submit(
                        run_program,
                        program,
                        backend,
                        (),
                        arguments,
                        default_sampling_para,
                        False,
                        True,
                    )
                )
                if progress_bar:
                    # 每个 future 完成时更新进度条
                    futures[-1].add_done_callback(lambda _: pbar.update())

            # 等待所有 future 完成并收集结果
            rets = [f.result() for f in futures]
        # 最后一个状态显式 sync，确保所有队列任务彻底完成
        rets[-1].sync()

        if progress_bar:
            pbar.close()

    return rets


def _run_program_batch_generator(
    program,
    backend,
    batch_arguments,
    default_sampling_para,
    num_threads,
    progress_bar,
):
    """Helper function that yields results one by one using chunking to avoid overwhelming ThreadPoolExecutor."""
    # 生成器版批量执行：逐个 yield 结果，单线程时直接顺序迭代
    if num_threads == 1:
        iterator = tqdm.tqdm(batch_arguments) if progress_bar else batch_arguments
        for arguments in iterator:
            yield run_program(
                program,
                backend,
                (),
                arguments,
                default_sampling_para,
                False,
                True,
            )
    else:
        # 多线程版：分块提交任务，避免 ThreadPoolExecutor 任务队列过长
        pbar = tqdm.tqdm(total=len(batch_arguments)) if progress_bar else None

        # Process in chunks to avoid overwhelming ThreadPoolExecutor
        # Otherwise, ThreadPoolExecutor.submit will block after adding certain number of tasks
        # so we will never reach "yield" until all tasks are done
        # 每次最多提交 200 个任务，确保可以及时 yield 结果（否则全部提交后才能 yield）
        chunk_size = 200

        with ThreadPoolExecutor(num_threads) as executor:
            for chunk_start in range(0, len(batch_arguments), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(batch_arguments))
                chunk_futures = []

                # Submit chunk of tasks
                # 提交当前块中的所有任务
                for i in range(chunk_start, chunk_end):
                    future = executor.submit(
                        run_program,
                        program,
                        backend,
                        (),
                        batch_arguments[i],
                        default_sampling_para,
                        False,
                        True,
                    )
                    if pbar:
                        future.add_done_callback(lambda _: pbar.update())
                    chunk_futures.append(future)

                # Yield results from this chunk as they complete
                # 按提交顺序 yield 当前块的结果
                for future in chunk_futures:
                    yield future.result()

        if pbar:
            pbar.close()


def cache_program(program, backend):
    # 提取程序的公共常量前缀并调用 backend.cache_prefix() 预热 KV cache
    # 只有前缀长度 > 64 个字符时才值得缓存（避免过短前缀的缓存开销）
    from sglang.lang.tracer import extract_prefix_by_tracing

    prefix = extract_prefix_by_tracing(program, backend)
    if prefix and len(prefix) > 64:
        backend.cache_prefix(prefix)


# 流式响应中需要增量合并的 meta_info 键名集合（这些字段是 list，需要拼接而非覆盖）
_INCREMENTAL_STREAMING_META_INFO_KEYS = (
    "output_token_logprobs",
    "output_top_logprobs",
    "output_token_ids_logprobs",
)


def _merge_stream_meta_info(
    pending_meta_info: dict[str, Any] | None,
    meta_info: dict[str, Any],
) -> dict[str, Any]:
    # 合并流式响应的 meta_info：对增量字段（logprob 列表）做拼接，其余字段取最新值
    if pending_meta_info is None:
        # 无历史积压时直接返回当前 meta_info
        return meta_info

    merged_meta_info = dict(meta_info)
    for key in _INCREMENTAL_STREAMING_META_INFO_KEYS:
        if key not in meta_info and key not in pending_meta_info:
            continue
        # 将历史积压列表与当前列表拼接
        merged_meta_info[key] = list(pending_meta_info.get(key, [])) + list(
            meta_info.get(key, [])
        )
    return merged_meta_info


class StreamExecutor:
    """A stream executor that executes SGL expressions in a background thread."""
    # StreamExecutor：SGL 程序的后台执行引擎，以队列 + 工作线程方式顺序处理所有 IR 节点

    def __init__(
        self,
        backend,
        arguments,
        default_sampling_para,
        chat_template,
        stream,
        num_api_spec_tokens=None,
        use_thread=True,
    ):
        from sglang.lang.backend.base_backend import BaseBackend

        # 唯一会话 ID（用于后端 KV cache 的 slot 标识）
        self.sid = uuid.uuid4().hex
        self.backend: BaseBackend = backend
        # 程序参数字典（用户传入的关键字参数）
        self.arguments: Dict[str, Any] = arguments
        # 默认采样参数（可被 sgl.gen() 的参数覆盖）
        self.default_sampling_para = default_sampling_para
        # 是否启用流式生成模式
        self.stream = stream

        self.variables = {}  # Dict[name: str -> value: str]  # 变量名 → 生成值映射
        self.variable_event = {}  # Dict[name: str -> event: threading.Event]  # 变量完成事件
        self.meta_info = {}  # Dict[name: str -> info: str]  # 变量元信息（logprob 等）
        # 执行是否已完成标志（供流式迭代器检测结束条件）
        self.is_finished = False
        # 工作线程中发生的错误（None 表示无错误）
        self.error_ = None

        # For completion
        self.text_ = ""  # The full text  # 已生成的完整文本缓冲区

        # For chat
        self.messages_ = []  # The messages in the OpenAI API format  # chat 模式下的消息列表
        # 获取聊天模板（用于插入角色前缀/后缀和 stop 字符串）
        self.chat_template = chat_template or self.backend.get_chat_template()
        # 当前活跃角色名（None 表示不在任何角色块中）
        self.cur_role = None
        # 当前角色块在 text_ 中的起始位置（用于提取角色内容）
        self.cur_role_begin_pos = None

        # For vision
        # 所有历史图像列表（path, base64）
        self.images_ = []
        # 当前角色块中的图像列表（角色结束时追加到消息）
        self.cur_images = []

        # For fork/join
        # fork 时父状态 text_ 的长度（子状态从此位置截取增量文本）
        self.fork_start_text_pos = None

        # For speculative execution
        # API 投机执行的 token 预算（None 表示不使用投机执行）
        self.num_api_spec_tokens = num_api_spec_tokens
        # 投机预取的文本缓冲（尚未消耗的推测文本）
        self.speculated_text = ""

        # Worker thread
        # use_thread=False 时同步执行（用于追踪模式或测试）
        self.use_thread = use_thread
        if self.use_thread:
            # 任务队列：submit() 向队列写入 IR 节点，工作线程从中取出执行
            self.queue = queue.Queue()

            def _run_worker_in_context():
                self._thread_worker_func()

            # 使用 contextvars.copy_context() 将父线程的上下文变量复制到工作线程
            self.worker = threading.Thread(
                target=contextvars.copy_context().run, args=(_run_worker_in_context,)
            )
            self.worker.start()

        # For streaming
        if stream:
            # 流式模式：全文流事件和各变量的流事件
            self.stream_text_event = threading.Event()
            self.stream_var_event = {}
        else:
            self.stream_text_event = None
            self.stream_var_event = None

    def submit(self, expr: SglExpr):
        # 提交 IR 节点到执行队列（同时初始化变量事件，确保 get_var 等待有效）
        self._init_var_event(expr)

        if self.use_thread:
            # 线程模式：放入队列，由工作线程异步处理
            self.queue.put(expr)
        else:
            # 同步模式：直接执行（用于追踪或单线程场景）
            self._execute(expr)

    def sync(self):
        # 等待队列中所有已提交任务执行完毕（queue.join() 阻塞直到 task_done 均调用）
        if self.use_thread:
            self.queue.join()

    def get_var(self, name):
        # 获取变量值（若变量尚未生成，阻塞等待其完成事件）
        if name in self.variable_event:
            self.variable_event[name].wait()
        return self.variables[name]

    def set_var(self, name, value):
        # 直接设置变量值（用于 separate_reasoning 等后处理操作）
        self.variables[name] = value

    def get_meta_info(self, name, timeout=None):
        # 获取变量的元信息（logprob 等），可设置超时
        if name in self.variable_event:
            got = self.variable_event[name].wait(timeout)
            if not got:
                raise TimeoutError(f"Timeout while waiting for event '{name}'")
        ret = self.meta_info.get(name, None)
        return ret

    def fork(
        self,
        size: int = 1,
        position_ids_offset: Optional[List[int]] = None,
    ):
        # fork：将当前状态复制为 size 个子执行器（用于并行分支生成）
        if size > 1 and str(self.text_):
            # 多分支时需要先提交 CommitLazy（确保 KV cache 刷新）
            self.submit(SglCommitLazy())

        # 等待当前所有操作完成，保证 fork 快照的一致性
        self.sync()
        size = int(size)

        # 创建 size 个子 StreamExecutor，共享 backend 和采样参数
        exes = [
            StreamExecutor(
                self.backend,
                self.arguments,
                self.default_sampling_para,
                self.chat_template,
                self.stream,
            )
            for _ in range(size)
        ]
        for i in range(size):
            # 复制父状态的变量、文本、消息、角色信息和图像列表
            exes[i].variables = dict(self.variables)
            exes[i].text_ = str(self.text_)
            exes[i].messages_ = list(self.messages_)
            exes[i].cur_role = self.cur_role
            exes[i].cur_role_begin_pos = self.cur_role_begin_pos
            # 记录 fork 起始位置（join 时子状态只需贡献此位置之后的增量文本）
            exes[i].fork_start_text_pos = len(self.text_)
            exes[i].images_ = list(self.images_)

            # TODO(ying): handle API speculative execution

        return exes

    def text(self):
        # 获取完整文本（先 sync 确保所有节点已执行）
        self.sync()
        return self.text_

    def messages(self):
        # 获取 OpenAI chat 格式的消息列表（先 sync）
        self.sync()
        return self.messages_

    def error(self):
        # 获取工作线程中发生的错误（先 sync）
        self.sync()
        return self.error_

    def end(self):
        # 结束执行器：向队列发送 None 哨兵值，通知工作线程退出
        if self.use_thread:
            if self.worker.is_alive():
                self.queue.put(None)
        # 通知后端释放与此 sid 关联的 KV cache slot
        self.backend.end_program(self)

    def _thread_worker_func(self):
        # 工作线程主循环：从队列取出 IR 节点逐个执行，直到收到 None 哨兵
        error = None

        while True:
            expr = self.queue.get()
            if expr is None:
                # 收到退出信号，标记任务完成后退出循环
                self.queue.task_done()
                break

            try:
                self._execute(expr)
            except Exception as e:
                # 记录错误但不重新抛出，确保队列清理能继续进行
                warnings.warn(f"Error in stream_executor: {get_exception_traceback()}")
                error = e
                break
            # 标记当前任务已完成（供 sync()/queue.join() 使用）
            self.queue.task_done()
            if self.stream_text_event:
                # 流式模式：每执行一个节点后通知文本流等待方
                self.stream_text_event.set()

        # Clean the queue and events
        # 出错时：清空队列中的剩余任务，并设置所有变量事件（解除等待方的阻塞）
        if error is not None:
            try:
                while True:
                    self.queue.task_done()
                    self.queue.get_nowait()
            except queue.Empty:
                pass
            # 设置所有变量事件，让等待中的 get_var() 调用能够返回（返回错误状态）
            for name in self.variable_event:
                self.variable_event[name].set()
            if self.stream_var_event:
                for name in self.stream_var_event:
                    self.stream_var_event[name].set()
            self.error_ = error

        if self.stream_text_event:
            # 工作线程结束时发出最后一次流事件通知，唤醒 text_iter 的等待
            self.stream_text_event.set()

        # 标记执行器已完成（text_iter/text_async_iter 用于检测结束条件）
        self.is_finished = True

    def _execute(self, other):
        # IR 节点分发器：根据节点类型调用对应的执行方法
        if isinstance(other, str):
            other = SglConstantText(other)

        assert isinstance(other, SglExpr), f"{other}"

        if isinstance(other, SglConstantText):
            # 常量文本：直接追加到 text_
            self._execute_fill(other.value)
        elif isinstance(other, SglGen):
            # 文本生成：调用 backend.generate() 或流式 generate_stream()
            self._execute_gen(other)
        elif isinstance(other, SglSelect):
            # 候选选择：调用 backend.select() 获取最优候选
            self._execute_select(other)
        elif isinstance(other, SglExprList):
            # 表达式列表：递归执行每个子节点
            for x in other.expr_list:
                self._execute(x)
        elif isinstance(other, SglRoleBegin):
            # 角色开始：插入角色前缀（和可能的默认 system 消息）
            self._execute_role_begin(other)
        elif isinstance(other, SglRoleEnd):
            # 角色结束：插入角色后缀，记录消息到 messages_ 列表
            self._execute_role_end(other)
        elif isinstance(other, SglImage):
            # 图像：编码为 base64，追加到 images_ 和 cur_images
            self._execute_image(other)
        elif isinstance(other, SglVideo):
            # 视频：编码多帧为 base64，追加到 images_ 和 cur_images
            self._execute_video(other)
        elif isinstance(other, SglVariable):
            # 变量引用：从源 StreamExecutor 中读取变量值并填充
            self._execute_variable(other)
        elif isinstance(other, SglVarScopeBegin):
            # 变量作用域开始：记录当前文本位置
            self._execute_var_scope_begin(other)
        elif isinstance(other, SglVarScopeEnd):
            # 变量作用域结束：截取作用域内容作为变量值
            self._execute_var_scope_end(other)
        elif isinstance(other, SglCommitLazy):
            # 提交懒执行：触发后端执行积累的延迟操作（如投机执行批次）
            self._execute_commit_lazy_operations(other)
        elif isinstance(other, SglConcateAndAppend):
            if (
                global_config.enable_parallel_encoding
                and self.backend.support_concate_and_append
            ):
                # 并行编码模式：通过后端 KV cache 拼接（更高效）
                self._execute_concatenate_and_append_kv_cache(other)
            else:
                # 普通模式：文本拼接（不利用 KV cache 共享）
                self._execute_concatenate_and_append_text(other)
        elif isinstance(other, SglSeparateReasoning):
            # 推理分离：将思维链与最终答案分开存储到不同变量
            self._execute_separate_reasoning(other)
        else:
            raise ValueError(f"Unknown type: {type(other)}")

    def _execute_fill(self, value: str, prefix=False):
        # 填充文本到 text_ 缓冲区，同时维护投机执行的 speculated_text
        value = str(value)

        if (
            self.cur_role == "assistant"
            and self.num_api_spec_tokens is not None
            and self.backend.is_chat_model
            and not prefix
        ):
            # 在 assistant 角色内且启用了 chat 模型投机执行：
            # 将文本片段追加到后端的 spec_format 中（延迟执行）
            self.backend.spec_fill(value)
            return

        if self.speculated_text.startswith(value):
            # 投机文本与实际文本匹配：消耗已预测的部分
            self.speculated_text = self.speculated_text[len(value) :]
        else:
            # 不匹配：清空投机缓冲（下次 gen 需重新生成）
            self.speculated_text = ""

        self.text_ += value

    def _execute_image(self, expr: SglImage):
        # 执行图像节点：将图像路径编码为 base64 并追加到图像列表
        path = expr.path

        base64_data = encode_image_base64(path)

        # 追加到全量历史图像列表和当前角色块的图像列表
        self.images_.append((path, base64_data))
        self.cur_images.append((path, base64_data))
        # 在文本中插入图像占位符（模型看到此 token 时知道此处有图像）
        self.text_ += self.chat_template.image_token

    def _execute_video(self, expr: SglVideo):
        # 执行视频节点：提取多帧并编码为 base64
        path = expr.path
        num_frames = expr.num_frames

        # 将视频按帧数采样并编码为 base64 列表
        base64_data = encode_video_base64(path, num_frames)

        self.images_.append((path, base64_data))
        self.cur_images.append((path, base64_data))
        # 与图像相同，插入占位符 token
        self.text_ += self.chat_template.image_token

    def _spec_gen(self, sampling_params):
        # 投机生成（Completion 模式）：利用预先生成的 speculated_text 减少 API 调用次数
        stop = sampling_params.stop
        max_new_tokens = sampling_params.max_new_tokens
        meta_info = {}

        def regen():
            nonlocal meta_info
            # 重新生成：放宽 stop 和 max_new_tokens 参数，批量预取更多 token
            sampling_params.max_new_tokens = max(
                sampling_params.max_new_tokens, self.num_api_spec_tokens
            )
            # 不设 stop，让模型尽可能多生成（后续由 find_stop() 软截断）
            sampling_params.stop = None
            self.speculated_text, meta_info = self.backend.generate(
                self, sampling_params=sampling_params
            )

        def find_stop():
            # 在 speculated_text 中查找最早的 stop 字符串位置
            if isinstance(stop, str):
                return self.speculated_text.find(stop)
            elif isinstance(stop, (tuple, list)):
                pos = -1
                for stop_str in stop:
                    stop_pos = self.speculated_text.find(stop_str)
                    if stop_pos != -1 and (pos == -1 or stop_pos < pos):
                        pos = stop_pos
                return pos
            else:
                raise Exception("Wrong type of stop in sampling parameters.")

        if stop is None:
            # 无 stop 条件：按 max_new_tokens 截取
            if len(self.speculated_text) < max_new_tokens:
                # 缓冲不足时重新生成
                regen()
            comp = self.speculated_text[:max_new_tokens]
            self.speculated_text = self.speculated_text[max_new_tokens:]
        elif isinstance(stop, (str, list, tuple)):
            if self.speculated_text == "":
                # 缓冲已空，需要重新生成
                regen()
            stop_pos = find_stop()
            if stop_pos == -1:
                # 在现有缓冲中未找到 stop，重新生成更多 token 后再次查找
                regen()
            stop_pos = find_stop()
            if stop_pos == -1:
                # 重新生成后仍未找到 stop：按 max_new_tokens 强制截断
                stop_pos = min(
                    sampling_params.max_new_tokens,
                    len(self.speculated_text),
                )
            # 截取 stop 之前的内容作为本次生成结果
            comp = self.speculated_text[:stop_pos]
            self.speculated_text = self.speculated_text[stop_pos:]
        else:
            raise ValueError("Wrong type of stop in sampling parameters.")

        return comp, meta_info

    def _execute_gen(self, expr: SglGen):
        # 执行生成节点：调用后端 generate()/generate_stream() 并存储结果
        sampling_params = self._resolve_sampling_params(expr.sampling_params)
        name = expr.name
        if not self.stream:
            # 非流式模式
            if self.num_api_spec_tokens is None:
                # 普通生成：直接调用后端 generate()
                comp, meta_info = self.backend.generate(
                    self,
                    sampling_params=sampling_params,
                )

            else:
                if self.backend.is_chat_model:
                    # Speculative execution on models with only chat interface.
                    # Store the calls into a temporary list.
                    # They will be lazily executed later.
                    # Chat 模型投机执行：将 gen 调用存入后端临时列表，等 role_end 时批量执行
                    comp, meta_info = self.backend.generate(
                        self,
                        sampling_params=sampling_params,
                        spec_var_name=name,
                    )
                    return

                else:  # Speculative execution on models with completion interface
                    # Completion 模型投机执行：使用本地 speculated_text 缓冲
                    comp, meta_info = self._spec_gen(sampling_params)
            if isinstance(comp, list):
                # n>1 时返回列表，只将第一个追加到文本
                self.text_ += comp[0]
            else:
                assert isinstance(comp, str)
                self.text_ += comp

            # 存储变量值、元信息，并触发等待方
            self.variables[name] = comp
            self.meta_info[name] = meta_info
            self.variable_event[name].set()
        else:
            # 流式模式：通过生成器逐步接收 chunk 并更新变量
            assert (
                self.num_api_spec_tokens is None
            ), "stream is not supported with api speculative execution"
            generator = self.backend.generate_stream(
                self, sampling_params=sampling_params
            )

            # 变量初始化为空字符串，触发 stream_var_event 让等待方开始监听
            self.variables[name] = ""
            self.stream_var_event[name].set()

            for comp, meta_info in generator:
                # 每收到一个 chunk：追加到 text_ 和变量，触发流事件通知等待方
                self.text_ += comp
                self.variables[name] += comp
                self.meta_info[name] = meta_info
                self.stream_var_event[name].set()
                self.stream_text_event.set()

            # 生成完毕：设置最终完成事件（variable_event）和流事件
            self.variable_event[name].set()
            self.stream_var_event[name].set()

    def _execute_select(self, expr: SglSelect):
        # 执行候选选择节点：调用后端 select() 返回最优候选决策
        choices_decision = self.backend.select(
            self, expr.choices, expr.temperature, expr.choices_method
        )
        if expr.name is not None:
            # 将选择结果和元信息存入变量，并触发完成事件
            name = expr.name
            self.variables[name] = choices_decision.decision
            self.meta_info[name] = choices_decision.meta_info
            self.variable_event[name].set()
            if self.stream_var_event:
                self.stream_var_event[name].set()
        # 将选择结果追加到文本（无论是否有 name）
        self.text_ += choices_decision.decision

    def _execute_variable(self, expr: SglVariable):
        # 执行变量引用节点：从源 StreamExecutor 中等待并获取变量值后填充
        src_executor = expr.source_stream_executor
        value = src_executor.get_var(expr.name)
        self._execute_fill(value)

    def _execute_role_begin(self, expr: SglRoleBegin):
        # 执行角色开始节点：插入默认 system 消息（如需要），然后填充角色前缀
        assert self.cur_role is None, "Nested roles are not allowed."

        if len(self.messages_) == 0 and expr.role != "system":
            # Insert the default system message
            # 首条消息非 system 时，自动插入模板默认的 system 消息
            default_system = self.chat_template.default_system_prompt
            if default_system:
                self._execute_role_begin(SglRoleBegin("system"))
                self._execute_fill(default_system)
                self._execute_role_end(SglRoleEnd("system"))

        self.cur_role = expr.role

        # 获取角色前缀（如 "[INST] " 或 "<|im_start|>user\n"）
        prefix, _ = self.chat_template.get_prefix_and_suffix(expr.role, self.messages_)

        # prefix=True 表示这是角色前缀，不走投机执行路径
        self._execute_fill(prefix, prefix=True)
        # 记录角色内容起始位置（用于 role_end 时提取纯内容）
        self.cur_role_begin_pos = len(self.text_)

    def _execute_role_end(self, expr: SglRoleEnd):
        # 执行角色结束节点：处理投机执行、填充后缀、构建消息对象
        if (
            self.cur_role == "assistant"
            and self.num_api_spec_tokens is not None
            and self.backend.is_chat_model
        ):
            # Execute the stored lazy generation calls
            # 在 assistant 角色结束时批量执行所有积累的投机生成调用
            self.backend.role_end_generate(self)
        self.cur_role = None

        # 提取角色内容（去除前导空白，避免重复模板空格）
        new_text = self.text_[self.cur_role_begin_pos :].lstrip()

        # 获取角色后缀（如 " [/INST]" 或 "<|im_end|>\n"）
        _, suffix = self.chat_template.get_prefix_and_suffix(expr.role, self.messages_)
        self._execute_fill(suffix)

        if self.cur_images:
            # OpenAI vision API format
            # 有图像时：构建多模态消息格式（content 为列表，包含文本和图像）
            last_msg = {
                "role": expr.role,
                "content": [{"type": "text", "text": new_text}],
            }
            for image_path, image_base64_data in self.cur_images:
                last_msg["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64_data}"
                        },
                    }
                )
            # 将多模态消息追加到消息列表并清空当前图像列表
            self.messages_.append(last_msg)
            self.cur_images = []
        else:
            # OpenAI chat API format
            # 纯文本消息：直接追加 {role, content} 对象
            self.messages_.append({"role": expr.role, "content": new_text})

    def _execute_var_scope_begin(self, expr: SglVarScopeBegin):
        # 变量作用域开始：将当前 text_ 长度存入变量（作为起始位置标记）
        self.variables[expr.name] = int(len(self.text_))

    def _execute_var_scope_end(self, expr: SglVarScopeEnd):
        # 变量作用域结束：截取起始位置到当前的文本作为变量值
        self.variables[expr.name] = self.text_[self.variables[expr.name] :]
        self.variable_event[expr.name].set()

    def _execute_commit_lazy_operations(self, expr: SglCommitLazy):
        # 提交懒执行：触发后端处理积累的延迟操作（如刷新 KV cache 或执行投机批次）
        self.backend.commit_lazy_operations(self)

    def _execute_concatenate_and_append_text(self, expr: SglConcateAndAppend):
        # 文本拼接模式：等待所有子状态完成，将各子状态的增量文本拼接后填充到当前状态
        new_text = ""
        for s in expr.states:
            exe = s.stream_executor
            # 等待子状态执行完毕
            exe.sync()
            # 只取 fork 之后产生的增量文本
            new_text += exe.text_[exe.fork_start_text_pos :]

        self._execute_fill(new_text)

    def _execute_concatenate_and_append_kv_cache(self, expr: SglConcateAndAppend):
        # KV cache 拼接模式：将子状态的 KV cache 拼接到当前状态（并行编码优化）
        self_len = len(self.text_)

        # 第一轮：向各子状态提交 CommitLazy，确保 KV cache 已刷新
        for i, s in enumerate(expr.states):
            exe = s.stream_executor
            exe.submit(SglCommitLazy())

        # 第二轮：等待各子状态完成并拼接文本
        for i, s in enumerate(expr.states):
            exe = s.stream_executor
            exe.sync()
            # 验证 fork 起始位置与当前状态的文本长度一致（保证 KV cache 对齐）
            assert exe.fork_start_text_pos == self_len
            self.text_ += exe.text_[exe.fork_start_text_pos :]

        # 收集各子状态的 sid，调用后端进行 KV cache 拼接
        src_rids = [state.stream_executor.sid for state in expr.states]
        self.backend.concatenate_and_append(src_rids, self.sid)

    def _execute_separate_reasoning(self, expr: SglSeparateReasoning):
        # 执行推理分离节点：将已生成变量的内容分割为推理链和最终答案，分别存储
        if self.stream:
            # separate reasoning for stream is not supported
            # 流式模式暂不支持推理分离（需要完整文本后才能解析）
            return

        if (
            self.cur_role == "assistant"
            and self.num_api_spec_tokens is not None
            and self.backend.is_chat_model
        ):
            # Execute the stored lazy generation calls
            # 确保投机执行的生成调用已经执行完毕
            self.backend.role_end_generate(self)

        from sglang.srt.parser.reasoning_parser import ReasoningParser

        reasoning_parser = ReasoningParser(expr.model_type)
        other = expr.expr
        if not other:
            return
        elif isinstance(other, SglGen) or isinstance(other, SglSelect):
            # 获取生成变量的当前文本值
            cur_text = self.get_var(other.name)
            # 调用 reasoning_parser 将文本分割为推理链和最终答案
            reasoning, normal_text = reasoning_parser.parse_non_stream(cur_text)
            # 生成推理链变量名（通常为原变量名加 "_reasoning" 后缀）
            reasoning_name = expr.process_name_for_reasoning(other.name)
            # 更新原变量为纯答案文本，创建推理链变量
            self.set_var(other.name, normal_text)
            self.set_var(reasoning_name, reasoning)
            # the variable is ready to be used
            # 触发推理链变量的完成事件（解除等待方阻塞）
            self.variable_event[reasoning_name].set()
            # 更新 text_：从角色开始位置截取，替换为纯答案文本
            self.text_ = self.text_[: self.cur_role_begin_pos] + normal_text
        elif isinstance(other, SglExprList):
            # 递归处理列表中的每个 gen/select 节点
            for x in other.expr_list:
                self._execute_separate_reasoning(
                    SglSeparateReasoning(expr.model_type, x)
                )

    def _init_var_event(self, expr):
        # 在 submit() 时提前初始化变量事件（确保 get_var() 能够正确 wait）
        if isinstance(
            expr, (SglGen, SglSelect, SglVarScopeBegin, SglSeparateReasoning)
        ):
            # 为有名称的生成/选择/作用域/推理分离节点创建完成事件
            self.variable_event[expr.name] = threading.Event()
            if self.stream:
                # 流式模式下还需要创建流事件（供 text_iter 逐步通知）
                self.stream_var_event[expr.name] = threading.Event()
        elif isinstance(expr, SglExprList):
            # 列表：递归初始化每个子节点的事件
            for e in expr.expr_list:
                self._init_var_event(e)

    def _resolve_sampling_params(self, sampling_params):
        """
        Construct sampling param based on default + override values

        The default values of sampling are populated in `default_sampling_para` via sgl.function.run(...sampling_args)
        , and `sampling_params` contains the override values from sgl.gen().

        Here we use default_sampling_para as the base and override the values if they exist in `sampling_params`.
        It also extends the stop tokens based on the chat template.
        """
        # 合并采样参数：以 default_sampling_para 为基础，sgl.gen() 中指定的参数优先覆盖

        # deepcopy is required because the dict has lists inside
        # 必须深拷贝，因为 stop/stop_token_ids 等字段可能是列表，避免共享引用导致污染
        clone = copy.deepcopy(self.default_sampling_para)

        # 逐个检查 sampling_params 中的覆盖字段，非 None 才覆盖（None 表示未指定）
        for item in [
            "max_new_tokens",
            "min_new_tokens",
            "n",
            "stop",
            "stop_token_ids",
            "stop_regex",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "frequency_penalty",
            "presence_penalty",
            "ignore_eos",
            "return_logprob",
            "logprob_start_len",
            "top_logprobs_num",
            "return_text_in_logprobs",
            "dtype",
            "regex",
            "json_schema",
        ]:
            value = getattr(sampling_params, item, None)
            if value is not None:
                setattr(clone, item, value)

        if self.chat_template.stop_str:
            # 将聊天模板的 stop 字符串（如 "</s>" "<|im_end|>"）追加到 stop 列表
            if clone.stop == ():
                clone.stop = []
            elif isinstance(clone.stop, str):
                clone.stop = [clone.stop]
            clone.stop += self.chat_template.stop_str

        return clone

    def __del__(self):
        # 析构时发送结束信号（防止工作线程泄漏）
        self.end()


class ProgramState:
    """The state of an SGL program."""
    # ProgramState：用户程序中操作的状态对象，封装 StreamExecutor 提供 API 接口

    def __init__(self, stream_executor: StreamExecutor):
        # 持有对应的 StreamExecutor（所有 IR 操作均委托给它执行）
        self.stream_executor = stream_executor

    def _role_common(self, name: str, expr: Optional[SglExpr] = None):
        # 内部辅助：支持两种角色使用方式——直接表达式或上下文管理器（with 语句）
        if expr is not None:
            # 有内容时：构建完整角色节点并提交执行
            role_expr = SglExprList([SglRoleBegin(name), expr, SglRoleEnd(name)])
            self.stream_executor.submit(role_expr)
            return role_expr
        else:
            # 无内容时：返回上下文管理器，支持 `with s.user():` 语法
            @contextmanager
            def role_scope():
                self.stream_executor.submit(SglRoleBegin(name))
                yield
                self.stream_executor.submit(SglRoleEnd(name))

            return role_scope()

    def system(self, expr: Optional[SglExpr] = None):
        # 创建/进入 system 角色块
        return self._role_common("system", expr)

    def user(self, expr: Optional[SglExpr] = None):
        # 创建/进入 user 角色块
        return self._role_common("user", expr)

    def assistant(self, expr: Optional[SglExpr] = None):
        # 创建/进入 assistant 角色块
        return self._role_common("assistant", expr)

    @contextmanager
    def var_scope(self, name: str):
        # 变量作用域上下文管理器：将 with 块中生成的文本捕获为命名变量
        self.stream_executor.submit(SglVarScopeBegin(name))
        yield
        self.stream_executor.submit(SglVarScopeEnd(name))

    def fork(
        self,
        size: int = 1,
        position_ids_offset: Optional[List[int]] = None,
    ):
        # fork：将当前状态复制为 size 个子 ProgramState（用于并行分支生成）
        stream_executors = self.stream_executor.fork(size, position_ids_offset)
        # 将每个子 StreamExecutor 包装为 ProgramState
        states = [ProgramState(x) for x in stream_executors]
        # 包装为 ProgramStateGroup，方便后续 join 操作
        state_group = ProgramStateGroup(states, self)
        return state_group

    @contextmanager
    def copy(self, position_ids_offset: Optional[List[int]] = None):
        # copy：创建单个副本的便捷上下文管理器（等价于 fork(1) + join）
        state_group = self.fork(1, position_ids_offset)
        try:
            yield state_group[0]
        finally:
            # with 块结束时自动 join（释放子状态资源）
            state_group.join()

    def text(self):
        # 获取完整生成文本（阻塞等待执行完毕）
        return self.stream_executor.text()

    def messages(self):
        # 获取 OpenAI chat 格式的消息列表（阻塞等待执行完毕）
        return self.stream_executor.messages()

    def sync(self):
        # 等待所有提交的 IR 节点执行完毕
        return self.stream_executor.sync()

    def error(self):
        # 获取执行过程中的错误信息（若无错误则返回 None）
        return self.stream_executor.error()

    def text_iter(self, var_name: Optional[str] = None):
        # 流式文本迭代器：逐步 yield 生成的文本 chunk（同步生成器）
        if self.stream_executor.stream:
            # 流式模式：通过事件等待机制逐步获取增量文本
            prev = 0
            if var_name is None:
                # 无变量名：迭代全量文本流
                event = self.stream_executor.stream_text_event
                while True:
                    event.wait()
                    event.clear()
                    out = str(self.stream_executor.text_[prev:])
                    prev += len(out)
                    if out:
                        yield out
                    if self.stream_executor.is_finished:
                        break
            else:
                # 有变量名：等待变量的流事件被初始化，然后迭代该变量的增量文本
                event = None
                while not event:
                    if var_name in self.stream_executor.stream_var_event:
                        event = self.stream_executor.stream_var_event[var_name]
                    if self.stream_executor.is_finished:
                        yield ""
                        return

                while True:
                    event.wait()
                    event.clear()
                    out = str(self.stream_executor.variables[var_name][prev:])
                    prev += len(out)
                    if out:
                        yield out
                    # variable_event 被设置表示该变量已生成完毕
                    if self.stream_executor.variable_event[var_name].is_set():
                        break
        else:
            # 非流式模式：直接返回完整文本（单次 yield）
            if var_name is None:
                yield self.text()
            else:
                yield self.get_var(var_name)

    async def text_async_iter(
        self, var_name: Optional[str] = None, return_meta_data: bool = False
    ):
        # 流式文本异步迭代器：与 text_iter 类似，但以 async for 方式使用
        # 通过 loop.run_in_executor 将 threading.Event.wait 包装为协程，避免阻塞事件循环
        loop = asyncio.get_running_loop()

        if self.stream_executor.stream:
            prev = 0
            if var_name is None:
                # 无变量名：异步迭代全量文本流
                event = self.stream_executor.stream_text_event
                while True:
                    # 在线程池中等待事件（不阻塞当前事件循环）
                    await loop.run_in_executor(None, event.wait)
                    event.clear()
                    out = str(self.stream_executor.text_[prev:])
                    prev += len(out)
                    if out:
                        yield out
                    if self.stream_executor.is_finished:
                        break
            else:
                # 有变量名：等待该变量的流事件初始化，然后异步迭代增量文本
                event = None
                pending_meta_info = None
                while not event:
                    if var_name in self.stream_executor.stream_var_event:
                        event = self.stream_executor.stream_var_event[var_name]
                    if self.stream_executor.is_finished:
                        yield ""
                        return

                while True:
                    await loop.run_in_executor(None, event.wait)
                    event.clear()
                    out = str(self.stream_executor.variables[var_name][prev:])
                    meta_info = self.stream_executor.meta_info.get(var_name)
                    prev += len(out)
                    if out:
                        if return_meta_data:
                            # return_meta_data=True 时：与 meta_info 一起 yield
                            assert meta_info is not None
                            merged_meta_info = _merge_stream_meta_info(
                                pending_meta_info,
                                meta_info,
                            )
                            pending_meta_info = None
                            yield out, merged_meta_info
                        else:
                            yield out
                    elif return_meta_data and meta_info is not None:
                        # 无文本但有 meta_info：积压到 pending_meta_info，下次一起 yield
                        pending_meta_info = _merge_stream_meta_info(
                            pending_meta_info,
                            meta_info,
                        )
                    if self.stream_executor.variable_event[var_name].is_set():
                        break
        else:
            # 非流式模式：直接 yield 完整结果
            if var_name is None:
                yield self.text()
            else:
                yield self.get_var(var_name)

    def get_var(self, name):
        # 获取命名变量的值（阻塞等待变量生成完毕）
        return self.stream_executor.get_var(name)

    def set_var(self, name, value):
        # 直接设置变量值（用于推理分离等后处理）
        return self.stream_executor.set_var(name, value)

    def get_meta_info(self, name):
        # 获取变量的元信息（logprob 等）
        return self.stream_executor.get_meta_info(name)

    def __iadd__(self, other):
        # 重载 += 运算符：将 IR 节点提交给 StreamExecutor 执行（s += sgl.gen("answer")）
        if other is None:
            raise ValueError("Tried to append None to state.")
        self.stream_executor.submit(other)
        return self

    def __getitem__(self, name):
        # 重载 [] 取值运算符：s["answer"] 等价于 s.get_var("answer")
        return self.get_var(name)

    def __setitem__(self, name, value):
        # 重载 [] 赋值运算符：s["answer"] = value 等价于 s.set_var("answer", value)
        self.set_var(name, value)

    def __contains__(self, name):
        # 重载 in 运算符：检查变量是否已存在（如 "answer" in s）
        return name in self.stream_executor.variables

    def __del__(self):
        # 析构时通知 StreamExecutor 结束（防止工作线程泄漏）
        self.stream_executor.end()

    def __repr__(self) -> str:
        # 调试打印：显示当前完整文本
        return f"ProgramState({self.text()})"


class ProgramStateGroup:
    # ProgramStateGroup：fork 产生的多个 ProgramState 的容器，支持 join 操作
    def __init__(
        self, states: List[ProgramState], src_state: Optional[ProgramState] = None
    ):
        # states：子状态列表（fork 产生的各分支）
        self.states = states
        # src_state：父状态（join 时将子状态的变量合并回父状态）
        self.src_state = src_state

    def join(self, mode: str = "gather_variable"):
        # join：将所有子状态的结果合并回父状态，并结束子状态
        if mode == "gather_variable":
            # Copy variables back
            # 变量收集模式：将各子状态中的新变量（父状态没有的）收集到父状态的变量字典
            src_vars = self.src_state.stream_executor.variables
            src_var_set = set(src_vars.keys())
            for child_state in self.states:
                # 等待子状态执行完毕
                child_state.stream_executor.sync()
                child_vars = child_state.stream_executor.variables
                # 仅收集子状态独有的新变量（fork 之后新生成的）
                new_vars = set(child_vars.keys()) - src_var_set

                for k in new_vars:
                    if k in src_vars:
                        # 若父状态已有同名变量，将子状态值追加到列表（多分支聚合）
                        src_vars[k].append(child_vars[k])
                    else:
                        # 新变量：创建列表以容纳所有分支的值
                        src_vars[k] = [child_vars[k]]
        elif mode == "concate_and_append":
            # Concatenate and append KV cache
            # KV cache 拼接模式：将子状态的文本和 KV cache 拼接到父状态
            self.src_state += SglConcateAndAppend(self.states)
            # Need a sync here. Otherwise, `states` can be deleted.
            # 必须 sync，防止 states 被提前析构（导致 KV cache 拼接操作异常）
            self.src_state.stream_executor.sync()
        else:
            raise ValueError(f"Invalid join mode: {mode}")

        # 结束所有子状态的 StreamExecutor（释放资源）
        for s in self.states:
            s.stream_executor.end()

    def __getitem__(self, i: int):
        # 通过索引访问第 i 个子状态
        return self.states[i]

    def __setitem__(self, i: int, value):
        # 赋值检查（仅验证一致性，不支持实际替换）
        assert self.states[i] == value

    def __iadd__(self, other):
        # 重载 += 运算符：批量向所有子状态提交相同或不同的 IR 节点
        if isinstance(other, Callable):
            # lambda function
            # 可调用对象（lambda）：对每个分支传入分支索引，返回对应的 IR 节点
            for i in range(len(self.states)):
                self.states[i] += other(i)
        elif isinstance(other, SglExpr):
            # 单个 IR 节点：广播到所有子状态（每个分支执行相同操作）
            for i in range(len(self.states)):
                self.states[i] += other
        elif isinstance(other, (list, tuple)):
            # 列表/元组：按索引为每个子状态分配不同的 IR 节点
            for i in range(len(self.states)):
                self.states[i] += other[i]
        else:
            raise ValueError(f"Invalid value: {other}")

        return self

