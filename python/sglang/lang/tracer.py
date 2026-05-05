"""Tracing a program."""
# 追踪程序执行以生成 IR（中间表示）节点树，用于分析前缀、延迟执行等

import uuid
from typing import Any, Dict, List, Optional

# 导入后端基类
from sglang.lang.backend.base_backend import BaseBackend
# 导入程序状态基类和状态组
from sglang.lang.interpreter import ProgramState, ProgramStateGroup
# 导入所有 IR 节点类型
from sglang.lang.ir import (
    SglArgument,       # 函数参数节点
    SglConstantText,   # 常量文本节点
    SglExpr,           # 表达式基类
    SglExprList,       # 表达式列表节点
    SglFork,           # fork 节点（并行分支）
    SglGen,            # 生成节点
    SglGetForkItem,    # 获取 fork 中第 i 个分支
    SglRoleBegin,      # 角色开始节点（chat 模式）
    SglRoleEnd,        # 角色结束节点（chat 模式）
    SglSelect,         # 选择节点
    SglVariable,       # 变量节点
    SglVarScopeBegin,  # 变量作用域开始
    SglVarScopeEnd,    # 变量作用域结束
)


# 用于中断 only_trace_prefix 模式下追踪的异常（遇到非常量节点时抛出）
class StopTracing(Exception):
    pass


def extract_prefix_by_tracing(program, backend):
    # 通过追踪提取程序的公共常量前缀（用于 KV cache 预热）
    # Create dummy arguments
    # 创建虚拟参数（值为 None 的 SglArgument 节点）
    dummy_arguments = {name: SglArgument(name, None) for name in program.arg_names}
    arguments = dummy_arguments
    # 合并绑定参数（bind() 时传入的固定参数）
    arguments.update(program.bind_arguments)

    # Trace
    # 创建仅追踪前缀的 Tracer（遇到非常量节点时停止）
    tracer = TracerProgramState(backend, arguments, only_trace_prefix=True)
    try:
        with TracingScope(tracer):
            tracer.ret_value = program.func(tracer, **arguments)
    except (StopTracing, TypeError, AttributeError):
        # Some exceptions may not be caught
        # 追踪提前终止是正常情况（前缀提取到非常量节点即停止）
        pass

    # Run and cache prefix
    # 遍历追踪到的节点，连续提取常量文本拼接为前缀
    prefix = ""
    for expr in tracer.flatten_nodes():
        if isinstance(expr, SglConstantText):
            prefix += expr.value
        else:
            # 遇到非常量节点（如 SglGen）则前缀提取结束
            break
    return prefix


def trace_program(program, arguments, backend):
    # 完整追踪程序，生成完整的 IR 节点树（用于分析程序结构）
    # Create dummy backend
    # 若未提供后端则使用空基类（仅用于追踪，不实际执行）
    if backend is None:
        backend = BaseBackend()

    # Create dummy arguments
    # 为未绑定的参数创建虚拟节点
    dummy_arguments = {
        name: SglArgument(name, None)
        for name in program.arg_names
        if name not in arguments
    }
    arguments.update(dummy_arguments)
    # 合并程序级绑定参数
    arguments.update(program.bind_arguments)

    # Trace
    # 完整追踪（only_trace_prefix=False），记录所有节点
    tracer = TracerProgramState(backend, arguments, only_trace_prefix=False)
    with TracingScope(tracer):
        tracer.ret_value = program.func(tracer, **arguments)
    return tracer


# 追踪状态：继承 ProgramState，在执行时记录 IR 节点而非真正调用后端
class TracerProgramState(ProgramState):
    def __init__(self, backend, arguments, only_trace_prefix):
        # 为追踪器分配唯一 ID
        self.pid = uuid.uuid4().hex
        self.backend = backend
        # 参数字典（含虚拟参数和绑定参数）
        self.arguments: Dict[str, Any] = arguments
        # 是否仅追踪前缀（遇到非常量节点时抛出 StopTracing）
        self.only_trace_prefix = only_trace_prefix

        # 若后端是 Runtime（包含 endpoint 属性），取其 endpoint 作为实际后端
        if hasattr(backend, "endpoint"):
            self.backend = backend.endpoint

        # 已追踪到的 IR 节点列表
        self.nodes = []
        # 指向最后一个节点的指针（用于构建链式 prev_node 关系）
        self.last_node = None
        # 变量字典（name → SglVariable 节点）
        self.variables = {}
        self.ret_value = None

        # For completion
        # （完成类型无需额外状态）

        # For chat
        # 聊天消息列表（追踪时不填充实际内容，仅记录角色）
        self.messages_ = []
        self.cur_role = None
        # 从后端获取聊天模板（用于生成角色前缀/后缀）
        self.chat_template = self.backend.get_chat_template()

        # For multi states
        # 子状态列表（fork 产生的子状态）
        self.child_states = []

        # 若当前在 TracingScope 中，则将自身注册为子状态
        cur_scope = TracingScope.get_current_scope()
        if cur_scope is not None:
            cur_scope.add_child_state(self)

    ##################################
    ########### Public API ###########
    ##################################

    def fork(self, size: int = 1, position_ids_offset: Optional[List[int]] = None):
        # 追踪 fork 操作：创建 SglFork 节点和 size 个子追踪状态
        assert size >= 1

        # 仅追踪前缀模式下不允许 fork（前缀必须是线性的）
        if self.only_trace_prefix:
            raise StopTracing()

        # 创建 fork IR 节点，挂接到当前链尾
        fork_node = SglFork(size)
        fork_node.prev_node = self.last_node

        # 为每个分支创建独立的追踪状态
        states = [
            TracerProgramState(self.backend, self.arguments, self.only_trace_prefix)
            for _ in range(size)
        ]

        for i in range(size):
            # 每个分支的入口节点是 SglGetForkItem(i)，链接到 fork_node
            node = SglGetForkItem(i)
            node.prev_node = fork_node
            states[i].last_node = node
            # 复制父状态的变量、消息和角色信息到各分支
            states[i].variables = dict(self.variables)
            states[i].messages_ = list(self.messages_)
            states[i].cur_role = self.cur_role
            states[i].chat_template = self.chat_template

        # 将所有分支状态包装为 ProgramStateGroup 返回
        state_group = ProgramStateGroup(states, self)

        return state_group

    ##################################
    ########## Internal API ##########
    ##################################

    def _append_node(self, other: SglExpr):
        # 将节点追加到节点列表，并维护 prev_node 链
        self.nodes.append(other)
        other.prev_node = self.last_node
        self.last_node = other

    def _execute(self, other: SglExpr):
        # 统一分发执行：根据节点类型调用对应的追踪处理方法
        if isinstance(other, str):
            other = SglConstantText(other)

        # 设置节点所属程序 ID
        other.pid = self.pid

        if isinstance(other, SglConstantText):
            self._execute_fill(other)
        elif isinstance(other, SglGen):
            self._execute_gen(other)
        elif isinstance(other, SglSelect):
            self._execute_select(other)
        elif isinstance(other, SglExprList):
            # 表达式列表：递归执行每个子表达式
            for x in other.expr_list:
                self._execute(x)
        elif isinstance(other, SglRoleBegin):
            self._execute_role_begin(other)
        elif isinstance(other, SglRoleEnd):
            self._execute_role_end(other)
        elif isinstance(other, SglVarScopeBegin):
            self._execute_var_scope_begin(other)
        elif isinstance(other, SglVarScopeEnd):
            self._execute_var_scope_end(other)
        else:
            # 未知节点类型：仅追踪前缀时停止，否则直接追加
            if self.only_trace_prefix:
                raise StopTracing()
            else:
                self._append_node(other)

        return self

    def __iadd__(self, other):
        # 重载 += 运算符，使 `s += expr` 等价于 `s._execute(expr)`
        self._execute(other)
        return self

    def _execute_fill(self, expr: SglConstantText):
        # 追踪常量文本填充：直接追加节点
        if isinstance(expr, str):
            expr = SglConstantText(expr)
        self._append_node(expr)

    def _execute_gen(self, expr: SglGen):
        # 追踪生成节点：创建对应的 SglVariable 并追加
        name = expr.name if expr.name is not None else "gen_" + str(len(self.variables))
        new_node = SglVariable(name, source=expr)
        self.variables[name] = new_node
        self._append_node(expr)

    def _execute_select(self, expr: SglSelect):
        # 追踪选择节点：创建对应的 SglVariable 并追加
        name = (
            expr.name if expr.name is not None else "select_" + str(len(self.variables))
        )
        new_node = SglVariable(name, source=expr)
        self.variables[name] = new_node
        self._append_node(expr)

    def _execute_role_begin(self, expr: SglRoleBegin):
        # 追踪角色开始：插入默认 system 消息（若需要），然后填充角色前缀
        assert self.cur_role is None, "Nested roles are not allowed."

        if len(self.messages_) == 0 and expr.role != "system":
            # Insert default system message
            # 首条消息非 system 时插入默认 system 消息
            default_system = self.chat_template.default_system_prompt
            if default_system:
                self._execute_role_begin(SglRoleBegin("system"))
                self._execute_fill(default_system)
                self._execute_role_end(SglRoleEnd("system"))

        self.cur_role = expr.role

        # 获取该角色的前缀和后缀
        prefix, suffix = self.chat_template.get_prefix_and_suffix(
            expr.role, self.messages_
        )

        # 追踪前缀填充
        self._execute_fill(prefix)

    def _execute_role_end(self, expr: SglRoleEnd):
        # 追踪角色结束：填充后缀并记录消息
        prefix, suffix = self.chat_template.get_prefix_and_suffix(
            expr.role, self.messages_
        )

        # 追踪后缀填充
        self._execute_fill(suffix)

        # 向消息列表追加该角色的占位消息（content 为空，追踪时不填充实际内容）
        self.messages_.append({"role": expr.role, "content": ""})

        self.cur_role = None

    def _execute_var_scope_end(self, expr: SglVarScopeEnd):
        # 变量作用域结束：将当前最后节点作为变量的来源
        new_node = SglVariable(expr.name, source=self.last_node)
        self.variables[expr.name] = new_node

    def get_var(self, name):
        # 按名称获取变量值（优先从参数字典中取，其次从 variables 中取）
        ret = self.arguments.get(name, None)
        if ret is not None:
            return ret

        # 返回变量的副本（避免共享引用）
        v = self.variables[name]
        return SglVariable(v.name, v.source)

    def flatten_nodes(self):
        # 将节点树展平为顺序列表（递归展开 SglExprList）
        def traverse(cur):
            if isinstance(cur, SglExprList):
                for child in cur.expr_list:
                    traverse(child)
            else:
                ret.append(cur)

        ret = []
        for x in self.nodes:
            traverse(x)
        return ret

    def __del__(self):
        # 析构时无需特殊处理
        pass


# 追踪作用域：通过上下文管理器维护当前活跃的追踪器，支持嵌套
class TracingScope:
    # 类变量：当前最内层的追踪作用域（None 表示不在追踪中）
    cur_scope = None

    def __init__(self, tracer_state: TracerProgramState):
        self.tracer_state = tracer_state
        # 保存上一层作用域（用于嵌套支持）
        self.last_scope = TracingScope.cur_scope

    def __enter__(self):
        # 进入作用域：将自身设为当前作用域
        TracingScope.cur_scope = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # 退出作用域：恢复上一层作用域
        TracingScope.cur_scope = self.last_scope

    @staticmethod
    def get_current_scope():
        # 获取当前最内层的追踪作用域
        return TracingScope.cur_scope

    def add_child_state(self, state: TracerProgramState):
        # 将子状态注册到当前及所有祖先作用域的追踪器中
        cur_scope = self
        while cur_scope is not None:
            cur_scope.tracer_state.child_states.append(state)
            cur_scope = cur_scope.last_scope
