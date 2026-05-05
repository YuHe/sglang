# EAGLE Worker 模块：实现 EAGLE 投机解码的草稿 worker
# 包含草稿 token 生成、token 树构建、验证输入准备等核心功能
import logging
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple

import torch

# 导入分布式通信工具（张量并行组）
from sglang.srt.distributed import get_tp_group
# 导入 NPU 硬件后端的 EAGLE 草稿计算图 runner
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_npu_graph_runner import (
    EAGLEDraftNpuGraphRunner,
)
# 导入数据并行注意力机制的 TP 组获取工具
from sglang.srt.layers.dp_attention import get_attention_tp_group
# 导入 logits 处理器输出数据结构
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
# 导入 MoE（混合专家）投机解码上下文管理器
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
# 导入投机解码 v1 的 logprob 输出添加工具
from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v1
# 导入权重更新请求数据结构
from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
# 导入调度批次数据结构
from sglang.srt.managers.schedule_batch import ScheduleBatch
# 导入生成批次结果数据结构
from sglang.srt.managers.scheduler import GenerationBatchResult
# 导入张量并行 model worker 基类
from sglang.srt.managers.tp_worker import TpModelWorker
# 导入内存缓存中的 token slot 分配工具
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
# 导入 CUDA 计算图 runner
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
# 导入前向传播批次信息（隐藏状态捕获模式、前向批次、前向模式）
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
# 导入请求时间统计工具
from sglang.srt.observability.req_time_stats import set_time_batch
# 导入全局追踪开关
from sglang.srt.observability.trace import get_global_tracing_enabled
# 导入服务器参数配置
from sglang.srt.server_args import ServerArgs
# 导入自适应投机解码控制器和运行时状态
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
)
# 导入草稿后端工厂，用于创建注意力后端
from sglang.srt.speculative.draft_utils import DraftBackendFactory
# 导入 EAGLE 草稿 CUDA 计算图 runner
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
# 导入 EAGLE 草稿 extend 阶段的 CUDA 计算图 runner
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
# 导入 EAGLE 草稿输入、验证输入、验证输出数据结构
from sglang.srt.speculative.eagle_info import (
    EagleDraftInput,
    EagleVerifyInput,
    EagleVerifyOutput,
)
# 导入 EAGLE 工具函数：高效 token 树构建、草稿结果整理
from sglang.srt.speculative.eagle_utils import (
    build_tree_kernel_efficient,
    organize_draft_results,
)
# 导入投机解码算法枚举（EAGLE、EAGLE2、EAGLE3等）
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
# 导入投机解码通用工具函数
from sglang.srt.speculative.spec_utils import (
    assign_draft_cache_locs,      # 分配草稿 KV cache 位置
    draft_tp_context,              # 草稿模型的 TP 上下文管理器
    fast_topk,                     # 快速 top-k 选择
    generate_token_bitmask,        # 生成 token 位掩码（用于约束解码）
    get_last_loc_large_page_size_large_top_k,  # 大页面大 top-k 场景的最后位置获取
    load_token_map,                # 加载热门 token 映射表
    maybe_detect_nan,              # 可选的 NaN 检测（调试用）
    maybe_detect_oob,              # 可选的越界检测（调试用）
    select_top_k_tokens,           # 选择 top-k tokens
)
# 导入通用工具函数
from sglang.srt.utils import (
    MultiprocessingSerializer,     # 多进程序列化工具
    empty_context,                 # 空上下文管理器（no-op）
    get_available_gpu_memory,      # 获取可用 GPU 内存
    is_cuda,                       # 是否 CUDA 设备
    is_musa,                       # 是否摩尔线程 MUSA 设备
    is_npu,                        # 是否昇腾 NPU 设备
    next_power_of_2,               # 向上取最近 2 的幂
)
# 导入 torch 猴子补丁（修复某些 reduction 操作兼容性）
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

# 全局设备类型标志（在模块加载时确定，避免重复检测）
_is_npu = is_npu()
_is_musa = is_musa()

# 仅在 CUDA 设备上导入 sgl_kernel 的 segment_packbits（位压缩工具）
if is_cuda():
    from sgl_kernel import segment_packbits  # noqa: F401

logger = logging.getLogger(__name__)


# EAGLEWorker：EAGLE 投机解码的草稿 worker，继承自 TpModelWorker
# 负责生成候选 token 树，供目标模型并行验证
class EAGLEWorker(TpModelWorker):

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,  # 目标（验证）模型 worker
    ):
        # Parse arguments
        self.server_args = server_args
        # top-k：每步草稿模型保留的候选 token 数量
        self.topk = server_args.speculative_eagle_topk
        # 投机解码步数（草稿模型展开多少步）
        self.speculative_num_steps = server_args.speculative_num_steps
        # 每次投机解码产生的草稿 token 总数
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.gpu_id = gpu_id
        self.device = server_args.device
        # 保存目标 worker 引用，用于共享内存池和 lm_head
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        # 解析投机解码算法类型（EAGLE/EAGLE2/EAGLE3等）
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Adaptive speculative
        # 自适应投机解码控制器，可根据运行时统计动态调整投机参数
        self.adaptive_controller: Optional[AdaptiveController] = None
        if server_args.speculative_adaptive:
            self.adaptive_controller = AdaptiveController(
                self, config_path=server_args.speculative_adaptive_config
            )

        # Override the context length of the draft model to be the same as the target model.
        # 草稿模型的上下文长度需与目标模型一致，确保 KV cache 兼容
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # Do not capture cuda graph in `super().__init__()`
        # It will be captured later.
        # 先禁用 CUDA graph，避免在父类初始化时提前捕获（需要在正确的 TP 上下文中捕获）
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True
        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        # 与目标 worker 共享 token pool allocator，减少内存分配开销
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Load hot token ids
        # 热门 token 映射：EAGLE 草稿模型只预测常见的"热门"token，减少词表扫描开销
        if self.speculative_algorithm.is_eagle3():
            # EAGLE3 模型内置 hot token 映射，无需外部指定
            if server_args.speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None
        elif server_args.speculative_token_map is not None:
            # 从文件加载热门 token ID 列表
            self.hot_token_id = load_token_map(server_args.speculative_token_map)
            # 通过 json_model_override_args 传递热门词表大小给模型
            server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            # 不使用热门 token 映射，使用完整词表
            self.hot_token_id = None

        # Init draft worker
        # 对于 EAGLE3 + DP 注意力，需要在特定的 TP 组上下文中初始化
        if server_args.enable_dp_attention and self.speculative_algorithm.is_eagle3():
            ctx = draft_tp_context(get_attention_tp_group())
        else:
            ctx = empty_context()
        # 在 MoE 投机后端上下文中初始化父类（加载草稿模型权重）
        with (
            ctx
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            super().__init__(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,  # FIXME
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,  # 标记为草稿 worker
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                memory_pool_config=target_worker.model_runner.memory_pool_config,
            )

        # 从目标模型获取 embedding 层和 lm_head 层（两者共享权重以节省内存）
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()

        if self.speculative_algorithm.is_eagle3():
            # most cases EAGLE3 models don't share lm_head
            # but some models (e.g. nvidia/gpt-oss-120b-Eagle3) shares
            # EAGLE3 多数情况下不共享 lm_head，但部分模型（如 nvidia/gpt-oss-120b-Eagle3）会共享
            if (
                hasattr(self.draft_model_runner.model, "load_lm_head_from_target")
                and self.draft_model_runner.model.load_lm_head_from_target
            ):
                # 共享 embedding 和 lm_head（完整共享）
                self.draft_model_runner.model.set_embed_and_head(embed, head)
            else:
                # 仅共享 embedding，不共享 lm_head
                self.draft_model_runner.model.set_embed(embed)

            # grab hot token ids
            # 从草稿模型中获取内置的热门 token ID（EAGLE3 模型内置此信息）
            if self.draft_model_runner.model.hot_token_id is not None:
                self.hot_token_id = self.draft_model_runner.model.hot_token_id.to(
                    embed.device
                )

        else:
            if self.hot_token_id is not None:
                # 克隆 lm_head 并只保留热门 token 对应的行（减小输出投影矩阵规模）
                head = head.clone()
                self.hot_token_id = self.hot_token_id.to(head.device)
                head.data = head.data[self.hot_token_id]

            # Share the embedding and lm_head
            # 草稿模型与目标模型共享 embedding 和 lm_head，减少显存占用
            self.draft_model_runner.model.set_embed_and_head(embed, head)

        # Init attention backend and cuda graphs
        # 恢复 CUDA graph 设置，为后续的注意力后端和计算图初始化做准备
        self.draft_model_runner.server_args.disable_cuda_graph = (
            backup_disable_cuda_graph
        )
        # 根据是否启用 DP 注意力选择正确的 TP 上下文管理器
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        # EAGLE3 使用辅助隐藏状态（aux_hidden_state）来增强草稿质量
        self.eagle_use_aux_hidden_state = False
        if self.speculative_algorithm.is_eagle3():
            self.eagle_use_aux_hidden_state = True
            # 从 HuggingFace 配置中读取 EAGLE3 特定的 eagle_config
            eagle_config = getattr(
                self.draft_model_runner.model_config.hf_config, "eagle_config", {}
            )
            self.eagle_use_aux_hidden_state = eagle_config.get(
                "use_aux_hidden_state", True
            )
        # 在正确的 TP 上下文和 MoE 后端上下文中初始化注意力后端和 CUDA 计算图
        with self.draft_tp_context(
            self.draft_model_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.init_attention_backend()
            self.init_cuda_graphs()
            # 注册当前运行时状态到自适应控制器（用于动态切换投机步数）
            if self.adaptive_controller is not None:
                self.adaptive_controller.register(
                    SpecRuntimeState(
                        speculative_num_steps=self.speculative_num_steps,
                        speculative_num_draft_tokens=self.speculative_num_draft_tokens,
                        draft_attn_backend=self.draft_attn_backend,
                        cuda_graph_runner=self.cuda_graph_runner,
                        target_attn_backend=self.target_worker.model_runner.attn_backend,
                        target_graph_runner=self.target_worker.model_runner.graph_runner,
                        draft_extend_attn_backend=self.draft_extend_attn_backend,
                        cuda_graph_runner_for_draft_extend=self.cuda_graph_runner_for_draft_extend,
                    )
                )
                self.adaptive_controller.init_states()

        # Some dummy tensors
        # 预分配标量张量，避免在推理热路径中动态分配（减少内存碎片）
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

    def init_attention_backend(self):
        # 创建多步草稿注意力后端和 CUDA 计算图 runner
        # DraftBackendFactory 根据配置自动选择合适的注意力实现（FlashAttention/RadixAttention等）
        draft_backend_factory = DraftBackendFactory(
            self.server_args,
            self.draft_model_runner,
            self.topk,
            self.speculative_num_steps,
        )

        # Initialize decode attention backend
        # 草稿解码注意力后端：用于多步草稿 token 生成阶段（decode 模式）
        self.draft_attn_backend = draft_backend_factory.create_decode_backend()

        # Initialize draft extend attention backend (respects speculative_attention_mode setting)
        # 草稿 extend 注意力后端：用于首次 prefill 阶段，根据 speculative_attention_mode 设置
        self.draft_extend_attn_backend = (
            draft_backend_factory.create_draft_extend_backend()
        )

        # 将注意力后端注册到草稿模型 runner（供前向传播使用）
        self.draft_model_runner.draft_attn_backend = self.draft_attn_backend

    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        # 初始化草稿解码 CUDA graph runner 和 extend CUDA graph runner
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        if self.server_args.disable_cuda_graph:
            return

        # 根据硬件类型选择对应的草稿 CUDA graph runner 实现
        Device2DraftCudaGraphRunner = {
            "npu": EAGLEDraftNpuGraphRunner,   # 昇腾 NPU 专用实现
            "cuda": EAGLEDraftCudaGraphRunner,  # NVIDIA CUDA 实现
            "musa": EAGLEDraftCudaGraphRunner,  # 摩尔线程 MUSA 实现（复用 CUDA runner）
        }
        # Capture draft
        # 仅在多步投机（步数>1）时捕获草稿 CUDA graph（单步不值得捕获开销）
        if self.speculative_num_steps > 1:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
            )
            # 捕获草稿模型的 CUDA 计算图（固化 GPU 执行序列以减少 CPU 调度开销）
            self.cuda_graph_runner = Device2DraftCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
            )

        # Capture extend
        # 捕获 extend 阶段的 CUDA graph（NPU 不支持 extend graph）
        if self.draft_extend_attn_backend and not _is_npu:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft extend cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
            )
            # 捕获 extend（prefill）阶段的 CUDA 计算图
            self.cuda_graph_runner_for_draft_extend = EAGLEDraftExtendCudaGraphRunner(
                self
            )
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft extend cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
            )

    def apply_runtime_state(self, state: SpecRuntimeState):
        """Apply a pre-built runtime state to this worker."""
        # 应用预构建的运行时状态（自适应投机解码切换步数时使用）
        if self.speculative_num_steps == state.speculative_num_steps:
            # 步数未变化，无需切换
            return

        logger.info(
            "Switch adaptive runtime state: "
            f"steps {self.speculative_num_steps} -> {state.speculative_num_steps}, "
            f"draft_tokens {self.speculative_num_draft_tokens} -> "
            f"{state.speculative_num_draft_tokens}"
        )

        # 更新投机步数和草稿 token 数量
        self.speculative_num_steps = state.speculative_num_steps
        self.speculative_num_draft_tokens = state.speculative_num_draft_tokens
        # Draft stage
        # 切换草稿注意力后端和 CUDA graph runner
        self.draft_attn_backend = state.draft_attn_backend
        self.draft_model_runner.draft_attn_backend = state.draft_attn_backend
        self.cuda_graph_runner = state.cuda_graph_runner
        # Verify stage
        # 切换目标（验证）模型的注意力后端和 graph runner
        self.target_worker.model_runner.attn_backend = state.target_attn_backend
        self.target_worker.model_runner.graph_runner = state.target_graph_runner
        # Extend stage
        # 切换 extend（prefill）阶段的注意力后端和 CUDA graph runner
        self.draft_extend_attn_backend = state.draft_extend_attn_backend
        self.cuda_graph_runner_for_draft_extend = (
            state.cuda_graph_runner_for_draft_extend
        )
        # Sync server_args
        # 同步 server_args 中的步数配置，确保后续逻辑读取到正确值
        self.server_args.speculative_num_steps = state.speculative_num_steps
        self.server_args.speculative_num_draft_tokens = (
            state.speculative_num_draft_tokens
        )

    def build_adaptive_runtime_state(
        self, speculative_num_steps: int, speculative_num_draft_tokens: int
    ) -> SpecRuntimeState:
        """Build a SpecRuntimeState for the given step configuration."""
        # 构建指定步数配置的运行时状态（注意力后端+CUDA graph），用于自适应切换
        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)

        with self._override_worker_state(
            speculative_num_steps, speculative_num_draft_tokens
        ):
            # Reuse existing init methods for draft attention backend and cuda graphs
            # 在临时覆盖步数参数的上下文中重新初始化注意力后端和 CUDA graph
            self.init_attention_backend()
            self.init_cuda_graphs()

            # Capture target attention backend and CUDA graph
            # 为目标模型创建与新步数配置对应的注意力后端
            target_model_runner = self.target_worker.model_runner
            backup_init = target_model_runner.init_new_workspace
            try:
                # 创建新的目标模型注意力后端（init_new_workspace=True 分配新的 workspace）
                target_attn_backend = target_model_runner._get_attention_backend(
                    init_new_workspace=True
                )
            finally:
                # 恢复 init_new_workspace 标志，避免影响后续正常初始化
                target_model_runner.init_new_workspace = backup_init

            # 为目标模型创建对应步数的 CUDA graph runner（如未禁用）
            target_graph_runner = None
            if not self.server_args.disable_cuda_graph:
                target_graph_runner = CudaGraphRunner(
                    target_model_runner,
                    attn_backend=target_attn_backend,
                    speculative_num_steps=speculative_num_steps,
                    speculative_num_draft_tokens=speculative_num_draft_tokens,
                )

            # 将草稿和目标两侧的状态打包成 SpecRuntimeState
            state = SpecRuntimeState(
                speculative_num_steps=speculative_num_steps,
                speculative_num_draft_tokens=speculative_num_draft_tokens,
                # Draft stage
                draft_attn_backend=self.draft_attn_backend,
                cuda_graph_runner=self.cuda_graph_runner,
                # Verify stage
                target_attn_backend=target_attn_backend,
                target_graph_runner=target_graph_runner,
                # Extend stage
                draft_extend_attn_backend=self.draft_extend_attn_backend,
                cuda_graph_runner_for_draft_extend=self.cuda_graph_runner_for_draft_extend,
            )

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Built adaptive runtime state steps={speculative_num_steps}: "
            f"elapsed={time.perf_counter() - tic:.2f}s, "
            f"mem={(before_mem - after_mem):.2f}GB"
        )

        return state

    @contextmanager
    def _override_worker_state(
        self, speculative_num_steps: int, speculative_num_draft_tokens: int
    ):
        """Temporarily override server_args and worker attributes for graph capture."""
        # 临时覆盖步数相关参数（用于自适应状态构建），退出时自动恢复
        sa = self.server_args
        # 备份当前所有与步数相关的状态，以便 finally 块中恢复
        backup = (
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            self.draft_attn_backend,
            self.draft_extend_attn_backend,
            getattr(self.draft_model_runner, "draft_attn_backend", None),
            getattr(self, "cuda_graph_runner", None),
            getattr(self, "cuda_graph_runner_for_draft_extend", None),
            sa.speculative_num_steps,
            sa.speculative_num_draft_tokens,
        )
        # 临时修改为目标步数配置
        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        sa.speculative_num_steps = speculative_num_steps
        sa.speculative_num_draft_tokens = speculative_num_draft_tokens
        try:
            yield
        finally:
            # 无论是否异常，都恢复原始状态
            (
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
                self.draft_attn_backend,
                self.draft_extend_attn_backend,
                self.draft_model_runner.draft_attn_backend,
                self.cuda_graph_runner,
                self.cuda_graph_runner_for_draft_extend,
                sa.speculative_num_steps,
                sa.speculative_num_draft_tokens,
            ) = backup

    @property
    def draft_model_runner(self):
        # 草稿模型 runner 即为本 worker 的 model_runner（继承自 TpModelWorker）
        return self.model_runner

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Run speculative decoding forward.

        NOTE: Many states of batch is modified as you go through. It is not guaranteed that
        the final output batch have the same state as the input.

        Args:
            batch: The batch to run forward. The state of the batch is modified as it runs.
        Returns:
            A tuple of the final logit output of the target model, next tokens accepted,
            the batch id (used for overlap schedule), and number of accepted tokens.
        """
        # 根据批次的前向模式分流：extend（prefill）模式 vs decode（投机推理）模式
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            # === Prefill（extend）阶段：目标模型先 prefill，再草稿模型 prefill ===
            (
                logits_output,
                next_token_ids,
                seq_lens_cpu,
                can_run_cuda_graph,
            ) = self.forward_target_extend(batch)
            # 在草稿模型的 TP 上下文中运行草稿 extend（利用目标模型的隐藏状态初始化草稿 KV cache）
            with self.draft_tp_context(
                self.draft_model_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                self.forward_draft_extend(
                    batch,
                    logits_output.hidden_states,  # 目标模型输出的完整隐藏状态
                    next_token_ids,
                    seq_lens_cpu,
                    logits_output.mm_input_embeds,  # 多模态输入 embedding（如有）
                )
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_drafts=0,  # prefill 阶段不产生接受的草稿 token
                can_run_cuda_graph=can_run_cuda_graph,
            )
        else:
            # === Decode（投机推理）阶段：草稿 → 验证 → extend ===
            set_time_batch(batch.reqs, "set_spec_draft_start_time", trace_only=True)

            # Step 1: 草稿阶段 —— 草稿模型多步展开，生成候选 token 树
            with self.draft_tp_context(
                self.draft_model_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                spec_info = self.draft(batch)

            set_time_batch(batch.reqs, "set_spec_draft_end_time", trace_only=True)
            set_time_batch(batch.reqs, "set_spec_verify_start_time", trace_only=True)

            # Step 2: 验证阶段 —— 目标模型并行验证候选 token 树，返回接受结果
            logits_output, verify_output, model_worker_batch, can_run_cuda_graph = (
                self.verify(batch, spec_info)
            )

            # 记录每个请求的接受 token 数（用于追踪分析）
            if get_global_tracing_enabled():
                for idx, req in enumerate(batch.reqs):
                    accepted = verify_output.num_accepted_drafts_per_req_cpu[idx]
                    req.time_stats.set_spec_verify_end_time(accepted_tokens=accepted)

            set_time_batch(
                batch.reqs, "set_spec_draft_extend_start_time", trace_only=True
            )

            # Step 3: 草稿 extend 阶段 —— 用验证后接受的 token 更新草稿模型的 KV cache
            with self.draft_tp_context(
                self.draft_model_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                # NOTE: We should use `check_forward_draft_extend_after_decode`
                # when DP attention is enabled, but it is slow. Skip it for now.
                # 若有接受的 token（decode 未完成）或 DP 注意力模式，执行 extend 更新
                if (
                    self.server_args.enable_dp_attention
                    or batch.spec_info.verified_id.shape[0] > 0
                ):
                    # decode is not finished
                    self.forward_draft_extend_after_decode(batch)

            set_time_batch(
                batch.reqs, "set_spec_draft_extend_end_time", trace_only=True
            )

            # 通知自适应控制器本轮验证结果，用于调整下一轮的投机步数
            controller = getattr(self, "adaptive_controller", None)
            if controller is not None:
                controller.on_verify_complete(
                    verify_output.num_accepted_drafts_per_req_cpu
                )

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=verify_output.verified_id,  # 最终接受的 token IDs
                num_accepted_drafts=sum(verify_output.num_accepted_drafts_per_req_cpu),
                num_accepted_drafts_per_req_cpu=verify_output.num_accepted_drafts_per_req_cpu,
                can_run_cuda_graph=can_run_cuda_graph,
            )

    def check_forward_draft_extend_after_decode(self, batch: ScheduleBatch):
        # 检查 decode 后是否需要运行草稿 extend（即是否有 token 被接受）
        local_need_forward = batch.spec_info.verified_id.shape[0] > 0
        if not self.server_args.enable_dp_attention:
            return local_need_forward

        # DP 注意力模式下需要全局 all_reduce，确保所有 DP rank 一致决定是否 forward
        global_need_forward = torch.tensor(
            [
                (local_need_forward),
            ],
            dtype=torch.int64,
        )
        torch.distributed.all_reduce(
            global_need_forward, group=get_tp_group().cpu_group
        )
        global_need_forward_cnt = global_need_forward[0].item()
        need_forward = global_need_forward_cnt > 0
        return need_forward

    def forward_target_extend(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, torch.Tensor, Optional[torch.Tensor], bool]:
        """Run the target extend.

        Args:
            batch: The batch to run. States could be modified.

        Returns:
            logits_output: The output of logits. It will contain the full hidden states.
            next_token_ids: Next token ids generated.
            seq_lens_cpu: CPU copy of sequence lengths for the draft prefill path.
            can_run_cuda_graph: Whether the target prefill ran with cuda graph.
        """
        # Forward with the target model and get hidden states.
        # We need the full hidden states to prefill the KV cache of the draft model.
        # 以 FULL 隐藏状态捕获模式运行目标模型，获取所有位置的隐藏状态（草稿模型 prefill 需要）
        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
        logits_output, next_token_ids = (
            batch_result.logits_output,
            batch_result.next_token_ids,
        )
        return (
            logits_output,
            next_token_ids,
            model_worker_batch.seq_lens_cpu,
            batch_result.can_run_cuda_graph,
        )

    def _draft_preprocess_decode(self, batch: ScheduleBatch):
        # 草稿模型 decode 前处理：滑动窗口 KV 驱逐、计数器更新、内存分配
        batch.maybe_evict_swa()
        for req in batch.reqs:
            # 每次 decode 递增请求的批次索引（用于惩罚器等状态追踪）
            req.decode_batch_idx += 1

        # Parse args
        num_seqs = batch.batch_size()
        spec_info = batch.spec_info

        # Accumulate penalty
        # 若启用了重复惩罚等采样约束，累积已生成的 token（宽松模式：只累积已验证的 token）
        if batch.sampling_info.penalizer_orchestrator.is_required:
            # This is a relaxed version of penalties for speculative decoding.
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                spec_info.verified_id.to(torch.int64)
            )

        # Allocate cache locations
        # Layout of the out_cache_loc
        # [       topk 0         ] [       topk 1         ]
        # [iter=0, iter=1, iter=2] [iter=0, iter=1, iter=2]
        # KV cache 位置分配：为 speculative_num_steps * topk 条路径各分配 KV slot
        if self.page_size == 1:
            # 非分页模式：简单分配 steps * topk 个连续 slot
            alloc_len_per_decode = self.speculative_num_steps * self.topk
            # TODO: We only need self.speculative_num_steps - 1 * topk cache loc
            out_cache_loc, token_to_kv_pool_state_backup = alloc_token_slots(
                batch.tree_cache,
                num_seqs * alloc_len_per_decode,
                backup_state=True,  # 备份状态，验证后若全部拒绝可回滚
            )
        else:
            # 分页模式（page_size > 1）：需要处理部分页的复制和对齐
            if self.topk == 1:
                # topk=1 时无需复制页面，直接获取最后位置
                prefix_lens, seq_lens, last_loc = get_last_loc_large_page_size_top_k_1(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    batch.seq_lens,
                    self.speculative_num_steps,
                )
                prefix_lens_cpu = batch.seq_lens_cpu
                seq_lens_cpu = batch.seq_lens_cpu + self.speculative_num_steps
                extend_num_tokens = num_seqs * self.speculative_num_steps
            else:
                # In this case, the last partial page needs to be duplicated.
                # KV cache layout in batch.req_to_token_pool.req_to_token:
                #
                # | -------- | -- xxxx .. | -- xxxx .. | -- xxxx .. |
                #    prefix     top-k = 0    tok-k = 1    top-k = 2
                #
                #  "-" means prefix tokens
                # topk > 1 时，最后的部分页面需要为每条 topk 路径复制一份
                (
                    prefix_lens,
                    seq_lens,
                    last_loc,
                    self.num_new_pages_per_topk,
                    self.extend_lens,
                    last_page_lens,
                ) = get_last_loc_large_page_size_large_top_k(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    batch.seq_lens,
                    self.speculative_num_steps,
                    self.topk,
                    self.page_size,
                )
                prefix_lens_cpu = batch.seq_lens_cpu
                last_page_lens_cpu = prefix_lens_cpu % self.page_size
                # 计算每个 topk 路径需要的新页面数量
                num_new_pages_per_topk = (
                    last_page_lens_cpu + self.speculative_num_steps + self.page_size - 1
                ) // self.page_size
                # 计算每个序列的目标总长度（按页对齐，再加上 topk 路径的新页）
                seq_lens_cpu = (
                    prefix_lens_cpu // self.page_size * self.page_size
                    + num_new_pages_per_topk * (self.page_size * self.topk)
                )
                extend_num_tokens = torch.sum((seq_lens_cpu - prefix_lens_cpu)).item()

            # 分配分页模式下的 extend token slots
            out_cache_loc, token_to_kv_pool_state_backup = (
                alloc_paged_token_slots_extend(
                    batch.tree_cache,
                    prefix_lens,
                    prefix_lens_cpu,
                    seq_lens,
                    seq_lens_cpu,
                    last_loc,
                    extend_num_tokens,
                    backup_state=True,
                )
            )

        # 分页 + topk > 1 时需要为页面复制准备 source/target cache 位置张量
        if self.page_size > 1 and self.topk > 1:
            last_page_lens_cumsum = torch.cumsum(last_page_lens, dim=0)
            duplicate_cache_len = torch.sum(last_page_lens_cpu).item() * (self.topk - 1)
            # target_cache_loc / source_cache_loc 用于页面内容复制
            target_cache_loc = torch.zeros(
                duplicate_cache_len, dtype=torch.int32, device=self.device
            )
            source_cache_loc = torch.zeros(
                duplicate_cache_len, dtype=torch.int32, device=self.device
            )
        else:
            # When source_cache_loc is not needed, simply skip
            # 无需复制时置为 None，assign_draft_cache_locs 会跳过相关逻辑
            duplicate_cache_len = 0
            source_cache_loc, target_cache_loc, last_page_lens_cumsum = None, None, None

        # 使用 Triton kernel 将分配好的 cache slots 写入 req_to_token_pool
        assign_draft_cache_locs[(num_seqs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            self.extend_lens,
            self.num_new_pages_per_topk,
            out_cache_loc,
            source_cache_loc,
            target_cache_loc,
            last_page_lens_cumsum,
            duplicate_cache_len,
            batch.req_to_token_pool.req_to_token.shape[1],
            self.topk,
            self.speculative_num_steps,
            self.page_size,
            next_power_of_2(num_seqs),
            next_power_of_2(self.speculative_num_steps + self.page_size),
        )

        if self.page_size > 1 and self.topk > 1:
            if duplicate_cache_len > 0:
                # 将 prefix 最后一页的 KV 数据复制到各 topk 路径的对应页（使 topk 路径从相同起点出发）
                self.draft_model_runner.token_to_kv_pool.move_kv_cache(
                    target_cache_loc, source_cache_loc
                )
            # Remove padded slots
            # TODO: We only need self.speculative_num_steps - 1 cache loc
            # 裁剪掉分页对齐时填充的多余 slot，只保留实际需要的位置
            out_cache_loc = out_cache_loc[
                : num_seqs * self.topk * self.speculative_num_steps
            ]

        # 将最终的 out_cache_loc 写入 batch，同时更新序列长度之和和其他状态
        batch.out_cache_loc = out_cache_loc
        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()
        batch.return_hidden_states = False
        # positions：每个 topk 路径共享相同的起始位置（序列当前长度），用于 RoPE 位置编码
        spec_info.positions = batch.seq_lens.repeat_interleave(self.topk, dim=0)
        # 恢复 token_to_kv_pool 备份状态，使分配的 draft KV slots 不计入永久分配（验证后按需保留）
        self.token_to_kv_pool_allocator.restore_state(token_to_kv_pool_state_backup)

    def _draft_preprocess_idle(self, batch: ScheduleBatch):
        # idle 模式下创建占位 EagleDraftInput（用于 DP 注意力中空批次的对齐填充）
        batch.spec_info = EagleDraftInput.create_idle_input(
            device=self.device,
            hidden_size=self.model_config.spec_hidden_size,
            dtype=self.model_config.dtype,
            topk=self.topk,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )

    def draft(self, batch: ScheduleBatch):
        # 草稿阶段主函数：预处理 → 多步展开 → token 树构建
        # Parse args
        if batch.forward_mode.is_idle():
            # idle 模式：创建空的占位输入（不做实际推理）
            self._draft_preprocess_idle(batch)
        else:
            # decode 模式：分配 KV cache 位置并准备草稿输入
            self._draft_preprocess_decode(batch)

        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        # 只需捕获最后一步的隐藏状态（用于下一轮草稿展开的初始状态）
        spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        # 每个请求产生 topk 个 token
        spec_info.num_tokens_per_req = self.topk
        spec_info.num_tokens_for_logprob_per_req = self.topk
        batch.return_hidden_states = False

        # Get forward batch
        # 构建草稿模型专用的前向传播批次对象
        model_worker_batch = batch.get_model_worker_batch()
        assert model_worker_batch.capture_hidden_mode == CaptureHiddenMode.LAST
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.draft_model_runner
        )
        # 检查是否可以使用 CUDA graph 加速（batch size 和 seq len 需在捕获范围内）
        can_cuda_graph = self.cuda_graph_runner and self.cuda_graph_runner.can_run(
            forward_batch
        )
        if can_cuda_graph:
            # CUDA graph replay 模式：直接重放预捕获的计算图，CPU-GPU 通信开销极低
            parent_list, top_scores_index, draft_tokens = self.cuda_graph_runner.replay(
                forward_batch
            )
        else:
            # 非 CUDA graph 模式：逐步运行前向传播
            forward_batch.can_run_dp_cuda_graph = False
            if (
                not forward_batch.forward_mode.is_idle()
                and self.speculative_num_steps > 1
            ):
                # Skip attention backend init for idle mode or 1-step draft
                # 初始化注意力后端元数据（多步草稿需要提前计算 KV 分块信息等）
                self.draft_attn_backend.init_forward_metadata(forward_batch)
            # Run forward steps
            # 执行多步草稿前向传播，返回 token 树的父节点列表、得分索引和草稿 token
            parent_list, top_scores_index, draft_tokens = self.draft_forward(
                forward_batch
            )

        if batch.forward_mode.is_idle():
            # idle 批次不需要真实的验证输入，返回占位对象
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        # 构建高效 token 树：将多步 topk 结果组织成树形结构（带掩码、位置、检索索引等）
        (
            tree_mask,           # 自定义注意力掩码（树形因果掩码）
            position,            # 每个草稿 token 的位置索引
            retrieve_index,      # 从树节点到草稿 token 的检索映射
            retrieve_next_token, # 树节点的下一个 token（用于验证时的接受/拒绝）
            retrieve_next_sibling, # 树节点的兄弟节点（同层 topk 候选）
            draft_tokens,        # 草稿 token 序列
        ) = build_tree_kernel_efficient(
            spec_info.verified_id,  # 上一轮已验证接受的 token（作为树的根）
            parent_list,
            top_scores_index,
            draft_tokens,
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
        )

        # 返回验证输入：包含完整的 token 树信息，供目标模型并行验证
        return EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,            # 树形注意力掩码（非标准因果掩码）
            positions=position,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=CaptureHiddenMode.FULL,  # 验证时需要所有位置的隐藏状态
            seq_lens_sum=forward_batch.seq_lens_sum,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
        )

    def draft_forward(self, forward_batch: ForwardBatch):
        # 草稿模型多步前向传播：通过 topk 选择扩展候选 token 树
        # Parse args
        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        out_cache_loc = forward_batch.out_cache_loc
        # 从 spec_info 中取出：topk 概率、topk 索引、隐藏状态（草稿模型的输入特征）
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        # 调试检测：检查初始 topk_p 中是否有 NaN（异常概率值）
        maybe_detect_nan(topk_p, "draft_forward: NaN in initial topk_p from spec_info")

        # 若使用热门 token 映射，将压缩词表索引映射回原始词表索引
        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]
        # TODO: We only need self.speculative_num_steps - 1 cache loc
        # 重排 cache loc 布局：[batch, topk, steps] → [steps, batch*topk]
        # 使每一步的 cache loc 连续排列，方便按步骤切片
        out_cache_loc = out_cache_loc.reshape(
            forward_batch.batch_size, self.topk, self.speculative_num_steps
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
            self.speculative_num_steps, -1
        )

        # Return values
        # 收集每步的树信息：得分列表、token 列表、父节点列表
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        # Forward multiple steps
        # 逐步展开草稿 token 树（每步生成 topk 个候选）
        scores = None
        for i in range(self.speculative_num_steps):
            # select_top_k_tokens：根据累积得分选择当前步最优的 topk 路径
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])    # 该步各节点的得分
            token_list.append(tree_info[1])    # 该步各节点的 token ID
            parents_list.append(tree_info[2])  # 该步各节点的父节点索引

            # We don't need to run the last forward. we get 1 token from draft prefill and (#spec steps - 1) tokens here
            # 最后一步不需要再跑 forward（已有足够 token 构成树）
            if i == self.speculative_num_steps - 1:
                break

            # Set inputs
            # 准备下一步的输入：input_ids 为当前步选出的 token
            forward_batch.input_ids = input_ids
            # This is a temporary fix for the case that the user is using standalone
            # speculative decoding and the draft model architecture is gpt-oss. gpt-oss
            # rope kernel needs cache_loc to be contiguous.
            # gpt-oss RoPE kernel 要求 cache_loc 连续（独立投机解码时的兼容性修复）
            if (
                self.server_args.speculative_algorithm == "STANDALONE"
                and self.model_config.hf_config.architectures[0] == "GptOssForCausalLM"
            ):
                out_cache_loc = out_cache_loc.contiguous()
            # 切换到当前步的 KV cache 写入位置
            forward_batch.out_cache_loc = out_cache_loc[i]
            # 位置加 1（每步 token 位置递增）
            forward_batch.positions.add_(1)
            # 切换到当前步对应的注意力后端（不同步可能有不同的 KV 长度）
            forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            # 更新草稿模型的隐藏状态（下一步的输入特征）
            spec_info.hidden_states = hidden_states

            # Run forward
            # 执行草稿模型第 i 步前向传播（skip_attn_backend_init=True 跳过重复初始化）
            logits_output = self.draft_model_runner.forward(
                forward_batch, skip_attn_backend_init=True
            ).logits_output
            # 检测 logits NaN（可能由梯度爆炸或数值溢出引起）
            maybe_detect_nan(logits_output.next_token_logits, f"draft_forward step {i}")
            # softmax 获取下一步的 token 概率分布
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            # fast_topk：高效选出概率最高的 topk 个 token 及其概率
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            # 调试检测：确保 topk 索引不越过词表边界
            maybe_detect_oob(
                topk_index,
                0,
                logits_output.next_token_logits.shape[-1],
                f"draft_forward step {i}: topk_index OOB vs vocab_size={logits_output.next_token_logits.shape[-1]}",
            )
            # 若使用热门 token 映射，将压缩词表索引映射回原始词表索引
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            # 更新隐藏状态供下一步使用
            hidden_states = logits_output.hidden_states

        # 整理草稿结果：从每步的 token/score/parent 列表中选出最优路径
        # 返回父节点列表、得分索引和最终草稿 token 序列
        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )

        return parent_list, top_scores_index, draft_tokens

    def clear_cache_pool(self):
        # allocator and kv cache pool are shared with target worker
        # KV cache 池与目标 worker 共享，此处无需清理（由目标 worker 负责）
        pass

    def verify(self, batch: ScheduleBatch, spec_info: EagleVerifyInput):
        # 验证阶段：目标模型并行验证草稿 token 树，采用接受/拒绝机制
        # 保存验证前的序列长度（用于 Mamba 等有状态模型的状态更新）
        seq_lens_pre_verify = batch.seq_lens.clone()
        # 准备验证所需的 KV cache 位置（在 req_to_token_pool 中分配验证 slot）
        spec_info.prepare_for_verify(batch, self.page_size)
        # 验证时每个请求处理 speculative_num_steps + 1 个 token（+1 为最终接受的 token）
        spec_info.num_tokens_per_req = self.speculative_num_steps + 1
        batch.return_hidden_states = False
        # 切换批次模式为 TARGET_VERIFY（目标模型验证）
        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = spec_info

        # 构建目标模型的前向批次（包含完整的草稿 token 树）
        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=spec_info.seq_lens_cpu
        )
        assert model_worker_batch.capture_hidden_mode == spec_info.capture_hidden_mode

        # 若请求有语法约束（结构化输出），提前将草稿 token 和树结构移到 CPU
        # （CPU 操作与 GPU 前向传播并行，减少等待时间）
        if batch.has_grammar:
            retrieve_next_token_cpu = spec_info.retrieve_next_token.cpu()
            retrieve_next_sibling_cpu = spec_info.retrieve_next_sibling.cpu()
            draft_tokens_cpu = spec_info.draft_token.view(
                spec_info.retrieve_next_token.shape
            ).cpu()

        # Forward
        # 目标模型以 TARGET_VERIFY 模式并行处理整棵 token 树
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        vocab_mask = None
        if batch.has_grammar:
            # Generate the logit mask for structured output.
            # Overlap the CPU operations for bitmask generation with the forward pass.
            # 生成结构化输出的 logit 掩码（与 GPU 前向传播重叠，减少等待）
            vocab_mask = generate_token_bitmask(
                batch.reqs,
                spec_info,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )

            if vocab_mask is not None:
                assert spec_info.grammar is not None
                # 将掩码移到 GPU，与目标模型 logits 在同一设备上进行掩码操作
                vocab_mask = vocab_mask.to(spec_info.retrieve_next_token.device)
                # NOTE (sk): otherwise, this vocab mask will be the one from the previous extend stage
                # and will be applied to produce wrong results
                # 清除 batch 上遗留的旧 vocab_mask，避免 extend 阶段的掩码被误用
                batch.sampling_info.vocab_mask = None

        # 检测目标模型 logits 是否有 NaN（数值异常检测）
        maybe_detect_nan(logits_output.next_token_logits, "verify: target model logits")

        # 将目标模型隐藏状态传给 spec_info（后续 extend 阶段使用）
        spec_info.hidden_states = logits_output.hidden_states
        # 执行接受/拒绝验证：比较草稿 token 与目标分布，返回验证结果
        res: EagleVerifyOutput = spec_info.verify(
            batch,
            logits_output,
            self.token_to_kv_pool_allocator,
            self.page_size,
            vocab_mask,
        )

        # Post process based on verified outputs.
        # Pick indices that we care (accepted)
        # 只保留被接受的 token 对应的 logits 和隐藏状态（过滤掉拒绝的草稿 token）
        logits_output.next_token_logits = logits_output.next_token_logits[
            res.accepted_indices
        ]
        logits_output.hidden_states = logits_output.hidden_states[res.accepted_indices]

        # Mamba/GDN/Lightning 等有状态模型需要在验证后更新内部状态
        if (
            self.target_worker.model_runner.hybrid_gdn_config is not None
            or self.target_worker.model_runner.mamba2_config is not None
            or self.target_worker.model_runner.hybrid_lightning_config is not None
        ):
            self._mamba_verify_update(
                batch, res, logits_output, spec_info, seq_lens_pre_verify
            )

        # 若需要 logprob，添加投机解码 v1 格式的 logprob 输出
        if batch.return_logprob:
            add_output_logprobs_for_spec_v1(batch, res, logits_output)

        # Prepare the batch for the next draft forwards.
        # 恢复 batch 到 decode 模式，并将验证结果中的草稿输入设为下一轮的 spec_info
        batch.forward_mode = (
            ForwardMode.DECODE if not batch.forward_mode.is_idle() else ForwardMode.IDLE
        )
        batch.spec_info = res.draft_input

        return logits_output, res, model_worker_batch, can_run_cuda_graph

    def _mamba_verify_update(
        self,
        batch: ScheduleBatch,
        res: EagleVerifyOutput,
        logits_output: LogitsProcessorOutput,
        spec_info: EagleVerifyInput,
        seq_lens_pre_verify: torch.Tensor,
    ):
        # Mamba 等 SSM 模型在投机验证后需要更新内部状态（选择正确的缓存步骤）
        # Under DP attention, some ranks can be IDLE during target verify and never
        # initialize mamba forward metadata for this step.
        if batch.forward_mode.is_idle():
            return

        # 计算每个请求实际接受的 token 数（+1 包含最后生成的新 token）
        accepted_length = (
            torch.tensor(
                res.num_accepted_drafts_per_req_cpu,
                device=logits_output.hidden_states.device,
                dtype=torch.int64,
            )
            + 1
        )
        cumulative_accepted_lengths = torch.cumsum(accepted_length, dim=0)
        # prepend 0 to the cumulative_accepted_lengths
        # 在累积长度前面补 0，用于提取每个请求的第一个接受 token 的全局索引
        accepted_indices_start = torch.cat(
            [
                torch.zeros(
                    1,
                    dtype=cumulative_accepted_lengths.dtype,
                    device=cumulative_accepted_lengths.device,
                ),
                cumulative_accepted_lengths[:-1],
            ]
        )
        # 每个请求在 draft_token_num 步中的起始偏移（用于计算相对步数）
        accepted_indices_offset = torch.arange(
            0,
            len(batch.seq_lens) * batch.spec_info.draft_token_num,
            step=batch.spec_info.draft_token_num,
            dtype=accepted_indices_start.dtype,
            device=accepted_indices_start.device,
        )

        # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
        # res.accepted_indices.shape[0] > 0 skips DP attn idle batch
        # topk > 1 时树形注意力掩码的索引映射更复杂，需要通过 accepted_indices 计算步数
        if spec_info.topk > 1 and res.accepted_indices.shape[0] > 0:
            # accepted_indices=[0,2,3,4,5,7,9,10,11], accepted_length=[4, 3, 2], cumulative_accepted_lengths=[4, 7, 9]
            # first_token_indices_per_req=prepend(0, accepted_indices[cumulative_accepted_lengths[:-1]]) = [0, 5, 10]
            # last_token_indices_per_req=accepted_indices[cumulative_accepted_lengths - 1] = [4, 9, 11] (last token ID of each req)
            # max_relative_indices_per_req = [4,4,1]; those are the per-req spec-decoding step offsets that contain the correct mamba caches
            # first_token_indices_per_req = res.accepted_indices[accepted_indices_start]
            # 取每个请求最后一个接受的 token 的全局索引，减去起始偏移得到相对步数
            accepted_steps = (
                res.accepted_indices[cumulative_accepted_lengths - 1]
                - accepted_indices_offset
            )
        else:
            # topk=1 时直接用接受长度作为步数
            accepted_steps = accepted_length - 1

        if batch.mamba_track_indices is not None:
            # If after verify, the request's seq_lens has crossed a mamba track interval,
            # we need to update the mamba state for the request at the crossing point.
            # 若验证后某些请求跨越了 Mamba track 区间，需在该点更新状态（用于状态持久化）
            mamba_track_interval = self.server_args.mamba_track_interval
            # 检查哪些请求的序列长度跨越了 track 区间边界
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != batch.seq_lens // mamba_track_interval
            )
            tracking_point = (
                batch.seq_lens // mamba_track_interval * mamba_track_interval
            )
            # 计算 track 点在验证步骤序列中的偏移位置
            to_track_ith = torch.clamp(tracking_point - seq_lens_pre_verify - 1, min=0)
            mamba_steps_to_track = torch.where(
                to_track_mask,
                res.accepted_indices[to_track_ith + accepted_indices_start]
                - accepted_indices_offset,
                -1,  # -1 表示该请求无需 track
            )
        else:
            mamba_steps_to_track = None

        # 用正确的接受步骤更新 Mamba 模型的状态缓存
        self.target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify(
            accepted_steps=accepted_steps,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=self.target_worker.model_runner.model,
        )

    def forward_draft_extend(
        self,
        batch: ScheduleBatch,
        hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        mm_input_embeds: Optional[torch.Tensor] = None,
    ):
        """Run draft model extend. This API modifies the states of the batch.

        Args:
            batch: The batch to run.
            hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
        """
        # 用目标模型的隐藏状态和生成的 token 构建草稿模型的 extend 输入
        batch.spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            verified_id=next_token_ids,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )
        batch.return_hidden_states = False
        # 准备 extend 阶段的位置信息和 KV cache 写入位置
        batch.spec_info.prepare_for_extend(batch)
        # extend 阶段只需捕获最后一步的隐藏状态（用于后续草稿 decode 的初始状态）
        batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=seq_lens_cpu
        )
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.draft_model_runner
        )
        forward_batch.return_logprob = False
        # 若有多模态 embedding，传递给草稿模型（视觉特征等）
        if mm_input_embeds is not None:
            forward_batch.mm_input_embeds = mm_input_embeds
        # 运行草稿模型 extend 前向传播，填充草稿模型的 KV cache
        logits_output = self.draft_model_runner.forward(forward_batch).logits_output
        maybe_detect_nan(logits_output.next_token_logits, "draft_extend_for_prefill")
        assert isinstance(forward_batch.spec_info, EagleDraftInput)
        assert forward_batch.spec_info is batch.spec_info
        # 将 extend 输出（logits + hidden_states）转换为下一轮 decode 的 topk_p/topk_index
        self.capture_for_decode(logits_output, forward_batch.spec_info)

    def forward_draft_extend_after_decode(self, batch: ScheduleBatch):
        # decode 后的草稿 extend：用验证接受的 token 更新草稿模型的 KV cache
        assert isinstance(batch.spec_info, EagleDraftInput)
        # Backup fields that will be modified in-place
        # 备份即将被原地修改的字段（extend 会修改 seq_lens 等）
        seq_lens_backup = batch.seq_lens.clone()
        seq_lens_cpu_backup = batch.seq_lens_cpu.clone()
        req_pool_indices_backup = batch.req_pool_indices
        num_accepted_drafts_backup = batch.spec_info.num_accepted_drafts.clone()
        num_accepted_tokens_backup = batch.spec_info.num_accepted_tokens.clone()
        return_logprob_backup = batch.return_logprob

        input_is_idle = batch.forward_mode.is_idle()

        if not input_is_idle and batch.spec_info.verified_id.numel() == 0:
            # 所有 token 被拒绝，创建 idle 批次（跳过真实前向传播）
            batch = batch.copy()
            batch.prepare_for_idle()
            # EAGLE3 使用辅助隐藏状态时需要更大的 hidden_size（3倍）
            hidden_size = (
                self.model_config.hidden_size * 3
                if self.speculative_algorithm.is_eagle3()
                and self.eagle_use_aux_hidden_state
                else self.model_config.spec_hidden_size
            )
            # 创建空的草稿输入占位符（idle 模式不做实际计算）
            batch.spec_info = EagleDraftInput.create_idle_input(
                device=self.device,
                hidden_size=hidden_size,
                dtype=self.model_config.dtype,
                topk=self.topk,
                capture_hidden_mode=CaptureHiddenMode.LAST,
            )

        # 设置 extend 阶段处理的 token 数量（speculative_num_steps + 1 个接受 token）
        batch.spec_info.num_tokens_per_req = self.speculative_num_steps + 1
        batch.spec_info.num_tokens_for_logprob_per_req = 1
        # 准备 extend 后处理（选择验证接受路径上的 token，更新位置和 KV cache 写入位置）
        batch.spec_info.prepare_extend_after_decode(
            batch,
            self.speculative_num_steps,
        )
        # 切换批次模式为 DRAFT_EXTEND（草稿 extend）
        batch.forward_mode = (
            ForwardMode.DRAFT_EXTEND
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )

        batch.return_hidden_states = False
        model_worker_batch = batch.get_model_worker_batch()
        assert model_worker_batch.capture_hidden_mode == CaptureHiddenMode.LAST
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.draft_model_runner
        )
        # 计算序列长度之和（用于注意力后端的批次信息初始化）
        if forward_batch.seq_lens_cpu is not None:
            forward_batch.seq_lens_sum = forward_batch.seq_lens_cpu.sum().item()
        else:
            forward_batch.seq_lens_sum = batch.seq_lens.sum().item()

        # Run
        # 检查是否可以使用 extend CUDA graph 加速
        can_cuda_graph = (
            self.cuda_graph_runner_for_draft_extend
            and self.cuda_graph_runner_for_draft_extend.can_run(forward_batch)
        )
        if can_cuda_graph:
            # CUDA graph 模式：直接读取预计算的 topk_p/topk_index 和 hidden_states
            logits_output = self.cuda_graph_runner_for_draft_extend.replay(
                forward_batch
            )
            forward_batch.spec_info.topk_p, forward_batch.spec_info.topk_index = (
                logits_output.topk_p,
                logits_output.topk_index,
            )
            forward_batch.spec_info.hidden_states = logits_output.hidden_states
        else:
            # 非 CUDA graph 模式：逐步初始化注意力后端并运行前向传播
            forward_batch.can_run_dp_cuda_graph = False
            if not forward_batch.forward_mode.is_idle():
                # 选择 extend 阶段专用后端或默认后端
                attn_backend = (
                    self.draft_extend_attn_backend
                    or self.draft_model_runner.attn_backend
                )
                attn_backend.init_forward_metadata(forward_batch)
                forward_batch.attn_backend = attn_backend
            logits_output = self.draft_model_runner.forward(
                forward_batch, skip_attn_backend_init=True
            ).logits_output
            # 提取 topk_p/topk_index，供下一轮草稿 decode 使用
            self.capture_for_decode(logits_output, forward_batch.spec_info)

        # 检测 extend 输出 logits 的 NaN
        maybe_detect_nan(
            logits_output.next_token_logits,
            f"draft_extend_after_decode (cuda_graph={can_cuda_graph})",
        )

        # Restore backup.
        # This is because `seq_lens` can be modified in `prepare_extend_after_decode`
        # 恢复备份字段（extend 过程中 seq_lens 等字段被修改，此处还原供外层逻辑使用）
        batch.forward_mode = (
            ForwardMode.DECODE if not input_is_idle else ForwardMode.IDLE
        )
        batch.seq_lens = seq_lens_backup
        batch.seq_lens_cpu = seq_lens_cpu_backup
        batch.req_pool_indices = req_pool_indices_backup
        batch.spec_info.num_accepted_drafts = num_accepted_drafts_backup
        batch.spec_info.num_accepted_tokens = num_accepted_tokens_backup
        batch.return_logprob = return_logprob_backup

    def capture_for_decode(
        self, logits_output: LogitsProcessorOutput, draft_input: EagleDraftInput
    ):
        # 将 extend/prefill 的输出转换为 decode 阶段的草稿输入
        # 计算概率并提取 topk（后续 decode 草稿展开的起点）
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        draft_input.topk_p, draft_input.topk_index = fast_topk(probs, self.topk, dim=-1)
        # 保存隐藏状态，作为草稿模型下一步的上下文特征
        draft_input.hidden_states = logits_output.hidden_states

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        # 从张量更新模型权重（在线权重更新接口，同时更新草稿模型和目标模型）
        monkey_patch_torch_reductions()
        named_tensors = MultiprocessingSerializer.deserialize(
            recv_req.serialized_named_tensors[self.tp_rank]
        )
        # 先更新草稿模型权重
        success, message = self.model_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        if not success:
            return success, message

        # 再更新目标模型权重（两者需保持同步）
        success, message = self.target_worker.model_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        return success, message


# 辅助函数：分页模式下 topk=1 时获取序列最后一个 token 的 KV cache 位置
# 使用 torch.compile 进行 JIT 编译优化（NPU/MUSA 不支持，禁用）
@torch.compile(dynamic=True, disable=(_is_npu or _is_musa))
def get_last_loc_large_page_size_top_k_1(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens,
    speculative_num_steps: int,
):
    prefix_lens = seq_lens
    # 目标序列长度 = 当前长度 + 投机步数（为草稿 token 预留位置）
    seq_lens = prefix_lens + speculative_num_steps
    # 获取每个序列最后一个已存储 token 的 KV cache 位置（分页模式下的边界位置）
    last_loc = get_last_loc(
        req_to_token,
        req_pool_indices,
        prefix_lens,
    )
    return prefix_lens, seq_lens, last_loc
