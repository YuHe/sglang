import contextlib
import logging
from typing import Optional, Tuple

import torch

# 导入环境变量配置（如是否启用 overlap plan stream）
from sglang.srt.environ import envs
# MoE 后端上下文管理器，用于隔离草稿模型的 MoE 通信
from sglang.srt.layers.moe.utils import speculative_moe_backend_context
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_utils import TreeMaskMode
# EagleDraftWorker: 草稿模型基类；EAGLEWorkerV2: EAGLE v2 完整 Worker 基类
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker, EAGLEWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import draft_tp_context
from sglang.srt.utils import empty_context, get_bool_env_var, is_cuda

# 仅在 CUDA 环境下导入位压缩工具（用于树掩码打包）
if is_cuda():
    from sgl_kernel import segment_packbits  # noqa: F401

logger = logging.getLogger(__name__)
# 环境变量：是否返回原始目标模型 logprob
SGLANG_RETURN_ORIGINAL_LOGPROB = get_bool_env_var("SGLANG_RETURN_ORIGINAL_LOGPROB")


def _get_plan_stream(
    device: str,
) -> Tuple[any, contextlib.AbstractContextManager]:
    # 若启用 overlap plan stream，创建独立的 CUDA 流用于并行计划计算
    # 这样推理计划（plan）可与主流（main stream）的计算重叠，提升吞吐量
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        plan_stream = torch.get_device_module(device).Stream()
        plan_stream_ctx = torch.get_device_module(device).stream(plan_stream)
        return plan_stream, plan_stream_ctx
    else:
        # 不启用时返回 None 和空上下文（无额外开销）
        return None, contextlib.nullcontext()


# StandaloneDraftWorker: 独立草稿模型 Worker（V2）
# 与 EagleDraftWorker 的区别：不共享目标模型的 embeddings 和 lm_head
class StandaloneDraftWorker(EagleDraftWorker):
    """Custom EagleDraftWorker that doesn't share embeddings/lm_head with target model."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: int,
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # copy args
        # 保存所有参数，供后续初始化注意力后端和 CUDA Graph 使用
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.nccl_port = nccl_port
        self.target_worker = target_worker
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank

        # Args for easy access
        # 常用参数的快捷访问属性
        self.device = server_args.device
        # EAGLE top-k：每步草稿候选数
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Set constant
        # 设置每次 decode 阶段的 KV 缓存分配长度（取两种策略的较大值）
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        EagleDraftInput.ALLOC_LEN_PER_DECODE = max(
            self.speculative_num_steps * self.topk, self.speculative_num_draft_tokens
        )

        # Do not capture cuda graph in `TpModelWorker` init,
        # will capture later with init_cuda_graphs()
        # 暂时禁用 CUDA Graph 捕获，待注意力后端就绪后再捕获
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True

        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        # 共享目标模型的内存池（req→token 映射 + KV 缓存分配器）
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        with empty_context():
            # Init draft worker
            # 在空上下文中初始化草稿模型（TpModelWorker 实例）
            self.draft_worker = TpModelWorker(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,  # FIXME
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                # 标记为草稿 Worker，模型加载走草稿路径
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                memory_pool_config=target_worker.model_runner.memory_pool_config,
            )

        # Alias for better readability
        # model_runner 的别名，方便后续引用
        self.draft_runner = self.draft_worker.model_runner

        # 初始化 token 映射（热门 token 索引）
        self.init_token_map()
        # 初始化语言模型头（standalone 版本使用草稿模型自身的 lm_head）
        self.init_lm_head()

        # Init attention backend and cuda graphs
        # 恢复 CUDA Graph 开关并在 TP 上下文中初始化注意力后端和捕获 Graph
        self.draft_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        with self.draft_tp_context(
            self.draft_runner.tp_group
        ), speculative_moe_backend_context():
            self.init_attention_backend()
            self.init_cuda_graphs()
        # 默认使用完整树掩码模式
        self.tree_mask_mode = TreeMaskMode.FULL_MASK

        # 获取规划流（用于 overlap 优化）
        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

    def init_lm_head(self):
        """Override to prevent sharing embeddings and lm_head with target model."""
        # For standalone worker, we don't share embeddings and lm_head
        # The draft model uses its own embeddings and lm_head
        # standalone 模式下草稿模型有独立的 lm_head，无需从目标模型共享，因此此方法为空
        pass


# StandaloneWorkerV2: EAGLE V2 独立工作进程，组合目标模型和独立草稿模型
class StandaloneWorkerV2(EAGLEWorkerV2):

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
        target_worker: TpModelWorker,
    ):
        # Parse arguments
        # 保存并初始化基本参数
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.gpu_id = gpu_id
        self.device = server_args.device
        # 目标模型 Worker，用于验证阶段（使用 _target_worker 私有属性以匹配 V2 接口）
        self._target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # 从目标模型获取内存池，草稿模型与目标模型共享分配器
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Override the context length of the draft model to be the same as the target model.
        # 草稿模型上下文长度与目标模型对齐
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # Create our custom draft worker that doesn't share embeddings/lm_head
        # 创建不共享 embeddings/lm_head 的独立草稿 Worker
        self._draft_worker = StandaloneDraftWorker(
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            attn_cp_rank,
            moe_dp_rank,
            nccl_port,
            target_worker,
        )

        # Some dummy tensors
        # 预分配标量占位张量，避免推理时重复分配
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        # 获取规划流（支持 overlap 优化）
        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)
