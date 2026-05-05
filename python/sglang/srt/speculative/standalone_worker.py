import logging
from typing import Optional

import torch

# 导入 MoE 专家并行相关的上下文管理器，用于投机解码时隔离 MoE 通信
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
# 导入张量并行 Worker，StandaloneWorker 将通过其初始化草稿模型
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
# 导入 EAGLE Worker 作为基类
from sglang.srt.speculative.eagle_worker import EAGLEWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
# draft_tp_context: 草稿模型专用 TP 通信上下文；load_token_map: 加载热门 token 映射
from sglang.srt.speculative.spec_utils import draft_tp_context, load_token_map
from sglang.srt.utils import empty_context, get_bool_env_var, is_cuda

# 仅在 CUDA 环境下导入位压缩工具（用于树掩码的高效打包）
if is_cuda():
    from sgl_kernel import segment_packbits  # noqa: F401

logger = logging.getLogger(__name__)
# 环境变量控制：是否返回原始目标模型的 logprob（而非投机解码修正后的值）
SGLANG_RETURN_ORIGINAL_LOGPROB = get_bool_env_var("SGLANG_RETURN_ORIGINAL_LOGPROB")


# StandaloneWorker：独立草稿模型 Worker，继承 EAGLEWorker
# 草稿模型与目标模型共享 KV 缓存分配器，但各自维护独立的 KV 缓存池
class StandaloneWorker(EAGLEWorker):

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
        # 保存服务器配置参数
        self.server_args = server_args
        # EAGLE top-k：每步生成的草稿候选 token 数
        self.topk = server_args.speculative_eagle_topk
        # 每次推理的投机步数
        self.speculative_num_steps = server_args.speculative_num_steps
        # 草稿 token 总数（= speculative_num_steps + 1）
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.gpu_id = gpu_id
        self.device = server_args.device
        # 持有目标模型 Worker 的引用，用于验证阶段
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        # 解析投机解码算法类型（EAGLE / EAGLE2 等）
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Override the context length of the draft model to be the same as the target model.
        # 草稿模型上下文长度必须与目标模型一致，保证 KV 缓存形状匹配
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # Do not capture cuda graph in `super().__init__()`
        # It will be captured later.
        # 临时禁用 CUDA Graph 捕获，待注意力后端就绪后再捕获
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True
        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        # 草稿模型与目标模型共享内存池（req→token 映射池 + KV 缓存分配器）
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Load hot token ids
        # 加载热门 token 映射（用于 EAGLE2 的 token 压缩优化）
        if server_args.speculative_token_map is not None:
            self.hot_token_id = load_token_map(server_args.speculative_token_map)
            # 将热门词表大小注入模型参数，使草稿模型只预测热门 token
            server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None

        # Init draft worker
        # 在空上下文和 MoE 上下文中初始化草稿模型（以 TpModelWorker 方式）
        with empty_context(), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            TpModelWorker.__init__(
                self,
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,  # FIXME
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                # 标记为草稿 Worker，使模型加载走草稿路径
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                memory_pool_config=target_worker.model_runner.memory_pool_config,
            )

        # Init attention backend and cuda graphs
        # 恢复 CUDA Graph 开关，并在正确的 TP 上下文中初始化注意力后端和捕获 Graph
        self.draft_model_runner.server_args.disable_cuda_graph = (
            backup_disable_cuda_graph
        )
        # 若启用了 DP Attention，使用草稿模型专用的 TP 通信上下文
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        with self.draft_tp_context(
            self.draft_model_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            # 初始化注意力后端（FlashAttention / FlashInfer 等）
            self.init_attention_backend()
            # 捕获 CUDA Graph，加速草稿模型的后续推理
            self.init_cuda_graphs()

        # Some dummy tensors
        # 预分配标量张量，避免推理时重复分配，占位用
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)
