# EAGLE Worker v2 模块：EAGLE 投机解码草稿 worker 的改进版本
# 相比 v1：使用新的 BaseDraftWorker/BaseSpecWorker 基类，支持流水线重叠、更细粒度的模块化
import contextlib
import logging
import time
from typing import List, Optional, Tuple

import torch

# 导入环境变量配置（如 SGLANG_ENABLE_OVERLAP_PLAN_STREAM）
from sglang.srt.environ import envs
# 导入 NPU 硬件的 extend 阶段 CUDA graph runner
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_extend_npu_graph_runner import (
    EAGLEDraftExtendNpuGraphRunner,
)
# 导入 NPU 硬件的草稿 CUDA graph runner
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_npu_graph_runner import (
    EAGLEDraftNpuGraphRunner,
)
# 导入 Triton 和 TRT-LLM MLA 注意力后端（用于判断是否支持 extend graph）
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
)
# 导入 DP 注意力的 TP 组获取工具
from sglang.srt.layers.dp_attention import get_attention_tp_group
# 导入 MoE 投机后端上下文管理器
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
# 导入投机解码 v2 的 logprob 计算工具
from sglang.srt.layers.utils.logprob import compute_spec_v2_logprobs
# 导入权重更新相关请求数据结构
from sglang.srt.managers.io_struct import (
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
)
# 导入模型 worker 批次和生成结果数据结构
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
# 导入前向传播批次信息（隐藏状态捕获模式和前向批次）
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.server_args import ServerArgs
# 导入投机解码基类（v2 架构使用 BaseDraftWorker 和 BaseSpecWorker）
from sglang.srt.speculative.base_spec_worker import BaseDraftWorker, BaseSpecWorker
# 导入草稿后端工厂
from sglang.srt.speculative.draft_utils import DraftBackendFactory
# 导入草稿 CUDA graph runner（CUDA/NPU/MUSA 版本）
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
# 导入 extend 阶段 CUDA graph runner
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
# 导入 EAGLE 草稿输入和验证输入数据结构
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
# 导入 v2 版 EAGLE 信息工具函数（extend cache 分配、接受 token 填充等）
from sglang.srt.speculative.eagle_info_v2 import (
    assign_extend_cache_locs,     # 分配 extend 阶段的 KV cache 位置
    fill_accepted_out_cache_loc,  # 填充被接受 token 的 cache 位置
    fill_new_verified_id,         # 填充新的验证 token ID
)
# 导入 token 树构建工具（含 TreeMaskMode 枚举）
from sglang.srt.speculative.eagle_utils import TreeMaskMode, build_tree_kernel_efficient
# 导入投机解码算法枚举
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
# 导入通用投机解码工具函数
from sglang.srt.speculative.spec_utils import (
    draft_tp_context,        # 草稿模型 TP 上下文管理器
    generate_token_bitmask,  # 结构化输出 token 位掩码生成
    load_token_map,          # 加载热门 token 映射
    maybe_detect_nan,        # NaN 检测
    maybe_detect_oob,        # 越界检测
    select_top_k_tokens,     # 选择 topk tokens
)
# 导入通用工具函数
from sglang.srt.utils.common import (
    MultiprocessingSerializer,   # 多进程序列化
    empty_context,               # 空上下文管理器
    fast_topk,                   # 快速 top-k 选择
    get_available_gpu_memory,    # 获取可用 GPU 内存
    is_cuda,                     # CUDA 检测
    is_hip,                      # ROCm HIP 检测
    is_musa,                     # MUSA 检测
    is_npu,                      # NPU 检测
    next_power_of_2,             # 向上取最近 2 的幂
)
# 导入 torch 猴子补丁
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

# 全局设备类型标志
_is_npu = is_npu()
_is_cuda = is_cuda()
_is_musa = is_musa()
_is_hip = is_hip()

logger = logging.getLogger(__name__)


def _get_plan_stream(
    device: str,
) -> Tuple[any, contextlib.AbstractContextManager]:
    # 获取用于注意力元数据计算的独立 CUDA stream（用于与主 stream 重叠执行）
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        # 创建独立的 plan stream 和对应上下文管理器
        plan_stream = torch.get_device_module(device).Stream()
        plan_stream_ctx = torch.get_device_module(device).stream(plan_stream)
        return plan_stream, plan_stream_ctx
    else:
        # 不启用重叠时，使用空上下文（与主 stream 串行执行）
        return None, contextlib.nullcontext()


# EagleDraftWorker v2：EAGLE 草稿 worker 的 v2 实现
# 使用 BaseDraftWorker 基类，支持更细粒度的接口分离
class EagleDraftWorker(BaseDraftWorker):
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
        target_worker: TpModelWorker,  # 目标（验证）模型 worker
    ):
        # copy args
        # 保存服务器配置和 worker 标识参数
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
        self.device = server_args.device
        # top-k 数量：每步草稿展开保留的候选 token 数
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        # 投机解码算法类型（EAGLE/EAGLE2/EAGLE3）
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Do not capture cuda graph in `TpModelWorker` init,
        # will capture later with init_cuda_graphs()
        # 先禁用 CUDA graph，避免在 TpModelWorker 初始化时提前捕获
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True

        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        # 与目标 worker 共享请求-token 映射池和 KV cache 分配器
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Init draft worker
        # EAGLE3 + DP 注意力时需要使用专用 TP 上下文
        if server_args.enable_dp_attention and self.speculative_algorithm.is_eagle3():
            ctx = draft_tp_context(get_attention_tp_group())
        else:
            ctx = empty_context()
        with (
            ctx
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            # Init draft worker
            # 初始化草稿 TpModelWorker（加载草稿模型权重和 KV cache）
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
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                memory_pool_config=target_worker.model_runner.memory_pool_config,
            )

        # Alias for better readability
        # draft_runner 是草稿模型的 model_runner（简化引用）
        self.draft_runner = self.draft_worker.model_runner
        # EAGLE3 可选使用辅助隐藏状态（来自目标模型多层特征）
        self.eagle_use_aux_hidden_state = False
        if self.speculative_algorithm.is_eagle3():
            eagle_config = getattr(
                self.draft_runner.model_config.hf_config, "eagle_config", {}
            )
            self.eagle_use_aux_hidden_state = eagle_config.get(
                "use_aux_hidden_state", True
            )
        # 初始化热门 token 映射（减少词表搜索范围）
        self.init_token_map()
        # 初始化 embedding 和 lm_head 共享（与目标模型共享权重）
        self.init_lm_head()

        # Init attention backend and cuda graphs
        # 恢复 CUDA graph 设置，准备初始化注意力后端
        self.draft_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        with self.draft_tp_context(
            self.draft_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.init_attention_backend()
            self.init_cuda_graphs()

        # token 树掩码模式（FULL_MASK：完整树形注意力掩码）
        self.tree_mask_mode = TreeMaskMode.FULL_MASK

        # 获取用于注意力 plan 与前向传播重叠的独立 stream
        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

    def init_token_map(self):
        # Load hot token ids
        # 加载热门 token 映射表（EAGLE 草稿模型的词表压缩）
        if self.speculative_algorithm.is_eagle3():
            # EAGLE3 模型内置热门 token 映射，忽略外部指定的映射文件
            if self.server_args.speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None
        elif self.server_args.speculative_token_map is not None:
            # 从文件加载热门 token ID 列表，并更新模型参数
            self.hot_token_id = load_token_map(self.server_args.speculative_token_map)
            self.server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None

    def init_lm_head(self):
        # 从目标模型获取 embedding 和 lm_head 并共享给草稿模型
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        if self.speculative_algorithm.is_eagle3():
            # most cases EAGLE3 models don't share lm_head
            # but some models (e.g. nvidia/gpt-oss-120b-Eagle3) shares
            if (
                hasattr(self.draft_runner.model, "load_lm_head_from_target")
                and self.draft_runner.model.load_lm_head_from_target
            ):
                # 完整共享 embedding + lm_head
                self.draft_runner.model.set_embed_and_head(embed, head)
            else:
                # 仅共享 embedding
                self.draft_runner.model.set_embed(embed)

            # grab hot token ids
            # 从 EAGLE3 模型中提取内置的热门 token ID
            if self.draft_runner.model.hot_token_id is not None:
                self.hot_token_id = self.draft_runner.model.hot_token_id.to(
                    embed.device
                )

        else:
            if self.hot_token_id is not None:
                # 克隆 lm_head 并只保留热门 token 对应行
                head = head.clone()
                self.hot_token_id = self.hot_token_id.to(head.device)
                head.data = head.data[self.hot_token_id]

            # Share the embedding and lm_head
            # 草稿模型与目标模型共享 embedding 和 lm_head（节省显存）
            self.draft_runner.model.set_embed_and_head(embed, head)

    def init_attention_backend(self):
        # Create multi-step attn backends and cuda graph runners
        # 初始化草稿 decode 和 extend 两种注意力后端

        self.has_prefill_wrapper_verify = False
        self.draft_extend_attn_backend = None

        # 使用工厂模式创建注意力后端（自动适配不同的后端实现）
        draft_backend_factory = DraftBackendFactory(
            self.server_args,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )

        # Initialize decode attention backend
        # 草稿 decode 注意力后端：用于多步 token 树展开
        self.draft_attn_backend = draft_backend_factory.create_decode_backend()

        # Initialize draft extend attention backend (respects speculative_attention_mode setting)
        # 草稿 extend 注意力后端：用于 prefill 和 decode 后更新 KV cache
        self.draft_extend_attn_backend = (
            draft_backend_factory.create_draft_extend_backend()
        )

        # 将 decode 注意力后端注册到草稿 runner
        self.draft_runner.draft_attn_backend = self.draft_attn_backend
        self.tree_mask_mode = TreeMaskMode.FULL_MASK

    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        # 捕获草稿 decode 和 extend 两种 CUDA 计算图
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        if self.server_args.disable_cuda_graph:
            return

        # MindSpore 框架不支持 CUDA graph
        if self.server_args.model_impl == "mindspore":
            return

        # 不同硬件平台的草稿 decode CUDA graph runner 映射
        Device2DraftCudaGraphRunner = {
            "npu": EAGLEDraftNpuGraphRunner,
            "cuda": EAGLEDraftCudaGraphRunner,
            "musa": EAGLEDraftCudaGraphRunner,
        }
        # Capture draft
        # 多步投机（>1步）时捕获草稿 decode CUDA graph
        if self.speculative_num_steps > 1:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
            )
            self.cuda_graph_runner = Device2DraftCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
            )

        # 不同硬件平台的草稿 extend CUDA graph runner 映射
        Device2ExtendCudaGraphRunner = {
            "npu": EAGLEDraftExtendNpuGraphRunner,
            "cuda": EAGLEDraftExtendCudaGraphRunner,
            "musa": EAGLEDraftCudaGraphRunner,
        }
        supports_hip_aiter_draft_extend_graph = False
        if _is_hip:
            # Keep import local so non-HIP environments do not require aiter.
            # ROCm HIP 设备：检查是否使用 Aiter 多步草稿后端（支持 extend graph）
            from sglang.srt.layers.attention.aiter_backend import (
                AiterMultiStepDraftBackend,
            )

            supports_hip_aiter_draft_extend_graph = isinstance(
                self.draft_attn_backend, AiterMultiStepDraftBackend
            )

        # CUDA/MUSA 设备：检查是否使用 Triton 或 TRT-LLM MLA 后端（支持 extend graph）
        supports_cuda_draft_extend_graph = (_is_cuda or _is_musa) and (
            isinstance(self.draft_extend_attn_backend, TritonAttnBackend)
            or isinstance(self.draft_extend_attn_backend, TRTLLMMLABackend)
        )
        # Capture extend
        # TODO: support draft extend cuda graph for more attention backends
        # 仅在满足条件的后端上捕获 extend CUDA graph
        if self.draft_extend_attn_backend and (
            _is_npu
            or supports_cuda_draft_extend_graph
            or supports_hip_aiter_draft_extend_graph
        ):
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft extend cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
            )
            self.cuda_graph_runner_for_draft_extend = Device2ExtendCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft extend cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
            )

    def draft(self, model_worker_batch: ModelWorkerBatch):
        # 草稿阶段主入口：准备输入 → 多步展开 → token 树构建
        draft_input: EagleDraftInput = model_worker_batch.spec_info
        # 准备 v2 草稿前向传播批次（分配 KV cache、构建前向批次、判断是否可用 CUDA graph）
        forward_batch, can_cuda_graph = draft_input.prepare_for_v2_draft(
            self.req_to_token_pool,
            model_worker_batch,
            self.cuda_graph_runner,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )

        # Run draft
        if can_cuda_graph:
            # CUDA graph 加速：重放预捕获的多步草稿计算图
            parent_list, top_scores_index, draft_tokens = self.cuda_graph_runner.replay(
                forward_batch,
            )
        else:
            if (
                not forward_batch.forward_mode.is_idle()
                and self.speculative_num_steps > 1
            ):
                # Skip attention backend init for 1-step draft,
                # `draft_forward` only does sample in this case.
                # 多步模式需要初始化注意力后端元数据（1步模式仅采样，无需初始化）
                self.draft_attn_backend.init_forward_metadata(forward_batch)
            parent_list, top_scores_index, draft_tokens = self.draft_forward(
                forward_batch
            )

        if model_worker_batch.forward_mode.is_idle():
            # idle 模式：直接返回空的验证输入
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        # Build tree mask
        # Directly write to cuda graph buffers for verify attn
        # v2 改进：直接写入验证阶段 CUDA graph 的预分配 buffer（减少数据复制）
        tree_mask_buf, position_buf = (
            self.target_worker.model_runner.attn_backend.get_verify_buffers_to_fill_after_draft()
        )

        # 从多步 topk 结果中高效构建 token 树（掩码、位置、检索索引等）
        (
            tree_mask,
            position,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            draft_input.verified_id,
            parent_list,
            top_scores_index,
            draft_tokens,
            model_worker_batch.seq_lens,
            model_worker_batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            self.tree_mask_mode,    # 树掩码模式（FULL_MASK/其他）
            tree_mask_buf,          # 预分配的树掩码 buffer
            position_buf,           # 预分配的位置 buffer
        )

        # 返回验证输入（v2 中部分字段 lazy 初始化，capture_hidden_mode 由上层设置）
        return EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=None,  # v2 中由 EAGLEWorkerV2 负责设置
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )

    def draft_forward(self, forward_batch: ForwardBatch):
        # 草稿模型多步前向传播：与 eagle_worker.py 的同名函数逻辑相同
        # Parse args
        spec_info: EagleDraftInput = forward_batch.spec_info
        out_cache_loc = forward_batch.out_cache_loc
        # 提取草稿输入：topk 概率、topk 索引、隐藏状态
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        # 检测初始 topk_p 中的 NaN（数值异常检测）
        maybe_detect_nan(topk_p, "draft_forward: NaN in initial topk_p from spec_info")

        # 若使用热门 token 映射，将压缩词表索引映射回原始词表索引
        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]

        # 重排 cache loc 布局：[batch, topk, steps] → [steps, batch*topk]
        out_cache_loc = out_cache_loc.reshape(
            forward_batch.batch_size, self.topk, self.speculative_num_steps
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
            self.speculative_num_steps, -1
        )

        # Return values
        # 收集每步的树信息
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        # Forward multiple steps
        # 多步前向传播，逐步展开 token 树
        scores = None
        for i in range(self.speculative_num_steps):
            # 根据累积得分选取当前步最优的 topk 路径
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            # We don't need to run the last forward. we get 1 token from draft prefill and (#spec steps - 1) tokens here
            # 最后一步不需要运行 forward（已有足够 token 构成树）
            if i == self.speculative_num_steps - 1:
                break

            # Set inputs
            # 准备下一步的输入 token 和 KV cache 写入位置
            forward_batch.input_ids = input_ids
            forward_batch.out_cache_loc = out_cache_loc[i]
            forward_batch.positions.add_(1)
            # 切换到当前步对应的注意力后端
            forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states

            # Run forward
            # 执行草稿模型第 i 步前向传播
            logits_output = self.draft_runner.forward(
                forward_batch, skip_attn_backend_init=True
            ).logits_output
            maybe_detect_nan(logits_output.next_token_logits, f"draft_forward step {i}")
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            maybe_detect_oob(
                topk_index,
                0,
                logits_output.next_token_logits.shape[-1],
                f"draft_forward step {i}: topk_index OOB vs vocab_size={logits_output.next_token_logits.shape[-1]}",
            )
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = logits_output.hidden_states

        # Organize the results
        # 整理多步结果：将得分展平并选出最优的 speculative_num_draft_tokens-1 个候选
        score_list = torch.cat(score_list, dim=1).flatten(
            1
        )  # b, n, topk; n= 1 + (num_steps-1) * self.topk
        ss_token_list = torch.cat(
            token_list, dim=1
        )  # b, (self.topk + (num_steps-1) * self.topk)
        # 从所有候选中选出得分最高的 speculative_num_draft_tokens-1 个
        top_scores = torch.topk(
            score_list, self.speculative_num_draft_tokens - 1, dim=-1
        )
        top_scores_index = top_scores.indices
        # 排序以保证 token 树的拓扑顺序（父节点索引单调递增）
        top_scores_index = torch.sort(top_scores_index).values
        maybe_detect_oob(
            top_scores_index,
            0,
            ss_token_list.shape[1],
            "draft_forward: top_scores_index OOB for gather on ss_token_list",
        )
        # 根据排序后的索引收集对应的草稿 token
        draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

        # 构建父节点列表（排除最后一步，因为最后一步不需要再展开）
        if len(parents_list) > 1:
            parent_list = torch.cat(parents_list[:-1], dim=1)
        else:
            # 单步时无父节点列表（空张量）
            batch_size = parents_list[0].shape[0]
            parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

        return parent_list, top_scores_index, draft_tokens

    def draft_extend(self):
        # 占位方法（v2 中 extend 逻辑由 _draft_extend_for_prefill 和 _draft_extend_for_decode 处理）
        pass

    def _draft_extend_for_prefill(
        self,
        batch: ModelWorkerBatch,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        mm_input_embeds: Optional[torch.Tensor] = None,
    ):
        """
        Run draft model extend to correctly fill the KV cache.

        Args:
            batch: The batch to run.
            target_hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
        """
        # Construct input_ids
        # 将 input_ids 向后移动一位（原始序列 + 目标模型生成的新 token）
        if not batch.forward_mode.is_idle():
            pt = 0
            for i, extend_len in enumerate(batch.extend_seq_lens):
                input_ids = batch.input_ids[pt : pt + extend_len]
                # 移除首个 token，追加新生成的 token 到末尾（EAGLE 的 shift 操作）
                batch.input_ids[pt : pt + extend_len] = torch.cat(
                    (input_ids[1:], next_token_ids[i].reshape(1))
                )
                pt += extend_len

        # Construct spec_info
        # 构建草稿输入（以目标模型的隐藏状态和新生成 token 为初始状态）
        next_draft_input = EagleDraftInput(
            hidden_states=target_hidden_states,
            verified_id=next_token_ids,
            new_seq_lens=batch.seq_lens,
            # draft mode is same with decode mode, only 1 token per req
            # prefill 后草稿 extend 每个请求只处理 1 个 token
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )

        batch.spec_info = next_draft_input

        # Run forward
        # 运行草稿模型 prefill extend，填充草稿 KV cache
        forward_batch = ForwardBatch.init_new(batch, self.draft_runner)
        forward_batch.return_logprob = False
        if mm_input_embeds is not None:
            forward_batch.mm_input_embeds = mm_input_embeds
        logits_output = self.draft_runner.forward(forward_batch).logits_output
        maybe_detect_nan(logits_output.next_token_logits, "draft_extend_for_prefill")

        # Update spec_info for the next draft step
        # 从 logits 计算 topk，更新草稿输入供下一步 decode 使用
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        next_draft_input.topk_p, next_draft_input.topk_index = fast_topk(
            probs, self.topk, dim=-1
        )
        next_draft_input.hidden_states = logits_output.hidden_states
        return next_draft_input

    def _draft_extend_for_decode(
        self, batch: ModelWorkerBatch, batch_result: GenerationBatchResult
    ):
        # Batch 2: Draft extend
        # decode 后的草稿 extend：用验证接受的 token 更新草稿模型 KV cache
        # 使用目标模型验证后的隐藏状态初始化草稿 extend
        draft_input = EagleDraftInput(
            hidden_states=batch_result.logits_output.hidden_states,
            num_tokens_per_req=self.speculative_num_steps + 1,
            num_tokens_for_logprob_per_req=self.speculative_num_steps + 1,
        )
        # 计算每个请求最后一个接受 token 在 hidden_states 中的索引
        select_index = (
            torch.arange(len(batch.seq_lens), device=self.device)
            * self.speculative_num_draft_tokens
            + batch_result.accept_lens
            - 1
        )

        # Prepare for draft extend in a separate stream
        # 在独立 stream 中准备 extend 批次（与主 stream 计算重叠）
        with self.plan_stream_ctx:
            forward_batch = draft_input.prepare_for_extend_to_fill_draft_kvcache(
                batch,
                batch_result.next_token_ids,
                self.speculative_num_draft_tokens,
                self.draft_runner,
                self.cuda_graph_runner_for_draft_extend,
            )

        if self.plan_stream:
            # 等待 plan stream 完成，确保主 stream 使用最新的批次信息
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )

        if forward_batch.spec_info.num_accepted_drafts is None:
            # `batch_result.accept_lens` already includes the bonus token, so use it
            # directly for `num_accepted_tokens` and subtract 1 for `num_accepted_drafts`.
            # accept_lens 包含 bonus token，直接作为 num_accepted_tokens
            forward_batch.spec_info.num_accepted_drafts = batch_result.accept_lens - 1
            forward_batch.spec_info.num_accepted_tokens = batch_result.accept_lens

        # Run draft extend batch in the main compute stream
        # 在主计算 stream 中运行草稿 extend（可使用 CUDA graph 加速）
        can_cuda_graph = (
            self.cuda_graph_runner_for_draft_extend
            and self.cuda_graph_runner_for_draft_extend.can_run(forward_batch)
        )
        if can_cuda_graph:
            draft_logits_output = self.cuda_graph_runner_for_draft_extend.replay(
                forward_batch
            )
        else:
            draft_logits_output = self.draft_runner.forward(
                forward_batch, skip_attn_backend_init=True
            ).logits_output

        maybe_detect_nan(
            draft_logits_output.next_token_logits,
            f"draft_extend_for_decode (cuda_graph={can_cuda_graph})",
        )

        # Reorganize the spec info for the next batch
        # 只保留每个请求最后一个接受 token 对应的 logits 和隐藏状态
        draft_logits_output.next_token_logits = draft_logits_output.next_token_logits[
            select_index
        ]
        draft_logits_output.hidden_states = draft_logits_output.hidden_states[
            select_index
        ]
        probs = torch.softmax(draft_logits_output.next_token_logits, dim=-1)
        # 计算下一轮草稿 decode 的 topk 概率和索引
        ret_topk_p, ret_topk_index = fast_topk(probs, self.topk, dim=-1)
        ret_hidden_states = draft_logits_output.hidden_states

        # Construct the return values
        # 将计算结果写入下一轮草稿输入对象（由 batch_result 携带）
        next_draft_input = batch_result.next_draft_input
        (
            next_draft_input.topk_p,
            next_draft_input.topk_index,
            next_draft_input.hidden_states,
        ) = (
            ret_topk_p,
            ret_topk_index,
            ret_hidden_states,
        )


class EAGLEWorkerV2(BaseSpecWorker):
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
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.tp_rank = tp_rank
        self.gpu_id = gpu_id
        self.device = server_args.device
        self._target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # 从目标 worker 获取共享的内存池
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Override the context length of the draft model to be the same as the target model.
        # 草稿模型上下文长度与目标模型对齐（确保 KV cache 结构兼容）
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # 创建 EagleDraftWorker（封装草稿模型的所有操作）
        self._draft_worker = EagleDraftWorker(
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
        # 预分配标量张量，避免热路径动态分配
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        # 获取用于流水线重叠的独立 plan stream
        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

    @property
    def target_worker(self):
        return self._target_worker

    @property
    def draft_worker(self):
        return self._draft_worker

    def clear_cache_pool(self):
        # allocator and kv cache pool are shared with target worker, which are cleared in scheduler
        # KV cache 池与目标 worker 共享，由 scheduler 负责清理
        pass

    def forward_batch_generation(self, model_worker_batch: ModelWorkerBatch):
        # 根据批次模式（extend/decode）选择不同的处理路径
        if (
            model_worker_batch.forward_mode.is_extend()
            or model_worker_batch.is_extend_in_batch
        ):
            # Target prefill
            # Step 1: 以 FULL 模式运行目标模型 prefill（获取所有位置的隐藏状态）
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
            batch_output = self.target_worker.forward_batch_generation(
                model_worker_batch
            )

            # Draft prefill
            # Step 2: 以目标模型输出的隐藏状态运行草稿模型 extend（填充草稿 KV cache）
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
            with self.draft_worker.draft_tp_context(
                self.draft_worker.draft_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                batch_output.next_draft_input = (
                    self.draft_worker._draft_extend_for_prefill(
                        model_worker_batch,
                        batch_output.logits_output.hidden_states,
                        batch_output.next_token_ids,
                        batch_output.logits_output.mm_input_embeds,
                    )
                )
                return batch_output
        else:
            # decode（投机推理）模式：草稿 → 验证 → extend
            if model_worker_batch.spec_info is None:
                # 若无 spec_info，创建 idle 输入（DP 注意力中空批次对齐用）
                model_worker_batch.spec_info = EagleDraftInput.create_idle_input(
                    device=self.device,
                    hidden_size=self.target_worker.model_config.spec_hidden_size,
                    dtype=self.target_worker.model_config.dtype,
                    topk=self.topk,
                    capture_hidden_mode=CaptureHiddenMode.LAST,
                )
            # Step 1: 草稿阶段 —— 生成候选 token 树
            with self.draft_worker.draft_tp_context(
                self.draft_worker.draft_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                verify_input: EagleVerifyInput = self.draft_worker.draft(
                    model_worker_batch
                )
            assert verify_input.is_verify_input()
            # Record a CUDA event after draft() GPU work is dispatched.
            # This event will be waited on by plan_stream in verify()
            # to ensure draft CUDA graph kernels finish before plan_stream
            # begins metadata preparation.
            # 记录草稿完成的 CUDA event，供 plan_stream 等待（精确控制重叠边界）
            if self.plan_stream:
                self._draft_done_event = torch.get_device_module(self.device).Event()
                self._draft_done_event.record()
            model_worker_batch.spec_info = verify_input
            # Step 2: 验证阶段 —— 目标模型并行验证 token 树
            batch_output = self.verify(model_worker_batch)
            # Step 3: 草稿 extend 阶段 —— 更新草稿 KV cache（使用验证接受的路径）
            with self.draft_worker.draft_tp_context(
                self.draft_worker.draft_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                self.draft_worker._draft_extend_for_decode(
                    model_worker_batch, batch_output
                )
            return batch_output

    def verify(self, batch: ModelWorkerBatch):
        # Since batch.seq_lens is allocated in another stream, we need
        # record_stream() to prevent pytorch gc and reuse the gpu memory
        # while forward_stream is still running.
        # 防止 PyTorch GC 回收跨 stream 使用的 seq_lens 张量
        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )

        # Parse args
        verify_input: EagleVerifyInput = batch.spec_info
        # 验证时每个请求处理 speculative_num_steps + 1 个 token
        verify_input.num_tokens_per_req = self.speculative_num_steps + 1
        bs = len(batch.seq_lens)

        # Batch 1: Target verify
        # Prepare for target verify in a separate stream
        # 在独立 plan_stream 中准备验证批次（与主 stream 的草稿计算重叠）
        with self.plan_stream_ctx:
            # Wait for the draft CUDA graph to finish before plan_stream
            # begins its work. Using an event is more targeted than
            # wait_stream(main_stream) — it only waits for draft GPU
            # work, not all queued main_stream operations.
            # 等待草稿 CUDA graph 完成（精确等待，不阻塞无关的主 stream 操作）
            if self.plan_stream and hasattr(self, "_draft_done_event"):
                self.plan_stream.wait_event(self._draft_done_event)
            # 准备验证前向批次（分配 KV cache、设置注意力掩码等）
            verify_forward_batch, can_run_cuda_graph = (
                verify_input.prepare_for_v2_verify(
                    self.req_to_token_pool,
                    batch,
                    self.target_worker,
                )
            )

        # Correct some buffers due to the overlap plan
        # 等待 plan_stream 完成，确保批次信息已更新到最新值
        if self.plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )

            # Some values such as custom_mask and position depend on the output of draft,
            # so the previous plan step used the wrong values. Here, we need to run the related
            # computation again to update them to the correct values.
            # 重新计算依赖草稿输出的缓冲区（custom_mask 和 position 在 plan_stream 中使用了旧值）
            self.target_worker.model_runner.attn_backend.update_verify_buffers_to_fill_after_draft(
                verify_input,
                (
                    self.target_worker.model_runner.graph_runner.bs
                    if can_run_cuda_graph
                    else None
                ),
            )

        # Prepare grammar data on CPU if needed
        # 若有语法约束，提前将草稿 token 和树结构移到 CPU（与 GPU 验证重叠）
        if batch.has_grammar:
            retrieve_next_token_cpu = verify_input.retrieve_next_token.cpu()
            retrieve_next_sibling_cpu = verify_input.retrieve_next_sibling.cpu()
            draft_tokens_cpu = verify_input.draft_token.view(
                verify_input.retrieve_next_token.shape
            ).cpu()

        # Run target verify batch in the main compute stream (GPU compute)
        # 目标模型验证前向传播（使用预准备的 verify_forward_batch）
        forward_batch_output = self.target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        logits_output = forward_batch_output.logits_output

        # Generate vocab mask for constrained decoding
        vocab_mask = None
        if batch.has_grammar:
            # Generate the logit mask for structured output.
            # 生成结构化输出的 token 位掩码（与 GPU 验证重叠完成）
            vocab_mask = generate_token_bitmask(
                batch.reqs,
                verify_input,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )

            if vocab_mask is not None:
                assert verify_input.grammar is not None
                vocab_mask = vocab_mask.to(verify_input.retrieve_next_token.device)
                # NOTE: otherwise, this vocab mask will be the one from the previous extend stage
                # and will be applied to produce wrong results
                # 清除旧的 vocab_mask，避免 extend 阶段的掩码被误用
                batch.sampling_info.vocab_mask = None

        # Sample
        # 检测 logits NaN 并执行接受/拒绝采样
        maybe_detect_nan(logits_output.next_token_logits, "verify: target model logits")
        (
            predict,        # 所有位置（包括草稿 token）的采样结果
            accept_lens,    # 每个请求接受的 token 数（含 bonus token）
            accept_index,   # 被接受 token 的全局索引
        ) = verify_input.sample(batch, logits_output, vocab_mask)
        # 计算验证后的新序列长度
        new_seq_lens = batch.seq_lens + accept_lens

        # Update mamba state for hybrid GDN models after verification
        # Mamba/GDN 等有状态模型需要在验证后更新内部状态缓存
        if (
            self.target_worker.model_runner.hybrid_gdn_config is not None
            or self.target_worker.model_runner.mamba2_config is not None
        ):
            self._mamba_verify_update(
                batch, verify_input, accept_lens, accept_index, bs
            )

        # 记录验证完成事件（供 extend 阶段等待）
        verify_done = torch.get_device_module(self.device).Event()
        verify_done.record()

        if not batch.forward_mode.is_idle():
            # 提取每个请求最后接受的 token ID（Triton kernel 实现）
            all_verified_id = predict[accept_index]
            verified_id = torch.empty_like(accept_lens, dtype=torch.int32)
            fill_new_verified_id[(bs,)](
                all_verified_id,
                accept_lens,
                verified_id,
                self.speculative_num_draft_tokens,
            )
        else:
            # idle 模式无接受 token
            verified_id = torch.empty((0,), device=self.device, dtype=torch.int32)

        if batch.return_logprob and not batch.forward_mode.is_idle():
            # 计算投机解码 v2 的 logprob（按接受路径整理输出概率）
            compute_spec_v2_logprobs(
                batch, logits_output, predict, accept_index, self.speculative_num_steps
            )

        # Construct the next draft input
        # 构建下一轮草稿输入（携带 verified_id 和新序列长度）
        next_draft_input = EagleDraftInput(
            verified_id=verified_id,
            new_seq_lens=new_seq_lens,
            verify_done=verify_done,  # 传递验证完成事件，供 extend 阶段同步
        )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            accept_lens=accept_lens,
            routed_experts_output=forward_batch_output.routed_experts_output,
        )

    def _mamba_verify_update(
        self,
        batch: ModelWorkerBatch,
        verify_input: EagleVerifyInput,
        accept_lens: torch.Tensor,
        accept_index: torch.Tensor,
        bs: int,
    ):
        """Update mamba state for hybrid GDN models after verification."""
        # `accept_lens` already includes the bonus token (drafts + 1 per req).
        # Mamba 状态更新：根据接受的投机步数选择正确的缓存状态
        accepted_length_with_bonus = accept_lens
        if not batch.forward_mode.is_idle() and accept_index.numel() > 0:
            if verify_input.topk != 1:
                raise ValueError("Spec v2 currently only supports topk = 1.")

            # 计算每个请求的起始偏移（用于将全局接受索引转换为相对步数）
            accepted_indices_offset = torch.arange(
                0,
                bs * self.speculative_num_draft_tokens,
                step=self.speculative_num_draft_tokens,
                dtype=accepted_length_with_bonus.dtype,
                device=accepted_length_with_bonus.device,
            )
            # 接受的步数 = 接受长度 - 1（减去 bonus token）
            accepted_steps = accepted_length_with_bonus - 1

            if batch.mamba_track_indices is not None:
                # If after verify, the request's seq_lens has crossed a mamba track interval,
                # we need to update the mamba state for the request at the crossing point.
                # 检测是否跨越 Mamba track 区间（需要在该点持久化 Mamba 状态）
                seq_lens_pre_verify = batch.seq_lens
                seq_lens_post_verify = batch.seq_lens + accepted_length_with_bonus
                mamba_track_interval = self.server_args.mamba_track_interval
                to_track_mask = (
                    seq_lens_pre_verify // mamba_track_interval
                    != seq_lens_post_verify // mamba_track_interval
                )
                tracking_point = (
                    seq_lens_post_verify // mamba_track_interval * mamba_track_interval
                )
                # 计算需要 track 的步骤偏移
                to_track_ith = torch.clamp(
                    tracking_point - seq_lens_pre_verify - 1, min=0
                ).to(torch.int64)
                req_idx = torch.arange(
                    bs,
                    dtype=torch.int64,
                    device=accepted_length_with_bonus.device,
                )
                candidate_track_steps = (
                    accept_index[req_idx, to_track_ith] - accepted_indices_offset
                )
                mamba_steps_to_track = torch.where(
                    to_track_mask,
                    candidate_track_steps,
                    torch.full_like(candidate_track_steps, -1),  # -1 表示无需 track
                )
            else:
                mamba_steps_to_track = None

            # 用正确的接受步骤更新 Mamba 模型状态缓存
            self.target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify(
                accepted_steps=accepted_steps,
                mamba_track_indices=batch.mamba_track_indices,
                mamba_steps_to_track=mamba_steps_to_track,
                model=self.target_worker.model_runner.model,
            )

    def move_accepted_tokens_to_target_kvcache(
        self,
        batch: ModelWorkerBatch,
        accept_index: torch.Tensor,
        num_accepted_drafts: torch.Tensor,
    ):
        """
        Move accepted tokens to the target KV cache.

        Args:
            batch: The batch to run.
            accept_index: The index of the accepted tokens.
            num_accepted_drafts: The length of the accepted tokens.
        """
        # 将验证接受的 token KV cache 数据从草稿位置移到目标模型的正式位置
        bs = len(batch.seq_lens)
        size = bs * self.speculative_num_draft_tokens

        tgt_cache_loc = torch.zeros(
            size,
            dtype=torch.int64,
            device=self.device,
        )
        accepted_out_cache_loc = torch.zeros(
            size, dtype=torch.int64, device=self.device
        )
        # 使用 Triton kernel 分配 extend 阶段的 cache 位置
        assign_extend_cache_locs[(bs,)](
            batch.req_pool_indices,
            self.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + num_accepted_drafts,
            tgt_cache_loc,
            self.req_to_token_pool.req_to_token.shape[1],
            next_power_of_2(bs),
        )
        # 使用 Triton kernel 将接受的草稿 token 的 cache 位置填入 accepted_out_cache_loc
        fill_accepted_out_cache_loc[(size,)](
            accept_index,
            batch.out_cache_loc,
            accepted_out_cache_loc,
            next_power_of_2(size),
        )
        # 将草稿 KV cache 中接受路径的数据移到目标模型 KV cache 的正式位置
        self.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
            tgt_cache_loc, accepted_out_cache_loc
        )

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        # 从磁盘更新草稿模型权重（只更新草稿模型，目标模型单独管理）
        success, message = self._draft_worker.draft_runner.update_weights_from_disk(
            recv_req.model_path,
            recv_req.load_format,
            recapture_cuda_graph=recv_req.recapture_cuda_graph,
        )
        if not success:
            return success, message
        return True, "Succeeded to update model weights."

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        # 通过 IPC（进程间通信）更新草稿模型权重
        success, message = self._draft_worker.draft_runner.update_weights_from_ipc(
            recv_req
        )
        if not success:
            return success, message
        return True, "Succeeded to update model weights."

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        # 从张量更新草稿和目标模型权重（在线权重更新接口）
        monkey_patch_torch_reductions()
        named_tensors = MultiprocessingSerializer.deserialize(
            recv_req.serialized_named_tensors[self.tp_rank]
        )
        # 先更新草稿模型权重
        success, message = self.draft_worker.draft_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        if not success:
            return success, message

        # 再更新目标模型权重（两者保持同步）
        success, message = self.target_worker.model_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        return success, message

