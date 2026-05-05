# DFlash Worker 模块：DFlash 投机解码 worker
# DFlash 是一种基于"草稿 token 块"的投机解码方法，草稿模型一次生成整块（block_size）候选 token
# 与 EAGLE 的树形展开不同，DFlash 采用线性草稿，草稿模型直接输出 block_size 个 token
import logging
import math
from copy import deepcopy
from typing import Optional, Union

import torch

# 导入分布式 TP 组工具
from sglang.srt.distributed import get_tp_group
# 导入批次数据结构
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
# 导入张量并行 model worker 基类
from sglang.srt.managers.tp_worker import TpModelWorker
# 导入 KV cache 位置获取工具
from sglang.srt.mem_cache.common import get_last_loc
# 导入前向传播批次信息
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
# 导入服务器参数和全局参数管理工具
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
# 导入 DFlash 草稿输入和验证输入数据结构
from sglang.srt.speculative.dflash_info import DFlashDraftInput, DFlashVerifyInput
# 导入 DFlash 相关工具函数
from sglang.srt.speculative.dflash_utils import (
    can_dflash_use_fused_qkv_proj,             # 检查是否可以使用融合 QKV 投影优化
    is_dflash_sampling_verify_available,         # 检查采样验证是否可用
    parse_dflash_draft_config,                   # 解析 DFlash 草稿模型配置
    resolve_dflash_verify_mask_policy,           # 解析验证阶段的掩码策略
)
# 导入投机解码算法枚举
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
# 导入请求-token 池分配函数
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func
# 导入 CUDA 检测工具
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)

# 全局延迟加载的融合 KV 物化助手（只在 CUDA 且条件满足时初始化）
_FusedKVMaterializeHelper = None


def _get_fused_kv_materialize_helper():
    # 延迟加载 FusedKVMaterializeHelper（避免在非 CUDA 环境中导入 sgl_kernel）
    global _FusedKVMaterializeHelper
    if _FusedKVMaterializeHelper is None:
        from sglang.srt.speculative.triton_ops.fused_kv_materialize import (
            FusedKVMaterializeHelper,
        )

        _FusedKVMaterializeHelper = FusedKVMaterializeHelper
    return _FusedKVMaterializeHelper


class DFlashWorker:
    """DFlash speculative decoding worker (spec-v1, tp>=1/pp=1)."""
    # DFlash 投机解码 worker，支持张量并行（tp>=1），不支持流水线并行（pp=1）

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
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank
        self.nccl_port = nccl_port
        # 目标（验证）模型 worker
        self.target_worker = target_worker
        # 与目标 worker 共享 model_runner（DFlash 的目标模型就是同一个 runner）
        self.model_runner = target_worker.model_runner
        self.page_size = server_args.page_size
        # 草稿滑动窗口大小（None 表示不使用窗口限制，使用完整上下文）
        self.draft_window_size: Optional[int] = (
            int(server_args.speculative_dflash_draft_window_size)
            if server_args.speculative_dflash_draft_window_size is not None
            else None
        )
        # 是否使用紧凑草稿 cache（启用窗口时使用，只保留窗口内的 KV）
        self.use_compact_draft_cache = self.draft_window_size is not None
        self.device = target_worker.device

        # 标志位：是否已打印过采样降级警告（避免刷屏）
        self._warned_sampling_fallback = False
        # 标志位：是否已打印过首次验证日志
        self._logged_first_verify = False

        # Draft runner (separate KV cache + attention backend).
        # Without draft windowing, the draft worker aliases the target request->token
        # mapping and allocation state. With draft windowing enabled, the draft worker
        # keeps a private compact req->token table over the same global KV index space,
        # so radix-cache/prefix-hit KV remains reusable while draft attention sees only
        # the recent window.
        # 草稿 runner 说明：
        # - 无窗口：共享目标模型的 req->token 映射表（完整上下文注意力）
        # - 有窗口：使用独立的紧凑映射表（只保留最近 draft_window_size 个 token 的 KV）
        target_req_to_token_pool, target_token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        # 无窗口时共享 req_to_token_pool，有窗口时草稿 worker 维护独立的映射
        shared_req_to_token_pool = (
            None if self.use_compact_draft_cache else target_req_to_token_pool
        )
        # 深拷贝 server_args，独立配置草稿 worker（避免污染目标模型配置）
        draft_server_args = deepcopy(server_args)
        draft_server_args.skip_tokenizer_init = True
        # 解析草稿注意力后端（DFlash 支持 flashinfer/fa3/fa4/triton）
        draft_backend = draft_server_args.speculative_draft_attention_backend
        supported_draft_backends = ("flashinfer", "fa3", "fa4", "triton")
        if draft_backend is None:
            draft_backend, _ = draft_server_args.get_attention_backends()
        if draft_backend is None:
            # Use triton on ROCm (no FlashInfer), flashinfer on CUDA
            # ROCm（HIP）无 FlashInfer，使用 Triton；CUDA 默认使用 FlashInfer
            import torch as _torch

            draft_backend = "triton" if _torch.version.hip else "flashinfer"
        elif draft_backend == "trtllm_mha":
            import torch as _torch

            _fb = "triton" if _torch.version.hip else "flashinfer"
            logger.warning(
                "DFLASH draft worker does not support 'trtllm_mha' because the "
                "draft path requires non-causal attention. Falling back to "
                "'%s'.",
                _fb,
            )
            # DFlash 草稿模型需要非因果注意力（bidirectional），不支持 trtllm_mha
            draft_backend = _fb
        elif draft_backend not in supported_draft_backends:
            import torch as _torch

            _fb = "triton" if _torch.version.hip else "flashinfer"
            logger.warning(
                "DFLASH draft worker only supports attention_backend in %s for now, "
                "but got %r. Falling back to '%s'.",
                supported_draft_backends,
                draft_backend,
                _fb,
            )
            draft_backend = _fb
        # Make the draft worker backend explicit and self-contained (no further overrides).
        # 显式设置草稿 worker 使用的注意力后端，清除其他注意力相关配置
        draft_server_args.speculative_draft_attention_backend = None
        draft_server_args.prefill_attention_backend = None
        draft_server_args.decode_attention_backend = None
        draft_server_args.attention_backend = draft_backend
        # Keep draft context length aligned with the target.
        # 草稿模型上下文长度与目标模型对齐
        draft_server_args.context_length = (
            target_worker.model_runner.model_config.context_len
        )
        # 保存当前全局 server_args（DFlash 草稿 worker 初始化会修改全局状态，需要恢复）
        saved_server_args = get_global_server_args()
        # 初始化草稿 TpModelWorker（加载草稿模型权重）
        self.draft_worker = TpModelWorker(
            server_args=draft_server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            moe_ep_rank=moe_ep_rank,
            pp_rank=0,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            dp_rank=dp_rank,
            nccl_port=nccl_port,
            is_draft_worker=True,
            req_to_token_pool=shared_req_to_token_pool,
            token_to_kv_pool_allocator=target_token_to_kv_pool_allocator,
            memory_pool_config=target_worker.model_runner.memory_pool_config,
        )
        # 恢复全局 server_args（避免草稿 worker 初始化副作用）
        set_global_server_args_for_scheduler(saved_server_args)
        self.draft_model_runner = self.draft_worker.model_runner
        self.draft_model = self.draft_model_runner.model
        # 解析草稿模型的 DFlash 专属配置（block_size、mask_token 等）
        draft_config = parse_dflash_draft_config(
            draft_hf_config=self.draft_model_runner.model_config.hf_config
        )
        if server_args.speculative_num_draft_tokens is None:
            # Should not happen (ServerArgs should have inferred it), but keep a fallback.
            # 从模型配置中推断 block_size（每次投机生成的 token 数量）
            self.block_size = int(draft_config.resolve_block_size(default=16))
        else:
            # 使用用户指定的 speculative_num_draft_tokens 作为 block_size
            self.block_size = int(server_args.speculative_num_draft_tokens)
            model_block_size = draft_config.block_size
            if model_block_size is None:
                model_block_size = getattr(self.draft_model, "block_size", None)
            if model_block_size is not None and int(model_block_size) != int(
                self.block_size
            ):
                # 用户指定值与模型配置不一致时警告（可能导致精度问题）
                logger.warning(
                    "DFLASH block size mismatch: using speculative_num_draft_tokens=%s but draft config block_size=%s.",
                    self.block_size,
                    model_block_size,
                )

        # mask_token：DFlash 用于占位的掩码 token（填充未生成的 draft slot）
        self._mask_token = draft_config.mask_token
        self._mask_token_id_override = draft_config.mask_token_id
        # 解析掩码 token 的实际 ID（从词表中查找或使用覆盖值）
        self._mask_token_id = self._resolve_mask_token_id(
            mask_token=self._mask_token,
            mask_token_id=self._mask_token_id_override,
        )
        if self.tp_rank == 0:
            logger.info(
                "Initialized DFLASH draft runner. attention_backend=%s, model=%s, block_size=%s, draft_window_size=%s, compact_cache=%s",
                getattr(draft_server_args, "attention_backend", None),
                self.draft_model.__class__.__name__,
                self.block_size,
                self.draft_window_size,
                self.use_compact_draft_cache,
            )
            logger.info(
                "DFLASH draft runner ready. mask_token=%s, mask_token_id=%s, mask_token_id_override=%s",
                self._mask_token,
                self._mask_token_id,
                self._mask_token_id_override,
            )

        # 位置偏移张量：[0, 1, ..., block_size-1]，用于计算草稿 token 的位置编码
        self._block_pos_offsets = torch.arange(
            self.block_size, device=self.device, dtype=torch.int64
        )
        # 草稿 block buffer（批量草稿前向传播的复用缓冲区，懒初始化）
        self._draft_block_ids_buf: Optional[torch.Tensor] = None  # [cap_bs, block_size]
        self._draft_block_positions_buf: Optional[torch.Tensor] = (
            None  # [cap_bs, block_size]
        )
        self._draft_block_tokens_buf: Optional[torch.Tensor] = (
            None  # [cap_bs, block_size]
        )
        self._draft_block_end_buf: Optional[torch.Tensor] = None  # [cap_bs]
        self._draft_seq_lens_cpu_buf: Optional[torch.Tensor] = None  # [cap_bs] on CPU
        # 草稿块的 spec_info 对象（用于传递给目标模型验证）
        self._draft_block_spec_info = DFlashVerifyInput(
            draft_token=torch.empty((0,), dtype=torch.long, device=self.device),
            positions=torch.empty((0,), dtype=torch.int64, device=self.device),
            draft_token_num=int(self.block_size),
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        # 贪婪采样用的缓冲区（用于快速选择最优 draft token）
        self._draft_greedy_gathered_max_buf: Optional[torch.Tensor] = None
        self._draft_greedy_gathered_ids_buf: Optional[torch.Tensor] = None
        self._draft_greedy_gather_cap: int = 0
        self._draft_greedy_best_rank_buf: Optional[torch.Tensor] = None
        self._draft_greedy_rank_index_buf: Optional[torch.Tensor] = None
        self._draft_greedy_selected_ids_buf: Optional[torch.Tensor] = None
        self._draft_greedy_index_cap: int = 0

        # 是否使用融合 KV 物化（将 QKV 投影和 KV cache 写入合并为单次操作，仅 CUDA）
        self._use_fused_kv_materialize = is_cuda()
        self._fused_kv_helper: Optional[object] = None
        if self._use_fused_kv_materialize:
            self._init_fused_kv_helper()

    def _init_fused_kv_helper(self) -> None:
        """Initialize the fused KV materialization helper with pre-stacked weights."""
        # 初始化融合 KV 物化助手：将所有层的 QKV 权重预堆叠，实现批量 KV 投影
        try:
            layers = self.draft_model.layers
            fused_disable_reason: Optional[str] = None

            if len(layers) == 0:
                fused_disable_reason = "no layers found"

            for layer_idx, layer in enumerate(layers):
                attn = layer.self_attn
                # 检查 QKV 投影是否满足融合条件（权重形状、量化方式等）
                eligible, reason = can_dflash_use_fused_qkv_proj(attn.qkv_proj)
                if not eligible:
                    fused_disable_reason = f"{reason}: layer={layer_idx}"
                    break

                # Keep semantics aligned with set_kv_buffer scaling behavior.
                # 检查 KV 量化缩放因子（非单位缩放不支持融合路径）
                k_scale = getattr(attn.attn, "k_scale", None)
                v_scale = getattr(attn.attn, "v_scale", None)
                if k_scale is not None and not math.isclose(float(k_scale), 1.0):
                    fused_disable_reason = (
                        "non-unit k_scale is not supported for fused KV path: "
                        f"layer={layer_idx}, k_scale={k_scale}"
                    )
                    break
                if v_scale is not None and not math.isclose(float(v_scale), 1.0):
                    fused_disable_reason = (
                        "non-unit v_scale is not supported for fused KV path: "
                        f"layer={layer_idx}, v_scale={v_scale}"
                    )
                    break

                # 检查 RoPE 样式（非 NeoX 风格的 RoPE 不支持融合路径）
                rope_is_neox_style = bool(
                    getattr(attn.rotary_emb, "is_neox_style", True)
                )
                if not rope_is_neox_style:
                    fused_disable_reason = (
                        "non-neox RoPE is not supported for fused KV path: "
                        f"layer={layer_idx}, rope_is_neox_style={rope_is_neox_style}"
                    )
                    break

            if fused_disable_reason is not None:
                if self.tp_rank == 0:
                    logger.info(
                        "DFLASH fused KV materialization disabled: %s",
                        fused_disable_reason,
                    )
                self._use_fused_kv_materialize = False
                self._fused_kv_helper = None
                return

            # 所有检查通过，初始化融合 KV 助手
            FusedKVMaterializeHelper = _get_fused_kv_materialize_helper()
            first_attn = layers[0].self_attn
            rotary_emb = first_attn.rotary_emb

            # 预堆叠所有层的 KV 权重，供批量投影使用
            self._fused_kv_helper = FusedKVMaterializeHelper(
                layers=layers,
                rotary_emb=rotary_emb,
                num_kv_heads=first_attn.num_kv_heads,
                head_dim=first_attn.head_dim,
                device=self.device,
            )
            if self.tp_rank == 0:
                logger.info(
                    "DFLASH fused KV materialization enabled. "
                    "n_layers=%d, num_kv_heads=%d, head_dim=%d",
                    len(layers),
                    first_attn.num_kv_heads,
                    first_attn.head_dim,
                )
        except Exception as e:
            # 初始化失败时降级到逐层串行路径
            logger.warning(
                "DFLASH fused KV initialization failed, falling back to sequential path: %s",
                e,
            )
            self._use_fused_kv_materialize = False
            self._fused_kv_helper = None

    def _ensure_draft_block_buffers(self, bs: int) -> None:
        # 确保草稿 block buffer 容量足够（按需扩容，避免每批次重新分配）
        cap = (
            0
            if self._draft_block_ids_buf is None
            else int(self._draft_block_ids_buf.shape[0])
        )
        if cap >= int(bs):
            return

        # 倍增扩容策略（减少重分配次数）
        new_cap = max(int(bs), cap * 2 if cap > 0 else int(bs))
        device = self.device
        block_size = int(self.block_size)
        # 草稿 token 的 KV cache 位置 ID [batch, block_size]
        self._draft_block_ids_buf = torch.empty(
            (new_cap, block_size), dtype=torch.long, device=device
        )
        # 草稿 token 的位置索引（用于 RoPE）[batch, block_size]
        self._draft_block_positions_buf = torch.empty(
            (new_cap, block_size), dtype=torch.int64, device=device
        )
        # 草稿 token 的 ID（初始化为 mask_token）[batch, block_size]
        self._draft_block_tokens_buf = torch.empty(
            (new_cap, block_size), dtype=torch.long, device=device
        )
        # 每个序列草稿 block 的有效长度（用于变长 mask 注意力）[batch]
        self._draft_block_end_buf = torch.empty(
            (new_cap,), dtype=torch.int32, device=device
        )
        # CPU 端的序列长度缓冲区（避免频繁 GPU-CPU 同步）[batch]
        self._draft_seq_lens_cpu_buf = torch.empty(
            (new_cap,), dtype=torch.int32, device="cpu"
        )

    def __getattr__(self, name):
        # Delegate anything not implemented yet to the target worker.
        # 委托未实现的属性/方法到目标 worker（DFlash worker 复用目标 worker 的大量接口）
        return getattr(self.target_worker, name)

    def clear_cache_pool(self):
        # The target worker owns the shared KV allocator/cache. For the compact
        # sliding-window path, the draft req->token view is rebuilt from committed
        # target state before each draft forward, so there is nothing persistent
        # to flush here.
        # KV 分配器由目标 worker 拥有；窗口模式下，草稿 req->token 视图在每次前向传播前重建
        # 因此此处无需持久状态清理
        pass

    def _gather_req_to_token_masked(
        self,
        *,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        pos2d: torch.Tensor,
        mask: torch.Tensor,
        context: str,
    ) -> torch.Tensor:
        # 从 req_to_token 表中按 mask 安全地 gather KV cache 位置（支持不规则批次）
        if pos2d.ndim != 2:
            raise RuntimeError(
                f"{context} expected 2D positions, got shape={tuple(pos2d.shape)}."
            )
        if mask.shape != pos2d.shape:
            raise RuntimeError(
                f"{context} mask/position shape mismatch: {tuple(mask.shape)} vs {tuple(pos2d.shape)}."
            )

        # 统一类型保证 gather 索引正确
        if req_pool_indices.dtype != torch.int64:
            req_pool_indices = req_pool_indices.to(torch.int64)
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)

        table_width = int(req_to_token.shape[1])
        if table_width <= 0:
            if bool(mask.any().item()):
                raise RuntimeError(
                    f"{context} req_to_token table is empty but gather mask is non-empty."
                )
            return torch.empty((0,), dtype=torch.int64, device=self.device)

        # Only the masked-off rectangular padding can be out of range in the normal
        # ragged-batch case. Replace those don't-care columns with a valid in-range
        # position before the gather so the kernel only sees real positions.
        # 将 padding 列替换为 0（有效位置），防止越界；gather 后再用 mask 筛选有效结果
        safe_pos2d = pos2d.masked_fill(~mask, 0)
        return req_to_token[req_pool_indices[:, None], safe_pos2d][mask].to(torch.int64)

    def _gather_req_to_token_segments(
        self,
        *,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        start: torch.Tensor | None,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        # 从 req_to_token 中按 [start, start+length) 范围 gather KV cache 位置（连续段）
        lengths = lengths.to(torch.int64)
        if lengths.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        max_len = int(lengths.max().item())
        if max_len <= 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)

        if req_pool_indices.dtype != torch.int64:
            req_pool_indices = req_pool_indices.to(torch.int64)
        # 构建矩形偏移矩阵 [1, max_len]，再广播为 [bs, max_len]
        offsets = torch.arange(
            max_len, device=self.device, dtype=torch.int64
        ).unsqueeze(0)
        if start is None:
            # 无起始偏移时，直接从序列头开始
            pos2d = offsets.expand(req_pool_indices.shape[0], -1)
        else:
            # 有起始偏移时，加上 start（绝对 token 位置）
            pos2d = start.to(torch.int64).unsqueeze(1) + offsets
        # 布尔 mask：只保留 offset < lengths 的有效位置（处理变长序列）
        mask = offsets < lengths.unsqueeze(1)
        return self._gather_req_to_token_masked(
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            pos2d=pos2d,
            mask=mask,
            context="DFLASH req_to_token segment gather",
        )

    def _compute_compact_draft_seq_lens(self, seq_lens: torch.Tensor) -> torch.Tensor:
        # 计算 compact（滑动窗口）模式下草稿序列的有效长度
        # compact 模式仅保留最近 draft_window_size 个 token 的 KV cache
        assert self.draft_window_size is not None
        visible_lens = torch.clamp(
            seq_lens.to(dtype=torch.int32, device=self.device),
            max=int(self.draft_window_size),
        )
        if self.page_size <= 1:
            # 非分页模式：直接返回裁剪后的可见长度
            return visible_lens

        # Paged FA backends derive the page table from local token positions, so the
        # compact suffix must start on a page boundary. Keep up to page_size - 1 extra
        # tokens on the left to preserve valid local page structure.
        # 分页注意力要求 compact 窗口从页边界开始，避免无效的局部 page 结构
        seq_lens_i64 = seq_lens.to(torch.int64)
        visible_lens_i64 = visible_lens.to(torch.int64)
        visible_start = seq_lens_i64 - visible_lens_i64
        # 向下对齐到页边界（最多额外保留 page_size-1 个 token）
        aligned_start = visible_start - torch.remainder(visible_start, self.page_size)
        return (seq_lens_i64 - aligned_start).to(torch.int32)

    def _resolve_mask_token_id(
        self, *, mask_token: str, mask_token_id: Optional[int] = None
    ) -> int:
        # 解析 DFlash mask token 的词表 ID（用于填充草稿 block 中的未知位置）
        # mask token 是一个特殊 token，在草稿 block 中占位，模型通过双向注意力预测其真实 token
        if not isinstance(mask_token, str) or not mask_token:
            raise ValueError(
                f"DFLASH mask_token must be a non-empty string, got {mask_token!r}."
            )

        vocab_size = int(self.target_worker.model_runner.model_config.vocab_size)
        if mask_token_id is not None:
            # 配置中已显式指定 mask_token_id：做边界校验和 tokenizer 一致性检验
            resolved_id = int(mask_token_id)
            if resolved_id >= vocab_size:
                raise ValueError(
                    "DFLASH mask_token_id is outside the target vocab size. "
                    f"mask_token_id={resolved_id}, vocab_size={vocab_size}. "
                    f"This likely means mask_token={mask_token!r} requires vocab expansion beyond the model's embedding size. "
                    "SGLang does not support resizing target embeddings for DFLASH yet."
                )

            tokenizer = getattr(self.target_worker, "tokenizer", None)
            if tokenizer is not None:
                # 校验 tokenizer 词表中相同 token 的 ID 与配置一致
                token_id_from_vocab = tokenizer.get_vocab().get(mask_token, None)
                if (
                    token_id_from_vocab is not None
                    and int(token_id_from_vocab) != resolved_id
                ):
                    raise ValueError(
                        "DFLASH config mismatch: dflash_config.mask_token_id conflicts with tokenizer vocab id "
                        f"for dflash_config.mask_token. mask_token={mask_token!r}, "
                        f"mask_token_id={resolved_id}, tokenizer_vocab_id={int(token_id_from_vocab)}."
                    )
            return resolved_id

        # 未指定 mask_token_id 时，尝试从 tokenizer 自动解析
        tokenizer = getattr(self.target_worker, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                "DFLASH requires tokenizer initialization when dflash_config.mask_token_id is not set "
                "(skip_tokenizer_init is not supported in this mode)."
            )

        resolved_id = None
        # 优先从 tokenizer.mask_token 属性获取（标准 HF tokenizer 接口）
        if getattr(tokenizer, "mask_token", None) == mask_token:
            resolved_id = getattr(tokenizer, "mask_token_id", None)

        if resolved_id is None:
            # Prefer checking the explicit vocab mapping first.
            # 再从词表字典中查找（兼容自定义 tokenizer）
            vocab = tokenizer.get_vocab()
            resolved_id = vocab.get(mask_token, None)

        if resolved_id is None:
            # Mirror the reference DFlash HF demo by adding the mask token to the tokenizer.
            # This is safe only when the resulting id stays within the target model vocab size.
            # 仍未找到时，动态添加特殊 token 到 tokenizer（与参考实现对齐）
            added = tokenizer.add_special_tokens({"mask_token": mask_token})
            resolved_id = getattr(tokenizer, "mask_token_id", None)
            if resolved_id is None:
                resolved_id = tokenizer.convert_tokens_to_ids(mask_token)

            if added and self.tp_rank == 0:
                logger.info(
                    "Added DFLASH mask token to tokenizer. token=%s, mask_token_id=%s, tokenizer_len=%s, model_vocab_size=%s",
                    mask_token,
                    resolved_id,
                    len(tokenizer),
                    vocab_size,
                )

        if resolved_id is None or int(resolved_id) < 0:
            raise ValueError(
                "DFLASH requires resolving a mask token id, but it could not be resolved. "
                f"mask_token={mask_token!r}."
            )

        # 最终校验：mask token ID 不能超出目标模型词表范围（不支持 embedding 扩容）
        if resolved_id >= vocab_size:
            raise ValueError(
                "DFLASH mask_token_id is outside the target vocab size. "
                f"mask_token_id={resolved_id}, vocab_size={vocab_size}. "
                f"This likely means mask_token={mask_token!r} requires vocab expansion beyond the model's embedding size. "
                "SGLang does not support resizing target embeddings for DFLASH yet."
            )

        return int(resolved_id)

    def _prepare_for_speculative_decoding(
        self, batch: ScheduleBatch, draft_input: DFlashDraftInput
    ):
        # 投机解码准备阶段：填充草稿 block 并组装 DFlashVerifyInput 供目标模型验证
        # 仅在 DECODE 模式下执行；EXTEND（prefill）和 IDLE 跳过
        if batch.forward_mode.is_extend() or batch.forward_mode.is_idle():
            return

        # DFlash 不支持语法约束（调度器应在接受请求时已拒绝）
        if batch.has_grammar:
            raise RuntimeError(
                "Invariant broken: DFLASH batch has grammar constraints, but scheduler should have rejected this request."
            )
        if batch.sampling_info is not None and not batch.sampling_info.is_all_greedy:
            # 非贪心采样验证需要特殊实现，未构建时降级到 greedy argmax
            if (
                not is_dflash_sampling_verify_available()
                and not self._warned_sampling_fallback
                and self.tp_rank == 0
            ):
                logger.warning(
                    "DFLASH non-greedy verification is unavailable on this build/device; "
                    "falling back to greedy argmax verification."
                )
                self._warned_sampling_fallback = True

        bs = batch.batch_size()

        # --- 1) 将上一步目标模型产出的 hidden states 写入草稿 KV cache ---
        # 必须在 radix cache 暴露新 token 前完成（防止前缀命中引用空缺的草稿 KV）
        self._append_target_hidden_to_draft_kv(batch, draft_input)

        # 获取目标模型的 embedding 层和 lm_head（DFlash 草稿模型复用目标模型的 lm_head）
        target_model = self.target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()
        lm_head = getattr(target_model, "lm_head", None)
        if (
            lm_head is None
            or not hasattr(lm_head, "weight")
            or not hasattr(lm_head, "shard_indices")
        ):
            raise RuntimeError(
                "DFLASH requires the target model to expose a vocab-parallel `lm_head` with `weight` and "
                "`shard_indices` attributes."
            )

        # --- 2) 用草稿模型生成非因果 block（固定大小 block_size 个 token）---
        self._ensure_draft_block_buffers(bs)
        assert self._draft_block_ids_buf is not None
        assert self._draft_block_positions_buf is not None
        assert self._draft_block_tokens_buf is not None
        assert self._draft_block_end_buf is not None
        assert self._draft_seq_lens_cpu_buf is not None

        # block_ids[:, 0] = 上一步验证接受的 token；其余位置填 mask_token（待预测）
        block_ids = self._draft_block_ids_buf[:bs]
        block_ids.fill_(int(self._mask_token_id))
        block_ids[:, 0].copy_(draft_input.verified_id.to(torch.long))

        # 将 token ID 转为 embedding 向量作为草稿模型输入
        noise_embedding = embed_module(block_ids)
        input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])

        # For spec-v1, the draft KV cache is always materialized before drafting the
        # next block. `target_prefix_lens` stay absolute for RoPE; `draft_prefix_lens`
        # are the logical resident lengths in the draft-local cache.
        # target_prefix_lens：目标序列绝对长度（用于 RoPE 位置编码）
        # draft_prefix_lens：草稿 KV cache 中的逻辑前缀长度
        target_prefix_lens = batch.seq_lens  # int32, device
        draft_prefix_lens = draft_input.draft_seq_lens
        if draft_prefix_lens.dtype != torch.int32:
            draft_prefix_lens = draft_prefix_lens.to(torch.int32)
        if draft_prefix_lens.device != self.device:
            draft_prefix_lens = draft_prefix_lens.to(self.device, non_blocking=True)

        # 构建草稿 block 的 RoPE 位置：从 target_prefix_lens 开始，连续 block_size 个位置
        positions_2d = self._draft_block_positions_buf[:bs]
        torch.add(
            target_prefix_lens.unsqueeze(1), self._block_pos_offsets, out=positions_2d
        )
        positions = positions_2d.reshape(-1)

        # 计算草稿 block 在 draft KV cache 中的写入范围 [block_start, block_end)
        block_start = draft_prefix_lens
        block_end = self._draft_block_end_buf[:bs]
        torch.add(block_start, int(self.block_size), out=block_end)

        # CPU 端序列长度（避免频繁 GPU-CPU 同步，仅在分页路径需要）
        seq_lens_cpu = self._draft_seq_lens_cpu_buf[:bs]
        seq_lens_cpu.copy_(draft_prefix_lens.to(device="cpu", dtype=torch.int32))
        # 备份分配器状态，确保草稿 block 不污染目标 KV 分配（用后恢复）
        allocator = self.draft_model_runner.token_to_kv_pool_allocator
        token_to_kv_pool_state_backup = allocator.backup_state()
        try:
            if self.page_size == 1:
                # 非分页：直接分配连续 KV 槽位
                block_cache_loc = allocator.alloc(bs * self.block_size)
            else:
                # 分页：需要知道上一页的最后位置，以便在同一 page 内续写
                block_end_cpu = seq_lens_cpu + int(self.block_size)
                last_loc = get_last_loc(
                    self.draft_model_runner.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    block_start,
                )
                block_cache_loc = allocator.alloc_extend(
                    block_start,
                    seq_lens_cpu,
                    block_end,
                    block_end_cpu,
                    last_loc,
                    bs * self.block_size,
                )
            if block_cache_loc is None:
                raise RuntimeError(
                    f"DFLASH draft OOM when allocating {bs * self.block_size} block tokens."
                )

            # 将草稿 block 的 KV 位置写入 req_to_token 表（供注意力 kernel 使用）
            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                block_start,
                block_end,
                block_cache_loc,
                bs,
            )

            # Use TARGET_VERIFY mode (cuda-graphable) to run a fixed-size draft block.
            # In this mode, `seq_lens` stores the prefix lengths; attention backends
            # 用 TARGET_VERIFY 模式运行草稿前向（固定大小，可捕获 CUDA graph）
            # seq_lens 存储前缀长度；注意力 backend 通过 draft_token_num 推导 kv_len
            # derive kv_len by adding `draft_token_num`.
            # _draft_block_spec_info 是一个固定的 DFlashDraftSpecInfo 对象，记录 block_size
            draft_spec_info = self._draft_block_spec_info
            seq_lens = draft_prefix_lens
            seq_lens_sum = int(draft_prefix_lens.sum().item())
            # 构造草稿前向的 ForwardBatch（TARGET_VERIFY 模式，复用草稿模型自己的 KV 资源）
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.TARGET_VERIFY,
                batch_size=bs,
                input_ids=block_ids.flatten(),
                req_pool_indices=batch.req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=block_cache_loc,
                seq_lens_sum=seq_lens_sum,
                seq_lens_cpu=seq_lens_cpu,
                positions=positions,
                req_to_token_pool=self.draft_model_runner.req_to_token_pool,
                token_to_kv_pool=self.draft_model_runner.token_to_kv_pool,
                attn_backend=self.draft_model_runner.attn_backend,
                input_embeds=input_embeds,         # 输入为 embedding 向量（含 mask token）
                spec_algorithm=SpeculativeAlgorithm.DFLASH,
                spec_info=draft_spec_info,
                capture_hidden_mode=CaptureHiddenMode.NULL,  # 草稿模型不需要捕获 hidden
            )

            with torch.inference_mode():
                # 运行草稿模型前向，获取每个位置的 hidden states（不需要 logits）
                draft_logits_output = self.draft_model_runner.forward(
                    forward_batch
                ).logits_output
        finally:
            # Drop the speculative block from the shared allocator (EAGLE3-style).
            # 草稿前向完成后立即归还草稿 KV 分配（只需 hidden states，不需要保留 KV）
            allocator.restore_state(token_to_kv_pool_state_backup)

        # 草稿模型输出 hidden states：[bs * block_size, hidden_dim]
        draft_hidden = draft_logits_output.hidden_states
        if draft_hidden is None:
            raise RuntimeError("DFLASH draft model returned no hidden states.")
        draft_hidden = draft_hidden.view(bs, self.block_size, -1)
        # 对 [1:] 位置的 hidden states 做 greedy argmax（位置 0 是已知 verified_id）
        # 结果：[bs, block_size-1] 的草稿 token 预测
        draft_next = self._greedy_sample_from_vocab_parallel_head(
            hidden_states=draft_hidden[:, 1:, :].reshape(-1, draft_hidden.shape[-1]),
            lm_head=lm_head,
        ).view(bs, self.block_size - 1)
        # 组装完整草稿 block：[verified_id, pred_1, pred_2, ..., pred_{block_size-1}]
        draft_tokens = self._draft_block_tokens_buf[:bs]
        draft_tokens[:, 0].copy_(block_ids[:, 0])
        draft_tokens[:, 1:].copy_(draft_next)
        positions = positions_2d.reshape(-1)

        # 构建 DFlashVerifyInput，供目标模型做 block 级并行验证
        verify_input = DFlashVerifyInput(
            draft_token=draft_tokens.reshape(-1),
            positions=positions,
            draft_token_num=self.block_size,
        )
        # 解析注意力 backend 对应的验证 mask 构建策略
        _, build_custom_mask = resolve_dflash_verify_mask_policy(
            self.model_runner.attn_backend
        )
        # 填充验证所需的 KV 位置、注意力 mask 等元数据
        verify_input.prepare_for_verify(
            batch,
            self.page_size,
            build_custom_mask=build_custom_mask,
        )

        # 将 batch forward_mode 切换为 TARGET_VERIFY（非 IDLE 时）
        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = verify_input
        batch.return_hidden_states = False

    def _greedy_sample_from_vocab_parallel_head(
        self,
        *,
        hidden_states: torch.Tensor,
        lm_head,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        """Greedy argmax over the target LM head in a TP-safe way.

        We cannot materialize full logits for large vocabularies efficiently, and with
        TP>1 each rank only owns a shard of the LM head weight. This computes the
        per-rank max, gathers candidates across TP ranks, and selects the global max.
        """
        # TP-safe 贪心采样：各 rank 持有 lm_head 权重的不同分片（vocab 切片）
        # 不能直接 argmax 全量 logits（内存爆炸），改为按 rank 局部 argmax 再 all_gather 选全局最大

        if hidden_states.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=hidden_states.device)

        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)

        if not hasattr(lm_head, "weight") or not hasattr(lm_head, "shard_indices"):
            raise RuntimeError(
                "DFLASH greedy sampling requires a vocab-parallel head with `weight` and `shard_indices`."
            )

        # shard_indices 记录本 rank 负责的词表范围（base vocab + added vocab 各自的起始/长度）
        shard = lm_head.shard_indices
        weight = lm_head.weight  # [local_vocab_padded, hidden]
        weight_dtype = weight.dtype

        # Valid ranges in the local shard (excluding padding):
        #   base vocab:  [0, num_org)
        #   added vocab: [num_org_padded, num_org_padded + num_added)
        # 本 rank 负责的基础词表大小（不含 padding）和扩展词表大小
        num_org = int(shard.num_org_elements)
        num_org_padded = int(shard.num_org_elements_padded)
        num_added = int(shard.num_added_elements)
        org_vocab_start = int(shard.org_vocab_start_index)
        added_vocab_start = int(shard.added_vocab_start_index)

        num_tokens = int(hidden_states.shape[0])
        out_token_ids = torch.empty(
            (num_tokens,), dtype=torch.long, device=hidden_states.device
        )
        def _cast_hs(x: torch.Tensor) -> torch.Tensor:
            # 确保 hidden states 与 lm_head 权重数据类型一致（如 bf16/fp16 混合精度）
            return x if x.dtype == weight_dtype else x.to(weight_dtype)

        # Fast path (common): single-rank greedy sampling over the base vocab shard.
        # Avoids extra max/id bookkeeping that is only needed for TP sync or added vocab.
        # 快速路径：单 rank 且无扩展词表时，直接逐 chunk matmul + argmax（最常见场景）
        if tp_size == 1 and num_added == 0:
            for start in range(0, num_tokens, int(chunk_size)):
                end = min(num_tokens, start + int(chunk_size))
                hs = _cast_hs(hidden_states[start:end])
                if num_org > 0:
                    # matmul(hs, W^T) + argmax → 局部词表中的最优 token 索引
                    base_logits = torch.matmul(hs, weight[:num_org].T)
                    out_token_ids[start:end] = (
                        torch.argmax(base_logits, dim=-1).to(torch.long)
                        + org_vocab_start
                    )
                else:
                    out_token_ids[start:end] = 0
            return out_token_ids

        # TP>1 或有扩展词表时的通用路径：逐 chunk 计算局部 argmax 后 all_gather 选全局最大
        for start in range(0, num_tokens, int(chunk_size)):
            end = min(num_tokens, start + int(chunk_size))
            hs = _cast_hs(hidden_states[start:end])
            chunk_len = int(hs.shape[0])

            # Base vocab logits.
            # 计算本 rank 基础词表对应的 logits，取最大值及其索引
            if num_org > 0:
                base_logits = torch.matmul(hs, weight[:num_org].T)
                local_max, local_arg = torch.max(base_logits, dim=-1)
            else:
                # 本 rank 没有基础词表 token：初始化为负无穷（会被其他 rank 的值覆盖）
                local_max = torch.full(
                    (chunk_len,),
                    torch.finfo(weight_dtype).min,
                    dtype=weight_dtype,
                    device=hs.device,
                )
                local_arg = torch.zeros(
                    (chunk_len,), dtype=torch.int64, device=hs.device
                )

            # Added vocab logits (e.g., LoRA-added embeddings), if present.
            # 扩展词表（如 LoRA 添加的 embedding）：与基础词表竞争 argmax
            if num_added > 0:
                added_slice_start = num_org_padded
                added_slice_end = num_org_padded + num_added
                added_logits = torch.matmul(
                    hs, weight[added_slice_start:added_slice_end].T
                )
                added_max, added_arg = torch.max(added_logits, dim=-1)
                use_added = added_max > local_max
                local_max = torch.where(use_added, added_max, local_max)
                # For base/added conversion below, keep local_arg expressed in the full local
                # weight index space (base + padding + added), matching `lm_head.weight`.
                # 统一用全局 weight 索引空间表示局部 argmax（方便后续 ID 转换）
                local_arg = torch.where(
                    use_added, added_arg.to(local_arg.dtype) + num_org_padded, local_arg
                )

            # Convert local argmax indices to global token ids.
            # 将局部词表索引转换为全局 token ID（加上本 rank 词表起始偏移）
            if num_added == 0:
                local_arg.add_(org_vocab_start)
                global_ids = local_arg
            else:
                # 需要区分来自基础词表还是扩展词表的 ID 转换
                global_ids = torch.empty(
                    (chunk_len,), dtype=torch.int64, device=hs.device
                )
                is_base = local_arg < num_org
                global_ids[is_base] = org_vocab_start + local_arg[is_base]
                global_ids[~is_base] = added_vocab_start + (
                    local_arg[~is_base] - num_org_padded
                )

            if tp_size == 1:
                # 单 rank 无需 all_gather，直接写出
                out_token_ids[start:end] = global_ids.to(torch.long)
                continue

            # Gather per-rank maxima and associated global ids, then select the global max.
            # TP>1：all_gather 各 rank 的 (local_max, global_ids)，选全局最大 logit 对应的 token
            needed = tp_size * chunk_len
            chunk_cap = int(chunk_size)
            if (
                self._draft_greedy_gather_cap < needed
                or self._draft_greedy_gathered_max_buf is None
                or self._draft_greedy_gathered_ids_buf is None
                or self._draft_greedy_gathered_max_buf.dtype != local_max.dtype
                or self._draft_greedy_gathered_max_buf.device != hs.device
            ):
                # Allocate enough space for the max chunk size to avoid reallocations.
                # 按最大 chunk 容量预分配 gather 缓冲区（减少重分配次数）
                cap = tp_size * chunk_cap
                self._draft_greedy_gathered_max_buf = torch.empty(
                    (cap,), dtype=local_max.dtype, device=hs.device
                )
                self._draft_greedy_gathered_ids_buf = torch.empty(
                    (cap,), dtype=global_ids.dtype, device=hs.device
                )
                self._draft_greedy_gather_cap = cap

            if (
                self._draft_greedy_index_cap < chunk_len
                or self._draft_greedy_best_rank_buf is None
                or self._draft_greedy_rank_index_buf is None
                or self._draft_greedy_selected_ids_buf is None
                or self._draft_greedy_best_rank_buf.device != hs.device
                or self._draft_greedy_selected_ids_buf.device != hs.device
            ):
                self._draft_greedy_best_rank_buf = torch.empty(
                    (chunk_cap,), dtype=torch.int64, device=hs.device
                )
                self._draft_greedy_rank_index_buf = torch.empty(
                    (1, chunk_cap), dtype=torch.int64, device=hs.device
                )
                self._draft_greedy_selected_ids_buf = torch.empty(
                    (1, chunk_cap), dtype=torch.int64, device=hs.device
                )
                self._draft_greedy_index_cap = chunk_cap

            gathered_max = self._draft_greedy_gathered_max_buf[:needed]
            gathered_ids = self._draft_greedy_gathered_ids_buf[:needed]

            # all_gather 各 rank 的局部最大 logit 值和对应全局 token ID
            tp_group.all_gather_into_tensor(gathered_max, local_max.contiguous())
            tp_group.all_gather_into_tensor(gathered_ids, global_ids.contiguous())
            # reshape 为 [tp_size, chunk_len] 便于逐 token 选最优 rank
            gathered_max = gathered_max.view(tp_size, chunk_len)
            gathered_ids = gathered_ids.view(tp_size, chunk_len)

            # 选择每个 token 在哪个 rank 的 logit 最大
            best_rank = self._draft_greedy_best_rank_buf[:chunk_len]
            torch.argmax(gathered_max, dim=0, out=best_rank)

            # 用 best_rank 索引到 gathered_ids，取出全局 greedy token ID
            rank_index = self._draft_greedy_rank_index_buf[:, :chunk_len]
            rank_index[0].copy_(best_rank)
            selected_ids = self._draft_greedy_selected_ids_buf[:, :chunk_len]
            torch.gather(gathered_ids, 0, rank_index, out=selected_ids)
            out_token_ids[start:end].copy_(selected_ids.view(-1))

        return out_token_ids

    def _append_target_hidden_to_draft_kv(
        self,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInput,
    ) -> None:
        """Materialize the target hidden-state features into the draft KV cache.

        This must be run before exposing new tokens to radix cache (prefix hits), otherwise
        another request could reuse target KV indices without having draft KV values.
        """
        # 将目标模型的 hidden states 写入草稿 KV cache
        # 草稿模型利用这些特征做非因果注意力（DFLASH 区别于 EAGLE 的关键：直接用 target hidden）
        # 必须在 radix cache 暴露新 token 前执行（否则会有 KV 缺失的前缀命中）

        bs = batch.batch_size()
        device = self.model_runner.device

        if draft_input.target_hidden is None:
            raise RuntimeError(
                "DFLASH draft state missing target_hidden context features."
            )
        if draft_input.ctx_lens.numel() != bs:
            raise RuntimeError(
                f"DFLASH ctx_lens length mismatch: got {draft_input.ctx_lens.numel()} for bs={bs}."
            )
        if draft_input.draft_seq_lens.numel() != bs:
            raise RuntimeError(
                f"DFLASH draft_seq_lens length mismatch: got {draft_input.draft_seq_lens.numel()} for bs={bs}."
            )

        total_ctx = int(draft_input.target_hidden.shape[0])
        if total_ctx <= 0:
            # 无新 context token 可追加，直接返回（等待下轮）
            draft_input.ctx_lens = torch.zeros_like(draft_input.ctx_lens)
            draft_input.target_hidden = draft_input.target_hidden[:0]
            return

        # target req_to_token：目标模型的 KV cache 位置表（隐状态来自此处）
        # draft req_to_token：草稿模型的 KV cache 位置表（写入目标）
        target_req_to_token = batch.req_to_token_pool.req_to_token
        draft_req_to_token = self.draft_model_runner.req_to_token_pool.req_to_token

        req_pool_indices = batch.req_pool_indices
        if req_pool_indices.dtype != torch.int64:
            req_pool_indices = req_pool_indices.to(torch.int64)

        # ctx_lens：本批次每个序列新增的 context token 数（目标模型产出的 hidden 数量）
        ctx_lens = draft_input.ctx_lens
        if ctx_lens.dtype != torch.int32:
            ctx_lens = ctx_lens.to(torch.int32)
        if ctx_lens.device != device:
            ctx_lens = ctx_lens.to(device, non_blocking=True)
        # ctx_start：新 context 在目标序列中的起始位置（从右侧 ctx_lens 个 token）
        ctx_start = batch.seq_lens.to(torch.int64) - ctx_lens.to(torch.int64)

        if bs == 1:
            # Fast path for single request.
            # 单请求快速路径：直接切片 _block_pos_offsets（避免 arange 分配）
            max_ctx = int(total_ctx)
            if max_ctx <= self._block_pos_offsets.numel():
                r = self._block_pos_offsets[:max_ctx]
            else:
                r = torch.arange(max_ctx, device=device, dtype=torch.int64)
            pos2d = ctx_start[:, None] + r[None, :]  # [1, ctx]
            cache2d = target_req_to_token[req_pool_indices[:, None], pos2d]  # [1, ctx]
            ctx_cache_loc = cache2d.reshape(-1).to(torch.int64)  # [ctx]
            ctx_positions = pos2d.reshape(-1)  # [ctx]
        else:
            # In decode mode, ctx_lens <= block_size so we can skip the .item() sync.
            # decode 模式下 ctx_lens <= block_size，可跳过 .item() 同步（节省 CPU-GPU 往返）
            if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
                max_ctx = int(ctx_lens.max().item())
            else:
                max_ctx = int(self.block_size)
            if max_ctx <= 0:
                raise RuntimeError(f"DFLASH invalid max_ctx={max_ctx} for KV append.")

            if max_ctx <= self._block_pos_offsets.numel():
                r = self._block_pos_offsets[:max_ctx]
            else:
                r = torch.arange(max_ctx, device=device, dtype=torch.int64)
            r = r[None, :]  # [1, max_ctx]
            pos2d = ctx_start[:, None] + r  # [bs, max_ctx]
            # 布尔 mask：只保留 offset < ctx_lens 的有效 token（不规则批次）
            mask = r < ctx_lens[:, None]

            # Batched gather of cache locations and positions.
            # 批量 gather 目标 req_to_token 中的 KV cache 位置（变长安全）
            ctx_cache_loc = self._gather_req_to_token_masked(
                req_to_token=target_req_to_token,
                req_pool_indices=req_pool_indices,
                pos2d=pos2d,
                mask=mask,
                context="DFLASH target hidden KV append",
            )  # [sum(ctx_lens)]
            ctx_positions = pos2d[mask]  # [sum(ctx_lens)]

        with torch.inference_mode():
            # 草稿模型将目标 hidden states 投影到草稿 KV 空间（线性变换）
            ctx_hidden = self.draft_model.project_target_hidden(
                draft_input.target_hidden
            )  # [sum(ctx), hidden]
            if ctx_hidden.shape[0] != ctx_cache_loc.numel():
                raise RuntimeError(
                    f"DFLASH ctx_hidden/cache_loc mismatch: {ctx_hidden.shape[0]} vs {ctx_cache_loc.numel()}."
                )

            # 优先使用 fused KV 物化路径（单次 kernel 处理所有层）
            if self._use_fused_kv_materialize and self._fused_kv_helper is not None:
                try:
                    self._append_target_hidden_fused(
                        ctx_hidden, ctx_positions, ctx_cache_loc
                    )
                except Exception as e:
                    # fused 路径失败时降级到逐层串行路径
                    logger.warning(
                        "DFLASH fused KV append failed; falling back to sequential path: %s",
                        e,
                    )
                    self._use_fused_kv_materialize = False
                    self._fused_kv_helper = None
                    self._append_target_hidden_sequential(
                        ctx_hidden, ctx_positions, ctx_cache_loc
                    )
            else:
                # 逐层串行 KV 写入（无 fused kernel 时的 fallback）
                self._append_target_hidden_sequential(
                    ctx_hidden, ctx_positions, ctx_cache_loc
                )

        if self.use_compact_draft_cache:
            # compact（滑动窗口）模式：草稿 req_to_token 只保存最近 draft_window_size 个 token
            new_draft_seq_lens = self._compute_compact_draft_seq_lens(batch.seq_lens)
            suffix_start = batch.seq_lens.to(torch.int64) - new_draft_seq_lens.to(
                torch.int64
            )
            # 从目标 req_to_token 中 gather 窗口内的 KV cache 位置
            suffix_cache_loc = self._gather_req_to_token_segments(
                req_to_token=target_req_to_token,
                req_pool_indices=req_pool_indices,
                start=suffix_start,
                lengths=new_draft_seq_lens,
            )
            # 将窗口内的 KV 位置写入草稿 req_to_token（从位置 0 开始覆盖）
            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                draft_req_to_token,
                torch.zeros_like(new_draft_seq_lens),
                new_draft_seq_lens,
                suffix_cache_loc,
                bs,
            )
            draft_input.draft_seq_lens = new_draft_seq_lens
        else:
            # 非 compact 模式：草稿前缀长度等于目标前缀长度（共享全量 KV）
            draft_input.draft_seq_lens = batch.seq_lens.to(dtype=torch.int32)
        # 清空 ctx_lens 和 target_hidden（已写入 draft KV，无需重复处理）
        draft_input.ctx_lens = torch.zeros_like(ctx_lens)
        draft_input.target_hidden = draft_input.target_hidden[:0]

    def _append_target_hidden_sequential(
        self,
        ctx_hidden: torch.Tensor,
        ctx_positions: torch.Tensor,
        ctx_cache_loc: torch.Tensor,
    ) -> None:
        # 逐层串行将 hidden states 投影为 KV 并写入 draft KV cache
        # 对每个草稿 Transformer 层：kv_proj → k_norm → k_rope → set_kv_buffer
        for layer in self.draft_model.layers:
            attn = layer.self_attn
            k, v = attn.kv_proj_only(ctx_hidden)
            # 对 K 做归一化（如 RMSNorm）和旋转位置编码（RoPE）
            k = attn.apply_k_norm(k)
            k = attn.apply_k_rope(ctx_positions, k)
            k = k.view(-1, attn.num_kv_heads, attn.head_dim)
            v = v.view(-1, attn.num_kv_heads, attn.head_dim)
            # 将 K/V 写入对应 KV cache 槽位（ctx_cache_loc 指定写入位置）
            self.draft_model_runner.token_to_kv_pool.set_kv_buffer(
                attn.attn,
                ctx_cache_loc,
                k,
                v,
                attn.attn.k_scale,
                attn.attn.v_scale,
            )

    def _append_target_hidden_fused(
        self,
        ctx_hidden: torch.Tensor,
        ctx_positions: torch.Tensor,
        ctx_cache_loc: torch.Tensor,
    ) -> None:
        """Fused KV materialization using batched projection + Triton kernel."""
        # 融合 KV 物化路径：使用 Triton kernel 将所有层的 KV 投影和写入合并为一次 pass
        # 性能优势：减少 Python 层循环开销和中间张量分配（对大 batch 效果显著）
        token_to_kv_pool = self.draft_model_runner.token_to_kv_pool
        layers = self.draft_model.layers

        def _write_layer_kv(
            layer_idx: int, cache_k: torch.Tensor, cache_v: torch.Tensor
        ) -> None:
            # 回调函数：将第 layer_idx 层的 K/V 写入对应 KV cache 槽位
            attn = layers[layer_idx].self_attn.attn
            token_to_kv_pool.set_kv_buffer(
                attn,
                ctx_cache_loc,
                cache_k,
                cache_v,
                attn.k_scale,
                attn.v_scale,
            )

        # 调用 fused kernel：一次性对所有层做 proj → norm → rope → 写 KV
        self._fused_kv_helper.materialize(
            ctx_hidden=ctx_hidden,
            positions=ctx_positions,
            write_layer_kv=_write_layer_kv,
        )

    def _update_target_mamba_state_after_verify(
        self,
        *,
        batch: ScheduleBatch,
        seq_lens_pre_verify: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        """Commit Mamba intermediate states for accepted verify steps.

        During TARGET_VERIFY, Mamba kernels run with `disable_state_update=True` and
        cache per-step intermediate states. After acceptance, we need to commit the
        state corresponding to each request's last accepted step.
        """
        # Mamba SSM 状态更新：TARGET_VERIFY 期间禁用了状态写入，验证接受后需手动提交
        # 找到每个请求最后一个被接受步骤对应的 Mamba 状态并写回
        attn_backend = self.target_worker.model_runner.attn_backend
        if not hasattr(attn_backend, "update_mamba_state_after_mtp_verify"):
            # 非 Mamba 模型跳过（不支持此接口）
            return

        # accepted_steps[i] = commit_lens[i] - 1（0-indexed 的最后接受步）
        accepted_steps = commit_lens.to(torch.int64) - 1
        mamba_steps_to_track = None

        if batch.mamba_track_indices is not None:
            # 计算需要追踪的 Mamba 状态快照（用于序列从某个间隔点继续时的状态恢复）
            mamba_track_interval = self.server_args.mamba_track_interval
            # to_track_mask：该批次中哪些请求跨越了 track 间隔边界
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != batch.seq_lens // mamba_track_interval
            )
            # tracking_point：该请求最近一个 track 点的绝对位置
            tracking_point = (
                batch.seq_lens // mamba_track_interval * mamba_track_interval
            )
            # to_track_ith：需要追踪的步骤是验证序列中的第几步（相对于 seq_lens_pre_verify）
            to_track_ith = torch.clamp(tracking_point - seq_lens_pre_verify - 1, min=0)
            # can_track_mask：只有实际被接受（commit_lens > to_track_ith）时才追踪
            can_track_mask = to_track_mask & (
                to_track_ith < commit_lens.to(to_track_ith.dtype)
            )
            # -1 表示不追踪该请求的 Mamba 状态
            mamba_steps_to_track = torch.where(
                can_track_mask,
                to_track_ith.to(torch.int64),
                torch.full_like(to_track_ith, -1, dtype=torch.int64),
            )

        # 调用 attn_backend 的 Mamba 状态更新接口，提交接受步骤对应的状态
        attn_backend.update_mamba_state_after_mtp_verify(
            accepted_steps=accepted_steps,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=self.target_worker.model_runner.model,
        )

    def forward_batch_generation(
        self,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        **kwargs,
    ) -> GenerationBatchResult:
        # DFlash 主入口：处理 prefill（extend）和 speculative decode（target_verify）两条路径
        if getattr(batch, "return_logprob", False):
            raise RuntimeError(
                "Invariant broken: DFLASH batch requested return_logprob, but scheduler should have rejected this request."
            )

        if isinstance(batch, ModelWorkerBatch):
            # Should not happen for spec-v1 (non-overlap) scheduling, but keep a sane fallback.
            # ModelWorkerBatch 不应在 spec-v1 路径出现，fallback 到目标 worker
            return self.target_worker.forward_batch_generation(batch, **kwargs)

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            # --- Prefill 路径 ---
            # 以 FULL hidden capture 模式运行目标模型前向（收集所有层 hidden states）
            model_worker_batch = batch.get_model_worker_batch()
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL

            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch, **kwargs
            )
            logits_output, next_token_ids = (
                batch_result.logits_output,
                batch_result.next_token_ids,
            )
            if logits_output.hidden_states is None:
                raise RuntimeError(
                    "DFLASH requires target aux hidden capture for prefill, but got None. "
                    "Make sure the target model has DFlash layers-to-capture configured."
                )

            if (
                model_worker_batch.extend_seq_lens is None
                or model_worker_batch.extend_prefix_lens is None
            ):
                raise RuntimeError(
                    "DFLASH expected extend_seq_lens / extend_prefix_lens to be populated in extend mode, but got None."
                )

            # Materialize the prompt tokens into the draft KV cache immediately. This is required
            # for radix cache support, since the scheduler may update radix after prefill returns.
            # Prefill 结束后立即将提示 token 对应的 hidden states 写入草稿 KV cache
            # （radix cache 更新在 prefill 返回后进行，需提前写入草稿 KV 防止引用空缺）
            device = next_token_ids.device

            def _to_int32_device_tensor(x, *, device=device):
                # 辅助函数：将张量或整数列表转为 int32 设备张量
                if isinstance(x, torch.Tensor):
                    if x.device != device:
                        x = x.to(device, non_blocking=True)
                    return x if x.dtype == torch.int32 else x.to(torch.int32)
                return torch.tensor(x, dtype=torch.int32, device=device)

            extend_seq_lens = _to_int32_device_tensor(
                model_worker_batch.extend_seq_lens
            )
            # 构造初始 DFlashDraftInput：
            # - verified_id：prefill 的最后一个 next token
            # - target_hidden：目标模型捕获的 hidden states
            # - ctx_lens：extend 的新增长度（需写入草稿 KV）
            # - draft_seq_lens：compact 模式为 0（从头建），否则用 extend_prefix_lens
            draft_input = DFlashDraftInput(
                verified_id=next_token_ids.to(torch.int64),
                target_hidden=logits_output.hidden_states,
                ctx_lens=extend_seq_lens,
                draft_seq_lens=(
                    torch.zeros_like(extend_seq_lens)
                    if self.use_compact_draft_cache
                    else _to_int32_device_tensor(model_worker_batch.extend_prefix_lens)
                ),
            )
            # 将 prefill hidden states 物化到草稿 KV cache
            self._append_target_hidden_to_draft_kv(batch, draft_input)
            batch.spec_info = draft_input

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_drafts=0,
                can_run_cuda_graph=batch_result.can_run_cuda_graph,
            )

        # Decode / target-verify stage.
        # --- Decode 路径：草稿 → 验证 → 接受/拒绝 ---
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInput):
            raise RuntimeError(
                "DFLASH decode requires DFlashDraftInput state on the running batch. "
                "This usually means the request did not complete the prefill stage."
            )

        # 步骤 1：草稿生成（填充 block + 非因果前向 + greedy 采样）
        self._prepare_for_speculative_decoding(batch, draft_input)

        model_worker_batch = batch.get_model_worker_batch()
        assert model_worker_batch.forward_mode.is_target_verify()
        verify_input = model_worker_batch.spec_info
        assert isinstance(verify_input, DFlashVerifyInput)
        # 如果目标模型含 Mamba 层，需要在验证后提交 SSM 状态
        need_mamba_verify_commit = hasattr(
            self.target_worker.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )
        seq_lens_pre_verify = (
            batch.seq_lens.clone() if need_mamba_verify_commit else None
        )

        # 步骤 2：目标模型验证（TARGET_VERIFY 模式，对草稿 block 做并行验证）
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True, **kwargs
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        # 步骤 3：接受/拒绝判断，得到新的 verified_id、提交长度、下一步 hidden states
        (
            new_verified_id,
            commit_lens,
            next_target_hidden,
            num_accepted_drafts_per_req_cpu,
        ) = verify_input.verify(
            batch=batch,
            logits_output=logits_output,
            page_size=self.page_size,
        )
        # 步骤 4：Mamba 状态提交（仅 SSM 模型需要）
        if need_mamba_verify_commit:
            assert seq_lens_pre_verify is not None
            self._update_target_mamba_state_after_verify(
                batch=batch,
                seq_lens_pre_verify=seq_lens_pre_verify,
                commit_lens=commit_lens,
            )

        # Update draft state for the next iteration. Also materialize the committed verify tokens
        # into the draft KV cache immediately so radix cache entries are safe to reuse.
        # 步骤 5：更新草稿状态（verified_id / target_hidden / ctx_lens）并立即写入草稿 KV
        draft_input.verified_id = new_verified_id
        draft_input.target_hidden = next_target_hidden
        draft_input.ctx_lens = commit_lens
        self._append_target_hidden_to_draft_kv(batch, draft_input)
        batch.spec_info = draft_input
        # 重置 forward_mode 为 DECODE（供下一轮调度使用）
        batch.forward_mode = ForwardMode.DECODE

        num_accepted_drafts = sum(num_accepted_drafts_per_req_cpu)
        if not self._logged_first_verify and self.tp_rank == 0:
            logger.info(
                "DFLASH verify completed. num_accepted_drafts_per_req=%s",
                num_accepted_drafts_per_req_cpu,
            )
            self._logged_first_verify = True

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=new_verified_id,
            num_accepted_drafts=num_accepted_drafts,
            num_accepted_drafts_per_req_cpu=num_accepted_drafts_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
        )
