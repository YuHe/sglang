from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# TP 通信组（用于跨 TP rank 广播采样结果）
from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,  # DP attention 模式下的 TP 通信组
    is_dp_attention_enabled,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
# 获取每次 decode 预分配的 KV slot 数（spec-v2 超量预分配策略）
from sglang.srt.managers.utils import get_alloc_len_per_decode
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.model_runner import ModelRunner
# apply_scaling_penalties: 乘性惩罚（repetition penalty）应用函数
from sglang.srt.sampling.penaltylib.repetition_penalty import apply_scaling_penalties
from sglang.srt.server_args import get_global_server_args
# verify_tree_greedy_func: 贪心树验证（跨平台实现）
from sglang.srt.speculative.eagle_utils import verify_tree_greedy_func
from sglang.srt.speculative.spec_utils import (
    SIMULATE_ACC_LEN,              # 模拟接受长度（调试用，生产环境为 0）
    generate_simulated_accept_index,  # 生成模拟接受索引（绕过真实验证）
)
from sglang.srt.utils.common import is_cuda, is_hip, is_musa, is_npu, next_power_of_2

# 平台标志：用于在 kernel 选择和条件分支中避免重复调用函数
_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_musa = is_musa()

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput

# CUDA/MUSA 平台：加载采样相关 kernel
if is_cuda() or is_musa():
    from sgl_kernel import (
        top_k_renorm_prob,                         # top-k 截断并重归一化
        top_p_renorm_prob,                         # top-p (nucleus) 截断并重归一化
        tree_speculative_sampling_target_only,     # 树形投机采样（仅目标概率）
    )


@triton.jit
def assign_draft_cache_locs_page_size_1(
    req_pool_indices,
    req_to_token,
    seq_lens,
    out_cache_loc,
    pool_len: tl.constexpr,
    topk: tl.constexpr,
    speculative_num_steps: tl.constexpr,
):
    # Triton kernel：将每个请求已分配的草稿 KV slot 位置复制到 out_cache_loc
    # Grid: (bs,) — 每个请求对应一个 program
    # 使用场景：spec-v2 decode 阶段，从 req_to_token_pool 中读取已提前分配的 slot 位置
    BLOCK_SIZE: tl.constexpr = 128
    pid = tl.program_id(axis=0)

    # 每个请求需复制 topk * speculative_num_steps 个 slot 位置
    copy_len = topk * speculative_num_steps
    out_cache_ptr = out_cache_loc + pid * topk * speculative_num_steps

    # 从 req_to_token[req_pool_indices[pid], seq_lens[pid]:] 复制 slot 位置
    kv_start = tl.load(seq_lens + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len
    num_loop = tl.cdiv(copy_len, BLOCK_SIZE)
    for i in range(num_loop):
        copy_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = copy_offset < copy_len
        data = tl.load(token_pool + kv_start + copy_offset, mask=mask)
        tl.store(out_cache_ptr + copy_offset, data, mask=mask)


@dataclass
class EagleDraftInputV2Mixin:
    def prepare_for_decode(self: EagleDraftInput, batch: ScheduleBatch):
        # 可能触发 SWA（Sliding Window Attention）的 KV 缓存驱逐
        batch.maybe_evict_swa()

        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

        bs = batch.batch_size()

        # 等待目标模型验证完成（overlap scheduler 下验证与草稿并行）
        # Now seq_lens is correct
        batch.maybe_wait_verify_done()

        # 累积惩罚项（宽松版本：仅基于最后一个 token，而非完整 output_ids）
        if batch.sampling_info.penalizer_orchestrator.is_required:
            output_ids = torch.tensor(
                [
                    (
                        req.output_ids[-1]
                        if len(req.output_ids)
                        else req.origin_input_ids[-1]
                    )
                    for req in batch.reqs
                ],
                dtype=torch.int64,
                device=batch.device,
            )
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                output_ids
            )

        page_size = batch.token_to_kv_pool_allocator.page_size
        cur_kv_lens_cpu = []
        nxt_kv_lens_cpu = []
        num_needed_tokens = 0
        alloc_len_per_decode = get_alloc_len_per_decode()
        for r in batch.reqs:
            # Over-allocation happens here
            # 超量预分配：每次多分配 2 * alloc_len_per_decode 个 slot（减少频繁分配开销）
            x = r.kv_committed_len + 2 * alloc_len_per_decode - r.kv_allocated_len
            cur_kv_lens_cpu.append(r.kv_allocated_len)
            nxt_kv_lens_cpu.append(r.kv_allocated_len + x)
            num_needed_tokens += x
            r.kv_allocated_len += x
            r.decode_batch_idx += 1
            # Pre-claim bonus slot here (like normal decode); resolve subtracts 1.
            # 预占 bonus slot（正常 decode 逻辑兼容）；resolve 阶段会减 1
            r.kv_committed_len += 1

        cur_kv_lens_cpu = torch.tensor(cur_kv_lens_cpu, dtype=torch.int32, device="cpu")
        nxt_kv_lens_cpu = torch.tensor(nxt_kv_lens_cpu, dtype=torch.int32, device="cpu")

        if page_size == 1:
            # 非分页模式：直接分配连续 slot
            out_cache_loc = alloc_token_slots(batch.tree_cache, num_needed_tokens)
        else:
            # 分页模式：基于最后 loc 扩展分配分页 slot
            cur_kv_lens = cur_kv_lens_cpu.to(device=batch.device)
            nxt_kv_lens = nxt_kv_lens_cpu.to(device=batch.device)
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                cur_kv_lens,
            )
            out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                cur_kv_lens,
                cur_kv_lens_cpu,
                nxt_kv_lens,
                nxt_kv_lens_cpu,
                last_loc,
                num_needed_tokens,
            )

        # 将新分配的 slot 写入 req_to_token_pool 的页表映射
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            cur_kv_lens_cpu.to(device=batch.device),
            nxt_kv_lens_cpu.to(device=batch.device),
            out_cache_loc,
            bs,
        )

        # FIXME(lsyin): make this sync optional
        # 同步 seq_lens 到 CPU（overlap scheduler 下需要精确的 seq_lens_cpu）
        batch.seq_lens_cpu = batch.seq_lens.cpu()
        batch.seq_lens_sum = batch.seq_lens_cpu.sum().item()

    def prepare_for_v2_draft(
        self: EagleDraftInput,
        req_to_token_pool: ReqToTokenPool,
        batch: ModelWorkerBatch,
        cuda_graph_runner: EAGLEDraftCudaGraphRunner,
        draft_model_runner: ModelRunner,
        topk: int,
        num_steps: int,
    ):
        if not batch.forward_mode.is_idle():
            bs = len(batch.seq_lens)

            # 为草稿 decode 阶段分配 out_cache_loc（从 req_to_token_pool 中读取已预分配的 slot）
            batch.out_cache_loc = torch.empty(
                (bs * topk * num_steps,),
                dtype=torch.int64,
                device=batch.input_ids.device,
            )
            # FIXME(lsyin): align with the default code path
            # 调用 Triton kernel 填充 out_cache_loc（page_size=1 的快速路径）
            assign_draft_cache_locs_page_size_1[(bs,)](
                batch.req_pool_indices,
                req_to_token_pool.req_to_token,
                batch.seq_lens,
                batch.out_cache_loc,
                req_to_token_pool.req_to_token.shape[1],
                topk,
                num_steps,
            )

        # 构造草稿 ForwardBatch
        # num_tokens_per_req = topk：每个请求有 topk 个草稿 token 输入
        self.num_tokens_per_req = topk
        self.num_tokens_for_logprob_per_req = topk
        batch.capture_hidden_mode = CaptureHiddenMode.LAST
        # positions: 每个请求的草稿 token 共享相同位置（= 当前 seq_len）
        self.positions = batch.seq_lens.repeat_interleave(topk, dim=0)
        forward_batch = ForwardBatch.init_new(batch, draft_model_runner)
        # 判断是否可以使用 CUDA Graph replay
        can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run(forward_batch)
        return forward_batch, can_cuda_graph

    def prepare_for_extend_to_fill_draft_kvcache(
        self,
        batch: ModelWorkerBatch,
        predict: torch.Tensor,
        num_draft_tokens: int,
        draft_model_runner: Any,
        cuda_graph_runner: Any,
    ):
        # 在 extend 阶段将草稿 token 写入 draft KV 缓存（DRAFT_EXTEND_V2 模式）
        seq_lens_cpu_ = batch.seq_lens_cpu
        extend_num_tokens = len(batch.seq_lens) * num_draft_tokens

        batch.spec_info = self
        batch.input_ids = predict
        # 更新 seq_lens 和 seq_lens_sum（加入本轮草稿 token）
        batch.seq_lens = batch.seq_lens + num_draft_tokens
        batch.seq_lens_cpu = batch.seq_lens_cpu + num_draft_tokens
        batch.seq_lens_sum += extend_num_tokens
        # 每个请求 extend num_draft_tokens 个 token
        batch.extend_seq_lens = [num_draft_tokens for _ in range(len(batch.seq_lens))]
        batch.extend_prefix_lens = seq_lens_cpu_.tolist()
        batch.extend_num_tokens = extend_num_tokens
        # FULL 模式：捕获所有 token 的隐状态（供后续草稿 KV 物化使用）
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.DRAFT_EXTEND_V2
        )
        forward_batch = ForwardBatch.init_new(batch, draft_model_runner)
        can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run(forward_batch)
        if not batch.forward_mode.is_idle() and not can_cuda_graph:
            # 无 CUDA Graph：手动初始化注意力元数据
            draft_model_runner.attn_backend.init_forward_metadata(forward_batch)
        return forward_batch


@dataclass
class EagleVerifyInputV2Mixin:
    def prepare_for_v2_verify(
        self: EagleVerifyInput,
        req_to_token_pool: ReqToTokenPool,
        batch: ModelWorkerBatch,
        target_worker: TpModelWorker,
    ):
        if not batch.forward_mode.is_idle():
            # 为目标模型验证阶段分配 out_cache_loc
            bs = len(batch.req_pool_indices)
            batch.input_ids = self.draft_token
            device = batch.input_ids.device
            # 从 req_to_token_pool 中读取验证 token 范围内的 KV slot 位置
            batch.out_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=req_to_token_pool.req_to_token,
                start_offset=batch.seq_lens,
                end_offset=batch.seq_lens + self.draft_token_num,
                batch_size=bs,
                draft_token_num=self.draft_token_num,
                device=device,
            )

            # Set mamba_track_indices for mamba prefix-cache state tracking
            # Mamba 状态跟踪索引：用于 ping-pong 缓冲区管理
            if get_global_server_args().enable_mamba_extra_buffer():
                mapping = (
                    req_to_token_pool.req_index_to_mamba_ping_pong_track_buffer_mapping
                )
                req_pool_idx_tensor = batch.req_pool_indices.to(
                    device=mapping.device, dtype=torch.int64
                )
                track_col_idx = torch.tensor(
                    [req.mamba_next_track_idx for req in batch.reqs],
                    dtype=torch.int64,
                    pin_memory=True,
                ).to(mapping.device, non_blocking=True)
                batch.mamba_track_indices = mapping[
                    req_pool_idx_tensor, track_col_idx
                ].to(dtype=torch.int64)
                batch.mamba_track_mask = None
                batch.mamba_track_seqlens = None

        # 切换 forward_mode 为 TARGET_VERIFY，捕获 FULL 隐状态
        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.TARGET_VERIFY
        )
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        verify_forward_batch = ForwardBatch.init_new(batch, target_worker.model_runner)

        # 判断目标模型验证是否可以使用 CUDA Graph replay
        can_run_cuda_graph = bool(
            target_worker.model_runner.graph_runner
            and target_worker.model_runner.graph_runner.can_run(verify_forward_batch)
        )
        if can_run_cuda_graph:
            # CUDA Graph replay 前的准备工作
            target_worker.model_runner.graph_runner.replay_prepare(verify_forward_batch)
        else:
            if not batch.forward_mode.is_idle():
                # 无 CUDA Graph：手动初始化注意力元数据
                target_worker.model_runner.attn_backend.init_forward_metadata(
                    verify_forward_batch
                )

        return verify_forward_batch, can_run_cuda_graph

    def sample(
        self: EagleVerifyInput,
        batch: ModelWorkerBatch,
        logits_output: LogitsProcessorOutput,
        vocab_mask: torch.Tensor = None,
    ):
        """
        Verify and find accepted tokens based on logits output and batch
        (which contains spec decoding information).
        """
        # idle 模式直接返回空张量
        if batch.forward_mode.is_idle():
            predict = torch.empty(0, dtype=torch.int32, device=batch.input_ids.device)
            num_accepted_drafts = torch.empty(
                0, dtype=torch.int32, device=batch.input_ids.device
            )
            accept_index = torch.empty(
                0, dtype=torch.int32, device=batch.input_ids.device
            )
            return predict, num_accepted_drafts, accept_index

        bs = len(batch.seq_lens)
        sampling_info = batch.sampling_info
        next_token_logits = logits_output.next_token_logits
        device = batch.input_ids.device

        # 应用加性惩罚（repetition/frequency penalty 的加法形式）
        # This is a relaxed version of penalties for speculative decoding.
        if sampling_info.acc_additive_penalties is not None:
            next_token_logits.add_(
                torch.repeat_interleave(
                    sampling_info.acc_additive_penalties, self.draft_token_num, dim=0
                )
            )
        # 应用乘性惩罚（presence penalty 的乘法形式）
        if sampling_info.acc_scaling_penalties is not None:
            apply_scaling_penalties(
                next_token_logits,
                torch.repeat_interleave(
                    sampling_info.acc_scaling_penalties, self.draft_token_num, dim=0
                ),
            )
        # 应用 logit_bias（固定 token 偏置）
        if sampling_info.logit_bias is not None:
            next_token_logits.add_(
                torch.repeat_interleave(
                    sampling_info.logit_bias, self.draft_token_num, dim=0
                )
            )

        # 应用 grammar 词表掩码（结构化输出）
        if vocab_mask is not None:
            assert self.grammar is not None
            self.grammar.apply_vocab_mask(
                logits=next_token_logits, vocab_mask=vocab_mask
            )

        # candidates: [bs, draft_token_num]，草稿 token 矩阵
        candidates = self.draft_token.reshape(bs, self.draft_token_num)
        predict_shape = list(next_token_logits.shape)[:-1]
        # predict: 展平的预测 token 序列（含 bonus token，长度 = bs * (spec_steps + 1) + bs）
        predict = torch.zeros(predict_shape, dtype=torch.int32, device=device).flatten()
        # accept_index: [bs, spec_steps + 1]，-1 表示未接受
        accept_index = torch.full(
            (bs, self.spec_steps + 1), -1, dtype=torch.int32, device=device
        )
        num_accepted_drafts = torch.empty((bs,), dtype=torch.int32, device=device)

        # 根据平台和采样策略选择验证方式
        if sampling_info.is_all_greedy or _is_npu or _is_hip:
            # 贪心验证（NPU/AMD 强制使用贪心）
            target_predict = torch.argmax(next_token_logits, dim=-1)
            target_predict = target_predict.reshape(bs, self.draft_token_num)
            predict, accept_index, num_accepted_drafts = verify_tree_greedy_func(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=num_accepted_drafts,  # mutable
                candidates=candidates,
                retrieve_index=self.retrieve_index,
                retrieve_next_token=self.retrieve_next_token,
                retrieve_next_sibling=self.retrieve_next_sibling,
                target_predict=target_predict,
                topk=self.topk,
            )
        else:
            # 采样验证：应用 temperature + top-k + top-p 后执行拒绝采样
            expanded_temperature = torch.repeat_interleave(
                sampling_info.temperatures, self.draft_token_num, dim=0
            )  # (bs * num_draft_tokens, 1)

            target_probs = F.softmax(
                next_token_logits / expanded_temperature, dim=-1
            )  # (bs * num_draft_tokens, vocab_size)
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ks, self.draft_token_num, dim=0
                ),
            )  # (bs * num_draft_tokens, vocab_size)
            target_probs = top_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ps, self.draft_token_num, dim=0
                ),
            )
            target_probs = target_probs.reshape(bs, self.draft_token_num, -1)
            # draft_probs 全零（target_only 模式）
            draft_probs = torch.zeros_like(target_probs)

            # 为拒绝采样和最终采样生成均匀随机数
            coins = torch.rand_like(candidates, dtype=torch.float32, device=device)
            coins_for_final_sampling = torch.rand(
                (bs,), dtype=torch.float32, device=device
            )

            tree_speculative_sampling_target_only(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=num_accepted_drafts,  # mutable
                candidates=candidates,
                # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
                retrive_index=self.retrieve_index,
                retrive_next_token=self.retrieve_next_token,
                retrive_next_sibling=self.retrieve_next_sibling,
                uniform_samples=coins,
                uniform_samples_for_final_sampling=coins_for_final_sampling,
                target_probs=target_probs,
                draft_probs=draft_probs,
                threshold_single=get_global_server_args().speculative_accept_threshold_single,
                threshold_acc=get_global_server_args().speculative_accept_threshold_acc,
                deterministic=True,
            )

            # Sync sampling results across TP ranks: different GPUs may
            # produce slightly different target_probs due to floating-point
            # non-determinism in softmax/top_k/top_p, causing different
            # sampled tokens. Broadcast from rank 0 to ensure consistency.
            # TP 广播：确保所有 rank 使用相同的采样结果（避免浮点误差导致不一致）
            tp_group = (
                get_attention_tp_group()
                if is_dp_attention_enabled()
                else get_tp_group()
            )
            if tp_group.world_size > 1:
                tp_group.broadcast(predict, src=0)
                tp_group.broadcast(accept_index, src=0)
                tp_group.broadcast(num_accepted_drafts, src=0)

        if SIMULATE_ACC_LEN > 0:
            # Do simulation
            # 调试模式：用模拟接受长度替换真实验证结果（用于性能分析）
            accept_index = generate_simulated_accept_index(
                accept_index=accept_index,
                predict=predict,  # mutable
                num_accepted_drafts=num_accepted_drafts,  # mutable
                simulate_acc_len=SIMULATE_ACC_LEN,
                bs=bs,
                spec_steps=self.spec_steps,
            )

        # `num_accepted_drafts` stays drafts-only inside this function; the returned
        # tensor includes the trailing/bonus token via out-of-place +1 so the
        # name no longer flips semantics mid-function (naming doc C2).
        # 返回 num_accepted_drafts + 1（包含 bonus token），对外语义为 num_accepted_tokens
        return predict, num_accepted_drafts + 1, accept_index


@triton.jit
def fill_new_verified_id(
    verified_id,
    accept_lens,
    new_verified_id,
    num_draft_tokens: tl.constexpr,
):
    # NOTE: we cannot fuse any in-place operations of `accept_lens` inside this kernel
    # because this kernel reads accept_lens
    # Triton kernel：从 verified_id 中提取每个请求最后一个被接受 token 的 ID
    # Grid: (bs,) — 每个请求对应一个 program
    pid = tl.program_id(axis=0)
    # accept_lens 包含 bonus token；最后一个被接受的 slot 在 accept_lens - 1 处
    accept_len = tl.load(accept_lens + pid)

    # 计算该请求最后一个被接受 token 在展平 verified_id 中的索引
    verified_id_idx = num_draft_tokens * pid + accept_len - 1
    verified_id_data = tl.load(verified_id + verified_id_idx)
    tl.store(new_verified_id + pid, verified_id_data)


@triton.jit
def fill_accepted_out_cache_loc(
    accept_index,
    out_cache_loc,
    accepted_out_cache_loc,
    size_upper: tl.constexpr,
):
    # Triton kernel：将接受的 token 的 KV slot 位置压缩到 accepted_out_cache_loc
    # 通过前缀和计算每个接受 token 在输出中的目标位置（前缀和 = 之前所有被接受 token 的数量）
    pid = tl.program_id(axis=0)
    offset = tl.arange(0, size_upper)

    # 计算 pid 位置之前有多少个非 -1 的接受 token（即目标写入位置 dst）
    masks = (tl.load(accept_index + offset, offset < pid, other=-1) != -1).to(tl.int64)
    dst = tl.sum(masks)
    src = tl.load(accept_index + pid)
    if src > -1:
        # 若当前位置被接受，从 out_cache_loc[src] 读取 slot 并写入 accepted_out_cache_loc[dst]
        value = tl.load(out_cache_loc + src)
        tl.store(accepted_out_cache_loc + dst, value)


@triton.jit
def assign_extend_cache_locs(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    # Triton kernel：将每个请求 [start_offset, end_offset) 范围内的 KV slot 位置
    # 复制到展平的 out_cache_loc（target verify 阶段的 KV slot 分配）
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    # 计算当前请求在展平 out_cache_loc 中的起始偏移（= 所有前序请求的 token 数之和）
    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0)
    out_offset = tl.sum(end - start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset

    load_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    save_offset = tl.arange(0, BLOCK_SIZE)

    # 分块复制 [kv_start, kv_end) 范围内的 slot 位置
    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = load_offset < kv_end
        data = tl.load(token_pool + load_offset, mask=mask)
        tl.store(out_cache_ptr + save_offset, data, mask=mask)
        load_offset += BLOCK_SIZE
        save_offset += BLOCK_SIZE


def assign_extend_cache_locs_func(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
    device,
) -> torch.Tensor:
    # Python 包装函数：根据平台选择合适的 kernel 分配 extend KV slot 位置
    if _is_cuda or _is_hip or _is_musa:
        out_cache_loc = torch.empty(
            (batch_size * draft_token_num,),
            dtype=torch.int64,
            device=device,
        )
        # 调用 Triton kernel（CUDA/HIP/MUSA 平台）
        assign_extend_cache_locs[(batch_size,)](
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            req_to_token.shape[1],
            next_power_of_2(batch_size),
        )

        return out_cache_loc

    elif _is_npu:
        # NPU 平台：使用 torch.ops.npu.cache_loc_update（int32 输出）
        out_cache_loc = torch.empty(
            (batch_size * draft_token_num,),
            dtype=torch.int32,
            device=device,
        )
        torch.ops.npu.cache_loc_update(
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
        )

        return out_cache_loc
