"""FlashInfer-based kernels for GDN (Gated Delta Network) linear attention.

Both SM90 and SM100+ use the same pool layout: [pool, HV, V, K] (K-last).

SM90 (Hopper): full support — decode, prefill, MTP.  State dtype: fp32.
SM100+ (Blackwell+): decode-only with bf16 state.  More support on the way.

Requires flashinfer >= 0.6.4 (SM90) or >= 0.6.5 (SM100+).
"""

# 导入日志模块
import logging
# 导入 os，用于设置环境变量（禁用 FlashInfer 版本检查）
import os
# Optional 类型注解，用于延迟初始化的全局变量
from typing import Optional

# 导入 PyTorch，用于 Tensor 操作和 CUDA 设备检测
import torch

# 导入线性注意力 Kernel 抽象基类
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import for FlashInfer GDN kernels
# ---------------------------------------------------------------------------
# 延迟导入 FlashInfer GDN Kernel：避免启动时强制依赖 FlashInfer 库
_flashinfer_gdn_available: Optional[bool] = None        # 是否可用（None=未检测）
_flashinfer_chunk_gated_delta_rule = None                # prefill 分块 GDN 函数
_flashinfer_gated_delta_rule_mtp = None                  # MTP（multi-token prediction）验证函数
_flashinfer_gated_delta_rule_decode = None               # decode 单步更新函数


def _get_flashinfer_gdn_kernels():
    """Lazy import for FlashInfer GDN prefill, decode and verify (MTP) kernels.

    Returns (available, prefill_fn, mtp_fn, decode_fn).
    """
    global _flashinfer_gdn_available, _flashinfer_chunk_gated_delta_rule, _flashinfer_gated_delta_rule_mtp, _flashinfer_gated_delta_rule_decode
    if _flashinfer_gdn_available is None:
        try:
            # 禁用 FlashInfer 版本检查（避免版本不匹配导致的运行时错误）
            os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

            # 导入 FlashInfer 的 GDN decode 和 prefill Kernel
            from flashinfer.gdn_decode import (
                gated_delta_rule_decode_pretranspose,  # decode 阶段：预转置输入的 GDN Kernel
                gated_delta_rule_mtp,                  # MTP 验证（multi-token prediction）Kernel
            )
            from flashinfer.gdn_prefill import chunk_gated_delta_rule  # prefill 分块 Kernel

            _flashinfer_chunk_gated_delta_rule = chunk_gated_delta_rule
            _flashinfer_gated_delta_rule_mtp = gated_delta_rule_mtp
            _flashinfer_gated_delta_rule_decode = gated_delta_rule_decode_pretranspose
            # 仅 SM90+ 且 CUDA 可用时标记为可用（SM90=Hopper, SM100=Blackwell）
            _flashinfer_gdn_available = (
                torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9
            )
            if _flashinfer_gdn_available:
                logger.info("FlashInfer GDN kernels loaded successfully")
        except (ImportError, RuntimeError) as e:
            # FlashInfer 未安装或不支持当前平台时静默降级
            logger.warning(f"FlashInfer GDN kernels not available: {e}")
            _flashinfer_gdn_available = False
            _flashinfer_gated_delta_rule_decode = None
    return (
        _flashinfer_gdn_available,
        _flashinfer_chunk_gated_delta_rule,
        _flashinfer_gated_delta_rule_mtp,
        _flashinfer_gated_delta_rule_decode,
    )


# ---------------------------------------------------------------------------
# Kernel implementation
# ---------------------------------------------------------------------------


# FlashInfer 实现的 GDN 线性注意力 Kernel（支持 SM90 和 SM100+）
class FlashInferGDNKernel(LinearAttnKernelBase):
    """FlashInfer kernel for GDN with K-last SSM state layout.

    SM90 (Hopper): decode uses gather/scatter; prefill and MTP verify supported.
    SM100+ (Blackwell+): decode uses pool API (initial_state_indices); prefill
    and MTP verify are not supported (use Triton backend for those).

    Requires flashinfer >= 0.6.4 (SM90) or >= 0.6.5 (SM100+).
    """

    def __init__(self):
        # 延迟加载 FlashInfer GDN Kernel 并缓存函数指针
        (
            available,
            self._prefill_fn,   # prefill Kernel 函数
            self._mtp_fn,       # MTP 验证 Kernel 函数
            self._decode_fn,    # decode Kernel 函数
        ) = _get_flashinfer_gdn_kernels()

        # 若 FlashInfer 不可用，抛出运行时错误
        if not available:
            raise RuntimeError(
                "FlashInfer GDN kernels are not available. "
                "Requires SM90+ and FlashInfer with GDN kernel support."
            )
        if self._decode_fn is None:
            raise RuntimeError("FlashInfer GDN decode kernel is unavailable.")

        # 获取 GPU 计算能力主版本号，判断是否需要使用 state pool API
        sm_major = torch.cuda.get_device_capability()[0]
        # SM100+（Blackwell+）使用 state pool API（initial_state_indices），SM90（Hopper）使用 gather/scatter
        self.use_state_pool = sm_major != 9

        # SM90 需要 prefill 和 MTP 验证 Kernel
        if sm_major == 9:
            if self._prefill_fn is None:
                raise RuntimeError("FlashInfer GDN prefill kernel is unavailable.")
            if self._mtp_fn is None:
                raise RuntimeError("FlashInfer GDN MTP (verify) kernel is unavailable.")

        logger.info("Using FlashInfer GDN kernels")

    # ---- decode ----

    def decode(
        self,
        q: torch.Tensor,        # Query 张量（decode 阶段）
        k: torch.Tensor,        # Key 张量
        v: torch.Tensor,        # Value 张量
        a: torch.Tensor,        # 输入门控因子 a
        b: torch.Tensor,        # 遗忘门控因子 b
        *,
        A_log: torch.Tensor,         # 对数衰减参数 A_log
        dt_bias: torch.Tensor,       # 时间步长偏置
        ssm_states: torch.Tensor,    # SSM 隐状态池 [pool, HV, V, K]（K-last 布局）
        cache_indices: torch.Tensor, # 每个请求对应的状态槽位索引
        query_start_loc: torch.Tensor,  # Query 累积起始位置（CSR 格式）
        **kwargs,
    ) -> torch.Tensor:
        # 从张量形状中提取维度信息
        batch_size = cache_indices.shape[0]
        num_heads = q.shape[2]
        head_k_dim = q.shape[3]
        num_v_heads = v.shape[2]
        head_v_dim = v.shape[3]

        # 将输入张量 reshape 为 FlashInfer 要求的 [B, T, H, D] 格式
        query_fi = q.view(batch_size, 1, num_heads, head_k_dim)
        key_fi = k.view(batch_size, 1, num_heads, head_k_dim)
        value_fi = v.view(batch_size, 1, num_v_heads, head_v_dim)
        a_fi = a.view(batch_size, 1, num_v_heads)  # 输入门控 reshape
        b_fi = b.view(batch_size, 1, num_v_heads)  # 遗忘门控 reshape

        if self.use_state_pool:
            # SM100+ 路径：使用 state pool API（通过 initial_state_indices 直接索引）
            output_fi, _ = self._decode_fn(
                q=query_fi,
                k=key_fi,
                v=value_fi,
                state=None,                          # 无独立 state 张量（由 pool 管理）
                A_log=A_log.detach().float(),         # 转为 float32 确保精度
                a=a_fi,
                dt_bias=dt_bias.detach(),
                b=b_fi,
                use_qk_l2norm=True,                  # QK L2 归一化
                initial_state=ssm_states,             # 完整状态池
                initial_state_indices=cache_indices,  # 按索引访问状态
            )
        else:
            # TODO: Once FlashInfer PR#2521 is merged for SM90, gather/scatter
            # will no longer be needed here.
            # SM90 路径：需要先从状态池中 gather 出对应状态（后续可优化为池 API）
            state_batch = ssm_states[cache_indices]  # 按索引收集状态 [B, HV, V, K]
            output_fi, new_state = self._decode_fn(
                q=query_fi,
                k=key_fi,
                v=value_fi,
                state=state_batch,          # 独立 state 张量（SM90 模式）
                A_log=A_log.detach(),
                a=a_fi,
                dt_bias=dt_bias.detach(),
                b=b_fi,
                scale=None,
                output=None,
                use_qk_l2norm=True,
            )
            # 将更新后的状态写回状态池（scatter 操作）
            ssm_states[cache_indices] = new_state

        # 输出 reshape：[B, 1, HV, V] -> [1, B, HV, V]（匹配 decode 路径标准输出布局）
        return output_fi.view(1, batch_size, num_v_heads, head_v_dim)

    # ---- extend (prefill) ----

    def extend(
        self,
        q: torch.Tensor,    # Query 张量（prefill 阶段）
        k: torch.Tensor,    # Key 张量
        v: torch.Tensor,    # Value 张量
        g: torch.Tensor,    # 输入门控（对应 alpha，exp(g) 后使用）
        beta: torch.Tensor, # 遗忘门控 beta
        *,
        ssm_states: torch.Tensor,    # SSM 状态池
        cache_indices: torch.Tensor, # 状态槽位索引
        query_start_loc: torch.Tensor,  # 累积序列长度（CSR 格式）
        **kwargs,
    ) -> tuple:
        # SM100+ 不支持 prefill（FlashInfer 限制），需回退到 Triton 后端
        if self.use_state_pool:
            raise NotImplementedError(
                "FlashInfer GDN prefill is not supported on SM100+. "
                "Use --linear-attn-prefill-backend triton."
            )

        # SM90: chunked prefill using FlashInfer GDN prefill kernel.
        # SM90（Hopper）：使用 FlashInfer GDN prefill Kernel 进行分块前向计算
        from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd

        total_seq_len = q.shape[1]
        num_v_heads = v.shape[2]
        head_v_dim = v.shape[3]

        # L2 归一化 Q/K（FlashInfer 要求在外部归一化，use_qk_l2norm_in_kernel=False）
        q_fi = l2norm_fwd(q[0].contiguous())
        k_fi = l2norm_fwd(k[0].contiguous())
        v_fi = v[0].contiguous()

        # g (alpha) and beta: [1, seq, HV] -> [seq, HV], float32 for FlashInfer
        # alpha = exp(g)，将对数门控转换为实际门控值（float32 保证精度）
        alpha_fi = torch.exp(g[0].to(torch.float32))
        beta_fi = beta[0].to(torch.float32)

        # 累积序列长度转换为 int64（FlashInfer API 要求）
        cu_seqlens_fi = query_start_loc.to(torch.int64)

        # Remap negative padding indices to sentinel slot
        # 将负数填充索引（无效请求）映射到状态池最后一个槽位（哨兵槽位）
        ssm_cache_indices = torch.where(
            cache_indices >= 0,
            cache_indices,
            ssm_states.shape[0] - 1,  # 最后一个槽位作为哨兵（padding 请求）
        ).to(torch.int64)

        # FlashInfer requires float32 initial state, K-last layout [B, HV, V, K]
        # 从状态池中提取初始状态并转换为 float32（FlashInfer 精度要求）
        initial_state_fi = ssm_states[ssm_cache_indices].to(torch.float32)

        # 调用 FlashInfer GDN prefill Kernel（输出注意力结果和更新后的状态）
        output_fi, output_state_fi = self._prefill_fn(
            q=q_fi,
            k=k_fi,
            v=v_fi,
            g=alpha_fi,                      # 输入门控
            beta=beta_fi,                    # 遗忘门控
            scale=None,                      # 缩放因子（None=使用默认）
            initial_state=initial_state_fi,  # 初始 KV 状态
            output_final_state=True,         # 输出最终状态（用于后续 decode）
            cu_seqlens=cu_seqlens_fi,        # 累积序列长度
            use_qk_l2norm_in_kernel=False,   # 已在外部归一化，不在 Kernel 内重复归一化
        )

        # Write back state to pool
        # 将更新后的状态写回状态池（scatter 操作，仅写有效请求的槽位）
        ssm_states.index_copy_(
            0,
            ssm_cache_indices,
            output_state_fi.to(ssm_states.dtype),  # 转换回状态池的数据类型
        )

        # Output: [seq, HV, V] -> [1, seq, HV, V]（添加 batch 维度）
        core_attn_out = output_fi.view(1, total_seq_len, num_v_heads, head_v_dim)

        # Return (output, last_recurrent_state, h) to match Triton kernel interface.
        # h=None since FlashInfer doesn't provide intermediate states.
        # 返回三元组以匹配 Triton Kernel 接口（中间状态 h 不可用，返回 None）
        return core_attn_out, None, None

    # ---- target_verify (MTP) ----

    def target_verify(
        self,
        A_log: torch.Tensor,      # 对数衰减参数
        dt_bias: torch.Tensor,    # 时间步长偏置
        q: torch.Tensor,          # Query 张量（待验证的草稿 token）
        k: torch.Tensor,          # Key 张量
        v: torch.Tensor,          # Value 张量
        a: torch.Tensor,          # 输入门控因子 a
        b: torch.Tensor,          # 遗忘门控因子 b
        *,
        ssm_states: torch.Tensor,    # SSM 状态池
        cache_indices: torch.Tensor, # 状态槽位索引
        query_start_loc: torch.Tensor,          # Query 累积起始位置
        intermediate_states_buffer: torch.Tensor,  # 中间状态缓冲区（并行验证用）
        intermediate_state_indices: torch.Tensor,  # 中间状态索引
        cache_steps: int,                          # 投机解码步数（草稿 token 数）
        retrieve_parent_token: torch.Tensor,       # 父 token 索引（树注意力）
        **kwargs,
    ) -> torch.Tensor:
        # SM100+ 不支持 MTP 验证，需回退到 Triton 后端
        if self.use_state_pool:
            raise NotImplementedError(
                "FlashInfer GDN MTP verify is not yet supported on SM100+."
            )

        # SM90: MTP verify using FlashInfer gated_delta_rule_mtp kernel.
        # FlashInfer MTP Kernel 仅支持 topk=1（retrieve_parent_token 必须为 None）
        if retrieve_parent_token is not None:
            raise RuntimeError(
                "FlashInfer GDN verify kernel only supports topk=1 "
                "(retrieve_parent_token must be None)."
            )

        # 计算 MTP 验证的张量形状
        seq_len = q.shape[1]
        batch_size = query_start_loc.shape[0] - 1
        draft_token_num = seq_len // batch_size  # 每个请求的草稿 token 数

        num_heads = q.shape[2]
        head_k_dim = q.shape[3]
        num_v_heads = v.shape[2]
        head_v_dim = v.shape[3]

        # 将张量 reshape 为 FlashInfer MTP Kernel 要求的 [B, draft_num, H, D] 格式
        query_mtp = q.view(batch_size, draft_token_num, num_heads, head_k_dim)
        key_mtp = k.view(batch_size, draft_token_num, num_heads, head_k_dim)
        value_mtp = v.view(batch_size, draft_token_num, num_v_heads, head_v_dim)

        # MTP 模式需要 a, b, A_log, dt_bias 全部提供
        if a is None or b is None or A_log is None or dt_bias is None:
            raise RuntimeError(
                "FlashInfer GDN MTP kernel requires a, b, A_log, dt_bias."
            )

        # 门控因子 reshape：[1, seq, HV] -> [B, draft_num, HV]
        a_mtp = a.view(batch_size, draft_token_num, num_v_heads)
        b_mtp = b.view(batch_size, draft_token_num, num_v_heads)

        # 调用 FlashInfer MTP 验证 Kernel（并行验证所有草稿 token）
        output_fi, _ = self._mtp_fn(
            q=query_mtp,
            k=key_mtp,
            v=value_mtp,
            initial_state=ssm_states,                        # 状态池
            initial_state_indices=cache_indices,              # 状态索引
            A_log=A_log.detach(),
            a=a_mtp,
            dt_bias=dt_bias.detach(),
            b=b_mtp,
            scale=None,
            output=None,
            intermediate_states_buffer=intermediate_states_buffer,  # 中间状态缓冲区
            disable_state_update=True,                        # 禁止主状态更新（验证模式）
            use_qk_l2norm=True,
        )

        # 输出 reshape：[B, draft_num, HV, V] -> [1, seq, HV, V]（统一输出布局）
        return output_fi.view(1, seq_len, num_v_heads, head_v_dim)
