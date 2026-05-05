# 导入类型提示模块
from typing import Optional, Tuple

# 导入 PyTorch 和 Triton 相关库
import torch
import triton
import triton.language as tl


# 使用 Triton JIT 编译的 GPU Kernel：用于合并前缀（prefix）和后缀（suffix）注意力状态
@triton.jit
def merge_state_kernel(
    output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE] v_merged — 合并后的输出值
    output_lse,  # [NUM_TOKENS, NUM_HEADS] s_merged — 合并后的 log-sum-exp
    prefix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE] v_a — 前缀注意力输出值
    prefix_lse,  # [NUM_TOKENS, NUM_HEADS] s_a — 前缀注意力 log-sum-exp
    suffix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE] v_b — 后缀注意力输出值
    suffix_lse,  # [NUM_TOKENS, NUM_HEADS] s_b — 后缀注意力 log-sum-exp
    HEAD_SIZE: tl.constexpr,        # 每个注意力头的实际维度大小（编译期常量）
    PADDED_HEAD_SIZE: tl.constexpr, # 填充后的头维度大小，为 2 的幂次（便于向量化）
    OUTPUT_LSE: tl.constexpr,       # 是否输出合并后的 log-sum-exp（编译期常量）
):
    # 获取当前 Kernel 处理的 token 索引（第 0 维 program_id）
    token_idx = tl.program_id(0)
    # 获取总 token 数量
    num_tokens = tl.num_programs(0)
    # 获取当前 Kernel 处理的注意力头索引（第 1 维 program_id）
    head_idx = tl.program_id(1)
    # 获取总注意力头数量
    num_heads = tl.num_programs(1)

    # 从全局内存加载前缀部分的 log-sum-exp 值
    p_lse = tl.load(prefix_lse + token_idx * num_heads + head_idx)
    # 从全局内存加载后缀部分的 log-sum-exp 值
    s_lse = tl.load(suffix_lse + token_idx * num_heads + head_idx)
    # 若 lse 为 +inf（表示空序列），则转换为 -inf 以便后续正确合并
    p_lse = float("-inf") if p_lse == float("inf") else p_lse
    s_lse = float("-inf") if s_lse == float("inf") else s_lse

    # 计算两者的最大 lse，用于数值稳定的 log-sum-exp 合并
    max_lse = tl.maximum(p_lse, s_lse)
    # 将各自 lse 减去最大值，防止指数溢出
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    # 合并后的归一化因子：exp(p_lse - max) + exp(s_lse - max)
    out_se = tl.exp(p_lse) + tl.exp(s_lse)

    # 若需要输出 lse，则计算并存储合并后的 log-sum-exp
    if OUTPUT_LSE:
        # 合并后的 lse = log(out_se) + max_lse，还原数值尺度
        out_lse = tl.log(out_se) + max_lse
        # 将合并后的 lse 写回全局内存
        tl.store(output_lse + token_idx * num_heads + head_idx, out_lse)

    # 生成头维度的索引范围（用于向量化加载）
    head_arange = tl.arange(0, PADDED_HEAD_SIZE)
    # 生成掩码，过滤掉填充部分（仅保留有效的 HEAD_SIZE 个元素）
    head_mask = head_arange < HEAD_SIZE
    # 从全局内存加载前缀注意力的输出向量 v_a（按 mask 读取）
    p_out = tl.load(
        prefix_output
        + token_idx * num_heads * HEAD_SIZE
        + head_idx * HEAD_SIZE
        + head_arange,
        mask=head_mask,
    )
    # 从全局内存加载后缀注意力的输出向量 v_b（按 mask 读取）
    s_out = tl.load(
        suffix_output
        + token_idx * num_heads * HEAD_SIZE
        + head_idx * HEAD_SIZE
        + head_arange,
        mask=head_mask,
    )

    # 计算前缀部分的混合权重：exp(p_lse) / out_se
    p_scale = tl.exp(p_lse) / out_se
    # 计算后缀部分的混合权重：exp(s_lse) / out_se
    s_scale = tl.exp(s_lse) / out_se
    # 加权合并：out = v_a * p_scale + v_b * s_scale
    out = p_out * p_scale + s_out * s_scale
    # 将合并后的输出向量写回全局内存（按 mask 写入）
    tl.store(
        output + token_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE + head_arange,
        out,
        mask=head_mask,
    )


# Python 封装函数：调用 merge_state_kernel 合并前缀和后缀注意力的输出状态
def merge_state_triton(
    prefix_output: torch.Tensor,  # 前缀注意力输出 [num_tokens, num_heads, head_size]
    prefix_lse: torch.Tensor,     # 前缀注意力 log-sum-exp [num_tokens, num_heads]
    suffix_output: torch.Tensor,  # 后缀注意力输出 [num_tokens, num_heads, head_size]
    suffix_lse: torch.Tensor,     # 后缀注意力 log-sum-exp [num_tokens, num_heads]
    output: Optional[torch.Tensor] = None,     # 可选：预分配的合并输出张量
    output_lse: Optional[torch.Tensor] = None, # 可选：预分配的合并 lse 张量
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # Avoid creating new tensors if they are already provided
    # 若未提供输出张量，则分配与 prefix_output 形状相同的空张量
    if output is None:
        output = torch.empty_like(prefix_output)
    # 若未提供 lse 输出张量，则分配与 prefix_lse 形状相同的空张量
    if output_lse is None:
        output_lse = torch.empty_like(prefix_lse)

    # 从输出张量中提取形状信息
    num_tokens = output.shape[0]       # token 总数
    num_query_heads = output.shape[1]  # 查询头数量
    head_size = output.shape[2]        # 每个头的实际维度
    # 计算填充后的头维度（向上取最近的 2 的幂，便于 Triton 向量化）
    padded_head_size = triton.next_power_of_2(head_size)

    # 启动 Triton Kernel，grid = (num_tokens, num_query_heads)，每个线程块处理一个 (token, head) 对
    merge_state_kernel[(num_tokens, num_query_heads)](
        output,
        output_lse,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        head_size,
        padded_head_size,
        output_lse is not None,  # 是否需要输出 lse
    )
    # 返回合并后的输出和 lse
    return output, output_lse
