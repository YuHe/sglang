# 类型提示：Callable 用于描述工厂函数，List/Optional 用于容器和可选参数
from typing import TYPE_CHECKING, Callable, List, Optional

import torch

# 导入 TBO（Two-Batch-Overlap）批次重叠调度工具
from sglang.srt.batch_overlap import two_batch_overlap
# 导入注意力后端基类
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
# 导入推测解码输入信息
from sglang.srt.speculative.spec_info import SpecInput

if TYPE_CHECKING:
    # 仅类型检查时导入，避免循环依赖
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


class TboAttnBackend(AttentionBackend):
    # TBO（Two-Batch-Overlap）注意力后端：将批次拆分为两个子批次并发执行，以隐藏内存带宽延迟
    def __init__(self, primary: AttentionBackend, children: List[AttentionBackend]):
        super().__init__()
        self.primary = primary    # 主后端，处理完整批次的前向计算
        self.children = children  # 子后端列表（通常为 2 个），用于 TBO 并发计算

    @classmethod
    def init_new(cls, creator: Callable[[], AttentionBackend]):
        # 工厂方法：通过 creator 函数创建 1 个主后端和 2 个子后端
        return cls(
            primary=creator(),
            children=[creator() for _ in range(2)],  # 创建 2 个子后端
        )

    def init_forward_metadata(self, forward_batch: "ForwardBatch"):
        # 初始化主批次和所有子批次的前向元数据
        self.primary.init_forward_metadata(forward_batch=forward_batch)
        if forward_batch.tbo_children is not None:
            # 若存在 TBO 子批次，逐个初始化对应子后端的元数据
            for child, forward_batch_child in zip(
                self.children, forward_batch.tbo_children, strict=True
            ):
                if forward_batch_child.batch_size > 0:
                    # 仅对非空子批次进行初始化
                    child.init_forward_metadata(forward_batch=forward_batch_child)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        # 初始化主后端和所有子后端的 CUDA Graph 全局状态
        self.primary.init_cuda_graph_state(max_bs=max_bs, max_num_tokens=max_num_tokens)
        for item in self.children:
            # TODO for children, maybe can provide *smaller* max_bs to optimize
            # 子后端目前使用相同的 max_bs，未来可优化为更小的值以节省内存
            item.init_cuda_graph_state(max_bs=max_bs, max_num_tokens=max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: "ForwardMode",
        spec_info: Optional[SpecInput],
    ):
        # CUDA Graph 捕获阶段：主后端先捕获，再分别捕获两个子后端
        self.primary.init_forward_metadata_capture_cuda_graph(
            bs=bs,
            num_tokens=num_tokens,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
        )

        # 为两个子后端分别计算分割后的参数并捕获 CUDA Graph
        self._init_forward_metadata_cuda_graph_children(
            fn_name="init_forward_metadata_capture_cuda_graph",
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
            capture_num_tokens=num_tokens,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: "ForwardMode",
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        # CUDA Graph 回放阶段：主后端先回放，再分别回放两个子后端
        self.primary.init_forward_metadata_replay_cuda_graph(
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_sum=seq_lens_sum,
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
            seq_lens_cpu=seq_lens_cpu,
        )

        # 为两个子后端分别计算分割后的参数并回放 CUDA Graph
        self._init_forward_metadata_cuda_graph_children(
            fn_name="init_forward_metadata_replay_cuda_graph",
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
            replay_seq_lens_sum=seq_lens_sum,
            replay_seq_lens_cpu=seq_lens_cpu,
        )

    def _init_forward_metadata_cuda_graph_children(
        self,
        fn_name: str,
        # common args
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: "ForwardMode",
        spec_info: Optional[SpecInput],
        # capture args
        capture_num_tokens: int = None,
        # replay args
        replay_seq_lens_sum: int = None,
        replay_seq_lens_cpu: Optional[torch.Tensor] = None,
    ):
        # 内部方法：计算批次分割点，并分别为左右子后端调用对应的 CUDA Graph 初始化函数
        token_num_per_seq = two_batch_overlap.get_token_num_per_seq(
            forward_mode=forward_mode, spec_info=spec_info
        )
        if fn_name == "init_forward_metadata_capture_cuda_graph":
            # 捕获时要求 num_tokens 必须等于 bs * token_num_per_seq
            assert (
                capture_num_tokens == bs * token_num_per_seq
            ), "For target-verify or decode mode, num_tokens should be equal to token_num_per_seq * bs"
        num_tokens = bs * token_num_per_seq  # 总 token 数

        # 计算 CUDA Graph 回放时的序列/token 维度分割索引
        tbo_split_seq_index, tbo_split_token_index = (
            two_batch_overlap.compute_split_indices_for_cuda_graph_replay(
                forward_mode=forward_mode,
                cuda_graph_num_tokens=num_tokens,
                spec_info=spec_info,
            )
        )

        # 根据分割点计算左右子批次的 token 数和序列数
        num_tokens_child_left = tbo_split_token_index
        num_tokens_child_right = num_tokens - tbo_split_token_index
        bs_child_left = tbo_split_seq_index
        bs_child_right = bs - bs_child_left

        # 确保左右两侧均有非零的 token 数（否则无法构成有效的 TBO 分割）
        assert (
            num_tokens_child_left > 0 and num_tokens_child_right > 0
        ), f"{num_tokens_child_left=} {num_tokens_child_right=} {forward_mode=} {num_tokens=}"

        # 构建分割前的公共参数字典，供左右分割函数复用
        common_pre_split_args = dict(
            fn_name=fn_name,
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
            capture_num_tokens=capture_num_tokens,
            replay_seq_lens_sum=replay_seq_lens_sum,
            replay_seq_lens_cpu=replay_seq_lens_cpu,
        )

        # 分别构建左、右子批次的参数（按序列维度切片）
        args_left = _init_forward_metadata_cuda_graph_split(
            output_bs=bs_child_left,
            seq_slice=slice(None, tbo_split_seq_index),
            **common_pre_split_args,
        )
        args_right = _init_forward_metadata_cuda_graph_split(
            output_bs=bs_child_right,
            seq_slice=slice(tbo_split_seq_index, None),
            **common_pre_split_args,
        )

        # 对左右子后端分别调用对应的 CUDA Graph 初始化函数
        child_left, child_right = self.children
        getattr(child_left, fn_name)(**args_left)
        getattr(child_right, fn_name)(**args_right)

    def get_cuda_graph_seq_len_fill_value(self):
        # 获取填充值，并断言所有子后端与主后端保持一致
        ans = self.primary.get_cuda_graph_seq_len_fill_value()
        for child in self.children:
            assert ans == child.get_cuda_graph_seq_len_fill_value()
        return ans

    def forward(self, *args, **kwargs):
        # 前向传播委托给主后端（子后端在 TBO 并发调度中由外部驱动）
        return self.primary.forward(*args, **kwargs)

    def forward_extend(self, *args, **kwargs):
        # extend 阶段委托给主后端
        return self.primary.forward_extend(*args, **kwargs)

    def forward_decode(self, *args, **kwargs):
        # decode 阶段委托给主后端
        return self.primary.forward_decode(*args, **kwargs)

    def get_indexer_metadata(self, layer_id: int, forward_batch: "ForwardBatch"):
        # 获取 NSA 索引器元数据，委托给主后端
        return self.primary.get_indexer_metadata(layer_id, forward_batch)


def _init_forward_metadata_cuda_graph_split(
    fn_name: str,
    seq_slice: slice,     # 当前子批次在完整批次中的序列切片范围
    output_bs: int,       # 子批次的序列数
    # common args
    bs: int,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    encoder_lens: Optional[torch.Tensor],
    forward_mode: "ForwardMode",
    spec_info: Optional[SpecInput],
    # capture args
    capture_num_tokens: int = None,
    # replay args
    replay_seq_lens_sum: int = None,
    replay_seq_lens_cpu: Optional[torch.Tensor] = None,
):
    # 根据序列切片，构建子批次 CUDA Graph 初始化所需的参数字典
    token_num_per_seq = two_batch_overlap.get_token_num_per_seq(
        forward_mode=forward_mode, spec_info=spec_info
    )
    # 目前不支持 encoder_lens 的分割
    assert encoder_lens is None, "encoder_lens is not supported yet"
    if spec_info is not None:
        # 若存在推测解码信息，按序列/token 范围分割 spec_info
        output_spec_info = two_batch_overlap.split_spec_info(
            spec_info=spec_info,
            start_seq_index=seq_slice.start if seq_slice.start is not None else 0,
            end_seq_index=seq_slice.stop if seq_slice.stop is not None else bs,
            start_token_index=(
                seq_slice.start * token_num_per_seq
                if seq_slice.start is not None
                else 0
            ),
            end_token_index=(
                seq_slice.stop * token_num_per_seq
                if seq_slice.stop is not None
                else bs * token_num_per_seq
            ),
        )

    else:
        output_spec_info = None  # 无推测解码信息时置为 None
    # 构建基础参数字典（按序列切片提取对应的 req_pool_indices 和 seq_lens）
    ans = dict(
        bs=output_bs,
        req_pool_indices=req_pool_indices[seq_slice],
        seq_lens=seq_lens[seq_slice],
        # directly forward
        forward_mode=forward_mode,
        # ignore
        encoder_lens=None,
        spec_info=output_spec_info,
    )

    if fn_name == "init_forward_metadata_capture_cuda_graph":
        # 捕获阶段：验证并补充 num_tokens 参数
        assert (
            capture_num_tokens == bs * token_num_per_seq
        ), "Only support num_tokens==bs * token_num_per_seq for target-verify or decode mode"
        ans.update(
            dict(
                num_tokens=output_bs * token_num_per_seq,  # 子批次 token 总数
            )
        )
    elif fn_name == "init_forward_metadata_replay_cuda_graph":
        # 回放阶段：从 CPU 序列长度中提取子批次切片，并计算其 sum
        output_seq_lens_cpu = replay_seq_lens_cpu[seq_slice]
        ans.update(
            dict(
                seq_lens_sum=output_seq_lens_cpu.sum().item(),  # 子批次序列长度之和
                seq_lens_cpu=output_seq_lens_cpu,
            )
        )
    else:
        raise NotImplementedError

    return ans
