import logging
from typing import TYPE_CHECKING

# 导入线性注意力模型注册表工具（用于混合 GDN 等模型）
from sglang.srt.configs.linear_attn_model_registry import (
    get_linear_attn_config,
    import_backend_class,
)
# 导入设备能力查询和 MUSA 设备检测工具
from sglang.srt.utils import get_device_capability, is_musa

# 一次性检测是否为 MUSA 设备（摩尔线程 GPU）
_is_musa = is_musa()

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    # evade circular imports
    # 仅类型检查时导入，避免运行时循环引用
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.model_executor.model_runner import ModelRunner

# 全局注意力后端注册表：name -> 工厂函数
ATTENTION_BACKENDS = {}


def register_attention_backend(name):
    # 装饰器工厂：将后端工厂函数注册到 ATTENTION_BACKENDS 字典中
    def decorator(fn):
        ATTENTION_BACKENDS[name] = fn
        return fn

    return decorator


@register_attention_backend("flashinfer")
def create_flashinfer_backend(runner):
    # 创建 FlashInfer 注意力后端：支持标准注意力和 MLA（多头潜在注意力）两种模式
    import torch

    if not runner.use_mla_backend:
        from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

        # Init streams
        # 为 EAGLE 推测解码初始化专用 CUDA 流，避免与主流竞争
        if runner.server_args.speculative_algorithm == "EAGLE":
            if (
                not hasattr(runner, "plan_stream_for_flashinfer")
                or not runner.plan_stream_for_flashinfer
            ):
                runner.plan_stream_for_flashinfer = torch.cuda.Stream()
        return FlashInferAttnBackend(
            runner, init_new_workspace=runner.init_new_workspace
        )
    else:
        # MLA 模式使用专用的 FlashInfer MLA 后端
        from sglang.srt.layers.attention.flashinfer_mla_backend import (
            FlashInferMLAAttnBackend,
        )

        return FlashInferMLAAttnBackend(runner)


@register_attention_backend("trtllm_mla")
def create_trtllm_mla_backend(runner):
    # 创建 TRT-LLM MLA 后端（仅适用于 MLA 模型）
    if not runner.use_mla_backend:
        raise ValueError("trtllm_mla backend can only be used with MLA models.")
    from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend

    return TRTLLMMLABackend(runner)


@register_attention_backend("aiter")
def create_aiter_backend(runner):
    # 创建 Aiter 注意力后端（AMD ROCm 优化后端）
    from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

    return AiterAttnBackend(runner)


@register_attention_backend("wave")
def create_wave_backend(runner):
    # 创建 Wave 注意力后端（持续批处理优化后端）
    from sglang.srt.layers.attention.wave_backend import WaveAttnBackend

    return WaveAttnBackend(runner)


@register_attention_backend("ascend")
def create_ascend_backend(runner):
    # 创建昇腾 NPU 注意力后端
    from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
        AscendAttnBackend,
    )

    return AscendAttnBackend(runner)


@register_attention_backend("nsa")
def create_nsa_backend(runner):
    # 创建 NSA（Native Sparse Attention）稀疏注意力后端
    from sglang.srt.layers.attention.nsa_backend import NativeSparseAttnBackend

    return NativeSparseAttnBackend(runner)


@register_attention_backend("triton")
def create_triton_backend(runner):
    # 创建基于 Triton 内核的注意力后端（不支持编码器-解码器交叉注意力）
    assert not runner.model_config.is_encoder_decoder, (
        "Cross attention is not supported in the triton attention backend. "
        "Please use `--attention-backend flashinfer`."
    )
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

    return TritonAttnBackend(runner)


@register_attention_backend("torch_native")
def create_torch_native_backend(runner):
    # 创建基于 PyTorch 原生 SDPA 的注意力后端
    from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend

    return TorchNativeAttnBackend(runner)


@register_attention_backend("flex_attention")
def create_flex_attention_backend(runner):
    # 创建基于 PyTorch Flex Attention 的注意力后端（支持自定义块掩码）
    from sglang.srt.layers.attention.torch_flex_backend import TorchFlexAttnBackend

    return TorchFlexAttnBackend(runner)


@register_attention_backend("flashmla")
def create_flashmla_backend(runner):
    # 创建 FlashMLA 注意力后端（专为 MLA 结构优化的 Flash 内核）
    from sglang.srt.layers.attention.flashmla_backend import FlashMLABackend

    return FlashMLABackend(runner)


@register_attention_backend("fa3")
def create_flashattention_v3_backend(runner):
    # 创建 FlashAttention v3 后端（需要 SM80+ 或 MUSA MP31+ 硬件支持）

    major, minor = get_device_capability()
    if not _is_musa:
        # NVIDIA GPU：要求 SM80（A100）或 SM90（H100）
        assert (major == 8 and not runner.use_mla_backend) or major == 9, (
            "FlashAttention v3 Backend requires SM>=80 and SM<=90. "
            "Please use `--attention-backend flashinfer`."
        )
        from sglang.srt.layers.attention.flashattention_backend import (
            FlashAttentionBackend,
        )

        return FlashAttentionBackend(runner)
    else:
        # MUSA GPU：要求 MP31+
        assert major == 3 and minor >= 1, (
            "FlashAttention v3 Backend requires MP>=31. "
            "Please use `--attention-backend triton`."
        )
        from sglang.srt.hardware_backend.musa.attention import (
            MusaFlashAttentionBackend,
        )

        return MusaFlashAttentionBackend(runner)


@register_attention_backend("fa4")
def create_flashattention_v4_backend(runner):
    # 创建 FlashAttention v4 后端（指定使用第 4 版实现）
    from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend

    return FlashAttentionBackend(runner, fa_impl_ver=4)


@register_attention_backend("cutlass_mla")
def create_cutlass_mla_backend(runner):
    # 创建 Cutlass MLA 后端（decode 阶段使用 Cutlass 高效矩阵运算内核）
    from sglang.srt.layers.attention.cutlass_mla_backend import CutlassMLABackend

    return CutlassMLABackend(runner)


@register_attention_backend("trtllm_mha")
def create_trtllm_mha_backend(runner):
    # 创建 TRT-LLM 多头注意力后端（仅适用于非 MLA 模型）
    if runner.use_mla_backend:
        raise ValueError("trtllm_mha backend can only be used with non-MLA models.")
    from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

    return TRTLLMHAAttnBackend(runner)


@register_attention_backend("intel_amx")
def create_intel_amx_backend(runner):
    # 创建 Intel AMX 注意力后端（用于 Intel CPU 上的高效矩阵运算）
    from sglang.srt.layers.attention.intel_amx_backend import IntelAMXAttnBackend

    return IntelAMXAttnBackend(runner)


@register_attention_backend("dual_chunk_flash_attn")
def create_dual_chunk_flash_attn_backend(runner):
    # 创建双块 Flash Attention 后端（适合超长序列分块注意力计算）
    from sglang.srt.layers.attention.dual_chunk_flashattention_backend import (
        DualChunkFlashAttentionBackend,
    )

    return DualChunkFlashAttentionBackend(runner)


def attn_backend_wrapper(runner: "ModelRunner", full_attn_backend: "AttentionBackend"):
    """
    Wrapper for special models like hybrid GDN, so we don't
    need to change the code of the original attention backend.
    """
    # 为混合线性注意力模型（如 Mamba/GDN 混合架构）包装注意力后端
    # 将全局注意力后端与线性注意力后端组合为 HybridLinearAttnBackend
    assert not (
        runner.hybrid_gdn_config is not None and runner.use_mla_backend
    ), "hybrid_gdn can only be used with non-MLA models."

    if cfg := runner.mambaish_config:
        # 检测并导入线性注意力相关环境依赖
        from sglang.srt.layers.attention.fla.utils import check_environments
        from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
        from sglang.srt.layers.attention.linear.lightning_backend import (
            LightningAttentionBackend,
        )
        from sglang.srt.layers.attention.linear.utils import (
            initialize_linear_attn_config,
        )
        from sglang.srt.utils import is_blackwell, is_npu

        if not is_npu():
            # CUDA 路径：导入 HybridLinearAttnBackend、Mamba2 和 GDN 后端
            from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
                HybridLinearAttnBackend,
                Mamba2AttnBackend,
            )
            from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
        else:
            # NPU 路径：使用昇腾专用的混合线性注意力后端
            from sglang.srt.hardware_backend.npu.attention.ascend_gdn_backend import (
                AscendGDNAttnBackend as GDNAttnBackend,
            )
            from sglang.srt.hardware_backend.npu.attention.ascend_hybrid_linear_attn_backend import (
                AscendHybridLinearAttnBackend as HybridLinearAttnBackend,
            )
            from sglang.srt.hardware_backend.npu.attention.ascend_hybrid_linear_attn_backend import (
                AscendMamba2AttnBackend as Mamba2AttnBackend,
            )

        # 初始化线性注意力环境和配置
        check_environments()
        initialize_linear_attn_config(runner.server_args)
        if runner.hybrid_gdn_config is not None:
            # GDN 混合模型：检查 Blackwell/NPU 的后端兼容性
            if is_blackwell():
                assert (
                    runner.server_args.attention_backend == "triton"
                    or runner.server_args.attention_backend == "trtllm_mha"
                    or runner.server_args.attention_backend == "fa4"
                    or runner.server_args.attention_backend == "flashinfer"
                ), "triton, trtllm_mha, fa4, or flashinfer backend are the only supported backends on Blackwell GPUs for hybrid GDN models, use --attention-backend to specify the backend."
            if is_npu():
                assert (
                    runner.server_args.attention_backend == "ascend"
                ), "ascend backend is the only supported backend on NPU for hybrid GDN models, use --attention-backend ascend to specify the backend."
            logger.info(f"Using hybrid linear attention backend for hybrid GDN models.")
            linear_attn_backend = GDNAttnBackend(runner)
        elif runner.mamba2_config is not None:
            # Mamba2 混合模型：使用 Mamba2 线性注意力后端
            linear_attn_backend = Mamba2AttnBackend(runner)
        elif runner.kimi_linear_config is not None:
            # Kimi 线性注意力模型：使用 KDA 后端
            linear_attn_backend = KDAAttnBackend(runner)
        elif runner.hybrid_lightning_config is not None:
            # Lightning Attention 混合模型：使用 LightningAttention 后端
            linear_attn_backend = LightningAttentionBackend(runner)
        else:
            # 通用路径：从模型注册表查找对应的线性注意力后端
            spec_result = get_linear_attn_config(runner.model_config.hf_config)
            if spec_result is not None:
                spec, _ = spec_result
                BackendClass = import_backend_class(spec.backend_class_name)
                linear_attn_backend = BackendClass(runner)
            else:
                raise ValueError(
                    "Expected hybrid GDN or NemotronH models, but got unknown model. "
                    "If this is a custom hybrid model, use register_linear_attn_model() "
                    "from sglang.srt.configs.linear_attn_model_registry."
                )
        # 获取模型中使用全注意力的层 ID 列表
        full_attn_layers = cfg.full_attention_layer_ids
        # 组合全注意力后端与线性注意力后端，返回混合后端
        return HybridLinearAttnBackend(
            full_attn_backend, linear_attn_backend, full_attn_layers
        )

    # 非混合模型：直接返回原始全注意力后端
    return full_attn_backend


@register_attention_backend("intel_xpu")
def create_intel_xpu_backend(runner):
    # 创建 Intel XPU（GPU）注意力后端
    from sglang.srt.layers.attention.xpu_backend import XPUAttentionBackend

    return XPUAttentionBackend(runner)
