import logging

from sglang.srt.server_args import ServerArgs, get_global_server_args
from sglang.srt.utils.common import is_blackwell, is_musa

logger = logging.getLogger(__name__)


# 草稿模型注意力后端工厂：根据配置自动选择并创建适合的注意力后端实例
# 支持 decode 阶段（多步草稿生成）和 extend 阶段（KV 缓存追赶）两种场景
class DraftBackendFactory:
    def __init__(
        self,
        server_args: ServerArgs,
        draft_model_runner,
        topk: int,
        speculative_num_steps: int,
    ):
        # 保存服务器参数、草稿模型 runner、top-k 值和投机步数
        self.server_args = server_args
        self.draft_model_runner = draft_model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        # 用户可指定专用的草稿模型注意力后端，否则复用通用后端
        self.draft_attn_backend = server_args.speculative_draft_attention_backend

    def _create_backend(
        self, backend_name: str, backend_map: dict, error_template: str
    ):
        # 优先使用草稿模型专属后端，否则从 server_args 中读取对应阶段的后端类型
        backend_type = (
            self.draft_attn_backend
            if self.draft_attn_backend
            else getattr(self.server_args, backend_name)
        )
        # 若仍为 None，则回退到通用注意力后端
        if backend_type is None:
            backend_type = self.server_args.attention_backend

        # 检查后端类型是否受支持
        if backend_type not in backend_map:
            raise ValueError(error_template.format(backend_type=backend_type))

        # 调用对应的创建函数并返回后端实例
        return backend_map[backend_type]()

    def create_decode_backend(self):
        # 若只有 1 步投机，不需要 decode 阶段的草稿后端（直接由目标模型验证）
        if self.speculative_num_steps == 1:
            return None

        # decode 阶段：草稿模型多步自回归生成所用的注意力后端映射
        backend_map = {
            "flashinfer": self._create_flashinfer_decode_backend,
            "triton": self._create_triton_decode_backend,
            "aiter": self._create_aiter_decode_backend,
            "fa3": self._create_fa3_decode_backend,
            # Blackwell GPU 使用 triton 代替 fa3
            "hybrid_linear_attn": (
                self._create_fa3_decode_backend
                if not is_blackwell()
                else self._create_triton_decode_backend
            ),
            "flashmla": self._create_flashmla_decode_backend,
            "trtllm_mha": self._create_trtllm_mha_decode_backend,
            "trtllm_mla": self._create_trtllm_mla_decode_backend,
            "nsa": self._create_nsa_decode_backend,
            "ascend": self._create_ascend_decode_backend,
            "fa4": self._create_fa4_decode_backend,
        }

        return self._create_backend(
            "decode_attention_backend",
            backend_map,
            "EAGLE is not supported in decode attention backend {backend_type}",
        )

    def create_draft_extend_backend(self):
        # extend 阶段：验证后草稿模型追赶 KV 缓存所用的注意力后端映射
        backend_map = {
            "flashinfer": self._create_flashinfer_prefill_backend,
            "triton": self._create_triton_prefill_backend,
            "aiter": self._create_aiter_prefill_backend,
            "fa3": self._create_fa3_prefill_backend,
            # Blackwell GPU 使用 triton 代替 fa3
            "hybrid_linear_attn": (
                self._create_fa3_prefill_backend
                if not is_blackwell()
                else self._create_triton_prefill_backend
            ),
            "flashmla": self._create_flashmla_prefill_backend,
            "trtllm_mha": self._create_trtllm_mha_prefill_backend,
            "trtllm_mla": self._create_trtllm_mla_prefill_backend,
            "nsa": self._create_nsa_prefill_backend,
            "ascend": self._create_ascend_prefill_backend,
            "fa4": self._create_fa4_prefill_backend,
        }
        # 根据 speculative_attention_mode 决定使用 decode 还是 prefill 后端类型名
        backend_name = (
            "decode_attention_backend"
            if self.server_args.speculative_attention_mode == "decode"
            else "prefill_attention_backend"
        )
        return self._create_backend(
            backend_name,
            backend_map,
            "EAGLE is not supported in attention backend {backend_type}",
        )

    def _create_nsa_decode_backend(self):
        # 原生稀疏注意力（NSA）多步草稿后端
        from sglang.srt.layers.attention.nsa_backend import (
            NativeSparseAttnMultiStepBackend,
        )

        return NativeSparseAttnMultiStepBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_nsa_prefill_backend(self):
        # NSA prefill 后端（用于 extend 阶段）
        from sglang.srt.layers.attention.nsa_backend import NativeSparseAttnBackend

        return NativeSparseAttnBackend(self.draft_model_runner, skip_prefill=False)

    def _create_flashinfer_decode_backend(self):
        # FlashInfer 多步草稿 decode 后端，支持 MLA（Multi-head Latent Attention）和标准 MHA
        if not get_global_server_args().use_mla_backend:
            from sglang.srt.layers.attention.flashinfer_backend import (
                FlashInferMultiStepDraftBackend,
            )

            return FlashInferMultiStepDraftBackend(
                self.draft_model_runner, self.topk, self.speculative_num_steps
            )
        else:
            # MLA 模型使用专用的 FlashInfer MLA 多步后端
            from sglang.srt.layers.attention.flashinfer_mla_backend import (
                FlashInferMLAMultiStepDraftBackend,
            )

            return FlashInferMLAMultiStepDraftBackend(
                self.draft_model_runner, self.topk, self.speculative_num_steps
            )

    def _create_triton_decode_backend(self):
        # Triton 多步草稿 decode 后端（纯 Python Triton kernel，兼容性好）
        from sglang.srt.layers.attention.triton_backend import (
            TritonMultiStepDraftBackend,
        )

        return TritonMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_aiter_decode_backend(self):
        # AIter 多步草稿 decode 后端（AMD ROCm 专用优化后端）
        from sglang.srt.layers.attention.aiter_backend import AiterMultiStepDraftBackend

        return AiterMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_fa_decode_backend(self, fa_impl_ver: int = 3):
        # FlashAttention 多步草稿 decode 后端（支持 FA3 和 FA4）
        if not is_musa():
            from sglang.srt.layers.attention.flashattention_backend import (
                FlashAttentionMultiStepBackend,
            )
        else:
            # MUSA（摩尔线程 GPU）使用专用 FlashAttention 实现
            from sglang.srt.hardware_backend.musa.attention.flashattention_backend import (
                MusaFlashAttentionMultiStepBackend as FlashAttentionMultiStepBackend,
            )

        return FlashAttentionMultiStepBackend(
            self.draft_model_runner,
            self.topk,
            self.speculative_num_steps,
            fa_impl_ver=fa_impl_ver,
        )

    def _create_fa3_decode_backend(self):
        # FA3 版本的多步草稿 decode 后端
        return self._create_fa_decode_backend(fa_impl_ver=3)

    def _create_fa4_decode_backend(self):
        # FA4 版本的多步草稿 decode 后端
        return self._create_fa_decode_backend(fa_impl_ver=4)

    def _create_flashmla_decode_backend(self):
        # FlashMLA 多步草稿 decode 后端（针对 MLA 模型优化）
        from sglang.srt.layers.attention.flashmla_backend import (
            FlashMLAMultiStepDraftBackend,
        )

        return FlashMLAMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_trtllm_mha_decode_backend(self):
        # TensorRT-LLM MHA 多步草稿 decode 后端
        from sglang.srt.layers.attention.trtllm_mha_backend import (
            TRTLLMHAAttnMultiStepDraftBackend,
        )

        return TRTLLMHAAttnMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_trtllm_mla_decode_backend(self):
        # TensorRT-LLM MLA 多步草稿 decode 后端，仅支持 MLA 模型
        if not get_global_server_args().use_mla_backend:
            raise ValueError(
                "trtllm_mla backend requires MLA model (use_mla_backend=True)."
            )

        from sglang.srt.layers.attention.trtllm_mla_backend import (
            TRTLLMMLAMultiStepDraftBackend,
        )

        return TRTLLMMLAMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_ascend_decode_backend(self):
        # Ascend NPU 多步草稿 decode 后端
        from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
            AscendAttnMultiStepDraftBackend,
        )

        return AscendAttnMultiStepDraftBackend(
            self.draft_model_runner, self.topk, self.speculative_num_steps
        )

    def _create_flashinfer_prefill_backend(self):
        # FlashInfer prefill 后端（用于 extend 阶段），支持 MHA 和 MLA
        if not get_global_server_args().use_mla_backend:
            from sglang.srt.layers.attention.flashinfer_backend import (
                FlashInferAttnBackend,
            )

            return FlashInferAttnBackend(self.draft_model_runner, skip_prefill=False)
        else:
            # MLA 模型使用 FlashInfer MLA prefill 后端
            from sglang.srt.layers.attention.flashinfer_mla_backend import (
                FlashInferMLAAttnBackend,
            )

            return FlashInferMLAAttnBackend(self.draft_model_runner, skip_prefill=False)

    def _create_triton_prefill_backend(self):
        # Triton prefill 后端（extend 阶段）
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        return TritonAttnBackend(self.draft_model_runner, skip_prefill=False)

    def _create_aiter_prefill_backend(self):
        # AIter prefill 后端（extend 阶段，AMD ROCm）
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

        return AiterAttnBackend(self.draft_model_runner, skip_prefill=False)

    def _create_fa_prefill_backend(self, fa_impl_ver: int = 3):
        # FlashAttention prefill 后端（extend 阶段），支持 FA3 和 FA4
        if not is_musa():
            from sglang.srt.layers.attention.flashattention_backend import (
                FlashAttentionBackend,
            )
        else:
            # MUSA 专用实现
            from sglang.srt.hardware_backend.musa.attention.flashattention_backend import (
                MusaFlashAttentionBackend as FlashAttentionBackend,
            )
        return FlashAttentionBackend(
            self.draft_model_runner, skip_prefill=False, fa_impl_ver=fa_impl_ver
        )

    def _create_fa3_prefill_backend(self):
        # FA3 版本 prefill 后端
        return self._create_fa_prefill_backend(fa_impl_ver=3)

    def _create_fa4_prefill_backend(self):
        # FA4 版本 prefill 后端
        return self._create_fa_prefill_backend(fa_impl_ver=4)

    def _create_trtllm_mha_prefill_backend(self):
        # TensorRT-LLM MHA prefill 后端（extend 阶段）
        from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

        return TRTLLMHAAttnBackend(self.draft_model_runner, skip_prefill=False)

    def _create_trtllm_mla_prefill_backend(self):
        # TensorRT-LLM MLA prefill 后端，仅支持 MLA 模型
        if not get_global_server_args().use_mla_backend:
            raise ValueError(
                "trtllm_mla backend requires MLA model (use_mla_backend=True)."
            )

        from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend

        return TRTLLMMLABackend(self.draft_model_runner, skip_prefill=False)

    def _create_ascend_prefill_backend(self):
        # Ascend NPU prefill 后端（extend 阶段）
        from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
            AscendAttnBackend,
        )

        return AscendAttnBackend(self.draft_model_runner)

    def _create_flashmla_prefill_backend(self):
        # FlashMLA prefill 后端目前暂未支持 draft extend，返回 None
        logger.warning(
            "flashmla prefill backend is not yet supported for draft extend."
        )
        return None
