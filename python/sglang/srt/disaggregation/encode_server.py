# 编码服务器模块：负责多模态（图像/视频/音频）特征提取，并将 embedding 传输给 Prefill 端
# 在 PD 分离架构中，encode_server 作为独立服务运行，对多模态输入进行 ViT 编码后
# 通过 ZMQ 或 Mooncake 传输引擎将 embedding 发送到 Prefill 调度器
import asyncio
import concurrent.futures
import ctypes
import logging
import multiprocessing as mp
import os
import pickle
import time
import traceback
from http import HTTPStatus
from typing import Dict, List, Optional, Set, Tuple, Union

import aiohttp
import numpy as np
import torch
import uvicorn
import zmq
import zmq.asyncio
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse, Response
from transformers import AutoProcessor

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.encode_receiver import EmbeddingData
from sglang.srt.distributed.parallel_state import (
    get_default_distributed_backend,
    get_mooncake_transfer_engine,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.managers.io_struct import ProfileReq, ProfileReqInput, ProfileReqType
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.srt.model_loader import get_model
from sglang.srt.multimodal.processors.qwen_vl import preprocess_video
from sglang.srt.server_args import (
    PortArgs,
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.srt.utils import (
    load_audio,
    load_image,
    load_video,
    random_uuid,
)
from sglang.srt.utils.network import (
    NetworkAddress,
    config_socket,
    get_local_ip_auto,
    get_zmq_socket,
)

logger = logging.getLogger(__name__)

# 健康检查的超时时间（秒）
HEALTH_CHECK_TIMEOUT = 10

# Minimal 32x32 black PNG for health check dummy encode
# 用于健康检查的最小黑色 PNG 图片（32x32，Base64 编码）
MINIMUM_PNG_PICTURE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAbUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGBcua9HOR7Y6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=="

# Minimal WAV: 16kHz mono 16-bit PCM, 160 samples (0.01s) of silence
# 用于健康检查的最小静音 WAV 音频（16kHz 单声道，160 样本，Base64 编码）
MINIMUM_WAV_SILENCE_BASE64 = "UklGRmQBAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YUABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="

# 请求 ID 到接收端点列表的映射，用于 zmq_to_scheduler 传输模式
rid_lock = asyncio.Lock()
rid_to_receive_endpoint: Dict[str, List[str]] = dict()
# 每个请求 ID 期望的接收端点数量
rid_to_receive_count: Dict[str, int] = dict()
# 记录每个请求 ID 对应的错误信息
rid_to_err_msg: Dict[str, str] = dict()
# 用于等待接收端点注册的条件变量字典锁
cond_dict_lock = asyncio.Lock()
# 每个请求 ID 对应的异步条件变量，用于通知 send_with_url 有新端点到达
rid_to_cond: Dict[str, asyncio.Condition] = {}

# 是否使用 GPU 进行图像预处理（由环境变量控制）
use_image_processor_gpu = (
    int(os.getenv("SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU", "0")) == 1
)


# 多模态编码错误基类，携带 HTTP 状态码
class MMError(Exception):
    def __init__(self, message, code=HTTPStatus.INTERNAL_SERVER_ERROR):
        self.message = message
        self.code = code
        super().__init__(self.message)


# 客户端请求格式错误（400）
class BadRequestError(MMError):
    def __init__(self, message):
        super().__init__(message, code=HTTPStatus.BAD_REQUEST)


# 服务端内部错误（500）
class InternalError(MMError):
    def __init__(self, message):
        super().__init__(message, code=HTTPStatus.INTERNAL_SERVER_ERROR)


class TensorWrapper:
    """Wrapper to keep tensor alive while exposing buffer for zero-copy.
    零拷贝传输辅助类：持有张量引用防止被 GC 回收，同时暴露底层内存缓冲区"""

    def __init__(self, tensor):
        # Ensure tensor is on CPU and contiguous
        # 确保张量在 CPU 上且内存连续，用于 ZMQ 零拷贝发送
        if tensor.is_cuda:
            tensor = tensor.cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # Keep tensor reference
        # 持有张量引用，防止在缓冲区被消费前被垃圾回收
        self.tensor = tensor
        self.shape = list(tensor.shape)
        self.dtype = tensor.dtype

    def __buffer__(self):
        # 通过 ctypes 将张量底层内存暴露为 memoryview，实现零拷贝传输
        data_ptr = self.tensor.data_ptr()
        total_bytes = self.tensor.numel() * self.tensor.element_size()
        c_obj = (ctypes.c_char * total_bytes).from_address(data_ptr)
        c_obj._keep_alive_ref = self
        return memoryview(c_obj)


def _convert(data):
    # 将 numpy 数组或数值列表统一转换为 torch.Tensor，保持原有张量不变
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.tensor(data)
    elif isinstance(data, list) and isinstance(data[0], np.ndarray):
        return torch.tensor(np.array(data))
    elif isinstance(data, list) and isinstance(data[0], (int, float)):
        return torch.tensor(data)
    else:
        return data


# 各模态对应的网格尺寸属性名（用于从处理器输出中提取 patch grid 维度）
_mm_grid_attrs = {
    # Kimi K2.5 HF processor uses grid_thws (see base_processor.ATTR_NAME_TO_MODALITY).
    Modality.IMAGE: ["image_grid_thw", "image_grid_hws", "grid_thws"],
    Modality.VIDEO: ["video_grid_thw"],
    Modality.AUDIO: ["audio_feature_lens_raw"],
}

# 各模态对应的特征张量属性名（用于从处理器输出中提取 pixel_values 等特征）
_mm_feature_attrs = {
    Modality.IMAGE: ["pixel_values"],
    Modality.VIDEO: ["pixel_values_videos"],
    Modality.AUDIO: ["input_features"],
}


def _get_mm_grid_dim(mm_inputs, modality, model_type: Optional[str] = None):
    # 从处理器输出中提取指定模态的网格维度（patch grid），支持多种模型的属性名差异
    # Kimi K2.5 vision processor only emits `grid_thws`; prefer it over generic keys
    # so we never pick a mis-typed or stale `image_grid_hws` field from kwargs.
    attrs = _mm_grid_attrs[modality]
    if (model_type or "").lower() in [
        "kimi_k25",
        "kimi_vl",
    ] and modality == Modality.IMAGE:
        attrs = ("grid_thws", "image_grid_thw", "image_grid_hws")
    for attr in attrs:
        if attr in mm_inputs and mm_inputs[attr] is not None:
            return mm_inputs[attr]
    raise ValueError(f"Grid dim ({_mm_grid_attrs[modality]}) not found in {mm_inputs}")


def _get_mm_feature(mm_inputs, modality):
    # 从处理器输出中提取指定模态的原始特征张量（如 pixel_values）
    for attr in _mm_feature_attrs[modality]:
        if attr in mm_inputs:
            return mm_inputs[attr]
    raise ValueError(
        f"Feature attrs ({_mm_feature_attrs[modality]}) not found in {mm_inputs}"
    )


def _build_mm_aux_data(mm_inputs):
    """
    Build auxiliary data for video modality.
    构建视频模态的辅助元数据（时间戳、每格时间等），用于旋转位置编码计算
    """
    aux_data = {
        "video_timestamps": mm_inputs.get("video_timestamps", None),
        "second_per_grid_ts": mm_inputs.get("second_per_grid_ts", None),
    }
    return aux_data


class MMEncoder:
    # 多模态编码器：在 PD 分离的 Prefill 端负责将图像/视频/音频输入编码为 embedding
    # 支持张量并行（TP），所有 TP rank 协同运行 ViT/视觉编码器
    def __init__(
        self,
        server_args: ServerArgs,
        schedule_path=None,
        dist_init_method=None,
        rank: int = 0,
    ):
        logger.info(f"init MMEncoder {rank}/{server_args.tp_size}")
        self.server_args = server_args
        # 将全局 ServerArgs 注入调度器模块
        set_global_server_args_for_scheduler(server_args)
        self.rank = rank
        self.profiler = EncoderProfiler(rank)
        # 分别加载图像/视频/音频处理器
        self._load_mm_processor(server_args)

        # 从 ServerArgs 构建模型配置
        self.model_config = ModelConfig.from_server_args(
            server_args,
        )
        self.load_config = LoadConfig(
            load_format=server_args.load_format,
            download_dir=server_args.download_dir,
            model_loader_extra_config=server_args.model_loader_extra_config,
            remote_instance_weight_loader_seed_instance_ip=server_args.remote_instance_weight_loader_seed_instance_ip,
            remote_instance_weight_loader_seed_instance_service_port=server_args.remote_instance_weight_loader_seed_instance_service_port,
            remote_instance_weight_loader_send_weights_group_ports=server_args.remote_instance_weight_loader_send_weights_group_ports,
        )
        # 获取模型类型（小写），用于各模型的特殊处理分支
        self.model_type = getattr(
            self.model_config.hf_config, "model_type", "unknown"
        ).lower()

        self.device = server_args.device
        # 每个 rank 对应的 GPU ID（base_gpu_id + rank 偏移）
        self.gpu_id = server_args.base_gpu_id + rank

        self.device_config = DeviceConfig(
            device=self.device,
            gpu_id=self.gpu_id,
        )

        # 设置当前进程使用的 GPU 设备
        torch.get_device_module(self.device).set_device(self.gpu_id)

        self.use_image_processor_gpu = (
            use_image_processor_gpu and not server_args.disable_fast_image_processor
        )
        # 构建并校验视觉/音频处理配置（fps、max_frames 等）
        self._build_vision_config(server_args.mm_process_config)

        # 初始化分布式环境，支持多卡张量并行
        init_distributed_environment(
            backend=get_default_distributed_backend(self.device),
            world_size=server_args.tp_size,
            rank=rank,
            distributed_init_method=dist_init_method,
            local_rank=rank,
        )
        initialize_model_parallel(tensor_model_parallel_size=server_args.tp_size)
        initialize_dp_attention(server_args, self.model_config)

        # 加载视觉语言模型（仅视觉编码器部分参与推理）
        self.model = get_model(
            model_config=self.model_config,
            load_config=self.load_config,
            device_config=self.device_config,
        )

        # 异步 ZMQ 上下文和同步 ZMQ 上下文（线程池中使用）
        self.context = zmq.asyncio.Context(2)
        self.sync_context = zmq.Context()  # Reuse sync context for thread pool
        # 线程池用于并行发送 ZMQ 消息（避免阻塞 asyncio 事件循环）
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

        # 本地 embedding 缓存，大小由 SGLANG_VLM_CACHE_SIZE_MB 控制（默认 4096MB）
        embedding_cache_size = int(os.environ.get("SGLANG_VLM_CACHE_SIZE_MB", "4096"))
        self.mm_cache = MultiModalStaticCache(embedding_cache_size * 1024 * 1024)
        self.mm_cache_lock = asyncio.Lock()

        # 多模态数据 I/O 线程池，用于并发加载图像/视频/音频文件
        self.io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(os.environ.get("SGLANG_ENCODER_MM_LOAD_WORKERS", 4))
        )
        # 发送超时时间
        self.send_timeout = envs.SGLANG_ENCODER_SEND_TIMEOUT.get()

        # 如果提供了调度路径，创建 ZMQ PULL socket 从调度器接收编码请求
        if schedule_path is not None:
            self.schedule_socket = get_zmq_socket(
                self.context, zmq.PULL, schedule_path, True
            )
        # 后台异步任务集合，用于追踪并发的发送任务
        self.background_tasks: Set[asyncio.Task] = set()

        # 若启用全局多模态缓存（跨请求共享），初始化 EmbeddingCacheController
        if self.server_args.enable_mm_global_cache:
            from sglang.srt.mem_cache.storage.mooncake_store.embedding_cache_controller import (
                EmbeddingCacheController,
            )

            hidden_dims = self._infer_embedding_dims()
            self.mm_global_cache = EmbeddingCacheController(
                rank,
                server_args.tp_size,
                hidden_dims=hidden_dims,
                tp_group=get_tp_group().cpu_group,
                all_rank_get=False,
            )
        else:
            self.mm_global_cache = None

        # rank 0 负责实际的 embedding 发送，其余 rank 仅参与分布式推理
        if self.rank == 0:
            logger.info(
                f"Using transfer backend: {self.server_args.encoder_transfer_backend}"
            )

            # 若使用 Mooncake 传输引擎，初始化 RDMA 传输引擎用于高速 KV 传输
            if self.server_args.encoder_transfer_backend == "mooncake":
                self.local_ip = get_local_ip_auto()

                self.engine = get_mooncake_transfer_engine()
                if self.engine is None:
                    from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
                        init_mooncake_transfer_engine,
                    )

                    self.engine = init_mooncake_transfer_engine(
                        hostname=self.local_ip,
                        gpu_id=self.gpu_id,
                        ib_device=(
                            self.server_args.disaggregation_ib_device
                            or self.server_args.mooncake_ib_device
                        ),
                    )

            # 待发送的 embedding 字典：req_id -> EmbeddingData
            self.embedding_to_send = dict()

        logger.info(f"rank {rank} init finish ")

    def _infer_embedding_dims(self) -> dict:
        """Infer per-modality embedding dimensions from hf_config at init time.
        根据 hf_config 推断各模态 embedding 的隐藏维度，用于全局缓存分配"""
        default = self.model_config.hidden_size
        hf_cfg = self.model_config.hf_config
        # Omni 模型（如 Qwen3-Omni）的 thinker 子配置
        thinker_cfg = getattr(hf_cfg, "thinker_config", None)
        dims = {
            Modality.IMAGE: default,
            Modality.VIDEO: default,
            Modality.AUDIO: default,
        }

        # 优先从 thinker_config 或顶层 hf_config 获取视觉配置
        vision_cfg = getattr(thinker_cfg, "vision_config", None) or getattr(
            hf_cfg, "vision_config", None
        )
        if vision_cfg is not None:
            out_hs = getattr(vision_cfg, "out_hidden_size", None)
            if out_hs is not None:
                # 若存在 deepstack 多尺度索引，维度需乘以层数
                ds = getattr(vision_cfg, "deepstack_visual_indexes", None)
                vis_dim = (
                    out_hs * (1 + len(ds))
                    if isinstance(ds, (list, tuple)) and ds
                    else out_hs
                )
                dims[Modality.IMAGE] = vis_dim
                dims[Modality.VIDEO] = vis_dim

        # 推断音频编码器输出维度
        audio_cfg = getattr(thinker_cfg, "audio_config", None) or getattr(
            hf_cfg, "audio_config", None
        )
        if audio_cfg is not None:
            for attr in ("output_dim", "d_model"):
                val = getattr(audio_cfg, attr, None)
                if val and int(val) > 0:
                    dims[Modality.AUDIO] = int(val)
                    break

        logger.info(f"Global cache embedding dims: {dims}")
        return dims

    def _build_vision_config(self, mm_process_config):
        """
        Validate vision config, used for image/video/audio.
        If not provided, keep default values.
        校验并补全视觉/音频处理配置，设置各模态的默认参数（fps、padding 等）
        """
        self.vision_config = (
            mm_process_config.get("vision_config", {})
            if mm_process_config is not None
            else {}
        )
        for modality_str in ["image", "video", "audio"]:
            # 确保每个模态都有对应的配置字典
            if not self.vision_config.get(modality_str, None):
                self.vision_config[modality_str] = {}
            # 若使用 GPU 图像处理器，注入设备信息
            if self.use_image_processor_gpu:
                self.vision_config[modality_str]["device"] = self.device

            if modality_str == "video":
                # 视频模态默认参数：采样帧率 2fps，最大帧数 768，最小帧数 4
                video_defaults = {"fps": 2.0, "max_frames": 768, "min_frames": 4}
                for k, v in video_defaults.items():
                    self.vision_config["video"].setdefault(k, v)

            if modality_str == "audio":
                # 音频模态默认启用 attention_mask 返回
                if "return_attention_mask" not in self.vision_config["audio"]:
                    self.vision_config["audio"]["return_attention_mask"] = True
                if "padding" not in self.vision_config["audio"]:
                    if self.model_type == "qwen2_audio":
                        # For Qwen2Audio, use padding="max_length"
                        # (same as https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_audio/processing_qwen2_audio.py#L93)
                        self.vision_config["audio"]["padding"] = "max_length"
                    else:
                        self.vision_config["audio"]["padding"] = True
                if "truncation" not in self.vision_config["audio"]:
                    # keep same logic as base_processor.py
                    # 特定模型（Gemma3n、Qwen2Audio 等）不做截断
                    if (
                        hasattr(self, "audio_processor")
                        and self.audio_processor is not None
                    ):
                        if self.audio_processor.__class__.__name__ in {
                            "Gemma3nProcessor",
                            "GlmAsrProcessor",
                            "Qwen2AudioProcessor",
                            "Qwen3OmniMoeProcessor",
                        }:
                            self.vision_config["audio"]["truncation"] = False

    def _load_mm_processor(self, server_args: ServerArgs):
        """
        Load image/video/audio processor separately,
        avoid issues with AutoProcessor not recognizing certain models
        分别加载图像/视频/音频处理器，避免 AutoProcessor 对特殊模型的兼容问题
        """
        from transformers import AutoImageProcessor, AutoVideoProcessor

        try:
            # 加载图像处理器（支持快速处理器）
            self.image_processor = AutoImageProcessor.from_pretrained(
                server_args.tokenizer_path or server_args.model_path,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                use_fast=not server_args.disable_fast_image_processor,
            )
        except Exception as e:
            logger.warning(f"Failed to load image processor: {e}")
            self.image_processor = None

        try:
            # 加载视频处理器（支持快速处理器）
            self.video_processor = AutoVideoProcessor.from_pretrained(
                server_args.tokenizer_path or server_args.model_path,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                use_fast=not server_args.disable_fast_image_processor,
            )
        except Exception as e:
            logger.warning(f"Failed to load video processor: {e}")
            self.video_processor = None

        try:
            # Note: AutoProcessor is used for audio processor
            # 使用 AutoProcessor 加载音频处理器（含 feature_extractor）
            _audio_proc = AutoProcessor.from_pretrained(
                server_args.tokenizer_path or server_args.model_path,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                use_fast=not server_args.disable_fast_image_processor,
            )
            if not hasattr(_audio_proc, "feature_extractor"):
                logger.warning(
                    "Loaded AutoProcessor has no feature_extractor attribute, "
                    "audio processing will be unavailable."
                )
                self.audio_processor = None
            else:
                self.audio_processor = _audio_proc
        except Exception as e:
            logger.warning(f"Failed to load audio processor: {e}")
            self.audio_processor = None

    def _load_single_item(
        self,
        data,
        modality: Modality,
        frame_count_limit=None,
        audio_sample_rate: Optional[int] = None,
        discard_alpha_channel=True,
    ):
        """
        Load a single multimodal data.
        If data is precomputed, returns directly.
        Static method that can be pickled for multiprocessing
        加载单条多模态数据（图像/视频/音频），若已预处理则直接返回
        """
        # 若 data 已是字典（预计算数据），直接返回
        if isinstance(data, dict):
            return data
        try:
            if modality == Modality.IMAGE:
                img, _ = load_image(data, False)
                if (
                    discard_alpha_channel
                    and not isinstance(img, torch.Tensor)
                    and img.mode != "RGB"
                ):
                    # Needed only when `img` is a PIL image
                    # 丢弃 alpha 通道，转换为 RGB 模式
                    img = img.convert("RGB")
                return img
            elif modality == Modality.VIDEO:
                return load_video(data, frame_count_limit)
            elif modality == Modality.AUDIO:
                return load_audio(data, audio_sample_rate)

        except Exception as e:
            raise RuntimeError(f"Error while loading data {data}: {e}")

    def submit_data_loading_tasks(self, items, modalities):
        # 向 I/O 线程池提交多条多模态数据加载任务，返回 Future 列表
        futures = []
        task_info = []

        for data, modality in zip(items, modalities):
            if modality is not None:
                futures.append(
                    self.io_executor.submit(
                        self._load_single_item,
                        data,
                        modality,
                    )
                )
                task_info.append((modality, data))
        return futures, task_info

    def _get_feat_extract_output_lengths(self, feature_lens):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        计算音频编码器卷积层输出序列长度，不同模型（qwen2_audio、qwen3_asr 等）有不同的下采样公式
        """
        # qwen2_audio/qwen2.5_omni
        if self.model_type in ["qwen2_audio", "qwen2_5_omni"]:
            # 两级下采样：先 /2 再 /2
            input_length = (feature_lens - 1) // 2 + 1
            return (input_length - 2) // 2 + 1
        # qwen3_asr / qwen3_omni_moe (same audio encoder architecture)
        elif self.model_type in ["qwen3_asr", "qwen3_omni_moe"]:
            # 每 100 帧为一块，块内下采样两次，块间按 13 合并
            input_lengths_leave = feature_lens % 100
            feat_lengths = (input_lengths_leave - 1) // 2 + 1
            output_lengths = (
                ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (feature_lens // 100) * 13
            )
            return output_lengths
        else:
            # fallback to original HF audio sample logic for other models
            logger.warning(
                f"Fallback to original HF audio sample logic for {self.model_type}"
            )
            input_length = (feature_lens - 1) // 2 + 1
            return (input_length - 2) // 2 + 1

    async def _flatten_and_load_videos(self, mm_items):
        # 并发加载所有视频帧，并对 Qwen 系列模型做帧采样预处理
        if not isinstance(mm_items, (list, tuple)):
            mm_items = [mm_items]

        futures, _ = self.submit_data_loading_tasks(
            mm_items, [Modality.VIDEO] * len(mm_items)
        )
        # 将线程 Future 包装为异步 Future 并发等待
        async_futures = [asyncio.wrap_future(f) for f in futures]
        video_items = await asyncio.gather(*async_futures)

        video_processor_kwargs = {}
        if "qwen" in self.model_type:
            # for qwen-series model, do sample frames before preprocess
            # Qwen 系列模型需要在调用视频处理器前先对帧进行采样
            video_processed = [
                await preprocess_video(
                    video, video_config=self.vision_config.get("video", {})
                )
                for video in video_items
            ]
            videos, video_metadata = map(list, zip(*video_processed))
            video_processor_kwargs["do_sample_frames"] = False
            if video_metadata:
                video_processor_kwargs["video_metadata"] = video_metadata
            return videos, video_processor_kwargs
        else:
            raise NotImplementedError(
                f"Video processing is not supported for {self.model_type} model."
            )

    async def _flatten_and_load_data_by_modality(self, mm_items, modality):
        """
        Flatten mm_items structure, load multimodal data concurrently, and restore original structure.
        将嵌套的多模态数据结构展平，并发加载后还原原始嵌套结构

        Returns:
            Same structure as load_mm_items would return, support for image/audio
        """
        # Handle single mm_item (not a list)
        # 处理单个多模态项（非列表）
        if not isinstance(mm_items, (list, tuple)):
            futures, _ = self.submit_data_loading_tasks([mm_items], [modality])
            return await asyncio.wrap_future(futures[0])

        # Handle nested list (list of lists)
        # 处理嵌套列表（多组多模态数据）
        if len(mm_items) > 0 and isinstance(mm_items[0], (list, tuple)):
            # Flatten nested structure
            # 展平嵌套结构，记录每个元素所属的组索引
            flat_data = []
            flat_indices = []  # Track which group each item belongs to
            for group_idx, item_group in enumerate(mm_items):
                for item in item_group:
                    flat_data.append(item)
                    flat_indices.append(group_idx)

            # Submit all tasks concurrently
            # 并发提交所有加载任务
            futures, _ = self.submit_data_loading_tasks(
                flat_data, [modality] * len(flat_data)
            )

            # Wait for all tasks to complete asynchronously
            # 异步等待所有加载任务完成
            async_futures = [asyncio.wrap_future(f) for f in futures]
            results = await asyncio.gather(*async_futures)

            # Restore nested structure
            # 还原嵌套结构
            nested_results = [[] for _ in range(len(mm_items))]
            for idx, result in zip(flat_indices, results):
                nested_results[idx].append(result)

            return nested_results

        # Handle simple list
        # 处理普通列表
        else:
            futures, _ = self.submit_data_loading_tasks(
                mm_items, [modality] * len(mm_items)
            )
            # Wait for all tasks to complete asynchronously
            async_futures = [asyncio.wrap_future(f) for f in futures]
            return await asyncio.gather(*async_futures)

    def get_num_patches(
        self, grid: Union[torch.Tensor, List[int]], modality: Modality
    ) -> int:
        """Calculate number of raw patches (before merge/sampling). Used for pixel_values slicing.
        计算原始 patch 数量（合并/采样前），用于切分 pixel_values 张量"""
        if modality == Modality.AUDIO:
            return int(grid.item())
        else:
            # 图像/视频：T * H * W 三维 grid
            return int(grid[0] * grid[1] * grid[2])

    def _kimi_tokens_from_patch_grid(self, grid: Union[torch.Tensor, List[int]]) -> int:
        """MoonViT + tpool: output len is (h//mh)*(w//mw); temporal dim is pooled (not t*h*w/merge^2).
        Kimi 模型的 token 数计算：时间维度已池化，仅按空间维度和合并核大小计算"""
        if isinstance(grid, torch.Tensor):
            flat = grid.flatten()
            _t, h, w = (int(x) for x in flat[:3].tolist())
        else:
            _t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
        merge_h, merge_w = self.model_config.hf_config.vision_config.merge_kernel_size
        return (h * w) // (merge_h * merge_w)

    def get_num_tokens(
        self, grid: Union[torch.Tensor, List[int]], modality: Modality
    ) -> int:
        """Calculate number of tokens (after 2x2 merge). Used for mm_embedding slicing.
        计算合并后的 token 数量，用于切分最终 mm_embedding 张量"""
        if modality == Modality.AUDIO:
            # 音频 token 数需经过卷积下采样公式计算
            input_length = self.get_num_patches(grid, modality)
            return self._get_feat_extract_output_lengths(input_length)
        else:
            # Kimi 模型使用特殊的 token 计数方式
            if (
                self.model_type in ["kimi_k25", "kimi_vl"]
                and modality == Modality.IMAGE
            ):
                return self._kimi_tokens_from_patch_grid(grid)
            # 标准视觉模型：patch 数除以 merge_size^2（默认 2x2=4）
            merge_size = getattr(self.image_processor, "merge_size", 2)
            return self.get_num_patches(grid, modality) // (merge_size**2)

    def slice_embedding(
        self, mm_embedding: torch.Tensor, grid_thw: List, modality: Modality
    ) -> List[torch.Tensor]:
        """Slice a concatenated embedding tensor into individual image embeddings.
        将拼接的 embedding 张量按各图像的 token 数切分为独立 embedding 列表"""
        slices, offset = [], 0
        for grid in grid_thw:
            count = self.get_num_tokens(grid, modality)
            slices.append(mm_embedding[offset : offset + count])
            offset += count
        return slices

    def _calculate_hashes_from_features(
        self, mm_feature: torch.Tensor, grid_thw: List, modality: Modality
    ) -> List[str]:
        """CPU Task: Compute hashes based on processed feature patches.
        对每个多模态项的原始特征计算哈希值，用于全局缓存命中检测"""
        hashes, offset = [], 0
        logger.info(f"{mm_feature.shape=} with {modality=}")
        for grid in grid_thw:
            num_patches = self.get_num_patches(grid, modality)
            feature_slice = mm_feature[offset : offset + num_patches]
            tmp_item = MultimodalDataItem(modality=modality, feature=feature_slice)
            tmp_item.set_pad_value()
            hashes.append(tmp_item.hash)
            offset += num_patches
        return hashes

    async def _encode_missing(
        self,
        mm_feature: torch.Tensor,
        mm_inputs: dict,
        indices: List[int],
        modality: Modality = Modality.IMAGE,
        get_feature_fn=None,
    ) -> List[torch.Tensor]:
        """
        GPU Task: Run ViT inference ONLY on the subset of mm items missing from the cache.
        仅对缓存未命中的多模态项运行 ViT 推理（避免重复计算缓存命中的项）
        """
        grid_thw = _get_mm_grid_dim(mm_inputs, modality, self.model_type)

        # 1. Slice mm_feature to get only the patches for missing mm items
        # 按 patch 偏移量计算各项的起止位置
        sub_feature_list = []
        offsets = [0]
        curr = 0
        for g in grid_thw:
            curr += self.get_num_patches(g, modality)
            offsets.append(curr)

        # 提取仅缺失项的特征切片
        for idx in indices:
            sub_feature_list.append(mm_feature[offsets[idx] : offsets[idx + 1]])

        sub_feature = torch.cat(sub_feature_list, dim=0)

        # 构建缺失项的 MultimodalDataItem
        mm_item = MultimodalDataItem.from_dict(
            {
                "modality": modality,
                "feature": _convert(sub_feature),
            }
        )

        # 将其他辅助属性（grid、attention_mask 等）注入 mm_item
        for k, v in mm_inputs.items():
            if k in _mm_feature_attrs.get(modality, []):
                continue
            val = _convert(v)
            if k in _mm_grid_attrs.get(modality, []):
                # grid 属性需要按缺失索引切片
                mm_item.set(k, val[indices])
            else:
                mm_item.set(k, val)

        # 在推理模式下运行 ViT，输出 embedding reshape 为 (tokens, hidden_dim)
        with torch.inference_mode():
            new_embeddings = get_feature_fn([mm_item]).cpu()
            if new_embeddings.ndim != 2:
                new_embeddings = new_embeddings.reshape(-1, new_embeddings.shape[-1])

        # 将合并的 embedding 按各缺失项的 token 数切分为列表
        sub_grids = [grid_thw[i] for i in indices]
        return self.slice_embedding(new_embeddings, sub_grids, modality)

    async def encode_with_global_cache(
        self,
        mm_items,
        modality: Modality,
        req_id: str,
        num_parts: int,
        part_idx: int,
        hashes: Optional[List[str]] = None,
    ) -> torch.Tensor:
        # 使用全局多模态缓存进行编码：对缓存命中的项直接取出 embedding，
        # 对缓存未命中的项运行 ViT 推理，最后拼接并存储到全局缓存
        # mm_inputs: dict
        mm_inputs, get_feature_fn = await self._process_mm_items(mm_items, modality)
        grid_thw = _get_mm_grid_dim(mm_inputs, modality, self.model_type)
        mm_feature = _convert(_get_mm_feature(mm_inputs, modality))
        num_items = len(grid_thw)

        # Step 1: Rank 0 checks global cache and broadcasts hit/miss mask to all ranks.
        # rank 0 检查全局缓存，生成命中/未命中掩码并广播给所有 rank
        if self.rank == 0:
            if hashes is None:
                # 若未预先提供哈希值，从特征张量计算
                mm_hashes = self._calculate_hashes_from_features(
                    mm_feature, grid_thw, modality
                )
            else:
                mm_hashes = hashes
            exist_mask = await self.mm_global_cache.batch_is_exist(mm_hashes)
            mask_tensor = torch.tensor(
                [1 if e else 0 for e in exist_mask], dtype=torch.int32
            )
        else:
            mm_hashes = None
            mask_tensor = torch.zeros(num_items, dtype=torch.int32)

        # 多 rank 下通过 distributed broadcast 同步命中掩码
        if self.server_args.tp_size > 1:
            torch.distributed.broadcast(
                mask_tensor,
                src=0,
                group=self.mm_global_cache.prefetch_tp_group,
            )

        exist_mask = [m.item() == 1 for m in mask_tensor]
        missing_indices = [i for i, e in enumerate(exist_mask) if not e]
        hit_indices = [i for i, e in enumerate(exist_mask) if e]

        # Step 2: All ranks run ViT together on cache-miss images.
        # 所有 rank 协同对缓存未命中的项运行 ViT 推理
        new_slices = []
        if missing_indices:
            new_slices = await self._encode_missing(
                mm_feature, mm_inputs, missing_indices, modality, get_feature_fn
            )

        # Step 3: Rank 0 prefetches cache-hit embeddings from global cache.
        # rank 0 从全局缓存预取命中项的 embedding
        prefetch_status = torch.tensor([1], dtype=torch.int32)

        if self.rank == 0:
            if hit_indices:
                hit_hashes = [mm_hashes[i] for i in hit_indices]
                hit_tokens = [
                    self.get_num_tokens(grid_thw[i], modality) for i in hit_indices
                ]
                self.mm_global_cache.prefetch(req_id, hit_hashes, hit_tokens, modality)

                try:

                    async def _wait_prefetch():
                        # 轮询等待预取完成（间隔 5ms）
                        while not self.mm_global_cache.check_prefetch_progress(req_id):
                            await asyncio.sleep(0.005)

                    await asyncio.wait_for(_wait_prefetch(), timeout=60.0)
                except (asyncio.TimeoutError, Exception) as e:
                    logger.error(
                        f"Prefetch failed for req {req_id}: {e}. "
                        f"Falling back to ViT for {len(hit_indices)} hit items."
                    )
                    # 预取失败，标记为 0，触发回退重新用 ViT 计算
                    prefetch_status[0] = 0

        # Step 4: Broadcast prefetch result to all ranks so they stay in sync.
        # 广播预取状态，确保所有 rank 在同一状态下执行后续逻辑
        if self.server_args.tp_size > 1:
            torch.distributed.broadcast(
                prefetch_status,
                src=0,
                group=self.mm_global_cache.prefetch_tp_group,
            )

        # Step 5: If prefetch failed, all ranks fallback to ViT for the hit mm items.
        # 预取失败时，所有 rank 对命中项重新运行 ViT 作为回退
        if prefetch_status.item() == 0 and hit_indices:
            logger.info(
                f"Req {req_id}: Prefetch failed, all ranks running ViT fallback "
                f"for {len(hit_indices)} mm items."
            )
            fallback_slices = await self._encode_missing(
                mm_feature, mm_inputs, hit_indices, modality, get_feature_fn
            )
        else:
            fallback_slices = None

        # Step 6: Rank 0 assembles final embedding and prepares for sending.
        # rank 0 组装最终 embedding：将新计算的和从缓存读取的按原始顺序拼接
        if self.rank == 0:
            final_slices = [None] * num_items

            for i, idx in enumerate(missing_indices):
                final_slices[idx] = new_slices[i]

            # Fill in cache-hit embeddings (from prefetch or fallback)
            # 填入命中缓存的 embedding（预取成功时从缓存读取，失败时用回退计算结果）
            if prefetch_status.item() == 1 and hit_indices:
                cached_slices = self.mm_global_cache.get_embeddings(
                    [mm_hashes[i] for i in hit_indices]
                )
                for i, idx in enumerate(hit_indices):
                    final_slices[idx] = cached_slices[i]
            elif fallback_slices is not None:
                for i, idx in enumerate(hit_indices):
                    final_slices[idx] = fallback_slices[i]

            mm_embedding = torch.cat(final_slices, dim=0)

            # Background insert: store newly computed embeddings into global cache.
            # Includes both original misses and fallback-recomputed hits.
            # 后台异步将新计算的 embedding 插入全局缓存（含 miss 项和回退重算的 hit 项）
            all_new_hashes = [mm_hashes[i] for i in missing_indices]
            all_new_slices = list(new_slices)
            if fallback_slices is not None:
                all_new_hashes += [mm_hashes[i] for i in hit_indices]
                all_new_slices += list(fallback_slices)

            if all_new_hashes:

                async def _background_insert():
                    await asyncio.to_thread(
                        self.mm_global_cache.insert_batch,
                        all_new_hashes,
                        all_new_slices,
                    )

                task = asyncio.create_task(_background_insert())
                self.background_tasks.add(task)
                task.add_done_callback(self.background_tasks.discard)

            # 构建辅助数据（视频时间戳等）并保存待发送的 EmbeddingData
            aux_data = _build_mm_aux_data(mm_inputs)
            self.embedding_to_send[req_id] = EmbeddingData(
                req_id,
                num_parts,
                part_idx,
                grid_thw,
                modality,
                mm_embedding,
                **aux_data,
            )
            return (
                mm_embedding.nbytes,
                mm_embedding.shape[0],
                mm_embedding.shape[1],
                None,
                None,
            )
        else:
            # 非 rank 0 只参与分布式推理，不负责发送
            return (0, 0, 0, None, None)

    async def _flatten_and_load_audios(self, mm_items):
        """
        Flatten mm_items structure, load audios concurrently, and restore original structure.
        并发加载音频数据，复用通用的展平/还原逻辑
        """
        return await self._flatten_and_load_data_by_modality(mm_items, Modality.AUDIO)

    async def _flatten_and_load_images(self, mm_items):
        """
        Flatten mm_items structure, load images concurrently, and restore original structure.
        并发加载图像数据，复用通用的展平/还原逻辑
        """
        return await self._flatten_and_load_data_by_modality(mm_items, Modality.IMAGE)

    def _calculate_timestamps(self, indices, video_fps: float, merge_size: int = 2):
        """Calculate timestamps for video frames, used for qwen3_vl models.
        计算视频帧的时间戳，供 Qwen3-VL 的旋转位置编码使用"""
        # refer to https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/processing_qwen3_vl.py#L255
        if not isinstance(indices, list):
            indices = indices.tolist()
        # 若帧数不是 merge_size 的整数倍，用最后一帧补齐
        if len(indices) % merge_size != 0:
            indices.extend(
                indices[-1] for _ in range(merge_size - len(indices) % merge_size)
            )
        timestamps = [idx / video_fps for idx in indices]
        # Frames are merged by merge_size, so we need to average the timestamps
        # between the first/last frame within the temporal patch
        # 对每个时间 patch 内的帧取首尾平均作为该 patch 的时间戳
        timestamps = [
            (timestamps[i] + timestamps[i + merge_size - 1]) / 2
            for i in range(0, len(timestamps), merge_size)
        ]
        return timestamps

    @staticmethod
    def _flatten_nested_items(items):
        # 递归展平嵌套列表，用于处理 Kimi 等模型的多层嵌套输入
        if not isinstance(items, (list, tuple)):
            return [items]

        flat = []
        for item in items:
            if isinstance(item, (list, tuple)):
                flat.extend(MMEncoder._flatten_nested_items(item))
            else:
                flat.append(item)
        return flat

    def _normalize_kimi_encoder_images(self, images):
        """Normalize Kimi image inputs for the image processor call.
        将 Kimi 系列模型的图像输入规范化为处理器所期望的格式（媒体字典列表）"""
        from PIL import Image as PILImage

        def wrap_one(img):
            # 将单张图片包装为 Kimi K2.5 处理器要求的媒体字典格式
            if isinstance(img, dict) and img.get("type") in ("image", "video_chunk"):
                return [img]
            if isinstance(img, PILImage.Image):
                return [{"type": "image", "image": img}]
            return [img]

        if not images:
            return images

        # Disagg may supply nested lists from grouped routing.
        # 分离架构可能传入嵌套列表，先展平
        images = self._flatten_nested_items(images)

        # Kimi-VL image processor expects a flat list of concrete images.
        # Kimi-VL 处理器期望接收具体图片对象列表（非媒体字典）
        if self.model_type == "kimi_vl":
            normalized = []
            for img in images:
                if (
                    isinstance(img, dict)
                    and img.get("type") == "image"
                    and "image" in img
                ):
                    inner = img["image"]
                    if isinstance(inner, (list, tuple)):
                        normalized.extend(self._flatten_nested_items(inner))
                    else:
                        normalized.append(inner)
                else:
                    normalized.append(img)
            return normalized

        # Kimi-K2.5 vision processor expects media dicts.
        # Kimi-K2.5 视觉处理器期望接收媒体字典列表
        normalized = []
        for img in images:
            wrapped = wrap_one(img)
            for media in wrapped:
                # Some pipelines may produce {"type": "image", "image": [PIL]}.
                # Split it into one media item per concrete image object.
                # 若 image 字段包含列表，拆分为多个独立媒体字典
                if (
                    isinstance(media, dict)
                    and media.get("type") == "image"
                    and isinstance(media.get("image"), (list, tuple))
                ):
                    for inner in self._flatten_nested_items(media["image"]):
                        normalized.append({**media, "image": inner})
                else:
                    normalized.append(media)

        return normalized

    async def _process_mm_items(self, mm_items, modality):
        # 根据模态调用对应的处理器，返回处理器输出字典和特征提取函数
        if modality == Modality.IMAGE and self.image_processor:
            # 加载图像并调用图像处理器
            images = await self._flatten_and_load_images(mm_items)
            image_config = self.vision_config.get("image", {})
            if self.model_type in ["kimi_k25", "kimi_vl"]:
                images = self._normalize_kimi_encoder_images(images)
            processor_input = self.image_processor(images=images, **image_config)
            if hasattr(self.model, "thinker"):  # for omni models
                get_feature_method = self.model.thinker.get_image_feature
            else:
                get_feature_method = self.model.get_image_feature
        elif modality == Modality.VIDEO and self.video_processor:
            # 加载并采样视频帧，调用视频处理器
            videos, video_processor_kwargs = await self._flatten_and_load_videos(
                mm_items
            )
            processor_input = self.video_processor(
                videos=videos, **video_processor_kwargs
            )
            # Get additional video metadata
            # 为 Qwen3-VL 系列模型计算视频时间戳（用于 RoPE）
            if (
                self.model_type
                in ["qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"]
                and video_processor_kwargs.get("video_metadata", None) is not None
            ):
                # For qwen3-vl/qwen3.5 models, we need to store the video timestamps
                video_metadata = video_processor_kwargs["video_metadata"]
                try:
                    merge_size = (
                        self.model_config.hf_config.vision_config.spatial_merge_size
                    )
                except (AttributeError, KeyError):
                    merge_size = 2  # Default merge_size

                video_timestamps = []
                for metadata in video_metadata:
                    video_fps = metadata.get("fps", None) or 24  # original video fps
                    frames_indices = metadata.get("frames_indices", None)
                    timestamps = self._calculate_timestamps(
                        frames_indices, video_fps, merge_size
                    )
                    video_timestamps.append(timestamps)
                processor_input["video_timestamps"] = video_timestamps
            elif (
                self.model_type in ["qwen2_5_vl", "qwen2_5_omni", "qwen3_omni_moe"]
                and processor_input.get("video_grid_thw", None) is not None
            ):
                # For omni/qwen2_5_vl models, calculate second_per_grid_ts for rotary embedding
                # 为 Qwen2.5-VL/Omni 计算每个视频 patch 对应的时间（秒）
                video_grid_thw = processor_input["video_grid_thw"]
                try:
                    temporal_patch_size = self.video_processor.temporal_patch_size
                except AttributeError:
                    temporal_patch_size = 2  # Default temporal_patch_size
                # get sampled fps, default: 2
                fps_list = [
                    self.vision_config.get("video", {}).get("fps", None) or 2
                ] * len(video_grid_thw)
                second_per_grid_ts = [(temporal_patch_size / fps) for fps in fps_list]
                second_per_grid_ts_tensor = torch.tensor(
                    second_per_grid_ts, dtype=torch.float32
                )
                processor_input["second_per_grid_ts"] = second_per_grid_ts_tensor

            if hasattr(self.model, "thinker"):  # for omni models
                get_feature_method = self.model.thinker.get_video_feature
            else:
                get_feature_method = self.model.get_video_feature
        elif modality == Modality.AUDIO and self.audio_processor:
            # 加载音频并调用特征提取器（feature_extractor）
            audios = await self._flatten_and_load_audios(mm_items)
            audio_config = self.vision_config.get("audio", {})
            processor_input = self.audio_processor.feature_extractor(
                audios, **audio_config
            )
            # 将 attention_mask 重命名为 feature_attention_mask 以匹配模型接口
            processor_input["feature_attention_mask"] = processor_input.pop(
                "attention_mask"
            )
            # convert to same format as image/video
            # 计算每条音频的有效帧长度，用于 token 数计算
            input_lengths = torch.tensor(
                processor_input["feature_attention_mask"].sum(-1), dtype=torch.long
            )
            processor_input["audio_feature_lens_raw"] = input_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)
            processor_input["audio_feature_lens"] = output_lengths
            if hasattr(self.model, "thinker"):  # for omni models
                get_feature_method = self.model.thinker.get_audio_feature
            else:
                get_feature_method = self.model.get_audio_feature
        else:
            raise ValueError(
                f"Currently only support image, video and audio modalities, {modality} modality has no processor available."
            )

        return processor_input, get_feature_method

    async def _encode(self, mm_items, modality: Modality) -> torch.Tensor:
        # 执行单次多模态编码：处理输入、可选缓存查找、ViT 推理、缓存写入
        try:
            mm_inputs, get_feature_fn = await self._process_mm_items(mm_items, modality)
        except NotImplementedError as e:
            raise InternalError(f"Not implemented error: {str(e)}")
        except Exception as e:
            raise BadRequestError(f"Failed to process mm items: {str(e)}")
        try:
            # support mm_cache
            # 尝试从本地 prefix 缓存中查找已有 embedding，避免重复编码
            mm_embedding = None
            mm_hash = None

            mm_item = MultimodalDataItem.from_dict(
                {
                    "modality": modality,
                    "feature": _convert(_get_mm_feature(mm_inputs, modality)),
                }
            )
            # 将处理器输出的其他属性注入 mm_item（grid、attention_mask 等）
            for k, v in mm_inputs.items():
                if k in _mm_feature_attrs[modality]:
                    continue
                mm_item.set(k, _convert(v))

            if self.server_args.enable_prefix_mm_cache:
                mm_item.set_pad_value()
                mm_hash = MultiModalStaticCache.combine_hashes([mm_item.hash])
                async with self.mm_cache_lock:
                    # 检查本地 embedding 缓存是否命中
                    mm_cache = self.mm_cache.get([mm_item.hash])
                    if mm_cache is not None:
                        mm_embedding = mm_cache.embedding

            if mm_embedding is None:
                # 缓存未命中，运行 ViT 推理获取 embedding
                with torch.inference_mode():
                    mm_embedding: torch.Tensor = get_feature_fn([mm_item])
                    mm_embedding = mm_embedding.cpu()
                if len(mm_embedding.shape) != 2:
                    mm_embedding = mm_embedding.reshape(-1, mm_embedding.shape[-1])

            if self.server_args.enable_prefix_mm_cache:
                # 将新计算的 embedding 写入本地缓存
                async with self.mm_cache_lock:
                    self.mm_cache.set(mm_hash, EmbeddingResult(embedding=mm_embedding))
            if self.profiler is not None:
                self.profiler.step()

            aux_data = _build_mm_aux_data(mm_inputs)
            return (
                _get_mm_grid_dim(mm_inputs, modality, self.model_type),
                mm_embedding,
                aux_data,
            )
        except BadRequestError as e:
            raise BadRequestError(f"Bad request error: {str(e)}")
        except Exception as e:
            raise InternalError(f"Internal encoding error: {str(e)}")

    async def _send(
        self,
        embedding: torch.Tensor,
        mm_data: EmbeddingData,
        session_id=None,
        buffer_address=None,
        prefill_host=None,
        embedding_port=None,
        url=None,
    ):
        # 将编码结果发送到 Prefill 端：支持 Mooncake RDMA 传输和 ZMQ 消息传输两种方式
        if self.server_args.encoder_transfer_backend == "mooncake":
            # Mooncake 传输：注册内存 -> RDMA 同步传输 -> 注销内存
            self.engine.register(embedding.data_ptr(), embedding.nbytes)
            self.engine.transfer_sync(
                session_id, embedding.data_ptr(), buffer_address, embedding.nbytes
            )
            self.engine.deregister(embedding.data_ptr())

            # Mooncake 传输完成后，清空 embedding 数据（已通过 RDMA 传输到 Decode 端）
            mm_data.embedding = None

        # Send ack/data
        # 解析目标端点地址（URL 或 host+port 两种形式）
        if url is not None:
            endpoint = NetworkAddress.parse(url).to_tcp()
        else:
            endpoint = NetworkAddress(prefill_host, embedding_port).to_tcp()
        logger.info(f"{endpoint = }")

        # Serialize data
        # 根据传输后端选择序列化方式
        if self.server_args.encoder_transfer_backend == "mooncake":
            # Mooncake：embedding 已传输，只发送元数据
            serialized_data = pickle.dumps(mm_data)
            buffer = None
        else:
            # ZMQ：分离元数据和 embedding 数据，使用零拷贝发送
            new_mm_data = mm_data.copy_without_embedding()
            if new_mm_data.error_msg is not None:
                buffer = None
                serialized_data = pickle.dumps(new_mm_data)
            else:
                embedding_tensor = TensorWrapper(mm_data.embedding)
                serialized_data = pickle.dumps(new_mm_data)
                buffer = embedding_tensor.__buffer__()

        # Use thread pool executor for parallel ZMQ send operations
        # 在线程池中执行 ZMQ 发送（避免阻塞 asyncio 事件循环）
        def send_with_socket():
            # 为每次发送创建新的 ZMQ PUSH socket（线程安全）
            sock = self.sync_context.socket(zmq.PUSH)
            config_socket(sock, zmq.PUSH)
            try:
                sock.connect(endpoint)
                if buffer is not None:
                    # 多帧发送：元数据帧 + embedding 数据帧（零拷贝）
                    sock.send_multipart([serialized_data, buffer], copy=False)
                else:
                    sock.send_multipart([serialized_data], copy=False)
            finally:
                sock.close()

        await asyncio.get_event_loop().run_in_executor(self.executor, send_with_socket)

    async def encode(self, mm_items, modality: Modality, req_id, num_parts, part_idx):
        # 对外暴露的编码入口：调用 _encode 并将结果包装为 EmbeddingData 保存
        try:
            grid_dim, mm_embedding, aux_data = await self._encode(mm_items, modality)

            if self.rank == 0:
                # rank 0 创建 EmbeddingData 并保存，等待后续 send 调用发送
                mm_data = EmbeddingData(
                    req_id,
                    num_parts,
                    part_idx,
                    grid_dim,
                    modality,
                    mm_embedding,
                    **aux_data,
                )
                self.embedding_to_send[req_id] = mm_data
            return (
                mm_embedding.nbytes,
                mm_embedding.shape[0],
                mm_embedding.shape[1],
                None,
                None,
            )
        except Exception as e:
            # 编码失败时创建错误 EmbeddingData，供后续发送错误响应给 Prefill 端
            error_code = getattr(e, "code", HTTPStatus.INTERNAL_SERVER_ERROR)
            error_msg = str(e)
            logger.error(f"Rank {self.rank} encode failed: {error_msg} {error_code = }")
            if self.rank == 0:
                mm_data = EmbeddingData(
                    req_id,
                    num_parts,
                    part_idx,
                    None,
                    modality,
                    error_msg=error_msg,
                    error_code=error_code,
                )
                self.embedding_to_send[req_id] = mm_data
                logger.debug(f"Created error EmbeddingData: {mm_data}")
            return 0, 0, 0, error_msg, error_code

    # For zmq_to_tokenizer zmq_to_scheduler and mooncake
    # 供 zmq_to_tokenizer/zmq_to_scheduler/mooncake 三种传输模式使用的发送接口
    async def send(
        self, req_id, prefill_host, embedding_port, session_id=None, buffer_address=None
    ):
        # 从待发送字典中取出对应的 EmbeddingData 并发送
        mm_data: EmbeddingData = self.embedding_to_send[req_id]
        await self._send(
            mm_data.embedding,
            mm_data,
            session_id=session_id,
            buffer_address=buffer_address,
            prefill_host=prefill_host,
            embedding_port=embedding_port,
        )

    # For zmq_to_scheduler
    # 供 zmq_to_scheduler 模式使用：等待 Decode 端注册接收 URL 后发送
    async def send_with_url(
        self,
        req_id,
    ):
        mm_data = self.embedding_to_send.get(req_id)
        if not mm_data:
            return
        sent_urls: Set[str] = set()
        all_tasks: List[Tuple[asyncio.Task, str]] = []
        start_time = asyncio.get_running_loop().time()
        timeout = self.send_timeout
        # 获取或创建该请求的条件变量，用于等待新端点注册
        cond = await get_condition(req_id)

        try:
            while True:
                async with rid_lock:
                    current_targets = rid_to_receive_endpoint.get(req_id, set()).copy()
                    expected_count = rid_to_receive_count.get(req_id)

                # 找到尚未发送的新端点
                new_targets = current_targets - sent_urls

                if new_targets:
                    logger.info(
                        f"Found {len(new_targets)} new endpoints for {req_id}. Starting tasks..."
                    )
                    for url in new_targets:
                        # 为每个新端点异步发送 embedding
                        task = asyncio.create_task(
                            self._send(
                                mm_data.embedding,
                                mm_data,
                                url=url,
                            )
                        )
                        all_tasks.append((task, url))
                        sent_urls.add(url)  # Mark as handled immediately
                # 已发送给所有期望的端点，退出循环
                if expected_count is not None and len(sent_urls) >= expected_count:
                    logger.info(
                        f"All {expected_count} endpoints initiated for {req_id}. Breaking loop."
                    )
                    break
                remaining = timeout - (asyncio.get_running_loop().time() - start_time)
                if remaining <= 0:
                    logger.error(
                        f"[{req_id}] Timeout! Sent {len(sent_urls)}/{expected_count}"
                    )
                    break

                # 等待新端点注册的通知（有超时保护）
                async with cond:
                    try:
                        await asyncio.wait_for(cond.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        continue

            if all_tasks:
                logger.info(
                    f"Loop finished. Awaiting completion of {len(all_tasks)} sending tasks..."
                )
                tasks_only = [t[0] for t in all_tasks]
                # 等待所有发送任务完成，收集异常
                results = await asyncio.gather(*tasks_only, return_exceptions=True)

                # Process results and log errors
                for i, result in enumerate(results):
                    url = all_tasks[i][1]  # Retrieve URL associated with the task
                    if isinstance(result, Exception):
                        logger.error(f"Failed to send to {url}: {result}")
                    else:
                        logger.debug(f"Successfully sent to {url}")

            logger.info(f"All tasks completed for req_id: {req_id}")

        finally:
            # 清理该请求的所有状态数据
            logger.info(f"Cleaning up resources for req_id {req_id}")
            async with rid_lock:
                rid_to_receive_endpoint.pop(req_id, None)
                rid_to_receive_count.pop(req_id, None)
            async with cond_dict_lock:
                rid_to_cond.pop(req_id, None)
            self.embedding_to_send.pop(req_id, None)

    async def get_embedding_port(self, prefill_url):
        # 向 Prefill 端查询 embedding 接收端口（用于动态协商）
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=1800)
        ) as session:
            response = await session.post(
                f"{prefill_url}/embedding_bootstrap",
                json={"embedding_port": None},
            )
            response_json = await response.json()
            return response_json["embedding_port"]


class EncoderProfiler:
    # 编码器性能剖析器：支持按步骤自动停止的 PyTorch Profiler 封装
    def __init__(self, rank: int):
        self.rank = rank
        self.profiler = None
        self.steps_left = None
        self.output_dir = None
        self.prefix = None
        self.profile_id = None

    def start(self, obj: ProfileReq):
        # 启动 Profiler：创建 torch.profiler.profile 实例并开始采集
        if self.profiler is not None:
            return False, "profiling already running"

        output_dir = obj.output_dir or os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp")
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.prefix = obj.profile_prefix or "encoder"
        self.profile_id = str(time.time())

        activities = obj.activities or ["CPU", "GPU"]
        torch_activities = []
        if "CPU" in activities:
            torch_activities.append(torch.profiler.ProfilerActivity.CPU)
        if "GPU" in activities:
            torch_activities.append(torch.profiler.ProfilerActivity.CUDA)

        profile_memory = "MEM" in activities
        if not torch_activities and not profile_memory:
            return False, "no supported activities"

        # 配置并启动 PyTorch Profiler
        self.profiler = torch.profiler.profile(
            activities=torch_activities,
            with_stack=True if obj.with_stack is None else obj.with_stack,
            record_shapes=False if obj.record_shapes is None else obj.record_shapes,
            profile_memory=profile_memory,
        )
        self.profiler.start()
        self.steps_left = obj.num_steps
        logger.info(
            f"Encoder profiling started. output_dir={self.output_dir} profile_id={self.profile_id}"
        )
        return True, None

    def step(self):
        # 每次编码后推进 Profiler 一步，达到步数上限时自动停止
        if self.profiler is None:
            return
        self.profiler.step()
        if self.steps_left is not None:
            self.steps_left -= 1
            if self.steps_left <= 0:
                self.stop()

    def stop(self):
        # 停止 Profiler 并将 Chrome Trace 导出到文件
        if self.profiler is None:
            return False, "profiling not running"
        self.profiler.stop()
        filename = f"{self.prefix}-rank{self.rank}-{self.profile_id}.trace.json"
        trace_path = os.path.join(self.output_dir, filename)
        self.profiler.export_chrome_trace(trace_path)
        logger.info("Encoder profiling saved to: %s", trace_path)
        self.profiler = None
        self.steps_left = None
        return True, None


# FastAPI 应用实例，提供编码服务的 HTTP 接口
app = FastAPI()
# 全局编码器实例（rank 0 使用）
encoder: Optional[MMEncoder] = None
# 向非 rank 0 进程发送请求的 ZMQ PUSH socket 列表
send_sockets: List[zmq.Socket] = []


async def run_encoder(
    server_args: ServerArgs, schedule_path, dist_init_method, rank: int
):
    # 非 rank 0 进程的异步主循环：通过 ZMQ PULL socket 接收并处理编码请求
    encoder = MMEncoder(server_args, schedule_path, dist_init_method, rank)
    while True:
        request = await encoder.schedule_socket.recv_pyobj()
        if isinstance(request, ProfileReq):
            # 处理 Profiler 控制请求（启动/停止）
            if request.type == ProfileReqType.START_PROFILE:
                if encoder.profiler is None:
                    encoder.profiler = EncoderProfiler(encoder.rank)
                encoder.profiler.start(request)
            else:
                encoder.profiler.stop()
        else:
            # 处理普通编码请求（转发自 rank 0 的 HTTP 请求）
            if encoder.mm_global_cache is not None:
                await encoder.encode_with_global_cache(
                    mm_items=request["mm_items"],
                    modality=Modality.from_str(request["modality"]),
                    req_id=request["req_id"],
                    num_parts=request["num_parts"],
                    part_idx=request["part_idx"],
                    hashes=request.get("hashes", None),
                )
            else:
                await encoder.encode(
                    mm_items=request["mm_items"],
                    modality=Modality.from_str(request["modality"]),
                    req_id=request["req_id"],
                    num_parts=request["num_parts"],
                    part_idx=request["part_idx"],
                )


def launch_encoder(server_args, schedule_path, dist_init_method, rank):
    # 在子进程中启动非 rank 0 的编码器（通过 asyncio.run 运行异步主循环）
    try:
        asyncio.run(run_encoder(server_args, schedule_path, dist_init_method, rank))
    except KeyboardInterrupt:
        logger.info(f"Exit rank {rank}")
    except Exception:
        traceback.print_exc()


def launch_server(server_args: ServerArgs):
    # 服务器启动入口：创建各 rank 的编码器子进程，rank 0 作为主进程运行 FastAPI 服务
    global encoder
    ctx = mp.get_context("spawn")
    zmq_ctx = zmq.Context(10)
    ipc_path_prefix = random_uuid()
    port_args = PortArgs.init_new(server_args)
    # 解析分布式初始化地址
    if server_args.dist_init_addr:
        na = NetworkAddress.parse(server_args.dist_init_addr)
        dist_init_method = na.to_tcp()
    else:
        dist_init_method = NetworkAddress(
            server_args.host or "127.0.0.1", port_args.nccl_port
        ).to_tcp()
    # 为每个非 rank 0 的 TP rank 创建子进程和对应的 IPC ZMQ socket
    for rank in range(1, server_args.tp_size):
        schedule_path = f"ipc:///tmp/{ipc_path_prefix}_schedule_{rank}"
        send_sockets.append(
            get_zmq_socket(zmq_ctx, zmq.PUSH, schedule_path, bind=False)
        )
        ctx.Process(
            target=launch_encoder,
            args=(server_args, schedule_path, dist_init_method, rank),
            daemon=True,
        ).start()
    # rank 0 直接在主进程中创建编码器并启动 uvicorn HTTP 服务
    encoder = MMEncoder(server_args, dist_init_method=dist_init_method)
    uvicorn.run(app, host=server_args.host, port=server_args.port)


async def get_condition(rid):
    # 获取或创建指定请求 ID 的异步条件变量（线程安全）
    async with cond_dict_lock:
        if rid not in rid_to_cond:
            rid_to_cond[rid] = asyncio.Condition()
        return rid_to_cond[rid]


@app.post("/encode")
async def handle_encode_request(request: dict):
    # HTTP 编码请求处理器：广播到所有 TP rank，执行编码，按传输后端发送 embedding
    req_id = request["req_id"]
    try:

        def start_background_send(req_id):
            # 在后台异步发送 embedding（不阻塞当前请求响应）
            task = asyncio.create_task(encoder.send_with_url(req_id=req_id))
            encoder.background_tasks.add(task)
            task.add_done_callback(encoder.background_tasks.discard)

        # broadcast request
        # 将编码请求广播给所有非 rank 0 的 TP 进程，确保分布式推理同步
        request.update({"enter_time": time.time()})
        for socket in send_sockets:
            socket.send_pyobj(request)
        if encoder.mm_global_cache is not None:
            nbytes, embedding_len, embedding_dim, error_msg, error_code = (
                await encoder.encode_with_global_cache(
                    mm_items=request["mm_items"],
                    modality=Modality.from_str(request["modality"]),
                    req_id=request["req_id"],
                    num_parts=request["num_parts"],
                    part_idx=request["part_idx"],
                    hashes=request.get("hashes", None),
                )
            )
        else:
            nbytes, embedding_len, embedding_dim, error_msg, error_code = (
                await encoder.encode(
                    mm_items=request["mm_items"],
                    modality=Modality.from_str(request["modality"]),
                    req_id=request["req_id"],
                    num_parts=request["num_parts"],
                    part_idx=request["part_idx"],
                )
            )

        if error_msg:
            # 编码出错时，仍需将错误信息发送到 Prefill 端
            if encoder.server_args.encoder_transfer_backend == "zmq_to_scheduler":
                if request["embedding_port"] is None:
                    start_background_send(req_id)
                else:
                    for port in request["embedding_port"]:
                        await encoder.send(
                            req_id=req_id,
                            prefill_host=request["prefill_host"],
                            embedding_port=port,
                        )
            return ORJSONResponse(
                status_code=error_code,
                content={"status": "error", "message": error_msg, "req_id": req_id},
            )
        if encoder.server_args.encoder_transfer_backend == "mooncake":
            # Mooncake 传输：返回 embedding 元数据（大小、形状），实际数据通过 RDMA 传输
            del request["mm_items"]
            request.update(
                {
                    "embedding_size": nbytes,
                    "embedding_len": embedding_len,
                    "embedding_dim": embedding_dim,
                }
            )
            return ORJSONResponse(content=request)
        elif encoder.server_args.encoder_transfer_backend == "zmq_to_scheduler":
            # ZMQ 调度器模式：通过 ZMQ 将 embedding 发送到调度器指定的端口
            logger.info(f"{request['embedding_port'] = }")
            if request["embedding_port"] is None:
                # 动态端口模式：等待调度器注册接收 URL
                await encoder.send_with_url(
                    req_id=request["req_id"],
                )
            else:
                assert type(request["embedding_port"]) == list
                # 静态端口模式：向指定端口列表并发发送
                tasks = []
                for embedding_port in request["embedding_port"]:
                    tasks.append(
                        encoder.send(
                            req_id=request["req_id"],
                            prefill_host=request["prefill_host"],
                            embedding_port=embedding_port,
                        )
                    )
                await asyncio.gather(*tasks)
                encoder.embedding_to_send.pop(request["req_id"], None)
            return ORJSONResponse(content=None)
        elif encoder.server_args.encoder_transfer_backend == "zmq_to_tokenizer":
            # ZMQ Tokenizer 模式：直接发送到 Tokenizer 指定端口
            await encoder.send(
                req_id=request["req_id"],
                prefill_host=request["prefill_host"],
                embedding_port=request["embedding_port"],
            )
            encoder.embedding_to_send.pop(request["req_id"], None)
            return ORJSONResponse(content=None)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Unexpected error in encoder logic for {req_id}: {error_msg}")
        rid_to_err_msg[req_id] = error_msg
        return ORJSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": error_msg,
                "req_id": req_id,
            },
        )


@app.post("/send")
async def handle_send_request(request: dict):
    # mooncake backend
    # Mooncake 传输后端专用接口：RDMA 传输完成后发送确认消息
    await encoder.send(
        req_id=request["req_id"],
        prefill_host=request["prefill_host"],
        embedding_port=request["embedding_port"],
        session_id=request["session_id"],
        buffer_address=request["buffer_address"],
    )
    encoder.embedding_to_send.pop(request["req_id"], None)
    return ORJSONResponse(content=None)


@app.post("/scheduler_receive_url")
async def handle_scheduler_receive_url_request(request: dict):
    # 调度器注册接收 URL 接口：Decode 端告知编码器自己的接收地址
    # 编码器据此知道应将 embedding 发往何处
    rid = request["req_id"]
    async with rid_lock:
        global rid_to_receive_endpoint
        if rid not in rid_to_receive_endpoint:
            rid_to_receive_endpoint[rid] = set()
            rid_to_receive_count[rid] = request["receive_count"]
        assert rid_to_receive_count[rid] == request["receive_count"]
        rid_to_receive_endpoint[rid].add(request["receive_url"])
    # 通知等待新端点的 send_with_url 协程
    cond = await get_condition(rid)
    async with cond:
        cond.notify_all()


@app.get("/health")
@app.get("/health_generate")
async def health_generate():
    """
    Health check endpoint for the encoder server.
    Performs a dummy encode to verify the encoder is functional.
    Returns 200 if the encoder is healthy, 503 otherwise.
    编码器健康检查：通过执行一次虚拟编码验证编码器功能是否正常
    """
    if encoder is None:
        return Response(status_code=503)

    # Skip the dummy encode when real requests are already in flight — the
    # ongoing traffic already proves liveness, matching the scheduler's
    # `is_fully_idle`-based health-check skip pattern.
    # 若有真实请求正在处理中，跳过虚拟编码直接返回健康（避免资源竞争）
    if encoder.embedding_to_send:
        return Response(status_code=200)

    # Pick the first available modality for the dummy encode
    # 选择第一个可用的模态进行虚拟编码
    if encoder.image_processor is not None:
        mm_items = [f"data:image/png;base64,{MINIMUM_PNG_PICTURE_BASE64}"]
        modality = Modality.IMAGE
    elif encoder.audio_processor is not None:
        mm_items = [f"data:audio/wav;base64,{MINIMUM_WAV_SILENCE_BASE64}"]
        modality = Modality.AUDIO
    else:
        # No processor available, fall back to liveness check only
        return Response(status_code=200)

    try:
        req_id = f"{HEALTH_CHECK_RID_PREFIX}_{time.time()}"

        dummy_request = {
            "mm_items": mm_items,
            "modality": modality.name,
            "req_id": req_id,
            "num_parts": 1,
            "part_idx": 0,
        }

        # Broadcast to other TP ranks so distributed ops stay in sync
        # 广播给其他 TP rank 确保分布式操作同步
        for socket in send_sockets:
            socket.send_pyobj(dummy_request)

        # Run encode on rank 0 with timeout
        # 在 rank 0 上执行编码，设置超时时间
        _, _, _, error_msg, _ = await asyncio.wait_for(
            encoder.encode(
                mm_items=mm_items,
                modality=modality,
                req_id=req_id,
                num_parts=1,
                part_idx=0,
            ),
            timeout=HEALTH_CHECK_TIMEOUT,
        )

        # Clean up stored embedding
        # 清理健康检查产生的临时 embedding
        encoder.embedding_to_send.pop(req_id, None)

        if error_msg:
            logger.error(f"Encoder health check failed: {error_msg}")
            return Response(status_code=503)

        return Response(status_code=200)

    except asyncio.TimeoutError:
        logger.error(f"Encoder health check timed out after {HEALTH_CHECK_TIMEOUT}s")
        return Response(status_code=503)
    except Exception as e:
        logger.error(f"Encoder health check failed: {e}")
        return Response(status_code=503)


@app.api_route("/start_profile", methods=["GET", "POST"])
async def start_profile_async(obj: Optional[ProfileReqInput] = None):
    # 启动性能剖析接口：支持 GET/POST，可配置剖析参数
    if encoder is None:
        return Response(content="encoder not ready\n", status_code=503)
    req = None
    if obj is None:
        req = ProfileReq(ProfileReqType.START_PROFILE)
    else:
        req = ProfileReq(
            type=ProfileReqType.START_PROFILE,
            output_dir=obj.output_dir,
            start_step=obj.start_step,
            num_steps=obj.num_steps,
            activities=obj.activities,
            with_stack=obj.with_stack,
            record_shapes=obj.record_shapes,
            profile_by_stage=obj.profile_by_stage,
            profile_id=str(time.time()),
            merge_profiles=obj.merge_profiles,
            profile_prefix=obj.profile_prefix,
            profile_stages=obj.profile_stages,
        )
    # 广播给所有 TP rank 同步启动 Profiler
    for socket in send_sockets:
        socket.send_pyobj(req)
    if encoder.profiler is None:
        encoder.profiler = EncoderProfiler(encoder.rank)
    ok, msg = encoder.profiler.start(req)
    if ok:
        detail = (
            f"Start profiling. output_dir={encoder.profiler.output_dir} "
            f"profile_id={encoder.profiler.profile_id}\n"
        )
        return Response(content=detail, status_code=200)
    return Response(
        content=(msg or "Start profiling failed.\n"), status_code=HTTPStatus.BAD_REQUEST
    )


@app.api_route("/stop_profile", methods=["GET", "POST"])
async def stop_profile_async():
    # 停止性能剖析接口：广播停止命令并保存 trace 文件
    if encoder is None:
        return Response(content="encoder not ready\n", status_code=503)
    if encoder.profiler is None:
        return Response(
            content="profiling not initialized\n", status_code=HTTPStatus.BAD_REQUEST
        )
    req = ProfileReq(ProfileReqType.STOP_PROFILE)
    # 广播给所有 TP rank 同步停止 Profiler
    for socket in send_sockets:
        socket.send_pyobj(req)
    ok, msg = encoder.profiler.stop()
    if ok:
        return Response(content="Stop profiling.\n", status_code=200)
    return Response(
        content=(msg or "Stop profiling failed.\n"), status_code=HTTPStatus.BAD_REQUEST
    )
