"""
gRPC Encoder Server for SGLang EPD (Encode-Prefill-Decode) mode.

This server provides gRPC-based encoding for multimodal inputs.

Usage:
    python -m sglang.launch_server --model-path <model> --encoder-only --grpc-mode
"""
# EPD（Encode-Prefill-Decode）模式下的 gRPC 编码服务器
# 负责将多模态输入（图像等）编码为 embedding，并通过 mooncake/zmq 传输到 prefill 节点

import asyncio
import logging
import multiprocessing as mp
import traceback
from concurrent import futures
from typing import List

import grpc
import zmq
import zmq.asyncio
# gRPC 健康检查协议，用于 Kubernetes 探针
from grpc_health.v1 import health_pb2, health_pb2_grpc
# gRPC 反射，方便客户端发现服务接口
from grpc_reflection.v1alpha import reflection
# SGLang 编码器 gRPC protobuf 定义
from smg_grpc_proto import sglang_encoder_pb2, sglang_encoder_pb2_grpc

from sglang.srt.disaggregation.encode_server import (
    MMEncoder,
    handle_scheduler_receive_url_request,
    launch_encoder,
)
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import random_uuid
from sglang.srt.utils.network import NetworkAddress, get_zmq_socket

logger = logging.getLogger(__name__)
# 简化 gRPC Servicer 基类引用
SGLangEncoderServicer = sglang_encoder_pb2_grpc.SglangEncoderServicer
add_SGLangEncoderServicer_to_server = (
    sglang_encoder_pb2_grpc.add_SglangEncoderServicer_to_server
)


# gRPC 健康检查服务实现，支持 Kubernetes liveness/readiness 探针
class EncoderHealthServicer(health_pb2_grpc.HealthServicer):
    """
    Standard gRPC health check service for encoder server.
    Implements grpc.health.v1.Health for Kubernetes probes.
    """

    OVERALL_SERVER = ""
    ENCODER_SERVICE = "sglang.grpc.encoder.SglangEncoder"

    def __init__(self):
        # 初始化时服务为非就绪状态，等待编码器完成初始化后切换
        self._serving = False

    def set_serving(self):
        # 标记服务为就绪（编码器初始化完成后调用）
        self._serving = True

    def set_not_serving(self):
        # 标记服务为非就绪（关闭时调用）
        self._serving = False

    async def Check(self, request, context) -> health_pb2.HealthCheckResponse:
        # 返回当前服务健康状态：SERVING 或 NOT_SERVING
        if self._serving:
            return health_pb2.HealthCheckResponse(
                status=health_pb2.HealthCheckResponse.SERVING
            )
        return health_pb2.HealthCheckResponse(
            status=health_pb2.HealthCheckResponse.NOT_SERVING
        )

    async def Watch(self, request, context):
        # Watch 接口：流式返回健康状态变化（此处简单返回当前状态）
        yield await self.Check(request, context)


# gRPC 编码器服务实现：接收 Encode/Send/SchedulerReceiveUrl 请求并分发给 MMEncoder
class SGLangEncoderServer(SGLangEncoderServicer):
    """
    gRPC service implementation for SGLang encoder.
    """

    def __init__(
        self,
        encoder: MMEncoder,
        send_sockets: List[zmq.Socket],
        server_args: ServerArgs,
    ):
        # 多模态编码器实例
        self.encoder = encoder
        # 向其他 TP rank 转发请求的 ZMQ PUSH socket 列表
        self.send_sockets = send_sockets
        self.server_args = server_args

    async def Encode(
        self, request: sglang_encoder_pb2.EncodeRequest, context
    ) -> sglang_encoder_pb2.EncodeResponse:
        # 处理编码请求：编码多模态输入并根据传输后端将 embedding 发送到 prefill 节点
        try:
            # 构建请求字典并广播给所有其他 TP rank（通过 ZMQ PUSH）
            request_dict = {
                "mm_items": list(request.mm_items),
                "req_id": request.req_id,
                "num_parts": request.num_parts,
                "part_idx": request.part_idx,
            }
            for socket in self.send_sockets:
                await socket.send_pyobj(request_dict)

            # gRPC encode is image-only; encoder.encode() requires modality
            # 调用编码器执行多模态编码，目前 gRPC 接口仅支持图像模态
            (
                nbytes,
                embedding_len,
                embedding_dim,
                error_msg,
                error_code,
            ) = await self.encoder.encode(
                mm_items=list(request.mm_items),
                modality=Modality.IMAGE,
                req_id=request.req_id,
                num_parts=request.num_parts,
                part_idx=request.part_idx,
            )
            if error_msg is not None:
                # 编码失败，设置 gRPC 错误码并返回空响应
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(error_msg)
                return sglang_encoder_pb2.EncodeResponse()

            if self.server_args.encoder_transfer_backend == "mooncake":
                # Mooncake 传输后端：返回 embedding 大小元数据，由 prefill 端主动拉取数据
                return sglang_encoder_pb2.EncodeResponse(
                    embedding_size=nbytes,
                    embedding_len=embedding_len,
                    embedding_dim=embedding_dim,
                )
            elif self.server_args.encoder_transfer_backend == "zmq_to_scheduler":
                # ZMQ 到 scheduler 后端：通过 ZMQ 将 embedding 推送到 prefill scheduler
                embedding_ports = list(request.embedding_port)
                logger.info(f"embedding_port = {embedding_ports}")
                if not embedding_ports:
                    # 无指定端口时使用 URL 方式发送
                    await self.encoder.send_with_url(req_id=request.req_id)
                else:
                    # 并发向所有目标端口发送 embedding
                    tasks = []
                    for embedding_port in embedding_ports:
                        tasks.append(
                            self.encoder.send(
                                req_id=request.req_id,
                                prefill_host=request.prefill_host,
                                embedding_port=embedding_port,
                            )
                        )
                    await asyncio.gather(*tasks)
                    # 发送完成后清理本地缓存
                    self.encoder.embedding_to_send.pop(request.req_id, None)
                return sglang_encoder_pb2.EncodeResponse()
            elif self.server_args.encoder_transfer_backend == "zmq_to_tokenizer":
                # ZMQ 到 tokenizer 后端：将 embedding 发送到 tokenizer 侧
                embedding_port = (
                    request.embedding_port[0] if request.embedding_port else 0
                )
                await self.encoder.send(
                    req_id=request.req_id,
                    prefill_host=request.prefill_host,
                    embedding_port=embedding_port,
                )
                self.encoder.embedding_to_send.pop(request.req_id, None)
                return sglang_encoder_pb2.EncodeResponse()

            return sglang_encoder_pb2.EncodeResponse()

        except Exception as e:
            # 统一异常处理：记录日志并返回 gRPC INTERNAL 错误
            logger.error(f"Encode error: {e}")
            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return sglang_encoder_pb2.EncodeResponse()

    async def Send(
        self, request: sglang_encoder_pb2.SendRequest, context
    ) -> sglang_encoder_pb2.SendResponse:
        # 显式 Send 请求：将已编码的 embedding 发送到指定 prefill 节点
        try:
            await self.encoder.send(
                req_id=request.req_id,
                prefill_host=request.prefill_host,
                embedding_port=request.embedding_port,
                session_id=request.session_id if request.session_id else None,
                buffer_address=(
                    request.buffer_address if request.buffer_address else None
                ),
            )
            # 发送完成后清理本地 embedding 缓存
            self.encoder.embedding_to_send.pop(request.req_id, None)
            return sglang_encoder_pb2.SendResponse()

        except Exception as e:
            logger.error(f"Send error: {e}")
            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return sglang_encoder_pb2.SendResponse()

    async def SchedulerReceiveUrl(
        self, request: sglang_encoder_pb2.SchedulerReceiveUrlRequest, context
    ) -> sglang_encoder_pb2.SchedulerReceiveUrlResponse:
        # 通知编码器 scheduler 的接收 URL，用于 zmq_to_scheduler 传输模式
        try:
            await handle_scheduler_receive_url_request(
                {
                    "req_id": request.req_id,
                    "receive_count": request.receive_count,
                    "receive_url": request.receive_url,
                }
            )
            return sglang_encoder_pb2.SchedulerReceiveUrlResponse()

        except Exception as e:
            logger.error(f"SchedulerReceiveUrl error: {e}")
            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return sglang_encoder_pb2.SchedulerReceiveUrlResponse()


async def serve_grpc_encoder(server_args: ServerArgs):
    # 启动 gRPC 编码器服务：初始化多进程编码器、构建 gRPC 服务并开始监听
    ctx = mp.get_context("spawn")
    zmq_ctx = zmq.asyncio.Context(10)
    # 使用随机 UUID 作为 IPC 路径前缀，避免多实例冲突
    ipc_path_prefix = random_uuid()
    port_args = PortArgs.init_new(server_args)

    # 解析或构建 dist_init 地址，用于各 TP rank 间的 NCCL 初始化
    if server_args.dist_init_addr:
        na = NetworkAddress.parse(server_args.dist_init_addr)
        dist_init_method = na.to_tcp()
    else:
        dist_init_method = NetworkAddress(
            server_args.host or "127.0.0.1", port_args.nccl_port
        ).to_tcp()

    # 为 TP rank 1..N-1 创建 ZMQ PUSH socket 和子进程编码器
    send_sockets: List[zmq.Socket] = []
    for rank in range(1, server_args.tp_size):
        schedule_path = f"ipc:///tmp/{ipc_path_prefix}_schedule_{rank}"
        send_sockets.append(
            get_zmq_socket(zmq_ctx, zmq.PUSH, schedule_path, bind=False)
        )
        # 以 spawn 方式启动各 TP rank 的编码器子进程
        ctx.Process(
            target=launch_encoder,
            args=(server_args, schedule_path, dist_init_method, rank),
            daemon=True,
        ).start()

    # 主 rank（rank 0）的编码器实例
    encoder = MMEncoder(server_args, dist_init_method=dist_init_method)

    # 创建异步 gRPC 服务器，配置消息大小上限为 256MB
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 1024 * 1024 * 256),
            ("grpc.max_receive_message_length", 1024 * 1024 * 256),
        ],
    )

    # 注册健康检查服务
    health_servicer = EncoderHealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    # 注册编码器服务
    encoder_servicer = SGLangEncoderServer(
        encoder=encoder,
        send_sockets=send_sockets,
        server_args=server_args,
    )
    add_SGLangEncoderServicer_to_server(encoder_servicer, server)

    # 启用 gRPC 反射，方便客户端工具（如 grpcurl）发现服务
    SERVICE_NAMES = (
        sglang_encoder_pb2.DESCRIPTOR.services_by_name["SglangEncoder"].full_name,
        "grpc.health.v1.Health",
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    # 绑定监听地址并启动服务器
    listen_addr = NetworkAddress(server_args.host, server_args.port).to_host_port_str()
    server.add_insecure_port(listen_addr)

    await server.start()
    logger.info(f"gRPC encoder server listening on {listen_addr}")

    # 服务器就绪后标记健康检查为 SERVING
    health_servicer.set_serving()

    try:
        # 阻塞等待服务终止
        await server.wait_for_termination()
    except KeyboardInterrupt:
        # 收到中断信号时优雅关闭，等待最多 5 秒
        logger.info("Shutting down gRPC encoder server...")
        health_servicer.set_not_serving()
        await server.stop(grace=5)
