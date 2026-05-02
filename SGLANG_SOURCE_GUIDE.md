# SGLang 源码导读：1 天达到 60% 认知水平

> 写给：对大模型推理有基础了解，但没有深入研究过推理框架源码的开发者
> 目标：理解 SGLang 的核心设计思想、关键代码路径、性能优化手段

---

## 第一部分：你需要先建立的基础概念（30 分钟）

在读代码之前，先建立几个心智模型。

### 1.1 LLM 推理的两个阶段

```
Prefill（预填充）：处理输入 prompt，一次性计算所有 token 的 KV cache
Decode（解码）：  每次生成一个 token，使用之前缓存的 KV cache
```

- Prefill 是**计算密集型**（compute-bound），大量并行矩阵乘法
- Decode 是**带宽密集型**（memory-bound），每步只生成 1 token 但要读取全部 KV cache
- 两者性能瓶颈完全不同，这是很多优化技术存在的根本原因

### 1.2 KV Cache 是什么

Transformer 的 Attention 计算需要所有历史 token 的 Key、Value 矩阵。
推理时把它们缓存起来（KV Cache），避免重复计算。
KV Cache 是显存的最大消耗者，也是推理框架的核心管理对象。

### 1.3 SGLang 解决的核心问题

| 问题 | SGLang 的解法 |
|---|---|
| 多请求共享系统 prompt，KV Cache 重复 | **RadixCache**：前缀 KV 共享 |
| GPU 在 decode 时利用率低 | **Chunked Prefill**：prefill+decode 混合批次 |
| 一次只生成 1 token 太慢 | **Speculative Decoding**：draft 模型预测多个 token |
| 调度 CPU 计算阻塞 GPU | **Overlap Scheduling**：CPU 调度与 GPU 计算并行 |
| 大模型无法单卡跑 | **Tensor Parallelism / Pipeline Parallelism** |
| KV Cache 显存不够用 | **Hierarchical Cache (HiCache)**：GPU→CPU→磁盘分级 |

---

## 第二部分：整体架构（1 小时）

### 2.1 仓库结构

```
sglang/
├── python/sglang/
│   ├── lang/          # 前端 DSL（用户写 sgl.gen()、sgl.select() 的地方）
│   ├── srt/           # Server Runtime —— 核心推理引擎（最重要）
│   ├── launch_server.py
│   └── bench_*.py     # 性能测试脚本
├── sgl-kernel/        # AOT 编译的 CUDA/C++ 内核
├── rust/              # Rust gRPC 服务器
└── test/              # 测试套件
```

**最重要的目录是 `python/sglang/srt/`**，这是本文的重点。

### 2.2 多进程运行时架构

SGLang 启动时会拉起**多个独立进程**，通过 **ZMQ** 消息队列通信：

```
                       ┌─────────────────────────────────────┐
HTTP 请求              │           主进程                     │
──────────────►  FastAPI HTTP Server                          │
                       │         ↓ async                      │
                       │  TokenizerManager                    │
                       │  (tokenize, manage sessions)         │
                       └──────────┬──────────────────────────┘
                                  │ ZMQ PUSH
                         ┌────────▼────────────────┐
                         │ DataParallelController  │  (仅 dp>1 时)
                         └────────┬────────────────┘
                                  │ ZMQ
               ┌──────────────────▼──────────────────────────┐
               │          Scheduler 子进程                     │
               │  (RadixCache + SchedulePolicy + Grammar)     │
               │              ↓                               │
               │       TpModelWorker                          │
               │              ↓ NCCL                          │
               │       ModelRunner (GPU)                      │
               │  (AttentionBackend + CudaGraph + LoRA)       │
               └──────────────────────────────────────────────┘
                                  │ ZMQ
                       ┌──────────▼──────────┐
                       │ DetokenizerManager  │
                       │ (token_ids → text)  │
                       └─────────────────────┘
```

**关键文件**：
- [python/sglang/srt/entrypoints/engine.py](python/sglang/srt/entrypoints/engine.py) — Engine 类，启动所有子进程
- [python/sglang/srt/entrypoints/http_server.py](python/sglang/srt/entrypoints/http_server.py) — FastAPI HTTP 服务器
- [python/sglang/srt/managers/tokenizer_manager.py](python/sglang/srt/managers/tokenizer_manager.py) — Tokenizer 管理器
- [python/sglang/srt/managers/scheduler.py](python/sglang/srt/managers/scheduler.py) — 调度器（核心）
- [python/sglang/srt/managers/detokenizer_manager.py](python/sglang/srt/managers/detokenizer_manager.py) — Detokenizer 管理器

### 2.3 一个请求的完整生命周期

```
1. POST /v1/chat/completions
   → http_server.py: FastAPI handler
   → tokenizer_manager.py: tokenize, 分配 rid, 管理 session

2. ZMQ → Scheduler
   → schedule_policy.py: 选择哪些请求进入本批次
   → radix_cache.py: match_prefix() 查找已有 KV 前缀
   → schedule_batch.py: 构造 ScheduleBatch

3. ScheduleBatch → ModelWorkerBatch → ForwardBatch
   → tp_worker.py: TpModelWorker.forward_batch()
   → model_runner.py: ModelRunner.forward_batch_generation()
   → attention backend: 执行 FlashInfer/Triton attention
   → sampler.py: 采样下一个 token

4. 生成 token ids → ZMQ → DetokenizerManager
   → 转换为文字 → ZMQ → TokenizerManager
   → SSE 流式返回给 HTTP 客户端
```

---

## 第三部分：核心模块深度解析（3 小时）

### 3.1 RadixCache：前缀共享的核心

**文件**：[python/sglang/srt/mem_cache/radix_cache.py](python/sglang/srt/mem_cache/radix_cache.py)

**核心思想**：用 Radix Tree（基数树）存储已计算的 KV Cache。
多个请求如果共享同一个前缀（例如相同的系统 prompt），它们的 KV Cache 只计算一次。

```
Tree 示例（token ids 作为 key）：
                  root
                 /    \
           [1,2,3]    [1,2,4]
           /    \          \
      [4,5]   [6,7]       [8,9]
        ↑         ↑           ↑
  req_A        req_B       req_C

req_A 和 req_B 共享前缀 [1,2,3] 的 KV Cache
```

**核心方法**：
```python
# match_prefix: 给定 token 序列，找到树中最长匹配前缀，返回已缓存的 KV indices
cache_result = radix_cache.match_prefix(
    MatchPrefixParams(rid=rid, key=token_ids, extra_key=lora_id)
)

# insert: 请求完成后，将新计算的 KV Cache 插入树
radix_cache.insert(InsertParams(rid=rid, key=all_token_ids, value=kv_indices))

# evict: LRU 策略，当显存不足时淘汰最少使用的 KV Cache 节点
radix_cache.evict(EvictParams(num_tokens=n))
```

**关键数据结构**：
- `TreeNode`：树节点，含 `key`（token ids）、`value`（KV cache indices）、`parent/children`
- `RadixKey`：`(token_ids_tuple, extra_key)` 作为查找 key，extra_key 用于 LoRA 区分

**相关文件**：
- [python/sglang/srt/mem_cache/memory_pool.py](python/sglang/srt/mem_cache/memory_pool.py) — 管理物理 KV 内存（`ReqToTokenPool` + `TokenToKVPoolAllocator`）
- [python/sglang/srt/mem_cache/evict_policy.py](python/sglang/srt/mem_cache/evict_policy.py) — LRU/LFU/FIFO 等淘汰策略
- [python/sglang/srt/mem_cache/hiradix_cache.py](python/sglang/srt/mem_cache/hiradix_cache.py) — 分级缓存（GPU→CPU→磁盘）

---

### 3.2 Scheduler：调度器

**文件**：[python/sglang/srt/managers/scheduler.py](python/sglang/srt/managers/scheduler.py)

Scheduler 是 SGLang 的大脑，负责决定：**每一步把哪些请求组成一个批次，如何调度 prefill 和 decode**。

Scheduler 是由多个 Mixin 组合而成的类：
```python
class Scheduler(
    SchedulerOutputProcessorMixin,   # 处理模型输出
    SchedulerUpdateWeightsMixin,     # 在线更新权重
    SchedulerProfilerMixin,          # 性能分析
    SchedulerMetricsMixin,           # Prometheus 指标
    SchedulerDisaggregationDecodeMixin,  # PD 解耦-解码侧
    SchedulerDisaggregationPrefillMixin, # PD 解耦-预填充侧
    ...
):
```

**核心调度循环**（简化）：
```python
while True:
    recv_requests()        # 从 ZMQ 接收新请求

    # 决定本轮的批次
    new_batch = get_next_batch_to_run()
    # └── schedule_policy.py: 选 decode 请求 + prefill 请求
    # └── 检查 KV cache 空间是否足够
    # └── 必要时 abort/preempt 低优先级请求

    # 发给 GPU 执行
    result = tp_worker.forward_batch(new_batch)

    process_batch_result(result)  # 更新 RadixCache，发送结果
```

**调度策略文件**：[python/sglang/srt/managers/schedule_policy.py](python/sglang/srt/managers/schedule_policy.py)
- `SchedulePolicy`：控制 prefill 和 decode 请求如何混合
- `PrefillAdder`：决定把哪些 prefill 请求加入当前批次（考虑内存、chunked prefill 等）
- LRU-based cache-aware ordering：优先调度能复用更多 KV Cache 的请求

---

### 3.3 批次数据结构三层抽象

**文件**：[python/sglang/srt/managers/schedule_batch.py](python/sglang/srt/managers/schedule_batch.py)

```
ScheduleBatch          # CPU 上，调度器视角
    ↓ to_model_worker_batch()
ModelWorkerBatch       # CPU→GPU 的桥梁
    ↓ prepare_forward_batch()
ForwardBatch           # GPU 上，模型执行视角（全是 torch.Tensor）
```

- `ScheduleBatch`：含 `List[Req]`，每个 `Req` 有完整的请求上下文（token ids、采样参数、KV 分配状态等）
- `ForwardBatch`：含 `input_ids`、`req_pool_indices`、`seq_lens`、`forward_mode` 等 GPU tensor

**ForwardMode 枚举**（[python/sglang/srt/model_executor/forward_batch_info.py](python/sglang/srt/model_executor/forward_batch_info.py)）：
```python
class ForwardMode(IntEnum):
    EXTEND       = ...  # Prefill（有 KV 前缀命中，扩展部分）
    DECODE       = ...  # Decode（每步生成 1 token）
    MIXED        = ...  # Chunked Prefill（prefill + decode 混合）
    IDLE         = ...  # DP 场景下的空闲 worker
    TARGET_VERIFY = ... # 投机解码：目标模型验证
    DRAFT_EXTEND  = ... # 投机解码：draft 模型扩展
    PREBUILT      = ... # PD 解耦：KV 已传输，直接 decode
```

---

### 3.4 ModelRunner：GPU 执行引擎

**文件**：[python/sglang/srt/model_executor/model_runner.py](python/sglang/srt/model_executor/model_runner.py)（约 3500 行，最核心的文件之一）

**主要职责**：
1. 初始化分布式环境（TP/PP 进程组，via `torch.distributed` + NCCL）
2. 加载模型权重（`model_loader/`）
3. 分配 KV Cache 物理内存（`memory_pool.py`）
4. 选择并初始化 Attention Backend
5. 创建并管理 CUDA Graph（用于 decode 加速）
6. 执行 `forward_batch_generation()` 和 `forward_batch_embedding()`

**内存分配逻辑**（关键）：
```python
# 启动时计算能分配多少 KV Cache
total_gpu_memory = get_total_gpu_memory()
available = total_gpu_memory * mem_fraction_static  # 默认 0.88
kv_cache_size = available - model_weights_size - activation_buffer_size
num_kv_tokens = kv_cache_size / (layers * heads * head_dim * 2 * 2)  # K+V, fp16
```

**CUDA Graph**：
- `CudaGraphRunner`：对固定 batch size 的 decode 做 CUDA Graph 捕获
- 捕获一次，之后 replay，消除 Python/CUDA API overhead
- 文件：[python/sglang/srt/model_executor/cuda_graph_runner.py](python/sglang/srt/model_executor/cuda_graph_runner.py)

---

### 3.5 Attention Backend：注意力计算的可插拔层

**文件**：[python/sglang/srt/layers/attention/](python/sglang/srt/layers/attention/)

模型中的每一个 Attention 层都是 `RadixAttention`，它把实际计算委托给 `AttentionBackend`：

```python
# layers/radix_attention.py
class RadixAttention(nn.Module):
    def forward(self, q, k, v, ...):
        return self.attn_backend.forward_extend(q, k, v, self, forward_batch)
        # 或
        return self.attn_backend.forward_decode(q, k, v, self, forward_batch)
```

**主要 Backend**：

| Backend | 适用场景 | 文件 |
|---|---|---|
| `flashinfer` | 默认，NVIDIA GPU | [flashinfer_backend.py](python/sglang/srt/layers/attention/flashinfer_backend.py) |
| `triton` | 通用 fallback | [triton_backend.py](python/sglang/srt/layers/attention/triton_backend.py) |
| `flashattention` | FA2/FA3 | [flashattention_backend.py](python/sglang/srt/layers/attention/flashattention_backend.py) |
| `flashinfer_mla` | DeepSeek MLA | [flashinfer_mla_backend.py](python/sglang/srt/layers/attention/flashinfer_mla_backend.py) |
| `aiter` | AMD GPU | [aiter_backend.py](python/sglang/srt/layers/attention/aiter_backend.py) |

**MLA（Multi-head Latent Attention）**：DeepSeek-V2/V3 的特殊注意力机制，将 KV 压缩到低秩空间，大幅降低 KV Cache 占用。SGLang 有专门的 MLA backend。

**AttentionBackend ABC**（[base_attn_backend.py](python/sglang/srt/layers/attention/base_attn_backend.py)）：
```python
class AttentionBackend:
    def init_forward_metadata(self, forward_batch): ...    # 每批次初始化
    def forward_extend(self, q, k, v, layer, forward_batch): ...  # prefill
    def forward_decode(self, q, k, v, layer, forward_batch): ...  # decode
```

---

### 3.6 Speculative Decoding：投机解码

**文件**：[python/sglang/srt/speculative/](python/sglang/srt/speculative/)

**核心思想**：用小的 draft 模型快速生成多个候选 token，然后用大的 target 模型并行验证，接受正确的 token。

```
Draft Model:  [tok_1] → [tok_2, tok_3, tok_4, tok_5]  （猜 4 个）
Target Model: 并行验证 [tok_2, tok_3, tok_4, tok_5]
              接受 tok_2, tok_3（正确），拒绝 tok_4（错误）
              → 本轮实际生成了 3 个 token（比普通 decode 快）
```

**算法**：

| 算法 | 文件 | 特点 |
|---|---|---|
| **EAGLE/EAGLE3** | [eagle_worker.py](python/sglang/srt/speculative/eagle_worker.py) | 训练一个预测下一个隐藏层特征的 draft head，接受率高 |
| **NGRAM** | [ngram_worker.py](python/sglang/srt/speculative/ngram_worker.py) | 从 prompt/历史中查找 n-gram，不需要额外模型 |
| **DFLASH** | [dflash_worker.py](python/sglang/srt/speculative/dflash_worker.py) | 专用于 flashinfer 的投机解码 |
| **Standalone** | [standalone_worker.py](python/sglang/srt/speculative/standalone_worker.py) | 独立的小模型作为 draft |

---

### 3.7 Chunked Prefill：混合批次

**ForwardMode.MIXED**：一个批次中同时包含 prefill（extend）token 和 decode token。

**为什么需要**：
- 纯 decode 批次：GPU 利用率低（每次只生成 1 token/请求）
- 纯 prefill 批次：decode 请求等待时延高
- 混合批次：把 prefill 分成小块（chunk），与 decode 混合，均衡 GPU 利用率

**配置**：
```bash
--chunked-prefill-size 8192   # 每批次最多处理多少个 prefill tokens
```

---

### 3.8 PD 解耦（Disaggregation）

**文件**：[python/sglang/srt/disaggregation/](python/sglang/srt/disaggregation/)

**思想**：把 Prefill 和 Decode 分到不同的 GPU 实例上跑。
- Prefill 实例：专门做计算密集的预填充
- Decode 实例：专门做内存密集的解码
- KV Cache 通过 NIXL/Mooncake/MoRI 等高速传输

**传输 Backend**：
- `nixl/`：NVIDIA NIXL (高速 KV 传输)
- `mooncake/`：月之暗面开发的传输框架
- `mori/`：另一种传输框架

---

### 3.9 Expert Parallelism & EPLB（针对 MoE 模型）

**文件**：[python/sglang/srt/eplb/](python/sglang/srt/eplb/)

MoE（Mixture of Experts）模型（如 DeepSeek-V3）有大量 Expert，需要跨 GPU 分布。

- **Expert Parallelism (EP)**：将不同 Expert 放在不同 GPU 上
- **EPLB（Expert Parallel Load Balancing）**：运行时根据 Expert 实际负载动态重新分配，避免热点 Expert 造成的负载不均

---

## 第四部分：前端 DSL（30 分钟，可选）

**文件**：[python/sglang/lang/](python/sglang/lang/)

SGLang 除了提供 OpenAI 兼容的 HTTP API，还提供了一个 Python DSL：

```python
import sglang as sgl

@sgl.function
def multi_turn_qa(s, question):
    s += sgl.system("你是一个助手")
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=200))

    s += sgl.user("请总结你的答案")
    s += sgl.assistant(sgl.gen("summary", max_tokens=50))

# 执行
state = multi_turn_qa.run(question="什么是 KV Cache?")
print(state["answer"])
```

**核心文件**：
- [python/sglang/lang/api.py](python/sglang/lang/api.py) — `gen`, `select`, `fork`, `image` 等 DSL 函数
- [python/sglang/lang/ir.py](python/sglang/lang/ir.py) — 中间表示（IR）
- [python/sglang/lang/interpreter.py](python/sglang/lang/interpreter.py) — DSL 执行器
- [python/sglang/lang/backend/runtime_endpoint.py](python/sglang/lang/backend/runtime_endpoint.py) — 连接到正在运行的 SGLang server

---

## 第五部分：量化与模型支持（30 分钟）

### 5.1 量化支持

**文件**：[python/sglang/srt/layers/quantization/](python/sglang/srt/layers/quantization/)

支持的量化格式：
- `fp8`：FP8 权重（W8A8 FP8）
- `fp4` / `nvfp4`：FP4 权重
- `awq`：AWQ 4-bit 量化
- `gptq`：GPTQ 4-bit 量化
- `bnb`：BitsAndBytes 4-bit/8-bit
- `gguf`：GGUF 格式（llama.cpp 格式）
- `mxfp8`：Microscaling FP8

### 5.2 模型支持

**文件**：[python/sglang/srt/models/](python/sglang/srt/models/)

184 个模型文件，重点模型：
- `llama.py`：Llama / Llama2 / Llama3
- `qwen2.py`、`qwen3.py`、`qwen2_moe.py`：Qwen 系列
- `deepseek_v2.py`、`deepseek_v3.py`：DeepSeek V2/V3（含 MLA + MoE）
- `gemma2.py`、`gemma3.py`：Gemma 系列
- `mistral.py`、`mixtral.py`：Mistral/Mixtral
- `phi3.py`：Phi-3

每个模型文件实现：
1. 模型的 `nn.Module`，使用 `RadixAttention` + SGLang 并行线性层
2. `load_weights` 方法（从 HuggingFace checkpoint 加载）
3. 注册到 `registry.py`（通过 `@register_model` 装饰器）

---

## 第六部分：sgl-kernel — 底层 CUDA 内核（30 分钟）

**目录**：[sgl-kernel/](sgl-kernel/)

与 `jit_kernel/`（Triton JIT 运行时编译）不同，`sgl-kernel` 是**预编译的 CUDA/C++ 内核**：

```
sgl-kernel/csrc/
├── attention/      # Flash Attention 变体
├── allreduce/      # 自定义 All-Reduce（比 NCCL 更快的小 tensor）
├── gemm/           # FP8/FP4 GEMM 内核
├── moe/            # MoE 内核（token routing + expert GEMM）
├── quantization/   # 量化/反量化内核
├── kvcacheio/      # KV Cache I/O（用于 PD 解耦传输）
└── mamba/          # Mamba SSM 内核
```

Python 绑定通过 `torch.ops` 注册，在 Python 层通过 `sgl_kernel.xxx()` 调用。

---

## 第七部分：一天学习路线图

### 上午（4 小时）：建立全局视图

1. **读 README**（30 分钟）：[README.md](README.md) — 了解功能和性能数字
2. **读架构注释**（30 分钟）：
   - [python/sglang/srt/managers/schedule_batch.py](python/sglang/srt/managers/schedule_batch.py) 的文件头注释（ScheduleBatch → ModelWorkerBatch → ForwardBatch 流程）
   - [python/sglang/srt/entrypoints/engine.py](python/sglang/srt/entrypoints/engine.py) 的文件头注释
3. **读 Engine 启动**（1 小时）：
   - [python/sglang/srt/entrypoints/engine.py](python/sglang/srt/entrypoints/engine.py) 的 `__init__` 和 `_launch_subprocesses`
   - 理解 ZMQ 如何连接各组件
4. **读 Scheduler 主循环**（1 小时）：
   - [python/sglang/srt/managers/scheduler.py](python/sglang/srt/managers/scheduler.py) 的 `__init__` 和 `event_loop`
5. **读 RadixCache**（1 小时）：
   - [python/sglang/srt/mem_cache/radix_cache.py](python/sglang/srt/mem_cache/radix_cache.py)
   - 重点：`match_prefix`, `insert`, `evict`，理解 TreeNode 结构

### 下午（4 小时）：深入关键模块

6. **读 ForwardBatch 和 ForwardMode**（45 分钟）：
   - [python/sglang/srt/model_executor/forward_batch_info.py](python/sglang/srt/model_executor/forward_batch_info.py)
7. **读 ModelRunner 的关键方法**（1.5 小时）：
   - [python/sglang/srt/model_executor/model_runner.py](python/sglang/srt/model_executor/model_runner.py)
   - 重点：`init_memory_pool`, `forward_batch_generation`，CUDA Graph 捕获部分
8. **读一个 Attention Backend**（1 小时）：
   - [python/sglang/srt/layers/attention/flashinfer_backend.py](python/sglang/srt/layers/attention/flashinfer_backend.py)
   - 重点：`init_forward_metadata`, `forward_extend`, `forward_decode`
9. **读一个模型实现**（45 分钟）：
   - [python/sglang/srt/models/llama.py](python/sglang/srt/models/llama.py)（最标准）
   - 看 `LlamaAttention` 如何使用 `RadixAttention`
   - 看 `load_weights` 如何映射 HuggingFace 权重

### 晚上（2 小时）：选读进阶主题

根据兴趣选 1-2 个：

- **投机解码**：[python/sglang/srt/speculative/eagle_worker.py](python/sglang/srt/speculative/eagle_worker.py)
- **量化**：[python/sglang/srt/layers/quantization/](python/sglang/srt/layers/quantization/)
- **DeepSeek MLA**：[python/sglang/srt/layers/attention/flashinfer_mla_backend.py](python/sglang/srt/layers/attention/flashinfer_mla_backend.py)
- **PD 解耦**：[python/sglang/srt/disaggregation/](python/sglang/srt/disaggregation/)
- **性能分析**：运行 `python -m sglang.bench_serving --help`

---

## 第八部分：常用问题索引

**Q: SGLang 如何选择 Attention Backend？**
A: 启动参数 `--attention-backend flashinfer|triton|flashattention|flashmla|...`，默认 `flashinfer`。
代码入口：[python/sglang/srt/model_executor/model_runner.py](python/sglang/srt/model_executor/model_runner.py) 中 `init_attention_backend()`

**Q: KV Cache 有多大？如何控制？**
A: 由 `--mem-fraction-static`（默认 0.88）控制，即 GPU 显存的 88% 用于模型 + KV Cache。
代码：`ModelRunner.init_memory_pool()` → `memory_pool.py`

**Q: Tensor Parallelism 如何工作？**
A: [python/sglang/srt/distributed/parallel_state.py](python/sglang/srt/distributed/parallel_state.py)，每个 TP rank 加载不同的权重分片，通过 NCCL All-Reduce 同步 Attention 输出。
线性层：[python/sglang/srt/layers/linear.py](python/sglang/srt/layers/linear.py) 中的 `ColumnParallelLinear`/`RowParallelLinear`

**Q: 如何添加新模型？**
A: 在 [python/sglang/srt/models/](python/sglang/srt/models/) 新建文件，继承标准接口，实现 `load_weights`，在 [registry.py](python/sglang/srt/models/registry.py) 注册。参考 `llama.py`。

**Q: 请求如何做约束解码（JSON Schema 等）？**
A: [python/sglang/srt/constrained/](python/sglang/srt/constrained/)，支持 XGrammar/LLGuidance/Outlines 三种 backend。
通过 `--grammar-backend xgrammar` 选择。

**Q: 如何做在线 LoRA 服务？**
A: [python/sglang/srt/lora/](python/sglang/srt/lora/)，启动时 `--enable-lora`，支持同一批次中混合多个 LoRA adapter。

---

## 附录：关键配置参数速查

```bash
# 基础启动
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3-8B-Instruct \
  --host 0.0.0.0 \
  --port 30000

# 性能调优
  --tp-size 4                    # Tensor Parallelism
  --dp-size 2                    # Data Parallelism（复制多份）
  --mem-fraction-static 0.90     # KV Cache 内存占比
  --chunked-prefill-size 8192    # Chunked Prefill chunk 大小
  --attention-backend flashinfer # Attention 计算后端

# 量化
  --quantization fp8             # FP8 量化
  --quantization awq             # AWQ INT4 量化

# 投机解码
  --speculative-algorithm EAGLE  # EAGLE 投机解码
  --speculative-draft-model-path ./eagle_model

# 约束解码
  --grammar-backend xgrammar     # JSON Schema 约束解码

# LoRA
  --enable-lora                  # 启用多 LoRA 服务
  --max-loras-per-batch 4

# 调试
  --log-level debug
  --disable-cuda-graph           # 关闭 CUDA Graph（方便调试）
```

---

## 附录：与 vLLM 的主要区别

| 特性 | SGLang | vLLM |
|---|---|---|
| KV Cache 共享 | Radix Tree（前缀感知） | PagedAttention（页式） |
| 前端 DSL | 有（sgl.gen, sgl.select） | 无 |
| MLA 支持 | 原生支持（DeepSeek V2/V3） | 支持 |
| Overlap Scheduling | 有（prefill-decode overlap） | 部分 |
| 投机解码算法 | EAGLE/EAGLE3/NGRAM/DFLASH | EAGLE/NGRAM等 |
| 约束解码 | XGrammar/LLGuidance/Outlines | Outlines/Guidance |
| PD 解耦 | 原生支持（NIXL/Mooncake/MoRI） | 部分支持 |
| EPLB | 有（运行时 Expert 负载均衡） | 部分 |
| AMD GPU | 有（AITER/WAVE backend） | 有 |

---

*本文档基于 SGLang 仓库 `/Users/heyu11/Code/sglang` 源码生成。如有疑问，欢迎继续提问。*
