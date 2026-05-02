# SGLang 数据流图 + 源码注释导读

> 阅读策略：先看数据流图建立全局感知，再看注释版源码理解细节。
> 文件中所有 `// ← 注释` 是为你加的学习注释，原始源码中没有。

---

## 一、最重要的一张图：请求全链路数据流

```
                        ┌──────────────────────────────────────────────────────────────┐
                        │                        主进程                                 │
                        │                                                              │
  POST /v1/chat  ──────►│  FastAPI                                                     │
                        │     │                                                        │
                        │     ▼ async                                                  │
                        │  TokenizerManager                                            │
                        │  ┌─────────────────────────────────────────────┐            │
                        │  │ 1. text → token_ids  (HuggingFace tokenizer)│            │
                        │  │ 2. 构造 GenerateReqInput                     │            │
                        │  │ 3. 分配 rid（请求唯一 ID）                    │            │
                        │  └──────────────────┬──────────────────────────┘            │
                        └─────────────────────┼────────────────────────────────────── ┘
                                              │ ZMQ PUSH（序列化的 Python 对象）
                        ┌─────────────────────▼────────────────────────────────────── ┐
                        │                   Scheduler 子进程                            │
                        │                                                              │
                        │  ┌─── event_loop() 主循环 ──────────────────────────────┐   │
                        │  │                                                       │   │
                        │  │  ① recv_requests()        接收新请求进 waiting_queue  │   │
                        │  │         │                                             │   │
                        │  │  ② get_next_batch_to_run()  调度决策                 │   │
                        │  │         │                                             │   │
                        │  │         ▼                                             │   │
                        │  │    RadixCache.match_prefix()  查命中的 KV 前缀        │   │
                        │  │         │                                             │   │
                        │  │         ▼                                             │   │
                        │  │    ScheduleBatch  (CPU 数据结构)                      │   │
                        │  │         │ .to_model_worker_batch()                   │   │
                        │  │         ▼                                             │   │
                        │  │    ModelWorkerBatch  (CPU→GPU 桥梁)                   │   │
                        │  │         │                                             │   │
                        │  │  ③ tp_worker.forward_batch_generation()              │   │
                        │  │         │                                             │   │
                        │  │         ▼                                             │   │
                        │  │    ModelRunner.forward_batch_generation()             │   │
                        │  │         │                                             │   │
                        │  │         ▼                                             │   │
                        │  │    ForwardBatch  (GPU Tensor)                        │   │
                        │  │         │ AttentionBackend.forward_extend/decode()   │   │
                        │  │         │ → FlashInfer / Triton 执行真正的 Attention  │   │
                        │  │         │                                             │   │
                        │  │         ▼                                             │   │
                        │  │    Sampler  采样下一个 token                          │   │
                        │  │         │                                             │   │
                        │  │  ④ process_batch_result()                            │   │
                        │  │         │ ├── 未结束：更新 Req.output_ids             │   │
                        │  │         │ └── 结束：RadixCache.cache_finished_req()  │   │
                        │  └─────────┼─────────────────────────────────────────── ┘   │
                        └────────── ┼──────────────────────────────────────────────── ┘
                                    │ ZMQ PUSH（output token ids）
                        ┌───────────▼──────────────────────────────────────────────── ┐
                        │              DetokenizerManager 子进程                        │
                        │   token_ids → 文字（增量 detokenize）                         │
                        └───────────────────────┬──────────────────────────────────── ┘
                                                │ ZMQ PUSH
                        ┌───────────────────────▼──────────────────────────────────── ┐
                        │  TokenizerManager（主进程）                                   │
                        │   → SSE 流式 / 一次性返回 HTTP Response                       │
                        └────────────────────────────────────────────────────────────── ┘
```

---

## 二、核心数据结构三层变换

这是理解 SGLang 代码的**第一关**，整个推理流水线数据就沿着这三层流转：

```
                    CPU（调度器）                CPU→GPU              GPU（模型）
                  ┌─────────────┐           ┌──────────────┐     ┌─────────────┐
                  │ScheduleBatch│  ────────► │ModelWorker   │───► │ForwardBatch │
                  │             │.to_model   │Batch         │     │             │
                  │ List[Req]   │_worker     │              │     │ Tensor-only │
                  │ 调度元信息   │_batch()    │ CPU 子集     │     │ GPU Tensors │
                  │ 树缓存引用   │            │ 去掉调度信息  │     │ input_ids   │
                  │ 采样参数    │            │ 保留模型需要  │     │ seq_lens    │
                  └─────────────┘           └──────────────┘     │ out_cache_loc│
                                                                  └─────────────┘

  源码位置：
  ScheduleBatch    → schedule_batch.py:1350
  ModelWorkerBatch → schedule_batch.py:2689
  ForwardBatch     → forward_batch_info.py（独立文件）
```

---

## 三、带注释的核心源码

### 3.1 三层数据结构（schedule_batch.py）

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 文件：python/sglang/srt/managers/schedule_batch.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ─── 层1：单个请求的完整状态 ─────────────────────────────
class Req(ReqDllmMixin):           # ← 继承 dLLM mixin，正常情况只看基础字段
    def __init__(self, rid, origin_input_text, origin_input_ids, sampling_params, ...):

        self.rid = rid                  # ← 请求唯一 ID（字符串）
        self.origin_input_ids = origin_input_ids  # ← 原始输入 token ids（来自 tokenizer）
        self.output_ids = []            # ← 已生成的 token ids（每 decode 步追加一个）
        self.fill_ids = []              # ← = origin_input_ids + output_ids（当前全序列）

        # KV Cache 相关 —— 理解这三个字段是理解内存管理的关键
        self.kv_committed_len = 0       # ← 已经"提交"到 RadixTree 的 KV 长度
        self.kv_allocated_len = 0       # ← 已经在物理内存池中分配的 KV 长度

        # 调度状态
        self.req_pool_idx = None        # ← 在 ReqToTokenPool 中的行号（分配后赋值）
        self.last_node = None           # ← 在 RadixTree 中匹配到的最后节点（用于 lock/unlock）
        self.cache_protected_len = 0    # ← 这之前的 KV 由 RadixTree 管理，不能随意释放

        # 采样参数（temperature、top_p、max_new_tokens 等都在这里）
        self.sampling_params: SamplingParams = sampling_params


# ─── 层2：一整个批次的调度视图（CPU 上） ──────────────────
class ScheduleBatch(ScheduleBatchDisaggregationDecodeMixin):
    reqs: List[Req]                     # ← 本批次所有请求，核心字段

    # 内存/缓存句柄（不是数据，是"管理器"引用）
    req_to_token_pool: ReqToTokenPool       # ← 管理"请求→token槽位"的映射
    token_to_kv_pool_allocator: ...         # ← 管理物理 KV Cache 内存页的分配器
    tree_cache: BasePrefixCache             # ← RadixTree 引用

    # 批次模式 —— 最重要的字段之一
    forward_mode: ForwardMode           # ← EXTEND/DECODE/MIXED，决定 attention 怎么算

    # GPU Tensor（由 prepare_for_extend/decode 填充）
    input_ids: torch.Tensor             # ← shape: [num_tokens], 本批次所有输入 token
    req_pool_indices: torch.Tensor      # ← shape: [batch_size], 每个请求在内存池中的行
    seq_lens: torch.Tensor              # ← shape: [batch_size], 每个请求当前序列长度
    out_cache_loc: torch.Tensor         # ← shape: [num_tokens], 新 token 的 KV 写入位置


# ─── 层3：TP Worker 视图（只含模型需要的字段） ──────────────
class ModelWorkerBatch:
    forward_mode: ForwardMode           # ← 透传
    input_ids: torch.Tensor             # ← 透传
    req_pool_indices: torch.Tensor      # ← 透传
    seq_lens: torch.Tensor              # ← 透传
    out_cache_loc: torch.Tensor         # ← 透传（新 KV 写哪里）

    # extend（prefill）专用字段
    extend_num_tokens: Optional[int]    # ← prefill 总 token 数
    extend_seq_lens: Optional[List[int]]     # ← 每个请求 extend 多少 token
    extend_prefix_lens: Optional[List[int]]  # ← 每个请求已命中 prefix 多少 token

    # LoRA
    lora_ids: Optional[List[str]]       # ← 每个请求用的 LoRA adapter id

    sampling_info: SamplingBatchInfo    # ← 批次采样参数（temperature等打包成GPU tensor）
```

---

### 3.2 ForwardMode：决定 Attention 怎么算（forward_batch_info.py）

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 文件：python/sglang/srt/model_executor/forward_batch_info.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ForwardMode(IntEnum):

    EXTEND = auto()
    # ↑ "扩展"模式 = Prefill
    # 请求刚来，KV Cache 的前缀部分已经命中（来自 RadixCache），
    # 只需要计算未命中的后段 token 的 KV。
    # Attention 需要 attend to 所有历史 token（包括缓存的前缀）。

    DECODE = auto()
    # ↑ 解码模式，每步生成 1 个 token。
    # 用 PagedAttention 风格：只计算新 token 的 Q，
    # attend to 历史所有 KV（从 KV Cache 读）。
    # GPU 利用率低，是 memory-bound 操作。

    MIXED = auto()
    # ↑ 混合模式 = Chunked Prefill。
    # 同一批次里既有 EXTEND（prefill 的一个 chunk）又有 DECODE 请求。
    # 提高 GPU 利用率的关键技术。

    IDLE = auto()
    # ↑ DP 场景：当前 worker 没有分到任何请求，空转一步（保持同步）。

    TARGET_VERIFY = auto()
    # ↑ 投机解码（Speculative Decoding）：
    # Target 大模型并行验证 draft 小模型预测的多个 token。

    DRAFT_EXTEND = auto()
    # ↑ 投机解码：Draft 小模型做 prefill/extend。

    PREBUILT = auto()
    # ↑ PD 解耦（Disaggregation）解码侧：
    # KV Cache 已经从 Prefill 实例传输过来，直接开始 decode，跳过 prefill。
```

---

### 3.3 RadixCache（radix_cache.py）：前缀 KV 共享

#### 先看数据结构

```
RadixCache 的树形结构示意：

                        root (TreeNode)
                       /              \
              [1,2,3,4]               [5,6,7]
           key=token_ids           key=token_ids
           value=KV indices        value=KV indices
             /         \
         [5,6]         [7,8,9]
        (req A)        (req B)

说明：
- 节点 key   = 这段 token ids（不是完整序列，是这段路径上新增的部分）
- 节点 value = 这段 token 对应的 KV Cache 物理索引（torch.Tensor, int64）
- 两个请求共享 [1,2,3,4] 的前 4 个 token，它们的 KV 只计算一次
- lock_ref   = 引用计数，>0 表示有请求正在使用，不能被淘汰
```

```python
class TreeNode:
    children = defaultdict(TreeNode)  # ← key: token_id（或 tuple(extra_key, token_id)）
    parent: TreeNode                  # ← 父节点（用于沿树向上遍历）
    key: RadixKey                     # ← 这段路径的 token ids
    value: Optional[torch.Tensor]     # ← 对应的 KV Cache 物理索引，None 表示已淘汰
    lock_ref: int                     # ← 引用计数，有请求使用时 > 0
    last_access_time: float           # ← LRU 淘汰用的时间戳
    host_value: Optional[torch.Tensor]  # ← HiCache：已 offload 到 CPU 内存的备份

class RadixKey:
    token_ids: List[int]              # ← token 序列（核心）
    extra_key: Optional[str]          # ← 命名空间隔离，如 lora_id 或 cache_salt
                                      #   相同 token_ids 但不同 extra_key → 不共享
    is_bigram: bool                   # ← EAGLE 投机解码用，两两 token 对作为 key
```

#### match_prefix：查命中前缀

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 调用时机：Scheduler 收到新请求，在 prefill 之前调用
# 目的：找出已经缓存的最长前缀，避免重复计算这部分的 KV
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
    key = params.key                      # ← 新请求的完整 token ids

    key = key.page_aligned(self.page_size)  # ← 对齐到 page_size（默认 1）

    value, last_node = self._match_prefix_helper(self.root_node, key)
    # ↑ 从根节点开始沿树往下走，尽可能匹配最长前缀
    # value：匹配到的所有节点的 KV indices 拼在一起
    # last_node：匹配结束的树节点（后续 insert 从这里接）

    # 注意：如果命中在一个节点的中间，会把该节点 split 成两个
    # 例：节点存了 [1,2,3,4,5]，但新请求只匹配到 [1,2,3]
    # 会把节点分裂为 [1,2,3] + [4,5]，精确对齐命中边界

    return MatchResult(
        device_indices=torch.cat(value),  # ← 命中的 KV Cache 物理索引（直接喂给 ModelRunner）
        last_device_node=last_node,       # ← 命中结束节点（之后 decode 完成时 insert 从这里开始）
    )

# 使用示例（scheduler.py 中）：
# cache_result = tree_cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids)))
# req.cache_protected_len = len(cache_result.device_indices)  # 命中了多少 token
# req.last_node = cache_result.last_device_node               # 记住终止节点
```

#### cache_finished_req：请求完成后写回缓存

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 调用时机：请求生成完毕（或被中断），把新计算的 KV 写入树
# 目的：下次有类似前缀的请求，可以直接命中
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cache_finished_req(self, req: Req):
    token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
    # ↑ 完整序列 = 输入 + 输出（只取已经写入 KV 的部分）

    kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, :len(token_ids)]
    # ↑ 从内存池中取出这些 token 对应的物理 KV 索引

    result = self.insert(InsertParams(key=RadixKey(token_ids), value=kv_indices))
    # ↑ 插入树中（内部会找公共前缀，只插入新增部分）

    # 关键：释放"重复"的 KV（已经在树中存在的部分，新分配的可以还回去）
    self.token_to_kv_pool_allocator.free(kv_indices[req.cache_protected_len:new_prefix_len])
    #                                                ↑ 之前就命中的        ↑ 刚刚 insert 进去的

    self.dec_lock_ref(req.last_node)
    # ↑ 解锁节点（引用计数 -1），允许被 LRU 淘汰
```

---

### 3.4 Scheduler 主循环（scheduler.py）

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 文件：python/sglang/srt/managers/scheduler.py
# Scheduler 是 SGLang 的大脑，在独立子进程中运行
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Scheduler(
    SchedulerOutputProcessorMixin,    # ← 处理模型输出（token ids → finish/continue）
    SchedulerUpdateWeightsMixin,      # ← 在线更新模型权重（RLHF 场景）
    SchedulerMetricsMixin,            # ← Prometheus 指标收集
    SchedulerDisaggregationDecodeMixin,  # ← PD 解耦解码侧逻辑
    SchedulerDisaggregationPrefillMixin, # ← PD 解耦预填充侧逻辑
    ...                               # ← 更多 Mixin，按需叠加功能
):
    def __init__(self, server_args, port_args, gpu_id, tp_rank, ...):
        self.init_model_config(server_args)     # ← 加载模型配置（层数、head数等）
        self.init_ipc_channels(port_args)       # ← 建立 ZMQ socket（收发消息）
        self.init_tokenizer(server_args)        # ← 初始化 tokenizer
        self.init_model_worker(...)             # ← 启动 TpModelWorker（加载模型权重）
        self.init_cache_with_memory_pool(...)   # ← 初始化 RadixCache + 物理内存池
        self.init_running_status(...)           # ← 初始化请求队列（waiting/running）
        self.init_chunked_prefill(...)          # ← Chunked Prefill 参数
        self.init_schedule_policy(...)          # ← 调度策略（LRU/cache-aware）
        self.init_grammar_backend(...)          # ← 约束解码（JSON Schema 等）

    def event_loop_normal(self):
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Scheduler 主循环（简化版）
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        while True:
            recv_requests()
            # ↑ 从 ZMQ socket 接收 TokenizerManager 发来的新请求
            # 解析成 Req 对象，放入 self.waiting_queue

            new_batch = self.get_next_batch_to_run()
            # ↑ 核心调度决策（详见 3.5）
            # 返回 ScheduleBatch 或 None（没有可调度的请求时）

            if new_batch is None:
                self.watchdog_last_forward_time = time.time()
                continue

            result = self.run_batch(new_batch)
            # ↑ 调用 tp_worker.forward_batch_generation()
            # tp_worker 再调用 model_runner.forward_batch_generation()
            # 最终在 GPU 上执行 attention + FFN + sample

            self.process_batch_result(new_batch, result)
            # ↑ 处理输出：
            #   - 每个请求追加 output_ids
            #   - 检查是否达到 stop condition（max_tokens / EOS / stop_str）
            #   - 已完成的请求：cache_finished_req() + 发送结果给 DetokenizerManager
            #   - 未完成的请求：留在 running_batch 继续下一轮 decode
```

---

### 3.5 调度决策（schedule_policy.py）

```
get_next_batch_to_run() 的决策树：

┌─────────────────────────────────────────────────────────┐
│  有 running_batch（正在 decode 的请求）？                  │
│                                                         │
│  YES ──► 先把 running_batch 做一步 DECODE                │
│          同时看能否再塞入一些 EXTEND 请求（Chunked Prefill）│
│                                                         │
│  NO ──► 从 waiting_queue 取请求做 EXTEND（prefill）       │
│         按"能复用更多 KV Cache"的顺序排序（cache-aware）   │
└─────────────────────────────────────────────────────────┘

关键约束（PrefillAdder 管理）：
1. 总 token 数不超过 max_prefill_tokens（避免 OOM）
2. 剩余 KV Cache 空间够用
3. running_batch 的 decode 不被饿死
```

---

### 3.6 内存管理两层模型（memory_pool.py）

```
物理内存布局：

  ReqToTokenPool（CPU 上的索引表）
  ┌────┬──────────────────────────────────────────────────┐
  │行号│  token 槽位序列                                   │
  ├────┼──────────────────────────────────────────────────┤
  │ 0  │ [42, 17, 8, 91, 3, ...]   ← req_0 的 KV 物理地址 │
  │ 1  │ [5, 22, 67, ...]          ← req_1 的 KV 物理地址  │
  │ 2  │ (空闲)                                            │
  └────┴──────────────────────────────────────────────────┘
         ↑ 每个数字是 KV Cache 物理内存页的编号

  TokenToKVPoolAllocator（管理上面数字的分配/释放）
  ┌─────────────────────────────────────────────────────────┐
  │ free_slots = [2, 3, 7, 9, ...]  ← 可用的 KV 物理页编号  │
  │ alloc(n)   → 取出 n 个编号                              │
  │ free(ids)  → 归还编号                                   │
  └─────────────────────────────────────────────────────────┘

  KVCache（GPU 上的实际 Tensor）
  ┌─────────────────────────────────────────────────────────┐
  │  shape: [num_pages, num_layers, 2, num_heads, head_dim] │
  │         ↑页数      ↑层数       ↑K和V  ↑头数   ↑每头维度 │
  │  通过物理页编号索引：kv_cache[page_id]                   │
  └─────────────────────────────────────────────────────────┘

三者关系：
  Req.req_pool_idx  ──► ReqToTokenPool[req_pool_idx]  → [page_id_0, page_id_1, ...]
                                                              ↓
                                                      KVCache[page_id_i]  → 实际 K/V tensor
```

---

## 四、关键流程的端到端追踪

### 4.1 一条新请求从入队到首个 token 输出

```
时序图（✓ = 发生，→ = 调用，数字 = 步骤顺序）

① HTTP POST /v1/chat/completions
  → tokenizer_manager.py: 分配 rid, tokenize
  → ZMQ PUSH: GenerateReqInput → Scheduler

② Scheduler.recv_requests()
  → 解析成 Req 对象
  → waiting_queue.append(req)

③ Scheduler.get_next_batch_to_run()
  → schedule_policy.PrefillAdder.add_one_req(req)
    → tree_cache.match_prefix(req 的 token_ids)
      → 找到命中前缀（例如命中了 200 个 token）
    → req.cache_protected_len = 200
    → 分配物理 KV 页：token_to_kv_pool_allocator.alloc(需要 extend 的 token 数)
    → req_to_token_pool.alloc_slot() → req.req_pool_idx = 0
  → 构造 ScheduleBatch(reqs=[req], forward_mode=EXTEND)

④ ScheduleBatch.to_model_worker_batch()
  → 提取 input_ids, seq_lens, out_cache_loc 等 → ModelWorkerBatch

⑤ ModelRunner.forward_batch_generation(model_worker_batch)
  → 构造 ForwardBatch（转成 GPU tensor）
  → attn_backend.init_forward_metadata(forward_batch)  # 初始化 attention 元数据
  → model.forward(input_ids, positions, forward_batch)  # 真正跑 transformer
    → 每一层的 RadixAttention.forward()
       → attn_backend.forward_extend(q, k, v, ...)      # FlashInfer CUDA kernel
       → 写入 KV Cache 到 out_cache_loc 指定的位置
  → logits_processor + sampler → 采样出 next_token_id

⑥ Scheduler.process_batch_result()
  → req.output_ids.append(next_token_id)
  → 检查是否结束（未结束）
  → ZMQ PUSH: output token id → DetokenizerManager

⑦ DetokenizerManager
  → tokenizer.decode([next_token_id]) → 文字（增量）
  → ZMQ PUSH → TokenizerManager → SSE chunk → HTTP client
```

---

### 4.2 RadixCache 命中前缀时节省了什么

```
假设：系统 prompt = "你是一个助手，请认真回答用户的问题。"（100 个 token）

请求 A（第一个来的）：
  - match_prefix → 命中 0 个 token（树是空的）
  - extend 100 个 token → 计算 100 个 token 的 KV
  - 完成后 cache_finished_req() → 插入树，存了 100 个 token 的 KV 索引

请求 B（后来的，同样的系统 prompt）：
  - match_prefix → 命中 100 个 token！
  - out_cache_loc 填入树中已有的 KV 索引
  - extend 只需计算 0 个 token（全部命中）
  - GPU 计算量节省 100 个 token 的 Attention + FFN！

                   B 节省的工作
                   ←── 100 tokens ──→
  ┌────────────────┬──────────────────────────────────┐
  │ 命中（从缓存读）│   需要重新计算（B 独有的用户输入）  │
  └────────────────┴──────────────────────────────────┘
```

---

### 4.3 Chunked Prefill（ForwardMode.MIXED）

```
场景：有 3 个请求在 decode，同时来了 1 个需要 prefill 的新请求（prompt 很长）

普通做法：
  Round 1: DECODE [req1, req2, req3]              → GPU 利用率低（只有 3 个 token）
  Round 2: EXTEND [new_req（全部 2000 token）]     → GPU 显存可能不够，延迟 decode

Chunked Prefill 做法（MIXED 模式）：
  Round 1: MIXED  [req1 decode, req2 decode, req3 decode, new_req 前 500 token]
  Round 2: MIXED  [req1 decode, req2 decode, req3 decode, new_req 后 500 token]
  Round 3: MIXED  [req1 decode, req2 decode, req3 decode, new_req 再 500 token]
  Round 4: MIXED  [req1 decode, req2 decode, req3 decode, new_req 最后 500 token]
  Round 5: DECODE [req1, req2, req3, new_req]  ← 全部进入 decode

优势：
  - decode 请求的延迟不增加（每轮都在推进）
  - 新请求的 prefill 被分散，不会 OOM
  - GPU 每轮有更多 token，利用率提升
```

---

## 五、读源码的推荐顺序

```
第 1 步（15 分钟）：
  forward_batch_info.py:80        ForwardMode 枚举（80行，全读）

第 2 步（20 分钟）：
  schedule_batch.py:574           Req.__init__（只看 __init__ 前 60 行）
  schedule_batch.py:1350          ScheduleBatch（只看字段定义，约 60 行）
  schedule_batch.py:2689          ModelWorkerBatch（全读，约 60 行）

第 3 步（30 分钟）：
  radix_cache.py:211              TreeNode（约 60 行）
  radix_cache.py:398              match_prefix（约 70 行）
  radix_cache.py:468              insert（约 20 行）
  radix_cache.py:488              cache_finished_req（约 40 行）← 最有价值

第 4 步（30 分钟）：
  model_runner.py                 全文搜索 "def forward_batch_generation"
                                  读这个函数（约 80 行）

第 5 步（30 分钟）：
  scheduler.py                    全文搜索 "def event_loop_normal"
                                  读这个函数（约 50 行）

以上 5 步完成后，你已经覆盖了 SGLang 的核心主干。
```

---

## 六、容易混淆的概念对照表

| 容易混淆的点 | 澄清 |
|---|---|
| `seq_lens` vs `extend_seq_lens` | `seq_lens`：包含已缓存前缀的完整长度；`extend_seq_lens`：只需要计算的新增部分长度 |
| `kv_committed_len` vs `kv_allocated_len` | `allocated`：物理页已分配但 KV 还没计算；`committed`：KV 已实际写入，可以插入树 |
| `match_prefix` 返回的 `device_indices` | 是物理 KV 页编号，不是 token ids！直接是内存地址索引 |
| `req_pool_idx` vs `rid` | `rid`：HTTP 请求的唯一 ID（字符串）；`req_pool_idx`：在内存池中的行号（整数） |
| `EXTEND` vs `DECODE` | `EXTEND` 包含 prefill 中有缓存命中的场景，不完全等于"从头计算" |
| `ForwardBatch` vs `ModelWorkerBatch` | `ModelWorkerBatch` 在 CPU 上，`ForwardBatch` 在 GPU 上（Tensor） |
| `tree_cache` vs `token_to_kv_pool_allocator` | `tree_cache` 管逻辑索引（哪些前缀被缓存）；`allocator` 管物理内存页 |

---

*搭配 [SGLANG_SOURCE_GUIDE.md](SGLANG_SOURCE_GUIDE.md) 一起食用效果更佳。*
*有疑问直接提问，可以针对任何一个函数或数据结构展开。*
