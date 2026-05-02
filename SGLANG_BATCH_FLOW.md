# SGLang 批量请求全链路：从 HTTP 到 Token 返回

> **阅读方式**：先看本页的总数据流图，然后按编号顺序点击跳转链接，对着注释读源码。
> 每一节都对应一个真实函数，注释直接解释"这段代码在做什么、为什么这样做"。

---

## 总蓝图：批量请求的完整数据流

```
多个 HTTP 请求同时进来
       │
       ▼
① [主进程] TokenizerManager.generate_request()         tokenizer_manager.py:511
       │  text → token_ids，构造 Req，通过 ZMQ 发给 Scheduler
       │
       ▼  ZMQ PUSH（每条请求独立发送，非批次）
       │
       ▼
② [Scheduler 子进程] recv_requests()                    scheduler.py:1590
       │  非阻塞轮询 ZMQ，一次性取出所有等待中的请求
       │  → 转成 Req 对象，放入 waiting_queue
       │
       ▼
③ [Scheduler] get_next_batch_to_run()                   scheduler.py:2388
       │  核心调度决策：从 waiting_queue + running_batch 凑一批
       │  └── _get_new_batch_prefill_raw()               scheduler.py:2527
       │        └── PrefillAdder：逐条拣选请求，查 RadixCache 命中
       │  返回 ScheduleBatch（可能包含 EXTEND + DECODE 混合）
       │
       ▼
④ [Scheduler] run_batch()                               scheduler.py:2864
       │  → batch.get_model_worker_batch()               ScheduleBatch 转 ModelWorkerBatch
       │  → model_worker.forward_batch_generation()      发给 GPU
       │    └── ModelRunner：ForwardBatch + AttentionBackend + Sampler
       │  返回 GenerationBatchResult（含 next_token_ids tensor）
       │
       ▼
⑤ [Scheduler] process_batch_result()                    scheduler.py:3046
       │  EXTEND 路径 → process_batch_result_prefill()   scheduler_output_processor_mixin.py:128
       │  DECODE 路径 → process_batch_result_decode()    scheduler_output_processor_mixin.py:392
       │    ├── req.output_ids.append(next_token_id)     每条请求追加一个 token
       │    ├── req.check_finished()                     检查停止条件
       │    ├── release_kv_cache() / cache_unfinished_req() 更新 RadixCache
       │    └── stream_output()                          打包发给 Detokenizer
       │
       ▼  ZMQ PUSH（BatchTokenIDOutput，整批打包）
       │
       ▼
⑥ [DetokenizerManager 子进程] event_loop()              detokenizer_manager.py:138
       │  → handle_batch_token_id_out()                  增量 detokenize
       │  → send_to_tokenizer（ZMQ PUSH，BatchStrOutput）
       │
       ▼
⑦ [主进程] TokenizerManager.handle_loop()               tokenizer_manager.py:1627
       │  → _handle_batch_output()                       找到对应请求的 asyncio Future
       │  → state.event.set()                            唤醒等待的协程
       │
       ▼
⑧ [主进程] HTTP SSE 流式返回给各个客户端
```

---

## 逐步注释源码

---

### ① TokenizerManager.generate_request — 入口

> **跳转**：[tokenizer_manager.py 第 511 行](python/sglang/srt/managers/tokenizer_manager.py#L511)

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# tokenizer_manager.py:511
# HTTP handler（FastAPI）调用这个 async 生成器来处理单个请求
# 它是 async generator，yield 出每个增量 token 的 HTTP chunk
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def generate_request(self, obj: GenerateReqInput, request):

    obj.normalize_batch_and_arguments()   # ← 统一格式，处理 batch 请求展开
    self._init_req_state(obj, request)    # ← 为这个请求创建 ReqState（含 asyncio.Event）
                                          #   rid_to_state[rid] = ReqState(obj, event, ...)

    # 单条请求路径（最常见）
    tokenized_obj = await self._tokenize_one_request(obj)
    # ↑ 调用 HuggingFace tokenizer 把 text → token_ids
    # 如果是多模态输入，还会处理图像特征

    self._send_one_request(tokenized_obj)
    # ↑ 通过 ZMQ PUSH socket 发给 Scheduler
    # self.send_to_scheduler.send_pyobj(tokenized_obj)

    async for response in self._wait_one_response(obj, request):
        yield response
    # ↑ 挂起等待，直到 handle_loop() 收到对应 rid 的结果
    # 每收到一个增量 token 就 yield 一次（SSE 流式）
```

**关键数据结构**：
- 每个请求进来后立刻在 `rid_to_state` 字典里登记一个 `ReqState`
- `ReqState` 里有一个 `asyncio.Event`，用于唤醒等待协程
- ZMQ 是**非阻塞发送**，发完立刻返回，不等 Scheduler 处理

---

### ② Scheduler.recv_requests — 非阻塞收包

> **跳转**：[scheduler.py 第 1590 行](python/sglang/srt/managers/scheduler.py#L1590)

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# scheduler.py:1590
# 每轮调度循环开始时调用，非阻塞地取出所有待处理请求
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def recv_requests(self):
    recv_reqs = []
    while True:
        try:
            recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
            # ↑ zmq.NOBLOCK = 非阻塞！没有消息立刻抛 ZMQError
            # 有消息就取出来（反序列化的 TokenizedGenerateReqInput 对象）
        except zmq.ZMQError:
            break  # ← 没有更多消息了，退出循环
        recv_reqs.append(recv_req)

    # 如果是 TP 多卡场景，tp_rank=0 收到后广播给其他 rank
    # （每个 TP rank 都是独立进程，需要保持批次同步）

    return recv_reqs
    # ↑ 返回本轮收到的所有请求（可能是 0 个、1 个或多个）
```

**批量的形成**：
- 多个 HTTP 请求几乎同时到达，都被 ZMQ 排在队列里
- `recv_requests()` 一次性把队列全部清空
- 这就是"批量"的来源：**并发请求被同一轮调度循环捞起来**

---

### ③ Scheduler.event_loop_normal — 主循环

> **跳转**：[scheduler.py 第 1469 行](python/sglang/srt/managers/scheduler.py#L1469)

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# scheduler.py:1469
# Scheduler 的无限循环，每次迭代处理一个批次
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def event_loop_normal(self):
    while True:
        # Step 1：收新请求放进 waiting_queue
        recv_reqs = self.recv_requests()           # ← 取出本轮所有新请求（可能0个）
        self.process_input_requests(recv_reqs)     # ← 解析成 Req，放进 waiting_queue

        # Step 2：决定这轮跑哪些请求
        batch = self.get_next_batch_to_run()       # ← 核心调度（见 ④）
        self.cur_batch = batch

        # Step 3：执行批次
        if batch:
            result = self.run_batch(batch)         # ← GPU 前向（见 ⑤）
            self.process_batch_result(batch, result)  # ← 处理输出（见 ⑥）
        else:
            self.on_idle()                         # ← 无任务时做 GC、指标上报等

        self.last_batch = batch
        # ↑ 保存本轮批次，下轮 get_next_batch_to_run 会用到
        # （判断上一轮是 prefill 还是 decode，影响合并策略）
```

**两种循环**：
- `event_loop_normal`：本轮 GPU 跑完才处理结果，再开始下一轮
- `event_loop_overlap`（[scheduler.py:1497](python/sglang/srt/managers/scheduler.py#L1497)）：上一轮 GPU 还在跑，CPU 已经在准备下一轮批次（overlap 优化）

---

### ④ Scheduler._get_new_batch_prefill_raw — 核心调度决策

> **跳转**：[scheduler.py 第 2527 行](python/sglang/srt/managers/scheduler.py#L2527)

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# scheduler.py:2527
# 从 waiting_queue 选出哪些请求做 prefill（EXTEND）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _get_new_batch_prefill_raw(self, ...):

    # 快速返回条件
    if self.running_batch.batch_is_full or len(self.waiting_queue) == 0:
        return None   # ← 已经满了，或没有等待的请求

    # 计算优先级（cache-aware 排序）
    self.policy.calc_priority(self.waiting_queue, self.running_batch)
    # ↑ 核心思想：优先调度能复用更多 RadixCache 的请求
    # 比如两个请求都有相同的 system prompt，优先调度它们让 KV 共享最大化

    # 构建 PrefillAdder，它负责"一条一条往批次里塞请求"
    adder = PrefillAdder(
        tree_cache=self.tree_cache,
        token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
        running_batch=self.running_batch,
        max_prefill_tokens=self.max_prefill_tokens,   # ← 每批 prefill token 数上限
        chunked_prefill_size=chunked_prefill_size,    # ← chunked prefill 的 chunk 大小
        ...
    )

    for req in self.waiting_queue:
        # 尝试把每个等待中的请求加入本批次
        result = adder.add_one_req(req)
        # add_one_req 内部会：
        #   1. tree_cache.match_prefix(req.token_ids)  → 查命中前缀
        #   2. 估算需要多少 KV Cache 空间
        #   3. 如果空间足够 → alloc KV → 加入批次
        #   4. 如果空间不够 → 标记 batch_is_full，停止添加
        if result == AddReqResult.NO_TOKEN:
            break   # ← token 预算用完，不能再加

    new_batch = adder.finalize()
    # ↑ 把选中的请求组成 ScheduleBatch，forward_mode = EXTEND

    # 如果同时有 decode 请求（is_mixed_chunk），合并成 MIXED 模式
    if self.is_mixed_chunk and not self.running_batch.is_empty():
        new_batch.merge_batch(self.running_batch)
        new_batch.forward_mode = ForwardMode.MIXED

    return new_batch
```

**get_next_batch_to_run 的完整逻辑**（[scheduler.py:2388](python/sglang/srt/managers/scheduler.py#L2388)）：

```
get_next_batch_to_run()
   │
   ├── 如果上一轮是 EXTEND（prefill）：
   │     把完成 prefill 的请求合并到 running_batch（进入 decode 阶段）
   │
   ├── get_new_batch_prefill()
   │     └── _get_new_batch_prefill_raw()  → 从 waiting_queue 凑 prefill 批次
   │
   └── 最终返回：
         有 prefill 候选 → 返回 prefill batch（EXTEND 或 MIXED）
         没有 prefill    → 返回 running_batch（纯 DECODE）
         两者都空         → 返回 None（idle）
```

---

### ⑤ Scheduler.run_batch — 发给 GPU 执行

> **跳转**：[scheduler.py 第 2864 行](python/sglang/srt/managers/scheduler.py#L2864)

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# scheduler.py:2864
# 把 ScheduleBatch 转换并送给 GPU 执行，拿回结果
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_batch(self, batch: ScheduleBatch):

    # 将 CPU 调度数据结构转换为 TP Worker 需要的格式
    worker_batch = batch.get_model_worker_batch()
    # ↑ ScheduleBatch → ModelWorkerBatch
    # 这一步把 List[Req] 里的数据提取出来，打包成 tensor
    # 比如：[req.fill_ids for req in reqs] → input_ids tensor
    #       [req.kv_committed_len for req in reqs] → extend_prefix_lens

    # 调用 TP Worker（在同一进程内，不通过 ZMQ）
    batch_result = self.model_worker.forward_batch_generation(worker_batch)
    # ↑ TpModelWorker.forward_batch_generation()
    #   └── ModelRunner.forward_batch_generation()
    #         ├── 构造 ForwardBatch（转成 GPU tensor）
    #         ├── attention_backend.init_forward_metadata(forward_batch)
    #         ├── model.forward(input_ids, positions, forward_batch)
    #         │     每一层：RadixAttention.forward() → attn_backend.forward_extend/decode()
    #         └── logits_processor + sampler → next_token_ids（GPU tensor）

    return batch_result
    # GenerationBatchResult 含：
    #   next_token_ids: torch.Tensor  shape [batch_size]，还在 GPU 上
    #   logits_output: LogitsProcessorOutput（logprobs 等附加信息）
    #   copy_done: cuda Event（异步 D2H copy 的完成信号）
```

**batch_size 是多少**：
- `batch.reqs` 里有多少个请求，`next_token_ids` 就有多少个元素
- EXTEND 模式：`input_ids` shape = `[所有请求的 extend token 总数]`（不是 batch_size！）
- DECODE 模式：`input_ids` shape = `[batch_size]`（每个请求 1 个 token）
- MIXED 模式：两者拼在一起

---

### ⑥-A process_batch_result_prefill — EXTEND 结果处理

> **跳转**：[scheduler_output_processor_mixin.py 第 128 行](python/sglang/srt/managers/scheduler_output_processor_mixin.py#L128)

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# scheduler_output_processor_mixin.py:128
# prefill（EXTEND）批次的结果处理
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def process_batch_result_prefill(self, batch, result):

    next_token_ids = result.next_token_ids.tolist()
    # ↑ GPU tensor → Python list，D2H copy（触发同步）

    for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
        if req.is_chunked > 0:
            continue  # ← chunked prefill 未完成，不生成 token，跳过

        # 生成第一个 token（prefill 结束，拿到第一个输出 token）
        req.output_ids.append(next_token_id)          # ← 追加到请求的输出列表

        req.check_finished()                           # ← 检查停止条件
        # check_finished 会检查：
        #   1. output_ids 是否包含 EOS token
        #   2. 是否达到 max_new_tokens
        #   3. 是否匹配 stop_str / stop_token_ids

        if req.finished():
            release_kv_cache(req, self.tree_cache)
            # ↑ 请求已完成，解锁 RadixCache 节点（引用计数 -1）
            # 这些 KV 页可以被 LRU 淘汰了
        else:
            self.tree_cache.cache_unfinished_req(req)
            # ↑ 请求未完成（还需要继续 decode），把已计算的 KV 插入树
            # 下一次调度这个请求时，match_prefix 会命中

    # 打包所有请求的输出，发给 DetokenizerManager
    self.stream_output(batch.reqs, batch.return_logprob)
    # ↑ 见 ⑦
```

---

### ⑥-B process_batch_result_decode — DECODE 结果处理

> **跳转**：[scheduler_output_processor_mixin.py 第 392 行](python/sglang/srt/managers/scheduler_output_processor_mixin.py#L392)

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# scheduler_output_processor_mixin.py:392
# decode 批次的结果处理（每轮生成 1 个 token/请求）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def process_batch_result_decode(self, batch, result):

    next_token_ids = result.next_token_ids.tolist()  # ← GPU → CPU

    self.token_to_kv_pool_allocator.free_group_begin()
    # ↑ 开启批量释放模式（把释放操作积累起来，最后一次性执行，更高效）

    for i, req in enumerate(batch.reqs):
        next_token_id = next_token_ids[i]

        req.output_ids.append(next_token_id)           # ← 追加这轮生成的 token

        req.check_finished()                           # ← 检查停止条件

        self._handle_finished_req(req, i, logits_output)
        # ↑ 如果 req.finished()：
        #     → release_kv_cache(req, self.tree_cache)  归还 KV Cache 物理页
        #       tree_cache.cache_finished_req(req)      插入 RadixTree 供复用

        if req.grammar is not None:
            req.grammar.accept_token(next_token_id)
            # ↑ 约束解码：通知 grammar backend 接受这个 token
            # 下一步采样时，grammar 会限制 logits，只允许合法的 token

    self.stream_output(batch.reqs, batch.return_logprob)
    # ↑ 打包发给 DetokenizerManager（见 ⑦）

    self.token_to_kv_pool_allocator.free_group_end()
    # ↑ 真正执行积累的 KV 页释放
```

---

### ⑦ stream_output_generation — 打包发送给 Detokenizer

> **跳转**：[scheduler_output_processor_mixin.py 第 944 行](python/sglang/srt/managers/scheduler_output_processor_mixin.py#L944)

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# scheduler_output_processor_mixin.py:944
# 把本轮所有请求的输出打包成一个 BatchTokenIDOutput，
# 通过 ZMQ 发给 DetokenizerManager
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def stream_output_generation(self, reqs, return_logprob, ...):

    rids = []          # ← 每个请求的 rid
    output_ids = []    # ← 每个请求的完整 output_ids（增量检测在 detokenizer 侧）
    finished_reasons = []
    # ... 其他字段

    for req in reqs:
        if req.finished() or req.is_streaming:
            # 只把"有新内容"的请求发出去（已完成 或 流式的）
            rids.append(req.rid)
            output_ids.append(req.output_ids)
            finished_reasons.append(req.finished_reason)
            # ...

    if rids:
        self.send_to_detokenizer.send_pyobj(
            BatchTokenIDOutput(
                rids=rids,
                output_ids=output_ids,           # ← token id 列表，不是文字
                finished_reasons=finished_reasons,
                # ...
            )
        )
        # ↑ 一次 ZMQ 发送，携带本轮所有请求的增量输出
        # 注意：即使是非流式请求，每一步也会发（detokenizer 侧缓存，最后合并）
```

---

### ⑧ DetokenizerManager.event_loop + handle_batch_token_id_out

> **跳转**：[detokenizer_manager.py 第 138 行](python/sglang/srt/managers/detokenizer_manager.py#L138)

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# detokenizer_manager.py:138
# 独立子进程，专职做 token_ids → 文字 的转换
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def event_loop(self):
    while True:
        recv_obj = self.recv_from_scheduler.recv_pyobj()
        # ↑ 阻塞等待来自 Scheduler 的消息

        output = self._request_dispatcher(recv_obj)
        # ↑ 根据类型分发：
        #   BatchTokenIDOutput → handle_batch_token_id_out()  生成文字
        #   BatchEmbeddingOutput → handle_batch_embedding_out()  直接透传

        if output is not None:
            self.send_to_tokenizer.send_pyobj(output)
            # ↑ 把 BatchStrOutput（含文字）发回给 TokenizerManager

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# detokenizer_manager.py（handle_batch_token_id_out 核心逻辑简化）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def handle_batch_token_id_out(self, recv_obj: BatchTokenIDOutput):
    # 增量 detokenize：每个请求维护一个 decode_status
    for rid, output_ids, finished_reason in zip(...):
        state = self.decode_status.get(rid)
        if state is None:
            # 新请求，初始化状态
            state = DecodeStatus(...)
            self.decode_status[rid] = state

        # 只 decode 新增的部分（state.read_offset 记录上次到哪里了）
        new_ids = output_ids[state.read_offset:]
        new_text = self.tokenizer.decode(new_ids, ...)
        # ↑ 增量解码！不是每次从头 decode 全部 output_ids
        # 这样对于长生成，detokenizer 不会越来越慢

        state.read_offset += len(new_ids)
        state.decoded_text += new_text

    # 打包成 BatchStrOutput 发回 TokenizerManager
    return BatchStrOutput(rids=..., decoded_texts=..., ...)
```

---

### ⑨ TokenizerManager.handle_loop — 收结果，唤醒 HTTP 协程

> **跳转**：[tokenizer_manager.py 第 1627 行](python/sglang/srt/managers/tokenizer_manager.py#L1627)

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# tokenizer_manager.py:1627
# 主进程的 asyncio 事件循环中运行的协程
# 不断从 ZMQ 接收 DetokenizerManager 发来的结果
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def handle_loop(self):
    while True:
        recv_obj = await self.recv_from_detokenizer.recv_pyobj()
        # ↑ 异步等待，不阻塞其他协程

        if isinstance(recv_obj, (BatchStrOutput, BatchTokenIDOutput)):
            await self._handle_batch_output(recv_obj)

async def _handle_batch_output(self, recv_obj):
    for i, rid in enumerate(recv_obj.rids):
        state = self.rid_to_state.get(rid)  # ← 找到对应请求的等待状态
        if state is None:
            continue  # ← 客户端已断开连接

        # 构建响应对象（包含本次增量文字）
        out = {
            "text": recv_obj.decoded_texts[i],
            "finish_reason": recv_obj.finished_reasons[i],
            "meta_info": {...}
        }

        state.out_list.append(out)       # ← 把结果放入这个请求的输出队列
        state.event.set()                # ← 唤醒 generate_request 里的等待协程！

# 被唤醒后，generate_request 里的 _wait_one_response 取出 out，yield 给 FastAPI
# FastAPI 把它写入 SSE 流，发送给 HTTP 客户端
```

---

## 批量处理的关键理解

### 批次是怎么形成的

```
时间轴：

t=0ms   req_A 到达，被 ZMQ 排队
t=1ms   req_B 到达，被 ZMQ 排队
t=2ms   req_C 到达，被 ZMQ 排队
t=5ms   Scheduler 跑完上一个批次，进入下一轮循环
        recv_requests() 一次性取出 [A, B, C]
        → 三条请求组成一个批次
```

### token 如何逐步返回给各个客户端

```
Round 1（EXTEND）：A、B、C 各做 prefill → 各生成 token_A1、token_B1、token_C1
        stream_output → DetokenizerManager → TokenizerManager
        → state_A.event.set()，state_B.event.set()，state_C.event.set()
        → 三个 HTTP 连接各自收到第 1 个 token

Round 2（DECODE）：A、B、C 同时 decode → 各生成 token_A2、token_B2、token_C2
        → 三个 HTTP 连接各自收到第 2 个 token

Round N：某个请求 check_finished() 返回 True（如 B 生成了 EOS）
        → B 从 running_batch 中移除
        → B 的 HTTP 连接收到 finish_reason，关闭 SSE 流
```

### 批次大小随时间变化

```
running_batch 的 batch_size 在动态变化：

  [A, B, C] → B 完成 → [A, C] → D 加入（prefill）→ [A, C, D] → ...

Scheduler 每轮都会：
  1. 把完成 prefill 的新请求合并进来（batch 变大）
  2. 把已完成 decode 的请求移除（batch 变小）
  3. 尝试从 waiting_queue 补充新请求
```

---

## 快速跳转索引

| 步骤 | 函数 | 文件 | 行号 |
|---|---|---|---|
| HTTP 接收 & tokenize | `generate_request` | [tokenizer_manager.py](python/sglang/srt/managers/tokenizer_manager.py#L511) | 511 |
| 主循环 | `event_loop_normal` | [scheduler.py](python/sglang/srt/managers/scheduler.py#L1469) | 1469 |
| 收新请求 | `recv_requests` | [scheduler.py](python/sglang/srt/managers/scheduler.py#L1590) | 1590 |
| 调度决策 | `get_next_batch_to_run` | [scheduler.py](python/sglang/srt/managers/scheduler.py#L2388) | 2388 |
| 选 prefill 请求 | `_get_new_batch_prefill_raw` | [scheduler.py](python/sglang/srt/managers/scheduler.py#L2527) | 2527 |
| 发给 GPU | `run_batch` | [scheduler.py](python/sglang/srt/managers/scheduler.py#L2864) | 2864 |
| 分发处理结果 | `process_batch_result` | [scheduler.py](python/sglang/srt/managers/scheduler.py#L3046) | 3046 |
| prefill 结果处理 | `process_batch_result_prefill` | [scheduler_output_processor_mixin.py](python/sglang/srt/managers/scheduler_output_processor_mixin.py#L128) | 128 |
| decode 结果处理 | `process_batch_result_decode` | [scheduler_output_processor_mixin.py](python/sglang/srt/managers/scheduler_output_processor_mixin.py#L392) | 392 |
| 请求完成处理 | `_handle_finished_req` | [scheduler_output_processor_mixin.py](python/sglang/srt/managers/scheduler_output_processor_mixin.py#L555) | 555 |
| 发往 Detokenizer | `stream_output_generation` | [scheduler_output_processor_mixin.py](python/sglang/srt/managers/scheduler_output_processor_mixin.py#L944) | 944 |
| Detokenizer 主循环 | `DetokenizerManager.event_loop` | [detokenizer_manager.py](python/sglang/srt/managers/detokenizer_manager.py#L138) | 138 |
| 唤醒 HTTP 协程 | `TokenizerManager.handle_loop` | [tokenizer_manager.py](python/sglang/srt/managers/tokenizer_manager.py#L1627) | 1627 |
