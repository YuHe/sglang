# SGLang 请求完整数据流：极端细化版

> **目标**：不逐行读源码，1 小时内掌握 SGLang 从请求入 → Token 返回的所有核心机制、特殊情况、加速优化。
>
> **阅读建议**：先整体看一遍蓝图，再跟着每个阶段深入，遇到不理解的概念跳到对应的"深入讲解"小节。

---

## 总体蓝图：5 进程 × 8 阶段

```
HTTP 请求
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│            主进程（FastAPI + asyncio）                   │
│  ┌──────────────────────────────────────────────────┐   │
│  │          TokenizerManager                        │   │
│  │  ① HTTP 接入 → 创建 ReqState                    │   │
│  │  ② tokenize(text → token_ids)                  │   │
│  │  ③ ZMQ PUSH → Scheduler                        │   │
│  │  ⑨ ZMQ PULL ← Detokenizer → 唤醒 HTTP 协程    │   │
│  └──────────────────────────────────────────────────┘   │
└────────────┬──────────────────────────┲━━━━━━━━━━━━━━━━━┘
             │ ZMQ PUSH                 ┃ ZMQ PUSH
             ▼                         ┃
┌──────────────────────────────────┐   ┃
│         Scheduler 子进程          │   ┃
│  ④ recv_requests (非阻塞)        │   ┃
│  ⑤ get_next_batch_to_run        │   ┃
│     └─ RadixCache prefix match  │   ┃
│     └─ PrefillAdder 选请求      │   ┃
│  ⑥ run_batch                    │   ┃
│     └─ ScheduleBatch →          │   ┃
│        ModelWorkerBatch →        │   ┃
│        TpModelWorker.forward()  │   ┃
│  ⑦ process_batch_result         │   ┃
│     └─ 检查 finish              │   ┃
│     └─ 更新 RadixCache          │   ┃
│     └─ ZMQ PUSH → Detokenizer  │   ┃
└──────────────────────────────────┘   ┃
                                       ┃ ZMQ PUSH
                                       ▼
                      ┌─────────────────────────────┐
                      │   DetokenizerManager 子进程  │
                      │  ⑧ token_ids → 文字（增量）  │
                      │  ZMQ PUSH → TokenizerManager│
                      └─────────────────────────────┘

GPU 侧（Scheduler 进程内）：
  TpModelWorker（主 TP rank）
    → ModelRunner.forward_batch_generation()
      → ForwardBatch（纯 GPU tensor）
      → attention_backend（FlashInfer/FlashAttention）
      → model.forward()
      → logits_processor + sampler
      → next_token_ids（GPU tensor D→H 异步拷贝）
```

---

## 阶段 ①：HTTP 接入 → 创建 ReqState

**文件**：`tokenizer_manager.py:511` `generate_request()`

### 正常路径
```
POST /v1/chat/completions
    │
    ▼ FastAPI route handler
    │  obj = GenerateReqInput(text, sampling_params, rid=uuid)
    ▼
generate_request(obj, fastapi_request)   ← async generator
    │
    ├─ normalize_batch_and_arguments()
    │    单条请求：is_single=True
    │    批量请求（List[str]）：展开成多条独立请求
    │
    ├─ _init_req_state(obj, request)
    │    rid_to_state[rid] = ReqState(
    │        event=asyncio.Event(),   ← 跨协程通知机制
    │        out_list=[],             ← 输出缓冲
    │        obj=obj
    │    )
    │
    ├─ is_pause_cond.wait_for(not is_pause)  ← 等待在线权重更新完成
    │
    └─ model_update_lock.reader_lock     ← 并发读锁（允许多请求同时持有）
```

### ReqState：HTTP 协程与调度循环的桥梁
```
HTTP 协程 (generate_request)         调度循环 (handle_loop)
         │                                    │
         │  rid_to_state[rid].event           │
         ├──────────── asyncio.Event ─────────┤
         │                                    │
await event.wait()              event.set() ──┘
         │                  （收到 Detokenizer 消息后）
         ▼
取 out_list 里的增量 token，yield 给 FastAPI
```

### 特殊情况
| 情况 | 处理 |
|---|---|
| `dp_size > 1` 且指定 `routed_dp_rank` | 验证合法性；路由到对应 DP 分组的 Scheduler |
| `tokenizer_worker_num > 1` | 附加 HTTP worker 信息，支持多 tokenizer 并行 |
| `language_only` 模式 | EPD 解耦：只做 encode，请求走特殊 disagg 路径 |
| 客户端立刻断连 | `create_abort_task()` 创建 2s 后的 abort 异步任务 |

---

## 阶段 ②：Tokenize（文本 → token_ids）

**文件**：`tokenizer_manager.py` `_tokenize_one_request()`

### 纯文本请求
```
input_text
    │
    ▼ apply_chat_template()（如果是 chat format）
    │
    ▼ HuggingFace tokenizer.encode()
    │
    ▼ TokenizedGenerateReqInput(
          rid=rid,
          origin_input_text=text,
          origin_input_ids=token_ids,        ← List[int]
          origin_input_ids_unpadded=token_ids,← 用于增量 detokenize 的偏移基准
          sampling_params=...,
          ...
      )
```

### 多模态请求（含图像）
```
input_text + image_data
    │
    ▼ mm_processor.process_mm_data_async()  ← 在线程池中异步执行
    │   ① 调用视觉处理器（ViT processor）：image → pixel_values tensor
    │   ② 每张图片计算 SHA-256 hash → 生成唯一 pad_value
    │       pad_value = MM_PAD_SHIFT_VALUE + (hash % 2^30)
    │       作用：不同图片 pad_value 不同 → RadixCache 不会混淆 KV
    │   ③ 把 <image> 占位符替换为 N 个 pad_value token（N=图像 patch 数）
    │
    ▼ origin_input_ids 含 pad_value 整数
      multimodal_inputs.mm_items = [
          MultimodalDataItem(
              modality="image",
              hash=sha256,
              pad_value=唯一整数,
              feature=pixel_values,   ← GPU tensor，在 forward 时才用
          )
      ]
```

**关键设计**：RadixCache 的 key 就是 `token_ids`（含 pad_value），唯一 pad_value 确保不同图片不会错误共享 KV Cache。

---

## 阶段 ③：ZMQ 发送到 Scheduler

**文件**：`tokenizer_manager.py` `_send_one_request()`

```python
self.send_to_scheduler.send_pyobj(tokenized_obj)
# send_pyobj = pickle 序列化 + zmq PUSH（非阻塞）
# 如果 dp_size > 1：根据 routed_dp_rank 选对应的 socket
```

**LoRA 处理**：`_validate_and_resolve_lora(obj)` 在发送前检查 `lora_id` 是否已加载，否则触发动态加载并等待。

---

## 阶段 ④：Scheduler 接收请求

**文件**：`scheduler.py:1590` `recv_requests()`

### 非阻塞拉取所有排队请求
```python
while True:
    recv_req = recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
    # 有消息取消息，无消息立刻 ZMQError → break
    recv_reqs.append(recv_req)
    if recv_limit_reached(len(recv_reqs)):
        break  # 单轮收包上限，防止饥饿
```

**关键**：这一步是"批量"自然形成的机制——并发的 HTTP 请求都堆在同一个 ZMQ 队列里，Scheduler 一次性清空，本轮就拿到了多条并发请求。

### process_input_requests：请求分类处理
```
recv_reqs 里的对象类型：
├─ TokenizedGenerateReqInput  → _add_request_to_queue(Req)
│    Req 对象创建，填充所有字段，推入 waiting_queue
├─ AbortReq                   → abort_request(rid)
├─ UpdateWeightReqInput       → update_weights(...)
├─ FlushCacheReq              → flush_cache()
└─ ProfileReq                 → start/stop profiler
```

**Req 对象关键字段**（完整见附录）：
```
rid                  请求唯一 ID
origin_input_ids     原始 token_ids（含图像 pad_value）
output_ids           已生成 token（空列表，逐步追加）
fill_ids             = origin_input_ids + output_ids（动态更新）
kv_committed_len     已插入 RadixCache 的 token 数
kv_allocated_len     已分配物理 KV 页的 token 数
prefix_indices       RadixCache 命中的 KV slot 物理索引（Tensor）
last_node            RadixTree 上最深匹配节点（用于 lock_ref）
is_chunked           0=非分块，>0=本请求正在分块 prefill
is_retracted         True=因 OOM 被踢回等待队列
finished_reason      None=进行中，否则=结束原因
req_pool_idx         在 ReqToTokenPool 中的行索引
```

---

## 阶段 ⑤：调度决策（最复杂的核心）

**文件**：`scheduler.py:2388` `get_next_batch_to_run()`

### 主调度决策流

```
get_next_batch_to_run()
    │
    ├─ 1. 超时检查
    │    _abort_on_waiting_timeout()   ← 等待超时的请求直接 abort
    │    _abort_on_running_timeout()   ← 运行超时的请求直接 abort
    │
    ├─ 2. 合并上轮 EXTEND 结果到 running_batch
    │    if last_batch.forward_mode == EXTEND:
    │        last_batch 里完成 prefill 的请求 → merge 进 running_batch
    │        (这些请求下轮开始 DECODE：每步生成 1 个 token)
    │
    ├─ 3. 尝试凑新 prefill 批次
    │    new_batch = get_new_batch_prefill()
    │       └─ _get_new_batch_prefill_raw()
    │
    └─ 4. 决策
         if new_batch:
             return new_batch          ← EXTEND（优先跑 prefill）
         elif running_batch 不空:
             return running_batch      ← DECODE（跑当前 decode 批次）
         else:
             return None               ← on_idle()
```

### _get_new_batch_prefill_raw：选哪些请求做 prefill

```
_get_new_batch_prefill_raw()
    │
    ├─ 快速返回判断：
    │    running_batch.batch_is_full → return None
    │    waiting_queue 空             → return None
    │
    ├─ policy.calc_priority(waiting_queue, running_batch)
    │    ← Cache-Aware 排序：
    │       优先调度能命中更多 RadixCache 前缀的请求
    │       → 最大化 KV 复用（同 system prompt 的请求聚批）
    │
    ├─ 遍历 waiting_queue，逐条调用 adder.add_one_req(req)
    │    ┌─────────────────────────────────────────────────┐
    │    │  add_one_req 内部：                             │
    │    │  1. tree_cache.match_prefix(req.fill_ids)       │
    │    │      → RadixCache 前缀匹配（见深入讲解）        │
    │    │  2. req.prefix_indices = 命中的 KV slot 索引   │
    │    │  3. req.extend_input_len = total - 命中 token 数│
    │    │  4. 估算需要的物理 KV 页数                     │
    │    │  5. 检查 token 预算和内存预算                  │
    │    │     如果超出：停止添加（或触发分块）            │
    │    └─────────────────────────────────────────────────┘
    │
    └─ 构建 ScheduleBatch(forward_mode=EXTEND, reqs=selected)
```

### 分块 Prefill（Chunked Prefill）

**触发条件**：`--chunked-prefill-size N`，单次 prefill 的 token 数超过 N。

```
请求 A：input_ids 长度 = 4096，N = 512
    │
    ├─ 第 1 轮：add_one_req 发现 extend_input_len > 512
    │    截断 req.extend_input_len = 512
    │    req.is_chunked = 1
    │    self.chunked_req = req
    │
    ├─ forward pass：处理 token[0:512]
    │    process_batch_result_prefill:
    │        tree_cache.cache_unfinished_req(req, chunked=True)
    │            → 把已计算的 512 token KV 插入 RadixTree
    │        req.is_chunked -= 1 → 0
    │        （还有剩余 → is_chunked 仍 > 0，继续）
    │
    ├─ 第 2 轮：init_next_round_input()
    │    重新做 prefix match：现在能命中 512 token
    │    extend_input_len = 4096 - 512 = 3584
    │    仍 > 512 → 再截断，is_chunked = 1
    │
    └─ 重复直到 is_chunked = 0 → 请求进入 DECODE 阶段
```

**效果**：长 prompt 请求不会独占 GPU 很久，DECODE 请求不会被长时间饿死（混合批次）。

### RadixCache 前缀匹配（深入讲解）

**文件**：`mem_cache/radix_cache.py`

```
RadixTree 结构（每个节点存储一段 token 序列对应的 KV slot 索引）：

root
├─ [sys_prompt_A token_ids] → KV slots [0..511]
│    ├─ [user_1 tokens] → KV slots [512..600]
│    └─ [user_2 tokens] → KV slots [512..650]
└─ [sys_prompt_B token_ids] → KV slots [700..900]

match_prefix(key=token_ids):
    1. key 按 page_size 对齐截断
    2. 沿树向下走，贪婪匹配
    3. 如匹配在节点中间结束 → split 节点
    4. 返回 MatchResult:
         device_indices = 命中的所有 KV slot 物理索引（Tensor）
         last_node     = 最深匹配节点
    5. inc_lock_ref(last_node)：
         从 last_node 到 root 每个节点 lock_ref += 1
         lock_ref 0→1：节点从"可驱逐"移到"受保护"

lock_ref 机制（防止 KV 被 LRU 驱逐）：
    请求持有 last_node → 节点被保护
    请求完成 → dec_lock_ref(last_node)
               lock_ref 1→0 → 回到"可驱逐"列表
               如果无子节点 → 加入 eviction_heap

evict(num_tokens)：
    从 eviction_heap（LRU 排序）弹出叶节点
    free(node.value)  → 归还物理 KV pages
    删除节点
    父节点变成叶节点 → 加入 eviction_heap
```

### KV Cache 物理内存管理

```
两层结构：

ReqToTokenPool：[max_reqs, max_context_len] int32 Tensor
    行 = 请求（由 req_pool_idx 索引）
    列 = 序列位置
    值 = 物理 KV slot 编号

TokenToKVPoolAllocator：
    free_pages：可用物理 slot 编号列表（int64 Tensor）

    alloc(need_size):
        select = free_pages[:need_size]
        free_pages = free_pages[need_size:]
        return select   # 或 None（OOM）

KVCache（MHATokenToKVPool）：
    k_buffer[layer] shape: [num_slots, head_num, head_dim]
    v_buffer[layer] shape: [num_slots, head_num, v_head_dim]
    slot 0 = dummy（padding token 写到这里，不影响正常 slot）

EXTEND 分配（alloc_for_extend）：
    1. 如 available < needed → evict_from_tree_cache(delta)
    2. alloc(extend_num_tokens) → out_cache_loc
    3. 仍 None → RuntimeError("Prefill out of memory")  ← crash
    4. write_req_to_token_pool（Triton kernel）：填 ReqToTokenPool

DECODE 分配（alloc_for_decode）：
    仅为每条请求分配 1 个新 slot（seq_len 位置）
```

---

## 阶段 ⑥：GPU Forward（run_batch）

**文件**：`scheduler.py:2864` `run_batch()`

### 数据变换三级流水

```
ScheduleBatch（CPU Python 对象，含 List[Req]）
    │
    ▼ get_model_worker_batch()
ModelWorkerBatch（CPU tensor，序列化后跨进程/线程传递）
    │  核心字段：
    │  input_ids:        [total_tokens] int32   ← 所有请求的 fill_ids 拼接
    │  req_pool_indices: [bs] int32             ← 每条请求的 req_pool_idx
    │  seq_lens:         [bs] int32             ← 每条请求当前总长度
    │  out_cache_loc:    [extend_tokens] int64  ← 新分配的 KV slot 索引
    │  extend_seq_lens:  [bs] int32             ← 每条请求需要 prefill 的长度
    │  extend_prefix_lens:[bs] int32            ← 每条请求已命中前缀长度
    │  sampling_info.grammars: List[grammar]    ← grammar 对象（如有）
    │
    ▼ TpModelWorker.forward_batch_generation(worker_batch)
ForwardBatch（GPU tensor，直接传给 model.forward()）
    │  通过 Triton kernel 从 req_to_token 读 KV slot 索引
    │  in_place 构建 attention metadata
    │
    ▼ model.forward(input_ids, positions, forward_batch)
    │  ├─ input embedding（or 多模态 embedding 替换）
    │  ├─ N × transformer layer
    │  │   ├─ RMSNorm
    │  │   ├─ QKV projection
    │  │   ├─ write K/V → KV cache（out_cache_loc 指定的 slot）
    │  │   └─ attention（FlashInfer/FlashAttention backend）
    │  │       EXTEND：causal attention over prefix_len + extend_len
    │  │       DECODE: attend to full seq_len（每条请求）
    │  └─ output projection → logits [total_tokens, vocab_size]
    │
    ▼ logits_processor
    │  ├─ 取每条请求最后 1 个 token 的 logits（EXTEND 取 extend 最后 1 个）
    │  ├─ 应用 temperature / top_p / top_k
    │  ├─ 如果有 grammar：apply vocab_mask（−∞ 掩盖不允许的 token）
    │  └─ 采样 → next_token_ids [bs]（GPU tensor）
    │
    ▼ 异步 D→H copy（cudaMemcpyAsync）
      next_token_ids CPU tensor（配合 copy_done event 同步）
```

### 多模态 forward 特殊路径
```
ForwardBatch.multimodal_inputs 存在时：
    在第一个 transformer layer 之前：
    ├─ 对 input_ids 中的 pad_value 位置，用 ViT 输出 embedding 替换
    │   pixel_values → vision_encoder → patch_embeddings
    │   根据 pad_value 找到位置 → embedding 写入对应行
    └─ 之后正常走 transformer（视觉 token 已嵌入）
```

---

## 阶段 ⑦：处理批次结果（process_batch_result）

**文件**：`scheduler_output_processor_mixin.py`

### 按 forward_mode 分支

```
process_batch_result(batch, result)
    ├─ DECODE  → process_batch_result_decode()
    ├─ EXTEND  → process_batch_result_prefill()
    ├─ MIXED   → process_batch_result_prefill()（含 decode 子集）
    └─ PREBUILT→ process_batch_result_prebuilt()（PD 解耦直接 decode）
```

### process_batch_result_prefill（EXTEND 批次）

```
for req, next_token_id in zip(batch.reqs, next_token_ids):

    if req.is_chunked > 0:
        # 分块 prefill 未完成，不产出 token
        tree_cache.cache_unfinished_req(req, chunked=True)
            → 插入已算完的 KV 到 RadixTree
        req.is_chunked -= 1
        continue

    # prefill 完成，产出第 1 个 token
    req.output_ids.append(next_token_id)
    req.check_finished()         ← 检查 EOS / max_tokens / stop_str

    if req.finished():
        release_kv_cache(req, tree_cache)
        # ↓ cache_finished_req：把 KV 插入 RadixTree（供后续请求复用）
        # ↓ dec_lock_ref(req.last_node)：解锁，允许 LRU 驱逐
        # ↓ free 不重复的物理 pages
    else:
        tree_cache.cache_unfinished_req(req)
        # 未完成：插入已算 KV，下轮 decode 继续
```

### process_batch_result_decode（DECODE 批次）

```
token_to_kv_pool_allocator.free_group_begin()
# ↑ 开启批量释放模式，本轮所有 free 先积累

for req, next_token_id in zip(batch.reqs, next_token_ids):
    req.output_ids.append(next_token_id)  # 追加本轮 token
    req.check_finished()
    _handle_finished_req(req, i, logits_output)

stream_output(batch.reqs)   # 发给 Detokenizer（见阶段 ⑦-续）
token_to_kv_pool_allocator.free_group_end()
# ↑ 批量释放：把本轮所有 freed KV page 一次性归还
```

### check_finished：停止条件检查
```
check_finished():
    ├─ 最后 token == EOS token                → FINISH_MATCHED_TOKEN
    ├─ len(output_ids) >= max_new_tokens       → FINISH_LENGTH
    ├─ output 文字包含 stop_str               → FINISH_MATCHED_STR
    ├─ to_finish 被外部设置（abort/超时）      → FINISH_ABORT
    └─ grammar.finished == True               → grammar 约束满足
```

### release_kv_cache（请求完成时的 KV 归还）
```
release_kv_cache(req, tree_cache):
    1. cache_finished_req(req)
       ├─ 读 req_to_token[req_pool_idx, :kv_committed_len]
       │    → 物理 KV slot 索引列表
       ├─ insert(key=token_ids, value=kv_indices) 到 RadixTree
       │    已在树里的部分（cache_protected_len 内）→ free 重复 pages
       └─ 保留新插入的 pages（供后续请求命中复用）
    2. dec_lock_ref(req.last_node)
       → 节点 lock_ref 归零 → 加入 eviction_heap
    3. free(kv_indices[cache_protected_len:new_prefix_len])
       → 归还不需要缓存的 pages
```

### stream_output → 发给 Detokenizer
```
stream_output_generation(reqs):
    for req in reqs:
        new_tokens = req.output_ids[req.send_token_offset:]
        # 仅发增量（已发的不重复发）
        req.send_token_offset = len(req.output_ids)

    BatchTokenIDOutput(
        rids=[req.rid, ...],
        decode_ids=[new_tokens_per_req, ...],
        read_offsets=[...],
        decoded_texts=[req.decoded_text, ...],   ← 已累积文字
        finished_reasons=[...],                  ← None 或 finish reason
        ...
    ) → ZMQ PUSH → DetokenizerManager
```

---

## 阶段 ⑧：DetokenizerManager（增量文字转换）

**文件**：`detokenizer_manager.py:138` `event_loop()`

```
event_loop():
    while True:
        recv_obj = recv_from_scheduler.recv_pyobj()  ← 阻塞等待
        output = _decode_batch_token_id_output(recv_obj)
        send_to_tokenizer.send_pyobj(output)          ← BatchStrOutput
```

### 增量 Detokenize 算法（关键！）

**问题**：直接 decode 最新 1 个 token 可能输出乱码（多字节 Unicode 被切断）。

**解决方案**：使用滑动窗口 + surr_offset（surrounding offset）

```
decode_status[rid]:
    decode_ids   = 所有已生成 token ids（累积）
    surr_offset  = 已安全输出到文字的 token 边界
    read_offset  = 上次完整 decode 到的位置

每轮：
    surr_ids = decode_ids[surr_offset : read_offset]  ← 已输出部分
    full_ids = decode_ids[surr_offset : len(decode_ids)]  ← 含新 token

    surr_text = tokenizer.decode(surr_ids)
    full_text  = tokenizer.decode(full_ids)
    new_text   = full_text[len(surr_text):]       ← 增量文字

    if new_text.endswith("▯"):                    ← Unicode 替换字符
        # 多字节字符被截断，还不能输出
        new_text = find_printable_text(new_text)  ← 取安全前缀
        不更新 surr_offset

    else:
        surr_offset = read_offset                 ← 提交安全边界
        read_offset = len(decode_ids)
        输出 new_text
```

`INIT_INCREMENTAL_DETOKENIZATION_OFFSET = 5`：初始 `surr_offset` 往前 5 个 token，给 tokenizer 上下文以正确合并字符。

---

## 阶段 ⑨：TokenizerManager 接收 → 唤醒 HTTP 协程

**文件**：`tokenizer_manager.py` `handle_loop()`

```
handle_loop():
    while True:
        recv_obj = await recv_from_detokenizer.recv_pyobj()
        await _handle_batch_output(recv_obj)

_handle_batch_output(BatchStrOutput):
    for rid, output_str in zip(recv_obj.rids, recv_obj.output_strs):
        state = rid_to_state[rid]
        state.out_list.append(output_str)     ← 追加增量文字
        state.event.set()                     ← 唤醒 HTTP 协程

↓（HTTP 协程被唤醒）

generate_request._wait_one_response():
    async for response in _yield_from_state(obj, state):
        yield response   ← 流式输出给客户端
```

---

## 特殊情况 1：OOM 与 Retraction（请求被踢回）

### 检测时机
```
update_running_batch() 在每轮 DECODE 之前：
    check_decode_mem()：
        预测 running_batch 下一步需要 bs 个新 KV slot
        if free_pages < needed:
            return False → 触发 retract_decode()
```

### Retraction 流程
```
retract_decode():
    1. 按 (len(output_ids) DESC, len(origin_input_ids) ASC) 排序
       ← 优先踢"输出最长"的请求（已生成多、再踢损失大于短的）

    2. 逐条 pop，release_kv_cache(req, is_insert=False)
       ← OOM 时不插入 RadixTree（只归还物理 pages，不保留缓存）

    3. req.reset_for_retract():
       req.is_retracted = True
       req.output_ids   = []        ← 清空已生成 token！重来
       req.kv_committed_len = 0
       req.prefix_indices   = empty
       req.last_node        = root

    4. 直到 check_decode_mem() = True 为止

    5. 极端情况：最后 1 条也 OOM
       → req.to_finish = FINISH_ABORT("Out of memory")
       → 该请求返回 500 错误给客户端

重新排队：
    _add_request_to_queue(req, is_retracted=True)
    → 回到 waiting_queue 等下一轮调度
    → 下次 match_prefix 可能重新命中 RadixCache（缓存可能还没被驱逐）
```

### 调整过载估计
```
retract_decode 返回 new_estimate_ratio：
    = (total_decoded + RETRACT_STEPS * bs) / total_max_new_tokens
    # 过度乐观的 new_token_ratio 被纠正，防止下一轮继续 OOM
```

---

## 特殊情况 2：客户端断连（Abort）

### 三条路径，覆盖所有阶段

```
TokenizerManager:
    HTTP handler 检测到 request.is_disconnected()
    → abort_request(rid)
    → send_to_scheduler.send_pyobj(AbortReq(rid))

Scheduler._handle_abort_req(rid):
    ┌─ 情况 A：req 在 waiting_queue
    │    pop from waiting_queue
    │    release 已分配 KV（disagg 模式）
    │    ZMQ push AbortReq back → TokenizerManager 清理 state
    │
    ├─ 情况 B：req 在 grammar_queue（等待 grammar 编译）
    │    cancel Future
    │    req.set_finish_with_abort()
    │    → 会再跑 1 次轻量 prefill（1 token），然后正常 finish
    │
    └─ 情况 C：req 在 running_batch（正在 DECODE）
         req.to_finish = FINISH_ABORT()
         → 下一轮 process_batch_result 时：
           req.check_finished() 检测到 to_finish
           → 走正常 finish 路径（release_kv_cache + stream_output）
           → 客户端收到最终 finish_reason=abort
```

---

## 特殊情况 3：Overlap 调度（重叠 CPU/GPU 执行）

**文件**：`scheduler.py:1510` `event_loop_overlap()`

### 核心思想：GPU 执行 N 批次时，CPU 处理 N-1 批次结果

```
event_loop_overlap():
    result_queue = deque()

    while True:
        recv_requests() + process_input_requests()   ← CPU
        batch = get_next_batch_to_run()              ← CPU 调度

        # 特殊情况：必须关闭 overlap
        if is_disable_overlap_for_batch(batch):
            pop_and_process()  ← 先把上批结果处理完

        if batch:
            batch_result = run_batch(batch)          ← 非阻塞：提交 GPU 任务
            result_queue.append((batch.copy(), batch_result))
            # batch.copy()：浅拷贝快照，防止 reqs 列表被下轮修改

        if last_batch and not disable_overlap:
            pop_and_process()
            # ↑ 此时 GPU 正在跑 batch N
            #   CPU 并行处理 batch N-1 的结果：
            #   - next_token_ids.wait()（D2H copy 等待）
            #   - check_finished, RadixCache 更新
            #   - ZMQ → Detokenizer
```

### 何时关闭 overlap（disable_overlap）
| 条件 | 原因 |
|---|---|
| 连续两个 EXTEND 批次 | 改善 TTFT（首 token 延迟） |
| spec_v2 + grammar + decode 在 result_queue | grammar 状态机必须在下批采样前推进 |
| `SGLANG_DISABLE_OVERLAP=1` | 调试用，强制关闭 |

**收益**：DECODE 密集场景吞吐提升 10–30%（隐藏了大部分 CPU 处理时延）。

---

## 特殊情况 4：投机解码（Speculative Decoding，EAGLE v2）

### 整体机制
```
普通 decode：1 个 GPU forward pass → 1 个 token
EAGLE v2：  1 个 GPU forward pass → 接受 N 个 token（N ≥ 1）

两个模型：
    Draft 模型（小）：快速生成候选 token 树
    Target 模型（大）：验证候选，选择接受

时间线：
    Draft pass:  D, D, D, D  ← 4 个候选 token
    Target pass: 验证全部 4 个 → 接受 3 个 + 生成第 4 个
    净效果：    1 次 forward ≈ 生成 3-4 个 token
```

### EAGLE v2 详细流程
```
run_batch(batch, spec_algorithm=EAGLE):
    │
    ├─ 1. draft 阶段
    │    draft_worker.draft(model_worker_batch)
    │    → EagleVerifyInput(
    │         draft_token,      ← 候选 token tree
    │         tree_mask,        ← 注意力掩码（树状依赖）
    │         position,         ← 每个候选 token 的位置编号
    │         retrieve_index,   ← 验证结果 → output 的映射
    │     )
    │
    ├─ 2. verify 阶段（target 模型）
    │    verify_input.prepare_for_v2_verify(...)
    │    → forward_mode = TARGET_VERIFY
    │    target_worker.forward_batch_generation(verify_forward_batch)
    │    → logits [draft_tree_size, vocab_size]
    │
    ├─ 3. 投机采样
    │    verify_input.sample(batch, logits_output, vocab_mask)
    │    对每个位置：
    │        p_target = softmax(logits[pos])
    │        p_draft  = draft_prob[pos]
    │        accept_prob = min(1, p_target[draft_token] / p_draft[draft_token])
    │        if random() < accept_prob: accept
    │        else: reject, sample from (p_target - p_draft)
    │    → (predict, accept_lens, accept_index)
    │      accept_lens[i] = 请求 i 本轮接受的 token 数
    │
    └─ 4. 结果
         GenerationBatchResult(
             next_token_ids=predict,   ← List[List[int]]（每条请求接受的 tokens）
             accept_lens=accept_lens,
         )
         process_batch_result_decode:
             req.output_ids.extend(next_token_id)  ← 一次追加多个 token
```

### Grammar + 投机解码
```
if batch.has_grammar:
    generate_token_bitmask(grammars, draft_token_tree)
    → vocab_mask：在整个 draft tree 的每个位置施加约束
    verify 阶段的 logits 会被 vocab_mask 掩盖（−∞）
```

---

## 特殊情况 5：Grammar 约束生成

**文件**：`grammar_manager.py`，`sampling_batch_info.py`

```
1. 请求携带 json_schema / regex / ebnf

2. TokenizerManager:
    _tokenize_one_request:
        process_req_with_grammar(req)
        → grammar_backend.get_cached_or_future_value(key)
        → 命中缓存：req.grammar = GrammarObject（同步）
        → 未命中：req.grammar = Future，req → grammar_queue

3. Scheduler（每轮）:
    get_ready_grammar_requests()
    → 轮询 Future.done()
    → 就绪的 req 移到 waiting_queue

4. 采样前:
    sampling_info.update_regex_vocab_mask()
    → vocab_mask[i] = req.grammar.fill_vocab_mask()
    → 在 logits 上加 −∞（不允许的 token）

5. 采样后:
    req.grammar.accept_token(next_token_id)
    → grammar 状态机前进到下一个状态
    → grammar.finished = True 时设置 req.check_finished()
```

---

## 特殊情况 6：PD 解耦（Prefill-Decode Disaggregation）

```
两个集群：
    Prefill 实例集群（大 GPU，用于跑 prefill）
    Decode  实例集群（小 GPU，用于跑 decode）

Prefill 实例处理过程：
    ① 正常跑 EXTEND forward
    ② 产出 KV Cache
    ③ 通过 RDMA/NVLink 传输 KV → Decode 实例
    ④ 发送 bootstrap 信号（host, port, room）

Decode 实例处理过程：
    ① 收到 KV 传输完成信号
    ② req.forward_mode = PREBUILT
    ③ process_batch_result_prebuilt：直接从 DECODE 开始
    ④ 不需要做 prefill

关键字段：
    req.disagg_kv_sender         ← KV 传输 sender 对象
    req.start_send_idx           ← 已发送到的 token 位置（支持分批传输）
    req.bootstrap_host/port/room ← Decode 实例连接信息
```

---

## 特殊情况 7：多卡并行（TP / PP / DP）

### Tensor Parallelism（TP）
```
每个 TP rank 是独立进程，运行相同的 Scheduler+ModelRunner
    TP rank 0：接收 ZMQ 消息（attn_tp_rank == 0）
    recv_requests → 通过 nccl broadcast 广播给其他 TP ranks
    每个 rank 处理 1/TP 份 attention heads
    TP all-reduce 后汇总 logits
```

### Pipeline Parallelism（PP）
```
pp_rank 0：接收请求，负责 embedding + 前几层
pp_rank N-1：负责最后几层 + sampling
中间层通过 PPProxyTensors 传递 hidden states
pp_rank 0 同时持有 chunked_req 状态
```

### Data Parallelism（DP）
```
dp_size 个独立 Scheduler 实例，各自维护独立的 waiting_queue
HTTP 层 round-robin 或按 routed_dp_rank 指定路由
Attention DP（attn_dp_rank）：attention 在 dp 维度切分
```

---

## 特殊情况 8：LoRA 动态加载

```
请求带 lora_id="my_adapter":
    _validate_and_resolve_lora(obj):
        if lora_id not in loaded_adapters:
            发送 LoadLora 请求 → Scheduler → TpModelWorker
            await lora_loaded_event  ← 阻塞直到加载完成
        obj.lora_id = resolved_id

每个 Req：
    req.lora_id → lora_ids Tensor → ModelWorkerBatch
    ModelRunner 在每层 QKV projection 时：
        output = base_weight(x) + lora_A(x) @ lora_B * scale
        根据 lora_ids 对不同 batch 行用不同 LoRA
```

---

## 完整数据字段附录

### Req 对象关键字段速查
```python
# 身份
rid: str                          # 请求唯一 ID（UUID）
origin_input_ids: List[int]       # tokenize 后的 token_ids
output_ids: List[int]             # 已生成的 token（逐步追加）
fill_ids: List[int]               # = origin_input_ids + output_ids

# KV Cache 状态
req_pool_idx: int                 # ReqToTokenPool 行索引
kv_committed_len: int             # 已插入 RadixTree 的 token 数
kv_allocated_len: int             # 已分配物理 KV pages 的 token 数
cache_protected_len: int          # RadixTree 中已保护（命中）的 token 数
prefix_indices: Tensor[int64]     # 命中的物理 KV slot 索引
last_node: TreeNode               # RadixTree 最深匹配节点
extend_input_len: int             # 本轮需实际 prefill 的 token 数

# 生命周期
is_chunked: int                   # 0=完整，>0=分块中
is_retracted: bool                # 是否被踢回（OOM）
finished_reason: BaseFinishReason # None=进行中
to_finish: BaseFinishReason       # 延迟 finish 信号

# 流式输出
send_token_offset: int            # 已发给 Detokenizer 的 token 偏移
decoded_text: str                 # 已累积的文字
read_offset: int                  # 增量 detokenize 进度

# 多模态
multimodal_inputs: MultimodalInputs  # 含 mm_items（图像 feature tensors）

# 采样
sampling_params: SamplingParams
grammar: GrammarObject            # grammar 约束对象（如有）
lora_id: str                     # LoRA adapter ID（如有）
```

### BatchTokenIDOutput 字段速查
```python
rids: List[str]                  # 本批次所有请求 rid
decode_ids: List[List[int]]      # 每条请求的增量 token ids
read_offsets: List[int]          # detokenize 进度
decoded_texts: List[str]         # 已累积文字（Detokenizer 用）
finished_reasons: List[...]      # None=流式中，否则=结束原因
prompt_tokens: List[int]         # input token 数
completion_tokens: List[int]     # 已生成 token 数
cached_tokens: List[int]         # RadixCache 命中的 token 数
```

---

## 性能优化点一览

| 优化 | 机制 | 收益 |
|---|---|---|
| RadixCache 前缀复用 | token_ids trie，LRU 驱逐，lock_ref 保护 | 相同 system prompt 请求共享 KV，省显存 + 加速 |
| Cache-Aware 调度 | 优先调度能命中更多前缀的请求 | 最大化 KV 复用率 |
| 分块 Prefill | 长 prompt 切成小块，与 decode 混合 | DECODE 请求不被饿死，吞吐均衡 |
| Overlap 调度 | GPU 跑 N 批时 CPU 处理 N-1 批结果 | DECODE 吞吐 +10–30% |
| 投机解码 | Draft model 预测 N token，Target 并行验证 | 低延迟场景吞吐 2–5x |
| 批量 KV 释放 | free_group_begin/end | 减少内存分配碎片，提升 DECODE 吞吐 |
| Triton 并行写 ReqToTokenPool | write_req_to_token_pool_triton kernel | 减少 CPU-GPU 同步等待 |
| 增量 Detokenize | surr_offset 滑动窗口 | 正确处理多字节 Unicode，最小化 decode 开销 |
| Grammar 异步编译 | Future + grammar_queue | grammar 编译不阻塞调度循环 |
| KV 批量传输（PD 解耦） | start_send_idx 分块发送 | 不阻塞 prefill 实例，流水线化 KV 传输 |

---

*文档生成基于 SGLang 源码静态分析。如有疑问直接跳到对应源码的【学习注释】标记。*
