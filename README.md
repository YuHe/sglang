<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="logo" width="400" margin="10px"></img>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://static.pepy.tech/badge/sglang?period=month)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)

</div>

---

<p align="center">
<a href="https://lmsys.org/blog/"><b>博客</b></a> |
<a href="https://docs.sglang.io/"><b>文档</b></a> |
<a href="https://roadmap.sglang.io/"><b>路线图</b></a> |
<a href="https://slack.sglang.io/"><b>加入 Slack</b></a> |
<a href="https://meet.sglang.io/"><b>每周开发会议</b></a> |
<a href="https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#slides"><b>演讲 Slides</b></a>
</p>

---

## 简介

SGLang 是一个面向大语言模型和多模态模型的**高性能推理服务框架**，专为从单卡到大规模分布式集群的各种部署场景设计，核心目标是实现低延迟、高吞吐的模型推理。

**核心能力：**

- **高效 Runtime**：RadixAttention 前缀缓存、零开销 CPU 调度器、Prefill-Decode 解耦（PD Disaggregation）、投机解码（Speculative Decoding）、Continuous Batching、Paged Attention、Tensor/Pipeline/Expert/Data 并行、结构化输出（Structured Outputs）、Chunked Prefill、量化（FP4/FP8/INT4/AWQ/GPTQ）、多 LoRA 批量服务。
- **广泛的模型支持**：覆盖主流语言模型（Llama、Qwen、DeepSeek、Kimi、GLM、GPT、Gemma、Mistral 等）、Embedding 模型、Reward 模型及扩散模型（WAN、Qwen-Image），兼容大多数 Hugging Face 模型和 OpenAI API。
- **多硬件支持**：支持 NVIDIA GPU（GB200/B300/H100/A100）、AMD GPU（MI355/MI300）、Intel Xeon CPU、Google TPU、Ascend NPU 等。
- **活跃社区**：开源项目，已在全球超过 40 万张 GPU 上部署，被 xAI、AMD、NVIDIA、LinkedIn、Google Cloud、Microsoft Azure 等主流机构广泛采用。
- **RL / 后训练支持**：作为多个前沿模型训练的 Rollout 后端，与 [verl](https://github.com/volcengine/verl)、[AReaL](https://github.com/inclusionAI/AReaL)、[slime](https://github.com/THUDM/slime) 等后训练框架深度集成。

**快速上手：**

- [安装 SGLang](https://docs.sglang.io/get_started/install.html)
- [快速开始](https://docs.sglang.io/basic_usage/send_request.html)
- [API 使用教程](https://docs.sglang.io/basic_usage/openai_api_completions.html)
- [贡献指南](https://docs.sglang.io/developer_guide/contribution_guide.html)

---

## 📖 本 Fork：中文源码学习资料

> 本仓库在官方 SGLang 基础上，新增了一套**面向源码学习**的中文注释和文档体系。
> 目标：不逐行读源码，1 小时内掌握 SGLang 从请求入 → Token 返回的所有核心机制。

### 学习文档导航

| 文档 | 说明 | 适合场景 |
|---|---|---|
| [SGLANG_SOURCE_GUIDE.md](./SGLANG_SOURCE_GUIDE.md) | 代码库全局导览：目录结构、核心模块、关键文件索引 | 第一次接触 SGLang 源码 |
| [SGLANG_BATCH_FLOW.md](./SGLANG_BATCH_FLOW.md) | 批量请求数据流蓝图（ASCII 流程图） | 快速建立宏观认知 |
| [SGLANG_ANNOTATED.md](./SGLANG_ANNOTATED.md) | 关键数据结构（Req、ScheduleBatch 等）注释说明 | 搞清楚字段含义时 |
| [SGLANG_REQUEST_FLOW.md](./SGLANG_REQUEST_FLOW.md) | **极端细化的完整请求数据流**，含所有特殊情况 | 深度掌握运转逻辑 |

### 源码内联注释

以下文件已添加 `【学习注释 ①②…】` 标记，可全局搜索 `【学习注释` 快速定位所有注释点：

| 文件 | 注释覆盖的关键函数 |
|---|---|
| [tokenizer_manager.py](./python/sglang/srt/managers/tokenizer_manager.py) | `generate_request`（全链路入口）、`handle_loop`（流式唤醒） |
| [scheduler.py](./python/sglang/srt/managers/scheduler.py) | `event_loop_normal`、`recv_requests`、`get_next_batch_to_run`、`run_batch`、`process_batch_result` |
| [scheduler_output_processor_mixin.py](./python/sglang/srt/managers/scheduler_output_processor_mixin.py) | `process_batch_result_prefill/decode`、`_handle_finished_req`、`stream_output_generation` |
| [detokenizer_manager.py](./python/sglang/srt/managers/detokenizer_manager.py) | `event_loop`（增量 detokenize） |

### 建议学习路线

```
1. 读 SGLANG_SOURCE_GUIDE.md       → 了解整体架构和进程拓扑（15 分钟）
2. 读 SGLANG_BATCH_FLOW.md         → 建立数据流直觉（10 分钟）
3. 读 SGLANG_REQUEST_FLOW.md       → 深入所有细节和特殊情况（30 分钟）
4. 左边开 SGLANG_REQUEST_FLOW.md   → 右边对照带【学习注释】的源码（随时查阅）
```

---

## 性能基准

详见各版本发布博客：[v0.2](https://lmsys.org/blog/2024-07-25-sglang-llama3/) · [v0.3](https://lmsys.org/blog/2024-09-04-sglang-v0-3/) · [v0.4](https://lmsys.org/blog/2024-12-04-sglang-v0-4/) · [大规模 EP](https://lmsys.org/blog/2025-05-05-large-scale-ep/) · [GB200 机架级并行](https://lmsys.org/blog/2025-09-25-gb200-part-2/) · [GB300 长上下文](https://lmsys.org/blog/2026-02-19-gb300-longctx/)

## 致谢

设计参考并复用了以下开源项目的代码：[Guidance](https://github.com/guidance-ai/guidance)、[vLLM](https://github.com/vllm-project/vllm)、[LightLLM](https://github.com/ModelTC/lightllm)、[FlashInfer](https://github.com/flashinfer-ai/flashinfer)、[Outlines](https://github.com/outlines-dev/outlines)、[LMQL](https://github.com/eth-sri/lmql)。
