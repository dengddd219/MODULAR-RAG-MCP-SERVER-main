# Modular RAG MCP Server

中文 | [English](#english)

一个面向知识库问答与 AI Assistant 工具调用场景的模块化 RAG 平台。项目从文档摄取、混合检索、答案生成、评测分析，到 MCP 工具暴露与可视化 Dashboard，形成了一条完整、可扩展、可观测的端到端链路。

## 中文
### 项目简介
`Modular RAG MCP Server` 是我基于MODULAR-RAG-MCP-SERVER-main设计并持续迭代的一个本地化、模块化 RAG 系统。它的目标不是只做一个“能回答问题”的检索增强生成 Demo，而是搭建一套可以真正用于工程验证、模型比较、检索调优、链路追踪和 AI 助手接入的完整平台。

这个项目围绕六个核心目标展开：
- 模块化：检索、重排、生成、评测、可视化彼此解耦，便于替换和扩展。
- 可观测：查询链路和摄取链路都能记录详细 trace，方便定位问题。
- 可评测：支持基于 golden set 的批量评估，以及基于 RAGAS 的质量打分。
- 可路由：支持意图识别、查询复杂度判断和多模型组合策略。
- 可接入：通过 MCP Server 暴露工具，便于 Copilot、Claude 等客户端调用。
- 可演示：提供完整 Streamlit Dashboard，适合展示、调试和对比实验。

### 解决的问题
传统的 RAG Demo 往往只关注“把文档塞进去，然后回答问题”，但一旦进入真实工程场景，就会暴露出很多问题：
- 文档摄取不可追踪，出错时不知道卡在哪一步。
- 检索只有单路向量召回，效果不稳定。
- 模型成本、时延和质量之间难以权衡。
- 缺少统一的评测基线，无法对不同策略做横向对比。
- AI Assistant 无法方便地把系统能力作为工具接入。

这个项目就是围绕这些问题建立的。它把“知识库系统”拆成多个可独立演进的能力模块，再通过统一配置和统一接口把它们串起来。

### 整体架构
项目主要由六层组成：

1. 文档摄取层
负责文件解析、分块、元数据增强、嵌入编码、向量写入、BM25 写入和图片索引。

2. 检索与路由层
负责查询预处理、意图识别、复杂度判断、Dense 检索、Sparse 检索、Graph 检索、融合和重排。

3. 响应生成层
负责把召回结果组织成上下文，调用 LLM 生成答案，并构建引用与多模态返回内容。

4. MCP 工具层
负责把查询、知识库列表、文档摘要等能力暴露成 MCP 工具，供外部 AI 客户端调用。

5. 可观测与评测层
负责记录查询与摄取 trace，运行 RAGAS / Custom evaluator，并输出指标与历史结果。

6. Dashboard 展示层
基于 Streamlit 提供总览、数据浏览、摄取管理、追踪分析、Chat、评测面板和 LLM Arena。

### 端到端工作流
#### 1. 摄取流程
一篇文档进入系统后，会经历以下步骤：
- Loader 解析原始文件内容，支持 `PDF` 和 `Markdown`。
- Chunker 将长文切分为适合检索和生成的文本块。
- Transform 模块可做块精炼、元数据增强和图片描述注入。
- Embedding 模块生成 Dense / Sparse 表示。
- Storage 模块把内容写入 Chroma、BM25 和图片索引。
- Integrity 模块记录文件哈希与摄取状态，支持去重与增量更新。

#### 2. 查询流程
一次查询进入系统后，会经历以下步骤：
- `QueryProcessor` 做规范化、关键词提取和过滤器解析。
- `IntentRouter` 进行垃圾流量拦截和业务意图判断。
- `QueryComplexityClassifier` 用于 simple / complex 路由决策。
- `HybridSearch` 组合 Dense、Sparse、Graph 三类召回。
- `Fusion` 使用 RRF 进行融合。
- `Reranker` 对候选结果进行二次排序。
- `RAGGenerator` 基于检索结果生成最终答案。
- `CitationGenerator` 和多模态组装模块输出引用和图片结果。

### 核心能力
#### 1. 模块化检索
- Dense Retrieval
- Sparse Retrieval (BM25)
- Graph Retrieval
- RRF Fusion
- Cross-Encoder / LLM Rerank

#### 2. 模型路由与意图能力
- 双层意图路由
- 文档级 `doc_intent` 分类
- 查询复杂度分类
- 小模型 + 大模型混合策略
- 多模型横向 benchmark

#### 3. 可观测性
- Query Trace
- Ingestion Trace
- 分阶段耗时记录
- 中间状态可视化
- Dashboard 中的瀑布图和详情面板

#### 4. 评测能力
- Golden test set 批量评测
- RAGAS 三项质量指标
- 自定义轻量评估器
- LLM Arena 批量压测
- 成本、时延、质量综合对比

### Dashboard 页面
当前 Dashboard 提供 8 个页面：
- `Overview`: 查看系统配置、知识库统计和追踪概况。
- `Chat`: 在 Dashboard 内直接进行 RAG 对话，支持历史会话与模型切换。
- `Data Browser`: 浏览文档、Chunk、元数据和图片。
- `Ingestion Manager`: 上传文件、执行摄取、查看并管理文档。
- `Ingestion Traces`: 查看摄取链路的阶段耗时和详细数据。
- `Query Traces`: 查看查询链路的检索、融合、重排和回答过程。
- `Evaluation Panel`: 运行检索评估并查看历史结果。
- `LLM Arena`: 进行单次对弈与批量压测，对比多模型组合策略。

Dashboard 还支持全局中英文界面切换。

### 目录结构
```text
src/
├── core/                  # 查询、响应、配置、trace 等核心流程
├── ingestion/             # 摄取 pipeline、分块、存储、transform
├── libs/                  # 各类可插拔后端实现
├── mcp_server/            # MCP 协议和工具暴露
└── observability/         # Dashboard、评测与日志

scripts/                   # 启动、摄取、查询、评测、训练与诊断脚本
tests/                     # unit / integration / e2e 测试
data/                      # 本地索引、模型、训练集、聊天记录等
logs/                      # trace、benchmark、evaluation 历史记录
config/                    # 主配置与 prompt 配置
```

### 技术特点
这个项目的重点不只是“功能全”，而是工程结构可解释、可验证、可替换：
- 配置驱动：LLM、Embedding、向量库、Reranker 均可通过配置切换。
- 工厂模式：不同 provider 通过 factory 统一实例化。
- 本地优先：支持本地模型、本地向量库、本地评测历史。
- 可扩展：新增检索器、重排器、评测器的成本较低。
- 实验友好：适合做 A/B 检索对比和多模型策略验证。

### 适用场景
这个项目适合以下用途：
- 个人或团队搭建本地知识库问答系统
- 为 AI Assistant 提供可调用的 MCP 工具
- 做 RAG 检索策略实验与评测
- 做多模型路由与成本优化研究
- 做课程、作品集或技术展示项目

### 快速启动
```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -e .
pip install -e ".[dev]"
pip install -e ".[train]"
```

配置好 `config/settings.yaml` 后，可以这样启动：

```bash
# 文档摄取
python scripts/ingest.py --path tests/fixtures/sample_documents/ --collection default

# 命令行查询
python scripts/query.py --query "What is Modular RAG?" --verbose

# 启动 Dashboard
python scripts/start_dashboard.py

# 启动 MCP Server
python -m src.mcp_server.server
```

### 常用脚本
- `scripts/ingest.py`: 执行文档摄取
- `scripts/query.py`: 命令行查询
- `scripts/evaluate.py`: 跑评测流程
- `scripts/start_dashboard.py`: 启动 Dashboard
- `scripts/train_intent_model.py`: 训练意图识别模型
- `scripts/train_query_complexity_model.py`: 训练复杂度分类模型
- `scripts/train_spam_model.py`: 训练垃圾流量分类模型
- `scripts/test_llm_arena.py`: 检查 LLM Arena 调用链路

### 当前项目状态
目前项目已经具备完整的原型系统能力：
- 从文档摄取到查询回答的主链路已打通
- Dashboard 多页面已可用
- LLM Arena 与 Evaluation Panel 已可用于对比实验
- 意图识别、复杂度分类和模型路由能力已接入
- RAGAS 与自定义评估已具备基础能力

这个项目仍然保留了继续迭代的空间，例如：
- 提升意图分类准确率
- 补全检索级 ground truth，完善 Hit Rate / MRR 统计
- 丰富 Graph Retrieval 的真实召回能力
- 增强 MCP Server 的生产化能力

到目前为止，这个项目已经不再只是一个能够跑通的 RAG Demo，而是一套相对完整的本地化 RAG 系统原型。从整体结构来看，它已经覆盖了文档摄取、混合检索、重排、答案生成、意图路由、评测体系以及可观测性等核心环节，并通过 MCP 工具接口和 Dashboard 进行统一展示。换句话说，这个项目已经从最初的“实现一个功能”逐渐发展成了一个具备模块化结构和工程组织的系统。整个开发过程中，我的关注点也经历了一次明显的转变：从一开始的“系统能不能跑起来”，逐渐转向“系统能力如何被证明、如何被量化”。

在搭建系统的过程中，也逐渐暴露出一些非常典型的工程问题。例如，早期的很多功能是以“修补式迭代”的方式加入的，但随着模块数量增加，我开始意识到需要通过配置驱动、统一抽象接口以及全局状态管理来保证系统结构的稳定性；在模型训练部分，最初只是关注模型是否能够跑通，但后来发现训练环境依赖、数据格式、脚本可复现性同样是工程的重要组成部分，否则模型能力很难真正沉淀为系统能力。另一个比较深的体会来自评测环节：在实际调试中发现，一些 benchmark 脚本的统计口径和真实链路并不完全一致，这让我意识到，评测体系本身如果设计不严谨，很容易产生“看起来很好但并不真实”的结果。相比单纯追求更高的分数，建立统一、可解释且不泄漏 ground truth 的评测标准其实更重要。

从当前状态来看，这个系统的架构完整度已经比较高，但在数据基础和指标闭环上仍然有不少可以继续完善的地方。例如，意图识别模块目前虽然已经建立了完整的训练和评估流程，但准确率提升空间仍然较大，后续可能需要从标签边界和语料质量上进一步优化，而不仅仅是依赖模型调参；检索链路虽然已经能够计算 Hit Rate 和 MRR 等指标，但如果缺少足够细粒度的 golden dataset，检索效果仍然很难得到有说服力的量化证明。此外，在系统性能方面，也需要进一步区分真实用户请求链路与评测链路的时间开销，否则评测工具本身可能会放大系统时延，影响对真实体验的判断。

回顾整个开发过程，这个项目最大的收获其实不只是实现了一套完整的 RAG 系统，而是逐渐建立了一种更偏工程化的思考方式：不再只关注系统是否能够回答问题，而是开始系统性地去区分哪些问题来自功能实现，哪些来自数据质量，哪些来自评测方法本身。只有把这些问题拆开，才能真正理解系统的能力边界，以及下一步应该优先优化的方向。从“把系统搭出来”，到“用数据证明系统到底有多强”，这其实是两个完全不同阶段的工作，而现在这个项目正好处在从前者迈向后者的过程中。


---

## English
### Project Overview
`Modular RAG MCP Server` is an end-to-end modular RAG platform that I designed and iterated from scratch. It is not just a simple retrieval-augmented QA demo. Instead, it is a full engineering-oriented system that covers document ingestion, hybrid retrieval, answer generation, benchmarking, observability, model routing, MCP tool exposure, and a visual dashboard.

The project is built around six goals:
- Modularity: retrieval, reranking, generation, evaluation, and visualization are loosely coupled.
- Observability: both ingestion and query pipelines are traceable.
- Evaluability: the system supports golden-set benchmarking and RAGAS-based quality scoring.
- Routability: intent recognition, complexity classification, and multi-model routing are supported.
- Integrability: system capabilities can be exposed as MCP tools to AI assistants.
- Demonstrability: a full Streamlit dashboard is provided for inspection, experiments, and demos.

### What This Project Solves
Many RAG demos stop at "ingest files and answer questions." Real engineering systems need much more:
- observable ingestion and query pipelines
- more stable retrieval than a single vector search path
- cost / latency / quality tradeoff analysis
- consistent evaluation baselines
- easy integration with AI assistant clients

This project addresses those gaps by decomposing a knowledge system into replaceable modules and reconnecting them through shared configuration, factories, and unified interfaces.

### High-Level Architecture
The system is organized into six major layers:

1. Ingestion Layer  
Parses documents, splits them into chunks, enriches metadata, generates embeddings, and writes data into vector, sparse, and image stores.

2. Retrieval and Routing Layer  
Handles query preprocessing, intent routing, complexity classification, dense retrieval, sparse retrieval, graph retrieval, fusion, and reranking.

3. Response Generation Layer  
Builds prompt context from retrieved chunks, generates answers through LLMs, and assembles citations or multimodal outputs.

4. MCP Tool Layer  
Exposes the system as MCP tools for external AI clients such as Claude or Copilot.

5. Observability and Evaluation Layer  
Records traces, computes evaluation metrics, stores benchmark history, and supports quality inspection.

6. Dashboard Layer  
Provides a Streamlit UI for system overview, data browsing, chat, evaluation, and model arena experimentation.

### End-to-End Workflow
#### 1. Ingestion Pipeline
When a document enters the system, it goes through:
- file loading
- chunking
- optional text refinement
- metadata enrichment
- image caption injection
- dense and sparse encoding
- vector / BM25 / image indexing
- integrity tracking for deduplication and incremental ingestion

#### 2. Query Pipeline
When a user sends a query, the system runs:
- query normalization and filter parsing
- intent routing
- complexity classification
- dense retrieval
- sparse retrieval
- optional graph retrieval
- RRF fusion
- reranking
- answer generation
- citation and multimodal assembly

### Core Capabilities
#### 1. Retrieval Stack
- Dense retrieval
- Sparse retrieval with BM25
- Graph retrieval
- Reciprocal Rank Fusion
- Optional cross-encoder or LLM reranking

#### 2. Routing and Intent Intelligence
- two-layer intent router
- document-level intent classification
- query complexity classification
- small-model + large-model hybrid routing
- multi-model benchmark support

#### 3. Observability
- query traces
- ingestion traces
- stage-level timing
- intermediate data inspection
- dashboard waterfall-style analysis

#### 4. Evaluation
- golden-set benchmarking
- RAGAS quality scoring
- lightweight custom evaluator
- LLM Arena batch benchmarking
- cost / latency / quality comparison

### Dashboard Pages
The dashboard currently contains 8 pages:
- `Overview`
- `Chat`
- `Data Browser`
- `Ingestion Manager`
- `Ingestion Traces`
- `Query Traces`
- `Evaluation Panel`
- `LLM Arena`

The dashboard also supports global bilingual UI switching between English and Chinese.

### Repository Structure
```text
src/
├── core/                  # core query, response, settings, tracing logic
├── ingestion/             # ingestion pipeline, chunking, transform, storage
├── libs/                  # pluggable backend implementations
├── mcp_server/            # MCP server protocol and tools
└── observability/         # dashboard, evaluation, logging

scripts/                   # launchers, ingestion, query, eval, training, diagnostics
tests/                     # unit, integration, and end-to-end tests
data/                      # local indexes, models, datasets, chat history
logs/                      # traces and benchmark history
config/                    # settings and prompt files
```

### Why This Project Is Strong
The value of this project is not only feature coverage, but also engineering structure:
- configuration-driven architecture
- factory-based backend abstraction
- local-first experimentation workflow
- traceable pipeline behavior
- benchmark-ready model comparison
- easy extensibility for new retrievers, rerankers, and evaluators

### Typical Use Cases
This project is suitable for:
- building a local knowledge-base QA system
- exposing enterprise knowledge tools to AI assistants through MCP
- experimenting with retrieval strategies
- testing hybrid model-routing strategies
- showcasing an end-to-end RAG engineering portfolio project

### Quick Start
```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -e .
pip install -e ".[dev]"
pip install -e ".[train]"
```

After configuring `config/settings.yaml`, you can run:

```bash
# ingest documents
python scripts/ingest.py --path tests/fixtures/sample_documents/ --collection default

# run CLI query
python scripts/query.py --query "What is Modular RAG?" --verbose

# start dashboard
python scripts/start_dashboard.py

# start MCP server
python -m src.mcp_server.server
```

### Useful Scripts
- `scripts/ingest.py`
- `scripts/query.py`
- `scripts/evaluate.py`
- `scripts/start_dashboard.py`
- `scripts/train_intent_model.py`
- `scripts/train_query_complexity_model.py`
- `scripts/train_spam_model.py`
- `scripts/test_llm_arena.py`

### Current Project Status
The project already has a functioning end-to-end prototype:
- the full ingestion-to-answer pipeline is connected
- the multi-page dashboard is usable
- LLM Arena and Evaluation Panel support side-by-side strategy comparison
- routing, intent, and complexity modules are integrated
- RAGAS and custom evaluation are available

There is still room for future improvement:
- improve intent-classification performance
- add chunk-level ground truth for retrieval metrics such as Hit Rate and MRR
- strengthen graph retrieval with richer graph signals
- harden MCP server behavior for more production-like usage

### Final Note
This project represents a full-stack RAG engineering effort rather than a narrow QA demo. It combines retrieval, orchestration, observability, evaluation, routing, and AI-tool integration into one coherent system. That makes it suitable not only as a technical showcase, but also as a practical foundation for continued iteration into a more production-ready knowledge platform.

## License
MIT
