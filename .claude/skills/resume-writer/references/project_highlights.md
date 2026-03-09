# 项目技术亮点清单（Modular RAG MCP Server）

> 基于当前 `DEV_SPEC`、`README` 与项目现状整理，供简历编写时按需选取。每个亮点附带"简历话术方向"和"可量化角度"，尽量优先使用当前版本已落地的能力，避免写成过时版本。

---

## 亮点 1：三路混合检索架构（Dense + Sparse + Graph + Rerank）

**技术要点**：
- 将 Hybrid Search 从传统双路检索升级为三路召回：Dense Retrieval + Sparse Retrieval（BM25）+ Graph Retriever
- 融合层支持动态多路 RRF（Reciprocal Rank Fusion），不依赖不同检索分数的绝对值标尺
- 检索后提供可插拔精排：`Cross-Encoder` / `LLM Rerank` / `None`
- 当精排超时、失败或关闭时自动回退到融合排序，保证主链路稳定可用
- 评估与可视化面板同步支持二路、三路及二路 vs 三路对比

**简历话术方向**：
- "主导实现多路混合检索架构，将 RAG 召回从 Dense+BM25 双路扩展到 Dense+Sparse+Graph 三路，并通过动态 RRF 融合提升复杂知识场景下的召回覆盖率"
- "设计'粗排召回 -> 精排重排'两阶段检索链路，支持 Cross-Encoder / LLM Rerank 可插拔切换，并通过失败回退机制保障线上稳定性"

**可量化角度**：Hit Rate@K、MRR、NDCG、三路相对二路的提升幅度、Rerank 前后 Top-1 命中率、查询延迟

---

## 亮点 2：智能查询路由网关（Spam Gate + Intent Router + Complexity Routing）

**技术要点**：
- 在 Query Engine 前置轻量垃圾拦截与安全网关，过滤无意义输入、恶意试探与纯闲聊流量
- 引入本地意图路由能力，支持业务意图识别并驱动不同检索/应答策略
- 新增本地 Query Complexity Classifier，为"简单问题走小模型、复杂问题走大模型"提供路由信号
- 查询侧过滤条件支持 `collection`、`doc_type`、`doc_intent` 等结构化约束
- Chat 页面只暴露 `快速 / 思考 / Pro` 三档模式，底层模型映射由系统根据基准结果动态管理

**简历话术方向**：
- "设计了面向 RAG 的智能查询网关，在检索前完成垃圾拦截、意图识别与复杂度分类，降低无效流量消耗并支撑模型路由决策"
- "将用户侧交互抽象为'快速/思考/Pro'三档模式，屏蔽底层模型复杂度，并基于离线评测结果动态调整后端路由"

**可量化角度**：垃圾流量拦截率、不同意图的分类准确率、简单/复杂问题路由占比、平均推理成本与延迟变化

---

## 亮点 3：全链路可插拔架构（抽象接口 + 工厂模式 + 配置驱动）

**技术要点**：
- 为 LLM、Embedding、Vision LLM、Splitter、Vector Store、Reranker、Evaluator 等核心组件定义统一抽象接口
- 采用工厂模式与 `settings.yaml` 配置驱动，实现"改配置不改代码"的 Provider 切换
- LLM 已支持 Azure OpenAI / OpenAI / Ollama / DeepSeek，多种部署形态可无缝切换
- Embedding 支持 Azure / OpenAI / Ollama，向量存储当前落地 Chroma，并预留后续扩展能力
- Prompt 模板外置到 `config/prompts/`，便于独立迭代与 A/B 调整

**简历话术方向**：
- "搭建了面向 RAG 的全链路可插拔架构，通过抽象接口 + 工厂模式 + 配置驱动，实现模型、检索、切分、评估等核心模块的零代码切换"
- "统一封装云端与本地 LLM/Embedding Provider，兼顾企业合规、成本优化与私有化部署诉求"

**可量化角度**：支持的 Provider 数量、可插拔组件数、配置切换耗时、A/B 实验迭代次数

---

## 亮点 4：智能数据摄取流水线（Load -> Split -> Transform -> Embed -> Upsert）

**技术要点**：
- 自研五阶段 Ingestion Pipeline，支持 CLI、Dashboard、离线批处理等多入口复用
- Loader 当前已支持 PDF 与 Markdown，PDF 通过 MarkItDown 转 canonical Markdown，Markdown 原生接入
- 使用 `RecursiveCharacterTextSplitter` 做结构感知切分，尽量保留标题、段落、列表、代码块语义边界
- Transform 阶段集成 Chunk Refiner、Metadata Enricher、Image Captioner 等增强步骤
- 通过文件 SHA256 与内容哈希实现增量摄取、重复跳过与幂等 Upsert
- Pipeline 支持 `on_progress` 回调，可直接驱动前端实时进度展示

**简历话术方向**：
- "实现统一可观测的数据摄取流水线，覆盖文档解析、语义切分、LLM 增强、双路编码与幂等上载，支撑本地知识库的持续更新"
- "基于文件哈希与内容哈希设计增量摄取机制，避免重复解析和重复向量化，显著降低离线处理成本"

**可量化角度**：处理文档数、Chunk 产出量、重复文件跳过率、摄取吞吐、单文档处理耗时

---

## 亮点 5：文档级意图标注与按意图分库视图

**技术要点**：
- 在摄取阶段复用本地分类模型，对整篇文档生成统一 `doc_intent` 标签
- 将意图标签写入所有 Chunk 的元数据，支持查询阶段按业务意图做精准过滤
- 自动构建 `data/documents_by_intent/{intent}/{collection}/` 物理视图，便于人工巡检和运营协作
- 形成"摄取侧意图标注 + 查询侧意图过滤"闭环，增强业务场景下的检索可控性

**简历话术方向**：
- "将文档级意图识别前移到数据摄取阶段，为每篇文档打上统一业务标签，并在查询阶段通过元数据过滤实现意图级精准召回"
- "设计按意图划分的原始文档视图，让算法标签结果可被产品、运营和开发共同巡检，提高 RAG 数据治理能力"

**可量化角度**：意图类别数、带标签文档占比、意图过滤命中率、标注一致性

---

## 亮点 6：MCP 标准集成与 Agent 可调用能力

**技术要点**：
- 遵循 MCP 标准，以 JSON-RPC 2.0 + Stdio Transport 方式实现本地知识检索服务
- 对外暴露 `query_knowledge_hub`、`list_collections`、`get_document_summary` 三个标准工具
- 支持 GitHub Copilot、Claude Desktop 等 MCP Client 即插即用
- 返回内容支持文本与图像两类内容，并附带结构化 Citation 信息
- 采用本地 stdio 通信，无需额外服务端口与网络暴露，适合私有知识库和桌面端集成

**简历话术方向**：
- "基于 MCP 标准实现知识检索 Server，使 GitHub Copilot、Claude Desktop 等 AI 助手可直接调用私有知识库"
- "设计带引用的结构化工具返回，兼容文本与图像内容，提升 AI 输出的可溯源性与可信度"

**可量化角度**：MCP 工具数、客户端适配数、工具调用成功率、端到端响应延迟

---

## 亮点 7：多模态 Image-to-Text 检索方案

**技术要点**：
- 采用 Image-to-Text 思路处理文档图片，避免引入额外的多模态向量库复杂度
- Loader 阶段提取图片引用，Transform 阶段使用 Vision LLM 生成结构化 Caption
- Caption 被注入到 Chunk 正文/Metadata 并进入统一检索空间，实现"搜文出图"
- 检索命中图片后，可读取本地图片并以 Base64 形式返回给 MCP Client 或 UI
- 该方案兼容现有文本检索链路，适合流程图、截图、图表等文档型图片场景

**简历话术方向**：
- "设计并落地 Image-to-Text 多模态检索方案，通过 Vision LLM 将图片语义映射到文本检索空间，在不重构主架构的前提下实现图文统一检索"
- "将图片描述与文本 Chunk 统一索引，使用户可通过自然语言检索文档中的图表、截图与流程图"

**可量化角度**：处理图片数、Caption 覆盖率、图片相关查询命中率、图文混合查询占比

---

## 亮点 8：八页面可观测性平台（Dashboard + Chat + LLM Arena）

**技术要点**：
- 基于 Streamlit 构建 8 页面管理平台：Overview、Chat、Data Browser、Ingestion Manager、Ingestion Traces、Query Traces、Evaluation Panel、LLM Arena
- 构建 Ingestion Trace + Query Trace 双链路白盒追踪，记录阶段耗时、候选数量、排序变化等中间状态
- Chat 页面在 Dashboard 内完成完整 RAG 闭环，支持会话持久化、引用折叠展示、运行时间展示与中英文引用文案自适应
- LLM Arena 支持单次交互测试与批量压测，提供动态排行榜、历史记录、进度恢复、质量/延迟/成本对比
- 页面渲染依赖 Trace 中的 `method` / `provider` 字段，更换底层组件后 UI 可自动适配

**简历话术方向**：
- "从 0 到 1 搭建 RAG 可观测性平台，以 8 页面 Dashboard 打通数据管理、链路追踪、在线对话、评估分析与模型压测"
- "通过白盒 Trace + 可视化分析，把 RAG 系统从黑盒问答升级为可定位、可解释、可调优的工程系统"

**可量化角度**：Dashboard 页面数、Trace 覆盖阶段数、历史运行记录数、问题定位耗时缩短比例

---

## 亮点 9：自动化评估与 Benchmark 闭环

**技术要点**：
- 集成 Ragas 与自定义指标，支持 Faithfulness、Answer Relevancy、Context Precision、Hit Rate、MRR 等评估维度
- Evaluation Panel 支持历史记录查看、历史重算、细节展示与结果对比图
- LLM Arena 支持批量策略评测、历史结果持久化、断点续跑与排行榜展示
- 最近版本统一采用质量/延迟/成本三维综合评分口径，便于不同策略横向比较
- 支持二路检索、三路检索及不同模型路由策略的量化对比，形成策略迭代闭环

**简历话术方向**：
- "搭建自动化评估与 Benchmark 闭环，将 Ragas 指标、自定义检索指标与成本/延迟分析统一到同一评测体系中，避免凭经验调参"
- "为模型路由和检索策略提供历史可追溯的排行榜与对比分析，使每次架构调整都有量化依据"

**可量化角度**：评估指标数、测试集规模、历史运行次数、不同策略的综合评分差异

---

## 亮点 10：Local-First 工程化与文档生命周期管理

**技术要点**：
- 采用 SQLite 持久化 `ingestion_history`、`image_index` 等关键索引信息，保持零外部数据库依赖
- DocumentManager 独立协调文档列表、详情查看、跨存储删除等生命周期操作
- 删除流程覆盖 Chroma、BM25、Image Storage、File Integrity 四类存储，保证状态一致性
- Chat 会话历史本地持久化，LLM Arena 压测进度与历史结果可恢复
- 保持本地优先、低部署成本的工程风格，便于个人项目展示、教学演示与快速落地

**简历话术方向**：
- "采用 Local-First 工程设计，通过 SQLite + 本地文件系统实现摄取历史、图片索引、聊天记录与评测历史持久化，做到低依赖、易部署、易演示"
- "实现跨多存储的一致性文档生命周期管理，支持文档浏览、详情查看、同步删除与重新摄取"

**可量化角度**：外部基础设施依赖数、跨存储协调操作数、删除成功率、环境搭建时长
