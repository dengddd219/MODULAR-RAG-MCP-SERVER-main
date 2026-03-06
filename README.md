# Modular RAG MCP Server

> 一个可插拔、可观测的模块化 RAG (检索增强生成) 服务框架，通过 MCP (Model Context Protocol) 协议对外暴露工具接口，支持 Copilot / Claude 等 AI 助手直接调用。

---

## 快速开始

### 1. 安装依赖

```bash
# 克隆仓库后进入项目目录
cd MODULAR-RAG-MCP-SERVER-main

# 创建虚拟环境（推荐）
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate
python scripts/start_dashboard.py
# 安装项目依赖
pip install -e .

# 开发/测试额外依赖（可选）
pip install -e ".[dev]"
```

### 2. 配置 API Key

编辑 `config/settings.yaml`，至少配置 LLM 与 Embedding 的 API 密钥与端点（见下方 [配置说明](#配置说明)）。  
也可通过环境变量设置，例如：

- **Azure OpenAI**：`AZURE_OPENAI_API_KEY`、`AZURE_OPENAI_ENDPOINT`（若在 settings 中未填 endpoint）
- **OpenAI**：`OPENAI_API_KEY`

### 3. 运行首次摄取

```bash
# 将示例文档摄入默认 collection
python scripts/ingest.py --path tests/fixtures/sample_documents/ --collection default

# 指定单文件
python scripts/ingest.py --path path/to/your.pdf --collection my_docs
```

### 4. 查询与 MCP

```bash
# 命令行查询
python scripts/query.py --query "你的问题" --verbose

# MCP Server 以 stdio 方式运行，供 Copilot / Claude 等连接（见下方 MCP 配置示例）
python -m src.mcp_server.server
```

---

## 配置说明

主配置文件为 **`config/settings.yaml`**。主要字段含义如下。

| 区块 | 字段 | 含义 |
|------|------|------|
| **llm** | `provider` | 取值：`openai` / `azure` / `ollama` / `deepseek` |
| | `model`, `deployment_name` | 模型名 / Azure 部署名 |
| | `azure_endpoint`, `api_key` | Azure/OpenAI 端点与密钥（可改用环境变量） |
| | `temperature`, `max_tokens` | 生成参数 |
| **embedding** | `provider` | `openai` / `azure` / `ollama` |
| | `model`, `dimensions` | 模型名与向量维度（如 1536） |
| | `azure_endpoint`, `deployment_name`, `api_key` | Azure Embedding 配置 |
| **vision_llm** | `enabled` | 是否启用多模态（如图片描述） |
| | `provider`, `model` | 同 LLM，用于图片理解 |
| **vector_store** | `provider` | 当前支持 `chroma` |
| | `persist_directory` | Chroma 持久化目录（如 `./data/db/chroma`） |
| | `collection_name` | 默认 collection 名 |
| **retrieval** | `dense_top_k`, `sparse_top_k`, `fusion_top_k` | 检索与融合数量 |
| | `rrf_k` | RRF 融合常数 |
| **rerank** | `enabled`, `provider` | 是否启用、类型（`none` / `cross_encoder` / `llm`） |
| | `model`, `top_k` | 模型与截断条数 |
| **observability** | `log_level` | 日志级别：DEBUG / INFO / WARNING / ERROR |
| | `trace_enabled`, `trace_file` | 是否写追踪、追踪文件路径（如 `./logs/traces.jsonl`） |
| **ingestion** | `chunk_size`, `chunk_overlap` | 分块大小与重叠 |
| | `splitter` | 分块方式：`recursive` / `semantic` / `fixed_length` |
| | `chunk_refiner.use_llm`, `metadata_enricher.use_llm` | 是否用 LLM 做块精炼/元数据增强 |

未填写的 `api_key` / `azure_endpoint` 会从环境变量读取（如 `AZURE_OPENAI_API_KEY`）。

---

## MCP 配置示例

MCP Server 使用 **stdio** 传输，需在客户端配置中指向本项目的 Python 与入口脚本。

### GitHub Copilot（mcp.json）

在 Copilot 使用的 `mcp.json` 中增加：

```json
{
  "mcpServers": {
    "modular-rag": {
      "command": "python",
      "args": [
        "-m",
        "src.mcp_server.server"
      ],
      "cwd": "C:/path/to/MODULAR-RAG-MCP-SERVER-main",
      "env": {}
    }
  }
}
```

将 `cwd` 改为你本机的项目根目录绝对路径。若需使用虚拟环境：

```json
"command": "C:/path/to/MODULAR-RAG-MCP-SERVER-main/.venv/Scripts/python.exe",
"args": ["-m", "src.mcp_server.server"],
"cwd": "C:/path/to/MODULAR-RAG-MCP-SERVER-main",
```

### Claude Desktop（claude_desktop_config.json）

在 Claude Desktop 配置目录下的 `claude_desktop_config.json` 中增加：

```json
{
  "mcpServers": {
    "modular-rag": {
      "command": "python",
      "args": ["-m", "src.mcp_server.server"],
      "cwd": "C:/path/to/MODULAR-RAG-MCP-SERVER-main"
    }
  }
}
```

同样将 `cwd` 改为实际项目路径；若用 venv，可将 `command` 改为 venv 中的 `python` 绝对路径。  
配置完成后重启 Copilot / Claude Desktop，即可在对话中调用 `query_knowledge_hub`、`list_collections`、`get_document_summary` 等工具。

---

## Dashboard 使用指南

基于 Streamlit 的六页管理平台，用于查看配置、浏览数据、管理摄取与查看追踪。

### 启动方式

```bash
# 方式一：脚本启动（推荐使用这个，然后会自动配置环境）
# .\.venv\Scripts\Activate.ps1 
# 配置环境不是base之后再运行
python scripts/start_dashboard.py

# 指定端口
python scripts/start_dashboard.py --port 8502

# 方式二：直接 Streamlit
streamlit run src/observability/dashboard/app.py --server.port 8501
```

浏览器访问 **http://localhost:8501**（或你所设端口）。

### 各页面功能

| 页面 | 功能 |
|------|------|
| **Overview（系统总览）** | 组件配置卡片、数据资产统计（collection 数量等） |
| **Data Browser（数据浏览器）** | 按 collection 查看文档列表、Chunk 内容与元数据、图片预览 |
| **Ingestion Manager（摄取管理）** | 文件上传、触发摄取、进度条、文档删除 |
| **Ingestion Traces（摄取追踪）** | 摄取历史、阶段耗时瀑布图、trace 详情 |
| **Query Traces（查询追踪）** | 查询历史、Dense/Sparse 对比、Rerank 前后结果 |
| **Evaluation Panel（评估面板）** | 运行评估、指标展示、历史趋势 |

截图示例可在运行后对上述各页截图保存，用于内部文档或 PR。

---

## 运行测试

```bash
# 单元测试（推荐先跑）
pytest -q tests/unit/

# 集成测试（可能依赖本地服务/配置）
pytest -q tests/integration/

# E2E 测试（MCP 子进程、Dashboard 冒烟等）
pytest -q tests/e2e/

# 全量测试
pytest -q

# 排除需真实 LLM 的测试
pytest -q -m "not llm"
```

常用单文件示例：

```bash
pytest -q tests/unit/test_smoke_imports.py
pytest -q tests/e2e/test_mcp_client.py
pytest -q tests/e2e/test_dashboard_smoke.py
```

---

## 全链路 E2E 验收（I5）

配置好 API Key 与依赖后，可按以下顺序验证整条链路：

1. **摄取**：`python scripts/ingest.py --path tests/fixtures/sample_documents/ --collection test`
2. **查询**：`python scripts/query.py --query "测试查询" --verbose`
3. **Dashboard**：`python scripts/start_dashboard.py`，在浏览器中确认总览、数据浏览、摄取/查询追踪等页面正常。
4. **评估**：`python scripts/evaluate.py`（需有 golden test set 或相应数据）。

全部通过即表示全链路 E2E 验收完成。

---

## 常见问题

### API Key 配置

- **现象**：摄取或查询报错 “API key not provided”。  
- **处理**：在 `config/settings.yaml` 中填写对应 `llm.api_key` / `embedding.api_key`，或设置环境变量 `AZURE_OPENAI_API_KEY` / `OPENAI_API_KEY`，并确保 endpoint（若用 Azure）正确。

### 依赖安装

- **现象**：`ModuleNotFoundError: No module named 'mcp'` 等。  
- **处理**：在项目根目录执行 `pip install -e .`，确保安装的是当前项目（含 `pyproject.toml` 中的依赖）。开发时建议 `pip install -e ".[dev]"`。  
- Windows 下若编码报错，可设置 `PYTHONIOENCODING=utf-8` 再运行脚本或测试。

### 连接与端点

- **现象**：Azure/OpenAI 请求超时或连接失败。  
- **处理**：检查 `settings.yaml` 中 `azure_endpoint` 是否带 `https://`、是否与 API Key 所在区域一致；本机代理或防火墙是否放行出站 HTTPS。

### MCP 客户端连不上 Server

- **现象**：Copilot / Claude 中看不到工具或调用失败。  
- **处理**：确认 `mcp.json` / `claude_desktop_config.json` 中 `cwd` 为项目根目录绝对路径；`command`/`args` 能在该 `cwd` 下成功执行 `python -m src.mcp_server.server`；重启客户端后再试。

---

## 项目概览与分支说明

- **Ingestion Pipeline**：PDF → 解析 → Chunk → Transform → Embedding → Upsert（支持多模态图片描述）
- **Hybrid Search**：Dense + Sparse (BM25) + RRF Fusion + 可选 Rerank
- **MCP Server**：stdio 暴露 `query_knowledge_hub`、`list_collections`、`get_document_summary`
- **Dashboard**：Streamlit 六页（总览 / 数据浏览 / 摄取管理 / 摄取追踪 / 查询追踪 / 评估面板）
- **Evaluation**：Ragas + Custom 评估与 golden test set 回归

详细架构与排期见 [DEV_SPEC.md](DEV_SPEC.md)。

| 分支 | 用途 |
|------|------|
| **main** | 最新代码，单 commit 完整快照 |
| **dev** | 开发历史，多 commit 记录 |
| **clean-start** | 仅骨架 + DEV_SPEC，进度清零，便于从零实现 |

---

## License

MIT
