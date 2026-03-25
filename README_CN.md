# Research Agent (Apple 10-K 财报分析助手)

English | 简体中文

本项目是一个基于 LangGraph 构建的 AI 智能研究助手，专门用于分析 Apple Inc. 的 SEC 10-K 财务报表。
它使用混合检索（关键词搜索 + 语义重排）与 Milvus 向量数据库相配合，为您提供精确、深度的财务信息分析与问答体验。

## Agent 与检索架构设计

本项目基于 LangGraph 设计了专门针对复杂长文本（如财报）的多步检索与分析反馈链路。主要包含以下三个核心环节：

### 1. 数据预处理与索引建立
- **文本分块**: 将结构化的苹果 10-K 财报 JSON 数据提取并使用 `RecursiveCharacterTextSplitter` 进行切片（默认 1500 字符大小，150 字符重叠），保证上下文语义连贯。
- **向量化嵌入 (Embedding)**: 调用基于 OpenAI 兼容 API 的 Qwen3-Embedding-8B 模型，对文本块进行批量向量化提取。
- **存储与元数据**: 向量被存入 Milvus 向量数据库（支持 Lite 或 Standalone）。存储的同时会保留丰富的元数据属性（如 `symbol` 股票代码，`fiscal_year` 财年，`section_title` 章节标题和 `form_type` 报表类型等），为后续的精准结构化条件过滤打下基础。

### 2. 文本检索与信息增强 (RAG Pipeline)
- **多维度并发检索**: 基于用户的问题拆分为不同的评估视角与关键词组，通过 `ThreadPoolExecutor` 对各个关键词组开启并发执行检索。这种并行设计大幅提升了复杂财报上下文的召回效率。
- **内容去重与清洗**: 汇总各线程的检索结果，根据 `page_content` 开展严格去重过滤，防止重复信息干扰大模型推理。
- **语义重排 (Reranking)**: 对召回后的初步材料，按不同评估视角（Perspective）分发给 Qwen3-Reranker-8B 模型进行深度语义重排得分排序，筛选出高匹配度的结构化 Markdown 内容，完成多维度信息增强。

### 3. 生成财报回答与分析报告
- **核心推理与输出**: Agent 的决策大脑（Graph）在获得由重排机制高度浓缩清洗后的背景上下文后，结合强大的对话大模型（如 MiniMax-M2.7），根据用户提出的问题及财报片段，进行综合分析与图表化总结。最终不仅能应对普通的 Q&A 问答，还能自动撰写逻辑严密的阶段性简要财务分析报告。

## 架构概览

- **Agent 核心**: 使用 LangGraph 构建的智能体 (`agent.graph:graph`)，支持多步检索与思考。
- **检索增强 (RAG)**:
  - 向量数据库：Milvus (支持独立版或 Lite 版)，采用混合检索策略。
  - 嵌入与重排模型：默认使用与 OpenAI 兼容的 API (如基于 Qwen3 的向量/重排服务)。
- **后端服务**: FastAPI 提供的上传交互服务 (`make api`)。
- **前端页面**: 基于 Next.js 的精美交互界面 (`make ui`)。

## 快速体验（生产级全栈环境）

### 1. 环境配置

复制环境变量配置模板，并根据你使用的模型供应商 API 填入对应的 API Key：
```bash
cp .env.example .env
```

### 2. 构建镜像与启动基础架构

在启动 Docker 前，我们需要先将 LangGraph 后端节点构建成 Docker 镜像：

```bash
make sync        # 安装环境与依赖
make build       # 自动调用 langgraph 构建生产镜像
```

接着，启动包含 Milvus 和打好的环境镜像的基础栈：

```bash
make up-prod
```

### 3. 数据向量化与入库（Ingestion）

等 Docker 环境全面就绪后，将自带的苹果财报文件 (`aapl_10k.json`) 注入到 Milvus 数据库中：

```bash
make ingest-prod # 读取并灌入苹果财报向量数据并处理关键字
```

### 4. 运行交互与调试

**启动本地后端与前端界面：**
- 运行前端 UI：
  ```bash
  make ui-install  # 初次拉取 Next.js 项目的依赖
  make ui          # 启动前端 UI 服务 (:3000)
  ```
- 运行交互API/智能体节点：
  ```bash
  make api         # 启动后台 FastAPI 服务 (:8080)
  make run         # 本地启动 LangGraph GUI (:2024)
  ```

## 常用开发命令字典

| 命令 | 描述 |
|---|---|
| `make up-prod` | 启动完整的生产级别 Docker (含独立版 Milvus 等组件) |
| `make ingest-prod`| 将本地苹果 JSON 财报向量化并存储到独立版 Milvus |
| `make dev` | 整合启动本地测试环境流 (基于 Procfile.dev) |

---
*更多细节请查看 [`CLAUDE.md`](./CLAUDE.md)。*
