# Research Agent (Apple 10-K 财报分析助手)

English | 简体中文

本项目是一个基于 LangGraph 构建的 AI 智能研究助手，专门用于分析 Apple Inc. 的 SEC 10-K 财务报表。
它使用混合检索（关键词搜索 + 语义重排）与 Milvus 向量数据库相配合，为您提供精确、深度的财务信息分析与问答体验。

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
