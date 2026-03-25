# Research Agent (Apple 10-K 财报分析助手)

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

我们可以使用项目提供的 `Makefile` 完整启动包含 Milvus、测试交互界面在内的所有服务。前提是已经安装了 Docker, [uv](https://github.com/astral-sh/uv), Node.js (推荐 bun/pnpm), 和 `make` 工具。

### 1. 环境配置

复制环境变量配置模板，并根据你使用的模型供应商 API 填入对应的 API Key（Chat LLM, Embedding, Reranker）：
```bash
cp .env.example .env
```
*(注意：请确保该文件内的相关配置正确映射到你在代码中设定的模型端点。`APP_MILVUS_URI` 在生产环境下将指向 Docker 提供的端口: http://localhost:19530 )*

### 2. 构建镜像与启动基础架构

在启动 Docker 前，我们需要先将 LangGraph 后端节点构建成 Docker 镜像（使用底层的 `langgraph cli`，这会在 `make build` 过程中经由 uv 执行）：

```bash
make sync        # 安装环境与依赖
make build       # 自动调用全局或虚拟环境中的 langgraph 构建生产镜像
```

接着，使用以下命令启动生产环境的基础栈（包含 Milvus Standalone、MinIO、etcd 等相关检索服务和打好的环境镜像）：



```bash
make up-prod
```
*可以通过 `docker ps` 或者 Attu 界面 (通常 http://localhost:8888) 确认服务是否启动完成。*

### 3. 数据向量化与入库（Ingestion）

等 Docker 环境全面就绪后，需要将自带的苹果财报文件 (`aapl_10k.json`) 向量化并注入到 Milvus 数据库中：

```bash
make sync        # 使用 uv 安装 Python 运行时环境依赖
make ingest-prod # 读取并灌入苹果财报向量数据并处理关键字
```
此步骤会比较耗时，请耐心等待数据切片、向量计算及插入完成。

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
  make run         # 本地启动 LangGraph GUI / 服务集成验证节点 (:2024)
  ```

现在你可以打开浏览器访问前端应用或 LangGraph Studio，与 AI 研究助手深入探讨并查询 Apple (AAPL) 财报的商业规划、风险提示或具体财务表现。

## 常用开发命令字典

| 命令 | 描述 |
|---|---|
| `make up-prod` / `down-prod` | 启动 / 停止完整的生产级别 Docker (含独立版 Milvus 等组件) |
| `make ingest-prod`| 将本地苹果 JSON 财报向量化并存储到独立版 Milvus (基于 `prod` 环境的接口) |
| `make milvus-up` / `milvus-down` | 仅启动 / 停止独立 Milvus 向量库 |
| `make dev` | 整合启动本地测试环境流 (基于 `Procfile.dev`) |
| `make test` | 快速执行 Pytest 单元与集成测试 |
| `make lint` / `format`| 代码风格与规范检查和格式化（Ruff） |

---

*如需查阅更多内部技术设计实现细节（如 Prompt 设计、Tool 使用流水线、向量搜索策略等），请查看仓库内的 [`CLAUDE.md`](./CLAUDE.md) 及其他 `docs/` 内容。*
