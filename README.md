# RAG Service — 知识库文档解析与检索增强生成服务

## 项目简介

RAG Service 是一个基于 **FastAPI** 的后端服务，核心能力是将多格式文档（PDF、Word、TXT、PPT、Excel、Markdown）解析为结构化数据，写入 Elasticsearch 向量索引，并提供 **混合检索 + Rerank** 的 RAG 搜索接口，为下游 LLM 提供高质量上下文。

---

## 技术栈

| 层级 | 技术 |
|------|------|
| Web 框架 | FastAPI + Uvicorn |
| 数据库 | PostgreSQL (SQLAlchemy ORM) |
| 搜索引擎 | Elasticsearch 8.x（向量检索 + BM25 混合查询） |
| 对象存储 | MinIO |
| Embedding | qwen-embedding（1024 维，通过 HTTP API 调用） |
| Rerank | bge-reranker-base（CrossEncoder，本地加载） |
| PDF 解析 | PyMuPDF + LLM 辅助目录检测与切分（Qwen3-8B） |
| 容器化 | Docker + Docker Compose |

---

## 项目结构

```
rag-service/
├── main.py                          # FastAPI 应用入口，加载 Rerank 模型、注册路由
├── requirements.txt                 # Python 依赖
├── Dockerfile                       # 生产镜像（基于基础镜像）
├── Dockerfile.base                  # 基础镜像（安装系统依赖 + pip 包）
├── docker-compose.yml               # 编排：基础镜像构建 + 主服务
│
├── app/                             # FastAPI 应用层
│   ├── api/                         # API 路由
│   │   ├── document_processing_api.py   # /document — 文档处理 & 状态查询
│   │   ├── rag_api.py                   # /rag — RAG 搜索
│   │   └── universal_rag_api.py         # /universal — 通用多格式文档处理
│   ├── core/                        # 核心基础设施
│   │   ├── database.py              # SQLAlchemy 引擎 & Session
│   │   └── exceptions.py           # 自定义异常类
│   ├── models/                      # 数据模型（SQLAlchemy ORM）
│   ├── repository/                  # 数据访问层
│   │   ├── document_repository.py   # 文档 CRUD
│   │   ├── file_repository.py       # 文件 CRUD
│   │   └── folder_repository.py     # 文件夹 CRUD（含递归子文件夹查询）
│   ├── services/                    # 业务逻辑层
│   │   ├── document_processing_service.py  # PDF 文档处理全流程
│   │   ├── universal_rag_service.py        # 通用多格式文档处理
│   │   ├── rag_service.py                  # RAG 搜索核心逻辑
│   │   ├── es_service.py                   # ES 检索（文档路由 + Chunk 混合召回）
│   │   ├── embedding.py                    # Embedding 获取（带 LRU 缓存）
│   │   ├── rerank.py                       # CrossEncoder Rerank
│   │   ├── minio_service.py                # MinIO 文件上传/下载
│   │   ├── folder_service.py               # 文件夹管理
│   │   └── parsers/                        # 多格式文档解析器
│   │       ├── base_parser.py          # 解析器基类（定义统一输出格式）
│   │       ├── word_parser.py          # Word (.docx/.doc) — 按标题样式切分
│   │       ├── txt_parser.py           # TXT — 按字符数分块
│   │       ├── ppt_parser.py           # PPT — 每张幻灯片一个节点
│   │       ├── excel_parser.py         # Excel — 每个 Sheet 一个节点
│   │       ├── md_parser.py            # Markdown — 按标题层级切分
│   │       └── pdf_simple_parser.py    # PDF 简单分块（无 LLM）
│   └── utils/                       # 工具函数
│       ├── response.py              # 统一 API 响应格式
│       └── uuid_generator.py        # UUID 生成
│
├── common/                          # 公共模块
│   ├── config.py                    # 全局配置（ES / PG / MinIO / Redis / Embedding）
│   ├── decorator.py                 # 装饰器
│   ├── doc_store/                   # ES 连接管理
│   │   ├── es_conn_base.py         # ES 连接基类
│   │   ├── es_conn_pool.py         # ES 连接池（全局单例 ES_CONN）
│   │   └── doc_store_base.py       # 文档存储基类
│   └── nlp/
│       └── embedding_client.py     # Embedding HTTP 客户端（qwen-embedding）
│
├── rag/                             # RAG 核心处理模块
│   ├── config.yaml                  # page_index LLM 参数配置
│   ├── page_index.py                # PDF PageIndex 切分（LLM 辅助，核心算法）
│   ├── json_to_es_converter_with_embedding.py  # PageIndex JSON → ES 扁平节点 + 向量写入
│   ├── build_router_es.py           # 文档路由生成（doc_summary_index）
│   ├── search_summary.py            # 文档路由搜索脚本
│   ├── run_router.py                # 路由生成运行脚本
│   └── utils.py                     # 工具函数（token 提取、配置加载等）
│
├── deepdoc/                         # DeepDoc 解析器
│   └── MiniDeepDocParser.py         # PDF 深度解析（OCR + 版面分析）
│
├── pageindex/                       # PageIndex 独立模块
│   ├── config.yaml                  # 独立配置
│   ├── page_index.py                # PageIndex 主逻辑
│   ├── page_index_md.py             # Markdown 输出版本
│   └── utils.py                     # 工具函数
│
└── pdf/                             # 测试用 PDF 文件及输出
```

---

## 核心流程

### 1. 文档处理流程

```
上传文档 → MinIO 存储 → 调用处理接口 → 解析 → ES 索引 → 状态更新
```

详细步骤：

1. **下载文件**：从 MinIO 的 `deepsearch` 桶下载源文件到临时目录
2. **解析文档**：根据文件扩展名分发到对应解析器
   - **PDF**：检测是否有目录（TOC）
     - 有目录 → `page_index()` LLM 精准切分（Qwen3-8B）
     - 无目录 + `use_llm_for_pdf_no_toc=True` → LLM 生成目录后切分
     - 无目录 + `use_llm_for_pdf_no_toc=False` → `PdfSimpleParser` 按字符快速分块
   - **Word**：按标题样式（Heading 1-6）切分章节
   - **TXT**：自动检测编码，按固定字符数分块
   - **PPT**：每张幻灯片作为一个节点
   - **Excel**：每个 Sheet 作为一个节点
   - **Markdown**：按标题层级（# ~ ######）切分
3. **向量索引**：`ESConverter` 将解析结果扁平化，生成 Embedding 写入 `page_index` 索引
4. **文档路由**：`DocumentRouter` 生成文档摘要路由写入 `doc_summary_index` 索引
5. **状态更新**：更新 PostgreSQL 中的文档状态为 `ready`

### 2. RAG 搜索流程

```
用户查询 → Embedding → 文档路由粗召回 → Chunk 混合检索 → 分数归一化 → Rerank → 上下文构建
```

详细步骤：

1. **Query Embedding**：将用户查询通过 `qwen-embedding` 转为 1024 维向量
2. **文档路由粗召回**：在 `doc_summary_index` 中通过 KNN 向量检索找到相关文档（`retrieve_docs`）
3. **Chunk 混合检索**：在 `page_index` 索引中对粗召回文档进行 **KNN 向量检索 + BM25 关键词匹配** 混合查询（`retrieve_chunks`）
4. **分数归一化**：Min-Max 归一化，过滤分数 < 0.6 的低质量 Chunk
5. **Rerank**（可选）：使用 `bge-reranker-base` CrossEncoder 对 Chunk 精排，过滤分数 < 0.6
6. **上下文构建**：取 Top-K Chunk，按 `【文档】【章节】【内容】` 格式拼接上下文返回

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/document/process/{doc_id}` | 处理文档（PDF 专用，非 PDF 自动委托给通用接口） |
| `GET` | `/document/status/{doc_id}` | 查询文档处理状态 |
| `POST` | `/rag/search` | RAG 搜索（需传 query, kb_ids, fd_ids 等） |
| `POST` | `/universal/process/{doc_id}` | 通用多格式文档处理（支持 PDF/Word/TXT/PPT/Excel/MD） |
| `GET` | `/universal/supported-formats` | 查询支持的文件格式列表 |
| `GET` | `/` | 健康检查 |

### RAG 搜索请求示例

```json
{
  "query": "低空经济发展趋势",
  "kb_ids": ["kb_id_1"],
  "fd_ids": ["fd_id_1"],
  "topk": 5,
  "use_rerank": true
}
```

### RAG 搜索响应示例

```json
{
  "code": 200,
  "data": {
    "query": "低空经济发展趋势",
    "doc_ids": ["doc_1", "doc_2"],
    "chunks": [
      {
        "doc_title": "低空经济研究报告",
        "section_path": ["第一章", "1.1 发展背景"],
        "original_snippet": "...",
        "_score": 0.92
      }
    ],
    "context": "【文档】低空经济研究报告\n【章节】第一章 > 1.1 发展背景\n【内容】..."
  }
}
```

---

## Elasticsearch 索引设计

| 索引名 | 用途 | 核心字段 |
|--------|------|----------|
| `page_index` | 文档 Chunk 向量索引 | `doc_id`, `kb_id`, `fd_id`, `embedding_text`, `embedding`(1024d), `original_snippet`, `section_path`, `page_num_int` |
| `doc_summary_index` | 文档路由索引（粗召回） | `doc_id`, `kb_id`, `folder`, `title`, `summary`, `routing_text`, `embedding`(1024d) |

---

## 外部依赖服务

| 服务 | 地址 | 用途 |
|------|------|------|
| PostgreSQL | 10.1.140.215:5435 | 文档元数据存储 |
| Elasticsearch | 10.1.140.215:12300 | 向量索引 + 全文检索 |
| MinIO | 10.1.140.215:19300 | 文件对象存储 |
| Redis | 10.1.140.215:16380 | 缓存（预留） |
| Embedding API | 10.1.141.33:8020 | qwen-embedding 向量化 |
| DeepSeek API | api.deepseek.com | LLM 调用（预留） |

> 配置集中在 `common/config.py`，生产环境建议迁移到环境变量。

---

## 部署方式

### Docker Compose（推荐）

```bash
# 1. 构建基础镜像（包含系统依赖 + pip 包，只需构建一次）
docker-compose build doc_parser_service_base

# 2. 构建并启动主服务
docker-compose up -d doc_parser_service
```

服务暴露在 `http://localhost:8000`。

### 本地开发

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python main.py
# 或
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 关键设计说明

- **两级检索**：先通过文档路由索引（`doc_summary_index`）粗召回相关文档，再在 Chunk 索引（`page_index`）中精确检索，减少无关文档的噪音干扰
- **混合检索**：Chunk 级别同时使用 KNN 向量相似度 + BM25 关键词匹配，兼顾语义匹配和精确匹配
- **分数归一化 + 阈值过滤**：混合检索的两种分数量纲不同，通过 Min-Max 归一化统一后，用 0.6 阈值过滤低质量结果
- **Rerank 可选**：支持 CrossEncoder 精排，模型加载失败时自动降级为按分数排序
- **PDF 智能切分**：自动检测 PDF 目录（TOC），有目录走 LLM 精准切分，无目录可选用 LLM 生成目录或快速字符分块，LLM 失败时自动降级
- **Embedding 缓存**：LRU 缓存（10000 条）避免重复向量化调用
- **统一解析器接口**：所有格式解析器继承 `BaseDocumentParser`，输出统一的 `page_index` JSON 结构，后续流程一致
