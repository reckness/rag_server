# RAG Service API 使用文档

> 服务地址: `http://localhost:8000`  |  Swagger 文档: `http://localhost:8000/docs`

---

## 一、`/pdf/process/{doc_id}` — 处理文档（与 /document 策略一致）

> 与 `/document/process/{doc_id}` 完全一致的处理流程  
> MinIO 下载 → page_index 解析 → ESConverter 写入 ES → DocumentRouter 写入路由 → 更新数据库  
> 前置条件: 文档记录已存在于 PostgreSQL，PDF 已上传到 MinIO

```bash
curl -X POST http://localhost:8000/pdf/process/{doc_id}
```

**处理流程:**
1. 查询数据库获取文档信息（`doc_id`、`source_path`、`kb_id`、`fd_id`）
2. 从 MinIO（`deepsearch` 桶）下载 PDF → 20%
3. page_index 解析生成 JSON → 50%（JSON 同步上传到 MinIO）
4. ESConverter 扁平化 + embedding → 写入 ES 向量索引 → 75%
5. DocumentRouter 生成文档路由 → 写入 `doc_summary_index` → 90%
6. 更新数据库状态为 `ready`、记录 `chunk_num`、`process_duration`、`llm_token` → 100%

**返回:**
```json
{
  "code": 200,
  "data": {
    "success": true,
    "document_id": "your_doc_id",
    "chunk_num": 12,
    "process_duration": 45.67
  }
}
```

**查询处理状态:**
```bash
curl http://localhost:8000/document/status/{doc_id}
```

---

## 二、`/pdf/process_by_path` — 通过本地路径处理（同步）

> 不依赖 MinIO / 数据库，直接从服务器本地路径读取 PDF  
> 处理 → 写入 ES → 直接返回结果

```bash
curl -X POST http://localhost:8000/pdf/process_by_path \
  -H "Content-Type: application/json" \
  -d '{
    "pdf_path": "/home/rag/rag-service/pdf/低空经济__开启立体城市空间新未来_劳莘.pdf",
    "mode": "chunked",
    "if_summary": false,
    "kb_id": "my_kb",
    "query": "低空经济的核心价值是什么？",
    "topk": 3
  }'
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `pdf_path` | string | **必填** | PDF 文件绝对路径 |
| `mode` | string | `chunked` | `simple` = 短文档；`chunked` = 长文档分块 |
| `if_summary` | bool | `true` | 是否生成 LLM 摘要 |
| `if_add_node_text` | bool | `true` | 是否保留原文 |
| `doc_id` | string | 自动生成 | 文档 ID |
| `kb_id` | string | `default_kb` | 知识库 ID |
| `fd_id` | string | `default_fd` | 文件夹 ID |
| `query` | string | 空 | 可选，传入后自动召回 |
| `topk` | int | `5` | 召回数量 |
| `llm_url` | string | `http://10.1.141.33:8001/v1/chat/completions` | LLM API 地址 |
| `llm_model` | string | `Qwen3-8B` | LLM 模型名称 |

**返回:**
```json
{
  "code": 200,
  "data": {
    "json_path": "/home/rag/rag-service/pdf/xxx_chunked_structure.json",
    "md_path": "/home/rag/rag-service/pdf/xxx_chunked.md",
    "es_info": { "chunk_num": 4, "doc_id": "260a158f...", "kb_id": "my_kb" },
    "recall": {
      "query": "低空经济的核心价值是什么？",
      "doc_ids": ["260a158f..."],
      "chunks": [ { "doc_title": "...", "section_path": [...], "original_snippet": "...", "_score": 5.67 } ],
      "context": "【文档】...【章节】...【内容】..."
    }
  }
}
```

> `recall` 字段仅在传入 `query` 时返回。

---

## 三、`/pdf/process_and_recall` — 处理 + 召回（一步到位）

> 与 `process_by_path` 类似，但 **`query` 为必填**，确保一定返回召回结果

```bash
curl -X POST http://localhost:8000/pdf/process_and_recall \
  -H "Content-Type: application/json" \
  -d '{
    "pdf_path": "/home/rag/rag-service/pdf/低空经济__开启立体城市空间新未来_劳莘.pdf",
    "mode": "chunked",
    "if_summary": false,
    "kb_id": "my_kb",
    "query": "低空经济的核心价值是什么？",
    "topk": 3
  }'
```

返回格式与 `process_by_path`（含 `query`）完全一致。

---

## 四、原有文档处理 API

### 4.1 `/document/process/{doc_id}`

```bash
curl -X POST http://localhost:8000/document/process/{doc_id}
```

> 与 `/pdf/process/{doc_id}` 逻辑一致。两个入口都可以使用。

### 4.2 `/document/status/{doc_id}`

```bash
curl http://localhost:8000/document/status/{doc_id}
```

### 4.3 `/universal/process/{doc_id}` — 通用多格式文档处理

```bash
curl -X POST "http://localhost:8000/universal/process/{doc_id}?use_llm_for_pdf_no_toc=true"
```

---

## 五、RAG 搜索 API

> 前缀: `/rag`  
> 适用于已写入 ES 的文档

```bash
curl -X POST http://localhost:8000/rag/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "低空经济的核心价值是什么？",
    "kb_ids": ["my_kb"],
    "topk": 3,
    "use_rerank": true
  }'
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | string | **必填** | 搜索查询 |
| `kb_ids` | list[string] | `null` | 知识库 ID 列表 |
| `fd_ids` | list[string] | `null` | 文件夹 ID 列表 |
| `topk` | int | `5` | 返回数量 |
| `use_rerank` | bool | `true` | 是否用 rerank 重排序（`http://10.1.141.33:8474/rerank`） |

---

## 六、接口对比

| | `/pdf/process/{doc_id}` | `/pdf/process_by_path` | `/pdf/process_and_recall` | `/rag/search` |
|---|---|---|---|---|
| **用途** | 完整 RAG 管线 | 本地路径处理+写入ES | 处理+写入ES+召回 | 仅搜索 |
| **输入** | 数据库 doc_id | PDF 路径 | PDF 路径 + query | query + kb_ids |
| **依赖** | MinIO + PG + ES + LLM | LLM + ES | LLM + ES | ES |
| **写入 ES** | 自动 | 自动 | 自动 | 不写入 |
| **更新数据库** | 是 | 否 | 否 | 否 |
| **返回召回** | 否 | 可选（传 query） | 是 | 是 |
| **Rerank** | 不涉及 | 不走 | 不走 | 可选 |

---

## 七、快速测试

```bash
# 1. 启动服务
cd /home/rag/rag-service && conda run -n rag python main.py

# 2. 通过 doc_id 处理（需要数据库+MinIO中有记录）
curl -X POST http://localhost:8000/pdf/process/{doc_id}

# 3. 通过本地路径处理 + 写入 ES + 召回
curl -X POST http://localhost:8000/pdf/process_by_path \
  -H "Content-Type: application/json" \
  -d '{
    "pdf_path": "/home/rag/rag-service/pdf/低空经济__开启立体城市空间新未来_劳莘.pdf",
    "mode": "chunked",
    "if_summary": false,
    "kb_id": "test_kb_001",
    "query": "低空经济的核心价值是什么？",
    "topk": 3
  }'

# 4. 搜索已索引的文档（走 rerank）
curl -X POST http://localhost:8000/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query": "低空经济的核心价值", "kb_ids": ["test_kb_001"], "topk": 3, "use_rerank": true}'

# 5. 搜索已索引的文档（不走 rerank）
curl -X POST http://localhost:8000/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query": "低空经济的核心价值", "kb_ids": ["test_kb_001"], "topk": 3, "use_rerank": false}'
```
