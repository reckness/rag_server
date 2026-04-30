# RAG API 接口文档

## 统一响应格式

除 `text/event-stream` 类型接口外，普通接口统一返回：

```json
{
  "code": 200,
  "message": "success",
  "data": {}
}
```

错误示例：

```json
{
  "code": 404,
  "message": "文档不存在",
  "data": null
}
```

---

## 1. RAG 检索接口

### 基本信息

- **接口路径**：`POST /rag/search`
- **Content-Type**：`application/json`
- **功能**：根据 query 在指定知识库/文件夹范围内进行三级 RAG 检索，返回召回的 chunk 与拼接后的 context。

### 请求参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `query` | `string` | 是 | 无 | 用户检索问题/关键词 |
| `kb_ids` | `string[]` 或 `null` | 否 | `null` | 限定知识库 ID 列表。建议必传，否则部分环境可能返回 500 或召回为空 |
| `fd_ids` | `string[]` 或 `null` | 否 | `null` | 限定文件夹 ID 列表。传入后会包含其子文件夹 |
| `topk` | `integer` 或 `null` | 否 | `5` | 最终返回 chunk 数量。小于等于 0 时按 5 处理 |
| `use_rerank` | `boolean` 或 `null` | 否 | `true` | 是否启用 rerank。开启后 `_score` 为归一化后的 rerank 分数 |

### 请求示例

```json
{
  "query": "人工智能 大模型 开源",
  "kb_ids": ["batch_test_kb_001"],
  "fd_ids": ["0"],
  "topk": 5,
  "use_rerank": true
}
```

### 成功响应示例

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "query": "人工智能 大模型 开源",
    "doc_ids": [
      "42c0c93e611d42a280358cd91dbd6177",
      "945344ebe13c49498cb175c6087ae62e"
    ],
    "chunks": [
      {
        "chunk_id": "（二）人工智能应用加速落地 - 段落2",
        "doc_id": "42c0c93e611d42a280358cd91dbd6177",
        "kb_id": "batch_test_kb_001",
        "chapter_id": "0001",
        "fd_id": "0",
        "doc_title": "第56次中国互联网络发展状况统计报告（2025年）-中国互联网络信息中心.pdf",
        "chapter_title": "前言",
        "section_title": "（二）人工智能应用加速落地",
        "page_num_int": [],
        "chunk_text": "一是技术创新不断加速。世界知识产权组织报告显示，我国已成为全球人工智能专利最大拥有国，占比达 60%...",
        "rerank_score": 6.293,
        "_score": 1.0
      }
    ],
    "context": "【文档】第56次中国互联网络发展状况统计报告（2025年）-中国互联网络信息中心.pdf\n【章节】前言 > （二）人工智能应用加速落地\n【内容】一是技术创新不断加速..."
  }
}
```

### 空结果响应示例

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "chunks": [],
    "context": ""
  }
}
```

### 返回字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `data.query` | `string` | 原始检索 query |
| `data.doc_ids` | `string[]` | 实际命中 chunk 所属的文档 ID 去重列表 |
| `data.chunks` | `object[]` | 召回 chunk 列表 |
| `data.context` | `string` | 根据 TopK chunk 拼接出的上下文 |
| `chunk_id` | `string` | chunk 标识，通常来自 JSON 节点标题 |
| `doc_id` | `string` | 文档 ID |
| `kb_id` | `string` | 知识库 ID |
| `chapter_id` | `string` | 章节节点 ID |
| `fd_id` | `string` | 文件夹 ID |
| `doc_title` | `string` | 文档标题 |
| `chapter_title` | `string` | 章节标题 |
| `section_title` | `string` | 小节标题 |
| `page_num_int` | `integer[]` | 页码列表，可能为空 |
| `chunk_text` | `string` | chunk 正文 |
| `rerank_score` | `number` | rerank 原始分数，仅 `use_rerank=true` 且 rerank 成功时返回 |
| `_score` | `number` | 最终排序分数。未 rerank 时为 RRF 分数；rerank 后为归一化分数 |

---

## 2. 文档批处理接口

### 基本信息

- **接口路径**：`POST /universal/process_bat/{doc_id}`
- **Content-Type**：无请求体
- **Response Content-Type**：`text/event-stream`
- **功能**：处理单个文档，实时返回处理进度。流程包括：下载源文件、必要时转 PDF、PDF 解析、JSON 上传、三级索引写入、更新文档状态。

### 路径参数

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `doc_id` | `string` | 是 | 待处理文档 ID。该文档需要已存在于数据库，且 `source_path` 不为空 |

### 请求示例

```bash
curl -N -X POST "http://localhost:8000/universal/process_bat/31a785b9e1c54367b394ff8949185a05"
```

### SSE 响应格式

接口返回 SSE 流，每条事件格式为：

```text
data: {"progress": 10}

data: {"progress": 20}

: heartbeat

data: {"progress": 100}
```

当前代码中流式事件主要返回 `progress` 字段。

### 进度含义

| progress | 说明 |
|---:|---|
| `10` | 文件下载完成 |
| `15` | 非 PDF 文件正在转换为 PDF |
| `20` | PDF 转换完成，或源文件本身已是 PDF |
| `25` | 完成 token 估算，确定 simple/chunked 解析模式 |
| `50` | 文件解析完成，JSON 已上传 |
| `90` | 三级索引写入完成 |
| `100` | 文档处理完成 |
| `-1` | 文档处理失败 |

### 后端最终处理结果结构

该接口的最终结果在服务内部为：

```json
{
  "doc_id": "31a785b9e1c54367b394ff8949185a05",
  "success": true,
  "mode": "chunked",
  "estimated_tokens": 120000,
  "chunk_num": 356,
  "process_duration": 83.51
}
```

注意：当前 SSE 实际输出只发送 `progress`，没有把上述 `result` 完整发送给客户端。

### 失败事件示例

```text
data: {"progress": -1}
```

失败时文档状态会被更新为 `error`，错误信息写入数据库的 `progress_msg`。

---

## 3. 文档 JSON 获取接口

### 基本信息

- **实际接口路径**：`GET /universal/doc-json/{doc_id}`
- **说明**：用户提到的 `/universal/doc_json/` 当前代码中不存在；实际路由使用中划线 `doc-json`，并且需要带 `doc_id`。
- **功能**：获取文档解析后的 JSON 树结构。可返回整篇 JSON，也可按 `chunk_id`、`node_id`、`chapter_id` 精确返回对应 JSON 节点。

### 路径参数

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `doc_id` | `string` | 是 | 文档 ID |

### Query 参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `chunk_id` | `string` | 否 | `null` | 按 chunk_id 精确返回对应 JSON 节点 |
| `node_id` | `string` | 否 | `null` | 按 node_id 精确返回对应 JSON 节点 |
| `chapter_id` | `string` | 否 | `null` | 按章节 node_id/title 返回对应章节 JSON 节点 |
| `include_children` | `boolean` | 否 | `true` | 返回节点时是否包含子节点 `nodes` |

优先级：

```text
chunk_id > node_id > chapter_id
```

即如果同时传多个参数，只会使用优先级最高的一个。

### 获取整篇 JSON 示例

```http
GET /universal/doc-json/31a785b9e1c54367b394ff8949185a05
```

响应示例：

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "doc_id": "31a785b9e1c54367b394ff8949185a05",
    "title": "工业互联网应用案例集（2023-2024年）-工业互联网产业联盟.pdf",
    "json": {
      "doc_name": "工业互联网应用案例集（2023-2024年）-工业互联网产业联盟",
      "structure": [
        {
          "title": "一、总体情况",
          "node_id": "0001",
          "start_page": 1,
          "end_page": 3,
          "nodes": [],
          "prefix_summary": "..."
        }
      ]
    }
  }
}
```

### 获取指定 chunk JSON 节点示例

```http
GET /universal/doc-json/31a785b9e1c54367b394ff8949185a05?chunk_id=（1）生产过程溯源%20-%20段落3&include_children=false
```

响应示例：

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "doc_id": "31a785b9e1c54367b394ff8949185a05",
    "title": "工业互联网应用案例集（2023-2024年）-工业互联网产业联盟.pdf",
    "query": {
      "chunk_id": "（1）生产过程溯源 - 段落3",
      "node_id": null,
      "chapter_id": null,
      "include_children": false
    },
    "match_count": 1,
    "matches": [
      {
        "path": [
          {
            "title": "3. 具体应用场景和应用模式",
            "node_id": "0030",
            "start_page": 12,
            "end_page": 16
          },
          {
            "title": "（1）生产过程溯源",
            "node_id": "0031",
            "start_page": null,
            "end_page": null
          },
          {
            "title": "（1）生产过程溯源 - 段落3",
            "node_id": "0032",
            "start_page": null,
            "end_page": null
          }
        ],
        "node": {
          "title": "（1）生产过程溯源 - 段落3",
          "node_id": "0032",
          "text": "利用工业互联网标识“一物一码”的特点..."
        }
      }
    ]
  }
}
```

### 获取指定章节 JSON 节点示例

```http
GET /universal/doc-json/31a785b9e1c54367b394ff8949185a05?chapter_id=0030&include_children=true
```

响应结构与 chunk 查询一致，但 `node` 中会包含该章节下的 `nodes` 子树。

### 未找到文档

```json
{
  "code": 404,
  "message": "文档不存在",
  "data": null
}
```

### 文档尚未处理

```json
{
  "code": 404,
  "message": "文档尚未完成处理，无可用 JSON",
  "data": null
}
```

### 未找到节点

```json
{
  "code": 404,
  "message": "未找到对应 JSON 节点: xxx",
  "data": null
}
```
