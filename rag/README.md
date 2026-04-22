# PageIndex PDF 处理流程使用指南

## 简介

`rag/page_index.py` 是 rag-service 的核心模块，负责将 PDF 文档解析为**层次化树结构**（目录检测 → 章节定位 → 节点摘要），输出结构化 JSON 文件，供下游 RAG 检索使用。

---

## 环境准备

### 1. 安装 Python 环境

```bash
# 项目使用 Python 3.11
# 创建虚拟环境（已存在可跳过）
python -m venv venv
```

### 2. 激活虚拟环境

```bash
# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 确认 LLM 服务可用

当前默认调用内网部署的 Qwen3-8B 模型：

| 配置项 | 值 |
|--------|-----|
| API 地址 | `http://10.1.141.33:8001/v1/chat/completions` |
| 模型 | `Qwen3-8B` |

> 如需更换模型或地址，修改 `rag/utils.py` 中 `llm_completion()` 和 `llm_acompletion()` 函数内的 `url` 和 `model` 字段。

---

## 快速开始

### 方式一：运行脚本（推荐）

1. 将 PDF 文件放入 `pdf/` 目录下

2. 修改 `run_pdf.py` 中的文件路径：

```python
PDF_PATH = os.path.join("pdf", "你的文件名.pdf")
```

3. 运行：

```bash
python run_pdf.py
```

4. 结果会自动保存到系统临时目录，控制台会打印输出路径和树结构预览。

### 方式二：在代码中调用

```python
import asyncio
from rag.page_index import page_index

async def main():
    result_path = await page_index(
        doc="pdf/你的文件名.pdf",
        # 以下为可选参数，不传则使用 rag/config.yaml 中的默认值
        model="Qwen3-8B",
        if_add_node_id="yes",
        if_add_node_summary="yes",
        if_add_node_text="yes",
        if_add_doc_description="no",
    )
    print(f"结果文件: {result_path}")

asyncio.run(main())
```

---

## 配置说明

默认配置位于 `rag/config.yaml`：

```yaml
model: "Qwen3-8B"                  # LLM 模型名称
toc_check_page_num: 10             # 检测目录时扫描的最大页数
max_page_num_each_node: 10         # 单节点最大页数（超过则递归拆分）
max_token_num_each_node: 10000     # 单节点最大 token 数（超过则递归拆分）
if_add_node_id: "yes"              # 是否为每个节点分配唯一 ID
if_add_node_summary: "yes"         # 是否为每个节点生成摘要
if_add_doc_description: "no"       # 是否生成文档级别的描述
if_add_node_text: "no"             # 是否在结果中包含节点原文
```

调用 `page_index()` 时传入的参数会覆盖配置文件中的对应值。

---

## 处理流程

```
输入 PDF
  │
  ▼
① 文本提取 —— PyPDF2 逐页提取内嵌文本 + 计算 token 数
  │
  ▼
② 目录检测 —— 逐页检查是否为目录页（LLM 判断）
  │
  ├── 有目录 + 有页码 → 提取目录 → 转 JSON → 页码偏移校准
  ├── 有目录 + 无页码 → 提取目录 → 转 JSON → 逐组匹配物理页码
  └── 无目录 ─────────→ 从原文直接生成目录树结构
  │
  ▼
③ 目录验证 —— 并发检查每个条目的页码是否正确（LLM 判断）
  │
  ├── 准确率 = 100%  → 直接使用
  ├── 准确率 > 60%   → 修复错误条目（最多重试 3 次）
  └── 准确率 ≤ 60%   → 降级到下一种模式重新处理
  │
  ▼
④ 大节点拆分 —— 对超过阈值的节点递归生成子结构
  │
  ▼
⑤ 后处理 —— 添加节点 ID / 原文 / 摘要 / 文档描述（按配置）
  │
  ▼
输出 JSON 文件
```

---

## 输出格式

结果为 JSON 文件，结构如下：

```json
{
  "doc_name": "低空经济__开启立体城市空间新未来_劳莘.pdf",
  "structure": [
    {
      "title": "行业观察",
      "node_id": "0001",
      "start_index": 1,
      "end_index": 5,
      "summary": "该部分文档主要探讨低空经济...",
      "text": "（可选）节点对应的原始文本",
      "nodes": [
        {
          "title": "低空经济的发展动能",
          "node_id": "0002",
          "start_index": 2,
          "end_index": 3,
          "summary": "...",
          "nodes": []
        }
      ]
    }
  ]
}
```

**字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `title` | string | 章节标题 |
| `node_id` | string | 节点唯一标识（4位编号） |
| `start_index` | int | 起始页码（1-indexed） |
| `end_index` | int | 结束页码 |
| `summary` | string | LLM 生成的章节摘要 |
| `text` | string | 节点对应的 PDF 原文（需开启 `if_add_node_text`） |
| `nodes` | array | 子节点列表（递归结构） |

---

## 查看结果

运行 `show_result.py` 可以以树形格式查看最近一次处理的结果：

```bash
python show_result.py
```

输出示例：

```
文档名: 低空经济__开启立体城市空间新未来_劳莘.pdf
顶层节点数: 1
结果文件: C:\Users\...\Temp\低空经济__开启立体城市空间新未来_劳莘.pdf.json
文件大小: 85.2 KB

├── 行业观察 [p1-5]  子节点:19
     摘要: 该部分文档主要探讨低空经济作为新兴经济模式的发展潜力与前景...
  ├── 低空经济的发展动能 [p2-2]  子节点:0
       摘要: 该部分文档主要介绍了中国通用航空及低空经济的发展现状...
  ├── 低空载人交通 [p3-2]  子节点:0
  ...
```

---

## 注意事项

1. **仅支持内嵌文本的 PDF** — 扫描件/图片型 PDF 无法提取文本，需先 OCR 处理
2. **LLM 依赖** — 处理过程中会多次调用 LLM（目录检测、结构生成、摘要等），需确保 LLM 服务在线
3. **处理时间** — 取决于 PDF 页数和 LLM 响应速度，5 页 PDF 约 1-2 分钟
4. **日志文件** — 每次处理会在 `logs/` 目录下生成详细日志（JSON 格式），可用于排查问题
