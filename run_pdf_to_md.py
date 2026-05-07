"""
PDF → Markdown → JSON 结构化处理流程
1. 使用 PyMuPDF 提取 PDF 文本
2. 调用 LLM (Qwen3-8B) 将提取的文本转换为带标题层级的 Markdown
3. 使用 page_index_md.py 的 md_to_tree() 将 Markdown 解析为 JSON 树结构
"""
import asyncio
import sys
import os
import json
import re
import requests
import pymupdf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.page_index_md import md_to_tree
from rag.utils import _detect_headers_footers, _remove_headers_footers


def build_doc_summary_from_chapter_summaries(structure):
    summaries = []
    for node in structure or []:
        title = (node.get("title") or "").strip()
        summary = (node.get("prefix_summary") or node.get("summary") or "").strip()
        if summary:
            summaries.append(f"{title}：{summary}" if title else summary)
    return "\n".join(summaries)


def generate_doc_summary_from_chapter_summaries(structure):
    chapter_summaries = build_doc_summary_from_chapter_summaries(structure)
    if not chapter_summaries.strip():
        return ""
    prompt = f"""你获得了一篇文档各章节的摘要，请基于这些章节摘要生成一段文章级总摘要。

要求：
1. 控制在 300-500 字
2. 概括文档主题、核心内容、关键结论和整体结构
3. 不要逐章罗列
4. 不要添加章节摘要中没有的信息
5. 直接返回总摘要，不要包含任何其他文本

各章节摘要：
{chapter_summaries}
"""
    try:
        return llm_call(prompt, max_tokens=2048)
    except Exception as e:
        print(f"[摘要] 文章总摘要生成失败，回退章节摘要拼接: {e}")
        return chapter_summaries


# ==================== 配置 ====================
PDF_PATH = os.path.join("pdf", "辽宁省低空经济高质量发展路径研究.pdf")
LLM_URL = "http://10.1.141.33:8080/v1/chat/completions"
LLM_MODEL = "qwen3.5-35b-int4"

# md_to_tree 参数
MODEL = "qwen3.5-35b-int4"
IF_THINNING = False
THINNING_THRESHOLD = 5000
SUMMARY_TOKEN_THRESHOLD = 200
IF_SUMMARY = True          # 是否生成摘要
IF_ADD_NODE_TEXT = True     # 是否保留节点文本


# ==================== Step 1: 提取 PDF 文本（自动去除页眉页脚） ====================
def extract_pdf_text(pdf_path):
    """使用 PyMuPDF 提取 PDF 每页文本，自动检测并去除页眉页脚，合并为一个字符串"""
    doc = pymupdf.open(pdf_path)
    output_md_path= './pdf/output.md'
    raw_texts = []
    for page in doc:
        raw_texts.append(page.get_text("text"))
    doc.close()

    # 自动检测页眉页脚
    header_lines, footer_lines = _detect_headers_footers(raw_texts)
    if header_lines:
        print(f"[Step 1] 检测到页眉: {header_lines}")
    if footer_lines:
        print(f"[Step 1] 检测到页脚: {footer_lines}")

    # 逐页去除页眉页脚并拼接
    pages = []
    for i, text in enumerate(raw_texts):
        cleaned = _remove_headers_footers(text, header_lines, footer_lines) if (header_lines or footer_lines) else text
        cleaned = cleaned.strip()
        if cleaned:
            pages.append(f"=== 第 {i+1} 页 ===\n{cleaned}")
    full_text = "\n\n".join(pages)
    ## 存储txt
    # with open(output_md_path, "w", encoding="utf-8") as f:
    #     f.write(full_text)
    print(f"[Step 1] PDF 文本提取完成，共 {len(pages)} 页，{len(full_text)} 字符（已去除页眉页脚）")
    return full_text


# ==================== Step 2: LLM 将文本转为 Markdown ====================
def llm_call(prompt, max_tokens=8192):
    """调用 LLM"""
    headers = {"Content-Type": "application/json", "Authorization": "Bearer "}
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0
    }
    resp = requests.post(LLM_URL, headers=headers, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def pdf_text_to_markdown(pdf_text):
    """调用 LLM 将 PDF 提取的纯文本转为带标题层级的 Markdown"""
    prompt = f"""你是一个文档结构化专家。下面是从 PDF 中提取的纯文本，包含多页内容。
请你将这些文本转换为结构良好的 Markdown 格式，要求：

1. 识别文档中的标题层级关系，使用 Markdown 标题语法（# ~ ######）标注
2. 文档大标题用 `#`（一级标题），主要章节用 `##`（二级标题），子章节用 `###`（三级标题），以此类推
3. 保留所有正文内容，不要遗漏任何信息
4. 移除所有分页标记、页眉页脚（如“=== 第 X 页 ===”、“- X -”或独立成行的页码）。
5. 删除文本中出现的具体人名、企业名称（如“***公司”）、联系方式等实体。确保清理后不影响主体上下文的连贯性。
5. 将跨页断开的段落合并为完整段落
6. 注意：页眉页脚已在提取阶段被 Python 脚本自动去除，无需再处理
7. 不要添加原文中没有的内容
8. 直接输出 Markdown，不要用代码块包裹，不要输出任何其他说明文字

PDF 提取文本：
{pdf_text}
"""
    print(f"[Step 2] 正在调用 LLM 转换为 Markdown...")
    md_content = llm_call(prompt)
    print(f"[Step 2] Markdown 生成完成，{len(md_content)} 字符")
    return md_content


# ==================== Step 3: 可复用的处理函数 ====================
async def process_pdf_simple(
    pdf_path: str,
    output_dir: str = "pdf",
    llm_url: str = LLM_URL,
    llm_model: str = LLM_MODEL,
    model: str = MODEL,
    if_thinning: bool = IF_THINNING,
    thinning_threshold: int = THINNING_THRESHOLD,
    summary_token_threshold: int = SUMMARY_TOKEN_THRESHOLD,
    if_summary: bool = IF_SUMMARY,
    if_add_node_text: bool = IF_ADD_NODE_TEXT,
):
    """
    PDF → Markdown → JSON 完整处理流程（适用于短文档）

    Args:
        pdf_path: PDF 文件路径
        output_dir: 输出目录
        llm_url: LLM API 地址
        llm_model: LLM 模型名称
        model: md_to_tree 使用的模型
        if_thinning: 是否进行树裁剪
        thinning_threshold: 裁剪阈值
        summary_token_threshold: 摘要 token 阈值
        if_summary: 是否生成摘要
        if_add_node_text: 是否保留节点文本

    Returns:
        dict: {"result": 结构化JSON, "json_path": JSON文件路径, "md_path": MD文件路径}
    """
    global LLM_URL, LLM_MODEL
    LLM_URL = llm_url
    LLM_MODEL = llm_model

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: 提取 PDF 文本
    pdf_text = extract_pdf_text(pdf_path)

    # Step 2: LLM 转 Markdown
    md_content = pdf_text_to_markdown(pdf_text)

    # 保存 Markdown 文件（md_to_tree 需要从磁盘读取，由调用方负责清理）
    md_path = os.path.join(output_dir, f"{pdf_name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[Step 2] Markdown 已保存至: {md_path}")

    # Step 3: 用 page_index_md 的 md_to_tree 处理
    print(f"\n[Step 3] 开始用 page_index_md 处理 Markdown → JSON...")
    result = await md_to_tree(
        md_path=md_path,
        if_thinning=if_thinning,
        min_token_threshold=thinning_threshold,
        if_add_node_summary='yes' if if_summary else 'no',
        summary_token_threshold=summary_token_threshold,
        model=model,
        if_add_node_text='yes' if if_add_node_text else 'no',
        if_add_node_id='yes',
    )
    result["summary"] = generate_doc_summary_from_chapter_summaries(result.get("structure", []))

    # 保存 JSON 结果（ESConverter / MinIO 上传需要从磁盘读取，由调用方负责清理）
    json_path = os.path.join(output_dir, f"{pdf_name}_md_structure.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[完成] JSON 结构已保存至: {json_path}")

    return {"result": result, "json_path": json_path, "md_path": md_path}


async def main():
    output = await process_pdf_simple(pdf_path=PDF_PATH)
    result = output["result"]

    # 打印结构预览
    print(f"\n文档名: {result.get('doc_name', '')}")
    print(f"结构预览:")

    def print_tree(nodes, indent=0):
        for node in nodes:
            title = node.get("title", "")
            summary = node.get("summary", node.get("prefix_summary", ""))
            summary_str = f" — {summary[:60]}..." if summary else ""
            print("  " * indent + f"├── {title}{summary_str}")
            if node.get("nodes"):
                print_tree(node["nodes"], indent + 1)

    print_tree(result.get("structure", []))


if __name__ == "__main__":
    asyncio.run(main())
