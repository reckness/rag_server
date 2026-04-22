"""
长文档 PDF → Markdown → JSON 分块处理流程
解决长文档 token 过大问题：
1. 检测 PDF 前几页中的目录（TOC），提取章节标题和页码
2. 按目录页码范围将 PDF 分块
3. 每个分块单独调用 LLM 转 Markdown，再生成子树
4. 最后合并所有子树为完整的树结构
"""
import asyncio
import sys
import os
import json
import re
import requests
import pymupdf
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.page_index_md import md_to_tree, extract_nodes_from_markdown, extract_node_text_content, build_tree_from_nodes, format_structure, generate_summaries_for_structure_md, write_node_id
from rag.utils import _detect_headers_footers, _remove_headers_footers

# ==================== 配置 ====================
PDF_PATH = os.path.join("pdf", "珠三角电子信息产业集群创新网络演化及其机理研究_王炜.pdf")
LLM_URL = "http://10.1.141.33:8001/v1/chat/completions"
LLM_MODEL = "Qwen3-8B"

MODEL = "Qwen3-8B"
IF_THINNING = False
THINNING_THRESHOLD = 5000
SUMMARY_TOKEN_THRESHOLD = 200
IF_SUMMARY = True
IF_ADD_NODE_TEXT = True

TOC_SCAN_PAGES = 15          # 扫描前 N 页寻找目录
MAX_TOKENS_PER_CHUNK = 6000  # 每个分块最大字符数（安全阈值）
CHUNK_TOKEN_THRESHOLD = 20000  # 章节 token 超过此阈值则按小节拆分


# ==================== Step 1: 提取 PDF 页面文本 ====================
def _table_data_to_markdown(data):
    """将表格数据 (list of lists) 转换为 Markdown 表格"""
    if not data or not data[0]:
        return ""

    num_cols = max(len(row) for row in data)

    # 清理单元格：None→""，换行→空格
    cleaned = []
    for row in data:
        cleaned_row = []
        for i in range(num_cols):
            cell = row[i] if i < len(row) else ""
            if cell is None:
                cell = ""
            else:
                cell = str(cell).replace("\n", " ").strip()
            cleaned_row.append(cell)
        cleaned.append(cleaned_row)

    # 构建 Markdown 表格
    lines = []
    # 表头
    lines.append("| " + " | ".join(cleaned[0]) + " |")
    # 分隔线
    lines.append("| " + " | ".join(["---"] * num_cols) + " |")
    # 数据行
    for row in cleaned[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _extract_page_with_tables(page):
    """提取单页文本，将表格区域替换为 Markdown 表格格式"""
    tabs = page.find_tables()

    if not tabs.tables:
        return page.get_text("text")

    # 收集表格信息，按垂直位置排序
    table_infos = []
    for tab in tabs.tables:
        bbox = tab.bbox  # (x0, y0, x1, y1)
        data = tab.extract()
        md_table = _table_data_to_markdown(data)
        if md_table:
            table_infos.append({
                'bbox': bbox,
                'md': md_table,
                'y0': bbox[1],
                'y1': bbox[3],
            })

    table_infos.sort(key=lambda t: t['y0'])

    # 按区域交替提取：非表格文本 + 表格 Markdown
    page_rect = page.rect
    parts = []
    current_y = page_rect.y0

    for tinfo in table_infos:
        # 提取表格上方的文本
        if tinfo['y0'] > current_y + 1:
            clip = pymupdf.Rect(page_rect.x0, current_y, page_rect.x1, tinfo['y0'])
            text = page.get_text("text", clip=clip).strip()
            if text:
                parts.append(text)

        # 插入 Markdown 表格
        parts.append(tinfo['md'])
        current_y = tinfo['y1']

    # 提取最后一个表格下方的文本
    if current_y < page_rect.y1 - 1:
        clip = pymupdf.Rect(page_rect.x0, current_y, page_rect.x1, page_rect.y1)
        text = page.get_text("text", clip=clip).strip()
        if text:
            parts.append(text)

    return "\n\n".join(parts)


def extract_all_pages(pdf_path):
    """提取 PDF 所有页面文本，表格自动转为 Markdown 表格格式，并去除页眉页脚"""
    doc = pymupdf.open(pdf_path)

    # 先用纯文本做页眉页脚检测
    plain_texts = [page.get_text("text") for page in doc]
    header_lines, footer_lines = _detect_headers_footers(plain_texts)
    if header_lines:
        print(f"[提取] 检测到页眉: {header_lines}")
    if footer_lines:
        print(f"[提取] 检测到页脚: {footer_lines}")

    # 再用表格感知提取每页内容
    table_page_count = 0
    cleaned_pages = []
    for i, page in enumerate(doc):
        tabs = page.find_tables()
        has_tables = bool(tabs.tables)
        if has_tables:
            table_page_count += 1
            text = _extract_page_with_tables(page)
        else:
            text = plain_texts[i]

        # 去除页眉页脚
        if header_lines or footer_lines:
            text = _remove_headers_footers(text, header_lines, footer_lines)
        cleaned_pages.append(text.strip())

    doc.close()
    print(f"[提取] 共 {len(cleaned_pages)} 页，其中 {table_page_count} 页包含表格")
    return cleaned_pages


# ==================== Step 2: 检测目录 ====================
def detect_toc_from_bookmarks(pdf_path):
    """尝试从 PDF 内嵌书签提取目录"""
    doc = pymupdf.open(pdf_path)
    toc = doc.get_toc(simple=True)  # [[level, title, page], ...]
    doc.close()

    if not toc:
        return None

    entries = []
    for level, title, page in toc:
        title = title.strip()
        if title and page > 0:
            entries.append({
                'level': level,
                'title': title,
                'page': page  # 1-indexed
            })

    if entries:
        print(f"[目录] 从 PDF 书签中提取到 {len(entries)} 个条目")
    return entries if entries else None


def detect_toc_from_text(cleaned_pages, scan_pages=15):
    """从前 N 页文本中检测目录页并解析条目"""
    # 目录页特征：包含大量 "标题...页码" 或 "标题 页码" 模式
    toc_page_indices = []

    for i in range(min(scan_pages, len(cleaned_pages))):
        text = cleaned_pages[i]
        if not text:
            continue

        # 检测"目录"/"目  录"/"CONTENTS" 关键词
        if re.search(r'目\s*录|contents|table\s+of\s+contents', text, re.IGNORECASE):
            # 检测是否有大量带页码的行（至少3行）
            dotted_lines = re.findall(r'.{2,}[\.…·]{3,}\s*\d+', text)
            spaced_lines = re.findall(r'.{2,}\s{2,}\d+\s*$', text, re.MULTILINE)
            if len(dotted_lines) + len(spaced_lines) >= 3:
                toc_page_indices.append(i)
                continue

        # 即使没有"目录"关键词，如果页面中超过 5 行有"标题...数字"模式也算目录页
        dotted_lines = re.findall(r'.{2,}[\.…·]{3,}\s*\d+', text)
        if len(dotted_lines) >= 5:
            toc_page_indices.append(i)

    if not toc_page_indices:
        print("[目录] 未在前几页中检测到目录")
        return None

    print(f"[目录] 检测到目录页: {[i+1 for i in toc_page_indices]}")

    # 合并所有目录页文本
    toc_text = "\n".join(cleaned_pages[i] for i in toc_page_indices)

    # 解析目录条目
    entries = parse_toc_text(toc_text)
    if entries:
        print(f"[目录] 从文本中解析出 {len(entries)} 个条目")
    return entries


def parse_toc_text(toc_text):
    """解析目录文本，提取标题和页码"""
    entries = []
    lines = toc_text.split('\n')

    # 模式1: "标题......页码" 或 "标题…………页码"
    pattern_dotted = re.compile(r'^(.+?)[\.…·]{2,}\s*(\d+)\s*$')
    # 模式2: "标题    页码"（多个空格分隔）
    pattern_spaced = re.compile(r'^(.+?)\s{3,}(\d+)\s*$')

    # 对于跨行的情况：标题一行，页码在下一行
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # 尝试匹配模式1
        m = pattern_dotted.match(line)
        if m:
            title = m.group(1).strip()
            page = int(m.group(2))
            if title and page > 0:
                level = _guess_toc_level(title)
                entries.append({'level': level, 'title': title, 'page': page})
            i += 1
            continue

        # 尝试匹配模式2
        m = pattern_spaced.match(line)
        if m:
            title = m.group(1).strip()
            page = int(m.group(2))
            if title and page > 0:
                level = _guess_toc_level(title)
                entries.append({'level': level, 'title': title, 'page': page})
            i += 1
            continue

        # 模式3: 标题在当前行，页码在下一行
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if re.match(r'^\d+$', next_line):
                page = int(next_line)
                # 检查当前行是否像标题（包含中文或字母，且有点号分隔）
                title_candidate = re.sub(r'[\.…·]+$', '', line).strip()
                if title_candidate and page > 0 and len(title_candidate) >= 2:
                    level = _guess_toc_level(title_candidate)
                    entries.append({'level': level, 'title': title_candidate, 'page': page})
                    i += 2
                    continue

        i += 1

    return entries


def _guess_toc_level(title):
    """根据标题格式猜测层级"""
    title = title.strip()
    # "第X章" → level 1
    if re.match(r'^第[一二三四五六七八九十\d]+章', title):
        return 1
    # "第X节" → level 2
    if re.match(r'^第[一二三四五六七八九十\d]+节', title):
        return 2
    # "一、" "二、" → level 2 or 3
    if re.match(r'^[一二三四五六七八九十]+、', title):
        return 3
    # "（一）" "（二）" → level 3 or 4
    if re.match(r'^（[一二三四五六七八九十]+）', title):
        return 4
    # 附录、参考文献 → level 1
    if re.match(r'^(附录|参考文献|致谢|摘\s*要|Abstract)', title):
        return 1
    return 2


# ==================== Step 3: 按目录分块 ====================
def build_chunks_from_toc(toc_entries, total_pages):
    """根据 TOC 条目生成分块（只取顶层章节做分块，避免过于碎片化）"""
    # 只取 level 1 的条目做分块边界
    top_entries = [e for e in toc_entries if e['level'] == 1]

    # 如果没有 level 1 条目，取 level <= 2
    if not top_entries:
        top_entries = [e for e in toc_entries if e['level'] <= 2]

    if not top_entries:
        top_entries = toc_entries

    chunks = []
    for i, entry in enumerate(top_entries):
        start_page = entry['page']  # 1-indexed
        if i + 1 < len(top_entries):
            end_page = top_entries[i + 1]['page'] - 1
        else:
            end_page = total_pages

        # 确保合理
        end_page = max(start_page, min(end_page, total_pages))

        chunks.append({
            'title': entry['title'],
            'level': entry['level'],
            'start_page': start_page,
            'end_page': end_page,
        })

    # 如果第一个章节不从第1页开始，添加前言块
    if chunks and chunks[0]['start_page'] > 1:
        chunks.insert(0, {
            'title': '前言',
            'level': 1,
            'start_page': 1,
            'end_page': chunks[0]['start_page'] - 1,
        })

    print(f"\n[分块] 共 {len(chunks)} 个分块:")
    for c in chunks:
        print(f"  - {c['title']} (p{c['start_page']}-{c['end_page']})")

    return chunks


def refine_chunks_by_token(chunks, toc_entries, cleaned_pages, token_threshold=CHUNK_TOKEN_THRESHOLD):
    """对超过 token 阈值的章节，按子节（level 2+）重新拆分"""
    if not toc_entries:
        return chunks

    refined = []
    for chunk in chunks:
        # 估算该章节的 token 数
        chunk_text = ""
        for p in range(chunk['start_page'] - 1, min(chunk['end_page'], len(cleaned_pages))):
            chunk_text += cleaned_pages[p]
        estimated_tokens = int(len(chunk_text) / 1.5)

        if estimated_tokens <= token_threshold:
            refined.append(chunk)
            continue

        # 超过阈值，在该章节页码范围内找子节点
        sub_entries = [
            e for e in toc_entries
            if e['level'] > chunk['level']
            and e['page'] >= chunk['start_page']
            and e['page'] <= chunk['end_page']
        ]

        if not sub_entries:
            # 没有子节点，无法拆分，保持原样
            print(f"  [拆分] {chunk['title']} ({estimated_tokens} tokens) 无子节可拆，保持原样")
            refined.append(chunk)
            continue

        print(f"  [拆分] {chunk['title']} ({estimated_tokens} tokens > {token_threshold})，"
              f"拆分为 {len(sub_entries)} 个子节")

        # 生成子块
        for j, sub in enumerate(sub_entries):
            sub_start = sub['page']
            if j + 1 < len(sub_entries):
                sub_end = sub_entries[j + 1]['page'] - 1
            else:
                sub_end = chunk['end_page']
            sub_end = max(sub_start, min(sub_end, chunk['end_page']))

            refined.append({
                'title': sub['title'],
                'level': sub['level'],
                'start_page': sub_start,
                'end_page': sub_end,
                '_parent_title': chunk['title'],
                '_parent_level': chunk['level'],
                '_parent_start': chunk['start_page'],
                '_parent_end': chunk['end_page'],
            })

        # 如果第一个子节不是从章节开头开始，添加章节开头的文本块
        if sub_entries[0]['page'] > chunk['start_page']:
            refined.insert(len(refined) - len(sub_entries), {
                'title': chunk['title'],
                'level': chunk['level'],
                'start_page': chunk['start_page'],
                'end_page': sub_entries[0]['page'] - 1,
                '_parent_title': chunk['title'],
                '_parent_level': chunk['level'],
                '_parent_start': chunk['start_page'],
                '_parent_end': chunk['end_page'],
            })

    print(f"\n[拆分后] 共 {len(refined)} 个分块:")
    for c in refined:
        parent = c.get('_parent_title')
        prefix = f"    └─ " if parent else "  - "
        print(f"{prefix}{c['title']} (p{c['start_page']}-{c['end_page']})")

    return refined


# ==================== Step 4: LLM 转 Markdown ====================
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


def chunk_text_to_markdown(chunk_title, chunk_text):
    """将分块文本通过 LLM 转换为 Markdown"""
    prompt = f"""你是一个文档结构化专家。下面是从 PDF 中提取的一个章节的纯文本。
该章节标题为：{chunk_title}

请将这些文本转换为结构良好的 Markdown 格式，要求：

1. 识别文档中的标题层级关系，使用 Markdown 标题语法（# ~ ######）标注
2. 章节大标题用 `##`（二级标题），子节用 `###`（三级标题），以此类推
3. 保留所有正文内容，不要遗漏任何信息
4. 移除所有分页标记（如"=== 第 X 页 ==="）、页码等
5. 将跨页断开的段落合并为完整段落
6. 不要添加原文中没有的内容
7. 直接输出 Markdown，不要用代码块包裹，不要输出任何其他说明文字

PDF 提取文本：
{chunk_text}
"""
    md_content = llm_call(prompt)
    return md_content


def pages_to_chunk_text(cleaned_pages, start_page, end_page):
    """将指定页范围的文本合并为分块文本"""
    parts = []
    for i in range(start_page - 1, min(end_page, len(cleaned_pages))):
        text = cleaned_pages[i]
        if text.strip():
            parts.append(f"=== 第 {i+1} 页 ===\n{text}")
    return "\n\n".join(parts)


# ==================== Step 5: 分块处理并合并 ====================
def build_subtree_from_markdown(md_content):
    """从 Markdown 内容构建子树（同步版本，不含摘要）"""
    node_list, markdown_lines = extract_nodes_from_markdown(md_content)
    if not node_list:
        return []
    nodes_with_content = extract_node_text_content(node_list, markdown_lines)
    tree = build_tree_from_nodes(nodes_with_content)
    return tree


def merge_subtrees(chunks, subtrees, toc_entries):
    """将所有子树合并为一棵完整的树（支持被拆分的子节归入父章节）"""
    merged = []
    # 用于收集被拆分章节的子节点：parent_title → chapter_node
    parent_map = {}

    for i, (chunk, subtree) in enumerate(zip(chunks, subtrees)):
        parent_title = chunk.get('_parent_title')

        # --- 被拆分的子节：归入父章节 ---
        if parent_title:
            if parent_title not in parent_map:
                # 创建父章节节点
                parent_map[parent_title] = {
                    'title': parent_title,
                    'start_page': chunk.get('_parent_start', chunk['start_page']),
                    'end_page': chunk.get('_parent_end', chunk['end_page']),
                    'nodes': []
                }
                merged.append(parent_map[parent_title])

            parent_node = parent_map[parent_title]
            # 更新父节点 end_page
            if chunk['end_page'] > parent_node.get('end_page', 0):
                parent_node['end_page'] = chunk['end_page']

            if not subtree:
                continue

            # 构建子节节点
            section_node = {
                'title': chunk['title'],
                'start_page': chunk['start_page'],
                'end_page': chunk['end_page'],
                'nodes': []
            }
            if len(subtree) == 1 and _titles_similar(subtree[0]['title'], chunk['title']):
                section_node['nodes'] = subtree[0].get('nodes', [])
                if subtree[0].get('text'):
                    section_node['text'] = subtree[0]['text']
            else:
                section_node['nodes'] = subtree

            parent_node['nodes'].append(section_node)
            continue

        # --- 普通章节（未被拆分）---
        if not subtree:
            continue

        chapter_node = {
            'title': chunk['title'],
            'start_page': chunk['start_page'],
            'end_page': chunk['end_page'],
            'nodes': []
        }

        if len(subtree) == 1 and _titles_similar(subtree[0]['title'], chunk['title']):
            chapter_node['nodes'] = subtree[0].get('nodes', [])
            if subtree[0].get('text'):
                chapter_node['text'] = subtree[0]['text']
        else:
            chapter_node['nodes'] = subtree

        merged.append(chapter_node)

    return merged


def _titles_similar(a, b):
    """模糊比较两个标题"""
    a = re.sub(r'\s+', '', (a or '').strip().lower())
    b = re.sub(r'\s+', '', (b or '').strip().lower())
    if not a or not b:
        return False
    return a == b or a in b or b in a


def assign_node_ids(tree, counter=None):
    """递归分配 node_id"""
    if counter is None:
        counter = [1]
    for node in tree:
        node['node_id'] = str(counter[0]).zfill(4)
        counter[0] += 1
        if node.get('nodes'):
            assign_node_ids(node['nodes'], counter)


def clean_tree(tree):
    """清理树节点，移除临时字段"""
    cleaned = []
    for node in tree:
        clean_node = {'title': node['title']}
        if 'node_id' in node:
            clean_node['node_id'] = node['node_id']
        if 'start_page' in node:
            clean_node['start_page'] = node['start_page']
        if 'end_page' in node:
            clean_node['end_page'] = node['end_page']
        if 'text' in node:
            clean_node['text'] = node['text']
        if 'summary' in node:
            clean_node['summary'] = node['summary']
        if 'prefix_summary' in node:
            clean_node['prefix_summary'] = node['prefix_summary']
        if node.get('nodes'):
            clean_node['nodes'] = clean_tree(node['nodes'])
        cleaned.append(clean_node)
    return cleaned


# ==================== Step 6: 无目录时的回退处理 ====================
def split_by_page_count(cleaned_pages, pages_per_chunk=10):
    """无目录时按固定页数分块"""
    chunks = []
    total = len(cleaned_pages)
    for start in range(0, total, pages_per_chunk):
        end = min(start + pages_per_chunk, total)
        chunks.append({
            'title': f'第 {start+1}-{end} 页',
            'level': 1,
            'start_page': start + 1,
            'end_page': end,
        })
    print(f"\n[分块] 无目录，按每 {pages_per_chunk} 页分块，共 {len(chunks)} 块")
    return chunks


# ==================== 主流程（可复用） ====================
async def process_pdf_chunked(
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
    toc_scan_pages: int = TOC_SCAN_PAGES,
    pages_per_chunk: int = 10,
):
    """
    长文档 PDF → 分块 Markdown → 合并 JSON 树结构

    Args:
        pdf_path: PDF 文件路径
        output_dir: 输出目录
        llm_url: LLM API 地址
        llm_model: LLM 模型名称
        model: 摘要生成使用的模型
        if_thinning: 是否进行树裁剪
        thinning_threshold: 裁剪阈值
        summary_token_threshold: 摘要 token 阈值
        if_summary: 是否生成摘要
        if_add_node_text: 是否保留节点文本
        toc_scan_pages: 扫描前 N 页寻找目录
        pages_per_chunk: 无目录时每块页数

    Returns:
        dict: {"result": 结构化JSON, "json_path": JSON文件路径, "md_path": MD文件路径}
    """
    global LLM_URL, LLM_MODEL
    LLM_URL = llm_url
    LLM_MODEL = llm_model

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: 提取所有页面文本
    print("=" * 60)
    print("[Step 1] 提取 PDF 文本")
    print("=" * 60)
    cleaned_pages = extract_all_pages(pdf_path)

    # Step 2: 检测目录
    print("\n" + "=" * 60)
    print("[Step 2] 检测目录")
    print("=" * 60)
    toc_entries = detect_toc_from_bookmarks(pdf_path)
    if not toc_entries:
        toc_entries = detect_toc_from_text(cleaned_pages, scan_pages=toc_scan_pages)

    # Step 3: 分块
    print("\n" + "=" * 60)
    print("[Step 3] 按目录分块")
    print("=" * 60)
    if toc_entries:
        chunks = build_chunks_from_toc(toc_entries, len(cleaned_pages))
    else:
        chunks = split_by_page_count(cleaned_pages, pages_per_chunk=pages_per_chunk)

    # Step 3.5: 对超大章节按子节拆分
    print("\n" + "=" * 60)
    print("[Step 3.5] 检查章节 token 数，超过 20000 的按小节拆分")
    print("=" * 60)
    chunks = refine_chunks_by_token(chunks, toc_entries, cleaned_pages)

    # Step 4: 逐块处理（并行，最多 2 个 LLM 请求同时发送）
    LLM_PARALLEL = 1
    print("\n" + "=" * 60)
    print(f"[Step 4] 逐块调用 LLM 生成 Markdown 并构建子树（并行={LLM_PARALLEL}）")
    print("=" * 60)

    # 预处理：提取每个分块的文本，标记空块
    chunk_texts = []
    for i, chunk in enumerate(chunks):
        chunk_text = pages_to_chunk_text(cleaned_pages, chunk['start_page'], chunk['end_page'])
        chunk_texts.append(chunk_text)

    def _process_one_chunk(idx):
        """处理单个分块：LLM 转 MD + 构建子树（在线程中执行）"""
        chunk = chunks[idx]
        chunk_text = chunk_texts[idx]
        char_count = len(chunk_text)
        print(f"\n--- 分块 {idx+1}/{len(chunks)}: {chunk['title']} "
              f"(p{chunk['start_page']}-{chunk['end_page']}, {char_count} 字符) ---")

        if not chunk_text.strip():
            print(f"  [跳过] 空分块")
            return idx, None, []

        md_content = chunk_text_to_markdown(chunk['title'], chunk_text)
        subtree = build_subtree_from_markdown(md_content)
        print(f"  [完成] 分块 {idx+1} MD {len(md_content)} 字符, 子树 {len(subtree)} 个根节点")
        return idx, md_content, subtree

    # 并行执行
    results = [None] * len(chunks)  # (md_content, subtree)
    with ThreadPoolExecutor(max_workers=LLM_PARALLEL) as executor:
        futures = {executor.submit(_process_one_chunk, i): i for i in range(len(chunks))}
        for future in as_completed(futures):
            idx, md_content, subtree = future.result()
            results[idx] = (md_content, subtree)

    # 按原始顺序整理结果
    subtrees = []
    all_md_parts = []
    for i, (md_content, subtree) in enumerate(results):
        subtrees.append(subtree)
        if md_content:
            chunk = chunks[i]
            all_md_parts.append(f"<!-- chunk: {chunk['title']} p{chunk['start_page']}-{chunk['end_page']} -->\n{md_content}")

    # 保存完整 Markdown
    md_path = os.path.join(output_dir, f"{pdf_name}_chunked.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(all_md_parts))
    print(f"\n[保存] 合并 Markdown → {md_path}")

    # Step 5: 合并子树
    print("\n" + "=" * 60)
    print("[Step 5] 合并所有子树")
    print("=" * 60)
    merged_tree = merge_subtrees(chunks, subtrees, toc_entries)
    assign_node_ids(merged_tree)

    # 可选：生成摘要
    if if_summary:
        print("\n[摘要] 正在为各节点生成摘要...")
        formatted = format_structure(merged_tree, order=['title', 'node_id', 'start_page', 'end_page', 'summary', 'prefix_summary', 'text', 'nodes'])
        formatted = await generate_summaries_for_structure_md(
            formatted,
            summary_token_threshold=summary_token_threshold,
            model=model
        )
        if not if_add_node_text:
            formatted = format_structure(formatted, order=['title', 'node_id', 'start_page', 'end_page', 'summary', 'prefix_summary', 'nodes'])
        merged_tree = formatted

    # Step 6: 保存结果
    print("\n" + "=" * 60)
    print("[Step 6] 保存结果")
    print("=" * 60)
    result = {
        'doc_name': pdf_name,
        'total_pages': len(cleaned_pages),
        'toc_detected': toc_entries is not None,
        'num_chunks': len(chunks),
        'structure': merged_tree,
    }

    json_path = os.path.join(output_dir, f"{pdf_name}_chunked_structure.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[完成] JSON 结构已保存至: {json_path}")

    return {"result": result, "json_path": json_path, "md_path": md_path}


async def main():
    output = await process_pdf_chunked(pdf_path=PDF_PATH)
    result = output["result"]

    # 打印结构预览
    print(f"\n文档名: {result.get('doc_name', '')}")
    print(f"总页数: {result.get('total_pages', '')}")
    print(f"分块数: {result.get('num_chunks', '')}")
    print(f"\n结构预览:")

    def print_tree(nodes, indent=0):
        for node in nodes:
            title = node.get("title", "")
            sp = node.get("start_page", "")
            ep = node.get("end_page", "")
            page_info = f" [p{sp}-{ep}]" if sp else ""
            summary = node.get("summary", node.get("prefix_summary", ""))
            summary_str = f" — {summary[:50]}..." if summary else ""
            print("  " * indent + f"├── {title}{page_info}{summary_str}")
            if node.get("nodes"):
                print_tree(node["nodes"], indent + 1)

    print_tree(result.get("structure", []))


if __name__ == "__main__":
    asyncio.run(main())
