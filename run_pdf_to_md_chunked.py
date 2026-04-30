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
import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.page_index_md import md_to_tree, extract_nodes_from_markdown, extract_node_text_content, build_tree_from_nodes, format_structure, generate_summaries_for_structure_md, write_node_id
from rag.utils import _detect_headers_footers, _remove_headers_footers

# ==================== 配置 ====================
PDF_PATH = os.path.join("pdf", "珠三角电子信息产业集群创新网络演化及其机理研究_王炜.pdf")
LLM_URL = "http://10.1.141.33:8080/v1/chat/completions"
LLM_MODEL = "qwen3.5-35b-int4"

MODEL = "qwen3.5-35b-int4"
IF_THINNING = False
THINNING_THRESHOLD = 5000
SUMMARY_TOKEN_THRESHOLD = 200
IF_SUMMARY = True
IF_ADD_NODE_TEXT = True

TOC_SCAN_PAGES = 15          # 扫描前 N 页寻找目录
MAX_TOKENS_PER_CHUNK = 6000  # 每个分块最大字符数（安全阈值）
CHUNK_TOKEN_THRESHOLD = 20000  # 章节 token 超过此阈值则按小节拆分

ENABLE_PDF_CLEANING = True   # 是否启用 PDF 文本清洗规则

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


def _normalize_pdf_whitespace(text):
    text = text or ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def _remove_common_page_noise(text, page_num=None):
    lines = text.split("\n")
    if not lines:
        return text

    page_patterns = [
        r"^\s*(?:第\s*)?\d{1,4}\s*(?:页)?\s*$",
        r"^\s*[-–—]\s*\d{1,4}\s*[-–—]\s*$",
        r"^\s*page\s+\d{1,4}(?:\s*/\s*\d{1,4})?\s*$",
        r"^\s*\d{1,4}\s*/\s*\d{1,4}\s*$",
    ]
    if page_num is not None:
        page_patterns.append(rf"^\s*(?:第\s*)?{page_num}\s*(?:页)?\s*$")

    candidates = set(range(min(3, len(lines))))
    candidates.update(range(max(0, len(lines) - 3), len(lines)))
    cleaned = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if idx in candidates and any(re.match(pattern, stripped, re.IGNORECASE) for pattern in page_patterns):
            continue
        if re.match(r"^\s*(?:扫描全能王|CamScanner|版权所有|Copyright\s+©?)", stripped, re.IGNORECASE):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _fix_pdf_line_breaks(text):
    protected_lines = []
    placeholders = {}
    for idx, line in enumerate(text.split("\n")):
        if line.lstrip().startswith("|"):
            key = f"__TABLE_LINE_{idx}__"
            placeholders[key] = line
            protected_lines.append(key)
        else:
            protected_lines.append(line)

    text = "\n".join(protected_lines)
    text = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff，。；：、“”‘’（）《》])\n(?=[\u4e00-\u9fff（《“‘])", "", text)
    text = re.sub(r"(?<=[A-Za-z0-9,;:])\n(?=[a-z(])", " ", text)
    for key, line in placeholders.items():
        text = text.replace(key, line)
    return text


def clean_pdf_page_text(text, page_num=None):
    text = _normalize_pdf_whitespace(text)
    text = _remove_common_page_noise(text, page_num)
    text = _fix_pdf_line_breaks(text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_all_pages(pdf_path):
    """提取 PDF 所有页面文本，表格自动转为 Markdown 表格格式，并去除页眉页脚"""
    doc = pymupdf.open(pdf_path)

    # 先用纯文本做页眉页脚检测
    plain_texts = []
    for page in doc:
        try:
            plain_texts.append(page.get_text("text"))
        except Exception as e:
            print(f"[提取] 警告: 第 {page.number + 1} 页无法提取文本: {e}")
            plain_texts.append("")
    header_lines, footer_lines = _detect_headers_footers(plain_texts)
    if header_lines:
        print(f"[提取] 检测到页眉: {header_lines}")
    if footer_lines:
        print(f"[提取] 检测到页脚: {footer_lines}")

    # 再用表格感知提取每页内容
    table_page_count = 0
    cleaned_pages = []
    for i, page in enumerate(doc):
        try:
            tabs = page.find_tables()
            has_tables = bool(tabs.tables)
        except Exception as e:
            print(f"[提取] 警告: 第 {i + 1} 页表格检测失败: {e}")
            has_tables = False
        if has_tables:
            table_page_count += 1
            try:
                text = _extract_page_with_tables(page)
            except Exception as e:
                print(f"[提取] 警告: 第 {i + 1} 页表格提取失败，回退到纯文本: {e}")
                text = plain_texts[i]
        else:
            text = plain_texts[i]

        # 去除页眉页脚
        if header_lines or footer_lines:
            text = _remove_headers_footers(text, header_lines, footer_lines)
        if ENABLE_PDF_CLEANING:
            text = clean_pdf_page_text(text, i + 1)
        cleaned_pages.append(text)

    doc.close()
    print(f"[提取] 共 {len(cleaned_pages)} 页，其中 {table_page_count} 页包含表格")
    return cleaned_pages


def extract_page_text_blocks(pdf_path):
    doc = pymupdf.open(pdf_path)
    page_blocks = []
    for page_idx, page in enumerate(doc, start=1):
        blocks = []
        try:
            raw_blocks = page.get_text("blocks")
        except Exception as e:
            print(f"[提取] 警告: 第 {page_idx} 页无法提取文本块: {e}")
            page_blocks.append(blocks)
            continue
        for block in raw_blocks:
            if len(block) < 5:
                continue
            x0, y0, x1, y1, text = block[:5]
            text = (text or "").strip()
            if not text:
                continue
            blocks.append({
                "page_index": page_idx,
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "text": text,
            })
        page_blocks.append(blocks)
    doc.close()
    return page_blocks


def _normalize_for_position_match(text):
    return re.sub(r"\s+", "", re.sub(r"^#{1,6}\s*", "", text or ""))


def _union_bbox(bboxes):
    return [
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    ]


def locate_text_positions(text, page_blocks, start_page=None, end_page=None):
    target = _normalize_for_position_match(text)
    if not target:
        return []

    locations = []
    page_start = max(1, start_page or 1)
    page_end = min(len(page_blocks), end_page or len(page_blocks))

    for page_idx in range(page_start, page_end + 1):
        blocks = page_blocks[page_idx - 1]
        hit_bboxes = []
        for block in blocks:
            block_text = _normalize_for_position_match(block.get("text", ""))
            if not block_text:
                continue
            if block_text in target or target in block_text:
                hit_bboxes.append(block["bbox"])
                continue
            sample_len = min(len(target), 80)
            if sample_len >= 20 and target[:sample_len] in block_text:
                hit_bboxes.append(block["bbox"])
                continue
            if sample_len >= 20 and target[-sample_len:] in block_text:
                hit_bboxes.append(block["bbox"])

        if hit_bboxes:
            locations.append({
                "page_index": page_idx,
                "bbox": _union_bbox(hit_bboxes),
            })

    return locations


def annotate_leaf_locations(nodes, page_blocks, inherited_start=None, inherited_end=None):
    for node in nodes:
        start_page = node.get("start_page", inherited_start)
        end_page = node.get("end_page", inherited_end)
        if node.get("nodes"):
            annotate_leaf_locations(node["nodes"], page_blocks, start_page, end_page)
            continue

        locations = locate_text_positions(node.get("text", ""), page_blocks, start_page, end_page)
        if locations:
            node["page_locations"] = locations
            node["page_index"] = [loc["page_index"] for loc in locations]
            node["bbox"] = locations[0]["bbox"] if len(locations) == 1 else [loc["bbox"] for loc in locations]
            node["start_page"] = locations[0]["page_index"]
            node["end_page"] = locations[-1]["page_index"]


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
    return entries


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

            # 块状目录检测：标题行和页码行分离排布
            # 统计纯数字行（页码候选）和非数字非空行（标题候选）
            page_lines = text.split('\n')
            num_only_lines = [l.strip() for l in page_lines if re.match(r'^\d{1,4}$', l.strip())]
            title_like_lines = [l.strip() for l in page_lines
                                if l.strip() and not re.match(r'^\d{1,4}$', l.strip())
                                and len(l.strip()) >= 2]
            if len(num_only_lines) >= 3 and len(title_like_lines) >= 3:
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

    # 模式4: 块状目录——标题行和页码行分离排布
    # 触发条件：条目不足，或已有条目的页码不是严格递增的（说明前面的模式匹配错误）
    pages_monotonic = all(
        entries[j]['page'] < entries[j+1]['page']
        for j in range(len(entries)-1)
    ) if len(entries) >= 2 else True
    if len(entries) < 3 or not pages_monotonic:
        raw_titles = []
        pages = []
        # 需要跳过的行（页眉页脚、目录关键词）
        skip_patterns = re.compile(
            r'^(目\s*录|contents|table\s+of\s+contents)$', re.IGNORECASE
        )
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if re.match(r'^\d{1,4}$', line):
                pages.append(int(line))
            else:
                if skip_patterns.match(line):
                    continue
                if len(line) >= 2:
                    raw_titles.append(line)

        # 合并 "第X章" 类编号行与副标题行
        # 支持两种排列：
        #   (A) 交替排列: 第一章, 研究与开发, 第二章, 技术性能, ...
        #   (B) 分块排列: 第一章, 第二章, ..., 附录, 研究与开发, 技术性能, ...
        chapter_pat = re.compile(r'^第[一二三四五六七八九十百\d]+[章节篇部]$')
        standalone_pat = re.compile(r'^(附录|参考文献|致谢|前言|摘\s*要|Abstract|报告核心要点|获取公共数据)', re.IGNORECASE)

        # 分离: 编号行、独立标题行、剩余行（潜在副标题）
        chapter_lines = []  # ("第一章", index)
        standalone_lines = []  # ("附录", index)
        other_lines = []  # ("研究与开发", index)
        for idx, t in enumerate(raw_titles):
            if chapter_pat.match(t):
                chapter_lines.append((t, idx))
            elif standalone_pat.match(t):
                standalone_lines.append((t, idx))
            else:
                other_lines.append((t, idx))

        titles = []
        if chapter_lines and len(other_lines) >= len(chapter_lines):
            # 先尝试模式A（交替排列）：每个章节编号后面紧跟副标题
            adjacent_pairs = True
            for ch_text, ch_idx in chapter_lines:
                # 检查 raw_titles 中下一个元素是否是非编号行
                if ch_idx + 1 < len(raw_titles) and not chapter_pat.match(raw_titles[ch_idx + 1]):
                    pass
                else:
                    adjacent_pairs = False
                    break

            if adjacent_pairs:
                # 模式A：逐个合并
                used = set()
                for ch_text, ch_idx in chapter_lines:
                    sub = raw_titles[ch_idx + 1]
                    titles.append(f"{ch_text} {sub}")
                    used.add(ch_idx)
                    used.add(ch_idx + 1)
                # 加入独立标题
                for st_text, st_idx in standalone_lines:
                    if st_idx not in used:
                        titles.append(st_text)
            else:
                # 模式B（分块排列）：编号行在前，副标题行在后，一一配对
                subtitles = [t for t, _ in other_lines[:len(chapter_lines)]]
                remaining = [t for t, _ in other_lines[len(chapter_lines):]]

                # 构建合并后的标题，并按原始位置排序
                merged = []
                for (ch_text, ch_idx), sub in zip(chapter_lines, subtitles):
                    merged.append((ch_idx, f"{ch_text} {sub}"))
                for st_text, st_idx in standalone_lines:
                    merged.append((st_idx, st_text))
                for rem_text in remaining:
                    merged.append((999, rem_text))  # 放末尾
                merged.sort(key=lambda x: x[0])
                titles = [t for _, t in merged]
        else:
            # 没有章节编号，直接用 raw_titles（跳过常见页眉）
            # 尝试简单的紧邻合并
            i_t = 0
            while i_t < len(raw_titles):
                t = raw_titles[i_t]
                if chapter_pat.match(t) and i_t + 1 < len(raw_titles):
                    next_t = raw_titles[i_t + 1]
                    if not chapter_pat.match(next_t) and not standalone_pat.match(next_t):
                        titles.append(f"{t} {next_t}")
                        i_t += 2
                        continue
                titles.append(t)
                i_t += 1

        # 找到最长递增页码子序列，长度应等于标题数
        if len(titles) >= 3 and len(pages) >= 3:
            def find_best_page_sequence(pg_list, n):
                """从页码列表中找到长度为 n 的递增子序列"""
                if len(pg_list) >= n:
                    tail = pg_list[-n:]
                    if all(tail[j] < tail[j+1] for j in range(len(tail)-1)):
                        return tail
                best = []
                for s in range(len(pg_list)):
                    seq = [pg_list[s]]
                    for k in range(s+1, len(pg_list)):
                        if pg_list[k] > seq[-1]:
                            seq.append(pg_list[k])
                    if len(seq) > len(best):
                        best = seq
                return best[:n] if len(best) >= n else best

            matched_pages = find_best_page_sequence(pages, len(titles))
            # 如果页码数不够，尝试截短 titles 以匹配
            if len(matched_pages) < len(titles) and len(matched_pages) >= 3:
                titles = titles[:len(matched_pages)]
            if len(matched_pages) == len(titles):
                entries = []
                for title, page in zip(titles, matched_pages):
                    level = _guess_toc_level(title)
                    entries.append({'level': level, 'title': title, 'page': page})
                print(f"[目录] 使用块状目录解析模式，匹配 {len(entries)} 个条目")

    return entries


def _guess_toc_level(title):
    """根据标题格式猜测层级"""
    title = title.strip()
    # "第X章"/"第X篇"/"第X部" → level 1
    if re.match(r'^第[一二三四五六七八九十百\d]+[章篇部]', title):
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
            # 没有子节点，按固定页数强制拆分
            pages_per_sub = max(1, int(token_threshold * 1.5 / (len(chunk_text) / (chunk['end_page'] - chunk['start_page'] + 1))))
            pages_per_sub = max(5, min(pages_per_sub, 15))  # 限制在5-15页
            total_chunk_pages = chunk['end_page'] - chunk['start_page'] + 1
            if total_chunk_pages <= pages_per_sub:
                print(f"  [拆分] {chunk['title']} ({estimated_tokens} tokens) 无子节可拆，保持原样")
                refined.append(chunk)
                continue

            n_subs = (total_chunk_pages + pages_per_sub - 1) // pages_per_sub
            print(f"  [拆分] {chunk['title']} ({estimated_tokens} tokens) 无子节可拆，"
                  f"按每 {pages_per_sub} 页强制拆为 {n_subs} 个子块")
            for si in range(n_subs):
                sub_start = chunk['start_page'] + si * pages_per_sub
                sub_end = min(chunk['start_page'] + (si + 1) * pages_per_sub - 1, chunk['end_page'])
                refined.append({
                    'title': f"{chunk['title']} (第{si+1}部分)",
                    'level': chunk['level'],
                    'start_page': sub_start,
                    'end_page': sub_end,
                    '_parent_title': chunk['title'],
                })
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
def llm_call(prompt, max_tokens=8192, retries=2):
    """调用 LLM（含重试）"""
    headers = {"Content-Type": "application/json", "Authorization": "Bearer "}
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0
    }
    for attempt in range(retries + 1):
        try:
            resp = requests.post(LLM_URL, headers=headers, json=payload, timeout=900)
            if resp.status_code != 200:
                print(f"  [LLM错误] status={resp.status_code}, prompt长度={len(prompt)}, "
                      f"响应={resp.text[:500]}", flush=True)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < retries:
                wait = 10 * (attempt + 1)
                print(f"  [LLM重试] {type(e).__name__}, {wait}s 后重试 ({attempt+1}/{retries})", flush=True)
                _time.sleep(wait)
            else:
                raise


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
8. 原始文章的目录进行删除。对原文语言有偏差的内容可以进行删除

PDF 提取文本：
{chunk_text}
"""
    # 动态调整 max_tokens，确保 prompt + max_tokens <= 模型上下文长度
    prompt_tokens = int(len(prompt) / 1.5)
    model_ctx = 32768
    available = model_ctx - prompt_tokens - 100  # 留 100 token 余量
    mt = max(2048, min(8192, available))
    md_content = llm_call(prompt, max_tokens=mt)
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
    if not toc_entries or len(toc_entries) < 3:
        if toc_entries:
            print(f"[目录] 书签目录仅 {len(toc_entries)} 个条目，尝试从文本中检测更完整的目录")
        text_toc = detect_toc_from_text(cleaned_pages, scan_pages=toc_scan_pages)
        if text_toc and len(text_toc) > len(toc_entries or []):
            toc_entries = text_toc

    # Step 3: 分块
    print("\n" + "=" * 60)
    print("[Step 3] 按目录分块")
    print("=" * 60)
    if toc_entries:
        chunks = build_chunks_from_toc(toc_entries, len(cleaned_pages))
    else:
        chunks = split_by_page_count(cleaned_pages, pages_per_chunk=pages_per_chunk)

    # Step 3.5: 对超大章节按子节（level 2+）重新拆分
    print("\n" + "=" * 60)
    print("[Step 3.5] 检查章节 token 数，超过 30000 的按小节拆分")
    print("=" * 60)
    chunks = refine_chunks_by_token(chunks, toc_entries, cleaned_pages)

    # Step 3.6: 丢弃参考文献/附录及其后的所有内容
    _DISCARD_PATTERNS = re.compile(
        r'^(参考文献|参考资料|references?|bibliography|附录|appendix|appendices)',
        re.IGNORECASE
    )
    truncated_chunks = []
    for chunk in chunks:
        if _DISCARD_PATTERNS.search(chunk['title'].strip()):
            print(f"\n[截断] 检测到 '{chunk['title']}'，丢弃该章节及后续所有内容")
            break
        truncated_chunks.append(chunk)
    if len(truncated_chunks) < len(chunks):
        print(f"  保留 {len(truncated_chunks)}/{len(chunks)} 个分块")
    chunks = truncated_chunks

    # Step 4: 逐块处理（动态并发，按字符容量池控制）
    CHAR_CAPACITY = 1000000  # 全局字符容量池
    print("\n" + "=" * 60)
    print(f"[Step 4] 逐块调用 LLM 生成 Markdown 并构建子树（字符容量池={CHAR_CAPACITY}）")
    print("=" * 60)

    # 预处理：提取每个分块的文本
    chunk_texts = []
    for i, chunk in enumerate(chunks):
        chunk_text = pages_to_chunk_text(cleaned_pages, chunk['start_page'], chunk['end_page'])
        chunk_texts.append(chunk_text)

    # 动态并发控制：字符容量池
    _cap_lock = threading.Lock()
    _cap_cond = threading.Condition(_cap_lock)
    _used_chars = [0]          # 当前占用的字符数
    _peak_concurrent = [0]     # 峰值并发数
    _running_count = [0]       # 当前并发任务数

    def _acquire_capacity(char_count):
        """获取字符容量，不足时阻塞等待"""
        with _cap_cond:
            while _used_chars[0] + char_count > CHAR_CAPACITY:
                _cap_cond.wait()
            _used_chars[0] += char_count
            _running_count[0] += 1
            if _running_count[0] > _peak_concurrent[0]:
                _peak_concurrent[0] = _running_count[0]
            print(f"  [调度] 占用 {char_count} 字符, 当前已用 {_used_chars[0]}/{CHAR_CAPACITY}, 并发 {_running_count[0]}")

    def _release_capacity(char_count):
        """释放字符容量，通知等待线程"""
        with _cap_cond:
            _used_chars[0] -= char_count
            _running_count[0] -= 1
            print(f"  [释放] 归还 {char_count} 字符, 当前已用 {_used_chars[0]}/{CHAR_CAPACITY}, 并发 {_running_count[0]}")
            _cap_cond.notify_all()

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

        # 获取容量（可能阻塞）
        _acquire_capacity(char_count)
        try:
            md_content = chunk_text_to_markdown(chunk['title'], chunk_text)
            subtree = build_subtree_from_markdown(md_content)
            print(f"  [完成] 分块 {idx+1} MD {len(md_content)} 字符, 子树 {len(subtree)} 个根节点")
            return idx, md_content, subtree
        finally:
            _release_capacity(char_count)

    # 并行执行（max_workers 设大，实际并发由容量池控制）
    step4_start = _time.time()
    results = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = {executor.submit(_process_one_chunk, i): i for i in range(len(chunks))}
        for future in as_completed(futures):
            idx, md_content, subtree = future.result()
            results[idx] = (md_content, subtree)
    step4_elapsed = _time.time() - step4_start
    print(f"\n[Step 4 统计] LLM 总耗时: {step4_elapsed:.1f}s, 峰值并发: {_peak_concurrent[0]}")

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

    # Step 5.5: 对叶子节点按段落进一步拆分
    print("\n" + "=" * 60)
    print("[Step 5.5] 对叶子节点按段落拆分")
    print("=" * 60)

    def split_leaf_by_paragraphs(nodes):
        """递归处理：将叶子节点的 text 按段落拆分为子节点"""
        split_count = 0
        for node in nodes:
            if node.get('nodes'):
                # 非叶子节点，递归处理子节点
                split_count += split_leaf_by_paragraphs(node['nodes'])
                continue

            text = (node.get('text') or '').strip()
            if not text:
                continue

            # 按空行分段
            paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
            if len(paragraphs) <= 1:
                continue

            # 第一段可能包含标题行+正文（单换行连接），需分离
            first_para = paragraphs[0]
            header_match = re.match(r'^(#{1,6}\s+.+?)(?:\n(.+))?$', first_para, re.DOTALL)
            if header_match:
                header_line = header_match.group(1).strip()
                body_after_header = (header_match.group(2) or '').strip()
                # 标题行留在父节点
                node['text'] = header_line
                child_paragraphs = []
                # 标题行后面紧跟的正文作为第一个段落子节点
                if body_after_header:
                    child_paragraphs.append(body_after_header)
                child_paragraphs.extend(paragraphs[1:])
            else:
                # 没有标题行，所有段落都拆为子节点
                node['text'] = ''
                child_paragraphs = paragraphs

            if not child_paragraphs:
                continue

            # 创建段落子节点
            node['nodes'] = []
            for j, para in enumerate(child_paragraphs, 1):
                para_node = {
                    'title': f"{node['title']} - 段落{j}",
                    'text': para,
                    'start_page': node.get('start_page'),
                    'end_page': node.get('end_page'),
                }
                node['nodes'].append(para_node)
            split_count += 1

        return split_count

    n_split = split_leaf_by_paragraphs(merged_tree)
    print(f"  拆分了 {n_split} 个叶子节点")

    assign_node_ids(merged_tree)

    page_blocks = extract_page_text_blocks(pdf_path)
    annotate_leaf_locations(merged_tree, page_blocks)

    # 可选：生成摘要（只对每一章生成一个总结，不逐叶子节点总结）
    if if_summary:
        from rag.utils import count_tokens
        SUMMARY_CHAR_CAPACITY = 45000

        print(f"\n[摘要] 正在为各章节并行生成摘要（字符容量池={SUMMARY_CHAR_CAPACITY}）...")
        formatted = format_structure(merged_tree, order=['title', 'node_id', 'start_page', 'end_page', 'page_index', 'bbox', 'page_locations', 'summary', 'prefix_summary', 'text', 'nodes'])

        def _collect_all_text(node):
            """递归收集一个节点及其所有子节点的文本"""
            parts = []
            t = (node.get('text') or '').strip()
            if t:
                parts.append(t)
            for child in (node.get('nodes') or []):
                parts.extend(_collect_all_text(child))
            return parts

        def _get_chapter_summary_sync(chapter_title, chapter_text):
            """同步调用 LLM 为一章生成摘要"""
            prompt = f"""你获得了文档某一章的内容，你的任务是生成该章的描述，说明该章涵盖的主要内容。

    章节标题: {chapter_title}
    章节文本: {chapter_text}
    
    直接返回描述，不要包含任何其他文本。
    """
            return llm_call(prompt, max_tokens=4096)

        # 获取顶级章节节点（formatted 是列表）
        top_chapters = formatted if isinstance(formatted, list) else [formatted]

        # 构建章节级任务：每章收集全部文本后生成一个摘要
        chapter_tasks = []  # (index, node, full_text, char_count)
        for i, chapter in enumerate(top_chapters):
            all_text_parts = _collect_all_text(chapter)
            full_text = "\n".join(all_text_parts)
            if not full_text.strip():
                continue
            # 截断过长文本以适应 LLM 上下文
            if len(full_text) > 15000:
                full_text = full_text[:15000]
            chapter_tasks.append((i, chapter, full_text, len(full_text)))

        print(f"  共 {len(top_chapters)} 个顶级章节, {len(chapter_tasks)} 个需要生成摘要")

        # 并行处理各章节摘要（字符容量池控制）
        if chapter_tasks:
            _sum_lock = threading.Lock()
            _sum_cond = threading.Condition(_sum_lock)
            _sum_used = [0]
            _sum_peak = [0]
            _sum_running = [0]

            def _acquire_sum_cap(cc):
                with _sum_cond:
                    while _sum_used[0] + cc > SUMMARY_CHAR_CAPACITY:
                        _sum_cond.wait()
                    _sum_used[0] += cc
                    _sum_running[0] += 1
                    if _sum_running[0] > _sum_peak[0]:
                        _sum_peak[0] = _sum_running[0]

            def _release_sum_cap(cc):
                with _sum_cond:
                    _sum_used[0] -= cc
                    _sum_running[0] -= 1
                    _sum_cond.notify_all()

            def _summarize_chapter(task_tuple):
                idx, chapter, text, cc = task_tuple
                _acquire_sum_cap(cc)
                try:
                    title = chapter.get('title', '')
                    summary = _get_chapter_summary_sync(title, text)
                    return idx, chapter, summary
                finally:
                    _release_sum_cap(cc)

            summary_start = _time.time()
            with ThreadPoolExecutor(max_workers=len(chapter_tasks)) as executor:
                futures = {executor.submit(_summarize_chapter, t): t for t in chapter_tasks}
                for future in as_completed(futures):
                    idx, chapter, summary = future.result()
                    # 摘要挂在顶级章节的 prefix_summary 上
                    chapter['prefix_summary'] = summary
            summary_elapsed = _time.time() - summary_start
            print(f"  [摘要统计] 耗时: {summary_elapsed:.1f}s, 峰值并发: {_sum_peak[0]}")

        if not if_add_node_text:
            formatted = format_structure(formatted, order=['title', 'node_id', 'start_page', 'end_page', 'page_index', 'bbox', 'page_locations', 'summary', 'prefix_summary', 'nodes'])
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
