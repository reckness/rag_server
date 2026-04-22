"""
Markdown 文档解析器
基于 # 标题符识别层级，复用 pageindex/page_index_md.py 的核心逻辑
"""
import os
import re
from typing import Dict, Any, List
from .base_parser import BaseDocumentParser


class MarkdownParser(BaseDocumentParser):
    """Markdown 文档解析器（.md）"""

    def parse(self, file_path: str, doc_name: str = None) -> Dict[str, Any]:
        if doc_name is None:
            doc_name = os.path.splitext(os.path.basename(file_path))[0]
        content = self._read_file(file_path)

        if not content.strip():
            return {
                "doc_name": doc_name,
                "structure": [self._make_node(doc_name, "", "0001")],
            }

        raw_sections = self._extract_sections(content)

        if not raw_sections:
            # 无标题，整体作为一个节点
            return {
                "doc_name": doc_name,
                "structure": [self._make_node(doc_name, content.strip(), "0001")],
            }

        structure = self._build_tree(raw_sections)
        return {"doc_name": doc_name, "structure": structure}

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _read_file(file_path: str) -> str:
        for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030", "latin-1"):
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, LookupError):
                continue
        raise ValueError(f"无法以任何已知编码读取文件: {file_path}")

    def _extract_sections(self, content: str) -> List[Dict]:
        """
        扫描 Markdown 文本，提取标题及对应的正文内容。
        正确处理代码块（``` ）内的 # 符号，不将其误识别为标题。
        """
        header_re = re.compile(r"^(#{1,6})\s+(.+)$")
        code_fence_re = re.compile(r"^```")

        lines = content.split("\n")
        sections: List[Dict] = []
        node_counter = [0]

        def next_id():
            node_counter[0] += 1
            return str(node_counter[0]).zfill(4)

        in_code_block = False
        current: Dict = None

        for line in lines:
            stripped = line.rstrip()

            # 切换代码块状态
            if code_fence_re.match(stripped):
                in_code_block = not in_code_block
                if current is not None:
                    current["text_parts"].append(line)
                continue

            if not in_code_block:
                match = header_re.match(stripped)
                if match:
                    if current is not None:
                        sections.append(current)
                    level = len(match.group(1))
                    title = match.group(2).strip()
                    current = {
                        "title": title,
                        "level": level,
                        "text_parts": [],
                        "node_id": next_id(),
                    }
                    continue

            # 普通内容行
            if current is not None:
                current["text_parts"].append(line)
            # 标题前的内容忽略（或可作为 "前言" 节点，这里从简）

        if current is not None:
            sections.append(current)

        return sections

    def _build_tree(self, sections: List[Dict]) -> List[Dict]:
        """将扁平章节列表按 level 构建树形结构"""
        root: List[Dict] = []
        stack: List[tuple] = []  # (level, node)

        for idx, sec in enumerate(sections):
            level = sec["level"]
            text = "\n".join(sec["text_parts"]).strip()
            node = {
                "title": sec["title"],
                "node_id": sec["node_id"],
                "text": text,
                "start_index": idx + 1,
                "end_index": idx + 1,
                "nodes": [],
            }

            while stack and stack[-1][0] >= level:
                stack.pop()

            if not stack:
                root.append(node)
            else:
                stack[-1][1]["nodes"].append(node)

            stack.append((level, node))

        self._clean_empty_nodes(root)
        return root

    def _clean_empty_nodes(self, nodes: List[Dict]):
        for node in nodes:
            if "nodes" in node:
                if not node["nodes"]:
                    del node["nodes"]
                else:
                    self._clean_empty_nodes(node["nodes"])
