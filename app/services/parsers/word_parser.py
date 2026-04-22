"""
Word 文档解析器
支持 .doc 和 .docx 格式，按标题层级切分章节
"""
import os
from typing import Dict, Any, List
from .base_parser import BaseDocumentParser


class WordParser(BaseDocumentParser):
    """Word 文档解析器（.docx / .doc）"""

    # Word 内置标题样式名称
    HEADING_STYLES = {
        "heading 1": 1, "heading 2": 2, "heading 3": 3,
        "heading 4": 4, "heading 5": 5, "heading 6": 6,
        # 中文版 Word
        "标题 1": 1, "标题 2": 2, "标题 3": 3,
        "标题 4": 4, "标题 5": 5, "标题 6": 6,
        "标题1": 1, "标题2": 2, "标题3": 3,
    }

    def parse(self, file_path: str, doc_name: str = None) -> Dict[str, Any]:
        try:
            from docx import Document
        except ImportError:
            raise ImportError("请安装 python-docx: pip install python-docx")

        doc = Document(file_path)
        if doc_name is None:
            doc_name = os.path.splitext(os.path.basename(file_path))[0]

        # 收集段落及其级别
        flat_nodes = []
        node_counter = [0]

        def next_id():
            node_counter[0] += 1
            return str(node_counter[0]).zfill(4)

        current_sections: Dict[int, Dict] = {}  # level -> node
        pending_text_parts: List[str] = []

        def flush_text(level: int) -> str:
            """将 pending_text_parts 中低于等于 level 的文本拼接"""
            return "\n".join(pending_text_parts)

        paragraphs = list(doc.paragraphs)

        # 先将段落按节点划分
        raw_sections = []
        current: Dict[str, Any] = None

        for para in paragraphs:
            style_name = para.style.name.lower() if para.style else ""
            level = self.HEADING_STYLES.get(para.style.name, None) if para.style else None

            if level is not None and para.text.strip():
                # 这是一个标题段落
                if current is not None:
                    raw_sections.append(current)
                current = {
                    "title": para.text.strip(),
                    "level": level,
                    "text_parts": [],
                    "node_id": next_id(),
                }
            else:
                # 普通内容段落
                text = para.text.strip()
                if text:
                    if current is not None:
                        current["text_parts"].append(text)
                    else:
                        # 文档开头还没有遇到标题，创建一个"前言"节点
                        if not raw_sections and current is None:
                            current = {
                                "title": "前言",
                                "level": 1,
                                "text_parts": [text],
                                "node_id": next_id(),
                            }
                        elif raw_sections and current is None:
                            raw_sections[-1]["text_parts"].append(text)

        if current is not None:
            raw_sections.append(current)

        # 如果没有找到任何标题，把所有文本作为一个节点
        if not raw_sections:
            all_text = "\n".join(
                p.text.strip() for p in paragraphs if p.text.strip()
            )
            raw_sections = [{
                "title": doc_name,
                "level": 1,
                "text_parts": [all_text],
                "node_id": "0001",
            }]

        # 构建树形结构
        structure = self._build_tree(raw_sections)

        return {
            "doc_name": doc_name,
            "structure": structure,
        }

    def _build_tree(self, raw_sections: List[Dict]) -> List[Dict]:
        """将扁平的章节列表按 level 构建为树形结构"""
        if not raw_sections:
            return []

        root = []
        stack = []  # (level, node)

        for idx, sec in enumerate(raw_sections):
            level = sec["level"]
            text = "\n".join(sec["text_parts"])
            node = {
                "title": sec["title"],
                "node_id": sec["node_id"],
                "text": text,
                "start_index": idx + 1,
                "end_index": idx + 1,
                "nodes": [],
            }

            # 弹出栈中所有 level >= 当前 level 的节点
            while stack and stack[-1][0] >= level:
                stack.pop()

            if not stack:
                root.append(node)
            else:
                stack[-1][1]["nodes"].append(node)

            stack.append((level, node))

        # 清理空 nodes 列表
        self._clean_empty_nodes(root)
        return root

    def _clean_empty_nodes(self, nodes: List[Dict]):
        for node in nodes:
            if "nodes" in node:
                if not node["nodes"]:
                    del node["nodes"]
                else:
                    self._clean_empty_nodes(node["nodes"])
