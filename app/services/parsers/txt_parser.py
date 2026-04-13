"""
纯文本文档解析器
支持 .txt 格式，按固定块大小切分
"""
import os
from typing import Dict, Any
from .base_parser import BaseDocumentParser


class TxtParser(BaseDocumentParser):
    """纯文本文档解析器（.txt）"""

    # 每个块的最大字符数（约 500 字）
    CHUNK_SIZE = 500
    # 块之间的重叠字符数
    OVERLAP = 50

    def parse(self, file_path: str, doc_name: str = None) -> Dict[str, Any]:
        if doc_name is None:
            doc_name = os.path.splitext(os.path.basename(file_path))[0]

        # 尝试多种编码读取
        text = None
        for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030", "latin-1"):
            try:
                with open(file_path, "r", encoding=enc) as f:
                    text = f.read()
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if text is None:
            raise ValueError(f"无法读取文件: {file_path}")

        # 按换行分割成段落
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

        if not paragraphs:
            return {
                "doc_name": doc_name,
                "structure": [{
                    "title": doc_name,
                    "node_id": "0001",
                    "text": "",
                    "start_index": 1,
                    "end_index": 1,
                }]
            }

        # 将段落合并成固定大小的块
        chunks = []
        current_chunk = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para)
            if current_len + para_len > self.CHUNK_SIZE and current_chunk:
                chunks.append("\n".join(current_chunk))
                # 保留最后一段作为重叠
                current_chunk = current_chunk[-1:] if self.OVERLAP > 0 else []
                current_len = len(current_chunk[0]) if current_chunk else 0

            current_chunk.append(para)
            current_len += para_len

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        # 构建结构
        structure = []
        for idx, chunk in enumerate(chunks):
            # 用第一行作为标题
            first_line = chunk.split("\n")[0][:50]
            title = first_line if first_line else f"片段 {idx + 1}"
            structure.append({
                "title": title,
                "node_id": str(idx + 1).zfill(4),
                "text": chunk,
                "start_index": idx + 1,
                "end_index": idx + 1,
            })

        return {
            "doc_name": doc_name,
            "structure": structure,
        }
