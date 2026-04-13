"""
PDF 简单解析器（无目录 PDF）
使用 PyMuPDF（fitz）按页提取文本，然后按固定 token 大小分块，
每个 chunk 对应一个节点，适合无目录、扫描件等 PDF。
"""
import os
from typing import Dict, Any, List
from .base_parser import BaseDocumentParser


class PdfSimpleParser(BaseDocumentParser):
    """
    PDF 简单解析器：适用于没有目录的 PDF。
    策略：逐页提取文本 → 按字符数分块 → 每块作为一个节点。
    若需要 LLM 生成目录，请使用 rag.page_index.process_no_toc。
    """

    # 每个节点的目标字符数（约 ~500 汉字 / ~1000 英文词）
    CHUNK_SIZE = 1500
    # 相邻块重叠字符数
    OVERLAP = 150

    def parse(self, file_path: str, doc_name: str = None) -> Dict[str, Any]:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("请安装 PyMuPDF: pip install pymupdf")

        if doc_name is None:
            doc_name = os.path.splitext(os.path.basename(file_path))[0]
        doc = fitz.open(file_path)

        # 1. 提取全文（保留页码信息）
        pages_text: List[str] = []
        for page in doc:
            text = page.get_text("text").strip()
            pages_text.append(text)
        doc.close()

        # 2. 按分块策略切割
        full_text = "\n".join(pages_text)
        chunks = self._chunk_text(full_text)

        # 3. 构造节点列表
        structure: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunks):
            # 取前 30 个字符作为标题预览
            title_preview = chunk[:30].replace("\n", " ").strip()
            title = f"第 {idx + 1} 段：{title_preview}..."

            node = self._make_node(
                title=title,
                text=chunk,
                node_id=str(idx + 1).zfill(4),
                start_index=idx + 1,
                end_index=idx + 1,
            )
            structure.append(node)

        if not structure:
            structure = [self._make_node(
                title=doc_name,
                text="（文档无可提取的文本内容）",
                node_id="0001",
            )]

        return {"doc_name": doc_name, "structure": structure}

    # ------------------------------------------------------------------

    def _chunk_text(self, text: str) -> List[str]:
        """将长文本按 CHUNK_SIZE 分块，相邻块有 OVERLAP 字符重叠"""
        if not text.strip():
            return []

        chunks: List[str] = []
        start = 0
        length = len(text)

        while start < length:
            end = min(start + self.CHUNK_SIZE, length)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            # 下一块从 (end - OVERLAP) 开始，确保有重叠
            start = max(end - self.OVERLAP, end) if end >= length else end - self.OVERLAP
            if start <= 0 or start >= end:
                # 防止死循环
                start = end

        return chunks
