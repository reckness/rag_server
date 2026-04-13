"""
文档解析器模块
支持多种文件格式的解析，统一输出为 page_index 兼容的 JSON 结构
"""
from .base_parser import BaseDocumentParser
from .word_parser import WordParser
from .txt_parser import TxtParser
from .ppt_parser import PptParser
from .excel_parser import ExcelParser
from .md_parser import MarkdownParser
from .pdf_simple_parser import PdfSimpleParser

__all__ = [
    "BaseDocumentParser",
    "WordParser",
    "TxtParser",
    "PptParser",
    "ExcelParser",
    "MarkdownParser",
    "PdfSimpleParser",
]
