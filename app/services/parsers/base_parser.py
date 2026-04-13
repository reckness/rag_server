"""
文档解析器基类
定义统一的解析接口，所有格式的解析器都继承此类
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseDocumentParser(ABC):
    """文档解析器基类"""

    @abstractmethod
    def parse(self, file_path: str, doc_name: str = None) -> Dict[str, Any]:
        """
        解析文档，返回统一格式的 page_index JSON 结构

        参数
        ----
        file_path: str - 文件路径
        doc_name: str - 文档名称，默认为 None，此时会从文件路径中提取

        返回格式：
        {
            "doc_name": "文档名称",
            "structure": [
                {
                    "title": "章节标题",
                    "node_id": "0001",
                    "text": "章节内容",
                    "start_index": 1,
                    "end_index": 2,
                    "nodes": [...]  # 子节点
                },
                ...
            ]
        }
        """
        raise NotImplementedError

    @staticmethod
    def _make_node(title: str, text: str, node_id: str,
                   start_index: int = 1, end_index: int = 1,
                   children: List[Dict] = None) -> Dict[str, Any]:
        """创建标准节点"""
        node = {
            "title": title,
            "node_id": node_id,
            "text": text,
            "start_index": start_index,
            "end_index": end_index,
        }
        if children:
            node["nodes"] = children
        return node
