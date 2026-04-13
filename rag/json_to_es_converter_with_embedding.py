#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tree-to-Flat JSON Converter for Elasticsearch with Position Tracking and Embedding
将树状 PageIndex JSON 转换为向量数据库所需的扁平化格式并存储到 Elasticsearch
添加了 page_num_int 字段用于文档位置跟踪
"""

import json
import os
import sys
import argparse
import time
import re
from collections import Counter
from typing import Dict, List, Any
from elasticsearch.helpers import bulk
import requests
import urllib3

# 添加项目根目录到路径，以便导入 common 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从配置文件导入配置
from common.config import ELASTICSEARCH_INDEX, ELASTICSEARCH_HOST

# 导入 Elasticsearch 连接池和基础类
from common.doc_store.es_conn_pool import ES_CONN

# 导入嵌入客户端
from common.nlp.embedding_client import get_embedding


# ==================== 文本清洗工具函数 ====================

def extract_lines_per_page(nodes):
    """按页收集文本"""
    page_lines = []
    for node in nodes:
        text = node.get("text", "")
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        page_lines.append(lines)
    return page_lines


def find_repeated_lines(page_lines, threshold=0.6):
    """
    出现超过60%页的行 → 页眉/页脚
    """
    total_pages = len(page_lines)
    # 当页数小于等于1时，不进行页眉/页脚检测
    if total_pages <= 1:
        return set()
    counter = Counter()

    for lines in page_lines:
        unique_lines = set(lines)
        for l in unique_lines:
            counter[l] += 1

    noise_lines = set()
    for line, count in counter.items():
        if count / total_pages >= threshold:
            noise_lines.add(line)

    return noise_lines


def clean_page_lines(lines, noise_lines):
    """清洗文本，去掉页眉页脚"""
    return [l for l in lines if l not in noise_lines]

def clean_text(text):
    """清洗文本"""
    if "【原始数据内容】:" in text:
        text = text.split("【原始数据内容】:")[-1]
    return text.replace("\n", " ").strip()


def truncate(text, max_length):
    """截断文本到指定长度"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


# ==========================================
# 辅助类与核心逻辑
# ==========================================

class ESConverter:
    def __init__(self, input_path: str, index_name: str, doc_id: str = "", kb_id: str = "", fd_id: str = "", doc_title: str = ""):
        self.input_path = input_path
        self.index_name = index_name
        self.doc_id = doc_id
        self.kb_id = kb_id
        self.fd_id = fd_id
        self.doc_title = doc_title
        self.flat_nodes = []
        self.processed_count = 0
        self.current_position = 0
        self.noise_lines = set()  # 存储检测到的页眉页脚
        # 如果没有提供文档标题，则从输入文件路径中提取
        if not self.doc_title:
            # 从输入文件路径中提取文件名作为文档标题
            base_name = os.path.basename(input_path)
            # 移除文件扩展名
            self.doc_title = os.path.splitext(base_name)[0]
        # 初始化 Elasticsearch 客户端
        self.es_client = None
    def log(self, msg: str, level: str = "INFO"):
        """标准日志输出"""
        print(f"[{level}] {msg}", flush=True)



    def _determine_section_hint(self, title: str, content: str) -> str:
        """
        智能推断章节类型
        """
        title_lower = title.lower() if title else ""
        content_lower = content.lower() if content else ""

        # 基于标题的结构判断
        if any(x in title_lower for x in ["content", "目录", "index", "table of contents"]):
            return "目录列表"
        elif any(x in title_lower for x in ["intro", "引言", "概述", "summary", "abstract"]):
            return "引言/概述"
        elif any(x in title_lower for x in ["conclusion", "结论", "结语"]):
            return "结论"
        elif any(x in title_lower for x in ["glossary", "术语", "definition"]):
            return "术语表/定义"
        elif any(x in title_lower for x in ["appendix", "appendices", "附录"]):
            return "附录"
        elif any(x in title_lower for x in ["reference", "参考"]):
            return "参考文献"
        elif any(x in title_lower for x in ["disclaimer", "声明", "legal", "copyright"]):
            return "法律声明"

        # 基于内容的特征判断
        if "table" in content_lower and ("row" in content_lower or "col" in content_lower):
            return "数据表格"
        elif any(x in content_lower for x in ["warning", "caution", "danger", "警告", "注意", "危险"]):
            return "安全警告"
        elif any(x in content_lower for x in ["step 1", "step 2", "步骤", "procedure", "流程"]):
            return "操作流程"
        elif any(x in content_lower for x in ["spec", "specification", "参数", "规格"]):
            return "技术规格"

        return "正文章节"

    def _generate_embedding_text(self, path_list: List[str], content: str, hint: str, node: Dict) -> str:
        """
        构造用于向量化的文本
        """
        # 组合最终的语义文本，使用优化版本
        embedding_text = f"""
        标题: {path_list[-1] if path_list else '根节点'}
        章节: {' > '.join(path_list)}
        类型: {hint}
        内容: {truncate(content, 500)}
        """
        return embedding_text

    def _recursive_walk(self, nodes: List[Dict], path: List[str], depth: int):
        """递归遍历树形结构并计算位置信息"""
        for node in nodes:
            # 1. 获取基础信息
            current_title = node.get("title", "Untitled").replace('\n', ' ').strip()
            current_path = path + [current_title]
            
            # 获取内容，优先用 text，其次用 content，最后为空
            original_text = node.get("text", node.get("content", "")).strip()
            
            # 清洗文本，去除页眉页脚
            if original_text and self.noise_lines:
                lines = [l.strip() for l in original_text.split("\n") if l.strip()]
                cleaned_lines = clean_page_lines(lines, self.noise_lines)
                original_text = "\n".join(cleaned_lines)
            
            # 2. 计算 page_num_int 数组
            # 优先使用输入 JSON 中的 start_index 和 end_index 字段
            start_index = node.get("start_index")
            end_index = node.get("end_index") + 1
            # 生成 page_num_int 数组
            page_num_int = list(range(start_index, end_index))
            # 更新当前位置
            self.current_position = end_index
            
            # 3. 处理 Node ID (优先用原有的，没有则生成)
            node_id = node.get("node_id", f"gen_id_{self.processed_count:05d}")

            # 4. 生成智能标签 (Hint)
            section_hint = self._determine_section_hint(current_title, original_text)

            # 5. 生成语义化 Embedding 文本
            display_text = original_text if original_text else "(无正文内容，仅作为章节标题存在)"
            # 进一步清洗文本，确保嵌入文本的质量
            cleaned_display_text = clean_text(display_text)
            embedding_text = self._generate_embedding_text(current_path, cleaned_display_text, section_hint, node)

            # 6. 生成向量嵌入
            embedding = get_embedding(embedding_text)
            if not embedding:
                # 如果获取嵌入失败，使用零向量作为 fallback
                from common.config import EMBEDDING_DIM
                embedding = [0.0] * EMBEDDING_DIM

            # 检查是否为叶子节点（无子节点）
            is_leaf = not ("nodes" in node and isinstance(node["nodes"], list) and len(node["nodes"]) > 0)
            
            # 只将叶子节点添加到扁平列表
            if is_leaf:
                # 7. 构建最终数据对象
                rag_item = {
                    "doc_id": self.doc_id,
                    "kb_id": self.kb_id,
                    "fd_id": self.fd_id,
                    "doc_title": self.doc_title,
                    "embedding_text": embedding_text,
                   "page_num_int": page_num_int,
                    "section_hint": section_hint,
                    "section_id": node_id,
                    "section_path": current_path,
                    "original_snippet": original_text,
                    "embedding": embedding,
                }

                self.flat_nodes.append(rag_item)
                self.processed_count += 1

            # 8. 递归处理子节点
            if "nodes" in node and isinstance(node["nodes"], list):
                self._recursive_walk(node["nodes"], current_path, depth + 1)

    def _init_elasticsearch(self):
        """初始化 Elasticsearch 客户端"""
        try:
            # 使用连接池获取 Elasticsearch 客户端
            self.es_client = ES_CONN.get_conn()
            # 测试连接
            info = self.es_client.info()
            self.log(f"Connected to Elasticsearch: {ELASTICSEARCH_HOST}")
            return True
        except Exception as e:
            self.log(f"Failed to connect to Elasticsearch: {str(e)}", "ERROR")
            import traceback
            traceback.print_exc()
            return False

    def _create_index(self):
        """创建 Elasticsearch 索引"""
        try:
            from common.config import EMBEDDING_DIM
            if not self.es_client.indices.exists(index=self.index_name):
                mapping = {
                    "mappings": {
                        "properties": {
                            "doc_id": {
                                "type": "keyword"
                            },
                            "kb_id": {
                                "type": "keyword"
                            },
                            "fd_id": {
                                "type": "keyword"
                            },
                            "embedding_text": {
                                "type": "text"
                            },
                            "section_hint": {
                                "type": "keyword"
                            },
                            "page_num_int": {
                                "type": "integer"
                            },
                            "doc_title": {
                                "type": "keyword"
                            },
                            "section_id": {
                                "type": "keyword"
                            },
                            "section_path": {
                                "type": "text"
                            },
                            "original_snippet": {
                                "type": "text"
                            },
                           "embedding": {
                                "type": "dense_vector",
                                "dims": EMBEDDING_DIM,
                                "index": True,
                                "similarity": "cosine"
                            }
                        }
                    }
                }
                self.es_client.indices.create(index=self.index_name, body=mapping)
                self.log(f"Created index: {self.index_name}")
            else:
                self.log(f"Index {self.index_name} already exists")
            return True
        except Exception as e:
            self.log(f"Failed to create index: {str(e)}", "ERROR")
            return False

    def _bulk_index(self):
        """批量索引数据到 Elasticsearch"""
        try:
            actions = [
                {
                    "_index": self.index_name,
                    "_source": item
                }
                for item in self.flat_nodes
            ]
            success, failed = bulk(self.es_client, actions)
            self.log(f"Bulk indexing completed: {success} succeeded, {failed} failed")
            return True
        except Exception as e:
            self.log(f"Failed to bulk index: {str(e)}", "ERROR")
            return False

    def run(self):
        """执行转换主流程"""
        from common.config import MODEL_NAME, EMBEDDING_DIM
        self.log(f"Starting Conversion: {self.input_path}")
        self.log(f"Target Elasticsearch: {ELASTICSEARCH_HOST}/{self.index_name}")
        self.log(f"Using embedding model: {MODEL_NAME} (dimensions: {EMBEDDING_DIM})")

        if not os.path.exists(self.input_path):
            self.log(f"Input file not found: {self.input_path}", "ERROR")
            return False

        # 初始化 Elasticsearch 客户端
        if not self._init_elasticsearch():
            return False

        # 创建索引
        if not self._create_index():
            return False

        try:
            # 读取文件 (utf-8-sig 兼容 Windows BOM)
            with open(self.input_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)

            # 解析根结构
            root_nodes = []
            if isinstance(data, list):
                root_nodes = data
                # 提取所有节点的文本，检测重复行（页眉页脚）
                nodes_with_text = [item for item in data if item.get("text", "")]
                page_lines = extract_lines_per_page(nodes_with_text)
                if page_lines:
                    self.noise_lines = find_repeated_lines(page_lines)
                    self.log(f"Detected {len(self.noise_lines)} noise lines (headers/footers)")
            elif isinstance(data, dict):
                # 尝试获取文档标题，优先使用构造函数传入的标题，否则使用 JSON 中的标题
                json_title = data.get("doc_name", data.get("title", ""))
                if not self.doc_title and json_title:
                    self.doc_title = json_title
                # 兼容不同的子节点键名 (structure 或 nodes)
                root_nodes = data.get("structure", data.get("nodes", []))
                
                # 提取所有节点的文本，检测重复行（页眉页脚）
                def extract_all_text(structure):
                    result = []
                    for node in structure:
                        if node.get("text", ""):
                            result.append(node)
                        if "nodes" in node:
                            result.extend(extract_all_text(node["nodes"]))
                    return result
                all_nodes = extract_all_text(root_nodes)
                page_lines = extract_lines_per_page(all_nodes)
                if page_lines:
                    self.noise_lines = find_repeated_lines(page_lines)
                    self.log(f"Detected {len(self.noise_lines)} noise lines (headers/footers)")

            self.log(f"Document identified: {self.doc_title}")
            
            # 开始递归
            self._recursive_walk(root_nodes, [], 1)

            # 批量索引到 Elasticsearch
            if not self._bulk_index():
                return False

            # 最终状态汇报
            self.log(f"Success! Processed {self.processed_count} segments.", "SUCCESS")
            self.log(f"Data indexed to Elasticsearch: {self.index_name}")
            return True

        except Exception as e:
            self.log(f"Critical Error: {str(e)}", "ERROR")
            import traceback
            traceback.print_exc()
            return False

    def run_skip_es(self):
        """执行转换主流程但跳过 Elasticsearch 存储，保存到本地文件"""
        from common.config import MODEL_NAME, EMBEDDING_DIM
        self.log(f"Starting Conversion: {self.input_path}")
        self.log(f"Skipping Elasticsearch indexing, will save to local file")
        self.log(f"Using embedding model: {MODEL_NAME} (dimensions: {EMBEDDING_DIM})")

        if not os.path.exists(self.input_path):
            self.log(f"Input file not found: {self.input_path}", "ERROR")
            return False

        try:
            # 读取文件 (utf-8-sig 兼容 Windows BOM)
            with open(self.input_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)

            # 解析根结构
            root_nodes = []
            if isinstance(data, list):
                root_nodes = data
                # 提取所有节点的文本，检测重复行（页眉页脚）
                nodes_with_text = [item for item in data if item.get("text", "")]
                page_lines = extract_lines_per_page(nodes_with_text)
                if page_lines:
                    self.noise_lines = find_repeated_lines(page_lines)
                    self.log(f"Detected {len(self.noise_lines)} noise lines (headers/footers)")
            elif isinstance(data, dict):
                # 尝试获取文档标题，优先使用构造函数传入的标题，否则使用 JSON 中的标题
                json_title = data.get("doc_name", data.get("title", ""))
                if not self.doc_title and json_title:
                    self.doc_title = json_title
                # 兼容不同的子节点键名 (structure 或 nodes)
                root_nodes = data.get("structure", data.get("nodes", []))
                
                # 提取所有节点的文本，检测重复行（页眉页脚）
                def extract_all_text(structure):
                    result = []
                    for node in structure:
                        if node.get("text", ""):
                            result.append(node)
                        if "nodes" in node:
                            result.extend(extract_all_text(node["nodes"]))
                    return result
                all_nodes = extract_all_text(root_nodes)
                page_lines = extract_lines_per_page(all_nodes)
                if page_lines:
                    self.noise_lines = find_repeated_lines(page_lines)
                    self.log(f"Detected {len(self.noise_lines)} noise lines (headers/footers)")

            self.log(f"Document identified: {self.doc_title}")
            
            # 开始递归
            self._recursive_walk(root_nodes, [], 1)

            # 生成输出文件路径
            base, ext = os.path.splitext(self.input_path)
            output_path = f"{base}_flat{ext}"

            # 保存到本地文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.flat_nodes, f, indent=2, ensure_ascii=False)

            # 最终状态汇报
            print(f"@@PROGRESS@@{{\"phase\": \"Converting\", \"current\": {self.processed_count}, \"total\": {self.processed_count}}}", flush=True)
            self.log(f"Success! Processed {self.processed_count} segments.", "SUCCESS")
            self.log(f"Data saved to local file: {output_path}")
            return True

        except Exception as e:
            self.log(f"Critical Error: {str(e)}", "ERROR")
            import traceback
            traceback.print_exc()
            return False

# ==========================================
# 命令行入口
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Tree-to-Flat JSON Converter for Elasticsearch with Position Tracking and Embedding")
    parser.add_argument("input", help="Input JSON file path (tree structure)")
    parser.add_argument("--index", default=ELASTICSEARCH_INDEX, help="Elasticsearch index name")
    parser.add_argument("--doc-id", default="", help="Document ID")
    parser.add_argument("--kb-id", default="", help="Knowledge Base ID")
    parser.add_argument("--fd-id", default="", help="Folder ID")
    parser.add_argument("--skip-es", action="store_true", help="Skip Elasticsearch indexing and save to local file instead")
    
    args = parser.parse_args()

    # 强制 stdout 使用 utf-8，防止 Windows 控制台乱码
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    # 由于使用连接池，不需要传入 es_host
    converter = ESConverter(args.input, args.index, args.doc_id, args.kb_id, args.fd_id)
    if args.skip_es:
        # 跳过 Elasticsearch，只转换并保存到本地文件
        success = converter.run_skip_es()
    else:
        success = converter.run()

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
