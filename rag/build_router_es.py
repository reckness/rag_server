#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Document Router Builder for Elasticsearch
生成文档路由并存入 Elasticsearch
"""

import json
import os
import sys
import argparse
import time
import re
import numpy as np
from collections import Counter
from datetime import datetime
from typing import Dict, List, Any
from elasticsearch.helpers import bulk

# 添加项目根目录到路径，以便导入 common 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从配置文件导入配置
from common.config import API_URL, MODEL_NAME, EMBEDDING_DIM, ELASTICSEARCH_HOST, ELASTICSEARCH_USER, ELASTICSEARCH_PASSWORD

# 导入 Elasticsearch 连接池
from common.doc_store.es_conn_pool import ES_CONN

# 导入嵌入客户端
from common.nlp.embedding_client import get_embedding

# 导入所需的库
from rank_bm25 import BM25Okapi
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ==================== 工具函数 ====================




def is_noise(text):
    """判断文本是否为噪音"""
    # 检查是否主要由空白字符组成
    if text.strip() == "":
        return True
    
    # 检查是否主要由标点符号组成
    punctuation = "，。！？；：""''（）【】[]()"
    if all(c in punctuation or c.isspace() for c in text):
        return True
    
    # 检查是否包含过多的重复字符
    if len(text) > 10 and any(text.count(c) > len(text) * 0.5 for c in text):
        return True
    
    return False


def is_valid_chunk(item):
    """判断 chunk 是否有效"""
    text = item.get("embedding_text", "")
    
    if len(text) < 80:
        return False
        
    if "目录" in item.get("section_hint", ""):
        return False
        
    if is_noise(text):
        return False
        
    return True


def calc_weight(item):
    """计算 chunk 权重"""
    weight = 1.0
    hint = item.get("section_hint", "")
    depth = item.get("metadata", {}).get("depth", 1)

    if "引言" in hint:
        weight += 1.2
    if "正文章节" in hint:
        weight += 0.6

    weight += max(0, 2 - depth) * 0.3
    return weight


def normalize_chunks(data):
    """标准化 chunks"""
    chunks = []
    for item in data:
        if not is_valid_chunk(item):
            continue

        # 文本已经在转换阶段清洗过，直接使用
        text = item.get("embedding_text", "")

        chunks.append({
            "text": text,
            "weight": calc_weight(item),
            "section": " > ".join(item.get("section_path", []))
        })
    return chunks


def split_sentences(text):
    """拆分句子"""
    sents = re.split(r'[。！？]', text)
    return [s.strip() for s in sents if len(s.strip()) > 20]


def extract_keywords_bm25(chunks, topk=10):
    """使用 BM25 提取关键词"""
    if not chunks:
        return []
    
    corpus = [c["text"].split() for c in chunks]
    if not corpus:
        return []
    
    try:
        bm25 = BM25Okapi(corpus)
        
        scores = {}
        for doc in corpus:
            for word in doc:
                scores[word] = scores.get(word, 0) + 1
        
        keywords = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [k for k, _ in keywords[:topk]]
    except:
        return []


def textrank(sentences):
    """改进版 TextRank"""
    texts = [s["sentence"] for s in sentences]

    # 检查文本是否为空或只包含停用词
    if not texts or all(len(text.strip()) == 0 for text in texts):
        # 如果文本为空，直接返回带有默认分数的句子
        for s in sentences:
            s["score"] = s["weight"] * 0.5
        return sentences

    try:
        tfidf = TfidfVectorizer().fit_transform(texts)
        # 检查是否有有效词汇
        if tfidf.shape[1] == 0:
            # 如果词汇表为空，直接返回带有默认分数的句子
            for s in sentences:
                s["score"] = s["weight"] * 0.5
            return sentences
        sim_matrix = cosine_similarity(tfidf)

        graph = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(graph)

        for i, s in enumerate(sentences):
            s["score"] = scores[i] * s["weight"]

        return sentences
    except ValueError as e:
        # 处理词汇表为空的情况
        if "empty vocabulary" in str(e):
            for s in sentences:
                s["score"] = s["weight"] * 0.5
            return sentences
        else:
            # 其他错误重新抛出
            raise


def similarity(a, b):
    """计算两个句子的相似度"""
    return len(set(a) & set(b)) / (len(set(a)) + 1e-5)


def mmr(sentences, topk=5, lambda_param=0.7):
    """MMR 去冗余"""
    if not sentences:
        return []
    
    selected = []
    candidates = sentences.copy()

    while len(selected) < topk and candidates:
        best = None
        best_score = -1

        for c in candidates:
            relevance = c["score"]

            diversity = 0
            for s in selected:
                diversity = max(diversity, similarity(c["sentence"], s["sentence"]))

            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity

            if mmr_score > best_score:
                best = c
                best_score = mmr_score

        selected.append(best)
        candidates.remove(best)

    return selected


def build_summary(vector_json):
    """构建摘要：生成全面覆盖文档内容的 routing_text 用于文档路由"""
    chunks = normalize_chunks(vector_json)

    # --- 1. 收集所有章节标题（去重保序） ---
    seen_sections = set()
    section_titles = []
    for item in vector_json:
        sp = item.get("section_path", [])
        for title in sp:
            title = title.strip()
            if title and title not in seen_sections and len(title) >= 2:
                seen_sections.add(title)
                section_titles.append(title)

    # --- 2. 收集每个顶级章节的首段文本（前200字） ---
    seen_top_sections = set()
    first_paragraphs = []
    for item in vector_json:
        sp = item.get("section_path", [])
        top_section = sp[0] if sp else ""
        if top_section and top_section not in seen_top_sections:
            seen_top_sections.add(top_section)
            text = item.get("embedding_text", "").strip()
            if text and len(text) > 30:
                first_paragraphs.append(text[:200])

    # --- 3. TextRank + MMR 提取代表性句子 ---
    sentences = []
    for c in chunks:
        for s in split_sentences(c["text"]):
            sentences.append({
                "sentence": s,
                "weight": c["weight"],
                "section": c["section"]
            })

    sentences = textrank(sentences)
    top_sentences = mmr(sentences, topk=6)
    summary = "。".join([s["sentence"] for s in top_sentences])

    # --- 4. BM25 关键词 ---
    keywords = extract_keywords_bm25(chunks)

    # --- 5. 组合 routing_text：标题目录 + 章节首段 + 代表性句子 + 关键词 ---
    parts = []
    if section_titles:
        parts.append("目录：" + "；".join(section_titles[:30]))
    if first_paragraphs:
        parts.append("各章节摘要：" + "。".join(first_paragraphs[:8]))
    if summary:
        parts.append("核心内容：" + summary)
    if keywords:
        parts.append("关键词：" + ",".join(keywords))

    routing_text = "\n".join(parts)
    # 截断到 embedding 模型安全长度
    if len(routing_text) > 6000:
        routing_text = routing_text[:6000]

    return {
        "summary": summary,
        "keywords": keywords,
        "routing_text": routing_text,
        "representative_sentences": [s["sentence"] for s in top_sentences]
    }

class DocumentRouter:
    """文档路由器，用于生成文档路由并存入 Elasticsearch"""
    
    def __init__(self, input_path: str, doc_id: str = "", kb_id: str = "", fd_id: str = "", doc_title: str = "", flat_nodes: list = None):
        """初始化文档路由器"""
        self.input_path = input_path
        self.doc_id = doc_id
        self.kb_id = kb_id
        self.fd_id = fd_id
        self.doc_title = doc_title
        self.flat_nodes = flat_nodes
        self.es_client = None
        # 如果没有提供文档标题，则从输入文件路径中提取
        if not self.doc_title:
            # 从输入文件路径中提取文件名作为文档标题
            base_name = os.path.basename(input_path)
            # 移除文件扩展名
            self.doc_title = os.path.splitext(base_name)[0]
    
    def log(self, msg: str, level: str = "INFO"):
        """标准日志输出"""
        print(f"[{level}] {msg}", flush=True)
    
    def _init_elasticsearch(self):
        """初始化 Elasticsearch 客户端"""
        try:
            # 使用连接池获取 Elasticsearch 客户端
            self.es_client = ES_CONN.get_conn()
            # 测试连接
            info = self.es_client.info()
            self.log(f"Connected to Elasticsearch: {ELASTICSEARCH_HOST}")
            self.log(f"Elasticsearch version: {info['version']['number']}")
            return True
        except Exception as e:
            self.log(f"Failed to connect to Elasticsearch: {str(e)}", "ERROR")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_index(self):
        """创建 Elasticsearch 索引"""
        index_name = "doc_summary_index"
        try:
            if not self.es_client.indices.exists(index=index_name):
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
                            "title": {
                                "type": "text"
                            },
                            "summary": {
                                "type": "text"
                            },
                            "keywords": {
                                "type": "keyword"
                            },
                            "topics": {
                                "type": "keyword"
                            },
                            "routing_text": {
                                "type": "text"
                            },
                            "embedding": {
                                "type": "dense_vector",
                                "dims": EMBEDDING_DIM,
                                "index": True,
                                "similarity": "cosine"
                            },
                            "doc_type": {
                                "type": "keyword"
                            },
                            "status": {
                                "type": "integer"
                            },
                            "create_time": {
                                "type": "date"
                            }
                        }
                    }
                }
                self.es_client.indices.create(index=index_name, body=mapping)
                self.log(f"Created index: {index_name}")
            else:
                self.log(f"Index {index_name} already exists")
            return True
        except Exception as e:
            self.log(f"Failed to create index: {str(e)}", "ERROR")
            return False
    
    def _generate_summary(self, data: Dict) -> str:
        """生成文档摘要"""
        summary = ""
        if isinstance(data, dict) and "structure" in data:
            structure = data["structure"]
            # 提取前几个节点的标题和内容
            for i, node in enumerate(structure[:3]):
                if "title" in node:
                    summary += f"标题: {node['title']}\n"
                if "text" in node:
                    text = node["text"][:200]  # 取前200字符
                    summary += f"内容: {text}...\n"
        elif isinstance(data, list):
            # 处理扁平化的 JSON 列表
            for i, item in enumerate(data[:3]):
                if "embedding_text" in item:
                    text = item["embedding_text"][:200]
                    summary += f"内容 {i+1}: {text}...\n"
        return summary
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词（简单实现，实际项目中可使用更复杂的 NLP 方法）"""
        # 简单实现，实际项目中可使用 TF-IDF、TextRank 等方法
        # 这里仅作为示例
        keywords = []
        # 可以根据实际需求实现更复杂的关键词提取逻辑
        return keywords
    
    def _extract_topics(self, text: str) -> List[str]:
        """提取主题（简单实现，实际项目中可使用更复杂的 NLP 方法）"""
        # 简单实现，实际项目中可使用主题模型等方法
        topics = []
        # 可以根据实际需求实现更复杂的主题提取逻辑
        return topics
    
    def _generate_routing_text(self, title: str, summary: str, keywords: List[str], topics: List[str]) -> str:
        """生成路由文本"""
        # 组合标题、摘要、关键词和主题
        routing_text = f"{title}\n{summary}\n"
        if keywords:
            routing_text += "关键词: " + " ".join(keywords) + "\n"
        if topics:
            routing_text += "主题: " + " ".join(topics) + "\n"
        return routing_text
    
    def _process_document(self):
        """处理文档，生成路由信息"""
        try:
            # 处理不同结构的数据
            vector_json = []
            data = None
            
            # 优先使用提供的 flat_nodes
            if self.flat_nodes:
                vector_json = self.flat_nodes
            else:
                # 读取文件
                with open(self.input_path, 'r', encoding='utf-8-sig') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    # 扁平化结构（已经在转换阶段处理过）
                    vector_json = data
                elif isinstance(data, dict):
                    # 树状结构
                    if "structure" in data:
                        # 树状结构应该先通过 json_to_es_converter_with_embedding.py 转换为扁平化结构
                        # 这里我们假设输入已经是扁平化结构
                        vector_json = data.get("structure", [])
            
            # 生成摘要、关键词和路由文本
            if vector_json:
                summary_info = build_summary(vector_json)
                summary = summary_info["summary"]
                keywords = summary_info["keywords"]
                routing_text = summary_info["routing_text"]
            else:
                # 回退到原始方法
                summary = self._generate_summary(data)
                keywords = []
                routing_text = summary

            # 确保 routing_text 包含文档标题
            if self.doc_title and self.doc_title not in (routing_text or ""):
                routing_text = f"文档标题：{self.doc_title}\n{routing_text or ''}"

            # routing_text 兜底：至少有标题
            if not routing_text or len(routing_text.strip()) < 10:
                routing_text = self.doc_title or "未知文档"
                self.log(f"routing_text 过短，使用文档标题兜底", "WARN")
            
            # 提取主题（简单实现）
            topics = []
            
            # 生成嵌入
            embedding = get_embedding(routing_text)
            if not embedding:
                # 如果获取嵌入失败，使用零向量作为 fallback
                embedding = [0.0] * EMBEDDING_DIM
            
            # 构建文档路由对象
            doc_router = {
                "doc_id": self.doc_id if self.doc_id else f"doc_{int(time.time())}",
                "kb_id": self.kb_id,
                "fd_id": self.fd_id,
                "title": self.doc_title,
                "summary": summary,
                "keywords": keywords,
                "topics": topics,
                "routing_text": routing_text,
                "embedding": embedding,
                "doc_type": "pdf",  # 默认为 PDF 类型，可根据实际情况调整
                "status": 1,  # 1 表示正常状态
                "create_time": datetime.now().isoformat()
            }
            
            return doc_router
        except Exception as e:
            self.log(f"Failed to process document: {str(e)}", "ERROR")
            import traceback
            traceback.print_exc()
            return None
    
    def _index_document(self, doc_router: Dict):
        """索引文档路由到 Elasticsearch"""
        try:
            index_name = "doc_summary_index"
            action = {
                "_index": index_name,
                "_source": doc_router
            }
            success, failed = bulk(self.es_client, [action])
            self.log(f"Indexing completed: {success} succeeded, {failed} failed")
            return True
        except Exception as e:
            self.log(f"Failed to index document: {str(e)}", "ERROR")
            return False
    
    def run(self):
        """执行文档路由生成和索引"""
        self.log(f"Starting Document Router: {self.input_path}")
        
        # 当没有提供 flat_nodes 时，检查文件是否存在
        if not self.flat_nodes and not os.path.exists(self.input_path):
            self.log(f"Input file not found: {self.input_path}", "ERROR")
            return False
        
        # 初始化 Elasticsearch 客户端
        if not self._init_elasticsearch():
            return False
        
        # 创建索引
        if not self._create_index():
            return False
        
        # 处理文档
        doc_router = self._process_document()
        if not doc_router:
            return False
        
        # 索引文档
        if not self._index_document(doc_router):
            return False
        
        self.log(f"Success! Document router generated and indexed.", "SUCCESS")
        return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Document Router Builder for Elasticsearch")
    parser.add_argument("input", help="Input JSON file path")
    parser.add_argument("--doc-id", default="", help="Document ID")
    parser.add_argument("--kb-id", default="", help="Knowledge Base ID")
    parser.add_argument("--fd-id", default="", help="Folder ID")
    
    args = parser.parse_args()
    
    # 强制 stdout 使用 utf-8，防止 Windows 控制台乱码
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    
    router = DocumentRouter(args.input, args.doc_id, args.kb_id, args.fd_id)
    success = router.run()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
