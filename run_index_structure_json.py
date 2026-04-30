"""
将 _chunked_structure.json 写入 doc_index / chapter_index / chunk_index 三级索引。

用法：
    python run_index_structure_json.py \
        --input pdf/珠三角电子信息产业集群创新网络演化及其机理研究_王炜_chunked_structure.json \
        --doc-id doc_001 --kb-id kb_001 --fd-id fd_001 \
        --industry-tag 电子信息
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.nlp.embedding_client import get_embedding
from common.config import (
    EMBEDDING_DIM,
    ELASTICSEARCH_DOC_INDEX,
    ELASTICSEARCH_CHAPTER_INDEX,
    ELASTICSEARCH_CHUNK_INDEX,
)
from common.doc_store.es_conn_pool import ES_CONN
from elasticsearch.helpers import bulk


# ==================== 工具函数 ====================

def _is_leaf(node: dict) -> bool:
    """判断是否为叶子节点"""
    return not node.get("nodes") or len(node["nodes"]) == 0


def _collect_leaves(node: dict, parent_title: str, chapter_id: str, chapter_title: str,
                    doc_id: str, kb_id: str, fd_id: str, doc_title: str,
                    leaves: list):
    """递归收集叶子节点，同时记录其上一级节点的 title (section_title)"""
    if _is_leaf(node):
        leaves.append({
            "chunk_id": node.get("title", ""),
            "doc_id": doc_id,
            "kb_id": kb_id,
            "chapter_id": chapter_id,
            "fd_id": fd_id,
            "doc_title": doc_title,
            "chapter_title": chapter_title,
            "section_title": parent_title,
            "page_num_int": list(range(
                node.get("start_page", 0),
                node.get("end_page", node.get("start_page", 0)) + 1
            )) if node.get("start_page") else [],
            "chunk_text": node.get("text", ""),
        })
    else:
        for child in node.get("nodes", []):
            _collect_leaves(
                child,
                parent_title=node.get("title", ""),
                chapter_id=chapter_id,
                chapter_title=chapter_title,
                doc_id=doc_id,
                kb_id=kb_id,
                fd_id=fd_id,
                doc_title=doc_title,
                leaves=leaves,
            )


# ==================== 主流程 ====================

def _ensure_index(es, index_name: str, mapping: dict, delete_if_exists: bool):
    """确保索引存在，不存在则创建"""
    if es.indices.exists(index=index_name):
        if delete_if_exists:
            es.indices.delete(index=index_name)
            print(f"  已删除旧索引: {index_name}")
        else:
            print(f"  索引已存在: {index_name}")
            return
    es.indices.create(index=index_name, body=mapping)
    print(f"  已创建索引: {index_name}")


def _build_mappings():
    """返回三个索引的 mapping"""
    doc_mapping = {
        "mappings": {"properties": {
            "doc_id": {"type": "keyword"},
            "kb_id": {"type": "keyword"},
            "title": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 512}}},
            "keywords": {"type": "keyword"},
            "industry_tag": {"type": "keyword"},
            "doc_type": {"type": "keyword"},
            "create_time": {"type": "date"},
            "report_year": {"type": "keyword"},
            "status": {"type": "integer"},
        }}
    }
    chapter_mapping = {
        "mappings": {"properties": {
            "doc_id": {"type": "keyword"},
            "kb_id": {"type": "keyword"},
            "fd_id": {"type": "keyword"},
            "chapter_id": {"type": "keyword"},
            "chapter_name": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 512}}},
            "chapter_summary": {"type": "text"},
            "chapter_keyword": {"type": "keyword"},
            "embedding": {"type": "dense_vector", "dims": EMBEDDING_DIM, "index": True, "similarity": "cosine"},
            "searchable_text": {"type": "text"},
            "status": {"type": "integer"},
        }}
    }
    chunk_mapping = {
        "mappings": {"properties": {
            "chunk_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "kb_id": {"type": "keyword"},
            "chapter_id": {"type": "keyword"},
            "fd_id": {"type": "keyword"},
            "doc_title": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 512}}},
            "chapter_title": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 512}}},
            "page_num_int": {"type": "integer"},
            "section_title": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 512}}},
            "chunk_text": {"type": "text"},
            "embedding": {"type": "dense_vector", "dims": EMBEDDING_DIM, "index": True, "similarity": "cosine"},
        }}
    }
    return doc_mapping, chapter_mapping, chunk_mapping


def index_structure_json(input_path: str, doc_id: str, kb_id: str, fd_id: str,
                         industry_tag: str = "", doc_type: str = "pdf",
                         delete_if_exists: bool = False):
    """读取 _chunked_structure.json 并写入三级索引"""

    print(f"[1/6] 读取文件: {input_path}")
    with open(input_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    doc_name = data.get("doc_name", "")
    structure = data.get("structure", [])
    print(f"  doc_name={doc_name}, chapters={len(structure)}")

    # ---------- 初始化 ES ----------
    print("[2/6] 初始化 ES 连接 & 创建索引")
    es = ES_CONN.get_conn()
    doc_mapping, chapter_mapping, chunk_mapping = _build_mappings()

    _ensure_index(es, ELASTICSEARCH_DOC_INDEX, doc_mapping, delete_if_exists)
    _ensure_index(es, ELASTICSEARCH_CHAPTER_INDEX, chapter_mapping, delete_if_exists)
    _ensure_index(es, ELASTICSEARCH_CHUNK_INDEX, chunk_mapping, delete_if_exists)

    # ==================== doc_index ====================
    print("[3/6] 写入 doc_index")
    doc_record = {
        "doc_id": doc_id,
        "kb_id": kb_id,
        "title": doc_name,
        "keywords": [],
        "industry_tag": industry_tag,
        "doc_type": doc_type,
        "create_time": datetime.now().isoformat(),
        "report_year": "",
        "status": 1,
    }
    es.index(index=ELASTICSEARCH_DOC_INDEX, id=doc_id, body=doc_record)
    print(f"  ✓ doc_index: doc_id={doc_id}")

    # ==================== chapter_index ====================
    print("[4/6] 构建 chapter_index 数据 & 生成 embedding")
    chapter_actions = []
    for ch in structure:
        chapter_id = ch.get("node_id", "")
        chapter_name = ch.get("title", "")
        chapter_summary = ch.get("prefix_summary", "")
        chapter_keyword = []  # 与之前逻辑一致，留空

        # searchable_text = chapter_name + chapter_summary + chapter_keyword
        searchable_parts = [chapter_name, chapter_summary]
        if chapter_keyword:
            searchable_parts.append(" ".join(chapter_keyword))
        searchable_text = "\n".join([p for p in searchable_parts if p])

        # embedding 基于 chapter_summary
        embedding_source = chapter_summary if chapter_summary else chapter_name
        print(f"  embedding chapter: {chapter_name[:30]}...")
        embedding = get_embedding(embedding_source)
        if not embedding:
            embedding = [0.0] * EMBEDDING_DIM

        chapter_actions.append({
            "_index": ELASTICSEARCH_CHAPTER_INDEX,
            "_id": chapter_id,
            "_source": {
                "doc_id": doc_id,
                "kb_id": kb_id,
                "fd_id": fd_id,
                "chapter_id": chapter_id,
                "chapter_name": chapter_name,
                "chapter_summary": chapter_summary,
                "chapter_keyword": chapter_keyword,
                "embedding": embedding,
                "searchable_text": searchable_text,
                "status": 1,
            }
        })

    print(f"  批量写入 chapter_index ({len(chapter_actions)} 条)...")
    success, failed = bulk(es, chapter_actions)
    print(f"  ✓ chapter_index: {success} succeeded, {failed} failed")

    # ==================== chunk_index ====================
    print("[5/6] 构建 chunk_index 数据 & 生成 embedding")
    chunk_records = []
    for ch in structure:
        chapter_id = ch.get("node_id", "")
        chapter_title = ch.get("title", "")

        # 递归收集叶子节点
        leaves = []
        for sub_node in ch.get("nodes", []):
            _collect_leaves(
                sub_node,
                parent_title=ch.get("title", ""),
                chapter_id=chapter_id,
                chapter_title=chapter_title,
                doc_id=doc_id,
                kb_id=kb_id,
                fd_id=fd_id,
                doc_title=doc_name,
                leaves=leaves,
            )

        # 如果顶级章节本身就是叶子（无子节点）
        if _is_leaf(ch):
            leaves.append({
                "chunk_id": ch.get("title", ""),
                "doc_id": doc_id,
                "kb_id": kb_id,
                "chapter_id": chapter_id,
                "fd_id": fd_id,
                "doc_title": doc_name,
                "chapter_title": chapter_title,
                "section_title": "",
                "page_num_int": list(range(
                    ch.get("start_page", 0),
                    ch.get("end_page", ch.get("start_page", 0)) + 1
                )) if ch.get("start_page") else [],
                "chunk_text": ch.get("text", ""),
            })

        chunk_records.extend(leaves)

    # 批量生成 embedding
    total = len(chunk_records)
    print(f"  共 {total} 个 chunk，开始生成 embedding...")
    for i, rec in enumerate(chunk_records):
        chunk_text = rec.get("chunk_text", "")
        if chunk_text:
            embedding = get_embedding(chunk_text)
        else:
            embedding = []
        if not embedding:
            embedding = [0.0] * EMBEDDING_DIM
        rec["embedding"] = embedding
        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(f"    embedding: {i + 1}/{total}")

    # 分批 bulk 写入
    print(f"  批量写入 chunk_index ({total} 条)...")
    batch_size = 50
    total_success = 0
    for start in range(0, total, batch_size):
        batch = chunk_records[start:start + batch_size]
        actions = [
            {"_index": ELASTICSEARCH_CHUNK_INDEX, "_source": r}
            for r in batch
        ]
        s, f = bulk(es, actions)
        total_success += s
    print(f"  ✓ chunk_index: {total_success} succeeded")

    # ==================== 汇总 ====================
    print("\n[6/6] 索引写入完成")
    print(f"  doc_index:     1 条")
    print(f"  chapter_index: {len(chapter_actions)} 条")
    print(f"  chunk_index:   {total} 条")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 _chunked_structure.json 写入三级 ES 索引")
    parser.add_argument("--input", required=True, help="输入 JSON 文件路径")
    parser.add_argument("--doc-id", default=f"doc_{int(time.time())}", help="文档 ID")
    parser.add_argument("--kb-id", default="", help="知识库 ID")
    parser.add_argument("--fd-id", default="", help="文件夹 ID")
    parser.add_argument("--industry-tag", default="", help="行业标签")
    parser.add_argument("--doc-type", default="pdf", help="文档类型")
    parser.add_argument("--delete-if-exists", action="store_true", help="若索引已存在则先删除重建")
    args = parser.parse_args()

    index_structure_json(
        input_path=args.input,
        doc_id=args.doc_id,
        kb_id=args.kb_id,
        fd_id=args.fd_id,
        industry_tag=args.industry_tag,
        doc_type=args.doc_type,
        delete_if_exists=args.delete_if_exists,
    )
