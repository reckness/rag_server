"""
三级索引写入模块：doc_index / chapter_index / chunk_index

可被 API 服务和独立脚本共同复用。
"""
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

from elasticsearch.helpers import bulk

from common.config import (
    EMBEDDING_DIM,
    ELASTICSEARCH_DOC_INDEX,
    ELASTICSEARCH_CHAPTER_INDEX,
    ELASTICSEARCH_CHUNK_INDEX,
)
from common.doc_store.es_conn_pool import ES_CONN
from common.nlp.embedding_client import get_embedding

logger = logging.getLogger(__name__)


# ==================== Mapping 定义 ====================

DOC_INDEX_MAPPING = {
    "mappings": {"properties": {
        "doc_id": {"type": "keyword"},
        "kb_id": {"type": "keyword"},
        "fd_id": {"type": "keyword"},
        "title": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 512}}},
        "keywords": {"type": "keyword"},
        "industry_tag": {"type": "keyword"},
        "doc_type": {"type": "keyword"},
        "create_time": {"type": "date"},
        "report_year": {"type": "keyword"},
        "status": {"type": "integer"},
    }}
}

CHAPTER_INDEX_MAPPING = {
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

CHUNK_INDEX_MAPPING = {
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
        "embedding": {"type": "dense_vector", "dims": EMBEDDING_DIM, "index": True, "similarity": "cosine"}
    }}
}


MIN_CHUNK_TEXT_LEN = 30


# ==================== 工具函数 ====================

def _is_leaf(node: dict) -> bool:
    return not node.get("nodes") or len(node["nodes"]) == 0


def _coalesce_page(value, inherited):
    return value if value not in (None, "") else inherited


def _collect_leaves(node: dict, parent_title: str, chapter_id: str,
                    chapter_title: str, doc_id: str, kb_id: str,
                    fd_id: str, doc_title: str, leaves: list,
                    inherited_start_page=None, inherited_end_page=None):
    """递归收集叶子节点"""
    start_page = _coalesce_page(node.get("start_page"), inherited_start_page)
    end_page = _coalesce_page(node.get("end_page"), inherited_end_page)
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
                start_page,
                (end_page or start_page) + 1
            )) if start_page else [],
            "chunk_text": node.get("text", ""),
        })
    else:
        for child in node.get("nodes", []):
            _collect_leaves(
                child,
                parent_title=node.get("title", ""),
                chapter_id=chapter_id,
                chapter_title=chapter_title,
                doc_id=doc_id, kb_id=kb_id, fd_id=fd_id,
                doc_title=doc_title, leaves=leaves,
                inherited_start_page=start_page,
                inherited_end_page=end_page,
            )


def _merge_short_chunks(leaves: list) -> list:
    """合并文本过短的叶子节点（纯标题）到下一个兄弟节点"""
    if not leaves:
        return leaves
    merged = []
    pending_title = ""
    pending_pages = []
    for leaf in leaves:
        text = leaf["chunk_text"].strip()
        if len(text) < MIN_CHUNK_TEXT_LEN:
            pending_title += (text + "\n")
            pending_pages.extend(leaf.get("page_num_int") or [])
            continue
        if pending_title:
            leaf = dict(leaf)
            leaf["chunk_text"] = pending_title + leaf["chunk_text"]
            leaf["page_num_int"] = sorted(set(pending_pages + (leaf.get("page_num_int") or [])))
            pending_title = ""
            pending_pages = []
        merged.append(leaf)
    if pending_title and merged:
        last = dict(merged[-1])
        last["chunk_text"] = last["chunk_text"] + "\n" + pending_title.strip()
        last["page_num_int"] = sorted(set((last.get("page_num_int") or []) + pending_pages))
        merged[-1] = last
    elif pending_title and not merged:
        merged.append(leaves[-1])
    return merged


def _generate_chapter_summary(ch: dict, max_len: int = 500) -> str:
    """当 prefix_summary 为空时，从子节点自动生成章节摘要"""
    parts = []
    for sub in ch.get("nodes", []):
        sub_title = sub.get("title", "")
        if sub_title:
            parts.append(sub_title)
        sub_text = sub.get("text", "").strip()
        if sub_text and len(sub_text) > MIN_CHUNK_TEXT_LEN:
            parts.append(sub_text[:200])
        for leaf in sub.get("nodes", []):
            leaf_text = leaf.get("text", "").strip()
            if leaf_text and len(leaf_text) > MIN_CHUNK_TEXT_LEN:
                parts.append(leaf_text[:200])
                break
    summary = "\n".join(parts)
    return summary[:max_len] if summary else ch.get("title", "")


def _ensure_index(es, index_name: str, mapping: dict):
    """如果索引不存在则创建"""
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=mapping)
        logger.info(f"Created index: {index_name}")


# ==================== 主入口 ====================

def write_to_three_indices(
    json_path: str,
    doc_id: str,
    kb_id: str,
    fd_id: str,
    doc_title: str = "",
    industry_tag: str = "",
    doc_type: str = "pdf",
) -> int:
    """
    读取 structure JSON，写入 doc_index / chapter_index / chunk_index。

    Parameters
    ----------
    json_path : str  — structure JSON 文件路径
    doc_id, kb_id, fd_id : str — 文档标识
    doc_title : str — 文档标题（可选，若为空则从 JSON doc_name 取）
    industry_tag : str — 行业标签
    doc_type : str — 文档类型

    Returns
    -------
    int — chunk 总数
    """
    # ---- 读取 JSON ----
    with open(json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    doc_name = doc_title or data.get("doc_name", "")
    structure = data.get("structure", [])

    # ---- ES 连接 & 确保索引存在 ----
    es = ES_CONN.get_conn()
    _ensure_index(es, ELASTICSEARCH_DOC_INDEX, DOC_INDEX_MAPPING)
    _ensure_index(es, ELASTICSEARCH_CHAPTER_INDEX, CHAPTER_INDEX_MAPPING)
    _ensure_index(es, ELASTICSEARCH_CHUNK_INDEX, CHUNK_INDEX_MAPPING)

    # ==================== 1. doc_index ====================
    doc_record = {
        "doc_id": doc_id,
        "kb_id": kb_id,
        "fd_id": fd_id,
        "title": doc_name,
        "keywords": [],
        "industry_tag": industry_tag,
        "doc_type": doc_type,
        "create_time": datetime.now().isoformat(),
        "report_year": "",
        "status": 1,
    }
    es.index(index=ELASTICSEARCH_DOC_INDEX, id=doc_id, body=doc_record)
    logger.info(f"doc_index: indexed doc_id={doc_id}")

    # ==================== 2. chapter_index ====================
    chapter_actions = []
    for ch in structure:
        chapter_id = ch.get("node_id", "")
        chapter_name = ch.get("title", "")
        chapter_summary = ch.get("prefix_summary", "").strip()
        if not chapter_summary or len(chapter_summary) < MIN_CHUNK_TEXT_LEN:
            chapter_summary = _generate_chapter_summary(ch)
        chapter_keyword = []

        searchable_parts = [chapter_name, chapter_summary]
        if chapter_keyword:
            searchable_parts.append(" ".join(chapter_keyword))
        searchable_text = "\n".join([p for p in searchable_parts if p])

        embedding_source = chapter_summary if chapter_summary else chapter_name
        embedding = get_embedding(embedding_source)
        if not embedding:
            embedding = [0.0] * EMBEDDING_DIM

        chapter_actions.append({
            "_index": ELASTICSEARCH_CHAPTER_INDEX,
            "_id": f"{doc_id}_{chapter_id}",
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

    if chapter_actions:
        success, failed = bulk(es, chapter_actions)
        logger.info(f"chapter_index: {success} succeeded, {failed} failed")

    # ==================== 3. chunk_index ====================
    chunk_records: List[Dict] = []
    for ch in structure:
        chapter_id = ch.get("node_id", "")
        chapter_title = ch.get("title", "")

        leaves: List[Dict] = []
        for sub_node in ch.get("nodes", []):
            _collect_leaves(
                sub_node,
                parent_title=ch.get("title", ""),
                chapter_id=chapter_id,
                chapter_title=chapter_title,
                doc_id=doc_id, kb_id=kb_id, fd_id=fd_id,
                doc_title=doc_name, leaves=leaves,
                inherited_start_page=ch.get("start_page"),
                inherited_end_page=ch.get("end_page"),
            )

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

        leaves = _merge_short_chunks(leaves)
        chunk_records.extend(leaves)

    # 生成 embedding 并写入
    total = len(chunk_records)
    for i, rec in enumerate(chunk_records):
        chunk_text = rec.get("chunk_text", "")
        embedding = get_embedding(chunk_text) if chunk_text else []
        if not embedding:
            embedding = [0.0] * EMBEDDING_DIM
        rec["embedding"] = embedding

    # 分批 bulk 写入
    batch_size = 50
    total_success = 0
    for start in range(0, total, batch_size):
        batch = chunk_records[start:start + batch_size]
        actions = [
            {"_index": ELASTICSEARCH_CHUNK_INDEX, "_source": r}
            for r in batch
        ]
        s, _ = bulk(es, actions)
        total_success += s

    logger.info(f"chunk_index: {total_success} succeeded out of {total}")

    # 刷新索引
    for idx in [ELASTICSEARCH_DOC_INDEX, ELASTICSEARCH_CHAPTER_INDEX, ELASTICSEARCH_CHUNK_INDEX]:
        es.indices.refresh(index=idx)

    return total
