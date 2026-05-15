from typing import List, Dict, Any
from common.config import (
    ELASTICSEARCH_INDEX,
    ELASTICSEARCH_DOC_INDEX,
    ELASTICSEARCH_CHAPTER_INDEX,
    ELASTICSEARCH_CHUNK_INDEX,
)
from common.doc_store.es_conn_pool import ES_CONN


class ESService:

    def __init__(self):
        self.es = ES_CONN.get_conn()

    # ================================================================
    # 第一重：doc_index 混合召回（宽松，有就召回）
    # ================================================================
    def retrieve_docs(self, query, kb_ids, fd_ids=None, query_vec=None, topk=20, rrf_k=60):
        """在 doc_index 中做 KNN + BM25 RRF 融合召回候选文档"""
        if not kb_ids:
            raise ValueError("kb_ids不能为空")
        if not isinstance(kb_ids, list):
            raise ValueError("kb_ids必须是列表")
        if fd_ids and not isinstance(fd_ids, list):
            fd_ids = [fd_ids]

        must = [{"terms": {"kb_id": kb_ids}}]
        if fd_ids:
            must.append({"terms": {"fd_id": fd_ids}})

        doc_filter = {"bool": {"must": must}}

        bm25_body = {
            "size": topk,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^3", "summary^2", "searchable_text", "keywords"],
                            }
                        }
                    ],
                    "filter": must,
                }
            },
            "_source": ["doc_id"],
        }

        bm25_res = self.es.search(index=ELASTICSEARCH_DOC_INDEX, body=bm25_body)

        knn_res = {"hits": {"hits": []}}
        if query_vec:
            knn_body = {
                "size": topk,
                "knn": {
                    "field": "embedding",
                    "query_vector": query_vec,
                    "k": topk,
                    "num_candidates": 500,
                    "filter": doc_filter,
                },
                "_source": ["doc_id"],
            }
            knn_res = self.es.search(index=ELASTICSEARCH_DOC_INDEX, body=knn_body)

        rrf_scores: Dict[str, float] = {}
        doc_meta: Dict[str, Dict] = {}

        for rank, hit in enumerate(knn_res["hits"]["hits"], start=1):
            key = hit["_id"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (rrf_k + rank)
            doc_meta[key] = hit["_source"]

        for rank, hit in enumerate(bm25_res["hits"]["hits"], start=1):
            key = hit["_id"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (rrf_k + rank)
            if key not in doc_meta:
                doc_meta[key] = hit["_source"]

        ranked_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:topk]

        seen = set()
        doc_ids = []
        for key, _ in ranked_docs:
            did = doc_meta[key]["doc_id"]
            if did not in seen:
                seen.add(did)
                doc_ids.append(did)

        return doc_ids

    # ================================================================
    # 第二重：chapter_index 章节召回（KNN + BM25 RRF，阈值偏低）
    # ================================================================
    def retrieve_chapters(self, query, query_vec, doc_ids, kb_ids=None,
                          topk=30, rrf_k=60, rrf_threshold=0.010):
        """在候选 doc 的章节中做 KNN + BM25 RRF 融合召回"""
        filters = []
        if doc_ids:
            filters.append({"terms": {"doc_id": doc_ids}})
        if kb_ids:
            filters.append({"terms": {"kb_id": kb_ids}})
        chapter_filter = {"bool": {"must": filters}} if filters else None

        # --- KNN ---
        knn_body = {
            "size": topk,
            "knn": {
                "field": "embedding",
                "query_vector": query_vec,
                "k": topk,
                "num_candidates": 500,
            },
            "_source": ["doc_id", "chapter_id", "chapter_name"],
        }
        if chapter_filter:
            knn_body["knn"]["filter"] = chapter_filter

        knn_res = self.es.search(index=ELASTICSEARCH_CHAPTER_INDEX, body=knn_body)

        # --- BM25 ---
        bm25_body = {
            "size": topk,
            "query": {
                "bool": {
                    "must": [{"match": {"searchable_text": {"query": query}}}],
                }
            },
            "_source": ["doc_id", "chapter_id", "chapter_name"],
        }
        if chapter_filter:
            bm25_body["query"]["bool"]["filter"] = chapter_filter

        bm25_res = self.es.search(index=ELASTICSEARCH_CHAPTER_INDEX, body=bm25_body)

        # --- RRF 融合 ---
        rrf_scores: Dict[str, float] = {}
        chapter_meta: Dict[str, Dict] = {}

        for rank, hit in enumerate(knn_res["hits"]["hits"], start=1):
            key = hit["_id"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (rrf_k + rank)
            chapter_meta[key] = hit["_source"]

        for rank, hit in enumerate(bm25_res["hits"]["hits"], start=1):
            key = hit["_id"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (rrf_k + rank)
            if key not in chapter_meta:
                chapter_meta[key] = hit["_source"]

        # 过滤 + 排序
        results = []
        for key, score in rrf_scores.items():
            if score >= rrf_threshold:
                results.append({**chapter_meta[key], "_score": score})
        results.sort(key=lambda x: x["_score"], reverse=True)

        # 提取 chapter_ids
        chapter_ids = [r["chapter_id"] for r in results]
        return chapter_ids, results

    # ================================================================
    # 第三重：chunk_index 精细召回（KNN + BM25 RRF，正常阈值）
    # ================================================================
    def retrieve_chunks(self, query, query_vec, chapter_ids, kb_ids=None, doc_ids=None,
                        topk=100, rrf_k=60, rrf_threshold=0.015):
        """在候选章节中做 KNN + BM25 RRF 精细召回"""
        filters = []
        if chapter_ids:
            filters.append({"terms": {"chapter_id": chapter_ids}})
        if kb_ids:
            filters.append({"terms": {"kb_id": kb_ids}})
        if doc_ids:
            filters.append({"terms": {"doc_id": doc_ids}})
        chunk_filter = {"bool": {"must": filters}} if filters else None

        # --- KNN ---
        knn_body = {
            "size": topk,
            "knn": {
                "field": "embedding",
                "query_vector": query_vec,
                "k": topk,
                "num_candidates": 500,
            },
            "_source": {"excludes": ["embedding"]},
        }
        if chunk_filter:
            knn_body["knn"]["filter"] = chunk_filter

        knn_res = self.es.search(index=ELASTICSEARCH_CHUNK_INDEX, body=knn_body)

        # --- BM25 ---
        bm25_body = {
            "size": topk,
            "query": {
                "bool": {
                    "must": [{"match": {"chunk_text": {"query": query}}}],
                }
            },
            "_source": {"excludes": ["embedding"]},
        }
        if chunk_filter:
            bm25_body["query"]["bool"]["filter"] = chunk_filter

        bm25_res = self.es.search(index=ELASTICSEARCH_CHUNK_INDEX, body=bm25_body)

        # --- RRF 融合 ---
        rrf_scores: Dict[str, float] = {}
        doc_sources: Dict[str, Dict] = {}

        for rank, hit in enumerate(knn_res["hits"]["hits"], start=1):
            key = hit["_id"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (rrf_k + rank)
            doc_sources[key] = hit["_source"]

        for rank, hit in enumerate(bm25_res["hits"]["hits"], start=1):
            key = hit["_id"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (rrf_k + rank)
            if key not in doc_sources:
                doc_sources[key] = hit["_source"]

        # 过滤 + 排序
        merged = []
        for key, score in rrf_scores.items():
            if score >= rrf_threshold:
                merged.append({**doc_sources[key], "_score": score})
        merged.sort(key=lambda x: x["_score"], reverse=True)

        return merged[:topk]

    # ================================================================
    # 二级检索：doc → chunk 直接召回（跳过 chapter 级）
    # ================================================================
    def retrieve_chunks_by_docs(self, query, query_vec, doc_ids, kb_ids=None,
                                topk=100, rrf_k=60):
        """跳过 chapter 级，直接在候选 doc 中做 KNN + BM25 RRF 精细召回"""
        filters = []
        if doc_ids:
            filters.append({"terms": {"doc_id": doc_ids}})
        if kb_ids:
            filters.append({"terms": {"kb_id": kb_ids}})
        chunk_filter = {"bool": {"must": filters}} if filters else None

        # --- KNN ---
        knn_body = {
            "size": topk,
            "knn": {
                "field": "embedding",
                "query_vector": query_vec,
                "k": topk,
                "num_candidates": 500,
            },
            "_source": {"excludes": ["embedding"]},
        }
        if chunk_filter:
            knn_body["knn"]["filter"] = chunk_filter

        knn_res = self.es.search(index=ELASTICSEARCH_CHUNK_INDEX, body=knn_body)

        # --- BM25 ---
        bm25_body = {
            "size": topk,
            "query": {
                "bool": {
                    "must": [{"match": {"chunk_text": {"query": query}}}],
                }
            },
            "_source": {"excludes": ["embedding"]},
        }
        if chunk_filter:
            bm25_body["query"]["bool"]["filter"] = chunk_filter

        bm25_res = self.es.search(index=ELASTICSEARCH_CHUNK_INDEX, body=bm25_body)

        # --- RRF 融合（无阈值截断，全部保留让 rerank 决定）---
        rrf_scores: Dict[str, float] = {}
        doc_sources: Dict[str, Dict] = {}

        for rank, hit in enumerate(knn_res["hits"]["hits"], start=1):
            key = hit["_id"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (rrf_k + rank)
            doc_sources[key] = hit["_source"]

        for rank, hit in enumerate(bm25_res["hits"]["hits"], start=1):
            key = hit["_id"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (rrf_k + rank)
            if key not in doc_sources:
                doc_sources[key] = hit["_source"]

        merged = []
        for key, score in rrf_scores.items():
            merged.append({**doc_sources[key], "_score": score})
        merged.sort(key=lambda x: x["_score"], reverse=True)

        return merged[:topk]