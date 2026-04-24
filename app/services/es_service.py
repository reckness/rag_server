from typing import List, Dict, Any
from common.config import ELASTICSEARCH_INDEX
from common.doc_store.es_conn_pool import ES_CONN


class ESService:

    def __init__(self):
        self.es = ES_CONN.get_conn()

    def retrieve_docs(self, query_vec, kb_ids, fd_ids=None, topk=5):
        # 验证参数
        if not kb_ids:
            raise ValueError("kb_ids不能为空")
        if not isinstance(kb_ids, list):
            raise ValueError("kb_ids必须是列表")
        if topk is None:
            topk = 5
        if fd_ids and not isinstance(fd_ids, list):
            fd_ids = [fd_ids]
        
        # 构建 KNN filter（ES 8.x 中 filter 必须在 knn 块内才会对 KNN 结果生效）
        knn_filters = [{"terms": {"kb_id": kb_ids}}]
        if fd_ids:
            knn_filters.append({"terms": {"folder": fd_ids}})

        body = {
            "size": topk,
            "knn": {
                "field": "embedding",
                "query_vector": query_vec,
                "k": topk,
                "num_candidates": 200,
                "filter": {"bool": {"must": knn_filters}}
            }
        }

        res = self.es.search(index="doc_summary_index", body=body)

        return [h["_source"]["doc_id"] for h in res["hits"]["hits"]]

    def retrieve_chunks(self, query, query_vec, doc_ids, topk=100, rrf_k=60):
        """混合检索：分别执行 KNN 和 BM25，使用 RRF（倒数秩融合）打分"""
        doc_filter = {"terms": {"doc_id": doc_ids}} if doc_ids else None

        # --- 1. KNN 向量检索 ---
        knn_body = {
            "size": topk,
            "knn": {
                "field": "embedding",
                "query_vector": query_vec,
                "k": topk,
                "num_candidates": 200,
            },
            "_source": {"excludes": ["embedding"]}
        }
        if doc_filter:
            knn_body["knn"]["filter"] = doc_filter
        knn_res = self.es.search(index=ELASTICSEARCH_INDEX, body=knn_body)

        # --- 2. BM25 文本检索 ---
        bm25_body = {
            "size": topk,
            "query": {
                "bool": {
                    "must": [
                        {"match": {"embedding_text": {"query": query}}}
                    ]
                }
            },
            "_source": {"excludes": ["embedding"]}
        }
        if doc_filter:
            bm25_body["query"]["bool"]["filter"] = doc_filter
        bm25_res = self.es.search(index=ELASTICSEARCH_INDEX, body=bm25_body)

        # --- 3. RRF 倒数秩融合 ---
        # RRF(d) = Σ 1 / (k + rank_i)，k 为平滑常数（默认60）
        rrf_scores = {}   # doc_id -> rrf_score
        doc_sources = {}  # doc_id -> _source

        for rank, hit in enumerate(knn_res["hits"]["hits"], start=1):
            doc_key = hit["_id"]
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0) + 1.0 / (rrf_k + rank)
            doc_sources[doc_key] = hit["_source"]

        for rank, hit in enumerate(bm25_res["hits"]["hits"], start=1):
            doc_key = hit["_id"]
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0) + 1.0 / (rrf_k + rank)
            if doc_key not in doc_sources:
                doc_sources[doc_key] = hit["_source"]

        # --- 4. 按 RRF 分数排序 ---
        merged = []
        for doc_key, score in rrf_scores.items():
            merged.append({**doc_sources[doc_key], "_score": score})

        merged.sort(key=lambda x: x["_score"], reverse=True)
        return merged[:topk]