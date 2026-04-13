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
        
        body = {
            "size": topk,
            "query": {
                "bool": {
                    "filter": []
                }
            },
            "knn": {
                "field": "embedding",
                "query_vector": query_vec,
                "k": topk,
                "num_candidates": 200  # 优化：增加候选数
            }
        }

        # 处理kb_ids
        body["query"]["bool"]["filter"].append({"terms": {"kb_id": kb_ids}})

        # 处理fd_ids
        if fd_ids:
            # 对于多个fd_ids，使用多个wildcard查询
            body["query"]["bool"]["filter"].append({"terms": {"folder": fd_ids}})
          

        res = self.es.search(index="doc_summary_index", body=body)

        return [h["_source"]["doc_id"] for h in res["hits"]["hits"]]

    def retrieve_chunks(self, query, query_vec, doc_ids, topk=100):
    
        body = {
            "size": topk,
            "query": {
                "bool": {
                    "must": [],
                    "should": [
                        {
                            "match": {
                                "embedding_text": {
                                    "query": query,
                                    "boost": 0.3
                                }
                            }
                        }
                    ]
                }
            },
            "knn": {
                "field": "embedding",
                "query_vector": query_vec,
                "k": topk,
                "num_candidates": 200  # 优化：增加候选数
            }
        }

        # 处理doc_ids
        if doc_ids:
            body["query"]["bool"]["must"].append({"terms": {"doc_id": doc_ids}})

        res = self.es.search(index=ELASTICSEARCH_INDEX, body=body)

        return [
            {**hit["_source"], "_score": hit["_score"]}
            for hit in res["hits"]["hits"]
        ]