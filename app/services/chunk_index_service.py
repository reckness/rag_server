import logging
from typing import Dict, List, Optional

from common.config import ELASTICSEARCH_CHUNK_INDEX, EMBEDDING_DIM
from common.doc_store.es_conn_pool import ES_CONN

logger = logging.getLogger(__name__)

# chunk_index mapping 定义
CHUNK_INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "chunk_id": {
                "type": "keyword"
            },
            "doc_id": {
                "type": "keyword"
            },
            "kb_id": {
                "type": "keyword"
            },
            "chapter_id": {
                "type": "keyword"
            },
            "fd_id": {
                "type": "keyword"
            },
            "doc_title": {
                "type": "text",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 512
                    }
                }
            },
            "chapter_title": {
                "type": "text",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 512
                    }
                }
            },
            "page_num_int": {
                "type": "integer"
            },
            "section_title": {
                "type": "text",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 512
                    }
                }
            },
            "chunk_text": {
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


class ChunkIndexService:
    """chunk_index 索引管理服务"""

    def __init__(self):
        self.es = ES_CONN.get_conn()
        self.index_name = ELASTICSEARCH_CHUNK_INDEX

    def create_index(self, delete_if_exists: bool = False) -> bool:
        """创建 chunk_index 索引

        Args:
            delete_if_exists: 若索引已存在是否先删除再重建
        Returns:
            是否创建成功
        """
        try:
            exists = self.es.indices.exists(index=self.index_name)
            if exists:
                if delete_if_exists:
                    self.es.indices.delete(index=self.index_name)
                    logger.info(f"Deleted existing index: {self.index_name}")
                else:
                    logger.info(f"Index {self.index_name} already exists, skipping creation")
                    return True

            self.es.indices.create(index=self.index_name, body=CHUNK_INDEX_MAPPING)
            logger.info(f"Created index: {self.index_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create index {self.index_name}: {e}")
            return False

    def index_chunk(self, chunk: Dict) -> bool:
        """写入单条 chunk 记录"""
        try:
            self.es.index(
                index=self.index_name,
                id=chunk.get("chunk_id"),
                body=chunk
            )
            return True
        except Exception as e:
            logger.error(f"Failed to index chunk: {e}")
            return False

    def bulk_index_chunks(self, chunks: List[Dict]) -> bool:
        """批量写入 chunk 记录"""
        try:
            from elasticsearch.helpers import bulk
            actions = [
                {
                    "_index": self.index_name,
                    "_id": ch.get("chunk_id"),
                    "_source": ch
                }
                for ch in chunks
            ]
            success, failed = bulk(self.es, actions)
            logger.info(f"Bulk indexing chunks: {success} succeeded, {failed} failed")
            return True
        except Exception as e:
            logger.error(f"Failed to bulk index chunks: {e}")
            return False

    def get_chunk(self, chunk_id: str) -> Optional[Dict]:
        """根据 chunk_id 获取 chunk"""
        try:
            res = self.es.get(index=self.index_name, id=chunk_id)
            return res["_source"]
        except Exception as e:
            logger.error(f"Failed to get chunk {chunk_id}: {e}")
            return None

    def delete_chunk(self, chunk_id: str) -> bool:
        """删除单条 chunk"""
        try:
            self.es.delete(index=self.index_name, id=chunk_id)
            return True
        except Exception as e:
            logger.error(f"Failed to delete chunk {chunk_id}: {e}")
            return False

    def delete_by_doc_id(self, doc_id: str) -> bool:
        """删除某个文档下的所有 chunk"""
        try:
            self.es.delete_by_query(
                index=self.index_name,
                body={"query": {"term": {"doc_id": doc_id}}}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete chunks for doc {doc_id}: {e}")
            return False

    def update_kb_id_by_doc_id(self, doc_id: str, new_kb_id: str) -> Optional[int]:
        """按 doc_id 更新 chunk 记录的 kb_id 字段"""
        try:
            res = self.es.update_by_query(
                index=self.index_name,
                body={
                    "script": {
                        "source": "ctx._source.kb_id = params.new_kb_id",
                        "lang": "painless",
                        "params": {"new_kb_id": new_kb_id}
                    },
                    "query": {"term": {"doc_id": doc_id}}
                },
                conflicts="proceed",
                refresh=True,
            )
            return int(res.get("updated", 0))
        except Exception as e:
            logger.error(f"Failed to update chunk kb_id for doc_id {doc_id}: {e}")
            return None

    def update_kb_id_by_fd_ids(self, fd_ids: List[str], new_kb_id: str) -> Optional[int]:
        """按 fd_id 批量更新 chunk 记录的 kb_id 字段"""
        try:
            res = self.es.update_by_query(
                index=self.index_name,
                body={
                    "script": {
                        "source": "ctx._source.kb_id = params.new_kb_id",
                        "lang": "painless",
                        "params": {"new_kb_id": new_kb_id}
                    },
                    "query": {"terms": {"fd_id": fd_ids}}
                },
                conflicts="proceed",
                refresh=True,
            )
            return int(res.get("updated", 0))
        except Exception as e:
            logger.error(f"Failed to update chunk kb_id for fd_ids {fd_ids}: {e}")
            return None

    def delete_by_chapter_id(self, chapter_id: str) -> bool:
        """删除某个章节下的所有 chunk"""
        try:
            self.es.delete_by_query(
                index=self.index_name,
                body={"query": {"term": {"chapter_id": chapter_id}}}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete chunks for chapter {chapter_id}: {e}")
            return False

    def search_chunks(self, doc_id: str = None, kb_id: str = None,
                      chapter_id: str = None, keyword: str = None,
                      size: int = 100) -> List[Dict]:
        """按条件搜索 chunk"""
        filters = []
        if doc_id:
            filters.append({"term": {"doc_id": doc_id}})
        if kb_id:
            filters.append({"term": {"kb_id": kb_id}})
        if chapter_id:
            filters.append({"term": {"chapter_id": chapter_id}})

        query = {"bool": {"must": filters}} if filters else {"match_all": {}}

        if keyword:
            query = {
                "bool": {
                    "must": filters + [{"match": {"chunk_text": keyword}}]
                }
            }

        body = {
            "size": size,
            "query": query,
            "_source": {"excludes": ["embedding"]}
        }
        res = self.es.search(index=self.index_name, body=body)
        return [hit["_source"] for hit in res["hits"]["hits"]]

    def vector_search(self, query_vec: List[float], doc_id: str = None,
                      kb_id: str = None, chapter_id: str = None,
                      fd_id: str = None, topk: int = 10) -> List[Dict]:
        """向量检索 chunk"""
        knn_filters = []
        if doc_id:
            knn_filters.append({"term": {"doc_id": doc_id}})
        if kb_id:
            knn_filters.append({"term": {"kb_id": kb_id}})
        if chapter_id:
            knn_filters.append({"term": {"chapter_id": chapter_id}})
        if fd_id:
            knn_filters.append({"term": {"fd_id": fd_id}})

        body = {
            "size": topk,
            "knn": {
                "field": "embedding",
                "query_vector": query_vec,
                "k": topk,
                "num_candidates": 200,
            },
            "_source": {"excludes": ["embedding"]}
        }
        if knn_filters:
            body["knn"]["filter"] = {"bool": {"must": knn_filters}}

        res = self.es.search(index=self.index_name, body=body)
        return [hit["_source"] for hit in res["hits"]["hits"]]
