import logging
from typing import Dict, List, Optional

from common.config import ELASTICSEARCH_CHAPTER_INDEX, EMBEDDING_DIM
from common.doc_store.es_conn_pool import ES_CONN

logger = logging.getLogger(__name__)

# chapter_index mapping 定义
CHAPTER_INDEX_MAPPING = {
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
            "chapter_id": {
                "type": "keyword"
            },
            "chapter_name": {
                "type": "text",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 512
                    }
                }
            },
            "chapter_summary": {
                "type": "text"
            },
            "chapter_keyword": {
                "type": "keyword"
            },
            "embedding": {
                "type": "dense_vector",
                "dims": EMBEDDING_DIM,
                "index": True,
                "similarity": "cosine"
            },
            "searchable_text": {
                "type": "text"
            },
            "status": {
                "type": "integer"
            }
        }
    }
}


class ChapterIndexService:
    """chapter_index 索引管理服务"""

    def __init__(self):
        self.es = ES_CONN.get_conn()
        self.index_name = ELASTICSEARCH_CHAPTER_INDEX

    def create_index(self, delete_if_exists: bool = False) -> bool:
        """创建 chapter_index 索引

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

            self.es.indices.create(index=self.index_name, body=CHAPTER_INDEX_MAPPING)
            logger.info(f"Created index: {self.index_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create index {self.index_name}: {e}")
            return False

    def index_chapter(self, chapter: Dict) -> bool:
        """写入单条章节记录"""
        try:
            self.es.index(
                index=self.index_name,
                id=chapter.get("chapter_id"),
                body=chapter
            )
            return True
        except Exception as e:
            logger.error(f"Failed to index chapter: {e}")
            return False

    def bulk_index_chapters(self, chapters: List[Dict]) -> bool:
        """批量写入章节记录"""
        try:
            from elasticsearch.helpers import bulk
            actions = [
                {
                    "_index": self.index_name,
                    "_id": ch.get("chapter_id"),
                    "_source": ch
                }
                for ch in chapters
            ]
            success, failed = bulk(self.es, actions)
            logger.info(f"Bulk indexing chapters: {success} succeeded, {failed} failed")
            return True
        except Exception as e:
            logger.error(f"Failed to bulk index chapters: {e}")
            return False

    def get_chapter(self, chapter_id: str) -> Optional[Dict]:
        """根据 chapter_id 获取章节"""
        try:
            res = self.es.get(index=self.index_name, id=chapter_id)
            return res["_source"]
        except Exception as e:
            logger.error(f"Failed to get chapter {chapter_id}: {e}")
            return None

    def delete_chapter(self, chapter_id: str) -> bool:
        """删除单条章节"""
        try:
            self.es.delete(index=self.index_name, id=chapter_id)
            return True
        except Exception as e:
            logger.error(f"Failed to delete chapter {chapter_id}: {e}")
            return False

    def delete_by_doc_id(self, doc_id: str) -> bool:
        """删除某个文档下的所有章节"""
        try:
            self.es.delete_by_query(
                index=self.index_name,
                body={"query": {"term": {"doc_id": doc_id}}}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete chapters for doc {doc_id}: {e}")
            return False

    def update_kb_id_by_doc_id(self, doc_id: str, new_kb_id: str) -> Optional[int]:
        """按 doc_id 更新章节记录的 kb_id 字段"""
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
            logger.error(f"Failed to update chapter kb_id for doc_id {doc_id}: {e}")
            return None

    def update_kb_id_by_fd_ids(self, fd_ids: List[str], new_kb_id: str) -> Optional[int]:
        """按 fd_id 批量更新章节记录的 kb_id 字段"""
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
            logger.error(f"Failed to update chapter kb_id for fd_ids {fd_ids}: {e}")
            return None

    def search_chapters(self, doc_id: str = None, kb_id: str = None,
                        keyword: str = None, size: int = 50) -> List[Dict]:
        """按条件搜索章节"""
        filters = []
        if doc_id:
            filters.append({"term": {"doc_id": doc_id}})
        if kb_id:
            filters.append({"term": {"kb_id": kb_id}})

        query = {"bool": {"must": filters}} if filters else {"match_all": {}}

        if keyword:
            query = {
                "bool": {
                    "must": filters + [{"match": {"searchable_text": keyword}}]
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
                      kb_id: str = None, topk: int = 10) -> List[Dict]:
        """向量检索章节"""
        knn_filters = []
        if doc_id:
            knn_filters.append({"term": {"doc_id": doc_id}})
        if kb_id:
            knn_filters.append({"term": {"kb_id": kb_id}})

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
