import logging
from datetime import datetime
from typing import Dict, List, Optional

from common.config import ELASTICSEARCH_DOC_INDEX
from common.doc_store.es_conn_pool import ES_CONN

logger = logging.getLogger(__name__)

# doc_index mapping 定义
DOC_INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "doc_id": {
                "type": "keyword"
            },
            "kb_id": {
                "type": "keyword"
            },
            "title": {
                "type": "text",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 512
                    }
                }
            },
            "keywords": {
                "type": "keyword"
            },
            "industry_tag": {
                "type": "keyword"
            },
            "doc_type": {
                "type": "keyword"
            },
            "create_time": {
                "type": "date"
            },
            "report_year": {
                "type": "keyword"
            },
            "status": {
                "type": "integer"
            }
        }
    }
}


class DocIndexService:
    """doc_index 索引管理服务"""

    def __init__(self):
        self.es = ES_CONN.get_conn()
        self.index_name = ELASTICSEARCH_DOC_INDEX

    def create_index(self, delete_if_exists: bool = False) -> bool:
        """创建 doc_index 索引

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

            self.es.indices.create(index=self.index_name, body=DOC_INDEX_MAPPING)
            logger.info(f"Created index: {self.index_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create index {self.index_name}: {e}")
            return False

    def index_doc(self, doc: Dict) -> bool:
        """写入单条文档记录"""
        try:
            self.es.index(index=self.index_name, id=doc.get("doc_id"), body=doc)
            return True
        except Exception as e:
            logger.error(f"Failed to index doc: {e}")
            return False

    def get_doc(self, doc_id: str) -> Optional[Dict]:
        """根据 doc_id 获取文档"""
        try:
            res = self.es.get(index=self.index_name, id=doc_id)
            return res["_source"]
        except Exception as e:
            logger.error(f"Failed to get doc {doc_id}: {e}")
            return None

    def delete_doc(self, doc_id: str) -> bool:
        """删除单条文档"""
        try:
            self.es.delete(index=self.index_name, id=doc_id)
            return True
        except Exception as e:
            logger.error(f"Failed to delete doc {doc_id}: {e}")
            return False

    def delete_by_doc_id(self, doc_id: str) -> bool:
        """按 doc_id 字段删除文档记录"""
        try:
            self.es.delete_by_query(
                index=self.index_name,
                body={"query": {"term": {"doc_id": doc_id}}}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete docs for doc_id {doc_id}: {e}")
            return False

    def update_kb_id_by_doc_id(self, doc_id: str, new_kb_id: str) -> Optional[int]:
        """按 doc_id 更新文档记录的 kb_id 字段"""
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
            logger.error(f"Failed to update kb_id for doc_id {doc_id}: {e}")
            return None

    def update_kb_id_by_fd_ids(self, fd_ids: List[str], new_kb_id: str) -> Optional[int]:
        """按 fd_id 批量更新文档记录的 kb_id 字段"""
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
            logger.error(f"Failed to update kb_id for fd_ids {fd_ids}: {e}")
            return None

    def search_docs(self, kb_id: str = None, doc_type: str = None,
                    industry_tag: str = None, status: int = None,
                    keyword: str = None, size: int = 50) -> List[Dict]:
        """按条件搜索文档"""
        filters = []
        if kb_id:
            filters.append({"term": {"kb_id": kb_id}})
        if doc_type:
            filters.append({"term": {"doc_type": doc_type}})
        if industry_tag:
            filters.append({"term": {"industry_tag": industry_tag}})
        if status is not None:
            filters.append({"term": {"status": status}})

        query = {"bool": {"must": filters}} if filters else {"match_all": {}}

        if keyword:
            query = {
                "bool": {
                    "must": filters + [{"match": {"title": keyword}}]
                }
            }

        body = {"size": size, "query": query, "sort": [{"create_time": {"order": "desc"}}]}
        res = self.es.search(index=self.index_name, body=body)
        return [hit["_source"] for hit in res["hits"]["hits"]]
