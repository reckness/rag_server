from typing import List, Dict, Any
import asyncio
from functools import lru_cache
from .es_service import ESService
from .embedding import get_embedding
from .rerank import rerank
from .folder_service import FolderService
from common.doc_store.es_conn_pool import ES_CONN
from common.config import ELASTICSEARCH_INDEX


class RagService:

    def __init__(self):
        self.es = ESService()

    async def search(self, req, db, model=None):
        query = req.query
        kb_ids = req.kb_ids
        fd_ids = req.fd_ids
        topk = req.topk or 5

        # 检查topk是否有效
        if topk <= 0:
            topk = 5
        #处理fd_ids
        if fd_ids:
            if isinstance(fd_ids, str):
                fd_ids = [fd_ids]
            # 获取所有子文件夹
            subfolders = FolderService.get_all_subfolders_by_ids(db, fd_ids)
            # 将Folder列表转换为fd_id列表
            subfolder_ids = [folder.fd_id for folder in subfolders]
            # 合并原始fd_ids和子文件夹fd_ids
            fd_ids = subfolder_ids

        # 1️⃣ embedding
        query_vec = await get_embedding(query)

        # 2️⃣ 文档路由（粗召回）
        doc_ids = self.es.retrieve_docs(query_vec, kb_ids, fd_ids)

        if not doc_ids:
            return {"chunks": [], "context": ""}

        # 3️⃣ chunk召回（混合检索）
        chunks = self.es.retrieve_chunks(query, query_vec, doc_ids)

        # 4️⃣ rerank（可选）
        if req.use_rerank:
            try:
                chunks = await rerank(query, chunks, model)
            except ImportError as e:
                # 处理缺少依赖的情况
                print(f"Rerank not available: {e}")
                # 回退到按分数排序
                chunks.sort(key=lambda x: x.get("_score", 0), reverse=True)
                chunks = chunks[:10]  # 限制为10个
        else:
            # 按分数排序
            chunks.sort(key=lambda x: x.get("_score", 0), reverse=True)
            chunks = chunks[:10]  # 限制为10个

        # 5️⃣ 上下文构建
        context = self._build_context(chunks, topk)

        # 移除不需要的字段，减少返回数据大小
        filtered_chunks = []
        for chunk in chunks[:topk]:
            filtered_chunk = {}
            for key, value in chunk.items():
                if key not in ["embedding_text", "embedding"]:
                    filtered_chunk[key] = value
            filtered_chunks.append(filtered_chunk)

        return {
            "query": query,
            "doc_ids": doc_ids,
            "chunks": filtered_chunks,
            "context": context
        }

    def _build_context(self, chunks, topk=5):
        """构建上下文"""
        selected = chunks[:topk]

        context_blocks = []

        for c in selected:
            text = c.get("original_snippet", "")
            section = " > ".join(c.get("section_path", []))

            block = f"""
                【文档】{c.get("doc_title")}
                【章节】{section}
                【内容】{text}
                """
            context_blocks.append(block)

        return "\n\n".join(context_blocks)

    def expand_neighbors(self, chunk, window=1):
        """扩展相邻页"""
        pages = chunk.get("page_num_int", [])

        expand_pages = []
        for p in pages:
            expand_pages.extend([p-1, p, p+1])

        # 过滤掉负数页码
        return [p for p in list(set(expand_pages)) if p > 0]

    def retrieve_expanded_chunks(self, query, query_vec, chunk, kb_ids, fd_ids=None, topk=10, db=None):
        """检索扩展的相邻页chunk"""
        expand_pages = self.expand_neighbors(chunk)
        doc_id = chunk.get("doc_id")

        # 构建查询
        body = {
            "size": topk,
            "query": {
                "bool": {
                    "must": [
                        {"term": {"doc_id": doc_id}}
                    ]
                }
            }
        }

        # 处理kb_ids
        if kb_ids:
            if isinstance(kb_ids, list):
                body["query"]["bool"]["must"].append({"terms": {"kb_id": kb_ids}})
            else:
                body["query"]["bool"]["must"].append({"term": {"kb_id": kb_ids}})

        # 处理fd_ids
        if fd_ids:
            if isinstance(fd_ids, str):
                fd_ids = [fd_ids]
            # 获取所有子文件夹
            if db:
                subfolders = FolderService.get_all_subfolders_by_ids(db, fd_ids)
                # 将Folder列表转换为fd_id列表
                subfolder_ids = [folder.fd_id for folder in subfolders]
                # 合并原始fd_ids和子文件夹fd_ids
                fd_ids = list(set(fd_ids + subfolder_ids))
            if isinstance(fd_ids, list):
                body["query"]["bool"]["must"].append({"terms": {"fd_id": fd_ids}})
            else:
                body["query"]["bool"]["must"].append({"term": {"fd_id": fd_ids}})

        # 添加页码过滤
        body["query"]["bool"]["must"].append({"terms": {"page_num_int": expand_pages}})

        # 执行查询
        res = self.es.es.search(index=ELASTICSEARCH_INDEX, body=body)

        expanded_chunks = []
        for hit in res["hits"]["hits"]:
            source = hit["_source"]
            source["_score"] = hit["_score"]
            expanded_chunks.append(source)

        return expanded_chunks