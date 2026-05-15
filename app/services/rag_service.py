from typing import List, Dict, Any
import math
import asyncio
from functools import lru_cache
from .es_service import ESService
from .embedding import get_embedding
from .rerank import rerank
from .folder_service import FolderService
from common.doc_store.es_conn_pool import ES_CONN
from common.config import ELASTICSEARCH_INDEX, ELASTICSEARCH_CHUNK_INDEX


def sigmoid(x: float) -> float:
    """Sigmoid 函数，将任意实数映射到 (0, 1)"""
    return 1.0 / (1.0 + math.exp(-x))


def minmax_normalize_scores(scores: List[float], max_value: float = 0.90) -> List[float]:
    """将分数列表归一化到 [0, max_value]。若分数相同，返回全 max_value。"""
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [max_value] * len(scores)
    return [((s - min_score) / (max_score - min_score)) * max_value for s in scores]


class ScoreNormalizer:
    def __init__(self, method='minmax'):
        """
        method: 'minmax', 'zscore', 'percentile'
        """
        self.method = method
        
    def normalize(self, results: List[Dict]) -> List[Dict]:
        """归一化搜索结果中的 _score"""
        
        if not results:
            return results
            
        # 提取所有分数
        scores = [hit['_score'] for hit in results]
        
        if self.method == 'minmax':
            normalized_scores = self._minmax_normalize(scores)
        elif self.method == 'zscore':
            normalized_scores = self._zscore_normalize(scores)
        elif self.method == 'percentile':
            normalized_scores = self._percentile_normalize(scores)
        else:
            normalized_scores = scores
            
        # 更新结果中的分数，将归一化后的分数直接赋值给 _score
        for hit, norm_score in zip(results, normalized_scores):
            hit['_score'] = norm_score
            
        return results
    
    def _minmax_normalize(self, scores: List[float]) -> List[float]:
        """Min-Max 归一化到 [0, 1]"""
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
            
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def _zscore_normalize(self, scores: List[float]) -> List[float]:
        """Z-Score 归一化"""
        import statistics
        
        mean = statistics.mean(scores)
        stdev = statistics.stdev(scores) if len(scores) > 1 else 1.0
        
        if stdev == 0:
            return [0.0] * len(scores)
            
        # 使用 tanh 将结果映射到 [-1, 1] 范围
        zscores = [(s - mean) / stdev for s in scores]
        return [max(-1.0, min(1.0, z)) for z in zscores]
    
    def _percentile_normalize(self, scores: List[float]) -> List[float]:
        """百分位归一化（基于排名）"""
        sorted_scores = sorted(scores)
        score_to_percentile = {}
        
        for i, score in enumerate(sorted_scores):
            percentile = i / (len(sorted_scores) - 1) if len(sorted_scores) > 1 else 1.0
            score_to_percentile[score] = percentile
            
        return [score_to_percentile[score] for score in scores]


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

        # 2️⃣ 第一重：doc_index 混合召回（宽松，有就召回）
        doc_ids = self.es.retrieve_docs(query, kb_ids, fd_ids, query_vec, topk=20)

        if not doc_ids:
            return {"chunks": [], "context": ""}

        # 3️⃣ 直接 chunk 召回（跳过 chapter 级，消除漏斗瓶颈）
        chunks = self.es.retrieve_chunks_by_docs(query, query_vec, doc_ids, kb_ids)

        # 5️⃣ rerank + Sigmoid 映射最终 _score
        if req.use_rerank:
            try:
                chunks.sort(key=lambda x: x.get("_score", 0), reverse=True)
                chunks = chunks[:30]
                chunks = await rerank(query, chunks, model)

                rerank_scores = []
                for c in chunks:
                    s = c.get("rerank_score")
                    if isinstance(s, (int, float)):
                        rerank_scores.append(float(s))

                # 只有存在有效 rerank_score 时才按 rerank 分数映射。
                # 否则回退到 RRF 分数，避免所有结果被映射为 0.5。
                if rerank_scores:
                    normalized = minmax_normalize_scores(rerank_scores)
                    idx = 0
                    for c in chunks:
                        s = c.get("rerank_score")
                        if isinstance(s, (int, float)):
                            c["_score"] = normalized[idx]
                            idx += 1
                        else:
                            c["_score"] = 0.0
                    chunks.sort(key=lambda x: x.get("_score", 0), reverse=True)
                else:
                    chunks.sort(key=lambda x: x.get("_score", 0), reverse=True)

                chunks = chunks[:10]
            except Exception as e:
                print(f"Rerank not available: {e}")
                # 回退到 RRF 排序
                chunks.sort(key=lambda x: x.get("_score", 0), reverse=True)
                chunks = chunks[:10]
        else:
            # 无 rerank，按 RRF 分数排序
            chunks.sort(key=lambda x: x.get("_score", 0), reverse=True)
            chunks = chunks[:10]

        # 5️⃣ 上下文构建
        context = self._build_context(chunks, topk)

        # 移除不需要的字段，减少返回数据大小
        filtered_chunks = []
        for chunk in chunks[:topk]:
            filtered_chunk = {}
            for key, value in chunk.items():
                if key not in ["embedding"]:
                    filtered_chunk[key] = value
            filtered_chunks.append(filtered_chunk)

        # 只保留实际有 chunk 命中的 doc_id
        hit_doc_ids = list(dict.fromkeys(
            c.get("doc_id") for c in filtered_chunks if c.get("doc_id")
        ))

        return {
            "query": query,
            "doc_ids": hit_doc_ids,
            "chunks": filtered_chunks,
            "context": context
        }

    def _build_context(self, chunks, topk=5):
        """构建上下文"""
        selected = chunks[:topk]

        context_blocks = []

        for c in selected:
            text = c.get("chunk_text", "")
            chapter = c.get("chapter_title", "")
            section = c.get("section_title", "")

            block = f"""
                【文档】{c.get("doc_title")}
                【章节】{chapter} > {section}
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
        res = self.es.es.search(index=ELASTICSEARCH_CHUNK_INDEX, body=body)

        expanded_chunks = []
        for hit in res["hits"]["hits"]:
            source = hit["_source"]
            source["_score"] = hit["_score"]
            expanded_chunks.append(source)

        return expanded_chunks