import asyncio
import httpx

from common.config import RERANK_TIMEOUT, RERANK_URL


async def rerank(query, chunks, model=None):
    """异步rerank，通过远程 HTTP API 调用"""
    try:
        if not chunks:
            return chunks

        documents = [
            c.get("embedding_text") or c.get("chunk_text") or ""
            for c in chunks
        ]
        if not any(documents):
            chunks.sort(key=lambda x: x.get("_score", 0), reverse=True)
            return chunks

        payload = {
            "query": query,
            "documents": documents,
        }

        async with httpx.AsyncClient(timeout=RERANK_TIMEOUT) as client:
            resp = await client.post(RERANK_URL, json=payload)
            resp.raise_for_status()
            result = resp.json()

        # 解析返回结果，兼容多种 rerank API 格式
        if "results" in result:
            results_list = result["results"]
            if results_list and "index" in results_list[0]:
                # 格式1: {"results": [{"index": 0, "relevance_score": 0.9}, ...]}
                for item in results_list:
                    idx = item["index"]
                    score = item.get("relevance_score", item.get("score", 0.0))
                    chunks[idx]["rerank_score"] = float(score)
            else:
                # 格式2: {"results": [{"document": "...", "score": 8.69}, ...]}
                # 结果按分数排序返回，通过 document 文本匹配回原始 chunks
                doc_to_score = {}
                for item in results_list:
                    doc_text = item.get("document", "")
                    score = item.get("relevance_score", item.get("score", 0.0))
                    doc_to_score[doc_text] = float(score)
                for c in chunks:
                    doc_text = c.get("embedding_text") or c.get("chunk_text") or ""
                    c["rerank_score"] = doc_to_score.get(doc_text, 0.0)
        elif "scores" in result:
            # 格式3: {"scores": [0.9, 0.8, ...]}
            for i, score in enumerate(result["scores"]):
                chunks[i]["rerank_score"] = float(score)
        else:
            print(f"Rerank: 未识别的返回格式: {list(result.keys())}")
            chunks.sort(key=lambda x: x.get("_score", 0), reverse=True)
            return chunks

        chunks.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        return chunks
    except Exception as e:
        # 处理其他异常
        print(f"Rerank error: {e}")
        # 回退到按分数排序
        chunks.sort(key=lambda x: x.get("_score", 0), reverse=True)
        return chunks