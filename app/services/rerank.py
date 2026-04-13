import asyncio


async def rerank(query, chunks, model=None):
    """异步rerank"""
    try:
        if model is None:
            # 模型加载失败时回退到按分数排序
            chunks.sort(key=lambda x: x.get("_score", 0), reverse=True)
            return chunks

        pairs = [(query, c["embedding_text"]) for c in chunks]

        scores = await asyncio.to_thread(model.predict, pairs)

        for i, c in enumerate(chunks):
            c["rerank_score"] = float(scores[i])

        chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

        return chunks
    except Exception as e:
        # 处理其他异常
        print(f"Rerank error: {e}")
        # 回退到按分数排序
        chunks.sort(key=lambda x: x.get("_score", 0), reverse=True)
        return chunks