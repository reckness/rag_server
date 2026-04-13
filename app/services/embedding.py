import asyncio
from functools import lru_cache


@lru_cache(maxsize=10000)
def cache_embedding(text):
    """缓存embedding结果"""
    from common.nlp.embedding_client import get_embedding
    return get_embedding(text)


async def get_embedding(text: str):
    """异步获取embedding"""
    return await asyncio.to_thread(cache_embedding, text)