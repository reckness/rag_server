#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Embedding Client for Vector Models
统一管理向量模型的连接和引用
"""

import requests
import urllib3
from common.config import API_URL, MODEL_NAME, EMBEDDING_DIM

class EmbeddingClient:
    """向量模型客户端，用于获取文本的向量嵌入"""
    
    def __init__(self):
        """初始化嵌入客户端"""
        self.api_url = API_URL
        self.model_name = MODEL_NAME
        self.embedding_dim = EMBEDDING_DIM
    
    def get_embedding(self, text: str) -> list:
        """获取文本的向量嵌入（使用本地 qwen-embedding 模型）"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer EMPTY"
        }
        payload = {
            "model": self.model_name,
            "input": [text[:8000]],  # 限制文本长度
            "dimensions": self.embedding_dim
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            if "data" in data and len(data["data"]) > 0:
                return data["data"][0]["embedding"]
            return []
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] 获取嵌入异常: {str(e)}")
        except Exception as e:
            print(f"[ERROR] 处理响应失败: {str(e)}")
        return []

# 创建全局嵌入客户端实例
EMBEDDING_CLIENT = EmbeddingClient()

# 便捷函数
def get_embedding(text: str) -> list:
    """获取文本的向量嵌入"""
    return EMBEDDING_CLIENT.get_embedding(text)
