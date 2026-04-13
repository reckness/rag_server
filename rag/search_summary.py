#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检索脚本，用于查询与特定关键词相关的文档
"""

import sys
import os
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从配置文件导入配置
from common.config import EMBEDDING_DIM
from common.doc_store.es_conn_pool import ES_CONN
from common.nlp.embedding_client import get_embedding

def search_summary(query):
    """搜索与指定关键词相关的文档"""
    # 获取 Elasticsearch 客户端
    es_client = ES_CONN.get_conn()
    
    # 获取查询向量
    query_embedding = get_embedding(query)
    
    # 构建混合检索查询（关键词+向量）
    search_body = {
        "size": 10,
        "query": {
            "bool": {
                "should": [
                    {
                        "match": {
                            "routing_text": {
                                "query": query,
                                "boost": 2
                            }
                        }
                    },
                    {
                        "match": {
                            "title": {
                                "query": query,
                                "boost": 3
                            }
                        }
                    },
                    {
                        "match": {
                            "summary": {
                                "query": query,
                                "boost": 1.5
                            }
                        }
                    }
                ]
            }
        },
        "highlight": {
            "fields": {
                "routing_text": {},
                "title": {},
                "summary": {}
            }
        }
    }
    
    # 执行搜索
    try:
        response = es_client.search(
            index="doc_summary_index",
            body=search_body
        )
        
        # 处理搜索结果
        print(f"找到 {response['hits']['total']['value']} 个相关文档:")
        print("-" * 80)
        
        for hit in response['hits']['hits']:
            source = hit['_source']
            print(f"文档标题: {source.get('title', '未知')}")
            print(f"文档ID: {source.get('doc_id', '未知')}")
            print(f"相似度得分: {hit['_score']}")
            print("-" * 80)
            
    except Exception as e:
        print(f"搜索失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="搜索与指定关键词相关的文档")
    parser.add_argument("query", help="搜索关键词")
    args = parser.parse_args()
    
    search_summary(args.query)
