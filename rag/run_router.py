#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
临时脚本，用于运行 DocumentRouter
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.build_router_es import DocumentRouter

if __name__ == "__main__":
    input_path = r"d:\xy\Python\deep-search\rag\vector_RAG_我国低空经济进入“载人时代”_经济热点分析 2025第14期.pdf_full.json"
    doc_id = "doc_低空经济"
    kb_id = "kb_低空经济"
    fd_id = "fd_低空经济"
    
    router = DocumentRouter(input_path, doc_id, kb_id, fd_id)
    success = router.run()
    
    if not success:
        sys.exit(1)
