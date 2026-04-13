#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 MiniDeepDocParser 解析器（简单版）
"""

import os
from deepdoc.MiniDeepDocParser import MiniDeepDocParser


def test_parser_initialization():
    """
    测试 MiniDeepDocParser 的初始化
    """
    print("开始测试 MiniDeepDocParser 初始化...")
    
    try:
        # 创建解析器
        parser = MiniDeepDocParser(
            lang='ch',          # 中文
            enable_layout=True, # 启用版面分析
            enable_table=True,  # 启用表格识别
            dpi=150,            # 图片分辨率
        )
        print("MiniDeepDocParser 初始化成功！")
        return True
    except Exception as e:
        print(f"MiniDeepDocParser 初始化失败: {e}")
        return False


if __name__ == "__main__":
    test_parser_initialization()
