#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 MiniDeepDocParser 解析器
"""

import os
import json
from deepdoc.MiniDeepDocParser import MiniDeepDocParser


def test_parse_pdf():
    """
    测试解析 PDF 文件
    """
    # PDF 文件路径
    pdf_path = r"d:\python\doc_parser_service\pdf\低空经济__开启立体城市空间新未来_劳莘.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"PDF 文件不存在: {pdf_path}")
        return
    
    print(f"开始解析 PDF 文件: {pdf_path}")
    
    # 读取 PDF 文件
    with open(pdf_path, 'rb') as f:
        pdf_data = f.read()
    
    # 创建解析器
    parser = MiniDeepDocParser(
        lang='ch',          # 中文
        enable_layout=True, # 启用版面分析
        enable_table=False, # 禁用表格识别，避免崩溃
        dpi=150,            # 图片分辨率
        model_dir=r"d:\python\doc_parser_service\models\paddle-orc",  # 自定义模型目录
    )
    
    # 定义进度回调
    def on_progress(progress, message):
        print(f"[{progress*100:.1f}%] {message}")
    
    # 执行解析
    boxes, markdown = parser.parse(pdf_data, callback=on_progress)
    
    print(f"\n解析完成！共识别 {len(boxes)} 个文本块")
    
    # 打印前 10 个块的信息
    print("\n前 10 个文本块:")
    for i, box in enumerate(boxes[:10]):
        print(f"{i+1}. [Page {box.page_number}, {box.layout_type}] {box.text[:50]}...")
    
    # 输出 Markdown 到文件
    output_dir = os.path.dirname(pdf_path)
    output_name = os.path.basename(pdf_path).replace('.pdf', '_output.md')
    output_path = os.path.join(output_dir, output_name)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown)
    print(f"\nMarkdown 输出已保存到: {output_path}")
    
    # 输出 JSON 到文件
    json_name = os.path.basename(pdf_path).replace('.pdf', '_output.json')
    json_path = os.path.join(output_dir, json_name)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(parser.get_boxes_json(), f, ensure_ascii=False, indent=2)
    print(f"JSON 输出已保存到: {json_path}")
    
    return boxes, markdown


if __name__ == "__main__":
    test_parse_pdf()
