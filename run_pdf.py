"""
调用 rag/page_index.py 处理 PDF 文档
"""
import asyncio
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.page_index import page_index

PDF_PATH = os.path.join("pdf", "低空经济__开启立体城市空间新未来_劳莘.pdf")

async def main():
    print(f"开始处理: {PDF_PATH}")
    result_path = await page_index(
        doc=PDF_PATH,
        if_add_node_id="yes",
        if_add_node_summary="yes",
        if_add_node_text="yes",
    )
    print(f"\n处理完成！结果保存在: {result_path}")
    
    # 打印树结构预览
    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    
    print(f"\n文档名: {result.get('doc_name', '')}")
    print(f"结构预览:")
    
    def print_tree(nodes, indent=0):
        for node in nodes:
            title = node.get("title", "")
            pages = f"[p{node.get('start_index', '?')}-{node.get('end_index', '?')}]"
            summary = node.get("summary", "")
            summary_str = f" — {summary[:50]}..." if summary else ""
            print("  " * indent + f"├── {title} {pages}{summary_str}")
            if node.get("nodes"):
                print_tree(node["nodes"], indent + 1)
    
    print_tree(result.get("structure", []))

if __name__ == "__main__":
    asyncio.run(main())
