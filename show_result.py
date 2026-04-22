"""查看 PDF 处理结果"""
import json, tempfile, os, glob

f = glob.glob(os.path.join(tempfile.gettempdir(), '*.pdf.json'))[-1]
data = json.load(open(f, 'r', encoding='utf-8'))

doc_name = data.get('doc_name', '')
structure = data.get('structure', [])

print(f"文档名: {doc_name}")
print(f"顶层节点数: {len(structure)}")
print(f"结果文件: {f}")
print(f"文件大小: {os.path.getsize(f) / 1024:.1f} KB")
print()

def show_tree(nodes, indent=0):
    for node in nodes:
        title = node.get('title', '')
        start = node.get('start_index', '?')
        end = node.get('end_index', '?')
        summary = node.get('summary', '')
        summary_short = (summary[:60] + '...') if summary and len(summary) > 60 else summary
        children = node.get('nodes', [])
        prefix = '  ' * indent + '├── '
        print(f"{prefix}{title} [p{start}-{end}]  子节点:{len(children)}")
        if summary_short:
            print(f"{'  ' * indent}     摘要: {summary_short}")
        if children:
            show_tree(children, indent + 1)

show_tree(structure)
