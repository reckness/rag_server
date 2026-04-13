#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to generate flattened JSON from the tree structure
"""

import os
import json

# Input and output paths
input_file = r"D:\python\deep-search-main\app_services\doc_parser_service\rag\json\珠三角电子信息产业集群创新网络演化及其机理研究_王炜.pdf.json"
output_dir = r"D:\python\deep-search-main\app_services\doc_parser_service\rag\json2"
output_file = os.path.join(output_dir, "珠三角电子信息产业集群创新网络演化及其机理研究_王炜_flat.json")

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

def clean_text(text):
    """Clean text by removing specific patterns and replacing newlines"""
    if "【原始数据内容】:" in text:
        text = text.split("【原始数据内容】:")[-1]
    return text.replace("\n", " ").strip()

def determine_section_hint(title, content):
    """Determine section hint based on title and content"""
    title_lower = title.lower() if title else ""
    content_lower = content.lower() if content else ""

    # Based on title
    if any(x in title_lower for x in ["content", "目录", "index", "table of contents"]):
        return "目录列表"
    elif any(x in title_lower for x in ["intro", "引言", "概述", "summary", "abstract"]):
        return "引言/概述"
    elif any(x in title_lower for x in ["conclusion", "结论", "结语"]):
        return "结论"
    elif any(x in title_lower for x in ["glossary", "术语", "definition"]):
        return "术语表/定义"
    elif any(x in title_lower for x in ["appendix", "appendices", "附录"]):
        return "附录"
    elif any(x in title_lower for x in ["reference", "参考"]):
        return "参考文献"
    elif any(x in title_lower for x in ["disclaimer", "声明", "legal", "copyright"]):
        return "法律声明"

    # Based on content
    if "table" in content_lower and ("row" in content_lower or "col" in content_lower):
        return "数据表格"
    elif any(x in content_lower for x in ["warning", "caution", "danger", "警告", "注意", "危险"]):
        return "安全警告"
    elif any(x in content_lower for x in ["step 1", "step 2", "步骤", "procedure", "流程"]):
        return "操作流程"
    elif any(x in content_lower for x in ["spec", "specification", "参数", "规格"]):
        return "技术规格"

    return "正文章节"

def generate_embedding_text(path_list, content, hint):
    """Generate embedding text"""
    embedding_text = f"""
    标题: {path_list[-1] if path_list else '根节点'}
    章节: {' > '.join(path_list)}
    类型: {hint}
    内容: {content[:500]}...
    """
    return embedding_text

def recursive_walk(nodes, path, depth, doc_title):
    """Recursively walk through the tree structure and generate flattened nodes"""
    flat_nodes = []
    processed_count = 0
    current_position = 0

    def _walk(nodes, path, depth):
        nonlocal processed_count, current_position
        for node in nodes:
            # 1. Get basic information
            current_title = node.get("title", "Untitled").replace('\n', ' ').strip()
            current_path = path + [current_title]
            
            # Get content, prioritize text, then content, finally empty
            original_text = node.get("text", node.get("content", "")).strip()
            
            # 2. Calculate page_num_int array
            # Use start_index and end_index from input JSON
            start_index = node.get("start_index")
            end_index = node.get("end_index") + 1 if node.get("end_index") is not None else start_index + 1
            # Generate page_num_int array
            page_num_int = list(range(start_index, end_index)) if start_index is not None else []
            # Update current position
            if end_index is not None:
                current_position = end_index
            
            # 3. Process Node ID (use existing if available, otherwise generate)
            node_id = node.get("node_id", f"gen_id_{processed_count:05d}")

            # 4. Generate section hint
            section_hint = determine_section_hint(current_title, original_text)

            # 5. Generate embedding text
            display_text = original_text if original_text else "(无正文内容，仅作为章节标题存在)"
            # Clean text to ensure quality
            cleaned_display_text = clean_text(display_text)
            embedding_text = generate_embedding_text(current_path, cleaned_display_text, section_hint)

            # Check if this is a leaf node (no children)
            is_leaf = not ("nodes" in node and isinstance(node["nodes"], list) and len(node["nodes"]) > 0)
            
            # Only add leaf nodes to the flat list
            if is_leaf:
                # 6. Build final data object
                rag_item = {
                    "doc_id": "",  # Empty for now
                    "kb_id": "",  # Empty for now
                    "fd_id": "",  # Empty for now
                    "doc_title": doc_title,
                    "embedding_text": embedding_text,
                    "page_num_int": page_num_int,
                    "section_hint": section_hint,
                    "section_id": node_id,
                    "section_path": current_path,
                    "original_snippet": original_text,
                    "embedding": []  # Empty for now
                }

                flat_nodes.append(rag_item)
                processed_count += 1

            # 7. Recursively process child nodes
            if "nodes" in node and isinstance(node["nodes"], list):
                _walk(node["nodes"], current_path, depth + 1)

    _walk(nodes, path, depth)
    return flat_nodes

def main():
    """Main function"""
    print(f"Processing input file: {input_file}")
    
    # Load input JSON
    try:
        with open(input_file, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        print("[OK] Input JSON loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load input JSON: {e}")
        return
    
    # Extract document title
    doc_title = data.get("doc_name", "Unknown Document")
    print(f"Document title: {doc_title}")
    
    # Extract structure
    structure = data.get("structure", [])
    print(f"Structure length: {len(structure)}")
    
    # Generate flattened nodes
    print("Generating flattened JSON...")
    flat_nodes = recursive_walk(structure, [], 1, doc_title)
    print(f"Generated {len(flat_nodes)} flattened nodes")
    
    # Save to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(flat_nodes, f, ensure_ascii=False, indent=2)
        print(f"[OK] Flattened JSON saved to: {output_file}")
    except Exception as e:
        print(f"[ERROR] Failed to save output JSON: {e}")
        return

if __name__ == "__main__":
    main()
