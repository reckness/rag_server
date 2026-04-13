#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for document processing pipeline
"""

import os
import sys
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock the ES connection to avoid actual connection attempts
class MockESConnection:
    def get_conn(self):
        return None

# Mock the ES_CONN before importing modules that use it
sys.modules['common.doc_store.es_conn_pool'] = type('obj', (object,), {'ES_CONN': MockESConnection()})()

# Test file path
test_file = "rag/json/珠三角电子信息产业集群创新网络演化及其机理研究_王炜.pdf.json"

def test_json_parsing():
    """Test JSON parsing"""
    print("Testing JSON parsing...")
    
    try:
        with open(test_file, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        
        print("[OK] JSON file loaded successfully")
        print(f"  Document name: {data.get('doc_name', 'Unknown')}")
        print(f"  Structure length: {len(data.get('structure', []))}")
        
        # Test structure parsing
        if 'structure' in data:
            structure = data['structure']
            print(f"  First structure item: {structure[0].get('title', 'No title')}")
            print(f"  First structure item text length: {len(structure[0].get('text', ''))}")
        
        return True
    except Exception as e:
        print(f"[ERROR] JSON parsing failed: {e}")
        return False

def test_document_router_logic():
    """Test document router logic without ES"""
    print("\nTesting document router logic...")
    
    try:
        # Import here after mocking
        from rag.build_router_es import build_summary
        
        # Load the JSON
        with open(test_file, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        
        # Extract structure
        structure = data.get('structure', [])
        
        # Create a simple flat structure for testing
        flat_nodes = []
        def extract_nodes(nodes, path=[]):
            for node in nodes:
                title = node.get('title', 'Untitled')
                text = node.get('text', '')
                current_path = path + [title]
                
                # Create a mock node similar to what json_to_es_converter produces
                flat_nodes.append({
                    'embedding_text': f"标题: {title}\n章节: {' > '.join(current_path)}\n内容: {text[:100]}...",
                    'section_hint': '正文章节',
                    'section_path': current_path,
                    'metadata': {'depth': len(path)}
                })
                
                # Recurse
                if 'nodes' in node:
                    extract_nodes(node['nodes'], current_path)
        
        extract_nodes(structure)
        
        # Test build_summary
        summary_info = build_summary(flat_nodes)
        print("[OK] Document router logic test passed")
        print(f"  Generated summary length: {len(summary_info['summary'])}")
        print(f"  Generated keywords: {summary_info['keywords']}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Document router logic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Starting document processing pipeline test...")
    print(f"Test file: {test_file}")
    
    # Test JSON parsing
    parsing_success = test_json_parsing()
    
    # Test document router logic
    router_success = test_document_router_logic()
    
    # Summary
    print("\nTest Summary:")
    print(f"JSON parsing: {'PASS' if parsing_success else 'FAIL'}")
    print(f"Document router logic: {'PASS' if router_success else 'FAIL'}")
    
    if parsing_success and router_success:
        print("\n All tests passed!")
        return 0
    else:
        print("\n Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
