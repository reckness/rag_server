import os
import re
import ssl
import json
import time
import copy
import asyncio
import logging
import requests
import urllib3
import yaml
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace as config

import tiktoken
import PyPDF2
from dotenv import load_dotenv

# 1. Network & Environment Config
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

# 2. API Config
load_dotenv()
# Check environment variable first, fallback to hardcoded
CHATGPT_API_KEY = os.getenv("API_KEY", "sk-2bf2356e4ccb4a6ea43b7ab433940c10")
API_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"

# --- Universal Fallback Object (Crash Preventer) ---
class UniversalFallback(dict):
    """
    Acts as both a dictionary (for toc detection) and a list (for toc generation).
    Prevents crashes when API returns HTML or fails.
    """
    def __init__(self):
        super().__init__()
        self['toc_detected'] = 'no' # Default safe value
        self['page_index_given_in_toc'] = 'no'
        self['completed'] = 'no'
    
    def extend(self, other):
        pass 
    
    def __iter__(self):
        return iter([])
    
    def __len__(self):
        return 0
        
    def get(self, key, default=None):
        return super().get(key, default)

def request_api_stream_sync(model, messages, timeout=180):
    target_model = "deepseek-chat" # 使用正确的模型名称
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CHATGPT_API_KEY}",
        "Accept": "application/json",
        "User-Agent": "PostmanRuntime/7.26.8"
    }
    
    payload = {
        "model": target_model,
        "messages": messages,
        "stream": True,
        "temperature": 0.1
    }
    
    urls_to_try = [
        "https://api.deepseek.com/v1/chat/completions"
    ]

    for url in urls_to_try:
        try:
            response = requests.post(
                url, 
                headers=headers, 
                json=payload, 
                timeout=timeout, 
                verify=False,
                stream=True 
            )
            
            if "text/html" in response.headers.get("Content-Type", ""):
                logging.warning(f"⚠️ URL {url} returned HTML (Login Page). Skipping...")
                continue
            
            # 401 Handling
            if response.status_code == 401:
                logging.error(f"⚠️ URL {url} failed with 401 Unauthorized. Check your API KEY.")
                continue

            if response.status_code != 200:
                logging.warning(f"⚠️ URL {url} failed with {response.status_code}")
                continue

            full_content = ""
            for line in response.iter_lines():
                if not line: continue
                line_str = line.decode('utf-8').strip()
                if line_str.startswith("data:"):
                    data_part = line_str[5:].strip()
                    if data_part == "[DONE]": break
                    try:
                        data_json = json.loads(data_part)
                        delta = data_json['choices'][0].get('delta', {})
                        if 'content' in delta:
                            content_str = delta['content']
                            full_content += content_str
                            
                            # === 【新增功能】 实时将字符输出到 stdout 供 GUI 捕获 ===
                            # DEBUG_AI_CHAR 是 pgui.py 识别的特殊标记
                            # 注释掉 DEBUG_AI_CHAR 输出，避免在命令行运行时产生大量输出
                            # print(f"DEBUG_AI_CHAR:{content_str}", flush=True)
                            # ===================================================

                    except: continue
            
            if full_content:
                return full_content
        except Exception as e:
            logging.error(f"Connection error to {url}: {e}")
            continue

    return "Error"

# --- Framework Adapters ---

def clean_deepseek_content(content):
    if not content or not isinstance(content, str): return ""
    # Remove thinking process tags
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    return content.strip()

def ChatGPT_API_with_finish_reason(model, prompt, api_key=None, chat_history=None):
    global global_token_count
    messages = chat_history + [{"role": "user", "content": prompt}] if chat_history else [{"role": "user", "content": prompt}]
    
    # 计算输入 token 数量
    input_text = "".join([msg["content"] for msg in messages])
    global_token_count['input'] += count_tokens(input_text, model)
    
    for i in range(3):
        raw = request_api_stream_sync(model, messages)
        if raw != "Error" and raw.strip():
            # 计算输出 token 数量
            output_text = clean_deepseek_content(raw)
            global_token_count['output'] += count_tokens(output_text, model)
            return output_text, "finished"
        print(f'************* API Retry ({i+1}) *************')
        time.sleep(3)
    return "Error", "failed"

def ChatGPT_API(model, prompt, api_key=None, chat_history=None):
    res, _ = ChatGPT_API_with_finish_reason(model, prompt, api_key, chat_history)
    return res

async def ChatGPT_API_async(model, prompt, api_key=None):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, ChatGPT_API, model, prompt)

def get_json_content(content):
    """Helper to extract pure JSON string from markdown code blocks"""
    if not content: return ""
    match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
    if match:
        return match.group(1)
    return content

def extract_json(content):
    if content == "Error" or not content: 
        return UniversalFallback()
    
    try:
        content = clean_deepseek_content(content)
        json_str = ""
        
        # Priority 1: Markdown code block
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Priority 2: Find outermost braces/brackets
            start_list = content.find("[")
            start_dict = content.find("{")
            
            if start_list != -1 and (start_dict == -1 or start_list < start_dict):
                start = start_list
                end = content.rfind("]")
            else:
                start = start_dict
                end = content.rfind("}")
                
            if start != -1 and end != -1:
                json_str = content[start:end+1]
        
        if not json_str: 
            return UniversalFallback()
        
        # Clean potential trailing commas or comments if needed (simple check)
        data = json.loads(json_str)
        return data

    except Exception as e:
        logging.error(f"JSON Parsing failed: {e}")
        return UniversalFallback()

# --- Helper Functions ---

# 全局 token 计数器
global_token_count = {
    'input': 0,
    'output': 0
}

def count_tokens(text, model=None):
    """计算文本的 token 数量"""
    if not text:
        return 0
    try:
        # 尝试使用 tiktoken 计算 token 数量
        if model and model.startswith('gpt'):
            encoding = tiktoken.encoding_for_model(model)
        else:
            # 默认为 cl100k_base 编码
            encoding = tiktoken.get_encoding('cl100k_base')
        return len(encoding.encode(text))
    except:
        #  fallback: 简单估算
        return len(text) // 2

def get_token_count():
    """获取全局 token 计数"""
    return global_token_count.copy()

def reset_token_count():
    """重置全局 token 计数"""
    global global_token_count
    global_token_count = {
        'input': 0,
        'output': 0
    }

def write_node_id(data, node_id=0):
    if isinstance(data, dict):
        data['node_id'] = str(node_id).zfill(4); node_id += 1
        for k in list(data.keys()):
            if 'nodes' in k: node_id = write_node_id(data[k], node_id)
    elif isinstance(data, list):
        for i in range(len(data)): node_id = write_node_id(data[i], node_id)
    return node_id

def get_nodes(structure):
    if isinstance(structure, dict):
        sn = copy.deepcopy(structure); sn.pop('nodes', None)
        nodes = [sn]
        for k in list(structure.keys()):
            if 'nodes' in k: nodes.extend(get_nodes(structure[k]))
        return nodes
    elif isinstance(structure, list):
        res = []
        for i in structure: res.extend(get_nodes(i))
        return res

def get_pdf_name(pdf_path):
    if hasattr(pdf_path, 'name'): return pdf_path.name
    return os.path.basename(pdf_path)

class JsonLogger:
    def __init__(self, file_path):
        name = get_pdf_name(file_path)
        self.filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("./logs", exist_ok=True); self.log_data = []
    def log(self, level, message, **kwargs):
        entry = {'message': str(message), 'level': level, 'timestamp': datetime.now().isoformat()}
        self.log_data.append(entry)
        with open(os.path.join("logs", self.filename), "w", encoding="utf-8") as f:
            json.dump(self.log_data, f, indent=2, ensure_ascii=False)
    def info(self, m): self.log("INFO", m)
    def error(self, m): self.log("ERROR", m)

def get_page_tokens(pdf_path, model=None):
    page_list = []
    reader = PyPDF2.PdfReader(pdf_path)
    for page in reader.pages:
        t = page.extract_text() or ""
        page_list.append((t, len(t)))
    return page_list

def list_to_tree(data):
    nodes, roots = {}, []
    for item in data:
        c = item.get('structure')
        # Ensure indices are integers
        s_idx = int(item.get('start_index', 0))
        e_idx = int(item.get('end_index', 0))
        
        node = {
            'title': item.get('title'), 
            'start_index': s_idx, 
            'end_index': e_idx, 
            'nodes': []
        }
        
        # Use node_id if available as fallback key
        key = str(c) if c else str(item.get('title'))
        nodes[key] = node
        
        # Find parent
        p_c = '.'.join(str(c).split('.')[:-1]) if '.' in str(c) else None
        
        if p_c and p_c in nodes: 
            nodes[p_c]['nodes'].append(node)
        else: 
            roots.append(node)
    return roots

def post_processing(structure, end_idx):
    """
    Refined post-processing to handle page boundaries more naturally.
    Prevents 'off-by-one' errors where a section spanning pages gets cut off.
    """
    for i, item in enumerate(structure):
        # Ensure conversion to int prevents TypeError
        curr_idx = int(item.get('physical_index', 0))
        
        if i < len(structure) - 1:
            next_idx_val = int(structure[i+1]['physical_index'])
            # OPTIMIZATION: 
            # If Section A starts on Pg 1, Section B starts on Pg 2.
            # A should technically end on Pg 2 (inclusive) if text flows.
            # We set end_index to next_idx (start of next section).
            # Python range(start, end) is exclusive, so range(1, 2) is just Pg 1.
            # To include Pg 2, we need range(1, 3). So end_index should be next_idx.
            
            # Logic: If next section starts on a later page, assume current section
            # continues until that page.
            if next_idx_val > curr_idx:
                item['end_index'] = next_idx_val
            else:
                item['end_index'] = curr_idx # Same page
        else:
            # Last item goes to the end of document
            item['end_index'] = end_idx
            
        item['start_index'] = curr_idx
        
    return list_to_tree(structure)

def reorder_dict(data, key_order):
    return {k: data[k] for k in key_order if k in data}

def format_structure(structure, order=None):
    if isinstance(structure, dict):
        if 'nodes' in structure: structure['nodes'] = format_structure(structure['nodes'], order)
        return reorder_dict(structure, order) if order else structure
    elif isinstance(structure, list):
        return [format_structure(i, order) for i in structure]
    return structure

class ConfigLoader:
    def __init__(self, default_path=None):
        if default_path is None: default_path = Path(__file__).parent / "config.yaml"
        # Create empty config if file doesn't exist
        if not os.path.exists(default_path):
            self._default_dict = {}
        else:
            with open(default_path, "r", encoding="utf-8") as f: self._default_dict = yaml.safe_load(f) or {}
    def load(self, user_opt=None) -> config:
        u_dict = vars(user_opt) if isinstance(user_opt, config) else (user_opt or {})
        return config(**{**self._default_dict, **u_dict})

def convert_physical_index_to_int(data):
    """Safely converts physical_index to integer in a list of dicts"""
    if isinstance(data, list):
        for i in range(len(data)):
            if 'physical_index' in data[i]:
                try:
                    # Handle both strings and ints, and strange formats like "Page 1"
                    val = str(data[i]['physical_index'])
                    m = re.search(r'(\d+)', val)
                    if m: 
                        data[i]['physical_index'] = int(m.group(1))
                    else:
                        # If no digit found, remove it or set to None to filter later
                        data[i]['physical_index'] = None
                except:
                    data[i]['physical_index'] = None
    return data

def get_text_of_pages(pdf_path, start, end, tag=True):
    reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    # Add bounds checking
    total_pages = len(reader.pages)
    start = max(1, start)
    
    # range is exclusive at the end, so min(end, total_pages) might miss the last page if end==total_pages
    # We want inclusive end. range(start-1, end) covers start-1 to end-1.
    # If we want to cover page 'end', we need range up to 'end'.
    
    loop_end = min(end, total_pages)
    
    for i in range(start-1, loop_end):
        t = reader.pages[i].extract_text() or ""
        text += f"<start_index_{i+1}>\n{t}\n<end_index_{i+1}>\n" if tag else t
    return text

# --- Missing Functions Fixed Below ---

def add_preface_if_needed(toc_list):
    if not toc_list:
        return toc_list
        
    first_item = toc_list[0]
    if not isinstance(first_item, dict):
        return toc_list
        
    start_index = first_item.get('start_index') or first_item.get('physical_index')
    
    try:
        if isinstance(start_index, str):
            nums = re.findall(r'\d+', start_index)
            start_val = int(nums[0]) if nums else 1
        else:
            start_val = int(start_index)
    except:
        start_val = 1
        
    if start_val > 1:
        preface_node = {
            'title': 'Preface / Abstract',
            'structure': '0',
            'start_index': 1,
            'end_index': start_val, # Overlap slightly
            'physical_index': 1,
            'level': 1
        }
        toc_list.insert(0, preface_node)
        
    return toc_list

def convert_page_to_int(toc_list):
    if not isinstance(toc_list, list):
        return toc_list
        
    for item in toc_list:
        if isinstance(item, dict) and 'page' in item:
            original_page = str(item['page'])
            nums = re.findall(r'\d+', original_page)
            if nums:
                item['page'] = int(nums[0])
            else:
                item['page'] = 1
        if isinstance(item, dict) and 'nodes' in item:
            convert_page_to_int(item['nodes'])
            
    return toc_list

def add_node_text(structure, page_list):
    """
    Populates the tree nodes with actual text from the PDF page_list
    based on start_index and end_index.
    """
    if isinstance(structure, list):
        for node in structure:
            add_node_text(node, page_list)
            
    elif isinstance(structure, dict):
        # Default to 0 if indices are missing or None
        start = int(structure.get('start_index') or 1)
        end = int(structure.get('end_index') or start)
        
        text_content = ""
        # PDF pages are 1-based, page_list is 0-based
        max_page = len(page_list)
        
        start_idx = max(0, start - 1)
        end_idx = min(max_page, end) # Loop until 'end' (exclusive in slicing if we used slicing)
        
        # Logic: if start=1, end=2. We want page 0 and page 1.
        # range(0, 2) gives 0, 1. Correct.
        
        for i in range(start_idx, end_idx):
            if i < len(page_list):
                # page_list[i] is a tuple (text, length)
                text_content += page_list[i][0] + "\n"
                
        structure['text'] = text_content
        
        # Recursively process children
        if 'nodes' in structure:
            add_node_text(structure['nodes'], page_list)

async def generate_summaries_for_structure(structure, model=None):
    """
    Generates summaries for every node in the tree structure using the LLM.
    """
    # Get all nodes flattened
    nodes = get_nodes(structure)
    tasks = []
    
    # Define the async worker for a single node
    async def summarize_node(node):
        text_content = node.get('text', '')
        if not text_content: 
            node['summary'] = ""
            return

        # Limit text to avoid token overflow, simple prompt
        prompt = f"Summarize the following section text in one concise sentence:\n\n{text_content[:4000]}"
        
        # Call the async API wrapper
        summary = await ChatGPT_API_async(model, prompt)
        node['summary'] = summary.strip()

    # Create tasks for all nodes
    for node in nodes:
        tasks.append(summarize_node(node))
        
    # Run all summary generations in parallel
    if tasks:
        await asyncio.gather(*tasks)
        
    return structure

def clean_page_numbers(data):
    """
    Recursively converts 'page_number' or 'page' values from strings to ints.
    Fixes: TypeError: unsupported operand type(s) for -: 'str' and 'int'
    """
    if isinstance(data, dict):
        for k, v in data.items():
            if k in ['page_number', 'page', 'physical_index', 'start_index', 'end_index']:
                try:
                    # Remove non-digit chars and convert
                    if isinstance(v, str):
                        digits = ''.join(filter(str.isdigit, v))
                        data[k] = int(digits) if digits else 0
                    elif isinstance(v, (float, int)):
                        data[k] = int(v)
                except:
                    pass
        # Recurse
        for k, v in data.items():
            clean_page_numbers(v)
            
    elif isinstance(data, list):
        for item in data:
            clean_page_numbers(item)
    return data

def remove_structure_text(structure):
    """
    Recursively removes the 'text' field from the structure to clean up the output.
    """
    if isinstance(structure, dict):
        structure.pop('text', None)
        if 'nodes' in structure:
            remove_structure_text(structure['nodes'])
    elif isinstance(structure, list):
        for item in structure:
            remove_structure_text(item)