import requests
import logging
import os
import textwrap
from datetime import datetime
import time
import json
import PyPDF2
import copy
import asyncio
import pymupdf
from io import BytesIO
from dotenv import load_dotenv
from transformers import AutoTokenizer
load_dotenv()
import logging
import yaml
from pathlib import Path
from types import SimpleNamespace as config

# Backward compatibility: support CHATGPT_API_KEY as alias for OPENAI_API_KEY
if not os.getenv("OPENAI_API_KEY") and os.getenv("CHATGPT_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("CHATGPT_API_KEY")

def count_tokens(text, model=None):
    """
    计算文本的token数量
    """
    if not text:
        return 0
    # 简单的token计数近似
    return len(text.split())

def calculate_token_size(payload, model_name="Qwen3-8B"):
    """
    计算请求的token大小
    
    参数:
    payload: 请求体
    model_name: 模型名称（可选）
    
    返回:
    int: 输入token数 如果计算成功
    int: 请求token大小（备用方案）如果计算失败
    """
    try:
        # 直接使用备用方案，避免从 Hugging Face 下载模型
        # 只提取消息内容进行简单计数
        messages = payload["messages"]
        text = "".join([msg["content"] for msg in messages])
        # 简单估算 token 数量（英文单词数 + 中文字符数）
        import re
        # 英文单词
        english_words = len(re.findall(r'\b\w+\b', text))
        # 中文字符
        chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
        # 其他字符
        other_chars = len(text) - len(re.findall(r'[\w\u4e00-\u9fa5]', text))
        # 估算 token 数量（英文单词按1个token，中文字符按1个token，其他字符按2个算1个token）
        input_tokens = english_words + chinese_chars + (other_chars // 2)
        return input_tokens
    except Exception as e:
        print(f"计算token时出错: {e}")
        # 备用方案
        request_tokens = count_tokens(json.dumps(payload))
        print(f"请求token大小: {request_tokens}")
        return request_tokens


def llm_completion(model, prompt, chat_history=None, return_finish_reason=False):
    """
    调用LLM模型完成文本生成
    
    参数:
    model: 模型名称
    prompt: 提示词
    chat_history: 聊天历史（可选）
    return_finish_reason: 是否返回完成原因（可选）
    
    返回:
    str: 模型生成的文本
    tuple: 如果return_finish_reason为True，返回(生成文本, 完成原因)
    """
    if model:
        model = model.removeprefix("litellm/")
    max_retries = 10
    messages = list(chat_history) + [{"role": "user", "content": prompt}] if chat_history else [{"role": "user", "content": prompt}]
    for i in range(max_retries):
        try:
            # 记录开始时间
            start_time = time.time()
            
            url = "http://10.1.141.33:8001/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer 7257b5ab-1cd1-424e-902d-79d08bc11442"
            }
            payload = {
                "model": "Qwen3-8B",
                "messages": messages,
                "chat_template_kwargs": {
                    "enable_thinking": False
                },
                "temperature": 0
            }
            print(f"\n===== 发送API请求 =====")
            print(f"URL: {url}")
            print(f"头部: {headers}")
            try:
                print(f"请求体: {json.dumps(payload, indent=2, ensure_ascii=False)}")
            except UnicodeEncodeError:
                print(f"请求体: {json.dumps(payload, indent=2, ensure_ascii=True)}")
            # 计算并打印请求token大小
            input_tokens = calculate_token_size(payload)
            if not isinstance(input_tokens, Exception):
                print(f"实际输入 token 数: {input_tokens}")
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            print(f"响应状态码: {response.status_code}")
            print(f"响应头: {dict(response.headers)}")
            print(f"响应内容长度: {len(response.text)}")
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            print(f"\n===== 模型回答 =====")
            # 限制输出长度为1000字
            if len(content) > 1000:
                print(content[:1000] + "...")
            else:
                print(content)
            
            # 计算并打印总调用时间
            end_time = time.time()
            total_time = end_time - start_time
            print(f"\n===== 调用时间 =====")
            print(f"调用时间: {total_time:.2f} 秒")
            
            if return_finish_reason:
                finish_reason = "max_output_reached" if data["choices"][0]["finish_reason"] == "length" else "finished"
                return content, finish_reason
            return content
        except requests.exceptions.RequestException as e:
            print(f"\n请求失败: {e}")
            import traceback
            traceback.print_exc()
            if i < max_retries - 1:
                time.sleep(1)
            else:
                logging.error('达到最大重试次数: ' + prompt)
                if return_finish_reason:
                    return "", "error"
                return ""
        except json.JSONDecodeError as e:
            print(f"\n解析响应失败: {e}")
            print("原始响应内容:")
            print(response.text[:500] + "..." if len(response.text) > 500 else response.text)
            import traceback
            traceback.print_exc()
            if i < max_retries - 1:
                time.sleep(1)
            else:
                logging.error('达到最大重试次数: ' + prompt)
                if return_finish_reason:
                    return "", "error"
                return ""
        except KeyError as e:
            print(f"\n响应格式错误: {e}")
            print("原始响应内容:")
            print(response.text)
            import traceback
            traceback.print_exc()
            if i < max_retries - 1:
                time.sleep(1)
            else:
                logging.error('达到最大重试次数: ' + prompt)
                if return_finish_reason:
                    return "", "error"
                return ""
        except Exception as e:
            print(f"\n其他错误: {e}")
            import traceback
            traceback.print_exc()
            if i < max_retries - 1:
                time.sleep(1)
            else:
                logging.error('达到最大重试次数: ' + prompt)
                if return_finish_reason:
                    return "", "error"
                return ""



async def llm_acompletion(model, prompt):
    """
    异步调用LLM模型完成文本生成
    
    参数:
    model: 模型名称
    prompt: 提示词
    
    返回:
    str: 模型生成的文本
    """
    if model:
        model = model.removeprefix("litellm/")
    max_retries = 10
    messages = [{"role": "user", "content": prompt}]
    for i in range(max_retries):
        try:
            # 记录开始时间
            start_time = time.time()
            
            url = "http://10.1.141.33:8001/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer 7257b5ab-1cd1-424e-902d-79d08bc11442"
            }
            payload = {
                "model": "Qwen3-8B",
                "messages": messages,
                "chat_template_kwargs": {
                    "enable_thinking": False
                },
                "temperature": 0
            }
            print(f"\n===== 发送API请求 =====")
            print(f"URL: {url}")
            print(f"头部: {headers}")
            try:
                print(f"请求体: {json.dumps(payload, indent=2, ensure_ascii=False)}")
            except UnicodeEncodeError:
                print(f"请求体: {json.dumps(payload, indent=2, ensure_ascii=True)}")

            # 计算并打印请求token大小
            input_tokens = calculate_token_size(payload)
            if not isinstance(input_tokens, Exception):
                print(f"实际输入 token 数: {input_tokens}")
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            print(f"响应状态码: {response.status_code}")
            print(f"响应头: {dict(response.headers)}")
            print(f"响应内容长度: {len(response.text)}")
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            print(f"\n===== 模型回答 =====")
            # 限制输出长度为1000字
            if len(content) > 1000:
                print(content[:1000] + "...")
            else:
                print(content)
            
            # 计算并打印总调用时间
            end_time = time.time()
            total_time = end_time - start_time
            print(f"\n===== 调用时间 =====")
            print(f"调用时间: {total_time:.2f} 秒")
            
            return content
        except requests.exceptions.RequestException as e:
            print(f"\n请求失败: {e}")
            import traceback
            traceback.print_exc()
            if i < max_retries - 1:
                await asyncio.sleep(1)
            else:
                logging.error('达到最大重试次数: ' + prompt)
                return ""
        except json.JSONDecodeError as e:
            print(f"\n解析响应失败: {e}")
            print("原始响应内容:")
            print(response.text[:500] + "..." if len(response.text) > 500 else response.text)
            import traceback
            traceback.print_exc()
            if i < max_retries - 1:
                await asyncio.sleep(1)
            else:
                logging.error('达到最大重试次数: ' + prompt)
                return ""
        except KeyError as e:
            print(f"\n响应格式错误: {e}")
            print("原始响应内容:")
            print(response.text)
            import traceback
            traceback.print_exc()
            if i < max_retries - 1:
                await asyncio.sleep(1)
            else:
                logging.error('达到最大重试次数: ' + prompt)
                return ""
        except Exception as e:
            print(f"\n其他错误: {e}")
            import traceback
            traceback.print_exc()
            if i < max_retries - 1:
                await asyncio.sleep(1)
            else:
                logging.error('达到最大重试次数: ' + prompt)
                return ""
            
def get_json_content(response):
    """
    从响应中提取JSON内容
    
    参数:
    response: 包含JSON的响应文本
    
    返回:
    str: 提取出的JSON内容
    """
    start_idx = response.find("```json")
    if start_idx != -1:
        start_idx += 7
        response = response[start_idx:]
        
    end_idx = response.rfind("```")
    if end_idx != -1:
        response = response[:end_idx]
    
    json_content = response.strip()
    return json_content
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
        # 直接使用简单估算方法，避免使用 tiktoken 库
        # 简单估算 token 数量（英文单词数 + 中文字符数）
        import re
        # 英文单词
        english_words = len(re.findall(r'\b\w+\b', text))
        # 中文字符
        chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
        # 其他字符
        other_chars = len(text) - len(re.findall(r'[\w\u4e00-\u9fa5]', text))
        # 估算 token 数量（英文单词按1个token，中文字符按1个token，其他字符按2个算1个token）
        return english_words + chinese_chars + (other_chars // 2)
    except:
        #  fallback: 最简单的估算
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
         

def extract_json(content):
    """
    从内容中提取并解析JSON
    
    参数:
    content: 包含JSON的内容
    
    返回:
    dict: 解析后的JSON对象
    """
    try:
        # 首先，尝试提取包含在 ```json 和 ``` 之间的JSON
        start_idx = content.find("```json")
        if start_idx != -1:
            start_idx += 7  # 调整索引，从分隔符后开始
            end_idx = content.rfind("```")
            json_content = content[start_idx:end_idx].strip()
        else:
            # 如果没有分隔符，假设整个内容都是JSON
            json_content = content.strip()

        # 清理可能导致解析错误的常见问题
        json_content = json_content.replace('None', 'null')  # 将Python的None替换为JSON的null
        json_content = json_content.replace('\n', ' ').replace('\r', ' ')  # 移除换行符
        json_content = ' '.join(json_content.split())  # 标准化空白字符

        # 尝试解析并返回JSON对象
        return json.loads(json_content)
    except json.JSONDecodeError as e:
        logging.error(f"提取JSON失败: {e}")
        # 如果初始解析失败，尝试进一步清理内容
        try:
            # 移除括号/大括号前的尾随逗号
            json_content = json_content.replace(',]', ']').replace(',}', '}')
            return json.loads(json_content)
        except:
            logging.error("即使清理后也无法解析JSON")
            return {}
    except Exception as e:
        logging.error(f"提取JSON时发生意外错误: {e}")
        return {}

def write_node_id(data, node_id=0):
    """
    为数据结构中的每个节点添加node_id
    
    参数:
    data: 要添加node_id的数据结构
    node_id: 起始node_id（可选）
    
    返回:
    int: 下一个可用的node_id
    """
    if isinstance(data, dict):
        data['node_id'] = str(node_id).zfill(4)
        node_id += 1
        for key in list(data.keys()):
            if 'nodes' in key:
                node_id = write_node_id(data[key], node_id)
    elif isinstance(data, list):
        for index in range(len(data)):
            node_id = write_node_id(data[index], node_id)
    return node_id

def get_nodes(structure):
    """
    从结构中提取所有节点
    
    参数:
    structure: 包含节点的数据结构
    
    返回:
    list: 所有节点的列表
    """
    if isinstance(structure, dict):
        structure_node = copy.deepcopy(structure)
        structure_node.pop('nodes', None)
        nodes = [structure_node]
        for key in list(structure.keys()):
            if 'nodes' in key:
                nodes.extend(get_nodes(structure[key]))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(get_nodes(item))
        return nodes
    
def structure_to_list(structure):
    """
    将结构转换为节点列表
    
    参数:
    structure: 要转换的数据结构
    
    返回:
    list: 节点列表
    """
    if isinstance(structure, dict):
        nodes = []
        nodes.append(structure)
        if 'nodes' in structure:
            nodes.extend(structure_to_list(structure['nodes']))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(structure_to_list(item))
        return nodes

    
def get_leaf_nodes(structure):
    """
    从结构中提取所有叶节点
    
    参数:
    structure: 包含节点的数据结构
    
    返回:
    list: 所有叶节点的列表
    """
    if isinstance(structure, dict):
        if not structure['nodes']:
            structure_node = copy.deepcopy(structure)
            structure_node.pop('nodes', None)
            return [structure_node]
        else:
            leaf_nodes = []
            for key in list(structure.keys()):
                if 'nodes' in key:
                    leaf_nodes.extend(get_leaf_nodes(structure[key]))
            return leaf_nodes
    elif isinstance(structure, list):
        leaf_nodes = []
        for item in structure:
            leaf_nodes.extend(get_leaf_nodes(item))
        return leaf_nodes

def is_leaf_node(data, node_id):
    """
    检查指定node_id的节点是否为叶节点
    
    参数:
    data: 包含节点的数据结构
    node_id: 要检查的节点ID
    
    返回:
    bool: 如果是叶节点返回True，否则返回False
    """
    # 辅助函数，根据node_id查找节点
    def find_node(data, node_id):
        if isinstance(data, dict):
            if data.get('node_id') == node_id:
                return data
            for key in data.keys():
                if 'nodes' in key:
                    result = find_node(data[key], node_id)
                    if result:
                        return result
        elif isinstance(data, list):
            for item in data:
                result = find_node(item, node_id)
                if result:
                    return result
        return None

    # 查找具有给定node_id的节点
    node = find_node(data, node_id)

    # 检查节点是否为叶节点
    if node and not node.get('nodes'):
        return True
    return False

def get_last_node(structure):
    """
    获取结构中的最后一个节点
    
    参数:
    structure: 包含节点的结构
    
    返回:
    object: 最后一个节点
    """
    return structure[-1]


def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本
    
    参数:
    pdf_path: PDF文件路径
    
    返回:
    str: 提取的文本
    """
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    # 返回文本而不是列表
    text=""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text+=page.extract_text()
    return text

def get_pdf_title(pdf_path):
    """
    获取PDF文件的标题
    
    参数:
    pdf_path: PDF文件路径
    
    返回:
    str: PDF文件的标题
    """
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    meta = pdf_reader.metadata
    title = meta.title if meta and meta.title else 'Untitled'
    return title

def get_text_of_pages(pdf_path, start_page, end_page, tag=True):
    """
    获取PDF文件中指定页面范围的文本
    
    参数:
    pdf_path: PDF文件路径
    start_page: 开始页面（1-based）
    end_page: 结束页面（1-based）
    tag: 是否添加页面标签（可选）
    
    返回:
    str: 提取的文本
    """
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page_num in range(start_page-1, end_page):
        page = pdf_reader.pages[page_num]
        page_text = page.extract_text()
        if tag:
            text += f"<start_index_{page_num+1}>\n{page_text}\n<end_index_{page_num+1}>\n"
        else:
            text += page_text
    return text

def get_first_start_page_from_text(text):
    """
    从文本中获取第一个开始页面的索引
    
    参数:
    text: 包含页面标签的文本
    
    返回:
    int: 第一个开始页面的索引，未找到返回-1
    """
    start_page = -1
    start_page_match = re.search(r'<start_index_(\d+)>', text)
    if start_page_match:
        start_page = int(start_page_match.group(1))
    return start_page

def get_last_start_page_from_text(text):
    """
    从文本中获取最后一个开始页面的索引
    
    参数:
    text: 包含页面标签的文本
    
    返回:
    int: 最后一个开始页面的索引，未找到返回-1
    """
    start_page = -1
    # 查找所有start_index标签的匹配
    start_page_matches = re.finditer(r'<start_index_(\d+)>', text)
    # 将迭代器转换为列表并获取最后一个匹配（如果存在）
    matches_list = list(start_page_matches)
    if matches_list:
        start_page = int(matches_list[-1].group(1))
    return start_page


def sanitize_filename(filename, replacement='-'):
    """
    清理文件名中的无效字符
    
    参数:
    filename: 原始文件名
    replacement: 替换字符（可选）
    
    返回:
    str: 清理后的文件名
    """
    # 在Linux中，只有 '/' 和 '\0'（空字符）在文件名中是无效的。
    # 空字符不能在字符串中表示，所以我们只处理 '/'。
    return filename.replace('/', replacement)

def get_pdf_name(pdf_path):
    """
    获取PDF文件的名称
    
    参数:
    pdf_path: PDF文件路径或BytesIO对象
    
    返回:
    str: PDF文件的名称
    """
    # 提取PDF名称
    if isinstance(pdf_path, str):
        pdf_name = os.path.basename(pdf_path)
    elif isinstance(pdf_path, BytesIO):
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        meta = pdf_reader.metadata
        pdf_name = meta.title if meta and meta.title else 'Untitled'
        pdf_name = sanitize_filename(pdf_name)
    return pdf_name


class JsonLogger:
    """
    JSON日志记录器
    """
    def __init__(self, file_path):
        # 提取PDF名称作为日志名称
        pdf_name = get_pdf_name(file_path)
            
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{pdf_name}_{current_time}.json"
        os.makedirs("./logs", exist_ok=True)
        # 初始化空列表存储所有消息
        self.log_data = []

    def log(self, level, message, **kwargs):
        """
        记录日志
        
        参数:
        level: 日志级别
        message: 日志消息
        **kwargs: 额外参数
        """
        if isinstance(message, dict):
            self.log_data.append(message)
        else:
            self.log_data.append({'message': message})
        # 添加新消息到日志数据
        
        # 将整个日志数据写入文件
        with open(self._filepath(), "w") as f:
            json.dump(self.log_data, f, indent=2)

    def info(self, message, **kwargs):
        """
        记录信息级别的日志
        
        参数:
        message: 日志消息
        **kwargs: 额外参数
        """
        self.log("INFO", message, **kwargs)

    def error(self, message, **kwargs):
        """
        记录错误级别的日志
        
        参数:
        message: 日志消息
        **kwargs: 额外参数
        """
        self.log("ERROR", message, **kwargs)

    def debug(self, message, **kwargs):
        """
        记录调试级别的日志
        
        参数:
        message: 日志消息
        **kwargs: 额外参数
        """
        self.log("DEBUG", message, **kwargs)

    def exception(self, message, **kwargs):
        """
        记录异常日志
        
        参数:
        message: 日志消息
        **kwargs: 额外参数
        """
        kwargs["exception"] = True
        self.log("ERROR", message, **kwargs)

    def _filepath(self):
        """
        获取日志文件路径
        
        返回:
        str: 日志文件路径
        """
        return os.path.join("logs", self.filename)
    



def list_to_tree(data):
    """
    将节点列表转换为树形结构
    
    参数:
    data: 节点列表
    
    返回:
    list: 树形结构
    """
    def get_parent_structure(structure):
        """辅助函数，获取父结构代码"""
        if not structure:
            return None
        parts = str(structure).split('.')
        return '.'.join(parts[:-1]) if len(parts) > 1 else None
    
    # 第一次遍历：创建节点并跟踪父子关系
    nodes = {}
    root_nodes = []
    
    for item in data:
        structure = item.get('structure')
        node = {
            'title': item.get('title'),
            'start_index': item.get('start_index'),
            'end_index': item.get('end_index'),
            'nodes': []
        }
        
        nodes[structure] = node
        
        # 查找父节点
        parent_structure = get_parent_structure(structure)
        
        if parent_structure:
            # 如果父节点存在，添加为子节点
            if parent_structure in nodes:
                nodes[parent_structure]['nodes'].append(node)
            else:
                root_nodes.append(node)
        else:
            # 没有父节点，这是根节点
            root_nodes.append(node)
    
    # 辅助函数，清理空的子节点数组
    def clean_node(node):
        if not node['nodes']:
            del node['nodes']
        else:
            for child in node['nodes']:
                clean_node(child)
        return node
    
    # 清理并返回树
    return [clean_node(node) for node in root_nodes]

def add_preface_if_needed(data):
    """
    如果需要，添加前言节点
    
    参数:
    data: 节点列表
    
    返回:
    list: 添加前言后的节点列表
    """
    if not isinstance(data, list) or not data:
        return data

    if data[0]['physical_index'] is not None and data[0]['physical_index'] > 1:
        preface_node = {
            "structure": "0",
            "title": "前言",
            "physical_index": 1,
        }
        data.insert(0, preface_node)
    return data



def get_page_tokens(pdf_path, model=None, pdf_parser="PyPDF2"):
    """
    获取PDF文件每页的文本和token数量
    
    参数:
    pdf_path: PDF文件路径或BytesIO对象
    model: 模型名称（可选）
    pdf_parser: PDF解析器（可选）
    
    返回:
    list: 每页的文本和token数量的列表
    """
    if pdf_parser == "PyPDF2":
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        page_list = []
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            token_length = len(page_text.split())
            page_list.append((page_text, token_length))
        return page_list
    elif pdf_parser == "PyMuPDF":
        if isinstance(pdf_path, BytesIO):
            pdf_stream = pdf_path
            doc = pymupdf.open(stream=pdf_stream, filetype="pdf")
        elif isinstance(pdf_path, str) and os.path.isfile(pdf_path) and pdf_path.lower().endswith(".pdf"):
            doc = pymupdf.open(pdf_path)
        page_list = []
        for page in doc:
            page_text = page.get_text()
            token_length = len(page_text.split())
            page_list.append((page_text, token_length))
        return page_list
    else:
        raise ValueError(f"不支持的PDF解析器: {pdf_parser}")

        

def get_text_of_pdf_pages(pdf_pages, start_page, end_page):
    """
    获取PDF页面列表中指定页面范围的文本
    
    参数:
    pdf_pages: PDF页面列表
    start_page: 开始页面（1-based）
    end_page: 结束页面（1-based）
    
    返回:
    str: 提取的文本
    """
    text = ""
    for page_num in range(start_page-1, end_page):
        text += pdf_pages[page_num][0]
    return text

def get_text_of_pdf_pages_with_labels(pdf_pages, start_page, end_page):
    """
    获取PDF页面列表中指定页面范围的文本，并添加页面标签
    
    参数:
    pdf_pages: PDF页面列表
    start_page: 开始页面（1-based）
    end_page: 结束页面（1-based）
    
    返回:
    str: 带有页面标签的文本
    """
    text = ""
    for page_num in range(start_page-1, end_page):
        text += f"<physical_index_{page_num+1}>\n{pdf_pages[page_num][0]}\n<physical_index_{page_num+1}>\n"
    return text

def get_number_of_pages(pdf_path):
    """
    获取PDF文件的页数
    
    参数:
    pdf_path: PDF文件路径
    
    返回:
    int: PDF文件的页数
    """
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    num = len(pdf_reader.pages)
    return num



def post_processing(structure, end_physical_index):
    """
    对结构进行后处理
    
    参数:
    structure: 节点列表
    end_physical_index: 结束物理索引
    
    返回:
    list: 处理后的树形结构或节点列表
    """
    # 首先在扁平列表中将page_number转换为start_index
    for i, item in enumerate(structure):
        item['start_index'] = item.get('physical_index')
        if i < len(structure) - 1:
            if structure[i + 1].get('appear_start') == 'yes':
                item['end_index'] = structure[i + 1]['physical_index']-1
            else:
                item['end_index'] = structure[i + 1]['physical_index']
        else:
            item['end_index'] = end_physical_index
    tree = list_to_tree(structure)
    if len(tree)!=0:
        return tree
    else:
        # 移除appear_start
        for node in structure:
            node.pop('appear_start', None)
            node.pop('physical_index', None)
        return structure

def clean_structure_post(data):
    """
    清理结构中的后处理字段
    
    参数:
    data: 要清理的数据结构
    
    返回:
    object: 清理后的数据结构
    """
    if isinstance(data, dict):
        data.pop('page_number', None)
        data.pop('start_index', None)
        data.pop('end_index', None)
        if 'nodes' in data:
            clean_structure_post(data['nodes'])
    elif isinstance(data, list):
        for section in data:
            clean_structure_post(section)
    return data

def remove_fields(data, fields=['text']):
    """
    从数据结构中移除指定字段
    
    参数:
    data: 要处理的数据结构
    fields: 要移除的字段列表（可选）
    
    返回:
    object: 移除指定字段后的数据结构
    """
    if isinstance(data, dict):
        return {k: remove_fields(v, fields)
            for k, v in data.items() if k not in fields}
    elif isinstance(data, list):
        return [remove_fields(item, fields) for item in data]
    return data

def print_toc(tree, indent=0):
    """
    打印目录结构
    
    参数:
    tree: 目录树结构
    indent: 缩进级别（可选）
    """
    for node in tree:
        print('  ' * indent + node['title'])
        if node.get('nodes'):
            print_toc(node['nodes'], indent + 1)

def print_json(data, max_len=40, indent=2):
    """
    打印JSON数据，简化长字符串
    
    参数:
    data: 要打印的JSON数据
    max_len: 字符串最大长度（可选）
    indent: 缩进级别（可选）
    """
    def simplify_data(obj):
        if isinstance(obj, dict):
            return {k: simplify_data(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [simplify_data(item) for item in obj]
        elif isinstance(obj, str) and len(obj) > max_len:
            return obj[:max_len] + '...'
        else:
            return obj
    
    simplified = simplify_data(data)
    print(json.dumps(simplified, indent=indent, ensure_ascii=False))


def remove_structure_text(data):
    """
    移除结构中的文本字段
    
    参数:
    data: 要处理的数据结构
    
    返回:
    object: 移除文本字段后的数据结构
    """
    if isinstance(data, dict):
        data.pop('text', None)
        if 'nodes' in data:
            remove_structure_text(data['nodes'])
    elif isinstance(data, list):
        for item in data:
            remove_structure_text(item)
    return data


def check_token_limit(structure, limit=110000):
    """
    检查结构中节点的token数量是否超过限制
    
    参数:
    structure: 要检查的数据结构
    limit: token数量限制（可选）
    """
    list = structure_to_list(structure)
    for node in list:
        num_tokens = count_tokens(node['text'], model=None)
        if num_tokens > limit:
            print(f"节点ID: {node['node_id']} 有 {num_tokens} 个token")
            print("开始索引:", node['start_index'])
            print("结束索引:", node['end_index'])
            print("标题:", node['title'])
            print("\n")


def convert_physical_index_to_int(data):
    """
    将物理索引转换为整数
    
    参数:
    data: 包含物理索引的数据结构或字符串
    
    返回:
    object: 转换后的结果
    """
    if isinstance(data, list):
        for i in range(len(data)):
            # 检查项目是否为字典且有'physical_index'键
            if isinstance(data[i], dict) and 'physical_index' in data[i]:
                if isinstance(data[i]['physical_index'], str):
                    if data[i]['physical_index'].startswith('<physical_index_'):
                        data[i]['physical_index'] = int(data[i]['physical_index'].split('_')[-1].rstrip('>').strip())
                    elif data[i]['physical_index'].startswith('physical_index_'):
                        data[i]['physical_index'] = int(data[i]['physical_index'].split('_')[-1].strip())
    elif isinstance(data, str):
        if data.startswith('<physical_index_'):
            data = int(data.split('_')[-1].rstrip('>').strip())
        elif data.startswith('physical_index_'):
            data = int(data.split('_')[-1].strip())
        # 检查数据是否为整数
        if isinstance(data, int):
            return data
        else:
            return None
    return data


def convert_page_to_int(data):
    """
    将页面号转换为整数
    
    参数:
    data: 包含页面号的节点列表
    
    返回:
    list: 转换后的节点列表
    """
    for item in data:
        if 'page' in item and isinstance(item['page'], str):
            try:
                item['page'] = int(item['page'])
            except ValueError:
                # 转换失败时保持原始值
                pass
    return data


def add_node_text(node, pdf_pages):
    """
    为节点添加文本内容
    
    参数:
    node: 节点或节点列表
    pdf_pages: PDF页面列表
    """
    if isinstance(node, dict):
        start_page = node.get('start_index')
        end_page = node.get('end_index')
        node['text'] = get_text_of_pdf_pages(pdf_pages, start_page, end_page)
        if 'nodes' in node:
            add_node_text(node['nodes'], pdf_pages)
    elif isinstance(node, list):
        for index in range(len(node)):
            add_node_text(node[index], pdf_pages)
    return


def add_node_text_with_labels(node, pdf_pages):
    """
    为节点添加带标签的文本内容
    
    参数:
    node: 节点或节点列表
    pdf_pages: PDF页面列表
    """
    if isinstance(node, dict):
        start_page = node.get('start_index')
        end_page = node.get('end_index')
        node['text'] = get_text_of_pdf_pages_with_labels(pdf_pages, start_page, end_page)
        if 'nodes' in node:
            add_node_text_with_labels(node['nodes'], pdf_pages)
    elif isinstance(node, list):
        for index in range(len(node)):
            add_node_text_with_labels(node[index], pdf_pages)
    return


async def generate_node_summary(node, model=None):
    """
    生成节点摘要
    
    参数:
    node: 要生成摘要的节点
    model: 模型名称（可选）
    
    返回:
    str: 节点的摘要
    """
    prompt = f"""你获得了文档的一部分，你的任务是生成该部分文档的描述，说明该部分文档涵盖的主要内容。

    部分文档文本: {node['text']}
    
    直接返回描述，不要包含任何其他文本。
    """
    response = await llm_acompletion(model, prompt)
    return response


async def generate_summaries_for_structure(structure, model=None):
    """
    为结构中的所有节点生成摘要
    
    参数:
    structure: 要生成摘要的数据结构
    model: 模型名称（可选）
    
    返回:
    object: 包含摘要的结构
    """
    nodes = structure_to_list(structure)
    tasks = [generate_node_summary(node, model=model) for node in nodes]
    summaries = await asyncio.gather(*tasks)
    
    for node, summary in zip(nodes, summaries):
        node['summary'] = summary
    return structure


def create_clean_structure_for_description(structure):
    """
    创建用于文档描述生成的干净结构，
    排除不必要的字段，如 'text'。
    """
    if isinstance(structure, dict):
        clean_node = {}
        # 只包含描述所需的必要字段
        for key in ['title', 'node_id', 'summary', 'prefix_summary']:
            if key in structure:
                clean_node[key] = structure[key]
        
        # 递归处理子节点
        if 'nodes' in structure and structure['nodes']:
            clean_node['nodes'] = create_clean_structure_for_description(structure['nodes'])
        
        return clean_node
    elif isinstance(structure, list):
        return [create_clean_structure_for_description(item) for item in structure]
    else:
        return structure


def generate_doc_description(structure, model=None):
    """
    生成文档描述
    
    参数:
    structure: 文档结构
    model: 模型名称（可选）
    
    返回:
    str: 文档的描述
    """
    prompt = f"""你是生成文档描述的专家。
    你获得了一份文档的结构。你的任务是为该文档生成一个一句话的描述，使其易于与其他文档区分。
        
    文档结构: {structure}
    
    直接返回描述，不要包含任何其他文本。
    """
    response = llm_completion(model, prompt)
    return response


def reorder_dict(data, key_order):
    """
    重新排序字典中的键
    
    参数:
    data: 要排序的字典
    key_order: 键的顺序列表
    
    返回:
    dict: 重新排序后的字典
    """
    if not key_order:
        return data
    return {key: data[key] for key in key_order if key in data}


def format_structure(structure, order=None):
    """
    格式化结构，重新排序键并清理空节点
    
    参数:
    structure: 要格式化的数据结构
    order: 键的顺序列表（可选）
    
    返回:
    object: 格式化后的结构
    """
    if not order:
        return structure
    if isinstance(structure, dict):
        if 'nodes' in structure:
            structure['nodes'] = format_structure(structure['nodes'], order)
        if not structure.get('nodes'):
            structure.pop('nodes', None)
        structure = reorder_dict(structure, order)
    elif isinstance(structure, list):
        structure = [format_structure(item, order) for item in structure]
    return structure


class ConfigLoader:
    """
    配置加载器
    """
    def __init__(self, default_path: str = None):
        """
        初始化配置加载器
        
        参数:
        default_path: 默认配置文件路径（可选）
        """
        if default_path is None:
            default_path = Path(__file__).parent / "config.yaml"
        self._default_dict = self._load_yaml(default_path)

    @staticmethod
    def _load_yaml(path):
        """
        加载YAML配置文件
        
        参数:
        path: 配置文件路径
        
        返回:
        dict: 配置字典
        """
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _validate_keys(self, user_dict):
        """
        验证配置键
        
        参数:
        user_dict: 用户配置字典
        """
        unknown_keys = set(user_dict) - set(self._default_dict)
        if unknown_keys:
            raise ValueError(f"未知的配置键: {unknown_keys}")

    def load(self, user_opt=None) -> config:
        """
        加载配置，将用户选项与默认值合并。
        
        参数:
        user_opt: 用户配置（可选）
        
        返回:
        config: 配置对象
        """
        if user_opt is None:
            user_dict = {}
        elif isinstance(user_opt, config):
            user_dict = vars(user_opt)
        elif isinstance(user_opt, dict):
            user_dict = user_opt
        else:
            raise TypeError("user_opt 必须是 dict, config(SimpleNamespace) 或 None")

        self._validate_keys(user_dict)
        merged = {**self._default_dict, **user_dict}
        return config(**merged)

def create_node_mapping(tree):
    """
    创建节点ID到节点的映射，用于快速查找。
    
    参数:
    tree: 节点树结构
    
    返回:
    dict: 节点ID到节点的映射
    """
    mapping = {}
    def _traverse(nodes):
        for node in nodes:
            if node.get('node_id'):
                mapping[node['node_id']] = node
            if node.get('nodes'):
                _traverse(node['nodes'])
    _traverse(tree)
    return mapping

def print_tree(tree, indent=0):
    """
    打印树结构
    
    参数:
    tree: 树结构
    indent: 缩进级别（可选）
    """
    for node in tree:
        summary = node.get('summary') or node.get('prefix_summary', '')
        summary_str = f"  —  {summary[:60]}..." if summary else ""
        print('  ' * indent + f"[{node.get('node_id', '?')}] {node.get('title', '')}{summary_str}")
        if node.get('nodes'):
            print_tree(node['nodes'], indent + 1)

def print_wrapped(text, width=100):
    """
    打印换行文本
    
    参数:
    text: 要打印的文本
    width: 每行宽度（可选）
    """
    for line in text.splitlines():
        print(textwrap.fill(line, width=width))

