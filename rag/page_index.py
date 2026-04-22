import os
import json
import copy
import math
import random
import re
import logging
from datetime import datetime
from rag.utils import *
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置结构化日志
class StructuredLogger:
    def __init__(self, doc):
        # Extract PDF name for logger name
        pdf_name = get_pdf_name(doc)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{pdf_name}_{current_time}.log"
        self.log_dir = "./logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
    def log(self, step, message):
        log_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 【{step}】{message}"
        print(log_message)
        with open(os.path.join(self.log_dir, self.filename), "a", encoding="utf-8") as f:
            f.write(log_message + "\n")

# 全局日志实例
logger = None

def get_logger():
    global logger
    return logger

def set_logger(new_logger):
    global logger
    logger = new_logger


################### 规则判断函数（策略一：用代码替代简单判断类LLM调用） #######################
def _rule_check_title_in_page(title, page_text):
    """规则检测：标题是否出现在页面文本中（利用PDF内嵌文本的干净特性）"""
    normalized_title = re.sub(r'\s+', '', title.strip().lower())
    normalized_page = re.sub(r'\s+', '', page_text.strip().lower())
    if not normalized_title:
        return None
    if normalized_title in normalized_page:
        return 'yes'
    title_len = len(normalized_title)
    if title_len < 3:
        return None
    best_ratio = 0
    for i in range(len(normalized_page) - title_len + 1):
        window = normalized_page[i:i + title_len]
        common = sum(1 for a, b in zip(normalized_title, window) if a == b)
        ratio = common / title_len
        if ratio > best_ratio:
            best_ratio = ratio
        if ratio >= 0.85:
            return 'yes'
    if best_ratio < 0.5:
        return 'no'
    return None


def _rule_detect_toc_page(page_text):
    """规则检测：该页是否为目录页"""
    text = page_text.strip()
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return None
    has_keyword = bool(re.search(r'table\s+of\s+contents|^contents$|目\s*录|目\s*次', text, re.MULTILINE | re.IGNORECASE))
    dot_leader_count = len(re.findall(r'\.{3,}|…{2,}|(?:\. ){3,}', text))
    lines_ending_with_number = sum(1 for l in lines if re.search(r'\d{1,4}\s*$', l))
    if has_keyword and (dot_leader_count >= 2 or lines_ending_with_number >= 3):
        return 'yes'
    if dot_leader_count >= 5:
        return 'yes'
    if len(lines) > 5 and lines_ending_with_number / len(lines) > 0.5:
        return 'yes'
    if not has_keyword and dot_leader_count == 0 and lines_ending_with_number < 2:
        return 'no'
    return None


def _rule_detect_page_index(toc_content):
    """规则检测：目录中有没有页码"""
    page_number_patterns = len(re.findall(r'[\.…:：]{2,}\s*\d+', toc_content))
    line_ending_numbers = len(re.findall(r'\S+\s+\d{1,4}\s*$', toc_content, re.MULTILINE))
    if page_number_patterns >= 3 or line_ending_numbers >= 3:
        return 'yes'
    if page_number_patterns == 0 and line_ending_numbers <= 1:
        return 'no'
    return None


def _is_likely_heading(title):
    """判断标题是否像真正的章节标题（而非正文片段）"""
    t = title.strip()
    if not t:
        return False
    # physical_index 标签误入
    if '<physical_index' in t:
        return False
    # 过长的标题
    if len(t) > 50:
        return False
    # 以标点/虚词结尾 → 正文片段
    if re.search(r'[，。、；：！？的了着过呢吗得]$', t):
        return False
    # 全是标点或数字
    if re.match(r'^[\d\s.、，。；：]+$', t):
        return False
    # 正向匹配：像真正的标题
    # 1) 编号开头: "1.1 xxx", "第一章 xxx", "(一) xxx", "一、xxx"
    if re.match(r'^[\d.]+\s', t):
        return True
    if re.match(r'^第[一二三四五六七八九十百千\d]+[章节部分篇]', t):
        return True
    if re.match(r'^[（(][一二三四五六七八九十\d]+[）)]', t):
        return True
    if re.match(r'^[一二三四五六七八九十]+[、.]', t):
        return True
    # 2) 短标题（≤12字且无明显正文特征）
    if len(t) <= 12:
        return True
    # 3) 中长标题但包含关键词
    if re.search(r'概述|总结|结论|引言|绪论|前言|摘要|附录|参考文献|发展|展望|分析|研究|方法|背景|目标|策略|规划|建设|驱动|价值|核心', t):
        return True
    return False


def _filter_invalid_toc_entries(toc_items):
    """后处理：过滤掉小模型生成的无效目录条目"""
    # 先统计每页的条目数，如果某页条目过多，说明模型在逐行提取
    from collections import Counter
    page_counts = Counter()
    for item in toc_items:
        pi = item.get('physical_index')
        if pi is not None:
            page_counts[pi] += 1

    filtered = []
    for item in toc_items:
        title = item.get('title', '')
        pi = item.get('physical_index')
        if not _is_likely_heading(title):
            continue
        # 如果某页条目 > 10，只保留有编号的标题
        if pi is not None and page_counts.get(pi, 0) > 10:
            t = title.strip()
            is_numbered = bool(re.match(r'^[\d.]+\s', t)) or \
                          bool(re.match(r'^第[一二三四五六七八九十百千\d]+', t)) or \
                          bool(re.match(r'^[（(][一二三四五六七八九十\d]+[）)]', t)) or \
                          bool(re.match(r'^[一二三四五六七八九十]+[、.]', t))
            if not is_numbered:
                continue
        filtered.append(item)
    return filtered


################### check title in page #########################################################
async def check_title_appearance(item, page_list, start_index=1, model=None):    
    title=item['title']
    if 'physical_index' not in item or item['physical_index'] is None:
        return {'list_index': item.get('list_index'), 'answer': 'no', 'title':title, 'page_number': None}
    
    
    page_number = item['physical_index']
    page_text = page_list[page_number-start_index][0]

    # 策略一：先用规则判断
    rule_result = _rule_check_title_in_page(title, page_text)
    if rule_result is not None:
        return {'list_index': item['list_index'], 'answer': rule_result, 'title': title, 'page_number': page_number}
    
    prompt = f"""
    你的任务是检查给定的章节是否在给定的页面文本中出现或开始。

    注意：进行模糊匹配，忽略页面文本中的任何空格不一致。

    给定的章节标题是 {title}。
    给定的页面文本是 {page_text}。
    
    回复格式：
    {{
        
        "thinking": <你认为该章节是否在页面文本中出现或开始的原因>
        "answer": "yes 或 no"（如果章节在页面文本中出现或开始则为 yes，否则为 no）
    }}
    直接返回最终的 JSON 结构。不要输出任何其他内容。"""

    response = await llm_acompletion(model=model, prompt=prompt)
    response = extract_json(response)
    if 'answer' in response:
        answer = response['answer']
    else:
        answer = 'no'
    return {'list_index': item['list_index'], 'answer': answer, 'title': title, 'page_number': page_number}


async def check_title_appearance_in_start(title, page_text, model=None, logger=None):    
    prompt = f"""
    你将获得当前章节标题和当前页面文本。
    你的任务是检查当前章节是否在给定页面文本的开头开始。
    如果在当前章节标题之前有其他内容，那么当前章节不在给定页面文本的开头开始。
    如果当前章节标题是给定页面文本中的第一个内容，那么当前章节在给定页面文本的开头开始。

    注意：进行模糊匹配，忽略页面文本中的任何空格不一致。

    给定的章节标题是 {title}。
    给定的页面文本是 {page_text}。
    
    回复格式：
    {{
        "thinking": <你认为该章节是否在页面文本中出现或开始的原因>
        "start_begin": "yes 或 no"（如果章节在页面文本开头开始则为 yes，否则为 no）
    }}
    直接返回最终的 JSON 结构。不要输出任何其他内容。"""

    response = await llm_acompletion(model=model, prompt=prompt)
    response = extract_json(response)
    logger = get_logger()
    if logger:
        logger.log("标题检查", f"响应: {response}")
    return response.get("start_begin", "no")


async def check_title_appearance_in_start_concurrent(structure, page_list, model=None):
    logger = get_logger()
    if logger:
        logger.log("标题检查", "并发检查标题开始出现")
    
    # skip items without physical_index
    for item in structure:
        if item.get('physical_index') is None:
            item['appear_start'] = 'no'

    # only for items with valid physical_index
    tasks = []
    valid_items = []
    for item in structure:
        if item.get('physical_index') is not None:
            page_text = page_list[item['physical_index'] - 1][0]
            tasks.append(check_title_appearance_in_start(item['title'], page_text, model=model))
            valid_items.append(item)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for item, result in zip(valid_items, results):
        if isinstance(result, Exception):
            if logger:
                logger.log("标题检查", f"检查 {item['title']} 开始时出错: {result}")
            item['appear_start'] = 'no'
        else:
            item['appear_start'] = result

    return structure


def toc_detector_single_page(content, model=None):
    # 策略一：先用规则判断
    rule_result = _rule_detect_toc_page(content)
    if rule_result is not None:
        _logger = get_logger()
        if _logger:
            _logger.log("目录检测", f"规则判断结果: {rule_result}（跳过LLM调用）")
        return rule_result

    prompt = f"""
    你的任务是检测给定文本中是否提供了目录。

    给定文本：{content}

    返回以下 JSON 格式：
    {{
        "thinking": <你认为给定文本中是否有目录的原因>
        "toc_detected": "<yes 或 no>",
    }}

    直接返回最终的 JSON 结构。不要输出任何其他内容。
    请注意：摘要、总结、符号列表、图表列表、表格列表等不是目录。"""

    response = llm_completion(model=model, prompt=prompt)
    json_content = extract_json(response)    
    return json_content.get('toc_detected', 'no')


def check_if_toc_extraction_is_complete(content, toc, model=None):
    prompt = f"""
    你获得了一份部分文档和一份目录。
    你的任务是检查目录是否完整，即是否包含了部分文档中的所有主要章节。

    回复格式：
    {{
        "thinking": <你认为目录是否完整的原因>
        "completed": "yes" 或 "no"
    }}
    直接返回最终的 JSON 结构。不要输出任何其他内容。"""

    prompt = prompt + '\n Document:\n' + content + '\n Table of contents:\n' + toc
    response = llm_completion(model=model, prompt=prompt)
    json_content = extract_json(response)
    return json_content['completed']


def check_if_toc_transformation_is_complete(content, toc, model=None):
    prompt = f"""
    你获得了一份原始目录和一份目录。
    你的任务是检查目录是否完整。

    回复格式：
    {{
        "thinking": <你认为清理后的目录是否完整的原因>
        "completed": "yes" 或 "no"
    }}
    直接返回最终的 JSON 结构。不要输出任何其他内容。"""

    prompt = prompt + '\n Raw Table of contents:\n' + content + '\n Cleaned Table of contents:\n' + toc
    response = llm_completion(model=model, prompt=prompt)
    json_content = extract_json(response)
    return json_content.get('completed', 'no')

def extract_toc_content(content, model=None):
    prompt = f"""
    你的任务是从给定文本中提取完整的目录，将 ... 替换为 :

    给定文本：{content}

    直接返回完整的目录内容。不要输出任何其他内容。"""

    response, finish_reason = llm_completion(model=model, prompt=prompt, return_finish_reason=True)
    
    if_complete = check_if_toc_transformation_is_complete(content, response, model)
    if if_complete == "yes" and finish_reason == "finished":
        return response
    
    chat_history = [
        {"role": "user", "content": prompt}, 
        {"role": "assistant", "content": response},    
    ]
    prompt = f"""请继续生成目录，直接输出结构的剩余部分"""
    new_response, finish_reason = llm_completion(model=model, prompt=prompt, chat_history=chat_history, return_finish_reason=True)
    response = response + new_response
    if_complete = check_if_toc_transformation_is_complete(content, response, model)
    
    attempt = 0
    max_attempts = 5

    while not (if_complete == "yes" and finish_reason == "finished"):
        attempt += 1
        if attempt > max_attempts:
            raise Exception('Failed to complete table of contents after maximum retries')

        chat_history = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        prompt = f"""请继续生成目录，直接输出结构的剩余部分"""
        new_response, finish_reason = llm_completion(model=model, prompt=prompt, chat_history=chat_history, return_finish_reason=True)
        response = response + new_response
        if_complete = check_if_toc_transformation_is_complete(content, response, model)
    
    return response

def detect_page_index(toc_content, model=None):
    logger = get_logger()
    if logger:
        logger.log("目录检查", "开始检测页面索引")
    # 策略一：先用规则判断
    rule_result = _rule_detect_page_index(toc_content)
    if rule_result is not None:
        if logger:
            logger.log("目录检查", f"规则判断页码存在: {rule_result}（跳过LLM调用）")
        return rule_result

    prompt = f"""
    你将获得一份目录。

    你的任务是检测目录中是否提供了页码/索引。

    给定文本：{toc_content}

    回复格式：
    {{
        "thinking": <你认为目录中是否提供了页码/索引的原因>
        "page_index_given_in_toc": "<yes 或 no>"
    }}
    直接返回最终的 JSON 结构。不要输出任何其他内容。"""

    response = llm_completion(model=model, prompt=prompt)
    json_content = extract_json(response)
    return json_content.get('page_index_given_in_toc', 'no')

def toc_extractor(page_list, toc_page_list, model):
    def transform_dots_to_colon(text):
        text = re.sub(r'\.{5,}', ': ', text)
        # Handle dots separated by spaces
        text = re.sub(r'(?:\. ){5,}\.?', ': ', text)
        return text
    
    toc_content = ""
    for page_index in toc_page_list:
        toc_content += page_list[page_index][0]
    toc_content = transform_dots_to_colon(toc_content)
    has_page_index = detect_page_index(toc_content, model=model)
    
    return {
        "toc_content": toc_content,
        "page_index_given_in_toc": has_page_index
    }




def toc_index_extractor(toc, content, model=None):
    logger = get_logger()
    if logger:
        logger.log("目录提取", "开始提取目录索引")
    toc_extractor_prompt = """
    你获得了一份 JSON 格式的目录和文档的几页内容，你的任务是将 physical_index 添加到 JSON 格式的目录中。

    提供的页面包含 <physical_index_X> 和 <physical_index_X> 这样的标签，用于指示页面 X 的物理位置。

    structure 变量是数字系统，表示目录中层级章节的索引。例如，第一节的结构索引为 1，第一个子节的结构索引为 1.1，第二个子节的结构索引为 1.2，依此类推。

    响应应采用以下 JSON 格式：
    [
        {
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>,
            "physical_index": "<physical_index_X>" (keep the format)
        },
        ...
    ]

    仅向提供的页面中存在的章节添加 physical_index。
    如果章节不在提供的页面中，不要向其添加 physical_index。
    直接返回最终的 JSON 结构。不要输出任何其他内容。"""

    prompt = toc_extractor_prompt + '\nTable of contents:\n' + str(toc) + '\nDocument pages:\n' + content
    response = llm_completion(model=model, prompt=prompt)
    json_content = extract_json(response)    
    return json_content



def toc_transformer(toc_content, model=None):
    logger = get_logger()
    if logger:
        logger.log("目录转换", "开始转换目录")
    # 策略三：Few-Shot 示例引导小模型输出正确 JSON
    init_prompt = """将下面的目录转为 JSON。

## 示例

输入：
第一章 绪论 : 1
  1.1 研究背景 : 2
  1.2 研究目的 : 5
第二章 方法 : 8

输出：
{"table_of_contents": [{"structure": "1", "title": "第一章 绪论", "page": 1}, {"structure": "1.1", "title": "1.1 研究背景", "page": 2}, {"structure": "1.2", "title": "1.2 研究目的", "page": 5}, {"structure": "2", "title": "第二章 方法", "page": 8}]}

## 规则
- structure 用数字层级（1, 1.1, 1.2, 2...）
- page 是页码数字，没有则填 null
- 一次性转换完整目录
- 只输出 JSON，不要输出任何其他内容"""

    prompt = init_prompt + '\n Given table of contents\n:' + toc_content
    last_complete, finish_reason = llm_completion(model=model, prompt=prompt, return_finish_reason=True)
    if_complete = check_if_toc_transformation_is_complete(toc_content, last_complete, model)
    if if_complete == "yes" and finish_reason == "finished":
        last_complete = extract_json(last_complete)
        cleaned_response=convert_page_to_int(last_complete['table_of_contents'])
        return cleaned_response
    
    last_complete = get_json_content(last_complete)
    attempt = 0
    max_attempts = 5
    while not (if_complete == "yes" and finish_reason == "finished"):
        attempt += 1
        if attempt > max_attempts:
            raise Exception('Failed to complete toc transformation after maximum retries')
        position = last_complete.rfind('}')
        if position != -1:
            last_complete = last_complete[:position+2]
        prompt = f"""
        你的任务是继续目录的 JSON 结构，直接输出 JSON 结构的剩余部分。
        响应应采用以下 JSON 格式：

        原始目录的 JSON 结构是：
        {toc_content}

        未完成的转换后目录 JSON 结构是：
        {last_complete}

        请继续 JSON 结构，直接输出 JSON 结构的剩余部分。"""

        new_complete, finish_reason = llm_completion(model=model, prompt=prompt, return_finish_reason=True)

        if new_complete.startswith('```json'):
            new_complete =  get_json_content(new_complete)
            last_complete = last_complete+new_complete

        if_complete = check_if_toc_transformation_is_complete(toc_content, last_complete, model)
        

    last_complete = extract_json(last_complete)

    cleaned_response=convert_page_to_int(last_complete['table_of_contents'])
    return cleaned_response
    



def find_toc_pages(start_page_index, page_list, opt):
    logger = get_logger()
    if logger:
        logger.log("目录检查", "开始查找目录页面")
    last_page_is_yes = False
    toc_page_list = []
    i = start_page_index
    
    while i < len(page_list):
        # Only check beyond max_pages if we're still finding TOC pages
        if i >= opt.toc_check_page_num and not last_page_is_yes:
            break
        detected_result = toc_detector_single_page(page_list[i][0],model=opt.model)
        if detected_result == 'yes':
            if logger:
                logger.log("目录检查", f"页面 {i} 有目录")
            toc_page_list.append(i)
            last_page_is_yes = True
        elif detected_result == 'no' and last_page_is_yes:
            if logger:
                logger.log("目录检查", f"找到最后一个有目录的页面: {i-1}")
            break
        i += 1
    
    if not toc_page_list and logger:
        logger.log("目录检查", "未找到目录")
        
    return toc_page_list

def remove_page_number(data):
    if isinstance(data, dict):
        data.pop('page_number', None)  
        for key in list(data.keys()):
            if 'nodes' in key:
                remove_page_number(data[key])
    elif isinstance(data, list):
        for item in data:
            remove_page_number(item)
    return data

def extract_matching_page_pairs(toc_page, toc_physical_index, start_page_index):
    pairs = []
    for phy_item in toc_physical_index:
        for page_item in toc_page:
            if phy_item.get('title') == page_item.get('title'):
                physical_index = phy_item.get('physical_index')
                if physical_index is not None and int(physical_index) >= start_page_index:
                    pairs.append({
                        'title': phy_item.get('title'),
                        'page': page_item.get('page'),
                        'physical_index': physical_index
                    })
    return pairs


def calculate_page_offset(pairs):
    differences = []
    for pair in pairs:
        try:
            physical_index = pair['physical_index']
            page_number = pair['page']
            difference = physical_index - page_number
            differences.append(difference)
        except (KeyError, TypeError):
            continue
    
    if not differences:
        return None
    
    difference_counts = {}
    for diff in differences:
        difference_counts[diff] = difference_counts.get(diff, 0) + 1
    
    most_common = max(difference_counts.items(), key=lambda x: x[1])[0]
    
    return most_common

def add_page_offset_to_toc_json(data, offset):
    for i in range(len(data)):
        if data[i].get('page') is not None and isinstance(data[i]['page'], int):
            data[i]['physical_index'] = data[i]['page'] + offset
            del data[i]['page']
    
    return data



def page_list_to_group_text(page_contents, token_lengths, max_tokens=4000, overlap_page=1):    
    num_tokens = sum(token_lengths)
    
    if num_tokens <= max_tokens:
        # merge all pages into one text
        page_text = "".join(page_contents)
        return [page_text]
    
    subsets = []
    current_subset = []
    current_token_count = 0

    expected_parts_num = math.ceil(num_tokens / max_tokens)
    average_tokens_per_part = math.ceil(((num_tokens / expected_parts_num) + max_tokens) / 2)
    
    for i, (page_content, page_tokens) in enumerate(zip(page_contents, token_lengths)):
        if current_token_count + page_tokens > average_tokens_per_part:

            subsets.append(''.join(current_subset))
            # Start new subset from overlap if specified
            overlap_start = max(i - overlap_page, 0)
            current_subset = page_contents[overlap_start:i]
            current_token_count = sum(token_lengths[overlap_start:i])
        
        # Add current page to the subset
        current_subset.append(page_content)
        current_token_count += page_tokens

    # Add the last subset if it contains any pages
    if current_subset:
        subsets.append(''.join(current_subset))
    
    logger = get_logger()
    if logger:
        logger.log("页面分组", f"将页面列表分成 {len(subsets)} 组")
    return subsets

def add_page_number_to_toc(part, structure, model=None):
    fill_prompt_seq = """
    你获得了文档的 JSON 结构和文档的部分内容。你的任务是检查结构中描述的标题是否在给定的部分文档中开始。

    提供的文本包含 <physical_index_X> 和 <physical_index_X> 这样的标签，用于指示页面 X 的物理位置。

    如果完整的目标章节在给定的部分文档中开始，插入带有 "start": "yes" 和 "start_index": "<physical_index_X>" 的给定 JSON 结构。

    如果完整的目标章节不在给定的部分文档中开始，插入 "start": "no"，"start_index": None。

    响应应采用以下格式。
        [
            {
                "structure": <structure index, "x.x.x" or None> (string),
                "title": <title of the section>,
                "start": "<yes or no>",
                "physical_index": "<physical_index_X> (keep the format)" or None
            },
            ...
        ]    
    给定的结构包含前一部分的结果，你需要填充当前部分的结果，不要更改之前的结果。
    直接返回最终的 JSON 结构。不要输出任何其他内容。"""

    prompt = fill_prompt_seq + f"\n\nCurrent Partial Document:\n{part}\n\nGiven Structure\n{json.dumps(structure, indent=2)}\n"
    current_json_raw = llm_completion(model=model, prompt=prompt)
    json_result = extract_json(current_json_raw)
    
    for item in json_result:
        if 'start' in item:
            del item['start']
    return json_result


def remove_first_physical_index_section(text):
    """
    Removes the first section between <physical_index_X> and <physical_index_X> tags,
    and returns the remaining text.
    """
    pattern = r'<physical_index_\d+>.*?<physical_index_\d+>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        # Remove the first matched section
        return text.replace(match.group(0), '', 1)
    return text

### add verify completeness
def generate_toc_continue(toc_content, part, model=None):
    logger = get_logger()
    if logger:
        logger.log("目录生成", "开始继续生成目录")
    # 策略五：只传最近5条结构作为上下文，避免上下文膨胀
    recent_items = toc_content[-5:] if len(toc_content) > 5 else toc_content
    # 策略三：Few-Shot 优化 prompt
    prompt = """继续从文本中提取层次树结构，返回**仅新增部分**的 JSON 数组。

## 示例

前一部分最后的结构：
[{"structure": "1.2", "title": "1.2 研究目的", "physical_index": "<physical_index_3>"}]

当前文本：
<physical_index_4>
第二章 方法
本章介绍研究方法...
<physical_index_4>

输出：
[{"structure": "2", "title": "第二章 方法", "physical_index": "<physical_index_4>"}]

## 规则
- structure 从上一部分的编号继续
- title 保留原始标题
- physical_index 保持 <physical_index_X> 格式
- 只输出新增章节的 JSON 数组，不要重复之前的内容"""

    prompt = prompt + '\nPrevious structure (last items):\n' + json.dumps(recent_items, ensure_ascii=False) + '\nGiven text:\n' + part
    response, finish_reason = llm_completion(model=model, prompt=prompt, return_finish_reason=True)
    result = extract_json(response)
    if isinstance(result, list) and len(result) > 0:
        return result
    if finish_reason == 'finished':
        return result if isinstance(result, list) else []
    else:
        if isinstance(result, list):
            return result
        logging.warning(f'generate_toc_continue finish_reason={finish_reason}, 返回空列表')
        return []

### add verify completeness
def generate_toc_init(part, model=None):
    logger = get_logger()
    if logger:
        logger.log("目录生成", "开始初始化生成目录")
    # 策略三：Few-Shot 示例引导小模型
    prompt = """从文本中提取层次树结构，返回 JSON 数组。

## 示例

输入：
<physical_index_1>
第一章 绪论
本章介绍研究背景...
<physical_index_1>

<physical_index_2>
1.1 研究背景
近年来人工智能快速发展...
第二章 方法
<physical_index_2>

输出：
[{"structure": "1", "title": "第一章 绪论", "physical_index": "<physical_index_1>"}, {"structure": "1.1", "title": "1.1 研究背景", "physical_index": "<physical_index_2>"}, {"structure": "2", "title": "第二章 方法", "physical_index": "<physical_index_2>"}]

## 规则
- structure 用数字层级（1, 1.1, 1.2, 2...）
- title 保留原始标题文字
- physical_index 保持 <physical_index_X> 格式
- 只输出 JSON 数组，不要输出任何其他内容"""

    prompt = prompt + '\nGiven text\n:' + part
    response, finish_reason = llm_completion(model=model, prompt=prompt, return_finish_reason=True)

    result = extract_json(response)
    if isinstance(result, list) and len(result) > 0:
        return result
    if finish_reason == 'finished':
        return result if isinstance(result, list) else []
    else:
        # max_output_reached: 尝试从截断的输出中提取已有的有效条目
        if isinstance(result, list):
            return result
        logging.warning(f'generate_toc_init finish_reason={finish_reason}, 返回空列表')
        return []

def process_no_toc(page_list, start_index=1, model=None):
    logger = get_logger()
    page_contents=[]
    token_lengths=[]
    for page_index in range(start_index, start_index+len(page_list)):
        page_text = f"<physical_index_{page_index}>\n{page_list[page_index-start_index][0]}\n<physical_index_{page_index}>\n\n"
        page_contents.append(page_text)
        token_lengths.append(count_tokens(page_text, model))
    group_texts = page_list_to_group_text(page_contents, token_lengths)
    if logger:
        logger.log("目录处理", f"group_texts 长度: {len(group_texts)}")

    toc_with_page_number= generate_toc_init(group_texts[0], model)
    for group_text in group_texts[1:]:
        toc_with_page_number_additional = generate_toc_continue(toc_with_page_number, group_text, model)    
        toc_with_page_number.extend(toc_with_page_number_additional)
    if logger:
        logger.log("目录处理", f"生成目录: {toc_with_page_number}")

    toc_with_page_number = convert_physical_index_to_int(toc_with_page_number)
    # 后处理：过滤小模型产生的无效条目
    toc_with_page_number = _filter_invalid_toc_entries(toc_with_page_number)
    if logger:
        logger.log("目录处理", f"过滤后目录: {toc_with_page_number}")

    return toc_with_page_number

def process_toc_no_page_numbers(toc_content, toc_page_list, page_list,  start_index=1, model=None):
    logger = get_logger()
    page_contents=[]
    token_lengths=[]
    toc_content = toc_transformer(toc_content, model)
    if logger:
        logger.log("目录处理", f"toc_transformer: {toc_content}")
    for page_index in range(start_index, start_index+len(page_list)):
        page_text = f"<physical_index_{page_index}>\n{page_list[page_index-start_index][0]}\n<physical_index_{page_index}>\n\n"
        page_contents.append(page_text)
        token_lengths.append(count_tokens(page_text, model))
    
    group_texts = page_list_to_group_text(page_contents, token_lengths)
    if logger:
        logger.log("目录处理", f"group_texts 长度: {len(group_texts)}")

    toc_with_page_number=copy.deepcopy(toc_content)
    for group_text in group_texts:
        toc_with_page_number = add_page_number_to_toc(group_text, toc_with_page_number, model)
    if logger:
        logger.log("目录处理", f"add_page_number_to_toc: {toc_with_page_number}")

    toc_with_page_number = convert_physical_index_to_int(toc_with_page_number)
    if logger:
        logger.log("目录处理", f"转换物理索引为整数: {toc_with_page_number}")

    return toc_with_page_number



def process_toc_with_page_numbers(toc_content, toc_page_list, page_list, toc_check_page_num=None, model=None):
    logger = get_logger()
    toc_with_page_number = toc_transformer(toc_content, model)
    if logger:
        logger.log("目录处理", f"toc_with_page_number: {toc_with_page_number}")

    toc_no_page_number = remove_page_number(copy.deepcopy(toc_with_page_number))
    
    start_page_index = toc_page_list[-1] + 1
    main_content = ""
    for page_index in range(start_page_index, min(start_page_index + toc_check_page_num, len(page_list))):
        main_content += f"<physical_index_{page_index+1}>\n{page_list[page_index][0]}\n<physical_index_{page_index+1}>\n\n"

    toc_with_physical_index = toc_index_extractor(toc_no_page_number, main_content, model)
    if logger:
        logger.log("目录处理", f"toc_with_physical_index: {toc_with_physical_index}")

    toc_with_physical_index = convert_physical_index_to_int(toc_with_physical_index)
    if logger:
        logger.log("目录处理", f"toc_with_physical_index: {toc_with_physical_index}")

    matching_pairs = extract_matching_page_pairs(toc_with_page_number, toc_with_physical_index, start_page_index)
    if logger:
        logger.log("目录处理", f"matching_pairs: {matching_pairs}")

    offset = calculate_page_offset(matching_pairs)
    if logger:
        logger.log("目录处理", f"offset: {offset}")

    toc_with_page_number = add_page_offset_to_toc_json(toc_with_page_number, offset)
    if logger:
        logger.log("目录处理", f"toc_with_page_number: {toc_with_page_number}")

    toc_with_page_number = process_none_page_numbers(toc_with_page_number, page_list, model=model)
    if logger:
        logger.log("目录处理", f"toc_with_page_number: {toc_with_page_number}")

    return toc_with_page_number



##check if needed to process none page numbers
def process_none_page_numbers(toc_items, page_list, start_index=1, model=None):
    for i, item in enumerate(toc_items):
        if "physical_index" not in item:
            # logger.info(f"fix item: {item}")
            # Find previous physical_index
            prev_physical_index = 0  # Default if no previous item exists
            for j in range(i - 1, -1, -1):
                if toc_items[j].get('physical_index') is not None:
                    prev_physical_index = toc_items[j]['physical_index']
                    break
            
            # Find next physical_index
            next_physical_index = -1  # Default if no next item exists
            for j in range(i + 1, len(toc_items)):
                if toc_items[j].get('physical_index') is not None:
                    next_physical_index = toc_items[j]['physical_index']
                    break

            page_contents = []
            for page_index in range(prev_physical_index, next_physical_index+1):
                # Add bounds checking to prevent IndexError
                list_index = page_index - start_index
                if list_index >= 0 and list_index < len(page_list):
                    page_text = f"<physical_index_{page_index}>\n{page_list[list_index][0]}\n<physical_index_{page_index}>\n\n"
                    page_contents.append(page_text)
                else:
                    continue

            item_copy = copy.deepcopy(item)
            del item_copy['page']
            result = add_page_number_to_toc(page_contents, item_copy, model)
            if isinstance(result[0]['physical_index'], str) and result[0]['physical_index'].startswith('<physical_index'):
                item['physical_index'] = int(result[0]['physical_index'].split('_')[-1].rstrip('>').strip())
                del item['page']
    
    return toc_items




def check_toc(page_list, opt=None):
    logger = get_logger()
    toc_page_list = find_toc_pages(start_page_index=0, page_list=page_list, opt=opt)
    if len(toc_page_list) == 0:
        if logger:
            logger.log("目录检查", "未找到目录")
        return {'toc_content': None, 'toc_page_list': [], 'page_index_given_in_toc': 'no'}
    else:
        if logger:
            logger.log("目录检查", "找到目录")
        toc_json = toc_extractor(page_list, toc_page_list, opt.model)

        if toc_json['page_index_given_in_toc'] == 'yes':
            if logger:
                logger.log("目录检查", "找到索引")
            return {'toc_content': toc_json['toc_content'], 'toc_page_list': toc_page_list, 'page_index_given_in_toc': 'yes'}
        else:
            current_start_index = toc_page_list[-1] + 1
            
            while (toc_json['page_index_given_in_toc'] == 'no' and 
                   current_start_index < len(page_list) and 
                   current_start_index < opt.toc_check_page_num):
                
                additional_toc_pages = find_toc_pages(
                    start_page_index=current_start_index,
                    page_list=page_list,
                    opt=opt
                )
                
                if len(additional_toc_pages) == 0:
                    break

                additional_toc_json = toc_extractor(page_list, additional_toc_pages, opt.model)
                if additional_toc_json['page_index_given_in_toc'] == 'yes':
                    if logger:
                        logger.log("目录检查", "找到索引")
                    return {'toc_content': additional_toc_json['toc_content'], 'toc_page_list': additional_toc_pages, 'page_index_given_in_toc': 'yes'}

                else:
                    current_start_index = additional_toc_pages[-1] + 1
            if logger:
                logger.log("目录检查", "未找到索引")
            return {'toc_content': toc_json['toc_content'], 'toc_page_list': toc_page_list, 'page_index_given_in_toc': 'no'}






################### fix incorrect toc #########################################################
async def single_toc_item_index_fixer(section_title, content, model=None):
    toc_extractor_prompt = """
    你获得了一个章节标题和文档的几页内容，你的任务是在部分文档中找到该章节开始页面的物理索引。

    提供的页面包含 <physical_index_X> 和 <physical_index_X> 这样的标签，用于指示页面 X 的物理位置。

    以 JSON 格式回复：
    {
        "thinking": <解释哪个由 <physical_index_X> 开始和结束的页面包含该章节的开始>,
        "physical_index": "<physical_index_X>" (保持格式)
    }
    直接返回最终的 JSON 结构。不要输出任何其他内容。"""

    prompt = toc_extractor_prompt + '\nSection Title:\n' + str(section_title) + '\nDocument pages:\n' + content
    response = await llm_acompletion(model=model, prompt=prompt)
    json_content = extract_json(response)    
    return convert_physical_index_to_int(json_content['physical_index'])



async def fix_incorrect_toc(toc_with_page_number, page_list, incorrect_results, start_index=1, model=None):
    logger = get_logger()
    if logger:
        logger.log("目录修复", f"开始修复不正确的目录，共 {len(incorrect_results)} 个不正确的结果")
    incorrect_indices = {result['list_index'] for result in incorrect_results}
    
    end_index = len(page_list) + start_index - 1
    
    incorrect_results_and_range_logs = []
    # Helper function to process and check a single incorrect item
    async def process_and_check_item(incorrect_item):
        list_index = incorrect_item['list_index']
        
        # Check if list_index is valid
        if list_index < 0 or list_index >= len(toc_with_page_number):
            # Return an invalid result for out-of-bounds indices
            return {
                'list_index': list_index,
                'title': incorrect_item['title'],
                'physical_index': incorrect_item.get('physical_index'),
                'is_valid': False
            }
        
        # Find the previous correct item
        prev_correct = None
        for i in range(list_index-1, -1, -1):
            if i not in incorrect_indices and i >= 0 and i < len(toc_with_page_number):
                physical_index = toc_with_page_number[i].get('physical_index')
                if physical_index is not None:
                    prev_correct = physical_index
                    break
        # If no previous correct item found, use start_index
        if prev_correct is None:
            prev_correct = start_index - 1
        
        # Find the next correct item
        next_correct = None
        for i in range(list_index+1, len(toc_with_page_number)):
            if i not in incorrect_indices and i >= 0 and i < len(toc_with_page_number):
                physical_index = toc_with_page_number[i].get('physical_index')
                if physical_index is not None:
                    next_correct = physical_index
                    break
        # If no next correct item found, use end_index
        if next_correct is None:
            next_correct = end_index
        
        incorrect_results_and_range_logs.append({
            'list_index': list_index,
            'title': incorrect_item['title'],
            'prev_correct': prev_correct,
            'next_correct': next_correct
        })

        page_contents=[]
        for page_index in range(prev_correct, next_correct+1):
            # Add bounds checking to prevent IndexError
            page_list_idx = page_index - start_index
            if page_list_idx >= 0 and page_list_idx < len(page_list):
                page_text = f"<physical_index_{page_index}>\n{page_list[page_list_idx][0]}\n<physical_index_{page_index}>\n\n"
                page_contents.append(page_text)
            else:
                continue
        content_range = ''.join(page_contents)
        
        physical_index_int = await single_toc_item_index_fixer(incorrect_item['title'], content_range, model)
        
        # Check if the result is correct
        check_item = incorrect_item.copy()
        check_item['physical_index'] = physical_index_int
        check_result = await check_title_appearance(check_item, page_list, start_index, model)

        return {
            'list_index': list_index,
            'title': incorrect_item['title'],
            'physical_index': physical_index_int,
            'is_valid': check_result['answer'] == 'yes'
        }

    # Process incorrect items concurrently
    tasks = [
        process_and_check_item(item)
        for item in incorrect_results
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for item, result in zip(incorrect_results, results):
        if isinstance(result, Exception):
            if logger:
                logger.log("目录修复", f"处理项目 {item} 时产生异常: {result}")
            continue
    results = [result for result in results if not isinstance(result, Exception)]

    # Update the toc_with_page_number with the fixed indices and check for any invalid results
    invalid_results = []
    for result in results:
        if result['is_valid']:
            # Add bounds checking to prevent IndexError
            list_idx = result['list_index']
            if 0 <= list_idx < len(toc_with_page_number):
                toc_with_page_number[list_idx]['physical_index'] = result['physical_index']
            else:
                # Index is out of bounds, treat as invalid
                invalid_results.append({
                    'list_index': result['list_index'],
                    'title': result['title'],
                    'physical_index': result['physical_index'],
                })
        else:
            invalid_results.append({
                'list_index': result['list_index'],
                'title': result['title'],
                'physical_index': result['physical_index'],
            })

    if logger:
        logger.log("目录修复", f"incorrect_results_and_range_logs: {incorrect_results_and_range_logs}")
        logger.log("目录修复", f"invalid_results: {invalid_results}")

    return toc_with_page_number, invalid_results



async def fix_incorrect_toc_with_retries(toc_with_page_number, page_list, incorrect_results, start_index=1, max_attempts=3, model=None):
    logger = get_logger()
    if logger:
        logger.log("目录修复", "开始修复不正确的目录")
    fix_attempt = 0
    current_toc = toc_with_page_number
    current_incorrect = incorrect_results

    while current_incorrect:
        if logger:
            logger.log("目录修复", f"修复 {len(current_incorrect)} 个不正确的结果")
        
        current_toc, current_incorrect = await fix_incorrect_toc(current_toc, page_list, current_incorrect, start_index, model)
                
        fix_attempt += 1
        if fix_attempt >= max_attempts:
            if logger:
                logger.log("目录修复", "达到最大修复尝试次数")
            break
    
    return current_toc, current_incorrect




################### verify toc #########################################################
async def verify_toc(page_list, list_result, start_index=1, N=None, model=None):
    logger = get_logger()
    if logger:
        logger.log("目录验证", "开始验证目录")
    # Find the last non-None physical_index
    last_physical_index = None
    for item in reversed(list_result):
        if item.get('physical_index') is not None:
            last_physical_index = item['physical_index']
            break
    
    # Early return if we don't have valid physical indices
    if last_physical_index is None or last_physical_index < len(page_list)/2:
        return 0, []
    
    # Determine which items to check
    if N is None:
        if logger:
            logger.log("目录验证", "检查所有项目")
        sample_indices = range(0, len(list_result))
    else:
        N = min(N, len(list_result))
        if logger:
            logger.log("目录验证", f"检查 {N} 个项目")
        sample_indices = random.sample(range(0, len(list_result)), N)

    # Prepare items with their list indices
    indexed_sample_list = []
    for idx in sample_indices:
        item = list_result[idx]
        # Skip items with None physical_index (these were invalidated by validate_and_truncate_physical_indices)
        if item.get('physical_index') is not None:
            item_with_index = item.copy()
            item_with_index['list_index'] = idx  # Add the original index in list_result
            indexed_sample_list.append(item_with_index)

    # Run checks concurrently
    tasks = [
        check_title_appearance(item, page_list, start_index, model)
        for item in indexed_sample_list
    ]
    results = await asyncio.gather(*tasks)
    
    # Process results
    correct_count = 0
    incorrect_results = []
    for result in results:
        if result['answer'] == 'yes':
            correct_count += 1
        else:
            incorrect_results.append(result)
    
    # Calculate accuracy
    checked_count = len(results)
    accuracy = correct_count / checked_count if checked_count > 0 else 0
    logger = get_logger()
    if logger:
        logger.log("目录验证", f"准确率: {accuracy*100:.2f}%")
    return accuracy, incorrect_results





################### main process #########################################################
async def meta_processor(page_list, mode=None, toc_content=None, toc_page_list=None, start_index=1, opt=None):
    logger = get_logger()
    if logger:
        logger.log("处理模式", f"模式: {mode}")
        logger.log("处理模式", f"开始索引: {start_index}")
    
    if mode == 'process_toc_with_page_numbers':
        toc_with_page_number = process_toc_with_page_numbers(toc_content, toc_page_list, page_list, toc_check_page_num=opt.toc_check_page_num, model=opt.model)
    elif mode == 'process_toc_no_page_numbers':
        toc_with_page_number = process_toc_no_page_numbers(toc_content, toc_page_list, page_list, model=opt.model)
    else:
        toc_with_page_number = process_no_toc(page_list, start_index=start_index, model=opt.model)
            
    toc_with_page_number = [item for item in toc_with_page_number if item.get('physical_index') is not None] 
    
    toc_with_page_number = validate_and_truncate_physical_indices(
        toc_with_page_number, 
        len(page_list), 
        start_index=start_index
    )
    
    accuracy, incorrect_results = await verify_toc(page_list, toc_with_page_number, start_index=start_index, model=opt.model)
        
    if logger:
        logger.log("处理结果", f"模式: process_toc_with_page_numbers")
        logger.log("处理结果", f"准确率: {accuracy}")
        logger.log("处理结果", f"错误结果: {incorrect_results}")
    if accuracy == 1.0 and len(incorrect_results) == 0:
        return toc_with_page_number
    if accuracy > 0.3 and len(incorrect_results) > 0:
        toc_with_page_number, incorrect_results = await fix_incorrect_toc_with_retries(toc_with_page_number, page_list, incorrect_results, start_index=start_index, max_attempts=3, model=opt.model)
        return toc_with_page_number
    else:
        if mode == 'process_toc_with_page_numbers':
            return await meta_processor(page_list, mode='process_toc_no_page_numbers', toc_content=toc_content, toc_page_list=toc_page_list, start_index=start_index, opt=opt)
        elif mode == 'process_toc_no_page_numbers':
            return await meta_processor(page_list, mode='process_no_toc', start_index=start_index, opt=opt)
        else:
            raise Exception('Processing failed')
        
 
async def process_large_node_recursively(node, page_list, opt=None):
    logger = get_logger()
    
    # 如果节点已经有子节点（来自原始 TOC 解析），跳过重新生成，避免重复子树
    if 'nodes' in node and node['nodes']:
        tasks = [
            process_large_node_recursively(child_node, page_list, opt)
            for child_node in node['nodes']
        ]
        await asyncio.gather(*tasks)
        return node
    
    node_page_list = page_list[node['start_index']-1:node['end_index']]
    token_num = sum([page[1] for page in node_page_list])
    
    if node['end_index'] - node['start_index'] > opt.max_page_num_each_node and token_num >= opt.max_token_num_each_node:
        if logger:
            logger.log("大节点处理", f"大节点: {node['title']} 开始索引: {node['start_index']} 结束索引: {node['end_index']} token数: {token_num}")

        node_toc_tree = await meta_processor(node_page_list, mode='process_no_toc', start_index=node['start_index'], opt=opt)
        node_toc_tree = await check_title_appearance_in_start_concurrent(node_toc_tree, page_list, model=opt.model)
        
        # Filter out items with None physical_index before post_processing
        valid_node_toc_items = [item for item in node_toc_tree if item.get('physical_index') is not None]
        
        if not valid_node_toc_items:
            return node
        
        # 模糊匹配：如果第一个子项标题与父节点标题相同/包含，跳过它
        if _titles_match(node['title'], valid_node_toc_items[0]['title']):
            remaining = valid_node_toc_items[1:]
            if remaining:
                node['nodes'] = post_processing(remaining, node['end_index'])
                node['end_index'] = remaining[0].get('start_index', node['end_index'])
        else:
            node['nodes'] = post_processing(valid_node_toc_items, node['end_index'])
            node['end_index'] = valid_node_toc_items[0].get('start_index', node['end_index'])
        
        # 递归处理新生成的子节点
        if 'nodes' in node and node['nodes']:
            tasks = [
                process_large_node_recursively(child_node, page_list, opt)
                for child_node in node['nodes']
            ]
            await asyncio.gather(*tasks)
    
    return node

def _titles_match(title_a, title_b):
    """模糊比较两个标题是否指同一章节（忽略空白，支持包含关系）"""
    a = re.sub(r'\s+', '', (title_a or '').strip().lower())
    b = re.sub(r'\s+', '', (title_b or '').strip().lower())
    if not a or not b:
        return False
    return a == b or a in b or b in a

async def tree_parser(page_list, opt, doc=None):
    check_toc_result = check_toc(page_list, opt)
    logger = get_logger()
    if logger:
        logger.log("目录检查", f"检查结果: {check_toc_result}")

    if check_toc_result.get("toc_content") and check_toc_result["toc_content"].strip() and check_toc_result["page_index_given_in_toc"] == "yes":
        toc_with_page_number = await meta_processor(
            page_list, 
            mode='process_toc_with_page_numbers', 
            start_index=1, 
            toc_content=check_toc_result['toc_content'], 
            toc_page_list=check_toc_result['toc_page_list'], 
            opt=opt)
    else:
        toc_with_page_number = await meta_processor(
            page_list, 
            mode='process_no_toc', 
            start_index=1, 
            opt=opt)

    toc_with_page_number = add_preface_if_needed(toc_with_page_number)
    toc_with_page_number = await check_title_appearance_in_start_concurrent(toc_with_page_number, page_list, model=opt.model)
    
    # Filter out items with None physical_index before post_processings
    valid_toc_items = [item for item in toc_with_page_number if item.get('physical_index') is not None]
    
    # 去重：移除标题和 physical_index 完全相同的重复条目
    seen = set()
    deduped_items = []
    for item in valid_toc_items:
        key = (item.get('title', '').strip(), item.get('physical_index'))
        if key not in seen:
            seen.add(key)
            deduped_items.append(item)
    valid_toc_items = deduped_items
    
    toc_tree = post_processing(valid_toc_items, len(page_list))
    
    # 合并重叠的根节点：如果第一个根节点的页面范围涵盖了后续根节点，
    # 将后续根节点合并为第一个根节点的子节点（修复 LLM 编号错误导致的重复子树）
    def _get_max_end_index(node):
        """递归获取节点及其所有后代的最大 end_index"""
        max_end = node.get('end_index', 0)
        for child in node.get('nodes', []):
            child_max = _get_max_end_index(child)
            if child_max > max_end:
                max_end = child_max
        return max_end
    
    if len(toc_tree) > 1:
        first_root = toc_tree[0]
        other_roots = list(toc_tree[1:])
        total_merged = 0
        
        # 迭代合并：每轮合并后 max_end 可能增大，需重新检查剩余根节点
        changed = True
        while changed and other_roots:
            changed = False
            first_root_max_end = _get_max_end_index(first_root)
            
            new_remaining = []
            for root_node in other_roots:
                root_start = root_node.get('start_index', 0)
                if root_start <= first_root_max_end + 1:
                    if 'nodes' not in first_root:
                        first_root['nodes'] = []
                    first_root['nodes'].append(root_node)
                    total_merged += 1
                    changed = True
                else:
                    new_remaining.append(root_node)
            other_roots = new_remaining
            
            if changed:
                first_root['end_index'] = _get_max_end_index(first_root)
        
        if total_merged > 0:
            toc_tree = [first_root] + other_roots
            if logger:
                logger.log("树合并", f"合并了 {total_merged} 个重叠根节点到第一个根节点下")
    
    tasks = [
        process_large_node_recursively(node, page_list, opt)
        for node in toc_tree
    ]
    await asyncio.gather(*tasks)
    
    return toc_tree


async def page_index_main(doc, opt=None):
    structured_logger = StructuredLogger(doc)
    set_logger(structured_logger)
    
    is_valid_pdf = (
        (isinstance(doc, str) and os.path.isfile(doc) and doc.lower().endswith(".pdf")) or 
        isinstance(doc, BytesIO)
    )
    if not is_valid_pdf:
        raise ValueError("不支持的输入类型。期望 PDF 文件路径或 BytesIO 对象。")

    structured_logger.log("PDF解析", "正在解析PDF...")
    page_list = get_page_tokens(doc, model=opt.model)

    structured_logger.log("PDF解析", f"总页数: {len(page_list)}")

    structure = await tree_parser(page_list, opt, doc=doc)
    if opt.if_add_node_id == 'yes':
        write_node_id(structure)    
    if opt.if_add_node_text == 'yes':
        add_node_text(structure, page_list)
    if opt.if_add_node_summary == 'yes':
        if opt.if_add_node_text == 'no':
            add_node_text(structure, page_list)
        await generate_summaries_for_structure(structure, model=opt.model)
        if opt.if_add_node_text == 'no':
            remove_structure_text(structure)
        if opt.if_add_doc_description == 'yes':
            # 创建一个精简的结构，用于描述生成，去除不必要的字段。
            clean_structure = create_clean_structure_for_description(structure)
            doc_description = generate_doc_description(clean_structure, model=opt.model)
            structure = format_structure(structure, order=['title', 'node_id', 'start_index', 'end_index', 'summary', 'text', 'nodes'])
            result = {
                'doc_name': get_pdf_name(doc),
                'doc_description': doc_description,
                'structure': structure,
            }
        else:
            structure = format_structure(structure, order=['title', 'node_id', 'start_index', 'end_index', 'summary', 'text', 'nodes'])
            result = {
                'doc_name': get_pdf_name(doc),
                'structure': structure,
            }
    else:
        structure = format_structure(structure, order=['title', 'node_id', 'start_index', 'end_index', 'summary', 'text', 'nodes'])
        result = {
            'doc_name': get_pdf_name(doc),
            'structure': structure,
        }

    # 保存结构到临时文件
    import tempfile
    pdf_name = get_pdf_name(doc)
    temp_dir = "./pdf/"
    json_save_path = os.path.join(temp_dir, f"{pdf_name}.json")
    
    with open(json_save_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"[SUCCESS] 完整召回数据已存至: {os.path.abspath(json_save_path)}")
    
    # 返回文件路径
    return json_save_path


async def page_index(doc, model=None, toc_check_page_num=None, max_page_num_each_node=None, max_token_num_each_node=None,
               if_add_node_id=None, if_add_node_summary=None, if_add_doc_description=None, if_add_node_text=None):
    
    user_opt = {
        arg: value for arg, value in locals().items()
        if arg != "doc" and value is not None
    }
    opt = ConfigLoader().load(user_opt)
    return await page_index_main(doc, opt)


def validate_and_truncate_physical_indices(toc_with_page_number, page_list_length, start_index=1):
    """
    Validates and truncates physical indices that exceed the actual document length.
    This prevents errors when TOC references pages that don't exist in the document (e.g. the file is broken or incomplete).
    """
    logger = get_logger()
    if not toc_with_page_number:
        return toc_with_page_number
    
    max_allowed_page = page_list_length + start_index - 1
    truncated_items = []
    
    for i, item in enumerate(toc_with_page_number):
        if item.get('physical_index') is not None:
            original_index = item['physical_index']
            if original_index > max_allowed_page:
                item['physical_index'] = None
                truncated_items.append({
                    'title': item.get('title', 'Unknown'),
                    'original_index': original_index
                })
                if logger:
                    logger.log("文档验证", f"移除了 '{item.get('title', 'Unknown')}' 的 physical_index (原索引: {original_index}, 超出文档范围)")
    
    if truncated_items and logger:
        logger.log("文档验证", f"总共移除的项目: {len(truncated_items)}")
        
    if logger:
        logger.log("文档验证", f"文档验证: {page_list_length} 页，最大允许索引: {max_allowed_page}")
        if truncated_items:
            logger.log("文档验证", f"截断了 {len(truncated_items)} 个超出文档长度的目录项目")
     
    return toc_with_page_number