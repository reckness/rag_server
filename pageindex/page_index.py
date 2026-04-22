import os
import json
import copy
import math
import random
import re
import logging
from datetime import datetime
from utils import *
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


################### check title in page #########################################################
async def check_title_appearance(item, page_list, start_index=1, model=None):    
    title=item['title']
    if 'physical_index' not in item or item['physical_index'] is None:
        return {'list_index': item.get('list_index'), 'answer': 'no', 'title':title, 'page_number': None}
    
    
    page_number = item['physical_index']
    page_text = page_list[page_number-start_index][0]

    
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
    # print('response', response)
    json_content = extract_json(response)    
    return json_content['toc_detected']


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
    return json_content['page_index_given_in_toc']

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
    init_prompt = """
    你获得了一份目录，你的任务是将整个目录转换为包含 table_of_contents 的 JSON 格式。

    structure 是数字系统，表示目录中层级章节的索引。例如，第一节的结构索引为 1，第一个子节的结构索引为 1.1，第二个子节的结构索引为 1.2，依此类推。

    响应应采用以下 JSON 格式：
    {
    table_of_contents: [
        {
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>,
            "page": <page number or None>,
        },
        ...
        ],
    }
    你应该一次性转换完整的目录。
    直接返回最终的 JSON 结构，不要输出任何其他内容。"""

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



def page_list_to_group_text(page_contents, token_lengths, max_tokens=20000, overlap_page=1):    
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
    prompt = """
    你是提取层次树结构的专家。
    你获得了前一部分的树结构和当前部分的文本。
    你的任务是从前一部分继续树结构，以包含当前部分。

    structure 变量是数字系统，表示目录中层级章节的索引。例如，第一节的结构索引为 1，第一个子节的结构索引为 1.1，第二个子节的结构索引为 1.2，依此类推。

    对于标题，你需要从文本中提取原始标题，只修复空格不一致的问题。

    提供的文本包含 <physical_index_X> 和 <physical_index_X> 这样的标签，用于指示页面 X 的开始和结束。
    
    对于 physical_index，你需要从文本中提取章节开始的物理索引。保持 <physical_index_X> 格式。

    响应应采用以下格式。
        [
            {
                "structure": <structure index, "x.x.x"> (string),
                "title": <title of the section, keep the original title>,
                "physical_index": "<physical_index_X> (keep the format)"
            },
            ...
        ]    

    直接返回最终 JSON 结构的附加部分。不要输出任何其他内容。"""

    prompt = prompt + '\nGiven text\n:' + part + '\nPrevious tree structure\n:' + json.dumps(toc_content, indent=2)
    response, finish_reason = llm_completion(model=model, prompt=prompt, return_finish_reason=True)
    if finish_reason == 'finished':
        return extract_json(response)
    else:
        raise Exception(f'finish reason: {finish_reason}')
    
### add verify completeness
def generate_toc_init(part, model=None):
    logger = get_logger()
    if logger:
        logger.log("目录生成", "开始初始化生成目录")
    prompt = """
    你是提取层次树结构的专家，你的任务是生成文档的树结构。

    structure 变量是数字系统，表示目录中层级章节的索引。例如，第一节的结构索引为 1，第一个子节的结构索引为 1.1，第二个子节的结构索引为 1.2，依此类推。

    对于标题，你需要从文本中提取原始标题，只修复空格不一致的问题。

    提供的文本包含 <physical_index_X> 和 <physical_index_X> 这样的标签，用于指示页面 X 的开始和结束。

    对于 physical_index，你需要从文本中提取章节开始的物理索引。保持 <physical_index_X> 格式。

    响应应采用以下格式。
        [
            {{
                "structure": <structure index, "x.x.x"> (string),
                "title": <title of the section, keep the original title>,
                "physical_index": "<physical_index_X> (keep the format)"
            }},
            
        ],


    直接返回最终的 JSON 结构。不要输出任何其他内容。"""

    prompt = prompt + '\nGiven text\n:' + part
    response, finish_reason = llm_completion(model=model, prompt=prompt, return_finish_reason=True)

    if finish_reason == 'finished':
         return extract_json(response)
    else:
        raise Exception(f'finish reason: {finish_reason}')

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
    if logger:
        logger.log("目录处理", f"转换物理索引为整数: {toc_with_page_number}")

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
    if accuracy > 0.6 and len(incorrect_results) > 0:
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
    node_page_list = page_list[node['start_index']-1:node['end_index']]
    token_num = sum([page[1] for page in node_page_list])
    
    if node['end_index'] - node['start_index'] > opt.max_page_num_each_node and token_num >= opt.max_token_num_each_node:
        if logger:
            logger.log("大节点处理", f"大节点: {node['title']} 开始索引: {node['start_index']} 结束索引: {node['end_index']} token数: {token_num}")

        node_toc_tree = await meta_processor(node_page_list, mode='process_no_toc', start_index=node['start_index'], opt=opt)
        node_toc_tree = await check_title_appearance_in_start_concurrent(node_toc_tree, page_list, model=opt.model)
        
        # Filter out items with None physical_index before post_processing
        valid_node_toc_items = [item for item in node_toc_tree if item.get('physical_index') is not None]
        
        if valid_node_toc_items and node['title'].strip() == valid_node_toc_items[0]['title'].strip():
            node['nodes'] = post_processing(valid_node_toc_items[1:], node['end_index'])
            node['end_index'] = valid_node_toc_items[1]['start_index'] if len(valid_node_toc_items) > 1 else node['end_index']
        else:
            node['nodes'] = post_processing(valid_node_toc_items, node['end_index'])
            node['end_index'] = valid_node_toc_items[0]['start_index'] if valid_node_toc_items else node['end_index']
        
    if 'nodes' in node and node['nodes']:
        tasks = [
            process_large_node_recursively(child_node, page_list, opt)
            for child_node in node['nodes']
        ]
        await asyncio.gather(*tasks)
    
    return node

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
    
    toc_tree = post_processing(valid_toc_items, len(page_list))
    tasks = [
        process_large_node_recursively(node, page_list, opt)
        for node in toc_tree
    ]
    await asyncio.gather(*tasks)
    
    return toc_tree


def page_index_main(doc, opt=None):
    structured_logger = StructuredLogger(doc)
    set_logger(structured_logger)
    
    is_valid_pdf = (
        (isinstance(doc, str) and os.path.isfile(doc) and doc.lower().endswith(".pdf")) or 
        isinstance(doc, BytesIO)
    )
    if not is_valid_pdf:
        raise ValueError("Unsupported input type. Expected a PDF file path or BytesIO object.")

    structured_logger.log("PDF解析", "正在解析PDF...")
    page_list = get_page_tokens(doc, model=opt.model)

    structured_logger.log("PDF解析", f"总页数: {len(page_list)}")
    structured_logger.log("PDF解析", f"总token数: {sum([page[1] for page in page_list])}")

    async def page_index_builder():
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
                # Create a clean structure without unnecessary fields for description generation
                clean_structure = create_clean_structure_for_description(structure)
                doc_description = generate_doc_description(clean_structure, model=opt.model)
                structure = format_structure(structure, order=['title', 'node_id', 'start_index', 'end_index', 'summary', 'text', 'nodes'])
                return {
                    'doc_name': get_pdf_name(doc),
                    'doc_description': doc_description,
                    'structure': structure,
                }
        structure = format_structure(structure, order=['title', 'node_id', 'start_index', 'end_index', 'summary', 'text', 'nodes'])
        return {
            'doc_name': get_pdf_name(doc),
            'structure': structure,
        }

    return asyncio.run(page_index_builder())


def page_index(doc, model=None, toc_check_page_num=None, max_page_num_each_node=None, max_token_num_each_node=None,
               if_add_node_id=None, if_add_node_summary=None, if_add_doc_description=None, if_add_node_text=None):
    
    user_opt = {
        arg: value for arg, value in locals().items()
        if arg != "doc" and value is not None
    }
    opt = ConfigLoader().load(user_opt)
    return page_index_main(doc, opt)


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