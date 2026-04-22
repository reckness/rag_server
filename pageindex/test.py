import requests
import json
import time

# 修复URL中的反引号
url = "http://10.1.141.33:8001/v1/chat/completions"

# 记录开始时间
start_time = time.time()

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer "
}

payload = {
    "model": "Qwen3-8B",
    "messages": [
        {"role": "user", "content": "你好，介绍一下你自己"}
    ],
    "chat_template_kwargs": {
        "enable_thinking": False
    },
}
print("开始发送请求...")
print(f"URL: {url}")
print(f"Headers: {headers}")
print(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")

try:
    # 发送请求
    print("\n正在发送POST请求...")
    response = requests.post(url, headers=headers, json=payload, timeout=300)
    print(f"响应状态码: {response.status_code}")
    print(f"响应头: {dict(response.headers)}")
    print(f"响应内容长度: {len(response.text)}")
    
    response.raise_for_status()  # 检查响应状态码
    
    # 解析响应
    print("\n正在解析响应...")
    data = response.json()
    
    print("\n===== 原始返回 =====")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    
    print("\n===== 模型回答 =====")
    print(data["choices"][0]["message"]["content"])
    
    # 计算并打印总调用时间
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n===== 调用总时间 =====")
    print(f"总调用时间: {total_time:.2f} 秒")
except requests.exceptions.RequestException as e:
    print(f"\n请求失败: {e}")
    import traceback
    traceback.print_exc()
except json.JSONDecodeError as e:
    print(f"\n解析响应失败: {e}")
    print("原始响应内容:")
    print(response.text[:500] + "..." if len(response.text) > 500 else response.text)
    import traceback
    traceback.print_exc()
except KeyError as e:
    print(f"\n响应格式错误: {e}")
    print("原始响应内容:")
    print(response.text)
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"\n其他错误: {e}")
    import traceback
    traceback.print_exc()