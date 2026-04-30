"""
重新处理所有 uploaded 状态的文档（跳过上传步骤）
"""
import os
import sys
import json
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import text
from app.core.database import SessionLocal

KB_ID = "batch_test_kb_001"
API_BASE = "http://localhost:8000"

db = SessionLocal()
rows = db.execute(text(
    "SELECT doc_id, title FROM document WHERE kb_id = :kb AND status = 'uploaded' ORDER BY create_time"
), {"kb": KB_ID}).fetchall()
db.close()

print(f"共 {len(rows)} 个待处理文档\n")

success = 0
failed = 0
for i, (doc_id, title) in enumerate(rows):
    print(f"\n[{i+1}/{len(rows)}] {title}")
    print(f"  doc_id: {doc_id}")
    try:
        resp = requests.post(
            f"{API_BASE}/universal/process_bat/{doc_id}",
            stream=True,
            timeout=1800,
        )
        last_progress = 0
        for line in resp.iter_lines(decode_unicode=True):
            if line and line.startswith("data:"):
                data = json.loads(line[5:].strip())
                p = data.get("progress", 0)
                if p > last_progress:
                    last_progress = p
                    print(f"  进度: {p}%", end="\r")
        if last_progress >= 100:
            print(f"  ✓ 完成 (100%)")
            success += 1
        elif last_progress == -1:
            print(f"  ✗ 处理失败")
            failed += 1
        else:
            print(f"  ? 进度停在 {last_progress}%")
            failed += 1
    except Exception as e:
        print(f"  ✗ 请求失败: {e}")
        failed += 1

print(f"\n{'='*50}")
print(f"处理完成: 成功 {success}, 失败 {failed}, 总计 {len(rows)}")
