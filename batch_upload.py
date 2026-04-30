"""
批量上传 dataset 目录下所有 PDF 到同一个知识库，
然后逐个调用 process_bat 处理。
"""
import os
import sys
import uuid
import time
import requests
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import text
from app.core.database import SessionLocal
from app.services.minio_service import MinioService

DATASET_DIR = "/home/rag/dataset/dataset"
BUCKET_NAME = "deepsearch"
KB_ID = "batch_test_kb_001"
FD_ID = "0"
API_BASE = "http://localhost:8000"

# 收集所有 PDF
pdf_files = []
for root, dirs, files in os.walk(DATASET_DIR):
    for f in files:
        if f.lower().endswith(".pdf"):
            pdf_files.append(os.path.join(root, f))

pdf_files.sort()
print(f"共找到 {len(pdf_files)} 个 PDF 文件\n")

minio = MinioService()
minio.ensure_bucket_exists(BUCKET_NAME)
db = SessionLocal()

doc_ids = []

# 1. 上传文件到 MinIO + 创建数据库记录
for i, pdf_path in enumerate(pdf_files):
    title = os.path.basename(pdf_path)
    file_size = os.path.getsize(pdf_path)
    doc_id = uuid.uuid4().hex

    # MinIO object path
    now = datetime.now()
    object_name = f"knowledge/file/{now.strftime('%Y/%m/%d')}/{uuid.uuid4()}.pdf"

    print(f"[{i+1}/{len(pdf_files)}] 上传: {title} ({file_size/1024/1024:.1f}MB)")
    minio.upload_file(BUCKET_NAME, object_name, pdf_path, content_type="application/pdf")

    # 创建数据库记录
    db.execute(text("""
        INSERT INTO document (doc_id, kb_id, fd_id, title, file_type, file_size, source_path, status, chunk_num, progress, process_duration, llm_token, create_time)
        VALUES (:doc_id, :kb_id, :fd_id, :title, :file_type, :file_size, :source_path, :status, 0, 0.0, 0.0, 0, :create_time)
    """), {
        "doc_id": doc_id,
        "kb_id": KB_ID,
        "fd_id": FD_ID,
        "title": title,
        "file_type": "pdf",
        "file_size": file_size,
        "source_path": object_name,
        "status": "uploaded",
        "create_time": now,
    })
    db.commit()
    doc_ids.append((doc_id, title))

print(f"\n✓ 全部上传完成，共 {len(doc_ids)} 个文档记录\n")

# 2. 逐个调用 process_bat 处理
success = 0
failed = 0
for i, (doc_id, title) in enumerate(doc_ids):
    print(f"\n[{i+1}/{len(doc_ids)}] 处理: {title}")
    print(f"  doc_id: {doc_id}")
    try:
        resp = requests.post(
            f"{API_BASE}/universal/process_bat/{doc_id}",
            stream=True,
            timeout=1800,  # 30分钟超时
        )
        last_progress = 0
        for line in resp.iter_lines(decode_unicode=True):
            if line and line.startswith("data:"):
                import json
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

db.close()
print(f"\n{'='*50}")
print(f"处理完成: 成功 {success}, 失败 {failed}, 总计 {len(doc_ids)}")
