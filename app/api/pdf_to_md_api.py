import os
import sys
import tempfile
from typing import Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..core.database import SessionLocal
from ..utils.response import ApiResponse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.config import ELASTICSEARCH_INDEX
from rag.json_to_es_converter_with_embedding import ESConverter
from rag.build_router_es import DocumentRouter


router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# POST /pdf/process/{doc_id}
# 从数据库获取文档 → MinIO 下载 → run_pdf_to_md_chunked 解析 → 写入 ES → 更新数据库
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "pdf")


@router.post("/process/{doc_id}", summary="处理文档（PDF解析 + 写入ES + 更新数据库）")
async def process_document(doc_id: str, db: Session = Depends(get_db)):
    """
    完整处理流程：

    1. 查询数据库获取文档信息（doc_id, source_path, kb_id, fd_id）
    2. 从 MinIO 下载 PDF
    3. run_pdf_to_md_chunked 解析生成结构化 JSON
    4. ESConverter 写入 ES 向量索引
    5. DocumentRouter 写入文档路由
    6. 上传 JSON 到 MinIO，更新数据库状态
    """
    from ..repository.document_repository import DocumentRepository
    from ..services.minio_service import MinioService
    from common.doc_store.es_conn_pool import ES_CONN

    # 1. 查询文档记录
    document = DocumentRepository.get_by_id(db, doc_id)
    if not document:
        return ApiResponse.error(code=404, message=f"ID 为 {doc_id} 的文档不存在")

    if not document.source_path:
        return ApiResponse.error(code=404, message="文档文件路径不存在")

    process_begin_at = datetime.now()
    progress = 0.0

    DocumentRepository.update(
        db, document.doc_id,
        progress=0.0,
        progress_msg="开始处理文档",
        process_begin_at=process_begin_at,
        process_duration=0.0,
    )

    temp_file_path = None

    try:
        # 2. 从 MinIO 下载文件
        minio_service = MinioService()
        bucket_name = "deepsearch"
        suffix = os.path.splitext(document.title)[1] or ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            temp_file_path = tmp.name

        minio_service.download_file(bucket_name, document.source_path, temp_file_path)
        progress = 20.0
        DocumentRepository.update(db, document.doc_id, progress=progress, progress_msg="文件下载完成，正在分析文档大小")

        # 3. 根据文档 token 数选择处理方式
        import fitz
        pdf_doc = fitz.open(temp_file_path)
        full_text = "".join(page.get_text() for page in pdf_doc)
        pdf_doc.close()
        # 中文约 1.5 字符/token
        estimated_tokens = int(len(full_text) / 1.5)

        LLM_URL = "http://10.1.141.33:8001/v1/chat/completions"
        LLM_MODEL = "Qwen3-8B"

        if estimated_tokens <= 8000:
            from run_pdf_to_md import process_pdf_simple
            output = await process_pdf_simple(
                pdf_path=temp_file_path,
                output_dir=OUTPUT_DIR,
                llm_url=LLM_URL, llm_model=LLM_MODEL, model=LLM_MODEL,
                if_summary=False, if_add_node_text=True,
            )
        else:
            from run_pdf_to_md_chunked import process_pdf_chunked
            output = await process_pdf_chunked(
                pdf_path=temp_file_path,
                output_dir=OUTPUT_DIR,
                llm_url=LLM_URL, llm_model=LLM_MODEL, model=LLM_MODEL,
                if_summary=False, if_add_node_text=True,
            )
        json_path = output["json_path"]

        # 上传 JSON 到 MinIO
        json_object_name = f"{document.doc_id}/{document.doc_id}.json"
        minio_service.upload_file(bucket_name, json_object_name, json_path)

        progress = 50.0
        DocumentRepository.update(
            db, document.doc_id,
            progress=progress,
            pageindex_path=json_object_name,
            progress_msg="文件解析完成",
        )

        # 4. ESConverter 写入 ES
        d_id = str(document.doc_id)
        k_id = str(document.kb_id)
        f_id = str(document.fd_id)

        converter = ESConverter(
            json_path,
            index_name=ELASTICSEARCH_INDEX,
            doc_id=d_id, kb_id=k_id, fd_id=f_id,
            doc_title=document.title,
        )
        converter.run()
        chunk_num = len(converter.flat_nodes)

        progress = 75.0
        DocumentRepository.update(
            db, document.doc_id,
            progress=progress, chunk_num=chunk_num,
            progress_msg="向量转换完成",
        )

        # 5. DocumentRouter 写入文档路由
        doc_router = DocumentRouter(
            json_path,
            doc_id=d_id, kb_id=k_id, fd_id=f_id,
            doc_title=document.title,
            flat_nodes=converter.flat_nodes,
        )
        doc_router.run()

        progress = 90.0
        DocumentRepository.update(db, document.doc_id, progress=progress, progress_msg="文档路由生成完成")

        # 刷新 ES 索引
        es = ES_CONN.get_conn()
        es.indices.refresh(index=ELASTICSEARCH_INDEX)
        es.indices.refresh(index="doc_summary_index")

        # 6. 完成
        process_duration = (datetime.now() - process_begin_at).total_seconds()
        DocumentRepository.update(
            db, document.doc_id,
            progress=100.0,
            progress_msg=f"处理完成，耗时 {process_duration:.2f} 秒",
            process_duration=process_duration,
            status="ready",
        )

        # 清理临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        mode_used = "simple" if estimated_tokens <= 8000 else "chunked"
        return ApiResponse.success(data={
            "success": True,
            "document_id": document.doc_id,
            "kb_id": k_id,
            "chunk_num": chunk_num,
            "estimated_tokens": estimated_tokens,
            "mode": mode_used,
            "json_path": json_path,
            "md_path": output.get("md_path"),
            "process_duration": process_duration,
        })

    except Exception as e:
        DocumentRepository.update(
            db, document.doc_id,
            progress=progress,
            progress_msg=f"处理失败: {str(e)}",
            status="error",
        )
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return ApiResponse.error(code=500, message=f"处理文档失败: {str(e)}")

