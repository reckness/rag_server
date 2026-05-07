"""
通用多格式文档 RAG API
提供文档处理端点，支持 PDF / Word / TXT / PPT / Excel / Markdown 等格式。
"""
import os
import sys
import json
import asyncio
import tempfile
from datetime import datetime
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import Dict, Any, AsyncGenerator, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.config import ELASTICSEARCH_INDEX, MINIO_IP, MINIO_PORT

from ..core.database import SessionLocal
from ..services.universal_rag_service import UniversalRagService
from ..services.minio_service import MinioService
from ..services.file_converter import convert_to_pdf, is_supported, SUPPORTED_EXTENSIONS
from ..repository.document_repository import DocumentRepository
from ..utils.response import ApiResponse
from ..core.exceptions import NotFoundException, InternalServerErrorException
from rag.multi_index_writer import write_to_three_indices

router = APIRouter()

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "pdf")
BUCKET_NAME = "deepsearch"
LLM_URL = "http://10.1.141.33:8080/v1/chat/completions"
LLM_MODEL = "qwen3.5-35b-int4"
PROCESS_BAT_SEMAPHORE = asyncio.Semaphore(2)


def _find_json_nodes_by_id(nodes, target_id: str, id_fields, path=None):
    path = path or []
    matches = []
    for node in nodes or []:
        current_path = path + [{
            "title": node.get("title", ""),
            "node_id": node.get("node_id", ""),
            "start_page": node.get("start_page"),
            "end_page": node.get("end_page"),
        }]
        if any(str(node.get(field, "")) == str(target_id) for field in id_fields):
            matches.append({
                "path": current_path,
                "node": node,
            })
        matches.extend(_find_json_nodes_by_id(
            node.get("nodes") or [],
            target_id,
            id_fields,
            current_path,
        ))
    return matches


def _strip_children(node: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(node)
    cleaned.pop("nodes", None)
    return cleaned


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post(
    "/process/{doc_id}",
    response_model=ApiResponse[Dict[str, Any]],
    summary="通用多格式文档处理（RAG）",
)
async def process_document_universal(
    doc_id: str,
    use_llm_for_pdf_no_toc: bool = Query(
        default=True,
        description=(
            "对无目录 PDF 是否使用 LLM 生成目录结构。\n"
            "True（默认）：LLM 驱动，切分效果更好，消耗 token；\n"
            "False：按字符分块，速度更快、零 token 消耗。"
        ),
    ),
    db: Session = Depends(get_db),
):
    """
    通用多格式文档处理接口。

    支持格式：**PDF、Word (.doc/.docx)、TXT、PPT (.ppt/.pptx)、Excel (.xls/.xlsx)、Markdown (.md)**

    处理流程：
    1. 从 MinIO 下载文件
    2. 根据文件扩展名分发到对应解析器
       - PDF：先检测是否含目录（TOC）
         - 有目录 → page_index LLM 精准切分
         - 无目录 + `use_llm_for_pdf_no_toc=True` → LLM 生成目录后切分
         - 无目录 + `use_llm_for_pdf_no_toc=False` → 按字符快速分块
       - 其他格式 → 格式专属解析器提取结构
    3. 写入 Elasticsearch（向量索引 + 文档路由）
    4. 更新文档处理状态
    """
    try:
        result = await UniversalRagService.process_document(
            db,
            doc_id,
            use_llm_for_pdf_no_toc=use_llm_for_pdf_no_toc,
        )
        return ApiResponse.success(data=result)
    except NotFoundException as e:
        return ApiResponse.error(code=404, message=e.detail)
    except InternalServerErrorException as e:
        return ApiResponse.error(code=500, message=e.detail)
    except Exception as e:
        return ApiResponse.error(code=500, message=f"服务器内部错误: {str(e)}")


@router.get(
    "/supported-formats",
    response_model=ApiResponse[Dict[str, Any]],
    summary="查询支持的文件格式",
)
def get_supported_formats():
    """返回当前支持的文件格式列表及说明"""
    return ApiResponse.success(
        data={
            "formats": {
                "pdf": {
                    "extensions": [".pdf"],
                    "description": "PDF 文档，自动检测目录（TOC）并选择最优切分策略",
                },
                "word": {
                    "extensions": [".docx", ".doc"],
                    "description": "Word 文档，按标题样式（Heading 1-6）切分章节",
                },
                "txt": {
                    "extensions": [".txt"],
                    "description": "纯文本，自动检测编码，按固定字符数分块",
                },
                "ppt": {
                    "extensions": [".pptx", ".ppt"],
                    "description": "演示文稿，每张幻灯片作为一个节点",
                },
                "excel": {
                    "extensions": [".xlsx", ".xls"],
                    "description": "Excel 表格，每个 Sheet 作为一个节点",
                },
                "markdown": {
                    "extensions": [".md", ".markdown"],
                    "description": "Markdown 文档，按标题（# 层级）切分",
                },
            }
        }
    )


@router.get(
    "/doc-json/{doc_id}",
    response_model=ApiResponse[Dict[str, Any]],
    summary="获取文档处理后的 JSON 结构",
)
def get_document_json(
    doc_id: str,
    chunk_id: Optional[str] = Query(default=None, description="按 chunk_id 精确返回对应 JSON 节点"),
    node_id: Optional[str] = Query(default=None, description="按 node_id 精确返回对应 JSON 节点"),
    chapter_id: Optional[str] = Query(default=None, description="按 chapter_id/node_id 返回对应章节 JSON 节点"),
    include_children: bool = Query(default=True, description="返回节点时是否包含其子节点 nodes"),
    db: Session = Depends(get_db),
):
    """
    返回文档经过解析处理后生成的 JSON 树状结构。

    该 JSON 包含文档的层级目录、各章节内容、摘要等信息。
    """
    document = DocumentRepository.get_by_id(db, doc_id)
    if not document:
        return ApiResponse.error(code=404, message="文档不存在")

    if not document.pageindex_path:
        return ApiResponse.error(code=404, message="文档尚未完成处理，无可用 JSON")

    try:
        minio_service = MinioService()
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        minio_service.download_file(BUCKET_NAME, document.pageindex_path, tmp_path)

        with open(tmp_path, "r", encoding="utf-8") as f:
            doc_json = json.load(f)

        os.unlink(tmp_path)

        target_id = chunk_id or node_id or chapter_id
        if target_id:
            id_fields = ["node_id", "chunk_id", "title"] if chunk_id else ["node_id", "title"]
            matches = _find_json_nodes_by_id(doc_json.get("structure", []), target_id, id_fields)
            if not matches:
                return ApiResponse.error(code=404, message=f"未找到对应 JSON 节点: {target_id}")

            if not include_children:
                for item in matches:
                    item["node"] = _strip_children(item["node"])

            return ApiResponse.success(data={
                "doc_id": doc_id,
                "title": document.title,
                "query": {
                    "chunk_id": chunk_id,
                    "node_id": node_id,
                    "chapter_id": chapter_id,
                    "include_children": include_children,
                },
                "match_count": len(matches),
                "matches": matches,
            })

        return ApiResponse.success(data={
            "doc_id": doc_id,
            "title": document.title,
            "json": doc_json,
        })
    except Exception as e:
        return ApiResponse.error(code=500, message=f"获取文档 JSON 失败: {str(e)}")


# ---------------------------------------------------------------------------
# 内部辅助：单个文档的「转 PDF → 解析 → 写入 ES」完整流程
# ---------------------------------------------------------------------------
async def _process_single_doc(db: Session, doc_id: str, progress_callback=None) -> Dict[str, Any]:
    """
    处理单个文档的完整流程：
    1. 从 MinIO 下载源文件
    2. 如果不是 PDF，用 LibreOffice 转换为 PDF
    3. 将 PDF 上传到 MinIO，更新 DB 的 pdf_path
    4. 运行 pdf/process 的解析逻辑（LLM 解析 + ES 写入 + 文档路由）

    progress_callback: 可选的异步回调函数 async fn(percent, message)
    """
    async def _notify(percent: float, msg: str):
        """更新 DB 进度 + 调用回调"""
        DocumentRepository.update(db, doc_id, progress=percent, progress_msg=msg)
        if progress_callback:
            await progress_callback(percent, msg)

    document = DocumentRepository.get_by_id(db, doc_id)
    if not document:
        return {"doc_id": doc_id, "success": False, "error": "文档不存在"}

    if not document.source_path:
        return {"doc_id": doc_id, "success": False, "error": "文档 source_path 为空"}

    process_begin_at = datetime.now()
    DocumentRepository.update(
        db, doc_id,
        progress=0.0,
        progress_msg="开始处理文档",
        process_begin_at=process_begin_at,
        process_duration=0.0,
    )
    if progress_callback:
        await progress_callback(0.0, "开始处理文档")

    temp_src_path = None
    temp_pdf_path = None
    output = None

    try:
        # --- 1. 从 MinIO 下载源文件 ---
        minio_service = MinioService()
        suffix = os.path.splitext(document.source_path)[1] or os.path.splitext(document.title)[1] or ""
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            temp_src_path = tmp.name

        minio_service.download_file(BUCKET_NAME, document.source_path, temp_src_path)
        await _notify(10.0, "文件下载完成")

        # --- 2. 转换为 PDF（如果不是 PDF）---
        ext = suffix.lower()
        if ext == ".pdf":
            temp_pdf_path = temp_src_path
            await _notify(20.0, "已是 PDF，跳过转换")
        elif ext in SUPPORTED_EXTENSIONS:
            await _notify(15.0, f"正在将 {ext} 转换为 PDF")
            source_file_url = f"http://{MINIO_IP}:{MINIO_PORT}/{BUCKET_NAME}{document.source_path}"
            temp_pdf_path = convert_to_pdf(temp_src_path, source_url=source_file_url)
            # 上传 PDF 到 MinIO，路径与 source_path 对应（仅替换扩展名）
            source_no_ext = os.path.splitext(document.source_path)[0]
            pdf_object_name = source_no_ext + ".pdf"
            minio_service.upload_file(BUCKET_NAME, pdf_object_name, temp_pdf_path, content_type="application/pdf")
            DocumentRepository.update(db, doc_id, pdf_path=pdf_object_name)
            await _notify(20.0, "PDF 转换并上传完成")
        else:
            return {"doc_id": doc_id, "success": False, "error": f"不支持的文件格式: {ext}"}

        # --- 3. 根据 token 数选择解析方式（与 /pdf/process 逻辑一致）---
        import fitz
        pdf_doc = fitz.open(temp_pdf_path)
        _page_texts = []
        for _p in pdf_doc:
            try:
                _page_texts.append(_p.get_text())
            except Exception:
                _page_texts.append("")
        full_text = "".join(_page_texts)
        pdf_doc.close()
        estimated_tokens = int(len(full_text) / 1.5)
        await _notify(25.0, f"文档估算 {estimated_tokens} tokens，使用 {'simple' if estimated_tokens <= 8000 else 'chunked'} 模式")

        if estimated_tokens <= 8000:
            from run_pdf_to_md import process_pdf_simple
            output = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: asyncio.run(process_pdf_simple(
                    pdf_path=temp_pdf_path,
                    output_dir=OUTPUT_DIR,
                    llm_url=LLM_URL, llm_model=LLM_MODEL, model=LLM_MODEL,
                    if_summary=True, if_add_node_text=True,
                ))
            )
            mode_used = "simple"
        else:
            from run_pdf_to_md_chunked import process_pdf_chunked
            output = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: asyncio.run(process_pdf_chunked(
                    pdf_path=temp_pdf_path,
                    output_dir=OUTPUT_DIR,
                    llm_url=LLM_URL, llm_model=LLM_MODEL, model=LLM_MODEL,
                    if_summary=True, if_add_node_text=True,
                ))
            )
            mode_used = "chunked"

        json_path = output["json_path"]

        # 上传 JSON 到 MinIO，路径与 source_path 对应（仅替换扩展名）
        source_no_ext = os.path.splitext(document.source_path)[0]
        json_object_name = source_no_ext + ".json"
        minio_service.upload_file(BUCKET_NAME, json_object_name, json_path)

        DocumentRepository.update(db, doc_id, pageindex_path=json_object_name)
        await _notify(50.0, "文件解析完成，JSON 已上传")

        # --- 4. 三级索引写入 (doc / chapter / chunk) ---
        d_id = str(document.doc_id)
        k_id = str(document.kb_id)
        f_id = str(document.fd_id)

        chunk_num = await asyncio.to_thread(
            write_to_three_indices,
            json_path=json_path,
            doc_id=d_id, kb_id=k_id, fd_id=f_id,
            doc_title=document.title,
        )

        DocumentRepository.update(db, doc_id, chunk_num=chunk_num)
        await _notify(90.0, f"三级索引写入完成，共 {chunk_num} 个 chunk")

        # --- 6. 完成 ---
        process_duration = (datetime.now() - process_begin_at).total_seconds()
        DocumentRepository.update(db, doc_id, process_duration=process_duration, status="ready")
        await _notify(100.0, f"处理完成，耗时 {process_duration:.2f} 秒")

        return {
            "doc_id": doc_id,
            "success": True,
            "mode": mode_used,
            "estimated_tokens": estimated_tokens,
            "chunk_num": chunk_num,
            "process_duration": process_duration,
        }

    except Exception as e:
        DocumentRepository.update(
            db, doc_id,
            progress_msg=f"处理失败: {str(e)}",
            status="error",
        )
        return {"doc_id": doc_id, "success": False, "error": str(e)}

    finally:
        # 清理临时文件（源文件、PDF、以及中间产物 MD/JSON）
        cleanup_paths = [temp_src_path, temp_pdf_path]
        try:
            if output and output.get("json_path"):
                cleanup_paths.append(output["json_path"])
            if output and output.get("md_path"):
                cleanup_paths.append(output["md_path"])
        except Exception:
            pass
        for p in cleanup_paths:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# POST /universal/process_bat/{doc_id}  — SSE 流式返回进度
# ---------------------------------------------------------------------------
@router.post(
    "/process_bat/{doc_id}",
    summary="处理文档（非PDF先转PDF，再解析+写入ES），SSE 流式返回进度",
)
async def process_bat_single(
    doc_id: str,
    db: Session = Depends(get_db),
):
    """
    通用文档处理接口，**实时返回处理进度（SSE）**。

    支持格式：**Word (.doc/.docx)、TXT、JSON、PPT (.ppt/.pptx)、Excel (.xls/.xlsx)、PDF**

    返回格式（text/event-stream）：
    ```
    data: {"progress": 10.0, "message": "文件下载完成"}
    data: {"progress": 25.0, "message": "文档估算 828 tokens，使用 simple 模式"}
    ...
    data: {"progress": 100.0, "message": "处理完成，耗时 18.01 秒", "result": {...}}
    ```
    """
    async def event_stream() -> AsyncGenerator[str, None]:
        progress_queue: asyncio.Queue = asyncio.Queue()
        last_sent = [0]  # 上次发送的进度值，用于保证最小步进 2

        async def on_progress(percent: float, msg: str):
            p = int(percent)
            # 最小步进 2（100% 始终发送）
            if p < 100 and p - last_sent[0] < 2:
                return
            last_sent[0] = p
            await progress_queue.put(p)

        async def run_queued_process():
            await progress_queue.put(0)
            async with PROCESS_BAT_SEMAPHORE:
                return await _process_single_doc(db, doc_id, progress_callback=on_progress)

        # 启动处理任务
        task = asyncio.create_task(run_queued_process())

        # 持续发送进度事件，直到任务完成
        while not task.done():
            try:
                p = await asyncio.wait_for(progress_queue.get(), timeout=1.0)
                yield f"data: {json.dumps({'progress': p})}\n\n"
            except asyncio.TimeoutError:
                yield f": heartbeat\n\n"

        # 排空队列中剩余的进度事件
        while not progress_queue.empty():
            p = await progress_queue.get()
            yield f"data: {json.dumps({'progress': p})}\n\n"

        # 如果处理失败，发送错误事件
        result = task.result()
        if not result.get("success"):
            yield f"data: {json.dumps({'progress': -1})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
