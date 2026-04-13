"""
通用多格式文档 RAG API
提供文档处理端点，支持 PDF / Word / TXT / PPT / Excel / Markdown 等格式。
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Dict, Any

from ..core.database import SessionLocal
from ..services.universal_rag_service import UniversalRagService
from ..utils.response import ApiResponse
from ..core.exceptions import NotFoundException, InternalServerErrorException

router = APIRouter()


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
