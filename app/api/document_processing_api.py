from fastapi import APIRouter, Depends, Body
from sqlalchemy.orm import Session
from typing import Dict, Any, List

from ..core.database import SessionLocal
from ..services.document_processing_service import DocumentProcessingService
from ..utils.response import ApiResponse
from ..core.exceptions import NotFoundException, InternalServerErrorException, BadRequestException

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/process/{doc_id}", response_model=ApiResponse[Dict[str, Any]], summary="处理文档")
async def process_document(doc_id: str, db: Session = Depends(get_db)):
    """
    处理文档的API接口
    
    - **doc_id**: 文档ID
    
    该接口会执行完整的文档处理流程：
    1. 查询文档记录，获取source_path
    2. 从MinIO获取文件
    3. 解析文件生成PageIndex JSON
    4. 转换为向量数据库所需的扁平化格式并存储到Elasticsearch
    5. 生成文档路由并存入Elasticsearch
    6. 更新文档信息
    """
    try:
        result = await DocumentProcessingService.process_document(db, doc_id)
        return ApiResponse.success(data=result)
    except NotFoundException as e:
        return ApiResponse.error(code=404, message=e.detail)
    except InternalServerErrorException as e:
        return ApiResponse.error(code=500, message=e.detail)
    except Exception as e:
        return ApiResponse.error(code=500, message=f"服务器内部错误: {str(e)}")


@router.put("/es/kb/by-folders", response_model=ApiResponse[Dict[str, Any]], summary="按文件夹批量更新文档在ES中的kb_id")
def update_documents_kb_id_by_fd_ids(
    fd_ids: List[str] = Body(..., embed=True),
    new_kb_id: str = Body(..., embed=True),
    db: Session = Depends(get_db)
):
    try:
        result = DocumentProcessingService.update_kb_id_by_fd_ids(db, fd_ids, new_kb_id)
        return ApiResponse.success(data=result)
    except BadRequestException as e:
        return ApiResponse.error(code=400, message=e.detail)
    except NotFoundException as e:
        return ApiResponse.error(code=404, message=e.detail)
    except InternalServerErrorException as e:
        return ApiResponse.error(code=500, message=e.detail)
    except Exception as e:
        return ApiResponse.error(code=500, message=f"服务器内部错误: {str(e)}")


@router.put("/es/kb/{doc_id}", response_model=ApiResponse[Dict[str, Any]], summary="更新文档在ES中的kb_id")
def update_document_kb_id(
    doc_id: str,
    new_kb_id: str = Body(..., embed=True),
    db: Session = Depends(get_db)
):
    """
    将指定文档在数据库和 Elasticsearch 中的 kb_id 更新为新的 kb_id。

    - **doc_id**: 文档ID
    - **new_kb_id**: 新的知识库ID
    """
    try:
        result = DocumentProcessingService.update_kb_id_by_doc_id(db, doc_id, new_kb_id)
        return ApiResponse.success(data=result)
    except BadRequestException as e:
        return ApiResponse.error(code=400, message=e.detail)
    except NotFoundException as e:
        return ApiResponse.error(code=404, message=e.detail)
    except InternalServerErrorException as e:
        return ApiResponse.error(code=500, message=e.detail)
    except Exception as e:
        return ApiResponse.error(code=500, message=f"服务器内部错误: {str(e)}")


@router.delete("/es/{doc_id}", response_model=ApiResponse[Dict[str, Any]], summary="删除文档在ES中的记录")
def delete_document_es_records(doc_id: str):
    """
    删除指定文档在 Elasticsearch 中的索引记录。

    - **doc_id**: 文档ID

    会按 doc_id 删除以下索引中的记录：
    1. doc_index
    2. chapter_index
    3. chunk_index
    """
    try:
        result = DocumentProcessingService.delete_es_records_by_doc_id(doc_id)
        return ApiResponse.success(data=result)
    except InternalServerErrorException as e:
        return ApiResponse.error(code=500, message=e.detail)
    except Exception as e:
        return ApiResponse.error(code=500, message=f"服务器内部错误: {str(e)}")


@router.get("/status/{doc_id}", response_model=ApiResponse[Dict[str, Any]], summary="获取文档处理状态")
def get_document_status(doc_id: str, db: Session = Depends(get_db)):
    """
    获取文档处理状态的API接口
    
    - **doc_id**: 文档ID
    
    返回文档的处理状态、进度、处理时长等信息
    """
    from ..repository.document_repository import DocumentRepository
    
    try:
        document = DocumentRepository.get_by_id(db, doc_id)
        if not document:
            return ApiResponse.error(code=404, message=f"文档ID {doc_id} 不存在")
        
        status_info = {
            "document_id": document.doc_id,
            "status": document.status,
            "progress": document.progress,
            "progress_msg": document.progress_msg,
            "process_begin_at": document.process_begin_at,
            "process_duration": document.process_duration,
            "chunk_num": document.chunk_num,
            "llm_token": document.llm_token
        }
        
        return ApiResponse.success(data=status_info)
    except Exception as e:
        return ApiResponse.error(code=500, message=f"服务器内部错误: {str(e)}")