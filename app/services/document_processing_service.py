from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
import os
import tempfile
import time
from datetime import datetime
from ..repository.document_repository import DocumentRepository
from ..repository.file_repository import FileRepository
from ..services.minio_service import MinioService
from ..services.doc_index_service import DocIndexService
from ..services.chapter_index_service import ChapterIndexService
from ..services.chunk_index_service import ChunkIndexService
from ..core.exceptions import NotFoundException, InternalServerErrorException, BadRequestException
from rag.page_index import page_index
from rag.json_to_es_converter_with_embedding import ESConverter
from rag.build_router_es import DocumentRouter

# 从配置文件导入Elasticsearch索引名称
from common.config import ELASTICSEARCH_INDEX
# 添加项目根目录到路径，以便导入 rag 模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 非 PDF 格式由通用服务处理
_NON_PDF_EXTS = {".docx", ".doc", ".txt", ".pptx", ".ppt", ".xlsx", ".xls", ".md", ".markdown"}


def add_progress_message(progress_list, last_step_time, step, result):
    """添加进度消息到指定列表"""
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
    
    # 计算步骤耗时
    duration = ""
    if last_step_time:
        step_duration = (current_time - last_step_time).total_seconds()
        duration = f" (耗时: {step_duration:.2f} 秒)"
    
    message = f"[{timestamp}] {step}: {result}{duration}"
    progress_list.append(message)
    print(message)  # 同时打印到控制台
    
    # 返回当前时间作为新的last_step_time
    return current_time


class DocumentProcessingService:
    @staticmethod
    def delete_es_records_by_doc_id(doc_id: str) -> Dict[str, Any]:
        """按 doc_id 删除 ES 中 doc/chapter/chunk 三类索引记录"""
        try:
            doc_deleted = DocIndexService().delete_by_doc_id(doc_id)
            chapter_deleted = ChapterIndexService().delete_by_doc_id(doc_id)
            chunk_deleted = ChunkIndexService().delete_by_doc_id(doc_id)
        except Exception as e:
            raise InternalServerErrorException(detail=f"删除 Elasticsearch 记录失败: {str(e)}")

        if not (doc_deleted and chapter_deleted and chunk_deleted):
            raise InternalServerErrorException(detail="部分索引删除失败，请检查服务日志")

        return {
            "success": True,
            "doc_id": doc_id,
            "deleted_indices": ["doc_index", "chapter_index", "chunk_index"],
        }

    @staticmethod
    def update_kb_id_by_doc_id(db: Session, doc_id: str, new_kb_id: str) -> Dict[str, Any]:
        """按 doc_id 将文件的 kb_id 更新为新的 kb_id（数据库 + ES）"""
        if not new_kb_id:
            raise BadRequestException(detail="new_kb_id 不能为空")

        document = DocumentRepository.get_by_id(db, doc_id)
        if not document:
            raise NotFoundException(detail=f"ID为 {doc_id} 的文档不存在")

        doc_updated = DocIndexService().update_kb_id_by_doc_id(doc_id, new_kb_id)
        chapter_updated = ChapterIndexService().update_kb_id_by_doc_id(doc_id, new_kb_id)
        chunk_updated = ChunkIndexService().update_kb_id_by_doc_id(doc_id, new_kb_id)

        if doc_updated is None or chapter_updated is None or chunk_updated is None:
            raise InternalServerErrorException(detail="更新 ES 的 kb_id 失败，请检查服务日志")

        DocumentRepository.update(db, doc_id, kb_id=new_kb_id)

        return {
            "success": True,
            "doc_id": doc_id,
            "new_kb_id": new_kb_id,
            "updated_counts": {
                "doc_index": doc_updated,
                "chapter_index": chapter_updated,
                "chunk_index": chunk_updated,
            }
        }

    @staticmethod
    def update_kb_id_by_fd_ids(db: Session, fd_ids: list, new_kb_id: str) -> Dict[str, Any]:
        """按多个 fd_id 批量将文件的 kb_id 更新为新的 kb_id（数据库 + ES）"""
        if not new_kb_id:
            raise BadRequestException(detail="new_kb_id 不能为空")
        if not fd_ids:
            raise BadRequestException(detail="fd_ids 不能为空")

        fd_ids = [str(fd_id) for fd_id in fd_ids if fd_id not in (None, "")]
        if not fd_ids:
            raise BadRequestException(detail="fd_ids 不能为空")

        documents = DocumentRepository.get_by_folders(db, fd_ids)
        if not documents:
            raise NotFoundException(detail=f"未找到 fd_id 属于 {fd_ids} 的文档")

        doc_ids = [str(document.doc_id) for document in documents]
        doc_updated = DocIndexService().update_kb_id_by_fd_ids(fd_ids, new_kb_id)
        chapter_updated = ChapterIndexService().update_kb_id_by_fd_ids(fd_ids, new_kb_id)
        chunk_updated = ChunkIndexService().update_kb_id_by_fd_ids(fd_ids, new_kb_id)

        if doc_updated is None or chapter_updated is None or chunk_updated is None:
            raise InternalServerErrorException(detail="更新 ES 的 kb_id 失败，请检查服务日志")

        db_updated = DocumentRepository.update_kb_id_by_folders(db, fd_ids, new_kb_id)

        return {
            "success": True,
            "fd_ids": fd_ids,
            "new_kb_id": new_kb_id,
            "updated_doc_count": db_updated,
            "updated_doc_ids": doc_ids,
            "updated_counts": {
                "database": db_updated,
                "doc_index": doc_updated,
                "chapter_index": chapter_updated,
                "chunk_index": chunk_updated,
            },
        }

    @staticmethod
    async def process_document(
        db: Session,
        doc_id: str
    ) -> Dict[str, Any]:
        """
        完整的文档解析流程
        1. 查询文档记录，获取source_path
        2. 从MinIO获取文件
        3. 解析文件生成PageIndex JSON
        4. 转换为向量数据库所需的扁平化格式并存储到Elasticsearch
        5. 生成文档路由并存入Elasticsearch
        6. 更新文档信息
        """
        # 1. 查询文档记录
        document = DocumentRepository.get_by_id(db, doc_id)
        if not document:
            raise NotFoundException(detail=f"ID为 {doc_id} 的文档不存在")
        
        # 2. 获取文件路径
        if not document.source_path:
            raise NotFoundException(detail="文档文件路径不存在")
        
        # 判断文件类型，非 PDF 格式委托给 UniversalRagService 处理
        file_ext = os.path.splitext(document.title)[1].lower()
        if file_ext in _NON_PDF_EXTS:
            from ..services.universal_rag_service import UniversalRagService
            return await UniversalRagService.process_document(db, doc_id)
    
        
        # 3. 记录开始时间
        process_begin_at = datetime.now()
        progress = 0.0
        llm_token = 0
        
        # 为每个请求创建独立的进度列表和时间戳
        processing_progress = []
        last_step_time = None
        
        # 添加进度消息
        last_step_time = add_progress_message(processing_progress, last_step_time, "初始化", "开始处理文档")
        
        # 更新文档处理状态
        # 将处理进度消息列表转换为字符串存储
        progress_messages = "\n".join(processing_progress)
        DocumentRepository.update(
            db,
            document.doc_id,
            progress=progress,
            progress_msg=progress_messages,
            process_begin_at=process_begin_at,
            process_duration=0.0
        )
        
        try:
            # 4. 从MinIO获取文件
            minio_service = MinioService()
            # 使用固定的桶名"document"来获取文档文件
            bucket_name = "deepsearch"
            source_path = document.source_path
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(document.title)[1], delete=False) as temp_file:
                temp_file_path = temp_file.name
            
            # 打印临时文件路径
            print(f"[DEBUG] 临时文件路径: {temp_file_path}")
            
            # 下载文件到临时路径
            minio_service.download_file(bucket_name, source_path, temp_file_path)
            progress = 20.0
            progress_msg = "文件下载完成"
            
            # 添加进度消息
            last_step_time = add_progress_message(processing_progress, last_step_time, "文件下载", "成功")
            
            # 将处理进度消息列表转换为字符串存储
            progress_messages = "\n".join(processing_progress)
            DocumentRepository.update(
                db,
                document.doc_id,
                progress=progress,
                progress_msg=progress_messages
            )
            
            # 5. 解析文件生成PageIndex JSON
            try:
                # 调用page_index函数解析文件
                # 注意：这里需要根据实际的page_index函数参数进行调整
                page_index_output = await page_index(
                    temp_file_path,
                    model="Qwen3-8B",  # 显式指定model参数
                    toc_check_page_num=10,  
                    max_page_num_each_node=10,
                    max_token_num_each_node=10000,
                    if_add_node_id="yes",
                    if_add_node_summary="no",
                    if_add_node_text="yes"
                )
                
                # 获取 LLM token 用量
                from rag.utils import get_token_count
                token_count = get_token_count()
                llm_token = token_count['input'] + token_count['output']
                
                # 打印page_index_output路径
                print(f"[DEBUG] page_index_output路径: {page_index_output}")
                
                # 将PageIndex JSON文件上传到MinIO
                json_filename = f"{document.doc_id}.json"
                json_object_name = f"{document.doc_id}/{json_filename}"
                
                # 上传文件到MinIO
                minio_service.upload_file(bucket_name, json_object_name, page_index_output)
                
                # 更新document的pageindex_path字段
                pageindex_path = json_object_name
                
                progress = 50.0
                progress_msg = "文件解析完成"
                
                # 添加进度消息
                last_step_time = add_progress_message(processing_progress, last_step_time, "文件解析", "成功")
                
                # 将处理进度消息列表转换为字符串存储
                progress_messages = "\n".join(processing_progress)
                DocumentRepository.update(
                    db,
                    document.doc_id,
                    progress=progress,
                    pageindex_path=pageindex_path,
                    progress_msg=progress_messages
                )
                
            except Exception as e:
                raise InternalServerErrorException(detail=f"解析文档失败: {str(e)}")
            
            # 6. 转换为向量数据库所需的扁平化格式并存储到Elasticsearch
            try:
                # 提取doc_id、kb_id、fd_id
                doc_id = str(document.doc_id)
                kb_id = str(document.kb_id)
                fd_id = str(document.fd_id)
                
                # 创建ESConverter实例并运行
                converter = ESConverter(
                    page_index_output,
                    index_name=ELASTICSEARCH_INDEX,
                    doc_id=doc_id,
                    kb_id=kb_id,
                    fd_id=fd_id,
                    doc_title=document.title
                )
                converter.run()
                
                # 计算chunk_num
                chunk_num = len(converter.flat_nodes)
                
                progress = 75.0
               
                
                # 添加进度消息
                last_step_time = add_progress_message(processing_progress, last_step_time, "向量转换", "成功")
                
                # 将处理进度消息列表转换为字符串存储
                progress_messages = "\n".join(processing_progress)
                DocumentRepository.update(
                    db,
                    document.doc_id,
                    progress=progress,
                    progress_msg=progress_messages,
                    chunk_num=chunk_num
                )
                
            except Exception as e:
                raise InternalServerErrorException(detail=f"转换为向量格式失败: {str(e)}")
            
            # 7. 生成文档路由并存入Elasticsearch
            print(f"document.title: {document.title}")
            try:
                # 创建DocumentRouter实例并运行，传入已经生成的扁平格式数据
                router = DocumentRouter(
                    page_index_output,
                    doc_id=doc_id,
                    kb_id=kb_id,
                    fd_id=fd_id,
                    doc_title=document.title,
                    flat_nodes=converter.flat_nodes
                )
                router.run()
                
                progress = 90.0
          
                
                # 添加进度消息
                last_step_time = add_progress_message(processing_progress, last_step_time, "文档路由生成", "成功")
                
                # 将处理进度消息列表转换为字符串存储
                progress_messages = "\n".join(processing_progress)
                DocumentRepository.update(
                    db,
                    document.doc_id,
                    progress=progress,
                    progress_msg=progress_messages
                )
                
            except Exception as e:
                raise InternalServerErrorException(detail=f"生成文档路由失败: {str(e)}")
            
            # 8. 计算处理时长
            process_duration = (datetime.now() - process_begin_at).total_seconds()
            
            # 9. 更新文档信息
            # 将处理进度消息列表转换为字符串存储
            progress_messages = "\n".join(processing_progress)
            DocumentRepository.update(
                db,
                document.doc_id,
                progress=100.0,
                progress_msg=progress_messages,
                process_duration=process_duration,
                llm_token=llm_token,
                status="ready"
            )
            
            # 添加进度消息
            last_step_time = add_progress_message(processing_progress, last_step_time, "处理完成", f"成功，耗时 {process_duration:.2f} 秒")
            
            # 10. 清理临时文件
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if os.path.exists(page_index_output):
                os.remove(page_index_output)
            
            return {
                "success": True,
                "message": "Document processed successfully",
                "document_id": document.doc_id
            }
            
        except Exception as e:
            # 添加进度消息
            last_step_time = add_progress_message(processing_progress, last_step_time, "处理失败", str(e))
            
            # 更新文档状态为错误
            # 将处理进度消息列表转换为字符串存储
            progress_messages = "\n".join(processing_progress)
            DocumentRepository.update(
                db,
                document.doc_id,
                progress=progress,
                progress_msg=progress_messages,
                status="error"
            )
            
            # 清理临时文件
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if 'page_index_output' in locals() and os.path.exists(page_index_output):
                os.remove(page_index_output)
            
            raise InternalServerErrorException(detail=f"处理文档失败: {str(e)}")