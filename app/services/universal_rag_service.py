"""
通用 RAG 服务（Universal RAG Service）
支持 PDF / Word / TXT / PPT / Excel / Markdown 等多格式文件的解析与 ES 索引。

处理流程：
  1. 根据文件扩展名分发到对应的解析器
  2. 对于 PDF：先检测是否有目录（TOC）
       有目录 → 走现有 page_index 流程（DocumentProcessingService.process_document）
       无目录 → 使用 PdfSimpleParser 分块后索引
  3. 其他格式 → 对应格式解析器解析后索引
  4. 将解析结果保存为临时 JSON 文件
  5. 调用 ESConverter 和 DocumentRouter 写入 Elasticsearch
"""
import os
import json
import tempfile
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from sqlalchemy.orm import Session

from ..repository.document_repository import DocumentRepository
from ..repository.file_repository import FileRepository
from ..services.minio_service import MinioService
from ..core.exceptions import NotFoundException, InternalServerErrorException
from .parsers import (
    WordParser,
    TxtParser,
    PptParser,
    ExcelParser,
    MarkdownParser,
    PdfSimpleParser,
)
from rag.utils import get_page_tokens, ConfigLoader
from rag.page_index import check_toc
from rag.page_index import page_index
from rag.utils import get_token_count, reset_token_count
from rag.json_to_es_converter_with_embedding import ESConverter
from common.config import ELASTICSEARCH_INDEX

logger = logging.getLogger(__name__)

# 文件扩展名 → 解析器映射（PDF 单独处理）
_EXT_PARSER_MAP = {
    ".docx": WordParser,
    ".doc": WordParser,
    ".txt": TxtParser,
    ".pptx": PptParser,
    ".ppt": PptParser,
    ".xlsx": ExcelParser,
    ".xls": ExcelParser,
    ".md": MarkdownParser,
    ".markdown": MarkdownParser,
}

# PDF 文件扩展名
_PDF_EXTS = {".pdf"}


def _add_progress(progress_list, last_step_time, step, result):
    """记录进度消息，返回新的时间戳"""
    current_time = datetime.now()
    ts = current_time.strftime("%Y-%m-%d %H:%M:%S")
    duration = ""
    if last_step_time:
        secs = (current_time - last_step_time).total_seconds()
        duration = f" (耗时: {secs:.2f} 秒)"
    msg = f"[{ts}] {step}: {result}{duration}"
    progress_list.append(msg)
    print(msg)
    return current_time


class UniversalRagService:
    """
    通用多格式文档 RAG 服务。
    提供 process_document 方法，与 DocumentProcessingService 接口兼容。
    """

    @staticmethod
    async def process_document(
        db: Session,
        doc_id: str,
        *,
        use_llm_for_pdf_no_toc: bool = False,
    ) -> Dict[str, Any]:
        """
        处理文档并写入 Elasticsearch。

        参数
        ----
        db                      : 数据库 Session
        doc_id                  : 文档 ID
        use_llm_for_pdf_no_toc  : 对无目录 PDF 是否启用 LLM 生成目录
                                  （True → 调用 page_index.process_no_toc，消耗 LLM token；
                                   False → 直接按字符分块，更快、零 token）
        """
        # ── 1. 查询文档记录 ──────────────────────────────────────────────
        document = DocumentRepository.get_by_id(db, doc_id)
        if not document:
            raise NotFoundException(detail=f"ID 为 {doc_id} 的文档不存在")
        if not document.source_path:
            raise NotFoundException(detail="文档文件路径不存在")

        process_begin_at = datetime.now()
        progress = 0.0
        llm_token = 0
        processing_progress = []
        last_step_time = None

        last_step_time = _add_progress(
            processing_progress, last_step_time, "初始化", "开始处理文档"
        )
        DocumentRepository.update(
            db,
            document.doc_id,
            progress=progress,
            progress_msg="\n".join(processing_progress),
            process_begin_at=process_begin_at,
            process_duration=0.0,
        )

        temp_file_path = None
        page_index_output = None

        try:
            # ── 2. 从 MinIO 下载文件 ────────────────────────────────────
            minio_service = MinioService()
            bucket_name = "deepsearch"
            source_path = document.source_path

            suffix = os.path.splitext(document.title)[1] or ".tmp"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                temp_file_path = tmp.name

            minio_service.download_file(bucket_name, source_path, temp_file_path)
            progress = 20.0
            last_step_time = _add_progress(
                processing_progress, last_step_time, "文件下载", "成功"
            )
            DocumentRepository.update(
                db,
                document.doc_id,
                progress=progress,
                progress_msg="\n".join(processing_progress),
            )

            # ── 3. 解析文件 → page_index JSON ──────────────────────────
            ext = os.path.splitext(document.title)[1].lower()

            llm_token = 0
            if ext in _PDF_EXTS:
                page_index_output, llm_token = await UniversalRagService._process_pdf(
                    temp_file_path,
                    document,
                    use_llm_for_pdf_no_toc=use_llm_for_pdf_no_toc,
                    db=db,
                )
            elif ext in _EXT_PARSER_MAP:
                page_index_output = UniversalRagService._process_other(
                    temp_file_path, ext, document
                )
            else:
                raise InternalServerErrorException(
                    detail=f"不支持的文件格式: {ext}"
                )

            progress = 50.0
            last_step_time = _add_progress(
                processing_progress, last_step_time, "文件解析", "成功"
            )

            # 上传 JSON 到 MinIO
            # 基于 source_path 生成 JSON 文件路径，将扩展名改为 .json
            base_name = os.path.splitext(source_path)[0]
            json_object_name = f"{base_name}.json"
            minio_service.upload_file(bucket_name, json_object_name, page_index_output)
            DocumentRepository.update(
                db,
                document.doc_id,
                progress=progress,
                pageindex_path=json_object_name,
                progress_msg="\n".join(processing_progress),
            )

            # ── 4. ES 向量索引 ──────────────────────────────────────────
           

            converter = ESConverter(
                page_index_output,
                index_name=ELASTICSEARCH_INDEX,
                doc_id=str(document.doc_id),
                kb_id=str(document.kb_id),
                fd_id=str(document.fd_id),
                doc_title=document.title,
            )
            converter.run()
            chunk_num = len(converter.flat_nodes)

            progress = 75.0
            last_step_time = _add_progress(
                processing_progress, last_step_time, "向量转换", "成功"
            )
            DocumentRepository.update(
                db,
                document.doc_id,
                progress=progress,
                progress_msg="\n".join(processing_progress),
                chunk_num=chunk_num,
            )

            # ── 5. 文档路由 ─────────────────────────────────────────────
            from rag.build_router_es import DocumentRouter

            router = DocumentRouter(
                page_index_output,
                doc_id=str(document.doc_id),
                kb_id=str(document.kb_id),
                fd_id=str(document.fd_id),
                doc_title=document.title,
                flat_nodes=converter.flat_nodes,
            )
            router.run()

            progress = 90.0
            last_step_time = _add_progress(
                processing_progress, last_step_time, "文档路由生成", "成功"
            )
            DocumentRepository.update(
                db,
                document.doc_id,
                progress=progress,
                progress_msg="\n".join(processing_progress),
            )

            # ── 6. 完成 ─────────────────────────────────────────────────
            process_duration = (datetime.now() - process_begin_at).total_seconds()
            _add_progress(
                processing_progress,
                last_step_time,
                "处理完成",
                f"成功，耗时 {process_duration:.2f} 秒",
            )
            DocumentRepository.update(
                db,
                document.doc_id,
                progress=100.0,
                progress_msg="\n".join(processing_progress),
                process_duration=process_duration,
                llm_token=llm_token,
                status="ready",
            )

            return {
                "success": True,
                "message": "Document processed successfully",
                "document_id": document.doc_id,
            }

        except Exception as e:
            _add_progress(processing_progress, last_step_time, "处理失败", str(e))
            DocumentRepository.update(
                db,
                document.doc_id,
                progress=progress,
                progress_msg="\n".join(processing_progress),
                status="error",
            )
            raise InternalServerErrorException(detail=f"处理文档失败: {str(e)}")

        finally:
            # 清理临时文件
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if page_index_output and os.path.exists(page_index_output):
                os.remove(page_index_output)

    # ══════════════════════════════════════════════════════════════════
    # 私有辅助方法
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    async def _process_pdf(
        file_path: str,
        document,
        *,
        use_llm_for_pdf_no_toc: bool,
        db: Session,
    ):
        """
        PDF 处理逻辑：
          - use_llm_for_pdf_no_toc=True（默认值）：
              直接调用 page_index()，其内部自动检测 TOC：
                有目录 → LLM 辅助精准切分
                无目录 → LLM 生成目录后切分
          - use_llm_for_pdf_no_toc=False：
              先用 rag/utils.get_page_tokens 提取页面文本，
              调用 check_toc 检测是否有目录：
                有目录 → 仍走 page_index() 全流程
                无目录 → PdfSimpleParser 按字符分块（快速、零 token）

        返回 (json_file_path, llm_token_count)
        """
        

        llm_token = 0

        if use_llm_for_pdf_no_toc:
            # ── LLM 全流程（page_index 内部自动处理有无目录）──
            reset_token_count()
            json_path = await page_index(
                file_path,
                model="Qwen3-8B",
                toc_check_page_num=10,
                max_page_num_each_node=10,
                max_token_num_each_node=10000,
                if_add_node_id="yes",
                if_add_node_summary="no",
                if_add_node_text="yes",
            )
            token_count = get_token_count()
            llm_token = token_count.get("input", 0) + token_count.get("output", 0)
            return json_path, llm_token

        # ── 快速模式：先检测 TOC，再决定是否走 LLM ──
        try:
            
            print("开始检测 TOC")
            # 构建临时 opt 供 check_toc 使用
            opt = ConfigLoader().load({"model": "Qwen3-8B", "toc_check_page_num": 10})
            page_list = get_page_tokens(file_path, model=opt.model)
            print("页面文本提取完成")
            toc_result = check_toc(page_list, opt)
            print("TOC 检测完成")
            has_toc = bool(toc_result.get("toc_content")) and toc_result.get("page_index_given_in_toc") == 'yes'
        except Exception as exc:
            logger.warning(f"TOC 检测失败，降级为简单分块: {exc}")
            has_toc = False

        if has_toc:
            print("检测到目录，走完整 page_index 流程")
            # 有目录 → 走完整 page_index 流程以获得精准结构
            reset_token_count()
            json_path = await page_index(
                file_path,
                model="Qwen3-8B",
                toc_check_page_num=10,
                max_page_num_each_node=10,
                max_token_num_each_node=10000,
                if_add_node_id="yes",
                if_add_node_summary="no",
                if_add_node_text="yes",
            )
            token_count = get_token_count()
            llm_token = token_count.get("input", 0) + token_count.get("output", 0)
            return json_path, llm_token

        # 无目录 + 不用 LLM → 简单分块（快速、零 token）
        print("无目录，按字符分块")
        parser = PdfSimpleParser()
        # 从 document.title 中提取文件名（不含扩展名）作为 doc_name
        doc_name = os.path.splitext(document.title)[0]
        result = parser.parse(file_path, doc_name=doc_name)
        json_path = UniversalRagService._save_result_to_json(result, document.doc_id)
        return json_path, 0

    @staticmethod
    def _process_other(file_path: str, ext: str, document) -> str:
        """解析非 PDF 文档，返回临时 JSON 文件路径"""
        parser_cls = _EXT_PARSER_MAP.get(ext)
        if parser_cls is None:
            raise InternalServerErrorException(detail=f"不支持的文件格式: {ext}")

        parser = parser_cls()
        # 从 document.title 中提取文件名（不含扩展名）作为 doc_name
        doc_name = os.path.splitext(document.title)[0]
        result = parser.parse(file_path, doc_name=doc_name)
        return UniversalRagService._save_result_to_json(result, document.doc_id)

    @staticmethod
    def _save_result_to_json(result: Dict[str, Any], doc_id: str) -> str:
        """
        将解析结果保存为与 page_index 输出格式兼容的临时 JSON 文件，
        返回文件路径。

        page_index 输出格式（ESConverter 期望的格式）：
        {
          "doc_name": "...",
          "structure": [...]
        }
        """
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
            prefix=f"rag_{doc_id}_",
        ) as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            return f.name
