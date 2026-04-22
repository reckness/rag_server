from fastapi import APIRouter
from .document_processing_api import router as document_processing_router
from .rag_api import router as rag_router
from .universal_rag_api import router as universal_rag_router
from .pdf_to_md_api import router as pdf_to_md_router

api_router = APIRouter()
api_router.include_router(document_processing_router, prefix="/document", tags=["document"])
api_router.include_router(rag_router, prefix="/rag", tags=["rag"])
api_router.include_router(universal_rag_router, prefix="/universal", tags=["universal-rag"])
api_router.include_router(pdf_to_md_router, prefix="/pdf", tags=["pdf-to-md"])