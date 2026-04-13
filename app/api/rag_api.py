from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session
from ..core.database import SessionLocal
from ..models.request import SearchRequest
from ..services.rag_service import RagService
from ..utils.response import ApiResponse

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

router = APIRouter()

rag_service = RagService()


@router.post("/search", response_model=ApiResponse, summary="RAG搜索")
async def search(req: SearchRequest, request: Request, db: Session = Depends(get_db)):
    """RAG搜索接口"""
    # 从应用状态获取模型
    model = getattr(request.app.state, "model", None)
    result = await rag_service.search(req, db, model)
    return ApiResponse.success(data=result)