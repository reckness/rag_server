from fastapi import FastAPI
from contextlib import asynccontextmanager
from app import api_router
from app.core.database import engine, Base
#from app.tasks.scheduler import start_scheduler

# 创建数据库表
Base.metadata.create_all(bind=engine)

# 加载模型函数
def load_model():
    """加载模型"""
    try:
        from sentence_transformers import CrossEncoder
        # 从本地加载模型
        model = CrossEncoder("./models/bge-reranker-base")
        print("模型加载成功")
        return model
    except ImportError as e:
        # 处理缺少依赖的情况
        print(f"Rerank not available: {e}")
        return None

# 处理上传的文档
async def process_uploaded_documents():
    """处理状态为uploaded的文档"""
    from app.repository.document_repository import DocumentRepository
    from app.services.document_processing_service import DocumentProcessingService
    from app.core.database import SessionLocal
    
    db = SessionLocal()
    try:
        # 获取所有状态为uploaded的文档
        uploaded_documents = DocumentRepository.get_by_status(db, "uploaded")
        
        if not uploaded_documents:
            print("没有需要处理的上传文档")
            return
        
        print(f"开始处理 {len(uploaded_documents)} 个上传文档")
        
        for document in uploaded_documents:
            try:
                print(f"处理文档: {document.title} (ID: {document.doc_id})")
                # 调用文档处理服务
                await DocumentProcessingService.process_document(db, document.doc_id)
                print(f"文档处理完成: {document.title}")
            except Exception as e:
                print(f"处理文档 {document.title} 时出错: {str(e)}")
                # 继续处理下一个文档
                continue
    finally:
        db.close()

# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 只在应用启动时加载一次
    app.state.model = load_model()
    
    # 处理上传的文档
    await process_uploaded_documents()
    
    yield
    # 清理资源

app = FastAPI(
    title="Deep Search API",
    description="知识库管理系统API",
    version="1.0.0",
    #lifespan=lifespan
)

# 包含API路由
app.include_router(api_router)

# 启动调度器
#start_scheduler()

@app.get("/")
def read_root():
    """根路径"""
    return {"message": "Welcome to Deep Search API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
