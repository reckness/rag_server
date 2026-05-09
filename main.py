import asyncio
import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app import api_router
from app.core.database import engine, Base
#from app.tasks.scheduler import start_scheduler

# 创建数据库表
Base.metadata.create_all(bind=engine)

# Rerank 配置（通过远程 HTTP API 调用）
RERANK_API_URL = "http://10.1.141.33:8474/rerank"
UPLOADED_DOCUMENT_POLL_INTERVAL = int(os.getenv("UPLOADED_DOCUMENT_POLL_INTERVAL", "5"))
UPLOADED_DOCUMENT_POLL_LIMIT = int(os.getenv("UPLOADED_DOCUMENT_POLL_LIMIT", "100"))
PROCESSING_DOC_IDS = set()

def load_model():
    """初始化 rerank（远程 API 模式，无需加载本地模型）"""
    print(f"Rerank API 地址: {RERANK_API_URL}")
    return RERANK_API_URL

async def process_uploaded_document(doc_id: str, title: str):
    from app.api.universal_rag_api import _process_single_doc

    try:
        print(f"开始解析上传文档: {title} (ID: {doc_id})")
        result = await _process_single_doc(None, doc_id)
        if result.get("success"):
            print(f"上传文档解析完成: {title} (ID: {doc_id})")
        else:
            print(f"上传文档解析失败: {title} (ID: {doc_id}), {result.get('error')}")
    except Exception as e:
        print(f"上传文档解析异常: {title} (ID: {doc_id}), {str(e)}")
    finally:
        PROCESSING_DOC_IDS.discard(doc_id)


async def poll_uploaded_documents():
    from app.repository.document_repository import DocumentRepository
    from app.core.database import SessionLocal

    while True:
        db = SessionLocal()
        try:
            uploaded_documents = DocumentRepository.get_uploaded_ordered(db, limit=UPLOADED_DOCUMENT_POLL_LIMIT)
            for document in uploaded_documents:
                doc_id = str(document.doc_id)
                if doc_id in PROCESSING_DOC_IDS:
                    continue
                PROCESSING_DOC_IDS.add(doc_id)
                asyncio.create_task(process_uploaded_document(doc_id, document.title))
        except Exception as e:
            print(f"轮询 uploaded 文档失败: {str(e)}")
        finally:
            db.close()
        await asyncio.sleep(UPLOADED_DOCUMENT_POLL_INTERVAL)

# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 只在应用启动时加载一次
    app.state.model = load_model()
    app.state.uploaded_document_poll_task = asyncio.create_task(poll_uploaded_documents())

    try:
        yield
    finally:
        app.state.uploaded_document_poll_task.cancel()
        try:
            await app.state.uploaded_document_poll_task
        except asyncio.CancelledError:
            pass

app = FastAPI(
    title="Deep Search API",
    description="知识库管理系统API",
    version="1.0.0",
    lifespan=lifespan
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
