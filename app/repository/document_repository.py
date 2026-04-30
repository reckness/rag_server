from typing import List, Optional
from sqlalchemy.orm import Session
from ..models.models import Document


class DocumentRepository:
    @staticmethod
    def create(db: Session, doc_data: dict) -> Document:
        """创建文档"""
        db_doc = Document(**doc_data)
        db.add(db_doc)
        db.commit()
        db.refresh(db_doc)
        return db_doc
    
    @staticmethod
    def get_by_id(db: Session, doc_id: str) -> Optional[Document]:
        """根据ID获取文档"""
        return db.query(Document).filter(Document.doc_id == doc_id).first()
    
    @staticmethod
    def get_by_kb(db: Session, kb_id: str, skip: int = 0, limit: int = 100) -> List[Document]:
        """根据知识库ID获取文档列表"""
        return db.query(Document).filter(Document.kb_id == kb_id).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_by_folder(db: Session, fd_id: str, kb_id: Optional[str] = None, skip: int = 0, limit: int = 100) -> List[Document]:
        """根据文件夹ID获取文档列表"""
        query = db.query(Document).filter(Document.fd_id == fd_id)
        if kb_id:
            query = query.filter(Document.kb_id == kb_id)
        return query.offset(skip).limit(limit).all()

    @staticmethod
    def get_by_folders(db: Session, fd_ids: List[str]) -> List[Document]:
        return db.query(Document).filter(Document.fd_id.in_(fd_ids)).all()
    
    @staticmethod
    def update_kb_id_by_folders(db: Session, fd_ids: List[str], new_kb_id: str) -> int:
        updated = db.query(Document).filter(Document.fd_id.in_(fd_ids)).update(
            {Document.kb_id: new_kb_id},
            synchronize_session=False
        )
        db.commit()
        return updated
    
    @staticmethod
    def get_by_status(db: Session, status: str, skip: int = 0, limit: int = 100) -> List[Document]:
        """根据状态获取文档列表"""
        return db.query(Document).filter(Document.status == status).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_all(db: Session, skip: int = 0, limit: int = 100) -> List[Document]:
        """获取文档列表"""
        return db.query(Document).offset(skip).limit(limit).all()
    
    @staticmethod
    def update(db: Session, doc_id: str, **kwargs) -> Optional[Document]:
        """更新文档信息"""
        db_doc = db.query(Document).filter(Document.doc_id == doc_id).first()
        if db_doc:
            for key, value in kwargs.items():
                setattr(db_doc, key, value)
            db.commit()
            db.refresh(db_doc)
        return db_doc
    
    @staticmethod
    def delete(db: Session, doc_id: str) -> bool:
        """删除文档"""
        db_doc = db.query(Document).filter(Document.doc_id == doc_id).first()
        if db_doc:
            db.delete(db_doc)
            db.commit()
            return True
        return False