from typing import List, Optional
from sqlalchemy.orm import Session
from ..models.models import File


class FileRepository:
    @staticmethod
    def create(db: Session, file_data: dict) -> File:
        """创建文件记录"""
        db_file = File(**file_data)
        db.add(db_file)
        db.commit()
        db.refresh(db_file)
        return db_file
    
    @staticmethod
    def get_by_id(db: Session, file_id: str) -> Optional[File]:
        """根据ID获取文件"""
        return db.query(File).filter(File.file_id == file_id).first()
    
    @staticmethod
    def get_by_bucket(db: Session, bucket_name: str, skip: int = 0, limit: int = 100) -> List[File]:
        """根据存储桶名称获取文件列表"""
        return db.query(File).filter(File.bucket_name == bucket_name).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_by_created_by(db: Session, created_by: str, skip: int = 0, limit: int = 100) -> List[File]:
        """根据创建人ID获取文件列表"""
        return db.query(File).filter(File.created_by == created_by).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_all(db: Session, skip: int = 0, limit: int = 100) -> List[File]:
        """获取文件列表"""
        return db.query(File).offset(skip).limit(limit).all()
    
    @staticmethod
    def update(db: Session, file_id: str, **kwargs) -> Optional[File]:
        """更新文件信息"""
        db_file = db.query(File).filter(File.file_id == file_id).first()
        if db_file:
            for key, value in kwargs.items():
                setattr(db_file, key, value)
            db.commit()
            db.refresh(db_file)
        return db_file
    
    @staticmethod
    def delete(db: Session, file_id: str) -> bool:
        """删除文件"""
        db_file = db.query(File).filter(File.file_id == file_id).first()
        if db_file:
            db.delete(db_file)
            db.commit()
            return True
        return False