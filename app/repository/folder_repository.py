from typing import List, Optional
from sqlalchemy.orm import Session
from ..models.models import Folder


class FolderRepository:
    @staticmethod
    def create(db: Session, folder_data: dict) -> Folder:
        """创建文件夹"""
        db_folder = Folder(**folder_data)
        db.add(db_folder)
        db.commit()
        db.refresh(db_folder)
        return db_folder
    
    @staticmethod
    def get_by_id(db: Session, fd_id: str) -> Optional[Folder]:
        """根据ID获取文件夹"""
        return db.query(Folder).filter(Folder.fd_id == fd_id).first()
    
    @staticmethod
    def get_by_kb(db: Session, kb_id: str, skip: int = 0, limit: int = 100) -> List[Folder]:
        """根据知识库ID获取文件夹列表"""
        return db.query(Folder).filter(Folder.kb_id == kb_id).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_by_parent(db: Session, parent_id: str, kb_id: Optional[str] = None, skip: int = 0, limit: int = 100) -> List[Folder]:
        """根据父文件夹ID获取子文件夹列表"""
        query = db.query(Folder).filter(Folder.parent_id == parent_id)
        if kb_id:
            query = query.filter(Folder.kb_id == kb_id)
        return query.offset(skip).limit(limit).all()
    
    @staticmethod
    def get_all(db: Session, skip: int = 0, limit: int = 100) -> List[Folder]:
        """获取文件夹列表"""
        return db.query(Folder).offset(skip).limit(limit).all()
    
    @staticmethod
    def update(db: Session, fd_id: str, **kwargs) -> Optional[Folder]:
        """更新文件夹信息"""
        db_folder = db.query(Folder).filter(Folder.fd_id == fd_id).first()
        if db_folder:
            for key, value in kwargs.items():
                setattr(db_folder, key, value)
            db.commit()
            db.refresh(db_folder)
        return db_folder
    
    @staticmethod
    def delete(db: Session, fd_id: str) -> bool:
        """删除文件夹"""
        db_folder = db.query(Folder).filter(Folder.fd_id == fd_id).first()
        if db_folder:
            db.delete(db_folder)
            db.commit()
            return True
        return False
    
    @staticmethod
    def folder_exists(db: Session, name: str, parent_id: str, kb_id: str, exclude_fd_id: Optional[str] = None) -> bool:
        """检查文件夹是否存在"""
        if parent_id is None or parent_id == "":
            parent_id = "0"
        query = db.query(Folder).filter(
            Folder.kb_id == kb_id,
            Folder.parent_id == parent_id,
            Folder.name == name
        )
        if exclude_fd_id:
            query = query.filter(Folder.fd_id != exclude_fd_id)
        existing_folder = query.first()
        return existing_folder is not None
    
    @staticmethod
    def get_all_subfolders(db: Session, fd_id: str) -> List[Folder]:
        """获取所有子文件夹（包括所有层级）"""
        # 获取当前文件夹
        current_folder = db.query(Folder).filter(Folder.fd_id == fd_id).first()
        if not current_folder:
            return []
        return db.query(Folder).filter(Folder.path.like(f"{fd_id}%")).all()
    
    @staticmethod
    def get_all_subfolders_by_ids(db: Session, fd_ids: List[str]) -> List[Folder]:
        """根据多个文件夹ID获取所有子文件夹（包括所有层级）"""
        if not fd_ids:
            return []
        
        # 构建查询条件
        from sqlalchemy import or_
        conditions = []
        for fd_id in fd_ids:
            conditions.append(Folder.path.like(f"{fd_id}%"))
        
        # 执行查询
        return db.query(Folder).filter(or_(*conditions)).all()