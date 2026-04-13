from typing import List, Optional
from sqlalchemy.orm import Session
from ..models.models import Folder
from ..utils.uuid_generator import generate_uuid
from ..repository.folder_repository import FolderRepository
from ..core.exceptions import NotFoundException


class FolderService:
    @staticmethod
    def create_folder(db: Session, name: str, parent_id: Optional[str], kb_id: str, path: str, level: int = 0, created_by: Optional[str] = None) -> Folder:
        """创建文件夹"""
        # 检查当前知识库下的当前层级是否存在相同名称的文件夹
        if FolderRepository.folder_exists(db, name, parent_id, kb_id):
            raise ValueError(f"Folder with name '{name}' already exists in the current directory")
        
        # 生成UUID作为fd_id
        fd_id = generate_uuid()
        folder_data = {
            "fd_id": fd_id,
            "name": name,
            "parent_id": parent_id,
            "kb_id": kb_id,
            "path": path,
            "level": level,
            "created_by": created_by
        }
        return FolderRepository.create(db, folder_data)
    
    @staticmethod
    def create_folder_with_path(db: Session, name: str, kb_id: str, parent_id: Optional[str] = None, created_by: Optional[str] = None) -> Folder:
        """创建文件夹并生成路径"""
      
        # 如果 parent_id 为 null 或空字符串，赋值为 "0"
        if parent_id is None or parent_id == "":
            parent_id = "0"
        
        # 检查当前知识库下的当前层级是否存在相同名称的文件夹
        if FolderRepository.folder_exists(db, name, parent_id, kb_id):
            raise ValueError(f"Folder with name '{name}' already exists in the current directory")
        
        # 生成UUID作为fd_id
        fd_id = generate_uuid()
        
        # 计算level和path
        if parent_id == "0":
            # 知识库的第一个文件夹
            level = 1
            # 生成path为fd_id
            path = f"{fd_id}"
        else:
            # 查询父文件夹
            parent_folder = FolderService.get_folder_by_id(db, parent_id)
            if not parent_folder:
                raise NotFoundException(detail="Parent folder not found")
            # 计算level
            level = parent_folder.level + 1
            # 生成path为父文件夹path加上当前fd_id
            path = f"{parent_folder.path},{fd_id}"
        
        # 直接创建文件夹，一次性设置所有字段
        folder_data = {
            "fd_id": fd_id,
            "name": name,
            "parent_id": parent_id,
            "kb_id": kb_id,
            "path": path,
            "level": level,
            "created_by": created_by
        }
        return FolderRepository.create(db, folder_data)
    
    @staticmethod
    def get_folder_by_id(db: Session, fd_id: str) -> Optional[Folder]:
        """根据ID获取文件夹"""
        return FolderRepository.get_by_id(db, fd_id)
    
    @staticmethod
    def get_folders(db: Session, skip: int = 0, limit: int = 100) -> List[Folder]:
        """获取文件夹列表"""
        return FolderRepository.get_all(db, skip, limit)
    
    @staticmethod
    def update_folder(db: Session, fd_id: str, **kwargs) -> Optional[Folder]:
        """更新文件夹信息"""
        # 获取要更新的文件夹
        db_folder = FolderRepository.get_by_id(db, fd_id)
        if not db_folder:
            raise NotFoundException(detail="Folder not found")
        
        # 检查是否更新了名称
        if 'name' in kwargs:
            # 获取当前文件夹的父ID和知识库ID
            parent_id = db_folder.parent_id
            kb_id = db_folder.kb_id
            # 检查新名称是否与同一层级下的其他文件夹重复
            if FolderRepository.folder_exists(db, kwargs['name'], parent_id, kb_id, fd_id):
                raise ValueError(f"Folder with name '{kwargs['name']}' already exists in the current directory")
        
        return FolderRepository.update(db, fd_id, **kwargs)
    
    @staticmethod
    def delete_folder(db: Session, fd_id: str) -> bool:
        """删除文件夹"""
        return FolderRepository.delete(db, fd_id)
    
    @staticmethod
    def get_folders_by_parent(db: Session, parent_id: str, kb_id: Optional[str] = None, skip: int = 0, limit: int = 100) -> List[Folder]:
        """根据父文件夹ID获取子文件夹列表"""
        return FolderRepository.get_by_parent(db, parent_id, kb_id, skip, limit)
    
    @staticmethod
    def get_all_subfolders(db: Session, fd_id: str) -> List[Folder]:
        """获取所有子文件夹（包括所有层级）"""
        return FolderRepository.get_all_subfolders(db, fd_id)
    
    @staticmethod
    def get_all_subfolders_by_ids(db: Session, fd_ids: List[str]) -> List[Folder]:
        """根据多个文件夹ID获取所有子文件夹（包括所有层级）"""
        return FolderRepository.get_all_subfolders_by_ids(db, fd_ids)