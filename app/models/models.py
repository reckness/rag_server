from sqlalchemy import Column, Integer, BigInteger, String, Text, ForeignKey, DateTime, CheckConstraint, UniqueConstraint, Float
from sqlalchemy.dialects.postgresql import BYTEA
from sqlalchemy.sql import func
from ..core.database import Base


class User(Base):
    __tablename__ = "user_info"
    
    id = Column(String(36), primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    created_by = Column(String(36), ForeignKey("user_info.id"))
    create_time = Column(DateTime(timezone=True), server_default=func.now())
    update_time = Column(DateTime(timezone=True), server_default=func.now())


class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"
    
    kb_id = Column(String(36), primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    type = Column(String(20), nullable=False)
    owner_id = Column(String(36), ForeignKey("user_info.id"), nullable=False)
    description = Column(Text)
    cover = Column(String(500))
    status = Column(Integer, default=1)
    file_count = Column(Integer, default=0)
    total_file_size = Column(BigInteger, default=0)
    file_reference_count = Column(Integer, default=0)
    create_time = Column(DateTime(timezone=True), server_default=func.now())
    update_time = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        CheckConstraint("type IN ('personal', 'team', 'public')"),
        CheckConstraint("status IN (0, 1, 2)"),
    )


class Folder(Base):
    __tablename__ = "folder"
    
    fd_id = Column(String(36), primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    parent_id = Column(String(36), ForeignKey("folder.fd_id"))
    kb_id = Column(String(36), ForeignKey("knowledge_base.kb_id"), nullable=False)
    path = Column(Text, nullable=False)
    level = Column(Integer, default=0)
    created_by = Column(String(36), ForeignKey("user_info.id"))
    create_time = Column(DateTime(timezone=True), server_default=func.now())
    update_time = Column(DateTime(timezone=True), server_default=func.now())


class Document(Base):
    __tablename__ = "document"
    
    doc_id = Column(String(36), primary_key=True, index=True)
    kb_id = Column(String(36), ForeignKey("knowledge_base.kb_id"), nullable=False)
    fd_id = Column(String(36), ForeignKey("folder.fd_id"), default='0')
    title = Column(String(500), nullable=False)
    file_type = Column(String(50))
    file_size = Column(BigInteger)
    source_path = Column(String(1000))
    pdf_path = Column(String(1000))
    pageindex_path = Column(String(1000))
    chunk_path = Column(String(1000))
    chunk_num = Column(Integer, nullable=False, default=0)
    progress = Column(Float, nullable=False, default=0.0)
    progress_msg = Column(Text)
    process_begin_at = Column(DateTime(timezone=True), nullable=True)
    process_duration = Column(Float, nullable=False, default=0.0)
    llm_token = Column(Integer, nullable=False, default=0)
    status = Column(String(20), default="uploaded")
    created_by = Column(String(36), ForeignKey("user_info.id"))
    create_time = Column(DateTime(timezone=True), server_default=func.now())
    update_time = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        CheckConstraint("status IN ('uploaded', 'ready', 'error')"),
    )



class UserKbRole(Base):
    __tablename__ = "user_kb_role"
    
    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(String(36), ForeignKey("user_info.id"), nullable=False)
    kb_id = Column(String(36), ForeignKey("knowledge_base.kb_id"), nullable=False)
    role = Column(String(20), nullable=False)
    
    __table_args__ = (
        UniqueConstraint("user_id", "kb_id"),
        CheckConstraint("role IN ('owner', 'editor', 'viewer')"),
    )


class File(Base):
    __tablename__ = "file"
    
    file_id = Column(String(36), primary_key=True, index=True)
    name = Column(String(500), nullable=False)
    path = Column(String(1000), nullable=False)
    file_type = Column(String(50))
    file_size = Column(BigInteger)
    bucket_name = Column(String(100), nullable=False)
    created_by = Column(String(36), ForeignKey("user_info.id"))
    create_time = Column(DateTime(timezone=True), server_default=func.now())
    update_time = Column(DateTime(timezone=True), server_default=func.now())