from typing import Generic, Optional, TypeVar, Any
from pydantic import BaseModel

T = TypeVar('T')


class ApiResponse(BaseModel, Generic[T]):
    """API 统一响应模型"""
    code: int = 200
    message: str = "success"
    data: Optional[T] = None

    @classmethod
    def success(cls, data: Optional[Any] = None, message: str = "success") -> 'ApiResponse':
        """成功响应"""
        return cls(code=200, message=message, data=data)

    @classmethod
    def error(cls, code: int = 400, message: str = "error", data: Optional[Any] = None) -> 'ApiResponse':
        """错误响应"""
        return cls(code=code, message=message, data=data)