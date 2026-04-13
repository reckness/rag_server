from fastapi import HTTPException, status


class AppException(HTTPException):
    """应用基础异常类"""
    def __init__(self, detail: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        super().__init__(status_code=status_code, detail=detail)


class NotFoundException(AppException):
    """资源未找到异常"""
    def __init__(self, detail: str = "Resource not found"):
        super().__init__(detail=detail, status_code=status.HTTP_404_NOT_FOUND)


class BadRequestException(AppException):
    """请求参数错误异常"""
    def __init__(self, detail: str = "Bad request"):
        super().__init__(detail=detail, status_code=status.HTTP_400_BAD_REQUEST)


class UnauthorizedException(AppException):
    """未授权异常"""
    def __init__(self, detail: str = "Unauthorized"):
        super().__init__(detail=detail, status_code=status.HTTP_401_UNAUTHORIZED)


class ForbiddenException(AppException):
    """禁止访问异常"""
    def __init__(self, detail: str = "Forbidden"):
        super().__init__(detail=detail, status_code=status.HTTP_403_FORBIDDEN)


class InternalServerErrorException(AppException):
    """服务器内部错误异常"""
    def __init__(self, detail: str = "Internal server error"):
        super().__init__(detail=detail, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ConflictException(AppException):
    """资源冲突异常"""
    def __init__(self, detail: str = "Conflict"):
        super().__init__(detail=detail, status_code=status.HTTP_409_CONFLICT)