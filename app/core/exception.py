from typing import Any, Optional, Dict

from fastapi import status
from fastapi import HTTPException
from starlette.responses import JSONResponse

from app.models.ModelResponse import ErrorData, ResponseError

async def http_exception_handler(request, exc):
    content = ResponseError(error=ErrorData(code=exc.status_code, message=exc.detail)).model_dump(mode='json')
    return JSONResponse(content, status_code=exc.status_code)

class AuthError(HTTPException):
    def __init__(
        self, detail: Any = None, headers: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(status.HTTP_401_UNAUTHORIZED, detail, headers)

class NotFoundError(HTTPException):
    def __init__(
        self, detail: Any = None, headers: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(status.HTTP_404_NOT_FOUND, detail, headers)
        
class ForbiddenError(HTTPException):
    def __init__(
        self, detail: Any = None, headers: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(status.HTTP_403_FORBIDDEN, detail, headers)
        
class InternalServerError(HTTPException):
    def __init__(
        self, detail: Any = None, headers: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(status.HTTP_500_INTERNAL_SERVER_ERROR, detail, headers)