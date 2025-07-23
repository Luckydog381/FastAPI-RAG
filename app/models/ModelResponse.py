from pydantic import BaseModel
from typing import Any

class ResponseSuccess(BaseModel):
    status: str = "SUCCESS"
    result: Any
class ErrorData(BaseModel):
    code: int
    message: str
class ResponseError(BaseModel):
    status: str = "ERROR"
    error: ErrorData
