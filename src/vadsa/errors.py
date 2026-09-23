"""Errors in the OpenAI shape: {"error": {"message", "type", "param", "code"}}."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException


class ErrorInfo(BaseModel):
    message: str
    type: str
    param: str | None
    code: str | None


class ErrorResponse(BaseModel):
    error: ErrorInfo


class ApiError(Exception):
    def __init__(
        self,
        status: int,
        message: str,
        *,
        code: str | None = None,
        param: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code
        self.param = param
        self.headers = headers


def error_response(
    status: int,
    message: str,
    *,
    code: str | None = None,
    param: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    info = ErrorInfo(
        message=message,
        type="server_error" if status >= 500 else "invalid_request_error",
        param=param,
        code=code,
    )
    body = ErrorResponse(error=info).model_dump()
    return JSONResponse(body, status_code=status, headers=headers)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError) -> JSONResponse:
        return error_response(
            exc.status, exc.message, code=exc.code, param=exc.param, headers=exc.headers
        )

    # unknown routes, wrong methods
    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return error_response(exc.status_code, str(exc.detail), headers=exc.headers)

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception) -> JSONResponse:
        return error_response(500, "Internal server error.", code="internal_error")
