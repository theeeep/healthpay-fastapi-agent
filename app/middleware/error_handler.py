from datetime import datetime

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse


class APIException(Exception):
    def __init__(
        self,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        message: str = "Internal Server Error",
        details: str | None = None,
    ):
        self.status_code = status_code
        self.message = message
        self.details = details
        super().__init__(self.message)


async def api_exception_handler(request: Request, exc: Exception):
    status_code = getattr(exc, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
    message = getattr(exc, "message", "Internal Server Error")
    details = getattr(exc, "details", str(exc))

    return JSONResponse(
        status_code=status_code,
        content={
            "status": status_code,
            "message": message,
            "details": details,
            "path": request.url.path,
            "timestamp": datetime.now().isoformat(),
            "request_id": request.state.request_id,
        },
    )


# Common exceptions for reuse
class BadRequestError(APIException):
    def __init__(self, message: str, details: str | None = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            message=message,
            details=details,
        )


class NotFoundError(APIException):
    def __init__(self, message: str, details: str | None = None):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            message=message,
            details=details,
        )


class UnauthorizedError(APIException):
    def __init__(self, message: str = "Unauthorized", details: str | None = None):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            message=message,
            details=details,
        )


def register_api_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers for the application."""
    handlers = {
        APIException: api_exception_handler,
        # Add more exception-handler pairs here as needed
    }

    for exception_class, handler in handlers.items():
        app.add_exception_handler(exception_class, handler)
