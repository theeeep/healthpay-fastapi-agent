from datetime import datetime
from typing import Generic, Optional, TypeVar

from fastapi import status
from pydantic import BaseModel, Field

T = TypeVar("T")


class Response(BaseModel, Generic[T]):
    status: int = Field(default=status.HTTP_200_OK)
    message: str = Field(default="Request processed successfully")
    data: Optional[T] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
