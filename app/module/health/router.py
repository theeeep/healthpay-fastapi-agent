from datetime import datetime

from fastapi import APIRouter

health_router = APIRouter()


@health_router.get("/health")
async def health_check():
    health_data = {"status": "API is running smoothly", "timestamp": datetime.now().isoformat()}
    return health_data
