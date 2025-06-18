# src/main.py
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config.settings import Config
from app.core.logger import logger
from app.middleware.error_handler import register_api_exception_handlers
from app.middleware.rate_limiter import RateLimiterMiddleware
from app.middleware.request_id import RequestIDMiddleware
from app.module.health.router import health_router
from app.module.process_claim.router import process_claim_router

logger.info("Main Application")


@asynccontextmanager
async def life_span(app: FastAPI):
    logger.info("🚀 Starting Application...")
    logger.info("✅ Application Started Successfully...")
    yield
    logger.info("🛑 Stopping Application...")
    logger.info("👋 Application Stopped Successfully...")


app = FastAPI(
    version=Config.APP_VERSION,
    title=Config.APP_NAME,
    description="AI-powered medical insurance claims processing system",
    lifespan=life_span,
    debug=Config.DEBUG,
)

# Add middleware
app.add_middleware(RequestIDMiddleware)
app.add_middleware(RateLimiterMiddleware, requests_per_minute=60)

# Register exception handlers
register_api_exception_handlers(app)

# Include routers
app.include_router(health_router, tags=["Health"])
app.include_router(process_claim_router, tags=["Claim Processing"])
# app.include_router(chat_router, prefix=f"/{version}", tags=["Chat"])
