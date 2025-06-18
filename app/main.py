from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import Config
from app.core.logger import logger
from app.middleware.error_handler import register_api_exception_handlers
from app.middleware.rate_limiter import RateLimiterMiddleware
from app.middleware.request_id import RequestIDMiddleware
from app.module.health.router import health_router
from app.module.process_claim.router import process_claim_router

logger.info("Main Application")

# API versioning from config
API_PREFIX = f"/api/{Config.API_VERSION}"


@asynccontextmanager
async def life_span(app: FastAPI):
    """Application lifecycle events"""
    logger.info("🚀 Starting Application...")
    logger.info("✅ Application Started Successfully...")
    yield
    logger.info("🛑 Stopping Application...")
    logger.info("👋 Application Stopped Successfully...")


app = FastAPI(
    version=Config.APP_VERSION,  # This shows full version in docs
    title=Config.APP_NAME,
    description="AI-powered medical insurance claims processing system",
    lifespan=life_span,
    debug=Config.DEBUG,
    docs_url=f"{API_PREFIX}/docs",
    redoc_url=f"{API_PREFIX}/redoc",
    openapi_url=f"{API_PREFIX}/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add security and monitoring middleware
app.add_middleware(RequestIDMiddleware)
app.add_middleware(RateLimiterMiddleware, requests_per_minute=60)

# Register exception handlers
register_api_exception_handlers(app)

# Include routers with versioned prefix
app.include_router(health_router, prefix=API_PREFIX, tags=["Health"])
app.include_router(process_claim_router, prefix=API_PREFIX, tags=["Claim Processing"])
