# src/main.py
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.logger import logger
from app.middleware.error_handler import register_api_exception_handlers
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


version = "v1"
title = "HealthPay API"
description = "HealthPay API"

app = FastAPI(
    version=version,
    title=title,
    description=description,
    lifespan=life_span,
)

register_api_exception_handlers(app)

app.include_router(health_router, tags=["Health"])
app.include_router(process_claim_router, tags=["Claim Processing"])
# app.include_router(chat_router, prefix=f"/{version}", tags=["Chat"])
