from __future__ import annotations

from fastapi import FastAPI

from app.api.router import api_router
from app.core.settings import settings


def create_app() -> FastAPI:
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    application.include_router(api_router)
    return application


app = create_app()
