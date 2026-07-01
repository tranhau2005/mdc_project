from __future__ import annotations
from contextlib import asynccontextmanager 
from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool
from app.core.settings import settings
from app.services.model_service import ModelService
@asynccontextmanager
async def lifespan(app: FastAPI): 
    model_service= ModelService(
        config_path=settings.model_config_path
    )
    app.state.model_serice = model_service
    try: 
        await run_in_threadpool(model_service.load)
        yield
    finally: 
        model_service.close()