"""Точка входу FastAPI."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__, config
from .api.routes import router
from .nlp.preprocessing import ensure_nltk
from .services.registry import registry

logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_nltk()
    registry.load_from_disk()
    logger.info("Сервіс запущено (модель навчена: %s)", registry.model.is_trained)
    yield


app = FastAPI(
    title="Text Summarizer API",
    description="Екстрактивна самаризація текстів: TF-IDF + Random Forest (NLTK, scikit-learn).",
    version=__version__, lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
app.include_router(router)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    errors = [{"field": ".".join(str(p) for p in e["loc"][1:]) or str(e["loc"][0]), "message": e["msg"]}
              for e in exc.errors()]
    first = errors[0] if errors else {"field": "", "message": "invalid"}
    return JSONResponse(status_code=422, content={"detail": f"{first['field']}: {first['message']}", "errors": errors})


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    logger.exception("Необроблена помилка: %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Внутрішня помилка сервера"})
