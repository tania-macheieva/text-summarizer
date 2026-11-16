"""Pydantic-схеми запитів і відповідей API."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from .config import MAX_TEXT_CHARS

Language = Literal["auto", "en", "uk"]
Algorithm = Literal["random_forest", "tfidf", "lead"]


class SummarizeRequest(BaseModel):
    text: str = Field(..., description="Текст документа", max_length=MAX_TEXT_CHARS)
    language: Language = "auto"
    algorithm: Algorithm = "random_forest"
    num_sentences: Optional[int] = Field(None, ge=1, le=100, description="Кількість речень у резюме")
    ratio: Optional[float] = Field(None, gt=0, le=1, description="Частка речень оригіналу в резюме (0–1]")

    @model_validator(mode="after")
    def _one_length_param(self):
        if self.num_sentences is not None and self.ratio is not None:
            raise ValueError("Вкажіть або num_sentences, або ratio, але не обидва")
        return self


class KeySentenceOut(BaseModel):
    index: int
    text: str
    score: float
    rank: int


class SummarizeResponse(BaseModel):
    summary: str
    key_sentences: list[KeySentenceOut]
    original_word_count: int
    summary_word_count: int
    compression_ratio: float
    processing_time: float
    model_name: str
    language: str
    original_sentence_count: int
    summary_sentence_count: int
    used_fallback: bool = False
    warnings: list[str] = []
    filename: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    model_trained_at: Optional[str] = None
    uptime_seconds: float


class AlgorithmInfo(BaseModel):
    id: str
    name: str
    description: str
    available: bool
    trained: bool = False
    metadata: dict = {}


class TrainRequest(BaseModel):
    dataset: Literal["auto", "demo", "bbc"] = "auto"
    max_documents: Optional[int] = Field(None, ge=20, le=5000)
    seed: int = Field(42, ge=0, le=2**31 - 1)
    n_estimators: Optional[list[int]] = Field(None, description="Сітка n_estimators (за замовчуванням [200, 400])")
    max_depth: Optional[list[Optional[int]]] = None
    min_samples_leaf: Optional[list[int]] = None
    save: bool = True


class TrainStatus(BaseModel):
    state: Literal["idle", "running", "done", "error"]
    message: str = ""
    progress: float = 0.0
    dataset: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None
