"""REST API."""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .. import __version__, config
from ..data.loader import DatasetError
from ..ml.summarizer import SummarizerError
from ..schemas import (AlgorithmInfo, HealthResponse, SummarizeRequest, SummarizeResponse,
                       TrainRequest, TrainStatus)
from ..services.file_parser import FileParseError, extract_text
from ..services.registry import read_json_artifact, registry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


def _build_response(res, filename: str | None = None) -> SummarizeResponse:
    return SummarizeResponse(
        summary=res.summary,
        key_sentences=[{"index": k.index, "text": k.text, "score": k.score, "rank": k.rank} for k in res.key_sentences],
        original_word_count=res.original_word_count, summary_word_count=res.summary_word_count,
        compression_ratio=res.compression_ratio, processing_time=res.processing_time, model_name=res.model_name,
        language=res.language, original_sentence_count=len(res.sentences),
        summary_sentence_count=len(res.selected_indices), used_fallback=res.used_fallback,
        warnings=res.warnings, filename=filename)


def _run(text: str, language: str, algorithm: str, num_sentences, ratio, filename=None) -> SummarizeResponse:
    try:
        res = registry.model.summarize(text, language, num_sentences, ratio, algorithm)
    except SummarizerError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    logger.info("summarize model=%s lang=%s words=%d->%d time=%.3fs", res.model_name, res.language,
                res.original_word_count, res.summary_word_count, res.processing_time)
    return _build_response(res, filename)


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    md = registry.model.metadata if registry.model.is_trained else {}
    return HealthResponse(status="ok", version=__version__, model_loaded=registry.model.is_trained,
                          model_trained_at=md.get("trained_at"),
                          uptime_seconds=round(time.time() - registry.started, 1))


@router.post("/summarize", response_model=SummarizeResponse, tags=["summarization"])
def summarize(req: SummarizeRequest):
    return _run(req.text, req.language, req.algorithm, req.num_sentences, req.ratio)


@router.post("/summarize/file", response_model=SummarizeResponse, tags=["summarization"])
async def summarize_file(
    file: UploadFile = File(..., description="TXT, PDF, or DOCX"),
    language: str = Form("auto"),
    algorithm: str = Form("random_forest"),
    num_sentences: int | None = Form(None),
    ratio: float | None = Form(None),
):
    if language not in ("auto", "en", "uk"):
        raise HTTPException(422, "language must be auto, en, or uk")
    if algorithm not in ("random_forest", "tfidf", "lead"):
        raise HTTPException(422, "Unknown algorithm")
    if num_sentences is not None and ratio is not None:
        raise HTTPException(422, "Specify either num_sentences or ratio, not both")
    if ratio is not None and not 0 < ratio <= 1:
        raise HTTPException(422, "ratio must be within (0, 1]")
    if num_sentences is not None and not 1 <= num_sentences <= 100:
        raise HTTPException(422, "num_sentences must be between 1 and 100")
    data = await file.read(config.MAX_UPLOAD_BYTES + 1)
    try:
        text = extract_text(file.filename or "", data)
    except FileParseError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    return _run(text, language, algorithm, num_sentences, ratio, file.filename)


@router.get("/models", response_model=list[AlgorithmInfo], tags=["models"])
def models():
    trained = registry.model.is_trained
    md = registry.model.metadata if trained else {}
    brief = {k: md[k] for k in ("dataset", "trained_at", "best_params", "ratio_used", "split_sizes", "seed") if k in md}
    return [
        AlgorithmInfo(id="random_forest", name="Random Forest (TF-IDF features)",
                      description="Sentence-importance classifier using 18 features; falls back to TF-IDF when no trained model is available.",
                      available=True, trained=trained, metadata=brief),
        AlgorithmInfo(id="tfidf", name="TF-IDF ranking",
                      description="Baseline that ranks sentences by their mean TF-IDF term weight.", available=True),
        AlgorithmInfo(id="lead", name="Lead-N",
                      description="Baseline that selects the first N sentences of the document.", available=True),
    ]


@router.get("/dataset", tags=["data"])
def dataset():
    info = read_json_artifact("dataset_info.json")
    if info is None:
        raise HTTPException(404, "Dataset information is unavailable. Run scripts/prepare_dataset.py")
    return info


@router.get("/evaluation", tags=["evaluation"])
def evaluation():
    res = read_json_artifact("evaluation.json")
    if res is None:
        raise HTTPException(404, "Evaluation results are unavailable. Run scripts/evaluate.py")
    return res


@router.post("/train", response_model=TrainStatus, status_code=202, tags=["training"])
def train(req: TrainRequest | None = None):
    """Train in the background while the current model remains available."""
    req = req or TrainRequest()
    try:
        job = registry.start_training(req)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (FileNotFoundError, DatasetError) as exc:
        raise HTTPException(404, str(exc)) from exc
    return TrainStatus(**job)


@router.get("/train/status", response_model=TrainStatus, tags=["training"])
def train_status():
    return TrainStatus(**registry.job)
