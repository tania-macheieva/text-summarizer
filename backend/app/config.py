"""Централізована конфігурація (шляхи, ліміти, параметри за замовчуванням)."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DEMO_DIR = Path(os.getenv("DEMO_DATA_DIR", DATA_DIR / "demo"))
BBC_DIR = Path(os.getenv("BBC_DATA_DIR", DATA_DIR / "bbc"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", BASE_DIR / "artifacts"))
MODEL_FILE = MODELS_DIR / "rf_summarizer.joblib"

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)  # train / validation / test

MAX_TEXT_CHARS = 200_000
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}
DEFAULT_RATIO = 0.4  # частка речень, що потрапляє в резюме (≈ середнє по BBC News Summary)

CORS_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if o.strip()
]
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
