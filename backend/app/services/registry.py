"""Реєстр моделі та фонове навчання (один екземпляр на процес)."""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import pandas as pd

from .. import config
from ..data.loader import is_bbc_layout, load_dataset
from ..data.splitter import split_dataset
from ..data.stats import compute_stats
from ..data.validation import validate_dataframe
from ..ml.summarizer import ExtractiveSummarizer
from ..ml.training import DEFAULT_GRID, train_model

logger = logging.getLogger(__name__)


def resolve_dataset_path(kind: str) -> tuple[Path, str]:
    """auto: повний BBC, якщо він є, інакше демонстраційний набір."""
    if kind == "bbc" or (kind == "auto" and is_bbc_layout(config.BBC_DIR)):
        if not is_bbc_layout(config.BBC_DIR):
            raise FileNotFoundError(f"BBC News Summary не знайдено в {config.BBC_DIR}")
        return config.BBC_DIR, "bbc"
    if not is_bbc_layout(config.DEMO_DIR):
        raise FileNotFoundError(f"Демонстраційний набір не знайдено в {config.DEMO_DIR}")
    return config.DEMO_DIR, "demo"


class ModelRegistry:
    def __init__(self):
        self.model = ExtractiveSummarizer()  # без навчання — працює fallback
        self._lock = threading.Lock()
        self.started = time.time()
        self.job = {"state": "idle", "message": "", "progress": 0.0}

    def load_from_disk(self) -> bool:
        try:
            self.model = ExtractiveSummarizer.load(config.MODEL_FILE)
            logger.info("Модель завантажено: %s", config.MODEL_FILE)
            return True
        except FileNotFoundError:
            logger.warning("Файл моделі не знайдено (%s) — використовується TF-IDF fallback", config.MODEL_FILE)
        except Exception as exc:
            logger.error("Не вдалося завантажити модель: %s", exc)
        self.model = ExtractiveSummarizer()
        return False

    # ----- навчання у фоновому потоці
    def start_training(self, req) -> dict:
        with self._lock:
            if self.job["state"] == "running":
                raise RuntimeError("Навчання вже виконується")
            path, name = resolve_dataset_path(req.dataset)  # може кинути FileNotFoundError
            self.job = {"state": "running", "message": "Запуск", "progress": 0.0, "dataset": name,
                        "started_at": pd.Timestamp.now("UTC").isoformat(timespec="seconds")}
        threading.Thread(target=self._train, args=(req, path, name), daemon=True).start()
        return dict(self.job)

    def _progress(self, msg: str, frac: float):
        self.job.update(message=msg, progress=round(frac, 3))

    def _train(self, req, path: Path, name: str):
        try:
            df, _ = load_dataset(path, max_docs=req.max_documents)
            df, _ = validate_dataframe(df)
            splits = split_dataset(df, seed=req.seed)
            grid = {k: (getattr(req, k) or DEFAULT_GRID[k]) for k in DEFAULT_GRID}
            model = train_model(splits, name, grid=grid, seed=req.seed, progress=self._progress)
            if req.save:
                model.save(config.MODEL_FILE)
            model.classifier.n_jobs = 1
            self.model = model
            md = model.metadata
            best = max(md["grid_results"], key=lambda r: r["rougeL_f1"])
            self.job.update(state="done", message="Готово", progress=1.0,
                            finished_at=pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
                            result={"dataset": name, "split_sizes": md["split_sizes"], "best_params": md["best_params"],
                                    "validation": best, "train_seconds": md["train_seconds"],
                                    "label_stats": md["label_stats"], "saved": bool(req.save)})
        except Exception as exc:  # noqa: BLE001
            logger.exception("Помилка навчання")
            self.job.update(state="error", message="Помилка", error=str(exc),
                            finished_at=pd.Timestamp.now("UTC").isoformat(timespec="seconds"))


registry = ModelRegistry()


def read_json_artifact(name: str) -> dict | None:
    p = config.ARTIFACTS_DIR / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))
