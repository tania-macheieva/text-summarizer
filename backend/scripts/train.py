"""Навчання моделі й збереження в models/rf_summarizer.joblib.

Запуск:  python -m scripts.train [--dataset bbc|demo|auto] [--max-docs N]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from app import config
from app.data.loader import load_dataset
from app.data.splitter import split_dataset
from app.data.validation import validate_dataframe
from app.ml.training import DEFAULT_GRID, train_model
from app.services.registry import resolve_dataset_path


def main(argv=None):
    logging.basicConfig(level="INFO", format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="auto", choices=["auto", "demo", "bbc"])
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--out", default=str(config.MODEL_FILE))
    a = ap.parse_args(argv)
    path, name = resolve_dataset_path(a.dataset)
    df, _ = load_dataset(path, a.max_docs)
    df, _ = validate_dataframe(df)
    splits = split_dataset(df, seed=a.seed)
    print({k: len(v) for k, v in splits.items()})
    model = train_model(splits, name, grid=DEFAULT_GRID, seed=a.seed)
    model.save(a.out)
    md = model.metadata
    print(json.dumps({"best_params": md["best_params"], "label_stats": md["label_stats"],
                      "train_seconds": md["train_seconds"], "saved": a.out}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
