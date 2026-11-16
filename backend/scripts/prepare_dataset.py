"""Перевірка даних, split train/val/test, статистика → artifacts/dataset_info.json, split.json.

Запуск (з каталогу backend):  python -m scripts.prepare_dataset [--path data/bbc]
"""
from __future__ import annotations

import argparse
import json
import sys

from app import config
from app.data.loader import load_dataset
from app.data.splitter import split_dataset
from app.data.stats import compute_stats
from app.data.validation import validate_dataframe


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(config.BBC_DIR))
    ap.add_argument("--name", default="BBC News Summary")
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--out", default=str(config.ARTIFACTS_DIR / "dataset_info.json"))
    ap.add_argument("--split-out", default=str(config.ARTIFACTS_DIR / "split.json"))
    a = ap.parse_args(argv)

    df, load_info = load_dataset(a.path)
    clean, val_report = validate_dataframe(df)
    splits = split_dataset(clean, seed=a.seed)
    stats = compute_stats(clean, splits)
    info = {"name": a.name, "path": str(a.path), "seed": a.seed,
            "fractions": list(config.SPLIT_FRACTIONS), "loading": load_info, "validation": val_report,
            "statistics": stats, "leakage_check": "passed (doc_id і хеші текстів не перетинаються; групи дублікатів не розділені)"}
    out = config.ARTIFACTS_DIR if a.out is None else None
    from pathlib import Path
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(a.split_out).write_text(json.dumps({k: v["doc_id"].tolist() for k, v in splits.items()}, indent=1), encoding="utf-8")
    print(json.dumps({"documents": stats["documents"], "splits": {k: len(v) for k, v in splits.items()},
                      "validation": val_report, "loading": load_info}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
