"""Статистика датасету — лише з реальних даних."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..nlp.preprocessing import clean_text, count_words, split_sentences, split_title


def _n_sent(text: str) -> int:
    return len(split_sentences(split_title(clean_text(text))[1], "en"))
from .splitter import duplicate_groups, text_hash


def _desc(values) -> dict:
    a = np.asarray(list(values), dtype=float)
    if a.size == 0:
        return {"mean": 0, "median": 0, "min": 0, "max": 0, "std": 0}
    return {"mean": round(float(a.mean()), 2), "median": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max()), "std": round(float(a.std()), 2)}


def compute_stats(df: pd.DataFrame, splits: dict[str, pd.DataFrame] | None = None) -> dict:
    tw = df["text"].map(count_words)
    rw = df["reference"].map(count_words)
    ns = df["text"].map(_n_sent)
    rs = df["reference"].map(_n_sent)
    exact = int(len(df) - df["text"].map(text_hash).nunique())
    groups = duplicate_groups(df)
    stats = {
        "documents": int(len(df)),
        "categories": {k: int(v) for k, v in df["category"].value_counts().sort_index().items()},
        "text_words": _desc(tw),
        "reference_words": _desc(rw),
        "text_sentences": _desc(ns),
        "reference_sentences": _desc(rs),
        "compression_words_mean": round(float((rw / tw).mean()), 4),
        "compression_sentences_mean": round(float((rs / ns).mean()), 4),
        "exact_duplicate_texts": exact,
        "documents_in_duplicate_groups": int(len(df) - len(set(groups))),
    }
    if splits:
        stats["splits"] = {k: {"documents": int(len(v)),
                               "categories": {c: int(n) for c, n in v["category"].value_counts().sort_index().items()}}
                           for k, v in splits.items()}
    return stats
