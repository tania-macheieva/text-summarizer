"""Порівняння алгоритмів на однаковій тестовій вибірці: ROUGE P/R/F, sentence-level P/R/F, стиснення, час."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd

from .labeling import greedy_oracle
from .rouge import rouge_from_tokens, rouge_tokens
from .summarizer import ExtractiveSummarizer

METHODS = {
    "random": "Випадковий вибір (контроль)",
    "lead": "Lead-N (baseline)",
    "tfidf": "TF-IDF ranking (baseline)",
    "tfidf_centrality": "TF-IDF + центральність (fallback)",
    "random_forest": "Random Forest",
}
ROUGE_KEYS = ("rouge1", "rouge2", "rougeL")


def _sentence_prf(pred: set[int], gold: set[int]) -> tuple[float, float, float]:
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def evaluate_method(model: ExtractiveSummarizer, df: pd.DataFrame, algorithm: str, ratio: float | None,
                    word_ratio: float | None = None) -> pd.DataFrame:
    """Повертає таблицю метрик по документах. Час — повний шлях model.summarize() (очищення → резюме)."""
    rows = []
    for row in df.itertuples(index=False):
        t0 = time.perf_counter()
        res = model.summarize(row.text, "en", ratio=ratio, algorithm=algorithm, word_ratio=word_ratio)
        elapsed = time.perf_counter() - t0
        ref_tok = rouge_tokens(row.reference)
        sc = rouge_from_tokens(ref_tok, rouge_tokens(res.summary))
        sent_tok = [rouge_tokens(s) for s in res.sentences]
        gold = set(np.flatnonzero(greedy_oracle(sent_tok, ref_tok)).tolist())
        sp, sr, sf = _sentence_prf(set(res.selected_indices), gold)
        rec = {"doc_id": row.doc_id, "time_ms": elapsed * 1000, "compression_words": res.compression_ratio,
               "sentences": len(res.sentences), "selected": len(res.selected_indices),
               "sent_precision": sp, "sent_recall": sr, "sent_f1": sf}
        for k in ROUGE_KEYS:
            for m in ("precision", "recall", "f1"):
                rec[f"{k}_{m}"] = sc[k][m]
        rows.append(rec)
    return pd.DataFrame(rows)


def evaluate_oracle(model: ExtractiveSummarizer, df: pd.DataFrame) -> pd.DataFrame:
    """Верхня межа екстрактивного підходу: речення, обрані greedy-oracle за еталоном (використовує reference!)."""
    rows = []
    for row in df.itertuples(index=False):
        res = model.summarize(row.text, "en", ratio=1.0, algorithm="lead")
        sent_tok = [rouge_tokens(s) for s in res.sentences]
        ref_tok = rouge_tokens(row.reference)
        idx = np.flatnonzero(greedy_oracle(sent_tok, ref_tok))
        hyp = [t for i in idx for t in sent_tok[i]]
        sc = rouge_from_tokens(ref_tok, hyp)
        total_words = sum(len(t) for t in sent_tok)
        rec = {"doc_id": row.doc_id, "compression_words": len(hyp) / max(1, total_words)}
        for k in ROUGE_KEYS:
            for m in ("precision", "recall", "f1"):
                rec[f"{k}_{m}"] = sc[k][m]
        rows.append(rec)
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> dict:
    out = {"documents": int(len(df))}
    for c in df.columns:
        if c == "doc_id":
            continue
        out[c] = round(float(df[c].mean()), 4)
        if c.endswith("_f1") or c == "time_ms":
            out[c + "_std"] = round(float(df[c].std(ddof=1)), 4)
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 42) -> dict:
    """Парний bootstrap різниці середніх (a − b): середнє, 95% ДІ, частка вибірок із різницею > 0."""
    diff = np.asarray(a) - np.asarray(b)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": round(float(diff.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "p_diff_positive": round(float((means > 0).mean()), 4),
            "significant": bool(lo > 0 or hi < 0),
            "wins": int((diff > 0).sum()), "ties": int((diff == 0).sum()), "losses": int((diff < 0).sum())}


def run_comparison(model: ExtractiveSummarizer, test_df: pd.DataFrame, ratio: float,
                   sweep_ratios: list[float] | None = None, word_ratio: float | None = None,
                   progress: Callable[[str], None] | None = None) -> dict:
    log = progress or (lambda m: None)
    per_doc: dict[str, pd.DataFrame] = {}
    methods = {}
    for key in METHODS:
        log(f"Оцінювання: {key}")
        per_doc[key] = evaluate_method(model, test_df, key, ratio)
        methods[key] = {"label": METHODS[key], **aggregate(per_doc[key])}
    log("Оцінювання: oracle")
    oracle = evaluate_oracle(model, test_df)
    oracle_agg = {"label": "Oracle greedy ROUGE-L (орієнтир, використовує еталон)", **aggregate(oracle)}

    tests = {}
    rf = per_doc["random_forest"]
    for base in ("random", "lead", "tfidf", "tfidf_centrality"):
        tests[f"random_forest_vs_{base}"] = {
            m: paired_bootstrap(rf[f"{m}_f1"].values, per_doc[base][f"{m}_f1"].values) for m in ROUGE_KEYS}

    sweep = []
    for r in (sweep_ratios or []):
        entry = {"ratio": r}
        for key in METHODS:
            d = per_doc[key] if abs(r - ratio) < 1e-9 else evaluate_method(model, test_df, key, r)
            entry[key] = {k + "_f1": round(float(d[k + "_f1"].mean()), 4) for k in ROUGE_KEYS}
            entry[key]["compression_words"] = round(float(d["compression_words"].mean()), 4)
        sweep.append(entry)
    equal_words = None
    if word_ratio:
        equal_words = {"word_ratio": round(word_ratio, 4), "methods": {}, "paired_bootstrap": {}}
        docs_w = {}
        for key in METHODS:
            log(f"Оцінювання (однаковий бюджет слів): {key}")
            docs_w[key] = evaluate_method(model, test_df, key, None, word_ratio=word_ratio)
            equal_words["methods"][key] = {"label": METHODS[key], **aggregate(docs_w[key])}
        for base in ("random", "lead", "tfidf", "tfidf_centrality"):
            equal_words["paired_bootstrap"][f"random_forest_vs_{base}"] = {
                m: paired_bootstrap(docs_w["random_forest"][f"{m}_f1"].values, docs_w[base][f"{m}_f1"].values)
                for m in ROUGE_KEYS}
    return {"test_documents": int(len(test_df)), "ratio": ratio, "methods": methods,
            "oracle": oracle_agg, "paired_bootstrap": tests, "ratio_sweep": sweep, "equal_word_budget": equal_words}
