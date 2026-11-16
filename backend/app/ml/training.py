"""Навчання: підготовка документів, greedy ROUGE-L розмітка, підбір гіперпараметрів RF на validation."""
from __future__ import annotations

import itertools
import logging
import platform
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

from ..nlp.preprocessing import Preprocessor, split_sentences
from .features import FEATURE_NAMES, FeatureExtractor, make_vectorizer
from .labeling import greedy_oracle
from .rouge import rouge_from_tokens, rouge_tokens
from .summarizer import ExtractiveSummarizer, num_to_select, select_top

logger = logging.getLogger(__name__)

DEFAULT_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [8, 16, None],
    "min_samples_leaf": [1, 5],
}


@dataclass
class PreparedDoc:
    doc_id: str
    sentences: list[str]
    tokens: list[list[str]]
    sent_rouge_tokens: list[list[str]]
    ref_tokens: list[str]
    feats: np.ndarray | None = None
    labels: np.ndarray | None = None


def prepare_docs(df: pd.DataFrame, language: str = "en") -> list[PreparedDoc]:
    pre = Preprocessor()
    docs = []
    for row in df.itertuples(index=False):
        p = pre.process(row.text, language)
        docs.append(PreparedDoc(
            doc_id=row.doc_id, sentences=p["sentences"], tokens=p["tokens"],
            sent_rouge_tokens=[rouge_tokens(s) for s in p["sentences"]],
            ref_tokens=rouge_tokens(row.reference),
        ))
    return docs


def attach_features_and_labels(docs: list[PreparedDoc], extractor: FeatureExtractor, with_labels: bool = True):
    for d in docs:
        d.feats, _ = extractor.extract(d.sentences, d.tokens)
        if with_labels:
            d.labels = greedy_oracle(d.sent_rouge_tokens, d.ref_tokens)
    return docs


def stack(docs: list[PreparedDoc]) -> tuple[np.ndarray, np.ndarray]:
    X = np.vstack([d.feats for d in docs])
    y = np.concatenate([d.labels for d in docs])
    return X, y


def mean_reference_ratio(df: pd.DataFrame) -> float:
    """Середня частка речень еталонного резюме відносно статті (на навчальній вибірці)."""
    pre = Preprocessor()
    ratios = []
    for row in df.itertuples(index=False):
        n = len(pre.process(row.text, "en")["sentences"])
        r = len(split_sentences(row.reference, "en"))
        if n:
            ratios.append(min(1.0, r / n))
    return float(np.mean(ratios))


def rouge_on_prepared(model, docs: list[PreparedDoc], ratio: float) -> dict[str, float]:
    """Середні ROUGE F1 для RF на підготовлених документах (швидко, без повторного видобування ознак)."""
    r1, r2, rl = [], [], []
    for d in docs:
        scores = model.predict_proba(d.feats)[:, 1]
        k = num_to_select(len(d.sentences), None, ratio)
        hyp = [t for i in select_top(scores, k) for t in d.sent_rouge_tokens[i]]
        s = rouge_from_tokens(d.ref_tokens, hyp)
        r1.append(s["rouge1"]["f1"]); r2.append(s["rouge2"]["f1"]); rl.append(s["rougeL"]["f1"])
    return {"rouge1_f1": float(np.mean(r1)), "rouge2_f1": float(np.mean(r2)), "rougeL_f1": float(np.mean(rl))}


def train_model(splits: dict[str, pd.DataFrame], dataset_name: str, grid: dict | None = None,
                seed: int = 42, progress: Callable[[str, float], None] | None = None) -> ExtractiveSummarizer:
    """Повний цикл навчання. Тест-вибірка тут НЕ використовується (лише train і val)."""
    from sklearn.ensemble import RandomForestClassifier

    def report(msg: str, frac: float):
        logger.info(msg)
        if progress:
            progress(msg, frac)

    t_start = time.perf_counter()
    train_df, val_df = splits["train"], splits["val"]
    report("Підготовка документів (очищення, речення, токени)", 0.05)
    train_docs, val_docs = prepare_docs(train_df), prepare_docs(val_df)

    report("Навчання TF-IDF на реченнях train", 0.2)
    vec = make_vectorizer(min_df=2)
    vec.fit([t for d in train_docs for t in d.tokens])
    extractor = FeatureExtractor(vec)

    report("Видобування ознак і greedy ROUGE-L розмітка", 0.3)
    attach_features_and_labels(train_docs, extractor)
    attach_features_and_labels(val_docs, extractor)
    Xtr, ytr = stack(train_docs)
    Xva, yva = stack(val_docs)
    ratio = round(mean_reference_ratio(train_df), 3)
    oracle_rl = float(np.mean([
        rouge_from_tokens(d.ref_tokens, [t for i in np.flatnonzero(d.labels) for t in d.sent_rouge_tokens[i]])["rougeL"]["f1"]
        for d in train_docs]))
    label_stats = {"train_sentences": int(len(ytr)), "train_positive": int(ytr.sum()),
                   "train_positive_rate": round(float(ytr.mean()), 4),
                   "val_sentences": int(len(yva)), "val_positive_rate": round(float(yva.mean()), 4),
                   "oracle_rougeL_f1_train": round(oracle_rl, 4)}

    grid = grid or DEFAULT_GRID
    keys = list(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]
    results, best, best_score = [], None, -1.0
    for i, params in enumerate(combos):
        report(f"Підбір гіперпараметрів RF {i + 1}/{len(combos)}: {params}", 0.4 + 0.5 * i / len(combos))
        clf = RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=seed, **params)
        clf.fit(Xtr, ytr)
        val_rouge = rouge_on_prepared(clf, val_docs, ratio)
        proba = clf.predict_proba(Xva)[:, 1]
        row = {**params, **{k: round(v, 4) for k, v in val_rouge.items()},
               "val_auc": round(float(roc_auc_score(yva, proba)), 4),
               "val_ap": round(float(average_precision_score(yva, proba)), 4)}
        results.append(row)
        if val_rouge["rougeL_f1"] > best_score:
            best_score, best = val_rouge["rougeL_f1"], (params, clf)
    params, clf = best
    importances = sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1])
    model = ExtractiveSummarizer(clf, vec, metadata={
        "algorithm": "RandomForestClassifier + TF-IDF",
        "dataset": dataset_name, "seed": seed,
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "ratio_used": ratio, "best_params": {k: v for k, v in params.items()},
        "selection_metric": "val ROUGE-L F1", "grid_results": results, "label_stats": label_stats,
        "feature_importances": [{"feature": k, "importance": round(float(v), 4)} for k, v in importances],
        "tfidf_vocabulary_size": int(len(vec.vocabulary_)),
        "train_seconds": round(time.perf_counter() - t_start, 1),
        "trained_at": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "numpy": np.__version__},
    })
    report("Навчання завершено", 1.0)
    return model
