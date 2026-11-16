"""Порівняння алгоритмів на тестовій вибірці → artifacts/evaluation.json, per_document_metrics.csv, графіки.

Запуск:  python -m scripts.evaluate [--dataset bbc]
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from pathlib import Path

import pandas as pd

from app import config
from app.data.loader import load_dataset
from app.data.splitter import split_dataset
from app.data.validation import validate_dataframe
from app.ml.evaluation import ROUGE_KEYS, run_comparison
from app.ml.summarizer import ExtractiveSummarizer
from app.nlp.preprocessing import count_words
from app.services.registry import resolve_dataset_path


def make_charts(res: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(res["methods"])
    labels = {"random": "Random", "lead": "Lead-N", "tfidf": "TF-IDF", "tfidf_centrality": "TF-IDF+\nцентральність", "random_forest": "Random\nForest"}
    colors = ["#d1d5db", "#9ca3af", "#60a5fa", "#fbbf24", "#16a34a"]
    # 1) ROUGE F1
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(ROUGE_KEYS)); w = 0.16
    for i, k in enumerate(keys):
        vals = [res["methods"][k][f"{r}_f1"] for r in ROUGE_KEYS]
        bars = ax.bar(x + (i - 2) * w, vals, w, label=labels[k].replace("\n", " "), color=colors[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x, ["ROUGE-1", "ROUGE-2", "ROUGE-L"]); ax.set_ylabel("F1"); ax.set_ylim(0, 0.9)
    ax.set_title("ROUGE F1 на тестовій вибірці"); ax.legend(fontsize=8, ncol=3); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(out_dir / "rouge_f1_comparison.png", dpi=200); plt.close(fig)
    # 2) Precision / Recall / F1 для ROUGE-1 і ROUGE-L
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, r, title in zip(axes, ROUGE_KEYS, ["ROUGE-1", "ROUGE-2", "ROUGE-L"]):
        x = np.arange(3)
        for i, k in enumerate(keys):
            vals = [res["methods"][k][f"{r}_{m}"] for m in ("precision", "recall", "f1")]
            ax.bar(x + (i - 2) * w, vals, w, color=colors[i], label=labels[k].replace("\n", " "))
        ax.set_xticks(x, ["P", "R", "F1"]); ax.set_title(title); ax.grid(axis="y", alpha=.3)
    axes[0].set_ylabel("значення"); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / "rouge_prf.png", dpi=200); plt.close(fig)
    # 3) Залежність від частки стиснення
    if res.get("ratio_sweep"):
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        rs = [e["ratio"] for e in res["ratio_sweep"]]
        for i, k in enumerate(keys):
            ax.plot(rs, [e[k]["rougeL_f1"] for e in res["ratio_sweep"]], marker="o", color=colors[i], label=labels[k].replace("\n", " "))
        ax.set_xlabel("частка речень оригіналу в резюме"); ax.set_ylabel("ROUGE-L F1"); ax.grid(alpha=.3); ax.legend(fontsize=8)
        ax.set_title("ROUGE-L F1 залежно від довжини резюме"); fig.tight_layout()
        fig.savefig(out_dir / "rougeL_vs_ratio.png", dpi=200); plt.close(fig)
    # 4) Важливість ознак
    fi = res["model"]["feature_importances"][:12][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh([f["feature"] for f in fi], [f["importance"] for f in fi], color="#16a34a")
    ax.set_xlabel("важливість (Gini)"); ax.set_title("Важливість ознак Random Forest (топ-12)"); ax.grid(axis="x", alpha=.3)
    fig.tight_layout(); fig.savefig(out_dir / "feature_importance.png", dpi=200); plt.close(fig)


def main(argv=None):
    logging.basicConfig(level="WARNING")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="auto", choices=["auto", "demo", "bbc"])
    ap.add_argument("--model", default=str(config.MODEL_FILE))
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--out", default=str(config.ARTIFACTS_DIR))
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model = ExtractiveSummarizer.load(Path(a.model))
    path, name = resolve_dataset_path(a.dataset)
    df, _ = load_dataset(path)
    df, _ = validate_dataframe(df)
    splits = split_dataset(df, seed=a.seed)
    test, train = splits["test"], splits["train"]
    ratio = model.metadata["ratio_used"]
    word_ratio = float((train["reference"].map(count_words) / train["text"].map(count_words)).mean())  # лише train
    res = run_comparison(model, test, ratio, sweep_ratios=sorted({0.2, 0.3, ratio, 0.5}),
                         word_ratio=word_ratio, progress=print)
    md = model.metadata
    res["dataset"] = name
    res["model"] = {k: md[k] for k in ("algorithm", "dataset", "seed", "best_params", "split_sizes", "ratio_used",
                                       "grid_results", "label_stats", "feature_importances", "train_seconds",
                                       "trained_at", "versions", "tfidf_vocabulary_size")}
    res["environment"] = {"python": platform.python_version(), "platform": platform.platform(),
                          "inference_n_jobs": 1, "timing": "повний шлях summarize(): очищення → речення → ознаки → оцінка → резюме"}
    res["generated_at"] = pd.Timestamp.now("UTC").isoformat(timespec="seconds")
    (out / "evaluation.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    make_charts(res, out / "charts")
    print(json.dumps({k: {m: v[m] for m in ("rouge1_f1", "rouge2_f1", "rougeL_f1", "time_ms")} for k, v in res["methods"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
