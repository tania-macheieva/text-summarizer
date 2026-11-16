"""Формує невеликий демонстраційний набір (n документів, стратифіковано за категоріями) зі структури BBC.

Запуск:  python -m scripts.build_demo_data --src data/bbc --dst data/demo --per-category 12
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from app.data.loader import load_dataset
from app.data.splitter import duplicate_groups
from app.data.validation import validate_dataframe


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/bbc")
    ap.add_argument("--dst", default="data/demo")
    ap.add_argument("--per-category", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    df, _ = load_dataset(Path(a.src))
    df, _ = validate_dataframe(df)
    df["_g"] = duplicate_groups(df)
    df = df.drop_duplicates("_g")  # демо-набір без дублікатів
    rng = np.random.RandomState(a.seed)
    dst = Path(a.dst)
    if dst.exists():
        shutil.rmtree(dst)
    picked = []
    for cat, sub in df.groupby("category"):
        idx = rng.permutation(len(sub))[: a.per_category]
        picked.append(sub.iloc[sorted(idx)])
    src = Path(a.src)
    n = 0
    for sub in picked:
        for doc_id in sub["doc_id"]:
            cat, name = doc_id.split("/")
            for folder in ("News Articles", "Summaries"):
                (dst / folder / cat).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src / folder / cat / f"{name}.txt", dst / folder / cat / f"{name}.txt")
            n += 1
    (dst / "README.md").write_text(
        "# Демонстраційний набір\n\n"
        f"{n} документів ({a.per_category} на категорію), випадково (seed={a.seed}) вибрані зі структури "
        "BBC News Summary (News Articles / Summaries). Призначення — запуск системи й швидке навчання без "
        "завантаження повного датасету. Для повноцінних експериментів використовуйте повний набір "
        "(https://www.kaggle.com/datasets/pariza/bbc-news-summary) у `backend/data/bbc/`.\n\n"
        "Авторські права на тексти статей належать BBC; дані використовуються лише в навчальних цілях "
        "(D. Greene, P. Cunningham, ICML 2006).\n", encoding="utf-8")
    print(f"Демо-набір: {n} документів → {dst}")


if __name__ == "__main__":
    main()
