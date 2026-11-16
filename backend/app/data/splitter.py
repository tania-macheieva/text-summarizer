"""Розділення train/validation/test без витоку: дублікати та майже-дублікати потрапляють в одну вибірку."""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..config import RANDOM_SEED, SPLIT_FRACTIONS

NEAR_DUP_THRESHOLD = 0.9


def text_hash(text: str) -> str:
    norm = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


def duplicate_groups(df: pd.DataFrame, near_threshold: float = NEAR_DUP_THRESHOLD) -> np.ndarray:
    """Групові мітки: точні дублікати (хеш) та майже-дублікати (косинус TF-IDF ≥ threshold) — разом."""
    n = len(df)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    first: dict[str, int] = {}
    for i, t in enumerate(df["text"]):
        h = text_hash(t)
        if h in first:
            union(first[h], i)
        else:
            first[h] = i
    if n > 1 and near_threshold < 1.0:
        mat = TfidfVectorizer(stop_words="english", min_df=1, sublinear_tf=True).fit_transform(df["text"])
        step = 500
        for s in range(0, n, step):
            sim = cosine_similarity(mat[s : s + step], mat)
            ii, jj = np.where(sim >= near_threshold)
            for a, b in zip(ii + s, jj):
                if a < b:
                    union(int(a), int(b))
    return np.array([find(i) for i in range(n)])


def split_dataset(df: pd.DataFrame, seed: int = RANDOM_SEED,
                  fractions: tuple[float, float, float] = SPLIT_FRACTIONS) -> dict[str, pd.DataFrame]:
    """Розділення за групами дублікатів зі стратифікацією за категорією (де вона є) і фіксованим seed."""
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("Сума часток train/val/test має дорівнювати 1")
    df = df.reset_index(drop=True).copy()
    df["_group"] = duplicate_groups(df)
    rng = np.random.RandomState(seed)
    assign = {}
    # представник групи → категорія першого елемента
    groups = df.groupby("_group").agg(cat=("category", "first"), size=("doc_id", "size")).reset_index()
    for _, sub in groups.groupby("cat"):
        gids = sub["_group"].values.copy()
        sizes = dict(zip(sub["_group"], sub["size"]))
        rng.shuffle(gids)
        total = sum(sizes[g] for g in gids)
        targets = [fractions[0] * total, (fractions[0] + fractions[1]) * total]
        acc = 0
        for g in gids:
            name = "train" if acc < targets[0] else "val" if acc < targets[1] else "test"
            assign[g] = name
            acc += sizes[g]
    df["_split"] = df["_group"].map(assign)
    out = {k: df[df["_split"] == k].drop(columns=["_group", "_split"]).reset_index(drop=True)
           for k in ("train", "val", "test")}
    assert_no_leakage(out)
    return out


def assert_no_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """Перевіряє, що doc_id та тексти (за хешем) не перетинаються між вибірками."""
    names = list(splits)
    hashes = {k: set(splits[k]["text"].map(text_hash)) for k in names}
    ids = {k: set(splits[k]["doc_id"]) for k in names}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if ids[a] & ids[b]:
                raise AssertionError(f"Витік: однакові doc_id у {a} та {b}")
            if hashes[a] & hashes[b]:
                raise AssertionError(f"Витік: однакові тексти у {a} та {b}")
