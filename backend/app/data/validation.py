"""Перевірка якості даних: порожні, надто короткі записи, некоректні пари."""
from __future__ import annotations

import pandas as pd

from ..nlp.preprocessing import clean_text, count_words, split_sentences

MIN_TEXT_WORDS = 30
MIN_REF_WORDS = 5
MIN_SENTENCES = 3


def validate_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Повертає очищений DataFrame та звіт: скільки записів і чому відкинуто."""
    report = {"input_rows": int(len(df)), "dropped": {}}
    required = {"doc_id", "text", "reference"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Відсутні обов'язкові колонки: {sorted(missing)}")
    df = df.copy()
    df["text"] = df["text"].fillna("").map(clean_text)
    df["reference"] = df["reference"].fillna("").map(clean_text)

    reasons: dict[str, pd.Series] = {}
    reasons["empty_text"] = df["text"].str.len() == 0
    reasons["empty_reference"] = df["reference"].str.len() == 0
    tw = df["text"].map(count_words)
    rw = df["reference"].map(count_words)
    reasons["text_too_short"] = (tw < MIN_TEXT_WORDS) & ~reasons["empty_text"]
    reasons["reference_too_short"] = (rw < MIN_REF_WORDS) & ~reasons["empty_reference"]
    reasons["reference_longer_than_text"] = (rw >= tw) & (tw > 0)
    drop = pd.Series(False, index=df.index)
    for name, mask in reasons.items():
        newly = mask & ~drop
        report["dropped"][name] = int(newly.sum())
        drop |= mask
    df = df[~drop].copy()
    # документи, що містять менше MIN_SENTENCES речень
    few = df["text"].map(lambda t: len(split_sentences(t, "en")) < MIN_SENTENCES)
    report["dropped"]["too_few_sentences"] = int(few.sum())
    df = df[~few].reset_index(drop=True)
    report["output_rows"] = int(len(df))
    report["total_dropped"] = report["input_rows"] - report["output_rows"]
    return df, report
