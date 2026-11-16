"""Завантаження датасетів: структура Articles/Summaries (BBC News Summary) та CSV."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..nlp.preprocessing import fix_missing_spaces

logger = logging.getLogger(__name__)

ARTICLES_DIR_NAMES = ("News Articles", "Articles", "News_Articles")
SUMMARIES_DIR_NAMES = ("Summaries", "Summary")
TEXT_COLUMNS = ("text", "article", "document", "body", "content", "story")
SUMMARY_COLUMNS = ("summary", "summaries", "highlights", "reference", "abstract")
CATEGORY_COLUMNS = ("category", "label", "topic", "source")

COLUMNS = ["doc_id", "category", "text", "reference"]


class DatasetError(ValueError):
    """Некоректна структура або вміст датасету."""


def read_text_file(path: Path, counter: dict | None = None) -> str:
    """Читає файл у UTF-8; якщо не вдається — cp1252 (у BBC News Summary є такі файли)."""
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        if counter is not None:
            counter["non_utf8_files"] = counter.get("non_utf8_files", 0) + 1
        return raw.decode("cp1252", errors="replace")


def _find_dir(root: Path, names: tuple[str, ...]) -> Path | None:
    for n in names:
        if (root / n).is_dir():
            return root / n
    return None


def is_bbc_layout(root: Path) -> bool:
    root = Path(root)
    return root.is_dir() and _find_dir(root, ARTICLES_DIR_NAMES) is not None and _find_dir(root, SUMMARIES_DIR_NAMES) is not None


def load_bbc_directory(root: Path, max_docs: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Читає <root>/News Articles/<category>/NNN.txt та <root>/Summaries/<category>/NNN.txt."""
    root = Path(root)
    art_dir = _find_dir(root, ARTICLES_DIR_NAMES)
    sum_dir = _find_dir(root, SUMMARIES_DIR_NAMES)
    if art_dir is None or sum_dir is None:
        raise DatasetError(f"У {root} не знайдено каталогів Articles/Summaries")
    info = {"non_utf8_files": 0, "articles_without_summary": 0, "summaries_without_article": 0}
    rows = []
    art_files = {(p.parent.name, p.name): p for p in sorted(art_dir.glob("*/*.txt"))}
    sum_files = {(p.parent.name, p.name): p for p in sorted(sum_dir.glob("*/*.txt"))}
    info["articles_without_summary"] = len(set(art_files) - set(sum_files))
    info["summaries_without_article"] = len(set(sum_files) - set(art_files))
    for key in sorted(set(art_files) & set(sum_files)):
        cat, name = key
        text = read_text_file(art_files[key], info)
        ref = read_text_file(sum_files[key], info)
        rows.append({"doc_id": f"{cat}/{name[:-4]}", "category": cat, "text": text,
                     "reference": fix_missing_spaces(ref)})
        if max_docs and len(rows) >= max_docs:
            break
    if not rows:
        raise DatasetError(f"У {root} не знайдено жодної пари стаття–резюме")
    return pd.DataFrame(rows, columns=COLUMNS), info


def _pick(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    low = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None


def load_csv(path: Path, text_column: str | None = None, summary_column: str | None = None,
             max_docs: int | None = None) -> tuple[pd.DataFrame, dict]:
    """CSV із колонками тексту та резюме (назви визначаються автоматично або задаються явно)."""
    path = Path(path)
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp1252")
    except Exception as exc:  # зіпсований CSV
        raise DatasetError(f"Не вдалося прочитати CSV: {exc}") from exc
    tcol = text_column or _pick(list(df.columns), TEXT_COLUMNS)
    scol = summary_column or _pick(list(df.columns), SUMMARY_COLUMNS)
    if tcol is None or scol is None or tcol not in df.columns or scol not in df.columns:
        raise DatasetError(
            f"Не знайдено колонки тексту/резюме. Наявні: {list(df.columns)}; "
            f"очікуються одна з {TEXT_COLUMNS} та одна з {SUMMARY_COLUMNS}"
        )
    ccol = _pick(list(df.columns), CATEGORY_COLUMNS)
    out = pd.DataFrame({
        "doc_id": [f"{path.stem}/{i}" for i in range(len(df))],
        "category": df[ccol].astype(str) if ccol else "unknown",
        "text": df[tcol].fillna("").astype(str),
        "reference": df[scol].fillna("").astype(str).map(fix_missing_spaces),
    })
    if max_docs:
        out = out.head(max_docs)
    return out, {"source_file": path.name, "text_column": tcol, "summary_column": scol}


def load_dataset(path: Path, max_docs: int | None = None) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    if path.is_dir():
        if is_bbc_layout(path):
            return load_bbc_directory(path, max_docs)
        csvs = sorted(path.glob("*.csv"))
        if csvs:
            return load_csv(csvs[0], max_docs=max_docs)
        raise DatasetError(f"Невідома структура датасету в {path}")
    if path.suffix.lower() == ".csv" and path.exists():
        return load_csv(path, max_docs=max_docs)
    raise DatasetError(f"Датасет не знайдено: {path}")
