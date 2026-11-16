"""Тести даних: завантаження, валідація, split без витоку, CSV, кодування."""
import pandas as pd
import pytest

from app.data.loader import DatasetError, load_bbc_directory, load_csv, load_dataset, read_text_file
from app.data.splitter import assert_no_leakage, duplicate_groups, split_dataset, text_hash
from app.data.stats import compute_stats
from app.data.validation import validate_dataframe
from app import config


def test_load_demo_structure():
    df, info = load_bbc_directory(config.DEMO_DIR)
    assert len(df) == 60 and set(df.columns) == {"doc_id", "category", "text", "reference"}
    assert df["category"].nunique() == 5 and info["articles_without_summary"] == 0


def test_reference_spaces_are_restored():
    df, _ = load_bbc_directory(config.DEMO_DIR)
    assert not df["reference"].str.contains(r"[a-z][.!?][A-Z]").any()


def test_non_utf8_fallback(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes("Price rose by £5".encode("cp1252"))
    info = {}
    assert "£5" in read_text_file(p, info) and info["non_utf8_files"] == 1


def test_load_csv_and_aliases(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"Article": ["a b c d. e f g."], "Highlights": ["a b."], "label": ["x"]}).to_csv(p, index=False)
    df, info = load_csv(p)
    assert len(df) == 1 and info["text_column"] == "Article" and df.loc[0, "category"] == "x"
    assert load_dataset(p)[0].shape[0] == 1


def test_load_csv_bad_columns_and_missing(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(p, index=False)
    with pytest.raises(DatasetError):
        load_csv(p)
    with pytest.raises(DatasetError):
        load_dataset(tmp_path / "missing")


def test_validation_drops_bad_rows():
    long = " ".join(f"word{i}" for i in range(80)) + ". " + "Another sentence here. " * 5
    df = pd.DataFrame({
        "doc_id": list("abcde"), "category": "x",
        "text": [long, "", "too short text.", long, long],
        "reference": ["word1 word2 word3 word4 word5 word6.", "x y z a b c d.", "q w e r t y u.", "", long + " more"],
    })
    clean, rep = validate_dataframe(df)
    assert list(clean["doc_id"]) == ["a"] and rep["total_dropped"] == 4
    assert rep["dropped"]["empty_text"] == 1 and rep["dropped"]["empty_reference"] == 1
    with pytest.raises(ValueError):
        validate_dataframe(pd.DataFrame({"x": [1]}))


def test_split_no_leakage_and_reproducible():
    df, _ = load_bbc_directory(config.DEMO_DIR)
    a, b = split_dataset(df, seed=42), split_dataset(df, seed=42)
    assert all(a[k]["doc_id"].tolist() == b[k]["doc_id"].tolist() for k in a)
    assert sum(len(v) for v in a.values()) == len(df)
    assert_no_leakage(a)
    c = split_dataset(df, seed=1)
    assert a["test"]["doc_id"].tolist() != c["test"]["doc_id"].tolist()


def test_duplicates_stay_in_one_split():
    """Кожен документ дублюється; без групування пари неминуче розійшлися б між вибірками."""
    df, _ = load_bbc_directory(config.DEMO_DIR)
    dup = df.copy()
    dup["doc_id"] = ["dup/" + d for d in df["doc_id"]]
    dup["text"] = dup["text"] + "  "  # відрізняється лише пробілами
    full = pd.concat([df, dup], ignore_index=True)
    splits = split_dataset(full, seed=3)
    where = {d: k for k, v in splits.items() for d in v["doc_id"]}
    assert all(where[d] == where["dup/" + d] for d in df["doc_id"])
    assert_no_leakage(splits)


def test_leakage_detector_fires():
    df, _ = load_bbc_directory(config.DEMO_DIR)
    with pytest.raises(AssertionError):
        assert_no_leakage({"train": df.head(10), "test": df.head(5)})
    t = df.iloc[:1].copy(); t2 = t.copy(); t2["doc_id"] = "other"
    with pytest.raises(AssertionError):
        assert_no_leakage({"train": t, "test": t2})
    assert text_hash(" A  b ") == text_hash("a b")


def test_stats_from_real_data():
    df, _ = load_bbc_directory(config.DEMO_DIR)
    st = compute_stats(df, split_dataset(df))
    assert st["documents"] == 60 and sum(st["categories"].values()) == 60
    assert 0 < st["compression_words_mean"] < 1 and st["splits"]["train"]["documents"] > 30
