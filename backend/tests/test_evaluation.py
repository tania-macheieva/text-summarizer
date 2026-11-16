"""Тести модуля порівняння моделей."""
import numpy as np

from app.ml.evaluation import evaluate_method, paired_bootstrap, run_comparison, aggregate


def test_paired_bootstrap_detects_difference():
    rng = np.random.RandomState(0)
    a = rng.rand(100) + 0.3
    b = a - 0.3 + rng.rand(100) * 0.01
    r = paired_bootstrap(a, b, n_boot=500)
    assert r["significant"] and r["ci95"][0] > 0.2
    same = paired_bootstrap(a, a.copy(), n_boot=200)
    assert same["mean_diff"] == 0 and not same["significant"]


def test_evaluate_method_columns(trained_model, demo_splits):
    df = evaluate_method(trained_model, demo_splits["test"], "random_forest", 0.4)
    for c in ("rouge1_precision", "rouge1_recall", "rouge1_f1", "rougeL_f1", "compression_words", "time_ms", "sent_f1"):
        assert c in df.columns
    assert ((df["rouge1_f1"] >= 0) & (df["rouge1_f1"] <= 1)).all() and (df["time_ms"] > 0).all()
    agg = aggregate(df)
    assert agg["documents"] == len(demo_splits["test"])


def test_run_comparison_structure(trained_model, demo_splits):
    res = run_comparison(trained_model, demo_splits["test"], 0.4, sweep_ratios=[0.3], word_ratio=0.45)
    assert {"random", "lead", "tfidf", "tfidf_centrality", "random_forest"} <= set(res["methods"])
    assert res["equal_word_budget"]["methods"]["lead"]["rouge1_f1"] >= 0
    assert "random_forest_vs_lead" in res["paired_bootstrap"] and res["ratio_sweep"][0]["ratio"] == 0.3
