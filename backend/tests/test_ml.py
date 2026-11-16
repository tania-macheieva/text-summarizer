"""Тести ML-ядра: препроцесинг, TF-IDF, ознаки, розмітка, Random Forest, збереження, fallback, ROUGE."""
import numpy as np
import pytest
from rouge_score import rouge_scorer

from app.ml import rouge
from app.ml.features import FEATURE_NAMES, FeatureExtractor, make_vectorizer
from app.ml.labeling import greedy_oracle
from app.ml.summarizer import (ExtractiveSummarizer, SummarizerError, num_to_select, select_top)
from app.ml.training import attach_features_and_labels, prepare_docs, stack
from app.nlp.preprocessing import Preprocessor, clean_text, count_words, split_sentences, tokenize
from tests.conftest import SAMPLE_EN, SAMPLE_UK


# ------------------------------------------------------------------ препроцесинг
def test_clean_text_removes_control_chars_and_keeps_paragraphs():
    out = clean_text("Hello\x00  world\u200b.\r\n\r\n\r\n\r\nNext   para.")
    assert out == "Hello world.\n\nNext para."


def test_sentence_split_nltk_handles_abbreviations():
    s = split_sentences("Mr. Smith went to Washington. He met Dr. Jones there. They talked.", "en")
    assert len(s) == 3


def test_sentence_split_ukrainian_abbreviations():
    s = split_sentences("Це відбулося у т.д. випадках. Міністр І. Петренко заявив про це. Чи правда?", "uk")
    assert len(s) == 3


def test_title_is_separated():
    doc = Preprocessor().process(SAMPLE_EN)
    assert doc["title"] == "Labour plans maternity pay rise" and doc["language"] == "en"
    assert doc["sentences"][0].startswith("Maternity pay")


def test_language_detection():
    assert Preprocessor().process(SAMPLE_UK)["language"] == "uk"
    assert Preprocessor().process(SAMPLE_EN)["language"] == "en"


def test_tokenize_removes_stopwords_and_stems():
    assert tokenize("The cats are running quickly", "en") == ["cat", "run", "quickli"]
    assert "і" not in tokenize("Кіт і собака бігають", "uk")


# ------------------------------------------------------------------ TF-IDF та ознаки
def test_tfidf_vectorizer_idf_downweights_common_terms():
    docs = [["cat", "dog"], ["cat", "fish"], ["cat", "bird"], ["rare", "cat"]]
    vec = make_vectorizer().fit(docs)
    idf = dict(zip(vec.get_feature_names_out(), vec.idf_))
    assert idf["cat"] < idf["rare"]
    m = vec.transform([["cat", "rare"]]).toarray()[0]
    names = list(vec.get_feature_names_out())
    assert m[names.index("rare")] > m[names.index("cat")] > 0


def test_feature_extractor_shape_and_ranges():
    doc = Preprocessor().process(SAMPLE_EN)
    feats, mode = FeatureExtractor().extract(doc["sentences"], doc["tokens"])
    assert feats.shape == (len(doc["sentences"]), len(FEATURE_NAMES))
    assert np.isfinite(feats).all() and mode == "local"
    f = dict(zip(FEATURE_NAMES, feats.T))
    assert f["pos_rel"][0] == 0 and f["pos_rel"][-1] == 1 and f["is_first"][0] == 1
    assert ((f["tfidf_rank"] > 0) & (f["tfidf_rank"] <= 1)).all()
    assert ((f["sim_doc"] >= -1e-9) & (f["sim_doc"] <= 1 + 1e-9)).all()
    assert abs(f["centrality_pr"].mean() - 1.0) < 1e-6


def test_feature_extractor_single_sentence_and_empty_tokens():
    feats, _ = FeatureExtractor().extract(["Only one."], [["one"]])
    assert feats.shape[0] == 1 and np.isfinite(feats).all()
    feats, _ = FeatureExtractor().extract(["the the"], [[]])
    assert np.isfinite(feats).all()


# ------------------------------------------------------------------ розмітка
def test_greedy_oracle_is_deterministic_and_meaningful():
    sents = [["a", "b", "c"], ["x", "y"], ["d", "e", "f"], ["q", "w"]]
    ref = ["a", "b", "c", "d", "e", "f"]
    l1, l2 = greedy_oracle(sents, ref), greedy_oracle(sents, ref)
    assert (l1 == l2).all() and l1.tolist() == [1, 0, 1, 0]


def test_oracle_labels_on_real_pairs(demo_splits):
    docs = attach_features_and_labels(prepare_docs(demo_splits["train"].head(10)), FeatureExtractor())
    X, y = stack(docs)
    assert 0 < y.mean() < 0.7 and X.shape[0] == y.shape[0]


# ------------------------------------------------------------------ вибір речень
def test_select_top_keeps_original_order():
    assert select_top(np.array([0.1, 0.9, 0.3, 0.8, 0.2]), 2) == [1, 3]


def test_num_to_select():
    assert num_to_select(10, None, 0.3) == 3
    assert num_to_select(10, 50, None) == 10
    assert num_to_select(3, None, 0.01) == 1
    with pytest.raises(SummarizerError):
        num_to_select(10, 2, 0.3)
    with pytest.raises(SummarizerError):
        num_to_select(10, 0, None)


# ------------------------------------------------------------------ Random Forest
def test_random_forest_trains_and_predicts(trained_model):
    assert trained_model.is_trained
    doc = Preprocessor().process(SAMPLE_EN)
    feats, _ = trained_model.extractor().extract(doc["sentences"], doc["tokens"])
    scores, name, fallback = trained_model.score_features(feats)
    assert name == "random_forest" and not fallback
    assert scores.shape == (len(doc["sentences"]),) and ((scores >= 0) & (scores <= 1)).all()


def test_training_is_reproducible(demo_splits):
    from app.ml.training import train_model
    grid = {"n_estimators": [20], "max_depth": [4], "min_samples_leaf": [2]}
    a = train_model(demo_splits, "demo", grid=grid, seed=7)
    b = train_model(demo_splits, "demo", grid=grid, seed=7)
    X = np.random.RandomState(0).rand(20, len(FEATURE_NAMES))
    assert np.allclose(a.classifier.predict_proba(X), b.classifier.predict_proba(X))


def test_fit_requires_both_classes():
    with pytest.raises(SummarizerError):
        ExtractiveSummarizer().fit(np.zeros((5, len(FEATURE_NAMES))), np.zeros(5))


def test_save_and_load_roundtrip(trained_model, tmp_path):
    path = trained_model.save(tmp_path / "m.joblib")
    loaded = ExtractiveSummarizer.load(path)
    a = trained_model.summarize(SAMPLE_EN, ratio=0.4)
    b = loaded.summarize(SAMPLE_EN, ratio=0.4)
    assert a.summary == b.summary and np.allclose(a.scores, b.scores)
    assert loaded.metadata["dataset"] == "demo"


def test_load_missing_and_save_untrained(tmp_path):
    with pytest.raises(FileNotFoundError):
        ExtractiveSummarizer.load(tmp_path / "none.joblib")
    with pytest.raises(SummarizerError):
        ExtractiveSummarizer().save(tmp_path / "x.joblib")


# ------------------------------------------------------------------ fallback
def test_fallback_without_trained_model():
    res = ExtractiveSummarizer().summarize(SAMPLE_EN, num_sentences=3)
    assert res.used_fallback and "fallback" in res.model_name and res.warnings
    assert len(res.selected_indices) == 3 and res.summary


def test_fallback_prefers_central_sentences():
    text = ("Solar panels convert sunlight into electricity. Solar panels are cheap now. "
            "My cat sleeps all day. Solar energy is growing as panels get cheaper and electricity demand rises. "
            "The weather was nice yesterday.")
    res = ExtractiveSummarizer().summarize(text, num_sentences=2)
    assert all("olar" in res.sentences[i] for i in res.selected_indices)


def test_summarize_errors():
    m = ExtractiveSummarizer()
    for bad in ("", "   ", "... !!!"):
        with pytest.raises(SummarizerError):
            m.summarize(bad)
    with pytest.raises(SummarizerError):
        m.summarize(SAMPLE_EN, language="fr")


def test_word_ratio_gives_comparable_length(trained_model):
    short = trained_model.summarize(SAMPLE_EN, word_ratio=0.3)
    long_ = trained_model.summarize(SAMPLE_EN, word_ratio=0.7)
    assert short.summary_word_count < long_.summary_word_count


# ------------------------------------------------------------------ ROUGE
def test_bit_lcs_matches_dp():
    rng = np.random.RandomState(0)
    for _ in range(200):
        a = [str(x) for x in rng.randint(0, 8, rng.randint(0, 30))]
        b = [str(x) for x in rng.randint(0, 8, rng.randint(0, 30))]
        assert rouge.lcs_length(a, b) == rouge.lcs_length_dp(a, b)


def test_rouge_matches_reference_package():
    ref = ("The Tories dismissed the maternity pay plan as desperate, while the Liberal Democrats said it was "
           "misdirected. Ministers will consult on other proposals.")
    hyp = "Maternity pay would rise to nine months; the Tories dismissed the plan as desperate."
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True).score(ref, hyp)
    mine = rouge.rouge_scores(ref, hyp)
    for k in ("rouge1", "rouge2", "rougeL"):
        for m, attr in (("precision", "precision"), ("recall", "recall"), ("f1", "fmeasure")):
            assert mine[k][m] == pytest.approx(getattr(sc[k], attr), abs=1e-9)


def test_rouge_edge_cases():
    assert rouge.rouge_scores("same text here", "same text here")["rougeL"]["f1"] == 1.0
    assert rouge.rouge_scores("abc def", "xyz")["rouge1"]["f1"] == 0.0
    assert rouge.rouge_scores("abc", "")["rouge1"]["precision"] == 0.0
    assert rouge.rouge_scores("Привіт світ", "привіт світ")["rouge1"]["f1"] == 1.0
