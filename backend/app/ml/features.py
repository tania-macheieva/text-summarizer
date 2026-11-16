"""Видобування ознак речень: TF-IDF, позиція, довжина, ключові слова, центральність, схожість з документом."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ..nlp.preprocessing import count_words, has_number, has_quote

FEATURE_NAMES = [
    "tfidf_mean",        # середня TF-IDF вага термів речення
    "tfidf_sum_rel",     # сума TF-IDF / максимум у документі
    "tfidf_rank",        # перцентильний ранг за TF-IDF у документі
    "pos_rel",           # відносна позиція (0 — початок, 1 — кінець)
    "pos_inv",           # 1/(i+1): виграш для перших речень
    "is_first",          # перше речення
    "len_words",         # довжина у словах
    "len_rel",           # довжина відносно середньої по документу
    "keyword_count",     # кількість ключових слів документа в реченні
    "keyword_ratio",     # частка термів речення, що є ключовими
    "centrality_mean",   # середня косинусна схожість з іншими реченнями
    "centrality_pr",     # TextRank (PageRank по графу схожості)
    "centrality_rank",   # ранг за PageRank
    "sim_doc",           # косинусна схожість із вектором документа
    "sim_doc_rank",      # ранг за схожістю з документом
    "has_number",        # наявність цифр
    "has_quote",         # наявність прямої мови
    "log_n_sentences",   # ln(1 + кількість речень у документі)
]
F = {name: i for i, name in enumerate(FEATURE_NAMES)}


def _identity(x):  # має бути на рівні модуля, щоб vectorizer серіалізувався joblib
    return x


def make_vectorizer(min_df: int = 1, max_features: int | None = None) -> TfidfVectorizer:
    """TF-IDF по реченнях; вхід — готові списки токенів; без L2-нормування (його виконуємо вручну)."""
    return TfidfVectorizer(
        analyzer=_identity, lowercase=False, norm=None, sublinear_tf=True,
        smooth_idf=True, min_df=min_df, max_features=max_features, dtype=np.float64,
    )


def vocabulary_coverage(vectorizer: TfidfVectorizer, tokens: list[list[str]]) -> float:
    vocab = vectorizer.vocabulary_
    total = sum(len(t) for t in tokens)
    if total == 0:
        return 0.0
    return sum(1 for ts in tokens for t in ts if t in vocab) / total


def _pagerank(sim: np.ndarray, damping: float = 0.85, iters: int = 50) -> np.ndarray:
    n = sim.shape[0]
    if n == 1:
        return np.ones(1)
    row = sim.sum(axis=1, keepdims=True)
    trans = np.divide(sim, row, out=np.full_like(sim, 1.0 / n), where=row > 0)
    pr = np.full(n, 1.0 / n)
    for _ in range(iters):
        new = (1 - damping) / n + damping * trans.T @ pr
        if np.abs(new - pr).sum() < 1e-9:
            pr = new
            break
        pr = new
    return pr * n  # нормуємо так, щоб середнє = 1


def _rank(x: np.ndarray) -> np.ndarray:
    return rankdata(x, method="average") / len(x)


class FeatureExtractor:
    """Перетворює список речень документа на матрицю ознак (n_sentences × len(FEATURE_NAMES)).

    Якщо передано глобальний TF-IDF-векторизатор (навчений на корпусі) і словник покриває
    щонайменше ``min_coverage`` токенів документа — використовується він. Інакше (інша мова,
    специфічна лексика, відсутність моделі) TF-IDF навчається локально на реченнях документа.
    """

    def __init__(self, vectorizer: TfidfVectorizer | None = None, min_coverage: float = 0.5,
                 top_keywords: int = 10):
        self.vectorizer = vectorizer
        self.min_coverage = min_coverage
        self.top_keywords = top_keywords

    def tfidf_matrix(self, tokens: list[list[str]]):
        use_global = (
            self.vectorizer is not None
            and vocabulary_coverage(self.vectorizer, tokens) >= self.min_coverage
        )
        if use_global:
            return self.vectorizer.transform(tokens), "global"
        if not any(tokens):
            import scipy.sparse as sp

            return sp.csr_matrix((len(tokens), 1)), "empty"
        return make_vectorizer().fit_transform(tokens), "local"

    def extract(self, sentences: list[str], tokens: list[list[str]]) -> tuple[np.ndarray, str]:
        n = len(sentences)
        if n == 0:
            return np.zeros((0, len(FEATURE_NAMES))), "empty"
        W, mode = self.tfidf_matrix(tokens)
        Wn = normalize(W, norm="l2")
        tok_len = np.array([max(1, len(t)) for t in tokens], dtype=float)

        tfidf_sum = np.asarray(W.sum(axis=1)).ravel()
        tfidf_mean = tfidf_sum / tok_len
        tfidf_sum_rel = tfidf_sum / tfidf_sum.max() if tfidf_sum.max() > 0 else tfidf_sum

        # косинусна схожість між реченнями та з вектором документа
        sim = (Wn @ Wn.T).toarray()
        np.fill_diagonal(sim, 0.0)
        centrality_mean = sim.sum(axis=1) / max(1, n - 1)
        centrality_pr = _pagerank(sim)
        doc_vec = normalize(np.asarray(W.sum(axis=0)))
        sim_doc = np.asarray(Wn @ doc_vec.T).ravel()

        # ключові слова документа: топ-K термів за сумарною TF-IDF вагою
        col = np.asarray(W.sum(axis=0)).ravel()
        k = int(min(self.top_keywords, (col > 0).sum()))
        binary = (W > 0).astype(float).tocsc()
        if k > 0:
            kw_idx = np.argpartition(-col, k - 1)[:k]
            kw_count = np.asarray(binary[:, kw_idx].sum(axis=1)).ravel()
        else:
            kw_count = np.zeros(n)
        distinct = np.maximum(1.0, np.asarray(binary.sum(axis=1)).ravel())
        kw_ratio = kw_count / distinct

        words = np.array([count_words(s) for s in sentences], dtype=float)
        idx = np.arange(n)
        feats = np.column_stack([
            tfidf_mean, tfidf_sum_rel, _rank(tfidf_sum),
            idx / max(1, n - 1), 1.0 / (idx + 1), (idx == 0).astype(float),
            words, words / max(1e-9, words.mean()),
            kw_count, kw_ratio,
            centrality_mean, centrality_pr, _rank(centrality_pr),
            sim_doc, _rank(sim_doc),
            np.array([has_number(s) for s in sentences], dtype=float),
            np.array([has_quote(s) for s in sentences], dtype=float),
            np.full(n, np.log1p(n)),
        ])
        return feats, mode
