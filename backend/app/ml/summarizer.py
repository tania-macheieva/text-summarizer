"""Extractive summarization using TF-IDF features and a Random Forest classifier."""
from __future__ import annotations

import logging
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ..config import DEFAULT_RATIO
from ..nlp.preprocessing import Preprocessor, count_words
from .features import F, FEATURE_NAMES, FeatureExtractor

logger = logging.getLogger(__name__)

MODEL_VERSION = 1
ALGORITHMS = ("random_forest", "tfidf", "lead")


class SummarizerError(ValueError):
    """Invalid summarization input or parameters."""


@dataclass
class KeySentence:
    index: int      # позиція в документі (з 0)
    text: str
    score: float    # важливість [0..1] (імовірність класу «важливе» або нормований бал)
    rank: int       # 1 — найважливіше


@dataclass
class SummaryResult:
    summary: str
    key_sentences: list[KeySentence]
    sentences: list[str]
    selected_indices: list[int]
    scores: list[float]
    language: str
    model_name: str
    used_fallback: bool
    tfidf_mode: str
    original_word_count: int
    summary_word_count: int
    compression_ratio: float
    processing_time: float
    warnings: list[str] = field(default_factory=list)


def tfidf_rank_scores(feats: np.ndarray) -> np.ndarray:
    """Базовий алгоритм «TF-IDF ranking»: середня TF-IDF вага термів речення."""
    return feats[:, F["tfidf_mean"]].copy()


def heuristic_scores(feats: np.ndarray) -> np.ndarray:
    """Змістовний fallback без навчання: TF-IDF + схожість із документом + центральність.

    Ваги (0.5 / 0.3 / 0.2) задані експертно й не підбиралися на даних.
    """
    return (0.5 * feats[:, F["tfidf_rank"]] + 0.3 * feats[:, F["sim_doc_rank"]]
            + 0.2 * feats[:, F["centrality_rank"]])


def lead_scores(n: int) -> np.ndarray:
    """Lead-N: чим ближче до початку, тим вищий бал."""
    return np.arange(n, 0, -1, dtype=float)


def num_to_select(n_sentences: int, num_sentences: int | None, ratio: float | None) -> int:
    if num_sentences is not None and ratio is not None:
        raise SummarizerError("Specify either num_sentences or ratio, not both.")
    if num_sentences is not None:
        if num_sentences < 1:
            raise SummarizerError("num_sentences must be at least 1.")
        return min(num_sentences, n_sentences)
    r = DEFAULT_RATIO if ratio is None else ratio
    if not 0 < r <= 1:
        raise SummarizerError("ratio must be within (0, 1].")
    return max(1, min(n_sentences, int(round(r * n_sentences))))


def select_top(scores: np.ndarray, k: int) -> list[int]:
    """TOP-k речень за балом; повертає індекси в ПОЧАТКОВОМУ порядку документа."""
    order = np.argsort(-scores, kind="stable")[:k]
    return sorted(int(i) for i in order)


class ExtractiveSummarizer:
    def __init__(self, classifier: RandomForestClassifier | None = None, vectorizer=None,
                 metadata: dict | None = None):
        self.classifier = classifier
        self.vectorizer = vectorizer
        self.metadata = metadata or {}
        self.preprocessor = Preprocessor()

    # ------------------------------------------------------------------ стан моделі
    @property
    def is_trained(self) -> bool:
        return self.classifier is not None

    def extractor(self) -> FeatureExtractor:
        return FeatureExtractor(self.vectorizer)

    # ------------------------------------------------------------------ навчання
    def fit(self, X: np.ndarray, y: np.ndarray, vectorizer=None, **rf_params) -> "ExtractiveSummarizer":
        """Навчає Random Forest на ознаках речень X та мітках y (greedy ROUGE-L oracle)."""
        if len(X) == 0 or len(set(np.asarray(y).tolist())) < 2:
            raise SummarizerError("Training requires examples from both classes.")
        params = dict(n_estimators=300, max_depth=12, min_samples_leaf=3,
                      class_weight="balanced_subsample", n_jobs=-1, random_state=42)
        params.update(rf_params)
        self.classifier = RandomForestClassifier(**params).fit(X, y)
        self.vectorizer = vectorizer
        return self

    # ------------------------------------------------------------------ збереження
    def save(self, path: Path) -> Path:
        if not self.is_trained:
            raise SummarizerError("There is no trained model to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"version": MODEL_VERSION, "classifier": self.classifier, "vectorizer": self.vectorizer,
                     "feature_names": FEATURE_NAMES, "metadata": self.metadata}, path, compress=3)
        return path

    @classmethod
    def load(cls, path: Path) -> "ExtractiveSummarizer":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        blob = joblib.load(path)
        if blob.get("feature_names") != FEATURE_NAMES:
            raise SummarizerError("The model was trained with a different feature set. Retrain it.")
        clf = blob["classifier"]
        clf.n_jobs = 1  # передбачення по одному документу швидше без пулу потоків
        return cls(clf, blob["vectorizer"], blob.get("metadata"))

    # ------------------------------------------------------------------ оцінювання речень
    def score_features(self, feats: np.ndarray, algorithm: str = "random_forest", seed: int = 0) -> tuple[np.ndarray, str, bool]:
        """Повертає (бали, назва моделі, чи використано fallback)."""
        if algorithm == "random_forest":
            if self.is_trained:
                return self.classifier.predict_proba(feats)[:, 1], "random_forest", False
            return heuristic_scores(feats), "tfidf_centrality (fallback)", True
        if algorithm == "tfidf":
            return tfidf_rank_scores(feats), "tfidf", False
        if algorithm == "tfidf_centrality":
            return heuristic_scores(feats), "tfidf_centrality", False
        if algorithm == "lead":
            return lead_scores(len(feats)), "lead", False
        if algorithm == "random":  # контрольний baseline для експериментів (детермінований за seed)
            return np.random.RandomState(seed).rand(len(feats)), "random", False
        raise SummarizerError(f"Unknown algorithm: {algorithm}")

    # ------------------------------------------------------------------ самаризація
    def summarize(self, text: str, language: str = "auto", num_sentences: int | None = None,
                  ratio: float | None = None, algorithm: str = "random_forest",
                  word_ratio: float | None = None) -> SummaryResult:
        """word_ratio — альтернативний спосіб задати довжину: резюме набирається в порядку рангу,
        доки не досягне частки word_ratio від кількості слів оригіналу (для порівняння за однакового бюджету слів)."""
        t0 = time.perf_counter()
        if not isinstance(text, str) or not text.strip():
            raise SummarizerError("The text is empty.")
        if language not in ("auto", "en", "uk"):
            raise SummarizerError(f"Unsupported language: {language}")
        doc = self.preprocessor.process(text, language)
        sentences, tokens = doc["sentences"], doc["tokens"]
        if not sentences or not any(tokens):
            raise SummarizerError("No analyzable words were found in the text.")
        n = len(sentences)
        if word_ratio is not None and (num_sentences is not None or ratio is not None):
            raise SummarizerError("word_ratio cannot be combined with num_sentences or ratio.")
        k = num_to_select(n, num_sentences, ratio) if word_ratio is None else n
        feats, tfidf_mode = self.extractor().extract(sentences, tokens)
        scores, model_name, fallback = self.score_features(feats, algorithm, seed=zlib.crc32(text.encode("utf-8")))
        if word_ratio is None:
            chosen = select_top(scores, k)
        else:
            if not 0 < word_ratio <= 1:
                raise SummarizerError("word_ratio must be within (0, 1].")
            wc = [count_words(x) for x in sentences]
            target, acc, picked = word_ratio * sum(wc), 0, []
            for i in np.argsort(-scores, kind="stable"):
                if acc >= target and picked:
                    break
                picked.append(int(i)); acc += wc[i]
            chosen = sorted(picked)
        summary = " ".join(sentences[i] for i in chosen)

        lo, hi = float(scores.min()), float(scores.max())
        norm = (scores - lo) / (hi - lo) if hi > lo else np.ones_like(scores)
        ranking = np.argsort(-scores, kind="stable")
        rank_of = {int(i): r + 1 for r, i in enumerate(ranking)}
        shown = scores if (algorithm == "random_forest" and not fallback) else norm
        keys = sorted(
            (KeySentence(i, sentences[i], round(float(shown[i]), 4), rank_of[i]) for i in chosen),
            key=lambda ks: ks.rank,
        )
        ow, sw = count_words(" ".join(sentences)), count_words(summary)
        warnings = []
        if fallback:
            warnings.append("No trained model is available; using the TF-IDF fallback.")
        if doc["language"] == "uk" and self.is_trained and algorithm == "random_forest":
            warnings.append("The model was trained on English text; its quality for Ukrainian has not been evaluated.")
        return SummaryResult(
            summary=summary, key_sentences=keys, sentences=sentences, selected_indices=chosen,
            scores=[float(s) for s in scores], language=doc["language"], model_name=model_name,
            used_fallback=fallback, tfidf_mode=tfidf_mode, original_word_count=ow, summary_word_count=sw,
            compression_ratio=round(sw / ow, 4) if ow else 0.0,
            processing_time=round(time.perf_counter() - t0, 5), warnings=warnings,
        )
