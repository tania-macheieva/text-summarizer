"""ROUGE-1/2/L (Precision, Recall, F1).

Токенізація й стемінг сумісні зі стандартним пакетом ``rouge-score`` для англійських текстів
(перевіряється в тестах), але працюють і з кирилицею. ROUGE-L — LCS на рівні всього тексту
(аналог ``rougeL`` у rouge-score). LCS обчислюється бітово-паралельним алгоритмом (Hyyrö),
що на порядки швидше за класичне ДП і потрібне для greedy-розмітки.
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

from ..nlp.language import english_stemmer

_TOK = re.compile(r"[^\W_]+", re.UNICODE)


@lru_cache(maxsize=300_000)
def _stem(tok: str) -> str:
    if len(tok) > 3 and tok.isascii():
        return english_stemmer().stem(tok)
    return tok


def rouge_tokens(text: str, stem: bool = True) -> list[str]:
    toks = _TOK.findall(text.lower())
    return [_stem(t) for t in toks] if stem else toks


def lcs_length(a: list[str], b: list[str]) -> int:
    """Довжина найдовшої спільної підпослідовності (бітово-паралельний алгоритм)."""
    if not a or not b:
        return 0
    m = len(b)
    masks: dict[str, int] = {}
    for i, t in enumerate(b):
        masks[t] = masks.get(t, 0) | (1 << i)
    full = (1 << m) - 1
    v = full
    for t in a:
        u = v & masks.get(t, 0)
        v = ((v + u) | (v - u)) & full
    return m - bin(v).count("1")


def lcs_length_dp(a: list[str], b: list[str]) -> int:
    """Еталонна реалізація ДП (для перевірки в тестах)."""
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def _prf(overlap: int, n_ref: int, n_hyp: int) -> dict[str, float]:
    p = overlap / n_hyp if n_hyp else 0.0
    r = overlap / n_ref if n_ref else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f}


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rouge_n_tokens(ref: list[str], hyp: list[str], n: int) -> dict[str, float]:
    rn, hn = _ngrams(ref, n), _ngrams(hyp, n)
    overlap = sum((rn & hn).values())
    return _prf(overlap, sum(rn.values()), sum(hn.values()))


def rouge_l_tokens(ref: list[str], hyp: list[str]) -> dict[str, float]:
    return _prf(lcs_length(ref, hyp), len(ref), len(hyp))


def rouge_l_f1(ref: list[str], hyp: list[str]) -> float:
    return rouge_l_tokens(ref, hyp)["f1"]


def rouge_from_tokens(ref: list[str], hyp: list[str]) -> dict[str, dict[str, float]]:
    return {
        "rouge1": rouge_n_tokens(ref, hyp, 1),
        "rouge2": rouge_n_tokens(ref, hyp, 2),
        "rougeL": rouge_l_tokens(ref, hyp),
    }


def rouge_scores(reference: str, hypothesis: str, stem: bool = True) -> dict[str, dict[str, float]]:
    """ROUGE-1, ROUGE-2, ROUGE-L для пари текстів."""
    return rouge_from_tokens(rouge_tokens(reference, stem), rouge_tokens(hypothesis, stem))
