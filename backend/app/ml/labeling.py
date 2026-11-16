"""Формування навчальних міток: greedy ROUGE-L oracle (без випадковості)."""
from __future__ import annotations

import numpy as np

from .rouge import rouge_l_f1


def greedy_oracle(sent_tokens: list[list[str]], ref_tokens: list[str], max_sentences: int | None = None) -> np.ndarray:
    """Жадібно додає речення, що найбільше збільшує ROUGE-L F1 відносно еталонного резюме.

    Зупиняється, коли жодне речення не покращує F1. Кандидат завжди складається з обраних
    речень у початковому порядку документа. Повертає бінарний вектор міток (1 — «важливе»).
    """
    n = len(sent_tokens)
    labels = np.zeros(n, dtype=int)
    if n == 0 or not ref_tokens:
        return labels
    selected: list[int] = []
    best = 0.0
    limit = max_sentences or n
    while len(selected) < limit:
        gain_idx, gain_f = -1, best
        for i in range(n):
            if labels[i]:
                continue
            cand = sorted(selected + [i])
            hyp = [t for j in cand for t in sent_tokens[j]]
            f = rouge_l_f1(ref_tokens, hyp)
            if f > gain_f + 1e-12:
                gain_idx, gain_f = i, f
        if gain_idx < 0:
            break
        selected.append(gain_idx)
        labels[gain_idx] = 1
        best = gain_f
    return labels
