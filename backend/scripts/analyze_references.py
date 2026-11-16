"""Аналіз еталонних резюме: екстрактивність і порядок речень відносно статті (весь датасет).

Запуск:  python -m scripts.analyze_references [--path data/bbc]
Результат: artifacts/reference_analysis.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from app import config
from app.data.loader import load_dataset
from app.nlp.preprocessing import clean_text, split_sentences


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(config.BBC_DIR))
    ap.add_argument("--out", default=str(config.ARTIFACTS_DIR / "reference_analysis.json"))
    a = ap.parse_args(argv)
    df, _ = load_dataset(Path(a.path))
    total = matched = 0
    taus, in_order, rel_pos = [], 0, []
    docs_ge3 = 0
    for row in df.itertuples(index=False):
        art = norm(clean_text(row.text))
        offsets = []
        for s in split_sentences(clean_text(row.reference), "en"):
            total += 1
            pos = art.find(norm(s))
            if pos >= 0:
                matched += 1
                offsets.append(pos)
                rel_pos.append(pos / max(1, len(art)))
        if len(offsets) >= 3:
            docs_ge3 += 1
            tau = kendalltau(range(len(offsets)), offsets)[0]
            if not np.isnan(tau):
                taus.append(tau)
            in_order += offsets == sorted(offsets)
    res = {
        "documents": int(len(df)),
        "reference_sentences": total,
        "reference_sentences_verbatim_in_article": matched,
        "verbatim_share": round(matched / total, 4),
        "documents_with_ge3_matched_sentences": docs_ge3,
        "documents_reference_order_equals_article_order": int(in_order),
        "share_documents_in_article_order": round(in_order / docs_ge3, 4),
        "mean_kendall_tau_reference_vs_article_order": round(float(np.mean(taus)), 4),
        "mean_relative_position_of_reference_sentences": round(float(np.mean(rel_pos)), 4),
        "method": "речення еталона шукаються як підрядок у статті (після нормалізації пробілів); позиція — зсув символів",
    }
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
