"""Tiny BM25 helper for the Streamlit demo. No dataset download required."""

from __future__ import annotations

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return [tok for tok in text.lower().replace("|", " ").split() if tok]


def query2doc_concat(query: str, pseudo_doc: str, n_repeat: int = 5) -> str:
    return ((query + " ") * n_repeat) + pseudo_doc


def rank_paragraphs(query: str, paragraphs: list[dict]) -> list[dict]:
    corpus = [tokenize(p["text"]) for p in paragraphs]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query))
    ranked = []
    for para, score in sorted(zip(paragraphs, scores), key=lambda x: -x[1]):
        ranked.append({**para, "score": float(score)})
    return ranked
