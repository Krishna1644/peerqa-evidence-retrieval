"""Convert Hugging Face PeerQA into JSONL files for Query2Doc (Approach 1).

Approach 1 expects:
  data/peerqa/corpus.jsonl
  data/peerqa/qa_pairs.jsonl

Approaches 2 and 3 load PeerQA from Hugging Face directly and do not need this step.

Usage:
  python prepare_data.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_utils import (
    build_para_qrels,
    build_paragraph_corpus,
    load_peerqa,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_corpus_rows(para_corpus: dict) -> list[dict]:
    rows = []
    for paper_id, paragraphs in para_corpus.items():
        for para in paragraphs:
            chunk_id = str(para["pidx"])
            rows.append({
                "chunk_id": chunk_id,
                "paper_id": paper_id,
                "pidx": para["pidx"],
                "heading": para.get("heading", ""),
                "text": para["text"],
            })
    return rows


def build_qa_rows(qa, para_corpus: dict, gold_qrels: dict) -> list[dict]:
    rows = []
    for q in qa:
        qid = q["question_id"]
        paper_id = q["paper_id"]
        if not q.get("answerable_mapped"):
            continue
        if qid not in gold_qrels or paper_id not in para_corpus:
            continue
        gold = sorted(gold_qrels[qid])
        rows.append({
            "question_id": qid,
            "paper_id": paper_id,
            "question": q["question"],
            "n_relevant": len(gold),
            "relevant_chunk_ids": gold,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PeerQA to JSONL for Approach 1")
    parser.add_argument("--out-dir", type=Path, default=Path("data/peerqa"))
    args = parser.parse_args()

    print("Loading PeerQA from Hugging Face...")
    print("Note: datasets==2.19.0 is required; newer versions can break PeerQA loading.")
    qa, papers, qrels_para, _qrels_sent = load_peerqa()
    para_corpus = build_paragraph_corpus(papers)
    gold_qrels = build_para_qrels(qrels_para)

    corpus_rows = build_corpus_rows(para_corpus)
    qa_rows = build_qa_rows(qa, para_corpus, gold_qrels)

    corpus_path = args.out_dir / "corpus.jsonl"
    qa_path = args.out_dir / "qa_pairs.jsonl"
    write_jsonl(corpus_path, corpus_rows)
    write_jsonl(qa_path, qa_rows)

    papers_in_eval = {row["paper_id"] for row in qa_rows}
    print(f"Wrote {len(corpus_rows):,} paragraph chunks -> {corpus_path}")
    print(f"Wrote {len(qa_rows):,} evidence-mapped questions -> {qa_path}")
    print(f"Unique papers in eval set: {len(papers_in_eval)}")


if __name__ == "__main__":
    main()
