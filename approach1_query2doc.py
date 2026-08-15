"""
Approach 1 (v2): Query2Doc-Augmented BM25 with RRF
====================================================
Improved query rewriting for lexical retrieval. Replaces the v1 paraphrase
strategy with two evidence-backed changes:

  1. bm25s + PyStemmer + English stopwords  (Lucene-equivalent analyzer)
     -> closes the tokenizer gap with the published PeerQA BM25 baseline.

  2. Query2Doc rewriting (Wang et al., EMNLP 2023)
     -> generate a HYPOTHETICAL ANSWER PASSAGE (not a paraphrase),
        concatenate as `(query + ' ') * 5 + passage`. The 5x repetition
        keeps original keywords salient against the longer pseudo-doc.
        Reported gains: +15 nDCG@10 on TREC DL.

Three rankings reported:
  A. BM25 (original query)
  B. BM25 (Query2Doc-augmented query)
  C. RRF fusion of A + B

Reads JSONL from data/peerqa/, writes metrics to results/.

Usage:
  python approach1_query2doc.py                 # full pipeline
  python approach1_query2doc.py --dry-run 5     # preview 5 pseudo-docs
  python approach1_query2doc.py --skip-rewrite  # use cached pseudo-docs only
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import bm25s
import numpy as np
import Stemmer
from openai import OpenAI
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent

# ── Data Loading ─────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_data(data_dir: Path):
    """Load corpus and QA pairs. Returns (paper_chunks, qa_eval)."""
    corpus = load_jsonl(data_dir / "corpus.jsonl")
    qa_pairs = load_jsonl(data_dir / "qa_pairs.jsonl")

    qa_eval = [q for q in qa_pairs if q["n_relevant"] > 0]

    paper_chunks: dict[str, list[dict]] = defaultdict(list)
    for chunk in corpus:
        paper_chunks[chunk["paper_id"]].append(chunk)
    for pid in paper_chunks:
        paper_chunks[pid].sort(key=lambda c: c["pidx"])

    print(f"  Corpus chunks : {len(corpus):,}")
    print(f"  QA pairs total: {len(qa_pairs):,}")
    print(f"  QA with evidence (eval set): {len(qa_eval):,}")
    print(f"  Unique papers in eval set  : {len(set(q['paper_id'] for q in qa_eval)):,}")

    return paper_chunks, qa_eval


# ── Query2Doc: LLM Pseudo-Document Generation ───────────────────────────────

SYSTEM_PROMPT = (
    "You are a scientific paper writing assistant. Given a question that a "
    "peer reviewer might ask about a scientific paper, you write a short "
    "passage from the paper that would directly contain the answer. Use "
    "formal academic language and the precise technical terminology that "
    "the paper authors would use. Do not address the reviewer; write as if "
    "the passage is excerpted from the paper itself."
)

USER_PROMPT_TEMPLATE = (
    "Write a passage of 2-4 sentences that would appear in a scientific "
    "paper and would directly answer the following question. Use specific "
    "technical terminology. Output ONLY the passage, no preamble.\n\n"
    "Question: {question}\n\nPassage:"
)


# Per-1M-token pricing (USD) - used for upfront cost estimate
MODEL_PRICING = {
    "gpt-4o-mini":     {"in": 0.15, "out": 0.60},
    "gpt-4o":          {"in": 2.50, "out": 10.00},
    "gpt-4.1-mini":    {"in": 0.40, "out": 1.60},
    "gpt-4.1":         {"in": 2.00, "out": 8.00},
}


def estimate_cost(n_queries: int, model: str) -> str:
    """Rough per-run cost estimate (150 input tok, 250 output tok per call)."""
    if model not in MODEL_PRICING:
        return f"  Cost estimate unavailable for model '{model}'"
    p = MODEL_PRICING[model]
    in_tok = n_queries * 150
    out_tok = n_queries * 250
    cost = in_tok * p["in"] / 1_000_000 + out_tok * p["out"] / 1_000_000
    return f"  Estimated cost: ${cost:.2f} for {n_queries} queries with {model}"


def generate_pseudo_docs(
    qa_pairs: list[dict],
    cache_path: Path,
    model: str = "gpt-4o",
    dry_run: int = 0,
    api_key: str | None = None,
) -> dict[str, str]:
    """Generate Query2Doc pseudo-documents via OpenAI API with caching."""

    cached: dict[str, str] = {}
    if cache_path.exists():
        for row in load_jsonl(cache_path):
            cached[row["question_id"]] = row["pseudo_doc"]
        print(f"  Loaded {len(cached)} cached pseudo-docs")

    to_generate = [q for q in qa_pairs if q["question_id"] not in cached]

    if dry_run > 0:
        to_generate = to_generate[:dry_run]
        print(f"\n  DRY RUN: generating {len(to_generate)} pseudo-docs for preview\n")
    else:
        print(f"  Need to generate: {len(to_generate)} pseudo-docs")
        if to_generate:
            print(estimate_cost(len(to_generate), model))

    if not to_generate:
        print("  All pseudo-docs already cached - skipping API calls")
        return cached

    client = OpenAI(api_key=api_key) if api_key else OpenAI()

    with open(cache_path, "a", encoding="utf-8") as cache_f:
        for qa in tqdm(to_generate, desc="  Generating pseudo-docs"):
            question = qa["question"]
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(question=question)},
            ]

            pseudo = None
            for attempt in range(3):
                try:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.3,
                        top_p=1,
                        max_tokens=300,
                    )
                    pseudo = completion.choices[0].message.content.strip()
                    pseudo = pseudo.strip('"').strip("'").strip()
                    break
                except Exception as e:
                    print(f"\n    Retry {attempt+1}/3: {e}")
                    time.sleep(3 * (attempt + 1))

            if pseudo is None or len(pseudo) < 10:
                print(f"\n    FAILED for {qa['question_id']} - using question as pseudo-doc")
                pseudo = question

            cached[qa["question_id"]] = pseudo

            cache_f.write(json.dumps({
                "question_id": qa["question_id"],
                "original": question,
                "pseudo_doc": pseudo,
            }, ensure_ascii=False) + "\n")
            cache_f.flush()

            if dry_run > 0:
                print(f"\n    QUESTION   : {question}")
                print(f"    PSEUDO-DOC : {pseudo}")

            time.sleep(0.05)

    return cached


def query2doc_concat(query: str, pseudo_doc: str, n_repeat: int = 5) -> str:
    """Build Query2Doc augmented query: (q + ' ') * n + pseudo_doc.

    The 5x repetition is from the original Query2Doc paper - it keeps the
    original query terms salient against the longer pseudo-document so BM25
    doesn't get drowned out by pseudo-doc tokens.
    """
    return ((query + " ") * n_repeat) + pseudo_doc


# ── BM25 with bm25s (Lucene-equivalent analyzer) ────────────────────────────

# PyStemmer's Porter stemmer + bm25s' built-in English stopwords match what
# Pyserini's Lucene EnglishAnalyzer does. This closes most of the gap to
# the published BM25 baseline (0.4288 paragraph MRR).
_STEMMER = Stemmer.Stemmer("english")


def tokenize_for_bm25(texts: list[str]) -> list[list[str]]:
    """Tokenize a list of texts: lowercase + strip punctuation + stopwords + stem."""
    return bm25s.tokenize(
        texts,
        stopwords="en",
        stemmer=_STEMMER,
        show_progress=False,
    )


def build_bm25_index(chunks: list[dict]) -> tuple:
    """Build a bm25s index. Returns (retriever, chunk_ids)."""
    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]
    tokens = tokenize_for_bm25(texts)
    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=False)
    return retriever, chunk_ids


def bm25_score_all(query: str, retriever, chunk_ids: list[str]) -> dict[str, float]:
    """Score ALL documents in the index against the query.

    bm25s.retrieve returns top-k; we set k = N to score every doc.
    """
    tokens = tokenize_for_bm25([query])
    n_docs = len(chunk_ids)
    results, scores = retriever.retrieve(
        tokens, k=n_docs, show_progress=False
    )
    # results[0] is array of doc indices, scores[0] is array of scores
    out = {}
    for doc_idx, score in zip(results[0], scores[0]):
        out[chunk_ids[doc_idx]] = float(score)
    return out


# ── Reciprocal Rank Fusion ───────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranking_a: dict[str, float],
    ranking_b: dict[str, float],
    k: int = 60,
) -> dict[str, float]:
    """Fuse two score dicts via RRF. Returns {chunk_id: rrf_score}."""
    sorted_a = sorted(ranking_a.items(), key=lambda x: -x[1])
    sorted_b = sorted(ranking_b.items(), key=lambda x: -x[1])

    rank_a = {cid: r for r, (cid, _) in enumerate(sorted_a, 1)}
    rank_b = {cid: r for r, (cid, _) in enumerate(sorted_b, 1)}

    all_ids = set(rank_a.keys()) | set(rank_b.keys())
    n = len(all_ids) + 1

    fused = {}
    for cid in all_ids:
        fused[cid] = 1.0 / (k + rank_a.get(cid, n)) + 1.0 / (k + rank_b.get(cid, n))
    return fused


# ── Retrieval Pipeline ───────────────────────────────────────────────────────

def run_retrieval(
    qa_pairs: list[dict],
    paper_chunks: dict[str, list[dict]],
    pseudo_docs: dict[str, str],
    rrf_k: int = 60,
    n_repeat: int = 5,
) -> tuple[dict, dict, dict]:
    """Run BM25 with original query, Query2Doc-augmented query, and RRF fusion."""
    run_original = {}
    run_q2d = {}
    run_fused = {}

    paper_questions: dict[str, list[dict]] = defaultdict(list)
    for qa in qa_pairs:
        paper_questions[qa["paper_id"]].append(qa)

    for paper_id, questions in tqdm(paper_questions.items(), desc="  Papers"):
        chunks = paper_chunks.get(paper_id, [])
        if not chunks:
            continue

        retriever, chunk_ids = build_bm25_index(chunks)

        for qa in questions:
            qid = qa["question_id"]
            original_query = qa["question"]
            pseudo = pseudo_docs.get(qid, "")
            if pseudo:
                augmented_query = query2doc_concat(original_query, pseudo, n_repeat=n_repeat)
            else:
                augmented_query = original_query

            scores_orig = bm25_score_all(original_query, retriever, chunk_ids)
            scores_q2d = bm25_score_all(augmented_query, retriever, chunk_ids)
            scores_fused = reciprocal_rank_fusion(scores_orig, scores_q2d, k=rrf_k)

            run_original[qid] = scores_orig
            run_q2d[qid] = scores_q2d
            run_fused[qid] = scores_fused

    return run_original, run_q2d, run_fused


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(
    run: dict[str, dict[str, float]],
    qa_pairs: list[dict],
    ks: list[int] = [1, 5, 10, 20],
) -> dict[str, float]:
    """Compute MRR and Recall@k over the eval set."""
    mrr_scores = []
    recall_at_k = {k: [] for k in ks}

    for qa in qa_pairs:
        qid = qa["question_id"]
        relevant = set(qa["relevant_chunk_ids"])
        if qid not in run:
            continue

        ranked = sorted(run[qid].items(), key=lambda x: -x[1])
        ranked_ids = [cid for cid, _ in ranked]

        rr = 0.0
        for rank, cid in enumerate(ranked_ids, 1):
            if cid in relevant:
                rr = 1.0 / rank
                break
        mrr_scores.append(rr)

        for k in ks:
            top_k = set(ranked_ids[:k])
            recall_at_k[k].append(len(top_k & relevant) / len(relevant))

    metrics = {"MRR": float(np.mean(mrr_scores))}
    for k in ks:
        metrics[f"Recall@{k}"] = float(np.mean(recall_at_k[k]))
    return metrics


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Approach 1 v2: Query2Doc-Augmented BM25 + RRF"
    )
    parser.add_argument("--data-dir", type=Path, default=SCRIPT_DIR / "data" / "peerqa")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="OpenAI model. gpt-4o ~$1.10/run, gpt-4o-mini ~$0.05/run")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--n-repeat", type=int, default=5,
                        help="Query2Doc query repetition factor (default 5 from paper)")
    parser.add_argument("--dry-run", type=int, default=0,
                        help="Generate N pseudo-docs and print them, skip retrieval")
    parser.add_argument("--skip-rewrite", action="store_true",
                        help="Use cached pseudo-docs only, no API calls")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenAI API key (fallback if OPENAI_API_KEY not set)")
    args = parser.parse_args()

    print("=" * 70)
    print("APPROACH 1 v2: Query2Doc-Augmented BM25 + RRF")
    print("=" * 70)

    # [1/4] Load data
    print("\n[1/4] Loading data ...")
    paper_chunks, qa_eval = load_data(args.data_dir)

    # [2/4] Generate pseudo-docs
    print("\n[2/4] Query2Doc pseudo-document generation ...")
    cache_path = args.data_dir / "query2doc_passages.jsonl"

    if args.skip_rewrite:
        print("  --skip-rewrite: loading cached pseudo-docs only")
        pseudo_docs = {}
        if cache_path.exists():
            for row in load_jsonl(cache_path):
                pseudo_docs[row["question_id"]] = row["pseudo_doc"]
        print(f"  Loaded {len(pseudo_docs)} cached pseudo-docs")
    else:
        pseudo_docs = generate_pseudo_docs(
            qa_eval, cache_path, model=args.model,
            dry_run=args.dry_run, api_key=args.api_key,
        )

    if args.dry_run > 0:
        print("\n  Dry run complete - exiting before retrieval.")
        return

    # [3/4] BM25 retrieval
    print("\n[3/4] Running BM25 retrieval (bm25s + Porter stemmer + stopwords) ...")
    run_orig, run_q2d, run_fused = run_retrieval(
        qa_eval, paper_chunks, pseudo_docs,
        rrf_k=args.rrf_k, n_repeat=args.n_repeat,
    )

    # [4/4] Evaluate
    print("\n[4/4] Evaluating ...")
    metrics_orig = evaluate(run_orig, qa_eval)
    metrics_q2d = evaluate(run_q2d, qa_eval)
    metrics_fused = evaluate(run_fused, qa_eval)

    # Results table
    print("\n" + "=" * 70)
    print("APPROACH 1 v2 RESULTS")
    print("=" * 70)
    header = f"{'Method':<26} {'MRR':>7} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'R@20':>7}"
    print(header)
    print("-" * 70)
    for label, m in [
        ("BM25 (original)", metrics_orig),
        ("BM25 (Query2Doc q5+d)", metrics_q2d),
        ("BM25 (RRF fused)", metrics_fused),
    ]:
        print(
            f"{label:<26} "
            f"{m['MRR']:>7.4f} "
            f"{m['Recall@1']:>7.4f} "
            f"{m['Recall@5']:>7.4f} "
            f"{m['Recall@10']:>7.4f} "
            f"{m['Recall@20']:>7.4f}"
        )
    print("-" * 70)
    print(f"{'PeerQA paper baseline':<26} {'0.4288':>7}       -       -       -       -")
    print("=" * 70)

    # Save results
    results = {
        "bm25_original": metrics_orig,
        "bm25_query2doc": metrics_q2d,
        "bm25_fused": metrics_fused,
        "config": {
            "model": args.model,
            "bm25_library": "bm25s",
            "tokenizer": "lowercase + Porter stemmer (PyStemmer) + English stopwords",
            "n_repeat": args.n_repeat,
            "rrf_k": args.rrf_k,
            "n_queries_evaluated": len(qa_eval),
            "method": "Query2Doc (Wang et al., EMNLP 2023)",
        },
    }

    out_dir = SCRIPT_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "approach1_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    runs_path = out_dir / "approach1_runs.json"
    with open(runs_path, "w", encoding="utf-8") as f:
        json.dump({
            "bm25_original": run_orig,
            "bm25_query2doc": run_q2d,
            "bm25_fused": run_fused,
        }, f)
    print(f"Full runs saved to {runs_path}")


if __name__ == "__main__":
    main()
