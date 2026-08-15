import argparse
import json
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from data_utils import (
    load_peerqa, build_paragraph_corpus, build_sentence_corpus,
    build_title_map, build_para_qrels, build_sent_qrels,
    evaluate, print_results,
)


def tokenize(text):
    return text.lower().split()


def prepend_title(text, title, chunk_type="Paragraph"):
    if title:
        return f"Title: {title} {chunk_type}: {text}"
    return text


def run_bm25_paragraph(qa, papers, qrels_para):
    para_corpus = build_paragraph_corpus(papers)
    gold_qrels = build_para_qrels(qrels_para)
    title_map = build_title_map(papers)

    questions = [
        q for q in qa
        if q["answerable_mapped"]
        and q["question_id"] in gold_qrels
        and q["paper_id"] in para_corpus
    ]
    print(f"Evaluating on {len(questions)} questions (paragraph level)...")

    results_plain, results_titled = [], []
    for q in tqdm(questions, desc="BM25 paragraphs"):
        paper_id = q["paper_id"]
        paragraphs = para_corpus[paper_id]
        title = title_map.get(paper_id, "")
        gold_ids = gold_qrels[q["question_id"]]
        plain_docs = [p["text"] for p in paragraphs]
        titled_docs = [prepend_title(p["text"], title, "Paragraph") for p in paragraphs]
        para_ids = [str(p["pidx"]) for p in paragraphs]
        query_tokens = tokenize(q["question"])

        bm25 = BM25Okapi([tokenize(d) for d in plain_docs])
        scores = bm25.get_scores(query_tokens)
        ranked_plain = [para_ids[i] for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)]

        bm25t = BM25Okapi([tokenize(d) for d in titled_docs])
        scores_t = bm25t.get_scores(query_tokens)
        ranked_titled = [para_ids[i] for i in sorted(range(len(scores_t)), key=lambda i: scores_t[i], reverse=True)]

        results_plain.append({"ranked_ids": ranked_plain, "gold_ids": gold_ids})
        results_titled.append({"ranked_ids": ranked_titled, "gold_ids": gold_ids})

    return evaluate(results_plain), evaluate(results_titled)


def run_bm25_sentence(qa, papers, qrels_sent):
    sent_corpus = build_sentence_corpus(papers)
    gold_qrels = build_sent_qrels(qrels_sent)
    title_map = build_title_map(papers)

    questions = [
        q for q in qa
        if q["answerable_mapped"]
        and q["question_id"] in gold_qrels
        and q["paper_id"] in sent_corpus
    ]
    print(f"Evaluating on {len(questions)} questions (sentence level)...")

    results_plain, results_titled = [], []
    for q in tqdm(questions, desc="BM25 sentences"):
        paper_id = q["paper_id"]
        sentences = sent_corpus[paper_id]
        title = title_map.get(paper_id, "")
        gold_ids = gold_qrels[q["question_id"]]
        plain_docs = [s["text"] for s in sentences]
        titled_docs = [prepend_title(s["text"], title, "Sentence") for s in sentences]
        sent_ids = [f"{s['pidx']}/{s['sidx']}" for s in sentences]
        query_tokens = tokenize(q["question"])

        bm25 = BM25Okapi([tokenize(d) for d in plain_docs])
        scores = bm25.get_scores(query_tokens)
        ranked_plain = [sent_ids[i] for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)]

        bm25t = BM25Okapi([tokenize(d) for d in titled_docs])
        scores_t = bm25t.get_scores(query_tokens)
        ranked_titled = [sent_ids[i] for i in sorted(range(len(scores_t)), key=lambda i: scores_t[i], reverse=True)]

        results_plain.append({"ranked_ids": ranked_plain, "gold_ids": gold_ids})
        results_titled.append({"ranked_ids": ranked_titled, "gold_ids": gold_ids})

    return evaluate(results_plain), evaluate(results_titled)


def main(level="paragraph"):
    print("Loading PeerQA dataset...")
    qa, papers, qrels_para, qrels_sent = load_peerqa()

    if level == "paragraph":
        plain_metrics, titled_metrics = run_bm25_paragraph(qa, papers, qrels_para)
        print_results("BM25 Paragraph (No Title)", plain_metrics)
        print_results("BM25 Paragraph (+Title)", titled_metrics)
        results = {"BM25_paragraph_plain": plain_metrics, "BM25_paragraph_titled": titled_metrics}
    else:
        plain_metrics, titled_metrics = run_bm25_sentence(qa, papers, qrels_sent)
        print_results("BM25 Sentence (No Title)", plain_metrics)
        print_results("BM25 Sentence (+Title)", titled_metrics)
        results = {"BM25_sentence_plain": plain_metrics, "BM25_sentence_titled": titled_metrics}

    from pathlib import Path
    out = Path("results") / f"baseline_{level}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["paragraph", "sentence"], default="paragraph")
    args = parser.parse_args()
    main(args.level)