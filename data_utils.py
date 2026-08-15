from __future__ import annotations
from collections import defaultdict
from datasets import load_dataset


def load_peerqa():
    print("Loading QA pairs...")
    qa = load_dataset("UKPLab/PeerQA", "qa", trust_remote_code=True)["test"]
    print("Loading paper text...")
    papers = load_dataset("UKPLab/PeerQA", "papers", trust_remote_code=True)["test"]
    print("Loading paragraph qrels...")
    qrels_para = load_dataset("UKPLab/PeerQA", "qrels-paragraphs", trust_remote_code=True)["test"]
    print("Loading sentence qrels...")
    qrels_sent = load_dataset("UKPLab/PeerQA", "qrels-sentences", trust_remote_code=True)["test"]
    return qa, papers, qrels_para, qrels_sent


def build_paragraph_corpus(papers):
    para_sentences = defaultdict(list)
    para_meta = {}
    for row in papers:
        if row.get("type") != "sentence":
            continue
        key = (row["paper_id"], row["pidx"])
        para_sentences[key].append(row["content"])
        para_meta[key] = {"heading": row.get("last_heading") or ""}
    paper_paragraphs = defaultdict(list)
    for (paper_id, pidx), sentences in sorted(
        para_sentences.items(), key=lambda x: (x[0][0], x[0][1])
    ):
        text = " ".join(sentences).strip()
        if not text:
            continue
        paper_paragraphs[paper_id].append({
            "pidx": pidx,
            "text": text,
            "heading": para_meta[(paper_id, pidx)]["heading"],
        })
    return dict(paper_paragraphs)


def build_sentence_corpus(papers):
    paper_sentences = defaultdict(list)
    for row in papers:
        if row.get("type") != "sentence" or not row["content"].strip():
            continue
        paper_sentences[row["paper_id"]].append({
            "pidx": row["pidx"],
            "sidx": row["sidx"],
            "text": row["content"],
            "heading": row.get("last_heading") or "",
        })
    return dict(paper_sentences)


def build_title_map(papers):
    titles = {}
    for row in papers:
        if row.get("type") == "title" and row["paper_id"] not in titles:
            titles[row["paper_id"]] = row["content"]
    return titles


def build_abstract_map(papers):
    abstracts = {}
    for row in papers:
        if row.get("type") == "sentence" and row["paper_id"] not in abstracts:
            abstracts[row["paper_id"]] = row["content"]
    return abstracts


def build_para_qrels(qrels_para):
    qrels = defaultdict(set)
    for row in qrels_para:
        qrels[row["question_id"]].add(str(row["idx"]))
    return dict(qrels)


def build_sent_qrels(qrels_sent):
    qrels = defaultdict(set)
    for row in qrels_sent:
        qrels[row["question_id"]].add(str(row["idx"]))
    return dict(qrels)


def compute_mrr(ranked_ids, gold_ids):
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in gold_ids:
            return 1.0 / rank
    return 0.0


def compute_recall_at_k(ranked_ids, gold_ids, k=10):
    return 1.0 if set(ranked_ids[:k]) & gold_ids else 0.0


def evaluate(results, k=10):
    mrr_scores = [compute_mrr(r["ranked_ids"], r["gold_ids"]) for r in results]
    recall_scores = [compute_recall_at_k(r["ranked_ids"], r["gold_ids"], k) for r in results]
    return {
        "MRR": round(sum(mrr_scores) / len(mrr_scores), 4),
        f"Recall@{k}": round(sum(recall_scores) / len(recall_scores), 4),
        "n_queries": len(results),
    }


def print_results(label, metrics):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")