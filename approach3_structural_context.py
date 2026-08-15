import argparse
import json
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from data_utils import (
    load_peerqa, build_paragraph_corpus, build_sentence_corpus,
    build_title_map, build_abstract_map, build_para_qrels, build_sent_qrels,
    evaluate, print_results,
)


def tokenize(text):
    return text.lower().split()


# --- Sentence-level context formats ---

def fmt_plain(s, title, abstract):
    return s["text"]

def fmt_title_only(s, title, abstract):
    if title:
        return f"Title: {title} Sentence: {s['text']}"
    return s["text"]

def fmt_title_section(s, title, abstract):
    parts = [p for p in [title, s["heading"], s["text"]] if p]
    return " | ".join(parts)

def fmt_title_section_position(s, title, abstract):
    position = f"Para {s['pidx']} Sent {s['sidx']}"
    parts = [p for p in [title, s["heading"], position, s["text"]] if p]
    return " | ".join(parts)

def fmt_full_rich(s, title, abstract):
    snippet = (abstract[:150] + "...") if len(abstract) > 150 else abstract
    position = f"Para {s['pidx']} Sent {s['sidx']}"
    parts = [p for p in [snippet, title, s["heading"], position, s["text"]] if p]
    return " | ".join(parts)

def fmt_section_only(s, title, abstract):
    parts = [p for p in [s["heading"], s["text"]] if p]
    return " | ".join(parts)


SENTENCE_FORMATS = {
    "plain":             fmt_plain,
    "title_only":        fmt_title_only,
    "title+section":     fmt_title_section,
    "title+section+pos": fmt_title_section_position,
    "full_rich":         fmt_full_rich,
    "section_only":      fmt_section_only,
}

# --- Paragraph-level context formats ---

def fmt_para_plain(p, title, abstract):
    return p["text"]

def fmt_para_title(p, title, abstract):
    if title:
        return f"Title: {title} Paragraph: {p['text']}"
    return p["text"]

def fmt_para_title_section(p, title, abstract):
    parts = [x for x in [title, p["heading"], p["text"]] if x]
    return " | ".join(parts)

def fmt_para_full(p, title, abstract):
    snippet = (abstract[:150] + "...") if len(abstract) > 150 else abstract
    parts = [x for x in [snippet, title, p["heading"], p["text"]] if x]
    return " | ".join(parts)


PARAGRAPH_FORMATS = {
    "plain":         fmt_para_plain,
    "title_only":    fmt_para_title,
    "title+section": fmt_para_title_section,
    "full_rich":     fmt_para_full,
}


def run_sentences(qa, papers, qrels_sent, title_map, abstract_map):
    sent_corpus = build_sentence_corpus(papers)
    gold_qrels = build_sent_qrels(qrels_sent)
    questions = [
        q for q in qa
        if q["answerable_mapped"]
        and q["question_id"] in gold_qrels
        and q["paper_id"] in sent_corpus
    ]
    print(f"Evaluating {len(questions)} questions at sentence level...")
    all_results = {fmt: [] for fmt in SENTENCE_FORMATS}
    for q in tqdm(questions, desc="Approach 3 sentences"):
        paper_id = q["paper_id"]
        sentences = sent_corpus[paper_id]
        title = title_map.get(paper_id, "")
        abstract = abstract_map.get(paper_id, "")
        gold_ids = gold_qrels[q["question_id"]]
        sent_ids = [f"{s['pidx']}/{s['sidx']}" for s in sentences]
        query_tokens = tokenize(q["question"])
        for fmt_name, fmt_fn in SENTENCE_FORMATS.items():
            docs = [fmt_fn(s, title, abstract) for s in sentences]
            bm25 = BM25Okapi([tokenize(d) for d in docs])
            scores = bm25.get_scores(query_tokens)
            ranked = [sent_ids[i] for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)]
            all_results[fmt_name].append({"ranked_ids": ranked, "gold_ids": gold_ids})
    return {fmt: evaluate(results) for fmt, results in all_results.items()}


def run_paragraphs(qa, papers, qrels_para, title_map, abstract_map):
    para_corpus = build_paragraph_corpus(papers)
    gold_qrels = build_para_qrels(qrels_para)
    questions = [
        q for q in qa
        if q["answerable_mapped"]
        and q["question_id"] in gold_qrels
        and q["paper_id"] in para_corpus
    ]
    print(f"Evaluating {len(questions)} questions at paragraph level...")
    all_results = {fmt: [] for fmt in PARAGRAPH_FORMATS}
    for q in tqdm(questions, desc="Approach 3 paragraphs"):
        paper_id = q["paper_id"]
        paragraphs = para_corpus[paper_id]
        title = title_map.get(paper_id, "")
        abstract = abstract_map.get(paper_id, "")
        gold_ids = gold_qrels[q["question_id"]]
        para_ids = [str(p["pidx"]) for p in paragraphs]
        query_tokens = tokenize(q["question"])
        for fmt_name, fmt_fn in PARAGRAPH_FORMATS.items():
            docs = [fmt_fn(p, title, abstract) for p in paragraphs]
            bm25 = BM25Okapi([tokenize(d) for d in docs])
            scores = bm25.get_scores(query_tokens)
            ranked = [para_ids[i] for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)]
            all_results[fmt_name].append({"ranked_ids": ranked, "gold_ids": gold_ids})
    return {fmt: evaluate(results) for fmt, results in all_results.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["paragraph", "sentence", "both"], default="both")
    args = parser.parse_args()

    print("Loading PeerQA...")
    qa, papers, qrels_para, qrels_sent = load_peerqa()
    title_map = build_title_map(papers)
    abstract_map = build_abstract_map(papers)

    all_results = {}

    if args.level in ("sentence", "both"):
        print("\n--- Sentence Level ---")
        sent_results = run_sentences(qa, papers, qrels_sent, title_map, abstract_map)
        for fmt, metrics in sent_results.items():
            print_results(f"[Sentence] {fmt}", metrics)
        all_results["sentence"] = sent_results

    if args.level in ("paragraph", "both"):
        print("\n--- Paragraph Level ---")
        para_results = run_paragraphs(qa, papers, qrels_para, title_map, abstract_map)
        for fmt, metrics in para_results.items():
            print_results(f"[Paragraph] {fmt}", metrics)
        all_results["paragraph"] = para_results

    from pathlib import Path
    out_path = Path("results") / f"approach3_{args.level}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\n" + "="*65)
    print("SUMMARY TABLE")
    print("="*65)
    for level, level_results in all_results.items():
        print(f"\n  [{level.upper()}]")
        print(f"  {'Format':<30} {'MRR':>8} {'Recall@10':>12}")
        print(f"  {'-'*52}")
        for fmt, m in level_results.items():
            marker = " <- baseline" if fmt == "title_only" else ""
            print(f"  {fmt:<30} {m['MRR']:>8.4f} {m['Recall@10']:>12.4f}{marker}")


if __name__ == "__main__":
    main()