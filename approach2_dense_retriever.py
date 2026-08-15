import argparse
import json
import random
import numpy as np
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from data_utils import (
    load_peerqa, build_paragraph_corpus, build_sentence_corpus,
    build_para_qrels, build_sent_qrels, evaluate, print_results,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 16
EPOCHS = 3


def embed_and_rank(model, questions, corpus_map, id_fn, text_fn, gold_qrels):
    results = []
    for q in tqdm(questions, desc="Dense retrieval"):
        paper_id = q["paper_id"]
        items = corpus_map[paper_id]
        docs = text_fn(items)
        doc_ids = id_fn(items)
        gold_ids = gold_qrels[q["question_id"]]
        if not docs:
            continue
        q_emb = model.encode(q["question"], convert_to_tensor=True, normalize_embeddings=True)
        d_embs = model.encode(docs, convert_to_tensor=True, normalize_embeddings=True, batch_size=64)
        scores = (q_emb @ d_embs.T).cpu().numpy()
        ranked = [doc_ids[i] for i in np.argsort(scores)[::-1]]
        results.append({"ranked_ids": ranked, "gold_ids": gold_ids})
    return results


def build_training_examples(questions, corpus_map, id_fn, text_fn, gold_qrels):
    examples = []
    for q in questions:
        paper_id = q["paper_id"]
        if paper_id not in corpus_map:
            continue
        items = corpus_map[paper_id]
        docs = text_fn(items)
        doc_ids = id_fn(items)
        gold_ids = gold_qrels.get(q["question_id"], set())
        if not gold_ids:
            continue
        for doc_id, doc_text in zip(doc_ids, docs):
            if doc_id in gold_ids:
                examples.append(InputExample(texts=[q["question"], doc_text]))
    return examples


def kfold_split(items, k, fold):
    n = len(items)
    fold_size = n // k
    val_start = fold * fold_size
    val_end = val_start + fold_size if fold < k - 1 else n
    return items[:val_start] + items[val_end:], items[val_start:val_end]


def run_approach2(qa, papers, qrels, level, n_folds=5, zero_shot_only=False):
    if level == "paragraph":
        corpus_map = build_paragraph_corpus(papers)
        gold_qrels = build_para_qrels(qrels)
        id_fn = lambda items: [str(p["pidx"]) for p in items]
        text_fn = lambda items: [p["text"] for p in items]
    else:
        corpus_map = build_sentence_corpus(papers)
        gold_qrels = build_sent_qrels(qrels)
        id_fn = lambda items: [f"{s['pidx']}/{s['sidx']}" for s in items]
        text_fn = lambda items: [s["text"] for s in items]

    all_questions = [
        q for q in qa
        if q["answerable_mapped"]
        and q["question_id"] in gold_qrels
        and q["paper_id"] in corpus_map
    ]
    print(f"Total questions: {len(all_questions)}")

    print(f"\nLoading base model: {BASE_MODEL}")
    base_model = SentenceTransformer(BASE_MODEL)

    print("\nEvaluating zero-shot...")
    zs_results = embed_and_rank(base_model, all_questions, corpus_map, id_fn, text_fn, gold_qrels)
    zs_metrics = evaluate(zs_results)
    print_results(f"MiniLM Zero-Shot ({level})", zs_metrics)

    if zero_shot_only:
        return {"zero_shot": zs_metrics}

    print(f"\nStarting {n_folds}-fold fine-tuning...")
    fold_metrics = []
    random.shuffle(all_questions)

    for fold in range(n_folds):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        train_qs, val_qs = kfold_split(all_questions, n_folds, fold)
        train_examples = build_training_examples(train_qs, corpus_map, id_fn, text_fn, gold_qrels)
        print(f"  Train: {len(train_examples)} examples, Val: {len(val_qs)} questions")
        if not train_examples:
            continue
        model = SentenceTransformer(BASE_MODEL)
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
        train_loss = losses.MultipleNegativesRankingLoss(model)
        warmup_steps = int(len(train_dataloader) * EPOCHS * 0.1)
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=EPOCHS,
            warmup_steps=warmup_steps,
            show_progress_bar=True,
        )
        val_results = embed_and_rank(model, val_qs, corpus_map, id_fn, text_fn, gold_qrels)
        fold_m = evaluate(val_results)
        print_results(f"Fold {fold+1}", fold_m)
        fold_metrics.append(fold_m)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if fold_metrics:
        finetuned_metrics = {
            "MRR": round(float(np.mean([m["MRR"] for m in fold_metrics])), 4),
            "MRR_std": round(float(np.std([m["MRR"] for m in fold_metrics])), 4),
            "Recall@10": round(float(np.mean([m["Recall@10"] for m in fold_metrics])), 4),
            "n_folds": n_folds,
        }
        print_results(f"MiniLM Fine-tuned {n_folds}-fold avg ({level})", finetuned_metrics)
    else:
        finetuned_metrics = {}

    return {"zero_shot": zs_metrics, "finetuned": finetuned_metrics, "per_fold": fold_metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["paragraph", "sentence"], default="paragraph")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--zero_shot_only", action="store_true")
    args = parser.parse_args()

    print("Loading PeerQA...")
    qa, papers, qrels_para, qrels_sent = load_peerqa()
    qrels = qrels_para if args.level == "paragraph" else qrels_sent

    results = run_approach2(qa, papers, qrels, args.level, args.folds, args.zero_shot_only)

    from pathlib import Path
    out_path = Path("results") / f"approach2_{args.level}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()