# Approach 2 & Approach 3 — Results and Notes

This document summarizes **dense retrieval (Approach 2)** and **BM25 with structural document context (Approach 3)** on your PeerQA runs. Metrics come from `results_approach2_paragraph.json` and `results_approach3_both.json` unless stated otherwise.

---

## Shared evaluation setup

Both approaches use the same helpers in `data_utils.py`:

- **Corpus**: Hugging Face `UKPLab/PeerQA` (papers + qrels).
- **Questions**: `answerable_mapped` is true, the question appears in qrels, and the paper exists in the built corpus.
- **Metrics**: mean reciprocal rank (MRR) and Recall@10 (binary per query: whether any gold hit appears in the top 10).
- **Subset size**: **136 queries** (see `n_queries` in Approach 3 and per-fold counts in Approach 2). This is smaller than the full PeerQA test split used in the original paper, so numbers are **not directly comparable** to published PeerQA leaderboard values.

`python compare_results.py` runs successfully with these files and lists the same MRR / Recall@10 values in its tables.

---

## Verification checklist

| Check | Outcome |
|--------|---------|
| JSON schema vs scripts | `results_approach2_paragraph.json` matches `approach2_dense_retriever.py` (`zero_shot`, `finetuned`, `per_fold`). `results_approach3_both.json` matches `approach3_structural_context.py` (`sentence` / `paragraph` × format names). |
| Query count | Approach 3: all cells report `n_queries: 136`. Approach 2 CV: `27 + 27 + 27 + 27 + 28 = 136` validation questions across folds. |
| Consistency with BM25 baseline | Paragraph **plain** and **title_only** in Approach 3 match `results_baseline_paragraph.json` exactly (MRR 0.3467 / 0.3471; Recall@10 0.6471 / 0.6765), as expected because Approach 3’s `plain` / `title_only` formats are the same indexing strategy as the baseline. |
| Reproducibility | Approach 2 uses fixed seeds (`SEED = 42` in code). Re-running on the same data should match unless library or hardware nondeterminism changes. |

---

## Approach 2 — Dense retriever (`all-MiniLM-L6-v2`)

### What it does

1. **Corpus text**: For paragraph level, each document is the **paragraph body only** (no title or heading prepended). The BM25 baseline’s “titled” variant instead prepends `Title: … Paragraph: …` to each **document** string before tokenization (`baseline_bm25.py`).
2. **Zero-shot**: Encode each question and all paragraphs in that paper with the pretrained MiniLM model, cosine similarity via normalized dot product, rank by score.
3. **Fine-tuned (5-fold)**: Questions are shuffled once, then split into 5 folds. For each fold, **train** on (question, gold paragraph text) pairs from the other four folds using **MultipleNegativesRankingLoss** (3 epochs, batch 16), **validate** on the held-out fold, then average metrics across folds. The reported **fine-tuned** line is the **mean** of the five fold-level scores; `MRR_std` is the standard deviation across folds.

### Results (paragraph level only — file on disk)

| Setting | MRR | Recall@10 | Notes |
|---------|-----|-----------|--------|
| Zero-shot | **0.3853** | **0.7059** | Full 136-query evaluation. |
| Fine-tuned (5-fold mean) | **0.4012** | **0.6989** | `MRR_std` = **0.0744** (high variance across folds). |

### Per-fold validation (for context)

| Fold | MRR | Recall@10 | `n_queries` |
|------|-----|-----------|-------------|
| 1 | 0.4774 | 0.7778 | 27 |
| 2 | 0.4349 | 0.7407 | 27 |
| 3 | 0.2596 | 0.5926 | 27 |
| 4 | 0.4259 | 0.7407 | 27 |
| 5 | 0.4084 | 0.6429 | 28 |

Fold 3 is much weaker than the others, which drives the high `MRR_std`. That often happens when easy/hard questions cluster after a single shuffle, or when the small validation slice is noisy.

### How to read it

- Dense retrieval **clearly beats** BM25+title on this subset at paragraph level (baseline titled MRR 0.3471 vs 0.3853 zero-shot and 0.4012 fine-tuned mean).
- Fine-tuning **improves mean MRR** slightly vs zero-shot; Recall@10 is essentially unchanged.
- Sentence-level Approach 2 was **not** saved in this repo (`results_approach2_sentence.json` absent), so those numbers are not documented here.

---

## Approach 3 — BM25 with structural context strings

### What it does

Still **BM25** (`rank_bm25`), but each **candidate document** (sentence or paragraph) is replaced by a **string template** that can include title, section heading (`last_heading`), optional position tokens, and for `full_rich` a short “abstract” snippet. The **query** remains the reviewer question; only the **indexed document text** changes per format.

Templates are defined in `approach3_structural_context.py` (`SENTENCE_FORMATS`, `PARAGRAPH_FORMATS`). For each question and format, the code tokenizes the question, builds BM25 over all sentences (or paragraphs) of that paper in that format, and scores.

**Implementation detail:** `build_abstract_map` in `data_utils.py` stores the **first** `sentence`-type row encountered per `paper_id` as the “abstract” string. Depending on PeerQA row order, that may be an abstract-like span or simply the start of the paper body. Treat `full_rich` as “early-paper snippet + rich metadata + text,” not necessarily a curated abstract field.

### Results — paragraph level

| Format | MRR | Recall@10 |
|--------|-----|-----------|
| plain | 0.3467 | 0.6471 |
| title_only | 0.3471 | 0.6765 |
| **title+section** | **0.3528** | 0.6765 |
| full_rich | 0.3434 | 0.6618 |

**Takeaway:** Concatenating **title** and **section heading** with the paragraph text gives the **best MRR** among paragraph formats here. `full_rich` adds a leading snippet and slightly **hurts** MRR vs `title+section`, possibly because the extra tokens add noise for BM25 or the snippet is not a clean abstract.

### Results — sentence level

| Format | MRR | Recall@10 |
|--------|-----|-----------|
| plain | 0.2388 | 0.3897 |
| title_only | 0.2492 | 0.3750 |
| title+section | 0.2492 | 0.4118 |
| **title+section+pos** | **0.2544** | 0.4412 |
| full_rich | 0.2453 | **0.4632** |
| section_only | 0.2495 | 0.4191 |

**Takeaway:** Best **MRR** is **`title+section+pos`** (title, heading, “Para p Sent s”, sentence). Best **Recall@10** is **`full_rich`** at the cost of lower MRR — richer strings surface more gold hits in the top 10 but not necessarily at rank 1.

---

## Suggested one-line descriptions (for a report)

- **Approach 2**: Cross-encoder-style training is approximated with a **bi-encoder** (MiniLM); in-domain fine-tuning with multiple negatives improves mean paragraph MRR on a 5-fold split, with noticeable fold-to-fold variance.
- **Approach 3**: **Lexical retrieval** gains from **denormalizing** each candidate into a pseudo-document that carries **citation-relevant metadata** (title, section, position), improving ranking especially when that metadata overlaps question wording.

---

## Files referenced

| File | Role |
|------|------|
| `approach2_dense_retriever.py` | Approach 2 implementation |
| `approach3_structural_context.py` | Approach 3 implementation |
| `results_approach2_paragraph.json` | Saved Approach 2 metrics |
| `results_approach3_both.json` | Saved Approach 3 metrics |
| `data_utils.py` | Loading, qrels, MRR / Recall@10 |
