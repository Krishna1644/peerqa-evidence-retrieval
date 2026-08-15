# Approach 2 — Supervised Dense Retriever (Contrastive Fine-Tuning of MiniLM)

> **Headline result**: Supervised contrastive fine-tuning of `all-MiniLM-L6-v2` on PeerQA question–paragraph pairs reaches **MRR 0.3618 (±0.0196)** on 3-fold cross-validation — a **+10.1% relative improvement** over the zero-shot MiniLM baseline (MRR 0.3286), with no external API calls, no additional data, and no proprietary infrastructure.

---

## 1. Executive Summary

This work targets the **semantic gap** in PeerQA evidence retrieval: reviewer questions and paper paragraphs share little surface-level vocabulary, making lexical methods like BM25 weak, but off-the-shelf dense retrievers are also disadvantaged because they are pre-trained on general-domain text (MS-MARCO, NLI) rather than scientific peer review. We bridge this gap by **fine-tuning a sentence transformer directly on labeled PeerQA question–paragraph pairs** using contrastive learning with same-paper hard negatives.

### Final results (n=383 questions with mapped evidence)

| Method                            | MRR    | R@1    | R@5    | R@10   | R@20   |
|-----------------------------------|--------|--------|--------|--------|--------|
| Zero-shot MiniLM (baseline)       | 0.3286 | 0.1570 | 0.3766 | 0.5360 | 0.6929 |
| **Supervised CV (3-fold, MiniLM)**| **0.3618** | **0.1722** | **0.4207** | **0.5754** | **0.7362** |
| *PeerQA paper Contriever (zs)*    | *0.3494* | — | — | — | — |
| *PeerQA paper Dragon+ (best)*     | *0.4845* | — | — | *0.6817* | — |

### Key findings

1. **Supervised fine-tuning on PeerQA labels gives +10.1% relative MRR** over the same model used zero-shot — without any external data, API calls, or additional corpora.
2. **The fine-tuned model surpasses the published Contriever zero-shot baseline (0.3494)**, a larger dense retriever, using only lightweight supervision on MiniLM.
3. **Same-paper hard negatives are critical**: training pairs use paragraphs from the same paper as negatives, which share domain vocabulary and are substantially harder than random corpus negatives. This forces the model to learn fine-grained semantic distinctions, not just domain adaptation.
4. **R@5 and R@20 improve significantly** (+4.4 and +4.3 absolute points), indicating the fine-tuned model systematically pushes relevant paragraphs higher up the ranked list across the board.

---

## 2. Method

### 2.1 Pipeline overview

```
corpus.jsonl ──────────────────────────────┐
qa_pairs.jsonl ──► build_supervised_pairs  │
                       │                   │
                       ▼                   ▼
              (question, gold_para,   per-paper
               hard_negative)         chunk index
                       │
                       ▼
              MultipleNegativesRankingLoss
              all-MiniLM-L6-v2 fine-tuning
              [3-fold cross-validation]
                       │
                       ▼
              embed_and_retrieve (cosine sim)
                       │
                       ▼
              MRR / Recall@{1,5,10,20}
```

A stronger base model such as Contriever or Dragon+ was not fine-tuned due to computational constraints (CPU-only environment). The supervision strategy is model-agnostic; applying it to a larger base would likely yield higher absolute performance.

For each fold of the 3-fold cross-validation:
1. The 383 labeled questions are shuffled (seed=42) and split 2/3 train, 1/3 validation.
2. Training pairs are built from the training split: for each question, every gold paragraph is paired with 1 same-paper hard negative.
3. The base `all-MiniLM-L6-v2` model is fine-tuned for 1 epoch using `MultipleNegativesRankingLoss`.
4. The fine-tuned model encodes all paragraphs for papers in the validation fold, and questions are ranked by cosine similarity.
5. MRR and Recall@{1,5,10,20} are computed on the held-out validation questions. Final reported metrics are averaged across all 3 folds.

### 2.2 Base model

- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
  - 6-layer, 22M parameter MiniLM distilled for sentence embeddings
  - Pre-trained on a large mixture of NLI, MS-MARCO, and paraphrase datasets
  - Produces 384-dimensional L2-normalized embeddings
  - Selected for its speed on CPU (no GPU available in this environment) while maintaining strong zero-shot dense retrieval quality

### 2.3 Supervised contrastive training

**Training pair construction** (`build_supervised_pairs`):

For each training question:
- All paragraphs listed in `relevant_chunk_ids` are used as gold positives
- 1 random paragraph from the **same paper** is sampled as a hard negative
- This yields an `InputExample` with texts `[question, gold_paragraph, hard_negative]`

Fold sizes (after 3-fold split of 383 questions, seed=42):
| Fold | Train pairs | Val questions |
|------|------------|---------------|
| 1    | 442        | 127           |
| 2    | 437        | 127           |
| 3    | 423        | 129           |

**Loss function**: `MultipleNegativesRankingLoss` (MNRL)
- Treats every other (question, paragraph) pair in the batch as an implicit negative
- Combined with the explicit same-paper hard negatives from the `InputExample`, the loss jointly optimizes against both random and hard negatives within each batch
- Particularly well-suited for bi-encoder training where no pre-mined hard negative pool is available

**Training hyperparameters**:
| Parameter | Value |
|-----------|-------|
| Base model | `all-MiniLM-L6-v2` |
| Epochs | 1 |
| Batch size | 16 |
| Warmup steps | 10% of total steps |
| Loss | MultipleNegativesRankingLoss |
| Optimizer | AdamW (HuggingFace Trainer default) |
| Device | CPU |
| Seed | 42 |

Observed training time per fold: **~2 minutes** on CPU.

### 2.4 Retrieval

**Embedding** (`embed_and_retrieve`):
- Documents are encoded per-paper: all paragraph chunks for a paper are batched and encoded together (`batch_size=64`)
- Questions are encoded separately per paper
- Both question and document embeddings are **L2-normalized** before scoring
- Retrieval score is the **dot product of normalized embeddings = cosine similarity**
- Retrieval is **per-paper** (matching the PeerQA evaluation protocol): each question is ranked against only the paragraphs of its source paper (~137 paragraphs/paper on average)

### 2.5 Evaluation

Metrics are computed over the 383 questions with at least one mapped evidence paragraph:
- **MRR** (Mean Reciprocal Rank): primary metric, matches PeerQA paper
- **Recall@{1, 5, 10, 20}**: fraction of questions for which a relevant paragraph appears in the top-k

Final reported metrics for the supervised model are the **mean across 3 held-out validation folds**, with MRR standard deviation reported to quantify cross-fold stability.

---

## 3. Results & Analysis

### 3.1 Per-fold breakdown

| Fold | Train pairs | Val Qs | MRR    | R@1    | R@5    | R@10   | R@20   |
|------|------------|--------|--------|--------|--------|--------|--------|
| 1    | 442        | 127    | 0.3518 | 0.1606 | 0.4354 | 0.5830 | 0.7290 |
| 2    | 437        | 127    | 0.3893 | 0.2023 | 0.4099 | 0.5528 | 0.7468 |
| 3    | 423        | 129    | 0.3444 | 0.1535 | 0.4169 | 0.5903 | 0.7327 |
| **Avg** | — | — | **0.3618** | **0.1722** | **0.4207** | **0.5754** | **0.7362** |
| Std  | — | — | ±0.0196 | — | — | — | — |

The MRR standard deviation of ±0.0196 indicates moderate but expected cross-fold variance given the small labeled dataset (383 questions total). Fold 2 is notably stronger (MRR 0.3893), which we attribute to random variation in which questions are held out — some folds may contain a higher proportion of "easier" questions where the relevant paragraph has high lexical overlap with the question.

### 3.2 Improvement over zero-shot baseline

| Metric | Zero-shot MiniLM | Supervised (3-fold avg) | Δ abs  | Δ rel   |
|--------|-----------------|------------------------|--------|---------|
| MRR    | 0.3286          | 0.3618                 | +0.0332 | +10.1%  |
| R@1    | 0.1570          | 0.1722                 | +0.0152 | +9.7%   |
| R@5    | 0.3766          | 0.4207                 | +0.0441 | +11.7%  |
| R@10   | 0.5360          | 0.5754                 | +0.0394 | +7.4%   |
| R@20   | 0.6929          | 0.7362                 | +0.0433 | +6.2%   |

**R@5 shows the largest relative gain (+11.7%)**, meaning the fine-tuned model is most effective at consolidating relevant paragraphs into the top-5 — the range most useful for a downstream RAG pipeline that passes a few retrieved passages to a generator.

### 3.3 Comparison to published baselines

| Method                  | MRR    | Source |
|------------------------|--------|--------|
| Zero-shot MiniLM        | 0.3286 | This work |
| **Supervised MiniLM CV**| **0.3618** | **This work** |
| Contriever (zero-shot)  | 0.3494 | PeerQA paper |
| Dragon+ (best)          | 0.4845 | PeerQA paper |

Our fine-tuned MiniLM (**0.3618**) exceeds the published zero-shot Contriever baseline (**0.3494**) — a substantially larger model — demonstrating that even a small amount of in-domain supervision on a lightweight model can outperform a larger zero-shot retriever on this task.

The gap to Dragon+ (0.4845) remains large. Dragon+ is a purpose-built dense retriever fine-tuned on large-scale retrieval corpora with progressive training, representing the state of the art on PeerQA.

### 3.4 Why same-paper hard negatives matter

The core design choice of using same-paper paragraphs as hard negatives is motivated by the structure of PeerQA. Reviewer questions are about a specific paper, and the retrieval task is always within that paper. Random negatives from the full 28,496-chunk corpus would be trivially easy to distinguish (different domain, different terminology) and would not teach the model to discriminate between topically-related paragraphs from the same source document. By sampling negatives from the same paper, each training example forces the model to learn: *"this paragraph on method X, not method Y (also from the same paper), is what this question is asking about."*

---

## 4. Implementation Details

### 4.1 Data flow

```
peerqa_data/corpus.jsonl ──────────────────┐
peerqa_data/qa_pairs.jsonl ────────────────┤
                                           ▼
                           step2_approach2_dense_retriever.py
                                 │
                                 ├── [Zero-shot] embed + retrieve → run_zs
                                 │
                                 └── [3-fold CV]
                                       ├── Fold 1: train → eval → metrics_1
                                       ├── Fold 2: train → eval → metrics_2
                                       └── Fold 3: train → eval → metrics_3
                                                         │
                                                         ▼
                                         peerqa_data/approach2_results.json
                                         peerqa_data/approach2_runs.json
```

### 4.2 Corpus statistics (input)

| Statistic | Value |
|-----------|-------|
| Total paragraph chunks | 28,496 |
| Unique papers in corpus | 208 |
| Total QA pairs | 579 |
| QA pairs with mapped evidence (eval set) | 383 |
| Unique papers in eval set | 167 |
| Avg paragraphs per paper (eval) | ~137 |

### 4.3 Run configuration (as executed)

```
python step2_approach2_dense_retriever.py \
    --skip-ict \
    --folds 3 \
    --sup-epochs 1 \
    --save-models
```

Full config recorded in `approach2_results.json`:
```json
{
  "base_model": "sentence-transformers/all-MiniLM-L6-v2",
  "device": "cpu",
  "ict_pairs_used": 0,
  "sup_epochs": 1,
  "sup_batch": 16,
  "n_folds": 3,
  "n_queries_evaluated": 383
}
```

### 4.4 Output files

| File | Contents |
|------|----------|
| `approach2_results.json` | Aggregated metrics (zero-shot, supervised avg, per-fold, config) |
| `approach2_runs.json` | Full per-question rankings for zero-shot and supervised runs |
| `approach2_run_output.txt` | Full terminal output including training loss and per-fold metrics |

---

## 5. Limitations & Future Work

### 5.1 Limitations

1. **Small labeled training set.** With only 383 labeled questions across 3 folds, each fold trains on ~280 questions (~440 pairs). Cross-fold variance (MRR std ±0.0196) is moderate and a larger labeled set would stabilize results.

2. **1 epoch of fine-tuning.** Training for a single epoch was necessary given CPU constraints and time budget. Additional epochs and a learning rate schedule could yield further improvement — the train loss (1.59–1.69) suggests the model has not converged, and more training would likely help.

3. **Single hard negative per example.** Only 1 same-paper negative is sampled per training pair. In practice, mining multiple hard negatives (e.g., top-k retrieved by the zero-shot model that are not relevant) has been shown to improve contrastive training.

4. **CPU-only training.** All training was performed on CPU, limiting batch size and epoch count. GPU training would allow larger batches (stronger in-batch negatives), more epochs, and a larger base model.

5. **MiniLM embedding dimension (384).** Larger models like `all-mpnet-base-v2` (768-dim) or purpose-built retrieval models like Contriever would likely yield higher ceilings, even with the same supervision.

### 5.2 Future work

1. **More training epochs with learning rate warmup/decay.** Even on CPU, 3–5 epochs with proper scheduling would likely push MRR toward 0.39–0.40.

2. **Mined hard negatives.** Run the zero-shot model on training questions, retrieve top-20, filter out relevant, and use top-retrieved non-relevant paragraphs as hard negatives for a second training pass (BM25-negatives or ANN-negatives).

3. **Larger base model.** Swap `all-MiniLM-L6-v2` for `all-mpnet-base-v2` or `msmarco-distilbert-base-v4`. The supervision strategy is model-agnostic and a stronger base should translate directly into better fine-tuned performance.

4. **Combine with Approach 1.** RRF or learned fusion of the supervised dense retriever with Query2Doc-BM25 (Approach 1) is the natural next step. Dense and lexical methods make complementary errors — BM25 fails on semantic paraphrase, dense fails on exact term lookup — and fusion is expected to outperform either component.

5. **Full 5-fold CV with more epochs.** Given sufficient time, a 5-fold CV with 3 supervised epochs would both stabilize the estimate (lower std) and improve the mean MRR.

---

## 6. Numbers to cite in the final report

> **Single-line summary**: *Supervised contrastive fine-tuning of `all-MiniLM-L6-v2` on PeerQA question–paragraph pairs with same-paper hard negatives reaches MRR 0.3618 (±0.0196) on 3-fold CV (n=383), a +10.1% relative improvement over zero-shot MiniLM (0.3286), and surpasses the published PeerQA Contriever zero-shot baseline (0.3494).*

### Key numbers for tables/figures

| Comparison | Δ MRR abs | Δ MRR rel |
|-----------|-----------|-----------|
| Supervised vs. zero-shot MiniLM | +0.0332 | +10.1% |
| Supervised vs. Contriever (published) | +0.0124 | +3.5% |
| Gap to Dragon+ (published) | −0.1227 | −25.4% |

### Bibliography (key references)

1. Reimers, N., & Gurevych, I. (2019). **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.** EMNLP 2019.
2. Henderson, M., et al. (2017). **Efficient Natural Language Response Suggestion for Smart Reply.** arXiv:1705.00652. *(In-batch negatives training strategy.)*
3. Karpukhin, V., et al. (2020). **Dense Passage Retrieval for Open-Domain Question Answering (DPR).** EMNLP 2020.
4. Xiong, L., et al. (2021). **Approximate Nearest Neighbor Negative Contrastive Estimation for Dense Text Retrieval (ANCE).** ICLR 2021.
5. Baumgartner, T., Briscoe, T., & Gurevych, I. (2025). **PeerQA: A Scientific Question Answering Dataset from Peer Reviews.** NAACL 2025 (Outstanding Paper).
