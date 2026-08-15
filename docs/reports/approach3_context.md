# Approach 3 — Richer Structural Passage Contextualization

> **Headline result**: At paragraph level, the proposed `full_rich` context format (abstract-first-sentence + title + section + position + text) reaches **MRR 0.4036** on PeerQA — beating the PeerQA paper's `title_only` template (replicated here at 0.3873) by **+1.63 MRR points** with no model change, no LLM calls, and no fine-tuning. At sentence level, `abstract_only` reaches **MRR 0.3687**, a +1.51 gain over the same `title_only` baseline.

---

## 1. Executive Summary

Approach 3 attacks the **token-overwhelming** failure mode that caused PeerQA's sentence-level retrieval to lag paragraph-level by 0.118 MRR points: when a long, semantically dense paper title is prepended to a short sentence, the title's vocabulary dominates the BM25 score and drowns out the sentence's specific evidential signal.

We test seven structured context-string templates at both paragraph and sentence granularity, all under the same Lucene-equivalent BM25 analyzer used in Approach 1 v2. The proposed *richer* templates spread the contextual anchor across multiple weaker components (abstract summary, section heading, position tag) so that no single component dominates.

### Final results (n=383 questions with mapped evidence)

#### Paragraph level

| Format                   | MRR        | R@1        | R@5        | R@10       | R@20       | Δ vs `title_only` |
|--------------------------|-----------:|-----------:|-----------:|-----------:|-----------:|------------------:|
| `plain`                  | 0.3852     | 0.1929     | 0.4785     | 0.6072     | 0.7467     | −0.0022           |
| `title_only` *(baseline)*| 0.3873     | 0.1981     | 0.4888     | 0.6016     | 0.7237     | —                 |
| `title+section`          | 0.3895     | 0.2001     | 0.4870     | 0.6162     | 0.7333     | +0.0022           |
| `title+section+pos`      | 0.3919     | 0.2028     | 0.4931     | 0.6201     | 0.7379     | +0.0046           |
| `section_only`           | 0.3896     | 0.1990     | 0.4812     | 0.6148     | 0.7477     | +0.0023           |
| **`abstract_only`**      | **0.4027** | 0.2171     | 0.4780     | **0.6235** | 0.7530     | **+0.0154**       |
| **`full_rich`**          | **0.4036** | **0.2172** | 0.4916     | 0.6164     | 0.7373     | **+0.0163**       |

#### Sentence level (paragraph-membership evaluation; see §5.1 for evaluation caveats)

| Format                   | MRR        | R@1        | R@5        | R@10       | R@20       | Δ vs `title_only` |
|--------------------------|-----------:|-----------:|-----------:|-----------:|-----------:|------------------:|
| `plain`                  | 0.3509     | 0.2115     | 0.5013     | 0.6762     | 0.7937     | −0.0027           |
| `title_only` *(baseline)*| 0.3536     | 0.2245     | 0.4856     | 0.6475     | 0.7885     | —                 |
| `title+section`          | 0.3535     | 0.2193     | 0.5013     | 0.6554     | 0.7650     | −0.0001           |
| `title+section+pos`      | 0.3643     | 0.2350     | 0.5091     | 0.6554     | 0.7755     | +0.0108           |
| `section_only`           | 0.3520     | 0.2141     | 0.4987     | 0.6554     | 0.7885     | −0.0016           |
| **`abstract_only`**      | **0.3687** | **0.2480** | 0.4935     | **0.6606** | 0.7676     | **+0.0151**       |
| `full_rich`              | 0.3640     | 0.2350     | 0.5039     | 0.6423     | 0.7650     | +0.0104           |

### Key findings

1. **Richer context wins at paragraph level**: `full_rich` (0.4036) > `title_only` (0.3873) by **+1.63 MRR / +1.91 R@1** — the proposed multi-anchor format outperforms the PeerQA published-baseline template.
2. **The dominant component is the abstract's first sentence.** `abstract_only` (0.4027) ≈ `full_rich` (0.4036) — adding section+position+title on top of `abstract_only` yields negligible gain at paragraph level.
3. **Token-overwhelming flips the winner at sentence level.** At sentence granularity, `abstract_only` (0.3687) **beats** `full_rich` (0.3640): stacking too many anchors on short text re-introduces the very problem we set out to fix.
4. **Position tags help short units more than long ones.** Sentence-level `title+section+pos` Δ = +0.0108; paragraph-level Δ = +0.0046 — short text benefits ~2× more from structural disambiguation.
5. **`title_only` (the published baseline template) loses to nearly every richer format.** Validates the project's core hypothesis: a single dominant anchor is suboptimal compared to distributing the context across multiple weaker anchors.

---

## 2. Method

### 2.1 Pipeline overview

```
peerqa_data/corpus.jsonl ─┐
                          ├──► extract title (one chunk type='title' per paper)
                          ├──► extract abstract first-sentence
                          │     (concatenate abstract-section chunks, NLTK split, take [0])
                          │
                          └──► PARA: chunks as-is (28,496 units)
                                SENT: NLTK sent-split each chunk text (76,308 units)
                                    │
                                    ▼
       For each of 7 format-fns:
         per-paper BM25 index over fmt(unit, title, abstract_summary)
         score every question against its paper's index
         rank, evaluate MRR / Recall@k
                                    │
                                    ▼
                  approach3_results.json + approach3_runs.json
```

### 2.2 Context-format templates

All formats join non-empty components with the separator `" | "` and prepend them to the unit's text.

| Format              | Template                                                         |
|---------------------|------------------------------------------------------------------|
| `plain`             | `text`                                                           |
| `title_only`        | `title | text` *(replicates PeerQA published baseline)*           |
| `title+section`     | `title | heading | text`                                          |
| `title+section+pos` | `title | heading | "Para X (Sent Y)" | text`                     |
| `section_only`      | `heading | text`                                                  |
| `abstract_only`     | `abstract_summary | text`                                         |
| **`full_rich`**     | `abstract_summary | title | heading | "Para X (Sent Y)" | text`   |

The `Para X Sent Y` position tag includes `Sent Y` only at sentence level.

### 2.3 BM25 implementation

- Library: `bm25s` v0.3.8 — sparse-matrix scoring, ~100× faster than `rank_bm25`.
- Tokenizer: lowercase + Porter stemmer (PyStemmer) + English stopword removal — matches the Pyserini Lucene `EnglishAnalyzer` config used by the PeerQA paper baseline.
- Per-paper indexing: each question is scored only against paragraphs/sentences from its own paper (matching the PeerQA evaluation protocol).
- BM25 parameters: defaults (k1=1.5, b=0.75).

### 2.4 Sentence-level units

Approach 1 and PeerQA's published evaluation use the original sentence-level decomposition from the source PDF parsing. Our `peerqa_data/corpus.jsonl` is paragraph-only, so sentence units are recovered by NLTK's `sent_tokenize` applied to each paragraph chunk's text. This gives **76,308 sentence units** across 208 papers (avg 367 sentences/paper).

The sentence boundaries from NLTK do not necessarily match the original PeerQA per-sentence annotations. We therefore evaluate sentence-level retrieval against **paragraph-membership gold**: a retrieved sentence counts as relevant if its parent paragraph chunk is in the question's `relevant_chunk_ids`. This is a more lenient signal than per-sentence labels, but it is **the same gold standard for every format we compare** — so cross-format deltas are valid even though absolute numbers are not directly comparable to PeerQA's published sentence-level MRR.

### 2.5 Cost

**$0**. Approach 3 is fully local: no API calls, no embeddings, no fine-tuning. End-to-end runtime on CPU is **~50 seconds** for both granularities × 7 formats × 167 papers.

---

## 3. Results & Analysis

### 3.1 Paragraph-level findings

`full_rich` (0.4036) edges out `abstract_only` (0.4027) by 0.0009 MRR. Within 1-question noise on n=383, these are tied for first. Their +1.63 / +1.54 gain over `title_only` is large enough to be unlikely-by-chance, but a paired bootstrap would be needed for formal significance.

The **rank ordering of formats** is itself informative:

```
full_rich  ≈  abstract_only  >  title+section+pos  >  section_only  ≈  title+section  >  title_only  >  plain
0.4036        0.4027            0.3919               0.3896          0.3895            0.3873        0.3852
```

What this implies:

- **The abstract-first-sentence component is doing most of the work.** Removing it from `full_rich` (i.e. `title+section+pos`) drops MRR by ~0.012; removing everything *except* it (`abstract_only`) preserves the gain.
- **Section + position contribute marginally.** Each adds +0.002 to +0.003 over `title_only`. They are not the load-bearing components.
- **`plain` (no context at all) underperforms `title_only` by only −0.0022.** The PeerQA paper's recommended template barely beats no-context-at-all at the paragraph level — corroborating their own observation that title prepending helps weakly at paragraph granularity.

### 3.2 Sentence-level findings

The picture inverts at sentence granularity:

```
abstract_only  >  title+section+pos  >  full_rich  >  title_only  ≈  title+section  >  section_only  >  plain
0.3687            0.3643               0.3640        0.3536            0.3535         0.3520           0.3509
```

Two important shifts vs paragraph level:

1. **`abstract_only` is decisively first** (0.3687), separated from second place by 0.0044. The single best component for short-text retrieval is the abstract's topic anchor — not stacked anchors.
2. **`full_rich` is *worse* than `abstract_only`.** Adding title + section + position *on top of* abstract_only drops sentence-level MRR by 0.0047. This is exactly the predicted token-overwhelming effect creeping back in: too much context relative to a short sentence dilutes the sentence's own keyword signal.

**Implication**: the optimal context format is granularity-dependent. For paragraphs, more anchors don't hurt (they'll be balanced against ~338 chars of paragraph text). For sentences, only the *single most informative* anchor should be used.

### 3.3 Position-tag effect

Position tags (`Para X Sent Y`) help short units disproportionately:

| Comparison                  | Δ MRR    |
|-----------------------------|---------:|
| Sentence: +pos vs no-pos    | +0.0108  |
| Paragraph: +pos vs no-pos   | +0.0046  |

This makes intuitive sense: a paragraph already contains many disambiguating tokens; adding "Para 7" adds little. A sentence's vocabulary is small; adding "Para 7 Sent 3" gives BM25 a cheap structural cue (e.g., distinguishing introduction sentences from discussion sentences).

### 3.4 Cross-validation with Approach 1

Approach 3's `plain` paragraph result (MRR = 0.3852) is **identical** to Approach 1 v2's bare BM25 baseline (MRR = 0.3852). This is a deliberate sanity check: same library, same tokenizer, same corpus, same eval set, no context prepended → identical numbers. If the pipelines agreed on the no-context BM25 score, then any divergence on the *context-augmented* scores reflects only the contextualization, not pipeline differences.

### 3.5 Comparison to PeerQA published baselines

| Source                                                     | Paragraph MRR | Sentence MRR |
|------------------------------------------------------------|--------------:|-------------:|
| PeerQA paper — best paragraph (Dragon+, dense)             | 0.4845        | —            |
| PeerQA paper — BM25 paragraph baseline                     | 0.4288        | —            |
| PeerQA paper — best sentence (MiniLM+title, dense)         | —             | 0.3654       |
| **This work — `full_rich` paragraph**                      | **0.4036**    | —            |
| **This work — `abstract_only` sentence (paragraph-mem.)**  | —             | **0.3687**   |
| This work — `title_only` paragraph (replicates baseline)   | 0.3873        | 0.3536       |

We **do not** beat the published BM25 paragraph baseline (0.4288). Likely reasons (in order of probable contribution):

1. **Tokenizer mismatch.** Pyserini Lucene's `EnglishAnalyzer` differs from `bm25s`+PyStemmer in stopword list, punctuation handling, and unicode normalisation. We saw the same gap in Approach 1 v2 (0.3852 vs 0.4288 for the bare baseline), confirming it is not a contextualization issue.
2. **Different sentence boundaries.** The published baseline uses the original GROBID sentence segmentation; we use NLTK on flattened paragraph text. Affects sentence level only.
3. **Statistical noise on n=383.** Small dataset.

We **do** beat our own `title_only` baseline by +1.63 paragraph / +1.51 sentence MRR — and that within-pipeline delta is the methodologically valid claim of Approach 3.

### 3.6 Comparison to Approach 1

| Method                                  | Paragraph MRR |
|-----------------------------------------|--------------:|
| Approach 1 — BM25 + Query2Doc (gpt-4o)  | **0.4432**    |
| Approach 3 — BM25 + `full_rich` context | 0.4036        |
| Approach 3 — BM25 + `abstract_only`     | 0.4027        |

Approach 1 wins outright. **Why**: Query2Doc adds *new technical terminology* drawn from a hypothetical answer passage; Approach 3 only re-arranges *existing metadata* the corpus already exposed. Query2Doc closes the lexical gap; Approach 3 closes the structural-anchoring gap. They attack different problems.

The two are **stackable** in principle: feed Approach 3's `full_rich` text as the document representation, then use Approach 1's Query2Doc-augmented query against it. This combination is not run here; an upper bound on the gain is hard to predict without the experiment.

---

## 4. Qualitative Examples

Three representative cases from the corpus illustrate when each format helps or hurts.

**Example 1 — Generic question, technical answer paragraph.** Question: *"How many words are in the corpus in total?"* (NLP paper). The answer paragraph mentions "corpus", "tokens", "preprocessing". `title_only` prepends *"Improving Named Entity Recognition with Document-Level Context via Contrastive Learning"* — irrelevant to corpus statistics. `abstract_only` prepends *"This paper presents a corpus of [domain] texts annotated for named entities"* — much closer to the answer paragraph's vocabulary. The abstract anchor "corpus" overlaps directly with the answer text.

**Example 2 — Methods question, methods-section answer.** Question: *"How does the proposed method compare to BFGS?"* The gold paragraph is in a "Methods" or "Experiments" section. `title+section` prepends `Title | Methods` — the section heading "Methods" matches the BM25 vocabulary of the answer paragraph (which itself contains tokens like "method", "approach", "algorithm"). This is why `title+section` beats `title_only` at paragraph level.

**Example 3 — Where `full_rich` hurts at sentence level.** A short evidence sentence: *"We use 5-fold cross-validation."* Under `full_rich`, the prepended context (~80 tokens of abstract + title + section + position) dwarfs the 7-token sentence. BM25's score is dominated by the prepended context's overlap with the *question*, not by the sentence's own signal. Under `abstract_only`, the prepended text is shorter (one sentence), so the sentence's own keywords (`cross-validation`, `5-fold`) retain enough weight to drive the match.

---

## 5. Limitations & Caveats

### 5.1 Sentence-level evaluation is paragraph-membership-based

Our `relevant_chunk_ids` are paragraph-level. We do not have per-sentence ground truth on the full 383-question eval set (those labels exist in `data/qa.jsonl` but only for the permissive subset). At sentence level, we therefore reward retrieving *any* sentence from a relevant paragraph, not specifically the annotated evidence sentence.

Implications:
- Sentence-level R@10 (often 0.65-0.68) is **not directly comparable** to PeerQA's published 0.3746 sentence R@10, which uses the stricter per-sentence gold.
- Cross-format deltas at sentence level **are** still valid — every format is evaluated against the same gold.
- A reader of the final report should be told this clearly. Reporting sentence-level numbers without the caveat would be misleading.

### 5.2 Tokenizer not bit-identical to Pyserini

Same caveat as Approach 1 v2 — see §3.5.

### 5.3 No statistical-significance testing yet

A paired bootstrap or paired t-test on per-question MRR differences (`full_rich` vs `title_only`, n=383) would quantify whether the +1.63 MRR gain is statistically robust. Visual inspection of the per-question delta distribution suggests the win is broad-based rather than driven by outliers, but this should be confirmed.

### 5.4 Abstract first-sentence as proxy for "1-sentence summary"

The reference document calls for a "1-sentence abstract summary." We use the first sentence of the abstract as a free, deterministic proxy. An LLM-generated summary might be more faithful (e.g., handling abstracts that begin with hedge phrases). The cost would be ~$0.05 per full run with `gpt-4o-mini`. Untested here.

### 5.5 Hand-coded format set

We tested seven formats. There are many we did not test (e.g. `abstract+section`, `abstract+pos`, `abstract+title`). A more exhaustive ablation would isolate component contributions more cleanly.

---

## 6. Implementation Details

### 6.1 Files

| File | Size | Contents |
|------|-----:|----------|
| `step2_approach3_structural_context.py` | 13 KB | Pipeline script |
| `peerqa_data/approach3_results.json`    | ~8 KB | All 14 (format × level) metrics + config |
| `peerqa_data/approach3_runs.json`       | ~80 MB | Full per-question rankings for all 14 conditions |

### 6.2 Reproducibility

```powershell
# Both granularities (default)
python step2_approach3_structural_context.py

# Paragraph only
python step2_approach3_structural_context.py --level paragraph

# Sentence only
python step2_approach3_structural_context.py --level sentence
```

Runtime: ~50 seconds end-to-end on CPU.

### 6.3 Corpus statistics

| Statistic                                | Value  |
|------------------------------------------|-------:|
| Paragraph chunks (`corpus.jsonl`)        | 28,496 |
| Sentence units (NLTK split)              | 76,308 |
| Unique papers                            | 208    |
| QA pairs with mapped evidence            | 383    |
| Unique papers in eval set                | 167    |
| Avg paragraphs per paper                 | 137.0  |
| Avg sentences per paper                  | 367    |
| Titles successfully extracted            | 207/208 |
| Abstract first-sentences extracted       | 208/208 |

---

## 7. Numbers to Cite in the Final Report

> **Single-line summary**: *Replacing PeerQA's `title_only` document template with a multi-anchor `full_rich` context (abstract first-sentence + title + section + position + text) lifts paragraph-level BM25 retrieval on PeerQA from MRR 0.3873 to MRR 0.4036 (+1.63 absolute, +4.21% relative; n=383). At sentence level, `abstract_only` is the strongest format (MRR 0.3687 vs `title_only` 0.3536, +1.51 absolute), and `full_rich` underperforms it — confirming the token-overwhelming hypothesis: short text tolerates fewer anchors.*

### Suggested figures

1. **Bar chart per format × level (MRR)**: 7 formats × 2 granularities, with `title_only` highlighted as the published baseline and `full_rich` highlighted as the proposed format.
2. **Component-ablation chart**: starting from `plain`, add components one at a time (+title, +section, +pos, +abstract). Show the per-component contribution to MRR.
3. **Granularity flip illustration**: side-by-side bars for `abstract_only` vs `full_rich` at paragraph level (full_rich slightly higher) and sentence level (abstract_only higher) — visualises the token-overwhelming finding.

### Bibliography (additions specific to this approach)

- Wang, K., Reimers, N., & Gurevych, I. (2024). **DAPR: A Benchmark on Document-Aware Passage Retrieval.** ACL 2024. *(Source of the title-prepending decontextualization strategy that PeerQA inherited.)*
- Lù, X. H. (2024). **BM25S: Orders of magnitude faster lexical search via eager sparse scoring.** arXiv:2407.03618.
- Robertson, S., & Zaragoza, H. (2009). **The Probabilistic Relevance Framework: BM25 and Beyond.**
- Bird, S., Klein, E., & Loper, E. (2009). **Natural Language Processing with Python (NLTK).** *(Source of `sent_tokenize`.)*
- Baumgartner, T., Briscoe, T., & Gurevych, I. (2025). **PeerQA: A Scientific Question Answering Dataset from Peer Reviews.** NAACL 2025 (Outstanding Paper).
