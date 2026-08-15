# Scientific Evidence Retrieval for PeerQA

Find the paragraph in a scientific paper that answers a peer-review question.

This is a retrieval project on [PeerQA](https://aclanthology.org/2025.naacl-long.22/) (NAACL 2025 Outstanding Paper). Papers average ~12k tokens. Reviewers ask informal questions; authors annotated the evidence. We compare three complementary retrievers: **Query2Doc + BM25**, a **fine-tuned MiniLM dense retriever**, and **structural context** for BM25.

## Headline result

On the **383** mapped-evidence PeerQA questions (paragraph level):

| Method | MRR | Recall@1 | Recall@10 |
|---|---:|---:|---:|
| BM25 (our original-query baseline) | 0.3852 | 0.1929 | 0.6072 |
| **BM25 + Query2Doc (gpt-4o)** | **0.4432** | **0.2500** | **0.6527** |
| BM25 + RRF fusion | 0.4174 | 0.2278 | 0.6323 |
| PeerQA paper BM25 | 0.4288 | — | 0.6817 |
| PeerQA paper Dragon+ (best published) | 0.4845 | — | 0.6817 |

Query2Doc beats the published BM25 baseline on MRR and improves **Recall@1 by +30% relative** over our own BM25. Reciprocal Rank Fusion *hurt*: when one ranker is much stronger, averaging it with the weaker ranking pulls gold evidence down.

![Pipeline](docs/figures/pipeline.png)

## Try the demo (no dataset download)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run demo/app.py
```

The demo uses small self-contained papers so recruiters can rank BM25 vs Query2Doc in the browser. It does **not** redistribute PeerQA paper text. Set `OPENAI_API_KEY` only if you want live pseudo-documents for custom questions.

## Methods

1. **Query2Doc + BM25** (`approach1_query2doc.py`) — LLM writes a hypothetical answer passage; query is `(question × 5) + passage`; Lucene-style BM25 via `bm25s` + Porter stemming. This is the primary contribution in this repo.
2. **Dense retriever** (`approach2_dense_retriever.py`) — `all-MiniLM-L6-v2` zero-shot vs 5-fold contrastive fine-tuning (`MultipleNegativesRankingLoss`).
3. **Structural context** (`approach3_structural_context.py`) — prepend title / section / position / abstract snippet and ablate BM25 formats.

Retrieval is **per paper**: rank candidate paragraphs (or sentences) inside the source document, matching the PeerQA protocol.

## Reproduce the full experiments

PeerQA loads from Hugging Face. Pin `datasets==2.19.0` — newer versions can break this dataset.

```powershell
pip install -r requirements.txt

# Query2Doc (needs data/peerqa JSONL + OPENAI_API_KEY for a fresh run)
python prepare_data.py
python approach1_query2doc.py                  # or --skip-rewrite if passages are cached
python approach1_query2doc.py --dry-run 5      # preview pseudo-docs only

# Hugging Face baselines / dense / context ablations
python baseline_bm25.py --level paragraph
python approach3_structural_context.py --level both
python approach2_dense_retriever.py --level paragraph --zero_shot_only
python approach2_dense_retriever.py --level paragraph --folds 5   # GPU recommended
python compare_results.py
```

Cached Query2Doc passages live at `data/peerqa/query2doc_passages.jsonl` (gitignored). Metrics from completed runs are in `results/`.

The Hugging Face scripts may evaluate a smaller permissive-license slice (**n=136**). Do not mix those numbers with the 383-question Query2Doc table above. See `docs/reports/hf-subset-notes.md`.

## Repo layout

```
demo/                  Streamlit app + ranking helper
approach1_query2doc.py Query2Doc + BM25 (primary method)
approach2_dense_retriever.py
approach3_structural_context.py
baseline_bm25.py
prepare_data.py        Hugging Face PeerQA → JSONL
results/               Saved metrics
docs/                  Report, poster, figures, method writeups
```

## Team

Virginia Tech CS 5624 (Spring 2026):

- **Gopala Krishna Mattaparthi** — Query2Doc / lexical retrieval
- **Hemansh Adunoor** — dense retriever fine-tuning
- **Krishna Inukonda** — structural contextualization

## Writeup

- [Final report (PDF)](docs/final-report.pdf)
- [Poster (PDF)](docs/poster.pdf)
- [Query2Doc notes](docs/reports/approach1_query2doc.md)

## License and data

Code is MIT (see `LICENSE`). PeerQA is [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Download it yourself; this repo does not bundle the papers.

```bibtex
@inproceedings{baumgartner-etal-2025-peerqa,
  title = "{P}eer{QA}: A Scientific Question Answering Dataset from Peer Reviews",
  author = {Baumg{\"a}rtner, Tim and Briscoe, Ted and Gurevych, Iryna},
  booktitle = "NAACL 2025",
  year = "2025"
}
```
