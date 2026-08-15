"""Interactive demo for scientific evidence retrieval.

Run from the repo root:
  streamlit run demo/app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))

from retriever import query2doc_concat, rank_paragraphs  # noqa: E402

EXAMPLES_PATH = ROOT / "demo" / "examples.json"
HEADLINE_PATH = ROOT / "results" / "headline.json"


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def gold_rank(ranked: list[dict], gold_ids: set[str]) -> int | None:
    for i, row in enumerate(ranked, start=1):
        if row["id"] in gold_ids:
            return i
    return None


def render_ranked(ranked: list[dict], gold_ids: set[str], key_prefix: str) -> None:
    for i, row in enumerate(ranked, start=1):
        is_gold = row["id"] in gold_ids
        label = f"#{i}  ·  {row['heading']}  ·  score {row['score']:.3f}"
        if is_gold:
            label += "  ·  GOLD EVIDENCE"
        with st.expander(label, expanded=is_gold or i == 1):
            st.write(row["text"])


def maybe_generate_pseudo_doc(question: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=220,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You write a short passage that might appear in a scientific paper "
                        "and would contain the answer to a reviewer question. Use technical "
                        "terms. Output only the passage."
                    ),
                },
                {"role": "user", "content": question},
            ],
        )
        return completion.choices[0].message.content.strip()
    except Exception as exc:  # pragma: no cover - demo fallback
        st.warning(f"Live Query2Doc failed: {exc}")
        return None


st.set_page_config(
    page_title="PeerQA Evidence Retrieval",
    layout="wide",
)

examples_payload = load_json(EXAMPLES_PATH)
headline = load_json(HEADLINE_PATH)
examples = examples_payload["examples"]
example_map = {item["id"]: item for item in examples}

st.title("Scientific evidence retrieval")
st.caption(
    "Find the paragraph in a paper that answers a peer-review question. "
    "This demo shows Query2Doc + BM25 from our PeerQA project."
)

tabs = st.tabs(["Try it", "Results", "Method"])

with tabs[0]:
    st.markdown(examples_payload["disclaimer"])
    choice = st.selectbox(
        "Example",
        options=[item["id"] for item in examples],
        format_func=lambda eid: example_map[eid]["title"],
    )
    example = example_map[choice]
    gold_ids = set(example["gold_ids"])

    st.subheader("Reviewer question")
    st.write(example["question"])
    st.caption(example["why_it_matters"])

    with st.expander("Query2Doc pseudo-document used for this example"):
        st.write(example["pseudo_doc"])

    orig = rank_paragraphs(example["question"], example["paragraphs"])
    q2d_query = query2doc_concat(example["question"], example["pseudo_doc"])
    q2d = rank_paragraphs(q2d_query, example["paragraphs"])
    orig_rank = gold_rank(orig, gold_ids)
    q2d_rank = gold_rank(q2d, gold_ids)

    m1, m2, m3 = st.columns(3)
    m1.metric("Gold rank · original BM25", f"#{orig_rank}" if orig_rank else "miss")
    m2.metric("Gold rank · Query2Doc BM25", f"#{q2d_rank}" if q2d_rank else "miss")
    delta = None
    if orig_rank and q2d_rank:
        delta = orig_rank - q2d_rank
    m3.metric("Ranks gained", delta if delta is not None else "—")

    left, right = st.columns(2)
    with left:
        st.markdown("**BM25 with the raw question**")
        render_ranked(orig, gold_ids, "orig")
    with right:
        st.markdown("**BM25 with Query2Doc (question × 5 + pseudo-doc)**")
        render_ranked(q2d, gold_ids, "q2d")

    st.divider()
    st.subheader("Custom question on this paper")
    custom = st.text_input(
        "Type a question",
        placeholder="e.g. What optimizer baselines were compared?",
    )
    if custom:
        live_pseudo = maybe_generate_pseudo_doc(custom)
        if live_pseudo:
            st.caption("Used a live gpt-4o-mini pseudo-document because OPENAI_API_KEY is set.")
            with st.expander("Generated pseudo-document"):
                st.write(live_pseudo)
            custom_ranked = rank_paragraphs(
                query2doc_concat(custom, live_pseudo),
                example["paragraphs"],
            )
        else:
            st.caption("No API key set — ranking the custom question with BM25 only.")
            custom_ranked = rank_paragraphs(custom, example["paragraphs"])
        render_ranked(custom_ranked, gold_ids, "custom")

with tabs[1]:
    st.markdown(
        f"**Eval set:** {headline['eval_set']}  ·  **Unit:** {headline['granularity']} retrieval"
    )
    st.caption(headline["note"])
    rows = []
    for item in headline["methods"]:
        rows.append({
            "Method": item["method"],
            "MRR": item["mrr"],
            "Recall@1": item["recall_at_1"],
            "Recall@10": item["recall_at_10"],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    fig_mrr = ROOT / "docs" / "figures" / "mrr.png"
    fig_recall = ROOT / "docs" / "figures" / "recall.png"
    cols = st.columns(2)
    if fig_mrr.exists():
        cols[0].image(str(fig_mrr), caption="Paragraph MRR")
    if fig_recall.exists():
        cols[1].image(str(fig_recall), caption="Recall")
    st.markdown(
        "Full writeup: [final report](../docs/final-report.pdf) · "
        "[Approach 1 notes](../docs/reports/approach1_query2doc.md)"
    )

with tabs[2]:
    pipeline = ROOT / "docs" / "figures" / "pipeline.png"
    if pipeline.exists():
        st.image(str(pipeline), caption="System pipeline")
    st.markdown(
        """
        Reviewer questions and paper paragraphs often do not share vocabulary.
        This project attacks that **lexical gap** with Query2Doc:

        1. Send the reviewer question to an LLM.
        2. Ask it to write a *hypothetical paper passage* (not a paraphrase).
        3. Build the BM25 query as `(question + " ") × 5 + pseudo-document`.
        4. Rank paragraphs *inside the source paper only*.

        Repeating the question 5× is the original Query2Doc trick: it keeps the
        real keywords from being drowned by the longer generated passage.

        **RRF fusion hurt** in our full run (0.4174 vs 0.4432 MRR). When one
        ranker dominates, averaging it with the weaker original ranking pulls
        the gold paragraph down.

        Query2Doc is for *retrieval*, not answering. The corpus-size example
        shows the model inventing a fake word count. That is acceptable for BM25
        term expansion and catastrophic if you treated the pseudo-doc as truth.
        """
    )
