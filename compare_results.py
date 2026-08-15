import json
import glob


def load_json(path):
    with open(path) as f:
        return json.load(f)


def print_table(rows, title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")
    print(f"  {'Method':<40} {'MRR':>8} {'Recall@10':>12}")
    print(f"  {'-'*62}")
    for label, mrr, recall in rows:
        print(f"  {label:<40} {mrr:>8.4f} {recall:>12.4f}")


def main():
    rows_para = [
        ("BM25+Title [PeerQA published]",  0.4288, 0.0),
        ("Dragon+ [PeerQA published best]", 0.4845, 0.0),
    ]
    rows_sent = [
        ("BM25+Title [PeerQA published]",  0.3654, 0.0),
        ("MiniLM zero-shot [PeerQA pub.]", 0.3654, 0.0),
    ]

    for path in glob.glob("results/baseline_*.json"):
        level = "paragraph" if "paragraph" in path else "sentence"
        data = load_json(path)
        for key, m in data.items():
            row = (f"[Ours] {key}", m["MRR"], m.get("Recall@10", 0))
            (rows_para if level == "paragraph" else rows_sent).append(row)

    for path in glob.glob("results/approach1*.json"):
        data = load_json(path)
        for key, m in data.items():
            if not isinstance(m, dict) or "MRR" not in m:
                continue
            rows_para.append((f"[A1] {key}", m["MRR"], m.get("Recall@10", 0)))

    for path in glob.glob("results/approach2_*.json"):
        level = "paragraph" if "paragraph" in path else "sentence"
        data = load_json(path)
        for key, m in data.items():
            if isinstance(m, dict) and "MRR" in m:
                row = (f"[A2] {key}", m["MRR"], m.get("Recall@10", 0))
                (rows_para if level == "paragraph" else rows_sent).append(row)

    for path in glob.glob("results/approach3_*.json"):
        data = load_json(path)
        for level, level_data in data.items():
            for fmt, m in level_data.items():
                row = (f"[A3] {fmt}", m["MRR"], m.get("Recall@10", 0))
                (rows_para if level == "paragraph" else rows_sent).append(row)

    print_table(rows_para, "PARAGRAPH-LEVEL RESULTS")
    print_table(rows_sent, "SENTENCE-LEVEL RESULTS")

    print("\n\n% LaTeX table")
    print(r"\begin{table}[h]\centering")
    print(r"\begin{tabular}{lcc}\hline")
    print(r"\textbf{Method} & \textbf{MRR} & \textbf{Recall@10} \\\hline")
    print(r"\multicolumn{3}{l}{\textit{Paragraph Level}} \\")
    for label, mrr, recall in rows_para:
        print(f"{label} & {mrr:.4f} & {recall:.4f} \\\\")
    print(r"\hline\multicolumn{3}{l}{\textit{Sentence Level}} \\")
    for label, mrr, recall in rows_sent:
        print(f"{label} & {mrr:.4f} & {recall:.4f} \\\\")
    print(r"\hline\end{tabular}")
    print(r"\caption{Retrieval results on PeerQA.}\label{tab:results}\end{table}")


if __name__ == "__main__":
    main()
