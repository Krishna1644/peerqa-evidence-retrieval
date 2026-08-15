import json
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))

from retriever import query2doc_concat, rank_paragraphs  # noqa: E402


class RankerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        payload = json.loads((ROOT / "demo" / "examples.json").read_text(encoding="utf-8"))
        cls.examples = {item["id"]: item for item in payload["examples"]}

    def test_query2doc_repeats_question(self):
        out = query2doc_concat("hello world", "pseudo", n_repeat=5)
        self.assertTrue(out.startswith("hello world hello world"))
        self.assertTrue(out.endswith("pseudo"))

    def test_query2doc_lifts_gold_on_quasi_newton(self):
        ex = self.examples["quasi-newton"]
        gold = set(ex["gold_ids"])
        orig = rank_paragraphs(ex["question"], ex["paragraphs"])
        q2d = rank_paragraphs(
            query2doc_concat(ex["question"], ex["pseudo_doc"]),
            ex["paragraphs"],
        )
        orig_rank = next(i for i, row in enumerate(orig, 1) if row["id"] in gold)
        q2d_rank = next(i for i, row in enumerate(q2d, 1) if row["id"] in gold)
        self.assertEqual(q2d_rank, 1)
        self.assertLessEqual(q2d_rank, orig_rank)


if __name__ == "__main__":
    unittest.main()
