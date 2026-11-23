import unittest
import numpy as np
from clap_textspan.span import fit, projection_scores
from clap_textspan.data import cohort, resampled_length


class Correctness(unittest.TestCase):
    def test_span_matches_explicit_residual_greedy(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(50, 12))
        bank = rng.normal(size=(40, 12))
        fitted = fit(x, bank, k=5, subspace=np.eye(12))
        r = x - x.mean(0)
        d = bank.copy()
        indices = []
        for _ in range(5):
            scores = np.std(r @ d.T, axis=0, ddof=1)
            j = int(np.argmax(scores))
            indices.append(j)
            q = d[j] / np.linalg.norm(d[j])
            r -= np.outer(r @ q, q)
            d -= np.outer(d @ q, q)
        self.assertEqual(indices, fitted["indices"].tolist())
        q = fitted["basis"]
        self.assertTrue(np.all(np.diff(fitted["explained"]) >= -1e-12))
        np.testing.assert_allclose(q.T @ q, np.eye(5), atol=1e-10)

    def test_projection_scores_are_signed_and_nested(self):
        x = np.array([[-2.0, 1.0], [3.0, -4.0]])
        scores = projection_scores(x, np.eye(2), np.eye(2))
        np.testing.assert_array_equal(scores[0], np.zeros((2, 2)))
        np.testing.assert_array_equal(scores[-1], x)

    def test_frozen_split_is_source_disjoint(self):
        fit, evaluation, keep, _ = cohort()
        sources = {r["source_id"] for r in fit}
        self.assertEqual(int(keep.sum()), 7962)
        self.assertTrue(
            all(r["source_id"] not in sources for r, k in zip(evaluation, keep) if k)
        )
        self.assertEqual(
            len({r["source_id"] for r, k in zip(evaluation, keep) if k}), 1228
        )

    def test_round_half_up(self):
        self.assertEqual(resampled_length(1, 96000), 1)
        self.assertEqual(resampled_length(44100, 44100), 48000)

    def test_frozen_gaussian_matches_seed_and_magnitude(self):
        from clap_textspan import ROOT

        with np.load(ROOT / "assets/text-embeddings.npz") as z:
            np.testing.assert_array_equal(
                z["gaussian"],
                np.random.default_rng(4503)
                .standard_normal((800, 512))
                .astype(np.float32),
            )
            np.testing.assert_allclose(np.linalg.norm(z["llm"], axis=1), 1, atol=2e-7)

    def test_paper_table_and_class_groups_trace_to_frozen_descriptions(self):
        import csv, json
        from clap_textspan import ROOT

        def rows(name):
            with (ROOT / name).open() as f:
                return list(csv.DictReader(f))

        descriptions = rows("reference/descriptions.csv")
        by_rank = {
            (int(r["block"]), int(r["head"]), int(r["rank"])): r for r in descriptions
        }
        for head in rows("reference/selected-heads.csv"):
            for k in range(1, 6):
                self.assertEqual(
                    head[f"rank{k}"],
                    by_rank[int(head["block"]), int(head["head"]), k]["text"],
                )
        mapping = {
            m["phrase_id"]: m["urbansound_labels"]
            for m in json.loads((ROOT / "assets/mapping.json").read_text())["mappings"]
        }
        groups = json.loads((ROOT / "assets/groups.json").read_text())["groups"]
        for group in groups:
            predicted = [
                h
                for h in range(64)
                if any(
                    group["label"] in mapping[by_rank[h // 32, h % 32, k]["phrase_id"]]
                    for k in range(1, 4)
                )
            ]
            self.assertEqual(group["heads"], predicted)
            for control in group["random_heads"]:
                self.assertEqual(len(control), len(predicted))
                self.assertTrue(set(control).isdisjoint(predicted))


if __name__ == "__main__":
    unittest.main()
