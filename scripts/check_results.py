"""Verify both archived selectors, the published performance curves and Table 1."""

from pathlib import Path

import numpy as np
import pandas as pd
from check_table1 import table1

from structural_signatures.config import PRESETS
from structural_signatures.replay import load_scores, select

ROOT = Path(__file__).resolve().parents[1]


def compare_predictions(actual, saved):
    keys = ["id", "sample_size"]
    left = actual.set_index(keys, verify_integrity=True).sort_index()
    right = saved.set_index(keys, verify_integrity=True).sort_index()
    pd.testing.assert_index_equal(left.index, right.index)
    for col in right:
        if col in ("predicted_score", "majority_score", "score_margin"):
            np.testing.assert_allclose(
                left[col], right[col], rtol=0, atol=1e-12, err_msg=col
            )
        else:
            pd.testing.assert_series_equal(
                left[col], right[col], check_dtype=False, obj=col
            )


def main():
    source = ROOT / "data/math500"
    archive = ROOT / "data/reference"
    scores = load_scores(
        source / "answers.csv.gz",
        source / "precedence.csv.gz",
        PRESETS["paper_w5"],
        range(5, 100),
        expected_ids=500,
    )
    published = pd.read_csv(archive / "figure2_from_pdf.csv").set_index("sample_size")
    for name, cfg in PRESETS.items():
        predictions, curve = select(scores, cfg)
        saved = pd.read_csv(
            archive / f"w{cfg.window}_predictions.csv.gz",
            dtype={"id": str, "predicted_answer": str, "majority_answer": str},
            keep_default_na=False,
            float_precision="round_trip",
        )
        compare_predictions(predictions, saved)
        curve = curve.set_index("sample_size").reindex(published.index)
        match = curve[published.columns].eq(published).all(axis=1)
        if name == "paper_w5" and not match.all():
            raise AssertionError("Mismatch with the published Figure 2")
        print(
            f"{name}: {len(predictions):,} saved predictions match; {int(match.sum())}/95 plotted budgets match"
        )
    print(
        f"Table 1: {len(table1())} rows match the displayed precision with the archived settings and corresponding cohorts"
    )


if __name__ == "__main__":
    main()
