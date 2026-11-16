from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from structural_signatures.cli import main
from structural_signatures.config import PRESETS
from structural_signatures.legacy import audit_answer_local_prefix_allpairs_v8 as base
from structural_signatures.replay import check_coverage, load_scores, select
from structural_signatures.scoring import FrozenScores, LegacyScores
from structural_signatures.traces import build_question, prepare

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "data/example"
CFG = PRESETS["historical_w3"]


def sorted_frame(table, columns, keys):
    return table[columns].sort_values(keys).reset_index(drop=True)


def test_upstream_and_selector_on_real_repair(tmp_path):
    stages = tmp_path / "graphs"
    main(
        [
            "build",
            "--traces",
            str(EXAMPLE / "q315_traces.csv.gz"),
            "--scores",
            str(EXAMPLE / "q315_scores.csv.gz"),
            "--budgets",
            "5-25",
            "--out",
            str(stages),
        ]
    )
    for name in ["answers", "precedence"]:
        read = {"dtype": {"answer": str, "id": str}, "keep_default_na": False}
        expected = pd.read_csv(EXAMPLE / f"q315_expected_{name}.csv.gz", **read)
        actual = pd.read_csv(stages / f"{name}.csv.gz", **read)
        keys = ["id", "sample_size", "answer"]
        if name == "precedence":
            keys.extend(["from_community", "to_community"])
        pd.testing.assert_frame_equal(
            sorted_frame(expected, expected.columns, keys),
            sorted_frame(actual, expected.columns, keys),
            check_dtype=False,
            rtol=0,
            atol=1e-12,
        )
    output = tmp_path / "selection"
    main(
        [
            "replay",
            "--answers",
            str(stages / "answers.csv.gz"),
            "--precedence",
            str(stages / "precedence.csv.gz"),
            "--budgets",
            "5-25",
            "--preset",
            "historical_w3",
            "--expected-ids",
            "1",
            "--out",
            str(output),
        ]
    )
    pred = pd.read_csv(
        output / "predictions.csv.gz",
        dtype={"predicted_answer": str, "majority_answer": str},
    )
    row = pred[pred.sample_size == 25].iloc[0]
    assert (
        row.predicted_answer,
        row.majority_answer,
        row.selected_raw_count,
        row.majority_raw_count,
    ) == ("41", "1", 6, 7)
    assert row.repair == 1
    raw = pd.read_csv(EXAMPLE / "q315_traces.csv.gz")
    order = pd.read_csv(stages / "trace_order.csv.gz")
    assert (
        len(order) == len(raw)
        and order.retained_position.isna().sum() == raw.final_answer_new.isna().sum()
    )


def test_future_traces_do_not_change_earlier_graphs():
    full = pd.read_csv(EXAMPLE / "q315_traces.csv.gz", dtype={"final_answer_new": str})
    short = full.iloc[: full.final_answer_new.dropna().index[14] + 1]
    scorer = FrozenScores(EXAMPLE / "q315_scores.csv.gz")

    def graph(frame):
        rows = []
        for candidate, edges in build_question(
            prepare("315", frame), CFG, [15], scorer
        ):
            # Archived any-hit labels use all retained traces for each answer.
            candidate.pop("is_correct_answer")
            rows.append((candidate, edges))
        return rows

    assert graph(full) == graph(short)


def test_original_clustering_and_precedence_invariants():
    base.self_test()


def test_missing_budget_and_absent_graph(tmp_path):
    example = pd.read_csv(
        EXAMPLE / "q315_expected_answers.csv.gz", dtype={"answer": str}
    )
    with pytest.raises(ValueError, match="coverage"):
        check_coverage(example[example.sample_size != 12], range(5, 26), 1)
    empty = tmp_path / "edges.csv"
    pd.read_csv(EXAMPLE / "q315_expected_precedence.csv.gz", nrows=0).to_csv(
        empty, index=False
    )
    scores = load_scores(
        EXAMPLE / "q315_expected_answers.csv.gz", empty, CFG, range(5, 26), 1
    )
    assert scores.excess_edges.eq(0).all()
    pred, _ = select(scores, CFG)
    assert pred.predicted_answer.equals(pred.majority_answer)
    with pytest.raises(ValueError, match="threshold"):
        edges = pd.read_csv(EXAMPLE / "q315_expected_precedence.csv.gz")
        edges["precedence_min_consistency"] = 0.95
        edges.to_csv(empty, index=False)
        load_scores(
            EXAMPLE / "q315_expected_answers.csv.gz", empty, CFG, range(5, 26), 1
        )


def test_cache_direction_and_missing_key():
    source = FrozenScores(EXAMPLE / "q315_scores.csv.gz")
    assert source.pair_key("B", "A") == source.pair_key("A", "B")
    source.rows[source.pair_key("A", "B")] = (2.5, -1.5)
    a, b = source.directional_logits([("A", "B"), ("B", "A")])
    np.testing.assert_array_equal(a, [2.5, -1.5])
    np.testing.assert_array_equal(b, [-1.5, 2.5])
    with pytest.raises(KeyError, match="absent"):
        source.directional_logits([("unknown", "pair")])


def test_sqlite_shard_is_read_only(tmp_path):
    import sqlite3

    path = tmp_path / "historical.sqlite"
    fixture = FrozenScores(EXAMPLE / "q315_scores.csv.gz")
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE directional_logits (pair_key TEXT PRIMARY KEY, canonical_left_to_right REAL, canonical_right_to_left REAL)"
        )
        db.execute(
            "INSERT INTO directional_logits VALUES (?, ?, ?)",
            (fixture.pair_key("A", "B"), 1.5, -2),
        )
    source = LegacyScores(path)
    try:
        a, b = source.directional_logits([("B", "A")])
        assert (a[0], b[0]) == (-2, 1.5)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            source.connection.execute("DELETE FROM directional_logits")
    finally:
        source.close()
