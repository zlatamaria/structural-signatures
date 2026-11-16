from itertools import combinations

import pandas as pd
import pytest

from structural_signatures.config import METRIC, PRESETS
from structural_signatures.graphs import maximum_excess, most_cross_linked, shortcuts
from structural_signatures.replay import select
from structural_signatures.transitions import decompose

CFG = PRESETS["historical_w3"]


def test_most_excess_is_not_most_nodes_and_edges_are_unique():
    tree = [(i, i + 1) for i in range(9)]
    clique = list(combinations([11, 12, 13, 14], 2))
    assert maximum_excess(tree) == 0
    graph, mu = most_cross_linked(tree + clique + [clique[0]])
    assert set(graph) == {11, 12, 13, 14}
    assert mu == maximum_excess(tree + clique + [clique[0]]) == 3


def test_shortcut_support_direction_and_two_realizations():
    frame = pd.DataFrame(
        [
            (4, 2, 5, 0.8, 1, 3),
            (2, 1, 3, 1.0, 3, 0),
            (1, 3, 3, 1.0, 0, 3),
            (3, 1, 4, 0.75, 1, 2),
            (5, 3, 2, 1.0, 1, 1),
        ],
        columns=[
            "from_community",
            "to_community",
            "cooccurrence_support",
            "ordering_consistency",
            "adjacent_forward_count",
            "nonadjacent_forward_count",
        ],
    )
    edges = shortcuts(frame, CFG)
    assert list(zip(edges.from_community, edges.to_community)) == [(4, 2)]


def scores(rows):
    return pd.DataFrame(
        [
            dict(
                id="q",
                sample_size=k,
                answer=a,
                raw_count=n,
                first_trace_position=p,
                is_correct_answer=(a == "a"),
                **{METRIC: s},
                **CFG.filters,
            )
            for k, a, n, p, s in rows
        ]
    )


def test_rolling_starts_at_candidate_appearance():
    frame = scores(
        [
            (5, "a", 3, 0, 0.2),
            (6, "a", 3, 0, 0.2),
            (7, "a", 3, 0, 0.2),
            (7, "b", 2, 4, 0.4),
            (8, "a", 4, 0, 0.2),
            (8, "b", 2, 4, 0),
        ]
    )
    pred, _ = select(frame, CFG)
    assert pred.predicted_answer.tolist() == ["a", "a", "b", "a"]
    assert pred.predicted_score.tolist() == pytest.approx([0.2, 0.2, 0.4, 0.2])


def test_exact_selection_order_and_singletons():
    # Close scores count as ties only in diagnostics.
    both = scores([(5, "a", 3, 0, 1), (5, "b", 2, 3, 1 + 1e-10)])
    result, _ = select(both, CFG)
    assert result.iloc[0].predicted_answer == "b"
    assert result.iloc[0].metric_ties == 2
    # All are eligible when every answer is a singleton.
    alone = scores([(5, "z", 1, 1, 0), (5, "b", 1, 0, 0), (5, "a", 1, 0, 0)])
    result, _ = select(alone, CFG)
    assert (result.iloc[0].predicted_answer, result.iloc[0].majority_answer) == (
        "a",
        "a",
    )


def test_signed_switches_count_twice():
    data = pd.DataFrame(
        {
            "id": ["q"] * 3,
            "sample_size": [5, 6, 7],
            "prediction_correct": [1, 0, 1],
            "majority_correct": [0, 1, 0],
        }
    )
    counts, impacts = decompose(data)
    assert counts.loc[6, "Repair became harm"] == 1
    assert impacts.sum(axis=1).to_list() == [0, -2, 2]
