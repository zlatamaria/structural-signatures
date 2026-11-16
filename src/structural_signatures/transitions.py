"""Changes in question-level contribution to net repairs across adjacent prefixes."""

import numpy as np
import pandas as pd

EVENTS = {
    "New repair": 1,
    "Harm resolved": 1,
    "Majority caught up": -1,
    "Structural repair disappeared": -1,
    "Majority became correct": -1,
    "Structural drift": -1,
    "Repair became harm": -2,
    "Harm became repair": 2,
}


def decompose(predictions):
    work = predictions.sort_values(["id", "sample_size"]).copy()
    if work.duplicated(["id", "sample_size"]).any():
        raise ValueError(
            "Select one configuration and metric before decomposing transitions"
        )
    previous = work.groupby("id")[
        ["sample_size", "prediction_correct", "majority_correct"]
    ].shift()
    adjacent = previous.sample_size.notna()
    if not ((work.sample_size[adjacent] - previous.sample_size[adjacent]) == 1).all():
        raise ValueError("Transition analysis requires consecutive budgets")
    old = previous.prediction_correct - previous.majority_correct
    new = work.prediction_correct.astype(int) - work.majority_correct.astype(int)
    structural = work.prediction_correct.eq(1)
    previous_structural = previous.prediction_correct.eq(1)
    conditions = [
        (old == 0) & (new == 1),
        (old == -1) & (new == 0),
        (old == 1) & (new == 0) & structural,
        (old == 1) & (new == 0) & ~structural,
        (old == 0) & (new == -1) & ~previous_structural,
        (old == 0) & (new == -1) & previous_structural,
        (old == 1) & (new == -1),
        (old == -1) & (new == 1),
    ]
    event = pd.Series(np.select(conditions, list(EVENTS), default=""), index=work.index)
    weight = event.map(EVENTS).fillna(0).astype(int)
    if not np.array_equal(weight[adjacent], (new - old)[adjacent]):
        raise ValueError("Event weights do not reconcile to the change in net repairs")
    counts = pd.crosstab(work.sample_size, event).reindex(
        columns=list(EVENTS), fill_value=0
    )
    return counts, counts.mul(pd.Series(EVENTS), axis="columns")
