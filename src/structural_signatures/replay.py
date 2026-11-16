"""Evaluate the published metric from compact or full v8 CSV tables."""

import numpy as np
import pandas as pd

from .config import METRIC
from .graphs import maximum_excess, shortcuts
from .legacy import evaluate_saved_shortcut_graph_metrics_consistency as original

KEY = ["id", "sample_size", "answer"]
ANSWER = ["raw_count", "first_trace_position", "is_correct_answer"]
EDGE = [
    "from_community",
    "to_community",
    "cooccurrence_support",
    "ordering_consistency",
    "adjacent_forward_count",
    "nonadjacent_forward_count",
    "precedence_min_consistency",
]


def filtered_chunks(path, fields, cfg, sizes, chunksize):
    columns = [*KEY, *cfg.filters, *fields]
    for chunk in pd.read_csv(
        path,
        usecols=columns,
        dtype={"answer": str, "id": str},
        keep_default_na=False,
        chunksize=chunksize,
    ):
        mask = chunk.sample_size.isin(sizes)
        for column, value in cfg.filters.items():
            if column == "edge_quantile":
                mask &= np.isclose(chunk[column], value)
            else:
                mask &= chunk[column].eq(value)
        if mask.any():
            yield chunk.loc[mask, columns].copy()


def check_coverage(answers, sizes, expected_ids=None):
    if answers.empty or answers.duplicated(KEY).any():
        raise ValueError(
            "Answer table is empty or contains duplicate question/prefix/answer keys"
        )
    if (answers.answer.eq("") | answers.id.eq("")).any():
        raise ValueError("Answer identifiers and question IDs must be nonempty")
    for column in ["raw_count", "first_trace_position"]:
        v = pd.to_numeric(answers[column], errors="raise")
        if (
            not np.isfinite(v).all()
            or (v < int(column == "raw_count")).any()
            or (v % 1 != 0).any()
        ):
            raise ValueError(f"Invalid {column}")
    n = answers.id.nunique()
    coverage = (
        answers.groupby("sample_size").id.nunique().reindex(list(sizes), fill_value=0)
    )
    if (coverage != n).any() or (expected_ids is not None and n != expected_ids):
        raise ValueError(
            f"Incomplete budget coverage: expected {expected_ids or n} questions at every budget; found {coverage.to_dict()}"
        )
    if (
        answers.groupby(["id", "sample_size"])
        .raw_count.sum()
        .reset_index()
        .eval("raw_count > sample_size")
        .any()
    ):
        raise ValueError("Answer frequencies exceed the requested prefix")


def load_scores(
    answer_path, edge_path, cfg, sizes, expected_ids=None, chunksize=160_000
):
    parts = list(filtered_chunks(answer_path, ANSWER, cfg, sizes, chunksize))
    if not parts:
        raise ValueError("No answers match this configuration and budget range")
    answers = pd.concat(parts, ignore_index=True)
    check_coverage(answers, sizes, expected_ids)
    labels = (
        answers.is_correct_answer.astype(str)
        .str.lower()
        .map({"true": True, "1": True, "false": False, "0": False})
    )
    if labels.isna().any():
        raise ValueError("Invalid is_correct_answer label")
    answers["is_correct_answer"] = labels
    hard_parts = []
    for chunk in filtered_chunks(edge_path, EDGE, cfg, sizes, chunksize):
        floor = pd.to_numeric(chunk.precedence_min_consistency)
        if not np.isfinite(floor).all() or (floor > cfg.consistency + 1e-12).any():
            raise ValueError(
                "Requested consistency is below the threshold recorded in the precedence table"
            )
        members = shortcuts(chunk, cfg)
        if not members.empty:
            hard_parts.append(members[[*KEY, "from_community", "to_community"]])
    answers = answers.set_index(KEY)
    answers["excess_edges"] = 0
    if hard_parts:
        edges = pd.concat(hard_parts, ignore_index=True)
        groups = edges.groupby(KEY, sort=False)
        counts = {
            key: maximum_excess(zip(group.from_community, group.to_community))
            for key, group in groups
        }
        values = pd.Series(counts, name="excess_edges")
        if not values.index.isin(answers.index).all():
            raise ValueError(
                "Precedence table contains a graph without a candidate answer"
            )
        answers["excess_edges"] = values.reindex(answers.index, fill_value=0).astype(
            int
        )
    answers[METRIC] = answers.excess_edges / answers.raw_count
    return answers.reset_index()


def select(scores, cfg):
    ordered = scores.sort_values(["id", "answer", "sample_size"], kind="stable").copy()
    rolling = f"{METRIC}__rolling_mean_w{cfg.window}"
    means = (
        ordered.groupby(["id", "answer"], sort=False)[METRIC]
        .rolling(cfg.window, min_periods=1)
        .mean()
    )
    ordered[rolling] = means.droplevel([0, 1]).reindex(ordered.index)
    predictions = original.select_answers(
        ordered, [rolling], cfg.minimum_answer_frequency
    )
    counts = (
        ordered.groupby(["id", "sample_size"]).raw_count.sum().rename("retained_budget")
    )
    predictions = predictions.join(counts, on=["id", "sample_size"])
    return predictions, original.summarize_accuracy(predictions)
