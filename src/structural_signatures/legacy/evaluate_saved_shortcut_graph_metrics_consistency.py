#!/usr/bin/env python3
"""
Evaluate graph metrics recoverable from saved answer_metrics and precedence_details.

The script does NOT rerun the reranker, clustering, or checkpoint extraction.
It reconstructs answer-local precedence/shortcut graphs for one fixed experiment
configuration, calculates static and prefix-dynamic graph metrics, selects the
final answer independently with each metric, compares against majority vote,
and saves accuracy/gain plots.

Inputs may be .csv or .csv.gz.

Typical use:
    python -u evaluate_saved_shortcut_graph_metrics.py \
      --answer-metrics answer_metrics_compact.csv.gz \
      --precedence-details precedence_details_compact.csv.gz \
      --output-dir saved_graph_metrics_c5_t3 \
      --community-min-support 5 \
      --transition-min-support 3 \
      --precedence-min-consistency 0.90 \
      --minimum-answer-frequency 2 \
      --sample-sizes 5-99 \
      --rank-sample-sizes 61-99 \
      --required-n-ids 500 \
      --top-n 5

Outputs:
    answer_graph_metrics.csv.gz
    graph_metric_predictions.csv.gz
    graph_metric_accuracy_by_sample_size.csv
    metric_summary_rank_range.csv
    metric_summary_by_range.csv
    top_metrics_accuracy_vs_majority.png
    top_metrics_gain_vs_majority.png
    run_metadata.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **_: Any):
        return iterable


CONFIG_COLUMNS = [
    "score_mode",
    "edge_quantile",
    "cohesion_mode",
    "community_min_support",
    "transition_min_support",
]

ANSWER_KEY_COLUMNS = [
    "id",
    "sample_size",
    *CONFIG_COLUMNS,
    "answer",
]

# These metrics are calculated directly from the hard shortcut graph:
# stable pair + adjacent_count > 0 + nonadjacent_count > 0.
HARD_STATIC_METRICS = [
    "largest_component_mass",
    "largest_component_edges",
    "largest_component_nodes",
    "largest_component_edges_per_trace",
    "largest_component_nodes_per_trace",
    "largest_component_excess_edges",
    "largest_component_excess_edges_per_trace",
    "largest_component_excess_edges_per_node",
    "largest_component_excess_edge_fraction",
    "largest_component_density",
    "largest_component_mean_degree",
    "largest_component_nonbridge_edges",
    "largest_component_nonbridge_edges_per_trace",
    "largest_component_nonbridge_edge_fraction",
    "largest_component_two_core_nodes",
    "largest_component_two_core_edges",
    "largest_component_two_core_edges_per_trace",
    "largest_component_two_core_node_fraction",
    "largest_biconnected_cycle_nodes",
    "largest_biconnected_cycle_edges",
    "largest_biconnected_cycle_edges_per_trace",
    "largest_component_effective_edge_count",
    "largest_component_effective_edge_ratio",
    "largest_component_weighted_excess_mass",
    "largest_component_weighted_excess_mass_fraction",
]

# Existing balance/coexistence/path/tree formulations, all derived from the
# same hard shortcut graph.
WEIGHTED_STATIC_METRICS = [
    "balanced_largest_component_mass",
    "coexistence_largest_component_mass",
    "gap_balanced_largest_component_mass",
    "balanced_effective_edge_count",
    "balanced_effective_edge_evidence",
    "balanced_max_spanning_tree_mass",
    "balanced_weighted_excess_mass",
    "balanced_weighted_excess_mass_fraction",
    "balanced_tree_bottleneck_score",
    "balanced_transitive_reduction_mass",
    "balanced_transitive_reduction_edges",
    "balanced_max_weight_path_mass",
    "balanced_max_weight_path_nodes",
]

# Soft graph: all stable supported precedence pairs are retained. A Jeffreys-
# smoothed adjacent/nonadjacent balance gives one-sided evidence a small rather
# than exactly zero weight. This does not recover pairs removed by the saved
# transition-support threshold.
SOFT_STATIC_METRICS = [
    "soft_largest_component_mass",
    "soft_effective_edge_count",
    "soft_max_spanning_tree_mass",
    "soft_weighted_excess_mass",
    "soft_weighted_excess_mass_fraction",
    "soft_transitive_reduction_mass",
    "soft_max_weight_path_mass",
]

STATIC_METRICS = (
    HARD_STATIC_METRICS
    + WEIGHTED_STATIC_METRICS
    + SOFT_STATIC_METRICS
)

DYNAMIC_BASE_METRICS = [
    "largest_component_excess_edges",
    "largest_component_excess_edges_per_trace",
    "largest_component_weighted_excess_mass",
    "balanced_weighted_excess_mass",
    "soft_weighted_excess_mass",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer-metrics", required=True)
    parser.add_argument("--precedence-details", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--community-min-support", type=int, required=True)
    parser.add_argument("--transition-min-support", type=int, required=True)
    parser.add_argument("--minimum-answer-frequency", type=int, default=2)
    parser.add_argument("--score-mode", default="raw_min")
    parser.add_argument("--edge-quantile", type=float, default=0.90)
    parser.add_argument("--cohesion-mode", default="min")
    parser.add_argument(
        "--precedence-min-consistency",
        type=float,
        default=None,
        help=(
            "Offline consistency threshold used to reconstruct stable/shortcut "
            "edges. When omitted, use the single value stored in the input."
        ),
    )

    parser.add_argument(
        "--sample-sizes",
        default="all",
        help="all, comma-separated values, or inclusive ranges such as 5-99.",
    )
    parser.add_argument(
        "--rank-sample-sizes",
        default="all",
        help=(
            "Sample sizes used to rank top metrics. Curves are still written "
            "for all --sample-sizes. Example: 61-99."
        ),
    )
    parser.add_argument(
        "--metrics",
        default="all",
        help="all, static, or comma-separated metric names.",
    )
    parser.add_argument("--growth-window", type=int, default=5)
    parser.add_argument("--required-n-ids", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--precedence-chunksize", type=int, default=400_000)
    parser.add_argument("--plot-dpi", type=int, default=180)
    return parser.parse_args()


def parse_int_spec(value: str) -> set[int] | None:
    value = value.strip().lower()
    if value == "all":
        return None

    result: set[int] = set()
    for token in re.split(r"[,\s]+", value):
        if not token:
            continue
        match = re.fullmatch(r"(-?\d+)-(-?\d+)", token)
        if match:
            left, right = map(int, match.groups())
            step = 1 if right >= left else -1
            result.update(range(left, right + step, step))
        else:
            result.add(int(token))
    return result


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    text = series.astype(str).str.strip().str.lower()
    mapped = text.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "1.0": True,
            "0.0": False,
        }
    )
    numeric = pd.to_numeric(series, errors="coerce")
    return mapped.where(mapped.notna(), numeric.eq(1)).fillna(False).astype(bool)


def numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
    )


def filter_fixed_config(
    frame: pd.DataFrame,
    *,
    community_min_support: int,
    transition_min_support: int,
    score_mode: str,
    edge_quantile: float,
    cohesion_mode: str,
    sample_sizes: set[int] | None,
) -> pd.DataFrame:
    mask = (
        numeric(frame["community_min_support"], -1).eq(community_min_support)
        & numeric(frame["transition_min_support"], -1).eq(transition_min_support)
        & frame["score_mode"].astype(str).eq(score_mode)
        & numeric(frame["edge_quantile"], np.nan).round(8).eq(round(edge_quantile, 8))
        & frame["cohesion_mode"].astype(str).eq(cohesion_mode)
    )
    if sample_sizes is not None:
        mask &= numeric(frame["sample_size"], -1).astype(int).isin(sample_sizes)
    return frame.loc[mask].copy()


def effective_count(weights: Sequence[float]) -> float:
    values = np.asarray([value for value in weights if value > 0], dtype=float)
    if values.size == 0:
        return 0.0
    denominator = float(np.square(values).sum())
    if denominator <= 0:
        return 0.0
    return float(values.sum() ** 2 / denominator)


def hard_balance(adjacent: float, nonadjacent: float) -> float:
    total = adjacent + nonadjacent
    if total <= 0:
        return 0.0
    return 2.0 * min(adjacent, nonadjacent) / total


def soft_balance(adjacent: float, nonadjacent: float) -> float:
    """Jeffreys-smoothed balance in [0,1]."""
    total = adjacent + nonadjacent
    p = (adjacent + 0.5) / (total + 1.0)
    return float(4.0 * p * (1.0 - p))


def add_or_replace_edge(
    graph: nx.DiGraph,
    source: Any,
    target: Any,
    attributes: dict[str, float],
    preferred_weight: str,
) -> None:
    if source == target:
        return
    if graph.has_edge(source, target):
        if attributes[preferred_weight] > graph[source][target][preferred_weight]:
            graph[source][target].update(attributes)
    else:
        graph.add_edge(source, target, **attributes)


def undirected_graph(
    directed: nx.DiGraph,
    weight_name: str,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(directed.nodes())
    for left, right, attributes in directed.edges(data=True):
        if graph.has_edge(left, right):
            if float(attributes[weight_name]) > float(graph[left][right][weight_name]):
                graph[left][right].update(dict(attributes))
        else:
            graph.add_edge(left, right, **dict(attributes))
    return graph


def maximum_spanning_tree_values(
    graph: nx.Graph,
    weight_name: str,
) -> tuple[float, float]:
    if graph.number_of_edges() == 0:
        return 0.0, 0.0
    tree = nx.maximum_spanning_tree(graph, weight=weight_name)
    weights = [
        float(attributes[weight_name])
        for _, _, attributes in tree.edges(data=True)
    ]
    return (
        float(sum(weights)),
        float(min(weights)) if weights else 0.0,
    )


def greedy_acyclic_graph(
    directed: nx.DiGraph,
    weight_name: str,
) -> nx.DiGraph:
    dag = nx.DiGraph()
    dag.add_nodes_from(directed.nodes())

    edges = sorted(
        directed.edges(data=True),
        key=lambda item: (
            -float(item[2][weight_name]),
            -float(item[2].get("base_weight", 0.0)),
            str(item[0]),
            str(item[1]),
        ),
    )

    for source, target, attributes in edges:
        if nx.has_path(dag, target, source):
            continue
        dag.add_edge(source, target, **dict(attributes))
    return dag


def maximum_weight_path(
    dag: nx.DiGraph,
    weight_name: str,
) -> tuple[float, int]:
    if dag.number_of_edges() == 0:
        return 0.0, 0

    order = list(nx.topological_sort(dag))
    distance = {node: 0.0 for node in order}
    node_count = {node: 1 for node in order}

    for source in order:
        for target, attributes in dag[source].items():
            candidate = distance[source] + float(attributes[weight_name])
            candidate_nodes = node_count[source] + 1
            if (
                candidate > distance[target]
                or (
                    math.isclose(candidate, distance[target])
                    and candidate_nodes > node_count[target]
                )
            ):
                distance[target] = candidate
                node_count[target] = candidate_nodes

    terminal = max(
        order,
        key=lambda node: (distance[node], node_count[node], str(node)),
    )
    return float(distance[terminal]), int(node_count[terminal])


def transitive_reduction_values(
    directed: nx.DiGraph,
    weight_name: str,
) -> tuple[float, int]:
    dag = greedy_acyclic_graph(directed, weight_name)
    if dag.number_of_edges() == 0:
        return 0.0, 0
    reduced = nx.transitive_reduction(dag)
    mass = sum(
        float(dag[left][right][weight_name])
        for left, right in reduced.edges()
    )
    return float(mass), int(reduced.number_of_edges())


def biconnected_cycle_stats(graph: nx.Graph) -> tuple[int, int]:
    best_nodes = 0
    best_edges = 0
    for nodes in nx.biconnected_components(graph):
        if len(nodes) < 3:
            continue
        subgraph = graph.subgraph(nodes)
        # At least one independent cycle.
        if subgraph.number_of_edges() < subgraph.number_of_nodes():
            continue
        if (
            subgraph.number_of_edges() > best_edges
            or (
                subgraph.number_of_edges() == best_edges
                and subgraph.number_of_nodes() > best_nodes
            )
        ):
            best_nodes = subgraph.number_of_nodes()
            best_edges = subgraph.number_of_edges()
    return int(best_nodes), int(best_edges)


def hard_component_stats(
    directed: nx.DiGraph,
    nodes: set[Any],
    raw_count: int,
) -> dict[str, float]:
    subgraph = directed.subgraph(nodes).copy()
    graph = undirected_graph(subgraph, "base_weight")

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    base_weights = [
        float(attributes["base_weight"])
        for _, _, attributes in subgraph.edges(data=True)
    ]
    balanced_weights = [
        float(attributes["balanced_weight"])
        for _, _, attributes in subgraph.edges(data=True)
    ]
    coexistence_weights = [
        float(attributes["coexistence_weight"])
        for _, _, attributes in subgraph.edges(data=True)
    ]
    gap_weights = [
        float(attributes["gap_balanced_weight"])
        for _, _, attributes in subgraph.edges(data=True)
    ]

    base_mass = float(sum(base_weights))
    balanced_mass = float(sum(balanced_weights))
    coexistence_mass = float(sum(coexistence_weights))
    gap_mass = float(sum(gap_weights))

    excess_edges = max(n_edges - n_nodes + 1, 0) if n_nodes else 0
    density = nx.density(graph) if n_nodes > 1 else 0.0
    mean_degree = 2.0 * n_edges / n_nodes if n_nodes else 0.0

    bridges = list(nx.bridges(graph)) if n_edges else []
    nonbridge_edges = max(n_edges - len(bridges), 0)

    if n_nodes:
        two_core = nx.k_core(graph, k=2)
        two_core_nodes = two_core.number_of_nodes()
        two_core_edges = two_core.number_of_edges()
    else:
        two_core_nodes = 0
        two_core_edges = 0

    biconnected_nodes, biconnected_edges = biconnected_cycle_stats(graph)

    base_tree_mass, _ = maximum_spanning_tree_values(graph, "base_weight")
    balanced_tree_mass, balanced_tree_bottleneck = maximum_spanning_tree_values(
        graph, "balanced_weight"
    )

    base_excess_mass = max(base_mass - base_tree_mass, 0.0)
    balanced_excess_mass = max(balanced_mass - balanced_tree_mass, 0.0)

    balanced_reduction_mass, balanced_reduction_edges = (
        transitive_reduction_values(subgraph, "balanced_weight")
    )
    balanced_dag = greedy_acyclic_graph(subgraph, "balanced_weight")
    balanced_path_mass, balanced_path_nodes = maximum_weight_path(
        balanced_dag,
        "balanced_weight",
    )

    effective_edges = effective_count(base_weights)
    balanced_effective_edges = effective_count(balanced_weights)
    mean_balanced_weight = (
        float(np.mean(balanced_weights)) if balanced_weights else 0.0
    )

    denominator = max(raw_count, 1)

    return {
        "largest_component_mass": base_mass,
        "largest_component_edges": float(n_edges),
        "largest_component_nodes": float(n_nodes),
        "largest_component_edges_per_trace": n_edges / denominator,
        "largest_component_nodes_per_trace": n_nodes / denominator,
        "largest_component_excess_edges": float(excess_edges),
        "largest_component_excess_edges_per_trace": excess_edges / denominator,
        "largest_component_excess_edges_per_node": (
            excess_edges / n_nodes if n_nodes else 0.0
        ),
        "largest_component_excess_edge_fraction": (
            excess_edges / n_edges if n_edges else 0.0
        ),
        "largest_component_density": float(density),
        "largest_component_mean_degree": float(mean_degree),
        "largest_component_nonbridge_edges": float(nonbridge_edges),
        "largest_component_nonbridge_edges_per_trace": (
            nonbridge_edges / denominator
        ),
        "largest_component_nonbridge_edge_fraction": (
            nonbridge_edges / n_edges if n_edges else 0.0
        ),
        "largest_component_two_core_nodes": float(two_core_nodes),
        "largest_component_two_core_edges": float(two_core_edges),
        "largest_component_two_core_edges_per_trace": (
            two_core_edges / denominator
        ),
        "largest_component_two_core_node_fraction": (
            two_core_nodes / n_nodes if n_nodes else 0.0
        ),
        "largest_biconnected_cycle_nodes": float(biconnected_nodes),
        "largest_biconnected_cycle_edges": float(biconnected_edges),
        "largest_biconnected_cycle_edges_per_trace": (
            biconnected_edges / denominator
        ),
        "largest_component_effective_edge_count": effective_edges,
        "largest_component_effective_edge_ratio": (
            effective_edges / n_edges if n_edges else 0.0
        ),
        "largest_component_weighted_excess_mass": base_excess_mass,
        "largest_component_weighted_excess_mass_fraction": (
            base_excess_mass / base_mass if base_mass > 0 else 0.0
        ),
        "balanced_largest_component_mass": balanced_mass,
        "coexistence_largest_component_mass": coexistence_mass,
        "gap_balanced_largest_component_mass": gap_mass,
        "balanced_effective_edge_count": balanced_effective_edges,
        "balanced_effective_edge_evidence": (
            balanced_effective_edges * mean_balanced_weight
        ),
        "balanced_max_spanning_tree_mass": balanced_tree_mass,
        "balanced_weighted_excess_mass": balanced_excess_mass,
        "balanced_weighted_excess_mass_fraction": (
            balanced_excess_mass / balanced_mass if balanced_mass > 0 else 0.0
        ),
        "balanced_tree_bottleneck_score": (
            balanced_tree_bottleneck * n_nodes
        ),
        "balanced_transitive_reduction_mass": balanced_reduction_mass,
        "balanced_transitive_reduction_edges": float(balanced_reduction_edges),
        "balanced_max_weight_path_mass": balanced_path_mass,
        "balanced_max_weight_path_nodes": float(balanced_path_nodes),
    }


def soft_component_stats(
    directed: nx.DiGraph,
    nodes: set[Any],
) -> dict[str, float]:
    subgraph = directed.subgraph(nodes).copy()
    graph = undirected_graph(subgraph, "soft_weight")

    weights = [
        float(attributes["soft_weight"])
        for _, _, attributes in subgraph.edges(data=True)
    ]
    mass = float(sum(weights))
    effective_edges = effective_count(weights)

    tree_mass, _ = maximum_spanning_tree_values(graph, "soft_weight")
    excess_mass = max(mass - tree_mass, 0.0)
    reduction_mass, _ = transitive_reduction_values(subgraph, "soft_weight")
    dag = greedy_acyclic_graph(subgraph, "soft_weight")
    path_mass, _ = maximum_weight_path(dag, "soft_weight")

    return {
        "soft_largest_component_mass": mass,
        "soft_effective_edge_count": effective_edges,
        "soft_max_spanning_tree_mass": tree_mass,
        "soft_weighted_excess_mass": excess_mass,
        "soft_weighted_excess_mass_fraction": (
            excess_mass / mass if mass > 0 else 0.0
        ),
        "soft_transitive_reduction_mass": reduction_mass,
        "soft_max_weight_path_mass": path_mass,
    }


def maximum_component_metrics(
    graph: nx.DiGraph,
    metrics: Sequence[str],
    component_function,
    *component_args,
) -> dict[str, float]:
    if graph.number_of_edges() == 0:
        return {metric: 0.0 for metric in metrics}

    rows = [
        component_function(graph, set(nodes), *component_args)
        for nodes in nx.weakly_connected_components(graph)
    ]
    return {
        metric: float(max((row[metric] for row in rows), default=0.0))
        for metric in metrics
    }


def calculate_answer_graph_metrics(
    group: pd.DataFrame,
    raw_count: int,
    precedence_min_consistency: float,
) -> dict[str, float]:
    hard_graph = nx.DiGraph()
    soft_graph = nx.DiGraph()

    for row in group.itertuples(index=False):
        consistency = float(row.ordering_consistency)
        if consistency < precedence_min_consistency:
            continue

        support_fraction = float(row.support_fraction)
        direction_strength = float(row.direction_strength)
        adjacent = float(row.adjacent_forward_count)
        nonadjacent = float(row.nonadjacent_forward_count)
        mean_gap = (
            float(row.mean_forward_gap)
            if pd.notna(row.mean_forward_gap)
            else 1.0
        )

        base_weight = support_fraction * direction_strength
        balance = hard_balance(adjacent, nonadjacent)
        balanced_weight = base_weight * balance
        coexistence_weight = (
            direction_strength
            * 2.0
            * min(adjacent, nonadjacent)
            / max(raw_count, 1)
        )
        gap_multiplier = min(
            math.log1p(max(mean_gap, 0.0)),
            math.log(4.0),
        )
        gap_balanced_weight = balanced_weight * gap_multiplier

        soft_weight = base_weight * soft_balance(adjacent, nonadjacent)
        soft_attributes = {
            "base_weight": base_weight,
            "soft_weight": soft_weight,
        }
        add_or_replace_edge(
            soft_graph,
            row.from_community,
            row.to_community,
            soft_attributes,
            "soft_weight",
        )

        if adjacent > 0 and nonadjacent > 0:
            hard_attributes = {
                "base_weight": base_weight,
                "balanced_weight": balanced_weight,
                "coexistence_weight": coexistence_weight,
                "gap_balanced_weight": gap_balanced_weight,
            }
            add_or_replace_edge(
                hard_graph,
                row.from_community,
                row.to_community,
                hard_attributes,
                "base_weight",
            )

    output = maximum_component_metrics(
        hard_graph,
        HARD_STATIC_METRICS + WEIGHTED_STATIC_METRICS,
        hard_component_stats,
        raw_count,
    )
    output.update(
        maximum_component_metrics(
            soft_graph,
            SOFT_STATIC_METRICS,
            soft_component_stats,
        )
    )
    output["n_hard_shortcut_components"] = float(
        nx.number_weakly_connected_components(hard_graph)
        if hard_graph.number_of_nodes()
        else 0
    )
    output["n_soft_stable_components"] = float(
        nx.number_weakly_connected_components(soft_graph)
        if soft_graph.number_of_nodes()
        else 0
    )
    return output


def load_answers(
    path: Path,
    args: argparse.Namespace,
    sample_sizes: set[int] | None,
) -> pd.DataFrame:
    required = {
        *ANSWER_KEY_COLUMNS,
        "raw_count",
        "first_trace_position",
        "is_correct_answer",
    }
    usecols = None
    frame = pd.read_csv(path, low_memory=False, usecols=usecols)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"answer_metrics is missing columns: {missing}")

    frame = filter_fixed_config(
        frame,
        community_min_support=args.community_min_support,
        transition_min_support=args.transition_min_support,
        score_mode=args.score_mode,
        edge_quantile=args.edge_quantile,
        cohesion_mode=args.cohesion_mode,
        sample_sizes=sample_sizes,
    )

    keep = [
        *ANSWER_KEY_COLUMNS,
        "raw_count",
        "full_raw_count",
        "first_trace_position",
        "is_correct_answer",
        "sample_majority_answer",
        "sample_majority_is_correct",
        "n_local_communities",
        "n_shortcut_surviving_pairs",
        "shortcut_survival_mass",
    ]
    frame = frame[[column for column in keep if column in frame.columns]].copy()
    frame["answer"] = frame["answer"].astype(str)
    frame["raw_count"] = numeric(frame["raw_count"], 0).astype(int)
    frame["first_trace_position"] = numeric(
        frame["first_trace_position"],
        np.inf,
    )
    frame["is_correct_answer"] = bool_series(frame["is_correct_answer"])

    duplicated = frame.duplicated(ANSWER_KEY_COLUMNS, keep=False)
    if duplicated.any():
        raise ValueError(
            f"answer_metrics contains {int(duplicated.sum())} duplicated answer keys."
        )
    return frame


def load_precedence(
    path: Path,
    args: argparse.Namespace,
    sample_sizes: set[int] | None,
) -> tuple[pd.DataFrame, float]:
    required = {
        *ANSWER_KEY_COLUMNS,
        "precedence_min_consistency",
        "from_community",
        "to_community",
        "support_fraction",
        "ordering_consistency",
        "direction_strength",
        "adjacent_forward_count",
        "nonadjacent_forward_count",
        "shortcut_survives",
        "mean_forward_gap",
    }

    keep = list(required)
    parts: list[pd.DataFrame] = []
    consistency_values: set[float] = set()

    for chunk in pd.read_csv(
        path,
        chunksize=args.precedence_chunksize,
        low_memory=False,
        usecols=lambda column: column in required,
    ):
        missing = sorted(required - set(chunk.columns))
        if missing:
            raise ValueError(f"precedence_details is missing columns: {missing}")

        chunk = filter_fixed_config(
            chunk,
            community_min_support=args.community_min_support,
            transition_min_support=args.transition_min_support,
            score_mode=args.score_mode,
            edge_quantile=args.edge_quantile,
            cohesion_mode=args.cohesion_mode,
            sample_sizes=sample_sizes,
        )
        if chunk.empty:
            continue

        values = numeric(
            chunk["precedence_min_consistency"],
            np.nan,
        ).dropna().unique()
        consistency_values.update(float(value) for value in values)

        chunk = chunk[keep].copy()
        chunk["answer"] = chunk["answer"].astype(str)
        for column in [
            "from_community",
            "to_community",
            "support_fraction",
            "ordering_consistency",
            "direction_strength",
            "adjacent_forward_count",
            "nonadjacent_forward_count",
            "mean_forward_gap",
        ]:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        chunk["shortcut_survives"] = bool_series(chunk["shortcut_survives"])
        chunk = chunk.dropna(
            subset=[
                "from_community",
                "to_community",
                "support_fraction",
                "ordering_consistency",
                "direction_strength",
            ]
        )
        parts.append(chunk)

    if not parts:
        raise ValueError("No precedence rows remain after filtering.")
    if len(consistency_values) != 1:
        raise ValueError(
            "Expected exactly one saved precedence_min_consistency value, found: "
            f"{sorted(consistency_values)}"
        )

    saved_consistency = next(iter(consistency_values))
    effective_consistency = (
        float(args.precedence_min_consistency)
        if args.precedence_min_consistency is not None
        else float(saved_consistency)
    )

    if not 0.5 <= effective_consistency <= 1.0:
        raise ValueError(
            "--precedence-min-consistency must be in [0.5, 1.0]."
        )

    if effective_consistency + 1e-12 < saved_consistency:
        raise ValueError(
            "The requested offline consistency threshold is lower than the "
            "threshold used to save the original shortcut details. A stricter "
            "threshold can be reconstructed exactly; a lower threshold may "
            "require rows that were not preserved. "
            f"saved={saved_consistency}, requested={effective_consistency}"
        )

    print(
        f"[precedence] saved consistency={saved_consistency}, "
        f"offline consistency={effective_consistency}"
    )

    return pd.concat(parts, ignore_index=True), effective_consistency


def build_answer_metric_table(
    answers: pd.DataFrame,
    precedence: pd.DataFrame,
    precedence_min_consistency: float,
) -> pd.DataFrame:
    answer_lookup = answers.set_index(ANSWER_KEY_COLUMNS)["raw_count"]
    rows: list[dict[str, Any]] = []

    grouped = precedence.groupby(ANSWER_KEY_COLUMNS, dropna=False, sort=False)
    progress = tqdm(
        grouped,
        total=grouped.ngroups,
        desc="Calculating saved graph metrics",
        unit="answer-graph",
        dynamic_ncols=True,
    )

    unmatched = 0
    for key, group in progress:
        if not isinstance(key, tuple):
            key = (key,)
        try:
            raw_count = int(answer_lookup.loc[key])
        except KeyError:
            unmatched += 1
            continue

        row = dict(zip(ANSWER_KEY_COLUMNS, key))
        row.update(
            calculate_answer_graph_metrics(
                group,
                raw_count=raw_count,
                precedence_min_consistency=precedence_min_consistency,
            )
        )
        rows.append(row)

    if unmatched:
        print(f"[warning] skipped {unmatched:,} precedence graph keys absent from answers")

    graph_metrics = pd.DataFrame(rows)
    merged = answers.merge(
        graph_metrics,
        on=ANSWER_KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    metric_columns = [
        *STATIC_METRICS,
        "n_hard_shortcut_components",
        "n_soft_stable_components",
    ]
    for metric in metric_columns:
        merged[metric] = numeric(merged.get(metric, pd.Series(index=merged.index)), 0.0)
    return merged


def add_dynamic_metrics(
    frame: pd.DataFrame,
    window: int,
) -> tuple[pd.DataFrame, list[str]]:
    if window < 1:
        raise ValueError("--growth-window must be positive")

    output = frame.sort_values(
        ["id", "answer", *CONFIG_COLUMNS, "sample_size"],
        kind="mergesort",
    ).copy()
    group_columns = ["id", "answer", *CONFIG_COLUMNS]
    dynamic_metrics: list[str] = []

    grouped = output.groupby(group_columns, dropna=False, sort=False)

    for metric in DYNAMIC_BASE_METRICS:
        lag = grouped[metric].shift(window)
        delta = output[metric] - lag

        growth_name = f"{metric}__growth_w{window}"
        output[growth_name] = delta.fillna(0.0)
        dynamic_metrics.append(growth_name)

        lag_count = grouped["raw_count"].shift(window)
        added_traces = output["raw_count"] - lag_count
        per_trace_name = f"{metric}__growth_per_new_trace_w{window}"
        output[per_trace_name] = np.where(
            added_traces > 0,
            delta / added_traces,
            0.0,
        )
        output[per_trace_name] = numeric(output[per_trace_name], 0.0)
        dynamic_metrics.append(per_trace_name)

        rolling_name = f"{metric}__rolling_mean_w{window}"
        rolling = (
            grouped[metric]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=group_columns, drop=True)
        )
        output[rolling_name] = rolling.reindex(output.index).fillna(0.0)
        dynamic_metrics.append(rolling_name)

    return output, dynamic_metrics


def select_metric_names(
    specification: str,
    dynamic_metrics: Sequence[str],
) -> list[str]:
    available = list(STATIC_METRICS) + list(dynamic_metrics)
    value = specification.strip()
    if value.lower() == "all":
        return available
    if value.lower() == "static":
        return list(STATIC_METRICS)

    requested = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}")
    return requested


def select_answers(
    candidates: pd.DataFrame,
    metrics: Sequence[str],
    minimum_answer_frequency: int,
) -> pd.DataFrame:
    """Vectorized answer selection: one global sort per metric."""
    group_columns = ["id", "sample_size", *CONFIG_COLUMNS]

    work = candidates.copy()
    group_has_eligible = (
        work["raw_count"].ge(minimum_answer_frequency)
        .groupby([work[column] for column in group_columns], dropna=False)
        .transform("any")
    )
    pool = work.loc[
        work["raw_count"].ge(minimum_answer_frequency)
        | ~group_has_eligible
    ].copy()

    majority = (
        work.sort_values(
            group_columns
            + ["raw_count", "first_trace_position", "answer"],
            ascending=[True] * len(group_columns) + [False, True, True],
            kind="mergesort",
        )
        .drop_duplicates(group_columns, keep="first")
        [
            group_columns
            + [
                "answer",
                "raw_count",
                "is_correct_answer",
            ]
        ]
        .rename(
            columns={
                "answer": "majority_answer",
                "raw_count": "majority_raw_count",
                "is_correct_answer": "majority_correct",
            }
        )
    )
    majority["majority_correct"] = majority["majority_correct"].astype(bool)

    candidate_counts = (
        work.groupby(group_columns, dropna=False)
        .size()
        .rename("n_candidate_answers")
        .reset_index()
    )
    eligible_counts = (
        pool.groupby(group_columns, dropna=False)
        .size()
        .rename("n_eligible_answers")
        .reset_index()
    )

    prediction_parts: list[pd.DataFrame] = []
    progress = tqdm(
        metrics,
        total=len(metrics),
        desc="Selecting final answers",
        unit="metric",
        dynamic_ncols=True,
    )

    for metric in progress:
        metric_pool = pool[group_columns + [
            "answer",
            "raw_count",
            "first_trace_position",
            "is_correct_answer",
            metric,
        ]].copy()
        metric_pool[metric] = numeric(metric_pool[metric], 0.0)

        # Top score and number of tied answers in every question-prefix group.
        top_scores = (
            metric_pool.groupby(group_columns, dropna=False)[metric]
            .max()
            .rename("predicted_score")
            .reset_index()
        )
        metric_pool = metric_pool.merge(
            top_scores,
            on=group_columns,
            how="left",
            validate="many_to_one",
        )
        metric_pool["is_top_score"] = np.isclose(
            metric_pool[metric],
            metric_pool["predicted_score"],
        )
        tie_counts = (
            metric_pool.groupby(group_columns, dropna=False)["is_top_score"]
            .sum()
            .astype(int)
            .rename("metric_ties")
            .reset_index()
        )

        # Best score strictly below the tied maximum, for a descriptive margin.
        runner_up = (
            metric_pool.loc[~metric_pool["is_top_score"]]
            .groupby(group_columns, dropna=False)[metric]
            .max()
            .rename("runner_up_score")
            .reset_index()
        )

        selected = (
            metric_pool.sort_values(
                group_columns
                + [metric, "raw_count", "first_trace_position", "answer"],
                ascending=[True] * len(group_columns)
                + [False, False, True, True],
                kind="mergesort",
            )
            .drop_duplicates(group_columns, keep="first")
            [
                group_columns
                + [
                    "answer",
                    "raw_count",
                    "is_correct_answer",
                    metric,
                ]
            ]
            .rename(
                columns={
                    "answer": "predicted_answer",
                    "raw_count": "selected_raw_count",
                    "is_correct_answer": "prediction_correct",
                    metric: "predicted_score",
                }
            )
        )
        selected["prediction_correct"] = (
            selected["prediction_correct"].astype(bool)
        )

        result = (
            selected.merge(
                majority,
                on=group_columns,
                how="left",
                validate="one_to_one",
            )
            .merge(
                tie_counts,
                on=group_columns,
                how="left",
                validate="one_to_one",
            )
            .merge(
                runner_up,
                on=group_columns,
                how="left",
                validate="one_to_one",
            )
            .merge(
                candidate_counts,
                on=group_columns,
                how="left",
                validate="one_to_one",
            )
            .merge(
                eligible_counts,
                on=group_columns,
                how="left",
                validate="one_to_one",
            )
        )

        # Obtain the majority's value for this metric without another sort.
        majority_scores = work[
            group_columns + ["answer", metric]
        ].rename(
            columns={
                "answer": "majority_answer",
                metric: "majority_score",
            }
        )
        result = result.merge(
            majority_scores,
            on=group_columns + ["majority_answer"],
            how="left",
            validate="one_to_one",
        )

        result["runner_up_score"] = result["runner_up_score"].fillna(
            result["predicted_score"]
        )
        result["score_margin"] = (
            result["predicted_score"] - result["runner_up_score"]
        )
        result["metric"] = metric
        result["minimum_answer_frequency"] = minimum_answer_frequency
        result["repair"] = (
            result["prediction_correct"] & ~result["majority_correct"]
        ).astype(int)
        result["harm"] = (
            ~result["prediction_correct"] & result["majority_correct"]
        ).astype(int)
        result["net_repair"] = result["repair"] - result["harm"]
        result["changed_from_majority"] = (
            result["predicted_answer"] != result["majority_answer"]
        )
        result["all_scores_zero"] = np.isclose(
            result["predicted_score"],
            0.0,
        )
        result["prediction_correct"] = result["prediction_correct"].astype(int)
        result["majority_correct"] = result["majority_correct"].astype(int)

        prediction_parts.append(
            result[
                group_columns
                + [
                    "minimum_answer_frequency",
                    "metric",
                    "predicted_answer",
                    "predicted_score",
                    "selected_raw_count",
                    "prediction_correct",
                    "majority_answer",
                    "majority_score",
                    "majority_raw_count",
                    "majority_correct",
                    "repair",
                    "harm",
                    "net_repair",
                    "changed_from_majority",
                    "all_scores_zero",
                    "metric_ties",
                    "score_margin",
                    "n_candidate_answers",
                    "n_eligible_answers",
                ]
            ]
        )

    return pd.concat(prediction_parts, ignore_index=True)


def summarize_accuracy(predictions: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "sample_size",
        *CONFIG_COLUMNS,
        "minimum_answer_frequency",
        "metric",
    ]

    summary = (
        predictions.groupby(group_columns, dropna=False)
        .agg(
            n_ids=("id", "nunique"),
            correct=("prediction_correct", "sum"),
            majority_correct=("majority_correct", "sum"),
            repairs=("repair", "sum"),
            harms=("harm", "sum"),
            net_repairs=("net_repair", "sum"),
            changed_answers=("changed_from_majority", "sum"),
            zero_score_fallbacks=("all_scores_zero", "sum"),
            total_metric_ties=("metric_ties", "sum"),
            mean_score_margin=("score_margin", "mean"),
        )
        .reset_index()
    )
    summary["accuracy"] = summary["correct"] / summary["n_ids"]
    summary["majority_accuracy"] = (
        summary["majority_correct"] / summary["n_ids"]
    )
    summary["gain"] = summary["accuracy"] - summary["majority_accuracy"]
    summary["change_rate"] = summary["changed_answers"] / summary["n_ids"]
    summary["zero_score_rate"] = (
        summary["zero_score_fallbacks"] / summary["n_ids"]
    )
    return summary


def range_label(sample_size: int) -> str:
    if 5 <= sample_size <= 10:
        return "5-10"
    if 11 <= sample_size <= 30:
        return "11-30"
    if 31 <= sample_size <= 60:
        return "31-60"
    if 61 <= sample_size <= 81:
        return "61-81"
    if 82 <= sample_size <= 99:
        return "82-99"
    return "other"


def metric_summary(
    accuracy: pd.DataFrame,
    rank_sizes: set[int] | None,
    required_n_ids: int | None,
) -> pd.DataFrame:
    selected = accuracy.copy()
    if rank_sizes is not None:
        selected = selected[selected["sample_size"].isin(rank_sizes)].copy()
    if required_n_ids is not None:
        selected = selected[selected["n_ids"].eq(required_n_ids)].copy()

    summary = (
        selected.groupby("metric", as_index=False)
        .agg(
            n_sample_sizes=("sample_size", "nunique"),
            first_sample_size=("sample_size", "min"),
            last_sample_size=("sample_size", "max"),
            mean_accuracy=("accuracy", "mean"),
            mean_majority_accuracy=("majority_accuracy", "mean"),
            mean_gain=("gain", "mean"),
            median_gain=("gain", "median"),
            min_gain=("gain", "min"),
            max_gain=("gain", "max"),
            better_sizes=("gain", lambda values: int((values > 0).sum())),
            equal_sizes=("gain", lambda values: int((values == 0).sum())),
            worse_sizes=("gain", lambda values: int((values < 0).sum())),
            total_repairs=("repairs", "sum"),
            total_harms=("harms", "sum"),
            total_net_repairs=("net_repairs", "sum"),
            mean_change_rate=("change_rate", "mean"),
            mean_zero_score_rate=("zero_score_rate", "mean"),
        )
        .sort_values(
            ["mean_gain", "min_gain", "better_sizes", "mean_accuracy"],
            ascending=[False, False, False, False],
        )
        .reset_index(drop=True)
    )
    return summary


def summary_by_range(accuracy: pd.DataFrame) -> pd.DataFrame:
    frame = accuracy.copy()
    frame["sample_range"] = frame["sample_size"].map(range_label)
    return (
        frame.groupby(["sample_range", "metric"], as_index=False)
        .agg(
            n_sample_sizes=("sample_size", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            mean_majority_accuracy=("majority_accuracy", "mean"),
            mean_gain=("gain", "mean"),
            min_gain=("gain", "min"),
            max_gain=("gain", "max"),
            better_sizes=("gain", lambda values: int((values > 0).sum())),
            equal_sizes=("gain", lambda values: int((values == 0).sum())),
            worse_sizes=("gain", lambda values: int((values < 0).sum())),
            total_repairs=("repairs", "sum"),
            total_harms=("harms", "sum"),
            total_net_repairs=("net_repairs", "sum"),
        )
        .sort_values(
            ["sample_range", "mean_gain", "min_gain"],
            ascending=[True, False, False],
        )
        .reset_index(drop=True)
    )


def plot_top_metrics(
    accuracy: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    selected = accuracy.copy()
    if args.required_n_ids is not None:
        selected = selected[selected["n_ids"].eq(args.required_n_ids)].copy()

    top_metrics = summary.head(args.top_n)["metric"].tolist()
    top_curves = selected[selected["metric"].isin(top_metrics)].copy()

    duplicates = top_curves.duplicated(["metric", "sample_size"], keep=False)
    if duplicates.any():
        duplicate_preview = top_curves.loc[
            duplicates,
            [
                "metric",
                "sample_size",
                "community_min_support",
                "transition_min_support",
                "n_ids",
            ],
        ].sort_values(["metric", "sample_size"])
        raise ValueError(
            "Multiple rows remain for metric/sample-size pairs:\n"
            + duplicate_preview.head(30).to_string(index=False)
        )

    majority_curve = (
        top_curves[["sample_size", "majority_accuracy"]]
        .drop_duplicates("sample_size")
        .sort_values("sample_size")
    )

    title_suffix = (
        f"Community support = {args.community_min_support}, "
        f"transition support = {args.transition_min_support}"
    )
    if args.required_n_ids is not None:
        title_suffix += f", N = {args.required_n_ids}"

    plt.figure(figsize=(14, 7))
    for metric in top_metrics:
        curve = top_curves[top_curves["metric"].eq(metric)].sort_values(
            "sample_size"
        )
        plt.plot(
            curve["sample_size"],
            curve["accuracy"],
            marker="o",
            markersize=2.5,
            linewidth=1.4,
            label=metric,
        )
    plt.plot(
        majority_curve["sample_size"],
        majority_curve["majority_accuracy"],
        linestyle="--",
        linewidth=2.5,
        label="Majority vote",
    )
    plt.xlabel("Number of sampled traces")
    plt.ylabel("Accuracy")
    plt.title(f"Top {len(top_metrics)} saved graph metrics vs majority\n{title_suffix}")
    plt.grid(alpha=0.3)
    plt.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    accuracy_path = output_dir / "top_metrics_accuracy_vs_majority.png"
    plt.savefig(accuracy_path, dpi=args.plot_dpi, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(14, 7))
    for metric in top_metrics:
        curve = top_curves[top_curves["metric"].eq(metric)].sort_values(
            "sample_size"
        )
        plt.plot(
            curve["sample_size"],
            curve["gain"],
            marker="o",
            markersize=2.5,
            linewidth=1.4,
            label=metric,
        )
    plt.axhline(0.0, linestyle="--", linewidth=2, label="Majority baseline")
    plt.xlabel("Number of sampled traces")
    plt.ylabel("Accuracy gain over majority")
    plt.title(f"Gain of top {len(top_metrics)} saved graph metrics\n{title_suffix}")
    plt.grid(alpha=0.3)
    plt.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    gain_path = output_dir / "top_metrics_gain_vs_majority.png"
    plt.savefig(gain_path, dpi=args.plot_dpi, bbox_inches="tight")
    plt.close()

    return top_metrics


def main() -> None:
    args = parse_args()
    if args.top_n < 1:
        raise ValueError("--top-n must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_sizes = parse_int_spec(args.sample_sizes)
    rank_sizes = parse_int_spec(args.rank_sample_sizes)

    answers = load_answers(Path(args.answer_metrics), args, sample_sizes)
    if answers.empty:
        raise ValueError("No answer rows remain after filtering.")
    print(
        f"[answers] rows={len(answers):,}, ids={answers['id'].nunique():,}, "
        f"sample sizes={answers['sample_size'].min()}-"
        f"{answers['sample_size'].max()}"
    )

    precedence, precedence_min_consistency = load_precedence(
        Path(args.precedence_details),
        args,
        sample_sizes,
    )
    print(
        f"[precedence] rows={len(precedence):,}, "
        f"answer graphs={precedence.groupby(ANSWER_KEY_COLUMNS, dropna=False).ngroups:,}, "
        f"consistency threshold={precedence_min_consistency}"
    )

    answer_graphs = build_answer_metric_table(
        answers,
        precedence,
        precedence_min_consistency,
    )
    answer_graphs, dynamic_metrics = add_dynamic_metrics(
        answer_graphs,
        window=args.growth_window,
    )
    metrics = select_metric_names(args.metrics, dynamic_metrics)

    answer_graphs.to_csv(
        output_dir / "answer_graph_metrics.csv.gz",
        index=False,
        compression="gzip",
    )

    predictions = select_answers(
        answer_graphs,
        metrics=metrics,
        minimum_answer_frequency=args.minimum_answer_frequency,
    )
    predictions.to_csv(
        output_dir / "graph_metric_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    accuracy = summarize_accuracy(predictions)
    accuracy.to_csv(
        output_dir / "graph_metric_accuracy_by_sample_size.csv",
        index=False,
    )

    summary = metric_summary(
        accuracy,
        rank_sizes=rank_sizes,
        required_n_ids=args.required_n_ids,
    )
    if summary.empty:
        raise ValueError(
            "No rows remain for metric ranking. Check --rank-sample-sizes and "
            "--required-n-ids."
        )
    summary.to_csv(
        output_dir / "metric_summary_rank_range.csv",
        index=False,
    )

    ranged = summary_by_range(accuracy)
    ranged.to_csv(
        output_dir / "metric_summary_by_range.csv",
        index=False,
    )

    top_metrics = plot_top_metrics(
        accuracy,
        summary,
        output_dir,
        args,
    )

    metadata = {
        "answer_metrics": str(args.answer_metrics),
        "precedence_details": str(args.precedence_details),
        "community_min_support": args.community_min_support,
        "transition_min_support": args.transition_min_support,
        "score_mode": args.score_mode,
        "edge_quantile": args.edge_quantile,
        "cohesion_mode": args.cohesion_mode,
        "precedence_min_consistency": precedence_min_consistency,
        "minimum_answer_frequency": args.minimum_answer_frequency,
        "sample_sizes": args.sample_sizes,
        "rank_sample_sizes": args.rank_sample_sizes,
        "growth_window": args.growth_window,
        "required_n_ids": args.required_n_ids,
        "evaluated_metrics": metrics,
        "top_metrics": top_metrics,
        "n_answer_rows": int(len(answer_graphs)),
        "n_unique_ids": int(answer_graphs["id"].nunique()),
        "n_precedence_rows": int(len(precedence)),
        "notes": [
            "Hard graph metrics reconstruct shortcut membership from ordering consistency plus positive adjacent and nonadjacent counts.",
            "Soft metrics use all saved stable pairs and Jeffreys-smoothed adjacent/nonadjacent balance.",
            "Pairs below the saved transition_min_support cannot be recovered offline.",
            "All-zero metric ties fall back to the majority through raw_count tie-breaking.",
        ],
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    display_columns = [
        "metric",
        "n_sample_sizes",
        "mean_accuracy",
        "mean_majority_accuracy",
        "mean_gain",
        "min_gain",
        "max_gain",
        "better_sizes",
        "equal_sizes",
        "worse_sizes",
        "total_repairs",
        "total_harms",
        "total_net_repairs",
    ]
    print("\nTop metrics in ranking range:")
    print(summary[display_columns].head(max(args.top_n, 20)).to_string(index=False))
    print(f"\nSaved analysis to: {output_dir}")


if __name__ == "__main__":
    main()
