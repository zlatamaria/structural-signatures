#!/usr/bin/env python3
# VERSION: answer-local-prefix-community-v8-optimized-resume-2026-08-06
"""
Optimized executable for ``audit_answer_local_prefix_allpairs_v8.py``.

Keep this file in the same directory as the original v8 script. It imports the
original implementation and replaces only the expensive per-question loop.
The metric definitions and final evaluation code remain those of v8.

Main optimizations
------------------
1. Pre-score all step pairs needed by the largest requested prefix once per
   question/seed, in large chunks. Smaller prefixes reuse the in-memory logits.
2. Build prefix pair lists incrementally instead of reconstructing every
   within-answer Cartesian product for k=2,3,...,99.
3. Run constrained clustering once per prefix/cohesion configuration with
   support=1, then filter and renumber the same clusters for community supports
   2,3,5. Community support does not affect the merge decisions in v8.
4. Calculate community summaries once per community-support configuration,
   rather than once per transition-support value.
5. Build checkpoint sequences once per community-support configuration and
   reuse them for every transition and precedence support threshold.
6. Precompute raw precedence counts once, then apply support thresholds cheaply.
7. Skip direct trace-pair metrics by default. They are unrelated to community,
   transition, precedence, shortcut, and chain metrics.
8. Make large detail outputs independently optional. Transition details remain
   enabled by default, as requested.

The following outputs are preserved by default:
- answer_metrics.csv (all community, transition, and precedence metric values)
- transition_details.csv
- precedence_details.csv
- metric_predictions.csv / metric_accuracy.csv
- majority learning curves and run metadata

By default, these expensive diagnostic outputs are omitted:
- trace_pair_metrics.csv
- trace_checkpoint_schemas.csv
- community_details.csv
- stage_diagnostics.csv

Use the corresponding --save-* flags below when an audit requires them.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **_: Any):
        return iterable


def load_base_module():
    base_path = Path(__file__).with_name("audit_answer_local_prefix_allpairs_v8.py")
    if not base_path.exists():
        raise FileNotFoundError(
            f"Expected the original v8 script beside this file: {base_path}"
        )
    spec = importlib.util.spec_from_file_location(
        "audit_answer_local_prefix_allpairs_v8_base", base_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {base_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()

DIRECT_TRACE_PAIR_METRICS = {
    "mean_best_trace_pair_score",
    "mean_top2_trace_pair_score",
    "mean_greedy_one_to_one_score",
    "mean_order_consistency",
    "trace_pair_match_coverage",
    "trace_pair_two_match_coverage",
}

TRANSITION_ORDERING_METRICS = (
    "n_shared_transition_types",
    "shared_transition_types_per_trace",
    "transition_trace_coverage",
    "transition_support_gini_simpson",
    "effective_shared_transitions_per_trace",
    "transition_set_diversity_gini_simpson",
    "transition_sequence_diversity_gini_simpson",
    "mean_pairwise_transition_jaccard_distance",
    "supported_pairwise_transition_jaccard_distance",
    "coverage_weighted_transition_set_diversity",
    "fraction_branching_checkpoints",
    "support_weighted_branching_fraction",
    "max_transition_out_degree",
    "late_shared_transition_fraction",
    "cohesion_weighted_transition_support",
    "n_supported_precedence_pairs",
    "n_stable_precedence_pairs",
    "stable_precedence_pairs_per_trace",
    "supported_precedence_mass",
    "precedence_reversal_rate",
    "n_shortcut_surviving_pairs",
    "shortcut_survival_mass",
    "consensus_chain_length",
    "consensus_chain_fraction",
    "atomic_ordered_backbone_mass",
)


# ---------------------------------------------------------------------------
# Pair prescoring and incremental prefix construction
# ---------------------------------------------------------------------------


def count_all_pairs_within_answer(
    question: Any, trace_ids: Sequence[int]
) -> int:
    total = 0
    trace_ids = list(trace_ids)
    for left_index, left_trace in enumerate(trace_ids):
        n_left = len(question.unit_ids_by_trace.get(left_trace, ()))
        for right_trace in trace_ids[left_index + 1 :]:
            total += n_left * len(question.unit_ids_by_trace.get(right_trace, ()))
    return total


def iter_all_pairs_within_answer_chunks(
    *,
    question: Any,
    trace_ids: Sequence[int],
    chunk_size: int,
) -> Iterator[list[tuple[int, int]]]:
    """Yield cross-trace unit pairs without creating one huge Python list."""
    chunk: list[tuple[int, int]] = []
    trace_ids = list(trace_ids)
    for left_index, left_trace in enumerate(trace_ids):
        left_units = question.unit_ids_by_trace.get(left_trace, ())
        for right_trace in trace_ids[left_index + 1 :]:
            right_units = question.unit_ids_by_trace.get(right_trace, ())
            for left in left_units:
                for right in right_units:
                    chunk.append((left, right))
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
    if chunk:
        yield chunk


def prescore_largest_prefix(
    *,
    question: Any,
    prefix_trace_ids: Sequence[int],
    score_store: Any,
    chunk_size: int,
    show_progress: bool,
) -> None:
    traces_by_answer: dict[str, list[int]] = defaultdict(list)
    for trace_id in prefix_trace_ids:
        traces_by_answer[str(question.answers_by_trace[trace_id])].append(int(trace_id))

    total_pairs = sum(
        count_all_pairs_within_answer(question, trace_ids)
        for trace_ids in traces_by_answer.values()
    )
    progress = tqdm(
        total=total_pairs,
        desc=f"prescore id={question.question_id}",
        unit="pair",
        leave=False,
        disable=not show_progress,
        dynamic_ncols=True,
    )
    try:
        for trace_ids in traces_by_answer.values():
            for chunk in iter_all_pairs_within_answer_chunks(
                question=question,
                trace_ids=trace_ids,
                chunk_size=chunk_size,
            ):
                score_store.ensure(chunk, show_progress_bar=False)
                progress.update(len(chunk))
    finally:
        progress.close()


@dataclass
class PrefixPairState:
    traces_by_answer: dict[str, list[int]]
    pairs_by_answer: dict[str, list[tuple[int, int]]]
    observed_answers: list[str]
    prefix_trace_ids: list[int]


def empty_prefix_state() -> PrefixPairState:
    return PrefixPairState(
        traces_by_answer=defaultdict(list),
        pairs_by_answer=defaultdict(list),
        observed_answers=[],
        prefix_trace_ids=[],
    )


def extend_prefix_state(
    *,
    state: PrefixPairState,
    question: Any,
    ordered_trace_ids: Sequence[int],
    new_prefix_count: int,
) -> None:
    """Add only the newly exposed traces and their within-answer pairs."""
    old_count = len(state.prefix_trace_ids)
    if new_prefix_count < old_count:
        raise ValueError("Prefix counts must be processed in nondecreasing order.")
    for trace_id in ordered_trace_ids[old_count:new_prefix_count]:
        trace_id = int(trace_id)
        answer = str(question.answers_by_trace[trace_id])
        if answer not in state.traces_by_answer:
            state.observed_answers.append(answer)
        new_units = question.unit_ids_by_trace.get(trace_id, ())
        for prior_trace in state.traces_by_answer[answer]:
            prior_units = question.unit_ids_by_trace.get(prior_trace, ())
            state.pairs_by_answer[answer].extend(
                (left, right) for left in prior_units for right in new_units
            )
        state.traces_by_answer[answer].append(trace_id)
        state.prefix_trace_ids.append(trace_id)


# ---------------------------------------------------------------------------
# Reuse one merge result for all community support thresholds
# ---------------------------------------------------------------------------


def filter_build_for_support(
    *,
    base_build: Any,
    units: Sequence[Any],
    n_answer_units: int,
    min_trace_support: int,
    support_impossible: bool,
) -> Any:
    if support_impossible:
        return BASE.CommunityBuildResult(
            communities=[],
            node_to_community={},
            diagnostics={
                **dict(base_build.diagnostics),
                "n_retained_communities": 0,
                "n_nodes_in_communities": 0,
                "n_units_answer": n_answer_units,
                "unit_coverage_fraction": 0.0,
                "support_impossible": 1,
            },
            rejection_rows=list(base_build.rejection_rows),
        )

    retained_old = [
        community
        for community in base_build.communities
        if community.support >= min_trace_support
    ]
    communities: list[Any] = []
    node_to_community: dict[int, int] = {}
    for new_id, old in enumerate(retained_old):
        community = BASE.LocalCommunity(
            community_id=new_id,
            nodes=list(old.nodes),
            trace_ids=list(old.trace_ids),
            support=int(old.support),
            min_score=float(old.min_score),
            q10_score=float(old.q10_score),
            mean_score=float(old.mean_score),
            representative_text=str(old.representative_text),
            mean_position=float(old.mean_position),
        )
        communities.append(community)
        for node in community.nodes:
            node_to_community[int(node)] = new_id

    diagnostics = dict(base_build.diagnostics)
    diagnostics["n_retained_communities"] = len(communities)
    diagnostics["n_nodes_in_communities"] = len(node_to_community)
    diagnostics["n_units_answer"] = n_answer_units
    diagnostics["unit_coverage_fraction"] = (
        len(node_to_community) / n_answer_units if n_answer_units else 0.0
    )
    diagnostics["max_community_size"] = max(
        (len(community.nodes) for community in communities), default=0
    )
    diagnostics["mean_community_size"] = float(
        np.mean([len(community.nodes) for community in communities])
        if communities
        else 0.0
    )
    diagnostics["reject_insufficient_trace_support"] = sum(
        len(community.nodes) > 1 and community.support < min_trace_support
        for community in base_build.communities
    )
    diagnostics["support_impossible"] = 0
    return BASE.CommunityBuildResult(
        communities=communities,
        node_to_community=node_to_community,
        diagnostics=diagnostics,
        rejection_rows=list(base_build.rejection_rows),
    )


# ---------------------------------------------------------------------------
# Shared sequence context for transition and precedence metrics
# ---------------------------------------------------------------------------


@dataclass
class StructuralSequenceContext:
    sequences: dict[int, list[int]]
    trace_ids: list[int]
    communities: list[Any]
    community_by_id: dict[int, Any]
    transition_support: Counter
    trace_transition_sets: dict[int, set[tuple[int, int]]]
    trace_transition_sequences: dict[int, tuple[tuple[int, int], ...]]
    precedence_records: list[dict[str, Any]]


def prepare_structural_sequence_context(
    *,
    question: Any,
    trace_ids: Sequence[int],
    communities: Sequence[Any],
    node_to_community: dict[int, int],
) -> StructuralSequenceContext:
    trace_ids = [int(value) for value in trace_ids]
    sequences = BASE.build_trace_sequences(
        question=question,
        trace_ids=trace_ids,
        node_to_community=node_to_community,
    )

    transition_support: Counter[tuple[int, int]] = Counter()
    trace_transition_sets: dict[int, set[tuple[int, int]]] = {}
    trace_transition_sequences: dict[int, tuple[tuple[int, int], ...]] = {}
    for trace_id, sequence in sequences.items():
        transitions = tuple(
            (left, right)
            for left, right in zip(sequence, sequence[1:])
            if left != right
        )
        trace_transition_sequences[trace_id] = transitions
        trace_transition_sets[trace_id] = set(transitions)
        transition_support.update(set(transitions))

    community_by_id = {
        community.community_id: community for community in communities
    }
    community_ids = sorted(community_by_id)
    positions = {
        trace_id: {
            community_id: index for index, community_id in enumerate(sequence)
        }
        for trace_id, sequence in sequences.items()
    }
    precedence_records: list[dict[str, Any]] = []
    for left_index, first in enumerate(community_ids):
        for second in community_ids[left_index + 1 :]:
            first_before = 0
            second_before = 0
            first_before_adjacent = 0
            second_before_adjacent = 0
            first_before_gaps: list[int] = []
            second_before_gaps: list[int] = []
            for trace_id in trace_ids:
                trace_positions = positions.get(trace_id, {})
                if first not in trace_positions or second not in trace_positions:
                    continue
                first_position = trace_positions[first]
                second_position = trace_positions[second]
                gap = abs(second_position - first_position)
                if first_position < second_position:
                    first_before += 1
                    first_before_gaps.append(gap)
                    if gap == 1:
                        first_before_adjacent += 1
                elif second_position < first_position:
                    second_before += 1
                    second_before_gaps.append(gap)
                    if gap == 1:
                        second_before_adjacent += 1
            precedence_records.append(
                {
                    "first": first,
                    "second": second,
                    "first_before": first_before,
                    "second_before": second_before,
                    "first_before_adjacent": first_before_adjacent,
                    "second_before_adjacent": second_before_adjacent,
                    "first_before_gaps": first_before_gaps,
                    "second_before_gaps": second_before_gaps,
                }
            )

    return StructuralSequenceContext(
        sequences=sequences,
        trace_ids=trace_ids,
        communities=list(communities),
        community_by_id=community_by_id,
        transition_support=transition_support,
        trace_transition_sets=trace_transition_sets,
        trace_transition_sequences=trace_transition_sequences,
        precedence_records=precedence_records,
    )


def transition_metrics_from_context(
    *,
    question: Any,
    answer: str,
    context: StructuralSequenceContext,
    transition_min_support: int,
    late_position_threshold: float,
    save_details: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    support = context.transition_support
    qualifying = {
        transition
        for transition, count in support.items()
        if count >= transition_min_support
    }
    supports = np.asarray([support[item] for item in sorted(qualifying)], dtype=float)
    concentration = BASE.entropy_concentration(supports)
    n_traces = len(context.trace_ids)

    filtered_sets: list[set[tuple[int, int]]] = []
    filtered_sequences: list[tuple[tuple[int, int], ...]] = []
    for trace_id in context.trace_ids:
        filtered_sequence = tuple(
            transition
            for transition in context.trace_transition_sequences.get(trace_id, ())
            if transition in qualifying
        )
        filtered_sequences.append(filtered_sequence)
        filtered_sets.append(set(filtered_sequence))

    jaccard_distances: list[float] = []
    supported_jaccard_sum = 0.0
    n_trace_pairs_total = 0
    n_trace_pairs_both_nonempty = 0
    for left in range(len(filtered_sets)):
        for right in range(left + 1, len(filtered_sets)):
            n_trace_pairs_total += 1
            union = filtered_sets[left] | filtered_sets[right]
            if union:
                distance = (
                    1.0
                    - len(filtered_sets[left] & filtered_sets[right]) / len(union)
                )
                jaccard_distances.append(distance)
            else:
                distance = 0.0
            if filtered_sets[left] and filtered_sets[right]:
                supported_jaccard_sum += distance
                n_trace_pairs_both_nonempty += 1

    supported_pairwise_jaccard = (
        supported_jaccard_sum / n_trace_pairs_total
        if n_trace_pairs_total
        else 0.0
    )
    transition_pair_evidence_coverage = (
        n_trace_pairs_both_nonempty / n_trace_pairs_total
        if n_trace_pairs_total
        else 0.0
    )
    set_stats = BASE.signature_stats(
        [tuple(sorted(values)) for values in filtered_sets], n_traces
    )
    sequence_stats = BASE.signature_stats(filtered_sequences, n_traces)

    adjacency: dict[int, set[int]] = defaultdict(set)
    for left, right in qualifying:
        adjacency[left].add(right)
    out_degrees = np.asarray(
        [len(values) for values in adjacency.values()], dtype=float
    )

    community_by_id = context.community_by_id
    branching_weight_total = 0.0
    branching_weight_mass = 0.0
    for community_id, destinations in adjacency.items():
        community = community_by_id.get(community_id)
        if community is None:
            continue
        weight = community.support / max(n_traces, 1)
        branching_weight_total += weight
        if len(destinations) > 1:
            branching_weight_mass += weight
    support_weighted_branching = (
        branching_weight_mass / branching_weight_total
        if branching_weight_total > 0
        else 0.0
    )

    late_transitions = {
        transition
        for transition in qualifying
        if min(
            community_by_id[transition[0]].mean_position,
            community_by_id[transition[1]].mean_position,
        )
        >= late_position_threshold
    }
    weighted_transition_support = 0.0
    detail_rows: list[dict[str, Any]] = []
    for left, right in sorted(qualifying):
        left_community = community_by_id[left]
        right_community = community_by_id[right]
        edge_cohesion = min(left_community.min_score, right_community.min_score)
        weighted_transition_support += (
            support[(left, right)] / max(n_traces, 1)
        ) * edge_cohesion
        if save_details:
            detail_rows.append(
                {
                    "id": question.question_id,
                    "answer": answer,
                    "transition_min_support": transition_min_support,
                    "from_community": left,
                    "to_community": right,
                    "support": support[(left, right)],
                    "support_fraction": support[(left, right)] / max(n_traces, 1),
                    "from_text": left_community.representative_text,
                    "to_text": right_community.representative_text,
                    "from_position": left_community.mean_position,
                    "to_position": right_community.mean_position,
                    "endpoint_min_cohesion": edge_cohesion,
                    "is_late": (left, right) in late_transitions,
                }
            )

    trace_with_transition = sum(bool(values) for values in filtered_sets)
    mean_jaccard = float(
        np.mean(jaccard_distances) if jaccard_distances else 0.0
    )
    return (
        {
            "n_shared_transition_types": float(len(qualifying)),
            "shared_transition_types_per_trace": len(qualifying) / max(n_traces, 1),
            "transition_trace_coverage": trace_with_transition / max(n_traces, 1),
            "transition_support_gini_simpson": concentration["gini_simpson"],
            "effective_shared_transitions_per_trace": (
                concentration["effective_number"] / max(n_traces, 1)
            ),
            "transition_set_diversity_gini_simpson": set_stats["gini_simpson"],
            "transition_sequence_diversity_gini_simpson": sequence_stats[
                "gini_simpson"
            ],
            "mean_pairwise_transition_jaccard_distance": mean_jaccard,
            "mean_pairwise_transition_jaccard_similarity": (
                1.0 - mean_jaccard if jaccard_distances else 1.0
            ),
            "supported_pairwise_transition_jaccard_distance": float(
                supported_pairwise_jaccard
            ),
            "transition_pair_evidence_coverage": float(
                transition_pair_evidence_coverage
            ),
            "n_transition_trace_pairs_total": float(n_trace_pairs_total),
            "n_transition_trace_pairs_both_nonempty": float(
                n_trace_pairs_both_nonempty
            ),
            "coverage_weighted_transition_set_diversity": float(
                (trace_with_transition / max(n_traces, 1))
                * set_stats["gini_simpson"]
            ),
            "fraction_branching_checkpoints": float(
                np.mean(out_degrees > 1) if out_degrees.size else 0.0
            ),
            "support_weighted_branching_fraction": float(
                support_weighted_branching
            ),
            "max_transition_out_degree": float(
                out_degrees.max() if out_degrees.size else 0.0
            ),
            "late_shared_transition_fraction": (
                len(late_transitions) / len(qualifying) if qualifying else 0.0
            ),
            "cohesion_weighted_transition_support": float(
                weighted_transition_support
            ),
            "n_trace_sequences_with_communities": float(
                sum(bool(sequence) for sequence in context.sequences.values())
            ),
        },
        detail_rows,
    )


def precedence_metrics_from_context(
    *,
    question: Any,
    answer: str,
    context: StructuralSequenceContext,
    precedence_min_support: int,
    precedence_min_consistency: float,
    save_details: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    community_by_id = context.community_by_id
    n_traces = len(context.trace_ids)
    supported_mass = 0.0
    shortcut_mass = 0.0
    reversal_numerator = 0.0
    reversal_denominator = 0.0
    n_supported = 0
    n_stable = 0
    n_shortcut = 0
    detail_rows: list[dict[str, Any]] = []
    stable_edges: list[tuple[int, int, float, int]] = []

    for record in context.precedence_records:
        first = int(record["first"])
        second = int(record["second"])
        first_before = int(record["first_before"])
        second_before = int(record["second_before"])
        support = first_before + second_before
        if support < precedence_min_support:
            continue
        n_supported += 1

        if first_before >= second_before:
            source, target = first, second
            directional_count = first_before
            reverse_count = second_before
            adjacent_count = int(record["first_before_adjacent"])
            directional_gaps = record["first_before_gaps"]
        else:
            source, target = second, first
            directional_count = second_before
            reverse_count = first_before
            adjacent_count = int(record["second_before_adjacent"])
            directional_gaps = record["second_before_gaps"]

        consistency = directional_count / support
        reversal_rate = reverse_count / support
        direction_strength = max(0.0, 2.0 * consistency - 1.0)
        support_fraction = support / max(n_traces, 1)
        nonadjacent_count = directional_count - adjacent_count
        stable = consistency >= precedence_min_consistency
        shortcut_survives = bool(
            stable and adjacent_count > 0 and nonadjacent_count > 0
        )

        supported_mass += support_fraction * direction_strength
        reversal_numerator += reverse_count
        reversal_denominator += support
        if stable:
            n_stable += 1
            stable_edges.append((source, target, consistency, support))
        if shortcut_survives:
            n_shortcut += 1
            shortcut_mass += support_fraction * direction_strength

        if save_details:
            detail_rows.append(
                {
                    "id": question.question_id,
                    "answer": answer,
                    "precedence_min_support": precedence_min_support,
                    "from_community": source,
                    "to_community": target,
                    "cooccurrence_support": support,
                    "support_fraction": support_fraction,
                    "forward_count": directional_count,
                    "reverse_count": reverse_count,
                    "ordering_consistency": consistency,
                    "reversal_rate": reversal_rate,
                    "direction_strength": direction_strength,
                    "adjacent_forward_count": adjacent_count,
                    "nonadjacent_forward_count": nonadjacent_count,
                    "shortcut_survives": shortcut_survives,
                    "mean_forward_gap": (
                        float(np.mean(directional_gaps))
                        if directional_gaps
                        else np.nan
                    ),
                    "from_text": community_by_id[source].representative_text,
                    "to_text": community_by_id[target].representative_text,
                    "from_position": community_by_id[source].mean_position,
                    "to_position": community_by_id[target].mean_position,
                }
            )

    acyclic_adjacency: dict[int, set[int]] = defaultdict(set)
    cycle_conflicts = 0
    for source, target, consistency, support in sorted(
        stable_edges,
        key=lambda value: (-value[2], -value[3], value[0], value[1]),
    ):
        if BASE._path_exists(acyclic_adjacency, target, source):
            cycle_conflicts += 1
            continue
        acyclic_adjacency[source].add(target)

    chain_length = BASE._longest_path_nodes(acyclic_adjacency)
    n_communities = len(community_by_id)
    return (
        {
            "n_supported_precedence_pairs": float(n_supported),
            "n_stable_precedence_pairs": float(n_stable),
            "stable_precedence_pairs_per_trace": n_stable / max(n_traces, 1),
            "supported_precedence_mass": float(supported_mass),
            "precedence_reversal_rate": float(
                reversal_numerator / reversal_denominator
                if reversal_denominator > 0
                else 0.0
            ),
            "n_shortcut_surviving_pairs": float(n_shortcut),
            "shortcut_survival_mass": float(shortcut_mass),
            "consensus_chain_length": float(chain_length),
            "consensus_chain_fraction": float(
                chain_length / n_communities if n_communities else 0.0
            ),
            "n_precedence_cycle_conflicts": float(cycle_conflicts),
        },
        detail_rows,
    )


# ---------------------------------------------------------------------------
# Optimized per-question loop
# ---------------------------------------------------------------------------


def empty_direct_values() -> dict[str, float]:
    return {
        "mean_best_trace_pair_score": np.nan,
        "mean_top2_trace_pair_score": np.nan,
        "mean_greedy_one_to_one_score": 0.0,
        "mean_order_consistency": 0.0,
        "trace_pair_match_coverage": 0.0,
        "trace_pair_two_match_coverage": 0.0,
        "n_trace_pairs": 0.0,
    }


def optimized_process_question(
    *,
    question: Any,
    experiment_mode: str,
    pipeline: Any,
    embedder: Any | None,
    reranker: Any,
    args: argparse.Namespace,
    category: str,
) -> tuple[Any, ...]:
    if experiment_mode != "sampled_allpairs":
        # This wrapper is optimized for the exact-prefix all-pairs experiment.
        return BASE.process_question_original(
            question=question,
            experiment_mode=experiment_mode,
            pipeline=pipeline,
            embedder=embedder,
            reranker=reranker,
            args=args,
            category=category,
        )

    score_store = BASE.PairScoreStore(units=question.units, reranker=reranker)
    unit_artifact_rows: list[dict[str, Any]] = []
    sampling_artifact_rows: list[dict[str, Any]] = []
    pair_score_artifact_rows: list[dict[str, Any]] = []
    if args.save_recalculation_artifacts:
        for unit in question.units:
            unit_artifact_rows.append(
                {
                    "dataset_split": args.dataset_split,
                    "id": question.question_id,
                    "unit_id": unit.unit_id,
                    "trace_id": unit.trace_id,
                    "answer": unit.answer,
                    "is_correct_answer": question.is_correct_by_answer.get(
                        unit.answer, False
                    ),
                    "full_raw_count": question.raw_count_by_answer.get(
                        unit.answer, 0
                    ),
                    "local_index": unit.local_index,
                    "relative_position": unit.position,
                    "raw_step_text": unit.text,
                    "embedding_text": unit.embedding_text,
                    "step_chars": len(unit.text),
                    "step_words": BASE.checkpoint_word_count(unit.text),
                }
            )

    answer_rows: list[dict[str, Any]] = []
    community_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    precedence_rows: list[dict[str, Any]] = []
    trace_schema_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    trace_pair_rows: list[dict[str, Any]] = []

    n_question_traces = len(question.answers_by_trace)
    requested_prefixes = [
        (
            int(sample_size),
            n_question_traces
            if int(sample_size) <= 0
            else min(int(sample_size), n_question_traces),
        )
        for sample_size in args.sample_size_values
    ]
    # Prefix states can be updated only in increasing actual-prefix order.
    requested_prefixes.sort(key=lambda value: (value[1], value[0]))

    for sampling_seed in args.sampling_seed_values:
        prefix_order = BASE.question_trace_order(
            question, sampling_seed, args.prefix_order
        )
        max_prefix_count = max((count for _, count in requested_prefixes), default=0)
        prescore_largest_prefix(
            question=question,
            prefix_trace_ids=prefix_order[:max_prefix_count],
            score_store=score_store,
            chunk_size=args.prescore_pair_chunk_size,
            show_progress=args.show_prescore_progress,
        )

        state = empty_prefix_state()
        prefix_position = {
            int(trace_id): position for position, trace_id in enumerate(prefix_order)
        }

        for sample_size, prefix_count in requested_prefixes:
            extend_prefix_state(
                state=state,
                question=question,
                ordered_trace_ids=prefix_order,
                new_prefix_count=prefix_count,
            )
            prefix_trace_ids = list(state.prefix_trace_ids)
            observed_answers = list(state.observed_answers)
            selected_traces_by_answer = {
                answer: list(state.traces_by_answer[answer])
                for answer in observed_answers
            }
            answer_candidate_pairs = {
                answer: state.pairs_by_answer[answer]
                for answer in observed_answers
            }
            observed_count_by_answer = {
                answer: len(selected_traces_by_answer[answer])
                for answer in observed_answers
            }
            observed_first_position_by_answer = {
                answer: min(
                    prefix_position[trace_id]
                    for trace_id in selected_traces_by_answer[answer]
                )
                for answer in observed_answers
            }

            if args.save_recalculation_artifacts:
                for answer in observed_answers:
                    selected = selected_traces_by_answer[answer]
                    all_trace_ids = question.traces_by_answer[answer]
                    sampling_artifact_rows.append(
                        {
                            "dataset_split": args.dataset_split,
                            "id": question.question_id,
                            "sampling_seed": sampling_seed,
                            "prefix_order": args.prefix_order,
                            "requested_sample_size": sample_size,
                            "actual_prefix_size": prefix_count,
                            "prefix_trace_ids_json": BASE.json_dumps(
                                prefix_trace_ids
                            ),
                            "answer": answer,
                            "is_correct_answer": question.is_correct_by_answer[
                                answer
                            ],
                            "raw_count": len(selected),
                            "full_raw_count": question.raw_count_by_answer[answer],
                            "available_trace_count": len(all_trace_ids),
                            "sampled_trace_count": len(selected),
                            "sampled_trace_ids_json": BASE.json_dumps(selected),
                            "sampled_all_available": len(selected)
                            == len(all_trace_ids),
                        }
                    )

            majority_fields = BASE.sample_size_majority_fields(
                question=question,
                prefix_trace_ids=prefix_trace_ids,
            )

            for score_mode in args.score_mode_values:
                for answer in observed_answers:
                    all_trace_ids = question.traces_by_answer[answer]
                    selected_trace_ids = selected_traces_by_answer[answer]
                    unit_ids = [
                        unit_id
                        for trace_id in selected_trace_ids
                        for unit_id in question.unit_ids_by_trace.get(trace_id, ())
                    ]
                    candidate_pairs = answer_candidate_pairs[answer]
                    candidate_edges = (
                        score_store.scored_edges(candidate_pairs, score_mode)
                        if candidate_pairs
                        else []
                    )
                    final_edges = BASE.top_k_edges(
                        edges=candidate_edges,
                        unit_ids=unit_ids,
                        k=0,
                        edge_mode=args.local_edge_mode,
                    )
                    finite_scores = np.asarray(
                        [edge[2] for edge in final_edges], dtype=float
                    )

                    for edge_quantile in args.edge_quantile_values:
                        edge_threshold = (
                            float(np.quantile(finite_scores, edge_quantile))
                            if finite_scores.size
                            else float("inf")
                        )

                        direct_values_once = empty_direct_values()
                        pair_details_once: list[dict[str, Any]] = []
                        if (
                            args.compute_direct_trace_pair_metrics
                            and len(selected_trace_ids) >= 2
                            and np.isfinite(edge_threshold)
                        ):
                            allowed_direct_pairs = {
                                BASE.PairScoreStore.key(left, right)
                                for left, right, _ in final_edges
                            }
                            direct_values_once, pair_details_once = (
                                BASE.trace_pair_metrics(
                                    question=question,
                                    answer=answer,
                                    trace_ids=selected_trace_ids,
                                    score_store=score_store,
                                    score_mode=score_mode,
                                    match_threshold=edge_threshold,
                                    allowed_pairs=allowed_direct_pairs,
                                )
                            )

                        pair_details_saved_for_threshold = False
                        for cohesion_mode in args.cohesion_mode_values:
                            # Merge once. min_trace_support affects only the final
                            # retention filter in v8, not any merge decision.
                            if not np.isfinite(edge_threshold):
                                base_build = BASE.CommunityBuildResult(
                                    communities=[],
                                    node_to_community={},
                                    diagnostics={
                                        "n_proposal_edges": len(final_edges),
                                        "n_edges_at_threshold": 0,
                                        "n_merge_attempts": 0,
                                        "accepted_merges": 0,
                                        "n_retained_communities": 0,
                                        "n_nodes_in_communities": 0,
                                        "n_units_answer": len(unit_ids),
                                        "unit_coverage_fraction": 0.0,
                                    },
                                    rejection_rows=[],
                                )
                            else:
                                base_build = BASE.build_constrained_communities(
                                    question_id=question.question_id,
                                    answer=answer,
                                    units=question.units,
                                    unit_ids=unit_ids,
                                    proposal_edges=final_edges,
                                    score_store=score_store,
                                    score_mode=score_mode,
                                    edge_threshold=edge_threshold,
                                    cohesion_mode=cohesion_mode,
                                    cohesion_margin=args.cohesion_margin,
                                    min_trace_support=1,
                                    max_rejections=args.max_rejections_per_answer,
                                )

                            for community_min_support in args.community_support_values:
                                if (
                                    sample_size > 0
                                    and community_min_support > sample_size
                                ):
                                    continue
                                support_impossible = (
                                    len(selected_trace_ids) < community_min_support
                                )
                                build = filter_build_for_support(
                                    base_build=base_build,
                                    units=question.units,
                                    n_answer_units=len(unit_ids),
                                    min_trace_support=community_min_support,
                                    support_impossible=support_impossible,
                                )

                                base_config = {
                                    "dataset_split": args.dataset_split,
                                    "id": question.question_id,
                                    "category": category,
                                    "experiment_mode": experiment_mode,
                                    "sampling_seed": sampling_seed,
                                    "prefix_order": args.prefix_order,
                                    "sample_size": sample_size,
                                    "score_mode": score_mode,
                                    "edge_quantile": edge_quantile,
                                    "local_neighbor_k": 0,
                                    "cohesion_mode": cohesion_mode,
                                    "community_min_support": community_min_support,
                                }

                                if args.save_trace_schemas:
                                    for schema_row in BASE.build_trace_schema_rows(
                                        question=question,
                                        answer=answer,
                                        trace_ids=selected_trace_ids,
                                        communities=build.communities,
                                        node_to_community=build.node_to_community,
                                    ):
                                        trace_schema_rows.append(
                                            {
                                                **base_config,
                                                **schema_row,
                                                "raw_count": observed_count_by_answer[
                                                    answer
                                                ],
                                                "full_raw_count": question.raw_count_by_answer[
                                                    answer
                                                ],
                                            }
                                        )

                                community_values = BASE.community_metrics(
                                    question=question,
                                    communities=build.communities,
                                    n_traces=len(selected_trace_ids),
                                    n_units=len(unit_ids),
                                )
                                sequence_context = prepare_structural_sequence_context(
                                    question=question,
                                    trace_ids=selected_trace_ids,
                                    communities=build.communities,
                                    node_to_community=build.node_to_community,
                                )

                                for transition_min_support in args.transition_support_values:
                                    if (
                                        sample_size > 0
                                        and transition_min_support > sample_size
                                    ):
                                        continue
                                    transition_values, details = (
                                        transition_metrics_from_context(
                                            question=question,
                                            answer=answer,
                                            context=sequence_context,
                                            transition_min_support=transition_min_support,
                                            late_position_threshold=args.late_position_threshold,
                                            save_details=args.save_transition_details,
                                        )
                                    )
                                    precedence_values, precedence_details = (
                                        precedence_metrics_from_context(
                                            question=question,
                                            answer=answer,
                                            context=sequence_context,
                                            precedence_min_support=transition_min_support,
                                            precedence_min_consistency=args.precedence_min_consistency,
                                            save_details=args.save_precedence_details,
                                        )
                                    )
                                    precedence_values[
                                        "atomic_ordered_backbone_mass"
                                    ] = float(
                                        precedence_values[
                                            "supported_precedence_mass"
                                        ]
                                        / math.sqrt(
                                            max(
                                                community_values[
                                                    "mean_checkpoint_words"
                                                ],
                                                1.0,
                                            )
                                        )
                                    )

                                    if not candidate_pairs:
                                        evidence_stage = "candidate_failure"
                                    elif not final_edges:
                                        evidence_stage = "edge_selection_failure"
                                    elif not build.communities:
                                        evidence_stage = "community_failure"
                                    elif (
                                        transition_values[
                                            "n_shared_transition_types"
                                        ]
                                        <= 0
                                    ):
                                        evidence_stage = "transition_failure"
                                    else:
                                        evidence_stage = (
                                            "transition_evidence_available"
                                        )

                                    if not build.communities:
                                        order_evidence_stage = "community_failure"
                                    elif (
                                        precedence_values[
                                            "n_supported_precedence_pairs"
                                        ]
                                        <= 0
                                    ):
                                        order_evidence_stage = (
                                            "no_supported_precedence"
                                        )
                                    elif (
                                        precedence_values[
                                            "n_stable_precedence_pairs"
                                        ]
                                        <= 0
                                    ):
                                        order_evidence_stage = (
                                            "precedence_reversal_failure"
                                        )
                                    elif (
                                        precedence_values[
                                            "n_shortcut_surviving_pairs"
                                        ]
                                        > 0
                                    ):
                                        order_evidence_stage = (
                                            "shortcut_evidence_available"
                                        )
                                    else:
                                        order_evidence_stage = (
                                            "stable_precedence_available"
                                        )

                                    answer_rows.append(
                                        {
                                            **base_config,
                                            "transition_min_support": transition_min_support,
                                            "answer": answer,
                                            "evidence_stage": evidence_stage,
                                            "order_evidence_stage": order_evidence_stage,
                                            "raw_count": observed_count_by_answer[answer],
                                            "full_raw_count": question.raw_count_by_answer[
                                                answer
                                            ],
                                            "available_trace_count": len(
                                                all_trace_ids
                                            ),
                                            "sampled_trace_count": len(
                                                selected_trace_ids
                                            ),
                                            "sampled_all_available": bool(
                                                len(selected_trace_ids)
                                                == len(all_trace_ids)
                                            ),
                                            "sampling_fraction": (
                                                len(selected_trace_ids)
                                                / len(all_trace_ids)
                                                if all_trace_ids
                                                else 0.0
                                            ),
                                            "first_trace_position": observed_first_position_by_answer[
                                                answer
                                            ],
                                            "is_correct_answer": question.is_correct_by_answer[
                                                answer
                                            ],
                                            **majority_fields,
                                            "edge_threshold": edge_threshold,
                                            "n_units_answer": len(unit_ids),
                                            "n_embedding_candidate_pairs_within_answer": len(
                                                candidate_pairs
                                            ),
                                            "n_final_local_edges": len(final_edges),
                                            **community_values,
                                            **transition_values,
                                            **precedence_values,
                                            **direct_values_once,
                                        }
                                    )

                                    if args.save_transition_details:
                                        transition_rows.extend(
                                            {
                                                **base_config,
                                                "transition_min_support": transition_min_support,
                                                **detail,
                                            }
                                            for detail in details
                                        )
                                    if args.save_precedence_details:
                                        precedence_rows.extend(
                                            {
                                                **base_config,
                                                "transition_min_support": transition_min_support,
                                                "precedence_min_consistency": args.precedence_min_consistency,
                                                **detail,
                                            }
                                            for detail in precedence_details
                                        )
                                    if (
                                        args.save_trace_pair_details
                                        and not pair_details_saved_for_threshold
                                        and transition_min_support
                                        == args.transition_support_values[0]
                                    ):
                                        trace_pair_rows.extend(
                                            {**base_config, **detail}
                                            for detail in pair_details_once
                                        )
                                        pair_details_saved_for_threshold = True

                                if args.save_community_details:
                                    for community in build.communities:
                                        community_rows.append(
                                            {
                                                **base_config,
                                                "answer": answer,
                                                "community_id": community.community_id,
                                                "support": community.support,
                                                "support_fraction": community.support
                                                / max(len(selected_trace_ids), 1),
                                                "n_nodes": len(community.nodes),
                                                "trace_ids_json": BASE.json_dumps(
                                                    community.trace_ids
                                                ),
                                                "node_ids_json": BASE.json_dumps(
                                                    community.nodes
                                                ),
                                                "member_texts_json": BASE.json_dumps(
                                                    [
                                                        question.units[node].text
                                                        for node in community.nodes
                                                    ]
                                                ),
                                                "representative_text": community.representative_text,
                                                "min_score": community.min_score,
                                                "q10_score": community.q10_score,
                                                "mean_score": community.mean_score,
                                                "mean_position": community.mean_position,
                                            }
                                        )

                                if args.max_rejections_per_answer > 0:
                                    rejection_rows.extend(
                                        {**base_config, **row}
                                        for row in build.rejection_rows
                                    )

                                if args.save_stage_diagnostics:
                                    diagnostics = {
                                        **base_config,
                                        "answer": answer,
                                        "raw_count": observed_count_by_answer[answer],
                                        "full_raw_count": question.raw_count_by_answer[
                                            answer
                                        ],
                                        "available_trace_count": len(all_trace_ids),
                                        "sampled_trace_count": len(
                                            selected_trace_ids
                                        ),
                                        "sampled_all_available": bool(
                                            len(selected_trace_ids)
                                            == len(all_trace_ids)
                                        ),
                                        "sampling_fraction": (
                                            len(selected_trace_ids)
                                            / len(all_trace_ids)
                                            if all_trace_ids
                                            else 0.0
                                        ),
                                        "n_units_answer": len(unit_ids),
                                        "n_global_embedding_candidates": 0,
                                        "n_within_answer_candidate_pairs": len(
                                            candidate_pairs
                                        ),
                                        "n_final_local_edges": len(final_edges),
                                        "edge_threshold": edge_threshold,
                                        "candidate_score_q50": (
                                            float(np.quantile(finite_scores, 0.50))
                                            if finite_scores.size
                                            else np.nan
                                        ),
                                        "candidate_score_q90": (
                                            float(np.quantile(finite_scores, 0.90))
                                            if finite_scores.size
                                            else np.nan
                                        ),
                                        **build.diagnostics,
                                    }
                                    if not candidate_pairs:
                                        diagnostics[
                                            "failure_stage"
                                        ] = "no_within_answer_candidates"
                                    elif not final_edges:
                                        diagnostics[
                                            "failure_stage"
                                        ] = "no_final_edges"
                                    elif not build.communities:
                                        diagnostics[
                                            "failure_stage"
                                        ] = "no_supported_communities"
                                    else:
                                        diagnostics[
                                            "failure_stage"
                                        ] = "community_evidence_available"
                                    diagnostic_rows.append(diagnostics)

    if args.save_recalculation_artifacts:
        for (left, right), (forward_raw, reverse_raw) in sorted(
            score_store.raw.items()
        ):
            left_unit = question.units[left]
            right_unit = question.units[right]
            pair_score_artifact_rows.append(
                {
                    "dataset_split": args.dataset_split,
                    "id": question.question_id,
                    "left_unit_id": left,
                    "right_unit_id": right,
                    "left_trace_id": left_unit.trace_id,
                    "right_trace_id": right_unit.trace_id,
                    "left_answer": left_unit.answer,
                    "right_answer": right_unit.answer,
                    "same_answer": left_unit.answer == right_unit.answer,
                    "left_text": left_unit.text,
                    "right_text": right_unit.text,
                    "forward_raw_logit": forward_raw,
                    "reverse_raw_logit": reverse_raw,
                    "raw_mean": BASE.score_from_directional(
                        forward_raw, reverse_raw, "raw_mean"
                    ),
                    "raw_min": BASE.score_from_directional(
                        forward_raw, reverse_raw, "raw_min"
                    ),
                    "sigmoid_mean": BASE.score_from_directional(
                        forward_raw, reverse_raw, "sigmoid_mean"
                    ),
                    "sigmoid_min": BASE.score_from_directional(
                        forward_raw, reverse_raw, "sigmoid_min"
                    ),
                }
            )

    return (
        answer_rows,
        community_rows,
        transition_rows,
        precedence_rows,
        trace_schema_rows,
        diagnostic_rows,
        rejection_rows + trace_pair_rows,
        unit_artifact_rows,
        sampling_artifact_rows,
        pair_score_artifact_rows,
    )


# Preserve a callable reference before monkey-patching.
BASE.process_question_original = BASE.process_question
BASE.process_question = optimized_process_question




# ---------------------------------------------------------------------------
# Safe resume planning
# ---------------------------------------------------------------------------


def _read_distinct_ids_from_csv(
    path: Path,
    *,
    id_column: str,
    chunksize: int = 250_000,
) -> set[str]:
    """Read only one ID column from a potentially very large CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint file does not exist: {path}")

    result: set[str] = set()
    try:
        reader = pd.read_csv(
            path,
            usecols=[id_column],
            chunksize=chunksize,
            low_memory=False,
        )
        for chunk in reader:
            result.update(
                normalized
                for normalized in chunk[id_column].map(normalize_question_id)
                if normalized
            )
    except ValueError as error:
        raise ValueError(
            f"Could not read ID column {id_column!r} from {path}: {error}"
        ) from error
    return result


def _input_id_order(path: Path, *, id_column: str) -> list[str]:
    """Return question IDs once, preserving their order in the expanded CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")
    frame = pd.read_csv(path, usecols=[id_column], low_memory=False)
    return list(
        dict.fromkeys(
            normalized
            for normalized in frame[id_column].map(normalize_question_id)
            if normalized
        )
    )


def prepare_resume_selection(args: argparse.Namespace) -> dict[str, Any]:
    """
    Determine a safe contiguous completed prefix and select only later IDs.

    The failed output directory is never modified.  The resumed computation
    must use a different --output-dir.  This avoids overwriting the last valid
    atomic CSV snapshots and avoids mixing tables saved at different moments.
    """
    resume_dir_value = getattr(args, "resume_from_output_dir", None)
    manual_after = getattr(args, "resume_after_id", None)
    if not resume_dir_value and manual_after is None:
        return {
            "resume_enabled": False,
            "resume_from_output_dir": None,
            "safe_last_completed_id": None,
            "n_safe_completed_ids": 0,
            "n_remaining_ids": None,
        }

    if args.question_ids or args.question_ids_file:
        raise ValueError(
            "Do not combine --resume-from-output-dir/--resume-after-id with "
            "--question-ids or --question-ids-file. Resume chooses the IDs."
        )

    original_input = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_ids = _input_id_order(
        original_input,
        id_column=args.question_id_column,
    )
    if not input_ids:
        raise ValueError("The input contains no question IDs.")

    checked_files: list[str] = []
    id_counts: dict[str, int] = {}

    if manual_after is not None:
        normalized_after = normalize_question_id(manual_after)
        if normalized_after not in input_ids:
            raise ValueError(
                f"--resume-after-id={manual_after!r} is not present in the input."
            )
        prefix_length = input_ids.index(normalized_after) + 1
        method = "manual_after_id"
    else:
        resume_dir = Path(resume_dir_value).expanduser().resolve()
        if output_dir == resume_dir:
            raise ValueError(
                "For safety, resumed --output-dir must differ from "
                "--resume-from-output-dir. Use a new part directory."
            )
        if not resume_dir.exists():
            raise FileNotFoundError(
                f"Failed output directory does not exist: {resume_dir}"
            )

        # answer_metrics is guaranteed to contain rows for every successfully
        # processed question. Trace schemas and stage diagnostics are also
        # guaranteed per question when the corresponding save flags are on.
        checkpoint_paths = [resume_dir / "answer_metrics.csv"]
        if args.save_trace_schemas:
            checkpoint_paths.append(resume_dir / "trace_checkpoint_schemas.csv")
        if args.save_stage_diagnostics:
            checkpoint_paths.append(resume_dir / "stage_diagnostics.csv")

        id_sets: list[set[str]] = []
        for path in checkpoint_paths:
            ids = _read_distinct_ids_from_csv(
                path,
                id_column=args.question_id_column,
            )
            id_sets.append(ids)
            checked_files.append(str(path))
            id_counts[path.name] = len(ids)

        common_ids = set.intersection(*id_sets) if id_sets else set()
        prefix_length = 0
        for question_id in input_ids:
            if question_id not in common_ids:
                break
            prefix_length += 1
        method = "longest_common_contiguous_prefix"

    reprocess_last = int(getattr(args, "resume_reprocess_last", 0))
    if reprocess_last < 0:
        raise ValueError("--resume-reprocess-last must be non-negative.")
    prefix_length = max(0, prefix_length - reprocess_last)

    completed_ids = input_ids[:prefix_length]
    remaining_ids = input_ids[prefix_length:]
    safe_last_id = completed_ids[-1] if completed_ids else None

    metadata = {
        "resume_enabled": True,
        "resume_method": method,
        "original_input": str(original_input),
        "resume_from_output_dir": (
            str(Path(resume_dir_value).expanduser().resolve())
            if resume_dir_value
            else None
        ),
        "resumed_output_dir": str(output_dir),
        "checkpoint_files_checked": checked_files,
        "checkpoint_file_distinct_id_counts": id_counts,
        "safe_last_completed_id": safe_last_id,
        "n_safe_completed_ids": len(completed_ids),
        "n_remaining_ids": len(remaining_ids),
        "resume_reprocess_last": reprocess_last,
        "first_remaining_id": remaining_ids[0] if remaining_ids else None,
        "last_remaining_id": remaining_ids[-1] if remaining_ids else None,
        "remaining_question_ids": remaining_ids,
    }
    (output_dir / "resume_plan.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        "[resume] safe completed prefix: "
        f"{len(completed_ids)} IDs; last={safe_last_id!r}",
        flush=True,
    )
    print(
        "[resume] remaining: "
        f"{len(remaining_ids)} IDs; "
        f"first={metadata['first_remaining_id']!r}; "
        f"last={metadata['last_remaining_id']!r}",
        flush=True,
    )
    print(f"[resume] plan: {output_dir / 'resume_plan.json'}", flush=True)

    if not remaining_ids:
        print("[resume] nothing remains to process.", flush=True)
        return metadata

    # Reuse the selected-ID filtering already implemented below.
    args.question_ids = ",".join(remaining_ids)
    return metadata


# ---------------------------------------------------------------------------
# Optional question-ID filtering
# ---------------------------------------------------------------------------


def normalize_question_id(value: Any) -> str:
    """Normalize numeric-looking IDs without changing nonnumeric IDs."""
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return ""
        if float(value).is_integer():
            return str(int(value))
        return str(value).strip()

    text = str(value).strip()
    if not text:
        return ""

    # Normalize values such as "17.0" to "17", while preserving IDs such as
    # "q17", UUIDs, and strings with leading zeros.
    try:
        numeric = float(text)
    except ValueError:
        return text
    if (
        numeric.is_integer()
        and "." in text
        and text.replace(".", "", 1).replace("-", "", 1).isdigit()
    ):
        return str(int(numeric))
    return text


def parse_question_id_spec(spec: str | None) -> list[str]:
    """
    Parse comma/whitespace-separated IDs and inclusive integer ranges.

    Examples:
        "0,3,8"
        "0-9,25,31-35"
        "q1 q2 q3"
    """
    if not spec:
        return []

    tokens = (
        str(spec)
        .replace("\n", ",")
        .replace("\t", ",")
        .replace(" ", ",")
        .split(",")
    )
    result: list[str] = []
    seen: set[str] = set()

    for raw_token in tokens:
        token = raw_token.strip()
        if not token:
            continue

        # Treat a-b as a range only when both endpoints are integers.
        range_parts = token.split("-", 1)
        is_integer_range = (
            len(range_parts) == 2
            and range_parts[0].strip().lstrip("+").isdigit()
            and range_parts[1].strip().lstrip("+").isdigit()
        )
        if is_integer_range:
            start = int(range_parts[0])
            end = int(range_parts[1])
            step = 1 if end >= start else -1
            expanded = [str(value) for value in range(start, end + step, step)]
        else:
            expanded = [normalize_question_id(token)]

        for value in expanded:
            if value and value not in seen:
                result.append(value)
                seen.add(value)

    return result


def read_question_ids_file(
    path: str | Path,
    *,
    id_column: str,
) -> list[str]:
    """
    Read selected IDs from a TXT/list file or from a CSV/TSV table.

    For tabular files, use ``id_column`` when present; otherwise a one-column
    table is accepted.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Question-ID file does not exist: {path}")

    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        table = pd.read_csv(path, sep=sep, low_memory=False)
        if id_column in table.columns:
            values = table[id_column].tolist()
        elif len(table.columns) == 1:
            values = table.iloc[:, 0].tolist()
        else:
            raise ValueError(
                f"{path} has no '{id_column}' column and has multiple columns: "
                f"{list(table.columns)}"
            )
        ids = [normalize_question_id(value) for value in values]
        return list(dict.fromkeys(value for value in ids if value))

    return parse_question_id_spec(path.read_text(encoding="utf-8"))


def prepare_selected_question_input(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """
    Filter the expanded input CSV before the unchanged v8 loader sees it.

    Returns metadata describing the selection. With no selection flags, the
    original input path is left untouched.
    """
    requested_ids: list[str] = []
    requested_ids.extend(parse_question_id_spec(args.question_ids))

    if args.question_ids_file:
        requested_ids.extend(
            read_question_ids_file(
                args.question_ids_file,
                id_column=args.question_id_column,
            )
        )

    # Preserve the user's order while removing duplicates.
    requested_ids = list(dict.fromkeys(requested_ids))
    if not requested_ids:
        return {
            "question_filter_enabled": False,
            "original_input": str(args.input),
            "filtered_input": None,
            "requested_question_ids": [],
            "found_question_ids": [],
            "missing_question_ids": [],
            "n_selected_questions": None,
            "n_selected_rows": None,
        }

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")

    data = pd.read_csv(input_path, low_memory=False)
    if args.question_id_column not in data.columns:
        raise ValueError(
            f"Input is missing question ID column '{args.question_id_column}'. "
            f"Available columns: {list(data.columns)}"
        )

    normalized_input_ids = data[args.question_id_column].map(normalize_question_id)
    available_ids = set(normalized_input_ids)
    requested_set = set(requested_ids)
    found_ids = [value for value in requested_ids if value in available_ids]
    missing_ids = [value for value in requested_ids if value not in available_ids]

    if missing_ids and not args.allow_missing_question_ids:
        preview = ", ".join(missing_ids[:20])
        suffix = " ..." if len(missing_ids) > 20 else ""
        raise ValueError(
            f"{len(missing_ids)} requested question IDs were not found in "
            f"{input_path}: {preview}{suffix}. "
            "Use --allow-missing-question-ids to ignore them."
        )
    if not found_ids:
        raise ValueError("None of the requested question IDs were found.")

    filtered = data.loc[normalized_input_ids.isin(requested_set)].copy()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_path = output_dir / "_selected_questions_input.csv"
    filtered.to_csv(filtered_path, index=False)

    # The original v8 code now reads only the selected rows.
    args.input = str(filtered_path)

    metadata = {
        "question_filter_enabled": True,
        "original_input": str(input_path),
        "filtered_input": str(filtered_path),
        "question_id_column": args.question_id_column,
        "requested_question_ids": requested_ids,
        "found_question_ids": found_ids,
        "missing_question_ids": missing_ids,
        "n_requested_questions": len(requested_ids),
        "n_selected_questions": len(set(found_ids)),
        "n_selected_rows": int(len(filtered)),
    }
    (output_dir / "selected_question_ids.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        "[question filter] selected "
        f"{metadata['n_selected_questions']} questions / "
        f"{metadata['n_selected_rows']} rows"
    )
    if missing_ids:
        print(
            "[question filter] ignored missing IDs: "
            + ", ".join(missing_ids[:20])
            + (" ..." if len(missing_ids) > 20 else "")
        )
    print(f"[question filter] filtered input: {filtered_path}")

    return metadata

def requested_metrics_from_args(args: argparse.Namespace) -> list[str]:
    if args.metrics == "transition_ordering":
        return list(TRANSITION_ORDERING_METRICS)
    if args.metrics == "selected":
        return list(BASE.SELECTED_METRICS)
    if args.metrics == "focused":
        return list(BASE.FOCUSED_METRICS)
    if args.metrics == "default":
        return list(BASE.DEFAULT_METRICS)
    return BASE.parse_str_list(args.metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = BASE.build_parser("sampled_allpairs")
    parser.set_defaults(save_every=0)
    parser.add_argument(
        "--prescore-pair-chunk-size",
        type=int,
        default=200_000,
        help=(
            "Maximum unit pairs submitted to the cache/reranker in one prescore "
            "chunk. This is not the GPU batch size."
        ),
    )
    parser.add_argument(
        "--show-prescore-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a per-question progress bar for full-prefix pair prescoring.",
    )
    parser.add_argument(
        "--compute-direct-trace-pair-metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Calculate the expensive direct trace-pair matching metrics. "
            "Disabled by default because transition/precedence metrics do not use them."
        ),
    )
    parser.add_argument(
        "--save-transition-details",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save transition_details.csv. Transition summary metrics are always calculated.",
    )
    parser.add_argument(
        "--save-precedence-details",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save precedence_details.csv. Precedence summary metrics are always calculated.",
    )
    parser.add_argument(
        "--save-community-details",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save community_details.csv.",
    )
    parser.add_argument(
        "--save-trace-schemas",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save trace_checkpoint_schemas.csv.",
    )
    parser.add_argument(
        "--save-stage-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save stage_diagnostics.csv.",
    )
    parser.add_argument(
        "--save-trace-pair-details",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save trace_pair_metrics.csv; requires direct trace-pair calculation.",
    )
    parser.add_argument(
        "--question-ids",
        default=None,
        help=(
            "Run only selected question IDs. Accepts comma/whitespace-separated "
            "IDs and inclusive integer ranges, e.g. '0,3,10-25'."
        ),
    )
    parser.add_argument(
        "--question-ids-file",
        default=None,
        help=(
            "TXT, CSV, or TSV file containing selected question IDs. For a "
            "table, --question-id-column is used; a one-column table is also accepted."
        ),
    )
    parser.add_argument(
        "--question-id-column",
        default="id",
        help="Question-ID column in both the expanded input and an optional ID table.",
    )
    parser.add_argument(
        "--allow-missing-question-ids",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ignore requested IDs that are absent from the input instead of failing.",
    )
    parser.add_argument(
        "--resume-from-output-dir",
        default=None,
        help=(
            "Existing failed/partial output directory. The script finds the "
            "longest contiguous input-ID prefix present in every guaranteed "
            "checkpoint file and processes only later IDs."
        ),
    )
    parser.add_argument(
        "--resume-after-id",
        default=None,
        help=(
            "Manual alternative to automatic detection: process only IDs after "
            "this ID in input order."
        ),
    )
    parser.add_argument(
        "--resume-reprocess-last",
        type=int,
        default=0,
        help=(
            "Conservatively reprocess this many IDs before the detected resume "
            "point. The resumed output remains in a separate directory."
        ),
    )
    parser.add_argument(
        "--resume-dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write/print the resume plan and exit before loading the models.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.prescore_pair_chunk_size < 1:
        raise ValueError("--prescore-pair-chunk-size must be positive.")

    requested_metrics = requested_metrics_from_args(args)
    if args.metrics == "transition_ordering":
        # The base v8 evaluator accepts explicit comma-separated names.
        args.metrics = ",".join(TRANSITION_ORDERING_METRICS)
    requested_direct = sorted(DIRECT_TRACE_PAIR_METRICS & set(requested_metrics))
    if requested_direct and not args.compute_direct_trace_pair_metrics:
        raise ValueError(
            "These requested metrics require --compute-direct-trace-pair-metrics: "
            f"{requested_direct}"
        )
    if args.save_trace_pair_details and not args.compute_direct_trace_pair_metrics:
        raise ValueError(
            "--save-trace-pair-details requires --compute-direct-trace-pair-metrics."
        )

    resume_metadata = prepare_resume_selection(args)
    if args.resume_dry_run:
        if not resume_metadata.get("resume_enabled"):
            raise ValueError("--resume-dry-run requires a resume option.")
        return
    if resume_metadata.get("resume_enabled") and resume_metadata.get("n_remaining_ids") == 0:
        return
    selection_metadata = prepare_selected_question_input(args)
    BASE.run_experiment(args, "sampled_allpairs")
    if args.self_test:
        return
    output_dir = Path(args.output_dir)
    optimization_metadata = {
        "base_script": "audit_answer_local_prefix_allpairs_v8.py",
        "optimized_script": Path(__file__).name,
        "prescore_pair_chunk_size": args.prescore_pair_chunk_size,
        "compute_direct_trace_pair_metrics": args.compute_direct_trace_pair_metrics,
        "save_transition_details": args.save_transition_details,
        "save_precedence_details": args.save_precedence_details,
        "save_community_details": args.save_community_details,
        "save_trace_schemas": args.save_trace_schemas,
        "save_stage_diagnostics": args.save_stage_diagnostics,
        "save_trace_pair_details": args.save_trace_pair_details,
        "question_selection": selection_metadata,
        "resume": resume_metadata,
        "optimizations": [
            "largest-prefix pair prescoring",
            "incremental prefix pair construction",
            "one community merge per prefix followed by support filtering",
            "community metrics computed once per community support",
            "shared transition/precedence sequence context",
            "precedence counts precomputed before support sweep",
            "direct trace-pair metrics optional",
            "detail outputs independently optional",
            "optional pre-run filtering to selected question IDs",
            "safe resume from the longest common contiguous saved-ID prefix",
        ],
    }
    (output_dir / "optimization_metadata.json").write_text(
        json.dumps(optimization_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
