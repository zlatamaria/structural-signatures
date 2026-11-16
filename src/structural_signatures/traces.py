"""Build v8-compatible answer and precedence tables from expanded traces."""

import csv
import gzip
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd

from .legacy import audit_answer_local_prefix_allpairs_v8_optimized_resume as v8
from .legacy import checkpoint_route_convergence_punctuation as punctuation
from .replay import ANSWER, EDGE, KEY


def prepare(question_id, traces):
    return v8.BASE.prepare_question(
        question_id=question_id,
        frame=traces,
        pipeline=punctuation,
        text_col="completion",
        answer_col="final_answer_new",
        hit_col="row_hit",
        min_unit_chars=1,
        max_unit_chars=900,
        max_units_per_trace=96,
        unit_context="current",
        minimum_unit_position=0.0,
    )


def build_question(question, cfg, sizes, scorer):
    ordered = list(range(len(question.answers_by_trace)))
    store = v8.BASE.PairScoreStore(units=question.units, reranker=scorer)
    v8.prescore_largest_prefix(
        question=question,
        prefix_trace_ids=ordered[: max(sizes)],
        score_store=store,
        chunk_size=50_000,
        show_progress=False,
    )
    state = v8.empty_prefix_state()
    for k in sizes:
        retained = min(k, len(ordered))
        v8.extend_prefix_state(
            state=state,
            question=question,
            ordered_trace_ids=ordered,
            new_prefix_count=retained,
        )
        for answer in state.observed_answers:
            trace_ids = state.traces_by_answer[answer]
            pairs = state.pairs_by_answer[answer]
            proposals = store.scored_edges(pairs, cfg.score_mode)
            threshold = (
                float(np.quantile([p[2] for p in proposals], cfg.edge_quantile))
                if pairs
                else np.inf
            )
            key = {
                "id": question.question_id,
                "sample_size": k,
                "answer": answer,
                **cfg.filters,
            }
            candidate = dict(
                key,
                raw_count=len(trace_ids),
                retained_budget=retained,
                first_trace_position=trace_ids[0],
                is_correct_answer=question.is_correct_by_answer[answer],
                edge_threshold=threshold,
                n_local_communities=0,
            )
            details = []
            if len(trace_ids) >= cfg.community_min_support:
                units = [u for t in trace_ids for u in question.unit_ids_by_trace[t]]
                build = v8.BASE.build_constrained_communities(
                    question_id=question.question_id,
                    answer=answer,
                    units=question.units,
                    unit_ids=units,
                    proposal_edges=proposals,
                    score_store=store,
                    score_mode=cfg.score_mode,
                    edge_threshold=threshold,
                    cohesion_mode=cfg.cohesion_mode,
                    cohesion_margin=0,
                    min_trace_support=cfg.community_min_support,
                    max_rejections=0,
                )
                candidate["n_local_communities"] = len(build.communities)
                context = v8.prepare_structural_sequence_context(
                    question=question,
                    trace_ids=trace_ids,
                    communities=build.communities,
                    node_to_community=build.node_to_community,
                )
                _, rows = v8.precedence_metrics_from_context(
                    question=question,
                    answer=answer,
                    context=context,
                    precedence_min_support=cfg.transition_min_support,
                    precedence_min_consistency=cfg.consistency,
                    save_details=True,
                )
                details = [
                    dict(row, precedence_min_consistency=cfg.consistency, **key)
                    for row in rows
                ]
            yield candidate, details


def build_tables(path, directory, cfg, sizes, scorer):
    fields = ["id", "completion", "final_answer_new", "row_hit"]
    traces = pd.read_csv(
        path, usecols=fields, dtype={"id": str, "final_answer_new": str}
    )
    if traces.empty or traces.id.isna().any():
        raise ValueError(
            "Expanded traces must include a nonempty question ID on every row"
        )
    common = [*KEY, *cfg.filters]
    with ExitStack() as stack:

        def writer(filename, columns):
            stream = stack.enter_context(
                gzip.open(
                    Path(directory) / filename, "wt", newline="", encoding="utf-8"
                )
            )
            obj = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            obj.writeheader()
            return obj

        answers = writer(
            "answers.csv.gz",
            common
            + ANSWER
            + ["retained_budget", "edge_threshold", "n_local_communities"],
        )
        precedence = writer(
            "precedence.csv.gz",
            common
            + EDGE
            + [
                "forward_count",
                "reverse_count",
                "support_fraction",
                "direction_strength",
                "mean_forward_gap",
                "shortcut_survives",
            ],
        )
        positions = writer(
            "trace_order.csv.gz", ["id", "input_position", "retained_position"]
        )
        for qid, group in traces.groupby("id", sort=False):
            question = prepare(qid, group)
            retained = 0
            for pos, value in enumerate(group.final_answer_new):
                valid = punctuation.normalize_answer(value) is not None
                positions.writerow(
                    {
                        "id": qid,
                        "input_position": pos,
                        "retained_position": retained if valid else "",
                    }
                )
                retained += int(valid)
            for candidate, edges in build_question(question, cfg, sizes, scorer):
                answers.writerow(candidate)
                precedence.writerows(edges)
    return int(traces.id.nunique())
