"""Check every archived candidate count and tie-break position against the expanded CSV."""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from structural_signatures.config import PRESETS, budgets
from structural_signatures.legacy.checkpoint_route_convergence_punctuation import (
    normalize_answer,
    parse_bool,
)
from structural_signatures.replay import ANSWER, filtered_chunks


def audit(traces, answers, sizes):
    source = pd.read_csv(
        traces,
        usecols=["id", "final_answer_new", "row_hit"],
        dtype={"id": str, "final_answer_new": str},
    )
    source["answer"] = source.final_answer_new.map(normalize_answer)
    source["hit"] = source.row_hit.map(parse_bool)
    valid = source.dropna(subset=["answer"])
    cfg = PRESETS["paper_w5"]
    parts = list(filtered_chunks(answers, ANSWER, cfg, sizes, chunksize=200_000))
    archive = pd.concat(parts, ignore_index=True)
    archive["is_correct_answer"] = (
        archive.is_correct_answer.astype(str)
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    if archive.is_correct_answer.isna().any() or set(archive.id) != set(source.id):
        raise ValueError("The answer labels or question population differ")
    observed = archive.groupby("id")
    total = short = 0
    for qid, group in valid.groupby("id", sort=False):
        sequence = group.answer.to_list()
        correct = group.groupby("answer").hit.any().to_dict()
        first = {}
        counts = Counter()
        rows = []
        for pos, answer in enumerate(sequence[: max(sizes)]):
            first.setdefault(answer, pos)
            counts[answer] += 1
            k = pos + 1
            if k in sizes:
                rows.extend((k, a, n, first[a], correct[a]) for a, n in counts.items())
        for k in sizes:
            if k > len(sequence):
                short += 1
                rows.extend((k, a, n, first[a], correct[a]) for a, n in counts.items())
        columns = [
            "sample_size",
            "answer",
            "raw_count",
            "first_trace_position",
            "is_correct_answer",
        ]
        raw = pd.DataFrame(rows, columns=columns)
        saved = observed.get_group(qid)[columns]

        def ordered(frame):
            return frame.sort_values(["sample_size", "answer"]).reset_index(drop=True)

        pd.testing.assert_frame_equal(
            ordered(raw), ordered(saved), check_dtype=False, obj=f"question {qid}"
        )
        total += len(raw)
    if total != len(archive):
        raise ValueError("Retained trace rows do not cover all archived questions")
    retained = valid.groupby("id").size()
    return {
        "raw_traces": len(source),
        "retained_traces": len(valid),
        "rejected_answers": len(source) - len(valid),
        "questions": len(retained),
        "budgets": len(sizes),
        "matched_candidate_rows": total,
        "matched_question_prefixes": len(retained) * len(sizes),
        "short_prefixes": short,
        "questions_short_at_last_budget": int((retained < max(sizes)).sum()),
        "minimum_retained_traces": int(retained.min()),
        "conflicting_answer_labels": int(
            (valid.groupby(["id", "answer"]).hit.nunique() > 1).sum()
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--answers", required=True, type=Path)
    parser.add_argument("--budgets", default="5-99")
    args = parser.parse_args()
    print(json.dumps(audit(args.traces, args.answers, budgets(args.budgets)), indent=2))
