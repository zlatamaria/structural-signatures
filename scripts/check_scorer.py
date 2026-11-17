"""Compare four fresh bidirectional predictions with the historical cache."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from structural_signatures.cli import model_options, reranker_arguments
from structural_signatures.scoring import FrozenScores, ModelScores
from structural_signatures.traces import prepare, v8

ROOT = Path(__file__).resolve().parents[1]


def check(cache, device, **options):
    example = ROOT / "data/example"
    frozen = FrozenScores(example / "q315_scores.csv.gz")
    chosen = sorted(frozen.rows)[:4]
    selected = set(chosen)
    traces = pd.read_csv(
        example / "q315_traces.csv.gz", dtype={"final_answer_new": str}
    )
    question = prepare("315", traces)
    state = v8.empty_prefix_state()
    v8.extend_prefix_state(
        state=state,
        question=question,
        ordered_trace_ids=list(range(len(question.answers_by_trace))),
        new_prefix_count=25,
    )
    lookup = {}
    for answer_pairs in state.pairs_by_answer.values():
        for u, v in answer_pairs:
            texts = (question.units[u].text, question.units[v].text)
            key = frozen.pair_key(*texts)
            if key in selected:
                lookup[key] = tuple(sorted(texts))
    pairs = [lookup[key] for key in chosen]
    ref = np.column_stack(frozen.directional_logits(pairs))
    model = ModelScores(cache, device, **{"batch_size": 4, **options})
    try:
        current = np.column_stack(model.directional_logits(pairs))
        again = np.column_stack(model.directional_logits(pairs))
        np.testing.assert_array_equal(current, again)
        return {
            "identity": model.identity,
            "keys": chosen,
            "archive": ref.tolist(),
            "fresh": current.tolist(),
            "max_abs_error": float(np.abs(ref - current).max()),
            "new_pairs": model.n_new_pairs,
            "cache_hits": model.n_cached_pairs,
        }
    finally:
        model.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    reranker_arguments(parser, batch_size=4)
    args = parser.parse_args()
    print(json.dumps(check(args.cache, args.device, **model_options(args)), indent=2))
