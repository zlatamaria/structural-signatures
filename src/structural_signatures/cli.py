"""Entry points for trace construction and saved-table evaluation."""

import argparse
from dataclasses import asdict, replace
from pathlib import Path

from .config import EXCESS_METRIC, METRIC, PRESETS, Segmentation, budgets
from .io import run_directory, write_metadata
from .legacy.audit_answer_local_prefix_allpairs_v8 import (
    COHESION_MODES,
    DEFAULT_RERANKER_MODEL,
    SCORE_MODES,
)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def fraction(value):
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def reranker_arguments(parser, batch_size=64):
    group = parser.add_argument_group("fresh directional scoring")
    group.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANKER_MODEL,
        help=f"Hugging Face ID of a scalar CrossEncoder (default {DEFAULT_RERANKER_MODEL})",
    )
    group.add_argument(
        "--reranker-revision",
        help="Hub commit, tag or branch. The paper model defaults to its archived snapshot",
    )
    group.add_argument(
        "--reranker-max-length",
        type=positive_int,
        default=256,
        help="maximum input tokens per ordered pair (default 256)",
    )
    group.add_argument(
        "--reranker-batch-size",
        type=positive_int,
        default=batch_size,
        help=f"ordered pairs per model batch (default {batch_size})",
    )
    group.add_argument(
        "--device", default="cpu", help="CrossEncoder device (default cpu)"
    )


def model_options(args):
    return {
        "model_name": args.reranker_model,
        "revision": args.reranker_revision,
        "batch_size": args.reranker_batch_size,
        "max_length": args.reranker_max_length,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(prog="structural-signatures", description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    build = subs.add_parser(
        "build", help="build answer and precedence tables from expanded traces"
    )
    build.add_argument(
        "--traces",
        type=Path,
        required=True,
        help="one CSV row per sampled completion, in draw order",
    )
    scorer = build.add_mutually_exclusive_group(required=True)
    scorer.add_argument("--scores", type=Path, help="frozen directional-score CSV[.gz]")
    scorer.add_argument(
        "--legacy-cache", type=Path, help="existing v8 SQLite shard, opened read-only"
    )
    scorer.add_argument(
        "--model-cache",
        type=Path,
        help="directory for model-specific directional-score databases",
    )
    reranker_arguments(build)
    segment = build.add_argument_group("reasoning units")
    segment.add_argument(
        "--max-unit-chars",
        type=positive_int,
        default=900,
        help="split longer sentences at this length (default 900)",
    )
    segment.add_argument(
        "--max-units-per-trace",
        type=positive_int,
        default=96,
        help="evenly subsample longer traces (default 96 units)",
    )
    segment.add_argument(
        "--minimum-unit-position",
        type=fraction,
        default=0.0,
        help="earliest relative position to keep (default 0)",
    )
    replay = subs.add_parser(
        "replay", help="reconstruct graphs and select an answer at every prefix"
    )
    replay.add_argument("--answers", type=Path, required=True)
    replay.add_argument("--precedence", type=Path, required=True)
    replay.add_argument(
        "--expected-ids",
        type=positive_int,
        help="require this many questions at every budget",
    )
    replay.add_argument(
        "--window",
        type=positive_int,
        help="available consecutive prefixes in the trailing mean (preset default 5 or 3)",
    )
    replay.add_argument(
        "--minimum-answer-frequency",
        type=positive_int,
        help="minimum candidate votes (default 2)",
    )
    replay.add_argument(
        "--metric",
        choices=(METRIC, EXCESS_METRIC),
        help=f"candidate score before smoothing (default {METRIC})",
    )
    for sub in [build, replay]:
        sub.add_argument("--preset", choices=PRESETS, default="paper_w5")
        sub.add_argument(
            "--budgets",
            default="5-99",
            metavar="FIRST-LAST",
            help="inclusive consecutive prefixes, or a single size (default 5-99)",
        )
        sub.add_argument("--out", required=True, type=Path, help="new run directory")
        graph = sub.add_argument_group("checkpoint and precedence configuration")
        graph.add_argument(
            "--score-mode",
            choices=SCORE_MODES,
            help="combine the two directional scores (default raw_min)",
        )
        graph.add_argument(
            "--edge-quantile",
            type=fraction,
            help="within-answer similarity quantile (default 0.90)",
        )
        graph.add_argument(
            "--cohesion-mode",
            choices=COHESION_MODES,
            help="cross-community merge statistic (default min)",
        )
        graph.add_argument(
            "--community-min-support",
            type=positive_int,
            help="distinct traces per checkpoint (default 5)",
        )
        graph.add_argument(
            "--transition-min-support",
            type=positive_int,
            help="traces containing both checkpoints (default 3)",
        )
        graph.add_argument(
            "--consistency",
            type=fraction,
            help="minimum fraction in the dominant direction (default 0.80)",
        )
    args = parser.parse_args(argv)
    preset = PRESETS[args.preset]
    overrides = {
        key: getattr(args, key)
        for key in asdict(preset)
        if getattr(args, key, None) is not None
    }
    cfg = replace(preset, **overrides)
    if cfg.consistency < 0.5:
        parser.error("--consistency must be between 0.5 and 1")
    try:
        sizes = budgets(args.budgets)
    except ValueError as error:
        parser.error(str(error))
    if (
        args.command == "build"
        and not args.model_cache
        and (
            args.reranker_model != DEFAULT_RERANKER_MODEL
            or args.reranker_revision is not None
            or args.reranker_max_length != 256
        )
    ):
        parser.error(
            "The archived scores use the paper Qwen model and 256 tokens. "
            "Use --model-cache for a different model, revision or length."
        )
    with run_directory(args.out) as directory:
        fields = {
            "command": args.command,
            "preset": args.preset,
            "config": asdict(cfg),
            "requested_budgets": list(sizes),
        }
        if args.command == "build":
            from .scoring import PROMPT_HASH, FrozenScores, LegacyScores, ModelScores
            from .traces import build_tables

            files = {"traces": args.traces}
            if args.scores or args.legacy_cache:
                files["directional_scores"] = args.scores or args.legacy_cache
                model = (
                    FrozenScores(args.scores)
                    if args.scores
                    else LegacyScores(args.legacy_cache)
                )
                identity = {
                    "mode": "frozen_csv" if args.scores else "read_only_sqlite",
                    "model": model.model_name,
                    "prompt_sha256": PROMPT_HASH,
                    "max_length": model.max_length,
                }
            else:
                model = ModelScores(
                    args.model_cache, args.device, **model_options(args)
                )
                identity = dict(model.identity, batch_size=model.batch_size)
            segmentation = Segmentation(
                args.max_unit_chars,
                args.max_units_per_trace,
                args.minimum_unit_position,
            )
            try:
                n_ids = build_tables(
                    args.traces, directory, cfg, sizes, model, segmentation
                )
            finally:
                if isinstance(model, (LegacyScores, ModelScores)):
                    model.close()
            write_metadata(
                directory,
                files,
                n_ids=n_ids,
                scorer=identity,
                segmentation=asdict(segmentation),
                **fields,
            )
        else:
            from .replay import evaluate, load_scores

            scores = load_scores(
                args.answers, args.precedence, cfg, sizes, args.expected_ids
            )
            scored, predictions, accuracy = evaluate(scores, cfg)
            scored.sort_index().to_csv(directory / "answer_scores.csv.gz", index=False)
            predictions.to_csv(directory / "predictions.csv.gz", index=False)
            accuracy.to_csv(directory / "accuracy.csv", index=False)
            write_metadata(
                directory,
                {"answers": args.answers, "precedence": args.precedence},
                n_ids=int(scores.id.nunique()),
                answer_rows=len(scores),
                question_prefixes=len(predictions),
                **fields,
            )
    print(args.out)


if __name__ == "__main__":
    main()
