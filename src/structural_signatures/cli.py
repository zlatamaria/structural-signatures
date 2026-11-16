"""Entry points for trace construction and saved-table evaluation."""

import argparse
from dataclasses import asdict
from pathlib import Path

from .config import PRESETS, budgets
from .io import run_directory, write_metadata


def main(argv=None):
    parser = argparse.ArgumentParser(prog="structural-signatures", description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    build = subs.add_parser(
        "build", help="build answer and precedence tables from expanded traces"
    )
    build.add_argument("--traces", type=Path, required=True)
    scorer = build.add_mutually_exclusive_group(required=True)
    scorer.add_argument("--scores", type=Path, help="frozen directional-score CSV[.gz]")
    scorer.add_argument(
        "--legacy-cache", type=Path, help="existing v8 SQLite shard, opened read-only"
    )
    scorer.add_argument(
        "--model-cache",
        type=Path,
        help="directory for new Qwen directional-score databases",
    )
    build.add_argument(
        "--device", default="cpu", help="CrossEncoder device, e.g. cpu or cuda"
    )
    replay = subs.add_parser(
        "replay", help="reconstruct graphs and select an answer at every prefix"
    )
    replay.add_argument("--answers", type=Path, required=True)
    replay.add_argument("--precedence", type=Path, required=True)
    replay.add_argument(
        "--expected-ids", type=int, help="require this many questions at every budget"
    )
    for sub in [build, replay]:
        sub.add_argument("--preset", choices=PRESETS, default="paper_w5")
        sub.add_argument("--budgets", default="5-99", metavar="FIRST-LAST")
        sub.add_argument("--out", required=True, type=Path, help="new run directory")
    args = parser.parse_args(argv)
    cfg, sizes = PRESETS[args.preset], budgets(args.budgets)
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
                model = ModelScores(args.model_cache, args.device)
                identity = model.identity
            try:
                n_ids = build_tables(args.traces, directory, cfg, sizes, model)
            finally:
                if isinstance(model, (LegacyScores, ModelScores)):
                    model.close()
            write_metadata(directory, files, n_ids=n_ids, scorer=identity, **fields)
        else:
            from .replay import load_scores, select

            scores = load_scores(
                args.answers, args.precedence, cfg, sizes, args.expected_ids
            )
            predictions, accuracy = select(scores, cfg)
            scores.to_csv(directory / "answer_scores.csv.gz", index=False)
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
