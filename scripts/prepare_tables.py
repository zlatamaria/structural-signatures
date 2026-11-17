"""Project a merged v8 run into the two portable paper-configuration tables."""

import argparse
import csv
import gzip
from io import TextIOWrapper
from pathlib import Path

from structural_signatures.config import PRESETS
from structural_signatures.io import run_directory, write_metadata
from structural_signatures.replay import ANSWER, EDGE, KEY, filtered_chunks


def project(source, destination, fields, cfg):
    columns = [*KEY, *cfg.filters, *fields]
    rows = 0
    with (
        destination.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz,
        TextIOWrapper(gz, encoding="utf-8", newline="") as output,
    ):
        csv.writer(output, lineterminator="\n").writerow(columns)
        for part in filtered_chunks(source, fields, cfg, range(5, 100), 160_000):
            part.to_csv(output, index=False, columns=columns, header=False)
            rows += len(part)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--precedence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = PRESETS["paper_w5"]
    with run_directory(args.out) as directory:
        n_answers = project(args.answers, directory / "answers.csv.gz", ANSWER, cfg)
        n_pairs = project(args.precedence, directory / "precedence.csv.gz", EDGE, cfg)
        write_metadata(
            directory,
            {"answers": args.answers, "precedence": args.precedence},
            config=cfg.filters,
            candidate_rows=n_answers,
            precedence_rows=n_pairs,
        )
