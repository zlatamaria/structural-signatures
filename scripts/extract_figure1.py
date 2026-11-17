"""Extract six published panels from the repository replay tables."""

import json
from pathlib import Path

import pandas as pd

from structural_signatures.config import PRESETS
from structural_signatures.replay import ANSWER, EDGE, KEY, filtered_chunks

ROOT = Path(__file__).resolve().parents[1]


def extract(folder=ROOT / "data/figure1", source=ROOT / "data/math500"):
    panels = pd.DataFrame(json.loads((folder / "panels.json").read_text()))
    cases = panels[KEY]
    sizes = sorted(set(cases.sample_size))
    for name, fields in [("answers", ANSWER), ("precedence", EDGE)]:
        sections = []
        for chunk in filtered_chunks(
            source / f"{name}.csv.gz", fields, PRESETS["historical_w3"], sizes, 200000
        ):
            sections.append(
                chunk.merge(cases, on=KEY, how="inner", validate="many_to_one")
            )
        subset = pd.concat(sections, ignore_index=True)
        if name == "answers" and len(subset) != len(panels):
            raise ValueError("Missing Figure 1 candidates")
        subset.to_csv(folder / f"{name}.csv", index=False)


if __name__ == "__main__":
    extract()
