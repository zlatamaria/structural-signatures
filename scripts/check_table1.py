"""Recompute the displayed development means from archived correct counts."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def table1():
    folder = ROOT / "data/reference"
    printed = pd.read_csv(folder / "table1_printed.csv").set_index("row")
    curves = pd.read_csv(folder / "dev_curves.csv.gz")
    rows = []
    for name, expected in printed.iterrows():
        data = curves[curves.row == name].sort_values("sample_size")
        if data.sample_size.tolist() != list(range(5, 100)):
            raise ValueError(f"Missing development budgets for {name}")
        reference = curves[curves.row == data.reference.iloc[0]].sort_values(
            "sample_size"
        )
        if not np.array_equal(
            reference[["sample_size", "n_ids", "majority_correct"]].to_numpy(),
            data[["sample_size", "n_ids", "majority_correct"]].to_numpy(),
        ):
            raise ValueError(f"Mismatched baseline for {name}")
        record = {
            "row": name,
            "n": int(data.n_ids.iloc[0]),
            "window": int(data.window.iloc[0]),
            "matched_reference": data.reference.iloc[0],
        }
        gain = (
            100
            * (data.correct - data.majority_correct).to_numpy()
            / data.n_ids.to_numpy()
        )
        baseline = (
            100
            * (reference.correct - reference.majority_correct).to_numpy()
            / reference.n_ids.to_numpy()
        )
        for label, start in [("full", 5), ("late", 61)]:
            use = data.sample_size.to_numpy() >= start
            record[label + "_gain"] = gain[use].mean()
            record[label + "_delta"] = (gain[use] - baseline[use]).mean()
        for key in expected.index:
            if abs(record[key] - expected[key]) > 0.000501:
                raise ValueError(
                    f"Printed and archived values differ for {name}, {key}"
                )
        rows.append(record)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(table1().round(3).to_string(index=False))
