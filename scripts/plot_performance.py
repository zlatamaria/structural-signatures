"""Draw the test performance and transition curves from archived question-level choices."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from structural_signatures.transitions import decompose

ROOT = Path(__file__).resolve().parents[1]
COLORS = [
    "#16896a",
    "#75b39c",
    "#d68348",
    "#a84639",
    "#e1aa70",
    "#b65e81",
    "#6c3d85",
    "#4d7893",
]


def draw(predictions, out):
    _counts, signed = decompose(predictions)
    grouped = predictions.groupby("sample_size")
    curve = grouped[["prediction_correct", "majority_correct"]].mean() * 100
    net = grouped.net_repair.sum()
    np.testing.assert_array_equal(
        signed.sum(axis=1).iloc[1:].values, net.diff().iloc[1:].values
    )
    fig, axes = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(10, 7.3),
        gridspec_kw={"height_ratios": [1, 1.1, 1.35]},
    )
    fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.17, hspace=0.48)
    for ax, title in zip(
        axes,
        [
            "Net gain over majority",
            "Change from the previous prefix",
            "Matched accuracy",
        ],
    ):
        ax.spines[["right", "top"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#c8cbd0")
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e9e9ed", linewidth=0.8)
        ax.tick_params(color="#c1c3c9", labelcolor="#36363c", labelsize=9)
        ax.set_title(title, loc="left", fontsize=12, weight="bold", color="#252630")
        for a, b in [(45, 60), (85, 90)]:
            ax.axvspan(a, b, color="#b7a4c5", alpha=0.11, linewidth=0)
    axes[0].plot(net.index, net, color="#684699", linewidth=2)
    axes[0].axhline(0, color="#89868c", lw=0.8)
    axes[0].set_ylabel("Questions")
    axes[0].yaxis.set_major_locator(MaxNLocator(5, integer=True))
    axes[0].scatter([99], [net.loc[99]], s=30, color="#684699", zorder=4)
    pos, neg = np.zeros(len(signed)), np.zeros(len(signed))
    for (event, data), color in zip(signed.items(), COLORS):
        if not data.any():
            continue
        bases = np.where(data > 0, pos, neg)
        axes[1].bar(
            signed.index, data, bottom=bases, width=0.88, label=event, color=color
        )
        pos += np.maximum(data, 0)
        neg += np.minimum(data, 0)
    axes[1].axhline(0, color="#78767e", lw=0.8)
    axes[1].set_ylabel("Questions")
    axes[1].yaxis.set_major_locator(MaxNLocator(5, integer=True))
    axes[2].plot(
        curve.index,
        curve.prediction_correct,
        color="#167c65",
        lw=2.1,
        label="Structural selector",
    )
    axes[2].plot(
        curve.index,
        curve.majority_correct,
        color="#60616b",
        lw=1.5,
        ls="--",
        label="Majority vote",
    )
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].set_xlabel("Requested traces per question")
    axes[2].set_xlim(4, 100)
    axes[2].set_xticks([5, 15, 25, 35, 45, 55, 65, 75, 85, 99])
    axes[2].legend(frameon=False, loc="lower right", fontsize=9)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3, frameon=False, fontsize=8)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "data/reference/w5_predictions.csv.gz",
    )
    cli.add_argument("--out", type=Path, default=ROOT / "docs/assets/performance.png")
    args = cli.parse_args()
    draw(pd.read_csv(args.predictions, dtype={"id": str}), args.out)
