"""Read vector coordinates from the ACL edition of the paper (optional pdfplumber)."""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pdfplumber


def figure2(page):
    curves = [item for item in page.curves if len(item["pts"]) == 95]
    grid = sorted(
        {
            line["top"]
            for line in page.lines
            if line["width"] > 250 and line["stroking_color"] == 0.6901960784
        }
    )
    if len(grid) != 15 or len(curves) != 3:
        raise ValueError("Expected the vector artwork from the seven-page ACL edition")
    # Tick labels run from 10 to 0 on the upper axis and .625 to .525 below.
    top_step = (grid[4] - grid[0]) / 10
    accuracy_step = (grid[14] - grid[10]) / 0.10
    points = [np.asarray(curve["pts"]) for curve in curves]
    np.testing.assert_allclose(points[0][:, 0], points[1][:, 0])
    np.testing.assert_allclose(points[0][:, 0], points[2][:, 0])
    net = (grid[4] - points[0][:, 1]) / top_step
    correct = 500 * (0.625 + (grid[10] - points[1][:, 1]) / accuracy_step)
    majority = 500 * (0.625 + (grid[10] - points[2][:, 1]) / accuracy_step)
    result = pd.DataFrame(
        {"net_repairs": net, "correct": correct, "majority_correct": majority}
    )
    np.testing.assert_allclose(result, result.round(), atol=0.0001, rtol=0)
    result = result.round().astype(int)
    result.insert(0, "sample_size", range(5, 100))
    return result


def figure1_layout(page):
    # These bounds locate the six panels inside Figure 1 on printed page 195.
    left, panel_width = 70.9, 75.6
    top, height = 80, 48
    nodes = []
    for word in page.extract_words(x_tolerance=0.55, y_tolerance=0.5):
        if not (82 < word["top"] < 125 and re.fullmatch(r"C\d+", word["text"])):
            continue
        x = (word["x0"] + word["x1"]) / 2
        y = (word["top"] + word["bottom"]) / 2
        panel = int((x - left) // panel_width)
        nodes.append(
            {
                "panel": panel + 1,
                "node": int(word["text"][1:]),
                "x": (x - left) / panel_width - panel,
                "y": 1 - (y - top) / height,
            }
        )
    return pd.DataFrame(nodes).sort_values(["panel", "node"]).reset_index(drop=True)


def figure1_edges(page, layout):
    """Read arrow tails, tips and highlight colors from the printed artwork."""
    coordinates = np.column_stack(
        [70.9 + 75.6 * (layout.panel - 1 + layout.x), 80 + 48 * (1 - layout.y)]
    )

    def nearest(point):
        return layout.iloc[np.square(coordinates - point).sum(axis=1).argmin()]

    rows = []
    for i, curve in enumerate(page.curves):
        color = curve["stroking_color"]
        if (
            curve["top"] > 130
            or curve["fill"]
            or not isinstance(color, tuple)
            or len(curve["pts"]) != 2
        ):
            continue
        head = np.asarray(page.curves[i + 1]["pts"][:3])
        tip = head[1]
        direction = tip - (head[0] + head[2]) / 2
        # The arrow stops at a node's outline. Continue about one radius to its center.
        first = nearest(curve["pts"][0])
        last = nearest(tip + 3 * direction / np.linalg.norm(direction))
        if first.panel != last.panel:
            raise ValueError("Arrow crosses a panel boundary")
        rows.append(
            {
                "panel": int(first.panel),
                "from_community": int(first.node),
                "to_community": int(last.node),
                "highlighted": abs(color[0] - 0.48235) < 0.001,
            }
        )
    return pd.DataFrame(rows).sort_values(["panel", "from_community", "to_community"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument(
        "--out", type=Path, required=True, help="directory for the two CSVs"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with pdfplumber.open(args.pdf) as paper:
        figure2(paper.pages[3]).to_csv(args.out / "figure2_from_pdf.csv", index=False)
        layout = figure1_layout(paper.pages[1])
        layout.to_csv(args.out / "layout.csv", index=False, float_format="%.8f")
        figure1_edges(paper.pages[1], layout).to_csv(
            args.out / "printed_edges.csv", index=False
        )
