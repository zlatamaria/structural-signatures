"""The shared graph construction and the two documented inference windows."""

from dataclasses import dataclass

METRIC = "largest_component_excess_edges_per_trace"
EXCESS_METRIC = "largest_component_excess_edges"


@dataclass(frozen=True)
class Config:
    window: int
    score_mode: str = "raw_min"
    edge_quantile: float = 0.90
    cohesion_mode: str = "min"
    community_min_support: int = 5
    transition_min_support: int = 3
    consistency: float = 0.80
    minimum_answer_frequency: int = 2
    metric: str = METRIC

    @property
    def filters(self):
        return {
            k: getattr(self, k)
            for k in (
                "score_mode",
                "edge_quantile",
                "cohesion_mode",
                "community_min_support",
                "transition_min_support",
            )
        }


PRESETS = {"paper_w5": Config(window=5), "historical_w3": Config(window=3)}


@dataclass(frozen=True)
class Segmentation:
    max_unit_chars: int = 900
    max_units_per_trace: int = 96
    minimum_unit_position: float = 0.0


def budgets(spec):
    """A consecutive interval preserves the meaning of a trailing prefix mean."""
    parts = spec.split("-")
    if len(parts) > 2 or any(not p.isdigit() for p in parts):
        raise ValueError("Use a single budget or a consecutive interval, e.g. 5-99")
    first, last = int(parts[0]), int(parts[-1])
    if first < 5 or last < first:
        raise ValueError("Budgets must increase and start at 5 or above")
    return range(first, last + 1)
