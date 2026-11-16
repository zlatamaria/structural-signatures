#!/usr/bin/env python3
# VERSION: answer-local-prefix-community-v8-2026-07-30
"""
Prefix-budget study for answer-local, constrained checkpoint communities.

For every requested prefix size k (for example 2 through 99), this executable:

    takes exactly the first k traces of each question
      -> calculates majority vote on those exact same k traces
      -> groups the observed prefix traces by their observed answer
      -> reranks every cross-trace step pair within each observed answer group
      -> builds answer-local constrained checkpoint communities
      -> calculates checkpoint, transition, ordering, shortcut and length metrics
      -> selects among answers observed in the prefix
      -> compares the structural selector with the matched prefix majority

Thus ``sample_size`` means traces per QUESTION, not traces per answer. If the
first two traces have answers A and B, each answer has one observed trace;
shared communities cannot yet be formed and the structural method falls back to
the majority/tie rule on those same two traces. As k grows, an answer becomes
structurally evaluable only after at least two of its traces have appeared.

By default the prefix follows the input row order within each question. The
optional ``--prefix-order shuffled`` creates one deterministic nested random
order per question and seed; prefixes remain nested as k increases.

Four community rules
--------------------
1. Answer-local scope.
2. At most one reasoning step from each trace per community.
3. Minimum distinct-trace support.
4. Complete-linkage-like cohesion: a bridge edge alone cannot merge clusters.

Important outputs
-----------------
answer_metrics.csv
    One row per observed answer and experiment configuration. ``raw_count`` is
    the answer count in the current k-trace prefix; ``full_raw_count`` is its
    count among all available traces.
metric_predictions.csv / metric_accuracy.csv
    Structural predictions and gain/loss against the exact k-trace prefix
    majority and the full-trace majority.
sample_size_majority_baselines.csv / sample_size_baseline_accuracy.csv
    Majority-vote learning curves on the exact same nested prefixes.
sampling_manifest.csv
    Exact prefix trace IDs and per-answer observed trace IDs for every k.
reasoning_units.csv.gz / reranker_pair_scores.csv.gz
    Optional low-level recalculation artifacts.
community_details.csv / transition_details.csv / precedence_details.csv
    Auditable community and ordering structures.
trace_checkpoint_schemas.csv
    Retained checkpoint skeleton for every trace present in the prefix.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import sqlite3
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_: Any):
        return iterable


DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_INSTRUCTION = """
Judge whether the Document expresses the same mathematical reasoning checkpoint
as the Query. Treat paraphrases and algebraically equivalent formulations as
matches. The two steps must express the same intermediate claim, mathematical
transformation, or result. Reject pairs that merely discuss the same problem
or share vocabulary. Differences in numerical value, sign, operator,
inequality direction, condition, assumptions, or conclusion mean that the
checkpoints are not equivalent.
""".strip()

SCORE_MODES = ("raw_mean", "raw_min", "sigmoid_mean", "sigmoid_min")
COHESION_MODES = ("min", "q10")


SQLITE_HEADER = b"SQLite format 3\x00"

def _sqlite_sidecars(path: Path) -> list[Path]:
    return [Path(str(path) + "-wal"), Path(str(path) + "-shm"), Path(str(path) + "-journal")]

def quarantine_invalid_sqlite(path: str | Path, *, label: str) -> Path:
    """Validate an SQLite file; move invalid/corrupt files aside and start fresh."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return path
    invalid_reason = None
    try:
        if path.is_dir():
            invalid_reason = "path is a directory"
        elif path.stat().st_size == 0:
            invalid_reason = "empty file"
        else:
            with path.open("rb") as handle:
                header = handle.read(16)
            if header != SQLITE_HEADER:
                invalid_reason = f"invalid SQLite header {header!r}"
            else:
                connection = sqlite3.connect(path, timeout=10.0)
                try:
                    # Lightweight validation: confirm that SQLite can open
                    # the database and read its schema.
                    connection.execute(
                        "SELECT name FROM sqlite_master LIMIT 1"
                    ).fetchone()
                finally:
                    connection.close()
    except sqlite3.DatabaseError as error:
        invalid_reason = f"SQLite validation failed: {error}"

    if invalid_reason is None:
        return path

    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.invalid-{stamp}")
    path.replace(backup)
    for sidecar in _sqlite_sidecars(path):
        if sidecar.exists():
            sidecar.unlink()
    print(
        f"[cache repair] {label} was not a valid SQLite database: {invalid_reason}\n"
        f"[cache repair] moved it to: {backup}\n"
        f"[cache repair] a fresh database will be created at: {path}",
        flush=True,
    )
    return path

def retry_sqlite(operation, *, attempts: int = 10, initial_sleep: float = 0.25):
    """Retry transient SQLite lock/busy errors with exponential backoff."""
    sleep_seconds = initial_sleep
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except sqlite3.OperationalError as error:
            message = str(error).lower()
            if "locked" not in message and "busy" not in message:
                raise
            if attempt >= attempts:
                raise
            time.sleep(sleep_seconds)
            sleep_seconds = min(sleep_seconds * 2.0, 5.0)


# All listed metrics are maximized unless included here.
MINIMIZE_METRICS = {
    "mean_pairwise_transition_jaccard_similarity",  # diversity-oriented use
    "precedence_reversal_rate",
    "pair_fragmentation_ratio",
    "mean_checkpoint_words",
    "median_checkpoint_words",
}

DEFAULT_METRICS = (
    # Community amount, support, coverage and cohesion.
    "n_local_communities",
    "local_communities_per_trace",
    "community_unit_coverage",
    "community_trace_coverage",
    "mean_community_support_fraction",
    "mean_community_min_score",
    "mean_community_q10_score",
    "cohesion_weighted_support",
    "effective_local_communities_per_trace",
    # Simple recurrence, fragmentation and step-length summaries.
    "mean_checkpoint_schema_length",
    "deep_recurrent_checkpoint_mass",
    "pair_fragmentation_ratio",
    "mean_checkpoint_words",
    "median_checkpoint_words",
    "atomic_recurrent_checkpoint_mass",
    # Recurrent transition richness and diversity.
    "n_shared_transition_types",
    "shared_transition_types_per_trace",
    "transition_trace_coverage",
    "transition_support_gini_simpson",
    "effective_shared_transitions_per_trace",
    "transition_set_diversity_gini_simpson",
    "transition_sequence_diversity_gini_simpson",
    "mean_pairwise_transition_jaccard_distance",
    # Diversity receives credit only when both traces contain recurrent
    # transition evidence. This fixes empty-vs-nonempty pairs being treated
    # as maximally diverse.
    "supported_pairwise_transition_jaccard_distance",
    # Whole-route diversity discounted by the fraction of sampled traces that
    # actually contain qualifying transitions.
    "coverage_weighted_transition_set_diversity",
    "fraction_branching_checkpoints",
    # Branching checkpoint mass weighted by answer-local trace support.
    "support_weighted_branching_fraction",
    "max_transition_out_degree",
    "late_shared_transition_fraction",
    "cohesion_weighted_transition_support",
    # Order-aware checkpoint backbone and shortcut survival.
    "n_supported_precedence_pairs",
    "n_stable_precedence_pairs",
    "stable_precedence_pairs_per_trace",
    "supported_precedence_mass",
    "precedence_reversal_rate",
    "n_shortcut_surviving_pairs",
    "shortcut_survival_mass",
    "consensus_chain_length",
    "consensus_chain_fraction",
    "atomic_ordered_backbone_mass",
    # Direct trace-pair evidence; these do not require communities.
    "mean_best_trace_pair_score",
    "mean_top2_trace_pair_score",
    "mean_greedy_one_to_one_score",
    "mean_order_consistency",
    "trace_pair_match_coverage",
    "trace_pair_two_match_coverage",
)


FOCUSED_METRICS = (
    # Existing baselines.
    "n_local_communities",
    "n_shared_transition_types",
    "supported_pairwise_transition_jaccard_distance",
    # Recurrence, fragmentation and length.
    "mean_checkpoint_schema_length",
    "deep_recurrent_checkpoint_mass",
    "pair_fragmentation_ratio",
    "mean_checkpoint_words",
    "atomic_recurrent_checkpoint_mass",
    # Ordering and shortcut survival.
    "supported_precedence_mass",
    "precedence_reversal_rate",
    "shortcut_survival_mass",
    "consensus_chain_length",
    "atomic_ordered_backbone_mass",
)


# Curated train/test suite.  The metrics are intentionally conceptually distinct:
# checkpoint richness, recurrence depth, exact transition richness, old and
# supported Jaccard diversity, route-signature diversity, branching, stable
# ordering, shortcut survival, consensus-chain structure and checkpoint length.
# This avoids giving duplicated metrics extra weight while preserving all
# promising signals from the previous experiments.
SELECTED_METRICS = (
    "n_local_communities",
    "deep_recurrent_checkpoint_mass",
    "n_shared_transition_types",
    "mean_pairwise_transition_jaccard_distance",
    "supported_pairwise_transition_jaccard_distance",
    "coverage_weighted_transition_set_diversity",
    "max_transition_out_degree",
    "precedence_reversal_rate",
    "supported_precedence_mass",
    "shortcut_survival_mass",
    "consensus_chain_length",
    "mean_checkpoint_words",
    "atomic_ordered_backbone_mass",
)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

_WORD_PATTERN = re.compile(r"\S+")

def checkpoint_word_count(text: str) -> int:
    """Simple, transparent whitespace-token length for one reasoning step."""
    return len(_WORD_PATTERN.findall(str(text)))

def parse_int_list(value: str) -> list[int]:
    """Parse comma-separated integers and inclusive ranges such as ``2-99``."""
    result: set[int] = set()
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        range_match = re.fullmatch(r"(-?\d+)\s*-\s*(-?\d+)", item)
        if range_match:
            start = int(range_match.group(1))
            stop = int(range_match.group(2))
            step = 1 if stop >= start else -1
            result.update(range(start, stop + step, step))
        else:
            result.add(int(item))
    values = sorted(result)
    if not values:
        raise ValueError("Expected at least one integer or inclusive range.")
    return values


def parse_float_list(value: str) -> list[float]:
    result = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not result:
        raise ValueError("Expected at least one float.")
    return result


def parse_str_list(value: str) -> list[str]:
    result = list(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    if not result:
        raise ValueError("Expected at least one value.")
    return result


def stable_sigmoid(values: np.ndarray | float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    result = np.empty_like(array)
    positive = array >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exp_values = np.exp(array[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result.astype(np.float32)


def score_from_directional(
    forward: np.ndarray | float,
    reverse: np.ndarray | float,
    mode: str,
) -> np.ndarray:
    forward_array = np.asarray(forward, dtype=np.float32)
    reverse_array = np.asarray(reverse, dtype=np.float32)
    if mode == "raw_mean":
        return 0.5 * (forward_array + reverse_array)
    if mode == "raw_min":
        return np.minimum(forward_array, reverse_array)
    forward_sigmoid = stable_sigmoid(forward_array)
    reverse_sigmoid = stable_sigmoid(reverse_array)
    if mode == "sigmoid_mean":
        return 0.5 * (forward_sigmoid + reverse_sigmoid)
    if mode == "sigmoid_min":
        return np.minimum(forward_sigmoid, reverse_sigmoid)
    raise ValueError(f"Unknown score mode: {mode}")


def entropy_concentration(weights: Sequence[float]) -> dict[str, float]:
    values = np.asarray(list(weights), dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return {
            "entropy": 0.0,
            "effective_number": 0.0,
            "hhi": 0.0,
            "gini_simpson": 0.0,
        }
    probabilities = values / values.sum()
    entropy = float(-(probabilities * np.log(probabilities + 1e-12)).sum())
    hhi = float(np.square(probabilities).sum())
    return {
        "entropy": entropy,
        "effective_number": float(np.exp(entropy)),
        "hhi": hhi,
        "gini_simpson": float(1.0 - hhi),
    }


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_write_csv_gzip(frame: pd.DataFrame, path: Path) -> None:
    """Atomically write a gzip-compressed CSV."""
    temporary = Path(str(path) + ".tmp")
    frame.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(path)


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def resolve_hf_snapshot(
    *, repo_id: str, cache_dir: str | Path, local_files_only: bool
) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover
        raise ImportError("pip install -U huggingface-hub") from error
    cache_root = Path(cache_dir).expanduser()
    if local_files_only and not cache_root.exists():
        raise FileNotFoundError(cache_root)
    return snapshot_download(
        repo_id=repo_id,
        cache_dir=str(cache_root),
        local_files_only=local_files_only,
    )


def load_pipeline(path: Path):
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    module_dir = str(path.parent)
    added = module_dir not in sys.path
    if added:
        sys.path.insert(0, module_dir)
    try:
        importlib.invalidate_caches()
        existing = sys.modules.get(path.stem)
        if existing is not None and Path(getattr(existing, "__file__", "")).resolve() != path:
            del sys.modules[path.stem]
        return importlib.import_module(path.stem)
    finally:
        if added:
            try:
                sys.path.remove(module_dir)
            except ValueError:
                pass


# ---------------------------------------------------------------------------
# Persistent reranker cache
# ---------------------------------------------------------------------------

class CachedDirectionalReranker:
    """Cache raw logits in canonical text order; derive score modes later."""

    def __init__(
        self,
        *,
        model_name: str,
        hf_cache_dir: str | Path,
        cache_path: str | Path,
        prompt: str,
        batch_size: int,
        max_length: int,
        device: str | None,
        local_files_only: bool,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:  # pragma: no cover
            raise ImportError(
                "pip install -U 'sentence-transformers>=5.2' "
                "'transformers>=4.51' torch"
            ) from error

        snapshot = resolve_hf_snapshot(
            repo_id=model_name,
            cache_dir=hf_cache_dir,
            local_files_only=local_files_only,
        )
        self.model_name = model_name
        self.snapshot = snapshot
        self.prompt = prompt.strip()
        self.prompt_hash = hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.model = CrossEncoder(
            snapshot,
            device=device,
            local_files_only=True,
            max_length=self.max_length,
            prompts={"checkpoint_equivalence": self.prompt},
            default_prompt_name="checkpoint_equivalence",
        )

        cache_path = quarantine_invalid_sqlite(cache_path, label="reranker cache")
        self.cache_path = cache_path
        self.connection = sqlite3.connect(cache_path, timeout=120.0)
        self.connection.execute("PRAGMA busy_timeout=120000")
        # DELETE is more reliable than WAL on NFS/network filesystems.
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS directional_logits (
                pair_key TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                max_length INTEGER NOT NULL,
                canonical_left TEXT NOT NULL,
                canonical_right TEXT NOT NULL,
                canonical_left_to_right REAL NOT NULL,
                canonical_right_to_left REAL NOT NULL
            )
            """
        )
        self.connection.commit()
        self.n_new_pairs = 0
        self.n_cached_pairs = 0

    @staticmethod
    def canonical_pair(left: str, right: str) -> tuple[str, str, bool]:
        left = str(left)
        right = str(right)
        if left <= right:
            return left, right, False
        return right, left, True

    def pair_key(self, left: str, right: str) -> str:
        canonical_left, canonical_right, _ = self.canonical_pair(left, right)
        payload = "\x1f".join(
            [
                self.model_name,
                self.prompt_hash,
                str(self.max_length),
                canonical_left,
                canonical_right,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _read(self, keys: Sequence[str]) -> dict[str, tuple[float, float]]:
        found: dict[str, tuple[float, float]] = {}
        unique_keys = list(dict.fromkeys(keys))
        for start in range(0, len(unique_keys), 700):
            chunk = unique_keys[start : start + 700]
            placeholders = ",".join("?" for _ in chunk)
            query = (
                "SELECT pair_key, canonical_left_to_right, "
                "canonical_right_to_left FROM directional_logits "
                f"WHERE pair_key IN ({placeholders})"
            )
            rows = retry_sqlite(
                lambda: list(self.connection.execute(query, chunk)),
                attempts=10,
            )
            for key, forward, reverse in rows:
                found[str(key)] = (float(forward), float(reverse))
        return found

    def directional_logits(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        show_progress_bar: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not pairs:
            empty = np.empty(0, dtype=np.float32)
            return empty, empty

        canonical: list[tuple[str, str, bool]] = [
            self.canonical_pair(left, right) for left, right in pairs
        ]
        keys = [self.pair_key(left, right) for left, right in pairs]
        cached = self._read(keys)

        missing: dict[str, tuple[str, str]] = {}
        for key, (left, right, _) in zip(keys, canonical):
            if key not in cached:
                missing.setdefault(key, (left, right))

        if missing:
            missing_keys = list(missing)
            canonical_pairs = [missing[key] for key in missing_keys]
            reverse_pairs = [(right, left) for left, right in canonical_pairs]
            forward = np.asarray(
                self.model.predict(
                    canonical_pairs,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=show_progress_bar,
                ),
                dtype=np.float32,
            ).reshape(-1)
            reverse = np.asarray(
                self.model.predict(
                    reverse_pairs,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=show_progress_bar,
                ),
                dtype=np.float32,
            ).reshape(-1)
            rows = []
            for key, (left, right), value_forward, value_reverse in zip(
                missing_keys, canonical_pairs, forward, reverse
            ):
                cached[key] = (float(value_forward), float(value_reverse))
                rows.append(
                    (
                        key,
                        self.model_name,
                        self.prompt_hash,
                        self.max_length,
                        left,
                        right,
                        float(value_forward),
                        float(value_reverse),
                    )
                )
            def _write_rows() -> None:
                self.connection.executemany(
                    """
                    INSERT OR REPLACE INTO directional_logits (
                        pair_key, model_name, prompt_hash, max_length,
                        canonical_left, canonical_right,
                        canonical_left_to_right, canonical_right_to_left
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                self.connection.commit()
            retry_sqlite(_write_rows, attempts=10)
            self.n_new_pairs += len(rows)

        requested_forward: list[float] = []
        requested_reverse: list[float] = []
        for key, (_, _, swapped) in zip(keys, canonical):
            canonical_forward, canonical_reverse = cached[key]
            if swapped:
                requested_forward.append(canonical_reverse)
                requested_reverse.append(canonical_forward)
            else:
                requested_forward.append(canonical_forward)
                requested_reverse.append(canonical_reverse)
        self.n_cached_pairs += len(keys) - len(missing)
        return (
            np.asarray(requested_forward, dtype=np.float32),
            np.asarray(requested_reverse, dtype=np.float32),
        )

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


# ---------------------------------------------------------------------------
# Question representation and candidates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Unit:
    unit_id: int
    trace_id: int
    answer: str
    local_index: int
    position: float
    text: str
    embedding_text: str


@dataclass
class QuestionData:
    question_id: Any
    answers_by_trace: list[str]
    hits_by_trace: list[bool]
    units: list[Unit]
    unit_ids_by_trace: dict[int, list[int]]
    traces_by_answer: dict[str, list[int]]
    raw_count_by_answer: dict[str, int]
    first_trace_by_answer: dict[str, int]
    is_correct_by_answer: dict[str, bool]


def prepare_question(
    *,
    question_id: Any,
    frame: pd.DataFrame,
    pipeline: Any,
    text_col: str,
    answer_col: str,
    hit_col: str,
    min_unit_chars: int,
    max_unit_chars: int,
    max_units_per_trace: int,
    unit_context: str,
    minimum_unit_position: float,
) -> QuestionData:
    frame = frame.copy().reset_index(drop=True)
    answers: list[str] = []
    hits: list[bool] = []
    retained_rows: list[int] = []
    for index, row in frame.iterrows():
        answer = pipeline.normalize_answer(row[answer_col])
        if answer is None:
            continue
        answers.append(str(answer))
        hits.append(bool(pipeline.parse_bool(row[hit_col])))
        retained_rows.append(index)
    frame = frame.iloc[retained_rows].reset_index(drop=True)
    if not answers:
        raise ValueError(f"No normalized answers for id={question_id}")

    units: list[Unit] = []
    unit_ids_by_trace: dict[int, list[int]] = defaultdict(list)
    for trace_id, text in enumerate(frame[text_col].astype(str)):
        extracted = pipeline.extract_reasoning_units(
            text,
            min_unit_chars,
            max_unit_chars,
            max_units_per_trace,
        )
        if not extracted:
            extracted = [text.strip() or "[empty]"]
        denominator = max(len(extracted) - 1, 1)
        for local_index, unit_text in enumerate(extracted):
            position = local_index / denominator if len(extracted) > 1 else 0.5
            if position + 1e-12 < minimum_unit_position:
                continue
            unit_id = len(units)
            embedding_text = pipeline.make_embedding_unit(
                extracted, local_index, unit_context
            )
            units.append(
                Unit(
                    unit_id=unit_id,
                    trace_id=trace_id,
                    answer=answers[trace_id],
                    local_index=local_index,
                    position=float(position),
                    text=str(unit_text),
                    embedding_text=str(embedding_text),
                )
            )
            unit_ids_by_trace[trace_id].append(unit_id)

    traces_by_answer: dict[str, list[int]] = defaultdict(list)
    for trace_id, answer in enumerate(answers):
        traces_by_answer[answer].append(trace_id)
    raw_count = {answer: len(trace_ids) for answer, trace_ids in traces_by_answer.items()}
    first_trace = {answer: min(trace_ids) for answer, trace_ids in traces_by_answer.items()}
    is_correct = {
        answer: any(hits[trace_id] for trace_id in trace_ids)
        for answer, trace_ids in traces_by_answer.items()
    }
    return QuestionData(
        question_id=question_id,
        answers_by_trace=answers,
        hits_by_trace=hits,
        units=units,
        unit_ids_by_trace=dict(unit_ids_by_trace),
        traces_by_answer=dict(traces_by_answer),
        raw_count_by_answer=raw_count,
        first_trace_by_answer=first_trace,
        is_correct_by_answer=is_correct,
    )


def build_embedding_knn_candidates(
    *,
    embeddings: np.ndarray,
    units: Sequence[Unit],
    k: int,
    edge_mode: str,
) -> list[tuple[int, int, float]]:
    """Global cross-trace embedding kNN candidates for a whole question."""
    n_units = len(units)
    if n_units < 2 or k < 1:
        return []
    similarities = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    trace_ids = np.asarray([unit.trace_id for unit in units], dtype=int)
    neighbor_sets: list[set[int]] = []
    for left in range(n_units):
        candidates = np.flatnonzero(trace_ids != trace_ids[left])
        if candidates.size == 0:
            neighbor_sets.append(set())
            continue
        values = similarities[left, candidates]
        take = min(k, len(candidates))
        if take == len(candidates):
            selected = candidates
        else:
            local = np.argpartition(values, -take)[-take:]
            selected = candidates[local]
        neighbor_sets.append({int(value) for value in selected})

    edges: dict[tuple[int, int], float] = {}
    for left in range(n_units):
        for right in neighbor_sets[left]:
            low, high = sorted((left, right))
            if low == high:
                continue
            mutual = left in neighbor_sets[right]
            keep = mutual if edge_mode == "mutual" else True
            if keep:
                edges[(low, high)] = float(similarities[low, high])
    if edge_mode == "union":
        for right in range(n_units):
            for left in neighbor_sets[right]:
                low, high = sorted((left, right))
                if low != high:
                    edges[(low, high)] = float(similarities[low, high])
    return [(left, right, score) for (left, right), score in edges.items()]


def build_all_pairs_within_answer(
    *, question: QuestionData, trace_ids: Sequence[int]
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    trace_ids = list(trace_ids)
    for left_index, left_trace in enumerate(trace_ids):
        left_units = question.unit_ids_by_trace.get(left_trace, [])
        for right_trace in trace_ids[left_index + 1 :]:
            right_units = question.unit_ids_by_trace.get(right_trace, [])
            pairs.extend((left, right) for left in left_units for right in right_units)
    return pairs


def top_k_edges(
    *,
    edges: Sequence[tuple[int, int, float]],
    unit_ids: Sequence[int],
    k: int,
    edge_mode: str,
) -> list[tuple[int, int, float]]:
    if k <= 0:
        return list(edges)
    allowed = set(int(value) for value in unit_ids)
    adjacency: dict[int, list[tuple[int, float]]] = defaultdict(list)
    score_map: dict[tuple[int, int], float] = {}
    for left, right, score in edges:
        if left not in allowed or right not in allowed:
            continue
        adjacency[left].append((right, float(score)))
        adjacency[right].append((left, float(score)))
        score_map[(min(left, right), max(left, right))] = float(score)
    neighbors: dict[int, set[int]] = {}
    for unit_id in allowed:
        ordered = sorted(adjacency.get(unit_id, []), key=lambda item: (-item[1], item[0]))
        neighbors[unit_id] = {candidate for candidate, _ in ordered[:k]}
    selected: list[tuple[int, int, float]] = []
    for (left, right), score in score_map.items():
        left_has = right in neighbors[left]
        right_has = left in neighbors[right]
        keep = left_has and right_has if edge_mode == "mutual" else left_has or right_has
        if keep:
            selected.append((left, right, score))
    return selected


# ---------------------------------------------------------------------------
# Raw-pair score store and closure scoring
# ---------------------------------------------------------------------------

class PairScoreStore:
    """Question-local raw directional score store backed by the global cache."""

    def __init__(
        self,
        *,
        units: Sequence[Unit],
        reranker: CachedDirectionalReranker,
    ) -> None:
        self.units = units
        self.reranker = reranker
        self.raw: dict[tuple[int, int], tuple[float, float]] = {}

    @staticmethod
    def key(left: int, right: int) -> tuple[int, int]:
        return (left, right) if left < right else (right, left)

    def ensure(self, pairs: Sequence[tuple[int, int]], show_progress_bar: bool = False) -> None:
        missing = []
        seen = set()
        for left, right in pairs:
            if left == right:
                continue
            key = self.key(left, right)
            if key not in self.raw and key not in seen:
                missing.append(key)
                seen.add(key)
        if not missing:
            return
        text_pairs = [
            (self.units[left].text, self.units[right].text) for left, right in missing
        ]
        forward, reverse = self.reranker.directional_logits(
            text_pairs, show_progress_bar=show_progress_bar
        )
        for (left, right), value_forward, value_reverse in zip(missing, forward, reverse):
            # left < right and text-pair order follows unit order.
            self.raw[(left, right)] = (float(value_forward), float(value_reverse))

    def score(self, left: int, right: int, mode: str) -> float:
        if left == right:
            return float("inf")
        key = self.key(left, right)
        if key not in self.raw:
            raise KeyError(key)
        forward, reverse = self.raw[key]
        if left > right:
            forward, reverse = reverse, forward
        return float(score_from_directional(forward, reverse, mode))

    def scored_edges(
        self, pairs: Sequence[tuple[int, int]], mode: str
    ) -> list[tuple[int, int, float]]:
        return [(left, right, self.score(left, right, mode)) for left, right in pairs]


# ---------------------------------------------------------------------------
# Constrained answer-local communities
# ---------------------------------------------------------------------------

@dataclass
class LocalCommunity:
    community_id: int
    nodes: list[int]
    trace_ids: list[int]
    support: int
    min_score: float
    q10_score: float
    mean_score: float
    representative_text: str
    mean_position: float


@dataclass
class CommunityBuildResult:
    communities: list[LocalCommunity]
    node_to_community: dict[int, int]
    diagnostics: dict[str, Any]
    rejection_rows: list[dict[str, Any]]


def cross_cluster_pairs(left_nodes: Iterable[int], right_nodes: Iterable[int]) -> list[tuple[int, int]]:
    return [(left, right) for left in left_nodes for right in right_nodes if left != right]


def cohesion_summary(
    *,
    left_nodes: Sequence[int],
    right_nodes: Sequence[int],
    score_store: PairScoreStore,
    score_mode: str,
) -> tuple[float, float, float, list[float]]:
    pairs = cross_cluster_pairs(left_nodes, right_nodes)
    score_store.ensure(pairs)
    values = [score_store.score(left, right, score_mode) for left, right in pairs]
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return float("inf"), float("inf"), float("inf"), values
    return (
        float(array.min()),
        float(np.quantile(array, 0.10)),
        float(array.mean()),
        values,
    )


def internal_community_scores(
    nodes: Sequence[int], score_store: PairScoreStore, score_mode: str
) -> tuple[float, float, float]:
    pairs = [
        (nodes[left], nodes[right])
        for left in range(len(nodes))
        for right in range(left + 1, len(nodes))
    ]
    if not pairs:
        return float("nan"), float("nan"), float("nan")
    score_store.ensure(pairs)
    values = np.asarray(
        [score_store.score(left, right, score_mode) for left, right in pairs],
        dtype=float,
    )
    return float(values.min()), float(np.quantile(values, 0.10)), float(values.mean())


def build_constrained_communities(
    *,
    question_id: Any,
    answer: str,
    units: Sequence[Unit],
    unit_ids: Sequence[int],
    proposal_edges: Sequence[tuple[int, int, float]],
    score_store: PairScoreStore,
    score_mode: str,
    edge_threshold: float,
    cohesion_mode: str,
    cohesion_margin: float,
    min_trace_support: int,
    max_rejections: int,
) -> CommunityBuildResult:
    """Greedy complete-linkage-style merging under four explicit rules."""
    unit_ids = sorted(set(int(value) for value in unit_ids))
    parent = {unit_id: unit_id for unit_id in unit_ids}
    members = {unit_id: {unit_id} for unit_id in unit_ids}
    traces = {unit_id: {units[unit_id].trace_id} for unit_id in unit_ids}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def merge_roots(left_root: int, right_root: int) -> int:
        if len(members[left_root]) < len(members[right_root]):
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        members[left_root].update(members.pop(right_root))
        traces[left_root].update(traces.pop(right_root))
        return left_root

    diagnostics = Counter()
    rejection_rows: list[dict[str, Any]] = []
    sorted_edges = sorted(
        [edge for edge in proposal_edges if edge[2] >= edge_threshold],
        key=lambda edge: (-edge[2], edge[0], edge[1]),
    )
    diagnostics["n_proposal_edges"] = len(proposal_edges)
    diagnostics["n_edges_at_threshold"] = len(sorted_edges)

    for left, right, edge_score in sorted_edges:
        diagnostics["n_merge_attempts"] += 1
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            diagnostics["reject_already_same_cluster"] += 1
            continue

        # Rule 2: at most one node per trace.
        overlap = traces[left_root] & traces[right_root]
        if overlap:
            diagnostics["reject_trace_overlap"] += 1
            if len(rejection_rows) < max_rejections:
                rejection_rows.append(
                    {
                        "id": question_id,
                        "answer": answer,
                        "score_mode": score_mode,
                        "edge_threshold": edge_threshold,
                        "cohesion_mode": cohesion_mode,
                        "reason": "trace_overlap",
                        "left_unit": left,
                        "right_unit": right,
                        "edge_score": edge_score,
                        "overlap_traces_json": json_dumps(sorted(overlap)),
                        "left_text": units[left].text,
                        "right_text": units[right].text,
                    }
                )
            continue

        # Rule 4: bridge-resistant cohesive merging.
        cross_min, cross_q10, cross_mean, _ = cohesion_summary(
            left_nodes=sorted(members[left_root]),
            right_nodes=sorted(members[right_root]),
            score_store=score_store,
            score_mode=score_mode,
        )
        criterion = cross_min if cohesion_mode == "min" else cross_q10
        required = edge_threshold + cohesion_margin
        if criterion < required:
            diagnostics["reject_low_cohesion"] += 1
            if len(rejection_rows) < max_rejections:
                rejection_rows.append(
                    {
                        "id": question_id,
                        "answer": answer,
                        "score_mode": score_mode,
                        "edge_threshold": edge_threshold,
                        "cohesion_mode": cohesion_mode,
                        "reason": "low_cohesion",
                        "left_unit": left,
                        "right_unit": right,
                        "edge_score": edge_score,
                        "cross_min": cross_min,
                        "cross_q10": cross_q10,
                        "cross_mean": cross_mean,
                        "required_cohesion": required,
                        "left_cluster_size": len(members[left_root]),
                        "right_cluster_size": len(members[right_root]),
                        "left_text": units[left].text,
                        "right_text": units[right].text,
                    }
                )
            continue

        merge_roots(left_root, right_root)
        diagnostics["accepted_merges"] += 1

    roots = sorted({find(unit_id) for unit_id in unit_ids})
    diagnostics["n_clusters_before_support"] = len(roots)
    retained_nodes: list[list[int]] = []
    for root in roots:
        cluster_nodes = sorted(members[find(root)])
        cluster_traces = {units[node].trace_id for node in cluster_nodes}
        # Rule 3: minimum distinct-trace support.
        if len(cluster_traces) >= min_trace_support:
            retained_nodes.append(cluster_nodes)
        elif len(cluster_nodes) > 1:
            diagnostics["reject_insufficient_trace_support"] += 1

    communities: list[LocalCommunity] = []
    node_to_community: dict[int, int] = {}
    for community_id, nodes in enumerate(
        sorted(retained_nodes, key=lambda values: (min(values), len(values)))
    ):
        min_score, q10_score, mean_score = internal_community_scores(
            nodes, score_store, score_mode
        )
        representative = min(
            nodes,
            key=lambda node: (
                abs(units[node].position - np.mean([units[x].position for x in nodes])),
                len(units[node].text),
                node,
            ),
        )
        trace_ids = sorted({units[node].trace_id for node in nodes})
        community = LocalCommunity(
            community_id=community_id,
            nodes=nodes,
            trace_ids=trace_ids,
            support=len(trace_ids),
            min_score=min_score,
            q10_score=q10_score,
            mean_score=mean_score,
            representative_text=units[representative].text,
            mean_position=float(np.mean([units[node].position for node in nodes])),
        )
        communities.append(community)
        for node in nodes:
            node_to_community[node] = community_id

    diagnostics["n_retained_communities"] = len(communities)
    diagnostics["n_nodes_in_communities"] = len(node_to_community)
    diagnostics["n_units_answer"] = len(unit_ids)
    diagnostics["unit_coverage_fraction"] = (
        len(node_to_community) / len(unit_ids) if unit_ids else 0.0
    )
    diagnostics["max_community_size"] = max(
        (len(community.nodes) for community in communities), default=0
    )
    diagnostics["mean_community_size"] = float(
        np.mean([len(community.nodes) for community in communities])
        if communities else 0.0
    )
    return CommunityBuildResult(
        communities=communities,
        node_to_community=node_to_community,
        diagnostics=dict(diagnostics),
        rejection_rows=rejection_rows,
    )


# ---------------------------------------------------------------------------
# Transition and direct pairwise metrics
# ---------------------------------------------------------------------------


def build_trace_sequences(
    *,
    question: QuestionData,
    trace_ids: Sequence[int],
    node_to_community: dict[int, int],
) -> dict[int, list[int]]:
    sequences: dict[int, list[int]] = {}
    for trace_id in trace_ids:
        sequence: list[int] = []
        for unit_id in sorted(
            question.unit_ids_by_trace.get(trace_id, []),
            key=lambda value: question.units[value].local_index,
        ):
            community = node_to_community.get(unit_id)
            if community is None:
                continue
            if not sequence or sequence[-1] != community:
                sequence.append(community)
        sequences[trace_id] = sequence
    return sequences


def signature_stats(signatures: Sequence[tuple[Any, ...]], n_traces: int) -> dict[str, float]:
    nonempty = [signature for signature in signatures if signature]
    counts = Counter(nonempty)
    concentration = entropy_concentration(counts.values())
    return {
        "n_unique": float(len(counts)),
        "unique_ratio": len(counts) / max(n_traces, 1),
        "effective_per_trace": concentration["effective_number"] / max(n_traces, 1),
        "gini_simpson": concentration["gini_simpson"],
    }


def transition_metrics(
    *,
    question: QuestionData,
    answer: str,
    trace_ids: Sequence[int],
    communities: Sequence[LocalCommunity],
    node_to_community: dict[int, int],
    transition_min_support: int,
    late_position_threshold: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    sequences = build_trace_sequences(
        question=question,
        trace_ids=trace_ids,
        node_to_community=node_to_community,
    )
    support: Counter[tuple[int, int]] = Counter()
    trace_transition_sets: dict[int, set[tuple[int, int]]] = {}
    trace_transition_sequences: dict[int, tuple[tuple[int, int], ...]] = {}
    for trace_id, sequence in sequences.items():
        transitions = tuple(
            (left, right)
            for left, right in zip(sequence, sequence[1:])
            if left != right
        )
        trace_transition_sequences[trace_id] = transitions
        trace_transition_sets[trace_id] = set(transitions)
        support.update(set(transitions))

    qualifying = {
        transition for transition, count in support.items()
        if count >= transition_min_support
    }
    supports = np.asarray([support[item] for item in sorted(qualifying)], dtype=float)
    concentration = entropy_concentration(supports)
    n_traces = len(trace_ids)

    filtered_sets: list[set[tuple[int, int]]] = []
    filtered_sequences: list[tuple[tuple[int, int], ...]] = []
    for trace_id in trace_ids:
        filtered_sequence = tuple(
            transition for transition in trace_transition_sequences.get(trace_id, ())
            if transition in qualifying
        )
        filtered_sequences.append(filtered_sequence)
        filtered_sets.append(set(filtered_sequence))

    # Existing Jaccard behavior: one-empty/one-nonempty pairs receive distance
    # 1, while empty/empty pairs are ignored. Keep it for backward comparison.
    jaccard_distances: list[float] = []

    # New supported diversity: a trace pair receives diversity credit only
    # when both traces contain at least one qualifying transition. Pairs with
    # missing evidence on either side contribute zero, and the denominator is
    # all sampled trace pairs.
    supported_jaccard_sum = 0.0
    n_trace_pairs_total = 0
    n_trace_pairs_both_nonempty = 0

    for left in range(len(filtered_sets)):
        for right in range(left + 1, len(filtered_sets)):
            n_trace_pairs_total += 1
            union = filtered_sets[left] | filtered_sets[right]
            if union:
                distance = (
                    1.0
                    - len(filtered_sets[left] & filtered_sets[right]) / len(union)
                )
                jaccard_distances.append(distance)
            else:
                distance = 0.0

            if filtered_sets[left] and filtered_sets[right]:
                supported_jaccard_sum += distance
                n_trace_pairs_both_nonempty += 1

    supported_pairwise_jaccard = (
        supported_jaccard_sum / n_trace_pairs_total
        if n_trace_pairs_total
        else 0.0
    )
    transition_pair_evidence_coverage = (
        n_trace_pairs_both_nonempty / n_trace_pairs_total
        if n_trace_pairs_total
        else 0.0
    )

    set_stats = signature_stats(
        [tuple(sorted(values)) for values in filtered_sets], n_traces
    )
    sequence_stats = signature_stats(filtered_sequences, n_traces)

    adjacency: dict[int, set[int]] = defaultdict(set)
    for left, right in qualifying:
        adjacency[left].add(right)
    out_degrees = np.asarray([len(values) for values in adjacency.values()], dtype=float)

    community_by_id = {community.community_id: community for community in communities}

    # Weight branching by how broadly the checkpoint itself is supported.
    # A branch at a checkpoint occurring in all sampled traces counts more
    # than a branch at a checkpoint occurring in only the minimum number.
    branching_weight_total = 0.0
    branching_weight_mass = 0.0
    for community_id, destinations in adjacency.items():
        community = community_by_id.get(community_id)
        if community is None:
            continue
        weight = community.support / max(n_traces, 1)
        branching_weight_total += weight
        if len(destinations) > 1:
            branching_weight_mass += weight
    support_weighted_branching = (
        branching_weight_mass / branching_weight_total
        if branching_weight_total > 0
        else 0.0
    )

    late_transitions = {
        transition for transition in qualifying
        if min(
            community_by_id[transition[0]].mean_position,
            community_by_id[transition[1]].mean_position,
        ) >= late_position_threshold
    }
    weighted_transition_support = 0.0
    detail_rows: list[dict[str, Any]] = []
    for left, right in sorted(qualifying):
        left_community = community_by_id[left]
        right_community = community_by_id[right]
        edge_cohesion = min(left_community.min_score, right_community.min_score)
        weighted_transition_support += (
            support[(left, right)] / max(n_traces, 1)
        ) * edge_cohesion
        detail_rows.append(
            {
                "id": question.question_id,
                "answer": answer,
                "transition_min_support": transition_min_support,
                "from_community": left,
                "to_community": right,
                "support": support[(left, right)],
                "support_fraction": support[(left, right)] / max(n_traces, 1),
                "from_text": left_community.representative_text,
                "to_text": right_community.representative_text,
                "from_position": left_community.mean_position,
                "to_position": right_community.mean_position,
                "endpoint_min_cohesion": edge_cohesion,
                "is_late": (left, right) in late_transitions,
            }
        )

    trace_with_transition = sum(bool(values) for values in filtered_sets)
    return (
        {
            "n_shared_transition_types": float(len(qualifying)),
            "shared_transition_types_per_trace": len(qualifying) / max(n_traces, 1),
            "transition_trace_coverage": trace_with_transition / max(n_traces, 1),
            "transition_support_gini_simpson": concentration["gini_simpson"],
            "effective_shared_transitions_per_trace": (
                concentration["effective_number"] / max(n_traces, 1)
            ),
            "transition_set_diversity_gini_simpson": set_stats["gini_simpson"],
            "transition_sequence_diversity_gini_simpson": sequence_stats["gini_simpson"],
            "mean_pairwise_transition_jaccard_distance": float(
                np.mean(jaccard_distances) if jaccard_distances else 0.0
            ),
            "mean_pairwise_transition_jaccard_similarity": float(
                1.0 - np.mean(jaccard_distances) if jaccard_distances else 1.0
            ),
            "supported_pairwise_transition_jaccard_distance": float(
                supported_pairwise_jaccard
            ),
            "transition_pair_evidence_coverage": float(
                transition_pair_evidence_coverage
            ),
            "n_transition_trace_pairs_total": float(n_trace_pairs_total),
            "n_transition_trace_pairs_both_nonempty": float(
                n_trace_pairs_both_nonempty
            ),
            "coverage_weighted_transition_set_diversity": float(
                (trace_with_transition / max(n_traces, 1))
                * set_stats["gini_simpson"]
            ),
            "fraction_branching_checkpoints": float(
                np.mean(out_degrees > 1) if out_degrees.size else 0.0
            ),
            "support_weighted_branching_fraction": float(
                support_weighted_branching
            ),
            "max_transition_out_degree": float(
                out_degrees.max() if out_degrees.size else 0.0
            ),
            "late_shared_transition_fraction": (
                len(late_transitions) / len(qualifying) if qualifying else 0.0
            ),
            "cohesion_weighted_transition_support": float(weighted_transition_support),
            "n_trace_sequences_with_communities": float(
                sum(bool(sequence) for sequence in sequences.values())
            ),
        },
        detail_rows,
    )


def greedy_one_to_one_matches(
    *,
    left_units: Sequence[int],
    right_units: Sequence[int],
    score_store: PairScoreStore,
    score_mode: str,
    match_threshold: float,
    allowed_pairs: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int, float]]:
    pairs = [
        (left, right)
        for left in left_units
        for right in right_units
        if allowed_pairs is None
        or PairScoreStore.key(left, right) in allowed_pairs
    ]
    # In the KNN experiment, do not silently score omitted pairs here: direct
    # metrics must use the same candidate budget as the community experiment.
    if allowed_pairs is None:
        score_store.ensure(pairs)
    scored = sorted(
        [
            (left, right, score_store.score(left, right, score_mode))
            for left, right in pairs
        ],
        key=lambda item: (-item[2], item[0], item[1]),
    )
    used_left: set[int] = set()
    used_right: set[int] = set()
    selected: list[tuple[int, int, float]] = []
    for left, right, score in scored:
        if score < match_threshold:
            break
        if left in used_left or right in used_right:
            continue
        used_left.add(left)
        used_right.add(right)
        selected.append((left, right, score))
    return selected


def order_consistency(matches: Sequence[tuple[int, int, float]], units: Sequence[Unit]) -> float:
    if len(matches) < 2:
        return 0.0
    ordered = sorted(matches, key=lambda item: units[item[0]].local_index)
    right_positions = [units[right].local_index for _, right, _ in ordered]
    total = 0
    consistent = 0
    for left in range(len(right_positions)):
        for right in range(left + 1, len(right_positions)):
            total += 1
            consistent += int(right_positions[left] < right_positions[right])
    return consistent / total if total else 0.0


def trace_pair_metrics(
    *,
    question: QuestionData,
    answer: str,
    trace_ids: Sequence[int],
    score_store: PairScoreStore,
    score_mode: str,
    match_threshold: float,
    allowed_pairs: set[tuple[int, int]] | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    best_scores: list[float] = []
    top2_scores: list[float] = []
    matching_scores: list[float] = []
    order_scores: list[float] = []
    coverage: list[float] = []
    two_coverage: list[float] = []
    details: list[dict[str, Any]] = []

    for left_index, left_trace in enumerate(trace_ids):
        left_units = question.unit_ids_by_trace.get(left_trace, [])
        for right_trace in trace_ids[left_index + 1 :]:
            right_units = question.unit_ids_by_trace.get(right_trace, [])
            pairs = [
                (left, right)
                for left in left_units
                for right in right_units
                if allowed_pairs is None
                or PairScoreStore.key(left, right) in allowed_pairs
            ]
            if allowed_pairs is None:
                score_store.ensure(pairs)
            scores = np.asarray(
                [score_store.score(left, right, score_mode) for left, right in pairs],
                dtype=float,
            )
            best = float(scores.max()) if scores.size else float("nan")
            top2 = float(np.mean(np.sort(scores)[-min(2, len(scores)) :])) if scores.size else float("nan")
            matches = greedy_one_to_one_matches(
                left_units=left_units,
                right_units=right_units,
                score_store=score_store,
                score_mode=score_mode,
                match_threshold=match_threshold,
                allowed_pairs=allowed_pairs,
            )
            matching_score = float(np.mean([item[2] for item in matches])) if matches else 0.0
            order_score = order_consistency(matches, question.units)
            best_scores.append(best)
            top2_scores.append(top2)
            matching_scores.append(matching_score)
            order_scores.append(order_score)
            coverage.append(float(len(matches) >= 1))
            two_coverage.append(float(len(matches) >= 2))
            details.append(
                {
                    "id": question.question_id,
                    "answer": answer,
                    "score_mode": score_mode,
                    "left_trace": left_trace,
                    "right_trace": right_trace,
                    "n_left_units": len(left_units),
                    "n_right_units": len(right_units),
                    "n_available_scored_pairs": len(pairs),
                    "available_pair_fraction": (
                        len(pairs) / max(len(left_units) * len(right_units), 1)
                    ),
                    "best_score": best,
                    "top2_mean_score": top2,
                    "n_one_to_one_matches": len(matches),
                    "mean_one_to_one_score": matching_score,
                    "order_consistency": order_score,
                    "matches_json": json_dumps(
                        [
                            {
                                "left_unit": left,
                                "right_unit": right,
                                "score": score,
                                "left_text": question.units[left].text,
                                "right_text": question.units[right].text,
                            }
                            for left, right, score in matches
                        ]
                    ),
                }
            )
    return (
        {
            "mean_best_trace_pair_score": float(np.nanmean(best_scores)) if best_scores else np.nan,
            "mean_top2_trace_pair_score": float(np.nanmean(top2_scores)) if top2_scores else np.nan,
            "mean_greedy_one_to_one_score": float(np.mean(matching_scores)) if matching_scores else 0.0,
            "mean_order_consistency": float(np.mean(order_scores)) if order_scores else 0.0,
            "trace_pair_match_coverage": float(np.mean(coverage)) if coverage else 0.0,
            "trace_pair_two_match_coverage": float(np.mean(two_coverage)) if two_coverage else 0.0,
            "n_trace_pairs": float(len(coverage)),
        },
        details,
    )


def _path_exists(adjacency: dict[int, set[int]], start: int, goal: int) -> bool:
    """Return whether goal is reachable from start in a small directed graph."""
    stack = [start]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if node == goal:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adjacency.get(node, ()))
    return False


def _longest_path_nodes(adjacency: dict[int, set[int]]) -> int:
    """Longest path length in nodes for an acyclic directed graph."""
    nodes = set(adjacency)
    for targets in adjacency.values():
        nodes.update(targets)
    if not nodes:
        return 0
    indegree = {node: 0 for node in nodes}
    for source, targets in adjacency.items():
        for target in targets:
            indegree[target] += 1
    queue = sorted(node for node, degree in indegree.items() if degree == 0)
    distance = {node: 1 for node in nodes}
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for target in sorted(adjacency.get(node, ())):
            distance[target] = max(distance[target], distance[node] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
                queue.sort()
    if visited != len(nodes):
        return 0
    return max(distance.values(), default=0)


def precedence_metrics(
    *,
    question: QuestionData,
    answer: str,
    trace_ids: Sequence[int],
    communities: Sequence[LocalCommunity],
    node_to_community: dict[int, int],
    precedence_min_support: int,
    precedence_min_consistency: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """
    Measure order relations that survive insertions, omissions and shortcuts.

    For every pair of retained communities, count in how many traces the first
    occurs before the second and vice versa. Exact adjacency is not required.
    A shortcut-surviving pair is adjacent in at least one trace and separated
    by intermediate retained checkpoints in at least one other trace while its
    direction remains stable.
    """
    sequences = build_trace_sequences(
        question=question,
        trace_ids=trace_ids,
        node_to_community=node_to_community,
    )
    positions = {
        trace_id: {community_id: index for index, community_id in enumerate(sequence)}
        for trace_id, sequence in sequences.items()
    }
    community_by_id = {community.community_id: community for community in communities}
    community_ids = sorted(community_by_id)
    n_traces = len(trace_ids)

    supported_mass = 0.0
    shortcut_mass = 0.0
    reversal_numerator = 0.0
    reversal_denominator = 0.0
    n_supported = 0
    n_stable = 0
    n_shortcut = 0
    detail_rows: list[dict[str, Any]] = []
    stable_edges: list[tuple[int, int, float, int]] = []

    for left_index, first in enumerate(community_ids):
        for second in community_ids[left_index + 1:]:
            first_before = 0
            second_before = 0
            first_before_adjacent = 0
            second_before_adjacent = 0
            first_before_gaps: list[int] = []
            second_before_gaps: list[int] = []

            for trace_id in trace_ids:
                trace_positions = positions.get(trace_id, {})
                if first not in trace_positions or second not in trace_positions:
                    continue
                first_position = trace_positions[first]
                second_position = trace_positions[second]
                gap = abs(second_position - first_position)
                if first_position < second_position:
                    first_before += 1
                    first_before_gaps.append(gap)
                    if gap == 1:
                        first_before_adjacent += 1
                elif second_position < first_position:
                    second_before += 1
                    second_before_gaps.append(gap)
                    if gap == 1:
                        second_before_adjacent += 1

            support = first_before + second_before
            if support < precedence_min_support:
                continue
            n_supported += 1

            if first_before >= second_before:
                source, target = first, second
                directional_count = first_before
                reverse_count = second_before
                adjacent_count = first_before_adjacent
                directional_gaps = first_before_gaps
            else:
                source, target = second, first
                directional_count = second_before
                reverse_count = first_before
                adjacent_count = second_before_adjacent
                directional_gaps = second_before_gaps

            consistency = directional_count / support
            reversal_rate = reverse_count / support
            direction_strength = max(0.0, 2.0 * consistency - 1.0)
            support_fraction = support / max(n_traces, 1)
            nonadjacent_count = directional_count - adjacent_count
            stable = consistency >= precedence_min_consistency
            shortcut_survives = bool(
                stable and adjacent_count > 0 and nonadjacent_count > 0
            )

            supported_mass += support_fraction * direction_strength
            reversal_numerator += reverse_count
            reversal_denominator += support
            if stable:
                n_stable += 1
                stable_edges.append((source, target, consistency, support))
            if shortcut_survives:
                n_shortcut += 1
                shortcut_mass += support_fraction * direction_strength

            detail_rows.append(
                {
                    "id": question.question_id,
                    "answer": answer,
                    "precedence_min_support": precedence_min_support,
                    "from_community": source,
                    "to_community": target,
                    "cooccurrence_support": support,
                    "support_fraction": support_fraction,
                    "forward_count": directional_count,
                    "reverse_count": reverse_count,
                    "ordering_consistency": consistency,
                    "reversal_rate": reversal_rate,
                    "direction_strength": direction_strength,
                    "adjacent_forward_count": adjacent_count,
                    "nonadjacent_forward_count": nonadjacent_count,
                    "shortcut_survives": shortcut_survives,
                    "mean_forward_gap": float(np.mean(directional_gaps))
                    if directional_gaps else np.nan,
                    "from_text": community_by_id[source].representative_text,
                    "to_text": community_by_id[target].representative_text,
                    "from_position": community_by_id[source].mean_position,
                    "to_position": community_by_id[target].mean_position,
                }
            )

    # A majority-precedence graph can still contain a cycle across three or
    # more pairs. Add the strongest relations first and skip only relations
    # that would create a directed cycle, then measure its longest chain.
    acyclic_adjacency: dict[int, set[int]] = defaultdict(set)
    cycle_conflicts = 0
    for source, target, consistency, support in sorted(
        stable_edges,
        key=lambda value: (-value[2], -value[3], value[0], value[1]),
    ):
        if _path_exists(acyclic_adjacency, target, source):
            cycle_conflicts += 1
            continue
        acyclic_adjacency[source].add(target)

    chain_length = _longest_path_nodes(acyclic_adjacency)
    n_communities = len(community_ids)
    return (
        {
            "n_supported_precedence_pairs": float(n_supported),
            "n_stable_precedence_pairs": float(n_stable),
            "stable_precedence_pairs_per_trace": n_stable / max(n_traces, 1),
            "supported_precedence_mass": float(supported_mass),
            "precedence_reversal_rate": float(
                reversal_numerator / reversal_denominator
                if reversal_denominator > 0 else 0.0
            ),
            "n_shortcut_surviving_pairs": float(n_shortcut),
            "shortcut_survival_mass": float(shortcut_mass),
            "consensus_chain_length": float(chain_length),
            "consensus_chain_fraction": float(
                chain_length / n_communities if n_communities else 0.0
            ),
            "n_precedence_cycle_conflicts": float(cycle_conflicts),
        },
        detail_rows,
    )


def build_trace_schema_rows(
    *,
    question: QuestionData,
    answer: str,
    trace_ids: Sequence[int],
    communities: Sequence[LocalCommunity],
    node_to_community: dict[int, int],
) -> list[dict[str, Any]]:
    """One auditable retained checkpoint skeleton per sampled trace."""
    community_by_id = {community.community_id: community for community in communities}
    rows: list[dict[str, Any]] = []
    for trace_id in trace_ids:
        entries: list[tuple[int, int, str, int, int]] = []
        for unit_id in sorted(
            question.unit_ids_by_trace.get(trace_id, []),
            key=lambda value: question.units[value].local_index,
        ):
            community_id = node_to_community.get(unit_id)
            if community_id is None:
                continue
            unit = question.units[unit_id]
            community = community_by_id[community_id]
            entries.append(
                (
                    unit.local_index,
                    community_id,
                    unit.text,
                    checkpoint_word_count(unit.text),
                    community.support,
                )
            )
        community_sequence = [entry[1] for entry in entries]
        word_lengths = [entry[3] for entry in entries]
        transition_sequence = [
            f"C{left}->C{right}"
            for left, right in zip(community_sequence, community_sequence[1:])
            if left != right
        ]
        schema_with_lengths = " -> ".join(
            f"C{community_id}[words={word_count};support={support}]"
            for _, community_id, _, word_count, support in entries
        )
        rows.append(
            {
                "id": question.question_id,
                "answer": answer,
                "trace_id": trace_id,
                "is_correct_answer": bool(question.hits_by_trace[trace_id]),
                "raw_count": question.raw_count_by_answer[answer],
                "n_retained_checkpoints": len(entries),
                "community_sequence": " -> ".join(
                    f"C{community_id}" for community_id in community_sequence
                ),
                "transition_sequence": " | ".join(transition_sequence),
                "step_word_lengths_json": json_dumps(word_lengths),
                "raw_step_texts_json": json_dumps([entry[2] for entry in entries]),
                "mean_checkpoint_words": float(np.mean(word_lengths))
                if word_lengths else 0.0,
                "median_checkpoint_words": float(np.median(word_lengths))
                if word_lengths else 0.0,
                "total_checkpoint_words": int(sum(word_lengths)),
                "schema_with_lengths": schema_with_lengths,
            }
        )
    return rows


def community_metrics(
    *,
    question: QuestionData,
    communities: Sequence[LocalCommunity],
    n_traces: int,
    n_units: int,
) -> dict[str, float]:
    supports = [community.support for community in communities]
    support_fractions = [value / max(n_traces, 1) for value in supports]
    concentration = entropy_concentration(supports)
    traces_covered = set(
        trace_id for community in communities for trace_id in community.trace_ids
    )
    nodes_covered = sum(len(community.nodes) for community in communities)
    cohesion_weighted = sum(
        (community.support / max(n_traces, 1)) * community.min_score
        for community in communities
    )
    occurrence_word_lengths = [
        checkpoint_word_count(question.units[node].text)
        for community in communities
        for node in community.nodes
    ]
    mean_words = float(np.mean(occurrence_word_lengths)) if occurrence_word_lengths else 0.0
    median_words = float(np.median(occurrence_word_lengths)) if occurrence_word_lengths else 0.0
    mean_schema_length = sum(supports) / max(n_traces, 1)
    deep_recurrent_mass = float(sum(value * value for value in support_fractions))
    total_occurrence_mass = float(sum(supports))
    pair_fragmentation = (
        sum(value for value in supports if value == 2) / total_occurrence_mass
        if total_occurrence_mass > 0 else 0.0
    )
    return {
        "n_local_communities": float(len(communities)),
        "local_communities_per_trace": len(communities) / max(n_traces, 1),
        "community_unit_coverage": nodes_covered / max(n_units, 1),
        "community_trace_coverage": len(traces_covered) / max(n_traces, 1),
        "mean_community_support_fraction": float(
            np.mean(support_fractions) if supports else 0.0
        ),
        "mean_community_min_score": float(
            np.mean([community.min_score for community in communities])
            if communities else 0.0
        ),
        "mean_community_q10_score": float(
            np.mean([community.q10_score for community in communities])
            if communities else 0.0
        ),
        "cohesion_weighted_support": float(cohesion_weighted),
        "effective_local_communities_per_trace": (
            concentration["effective_number"] / max(n_traces, 1)
        ),
        "mean_checkpoint_schema_length": float(mean_schema_length),
        "deep_recurrent_checkpoint_mass": deep_recurrent_mass,
        "pair_fragmentation_ratio": float(pair_fragmentation),
        "mean_checkpoint_words": mean_words,
        "median_checkpoint_words": median_words,
        "atomic_recurrent_checkpoint_mass": float(
            deep_recurrent_mass / math.sqrt(max(mean_words, 1.0))
        ),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def metric_direction(metric: str) -> str:
    return "min" if metric in MINIMIZE_METRICS else "max"


def choose_majority(group: pd.DataFrame) -> pd.Series:
    return group.sort_values(
        ["raw_count", "first_trace_position", "answer"],
        ascending=[False, True, True],
        kind="mergesort",
    ).iloc[0]


def choose_by_metric(
    group: pd.DataFrame, metric: str, minimum_frequency: int
) -> tuple[pd.Series, str, int]:
    majority = choose_majority(group)
    eligible = group.loc[group["raw_count"] >= minimum_frequency].copy()
    if eligible.empty:
        return majority, "fallback_no_frequency", 0
    eligible[metric] = pd.to_numeric(eligible[metric], errors="coerce")
    eligible = eligible.loc[np.isfinite(eligible[metric].to_numpy(dtype=float))]
    if eligible.empty:
        return majority, "fallback_no_metric", 0
    direction = metric_direction(metric)
    best = eligible[metric].min() if direction == "min" else eligible[metric].max()
    tied = eligible.loc[np.isclose(eligible[metric], best, rtol=1e-10, atol=1e-12)]
    selected = tied.sort_values(
        ["raw_count", "first_trace_position", "answer"],
        ascending=[False, True, True],
        kind="mergesort",
    ).iloc[0]
    return selected, "metric" if len(tied) == 1 else "metric_tie", len(eligible)


def normalized_metric_margin(group: pd.DataFrame, metric: str, selected: pd.Series, majority: pd.Series) -> float:
    values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    span = float(values.max() - values.min())
    if span <= 1e-12:
        return 0.0
    selected_value = float(selected[metric])
    majority_value = float(majority[metric])
    difference = selected_value - majority_value
    if metric_direction(metric) == "min":
        difference = -difference
    return float(difference / span)


def evaluate_answer_metrics(
    *,
    answer_frame: pd.DataFrame,
    metrics: Sequence[str],
    frequency_thresholds: Sequence[int],
    override_margins: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions: list[dict[str, Any]] = []
    override_predictions: list[dict[str, Any]] = []
    config_columns = [
        "dataset_split",
        "experiment_mode",
        "sampling_seed",
        "prefix_order",
        "sample_size",
        "score_mode",
        "edge_quantile",
        "local_neighbor_k",
        "cohesion_mode",
        "community_min_support",
        "transition_min_support",
    ]
    group_columns = config_columns + ["id"]
    for keys, group in answer_frame.groupby(group_columns, dropna=False, sort=False):
        config = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,)))
        majority = choose_majority(group)
        for minimum_frequency in frequency_thresholds:
            for metric in metrics:
                if metric not in group.columns:
                    continue
                selected, reason, n_eligible = choose_by_metric(
                    group, metric, minimum_frequency
                )
                majority_correct = bool(majority["is_correct_answer"])
                selected_correct = bool(selected["is_correct_answer"])
                changed = str(selected["answer"]) != str(majority["answer"])
                margin = normalized_metric_margin(group, metric, selected, majority)
                sample_majority_answer = str(group["sample_majority_answer"].iloc[0])
                sample_majority_correct = bool(group["sample_majority_is_correct"].iloc[0])
                budget_majority_answer = str(group["budget_majority_answer"].iloc[0])
                budget_majority_correct = bool(group["budget_majority_is_correct"].iloc[0])
                changed_vs_sample_majority = str(selected["answer"]) != sample_majority_answer
                changed_vs_budget_majority = str(selected["answer"]) != budget_majority_answer
                base = {
                    **config,
                    "minimum_answer_frequency": minimum_frequency,
                    "metric": metric,
                    "selection_direction": metric_direction(metric),
                    "selected_answer": str(selected["answer"]),
                    "selected_metric_value": float(pd.to_numeric(selected[metric], errors="coerce")),
                    "selected_raw_count": int(selected["raw_count"]),
                    "selected_is_correct": selected_correct,
                    "majority_answer": str(majority["answer"]),
                    "majority_raw_count": int(majority["raw_count"]),
                    "majority_is_correct": majority_correct,
                    "changed_vs_majority": changed,
                    "repair": changed and (not majority_correct) and selected_correct,
                    "harm": changed and majority_correct and (not selected_correct),
                    "structural_trace_budget": int(group["structural_trace_budget"].iloc[0]),
                    "sample_majority_answer": sample_majority_answer,
                    "sample_majority_vote_count": int(group["sample_majority_vote_count"].iloc[0]),
                    "sample_majority_n_traces": int(group["sample_majority_n_traces"].iloc[0]),
                    "sample_majority_is_correct": sample_majority_correct,
                    "changed_vs_sample_majority": changed_vs_sample_majority,
                    "repair_vs_sample_majority": (
                        changed_vs_sample_majority and (not sample_majority_correct) and selected_correct
                    ),
                    "harm_vs_sample_majority": (
                        changed_vs_sample_majority and sample_majority_correct and (not selected_correct)
                    ),
                    "budget_majority_answer": budget_majority_answer,
                    "budget_majority_vote_count": int(group["budget_majority_vote_count"].iloc[0]),
                    "budget_majority_n_traces": int(group["budget_majority_n_traces"].iloc[0]),
                    "budget_majority_is_correct": budget_majority_correct,
                    "changed_vs_budget_majority": changed_vs_budget_majority,
                    "repair_vs_budget_majority": (
                        changed_vs_budget_majority and (not budget_majority_correct) and selected_correct
                    ),
                    "harm_vs_budget_majority": (
                        changed_vs_budget_majority and budget_majority_correct and (not selected_correct)
                    ),
                    "selection_reason": reason,
                    "n_eligible_answers": n_eligible,
                    "normalized_margin_vs_majority": margin,
                    "category": group["category"].iloc[0] if "category" in group else "",
                }
                predictions.append(base)
                for required_margin in override_margins:
                    override_selected = selected if changed and margin >= required_margin else majority
                    override_answer = str(override_selected["answer"])
                    override_changed = override_answer != str(majority["answer"])
                    override_correct = bool(override_selected["is_correct_answer"])
                    override_changed_vs_sample = override_answer != sample_majority_answer
                    override_changed_vs_budget = override_answer != budget_majority_answer
                    override_predictions.append(
                        {
                            **base,
                            "required_override_margin": required_margin,
                            "selected_answer": override_answer,
                            "selected_is_correct": override_correct,
                            "changed_vs_majority": override_changed,
                            "repair": override_changed and (not majority_correct) and override_correct,
                            "harm": override_changed and majority_correct and (not override_correct),
                            "changed_vs_sample_majority": override_changed_vs_sample,
                            "repair_vs_sample_majority": (
                                override_changed_vs_sample and (not sample_majority_correct) and override_correct
                            ),
                            "harm_vs_sample_majority": (
                                override_changed_vs_sample and sample_majority_correct and (not override_correct)
                            ),
                            "changed_vs_budget_majority": override_changed_vs_budget,
                            "repair_vs_budget_majority": (
                                override_changed_vs_budget and (not budget_majority_correct) and override_correct
                            ),
                            "harm_vs_budget_majority": (
                                override_changed_vs_budget and budget_majority_correct and (not override_correct)
                            ),
                            "selection_reason": (
                                "override" if override_changed else "keep_majority"
                            ),
                        }
                    )

    prediction_frame = pd.DataFrame(predictions)
    override_frame = pd.DataFrame(override_predictions)

    def summarize(frame: pd.DataFrame, extra: Sequence[str]) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame()
        columns = config_columns + list(extra) + [
            "minimum_answer_frequency", "metric", "selection_direction"
        ]
        rows = []
        for keys, subset in frame.groupby(columns, dropna=False, sort=False):
            config = dict(zip(columns, keys if isinstance(keys, tuple) else (keys,)))
            rows.append(
                {
                    **config,
                    "n_ids": int(len(subset)),
                    "correct": int(subset["selected_is_correct"].astype(bool).sum()),
                    "accuracy": float(subset["selected_is_correct"].astype(bool).mean()),
                    "majority_correct": int(subset["majority_is_correct"].astype(bool).sum()),
                    "majority_accuracy": float(subset["majority_is_correct"].astype(bool).mean()),
                    "sample_majority_correct": int(subset["sample_majority_is_correct"].astype(bool).sum()),
                    "sample_majority_accuracy": float(subset["sample_majority_is_correct"].astype(bool).mean()),
                    "budget_majority_correct": int(subset["budget_majority_is_correct"].astype(bool).sum()),
                    "budget_majority_accuracy": float(subset["budget_majority_is_correct"].astype(bool).mean()),
                    "accuracy_gain_vs_full_majority": float(
                        subset["selected_is_correct"].astype(bool).mean()
                        - subset["majority_is_correct"].astype(bool).mean()
                    ),
                    "accuracy_gain_vs_sample_majority": float(
                        subset["selected_is_correct"].astype(bool).mean()
                        - subset["sample_majority_is_correct"].astype(bool).mean()
                    ),
                    "accuracy_gain_vs_budget_majority": float(
                        subset["selected_is_correct"].astype(bool).mean()
                        - subset["budget_majority_is_correct"].astype(bool).mean()
                    ),
                    "changed_vs_majority": int(subset["changed_vs_majority"].astype(bool).sum()),
                    "repairs": int(subset["repair"].astype(bool).sum()),
                    "harms": int(subset["harm"].astype(bool).sum()),
                    "net_repairs": int(subset["repair"].astype(bool).sum() - subset["harm"].astype(bool).sum()),
                    "changed_vs_sample_majority": int(subset["changed_vs_sample_majority"].astype(bool).sum()),
                    "repairs_vs_sample_majority": int(subset["repair_vs_sample_majority"].astype(bool).sum()),
                    "harms_vs_sample_majority": int(subset["harm_vs_sample_majority"].astype(bool).sum()),
                    "net_repairs_vs_sample_majority": int(
                        subset["repair_vs_sample_majority"].astype(bool).sum()
                        - subset["harm_vs_sample_majority"].astype(bool).sum()
                    ),
                    "changed_vs_budget_majority": int(subset["changed_vs_budget_majority"].astype(bool).sum()),
                    "repairs_vs_budget_majority": int(subset["repair_vs_budget_majority"].astype(bool).sum()),
                    "harms_vs_budget_majority": int(subset["harm_vs_budget_majority"].astype(bool).sum()),
                    "net_repairs_vs_budget_majority": int(
                        subset["repair_vs_budget_majority"].astype(bool).sum()
                        - subset["harm_vs_budget_majority"].astype(bool).sum()
                    ),
                    "mean_structural_trace_budget": float(subset["structural_trace_budget"].mean()),
                    "mean_normalized_margin": float(subset["normalized_margin_vs_majority"].mean()),
                }
            )
        return pd.DataFrame(rows).sort_values(
            ["accuracy", "net_repairs", "repairs"],
            ascending=[False, False, False],
            kind="mergesort",
        )

    return (
        prediction_frame,
        summarize(prediction_frame, []),
        override_frame,
        summarize(override_frame, ["required_override_margin"]),
    )


# ---------------------------------------------------------------------------
# Per-question experiment execution
# ---------------------------------------------------------------------------


def question_trace_order(
    question: QuestionData,
    seed: int,
    prefix_order: str,
) -> list[int]:
    """Return one nested question-level trace order.

    ``input`` preserves the row order in the input CSV. ``shuffled`` uses one
    deterministic question-level permutation, so prefix k is always contained
    in prefix k+1.
    """
    n_traces = len(question.answers_by_trace)
    if prefix_order == "input":
        return list(range(n_traces))
    if prefix_order != "shuffled":
        raise ValueError(f"Unknown prefix order: {prefix_order}")
    stable = int(
        hashlib.sha256(
            f"question-prefix|{seed}|{question.question_id}".encode()
        ).hexdigest()[:16],
        16,
    )
    rng = np.random.default_rng(stable)
    return [int(value) for value in rng.permutation(n_traces).tolist()]


def majority_from_trace_order(
    question: QuestionData, trace_ids_in_order: Sequence[int]
) -> dict[str, Any]:
    """Majority answer with deterministic first-occurrence tie breaking."""
    trace_ids = [int(value) for value in trace_ids_in_order]
    if not trace_ids:
        raise ValueError("Cannot calculate majority from an empty trace sample.")
    counts: Counter[str] = Counter()
    first_position: dict[str, int] = {}
    for position, trace_id in enumerate(trace_ids):
        answer = str(question.answers_by_trace[trace_id])
        counts[answer] += 1
        first_position.setdefault(answer, position)
    answer = sorted(
        counts,
        key=lambda value: (-counts[value], first_position[value], value),
    )[0]
    return {
        "answer": answer,
        "vote_count": int(counts[answer]),
        "n_traces": int(len(trace_ids)),
        "is_correct": bool(question.is_correct_by_answer.get(answer, False)),
    }


def sample_size_majority_fields(
    *,
    question: QuestionData,
    prefix_trace_ids: Sequence[int],
) -> dict[str, Any]:
    """Return matched-prefix and full majority baselines.

    In v8 the structural method and ``sample_majority`` use the exact same
    question-level prefix. ``budget_majority`` is retained as a backward-
    compatible alias and is therefore identical to ``sample_majority``.
    """
    prefix = [int(value) for value in prefix_trace_ids]
    prefix_majority = majority_from_trace_order(question, prefix)
    full_majority = majority_from_trace_order(
        question, list(range(len(question.answers_by_trace)))
    )
    return {
        "structural_trace_budget": int(len(prefix)),
        "prefix_trace_ids_json": json_dumps(prefix),
        "sample_majority_answer": prefix_majority["answer"],
        "sample_majority_vote_count": prefix_majority["vote_count"],
        "sample_majority_n_traces": prefix_majority["n_traces"],
        "sample_majority_is_correct": prefix_majority["is_correct"],
        # Backward-compatible aliases. These are intentionally identical.
        "budget_majority_answer": prefix_majority["answer"],
        "budget_majority_vote_count": prefix_majority["vote_count"],
        "budget_majority_n_traces": prefix_majority["n_traces"],
        "budget_majority_is_correct": prefix_majority["is_correct"],
        "full_majority_answer": full_majority["answer"],
        "full_majority_vote_count": full_majority["vote_count"],
        "full_majority_n_traces": full_majority["n_traces"],
        "full_majority_is_correct": full_majority["is_correct"],
    }


def process_question(
    *,
    question: QuestionData,
    experiment_mode: str,
    pipeline: Any,
    embedder: Any | None,
    reranker: CachedDirectionalReranker,
    args: argparse.Namespace,
    category: str,
) -> tuple[Any, ...]:
    score_store = PairScoreStore(units=question.units, reranker=reranker)
    unit_artifact_rows: list[dict[str, Any]] = []
    sampling_artifact_rows: list[dict[str, Any]] = []
    pair_score_artifact_rows: list[dict[str, Any]] = []
    if args.save_recalculation_artifacts:
        for unit in question.units:
            unit_artifact_rows.append({
                "dataset_split": args.dataset_split,
                "id": question.question_id,
                "unit_id": unit.unit_id,
                "trace_id": unit.trace_id,
                "answer": unit.answer,
                "is_correct_answer": question.is_correct_by_answer.get(unit.answer, False),
                "full_raw_count": question.raw_count_by_answer.get(unit.answer, 0),
                "local_index": unit.local_index,
                "relative_position": unit.position,
                "raw_step_text": unit.text,
                "embedding_text": unit.embedding_text,
                "step_chars": len(unit.text),
                "step_words": checkpoint_word_count(unit.text),
            })

    # Candidate generation and initial reranking are shared by all downstream
    # configurations. For sampled all-pairs, pairs differ by sampling seed/size.
    global_candidate_pairs: list[tuple[int, int]] = []
    global_embedding_edges: list[tuple[int, int, float]] = []
    if experiment_mode == "knn":
        if embedder is None:
            raise RuntimeError("KNN mode requires an embedding model.")
        embeddings = pipeline.preprocess_embeddings(
            embedder.encode([unit.embedding_text for unit in question.units]),
            "none",
            0,
        )
        global_embedding_edges = build_embedding_knn_candidates(
            embeddings=embeddings,
            units=question.units,
            k=args.embedding_candidate_k,
            edge_mode=args.embedding_candidate_edge_mode,
        )
        global_candidate_pairs = [(left, right) for left, right, _ in global_embedding_edges]
        score_store.ensure(global_candidate_pairs, show_progress_bar=False)

    answer_rows: list[dict[str, Any]] = []
    community_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    precedence_rows: list[dict[str, Any]] = []
    trace_schema_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    trace_pair_rows: list[dict[str, Any]] = []

    sampling_seeds = args.sampling_seed_values if experiment_mode == "sampled_allpairs" else [-1]
    sample_sizes = args.sample_size_values if experiment_mode == "sampled_allpairs" else [0]

    for sampling_seed in sampling_seeds:
        # One question-level order is shared by majority and community analysis.
        prefix_order = question_trace_order(
            question, sampling_seed, args.prefix_order
        )
        for sample_size in sample_sizes:
            prefix_count = (
                len(prefix_order)
                if sample_size <= 0
                else min(int(sample_size), len(prefix_order))
            )
            prefix_trace_ids = prefix_order[:prefix_count]
            prefix_position = {
                int(trace_id): position
                for position, trace_id in enumerate(prefix_trace_ids)
            }
            observed_answers = list(
                dict.fromkeys(
                    str(question.answers_by_trace[trace_id])
                    for trace_id in prefix_trace_ids
                )
            )
            selected_traces_by_answer: dict[str, list[int]] = {
                answer: [] for answer in observed_answers
            }
            for trace_id in prefix_trace_ids:
                selected_traces_by_answer[
                    str(question.answers_by_trace[trace_id])
                ].append(int(trace_id))

            observed_count_by_answer = {
                answer: len(trace_ids)
                for answer, trace_ids in selected_traces_by_answer.items()
            }
            observed_first_position_by_answer = {
                answer: min(prefix_position[trace_id] for trace_id in trace_ids)
                for answer, trace_ids in selected_traces_by_answer.items()
            }

            answer_candidate_pairs: dict[str, list[tuple[int, int]]] = {}
            if experiment_mode == "sampled_allpairs":
                for answer in observed_answers:
                    all_trace_ids = question.traces_by_answer[answer]
                    selected = selected_traces_by_answer[answer]
                    if args.save_recalculation_artifacts:
                        sampling_artifact_rows.append({
                            "dataset_split": args.dataset_split,
                            "id": question.question_id,
                            "sampling_seed": sampling_seed,
                            "prefix_order": args.prefix_order,
                            "requested_sample_size": sample_size,
                            "actual_prefix_size": prefix_count,
                            "prefix_trace_ids_json": json_dumps(prefix_trace_ids),
                            "answer": answer,
                            "is_correct_answer": question.is_correct_by_answer[answer],
                            "raw_count": len(selected),
                            "full_raw_count": question.raw_count_by_answer[answer],
                            "available_trace_count": len(all_trace_ids),
                            "sampled_trace_count": len(selected),
                            "sampled_trace_ids_json": json_dumps(selected),
                            "sampled_all_available": len(selected) == len(all_trace_ids),
                        })
                    answer_candidate_pairs[answer] = build_all_pairs_within_answer(
                        question=question, trace_ids=selected
                    )
                all_pairs = list(
                    dict.fromkeys(
                        pair for pairs in answer_candidate_pairs.values() for pair in pairs
                    )
                )
                score_store.ensure(all_pairs, show_progress_bar=False)
            else:
                observed_answers = list(question.traces_by_answer)
                selected_traces_by_answer = {
                    answer: list(trace_ids)
                    for answer, trace_ids in question.traces_by_answer.items()
                }
                observed_count_by_answer = dict(question.raw_count_by_answer)
                observed_first_position_by_answer = dict(question.first_trace_by_answer)
                prefix_trace_ids = list(range(len(question.answers_by_trace)))
                prefix_count = len(prefix_trace_ids)
                for answer, trace_ids in selected_traces_by_answer.items():
                    trace_set = set(trace_ids)
                    answer_candidate_pairs[answer] = [
                        (left, right)
                        for left, right in global_candidate_pairs
                        if question.units[left].trace_id in trace_set
                        and question.units[right].trace_id in trace_set
                    ]

            majority_fields = sample_size_majority_fields(
                question=question,
                prefix_trace_ids=prefix_trace_ids,
            )

            for score_mode in args.score_mode_values:
                for answer in observed_answers:
                    all_trace_ids = question.traces_by_answer[answer]
                    selected_trace_ids = selected_traces_by_answer[answer]
                    unit_ids = [
                        unit_id
                        for trace_id in selected_trace_ids
                        for unit_id in question.unit_ids_by_trace.get(trace_id, [])
                    ]
                    candidate_pairs = answer_candidate_pairs[answer]
                    if candidate_pairs:
                        candidate_edges = score_store.scored_edges(candidate_pairs, score_mode)
                    else:
                        candidate_edges = []

                    local_k_values = args.local_neighbor_k_values if experiment_mode == "knn" else [0]
                    for local_neighbor_k in local_k_values:
                        final_edges = top_k_edges(
                            edges=candidate_edges,
                            unit_ids=unit_ids,
                            k=local_neighbor_k,
                            edge_mode=args.local_edge_mode,
                        )
                        finite_scores = np.asarray([edge[2] for edge in final_edges], dtype=float)
                        for edge_quantile in args.edge_quantile_values:
                            if finite_scores.size:
                                edge_threshold = float(np.quantile(finite_scores, edge_quantile))
                            else:
                                edge_threshold = float("inf")
                            # Direct pairwise diagnostics use exactly the same
                            # scored-pair budget as this configuration. They are
                            # independent of cohesion/support sweeps, so compute
                            # them once here rather than repeatedly.
                            allowed_direct_pairs = {
                                PairScoreStore.key(left, right)
                                for left, right, _ in final_edges
                            }
                            if len(selected_trace_ids) >= 2 and np.isfinite(edge_threshold):
                                direct_values_once, pair_details_once = trace_pair_metrics(
                                    question=question,
                                    answer=answer,
                                    trace_ids=selected_trace_ids,
                                    score_store=score_store,
                                    score_mode=score_mode,
                                    match_threshold=edge_threshold,
                                    allowed_pairs=allowed_direct_pairs,
                                )
                            else:
                                direct_values_once, pair_details_once = (
                                    {
                                        "mean_best_trace_pair_score": np.nan,
                                        "mean_top2_trace_pair_score": np.nan,
                                        "mean_greedy_one_to_one_score": 0.0,
                                        "mean_order_consistency": 0.0,
                                        "trace_pair_match_coverage": 0.0,
                                        "trace_pair_two_match_coverage": 0.0,
                                        "n_trace_pairs": 0.0,
                                    },
                                    [],
                                )
                            pair_details_saved = False
                            for cohesion_mode in args.cohesion_mode_values:
                                for community_min_support in args.community_support_values:
                                    # Do not materialize configurations that are
                                    # impossible for every answer at this requested
                                    # sample size (e.g. support 5 with sample size 3).
                                    if (
                                        experiment_mode == "sampled_allpairs"
                                        and sample_size > 0
                                        and community_min_support > sample_size
                                    ):
                                        continue
                                    if len(selected_trace_ids) < community_min_support:
                                        # Still create zero-evidence rows so final-answer
                                        # selection can fall back to majority.
                                        support_impossible = True
                                    else:
                                        support_impossible = False
                                    if support_impossible or not np.isfinite(edge_threshold):
                                        build = CommunityBuildResult(
                                            communities=[],
                                            node_to_community={},
                                            diagnostics={
                                                "n_proposal_edges": len(final_edges),
                                                "n_edges_at_threshold": 0,
                                                "n_merge_attempts": 0,
                                                "accepted_merges": 0,
                                                "n_retained_communities": 0,
                                                "n_nodes_in_communities": 0,
                                                "n_units_answer": len(unit_ids),
                                                "unit_coverage_fraction": 0.0,
                                                "support_impossible": int(support_impossible),
                                            },
                                            rejection_rows=[],
                                        )
                                    else:
                                        build = build_constrained_communities(
                                            question_id=question.question_id,
                                            answer=answer,
                                            units=question.units,
                                            unit_ids=unit_ids,
                                            proposal_edges=final_edges,
                                            score_store=score_store,
                                            score_mode=score_mode,
                                            edge_threshold=edge_threshold,
                                            cohesion_mode=cohesion_mode,
                                            cohesion_margin=args.cohesion_margin,
                                            min_trace_support=community_min_support,
                                            max_rejections=args.max_rejections_per_answer,
                                        )

                                    base_config = {
                                        "dataset_split": args.dataset_split,
                                        "id": question.question_id,
                                        "category": category,
                                        "experiment_mode": experiment_mode,
                                        "sampling_seed": sampling_seed,
                                        "prefix_order": args.prefix_order,
                                        "sample_size": sample_size,
                                        "score_mode": score_mode,
                                        "edge_quantile": edge_quantile,
                                        "local_neighbor_k": local_neighbor_k,
                                        "cohesion_mode": cohesion_mode,
                                        "community_min_support": community_min_support,
                                    }

                                    for schema_row in build_trace_schema_rows(
                                        question=question,
                                        answer=answer,
                                        trace_ids=selected_trace_ids,
                                        communities=build.communities,
                                        node_to_community=build.node_to_community,
                                    ):
                                        trace_schema_rows.append({
                                            **base_config,
                                            **schema_row,
                                            "raw_count": observed_count_by_answer[answer],
                                            "full_raw_count": question.raw_count_by_answer[answer],
                                        })

                                    for transition_min_support in args.transition_support_values:
                                        if (
                                            experiment_mode == "sampled_allpairs"
                                            and sample_size > 0
                                            and transition_min_support > sample_size
                                        ):
                                            continue
                                        community_values = community_metrics(
                                            question=question,
                                            communities=build.communities,
                                            n_traces=len(selected_trace_ids),
                                            n_units=len(unit_ids),
                                        )
                                        transition_values, details = transition_metrics(
                                            question=question,
                                            answer=answer,
                                            trace_ids=selected_trace_ids,
                                            communities=build.communities,
                                            node_to_community=build.node_to_community,
                                            transition_min_support=transition_min_support,
                                            late_position_threshold=args.late_position_threshold,
                                        )
                                        precedence_values, precedence_details = precedence_metrics(
                                            question=question,
                                            answer=answer,
                                            trace_ids=selected_trace_ids,
                                            communities=build.communities,
                                            node_to_community=build.node_to_community,
                                            precedence_min_support=transition_min_support,
                                            precedence_min_consistency=args.precedence_min_consistency,
                                        )
                                        precedence_values["atomic_ordered_backbone_mass"] = float(
                                            precedence_values["supported_precedence_mass"]
                                            / math.sqrt(max(community_values["mean_checkpoint_words"], 1.0))
                                        )
                                        direct_values = direct_values_once
                                        pair_details = pair_details_once
                                        if not candidate_pairs:
                                            evidence_stage = "candidate_failure"
                                        elif not final_edges:
                                            evidence_stage = "edge_selection_failure"
                                        elif not build.communities:
                                            evidence_stage = "community_failure"
                                        elif transition_values["n_shared_transition_types"] <= 0:
                                            evidence_stage = "transition_failure"
                                        else:
                                            evidence_stage = "transition_evidence_available"

                                        if not build.communities:
                                            order_evidence_stage = "community_failure"
                                        elif precedence_values["n_supported_precedence_pairs"] <= 0:
                                            order_evidence_stage = "no_supported_precedence"
                                        elif precedence_values["n_stable_precedence_pairs"] <= 0:
                                            order_evidence_stage = "precedence_reversal_failure"
                                        elif precedence_values["n_shortcut_surviving_pairs"] > 0:
                                            order_evidence_stage = "shortcut_evidence_available"
                                        else:
                                            order_evidence_stage = "stable_precedence_available"
                                        answer_rows.append(
                                            {
                                                **base_config,
                                                "transition_min_support": transition_min_support,
                                                "answer": answer,
                                                "evidence_stage": evidence_stage,
                                                "order_evidence_stage": order_evidence_stage,
                                                "raw_count": observed_count_by_answer[answer],
                                                "full_raw_count": question.raw_count_by_answer[answer],
                                                "available_trace_count": len(all_trace_ids),
                                                "sampled_trace_count": len(selected_trace_ids),
                                                "sampled_all_available": bool(
                                                    len(selected_trace_ids) == len(all_trace_ids)
                                                ),
                                                "sampling_fraction": (
                                                    len(selected_trace_ids) / len(all_trace_ids)
                                                    if all_trace_ids else 0.0
                                                ),
                                                "first_trace_position": observed_first_position_by_answer[answer],
                                                "is_correct_answer": question.is_correct_by_answer[answer],
                                                **majority_fields,
                                                "edge_threshold": edge_threshold,
                                                "n_units_answer": len(unit_ids),
                                                "n_embedding_candidate_pairs_within_answer": len(candidate_pairs),
                                                "n_final_local_edges": len(final_edges),
                                                **community_values,
                                                **transition_values,
                                                **precedence_values,
                                                **direct_values,
                                            }
                                        )
                                        for detail in details:
                                            transition_rows.append(
                                                {
                                                    **base_config,
                                                    "transition_min_support": transition_min_support,
                                                    **detail,
                                                }
                                            )
                                        for detail in precedence_details:
                                            precedence_rows.append(
                                                {
                                                    **base_config,
                                                    "transition_min_support": transition_min_support,
                                                    "precedence_min_consistency": args.precedence_min_consistency,
                                                    **detail,
                                                }
                                            )
                                        # Pair details do not depend on community or
                                        # transition support; store only once for this
                                        # score/threshold/local-k configuration.
                                        if (
                                            not pair_details_saved
                                            and transition_min_support == args.transition_support_values[0]
                                        ):
                                            for detail in pair_details:
                                                trace_pair_rows.append({**base_config, **detail})
                                            pair_details_saved = True

                                    for community in build.communities:
                                        community_rows.append(
                                            {
                                                **base_config,
                                                "answer": answer,
                                                "community_id": community.community_id,
                                                "support": community.support,
                                                "support_fraction": community.support / max(len(selected_trace_ids), 1),
                                                "n_nodes": len(community.nodes),
                                                "trace_ids_json": json_dumps(community.trace_ids),
                                                "node_ids_json": json_dumps(community.nodes),
                                                "member_texts_json": json_dumps(
                                                    [question.units[node].text for node in community.nodes]
                                                ),
                                                "representative_text": community.representative_text,
                                                "min_score": community.min_score,
                                                "q10_score": community.q10_score,
                                                "mean_score": community.mean_score,
                                                "mean_position": community.mean_position,
                                            }
                                        )
                                    rejection_rows.extend(
                                        {**base_config, **row} for row in build.rejection_rows
                                    )

                                    diagnostics = {
                                        **base_config,
                                        "answer": answer,
                                        "raw_count": observed_count_by_answer[answer],
                                        "full_raw_count": question.raw_count_by_answer[answer],
                                        "available_trace_count": len(all_trace_ids),
                                        "sampled_trace_count": len(selected_trace_ids),
                                        "sampled_all_available": bool(
                                            len(selected_trace_ids) == len(all_trace_ids)
                                        ),
                                        "sampling_fraction": (
                                            len(selected_trace_ids) / len(all_trace_ids)
                                            if all_trace_ids else 0.0
                                        ),
                                        "n_units_answer": len(unit_ids),
                                        "n_global_embedding_candidates": len(global_candidate_pairs),
                                        "n_within_answer_candidate_pairs": len(candidate_pairs),
                                        "n_final_local_edges": len(final_edges),
                                        "edge_threshold": edge_threshold,
                                        "candidate_score_q50": float(np.quantile(finite_scores, 0.50)) if finite_scores.size else np.nan,
                                        "candidate_score_q90": float(np.quantile(finite_scores, 0.90)) if finite_scores.size else np.nan,
                                        **build.diagnostics,
                                    }
                                    if not candidate_pairs:
                                        diagnostics["failure_stage"] = "no_within_answer_candidates"
                                    elif not final_edges:
                                        diagnostics["failure_stage"] = "no_final_edges"
                                    elif not build.communities:
                                        diagnostics["failure_stage"] = "no_supported_communities"
                                    else:
                                        diagnostics["failure_stage"] = "community_evidence_available"
                                    diagnostic_rows.append(diagnostics)

    if args.save_recalculation_artifacts:
        for (left, right), (forward_raw, reverse_raw) in sorted(score_store.raw.items()):
            left_unit = question.units[left]
            right_unit = question.units[right]
            pair_score_artifact_rows.append({
                "dataset_split": args.dataset_split,
                "id": question.question_id,
                "left_unit_id": left,
                "right_unit_id": right,
                "left_trace_id": left_unit.trace_id,
                "right_trace_id": right_unit.trace_id,
                "left_answer": left_unit.answer,
                "right_answer": right_unit.answer,
                "same_answer": left_unit.answer == right_unit.answer,
                "left_text": left_unit.text,
                "right_text": right_unit.text,
                "forward_raw_logit": forward_raw,
                "reverse_raw_logit": reverse_raw,
                "raw_mean": score_from_directional(forward_raw, reverse_raw, "raw_mean"),
                "raw_min": score_from_directional(forward_raw, reverse_raw, "raw_min"),
                "sigmoid_mean": score_from_directional(forward_raw, reverse_raw, "sigmoid_mean"),
                "sigmoid_min": score_from_directional(forward_raw, reverse_raw, "sigmoid_min"),
            })

    return (
        answer_rows,
        community_rows,
        transition_rows,
        precedence_rows,
        trace_schema_rows,
        diagnostic_rows,
        rejection_rows + trace_pair_rows,
        unit_artifact_rows,
        sampling_artifact_rows,
        pair_score_artifact_rows,
    )


# ---------------------------------------------------------------------------
# CLI and orchestration
# ---------------------------------------------------------------------------


def build_parser(experiment_mode: str = "knn") -> argparse.ArgumentParser:
    description = (
        "Answer-local constrained communities using global embedding KNN + reranker."
        if experiment_mode == "knn"
        else "Answer-local constrained communities on exact question-level trace prefixes."
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", required=True)
    parser.add_argument("--pipeline-script", default="checkpoint_route_convergence_punctuation.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--dataset-split", default="unspecified", help="Label written to every output row, e.g. train or test.")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--text-col", default="completion")
    parser.add_argument("--answer-col", default="final_answer_new")
    parser.add_argument("--hit-col", default="row_hit")

    parser.add_argument("--hf-cache-dir", default="/workspace-SR003.nfs2/.cache/hub")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--allow-model-downloads", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-cache-path", default="qwen3_checkpoint_embeddings.sqlite")
    parser.add_argument("--reranker-cache-path", default="qwen3_checkpoint_directional_raw_logits.sqlite")
    parser.add_argument(
        "--local-cache-dir",
        default=None,
        help=(
            "Optional local filesystem directory for both SQLite caches. "
            "Recommended on NFS systems, e.g. /tmp/$USER/checkpoint_short50_cache."
        ),
    )
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--reranker-batch-size", type=int, default=64)
    parser.add_argument("--embedding-max-length", type=int, default=256)
    parser.add_argument("--reranker-max-length", type=int, default=256)
    parser.add_argument("--reranker-instruction", default=DEFAULT_INSTRUCTION)

    parser.add_argument("--unit-context", choices=["current", "previous_current"], default="current")
    parser.add_argument("--min-unit-chars", type=int, default=1)
    parser.add_argument("--max-unit-chars", type=int, default=900)
    parser.add_argument("--max-units-per-trace", type=int, default=96)
    parser.add_argument("--minimum-unit-position", type=float, default=0.0)
    parser.add_argument("--late-position-threshold", type=float, default=0.50)
    parser.add_argument(
        "--precedence-min-consistency",
        type=float,
        default=0.80,
        help=(
            "Minimum fraction of co-occurring traces agreeing on the same "
            "checkpoint order before the precedence relation is called stable."
        ),
    )

    # Experiment 1 candidate settings.
    parser.add_argument("--embedding-candidate-k", type=int, default=50)
    parser.add_argument("--embedding-candidate-edge-mode", choices=["mutual", "union"], default="union")
    parser.add_argument("--local-neighbor-k-values", default="1,2,3,5")
    parser.add_argument("--local-edge-mode", choices=["mutual", "union"], default="mutual")

    # Prefix-budget settings. sample_size is traces per QUESTION.
    parser.add_argument(
        "--sample-sizes",
        default="2-99",
        help="Question-level prefix sizes, e.g. 2-99 or 2,3,5,10,20,99.",
    )
    parser.add_argument(
        "--prefix-order",
        choices=["input", "shuffled"],
        default="input",
        help=(
            "Use the first k input rows per question, or one deterministic "
            "nested shuffled order per question/seed."
        ),
    )
    parser.add_argument(
        "--sampling-seeds",
        default="17",
        help="Used only when --prefix-order shuffled; retained in output metadata.",
    )

    # Shared community sweeps.
    parser.add_argument("--score-modes", default="raw_min")
    parser.add_argument("--edge-quantiles", default="0.90")
    parser.add_argument("--cohesion-modes", default="min")
    parser.add_argument("--cohesion-margin", type=float, default=0.0)
    parser.add_argument("--community-min-supports", default="2,3,4,5")
    parser.add_argument("--transition-min-supports", default="2,3,4,5")
    parser.add_argument("--minimum-answer-frequencies", default="2,3,5")
    parser.add_argument("--override-margins", default="0.0")
    parser.add_argument(
        "--metrics",
        default="selected",
        help=(
            "'selected' evaluates the curated train/test metric suite; "
            "'focused' evaluates the compact hypothesis-driven set; "
            "'default' evaluates every legacy and new metric; otherwise pass "
            "a comma-separated metric list."
        ),
    )
    parser.add_argument(
        "--save-recalculation-artifacts",
        action="store_true",
        help=(
            "Save all extracted reasoning units, sampled-trace selections and "
            "raw directional reranker pair scores as compressed CSV files. "
            "These files allow later metric/community recalculation without "
            "another model pass."
        ),
    )

    parser.add_argument("--max-rejections-per-answer", type=int, default=30)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--max-ids", type=int, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def self_test() -> None:
    """Small no-model test of trace exclusivity and bridge rejection."""
    units = [
        Unit(0, 0, "a", 0, 0.0, "A0", "A0"),
        Unit(1, 1, "a", 0, 0.0, "B0", "B0"),
        Unit(2, 2, "a", 0, 0.0, "C0", "C0"),
        Unit(3, 0, "a", 1, 1.0, "A1", "A1"),
    ]
    class FakeStore:
        values = {
            (0, 1): 0.95,
            (1, 2): 0.95,
            (0, 2): 0.10,
            (2, 3): 0.99,
            (0, 3): 0.99,
            (1, 3): 0.10,
        }
        def ensure(self, pairs, show_progress_bar=False):
            return None
        def score(self, left, right, mode):
            return self.values[(min(left, right), max(left, right))]
    result = build_constrained_communities(
        question_id=0,
        answer="a",
        units=units,
        unit_ids=[0, 1, 2, 3],
        proposal_edges=[(0, 1, 0.99), (1, 2, 0.98), (0, 3, 0.97)],
        score_store=FakeStore(),
        score_mode="sigmoid_mean",
        edge_threshold=0.90,
        cohesion_mode="min",
        cohesion_margin=0.0,
        min_trace_support=2,
        max_rejections=20,
    )
    assert len(result.communities) == 1
    assert set(result.communities[0].nodes) == {0, 1}
    assert result.diagnostics.get("reject_low_cohesion", 0) >= 1
    assert result.diagnostics.get("reject_trace_overlap", 0) >= 1

    # Order-aware sanity check: C0 precedes C2 in all three traces, directly
    # once and with an inserted retained checkpoint twice.
    sequence_units = [
        Unit(0, 0, "a", 0, 0.0, "S0", "S0"),
        Unit(1, 0, "a", 1, 1.0, "S2", "S2"),
        Unit(2, 1, "a", 0, 0.0, "S0", "S0"),
        Unit(3, 1, "a", 1, 0.5, "S1", "S1"),
        Unit(4, 1, "a", 2, 1.0, "S2", "S2"),
        Unit(5, 2, "a", 0, 0.0, "S0", "S0"),
        Unit(6, 2, "a", 1, 0.5, "S1", "S1"),
        Unit(7, 2, "a", 2, 1.0, "S2", "S2"),
    ]
    sequence_question = QuestionData(
        question_id=1,
        answers_by_trace=["a", "a", "a"],
        hits_by_trace=[True, True, True],
        units=sequence_units,
        unit_ids_by_trace={0: [0, 1], 1: [2, 3, 4], 2: [5, 6, 7]},
        traces_by_answer={"a": [0, 1, 2]},
        raw_count_by_answer={"a": 3},
        first_trace_by_answer={"a": 0},
        is_correct_by_answer={"a": True},
    )
    sequence_communities = [
        LocalCommunity(0, [0, 2, 5], [0, 1, 2], 3, 5, 5, 5, "C0", 0.0),
        LocalCommunity(1, [3, 6], [1, 2], 2, 5, 5, 5, "C1", 0.5),
        LocalCommunity(2, [1, 4, 7], [0, 1, 2], 3, 5, 5, 5, "C2", 1.0),
    ]
    sequence_mapping = {0: 0, 2: 0, 5: 0, 3: 1, 6: 1, 1: 2, 4: 2, 7: 2}
    order_values, _ = precedence_metrics(
        question=sequence_question,
        answer="a",
        trace_ids=[0, 1, 2],
        communities=sequence_communities,
        node_to_community=sequence_mapping,
        precedence_min_support=2,
        precedence_min_consistency=0.8,
    )
    assert order_values["precedence_reversal_rate"] == 0.0
    assert order_values["n_shortcut_surviving_pairs"] >= 1
    assert order_values["consensus_chain_length"] >= 3
    print("Self-test passed.")


def run_experiment(args: argparse.Namespace, experiment_mode: str) -> None:
    if args.self_test:
        self_test()
        return
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args.score_mode_values = parse_str_list(args.score_modes)
    invalid_modes = set(args.score_mode_values) - set(SCORE_MODES)
    if invalid_modes:
        raise ValueError(f"Invalid score modes: {sorted(invalid_modes)}")
    args.edge_quantile_values = parse_float_list(args.edge_quantiles)
    args.cohesion_mode_values = parse_str_list(args.cohesion_modes)
    invalid_cohesion = set(args.cohesion_mode_values) - set(COHESION_MODES)
    if invalid_cohesion:
        raise ValueError(f"Invalid cohesion modes: {sorted(invalid_cohesion)}")
    args.community_support_values = parse_int_list(args.community_min_supports)
    args.transition_support_values = parse_int_list(args.transition_min_supports)
    args.frequency_threshold_values = parse_int_list(args.minimum_answer_frequencies)
    args.override_margin_values = parse_float_list(args.override_margins)
    args.local_neighbor_k_values = parse_int_list(args.local_neighbor_k_values)
    args.sample_size_values = parse_int_list(args.sample_sizes)
    args.sampling_seed_values = parse_int_list(args.sampling_seeds)
    if not 0.5 <= args.precedence_min_consistency <= 1.0:
        raise ValueError("--precedence-min-consistency must be in [0.5, 1.0].")

    input_frame = pd.read_csv(args.input)
    required = {args.id_col, args.text_col, args.answer_col, args.hit_col}
    missing = required - set(input_frame.columns)
    if missing:
        raise ValueError(f"Input missing columns: {sorted(missing)}")

    category_by_id: dict[str, str] = {}
    if args.manifest:
        manifest = pd.read_csv(args.manifest)
        category_column = next(
            (column for column in ("experiment_category", "category", "selection_category", "selection_stratum", "stratum") if column in manifest),
            None,
        )
        if category_column is not None and args.id_col in manifest:
            category_by_id = {
                str(row[args.id_col]): str(row[category_column])
                for _, row in manifest.iterrows()
            }

    pipeline_path = Path(args.pipeline_script)
    if not pipeline_path.exists():
        candidate = Path(__file__).resolve().parent / args.pipeline_script
        pipeline_path = candidate if candidate.exists() else pipeline_path
    pipeline = load_pipeline(pipeline_path)

    local_files_only = not args.allow_model_downloads

    if args.local_cache_dir:
        local_cache_root = Path(os.path.expandvars(args.local_cache_dir)).expanduser()
        local_cache_root.mkdir(parents=True, exist_ok=True)
        args.embedding_cache_path = str(local_cache_root / Path(args.embedding_cache_path).name)
        args.reranker_cache_path = str(local_cache_root / Path(args.reranker_cache_path).name)
        print(f"Using local SQLite cache directory: {local_cache_root}", flush=True)

    args.embedding_cache_path = str(
        quarantine_invalid_sqlite(args.embedding_cache_path, label="embedding cache")
    )
    args.reranker_cache_path = str(
        quarantine_invalid_sqlite(args.reranker_cache_path, label="reranker cache")
    )

    reranker = CachedDirectionalReranker(
        model_name=args.reranker_model,
        hf_cache_dir=args.hf_cache_dir,
        cache_path=args.reranker_cache_path,
        prompt=args.reranker_instruction,
        batch_size=args.reranker_batch_size,
        max_length=args.reranker_max_length,
        device=args.device,
        local_files_only=local_files_only,
    )

    embedder = None
    if experiment_mode == "knn":
        embedding_snapshot = resolve_hf_snapshot(
            repo_id=args.embedding_model,
            cache_dir=args.hf_cache_dir,
            local_files_only=local_files_only,
        )
        embedder = pipeline.CachedSentenceTransformer(
            model_name=embedding_snapshot,
            cache_path=args.embedding_cache_path,
            batch_size=args.embedding_batch_size,
            device=args.device,
        )

    id_order = input_frame[args.id_col].drop_duplicates().tolist()
    if args.max_ids is not None:
        id_order = id_order[: args.max_ids]

    all_answers: list[dict[str, Any]] = []
    all_communities: list[dict[str, Any]] = []
    all_transitions: list[dict[str, Any]] = []
    all_precedence: list[dict[str, Any]] = []
    all_trace_schemas: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []
    all_misc: list[dict[str, Any]] = []
    all_units: list[dict[str, Any]] = []
    all_sampling: list[dict[str, Any]] = []
    all_pair_scores: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    def save_partial() -> None:
        if all_answers:
            atomic_write_csv(pd.DataFrame(all_answers), output_dir / "answer_metrics.csv")
        if all_communities:
            atomic_write_csv(pd.DataFrame(all_communities), output_dir / "community_details.csv")
        if all_transitions:
            atomic_write_csv(pd.DataFrame(all_transitions), output_dir / "transition_details.csv")
        if all_precedence:
            atomic_write_csv(pd.DataFrame(all_precedence), output_dir / "precedence_details.csv")
        if all_trace_schemas:
            atomic_write_csv(pd.DataFrame(all_trace_schemas), output_dir / "trace_checkpoint_schemas.csv")
        if all_diagnostics:
            atomic_write_csv(pd.DataFrame(all_diagnostics), output_dir / "stage_diagnostics.csv")
        if all_misc:
            misc = pd.DataFrame(all_misc)
            rejection = misc.loc[misc.get("reason", pd.Series(index=misc.index, dtype=object)).notna()] if "reason" in misc else pd.DataFrame()
            pair_rows = misc.loc[misc.get("left_trace", pd.Series(index=misc.index, dtype=float)).notna()] if "left_trace" in misc else pd.DataFrame()
            if not rejection.empty:
                atomic_write_csv(rejection, output_dir / "merge_rejections.csv")
            if not pair_rows.empty:
                atomic_write_csv(pair_rows, output_dir / "trace_pair_metrics.csv")
        if errors:
            atomic_write_csv(pd.DataFrame(errors), output_dir / "processing_errors.csv")

    try:
        for number, question_id in enumerate(
            tqdm(id_order, desc=f"{experiment_mode} pilot", total=len(id_order)), start=1
        ):
            try:
                question = prepare_question(
                    question_id=question_id,
                    frame=input_frame.loc[input_frame[args.id_col] == question_id],
                    pipeline=pipeline,
                    text_col=args.text_col,
                    answer_col=args.answer_col,
                    hit_col=args.hit_col,
                    min_unit_chars=args.min_unit_chars,
                    max_unit_chars=args.max_unit_chars,
                    max_units_per_trace=args.max_units_per_trace,
                    unit_context=args.unit_context,
                    minimum_unit_position=args.minimum_unit_position,
                )
                (
                    answers,
                    communities,
                    transitions,
                    precedence,
                    trace_schemas,
                    diagnostics,
                    misc,
                    units_artifact,
                    sampling_artifact,
                    pair_scores_artifact,
                ) = process_question(
                    question=question,
                    experiment_mode=experiment_mode,
                    pipeline=pipeline,
                    embedder=embedder,
                    reranker=reranker,
                    args=args,
                    category=category_by_id.get(str(question_id), ""),
                )
                all_answers.extend(answers)
                all_communities.extend(communities)
                all_transitions.extend(transitions)
                all_precedence.extend(precedence)
                all_trace_schemas.extend(trace_schemas)
                all_diagnostics.extend(diagnostics)
                all_misc.extend(misc)
                all_units.extend(units_artifact)
                all_sampling.extend(sampling_artifact)
                all_pair_scores.extend(pair_scores_artifact)
            except Exception as error:
                errors.append(
                    {
                        "id": question_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
                print(f"Failed id={question_id}: {type(error).__name__}: {error}")
                if args.fail_fast:
                    raise
            if args.save_every > 0 and number % args.save_every == 0:
                save_partial()

        save_partial()
        if args.save_recalculation_artifacts:
            if all_units:
                atomic_write_csv_gzip(
                    pd.DataFrame(all_units), output_dir / "reasoning_units.csv.gz"
                )
            if all_sampling:
                atomic_write_csv(
                    pd.DataFrame(all_sampling), output_dir / "sampling_manifest.csv"
                )
            if all_pair_scores:
                atomic_write_csv_gzip(
                    pd.DataFrame(all_pair_scores), output_dir / "reranker_pair_scores.csv.gz"
                )

        answer_frame = pd.DataFrame(all_answers)
        if answer_frame.empty:
            raise RuntimeError("No answer metrics produced.")

        baseline_key_columns = [
            "dataset_split", "experiment_mode", "sampling_seed", "prefix_order", "sample_size", "id"
        ]
        baseline_value_columns = [
            "category", "structural_trace_budget", "prefix_trace_ids_json",
            "sample_majority_answer", "sample_majority_vote_count",
            "sample_majority_n_traces", "sample_majority_is_correct",
            "budget_majority_answer", "budget_majority_vote_count",
            "budget_majority_n_traces", "budget_majority_is_correct",
            "full_majority_answer", "full_majority_vote_count",
            "full_majority_n_traces", "full_majority_is_correct",
        ]
        baseline_frame = (
            answer_frame[baseline_key_columns + baseline_value_columns]
            .drop_duplicates(subset=baseline_key_columns)
            .sort_values(baseline_key_columns, kind="mergesort")
        )
        atomic_write_csv(
            baseline_frame, output_dir / "sample_size_majority_baselines.csv"
        )
        baseline_accuracy = (
            baseline_frame.groupby(
                ["dataset_split", "experiment_mode", "sampling_seed", "prefix_order", "sample_size"],
                dropna=False,
                sort=False,
            )
            .agg(
                n_ids=("id", "nunique"),
                full_majority_correct=("full_majority_is_correct", "sum"),
                full_majority_accuracy=("full_majority_is_correct", "mean"),
                sample_majority_correct=("sample_majority_is_correct", "sum"),
                sample_majority_accuracy=("sample_majority_is_correct", "mean"),
                budget_majority_correct=("budget_majority_is_correct", "sum"),
                budget_majority_accuracy=("budget_majority_is_correct", "mean"),
                mean_structural_trace_budget=("structural_trace_budget", "mean"),
                median_structural_trace_budget=("structural_trace_budget", "median"),
            )
            .reset_index()
            .sort_values(["sampling_seed", "prefix_order", "sample_size"], kind="mergesort")
        )
        atomic_write_csv(
            baseline_accuracy, output_dir / "sample_size_baseline_accuracy.csv"
        )

        if args.metrics == "selected":
            metrics = list(SELECTED_METRICS)
        elif args.metrics == "focused":
            metrics = list(FOCUSED_METRICS)
        elif args.metrics == "default":
            metrics = list(DEFAULT_METRICS)
        else:
            metrics = parse_str_list(args.metrics)
        missing_metrics = [metric for metric in metrics if metric not in answer_frame]
        if missing_metrics:
            raise ValueError(f"Missing requested metrics: {missing_metrics}")
        predictions, accuracy, overrides, override_accuracy = evaluate_answer_metrics(
            answer_frame=answer_frame,
            metrics=metrics,
            frequency_thresholds=args.frequency_threshold_values,
            override_margins=args.override_margin_values,
        )
        atomic_write_csv(predictions, output_dir / "metric_predictions.csv")
        atomic_write_csv(accuracy, output_dir / "metric_accuracy.csv")
        atomic_write_csv(
            accuracy.sort_values(
                ["sampling_seed", "prefix_order", "sample_size", "metric", "accuracy"],
                ascending=[True, True, True, False],
                kind="mergesort",
            ),
            output_dir / "metric_vs_majority_by_sample_size.csv",
        )
        atomic_write_csv(overrides, output_dir / "override_predictions.csv")
        atomic_write_csv(override_accuracy, output_dir / "override_accuracy.csv")
        atomic_write_csv(accuracy.head(100), output_dir / "best_metric_configs.csv")

        metadata = {
            "dataset_split": args.dataset_split,
            "experiment_mode": experiment_mode,
            "input": args.input,
            "manifest": args.manifest,
            "pipeline_script": str(pipeline_path),
            "n_ids": len(id_order),
            "embedding_model": args.embedding_model if experiment_mode == "knn" else None,
            "embedding_cache_path": args.embedding_cache_path if experiment_mode == "knn" else None,
            "reranker_cache_path": args.reranker_cache_path,
            "reranker_model": args.reranker_model,
            "selected_metrics": list(SELECTED_METRICS),
            "metrics_evaluated": metrics,
            "save_recalculation_artifacts": bool(args.save_recalculation_artifacts),
            "score_modes": args.score_mode_values,
            "edge_quantiles": args.edge_quantile_values,
            "cohesion_modes": args.cohesion_mode_values,
            "community_min_supports": args.community_support_values,
            "transition_min_supports": args.transition_support_values,
            "embedding_candidate_k": args.embedding_candidate_k if experiment_mode == "knn" else None,
            "local_neighbor_k_values": args.local_neighbor_k_values if experiment_mode == "knn" else [0],
            "sample_sizes": args.sample_size_values if experiment_mode == "sampled_allpairs" else None,
            "sample_size_majority_baselines": {
                "sample_majority": "majority on the exact same k-trace question prefix used for structural analysis",
                "budget_majority": "backward-compatible alias of sample_majority in prefix mode",
                "full_majority": "majority using all available traces",
            },
            "sampling_seeds": args.sampling_seed_values if experiment_mode == "sampled_allpairs" else None,
            "prefix_order": args.prefix_order if experiment_mode == "sampled_allpairs" else None,
            "sample_size_semantics": "number of traces per question prefix",
            "four_community_rules": [
                "answer_local_scope",
                "one_step_per_trace",
                "minimum_distinct_trace_support",
                "complete_linkage_like_cohesion",
            ],
            "new_transition_metrics": [
                "supported_pairwise_transition_jaccard_distance",
                "coverage_weighted_transition_set_diversity",
                "support_weighted_branching_fraction",
            ],
            "ordered_backbone_metrics": [
                "supported_precedence_mass",
                "precedence_reversal_rate",
                "shortcut_survival_mass",
                "consensus_chain_length",
                "atomic_ordered_backbone_mass",
            ],
            "recurrence_length_metrics": [
                "mean_checkpoint_schema_length",
                "deep_recurrent_checkpoint_mass",
                "pair_fragmentation_ratio",
                "mean_checkpoint_words",
                "atomic_recurrent_checkpoint_mass",
            ],
            "precedence_min_consistency": args.precedence_min_consistency,
            "reranker_new_pairs": reranker.n_new_pairs,
            "reranker_cached_pairs": reranker.n_cached_pairs,
            "n_errors": len(errors),
        }
        (output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        metric_dictionary = pd.DataFrame([
            {"metric": "n_local_communities", "direction": "max", "family": "checkpoint_richness", "description": "Number of retained answer-local checkpoint communities."},
            {"metric": "deep_recurrent_checkpoint_mass", "direction": "max", "family": "checkpoint_recurrence", "description": "Sum of squared community support fractions; rewards broad, deeply recurrent checkpoint structure."},
            {"metric": "n_shared_transition_types", "direction": "max", "family": "transition_richness", "description": "Number of exact adjacent transitions recurring in enough sampled traces."},
            {"metric": "mean_pairwise_transition_jaccard_distance", "direction": "max", "family": "partial_route_heterogeneity", "description": "Legacy pairwise transition-set Jaccard distance; preserves empty-vs-nonempty behavior for comparison."},
            {"metric": "supported_pairwise_transition_jaccard_distance", "direction": "max", "family": "supported_route_diversity", "description": "Transition-set diversity credited only when both traces contain qualifying transitions."},
            {"metric": "coverage_weighted_transition_set_diversity", "direction": "max", "family": "route_signature_diversity", "description": "Transition-set signature diversity weighted by trace coverage."},
            {"metric": "max_transition_out_degree", "direction": "max", "family": "branching", "description": "Maximum number of recurrent continuations from one checkpoint."},
            {"metric": "precedence_reversal_rate", "direction": "min", "family": "order_consistency", "description": "Support-weighted rate at which co-occurring checkpoint pairs reverse order across traces."},
            {"metric": "supported_precedence_mass", "direction": "max", "family": "ordered_backbone", "description": "Mass of recurrent checkpoint precedence relations weighted by support and direction consistency."},
            {"metric": "shortcut_survival_mass", "direction": "max", "family": "shortcut_survival", "description": "Stable precedence relations adjacent in some traces and separated by inserted checkpoints in others."},
            {"metric": "consensus_chain_length", "direction": "max", "family": "ordered_backbone", "description": "Longest acyclic chain in the stable checkpoint-precedence graph."},
            {"metric": "mean_checkpoint_words", "direction": "min", "family": "checkpoint_atomicity", "description": "Mean whitespace-token length of retained checkpoint steps."},
            {"metric": "atomic_ordered_backbone_mass", "direction": "max", "family": "ordered_backbone_atomicity", "description": "Supported precedence mass normalized by the square root of mean checkpoint length."},
        ])
        atomic_write_csv(metric_dictionary, output_dir / "metric_dictionary.csv")
        print("\nBEST METRIC CONFIGURATIONS")
        print(accuracy.head(20).to_string(index=False))
        print(f"\nSaved to {output_dir.resolve()}")
    finally:
        reranker.close()
        if embedder is not None:
            embedder.close()


def main() -> None:
    parser = build_parser("sampled_allpairs")
    args = parser.parse_args()
    run_experiment(args, "sampled_allpairs")


if __name__ == "__main__":
    main()
