#!/usr/bin/env python3
"""
Punctuation-delimited checkpoint-route convergence voting over reasoning traces.

This script implements two related, verifier-free hypotheses.

SIMPLE PAIRWISE VERSION
-----------------------
For a one-to-one matched sentence pair (u, v) from two traces:

    checkpoint_similarity = similarity(sentence_u, sentence_v)
    prefix_similarity     = similarity(prefix_before_u, prefix_before_v)

    independent_mass = checkpoint_similarity * (1 - prefix_similarity)
    repeated_mass    = checkpoint_similarity * prefix_similarity

A pair of traces has high independent convergence when it reaches similar
intermediate conclusions from dissimilar prior reasoning. It has high repeated
path mass when both the checkpoint and the route leading to it are similar.

STRONG CHECKPOINT-COMMUNITY VERSION
-----------------------------------
1. Construct a mutual-kNN graph over sentence units from different traces.
2. Detect weighted checkpoint communities with greedy modularity.
3. For each answer and checkpoint, compute:
       - within-answer prevalence;
       - prevalence outside the answer;
       - contrastive checkpoint specificity;
       - effective rank of prefix embeddings leading to the checkpoint.
4. Aggregate answer-specific checkpoint evidence and parent-prefix diversity.

Sentence extraction ignores explicit Step 1/2 grouping: every detected sentence is a separate unit.

Both versions optionally use size-matched random trace groups to calibrate their
scores as z-scores. Several coefficient-free selectors are evaluated against
majority voting, including geometric rank products of frequency and structure.

The script reuses a persistent SQLite embedding cache and accepts arbitrary
SentenceTransformers-compatible Hugging Face model IDs.

Example
-------
python checkpoint_route_convergence_voting.py \
    --input math500_test_expanded.csv \
    --output-dir checkpoint_route_run \
    --cache-path shared_matrix_embeddings.sqlite \
    --models tbs17/MathBERT \
    --n-random-simple 200 \
    --n-random-checkpoint 50

Outputs
-------
OUTPUT_DIR/
  run_config.json
  combined_question_results.csv
  combined_selector_summary.csv
  MODEL_SLUG/
    question_results.csv
    answer_scores.csv
    selector_summary.csv
    steps.csv
    pairwise_convergence.csv
    checkpoint_communities.csv
    checkpoint_answer_stats.csv
    changed_cases.csv
    recoveries.csv
    harms.csv
    wrong_to_wrong.csv
    matrices/*.npz
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

try:
    import networkx as nx
except ImportError as exc:
    raise ImportError("networkx is required: pip install networkx") from exc

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    models: tuple[str, ...] = ("tbs17/MathBERT",)
    text_col: str = "completion"
    id_col: str = "id"
    answer_col: str = "final_answer_new"
    hit_col: str = "row_hit"
    batch_size: int = 64
    device: str | None = None

    unit_segmentation: str = "sentence"
    min_unit_chars: int = 1
    max_unit_chars: int = 900
    max_units_per_trace: int = 128
    unit_context: str = "current"  # current | previous_current

    embedding_preprocess: str = "none"  # none | center | remove_pc
    shared_pc_count: int = 1

    commonness_stat: str = "median"  # median | mean | quantile
    commonness_quantile: float = 0.50
    commonness_pair: str = "mean"  # mean | max | min
    residualization: str = "subtract"  # none | subtract | relative

    min_prefix_steps: int = 1
    prefix_similarity: str = "positive_cosine"  # positive_cosine | scaled_cosine

    checkpoint_knn: int = 5
    checkpoint_edge_mode: str = "mutual"  # mutual | union
    checkpoint_min_similarity: float = 0.0
    checkpoint_min_trace_support: int = 2
    checkpoint_resolution: float = 1.0

    n_random_simple: int = 200
    n_random_checkpoint: int = 50
    random_batch_size: int = 256
    checkpoint_random_max_count: int = 15  # 0 = all sizes
    seed: int = 42

    save_pairwise_top_k: int = 100
    matrix_compression: bool = True


SELECTORS = (
    "simple_raw_advantage",
    "simple_z_convergence",
    "simple_count_rank_product",
    "checkpoint_raw_independent",
    "checkpoint_z_independent",
    "checkpoint_count_rank_product",
    "combined_count_rank_product",
    "majority_tiebreak_combined",
)


# ---------------------------------------------------------------------------
# Persistent embedding cache
# ---------------------------------------------------------------------------


class SQLiteEmbeddingCache:
    def __init__(self, path: str | Path, model_name: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                model_name TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                text TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY (model_name, text_hash)
            )
            """
        )
        self.conn.commit()

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_many(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        unique_texts = list(dict.fromkeys(texts))
        output: dict[str, np.ndarray] = {}
        if not unique_texts:
            return output

        hashes = [self._hash(text) for text in unique_texts]
        for start in range(0, len(hashes), 800):
            batch = hashes[start:start + 800]
            placeholders = ",".join("?" for _ in batch)
            rows = self.conn.execute(
                f"SELECT text_hash, text, dim, vector FROM embeddings "
                f"WHERE model_name = ? AND text_hash IN ({placeholders})",
                [self.model_name, *batch],
            ).fetchall()
            for text_hash, text, dim, blob in rows:
                if self._hash(text) != text_hash:
                    continue
                output[text] = np.frombuffer(
                    blob, dtype=np.float32, count=int(dim)
                ).copy()
        return output

    def put_many(self, items: Sequence[tuple[str, np.ndarray]]) -> None:
        rows = []
        for text, vector in items:
            vector = np.asarray(vector, dtype=np.float32)
            rows.append(
                (
                    self.model_name,
                    self._hash(text),
                    text,
                    int(vector.shape[0]),
                    sqlite3.Binary(vector.tobytes()),
                )
            )
        if rows:
            self.conn.executemany(
                "INSERT OR REPLACE INTO embeddings "
                "(model_name, text_hash, text, dim, vector) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            self.conn.commit()

    def close(self) -> None:
        self.conn.close()


class CachedSentenceTransformer:
    def __init__(
        self,
        model_name: str,
        cache_path: str | Path,
        batch_size: int,
        device: str | None,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self.cache = SQLiteEmbeddingCache(cache_path, model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        cached = self.cache.get_many(texts)
        missing = list(dict.fromkeys(text for text in texts if text not in cached))
        if missing:
            vectors = self.model.encode(
                missing,
                batch_size=self.batch_size,
                show_progress_bar=len(missing) >= self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            self.cache.put_many(list(zip(missing, vectors)))
            cached.update({text: vector for text, vector in zip(missing, vectors)})

        matrix = np.stack([cached[text] for text in texts]).astype(np.float32)
        return l2_normalize(matrix)

    def close(self) -> None:
        self.cache.close()


# ---------------------------------------------------------------------------
# Reasoning-unit extraction
# ---------------------------------------------------------------------------


FINAL_MARKER_RE = re.compile(
    r"(?im)^\s*(?:final\s+answer|the\s+answer\s+is|answer)\s*[:：]"
)
STEP_LABEL_RE = re.compile(
    r"(?im)^\s*(?:step\s*\d+|\d+[\.)])\s*[:：.-]?\s*"
)
SENTENCE_BOUNDARY_RE = re.compile(
    r"(?<=[.!?])(?:[\"'”’\)\]]*)\s+(?=[A-Z0-9\\$\(\[])"
)


def clean_text(text: object) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    value = str(text).replace("\r\n", "\n").replace("\r", "\n")
    marker = FINAL_MARKER_RE.search(value)
    if marker:
        value = value[:marker.start()]
    return re.sub(r"[ \t]+", " ", value).strip()


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    """
    Keep a sentence intact whenever possible. Extremely long sentences are
    chunked only to respect the encoder input safety limit.
    """
    sentence = re.sub(r"\s+", " ", sentence).strip()
    if not sentence:
        return []
    if max_chars <= 0 or len(sentence) <= max_chars:
        return [sentence]

    parts: list[str] = []
    remaining = sentence
    while len(remaining) > max_chars:
        boundary = remaining.rfind(" ", 0, max_chars + 1)
        if boundary <= 0:
            boundary = max_chars
        parts.append(remaining[:boundary].strip())
        remaining = remaining[boundary:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def _limit_units(units: list[str], maximum: int) -> list[str]:
    if maximum <= 0 or len(units) <= maximum:
        return units
    indices = np.linspace(0, len(units) - 1, num=maximum, dtype=int)
    return [units[index] for index in sorted(set(indices.tolist()))]


def extract_reasoning_units(
    text: object,
    min_chars: int,
    max_chars: int,
    max_units: int,
) -> list[str]:
    """
    Extract units using only explicit sentence-ending punctuation.

    A boundary is created only after `.`, `!`, or `?`.

    Newlines, colons, and numbered step labels are treated as formatting and
    converted to spaces. Equation lines remain attached to the surrounding
    explanation unless they themselves end with sentence punctuation.
    """
    body = clean_text(text)
    if not body:
        return []

    body = STEP_LABEL_RE.sub("", body)
    body = re.sub(r"\s+", " ", body).strip()

    if not body:
        return []

    raw_sentences = SENTENCE_BOUNDARY_RE.split(body)

    units: list[str] = []
    for raw_sentence in raw_sentences:
        sentence = re.sub(r"\s+", " ", raw_sentence).strip()
        sentence = STEP_LABEL_RE.sub("", sentence).strip()

        if not sentence:
            continue

        meaningful = re.sub(r"[^A-Za-z0-9]+", "", sentence)
        if not meaningful:
            continue

        units.extend(
            _split_long_sentence(
                sentence,
                max_chars=max_chars,
            )
        )

    return _limit_units(units, max_units)


def make_embedding_unit(units: Sequence[str], index: int, mode: str) -> str:
    if mode == "current":
        return units[index]
    if mode == "previous_current":
        if index == 0:
            return units[index]
        return f"Previous: {units[index - 1]}\nCurrent: {units[index]}"
    raise ValueError(f"Unsupported unit_context={mode!r}")


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------


def normalize_answer(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    value = re.sub(r"\s+", " ", str(value)).strip()
    return value or None


def parse_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f", ""}:
        return False
    raise ValueError(f"Cannot parse bool: {value!r}")


def model_slug(model_name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "__", model_name).strip("_")
    return value or hashlib.sha1(model_name.encode()).hexdigest()[:10]


def question_file_stem(question_key: str) -> str:
    digest = hashlib.sha1(question_key.encode("utf-8")).hexdigest()[:12]
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", question_key)[-60:]
    return f"{readable}__{digest}"


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def preprocess_embeddings(matrix: np.ndarray, mode: str, pc_count: int) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float32).copy()
    if x.size == 0 or mode == "none":
        return l2_normalize(x)
    x -= x.mean(axis=0, keepdims=True)
    if mode == "center":
        return l2_normalize(x)
    if mode != "remove_pc":
        raise ValueError(mode)
    k = max(0, min(int(pc_count), min(x.shape) - 1))
    if k > 0 and np.any(np.abs(x) > 1e-12):
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        pcs = vt[:k]
        x -= (x @ pcs.T) @ pcs
    return l2_normalize(x)


def scale_cosine(cosine: np.ndarray) -> np.ndarray:
    return np.clip((cosine + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)


def statistic(values: Sequence[float], mode: str, quantile: float) -> float:
    array = np.asarray(values, dtype=np.float32)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0
    if mode == "median":
        return float(np.median(array))
    if mode == "mean":
        return float(array.mean())
    if mode == "quantile":
        return float(np.quantile(array, quantile))
    raise ValueError(mode)


def effective_rank(kernel: np.ndarray) -> float:
    n = kernel.shape[0]
    if n == 0:
        return float("nan")
    symmetric = (kernel + kernel.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(symmetric)
    positive = np.clip(eigenvalues, 0.0, None)
    total = float(positive.sum())
    squares = float(np.square(positive).sum())
    if total <= 1e-12 or squares <= 1e-12:
        return 1.0
    return float((total * total) / squares)


def normal_cdf(z: float) -> float:
    if not np.isfinite(z):
        return float("nan")
    return float(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def calibrate(observed: float, null_values: np.ndarray) -> dict[str, float]:
    values = np.asarray(null_values, dtype=float)
    values = values[np.isfinite(values)]
    if not np.isfinite(observed) or values.size == 0:
        return {"mean": np.nan, "std": np.nan, "z": np.nan, "percentile": np.nan}
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    if std > 0:
        z = float((observed - mean) / std)
    else:
        z = 0.0 if np.isclose(observed, mean) else (
            float("inf") if observed > mean else float("-inf")
        )
    percentile = float((np.sum(values <= observed) + 1) / (values.size + 1))
    return {"mean": mean, "std": std, "z": z, "percentile": percentile}


def first_occurrence_majority(answers: Sequence[str]) -> tuple[str, int, list[str]]:
    counts = Counter(answers)
    maximum = max(counts.values())
    candidates = []
    seen = set()
    for answer in answers:
        if counts[answer] == maximum and answer not in seen:
            seen.add(answer)
            candidates.append(answer)
    return candidates[0], maximum, candidates


def answer_correctness(answers: Sequence[str], hits: Sequence[bool]) -> dict[str, bool]:
    output: dict[str, bool] = {}
    for answer, hit in zip(answers, hits):
        output[answer] = output.get(answer, False) or bool(hit)
    return output


def dense_rank_desc(values: pd.Series) -> pd.Series:
    return values.rank(method="min", ascending=False, na_option="bottom")


# ---------------------------------------------------------------------------
# Unit commonness and residualization
# ---------------------------------------------------------------------------


def compute_unit_commonness(
    similarity: np.ndarray,
    trace_indices: Sequence[np.ndarray],
    unit_trace: np.ndarray,
    config: Config,
) -> np.ndarray:
    output = np.zeros(similarity.shape[0], dtype=np.float32)
    for unit in range(similarity.shape[0]):
        own_trace = int(unit_trace[unit])
        per_trace = []
        for trace, indices in enumerate(trace_indices):
            if trace == own_trace or indices.size == 0:
                continue
            per_trace.append(float(similarity[unit, indices].max()))
        output[unit] = statistic(
            per_trace,
            config.commonness_stat,
            config.commonness_quantile,
        )
    return output


def pair_baseline(source: np.ndarray, target: np.ndarray, mode: str) -> np.ndarray:
    a = np.asarray(source, dtype=np.float32)[:, None]
    b = np.asarray(target, dtype=np.float32)[None, :]
    if mode == "mean":
        return 0.5 * (a + b)
    if mode == "max":
        return np.maximum(a, b)
    if mode == "min":
        return np.minimum(a, b)
    raise ValueError(mode)


def residualize(
    block: np.ndarray,
    source_commonness: np.ndarray,
    target_commonness: np.ndarray,
    config: Config,
) -> np.ndarray:
    baseline = pair_baseline(source_commonness, target_commonness, config.commonness_pair)
    if config.residualization == "none":
        return block.astype(np.float32)
    if config.residualization == "subtract":
        return np.clip(block - baseline, 0.0, 1.0).astype(np.float32)
    if config.residualization == "relative":
        return np.clip(
            (block - baseline) / np.clip(1.0 - baseline, 1e-8, None),
            0.0,
            1.0,
        ).astype(np.float32)
    raise ValueError(config.residualization)


# ---------------------------------------------------------------------------
# Simple pairwise convergence-gap method
# ---------------------------------------------------------------------------


def one_to_one_alignment(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if block.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    return linear_sum_assignment(-block)


def normalized_matching_score(block: np.ndarray, normalization: str) -> float:
    if block.size == 0:
        return 0.0
    rows, cols = one_to_one_alignment(block)
    total = float(block[rows, cols].sum())
    m, n = block.shape
    if normalization == "max":
        denominator = max(m, n)
    elif normalization == "min":
        denominator = min(m, n)
    elif normalization == "mean":
        denominator = 0.5 * (m + n)
    elif normalization == "matches":
        denominator = len(rows)
    else:
        raise ValueError(normalization)
    return total / max(float(denominator), 1.0)


def compute_prefix_embeddings(
    embeddings: np.ndarray,
    trace_indices: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Mean embedding of all units before each unit, cached once per question."""
    prefix = np.zeros_like(embeddings, dtype=np.float32)
    valid = np.zeros(embeddings.shape[0], dtype=bool)
    for indices in trace_indices:
        if indices.size <= 1:
            continue
        cumulative = np.cumsum(embeddings[indices], axis=0)
        for local in range(1, len(indices)):
            vector = cumulative[local - 1] / local
            norm = float(np.linalg.norm(vector))
            if norm > 1e-12:
                prefix[indices[local]] = vector / norm
                valid[indices[local]] = True
    return prefix, valid


def prefix_similarity_value(
    a: np.ndarray,
    b: np.ndarray,
    mode: str,
) -> float:
    cosine = float(np.dot(a, b))
    if mode == "positive_cosine":
        return float(np.clip(cosine, 0.0, 1.0))
    if mode == "scaled_cosine":
        return float(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0))
    raise ValueError(mode)


def pair_convergence_metrics(
    block: np.ndarray,
    global_i: np.ndarray,
    global_j: np.ndarray,
    prefix_embeddings: np.ndarray,
    prefix_valid: np.ndarray,
    min_prefix_steps: int,
    prefix_similarity_mode: str,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Compute convergence/repetition for one pair of traces."""
    if block.size == 0:
        empty = {
            "shared_checkpoint_mass": 0.0,
            "independent_mass": 0.0,
            "repeated_mass": 0.0,
            "independent_fraction": np.nan,
            "advantage": 0.0,
            "n_checkpoint_matches": 0,
        }
        return empty, []

    rows, cols = one_to_one_alignment(block)
    denominator = max(block.shape)
    matched_rows = []
    independent_total = 0.0
    repeated_total = 0.0
    checkpoint_total = 0.0

    for u, v in zip(rows.tolist(), cols.tolist()):
        checkpoint_similarity = float(block[u, v])
        if checkpoint_similarity <= 0:
            continue
        if u < min_prefix_steps or v < min_prefix_steps:
            continue

        unit_i = int(global_i[u])
        unit_j = int(global_j[v])
        if not prefix_valid[unit_i] or not prefix_valid[unit_j]:
            continue

        prefix_similarity = prefix_similarity_value(
            prefix_embeddings[unit_i],
            prefix_embeddings[unit_j],
            prefix_similarity_mode,
        )
        independent = checkpoint_similarity * (1.0 - prefix_similarity)
        repeated = checkpoint_similarity * prefix_similarity

        checkpoint_total += checkpoint_similarity
        independent_total += independent
        repeated_total += repeated
        matched_rows.append(
            {
                "source_unit": int(u),
                "target_unit": int(v),
                "checkpoint_similarity": checkpoint_similarity,
                "prefix_similarity": prefix_similarity,
                "independent_contribution": independent,
                "repeated_contribution": repeated,
            }
        )

    independent_fraction = (
        independent_total / checkpoint_total if checkpoint_total > 0 else np.nan
    )
    metrics = {
        "shared_checkpoint_mass": checkpoint_total / max(float(denominator), 1.0),
        "independent_mass": independent_total / max(float(denominator), 1.0),
        "repeated_mass": repeated_total / max(float(denominator), 1.0),
        "independent_fraction": independent_fraction,
        "advantage": (independent_total - repeated_total) / max(float(denominator), 1.0),
        "n_checkpoint_matches": len(matched_rows),
    }
    return metrics, matched_rows


def build_pairwise_convergence_matrices(
    unit_similarity: np.ndarray,
    trace_indices: Sequence[np.ndarray],
    commonness: np.ndarray,
    prefix_embeddings: np.ndarray,
    prefix_valid: np.ndarray,
    config: Config,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    n = len(trace_indices)
    matrices = {
        "shared_checkpoint_mass": np.zeros((n, n), dtype=np.float32),
        "independent_mass": np.zeros((n, n), dtype=np.float32),
        "repeated_mass": np.zeros((n, n), dtype=np.float32),
        "independent_fraction": np.full((n, n), np.nan, dtype=np.float32),
        "advantage": np.zeros((n, n), dtype=np.float32),
        "n_checkpoint_matches": np.zeros((n, n), dtype=np.float32),
    }
    np.fill_diagonal(matrices["independent_fraction"], 0.0)
    detail_rows: list[dict[str, Any]] = []

    for i in range(n):
        idx_i = trace_indices[i]
        for j in range(i + 1, n):
            idx_j = trace_indices[j]
            if idx_i.size == 0 or idx_j.size == 0:
                adjusted = np.empty((len(idx_i), len(idx_j)), dtype=np.float32)
            else:
                raw = unit_similarity[np.ix_(idx_i, idx_j)]
                adjusted = residualize(
                    raw,
                    commonness[idx_i],
                    commonness[idx_j],
                    config,
                )
            metrics, details = pair_convergence_metrics(
                adjusted,
                idx_i,
                idx_j,
                prefix_embeddings,
                prefix_valid,
                config.min_prefix_steps,
                config.prefix_similarity,
            )

            for name, value in metrics.items():
                matrices[name][i, j] = matrices[name][j, i] = value
            for row in details:
                detail_rows.append({"trace_i": i, "trace_j": j, **row})

    return matrices, detail_rows

def group_pair_mean(matrix: np.ndarray, indices: np.ndarray) -> float:
    indices = np.asarray(indices, dtype=int)
    if len(indices) < 2:
        return np.nan
    sub = matrix[np.ix_(indices, indices)]
    return float(sub[np.triu_indices(len(indices), k=1)].mean())


def random_group_pair_means(
    matrix: np.ndarray,
    group_size: int,
    n_random: int,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = matrix.shape[0]
    output = np.full(n_random, np.nan, dtype=np.float64)
    if group_size < 2:
        return output

    start = 0
    while start < n_random:
        batch = min(batch_size, n_random - start)
        random_values = rng.random((batch, n))
        sampled = np.argpartition(
            random_values,
            kth=group_size - 1,
            axis=1,
        )[:, :group_size]
        membership = np.zeros((batch, n), dtype=np.float64)
        membership[np.arange(batch)[:, None], sampled] = 1.0
        total = np.einsum("bi,ij,bj->b", membership, matrix, membership, optimize=True)
        diagonal = membership @ np.diag(matrix)
        output[start:start + batch] = (total - diagonal) / (
            group_size * (group_size - 1)
        )
        start += batch
    return output


# ---------------------------------------------------------------------------
# Strong checkpoint-community method
# ---------------------------------------------------------------------------


def build_checkpoint_communities(
    adjusted_similarity: np.ndarray,
    unit_trace: np.ndarray,
    config: Config,
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    n_units = adjusted_similarity.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(n_units))

    directed_neighbors: list[set[int]] = []
    for unit in range(n_units):
        candidates = np.flatnonzero(unit_trace != unit_trace[unit])
        if candidates.size == 0:
            directed_neighbors.append(set())
            continue
        scores = adjusted_similarity[unit, candidates]
        k = min(max(config.checkpoint_knn, 1), candidates.size)
        top_local = np.argpartition(-scores, kth=k - 1)[:k]
        selected = {
            int(candidates[index])
            for index in top_local
            if float(scores[index]) >= config.checkpoint_min_similarity
            and float(scores[index]) > 0
        }
        directed_neighbors.append(selected)

    for i in range(n_units):
        for j in directed_neighbors[i]:
            if i >= j:
                continue
            if config.checkpoint_edge_mode == "mutual":
                keep = i in directed_neighbors[j]
            elif config.checkpoint_edge_mode == "union":
                keep = True
            else:
                raise ValueError(config.checkpoint_edge_mode)
            if keep:
                graph.add_edge(i, j, weight=float(adjusted_similarity[i, j]))

    if graph.number_of_edges() == 0:
        communities = [{node} for node in graph.nodes]
    else:
        communities = list(
            nx.algorithms.community.greedy_modularity_communities(
                graph,
                weight="weight",
                resolution=config.checkpoint_resolution,
            )
        )

    retained: list[list[int]] = []
    community_rows: list[dict[str, Any]] = []
    for community in communities:
        nodes = sorted(int(node) for node in community)
        trace_support = len(set(unit_trace[nodes].tolist()))
        if trace_support < config.checkpoint_min_trace_support:
            continue
        retained.append(nodes)
        community_rows.append(
            {
                "checkpoint_id": len(retained) - 1,
                "n_units": len(nodes),
                "trace_support": trace_support,
            }
        )
    return retained, community_rows


def choose_one_occurrence_per_trace(
    nodes: Sequence[int],
    adjusted_similarity: np.ndarray,
    unit_trace: np.ndarray,
) -> list[int]:
    nodes = list(nodes)
    if len(nodes) == 1:
        return nodes
    sub = adjusted_similarity[np.ix_(nodes, nodes)]
    centrality = sub.mean(axis=1)
    by_trace: dict[int, list[int]] = {}
    for local, node in enumerate(nodes):
        by_trace.setdefault(int(unit_trace[node]), []).append(local)
    selected = []
    for locals_ in by_trace.values():
        best_local = max(locals_, key=lambda index: float(centrality[index]))
        selected.append(nodes[best_local])
    return selected


def prefix_embedding_for_unit(
    unit: int,
    embeddings: np.ndarray,
    trace_indices: Sequence[np.ndarray],
    unit_trace: np.ndarray,
) -> np.ndarray | None:
    trace = int(unit_trace[unit])
    indices = trace_indices[trace]
    positions = np.flatnonzero(indices == unit)
    if positions.size == 0:
        return None
    local = int(positions[0])
    if local <= 0:
        return None
    prefix = embeddings[indices[:local]]
    vector = prefix.mean(axis=0)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return None
    return (vector / norm).astype(np.float32)


def prepare_checkpoint_objects(
    communities: Sequence[Sequence[int]],
    adjusted_similarity: np.ndarray,
    embeddings: np.ndarray,
    trace_indices: Sequence[np.ndarray],
    unit_trace: np.ndarray,
    unit_index: np.ndarray,
    unit_position: np.ndarray,
    unit_text: Sequence[str],
    answers: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    objects = []
    rows = []

    for checkpoint_id, nodes in enumerate(communities):
        selected = choose_one_occurrence_per_trace(
            nodes,
            adjusted_similarity,
            unit_trace,
        )
        sub = adjusted_similarity[np.ix_(nodes, nodes)]
        medoid_local = int(np.argmax(sub.mean(axis=1)))
        representative = int(nodes[medoid_local])

        occurrences = []
        answer_counts: Counter[str] = Counter()
        for node in selected:
            trace = int(unit_trace[node])
            prefix = prefix_embedding_for_unit(
                node,
                embeddings,
                trace_indices,
                unit_trace,
            )
            occurrences.append(
                {
                    "trace": trace,
                    "unit": int(node),
                    "prefix": prefix,
                }
            )
            answer_counts[answers[trace]] += 1

        objects.append(
            {
                "checkpoint_id": checkpoint_id,
                "nodes": list(nodes),
                "occurrences": occurrences,
                "representative_unit": representative,
            }
        )
        rows.append(
            {
                "checkpoint_id": checkpoint_id,
                "n_units": len(nodes),
                "trace_support": len(selected),
                "representative_trace": int(unit_trace[representative]),
                "representative_unit_index": int(unit_index[representative]),
                "representative_position": float(unit_position[representative]),
                "representative_text": unit_text[representative],
                "mean_position": float(np.mean(unit_position[list(nodes)])),
                "answer_support_json": json.dumps(answer_counts, ensure_ascii=False),
            }
        )

    return objects, rows


def checkpoint_group_metrics(
    group_indices: np.ndarray,
    checkpoint_objects: Sequence[dict[str, Any]],
    n_traces: int,
    return_details: bool = False,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    group = set(int(index) for index in group_indices)
    n_group = len(group)
    outside_size = n_traces - n_group

    evidence_mass = 0.0
    weighted_parent_diversity = 0.0
    n_usable = 0
    details: list[dict[str, Any]] = []

    for checkpoint in checkpoint_objects:
        occurrences = checkpoint["occurrences"]
        inside = [occ for occ in occurrences if occ["trace"] in group]
        outside_count = sum(occ["trace"] not in group for occ in occurrences)
        n_inside = len(inside)
        if n_inside < 2:
            continue

        within_prevalence = n_inside / max(n_group, 1)
        outside_prevalence = (
            outside_count / outside_size if outside_size > 0 else 0.0
        )
        contrast = max(0.0, within_prevalence - outside_prevalence)
        if contrast <= 0:
            continue

        prefixes = [occ["prefix"] for occ in inside if occ["prefix"] is not None]
        if len(prefixes) >= 2:
            prefix_matrix = np.stack(prefixes)
            # Positive cosine kernel: orthogonal prefixes contribute 0 rather
            # than 0.5, making effective rank a clearer route-diversity signal.
            prefix_kernel = np.clip(
                prefix_matrix @ prefix_matrix.T, 0.0, 1.0
            ).astype(np.float32)
            np.fill_diagonal(prefix_kernel, 1.0)
            parent_rank = effective_rank(prefix_kernel)
            parent_diversity_ratio = parent_rank / len(prefixes)
        else:
            parent_rank = np.nan
            parent_diversity_ratio = 0.0

        weight = within_prevalence * contrast
        contribution = weight * parent_diversity_ratio
        evidence_mass += weight
        weighted_parent_diversity += contribution
        n_usable += 1

        if return_details:
            details.append(
                {
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "n_inside": n_inside,
                    "n_outside": outside_count,
                    "within_prevalence": within_prevalence,
                    "outside_prevalence": outside_prevalence,
                    "contrast": contrast,
                    "n_prefixes": len(prefixes),
                    "parent_effective_rank": parent_rank,
                    "parent_diversity_ratio": parent_diversity_ratio,
                    "weight": weight,
                    "independent_contribution": contribution,
                }
            )

    if evidence_mass > 0:
        parent_diversity = weighted_parent_diversity / evidence_mass
    else:
        parent_diversity = 0.0

    metrics = {
        "checkpoint_evidence_mass": evidence_mass,
        "checkpoint_parent_diversity": parent_diversity,
        "checkpoint_independent_score": weighted_parent_diversity,
        "n_usable_checkpoints": n_usable,
    }
    return metrics, details


def random_checkpoint_metrics(
    group_size: int,
    n_random: int,
    checkpoint_objects: Sequence[dict[str, Any]],
    n_traces: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    storage = {
        "checkpoint_evidence_mass": np.full(n_random, np.nan),
        "checkpoint_parent_diversity": np.full(n_random, np.nan),
        "checkpoint_independent_score": np.full(n_random, np.nan),
    }
    for draw in range(n_random):
        group = rng.choice(n_traces, size=group_size, replace=False)
        metrics, _ = checkpoint_group_metrics(
            group,
            checkpoint_objects,
            n_traces,
            return_details=False,
        )
        for name in storage:
            storage[name][draw] = metrics[name]
    return storage


# ---------------------------------------------------------------------------
# Answer scoring and selectors
# ---------------------------------------------------------------------------


def score_answer_candidates(
    answers: Sequence[str],
    pair_matrices: dict[str, np.ndarray],
    checkpoint_objects: Sequence[dict[str, Any]],
    config: Config,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    answer_order = []
    seen = set()
    for answer in answers:
        if answer not in seen:
            seen.add(answer)
            answer_order.append(answer)

    raw_counts = Counter(answers)
    count_levels = sorted(set(raw_counts.values()), reverse=True)
    frequency_rank = {count: rank + 1 for rank, count in enumerate(count_levels)}
    n_traces = len(answers)

    simple_null_cache: dict[int, dict[str, np.ndarray]] = {}
    checkpoint_null_cache: dict[int, dict[str, np.ndarray]] = {}
    detail_rows: list[dict[str, Any]] = []
    rows = []

    for answer in answer_order:
        indices = np.flatnonzero(np.asarray(answers, dtype=object) == answer)
        count = len(indices)

        independent = group_pair_mean(pair_matrices["independent_mass"], indices)
        repeated = group_pair_mean(pair_matrices["repeated_mass"], indices)
        advantage = group_pair_mean(pair_matrices["advantage"], indices)
        independent_fraction = group_pair_mean(
            pair_matrices["independent_fraction"], indices
        )

        if count >= 2 and config.n_random_simple > 0:
            if count not in simple_null_cache:
                simple_null_cache[count] = {
                    "independent": random_group_pair_means(
                        pair_matrices["independent_mass"],
                        count,
                        config.n_random_simple,
                        config.random_batch_size,
                        rng,
                    ),
                    "repeated": random_group_pair_means(
                        pair_matrices["repeated_mass"],
                        count,
                        config.n_random_simple,
                        config.random_batch_size,
                        rng,
                    ),
                    "advantage": random_group_pair_means(
                        pair_matrices["advantage"],
                        count,
                        config.n_random_simple,
                        config.random_batch_size,
                        rng,
                    ),
                }
            simple_ind_cal = calibrate(independent, simple_null_cache[count]["independent"])
            simple_rep_cal = calibrate(repeated, simple_null_cache[count]["repeated"])
            simple_adv_cal = calibrate(advantage, simple_null_cache[count]["advantage"])
        else:
            simple_ind_cal = calibrate(np.nan, np.asarray([]))
            simple_rep_cal = calibrate(np.nan, np.asarray([]))
            simple_adv_cal = calibrate(np.nan, np.asarray([]))

        checkpoint_metrics, checkpoint_details = checkpoint_group_metrics(
            indices,
            checkpoint_objects,
            n_traces,
            return_details=True,
        )
        for detail in checkpoint_details:
            detail_rows.append({"answer": answer, **detail})

        calibrate_checkpoint = (
            config.n_random_checkpoint > 0
            and count >= 2
            and (
                config.checkpoint_random_max_count == 0
                or count <= config.checkpoint_random_max_count
            )
        )
        if calibrate_checkpoint:
            if count not in checkpoint_null_cache:
                checkpoint_null_cache[count] = random_checkpoint_metrics(
                    count,
                    config.n_random_checkpoint,
                    checkpoint_objects,
                    n_traces,
                    rng,
                )
            checkpoint_score_cal = calibrate(
                checkpoint_metrics["checkpoint_independent_score"],
                checkpoint_null_cache[count]["checkpoint_independent_score"],
            )
            checkpoint_mass_cal = calibrate(
                checkpoint_metrics["checkpoint_evidence_mass"],
                checkpoint_null_cache[count]["checkpoint_evidence_mass"],
            )
            checkpoint_div_cal = calibrate(
                checkpoint_metrics["checkpoint_parent_diversity"],
                checkpoint_null_cache[count]["checkpoint_parent_diversity"],
            )
        else:
            checkpoint_score_cal = calibrate(np.nan, np.asarray([]))
            checkpoint_mass_cal = calibrate(np.nan, np.asarray([]))
            checkpoint_div_cal = calibrate(np.nan, np.asarray([]))

        simple_high = normal_cdf(simple_ind_cal["z"])
        repeat_low = normal_cdf(-simple_rep_cal["z"])
        if np.isfinite(simple_high) and np.isfinite(repeat_low):
            simple_structure = math.sqrt(simple_high * repeat_low)
        else:
            simple_structure = 0.0

        checkpoint_mass_high = normal_cdf(checkpoint_mass_cal["z"])
        checkpoint_div_high = normal_cdf(checkpoint_div_cal["z"])
        if np.isfinite(checkpoint_mass_high) and np.isfinite(checkpoint_div_high):
            checkpoint_calibrated_structure = math.sqrt(
                checkpoint_mass_high * checkpoint_div_high
            )
        else:
            checkpoint_calibrated_structure = np.nan

        rows.append(
            {
                "answer": answer,
                "raw_count": count,
                "raw_share": count / n_traces,
                "frequency_rank": frequency_rank[count],
                "simple_independent_mass": independent,
                "simple_repeated_mass": repeated,
                "simple_advantage": advantage,
                "simple_independent_fraction": independent_fraction,
                "simple_independent_z": simple_ind_cal["z"],
                "simple_repeated_z": simple_rep_cal["z"],
                "simple_advantage_z": simple_adv_cal["z"],
                "simple_structure_score": simple_structure,
                **checkpoint_metrics,
                "checkpoint_independent_z": checkpoint_score_cal["z"],
                "checkpoint_evidence_z": checkpoint_mass_cal["z"],
                "checkpoint_parent_diversity_z": checkpoint_div_cal["z"],
                "checkpoint_raw_structure_score": float(
                    checkpoint_metrics["checkpoint_independent_score"]
                ),
                "checkpoint_calibrated_structure_score": (
                    checkpoint_calibrated_structure
                ),
            }
        )

    frame = pd.DataFrame(rows)
    frame["count_rank"] = dense_rank_desc(frame["raw_count"])
    frame["simple_structure_rank"] = dense_rank_desc(frame["simple_structure_score"])
    frame["checkpoint_raw_structure_rank"] = dense_rank_desc(
        frame["checkpoint_raw_structure_score"]
    )
    frame["checkpoint_calibrated_structure_rank"] = dense_rank_desc(
        frame["checkpoint_calibrated_structure_score"]
    )
    # Coefficient-free structural agreement: geometric rank product of the
    # simple convergence signal and the raw checkpoint-community signal.
    frame["combined_structure_score"] = 1.0 / np.sqrt(
        frame["simple_structure_rank"]
        * frame["checkpoint_raw_structure_rank"]
    )
    frame["combined_structure_rank"] = dense_rank_desc(
        frame["combined_structure_score"]
    )

    frame["simple_count_rank_product"] = 1.0 / np.sqrt(
        frame["count_rank"] * frame["simple_structure_rank"]
    )
    frame["checkpoint_count_rank_product"] = 1.0 / np.sqrt(
        frame["count_rank"] * frame["checkpoint_raw_structure_rank"]
    )
    frame["combined_count_rank_product"] = 1.0 / np.cbrt(
        frame["count_rank"]
        * frame["simple_structure_rank"]
        * frame["checkpoint_raw_structure_rank"]
    )
    return frame, detail_rows


def select_best(frame: pd.DataFrame, score: str, answer_order: dict[str, int]) -> pd.Series:
    work = frame.copy()
    work["_score"] = pd.to_numeric(work[score], errors="coerce").fillna(-np.inf)
    work["_order"] = work["answer"].map(answer_order)
    return work.sort_values(
        ["_score", "raw_count", "_order"],
        ascending=[False, False, True],
        kind="mergesort",
    ).iloc[0]


def choose_selector(
    frame: pd.DataFrame,
    selector: str,
    majority_answer: str,
    answer_order: dict[str, int],
) -> tuple[pd.Series, str]:
    mapping = {
        "simple_raw_advantage": "simple_advantage",
        "simple_z_convergence": "simple_structure_score",
        "simple_count_rank_product": "simple_count_rank_product",
        "checkpoint_raw_independent": "checkpoint_independent_score",
        "checkpoint_z_independent": "checkpoint_calibrated_structure_score",
        "checkpoint_count_rank_product": "checkpoint_count_rank_product",
        "combined_count_rank_product": "combined_count_rank_product",
    }
    if selector in mapping:
        score = mapping[selector]
        return select_best(frame, score, answer_order), score
    if selector == "majority_tiebreak_combined":
        maximum = int(frame["raw_count"].max())
        tied = frame[frame["raw_count"] == maximum]
        if len(tied) == 1:
            return tied.iloc[0], "combined_structure_score"
        return select_best(tied, "combined_structure_score", answer_order), (
            "combined_structure_score"
        )
    raise KeyError(selector)


def case_type(majority_correct: bool, selected_correct: bool) -> str:
    if not majority_correct and selected_correct:
        return "recovery"
    if majority_correct and not selected_correct:
        return "harm"
    if not majority_correct and not selected_correct:
        return "wrong_to_wrong"
    return "correct_to_correct"


def summarize_selectors(question_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    majority = question_results["majority_correct"].astype(bool)
    rows.append(
        {
            "model": question_results["model"].iloc[0],
            "selector": "majority",
            "n_questions": len(question_results),
            "correct": int(majority.sum()),
            "accuracy": float(majority.mean()),
            "non_majority_choices": 0,
            "majority_failures_recovered": 0,
            "correct_majorities_harmed": 0,
            "net_repairs": 0,
            "wrong_to_wrong_changes": 0,
            "correct_to_correct_changes": 0,
        }
    )
    for selector in SELECTORS:
        selected = question_results[f"{selector}_correct"].astype(bool)
        changed = question_results[f"{selector}_changed_majority"].astype(bool)
        recoveries = (~majority) & selected & changed
        harms = majority & (~selected) & changed
        wrong = (~majority) & (~selected) & changed
        c2c = majority & selected & changed
        rows.append(
            {
                "model": question_results["model"].iloc[0],
                "selector": selector,
                "n_questions": len(question_results),
                "correct": int(selected.sum()),
                "accuracy": float(selected.mean()),
                "non_majority_choices": int(changed.sum()),
                "majority_failures_recovered": int(recoveries.sum()),
                "correct_majorities_harmed": int(harms.sum()),
                "net_repairs": int(recoveries.sum() - harms.sum()),
                "wrong_to_wrong_changes": int(wrong.sum()),
                "correct_to_correct_changes": int(c2c.sum()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-question processing
# ---------------------------------------------------------------------------


def process_question(
    group: pd.DataFrame,
    question_key: str,
    source_file: str,
    model_name: str,
    embedder: CachedSentenceTransformer,
    config: Config,
    model_dir: Path,
    rng: np.random.Generator,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    g = group.copy().reset_index(drop=True)
    g["_answer"] = g[config.answer_col].map(normalize_answer)
    g["_hit"] = g[config.hit_col].map(parse_bool)
    g = g[g["_answer"].notna()].reset_index(drop=True)
    if g.empty:
        raise ValueError("No non-empty answers")

    answers = g["_answer"].astype(str).tolist()
    hits = g["_hit"].astype(bool).tolist()
    majority_answer, majority_count, majority_candidates = first_occurrence_majority(
        answers
    )
    correct_map = answer_correctness(answers, hits)
    majority_correct = bool(correct_map[majority_answer])

    units_by_trace = [
        extract_reasoning_units(
            text,
            config.min_unit_chars,
            config.max_unit_chars,
            config.max_units_per_trace,
        )
        for text in g[config.text_col]
    ]

    unit_trace: list[int] = []
    unit_index: list[int] = []
    unit_position: list[float] = []
    unit_text: list[str] = []
    embedding_text: list[str] = []
    trace_indices: list[np.ndarray] = []
    cursor = 0

    for trace, units in enumerate(units_by_trace):
        start = cursor
        denominator = max(len(units) - 1, 1)
        for local, text in enumerate(units):
            unit_trace.append(trace)
            unit_index.append(local)
            unit_position.append(local / denominator if len(units) > 1 else 0.5)
            unit_text.append(text)
            embedding_text.append(make_embedding_unit(units, local, config.unit_context))
            cursor += 1
        trace_indices.append(np.arange(start, cursor, dtype=int))

    unit_trace_arr = np.asarray(unit_trace, dtype=int)
    unit_index_arr = np.asarray(unit_index, dtype=int)
    unit_position_arr = np.asarray(unit_position, dtype=np.float32)

    if embedding_text:
        embeddings = preprocess_embeddings(
            embedder.encode(embedding_text),
            config.embedding_preprocess,
            config.shared_pc_count,
        )
        unit_similarity = scale_cosine(embeddings @ embeddings.T)
        commonness = compute_unit_commonness(
            unit_similarity,
            trace_indices,
            unit_trace_arr,
            config,
        )
        full_adjusted = residualize(
            unit_similarity,
            commonness,
            commonness,
            config,
        )
    else:
        embeddings = np.empty((0, 0), dtype=np.float32)
        unit_similarity = np.empty((0, 0), dtype=np.float32)
        commonness = np.empty(0, dtype=np.float32)
        full_adjusted = np.empty((0, 0), dtype=np.float32)

    prefix_embeddings, prefix_valid = compute_prefix_embeddings(
        embeddings, trace_indices
    ) if embeddings.size else (
        np.empty_like(embeddings), np.zeros(len(unit_text), dtype=bool)
    )
    pair_matrices, pair_detail = build_pairwise_convergence_matrices(
        unit_similarity,
        trace_indices,
        commonness,
        prefix_embeddings,
        prefix_valid,
        config,
    )

    if full_adjusted.size:
        communities, _ = build_checkpoint_communities(
            full_adjusted,
            unit_trace_arr,
            config,
        )
        checkpoint_objects, checkpoint_rows = prepare_checkpoint_objects(
            communities,
            full_adjusted,
            embeddings,
            trace_indices,
            unit_trace_arr,
            unit_index_arr,
            unit_position_arr,
            unit_text,
            answers,
        )
    else:
        checkpoint_objects, checkpoint_rows = [], []

    answer_frame, checkpoint_answer_details = score_answer_candidates(
        answers,
        pair_matrices,
        checkpoint_objects,
        config,
        rng,
    )
    answer_frame["is_correct_answer"] = answer_frame["answer"].map(correct_map)

    answer_order = {}
    for index, answer in enumerate(answers):
        answer_order.setdefault(answer, index)

    question_result: dict[str, Any] = {
        "model": model_name,
        "question_key": question_key,
        "source_file": source_file,
        "id": g[config.id_col].iloc[0] if config.id_col in g else "all",
        "n_traces": len(g),
        "n_answers": len(answer_frame),
        "n_correct_traces": int(sum(hits)),
        "has_correct_trace": bool(any(hits)),
        "majority_answer": majority_answer,
        "majority_count": majority_count,
        "n_top_count_answers": len(majority_candidates),
        "majority_is_unique": len(majority_candidates) == 1,
        "majority_correct": majority_correct,
        "correct_answers_json": json.dumps(
            [answer for answer, correct in correct_map.items() if correct],
            ensure_ascii=False,
        ),
    }

    changed_rows = []
    for selector in SELECTORS:
        selected, score_column = choose_selector(
            answer_frame,
            selector,
            majority_answer,
            answer_order,
        )
        selected_answer = str(selected["answer"])
        selected_correct = bool(correct_map[selected_answer])
        changed = selected_answer != majority_answer
        majority_row = answer_frame[answer_frame["answer"] == majority_answer].iloc[0]
        selected_score = float(selected[score_column]) if pd.notna(selected[score_column]) else np.nan
        majority_score = (
            float(majority_row[score_column])
            if pd.notna(majority_row[score_column])
            else np.nan
        )

        question_result[f"{selector}_answer"] = selected_answer
        question_result[f"{selector}_correct"] = selected_correct
        question_result[f"{selector}_changed_majority"] = changed
        question_result[f"{selector}_selected_count"] = int(selected["raw_count"])
        question_result[f"{selector}_selected_frequency_rank"] = int(
            selected["frequency_rank"]
        )
        question_result[f"{selector}_selected_score"] = selected_score
        question_result[f"{selector}_margin_over_majority"] = (
            selected_score - majority_score
            if np.isfinite(selected_score) and np.isfinite(majority_score)
            else np.nan
        )

        if changed:
            changed_rows.append(
                {
                    "model": model_name,
                    "question_key": question_key,
                    "selector": selector,
                    "case_type": case_type(majority_correct, selected_correct),
                    "majority_answer": majority_answer,
                    "majority_count": majority_count,
                    "majority_correct": majority_correct,
                    "selected_answer": selected_answer,
                    "selected_count": int(selected["raw_count"]),
                    "selected_frequency_rank": int(selected["frequency_rank"]),
                    "selected_correct": selected_correct,
                    "selected_score": selected_score,
                    "majority_score": majority_score,
                    "margin_over_majority": question_result[
                        f"{selector}_margin_over_majority"
                    ],
                    "matrix_path": str(
                        model_dir / "matrices" / f"{question_file_stem(question_key)}.npz"
                    ),
                }
            )

    base = {
        "model": model_name,
        "question_key": question_key,
        "source_file": source_file,
        "id": g[config.id_col].iloc[0] if config.id_col in g else "all",
    }

    answer_rows = [{**base, **row} for row in answer_frame.to_dict("records")]
    step_rows = []
    for unit, text in enumerate(unit_text):
        trace = int(unit_trace_arr[unit])
        step_rows.append(
            {
                **base,
                "trace_index": trace,
                "completion_idx": g.loc[trace].get("completion_idx", trace),
                "answer": answers[trace],
                "row_hit": hits[trace],
                "unit_global_index": unit,
                "unit_index": int(unit_index_arr[unit]),
                "relative_position": float(unit_position_arr[unit]),
                "unit_text": text,
                "unit_commonness": float(commonness[unit]) if len(commonness) else np.nan,
            }
        )

    pair_rows = []
    for row in pair_detail:
        i, j = int(row["trace_i"]), int(row["trace_j"])
        global_i = int(trace_indices[i][row["source_unit"]])
        global_j = int(trace_indices[j][row["target_unit"]])
        pair_rows.append(
            {
                **base,
                **row,
                "trace_i_answer": answers[i],
                "trace_j_answer": answers[j],
                "trace_i_hit": hits[i],
                "trace_j_hit": hits[j],
                "source_step_text": unit_text[global_i],
                "target_step_text": unit_text[global_j],
            }
        )
    pair_rows = sorted(
        pair_rows,
        key=lambda row: row["independent_contribution"],
        reverse=True,
    )[: config.save_pairwise_top_k]

    checkpoint_rows = [{**base, **row} for row in checkpoint_rows]
    checkpoint_answer_rows = [{**base, **row} for row in checkpoint_answer_details]

    matrix_dir = model_dir / "matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = matrix_dir / f"{question_file_stem(question_key)}.npz"
    payload = {
        **{name: matrix.astype(np.float32) for name, matrix in pair_matrices.items()},
        "answers": np.asarray(answers, dtype=str),
        "row_hit": np.asarray(hits, dtype=bool),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "model": model_name,
                    "question_key": question_key,
                    "source_file": source_file,
                    "config": asdict(config),
                },
                ensure_ascii=False,
            )
        ),
    }
    if config.matrix_compression:
        np.savez_compressed(matrix_path, **payload)
    else:
        np.savez(matrix_path, **payload)

    return (
        question_result,
        answer_rows,
        step_rows,
        pair_rows,
        checkpoint_rows,
        checkpoint_answer_rows,
        changed_rows,
    )


# ---------------------------------------------------------------------------
# Input and CLI
# ---------------------------------------------------------------------------


def expand_inputs(patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        candidate = Path(pattern)
        if candidate.is_dir():
            paths.extend(sorted(candidate.glob("*.csv")))
            continue
        matches = [Path(path) for path in glob.glob(pattern)]
        if matches:
            paths.extend(sorted(matches))
        elif candidate.exists():
            paths.append(candidate)
    unique = []
    seen = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    if not unique:
        raise FileNotFoundError(f"No CSV inputs matched {list(patterns)}")
    return unique


def validate_columns(frame: pd.DataFrame, config: Config, path: Path) -> None:
    required = {config.text_col, config.answer_col, config.hit_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-path", default="checkpoint_route_embeddings.sqlite")
    parser.add_argument("--models", nargs="+", default=list(Config.models))
    parser.add_argument("--text-col", default=Config.text_col)
    parser.add_argument("--id-col", default=Config.id_col)
    parser.add_argument("--answer-col", default=Config.answer_col)
    parser.add_argument("--hit-col", default=Config.hit_col)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--device", default=None)

    parser.add_argument(
        "--min-unit-chars",
        type=int,
        default=Config.min_unit_chars,
        help=(
            "Sentence units are never merged. This value is retained for "
            "compatibility; short mathematical sentences are still preserved."
        ),
    )
    parser.add_argument("--max-unit-chars", type=int, default=Config.max_unit_chars)
    parser.add_argument(
        "--max-units-per-trace",
        type=int,
        default=Config.max_units_per_trace,
        help="Maximum number of sentence units retained per trace.",
    )
    parser.add_argument(
        "--unit-context", choices=["current", "previous_current"], default=Config.unit_context
    )
    parser.add_argument(
        "--embedding-preprocess",
        choices=["none", "center", "remove_pc"],
        default=Config.embedding_preprocess,
    )
    parser.add_argument("--shared-pc-count", type=int, default=Config.shared_pc_count)
    parser.add_argument(
        "--commonness-stat",
        choices=["median", "mean", "quantile"],
        default=Config.commonness_stat,
    )
    parser.add_argument("--commonness-quantile", type=float, default=Config.commonness_quantile)
    parser.add_argument(
        "--commonness-pair", choices=["mean", "max", "min"], default=Config.commonness_pair
    )
    parser.add_argument(
        "--residualization",
        choices=["none", "subtract", "relative"],
        default=Config.residualization,
    )

    parser.add_argument("--min-prefix-steps", type=int, default=Config.min_prefix_steps)
    parser.add_argument(
        "--prefix-similarity",
        choices=["positive_cosine", "scaled_cosine"],
        default=Config.prefix_similarity,
    )

    parser.add_argument("--checkpoint-knn", type=int, default=Config.checkpoint_knn)
    parser.add_argument(
        "--checkpoint-edge-mode",
        choices=["mutual", "union"],
        default=Config.checkpoint_edge_mode,
    )
    parser.add_argument(
        "--checkpoint-min-similarity",
        type=float,
        default=Config.checkpoint_min_similarity,
    )
    parser.add_argument(
        "--checkpoint-min-trace-support",
        type=int,
        default=Config.checkpoint_min_trace_support,
    )
    parser.add_argument(
        "--checkpoint-resolution", type=float, default=Config.checkpoint_resolution
    )

    parser.add_argument("--n-random-simple", type=int, default=Config.n_random_simple)
    parser.add_argument(
        "--n-random-checkpoint", type=int, default=Config.n_random_checkpoint
    )
    parser.add_argument("--random-batch-size", type=int, default=Config.random_batch_size)
    parser.add_argument(
        "--checkpoint-random-max-count",
        type=int,
        default=Config.checkpoint_random_max_count,
        help="0 calibrates checkpoint scores for all group sizes.",
    )
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--save-pairwise-top-k", type=int, default=Config.save_pairwise_top_k)
    parser.add_argument("--no-matrix-compression", action="store_true")
    parser.add_argument("--disable-progress", action="store_true")
    return parser.parse_args()


def write_report(output_dir: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# Checkpoint-route convergence analysis",
        "",
        "## Majority comparison",
        "",
        summary.to_markdown(index=False) if not summary.empty else "_No results._",
        "",
        "## Interpretation",
        "",
        "- `simple_*` selectors test whether matched intermediate conclusions are reached from dissimilar prefixes rather than repeated paths.",
        "- `checkpoint_*` selectors use checkpoint communities and the effective rank of prefix embeddings leading into answer-specific checkpoints.",
        "- `*_count_rank_product` combines frequency and structure through geometric rank products, without learned coefficients.",
        "- Positive `net_repairs` means the selector repairs more majority failures than correct majorities it harms.",
    ]
    (output_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = Config(
        models=tuple(args.models),
        text_col=args.text_col,
        id_col=args.id_col,
        answer_col=args.answer_col,
        hit_col=args.hit_col,
        batch_size=args.batch_size,
        device=args.device,
        min_unit_chars=args.min_unit_chars,
        max_unit_chars=args.max_unit_chars,
        max_units_per_trace=args.max_units_per_trace,
        unit_context=args.unit_context,
        embedding_preprocess=args.embedding_preprocess,
        shared_pc_count=args.shared_pc_count,
        commonness_stat=args.commonness_stat,
        commonness_quantile=args.commonness_quantile,
        commonness_pair=args.commonness_pair,
        residualization=args.residualization,
        min_prefix_steps=args.min_prefix_steps,
        prefix_similarity=args.prefix_similarity,
        checkpoint_knn=args.checkpoint_knn,
        checkpoint_edge_mode=args.checkpoint_edge_mode,
        checkpoint_min_similarity=args.checkpoint_min_similarity,
        checkpoint_min_trace_support=args.checkpoint_min_trace_support,
        checkpoint_resolution=args.checkpoint_resolution,
        n_random_simple=args.n_random_simple,
        n_random_checkpoint=args.n_random_checkpoint,
        random_batch_size=args.random_batch_size,
        checkpoint_random_max_count=args.checkpoint_random_max_count,
        seed=args.seed,
        save_pairwise_top_k=args.save_pairwise_top_k,
        matrix_compression=not args.no_matrix_compression,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = expand_inputs(args.input)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": asdict(config),
                "inputs": [str(path) for path in input_paths],
                "cache_path": args.cache_path,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    frames = []
    for path in input_paths:
        frame = pd.read_csv(path)
        validate_columns(frame, config, path)
        if config.id_col not in frame.columns:
            frame[config.id_col] = "all"
        frames.append((path, frame))

    all_question_results = []
    all_summaries = []
    rng_master = np.random.SeedSequence(config.seed)

    for model_number, model_name in enumerate(config.models):
        model_dir = output_dir / model_slug(model_name)
        model_dir.mkdir(parents=True, exist_ok=True)
        embedder = CachedSentenceTransformer(
            model_name,
            args.cache_path,
            config.batch_size,
            config.device,
        )
        model_rng = np.random.default_rng(rng_master.spawn(1)[0])

        question_results = []
        answer_rows = []
        step_rows = []
        pair_rows = []
        checkpoint_rows = []
        checkpoint_answer_rows = []
        changed_rows = []
        failed_rows = []

        question_items = []
        for path, frame in frames:
            for qid, group in frame.groupby(config.id_col, sort=False, dropna=False):
                question_items.append((path, qid, group))

        iterator = question_items
        if not args.disable_progress:
            iterator = tqdm(
                question_items,
                desc=f"{model_name}",
                unit="question",
            )

        try:
            for path, qid, group in iterator:
                question_key = f"{path.name}::{qid}"
                try:
                    outputs = process_question(
                        group,
                        question_key,
                        path.name,
                        model_name,
                        embedder,
                        config,
                        model_dir,
                        model_rng,
                    )
                    (
                        question_result,
                        current_answers,
                        current_steps,
                        current_pairs,
                        current_checkpoints,
                        current_checkpoint_answers,
                        current_changed,
                    ) = outputs
                    question_results.append(question_result)
                    answer_rows.extend(current_answers)
                    step_rows.extend(current_steps)
                    pair_rows.extend(current_pairs)
                    checkpoint_rows.extend(current_checkpoints)
                    checkpoint_answer_rows.extend(current_checkpoint_answers)
                    changed_rows.extend(current_changed)
                except Exception as error:
                    failed_rows.append(
                        {"question_key": question_key, "error": repr(error)}
                    )
        finally:
            embedder.close()

        qdf = pd.DataFrame(question_results)
        adf = pd.DataFrame(answer_rows)
        sdf = pd.DataFrame(step_rows)
        pdf = pd.DataFrame(pair_rows)
        cdf = pd.DataFrame(checkpoint_rows)
        cadf = pd.DataFrame(checkpoint_answer_rows)
        chdf = pd.DataFrame(changed_rows)
        fdf = pd.DataFrame(failed_rows)

        qdf.to_csv(model_dir / "question_results.csv", index=False)
        adf.to_csv(model_dir / "answer_scores.csv", index=False)
        sdf.to_csv(model_dir / "steps.csv", index=False)
        pdf.to_csv(model_dir / "pairwise_convergence.csv", index=False)
        cdf.to_csv(model_dir / "checkpoint_communities.csv", index=False)
        cadf.to_csv(model_dir / "checkpoint_answer_stats.csv", index=False)
        chdf.to_csv(model_dir / "changed_cases.csv", index=False)
        fdf.to_csv(model_dir / "failed_questions.csv", index=False)

        if not chdf.empty:
            for case in ["recovery", "harm", "wrong_to_wrong", "correct_to_correct"]:
                chdf[chdf["case_type"] == case].to_csv(
                    model_dir / f"{case if case != 'wrong_to_wrong' else 'wrong_to_wrong'}.csv",
                    index=False,
                )

        if not qdf.empty:
            summary = summarize_selectors(qdf)
        else:
            summary = pd.DataFrame()
        summary.to_csv(model_dir / "selector_summary.csv", index=False)
        write_report(model_dir, summary)

        all_question_results.extend(question_results)
        all_summaries.append(summary)

    combined_q = pd.DataFrame(all_question_results)
    combined_s = (
        pd.concat(all_summaries, ignore_index=True)
        if all_summaries
        else pd.DataFrame()
    )
    combined_q.to_csv(output_dir / "combined_question_results.csv", index=False)
    combined_s.to_csv(output_dir / "combined_selector_summary.csv", index=False)
    write_report(output_dir, combined_s)
    print("\nSelector summary:")
    print(combined_s.to_string(index=False) if not combined_s.empty else "No results")
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
