"""Historical directional predictions and a revision-scoped cache for new runs."""

import json
import sqlite3
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from .legacy.audit_answer_local_prefix_allpairs_v8 import (
    DEFAULT_INSTRUCTION,
    DEFAULT_RERANKER_MODEL,
    CachedDirectionalReranker,
)

SNAPSHOT = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
PROMPT = DEFAULT_INSTRUCTION.strip()
PROMPT_HASH = sha256(PROMPT.encode()).hexdigest()


class ArchivedScores:
    """Look up canonical v8 keys, then restore the requested text direction."""

    model_name = DEFAULT_RERANKER_MODEL
    prompt_hash = PROMPT_HASH
    max_length = 256
    canonical_pair = staticmethod(CachedDirectionalReranker.canonical_pair)
    pair_key = CachedDirectionalReranker.pair_key

    def directional_logits(self, pairs, *, show_progress_bar=False):
        keys = [self.pair_key(left, right) for left, right in pairs]
        cached = self.lookup(keys)
        forward, reverse = [], []
        for (left, right), key in zip(pairs, keys):
            if key not in cached:
                raise KeyError(
                    f"Pair {key} is absent from the supplied directional scores"
                )
            a, b = cached[key]
            if left > right:
                a, b = b, a
            forward.append(a)
            reverse.append(b)
        return np.asarray(forward, dtype=np.float32), np.asarray(
            reverse, dtype=np.float32
        )


class FrozenScores(ArchivedScores):
    """Portable subset exported from a legacy cache."""

    def __init__(self, path):
        table = pd.read_csv(path, float_precision="round_trip")
        if table.pair_key.duplicated().any():
            raise ValueError("Duplicate directional score key")
        values = table[["canonical_left_to_right", "canonical_right_to_left"]]
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError("Non-finite cached score")
        self.rows = dict(zip(table.pair_key, values.itertuples(index=False, name=None)))

    def lookup(self, keys):
        return self.rows


class LegacyScores(ArchivedScores):
    """Query an existing SQLite shard, opened read-only, with the v8 key lookup."""

    lookup = CachedDirectionalReranker._read

    def __init__(self, path):
        self.connection = sqlite3.connect(
            Path(path).resolve().as_uri() + "?mode=ro", uri=True
        )

    def close(self):
        self.connection.close()


class ModelScores(CachedDirectionalReranker):
    """Use the recovered CrossEncoder calls and retain its scalar outputs verbatim."""

    def __init__(
        self,
        cache_dir,
        device,
        revision=None,
        batch_size=64,
        *,
        model_name=DEFAULT_RERANKER_MODEL,
        max_length=256,
    ):
        from huggingface_hub import snapshot_download
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.prompt = PROMPT
        self.prompt_hash = PROMPT_HASH
        self.max_length = max_length
        self.batch_size = batch_size
        if revision is None and model_name == DEFAULT_RERANKER_MODEL:
            revision = SNAPSHOT
        # Resolve a movable Hub reference once. CrossEncoder fetches the weights
        # and tokenizer it uses at that commit, including from an offline cache.
        self.snapshot = snapshot_download(
            repo_id=self.model_name, revision=revision, allow_patterns=["config.json"]
        )
        resolved_revision = Path(self.snapshot).name
        self.model = CrossEncoder(
            self.model_name,
            revision=resolved_revision,
            device=device,
            max_length=self.max_length,
            prompts={"checkpoint_equivalence": PROMPT},
            default_prompt_name="checkpoint_equivalence",
        )
        if self.model.num_labels != 1:
            raise ValueError(
                "The reranker must return one scalar score for each text pair"
            )
        self.identity = {
            "schema": 1,
            "model": self.model_name,
            "snapshot": resolved_revision,
            "prompt_sha256": PROMPT_HASH,
            "max_length": self.max_length,
            "api": "CrossEncoder.predict",
            "device": str(self.model.device),
            "dtype": str(next(self.model.model.parameters()).dtype),
            "packages": {
                name: version(name)
                for name in ("sentence-transformers", "transformers", "torch", "numpy")
            },
        }
        identity = json.dumps(self.identity, sort_keys=True)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True, parents=True)
        self.cache_path = cache_dir / (
            sha256(identity.encode()).hexdigest() + ".sqlite"
        )
        self.connection = sqlite3.connect(self.cache_path, timeout=120)
        with self.connection:
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS identity (value TEXT PRIMARY KEY)"
            )
            self.connection.execute(
                "INSERT OR IGNORE INTO identity VALUES (?)", (identity,)
            )
            if (
                self.connection.execute("SELECT COUNT(*) FROM identity").fetchone()[0]
                != 1
            ):
                raise ValueError("Directional cache belongs to a different scorer")
            self.connection.execute("""CREATE TABLE IF NOT EXISTS directional_logits (
                pair_key TEXT PRIMARY KEY, model_name TEXT, prompt_hash TEXT, max_length INTEGER,
                canonical_left TEXT, canonical_right TEXT, canonical_left_to_right REAL,
                canonical_right_to_left REAL)""")
        self.n_new_pairs = self.n_cached_pairs = 0
