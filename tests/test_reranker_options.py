"""Exercise Hub revision resolution and cache separation without model downloads."""

import sys
from types import SimpleNamespace

from structural_signatures import scoring


def test_model_and_revision_are_forwarded_and_caches_are_isolated(
    tmp_path, monkeypatch
):
    resolutions, loads = [], []

    def snapshot_download(repo_id, revision, allow_patterns):
        resolutions.append((repo_id, revision))
        commit = {"experiment": "alternate-commit-2", None: "alternate-commit-1"}.get(
            revision, revision
        )
        return str(tmp_path / "snapshots" / commit)

    class Encoder:
        num_labels = 1

        def __init__(self, name, *, revision, device, **kwargs):
            loads.append((name, revision, kwargs["max_length"]))
            self.device = device
            self.model = SimpleNamespace(
                parameters=lambda: iter([SimpleNamespace(dtype="float32")])
            )

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=Encoder)
    )
    monkeypatch.setattr(scoring, "version", lambda package: "test")
    paper = scoring.DEFAULT_RERANKER_MODEL
    paths = []
    for model, rev in [
        (paper, None),
        ("other/model", None),
        ("other/model", "experiment"),
        ("other/model", None),
    ]:
        scorer = scoring.ModelScores(
            tmp_path, "cpu", revision=rev, model_name=model, max_length=128
        )
        try:
            paths.append(scorer.cache_path)
            assert (
                scorer.identity["model"],
                scorer.identity["snapshot"],
                scorer.max_length,
            ) == loads[-1]
        finally:
            scorer.close()
    assert resolutions[:3] == [
        (paper, scoring.SNAPSHOT),
        ("other/model", None),
        ("other/model", "experiment"),
    ]
    assert loads[:3] == [
        (paper, scoring.SNAPSHOT, 128),
        ("other/model", "alternate-commit-1", 128),
        ("other/model", "alternate-commit-2", 128),
    ]
    assert len(set(paths)) == 3 and paths[1] == paths[-1]
