"""File identities and complete run directories."""

import json
import platform
from contextlib import contextmanager
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_metadata(directory, files, **fields):
    metadata = {
        **fields,
        "inputs": {
            name: {"file": Path(path).name, "sha256": file_sha256(path)}
            for name, path in files.items()
        },
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in (
                "structural-signatures",
                "numpy",
                "pandas",
                "networkx",
                "scipy",
            )
        },
    }
    (directory / "run.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


@contextmanager
def run_directory(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Choose a new run directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".building-", dir=destination.parent) as scratch:
        staged = Path(scratch) / "run"
        staged.mkdir()
        yield staged
        staged.rename(destination)
