"""Run manifests: record exactly what a result file was produced from.

`write_manifest` stores the git commit, whether the working tree was dirty,
the SHA-256 and size of every input and output file, package versions, the
command line, seeds and a timestamp, as JSON next to the results. This does
not make a run reproducible by itself, but it makes "which data and code
produced this number" answerable, which published results increasingly need.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Dict, Iterable, Optional

_PACKAGES = ("numpy", "pandas", "scikit-learn", "xgboost", "scipy", "joblib")


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _git(args, cwd) -> Optional[str]:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception:  # noqa: BLE001 - manifest must never break a run
        return None


def _file_entry(path) -> Dict[str, object]:
    p = Path(path)
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)}


def build_manifest(inputs: Iterable, outputs: Iterable = (), seeds=None, extra: Optional[dict] = None, repo_dir=".") -> dict:
    versions = {}
    for name in _PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    status = _git(["status", "--porcelain"], repo_dir)
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": sys.argv,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git(["rev-parse", "HEAD"], repo_dir),
        "git_dirty": bool(status),
        "packages": versions,
        "seeds": list(seeds) if seeds is not None else None,
        "inputs": [_file_entry(p) for p in inputs],
        "outputs": [_file_entry(p) for p in outputs if Path(p).exists()],
        "extra": extra or {},
    }


def write_manifest(path, **kwargs) -> dict:
    manifest = build_manifest(**kwargs)
    Path(path).write_text(json.dumps(manifest, indent=2, default=str))
    return manifest
