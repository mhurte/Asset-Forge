from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


def read_json(path):
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def resolve_path(value, base):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(base) / path).resolve()


def derived_seed(seed, *keys):
    return int(digest([seed, *keys])[:16], 16)


def git_revision(repo):
    repo = Path(repo).resolve()
    try:
        revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                          stderr=subprocess.PIPE, text=True).strip()
        status = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"],
                                        stderr=subprocess.PIPE, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"Cannot record Git provenance for {repo}: {exc}") from exc
    if status:
        raise ValueError(f"Dependency checkout must be clean for reproducible builds: {repo}")
    return {"path": str(repo), "commit": revision}
