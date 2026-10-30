"""Prepare and route SAM3D's Torch Hub dependencies for offline compute nodes."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import subprocess
from urllib.parse import urlparse

from .common import file_digest, read_json, write_json

SOURCES = read_json(Path(__file__).resolve().parents[1] / "environments/level0-offline-sources.json")
RECOVERY = "Run python3 assetgen.py --setup --resume with your checkpoint paths (or --from-run migration input) on the online front node."


def collect_dino_requests(pipeline):
    """Read actual checkpoint configs; never guess the DINO variant or pretrained flag."""
    import yaml
    visited, requests = set(), {}
    def visit_file(path):
        path = path.resolve()
        if path in visited:
            return
        visited.add(path)
        walk(yaml.safe_load(path.read_text()), path.parent)
    def walk(value, base):
        if isinstance(value, list):
            for item in value:
                walk(item, base)
        elif isinstance(value, dict):
            if str(value.get("_target_", "")).endswith(".dino.Dino"):
                source = value.get("source", "github")
                repo = value.get("repo_or_dir", "facebookresearch/dinov2")
                if source != "local":
                    if repo != "facebookresearch/dinov2":
                        raise ValueError(f"Unrecognized DINO source {repo!r}; configure its offline source explicitly")
                    request = {"model": value.get("dino_model", "dinov2_vitb14"),
                               "kwargs": value.get("backbone_kwargs") or {}}
                    if "${" in json.dumps(request):
                        raise ValueError("Unresolved interpolation in DINO config; supply a resolved checkpoint configuration")
                    requests[json.dumps(request, sort_keys=True)] = request
            for key, item in value.items():
                if key.endswith("_config_path") and isinstance(item, str):
                    candidate = base / item
                    if candidate.suffix.lower() in {".yaml", ".yml"}:
                        if not candidate.is_file():
                            raise FileNotFoundError(f"Referenced SAM3D config missing: {candidate}")
                        visit_file(candidate)
                walk(item, base)
    visit_file(Path(pipeline))
    if not requests:
        raise ValueError("No GitHub DINO configurations found in the checkpoint configuration tree")
    return list(requests.values())


def read_prepared(tools):
    tools = Path(tools)
    path = tools / "offline-dependencies.json"
    if not path.is_file():
        raise FileNotFoundError(f"Offline SAM3D dependencies have not been prepared. {RECOVERY}")
    manifest = read_json(path)
    repo = tools / "dinov2"
    if (manifest.get("schema_version") != 1 or manifest.get("dinov2_commit") != SOURCES["dinov2"]["commit"] or
            not (repo / "hubconf.py").is_file() or
            file_digest(repo / "hubconf.py") != manifest.get("hubconf_sha256")):
        raise ValueError(f"Pinned DINOv2 source is missing or changed. {RECOVERY}")
    for name, digest in manifest.get("checkpoints", {}).items():
        # Full checkpoint hashes are recorded during preparation; presence is
        # checked per inference to avoid rehashing multi-GB weights each asset.
        if not (tools / "torch-cache/hub/checkpoints" / name).is_file():
            raise FileNotFoundError(f"Cached DINO checkpoint {name} is missing. {RECOVERY}")
    return manifest


@contextmanager
def offline_torch_hub(tools, prepared=None):
    """Scope a local-source adapter to one inference; upstream repositories stay unchanged."""
    import torch
    tools = Path(tools)
    manifest = read_prepared(tools) if prepared is None else prepared
    original_load = torch.hub.load
    original_weights = torch.hub.load_state_dict_from_url
    original_directory = torch.hub.get_dir()
    torch.hub.set_dir(str(tools / "torch-cache/hub"))
    def load(repo_or_dir, model, *args, source="github", **kwargs):
        if source == "github":
            if repo_or_dir != "facebookresearch/dinov2":
                raise ValueError(f"Unprepared online Torch Hub source: {repo_or_dir!r}. {RECOVERY}")
            model_kwargs = {k: v for k, v in kwargs.items()
                            if k not in {"verbose", "force_reload", "skip_validation", "trust_repo"}}
            request = {"model": model, "kwargs": model_kwargs}
            if request not in manifest["requests"]:
                raise ValueError(f"DINO variant/options were not prepared: {request!r}. {RECOVERY}")
            repo_or_dir, source = str(tools / "dinov2"), "local"
        return original_load(repo_or_dir, model, *args, source=source, **kwargs)
    def weights(url, model_dir=None, *args, **kwargs):
        directory = Path(model_dir) if model_dir else Path(torch.hub.get_dir()) / "checkpoints"
        name = kwargs.get("file_name") or Path(urlparse(url).path).name
        if not (directory / name).is_file():
            raise FileNotFoundError(f"Uncached Torch Hub weights requested: {url}. {RECOVERY}")
        return original_weights(url, model_dir, *args, **kwargs)
    torch.hub.load, torch.hub.load_state_dict_from_url = load, weights
    try:
        yield manifest
    finally:
        torch.hub.load, torch.hub.load_state_dict_from_url = original_load, original_weights
        torch.hub.set_dir(original_directory)


def prepare(pipeline, tools):
    import torch
    tools = Path(tools).absolute()
    tools.mkdir(parents=True, exist_ok=True)
    requests = collect_dino_requests(pipeline)
    repo, source = tools / "dinov2", SOURCES["dinov2"]
    def git(*args):
        subprocess.run(["git", *map(str, args)], check=True)
    if not repo.exists():
        git("init", repo)
        git("-C", repo, "remote", "add", "origin", source["url"])
        git("-C", repo, "fetch", "--depth=1", "origin", source["commit"])
        git("-C", repo, "checkout", "--detach", "FETCH_HEAD")
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual != source["commit"]:
        raise ValueError(f"Unexpected DINO revision in {repo}: {actual}")
    cache = tools / "torch-cache/hub"
    torch.hub.set_dir(str(cache))
    for request in requests:
        print("Preparing DINO on CPU:", json.dumps(request, sort_keys=True), flush=True)
        model = torch.hub.load(str(repo), request["model"], source="local", **request["kwargs"])
        del model
        gc.collect()
    checkpoints = {p.name: file_digest(p) for p in sorted((cache / "checkpoints").glob("*")) if p.is_file()}
    manifest = {"schema_version": 1, "dinov2_commit": actual, "hubconf_sha256": file_digest(repo / "hubconf.py"),
                "pipeline": str(Path(pipeline).absolute()), "requests": requests, "checkpoints": checkpoints}
    # Re-run exact loads with downloads forbidden: confirms the cache is sufficient.
    with offline_torch_hub(tools, prepared=manifest):
        for request in requests:
            model = torch.hub.load("facebookresearch/dinov2", request["model"], **request["kwargs"])
            del model
            gc.collect()
    write_json(tools / "offline-dependencies.json", manifest)
    print("Offline DINO source/weights check passed; no GPU was required.", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tools", type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", type=Path, help="Prior level 0 run containing configs/mesh.json")
    group.add_argument("--pipeline", type=Path, help="SAM3D pipeline YAML")
    a = p.parse_args()
    pipeline = Path(read_json(a.run / "configs/mesh.json")["pipeline_config_path"]) if a.run else a.pipeline
    prepare(pipeline, a.tools)


if __name__ == "__main__":
    main()
