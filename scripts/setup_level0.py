#!/usr/bin/env python3
"""Fresh Linux x86-64 prefixes; curated dependencies, pinned upstream sources.

Online setup is separate from the 1h offline inference job. Native CUDA builds
need a suitable build node and can take considerably longer than one hour.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads((ROOT / "environments/level0-sources.json").read_text())
# These earlier recipes use the same Python, Torch and CUDA builds; fixes only
# correct the packaging pin and discover the Conda CUDA development paths.
COMPATIBLE_PREVIOUS_RECIPES = {
    "1c53e4d59bbed3bc78182dedc22cddf094945d66ae5c7fd0b9812eaeb86b53cd",  # Same native builds; Blender moved to an optional installer.
    "d74c1bdc84cd85a2812b6c13ae18ac5f109eccb39a4df6e2bfa19bafedfc586a",
    "f35576a67df7326906b34aabaff89fb9d02024d39530573d87beb0dc9a01002c",
}


def claim_prefixes(prefixes, recipe, resume):
    markers = []
    for prefix in prefixes:
        marker = prefix.with_name(prefix.name + ".assetforge-recipe")
        previous = marker.read_text().strip() if marker.is_file() else None
        if prefix.exists() and (not resume or previous not in {recipe, *COMPATIBLE_PREVIOUS_RECIPES}):
            raise ValueError(f"Refusing unmanaged/existing prefix {prefix}; choose ASSETFORGE_ENV_ROOT or use --resume for this recipe")
        markers.append(marker)
    # Validate both prefixes before changing either ownership marker.
    for marker in markers:
        marker.write_text(recipe)


def run(command, **kwargs):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, **kwargs)


def configure_cuda_paths(prefix, environment):
    """Expose Conda's split CUDA layout to the host compiler as well as nvcc."""
    target = prefix / "targets/x86_64-linux"
    candidates = [target / "include", prefix / "include"]
    includes = [folder for folder in candidates if folder.is_dir()]
    if not any((folder / "cuda_runtime_api.h").is_file() for folder in includes):
        raise FileNotFoundError("CUDA development header cuda_runtime_api.h missing; searched " +
                                ", ".join(map(str, candidates)) +
                                ". The backend prefix needs the CUDA 12.1 development toolkit.")
    libraries = [folder for folder in (target / "lib", prefix / "lib64", prefix / "lib") if folder.is_dir()]
    def prepend(key, folders):
        old = environment.get(key, "").split(os.pathsep)
        paths = list(dict.fromkeys([*map(str, folders), *filter(None, old)]))
        environment[key] = os.pathsep.join(paths)
    # PyTorch 2.5 CUDAExtension adds CUDA_HOME/include, but the host C++ compiler
    # does not automatically search targets/x86_64-linux/include like nvcc does.
    for key in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
        prepend(key, includes)
    for key in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
        prepend(key, libraries)
    for folder in includes:
        if (folder / "cub").is_dir():
            environment["CUB_HOME"] = str(folder)
            break
    print("CUDA development includes:", ", ".join(map(str, includes)), flush=True)
    print("CUDA development libraries:", ", ".join(map(str, libraries)), flush=True)


def check_cuda_toolchain(tools):
    """Compile/link only: no GPU, NVIDIA driver, or cuda.is_available() needed."""
    source = tools / "cuda-build-probe.cpp"
    source.write_text("#include <cuda_runtime_api.h>\n"
                      "int main() { int v = 0; return (int)cudaRuntimeGetVersion(&v); }\n")
    run([os.environ["CXX"], source, "-lcudart", "-o", tools / "cuda-build-probe"])
    # Deliberately do not execute this binary on the build/front node.
    print("CUDA header and host-link check passed; GPU execution deferred to the A100 job.", flush=True)


def download(url, target):
    if not target.is_file():
        partial = target.with_name(target.name + ".partial")
        urllib.request.urlretrieve(url, partial)
        partial.replace(target)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def extract(archive, destination, *, strip=0):
    destination.mkdir(parents=True, exist_ok=True)
    links = []
    with tarfile.open(archive) as source:
        # Blender ships shared-library symlinks. Keep only links within the destination.
        for member in source.getmembers():
            parts = Path(member.name).parts[strip:]
            if not parts or member.isdir():
                continue
            target = destination.joinpath(*parts)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError("Unsafe archive member: " + member.name)
            if member.issym():
                if not (target.parent / member.linkname).resolve().is_relative_to(destination.resolve()):
                    raise ValueError("Unsafe archive symlink: " + member.name)
                links.append((target, member.linkname))
                continue
            if not member.isfile():
                raise ValueError("Unsupported archive member: " + member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as incoming, target.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            target.chmod(member.mode & 0o777)
    for target, link in links:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()
        target.symlink_to(link)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-root", type=Path, required=True)
    p.add_argument("--mamba", help="Existing micromamba/mamba/conda executable; otherwise bootstrap micromamba")
    p.add_argument("--resume", action="store_true", help="Resume only prefixes created by this recipe")
    a = p.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        p.error("Setup requires Linux x86_64; A100 execution is validated, other GPUs are untested")
    if sys.version_info < (3, 9):
        p.error("Load a Python >=3.9 module to bootstrap setup; managed environments use Python 3.11")
    main_env = a.env_root.absolute()
    backend = main_env.with_name(main_env.name + "_sam3d")
    tools = main_env.parent / "assetforge-tools"
    tools.mkdir(parents=True, exist_ok=True)
    recipe = hashlib.sha256(b"".join((ROOT / "environments" / name).read_bytes()
        for name in ("level0-image.txt", "level0-sam3d.txt", "level0-sources.json")) + (Path(__file__).read_bytes() + os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0").encode())).hexdigest()
    try:
        claim_prefixes((main_env, backend), recipe, a.resume)
    except ValueError as exc:
        p.error(str(exc))
    # A resumed install must pass all probes again before it can be used.
    (main_env / "setup-complete.json").unlink(missing_ok=True)
    cache = tools / "downloads"
    cache.mkdir(exist_ok=True)
    os.environ["PIP_CACHE_DIR"] = str(tools / "pip-cache")
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ.pop("PYTHONPATH", None)
    os.environ.pop("PYTHONHOME", None)
    os.environ["MAMBA_ROOT_PREFIX"] = str(tools / "mamba-root")
    os.environ["TORCH_EXTENSIONS_DIR"] = str(tools / "torch-extensions")
    mamba = a.mamba or shutil.which("micromamba") or shutil.which("mamba") or shutil.which("conda")
    downloads = {}
    if not mamba:
        archive = cache / "micromamba.tar.bz2"
        downloads[archive.name] = download("https://micro.mamba.pm/api/micromamba/linux-64/" + SOURCES["micromamba_version"], archive)
        extract(archive, tools / "micromamba")
        mamba = str(tools / "micromamba/bin/micromamba")
    for prefix, packages in ((main_env, ["python=3.11", "pip", "libgl", "libegl", "libxkbcommon", "xorg-libx11", "xorg-libxi", "xorg-libxrender"]),
                              (backend, ["python=3.11", "pip", "cuda-toolkit=12.1", "gcc_linux-64=12", "gxx_linux-64=12", "cmake=3.31", "ninja"])):
        # Install into a directory containing only our marker on the first run.
        if not (prefix / "conda-meta").is_dir():
            run([mamba, "create", "-y", "--prefix", prefix, "--override-channels", "-c", "conda-forge", "-c", "nvidia", *packages])
    main_python, python = main_env / "bin/python", backend / "bin/python"
    def pip(interpreter, *args):
        run([interpreter, "-m", "pip", *args])
    for interpreter in (main_python, python):
        pip(interpreter, "install", "pip==25.3", "setuptools==75.8.0", "wheel==0.45.1")
    pip(main_python, "install", "torch==2.7.1", "torchvision==0.22.1", "--index-url", "https://download.pytorch.org/whl/cu126")
    pip(main_python, "install", "-r", ROOT / "environments/level0-image.txt")
    pip(python, "install", "torch==2.5.1", "torchvision==0.20.1", "--index-url", "https://download.pytorch.org/whl/cu121")
    # Constraints stop native packages from silently upgrading the Torch/CUDA pair.
    constraints = tools / "sam3d-constraints.txt"
    constraints.write_text("torch==2.5.1+cu121\ntorchvision==0.20.1+cu121\nnumpy==1.26.4\n")
    pip(python, "install", "-c", constraints, "-r", ROOT / "environments/level0-sam3d.txt")
    pip(python, "install", "--no-deps", "xformers==0.0.28.post3")
    pip(python, "install", "-c", constraints, "kaolin==0.17.0", "--find-links", "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html")
    os.environ.update(CONDA_PREFIX=str(backend), CUDA_HOME=str(backend), FORCE_CUDA="1",
                      TORCH_CUDA_ARCH_LIST=os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0"), MAX_JOBS=os.environ.get("MAX_JOBS", "4"),
                      CC=str(backend / "bin/x86_64-conda-linux-gnu-cc"),
                      CXX=str(backend / "bin/x86_64-conda-linux-gnu-c++"))
    os.environ["PATH"] = str(backend / "bin") + os.pathsep + os.environ["PATH"]
    configure_cuda_paths(backend, os.environ)
    check_cuda_toolchain(tools)
    locations = {}
    for name in ("sam3d", "utils3d", "moge", "pytorch3d", "gsplat"):
        spec = SOURCES[name]
        folder = tools / ("sam-3d-objects" if name == "sam3d" else name)
        if not folder.exists():
            run(["git", "init", folder])
            run(["git", "-C", folder, "remote", "add", "origin", spec["url"]])
            run(["git", "-C", folder, "fetch", "--depth=1", "origin", spec["commit"]])
            run(["git", "-C", folder, "checkout", "--detach", "FETCH_HEAD"])
        actual = subprocess.check_output(["git", "-C", str(folder), "rev-parse", "HEAD"], text=True).strip()
        if actual != spec["commit"]:
            raise ValueError(f"Wrong upstream revision in {folder}: {actual}")
        run(["git", "-C", folder, "submodule", "update", "--init", "--recursive"])
        locations[name] = folder
        if name != "sam3d":
            pip(python, "install", "--no-deps", "--no-build-isolation", folder)
    pip(python, "install", "--no-build-isolation", "--no-deps", "flash-attn==2.8.3")
    # Keep upstream importable without invoking its broad, outdated package requirements.
    site = subprocess.check_output([str(python), "-c", "import site; print(site.getsitepackages()[0])"], text=True).strip()
    Path(site, "assetforge-sam3d.pth").write_text(str(locations["sam3d"]) + "\n")
    patch = cache / "hydra-utils.py"
    downloads[patch.name] = download(SOURCES["hydra_patch"], patch)
    hydra_root = subprocess.check_output([str(python), "-c", "import hydra,pathlib; print(pathlib.Path(hydra.__file__).parent)"], text=True).strip()
    shutil.copy2(patch, Path(hydra_root) / "core/utils.py")
    run([main_python, "-c", "import torch,diffusers,accelerate,yaml; from transformers import Sam3Model,Sam3Processor; print(torch.__version__)"])
    run([python, "-c", "import sys,runpy; sys.path.insert(0,sys.argv[1]); runpy.run_path(sys.argv[1]+'/inference.py'); import flash_attn,gsplat,kaolin,spconv.pytorch; print('SAM3D import closure OK')", locations["sam3d"] / "notebook"])
    for prefix, interpreter in ((main_env, main_python), (backend, python)):
        run([interpreter, "-m", "pip", "check"])
        with (prefix / "pip-freeze.txt").open("w") as target:
            run([interpreter, "-m", "pip", "freeze", "--all"], stdout=target)
        with (prefix / "conda-explicit.txt").open("w") as target:
            run([mamba, "list", "--prefix", prefix, "--explicit"], stdout=target)
    (main_env / "setup-complete.json").write_text(json.dumps({"recipe_sha256": recipe, "sources": SOURCES,
        "download_sha256": downloads, "sam3d_prefix": str(backend), "gpu_inference_tested": False}, indent=2))
    print("Setup import probes passed. A100 kernel/model validation happens in the level 0 job.")


if __name__ == "__main__":
    main()
