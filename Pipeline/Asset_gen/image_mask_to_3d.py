"""Offline image-and-mask to 3D asset generation with SAM 3D Objects.

This module can be imported by a larger pipeline or run directly as a CLI.
All model code and checkpoint paths are supplied through a local JSON config.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path(__file__).with_name("image_mask_to_3d_config.json")


class AssetGenerationError(RuntimeError):
    """Raised when SAM 3D Objects cannot produce or export an asset."""


class ConfigurationError(ValueError):
    """Raised when the offline model configuration is invalid."""


@dataclass(frozen=True)
class Asset3DResult:
    """A generated asset and the source files used to reconstruct it."""

    asset_path: Path
    image_path: Path
    mask_path: Path
    seed: int
    format: str = "ply"
    mesh_path: Path | None = None


def export_assets(
    output: Mapping[str, Any], destination: Path, *, export_glb: bool = True
) -> Path | None:
    """Save the splat and, by default, the mesh from one inference result."""

    if not isinstance(output, Mapping):
        raise AssetGenerationError("SAM 3D Objects returned a non-mapping output.")
    save_ply = getattr(output.get("gs"), "save_ply", None)
    if not callable(save_ply):
        raise AssetGenerationError("SAM 3D Objects returned no saveable 'gs' asset.")
    mesh_path = destination.with_suffix(".glb") if export_glb else None
    export_mesh = getattr(output.get("glb"), "export", None)
    if export_glb and not callable(export_mesh):
        raise AssetGenerationError(
            "SAM 3D Objects returned no exportable 'glb' mesh. Enable both "
            "'gaussian' and 'mesh' in the upstream pipeline's decode_formats, "
            "or set export_glb=false for PLY-only output."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        save_ply(str(destination))
        if mesh_path is not None:
            export_mesh(str(mesh_path))
    except Exception as exc:
        raise AssetGenerationError(f"SAM 3D export failed: {exc}") from exc
    for path in (destination, mesh_path):
        if path is not None and (not path.is_file() or path.stat().st_size == 0):
            raise AssetGenerationError(f"SAM 3D export did not create a nonempty {path}.")
    return mesh_path


def _load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ConfigurationError("Configuration root must be a JSON object.")
    return config, path.parent


def _resolve_path(
    value: Any,
    label: str,
    config_dir: Path,
    *,
    expect_directory: bool = False,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{label} is not configured.")

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    path = path.resolve()

    exists = path.is_dir() if expect_directory else path.is_file()
    expected = "directory" if expect_directory else "file"
    if not exists:
        raise ConfigurationError(
            f"{label} must point to an existing local {expected}: {path}"
        )
    return path


def _configure_offline_environment() -> None:
    # These are recognized by common Hugging Face dependencies and prevent
    # accidental network access if an upstream dependency changes behavior.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def _load_inputs(
    image_path: Path,
    mask_path: Path,
    mask_threshold: int,
) -> tuple[Any, Any]:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise AssetGenerationError(
            "NumPy and Pillow are required to load image inputs."
        ) from exc

    if not 0 <= mask_threshold <= 255:
        raise ConfigurationError("mask_threshold must be between 0 and 255.")

    with Image.open(image_path) as source_image:
        image = np.asarray(source_image.convert("RGB"), dtype=np.uint8)
    with Image.open(mask_path) as source_mask:
        mask = np.asarray(source_mask.convert("L"), dtype=np.uint8)

    if image.shape[:2] != mask.shape:
        raise ValueError(
            "Image and mask dimensions must match: "
            f"image={image.shape[1]}x{image.shape[0]}, "
            f"mask={mask.shape[1]}x{mask.shape[0]}."
        )

    binary_mask = mask > mask_threshold
    if not binary_mask.any():
        raise ValueError(
            f"Mask contains no foreground pixels above {mask_threshold}."
        )
    if binary_mask.all():
        LOGGER.warning("The mask covers the entire image.")

    return image, binary_mask


def _load_inference_module(
    repository_path: Path,
    inference_module_path: Path,
) -> ModuleType:
    # The official notebook wrapper imports both repository packages and helper
    # modules located beside inference.py.
    for path in (repository_path, inference_module_path.parent):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    module_name = "_local_sam3d_objects_inference"
    spec = importlib.util.spec_from_file_location(
        module_name, inference_module_path
    )
    if spec is None or spec.loader is None:
        raise AssetGenerationError(
            f"Cannot import inference module: {inference_module_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _create_inference(
    repository_path: Path,
    inference_module_path: Path,
    pipeline_config_path: Path,
    compile_model: bool,
) -> Any:
    try:
        module = _load_inference_module(
            repository_path, inference_module_path
        )
    except Exception as exc:
        raise AssetGenerationError(
            "Could not import the local SAM 3D Objects inference wrapper. "
            "Confirm that its offline environment and dependencies are "
            f"installed. Original error: {type(exc).__name__}: {exc}"
        ) from exc

    inference_class = getattr(module, "Inference", None)
    if inference_class is None:
        raise AssetGenerationError(
            f"No Inference class found in {inference_module_path}."
        )

    try:
        return inference_class(
            str(pipeline_config_path),
            compile=compile_model,
        )
    except Exception as exc:
        raise AssetGenerationError(
            "Failed to initialize SAM 3D Objects from local configuration. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


def _output_path(
    image_path: Path,
    mask_path: Path,
    output_dir: str | Path | None,
    output_path: str | Path | None,
    suffix: str,
) -> Path:
    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        if destination.suffix.lower() != ".ply":
            raise ValueError("SAM 3D output_path must use the .ply extension.")
        return destination

    directory = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else image_path.parent
    )
    filename = f"{mask_path.stem}{suffix}.ply"
    return directory / filename


def _resolve_python_executable(
    value: Any,
    config_dir: Path,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(
            "python_executable must be configured when isolate_process is true."
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    path = path.resolve()
    if not path.is_file():
        raise ConfigurationError(f"python_executable does not exist: {path}")
    return path


def _run_isolated(
    python_executable: Path,
    image_path: Path,
    mask_path: Path,
    config_path: Path,
    destination: Path,
    seed: int,
    timeout_seconds: float,
    export_glb: bool = True,
) -> None:
    if timeout_seconds <= 0:
        raise ConfigurationError("timeout_seconds must be greater than zero.")

    temporary_output = destination.with_name(
        f".{destination.name}.sam3d-worker.tmp.ply"
    )
    temporary_output.unlink(missing_ok=True)
    temporary_mesh = temporary_output.with_suffix(".glb")
    temporary_mesh.unlink(missing_ok=True)
    command = [
        str(python_executable),
        str(Path(__file__).resolve()),
        str(image_path),
        str(mask_path),
        "--config",
        str(config_path),
        "--output",
        str(temporary_output),
        "--seed",
        str(seed),
        "--in-process-worker",
    ]
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["HF_DATASETS_OFFLINE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"

    try:
        completed = subprocess.run(
            command,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        temporary_output.unlink(missing_ok=True)
        temporary_mesh.unlink(missing_ok=True)
        raise AssetGenerationError(
            f"Isolated SAM 3D Objects exceeded its "
            f"{timeout_seconds:g}-second timeout."
        ) from exc
    except OSError as exc:
        temporary_output.unlink(missing_ok=True)
        temporary_mesh.unlink(missing_ok=True)
        raise AssetGenerationError(
            f"Could not start isolated SAM 3D Objects process: {exc}"
        ) from exc

    if completed.returncode != 0:
        temporary_output.unlink(missing_ok=True)
        temporary_mesh.unlink(missing_ok=True)
        details = (completed.stderr or completed.stdout).strip()
        if len(details) > 2000:
            details = details[-2000:]
        reason = details or f"worker exited with code {completed.returncode}"
        raise AssetGenerationError(
            f"Isolated SAM 3D Objects failed: {reason}"
        )
    expected = [temporary_output, temporary_mesh] if export_glb else [temporary_output]
    if any(not path.is_file() or path.stat().st_size == 0 for path in expected):
        temporary_output.unlink(missing_ok=True)
        temporary_mesh.unlink(missing_ok=True)
        raise AssetGenerationError(
            "Isolated SAM 3D Objects did not produce all requested nonempty exports."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_output.replace(destination)
    if export_glb:
        temporary_mesh.replace(destination.with_suffix(".glb"))


def generate_3d_asset(
    image_path: str | Path,
    mask_path: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    seed: int | None = None,
    _allow_isolation: bool = True,
) -> Asset3DResult:
    """Generate a Gaussian-splat PLY and GLB mesh from the same inference.

    Args:
        image_path: Source image path.
        mask_path: Binary or grayscale object mask matching the image size.
        config_path: JSON file containing local SAM 3D Objects paths.
        output_dir: Destination directory used for the generated filename.
        output_path: Explicit PLY path; the GLB uses the same stem and directory.
        seed: Optional inference seed overriding the configured default.

    Returns:
        Metadata containing the generated 3D asset path.
    """

    source_image = Path(image_path).expanduser().resolve()
    source_mask = Path(mask_path).expanduser().resolve()
    if not source_image.is_file():
        raise FileNotFoundError(f"Input image not found: {source_image}")
    if not source_mask.is_file():
        raise FileNotFoundError(f"Input mask not found: {source_mask}")

    config_file = Path(config_path).expanduser().resolve()
    config, config_dir = _load_config(config_file)
    export_glb = config.get("export_glb", True)
    if not isinstance(export_glb, bool):
        raise ConfigurationError("export_glb must be a JSON boolean.")
    configured_seed = config.get("seed", 42)
    inference_seed = int(configured_seed if seed is None else seed)
    output_suffix = str(config.get("output_suffix", "_sam3d"))
    destination = _output_path(
        source_image,
        source_mask,
        output_dir,
        output_path,
        output_suffix,
    )

    if _allow_isolation and bool(config.get("isolate_process", False)):
        python_executable = _resolve_python_executable(
            config.get("python_executable"),
            config_dir,
        )
        _run_isolated(
            python_executable,
            source_image,
            source_mask,
            config_file,
            destination,
            inference_seed,
            float(config.get("timeout_seconds", 3600)),
            export_glb=export_glb,
        )
        return Asset3DResult(
            asset_path=destination,
            image_path=source_image,
            mask_path=source_mask,
            seed=inference_seed,
            mesh_path=destination.with_suffix(".glb") if export_glb else None,
        )

    repository_path = _resolve_path(
        config.get("repository_path"),
        "repository_path",
        config_dir,
        expect_directory=True,
    )
    inference_module_path = _resolve_path(
        config.get("inference_module_path"),
        "inference_module_path",
        config_dir,
    )
    pipeline_config_path = _resolve_path(
        config.get("pipeline_config_path"),
        "pipeline_config_path",
        config_dir,
    )

    mask_threshold = int(config.get("mask_threshold", 0))
    compile_model = bool(config.get("compile_model", False))

    _configure_offline_environment()
    image, mask = _load_inputs(
        source_image, source_mask, mask_threshold
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    inference = _create_inference(
        repository_path,
        inference_module_path,
        pipeline_config_path,
        compile_model,
    )
    try:
        output = inference(image, mask, seed=inference_seed)
    except Exception as exc:
        raise AssetGenerationError(
            "SAM 3D Objects inference failed. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    mesh_path = export_assets(output, destination, export_glb=export_glb)

    return Asset3DResult(
        asset_path=destination,
        image_path=source_image,
        mask_path=source_mask,
        seed=inference_seed,
        mesh_path=mesh_path,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SAM 3D Objects PLY and GLB exports from an image "
            "and mask using local model files."
        )
    )
    parser.add_argument("image", help="Path to the source image.")
    parser.add_argument("mask", help="Path to the matching object mask.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the offline SAM 3D Objects JSON configuration.",
    )
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "--output-dir",
        help="Directory for the generated PLY and GLB.",
    )
    destination.add_argument(
        "--output",
        help="Explicit .ply path; the GLB uses the same stem.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Inference seed overriding the configured default.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable informational logging.",
    )
    parser.add_argument(
        "--in-process-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    try:
        result = generate_3d_asset(
            args.image,
            args.mask,
            config_path=args.config,
            output_dir=args.output_dir,
            output_path=args.output,
            seed=args.seed,
            _allow_isolation=not args.in_process_worker,
        )
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1

    print(result.asset_path)
    if result.mesh_path is not None:
        print(result.mesh_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
