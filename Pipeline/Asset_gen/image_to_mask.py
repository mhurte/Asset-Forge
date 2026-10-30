"""Offline image-to-mask inference with SAM and SAM 3.

The module intentionally imports model dependencies only when a backend is
used. This keeps it importable by a larger pipeline even when one backend is
not installed in the current environment. Both backends are attempted so every
available mask is retained; one successful backend is enough to proceed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path(__file__).with_name("image_to_mask_config.json")
SUPPORTED_BACKENDS = ("sam", "sam3")


class MaskGenerationError(RuntimeError):
    """Raised when every requested segmentation backend fails."""


class BackendUnavailableError(RuntimeError):
    """Raised when a backend cannot run with the local configuration."""


@dataclass(frozen=True)
class MaskResult:
    """Result returned to callers integrating this module into a pipeline."""

    mask_paths: Mapping[str, Path]
    attempts: tuple[str, ...]
    failures: tuple[str, ...]

    @property
    def mask_path(self) -> Path:
        """Return the first generated mask for single-mask consumers."""

        return next(iter(self.mask_paths.values()))

    @property
    def backend(self) -> str:
        """Return the backend associated with the first generated mask."""

        return next(iter(self.mask_paths))


def _load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a JSON object.")
    return config


def _require_path(
    value: Any,
    label: str,
    config_dir: Path,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise BackendUnavailableError(f"{label} is not configured.")

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    path = path.resolve()
    if not path.exists():
        raise BackendUnavailableError(f"{label} does not exist: {path}")
    return path


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError as exc:
        raise BackendUnavailableError("PyTorch is not installed.") from exc
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_rgb_image(image_path: Path) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise BackendUnavailableError("Pillow is not installed.") from exc

    with Image.open(image_path) as image:
        return image.convert("RGB")


def _to_numpy_bool(mask: Any) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise BackendUnavailableError("NumPy is not installed.") from exc

    if hasattr(mask, "detach"):
        mask = mask.detach().to("cpu").numpy()
    array = np.asarray(mask)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise RuntimeError(f"Expected a 2D mask, received shape {array.shape}.")
    return array.astype(bool)


def _run_sam(
    image_path: Path,
    backend_config: Mapping[str, Any],
    config_dir: Path,
    device: str,
) -> Any:
    model_path = _require_path(
        backend_config.get("checkpoint_path"),
        "sam.checkpoint_path",
        config_dir,
    )

    try:
        import numpy as np
        from transformers import pipeline
    except ImportError as exc:
        raise BackendUnavailableError(
            "Transformers is required for SAM mask generation."
        ) from exc

    pipeline_device: int | str = -1
    if device.startswith("cuda"):
        pipeline_device = 0 if device == "cuda" else device

    try:
        generator = pipeline(
            "mask-generation",
            model=str(model_path),
            image_processor=str(model_path),
            device=pipeline_device,
            local_files_only=True,
        )
    except Exception as exc:
        raise BackendUnavailableError(
            "Could not load SAM from local Hugging Face files. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    generator_options = backend_config.get("automatic_mask_generator", {})
    if not isinstance(generator_options, dict):
        raise ValueError("sam.automatic_mask_generator must be a JSON object.")
    allowed_options = {
        "points_per_batch",
        "mask_threshold",
        "pred_iou_thresh",
        "stability_score_thresh",
        "stability_score_offset",
        "crops_nms_thresh",
        "crops_n_layers",
        "crop_overlap_ratio",
        "crop_n_points_downscale_factor",
    }
    call_options = {
        key: value
        for key, value in generator_options.items()
        if key in allowed_options
    }

    try:
        output = generator(_load_rgb_image(image_path), **call_options)
    except Exception as exc:
        raise RuntimeError(
            "SAM automatic mask generation failed. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(output, Mapping):
        raise RuntimeError("SAM returned a non-mapping output.")

    masks = output.get("masks")
    if masks is None:
        mask = output.get("mask")
        masks = [] if mask is None else [mask]
    if len(masks) == 0:
        raise RuntimeError("SAM returned no masks.")

    scores = output.get("scores")
    if scores is None:
        scores = output.get("score")
    if scores is None or isinstance(scores, (float, int)):
        scores = [float(scores or 0.0)] * len(masks)

    image = np.asarray(_load_rgb_image(image_path))
    image_area = image.shape[0] * image.shape[1]
    max_area_ratio = float(backend_config.get("max_area_ratio", 0.9))

    candidates = []
    for index, mask in enumerate(masks):
        array = _to_numpy_bool(mask)
        area = int(array.sum())
        if 0 < area <= image_area * max_area_ratio:
            score = float(scores[index]) if index < len(scores) else 0.0
            candidates.append((array, area, score))
    if not candidates:
        for index, mask in enumerate(masks):
            array = _to_numpy_bool(mask)
            area = int(array.sum())
            if area > 0:
                score = float(scores[index]) if index < len(scores) else 0.0
                candidates.append((array, area, score))
    if not candidates:
        raise RuntimeError("SAM returned only empty masks.")

    combination = backend_config.get("mask_combination", "union")
    if combination == "largest":
        selected, _, _ = max(candidates, key=lambda item: (item[1], item[2]))
        return selected
    if combination != "union":
        raise ValueError("sam.mask_combination must be 'union' or 'largest'")
    combined = np.zeros_like(candidates[0][0], dtype=bool)
    for candidate, _, _ in candidates:
        np.logical_or(combined, candidate, out=combined)
    return combined


def _run_backend_isolated(
    backend_name: str,
    image_path: Path,
    output_path: Path,
    config_path: Path,
    backend_config: Mapping[str, Any],
    requested_device: str,
    text_prompt: str | None = None,
) -> None:
    python_value = backend_config.get("python_executable")
    if python_value in (None, ""):
        python_executable = Path(sys.executable).resolve()
    else:
        python_executable = Path(str(python_value)).expanduser()
        if not python_executable.is_absolute():
            python_executable = config_path.parent / python_executable
        python_executable = python_executable.resolve()
        if not python_executable.is_file():
            raise BackendUnavailableError(
                f"{backend_name}.python_executable does not exist: "
                f"{python_executable}"
            )

    timeout_seconds = float(backend_config.get("timeout_seconds", 600))
    if timeout_seconds <= 0:
        raise ValueError("sam.timeout_seconds must be greater than zero.")

    temporary_output = output_path.with_name(
        f".{output_path.name}.sam-worker.tmp"
    )
    temporary_output.unlink(missing_ok=True)
    command = [
        str(python_executable),
        str(Path(__file__).resolve()),
        "--mask-worker",
        "--worker-backend",
        backend_name,
        "--worker-image",
        str(image_path),
        "--worker-output",
        str(temporary_output),
        "--config",
        str(config_path),
        "--worker-device",
        requested_device,
    ]
    if text_prompt:
        command.extend(["--worker-text", text_prompt])
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
        raise BackendUnavailableError(
            f"Isolated {backend_name} exceeded its "
            f"{timeout_seconds:g}-second timeout."
        ) from exc
    except OSError as exc:
        temporary_output.unlink(missing_ok=True)
        raise BackendUnavailableError(
            f"Could not start isolated {backend_name} process: {exc}"
        ) from exc

    if completed.returncode != 0:
        temporary_output.unlink(missing_ok=True)
        details = (completed.stderr or completed.stdout).strip()
        if len(details) > 4000:
            details = details[-4000:]
        reason = details or f"worker exited with code {completed.returncode}"
        raise BackendUnavailableError(
            f"Isolated {backend_name} failed: {reason}"
        )
    if not temporary_output.is_file():
        raise BackendUnavailableError(
            f"Isolated {backend_name} exited successfully without producing "
            "a mask."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output.replace(output_path)


def _run_sam3(
    image_path: Path,
    text_prompt: str | None,
    backend_config: Mapping[str, Any],
    config_dir: Path,
    device: str,
) -> Any:
    if not text_prompt:
        raise BackendUnavailableError(
            "SAM 3 requires a short text prompt for concept segmentation."
        )

    model_path = _require_path(
        backend_config.get("checkpoint_path"),
        "sam3.checkpoint_path",
        config_dir,
    )

    try:
        import numpy as np
        import torch
        import transformers
    except ImportError as exc:
        raise BackendUnavailableError(
            "Transformers is required for SAM 3 mask generation."
        ) from exc

    try:
        if hasattr(transformers, "Sam3Model") and hasattr(
            transformers, "Sam3Processor"
        ):
            model_class = transformers.Sam3Model
            processor_class = transformers.Sam3Processor
        else:
            if not hasattr(transformers, "AutoModelForMultimodalLM"):
                raise BackendUnavailableError(
                    "The installed Transformers package is too old for SAM 3: "
                    f"version {transformers.__version__} has neither "
                    "Sam3Model nor AutoModelForMultimodalLM."
                )
            model_class = transformers.AutoModelForMultimodalLM
            processor_class = transformers.AutoProcessor

        model = model_class.from_pretrained(str(model_path), local_files_only=True)
        processor = processor_class.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
    except Exception as exc:
        raise BackendUnavailableError(
            "Could not load SAM 3 from local Hugging Face files. "
            f"Transformers version: {transformers.__version__}. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    model.to(device)
    model.eval()

    image = _load_rgb_image(image_path)
    try:
        inputs = processor(images=image, text=text_prompt, return_tensors="pt")
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=float(backend_config.get("confidence_threshold", 0.5)),
            mask_threshold=float(backend_config.get("mask_threshold", 0.5)),
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]
    except Exception as exc:
        raise RuntimeError(
            "SAM 3 Transformers inference failed. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    masks = results.get("masks")
    if masks is None or len(masks) == 0:
        raise RuntimeError(f"SAM 3 found no instances for prompt: {text_prompt!r}")

    # A concept prompt may match multiple instances. The output mask represents
    # their union so downstream code receives one binary asset mask.
    arrays = [_to_numpy_bool(mask) for mask in masks]
    return np.logical_or.reduce(arrays)


def _save_mask(mask: Any, output_path: Path) -> None:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise BackendUnavailableError(
            "NumPy and Pillow are required to save masks."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")
    image.save(output_path, format="PNG")


def _backend_order(
    config: Mapping[str, Any],
    text_prompt: str | None,
    backend: str | None,
) -> list[str]:
    if backend:
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend: {backend}")
        fallback = [name for name in SUPPORTED_BACKENDS if name != backend]
        return [backend, *fallback]

    configured = config.get("backend_order")
    if configured is not None:
        if (
            not isinstance(configured, list)
            or not configured
            or any(name not in SUPPORTED_BACKENDS for name in configured)
        ):
            raise ValueError(
                "backend_order must be a non-empty list containing 'sam' "
                "and/or 'sam3'."
            )
        ordered = list(dict.fromkeys(configured))
        ordered.extend(name for name in SUPPORTED_BACKENDS if name not in ordered)
        return ordered

    return ["sam3", "sam"] if text_prompt else ["sam", "sam3"]


def _generate_backend_mask(backend_name, source, prompt, backend_config,
                           config_file, requested_device, output_path):
    isolated = bool(backend_config.get("isolate_process", backend_name == "sam"))
    if isolated:
        _run_backend_isolated(backend_name, source, output_path, config_file,
                              backend_config, requested_device, prompt)
    else:
        device = _resolve_device(requested_device)
        mask = (_run_sam(source, backend_config, config_file.parent, device)
                if backend_name == "sam" else
                _run_sam3(source, prompt, backend_config, config_file.parent, device))
        _save_mask(mask, output_path)
    # Reject empty, corrupt or incorrectly sized outputs before accepting success.
    from PIL import Image
    with Image.open(source) as image, Image.open(output_path) as mask:
        if mask.size != image.size or not mask.convert("L").getbbox():
            raise RuntimeError("Mask is empty or does not match the source image dimensions")


def generate_mask(
    image_path: str | Path,
    text_prompt: str | None = None,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path | None = None,
    backend: str | None = None,
    max_attempts: int | None = None,
) -> MaskResult:
    """Generate every possible local SAM and SAM 3 PNG mask.

    Args:
        image_path: Input raster image.
        text_prompt: Optional short noun phrase used by SAM 3.
        config_path: Offline model configuration JSON.
        output_dir: Output directory; defaults to the input image directory.
        backend: Optional backend to try first ("sam" or "sam3").
        max_attempts: Total attempts per enabled backend; config default is 3.

    Returns:
        A result containing all successfully generated backend mask paths.
        The compatibility properties ``mask_path`` and ``backend`` expose the
        first successful result for callers that only consume one mask.
    """

    source = Path(image_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input image not found: {source}")

    prompt = text_prompt.strip() if text_prompt and text_prompt.strip() else None
    config_file = Path(config_path).expanduser().resolve()
    config = _load_config(config_file)
    limit = config.get("max_attempts", 3) if max_attempts is None else max_attempts
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("max_attempts must be a positive integer")
    config_dir = config_file.parent
    requested_device = str(config.get("device", "auto"))
    destination_dir = (
        Path(output_dir).expanduser().resolve() if output_dir else source.parent
    )

    # These flags are also honored by common model dependencies. They prevent
    # accidental hub calls if a dependency changes its internal defaults.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    attempts: list[str] = []
    failures: list[str] = []
    mask_paths: dict[str, Path] = {}
    for backend_name in _backend_order(config, prompt, backend):
        attempts.append(backend_name)
        backend_config = config.get(backend_name, {})
        if not isinstance(backend_config, dict):
            failures.append(f"{backend_name}: configuration must be an object")
            continue
        if backend_config.get("enabled", True) is False:
            failures.append(f"{backend_name}: disabled in configuration")
            continue

        output_path = destination_dir / f"{source.stem}_mask_{backend_name}.png"
        for attempt in range(1, limit + 1):
            try:
                output_path.unlink(missing_ok=True)
                LOGGER.warning("Mask backend %s: attempt %d/%d", backend_name, attempt, limit)
                _generate_backend_mask(backend_name, source, prompt, backend_config,
                                       config_file, requested_device, output_path)
                mask_paths[backend_name] = output_path
                break
            except Exception as exc:
                output_path.unlink(missing_ok=True)
                message = f"{backend_name} attempt {attempt}/{limit}: {type(exc).__name__}: {exc}"
                failures.append(message)
                LOGGER.warning("Mask attempt failed. %s", message)

    if mask_paths:
        return MaskResult(
            mask_paths=dict(mask_paths),
            attempts=tuple(attempts),
            failures=tuple(failures),
        )

    details = "\n".join(f"  - {failure}" for failure in failures)
    raise MaskGenerationError(
        f"No mask backend succeeded for {source}:\n{details}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a binary image mask using offline SAM/SAM 3."
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to the input image.",
    )
    parser.add_argument(
        "--text",
        help="Optional short noun phrase for SAM 3, e.g. 'red chair'.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the local-only JSON configuration.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (defaults to the input image directory).",
    )
    parser.add_argument(
        "--backend",
        choices=SUPPORTED_BACKENDS,
        help="Backend to try first; both are still attempted.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable informational logging."
    )
    parser.add_argument("--sam-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mask-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-backend",
        choices=SUPPORTED_BACKENDS,
        default="sam",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-image", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    parser.add_argument("--worker-device", default="auto", help=argparse.SUPPRESS)
    parser.add_argument("--worker-text", help=argparse.SUPPRESS)
    return parser


def _run_mask_worker(args: argparse.Namespace) -> int:
    if not args.worker_image or not args.worker_output:
        LOGGER.error("Mask worker requires image and output paths.")
        return 2

    try:
        config_file = Path(args.config).expanduser().resolve()
        config = _load_config(config_file)
        backend_name = str(args.worker_backend)
        backend_config = config.get(backend_name, {})
        if not isinstance(backend_config, dict):
            raise ValueError(f"{backend_name} configuration must be an object.")
        image_path = Path(args.worker_image).expanduser().resolve()
        device = _resolve_device(str(args.worker_device))
        if backend_name == "sam":
            mask = _run_sam(
                image_path,
                backend_config,
                config_file.parent,
                device,
            )
        else:
            mask = _run_sam3(
                image_path,
                args.worker_text,
                backend_config,
                config_file.parent,
                device,
            )
        _save_mask(
            mask,
            Path(args.worker_output).expanduser().resolve(),
        )
    except Exception as exc:
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    if args.sam_worker or args.mask_worker:
        return _run_mask_worker(args)
    if not args.image:
        _build_parser().error("the following arguments are required: image")

    try:
        result = generate_mask(
            args.image,
            args.text,
            config_path=args.config,
            output_dir=args.output_dir,
            backend=args.backend,
        )
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1

    for backend_name, mask_path in result.mask_paths.items():
        print(f"{backend_name}: {mask_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
