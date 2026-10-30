"""Offline image-or-prompt to 3D asset generation pipeline.

The text-to-image module is imported only when a text prompt is used. This is
intentional: image-only runs must not load Diffusers or its dependencies.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path(__file__).with_name(
    "asset_generation_pipeline_config.json"
)


class PipelineError(RuntimeError):
    """Raised when the pipeline cannot produce any 3D asset."""


class ConfigurationError(ValueError):
    """Raised when the pipeline configuration is invalid."""


@dataclass(frozen=True)
class AssetPipelineResult:
    """Artifacts produced by the complete asset generation pipeline."""

    image_path: Path
    mask_paths: Mapping[str, Path]
    asset_paths: Mapping[str, Path]
    generated_image: bool
    prompt: str | None
    image_prompt: str | None
    failures: tuple[str, ...]
    mesh_paths: Mapping[str, Path] = field(default_factory=dict)

    @property
    def asset_path(self) -> Path:
        """Return the first generated asset for single-output consumers."""

        return next(iter(self.asset_paths.values()))

    @property
    def mask_path(self) -> Path:
        """Return the mask associated with the first generated asset."""

        return self.mask_paths[next(iter(self.asset_paths))]


def _load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline configuration not found: {path}")

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
    must_exist: bool,
    directory: bool = False,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{label} is not configured.")

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    path = path.resolve()

    if must_exist:
        exists = path.is_dir() if directory else path.is_file()
        if not exists:
            expected = "directory" if directory else "file"
            raise ConfigurationError(
                f"{label} must point to an existing {expected}: {path}"
            )
    return path


def _stage_directory(
    config: Mapping[str, Any],
    key: str,
    root: Path,
    default: str,
) -> Path:
    value = config.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{key} must be a non-empty path string.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _select_input(
    image_path: str | Path | None,
    prompt: str | None,
) -> tuple[Path | None, str | None]:
    normalized_prompt = prompt.strip() if prompt and prompt.strip() else None
    if (image_path is None) == (normalized_prompt is None):
        raise ValueError("Provide exactly one of image_path or prompt.")

    if image_path is None:
        return None, normalized_prompt

    source_image = Path(image_path).expanduser().resolve()
    if not source_image.is_file():
        raise FileNotFoundError(f"Input image not found: {source_image}")
    return source_image, None


def generate_asset_pipeline(
    *,
    image_path: str | Path | None = None,
    prompt: str | None = None,
    image_prompt: str | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> AssetPipelineResult:
    """Generate one or more 3D assets from an image or text prompt.

    Args:
        image_path: Existing source image. Mutually exclusive with prompt.
        prompt: Text used to generate the source image.
        image_prompt: Optional concept prompt enabling SAM 3 segmentation.
        config_path: Pipeline JSON containing child configs and output paths.

    Returns:
        All successful masks and their corresponding 3D assets. The SAM mask is
        attempted in every run; SAM 3 is additionally useful with image_prompt.
    """

    source_image, normalized_prompt = _select_input(image_path, prompt)
    normalized_image_prompt = (
        image_prompt.strip()
        if image_prompt and image_prompt.strip()
        else None
    )
    config, config_dir = _load_config(config_path)

    mask_config = _resolve_path(
        config.get("image_to_mask_config"),
        "image_to_mask_config",
        config_dir,
        must_exist=True,
    )
    asset_config = _resolve_path(
        config.get("image_mask_to_3d_config"),
        "image_mask_to_3d_config",
        config_dir,
        must_exist=True,
    )
    output_root = _resolve_path(
        config.get("output_root", "./pipeline_outputs"),
        "output_root",
        config_dir,
        must_exist=False,
        directory=True,
    )
    image_output_dir = _stage_directory(
        config, "image_output_directory", output_root, "images"
    )
    mask_output_dir = _stage_directory(
        config, "mask_output_directory", output_root, "masks"
    )
    asset_output_dir = _stage_directory(
        config, "asset_output_directory", output_root, "assets"
    )

    generated_image = source_image is None
    if generated_image:
        # Keep this import inside the prompt branch. Importing this module in an
        # image-only process must not import Diffusers or prompt_to_image.
        from prompt_to_image import generate_image

        prompt_config = _resolve_path(
            config.get("prompt_to_image_config"),
            "prompt_to_image_config",
            config_dir,
            must_exist=True,
        )
        image_result = generate_image(
            normalized_prompt or "",
            config_path=prompt_config,
            output_dir=image_output_dir,
            seed=(
                int(config["image_seed"])
                if config.get("image_seed") is not None
                else None
            ),
        )
        source_image = image_result.image_path

    assert source_image is not None

    # These modules are also imported at execution time to keep importing this
    # orchestrator itself lightweight.
    from image_to_mask import generate_mask

    mask_result = generate_mask(
        source_image,
        text_prompt=normalized_image_prompt,
        config_path=mask_config,
        output_dir=mask_output_dir,
        backend="sam3" if normalized_image_prompt else "sam",
    )

    from image_mask_to_3d import generate_3d_asset

    asset_paths: dict[str, Path] = {}
    mesh_paths: dict[str, Path] = {}
    failures = list(mask_result.failures)
    asset_seed = (
        int(config["asset_seed"])
        if config.get("asset_seed") is not None
        else None
    )
    for backend_name, mask_path_value in mask_result.mask_paths.items():
        try:
            asset_result = generate_3d_asset(
                source_image,
                mask_path_value,
                config_path=asset_config,
                output_dir=asset_output_dir,
                seed=asset_seed,
            )
            asset_paths[backend_name] = asset_result.asset_path
            if asset_result.mesh_path is not None:
                mesh_paths[backend_name] = asset_result.mesh_path
        except Exception as exc:
            message = (
                f"{backend_name} 3D generation: "
                f"{type(exc).__name__}: {exc}"
            )
            failures.append(message)
            LOGGER.warning(
                "3D generation failed for one mask; continuing. %s",
                message,
            )

    if not asset_paths:
        details = "\n".join(f"  - {failure}" for failure in failures)
        raise PipelineError(
            f"No 3D asset was generated from {source_image}:\n{details}"
        )

    return AssetPipelineResult(
        image_path=source_image,
        mask_paths=dict(mask_result.mask_paths),
        asset_paths=asset_paths,
        generated_image=generated_image,
        prompt=normalized_prompt,
        image_prompt=normalized_image_prompt,
        failures=tuple(failures),
        mesh_paths=mesh_paths,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SAM 3D Objects assets from an existing image or a "
            "text-generated image."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Path to an existing source image.")
    source.add_argument(
        "--prompt",
        help="Text prompt used to generate the source image.",
    )
    parser.add_argument(
        "--image-prompt",
        help="Optional object concept used by SAM 3 segmentation.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the asset pipeline JSON configuration.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable informational logging.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    try:
        result = generate_asset_pipeline(
            image_path=args.image,
            prompt=args.prompt,
            image_prompt=args.image_prompt,
            config_path=args.config,
        )
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1

    for backend_name, asset_path in result.asset_paths.items():
        print(f"{backend_name}: {asset_path}")
        if backend_name in result.mesh_paths:
            print(f"{backend_name} mesh: {result.mesh_paths[backend_name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
