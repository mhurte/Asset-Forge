"""Offline text-to-image generation using local FLUX weights.

The module can be imported by a larger pipeline or run directly as a CLI.
Every filesystem location is supplied through prompt_to_image_config.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path(__file__).with_name("prompt_to_image_config.json")


class ImageGenerationError(RuntimeError):
    """Raised when the local text-to-image pipeline cannot generate an image."""


class ConfigurationError(ValueError):
    """Raised when the offline model configuration is invalid."""


@dataclass(frozen=True)
class ImageGenerationResult:
    """Generated image metadata returned to pipeline callers."""

    image_path: Path
    prompt: str
    seed: int


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
    required: bool,
    expect_directory: bool = True,
) -> Path | None:
    if value in (None, ""):
        if required:
            raise ConfigurationError(f"{label} is not configured.")
        return None
    if not isinstance(value, str):
        raise ConfigurationError(f"{label} must be a path string or null.")

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    path = path.resolve()

    exists = path.is_dir() if expect_directory else path.is_file()
    if not exists:
        expected = "directory" if expect_directory else "file"
        raise ConfigurationError(
            f"{label} must point to an existing local {expected}: {path}"
        )
    return path


def _resolve_output_directory(value: Any, config_dir: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError("default_output_directory is not configured.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def _configure_offline_environment(cache_path: Path | None) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    if cache_path is not None:
        os.environ["HF_HOME"] = str(cache_path)


def _add_code_path(code_path: Path | None) -> None:
    if code_path is not None and str(code_path) not in sys.path:
        sys.path.insert(0, str(code_path))


def _resolve_device(requested: str, torch_module: Any) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def _resolve_dtype(value: Any, torch_module: Any) -> Any:
    dtype_name = str(value)
    supported = {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }
    if dtype_name not in supported:
        raise ConfigurationError(
            "torch_dtype must be 'bfloat16', 'float16', or 'float32'."
        )
    return supported[dtype_name]


def _load_pipeline(
    model_path: Path,
    code_path: Path | None,
    cache_path: Path | None,
    config: Mapping[str, Any],
) -> tuple[Any, Any, str]:
    _configure_offline_environment(cache_path)
    _add_code_path(code_path)

    try:
        import torch
        from diffusers import FluxPipeline
    except ImportError as exc:
        raise ImageGenerationError(
            "PyTorch, Transformers, Accelerate, and a Diffusers version with "
            "FluxPipeline must be installed in the offline environment."
        ) from exc

    device = _resolve_device(str(config.get("device", "auto")), torch)
    torch_dtype = _resolve_dtype(
        config.get("torch_dtype", "bfloat16"),
        torch,
    )
    load_options: dict[str, Any] = {
        "local_files_only": True,
        "torch_dtype": torch_dtype,
    }
    if cache_path is not None:
        load_options["cache_dir"] = str(cache_path)
    if config.get("variant") is not None:
        load_options["variant"] = str(config["variant"])
    if config.get("use_safetensors") is not None:
        load_options["use_safetensors"] = bool(config["use_safetensors"])

    try:
        pipeline = FluxPipeline.from_pretrained(
            str(model_path),
            **load_options,
        )
        memory_strategy = str(
            config.get("memory_strategy", "model_cpu_offload")
        )
        if device == "cuda" and memory_strategy == "model_cpu_offload":
            pipeline.enable_model_cpu_offload()
        elif device == "cuda" and memory_strategy == "sequential_cpu_offload":
            pipeline.enable_sequential_cpu_offload()
        elif memory_strategy == "direct":
            pipeline.to(device)
        else:
            raise ConfigurationError(
                "memory_strategy must be 'model_cpu_offload', "
                "'sequential_cpu_offload', or 'direct'. CPU execution "
                "requires 'direct'."
            )

        if bool(config.get("enable_vae_slicing", True)):
            pipeline.vae.enable_slicing()
        if bool(config.get("enable_vae_tiling", True)):
            pipeline.vae.enable_tiling()
        pipeline.set_progress_bar_config(
            disable=not bool(config.get("show_progress", False))
        )
    except Exception as exc:
        raise ImageGenerationError(
            "Failed to load the local Diffusers pipeline. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    return pipeline, torch, device


def _validated_inference_options(config: Mapping[str, Any]) -> dict[str, Any]:
    options = config.get("inference_options", {})
    if not isinstance(options, dict):
        raise ConfigurationError("inference_options must be a JSON object.")

    call_options = dict(options)
    call_options.setdefault("height", 1024)
    call_options.setdefault("width", 1024)
    call_options.setdefault("num_inference_steps", 4)
    call_options.setdefault("guidance_scale", 0.0)
    call_options.setdefault("max_sequence_length", 256)

    model_variant = str(config.get("model_variant", "schnell")).lower()
    if model_variant not in {"schnell", "dev"}:
        raise ConfigurationError("model_variant must be 'schnell' or 'dev'.")

    if model_variant == "schnell" and float(call_options["guidance_scale"]) != 0.0:
        raise ConfigurationError(
            "FLUX.1-schnell requires guidance_scale=0.0."
        )
    if model_variant == "schnell" and int(call_options["max_sequence_length"]) > 256:
        raise ConfigurationError(
            "FLUX.1-schnell max_sequence_length cannot exceed 256."
        )
    if int(call_options["num_inference_steps"]) < 1:
        raise ConfigurationError("num_inference_steps must be positive.")
    return call_options


def _safe_filename(prompt: str) -> str:
    filename = "".join(
        character
        if character.isalnum() or character in {"-", "_"}
        else "_"
        for character in prompt.strip()
    )
    return filename.strip("_")[:64] or "generated_image"


def _destination_path(
    prompt: str,
    output_dir: str | Path | None,
    output_path: str | Path | None,
    default_output_dir: Path,
    output_format: str,
) -> Path:
    extension = output_format.lower().lstrip(".")
    if not extension:
        raise ConfigurationError("output_format cannot be empty.")

    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        if destination.suffix.lower() != f".{extension}":
            raise ValueError(
                f"output_path must use the .{extension} extension."
            )
        return destination

    directory = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else default_output_dir
    )
    return directory / f"{_safe_filename(prompt)}.{extension}"


def generate_image(
    prompt: str,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    negative_prompt: str | None = None,
    seed: int | None = None,
) -> ImageGenerationResult:
    """Generate one image from text using only locally available model files."""

    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        raise ValueError("Prompt cannot be empty.")

    config, config_dir = _load_config(config_path)
    model_path = _resolve_path(
        config.get("model_path"),
        "model_path",
        config_dir,
        required=True,
    )
    code_path = _resolve_path(
        config.get("code_path"),
        "code_path",
        config_dir,
        required=False,
    )
    cache_path = _resolve_path(
        config.get("cache_path"),
        "cache_path",
        config_dir,
        required=False,
    )
    default_output_dir = _resolve_output_directory(
        config.get("default_output_directory", "./generated_images"),
        config_dir,
    )

    assert model_path is not None
    configured_seed = int(config.get("seed", 42))
    inference_seed = configured_seed if seed is None else int(seed)
    output_format = str(config.get("output_format", "png"))
    destination = _destination_path(
        normalized_prompt,
        output_dir,
        output_path,
        default_output_dir,
        output_format,
    )

    pipeline, torch, device = _load_pipeline(
        model_path,
        code_path,
        cache_path,
        config,
    )
    generator_device = str(config.get("generator_device", "cpu"))
    if generator_device == "auto":
        generator_device = device
    generator = torch.Generator(
        device=generator_device
    ).manual_seed(inference_seed)

    call_options = _validated_inference_options(config)
    call_options["generator"] = generator

    configured_negative_prompt = config.get("negative_prompt")
    selected_negative_prompt = (
        negative_prompt
        if negative_prompt is not None
        else configured_negative_prompt
    )
    if selected_negative_prompt:
        raise ConfigurationError(
            "FLUX.1-schnell does not use negative prompts with its required "
            "guidance_scale=0. Remove negative_prompt from the call/config."
        )

    prompt_suffix = str(config.get("prompt_suffix", "")).strip()
    model_prompt = normalized_prompt
    if prompt_suffix:
        model_prompt = f"{normalized_prompt}, {prompt_suffix}"

    try:
        output = pipeline(prompt=model_prompt, **call_options)
        images = getattr(output, "images", None)
        if not images:
            raise ImageGenerationError(
                "The Diffusers pipeline returned no images."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(destination)
    except ImageGenerationError:
        raise
    except Exception as exc:
        raise ImageGenerationError(
            "Text-to-image inference failed. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    if not destination.is_file():
        raise ImageGenerationError(
            f"Image generation completed without creating {destination}."
        )
    return ImageGenerationResult(
        image_path=destination,
        prompt=normalized_prompt,
        seed=inference_seed,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an image with local offline FLUX.1-schnell."
    )
    parser.add_argument("prompt", nargs="+", help="Text prompt to render.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the offline text-to-image JSON configuration.",
    )
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "--output-dir",
        help="Directory for the generated image.",
    )
    destination.add_argument(
        "--output",
        help="Explicit generated image path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Generation seed overriding the configured default.",
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
        result = generate_image(
            " ".join(args.prompt),
            config_path=args.config,
            output_dir=args.output_dir,
            output_path=args.output,
            seed=args.seed,
        )
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1

    print(result.image_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
