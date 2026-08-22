"""Image preprocessing helpers shared by dataset and embedding scripts."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any
import warnings


DEFAULT_MAX_IMAGE_PIXELS = 178_956_970


def _has_alpha(mode: str, info: dict[str, Any]) -> bool:
    return mode in {"RGBA", "LA"} or "transparency" in info


def _resized_image_path(path: Path, cache_dir: Path, max_pixels: int, extension: str) -> Path:
    digest = hashlib.sha1(
        f"{path.resolve()}:{path.stat().st_mtime_ns}:{path.stat().st_size}:{max_pixels}".encode("utf-8")
    ).hexdigest()[:16]
    return cache_dir / f"{path.stem}-{digest}{extension}"


def target_size(width: int, height: int, max_pixels: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")
    pixels = width * height
    if pixels <= max_pixels:
        return width, height
    scale = math.sqrt(max_pixels / pixels)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    while new_width * new_height > max_pixels:
        if new_width >= new_height:
            new_width -= 1
        else:
            new_height -= 1
    return new_width, new_height


def ensure_image_within_pixel_limit(
    path: Path,
    cache_dir: Path,
    max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> tuple[Path, dict[str, Any] | None]:
    if max_pixels <= 0:
        return path, None

    try:
        from PIL import Image
        from PIL import ImageFile
    except ImportError as exc:
        raise RuntimeError("Pillow is required to resize oversized images before embedding") from exc

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Image.open(path) as image:
                width, height = image.size
                pixels = width * height
                if pixels <= max_pixels:
                    return path, None

                new_width, new_height = target_size(width, height, max_pixels)
                has_alpha = _has_alpha(image.mode, image.info)
                extension = ".png" if has_alpha else ".jpg"
                out_path = _resized_image_path(path, cache_dir, max_pixels, extension)
                if out_path.exists() and out_path.stat().st_size > 0:
                    return out_path, {
                        "source_path": str(path),
                        "resized_path": str(out_path),
                        "source_width": width,
                        "source_height": height,
                        "source_pixels": pixels,
                        "resized_cached": True,
                    }

                cache_dir.mkdir(parents=True, exist_ok=True)
                image.draft("RGB", (new_width, new_height))
                resampling = getattr(Image, "Resampling", Image).LANCZOS
                image.thumbnail((new_width, new_height), resampling)
                if image.width * image.height > max_pixels:
                    image = image.resize(target_size(image.width, image.height, max_pixels), resampling)

                if has_alpha:
                    if image.mode != "RGBA":
                        image = image.convert("RGBA")
                    image.save(out_path, format="PNG", optimize=True)
                else:
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    image.save(out_path, format="JPEG", quality=90, optimize=True)

                return out_path, {
                    "source_path": str(path),
                    "resized_path": str(out_path),
                    "source_width": width,
                    "source_height": height,
                    "source_pixels": pixels,
                    "resized_width": image.width,
                    "resized_height": image.height,
                    "resized_pixels": image.width * image.height,
                    "resized_cached": False,
                }
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit
