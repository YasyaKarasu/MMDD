"""Evidence-source-preserving image views for S2-R4c FAST."""

from __future__ import annotations

import hashlib
import io
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .r4c_fast_types import FastUnit, ViewSpec

BASE_PIXELS = 262144
HIGHRES_PIXELS = 524288


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        source.seek(0)
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.load()
    return image


@lru_cache(maxsize=512)
def _original_view_sha(path_text: str) -> str:
    image = _open_rgb(Path(path_text))
    try:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=False, compress_level=9)
        return hashlib.sha256(buffer.getvalue()).hexdigest()
    finally:
        image.close()


def save_crop(
    source_path: Path, box: tuple[int, int, int, int], crop_folder: Path
) -> tuple[Path, str]:
    crop = _open_rgb(source_path).crop(box)
    crop_folder.mkdir(parents=True, exist_ok=True)
    temporary = crop_folder / "pending.png"
    crop.save(temporary, format="PNG", optimize=False, compress_level=9)
    data = temporary.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    path = crop_folder / f"{sha}.png"
    if not path.is_file():
        temporary.replace(path)
    else:
        temporary.unlink()
    return path, sha


def make_evidence_views(
    unit: FastUnit,
    objects: dict[str, Any],
    arm: str,
    crop_result: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    sources = []
    for asset_id in unit.evidence_ids:
        item = objects["evidence"][asset_id]
        if item["asset_type"] != "image":
            sources.append({
                "asset_id": asset_id,
                "asset_type": "text",
                "content": str(item.get("content", "")),
            })
            continue
        path = Path(str(item["local_path"]))
        max_pixels = HIGHRES_PIXELS if arm == "V1_FULL_HIGHRES" and asset_id == unit.focus_image_id else BASE_PIXELS
        views = [ViewSpec(
            source_asset_id=asset_id,
            view_id="ORIGINAL",
            local_path=str(path),
            pixel_box=None,
            max_pixels=max_pixels,
            view_sha256=_original_view_sha(str(path)),
        )]
        if (
            asset_id == unit.focus_image_id
            and arm in {"V2_RAEA_DUAL", "V3_CONSENSUS_DUAL"}
            and crop_result is not None
            and not crop_result.get("crop_fallback")
        ):
            views.append(ViewSpec(
                source_asset_id=asset_id,
                view_id="TIGHT_CROP",
                local_path=str(crop_result["crop_path"]),
                pixel_box=tuple(int(value) for value in crop_result["pixel_box"]),
                max_pixels=BASE_PIXELS,
                view_sha256=str(crop_result["crop_view_sha256"]),
            ))
        sources.append({"asset_id": asset_id, "asset_type": "image", "views": views})
    return sources


def bounded_image(view: ViewSpec) -> Image.Image:
    image = _open_rgb(Path(view.local_path))
    if image.width * image.height > view.max_pixels:
        scale = (view.max_pixels / (image.width * image.height)) ** 0.5
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image
