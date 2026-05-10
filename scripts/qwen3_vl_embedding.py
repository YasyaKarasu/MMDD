#!/usr/bin/env python
"""Qwen3-VL-Embedding-2B wrapper used by stage-1 scripts."""

from __future__ import annotations

import importlib.util
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from image_preprocessing import ensure_image_within_pixel_limit
from stage1_io import l2_normalize_array

LOG = logging.getLogger("stage1")
IMAGE_LIMIT_RE = re.compile(r"Image size \((\d+) pixels\) exceeds limit of (\d+) pixels")


def resolve_encoder_path(path: str | None = None) -> Path:
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates.extend(
        [
            Path("./Qwen3-VL-Embedding-2B"),
            Path("./hf_models/Qwen3-VL-Embedding-2B"),
            Path(__file__).resolve().parents[1] / "Qwen3-VL-Embedding-2B",
            Path(__file__).resolve().parents[1] / "hf_models" / "Qwen3-VL-Embedding-2B",
        ]
    )
    for candidate in candidates:
        if candidate.exists() and (candidate / "config.json").exists():
            return candidate.resolve()
    searched = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Qwen3-VL-Embedding-2B local model directory was not found. "
        "Pass --encoder_path explicitly. Searched:\n" + searched
    )


def _load_official_embedder(model_dir: Path):
    _check_official_runtime_dependencies()
    script_path = model_dir / "scripts" / "qwen3_vl_embedding.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Missing official Qwen embedder script: {script_path}")
    spec = importlib.util.spec_from_file_location("_local_qwen3_vl_embedding", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Qwen3VLEmbedder


def _check_official_runtime_dependencies() -> None:
    try:
        from importlib import metadata as importlib_metadata

        transformers_version = importlib_metadata.version("transformers")
    except importlib_metadata.PackageNotFoundError as exc:
        raise ImportError(
            "Qwen3-VL-Embedding requires transformers>=4.57.0,<5. "
            "Install project dependencies in the active Python environment."
        ) from exc

    try:
        transformers_major = int(transformers_version.split(".", 1)[0])
    except ValueError:
        transformers_major = 0
    if transformers_major >= 5:
        raise ImportError(
            "Qwen3-VL-Embedding-2B is not compatible with transformers "
            f"{transformers_version}; use transformers>=4.57.0,<5. "
            "For example: python -m pip install "
            "'transformers>=4.57.0,<5' 'qwen-vl-utils>=0.0.14'"
        )

    if importlib.util.find_spec("qwen_vl_utils") is None:
        raise ImportError(
            "Qwen3-VL-Embedding requires qwen-vl-utils>=0.0.14. "
            "For example: python -m pip install 'qwen-vl-utils>=0.0.14'"
        )


def image_limit_from_exception(exc: Exception) -> int | None:
    match = IMAGE_LIMIT_RE.search(str(exc))
    if not match:
        return None
    return int(match.group(2))


def resize_batch_images_for_limit(
    batch: list[dict[str, Any]],
    cache_dir: Path,
    limit: int,
) -> list[dict[str, Any]]:
    resized_batch: list[dict[str, Any]] = []
    for item in batch:
        image = item.get("image")
        if not image:
            resized_batch.append(item)
            continue
        resized_path, _record = ensure_image_within_pixel_limit(Path(str(image)), cache_dir, max_pixels=limit)
        if str(resized_path) == str(image):
            resized_batch.append(item)
        else:
            resized_item = dict(item)
            resized_item["image"] = str(resized_path)
            resized_batch.append(resized_item)
    return resized_batch


class Qwen3VLEmbeddingEncoder:
    """Frozen local Qwen3-VL-Embedding encoder.

    The default path resolution only accepts a local Qwen3-VL-Embedding-2B
    directory. No sentence-transformers, CLIP, MiniLM, or other substitute
    encoder is used by default.
    """

    def __init__(
        self,
        encoder_path: str | None = None,
        device: str = "cuda",
        dtype: str = "bf16",
        batch_size: int = 8,
        mock: bool = False,
        image_resize_cache_dir: str | None = None,
    ) -> None:
        self.batch_size = max(1, int(batch_size))
        self.mock = mock
        self.device = device
        self.dtype = dtype
        self.image_resize_cache_dir = Path(image_resize_cache_dir or ".qwen_resized_images")
        if mock:
            self.model_dir = Path("<mock>")
            self.model = None
            return

        import torch

        self.model_dir = resolve_encoder_path(encoder_path)
        torch_dtype = {
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }.get(dtype.lower())
        if torch_dtype is None:
            raise ValueError(f"Unsupported dtype {dtype!r}; use bf16, fp16, or fp32")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        Qwen3VLEmbedder = _load_official_embedder(self.model_dir)
        kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
        self.model = Qwen3VLEmbedder(model_name_or_path=str(self.model_dir), **kwargs)
        self.model.model.to(torch.device(device))
        self.model.model.eval()

    def _mock_encode(self, inputs: list[dict[str, Any]]) -> np.ndarray:
        vectors = []
        for item in inputs:
            payload = str(item.get("text", "")) + str(item.get("image", ""))
            seed = abs(hash(payload)) % (2**32)
            rng = np.random.default_rng(seed)
            vectors.append(rng.normal(size=32).astype("float32"))
        return l2_normalize_array(np.vstack(vectors))

    def _encode_batch(self, batch: list[dict[str, Any]]) -> np.ndarray:
        try:
            emb = self.model.process(batch, normalize=True)
        except Exception as exc:
            limit = image_limit_from_exception(exc)
            if limit is None or not any(item.get("image") for item in batch):
                raise
            resized_batch = resize_batch_images_for_limit(batch, self.image_resize_cache_dir, limit)
            LOG.warning("Retrying Qwen image embedding batch after resizing images to <= %s pixels", limit)
            emb = self.model.process(resized_batch, normalize=True)
        return emb.detach().float().cpu().numpy()

    def encode_items(self, items: list[dict[str, Any]]) -> np.ndarray:
        if not items:
            return np.zeros((0, 0), dtype="float32")
        if self.mock:
            return self._mock_encode(items)
        outputs = []
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            outputs.append(self._encode_batch(batch))
        return l2_normalize_array(np.vstack(outputs))

    def encode_texts(self, texts: list[str], instruction: str | None = None) -> np.ndarray:
        return self.encode_items([{"text": text, "instruction": instruction} for text in texts])

    def encode_images(self, images: list[str], prompts: list[str] | None = None, instruction: str | None = None) -> np.ndarray:
        prompts = prompts or [""] * len(images)
        return self.encode_items(
            [{"image": image, "text": prompt, "instruction": instruction} for image, prompt in zip(images, prompts)]
        )

    def encode_tables(self, serialized_tables: list[str], instruction: str | None = None) -> np.ndarray:
        return self.encode_texts(serialized_tables, instruction=instruction)
