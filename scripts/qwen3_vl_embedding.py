#!/usr/bin/env python
"""Qwen3-VL-Embedding-2B wrapper used by stage-1 scripts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

from stage1_io import l2_normalize_array


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
    ) -> None:
        self.batch_size = max(1, int(batch_size))
        self.mock = mock
        self.device = device
        self.dtype = dtype
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

    def encode_items(self, items: list[dict[str, Any]]) -> np.ndarray:
        if not items:
            return np.zeros((0, 0), dtype="float32")
        if self.mock:
            return self._mock_encode(items)
        outputs = []
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            emb = self.model.process(batch, normalize=True)
            outputs.append(emb.detach().float().cpu().numpy())
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
