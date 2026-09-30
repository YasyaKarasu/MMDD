"""Frozen local MiniLM cell embeddings used by the D-0.98 matcher.

The backend intentionally has no download path.  A run must point at a local
``all-MiniLM-L6-v2`` directory whose files are included in the experiment
receipt.  Embeddings are mean pooled with the attention mask and L2 normalized,
matching the historical matching contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from .column_data import file_hash


class MiniLMCellEmbeddingBackend:
    """Small, deterministic ``embed_texts`` backend for :func:`semantic_frozen_matches`."""

    def __init__(self, model_dir: Path, *, device: str = "cpu", max_length: int = 256,
                 batch_size: int = 256) -> None:
        if max_length < 1 or batch_size < 1:
            raise ValueError("max_length and batch_size must be positive")
        self.model_dir = Path(model_dir).resolve()
        if not self.model_dir.is_dir():
            raise FileNotFoundError(self.model_dir)
        self.device = torch.device(device)
        self.max_length = max_length
        self.batch_size = batch_size
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # keep the rest of src importable without transformers
            raise RuntimeError("MiniLM backend requires transformers") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(self.model_dir, local_files_only=True).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._cache: dict[str, torch.Tensor] = {}
        self.identity = {
            "model_dir": str(self.model_dir),
            "model_files": {p.name: file_hash(p) for p in sorted(self.model_dir.iterdir()) if p.is_file()},
            "pooling": "attention_mask_mean",
            "l2_normalize": True,
            "max_length": max_length,
            "dimension": int(getattr(self.model.config, "hidden_size", 0)),
            "dtype": "float32",
            "device": str(self.device),
        }

    @torch.inference_mode()
    def embed_texts(self, texts: Iterable[str]) -> torch.Tensor:
        values = [str(text) for text in texts]
        if not values:
            dimension = self.identity["dimension"]
            return torch.empty((0, dimension), dtype=torch.float32)
        missing = [text for text in dict.fromkeys(values) if text not in self._cache]
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length,
                                     return_tensors="pt").to(self.device)
            hidden = self.model(**encoded).last_hidden_state.float()
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.)
            pooled = F.normalize(pooled, p=2, dim=-1).cpu()
            self._cache.update({text: vector for text, vector in zip(batch, pooled, strict=True)})
        return torch.stack([self._cache[text] for text in values]).float()

    def vector(self, text: str):
        return self.embed_texts([text])[0].numpy()

    def matrix(self, texts: Iterable[str]):
        return self.embed_texts(texts).numpy()


def load_minilm_backend(model_dir: Path, *, device: str = "cpu", max_length: int = 256,
                        batch_size: int = 256) -> MiniLMCellEmbeddingBackend:
    """Construct the frozen backend and fail before any matching work on missing local weights."""
    return MiniLMCellEmbeddingBackend(model_dir, device=device, max_length=max_length,
                                      batch_size=batch_size)
