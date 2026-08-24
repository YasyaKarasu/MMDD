"""Cached frozen-Qwen features used by the Stage-1 models."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

OBJECT_TYPES = ("table", "text", "image")
TYPE_ALIASES = {
    "table": "table",
    "table_fragment": "table",
    "text": "text",
    "text_asset": "text",
    "image": "image",
    "image_asset": "image",
}


def normalize_object_type(value: str) -> str:
    try:
        return TYPE_ALIASES[value]
    except KeyError as exc:
        choices = ", ".join(sorted(TYPE_ALIASES))
        raise ValueError(f"Unknown object type {value!r}; expected one of: {choices}") from exc


@dataclass(frozen=True)
class ObjectFeatures:
    """The two frozen feature granularities consumed by Teacher and Student.

    ``hidden_states`` contains pooling-before hidden states. For tables,
    ``token_groups`` assigns each token to schema group 0 or an example-row
    group greater than 0 from the same encoder forward pass.
    """

    object_id: str
    object_type: str
    embedding: torch.Tensor
    hidden_states: torch.Tensor | None = None
    token_groups: torch.Tensor | None = None
    row_embeddings: torch.Tensor | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_type", normalize_object_type(self.object_type))
        if self.embedding.ndim != 1:
            raise ValueError(f"{self.object_id}: embedding must have shape [D]")
        if self.hidden_states is not None and self.hidden_states.ndim != 2:
            raise ValueError(f"{self.object_id}: hidden_states must have shape [tokens, D]")
        if self.hidden_states is not None and self.hidden_states.shape[1] != self.embedding.shape[0]:
            raise ValueError(f"{self.object_id}: embedding and hidden-state dimensions must match")
        if self.object_type == "table" and self.hidden_states is not None and self.token_groups is None:
            raise ValueError(f"{self.object_id}: table hidden_states require token_groups")
        if self.token_groups is not None:
            if self.hidden_states is None:
                raise ValueError(f"{self.object_id}: token_groups require hidden_states")
            if self.token_groups.ndim != 1 or self.token_groups.shape[0] != self.hidden_states.shape[0]:
                raise ValueError(f"{self.object_id}: token_groups must have one entry per hidden-state token")
            if torch.any(self.token_groups < 0):
                raise ValueError(f"{self.object_id}: token_groups must be non-negative")
            groups = torch.unique(self.token_groups, sorted=True)
            expected = torch.arange(len(groups), device=groups.device)
            if not torch.equal(groups, expected):
                raise ValueError(f"{self.object_id}: token_groups must be contiguous and start at schema group 0")
        if self.row_embeddings is not None:
            if self.object_type != "table":
                raise ValueError(f"{self.object_id}: only tables can have row_embeddings")
            if self.row_embeddings.ndim != 2 or self.row_embeddings.shape[1] != self.embedding.shape[0]:
                raise ValueError(f"{self.object_id}: row_embeddings must have shape [rows, D]")

    def to(self, device: torch.device, *, include_hidden: bool) -> ObjectFeatures:
        hidden = self.hidden_states
        groups = self.token_groups
        row_embeddings = self.row_embeddings
        if include_hidden:
            if hidden is None:
                raise ValueError(f"{self.object_id}: Teacher training requires hidden_states")
            hidden = hidden.to(device=device, dtype=torch.float32)
            groups = groups.to(device=device, dtype=torch.long) if groups is not None else None
        else:
            hidden = None
            groups = None
        if row_embeddings is not None:
            row_embeddings = row_embeddings.to(device=device, dtype=torch.float32)
        return ObjectFeatures(
            object_id=self.object_id,
            object_type=self.object_type,
            embedding=self.embedding.to(device=device, dtype=torch.float32),
            hidden_states=hidden,
            token_groups=groups,
            row_embeddings=row_embeddings,
        )


def _feature_from_payload(object_id: str, object_type: str, payload: Mapping[str, Any]) -> ObjectFeatures:
    embedding = payload.get("embedding")
    hidden_states = payload.get("hidden_states")
    token_groups = payload.get("token_groups")
    row_embeddings = payload.get("row_embeddings")
    if not isinstance(embedding, torch.Tensor):
        raise ValueError(f"{object_id}: feature payload has no tensor embedding")
    if hidden_states is not None and not isinstance(hidden_states, torch.Tensor):
        raise ValueError(f"{object_id}: hidden_states must be a tensor")
    if token_groups is not None and not isinstance(token_groups, torch.Tensor):
        raise ValueError(f"{object_id}: token_groups must be a tensor")
    if row_embeddings is not None and not isinstance(row_embeddings, torch.Tensor):
        raise ValueError(f"{object_id}: row_embeddings must be a tensor")
    return ObjectFeatures(
        object_id=object_id,
        object_type=object_type,
        embedding=embedding.detach().cpu().float(),
        hidden_states=hidden_states.detach().cpu().float() if hidden_states is not None else None,
        token_groups=token_groups.detach().cpu().long() if token_groups is not None else None,
        row_embeddings=row_embeddings.detach().cpu().float() if row_embeddings is not None else None,
    )


def _load_tensor_file(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch.
        return torch.load(path, map_location="cpu")


class FeatureStore:
    """Read a consolidated feature file or a lazy per-object feature directory."""

    def __init__(
        self,
        eager_features: Mapping[str, ObjectFeatures] | None = None,
        *,
        root: Path | None = None,
        index: Mapping[str, tuple[str, Path]] | None = None,
        cache_size: int = 128,
    ) -> None:
        self._eager = dict(eager_features or {})
        self._root = root
        self._index = dict(index or {})
        self._cache_size = max(0, cache_size)
        self._cache: OrderedDict[str, ObjectFeatures] = OrderedDict()
        if not self._eager and not self._index:
            raise ValueError("Feature store is empty")

    @classmethod
    def from_path(cls, path: Path, *, cache_size: int = 128) -> FeatureStore:
        if path.is_dir():
            return cls._from_directory(path, cache_size=cache_size)
        payload = _load_tensor_file(path)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path}: expected a mapping of object features")
        objects = payload.get("objects", payload)
        if not isinstance(objects, Mapping):
            raise ValueError(f"{path}: 'objects' must be a mapping")
        features: dict[str, ObjectFeatures] = {}
        for object_id, record in objects.items():
            if not isinstance(record, Mapping):
                raise ValueError(f"{path}: object {object_id!r} is not a mapping")
            object_type = record.get("object_type")
            if not isinstance(object_type, str):
                raise ValueError(f"{path}: object {object_id!r} has no object_type")
            features[str(object_id)] = _feature_from_payload(str(object_id), object_type, record)
        return cls(features)

    @classmethod
    def _from_directory(cls, root: Path, *, cache_size: int) -> FeatureStore:
        manifest = root / "manifest.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing feature manifest: {manifest}")
        root = root.resolve()
        index: dict[str, tuple[str, Path]] = {}
        with manifest.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                object_id = str(record["object_id"])
                object_type = normalize_object_type(str(record["object_type"]))
                relative_path = Path(record["feature_path"])
                feature_path = (root / relative_path).resolve()
                if not feature_path.is_relative_to(root):
                    raise ValueError(f"{manifest}:{line_number}: feature_path escapes the feature directory")
                if object_id in index:
                    raise ValueError(f"{manifest}:{line_number}: duplicate object_id {object_id!r}")
                index[object_id] = (object_type, feature_path)
        return cls(root=root, index=index, cache_size=cache_size)

    def __contains__(self, object_id: str) -> bool:
        return object_id in self._eager or object_id in self._index

    def __len__(self) -> int:
        return len(self._eager) + len(self._index)

    def object_ids(self) -> Iterable[str]:
        yield from self._eager
        yield from self._index

    def get(self, object_id: str) -> ObjectFeatures:
        if object_id in self._eager:
            return self._eager[object_id]
        if object_id in self._cache:
            value = self._cache.pop(object_id)
            self._cache[object_id] = value
            return value
        try:
            object_type, path = self._index[object_id]
        except KeyError as exc:
            raise KeyError(f"Feature cache has no object {object_id!r}") from exc
        payload = _load_tensor_file(path)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path}: expected a feature mapping")
        feature = _feature_from_payload(object_id, object_type, payload)
        if self._cache_size:
            self._cache[object_id] = feature
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return feature

    def dimensions(self) -> tuple[int, int | None]:
        first_id = next(iter(self.object_ids()))
        first = self.get(first_id)
        hidden_dim = first.hidden_states.shape[1] if first.hidden_states is not None else None
        return int(first.embedding.shape[0]), int(hidden_dim) if hidden_dim is not None else None
