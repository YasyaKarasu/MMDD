"""Cached frozen-Qwen features used by the Stage-1 models."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

    ``hidden_states`` contains the frozen states consumed by the Teacher.
    Current table caches store one vector per schema/example-row group and
    identify them with ``token_groups``. Historical caches may omit the groups
    because their table states were already pooled.
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

    def for_scoring(
        self, device: torch.device, *, include_hidden: bool
    ) -> ObjectFeatures:
        hidden = self.hidden_states
        groups = self.token_groups
        if include_hidden:
            if hidden is None:
                raise ValueError(f"{self.object_id}: Teacher training requires hidden_states")
            hidden = hidden.to(device=device, dtype=torch.float32)
            groups = groups.to(device=device, dtype=torch.long) if groups is not None else None
        else:
            hidden = None
            groups = None
        return ObjectFeatures(
            object_id=self.object_id,
            object_type=self.object_type,
            embedding=self.embedding.to(device=device, dtype=torch.float32),
            hidden_states=hidden,
            token_groups=groups,
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
    return torch.load(path, map_location="cpu", weights_only=True)


class FeatureStore:
    """Read consolidated features or a lazy two-tier per-object cache."""

    def __init__(
        self,
        eager_features: Mapping[str, ObjectFeatures] | None = None,
        *,
        index: Mapping[str, tuple[str, Path]] | None = None,
        teacher_index: Mapping[str, Path] | None = None,
        cache_size: int = 128,
    ) -> None:
        self._eager = dict(eager_features or {})
        self._index = dict(index or {})
        self._teacher_index = dict(teacher_index or {})
        self._cache_size = max(0, cache_size)
        self._cache: OrderedDict[tuple[str, bool], ObjectFeatures] = OrderedDict()
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
        teacher_index: dict[str, Path] = {}
        teacher_manifest = root / "teacher_manifest.jsonl"
        if teacher_manifest.is_file():
            with teacher_manifest.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    object_id = str(record["object_id"])
                    if object_id not in index:
                        raise ValueError(
                            f"{teacher_manifest}:{line_number}: Teacher object {object_id!r} "
                            "is absent from the base manifest"
                        )
                    declared_type = normalize_object_type(str(record["object_type"]))
                    if declared_type != index[object_id][0]:
                        raise ValueError(
                            f"{teacher_manifest}:{line_number}: object type disagrees with "
                            "the base manifest"
                        )
                    relative_path = Path(record["teacher_feature_path"])
                    feature_path = (root / relative_path).resolve()
                    if not feature_path.is_relative_to(root):
                        raise ValueError(
                            f"{teacher_manifest}:{line_number}: teacher_feature_path escapes "
                            "the feature directory"
                        )
                    if object_id in teacher_index:
                        raise ValueError(
                            f"{teacher_manifest}:{line_number}: duplicate object_id {object_id!r}"
                        )
                    teacher_index[object_id] = feature_path
        return cls(index=index, teacher_index=teacher_index, cache_size=cache_size)

    def object_ids(self) -> Iterable[str]:
        yield from self._eager
        yield from self._index

    def get(self, object_id: str, *, include_hidden: bool = True) -> ObjectFeatures:
        if object_id in self._eager:
            value = self._eager[object_id]
            if include_hidden or value.hidden_states is None:
                return value
            return ObjectFeatures(
                object_id=value.object_id,
                object_type=value.object_type,
                embedding=value.embedding,
                row_embeddings=value.row_embeddings,
            )
        cache_key = (object_id, include_hidden)
        if cache_key in self._cache:
            value = self._cache.pop(cache_key)
            self._cache[cache_key] = value
            return value
        try:
            object_type, path = self._index[object_id]
        except KeyError as exc:
            raise KeyError(f"Feature cache has no object {object_id!r}") from exc
        payload = _load_tensor_file(path)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path}: expected a feature mapping")
        if include_hidden and object_id in self._teacher_index:
            teacher_path = self._teacher_index[object_id]
            teacher_payload = _load_tensor_file(teacher_path)
            if not isinstance(teacher_payload, Mapping):
                raise ValueError(f"{teacher_path}: expected a Teacher feature mapping")
            if "hidden_states" not in teacher_payload:
                raise ValueError(
                    f"{teacher_path}: Teacher feature mapping has no hidden_states"
                )
            payload = dict(payload)
            for key in ("hidden_states", "token_groups"):
                if key in teacher_payload:
                    payload[key] = teacher_payload[key]
        elif not include_hidden:
            payload = {
                key: value
                for key, value in payload.items()
                if key not in {"hidden_states", "token_groups"}
            }
        feature = _feature_from_payload(object_id, object_type, payload)
        if self._cache_size:
            self._cache[cache_key] = feature
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return feature

    def dimensions(self) -> tuple[int, int | None]:
        first_id = next(iter(self.object_ids()))
        first = self.get(first_id, include_hidden=False)
        hidden = next(
            (
                feature.hidden_states
                for feature in self._eager.values()
                if feature.hidden_states is not None
            ),
            None,
        )
        if hidden is None and self._teacher_index:
            teacher_id = next(iter(self._teacher_index))
            hidden = self.get(teacher_id, include_hidden=True).hidden_states
        if hidden is None and not self._teacher_index:
            # Legacy directory caches kept both tiers in the base object file.
            hidden = self.get(first_id, include_hidden=True).hidden_states
        hidden_dim = hidden.shape[1] if hidden is not None else None
        return int(first.embedding.shape[0]), int(hidden_dim) if hidden_dim is not None else None
