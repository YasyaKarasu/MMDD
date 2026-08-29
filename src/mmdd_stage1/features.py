"""Cached frozen-Qwen features used by the Stage-1 models."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from mmdd_progress import progress

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
    Current table caches store one already-pooled vector per schema/example-row
    group. Historical raw-token caches identify those groups with
    ``token_groups``.
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
        self,
        device: torch.device,
        *,
        include_hidden: bool,
        hidden_dtype: torch.dtype | None = torch.float32,
    ) -> ObjectFeatures:
        hidden = self.hidden_states
        groups = self.token_groups
        if include_hidden:
            if hidden is None:
                raise ValueError(f"{self.object_id}: Teacher training requires hidden_states")
            hidden = hidden.to(device=device, dtype=hidden_dtype)
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
        hidden_states=hidden_states.detach().cpu() if hidden_states is not None else None,
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
        cache_bytes: int | None = None,
    ) -> None:
        self._eager = dict(eager_features or {})
        self._index = dict(index or {})
        self._teacher_index = dict(teacher_index or {})
        self._cache_size = max(0, cache_size)
        self._cache_byte_limit = (
            max(0, cache_bytes) if cache_bytes is not None else None
        )
        self._cache: OrderedDict[tuple[str, bool], ObjectFeatures] = OrderedDict()
        self._cache_bytes = 0
        self._hot_keys: set[tuple[str, bool]] = set()
        self._hot_cache: dict[tuple[str, bool], ObjectFeatures] = {}
        self._hot_cache_bytes = 0
        self._hot_cache_estimated_bytes = 0
        self._hot_cache_access_coverage = 0.0
        self._preloaded_features: dict[str, ObjectFeatures] = {}
        self._preloaded_embeddings: torch.Tensor | None = None
        if not self._eager and not self._index:
            raise ValueError("Feature store is empty")

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        cache_size: int = 128,
        cache_bytes: int | None = None,
    ) -> FeatureStore:
        if path.is_dir():
            return cls._from_directory(
                path, cache_size=cache_size, cache_bytes=cache_bytes
            )
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
        return cls(features, cache_bytes=cache_bytes)

    @classmethod
    def _from_directory(
        cls,
        root: Path,
        *,
        cache_size: int,
        cache_bytes: int | None,
    ) -> FeatureStore:
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
        return cls(
            index=index,
            teacher_index=teacher_index,
            cache_size=cache_size,
            cache_bytes=cache_bytes,
        )

    @staticmethod
    def _feature_bytes(features: ObjectFeatures) -> int:
        tensors = (
            features.embedding,
            features.hidden_states,
            features.token_groups,
            features.row_embeddings,
        )
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in tensors
            if tensor is not None
        )

    def estimated_feature_bytes(
        self, object_id: str, *, include_hidden: bool
    ) -> int:
        """Estimate cached bytes without loading an object's tensors."""

        if object_id in self._eager:
            value = self._eager[object_id]
            if not include_hidden and value.hidden_states is not None:
                value = ObjectFeatures(
                    object_id=value.object_id,
                    object_type=value.object_type,
                    embedding=value.embedding,
                    row_embeddings=value.row_embeddings,
                )
            return self._feature_bytes(value)
        try:
            _object_type, base_path = self._index[object_id]
        except KeyError as exc:
            raise KeyError(f"Feature cache has no object {object_id!r}") from exc
        size = base_path.stat().st_size
        if include_hidden and object_id in self._teacher_index:
            size += self._teacher_index[object_id].stat().st_size
        return size

    def configure_hot_cache(
        self,
        access_weights: Mapping[str, float],
        *,
        byte_budget: int | None = None,
        object_budget: int | None = None,
        include_hidden: bool,
    ) -> dict[str, int | float]:
        """Reserve a lazy frequency-aware cache within explicit budgets."""

        if byte_budget is not None and byte_budget < 0:
            raise ValueError("Hot-cache byte budget must be non-negative")
        if object_budget is not None and object_budget < 0:
            raise ValueError("Hot-cache object budget must be non-negative")
        if byte_budget is None and object_budget is None:
            raise ValueError("Hot cache requires a byte or object budget")
        if self._hot_cache:
            raise ValueError("Hot cache must be configured before it is populated")
        if (
            self._eager
            or byte_budget == 0
            or object_budget == 0
        ):
            return {
                "planned_objects": 0,
                "estimated_bytes": 0,
                "access_coverage": 0.0,
            }
        candidates = []
        total_weight = 0.0
        for object_id, raw_weight in access_weights.items():
            weight = float(raw_weight)
            if weight <= 0 or object_id not in self._index:
                continue
            size = (
                self.estimated_feature_bytes(
                    object_id, include_hidden=include_hidden
                )
                if byte_budget is not None
                else 0
            )
            priority = weight / max(size, 1) if byte_budget is not None else weight
            candidates.append((priority, weight, object_id, size))
            total_weight += weight
        candidates.sort(key=lambda value: (-value[0], -value[1], value[2]))
        selected = set()
        estimated_bytes = 0
        selected_weight = 0.0
        for _density, weight, object_id, size in candidates:
            if object_budget is not None and len(selected) >= object_budget:
                break
            if byte_budget is not None and estimated_bytes + size > byte_budget:
                continue
            selected.add((object_id, include_hidden))
            estimated_bytes += size
            selected_weight += weight
        self._hot_keys = selected
        self._hot_cache_estimated_bytes = estimated_bytes
        self._hot_cache_access_coverage = (
            selected_weight / total_weight if total_weight else 0.0
        )
        return {
            "planned_objects": len(selected),
            "estimated_bytes": estimated_bytes,
            "access_coverage": self._hot_cache_access_coverage,
        }

    def cache_info(self) -> dict[str, int | float | None]:
        return {
            "lru_objects": len(self._cache),
            "lru_bytes": self._cache_bytes,
            "lru_object_limit": self._cache_size,
            "lru_byte_limit": self._cache_byte_limit,
            "hot_objects": len(self._hot_cache),
            "hot_bytes": self._hot_cache_bytes,
            "planned_hot_objects": len(self._hot_keys),
            "planned_hot_estimated_bytes": self._hot_cache_estimated_bytes,
            "planned_hot_access_coverage": self._hot_cache_access_coverage,
        }

    def object_ids(self) -> Iterable[str]:
        yield from self._eager
        yield from self._index

    def object_type(self, object_id: str) -> str:
        """Return an object's normalized type without loading its tensors."""

        if object_id in self._eager:
            return self._eager[object_id].object_type
        try:
            return self._index[object_id][0]
        except KeyError as exc:
            raise KeyError(f"Feature cache has no object {object_id!r}") from exc

    def preload_embeddings(self, object_ids: Iterable[str]) -> int:
        """Load the referenced raw embeddings into one contiguous CPU tensor."""

        return int(self.preload_embedding_matrix(object_ids).shape[0])

    def preload_embedding_matrix(self, object_ids: Iterable[str]) -> torch.Tensor:
        """Preload raw embeddings and return their contiguous matrix."""

        unique_ids = list(dict.fromkeys(str(object_id) for object_id in object_ids))
        if not unique_ids:
            self._preloaded_features = {}
            self._preloaded_embeddings = torch.empty((0, self.embedding_dimension()))
            self._cache.clear()
            return self._preloaded_embeddings

        embeddings = torch.empty((len(unique_ids), self.embedding_dimension()))
        preloaded = {}
        object_ids = progress(
            unique_ids,
            desc="Preload embeddings",
            unit="object",
            leave=False,
        )
        for row, object_id in enumerate(object_ids):
            features = self.get(object_id, include_hidden=False)
            embeddings[row].copy_(features.embedding)
            preloaded[object_id] = ObjectFeatures(
                object_id=object_id,
                object_type=features.object_type,
                embedding=embeddings[row],
            )
        self._preloaded_features = preloaded
        self._preloaded_embeddings = embeddings
        self._cache.clear()
        self._cache_bytes = 0
        return embeddings

    def embedding_features(self, object_id: str) -> ObjectFeatures:
        """Return scoring features from the contiguous embedding tier when loaded."""

        features = self._preloaded_features.get(object_id)
        return features if features is not None else self.get(
            object_id, include_hidden=False
        )

    def embedding_dimension(self) -> int:
        first_id = next(iter(self.object_ids()))
        return int(self.embedding_features(first_id).embedding.shape[0])

    def teacher_dimension(self) -> int | None:
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
            first_id = next(iter(self.object_ids()))
            hidden = self.get(first_id, include_hidden=True).hidden_states
        return int(hidden.shape[1]) if hidden is not None else None

    def has_teacher_features(self, object_id: str) -> bool:
        """Return whether an object has cached hidden states without loading them."""

        if object_id in self._eager:
            return self._eager[object_id].hidden_states is not None
        if self._teacher_index:
            return object_id in self._teacher_index
        return self.get(object_id, include_hidden=True).hidden_states is not None

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
        if cache_key in self._hot_cache:
            return self._hot_cache[cache_key]
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
        feature_bytes = self._feature_bytes(feature)
        if cache_key in self._hot_keys:
            self._hot_cache[cache_key] = feature
            self._hot_cache_bytes += feature_bytes
        elif self._cache_size or self._cache_byte_limit:
            self._cache[cache_key] = feature
            self._cache_bytes += feature_bytes
            while self._cache and (
                (self._cache_size and len(self._cache) > self._cache_size)
                or (
                    self._cache_byte_limit is not None
                    and self._cache_bytes > self._cache_byte_limit
                )
            ):
                _evicted_key, evicted = self._cache.popitem(last=False)
                self._cache_bytes -= self._feature_bytes(evicted)
        return feature
