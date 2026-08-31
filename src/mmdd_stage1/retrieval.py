"""Per-type ANN indexes and the prescribed zero/one-hop retrieval."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mmdd_progress import progress

from .features import OBJECT_TYPES, FeatureStore, normalize_object_type
from .models import StudentJoinabilityModel
from .objectives import PathAggregator


def checkpoint_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_corpus_ids(path: Path, store: FeatureStore) -> dict[str, list[str]]:
    ids_by_type = {object_type: [] for object_type in OBJECT_TYPES}
    seen = set()
    with path.open(encoding="utf-8") as handle:
        lines = progress(handle, desc="Load corpus", unit="object", leave=False)
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            object_id = str(record["object_id"])
            if object_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate object_id {object_id!r}")
            seen.add(object_id)
            features = store.embedding_features(object_id)
            if "object_type" in record:
                declared_type = normalize_object_type(str(record["object_type"]))
                if declared_type != features.object_type:
                    raise ValueError(f"{path}:{line_number}: object type disagrees with the feature cache")
            ids_by_type[features.object_type].append(object_id)
    return ids_by_type


def build_indices(
    model: StudentJoinabilityModel,
    store: FeatureStore,
    ids_by_type: dict[str, list[str]],
    output_dir: Path,
    *,
    device: torch.device,
    checkpoint_sha256: str,
    corpus_sha256: str | None = None,
    batch_size: int = 1024,
    m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 100,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    import hnswlib

    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    type_records = {}
    with torch.no_grad():
        for object_type in progress(
            OBJECT_TYPES, desc="Build Student indexes", unit="type", leave=False
        ):
            object_ids = ids_by_type.get(object_type, [])
            if not object_ids:
                continue
            index = hnswlib.Index(space="ip", dim=model.student_dim)
            index.init_index(max_elements=len(object_ids), ef_construction=ef_construction, M=m)
            starts = range(0, len(object_ids), batch_size)
            for start in progress(
                starts,
                total=len(starts),
                desc=f"Index {object_type}",
                unit="batch",
                leave=False,
            ):
                batch_ids = object_ids[start : start + batch_size]
                embeddings = torch.stack(
                    [
                        store.embedding_features(object_id).embedding.to(
                            device=device, dtype=torch.float32
                        )
                        for object_id in batch_ids
                    ]
                )
                vectors = model.index_vector(embeddings, object_type).detach().cpu().numpy().astype("float32")
                labels = np.arange(start, start + len(batch_ids))
                index.add_items(vectors, labels)
            index.set_ef(ef_search)
            index_path = output_dir / f"{object_type}.hnsw"
            ids_path = output_dir / f"{object_type}_ids.json"
            index.save_index(str(index_path))
            ids_path.write_text(json.dumps(object_ids, ensure_ascii=False) + "\n", encoding="utf-8")
            type_records[object_type] = {
                "index_path": index_path.name,
                "ids_path": ids_path.name,
                "objects": len(object_ids),
            }
    manifest = {
        "format_version": 1,
        "space": "ip",
        "student_dim": model.student_dim,
        "student_checkpoint_sha256": checkpoint_sha256,
        "corpus_sha256": corpus_sha256,
        "hnsw_m": m,
        "ef_construction": ef_construction,
        "ef_search": ef_search,
        "types": type_records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_raw_embedding_indices(
    store: FeatureStore,
    ids_by_type: dict[str, list[str]],
    output_dir: Path,
    *,
    corpus_sha256: str,
    batch_size: int = 1024,
    m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 100,
) -> dict[str, Any]:
    """Build per-type ANN indexes from the frozen embeddings without a Student head."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    import hnswlib

    first_id = next(
        (object_id for object_type in OBJECT_TYPES for object_id in ids_by_type.get(object_type, [])),
        None,
    )
    if first_id is None:
        raise ValueError("Cannot build raw embedding indexes for an empty corpus")
    embedding_dim = int(store.embedding_features(first_id).embedding.shape[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    type_records = {}
    for object_type in progress(
        OBJECT_TYPES, desc="Build raw indexes", unit="type", leave=False
    ):
        object_ids = ids_by_type.get(object_type, [])
        if not object_ids:
            continue
        index = hnswlib.Index(space="ip", dim=embedding_dim)
        index.init_index(max_elements=len(object_ids), ef_construction=ef_construction, M=m)
        starts = range(0, len(object_ids), batch_size)
        for start in progress(
            starts,
            total=len(starts),
            desc=f"Raw index {object_type}",
            unit="batch",
            leave=False,
        ):
            batch_ids = object_ids[start : start + batch_size]
            vectors = torch.stack(
                [
                    store.embedding_features(object_id).embedding.float()
                    for object_id in batch_ids
                ]
            ).numpy().astype("float32")
            labels = np.arange(start, start + len(batch_ids))
            index.add_items(vectors, labels)
        index.set_ef(ef_search)
        index_path = output_dir / f"{object_type}.hnsw"
        ids_path = output_dir / f"{object_type}_ids.json"
        index.save_index(str(index_path))
        ids_path.write_text(json.dumps(object_ids, ensure_ascii=False) + "\n", encoding="utf-8")
        type_records[object_type] = {
            "index_path": index_path.name,
            "ids_path": ids_path.name,
            "objects": len(object_ids),
        }
    manifest = {
        "format_version": 1,
        "index_kind": "raw_embedding",
        "space": "ip",
        "embedding_dim": embedding_dim,
        "corpus_sha256": corpus_sha256,
        "hnsw_m": m,
        "ef_construction": ef_construction,
        "ef_search": ef_search,
        "types": type_records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


class StudentANNIndices:
    def __init__(
        self,
        model: StudentJoinabilityModel,
        store: FeatureStore,
        index_dir: Path,
        *,
        device: torch.device,
        checkpoint_sha256: str,
        corpus_sha256: str | None = None,
        destination_types: tuple[str, ...] | None = None,
    ) -> None:
        import hnswlib

        manifest_path = index_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != 1 or manifest.get("space") != "ip":
            raise ValueError(f"{manifest_path}: unsupported ANN index format")
        if manifest.get("student_dim") != model.student_dim:
            raise ValueError(f"{manifest_path}: Student dimension does not match the checkpoint")
        if manifest.get("student_checkpoint_sha256") != checkpoint_sha256:
            raise ValueError(f"{manifest_path}: indexes were built from a different Student checkpoint")
        if corpus_sha256 is not None and manifest.get("corpus_sha256") != corpus_sha256:
            raise ValueError(f"{manifest_path}: indexes were built from a different corpus")
        self.model = model
        self.store = store
        self.device = device
        self.indices = {}
        self.object_ids = {}
        self.ef_search = int(manifest["ef_search"])
        self._relation_queries: dict[tuple[str, str], np.ndarray] = {}
        selected_types = (
            set(OBJECT_TYPES)
            if destination_types is None
            else {normalize_object_type(value) for value in destination_types}
        )
        for object_type, record in manifest["types"].items():
            if object_type not in selected_types:
                continue
            index = hnswlib.Index(space="ip", dim=model.student_dim)
            index.load_index(str(index_dir / record["index_path"]), max_elements=int(record["objects"]))
            index.set_ef(self.ef_search)
            object_ids = json.loads((index_dir / record["ids_path"]).read_text(encoding="utf-8"))
            if len(object_ids) != int(record["objects"]):
                raise ValueError(f"{index_dir / record['ids_path']}: object count does not match the manifest")
            self.indices[object_type] = index
            self.object_ids[object_type] = object_ids

    @torch.no_grad()
    def search(self, source_id: str, destination_type: str, k: int) -> list[tuple[str, float]]:
        return self.search_many([source_id], destination_type, k)[0]

    @torch.no_grad()
    def search_many(
        self, source_ids: list[str], destination_type: str, k: int
    ) -> list[list[tuple[str, float]]]:
        destination_type = normalize_object_type(destination_type)
        if not source_ids:
            return []
        if k <= 0 or destination_type not in self.indices:
            return [[] for _source_id in source_ids]
        self.indices[destination_type].set_ef(max(self.ef_search, int(k)))

        missing_by_type: dict[str, list[Any]] = defaultdict(list)
        for source_id in dict.fromkeys(source_ids):
            key = (source_id, destination_type)
            if key not in self._relation_queries:
                source = self.store.embedding_features(source_id)
                missing_by_type[source.object_type].append(source)
        for source_type, features in missing_by_type.items():
            embeddings = torch.stack([value.embedding for value in features]).to(
                device=self.device, dtype=torch.float32
            )
            queries = self.model.relation_query(
                embeddings, source_type, destination_type
            )
            arrays = queries.detach().cpu().numpy().astype("float32")
            for features_value, array in zip(features, arrays):
                self._relation_queries[
                    (features_value.object_id, destination_type)
                ] = array

        query_array = np.stack(
            [self._relation_queries[(source_id, destination_type)] for source_id in source_ids]
        )
        count = len(self.object_ids[destination_type])
        labels, distances = self.indices[destination_type].knn_query(query_array, k=min(k, count))
        return [
            [
                (
                    self.object_ids[destination_type][int(label)],
                    1.0 - float(distance),
                )
                for label, distance in zip(row_labels, row_distances)
            ]
            for row_labels, row_distances in zip(labels, distances)
        ]


class RawEmbeddingANNIndices:
    """ANN search over frozen embeddings with no learned projection or relation."""

    def __init__(
        self,
        store: FeatureStore,
        index_dir: Path,
        *,
        corpus_sha256: str,
        destination_types: tuple[str, ...] | None = None,
    ) -> None:
        import hnswlib

        manifest_path = index_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("format_version") != 1
            or manifest.get("index_kind") != "raw_embedding"
            or manifest.get("space") != "ip"
        ):
            raise ValueError(f"{manifest_path}: unsupported raw embedding ANN format")
        if manifest.get("corpus_sha256") != corpus_sha256:
            raise ValueError(f"{manifest_path}: raw embedding index belongs to a different corpus")
        self.store = store
        self.embedding_dim = int(manifest["embedding_dim"])
        self.ef_search = int(manifest["ef_search"])
        self.indices = {}
        self.object_ids = {}
        selected_types = (
            set(OBJECT_TYPES)
            if destination_types is None
            else {normalize_object_type(value) for value in destination_types}
        )
        for object_type, record in manifest["types"].items():
            if object_type not in selected_types:
                continue
            index = hnswlib.Index(space="ip", dim=self.embedding_dim)
            index.load_index(
                str(index_dir / record["index_path"]),
                max_elements=int(record["objects"]),
            )
            index.set_ef(self.ef_search)
            object_ids = json.loads(
                (index_dir / record["ids_path"]).read_text(encoding="utf-8")
            )
            if len(object_ids) != int(record["objects"]):
                raise ValueError(
                    f"{index_dir / record['ids_path']}: object count does not match the manifest"
                )
            self.indices[object_type] = index
            self.object_ids[object_type] = object_ids

    def search(
        self, source_id: str, destination_type: str, k: int
    ) -> list[tuple[str, float]]:
        return self.search_many([source_id], destination_type, k)[0]

    def search_many(
        self, source_ids: list[str], destination_type: str, k: int
    ) -> list[list[tuple[str, float]]]:
        destination_type = normalize_object_type(destination_type)
        if not source_ids:
            return []
        if k <= 0 or destination_type not in self.indices:
            return [[] for _source_id in source_ids]
        self.indices[destination_type].set_ef(max(self.ef_search, int(k)))
        embeddings = torch.stack(
            [
                self.store.embedding_features(source_id).embedding
                for source_id in source_ids
            ]
        ).float()
        if embeddings.shape[1:] != (self.embedding_dim,):
            raise ValueError("Raw embedding dimension does not match the index")
        query_array = embeddings.numpy().astype("float32")
        count = len(self.object_ids[destination_type])
        labels, distances = self.indices[destination_type].knn_query(
            query_array, k=min(k, count)
        )
        return [
            [
                (
                    self.object_ids[destination_type][int(label)],
                    1.0 - float(distance),
                )
                for label, distance in zip(row_labels, row_distances)
            ]
            for row_labels, row_distances in zip(labels, distances)
        ]


def load_or_build_raw_embedding_indices(
    store: FeatureStore,
    ids_by_type: dict[str, list[str]],
    output_dir: Path,
    *,
    corpus_sha256: str,
    batch_size: int = 1024,
    m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 100,
) -> RawEmbeddingANNIndices:
    """Reuse one corpus-bound raw index, or build it once when absent."""

    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "corpus_sha256": corpus_sha256,
            "hnsw_m": m,
            "ef_construction": ef_construction,
            "ef_search": ef_search,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{manifest_path}: raw embedding index settings differ from this run")
    else:
        build_raw_embedding_indices(
            store,
            ids_by_type,
            output_dir,
            corpus_sha256=corpus_sha256,
            batch_size=batch_size,
            m=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
        )
    return RawEmbeddingANNIndices(
        store,
        output_dir,
        corpus_sha256=corpus_sha256,
    )


def _logsumexp(values: Iterable[float]) -> float:
    values = list(values)
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _aggregate_path_channels(
    paths: list[dict[str, Any]], aggregator: PathAggregator
) -> tuple[float | None, float | None]:
    direct_score = next(
        (
            float(path["path_score"])
            for path in paths
            if path["kind"] == "direct"
        ),
        None,
    )
    evidence_scores = [float(path["path_score"]) for path in paths if path["kind"] == "evidence"]
    if not evidence_scores:
        return direct_score, None
    if aggregator.evidence_aggregation == "logsumexp":
        evidence_score = _logsumexp(evidence_scores)
    elif aggregator.evidence_aggregation == "max":
        evidence_score = max(evidence_scores)
    elif aggregator.evidence_aggregation in {"topk_mean", "topk_sum"}:
        selected = sorted(evidence_scores, reverse=True)[: aggregator.top_k]
        evidence_score = sum(selected)
        if aggregator.evidence_aggregation == "topk_mean":
            evidence_score /= len(selected)
    elif aggregator.evidence_aggregation == "softmax_weighted_mean":
        maximum = max(evidence_scores)
        weights = [
            math.exp((value - maximum) / aggregator.temperature)
            for value in evidence_scores
        ]
        evidence_score = sum(
            value * weight for value, weight in zip(evidence_scores, weights)
        ) / sum(weights)
    elif aggregator.evidence_aggregation == "power_mean":
        minimum = min(evidence_scores)
        evidence_score = minimum + (
            sum((value - minimum) ** aggregator.power for value in evidence_scores)
            / len(evidence_scores)
        ) ** (1.0 / aggregator.power)
    else:
        evidence_score = sum(evidence_scores) * sum(
            value != 0.0 for value in evidence_scores
        )
    return direct_score, evidence_score


def _channel_ranks(results: list[dict[str, Any]], score_key: str) -> dict[str, int]:
    ranked = sorted(
        (result for result in results if result[score_key] is not None),
        key=lambda result: (-float(result[score_key]), str(result["target_id"])),
    )
    return {str(result["target_id"]): rank for rank, result in enumerate(ranked, 1)}


def _quantile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _normalize_values(
    values: list[float], method: str, *, temperature: float = 1.0
) -> list[float]:
    if method == "none":
        return list(values)
    if method == "zscore":
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        return [0.0 for _value in values] if std <= 1e-12 else [
            (value - mean) / std for value in values
        ]
    if method == "minmax":
        minimum = min(values)
        width = max(values) - minimum
        return [0.0 for _value in values] if width <= 1e-12 else [
            (value - minimum) / width for value in values
        ]
    if method == "softmax":
        if temperature <= 0:
            raise ValueError("score normalization temperature must be positive")
        maximum = max(values)
        exponentials = [
            math.exp((value - maximum) / temperature) for value in values
        ]
        total = sum(exponentials)
        return [value / total for value in exponentials]
    raise ValueError("score normalization must be one of: none, zscore, minmax, softmax")


def _normalized_channel_scores(
    results: list[dict[str, Any]],
    score_key: str,
    method: str,
    *,
    temperature: float,
) -> dict[str, float]:
    values = [float(result[score_key]) for result in results]
    normalized = _normalize_values(values, method, temperature=temperature)
    return {
        str(result["target_id"]): value
        for result, value in zip(results, normalized)
    }


def fuse_ranked_channels(
    direct: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    *,
    rrf_k: int = 60,
    fusion_mode: str = "rrf",
    direct_weight: float = 1.0,
    evidence_weight: float = 1.0,
    gated_evidence_min_paths: int = 2,
    gated_evidence_quantile: float = 0.75,
    score_normalization: str = "none",
    score_temperature: float = 1.0,
) -> list[dict[str, Any]]:
    """Fuse pre-ranked channels without repeating ANN retrieval."""

    if fusion_mode not in {
        "gated",
        "normalized_rrc",
        "normalized_score",
        "rrf",
        "weighted_rrf",
    }:
        raise ValueError(
            "fusion_mode must be one of: rrf, weighted_rrf, gated, "
            "normalized_score, normalized_rrc"
        )
    if score_normalization not in {"none", "zscore", "minmax", "softmax"}:
        raise ValueError(
            "score_normalization must be one of: none, zscore, minmax, softmax"
        )
    if fusion_mode in {"normalized_score", "normalized_rrc"} and score_normalization == "none":
        raise ValueError("normalized fusion requires score_normalization")
    if score_temperature <= 0:
        raise ValueError("score_temperature must be positive")

    direct_ranks = {
        str(result["target_id"]): rank for rank, result in enumerate(direct, 1)
    }
    evidence_ranks = {
        str(result["target_id"]): rank for rank, result in enumerate(evidence, 1)
    }
    results_by_id = {
        str(result["target_id"]): result for result in [*direct, *evidence]
    }
    evidence_values = [float(result["evidence_score"]) for result in evidence]
    evidence_threshold = (
        _quantile(evidence_values, gated_evidence_quantile)
        if fusion_mode == "gated" and evidence_values
        else None
    )
    weights = (
        (1.0, 1.0)
        if fusion_mode == "rrf"
        else (direct_weight, evidence_weight)
    )
    direct_values = (
        _normalized_channel_scores(
            direct,
            "direct_score",
            score_normalization,
            temperature=score_temperature,
        )
        if fusion_mode in {"normalized_score", "normalized_rrc"} and direct
        else {}
    )
    evidence_normalized_values = (
        _normalized_channel_scores(
            evidence,
            "evidence_score",
            score_normalization,
            temperature=score_temperature,
        )
        if fusion_mode in {"normalized_score", "normalized_rrc"} and evidence
        else {}
    )
    if fusion_mode == "normalized_rrc":
        direct_unit = (
            dict(
                zip(
                    direct_values,
                    _normalize_values(list(direct_values.values()), "minmax"),
                )
            )
            if direct_values
            else {}
        )
        evidence_unit = (
            dict(
                zip(
                    evidence_normalized_values,
                    _normalize_values(
                        list(evidence_normalized_values.values()), "minmax"
                    ),
                )
            )
            if evidence_normalized_values
            else {}
        )
    else:
        direct_unit = {}
        evidence_unit = {}
    fused = []
    for target_id, result in results_by_id.items():
        direct_rank = direct_ranks.get(target_id)
        evidence_rank = evidence_ranks.get(target_id)
        evidence_path_count = sum(
            path["kind"] == "evidence" for path in result["paths"]
        )
        evidence_allowed = fusion_mode != "gated" or (
            evidence_rank is not None
            and (
                evidence_path_count >= gated_evidence_min_paths
                or (
                    evidence_threshold is not None
                    and float(result["evidence_score"]) >= evidence_threshold
                )
            )
        )
        contributions = []
        if direct_rank is not None and weights[0] > 0:
            if fusion_mode == "normalized_score":
                contributions.append(weights[0] * direct_values[target_id])
            elif fusion_mode == "normalized_rrc":
                denominator = max(rrf_k + 1.0 - direct_unit[target_id], 1e-12)
                contributions.append(weights[0] / denominator)
            else:
                contributions.append(weights[0] / (rrf_k + direct_rank))
        if evidence_rank is not None and evidence_allowed and weights[1] > 0:
            if fusion_mode == "normalized_score":
                contributions.append(
                    weights[1] * evidence_normalized_values[target_id]
                )
            elif fusion_mode == "normalized_rrc":
                denominator = max(
                    rrf_k + 1.0 - evidence_unit[target_id], 1e-12
                )
                contributions.append(weights[1] / denominator)
            else:
                contributions.append(weights[1] / (rrf_k + evidence_rank))
        if contributions:
            result["score"] = sum(contributions)
            fused.append(result)
    return sorted(
        fused,
        key=lambda result: (-float(result["score"]), str(result["target_id"])),
    )


def _compact_result_paths(
    paths: list[dict[str, Any]], *, evidence_limit: int
) -> list[dict[str, Any]]:
    compact = []
    if any(path["kind"] == "direct" for path in paths):
        compact.append({"kind": "direct"})
    evidence_paths = [
        path for path in paths if path["kind"] == "evidence"
    ][:evidence_limit]
    compact.extend(
        {
            "kind": "evidence",
            "evidence_id": str(path["evidence_id"]),
            "path_score": float(path["path_score"]),
        }
        for path in evidence_paths
    )
    return compact


def retrieve_zero_one_hop(
    query_id: str,
    indices: StudentANNIndices,
    *,
    k: int = 10,
    gamma: int = 4,
    gamma_evidence: int = 2,
    direct_k: int | None = None,
    evidence_k: int | None = None,
    targets_per_evidence: int | None = None,
    result_k: int | None = None,
    evidence_types: tuple[str, ...] = ("text", "image"),
    evidence_aggregation: str = "logsumexp",
    evidence_top_k: int = 4,
    evidence_temperature: float = 1.0,
    evidence_power: float = 2.0,
    path_edge_normalization: str = "none",
    rrf_k: int = 60,
    fusion_mode: str = "rrf",
    direct_weight: float = 1.0,
    evidence_weight: float = 1.0,
    fusion_score_normalization: str = "none",
    fusion_score_temperature: float = 1.0,
    gated_evidence_min_paths: int = 2,
    gated_evidence_quantile: float = 0.75,
    evidence_modality_weights: dict[str, float] | None = None,
    path_result_k: int = 10,
    evidence_path_k: int | None = None,
) -> list[dict[str, Any]]:
    """Retrieve and rank with all paths, then retain only Stage-2 path detail."""

    if k <= 0:
        raise ValueError("k must be positive")
    result_k = k if result_k is None else result_k
    if min(result_k, path_result_k) < 0:
        raise ValueError("Retrieval k values must be non-negative")
    if evidence_path_k is not None and evidence_path_k < 0:
        raise ValueError("evidence_path_k must be non-negative")
    ranked = retrieve_zero_one_hop_detailed(
        query_id,
        indices,
        k=k,
        gamma=gamma,
        gamma_evidence=gamma_evidence,
        direct_k=direct_k,
        evidence_k=evidence_k,
        targets_per_evidence=targets_per_evidence,
        evidence_types=evidence_types,
        evidence_aggregation=evidence_aggregation,
        evidence_top_k=evidence_top_k,
        evidence_temperature=evidence_temperature,
        evidence_power=evidence_power,
        path_edge_normalization=path_edge_normalization,
        rrf_k=rrf_k,
        fusion_mode=fusion_mode,
        direct_weight=direct_weight,
        evidence_weight=evidence_weight,
        fusion_score_normalization=fusion_score_normalization,
        fusion_score_temperature=fusion_score_temperature,
        gated_evidence_min_paths=gated_evidence_min_paths,
        gated_evidence_quantile=gated_evidence_quantile,
        evidence_modality_weights=evidence_modality_weights,
    )["fused"]
    results = []
    retained_evidence = evidence_top_k if evidence_path_k is None else evidence_path_k
    for result_index, detailed in enumerate(ranked[:result_k]):
        result = {
            "target_id": detailed["target_id"],
            "score": detailed["score"],
        }
        if detailed["evidence_score"] is not None:
            result["evidence_score"] = detailed["evidence_score"]
        if result_index < path_result_k:
            result["paths"] = _compact_result_paths(
                detailed["paths"], evidence_limit=retained_evidence
            )
        results.append(result)
    return results


def _normalize_path_edges(
    paths_by_target: dict[str, list[dict[str, Any]]], normalization: str
) -> dict[str, list[dict[str, Any]]]:
    if normalization == "none":
        return {
            target_id: [dict(path) for path in paths]
            for target_id, paths in paths_by_target.items()
        }
    if normalization != "zscore":
        raise ValueError("path_edge_normalization must be one of: none, zscore")

    query_edges: dict[str, dict[str, float]] = defaultdict(dict)
    target_edges: dict[str, list[float]] = defaultdict(list)
    for paths in paths_by_target.values():
        for path in paths:
            if path["kind"] != "evidence":
                continue
            evidence_type = str(path["evidence_type"])
            query_edges[evidence_type][str(path["evidence_id"])] = float(
                path["query_evidence_score"]
            )
            target_edges[evidence_type].append(float(path["evidence_target_score"]))

    normalized_query_edges = {
        evidence_type: dict(
            zip(
                values,
                _normalize_values(list(values.values()), normalization),
            )
        )
        for evidence_type, values in query_edges.items()
    }
    target_stats: dict[str, tuple[float, float]] = {}
    for evidence_type, values in target_edges.items():
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        target_stats[evidence_type] = (mean, math.sqrt(variance))

    normalized = {}
    for target_id, paths in paths_by_target.items():
        normalized_paths = []
        for source in paths:
            path = dict(source)
            if path["kind"] == "evidence":
                evidence_type = str(path["evidence_type"])
                query_score = normalized_query_edges[evidence_type][
                    str(path["evidence_id"])
                ]
                mean, std = target_stats[evidence_type]
                target_score = (
                    0.0
                    if std <= 1e-12
                    else (float(path["evidence_target_score"]) - mean) / std
                )
                path["normalized_query_evidence_score"] = query_score
                path["normalized_evidence_target_score"] = target_score
                path["path_score"] = query_score + target_score
            normalized_paths.append(path)
        normalized[target_id] = normalized_paths
    return normalized


def rank_detailed_paths(
    paths_by_target: dict[str, list[dict[str, Any]]],
    *,
    aggregator: PathAggregator,
    path_edge_normalization: str = "none",
    rrf_k: int,
    fusion_mode: str,
    direct_weight: float,
    evidence_weight: float,
    fusion_score_normalization: str = "none",
    fusion_score_temperature: float = 1.0,
    gated_evidence_min_paths: int,
    gated_evidence_quantile: float,
) -> dict[str, list[dict[str, Any]]]:
    paths_by_target = _normalize_path_edges(
        paths_by_target, path_edge_normalization
    )
    results = []
    for target_id, paths in paths_by_target.items():
        paths.sort(key=lambda path: path["path_score"], reverse=True)
        direct_score, evidence_score = _aggregate_path_channels(paths, aggregator)
        results.append(
            {
                "target_id": target_id,
                "direct_score": direct_score,
                "evidence_score": evidence_score,
                "paths": paths,
            }
        )
    direct = sorted(
        (result for result in results if result["direct_score"] is not None),
        key=lambda result: (
            -float(result["direct_score"]),
            str(result["target_id"]),
        ),
    )
    evidence = sorted(
        (result for result in results if result["evidence_score"] is not None),
        key=lambda result: (
            -float(result["evidence_score"]),
            str(result["target_id"]),
        ),
    )
    fused = fuse_ranked_channels(
        direct,
        evidence,
        rrf_k=rrf_k,
        fusion_mode=fusion_mode,
        direct_weight=direct_weight,
        evidence_weight=evidence_weight,
        gated_evidence_min_paths=gated_evidence_min_paths,
        gated_evidence_quantile=gated_evidence_quantile,
        score_normalization=fusion_score_normalization,
        score_temperature=fusion_score_temperature,
    )
    return {"fused": fused, "direct": direct, "evidence": evidence}


def retrieve_zero_one_hop_detailed_many(
    query_ids: Sequence[str],
    indices: StudentANNIndices | RawEmbeddingANNIndices,
    *,
    k: int = 10,
    gamma: int = 4,
    gamma_evidence: int = 2,
    direct_k: int | None = None,
    evidence_k: int | None = None,
    targets_per_evidence: int | None = None,
    evidence_types: tuple[str, ...] = ("text", "image"),
    evidence_aggregation: str = "logsumexp",
    evidence_top_k: int = 4,
    evidence_temperature: float = 1.0,
    evidence_power: float = 2.0,
    path_edge_normalization: str = "none",
    rrf_k: int = 60,
    fusion_mode: str = "rrf",
    direct_weight: float = 1.0,
    evidence_weight: float = 1.0,
    fusion_score_normalization: str = "none",
    fusion_score_temperature: float = 1.0,
    gated_evidence_min_paths: int = 2,
    gated_evidence_quantile: float = 0.75,
    evidence_modality_weights: dict[str, float] | None = None,
    query_batch_size: int = 32,
) -> list[dict[str, list[dict[str, Any]]]]:
    """Return full rankings for many queries with batched ANN searches."""

    if k <= 0:
        raise ValueError("k must be positive")
    if gamma <= 0 or gamma_evidence <= 0:
        raise ValueError("gamma and gamma_evidence must be positive")
    if query_batch_size <= 0:
        raise ValueError("query_batch_size must be positive")
    direct_k = direct_k if direct_k is not None else math.ceil(gamma * k)
    evidence_k = evidence_k if evidence_k is not None else math.ceil(gamma_evidence * k)
    targets_per_evidence = (
        targets_per_evidence
        if targets_per_evidence is not None
        else math.ceil(gamma_evidence * k)
    )
    if min(direct_k, evidence_k, targets_per_evidence) < 0:
        raise ValueError("Retrieval k values must be non-negative")
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    if fusion_mode not in {
        "rrf",
        "weighted_rrf",
        "gated",
        "normalized_score",
        "normalized_rrc",
    }:
        raise ValueError(
            "fusion_mode must be one of: rrf, weighted_rrf, gated, "
            "normalized_score, normalized_rrc"
        )
    if path_edge_normalization not in {"none", "zscore"}:
        raise ValueError("path_edge_normalization must be one of: none, zscore")
    if fusion_score_normalization not in {"none", "zscore", "minmax", "softmax"}:
        raise ValueError(
            "fusion_score_normalization must be one of: none, zscore, minmax, softmax"
        )
    if fusion_score_temperature <= 0:
        raise ValueError("fusion_score_temperature must be positive")
    if direct_weight < 0 or evidence_weight < 0:
        raise ValueError("fusion weights must be non-negative")
    if fusion_mode != "rrf" and direct_weight == evidence_weight == 0:
        raise ValueError("at least one fusion weight must be positive")
    if gated_evidence_min_paths <= 0:
        raise ValueError("gated_evidence_min_paths must be positive")
    if not 0 <= gated_evidence_quantile <= 1:
        raise ValueError("gated_evidence_quantile must be in [0, 1]")
    if not query_ids:
        return []
    if len(query_ids) > query_batch_size:
        return [
            result
            for start in range(0, len(query_ids), query_batch_size)
            for result in retrieve_zero_one_hop_detailed_many(
                query_ids[start : start + query_batch_size],
                indices,
                k=k,
                gamma=gamma,
                gamma_evidence=gamma_evidence,
                direct_k=direct_k,
                evidence_k=evidence_k,
                targets_per_evidence=targets_per_evidence,
                evidence_types=evidence_types,
                evidence_aggregation=evidence_aggregation,
                evidence_top_k=evidence_top_k,
                evidence_temperature=evidence_temperature,
                evidence_power=evidence_power,
                path_edge_normalization=path_edge_normalization,
                rrf_k=rrf_k,
                fusion_mode=fusion_mode,
                direct_weight=direct_weight,
                evidence_weight=evidence_weight,
                fusion_score_normalization=fusion_score_normalization,
                fusion_score_temperature=fusion_score_temperature,
                gated_evidence_min_paths=gated_evidence_min_paths,
                gated_evidence_quantile=gated_evidence_quantile,
                evidence_modality_weights=evidence_modality_weights,
                query_batch_size=query_batch_size,
            )
        ]
    aggregator = PathAggregator(
        evidence_aggregation,
        evidence_top_k,
        temperature=evidence_temperature,
        power=evidence_power,
    )
    paths_by_query = [defaultdict(list) for _query_id in query_ids]
    for paths_by_target, direct_hits in zip(
        paths_by_query, indices.search_many(list(query_ids), "table", direct_k)
    ):
        for target_id, score in direct_hits:
            paths_by_target[target_id].append(
                {"kind": "direct", "path_score": score}
            )

    normalized_evidence_types = dict.fromkeys(
        normalize_object_type(value) for value in evidence_types
    )
    modality_weights = {
        normalize_object_type(key): float(value)
        for key, value in (evidence_modality_weights or {}).items()
    }
    if any(value < 0 for value in modality_weights.values()):
        raise ValueError("evidence modality weights must be non-negative")
    evidence_hits_by_query: list[list[tuple[str, float, str]]] = [
        [] for _query_id in query_ids
    ]
    for evidence_type in normalized_evidence_types:
        modality_weight = modality_weights.get(evidence_type, 1.0)
        if modality_weight <= 0:
            continue
        offset = math.log(modality_weight)
        for evidence_hits, hits in zip(
            evidence_hits_by_query,
            indices.search_many(list(query_ids), evidence_type, evidence_k),
        ):
            evidence_hits.extend(
                (evidence_id, query_evidence_score + offset, evidence_type)
                for evidence_id, query_evidence_score in hits
            )
    flattened_evidence = [
        (query_index, evidence_id, query_evidence_score, evidence_type)
        for query_index, evidence_hits in enumerate(evidence_hits_by_query)
        for evidence_id, query_evidence_score, evidence_type in evidence_hits
    ]
    target_hits = indices.search_many(
        [entry[1] for entry in flattened_evidence],
        "table",
        targets_per_evidence,
    )
    for (
        query_index,
        evidence_id,
        query_evidence_score,
        evidence_type,
    ), evidence_targets in zip(
        flattened_evidence, target_hits
    ):
        for target_id, evidence_target_score in evidence_targets:
            paths_by_query[query_index][target_id].append(
                {
                    "kind": "evidence",
                    "evidence_id": evidence_id,
                    "evidence_type": evidence_type,
                    "query_evidence_score": query_evidence_score,
                    "evidence_target_score": evidence_target_score,
                    "path_score": query_evidence_score + evidence_target_score,
                }
            )
    return [
        rank_detailed_paths(
            paths_by_target,
            aggregator=aggregator,
            path_edge_normalization=path_edge_normalization,
            rrf_k=rrf_k,
            fusion_mode=fusion_mode,
            direct_weight=direct_weight,
            evidence_weight=evidence_weight,
            fusion_score_normalization=fusion_score_normalization,
            fusion_score_temperature=fusion_score_temperature,
            gated_evidence_min_paths=gated_evidence_min_paths,
            gated_evidence_quantile=gated_evidence_quantile,
        )
        for paths_by_target in paths_by_query
    ]


def retrieve_zero_one_hop_detailed(
    query_id: str,
    indices: StudentANNIndices | RawEmbeddingANNIndices,
    *,
    k: int = 10,
    gamma: int = 4,
    gamma_evidence: int = 2,
    direct_k: int | None = None,
    evidence_k: int | None = None,
    targets_per_evidence: int | None = None,
    evidence_types: tuple[str, ...] = ("text", "image"),
    evidence_aggregation: str = "logsumexp",
    evidence_top_k: int = 4,
    evidence_temperature: float = 1.0,
    evidence_power: float = 2.0,
    path_edge_normalization: str = "none",
    rrf_k: int = 60,
    fusion_mode: str = "rrf",
    direct_weight: float = 1.0,
    evidence_weight: float = 1.0,
    fusion_score_normalization: str = "none",
    fusion_score_temperature: float = 1.0,
    gated_evidence_min_paths: int = 2,
    gated_evidence_quantile: float = 0.75,
    evidence_modality_weights: dict[str, float] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return full fused/direct/evidence rankings for one query."""

    return retrieve_zero_one_hop_detailed_many(
        [query_id],
        indices,
        k=k,
        gamma=gamma,
        gamma_evidence=gamma_evidence,
        direct_k=direct_k,
        evidence_k=evidence_k,
        targets_per_evidence=targets_per_evidence,
        evidence_types=evidence_types,
        evidence_aggregation=evidence_aggregation,
        evidence_top_k=evidence_top_k,
        evidence_temperature=evidence_temperature,
        evidence_power=evidence_power,
        path_edge_normalization=path_edge_normalization,
        rrf_k=rrf_k,
        fusion_mode=fusion_mode,
        direct_weight=direct_weight,
        evidence_weight=evidence_weight,
        fusion_score_normalization=fusion_score_normalization,
        fusion_score_temperature=fusion_score_temperature,
        gated_evidence_min_paths=gated_evidence_min_paths,
        gated_evidence_quantile=gated_evidence_quantile,
        evidence_modality_weights=evidence_modality_weights,
    )[0]
