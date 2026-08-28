"""Per-type ANN indexes and the prescribed zero/one-hop retrieval."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            object_id = str(record["object_id"])
            if object_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate object_id {object_id!r}")
            seen.add(object_id)
            features = store.get(object_id, include_hidden=False)
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
        for object_type in OBJECT_TYPES:
            object_ids = ids_by_type.get(object_type, [])
            if not object_ids:
                continue
            index = hnswlib.Index(space="ip", dim=model.student_dim)
            index.init_index(max_elements=len(object_ids), ef_construction=ef_construction, M=m)
            for start in range(0, len(object_ids), batch_size):
                batch_ids = object_ids[start : start + batch_size]
                embeddings = torch.stack(
                    [
                        store.get(object_id, include_hidden=False).embedding.to(
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
    embedding_dim = int(store.get(first_id, include_hidden=False).embedding.shape[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    type_records = {}
    for object_type in OBJECT_TYPES:
        object_ids = ids_by_type.get(object_type, [])
        if not object_ids:
            continue
        index = hnswlib.Index(space="ip", dim=embedding_dim)
        index.init_index(max_elements=len(object_ids), ef_construction=ef_construction, M=m)
        for start in range(0, len(object_ids), batch_size):
            batch_ids = object_ids[start : start + batch_size]
            vectors = torch.stack(
                [
                    store.get(object_id, include_hidden=False).embedding.float()
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
        for object_type, record in manifest["types"].items():
            index = hnswlib.Index(space="ip", dim=model.student_dim)
            index.load_index(str(index_dir / record["index_path"]), max_elements=int(record["objects"]))
            index.set_ef(int(manifest["ef_search"]))
            object_ids = json.loads((index_dir / record["ids_path"]).read_text(encoding="utf-8"))
            if len(object_ids) != int(record["objects"]):
                raise ValueError(f"{index_dir / record['ids_path']}: object count does not match the manifest")
            self.indices[object_type] = index
            self.object_ids[object_type] = object_ids

    @torch.no_grad()
    def search(self, source_id: str, destination_type: str, k: int) -> list[tuple[str, float]]:
        destination_type = normalize_object_type(destination_type)
        if k <= 0 or destination_type not in self.indices:
            return []
        source = self.store.get(source_id, include_hidden=False)
        embedding = source.embedding.to(device=self.device, dtype=torch.float32)
        query = self.model.relation_query(embedding, source.object_type, destination_type)
        query_array = query.detach().cpu().numpy().astype("float32").reshape(1, -1)
        count = len(self.object_ids[destination_type])
        labels, distances = self.indices[destination_type].knn_query(query_array, k=min(k, count))
        return [
            (self.object_ids[destination_type][int(label)], 1.0 - float(distance))
            for label, distance in zip(labels[0], distances[0])
        ]


class RawEmbeddingANNIndices:
    """ANN search over frozen embeddings with no learned projection or relation."""

    def __init__(
        self,
        store: FeatureStore,
        index_dir: Path,
        *,
        corpus_sha256: str,
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
        self.indices = {}
        self.object_ids = {}
        for object_type, record in manifest["types"].items():
            index = hnswlib.Index(space="ip", dim=self.embedding_dim)
            index.load_index(
                str(index_dir / record["index_path"]),
                max_elements=int(record["objects"]),
            )
            index.set_ef(int(manifest["ef_search"]))
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
        destination_type = normalize_object_type(destination_type)
        if k <= 0 or destination_type not in self.indices:
            return []
        embedding = self.store.get(source_id, include_hidden=False).embedding.float()
        if embedding.shape != (self.embedding_dim,):
            raise ValueError(f"{source_id}: raw embedding dimension does not match the index")
        query_array = embedding.numpy().astype("float32").reshape(1, -1)
        count = len(self.object_ids[destination_type])
        labels, distances = self.indices[destination_type].knn_query(
            query_array, k=min(k, count)
        )
        return [
            (self.object_ids[destination_type][int(label)], 1.0 - float(distance))
            for label, distance in zip(labels[0], distances[0])
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
    else:
        selected = sorted(evidence_scores, reverse=True)[: aggregator.top_k]
        evidence_score = sum(selected)
        if aggregator.evidence_aggregation == "topk_mean":
            evidence_score /= len(selected)
    return direct_score, evidence_score


def _channel_ranks(results: list[dict[str, Any]], score_key: str) -> dict[str, int]:
    ranked = sorted(
        (result for result in results if result[score_key] is not None),
        key=lambda result: (-float(result[score_key]), str(result["target_id"])),
    )
    return {str(result["target_id"]): rank for rank, result in enumerate(ranked, 1)}


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
    direct_k: int = 100,
    evidence_k: int = 50,
    targets_per_evidence: int = 50,
    result_k: int = 100,
    evidence_types: tuple[str, ...] = ("text", "image"),
    evidence_aggregation: str = "logsumexp",
    evidence_top_k: int = 4,
    rrf_k: int = 60,
    path_result_k: int = 10,
    evidence_path_k: int | None = None,
) -> list[dict[str, Any]]:
    """Retrieve and rank with all paths, then retain only Stage-2 path detail."""

    if min(result_k, path_result_k) < 0:
        raise ValueError("Retrieval k values must be non-negative")
    if evidence_path_k is not None and evidence_path_k < 0:
        raise ValueError("evidence_path_k must be non-negative")
    ranked = retrieve_zero_one_hop_detailed(
        query_id,
        indices,
        direct_k=direct_k,
        evidence_k=evidence_k,
        targets_per_evidence=targets_per_evidence,
        evidence_types=evidence_types,
        evidence_aggregation=evidence_aggregation,
        evidence_top_k=evidence_top_k,
        rrf_k=rrf_k,
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


def retrieve_zero_one_hop_detailed(
    query_id: str,
    indices: StudentANNIndices,
    *,
    direct_k: int = 100,
    evidence_k: int = 50,
    targets_per_evidence: int = 50,
    evidence_types: tuple[str, ...] = ("text", "image"),
    evidence_aggregation: str = "logsumexp",
    evidence_top_k: int = 4,
    rrf_k: int = 60,
) -> dict[str, list[dict[str, Any]]]:
    """Return full fused/direct/evidence rankings for evaluation."""

    if min(direct_k, evidence_k, targets_per_evidence) < 0:
        raise ValueError("Retrieval k values must be non-negative")
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    aggregator = PathAggregator(evidence_aggregation, evidence_top_k)
    paths_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for target_id, score in indices.search(query_id, "table", direct_k):
        paths_by_target[target_id].append({"kind": "direct", "path_score": score})

    normalized_evidence_types = dict.fromkeys(
        normalize_object_type(value) for value in evidence_types
    )
    for evidence_type in normalized_evidence_types:
        for evidence_id, query_evidence_score in indices.search(query_id, evidence_type, evidence_k):
            for target_id, evidence_target_score in indices.search(evidence_id, "table", targets_per_evidence):
                paths_by_target[target_id].append(
                    {
                        "kind": "evidence",
                        "evidence_id": evidence_id,
                        "path_score": query_evidence_score + evidence_target_score,
                    }
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
    direct_ranks = _channel_ranks(results, "direct_score")
    evidence_ranks = _channel_ranks(results, "evidence_score")
    for result in results:
        target_id = str(result["target_id"])
        direct_rank = direct_ranks.get(target_id)
        evidence_rank = evidence_ranks.get(target_id)
        result["score"] = sum(
            1.0 / (rrf_k + rank) for rank in (direct_rank, evidence_rank) if rank is not None
        )
    fused = sorted(
        results,
        key=lambda result: (-float(result["score"]), str(result["target_id"])),
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
    return {"fused": fused, "direct": direct, "evidence": evidence}
