"""Construct initial Stage-1 objects and training lists from dataset artifacts."""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from mmdd_dataset.utils import clean_text, get_cell, read_jsonl
from mmdd_dataset.wdc_runtime import iter_dataset_artifact


TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def _artifact_records(root: Path, name: str, *, required: bool = True) -> list[dict[str, Any]]:
    path = root / f"{name}.jsonl"
    if path.is_file():
        records = list(read_jsonl(path))
    else:
        manifest = root / "dataset_manifest.json"
        if not manifest.is_file():
            records = []
        else:
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            records = list(iter_dataset_artifact(root, name)) if metadata.get("complete") is True else []
    if required and not records:
        raise ValueError(f"Dataset artifact is empty or missing: {name}")
    return records


def serialize_table_parts(table: dict[str, Any], max_rows: int) -> list[str]:
    headers = [clean_text(column.get("column_name")) for column in table["columns"]]
    parts = ["Columns: " + " | ".join(headers)]
    for row in table["rows"][:max_rows]:
        values = [
            clean_text(get_cell(row, int(column["column_index"])).get("text"))
            for column in table["columns"]
        ]
        parts.append("Row: " + " | ".join(values))
    return parts


def _table_object(
    table: dict[str, Any],
    max_rows: int,
    *,
    embedding_role: str,
    cache_row_embeddings: bool = False,
) -> dict[str, Any]:
    parts = serialize_table_parts(table, max_rows)
    record = {
        "object_id": str(table.get("table_id") or table["object_id"]),
        "object_type": "table",
        "embedding_role": embedding_role,
        "text": "\n".join(parts),
        "table_parts": parts,
    }
    if cache_row_embeddings:
        record["row_routing_texts"] = [f"{parts[0]}\n{row}" for row in parts[1:]]
    return record


def _asset_object(asset: dict[str, Any], dataset_root: Path) -> dict[str, Any]:
    asset_id = str(asset["asset_id"])
    asset_type = str(asset["asset_type"])
    record = {
        "object_id": asset_id,
        "object_type": asset_type,
        "embedding_role": "evidence",
    }
    if asset_type == "text":
        record["text"] = clean_text(asset.get("content"))
    elif asset_type == "image":
        local_path = Path(str(asset.get("local_path", "")))
        if not local_path.is_absolute():
            local_path = dataset_root / local_path
        relative_path = dataset_root / str(asset.get("relative_path", ""))
        image_path = local_path if local_path.is_file() else relative_path
        if not image_path.is_file():
            raise FileNotFoundError(f"{asset_id}: image artifact has no local file")
        record["image"] = str(image_path.resolve())
    else:
        raise ValueError(f"{asset_id}: unsupported asset_type {asset_type!r}")
    return record


def _tokens(text: str) -> Counter[str]:
    return Counter(TOKEN_PATTERN.findall(text.casefold()))


def _semantic_index(
    target_text: dict[str, str],
) -> tuple[dict[str, list[tuple[str, float]]], dict[str, float]]:
    counts = {target_id: _tokens(text) for target_id, text in target_text.items()}
    document_frequency = Counter(token for values in counts.values() for token in values)
    document_count = len(counts)
    inverse_document_frequency = {
        token: math.log((document_count + 1) / (frequency + 1)) + 1.0
        for token, frequency in document_frequency.items()
    }
    postings: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for target_id, values in counts.items():
        weighted = {token: count * inverse_document_frequency[token] for token, count in values.items()}
        norm = math.sqrt(sum(value * value for value in weighted.values())) or 1.0
        for token, value in weighted.items():
            postings[token].append((target_id, value / norm))
    return postings, inverse_document_frequency


def _semantic_negative(
    query_text: str,
    allowed: set[str],
    excluded: set[str],
    postings: dict[str, list[tuple[str, float]]],
    inverse_document_frequency: dict[str, float],
) -> str | None:
    weighted = {
        token: count * inverse_document_frequency[token]
        for token, count in _tokens(query_text).items()
        if token in inverse_document_frequency
    }
    norm = math.sqrt(sum(value * value for value in weighted.values()))
    if not norm:
        return None
    scores: dict[str, float] = defaultdict(float)
    for token, value in weighted.items():
        for target_id, target_weight in postings[token]:
            if target_id in allowed and target_id not in excluded:
                scores[target_id] += value / norm * target_weight
    return max(scores, key=scores.get) if scores else None


def _pick_random(ids: list[str], excluded: set[str], rng: random.Random) -> str | None:
    if not ids:
        return None
    start = rng.randrange(len(ids))
    for offset in range(len(ids)):
        candidate = ids[(start + offset) % len(ids)]
        if candidate not in excluded:
            return candidate
    return None


def _structure_negative(
    query: dict[str, Any],
    split: str,
    excluded: set[str],
    structure_buckets: dict[tuple[str, int, int], list[str]],
) -> str | None:
    query_shape = (len(query["columns"]), len(query["rows"]))
    keys = sorted(
        (key for key in structure_buckets if key[0] == split),
        key=lambda key: (abs(key[1] - query_shape[0]), abs(key[2] - query_shape[1]), key),
    )
    for key in keys:
        candidate = next((value for value in structure_buckets[key] if value not in excluded), None)
        if candidate is not None:
            return candidate
    return None


def _evidence_by_target(
    targets: dict[str, dict[str, Any]],
    assets: list[dict[str, Any]],
    recoveries: Iterable[dict[str, Any]],
) -> dict[str, list[str]]:
    asset_ids = {str(asset["asset_id"]) for asset in assets}
    result: dict[str, list[str]] = defaultdict(list)
    for recovery in recoveries:
        target_id = str(recovery["target_table_id"])
        evidence_id = str(recovery.get("evidence", {}).get("asset_id", ""))
        if evidence_id in asset_ids and evidence_id not in result[target_id]:
            result[target_id].append(evidence_id)

    assets_by_source: dict[str, list[str]] = defaultdict(list)
    for asset in assets:
        source_id = asset.get("source_table_id")
        if source_id is not None:
            assets_by_source[str(source_id)].append(str(asset["asset_id"]))
    for target_id, target in targets.items():
        if not result[target_id]:
            result[target_id].extend(assets_by_source.get(str(target.get("source_table_id")), ()))
    return result


def _edge_evidence_ids(
    evidence_ids: Iterable[str],
    asset_types: dict[str, str],
    limit: int,
) -> list[str]:
    """Retain modality coverage before filling the remaining edge-positive budget."""

    if limit == 0:
        return []
    values = list(evidence_ids)
    selected = []
    for evidence_type in dict.fromkeys(asset_types[evidence_id] for evidence_id in values):
        selected.append(
            next(
                evidence_id
                for evidence_id in values
                if asset_types[evidence_id] == evidence_type
            )
        )
        if len(selected) >= limit:
            return selected
    for evidence_id in values:
        if evidence_id not in selected:
            selected.append(evidence_id)
        if len(selected) >= limit:
            break
    return selected


def _resolve_target_references(
    dataset_root: Path, records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    source_ids = {
        str(record["source_table_ref"]["source_table_id"])
        for record in records
        if "source_table_ref" in record
    }
    if not source_ids:
        return records
    sources = {
        str(record["source_table_id"]): record
        for record in _artifact_records(dataset_root, "source_tables")
        if str(record["source_table_id"]) in source_ids
    }
    missing = source_ids - sources.keys()
    if missing:
        raise KeyError(f"source_tables has no records for: {', '.join(sorted(missing))}")
    return [
        {
            **sources[str(record["source_table_ref"]["source_table_id"])],
            **{key: value for key, value in record.items() if key != "source_table_ref"},
            "object_id": str(record.get("table_id") or record["object_id"]),
        }
        if "source_table_ref" in record
        else record
        for record in records
    ]


def build_stage1_training_artifacts(
    dataset_root: Path,
    *,
    dataset_name: str,
    max_rows: int = 12,
    max_evidence_per_target: int = 8,
    seed: int = 13,
) -> dict[str, list[dict[str, Any]]]:
    if max_rows <= 0 or max_evidence_per_target < 0:
        raise ValueError("max_rows must be positive and max_evidence_per_target must be non-negative")
    queries = {
        str(record.get("table_id") or record["object_id"]): record
        for record in _artifact_records(dataset_root, "query_tables")
    }
    target_records = _resolve_target_references(
        dataset_root, _artifact_records(dataset_root, "data_lake_tables")
    )
    targets = {
        str(record.get("table_id") or record["object_id"]): record
        for record in target_records
    }
    assets = _artifact_records(dataset_root, "bridge_assets")
    qrels = _artifact_records(dataset_root, "qrels")
    recoveries = _artifact_records(dataset_root, "evidence_recoveries", required=False)

    positives_by_query: dict[str, list[str]] = defaultdict(list)
    for qrel in qrels:
        if float(qrel.get("rel", 1)) <= 0:
            continue
        query_id = str(qrel.get("query_table_id", qrel.get("query_id")))
        target_id = str(qrel.get("target_table_id", qrel.get("data_lake_table_id")))
        if query_id not in queries or target_id not in targets:
            raise KeyError(f"qrel references missing objects: {query_id} -> {target_id}")
        if target_id not in positives_by_query[query_id]:
            positives_by_query[query_id].append(target_id)

    query_objects = {
        query_id: _table_object(
            table,
            max_rows,
            embedding_role="query",
            cache_row_embeddings=True,
        )
        for query_id, table in queries.items()
    }
    target_objects = {
        target_id: _table_object(table, max_rows, embedding_role="target")
        for target_id, table in targets.items()
    }
    asset_objects = [_asset_object(asset, dataset_root) for asset in assets]
    object_ids = [*query_objects, *target_objects, *(record["object_id"] for record in asset_objects)]
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("Stage-1 object IDs must be globally unique")

    target_text = {target_id: record["text"] for target_id, record in target_objects.items()}
    postings, inverse_document_frequency = _semantic_index(target_text)
    evidence_by_target = _evidence_by_target(targets, assets, recoveries)
    asset_types = {str(asset["asset_id"]): str(asset["asset_type"]) for asset in assets}
    targets_by_split: dict[str, list[str]] = defaultdict(list)
    target_sets_by_split: dict[str, set[str]] = defaultdict(set)
    structure_buckets: dict[tuple[str, int, int], list[str]] = defaultdict(list)
    for target_id, target in targets.items():
        split = str(target.get("split", "train"))
        targets_by_split[split].append(target_id)
        target_sets_by_split[split].add(target_id)
        structure_buckets[(split, len(target["columns"]), len(target["rows"]))].append(target_id)

    targets_by_evidence: dict[str, set[str]] = defaultdict(set)
    evidence_by_split_type: dict[tuple[str, str], list[str]] = defaultdict(list)
    for target_id, evidence_ids in evidence_by_target.items():
        split = str(targets[target_id].get("split", "train"))
        for evidence_id in evidence_ids:
            targets_by_evidence[evidence_id].add(target_id)
            bucket = evidence_by_split_type[(split, asset_types[evidence_id])]
            if evidence_id not in bucket:
                bucket.append(evidence_id)

    edge_lists = []
    target_lists = []
    emitted_evidence_target_edges: set[tuple[str, str]] = set()
    for query_id, positive_ids in positives_by_query.items():
        query = queries[query_id]
        split = str(query.get("split", "train"))
        split_targets = targets_by_split[split]
        excluded = set(positive_ids)
        selected: dict[str, str] = {}

        semantic_id = _semantic_negative(
            query_objects[query_id]["text"],
            target_sets_by_split[split],
            excluded,
            postings,
            inverse_document_frequency,
        )
        if semantic_id is not None:
            selected["semantic_similar_non_joinable"] = semantic_id
            excluded.add(semantic_id)

        structure_id = _structure_negative(query, split, excluded, structure_buckets)
        if structure_id is not None:
            selected["type_structure_matched"] = structure_id
            excluded.add(structure_id)

        rng = random.Random(f"{seed}:{query_id}")
        corrupted_id = _pick_random(split_targets, excluded, rng)
        positive_evidence = evidence_by_target[positive_ids[0]][:max_evidence_per_target]
        if corrupted_id is not None and positive_evidence:
            selected["corrupted_path"] = corrupted_id
            excluded.add(corrupted_id)

        random_id = _pick_random(split_targets, excluded, rng)
        if random_id is not None:
            selected["random"] = random_id

        ordered_sources = (
            "random",
            "semantic_similar_non_joinable",
            "type_structure_matched",
            "corrupted_path",
        )
        negative_ids = [selected[source] for source in ordered_sources if source in selected]
        if not negative_ids:
            continue

        candidates = [
            {
                "target_id": positive_ids[0],
                "evidence_ids": positive_evidence,
                "negative_source": None,
            }
        ]
        for source in ordered_sources:
            target_id = selected.get(source)
            if target_id is None:
                continue
            candidate_evidence = (
                positive_evidence
                if source == "corrupted_path"
                else evidence_by_target[target_id][:max_evidence_per_target]
            )
            candidates.append(
                {
                    "target_id": target_id,
                    "evidence_ids": candidate_evidence,
                    "negative_source": source,
                }
            )

        target_negative_sources = {
            target_id: source for source, target_id in selected.items()
        }
        for positive_id in positive_ids:
            edge_lists.append(
                {
                    "query_id": query_id,
                    "source_type": "table",
                    "positive_id": positive_id,
                    "candidate_ids": [positive_id, *negative_ids],
                    "destination_type": "table",
                    "edge_kind": "query_to_target",
                    "negative_sources": target_negative_sources,
                    "dataset": dataset_name,
                    "split": split,
                }
            )

        positive_evidence_ids = []
        for positive_id in positive_ids:
            for evidence_id in _edge_evidence_ids(
                evidence_by_target[positive_id], asset_types, max_evidence_per_target
            ):
                if evidence_id not in positive_evidence_ids:
                    positive_evidence_ids.append(evidence_id)
        positive_evidence_set = set(positive_evidence_ids)

        evidence_negatives: dict[str, list[str]] = defaultdict(list)
        evidence_negative_sources: dict[str, str] = {}
        for source in ordered_sources:
            negative_target_id = selected.get(source)
            if negative_target_id is None:
                continue
            seen_type = set()
            for evidence_id in _edge_evidence_ids(
                evidence_by_target[negative_target_id], asset_types, max_evidence_per_target
            ):
                evidence_type = asset_types[evidence_id]
                if (
                    evidence_type in seen_type
                    or evidence_id in positive_evidence_set
                    or evidence_id in evidence_negatives[evidence_type]
                ):
                    continue
                seen_type.add(evidence_type)
                evidence_negatives[evidence_type].append(evidence_id)
                evidence_negative_sources[evidence_id] = source

        for evidence_type in {asset_types[evidence_id] for evidence_id in positive_evidence_ids}:
            if evidence_negatives[evidence_type]:
                continue
            fallback_id = _pick_random(
                evidence_by_split_type[(split, evidence_type)],
                positive_evidence_set,
                rng,
            )
            if fallback_id is not None:
                evidence_negatives[evidence_type].append(fallback_id)
                evidence_negative_sources[fallback_id] = "random"

        for positive_evidence_id in positive_evidence_ids:
            evidence_type = asset_types[positive_evidence_id]
            negative_evidence_ids = evidence_negatives[evidence_type]
            if not negative_evidence_ids:
                continue
            edge_lists.append(
                {
                    "query_id": query_id,
                    "source_type": "table",
                    "positive_id": positive_evidence_id,
                    "candidate_ids": [positive_evidence_id, *negative_evidence_ids],
                    "destination_type": evidence_type,
                    "edge_kind": "query_to_evidence",
                    "negative_sources": {
                        evidence_id: evidence_negative_sources[evidence_id]
                        for evidence_id in negative_evidence_ids
                    },
                    "dataset": dataset_name,
                    "split": split,
                }
            )

        for positive_target_id in positive_ids:
            for evidence_id in _edge_evidence_ids(
                evidence_by_target[positive_target_id], asset_types, max_evidence_per_target
            ):
                edge_key = (evidence_id, positive_target_id)
                if edge_key in emitted_evidence_target_edges:
                    continue
                excluded_targets = targets_by_evidence[evidence_id]
                evidence_target_negatives = [
                    target_id for target_id in negative_ids if target_id not in excluded_targets
                ]
                if not evidence_target_negatives:
                    fallback_id = _pick_random(split_targets, excluded_targets, rng)
                    if fallback_id is not None:
                        evidence_target_negatives.append(fallback_id)
                if not evidence_target_negatives:
                    continue
                emitted_evidence_target_edges.add(edge_key)
                edge_lists.append(
                    {
                        "query_id": evidence_id,
                        "source_type": asset_types[evidence_id],
                        "positive_id": positive_target_id,
                        "candidate_ids": [positive_target_id, *evidence_target_negatives],
                        "destination_type": "table",
                        "edge_kind": "evidence_to_target",
                        "negative_sources": {
                            target_id: target_negative_sources.get(target_id, "random")
                            for target_id in evidence_target_negatives
                        },
                        "dataset": dataset_name,
                        "split": split,
                    }
                )
        target_lists.append(
            {
                "query_id": query_id,
                "positive_target_id": positive_ids[0],
                "positive_target_ids": positive_ids,
                "candidates": candidates,
                "dataset": dataset_name,
                "split": split,
            }
        )

    if not target_lists:
        raise ValueError("No query has both a positive and a non-joinable target in the same split")
    return {
        "stage1_objects": [*query_objects.values(), *target_objects.values(), *asset_objects],
        "edge_lists": edge_lists,
        "target_lists": target_lists,
        "stage1_corpus": [
            *(
                {"object_id": target_id, "object_type": "table"}
                for target_id in target_objects
            ),
            *(
                {"object_id": record["object_id"], "object_type": record["object_type"]}
                for record in asset_objects
            ),
        ],
    }
