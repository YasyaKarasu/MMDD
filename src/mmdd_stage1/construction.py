"""Construct initial Stage-1 objects and training lists from dataset artifacts."""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mmdd_dataset.utils import clean_text, get_cell, read_jsonl
from mmdd_dataset.wdc_runtime import iter_dataset_artifact
from mmdd_progress import progress

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
DEFAULT_MAX_CELL_CHARS = 1024
TABLE_ROW_FORMATS = ("values", "named_cells")


def _artifact_records(root: Path, name: str, *, required: bool = True) -> list[dict[str, Any]]:
    manifest = root / "dataset_manifest.json"
    path = root / f"{name}.jsonl"
    if manifest.is_file():
        records = list(iter_dataset_artifact(root, name))
    elif path.is_file():
        records = list(read_jsonl(path))
    else:
        records = []
    if required and not records:
        raise ValueError(f"Dataset artifact is empty or missing: {name}")
    return records


def serialize_table_parts(
    table: dict[str, Any],
    max_rows: int,
    max_cell_chars: int = DEFAULT_MAX_CELL_CHARS,
    *,
    row_format: str = "values",
) -> list[str]:
    if row_format not in TABLE_ROW_FORMATS:
        raise ValueError(
            f"row_format must be one of: {', '.join(TABLE_ROW_FORMATS)}"
        )
    headers = [clean_text(column.get("column_name")) for column in table["columns"]]
    parts = ["Columns: " + " | ".join(headers)]
    for row in table["rows"][:max_rows]:
        values = []
        for column in table["columns"]:
            value = clean_text(
                get_cell(row, int(column["column_index"])).get("text")
            )
            values.append(value[:max_cell_chars].rstrip())
        if row_format == "named_cells":
            values = [
                f"{header or f'column_{index}'}: {value}"
                for index, (header, value) in enumerate(zip(headers, values))
            ]
        parts.append("Row: " + " | ".join(values))
    return parts


def _table_object(
    table: dict[str, Any],
    max_rows: int,
    max_cell_chars: int = DEFAULT_MAX_CELL_CHARS,
    *,
    embedding_role: str,
    row_format: str = "values",
) -> dict[str, Any]:
    parts = serialize_table_parts(
        table,
        max_rows,
        max_cell_chars,
        row_format=row_format,
    )
    record = {
        "object_id": str(table["table_id"]),
        "object_type": "table",
        "embedding_role": embedding_role,
        "table_parts": parts,
    }
    return record


def _asset_object(asset: dict[str, Any], dataset_root: Path) -> dict[str, Any]:
    asset_id = str(asset["asset_id"])
    asset_type = str(asset["asset_type"])
    record = {
        "object_id": asset_id,
        "object_type": asset_type,
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
    excluded: set[str],
    structure_buckets: dict[tuple[int, int], list[str]],
) -> str | None:
    query_shape = (len(query["columns"]), len(query["rows"]))
    keys = sorted(
        structure_buckets,
        key=lambda key: (abs(key[0] - query_shape[0]), abs(key[1] - query_shape[1]), key),
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


def _recovery_evidence(
    queries: dict[str, dict[str, Any]],
    targets: dict[str, dict[str, Any]],
    assets: list[dict[str, Any]],
    recoveries: Iterable[dict[str, Any]],
) -> dict[tuple[str, str], list[str]]:
    asset_ids = {str(asset["asset_id"]) for asset in assets}
    result: dict[tuple[str, str], list[str]] = defaultdict(list)
    for recovery in recoveries:
        query_id = str(recovery.get("query_table_id", ""))
        target_id = str(recovery.get("target_table_id", ""))
        evidence_id = str(recovery.get("evidence", {}).get("asset_id", ""))
        key = (query_id, target_id)
        if (
            query_id in queries
            and target_id in targets
            and evidence_id in asset_ids
            and evidence_id not in result[key]
        ):
            result[key].append(evidence_id)
    return result


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
    max_cell_chars: int = DEFAULT_MAX_CELL_CHARS,
    table_row_format: str = "values",
    seed: int = 13,
) -> dict[str, list[dict[str, Any]]]:
    if max_rows <= 0 or max_cell_chars <= 0:
        raise ValueError("max_rows and max_cell_chars must be positive")
    if table_row_format not in TABLE_ROW_FORMATS:
        raise ValueError(
            f"table_row_format must be one of: {', '.join(TABLE_ROW_FORMATS)}"
        )
    queries = {
        str(record["table_id"]): record
        for record in _artifact_records(dataset_root, "query_tables")
    }
    target_records = _resolve_target_references(
        dataset_root, _artifact_records(dataset_root, "data_lake_tables")
    )
    targets = {
        str(record["table_id"]): record
        for record in target_records
    }
    assets = _artifact_records(dataset_root, "bridge_assets")
    qrels = _artifact_records(dataset_root, "qrels")
    recoveries = _artifact_records(dataset_root, "evidence_recoveries", required=False)

    positives_by_query: dict[str, list[str]] = defaultdict(list)
    for qrel in qrels:
        if float(qrel.get("rel", 1)) <= 0:
            continue
        query_id = str(qrel["query_table_id"])
        target_id = str(qrel["target_table_id"])
        if query_id not in queries or target_id not in targets:
            raise KeyError(f"qrel references missing objects: {query_id} -> {target_id}")
        if target_id not in positives_by_query[query_id]:
            positives_by_query[query_id].append(target_id)

    query_objects = {
        query_id: _table_object(
            table,
            max_rows,
            max_cell_chars,
            embedding_role="query",
            row_format=table_row_format,
        )
        for query_id, table in queries.items()
    }
    target_objects = {
        target_id: _table_object(
            table,
            max_rows,
            max_cell_chars,
            embedding_role="target",
            row_format=table_row_format,
        )
        for target_id, table in targets.items()
    }
    asset_objects = [_asset_object(asset, dataset_root) for asset in assets]
    object_ids = [*query_objects, *target_objects, *(record["object_id"] for record in asset_objects)]
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("Stage-1 object IDs must be globally unique")

    target_text = {
        target_id: "\n".join(record["table_parts"])
        for target_id, record in target_objects.items()
    }
    postings, inverse_document_frequency = _semantic_index(target_text)
    evidence_by_target = _evidence_by_target(targets, assets, recoveries)
    recovery_evidence = _recovery_evidence(queries, targets, assets, recoveries)
    asset_types = {str(asset["asset_id"]): str(asset["asset_type"]) for asset in assets}
    target_ids = list(targets)
    target_id_set = set(target_ids)
    structure_buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
    for target_id, target in targets.items():
        structure_buckets[(len(target["columns"]), len(target["rows"]))].append(target_id)

    targets_by_evidence: dict[str, set[str]] = defaultdict(set)
    evidence_by_type: dict[str, list[str]] = defaultdict(list)
    for target_id, evidence_ids in evidence_by_target.items():
        for evidence_id in evidence_ids:
            targets_by_evidence[evidence_id].add(target_id)
            bucket = evidence_by_type[asset_types[evidence_id]]
            if evidence_id not in bucket:
                bucket.append(evidence_id)

    edge_lists = []
    target_lists = []
    emitted_evidence_target_edges: set[tuple[str, str]] = set()
    query_items = progress(
        positives_by_query.items(),
        total=len(positives_by_query),
        desc="Build Stage-1 examples",
        unit="query",
    )
    for query_id, direct_positive_target_ids in query_items:
        query = queries[query_id]
        split = str(query.get("split", "train"))
        evidence_positive_target_ids = [
            target_id
            for recovery_query_id, target_id in recovery_evidence
            if recovery_query_id == query_id
        ]
        if not evidence_positive_target_ids:
            evidence_positive_target_ids = [
                target_id
                for target_id in direct_positive_target_ids
                if evidence_by_target[target_id]
            ]
        if not evidence_positive_target_ids:
            evidence_positive_target_ids = [direct_positive_target_ids[0]]
        positive_evidence_by_target = {
            target_id: recovery_evidence.get((query_id, target_id), evidence_by_target[target_id])
            for target_id in evidence_positive_target_ids
        }
        positive_target_ids = list(
            dict.fromkeys(
                [*direct_positive_target_ids, *evidence_positive_target_ids]
            )
        )
        excluded = set(positive_target_ids)
        selected: dict[str, str] = {}

        semantic_id = _semantic_negative(
            "\n".join(query_objects[query_id]["table_parts"]),
            target_id_set,
            excluded,
            postings,
            inverse_document_frequency,
        )
        if semantic_id is not None:
            selected["semantic_similar_non_joinable"] = semantic_id
            excluded.add(semantic_id)

        structure_id = _structure_negative(query, excluded, structure_buckets)
        if structure_id is not None:
            selected["type_structure_matched"] = structure_id
            excluded.add(structure_id)

        rng = random.Random(f"{seed}:{query_id}")
        corrupted_id = _pick_random(target_ids, excluded, rng)
        positive_evidence = positive_evidence_by_target[
            evidence_positive_target_ids[0]
        ]
        if corrupted_id is not None and positive_evidence:
            selected["corrupted_path"] = corrupted_id
            excluded.add(corrupted_id)

        random_id = _pick_random(target_ids, excluded, rng)
        if random_id is not None:
            selected["random"] = random_id

        ordered_sources = (
            "random",
            "semantic_similar_non_joinable",
            "type_structure_matched",
            "corrupted_path",
        )
        negative_ids = [selected[source] for source in ordered_sources if source in selected]
        candidate_ids = list(
            dict.fromkeys(
                [
                    *positive_target_ids,
                    *negative_ids,
                ]
            )
        )
        if len(candidate_ids) < 2:
            continue

        candidates = []
        for target_id in candidate_ids:
            source = next((name for name, value in selected.items() if value == target_id), None)
            candidates.append(
                {
                    "target_id": target_id,
                    "evidence_ids": (
                        positive_evidence
                        if source == "corrupted_path"
                        else positive_evidence_by_target.get(
                            target_id, evidence_by_target[target_id]
                        )
                    ),
                }
            )

        direct_hard_ids = [
            target_id
            for target_id in evidence_positive_target_ids
            if target_id not in direct_positive_target_ids
        ]
        direct_negative_ids = list(dict.fromkeys([*direct_hard_ids, *negative_ids]))
        for positive_id in direct_positive_target_ids:
            edge_lists.append(
                {
                    "query_id": query_id,
                    "source_type": "table",
                    "positive_id": positive_id,
                    "candidate_ids": [positive_id, *direct_negative_ids],
                    "destination_type": "table",
                    "dataset": dataset_name,
                    "split": split,
                }
            )

        positive_evidence_ids = []
        for positive_id in evidence_positive_target_ids:
            for evidence_id in positive_evidence_by_target[positive_id]:
                if evidence_id not in positive_evidence_ids:
                    positive_evidence_ids.append(evidence_id)
        positive_evidence_set = set(positive_evidence_ids)

        evidence_negatives: dict[str, list[str]] = defaultdict(list)
        for candidate in candidates:
            if candidate["target_id"] in evidence_positive_target_ids:
                continue
            seen_type = set()
            for evidence_id in candidate["evidence_ids"]:
                evidence_type = asset_types[evidence_id]
                if (
                    evidence_type in seen_type
                    or evidence_id in positive_evidence_set
                    or evidence_id in evidence_negatives[evidence_type]
                ):
                    continue
                seen_type.add(evidence_type)
                evidence_negatives[evidence_type].append(evidence_id)

        for evidence_type in sorted({asset_types[evidence_id] for evidence_id in positive_evidence_ids}):
            if evidence_negatives[evidence_type]:
                continue
            fallback_id = _pick_random(
                evidence_by_type[evidence_type],
                positive_evidence_set,
                rng,
            )
            if fallback_id is not None:
                evidence_negatives[evidence_type].append(fallback_id)

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
                    "dataset": dataset_name,
                    "split": split,
                }
            )

        evidence_positive_target_set = set(evidence_positive_target_ids)
        for positive_target_id in evidence_positive_target_ids:
            for evidence_id in positive_evidence_by_target[positive_target_id]:
                edge_key = (evidence_id, positive_target_id)
                if edge_key in emitted_evidence_target_edges:
                    continue
                excluded_targets = (
                    targets_by_evidence[evidence_id]
                    | evidence_positive_target_set
                )
                evidence_target_negatives = [
                    target_id for target_id in candidate_ids if target_id not in excluded_targets
                ]
                if not evidence_target_negatives:
                    fallback_id = _pick_random(target_ids, excluded_targets, rng)
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
                        "dataset": dataset_name,
                        "split": split,
                    }
                )
        target_lists.append(
            {
                "query_id": query_id,
                "direct_positive_target_id": direct_positive_target_ids[0],
                "evidence_positive_target_id": evidence_positive_target_ids[0],
                "positive_target_ids": positive_target_ids,
                "candidates": candidates,
                "dataset": dataset_name,
                "split": split,
            }
        )

    if not target_lists:
        raise ValueError(
            "No query has both a positive and a non-joinable target in the shared data lake"
        )
    return {
        "stage1_objects": [*query_objects.values(), *target_objects.values(), *asset_objects],
        "edge_lists": edge_lists,
        "target_lists": target_lists,
        "stage1_corpus": [
            *(
                {"object_id": target_id}
                for target_id in target_objects
            ),
            *(
                {"object_id": record["object_id"]}
                for record in asset_objects
            ),
        ],
    }
