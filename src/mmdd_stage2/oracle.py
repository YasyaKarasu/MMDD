"""Oracle-positive examples for the Stage-2 candidate-column experiment."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mmdd_dataset.wdc_runtime import iter_dataset_artifact

from .data import Stage2ObjectIndex, local_column_index
from .verifier import EvidenceBundle

ORACLE_EVIDENCE_POLICY = (
    "sort text/image by asset_id; keep one of each when both exist; "
    "fill remaining slots by asset_id; deduplicate; top_k=4"
)


@dataclass(frozen=True)
class OracleColumnExample:
    dataset: str
    dataset_root: str
    split: str
    query_id: str
    target_id: str
    source_table_id: str
    chain_id: str
    gold_source_column: int
    gold_local_column: int
    gold_column_position: int
    candidate_column_indices: tuple[int, ...]
    positive_bundle: EvidenceBundle
    evidence_modalities: tuple[str, ...]
    evidence_count_before_truncation: int


class OracleDataError(ValueError):
    """Raised when an Oracle-positive dataset audit is not clean."""

    def __init__(self, message: str, audit: dict[str, Any]) -> None:
        super().__init__(message)
        self.audit = audit


def dataset_name(root: Path) -> str:
    name = root.name.lower()
    if "entitables" in name:
        return "entitables"
    if "wdc" in name:
        return "wdc"
    return root.name


def _distribution(values: Sequence[int]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(Counter(values).items())}


def _modality_bucket(modalities: Sequence[str]) -> str:
    kinds = set(modalities)
    if kinds == {"text"}:
        return "text_only"
    if kinds == {"image"}:
        return "image_only"
    if kinds == {"text", "image"}:
        return "text_image"
    return "unknown"


def select_oracle_evidence(
    evidence_ids: Sequence[str],
    evidence: dict[str, dict[str, Any]],
    *,
    top_k: int,
) -> tuple[str, ...]:
    """Apply the fixed, split-independent Oracle evidence policy."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    unique_ids = sorted(set(evidence_ids))
    text_ids = sorted(
        asset_id
        for asset_id in unique_ids
        if evidence[asset_id].get("asset_type") == "text"
    )
    image_ids = sorted(
        asset_id
        for asset_id in unique_ids
        if evidence[asset_id].get("asset_type") == "image"
    )
    unsupported = set(unique_ids) - set(text_ids) - set(image_ids)
    if unsupported:
        raise ValueError(
            "Unsupported Oracle evidence asset types: " + ", ".join(sorted(unsupported))
        )

    selected: list[str] = []
    if text_ids and image_ids:
        selected.extend((text_ids[0], image_ids[0]))
    for asset_id in sorted(unique_ids):
        if asset_id not in selected:
            selected.append(asset_id)
    return tuple(selected[:top_k])


def _resolve_image_path(root: Path, record: dict[str, Any]) -> Path:
    local = Path(str(record.get("local_path", "")))
    relative = root / str(record.get("relative_path", ""))
    if local.is_file():
        return local.resolve()
    if relative.is_file():
        return relative.resolve()
    return local


def _merge_objects(
    destination: dict[str, dict[str, Any]],
    owners: dict[str, str],
    records: dict[str, dict[str, Any]],
    *,
    dataset: str,
    kind: str,
    conflicts: list[dict[str, str]],
) -> None:
    for object_id, record in records.items():
        if object_id in owners and owners[object_id] != dataset:
            conflicts.append(
                {
                    "kind": kind,
                    "id": object_id,
                    "first_dataset": owners[object_id],
                    "second_dataset": dataset,
                }
            )
            continue
        owners[object_id] = dataset
        destination[object_id] = record


def _selected_artifact_records(
    root: Path, artifact: str, selected_ids: set[str]
) -> dict[str, dict[str, Any]]:
    id_key = "asset_id" if artifact == "bridge_assets" else "table_id"
    records: dict[str, dict[str, Any]] = {}
    for record in iter_dataset_artifact(root, artifact):
        object_id = str(record[id_key])
        if object_id in selected_ids:
            records[object_id] = record
    return records


def _resolve_source_backed_targets(
    root: Path, targets: dict[str, dict[str, Any]]
) -> None:
    source_ids = {
        str(record["source_table_ref"]["source_table_id"])
        for record in targets.values()
        if record.get("source_table_ref")
    }
    if not source_ids:
        return
    sources = {
        str(record["source_table_id"]): record
        for record in iter_dataset_artifact(root, "source_tables")
        if str(record["source_table_id"]) in source_ids
    }
    missing = source_ids - sources.keys()
    if missing:
        raise KeyError("source_tables has no records for: " + ", ".join(sorted(missing)))
    for target_id, target in list(targets.items()):
        reference = target.get("source_table_ref")
        if reference:
            targets[target_id] = {
                **sources[str(reference["source_table_id"])],
                **{key: value for key, value in target.items() if key != "source_table_ref"},
            }


def load_oracle_column_data(
    dataset_roots: Sequence[Path],
    *,
    splits: Sequence[str] = ("train", "dev", "test"),
    top_k_evidence: int = 4,
    strict: bool = True,
) -> tuple[list[OracleColumnExample], Stage2ObjectIndex, dict[str, Any]]:
    """Load audited gold target/evidence examples from canonical dataset roots."""

    roots = [Path(root) for root in dataset_roots]
    selected_splits = tuple(dict.fromkeys(splits))
    if not roots:
        raise ValueError("At least one dataset root is required")
    if not selected_splits or any(split not in {"train", "dev", "test"} for split in selected_splits):
        raise ValueError("splits must contain train, dev, and/or test")
    if top_k_evidence <= 0:
        raise ValueError("top_k_evidence must be positive")
    names = [dataset_name(root) for root in roots]
    if len(set(names)) != len(names):
        raise ValueError("Dataset roots must have distinct dataset names")

    examples: list[OracleColumnExample] = []
    merged = Stage2ObjectIndex({}, {}, {})
    owners: dict[str, dict[str, str]] = {
        "query": {},
        "target": {},
        "evidence": {},
    }
    conflicts: list[dict[str, str]] = []
    dataset_audits: dict[str, Any] = {}
    split_sources: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"source_table_ids": set(), "chain_ids": set()}
    )

    for root, dataset in zip(roots, names, strict=True):
        qrels = [
            record
            for record in iter_dataset_artifact(root, "qrels")
            if record.get("reason") == "model_recoverable_join_column"
            and str(record.get("split", "train")) in selected_splits
        ]
        duplicate_qrel_pairs = [
            f"{query_id}->{target_id}"
            for (query_id, target_id), count in Counter(
                (
                    str(record["query_table_id"]),
                    str(record["target_table_id"]),
                )
                for record in qrels
            ).items()
            if count > 1
        ]
        qrel_pairs = {
            (str(record["query_table_id"]), str(record["target_table_id"]))
            for record in qrels
        }
        recoveries: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in iter_dataset_artifact(root, "evidence_recoveries"):
            pair = (str(record["query_table_id"]), str(record["target_table_id"]))
            if pair in qrel_pairs and str(record.get("split", "train")) in selected_splits:
                recoveries[pair].append(record)

        query_ids = {pair[0] for pair in qrel_pairs}
        target_ids = {pair[1] for pair in qrel_pairs}
        recovery_evidence_ids = {
            str(record["evidence"]["asset_id"])
            for records in recoveries.values()
            for record in records
            if record.get("evidence", {}).get("asset_id") is not None
        }
        queries = _selected_artifact_records(root, "query_tables", query_ids)
        targets = _selected_artifact_records(root, "data_lake_tables", target_ids)
        _resolve_source_backed_targets(root, targets)
        evidence = _selected_artifact_records(
            root, "bridge_assets", recovery_evidence_ids
        )

        missing = {
            "query": sorted(query_ids - queries.keys()),
            "target": sorted(target_ids - targets.keys()),
            "evidence": sorted(recovery_evidence_ids - evidence.keys()),
            "image_file": [],
            "recovery": [],
            "split_mismatch": [],
            "column_mapping": [],
        }
        for asset_id, record in evidence.items():
            if record.get("asset_type") == "image":
                image_path = _resolve_image_path(root, record)
                if image_path.is_file():
                    record["local_path"] = str(image_path)
                else:
                    missing["image_file"].append(asset_id)

        usable_for_root: list[OracleColumnExample] = []
        for qrel in qrels:
            query_id = str(qrel["query_table_id"])
            target_id = str(qrel["target_table_id"])
            split = str(qrel.get("split", "train"))
            pair = (query_id, target_id)
            records = recoveries.get(pair, [])
            if not records:
                missing["recovery"].append(f"{query_id}->{target_id}")
                continue
            if query_id not in queries or target_id not in targets:
                continue
            if str(queries[query_id].get("split", split)) != split or str(
                targets[target_id].get("split", split)
            ) != split:
                missing["split_mismatch"].append(f"{query_id}->{target_id}")
                continue
            all_evidence_ids = sorted(
                {
                    str(record["evidence"]["asset_id"])
                    for record in records
                    if record.get("evidence", {}).get("asset_id") is not None
                }
            )
            if any(asset_id not in evidence for asset_id in all_evidence_ids):
                continue
            if any(
                evidence[asset_id].get("asset_type") == "image"
                and asset_id in missing["image_file"]
                for asset_id in all_evidence_ids
            ):
                continue
            selected = select_oracle_evidence(
                all_evidence_ids, evidence, top_k=top_k_evidence
            )
            target = targets[target_id]
            source_column = int(qrel["join_attribute"]["source_column_index"])
            try:
                local_column = local_column_index(target, source_column)
                position = next(
                    index
                    for index, column in enumerate(target["columns"])
                    if int(column["column_index"]) == local_column
                )
            except (KeyError, StopIteration):
                missing["column_mapping"].append(f"{query_id}->{target_id}:{source_column}")
                continue
            modalities = tuple(str(evidence[asset_id]["asset_type"]) for asset_id in selected)
            example = OracleColumnExample(
                dataset=dataset,
                dataset_root=str(root.resolve()),
                split=split,
                query_id=query_id,
                target_id=target_id,
                source_table_id=str(qrel.get("source_table_id", "")),
                chain_id=str(qrel.get("chain_id", "")),
                gold_source_column=source_column,
                gold_local_column=local_column,
                gold_column_position=position,
                candidate_column_indices=tuple(
                    int(column["column_index"]) for column in target["columns"]
                ),
                positive_bundle=EvidenceBundle(target_id, 0.0, selected),
                evidence_modalities=modalities,
                evidence_count_before_truncation=len(all_evidence_ids),
            )
            usable_for_root.append(example)
            split_sources[split]["source_table_ids"].add(example.source_table_id)
            split_sources[split]["chain_ids"].add(example.chain_id)

        examples.extend(usable_for_root)
        selected_query_ids = {example.query_id for example in usable_for_root}
        selected_target_ids = {example.target_id for example in usable_for_root}
        selected_evidence_ids = {
            asset_id
            for example in usable_for_root
            for asset_id in example.positive_bundle.evidence_ids
        }
        _merge_objects(
            merged.queries,
            owners["query"],
            {key: queries[key] for key in selected_query_ids},
            dataset=dataset,
            kind="query",
            conflicts=conflicts,
        )
        _merge_objects(
            merged.targets,
            owners["target"],
            {key: targets[key] for key in selected_target_ids},
            dataset=dataset,
            kind="target",
            conflicts=conflicts,
        )
        _merge_objects(
            merged.evidence,
            owners["evidence"],
            {key: evidence[key] for key in selected_evidence_ids},
            dataset=dataset,
            kind="evidence",
            conflicts=conflicts,
        )

        per_split: dict[str, Any] = {}
        for split in selected_splits:
            split_qrels = [
                record for record in qrels if str(record.get("split", "train")) == split
            ]
            split_examples = [item for item in usable_for_root if item.split == split]
            per_split[split] = {
                "qrels": len(split_qrels),
                "usable_examples": len(split_examples),
                "evidence_count_before_truncation": _distribution(
                    [item.evidence_count_before_truncation for item in split_examples]
                ),
                "evidence_count_selected": _distribution(
                    [len(item.positive_bundle.evidence_ids) for item in split_examples]
                ),
                "modality": dict(
                    sorted(Counter(_modality_bucket(item.evidence_modalities) for item in split_examples).items())
                ),
                "candidate_column_count": _distribution(
                    [len(item.candidate_column_indices) for item in split_examples]
                ),
                "gold_column_position": _distribution(
                    [item.gold_column_position for item in split_examples]
                ),
            }
        dataset_audits[dataset] = {
            "root": str(root.resolve()),
            "duplicate_qrel_pairs": sorted(duplicate_qrel_pairs),
            "missing": {key: sorted(value) for key, value in missing.items()},
            "splits": per_split,
        }

    leakage: dict[str, dict[str, list[str]]] = {}
    for left_index, left in enumerate(selected_splits):
        for right in selected_splits[left_index + 1 :]:
            key = f"{left}__{right}"
            leakage[key] = {
                field: sorted(split_sources[left][field] & split_sources[right][field])
                for field in ("source_table_ids", "chain_ids")
            }

    audit = {
        "format_version": 2,
        "training_source": "oracle_positive",
        "evidence_policy": ORACLE_EVIDENCE_POLICY,
        "top_k_evidence": top_k_evidence,
        "datasets": dataset_audits,
        "id_conflicts": conflicts,
        "cross_split_intersections": leakage,
        "total_usable_examples": len(examples),
    }
    failures = []
    for dataset, item in dataset_audits.items():
        if item["duplicate_qrel_pairs"]:
            failures.append(f"{dataset}: duplicate qrels")
        for kind, ids in item["missing"].items():
            if ids:
                failures.append(f"{dataset}: missing/invalid {kind} ({len(ids)})")
    if conflicts:
        failures.append(f"cross-dataset ID conflicts ({len(conflicts)})")
    for pair, intersections in leakage.items():
        for kind, ids in intersections.items():
            if any(ids):
                failures.append(f"{pair}: cross-split {kind} ({len(ids)})")
    if strict and failures:
        raise OracleDataError("Oracle-positive data audit failed: " + "; ".join(failures), audit)

    examples.sort(key=lambda item: (item.dataset, item.split, item.query_id, item.target_id))
    return examples, merged, audit


def oracle_examples_fingerprint(examples: Sequence[OracleColumnExample]) -> str:
    records = []
    for example in examples:
        record = asdict(example)
        record["positive_bundle"] = asdict(example.positive_bundle)
        records.append(record)
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
