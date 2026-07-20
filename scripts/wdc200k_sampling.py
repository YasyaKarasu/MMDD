"""Deterministic theoretical prefiltering and entity sampling for WDC 200K."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import build_mm_joinability_dataset as join_builder
from stage1_io import clean_text, stable_hash
from wdc200k_io import (
    AtomicJsonlShard,
    CompletedShard,
    PreWriteGuard,
    StageFingerprint,
    StageManifest,
    validate_completed_shard,
)


SAMPLING_SCHEMA_VERSION = "wdc200k-entity-sampling-v1"
ARTIFACTS = (
    "sampled_entities",
    "sampled_page_refs",
    "sampled_direct_image_refs",
    "prefilter_tables",
)
STRATA = ("both", "page_only", "direct_only")


@dataclass(frozen=True)
class SamplingPolicy:
    sampled_entities_per_table: int = 8
    entity_sampling_seed: int = 20260720
    query_rows_per_table: int = 5
    min_column_non_empty_ratio: float = 0.5
    min_recovered_value_ratio: float = 0.6
    min_recovery_denominator: int = 2
    min_rows_per_output_table: int = 2
    global_entity_budget: int | None = None

    def __post_init__(self) -> None:
        if self.sampled_entities_per_table < 3:
            raise ValueError("sampled_entities_per_table must be at least 3")
        if self.query_rows_per_table <= 0:
            raise ValueError("query_rows_per_table must be positive")
        if not 0.0 <= self.min_column_non_empty_ratio <= 1.0:
            raise ValueError("min_column_non_empty_ratio must be in [0, 1]")
        if not 0.0 <= self.min_recovered_value_ratio <= 1.0:
            raise ValueError("min_recovered_value_ratio must be in [0, 1]")
        if self.min_recovery_denominator < 0:
            raise ValueError("min_recovery_denominator must be non-negative")
        if self.min_rows_per_output_table <= 0:
            raise ValueError("min_rows_per_output_table must be positive")
        if self.global_entity_budget is not None and self.global_entity_budget < 0:
            raise ValueError("global_entity_budget must be non-negative")

    def validate_budget(self, *, eligible_tables: int) -> None:
        required = eligible_tables * self.sampled_entities_per_table
        if self.global_entity_budget is not None and self.global_entity_budget < required:
            raise ValueError(
                "global entity budget cannot cover every eligible table: "
                f"budget={self.global_entity_budget}, required={required}"
            )


@dataclass(frozen=True)
class SamplingResult:
    output_root: Path
    manifest_path: Path
    artifact_paths: dict[str, tuple[Path, ...]]
    eligible_tables: int
    rejected_tables: int
    sampled_entities: int
    sampled_page_refs: int
    sampled_direct_image_refs: int
    strata: dict[str, int]


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"JSONL record must be an object: {path}")
                yield record


def _http_url(record: Mapping[str, Any], field: str) -> bool:
    value = clean_text(record.get(field)).casefold()
    return value.startswith("http://") or value.startswith("https://")


def _appearance(entity: Mapping[str, Any]) -> tuple[str, Any]:
    appearances = entity.get("appears_in") or []
    if len(appearances) != 1 or not isinstance(appearances[0], dict):
        raise ValueError("structural entity must have exactly one appearance")
    appearance = appearances[0]
    return str(appearance["source_table_id"]), appearance["row_id"]


def _cells_by_index(source_table: Mapping[str, Any]) -> dict[Any, dict[int, str]]:
    result: dict[Any, dict[int, str]] = {}
    for fallback, row in enumerate(source_table.get("rows") or []):
        row_id = row.get("row_id", fallback)
        result[row_id] = {
            int(cell["column_index"]): clean_text(cell.get("text"))
            for cell in row.get("cells") or []
        }
    return result


def _rank(policy: SamplingPolicy, table_id: str, entity_id: str, purpose: str) -> str:
    return stable_hash(
        "wdc200k-entity-sampling",
        policy.entity_sampling_seed,
        table_id,
        purpose,
        entity_id,
        length=40,
    )


def sample_table_entities(
    source_table: Mapping[str, Any],
    entities: Iterable[dict[str, Any]],
    page_refs: Iterable[dict[str, Any]],
    direct_image_refs: Iterable[dict[str, Any]],
    policy: SamplingPolicy,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return sampled entity records and a table-level prefilter decision."""
    table_id = str(source_table["source_table_id"])
    decision: dict[str, Any] = {
        "source_table_id": table_id,
        "eligible": False,
        "reason": "",
        "sampled_entities": 0,
        "anchor_candidate_attribute": None,
    }
    if int(source_table.get("num_cols", 0)) < 3:
        decision["reason"] = "fewer_than_three_columns"
        return [], decision
    entity_columns = source_table.get("metadata", {}).get("candidate_entity_columns") or []
    if not entity_columns:
        decision["reason"] = "missing_entity_column"
        return [], decision
    entity_column = int(entity_columns[0])
    cells = _cells_by_index(source_table)
    non_empty_entity_rows = {
        row_id for row_id, values in cells.items() if clean_text(values.get(entity_column))
    }
    if len(non_empty_entity_rows) < policy.query_rows_per_table:
        decision["reason"] = "fewer_than_query_rows_non_empty_entities"
        return [], decision
    candidate_indexes = join_builder.candidate_attribute_columns(
        dict(source_table), entity_column, policy.min_column_non_empty_ratio
    )
    names = {
        int(column["column_index"]): str(column["column_name"])
        for column in source_table.get("columns") or []
    }
    candidates = [(index, names[index]) for index in candidate_indexes]
    if not candidates:
        decision["reason"] = "no_non_empty_candidate_attribute"
        return [], decision

    entities_by_id: dict[str, dict[str, Any]] = {}
    row_by_entity: dict[str, Any] = {}
    for entity in entities:
        entity_table, row_id = _appearance(entity)
        if entity_table != table_id:
            continue
        entity_id = str(entity["entity_id"])
        entities_by_id[entity_id] = entity
        row_by_entity[entity_id] = row_id
    page_by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for reference in page_refs:
        entity_id = str(reference.get("entity_id"))
        if entity_id in entities_by_id and _http_url(reference, "page_url"):
            page_by_entity[entity_id].append(reference)
    image_by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for reference in direct_image_refs:
        entity_id = str(reference.get("entity_id"))
        if entity_id in entities_by_id and _http_url(reference, "image_url"):
            image_by_entity[entity_id].append(reference)

    supporting: dict[int, list[str]] = {index: [] for index, _name in candidates}
    supported_names: dict[str, list[str]] = defaultdict(list)
    attemptable: list[str] = []
    for entity_id, row_id in row_by_entity.items():
        has_material = bool(page_by_entity[entity_id] or image_by_entity[entity_id])
        if row_id not in non_empty_entity_rows or not has_material:
            continue
        attemptable.append(entity_id)
        for column_index, column_name in candidates:
            if clean_text(cells.get(row_id, {}).get(column_index)):
                supporting[column_index].append(entity_id)
                supported_names[entity_id].append(column_name)
    viable = [
        (index, name, supporting[index])
        for index, name in candidates
        if len(supporting[index]) >= 3
    ]
    if not viable:
        decision["reason"] = "insufficient_material_support_for_candidate"
        return [], decision
    anchor_index, anchor_name, anchor_entities = min(
        viable, key=lambda item: (-len(item[2]), item[0], item[1])
    )
    del anchor_index

    selected: list[tuple[str, str]] = []
    anchor_order = sorted(
        set(anchor_entities),
        key=lambda entity_id: (_rank(policy, table_id, entity_id, "anchor"), entity_id),
    )
    selected.extend((entity_id, "anchor_support") for entity_id in anchor_order[:3])
    selected_ids = {entity_id for entity_id, _reason in selected}

    queues: dict[str, deque[str]] = {}
    for stratum in STRATA:
        members = []
        for entity_id in set(attemptable) - selected_ids:
            has_page = bool(page_by_entity[entity_id])
            has_image = bool(image_by_entity[entity_id])
            actual = "both" if has_page and has_image else "page_only" if has_page else "direct_only"
            if actual == stratum:
                members.append(entity_id)
        queues[stratum] = deque(
            sorted(
                members,
                key=lambda entity_id: (
                    _rank(policy, table_id, entity_id, f"stratum:{stratum}"),
                    entity_id,
                ),
            )
        )
    while len(selected) < policy.sampled_entities_per_table and any(queues.values()):
        for stratum in STRATA:
            if len(selected) >= policy.sampled_entities_per_table:
                break
            if queues[stratum]:
                entity_id = queues[stratum].popleft()
                selected.append((entity_id, "stratified_fill"))
                selected_ids.add(entity_id)

    output = []
    for sampling_rank, (entity_id, reason) in enumerate(selected, 1):
        entity = dict(entities_by_id[entity_id])
        row_id = row_by_entity[entity_id]
        has_page = bool(page_by_entity[entity_id])
        has_image = bool(image_by_entity[entity_id])
        stratum = "both" if has_page and has_image else "page_only" if has_page else "direct_only"
        entity.update(
            {
                "source_table_id": table_id,
                "source_row_id": row_id,
                "stratum": stratum,
                "sampling_rank": sampling_rank,
                "selection_reason": reason,
                "supporting_candidate_attributes": sorted(supported_names[entity_id]),
                "anchor_candidate_attribute": anchor_name,
            }
        )
        output.append(entity)
    decision.update(
        {
            "eligible": True,
            "reason": "eligible",
            "sampled_entities": len(output),
            "attemptable_entities": len(set(attemptable)),
            "anchor_candidate_attribute": anchor_name,
            "anchor_material_support": len(set(anchor_entities)),
        }
    )
    return output, decision


def _manifest_artifacts(manifest_path: Path, root: Path) -> dict[str, Path]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("complete") is not True:
        raise ValueError(f"structural manifest is incomplete: {manifest_path}")
    result: dict[str, Path] = {}
    for raw in payload.get("completed_shards") or []:
        completed = CompletedShard(
            path=str(raw["path"]),
            records=int(raw["records"]),
            bytes=int(raw["bytes"]),
            sha256=str(raw["sha256"]),
        )
        path = (root / completed.path).resolve()
        if (
            not path.is_relative_to(root.resolve())
            or not path.is_file()
            or path.stat().st_size != completed.bytes
        ):
            raise ValueError(f"structural shard metadata failed: {completed.path}")
        prefix = completed.path.split("/", 1)[0]
        if prefix in {"source_tables", "entities", "page_refs", "direct_image_refs"}:
            if prefix in result:
                raise ValueError(f"duplicate structural artifact {prefix}")
            result[prefix] = path
    required = {"source_tables", "entities", "page_refs", "direct_image_refs"}
    if set(result) != required:
        raise ValueError("structural manifest is missing sampling inputs")
    return result


def _relative_completed(completed: CompletedShard, path: Path, root: Path) -> CompletedShard:
    return CompletedShard(
        path=path.relative_to(root).as_posix(),
        records=completed.records,
        bytes=completed.bytes,
        sha256=completed.sha256,
    )


def _paths_from_manifest(manifest: StageManifest, root: Path) -> dict[str, tuple[Path, ...]]:
    grouped: dict[str, list[Path]] = {artifact: [] for artifact in ARTIFACTS}
    for completed in manifest.completed_shards:
        prefix = completed.path.split("/", 1)[0]
        if prefix in grouped:
            grouped[prefix].append(root / completed.path)
    return {key: tuple(sorted(paths)) for key, paths in grouped.items()}


def _result(root: Path, manifest: StageManifest) -> SamplingResult:
    paths = _paths_from_manifest(manifest, root)
    eligible = 0
    rejected = 0
    for path in paths["prefilter_tables"]:
        for record in _iter_jsonl(path):
            if bool(record["eligible"]):
                eligible += 1
            else:
                rejected += 1
    strata = {stratum: 0 for stratum in STRATA}
    sampled_entities = 0
    for path in paths["sampled_entities"]:
        for record in _iter_jsonl(path):
            sampled_entities += 1
            strata[str(record["stratum"])] += 1
    sampled_page_refs = sum(
        1
        for path in paths["sampled_page_refs"]
        for _record in _iter_jsonl(path)
    )
    sampled_direct_image_refs = sum(
        1
        for path in paths["sampled_direct_image_refs"]
        for _record in _iter_jsonl(path)
    )
    return SamplingResult(
        output_root=root,
        manifest_path=manifest.path,
        artifact_paths=paths,
        eligible_tables=eligible,
        rejected_tables=rejected,
        sampled_entities=sampled_entities,
        sampled_page_refs=sampled_page_refs,
        sampled_direct_image_refs=sampled_direct_image_refs,
        strata=strata,
    )


def sample_structural_artifacts(
    *,
    structural_output_root: Path,
    structural_manifests: Sequence[Path],
    output_root: Path,
    policy: SamplingPolicy,
    pre_write_guard: PreWriteGuard | None = None,
) -> SamplingResult:
    """Stream completed structural shards into durable sampled artifacts."""
    structural_output_root = Path(structural_output_root)
    output_root = Path(output_root)
    manifests = tuple(sorted(Path(path).resolve() for path in structural_manifests))
    if not manifests:
        raise ValueError("sampling requires structural manifests")
    fingerprint = StageFingerprint(
        stage="wdc200k_entity_sampling",
        input_fingerprint=stable_hash(
            SAMPLING_SCHEMA_VERSION,
            *(f"{path.as_posix()}:{_sha256_path(path)}" for path in manifests),
            length=40,
        ),
        parameter_fingerprint=stable_hash(
            SAMPLING_SCHEMA_VERSION, json.dumps(asdict(policy), sort_keys=True), length=40
        ),
        schema_version=SAMPLING_SCHEMA_VERSION,
    )
    manifest = StageManifest(
        output_root / "manifest.json", fingerprint, pre_write_guard=pre_write_guard
    )
    if manifest.complete:
        if not manifest.completed_shards or not all(
            validate_completed_shard(shard, output_root) for shard in manifest.completed_shards
        ):
            raise ValueError("completed sampling manifest failed shard validation")
        return _result(output_root, manifest)
    try:
        for index, structural_manifest in enumerate(manifests):
            expected_relatives = {
                artifact: f"{artifact}/part-{index:05d}.jsonl"
                for artifact in ARTIFACTS
            }
            recorded = {
                shard.path: shard for shard in manifest.completed_shards
            }
            reusable = {
                artifact
                for artifact, relative in expected_relatives.items()
                if relative in recorded
                and validate_completed_shard(recorded[relative], output_root)
            }
            invalid_recorded = [
                relative
                for relative in expected_relatives.values()
                if relative in recorded
                and not validate_completed_shard(recorded[relative], output_root)
            ]
            if invalid_recorded:
                raise ValueError(
                    "recorded sampling shards failed validation: "
                    + ", ".join(invalid_recorded)
                )
            if len(reusable) == len(ARTIFACTS):
                continue
            inputs = _manifest_artifacts(structural_manifest, structural_output_root)
            entities_by_table: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for entity in _iter_jsonl(inputs["entities"]):
                table_id, _row_id = _appearance(entity)
                entities_by_table[table_id].append(entity)
            pages_by_table: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for reference in _iter_jsonl(inputs["page_refs"]):
                pages_by_table[str(reference["source_table_id"])].append(reference)
            images_by_table: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for reference in _iter_jsonl(inputs["direct_image_refs"]):
                images_by_table[str(reference["source_table_id"])].append(reference)
            writers = {
                artifact: AtomicJsonlShard(
                    output_root / artifact / f"part-{index:05d}.jsonl",
                    pre_write_guard=pre_write_guard,
                )
                for artifact in ARTIFACTS
                if artifact not in reusable
            }
            try:
                for source_table in _iter_jsonl(inputs["source_tables"]):
                    table_id = str(source_table["source_table_id"])
                    sampled, decision = sample_table_entities(
                        source_table,
                        entities_by_table.get(table_id, ()),
                        pages_by_table.get(table_id, ()),
                        images_by_table.get(table_id, ()),
                        policy,
                    )
                    if "prefilter_tables" in writers:
                        writers["prefilter_tables"].write(decision)
                    selected_ids = {record["entity_id"] for record in sampled}
                    if "sampled_entities" in writers:
                        for record in sampled:
                            writers["sampled_entities"].write(record)
                    if "sampled_page_refs" in writers:
                        selected_pages = sorted(
                            (
                                reference
                                for reference in pages_by_table.get(table_id, ())
                                if reference.get("entity_id") in selected_ids
                            ),
                            key=lambda item: (
                                str(item.get("entity_id")),
                                str(item.get("url_key")),
                                str(item.get("page_url")),
                            ),
                        )
                        for reference in selected_pages:
                            writers["sampled_page_refs"].write(reference)
                    if "sampled_direct_image_refs" in writers:
                        selected_images = sorted(
                            (
                                reference
                                for reference in images_by_table.get(table_id, ())
                                if reference.get("entity_id") in selected_ids
                            ),
                            key=lambda item: (
                                str(item.get("entity_id")),
                                int(item.get("ordinal", 0)),
                                str(item.get("url_key")),
                            ),
                        )
                        for reference in selected_images:
                            writers["sampled_direct_image_refs"].write(reference)
                committed = []
                for artifact in ARTIFACTS:
                    if artifact not in writers:
                        continue
                    path = writers[artifact].path
                    completed = writers[artifact].commit()
                    committed.append(_relative_completed(completed, path, output_root))
                for completed in committed:
                    manifest.record_shard(completed)
            except BaseException:
                for writer in writers.values():
                    writer.abort()
                raise
        current = _result(output_root, manifest)
        policy.validate_budget(eligible_tables=current.eligible_tables)
        manifest.mark_complete()
    except BaseException:
        raise
    return _result(output_root, manifest)


def iter_sampled_records(
    result: SamplingResult, artifact: str
) -> Iterator[dict[str, Any]]:
    if artifact not in ARTIFACTS:
        raise ValueError(f"unknown sampled artifact: {artifact}")
    for path in result.artifact_paths[artifact]:
        yield from _iter_jsonl(path)
