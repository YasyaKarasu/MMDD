"""Deterministic, dev-only locks for the S2-R4c FAST experiment."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps

from .r4_common import file_hash, read_jsonl
from .r4_schedule import visible_query_rows
from .r4c_fast_types import FastUnit

COMPONENT_SALT = "R4C_FAST_COMPONENT_V1"
NATURAL_SALT = "R4C_FAST_LABEL_BLIND_V2"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _read_objects(r4: Path) -> dict[str, Any]:
    with gzip.open(r4 / "READER_OBJECTS.dev.jsonl.gz", "rt", encoding="utf-8") as handle:
        return json.loads(next(line for line in handle if line.strip()))


def _read_schedule(r4: Path) -> list[dict[str, Any]]:
    path = r4 / "RECOVERY_SCHEDULES" / "S1_JointGlobal.dev.json"
    return json.loads(path.read_text())["records"]


def _read_lock(r4: Path) -> dict[str, dict[str, Any]]:
    return {record["query_id"]: record for record in read_jsonl(r4 / "CANDIDATE_LOCK.dev.jsonl.gz")}


def _target(lock: dict[str, dict[str, Any]], query_id: str, target_id: str) -> dict[str, Any]:
    return next(item for item in lock[query_id]["targets"] if item["target_id"] == target_id)


def _column_name(objects: dict[str, Any], target_id: str, column_id: int) -> str:
    return str(next(
        column["column_name"]
        for column in objects["targets"][target_id]["columns"]
        if int(column["column_index"]) == int(column_id)
    ))


def _cells(row: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    ordered = sorted(row["cells"], key=lambda cell: int(cell["column_id"]))
    return tuple((str(cell["column_name"]), str(cell.get("text", ""))) for cell in ordered)


def _image_decodes(item: dict[str, Any], memo: dict[str, bool]) -> bool:
    path = str(item.get("local_path", ""))
    if path in memo:
        return memo[path]
    try:
        with Image.open(path) as image:
            image.seek(0)
            ImageOps.exif_transpose(image).convert("RGB").load()
        memo[path] = True
    except (OSError, ValueError, Image.DecompressionBombError):
        memo[path] = False
    return memo[path]


def build_component_candidates(
    r4: Path, r3: Path
) -> tuple[list[FastUnit], list[dict[str, Any]]]:
    objects = _read_objects(r4)
    lock = _read_lock(r4)
    schedule = _read_schedule(r4)
    gt_index: dict[tuple[str, str, int, int], dict[str, set[Any]]] = {}
    for row in read_jsonl(r3 / "ROW_GT.dev.jsonl"):
        key = (
            str(row["query_id"]),
            str(row["target_id"]),
            int(row["query_row_id"]),
            int(row["local_column_index"]),
        )
        entry = gt_index.setdefault(key, {"gold_values": set(), "target_rows": set(), "witness": set()})
        entry["gold_values"].add(str(row["gold_value_raw"]))
        entry["target_rows"].update(int(item) for item in row.get("target_row_ids", []))
        entry["witness"].update(str(item) for item in row.get("witness_evidence_ids", []))

    units: list[FastUnit] = []
    sidecars: list[dict[str, Any]] = []
    decode_memo: dict[str, bool] = {}
    seen: set[str] = set()
    for record in schedule:
        query_id = str(record["query_id"])
        rows = visible_query_rows(objects, query_id)
        for branch in record["branches"]:
            target_id = str(branch["target_id"])
            column_id = int(branch["column_id"])
            target = _target(lock, query_id, target_id)
            retained = tuple(str(item) for item in target["retained_evidence_ids"])
            for row in rows:
                key = (query_id, target_id, int(row["query_row_id"]), column_id)
                evaluation = gt_index.get(key)
                if evaluation is None:
                    continue
                witness = evaluation["witness"]
                valid = [
                    evidence_id
                    for evidence_id in retained
                    if evidence_id in witness
                    and objects["evidence"][evidence_id].get("asset_type") == "image"
                    and _image_decodes(objects["evidence"][evidence_id], decode_memo)
                ]
                if not valid:
                    continue
                unit = FastUnit.create(
                    query_id=query_id,
                    target_id=target_id,
                    column_id=column_id,
                    column_name=_column_name(objects, target_id, column_id),
                    query_row_id=int(row["query_row_id"]),
                    source_group=str(record["source_group"]),
                    cells=_cells(row),
                    evidence_ids=retained,
                    focus_image_id=valid[0],
                )
                if unit.unit_id in seen:
                    continue
                seen.add(unit.unit_id)
                units.append(unit)
                sidecars.append({
                    "unit_id": unit.unit_id,
                    "gold_values": sorted(evaluation["gold_values"]),
                    "gold_target_row_ids": sorted(evaluation["target_rows"]),
                    "witness_evidence_ids": sorted(witness),
                })
    return units, sidecars


def _unit_key(unit: FastUnit, salt: str) -> str:
    return _sha(
        f"{salt}|{unit.query_id}|{unit.target_id}|{unit.column_id}|"
        f"{unit.query_row_id}|{unit.focus_image_id}"
    )


def unit_shard(unit_id: str, shards: int) -> int:
    return int(hashlib.sha256(unit_id.encode()).hexdigest()[:16], 16) % shards


def _select(
    units: Iterable[FastUnit], *, salt: str, target_units: int, per_group: int, per_query: int
) -> list[FastUnit]:
    grouped: dict[str, list[FastUnit]] = defaultdict(list)
    for unit in units:
        grouped[unit.source_group].append(unit)
    query_counts: dict[str, int] = defaultdict(int)
    used_images: set[str] = set()
    selected: list[FastUnit] = []
    for group in sorted(grouped, key=lambda item: _sha(f"{salt}|{item}")):
        taken = 0
        for unit in sorted(grouped[group], key=lambda item: _unit_key(item, salt)):
            if taken >= per_group or query_counts[unit.query_id] >= per_query:
                continue
            if unit.focus_image_id in used_images:
                continue
            selected.append(unit)
            taken += 1
            query_counts[unit.query_id] += 1
            used_images.add(unit.focus_image_id)
            if len(selected) >= target_units:
                return selected
    return selected


def _json_line(record: dict[str, Any]) -> bytes:
    return (json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n").encode()


def write_deterministic_jsonl_gz(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            for record in records:
                handle.write(_json_line(record))
    temporary.replace(path)


def lock_component(r4: Path, r3: Path, out: Path, target_units: int = 48) -> dict[str, Any]:
    candidates, sidecars = build_component_candidates(r4, r3)
    by_unit = {row["unit_id"]: row for row in sidecars}
    selected = _select(
        candidates, salt=COMPONENT_SALT, target_units=target_units, per_group=2, per_query=2
    )
    write_deterministic_jsonl_gz(
        out / "COMPONENT_CANDIDATES.dev.jsonl.gz", (unit.to_dict() for unit in candidates)
    )
    write_deterministic_jsonl_gz(
        out / "COMPONENT_PILOT_LOCK.dev.jsonl.gz", (unit.to_dict() for unit in selected)
    )
    write_deterministic_jsonl_gz(
        out / "COMPONENT_EVAL_SIDECAR.dev.jsonl.gz", (by_unit[unit.unit_id] for unit in selected)
    )
    return {
        "status": "LOCKED" if len(selected) >= 32 else "BLOCKED_TOO_FEW_IMAGE_WITNESS_UNITS",
        "candidate_units": len(candidates),
        "locked_units": len(selected),
        "source_groups": len({unit.source_group for unit in selected}),
        "queries": len({unit.query_id for unit in selected}),
        "salt": COMPONENT_SALT,
        "target_units": target_units,
    }


def build_natural_candidates(r4: Path) -> list[FastUnit]:
    """Build label-blind candidates. This function deliberately has no R3/GT argument."""
    objects = _read_objects(r4)
    lock = _read_lock(r4)
    schedule = _read_schedule(r4)
    decode_memo: dict[str, bool] = {}
    units: list[FastUnit] = []
    for record in schedule:
        query_id = str(record["query_id"])
        rows = visible_query_rows(objects, query_id)
        for branch in record["branches"]:
            target_id = str(branch["target_id"])
            column_id = int(branch["column_id"])
            target = _target(lock, query_id, target_id)
            retained = tuple(str(item) for item in target["retained_evidence_ids"])
            images = [
                evidence_id
                for evidence_id in retained
                if objects["evidence"][evidence_id].get("asset_type") == "image"
                and _image_decodes(objects["evidence"][evidence_id], decode_memo)
            ]
            if not images:
                continue
            for row in rows:
                units.append(FastUnit.create(
                    query_id=query_id,
                    target_id=target_id,
                    column_id=column_id,
                    column_name=_column_name(objects, target_id, column_id),
                    query_row_id=int(row["query_row_id"]),
                    source_group=str(record["source_group"]),
                    cells=_cells(row),
                    evidence_ids=retained,
                    focus_image_id=images[0],
                ))
    return units


def lock_natural(
    r4: Path, out: Path, target_units: int = 48, target_groups: int = 32
) -> dict[str, Any]:
    candidates = build_natural_candidates(r4)
    grouped: dict[str, list[FastUnit]] = defaultdict(list)
    for unit in candidates:
        grouped[unit.source_group].append(unit)
    selected: list[FastUnit] = []
    used_images: set[str] = set()
    query_counts: dict[str, int] = defaultdict(int)
    chosen_groups: list[str] = []
    for group in sorted(grouped, key=lambda item: _sha(f"{NATURAL_SALT}|{item}")):
        ordered = sorted(grouped[group], key=lambda item: _unit_key(item, NATURAL_SALT))
        available = next(
            (unit for unit in ordered if unit.focus_image_id not in used_images), None
        )
        if available is None:
            continue
        selected.append(available)
        chosen_groups.append(group)
        used_images.add(available.focus_image_id)
        query_counts[available.query_id] += 1
        if len(chosen_groups) == target_groups:
            break
    if len(chosen_groups) == target_groups:
        for group in chosen_groups:
            ordered = sorted(grouped[group], key=lambda item: _unit_key(item, NATURAL_SALT))
            for unit in ordered:
                if (
                    unit.focus_image_id not in used_images
                    and query_counts[unit.query_id] < 2
                ):
                    selected.append(unit)
                    used_images.add(unit.focus_image_id)
                    query_counts[unit.query_id] += 1
                    break
            if len(selected) == target_units:
                break
    write_deterministic_jsonl_gz(
        out / "PILOT_LOCK.dev.jsonl.gz", (unit.to_dict() for unit in selected)
    )
    return {
        "status": "LOCKED" if len(chosen_groups) == target_groups and len(selected) == target_units
        else "BLOCKED_TOO_FEW_LABEL_BLIND_UNITS",
        "candidate_units": len(candidates),
        "locked_units": len(selected),
        "source_groups": len({unit.source_group for unit in selected}),
        "salt": NATURAL_SALT,
        "target_units": target_units,
        "target_groups": target_groups,
        "lock_sha256": file_hash(out / "PILOT_LOCK.dev.jsonl.gz"),
    }


def smoke_units(units: list[FastUnit]) -> list[FastUnit]:
    groups = list(dict.fromkeys(unit.source_group for unit in units))[:8]
    return [unit for unit in units if unit.source_group in groups]
