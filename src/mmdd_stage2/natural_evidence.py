"""Natural retrieved evidence as the reader condition for the column selector.

The production selector is trained with witness evidence (``O-O``/``O-R`` keys of the frozen
column inputs) and the no-evidence control is the empty list. This module adds a third, honest
source: the *natural* evidence the current Stage-1 retrieval actually retained for a
(query, target) pair, taken in retrieval order and capped at the reader evidence budget. Nothing
is oracle-derived, nothing is synthesised, and an empty retrieval stays empty.

It writes augmented column inputs (``COLUMN_INPUTS.<split>.jsonl`` carrying an extra
``NAT_E`` evidence key) so the existing job keys, feature cache and training loop apply unchanged.
The source-group holdout split is fixed by a hash of the source table, so the fit and holdout
partitions never share a source table and are reproducible across arms.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .column_data import digest, file_hash, read_jsonl, write_json, write_jsonl

NATURAL_EVIDENCE_KEY = "NAT_E"
HOLDOUT_SALT = "R7_SOURCE_GROUP_HOLDOUT_V1"
HOLDOUT_MODULUS = 10
HOLDOUT_RESIDUE = 0
MAX_EVIDENCE = 4


def retained_paths(record: Mapping[str, Any]) -> Mapping[str, Sequence[str]]:
    """Ordered retained evidence per target, accepting both Stage-1 export shapes."""
    pool = record.get("pool")
    if isinstance(pool, Mapping) and "retained_paths" in pool:
        return pool["retained_paths"]
    paths = record.get("retained_paths")
    if isinstance(paths, Mapping):
        return paths
    raise ValueError("stage1 record carries no retained_paths")


def natural_evidence_ids(record: Mapping[str, Any], target_id: str, *, max_evidence: int = MAX_EVIDENCE) -> list[str]:
    """Evidence ids the current Stage-1 retrieval retained for one target, in retrieval order."""
    if max_evidence < 1:
        raise ValueError("max_evidence must be positive")
    ids = list(retained_paths(record).get(target_id, ()))
    if len(ids) > max_evidence:
        return ids[:max_evidence]
    return ids


def holdout_partition(source_group: str, *, salt: str = HOLDOUT_SALT, modulus: int = HOLDOUT_MODULUS,
                      residue: int = HOLDOUT_RESIDUE) -> str:
    """Fixed fit/holdout partition of a source table; identical for every arm."""
    if modulus < 1 or not 0 <= residue < modulus:
        raise ValueError("invalid holdout modulus/residue")
    bucket = int(hashlib.sha256(f"{salt}|{source_group}".encode()).hexdigest(), 16)
    return "holdout" if bucket % modulus == residue else "fit"


def stage1_directory_loader(folder: Path) -> Callable[[str, str], Mapping[str, Any]]:
    """Lazy ``(split, query_id) -> record`` reader over a Stage-1 export directory.

    A missing record raises instead of quietly yielding empty evidence, because a wrong Stage-1
    path would otherwise produce a silently evidence-free run.
    """
    root = Path(folder)

    def load(split: str, query_id: str) -> Mapping[str, Any]:
        path = root / split / f"{query_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Stage-1 record missing for {split}/{query_id}: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    return load


def augment_inputs(items: Sequence[dict[str, Any]], stage1: Mapping[str, Mapping[str, Any]],
                   *, max_evidence: int = MAX_EVIDENCE, key: str = NATURAL_EVIDENCE_KEY,
                   groups: Mapping[str, str] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach natural evidence (and the source-group partition) to frozen column inputs.

    Every input keeps its existing evidence keys untouched, so witness-based arms still resolve.
    """
    funnel: dict[str, int] = {"pairs": 0, "pairs_with_natural_evidence": 0, "evidence_items": 0,
                              "fit_pairs": 0, "holdout_pairs": 0, "missing_stage1_record": 0, "source_group_fallback": 0}
    fit_groups: set[str] = set()
    holdout_groups: set[str] = set()
    augmented = []
    for item in items:
        record = stage1.get(item["query_id"])
        if record is None:
            funnel["missing_stage1_record"] += 1
            ids: list[str] = []
        else:
            ids = natural_evidence_ids(record, item["target_id"], max_evidence=max_evidence)
        group = (groups or {}).get(item["query_id"]) or item.get("source_group")
        if group is None:
            group = item["query_id"]
            funnel["source_group_fallback"] += 1
        partition = holdout_partition(str(group))
        (holdout_groups if partition == "holdout" else fit_groups).add(str(group))
        funnel["pairs"] += 1
        funnel["pairs_with_natural_evidence"] += int(bool(ids))
        funnel["evidence_items"] += len(ids)
        funnel[f"{partition}_pairs"] += 1
        augmented.append({**item, "evidence_ids": {**item.get("evidence_ids", {}), key: ids},
                          "natural_partition": partition, "source_group": str(group)})
    funnel["fit_groups"] = len(fit_groups)
    funnel["holdout_groups"] = len(holdout_groups)
    funnel["group_overlap"] = len(fit_groups & holdout_groups)
    return augmented, funnel


def build_inputs(r1: Path, output: Path, stage1: Mapping[str, Mapping[str, Any]] | Callable[[str, str], Mapping[str, Any]],
                 splits: Sequence[str] = ("train", "dev", "test"), *, max_evidence: int = MAX_EVIDENCE,
                 groups: Mapping[str, str] | None = None, key: str = NATURAL_EVIDENCE_KEY) -> dict[str, Any]:
    """Write ``NAT_E/COLUMN_INPUTS.<split>.jsonl`` next to the frozen column inputs.

    ``stage1`` is either a ``query_id -> record`` mapping or a ``(split, query_id) -> record``
    callable, so exports read from disk can be passed lazily.
    """
    folder = output / "NAT_E"
    folder.mkdir(parents=True, exist_ok=True)
    lookup = stage1 if callable(stage1) else (lambda split, query_id: stage1.get(query_id))
    manifest: dict[str, Any] = {"natural_evidence_key": key, "max_evidence": max_evidence,
                                "holdout_salt": HOLDOUT_SALT, "holdout_modulus": HOLDOUT_MODULUS,
                                "holdout_residue": HOLDOUT_RESIDUE, "splits": {}}
    all_ids: set[str] = set()
    for split in splits:
        items = read_jsonl(r1 / f"COLUMN_INPUTS.{split}.jsonl")
        augmented, funnel = augment_inputs(
            items, {item["query_id"]: lookup(split, item["query_id"]) for item in items},
            max_evidence=max_evidence, key=key, groups=groups)
        if funnel["group_overlap"]:
            raise ValueError("holdout partition leaks source groups")
        path = folder / f"COLUMN_INPUTS.{split}.jsonl"
        write_jsonl(path, augmented)
        all_ids.update(eid for item in augmented for eid in item["evidence_ids"][key])
        manifest["splits"][split] = {**funnel, "path": str(path), "sha256": file_hash(path), "rows": len(augmented)}
        print(f"NAT_E {split}: {len(augmented)} pairs, {funnel['pairs_with_natural_evidence']} with evidence, "
              f"fit/holdout groups {funnel['fit_groups']}/{funnel['holdout_groups']}", flush=True)
    # Keep the natural arm self-contained: it may use only assets already present in
    # the audited Stage-1 object store and never silently substitute oracle objects.
    objects_path = r1 / "OBJECTS.jsonl.gz"
    objects = read_jsonl(objects_path)[0] if objects_path.is_file() else {"evidence": {}}
    missing = sorted(all_ids - set(objects.get("evidence", {})))
    # Tiny fixtures and a staged Stage-1 export may carry paths before its object
    # sidecar is materialized.  Preserve that fact in the receipt; feature caching
    # will fail loudly if a non-empty run actually needs the missing assets.
    evidence_objects = [objects["evidence"][eid] for eid in sorted(all_ids) if eid in objects.get("evidence", {})]
    object_path = folder / "EVIDENCE_OBJECTS.jsonl.gz"
    write_jsonl(object_path, evidence_objects)
    manifest["objects_sha256"] = file_hash(object_path)
    manifest["evidence_object_count"] = len(evidence_objects)
    manifest["missing_evidence_objects"] = missing
    manifest["funnel_sha256"] = digest(manifest["splits"])
    write_json(folder / "MANIFEST.json", manifest)
    return manifest
