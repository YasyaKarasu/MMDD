"""Data loading, content alias extraction, and dataset access for CLEAN-QET v4.0."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Generator, Iterable, Sequence

from .config import Paths


def iter_jsonl(path: Path) -> Generator[dict[str, Any], None, None]:
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        with path.open("rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(p)


def write_jsonl_gz(path: Path, rows: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp.{os.getpid()}")
    with gzip.open(tmp, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(p)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def utf8_sorted(items: Iterable[str]) -> list[str]:
    return sorted(items, key=lambda s: s.encode("utf-8"))


def build_content_aliases(paths: Paths) -> dict[str, str]:
    """Identify canonical evidence IDs across bridge assets based on content hash.
    
    Returns mapping: asset_id -> canonical_id.
    """
    out_dir = paths.run_root
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical_map_path = paths.run_root / "CONTENT_ALIASES.json"
    canonical_gz_path = paths.run_root / "CONTENT_ALIASES.jsonl.gz"

    if canonical_map_path.exists() and canonical_gz_path.exists():
        summary = read_json(canonical_map_path)
        if summary.get("status") == "PASS":
            canonical = {}
            for row in iter_jsonl(canonical_gz_path):
                canonical[str(row["asset_id"])] = str(row["canonical_id"])
            return canonical

    asset_files = sorted((paths.dataset_root / "bridge_assets").glob("part-*.jsonl"))
    by_hash: dict[tuple[str, str], list[dict]] = defaultdict(list)
    total_assets = 0

    for part in asset_files:
        for row in iter_jsonl(part):
            asset_id = str(row["asset_id"])
            asset_type = str(row["asset_type"])
            content_hash = str(row.get("content_sha256") or row.get("asset_sha256") or asset_id)
            by_hash[(asset_type, content_hash)].append(
                {"asset_id": asset_id, "asset_type": asset_type, "content_hash": content_hash}
            )
            total_assets += 1

    canonical: dict[str, str] = {}
    rows_out: list[dict] = []
    canonical_objects = 0

    for (modality, chash), group in sorted(by_hash.items(), key=lambda x: (x[0][0], x[0][1])):
        # Canonical ID is the lexicographically smallest asset_id in the equivalence class
        sorted_group = sorted(group, key=lambda g: g["asset_id"].encode("utf-8"))
        canonical_id = sorted_group[0]["asset_id"]
        canonical_objects += 1
        for item in sorted_group:
            canonical[item["asset_id"]] = canonical_id
            rows_out.append(
                {
                    "asset_id": item["asset_id"],
                    "canonical_id": canonical_id,
                    "modality": modality,
                    "content_hash": chash,
                }
            )

    rows_out.sort(key=lambda r: r["asset_id"].encode("utf-8"))
    write_jsonl_gz(canonical_gz_path, rows_out)
    write_json(
        canonical_map_path,
        {
            "status": "PASS",
            "total_assets": total_assets,
            "canonical_objects": canonical_objects,
            "aliases_gz": str(canonical_gz_path),
            "sha256": sha256_file(canonical_gz_path),
        },
    )
    return canonical


def load_split_gt(paths: Paths, split: str, canonical_map: dict[str, str]) -> dict[str, dict]:
    """Load ground-truth qrels, source groups, and witness records for dev or test split."""
    if split not in ("dev", "test"):
        raise ValueError(f"invalid split: {split}")

    gold: dict[str, set[str]] = defaultdict(set)
    reasons: dict[str, set[str]] = defaultdict(set)
    implicit: dict[str, set[str]] = defaultdict(set)

    for row in iter_jsonl(paths.dataset_root / "qrels.jsonl"):
        if str(row.get("split")) != split or int(row.get("rel", 0)) <= 0:
            continue
        qid = str(row["query_table_id"])
        tid = str(row["target_table_id"])
        gold[qid].add(tid)
        reason = str(row.get("reason"))
        reasons[qid].add(reason)
        if reason == "model_recoverable_join_column":
            implicit[qid].add(tid)

    groups: dict[str, str] = {}
    for row in iter_jsonl(paths.dataset_root / "query_tables" / "part-00000.jsonl"):
        if str(row.get("split")) == split:
            groups[str(row["table_id"])] = str(row.get("source_table_id") or row["table_id"])

    witness: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in iter_jsonl(paths.dataset_root / "evidence_recoveries" / "part-00000.jsonl"):
        if str(row.get("split")) != split:
            continue
        qid = str(row["query_table_id"])
        tid = str(row["target_table_id"])
        evidence = row.get("evidence") or {}
        raw_asset = str(evidence.get("asset_id"))
        if raw_asset not in canonical_map:
            continue
        witness[qid][tid].add(canonical_map[raw_asset])

    out: dict[str, dict] = {}
    for qid in utf8_sorted(gold):
        kinds = reasons[qid]
        if kinds == {"model_recoverable_join_column"}:
            kind = "implicit"
        elif kinds == {"explicit_visible_join_column"}:
            kind = "explicit"
        else:
            kind = "mixed"
        out[qid] = {
            "G": utf8_sorted(gold[qid]),
            "implicit_G": utf8_sorted(implicit.get(qid, set())),
            "kind": kind,
            "source_group": groups.get(qid, qid),
            "W": {t: utf8_sorted(a) for t, a in sorted(witness.get(qid, {}).items())},
        }
    return out


def split_query_ids(paths: Paths, split: str) -> list[str]:
    return utf8_sorted(
        str(row["table_id"])
        for row in iter_jsonl(paths.dataset_root / "query_tables" / "part-00000.jsonl")
        if str(row.get("split")) == split
    )
