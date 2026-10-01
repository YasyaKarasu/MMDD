"""Data loading, content alias extraction, and dataset access for Stage-1 CQET."""
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


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp.{os.getpid()}")
    with tmp.open("wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
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
    """Load the immutable, pixel/text-content-derived preflight alias lock."""
    report_path = paths.run_root / "CONTENT_ALIAS_REPORT.json"
    archive_path = paths.run_root / "CONTENT_ALIASES.jsonl.gz"
    report = read_json(report_path)
    if report.get("status") != "PASS":
        raise RuntimeError("content alias preflight has not passed")
    declared = report.get("file", {})
    actual_sha = sha256_file(archive_path)
    if declared.get("sha256") != actual_sha:
        raise RuntimeError("content alias archive hash differs from the preflight lock")

    canonical: dict[str, str] = {}
    for row in iter_jsonl(archive_path):
        asset_id = str(row["asset_id"])
        canonical_id = str(row["canonical_evidence_id"])
        visible_hash = str(row.get("visible_content_sha256", ""))
        if not visible_hash or visible_hash == asset_id:
            raise ValueError(f"{asset_id}: missing or ID-derived content hash")
        canonical[asset_id] = canonical_id
    if len(canonical) != int(report["objects"]):
        raise RuntimeError("content alias archive row count changed")
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
            source_group = row.get("source_table_id")
            if not source_group:
                raise ValueError(f"{row['table_id']}: source_table_id is required for evaluation")
            groups[str(row["table_id"])] = str(source_group)

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
            "source_group": groups[qid],
            "W": {t: utf8_sorted(a) for t, a in sorted(witness.get(qid, {}).items())},
        }
    return out


def split_query_ids(paths: Paths, split: str) -> list[str]:
    return utf8_sorted(
        str(row["table_id"])
        for row in iter_jsonl(paths.dataset_root / "query_tables" / "part-00000.jsonl")
        if str(row.get("split")) == split
    )
