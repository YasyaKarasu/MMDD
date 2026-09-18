#!/usr/bin/env python
"""Experiment 1: offline witness annotation and pre-retention provenance tracing.

Zero training, zero retrieval, zero ranking change.  This script only *reads*
the frozen artifacts produced by the FINAL_RERANK round and attaches an offline
witness label plus a per-stage path-membership trail to them.  It never rewrites
a score, a candidate pool, a retained bag, or a ranking.

Witness label source
--------------------
`<dataset>/evidence_recoveries/part-*.jsonl` -- the dataset builder's own
triadic recovery records.  Each record is a `query_table -> asset -> target_table`
path whose `auto_check` reviews all carry `verdict == "supported"` under the
fail-closed policy `keep_source_canonical_supported_only_fail_closed`.  That is
a *verified* statement that the evidence asset supports recovering the hidden
join attribute that links the query entity to the target table row.

Important scope limits, recorded in every emitted label:

* The recovery set is a strict subset of the qrels set (verified: 0 recoveries
  fall outside qrels, 601 of 1279 dev qrels have no recovery).  A qrel without a
  recovery is therefore `unknown`, never `verified_negative`.
* Consequently this file proves a *lower bound* on witness availability.  It can
  never prove "no correct witness exists" and is never used to claim that.

Retention funnel replayed here (imported from production so semantics cannot drift)
----------------------------------------------------------------------------------
  raw bag            row["E_pre_retention"][t]["paths"], kind == "evidence"
  stage1 evidence    run_stage1_r11_task_e._evidence_paths      (filter + sort)
  stage2 dedup       run_stage1_r11_task_e._deduplicate         (content_key)
  stage3 top20       stage2[:20]
  stage4 retained    row["E_paths"][t]["retained_paths"]        (greedy budget 4)
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmdd_stage1.row_support import load_evidence_content_keys  # noqa: E402
from mmdd_stage1.r26_metrics import query_metrics  # noqa: E402
from run_stage1_r11_task_e import _deduplicate, _evidence_paths  # noqa: E402

DATASET = ROOT / "output_mm_joinability_entitables_20000_retry100_rounds5_qwen35_final_survivor_context_gaussian_v9"
RERANK = ROOT / "work/final_rerank_20260916/FINAL_RERANK"
OUT = ROOT / "work/witness_diagnostic_20260916"
CONTENT_KEYS = ROOT / "work/stage1_optimization_r10_20260907/taskB_g5/evidence_content_keys.jsonl"
WITNESS_ANNOTATION_SCOPE = "dev"
WITNESS_ANNOTATION_SOURCE = "dataset_builder_evidence_recoveries_auto_check_v6"
WITNESS_COMPLETENESS = "partial"
KS = (10, 20, 50)

# Ordered stage names; index is the furthest stage a witness survives to.
STAGES = ("absent", "pre_retention", "dedup", "top20", "retained")
STAGE_INDEX = {name: index for index, name in enumerate(STAGES)}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def rows(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_rows(path: Path, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    digest = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as handle:
        for record in records:
            line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            handle.write(line)
            digest.update(line.encode())
            count += 1
    return {"path": str(path), "records": count, "sha256": digest.hexdigest(),
            "bytes": path.stat().st_size}


def write_csv(path: Path, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return {"path": str(path), "rows": 0, "sha256": sha256_file(path)}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return {"path": str(path), "rows": len(records), "sha256": sha256_file(path)}


def write_json(path: Path, payload: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return {"path": str(path), "sha256": hashlib.sha256(text.encode()).hexdigest()}


# --------------------------------------------------------------------------- #
# witness labels
# --------------------------------------------------------------------------- #


def load_witness_labels() -> tuple[dict[tuple[str, str], dict[str, dict]], dict[str, Any]]:
    """(query_id, target_id) -> {asset_id: label record}. Dev split only."""

    labels: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    stats: Counter = Counter()
    paths = sorted(DATASET.glob("evidence_recoveries/part-*.jsonl"))
    if not paths:
        raise SystemExit(f"no evidence_recoveries under {DATASET}")
    for path in paths:
        for record in rows(path):
            stats[f"split:{record.get('split')}"] += 1
            if record.get("split") != WITNESS_ANNOTATION_SCOPE:
                continue
            evidence = record.get("evidence") or {}
            asset_id = str(evidence.get("asset_id") or "")
            if not asset_id:
                stats["skipped_missing_asset"] += 1
                continue
            review = record.get("auto_check") or {}
            reviews = review.get("reviews") or []
            policy = str(review.get("policy") or "")
            supported = bool(reviews) and all(
                str(item.get("verdict")) == "supported" for item in reviews
            )
            if not supported or policy != "keep_source_canonical_supported_only_fail_closed":
                stats["skipped_not_fail_closed_supported"] += 1
                continue
            key = (str(record["query_table_id"]), str(record["target_table_id"]))
            entry = labels[key].setdefault(
                asset_id,
                {
                    "query_id": key[0],
                    "target_id": key[1],
                    "evidence_id": asset_id,
                    "modality": str(evidence.get("asset_type") or ""),
                    "witness_label": "verified_positive",
                    "annotation_scope": WITNESS_ANNOTATION_SCOPE,
                    "annotation_source": WITNESS_ANNOTATION_SOURCE,
                    "completeness": WITNESS_COMPLETENESS,
                    "recovery_ids": [],
                    "query_row_ids": [],
                    "target_row_ids": [],
                    "join_column": None,
                    "join_value": None,
                    "chain_id": record.get("chain_id"),
                },
            )
            entry["recovery_ids"].append(str(record.get("recovery_id")))
            row_id = record.get("query_row_id")
            if row_id is not None and row_id not in entry["query_row_ids"]:
                entry["query_row_ids"].append(row_id)
            for target_row in record.get("target_row_ids") or []:
                if target_row not in entry["target_row_ids"]:
                    entry["target_row_ids"].append(target_row)
            attribute = record.get("recovered_attribute") or {}
            if attribute.get("column_name"):
                entry["join_column"] = str(attribute["column_name"])
                entry["join_value"] = attribute.get("value")
            stats["verified_positive"] += 1
    for entry in labels.values():
        for value in entry.values():
            value["recovery_ids"].sort()
            value["query_row_ids"].sort()
            value["target_row_ids"] = sorted(
                value["target_row_ids"], key=lambda item: str(item)
            )
    stats["pairs"] = len(labels)
    stats["assets"] = sum(len(value) for value in labels.values())
    return labels, dict(stats)


# --------------------------------------------------------------------------- #
# endpoints
# --------------------------------------------------------------------------- #


def load_endpoints() -> list[dict[str, Any]]:
    lock = json.loads((RERANK / "INPUT_LOCK.json").read_text())
    result = []
    for entry in lock["endpoints"]:
        own = Path(entry["own_rankings"]["path"])
        if not own.is_file():
            raise SystemExit(f"missing own rankings: {own}")
        observed = sha256_file(own)
        if observed != entry["own_rankings"]["sha256"]:
            raise SystemExit(
                f"{entry['endpoint']}: own rankings sha256 mismatch "
                f"{observed} != {entry['own_rankings']['sha256']}"
            )
        result.append(
            {
                "endpoint": entry["endpoint"],
                "family": entry["family"],
                "seed": entry["seed"],
                "own": own,
                "own_sha256": observed,
            }
        )
    return result


def load_cohorts() -> dict[tuple[str, str, str], list[str]]:
    """(endpoint, query_id, target_id) -> cohort names from the fixed-207 funnel."""

    path = RERANK / "strict_eo/fixed207_funnel.csv"
    result: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            result[(record["endpoint"], record["query_id"], record["target_id"])].append(
                record["cohort"]
            )
    return result


def load_saved_rankings(name: str) -> dict[tuple[str, str], dict[str, Any]]:
    """(endpoint, query_id) -> saved ranking row for a view artifact."""

    result: dict[tuple[str, str], dict[str, Any]] = {}
    for record in rows(RERANK / f"rankings/{name}.jsonl.gz"):
        result[(record["endpoint"], record["query_id"])] = record
    return result


# --------------------------------------------------------------------------- #
# funnel tracing
# --------------------------------------------------------------------------- #


def content_key(content_keys: dict[str, str], evidence_id: str) -> str | None:
    return content_keys.get(evidence_id)


def stage_of_witness(
    witness_id: str,
    stages: dict[str, Sequence[dict[str, Any]]],
    keys: dict[str, str],
) -> dict[str, Any]:
    """Where an evidence asset (or its content-equivalent) survives to.

    `exact` tracks the literal evidence id.  `content` tracks the dedup content
    key, because content dedup can drop an id while keeping an identical-content
    sibling alive -- the witness content is then still present under another id.
    """

    key = keys.get(witness_id)
    trail: dict[str, Any] = {"content_key": key}
    exact_stage = "absent"
    content_stage = "absent"
    for name in STAGES[1:]:
        ids = [str(path["evidence_id"]) for path in stages[name]]
        if witness_id in ids:
            exact_stage = name
        if key is not None:
            for path in stages[name]:
                if keys.get(str(path["evidence_id"])) == key:
                    content_stage = name
                    break
        if "rank" not in trail and witness_id in ids:
            trail["rank"] = ids.index(witness_id)
    trail["exact_stage"] = exact_stage
    trail["content_stage"] = content_stage
    trail["furthest"] = max(
        (exact_stage, content_stage), key=lambda value: STAGE_INDEX[value]
    )
    trail["exact_rank"] = None
    for name in STAGES[1:]:
        ids = [str(path["evidence_id"]) for path in stages[name]]
        if witness_id in ids:
            trail["exact_rank"] = ids.index(witness_id)
            break
    return trail


def build_stages(target: dict[str, Any], keys: dict[str, str]) -> dict[str, Sequence[dict[str, Any]]]:
    raw = [path for path in target.get("paths", []) if path.get("kind") == "evidence"]
    evidence = _evidence_paths(target.get("paths", []))
    dedup = _deduplicate(evidence, keys)
    return {
        "pre_retention": evidence,
        "dedup": dedup,
        "top20": dedup[:20],
        "retained": list(target.get("retained_paths", [])),
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--endpoints", nargs="*", default=None)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    started = utcnow()
    print(json.dumps({"event": "load_labels"}), flush=True)
    labels, label_stats = load_witness_labels()
    print(json.dumps({"event": "labels", **label_stats}), flush=True)

    content_keys, content_keys_sha = load_evidence_content_keys(CONTENT_KEYS)
    print(json.dumps({"event": "content_keys", "objects": len(content_keys)}), flush=True)

    endpoints = load_endpoints()
    if args.endpoints:
        wanted = set(args.endpoints)
        endpoints = [entry for entry in endpoints if entry["endpoint"] in wanted]
    cohorts = load_cohorts()

    qt_saved = load_saved_rankings("QT")
    student_saved = load_saved_rankings("Student_Path")
    teacher_saved = load_saved_rankings("Teacher_Path")

    c100_by_query: dict[tuple[str, str], dict[str, Any]] = {}
    for record in rows(RERANK / "candidates/per_query_C100.jsonl.gz"):
        c100_by_query[(record["endpoint"], record["query_id"])] = record
    path_membership: dict[tuple[str, str], dict[str, Any]] = {}
    for record in rows(RERANK / "candidates/path_membership.jsonl.gz"):
        path_membership[(record["endpoint"], record["query_id"])] = record

    retrieval_scores = sqlite3.connect(f"file:{RERANK / 'scores/teacher_pair_scores.sqlite'}?mode=ro", uri=True)
    cached_pairs = {
        (str(source), str(destination)): float(score)
        for source, destination, score in retrieval_scores.execute(
            "SELECT source_id, destination_id, score FROM scores"
        )
    }
    retrieval_scores.close()
    print(json.dumps({"event": "teacher_cache", "pairs": len(cached_pairs)}), flush=True)

    label_records: list[dict[str, Any]] = []
    diagnostic_records: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    pair_hashes: dict[str, Any] = {}
    population: dict[str, Any] = {}
    missing_teacher_pairs: set[tuple[str, str]] = set()

    for entry in endpoints:
        endpoint = entry["endpoint"]
        family = entry["family"]
        seed = entry["seed"]
        print(json.dumps({"event": "endpoint_start", "endpoint": endpoint}), flush=True)

        # Recompute the official metrics from the saved view rankings.
        per_view_rankings: dict[str, dict[str, Sequence[str]]] = defaultdict(dict)
        per_view_qrels: dict[str, dict[str, Sequence[str]]] = {}
        for (ep, query_id), record in qt_saved.items():
            if ep != endpoint:
                continue
            per_view_rankings["QT-only"][query_id] = record["ranking"]
            per_view_qrels[query_id] = record["positive_target_ids"]
        for (ep, query_id), record in student_saved.items():
            if ep != endpoint:
                continue
            for view in ("Student-D1-Path", "Student-LSE-Path"):
                per_view_rankings[view][query_id] = record["rankings"][view]
        for (ep, query_id), record in teacher_saved.items():
            if ep != endpoint:
                continue
            per_view_rankings["Teacher-LSE-Path"][query_id] = record["ranking"]

        for view, rankings in per_view_rankings.items():
            for kind in ("overall", "implicit", "explicit"):
                subset = {
                    q: truth
                    for q, truth in per_view_qrels.items()
                    if kind == "overall"
                    or qt_saved[(endpoint, q)]["query_kind"] == kind
                }
                if not subset:
                    continue
                metrics = {
                    key: sum(
                        query_metrics(rankings.get(q, []), truth, KS)[key]
                        for q, truth in subset.items()
                    )
                    / len(subset)
                    for key in ("recall@10", "recall@20", "recall@50")
                }
                metric_rows.append(
                    {
                        "endpoint": endpoint,
                        "family": family,
                        "seed": seed,
                        "view": view,
                        "kind": kind,
                        "queries": len(subset),
                        "recall@10": round(100 * metrics["recall@10"], 6),
                        "recall@20": round(100 * metrics["recall@20"], 6),
                        "recall@50": round(100 * metrics["recall@50"], 6),
                    }
                )

        query_count = 0
        funnel_cells: Counter = Counter()
        target_cells: Counter = Counter()
        for record in rows(entry["own"]):
            query_id = str(record["query_id"])
            if args.limit_queries is not None and query_count >= args.limit_queries:
                break
            query_count += 1
            key = (endpoint, query_id)
            c100_record = c100_by_query[key]
            c100 = [str(value) for value in c100_record["c100_ids"]]
            if c100 != [str(value) for value in record["rankings"]["Equal"][:100]]:
                raise SystemExit(f"{endpoint}/{query_id}: C100 differs from own Equal[:100]")
            positive = {str(value) for value in record["positive_target_ids"]}
            target_source = {str(k): str(v) for k, v in c100_record["target_source"].items()}
            query_kind = str(record["query_kind"])

            qt_rank = {
                str(t): index
                for index, t in enumerate(qt_saved[key]["ranking"])
            }
            student_rank = {
                view: {str(t): index for index, t in enumerate(student_saved[key]["rankings"][view])}
                for view in ("Student-D1-Path", "Student-LSE-Path")
            }
            teacher_rank = {
                str(t): index for index, t in enumerate(teacher_saved[key]["ranking"])
            }

            target_by_id = {str(row["target_id"]): row for row in record["E_paths"]}
            pre_by_id = {str(row["target_id"]): row for row in record["E_pre_retention"]}

            # Stage membership hashes.  The published multiset hashes come from
            # the frozen round; the replay hash is recomputed here so the two can
            # be compared.  Only positive targets get a per-stage trail hash.
            membership = path_membership[key]
            replay = {
                t: sorted(
                    str(p.get("evidence_id"))
                    for p in row.get("retained_paths", [])
                    if p.get("kind") == "evidence"
                )
                for t, row in sorted(target_by_id.items())
            }
            pair_hashes[f"{endpoint}/{query_id}"] = {
                "published_path_multiset_sha256": c100_record.get("path_multiset_sha256"),
                "published_retained_path_multiset_sha256": membership.get(
                    "retained_path_multiset_sha256"
                ),
                "replayed_retained_evidence_multiset_sha256": stable_sha(replay),
            }

            for target_id in c100:
                is_positive = target_id in positive
                target = target_by_id.get(target_id, {})
                pre = pre_by_id.get(target_id, {})
                if pre and target:
                    pre_ids = [str(p["evidence_id"]) for p in pre.get("paths", [])
                               if p.get("kind") == "evidence"]
                    ret_ids = [str(p["evidence_id"]) for p in target.get("retained_paths", [])]
                    if pre_ids != [str(p["evidence_id"]) for p in target.get("paths", [])
                                   if p.get("kind") == "evidence"]:
                        raise SystemExit(
                            f"{endpoint}/{query_id}/{target_id}: E_paths.paths != E_pre_retention.paths"
                        )
                witness = labels.get((query_id, target_id), {})
                # build_stages is only needed when a witness label must be traced.
                stages = (
                    build_stages(target, content_keys)
                    if (witness and target)
                    else {"pre_retention": [], "dedup": [], "top20": [], "retained": []}
                )

                if is_positive:
                    target_cells["c100_positive"] += 1
                    if witness:
                        target_cells["c100_positive_with_witness_label"] += 1
                    else:
                        target_cells["c100_positive_unknown"] += 1
                    if not target.get("retained_paths"):
                        target_cells["c100_positive_no_retained_path"] += 1
                cohort = list(cohorts.get((endpoint, query_id, target_id), []))
                ranks = {
                    "QT-only": qt_rank.get(target_id),
                    "Student-D1-Path": student_rank["Student-D1-Path"].get(target_id),
                    "Student-LSE-Path": student_rank["Student-LSE-Path"].get(target_id),
                    "Teacher-LSE-Path": teacher_rank.get(target_id),
                }

                # ---- witness-labelled evidence rows -------------------------
                for evidence_id, label in sorted(witness.items()):
                    trail = stage_of_witness(evidence_id, stages, content_keys)
                    path = next(
                        (p for p in stages["pre_retention"]
                         if str(p["evidence_id"]) == evidence_id),
                        None,
                    )
                    if path is None:
                        path = next(
                            (p for p in (target.get("paths") or [])
                             if p.get("kind") == "evidence"
                             and str(p.get("evidence_id")) == evidence_id),
                            None,
                        )
                    query_evidence = path.get("query_evidence_score") if path else None
                    evidence_target = path.get("evidence_target_score") if path else None
                    teacher_qe = cached_pairs.get((query_id, evidence_id))
                    teacher_et = cached_pairs.get((evidence_id, target_id))
                    if teacher_qe is None:
                        missing_teacher_pairs.add((query_id, evidence_id))
                    if teacher_et is None:
                        missing_teacher_pairs.add((evidence_id, target_id))
                    label_records.append(dict(label))
                    diagnostic_records.append(
                        {
                            "endpoint": endpoint,
                            "family": family,
                            "seed": seed,
                            "query_id": query_id,
                            "query_kind": query_kind,
                            "source_table_id": str(record["source_table_id"]),
                            "target_id": target_id,
                            "target_is_positive": is_positive,
                            "target_source": target_source.get(target_id),
                            "cohorts": cohort,
                            "ranks": ranks,
                            "evidence_id": evidence_id,
                            "modality": label["modality"],
                            "witness_label": "verified_positive",
                            "exact_stage": trail["exact_stage"],
                            "content_stage": trail["content_stage"],
                            "furthest_stage": trail["furthest"],
                            "exact_rank_in_stage": trail["exact_rank"],
                            "content_key": trail["content_key"],
                            "student_query_evidence_score": query_evidence,
                            "student_evidence_target_score": evidence_target,
                            "student_path_score": (
                                None if query_evidence is None or evidence_target is None
                                else query_evidence + evidence_target
                            ),
                            "teacher_query_evidence_score": teacher_qe,
                            "teacher_evidence_target_score": teacher_et,
                            "teacher_path_score": (
                                None if teacher_qe is None or teacher_et is None
                                else teacher_qe + teacher_et
                            ),
                            "join_column": label["join_column"],
                            "join_value": label["join_value"],
                            "recovery_ids": label["recovery_ids"],
                            "retained_path_count": len(target.get("retained_paths", [])),
                        }
                    )
                    funnel_cells[f"pos={is_positive}|stage={trail['furthest']}"] += 1

                # ---- positive target with no verified witness label --------
                # Emitted explicitly so the diagnostic file is complete over the
                # positive population and unknown coverage is computable without
                # the population side table.  This is `unknown`, never negative.
                if is_positive and not witness:
                    diagnostic_records.append(
                        {
                            "endpoint": endpoint,
                            "family": family,
                            "seed": seed,
                            "query_id": query_id,
                            "query_kind": query_kind,
                            "source_table_id": str(record["source_table_id"]),
                            "target_id": target_id,
                            "target_is_positive": True,
                            "target_source": target_source.get(target_id),
                            "cohorts": cohort,
                            "ranks": ranks,
                            "evidence_id": None,
                            "modality": None,
                            "witness_label": "unknown",
                            "exact_stage": None,
                            "content_stage": None,
                            "furthest_stage": None,
                            "exact_rank_in_stage": None,
                            "content_key": None,
                            "student_query_evidence_score": None,
                            "student_evidence_target_score": None,
                            "student_path_score": None,
                            "teacher_query_evidence_score": None,
                            "teacher_evidence_target_score": None,
                            "teacher_path_score": None,
                            "join_column": None,
                            "join_value": None,
                            "recovery_ids": [],
                            "retained_path_count": len(target.get("retained_paths", [])),
                        }
                    )

                # ---- competitor evidence check ------------------------------
                if not is_positive and target_id in set(qt_saved[key]["ranking"][:10]):
                    for path in target.get("retained_paths", []):
                        evidence_id = str(path["evidence_id"])
                        label_records.append(
                            {
                                "query_id": query_id,
                                "target_id": target_id,
                                "evidence_id": evidence_id,
                                "modality": str(path.get("evidence_type") or ""),
                                "witness_label": "unknown",
                                "annotation_scope": WITNESS_ANNOTATION_SCOPE,
                                "annotation_source": WITNESS_ANNOTATION_SOURCE,
                                "completeness": WITNESS_COMPLETENESS,
                            }
                        )
                        diagnostic_records.append(
                            {
                                "endpoint": endpoint,
                                "family": family,
                                "seed": seed,
                                "query_id": query_id,
                                "query_kind": query_kind,
                                "source_table_id": str(record["source_table_id"]),
                                "target_id": target_id,
                                "target_is_positive": False,
                                "target_source": target_source.get(target_id),
                                "cohorts": cohort,
                                "ranks": ranks,
                                "evidence_id": evidence_id,
                                "modality": str(path.get("evidence_type") or ""),
                                "witness_label": "unknown",
                                "exact_stage": "retained",
                                "content_stage": "retained",
                                "furthest_stage": "retained",
                                "exact_rank_in_stage": None,
                                "content_key": content_keys.get(evidence_id),
                                "student_query_evidence_score": path.get("query_evidence_score"),
                                "student_evidence_target_score": path.get("evidence_target_score"),
                                "student_path_score": path.get("path_score"),
                                "teacher_query_evidence_score": cached_pairs.get((query_id, evidence_id)),
                                "teacher_evidence_target_score": cached_pairs.get((evidence_id, target_id)),
                                "teacher_path_score": None,
                                "join_column": None,
                                "join_value": None,
                                "recovery_ids": [],
                            }
                        )
                        target_cells["competitor_retained_path_unknown"] += 1

        print(json.dumps({
            "event": "endpoint_done", "endpoint": endpoint,
            "queries": query_count,
            "funnel": dict(funnel_cells),
            "targets": dict(target_cells),
        }), flush=True)
        population[endpoint] = {
            "queries": query_count,
            "c100_targets": sum(target_cells[k] for k in ("c100_positive", "competitor_retained_path_unknown")),
            "c100_positive": target_cells["c100_positive"],
            "c100_positive_with_witness_label": target_cells["c100_positive_with_witness_label"],
            "c100_positive_unknown": target_cells["c100_positive_unknown"],
            "c100_positive_no_retained_path": target_cells["c100_positive_no_retained_path"],
            "competitor_retained_path_rows": target_cells["competitor_retained_path_unknown"],
        }
        write_json(OUT / "POPULATION.json", population)

    # ---------------------------- stage summary ---------------------------- #
    stage_summary: Counter = Counter()
    for record in diagnostic_records:
        if record["witness_label"] != "verified_positive":
            continue
        stage_summary[f"stage={record['furthest_stage']}|positive={record['target_is_positive']}"] += 1

    artifacts = {}
    artifacts["labels"] = write_rows(OUT / "WITNESS_LABELS.jsonl.gz", label_records)
    artifacts["diagnostic"] = write_rows(OUT / "WITNESS_DIAGNOSTIC.jsonl.gz", diagnostic_records)
    artifacts["metrics"] = write_csv(OUT / "official_metrics_recomputed.csv", metric_rows)
    artifacts["stage_summary"] = write_json(OUT / "stage_summary.json", dict(stage_summary))
    artifacts["content_keys"] = {"path": str(CONTENT_KEYS), "sha256": content_keys_sha}
    artifacts["stage_hashes"] = write_json(OUT / "PATH_STAGE_HASHES.json", pair_hashes)
    artifacts["run"] = write_json(
        OUT / "RUN.json",
        {
            "started_utc": started,
            "finished_utc": utcnow(),
            "label_stats": label_stats,
            "endpoints": [entry["endpoint"] for entry in endpoints],
            "own_sha256": {entry["endpoint"]: entry["own_sha256"] for entry in endpoints},
            "teacher_cache_pairs": len(cached_pairs),
            "missing_teacher_pairs": len(missing_teacher_pairs),
            "limits": {"training": 0, "retrieval": 0, "ranking_change": 0},
        },
    )
    print(json.dumps({"event": "artifacts", **{k: v for k, v in artifacts.items()
                                               if k in ("labels", "diagnostic", "metrics")}},
                     default=str), flush=True)
    with (OUT / "ARTIFACTS.json").open("w", encoding="utf-8") as handle:
        json.dump(artifacts, handle, indent=2, sort_keys=True, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
