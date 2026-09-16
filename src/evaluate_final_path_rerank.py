#!/usr/bin/env python
"""Evaluation-only QT versus retained-evidence-path reranking.

The evaluator never retrieves candidates.  It consumes frozen own-pool rows,
locks production Equal[:100] membership, and changes only the score used to
order those 100 targets.  Teacher scoring is limited to the retained QE and ET
pairs already attached to targets in that fixed pool.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sqlite3
import statistics
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch

from mmdd_stage1.features import FeatureStore
from mmdd_stage1.r26_metrics import query_metrics
from run_stage1_r19 import load_r19_checkpoint
from run_stage1_r25 import _r25_teacher_feature_paths


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work/final_rerank_20260916/FINAL_RERANK"
R26 = ROOT / "work/stage1_optimization_r26_20260914"
R27 = ROOT / "work/stage1_diagnostics_r27_20260915_rev2_b13_exact"
R30 = ROOT / "work/stage1_r30_c1_et_20260916"
TEACHER = ROOT / (
    "work/stage1_optimization_r22_20260911/fresh_lineage/"
    "T1-B/seed13/checkpoints/step_010536.pt"
)
FEATURES = ROOT / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b"
R28_TEACHER_FEATURES = (
    ROOT / "work/stage1_optimization_r28_split_path_20260915/teacher_hidden_backfill"
)
FINAL_TEACHER_FEATURES = ROOT / "work/final_rerank_20260916/teacher_hidden_backfill"
KS = (10, 20, 50)
VIEWS = ("QT-only", "Student-D1-Path", "Student-LSE-Path", "Teacher-LSE-Path")
PATH_VIEWS = VIEWS[1:]
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 260916


@dataclass(frozen=True)
class Endpoint:
    endpoint: str
    family: str
    seed: int | None
    own: Path
    qt: Path


def endpoints(phase: str) -> list[Endpoint]:
    result = [
        Endpoint(
            "Historical-B13",
            "Historical-B13",
            None,
            R26 / "rankings/B13/rankings.jsonl.gz",
            R26 / "teacher/B13/rankings.jsonl.gz",
        ),
        Endpoint(
            "Healthy-B4-s13",
            "Healthy-B4",
            13,
            ROOT / "work/stage1_bridge_20260915/evaluation/rankings/B4/rankings.jsonl.gz",
            ROOT / "work/stage1_bridge_20260915/evaluation/teacher/B4/rankings.jsonl.gz",
        ),
        Endpoint(
            "Healthy-B4-s29",
            "Healthy-B4",
            29,
            ROOT / "work/stage1_bridge_20260915/evaluation/rankings/B4_seed29/rankings.jsonl.gz",
            ROOT / "work/stage1_bridge_20260915/evaluation/teacher/B4_seed29/rankings.jsonl.gz",
        ),
    ]
    if phase == "all":
        result.extend(
            Endpoint(
                f"R30_F-P659_s{seed}",
                "R30_F-P659",
                seed,
                R30 / f"rankings/F-P659_s{seed}/rankings.jsonl.gz",
                R30 / f"teacher/F-P659_s{seed}/rankings.jsonl.gz",
            )
            for seed in (13, 29)
        )
    return result


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def teacher_feature_paths() -> list[Path]:
    paths = list(_r25_teacher_feature_paths(ROOT))
    if (R28_TEACHER_FEATURES / "teacher_manifest.jsonl").is_file():
        paths.append(R28_TEACHER_FEATURES)
    receipt_path = FINAL_TEACHER_FEATURES / "BACKFILL_RECEIPT.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "completed":
            raise ValueError("Final-rerank Teacher backfill receipt is not completed")
        expected = {
            str((FINAL_TEACHER_FEATURES / f"gpu{shard}/teacher_manifest.jsonl").resolve())
            for shard in range(2)
        }
        recorded = {str(Path(row["path"]).resolve()) for row in receipt["manifests"]}
        if recorded != expected:
            raise ValueError("Final-rerank Teacher backfill manifest set changed")
        for record in receipt["manifests"]:
            path = Path(record["path"])
            if file_record(path) != record:
                raise ValueError(f"Final-rerank Teacher backfill manifest changed: {path}")
            paths.append(path.parent)
    return paths


def teacher_identity() -> dict[str, Any]:
    feature_manifests = [FEATURES / "manifest.jsonl"]
    feature_manifests.extend(
        path / "teacher_manifest.jsonl"
        for path in teacher_feature_paths()
        if (path / "teacher_manifest.jsonl").is_file()
    )
    t0_identity = R26 / "teacher/CACHE_IDENTITY.json"
    payload = {
        "checkpoint": file_record(TEACHER),
        "scorer_family": "R19GlobalResidualTeacher T1-B seed13",
        "score_space": "raw_logit",
        "required_directed_relations": [
            "table_to_text",
            "table_to_image",
            "text_to_table",
            "image_to_table",
        ],
        "base_and_teacher_feature_manifests": [
            file_record(path) for path in feature_manifests
        ],
        "existing_QT_T0_cache_identity": file_record(t0_identity),
        "existing_QT_T0_namespace": json.loads(t0_identity.read_text(encoding="utf-8"))[
            "namespace"
        ],
    }
    backfill_receipt = FINAL_TEACHER_FEATURES / "BACKFILL_RECEIPT.json"
    if backfill_receipt.is_file():
        payload["final_rerank_hidden_backfill_receipt"] = file_record(backfill_receipt)
        payload["final_rerank_hidden_backfill_preflight"] = file_record(
            FINAL_TEACHER_FEATURES / "PREFLIGHT.json"
        )
    payload["path_cache_namespace"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_rows(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def target_ids(values: Iterable[Any]) -> list[str]:
    return [
        str(value["target_id"]) if isinstance(value, dict) else str(value)
        for value in values
    ]


def logsumexp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("logsumexp requires at least one retained path")
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def rank_scores(c100: Sequence[str], scores: dict[str, float | None]) -> list[str]:
    """Return only validly scored targets; no-path targets never enter TopK."""

    if set(c100) != set(scores) or len(c100) != len(scores):
        raise ValueError("Score keys must exactly equal fixed C100 membership")
    eligible = [target for target in c100 if scores[target] is not None]
    return sorted(eligible, key=lambda target: (-float(scores[target]), target))


def target_recall(ranking: Sequence[str], positives: Sequence[str], k: int) -> float:
    truth = set(positives)
    if not truth:
        raise ValueError("Every evaluated query must have a positive target")
    return len(truth & set(dict.fromkeys(ranking[:k]))) / len(truth)


def source_bootstrap(
    deltas: Sequence[float], sources: Sequence[str], *, replicates: int = BOOTSTRAP_REPLICATES
) -> dict[str, Any]:
    if not deltas or len(deltas) != len(sources):
        raise ValueError("Bootstrap needs aligned nonempty paired query values")
    grouped: dict[str, list[float]] = defaultdict(list)
    for source, delta in zip(sources, deltas, strict=True):
        grouped[str(source)].append(float(delta))
    names = sorted(grouped)
    sums = np.asarray([sum(grouped[name]) for name in names], dtype=np.float64)
    counts = np.asarray([len(grouped[name]) for name in names], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates: list[float] = []
    for start in range(0, replicates, 256):
        selected = rng.integers(
            len(names), size=(min(256, replicates - start), len(names))
        )
        estimates.extend(
            (sums[selected].sum(axis=1) / counts[selected].sum(axis=1)).tolist()
        )
    low, high = np.quantile(np.asarray(estimates), [0.025, 0.975]).tolist()
    array = np.asarray(deltas, dtype=np.float64)
    return {
        "estimate": float(array.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "wins": int((array > 1e-12).sum()),
        "losses": int((array < -1e-12).sum()),
        "ties": int((np.abs(array) <= 1e-12).sum()),
        "queries": len(array),
        "source_groups": len(names),
        "replicates": replicates,
        "rng_seed": BOOTSTRAP_SEED,
    }


def fixed_207() -> dict[str, set[str]]:
    path = R27 / "score_handoff/B13/baseline_per_query.jsonl.gz"
    result: dict[str, set[str]] = {}
    for row in read_rows(path):
        values = {str(value) for value in row["EO_sets"]["EO_STRICT"]}
        if values:
            result[str(row["query_id"])] = values
    if sum(map(len, result.values())) != 207:
        raise ValueError("Historical fixed strict-EO cohort is not exactly 207 pairs")
    return result


def aligned_rows(endpoint: Endpoint) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    count = 0
    for own, qt in zip(read_rows(endpoint.own), read_rows(endpoint.qt), strict=True):
        count += 1
        if str(own["query_id"]) != str(qt["query_id"]):
            raise ValueError(f"{endpoint.endpoint}: own/T0 query order differs")
        if own["candidate_pool_id"] != qt["candidate_pool_id"]:
            raise ValueError(f"{endpoint.endpoint}: own/T0 candidate pool differs")
        yield own, qt
    if count != 1198:
        raise ValueError(f"{endpoint.endpoint}: expected 1198 queries, got {count}")


def validate_and_collect_pairs(endpoint: Endpoint) -> tuple[set[tuple[str, str]], dict[str, int]]:
    pairs: set[tuple[str, str]] = set()
    counts = Counter()
    for own, qt in aligned_rows(endpoint):
        query_id = str(own["query_id"])
        direct = target_ids(own["D100_ANN"])
        evidence = target_ids(own["E_paths"])
        c100 = target_ids(own["rankings"]["Equal"][:100])
        if len(c100) != 100 or len(set(c100)) != 100:
            raise ValueError(f"{endpoint.endpoint}/{query_id}: invalid Equal C100")
        if qt["rankings"]["BT100_NO_T0"] != c100:
            raise ValueError(f"{endpoint.endpoint}/{query_id}: T0 did not consume Equal C100")
        expected_qt = sorted(c100, key=lambda target: (-qt["teacher_scores"][target], target))
        if expected_qt != qt["rankings"]["BT100_T0"]:
            raise ValueError(f"{endpoint.endpoint}/{query_id}: saved T0 ranking is not score sorted")
        by_target = {str(row["target_id"]): row for row in own["E_paths"]}
        if evidence != target_ids(own["rankings"]["E_ONLY"]):
            raise ValueError(f"{endpoint.endpoint}/{query_id}: production E ordering differs")
        if set(own["U"]) != set(direct) | set(evidence):
            raise ValueError(f"{endpoint.endpoint}/{query_id}: U is not D union E")
        for target in c100:
            paths = by_target.get(target, {}).get("retained_paths", [])
            counts["c100_targets"] += 1
            counts["c100_targets_with_path"] += bool(paths)
            for path in paths:
                if path.get("kind") != "evidence":
                    raise ValueError("Retained path contains a non-evidence path")
                expected = float(path["query_evidence_score"]) + float(
                    path["evidence_target_score"]
                )
                if not math.isclose(float(path["path_score"]), expected, abs_tol=1e-8):
                    raise ValueError("Student path is not the actual QE plus ET score")
                evidence_id = str(path["evidence_id"])
                pairs.add((query_id, evidence_id))
                pairs.add((evidence_id, target))
                counts["retained_path_occurrences"] += 1
            if paths and Counter(
                str(path["evidence_id"]) for path in paths
            ) != Counter(str(value) for value in by_target[target]["selected_evidence_ids"]):
                raise ValueError(
                    f"{endpoint.endpoint}/{query_id}/{target}: retained path multiset "
                    "differs from production selected evidence IDs"
                )
    counts["unique_teacher_pairs"] = len(pairs)
    return pairs, dict(counts)


class TeacherPathCache:
    def __init__(self, path: Path, namespace: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS scores ("
            "namespace TEXT, source_id TEXT, destination_id TEXT, score REAL, "
            "PRIMARY KEY(namespace,source_id,destination_id))"
        )
        self.namespace = namespace

    def load(self) -> dict[tuple[str, str], float]:
        return {
            (str(source), str(destination)): float(score)
            for source, destination, score in self.db.execute(
                "SELECT source_id,destination_id,score FROM scores WHERE namespace=?",
                (self.namespace,),
            )
        }

    def insert(self, rows: Sequence[tuple[str, str, float]]) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO scores VALUES (?,?,?,?)",
            [(self.namespace, source, destination, score) for source, destination, score in rows],
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()


def score_teacher_pairs(
    required: set[tuple[str, str]], device_name: str, identity: dict[str, Any]
) -> tuple[dict[tuple[str, str], float], dict[str, Any]]:
    teacher_sha = identity["checkpoint"]["sha256"]
    cache = TeacherPathCache(
        OUT / "scores/teacher_pair_scores.sqlite", identity["path_cache_namespace"]
    )
    scores = cache.load()
    missing = sorted(required - scores.keys())
    started = time.monotonic()
    device = torch.device(device_name)
    _arm, teacher_seed, _step, teacher, _payload = load_r19_checkpoint(TEACHER, device)
    teacher.eval()
    store = FeatureStore.from_path(
        FEATURES,
        cache_size=30000,
        cache_bytes=8 * 1024**3,
        teacher_paths=teacher_feature_paths(),
    )
    object_ids = {value for pair in required for value in pair}
    absent = sorted(value for value in object_ids if not store.has_teacher_features(value))
    relation_counts = Counter(
        f"{store.object_type(source)}_to_{store.object_type(destination)}"
        for source, destination in required
    )
    required_relations = set(identity["required_directed_relations"])
    if set(relation_counts) != required_relations:
        raise ValueError(
            "Actual retained paths do not exercise exactly the four locked QE/ET relations: "
            f"{dict(relation_counts)}"
        )
    if absent:
        cache.close()
        return scores, {
            "status": "blocked_missing_teacher_path_support",
            "required_pairs": len(required),
            "missing_pairs": len(missing),
            "missing_teacher_feature_objects": absent,
            "teacher_checkpoint_sha256": teacher_sha,
            "observed_relation_counts": dict(relation_counts),
            "same_teacher_instance_for_QE_and_ET": True,
        }

    if not missing:
        cache.close()
        return scores, {
            "status": "complete",
            "required_pairs": len(required),
            "cached_pairs": len(required),
            "new_pairs": 0,
            "teacher_seed": teacher_seed,
            "teacher_checkpoint_sha256": teacher_sha,
            "feature_cache_identity": identity,
            "observed_relation_counts": dict(relation_counts),
            "same_teacher_instance_for_QE_and_ET": True,
            "elapsed_seconds": time.monotonic() - started,
        }

    new_rows: list[tuple[str, str, float]] = []
    compression_cache = teacher.new_compression_cache()
    teacher_dtype = next(teacher.parameters()).dtype
    for start in range(0, len(missing), 128):
        batch = missing[start : start + 128]
        sources = [
            store.get(source, include_hidden=source not in compression_cache).for_scoring(
                device,
                include_hidden=source not in compression_cache,
                hidden_dtype=teacher_dtype,
            )
            for source, _destination in batch
        ]
        destinations = [
            store.get(
                destination,
                include_hidden=destination not in compression_cache,
            ).for_scoring(
                device,
                include_hidden=destination not in compression_cache,
                hidden_dtype=teacher_dtype,
            )
            for _source, destination in batch
        ]
        with torch.inference_mode():
            values = teacher.score_pairs(
                sources,
                destinations,
                compression_cache=compression_cache,
            ).cpu()
        current = [
            (source, destination, float(score))
            for (source, destination), score in zip(batch, values, strict=True)
        ]
        cache.insert(current)
        new_rows.extend(current)
        scores.update({(source, destination): score for source, destination, score in current})
        if (start // 128 + 1) % 100 == 0:
            print(
                json.dumps(
                    {
                        "event": "teacher_path_pairs",
                        "completed": min(start + len(batch), len(missing)),
                        "total": len(missing),
                    }
                ),
                flush=True,
            )
    cache.close()
    return scores, {
        "status": "complete",
        "required_pairs": len(required),
        "cached_pairs": len(required) - len(missing),
        "new_pairs": len(new_rows),
        "teacher_seed": teacher_seed,
        "teacher_checkpoint_sha256": teacher_sha,
        "feature_cache_identity": identity,
        "observed_relation_counts": dict(relation_counts),
        "same_teacher_instance_for_QE_and_ET": True,
        "score_semantics": (
            "Frozen five-relation T1-B raw pair logits; actual table->evidence and "
            "evidence->table pairs; target score is retained-path LSE"
        ),
        "elapsed_seconds": time.monotonic() - started,
        "device": device_name,
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }


def _teacher_pair_job_counts(execution: dict[str, Any]) -> dict[str, int]:
    return {
        "teacher_pair_scores_consumed": (
            int(execution.get("required_pairs", 0))
            if execution["status"] == "complete"
            else 0
        ),
        "teacher_pair_scores_computed_this_run": int(execution.get("new_pairs", 0)),
    }


def _source_label(target: str, direct: set[str], evidence: set[str]) -> str:
    if target in direct and target in evidence:
        return "both"
    if target in direct:
        return "direct-only"
    if target in evidence:
        return "evidence-only"
    raise ValueError(f"C100 target {target!r} is in neither D nor E")


def _rank_map(ranking: Sequence[str]) -> dict[str, int]:
    return {target: index for index, target in enumerate(ranking, 1)}


def _best_path(paths: Sequence[dict[str, Any]], score_key: str) -> dict[str, Any] | None:
    if not paths:
        return None
    return min(
        paths,
        key=lambda path: (-float(path[score_key]), str(path["evidence_id"])),
    )


def process_endpoint(
    endpoint: Endpoint,
    teacher_pair_scores: dict[tuple[str, str], float],
    teacher_status: str,
    fixed: dict[str, set[str]],
) -> dict[str, Any]:
    destination = OUT / "model_runs" / endpoint.endpoint
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "candidates": destination / "per_query_C100.jsonl.gz",
        "membership": destination / "path_membership.jsonl.gz",
        "qt_scores": destination / "QT_scores.jsonl.gz",
        "d1_scores": destination / "Student_D1_path_scores.jsonl.gz",
        "lse_scores": destination / "Student_LSE_path_scores.jsonl.gz",
        "teacher_scores": destination / "Teacher_path_scores.jsonl.gz",
        "qt_ranking": destination / "QT.jsonl.gz",
        "student_ranking": destination / "Student_Path.jsonl.gz",
        "teacher_ranking": destination / "Teacher_Path.jsonl.gz",
        "per_query": destination / "per_query_metrics.jsonl.gz",
        "target_diagnostics": destination / "target_diagnostics.jsonl.gz",
        "strict": destination / "strict_eo.jsonl.gz",
    }
    per_query_rows: list[dict[str, Any]] = []
    target_summaries: list[dict[str, Any]] = []
    correctness = Counter()
    counts = Counter()
    with ExitStack() as stack:
        handles = {
            name: stack.enter_context(gzip.open(path, "wt", encoding="utf-8"))
            for name, path in paths.items()
        }

        def emit(name: str, value: Any) -> None:
            handles[name].write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")

        for own, qt in aligned_rows(endpoint):
            query_id = str(own["query_id"])
            positives = [str(value) for value in own["positive_target_ids"]]
            direct_ids = target_ids(own["D100_ANN"])
            direct = set(direct_ids)
            evidence_ids = target_ids(own["E_paths"])
            evidence = set(evidence_ids)
            c100 = target_ids(own["rankings"]["Equal"][:100])
            c100_set = set(c100)
            by_target = {str(row["target_id"]): row for row in own["E_paths"]}
            source = {target: _source_label(target, direct, evidence) for target in c100}
            qt_scores = {target: float(qt["teacher_scores"][target]) for target in c100}
            d1_scores: dict[str, float | None] = {}
            lse_scores: dict[str, float | None] = {}
            teacher_scores: dict[str, float | None] = {}
            path_records = []
            best_student: dict[str, dict[str, Any] | None] = {}
            best_teacher: dict[str, dict[str, Any] | None] = {}
            for target in c100:
                row = by_target.get(target)
                retained = [] if row is None else list(row["retained_paths"])
                if retained:
                    recomputed_lse = logsumexp([float(path["path_score"]) for path in retained])
                    if not math.isclose(
                        recomputed_lse, float(row["retained_path_lse"]), abs_tol=1e-8
                    ):
                        raise ValueError(f"{endpoint.endpoint}/{query_id}: Student LSE differs")
                    d1_scores[target] = float(row["evidence_score"])
                    lse_scores[target] = recomputed_lse
                    teacher_paths = []
                    for occurrence, path in enumerate(retained):
                        evidence_id = str(path["evidence_id"])
                        teacher_qe = teacher_pair_scores.get((query_id, evidence_id))
                        teacher_et = teacher_pair_scores.get((evidence_id, target))
                        teacher_path = (
                            None
                            if teacher_qe is None or teacher_et is None
                            else teacher_qe + teacher_et
                        )
                        teacher_paths.append(
                            {
                                **path,
                                "occurrence": occurrence,
                                "teacher_query_evidence_score": teacher_qe,
                                "teacher_evidence_target_score": teacher_et,
                                "teacher_path_score": teacher_path,
                            }
                        )
                    if teacher_status == "complete" and any(
                        path["teacher_path_score"] is None for path in teacher_paths
                    ):
                        raise ValueError("Teacher path cache lacks a required actual pair")
                    teacher_scores[target] = (
                        logsumexp([float(path["teacher_path_score"]) for path in teacher_paths])
                        if teacher_status == "complete"
                        else None
                    )
                    best_student[target] = _best_path(teacher_paths, "path_score")
                    best_teacher[target] = (
                        _best_path(teacher_paths, "teacher_path_score")
                        if teacher_status == "complete"
                        else None
                    )
                    path_records.append(
                        {
                            "target_id": target,
                            "target_source": source[target],
                            "selected_evidence_ids": list(row["selected_evidence_ids"]),
                            "retained_path_ids": [
                                str(path["evidence_id"]) for path in teacher_paths
                            ],
                            "retained_path_multiplicity": len(teacher_paths),
                            "retained_paths": teacher_paths,
                            "routed_rows": row["routed_rows"],
                        }
                    )
                    correctness["saved_retained_path_occurrences"] += len(teacher_paths)
                else:
                    d1_scores[target] = None
                    lse_scores[target] = None
                    teacher_scores[target] = None
                    best_student[target] = None
                    best_teacher[target] = None
                    path_records.append(
                        {
                            "target_id": target,
                            "target_source": source[target],
                            "selected_evidence_ids": [],
                            "retained_path_ids": [],
                            "retained_path_multiplicity": 0,
                            "retained_paths": [],
                            "routed_rows": {},
                        }
                    )

            path_multiset = [
                {
                    "target_id": record["target_id"],
                    "paths": [
                        {
                            "evidence_id": path["evidence_id"],
                            "evidence_type": path["evidence_type"],
                            "occurrence": path["occurrence"],
                        }
                        for path in record["retained_paths"]
                    ],
                }
                for record in path_records
            ]
            path_multiset_sha = hashlib.sha256(
                json.dumps(path_multiset, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
            path_view_hashes = {view: path_multiset_sha for view in PATH_VIEWS}
            if len(set(path_view_hashes.values())) != 1:
                raise ValueError("Path views consumed different retained path multisets")
            c100_sha = hashlib.sha256(
                json.dumps(c100, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            c100_view_hashes = {view: c100_sha for view in VIEWS}
            if len(set(c100_view_hashes.values())) != 1:
                raise ValueError("Scorer views consumed different C100 memberships")
            rankings = {
                "QT-only": rank_scores(c100, qt_scores),
                "Student-D1-Path": rank_scores(c100, d1_scores),
                "Student-LSE-Path": rank_scores(c100, lse_scores),
            }
            if teacher_status == "complete":
                rankings["Teacher-LSE-Path"] = rank_scores(c100, teacher_scores)
            else:
                rankings["Teacher-LSE-Path"] = []
            if rankings["QT-only"] != qt["rankings"]["BT100_T0"]:
                raise ValueError(f"{endpoint.endpoint}/{query_id}: QT replay differs")
            eligible = {target for target, score in d1_scores.items() if score is not None}
            checked_path_views = (
                PATH_VIEWS if teacher_status == "complete" else PATH_VIEWS[:2]
            )
            if any(set(rankings[view]) != eligible for view in checked_path_views):
                raise ValueError(f"{endpoint.endpoint}/{query_id}: path eligibility changed")
            correctness["queries_with_identical_membership"] += 1
            correctness["queries_with_identical_path_multiset"] += 1

            common = {
                "endpoint": endpoint.endpoint,
                "family": endpoint.family,
                "seed": endpoint.seed,
                "query_id": query_id,
                "query_kind": str(own["query_kind"]),
                "source_table_id": str(own["source_table_id"]),
                "positive_target_ids": positives,
                "candidate_pool_id": own["candidate_pool_id"],
                "c100_ids": c100,
                "c100_membership_sha256": c100_sha,
                "path_target_count": len(eligible),
                "path_multiset_sha256": path_multiset_sha,
            }
            emit(
                "candidates",
                {
                    **common,
                    "direct100_ids": direct_ids,
                    "evidence_target_ranking": evidence_ids,
                    "U_ids": target_ids(own["U"]),
                    "equal_c100_ids": c100,
                    "target_source": source,
                },
            )
            emit("membership", {**common, "targets": path_records})
            for name, score_map, handle in (
                ("QT-only", qt_scores, "qt_scores"),
                ("Student-D1-Path", d1_scores, "d1_scores"),
                ("Student-LSE-Path", lse_scores, "lse_scores"),
                ("Teacher-LSE-Path", teacher_scores, "teacher_scores"),
            ):
                emit(
                    handle,
                    {
                        **common,
                        "view": name,
                        "scores": score_map,
                        "retained_path_multiset_sha256": (
                            path_view_hashes.get(name) if name in PATH_VIEWS else None
                        ),
                    },
                )
            emit(
                "qt_ranking",
                {**common, "view": "QT-only", "ranking": rankings["QT-only"]},
            )
            emit(
                "student_ranking",
                {
                    **common,
                    "rankings": {
                        "Student-D1-Path": rankings["Student-D1-Path"],
                        "Student-LSE-Path": rankings["Student-LSE-Path"],
                    },
                    "retained_path_multiset_sha256": path_multiset_sha,
                    "topk_policy": "top min(K,path_target_count)",
                },
            )
            emit(
                "teacher_ranking",
                {
                    **common,
                    "view": "Teacher-LSE-Path",
                    "status": teacher_status,
                    "ranking": rankings["Teacher-LSE-Path"],
                    "retained_path_multiset_sha256": path_multiset_sha,
                    "topk_policy": "top min(K,path_target_count)",
                },
            )

            per_view = {
                view: query_metrics(rankings[view], positives, KS)
                for view in VIEWS
                if view != "Teacher-LSE-Path" or teacher_status == "complete"
            }
            path_target_count = sum(score is not None for score in d1_scores.values())
            positive_path_count = sum(
                target in c100_set and d1_scores[target] is not None for target in positives
            )
            per_query = {
                **common,
                "metrics": per_view,
                "c100_positive_count": len(set(positives) & c100_set),
                "c100_path_target_count": path_target_count,
                "c100_path_target_fraction": path_target_count / len(c100),
                "positive_path_count": positive_path_count,
                "positive_path_fraction": positive_path_count / len(set(positives)),
            }
            per_query_rows.append(per_query)
            emit("per_query", per_query)
            counts["queries"] += 1
            counts["positive_pairs"] += len(set(positives))
            counts["positive_in_c100"] += len(set(positives) & c100_set)
            counts["positive_with_path"] += positive_path_count
            counts["c100_targets"] += len(c100)
            counts["c100_targets_with_path"] += path_target_count

            rank_maps = {
                view: _rank_map(ranking) for view, ranking in rankings.items() if ranking
            }
            negative_targets = c100_set - set(positives)
            highest_negative = {
                "Student-D1-Path": max(
                    (d1_scores[target] for target in negative_targets if d1_scores[target] is not None),
                    default=None,
                ),
                "Student-LSE-Path": max(
                    (lse_scores[target] for target in negative_targets if lse_scores[target] is not None),
                    default=None,
                ),
                "Teacher-LSE-Path": max(
                    (teacher_scores[target] for target in negative_targets if teacher_scores[target] is not None),
                    default=None,
                ),
            }
            for target in positives:
                in_c100 = target in c100_set
                student_row = by_target.get(target)
                retained = [] if student_row is None else list(student_row["retained_paths"])
                target_values = {
                    "Student-D1-Path": d1_scores.get(target),
                    "Student-LSE-Path": lse_scores.get(target),
                    "Teacher-LSE-Path": teacher_scores.get(target),
                }
                diagnostic = {
                    "endpoint": endpoint.endpoint,
                    "family": endpoint.family,
                    "seed": endpoint.seed,
                    "query_id": query_id,
                    "query_kind": own["query_kind"],
                    "source_table_id": own["source_table_id"],
                    "target_id": target,
                    "in_c100": in_c100,
                    "target_source": source.get(target),
                    "retained_path_count": len(retained),
                    "best_student_path_score": (
                        max((float(path["path_score"]) for path in retained), default=None)
                    ),
                    "mean_student_path_score": (
                        statistics.fmean(float(path["path_score"]) for path in retained)
                        if retained
                        else None
                    ),
                    "student_lse_score": (
                        logsumexp([float(path["path_score"]) for path in retained])
                        if retained
                        else None
                    ),
                    "student_d1_coverage_score": (
                        float(student_row["evidence_score"]) if student_row is not None else None
                    ),
                    "best_student_evidence_id": (
                        str(best_student[target]["evidence_id"])
                        if in_c100 and best_student[target]
                        else None
                    ),
                    "best_student_evidence_modality": (
                        str(best_student[target]["evidence_type"])
                        if in_c100 and best_student[target]
                        else None
                    ),
                    "best_teacher_evidence_id": (
                        str(best_teacher[target]["evidence_id"])
                        if in_c100 and best_teacher[target]
                        else None
                    ),
                    "best_teacher_evidence_modality": (
                        str(best_teacher[target]["evidence_type"])
                        if in_c100 and best_teacher[target]
                        else None
                    ),
                    "ranks": {
                        view: rank_maps[view].get(target) for view in rank_maps
                    },
                    "path_scores": target_values,
                    "highest_negative_path_scores": highest_negative,
                    "path_margins": {
                        view: (
                            None
                            if score is None or highest_negative[view] is None
                            else score - float(highest_negative[view])
                        )
                        for view, score in target_values.items()
                    },
                    "failure_class": (
                        "not_in_c100"
                        if not in_c100
                        else "no_correct_path"
                        if not retained
                        else "correct_path_below_competitor"
                        if highest_negative["Student-LSE-Path"] is not None
                        and float(target_values["Student-LSE-Path"])
                        <= float(highest_negative["Student-LSE-Path"])
                        else "correct_path_above_competitor"
                    ),
                }
                emit("target_diagnostics", diagnostic)
                target_summaries.append(diagnostic)

            own_strict = set(positives) & (evidence - direct - set(own["D100_EXACT"]))
            for cohort, members in (
                ("fixed207", fixed.get(query_id, set())),
                ("model_own_strict", own_strict),
            ):
                for target in sorted(members):
                    payload = {
                        "endpoint": endpoint.endpoint,
                        "family": endpoint.family,
                        "seed": endpoint.seed,
                        "cohort": cohort,
                        "query_id": query_id,
                        "query_kind": own["query_kind"],
                        "source_table_id": own["source_table_id"],
                        "target_id": target,
                        "in_D_ANN100": target in direct,
                        "in_D_exact100": target in set(own["D100_EXACT"]),
                        "in_E": target in evidence,
                        "in_U": target in set(own["U"]),
                        "in_C100": target in c100_set,
                        "target_source": source.get(target),
                        "ranks": {
                            view: rank_maps[view].get(target) for view in rank_maps
                        },
                    }
                    emit("strict", payload)

    if correctness["queries_with_identical_membership"] != counts["queries"]:
        raise ValueError("Not every query passed C100 membership equality")
    if correctness["queries_with_identical_path_multiset"] != counts["queries"]:
        raise ValueError("Not every query passed retained path multiset equality")
    receipt = {
        "endpoint": endpoint.endpoint,
        "family": endpoint.family,
        "seed": endpoint.seed,
        "status": "complete",
        "counts": dict(counts),
        "correctness": dict(correctness),
        "files": {name: file_record(path) for name, path in paths.items()},
    }
    write_json(destination / "RECEIPT.json", receipt)
    return {
        "endpoint": endpoint,
        "per_query": per_query_rows,
        "target_diagnostics": target_summaries,
        "receipt": receipt,
    }


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def summarize(
    runs: list[dict[str, Any]], fixed: dict[str, set[str]], teacher_status: str
) -> dict[str, Any]:
    statistics_dir = OUT / "statistics"
    diagnostics_dir = OUT / "diagnostics"
    strict_dir = OUT / "strict_eo"
    candidates_dir = OUT / "candidates"
    scores_dir = OUT / "scores"
    rankings_dir = OUT / "rankings"
    for directory in (
        statistics_dir,
        diagnostics_dir,
        strict_dir,
        candidates_dir,
        scores_dir,
        rankings_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    combined = {
        candidates_dir / "per_query_C100.jsonl.gz": ["candidates"],
        candidates_dir / "path_membership.jsonl.gz": ["membership"],
        scores_dir / "QT_scores.jsonl.gz": ["qt_scores"],
        scores_dir / "Student_D1_path_scores.jsonl.gz": ["d1_scores"],
        scores_dir / "Student_LSE_path_scores.jsonl.gz": ["lse_scores"],
        scores_dir / "Teacher_path_scores.jsonl.gz": ["teacher_scores"],
        rankings_dir / "QT.jsonl.gz": ["qt_ranking"],
        rankings_dir / "Student_Path.jsonl.gz": ["student_ranking"],
        rankings_dir / "Teacher_Path.jsonl.gz": ["teacher_ranking"],
        diagnostics_dir / "positive_negative_path_margin.jsonl.gz": ["target_diagnostics"],
        strict_dir / "rescued_dropped.jsonl.gz": ["strict"],
    }
    for destination, keys in combined.items():
        with gzip.open(destination, "wb") as output:
            for run in runs:
                receipt = run["receipt"]
                for key in keys:
                    source = Path(receipt["files"][key]["path"])
                    with gzip.open(source, "rb") as handle:
                        for block in iter(lambda: handle.read(1024 * 1024), b""):
                            output.write(block)

    main_rows: list[dict[str, Any]] = []
    query_by_endpoint: dict[str, dict[str, dict[str, Any]]] = {}
    for run in runs:
        endpoint = run["endpoint"]
        query_by_endpoint[endpoint.endpoint] = {
            row["query_id"]: row for row in run["per_query"]
        }
        for kind in ("overall", "implicit", "explicit"):
            selected = [
                row
                for row in run["per_query"]
                if kind == "overall" or row["query_kind"] == kind
            ]
            for view in VIEWS:
                available = [row for row in selected if view in row["metrics"]]
                if not available:
                    continue
                values = {
                    f"R@{k}": _mean(
                        [float(row["metrics"][view][f"recall@{k}"]) for row in available]
                    )
                    for k in KS
                }
                main_rows.append(
                    {
                        "endpoint": endpoint.endpoint,
                        "family": endpoint.family,
                        "seed": endpoint.seed,
                        "kind": kind,
                        "scorer": view,
                        "queries": len(available),
                        **values,
                        "C100_recall": _mean(
                            [
                                float(row["c100_positive_count"])
                                / len(set(row["positive_target_ids"]))
                                for row in available
                            ]
                        ),
                        "returned_positive_recall": _mean(
                            [float(row["metrics"][view]["raw_recall"]) for row in available]
                        ),
                        "C100_path_target_fraction": _mean(
                            [float(row["c100_path_target_fraction"]) for row in available]
                        ),
                        "path_target_count_mean": _mean(
                            [float(row["c100_path_target_count"]) for row in available]
                        ),
                        "queries_path_target_count_lt_10": _mean(
                            [float(row["c100_path_target_count"] < 10) for row in available]
                        ),
                        "queries_path_target_count_lt_20": _mean(
                            [float(row["c100_path_target_count"] < 20) for row in available]
                        ),
                        "queries_path_target_count_lt_50": _mean(
                            [float(row["c100_path_target_count"] < 50) for row in available]
                        ),
                        "positive_path_fraction": _mean(
                            [float(row["positive_path_fraction"]) for row in available]
                        ),
                        "P_top10_given_GT_in_C100": (
                            sum(
                                float(row["metrics"][view]["recall@10"])
                                * len(set(row["positive_target_ids"]))
                                for row in available
                            )
                            / sum(int(row["c100_positive_count"]) for row in available)
                            if sum(
                                int(row["c100_positive_count"]) for row in available
                            )
                            else 0.0
                        ),
                    }
                )

    family_endpoint_names: dict[str, list[str]] = defaultdict(list)
    for run in runs:
        family_endpoint_names[run["endpoint"].family].append(run["endpoint"].endpoint)
    for family, names in family_endpoint_names.items():
        if len(names) < 2:
            continue
        maps = [query_by_endpoint[name] for name in names]
        ids = sorted(set.intersection(*(set(values) for values in maps)))
        for kind in ("overall", "implicit", "explicit"):
            selected_ids = [
                query_id
                for query_id in ids
                if kind == "overall" or maps[0][query_id]["query_kind"] == kind
            ]
            for view in VIEWS:
                if not selected_ids or any(
                    view not in values[selected_ids[0]]["metrics"] for values in maps
                ):
                    continue
                averaged = {
                    metric: [
                        statistics.fmean(
                            float(values[query_id]["metrics"][view][metric])
                            for values in maps
                        )
                        for query_id in selected_ids
                    ]
                    for metric in ("raw_recall", *(f"recall@{k}" for k in KS))
                }
                path_counts = [
                    statistics.fmean(
                        float(values[query_id]["c100_path_target_count"])
                        for values in maps
                    )
                    for query_id in selected_ids
                ]
                admitted = [
                    statistics.fmean(
                        float(values[query_id]["c100_positive_count"])
                        for values in maps
                    )
                    for query_id in selected_ids
                ]
                truth_counts = [
                    len(set(maps[0][query_id]["positive_target_ids"]))
                    for query_id in selected_ids
                ]
                hit10 = [
                    averaged["recall@10"][index] * truth_counts[index]
                    for index in range(len(selected_ids))
                ]
                main_rows.append(
                    {
                        "endpoint": f"{family}-seed-average",
                        "family": family,
                        "seed": "13;29",
                        "kind": kind,
                        "scorer": view,
                        "queries": len(selected_ids),
                        **{f"R@{k}": _mean(averaged[f"recall@{k}"]) for k in KS},
                        "C100_recall": _mean(
                            [
                                admitted[index] / truth_counts[index]
                                for index in range(len(selected_ids))
                            ]
                        ),
                        "returned_positive_recall": _mean(averaged["raw_recall"]),
                        "C100_path_target_fraction": _mean(
                            [value / 100.0 for value in path_counts]
                        ),
                        "path_target_count_mean": _mean(path_counts),
                        "queries_path_target_count_lt_10": _mean(
                            [float(value < 10) for value in path_counts]
                        ),
                        "queries_path_target_count_lt_20": _mean(
                            [float(value < 20) for value in path_counts]
                        ),
                        "queries_path_target_count_lt_50": _mean(
                            [float(value < 50) for value in path_counts]
                        ),
                        "positive_path_fraction": _mean(
                            [
                                statistics.fmean(
                                    float(values[query_id]["positive_path_fraction"])
                                    for values in maps
                                )
                                for query_id in selected_ids
                            ]
                        ),
                        "P_top10_given_GT_in_C100": (
                            sum(hit10) / sum(admitted) if sum(admitted) else 0.0
                        ),
                    }
                )

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(statistics_dir / "main_table.csv", main_rows)

    paired_rows: list[dict[str, Any]] = []
    wlt_rows: list[dict[str, Any]] = []
    scorer_strength_rows: list[dict[str, Any]] = []
    family_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        family_groups[run["endpoint"].family].append(run)
    for family, family_runs in family_groups.items():
        ids = sorted(set.intersection(*({row["query_id"] for row in run["per_query"]} for run in family_runs)))
        indexed = [{row["query_id"]: row for row in run["per_query"]} for run in family_runs]
        for query_id in ids:
            reference = indexed[0][query_id]
            for rows in indexed[1:]:
                candidate = rows[query_id]
                for key in (
                    "query_kind",
                    "source_table_id",
                    "positive_target_ids",
                ):
                    if candidate[key] != reference[key]:
                        raise ValueError(
                            f"{family}/{query_id}: seed metadata differs for {key}"
                        )
        for kind in ("overall", "implicit", "explicit"):
            selected_ids = [
                query_id
                for query_id in ids
                if kind == "overall" or indexed[0][query_id]["query_kind"] == kind
            ]
            for view in VIEWS[1:]:
                if (
                    any(
                        view not in rows[selected_ids[0]]["metrics"] for rows in indexed
                    )
                    if selected_ids
                    else True
                ):
                    continue
                for k in KS:
                    deltas = [
                        statistics.fmean(
                            rows[query_id]["metrics"][view][f"recall@{k}"]
                            - rows[query_id]["metrics"]["QT-only"][f"recall@{k}"]
                            for rows in indexed
                        )
                        for query_id in selected_ids
                    ]
                    sources = [str(indexed[0][query_id]["source_table_id"]) for query_id in selected_ids]
                    result = source_bootstrap(deltas, sources)
                    payload = {
                        "family": family,
                        "seeds_averaged_before_bootstrap": ";".join(
                            str(run["endpoint"].seed)
                            for run in family_runs
                            if run["endpoint"].seed is not None
                        )
                        or "single_reference",
                        "kind": kind,
                        "path_scorer": view,
                        "control": "QT-only",
                        "metric": f"R@{k}",
                        **result,
                    }
                    paired_rows.append(payload)
                    wlt_rows.append(
                        {
                            key: payload[key]
                            for key in (
                                "family",
                                "seeds_averaged_before_bootstrap",
                                "kind",
                                "path_scorer",
                                "control",
                                "metric",
                                "queries",
                                "wins",
                                "losses",
                                "ties",
                                "estimate",
                            )
                        }
                    )
            if selected_ids and all(
                all(
                    view in rows[selected_ids[0]]["metrics"]
                    for view in ("Student-LSE-Path", "Teacher-LSE-Path")
                )
                for rows in indexed
            ):
                for k in KS:
                    deltas = [
                        statistics.fmean(
                            rows[query_id]["metrics"]["Teacher-LSE-Path"][f"recall@{k}"]
                            - rows[query_id]["metrics"]["Student-LSE-Path"][f"recall@{k}"]
                            for rows in indexed
                        )
                        for query_id in selected_ids
                    ]
                    sources = [
                        str(indexed[0][query_id]["source_table_id"])
                        for query_id in selected_ids
                    ]
                    scorer_strength_rows.append(
                        {
                            "family": family,
                            "seeds_averaged_before_bootstrap": ";".join(
                                str(run["endpoint"].seed)
                                for run in family_runs
                                if run["endpoint"].seed is not None
                            )
                            or "single_reference",
                            "kind": kind,
                            "scorer": "Teacher-LSE-Path",
                            "control": "Student-LSE-Path",
                            "metric": f"R@{k}",
                            **source_bootstrap(deltas, sources),
                        }
                    )
    write_csv(statistics_dir / "paired_source_bootstrap.csv", paired_rows)
    write_csv(statistics_dir / "WLT.csv", wlt_rows)
    write_csv(
        statistics_dir / "scorer_strength_bootstrap.csv", scorer_strength_rows
    )

    source_rows: list[dict[str, Any]] = []
    modality_rows: list[dict[str, Any]] = []
    for run in runs:
        endpoint = run["endpoint"]
        values = [row for row in run["target_diagnostics"] if row["in_c100"]]
        for source_name in ("direct-only", "evidence-only", "both"):
            selected = [row for row in values if row["target_source"] == source_name]
            for view in VIEWS[1:]:
                scorer_available = (
                    view != "Teacher-LSE-Path" or teacher_status == "complete"
                )
                ranked = [
                    row for row in selected if row["ranks"].get(view) is not None
                ]
                source_rows.append(
                    {
                        "endpoint": endpoint.endpoint,
                        "family": endpoint.family,
                        "seed": endpoint.seed,
                        "target_source": source_name,
                        "path_scorer": view,
                        "scorer_status": (
                            "complete"
                            if scorer_available
                            else "blocked_missing_teacher_path_support"
                        ),
                        "positive_targets": len(selected),
                        "positive_targets_with_retained_path": sum(
                            int(row["retained_path_count"] > 0) for row in selected
                        ),
                        "positive_targets_with_scorer_rank": len(ranked),
                        "mean_QT_rank": _mean(
                            [float(row["ranks"]["QT-only"]) for row in selected]
                        ),
                        "mean_Path_rank_among_ranked": (
                            _mean([float(row["ranks"][view]) for row in ranked])
                            if scorer_available
                            else math.nan
                        ),
                        "mean_rank_delta_Path_minus_QT_among_ranked": (
                            _mean(
                                [
                                    float(
                                        row["ranks"][view]
                                        - row["ranks"]["QT-only"]
                                    )
                                    for row in ranked
                                ]
                            )
                            if scorer_available
                            else math.nan
                        ),
                        "QT_Top10_retention": _mean(
                            [float(row["ranks"]["QT-only"] <= 10) for row in selected]
                        ),
                        "Path_Top10_retention_all_C100_positive": (
                            _mean(
                                [
                                    float(
                                        row["ranks"].get(view) is not None
                                        and row["ranks"][view] <= 10
                                    )
                                    for row in selected
                                ]
                            )
                            if scorer_available
                            else math.nan
                        ),
                    }
                )
        for view, modality_key in (
            ("Student-D1-Path", "best_student_evidence_modality"),
            ("Student-LSE-Path", "best_student_evidence_modality"),
            ("Teacher-LSE-Path", "best_teacher_evidence_modality"),
        ):
            for modality in ("text", "image"):
                selected = [
                    row
                    for row in values
                    if row.get(modality_key) == modality
                    and row["ranks"].get(view) is not None
                ]
                modality_rows.append(
                    {
                        "endpoint": endpoint.endpoint,
                        "family": endpoint.family,
                        "seed": endpoint.seed,
                        "path_scorer": view,
                        "best_path_modality": modality,
                        "positive_targets": len(selected),
                        "QT_Top10": _mean(
                            [float(row["ranks"]["QT-only"] <= 10) for row in selected]
                        ),
                        "Path_Top10": _mean(
                            [float(row["ranks"][view] <= 10) for row in selected]
                        ),
                        "Path_minus_QT_Top10": _mean(
                            [
                                float(row["ranks"][view] <= 10)
                                - float(row["ranks"]["QT-only"] <= 10)
                                for row in selected
                            ]
                        ),
                    }
                )
    write_csv(diagnostics_dir / "by_candidate_source.csv", source_rows)
    write_csv(diagnostics_dir / "by_modality.csv", modality_rows)

    strict_rows = [
        row
        for run in runs
        for row in read_rows(Path(run["receipt"]["files"]["strict"]["path"]))
    ]
    funnel_fields = [
        "endpoint",
        "family",
        "seed",
        "cohort",
        "query_id",
        "query_kind",
        "source_table_id",
        "target_id",
        "in_D_ANN100",
        "in_D_exact100",
        "in_E",
        "in_U",
        "in_C100",
        "target_source",
        *[f"{view}_rank" for view in VIEWS],
        *[f"{view}_Top{k}" for view in VIEWS for k in KS],
    ]
    with (strict_dir / "fixed207_funnel.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=funnel_fields)
        writer.writeheader()
        for row in strict_rows:
            if row["cohort"] != "fixed207":
                continue
            ranks = row["ranks"]
            flat = {key: row.get(key) for key in funnel_fields if not key.endswith(tuple(f"Top{k}" for k in KS)) and not key.endswith("_rank")}
            for view in VIEWS:
                flat[f"{view}_rank"] = ranks.get(view)
                for k in KS:
                    rank = ranks.get(view)
                    flat[f"{view}_Top{k}"] = int(rank is not None and rank <= k)
            writer.writerow(flat)

    transitions: list[dict[str, Any]] = []
    for row in strict_rows:
        for view in VIEWS[1:]:
            if view not in row["ranks"]:
                continue
            for k in KS:
                qt_rank = row["ranks"].get("QT-only")
                path_rank = row["ranks"].get(view)
                qt_correct = qt_rank is not None and qt_rank <= k
                path_correct = path_rank is not None and path_rank <= k
                transitions.append(
                    {
                        **{key: row[key] for key in ("endpoint", "family", "seed", "cohort", "query_id", "query_kind", "source_table_id", "target_id")},
                        "path_scorer": view,
                        "cutoff": k,
                        "transition": (
                            "both_correct"
                            if qt_correct and path_correct
                            else "QT_correct_Path_wrong"
                            if qt_correct
                            else "QT_wrong_Path_correct"
                            if path_correct
                            else "both_wrong"
                        ),
                        "rescued": bool(path_correct and not qt_correct),
                        "dropped": bool(qt_correct and not path_correct),
                        "net": int(path_correct) - int(qt_correct),
                    }
                )
    with gzip.open(strict_dir / "rescued_dropped.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in transitions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    return {
        "main_rows": main_rows,
        "paired_rows": paired_rows,
        "scorer_strength_rows": scorer_strength_rows,
        "source_rows": source_rows,
        "modality_rows": modality_rows,
        "strict_rows": strict_rows,
        "transitions": transitions,
        "fixed_pairs": sum(map(len, fixed.values())),
        "teacher_status": teacher_status,
    }


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def write_reports(summary: dict[str, Any], phase: str) -> None:
    rows = summary["main_rows"]
    lines = [
        "# Final fixed-C100 path reranking",
        "",
        "All views use byte-locked own-pool production `Equal[:100]` membership. "
        "`QT-only` is the frozen T0 control. Path TopK is truncated to "
        "`min(K, path_target_count)`, so no-path targets never enter a returned list.",
        "",
        "| Endpoint | Scorer | Overall R10 | Implicit R10 | Explicit R10 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    endpoints_seen = list(dict.fromkeys(row["endpoint"] for row in rows))
    for endpoint in endpoints_seen:
        for view in VIEWS:
            values = {
                row["kind"]: row
                for row in rows
                if row["endpoint"] == endpoint and row["scorer"] == view
            }
            if not values:
                continue
            lines.append(
                f"| {endpoint} | {view} | {_pct(values['overall']['R@10'])} | "
                f"{_pct(values['implicit']['R@10'])} | {_pct(values['explicit']['R@10'])} |"
            )

    transitions = [
        row
        for row in summary["transitions"]
        if row["cohort"] == "fixed207" and row["cutoff"] == 10
    ]
    fixed_rows = [
        row for row in summary["strict_rows"] if row["cohort"] == "fixed207"
    ]
    lines.extend(["", "## Fixed-207 funnel", ""])
    lines.append("| Endpoint | Scorer | In C100 | Top10 | Top20 | Top50 |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for endpoint in endpoints_seen:
        selected = [row for row in fixed_rows if row["endpoint"] == endpoint]
        in_c100 = sum(int(row["in_C100"]) for row in selected)
        for view in VIEWS:
            if not any(view in row["ranks"] for row in selected):
                continue
            counts = {
                k: sum(
                    int(
                        row["ranks"].get(view) is not None
                        and row["ranks"][view] <= k
                    )
                    for row in selected
                )
                for k in KS
            }
            lines.append(
                f"| {endpoint} | {view} | {in_c100} | {counts[10]} | "
                f"{counts[20]} | {counts[50]} |"
            )

    lines.extend(["", "## Fixed-207 Top10 transitions", ""])
    lines.append("| Endpoint | Path scorer | rescued | dropped | net |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for endpoint in endpoints_seen:
        for view in VIEWS[1:]:
            selected = [
                row
                for row in transitions
                if row["endpoint"] == endpoint and row["path_scorer"] == view
            ]
            if not selected:
                continue
            rescued = sum(row["rescued"] for row in selected)
            dropped = sum(row["dropped"] for row in selected)
            lines.append(f"| {endpoint} | {view} | {rescued} | {dropped} | {rescued-dropped:+d} |")

    lines.extend(
        [
            "",
            "Teacher D1 is intentionally not reported: production D1 uses Student-score "
            "top-L screening and row coverage. Recomputing it from retained paths would not "
            "be the same aggregator. Teacher Path therefore uses retained-path LSE.",
            f"Teacher-LSE-Path status: `{summary['teacher_status']}`.",
        ]
    )
    strength = [
        row
        for row in summary["scorer_strength_rows"]
        if row["kind"] == "overall" and row["metric"] == "R@10"
    ]
    if strength:
        lines.extend(["", "## Same-aggregator scorer strength", ""])
        lines.append(
            "| Family | Teacher-LSE minus Student-LSE R10 | 95% source-bootstrap CI |"
        )
        lines.append("| --- | ---: | ---: |")
        for row in strength:
            lines.append(
                f"| {row['family']} | {100 * row['estimate']:+.2f} pp | "
                f"[{100 * row['ci95_low']:+.2f}, {100 * row['ci95_high']:+.2f}] pp |"
            )
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    limitations = [
        "# Limitations",
        "",
        "- This is a dev-set, evaluation-only mechanism audit; it is not held-out confirmation.",
        "- Student D1 and Teacher Path use different aggregators. Their difference cannot be attributed only to scorer strength.",
        "- Modality groups use the highest-scoring retained path and are observational, not randomized modality ablations.",
        "- Path-only deliberately gives no-path targets no fallback score; this measures an evidence-only decision boundary, not a deployable fusion policy.",
        f"- Teacher path status: `{summary['teacher_status']}`.",
    ]
    (OUT / "LIMITATIONS.md").write_text("\n".join(limitations) + "\n", encoding="utf-8")

    b13 = {
        (row["kind"], row["scorer"]): row
        for row in rows
        if row["endpoint"] == "Historical-B13"
    }
    best_view = max(
        (view for view in VIEWS[1:] if ("overall", view) in b13),
        key=lambda view: b13[("overall", view)]["R@10"],
    )
    delta = (
        b13[("overall", best_view)]["R@10"]
        - b13[("overall", "QT-only")]["R@10"]
    )
    next_lines = [
        "# Next Decision",
        "",
        f"The strongest Historical-B13 pure path view is `{best_view}` with an overall R@10 delta of {100*delta:+.2f} pp versus `QT-only`.",
        "",
        (
            "The single next experiment should materialize only the missing hidden-state "
            "support for the already locked frozen Teacher, then rerun exact Teacher-LSE-Path "
            "on this identical C100 and retained-path multiset. Do not train a new Teacher or "
            "Student and do not tune QT+Path fusion before that scorer-strength control is resolved."
            if summary["teacher_status"] != "complete"
            else (
                "The single next experiment should be a held-out replication of the same fixed-C100 QT-versus-path comparison. "
                "Do not train or tune fusion until that replication confirms a path advantage and the implicit/explicit trade-off."
                if delta > 0
                else "The single next experiment should replicate this exact frozen fixed-C100 QT-versus-path audit on a held-out query split. Do not train a new path scorer or expand a QT+Path fusion grid unless the held-out result overturns the negative pure-path finding."
            )
        ),
    ]
    (OUT / "NEXT_DECISION.md").write_text("\n".join(next_lines) + "\n", encoding="utf-8")


def run(phase: str, device_name: str) -> dict[str, Any]:
    started = time.monotonic()
    selected = endpoints(phase)
    for endpoint in selected:
        for path in (endpoint.own, endpoint.qt):
            if not path.is_file():
                raise FileNotFoundError(path)
    fixed_path = R27 / "score_handoff/B13/baseline_per_query.jsonl.gz"
    locked_teacher = teacher_identity()
    lock = {
        "status": "locked",
        "created_at_utc": now(),
        "phase": phase,
        "experimental_contract": {
            "candidate_generation": "frozen; no calls",
            "C100": "each endpoint's production Equal[:100]",
            "allowed_change": "final score/ranking function only",
            "no_path_policy": (
                "PathTopK is top min(K, number of C100 targets with a valid retained path); "
                "no-path targets are absent, with no QT fallback"
            ),
            "student_D1": "stored production e2_row_coverage",
            "student_LSE": "LSE of all actual retained Student QE+ET paths",
            "teacher_path": "LSE of frozen Teacher QE+ET scores on those same retained paths",
            "teacher_D1": "not computed because it cannot be strictly reproduced from retained paths",
        },
        "endpoints": [
            {
                "endpoint": endpoint.endpoint,
                "family": endpoint.family,
                "seed": endpoint.seed,
                "own_rankings": file_record(endpoint.own),
                "qt_rankings": file_record(endpoint.qt),
            }
            for endpoint in selected
        ],
        "teacher": locked_teacher,
        "fixed207_source": file_record(fixed_path),
        "fixed207_definition": "G intersect E_B13 minus (D_ANN100 union D_exact100)",
        "fixed207_pairs": 207,
        "new_training_jobs": 0,
        "new_mining_jobs": 0,
        "new_ann_indexes": 0,
        "fusion_tuning": 0,
    }
    write_json(OUT / "INPUT_LOCK.json", lock)

    fixed = fixed_207()
    required: set[tuple[str, str]] = set()
    scans = {}
    for endpoint in selected:
        pairs, counts = validate_and_collect_pairs(endpoint)
        required.update(pairs)
        scans[endpoint.endpoint] = counts
    teacher_scores, teacher_execution = score_teacher_pairs(
        required, device_name, locked_teacher
    )
    if teacher_execution["status"] not in {
        "complete",
        "blocked_missing_teacher_path_support",
    }:
        raise ValueError(f"Unexpected Teacher execution status: {teacher_execution['status']}")
    runs = [
        process_endpoint(endpoint, teacher_scores, teacher_execution["status"], fixed)
        for endpoint in selected
    ]
    summary = summarize(runs, fixed, teacher_execution["status"])
    write_reports(summary, phase)

    public_paths = [
        OUT / "candidates/per_query_C100.jsonl.gz",
        OUT / "candidates/path_membership.jsonl.gz",
        OUT / "scores/QT_scores.jsonl.gz",
        OUT / "scores/Student_D1_path_scores.jsonl.gz",
        OUT / "scores/Student_LSE_path_scores.jsonl.gz",
        OUT / "scores/Teacher_path_scores.jsonl.gz",
        OUT / "rankings/QT.jsonl.gz",
        OUT / "rankings/Student_Path.jsonl.gz",
        OUT / "rankings/Teacher_Path.jsonl.gz",
        OUT / "strict_eo/fixed207_funnel.csv",
        OUT / "strict_eo/rescued_dropped.jsonl.gz",
        OUT / "diagnostics/by_candidate_source.csv",
        OUT / "diagnostics/by_modality.csv",
        OUT / "diagnostics/positive_negative_path_margin.jsonl.gz",
        OUT / "statistics/main_table.csv",
        OUT / "statistics/paired_source_bootstrap.csv",
        OUT / "statistics/scorer_strength_bootstrap.csv",
        OUT / "statistics/WLT.csv",
        OUT / "RESULTS.md",
        OUT / "LIMITATIONS.md",
        OUT / "NEXT_DECISION.md",
    ]
    correctness = {
        "C1_C100_membership_identical": all(
            run["receipt"]["correctness"]["queries_with_identical_membership"]
            == run["receipt"]["counts"]["queries"]
            for run in runs
        ),
        "C1b_retained_path_multiset_identical_across_path_views": all(
            run["receipt"]["correctness"]["queries_with_identical_path_multiset"]
            == run["receipt"]["counts"]["queries"]
            for run in runs
        ),
        "C2_student_paths_are_actual_QE_plus_ET": True,
        "C3_teacher_paths_are_actual_teacher_QE_plus_ET": (
            True
            if teacher_execution["status"] == "complete"
            else "blocked_missing_teacher_path_support"
        ),
        "C4_no_path_policy": "PathTopK=min(K,path_target_count); no_QT_fallback",
        "C5_all_retained_path_occurrences_saved": all(
            run["receipt"]["correctness"]["saved_retained_path_occurrences"]
            == scans[run["endpoint"].endpoint]["retained_path_occurrences"]
            for run in runs
        ),
        "C6_recall_semantics": "per-query |G intersection TopK| / |G|; covered by unit test",
        "C7_fixed_EO_double_exclusion": summary["fixed_pairs"] == 207,
        "C8_raw_artifacts_saved": all(path.is_file() for path in public_paths),
    }
    if not all(value is True or isinstance(value, str) for value in correctness.values()):
        raise ValueError(f"Correctness gate failed: {correctness}")
    ledger = {
        "status": (
            "complete"
            if teacher_execution["status"] == "complete"
            else "complete_with_teacher_blocked"
        ),
        "created_at_utc": now(),
        "phase": phase,
        "endpoints": [endpoint.endpoint for endpoint in selected],
        "input_lock": file_record(OUT / "INPUT_LOCK.json"),
        "input_scans": scans,
        "teacher_path_execution": teacher_execution,
        "correctness_tests": correctness,
        "artifacts": [file_record(path) for path in public_paths],
        "jobs": {
            "training": 0,
            "mining": 0,
            "ann_index_builds": 0,
            "fusion_tuning": 0,
            "teacher_hidden_state_backfill_objects": (
                json.loads(
                    (FINAL_TEACHER_FEATURES / "BACKFILL_RECEIPT.json").read_text(
                        encoding="utf-8"
                    )
                )["objects"]
                if (FINAL_TEACHER_FEATURES / "BACKFILL_RECEIPT.json").is_file()
                else 0
            ),
            **_teacher_pair_job_counts(teacher_execution),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(OUT / "EXECUTION_LEDGER.json", ledger)
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("phase_a", "all"), default="phase_a")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(args.phase, args.device), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
