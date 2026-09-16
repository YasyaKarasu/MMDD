"""Prepare and audit the B13 -> modern Stage-1 bridge ladder.

The bridge experiment is intentionally conservative.  It separates frozen
recipe evidence from model training/evaluation, materializes B2/B3 from one
modern base schedule, and records a resource-blocked receipt when CUDA is not
available.  Existing R25/R26 scores are controls only; they are never silently
relabelled as causal bridge stages.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work/stage1_bridge_20260915"
HIST = ROOT / "work/stage1_diagnostics_r27_20260915_rev2_b13_exact"
R12 = ROOT / "work/stage1_optimization_r12_20260908"
R13 = ROOT / "work/stage1_optimization_r13_20260909"
R24 = ROOT / "work/stage1_optimization_r24_20260913"
R25 = ROOT / "work/stage1_optimization_r25_final_20260914"
R26 = ROOT / "work/stage1_optimization_r26_20260914"
SEEDS = (13, 29)
RELATIONS = ("table->table", "table->text", "table->image", "text->table", "image->table")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def record(path: Path) -> dict[str, Any]:
    result = {"path": str(path.resolve()), "exists": path.is_file()}
    if path.is_file():
        result.update(bytes=path.stat().st_size, sha256=sha256(path))
    return result


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rows(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _input_paths(root: Path = ROOT) -> dict[str, Path]:
    return {
        "b0_verdict": root / HIST.relative_to(root) / "historical_replay/H_VERDICT.json",
        "b0_c1_step356": root / HIST.relative_to(root) / "historical_replay/C1/seed13/checkpoints/step_000356.pt",
        "b0_c2_step178": root / HIST.relative_to(root) / "historical_replay/C2/seed13/checkpoints/step_000178.pt",
        "historical_c1_schedule": R12 / "taskC_training/candidates_seed13_steps356/candidates.jsonl.gz",
        "modern_c1_seed13": R25 / "common/c1_selective_hard_seed13.jsonl.gz",
        "modern_c1_seed29": R25 / "common/c1_selective_hard_seed29.jsonl.gz",
        "historical_c2_graph": R12 / "taskC_training/c2_candidates_seed13/path_hard.jsonl",
        "modern_c2_graph": R24 / "path_pool/common_seed13.jsonl",
        "historical_c2_order": R13 / "taskD_witness_supervision/schedule_order.json",
        "modern_c2_order": R26 / "common/c2_order.jsonl",
        "modern_c2_teacher": R25 / "common/teacher_native_path_cache.jsonl.gz",
        "modern_c1_teacher13": R25 / "common/teacher_edge_cache_seed13.jsonl.gz",
        "modern_c1_teacher29": R25 / "common/teacher_edge_cache_seed29.jsonl.gz",
        "modern_current_c1_13": R25 / "training/C1/seed13/checkpoints/step_000659.pt",
        "modern_current_c1_29": R25 / "training/C1/seed29/checkpoints/step_000659.pt",
        "modern_current_c2_13": R26 / "training/O-NATIVE/seed13/checkpoints/step_000178.pt",
        "modern_current_c2_29": R26 / "training/O-NATIVE/seed29/checkpoints/step_000178.pt",
        "modern_current_c2_receipt13": R26 / "training/O-NATIVE/seed13/C2_COMPLETION_RECEIPT.json",
        "modern_current_c2_receipt29": R26 / "training/O-NATIVE/seed29/C2_COMPLETION_RECEIPT.json",
        "modern_current_c2_order": R26 / "common/c2_order.jsonl",
        "modern_current_c2_graph": R24 / "path_pool/common_seed13.jsonl",
        "modern_current_native_cache": R25 / "common/teacher_native_path_cache.jsonl.gz",
        "r26_order_statistics": R26 / "statistics/paired_source_bootstrap.jsonl",
        "bridge_base_schedule13": OUT / "schedules/seed13_steps659/base_full.jsonl.gz",
        "bridge_closure_schedule13": OUT / "schedules/seed13_steps659/closure_full.jsonl.gz",
        "bridge_base_schedule29": OUT / "schedules/seed29_steps659/base_full.jsonl.gz",
        "bridge_closure_schedule29": OUT / "schedules/seed29_steps659/closure_full.jsonl.gz",
        "train_edge_registry": R12 / "taskA_correctness/supervision/edge_lists.train_fit.jsonl",
        "feature_root": root / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b",
        "modern_raw_index": R25 / "common/raw_qwen_index",
    }


def resolve_inputs(root: Path = ROOT) -> dict[str, Any]:
    paths = _input_paths(root)
    resolved = {name: record(path) for name, path in paths.items()}
    required = {
        "b0_verdict", "b0_c1_step356", "b0_c2_step178", "historical_c1_schedule",
        "modern_c1_seed13", "modern_c1_seed29", "historical_c2_graph", "modern_c2_graph",
        "historical_c2_order", "modern_c2_order", "modern_c2_teacher",
        "bridge_base_schedule13", "bridge_closure_schedule13", "bridge_base_schedule29", "bridge_closure_schedule29",
        "modern_current_c1_13", "modern_current_c1_29", "modern_current_c2_13", "modern_current_c2_29",
        "modern_current_c2_receipt13", "modern_current_c2_receipt29", "modern_current_c2_order",
        "modern_current_c2_graph", "modern_current_native_cache",
        "r26_order_statistics",
    }
    missing = sorted(name for name in required if not resolved[name]["exists"])
    payload = {
        "format_version": 1,
        "status": "complete" if not missing else "partial",
        "required_missing": missing,
        "resolved": resolved,
        "b0_identity": "R27 historical fresh replay; not reconstructed from R25/R26 checkpoints",
        "created_by": str(Path(__file__).resolve()),
    }
    write_json(OUT / "EXECUTION_SOURCE_AUDIT.json", payload)
    return payload


def _relation(row: dict[str, Any]) -> str:
    return f"{row.get('source_type')}->{row.get('destination_type')}"


def _ids(row: dict[str, Any]) -> list[str]:
    return list(row.get("candidate_ids", ()))


def _positive_ids(row: dict[str, Any]) -> set[str]:
    return set(row.get("positive_ids", ()))


def _schedule_batches(path: Path) -> list[dict[str, Any]]:
    return list(rows(path))


def _schedule_summary(schedule: list[dict[str, Any]]) -> dict[str, Any]:
    flat = [row for batch in schedule for row in batch.get("examples", ())]
    rel = Counter(_relation(row) for row in flat)
    candidate_counts = [len(_ids(row)) for row in flat]
    return {
        "batches": len(schedule),
        "lists": len(flat),
        "relations": dict(rel),
        "candidate_count": {
            "min": min(candidate_counts, default=0),
            "median": sorted(candidate_counts)[len(candidate_counts) // 2] if candidate_counts else 0,
            "max": max(candidate_counts, default=0),
        },
        "query_order_sha256": stable_sha([row.get("query_id") for row in flat]),
    }


def compare_c1_schedules(historical: Path, modern: Path, *, batches: int = 356) -> dict[str, Any]:
    old = _schedule_batches(historical)[:batches]
    new = _schedule_batches(modern)[:batches]
    if len(old) != len(new):
        raise ValueError(f"C1 schedule length differs: {len(old)} vs {len(new)}")
    counts = Counter()
    per_relation: dict[str, Counter] = defaultdict(Counter)
    for old_batch, new_batch in zip(old, new):
        if len(old_batch.get("examples", ())) != len(new_batch.get("examples", ())):
            raise ValueError("aligned C1 batches have different list counts")
        for left, right in zip(old_batch["examples"], new_batch["examples"]):
            counts["aligned_lists"] += 1
            relation = _relation(left)
            if relation != _relation(right):
                counts["relation_changed"] += 1
                continue
            per_relation[relation]["lists"] += 1
            a, b = _ids(left), _ids(right)
            pa, pb = _positive_ids(left), _positive_ids(right)
            sa, sb = set(a), set(b)
            counts["candidate_order_equal"] += a == b
            counts["candidate_membership_equal"] += sa == sb
            counts["positive_membership_equal"] += pa == pb
            counts["negative_membership_equal"] += (sa - pa) == (sb - pb)
            counts["added_candidates"] += len(sb - sa)
            counts["removed_candidates"] += len(sa - sb)
            counts["added_positives"] += len(pb - pa)
            counts["removed_positives"] += len(pa - pb)
            per_relation[relation]["candidate_membership_equal"] += sa == sb
            per_relation[relation]["positive_membership_equal"] += pa == pb
            per_relation[relation]["added_positives"] += len(pb - pa)
            per_relation[relation]["removed_positives"] += len(pa - pb)
    return {
        "historical": {"path": str(historical), **_schedule_summary(old)},
        "modern": {"path": str(modern), **_schedule_summary(new)},
        "aligned_differences": dict(counts),
        "per_relation": {key: dict(value) for key, value in sorted(per_relation.items())},
        "comparison_scope": f"first {batches} registered batches; position aligned; no training",
    }


def compare_c1_teacher_scores() -> dict[str, Any]:
    """Check whether the R25 edge cache is genuinely a changed Teacher or a
    numerically equivalent reuse of the historical T_core pair scores."""
    manifest_path = R12 / "taskC_training/candidates_seed13_steps356/teacher_pairs.jsonl.gz"
    score_path = R12 / "taskC_training/teacher_pair_scores/scores.pt"
    if not manifest_path.is_file() or not score_path.is_file():
        return {"status": "missing_historical_pair_cache"}
    try:
        import torch
        scores_payload = torch.load(score_path, map_location="cpu", weights_only=True)
        scores = scores_payload["scores"]
    except Exception as exc:
        return {"status": "error", "error": repr(exc)}
    historical = {}
    for row in rows(manifest_path):
        historical[(row["source_id"], f"{row['source_type']}->{row['destination_type']}", row["destination_id"])] = float(scores[row["pair_id"]])
    result = {"status": "complete", "historical_pairs": len(historical), "seeds": {}}
    for seed in SEEDS:
        cache_path = R25 / f"common/teacher_edge_cache_seed{seed}.jsonl.gz"
        if not cache_path.is_file():
            result["seeds"][str(seed)] = {"status": "missing", "path": str(cache_path)}
            continue
        schedule_path = R25 / f"common/c1_selective_hard_seed{seed}.jsonl.gz"
        positive_by_query_relation: dict[tuple[str, str], set[str]] = {}
        if schedule_path.is_file():
            for batch in rows(schedule_path):
                for example in batch.get("examples", ()):
                    positive_by_query_relation[(example["query_id"], _relation(example))] = _positive_ids(example)
        overlap = 0; missing = 0; differences = []
        paired_scores: dict[str, list[array]] = defaultdict(lambda: [array("d"), array("d")])
        by_relation: dict[str, dict[str, Any]] = defaultdict(lambda: {
            "pairs": 0, "abs_sum": 0.0, "max_abs": 0.0,
            "old_sum": 0.0, "modern_sum": 0.0,
            "old_positive": [], "modern_positive": [],
            "old_negative": [], "modern_negative": [],
            "old_margins": [], "modern_margins": [],
        })
        for row in rows(cache_path):
            relation = row["relation"]
            positive_ids = positive_by_query_relation.get((row["query_id"], relation), set())
            old_positive: list[float] = []
            modern_positive: list[float] = []
            old_negative: list[float] = []
            modern_negative: list[float] = []
            for target_id, value in zip(row["candidate_ids"], row["scores"]):
                old = historical.get((row["query_id"], relation, target_id))
                if old is None:
                    missing += 1
                else:
                    value = float(value)
                    overlap += 1; differences.append(abs(old - value))
                    paired_scores[relation][0].append(old)
                    paired_scores[relation][1].append(value)
                    stats = by_relation[relation]
                    stats["pairs"] += 1
                    stats["abs_sum"] += abs(old - value)
                    stats["max_abs"] = max(stats["max_abs"], abs(old - value))
                    stats["old_sum"] += old
                    stats["modern_sum"] += value
                    if target_id in positive_ids:
                        old_positive.append(old)
                        modern_positive.append(value)
                    else:
                        old_negative.append(old)
                        modern_negative.append(value)
            stats = by_relation[relation]
            if old_positive and old_negative:
                old_margin = (sum(old_positive) / len(old_positive)) - (sum(old_negative) / len(old_negative))
                modern_margin = (sum(modern_positive) / len(modern_positive)) - (sum(modern_negative) / len(modern_negative))
                stats["old_margins"].append(old_margin)
                stats["modern_margins"].append(modern_margin)

            # Keep the score pools separate so the report can state whether a
            # Teacher change moved positives, negatives, or their margin.
            stats["old_positive"].extend(old_positive)
            stats["modern_positive"].extend(modern_positive)
            stats["old_negative"].extend(old_negative)
            stats["modern_negative"].extend(modern_negative)

        def _mean(values: list[float]) -> float | None:
            return sum(values) / len(values) if values else None

        per_relation = {}
        for relation, stats in sorted(by_relation.items()):
            old_margins, modern_margins = stats["old_margins"], stats["modern_margins"]
            margin_delta = [right - left for left, right in zip(old_margins, modern_margins)]
            per_relation[relation] = {
                "pairs": stats["pairs"],
                "mean_absolute_difference": stats["abs_sum"] / stats["pairs"] if stats["pairs"] else None,
                "max_absolute_difference": stats["max_abs"] if stats["pairs"] else None,
                "old_mean": stats["old_sum"] / stats["pairs"] if stats["pairs"] else None,
                "modern_mean": stats["modern_sum"] / stats["pairs"] if stats["pairs"] else None,
                "positive_pairs": len(stats["old_positive"]),
                "negative_pairs": len(stats["old_negative"]),
                "old_positive_mean": _mean(stats["old_positive"]),
                "modern_positive_mean": _mean(stats["modern_positive"]),
                "old_negative_mean": _mean(stats["old_negative"]),
                "modern_negative_mean": _mean(stats["modern_negative"]),
                "margin_queries": len(old_margins),
                "old_positive_minus_negative_margin_mean": _mean(old_margins),
                "modern_positive_minus_negative_margin_mean": _mean(modern_margins),
                "positive_minus_negative_margin_delta_mean": _mean(margin_delta),
                "pearson": _corr(paired_scores[relation][0], paired_scores[relation][1]),
                "spearman": _corr(_rank(paired_scores[relation][0]), _rank(paired_scores[relation][1])),
            }
        all_left = array("d")
        all_right = array("d")
        for left, right in paired_scores.values():
            all_left.extend(left)
            all_right.extend(right)
        result["seeds"][str(seed)] = {
            "status": "complete", "cache": record(cache_path), "overlap_pairs": overlap,
            "pairs_not_in_historical_cache": missing,
            "mean_absolute_difference": sum(differences) / len(differences) if differences else None,
            "max_absolute_difference": max(differences) if differences else None,
            "pearson": _corr(all_left, all_right),
            "spearman": _corr(_rank(all_left), _rank(all_right)),
            "correlation_pairs": len(all_left),
            "per_relation": per_relation,
            "interpretation": "R25 edge cache is numerically historical T_core reuse on overlapping pairs" if differences and max(differences) < 1e-4 else "Teacher drift or unavailable overlap",
        }
    return result


def compare_bridge_schedules() -> dict[str, Any]:
    """Audit the shared B2/B3 base and closure schedules against R25."""
    result = {}
    for seed in SEEDS:
        base_path = OUT / f"schedules/seed{seed}_steps659/base_full.jsonl.gz"
        closure_path = OUT / f"schedules/seed{seed}_steps659/closure_full.jsonl.gz"
        modern_path = R25 / f"common/c1_selective_hard_seed{seed}.jsonl.gz"
        if not all(path.is_file() for path in (base_path, closure_path, modern_path)):
            result[str(seed)] = {"status": "missing", "paths": [str(path) for path in (base_path, closure_path, modern_path)]}
            continue
        base = [row for batch in rows(base_path) for row in batch.get("examples", ())]
        closure = [row for batch in rows(closure_path) for row in batch.get("examples", ())]
        modern = [row for batch in rows(modern_path) for row in batch.get("examples", ())]
        first = len(base) and min(356 * 64, len(base))
        counts = Counter()
        by_relation: dict[str, Counter] = defaultdict(Counter)
        for left, right in zip(base[:first], closure[:first]):
            relation = _relation(left)
            old, new = set(left.get("candidate_ids", ())), set(right.get("candidate_ids", ()))
            inserted = new - old
            counts["lists"] += 1
            counts["lists_modified"] += bool(inserted)
            counts["positives_inserted"] += len(inserted)
            counts["negatives_evicted"] += len(old - new)
            by_relation[relation]["lists"] += 1
            by_relation[relation]["lists_modified"] += bool(inserted)
            by_relation[relation]["positives_inserted"] += len(inserted)
            by_relation[relation]["negatives_evicted"] += len(old - new)
        def schedule_semantics(row: dict[str, Any]) -> dict[str, Any]:
            return {key: row.get(key) for key in ("query_id", "source_type", "destination_type", "candidate_ids",
                                                   "positive_ids", "positive_id", "confirmed_labels", "dataset", "split",
                                                   "negative_semantics")}

        def schedule_raw_metadata(row: dict[str, Any]) -> dict[str, Any]:
            return {key: row.get(key) for key in ("raw_ann_negative_quota", "raw_ann_negatives_used")}

        ids_equal = [schedule_semantics(row) for row in closure[:first]] == [schedule_semantics(row) for row in modern[:first]]
        raw_metadata_equal = [schedule_raw_metadata(row) for row in closure[:first]] == [schedule_raw_metadata(row) for row in modern[:first]]

        result[str(seed)] = {
            "status": "complete",
            "base": {"path": str(base_path), "sha256": sha256(base_path), "batches": len(base) // 64 + bool(len(base) % 64), "lists": len(base)},
            "closure": {"path": str(closure_path), "sha256": sha256(closure_path), "batches": len(closure) // 64 + bool(len(closure) % 64), "lists": len(closure)},
            "modern_r25": {"path": str(modern_path), "sha256": sha256(modern_path), "lists": len(modern)},
            "first356": {"base_vs_closure": dict(counts),
                         "per_relation": {key: dict(value) for key, value in sorted(by_relation.items())},
                         "closure_equals_r25_consumed_semantics": ids_equal,
                         "closure_equals_r25_raw_metadata": raw_metadata_equal,
                         "raw_metadata_note": "base/closure bridge rows retain bookkeeping defaults (zero); raw ANN fields are not consumed by EdgeExample training",
                         "base_query_order_sha256": stable_sha([row["query_id"] for row in base[:first]]),
                         "closure_query_order_sha256": stable_sha([row["query_id"] for row in closure[:first]]),
                         "r25_query_order_sha256": stable_sha([row["query_id"] for row in modern[:first]])},
        }
    return result


def compare_modern_reference() -> dict[str, Any]:
    """Audit the external R25/R26 endpoint used as ``B-modern``.

    B-modern is not silently relabelled as a bridge-trained checkpoint: its
    R25 C1 endpoint and R26 O-NATIVE C2 endpoint are retained as external
    receipts, with the exact graph/order/cache identities recorded here.
    """
    paths = _input_paths()
    required_names = (
        "modern_current_c1_13", "modern_current_c1_29", "modern_current_c2_13", "modern_current_c2_29",
        "modern_current_c2_receipt13", "modern_current_c2_receipt29", "modern_current_c2_order",
        "modern_current_c2_graph", "modern_current_native_cache",
    )
    missing = [name for name in required_names if not paths[name].is_file()]
    if missing:
        return {"status": "missing", "missing": missing}
    receipts = {}
    for seed in SEEDS:
        receipt_path = paths[f"modern_current_c2_receipt{seed}"]
        receipt = read_json(receipt_path)
        signature = receipt.get("signature", {})
        receipts[str(seed)] = {
            "c1_parent": record(paths[f"modern_current_c1_{seed}"]),
            "c2_checkpoint": record(paths[f"modern_current_c2_{seed}"]),
            "c2_receipt": record(receipt_path),
            "execution_status": receipt.get("execution_status"),
            "scientific_validity": receipt.get("scientific_validity"),
            "optimizer_initial_state": receipt.get("optimizer_initial_state"),
            "optimizer_updates": receipt.get("optimizer_updates"),
            "coverage_lists": receipt.get("coverage_lists"),
            "signature_graph_sha256": signature.get("graph_sha256"),
            "signature_order_sha256": signature.get("order_sha256"),
            "signature_native_cache": signature.get("teacher_native_cache"),
        }
    bridge_b7 = {
        "c2_graph": record(R24 / "path_pool/common_seed13.jsonl"),
        "native_teacher_cache": record(R25 / "common/teacher_native_path_cache.jsonl.gz"),
        "historical_order": record(R13 / "taskD_witness_supervision/schedule_order.json"),
    }
    current = {
        "c2_graph": record(paths["modern_current_c2_graph"]),
        "native_teacher_cache": record(paths["modern_current_native_cache"]),
        "modern_order": record(paths["modern_current_c2_order"]),
    }
    historical_graph_ids = [row["query_id"] for row in rows(R12 / "taskC_training/c2_candidates_seed13/path_hard.jsonl")]
    historical_order_indices = read_json(R13 / "taskD_witness_supervision/schedule_order.json")["indices"]
    historical_order_ids = [historical_graph_ids[index] for index in historical_order_indices]
    modern_order_ids = [row["query_id"] for row in rows(paths["modern_current_c2_order"])]
    order_semantically_equal = historical_order_ids == modern_order_ids
    graph_sha_matches = all(value["signature_graph_sha256"] == current["c2_graph"]["sha256"] for value in receipts.values())
    order_sha_matches = all(value["signature_order_sha256"] == current["modern_order"]["sha256"] for value in receipts.values())
    native_cache_matches = all(
        isinstance(value["signature_native_cache"], dict)
        and value["signature_native_cache"].get("sha256") == current["native_teacher_cache"]["sha256"]
        for value in receipts.values()
    )
    evaluation_comparison = {}
    bridge_eval_root = OUT / "evaluation/rankings"
    r26_eval_root = R26 / "rankings"
    for seed in SEEDS:
        bridge_label = "B-modern" if seed == 13 else "B-modern_seed29"
        bridge_metrics_path = bridge_eval_root / bridge_label / "metrics.json"
        r26_metrics_path = r26_eval_root / f"R26-O-NATIVE/seed{seed}/step178/metrics.json"
        if not bridge_metrics_path.is_file() or not r26_metrics_path.is_file():
            evaluation_comparison[str(seed)] = {"status": "missing"}
            continue
        bridge_metrics, r26_metrics = read_json(bridge_metrics_path), read_json(r26_metrics_path)
        comparisons = {}
        for kind in ("overall", "implicit", "explicit"):
            for metric in ("D100_EXACT", "E_ONLY", "U", "QT_OVER_U"):
                for key in ("recall@10", "recall@20", "recall@50", "raw_recall"):
                    left = bridge_metrics[kind].get(metric, {}).get(key)
                    right = r26_metrics[kind].get(metric, {}).get(key)
                    if left is not None and right is not None:
                        comparisons[f"{kind}/{metric}/{key}"] = float(left) - float(right)
        evaluation_comparison[str(seed)] = {
            "status": "complete", "bridge_metrics": record(bridge_metrics_path),
            "r26_metrics": record(r26_metrics_path),
            "max_absolute_metric_difference": max((abs(value) for value in comparisons.values()), default=None),
            "metric_differences": comparisons,
            "interpretation": "same checkpoint/protocol; own ANN/evidence retrieval can differ because the index was rebuilt independently" if comparisons else None,
        }
    return {
        "status": "complete",
        "endpoint": "R25 C1 step659 + R26 O-NATIVE C2 step178",
        "seed_receipts": receipts,
        "bridge_B7_inputs": bridge_b7,
        "current_inputs": current,
        "identity_checks": {
            "current_graph_matches_receipts": graph_sha_matches,
            "current_order_matches_receipts": order_sha_matches,
            "current_native_cache_matches_receipts": native_cache_matches,
            "B7_graph_matches_current_graph": bridge_b7["c2_graph"]["sha256"] == current["c2_graph"]["sha256"],
            "B7_native_cache_matches_current_cache": bridge_b7["native_teacher_cache"]["sha256"] == current["native_teacher_cache"]["sha256"],
            "B7_order_is_historical_not_modern": bridge_b7["historical_order"]["sha256"] != current["modern_order"]["sha256"],
            "B7_order_semantically_equals_current_order": order_semantically_equal,
        },
        "evaluation_comparison": evaluation_comparison,
        "interpretation": "B-modern is the pre-existing R25/R26 endpoint; B7 uses the historical order manifest but its semantic query sequence is compared explicitly to the current R26 order. The bridge and current endpoint share the modern path graph and native path Teacher cache.",
    }


def audit_bridge_c2_provenance() -> dict[str, Any]:
    """Verify the graph actually consumed by B5/B6/B7, not only receipt labels."""
    historical_path = R12 / "taskC_training/c2_candidates_seed13/path_hard.jsonl"
    modern_path = R24 / "path_pool/common_seed13.jsonl"
    witness_path = R12 / "taskA_correctness/supervision/target_lists.train_fit.jsonl"
    historical = {row["query_id"]: row for row in rows(historical_path)}
    modern = {row["query_id"]: row for row in rows(modern_path)}
    witnesses = {row["query_id"]: row.get("positive_evidence_by_target", {}) for row in rows(witness_path)}
    result: dict[str, Any] = {"status": "complete", "stages": {}}
    for stage in ("B5", "B6", "B7"):
        for seed in SEEDS:
            consumed_path = OUT / f"training/{stage}/seed{seed}/C2/consumed_order.jsonl.gz"
            if not consumed_path.is_file():
                result["stages"][f"{stage}/seed{seed}"] = {"status": "missing"}
                continue
            observed = [example for batch in rows(consumed_path) for example in batch.get("examples", ())]
            candidate_order_equal_hist = 0
            candidate_order_equal_modern = 0
            evidence_equal_full = 0
            evidence_equal_first8 = 0
            targets = truncated_targets = removed_paths = removed_known_witnesses = 0
            teacher_mode_counts = Counter()
            teacher_cache_pairs = []
            modern_cache = {}
            if stage == "B7":
                cache_path = R25 / "common/teacher_native_path_cache.jsonl.gz"
                modern_cache = {row["query_id"]: row for row in rows(cache_path)}
            for example in observed:
                query_id = example["query_id"]
                old = historical[query_id]
                new = modern[query_id]
                old_candidates = old.get("candidates", ())
                new_candidates = new.get("candidates", ())
                observed_candidates = example.get("candidates", ())
                old_ids = [candidate["target_id"] for candidate in old_candidates]
                new_ids = [candidate["target_id"] for candidate in new_candidates]
                observed_ids = [candidate["target_id"] for candidate in observed_candidates]
                candidate_order_equal_hist += observed_ids == old_ids
                candidate_order_equal_modern += observed_ids == new_ids
                for observed_candidate in observed_candidates:
                    target_id = observed_candidate["target_id"]
                    old_candidate = next(candidate for candidate in old_candidates if candidate["target_id"] == target_id)
                    modern_candidate = next(candidate for candidate in new_candidates if candidate["target_id"] == target_id)
                    observed_paths = list(observed_candidate.get("evidence_ids", ()))
                    old_paths = list(old_candidate.get("evidence_ids", ()))
                    modern_paths = list(modern_candidate.get("evidence_ids", ()))
                    targets += 1
                    evidence_equal_full += observed_paths == old_paths
                    evidence_equal_first8 += observed_paths == old_paths[:8]
                    truncated_targets += len(old_paths) > 8 and observed_paths == old_paths[:8]
                    removed = set(old_paths) - set(observed_paths)
                    removed_paths += len(removed)
                    removed_known_witnesses += len(removed & set(witnesses.get(query_id, {}).get(target_id, ())))
                teacher_mode_counts[example.get("teacher_logit_mode")] += 1
                if stage == "B7":
                    cache_row = modern_cache[query_id]
                    cache_scores = dict(zip(cache_row["candidate_ids"], cache_row["evidence_logits"]))
                    observed_scores = dict(zip(observed_ids, example.get("teacher_evidence_logits", ())))
                    teacher_cache_pairs.extend((float(observed_scores[target]), float(cache_scores[target]))
                                               for target in observed_ids if target in cache_scores and target in observed_scores)
            result["stages"][f"{stage}/seed{seed}"] = {
                "status": "complete", "lists": len(observed), "targets": targets,
                "candidate_order_equal_historical": candidate_order_equal_hist,
                "candidate_order_equal_modern": candidate_order_equal_modern,
                "evidence_equal_historical_full": evidence_equal_full,
                "evidence_equal_historical_first8": evidence_equal_first8,
                "truncated_targets": truncated_targets, "removed_paths": removed_paths,
                "removed_known_witnesses": removed_known_witnesses,
                "teacher_logit_modes": dict(teacher_mode_counts),
                "actual_graph_source": str(historical_path.resolve()),
                "receipt_graph_label_is_modern_for_first8": stage in ("B6", "B7"),
                "B7_native_cache_score_pairs": len(teacher_cache_pairs),
                "B7_native_cache_max_abs_difference": max((abs(left - right) for left, right in teacher_cache_pairs), default=None),
            }
    result["interpretation"] = "B6/B7 consumed the historical path_hard candidate graph with evidence_ids truncated to first8; B7 additionally consumed the R25 native path Teacher cache. The modern R24 graph label in the legacy B6/B7 receipts is therefore a declared-input label, not the observed candidate source."
    return result


def compare_r26_order_control() -> dict[str, Any]:
    """Reuse the pre-registered R26 order control instead of retraining B8."""
    path = R26 / "statistics/paired_source_bootstrap.jsonl"
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    selected = [row for row in rows(path) if row.get("comparison") == "order_native" and row.get("kind") == "overall"]
    return {
        "status": "complete", "source": record(path), "comparison": "R26-O-NATIVE vs R25-B13-FULL external order control",
        "metrics": {row["metric"]: {key: row.get(key) for key in ("mean_delta", "bootstrap_95ci", "wins", "losses", "ties", "replicates", "seed")}
                    for row in selected},
        "interpretation": "external control only; it is not a same-parent B8 causal retrain and therefore B8 remains conditional/not_triggered",
    }


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2 + 1
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _corr(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    denominator = math.sqrt(sum((a - mean_left) ** 2 for a in left) * sum((b - mean_right) ** 2 for b in right))
    if not denominator:
        return None
    # Floating-point accumulation can produce a value a few ulps outside the
    # mathematical correlation interval when the two vectors are numerically
    # identical.  Keep the receipt interpretable as a correlation coefficient.
    return max(-1.0, min(1.0, numerator / denominator))


def _logit_stats(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    left = [a for a, _ in pairs]
    right = [b for _, b in pairs]
    if not pairs:
        return {"pairs": 0, "pearson": None, "spearman": None, "mean_absolute_difference": None, "max_absolute_difference": None}
    difference = [abs(a - b) for a, b in pairs]
    return {
        "pairs": len(pairs),
        "pearson": _corr(left, right),
        "spearman": _corr(_rank(left), _rank(right)),
        "mean_absolute_difference": sum(difference) / len(difference),
        "max_absolute_difference": max(difference),
        "left_mean": sum(left) / len(left),
        "right_mean": sum(right) / len(right),
    }


def compare_c2_graphs(historical: Path, modern: Path, modern_teacher: Path) -> dict[str, Any]:
    old = {row["query_id"]: row for row in rows(historical)}
    new = {row["query_id"]: row for row in rows(modern)}
    teacher = {row["query_id"]: row for row in rows(modern_teacher)}
    witness_path = R12 / "taskA_correctness/supervision/target_lists.train_fit.jsonl"
    witnesses = {row["query_id"]: row.get("positive_evidence_by_target", {}) for row in rows(witness_path)}
    bag = Counter()
    logit_pairs: dict[str, list[tuple[float, float]]] = {"direct": [], "evidence": []}
    positive_logit_pairs: dict[str, list[tuple[float, float]]] = {"direct": [], "evidence": []}
    margin_pairs: dict[str, list[tuple[float, float]]] = {"direct": [], "evidence": []}
    aligned_queries = sorted(set(old) & set(new) & set(teacher))
    for query_id in aligned_queries:
        old_candidates = {c["target_id"]: c for c in old[query_id].get("candidates", ())}
        new_candidates = {c["target_id"]: c for c in new[query_id].get("candidates", ())}
        teacher_positions = {target: index for index, target in enumerate(teacher[query_id].get("candidate_ids", ()))}
        positives = set(old[query_id].get("positive_target_ids", ()))
        query_scores = {
            channel: {"old_positive": [], "modern_positive": [], "old_negative": [], "modern_negative": []}
            for channel in ("direct", "evidence")
        }
        for target_id, old_candidate in old_candidates.items():
            if target_id not in new_candidates or target_id not in teacher_positions:
                bag["target_missing_in_modern"] += 1
                continue
            new_paths = list(new_candidates[target_id].get("evidence_ids", ()))
            old_paths = list(old_candidate.get("evidence_ids", ()))
            lost = set(old_paths) - set(new_paths)
            bag["targets"] += 1
            bag["positive_targets"] += target_id in positives
            bag["targets_with_removed_paths"] += bool(lost)
            bag["removed_paths"] += len(lost)
            bag["paths_beyond_8"] += max(0, len(old_paths) - 8)
            bag["paths_removed_beyond_first8"] += len(set(old_paths[8:]) - set(new_paths))
            bag["first8_equal"] += new_paths == old_paths[:8]
            bag["removed_known_witnesses"] += len(lost & set(witnesses.get(query_id, {}).get(target_id, ())))
            pos = teacher_positions[target_id]
            for channel, old_key, new_key in (("direct", "teacher_direct_logits", "direct_logits"), ("evidence", "teacher_evidence_logits", "evidence_logits")):
                old_values = old[query_id].get(old_key, ())
                new_values = teacher[query_id].get(new_key, ())
                old_value = old_values[ list(old_candidates).index(target_id) ] if target_id in old_candidates and list(old_candidates).index(target_id) < len(old_values) else None
                new_value = new_values[pos] if pos < len(new_values) else None
                if old_value is not None and new_value is not None:
                    logit_pairs[channel].append((float(old_value), float(new_value)))
                    if target_id in positives:
                        positive_logit_pairs[channel].append((float(old_value), float(new_value)))
                        query_scores[channel]["old_positive"].append(float(old_value))
                        query_scores[channel]["modern_positive"].append(float(new_value))
                    else:
                        query_scores[channel]["old_negative"].append(float(old_value))
                        query_scores[channel]["modern_negative"].append(float(new_value))
        for channel, values in query_scores.items():
            if values["old_positive"] and values["old_negative"]:
                old_margin = (sum(values["old_positive"]) / len(values["old_positive"])) - (sum(values["old_negative"]) / len(values["old_negative"]))
                modern_margin = (sum(values["modern_positive"]) / len(values["modern_positive"])) - (sum(values["modern_negative"]) / len(values["modern_negative"]))
                margin_pairs[channel].append((old_margin, modern_margin))
        for target_id, candidate in new_candidates.items():
            if len(candidate.get("evidence_ids", ())) > 8:
                bag["modern_targets_over_8"] += 1
    return {
        "aligned_queries": len(aligned_queries),
        "bag_differences": dict(bag),
        "teacher_logit_differences": {channel: _logit_stats(pairs) for channel, pairs in logit_pairs.items()},
        "positive_teacher_logit_differences": {channel: _logit_stats(pairs) for channel, pairs in positive_logit_pairs.items()},
        "positive_negative_margin_differences": {channel: _logit_stats(pairs) for channel, pairs in margin_pairs.items()},
        "comparison_scope": "historical full path graph aligned to modern graph and modern native path cache; known witnesses read from historical train_fit target lists; descriptive only",
    }


def compare_orders(historical_order: Path, modern_order: Path, graph: Path) -> dict[str, Any]:
    historical_graph = list(rows(graph))
    order = read_json(historical_order)["indices"]
    historical_ids = [historical_graph[index]["query_id"] for index in order]
    modern_ids = [row["query_id"] for row in rows(modern_order)]
    same = sum(left == right for left, right in zip(historical_ids, modern_ids))
    return {
        "historical_sha256": stable_sha(historical_ids),
        "modern_sha256": stable_sha(modern_ids),
        "equal": historical_ids == modern_ids,
        "same_positions": same,
        "historical_count": len(historical_ids),
        "modern_count": len(modern_ids),
        "interpretation": "R26 order result is cited unless B0-B7 leaves unexplained degradation",
    }


def _semantic_consumption(path: Path, stage: str) -> list[dict[str, Any]]:
    """Read only input-bearing fields from a consumed bridge trace."""
    result = []
    for batch in rows(path):
        examples = []
        for example in batch.get("examples", ()):
            if stage == "C1":
                keys = ("query_id", "candidate_ids", "positive_index", "source_type", "destination_type",
                        "positive_ids", "confirmed_labels", "teacher_logits", "teacher_checkpoint_sha256",
                        "teacher_logit_mode", "teacher_ensemble_alpha")
                values = {key: example.get(key) for key in keys}
            else:
                candidates = [{"target_id": value.get("target_id"), "evidence_ids": value.get("evidence_ids", ())}
                              for value in example.get("candidates", ())]
                keys = ("query_id", "direct_positive_index", "evidence_positive_index", "positive_target_ids",
                        "teacher_direct_logits", "teacher_evidence_logits", "teacher_checkpoint_sha256",
                        "teacher_logit_mode", "teacher_ensemble_alpha")
                values = {key: example.get(key) for key in keys}
                values["candidates"] = candidates
            examples.append(values)
        result.append({"step": batch.get("step"), "examples": examples})
    return result


def verify_b1_consumed_inputs() -> dict[str, Any]:
    """Verify B1 changed only initialization, not consumed supervision inputs."""
    historical_c1 = HIST / "historical_replay/C1/seed13/consumed_batches.jsonl.gz"
    historical_c2 = HIST / "historical_replay/C2/seed13/consumed_order.jsonl.gz"
    checks = []
    for seed in SEEDS:
        c1 = OUT / f"training/B1/seed{seed}/C1/consumed_batches.jsonl.gz"
        c2 = OUT / f"training/B1/seed{seed}/C2/consumed_order.jsonl.gz"
        if not c1.is_file() or not c2.is_file():
            checks.append({"seed": seed, "status": "missing"})
            continue
        old_c1, new_c1 = _semantic_consumption(historical_c1, "C1"), _semantic_consumption(c1, "C1")
        old_c2, new_c2 = _semantic_consumption(historical_c2, "C2"), _semantic_consumption(c2, "C2")
        checks.append({"seed": seed, "status": "equal" if old_c1 == new_c1 and old_c2 == new_c2 else "mismatch",
                       "C1_batches": len(new_c1), "C1_lists": sum(len(x["examples"]) for x in new_c1),
                       "C2_batches": len(new_c2), "C2_lists": sum(len(x["examples"]) for x in new_c2),
                       "C1_semantic_sha256": stable_sha(new_c1), "C2_semantic_sha256": stable_sha(new_c2)})
    return {"status": "verified" if checks and all(c["status"] == "equal" for c in checks) else "incomplete_or_mismatch",
            "checks": checks,
            "scope": "candidate IDs/order, positive masks, Teacher tensors, C2 graph/order; model state intentionally excluded"}


def verify_checkpoint_lineage() -> dict[str, Any]:
    """Check the parent hashes and fresh/continued semantics of the cliff path."""
    checks = []
    for seed in SEEDS:
        def receipt(stage: str, component: str) -> dict[str, Any]:
            return read_json(OUT / f"training/{stage}/seed{seed}/{component}/EXECUTION.json")
        b4_c1_path = OUT / f"training/B4/seed{seed}/C1/checkpoints/step_000356.pt"
        b5_c1_path = OUT / f"training/B5/seed{seed}/C1/checkpoints/step_000659.pt"
        b4_c1_step0 = OUT / f"training/B4/seed{seed}/C1/checkpoints/step_000000.pt"
        b4_c1_step178 = OUT / f"training/B4/seed{seed}/C1/checkpoints/step_000178.pt"
        b5_c1_step500 = OUT / f"training/B5/seed{seed}/C1/checkpoints/step_000500.pt"
        b4_c1, b5_c1 = receipt("B4", "C1"), receipt("B5", "C1")
        b5_c2, b6_c2, b7_c2 = receipt("B5", "C2"), receipt("B6", "C2"), receipt("B7", "C2")
        b5_consumed = OUT / f"training/B5/seed{seed}/C1/consumed_batches.jsonl.gz"
        b5_steps = [batch.get("step") for batch in rows(b5_consumed)]
        expected_b5_steps = list(range(357, 660))
        checks.append({
            "seed": seed,
            "B5_parent_is_B4_step356": b5_c1.get("parent", {}).get("sha256") == sha256(b4_c1_path),
            "B5_optimizer_continued": b5_c1.get("optimizer_initial_state") == "continued" and b5_c1.get("start_step") == 356,
            "B5_consumed_steps_357_to_659": b5_steps == expected_b5_steps,
            "B5_checkpoint_659_matches_receipt": b5_c1.get("checkpoints", {}).get("659", {}).get("checkpoint", {}).get("sha256") == sha256(b5_c1_path),
            "B5_trajectory_anchors_0_178_available": b4_c1_step0.is_file() and b4_c1_step178.is_file(),
            "B5_trajectory_checkpoints_356_500_659_available": b4_c1_path.is_file() and b5_c1_step500.is_file() and b5_c1_path.is_file(),
            "B5_trajectory_manifest": {
                "0": {"path": str(b4_c1_step0.resolve()), "sha256": sha256(b4_c1_step0), "origin": "B4 C1 parent anchor"},
                "178": {"path": str(b4_c1_step178.resolve()), "sha256": sha256(b4_c1_step178), "origin": "B4 C1 parent anchor"},
                "356": {"path": str(b4_c1_path.resolve()), "sha256": sha256(b4_c1_path), "origin": "B4 C1 continuation boundary"},
                "500": {"path": str(b5_c1_step500.resolve()), "sha256": sha256(b5_c1_step500), "origin": "B5 continued C1 run"},
                "659": {"path": str(b5_c1_path.resolve()), "sha256": sha256(b5_c1_path), "origin": "B5 continued C1 run"},
            },
            "B5_C2_parent_is_C1_step659": b5_c2.get("parent", {}).get("sha256") == sha256(b5_c1_path),
            "B6_C2_parent_is_inherited_C1_659": b6_c2.get("parent", {}).get("sha256") == sha256(OUT / f"training/B6/seed{seed}/C1/checkpoints/step_000659.pt"),
            "B7_C2_parent_is_inherited_C1_659": b7_c2.get("parent", {}).get("sha256") == sha256(OUT / f"training/B7/seed{seed}/C1/checkpoints/step_000659.pt"),
            "B6_C1_inherits_B5": read_json(OUT / f"training/B6/seed{seed}/C1/EXECUTION.json").get("parent", {}).get("sha256") == sha256(b5_c1_path),
            "B7_C1_inherits_B5": read_json(OUT / f"training/B7/seed{seed}/C1/EXECUTION.json").get("parent", {}).get("sha256") == sha256(b5_c1_path),
        })
    fields = [key for key in checks[0] if key != "seed"] if checks else []
    return {"status": "verified" if checks and all(all(row[field] for field in fields) for row in checks) else "mismatch", "checks": checks,
            "scope": "B4 step356 -> one continued B5 C1 run through659 -> B5/B6/B7 C2 parent hashes"}


def resource_status() -> dict[str, Any]:
    try:
        proc = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader"], capture_output=True, text=True, check=True)
        gpus = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        try:
            import torch
            cuda_available = bool(torch.cuda.is_available())
        except Exception as exc:
            cuda_available = False
            torch_error = repr(exc)
        payload = {"status": "available" if gpus and cuda_available else "blocked_resource", "gpus": gpus,
                   "torch_cuda_available": cuda_available}
        if not cuda_available and "torch_error" in locals():
            payload["torch_error"] = torch_error
        if gpus and not cuda_available:
            payload["reason"] = "nvidia-smi lists devices but PyTorch cannot initialize CUDA"
        return payload
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"status": "blocked_resource", "gpus": [], "error": str(exc)}


def bridge_lineage(inputs: dict[str, Any], resources: dict[str, Any]) -> dict[str, Any]:
    b1_receipts = [OUT / "training/B1" / f"seed{seed}" / stage / "EXECUTION.json" for seed in SEEDS for stage in ("C1", "C2")]
    b1_complete = all(path.is_file() and json.loads(path.read_text()).get("status") == "completed" for path in b1_receipts)
    def stage_status(stage: str) -> str:
        receipts = [OUT / "training" / stage / f"seed{seed}" / component / "EXECUTION.json"
                    for seed in SEEDS for component in ("C1", "C2")]
        present = [path for path in receipts if path.is_file()]
        if not present:
            return "planned"
        if all(json.loads(path.read_text()).get("status") == "completed" for path in receipts if path.is_file()) and len(present) == len(receipts):
            return "completed"
        return "in_progress"
    def c1_node_status(stage: str, checkpoint_step: int) -> str:
        """Status for a C1-only trajectory node inside the B5 continuation."""
        statuses = []
        for seed in SEEDS:
            receipt = OUT / "training" / stage / f"seed{seed}" / "C1" / "EXECUTION.json"
            checkpoint = OUT / "training" / stage / f"seed{seed}" / "C1" / "checkpoints" / f"step_{checkpoint_step:06d}.pt"
            statuses.append(receipt.is_file() and json.loads(receipt.read_text()).get("status") == "completed" and checkpoint.is_file())
        return "completed" if all(statuses) else ("in_progress" if any(statuses) else "planned")
    modern_reference_complete = all(
        path.is_file() for path in (
            R25 / "training/C1/seed13/checkpoints/step_000659.pt",
            R25 / "training/C1/seed29/checkpoints/step_000659.pt",
            R26 / "training/O-NATIVE/seed13/checkpoints/step_000178.pt",
            R26 / "training/O-NATIVE/seed29/checkpoints/step_000178.pt",
            R26 / "training/O-NATIVE/seed13/C2_COMPLETION_RECEIPT.json",
            R26 / "training/O-NATIVE/seed29/C2_COMPLETION_RECEIPT.json",
        )
    )
    stages = [
        ("B0", "historical exact", "R27 C1/C2 replay checkpoints", "ready"),
        ("B1", "modern initialization", "B0 schedule/logits/graph/order; only init changes; seed13+29 receipts", "completed" if b1_complete else "in_progress"),
        ("B2", "modern C1 schedule, closure off, first356", "one modern base schedule; closure off", stage_status("B2")),
        ("B3", "positive closure on", "B2 base lists; closure insertion only", stage_status("B3")),
        ("B4", "modern C1 Teacher logits", "B3 lists and 356 updates", stage_status("B4")),
        ("B5-356", "modern C1 at356", "B4 trajectory node; C1-only diagnostic checkpoint", c1_node_status("B5", 356)),
        ("B5-500", "modern C1 at500", "B4 same run continuation; C1-only diagnostic checkpoint", c1_node_status("B5", 500)),
        ("B5-659", "only more C1 training", "B4 same run continued to659; C1-only diagnostic checkpoint", c1_node_status("B5", 659)),
        ("B6", "Student first8 paths; historical full-bag Teacher", "B5-659 parent", stage_status("B6")),
        ("B7", "modern Teacher logits on first8", "B6 parent", stage_status("B7")),
        ("B8", "modern consumed order (conditional)", "only if B0-B7 leaves unexplained degradation", "not_triggered"),
        ("B-modern", "current modern pipeline equivalent", "external R25 C1 step659 + R26 O-NATIVE C2 step178 receipt", "completed" if modern_reference_complete else "planned"),
    ]
    all_bridge_stages_complete = all(stage_status(stage) == "completed" for stage in ("B2", "B3", "B4", "B5", "B6", "B7"))
    training = "completed" if all_bridge_stages_complete else (
        "ready_to_run" if resources["status"] == "available" else "blocked_resource_after_partial_training")
    completed_jobs = []
    for stage in ("B1", "B2", "B3", "B4", "B5", "B6", "B7"):
        for seed in SEEDS:
            for component in ("C1", "C2"):
                receipt = OUT / "training" / stage / f"seed{seed}" / component / "EXECUTION.json"
                if receipt.is_file() and json.loads(receipt.read_text()).get("status") == "completed":
                    completed_jobs.append(f"{stage}/{component}/seed{seed}")
    return {
        "format_version": 1,
        "stages": [{"stage": stage, "factor": factor, "parent_and_control": parent, "status": status} for stage, factor, parent, status in stages],
        "training_resource_status": training,
        "completed_jobs": completed_jobs,
        "resource": resources,
        "B0": {"verdict": inputs["resolved"].get("b0_verdict"), "lineage_origin": "R27 historical fresh replay"},
        "rules": ["one historical component per cumulative step", "B2/B3 share exact base lists", "B5 356->659 is one continued run", "B6/B7 consume historical candidate graph with first8 evidence transform", "B-modern is an external R25/R26 endpoint with receipt identities", "B8 is conditional and not retrained by default"],
    }


def audit(root: Path = ROOT) -> dict[str, Any]:
    inputs = resolve_inputs(root)
    paths = _input_paths(root)
    if inputs["required_missing"]:
        write_json(OUT / "BRIDGE_LINEAGE.json", bridge_lineage(inputs, resource_status()))
        return {"status": "blocked_missing_input", "missing": inputs["required_missing"]}
    c1 = compare_c1_schedules(paths["historical_c1_schedule"], paths["modern_c1_seed13"])
    c1_teacher = compare_c1_teacher_scores()
    bridge_schedules = compare_bridge_schedules()
    modern_reference = compare_modern_reference()
    c2_provenance = audit_bridge_c2_provenance()
    order_control = compare_r26_order_control()
    c2 = compare_c2_graphs(paths["historical_c2_graph"], paths["modern_c2_graph"], paths["modern_c2_teacher"])
    order = compare_orders(paths["historical_c2_order"], paths["modern_c2_order"], paths["historical_c2_graph"])
    consumption = verify_b1_consumed_inputs()
    lineage_checks = verify_checkpoint_lineage()
    resources = resource_status()
    payload = {
        "status": "audit_complete",
        "source_audit": record(OUT / "EXECUTION_SOURCE_AUDIT.json"),
        "C1_schedule_comparison": c1,
        "C1_teacher_score_comparison": c1_teacher,
        "B2_B3_schedule_comparison": bridge_schedules,
        "B-modern_reference": modern_reference,
        "C2_bridge_graph_provenance": c2_provenance,
        "B8_external_order_control": order_control,
        "C2_graph_and_teacher_comparison": c2,
        "C2_order_comparison": order,
        "B1_consumption_verification": consumption,
        "checkpoint_lineage_verification": lineage_checks,
        "lineage": bridge_lineage(inputs, resources),
        "causal_claims": "none; these are recipe/graph diagnostics until bridge checkpoints are trained and evaluated",
        "runtime": {"python": platform.python_version(), "torch": _torch_version(), "resource": resources},
    }
    write_json(OUT / "BRIDGE_AUDIT.json", payload)
    write_json(OUT / "BRIDGE_LINEAGE.json", payload["lineage"])
    expected_jobs = [f"{stage}/{component}/seed{seed}"
                     for stage in ("B2", "B3", "B4", "B5", "B6", "B7")
                     for component in ("C1", "C2") for seed in SEEDS]
    completed_set = set(payload["lineage"]["completed_jobs"])
    execution_status = "completed" if all(job in completed_set for job in expected_jobs) else (
        "blocked_resource" if resources["status"] != "available" else "in_progress")
    write_json(OUT / "TRAINING_EXECUTION.json", {
        "status": execution_status,
        "resource": resources,
        "completed_jobs": payload["lineage"]["completed_jobs"],
        "planned_jobs": [{"job": job, "status": "blocked_resource" if resources["status"] != "available" else "pending"}
                         for job in expected_jobs if job not in completed_set],
        "source_audit": record(OUT / "EXECUTION_SOURCE_AUDIT.json"),
    })
    return payload


def _torch_version() -> str | None:
    try:
        import torch
        return str(torch.__version__)
    except ImportError:
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "resource"), nargs="?", default="audit")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = resource_status() if args.command == "resource" else audit()
    print(json.dumps(result, ensure_ascii=False, indent=2))
