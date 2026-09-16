"""Run the preregistered R30 text-to-table candidate diagnostic.

The unit of comparison is an actual text evidence source consumed in Bridge
C1 batches 357--659.  Natural candidates come from the same-seed C1@356
Student's exact whole-table ranking and exclude every train-known positive for
that source.  Repeated schedule occurrences are retained, then averaged by
source for the G-ET bootstrap.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import load_corpus_ids
from prepare_stage1_r27 import ROOT
from run_stage1_bridge import sha256, write_json


OUT = ROOT / "work/stage1_r30_c1_et_20260916"
DIAGNOSTICS = OUT / "diagnostics_repaired"
FEATURES = ROOT / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b"
CORPUS = ROOT / "work/stage1_optimization_r10_20260907/stage1_data/stage1_corpus.jsonl"
REGISTRY = ROOT / "work/stage1_optimization_r12_20260908/taskA_correctness/supervision/edge_lists.train_fit.jsonl"
T0TRAIN = ROOT / "work/stage1_optimization_r22_20260911/manifests/full_natural.jsonl"
BRIDGE = ROOT / "work/stage1_bridge_20260915"
BOOTSTRAP_SEED = 260914
BOOTSTRAP_REPLICATES = 10_000


def read_rows(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_rows(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    temporary = path.with_suffix(path.suffix + ".tmp")
    with opener(temporary, "wt", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def load_occurrences(seed: int) -> list[dict[str, Any]]:
    path = BRIDGE / f"schedules/seed{seed}_steps659/closure_full.jsonl.gz"
    result = []
    for batch in read_rows(path):
        step = int(batch["step"])
        if step < 357:
            continue
        for position, row in enumerate(batch["examples"]):
            if row["source_type"] != "text" or row["destination_type"] != "table":
                continue
            positives = list(dict.fromkeys([
                *(str(value) for value in row.get("positive_ids", ())),
                str(row["positive_id"]),
            ]))
            result.append({
                "seed": seed,
                "occurrence_id": f"{step}:{position}",
                "step": step,
                "position": position,
                "source_id": str(row["query_id"]),
                "source_type": "text",
                "destination_type": "table",
                "candidate_ids": [str(value) for value in row["candidate_ids"]],
                "positive_ids": positives,
            })
    return result


def load_global_positives() -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for row in read_rows(REGISTRY):
        if row["source_type"] != "text" or row["destination_type"] != "table":
            continue
        source_id = str(row["query_id"])
        result[source_id].update(str(value) for value in row.get("positive_ids", ()))
        result[source_id].add(str(row["positive_id"]))
    return result


def load_t0train(source_ids: set[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for row in read_rows(T0TRAIN):
        if (
            row["source_type"] != "text"
            or row["destination_type"] != "table"
            or str(row["query_id"]) not in source_ids
        ):
            continue
        source_id = str(row["query_id"])
        current = result.setdefault(source_id, [])
        current[:] = list(dict.fromkeys([*current, *(str(value) for value in row.get("candidate_ids", ())) ]))
    return result


def matched_metrics(
    positive_ids: list[str],
    unknown_ids: list[str],
    scores: dict[str, float],
) -> dict[str, Any]:
    positives = list(dict.fromkeys(positive_ids))
    unknowns = sorted(set(unknown_ids), key=lambda value: (-scores[value], value))[:32]
    if not positives or not unknowns:
        return {
            "status": "insufficient_unknowns" if positives else "no_positives",
            "positive_count": len(positives),
            "unknown_count": len(unknowns),
            "hard32": unknowns,
            "margin": None,
            "violation": None,
            "positive_rank": None,
            "per_positive": [],
        }
    ranking = sorted([*positives, *unknowns], key=lambda value: (-scores[value], value))
    maximum_unknown = max(scores[value] for value in unknowns)
    per_positive = [
        {
            "positive_id": positive_id,
            "score": scores[positive_id],
            "margin": scores[positive_id] - maximum_unknown,
            "violation": float(maximum_unknown > scores[positive_id]),
            "rank": ranking.index(positive_id) + 1,
        }
        for positive_id in positives
    ]
    return {
        "status": "complete" if len(unknowns) == 32 else "insufficient_unknowns",
        "positive_count": len(positives),
        "unknown_count": len(unknowns),
        "hard32": unknowns,
        "margin": statistics.fmean(row["margin"] for row in per_positive),
        "violation": statistics.fmean(row["violation"] for row in per_positive),
        "positive_rank": statistics.fmean(row["rank"] for row in per_positive),
        "per_positive": per_positive,
    }


def bootstrap(values: list[float], *, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return {"n": 0, "mean": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    means = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPLICATES, 100):
        count = min(100, BOOTSTRAP_REPLICATES - start)
        samples = rng.integers(0, len(array), size=(count, len(array)))
        means[start:start + count] = array[samples].mean(axis=1)
    return {
        "n": len(array),
        "mean": float(array.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "replicates": BOOTSTRAP_REPLICATES,
        "rng_seed": seed,
        "unit": "distinct text evidence source",
    }


def aggregate_sources(occurrences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in occurrences:
        grouped[row["source_id"]].append(row)
    result = []
    for source_id, rows in sorted(grouped.items()):
        item: dict[str, Any] = {
            "seed": rows[0]["seed"],
            "source_id": source_id,
            "occurrences": len(rows),
        }
        for pool in ("native", "natural", "t0train"):
            valid = [row[pool] for row in rows if row.get(pool, {}).get("margin") is not None]
            item[pool] = {
                "status": "complete" if len(valid) == len(rows) else ("not_applicable" if not valid else "partial"),
                "valid_occurrences": len(valid),
                **({
                    key: statistics.fmean(float(value[key]) for value in valid)
                    for key in ("margin", "violation", "positive_rank")
                } if valid else {}),
            }
        if item["native"].get("margin") is not None and item["natural"].get("margin") is not None:
            item["natural_minus_native_margin"] = item["natural"]["margin"] - item["native"]["margin"]
            item["natural_minus_native_violation"] = item["natural"]["violation"] - item["native"]["violation"]
        result.append(item)
    return result


def summarize_seed(seed: int, source_rows: list[dict[str, Any]], occurrence_rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["natural_minus_native_margin"]) for row in source_rows if row.get("natural_minus_native_margin") is not None]
    violation_deltas = [float(row["natural_minus_native_violation"]) for row in source_rows if row.get("natural_minus_native_violation") is not None]
    outside = sum(int(row["natural_outside_native_above_any_positive"]) for row in occurrence_rows)
    return {
        "seed": seed,
        "occurrences": len(occurrence_rows),
        "sources": len(source_rows),
        "source_level_margin_natural_minus_native": bootstrap(deltas),
        "source_level_violation_natural_minus_native": bootstrap(violation_deltas),
        "natural_outside_native_above_any_positive_occurrence_count": outside,
        "mean_direction_lower": bool(deltas and statistics.fmean(deltas) < 0),
    }


def combine_gate() -> dict[str, Any] | None:
    seed_payloads = []
    source_rows = []
    for seed in (13, 29):
        summary_path = DIAGNOSTICS / f"ET_CANDIDATE_SUMMARY_seed{seed}.json"
        rows_path = DIAGNOSTICS / f"et_source_candidate_metrics_seed{seed}.jsonl.gz"
        if not summary_path.is_file() or not rows_path.is_file():
            return None
        seed_payloads.append(json.loads(summary_path.read_text()))
        source_rows.extend(read_rows(rows_path))
    deltas = [float(row["natural_minus_native_margin"]) for row in source_rows if row.get("natural_minus_native_margin") is not None]
    pooled = bootstrap(deltas)
    completeness = all(payload["completeness"]["status"] == "pass" for payload in seed_payloads)
    directions = all(payload["metrics"]["mean_direction_lower"] for payload in seed_payloads)
    outside_count = sum(payload["metrics"]["natural_outside_native_above_any_positive_occurrence_count"] for payload in seed_payloads)
    upper = pooled["ci95"][1]
    passed = completeness and directions and upper is not None and upper < 0 and outside_count > 0
    gate = {
        "gate": "G-ET",
        "status": "pass" if passed else "fail",
        "trigger_F_P_ETNAT_subject_to_G_F": passed,
        "criteria": {
            "type_id_score_and_closure_complete": completeness,
            "both_seed_margin_directions_lower": directions,
            "pooled_source_bootstrap_upper_below_zero": upper is not None and upper < 0,
            "legal_outside_native_competitor_above_positive_exists": outside_count > 0,
        },
        "seed_summaries": [payload["metrics"] for payload in seed_payloads],
        "pooled_source_level_margin_natural_minus_native": pooled,
        "outside_native_above_positive_occurrence_count": outside_count,
        "bootstrap_policy": "average repeated occurrences within source; equal-weight distinct (seed, source) units",
    }
    write_json(DIAGNOSTICS / "G_ET.json", gate)
    write_rows(DIAGNOSTICS / "et_source_candidate_metrics.jsonl.gz", source_rows)
    completeness_rows = [payload["completeness"] for payload in seed_payloads]
    write_json(DIAGNOSTICS / "candidate_type_and_score_completeness.json", {
        "status": "pass" if completeness else "fail",
        "seeds": completeness_rows,
    })
    return gate


@torch.inference_mode()
def run(seed: int, device_name: str, batch_size: int) -> dict[str, Any]:
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.cuda.set_device(device)
    torch.set_num_threads(4)
    occurrences = load_occurrences(seed)
    source_ids = sorted({row["source_id"] for row in occurrences})
    global_positives = load_global_positives()
    t0train = load_t0train(set(source_ids))
    schedule_path = BRIDGE / f"schedules/seed{seed}_steps659/closure_full.jsonl.gz"
    checkpoint = BRIDGE / f"training/B4/seed{seed}/C1/checkpoints/step_000356.pt"

    store = FeatureStore.from_path(FEATURES, cache_size=80_000)
    corpus_ids = load_corpus_ids(CORPUS, store)
    table_ids = corpus_ids["table"]
    table_position = {value: index for index, value in enumerate(table_ids)}
    table_set = set(table_ids)
    text_set = set(corpus_ids["text"])
    model = load_student(checkpoint, device).eval()
    target_blocks = []
    for start in range(0, len(table_ids), 4096):
        raw = torch.stack([store.embedding_features(value).embedding for value in table_ids[start:start + 4096]]).to(device)
        target_blocks.append(model.index_vector(raw, "table").detach())
    target_vectors = torch.cat(target_blocks)
    del target_blocks

    occurrences_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in occurrences:
        occurrences_by_source[row["source_id"]].append(row)
    invalid_sources = sorted(set(source_ids) - text_set)
    invalid_candidates = set()
    closure_mismatches = []
    for row in occurrences:
        invalid_candidates.update(set(row["candidate_ids"]) - table_set)
        expected = global_positives.get(row["source_id"], set())
        if set(row["positive_ids"]) != expected:
            closure_mismatches.append(row["occurrence_id"])
    for values in t0train.values():
        invalid_candidates.update(set(values) - table_set)

    reservoir_rows = []
    score_maps: dict[str, dict[str, float]] = {}
    for start in range(0, len(source_ids), batch_size):
        part = source_ids[start:start + batch_size]
        raw = torch.stack([store.embedding_features(value).embedding for value in part]).to(device)
        queries = model.relation_query(raw, "text", "table", source_role="query")
        matrix = queries @ target_vectors.T
        initial_k = min(512, len(table_ids))
        top_values, top_indices = matrix.topk(initial_k, dim=1)
        for offset, source_id in enumerate(part):
            known = global_positives.get(source_id, set())
            ranked = sorted(
                ((table_ids[int(index)], float(value)) for index, value in zip(top_indices[offset], top_values[offset], strict=True)),
                key=lambda item: (-item[1], item[0]),
            )
            legal = [(candidate_id, score) for candidate_id, score in ranked if candidate_id not in known]
            if len(legal) < 256:
                all_values = matrix[offset].cpu().tolist()
                legal = sorted(
                    ((candidate_id, float(all_values[index])) for index, candidate_id in enumerate(table_ids) if candidate_id not in known),
                    key=lambda item: (-item[1], item[0]),
                )
            legal = legal[:256]
            needed = set(known)
            for row in occurrences_by_source[source_id]:
                needed.update(row["candidate_ids"])
                needed.update(row["positive_ids"])
            needed.update(t0train.get(source_id, ()))
            needed.update(candidate_id for candidate_id, _score in legal)
            score_map = {candidate_id: float(matrix[offset, table_position[candidate_id]].cpu()) for candidate_id in needed}
            score_maps[source_id] = score_map
            reservoir_rows.append({
                "seed": seed,
                "source_id": source_id,
                "source_type": "text",
                "destination_type": "table",
                "global_train_known_positive_ids": sorted(known),
                "exact_top256_legal_unknown": [
                    {"candidate_id": candidate_id, "score": score}
                    for candidate_id, score in legal
                ],
                "complete": len(legal) == 256,
            })
        if start % (batch_size * 20) == 0:
            print(json.dumps({"event": "et_exact_sources", "seed": seed, "completed": min(start + batch_size, len(source_ids)), "total": len(source_ids)}), flush=True)

    reservoir_by_source = {row["source_id"]: row for row in reservoir_rows}
    occurrence_metrics = []
    score_missing = 0
    protection_errors = 0
    for row in occurrences:
        source_id = row["source_id"]
        known = global_positives[source_id]
        scores = score_maps[source_id]
        natural_ids = [item["candidate_id"] for item in reservoir_by_source[source_id]["exact_top256_legal_unknown"]]
        native_unknown = [value for value in row["candidate_ids"] if value not in known]
        natural_unknown = [value for value in natural_ids if value not in known]
        t0_ids = t0train.get(source_id)
        t0_unknown = [] if t0_ids is None else [value for value in t0_ids if value not in known]
        positive_ids = row["positive_ids"]
        needed = set(positive_ids) | set(native_unknown) | set(natural_unknown) | set(t0_unknown)
        score_missing += len(needed - set(scores))
        protection_errors += len((set(native_unknown) | set(natural_unknown) | set(t0_unknown)) & known)
        native = matched_metrics(positive_ids, native_unknown, scores)
        natural = matched_metrics(positive_ids, natural_unknown, scores)
        t0 = {"status": "not_applicable", "margin": None}
        if t0_ids is not None:
            t0 = matched_metrics(positive_ids, t0_unknown, scores)
        native_membership = set(row["candidate_ids"])
        positive_scores = [scores[value] for value in positive_ids]
        outside_above_any = [
            value for value in natural["hard32"]
            if value not in native_membership and scores[value] > min(positive_scores)
        ]
        outside_above_best = [
            value for value in natural["hard32"]
            if value not in native_membership and scores[value] > max(positive_scores)
        ]
        occurrence_metrics.append({
            **{key: row[key] for key in ("seed", "occurrence_id", "step", "position", "source_id")},
            "positive_ids": positive_ids,
            "native_width": len(row["candidate_ids"]),
            "native": native,
            "natural": natural,
            "t0train": t0,
            "hard32_overlap_native_natural": len(set(native["hard32"]) & set(natural["hard32"])),
            "natural_outside_native_above_any_positive": len(outside_above_any),
            "natural_outside_native_above_best_positive": len(outside_above_best),
            "outside_native_above_any_positive_ids": outside_above_any,
        })

    source_metrics = aggregate_sources(occurrence_metrics)
    completeness = {
        "seed": seed,
        "status": "pass" if not invalid_sources and not invalid_candidates and not closure_mismatches and score_missing == 0 and protection_errors == 0 and all(row["complete"] for row in reservoir_rows) else "fail",
        "sources_are_text": not invalid_sources,
        "candidates_are_tables": not invalid_candidates,
        "score_missing": score_missing,
        "closure_mismatch_occurrences": len(closure_mismatches),
        "known_positive_protection_errors": protection_errors,
        "natural_reservoir_shortfalls": sum(not row["complete"] for row in reservoir_rows),
        "invalid_source_count": len(invalid_sources),
        "invalid_candidate_count": len(invalid_candidates),
    }
    metrics = summarize_seed(seed, source_metrics, occurrence_metrics)
    summary = {
        "status": "complete",
        "seed": seed,
        "inputs": {
            "schedule": {"path": str(schedule_path.resolve()), "sha256": sha256(schedule_path)},
            "parent": {"path": str(checkpoint.resolve()), "sha256": sha256(checkpoint)},
            "registry": {"path": str(REGISTRY.resolve()), "sha256": sha256(REGISTRY)},
            "t0train": {"path": str(T0TRAIN.resolve()), "sha256": sha256(T0TRAIN)},
            "features": {"path": str((FEATURES / "manifest.jsonl").resolve()), "sha256": sha256(FEATURES / "manifest.jsonl")},
            "corpus": {"path": str(CORPUS.resolve()), "sha256": sha256(CORPUS)},
        },
        "completeness": completeness,
        "metrics": metrics,
        "t0train_aligned_sources": len(t0train),
        "scoring": "same-seed frozen C1@356 Student raw text->table logit; exact whole table lake",
    }
    write_rows(DIAGNOSTICS / f"natural_text_et_reservoir_seed{seed}.jsonl.gz", reservoir_rows)
    write_rows(DIAGNOSTICS / f"et_candidate_occurrence_metrics_seed{seed}.jsonl.gz", occurrence_metrics)
    write_rows(DIAGNOSTICS / f"et_source_candidate_metrics_seed{seed}.jsonl.gz", source_metrics)
    write_json(DIAGNOSTICS / f"ET_CANDIDATE_SUMMARY_seed{seed}.json", summary)
    gate = combine_gate()
    return {"seed_summary": summary, "combined_gate": gate}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(13, 29), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.device, args.batch_size), ensure_ascii=False))


if __name__ == "__main__":
    main()
