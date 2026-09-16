"""Compute the repaired R30 QE/ET witness diagnostics.

The scorer functions are shared with the original C1 diagnostic, while the
annotation mapping and all aggregations implement the R30 metric contract:
all positive targets, unioned evidence-to-target labels, target/query macro
means, explicit best-witness hits, and classified missing exact ranks.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

from diagnose_c1_evidence_collapse import (
    ann_relation_ranks,
    exact_relation_ranks,
    object_types,
    read_dev_rows,
    read_ids,
    sha256,
)
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from prepare_stage1_r27 import ROOT
from run_stage1_bridge import write_json


OUT = ROOT / "work/stage1_r30_c1_et_20260916"
DIAGNOSTICS = OUT / "diagnostics_repaired"
FEATURES = ROOT / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b"
DEV_QUERIES = OUT / "common/dev_queries.jsonl"
BRIDGE = ROOT / "work/stage1_bridge_20260915"
CUTOFFS = (1, 5, 10, 20, 50, 100)


def rows(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_rows(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def stable_probe(query_ids: list[str], count: int = 128) -> list[str]:
    return sorted(query_ids, key=lambda value: (hashlib.sha256(value.encode()).hexdigest(), value))[:count]


def checkpoint_spec(seed: int, name: str) -> tuple[Path, Path]:
    suffix = "" if seed == 13 else "_seed29"
    if name == "STOP356":
        return (
            BRIDGE / f"training/B5/seed{seed}/C1/checkpoints/step_000356.pt",
            BRIDGE / f"evaluation/indexes/B5_356{suffix}",
        )
    if name == "JOINT500":
        return (
            BRIDGE / f"training/B5/seed{seed}/C1/checkpoints/step_000500.pt",
            BRIDGE / f"evaluation/indexes/B5_500{suffix}",
        )
    if name == "JOINT659":
        return (
            BRIDGE / f"training/B5/seed{seed}/C1/checkpoints/step_000659.pt",
            BRIDGE / f"evaluation/indexes/B5_659{suffix}",
        )
    if name in {"F-P500", "F-P659"}:
        step = int(name.removeprefix("F-P"))
        generator = f"F-P{step}_s{seed}"
        return (
            OUT / f"C1/F-P/seed{seed}/checkpoints/step_{step:06d}.pt",
            OUT / f"indexes/{generator}",
        )
    if name in {"F-P-ETNAT500", "F-P-ETNAT659"}:
        step = int(name.removeprefix("F-P-ETNAT"))
        generator = f"F-P-ETNAT{step}_s{seed}"
        return (
            OUT / f"C1/F-P-ETNAT/seed{seed}/checkpoints/step_{step:06d}.pt",
            OUT / f"indexes/{generator}",
        )
    if name in {"C2-F-P89", "C2-F-P178"}:
        step = int(name.removeprefix("C2-F-P"))
        generator = f"C2-F-P{step}_s{seed}"
        return (
            OUT / f"C2-CHECK/F-P/seed{seed}/checkpoints/step_{step:06d}.pt",
            OUT / f"indexes/{generator}",
        )
    raise ValueError(name)


def annotation_map(
    dev_rows: list[dict[str, Any]], types: dict[str, str]
) -> tuple[
    dict[str, dict[str, dict[str, tuple[str, ...]]]],
    dict[str, set[str]],
]:
    annotations: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {}
    targets_by_evidence: dict[str, set[str]] = defaultdict(set)
    for row in dev_rows:
        query_id = str(row["query_id"])
        by_target = row.get("positive_evidence_by_target", {})
        annotations[query_id] = {}
        for raw_target in row["positive_target_ids"]:
            target_id = str(raw_target)
            evidence = tuple(dict.fromkeys(str(value) for value in by_target.get(target_id, ())))
            annotations[query_id][target_id] = {
                modality: tuple(value for value in evidence if types.get(value) == modality)
                for modality in ("text", "image")
            }
            for evidence_id in evidence:
                targets_by_evidence[evidence_id].add(target_id)
    return annotations, targets_by_evidence


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def relation_metrics(
    dev_rows: list[dict[str, Any]],
    annotations: dict[str, dict[str, dict[str, tuple[str, ...]]]],
    qe_ranks: dict[str, dict[str, int]],
    et_ranks: dict[str, dict[str, int]],
    modality: str,
) -> dict[str, Any]:
    per_query: dict[str, dict[str, float]] = {}
    eligible_targets = 0
    witness_links = 0
    for row in dev_rows:
        query_id = str(row["query_id"])
        target_rows = []
        for raw_target in row["positive_target_ids"]:
            target_id = str(raw_target)
            evidence = annotations[query_id][target_id][modality]
            if not evidence:
                continue
            eligible_targets += 1
            witness_links += len(evidence)
            values: dict[str, float] = {}
            for cutoff in CUTOFFS:
                qe_hits = [float(qe_ranks.get(query_id, {}).get(evidence_id, math.inf) <= cutoff) for evidence_id in evidence]
                et_hits = [float(et_ranks.get(evidence_id, {}).get(target_id, math.inf) <= cutoff) for evidence_id in evidence]
                values[f"qe_witness_recall@{cutoff}"] = sum(qe_hits) / len(qe_hits)
                values[f"et_witness_mean_hit@{cutoff}"] = sum(et_hits) / len(et_hits)
                values[f"et_best_witness_hit@{cutoff}"] = float(any(et_hits))
                values[f"path_exists@{cutoff}"] = float(
                    any(
                        qe_ranks.get(query_id, {}).get(evidence_id, math.inf) <= cutoff
                        and et_ranks.get(evidence_id, {}).get(target_id, math.inf) <= cutoff
                        for evidence_id in evidence
                    )
                )
            target_rows.append(values)
        if target_rows:
            per_query[query_id] = {
                key: sum(target[key] for target in target_rows) / len(target_rows)
                for key in target_rows[0]
            }
    summary = {
        key: sum(values[key] for values in per_query.values()) / len(per_query)
        for key in next(iter(per_query.values()), {})
    }
    return {
        "eligible_queries": len(per_query),
        "eligible_targets": eligible_targets,
        "witness_links": witness_links,
        "summary": summary,
        "per_query": per_query,
        "weighting": "witness mean within (q,t), target mean within q, query macro",
    }


def hub_metrics(top_rows: list[list[str]], fixed_targets: set[str] | None = None) -> dict[str, Any]:
    counts = Counter(value for row in top_rows for value in row)
    slots = sum(len(row) for row in top_rows)
    universe = sorted(set(value for row in top_rows for value in row))
    frequencies = sorted((counts[value] for value in universe))
    weighted = sum((2 * index - len(frequencies) - 1) * value for index, value in enumerate(frequencies, 1))
    return {
        "sources": len(top_rows),
        "slots": slots,
        "unique_ids": len(counts),
        "max_frequency": max(counts.values(), default=0),
        "max_frequency_items": sorted(value for value, count in counts.items() if count == max(counts.values(), default=0))[:20],
        "gini_over_fixed_lake_with_zeros": None,
        "gini_retrieved_universe": weighted / (len(frequencies) * slots) if frequencies and slots else 0.0,
        "fixed_reference_target_slot_share": (
            sum(counts[value] for value in fixed_targets) / slots if fixed_targets is not None and slots else None
        ),
        "top_items": counts.most_common(100),
    }


def build_static_annotations(*, write_artifacts: bool = True) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, dict[str, tuple[str, ...]]]],
    dict[str, set[str]],
    dict[str, str],
]:
    dev_rows = read_dev_rows()
    query_meta = {str(row["query_id"]): row for row in rows(DEV_QUERIES)}
    if set(query_meta) != {str(row["query_id"]) for row in dev_rows}:
        raise ValueError("dev query population differs from target annotations")
    evidence_ids = sorted(
        {
            str(evidence_id)
            for row in dev_rows
            for target_id in row["positive_target_ids"]
            for evidence_id in row.get("positive_evidence_by_target", {}).get(str(target_id), ())
        }
    )
    store = FeatureStore.from_path(FEATURES, cache_size=0)
    types = object_types(store, evidence_ids)
    annotations, targets_by_evidence = annotation_map(dev_rows, types)
    annotated_rows = []
    missing_rows = []
    references: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in dev_rows:
        query_id = str(row["query_id"])
        for raw_target in row["positive_target_ids"]:
            target_id = str(raw_target)
            by_modality = annotations[query_id][target_id]
            all_evidence = tuple(dict.fromkeys((*by_modality["text"], *by_modality["image"])))
            record = {
                "query_id": query_id,
                "target_id": target_id,
                "query_kind": query_meta[query_id]["query_kind"],
                "source_table_id": query_meta[query_id]["source_table_id"],
                "witnesses": {key: list(value) for key, value in by_modality.items()},
                "annotation_status": "present" if all_evidence else "missing",
            }
            annotated_rows.append(record)
            if not all_evidence:
                missing_rows.append({**record, "reason": "witness_annotation_missing", "kept_in_target_recall_denominator": True})
            for evidence_id in all_evidence:
                references[evidence_id].append((query_id, target_id))
    collisions = [
        {
            "evidence_id": evidence_id,
            "modality": types[evidence_id],
            "query_target_references": [{"query_id": q, "target_id": t} for q, t in values],
            "target_union": sorted(targets_by_evidence[evidence_id]),
        }
        for evidence_id, values in sorted(references.items())
        if len(values) > 1 or len(targets_by_evidence[evidence_id]) > 1
    ]
    if write_artifacts:
        write_rows(DIAGNOSTICS / "annotated_witness_population.jsonl.gz", annotated_rows)
        write_rows(DIAGNOSTICS / "shared_evidence_target_collisions.jsonl.gz", collisions)
        write_rows(DIAGNOSTICS / "exact_missing_rank_causes.jsonl.gz", missing_rows)
        write_json(
            DIAGNOSTICS / "ANNOTATION_SUMMARY.json",
            {
                "queries": len(dev_rows),
                "query_target_pairs": len(annotated_rows),
                "annotated_pairs": sum(row["annotation_status"] == "present" for row in annotated_rows),
                "missing_annotation_pairs": len(missing_rows),
                "eligible_queries": {
                    modality: sum(any(annotations[str(row["query_id"])][str(target)][modality] for target in row["positive_target_ids"]) for row in dev_rows)
                    for modality in ("text", "image")
                },
                "shared_evidence_records": len(collisions),
            },
        )
    return dev_rows, annotations, targets_by_evidence, types


@torch.inference_mode()
def run_seed(seed: int, names: list[str], device: torch.device) -> dict[str, Any]:
    started = time.monotonic()
    dev_rows, annotations, targets_by_evidence, types = build_static_annotations(write_artifacts=False)
    query_ids = [str(row["query_id"]) for row in dev_rows]
    probe_ids = stable_probe(query_ids)
    first_checkpoint, first_index = checkpoint_spec(seed, names[0])
    ids = read_ids(first_index)
    lake_sets = {key: set(value) for key, value in ids.items()}
    evidence_ids = sorted(types)
    store = FeatureStore.from_path(FEATURES, cache_size=0)
    preload = [*ids["table"], *ids["text"], *ids["image"], *query_ids, *evidence_ids]
    print(json.dumps({"event": "preload", "seed": seed, "objects": len(set(preload))}), flush=True)
    store.preload_embedding_matrix(preload)
    positive_by_query = {
        modality: {
            query_id: {
                evidence_id
                for target in annotations[query_id].values()
                for evidence_id in target[modality]
            }
            for query_id in query_ids
        }
        for modality in ("text", "image")
    }
    evidence_by_modality = {
        modality: sorted(value for value, object_type in types.items() if object_type == modality)
        for modality in ("text", "image")
    }
    positive_by_evidence = {
        modality: {evidence_id: targets_by_evidence[evidence_id] for evidence_id in evidence_by_modality[modality]}
        for modality in ("text", "image")
    }
    result_path = DIAGNOSTICS / f"RELATION_RESULTS_seed{seed}.json"
    per_path_path = DIAGNOSTICS / f"per_q_t_e_ranks_seed{seed}.jsonl.gz"
    missing_path = DIAGNOSTICS / f"exact_missing_rank_causes_seed{seed}.jsonl.gz"
    selected_names = set(names)
    per_path_rows = [row for row in rows(per_path_path) if row["checkpoint"] not in selected_names] if per_path_path.is_file() else []
    exact_missing = [row for row in rows(missing_path) if row["checkpoint"] not in selected_names] if missing_path.is_file() else []
    results: dict[str, Any] = json.loads(result_path.read_text()) if result_path.is_file() else {
        "seed": seed,
        "probe_query_ids": probe_ids,
        "inputs": {
            "features": {"path": str(FEATURES.resolve()), "sha256": sha256(FEATURES / "manifest.jsonl")},
            "queries": {"path": str(DEV_QUERIES.resolve()), "sha256": sha256(DEV_QUERIES)},
        },
        "checkpoints": {},
    }
    fixed_hub_targets: dict[tuple[str, str, str], set[str]] = {}
    stop = results.get("checkpoints", {}).get("STOP356", {})
    for modality in ("text", "image"):
        for backend in ("exact", "ann"):
            for relation in ("QE", "ET"):
                key = (backend, modality, relation)
                hub = stop.get("hub", {}).get(f"{relation}_{modality}_{backend}", {})
                fixed_hub_targets[key] = {str(item[0]) for item in hub.get("top_items", [])[:100]}
    for name in names:
        checkpoint, index_dir = checkpoint_spec(seed, name)
        if not checkpoint.is_file() or not (index_dir / "manifest.json").is_file():
            raise FileNotFoundError(checkpoint if not checkpoint.is_file() else index_dir / "manifest.json")
        checkpoint_sha = sha256(checkpoint)
        model = load_student(checkpoint, device).eval()
        destination_matrices = {}
        for destination_type in ("table", "text", "image"):
            raw = torch.stack([store.embedding_features(object_id).embedding for object_id in ids[destination_type]]).to(device)
            destination_matrices[destination_type] = model.index_vector(raw, destination_type).detach()
        state: dict[str, Any] = {
            "checkpoint": {"path": str(checkpoint.resolve()), "sha256": checkpoint_sha},
            "index": {"path": str(index_dir.resolve()), "manifest_sha256": sha256(index_dir / "manifest.json")},
            "relations": {},
            "hub": {},
        }
        rank_cache: dict[tuple[str, str], tuple[dict[str, dict[str, int]], dict[str, list[str]]]] = {}
        for modality in ("text", "image"):
            qe_exact, qe_exact_top = exact_relation_ranks(
                model, store, query_ids, "table", ids[modality], modality,
                positive_by_query[modality], device, destination_matrices[modality],
            )
            qe_ann, qe_ann_top = ann_relation_ranks(
                model, store, index_dir, checkpoint_sha, query_ids, modality,
                positive_by_query[modality], device,
            )
            et_sources = evidence_by_modality[modality]
            et_exact, et_exact_top = exact_relation_ranks(
                model, store, et_sources, modality, ids["table"], "table",
                positive_by_evidence[modality], device, destination_matrices["table"],
            )
            et_ann, et_ann_top = ann_relation_ranks(
                model, store, index_dir, checkpoint_sha, et_sources, "table",
                positive_by_evidence[modality], device,
            )
            rank_cache[(modality, "qe_exact")] = (qe_exact, qe_exact_top)
            rank_cache[(modality, "qe_ann")] = (qe_ann, qe_ann_top)
            rank_cache[(modality, "et_exact")] = (et_exact, et_exact_top)
            rank_cache[(modality, "et_ann")] = (et_ann, et_ann_top)
            state["relations"][modality] = {
                backend: relation_metrics(
                    dev_rows, annotations,
                    qe_exact if backend == "exact" else qe_ann,
                    et_exact if backend == "exact" else et_ann,
                    modality,
                )
                for backend in ("exact", "ann")
            }
            probe_evidence = sorted(
                {
                    evidence_id
                    for query_id in probe_ids
                    for target in annotations[query_id].values()
                    for evidence_id in target[modality]
                }
            )
            for backend, qe_top, et_top in (
                ("exact", qe_exact_top, et_exact_top),
                ("ann", qe_ann_top, et_ann_top),
            ):
                qe_rows = [qe_top[query_id][:50] for query_id in probe_ids]
                et_rows = [et_top[evidence_id][:50] for evidence_id in probe_evidence]
                for relation, top_rows in (("QE", qe_rows), ("ET", et_rows)):
                    key = (backend, modality, relation)
                    if name == "STOP356":
                        counts = Counter(value for values in top_rows for value in values)
                        fixed_hub_targets[key] = set(value for value, _count in counts.most_common(100))
                    state["hub"][f"{relation}_{modality}_{backend}"] = hub_metrics(
                        top_rows, fixed_hub_targets.get(key)
                    )
        for row in dev_rows:
            query_id = str(row["query_id"])
            for raw_target in row["positive_target_ids"]:
                target_id = str(raw_target)
                for modality in ("text", "image"):
                    for evidence_id in annotations[query_id][target_id][modality]:
                        qe_exact = rank_cache[(modality, "qe_exact")][0].get(query_id, {}).get(evidence_id)
                        qe_ann = rank_cache[(modality, "qe_ann")][0].get(query_id, {}).get(evidence_id)
                        et_exact = rank_cache[(modality, "et_exact")][0].get(evidence_id, {}).get(target_id)
                        et_ann = rank_cache[(modality, "et_ann")][0].get(evidence_id, {}).get(target_id)
                        record = {
                            "seed": seed,
                            "checkpoint": name,
                            "query_id": query_id,
                            "target_id": target_id,
                            "evidence_id": evidence_id,
                            "modality": modality,
                            "qe_exact_rank": qe_exact,
                            "qe_ann_rank": qe_ann,
                            "et_exact_rank": et_exact,
                            "et_ann_rank": et_ann,
                        }
                        per_path_rows.append(record)
                        if qe_exact is None:
                            exact_missing.append({**record, "edge": "QE", "reason": "evidence_not_in_modality_lake" if evidence_id not in lake_sets[modality] else "unexpected_rank_map_missing"})
                        if et_exact is None:
                            exact_missing.append({**record, "edge": "ET", "reason": "target_not_in_table_lake" if target_id not in lake_sets["table"] else "unexpected_rank_map_missing"})
        results["checkpoints"][name] = state
        del destination_matrices, model
        torch.cuda.empty_cache()
        print(json.dumps({"event": "checkpoint_complete", "seed": seed, "checkpoint": name, "elapsed_seconds": time.monotonic() - started}), flush=True)
    write_rows(per_path_path, per_path_rows)
    if exact_missing:
        write_rows(missing_path, exact_missing)
    results["runtime_seconds"] = time.monotonic() - started
    write_json(result_path, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(13, 29), required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        choices=(
            "STOP356", "JOINT500", "JOINT659", "F-P500", "F-P659",
            "F-P-ETNAT500", "F-P-ETNAT659", "C2-F-P89", "C2-F-P178",
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--annotations-only", action="store_true")
    args = parser.parse_args()
    if args.annotations_only:
        build_static_annotations()
        print(json.dumps({"status": "complete", "output": str(DIAGNOSTICS.resolve())}))
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    names = args.checkpoint or ["STOP356", "JOINT500", "JOINT659", "F-P500", "F-P659", "F-P-ETNAT500", "F-P-ETNAT659"]
    result = run_seed(args.seed, names, device)
    print(json.dumps({"status": "complete", "seed": args.seed, "checkpoints": list(result["checkpoints"])}))


if __name__ == "__main__":
    main()
