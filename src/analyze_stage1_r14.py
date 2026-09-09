#!/usr/bin/env python
"""Compute R14 exact-direct boundaries and cross-model pool/score attribution."""

from __future__ import annotations

import argparse
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.data import load_target_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import load_corpus_ids
from run_stage1_r13 import _paths as r13_paths
from run_stage1_r14 import freeze_plan


KS = (10, 20, 50)


def _read_jsonl_gz(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {
            str(record["query_id"]): record
            for record in (json.loads(line) for line in handle)
        }


def _r13_artifacts(root: Path) -> dict[str, dict[str, Path]]:
    base = root / "work/stage1_optimization_r13_20260909"
    return {
        "s0": {
            "checkpoint": Path(freeze_plan(root)["r13_dependencies"]["s0"]["path"]),
            "pool": base / "taskA_stage1_protocol/s0/evaluation_step0/path_pool.jsonl.gz",
            "rankings": base / "taskA_stage1_protocol/s0/evaluation_step0/rankings.jsonl.gz",
        },
        "p_s_target_only": {
            "checkpoint": Path(
                freeze_plan(root)["r13_dependencies"]["selected_recipe"]["checkpoint"]
            ),
            "pool": base
            / "taskD_witness_supervision/p_s_target_only/evaluation_step178/path_pool.jsonl.gz",
            "rankings": base
            / "taskD_witness_supervision/p_s_target_only/evaluation_step178/rankings.jsonl.gz",
        },
    }


@torch.inference_mode()
def _target_vectors(
    model,
    store: FeatureStore,
    target_ids: list[str],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    rows = []
    for start in range(0, len(target_ids), batch_size):
        ids = target_ids[start : start + batch_size]
        embeddings = torch.stack(
            [store.embedding_features(object_id).embedding for object_id in ids]
        ).to(device=device, dtype=torch.float32)
        rows.append(model.project(embeddings, "table", role="target"))
    return torch.cat(rows)


def _recall(
    ids: list[str], positives: set[str], k: int
) -> tuple[float, list[str]]:
    hits = sorted(positives & set(ids[:k]))
    return len(hits) / len(positives), hits


def _aggregate_matrix(
    per_query: list[dict[str, Any]]
) -> dict[str, dict[str, dict[str, float]]]:
    result = {}
    for kind in ("all", "implicit", "explicit"):
        rows = [
            row
            for row in per_query
            if kind == "all" or row["query_kind"] == kind
        ]
        by_k = {}
        for k in KS:
            values = {
                key: float(np.mean([row["matrix"][str(k)][key] for row in rows]))
                for key in ("R00", "R01", "R10", "R11")
            }
            values["delta_pool"] = 0.5 * (
                (values["R10"] - values["R00"])
                + (values["R11"] - values["R01"])
            )
            values["delta_score"] = 0.5 * (
                (values["R01"] - values["R00"])
                + (values["R11"] - values["R10"])
            )
            values["delta_total"] = values["R11"] - values["R00"]
            by_k[str(k)] = values
        result[kind] = by_k
    return result


@torch.inference_mode()
def analyze(args: argparse.Namespace) -> None:
    plan = freeze_plan(args.root)
    output = args.root / "work/stage1_optimization_r14_20260909/stage1_A_attribution"
    boundary_path = output / "exact_direct_boundary.json"
    matrix_path = output / "pool_score_cross_matrix.json"
    contributions_path = output / "query_level_contributions.jsonl.gz"
    if boundary_path.is_file() or matrix_path.is_file() or contributions_path.is_file():
        raise FileExistsError("Inspect existing R14 Task A outputs before rerunning")

    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    paths = r13_paths(args.root)
    artifacts = _r13_artifacts(args.root)
    for by_name in artifacts.values():
        for path in by_name.values():
            if not path.is_file():
                raise FileNotFoundError(path)
    pools = {
        name: _read_jsonl_gz(values["pool"])
        for name, values in artifacts.items()
    }
    rankings = {
        name: _read_jsonl_gz(values["rankings"])
        for name, values in artifacts.items()
    }
    examples = load_target_examples(paths["dev_targets"], split="dev")
    example_by_id = {example.query_id: example for example in examples}
    query_ids = [example.query_id for example in examples]
    if any(set(records) != set(query_ids) for records in [*pools.values(), *rankings.values()]):
        raise ValueError("Task A artifacts do not align to the frozen dev queries")

    store = FeatureStore.from_path(paths["features"], cache_size=260_000)
    ids_by_type = load_corpus_ids(paths["corpus"], store)
    target_ids = sorted(ids_by_type["table"])
    target_index = {target_id: index for index, target_id in enumerate(target_ids)}
    if any(
        target_id not in target_index
        for records in pools.values()
        for record in records.values()
        for target_id in record["paths_by_target"]
    ):
        raise ValueError("Natural pool contains a target outside the legal target corpus")
    preload_started = time.monotonic()
    store.preload_embeddings([*target_ids, *query_ids])
    preload_seconds = time.monotonic() - preload_started
    models = {
        name: load_student(values["checkpoint"], device).eval()
        for name, values in artifacts.items()
    }
    vectors = {
        name: _target_vectors(
            model, store, target_ids, device, args.batch_size
        )
        for name, model in models.items()
    }
    boundaries: dict[str, Any] = {
        "format_version": 1,
        "status": "complete",
        "legal_targets": len(target_ids),
        "queries": len(query_ids),
        "models": {},
    }
    per_query = []
    boundary_rows = defaultdict(list)
    started = time.monotonic()
    for query_id in query_ids:
        example = example_by_id[query_id]
        positives = set(example.positive_target_ids)
        query_embedding = store.embedding_features(query_id).embedding.to(
            device=device, dtype=torch.float32
        )
        full_scores = {}
        exact_orders = {}
        exact_ranks = {}
        for name, model in models.items():
            query_vector = model.relation_query(
                query_embedding, "table", "table", source_role="query"
            )
            scores = (query_vector @ vectors[name].T).float().cpu().numpy()
            order = np.argsort(-scores, kind="stable")
            ranks = np.empty(len(order), dtype=np.int32)
            ranks[order] = np.arange(1, len(order) + 1, dtype=np.int32)
            full_scores[name] = scores
            exact_orders[name] = order
            exact_ranks[name] = ranks

        matrix = {}
        pool_ids = {
            "0": list(pools["s0"][query_id]["paths_by_target"]),
            "1": list(pools["p_s_target_only"][query_id]["paths_by_target"]),
        }
        for k in KS:
            by_cell = {}
            for pool_key, score_name, cell in (
                ("0", "s0", "R00"),
                ("0", "p_s_target_only", "R01"),
                ("1", "s0", "R10"),
                ("1", "p_s_target_only", "R11"),
            ):
                ordered = sorted(
                    pool_ids[pool_key],
                    key=lambda target_id: (
                        -float(full_scores[score_name][target_index[target_id]]),
                        target_id,
                    ),
                )
                by_cell[cell] = _recall(ordered, positives, k)[0]
            matrix[str(k)] = by_cell

        query_row = {
            "query_id": query_id,
            "query_kind": example.query_kind,
            "positive_target_ids": sorted(positives),
            "positive_denominator": len(positives),
            "pool_sizes": {key: len(value) for key, value in pool_ids.items()},
            "matrix": matrix,
            "boundary": {},
        }
        for name in ("s0", "p_s_target_only"):
            record = pools[name][query_id]
            ranking = rankings[name][query_id]["rankings"]
            score = full_scores[name]
            exact_order = exact_orders[name]
            exact_top100 = [target_ids[index] for index in exact_order[:100]]
            exact_rank = exact_ranks[name]
            ann_ids = {
                target_id
                for target_id, paths_by_target in record["paths_by_target"].items()
                if any(path.get("kind") == "direct" for path in paths_by_target)
            }
            evidence_ids = {
                target_id
                for target_id, paths_by_target in record["paths_by_target"].items()
                if any(path.get("kind") == "evidence" for path in paths_by_target)
            }
            ann_ordered = sorted(
                ann_ids,
                key=lambda target_id: (
                    -float(score[target_index[target_id]]),
                    target_id,
                ),
            )
            natural_ordered = sorted(
                record["paths_by_target"],
                key=lambda target_id: (
                    -float(score[target_index[target_id]]),
                    target_id,
                ),
            )
            exact_union_ordered = sorted(
                set(exact_top100) | evidence_ids,
                key=lambda target_id: (
                    -float(score[target_index[target_id]]),
                    target_id,
                ),
            )
            saved_f1 = ranking["f1_union_direct"]
            saved_pure = ranking["pure_direct100"]
            checks = {}
            for k in KS:
                exact_ids = exact_top100[:k]
                ann_top = ann_ordered[:k]
                natural_top = natural_ordered[:k]
                exact_union_top = exact_union_ordered[:k]
                saved_f1_ids = saved_f1[str(k)]["target_ids"]
                saved_pure_ids = saved_pure[str(k)]["target_ids"]
                checks[str(k)] = {
                    "exact_union_equals_exact": exact_union_top == exact_ids,
                    "natural_recompute_equals_saved_f1": natural_top == saved_f1_ids,
                    "ann_recompute_equals_saved_pure": ann_top == saved_pure_ids,
                    "ann_exact_overlap": len(set(ann_top) & set(exact_ids)),
                    "ann_exact_overlap_rate": len(set(ann_top) & set(exact_ids)) / k,
                }
            f1_top10 = saved_f1["10"]["target_ids"]
            pure_top10 = saved_pure["10"]["target_ids"]
            entrants = sorted(set(f1_top10) - set(pure_top10))
            weighted_rescued = len(
                positives & (set(f1_top10) - set(pure_top10))
            ) / len(positives)
            weighted_displaced = len(
                positives & (set(pure_top10) - set(f1_top10))
            ) / len(positives)
            model_boundary = {
                "ann_d100_count": len(ann_ids),
                "evidence_target_count": len(evidence_ids),
                "checks": checks,
                "f1_new_entrants": [
                    {
                        "target_id": target_id,
                        "exact_direct_rank": int(
                            exact_rank[target_index[target_id]]
                        ),
                        "positive": target_id in positives,
                    }
                    for target_id in entrants
                ],
                "weighted_rescued_f1_minus_pure@10": weighted_rescued,
                "weighted_displaced_f1_minus_pure@10": weighted_displaced,
            }
            query_row["boundary"][name] = model_boundary
            boundary_rows[name].append(model_boundary)
        per_query.append(query_row)

    for name, rows in boundary_rows.items():
        checks = {
            str(k): {
                key: (
                    sum(row["checks"][str(k)][key] for row in rows)
                    if key != "ann_exact_overlap_rate"
                    else float(
                        np.mean(
                            [row["checks"][str(k)][key] for row in rows]
                        )
                    )
                )
                for key in (
                    "exact_union_equals_exact",
                    "natural_recompute_equals_saved_f1",
                    "ann_recompute_equals_saved_pure",
                    "ann_exact_overlap_rate",
                )
            }
            for k in KS
        }
        boundaries["models"][name] = {
            "checkpoint": str(artifacts[name]["checkpoint"].resolve()),
            "checkpoint_sha256": checkpoint_fingerprint(
                artifacts[name]["checkpoint"]
            ),
            "checks": checks,
            "f1_new_entrant_targets@10": sum(
                len(row["f1_new_entrants"]) for row in rows
            ),
            "weighted_rescued_f1_minus_pure@10": sum(
                row["weighted_rescued_f1_minus_pure@10"] for row in rows
            )
            / len(rows),
            "weighted_displaced_f1_minus_pure@10": sum(
                row["weighted_displaced_f1_minus_pure@10"] for row in rows
            )
            / len(rows),
            "all_exact_union_equalities_passed": all(
                row["checks"][str(k)]["exact_union_equals_exact"]
                for row in rows
                for k in KS
            ),
        }
    boundaries["cost"] = {
        "feature_preload_seconds": preload_seconds,
        "analysis_seconds": time.monotonic() - started,
        "device": args.device,
    }
    boundaries["plan_sha256"] = plan["plan_sha256"]
    write_json(boundary_path, boundaries)
    write_json(
        matrix_path,
        {
            "format_version": 1,
            "status": "complete",
            "definition": {
                "R00": "sort d_S0 over U_S0",
                "R01": "sort d_B13 over U_S0",
                "R10": "sort d_S0 over U_B13",
                "R11": "sort d_B13 over U_B13",
            },
            "aggregate": _aggregate_matrix(per_query),
            "queries": len(per_query),
            "plan_sha256": plan["plan_sha256"],
        },
    )
    with gzip.open(contributions_path, "wt", encoding="utf-8") as handle:
        for row in per_query:
            handle.write(json.dumps(row) + "\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "exact": str(boundary_path),
                "matrix": _aggregate_matrix(per_query)["all"]["10"],
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.root = arguments.root.resolve()
    analyze(arguments)
