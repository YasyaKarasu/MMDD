"""Audit R23 evidence admission and exact second-hop target concentration.

The R23 full-lake rankings retain the paths used to admit evidence targets.
This audit recomputes text/image -> table exact top-20 rankings for every
retained evidence node, then reports evidence-only positive recovery and hub
concentration at both ANN admission and exact ET levels.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import statistics
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.artifacts import checkpoint_fingerprint
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from run_stage1_r21 import out as r21_out, paths as r21_paths
from run_stage1_r23 import FINAL_STEP, out, read_rows, r23_paths, write_rows

ROOT = Path(__file__).resolve().parents[1]


def _stats(counter: collections.Counter[str], queries: int) -> dict[str, Any]:
    total = sum(counter.values())
    return {
        "unique_targets": len(counter),
        "universal_targets": sum(value == queries for value in counter.values()),
        "top20": counter.most_common(20),
        "top20_share": (sum(value for _, value in counter.most_common(20)) / total if total else 0.0),
        "total_target_occurrences": total,
    }


def _collect(rows: list[dict[str, Any]]) -> tuple[dict[tuple[str, str], set[str]], dict[str, Any]]:
    nodes: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    ann_counter: collections.Counter[str] = collections.Counter()
    ann_query_sets: list[set[str]] = []
    e_only: list[float] = []
    for row in rows:
        direct = set(map(str, row["direct_ann"]))
        evidence = set(map(str, row["evidence_ann"]))
        positives = set(map(str, row["positive_target_ids"]))
        ann_counter.update(evidence)
        ann_query_sets.append(evidence)
        e_only.append(len(positives & (evidence - direct)) / len(positives) if positives else 0.0)
        for target in row.get("evidence_paths", []):
            target_id = str(target["target_id"])
            for path in target.get("paths", []):
                if path.get("kind") != "evidence":
                    continue
                nodes[(str(path["evidence_id"]), str(path["evidence_type"]))].add(target_id)
    return nodes, {
        "queries": len(rows),
        "ann_evidence": _stats(ann_counter, len(rows)),
        "evidence_only_positive_recall": statistics.fmean(e_only) if e_only else 0.0,
    }


@torch.inference_mode()
def run(root: Path, arm: str, seed: int, step: int, device_name: str,
        checkpoint_path: Path | None = None, ranking_path: Path | None = None,
        output_dir: Path | None = None) -> dict[str, Any]:
    source = out(root) / "full_lake" / arm / f"seed{seed}_step{step:06d}"
    ranking_path = ranking_path or (source / "rankings.jsonl.gz")
    if not ranking_path.exists():
        raise FileNotFoundError(ranking_path)
    rows = list(read_rows(ranking_path))
    nodes, summary = _collect(rows)
    device = torch.device(device_name)
    store = FeatureStore.from_path(r23_paths(root)["features"], cache_size=100000)
    ck = checkpoint_path or (out(root) / arm / f"seed{seed}" / "checkpoints" / f"step_{step:06d}.pt")
    model = load_student(ck, device).eval()
    table_ids = json.loads((r21_out(root) / "indexes" / "Qwen-Raw" / "table_ids.json").read_text())
    table_embeddings = torch.stack([store.embedding_features(str(x)).embedding for x in table_ids]).to(device=device, dtype=torch.float32)
    table_vec = model.project(table_embeddings, "table", role="target")

    exact_by_node: dict[tuple[str, str], list[str]] = {}
    detail_rows: list[dict[str, Any]] = []
    for evidence_type in ("text", "image"):
        keys = sorted(key for key in nodes if key[1] == evidence_type)
        relation = model.relations[model.relation_key(evidence_type, "table")]
        for start in range(0, len(keys), 64):
            part = keys[start : start + 64]
            src = torch.stack([store.embedding_features(key[0]).embedding for key in part]).to(device=device, dtype=torch.float32)
            scores = model.project(src, evidence_type, role="evidence") @ relation @ table_vec.T
            _values, indices = scores.topk(k=min(20, len(table_ids)), dim=1)
            for key, index_row in zip(part, indices.cpu()):
                exact_ids = [str(table_ids[int(index)]) for index in index_row]
                exact_by_node[key] = exact_ids
                ann_ids = sorted(nodes[key])
                detail_rows.append({
                    "evidence_id": key[0],
                    "evidence_type": key[1],
                    "exact_top20": exact_ids,
                    "ann_target_ids": ann_ids,
                    "exact_top20_in_ann": len(set(exact_ids) & set(ann_ids)),
                    "ann_targets_retrieved": len(ann_ids),
                })

    exact_counter: collections.Counter[str] = collections.Counter()
    for ids in exact_by_node.values():
        exact_counter.update(ids)
    exact_query_sets: list[set[str]] = []
    exact_only_recall: list[float] = []
    for row in rows:
        node_keys = {
            (str(path["evidence_id"]), str(path["evidence_type"]))
            for target in row.get("evidence_paths", [])
            for path in target.get("paths", [])
            if path.get("kind") == "evidence"
        }
        exact_union = set().union(*(set(exact_by_node[key]) for key in node_keys if key in exact_by_node))
        exact_query_sets.append(exact_union)
        direct = set(map(str, row["direct_ann"]))
        positives = set(map(str, row["positive_target_ids"]))
        exact_only_recall.append(len(positives & (exact_union - direct)) / len(positives) if positives else 0.0)
    summary["exact_et"] = {
        "nodes": len(detail_rows),
        "nodes_by_modality": collections.Counter(row["evidence_type"] for row in detail_rows),
        "target_rankings": _stats(exact_counter, len(rows)),
        "mean_exact_top20_in_ann": statistics.fmean(row["exact_top20_in_ann"] / 20.0 for row in detail_rows) if detail_rows else 0.0,
        "mean_ann_targets_retrieved_per_node": statistics.fmean(row["ann_targets_retrieved"] for row in detail_rows) if detail_rows else 0.0,
        "query_union_unique_targets": len(set().union(*exact_query_sets)) if exact_query_sets else 0,
        "query_union_universal_targets": sum(sum(target in query for query in exact_query_sets) == len(rows) for target in set().union(*exact_query_sets)) if exact_query_sets else 0,
        "positive_recall_exact_et_union": statistics.fmean(len(set(map(str, row["positive_target_ids"])) & query) / len(row["positive_target_ids"]) if row["positive_target_ids"] else 0.0 for row, query in zip(rows, exact_query_sets)) if rows else 0.0,
        "e_only_positive_recall_exact_et_union": statistics.fmean(exact_only_recall) if exact_only_recall else 0.0,
    }
    summary.update({"status": "complete", "arm": arm, "seed": seed, "step": step,
                    "checkpoint_sha256": checkpoint_fingerprint(ck),
                    "table_corpus": len(table_ids), "ranking_source": str(ranking_path.resolve())})
    # Store a content hash for the checkpoint separately in the machine-readable
    # source manifest; the path/size above keeps this audit independent of a
    # second heavyweight checkpoint read.
    destination = output_dir or (source / "evidence_diagnostics")
    destination.mkdir(parents=True, exist_ok=True)
    write_rows(destination / "et_exact_nodes.jsonl.gz", detail_rows)
    (destination / "ET_EXACT.json").write_text(json.dumps(summary, indent=2, default=lambda value: dict(value)), encoding="utf-8")
    if output_dir is None:
        metrics_path = source / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        metrics["evidence_diagnostics"] = summary
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--step", type=int, default=FINAL_STEP)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--ranking-source", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve(), args.arm, args.seed, args.step, args.device,
                         args.checkpoint, args.ranking_source, args.output_dir), indent=2,
                     default=lambda value: dict(value)))


if __name__ == "__main__":
    main()
