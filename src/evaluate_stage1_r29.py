"""Evaluate R29 Student checkpoints on the locked exact Direct dev protocol."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from prepare_stage1_r28 import inputs
from run_stage1_r29 import OUT, ROOT, sha, read_rows


DEV = ROOT / "work/stage1_optimization_r26_20260914/common/dev_queries.jsonl"
CORPUS = ROOT / "work/stage1_optimization_r10_20260907/stage1_data/stage1_corpus.jsonl"
TABLE_IDS = ROOT / "work/stage1_diagnostics_r27_20260915_rev2_b13_exact/historical_replay/own_evaluation/indexes/H-C1-step000356/table_ids.json"
FEATURES = Path(inputs()["feature_manifest"]["path"]).parent


def dev_rows() -> list[dict[str, Any]]:
    return list(read_rows(DEV))


def bootstrap(values: list[float], groups: list[str], n: int = 10000) -> list[float]:
    # One source table is one source group in this locked dev population; keep
    # the implementation explicit so the report can distinguish it from train
    # query-level bootstrap.
    by_group: dict[str, list[float]] = {}
    for group, value in zip(groups, values, strict=True):
        by_group.setdefault(group, []).append(value)
    group_values = [sum(v) / len(v) for v in by_group.values()]
    if not group_values:
        return [float("nan"), float("nan")]
    g = torch.Generator().manual_seed(290915)
    samples = torch.tensor(group_values, dtype=torch.float64)[torch.randint(len(group_values), (n, len(group_values)), generator=g)]
    q = torch.quantile(samples.mean(dim=1), torch.tensor([.025, .975], dtype=torch.float64))
    return [float(q[0]), float(q[1])]


def evaluate(checkpoint: Path, label: str, *, device_name: str = "cpu", step: int = 534) -> dict[str, Any]:
    device = torch.device(device_name)
    model = load_student(checkpoint, device).eval()
    store = FeatureStore.from_path(FEATURES, cache_size=220000, cache_bytes=4 * 1024**3)
    # The locked parent ANN receipt already contains the exact corpus table ID
    # order; reading it avoids re-scanning the multimodal corpus file.
    table_ids = json.loads(TABLE_IDS.read_text())
    table_embeddings = torch.stack([store.embedding_features(x).embedding for x in table_ids]).float().to(device)
    queries = dev_rows()
    source_embeddings = torch.stack([store.embedding_features(str(row["query_id"])).embedding for row in queries]).float().to(device)
    with torch.inference_mode():
        targets = model.project(table_embeddings, "table", role="target")
        query_vectors = model.relation_query(source_embeddings, "table", "table", source_role="query")
        rankings: list[list[str]] = []
        for start in range(0, len(queries), 32):
            scores = query_vectors[start:start + 32] @ targets.T
            top = torch.topk(scores, k=50, dim=1).indices.cpu().tolist()
            rankings.extend([[table_ids[int(i)] for i in row] for row in top])
    records = []
    for row, ranking in zip(queries, rankings, strict=True):
        positives = set(str(x) for x in row["positive_target_ids"])
        records.append({"query_id": row["query_id"], "source_table_id": row["source_table_id"], "query_kind": row.get("query_kind"), "positive_target_ids": sorted(positives), "ranking_exact_D50": ranking, "hits": {f"R@{k}": int(bool(positives & set(ranking[:k]))) for k in (10, 20, 50)}})
    metrics: dict[str, Any] = {"label": label, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha(checkpoint), "queries": len(records), "device": device_name, "protocol": "query-macro exact Direct; failures remain in denominator", "teacher_full_u": "not_measured_r29_scope"}
    for k in (10, 20, 50):
        values = [float(x["hits"][f"R@{k}"]) for x in records]
        metrics[f"R@{k}"] = sum(values) / len(values)
        metrics[f"R@{k}_source_group_bootstrap95"] = bootstrap(values, [str(x["source_table_id"]) for x in records])
    probe = sorted({str(x["source_table_id"]): x for x in records}.values(), key=lambda x: (x["source_table_id"], x["query_id"]))[:128]
    top10 = {}
    top1 = {}
    for row in probe:
        for target in row["ranking_exact_D50"][:10]: top10[target] = top10.get(target, 0) + 1
        if row["ranking_exact_D50"]: top1[row["ranking_exact_D50"][0]] = top1.get(row["ranking_exact_D50"][0], 0) + 1
    metrics["hub_probe"] = {"n": len(probe), "max_top10_frequency": max(top10.values(), default=0), "max_top1_frequency": max(top1.values(), default=0), "distinct_top10_targets": len(top10), "top10_frequency": sorted(top10.items(), key=lambda x: (-x[1], x[0]))[:20]}
    out_name = "exact_rankings" if step == 534 else f"exact_rankings_epoch{step // 178}"
    out = OUT / "training" / label / "seed13" / out_name
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "rankings.jsonl.gz", "wt", encoding="utf-8") as f:
        for row in records: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("S-EDGE-FREEZE-P", "S-EDGE-FREEZE-R"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--step", type=int, default=534)
    args = parser.parse_args()
    checkpoint = OUT / "training" / args.arm / "seed13/checkpoints" / f"step_{args.step:06d}.pt"
    print(json.dumps(evaluate(checkpoint, args.arm, device_name=args.device, step=args.step), ensure_ascii=False))


if __name__ == "__main__":
    main()
