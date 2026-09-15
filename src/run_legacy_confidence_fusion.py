#!/usr/bin/env python3
"""Recover legal legacy Direct scores and materialize confidence-only F1.

The historical ranking exports omit score vectors.  This helper only uses
frozen, non-GT inputs: the raw Qwen feature cache, archived B13 detailed direct
paths, and the archived N-U checkpoint.  It fails closed if any Direct ANN
candidate cannot be assigned a score.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore


def _read(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _alpha(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    ordered = sorted(values, reverse=True)
    denominator = ordered[0] - ordered[-1] + 1e-8
    if denominator <= 0:
        return 1.0
    margin = max(0.0, min(1.0, (ordered[0] - ordered[1]) / denominator))
    return 1.0 - margin


def _load_tables(store: FeatureStore, ids: list[str], device: torch.device) -> tuple[dict[str, int], torch.Tensor]:
    vectors = torch.stack([store.embedding_features(identifier).embedding for identifier in ids]).to(device=device, dtype=torch.float32)
    return {identifier: index for index, identifier in enumerate(ids)}, vectors


def _raw_scores(rows: list[dict[str, Any]], store: FeatureStore, table_index: dict[str, int], table_vectors: torch.Tensor, device: torch.device) -> dict[str, list[float]]:
    output = {}
    with torch.inference_mode():
        for row in rows:
            query = store.embedding_features(str(row["query_id"])).embedding.to(device=device, dtype=torch.float32)
            ids = [str(value) for value in row.get("direct_ann", [])]
            if any(value not in table_index for value in ids):
                raise RuntimeError(f"raw Direct candidate missing from frozen table corpus: {row['query_id']}")
            values = query @ table_vectors[[table_index[value] for value in ids]].T
            output[str(row["query_id"])] = [float(value) for value in values.cpu()]
    return output


def _b13_scores(rows: list[dict[str, Any]], detailed: list[dict[str, Any]]) -> dict[str, list[float]]:
    by_query = {str(row["query_id"]): row for row in detailed}
    output = {}
    for row in rows:
        detail = by_query.get(str(row["query_id"]))
        if detail is None:
            raise RuntimeError(f"B13 detailed direct path missing: {row['query_id']}")
        score_map = {}
        for item in detail.get("direct", []):
            direct_paths = [path for path in item.get("paths", []) if path.get("kind") == "direct" and path.get("path_score") is not None]
            if len(direct_paths) != 1:
                raise RuntimeError(f"B13 direct path score incomplete: {row['query_id']} {item.get('target_id')}")
            score_map[str(item["target_id"])] = float(direct_paths[0]["path_score"])
        ids = [str(value) for value in row.get("direct_ann", [])]
        if any(value not in score_map for value in ids):
            raise RuntimeError(f"B13 Direct candidate missing detailed path score: {row['query_id']}")
        output[str(row["query_id"])] = [score_map[value] for value in ids]
    return output


def _nu_scores(rows: list[dict[str, Any]], checkpoint: Path, store: FeatureStore, table_index: dict[str, int], table_vectors: torch.Tensor, device: torch.device) -> dict[str, list[float]]:
    model = load_student(checkpoint, device).eval()
    projected_tables = model.project(table_vectors, "table", role="target")
    relation = model.relations[model.relation_key("table", "table")]
    output = {}
    with torch.inference_mode():
        for row in rows:
            query = store.embedding_features(str(row["query_id"])).embedding.to(device=device, dtype=torch.float32).unsqueeze(0)
            projected_query = model.project(query, "table", role="query") @ relation
            ids = [str(value) for value in row.get("direct_ann", [])]
            if any(value not in table_index for value in ids):
                raise RuntimeError(f"N-U Direct candidate missing from frozen table corpus: {row['query_id']}")
            values = projected_query @ projected_tables[[table_index[value] for value in ids]].T
            output[str(row["query_id"])] = [float(value) for value in values.squeeze(0).cpu()]
    return output


def _materialize(name: str, rows: list[dict[str, Any]], scores: dict[str, list[float]], output_root: Path, provenance: str) -> dict[str, Any]:
    output_rows = []
    recalls = {10: [], 20: [], 50: []}
    for row in rows:
        direct = [str(value) for value in row.get("direct_ann", [])]
        evidence = [str(value) for value in row.get("evidence_ann", [])]
        values = scores[str(row["query_id"])]
        if len(values) != len(direct):
            raise RuntimeError(f"Direct score vector length mismatch: {name}/{row['query_id']}")
        rank_d = {value: index + 1 for index, value in enumerate(direct)}
        rank_e = {value: index + 1 for index, value in enumerate(evidence)}
        candidate_ids = list(dict.fromkeys(direct + evidence))
        alpha = _alpha(values)
        ranking = sorted(candidate_ids, key=lambda value: (-(1.0 / (60 + rank_d[value]) if value in rank_d else 0.0) - alpha * (1.0 / (60 + rank_e[value]) if value in rank_e else 0.0), value))
        positives = set(map(str, row.get("positive_target_ids", [])))
        for k in recalls:
            recalls[k].append(float(bool(positives & set(ranking[:k]))))
        output_rows.append({"query_id": str(row["query_id"]), "query_kind": row.get("query_kind"), "positive_target_ids": sorted(positives), "fusion_method": "confidence-only", "ranking": ranking, "source_scorers": ["DIRECT_ANN", "QT_OVER_U"], "alpha": alpha, "direct_scores": dict(zip(direct, values)), "direct_scores_complete": True, "score_provenance": provenance, "online_gt_inputs": False})
    destination = output_root / name
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "confidence-only.jsonl.gz"
    with gzip.open(output, "wt", encoding="utf-8") as handle:
        for item in output_rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    metrics = {"queries": len(output_rows), "alpha_mean": sum(row["alpha"] for row in output_rows) / len(output_rows), "R@10": sum(recalls[10]) / len(recalls[10]), "R@20": sum(recalls[20]) / len(recalls[20]), "R@50": sum(recalls[50]) / len(recalls[50]), "direct_scores_complete": True, "score_provenance": provenance}
    (destination / "confidence-only.metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return {"source": name, "output": str(output.resolve()), "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--table-ids", required=True, type=Path)
    parser.add_argument("--raw-ranking", required=True, type=Path)
    parser.add_argument("--b13-ranking", required=True, type=Path)
    parser.add_argument("--b13-detailed", required=True, type=Path)
    parser.add_argument("--nu13-ranking", required=True, type=Path)
    parser.add_argument("--nu13-checkpoint", required=True, type=Path)
    parser.add_argument("--nu29-ranking", required=True, type=Path)
    parser.add_argument("--nu29-checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    store = FeatureStore.from_path(args.features.resolve(), cache_size=50000)
    table_ids = list(map(str, json.loads(args.table_ids.resolve().read_text(encoding="utf-8"))))
    table_index, table_vectors = _load_tables(store, table_ids, device)
    raw_rows = _read(args.raw_ranking.resolve())
    b13_rows = _read(args.b13_ranking.resolve())
    nu13_rows = _read(args.nu13_ranking.resolve())
    nu29_rows = _read(args.nu29_ranking.resolve())
    results = [
        _materialize("Qwen-Raw/seed13", raw_rows, _raw_scores(raw_rows, store, table_index, table_vectors, device), args.output_root.resolve(), "frozen Qwen feature dot product over archived Direct ANN IDs"),
        _materialize("B13/seed13", b13_rows, _b13_scores(b13_rows, _read(args.b13_detailed.resolve())), args.output_root.resolve(), "archived B13 detailed direct path_score for every Direct ANN ID"),
        _materialize("N-U/seed13", nu13_rows, _nu_scores(nu13_rows, args.nu13_checkpoint.resolve(), store, table_index, table_vectors, device), args.output_root.resolve(), "archived N-U seed13 checkpoint table->table score for every Direct ANN ID"),
        _materialize("N-U/seed29", nu29_rows, _nu_scores(nu29_rows, args.nu29_checkpoint.resolve(), store, table_index, table_vectors, device), args.output_root.resolve(), "archived N-U seed29 checkpoint table->table score for every Direct ANN ID"),
    ]
    print(json.dumps({"status": "complete", "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
