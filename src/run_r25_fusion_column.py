#!/usr/bin/env python3
"""Run contract §8 column-only promotion from frozen Qwen column vectors."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import torch


def _read(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _standardize(row: dict[str, Any]) -> tuple[list[str], list[str], list[str], list[float] | None]:
    if "scorer_ids" in row:
        direct = row["scorer_ids"]["DIRECT_ANN"]
        evidence = row["scorer_ids"].get("QT_OVER_U", row["scorer_ids"].get("QT_OVER_M", {}))
        return list(map(str, direct["ranking"])), list(map(str, evidence.get("ranking", []))), list(map(str, direct["candidate_ids"])), list(map(float, direct["scores"]))
    direct = list(map(str, row.get("direct_ann", row.get("U", []))))
    evidence = list(map(str, row.get("evidence_ann", [])))
    return direct, evidence, list(dict.fromkeys(direct + evidence)), None


def _index_vectors(vectors: dict[str, torch.Tensor]) -> dict[tuple[str, str], list[torch.Tensor]]:
    indexed: dict[tuple[str, str], list[torch.Tensor]] = {}
    for key, value in vectors.items():
        role, table_id, _column_index = key.split(":", 2)
        indexed.setdefault((role, table_id), []).append(value.float())
    return indexed


def _column_similarity(indexed: dict[tuple[str, str], list[torch.Tensor]], query_id: str, target_id: str) -> float | None:
    query = indexed.get(("query", query_id), [])
    target = indexed.get(("target", target_id), [])
    if not query or not target:
        return None
    return max(float(torch.dot(q, t)) for q in query for t in target)


def _reference_cdf(indexed: dict[tuple[str, str], list[torch.Tensor]], train_rows: list[dict[str, Any]]) -> list[list[float]]:
    per_query = []
    for row in train_rows:
        query_id = str(row["query_id"])
        values = []
        for item in row.get("results", []):
            score = _column_similarity(indexed, query_id, str(item["target_id"]))
            if score is not None:
                values.append(score)
        if values:
            per_query.append(values)
    return per_query


def _cdf(value: float, reference: list[list[float]]) -> float:
    if not reference:
        return 0.0
    return sum(sum(score <= value for score in values) / len(values) for values in reference) / len(reference)


def _run_source(name: str, rows: list[dict[str, Any]], indexed: dict[tuple[str, str], list[torch.Tensor]], reference: list[list[float]], output_root: Path) -> dict[str, Any]:
    output_rows = []
    missing = 0
    alpha_values = []
    recalls = {10: [], 20: [], 50: []}
    for row in rows:
        direct, evidence, candidates, direct_scores = _standardize(row)
        rank_d = {target: index + 1 for index, target in enumerate(direct)}
        rank_e = {target: index + 1 for index, target in enumerate(evidence)}
        c_values = {}
        alpha = {}
        for target in candidates:
            value = _column_similarity(indexed, str(row["query_id"]), target)
            c_values[target] = value
            if value is None:
                alpha[target] = 1.0
                missing += 1
            else:
                alpha[target] = 1.0 - _cdf(value, reference)
            alpha_values.append(alpha[target])
        def score(target: str) -> float:
            d_term = 1.0 / (60.0 + rank_d[target]) if target in rank_d else 0.0
            e_term = 1.0 / (60.0 + rank_e[target]) if target in rank_e else 0.0
            return d_term + alpha[target] * e_term
        ranking = sorted(candidates, key=lambda target: (-score(target), target))
        positives = set(map(str, row.get("positive_target_ids", [])))
        for k in recalls:
            recalls[k].append(float(bool(positives & set(ranking[:k]))))
        direct_top10 = [c_values[target] for target in direct[:10] if c_values.get(target) is not None]
        output_rows.append({
            "query_id": str(row["query_id"]),
            "query_kind": row.get("query_kind"),
            "positive_target_ids": sorted(positives),
            "fusion_method": "column-only",
            "ranking": ranking,
            "source_scorers": ["DIRECT_ANN", "QT_OVER_U"],
            "alpha": alpha,
            "column_similarity": c_values,
            "C_q_direct_top10": max(direct_top10) if direct_top10 else None,
            "missing_column_representation": sum(value is None for value in c_values.values()),
            "direct_scores_complete": direct_scores is not None,
            "online_gt_inputs": False,
        })
    destination = output_root / name
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "column-only.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metric = {"queries": len(output_rows), "missing_column_representation": missing, "alpha_mean": sum(alpha_values) / len(alpha_values) if alpha_values else 1.0, "alpha_lt_0_1": sum(value < 0.1 for value in alpha_values), "alpha_gt_0_9": sum(value > 0.9 for value in alpha_values), "R@10": sum(recalls[10]) / len(recalls[10]) if recalls[10] else 0.0, "R@20": sum(recalls[20]) / len(recalls[20]) if recalls[20] else 0.0, "R@50": sum(recalls[50]) / len(recalls[50]) if recalls[50] else 0.0}
    (destination / "column-only.metrics.json").write_text(json.dumps(metric, indent=2) + "\n", encoding="utf-8")
    return {"status": "complete", "method": "column-only", "output": str(path.resolve()), "metrics": metric}


def run(args: argparse.Namespace) -> None:
    vectors: dict[str, torch.Tensor] = {}
    for path in args.column_cache:
        payload = torch.load(Path(path).resolve(), map_location="cpu", weights_only=False)
        vectors.update(payload["vectors"])
    indexed = _index_vectors(vectors)
    reference = _reference_cdf(indexed, _read(Path(args.train_retrieval).resolve()))
    sources = [("B13-FULL/seed13", Path(args.r25_b13_13)), ("B13-FULL/seed29", Path(args.r25_b13_29)), ("SPLIT-QTKD/seed13", Path(args.r25_split_13)), ("SPLIT-QTKD/seed29", Path(args.r25_split_29)), ("Qwen-Raw/seed13", Path(args.raw)), ("B13/seed13", Path(args.b13)), ("N-U/seed13", Path(args.nu13)), ("N-U/seed29", Path(args.nu29))]
    seen = {name for name, _ in sources}
    ranking_root = Path(args.output_root).resolve().parent / "rankings"
    for path in sorted(ranking_root.glob("*/seed*/query_rankings.jsonl.gz")):
        name = f"{path.parent.parent.name}/{path.parent.name}"
        if name not in seen:
            sources.append((name, path))
            seen.add(name)
    results = [_run_source(name, _read(path.resolve()), indexed, reference, Path(args.output_root).resolve()) for name, path in sources]
    print(json.dumps({"status": "complete", "reference_queries": len(reference), "sources": results}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--column-cache", nargs="+", required=True)
    parser.add_argument("--train-retrieval", required=True)
    parser.add_argument("--r25-b13-13", required=True)
    parser.add_argument("--r25-b13-29", required=True)
    parser.add_argument("--r25-split-13", required=True)
    parser.add_argument("--r25-split-29", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--b13", required=True)
    parser.add_argument("--nu13", required=True)
    parser.add_argument("--nu29", required=True)
    parser.add_argument("--output-root", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
