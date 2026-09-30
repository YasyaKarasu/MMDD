"""Fresh B+IDF bridge fusion over a frozen Stage-1 candidate population.

The visible ``vis_row``/``vis_idf`` values are selector independent.  This
module only replaces the bridge tier with scores produced by the current fresh
run, so it never loads a Stage-2 checkpoint.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .bridge_first_rerank import rrf_fuse


def build_orders(base: Sequence[str], bridge: Mapping[str, float], vis_row: Mapping[str, float],
                 vis_idf: Mapping[str, float]) -> dict[str, tuple[list[str], list[str]]]:
    """Return the historical lexical and IDF pure/RRF orders exactly."""
    stage1 = {target: rank for rank, target in enumerate(base, 1)}
    tier1 = sorted((target for target in base if bridge[target] > 0),
                   key=lambda target: (-round(bridge[target], 8), stage1[target], target))
    rest = [target for target in base if bridge[target] == 0]
    lexical = tier1 + sorted((target for target in rest if vis_row[target] > 0),
                             key=lambda target: (-round(vis_row[target], 8), stage1[target], target))
    lexical += sorted((target for target in rest if vis_row[target] <= 0), key=lambda target: stage1[target])
    idf = tier1 + sorted((target for target in rest if vis_idf[target] > 0),
                         key=lambda target: (-round(vis_idf[target], 12),
                                             -round(vis_row[target], 8), stage1[target], target))
    idf += sorted((target for target in rest if vis_idf[target] <= 0), key=lambda target: stage1[target])
    return {"L_LEXICO": (lexical, rrf_fuse(base, lexical)),
            "L_LEXICO_IDF": (idf, rrf_fuse(base, idf))}


def fresh_bridge_scores(base: Sequence[str], score_record: Mapping[str, Any]) -> dict[str, float]:
    """Project a fresh selector score record onto its frozen candidate scope."""
    values = score_record.get("table_scores", {})
    return {target: float(values.get(target, 0.0)) for target in base}


def _metrics(order: Sequence[str], gold: set[str], k: int) -> dict[str, float]:
    prefix = list(order[:k])
    hits = sum(target in gold for target in prefix)
    precision = hits / k if k else 0.0
    dcg = sum((1.0 / math.log2(index + 2)) for index, target in enumerate(prefix) if target in gold)
    ideal = sum(1.0 / math.log2(index + 2) for index in range(min(k, len(gold))))
    return {f"R{k}": float(bool(hits)), f"P{k}": precision,
            f"NDCG{k}": dcg / ideal if ideal else 0.0}


def run_addendum(idf_scores: Mapping[str, Mapping[str, Any]], fresh_scores_dir: Path,
                 output: Path, *, arm: str = "SEL_E_FRESH", cutoffs: Sequence[int] = (1, 5, 10, 15, 20, 30, 50)) -> dict[str, Any]:
    """Fuse fresh bridge scores and write auditable per-query rows plus a receipt."""
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for query_id, frozen in idf_scores.items():
        split = str(frozen.get("split", ""))
        root = Path(fresh_scores_dir)
        source = root / split / f"{query_id}.json"
        if not source.is_file():
            source = root / arm / split / f"{query_id}.json"
        if not source.is_file():
            missing.append(query_id)
            continue
        score_record = json.loads(source.read_text())
        base = list(frozen["base"])
        bridge = fresh_bridge_scores(base, score_record)
        orders = build_orders(base, bridge, frozen["vis_row"], frozen["vis_idf"])
        gold = set(frozen["gold"])
        for policy, (pure, fused) in orders.items():
            for order_name, order in (("PURE", pure), ("RRF60", fused)):
                row = {"query_id": query_id, "split": split, "policy": policy,
                       "order": order_name, "source_group": frozen.get("source_group"),
                       "kind": frozen.get("kind"), "gold": sorted(gold), "ranking": order}
                for cutoff in cutoffs:
                    row.update(_metrics(order, gold, cutoff))
                rows.append(row)
    path = output / "PER_QUERY_B_PLUS_IDF.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    aggregates = []
    for split in sorted({row["split"] for row in rows}):
        for policy in sorted({row["policy"] for row in rows if row["split"] == split}):
            for order_name in ("PURE", "RRF60"):
                group = [row for row in rows if row["split"] == split and row["policy"] == policy
                         and row["order"] == order_name]
                if not group:
                    continue
                aggregates.append({"split": split, "policy": policy, "order": order_name,
                                   "queries": len(group),
                                   **{metric: sum(row[metric] for row in group) / len(group)
                                      for k in cutoffs for metric in (f"R{k}", f"P{k}", f"NDCG{k}")}})
    (output / "METRICS_B_PLUS_IDF.json").write_text(json.dumps(aggregates, indent=2) + "\n")
    receipt = {"arm": arm, "queries": len(idf_scores), "rows": len(rows),
               "scores_missing": len(missing), "missing_query_ids": missing,
               "stage2_checkpoint_used": False, "new_minilm_encodings": 0,
               "candidate_scores_source": "fresh selector bridge table_scores",
               "metrics_path": str(output / "METRICS_B_PLUS_IDF.json")}
    (output / "B_PLUS_IDF_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
