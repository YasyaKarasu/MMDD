"""Raw-ID evaluation, candidate funnels, and preregistered paired contrasts."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from . import SCHEMA_VERSION
from .data import utf8_sorted, write_json, write_jsonl_gz
from .evaluate import paired_bootstrap
from .labels import Labels
from .retrieval import PoolRecord


def rank_metrics(ranked: Sequence[str], gold: set[str]) -> dict[str, float]:
    if len(ranked) != len(set(ranked)):
        raise ValueError("duplicate target in ranking")
    if not gold:
        raise ValueError("ranking metric requires at least one gold target")
    values = {}
    for k in (10, 20, 30, 40, 50):
        values[f"R@{k}"] = len(set(ranked[:k]) & gold) / len(gold)
    recovered = len(set(ranked) & gold)
    values["target_coverage"] = recovered / len(gold)
    values["query_hit_rate"] = float(recovered > 0)
    values["oracle@10"] = min(10, recovered) / len(gold)
    return values


def candidate_metrics(pool: PoolRecord, gold: set[str]) -> dict[str, float]:
    collections = {
        "D100": [target for target, _ in pool.direct],
        "E": list(pool.pre_paths),
        "U": pool.U,
        "C150": pool.C150,
        "D150": [target for target, _ in pool.D150],
        "MatchedDirectC": [target for target, _ in pool.MatchedDirectC],
        "MatchedDirectU": [target for target, _ in pool.MatchedDirectU],
    }
    result = {"gold_count": float(len(gold))}
    for name, ids in collections.items():
        hits = len(set(ids) & gold)
        result[f"{name}_gold"] = float(hits)
        result[f"{name}_target_coverage"] = hits / len(gold)
        result[f"{name}_query_hit_rate"] = float(hits > 0)
        result[f"{name}_oracle@10"] = min(10, hits) / len(gold)
        result[f"{name}_size"] = float(len(ids))
    direct_ranked = [target for target, _ in pool.direct]
    result["Direct_ANN_R10"] = len(set(direct_ranked[:10]) & gold) / len(gold)
    return result


def summarize(rows: Mapping[str, dict], gt: Mapping[str, dict]) -> dict:
    fields = sorted({field for values in rows.values() for field in values})
    output = {}
    for segment in ("overall", "implicit", "explicit", "mixed", "unknown"):
        query_ids = [
            query_id for query_id in utf8_sorted(rows)
            if segment == "overall" or gt[query_id]["kind"] == segment
        ]
        output[segment] = {
            "queries": len(query_ids),
            **{
                field: float(np.mean([rows[q][field] for q in query_ids])) if query_ids else None
                for field in fields
            },
        }
    return output


def evaluate_matrix(
    pools: Mapping[str, PoolRecord],
    matrix: Mapping[str, Mapping[str, Mapping[str, dict]]],
    gt: Mapping[str, dict],
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_by_query = {
        query_id: candidate_metrics(pools[query_id], set(gt[query_id]["G"]))
        for query_id in utf8_sorted(pools)
    }
    teacher_summary = {}
    per_query_rows = []
    per_query_views: dict[str, dict[str, float]] = {query_id: {} for query_id in pools}
    for teacher, views in matrix.items():
        teacher_summary[teacher] = {}
        for view, rankings in views.items():
            rows = {}
            for query_id in utf8_sorted(rankings):
                metrics = rank_metrics(rankings[query_id]["target_ids"], set(gt[query_id]["G"]))
                rows[query_id] = metrics
                per_query_views[query_id][f"{teacher}.{view}.R@10"] = metrics["R@10"]
            teacher_summary[teacher][view] = summarize(rows, gt)

    candidate_summary = summarize(candidate_by_query, gt)
    for query_id in utf8_sorted(pools):
        per_query_rows.append({
            "query_id": query_id,
            "source_group": gt[query_id]["source_group"],
            "query_kind": gt[query_id]["kind"],
            **candidate_by_query[query_id],
            **per_query_views[query_id],
        })
    if per_query_rows:
        with (output_dir / "per_query_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_query_rows[0]))
            writer.writeheader()
            writer.writerows(per_query_rows)
    report = {
        "candidate": candidate_summary,
        "teacher": teacher_summary,
        "per_query": per_query_views,
    }
    write_json(output_dir / "METRICS.json", report)
    return report


def export_funnels(
    pools: Mapping[str, PoolRecord],
    rankings: Mapping[str, dict],
    gt: Mapping[str, dict],
    labels: Labels,
    output_dir: Path,
    *,
    seed: int,
    generator: str,
) -> None:
    strict_rows = []
    witness_rows = []
    strict_by_query: dict[str, dict[str, float]] = {}
    for query_id in utf8_sorted(pools):
        pool = pools[query_id]
        gold = set(gt[query_id]["G"])
        direct_ann = {target for target, _ in pool.direct}
        direct_exact = set(pool.direct_exact[:100])
        strict_gold = gold - (direct_ann | direct_exact)
        ranked = rankings[query_id]["target_ids"]
        rank = {target: i + 1 for i, target in enumerate(ranked)}
        strict_by_query[query_id] = {
            "gold": float(len(gold)),
            "strict_gold": float(len(strict_gold)),
        }
        for target_id in utf8_sorted(strict_gold):
            strict_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "seed": seed,
                    "generator": generator,
                    "query_id": query_id,
                    "target_id": target_id,
                    "G_EO": True,
                    "two_hop_reached": target_id in pool.pre_paths,
                    "in_U": target_id in pool.U,
                    "D1_nonempty_bag": bool(pool.retained_paths.get(target_id)),
                    "in_C150": target_id in pool.C150,
                    "teacher_top10": rank.get(target_id, 10**9) <= 10,
                    "teacher_top20": rank.get(target_id, 10**9) <= 20,
                    "teacher_top50": rank.get(target_id, 10**9) <= 50,
                    "teacher_rank": rank.get(target_id),
                }
            )
        for target_id in utf8_sorted(gt[query_id]["W"]):
            for evidence_id in gt[query_id]["W"][target_id]:
                modality = labels.modality[evidence_id]
                first_hit = evidence_id in {
                    e for e, _ in pool.first_hop.get(modality, ())
                }
                second_hit = any(
                    path.evidence_id == evidence_id
                    for path in pool.pre_paths.get(target_id, ())
                )
                witness_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "seed": seed,
                        "generator": generator,
                        "query_id": query_id,
                        "target_id": target_id,
                        "evidence_id": evidence_id,
                        "canonical_content_exists": True,
                        "QE20_hit": first_hit,
                        "modality": modality,
                        "ET50_hit": second_hit,
                        "D1_retained": evidence_id in pool.retained_paths.get(target_id, ()),
                        "target_in_C150": target_id in pool.C150,
                        "teacher_top10": rank.get(target_id, 10**9) <= 10,
                        "teacher_top20": rank.get(target_id, 10**9) <= 20,
                        "teacher_top50": rank.get(target_id, 10**9) <= 50,
                        "teacher_rank": rank.get(target_id),
                    }
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_gz(output_dir / "strict_EO.jsonl.gz", strict_rows)
    write_jsonl_gz(output_dir / "witness.jsonl.gz", witness_rows)
    strict_stages = (
        "two_hop_reached", "in_U", "D1_nonempty_bag", "in_C150",
        "teacher_top10", "teacher_top20", "teacher_top50",
    )
    rows_by_query = {
        query_id: [row for row in strict_rows if row["query_id"] == query_id]
        for query_id in strict_by_query
    }
    nonempty = [q for q, counts in strict_by_query.items() if counts["strict_gold"] > 0]
    strict_summary = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "generator": generator,
        "all_query_denominator": len(strict_by_query),
        "conditional_nonempty_query_denominator": len(nonempty),
        "strict_target_pairs": len(strict_rows),
        "stages": {},
    }
    for stage in strict_stages:
        per_all = [
            sum(bool(row[stage]) for row in rows_by_query[q]) / strict_by_query[q]["gold"]
            for q in strict_by_query
        ]
        per_conditional = [
            sum(bool(row[stage]) for row in rows_by_query[q]) / strict_by_query[q]["strict_gold"]
            for q in nonempty
        ]
        strict_summary["stages"][stage] = {
            "query_macro_all_normalized_by_full_gold": float(np.mean(per_all)) if per_all else None,
            "query_macro_conditional_normalized_by_strict_gold": (
                float(np.mean(per_conditional)) if per_conditional else None
            ),
            "micro_pairs": sum(bool(row[stage]) for row in strict_rows),
            "micro_pair_rate": (
                sum(bool(row[stage]) for row in strict_rows) / len(strict_rows)
                if strict_rows else None
            ),
        }
    write_json(output_dir / "strict_EO_SUMMARY.json", strict_summary)

    pair_groups: dict[tuple[str, str], list[dict]] = {}
    for row in witness_rows:
        pair_groups.setdefault((row["query_id"], row["target_id"]), []).append(row)
    witness_pair_rows = []
    witness_stages = (
        "canonical_content_exists", "QE20_hit", "ET50_hit", "D1_retained",
        "target_in_C150", "teacher_top10", "teacher_top20", "teacher_top50",
    )
    for (query_id, target_id), rows in sorted(
        pair_groups.items(), key=lambda item: (item[0][0].encode("utf-8"), item[0][1].encode("utf-8"))
    ):
        witness_pair_rows.append({
            "schema_version": SCHEMA_VERSION,
            "seed": seed,
            "generator": generator,
            "query_id": query_id,
            "target_id": target_id,
            "witness_count": len(rows),
            **{f"any_{stage}": any(bool(row[stage]) for row in rows) for stage in witness_stages},
        })
    write_jsonl_gz(output_dir / "witness_pairs.jsonl.gz", witness_pair_rows)


def bootstrap_contrast(
    left: Mapping[str, float],
    right: Mapping[str, float],
    gt: Mapping[str, dict],
) -> dict:
    if set(left) != set(right):
        raise ValueError("paired contrast query sets differ")
    deltas = {query_id: left[query_id] - right[query_id] for query_id in left}
    groups = {query_id: gt[query_id]["source_group"] for query_id in deltas}
    return paired_bootstrap(deltas, groups, replicates=10000, seed=20260925)
