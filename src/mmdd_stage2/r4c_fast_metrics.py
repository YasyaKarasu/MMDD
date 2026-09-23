"""Scoring and engineering gates for S2-R4c FAST."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any

from .row_r3_metrics import normalized_equal, numeric_equal, strict_equal

ARMS = ("V0_FULL_BASE", "V1_FULL_HIGHRES", "V2_RAEA_DUAL", "V3_CONSENSUS_DUAL")
CROP_ARMS = ("V2_RAEA_DUAL", "V3_CONSENSUS_DUAL")


def score_records(
    raw: list[dict[str, Any]], evaluation: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    eval_by_unit = {row["unit_id"]: row for row in evaluation}
    scored = []
    for row in raw:
        sidecar = eval_by_unit.get(row["unit_id"])
        if sidecar is None:
            continue
        gold_values = [str(value) for value in sidecar["gold_values"]]
        value = row.get("value")
        status = row.get("status")
        strict = status == "VALUE" and any(strict_equal(value, gold) for gold in gold_values)
        normalized = status == "VALUE" and any(normalized_equal(value, gold) for gold in gold_values)
        numeric = status == "VALUE" and any(numeric_equal(value, gold) for gold in gold_values)
        labels = row.get("evidence_label_map") or {}
        cited_assets = {labels[label] for label in row.get("evidence_ids", []) if label in labels}
        witness = set(sidecar.get("witness_evidence_ids", []))
        witness_cited = bool(cited_assets & witness)
        parsed_support = bool(row.get("evidence_ids")) and row.get("status") == "VALUE"
        scored.append({
            "unit_id": row["unit_id"],
            "arm": row["arm"],
            "dataset": "entitables",
            "query_id": row["query_id"],
            "source_group": row["source_group"],
            "strict_correct": int(strict),
            "normalized_correct": int(normalized),
            "numeric_correct": int(numeric),
            "supported_value_correct": int(strict and witness_cited),
            "known_witness_cited_correct": int(strict and witness_cited),
            "hallucinated_supported_value": int(parsed_support and not (strict and witness_cited)),
            "status_value": int(status == "VALUE"),
            "refused": int(status in {"INSUFFICIENT_EVIDENCE", "AMBIGUOUS"}),
            "parse_error": int(status == "PARSE_ERROR" or bool(row.get("parse_error"))),
            "crop_fallback": int(bool(row.get("crop_fallback"))),
            "elapsed_localizer_seconds": float(row.get("elapsed_localizer_seconds", 0.0)),
            "elapsed_generation_seconds": float(row.get("elapsed_generation_seconds", 0.0)),
            "total_elapsed_seconds": float(row.get("elapsed_localizer_seconds", 0.0))
            + float(row.get("elapsed_generation_seconds", 0.0)),
            "input_pixels": sum(int(view.get("input_pixels", 0)) for view in row.get("views", [])),
            "image_tokens": sum(int(view.get("image_tokens", 0)) for view in row.get("views", [])),
            "prompt_tokens": int(row.get("prompt_tokens", 0)),
            "generated_tokens": int(row.get("generated_tokens", 0)),
            "peak_gpu_memory_bytes": int(row.get("peak_gpu_memory_bytes", 0)),
        })
    return scored


def _query_macro(rows: list[dict[str, Any]], field: str) -> float | None:
    by_query: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_query[str(row["query_id"])].append(float(row[field]))
    return mean(mean(values) for values in by_query.values()) if by_query else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(fraction * len(ordered))))]


def summarize_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"units": 0}
    fields = (
        "strict_correct", "normalized_correct", "numeric_correct",
        "supported_value_correct", "known_witness_cited_correct",
        "hallucinated_supported_value", "refused", "parse_error", "crop_fallback",
    )
    summary: dict[str, Any] = {
        "units": len(rows),
        "queries": len({row["query_id"] for row in rows}),
        "source_groups": len({row["source_group"] for row in rows}),
    }
    for field in fields:
        summary[field] = mean(float(row[field]) for row in rows)
        summary[f"query_macro_{field}"] = _query_macro(rows, field)
    for field in (
        "total_elapsed_seconds", "elapsed_generation_seconds", "input_pixels", "image_tokens",
        "prompt_tokens", "generated_tokens", "peak_gpu_memory_bytes",
    ):
        values = [float(row[field]) for row in rows]
        summary[f"median_{field}"] = median(values)
        summary[f"p95_{field}"] = _percentile(values, 0.95)
    return summary


def paired_wlt(scored: list[dict[str, Any]], arm: str) -> dict[str, int]:
    by_key = {(row["unit_id"], row["arm"]): row for row in scored}
    wins = losses = ties = 0
    for unit_id in sorted({row["unit_id"] for row in scored}):
        base, candidate = by_key.get((unit_id, "V0_FULL_BASE")), by_key.get((unit_id, arm))
        if base is None or candidate is None:
            continue
        delta = candidate["strict_correct"] - base["strict_correct"]
        if delta > 0:
            wins += 1
        elif delta < 0:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "losses": losses, "ties": ties, "paired_units": wins + losses + ties}


def paired_bootstrap(
    scored: list[dict[str, Any]], arm: str, *, iterations: int = 10000, seed: int = 20260923
) -> dict[str, Any]:
    by_key = {(row["unit_id"], row["arm"]): row for row in scored}
    by_query: dict[tuple[str, str], list[float]] = defaultdict(list)
    for unit_id in {row["unit_id"] for row in scored}:
        base, candidate = by_key.get((unit_id, "V0_FULL_BASE")), by_key.get((unit_id, arm))
        if base is None or candidate is None:
            continue
        by_query[(base["source_group"], base["query_id"])].append(
            float(candidate["strict_correct"] - base["strict_correct"])
        )
    grouped: dict[str, list[float]] = defaultdict(list)
    for (group, _query), values in by_query.items():
        grouped[group].append(mean(values))
    keys = sorted(grouped)
    if not keys:
        return {"point": None, "low": None, "high": None, "groups": 0, "iterations": 0}
    point = sum(sum(grouped[key]) for key in keys) / sum(len(grouped[key]) for key in keys)
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        drawn = Counter(keys[rng.randrange(len(keys))] for _ in keys)
        numerator = sum(weight * sum(grouped[key]) for key, weight in drawn.items())
        denominator = sum(weight * len(grouped[key]) for key, weight in drawn.items())
        samples.append(numerator / denominator)
    samples.sort()
    return {
        "point": point,
        "low": samples[max(0, int(0.025 * len(samples)) - 1)],
        "high": samples[min(len(samples) - 1, int(0.975 * len(samples)))],
        "groups": len(keys),
        "iterations": iterations,
        "seed": seed,
    }


def component_summary(scored: list[dict[str, Any]]) -> dict[str, Any]:
    arms = {
        arm: summarize_arm([row for row in scored if row["arm"] == arm])
        for arm in ARMS
    }
    comparisons = {
        arm: {"wlt": paired_wlt(scored, arm), "bootstrap": paired_bootstrap(scored, arm)}
        for arm in ARMS if arm != "V0_FULL_BASE"
    }
    return {"arms": arms, "comparisons_vs_v0": comparisons}


def decide_gate(summary: dict[str, Any]) -> dict[str, Any]:
    arms = summary["arms"]
    if summary.get("evaluable_units", 32) < 32:
        return {"status": "STOP_INSUFFICIENT_EVALUABLE_UNITS", "continue_expand": False,
                "winner": None, "evaluable_units": summary["evaluable_units"],
                "reason": "Label-blind locked units cannot support a 2pp row-level accuracy comparison."}
    if any(arms.get(arm, {}).get("units", 0) < 32 for arm in ARMS):
        return {"status": "STOP_INSUFFICIENT_EVALUABLE_UNITS", "continue_expand": False}
    crop_candidates = list(CROP_ARMS)
    crop_winner = min(
        crop_candidates,
        key=lambda arm: (
            -float(arms[arm]["query_macro_strict_correct"]),
            float(arms[arm]["median_total_elapsed_seconds"]),
            0 if arm == "V2_RAEA_DUAL" else 1,
        ),
    )
    base, winner = arms["V0_FULL_BASE"], arms[crop_winner]
    highres = arms.get("V1_FULL_HIGHRES", {})
    def checks_for(arm: str) -> dict[str, bool]:
        candidate = arms[arm]
        wlt = summary["comparisons_vs_v0"][arm]["wlt"]
        return {
            "strict_gain_ge_2pp": candidate["query_macro_strict_correct"] - base["query_macro_strict_correct"] >= 0.02 - 1e-12,
            "wins_minus_losses_ge_2": wlt["wins"] - wlt["losses"] >= 2,
            "supported_not_down_gt_1pp": candidate["query_macro_supported_value_correct"]
            >= base["query_macro_supported_value_correct"] - 0.01 - 1e-12,
            "hallucinated_support_not_up_gt_1pp": candidate["query_macro_hallucinated_supported_value"]
            <= base["query_macro_hallucinated_supported_value"] + 0.01 + 1e-12,
            "latency_le_2_5x": candidate["median_total_elapsed_seconds"]
            <= 2.5 * base["median_elapsed_generation_seconds"],
        }
    checks = checks_for(crop_winner)
    crop_passed = all(checks.values())
    highres_checks = checks_for("V1_FULL_HIGHRES")
    prefer_highres = (
        highres["query_macro_strict_correct"] >= winner["query_macro_strict_correct"] - 0.01 - 1e-12
        and highres["median_total_elapsed_seconds"] < winner["median_total_elapsed_seconds"]
    )
    if all(highres_checks.values()) and (not crop_passed or prefer_highres):
        return {"status": "STOP_COMPLEX_CROP_USE_HIGHRES", "continue_expand": True,
                "winner": "V1_FULL_HIGHRES", "crop_winner": crop_winner,
                "checks": highres_checks, "crop_checks": checks}
    if crop_passed:
        return {"status": "CONTINUE_EXPAND", "continue_expand": True,
                "winner": crop_winner, "crop_winner": crop_winner, "checks": checks}
    return {
        "status": "STOP_USE_V0",
        "continue_expand": False,
        "winner": "V0_FULL_BASE",
        "crop_winner": crop_winner,
        "checks": checks,
        "highres_checks": highres_checks,
    }
