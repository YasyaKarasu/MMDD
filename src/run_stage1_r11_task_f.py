#!/usr/bin/env python
"""Evaluate R11 F0-F5 fusion on fixed dev and cal-fit path pools."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Sequence

import torch
from mmdd_progress import progress

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import checkpoint_fingerprint, fuse_ranked_channels
from mmdd_stage1.row_support import load_evidence_content_keys
from mmdd_stage1.selection import load_stage1_selection
from run_stage1_r11_task_e import (
    INTERVENTIONS,
    empty_intervention_stats,
    finalize_intervention_stats,
    intervene_paths,
    select_evidence,
)


RECALL_KS = (10, 20, 50)
FUSION_IDS = (
    "f0_d100_direct",
    "f1_union_direct",
    "f2_d100_rrf_e005",
    "f3_union_rrf_equal",
    "f4_lambda_0.25",
    "f4_lambda_0.5",
    "f5_reserved_half",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty channel")
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _scale(values: Sequence[float]) -> dict[str, float | int]:
    q10 = _percentile(values, 0.1)
    q50 = _percentile(values, 0.5)
    q90 = _percentile(values, 0.9)
    return {
        "count": len(values),
        "q10": q10,
        "q50": q50,
        "q90": q90,
        "denominator": max(q90 - q10, 1e-6),
    }


def _z(value: float, scale: dict[str, float | int]) -> float:
    return (float(value) - float(scale["q50"])) / float(scale["denominator"])


class DirectScorer:
    def __init__(
        self,
        metadata: dict[str, Any],
        store: FeatureStore,
        device: torch.device,
    ) -> None:
        self.store = store
        self.device = device
        self.system = str(metadata["system"])
        self.student = None
        self.selection_path = Path(metadata["selection"])
        self.checkpoint = None
        self.checkpoint_sha256 = None
        self.scored_pairs = 0
        if self.system == "student":
            selection = load_stage1_selection(self.selection_path)
            self.checkpoint = Path(selection["best_checkpoint"])
            self.checkpoint_sha256 = checkpoint_fingerprint(self.checkpoint)
            if self.checkpoint_sha256 != metadata.get("student_checkpoint_sha256"):
                raise ValueError("Path pool and Student checkpoint fingerprints differ")
            self.student = load_student(self.checkpoint, device).eval()
        elif self.system != "raw":
            raise ValueError(f"Unsupported path-pool system: {self.system}")

    def score(
        self, query_id: str, target_ids: Sequence[str], *, batch_size: int
    ) -> dict[str, float]:
        self.scored_pairs += len(target_ids)
        query = self.store.embedding_features(query_id)
        if self.student is None:
            query_embedding = query.embedding.detach().cpu().float()
            return {
                target_id: float(
                    torch.dot(
                        query_embedding,
                        self.store.embedding_features(target_id)
                        .embedding.detach()
                        .cpu()
                        .float(),
                    )
                )
                for target_id in target_ids
            }
        query_features = query.for_scoring(self.device, include_hidden=False)
        result = {}
        with torch.inference_mode():
            for start in range(0, len(target_ids), batch_size):
                batch_ids = list(target_ids[start : start + batch_size])
                targets = [
                    self.store.embedding_features(target_id).for_scoring(
                        self.device, include_hidden=False
                    )
                    for target_id in batch_ids
                ]
                scores = self.student.score_pairs_in_space(
                    [query_features] * len(targets), targets, "raw_logit"
                )
                result.update(
                    (target_id, float(score))
                    for target_id, score in zip(batch_ids, scores.detach().cpu())
                )
        return result


def _target_channels(
    record: dict[str, Any],
    *,
    retention: str,
    scorer: DirectScorer,
    store: FeatureStore,
    content_keys: dict[str, str],
    top_l: int,
    evidence_budget: int,
    pair_batch_size: int,
    intervention: str,
    intervention_stats: dict[str, float | int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    query_id = str(record["query_id"])
    union_ids = sorted(str(value) for value in record["paths_by_target"])
    direct_scores = scorer.score(query_id, union_ids, batch_size=pair_batch_size)
    support_cache: dict[str, list[float]] = {}
    direct = []
    evidence = []
    for target_id in union_ids:
        paths = intervene_paths(
            record["paths_by_target"][target_id],
            intervention=intervention,
            query_id=query_id,
            store=store,
            support_cache=support_cache,
            stats=intervention_stats,
        )
        selected, evidence_score = select_evidence(
            retention,
            paths,
            query_id=query_id,
            store=store,
            content_keys=content_keys,
            top_l=top_l,
            budget=evidence_budget,
            support_cache=support_cache,
        )
        row = {
            "target_id": target_id,
            "direct_score": direct_scores[target_id],
            "evidence_score": evidence_score,
            "selected_evidence_ids": selected,
            "paths": paths,
        }
        if any(path.get("kind") == "direct" for path in paths):
            direct.append(row)
        if evidence_score is not None:
            evidence.append(row)
    direct.sort(key=lambda row: (-float(row["direct_score"]), str(row["target_id"])))
    evidence.sort(
        key=lambda row: (-float(row["evidence_score"]), str(row["target_id"]))
    )
    return direct, evidence


def _union_direct(
    direct: Sequence[dict[str, Any]], evidence: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_target = {
        str(row["target_id"]): row for row in [*direct, *evidence]
    }
    return sorted(
        by_target.values(),
        key=lambda row: (-float(row["direct_score"]), str(row["target_id"])),
    )


def _linear_fusion(
    direct: Sequence[dict[str, Any]],
    evidence: Sequence[dict[str, Any]],
    *,
    direct_scale: dict[str, float | int],
    evidence_scale: dict[str, float | int],
    evidence_weight: float,
) -> list[dict[str, Any]]:
    evidence_by_target = {
        str(row["target_id"]): float(row["evidence_score"]) for row in evidence
    }
    result = []
    for row in direct:
        target_id = str(row["target_id"])
        direct_z = _z(float(row["direct_score"]), direct_scale)
        evidence_z = (
            _z(evidence_by_target[target_id], evidence_scale)
            if target_id in evidence_by_target
            else 0.0
        )
        score = (1.0 - evidence_weight) * direct_z + evidence_weight * evidence_z
        result.append(
            {
                **row,
                "score": score,
                "direct_z": direct_z,
                "evidence_z": evidence_z,
            }
        )
    return sorted(
        result, key=lambda row: (-float(row["score"]), str(row["target_id"]))
    )


def _take_unique(
    ranking: Sequence[dict[str, Any]], selected: set[str], count: int
) -> tuple[list[dict[str, Any]], int]:
    rows = []
    consumed = 0
    for consumed, row in enumerate(ranking, 1):
        target_id = str(row["target_id"])
        if target_id in selected:
            continue
        selected.add(target_id)
        rows.append(row)
        if len(rows) == count:
            break
    return rows, consumed


def reserved_channel_fusion(
    direct: Sequence[dict[str, Any]],
    evidence: Sequence[dict[str, Any]],
    *,
    k: int,
) -> list[dict[str, Any]]:
    selected: set[str] = set()
    direct_rows, direct_index = _take_unique(direct, selected, (k + 1) // 2)
    evidence_rows, evidence_index = _take_unique(evidence, selected, k // 2)
    result = []
    for index in range(max(len(direct_rows), len(evidence_rows))):
        if index < len(direct_rows):
            result.append(direct_rows[index])
        if index < len(evidence_rows):
            result.append(evidence_rows[index])
    streams = (iter(direct[direct_index:]), iter(evidence[evidence_index:]))
    exhausted = [False, False]
    turn = 0
    while len(result) < k and not all(exhausted):
        current = turn % 2
        turn += 1
        if exhausted[current]:
            continue
        for row in streams[current]:
            target_id = str(row["target_id"])
            if target_id not in selected:
                selected.add(target_id)
                result.append(row)
                break
        else:
            exhausted[current] = True
    return [{**row, "score": 1.0 / rank} for rank, row in enumerate(result, 1)]


def _empty() -> dict[str, Any]:
    return {
        "query_recall": {k: [] for k in RECALL_KS},
        "query_recall_by_kind": {
            kind: {k: [] for k in RECALL_KS} for kind in ("implicit", "explicit")
        },
        "query_recall_by_multiplicity": {
            kind: {k: [] for k in RECALL_KS} for kind in ("single", "multiple")
        },
        "valid_path": {k: 0 for k in RECALL_KS},
        "row_support": {k: [] for k in RECALL_KS},
        "multi_row_2": {k: 0 for k in RECALL_KS},
        "multi_row_3": {k: 0 for k in RECALL_KS},
        "valid_discovery": {k: 0 for k in RECALL_KS},
        "positive_rescued_vs_f1": {k: 0 for k in RECALL_KS},
        "positive_displaced_vs_f1": {k: 0 for k in RECALL_KS},
        "implicit_positive_pairs": 0,
        "raw_d100_outside_implicit_positive_pairs": 0,
        "per_query": [],
    }


def _selected_valid_rows(
    record: dict[str, Any], target: dict[str, Any] | None, target_id: str
) -> tuple[set[str], set[int]]:
    if target is None:
        return set(), set()
    selected = {str(value) for value in target.get("selected_evidence_ids", [])}
    expected = {
        str(value)
        for value in record.get("positive_evidence_by_target", {}).get(target_id, [])
    }
    valid = selected & expected
    rows_by_evidence = record.get("positive_evidence_rows_by_target", {}).get(
        target_id, {}
    )
    rows = set().union(
        *(
            {int(row) for row in rows_by_evidence.get(evidence_id, [])}
            for evidence_id in valid
        )
    )
    return valid, rows


def _accumulate(
    metrics: dict[str, Any],
    record: dict[str, Any],
    rankings: dict[int, Sequence[dict[str, Any]]],
    f1_rankings: dict[int, Sequence[dict[str, Any]]],
    raw_d100: set[str],
) -> None:
    positives = {str(value) for value in record["positive_target_ids"]}
    kind = str(record.get("query_kind", "unknown"))
    multiplicity = "single" if len(positives) == 1 else "multiple"
    implicit_targets = (
        {str(value) for value in record.get("positive_evidence_by_target", {})}
        if kind == "implicit"
        else set()
    )
    metrics["implicit_positive_pairs"] += len(implicit_targets)
    metrics["raw_d100_outside_implicit_positive_pairs"] += len(
        implicit_targets - raw_d100
    )
    per_query = {"query_id": str(record["query_id"]), "query_kind": kind}
    row_count = int(record.get("query_row_count") or 0)
    for k in RECALL_KS:
        ranking = list(rankings[k])
        top = {str(row["target_id"]) for row in ranking[:k]}
        f1_top = {str(row["target_id"]) for row in f1_rankings[k][:k]}
        by_target = {str(row["target_id"]): row for row in ranking[:k]}
        recall = len(top & positives) / len(positives)
        metrics["query_recall"][k].append(recall)
        if kind in metrics["query_recall_by_kind"]:
            metrics["query_recall_by_kind"][kind][k].append(recall)
        metrics["query_recall_by_multiplicity"][multiplicity][k].append(recall)
        metrics["positive_rescued_vs_f1"][k] += len((top - f1_top) & positives)
        metrics["positive_displaced_vs_f1"][k] += len((f1_top - top) & positives)
        valid_count = 0
        discovery_count = 0
        row_values = []
        multi2 = 0
        multi3 = 0
        for target_id in implicit_targets:
            valid, rows = _selected_valid_rows(
                record, by_target.get(target_id), target_id
            )
            valid_count += int(bool(valid))
            discovery_count += int(
                target_id not in raw_d100 and target_id in top and bool(valid)
            )
            row_values.append(len(rows) / row_count if row_count else 0.0)
            multi2 += int(len(rows) >= 2)
            multi3 += int(len(rows) >= 3)
        metrics["valid_path"][k] += valid_count
        metrics["valid_discovery"][k] += discovery_count
        metrics["row_support"][k].extend(row_values)
        metrics["multi_row_2"][k] += multi2
        metrics["multi_row_3"][k] += multi3
        per_query[f"recall@{k}"] = recall
        per_query[f"valid_path_count@{k},4"] = valid_count
        per_query[f"valid_discovery_count@{k},4"] = discovery_count
        per_query[f"row_support_sum@{k},4"] = sum(row_values)
    metrics["per_query"].append(per_query)


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _finalize(metrics: dict[str, Any]) -> dict[str, Any]:
    denominator = int(metrics["implicit_positive_pairs"])
    outside = int(metrics["raw_d100_outside_implicit_positive_pairs"])
    result: dict[str, Any] = {
        "queries": len(metrics["per_query"]),
        "implicit_positive_pairs": denominator,
        "raw_d100_outside_implicit_positive_pairs": outside,
        "per_query": metrics["per_query"],
    }
    for k in RECALL_KS:
        result[f"recall@{k}"] = _mean(metrics["query_recall"][k])
        result[f"implicit_recall@{k}"] = _mean(
            metrics["query_recall_by_kind"]["implicit"][k]
        )
        result[f"explicit_recall@{k}"] = _mean(
            metrics["query_recall_by_kind"]["explicit"][k]
        )
        result[f"single_positive_recall@{k}"] = _mean(
            metrics["query_recall_by_multiplicity"]["single"][k]
        )
        result[f"multiple_positive_recall@{k}"] = _mean(
            metrics["query_recall_by_multiplicity"]["multiple"][k]
        )
        result[f"valid_path_count@{k},4"] = metrics["valid_path"][k]
        result[f"valid_path@{k},4"] = (
            metrics["valid_path"][k] / denominator if denominator else 0.0
        )
        result[f"row_support@{k},4"] = _mean(metrics["row_support"][k])
        result[f"multi_row_2@{k},4"] = (
            metrics["multi_row_2"][k] / denominator if denominator else 0.0
        )
        result[f"multi_row_3@{k},4"] = (
            metrics["multi_row_3"][k] / denominator if denominator else 0.0
        )
        result[f"valid_discovery_count@{k},4"] = metrics["valid_discovery"][k]
        result[f"valid_discovery@{k},4"] = (
            metrics["valid_discovery"][k] / denominator if denominator else 0.0
        )
        result[f"valid_discovery_given_raw_d100_outside@{k},4"] = (
            metrics["valid_discovery"][k] / outside if outside else 0.0
        )
        result[f"positive_rescued_vs_f1@{k}"] = metrics[
            "positive_rescued_vs_f1"
        ][k]
        result[f"positive_displaced_vs_f1@{k}"] = metrics[
            "positive_displaced_vs_f1"
        ][k]
    return result


def _raw_d100_by_query(path: Path) -> dict[str, set[str]]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            result[str(record["query_id"])] = {
                str(target_id)
                for target_id, paths in record["paths_by_target"].items()
                if any(path.get("kind") == "direct" for path in paths)
            }
    return result


def _calibration_scales(
    path: Path,
    *,
    retention: str,
    scorer: DirectScorer,
    store: FeatureStore,
    content_keys: dict[str, str],
    top_l: int,
    evidence_budget: int,
    pair_batch_size: int,
    intervention: str,
    intervention_stats: dict[str, float | int],
) -> dict[str, dict[str, float | int]]:
    direct_values = []
    evidence_values = []
    with path.open(encoding="utf-8") as handle:
        for line in progress(handle, desc="Fit R11 F channel scales", unit="query"):
            record = json.loads(line)
            direct, evidence = _target_channels(
                record,
                retention=retention,
                scorer=scorer,
                store=store,
                content_keys=content_keys,
                top_l=top_l,
                evidence_budget=evidence_budget,
                pair_batch_size=pair_batch_size,
                intervention=intervention,
                intervention_stats=intervention_stats,
            )
            direct_values.extend(float(row["direct_score"]) for row in _union_direct(direct, evidence))
            evidence_values.extend(float(row["evidence_score"]) for row in evidence)
    return {"direct": _scale(direct_values), "evidence": _scale(evidence_values)}


def _quality_selection(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline = results["f1_union_direct"]
    eligible = []
    for name in FUSION_IDS:
        if name in {"f0_d100_direct", "f1_union_direct", "f2_d100_rrf_e005"}:
            continue
        row = results[name]
        overall_drop = baseline["recall@10"] - row["recall@10"]
        implicit_drop = baseline["implicit_recall@10"] - row["implicit_recall@10"]
        if overall_drop <= 0.02 + 1e-12 and implicit_drop <= 0.02 + 1e-12:
            eligible.append(name)
    selected = (
        min(
            eligible,
            key=lambda name: (
                -results[name]["row_support@10,4"],
                -results[name]["multi_row_2@10,4"],
                -results[name]["valid_path@10,4"],
                -results[name]["recall@10"],
                name,
            ),
        )
        if eligible
        else None
    )
    return {
        "quality_tolerance_absolute": 0.02,
        "eligible": eligible,
        "selected": selected,
        "comparison_order": [
            "row_support@10,4",
            "multi_row_2@10,4",
            "valid_path@10,4",
            "recall@10",
            "lower_measured_cost",
            "config_id",
        ],
    }


def _selection_result(
    results: dict[str, dict[str, Any]],
    *,
    evaluation_role: str,
    frozen_selection: str | None,
    frozen_selection_source: Path | None,
) -> dict[str, Any]:
    if evaluation_role == "dev":
        if frozen_selection is None and frozen_selection_source is None:
            return {
                "selection_scope": "dev",
                "reselected": True,
                **_quality_selection(results),
            }
        selection_scope = "dev_fixed_intervention"
    else:
        selection_scope = "r10_test_regression_historical_only"
    if frozen_selection is None or frozen_selection_source is None:
        raise ValueError(
            "A frozen evaluation requires --frozen-selection and "
            "--frozen-selection-source together"
        )
    if frozen_selection not in FUSION_IDS:
        raise ValueError(f"Unknown frozen fusion rule: {frozen_selection}")
    source = json.loads(frozen_selection_source.read_text(encoding="utf-8"))
    if source.get("selection", {}).get("selected") != frozen_selection:
        raise ValueError("Frozen rule does not match the dev selection artifact")
    return {
        "selection_scope": selection_scope,
        "reselected": False,
        "selected": frozen_selection,
        "frozen_selection_source": str(frozen_selection_source.resolve()),
        "frozen_selection_source_sha256": checkpoint_fingerprint(
            frozen_selection_source
        ),
        "eligible": None,
        "comparison_order": None,
        "quality_tolerance_absolute": 0.02,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# R11 Task F: {payload['system']}",
        "",
        "F1/F3/F4/F5 score every target in the D/E union with the same direct ",
        "function. F4 scales are frozen on cal-fit. F5 is allocated independently ",
        "for each K.",
        "",
        "| Rule | R@10 | Implicit R@10 | ValidPath | RowSupport | MultiRow2 | ValidDiscovery | Rescued | Displaced |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in FUSION_IDS:
        row = payload["results"][name]
        lines.append(
            f"| `{name}` | {row['recall@10']:.2%} | "
            f"{row['implicit_recall@10']:.2%} | {row['valid_path@10,4']:.2%} | "
            f"{row['row_support@10,4']:.2%} | {row['multi_row_2@10,4']:.2%} | "
            f"{row['valid_discovery_count@10,4']} | "
            f"{row['positive_rescued_vs_f1@10']} | "
            f"{row['positive_displaced_vs_f1@10']} |"
        )
    lines.extend(
        [
            "",
            f"Quality-eligible rules: `{payload['selection']['eligible']}`.",
            f"Selected rule: `{payload['selection']['selected']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    dev_pool = Path(args.dev_pool).resolve()
    cal_pool = Path(args.cal_fit_pool).resolve()
    raw_reference = Path(args.raw_reference_pool).resolve()
    dev_metadata = json.loads(
        dev_pool.with_suffix(dev_pool.suffix + ".metadata.json").read_text(
            encoding="utf-8"
        )
    )
    cal_metadata = json.loads(
        cal_pool.with_suffix(cal_pool.suffix + ".metadata.json").read_text(
            encoding="utf-8"
        )
    )
    for path, metadata in ((dev_pool, dev_metadata), (cal_pool, cal_metadata)):
        if metadata.get("output_sha256") != checkpoint_fingerprint(path):
            raise ValueError(f"{path}: path-pool fingerprint mismatch")
    if dev_metadata["system"] != cal_metadata["system"]:
        raise ValueError("Dev and cal-fit pools must use the same system")
    if dev_metadata.get("student_checkpoint_sha256") != cal_metadata.get(
        "student_checkpoint_sha256"
    ):
        raise ValueError("Dev and cal-fit pools must use the same checkpoint")
    expected_eval_split = "dev" if args.evaluation_role == "dev" else "test"
    if (
        dev_metadata["split"] != expected_eval_split
        or cal_metadata["split"] != "train"
    ):
        raise ValueError(
            f"Task F {args.evaluation_role} requires {expected_eval_split} and "
            "train-labelled cal-fit pools"
        )

    content_keys, content_sha256 = load_evidence_content_keys(
        Path(args.content_keys)
    )
    store = FeatureStore.from_path(
        Path(args.features), cache_size=args.feature_cache_size
    )
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    scorer = DirectScorer(dev_metadata, store, device)
    intervention_stats = empty_intervention_stats()
    scales = _calibration_scales(
        cal_pool,
        retention=args.retention,
        scorer=scorer,
        store=store,
        content_keys=content_keys,
        top_l=args.top_l,
        evidence_budget=args.evidence_budget,
        pair_batch_size=args.pair_batch_size,
        intervention=args.intervention,
        intervention_stats=intervention_stats,
    )
    raw_d100 = _raw_d100_by_query(raw_reference)
    metrics = {name: _empty() for name in FUSION_IDS}
    lambda_zero_exact = True
    f2_discovery_boundary = True
    records = 0
    with dev_pool.open(encoding="utf-8") as handle:
        for line in progress(handle, desc="Evaluate R11 Task F", unit="query"):
            record = json.loads(line)
            query_id = str(record["query_id"])
            direct, evidence = _target_channels(
                record,
                retention=args.retention,
                scorer=scorer,
                store=store,
                content_keys=content_keys,
                top_l=args.top_l,
                evidence_budget=args.evidence_budget,
                pair_batch_size=args.pair_batch_size,
                intervention=args.intervention,
                intervention_stats=intervention_stats,
            )
            union = _union_direct(direct, evidence)
            f1 = [{**row, "score": float(row["direct_score"])} for row in union]
            lambda_zero = _linear_fusion(
                union,
                evidence,
                direct_scale=scales["direct"],
                evidence_scale=scales["evidence"],
                evidence_weight=0.0,
            )
            lambda_zero_exact &= [row["target_id"] for row in f1] == [
                row["target_id"] for row in lambda_zero
            ]
            configurations = {
                "f0_d100_direct": [
                    {**row, "score": float(row["direct_score"])} for row in direct
                ],
                "f1_union_direct": f1,
                "f2_d100_rrf_e005": fuse_ranked_channels(
                    list(direct),
                    list(evidence),
                    rrf_k=60,
                    fusion_mode="weighted_rrf",
                    direct_weight=1.0,
                    evidence_weight=0.05,
                ),
                "f3_union_rrf_equal": fuse_ranked_channels(
                    list(union), list(evidence), rrf_k=60, fusion_mode="rrf"
                ),
                "f4_lambda_0.25": _linear_fusion(
                    union,
                    evidence,
                    direct_scale=scales["direct"],
                    evidence_scale=scales["evidence"],
                    evidence_weight=0.25,
                ),
                "f4_lambda_0.5": _linear_fusion(
                    union,
                    evidence,
                    direct_scale=scales["direct"],
                    evidence_scale=scales["evidence"],
                    evidence_weight=0.5,
                ),
            }
            f1_by_k = {k: f1 for k in RECALL_KS}
            for name in FUSION_IDS:
                rankings = (
                    {k: reserved_channel_fusion(union, evidence, k=k) for k in RECALL_KS}
                    if name == "f5_reserved_half"
                    else {k: configurations[name] for k in RECALL_KS}
                )
                _accumulate(
                    metrics[name],
                    record,
                    rankings,
                    f1_by_k,
                    raw_d100[query_id],
                )
            records += 1

    results = {name: _finalize(values) for name, values in metrics.items()}
    f2_discovery_boundary = all(
        results["f2_d100_rrf_e005"][f"valid_discovery_count@{k},4"] == 0
        for k in RECALL_KS
    )
    payload = {
        "format_version": 1,
        "experiment": "R11 Task F union-direct fusion",
        "system": dev_metadata["system"],
        "evaluation_role": args.evaluation_role,
        "retention": args.retention,
        "intervention": args.intervention,
        "intervention_stats": finalize_intervention_stats(intervention_stats),
        "dev_pool": str(dev_pool),
        "dev_pool_sha256": dev_metadata["output_sha256"],
        "cal_fit_pool": str(cal_pool),
        "cal_fit_pool_sha256": cal_metadata["output_sha256"],
        "raw_reference_pool": str(raw_reference),
        "raw_reference_pool_sha256": checkpoint_fingerprint(raw_reference),
        "content_keys_sha256": content_sha256,
        "channel_scales": scales,
        "f4_lambda_zero_matches_f1_exactly": lambda_zero_exact,
        "f2_valid_discovery_zero_boundary": f2_discovery_boundary,
        "queries": records,
        "results": results,
        "selection": _selection_result(
            results,
            evaluation_role=args.evaluation_role,
            frozen_selection=args.frozen_selection,
            frozen_selection_source=(
                Path(args.frozen_selection_source)
                if args.frozen_selection_source is not None
                else None
            ),
        ),
        "cost": {
            "cal_fit_and_dev_union_direct_pair_scores_recomputed": True,
            "union_direct_pair_scores": scorer.scored_pairs,
            "ann_calls_during_fusion": 0,
            "reader_calls": 0,
        },
    }
    output_dir = Path(args.output_dir)
    _write_json(output_dir / "metrics.json", payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics_per_query.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(records):
            handle.write(
                json.dumps(
                    {
                        "query_id": results[FUSION_IDS[0]]["per_query"][index]["query_id"],
                        "systems": {
                            name: results[name]["per_query"][index]
                            for name in FUSION_IDS
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    (output_dir / "RESULTS.md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"status": "pass", "output_dir": str(output_dir)}, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-pool", required=True)
    parser.add_argument("--cal-fit-pool", required=True)
    parser.add_argument("--raw-reference-pool", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--content-keys", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--retention",
        choices=("e0_top_quality", "e1_content_dedup", "e2_row_coverage"),
        required=True,
    )
    parser.add_argument("--top-l", type=int, default=20)
    parser.add_argument("--evidence-budget", type=int, default=4)
    parser.add_argument("--pair-batch-size", type=int, default=256)
    parser.add_argument("--feature-cache-size", type=int, default=60_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--intervention", choices=INTERVENTIONS, default="original_mixed"
    )
    parser.add_argument(
        "--evaluation-role",
        choices=("dev", "r10_test_regression"),
        default="dev",
    )
    parser.add_argument("--frozen-selection", choices=FUSION_IDS)
    parser.add_argument("--frozen-selection-source")
    args = parser.parse_args()
    if min(
        args.top_l,
        args.evidence_budget,
        args.pair_batch_size,
        args.feature_cache_size,
    ) <= 0:
        parser.error("Budgets and batch/cache sizes must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
