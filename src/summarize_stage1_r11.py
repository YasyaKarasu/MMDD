#!/usr/bin/env python
"""Validate and summarize the completed R11 Stage-1 experiment artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Hashable, Sequence

import numpy as np

from mmdd_stage1.retrieval import checkpoint_fingerprint


BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 13


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _query_groups(dataset_root: Path) -> dict[str, str]:
    result = {}
    for path in sorted((dataset_root / "query_tables").glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                result[str(record["table_id"])] = str(record["source_table_id"])
    if not result:
        raise ValueError(f"No query source groups found under {dataset_root}")
    return result


def grouped_paired_bootstrap(
    left: Sequence[dict[str, Any]],
    right: Sequence[dict[str, Any]],
    groups: dict[str, str],
    *,
    key: Callable[[dict[str, Any]], Hashable],
    query_id: Callable[[dict[str, Any]], str],
    numerator: Callable[[dict[str, Any]], float],
    denominator: Callable[[dict[str, Any]], float],
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    """Bootstrap a paired ratio-of-sums difference by source table."""

    left_by_key = {key(row): row for row in left}
    right_by_key = {key(row): row for row in right}
    if not left_by_key or left_by_key.keys() != right_by_key.keys():
        raise ValueError("Paired bootstrap requires the same non-empty population")
    by_group: dict[str, list[Hashable]] = defaultdict(list)
    for item_key, row in left_by_key.items():
        qid = query_id(row)
        if qid not in groups:
            raise ValueError(f"Missing source group for {qid}")
        by_group[groups[qid]].append(item_key)

    group_names = sorted(by_group)
    differences = np.zeros(len(group_names), dtype=np.float64)
    denominators = np.zeros(len(group_names), dtype=np.float64)
    left_numerator = 0.0
    right_numerator = 0.0
    for group_index, group in enumerate(group_names):
        for item_key in by_group[group]:
            left_row = left_by_key[item_key]
            right_row = right_by_key[item_key]
            left_denominator = float(denominator(left_row))
            right_denominator = float(denominator(right_row))
            if left_denominator != right_denominator or left_denominator < 0:
                raise ValueError("Paired records have unequal or negative denominators")
            left_value = float(numerator(left_row))
            right_value = float(numerator(right_row))
            differences[group_index] += left_value - right_value
            denominators[group_index] += left_denominator
            left_numerator += left_value
            right_numerator += right_value
    total_denominator = float(denominators.sum())
    if total_denominator <= 0:
        raise ValueError("Bootstrap estimand has no eligible denominator")

    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, iterations, 256):
        batch_size = min(256, iterations - start)
        selected = rng.integers(
            len(group_names), size=(batch_size, len(group_names))
        )
        batch_denominators = denominators[selected].sum(axis=1)
        valid = batch_denominators > 0
        estimates.extend(
            (
                differences[selected].sum(axis=1)[valid]
                / batch_denominators[valid]
            ).tolist()
        )
    estimates.sort()
    return {
        "groups": len(group_names),
        "eligible_groups": int(np.count_nonzero(denominators)),
        "records": len(left_by_key),
        "denominator": total_denominator,
        "left_numerator": left_numerator,
        "right_numerator": right_numerator,
        "iterations": iterations,
        "seed": seed,
        "valid_replicates": len(estimates),
        "difference": (left_numerator - right_numerator) / total_denominator,
        "ci95_low": estimates[int(0.025 * (len(estimates) - 1))],
        "ci95_high": estimates[int(0.975 * (len(estimates) - 1))],
    }


def _funnel(value: dict[str, Any]) -> dict[str, int | float]:
    return {
        key: value[key]
        for key in (
            "implicit_positive_pairs",
            "valid_pool_count",
            "valid_pool",
            "valid_b_count",
            "valid_b",
            "row_b",
            "q_to_e_pair_recall",
            "e_to_t_pair_recall_given_q_to_e",
        )
    }


def _stage_funnel(manifest: dict[str, Any], stage: str) -> dict[str, int | float]:
    record = manifest[stage]
    retrieval = record.get("endpoint_retrieval")
    if retrieval is None:
        retrieval = record["endpoint_record"]["dev_retrieval"]
    return _funnel(retrieval["evidence_funnel"])


def _e_metrics(payload: dict[str, Any], strategy: str) -> dict[str, Any]:
    row = payload["results"][strategy]
    return {
        key: row[key]
        for key in (
            "implicit_positive_pairs",
            "valid_b_count",
            "valid_b",
            "row_b",
            "multi_row_2_count",
            "multi_row_2",
            "multi_row_3_count",
            "multi_row_3",
            "selected_by_modality",
            "valid_selected_by_modality",
        )
    }


def _f_metrics(payload: dict[str, Any], rule: str) -> dict[str, Any]:
    row = payload["results"][rule]
    fields = (
        "queries",
        "implicit_positive_pairs",
        "raw_d100_outside_implicit_positive_pairs",
        "recall@10",
        "implicit_recall@10",
        "explicit_recall@10",
        "valid_path_count@10,4",
        "valid_path@10,4",
        "row_support@10,4",
        "multi_row_2@10,4",
        "multi_row_3@10,4",
        "valid_discovery_count@10,4",
        "valid_discovery@10,4",
        "valid_discovery_given_raw_d100_outside@10,4",
        "positive_rescued_vs_f1@10",
        "positive_displaced_vs_f1@10",
    )
    return {key: row[key] for key in fields}


def _path_curve(path: Path) -> list[dict[str, Any]]:
    history = _load(path)
    result = []
    for row in history["epochs"]:
        funnel = row["dev_retrieval"]["evidence_funnel"]
        result.append(
            {
                "epoch": row["epoch"],
                "cumulative_path_updates": row.get("cumulative_optimizer_updates", 0),
                "valid_pool_count": funnel["valid_pool_count"],
                "valid_b_count": funnel["valid_b_count"],
                "row_b": funnel["row_b"],
            }
        )
    return result


def _metric_rows(path: Path, rule: str) -> list[dict[str, Any]]:
    return _load(path)["results"][rule]["per_query"]


def _implicit_denominators(path: Path) -> dict[str, int]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            result[str(record["query_id"])] = (
                len(record.get("positive_evidence_by_target", {}))
                if record.get("query_kind") == "implicit"
                else 0
            )
    return result


def _run_registry(r11: Path) -> list[dict[str, Any]]:
    artifacts = [
        r11 / "taskA_protocol" / "baseline_funnel.json",
        *(r11 / "taskB_historical_mask").glob("*/manifest.json"),
        *(r11 / "taskC_clean").glob("*/manifest.json"),
        *(r11 / "taskD_controls").glob("*/manifest.json"),
        *(r11 / "taskD_controls").glob("*/metrics.json"),
        *(r11 / "taskE_fixed_pool").glob("**/metrics.json"),
        *(r11 / "taskF_fusion").glob("**/metrics.json"),
    ]
    rows = []
    for path in sorted({value.resolve() for value in artifacts if value.is_file()}):
        payload = _load(path)
        relative = path.relative_to(r11.resolve())
        elapsed = sum(
            float(value.get("elapsed_seconds_this_invocation", 0.0))
            for value in payload.values()
            if isinstance(value, dict)
        )
        rows.append(
            {
                "run_id": str(relative.parent).replace("/", "__"),
                "task": relative.parts[0],
                "status": "complete",
                "experiment": payload.get("experiment", "R11 Task A baseline"),
                "artifact": str(path),
                "artifact_sha256": checkpoint_fingerprint(path),
                "arm": payload.get("arm"),
                "regime": payload.get("regime"),
                "seed": payload.get("seed"),
                "system": payload.get("system"),
                "retention": payload.get("retention"),
                "intervention": payload.get("intervention", "original_mixed"),
                "evaluation_role": payload.get(
                    "evaluation_role",
                    "r10_test_regression"
                    if payload.get("split") == "test"
                    else "dev",
                ),
                "elapsed_seconds_recorded": elapsed,
            }
        )
    return rows


def _pct(value: float) -> str:
    return f"{value:.2%}"


def _pp(value: float) -> str:
    return f"{value * 100:+.3f}"


def _write_task_reports(r11: Path, summary: dict[str, Any]) -> None:
    b = summary["task_b"]
    b_ci = summary["bootstrap"]["b1_minus_b0_edge_endpoint_valid_pool"]
    b_lines = [
        "# R11 Task B：历史 global-positive mask 受控复测",
        "",
        "Task B 固定 R10 C4 初始化、R10 T0 Teacher 和历史评估条件，只改变完整正邻居 mask。`evidence_weight=0.05` 仅复现历史辅助融合读数，不进入训练 loss，也不作为 R11 主方法。",
        "",
        "| 臂 | edge 误负例 | edge ValidPool | path ValidPool | path ValidB | path RowB |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ("b0", "b1"):
        row = b[arm]
        b_lines.append(
            f"| `{arm}` | {row['known_positive_as_negative']} | "
            f"{row['edge_endpoint']['valid_pool_count']}/678 | "
            f"{row['path_endpoint']['valid_pool_count']}/678 | "
            f"{row['path_endpoint']['valid_b_count']}/678 | "
            f"{_pct(row['path_endpoint']['row_b'])} |"
        )
    b_lines.extend(
        [
            "",
            f"B1-B0 edge ValidPool 差值为 {_pp(b_ci['difference'])} 个百分点，95% source-table bootstrap CI [{_pp(b_ci['ci95_low'])}, {_pp(b_ci['ci95_high'])}]。修复使误负例归零，但未恢复证据池，结论是“修复必要但不足”。",
            "",
        ]
    )
    (r11 / "taskB_historical_mask" / "RESULTS.md").write_text(
        "\n".join(b_lines), encoding="utf-8"
    )

    c = summary["task_c"]
    c_lines = [
        "# R11 Task C：干净 Teacher 与 Student 排序探针",
        "",
        f"干净 token Teacher：edge macro R@1={_pct(c['teacher']['edge_macro_recall_at_1'])}，path evidence R@1={_pct(c['teacher']['path_evidence_recall_at_1'])}，path direct R@1={_pct(c['teacher']['path_direct_recall_at_1'])}。",
        "",
        "| 探针 | ValidPool | ValidB | RowB |",
        "| --- | ---: | ---: | ---: |",
    ]
    for arm in ("c0", "c1", "c2", "path_only"):
        row = c["probes"][arm]
        c_lines.append(
            f"| `{arm}` | {row['valid_pool_count']}/678 | "
            f"{row['valid_b_count']}/678 | {_pct(row['row_b'])} |"
        )
    c_lines.extend(
        [
            "",
            "C1/C2 按冻结字典序进入长程与多种子复测。长程最终 ValidPool：C1 seed13/17/23=`7/7/8`，C2=`19/18/18`；稳定复现的是训练坍缩，不是成功。C2 seed13 的 path 曲线为 `37 -> 23 -> 19`。连续低于 80% 后额外运行的端点记为计算超跑，不替代共同检查点。",
            "",
        ]
    )
    (r11 / "taskC_clean" / "RESULTS.md").write_text(
        "\n".join(c_lines), encoding="utf-8"
    )

    d = summary["task_d"]
    d3 = d["d3_affine"]
    d_lines = [
        "# R11 Task D：单因素训练与校准对照",
        "",
        "| 臂 | 条件 | ValidPool | ValidB | RowB |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for arm in ("d0_c2_probe", "d1_kd_off_probe", "d2_projection_frozen_probe", "d1_kd_off_long"):
        row = d[arm]
        d_lines.append(
            f"| `{arm}` | {row['condition']} | {row['metrics']['valid_pool_count']}/678 | "
            f"{row['metrics']['valid_b_count']}/678 | {_pct(row['metrics']['row_b'])} |"
        )
    d_lines.extend(
        [
            "",
            f"D3 保持同关系排名不变：`{d3['same_relation_ranking_unchanged']}`。cal_check evidence R@1 从 {_pct(d3['raw_evidence_recall@1'])} 降至 {_pct(d3['affine_evidence_recall@1'])}，equal-RRF R@1 从 {_pct(d3['raw_equal_rrf_recall@1'])} 降至 {_pct(d3['affine_equal_rrf_recall@1'])}，305 个 evidence 列表中 {d3['evidence_top_path_modality_switches']} 个最高路径模态切换；未观察到校准收益。",
            "",
            "D1 长程仍坍缩，KD 不是充分解释；D2 未达到 +2pp 触发条件，未跑 long。没有稳定的长程 Student 目标，因此 Teacher 必要性扩展未被当成主线完成项。",
            "",
        ]
    )
    (r11 / "taskD_controls" / "RESULTS.md").write_text(
        "\n".join(d_lines), encoding="utf-8"
    )

    e = summary["task_e"]
    e_ci = summary["bootstrap"]["raw_e2_minus_e1_row_b"]
    e_lines = [
        "# R11 Task E：固定池证据保留",
        "",
        "| 系统 | 策略 | ValidB | RowB | MultiRow2 | MultiRow3 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for system in ("raw", "d1_probe"):
        for strategy in ("e1_content_dedup", "e2_row_coverage"):
            row = e[system][strategy]
            e_lines.append(
                f"| `{system}` | `{strategy}` | {row['valid_b_count']}/678 | "
                f"{_pct(row['row_b'])} | {row['multi_row_2_count']}/678 | "
                f"{row['multi_row_3_count']}/678 |"
            )
    e_lines.extend(
        [
            "",
            f"Raw E2-E1 RowB 差值为 {_pp(e_ci['difference'])} 个百分点，95% source-table bootstrap CI [{_pp(e_ci['ci95_low'])}, {_pp(e_ci['ci95_high'])}]。Raw E2 的 MultiRow3 比 E1 少 3 对，不能描述为所有覆盖指标均改善。",
            "",
            "Raw 固定池干预：移除 image 后 E2 ValidB `184 -> 174`、RowB `9.56% -> 9.14%`；用预测同一行的非图像证据重复替换后结果与移除 image 相同，说明 exact-content 去重没有把重复票数当作新增行支持。wrong-attribute replacement 未执行，因为 256 条独立审计样本均保持 unknown；E3 和 supervised support predictor 均未触发。",
            "",
        ]
    )
    (r11 / "taskE_fixed_pool" / "RESULTS.md").write_text(
        "\n".join(e_lines), encoding="utf-8"
    )

    f = summary["task_f"]
    f_ci = summary["bootstrap"]["d1_f5_minus_f1_valid_discovery"]
    f_lines = [
        "# R11 Task F：证据准入与多模态干预",
        "",
        "| 系统 | 冻结/选择规则 | R@10 | implicit R@10 | ValidPath | RowSupport | ValidDiscovery |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("raw", "d1_probe", "text_only", "image_only", "r10_test_regression"):
        row = f[name]
        selected = row["selected"]
        values = row["selected_metrics"]
        denominator = values["implicit_positive_pairs"]
        f_lines.append(
            f"| `{name}` | `{selected}` | {_pct(values['recall@10'])} | "
            f"{_pct(values['implicit_recall@10'])} | "
            f"{values['valid_path_count@10,4']}/{denominator} | "
            f"{_pct(values['row_support@10,4'])} | "
            f"{values['valid_discovery_count@10,4']}/{denominator} |"
        )
    f_lines.extend(
        [
            "",
            f"D1 short Student 的 F5-F1 ValidDiscovery 差值为 {_pp(f_ci['difference'])} 个百分点（8/678），95% source-table bootstrap CI [{_pp(f_ci['ci95_low'])}, {_pp(f_ci['ci95_high'])}]。它是短程机制结果，不是稳定长程训练成功。",
            "",
            "Raw mixed F5 通过质量门槛并选中，但 text-only F5 略高于 mixed；image-only 未通过 F5 门槛并选择 F4 lambda=.25。因此本轮不能宣称 text/image 互补已经成立。固定池移除 image 不改变 R@10 或 ValidDiscovery，但 ValidPath `111 -> 110`、RowSupport `6.40% -> 6.34%`，表明图像贡献有限但非零。",
            "",
            "`r10_test_regression` 使用 dev 冻结的 Raw E2+F5，未在 test 重选：R@10=27.07%，implicit R@10=21.04%，ValidDiscovery=8/646（其中 Raw D100 外分母 396）。该集合已参与 R11 设计，只是历史回归，不是独立确认。",
            "",
        ]
    )
    (r11 / "taskF_fusion" / "RESULTS.md").write_text(
        "\n".join(f_lines), encoding="utf-8"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    r11 = root / "work" / "stage1_optimization_r11_20260908"
    dataset = (
        root
        / "output_mm_joinability_entitables_20000_retry100_rounds5_qwen35_final_survivor_context_gaussian_v9"
    )
    groups = _query_groups(dataset)

    baseline = _load(r11 / "taskA_protocol" / "baseline_funnel.json")
    provenance = _load(r11 / "taskA_protocol" / "label_provenance.json")
    teacher = _load(r11 / "taskC_clean" / "teacher" / "manifest.json")
    b_manifests = {
        arm: _load(r11 / "taskB_historical_mask" / arm / "manifest.json")
        for arm in ("b0", "b1")
    }
    c_probes = {
        arm: _load(r11 / "taskC_clean" / arm / "manifest.json")
        for arm in ("c0", "c1", "c2", "path_only")
    }
    c_longs = {
        arm: _load(r11 / "taskC_clean" / arm / "manifest.json")
        for arm in (
            "c1_long",
            "c1_long_seed17",
            "c1_long_seed23",
            "c2_long",
            "c2_long_seed17",
            "c2_long_seed23",
        )
    }
    d1 = _load(r11 / "taskD_controls" / "c2_d1_probe" / "manifest.json")
    d2 = _load(r11 / "taskD_controls" / "c2_d2_probe" / "manifest.json")
    d1_long = _load(r11 / "taskD_controls" / "c2_d1_long" / "manifest.json")
    d3 = _load(r11 / "taskD_controls" / "c2_d3_affine" / "metrics.json")

    e_payloads = {
        name: _load(path)
        for name, path in {
            "raw": r11 / "taskE_fixed_pool" / "raw" / "metrics.json",
            "d1_probe": r11 / "taskE_fixed_pool" / "d1_probe" / "metrics.json",
            "text_only": r11 / "taskE_fixed_pool" / "raw_text40" / "metrics.json",
            "image_only": r11 / "taskE_fixed_pool" / "raw_image40" / "metrics.json",
            "remove_image": r11 / "taskE_fixed_pool" / "interventions" / "raw_remove_image" / "metrics.json",
            "duplicate_same_row": r11 / "taskE_fixed_pool" / "interventions" / "raw_duplicate_same_row" / "metrics.json",
            "r10_test_regression": r11 / "taskE_fixed_pool" / "r10_test_regression_raw" / "metrics.json",
        }.items()
    }
    audit = _load(
        r11 / "taskE_fixed_pool" / "attribute_audit" / "metrics.json"
    )
    f_payloads = {
        name: _load(path)
        for name, path in {
            "raw": r11 / "taskF_fusion" / "raw_e2" / "metrics.json",
            "d1_probe": r11 / "taskF_fusion" / "d1_probe_e2" / "metrics.json",
            "text_only": r11 / "taskF_fusion" / "raw_text40_e2" / "metrics.json",
            "image_only": r11 / "taskF_fusion" / "raw_image40_e2" / "metrics.json",
            "remove_image": r11 / "taskF_fusion" / "interventions" / "raw_remove_image" / "metrics.json",
            "duplicate_same_row": r11 / "taskF_fusion" / "interventions" / "raw_duplicate_same_row" / "metrics.json",
            "r10_test_regression": r11 / "taskF_fusion" / "r10_test_regression_raw_e2_f5" / "metrics.json",
        }.items()
    }

    b_rows = {
        arm: manifest["edge_endpoint_eval"]["endpoint_record"]["dev_retrieval"]
        ["evidence_funnel"]["per_pair"]
        for arm, manifest in b_manifests.items()
    }
    e1_rows = e_payloads["raw"]["results"]["e1_content_dedup"]["per_pair"]
    e2_rows = e_payloads["raw"]["results"]["e2_row_coverage"]["per_pair"]
    f1_rows = _metric_rows(
        r11 / "taskF_fusion" / "d1_probe_e2" / "metrics.json",
        "f1_union_direct",
    )
    f5_rows = _metric_rows(
        r11 / "taskF_fusion" / "d1_probe_e2" / "metrics.json",
        "f5_reserved_half",
    )
    implicit_denominators = _implicit_denominators(
        r11 / "taskE_fixed_pool" / "d1_probe_dev.jsonl"
    )
    f1_bootstrap_rows = [
        {**row, "implicit_pairs": implicit_denominators[row["query_id"]]}
        for row in f1_rows
    ]
    f5_bootstrap_rows = [
        {**row, "implicit_pairs": implicit_denominators[row["query_id"]]}
        for row in f5_rows
    ]
    bootstraps = {
        "b1_minus_b0_edge_endpoint_valid_pool": grouped_paired_bootstrap(
            b_rows["b1"],
            b_rows["b0"],
            groups,
            key=lambda row: (row["query_id"], row["target_id"]),
            query_id=lambda row: str(row["query_id"]),
            numerator=lambda row: float(row["valid_pool"]),
            denominator=lambda _row: 1.0,
        ),
        "raw_e2_minus_e1_row_b": grouped_paired_bootstrap(
            e2_rows,
            e1_rows,
            groups,
            key=lambda row: (row["query_id"], row["target_id"]),
            query_id=lambda row: str(row["query_id"]),
            numerator=lambda row: float(row["row_b"]),
            denominator=lambda _row: 1.0,
        ),
        "d1_f5_minus_f1_valid_discovery": grouped_paired_bootstrap(
            f5_bootstrap_rows,
            f1_bootstrap_rows,
            groups,
            key=lambda row: row["query_id"],
            query_id=lambda row: str(row["query_id"]),
            numerator=lambda row: float(row["valid_discovery_count@10,4"]),
            denominator=lambda row: float(row["implicit_pairs"]),
        ),
    }

    c_probe_metrics = {
        arm: _stage_funnel(
            manifest, "path" if arm == "path_only" else "endpoint_eval"
        )
        for arm, manifest in c_probes.items()
    }
    c_long_metrics = {}
    for name, manifest in c_longs.items():
        history_path = Path(manifest["path"]["selected_checkpoint"] + ".history.json")
        c_long_metrics[name] = {
            "arm": manifest["arm"],
            "seed": manifest["seed"],
            "endpoint": _stage_funnel(manifest, "path"),
            "path_curve": _path_curve(history_path),
        }

    task_f = {}
    for name, payload in f_payloads.items():
        selected = payload["selection"]["selected"]
        task_f[name] = {
            "selected": selected,
            "selection": payload["selection"],
            "f1_union_direct": _f_metrics(payload, "f1_union_direct"),
            "selected_metrics": _f_metrics(payload, selected),
            "intervention_stats": payload.get("intervention_stats"),
        }

    summary = {
        "format_version": 1,
        "experiment": "Stage-1 optimization R11",
        "date": "2026-09-08",
        "protocol": {
            "dev_queries": baseline["systems"]["raw"]["queries"],
            "implicit_positive_pairs": baseline["systems"]["raw"][
                "evidence_funnel"
            ]["implicit_positive_pairs"],
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_unit": "source_table_id",
            "r10_test_regression_is_independent_confirmation": False,
            "independent_confirmation_available": False,
            "transductive_retrieval": True,
        },
        "task_a": {
            "status": provenance["status"],
            "fixed_false_negatives": provenance["in_batch_replay"][
                "fixed_total_known_positive_as_negative"
            ],
            "legacy_false_negatives": provenance["in_batch_replay"][
                "legacy_total_known_positive_as_negative"
            ],
            "raw": _funnel(baseline["systems"]["raw"]["evidence_funnel"]),
            "pca": _funnel(baseline["systems"]["pca"]["evidence_funnel"]),
        },
        "task_b": {
            arm: {
                "known_positive_as_negative": manifest["edge"]["endpoint_record"]
                ["in_batch_expansion"]["known_positive_as_negative"],
                "edge_endpoint": _stage_funnel(manifest, "edge_endpoint_eval"),
                "path_endpoint": _stage_funnel(manifest, "path"),
            }
            for arm, manifest in b_manifests.items()
        },
        "task_c": {
            "teacher": {
                "edge_macro_recall_at_1": teacher["edge"]["endpoint"]["dev_edge"]
                ["macro_recall@1"],
                "path_evidence_recall_at_1": teacher["path"]["endpoint"]
                ["dev_target_lists"]["evidence"]["recall@1"],
                "path_direct_recall_at_1": teacher["path"]["endpoint"]
                ["dev_target_lists"]["direct"]["recall@1"],
            },
            "probes": c_probe_metrics,
            "long_runs": c_long_metrics,
            "long_horizon_stable": False,
            "multi_seed_interpretation": (
                "Student variance under the same fixed seed-13 Teacher; stable "
                "reproduction of failure, not success"
            ),
        },
        "task_d": {
            "d0_c2_probe": {
                "condition": "C2, P/R trainable, KD on",
                "metrics": c_probe_metrics["c2"],
            },
            "d1_kd_off_probe": {
                "condition": "KD=0",
                "metrics": _stage_funnel(d1, "endpoint_eval"),
            },
            "d2_projection_frozen_probe": {
                "condition": "P frozen; R trainable",
                "metrics": _stage_funnel(d2, "endpoint_eval"),
            },
            "d1_kd_off_long": {
                "condition": "KD=0 long horizon",
                "metrics": _stage_funnel(d1_long, "path"),
            },
            "d3_affine": {
                "same_relation_ranking_unchanged": d3[
                    "same_relation_ranking_unchanged"
                ],
                **d3["path_ranking"],
            },
            "teacher_necessity_extension": (
                "not activated as a main-line claim because no stable long-horizon "
                "Student objective was established"
            ),
        },
        "task_e": {
            **{
                name: {
                    strategy: _e_metrics(payload, strategy)
                    for strategy in (
                        "e0_top_quality",
                        "e1_content_dedup",
                        "e2_row_coverage",
                    )
                }
                for name, payload in e_payloads.items()
            },
            "attribute_audit": audit,
            "wrong_attribute_replacement": {
                "executed": False,
                "reason": audit["e3_reason"],
            },
        },
        "task_f": task_f,
        "bootstrap": bootstraps,
        "conclusions": {
            "stable_long_horizon_pr_main_method": False,
            "dev_mechanism_rule": "f5_reserved_half",
            "dev_mechanism_systems": ["raw", "d1_probe_short"],
            "d1_short_is_complete_training_success": False,
            "global_positive_mask": "necessary implementation fix but insufficient",
            "kd": "not a sufficient explanation for long-horizon collapse",
            "affine_calibration": "no path-ranking or fusion benefit",
            "row_coverage": (
                "small positive RowB effect; Raw MultiRow3 tradeoff preserved"
            ),
            "multimodal_complementarity_established": False,
            "image_fixed_pool_contribution": "limited but non-zero",
            "r10_test_regression": "historical adaptive regression only",
            "independent_confirmation": "not available",
            "historical_evidence_weight_0_05": (
                "R10 mechanism-limiting control and auxiliary readout only; not "
                "the R11 primary method"
            ),
        },
    }
    _write_json(r11 / "bootstrap.json", bootstraps)
    _write_json(r11 / "final_summary.json", summary)
    registry = _run_registry(r11)
    runs_path = r11 / "runs.jsonl"
    runs_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in registry),
        encoding="utf-8",
    )
    _write_task_reports(r11, summary)

    final_lines = [
        "# Stage-1 Optimization R11 最终报告",
        "",
        "## 结论",
        "",
        "R11 修复了全局正邻居误作 in-batch 负例的问题，但没有得到稳定的长程 P/R 训练主方法。C1/C2 多种子一致复现长程证据池坍缩；D1 说明关闭 KD 仍会坍缩，D3 单调关系校准也没有带来路径排序收益。不能把短程 checkpoint 或冻结 P 消融包装成完整训练成功。",
        "",
        "论文机制层面仍有受控正证据：E2 在 Raw 固定池上相对 E1 将 RowB 从 9.44% 提高到 9.56%；F5 通过为 evidence 预留一半名额，在 Raw dev 上得到 5 个、D1 short Student 上得到 8 个 Raw D100 外的 ValidDiscovery。固定池移除 image 会损失 10 个 ValidB 正对和 1 个 F5 ValidPath，但 mixed 没有超过等预算 text-only，因此多模态互补尚未建立。",
        "",
        "`evidence_weight=0.05` 只属于 R10 历史机制限制对照及辅助 fused 读数，不是 R11 主方法。R11 的机制主读数来自 E2 行覆盖与 F5 evidence 预留准入。",
        "",
        "## 分项结果",
        "",
        "- Task A：两轮 fixed-mask 误负例为 0；历史 local mask 为 1,816。Raw/PCA ValidPool 为 216/204。",
        "- Task B：B0/B1 edge 端点均为 8/678，path 端点均为 9/678；修复必要但不足。",
        "- Task C：probe C1/C2 ValidPool 为 218/216；长程三种子 C1 为 7/7/8，C2 为 19/18/18。",
        "- Task D：D1 probe 为 220/678，D1 long 为 17/678；D2 probe 为 212/678且未触发 long；D3 无收益。",
        "- Task E：Raw E2 RowB=9.56%、ValidB=184/678、MultiRow2=106/678；D1 short E2 RowB=9.88%、ValidB=190/678。",
        "- Task F：Raw 与 D1 short 均由 dev 选择 F5；Raw R@10=27.57%、ValidDiscovery=5/678，D1 short R@10=26.44%、ValidDiscovery=8/678。",
        "- 历史回归：dev 冻结 Raw E2+F5 在 R10 test regression 上 R@10=27.07%、ValidDiscovery=8/646；该集合不是独立确认。",
        "",
        "## 统计与边界",
        "",
        "三组预注册主比较均使用 10,000 次、seed13、source_table_id 分组配对 bootstrap；区间见 `bootstrap.json`。多种子只刻画固定 seed13 Teacher 下的 Student 方差。属性审计 256 条全部因缺乏独立依据保持 unknown，因此 wrong-attribute replacement、E3 和 supervised support predictor 未执行。没有新的盲测 source groups，本轮结论尚无独立确认。",
        "",
        "详细表见各 Task 的 `RESULTS.md`，机器摘要见 `final_summary.json`，运行索引见 `runs.jsonl`。",
        "",
    ]
    (r11 / "FINAL.md").write_text("\n".join(final_lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "pass",
                "runs": len(registry),
                "output": str(r11 / "FINAL.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default=str(Path(__file__).resolve().parents[1])
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
