#!/usr/bin/env python
"""Package the reviewed R15 narrative and measured evidence as a portable report."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mmdd_stage1.artifacts import write_json


TITLE = "R15: Residual Failure and Evidence Discovery"
ARM_ORDER = ("s_full", "s_eoff", "l_full", "l_eoff", "n_full", "n_eoff")
ARM_LABELS = {
    "s_full": "S-full / B13", "s_eoff": "S-Eoff", "l_full": "L-full",
    "l_eoff": "L-Eoff", "n_full": "N-full", "n_eoff": "N-Eoff",
}

ENDPOINT_SQL = """SELECT
  key AS arm,
  json_extract(value, '$.all.f1_union_direct."10"') AS recall,
  json_extract(value, '$.implicit.f1_union_direct."10"') AS implicit_recall,
  json_extract(value, '$.explicit.f1_union_direct."10"') AS explicit_recall,
  json_extract(value, '$.all.f1_union_direct."20"') AS recall20,
  json_extract(value, '$.all.f1_union_direct."50"') AS candidate_recall50
FROM json_each(:payload, '$.arm_metrics')
ORDER BY CASE key WHEN 's_full' THEN 0 WHEN 's_eoff' THEN 1
  WHEN 'l_full' THEN 2 WHEN 'l_eoff' THEN 3 WHEN 'n_full' THEN 4 ELSE 5 END"""

ADMISSION_SQL = """SELECT key AS rule,
  json_extract(value, '$."CandidateRecall@50"') AS recall,
  json_extract(:payload, '$.metrics.implicit.rules.' || key || '."CandidateRecall@50"') AS implicit_recall,
  json_extract(:payload, '$.metrics.explicit.rules.' || key || '."CandidateRecall@50"') AS explicit_recall,
  0 AS random_repeats, NULL AS monte_carlo_sd
FROM json_each(:payload, '$.metrics.all.rules')
UNION ALL
SELECT key AS rule,
  json_extract(value, '$.all."CandidateRecall@50_mean"') AS recall,
  json_extract(value, '$.implicit."CandidateRecall@50_mean"') AS implicit_recall,
  json_extract(value, '$.explicit."CandidateRecall@50_mean"') AS explicit_recall,
  100 AS random_repeats, json_extract(value, '$.all.monte_carlo_sd') AS monte_carlo_sd
FROM json_each(:payload, '$.random_controls')"""

TIMELINE_SQL = """SELECT :arm AS arm, :step AS step,
  printf('%.4f%%', 100 * json_extract(:payload, '$.full_dev_direct.aggregate.all."exact_recall@10"')) AS exact_recall10,
  printf('%.4f%%', 100 * json_extract(:payload, '$.full_dev_direct.aggregate.all."ann_recall@10"')) AS ann_recall10,
  printf('%.2f%%', 100 * json_extract(:payload, '$.full_dev_direct.aggregate.all."ann_exact_overlap@100"')) AS overlap100"""

CONTRAST_SQL = """SELECT key AS contrast,
  printf('%+.4f', 100 * json_extract(value, '$.all.f1_union_direct."10".point_delta')) AS delta_pp,
  printf('[%+.4f, %+.4f]',
    100 * json_extract(value, '$.all.f1_union_direct."10".ci95_percentile[0]'),
    100 * json_extract(value, '$.all.f1_union_direct."10".ci95_percentile[1]')) AS ci95_pp,
  printf('%d / %d / %d',
    json_extract(value, '$.all.f1_union_direct."10".win_queries'),
    json_extract(value, '$.all.f1_union_direct."10".loss_queries'),
    json_extract(value, '$.all.f1_union_direct."10".tie_queries')) AS win_loss_tie,
  json_extract(value, '$.all.f1_union_direct."10".source_groups') AS source_groups
FROM json_each(:payload, '$.contrasts')
WHERE key IN ('l_eoff_minus_s_eoff', 'n_eoff_minus_l_eoff', 'n_eoff_minus_s_eoff', 'I_L', 'I_N', 'I_N_minus_L')"""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def query_json(payload: dict[str, Any], sql: str, **bindings: Any) -> list[dict[str, Any]]:
    """Execute the report's recorded SQL against the reviewed source JSON."""
    with sqlite3.connect(":memory:") as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(sql, {"payload": json.dumps(payload), **bindings})]


def source(root: Path, identifier: str, label: str, paths: list[Path], description: str, sql: str | None = None) -> dict[str, Any]:
    return {
        "id": identifier, "label": label, "path": str(paths[0].relative_to(root)),
        "query": {
            "engine": "SQLite JSON1; in-memory local research artifact extraction", "language": "sql",
            **({"sql": sql} if sql is not None else {}),
            "parameter_bindings": {"payload": "UTF-8 JSON contents of the declared input file; timeline also binds the checkpoint's arm and step"},
            "description": description,
            "tables_used": [str(path.relative_to(root)) for path in paths],
            "filters": {"split": "dev", "queries": 1198, "Student_seed": 13},
            "metric_definitions": {
                "Recall@K": "Mean over all queries of count(positive targets in first K)/count(all fixed positive targets for that query).",
                "CandidateRecall@50": "Recall over exactly 50 delivered target candidates, before Stage2.",
                "uncertainty": "Paired source-group bootstrap is conditional on fixed artifacts; random-control variation is Monte Carlo, not Student seed variation.",
            },
        },
        "sha256_by_file": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
    }


def make_chart(identifier: str, title: str, dataset: str, source_id: str, question: str) -> dict[str, Any]:
    return {
        "id": identifier, "title": title, "dataset": dataset, "sourceId": source_id,
        "type": "bar", "intent": "comparison", "question": question,
        "rationale": "Horizontal bars compare a small set of prespecified conditions on one common recall denominator; direct category labels identify conditions without a redundant legend.",
        "subtitle": "Dev: 1,198 queries, fixed positive-target denominator; percent",
        "showDescription": True, "valueFormat": "percent", "layout": "full",
        "palette": {"kind": "sequential", "name": "blue"},
        "labels": {"values": "all"},
        "settings": {"orientation": "horizontal", "groupMode": "single", "sort": "none", "showValues": True},
        "comparisonContext": {
            "denominator": "1198 query-macro averages with fixed per-query G_q",
            "grain": "prespecified arm or admission rule", "unit": "fraction displayed as percent",
        },
        "encodings": {
            "x": {"field": "condition", "type": "nominal", "label": "Condition"},
            "y": {"field": "recall", "type": "quantitative", "label": "Recall", "format": "percent"},
            "tooltip": [
                {"field": "queries", "type": "quantitative", "label": "Queries"},
                {"field": "implicit_recall", "type": "quantitative", "label": "Implicit recall", "format": "percent"},
                {"field": "explicit_recall", "type": "quantitative", "label": "Explicit recall", "format": "percent"},
            ],
        },
    }


def build(root: Path) -> Path:
    output = root / "work/stage1_optimization_r15_20260909"
    results_path = output / "RESULTS.md"
    results = results_path.read_text(encoding="utf-8")
    summary_path = output / "statistics/summary.json"
    interaction_path = output / "stageI_interaction/statistics/interaction_summary.json"
    candidate_path = output / "stageC_candidate_delivery/SUMMARY.json"
    gate_path = output / "stageG_correctness/GATE.json"
    interaction = load_json(interaction_path)
    candidate = load_json(candidate_path)
    gate = load_json(gate_path)
    if gate["status"] != "passed" or not candidate["validation"]["saved_rankings_reproduced"]:
        raise ValueError("Report inputs must pass G and saved ranking reproduction")
    endpoint_rows = []
    for row in query_json(interaction, ENDPOINT_SQL):
        arm = row.pop("arm")
        endpoint_rows.append({
            **row, "condition": ARM_LABELS[arm],
            "queries": 1198, "source_groups": 1000, "seed": 13,
            "projection": arm.split("_")[0].upper(), "evidence_loss": "full" in arm,
        })
    admission_rows = []
    admission_by_rule = {row["rule"]: row for row in query_json(candidate, ADMISSION_SQL)}
    for rule, label in (
        ("pure_direct100", "Pure direct"), ("f1_union_direct", "F1 union-direct"),
        ("union_rrf_equal", "Equal RRF"), ("pool_random", "Pool random (100)"),
        ("lake_random", "Lake random (100)"),
    ):
        admission_rows.append({
            **admission_by_rule[rule], "condition": label, "queries": 1198, "source_groups": 1000,
            "candidate_budget": 50,
            "stage2_run": False,
        })
    timeline_rows = []
    timeline_paths = []
    for arm in ("l_full", "l_eoff", "n_full", "n_eoff"):
        base_key = arm[0] + "_eoff"
        for step in (0, 45, 89, 178):
            if arm.endswith("full") or step == 0:
                values = gate["checkpoints"][base_key][f"step_{step:06d}"]
                path = Path(values["path"])
            else:
                path = output / f"stageI_interaction/{arm}_seed13/correctness/step_{step:06d}/metrics.json"
            timeline_paths.append(path)
            timeline_rows.append({
                "order": len(timeline_rows),
                **query_json(load_json(path), TIMELINE_SQL, arm=ARM_LABELS[arm], step=step)[0],
            })
    contrast_rows = []
    contrasts_by_name = {row["contrast"]: row for row in query_json(interaction, CONTRAST_SQL)}
    for name in ("l_eoff_minus_s_eoff", "n_eoff_minus_l_eoff", "n_eoff_minus_s_eoff", "I_L", "I_N", "I_N_minus_L"):
        contrast_rows.append({
            "order": len(contrast_rows), **contrasts_by_name[name],
        })
    sources = [
        source(root, "results", "R15 combined reviewed summary", [summary_path, results_path, interaction_path, candidate_path, gate_path, results_path.parent / "projection_and_optimizer_diagnostics.jsonl.gz", results_path.parent / "stageG_correctness/mean_vectors.jsonl.gz", results_path.parent / "RECOVERED_TRAINING_SOURCE.json", results_path.parent / "RUNTIME_SOURCE_CACHE_AUDIT.json", results_path.parent / "ARCHIVAL_RECOVERY_AUDIT.json", results_path.parent / "ORIGINAL_CASES_VALIDATION.json", root / "B13_evidence_only_known_witness_cases.csv"], "The combined summary materializes the numerical evidence behind the reviewed full technical narrative; the additional files record its upstream provenance, actual fixed-panel F/P/S0 mean vectors, hash-matched training-entrypoint recovery, complete R15 and partial R14 pre-run local bytecode-cache evidence, current original-CSV semantic identity, and remaining archival limits. Historical absence statements are retained as then-current records, not current CSV availability."),
        source(root, "interaction", "R15 interaction statistics", [interaction_path], "Six frozen seed13 arms and paired source-group bootstrap contrasts; no model selection on the report charts.", ENDPOINT_SQL + ";\n\n" + CONTRAST_SQL + ";"),
        source(root, "candidates", "R15 frozen B13 candidate audit", [candidate_path], "Recomputed fixed admission rules and prespecified 100-repeat random controls, all on the same B13 retrieval pool.", ADMISSION_SQL),
        source(root, "exact", "R15 exact and ANN timeline", list(dict.fromkeys(timeline_paths)), "Full dev exact and natural ANN direct retrieval at each saved checkpoint; step0 is byte-identical between each full/Eoff pair. SQL is executed once for each declared per-checkpoint metrics file, not the GATE summary.", TIMELINE_SQL),
    ]
    charts = [
        make_chart("six_arms", "F1 target Recall@10", "endpoint_rows", "interaction", "Does removing the evidence objective rescue either residual projection recipe?"),
        make_chart("candidate_admission", "B13 Candidate Recall@50", "admission_rows", "candidates", "Which fixed N50 admission rules preserve known relevant targets?"),
    ]
    tables = [
        {"id": "exact_timeline", "title": "Direct exact and ANN recall by checkpoint", "dataset": "timeline_rows", "sourceId": "exact",
         "subtitle": "1,198 dev queries; identical legal targets and score function", "showDescription": True,
         "defaultSort": {"field": "order", "direction": "asc"},
         "columns": [{"field": key, "label": label, "type": "number" if key in ("order", "step") else "text"} for key, label in (
             ("order", "Order"), ("arm", "Arm"), ("step", "Update"), ("exact_recall10", "Exact R@10"), ("ann_recall10", "ANN R@10"), ("overlap100", "Top100 overlap"))]},
        {"id": "interaction_table", "title": "Paired F1 Recall@10 contrasts", "dataset": "contrast_rows", "sourceId": "interaction",
         "subtitle": "Differences and 95% source-group bootstrap intervals, percentage points", "showDescription": True,
         "defaultSort": {"field": "order", "direction": "asc"},
         "columns": [{"field": key, "label": label, "type": "text", **({"movement": True} if key == "delta_pp" else {})} for key, label in (
             ("order", "Order"), ("contrast", "Contrast"), ("delta_pp", "Delta (pp)"), ("ci95_pp", "95% interval (pp)"), ("win_loss_tie", "Win / loss / tie"))]},
    ]
    source_by_id = {value["id"]: value for value in sources}
    for block in [*charts, *tables]:
        block["source"] = source_by_id[block["sourceId"]]
    extras = {
        "G": [
            {"id": "timeline_explanation", "type": "markdown", "sourceId": "exact", "body": "### Exact 与 ANN 随更新的对照\n\n下表直接比较四个残差配方保存点的全 dev 检索结果。训练终点的 exact Recall 同样很低，说明单独的近似索引误差不能解释失败；中间点只用于诊断，没有据此更换终点。"},
            {"id": "timeline_block", "type": "table", "tableId": "exact_timeline"},
        ],
        "I": [
            {"id": "endpoint_explanation", "type": "markdown", "sourceId": "interaction", "body": "### 六臂的主端点对照\n\n柱形使用同一全 dev query 宏平均、同一固定正例分母，展示 F1 Recall@10。L/N-Eoff 仍远低于两个原投影对照；E-loss 不是本次残差失败的必要条件。N-Eoff 与 L-Eoff 的差异不能作为健康模型上的非线性收益。"},
            {"id": "endpoint_chart_block", "type": "chart", "chartId": "six_arms"},
        ],
        "interaction": [
            {"id": "contrast_explanation", "type": "markdown", "sourceId": "interaction", "body": "### 交互与配方差异的统计边界\n\n下表保留原 query 权重，并按 source group 联合重采样。区间条件于本次固定初始化、候选、Teacher 与 seed13；它们不是完整训练流程或跨数据湖方差。交互项仍受两个残差臂的端点坍塌影响。"},
            {"id": "contrast_table_block", "type": "table", "tableId": "interaction_table"},
        ],
        "C": [
            {"id": "admission_explanation", "type": "markdown", "sourceId": "candidates", "body": "### 固定 N=50 的候选交付对照\n\n柱形展示真正交付的50个 target 的宏 Recall。池内随机和全湖随机都保持 RRF 的 direct 成员与外部槽位数，图中为预先固定的100次随机重复均值；这不是 Student 重复。RRF 候选交付的点估计更高，但 RRF−F1 的95%区间跨零，尚不能宣布总体交付 Recall 提升。其 Stage1 Top10 低于 F1，最终收益仍需同一 Stage2 验证。"},
            {"id": "admission_chart_block", "type": "chart", "chartId": "candidate_admission"},
        ],
    }
    # Retain every reviewed section; figures are additions in the matching flow.
    normalized = results.replace(str(root) + "/", "")
    normalized = re.sub(r"\A# [^\n]+\n", "", normalized).strip()
    sections = [value.strip() for value in re.split(r"(?m)(?=^## )", normalized) if value.strip()]
    blocks = [{"id": "report_title", "type": "markdown", "body": f"# {TITLE}"}]
    inserted = set()
    section_headings = []
    heading_patterns = {
        "G": r"^##\s+(?:G(?:[\s：:、.．]|$)|Exact 时间线)",
        "I": r"^##\s+(?:I(?:[\s：:、.．]|$)|主端点)",
        "interaction": r"^##\s+配对交互",
        "C": r"^##\s+(?:C(?:[\s：:、.．]|$)|B13)",
    }
    for index, body in enumerate(sections):
        heading = body.splitlines()[0]
        section_headings.append(heading)
        blocks.append({"id": f"narrative_{index:02d}", "type": "markdown", "body": body, "sourceId": "results"})
        for key in extras:
            if key not in inserted and re.search(heading_patterns[key], heading):
                blocks.extend(extras[key])
                inserted.add(key)
    for key in extras:
        if key not in inserted:
            blocks.extend(extras[key])
    generated_at = datetime.now(timezone.utc).isoformat()
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1, "surface": "report", "title": TITLE,
            "description": "R15 seed13 experiment, trained-score audit, and frozen B13 evidence discovery analysis.",
            "generatedAt": generated_at, "blocks": blocks, "charts": charts, "tables": tables,
            "cards": [], "sources": sources,
        },
        "snapshot": {
            "version": 1, "status": "ready", "generatedAt": generated_at,
            "datasets": {"endpoint_rows": endpoint_rows, "admission_rows": admission_rows,
                         "timeline_rows": timeline_rows, "contrast_rows": contrast_rows},
            "accessIssues": [],
        },
        "sources": sources,
    }
    artifact_path = output / "artifact.json"
    write_json(artifact_path, artifact)
    write_json(output / "REPORT_SOURCE_NOTES.json", {
        "delivery_mode": "html", "audience": "technical", "results_sections_preserved": section_headings,
        "required_structure": {
            "title": TITLE, "technical_summary": "RESULTS opening summary", "key_findings": "G/I/C narrative with native evidence",
            "definitions_and_scope": "RESULTS fixed query denominator and experimental protocol", "methodology": "G/I/C experimental design",
            "uncertainty_and_robustness": "RESULTS source-group intervals, exact checks, and null Stage2 verification",
            "recommended_next_steps": "RESULTS stop rules and follow-up intervention", "further_questions": "RESULTS competing explanations and conditional Stage2 verification",
        },
        "chart_map": [
            {"id": "six_arms", "family": "comparison", "type": "bar", "rows": 6, "question": charts[0]["question"], "source_id": "interaction"},
            {"id": "candidate_admission", "family": "comparison", "type": "bar", "rows": 5, "question": charts[1]["question"], "source_id": "candidates"},
        ],
        "same_family_reason": "Both charts compare finite prespecified conditions; checkpoint time series has only four points per arm and is retained as an exact lookup table.",
        "palette_policy": "single-root preferred; blue; category labels and semantic order provide non-color distinctions",
        "quantitative_sections_without_charts": "Geometry, witness retention, bootstrap intervals and checkpoint values are exact audit lookups; tables/narrative preserve their precision.",
        "independent_value_or_join_validation": None,
        "original_case_notebook_execution": {"path": "original_cases_validation.ipynb",
                                              "method": "Four plain-Python cells executed sequentially in one namespace",
                                              "Jupyter_kernel_used": False,
                                              "limitation": "nbformat, nbclient and ipykernel were unavailable; saved outputs are from direct Python execution, not Jupyter-kernel validation"},
        "html_qa": "Use the packaged report:deliver receipt; no additional renderer or browser installation.",
        "source_extraction": "The packaged validator requires SQL for native charts. The saved SQL actually extracts report datasets with SQLite JSON1 from the original reviewed JSON artifacts; no warehouse, network, or model calls are involved. Original file identities and hashes are retained alongside the extraction queries.",
    })
    return artifact_path


def package_report(artifact_path: Path, plugin_root: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["npm", "run", "report:deliver", "--", "--input", str(artifact_path),
         "--output", str(artifact_path.parent / "report.html")],
        cwd=plugin_root, capture_output=True, text=True, check=False,
    )
    output_lines = (result.stdout + "\n" + result.stderr).splitlines()
    receipt = next(json.loads(line) for line in reversed(output_lines) if line.startswith("{"))
    write_json(artifact_path.parent / "report.delivery.json", receipt)
    if result.returncode:
        raise RuntimeError(json.dumps(receipt))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--package", action="store_true")
    parser.add_argument("--plugin-root", type=Path)
    arguments = parser.parse_args()
    artifact = build(arguments.root)
    if arguments.package:
        if arguments.plugin_root is None:
            parser.error("--package requires --plugin-root")
        print(json.dumps(package_report(artifact, arguments.plugin_root)))
    else:
        print(artifact)
