#!/usr/bin/env python
"""Package the reviewed R10 narrative and frozen metrics as a report artifact.

The Data Analytics portable builder owns HTML rendering and browser validation.
This script only creates its canonical manifest, sources and reviewed snapshot.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CHART_QUERY = '''SELECT model, rule, label, rank, 'test' AS split, 13 AS seed,
 count(*) AS queries,
 sum(json_extract(q.value, '$.implicit_pair_count')) AS implicit_pairs,
 avg(json_extract(q.value, '$."recall@10"')) AS recall_at_10,
 avg(json_extract(q.value, '$."recall@20"')) AS recall_at_20,
 avg(json_extract(q.value, '$."recall@50"')) AS recall_at_50,
 avg(json_extract(q.value, '$."mrr@50"')) AS mrr_at_50,
 sum(json_extract(q.value, '$."valid_path_recall@10,4"') * json_extract(q.value, '$.implicit_pair_count'))
   / sum(json_extract(q.value, '$.implicit_pair_count')) AS valid_path_at_10_4,
 sum(json_extract(q.value, '$."row_support_coverage@10,4"') * json_extract(q.value, '$.implicit_pair_count'))
   / sum(json_extract(q.value, '$.implicit_pair_count')) AS row_support_at_10_4
FROM frozen_evaluations AS e, json_each(e.payload, '$.per_query') AS q
GROUP BY model, rule, label, rank ORDER BY rank'''

TABLE_QUERY = 'SELECT table_id, row_json FROM reviewed_report_rows ORDER BY table_id, row_index'


def markdown_cells(line: str) -> list[str]:
    return [cell.strip().replace("**", "").replace("`", "") for cell in line.strip().strip("|").split("|")]


def build(r10: Path, output: Path) -> dict[str, Any]:
    text = (r10 / "FINAL.md").read_text(encoding="utf-8")
    summary = json.loads((r10 / "final_summary.json").read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).isoformat()
    repo = r10.parent.parent
    rel = r10.relative_to(repo).as_posix()
    title = text.splitlines()[0].removeprefix("# ")
    evidence = ["FINAL.md", "final_summary.json", "taskA_protocol/inputs.json", "taskA_protocol/label_audit.json",
                "taskD_edge_labels/corruption_overlap_audit.json", "taskD_student_ablation/RESULTS.md",
                "taskD4_continuation/RESULTS.md", "taskB_g5/d4_mined_endpoint_exact_content/RESULTS.md",
                "stage1_closeout/frozen_protocol.json"]
    for pattern in ("stage1_closeout/evaluation/*.json", "taskC_pr_ablation/**/student_path.pt.selection.json",
                    "taskE_matched/**/student_path.pt.history.json", "taskE_student_controls/*/student_path.pt.selection.json",
                    "taskF_fusion/*/RESULTS.md"):
        evidence.extend(path.relative_to(r10).as_posix() for path in sorted(r10.glob(pattern)))
    for path in evidence:
        if not (r10 / path).is_file():
            raise FileNotFoundError(path)
    sources = [
        {
            "id": "reviewed_report", "label": "R10 reviewed research report and experimental evidence",
            "path": f"{rel}/FINAL.md",
            "query": {"engine": "local experiment artifacts", "language": "markdown",
                      "description": "Reviewed A–F dev narrative, fixed test results, conditional seed repeats, and limitations.",
                      "tables_used": [f"{rel}/{path}" for path in evidence],
                      "filters": ["EntiTables v9; no Stage-2 outcome claims; no WDC substitute"],
                      "metric_definitions": {"R@K": "Macro-average over queries of retrieved positive targets / all positive targets",
                                             "ValidPath@K,4": "Fraction of implicit positive query-target pairs retrieved with confirmed supporting evidence (B=4)",
                                             "RowSupport@K,4": "Mean over implicit positive pairs of distinct supported query rows / 5; missed target contributes 0"}},
        },
        {
            "id": "frozen_metrics", "label": "Frozen Stage-1 evaluation and source-group bootstrap",
            "path": f"{rel}/final_summary.json",
            "query": {"engine": "SQLite", "language": "sql", "sql": CHART_QUERY,
                      "description": "In-memory frozen_evaluations contains the selected rule result JSON from each original evaluation file. The query independently recomputes query means and pair-weighted path metrics; values are reconciled to final_summary.json. Bootstrap is computed separately by src/summarize_stage1_r10_closeout.py.",
                      "tables_used": ["main.frozen_evaluations", *[f"{rel}/stage1_closeout/evaluation/{name}_test.json" for name in
                                      ("raw", "pca_epoch0", "r5_repro", "c3a", "c4", "e01_s13", "p_frozen_s13")]],
                      "filters": ["split=test; seed=13; frozen before test; K=10; B=4"],
                      "metric_definitions": {"recall_at_10": "Query-macro recall, denominator 1166 queries",
                                             "valid_path_at_10_4": "Confirmed supported implicit positive-pair fraction, denominator 646 pairs",
                                             "row_support_at_10_4": "Pair-macro supported-row fraction, 5 rows per query"}},
        },
        {
            "id": "reviewed_tables", "label": "Reviewed research tables (exact report cells)",
            "path": f"{rel}/FINAL.md",
            "query": {"engine": "SQLite", "language": "sql", "sql": TABLE_QUERY,
                      "description": "src/build_stage1_r10_report.py materializes Markdown table rows from FINAL.md in an in-memory SQLite table, then reads them in report order. This preserves reviewed cells; it does not independently re-estimate historical results. Raw experimental sources are listed for audit.",
                      "tables_used": ["main.reviewed_report_rows", *[f"{rel}/{path}" for path in evidence]],
                      "filters": ["All 15 reviewed report tables; original displayed precision retained"]},
        },
    ]
    manifest: dict[str, Any] = {
        "version": 1, "surface": "report", "title": title,
        "description": "EntiTables v9 · Stage-1 · 2026-09-08 · 固定快照，Stage-2 已停止",
        "generatedAt": stamp, "cards": [], "charts": [], "tables": [], "sources": sources,
        "blocks": [{"id": "title", "type": "markdown", "body": f"# {title}"}],
    }
    datasets: dict[str, list[dict[str, Any]]] = {}
    comparisons = [(name, "f2_rrf_e005", label) for name, label in
                   (("raw", "Raw / F2"), ("pca_epoch0", "PCA / F2"), ("r5_repro", "r5 / F2"),
                    ("c3a", "C3a / F2"), ("c4", "C4 / F2"), ("e01_s13", "E01 / F2"))]
    comparisons += [("p_frozen_s13", rule, label) for rule, label in
                    (("f0_direct", "P-frozen / F0"), ("f2_rrf_e005", "P-frozen / F2"), ("f4_lambda_0.5", "P-frozen / F4"))]
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("CREATE TABLE frozen_evaluations (model TEXT, rule TEXT, label TEXT, rank INTEGER, payload TEXT)")
    connection.execute("CREATE TABLE reviewed_report_rows (table_id TEXT, row_index INTEGER, row_json TEXT)")
    for index, (name, rule, label) in enumerate(comparisons):
        payload = json.loads((r10 / "stage1_closeout/evaluation" / f"{name}_test.json").read_text(encoding="utf-8"))
        connection.execute("INSERT INTO frozen_evaluations VALUES (?, ?, ?, ?, ?)",
                           (name, rule, label, index + 1, json.dumps(payload["results"][rule])))
    datasets["test_comparisons"] = [dict(row) for row in connection.execute(CHART_QUERY)]
    for row in datasets["test_comparisons"]:
        expected = summary["entries"][row["model"] + "__test"][row["rule"]]
        for metric in ("recall_at_10", "valid_path_at_10_4", "row_support_at_10_4"):
            if abs(row[metric] - expected[metric]) > 1e-10:
                raise ValueError(f"SQL metric mismatch: {row['model']}/{metric}")
    for metric, label in (("recall_at_10", "目标召回 R@10"), ("valid_path_at_10_4", "有效路径 ValidPath@10,4")):
        manifest["charts"].append({
            "id": metric, "title": f"冻结 test：{label}", "type": "horizontalBar",
            "subtitle": "seed 13；目标召回按 1,166 queries，有效路径按 646 implicit 正对汇总",
            "dataset": "test_comparisons", "sourceId": "frozen_metrics", "valueFormat": "percent",
            "encodings": {
                "x": {"field": "label", "type": "nominal", "label": "模型 / 固定融合"},
                "y": {"field": metric, "type": "quantitative", "label": label},
                "tooltip": [{"field": "row_support_at_10_4", "type": "quantitative", "label": "行覆盖", "format": "percent"},
                            {"field": "queries", "type": "quantitative", "label": "queries"},
                            {"field": "implicit_pairs", "type": "quantitative", "label": "implicit 正对"}],
            },
        })
    sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)[1:]
    table_count = 0
    for section_index, section in enumerate(sections):
        lines = section.strip().splitlines()
        buffer = []
        subblock = 0

        def flush() -> None:
            nonlocal subblock
            if buffer:
                manifest["blocks"].append({"id": f"section_{section_index}_{subblock}", "type": "markdown",
                                           "body": "\n".join(buffer).strip(), "sourceId": "reviewed_report"})
                buffer.clear()
                subblock += 1

        index = 0
        while index < len(lines):
            line = lines[index]
            if not line.startswith("|"):
                buffer.append(line)
                index += 1
                continue
            flush()
            if lines[0].startswith("## 2.") and not any(block.get("chartId") == "recall_at_10" for block in manifest["blocks"]):
                for chart_id, body in (
                    ("recall_at_10", "横条从零开始，比较九个预先固定模型/融合配置的总体目标召回。E01 明显退化；P-frozen F4 比自身 F0 和 r5 都低。图中参考模型是单 seed，不据 test 再选择配置。"),
                    ("valid_path_at_10_4", "相同候选的有效路径排序不同于总体召回：P-frozen F4 相比自身 F0 有局部增益，但仍低于 Raw/r5/C4。此图分母为 implicit 正对；有效路径不等于成功补值。"),
                ):
                    manifest["blocks"].append({"id": f"note_{chart_id}", "type": "markdown", "body": body, "sourceId": "frozen_metrics"})
                    manifest["blocks"].append({"id": f"chart_{chart_id}", "type": "chart", "chartId": chart_id})
            headers = markdown_cells(line)
            index += 2  # Header followed by CommonMark separator row.
            rows = []
            while index < len(lines) and lines[index].startswith("|"):
                cells = markdown_cells(lines[index])
                if len(cells) != len(headers):
                    raise ValueError(f"Malformed report table: {lines[index]}")
                rows.append({"order": len(rows) + 1, **{f"c{i}": value for i, value in enumerate(cells)}})
                index += 1
            table_id = f"table_{table_count}"
            table_count += 1
            connection.executemany("INSERT INTO reviewed_report_rows VALUES (?, ?, ?)",
                                   [(table_id, row["order"], json.dumps(row, ensure_ascii=False)) for row in rows])
            datasets[table_id] = []
            manifest["tables"].append({
                "id": table_id, "title": lines[0].removeprefix("## "), "dataset": table_id,
                "sourceId": "reviewed_tables", "defaultSort": {"field": "order", "direction": "asc"},
                "columns": [{"field": "order", "label": "序", "format": "number"},
                            *[{"field": f"c{i}", "label": header, "type": "text"} for i, header in enumerate(headers)]],
            })
            manifest["blocks"].append({"id": f"block_{table_id}", "type": "table", "tableId": table_id})
        flush()
    for row in connection.execute(TABLE_QUERY):
        datasets[row["table_id"]].append(json.loads(row["row_json"]))
    connection.close()
    artifact = {"surface": "report", "manifest": manifest,
                "snapshot": {"version": 1, "generatedAt": stamp, "status": "ready", "datasets": datasets, "accessIssues": []},
                "sources": sources}
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(output), "charts": len(manifest["charts"]), "tables": table_count, "sections": len(sections)}))
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r10-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.r10_root.resolve(), args.output.resolve())
