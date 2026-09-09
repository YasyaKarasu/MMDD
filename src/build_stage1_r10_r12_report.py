"""Package the reviewed research audit with the shared portable report renderer."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from build_stage1_r10_report import markdown_cells


def build(repo: Path, output: Path, training_path: Path, coverage_path: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    training = json.loads(training_path.read_text())
    coverage = json.loads(coverage_path.read_text())
    # Preserve reviewed calculations and reconcile chart observations to source metrics.
    for rows in (training["c1"], training["c2"]):
        for row in rows:
            source = Path(row["source"])
            if not source.is_absolute():
                source = repo / source
            raw = json.loads(source.read_text())["retrieval"]["evidence_funnel"]
            assert row["valid_pool"] == raw["valid_pool_count"]
            assert row["denominator"] == raw["implicit_positive_pairs"]
            row["source"] = source.relative_to(repo).as_posix()
    for name, payload in (("training_data.json", training), ("coverage_data.json", coverage)):
        (output / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    narrative_path = repo / "docs/stage1_r10_r12_research_analysis_20260909.zh-CN.md"
    narrative = narrative_path.read_text()
    title = narrative.splitlines()[0].removeprefix("# ")
    stamp = datetime.now(timezone.utc).isoformat()
    r10 = "work/stage1_optimization_r10_20260907"
    r11 = "work/stage1_optimization_r11_20260908"
    r12 = "work/stage1_optimization_r12_20260908"
    out = output.relative_to(repo).as_posix()
    inputs = ["方案.md", "AGENTS.md", narrative_path.relative_to(repo).as_posix(),
              f"{r10}/FINAL.md", f"{r10}/REVIEW_20260908.md",
              f"{r11}/FINAL.md", f"{r11}/REVIEW_20260908.md",
              f"{r12}/FINAL.md", f"{r12}/statistics/summary.json",
              f"{r12}/statistics/bootstrap.json", f"{r12}/taskF_end_to_end/metrics.json",
              f"{r12}/taskD_retention/student_selected/original/metrics.json",
              f"{r12}/taskE_admission/student_selected_mixed_d1/metrics.json",
              f"{r12}/taskE_admission/student_selected_text40_d1/metrics.json",
              f"{r12}/taskE_admission/student_selected_image40_d1/metrics.json",
              "src/mmdd_stage2/r12_task_f.py", "src/mmdd_stage2/qwen.py",
              "src/mmdd_stage2/verifier.py", "src/mmdd_stage2/pipeline.py",
              "src/mmdd_stage1/training.py", f"{out}/recomputed_task_f.json"]
    training_sources = sorted({row["source"] for rows in (training["c1"], training["c2"]) for row in rows})
    coverage_sources = [Path(path).relative_to(repo).as_posix()
                        for path in coverage["authoritative_inputs"].values()]
    source_groups = [
        ("audit", "逐轮实验与实现审计", narrative_path.relative_to(repo).as_posix(), inputs,
         "Read plans, results and independent reviews; reconcile key counts and qualify historical protocols."),
        ("training", "R12同预算训练端点", f"{out}/training_data.json", training_sources,
         "Read retrieval.evidence_funnel from saved metrics; integer ValidPool denominator is 678 implicit pairs."),
        ("coverage", "按正对去重的确认恢复行", f"{out}/coverage_data.json", coverage_sources,
         coverage["method"]),
        ("mechanism", "R12原始补值与join输出复算", f"{out}/recomputed_task_f.json",
         [f"{r12}/taskF_end_to_end/{name}" for name in
          ("f_a_predictions.jsonl", "full_chain_predictions.jsonl", "metrics.json")],
         "src/analyze_stage1_r10_r12.py recounts predictions, identical systems once, and reconciles saved metrics."),
    ]
    sources = [{"id": key, "label": label, "path": path,
                "query": {"engine": "local experiment artifacts", "language": "python",
                          "description": description, "tables_used": tables,
                          "filters": ["EntiTables v9; existing artifacts as of 2026-09-09; exploratory development evidence"]}}
               for key, label, path, tables, description in source_groups]
    for _, _, _, paths, _ in source_groups:
        assert all((repo / path).is_file() for path in paths)
    labels = {"base": "C-base", "candidates": "C-candidates", "function": "C-function", "kd_off": "KD-off（base候选）"}
    screen = [{**row, "label": labels[row["arm"]], "rate": row["valid_pool"] / row["denominator"],
               "queries": 1198, "seed": 13} for row in training["c1"] if row["step"] == 356]
    split_labels = {"train_fit": "train-fit", "train_calibration": "train-calibration",
                    "dev": "dev", "test": "历史test regression"}
    split_rows = [{"split": key, "label": label, "denominator": values["implicit_positive_pairs"],
                   "supported_pairs": values["known_rows_at_least_3_pairs"],
                   "rate": values["known_rows_at_least_3_rate"],
                   "two_row_pairs": values["known_recovered_row_count_histogram"]["2"], "query_rows": 5}
                  for key, label in split_labels.items()
                  for values in [coverage["r10_protocol_splits"][key]]]
    datasets = {}
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE reviewed_chart_rows (dataset TEXT, row_index INTEGER, row_json TEXT)")
    for dataset, rows, source_id in (("c1_screen", screen, "training"),
                                     ("coverage_splits", split_rows, "coverage")):
        connection.executemany("INSERT INTO reviewed_chart_rows VALUES (?, ?, ?)",
            [(dataset, index, json.dumps(row, ensure_ascii=False)) for index, row in enumerate(rows)])
        query = f"SELECT row_json FROM reviewed_chart_rows WHERE dataset = '{dataset}' ORDER BY row_index"
        datasets[dataset] = [json.loads(row[0]) for row in connection.execute(query)]
        source = next(item for item in sources if item["id"] == source_id)
        source["query"].update(engine="SQLite", language="sql", sql=query)
        source["query"]["tables_used"].insert(0, "main.reviewed_chart_rows")
        source["query"]["description"] += (
            " build_stage1_r10_r12_report.py materializes reviewed Python calculations into an in-memory "
            "table and executes this query to preserve chart row order; SQL does not refit or relabel data."
        )
    connection.close()
    manifest = {"version": 1, "surface": "report", "title": title,
                "description": "R10–R12科研机制诊断 · 中文报告 · 2026年9月9日固定快照",
                "generatedAt": stamp, "cards": [], "charts": [], "tables": [],
                "sources": sources, "blocks": [{"id": "title", "type": "markdown", "body": f"# {title}"}]}
    for chart_id, chart_title, subtitle, source in (
        ("c1_screen", "356步时的完整正确路径覆盖", "R12 dev；678个implicit正对；同Teacher和Student更新预算；seed13", "training"),
        ("coverage_splits", "各划分至少3行确认支持的正对比例", "5行query；正对去重；R10协议；确认标签覆盖，不是真实可恢复性上界", "coverage"),
    ):
        manifest["charts"].append({"id": chart_id, "title": chart_title, "subtitle": subtitle,
            "type": "horizontalBar", "dataset": chart_id, "sourceId": source, "valueFormat": "percent",
            "encodings": {"x": {"field": "label", "type": "nominal", "label": "实验条件 / 划分"},
                          "y": {"field": "rate", "type": "quantitative", "label": "正对比例"},
                          "tooltip": [{"field": "denominator", "type": "quantitative", "label": "implicit正对分母"}]}})
    sections = re.split(r"(?=^## )", narrative, flags=re.MULTILINE)[1:]
    for section_index, section in enumerate(sections):
        lines = section.strip().splitlines()
        buffer = []
        index = 0

        def flush() -> None:
            if buffer:
                manifest["blocks"].append({"id": f"prose_{len(manifest['blocks'])}",
                    "type": "markdown", "body": "\n".join(buffer).strip(), "sourceId": "audit"})
                buffer.clear()

        while index < len(lines):
            line = lines[index]
            chart_match = re.fullmatch(r"<!-- chart:(\w+) -->", line)
            if chart_match:
                flush()
                chart_id = chart_match[1]
                manifest["blocks"].append({"id": f"chart_{chart_id}", "type": "chart", "chartId": chart_id})
                index += 1
            elif line.startswith("|"):
                flush()
                headers = markdown_cells(line)
                index += 2
                rows = []
                while index < len(lines) and lines[index].startswith("|"):
                    cells = markdown_cells(lines[index])
                    assert len(cells) == len(headers)
                    rows.append({"order": len(rows) + 1, **{f"c{n}": value for n, value in enumerate(cells)}})
                    index += 1
                table_id = f"table_{len(manifest['tables'])}"
                datasets[table_id] = rows
                manifest["tables"].append({"id": table_id, "title": lines[0].removeprefix("## "),
                    "dataset": table_id, "sourceId": "audit", "defaultSort": {"field": "order", "direction": "asc"},
                    "columns": [{"field": "order", "label": "序", "format": "number"},
                                *[{"field": f"c{n}", "label": value, "type": "text"} for n, value in enumerate(headers)]]})
                manifest["blocks"].append({"id": f"block_{table_id}", "type": "table", "tableId": table_id})
            else:
                buffer.append(line)
                index += 1
        flush()
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE reviewed_report_rows (table_id TEXT, row_index INTEGER, row_json TEXT)")
    for table in manifest["tables"]:
        table_id = table["dataset"]
        connection.executemany("INSERT INTO reviewed_report_rows VALUES (?, ?, ?)",
            [(table_id, row["order"], json.dumps(row, ensure_ascii=False)) for row in datasets[table_id]])
        datasets[table_id] = []
    table_query = "SELECT table_id, row_json FROM reviewed_report_rows ORDER BY table_id, row_index"
    for table_id, row_json in connection.execute(table_query):
        datasets[table_id].append(json.loads(row_json))
    connection.close()
    sources[0]["query"].update(engine="SQLite", language="sql", sql=table_query)
    sources[0]["query"]["tables_used"].insert(0, "main.reviewed_report_rows")
    sources[0]["query"]["description"] += (
        " The parser materializes all eight reviewed Markdown tables; this SQL preserves their exact "
        "displayed values, without treating narrative hypotheses as observed outcomes."
    )
    artifact = {"surface": "report", "manifest": manifest, "sources": sources,
                "snapshot": {"version": 1, "generatedAt": stamp, "status": "ready",
                             "datasets": datasets, "accessIssues": []}}
    (output / "artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"sections": len(sections), "tables": len(manifest["tables"]),
                      "charts": len(manifest["charts"]), "artifact": str(output / "artifact.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--coverage-data", type=Path, required=True)
    args = parser.parse_args()
    build(args.repo.resolve(), args.output.resolve(), args.training_data, args.coverage_data)
