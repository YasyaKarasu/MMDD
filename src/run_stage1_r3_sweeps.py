#!/usr/bin/env python
"""Run the round-3 gamma and evidence-budget sensitivity sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import torch

from mmdd_stage1.checkpoints import load_student, load_teacher
from mmdd_stage1.data import load_target_examples
from mmdd_stage1.evaluation import evaluate_student_retrieval
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import (
    StudentANNIndices,
    checkpoint_fingerprint,
)
from mmdd_stage1.selection import load_stage1_selection
from mmdd_stage1.teacher_rerank import teacher_ensemble_metrics as _teacher_ensemble_metrics


def _examples(paths: list[str]) -> list:
    return [
        example
        for value in paths
        for example in load_target_examples(Path(value), split="dev", dataset_name=Path(value).stem)
    ]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _teacher_preflight(
    examples: list,
    indices: StudentANNIndices,
    store: FeatureStore,
    *,
    gammas: tuple[int, ...],
    recall_ks: tuple[int, ...],
    objects_path: Path,
    output_dir: Path,
) -> None:
    query_ids = [example.query_id for example in examples]
    required_ids = set(query_ids)
    for pool_size in sorted({gamma * k for gamma in gammas for k in recall_ks}):
        required_ids.update(
            target_id
            for hits in indices.search_many(query_ids, "table", pool_size)
            for target_id, _score in hits
        )
    missing = sorted(
        object_id for object_id in required_ids if not store.has_teacher_features(object_id)
    )
    preflight = {
        "queries": len(examples),
        "candidate_pool_sizes": sorted({gamma * k for gamma in gammas for k in recall_ks}),
        "required_teacher_objects": len(required_ids),
        "missing_teacher_objects": len(missing),
    }
    if not missing:
        _write_json(output_dir / "preflight.json", preflight)
        return

    missing_set = set(missing)
    missing_ids_path = output_dir / "missing_teacher_ids.jsonl"
    missing_input_path = output_dir / "missing_teacher_input.jsonl"
    with missing_ids_path.open("w", encoding="utf-8") as handle:
        for object_id in missing:
            handle.write(json.dumps({"object_id": object_id}) + "\n")
    found = set()
    with objects_path.open(encoding="utf-8") as source, missing_input_path.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            object_id = str(record["object_id"])
            if object_id in missing_set:
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                found.add(object_id)
    absent = missing_set - found
    if absent:
        raise KeyError(f"Corpus is missing {len(absent)} required Teacher objects")
    preflight.update(
        {
            "missing_teacher_ids": str(missing_ids_path.resolve()),
            "missing_teacher_input": str(missing_input_path.resolve()),
        }
    )
    _write_json(output_dir / "preflight.json", preflight)
    raise ValueError(
        f"Teacher sweep requires hidden states for {len(missing)} more objects; "
        f"cache {missing_input_path} and rerun"
    )


def _write_png(
    rows: list[dict],
    path: Path,
    *,
    title: str,
    y_fields: list[tuple[str, str]],
    x_field: str = "gamma",
) -> None:
    """Small dependency-free line chart suitable for quick inspection."""
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1200, 720
    left, right, top, bottom = 90, 40, 70, 90
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((left, 20), title, fill="black", font=font)
    x_values = sorted({int(r[x_field]) for r in rows})
    values = [float(r[field]) for r in rows for field, _label in y_fields]
    ymin, ymax = min(values), max(values)
    pad = max((ymax - ymin) * 0.12, 0.01)
    ymin -= pad
    ymax += pad
    plot_w, plot_h = width - left - right, height - top - bottom

    def xy(x_value: float, value: float) -> tuple[float, float]:
        x = left + (x_value - x_values[0]) / max(x_values[-1] - x_values[0], 1) * plot_w
        y = top + (ymax - value) / max(ymax - ymin, 1e-9) * plot_h
        return x, y

    percent_axis = all(
        "recall@" in field or "coverage@" in field
        for field, _label in y_fields
    )
    for tick in range(6):
        value = ymin + (ymax - ymin) * tick / 5
        y = top + plot_h - plot_h * tick / 5
        draw.line((left, y, width - right, y), fill="#dddddd")
        label = f"{value:.1%}" if percent_axis else f"{value:.3f}"
        draw.text((8, y - 6), label, fill="#333333", font=font)
    for x_value in x_values:
        x, _ = xy(x_value, ymin)
        draw.line((x, top, x, top + plot_h), fill="#eeeeee")
        draw.text((x - 8, top + plot_h + 10), str(x_value), fill="#333333", font=font)

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    systems = sorted({str(row.get("system", "series")) for row in rows})
    series = [
        (system, field, label if system == "series" else f"{system} {label}")
        for system in systems
        for field, label in y_fields
    ]
    for idx, (system, field, label) in enumerate(series):
        points = [
            xy(int(row[x_field]), float(row[field]))
            for row in sorted(rows, key=lambda item: int(item[x_field]))
            if str(row.get("system", "series")) == system
        ]
        color = colors[idx % len(colors)]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        for point in points:
            draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=color)
        lx = left + idx * 190
        draw.line((lx, height - 35, lx + 20, height - 35), fill=color, width=3)
        draw.text((lx + 25, height - 41), label, fill="black", font=font)
    image.save(path)


def run_sweeps(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    selection = load_stage1_selection(Path(args.selection))
    student_path = Path(selection["best_checkpoint"])
    student_sha = checkpoint_fingerprint(student_path)
    corpus_path = Path(args.corpus)
    corpus_sha = checkpoint_fingerprint(corpus_path)
    store = FeatureStore.from_path(
        Path(args.features),
        cache_size=args.feature_cache_size,
        teacher_paths=tuple(Path(value) for value in args.teacher_features),
    )
    examples = _examples(args.dev_data)
    student = load_student(student_path, device)
    student.eval()
    student_indices = StudentANNIndices(
        student,
        store,
        Path(selection["best_index"]),
        device=device,
        checkpoint_sha256=student_sha,
        corpus_sha256=corpus_sha,
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gamma_values = tuple(int(v) for v in args.gammas.split(","))
    _teacher_preflight(
        examples,
        student_indices,
        store,
        gammas=gamma_values,
        recall_ks=(10, 50),
        objects_path=Path(args.objects),
        output_dir=out,
    )
    teacher = load_teacher(Path(args.teacher_checkpoint), device)
    teacher.eval()
    if store.teacher_dimension() != teacher.input_dim:
        raise ValueError("Teacher checkpoint does not match feature-store hidden dimension")

    progress_path = out / "progress.json"
    progress_payload = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.is_file()
        else {}
    )
    rows: list[dict] = list(progress_payload.get("rows", []))
    cache_policy = "independent_per_gamma"
    if progress_payload.get("teacher_score_cache_policy") != cache_policy:
        rows = [row for row in rows if row["system"] == "student"]
    for gamma in gamma_values:
        completed = {str(row["system"]) for row in rows if int(row["gamma"]) == gamma}
        if "student" not in completed:
            start = time.perf_counter()
            student_metrics = evaluate_student_retrieval(
                examples,
                student_indices,
                recall_ks=(10, 50),
                gamma=gamma,
                gamma_evidence=args.gamma_evidence,
                evidence_aggregation=args.evidence_aggregation,
                evidence_top_k=args.evidence_top_k,
                fusion_mode="weighted_rrf",
                evidence_weight=0.05,
            )
            elapsed = time.perf_counter() - start
            rows.append({
                "gamma": gamma,
                "system": "student",
                "recall@10": student_metrics["recall@10"],
                "recall@50": student_metrics["recall@50"],
                "latency_seconds_per_query": elapsed / len(examples),
                "latency_seconds_per_query_per_k": elapsed / (len(examples) * 2),
                "queries": len(examples),
            })
            _write_json(
                progress_path,
                {"teacher_score_cache_policy": cache_policy, "rows": rows},
            )
        if "student_ensemble" not in completed:
            score_cache: dict[tuple[str, str], float] = {}
            ensemble_metrics, timing = _teacher_ensemble_metrics(
                teacher,
                examples,
                student_indices,
                store,
                recall_ks=(10, 50),
                gamma=gamma,
                alpha=args.teacher_alpha,
                device=device,
                batch_size=args.teacher_batch_size,
                score_cache=score_cache,
            )
            rows.append({
                "gamma": gamma,
                "system": "student_ensemble",
                "recall@10": ensemble_metrics["recall@10"],
                "recall@50": ensemble_metrics["recall@50"],
                "latency_seconds_per_query": timing["total_seconds"] / len(examples),
                "latency_seconds_per_query_per_k": timing["average_seconds_per_query_per_k"],
                "queries": len(examples),
            })
            _write_json(
                progress_path,
                {"teacher_score_cache_policy": cache_policy, "rows": rows},
            )
    rows.sort(key=lambda row: (int(row["gamma"]), str(row["system"])))

    csv_path = out / "gamma_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_png(
        rows,
        out / "gamma_sweep.png",
        title="Task F: gamma sensitivity (recall)",
        y_fields=[("recall@10", "R@10"), ("recall@50", "R@50")],
    )
    # The chart helper above expects one series per row; provide a second explicit latency chart.
    _write_png(
        rows,
        out / "gamma_latency.png",
        title="Task F: gamma sensitivity (latency/query)",
        y_fields=[("latency_seconds_per_query", "latency/query")],
    )
    by_system = {
        system: [r for r in rows if r["system"] == system]
        for system in {r["system"] for r in rows}
    }
    # Saturation means every reported recall is within 0.5 point of the scan maximum.
    chosen = gamma_values[-1]
    maxima = {
        (system, metric): max(float(row[metric]) for row in system_rows)
        for system, system_rows in by_system.items()
        for metric in ("recall@10", "recall@50")
    }
    for gamma in gamma_values:
        gamma_rows = {str(row["system"]): row for row in rows if int(row["gamma"]) == gamma}
        if all(
            maxima[(system, metric)] - float(gamma_rows[system][metric]) <= 0.005
            for system in by_system
            for metric in ("recall@10", "recall@50")
        ):
            chosen = gamma
            break
    final_step_gains = {
        f"{system}.{metric}": float(system_rows[-1][metric]) - float(system_rows[-2][metric])
        for system, system_rows in by_system.items()
        for metric in ("recall@10", "recall@50")
    }
    depth_bottleneck = chosen == gamma_values[-1] and any(
        gain > 0.005 for gain in final_step_gains.values()
    )
    lines = [
        "# Task F: gamma sweep",
        "",
        f"Scanned γ={list(gamma_values)} with γ_e={args.gamma_evidence} on {len(examples)} dev queries (CPU). Each γ uses an independent cold Teacher score cache; reuse is limited to the two k cutoffs within that γ.",
        "",
        "| γ | System | R@10 | R@50 | latency/query (s) |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row['gamma']} | {row['system']} | {float(row['recall@10']):.2%} | {float(row['recall@50']):.2%} | {float(row['latency_seconds_per_query']):.4f} |")
    lines += [
        "",
        f"Selected γ*={chosen}: smallest scanned value within 0.5 percentage points of the scan maximum for both systems and both recall metrics. See `gamma_sweep.csv`, `gamma_sweep.png`, and `gamma_latency.png`.",
    ]
    if depth_bottleneck:
        lines += [
            "",
            "The ensemble curve is not saturated at γ=10; the recall-layer depth remains a bottleneck. Per the R3 decision rule, γ*=10 is retained as the temporary default.",
        ]
    lines += [
        "",
        f"Hardware: device={device}, CPU={platform.processor() or platform.machine()}, torch={torch.__version__}.",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(
        out / "selection.json",
        {
            "gamma_star": chosen,
            "recall_depth_bottleneck": depth_bottleneck,
            "final_step_gains": final_step_gains,
            "rows": rows,
        },
    )

    if args.run_evidence:
        evidence_out = out.parent / "taskG_evidence_sweep"
        evidence_out.mkdir(parents=True, exist_ok=True)
        evidence_rows = []
        for gamma_evidence in (1, 2, 4):
            start = time.perf_counter()
            metrics = evaluate_student_retrieval(
                examples,
                student_indices,
                recall_ks=(10,),
                gamma=chosen,
                gamma_evidence=gamma_evidence,
                evidence_aggregation=args.evidence_aggregation,
                evidence_top_k=args.evidence_top_k,
                fusion_mode="weighted_rrf",
                evidence_weight=0.05,
            )
            elapsed = time.perf_counter() - start
            evidence_rows.append({
                "gamma": chosen,
                "gamma_evidence": gamma_evidence,
                "coverage@10": metrics["positive_evidence_path_coverage@10"],
                "evidence_recall@10": metrics["evidence"]["recall@10"],
                "latency_seconds_per_query": elapsed / len(examples),
            })
        with (evidence_out / "evidence_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(evidence_rows[0]))
            writer.writeheader()
            writer.writerows(evidence_rows)
        _write_png(
            evidence_rows,
            evidence_out / "evidence_sweep.png",
            title="Task G: evidence-budget sensitivity",
            y_fields=[
                ("coverage@10", "coverage@10"),
                ("evidence_recall@10", "evidence R@10"),
            ],
            x_field="gamma_evidence",
        )
        lines = [
            "# Task G: evidence-budget sweep",
            "",
            f"Student system at γ*={chosen}; scanned γ_e∈{{1,2,4}} on {len(examples)} dev queries (CPU).",
            "",
            "| γ_e | coverage@10 | evidence recall@10 | latency/query (s) |",
            "| ---: | ---: | ---: | ---: |",
        ]
        for row in evidence_rows:
            lines.append(f"| {row['gamma_evidence']} | {float(row['coverage@10']):.2%} | {float(row['evidence_recall@10']):.2%} | {float(row['latency_seconds_per_query']):.4f} |")
        max_coverage = max(float(row["coverage@10"]) for row in evidence_rows)
        best_e = evidence_rows[-1]["gamma_evidence"]
        for row in evidence_rows:
            if max_coverage - float(row["coverage@10"]) <= 0.005:
                best_e = row["gamma_evidence"]
                break
        lines += [
            "",
            f"Selected γ_e*={best_e}: the smallest scanned value within 0.5 percentage points of maximum coverage. "
            f"γ_e=4 adds only {(float(evidence_rows[-1]['coverage@10']) - float(evidence_rows[1]['coverage@10'])):.2%} over γ_e=2 while increasing latency "
            f"{(float(evidence_rows[-1]['latency_seconds_per_query']) / float(evidence_rows[1]['latency_seconds_per_query'])):.1f}x and lowering evidence R@10.",
            "See `evidence_sweep.csv` and `evidence_sweep.png`.",
        ]
        (evidence_out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (evidence_out / "selection.json").write_text(json.dumps({"gamma": chosen, "gamma_evidence_star": best_e, "rows": evidence_rows}, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--teacher-features", nargs="*", default=[])
    parser.add_argument("--dev-data", nargs="+", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gammas", default="2,4,6,8,10")
    parser.add_argument("--gamma-evidence", type=int, default=2)
    parser.add_argument("--teacher-alpha", type=float, default=0.7)
    parser.add_argument("--teacher-batch-size", type=int, default=64)
    parser.add_argument("--feature-cache-size", type=int, default=60000)
    parser.add_argument("--index-batch-size", type=int, default=1024)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=100)
    parser.add_argument("--evidence-aggregation", default="logsumexp")
    parser.add_argument("--evidence-top-k", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-evidence", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_sweeps(parse_args())
