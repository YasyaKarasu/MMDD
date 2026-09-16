#!/usr/bin/env python
"""Evaluate the frozen strongest Bridge checkpoint on the EntiTables test split."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_stage1_r26 as retrieval
import evaluate_stage1_r26_teacher as teacher
from mmdd_dataset.wdc_runtime import iter_dataset_artifact
from prepare_stage1_r26 import file_record
from run_stage1_r21 import paths, read_rows, write_rows


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work/stage1_b4_test_20260916"
DATASET = (
    ROOT
    / "output_mm_joinability_entitables_20000_retry100_rounds5_qwen35_final_survivor_context_gaussian_v9"
)
TARGET_LISTS = (
    ROOT
    / "work/stage1_optimization_r10_20260907/taskA_protocol/lists/target_lists.test.jsonl"
)
SOURCE_EVAL = ROOT / "work/stage1_bridge_20260915/evaluation"
GENERATOR = "B4_seed29"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _checkpoint_spec() -> dict[str, Any]:
    inventory = json.loads((SOURCE_EVAL / "MODEL_INVENTORY.json").read_text())
    return next(row for row in inventory if row["generator_id"] == GENERATOR)


def _test_population() -> list[dict[str, Any]]:
    targets = list(read_rows(TARGET_LISTS))
    target_by_query = {str(row["query_id"]): row for row in targets}
    if len(target_by_query) != len(targets):
        raise ValueError("Duplicate query in frozen test target lists")

    query_meta = {
        str(row["table_id"]): row
        for row in iter_dataset_artifact(DATASET, "query_tables")
        if row.get("split") == "test"
    }
    qrels: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in iter_dataset_artifact(DATASET, "qrels"):
        if row.get("split") == "test":
            qrels[str(row["query_table_id"])].append(row)

    expected_ids = set(target_by_query)
    if set(query_meta) != expected_ids or set(qrels) != expected_ids:
        raise ValueError("Frozen target lists, query tables, and qrels have different test populations")

    population = []
    for target_row in targets:
        query_id = str(target_row["query_id"])
        labels = qrels[query_id]
        reasons = {str(row["reason"]) for row in labels}
        if reasons == {"model_recoverable_join_column"}:
            query_kind = "implicit"
        elif reasons == {"explicit_visible_join_column"}:
            query_kind = "explicit"
        else:
            raise ValueError(f"Unexpected test qrel reasons for {query_id}: {sorted(reasons)}")
        positive_ids = sorted(str(row["target_table_id"]) for row in labels)
        if positive_ids != sorted(map(str, target_row["positive_target_ids"])):
            raise ValueError(f"Frozen qrels changed for {query_id}")
        source_ids = {str(row["source_table_id"]) for row in labels}
        source_ids.add(str(query_meta[query_id]["source_table_id"]))
        if len(source_ids) != 1:
            raise ValueError(f"Source group changed for {query_id}")
        population.append(
            {
                "query_id": query_id,
                "query_kind": query_kind,
                "positive_target_ids": positive_ids,
                "source_table_id": source_ids.pop(),
            }
        )

    counts = {kind: sum(row["query_kind"] == kind for row in population) for kind in ("implicit", "explicit")}
    if len(population) != 1166 or counts != {"implicit": 583, "explicit": 583}:
        raise ValueError(f"Unexpected frozen test population: rows={len(population)}, kinds={counts}")
    corpus_ids = {str(row["object_id"]) for row in read_rows(paths(ROOT)["corpus"])}
    overlap = corpus_ids.intersection(row["query_id"] for row in population)
    if overlap:
        raise ValueError("Test queries overlap the retrieval corpus")
    return population


def prepare() -> dict[str, Any]:
    spec = _checkpoint_spec()
    checkpoint = Path(spec["checkpoint"])
    source_index = SOURCE_EVAL / "indexes" / GENERATOR
    source_protocol = SOURCE_EVAL / "PROTOCOL.json"
    source_dev_metrics = SOURCE_EVAL / "teacher" / GENERATOR / "metrics.json"
    b13_dev_metrics = ROOT / "work/stage1_optimization_r26_20260914/teacher/B13/metrics.json"
    population = _test_population()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "common").mkdir(exist_ok=True)
    write_rows(OUT / "common/test_queries.jsonl", population)
    # The audited R26 evaluator has a historical fixed filename. The lock below
    # records that this byte-identical file contains the frozen test population.
    shutil.copyfile(OUT / "common/test_queries.jsonl", OUT / "common/dev_queries.jsonl")
    shutil.copyfile(source_protocol, OUT / "PROTOCOL.json")
    write_json(OUT / "MODEL_INVENTORY.json", [spec])

    index_parent = OUT / "indexes"
    index_parent.mkdir(exist_ok=True)
    linked_index = index_parent / GENERATOR
    if linked_index.exists() or linked_index.is_symlink():
        if linked_index.resolve() != source_index.resolve():
            raise ValueError("Existing test index link has a different target")
    else:
        linked_index.symlink_to(source_index, target_is_directory=True)

    dev_metrics = json.loads(source_dev_metrics.read_text())["overall"]["BT100_T0"]
    b13_metrics = json.loads(b13_dev_metrics.read_text())["overall"]["BT100_T0"]
    lock = {
        "status": "frozen_before_test_evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "generator": GENERATOR,
            "criterion": "highest observed Bridge checkpoint C100+T0 dev Recall@10",
            "dev_recall@10": dev_metrics["recall@10"],
            "historical_B13_dev_recall@10": b13_metrics["recall@10"],
            "checkpoint": file_record(checkpoint),
            "source_dev_metrics": file_record(source_dev_metrics),
        },
        "test_population": {
            "queries": len(population),
            "implicit": sum(row["query_kind"] == "implicit" for row in population),
            "explicit": sum(row["query_kind"] == "explicit" for row in population),
            "source_groups": len({row["source_table_id"] for row in population}),
            "query_file": file_record(OUT / "common/test_queries.jsonl"),
            "r26_compatibility_alias": file_record(OUT / "common/dev_queries.jsonl"),
            "target_lists": file_record(TARGET_LISTS),
        },
        "protocol": file_record(OUT / "PROTOCOL.json"),
        "index": {
            "source": str(source_index.resolve()),
            "manifest": file_record(source_index / "manifest.json"),
        },
        "test_use_note": (
            "This split was used by the earlier R10 closeout, but not to train or select B4. "
            "Only the dev-selected B4 seed29 checkpoint is evaluated here."
        ),
    }
    lock_path = OUT / "TEST_LOCK.json"
    if lock_path.exists():
        previous = json.loads(lock_path.read_text())
        comparable = {key: value for key, value in lock.items() if key != "created_at_utc"}
        previous_comparable = {key: value for key, value in previous.items() if key != "created_at_utc"}
        if previous_comparable != comparable:
            raise ValueError("Existing test lock differs")
        return previous
    write_json(lock_path, lock)
    return lock


def run_own(device: str, index_threads: int) -> dict[str, Any]:
    prepare()
    retrieval.OUT = OUT
    retrieval.PRELOAD_ALL = True
    return retrieval.evaluate(GENERATOR, device, index_threads=index_threads, legacy_diagnostics=False)


def run_teacher(device: str) -> dict[str, Any]:
    prepare()
    teacher.OUT = OUT
    return teacher.run([GENERATOR], device, benchmark_queries=0, cache_name="T0_test_pairs.sqlite")


def _cluster_bootstrap(rows: list[dict[str, Any]], replicates: int = 10_000) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        positives = set(map(str, row["positive_target_ids"]))
        ranking = list(map(str, row["rankings"]["BT100_T0"]))
        grouped[str(row["source_table_id"])].append(len(positives.intersection(ranking[:10])) / len(positives))
    names = sorted(grouped)
    numerators = np.asarray([sum(grouped[name]) for name in names], dtype=np.float64)
    denominators = np.asarray([len(grouped[name]) for name in names], dtype=np.float64)
    observed = float(numerators.sum() / denominators.sum())
    rng = np.random.default_rng(260916)
    estimates = []
    for start in range(0, replicates, 256):
        selected = rng.integers(len(names), size=(min(256, replicates - start), len(names)))
        estimates.extend((numerators[selected].sum(axis=1) / denominators[selected].sum(axis=1)).tolist())
    low, high = np.quantile(np.asarray(estimates), [0.025, 0.975]).tolist()
    return {
        "estimate": observed,
        "ci95": [float(low), float(high)],
        "replicates": replicates,
        "rng_seed": 260916,
        "cluster": "source_table_id",
        "source_groups": len(names),
        "queries": int(denominators.sum()),
    }


def summarize() -> dict[str, Any]:
    lock = json.loads((OUT / "TEST_LOCK.json").read_text())
    own = json.loads((OUT / "rankings" / GENERATOR / "metrics.json").read_text())
    t0 = json.loads((OUT / "teacher" / GENERATOR / "metrics.json").read_text())
    teacher_rows = list(read_rows(OUT / "teacher" / GENERATOR / "rankings.jsonl.gz"))
    primary = t0["overall"]["BT100_T0"]
    bootstrap = _cluster_bootstrap(teacher_rows)
    if abs(primary["recall@10"] - bootstrap["estimate"]) > 1e-12:
        raise ValueError("Saved aggregate and bootstrap input disagree")
    t0_dev = {"recall@10": lock["selection"]["dev_recall@10"]}
    result = {
        "status": "complete",
        "generator": GENERATOR,
        "checkpoint": lock["selection"]["checkpoint"],
        "population": lock["test_population"],
        "primary_metric": "C100+T0 query-macro target Recall@10 (BT100_T0)",
        "dev": t0_dev,
        "test": {
            "overall": primary,
            "implicit": t0["implicit"]["BT100_T0"],
            "explicit": t0["explicit"]["BT100_T0"],
            "without_T0": t0["overall"]["BT100_NO_T0"],
            "student_direct_exact": own["overall"]["D100_EXACT"],
            "student_equal": own["overall"]["Equal"],
        },
        "dev_minus_test_recall@10": t0_dev["recall@10"] - primary["recall@10"],
        "test_recall@10_source_cluster_bootstrap": bootstrap,
        "scientific_scope": lock["test_use_note"],
    }
    write_json(OUT / "TEST_RESULTS.json", result)
    lines = [
        "# B4 seed29 frozen test evaluation",
        "",
        f"Checkpoint SHA-256: `{result['checkpoint']['sha256']}`",
        "",
        "| Split/slice | R@10 | R@20 | R@50 | Raw candidate recall |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| dev overall | {100 * result['dev']['recall@10']:.2f}% | - | - | - |",
        f"| test overall | {100 * primary['recall@10']:.2f}% | {100 * primary['recall@20']:.2f}% | {100 * primary['recall@50']:.2f}% | {100 * primary['raw_recall']:.2f}% |",
        f"| test implicit | {100 * result['test']['implicit']['recall@10']:.2f}% | {100 * result['test']['implicit']['recall@20']:.2f}% | {100 * result['test']['implicit']['recall@50']:.2f}% | {100 * result['test']['implicit']['raw_recall']:.2f}% |",
        f"| test explicit | {100 * result['test']['explicit']['recall@10']:.2f}% | {100 * result['test']['explicit']['recall@20']:.2f}% | {100 * result['test']['explicit']['recall@50']:.2f}% | {100 * result['test']['explicit']['raw_recall']:.2f}% |",
        "",
        f"Test R@10 source-group bootstrap 95% CI: [{100 * bootstrap['ci95'][0]:.2f}%, {100 * bootstrap['ci95'][1]:.2f}%].",
        f"Dev minus test R@10: {100 * result['dev_minus_test_recall@10']:+.2f} percentage points.",
        "",
        lock["test_use_note"],
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "own", "teacher", "summarize"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--index-threads", type=int, default=2)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare()
    elif args.command == "own":
        result = run_own(args.device, args.index_threads)
    elif args.command == "teacher":
        result = run_teacher(args.device)
    else:
        result = summarize()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
