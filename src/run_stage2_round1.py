#!/usr/bin/env python
"""Run the 2026-08-31 Stage-2 Oracle-positive round-1 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from mmdd_stage1.retrieval import checkpoint_fingerprint
from mmdd_stage1.selection import validate_stage2_gate
from mmdd_stage2.checkpoints import load_candidate_scorer
from mmdd_stage2.oracle import (
    ORACLE_EVIDENCE_POLICY,
    OracleDataError,
    load_oracle_column_data,
)
from mmdd_stage2.qwen import QwenStage2Backend
from mmdd_stage2.reader_cache import (
    build_reader_cache,
    deterministic_baselines,
    evaluate_scorer,
    load_reader_cache,
    mean_std,
    paired_bootstrap_ci,
    train_cached_scorer,
    write_json_atomic,
    write_jsonl_atomic,
)
from mmdd_stage2.verifier import CandidateColumnScorer


DEFAULT_ROOTS = (
    "output_mm_joinability_entitables_20000_retry100_rounds5_qwen35_queryonly_autocheck_v4_jsonrepair_luna_consensus",
    "output_wdc_webtable_2000_qwen35_unified_autocheck_v2_luna_consensus",
)
DEFAULT_OUTPUT = Path("work/stage2_round1_20260831")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _old_stage1_audit() -> dict[str, Any]:
    base = Path("work/stage1_stage2_wdc2k_entitables20k_v4_20260828")
    gate = base / "checkpoints/student_path.pt.selection.json"
    retrievals = [
        base / "retrieval_entitables_train.jsonl",
        base / "retrieval_wdc_train.jsonl",
    ]
    result: dict[str, Any] = {
        "status": "unavailable",
        "gate": str(gate),
        "retrievals": [],
        "missing_required_splits": [
            "entitables dev/test retrieval",
            "wdc dev/test retrieval",
        ],
        "note": "Old mixed-pipeline files are audited only and are not used as formal inputs.",
    }
    if not gate.is_file():
        result["error"] = "old selection manifest is missing"
        return result
    selection = _load_json(gate)
    checkpoint = Path(str(selection.get("best_checkpoint", "")))
    result["gate_stage2_allowed"] = selection.get("stage2_allowed")
    result["gate_checkpoint_sha256"] = selection.get("best_checkpoint_sha256")
    result["checkpoint_exists"] = checkpoint.is_file()
    result["checkpoint_actual_sha256"] = (
        checkpoint_fingerprint(checkpoint) if checkpoint.is_file() else None
    )
    for retrieval in retrievals:
        fingerprints = set()
        records = 0
        if retrieval.is_file():
            with retrieval.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        record = json.loads(line)
                        fingerprints.add(record.get("student_checkpoint_sha256"))
                        records += 1
        result["retrievals"].append(
            {
                "path": str(retrieval),
                "exists": retrieval.is_file(),
                "records": records,
                "student_checkpoint_sha256": sorted(
                    str(value) for value in fingerprints
                ),
            }
        )
    try:
        validate_stage2_gate(gate, retrievals)
    except (ValueError, FileNotFoundError) as error:
        result["validation_error"] = str(error)
    else:
        result["status"] = "valid_but_incomplete"
    return result


def _audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Stage-2 Round-1 Data Audit",
        "",
        f"- Training source: `{audit['training_source']}`",
        f"- Evidence policy: {audit['evidence_policy']}",
        f"- Total usable examples: {audit['total_usable_examples']:,}",
        "",
        "## Per-lake split counts",
        "",
        "| Lake | Split | Recoverable qrels | Usable examples | Text only | Image only | Text + image |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset, item in audit["datasets"].items():
        for split, values in item["splits"].items():
            modality = values["modality"]
            lines.append(
                f"| {dataset} | {split} | {values['qrels']:,} | "
                f"{values['usable_examples']:,} | {modality.get('text_only', 0):,} | "
                f"{modality.get('image_only', 0):,} | {modality.get('text_image', 0):,} |"
            )
    lines.extend(["", "## Missing or invalid assets", ""])
    for dataset, item in audit["datasets"].items():
        lines.append(f"### {dataset}")
        lines.append("")
        for kind, ids in item["missing"].items():
            rendered = ", ".join(ids) if ids else "none"
            lines.append(f"- {kind}: {len(ids)} ({rendered})")
        duplicates = item.get(
            "duplicate_qrel_pairs", item.get("duplicate_qrel_query_ids", [])
        )
        lines.append(
            f"- duplicate qrel pairs: {len(duplicates)} "
            f"({', '.join(duplicates) if duplicates else 'none'})"
        )
        lines.append("")
    lines.extend(["## Distributions", ""])
    for dataset, item in audit["datasets"].items():
        for split, values in item["splits"].items():
            lines.extend(
                [
                    f"### {dataset} / {split}",
                    "",
                    f"- Evidence before truncation: `{json.dumps(values['evidence_count_before_truncation'], sort_keys=True)}`",
                    f"- Evidence selected: `{json.dumps(values['evidence_count_selected'], sort_keys=True)}`",
                    f"- Candidate-column count: `{json.dumps(values['candidate_column_count'], sort_keys=True)}`",
                    f"- Gold-column position (zero-based): `{json.dumps(values['gold_column_position'], sort_keys=True)}`",
                    "",
                ]
            )
    lines.extend(["## Leakage and ID conflicts", ""])
    lines.append(f"- Cross-dataset ID conflicts: {len(audit['id_conflicts'])}")
    for pair, intersections in audit["cross_split_intersections"].items():
        lines.append(
            f"- {pair}: source tables={len(intersections['source_table_ids'])}, "
            f"chains={len(intersections['chain_ids'])}"
        )
    old = audit["old_stage1_retrieval_audit"]
    lines.extend(
        [
            "",
            "## Old Stage-1 retrieval/gate fingerprint audit",
            "",
            f"- Status: `{old['status']}`",
            f"- Gate expected checkpoint SHA-256: `{old.get('gate_checkpoint_sha256')}`",
            f"- Actual checkpoint SHA-256: `{old.get('checkpoint_actual_sha256')}`",
            f"- Validation error: {old.get('validation_error', 'none')}",
            f"- Missing formal inputs: {', '.join(old['missing_required_splits'])}",
            "- Result: Task D cannot use these legacy mixed-pipeline files as formal inputs.",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(args: argparse.Namespace) -> None:
    examples, _objects, audit = load_oracle_column_data(
        [Path(root) for root in args.dataset_root], strict=False
    )
    audit["old_stage1_retrieval_audit"] = _old_stage1_audit()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_dir / "data_audit.json", audit)
    (args.output_dir / "DATA_AUDIT.md").write_text(
        _audit_markdown(audit), encoding="utf-8"
    )
    try:
        load_oracle_column_data([Path(root) for root in args.dataset_root], strict=True)
    except OracleDataError:
        raise
    print(json.dumps({"examples": len(examples), "audit": str(args.output_dir / "data_audit.json")}, indent=2))


def _subset_examples(examples: list[Any], limit: int | None) -> list[Any]:
    if limit is None:
        return examples
    selected = []
    for dataset in sorted({example.dataset for example in examples}):
        for split in sorted({example.split for example in examples}):
            selected.extend(
                sorted(
                    (
                        example
                        for example in examples
                        if example.dataset == dataset and example.split == split
                    ),
                    key=lambda item: item.query_id,
                )[:limit]
            )
    return selected


def run_cache(args: argparse.Namespace) -> None:
    examples, objects, _audit = load_oracle_column_data(
        [Path(root) for root in args.dataset_root],
        splits=args.split,
        top_k_evidence=args.top_k_evidence,
        strict=True,
    )
    examples = _subset_examples(examples, args.limit_per_dataset)
    backend = QwenStage2Backend(
        Path(args.model_dir), device=args.device, dtype=args.dtype
    )
    manifests = []
    for dataset in sorted({example.dataset for example in examples}):
        for split in args.split:
            subset = [
                example
                for example in examples
                if example.dataset == dataset and example.split == split
            ]
            if not subset:
                continue
            cache_dir = args.cache_root / dataset / split
            manifests.append(
                build_reader_cache(
                    backend,
                    subset,
                    objects,
                    cache_dir,
                    model_path=Path(args.model_dir),
                    model_dtype=args.dtype,
                    top_k_evidence=args.top_k_evidence,
                    evidence_policy=ORACLE_EVIDENCE_POLICY,
                    shard_size=args.shard_size,
                )
            )
    print(json.dumps({"cache_root": str(args.cache_root), "manifests": manifests}, indent=2))


def _cache_dirs(root: Path, splits: tuple[str, ...]) -> list[Path]:
    paths = []
    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for split in splits:
            candidate = dataset_dir / split
            if candidate.is_dir():
                paths.append(candidate)
    return paths


def run_train(args: argparse.Namespace) -> None:
    records, cache_fingerprint = load_reader_cache(
        _cache_dirs(args.cache_root, ("train", "dev"))
    )
    train_records = [record for record in records if record["split"] == "train"]
    dev_records = [record for record in records if record["split"] == "dev"]
    hidden_dim = int(train_records[0]["open_states"].shape[1])
    for seed in args.seed:
        seed_dir = args.output_dir / "oracle_positive" / f"seed_{seed}"
        scorer, summary, _epoch_zero = train_cached_scorer(
            train_records,
            dev_records,
            hidden_dim=hidden_dim,
            seed=seed,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            output_dir=seed_dir,
            checkpoint_metadata={
                "dataset_roots": [str(Path(root).resolve()) for root in args.dataset_root],
                "evidence_policy": ORACLE_EVIDENCE_POLICY,
                "model_dir": str(Path(args.model_dir).resolve()),
                "reader_cache_fingerprint": cache_fingerprint,
                "top_k_evidence": 4,
            },
        )
        dev_metrics, dev_predictions = evaluate_scorer(
            scorer, dev_records, include_predictions=True
        )
        write_jsonl_atomic(seed_dir / "dev_predictions.jsonl", dev_predictions)
        write_json_atomic(
            seed_dir / "metrics.json",
            {"seed": seed, "selected_epoch": summary["selected_epoch"], "dev": dev_metrics},
        )
    print(json.dumps({"seeds": args.seed, "cache_fingerprint": cache_fingerprint}, indent=2))


def run_smoke_train(args: argparse.Namespace) -> None:
    records, cache_fingerprint = load_reader_cache(
        _cache_dirs(args.cache_root, ("train", "dev"))
    )
    train_records = [record for record in records if record["split"] == "train"]
    dev_records = [record for record in records if record["split"] == "dev"]
    hidden_dim = int(train_records[0]["open_states"].shape[1])
    runs = []
    scorers = []
    for name in ("run1", "run2"):
        scorer, summary, epoch_zero = train_cached_scorer(
            train_records,
            dev_records,
            hidden_dim=hidden_dim,
            seed=13,
            epochs=1,
            learning_rate=1e-3,
            weight_decay=1e-4,
            output_dir=args.output_dir / "smoke" / name,
            checkpoint_metadata={
                "dataset_roots": [str(Path(root).resolve()) for root in args.dataset_root],
                "evidence_policy": ORACLE_EVIDENCE_POLICY,
                "model_dir": str(Path(args.model_dir).resolve()),
                "reader_cache_fingerprint": cache_fingerprint,
                "top_k_evidence": 4,
            },
        )
        metrics, _ = evaluate_scorer(scorer, dev_records)
        initial_metrics, _ = evaluate_scorer(epoch_zero, dev_records)
        loaded = load_candidate_scorer(
            args.output_dir / "smoke" / name / "candidate.pt",
            torch.device("cpu"),
            expected_model_dir=Path(args.model_dir),
        )
        loaded_metrics, _ = evaluate_scorer(loaded, dev_records)
        runs.append(
            {
                "summary": summary,
                "metrics": metrics,
                "epoch_zero_metrics": initial_metrics,
                "loaded_checkpoint_metrics": loaded_metrics,
            }
        )
        scorers.append(scorer)
    deterministic = all(
        torch.equal(scorers[0].state_dict()[name], scorers[1].state_dict()[name])
        for name in scorers[0].state_dict()
    ) and runs[0]["metrics"] == runs[1]["metrics"]
    result = {
        "passed": bool(
            deterministic
            and all(run["summary"]["head_parameters_updated"] for run in runs)
            and all(
                torch.isfinite(torch.tensor(run["summary"]["history"][0]["mean_train_loss"]))
                for run in runs
            )
            and all(run["metrics"] == run["loaded_checkpoint_metrics"] for run in runs)
        ),
        "same_seed_deterministic": deterministic,
        "runs": runs,
    }
    write_json_atomic(args.output_dir / "smoke" / "smoke_result.json", result)
    if not result["passed"]:
        raise RuntimeError("Stage-2 smoke gate failed")
    print(json.dumps(result, indent=2))


def _macro(metrics: dict[str, Any]) -> float:
    return mean(
        float(item["column_accuracy@1"])
        for item in metrics["by_dataset"].values()
    )


def run_evaluate(args: argparse.Namespace) -> None:
    records, cache_fingerprint = load_reader_cache(
        _cache_dirs(args.cache_root, ("train", "dev", "test"))
    )
    splits = {
        split: [record for record in records if record["split"] == split]
        for split in ("train", "dev", "test")
    }
    hidden_dim = int(records[0]["open_states"].shape[1])
    seed_results = {}
    test_predictions_by_seed = {}
    for seed in args.seed:
        seed_dir = args.output_dir / "oracle_positive" / f"seed_{seed}"
        scorer = load_candidate_scorer(
            seed_dir / "candidate.pt",
            torch.device("cpu"),
            expected_model_dir=Path(args.model_dir),
        )
        torch.manual_seed(seed)
        epoch_zero = CandidateColumnScorer(hidden_dim)
        split_metrics = {}
        for split, split_records in splits.items():
            metrics, predictions = evaluate_scorer(
                scorer, split_records, include_predictions=split in {"dev", "test"}
            )
            split_metrics[split] = metrics
            if split == "dev":
                write_jsonl_atomic(seed_dir / "dev_predictions.jsonl", predictions)
            elif split == "test":
                write_jsonl_atomic(seed_dir / "test_predictions.jsonl", predictions)
                test_predictions_by_seed[seed] = predictions
        epoch_zero_metrics, _ = evaluate_scorer(epoch_zero, splits["test"])
        history = _load_json(seed_dir / "history.json")
        seed_results[str(seed)] = {
            "selected_epoch": history["selected_epoch"],
            "elapsed_seconds": history["elapsed_seconds"],
            "trained": split_metrics,
            "epoch_0_seeded_head_test": epoch_zero_metrics,
            "macro_accuracy": {
                split: _macro(metrics) for split, metrics in split_metrics.items()
            },
            "macro_train_dev_gap": _macro(split_metrics["train"])
            - _macro(split_metrics["dev"]),
            "macro_dev_test_gap": _macro(split_metrics["dev"])
            - _macro(split_metrics["test"]),
        }
        write_json_atomic(seed_dir / "metrics.json", seed_results[str(seed)])

    baselines = deterministic_baselines(splits["train"], splits["test"])
    seed13_predictions = test_predictions_by_seed[13]
    uniform_values = [item["uniform_expectation"] for item in baselines["per_sample"]]
    majority_values = [float(item["majority_correct"]) for item in baselines["per_sample"]]
    baselines.pop("per_sample")
    aggregate: dict[str, Any] = {
        "format_version": 1,
        "training_source": "oracle_positive",
        "reader_cache_fingerprint": cache_fingerprint,
        "seeds": seed_results,
        "baselines_test": baselines,
        "paired_bootstrap_seed_13": {
            "vs_uniform_random_expectation": paired_bootstrap_ci(
                seed13_predictions, uniform_values
            ),
            "vs_majority_column_position": paired_bootstrap_ci(
                seed13_predictions, majority_values
            ),
        },
        "three_seed": {},
    }
    for split in ("train", "dev", "test"):
        aggregate["three_seed"][split] = {
            "macro_column_accuracy@1": mean_std(
                [seed_results[str(seed)]["macro_accuracy"][split] for seed in args.seed]
            ),
            "overall_column_accuracy@1": mean_std(
                [
                    seed_results[str(seed)]["trained"][split]["column_accuracy@1"]
                    for seed in args.seed
                ]
            ),
            "by_dataset_column_accuracy@1": {
                dataset: mean_std(
                    [
                        seed_results[str(seed)]["trained"][split]["by_dataset"][dataset]["column_accuracy@1"]
                        for seed in args.seed
                    ]
                )
                for dataset in sorted(
                    seed_results[str(args.seed[0])]["trained"][split]["by_dataset"]
                )
            },
        }
    aggregate["three_seed"]["epoch_0_seeded_head_test"] = {
        "overall_column_accuracy@1": mean_std(
            [
                seed_results[str(seed)]["epoch_0_seeded_head_test"]["column_accuracy@1"]
                for seed in args.seed
            ]
        ),
        "by_dataset_column_accuracy@1": {
            dataset: mean_std(
                [
                    seed_results[str(seed)]["epoch_0_seeded_head_test"]["by_dataset"][dataset]["column_accuracy@1"]
                    for seed in args.seed
                ]
            )
            for dataset in sorted(
                seed_results[str(args.seed[0])]["epoch_0_seeded_head_test"]["by_dataset"]
            )
        },
    }
    write_json_atomic(args.output_dir / "summary_metrics.json", aggregate)
    print(json.dumps(aggregate["three_seed"], indent=2))


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _round1_conclusion(
    summary: dict[str, Any], audit: dict[str, Any]
) -> dict[str, Any]:
    test = summary["three_seed"]["test"]
    dev = summary["three_seed"]["dev"]
    baselines = summary["baselines_test"]
    improvements = {
        dataset: test["by_dataset_column_accuracy@1"][dataset]["mean"]
        - baselines["by_dataset"][dataset]["majority_column_position_accuracy@1"]
        for dataset in test["by_dataset_column_accuracy@1"]
    }
    macro_gap = abs(
        dev["macro_column_accuracy@1"]["mean"]
        - test["macro_column_accuracy@1"]["mean"]
    )
    all_seed_improvements_positive = all(
        summary["seeds"][str(seed)]["trained"]["test"]["by_dataset"][dataset]["column_accuracy@1"]
        > baselines["by_dataset"][dataset]["majority_column_position_accuracy@1"]
        for seed in (13, 17, 23)
        for dataset in improvements
    )
    leakage = any(
        ids
        for pair in audit["cross_split_intersections"].values()
        for ids in pair.values()
    ) or bool(audit["id_conflicts"])

    gold_position_counts: Counter[int] = Counter()
    for dataset in audit["datasets"].values():
        for split in dataset["splits"].values():
            gold_position_counts.update(
                {
                    int(position): int(count)
                    for position, count in split["gold_column_position"].items()
                }
            )
    degenerate_gold_position = len(gold_position_counts) == 1
    perfect_majority_baseline = all(
        values["majority_column_position_accuracy@1"] == 1.0
        for values in baselines["by_dataset"].values()
    )

    if (
        all(value >= 0.10 for value in improvements.values())
        and all_seed_improvements_positive
        and macro_gap <= 0.05
        and not leakage
    ):
        code = "A"
        decision = "A：Oracle 候选列头可学习，满足进入第二轮的固定门槛。"
    elif any(value < 0.03 for value in improvements.values()) or macro_gap > 0.05:
        code = "C"
        decision = (
            "C：至少一湖相对 majority baseline 的提升不足 3 个百分点；"
            "暂停昂贵的完整 Stage-2 流水线并先修复实验可识别性。"
        )
    else:
        code = "uncertain"
        decision = "不确定区间：保留本轮结果，第二轮先做错误分析，不直接扩大训练。"

    identifiability = "identifiable"
    if degenerate_gold_position and perfect_majority_baseline:
        identifiability = "confounded_by_degenerate_gold_column_position"
    return {
        "decision_code": code,
        "decision": decision,
        "improvement_over_majority_by_dataset": improvements,
        "macro_dev_test_gap": macro_gap,
        "all_seed_improvements_positive": all_seed_improvements_positive,
        "data_leakage_detected": leakage,
        "gold_column_position_counts": {
            str(position): count
            for position, count in sorted(gold_position_counts.items())
        },
        "degenerate_gold_column_position": degenerate_gold_position,
        "perfect_majority_baseline": perfect_majority_baseline,
        "identifiability": identifiability,
    }


def _reader_cache_summary(output_dir: Path) -> dict[str, Any]:
    cache_root = output_dir / "reader_cache" / "oracle_positive"
    manifests = [
        _load_json(path) for path in sorted(cache_root.glob("*/*/manifest.json"))
    ]
    cache_bytes = sum(
        path.stat().st_size for path in cache_root.rglob("*") if path.is_file()
    )
    return {
        "manifests": len(manifests),
        "samples": sum(int(item["sample_count"]) for item in manifests),
        "recorded_elapsed_seconds": sum(
            float(item["elapsed_seconds_this_run"]) for item in manifests
        ),
        "peak_gpu_memory_bytes": max(
            int(item["peak_gpu_memory_bytes"] or 0) for item in manifests
        ),
        "oom_retry_examples": sum(
            int(item["reader_oom_retry_examples"]) for item in manifests
        ),
        "cache_bytes": cache_bytes,
    }


def run_report(args: argparse.Namespace) -> None:
    summary = _load_json(args.output_dir / "summary_metrics.json")
    audit = _load_json(args.output_dir / "data_audit.json")
    conclusion = _round1_conclusion(summary, audit)
    summary["conclusion"] = conclusion
    write_json_atomic(args.output_dir / "summary_metrics.json", summary)

    test = summary["three_seed"]["test"]
    baselines = summary["baselines_test"]
    improvements = conclusion["improvement_over_majority_by_dataset"]
    macro_gap = conclusion["macro_dev_test_gap"]
    decision = conclusion["decision"]
    cache = _reader_cache_summary(args.output_dir)
    bootstrap = summary["paired_bootstrap_seed_13"]
    old_stage1 = audit["old_stage1_retrieval_audit"]
    epoch0 = summary["three_seed"]["epoch_0_seeded_head_test"]
    seed13_modality = summary["seeds"]["13"]["trained"]["test"][
        "by_evidence_modality"
    ]

    lines = [
        "# Stage-2 Round-1 Results",
        "",
        "## Executive conclusion",
        "",
        decision,
        "",
        f"训练后的线性头在两湖、三个 seed 的 test accuracy 都是 "
        f"{_pct(test['macro_column_accuracy@1']['mean'])}，但 majority-column-position baseline "
        f"同样是 {_pct(baselines['majority_column_position_accuracy@1'])}。"
        "相对 majority 的提升为 0 个百分点，seed 13 paired bootstrap 95% CI 也是 [0, 0]。",
        "",
        f"决定性限制是位置退化：全部 {audit['total_usable_examples']:,} 条 Oracle 样本的 gold "
        f"column position 都是 0（`{json.dumps(conclusion['gold_column_position_counts'], sort_keys=True)}`）。"
        "因此本轮无法区分模型是否学习了语义，还是仅学会始终选择第一列。100% accuracy 不能作为 "
        "Oracle 语义可学习性的证据。",
        "",
        "继续/停止结论：暂不启动昂贵的 FOCUS、row filling、值生成与 final verification；"
        "先修复候选列顺序和位置偏置，再重跑 Oracle 可学习性实验。",
        "",
        "## Main results",
        "",
        "| Lake | Trained test accuracy (mean ± std) | Majority baseline | Improvement | Uniform expectation |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset, values in test["by_dataset_column_accuracy@1"].items():
        baseline = baselines["by_dataset"][dataset]
        lines.append(
            f"| {dataset} | {_pct(values['mean'])} ± {_pct(values['std'])} | "
            f"{_pct(baseline['majority_column_position_accuracy@1'])} | "
            f"{_pct(improvements[dataset])} | {_pct(baseline['uniform_random_expectation'])} |"
        )
    lines.extend(
        [
            "",
            f"Epoch-0 seeded head 的整体 test accuracy 为 "
            f"{_pct(epoch0['overall_column_accuracy@1']['mean'])} ± "
            f"{_pct(epoch0['overall_column_accuracy@1']['std'])}；训练后相对 uniform expectation "
            f"提高 {_pct(bootstrap['vs_uniform_random_expectation']['mean_difference'])}，"
            f"95% CI=[{_pct(bootstrap['vs_uniform_random_expectation']['ci95_low'])}, "
            f"{_pct(bootstrap['vs_uniform_random_expectation']['ci95_high'])}]。",
            "",
            f"相对 majority baseline 的 paired bootstrap 差为 "
            f"{_pct(bootstrap['vs_majority_column_position']['mean_difference'])}，"
            f"95% CI=[{_pct(bootstrap['vs_majority_column_position']['ci95_low'])}, "
            f"{_pct(bootstrap['vs_majority_column_position']['ci95_high'])}]（10,000 次，seed 13）。",
            "",
            "| Seed | Selected epoch | Train macro acc. | Dev macro acc. | Test macro acc. | Epoch-0 overall test acc. | Head training time |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for seed in (13, 17, 23):
        item = summary["seeds"][str(seed)]
        lines.append(
            f"| {seed} | {item['selected_epoch']} | {_pct(item['macro_accuracy']['train'])} | "
            f"{_pct(item['macro_accuracy']['dev'])} | {_pct(item['macro_accuracy']['test'])} | "
            f"{_pct(item['epoch_0_seeded_head_test']['column_accuracy@1'])} | "
            f"{item['elapsed_seconds']:.2f}s |"
        )
    lines.extend(
        [
            "",
            f"三 seed 的宏平均 dev-test absolute gap 是 {_pct(macro_gap)}，没有观察到 dev/test 崩塌；"
            "但位置标签退化使这一泛化结果同样不可识别。",
            "",
            "## Stratified diagnostics",
            "",
            "| Evidence modality | EntiTables test n | WDC test n | Seed 13 combined accuracy |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for modality in ("text_only", "image_only", "text_image"):
        ent_count = audit["datasets"]["entitables"]["splits"]["test"]["modality"].get(
            modality, 0
        )
        wdc_count = audit["datasets"]["wdc"]["splits"]["test"]["modality"].get(
            modality, 0
        )
        metric = seed13_modality.get(modality)
        rendered_accuracy = _pct(metric["column_accuracy@1"]) if metric else "n/a"
        lines.append(
            f"| {modality} | {ent_count:,} | {wdc_count:,} | {rendered_accuracy} |"
        )
    lines.extend(
        [
            "",
            "所有 modality bucket 都是 100%，但这与 gold position 恒为 0 完全相容，不能归因于 image evidence。"
            "WDC test 仅有 1 条 image-only 和 1 条 text+image 样本，明确不足以作显著性结论。",
            "",
            "Test 的 576 条样本均有 2 个候选列；train/dev 另有 19 条单候选列样本。"
            "候选列数分层同样不能消除第一列恒为 gold 的混杂。",
            "",
            "## Required questions",
            "",
            "1. **Oracle 正例条件下是否可学习？** " + decision + " "
            "训练过程能把头优化到始终正确，但相对可部署的 majority baseline 没有提升，"
            "所以本轮不能证明语义可学习性。",
            "2. **两湖是否一致？** 两湖 trained 与 majority 都是 100%，结论一致：均被相同的位置偏置混杂。",
            f"3. **是否存在明显过拟合或位置偏置？** dev-test gap 为 {_pct(macro_gap)}，没有明显 split 崩塌；"
            "但存在决定性的严重位置偏置：所有 gold 都在 position 0。",
            "4. **Image evidence 是否有可描述的信号？** 没有可归因信号。所有 modality 都是 100%，"
            "且 WDC image 样本不足；位置混杂优先于 modality 解释。",
            "5. **真实 retrieval 的限制？** Task D 未执行。旧 retrieval SHA 为 "
            f"`{old_stage1['retrievals'][0]['student_checkpoint_sha256'][0]}`，gate checkpoint SHA 为 "
            f"`{old_stage1['gate_checkpoint_sha256']}`，且缺两湖 dev/test retrieval。"
            "没有伪造 gate 或 provenance，因此不能裁决端到端瓶颈。",
            "6. **是否值得进入 FOCUS/row filling/生成评测？** 暂不值得。先在计算 gold local column 后，"
            "对候选列展示顺序做稳定且与标签无关的置换，确保 train/dev/test 的 gold positions 非退化；"
            "再加入 position-only、列顺序交换 counterfactual 和语义被遮蔽对照后重跑本轮。",
            "",
            "## Reproducibility and scope",
            "",
            "- Backbone: local `hf_models/Qwen3.5-9B`, frozen/eval, bf16 reader cache. "
            "Commands fixed `CUDA_VISIBLE_DEVICES=1`; manifests therefore record mapped device `cuda:0`.",
            "- Trainable parameters: `CandidateColumnScorer` only.",
            "- Oracle evidence top-k: 4; epochs: 3; learning rate: 1e-3; weight decay: 1e-4; seeds: 13/17/23.",
            "- Checkpoint selection: per-lake dev accuracy macro mean, then lower dev NLL, then earlier epoch.",
            f"- Reader cache: {cache['samples']:,} samples in {cache['manifests']} complete manifests, "
            f"{cache['cache_bytes'] / (1024 ** 2):.1f} MiB, recorded successful-run time "
            f"{cache['recorded_elapsed_seconds'] / 60:.2f} min, peak allocated GPU memory "
            f"{cache['peak_gpu_memory_bytes'] / (1024 ** 3):.2f} GiB.",
            f"- OOM policy: deterministic first-frame RGB resize to at most 1 MP was used for "
            f"{cache['oom_retry_examples']} example; all other examples used processor-default image sizing.",
            f"- Linear-head training time across three seeds: "
            f"{sum(float(summary['seeds'][str(seed)]['elapsed_seconds']) for seed in (13, 17, 23)):.2f}s.",
            "- Data audit: no missing query/target/evidence/image, cross-split overlap, or cross-lake ID conflict.",
            "- This experiment measures Oracle gold-target/gold-evidence column selection, not end-to-end row filling or joinability verification.",
            "- Full commands are in `commands.sh`; cache/build and training timings are in cache manifests and per-seed histories.",
            "",
        ]
    )
    (args.output_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(decision)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", nargs="+", default=list(DEFAULT_ROOTS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-dir", default="hf_models/Qwen3.5-9B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    _common(audit)
    audit.set_defaults(function=run_audit)

    cache = subparsers.add_parser("cache")
    _common(cache)
    cache.add_argument("--cache-root", type=Path, required=True)
    cache.add_argument("--split", nargs="+", choices=("train", "dev", "test"), required=True)
    cache.add_argument("--limit-per-dataset", type=int)
    cache.add_argument("--device", default="cuda:0")
    cache.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    cache.add_argument("--top-k-evidence", type=int, default=4)
    cache.add_argument("--shard-size", type=int, default=64)
    cache.set_defaults(function=run_cache)

    train = subparsers.add_parser("train")
    _common(train)
    train.add_argument("--cache-root", type=Path, required=True)
    train.add_argument("--seed", nargs="+", type=int, default=[13, 17, 23])
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.set_defaults(function=run_train)

    smoke = subparsers.add_parser("smoke-train")
    _common(smoke)
    smoke.add_argument("--cache-root", type=Path, required=True)
    smoke.set_defaults(function=run_smoke_train)

    evaluate = subparsers.add_parser("evaluate")
    _common(evaluate)
    evaluate.add_argument("--cache-root", type=Path, required=True)
    evaluate.add_argument("--seed", nargs="+", type=int, default=[13, 17, 23])
    evaluate.set_defaults(function=run_evaluate)

    report = subparsers.add_parser("report")
    _common(report)
    report.set_defaults(function=run_report)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.function(arguments)
