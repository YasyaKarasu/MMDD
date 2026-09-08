#!/usr/bin/env python
"""Evaluate Stage-1 round-3 retrieval and Teacher-ensemble baselines.

The four systems are deliberately evaluated with independent ANN pools for every
reported k.  Teacher reranking is direct Q-to-table reranking, while the raw and
Student systems retain the prescribed zero/one-hop weighted-RRF path retrieval.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.checkpoints import load_path_aggregator, load_student, load_teacher
from mmdd_stage1.data import TargetExample, load_target_examples
from mmdd_stage1.evaluation import evaluate_student_retrieval
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.objectives import PATH_AGGREGATIONS, PathAggregator
from mmdd_stage1.retrieval import (
    RawEmbeddingANNIndices,
    StudentANNIndices,
    checkpoint_fingerprint,
    load_corpus_ids,
    load_or_build_raw_embedding_indices,
)
from mmdd_stage1.selection import load_stage1_selection, write_json
from mmdd_stage1.significance import paired_bootstrap_delta
from mmdd_stage1.teacher_rerank import teacher_ensemble_metrics as _teacher_ensemble_metrics

SYSTEMS = ("raw", "student", "raw_ensemble", "student_ensemble")


def _positive_ints(value: str) -> tuple[int, ...]:
    values = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def _system_names(value: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = sorted(set(values) - set(SYSTEMS))
    if not values or invalid:
        raise argparse.ArgumentTypeError(
            f"--systems must be a comma-separated subset of {', '.join(SYSTEMS)}"
        )
    return tuple(dict.fromkeys(values))


def _modality_weight(value: str) -> tuple[str, float]:
    modality, separator, raw_weight = value.partition("=")
    if separator != "=" or modality not in {"text", "image"}:
        raise argparse.ArgumentTypeError("modality weights must use text=VALUE or image=VALUE")
    try:
        weight = float(raw_weight)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("modality weight must be numeric") from exc
    if weight < 0:
        raise argparse.ArgumentTypeError("modality weight must be non-negative")
    return modality, weight


def _examples(paths: Sequence[str], max_queries: int | None) -> list[TargetExample]:
    examples = [
        example
        for value in paths
        for example in load_target_examples(
            Path(value), split="dev", dataset_name=Path(value).stem
        )
    ]
    return examples if max_queries is None else examples[:max_queries]


def _teacher_feature_preflight(
    examples: Sequence[TargetExample],
    indices_by_system: dict[str, RawEmbeddingANNIndices | StudentANNIndices],
    store: FeatureStore,
    *,
    recall_ks: Sequence[int],
    gamma: int,
    output_dir: Path,
    objects_path: Path | None,
) -> None:
    query_ids = [example.query_id for example in examples]
    pool_sizes = sorted({gamma * k for k in recall_ks})
    required_ids = set(query_ids)
    for indices in indices_by_system.values():
        for pool_size in pool_sizes:
            required_ids.update(
                target_id
                for hits in indices.search_many(query_ids, "table", pool_size)
                for target_id, _score in hits
            )
    missing = sorted(
        object_id for object_id in required_ids if not store.has_teacher_features(object_id)
    )
    payload: dict[str, Any] = {
        "systems": sorted(indices_by_system),
        "candidate_pool_sizes": pool_sizes,
        "required_teacher_objects": len(required_ids),
        "missing_teacher_objects": len(missing),
    }
    if missing:
        ids_path = output_dir / "missing_teacher_ids.jsonl"
        with ids_path.open("w", encoding="utf-8") as handle:
            for object_id in missing:
                handle.write(json.dumps({"object_id": object_id}) + "\n")
        payload["missing_teacher_ids"] = str(ids_path.resolve())
        if objects_path is not None:
            missing_set = set(missing)
            found = set()
            input_path = output_dir / "missing_teacher_input.jsonl"
            with objects_path.open(encoding="utf-8") as source, input_path.open(
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
            if found != missing_set:
                raise KeyError(
                    f"Object source is missing {len(missing_set - found)} Teacher candidates"
                )
            payload["missing_teacher_input"] = str(input_path.resolve())
        write_json(output_dir / "teacher_preflight.json", payload)
        raise ValueError(
            f"Teacher reranking requires hidden states for {len(missing)} additional objects"
        )
    write_json(output_dir / "teacher_preflight.json", payload)


def _comparison(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    *,
    recall_ks: Sequence[int],
    iterations: int,
    seed: int,
    reference_name: str = "raw_embedding",
) -> dict[str, Any] | None:
    if iterations <= 0:
        return None

    def scope(values: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
        return {
            f"recall@{k}": paired_bootstrap_delta(
                values["per_query"][f"recall@{k}"],
                baseline["per_query"][f"recall@{k}"],
                iterations=iterations,
                seed=seed,
            )
            for k in recall_ks
        }

    return {
        "reference": reference_name,
        "overall": scope(candidate, reference),
        "by_dataset": {
            dataset: scope(candidate["by_dataset"][dataset], reference["by_dataset"][dataset])
            for dataset in candidate["by_dataset"]
        },
    }


def _comparison_view(metrics: dict[str, Any], channel: str | None = None) -> dict[str, Any]:
    """Select the comparable per-query metric view from a system result."""

    if channel is None:
        return metrics
    return {
        "queries": metrics["queries"],
        "per_query": metrics["per_query"][channel],
        "by_dataset": {
            dataset: {
                "queries": values["queries"],
                "per_query": values["per_query"][channel],
            }
            for dataset, values in metrics["by_dataset"].items()
        },
    }


def _format_ci(comparison: dict[str, Any] | None, k: int) -> str:
    if comparison is None:
        return "—"
    delta = comparison["overall"][f"recall@{k}"]
    return f"Δ {delta['mean']:+.2%}, CI [{delta['ci_low']:+.2%}, {delta['ci_high']:+.2%}]"


def _markdown(payload: dict[str, Any]) -> str:
    recall_ks = payload["parameters"]["recall_ks"]
    max_k = max(recall_ks)
    rows = [
        f"# {payload.get('title', 'Task E: round-3 baselines')}",
        "",
        "Teacher-ensemble rows use direct Q→T reranking of an independently retrieved "
        "`gamma × k` candidate pool. Coverage is therefore not applicable to them; the "
        "raw and Student rows use the complete zero/one-hop weighted-RRF retrieval path.",
        "",
        "| System | "
        + " | ".join(f"R@{k}" for k in recall_ks)
        + f" | MRR@{max_k} | Coverage@10 | R@10 vs raw |",
        "| --- | " + " | ".join("---:" for _ in recall_ks) + " | ---: | ---: | --- |",
    ]
    for name, result in payload["systems"].items():
        metrics = result["metrics"]
        coverage = metrics.get("positive_evidence_path_coverage@10")
        rows.append(
            f"| {result['label']} | "
            + " | ".join(f"{metrics[f'recall@{k}']:.2%}" for k in recall_ks)
            + f" | {metrics[f'mrr@{max_k}']:.4f} | "
            + (f"{coverage:.2%}" if coverage is not None else "—")
            + f" | {_format_ci(result.get('vs_raw'), 10)} |"
        )
    deployment = payload["systems"].get("student_ensemble")
    if deployment is not None and deployment.get("vs_raw_ensemble") is not None:
        gap = deployment["vs_raw_ensemble"]["overall"]["recall@10"]
        rows.extend(
            [
                "",
                "## Deployment gap: system (4) vs system (3)",
                "",
                "Student + Teacher ensemble minus Raw + Teacher ensemble at R@10: "
                f"{gap['mean']:+.2%}, paired 95% CI "
                f"[{gap['ci_low']:+.2%}, {gap['ci_high']:+.2%}].",
            ]
        )
    rows.extend(["", "## R@10 by dataset (paired 95% CI vs matching raw channel)", ""])
    rows.extend(
        [
            "| System | Dataset | R@10 | Δ vs raw / 95% CI |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for result in payload["systems"].values():
        for dataset, metrics in result["metrics"]["by_dataset"].items():
            comparison = result.get("vs_raw")
            ci = "—"
            if comparison is not None:
                delta = comparison["by_dataset"][dataset]["recall@10"]
                ci = f"{delta['mean']:+.2%} [{delta['ci_low']:+.2%}, {delta['ci_high']:+.2%}]"
            rows.append(
                f"| {result['label']} | {dataset} | {metrics['recall@10']:.2%} | {ci} |"
            )
    rows.extend(
        [
            "",
            "## Parameters and timing",
            "",
            f"- `recall_ks={recall_ks}`, `gamma={payload['parameters']['gamma']}`, "
            f"`gamma_evidence={payload['parameters']['gamma_evidence']}`",
            f"- Teacher ensemble: α={payload['parameters']['teacher_alpha']:.3g}; "
            f"Teacher batch size={payload['parameters']['teacher_batch_size']}",
            f"- Device: `{payload['hardware']['device']}`; CUDA available: "
            f"`{payload['hardware']['cuda_available']}`",
        ]
    )
    for result in payload["systems"].values():
        timing = result["timing"]
        rows.append(
            f"- {result['label']}: {timing['average_seconds_per_query_per_k']:.4f} s/query/k "
            f"({timing['total_seconds']:.1f} s total)."
        )
    return "\n".join(rows) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    evidence_modality_weights = dict(
        getattr(args, "evidence_modality_weights", [])
    )
    evidence_types = tuple(getattr(args, "evidence_types", ("text", "image")))
    if args.gamma <= 0 or args.gamma_evidence <= 0 or args.teacher_batch_size <= 0:
        raise ValueError("gamma, gamma-evidence, and teacher-batch-size must be positive")
    if not 0 <= args.teacher_alpha <= 1:
        raise ValueError("--teacher-alpha must be in [0, 1]")
    if args.max_queries is not None and args.max_queries <= 0:
        raise ValueError("--max-queries must be positive")
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    selection = load_stage1_selection(Path(args.selection))
    checkpoint_path = Path(selection["best_checkpoint"])
    checkpoint_sha256 = checkpoint_fingerprint(checkpoint_path)
    if checkpoint_sha256 != selection["best_checkpoint_sha256"]:
        raise ValueError("Selection and Student checkpoint fingerprints differ")
    corpus_path = Path(args.corpus)
    corpus_sha256 = checkpoint_fingerprint(corpus_path)
    if corpus_sha256 != selection["corpus_sha256"]:
        raise ValueError("Selection and corpus fingerprints differ")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    store = FeatureStore.from_path(
        Path(args.features),
        cache_size=args.feature_cache_size,
        teacher_paths=tuple(Path(value) for value in args.teacher_features),
    )
    examples = _examples(args.dev_data, args.max_queries)
    student = load_student(checkpoint_path, device)
    student.eval()
    student_indices = StudentANNIndices(
        student,
        store,
        Path(selection["best_index"]),
        device=device,
        checkpoint_sha256=checkpoint_sha256,
        corpus_sha256=corpus_sha256,
    )
    need_raw = any(name in args.systems for name in ("raw", "raw_ensemble"))
    raw_indices = None
    if need_raw:
        raw_indices = load_or_build_raw_embedding_indices(
            store,
            load_corpus_ids(corpus_path, store),
            Path(selection["raw_embedding_index"]),
            corpus_sha256=corpus_sha256,
            batch_size=args.index_batch_size,
            m=args.hnsw_m,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
        )
    ensemble_indices: dict[str, RawEmbeddingANNIndices | StudentANNIndices] = {}
    if "raw_ensemble" in args.systems:
        assert raw_indices is not None
        ensemble_indices["raw_ensemble"] = raw_indices
    if "student_ensemble" in args.systems:
        ensemble_indices["student_ensemble"] = student_indices
    if ensemble_indices:
        _teacher_feature_preflight(
            examples,
            ensemble_indices,
            store,
            recall_ks=args.recall_ks,
            gamma=args.gamma,
            output_dir=output_dir,
            objects_path=Path(args.objects) if args.objects else None,
        )
    if args.preflight_only:
        if not ensemble_indices:
            raise ValueError("--preflight-only requires at least one ensemble system")
        payload = json.loads(
            (output_dir / "teacher_preflight.json").read_text(encoding="utf-8")
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload
    systems: dict[str, dict[str, Any]] = {}
    saved_aggregator = load_path_aggregator(checkpoint_path)
    evidence_aggregation = getattr(args, "evidence_aggregation", None)
    evidence_top_k = getattr(args, "evidence_top_k", None)
    evidence_temperature = getattr(args, "evidence_temperature", None)
    evidence_power = getattr(args, "evidence_power", None)
    path_combination = getattr(args, "path_combination", None)
    evidence_threshold = getattr(args, "evidence_threshold", None)
    aggregator = PathAggregator(
        evidence_aggregation or saved_aggregator.evidence_aggregation,
        evidence_top_k
        if evidence_top_k is not None
        else saved_aggregator.top_k,
        temperature=(
            evidence_temperature
            if evidence_temperature is not None
            else saved_aggregator.temperature
        ),
        power=(
            evidence_power
            if evidence_power is not None
            else saved_aggregator.power
        ),
        path_combination=(
            path_combination
            if path_combination is not None
            else saved_aggregator.path_combination
        ),
        threshold=(
            evidence_threshold
            if evidence_threshold is not None
            else saved_aggregator.threshold
        ),
    )
    fusion_mode = getattr(args, "fusion_mode", "weighted_rrf")
    direct_weight = getattr(args, "direct_weight", 1.0)
    evidence_weight = getattr(args, "evidence_weight", 0.05)
    score_normalization = getattr(args, "fusion_score_normalization", "none")
    score_temperature = getattr(args, "fusion_score_temperature", 1.0)
    path_edge_normalization = getattr(args, "path_edge_normalization", "none")

    def full_system(name: str, label: str, indices: RawEmbeddingANNIndices | StudentANNIndices) -> None:
        start = time.perf_counter()
        metrics = evaluate_student_retrieval(
            examples,
            indices,
            recall_ks=args.recall_ks,
            gamma=args.gamma,
            gamma_evidence=args.gamma_evidence,
            direct_k=args.direct_k,
            evidence_k=args.evidence_k,
            targets_per_evidence=args.targets_per_evidence,
            evidence_types=evidence_types,
            evidence_aggregation=aggregator.evidence_aggregation,
            evidence_top_k=aggregator.top_k,
            evidence_temperature=aggregator.temperature,
            evidence_power=aggregator.power,
            path_combination=aggregator.path_combination,
            evidence_threshold=aggregator.threshold,
            evidence_modality_weights=evidence_modality_weights,
            fusion_mode=fusion_mode,
            direct_weight=direct_weight,
            evidence_weight=evidence_weight,
            fusion_score_normalization=score_normalization,
            fusion_score_temperature=score_temperature,
            path_edge_normalization=path_edge_normalization,
            return_per_query=True,
        )
        elapsed = time.perf_counter() - start
        systems[name] = {
            "label": label,
            "kind": "zero_one_hop_weighted_rrf",
            "metrics": metrics,
            "timing": {
                "total_seconds": elapsed,
                "average_seconds_per_query_per_k": elapsed / (len(examples) * len(args.recall_ks)),
            },
        }

    if "raw" in args.systems:
        assert raw_indices is not None
        full_system("raw", "Raw embedding", raw_indices)
    if "student" in args.systems:
        full_system(
            "student",
            getattr(args, "student_label", "Final Student (epoch 0)"),
            student_indices,
        )

    if any(name in args.systems for name in ("raw_ensemble", "student_ensemble")):
        teacher_path = Path(args.teacher_checkpoint)
        teacher = load_teacher(
            teacher_path,
            device,
            table_tokens_per_group=getattr(
                args, "teacher_table_tokens_per_group", None
            ),
        )
        if store.teacher_dimension() != teacher.input_dim:
            raise ValueError("Teacher checkpoint does not match the feature store hidden dimension")
        teacher.eval()

        def ensemble_system(
            name: str,
            label: str,
            indices: RawEmbeddingANNIndices | StudentANNIndices,
        ) -> None:
            score_cache: dict[tuple[str, str], float] = {}
            metrics, timing = _teacher_ensemble_metrics(
                teacher,
                examples,
                indices,
                store,
                recall_ks=args.recall_ks,
                gamma=args.gamma,
                alpha=args.teacher_alpha,
                device=device,
                batch_size=args.teacher_batch_size,
                score_cache=score_cache,
            )
            systems[name] = {
                "label": label,
                "kind": "direct_teacher_ensemble",
                "metrics": metrics,
                "timing": timing,
            }

        if "raw_ensemble" in args.systems:
            assert raw_indices is not None
            ensemble_system("raw_ensemble", "Raw + Teacher ensemble", raw_indices)
        if "student_ensemble" in args.systems:
            ensemble_system("student_ensemble", "Student + Teacher ensemble", student_indices)

    raw_full = systems.get("raw")
    if raw_full is not None:
        raw_fused = _comparison_view(raw_full["metrics"], channel="fused")
        raw_direct = _comparison_view(raw_full["metrics"], channel="direct")
        for name, result in systems.items():
            if name == "raw":
                continue
            candidate = result["metrics"]
            reference = raw_direct
            if result["kind"] == "zero_one_hop_weighted_rrf":
                candidate = _comparison_view(candidate, channel="fused")
                reference = raw_fused
            result["vs_raw"] = _comparison(
                candidate,
                reference,
                recall_ks=args.recall_ks,
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed,
            )
    if "raw_ensemble" in systems and "student_ensemble" in systems:
        systems["student_ensemble"]["vs_raw_ensemble"] = _comparison(
            systems["student_ensemble"]["metrics"],
            systems["raw_ensemble"]["metrics"],
            recall_ks=args.recall_ks,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
            reference_name="raw_teacher_ensemble",
        )

    payload = {
        "format_version": 1,
        "title": getattr(args, "title", "Task E: round-3 baselines"),
        "selection": str(Path(args.selection).resolve()),
        "student_checkpoint": str(checkpoint_path.resolve()),
        "student_checkpoint_sha256": checkpoint_sha256,
        "teacher_checkpoint": str(Path(args.teacher_checkpoint).resolve()),
        "teacher_checkpoint_sha256": checkpoint_fingerprint(Path(args.teacher_checkpoint)),
        "corpus": str(corpus_path.resolve()),
        "corpus_sha256": corpus_sha256,
        "parameters": {
            "recall_ks": list(args.recall_ks),
            "gamma": args.gamma,
            "gamma_evidence": args.gamma_evidence,
            "direct_k": args.direct_k,
            "evidence_k": args.evidence_k,
            "targets_per_evidence": args.targets_per_evidence,
            "teacher_alpha": args.teacher_alpha,
            "teacher_batch_size": args.teacher_batch_size,
            "teacher_table_tokens_per_group": getattr(
                args, "teacher_table_tokens_per_group", None
            ),
            "evidence_aggregation": aggregator.evidence_aggregation,
            "evidence_top_k": aggregator.top_k,
            "evidence_temperature": aggregator.temperature,
            "evidence_power": aggregator.power,
            "path_combination": aggregator.path_combination,
            "evidence_threshold": aggregator.threshold,
            "fusion_mode": fusion_mode,
            "direct_weight": direct_weight,
            "evidence_weight": evidence_weight,
            "fusion_score_normalization": score_normalization,
            "fusion_score_temperature": score_temperature,
            "path_edge_normalization": path_edge_normalization,
            "evidence_types": list(evidence_types),
            "evidence_modality_weights": evidence_modality_weights,
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_seed": args.bootstrap_seed,
            "max_queries": args.max_queries,
        },
        "hardware": {
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cpu": platform.processor() or platform.machine(),
            "torch": torch.__version__,
        },
        "systems": systems,
    }
    write_json(output_dir / "metrics.json", payload)
    (output_dir / "RESULTS.md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument(
        "--teacher-features",
        nargs="*",
        default=[],
        help="Optional Teacher-only cache directories or teacher_manifest.jsonl paths.",
    )
    parser.add_argument("--dev-data", nargs="+", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument(
        "--objects",
        help="Full object JSONL used to materialize any Teacher-cache preflight misses.",
    )
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="Task E: round-3 baselines")
    parser.add_argument("--student-label", default="Final Student (epoch 0)")
    parser.add_argument("--systems", type=_system_names, default=SYSTEMS)
    parser.add_argument("--recall-ks", type=_positive_ints, default=(10, 20, 30, 40, 50))
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--gamma-evidence", type=int, default=2)
    parser.add_argument("--direct-k", type=int)
    parser.add_argument("--evidence-k", type=int)
    parser.add_argument("--targets-per-evidence", type=int)
    parser.add_argument("--teacher-alpha", type=float, default=0.7)
    parser.add_argument("--teacher-batch-size", type=int, default=16)
    parser.add_argument(
        "--teacher-table-tokens-per-group",
        type=int,
        help="Override the checkpoint's table schema/row token budget.",
    )
    parser.add_argument("--feature-cache-size", type=int, default=60_000)
    parser.add_argument("--index-batch-size", type=int, default=1024)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=100)
    parser.add_argument("--evidence-aggregation", choices=sorted(PATH_AGGREGATIONS))
    parser.add_argument("--evidence-top-k", type=int)
    parser.add_argument("--evidence-temperature", type=float)
    parser.add_argument("--evidence-power", type=float)
    parser.add_argument("--path-combination", choices=["sum", "min", "product"])
    parser.add_argument("--evidence-threshold", type=float)
    parser.add_argument(
        "--path-edge-normalization",
        choices=["none", "zscore"],
        default="none",
        help="Reference-only normalization of edge scores before path aggregation.",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=["rrf", "weighted_rrf", "gated", "normalized_score", "normalized_rrc"],
        default="weighted_rrf",
    )
    parser.add_argument("--direct-weight", type=float, default=1.0)
    parser.add_argument("--evidence-weight", type=float, default=0.05)
    parser.add_argument(
        "--fusion-score-normalization",
        choices=["none", "zscore", "minmax", "softmax"],
        default="none",
    )
    parser.add_argument("--fusion-score-temperature", type=float, default=1.0)
    parser.add_argument(
        "--evidence-types",
        nargs="+",
        choices=["text", "image"],
        default=["text", "image"],
    )
    parser.add_argument(
        "--evidence-modality-weights",
        nargs="*",
        type=_modality_weight,
        default=[],
        metavar="MODALITY=WEIGHT",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
