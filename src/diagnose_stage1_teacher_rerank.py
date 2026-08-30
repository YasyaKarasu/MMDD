#!/usr/bin/env python
"""Measure whether the Stage-1 Teacher improves raw top-k retrieval ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from mmdd_progress import progress
from mmdd_stage1.checkpoints import load_teacher
from mmdd_stage1.data import (
    TargetCandidate,
    TargetExample,
    load_edge_examples,
    load_target_examples,
)
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.protocol import validate_protocol_split
from mmdd_stage1.retrieval import (
    checkpoint_fingerprint,
    load_corpus_ids,
    load_or_build_raw_embedding_indices,
)
from mmdd_stage1.selection import write_json
from mmdd_stage1.significance import paired_bootstrap_delta
from mmdd_stage1.teacher_rerank import (
    evaluate_teacher_reranking,
    spearman_correlation,
)


def _max_recall_k(metrics: dict[str, Any]) -> int:
    values = [
        int(key.split("@", 1)[1])
        for key in metrics
        if key.startswith("recall@")
    ]
    if not values:
        raise ValueError("retrieval metrics contain no recall cutoff")
    return max(values)


def _max_mrr_k(metrics: dict[str, Any]) -> int:
    values = [
        int(key.split("@", 1)[1])
        for key in metrics
        if key.startswith("mrr@")
    ]
    if not values:
        raise ValueError("retrieval metrics contain no MRR cutoff")
    return max(values)


def _branch(raw_recall: float, reranked_recall: float) -> str:
    delta = reranked_recall - raw_recall
    if delta > 0.02:
        return "teacher_adds_retrieval_value"
    if delta < -0.02:
        return "teacher_is_harmful"
    return "teacher_has_no_incremental_value"


def _select_ensemble(payload: dict[str, Any]) -> dict[str, Any] | None:
    ensembles = payload.get("ensembles", [])
    if not ensembles:
        return None
    raw = payload["raw_direct"]
    wdc_dataset = next(
        (
            dataset
            for dataset in raw["by_dataset"]
            if dataset.lower().startswith("wdc")
        ),
        None,
    )
    eligible = ensembles
    if wdc_dataset is not None:
        raw_wdc = float(raw["by_dataset"][wdc_dataset]["recall@10"])
        guarded = [
            metrics
            for metrics in ensembles
            if float(metrics["by_dataset"][wdc_dataset]["recall@10"])
            >= raw_wdc
        ]
        if guarded:
            eligible = guarded
    best = max(
        eligible,
        key=lambda metrics: (
            float(metrics["recall@10"]),
            float(metrics[f"mrr@{_max_mrr_k(metrics)}"]),
            -float(metrics["alpha"]),
        ),
    )
    wdc_pass = wdc_dataset is None or (
        float(best["by_dataset"][wdc_dataset]["recall@10"])
        >= float(raw["by_dataset"][wdc_dataset]["recall@10"])
    )
    return {
        "alpha": float(best["alpha"]),
        "metrics": best,
        "wdc_dataset": wdc_dataset,
        "wdc_guard_satisfied": wdc_pass,
        "accepted": float(best["recall@10"]) >= 0.42 and wdc_pass,
    }


def _evidence_to_table_examples(args: argparse.Namespace) -> list[TargetExample]:
    examples = []
    for path_value in args.edge_dev_data or []:
        path = Path(path_value)
        for edge in load_edge_examples(
            path, split=args.dev_split, dataset_name=path.stem
        ):
            if edge.source_type not in {"text", "image"} or edge.destination_type != "table":
                continue
            examples.append(
                TargetExample(
                    edge.query_id,
                    tuple(TargetCandidate(candidate_id, ()) for candidate_id in edge.candidate_ids),
                    edge.positive_index,
                    edge.positive_index,
                    dataset=edge.dataset,
                    split=edge.split,
                    positive_target_ids=(edge.candidate_ids[edge.positive_index],),
                )
            )
    return examples


def _markdown(payload: dict[str, Any]) -> str:
    raw = payload["raw_direct"]
    reranked = payload["teacher_reranked"]
    max_k = _max_recall_k(reranked)
    rows = [
        "# Task 2b: Teacher rerank diagnostic",
        "",
        f"| Scope | Ranking | R@10 | R@{max_k} | MRR@{max_k} |",
        "| --- | --- | ---: | ---: | ---: |",
        f"| overall | raw | {raw['recall@10']:.2%} | "
        f"{raw[f'recall@{max_k}']:.2%} | {raw[f'mrr@{max_k}']:.4f} |",
        f"| overall | teacher | {reranked['recall@10']:.2%} | "
        f"{reranked[f'recall@{max_k}']:.2%} | "
        f"{reranked[f'mrr@{max_k}']:.4f} |",
    ]
    for dataset in payload["raw_direct"]["by_dataset"]:
        for name, metrics in (
            ("raw", payload["raw_direct"]["by_dataset"][dataset]),
            ("teacher", payload["teacher_reranked"]["by_dataset"][dataset]),
        ):
            rows.append(
                f"| {dataset} | {name} | {metrics['recall@10']:.2%} | "
                f"{metrics[f'recall@{max_k}']:.2%} | "
                f"{metrics[f'mrr@{max_k}']:.4f} |"
            )
    rows.extend(
        [
            "",
            f"- Decision: `{payload['decision']}`",
            f"- R@10 delta: {payload['recall@10_delta']:+.2%}",
            "- Paired R@10 delta / 95% CI: "
            f"{payload['paired_recall@10_vs_raw']['mean']:+.2%} "
            f"[{payload['paired_recall@10_vs_raw']['ci_low']:+.2%}, "
            f"{payload['paired_recall@10_vs_raw']['ci_high']:+.2%}]",
            f"- Mean per-query Spearman(raw, Teacher): {payload['spearman']['mean']:.4f}",
            f"- Median per-query Spearman(raw, Teacher): {payload['spearman']['median']:.4f}",
            "",
        ]
    )
    if payload.get("ensembles"):
        rows.extend(
            [
                "## Raw + Teacher z-score ensemble",
                "",
                f"| Alpha | Scope | R@10 | R@{max_k} | MRR@{max_k} |",
                "| ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for ensemble in payload["ensembles"]:
            alpha = float(ensemble["alpha"])
            rows.append(
                f"| {alpha:g} | overall | {ensemble['recall@10']:.2%} | "
                f"{ensemble[f'recall@{max_k}']:.2%} | "
                f"{ensemble[f'mrr@{max_k}']:.4f} |"
            )
            for dataset, metrics in ensemble["by_dataset"].items():
                rows.append(
                    f"| {alpha:g} | {dataset} | {metrics['recall@10']:.2%} | "
                    f"{metrics[f'recall@{max_k}']:.2%} | "
                    f"{metrics[f'mrr@{max_k}']:.4f} |"
                )
        selection = payload["ensemble_selection"]
        rows.extend(
            [
                "",
                f"- Selected alpha: `{selection['alpha']:g}`",
                f"- WDC guard satisfied: **{selection['wdc_guard_satisfied']}**",
                f"- Task C acceptance: **{selection['accepted']}**",
                "",
            ]
        )
    evidence_to_table = payload.get("evidence_to_table")
    if evidence_to_table is not None:
        evidence_max_k = _max_recall_k(evidence_to_table["teacher_reranked"])
        rows.extend(
            [
                "## Evidence→table rerank",
                "",
                f"| Scope | Ranking | R@10 | R@{evidence_max_k} | MRR@{evidence_max_k} |",
                "| --- | --- | ---: | ---: | ---: |",
                f"| overall | raw | {evidence_to_table['raw_direct']['recall@10']:.2%} | "
                f"{evidence_to_table['raw_direct'][f'recall@{evidence_max_k}']:.2%} | "
                f"{evidence_to_table['raw_direct'][f'mrr@{evidence_max_k}']:.4f} |",
                f"| overall | teacher | {evidence_to_table['teacher_reranked']['recall@10']:.2%} | "
                f"{evidence_to_table['teacher_reranked'][f'recall@{evidence_max_k}']:.2%} | "
                f"{evidence_to_table['teacher_reranked'][f'mrr@{evidence_max_k}']:.4f} |",
                "",
            ]
        )
        if evidence_to_table.get("ensembles"):
            rows.extend(
                [
                    f"| Alpha | Ensemble R@10 | R@{evidence_max_k} | MRR@{evidence_max_k} |",
                    "| ---: | ---: | ---: | ---: |",
                ]
            )
            for ensemble in evidence_to_table["ensembles"]:
                rows.append(
                    f"| {ensemble['alpha']:g} | {ensemble['recall@10']:.2%} | "
                    f"{ensemble[f'recall@{evidence_max_k}']:.2%} | "
                    f"{ensemble[f'mrr@{evidence_max_k}']:.4f} |"
                )
            rows.append("")
    return "\n".join(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.raw_top_k <= 0 or args.teacher_batch_size <= 0:
        raise ValueError("--raw-top-k and --teacher-batch-size must be positive")
    validate_protocol_split("dev_gate", args.dev_split)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    store = FeatureStore.from_path(
        Path(args.features), cache_size=args.feature_cache_size
    )
    teacher_path = Path(args.teacher_checkpoint)
    teacher = load_teacher(teacher_path, device)
    hidden_dim = store.teacher_dimension()
    if hidden_dim is None or teacher.input_dim != hidden_dim:
        raise ValueError("Teacher checkpoint does not match cached hidden states")
    examples = [
        example
        for path_value in args.dev_data
        for example in load_target_examples(
            Path(path_value), split=args.dev_split, dataset_name=Path(path_value).stem
        )
    ]
    evidence_to_table_examples = _evidence_to_table_examples(args)
    corpus_path = Path(args.corpus)
    corpus_sha256 = checkpoint_fingerprint(corpus_path)
    ids_by_type = load_corpus_ids(corpus_path, store)
    raw_indices = load_or_build_raw_embedding_indices(
        store,
        ids_by_type,
        Path(args.raw_index_root),
        corpus_sha256=corpus_sha256,
        batch_size=args.index_batch_size,
        m=args.hnsw_m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )

    raw_hits_by_query = [
        raw_indices.search(example.query_id, "table", args.raw_top_k)
        for example in progress(
            examples, desc="Teacher rerank preflight", unit="query"
        )
    ]
    evidence_to_table_hits = [
        raw_indices.search(example.query_id, "table", args.raw_top_k)
        for example in progress(
            evidence_to_table_examples,
            desc="Evidence-to-table rerank preflight",
            unit="query",
        )
    ]
    missing_teacher_ids = sorted(
        {
            object_id
            for example, raw_hits in [
                *zip(examples, raw_hits_by_query),
                *zip(evidence_to_table_examples, evidence_to_table_hits),
            ]
            for object_id in [
                example.query_id,
                *(target_id for target_id, _score in raw_hits),
            ]
            if not store.has_teacher_features(object_id)
        }
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if missing_teacher_ids:
        missing_path = output_dir / "missing_teacher_features.jsonl"
        missing_set = set(missing_teacher_ids)
        with missing_path.open("w", encoding="utf-8") as handle:
            for example, raw_hits in [
                *zip(examples, raw_hits_by_query),
                *zip(evidence_to_table_examples, evidence_to_table_hits),
            ]:
                candidate_ids = [
                    target_id
                    for target_id, _score in raw_hits
                    if target_id in missing_set
                ]
                if example.query_id in missing_set or candidate_ids:
                    handle.write(
                        json.dumps(
                            {
                                "query_id": example.query_id,
                                "candidate_ids": candidate_ids,
                                "split": "dev",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        preflight = {
            "format_version": 1,
            "queries": len(examples),
            "evidence_to_table_queries": len(evidence_to_table_examples),
            "raw_top_k": args.raw_top_k,
            "missing_teacher_objects": len(missing_teacher_ids),
            "missing_teacher_data": str(missing_path.resolve()),
        }
        write_json(output_dir / "preflight.json", preflight)
        raise ValueError(
            f"Teacher rerank requires hidden states for {len(missing_teacher_ids)} "
            f"additional objects; cache the IDs listed by {missing_path} and rerun"
        )
    write_json(
        output_dir / "preflight.json",
        {
            "format_version": 1,
            "queries": len(examples),
            "evidence_to_table_queries": len(evidence_to_table_examples),
            "raw_top_k": args.raw_top_k,
            "missing_teacher_objects": 0,
        },
    )

    evaluation = evaluate_teacher_reranking(
        teacher,
        examples,
        raw_hits_by_query,
        store,
        device=device,
        batch_size=args.teacher_batch_size,
        ensemble_alphas=args.ensemble_alphas,
        return_per_query=True,
    )
    raw_metrics = evaluation["raw_direct"]
    teacher_metrics = evaluation["teacher_reranked"]
    payload = {
        "format_version": 1,
        "queries": len(examples),
        "teacher_checkpoint": str(teacher_path.resolve()),
        "teacher_checkpoint_sha256": checkpoint_fingerprint(teacher_path),
        "corpus_sha256": corpus_sha256,
        "raw_top_k": args.raw_top_k,
        "raw_direct": raw_metrics,
        "teacher_reranked": teacher_metrics,
        "recall@10_delta": evaluation["recall@10_delta"],
        "paired_recall@10_vs_raw": paired_bootstrap_delta(
            teacher_metrics["per_query"]["recall@10"],
            raw_metrics["per_query"]["recall@10"],
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
        ),
        "spearman": evaluation["spearman"],
        "decision": _branch(
            float(raw_metrics["recall@10"]),
            float(teacher_metrics["recall@10"]),
        ),
        "ensembles": evaluation["ensembles"],
    }
    payload["ensemble_selection"] = _select_ensemble(payload)
    if evidence_to_table_examples:
        payload["evidence_to_table"] = evaluate_teacher_reranking(
            teacher,
            evidence_to_table_examples,
            evidence_to_table_hits,
            store,
            device=device,
            batch_size=args.teacher_batch_size,
            ensemble_alphas=args.ensemble_alphas,
            return_per_query=True,
        )
    selection = payload["ensemble_selection"]
    if selection is not None and selection["accepted"]:
        interface_path = output_dir / "ensemble_config.json"
        write_json(
            interface_path,
            {
                "format_version": 1,
                "score": "alpha*z(teacher)+(1-alpha)*z(raw_cosine)",
                "normalization": "per_query_candidate_list",
                "alpha": selection["alpha"],
                "teacher_checkpoint": str(teacher_path.resolve()),
                "teacher_checkpoint_sha256": payload[
                    "teacher_checkpoint_sha256"
                ],
                "raw_top_k": args.raw_top_k,
                "corpus_sha256": corpus_sha256,
            },
        )
        payload["ensemble_interface"] = str(interface_path.resolve())
    write_json(output_dir / "metrics.json", payload)
    report = _markdown(payload)
    (output_dir / "RESULTS.md").write_text(report, encoding="utf-8")
    summary_path = output_dir.parent / "RESULTS.md"
    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write(report + "\n")
    print(
        json.dumps(
            {
                "raw_direct": raw_metrics,
                "teacher_reranked": teacher_metrics,
                "recall@10_delta": payload["recall@10_delta"],
                "paired_recall@10_vs_raw": payload["paired_recall@10_vs_raw"],
                "spearman": payload["spearman"],
                "decision": payload["decision"],
                "ensembles": payload["ensembles"],
                "ensemble_selection": payload["ensemble_selection"],
                "evidence_to_table": payload.get("evidence_to_table"),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--dev-data", required=True, nargs="+")
    parser.add_argument(
        "--edge-dev-data",
        nargs="*",
        default=[],
        help="Optional edge lists for a text/image-to-table rerank diagnostic.",
    )
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--raw-index-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dev-split", default="dev", choices=["dev"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--raw-top-k", type=int, default=100)
    parser.add_argument("--teacher-batch-size", type=int, default=16)
    parser.add_argument(
        "--ensemble-alphas",
        type=float,
        nargs="*",
        default=[0.3, 0.4, 0.5, 0.6, 0.7],
        help="Teacher weights for per-query z-score raw/Teacher ensembles.",
    )
    parser.add_argument("--feature-cache-size", type=int, default=60_000)
    parser.add_argument("--index-batch-size", type=int, default=1024)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=100)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
