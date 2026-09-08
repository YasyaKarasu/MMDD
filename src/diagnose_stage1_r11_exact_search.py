#!/usr/bin/env python
"""Compare R11 raw/PCA ANN retrieval with exact inner-product search."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch

from mmdd_dataset.wdc_runtime import iter_dataset_artifact
from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.data import EdgeExample, TargetExample, load_edge_examples, load_target_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import (
    RawEmbeddingANNIndices,
    StudentANNIndices,
    load_corpus_ids,
)


def _source_groups(dataset_root: Path) -> dict[str, str]:
    return {
        str(row["table_id"]): str(row["source_table_id"])
        for row in iter_dataset_artifact(dataset_root, "query_tables")
    }


def _sample_by_source(
    values: list[Any],
    source: Callable[[Any], str],
    *,
    count: int,
    seed: str,
) -> list[Any]:
    ordered = sorted(values, key=lambda value: (source(value), str(value)))
    random.Random(seed).shuffle(ordered)
    selected = []
    seen = set()
    for value in ordered:
        group = source(value)
        if group in seen:
            continue
        selected.append(value)
        seen.add(group)
        if len(selected) == count:
            return selected
    for value in ordered:
        if value not in selected:
            selected.append(value)
        if len(selected) == count:
            break
    return selected


def _positive_sets(
    targets: list[TargetExample],
    edges: list[EdgeExample],
) -> tuple[dict[tuple[str, str], set[str]], dict[tuple[str, str], set[str]]]:
    query_positives: dict[tuple[str, str], set[str]] = defaultdict(set)
    for example in targets:
        query_positives[(example.query_id, "table")].update(
            example.positive_target_ids
        )
        evidence = example.positive_evidence_by_target or {}
    edge_positives: dict[tuple[str, str], set[str]] = defaultdict(set)
    for example in edges:
        edge_positives[(example.query_id, example.destination_type or "")].update(
            example.positive_ids
        )
        if example.source_type == "table":
            query_positives[
                (example.query_id, example.destination_type or "")
            ].update(example.positive_ids)
    return query_positives, edge_positives


@torch.no_grad()
def _exact_hits(
    source_ids: list[str],
    destination_type: str,
    destination_ids: list[str],
    store: FeatureStore,
    *,
    device: torch.device,
    model: Any | None,
    k: int,
) -> list[list[tuple[str, float]]]:
    destination_embeddings = torch.stack(
        [store.embedding_features(value).embedding for value in destination_ids]
    ).to(device=device, dtype=torch.float32)
    if model is not None:
        destination_vectors = model.index_vector(
            destination_embeddings, destination_type
        )
    else:
        destination_vectors = destination_embeddings
    results = []
    for start in range(0, len(source_ids), 16):
        batch_ids = source_ids[start : start + 16]
        source_embeddings = torch.stack(
            [store.embedding_features(value).embedding for value in batch_ids]
        ).to(device=device, dtype=torch.float32)
        if model is not None:
            source_type = store.embedding_features(batch_ids[0]).object_type
            if any(
                store.embedding_features(value).object_type != source_type
                for value in batch_ids
            ):
                raise ValueError("Exact-search batches must share a source type")
            source_vectors = model.relation_query(
                source_embeddings, source_type, destination_type
            )
        else:
            source_vectors = source_embeddings
        scores = source_vectors @ destination_vectors.T
        values, indices = torch.topk(scores, min(k, scores.shape[1]), dim=1)
        for row_values, row_indices in zip(values.cpu(), indices.cpu()):
            hits = [
                (destination_ids[int(index)], float(value))
                for value, index in zip(row_values, row_indices)
            ]
            results.append(sorted(hits, key=lambda item: (-item[1], item[0])))
    del destination_embeddings, destination_vectors
    return results


def _relation_metrics(
    source_ids: list[str],
    destination_type: str,
    k: int,
    positives: dict[tuple[str, str], set[str]],
    ann_hits: list[list[tuple[str, float]]],
    exact_hits: list[list[tuple[str, float]]],
) -> dict[str, Any]:
    rows = []
    for source_id, ann, exact in zip(source_ids, ann_hits, exact_hits):
        ann_ids = [value for value, _score in ann]
        exact_ids = [value for value, _score in exact]
        exact_rank = {value: rank for rank, value in enumerate(exact_ids, 1)}
        ann_rank = {value: rank for rank, value in enumerate(ann_ids, 1)}
        intersection = set(ann_ids) & set(exact_ids)
        relevant = positives.get((source_id, destination_type), set())
        rows.append(
            {
                "source_id": source_id,
                "ann_exact_overlap": len(intersection) / max(len(exact_ids), 1),
                "mean_intersection_rank_error": (
                    sum(abs(ann_rank[value] - exact_rank[value]) for value in intersection)
                    / len(intersection)
                    if intersection
                    else float(k)
                ),
                "ann_positive_hit": int(bool(set(ann_ids) & relevant)),
                "exact_positive_hit": int(bool(set(exact_ids) & relevant)),
            }
        )
    return {
        "sources": len(rows),
        "k": k,
        "ann_exact_overlap": sum(row["ann_exact_overlap"] for row in rows)
        / len(rows),
        "mean_intersection_rank_error": sum(
            row["mean_intersection_rank_error"] for row in rows
        )
        / len(rows),
        "ann_positive_recall": sum(row["ann_positive_hit"] for row in rows)
        / len(rows),
        "exact_positive_recall": sum(row["exact_positive_hit"] for row in rows)
        / len(rows),
        "per_source": rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    store = FeatureStore.from_path(Path(args.features), cache_size=120_000)
    corpus = Path(args.corpus).resolve()
    corpus_sha256 = checkpoint_fingerprint(corpus)
    ids_by_type = load_corpus_ids(corpus, store)
    targets = load_target_examples(Path(args.dev_targets), split="dev")
    edges = load_edge_examples(Path(args.dev_edges), split="dev")
    source_groups = _source_groups(Path(args.dataset_root))
    selected_queries = []
    for kind in ("implicit", "explicit"):
        selected_queries.extend(
            _sample_by_source(
                [value for value in targets if value.query_kind == kind],
                lambda value: source_groups[value.query_id],
                count=64,
                seed=f"{args.seed}:query:{kind}",
            )
        )
    evidence_to_group: dict[str, str] = {}
    for example in targets:
        for evidence_ids in (example.positive_evidence_by_target or {}).values():
            for evidence_id in evidence_ids:
                evidence_to_group.setdefault(
                    evidence_id, source_groups[example.query_id]
                )
    evidence_edges = [
        value
        for value in edges
        if value.source_type in {"text", "image"}
        and value.destination_type == "table"
        and value.query_id in evidence_to_group
    ]
    selected_evidence = []
    for modality in ("text", "image"):
        selected_evidence.extend(
            _sample_by_source(
                [value for value in evidence_edges if value.source_type == modality],
                lambda value: evidence_to_group[value.query_id],
                count=64,
                seed=f"{args.seed}:evidence:{modality}",
            )
        )
    query_positives, edge_positives = _positive_sets(targets, edges)
    relation_requests = [
        (
            "q_to_table",
            [value.query_id for value in selected_queries],
            "table",
            100,
            query_positives,
        ),
        (
            "q_to_text",
            [value.query_id for value in selected_queries],
            "text",
            20,
            query_positives,
        ),
        (
            "q_to_image",
            [value.query_id for value in selected_queries],
            "image",
            20,
            query_positives,
        ),
        *(
            (
                f"{modality}_to_table",
                [
                    value.query_id
                    for value in selected_evidence
                    if value.source_type == modality
                ],
                "table",
                20,
                edge_positives,
            )
            for modality in ("text", "image")
        ),
    ]
    checkpoint_path = Path(args.student_checkpoint).resolve()
    checkpoint_sha256 = checkpoint_fingerprint(checkpoint_path)
    student = load_student(checkpoint_path, device)
    systems = {
        "raw": (
            None,
            RawEmbeddingANNIndices(
                store,
                Path(args.raw_index),
                corpus_sha256=corpus_sha256,
            ),
        ),
        "pca": (
            student,
            StudentANNIndices(
                student,
                store,
                Path(args.student_index),
                device=device,
                checkpoint_sha256=checkpoint_sha256,
                corpus_sha256=corpus_sha256,
                score_space="raw_logit",
            ),
        ),
    }
    output: dict[str, Any] = {
        "format_version": 1,
        "seed": args.seed,
        "samples": {
            "queries": len(selected_queries),
            "query_kind": {
                kind: sum(value.query_kind == kind for value in selected_queries)
                for kind in ("implicit", "explicit")
            },
            "evidence": len(selected_evidence),
            "evidence_modality": {
                modality: sum(
                    value.source_type == modality for value in selected_evidence
                )
                for modality in ("text", "image")
            },
        },
        "systems": {},
    }
    for system_name, (model, indices) in systems.items():
        system_results = {}
        for name, source_ids, destination_type, k, positives in relation_requests:
            if not source_ids:
                continue
            ann = indices.search_many(source_ids, destination_type, k)
            exact = _exact_hits(
                source_ids,
                destination_type,
                ids_by_type[destination_type],
                store,
                device=device,
                model=model,
                k=k,
            )
            system_results[name] = _relation_metrics(
                source_ids,
                destination_type,
                k,
                positives,
                ann,
                exact,
            )
        output["systems"][system_name] = system_results
    write_json(Path(args.output), output)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "samples": output["samples"],
                "summary": {
                    system: {
                        relation: {
                            key: metrics[key]
                            for key in (
                                "ann_exact_overlap",
                                "ann_positive_recall",
                                "exact_positive_recall",
                            )
                        }
                        for relation, metrics in relations.items()
                    }
                    for system, relations in output["systems"].items()
                },
            },
            indent=2,
        )
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--dev-targets", required=True)
    parser.add_argument("--dev-edges", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--student-index", required=True)
    parser.add_argument("--raw-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
