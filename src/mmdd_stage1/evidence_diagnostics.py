"""Diagnostics for Stage-1 evidence annotation and raw retrieval reachability."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch
from mmdd_dataset.wdc_runtime import iter_dataset_artifact

from .features import FeatureStore


def _jsonl_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            yield record


def _ordered_add(values: dict[Any, list[str]], key: Any, value: str) -> None:
    bucket = values[key]
    if value not in bucket:
        bucket.append(value)


def _construction_source(
    pair: tuple[str, str],
    exact_evidence: dict[tuple[str, str], list[str]],
    target_recovery_evidence: dict[str, list[str]],
    source_evidence: dict[str, list[str]],
    target_sources: dict[str, str],
) -> tuple[str, list[str]]:
    if exact_evidence.get(pair):
        return "exact_recovery", exact_evidence[pair]
    target_id = pair[1]
    if target_recovery_evidence.get(target_id):
        return "target_recovery_fallback", target_recovery_evidence[target_id]
    source_id = target_sources.get(target_id, "")
    if source_evidence.get(source_id):
        return "source_asset_heuristic", source_evidence[source_id]
    return "none", []


def analyze_evidence_annotations(
    *,
    dataset_name: str,
    dataset_root: Path,
    target_lists_path: Path,
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reproduce construction.py evidence selection and audit its provenance."""

    queries = {
        str(record["table_id"]): {
            "split": str(record.get("split", "train")),
            "source_table_id": str(record.get("source_table_id", "")),
        }
        for record in iter_dataset_artifact(dataset_root, "query_tables")
    }
    target_sources = {
        str(record["table_id"]): str(record.get("source_table_id", ""))
        for record in iter_dataset_artifact(dataset_root, "data_lake_tables")
    }
    asset_types: dict[str, str] = {}
    source_evidence: dict[str, list[str]] = defaultdict(list)
    for asset in iter_dataset_artifact(dataset_root, "bridge_assets"):
        asset_id = str(asset["asset_id"])
        asset_types[asset_id] = str(asset["asset_type"])
        source_id = asset.get("source_table_id")
        if source_id is not None:
            source_evidence[str(source_id)].append(asset_id)

    qrel_pairs: set[tuple[str, str]] = set()
    qrel_metadata: dict[tuple[str, str], dict[str, str]] = {}
    qrel_split_disagreements = 0
    for qrel in iter_dataset_artifact(dataset_root, "qrels"):
        if float(qrel.get("rel", 1)) <= 0:
            continue
        query_id = str(qrel["query_table_id"])
        target_id = str(qrel["target_table_id"])
        query = queries.get(query_id)
        if query is None or target_id not in target_sources:
            continue
        if str(qrel.get("split", query["split"])) != query["split"]:
            qrel_split_disagreements += 1
        if query["split"] == split:
            pair = (query_id, target_id)
            qrel_pairs.add(pair)
            qrel_metadata[pair] = {
                "reason": str(qrel.get("reason", "")),
                "join_role": str(qrel.get("join_attribute", {}).get("role", "")),
            }

    exact_evidence: dict[tuple[str, str], list[str]] = defaultdict(list)
    target_recovery_evidence: dict[str, list[str]] = defaultdict(list)
    recovery_records = 0
    accepted_recovery_records = 0
    collapsed_row_recovery_records = 0
    split_recovery_records = 0
    invalid_recovery_references = Counter()
    recovery_pairs_outside_qrels = set()
    recovery_split_disagreements = 0
    recovered_attributes_not_hidden = 0
    supported_auto_checks = 0
    for recovery in iter_dataset_artifact(dataset_root, "evidence_recoveries"):
        recovery_records += 1
        query_id = str(recovery.get("query_table_id", ""))
        target_id = str(recovery.get("target_table_id", ""))
        evidence_id = str(recovery.get("evidence", {}).get("asset_id", ""))
        invalid = False
        if query_id not in queries:
            invalid_recovery_references["missing_query"] += 1
            invalid = True
        if target_id not in target_sources:
            invalid_recovery_references["missing_target"] += 1
            invalid = True
        if evidence_id not in asset_types:
            invalid_recovery_references["missing_evidence"] += 1
            invalid = True
        if invalid:
            continue
        accepted_recovery_records += 1
        pair = (query_id, target_id)
        query_split = queries[query_id]["split"]
        if query_split == split:
            split_recovery_records += 1
        if evidence_id in exact_evidence[pair]:
            collapsed_row_recovery_records += 1
        _ordered_add(exact_evidence, pair, evidence_id)
        _ordered_add(target_recovery_evidence, target_id, evidence_id)
        if str(recovery.get("split", query_split)) != query_split:
            recovery_split_disagreements += 1
        if pair not in qrel_pairs and query_split == split:
            recovery_pairs_outside_qrels.add(pair)
        if recovery.get("recovered_attribute", {}).get("hidden_in_query") is not True:
            recovered_attributes_not_hidden += 1
        if int(recovery.get("auto_check", {}).get("supported_attributes", 0)) > 0:
            supported_auto_checks += 1

    target_lists: dict[str, dict[str, Any]] = {}
    evidence_loss_rows = Counter()
    for record in _jsonl_records(target_lists_path):
        if str(record.get("split", "train")) != split:
            continue
        query_id = str(record["query_id"])
        if query_id in target_lists:
            raise ValueError(f"{target_lists_path}: duplicate query_id {query_id!r}")
        target_lists[query_id] = record
        candidates = {
            str(candidate["target_id"]): [
                str(value) for value in candidate.get("evidence_ids", [])
            ]
            for candidate in record["candidates"]
        }
        positive_id = str(record["evidence_positive_target_id"])
        evidence_mask = [bool(values) for values in candidates.values()]
        has_positive = bool(candidates.get(positive_id))
        evidence_loss_rows["rows"] += 1
        evidence_loss_rows["positive_has_evidence"] += has_positive
        evidence_loss_rows["at_least_two_candidates_have_evidence"] += sum(
            evidence_mask
        ) >= 2
        evidence_loss_rows["usable"] += has_positive and sum(evidence_mask) >= 2

    cases = []
    missing_target_list_queries = set()
    missing_positive_candidates = 0
    evidence_mismatches = 0
    source_counts = Counter()
    modality_query_counts = Counter()
    construction_positive_source_counts = Counter()
    qrel_reason_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    exact_recovery_ids_expected = 0
    exact_recovery_ids_present = 0
    for query_id, target_id in sorted(qrel_pairs):
        pair = (query_id, target_id)
        source, expected_evidence = _construction_source(
            pair,
            exact_evidence,
            target_recovery_evidence,
            source_evidence,
            target_sources,
        )
        source_counts[source] += 1
        qrel_reason = qrel_metadata[pair]["reason"] or "unspecified"
        qrel_reason_by_source[source][qrel_reason] += 1
        exact_recovery_ids_expected += len(exact_evidence.get(pair, ()))

        target_list = target_lists.get(query_id)
        construction_evidence: list[str] = []
        construction_positive_target_id = ""
        construction_positive_evidence: list[str] = []
        if target_list is None:
            missing_target_list_queries.add(query_id)
        else:
            candidates = {
                str(candidate["target_id"]): [
                    str(value) for value in candidate.get("evidence_ids", [])
                ]
                for candidate in target_list["candidates"]
            }
            if target_id not in candidates:
                missing_positive_candidates += 1
            else:
                construction_evidence = candidates[target_id]
                exact_recovery_ids_present += len(
                    set(exact_evidence.get(pair, ())) & set(construction_evidence)
                )
                if construction_evidence != expected_evidence:
                    evidence_mismatches += 1
            construction_positive_target_id = str(
                target_list["evidence_positive_target_id"]
            )
            construction_positive_evidence = candidates.get(
                construction_positive_target_id, []
            )
            positive_source, _ = _construction_source(
                (query_id, construction_positive_target_id),
                exact_evidence,
                target_recovery_evidence,
                source_evidence,
                target_sources,
            )
            construction_positive_source_counts[positive_source] += 1

        modalities = sorted(
            {asset_types[evidence_id] for evidence_id in expected_evidence}
        )
        for modality in modalities:
            modality_query_counts[modality] += 1
        cases.append(
            {
                "dataset": dataset_name,
                "query_id": query_id,
                "target_id": target_id,
                "qrel_reason": qrel_metadata[pair]["reason"],
                "qrel_join_role": qrel_metadata[pair]["join_role"],
                "annotation_source": source,
                "exact_recovery_evidence_ids": list(exact_evidence.get(pair, ())),
                "expected_construction_evidence_ids": list(expected_evidence),
                "construction_evidence_ids": construction_evidence,
                "construction_positive_target_id": construction_positive_target_id,
                "construction_positive_evidence_ids": construction_positive_evidence,
                "modalities": modalities,
            }
        )

    dev_recovery_pairs = {
        pair for pair in exact_evidence if queries[pair[0]]["split"] == split
    }
    summary = {
        "dataset": dataset_name,
        "dataset_root": str(dataset_root),
        "target_lists": str(target_lists_path),
        "split": split,
        "queries": len({query_id for query_id, _target_id in qrel_pairs}),
        "positive_qrel_pairs": len(qrel_pairs),
        "annotation_source_counts": dict(sorted(source_counts.items())),
        "qrel_reason_by_annotation_source": {
            source: dict(sorted(counts.items()))
            for source, counts in sorted(qrel_reason_by_source.items())
        },
        "construction_positive_source_counts": dict(
            sorted(construction_positive_source_counts.items())
        ),
        "queries_with_positive_evidence_by_modality": dict(
            sorted(modality_query_counts.items())
        ),
        "positive_pairs_with_any_evidence": sum(
            value for key, value in source_counts.items() if key != "none"
        ),
        "positive_pairs_with_exact_recovery": source_counts["exact_recovery"],
        "recovery_pairs_for_split": len(dev_recovery_pairs),
        "recovery_pairs_outside_positive_qrels": len(recovery_pairs_outside_qrels),
        "recovery_records_all_splits": recovery_records,
        "recovery_records_for_split": split_recovery_records,
        "accepted_recovery_records_all_splits": accepted_recovery_records,
        "additional_row_records_collapsed_by_query_target_evidence_all_splits": (
            collapsed_row_recovery_records
        ),
        "invalid_recovery_references": dict(sorted(invalid_recovery_references.items())),
        "recovery_split_disagreements": recovery_split_disagreements,
        "qrel_split_disagreements": qrel_split_disagreements,
        "recoveries_with_hidden_in_query_not_true": recovered_attributes_not_hidden,
        "recoveries_with_supported_auto_check": supported_auto_checks,
        "assets_with_source_table_id": sum(len(values) for values in source_evidence.values()),
        "evidence_loss_gating": {
            **dict(evidence_loss_rows),
            "usable_rate": (
                evidence_loss_rows["usable"] / evidence_loss_rows["rows"]
                if evidence_loss_rows["rows"]
                else 0.0
            ),
        },
        "construction_audit": {
            "target_list_queries": len(target_lists),
            "missing_target_list_queries": len(missing_target_list_queries),
            "missing_positive_candidates": missing_positive_candidates,
            "positive_candidate_evidence_mismatches": evidence_mismatches,
            "exact_recovery_ids_expected": exact_recovery_ids_expected,
            "exact_recovery_ids_present": exact_recovery_ids_present,
        },
    }
    return summary, cases


def _load_hnsw_index(
    index_dir: Path,
    object_type: str,
    *,
    ef_search: int,
) -> tuple[Any, list[str], int]:
    import hnswlib

    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    record = manifest["types"][object_type]
    dim = int(manifest["embedding_dim"])
    index = hnswlib.Index(space="ip", dim=dim)
    index.load_index(
        str(index_dir / record["index_path"]),
        max_elements=int(record["objects"]),
    )
    base_ef_search = int(manifest["ef_search"])
    index.set_ef(max(base_ef_search, ef_search))
    object_ids = json.loads(
        (index_dir / record["ids_path"]).read_text(encoding="utf-8")
    )
    return index, [str(value) for value in object_ids], base_ef_search


def _search_vectors(
    index: Any,
    vectors: torch.Tensor,
    *,
    k: int,
    batch_size: int,
    num_threads: int,
) -> Iterable[np.ndarray]:
    for start in range(0, len(vectors), batch_size):
        batch = vectors[start : start + batch_size].numpy().astype("float32")
        search_options = {"num_threads": num_threads} if num_threads > 0 else {}
        labels, _distances = index.knn_query(batch, k=k, **search_options)
        yield from labels


def _ranks_for_positive_ids(
    index: Any,
    object_ids: list[str],
    vectors: torch.Tensor,
    positive_ids: list[set[str]],
    *,
    rank_limit: int,
    batch_size: int,
    num_threads: int,
) -> list[dict[str, int]]:
    id_by_label = object_ids
    k = min(rank_limit, len(id_by_label))
    results = []
    for labels, expected in zip(
        _search_vectors(
            index,
            vectors,
            k=k,
            batch_size=batch_size,
            num_threads=num_threads,
        ),
        positive_ids,
    ):
        ranks = {}
        for rank, label in enumerate(labels, 1):
            object_id = id_by_label[int(label)]
            if object_id in expected:
                ranks[object_id] = rank
                if len(ranks) == len(expected):
                    break
        results.append(ranks)
    return results


def summarize_ranks(
    ranks: list[int | None],
    *,
    rank_limit: int,
    cutoffs: tuple[int, ...] = (1, 10, 50, 100, 1000),
) -> dict[str, Any]:
    """Summarize right-censored positive ranks without hiding the censoring."""

    denominator = len(ranks)
    observed = sorted(rank for rank in ranks if rank is not None)
    censored = denominator - len(observed)
    bounded = [rank if rank is not None else rank_limit + 1 for rank in ranks]
    bounded_median = median(bounded) if bounded else None
    median_is_observed = len(observed) >= denominator // 2 + 1
    return {
        "queries": denominator,
        "observed_within_rank_limit": len(observed),
        "censored_above_rank_limit": censored,
        "rank_limit": rank_limit,
        "median_best_positive_rank": bounded_median if median_is_observed else None,
        "median_best_positive_rank_label": (
            str(bounded_median) if median_is_observed else f">{rank_limit}"
        )
        if denominator
        else None,
        "observed_rank_quartiles": {
            "p25": float(np.percentile(observed, 25)) if observed else None,
            "p50": float(np.percentile(observed, 50)) if observed else None,
            "p75": float(np.percentile(observed, 75)) if observed else None,
        },
        "recall_at_rank": {
            str(cutoff): (
                sum(rank is not None and rank <= cutoff for rank in ranks) / denominator
                if denominator
                else 0.0
            )
            for cutoff in cutoffs
            if cutoff <= rank_limit
        },
    }


def rank_raw_evidence_paths(
    *,
    cases: list[dict[str, Any]],
    feature_store: FeatureStore,
    raw_index_dir: Path,
    evidence_rank_limit: int,
    target_rank_limit: int,
    batch_size: int = 32,
    num_threads: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Rank exact recovery evidence for Q->E, Q+T->E, and E->T."""

    ranked_cases = [case for case in cases if case["exact_recovery_evidence_ids"]]
    required_ids = []
    for case in ranked_cases:
        required_ids.extend([case["query_id"], case["target_id"]])
        required_ids.extend(case["exact_recovery_evidence_ids"])
    feature_store.preload_embeddings(required_ids)

    details = [
        {
            **case,
            "query_to_evidence_ranks": {},
            "online_query_to_evidence_ranks_at_50": {},
            "query_target_to_evidence_ranks": {},
            "evidence_to_target_ranks": {},
        }
        for case in ranked_cases
    ]
    evidence_type_by_id = {
        evidence_id: feature_store.embedding_features(evidence_id).object_type
        for case in ranked_cases
        for evidence_id in case["exact_recovery_evidence_ids"]
    }
    case_index = {
        (case["dataset"], case["query_id"], case["target_id"]): index
        for index, case in enumerate(details)
    }

    for modality in ("text", "image"):
        modality_cases = [
            case
            for case in ranked_cases
            if any(
                evidence_type_by_id[evidence_id] == modality
                for evidence_id in case["exact_recovery_evidence_ids"]
            )
        ]
        if not modality_cases:
            continue
        index, index_ids, base_ef_search = _load_hnsw_index(
            raw_index_dir,
            modality,
            ef_search=evidence_rank_limit,
        )
        query_vectors = torch.stack(
            [
                feature_store.embedding_features(case["query_id"]).embedding.float()
                for case in modality_cases
            ]
        )
        target_vectors = torch.stack(
            [
                feature_store.embedding_features(case["target_id"]).embedding.float()
                for case in modality_cases
            ]
        )
        conditioned_vectors = torch.nn.functional.normalize(
            query_vectors + target_vectors,
            dim=1,
        )
        positives = [
            {
                evidence_id
                for evidence_id in case["exact_recovery_evidence_ids"]
                if evidence_type_by_id[evidence_id] == modality
            }
            for case in modality_cases
        ]
        index.set_ef(base_ef_search)
        online_query_ranks = _ranks_for_positive_ids(
            index,
            index_ids,
            query_vectors,
            positives,
            rank_limit=50,
            batch_size=batch_size,
            num_threads=num_threads,
        )
        index.set_ef(max(base_ef_search, evidence_rank_limit))
        query_ranks = _ranks_for_positive_ids(
            index,
            index_ids,
            query_vectors,
            positives,
            rank_limit=evidence_rank_limit,
            batch_size=batch_size,
            num_threads=num_threads,
        )
        conditioned_ranks = _ranks_for_positive_ids(
            index,
            index_ids,
            conditioned_vectors,
            positives,
            rank_limit=evidence_rank_limit,
            batch_size=batch_size,
            num_threads=num_threads,
        )
        for case, online_ranks, raw_ranks, proxy_ranks in zip(
            modality_cases, online_query_ranks, query_ranks, conditioned_ranks
        ):
            detail = details[case_index[(case["dataset"], case["query_id"], case["target_id"])]]
            detail["query_to_evidence_ranks"].update(raw_ranks)
            detail["online_query_to_evidence_ranks_at_50"].update(online_ranks)
            detail["query_target_to_evidence_ranks"].update(proxy_ranks)
        del index

    evidence_targets: dict[str, set[str]] = defaultdict(set)
    for case in ranked_cases:
        for evidence_id in case["exact_recovery_evidence_ids"]:
            evidence_targets[evidence_id].add(case["target_id"])
    evidence_ids = sorted(evidence_targets)
    table_index, table_ids, _base_table_ef_search = _load_hnsw_index(
        raw_index_dir,
        "table",
        ef_search=target_rank_limit,
    )
    evidence_vectors = torch.stack(
        [feature_store.embedding_features(evidence_id).embedding.float() for evidence_id in evidence_ids]
    )
    target_rank_maps = _ranks_for_positive_ids(
        table_index,
        table_ids,
        evidence_vectors,
        [evidence_targets[evidence_id] for evidence_id in evidence_ids],
        rank_limit=target_rank_limit,
        batch_size=batch_size,
        num_threads=num_threads,
    )
    target_ranks = {
        evidence_id: ranks
        for evidence_id, ranks in zip(evidence_ids, target_rank_maps)
    }
    for detail in details:
        target_id = detail["target_id"]
        detail["evidence_to_target_ranks"] = {
            evidence_id: target_ranks[evidence_id].get(target_id)
            for evidence_id in detail["exact_recovery_evidence_ids"]
        }

    for detail in details:
        raw_values = list(detail["query_to_evidence_ranks"].values())
        online_values = list(detail["online_query_to_evidence_ranks_at_50"].values())
        proxy_values = list(detail["query_target_to_evidence_ranks"].values())
        target_values = [
            value
            for value in detail["evidence_to_target_ranks"].values()
            if value is not None
        ]
        detail["best_query_to_evidence_rank"] = min(raw_values, default=None)
        detail["best_online_query_to_evidence_rank_at_50"] = min(
            online_values, default=None
        )
        detail["best_query_target_to_evidence_rank"] = min(proxy_values, default=None)
        detail["best_evidence_to_target_rank"] = min(target_values, default=None)
        detail["raw_joint_reachable_at_50_50"] = any(
            detail["online_query_to_evidence_ranks_at_50"].get(evidence_id) is not None
            and detail["evidence_to_target_ranks"].get(evidence_id) is not None
            and detail["evidence_to_target_ranks"][evidence_id] <= 50
            for evidence_id in detail["exact_recovery_evidence_ids"]
        )
        detail["best_query_to_evidence_rank_by_modality"] = {}
        detail["best_query_target_to_evidence_rank_by_modality"] = {}
        detail["best_evidence_to_target_rank_by_modality"] = {}
        for modality in ("text", "image"):
            evidence_ids_for_modality = [
                evidence_id
                for evidence_id in detail["exact_recovery_evidence_ids"]
                if evidence_type_by_id[evidence_id] == modality
            ]
            if not evidence_ids_for_modality:
                continue
            detail["best_query_to_evidence_rank_by_modality"][modality] = min(
                (
                    detail["query_to_evidence_ranks"].get(evidence_id)
                    for evidence_id in evidence_ids_for_modality
                    if detail["query_to_evidence_ranks"].get(evidence_id) is not None
                ),
                default=None,
            )
            detail["best_query_target_to_evidence_rank_by_modality"][modality] = min(
                (
                    detail["query_target_to_evidence_ranks"].get(evidence_id)
                    for evidence_id in evidence_ids_for_modality
                    if detail["query_target_to_evidence_ranks"].get(evidence_id)
                    is not None
                ),
                default=None,
            )
            detail["best_evidence_to_target_rank_by_modality"][modality] = min(
                (
                    detail["evidence_to_target_ranks"].get(evidence_id)
                    for evidence_id in evidence_ids_for_modality
                    if detail["evidence_to_target_ranks"].get(evidence_id) is not None
                ),
                default=None,
            )
        detail["proxy_joint_reachable_at_50_50"] = any(
            detail["query_target_to_evidence_ranks"].get(evidence_id) is not None
            and detail["query_target_to_evidence_ranks"][evidence_id] <= 50
            and detail["evidence_to_target_ranks"].get(evidence_id) is not None
            and detail["evidence_to_target_ranks"][evidence_id] <= 50
            for evidence_id in detail["exact_recovery_evidence_ids"]
        )

    def grouped_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "query_to_evidence": summarize_ranks(
                [row["best_query_to_evidence_rank"] for row in rows],
                rank_limit=evidence_rank_limit,
            ),
            "online_query_to_evidence_recall_at_50": (
                sum(
                    row["best_online_query_to_evidence_rank_at_50"] is not None
                    for row in rows
                )
                / len(rows)
                if rows
                else 0.0
            ),
            "query_target_proxy_to_evidence": summarize_ranks(
                [row["best_query_target_to_evidence_rank"] for row in rows],
                rank_limit=evidence_rank_limit,
            ),
            "evidence_to_target": summarize_ranks(
                [row["best_evidence_to_target_rank"] for row in rows],
                rank_limit=target_rank_limit,
                cutoffs=(1, 10, 50, 100),
            ),
            "joint_reachability_at_50_50": {
                "raw_query": (
                    sum(row["raw_joint_reachable_at_50_50"] for row in rows) / len(rows)
                    if rows
                    else 0.0
                ),
                "query_target_proxy": (
                    sum(row["proxy_joint_reachable_at_50_50"] for row in rows) / len(rows)
                    if rows
                    else 0.0
                ),
                "queries": len(rows),
            },
        }

    by_dataset = {
        dataset: grouped_summary(
            [row for row in details if row["dataset"] == dataset]
        )
        for dataset in sorted({row["dataset"] for row in details})
    }
    by_modality = {}
    for modality in ("text", "image"):
        modality_rows = [
            row
            for row in details
            if modality in row["best_query_to_evidence_rank_by_modality"]
        ]
        by_modality[modality] = {
            "query_to_evidence": summarize_ranks(
                [
                    row["best_query_to_evidence_rank_by_modality"][modality]
                    for row in modality_rows
                ],
                rank_limit=evidence_rank_limit,
            ),
            "query_target_proxy_to_evidence": summarize_ranks(
                [
                    row["best_query_target_to_evidence_rank_by_modality"][modality]
                    for row in modality_rows
                ],
                rank_limit=evidence_rank_limit,
            ),
            "evidence_to_target": summarize_ranks(
                [
                    row["best_evidence_to_target_rank_by_modality"][modality]
                    for row in modality_rows
                ],
                rank_limit=target_rank_limit,
                cutoffs=(1, 10, 50, 100),
            ),
        }
    summary = {
        "definition": {
            "positive_evidence": "valid exact recovery evidence for the dev (query, target) qrel pair",
            "query_to_evidence_rank": "rank within the evidence modality's raw HNSW index",
            "query_target_proxy": "L2-normalized sum of frozen query and positive-target embeddings; this is a retrieval-formulation proxy, not the Teacher model",
            "evidence_to_target_rank": "rank in the raw table HNSW index",
            "joint_reachability": "at least one exact recovery has Q->E rank <= 50 and E->T rank <= 50 before path scoring/fusion",
        },
        "overall": grouped_summary(details),
        "by_dataset": by_dataset,
        "by_modality": by_modality,
    }
    return summary, details
