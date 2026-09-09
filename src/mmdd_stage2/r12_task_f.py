"""Frozen-sample execution and metrics for the R12 Task F mechanism test."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mmdd_dataset.utils import clean_text
from mmdd_dataset.wdc_runtime import iter_dataset_artifact
from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.significance import paired_bootstrap_delta
from torch.nn import functional as F

from .data import (
    Stage2ObjectIndex,
    column_name,
    column_values,
    load_stage2_index,
    local_column_index,
    row_values,
)
from .oracle import select_oracle_evidence
from .pipeline import (
    ColumnSelection,
    EvidenceVerification,
    LocalizedEvidence,
    RowPrediction,
    Stage2Backend,
    Stage2Verifier,
)
from .verifier import EvidenceBundle


@dataclass(frozen=True)
class TaskFInputs:
    sample: tuple[dict[str, Any], ...]
    supervision: dict[str, dict[str, Any]]
    queues: dict[str, dict[str, list[dict[str, Any]]]]
    queue_details: dict[str, dict[str, list[dict[str, Any]]]]
    frozen_fusion_id: str
    queues_identical: bool
    objects: Stage2ObjectIndex
    qrels: dict[tuple[str, str], dict[str, Any]]
    recovery_values: dict[tuple[str, str, int], dict[str, Any]]
    manifest: dict[str, Any]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _selected_supervision(path: Path, query_ids: set[str]) -> dict[str, dict[str, Any]]:
    records = {
        str(record["query_id"]): record
        for record in _load_jsonl(path)
        if str(record["query_id"]) in query_ids
    }
    missing = query_ids - records.keys()
    if missing:
        raise KeyError(f"Frozen Task F supervision misses {min(missing)}")
    return records


def _selected_pool_records(path: Path, query_ids: set[str]) -> dict[str, dict[str, Any]]:
    records = {
        str(record["query_id"]): record
        for record in _load_jsonl(path)
        if str(record["query_id"]) in query_ids
    }
    missing = query_ids - records.keys()
    if missing:
        raise KeyError(f"Frozen Task F path pool misses {min(missing)}")
    return records


def _queue_retrieval(
    queue: Sequence[dict[str, Any]], pool: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retrieval = []
    details = []
    for row in queue:
        target_id = str(row["target_id"])
        all_paths = pool["paths_by_target"].get(target_id)
        if not all_paths:
            raise KeyError(f"{pool['query_id']}->{target_id}: no frozen path detail")
        evidence_paths = {
            str(path["evidence_id"]): path
            for path in all_paths
            if path.get("kind") == "evidence"
        }
        selected_ids = [str(value) for value in row["selected_evidence_ids"]]
        missing_evidence = set(selected_ids) - evidence_paths.keys()
        if missing_evidence:
            raise KeyError(
                f"{pool['query_id']}->{target_id}: selected evidence has no path detail"
            )
        paths = [dict(evidence_paths[value]) for value in selected_ids]
        direct = next(
            (dict(path) for path in all_paths if path.get("kind") == "direct"), None
        )
        if direct is not None:
            paths.append(direct)
        if not paths:
            raise ValueError(f"{pool['query_id']}->{target_id}: empty Task F path set")
        final_rank = int(row["final_rank"])
        item = {
            "target_id": target_id,
            "score": -float(final_rank),
            "stage2_table_score": -float(final_rank),
            "paths": paths,
        }
        if selected_ids:
            item["evidence_score"] = -float(final_rank)
        retrieval.append(item)
        details.append(dict(row))
    return retrieval, details


def load_task_f_inputs(
    dataset_root: Path,
    sample_path: Path,
    supervision_path: Path,
    path_pool: Path,
    admission_metrics: Path,
) -> TaskFInputs:
    sample_payload = _load_json(sample_path)
    sample = tuple(sample_payload["selected"])
    query_ids = {str(record["query_id"]) for record in sample}
    if len(sample) != 128 or len(query_ids) != 128:
        raise ValueError("R12 Task F requires the frozen 128 unique queries")
    if Counter(record["kind"] for record in sample) != {"implicit": 96, "explicit": 32}:
        raise ValueError("R12 Task F frozen stratum counts changed")
    supervision = _selected_supervision(supervision_path, query_ids)
    for record in sample:
        saved = supervision[str(record["query_id"])]
        if str(saved["query_kind"]) != str(record["kind"]):
            raise ValueError(f"Frozen Task F identity changed for {record['query_id']}")

    admission = _load_json(admission_metrics)
    frozen_fusion_id = str(admission["selection"]["selected"])
    if frozen_fusion_id not in admission["results"]:
        raise ValueError("Frozen Task F fusion is absent from Task E")
    admission_by_system = {
        system: {
            str(record["query_id"]): record
            for record in admission["results"][rule]["per_query"]
        }
        for system, rule in (
            ("f1_union_direct", "f1_union_direct"),
            ("frozen_fusion", frozen_fusion_id),
        )
    }
    pool_records = _selected_pool_records(path_pool, query_ids)
    queues: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    queue_details: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for system, by_query in admission_by_system.items():
        for query_id in query_ids:
            if query_id not in by_query:
                raise KeyError(f"Task E {system} misses {query_id}")
            queue, details = _queue_retrieval(
                by_query[query_id]["final_top10"], pool_records[query_id]
            )
            queues[system][query_id] = queue
            queue_details[system][query_id] = details
    queues_identical = queues["f1_union_direct"] == queues["frozen_fusion"]

    target_ids = {
        str(row["target_id"])
        for system in queues.values()
        for rows in system.values()
        for row in rows
    }
    evidence_ids = {
        str(path["evidence_id"])
        for system in queues.values()
        for rows in system.values()
        for row in rows
        for path in row["paths"]
        if path.get("kind") == "evidence"
    }
    for record in supervision.values():
        for ids in record.get("positive_evidence_by_target", {}).values():
            evidence_ids.update(str(value) for value in ids)
    objects = load_stage2_index(
        dataset_root,
        query_ids=query_ids,
        target_ids=target_ids,
        evidence_ids=evidence_ids,
    )
    for record in sample:
        query_id = str(record["query_id"])
        if str(objects.queries[query_id].get("source_table_id")) != str(
            record["source_table_id"]
        ):
            raise ValueError(f"Frozen Task F source group changed for {query_id}")

    positive_pairs = {
        (query_id, str(target_id))
        for query_id, record in supervision.items()
        for target_id in record["positive_target_ids"]
        if str(target_id) in target_ids
    }
    qrels = {
        (str(record["query_table_id"]), str(record["target_table_id"])): record
        for record in iter_dataset_artifact(dataset_root, "qrels")
        if (str(record["query_table_id"]), str(record["target_table_id"]))
        in positive_pairs
    }
    recoverable_pairs = {
        pair
        for pair, record in qrels.items()
        if record.get("reason") == "model_recoverable_join_column"
    }
    recovery_values: dict[tuple[str, str, int], dict[str, Any]] = {}
    for record in iter_dataset_artifact(dataset_root, "evidence_recoveries"):
        pair = (str(record["query_table_id"]), str(record["target_table_id"]))
        if pair not in recoverable_pairs:
            continue
        key = (*pair, int(record["query_row_id"]))
        value = clean_text(record["recovered_attribute"].get("value"))
        evidence_id = str(record["evidence"]["asset_id"])
        existing = recovery_values.setdefault(
            key, {"value": value, "evidence_ids": []}
        )
        if existing["value"] != value:
            raise ValueError(f"Conflicting Task F row truth for {key}")
        if evidence_id not in existing["evidence_ids"]:
            existing["evidence_ids"].append(evidence_id)

    path_metadata = _load_json(path_pool.with_suffix(".jsonl.metadata.json"))
    if checkpoint_fingerprint(path_pool) != path_metadata["output_sha256"]:
        raise ValueError("Frozen Task F path-pool fingerprint changed")
    manifest = {
        "format_version": 1,
        "sample": str(sample_path.resolve()),
        "sample_sha256": checkpoint_fingerprint(sample_path),
        "supervision": str(supervision_path.resolve()),
        "supervision_sha256": checkpoint_fingerprint(supervision_path),
        "path_pool": str(path_pool.resolve()),
        "path_pool_sha256": checkpoint_fingerprint(path_pool),
        "admission_metrics": str(admission_metrics.resolve()),
        "admission_metrics_sha256": checkpoint_fingerprint(admission_metrics),
        "frozen_fusion_id": frozen_fusion_id,
        "queues_identical": queues_identical,
        "queries": len(sample),
        "implicit": sum(record["kind"] == "implicit" for record in sample),
        "explicit": sum(record["kind"] == "explicit" for record in sample),
        "objects": {
            "queries": len(objects.queries),
            "targets": len(objects.targets),
            "evidence": len(objects.evidence),
        },
        "recoverable_pairs_in_top10_union": len(recoverable_pairs),
        "recovery_truth_rows": len(recovery_values),
        "visible_input_policy": (
            "Qwen serializers receive columns/rows and evidence content only; hidden_attributes, "
            "provenance, qrels, and recovery truth remain evaluator-side"
        ),
    }
    return TaskFInputs(
        sample=sample,
        supervision=supervision,
        queues=dict(queues),
        queue_details=dict(queue_details),
        frozen_fusion_id=frozen_fusion_id,
        queues_identical=queues_identical,
        objects=objects,
        qrels=qrels,
        recovery_values=recovery_values,
        manifest=manifest,
    )


def _support_rows(record: dict[str, Any], target_id: str) -> dict[str, set[int]]:
    return {
        str(evidence_id): {int(row_id) for row_id in row_ids}
        for evidence_id, row_ids in record.get(
            "positive_evidence_rows_by_target", {}
        ).get(target_id, {}).items()
    }


def _wrong_attribute_evidence(
    record: dict[str, Any], target_id: str, *, count: int, seed: int
) -> tuple[str, ...]:
    own = set(record.get("positive_evidence_by_target", {}).get(target_id, []))
    candidates = {
        str(evidence_id)
        for other_target, ids in record.get("positive_evidence_by_target", {}).items()
        if str(other_target) != target_id
        for evidence_id in ids
        if str(evidence_id) not in own
    }
    ordered = sorted(
        candidates,
        key=lambda value: hashlib.sha256(
            f"{seed}\0{record['query_id']}\0{target_id}\0{value}".encode()
        ).digest(),
    )
    return tuple(ordered[:count]) if len(ordered) >= count else ()


def _no_evidence_recovery(
    verifier: Stage2Verifier,
    query: dict[str, Any],
    target: dict[str, Any],
    selection: ColumnSelection,
) -> EvidenceVerification:
    predictions = []
    empty = LocalizedEvidence(
        evidence_id="__empty_evidence__",
        evidence_type="text",
        text="",
        text_span_relevance=0.0,
    )
    for row in query["rows"]:
        predictions.append(
            RowPrediction(
                row_id=int(row["row_id"]),
                value=verifier.backend.generate_value(
                    row_values(query, row),
                    attribute_name=selection.column_name,
                    evidence=empty,
                ),
                evidence=None,
            )
        )
    generated = [row.value for row in predictions]
    target_values = column_values(target, selection.column_index)
    return EvidenceVerification(
        selection,
        tuple(predictions),
        verifier._semantic_check(generated, target_values),
    )


def _row_equivalence(
    backend: Stage2Backend,
    rows: Sequence[RowPrediction],
    *,
    query_id: str,
    target_id: str,
    truth: dict[tuple[str, str, int], dict[str, Any]],
    supports: dict[str, set[int]],
    threshold: float,
) -> list[dict[str, Any]]:
    comparable = []
    for row in rows:
        expected = truth.get((query_id, target_id, row.row_id))
        if expected is not None and row.value:
            comparable.append((row.row_id, row.value, str(expected["value"])))
    similarities = {}
    if comparable:
        values = [value for _, generated, expected in comparable for value in (generated, expected)]
        embeddings = backend.embed_texts(values)
        for index, (row_id, generated, expected) in enumerate(comparable):
            exact = clean_text(generated).casefold() == clean_text(expected).casefold()
            similarity = float(
                F.cosine_similarity(
                    embeddings[index * 2].unsqueeze(0),
                    embeddings[index * 2 + 1].unsqueeze(0),
                )[0]
            )
            similarities[row_id] = (exact or similarity >= threshold, similarity)
    result = []
    for row in rows:
        expected = truth.get((query_id, target_id, row.row_id))
        evidence_id = str(row.evidence["evidence_id"]) if row.evidence else None
        equivalent, similarity = similarities.get(row.row_id, (False, None))
        evidence_supported = (
            evidence_id is not None and row.row_id in supports.get(evidence_id, set())
        )
        result.append(
            {
                "row_id": row.row_id,
                "generated_value": row.value,
                "expected_value": expected["value"] if expected else None,
                "truth_available": expected is not None,
                "semantic_similarity": similarity,
                "model_value_correct": equivalent,
                "evidence_id": evidence_id,
                "evidence_supported": evidence_supported,
                "correct_value_recovery": equivalent and evidence_supported,
            }
        )
    return result


def _condition_result(
    verifier: Stage2Verifier,
    inputs: TaskFInputs,
    *,
    query_id: str,
    target_id: str,
    selection: ColumnSelection,
    evidence_ids: Sequence[str],
    condition: str,
) -> dict[str, Any]:
    query = inputs.objects.queries[query_id]
    target = inputs.objects.targets[target_id]
    supports = _support_rows(inputs.supervision[query_id], target_id)
    started = time.monotonic()
    routed_support_rows: set[int] = set()
    if condition == "no_evidence_same_generator":
        verification = _no_evidence_recovery(verifier, query, target, selection)
    else:
        assignments = verifier.evidence_router.assign(
            query_id, evidence_ids, row_count=len(query["rows"])
        )
        for evidence_id, row_position in assignments.items():
            row_id = int(query["rows"][row_position]["row_id"])
            if row_id in supports.get(str(evidence_id), set()):
                routed_support_rows.add(row_id)
        verification = verifier.recover_candidate(
            query,
            EvidenceBundle(target_id, 0.0, tuple(evidence_ids)),
            selection,
            inputs.objects.targets,
            inputs.objects.evidence,
        )
    row_metrics = _row_equivalence(
        verifier.backend,
        verification.rows,
        query_id=query_id,
        target_id=target_id,
        truth=inputs.recovery_values,
        supports=supports,
        threshold=verifier.similarity_threshold,
    )
    recoverable_rows = sum(row["truth_available"] for row in row_metrics)
    return {
        "condition": condition,
        "evidence_ids": list(evidence_ids),
        "routed_support_rows": sorted(routed_support_rows),
        "routed_support_all_rows": len(routed_support_rows) / len(query["rows"]),
        "routed_support_recoverable_rows": (
            len(routed_support_rows) / recoverable_rows if recoverable_rows else None
        ),
        "rows": row_metrics,
        "model_correct_values": sum(row["model_value_correct"] for row in row_metrics),
        "correct_value_recoveries": sum(
            row["correct_value_recovery"] for row in row_metrics
        ),
        "recoverable_rows": recoverable_rows,
        "joinable": verification.semantic_joinability.joinable,
        "verification": asdict(verification.semantic_joinability),
        "elapsed_seconds": time.monotonic() - started,
    }


def _existing_keys(path: Path, fields: Sequence[str]) -> set[tuple[str, ...]]:
    if not path.is_file():
        return set()
    return {
        tuple(str(record[field]) for field in fields) for record in _load_jsonl(path)
    }


def run_fa_diagnostics(
    verifier: Stage2Verifier,
    inputs: TaskFInputs,
    output_path: Path,
    *,
    seed: int = 13,
    top_k_evidence: int = 4,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _existing_keys(output_path, ("query_id", "target_id"))
    with output_path.open("a", encoding="utf-8") as handle:
        for sample in inputs.sample:
            query_id = str(sample["query_id"])
            details = inputs.queue_details["frozen_fusion"][query_id]
            for queue_row in details:
                target_id = str(queue_row["target_id"])
                if (query_id, target_id) in completed:
                    continue
                qrel = inputs.qrels.get((query_id, target_id))
                if not qrel or qrel.get("reason") != "model_recoverable_join_column":
                    continue
                retrieved_ids = tuple(
                    str(value) for value in queue_row["selected_evidence_ids"]
                )[:top_k_evidence]
                if not retrieved_ids:
                    continue
                target = inputs.objects.targets[target_id]
                gold_column = local_column_index(
                    target, int(qrel["join_attribute"]["source_column_index"])
                )
                selection = ColumnSelection(
                    target_id, gold_column, column_name(target, gold_column)
                )
                oracle_ids = select_oracle_evidence(
                    inputs.supervision[query_id]["positive_evidence_by_target"][target_id],
                    inputs.objects.evidence,
                    top_k=top_k_evidence,
                )
                wrong_ids = _wrong_attribute_evidence(
                    inputs.supervision[query_id],
                    target_id,
                    count=len(retrieved_ids),
                    seed=seed,
                )
                conditions = {
                    "no_evidence_same_generator": (),
                    "retrieved": retrieved_ids,
                    "oracle_evidence": oracle_ids,
                }
                if wrong_ids:
                    conditions["wrong_attribute"] = wrong_ids
                results = {
                    name: _condition_result(
                        verifier,
                        inputs,
                        query_id=query_id,
                        target_id=target_id,
                        selection=selection,
                        evidence_ids=evidence,
                        condition=name,
                    )
                    for name, evidence in conditions.items()
                }
                no_evidence = results["no_evidence_same_generator"]
                retrieved = results["retrieved"]
                no_correct_rows = {
                    row["row_id"]
                    for row in no_evidence["rows"]
                    if row["model_value_correct"]
                }
                newly_correct_supported = [
                    row["row_id"]
                    for row in retrieved["rows"]
                    if row["correct_value_recovery"]
                    and row["row_id"] not in no_correct_rows
                ]
                record = {
                    "query_id": query_id,
                    "source_table_id": str(sample["source_table_id"]),
                    "query_kind": str(sample["kind"]),
                    "target_id": target_id,
                    "gold_column_index": gold_column,
                    "gold_column_name": selection.column_name,
                    "raw_d100_member": bool(queue_row["raw_d100_member"]),
                    "wrong_attribute_status": (
                        "matched_count" if wrong_ids else "unavailable_equal_count"
                    ),
                    "conditions": results,
                    "newly_correct_supported_rows": newly_correct_supported,
                    "evidence_enabled_join": bool(
                        not no_evidence["joinable"]
                        and retrieved["joinable"]
                        and newly_correct_supported
                    ),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()


def run_full_chain(
    verifier: Stage2Verifier,
    inputs: TaskFInputs,
    output_path: Path,
    *,
    recovery_budget: int = 10,
    top_k_evidence: int = 4,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _existing_keys(output_path, ("system", "query_id"))
    with output_path.open("a", encoding="utf-8") as handle:
        for sample in inputs.sample:
            query_id = str(sample["query_id"])
            cached_result = None
            for system in ("f1_union_direct", "frozen_fusion"):
                if (system, query_id) in completed:
                    continue
                started = time.monotonic()
                if inputs.queues_identical and cached_result is not None:
                    payload = cached_result
                    reused = True
                else:
                    payload = verifier.verify(
                        inputs.objects.queries[query_id],
                        inputs.queues[system][query_id],
                        inputs.objects.targets,
                        inputs.objects.evidence,
                        recovery_budget=recovery_budget,
                        top_k_evidence=top_k_evidence,
                    ).to_dict()
                    reused = False
                    if inputs.queues_identical:
                        cached_result = payload
                record = {
                    "system": system,
                    "query_id": query_id,
                    "source_table_id": str(sample["source_table_id"]),
                    "query_kind": str(sample["kind"]),
                    "positive_target_ids": inputs.supervision[query_id][
                        "positive_target_ids"
                    ],
                    "queue_details": inputs.queue_details[system][query_id],
                    "identical_queue_result_reused": reused,
                    "elapsed_seconds": time.monotonic() - started,
                    "stage2": payload,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()


def _candidate_rows(stage2: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        *stage2["reranked_candidates"],
        *stage2["unattempted_candidates"],
    ]


def _full_chain_query_metrics(
    record: dict[str, Any],
    inputs: TaskFInputs,
    backend: Stage2Backend,
    *,
    similarity_threshold: float,
) -> dict[str, Any]:
    query_id = str(record["query_id"])
    positive_ids = {str(value) for value in record["positive_target_ids"]}
    details = {
        str(row["target_id"]): row for row in record["queue_details"]
    }
    true_final = 0
    false_final = 0
    positive_in_queue = 0
    correct_columns = 0
    correct_column_denominator = 0
    correct_values = 0
    recoverable_rows = 0
    outside_direct = 0
    candidates = []
    for candidate in _candidate_rows(record["stage2"]):
        target_id = str(candidate["target_id"])
        positive = target_id in positive_ids
        positive_in_queue += int(positive)
        final_joinable = bool(
            candidate["verification"] and candidate["verification"]["joinable"]
        )
        true_final += int(final_joinable and positive)
        false_final += int(final_joinable and not positive)
        qrel = inputs.qrels.get((query_id, target_id))
        selection = candidate.get("selection")
        column_correct = None
        if qrel and qrel.get("reason") == "model_recoverable_join_column":
            correct_column_denominator += 1
            gold_column = local_column_index(
                inputs.objects.targets[target_id],
                int(qrel["join_attribute"]["source_column_index"]),
            )
            column_correct = bool(
                candidate.get("column_accepted")
                and selection
                and int(selection["column_index"]) == gold_column
            )
            correct_columns += int(column_correct)
            truth_rows = {
                row_id: truth
                for (saved_query, saved_target, row_id), truth in inputs.recovery_values.items()
                if saved_query == query_id and saved_target == target_id
            }
            recoverable_rows += len(truth_rows)
            evidence_rows = candidate.get("branches", {}).get("evidence", {}).get("rows", [])
            row_predictions = tuple(
                RowPrediction(
                    row_id=int(row["row_id"]),
                    value=str(row["value"]),
                    evidence=row.get("evidence"),
                )
                for row in evidence_rows
            )
            row_metrics = (
                _row_equivalence(
                    backend,
                    row_predictions,
                    query_id=query_id,
                    target_id=target_id,
                    truth=inputs.recovery_values,
                    supports=_support_rows(inputs.supervision[query_id], target_id),
                    threshold=similarity_threshold,
                )
                if row_predictions
                else []
            )
            correct_values += sum(row["correct_value_recovery"] for row in row_metrics)
        else:
            row_metrics = []
        evidence_branch = candidate.get("branches", {}).get("evidence", {})
        outside = bool(
            positive
            and not details[target_id]["raw_d100_member"]
            and evidence_branch.get("verification")
            and evidence_branch["verification"]["joinable"]
            and any(row["correct_value_recovery"] for row in row_metrics)
        )
        outside_direct += int(outside)
        candidates.append(
            {
                "target_id": target_id,
                "positive": positive,
                "final_joinable": final_joinable,
                "final_branch": candidate.get("final_branch"),
                "column_correct": column_correct,
                "row_metrics": row_metrics,
                "outside_direct_final_join": outside,
            }
        )
    return {
        "system": record["system"],
        "query_id": query_id,
        "source_table_id": record["source_table_id"],
        "query_kind": record["query_kind"],
        "gold_targets": len(positive_ids),
        "positive_targets_in_queue": positive_in_queue,
        "true_final_joins": true_final,
        "false_final_joins": false_final,
        "correct_columns": correct_columns,
        "correct_column_denominator": correct_column_denominator,
        "correct_value_recoveries": correct_values,
        "recoverable_rows": recoverable_rows,
        "outside_direct_final_joins": outside_direct,
        "candidates": candidates,
    }


def select_task_f_audit_cases(
    fa_rows: Sequence[dict[str, Any]],
    query_metrics: Sequence[dict[str, Any]],
    *,
    failure_fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    if not 0.0 <= failure_fraction <= 1.0:
        raise ValueError("Task F audit failure fraction must be between zero and one")
    reasons: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in fa_rows:
        if row["evidence_enabled_join"]:
            reasons[(str(row["query_id"]), str(row["target_id"]))].add(
                "evidence_enabled_join_success"
            )
    for row in query_metrics:
        for candidate in row["candidates"]:
            if candidate["outside_direct_final_join"]:
                reasons[(str(row["query_id"]), str(candidate["target_id"]))].add(
                    "outside_direct_final_join_success"
                )

    eligible_failures = sorted(
        {
            (str(row["query_id"]), str(row["target_id"]))
            for row in fa_rows
            if not row["evidence_enabled_join"]
        }
        - reasons.keys(),
        key=lambda pair: hashlib.sha256(
            f"{seed}\0failure\0{pair[0]}\0{pair[1]}".encode()
        ).digest(),
    )
    failure_count = math.ceil(len(eligible_failures) * failure_fraction)
    for pair in eligible_failures[:failure_count]:
        reasons[pair].add("evidence_enabled_join_failure_sample")
    return [
        {
            "query_id": query_id,
            "target_id": target_id,
            "selection_reasons": sorted(values),
        }
        for (query_id, target_id), values in sorted(reasons.items())
    ]


def _review_evidence(record: dict[str, Any]) -> dict[str, Any]:
    modality = str(record["asset_type"])
    return {
        "evidence_id": str(record["asset_id"]),
        "modality": modality,
        "evidence_text": record.get("content") if modality == "text" else None,
        "image_path": record.get("local_path") if modality == "image" else None,
    }


def prepare_task_f_human_audit(
    inputs: TaskFInputs,
    backend: Stage2Backend,
    *,
    fa_path: Path,
    full_chain_path: Path,
    output_dir: Path,
    failure_fraction: float = 0.1,
    seed: int = 13,
    similarity_threshold: float = 0.8,
) -> dict[str, Any]:
    fa_rows = list(_load_jsonl(fa_path))
    full_rows = list(_load_jsonl(full_chain_path))
    query_metrics = [
        _full_chain_query_metrics(
            row,
            inputs,
            backend,
            similarity_threshold=similarity_threshold,
        )
        for row in full_rows
    ]
    selections = select_task_f_audit_cases(
        fa_rows,
        query_metrics,
        failure_fraction=failure_fraction,
        seed=seed,
    )
    fa_by_pair = {
        (str(row["query_id"]), str(row["target_id"])): row for row in fa_rows
    }
    metrics_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in query_metrics:
        for candidate in row["candidates"]:
            pair = (str(row["query_id"]), str(candidate["target_id"]))
            metrics_by_pair[pair].append(
                {
                    "system": row["system"],
                    "final_joinable": candidate["final_joinable"],
                    "final_branch": candidate["final_branch"],
                    "column_correct": candidate["column_correct"],
                    "row_metrics": candidate["row_metrics"],
                }
            )

    packets = []
    selection_rows = []
    templates = []
    for selection in selections:
        query_id = selection["query_id"]
        target_id = selection["target_id"]
        pair = (query_id, target_id)
        fa_row = fa_by_pair.get(pair)
        query = inputs.objects.queries[query_id]
        target = inputs.objects.targets[target_id]
        if fa_row is not None:
            column_index = int(fa_row["gold_column_index"])
            attribute_name = str(fa_row["gold_column_name"])
        else:
            qrel = inputs.qrels[pair]
            column_index = local_column_index(
                target, int(qrel["join_attribute"]["source_column_index"])
            )
            attribute_name = column_name(target, column_index)
        condition_rows = {}
        evidence_ids = set()
        if fa_row is not None:
            for condition, result in fa_row["conditions"].items():
                evidence_ids.update(str(value) for value in result["evidence_ids"])
                condition_rows[condition] = {
                    "evidence_ids": result["evidence_ids"],
                    "joinable": result["joinable"],
                    "rows": [
                        {
                            "row_id": row["row_id"],
                            "generated_value": row["generated_value"],
                            "evidence_id": row["evidence_id"],
                        }
                        for row in result["rows"]
                    ],
                }
        for metric in metrics_by_pair[pair]:
            evidence_ids.update(
                str(row["evidence_id"])
                for row in metric["row_metrics"]
                if row["evidence_id"] is not None
            )
        case_id = "r12_task_f_" + hashlib.sha256(
            f"{seed}\0{query_id}\0{target_id}".encode()
        ).hexdigest()[:16]
        packets.append(
            {
                "case_id": case_id,
                "query_id": query_id,
                "target_id": target_id,
                "query_kind": inputs.supervision[query_id]["query_kind"],
                "visible_query_rows": [row_values(query, row) for row in query["rows"]],
                "target_schema": [
                    {
                        "column_index": int(column["column_index"]),
                        "column_name": column_name(
                            target, int(column["column_index"])
                        ),
                    }
                    for column in target["columns"]
                ],
                "requested_attribute": attribute_name,
                "target_values": column_values(target, column_index),
                "conditions": condition_rows,
                "evidence": [
                    _review_evidence(inputs.objects.evidence[evidence_id])
                    for evidence_id in sorted(evidence_ids)
                ],
            }
        )
        selection_rows.append(
            {
                "case_id": case_id,
                **selection,
                "automated_f_a": fa_row,
                "automated_full_chain": metrics_by_pair[pair],
                "independently_confirmed": False,
            }
        )
        templates.append(
            {
                "case_id": case_id,
                "reviewer_id": None,
                "evidence_supports_requested_attribute": None,
                "generated_values_supported": None,
                "join_decision_correct": None,
                "notes": None,
            }
        )

    review_path = output_dir / "review_packets.jsonl"
    selection_path = output_dir / "automated_selection_do_not_show_reviewers.jsonl"
    template_path = output_dir / "review_template.jsonl"
    _write_jsonl(review_path, packets)
    _write_jsonl(selection_path, selection_rows)
    _write_jsonl(template_path, templates)
    success_cases = sum(
        any(reason.endswith("_success") for reason in row["selection_reasons"])
        for row in selections
    )
    failure_cases = sum(
        "evidence_enabled_join_failure_sample" in row["selection_reasons"]
        for row in selections
    )
    selected_success_pairs = {
        (row["query_id"], row["target_id"])
        for row in selections
        if any(reason.endswith("_success") for reason in row["selection_reasons"])
    }
    eligible_failures = sum(
        not row["evidence_enabled_join"]
        and (str(row["query_id"]), str(row["target_id"])) not in selected_success_pairs
        for row in fa_rows
    )
    manifest = {
        "format_version": 1,
        "status": "prepared_not_reviewed",
        "selection_policy": "all automated new successes plus a deterministic failure sample",
        "failure_fraction": failure_fraction,
        "seed": seed,
        "eligible_failure_pairs": eligible_failures,
        "success_cases": success_cases,
        "failure_sample_cases": failure_cases,
        "review_cases": len(packets),
        "independent_reviews_completed": 0,
        "review_packets": str(review_path.resolve()),
        "review_packets_sha256": checkpoint_fingerprint(review_path),
        "automated_selection": str(selection_path.resolve()),
        "automated_selection_sha256": checkpoint_fingerprint(selection_path),
        "review_template": str(template_path.resolve()),
        "review_template_sha256": checkpoint_fingerprint(template_path),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def _fa_condition_summary(fa_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    names = (
        "no_evidence_same_generator",
        "retrieved",
        "wrong_attribute",
        "oracle_evidence",
    )
    result = {}
    for name in names:
        rows = [row["conditions"][name] for row in fa_rows if name in row["conditions"]]
        recoverable = sum(int(row["recoverable_rows"]) for row in rows)
        model_correct = sum(int(row["model_correct_values"]) for row in rows)
        supported = sum(int(row["correct_value_recoveries"]) for row in rows)
        result[name] = {
            "pairs": len(rows),
            "joinable_pairs": sum(bool(row["joinable"]) for row in rows),
            "joinable_rate": (
                sum(bool(row["joinable"]) for row in rows) / len(rows) if rows else None
            ),
            "model_correct_values": model_correct,
            "correct_value_recoveries": supported,
            "recoverable_rows": recoverable,
            "model_value_accuracy": model_correct / recoverable if recoverable else None,
            "supported_value_recovery": supported / recoverable if recoverable else None,
        }
    return result


def summarize_task_f(
    inputs: TaskFInputs,
    backend: Stage2Backend,
    *,
    fa_path: Path,
    full_chain_path: Path,
    similarity_threshold: float = 0.8,
) -> dict[str, Any]:
    fa_rows = list(_load_jsonl(fa_path))
    full_rows = list(_load_jsonl(full_chain_path))
    if len({row["query_id"] for row in full_rows}) != 128 or len(full_rows) != 256:
        raise ValueError("Task F full-chain output is incomplete")
    query_metrics = [
        _full_chain_query_metrics(
            row,
            inputs,
            backend,
            similarity_threshold=similarity_threshold,
        )
        for row in full_rows
    ]
    fa_by_query = defaultdict(int)
    for row in fa_rows:
        fa_by_query[str(row["query_id"])] += int(row["evidence_enabled_join"])
    by_system = {}
    for system in ("f1_union_direct", "frozen_fusion"):
        rows = [row for row in query_metrics if row["system"] == system]
        true_joins = sum(row["true_final_joins"] for row in rows)
        false_joins = sum(row["false_final_joins"] for row in rows)
        gold_targets = sum(row["gold_targets"] for row in rows)
        positive_in_queue = sum(row["positive_targets_in_queue"] for row in rows)
        correct_columns = sum(row["correct_columns"] for row in rows)
        column_denominator = sum(row["correct_column_denominator"] for row in rows)
        correct_values = sum(row["correct_value_recoveries"] for row in rows)
        recoverable_rows = sum(row["recoverable_rows"] for row in rows)
        outside = sum(row["outside_direct_final_joins"] for row in rows)
        by_system[system] = {
            "queries": len(rows),
            "true_final_joins": true_joins,
            "false_final_joins": false_joins,
            "final_join_precision": true_joins / (true_joins + false_joins)
            if true_joins + false_joins
            else None,
            "final_join_recall_all_gold": true_joins / gold_targets,
            "final_join_recall_given_positive_in_queue": true_joins / positive_in_queue
            if positive_in_queue
            else None,
            "correct_column": correct_columns / column_denominator
            if column_denominator
            else None,
            "correct_column_numerator": correct_columns,
            "correct_column_denominator": column_denominator,
            "correct_value_recovery": correct_values / recoverable_rows
            if recoverable_rows
            else None,
            "correct_value_numerator": correct_values,
            "correct_value_denominator": recoverable_rows,
            "evidence_enabled_join_queries": sum(value > 0 for value in fa_by_query.values()),
            "evidence_enabled_join_pairs": sum(fa_by_query.values()),
            "outside_direct_final_join": outside,
        }
    by_key = {
        (row["system"], row["query_id"]): row for row in query_metrics
    }
    query_order = [str(row["query_id"]) for row in inputs.sample]
    evidence_left = [float(fa_by_query[query_id] > 0) for query_id in query_order]
    if not inputs.queues_identical:
        raise ValueError("F-a comparison requires diagnostics for both distinct queues")
    evidence_right = list(evidence_left)
    outside_left = [
        float(by_key[("frozen_fusion", query_id)]["outside_direct_final_joins"] > 0)
        for query_id in query_order
    ]
    outside_right = [
        float(by_key[("f1_union_direct", query_id)]["outside_direct_final_joins"] > 0)
        for query_id in query_order
    ]
    bootstrap = {
        "evidence_enabled_join": paired_bootstrap_delta(
            evidence_left, evidence_right, iterations=10_000, seed=13
        ),
        "outside_direct_final_join": paired_bootstrap_delta(
            outside_left, outside_right, iterations=10_000, seed=13
        ),
        "unit": "source_table_id",
        "source_groups": len({row["source_table_id"] for row in inputs.sample}),
        "note": (
            "The frozen Task E fusion is F1, so both paired inputs are identical."
            if inputs.queues_identical
            else None
        ),
    }
    return {
        "format_version": 1,
        "status": "automated_complete_human_audit_pending",
        "inputs": inputs.manifest,
        "by_system": by_system,
        "f_a": {
            "pairs": len(fa_rows),
            "wrong_attribute_matched": sum(
                row["wrong_attribute_status"] == "matched_count" for row in fa_rows
            ),
            "evidence_enabled_join_pairs": sum(
                row["evidence_enabled_join"] for row in fa_rows
            ),
            "newly_correct_supported_rows": sum(
                len(row["newly_correct_supported_rows"]) for row in fa_rows
            ),
            "by_condition": _fa_condition_summary(fa_rows),
        },
        "bootstrap": bootstrap,
        "limitations": [
            "Canonical qrels and recovery values provide automated reference labels; required independent human review remains pending.",
            "The Task E quality gate selected F1 itself as frozen fusion, so the primary fusion-minus-F1 effect is structurally zero.",
        ],
    }
