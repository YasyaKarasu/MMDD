#!/usr/bin/env python
"""Prepare and validate independent human reviews for the R12 experiments."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json


ATTRIBUTE_LABELS = {
    "confirmed_support",
    "confirmed_wrong_attribute",
    "confirmed_entity_or_value_conflict",
    "insufficient_information",
}
ATTRIBUTE_UNKNOWN = {"insufficient_information", "review_disagreement"}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _by_case(rows: Sequence[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        case_id = str(row.get("case_id") or "")
        if not case_id:
            raise ValueError(f"{path}: every review must have a case_id")
        if case_id in result:
            raise ValueError(f"{path}: duplicate case_id {case_id!r}")
        result[case_id] = row
    return result


def _attribute_template(case_id: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "reviewer_id": None,
        "reviewer_role": "independent_human",
        "reviewed_at_utc": None,
        "label": None,
        "supported_value": None,
        "evidence_locator": None,
        "notes": None,
    }


def _task_f_template(case_id: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "reviewer_id": None,
        "reviewer_role": "independent_human",
        "reviewed_at_utc": None,
        "evidence_supports_requested_attribute": None,
        "generated_values_supported": None,
        "join_decision_correct": None,
        "notes": None,
    }


def prepare_templates(output_root: Path) -> dict[str, Any]:
    task_b = output_root / "taskB_attribute_audit"
    task_f = output_root / "taskF_end_to_end" / "human_audit"
    primary_packets = _load_jsonl(task_b / "review_packets.jsonl")
    second_packets = _load_jsonl(task_b / "second_review_packets.jsonl")
    task_f_packets = _load_jsonl(task_f / "review_packets.jsonl")
    _by_case(primary_packets, task_b / "review_packets.jsonl")
    _by_case(second_packets, task_b / "second_review_packets.jsonl")
    _by_case(task_f_packets, task_f / "review_packets.jsonl")
    paths = {
        "attribute_primary": task_b / "primary_reviews.jsonl",
        "attribute_second": task_b / "second_reviews.jsonl",
        "task_f": task_f / "human_reviews.jsonl",
    }
    templates = {
        "attribute_primary": [
            _attribute_template(str(row["case_id"])) for row in primary_packets
        ],
        "attribute_second": [
            _attribute_template(str(row["case_id"])) for row in second_packets
        ],
        "task_f": [_task_f_template(str(row["case_id"])) for row in task_f_packets],
    }
    for name, path in paths.items():
        if not path.exists():
            _write_jsonl(path, templates[name])
    instructions = "\n".join(
        [
            "# R12 independent review instructions",
            "",
            "Only independent humans may complete these submissions. Model assistance "
            "and historical labels are not human review.",
            "",
            "## Task B primary review",
            "",
            "Read `taskB_attribute_audit/review_packets.jsonl` and edit only "
            "`taskB_attribute_audit/primary_reviews.jsonl`. Do not inspect "
            "`model_assistance.jsonl` or historical-reference files before reviewing.",
            "",
            "Use exactly one label per case:",
            "",
            "- `confirmed_support`: the evidence supports the requested attribute for "
            "the specified entity/row; record `supported_value` and a precise "
            "`evidence_locator`.",
            "- `confirmed_wrong_attribute`: the evidence concerns the entity but does "
            "not support the requested attribute.",
            "- `confirmed_entity_or_value_conflict`: the evidence concerns another "
            "entity or contradicts the candidate value.",
            "- `insufficient_information`: the evidence is inadequate to decide.",
            "",
            "## Task B second review",
            "",
            "A different human reads `second_review_packets.jsonl` and edits only "
            "`second_reviews.jsonl`, without seeing primary submissions. Reviewer IDs "
            "must differ on every double-reviewed case.",
            "",
            "## Task F review",
            "",
            "Read `taskF_end_to_end/human_audit/review_packets.jsonl` and edit "
            "`taskF_end_to_end/human_audit/human_reviews.jsonl`. All three judgment "
            "fields must be JSON booleans (`true` or `false`).",
            "",
            "Every submission requires a stable `reviewer_id`, "
            "`reviewer_role=independent_human`, and an ISO-8601 `reviewed_at_utc`.",
            "",
            "After all files are complete, run:",
            "",
            "```bash",
            "PYTHONPATH=src conda run -n MMDD python "
            "src/finalize_stage1_r12_reviews.py finalize \\",
            f"  --output-root {output_root.resolve()}",
            "```",
            "",
        ]
    )
    (output_root / "HUMAN_REVIEW_INSTRUCTIONS.md").write_text(
        instructions, encoding="utf-8"
    )
    workflow = {
        "format_version": 1,
        "status": "prepared_not_reviewed",
        "attribute_labels": sorted(ATTRIBUTE_LABELS),
        "attribute_primary_cases": len(primary_packets),
        "attribute_second_cases": len(second_packets),
        "task_f_cases": len(task_f_packets),
        "submissions": {
            name: {
                "path": str(path.resolve()),
                "sha256_at_preparation": checkpoint_fingerprint(path),
            }
            for name, path in paths.items()
        },
        "independence_policy": (
            "reviewer_role must be independent_human; the two Task B reviewers "
            "must have different reviewer_id values for every double-reviewed case"
        ),
        "resolution_policy": (
            "matching labels are accepted; double-review disagreements become "
            "review_disagreement and remain unknown"
        ),
    }
    write_json(output_root / "HUMAN_REVIEW_WORKFLOW.json", workflow)
    return workflow


def _validate_reviewer(row: dict[str, Any], path: Path, case_id: str) -> None:
    reviewer_id = str(row.get("reviewer_id") or "").strip()
    if not reviewer_id:
        raise ValueError(f"{path}: {case_id}: reviewer_id is required")
    if row.get("reviewer_role") != "independent_human":
        raise ValueError(
            f"{path}: {case_id}: reviewer_role must be independent_human"
        )
    reviewed_at = str(row.get("reviewed_at_utc") or "").strip()
    if not reviewed_at:
        raise ValueError(f"{path}: {case_id}: reviewed_at_utc is required")
    try:
        reviewed = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"{path}: {case_id}: reviewed_at_utc must be ISO-8601"
        ) from exc
    if reviewed.utcoffset() != timezone.utc.utcoffset(reviewed):
        raise ValueError(f"{path}: {case_id}: reviewed_at_utc must be UTC")


def _reviewer_id(row: dict[str, Any]) -> str:
    return str(row.get("reviewer_id") or "").strip()


def _validate_exact_cases(
    rows: Sequence[dict[str, Any]], expected: set[str], path: Path
) -> dict[str, dict[str, Any]]:
    indexed = _by_case(rows, path)
    missing = expected - indexed.keys()
    extra = indexed.keys() - expected
    if missing or extra:
        raise ValueError(
            f"{path}: review cases differ from the frozen packet; "
            f"missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}"
        )
    return indexed


def _validate_attribute_review(
    row: dict[str, Any], path: Path, case_id: str
) -> None:
    _validate_reviewer(row, path, case_id)
    label = row.get("label")
    if label not in ATTRIBUTE_LABELS:
        raise ValueError(f"{path}: {case_id}: invalid attribute label {label!r}")
    if label == "confirmed_support":
        if not str(row.get("supported_value") or "").strip():
            raise ValueError(f"{path}: {case_id}: supported_value is required")
        if not str(row.get("evidence_locator") or "").strip():
            raise ValueError(f"{path}: {case_id}: evidence_locator is required")


def _rate_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["resolved_label"]) for row in rows)
    total = len(rows)
    weights = Counter()
    for row in rows:
        weights[str(row["resolved_label"])] += float(row["sampling_weight"])
    weight_total = sum(weights.values())
    return {
        "cases": total,
        "counts": dict(sorted(counts.items())),
        "confirmed_support_rate": counts["confirmed_support"] / total if total else None,
        "unknown_rate": (
            sum(counts[label] for label in ATTRIBUTE_UNKNOWN) / total
            if total
            else None
        ),
        "weighted_label_distribution": {
            label: value / weight_total for label, value in sorted(weights.items())
        },
        "weighted_estimate_scope": "conditional sampled candidate pool",
    }


def _support_model_gate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    train_rows = [row for row in rows if row["split"] == "train_fit"]
    positives = [
        row for row in train_rows if row["resolved_label"] == "confirmed_support"
    ]
    negatives = [
        row
        for row in train_rows
        if row["resolved_label"]
        in {"confirmed_wrong_attribute", "confirmed_entity_or_value_conflict"}
    ]
    positive_groups = {str(row["source_table_id"]) for row in positives}
    negative_groups = {str(row["source_table_id"]) for row in negatives}
    result = {
        "positive_cases": len(positives),
        "negative_cases": len(negatives),
        "positive_source_groups": len(positive_groups),
        "negative_source_groups": len(negative_groups),
    }
    result["eligible"] = bool(
        len(positives) >= 64
        and len(negatives) >= 64
        and len(positive_groups) >= 20
        and len(negative_groups) >= 20
    )
    return result


def finalize_attribute_reviews(output_root: Path) -> dict[str, Any]:
    task_b = output_root / "taskB_attribute_audit"
    primary_path = task_b / "primary_reviews.jsonl"
    second_path = task_b / "second_reviews.jsonl"
    packet_path = task_b / "review_packets.jsonl"
    second_packet_path = task_b / "second_review_packets.jsonl"
    packet_ids = set(_by_case(_load_jsonl(packet_path), packet_path))
    second_ids = set(
        _by_case(_load_jsonl(second_packet_path), second_packet_path)
    )
    if not second_ids <= packet_ids:
        raise ValueError("Second-review packet is not a subset of the primary packet")
    primary = _validate_exact_cases(_load_jsonl(primary_path), packet_ids, primary_path)
    second = _validate_exact_cases(_load_jsonl(second_path), second_ids, second_path)
    for case_id, row in primary.items():
        _validate_attribute_review(row, primary_path, case_id)
    for case_id, row in second.items():
        _validate_attribute_review(row, second_path, case_id)
        if _reviewer_id(row) == _reviewer_id(primary[case_id]):
            raise ValueError(
                f"{second_path}: {case_id}: double review must use another reviewer"
            )

    candidates = _by_case(
        _load_jsonl(task_b / "selected_candidates.jsonl"),
        task_b / "selected_candidates.jsonl",
    )
    if candidates.keys() != packet_ids:
        raise ValueError("Selected candidate IDs differ from the frozen review packet")
    resolved = []
    for case_id in sorted(packet_ids):
        first = primary[case_id]
        other = second.get(case_id)
        label = str(first["label"])
        agreement = None
        if other is not None:
            agreement = label == other["label"]
            if not agreement:
                label = "review_disagreement"
        candidate = candidates[case_id]
        resolved.append(
            {
                **candidate,
                "case_id": case_id,
                "resolved_label": label,
                "double_reviewed": other is not None,
                "double_review_agreement": agreement,
                "reviewer_ids": [
                    _reviewer_id(first),
                    *([_reviewer_id(other)] if other is not None else []),
                ],
                "supported_value": (
                    first.get("supported_value")
                    if label == "confirmed_support"
                    else None
                ),
                "evidence_locator": (
                    first.get("evidence_locator")
                    if label == "confirmed_support"
                    else None
                ),
            }
        )

    by_bucket = {}
    for split in ("train_fit", "dev"):
        for modality in ("text", "image"):
            name = f"{split}/{modality}"
            by_bucket[name] = _rate_summary(
                [
                    row
                    for row in resolved
                    if row["split"] == split and row["modality"] == modality
                ]
            )
    support_model_gate = _support_model_gate(resolved)
    supports_by_context: dict[tuple[str, str, int, int], set[str]] = defaultdict(set)
    supported_attributes: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in resolved:
        if row["resolved_label"] != "confirmed_support":
            continue
        context = (
            str(row["query_id"]),
            str(row["target_id"]),
            int(row["row_id"]),
            int(row["source_column_id"]),
        )
        supports_by_context[context].add(str(row["evidence_id"]))
        supported_attributes[(context[0], context[1])].add(context[3])
    confirmed_support = [
        row for row in resolved if row["resolved_label"] == "confirmed_support"
    ]
    routed_correct = sum(
        int(row.get("routing_row") == row.get("row_id")) for row in confirmed_support
    )
    disagreements = sum(
        row["double_reviewed"] and not row["double_review_agreement"]
        for row in resolved
    )
    payload = {
        "format_version": 1,
        "status": "complete",
        "reviews": len(resolved),
        "double_reviews": len(second),
        "double_review_fraction": len(second) / len(resolved),
        "double_review_minimum_met": len(second) >= math.ceil(0.2 * len(resolved)),
        "double_review_disagreements": disagreements,
        "overall": _rate_summary(resolved),
        "by_bucket": by_bucket,
        "actual_routing": {
            "confirmed_support_cases": len(confirmed_support),
            "correct_row": routed_correct,
            "wrong_row": len(confirmed_support) - routed_correct,
            "wrong_row_rate": (
                (len(confirmed_support) - routed_correct) / len(confirmed_support)
                if confirmed_support
                else None
            ),
        },
        "image_independent_attribute_support": _rate_summary(
            [row for row in resolved if row["modality"] == "image"]
        ),
        "pairs_with_multiple_confirmed_bridge_attributes": sum(
            len(columns) > 1 for columns in supported_attributes.values()
        ),
        "support_model_gate": support_model_gate,
        "intervention_material": {
            "confirmed_wrong_attribute_cases": sum(
                row["resolved_label"] == "confirmed_wrong_attribute"
                for row in resolved
            ),
            "same_row_multiple_confirmed_evidence_contexts": sum(
                len(evidence_ids) >= 2
                for evidence_ids in supports_by_context.values()
            ),
        },
        "inputs": {
            "primary": str(primary_path.resolve()),
            "primary_sha256": checkpoint_fingerprint(primary_path),
            "second": str(second_path.resolve()),
            "second_sha256": checkpoint_fingerprint(second_path),
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_jsonl(task_b / "resolved_labels.jsonl", resolved)
    payload["resolved_labels"] = str((task_b / "resolved_labels.jsonl").resolve())
    payload["resolved_labels_sha256"] = checkpoint_fingerprint(
        task_b / "resolved_labels.jsonl"
    )
    write_json(task_b / "review_summary.json", payload)
    return payload


def finalize_task_f_reviews(output_root: Path) -> dict[str, Any]:
    task_f = output_root / "taskF_end_to_end" / "human_audit"
    review_path = task_f / "human_reviews.jsonl"
    packet_path = task_f / "review_packets.jsonl"
    packet_ids = set(_by_case(_load_jsonl(packet_path), packet_path))
    reviews = _validate_exact_cases(_load_jsonl(review_path), packet_ids, review_path)
    fields = (
        "evidence_supports_requested_attribute",
        "generated_values_supported",
        "join_decision_correct",
    )
    for case_id, row in reviews.items():
        _validate_reviewer(row, review_path, case_id)
        for field in fields:
            if not isinstance(row.get(field), bool):
                raise ValueError(f"{review_path}: {case_id}: {field} must be boolean")
    payload = {
        "format_version": 1,
        "status": "complete",
        "reviews": len(reviews),
        "counts": {
            field: sum(bool(row[field]) for row in reviews.values()) for field in fields
        },
        "input": str(review_path.resolve()),
        "input_sha256": checkpoint_fingerprint(review_path),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(task_f / "review_summary.json", payload)
    return payload


def finalize_reviews(output_root: Path) -> dict[str, Any]:
    result = {
        "attribute": finalize_attribute_reviews(output_root),
        "task_f": finalize_task_f_reviews(output_root),
    }
    with (output_root / "runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "task": "B and F independent human review finalization",
                    "status": "complete",
                    "command": [sys.executable, *sys.argv],
                    "output": str(output_root / "taskB_attribute_audit/review_summary.json"),
                    "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            + "\n"
        )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "finalize"))
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    output_root = args.output_root.resolve()
    result = (
        prepare_templates(output_root)
        if args.command == "prepare"
        else finalize_reviews(output_root)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
