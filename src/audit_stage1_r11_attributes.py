#!/usr/bin/env python
"""Build the fixed R11 train/dev attribute-consistency audit sample."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mmdd_stage1.retrieval import checkpoint_fingerprint


def _priority(seed: int, *values: str) -> str:
    return hashlib.sha256(
        (str(seed) + "\0" + "\0".join(values)).encode("utf-8")
    ).hexdigest()


def _source_map(path: Path) -> dict[str, str]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result[str(row["table_id"])] = str(row["source_table_id"])
    return result


def _candidates(
    path: Path, split_name: str, source_by_query: dict[str, str]
) -> list[dict[str, Any]]:
    result = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            query_id = str(record["query_id"])
            positives = {str(value) for value in record["positive_target_ids"]}
            expected = {
                str(target_id): {str(value) for value in evidence_ids}
                for target_id, evidence_ids in record.get(
                    "positive_evidence_by_target", {}
                ).items()
            }
            rows = record.get("positive_evidence_rows_by_target", {})
            seen = set()
            for target_id, paths in record["paths_by_target"].items():
                for path_row in paths:
                    if path_row.get("kind") != "evidence":
                        continue
                    evidence_id = str(path_row["evidence_id"])
                    key = (str(target_id), evidence_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    confirmed = (
                        str(target_id) in positives
                        and evidence_id in expected.get(str(target_id), set())
                    )
                    result.append(
                        {
                            "split": split_name,
                            "query_id": query_id,
                            "source_table_id": source_by_query[query_id],
                            "target_id": str(target_id),
                            "evidence_id": evidence_id,
                            "evidence_type": str(path_row["evidence_type"]),
                            "path_score": float(path_row["path_score"]),
                            "label": (
                                "confirmed_correct_support" if confirmed else "unknown"
                            ),
                            "confirmed_support_rows": [
                                int(value)
                                for value in rows.get(str(target_id), {}).get(
                                    evidence_id, []
                                )
                            ],
                            "wrong_attribute": None,
                            "wrong_entity_or_value": None,
                            "review_policy": (
                                "Existing recovery confirms support"
                                if confirmed
                                else "No complete negative evidence; manual review required"
                            ),
                        }
                    )
    return result


def _sample(
    candidates: list[dict[str, Any]], *, modality: str, count: int, seed: int
) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        if row["evidence_type"] == modality:
            by_source[row["source_table_id"]].append(row)
    for source_id, rows in by_source.items():
        rows.sort(
            key=lambda row: _priority(
                seed,
                row["split"],
                source_id,
                row["query_id"],
                row["target_id"],
                row["evidence_id"],
            )
        )
    source_ids = sorted(
        by_source, key=lambda source_id: _priority(seed, modality, source_id)
    )
    result = []
    offset = 0
    while len(result) < count:
        added = 0
        for source_id in source_ids:
            rows = by_source[source_id]
            if offset < len(rows):
                result.append(rows[offset])
                added += 1
                if len(result) == count:
                    break
        if not added:
            break
        offset += 1
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_by_query = _source_map(Path(args.query_tables))
    selected = []
    pool_rows = {}
    for split_name, value in (("train_fit", args.train_pool), ("dev", args.dev_pool)):
        path = Path(value).resolve()
        candidates = _candidates(path, split_name, source_by_query)
        pool_rows[split_name] = {
            "path": str(path),
            "sha256": checkpoint_fingerprint(path),
            "candidate_paths": len(candidates),
        }
        for modality in ("text", "image"):
            selected.extend(
                _sample(
                    candidates,
                    modality=modality,
                    count=args.per_bucket,
                    seed=args.seed,
                )
            )
    selected.sort(
        key=lambda row: (
            row["split"], row["evidence_type"], row["source_table_id"], row["query_id"]
        )
    )
    counts = Counter(
        (row["split"], row["evidence_type"], row["label"]) for row in selected
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "audit_candidates.jsonl"
    with sample_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    payload = {
        "format_version": 1,
        "seed": args.seed,
        "sampling_policy": "source-diverse hash order within split and modality; model-score blind",
        "requested_per_split_modality": args.per_bucket,
        "pools": pool_rows,
        "selected_paths": len(selected),
        "selected_source_groups": len({row["source_table_id"] for row in selected}),
        "counts": [
            {
                "split": split,
                "modality": modality,
                "label": label,
                "count": count,
            }
            for (split, modality, label), count in sorted(counts.items())
        ],
        "confirmed_wrong_attribute_paths": 0,
        "confirmed_wrong_attribute_source_groups": 0,
        "support_predictor_prerequisite": {
            "confirmed_negative_query_evidence_pairs": 0,
            "eligible": False,
            "reason": "Unverified paths remain unknown; only positive recovery labels are available."
        },
        "e3_triggered": False,
        "e3_reason": "No independently confirmed wrong-attribute paths and no confirmed negative QE labels.",
        "sample": str(sample_path.resolve()),
        "sample_sha256": checkpoint_fingerprint(sample_path),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "RESULTS.md").write_text(
        "# R11 Attribute Consistency Audit\n\n"
        f"Selected {len(selected)} fixed candidate paths across train-fit/dev and text/image. "
        "Existing recovery records can confirm positive support, but do not establish "
        "complete negative or wrong-attribute labels for the remaining candidates; those "
        "records remain `unknown`. E3 and a supervised row-support classifier were not "
        "triggered.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "pass", "output_dir": str(output_dir)}, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pool", required=True)
    parser.add_argument("--dev-pool", required=True)
    parser.add_argument("--query-tables", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-bucket", type=int, default=64)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    if args.per_bucket <= 0:
        parser.error("--per-bucket must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
