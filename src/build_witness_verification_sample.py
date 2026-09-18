#!/usr/bin/env python
"""Build score-free verification packets for the Experiment-1 witness labels.

The judge sees only what the retrieval model itself sees (the frozen object
serialization) and never a model score, a rank, or a candidate list.  Packets
are drawn from four groups so the check is not restricted to confirmed
positives:

  retained_witness        positive target whose verified witness survived budget-4
  unretrieved_witness     positive target whose verified witness never entered the
                          pre-retention bag
  unknown_positive        positive target with no witness label at all
  top10_competitor        non-GT target that QT puts in the Top-10

Group `unknown_positive` and group `top10_competitor` are the ones that matter:
they attack the two ways this whole audit could be circular -- label-source
incompleteness on positives, and unlabelled false negatives among competitors.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "work/witness_diagnostic_20260916"
RERANK = ROOT / "work/final_rerank_20260916/FINAL_RERANK"
DATASET = ROOT / "output_mm_joinability_entitables_20000_retry100_rounds5_qwen35_final_survivor_context_gaussian_v9"
OBJECTS = ROOT / "work/stage1_optimization_r10_20260907/stage1_data/stage1_objects.jsonl"
SEED = 260916


def rows(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-group", type=int, default=120)
    args = parser.parse_args()
    rng = random.Random(SEED)

    # Which evidence a group should be judged on.
    group_pick: dict[tuple[str, str, str], tuple[str, str]] = {}
    for record in rows(IN / "WITNESS_DIAGNOSTIC.jsonl.gz"):
        key = (record["endpoint"], record["query_id"], record["target_id"])
        if record["witness_label"] == "verified_positive":
            if record["furthest_stage"] == "retained":
                group_pick.setdefault(key, ("retained_witness", record["evidence_id"]))
            elif record["furthest_stage"] == "absent":
                if key not in group_pick:
                    prior = group_pick.get(key)
                    if prior is None:
                        group_pick[key] = ("unretrieved_witness", record["evidence_id"])
        elif record["witness_label"] == "unknown" and record["target_is_positive"]:
            group_pick.setdefault(key, ("unknown_positive", None))

    buckets: dict[str, list[tuple[tuple[str, str, str], str | None]]] = defaultdict(list)
    for key, value in group_pick.items():
        buckets[value[0]].append((key, value[1]))

    # Competitors: pull the evidence from the frozen retained bags.
    membership = {
        (record["endpoint"], record["query_id"]): record
        for record in rows(RERANK / "candidates/path_membership.jsonl.gz")
    }
    qt_rankings = {
        (record["endpoint"], record["query_id"]): record
        for record in rows(RERANK / "rankings/QT.jsonl.gz")
    }
    competitor_candidates: list[tuple[tuple[str, str, str], str | None]] = []
    unknown_candidates: list[tuple[tuple[str, str, str], str | None]] = []
    for (endpoint, query_id), record in membership.items():
        qt = qt_rankings[(endpoint, query_id)]
        positives = {str(value) for value in qt["positive_target_ids"]}
        for target in record["targets"]:
            target_id = str(target["target_id"])
            if not target.get("retained_paths"):
                continue
            best = max(
                target["retained_paths"], key=lambda path: float(path.get("path_score") or 0.0)
            )
            candidate = ((endpoint, query_id, target_id), str(best["evidence_id"]))
            if target_id in positives:
                # Only fill the ones the label pass left unknown.
                if group_pick.get((endpoint, query_id, target_id), (None,))[0] == "unknown_positive":
                    unknown_candidates.append(candidate)
                continue
            rank = qt["ranking"][:10].index(target_id) if target_id in [
                str(value) for value in qt["ranking"][:10]
            ] else None
            if rank is not None:
                competitor_candidates.append(candidate)

    chosen: list[dict[str, Any]] = []
    for group, items in sorted(buckets.items()):
        if group == "unknown_positive":
            # Handled below, where an actual retained evidence can be attached.
            continue
        rng.shuffle(items)
        for key, evidence_id in items[: args.per_group]:
            chosen.append({"group": group, "endpoint": key[0], "query_id": key[1],
                           "target_id": key[2], "evidence_id": evidence_id})
    rng.shuffle(competitor_candidates)
    for key, evidence_id in competitor_candidates[: args.per_group]:
        chosen.append({"group": "top10_competitor", "endpoint": key[0], "query_id": key[1],
                       "target_id": key[2], "evidence_id": evidence_id})
    rng.shuffle(unknown_candidates)
    for key, evidence_id in unknown_candidates[: args.per_group]:
        chosen.append({"group": "unknown_positive", "endpoint": key[0], "query_id": key[1],
                       "target_id": key[2], "evidence_id": evidence_id})

    # Resolve content for exactly the objects the packets need.
    need = {item["query_id"] for item in chosen}
    need |= {item["target_id"] for item in chosen}
    need |= {item["evidence_id"] for item in chosen if item["evidence_id"]}
    content: dict[str, dict[str, Any]] = {}
    for record in rows(OBJECTS):
        object_id = str(record.get("object_id"))
        if object_id in need:
            content[object_id] = record
            if len(content) == len(need):
                break
    missing = sorted(need - content.keys())

    packets = []
    for item in chosen:
        evidence_id = item["evidence_id"]
        query = content.get(item["query_id"], {})
        target = content.get(item["target_id"], {})
        evidence = content.get(evidence_id, {}) if evidence_id else {}
        modality = evidence.get("object_type")
        packets.append(
            {
                **item,
                "query_columns_rows": query.get("table_parts"),
                "target_columns_rows": target.get("table_parts"),
                "evidence_modality": modality,
                "evidence_text": evidence.get("text"),
                "evidence_image": evidence.get("image"),
            }
        )

    IN.mkdir(parents=True, exist_ok=True)
    out = IN / "VERIFICATION_PACKETS.jsonl"
    digest = hashlib.sha256()
    with out.open("w", encoding="utf-8") as handle:
        for packet in packets:
            line = json.dumps(packet, ensure_ascii=False, sort_keys=True) + "\n"
            handle.write(line)
            digest.update(line.encode())
    stats = defaultdict(int)
    for packet in packets:
        stats[packet["group"]] += 1
        stats[f"{packet['group']}|modality={packet['evidence_modality']}"] += 1
    print(json.dumps({
        "packets": len(packets),
        "groups": dict(stats),
        "missing_objects": len(missing),
        "sha256": digest.hexdigest(),
        "path": str(out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
