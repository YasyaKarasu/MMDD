#!/usr/bin/env python
"""Lock the R12 mechanism sample from labels before inspecting new outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from mmdd_dataset.wdc_runtime import iter_dataset_artifact
from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.data import load_target_examples


def sample_queries(dataset_root: Path, targets_path: Path, output: Path, seed: int) -> None:
    if output.exists():
        raise FileExistsError("The mechanism sample is already locked")
    targets = load_target_examples(targets_path, split="dev")
    queries = {str(row["table_id"]): row
               for row in iter_dataset_artifact(dataset_root, "query_tables")}
    asset_types = {str(row["asset_id"]): str(row["asset_type"])
                   for row in iter_dataset_artifact(dataset_root, "bridge_assets")}
    pools: dict[str, list[dict]] = defaultdict(list)
    for example in targets:
        query = queries[example.query_id]
        attributes = sorted(str(row["column_name"]).strip().casefold()
                            for row in query.get("hidden_attributes", []))
        modalities = sorted({asset_types[evidence_id]
                             for ids in (example.positive_evidence_by_target or {}).values()
                             for evidence_id in ids})
        row = {"query_id": example.query_id, "source_table_id": query["source_table_id"],
               "kind": example.query_kind, "attributes": attributes, "modalities": modalities}
        row["stratum"] = json.dumps([example.query_kind, modalities, attributes], sort_keys=True)
        row["hash"] = hashlib.sha256(f"{seed}:{example.query_id}".encode()).hexdigest()
        pools[example.query_kind].append(row)
    selected = []
    groups = set()
    for kind, count in (("implicit", 96), ("explicit", 32)):
        strata: dict[str, list[dict]] = defaultdict(list)
        for row in sorted(pools[kind], key=lambda value: value["hash"]):
            strata[row["stratum"]].append(row)
        order = sorted(strata, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
        chosen = []
        while len(chosen) < count:
            changed = False
            for key in order:
                while strata[key] and strata[key][0]["source_table_id"] in groups:
                    strata[key].pop(0)
                if not strata[key]:
                    continue
                row = strata[key].pop(0)
                chosen.append(row)
                groups.add(row["source_table_id"])
                changed = True
                if len(chosen) == count:
                    break
            if not changed:
                break
        selected.extend(chosen)
    write_json(output, {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
        "source_targets": str(targets_path), "source_targets_sha256": checkpoint_fingerprint(targets_path),
        "sampling": "Fixed hash within exact attribute/modality strata, round-robin strata, unique source groups",
        "system_outputs_used": False, "case_enrichment": False,
        "requested": {"implicit": 96, "explicit": 32},
        "actual": dict(Counter(row["kind"] for row in selected)),
        "source_groups": len(groups), "selected": selected, "pool": dict(pools),
        "reader_input_policy": "This file contains audit-only stratification labels; never send it to the reader",
    })
    print(json.dumps({"output": str(output), "actual": dict(Counter(row["kind"] for row in selected))}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dev-targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    sample_queries(args.dataset_root, args.dev_targets, args.output, args.seed)
