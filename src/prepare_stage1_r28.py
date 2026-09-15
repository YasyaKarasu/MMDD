"""Resolve the preregistered R28 identities and freeze the full graph schedule."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import random

import torch

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.data import load_edge_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.scoring import global_edge_positive_ids, target_positive_mask
from prepare_stage1_r27 import ROOT, record, rows, sha, stable_sha, write_json
from run_stage1_r13 import _merge_witness_metadata
from run_stage1_r19 import load_r19_checkpoint
from run_stage1_r25 import _r25_teacher_feature_paths

OUT = ROOT / "work/stage1_optimization_r28_split_path_20260915"
LOCK = ROOT / "mmdd_r27_review/R28_INPUT_LOCK.json"
FAMILIES = {"T-EDGE-CONT": "EDGE", "T-PATH-SPLIT-LSE": "SPLIT-LSE",
            "T-PATH-SPLIT-COV": "SPLIT-COV", "S-EDGE-LONG": "EDGE",
            "S-PATH-LONG": "SPLIT-LSE", "S-COV-LONG": "SPLIT-COV"}


def inputs() -> dict:
    return json.loads((OUT / "R28_RESOLVED_INPUTS.json").read_text())["inputs"]


def registry() -> dict:
    return global_edge_positive_ids(load_edge_examples(Path(inputs()["edge_positive_registry"]["path"]), split="train"))


def feature_store(teacher: bool) -> FeatureStore:
    teacher_paths = _r25_teacher_feature_paths(ROOT) if teacher else []
    supplement = OUT / "teacher_hidden_backfill"
    receipt_path = supplement / "BACKFILL_RECEIPT.json"
    if teacher and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert receipt["status"] == "completed"
        assert sha(supplement / "teacher_manifest.jsonl") == receipt["manifest"]["sha256"]
        teacher_paths.append(supplement)
    return FeatureStore.from_path(Path(inputs()["feature_manifest"]["path"]).parent,
        cache_size=120000 if not teacher else 24000, cache_bytes=4*1024**3 if teacher else None,
        teacher_paths=teacher_paths)


def prepare() -> dict:
    torch.set_num_threads(2)
    lock = json.loads(LOCK.read_text())
    resolved = {}
    for name, item in lock.items():
        if isinstance(item, dict) and "path" in item and "sha256" in item:
            rec = record(ROOT / item["path"])
            resolved[name] = {**rec, "expected_sha256": item["sha256"],
                              "matches": rec.get("sha256") == item["sha256"]}
    write_json(OUT / "R28_RESOLVED_INPUTS.json", {"lock": record(LOCK), "inputs": resolved})
    if not all(r["matches"] for r in resolved.values()):
        raise ValueError("R28 input identity mismatch; see R28_RESOLVED_INPUTS.json")
    # Extra dependencies are pinned before any updates, including actual hidden-feature manifests.
    additional = {
        "historical_order": ROOT / "work/stage1_optimization_r13_20260909/taskD_witness_supervision/schedule_order.json",
        "population": ROOT / "work/stage1_optimization_r26_20260914/common/dev_queries.jsonl",
        "corpus": ROOT / "work/stage1_optimization_r10_20260907/stage1_data/stage1_corpus.jsonl",
        "objects": ROOT / "work/stage1_optimization_r10_20260907/stage1_data/stage1_objects.jsonl",
        "qrels_source": ROOT / "work/stage1_optimization_r16_20260910/candidate_pools.jsonl.gz",
        "teacher_lineage_runner": ROOT / "src/run_stage1_r22_f1.py",
        "teacher_lineage_receipt": ROOT / "work/stage1_optimization_r22_20260911/fresh_lineage/T1-B/seed13/config.json",
    }
    additional.update({f"teacher_manifest_{i}": p / "teacher_manifest.jsonl"
                       for i, p in enumerate(_r25_teacher_feature_paths(ROOT))})
    frozen = {**resolved, **{k: record(p) for k, p in additional.items()}}
    if any(not r["exists"] for r in frozen.values()):
        raise ValueError("Missing additional input dependency")
    write_json(OUT / "INPUT_HASHES.json", frozen)
    examples = _merge_witness_metadata(ROOT)
    known = registry()
    violations = []
    for example in examples:
        present = set(c.target_id for c in example.candidates) & known.get((example.query_id, "table", "table"), set())
        masks = {channel: target_positive_mask([example], len(example.candidates), torch.device("cpu"), channel=channel)[0]
                 for channel in ("direct", "evidence")}
        for j, c in enumerate(example.candidates):
            for channel, mask in masks.items():
                if c.target_id in present and (channel == "direct" or c.evidence_ids) and not mask[j]:
                    violations.append({"query_id": example.query_id, "target_id": c.target_id, "channel": channel})
    write_json(OUT / "TARGET_CLOSURE.json", {"queries": len(examples), "violations": violations})
    if violations:
        raise ValueError("Present train-known positive masked negative")
    original_order = json.loads(additional["historical_order"].read_text())["indices"]
    assert sorted(original_order) == list(range(len(examples)))
    orders = {}
    for seed in lock["seeds"]:
        orders[str(seed)] = []
        for epoch in range(1, 6):
            order = list(original_order)
            if (seed, epoch) != (13, 1):
                random.Random(f"r28-query-order:{seed}:{epoch}").shuffle(order)
            orders[str(seed)].append(order)
    write_json(OUT / "common/orders.json", orders)
    teacher = load_r19_checkpoint(Path(resolved["teacher_parent"]["path"]), torch.device("cpu"))[3]
    student = load_student(Path(resolved["student_parent"]["path"]), torch.device("cpu"))
    from run_stage1_r22_f1 import TEACHER_LR, TEACHER_WD
    parent_payload = torch.load(resolved["teacher_parent"]["path"], map_location="cpu", weights_only=True)
    parent_groups = [{k: v for k, v in g.items() if k != "params"}
                     for g in parent_payload.get("optimizer_state_dict", {}).get("param_groups", [])]
    if parent_groups:
        assert all(g["lr"] == TEACHER_LR and g["weight_decay"] == TEACHER_WD for g in parent_groups)
    manifest = {"version": lock["version"], "graph_examples": len(examples),
        "bag_policy": "full historical bag, no truncation", "path_count": sum(len(c.evidence_ids) for e in examples for c in e.candidates),
        "max_bag": max(len(c.evidence_ids) for e in examples for c in e.candidates),
        "orders": record(OUT / "common/orders.json"), "order_hashes": {s: [stable_sha(o) for o in os] for s, os in orders.items()},
        "teacher": {"batch": 8, "updates_per_epoch": math.ceil(len(examples)/8), "total_updates": 5*math.ceil(len(examples)/8),
                    "lr": TEACHER_LR, "weight_decay": TEACHER_WD, "historical_optimizer_groups": parent_groups,
                    "lr_resolution": "actual T1-B lineage uses 5e-5, overriding plan's unverified 1e-5 suggestion"},
        "student": {"batch": 64, "updates_per_epoch": math.ceil(len(examples)/64), "total_updates": 5*math.ceil(len(examples)/64),
                    "relation_lr": 1e-5, "projection_lr": 1e-6, "weight_decay": .01, "anchor": .1,
                    "anchor_boundary": "reset projection stage anchor at C1 to C2, preserve relation identity anchor", "KD": 0,
                    "historical_sanity": "same parent, full graph, seed13 epoch1 order, split SUP and anchor; KD intentionally disabled"},
        "edge_reduction": "query mean of D + 0.5 mean(QE lists) + 0.5 mean(ET lists); inactive lists contribute zero",
        "trainable": {kind: {"names": [n for n,p in m.named_parameters() if p.requires_grad],
                            "names_hash": stable_sha([n for n,p in m.named_parameters() if p.requires_grad]),
                            "count": sum(p.numel() for p in m.parameters() if p.requires_grad)}
                      for kind,m in (("teacher",teacher),("student",student))},
        "seeds": lock["seeds"], "trajectory_epochs": lock["trajectory_epochs"], "arms": FAMILIES,
        "numerics": "float32, no AMP/scheduler/clipping; fresh AdamW; same logical reduction in all arms",
        "status": "prepared_pending_objective_smoke"}
    previous = OUT / "R28_PREPARED_MANIFEST.json"
    if previous.exists() and json.loads(previous.read_text()) != manifest:
        raise ValueError("Prepared manifest differs; inspect before replacing")
    write_json(previous, manifest)
    write_json(OUT / "EXECUTION_LEDGER.json", {"G0": "pass", "G1": "pending", "status": "prepared", "jobs": []})
    return {"G0": "pass", "graph_examples": len(examples), "teacher": manifest["teacher"], "student": manifest["student"]}


if __name__ == "__main__":
    print(json.dumps(prepare()))
