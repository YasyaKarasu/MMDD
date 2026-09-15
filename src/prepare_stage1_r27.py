"""Resolve R27-rev2 historical identities and audit the actual positive masks."""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import torch

from mmdd_stage1.data import load_edge_examples
from mmdd_stage1.scoring import edge_positive_key, edge_positive_mask, global_edge_positive_ids, target_positive_mask
from run_stage1_r12_task_c import _examples, _score_payload
from run_stage1_r13 import _merge_witness_metadata

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work/stage1_diagnostics_r27_20260915_rev2_b13_exact"
R12 = ROOT / "work/stage1_optimization_r12_20260908"
R13 = ROOT / "work/stage1_optimization_r13_20260909"
R26 = ROOT / "work/stage1_optimization_r26_20260914"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def rows(path: Path):
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as handle:
        for line in handle:
            yield json.loads(line)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def record(path: Path) -> dict:
    return {"path": str(path), "exists": path.is_file(), **({"bytes": path.stat().st_size, "sha256": sha(path)} if path.is_file() else {})}


def prepare() -> dict:
    torch.set_num_threads(2)
    hist = OUT / "historical_replay"
    lock = read_json(ROOT / "mmdd_r26_review/R27_HISTORICAL_REPLAY_INPUT_LOCK.json")
    schedule = read_json(R12 / "taskC_training/candidates_seed13_steps356/manifest.json")
    teacher = read_json(R12 / "taskC_training/teacher_pair_scores/manifest.json")
    c1 = read_json(R12 / "taskC_training/c_candidates_seed13/manifest.json")
    c2 = read_json(R13 / "taskD_witness_supervision/schedule_manifest.json")
    expected = {
        "historical_c1_schedule": schedule["arms"]["candidates"]["schedule_sha256"],
        "historical_c1_schedule_manifest": teacher["candidate_manifest_sha256"],
        "historical_c1_pair_ids": teacher["pair_manifest_sha256"],
        "historical_c1_teacher_scores": teacher["scores_sha256"],
        "historical_c1_reference": c1["checkpoints"]["356"]["checkpoint_sha256"],
        "historical_c2_full_graph": c2["path_hard"]["sha256"],
        "historical_c2_graph_metadata": c2["path_hard"]["metadata_sha256"],
        "historical_c2_witness_metadata": c2["witness_source"]["sha256"],
        "historical_c2_order": c2["order_sha256"],
    }
    resolved = []
    for item in lock["replay_inputs"]:
        name = item["logical_id"]
        path = ROOT / (teacher["scores"] if name == "historical_c1_teacher_scores" else item["path_hint"])
        rec = {**item, **record(path)}
        rec["expected_sha256"] = expected.get(name, item["expected_sha256"])
        rec["identity_status"] = ("missing" if not rec["exists"] else "verified_historical_hash" if rec["expected_sha256"] == rec["sha256"] else "hash_mismatch" if rec["expected_sha256"] else "historical_manifest_provenance_no_independent_hash")
        rec["schema_summary"] = path.suffix
        rec["source_stage"] = "R12/R13 historical or R26 recovery"
        rec["locally_recheckable"] = rec["exists"]
        resolved.append(rec)
    write_json(hist / "H_RESOLVED_INPUTS.json", resolved)
    write_json(hist / "frozen_input_hashes.json", {r["logical_id"]: r.get("sha256") for r in resolved})
    write_json(hist / "R27_HISTORICAL_REPLAY_INPUT_LOCK.json", lock)
    bad = [r for r in resolved if r["required_for_training"] and r["identity_status"] in ("missing", "hash_mismatch")]
    if bad:
        write_json(hist / "H_EXECUTION_LEDGER.json", {"GH0": "blocked_missing_historical_input", "inputs": bad})
        return {"GH0": "blocked", "bad": bad}
    # Check the mask used by score_edge_batch, not just the schedule's ANN provenance.
    known = global_edge_positive_ids(load_edge_examples(R12 / "taskA_correctness/supervision/edge_lists.train_fit.jsonl", split="train"))
    scores, teacher_manifest = _score_payload(R12)
    counts = Counter()
    pollution = []
    consumed = []
    pair_checks = {}
    feature_ids = set()
    for batch in rows(R12 / "taskC_training/candidates_seed13_steps356/candidates.jsonl.gz"):
        examples = _examples(batch["examples"], scores, teacher_manifest["teacher_checkpoint_sha256"])
        mask = edge_positive_mask(examples, max(len(e.candidate_ids) for e in examples), torch.device("cpu"))
        counts["batches"] += 1
        consumed.append({"step": batch["step"], "batch_sha256": stable_sha(batch), "query_ids": [e.query_id for e in examples]})
        for i, (e, raw) in enumerate(zip(examples, batch["examples"])):
            present = set(e.candidate_ids) & known[edge_positive_key(e)]
            actual = {t for j, t in enumerate(e.candidate_ids) if bool(mask[i, j])}
            missing = present - actual
            contradictions = [t for t, label in zip(e.candidate_ids, e.confirmed_labels) if t in present and label == 0]
            counts["lists"] += 1
            counts["present_positives"] += len(present)
            counts["absent_positives"] += len(known[edge_positive_key(e)] - set(e.candidate_ids))
            counts[f"relation/{e.source_type}_to_{e.destination_type}"] += 1
            if missing or contradictions:
                pollution.append({"stage": "C1", "step": batch["step"], "row": i, "query_id": e.query_id, "missing_mask_ids": sorted(missing), "confirmed_negative_ids": contradictions})
            feature_ids.update((e.query_id, *e.candidate_ids))
            for pid, target in zip(raw["candidate_pair_ids"], e.candidate_ids):
                value = (e.query_id, target, e.source_type, e.destination_type)
                if pid in pair_checks:
                    assert pair_checks[pid] == value
                pair_checks[pid] = value
    assert counts["batches"] == 356 and counts["lists"] == 22784
    for pair in rows(R12 / "taskC_training/candidates_seed13_steps356/teacher_pairs.jsonl.gz"):
        pid = pair["pair_id"]
        if pid in pair_checks:
            assert pair_checks.pop(pid) == tuple(pair[k] for k in ("source_id", "destination_id", "source_type", "destination_type"))
    assert not pair_checks
    write_json(hist / "C1_schedule_audit.json", {"counts": counts, "batches": consumed, "pair_mapping": "all_consumed_pairs_verified", "teacher": record(Path(teacher["scores"]))})
    examples = _merge_witness_metadata(ROOT)
    target_known = defaultdict(set)
    for row in rows(R12 / "taskA_correctness/supervision/target_lists.train_fit.jsonl"):
        target_known[row["query_id"]].update(row["positive_target_ids"])
    c2_counts = Counter()
    for e in examples:
        feature_ids.add(e.query_id)
        present = target_known[e.query_id] & {c.target_id for c in e.candidates}
        for channel in ("direct", "evidence"):
            mask = target_positive_mask([e], len(e.candidates), torch.device("cpu"), channel=channel)
            actual = {c.target_id for j, c in enumerate(e.candidates) if bool(mask[0, j])}
            if present - actual:
                pollution.append({"stage": "C2", "query_id": e.query_id, "channel": channel, "missing_mask_ids": sorted(present - actual)})
        for c in e.candidates:
            feature_ids.update((c.target_id, *c.evidence_ids))
            c2_counts["targets"] += 1
            c2_counts["paths"] += len(c.evidence_ids)
            c2_counts["targets_over_8"] += len(c.evidence_ids) > 8
            c2_counts["paths_beyond_8"] += max(0, len(c.evidence_ids)-8)
            c2_counts["duplicate_path_ids"] += len(c.evidence_ids)-len(set(c.evidence_ids))
    order = read_json(R13 / "taskD_witness_supervision/schedule_order.json")["indices"]
    assert len(examples) == 11390 and sorted(order) == list(range(11390))
    write_json(hist / "present_positive_closure_audit.json", {"status": "pass" if not pollution else "failed_correctness_positive_pollution", "C1": counts, "C2": dict(c2_counts, examples=len(examples)), "pollution": pollution, "actual_mask_functions": ["edge_positive_mask", "target_positive_mask"], "train_only": True, "absent_positives_inserted": False})
    features = ROOT / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b"
    feature_records = []
    for row in rows(features / "manifest.jsonl"):
        if row["object_id"] in feature_ids:
            feature_records.append({"object_id": row["object_id"], "source_fingerprint": row.get("source_fingerprint"), **record(features / row["feature_path"])})
    assert {r["object_id"] for r in feature_records} == feature_ids and all(r["exists"] for r in feature_records)
    with gzip.open(hist / "consumed_feature_files.jsonl.gz", "wt") as handle:
        for rec in feature_records:
            handle.write(json.dumps(rec) + "\n")
    write_json(hist / "H_EXECUTION_LEDGER.json", {"GH0": "pass_with_historical_runtime_unknown", "GH1": "pass" if not pollution else "failed_correctness_positive_pollution", "GH2": "pending", "H1": {"planned": True, "executed": False}, "H2": {"planned": True, "executed": False}, "historical_feature_bytes": "current files individually hashed; historical manifest locks object paths/source fingerprints, not historical tensor bytes"})
    for path in (ROOT / "src").rglob("*.py"):
        dest = OUT / "source_snapshot" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    write_json(OUT / "PROTOCOL_LOCK.json", {"version": lock["version"], "plans": [record(ROOT / "mmdd_r26_review" / name) for name in ("R27_CODEX_PROMPT.md", "R27_EXPERIMENT_PLAN.md", "R27_REVISION_NOTES.md")], "numeric_atol": 1e-6, "numeric_rtol": 1e-5, "float64_aggregation_atol": 1e-8, "gpu_spotcheck_atol": 1e-5})
    return {"GH0": "pass_with_historical_runtime_unknown", "GH1": "pass" if not pollution else "failed", "features": len(feature_records), "C1": counts, "C2": c2_counts}


if __name__ == "__main__":
    print(json.dumps(prepare()))
