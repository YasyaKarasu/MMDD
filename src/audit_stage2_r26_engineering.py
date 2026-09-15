"""Run the frozen 32 train-side records through actual reader/localizer/generator."""
from __future__ import annotations

from collections import Counter
import json
import hashlib
from pathlib import Path

import torch

from mmdd_stage2.checkpoints import load_candidate_scorer
from mmdd_stage2.data import load_stage2_index, row_values
from mmdd_stage2.pipeline import Stage2Verifier
from mmdd_stage2.r26_generation import R26QwenBackend
from mmdd_stage2.verifier import EvidenceBundle
from mmdd_stage1.features import FeatureStore
from prepare_stage1_r26 import ROOT, OUT, file_record
from run_stage1_r21 import read_rows, write_rows
from run_stage1_r25 import _json, out as r25_out

DATASET = ROOT / "output_mm_joinability_entitables_20000_retry100_rounds5_qwen35_final_survivor_context_gaussian_v9"


def run() -> dict:
    torch.set_num_threads(2)
    directory = OUT / "stage2/engineering"
    directory.mkdir(parents=True, exist_ok=True)
    source = r25_out(ROOT) / "stage2/engineering_records_32.jsonl"
    records = list(read_rows(source))
    train_path = ROOT / "work/stage1_optimization_r12_20260908/taskA_correctness/supervision/target_lists.train_fit.jsonl"
    train_rows = list(read_rows(train_path))
    train_ids = {r["query_id"] for r in train_rows}
    invalid_old = [r["query_id"] for r in records if r["query_id"] not in train_ids]
    if len(records) != 32 or invalid_old:
        store = FeatureStore.from_path(ROOT / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b")
        selected, used_queries = [], set()
        ordered = sorted(train_rows, key=lambda r: hashlib.sha256(r["query_id"].encode()).hexdigest())
        for modality in ("text", "image"):
            count = 0
            for row in ordered:
                if row["query_id"] in used_queries:
                    continue
                for target_id, witnesses in sorted(row.get("positive_evidence_by_target", {}).items()):
                    ids = [e for e in sorted(witnesses) if store.embedding_features(e).object_type == modality][:4]
                    if not ids:
                        continue
                    selected.append({"record_id": f"train_eng_{len(selected):03d}", "query_id": row["query_id"],
                                     "target_id": target_id, "evidence_ids": ids, "evidence_modalities": [modality],
                                     "source": "train_fit_independently_labeled_witness"})
                    used_queries.add(row["query_id"])
                    count += 1
                    break
                if count == 16:
                    break
        records = selected
        _json(directory / "POPULATION_REPAIR.json", {"old_source": file_record(source), "old_non_train_queries": invalid_old,
              "new_source": file_record(train_path), "selection": "SHA256(query_id) ascending; 16 text and 16 image; disjoint train queries; explicit witness labels only"})
    if len(records) != 32 or any(r["query_id"] not in train_ids for r in records):
        raise ValueError("Could not freeze 32 train-only engineering records")
    write_rows(directory / "records.jsonl", records)
    objects = load_stage2_index(DATASET, query_ids={r["query_id"] for r in records}, target_ids={r["target_id"] for r in records},
                                evidence_ids={e for r in records for e in r["evidence_ids"]})
    model_dir = ROOT / "hf_models/Qwen3.5-9B"
    scorer_path = r25_out(ROOT) / "stage2/r25_b13_column_scorer.pt"
    scorer = load_candidate_scorer(scorer_path, torch.device("cpu"), expected_model_dir=model_dir)
    backend = R26QwenBackend(model_dir, device="cuda:1", dtype="bf16")
    scorer.to(backend.device)
    verifier = Stage2Verifier(backend, scorer)
    _json(directory / "INPUTS.json", {"records": file_record(source), "scorer": file_record(scorer_path), "template": backend.template_audit,
         "model_config": file_record(model_dir / "config.json"), "dataset_root": str(DATASET), "query_source": "train_fit", "condition": "Real-crop"})
    output = []
    for i, record in enumerate(records):
        query = objects.queries[record["query_id"]]
        target = objects.targets[record["target_id"]]
        evidence = {e: objects.evidence[e] for e in record["evidence_ids"]}
        bundle = EvidenceBundle(record["target_id"], 0., tuple(record["evidence_ids"]))
        result = {"record_id": record["record_id"], "query_id": record["query_id"], "evidence": []}
        try:
            selection = verifier.score_candidates(query, [bundle], {record["target_id"]: target}, evidence)[0].selection
            result["predicted_attribute"] = selection.column_name
            visible_row = row_values(query, query["rows"][0])
            for eid, original in evidence.items():
                backend.generation_context = {"record_id": record["record_id"], "query_id": record["query_id"]}
                try:
                    localized = backend.localize_evidence(visible_row, attribute_name=selection.column_name, evidence=original)
                    value = backend.generate_value(visible_row, attribute_name=selection.column_name, evidence=localized)
                    entry = {"evidence_id": eid, "evidence_type": localized.evidence_type, "value": value,
                             "status": backend.generation_records[-1]["status"], "span": localized.text, "box": localized.box}
                except (ValueError, RuntimeError, OSError) as exc:
                    entry = {"evidence_id": eid, "status": "failed", "error_type": type(exc).__name__}
                result["evidence"].append(entry)
        except (ValueError, RuntimeError, OSError) as exc:
            result["status"] = "reader_failed"
            result["error_type"] = type(exc).__name__
        output.append(result)
        write_rows(directory / "results.jsonl", output)
        write_rows(directory / "raw_completions.jsonl", backend.generation_records)
        torch.cuda.empty_cache()
        print(json.dumps({"record": i + 1, "total": len(records), "states": dict(Counter(e["status"] for e in result["evidence"]))}), flush=True)
    states = Counter(e["status"] for row in output for e in row["evidence"])
    summary = {"records": len(output), "states": dict(states), "valid_value_modalities": sorted({e["evidence_type"] for row in output for e in row["evidence"] if e["status"] == "valid_value"}),
               "synthetic_correctness": file_record(OUT / "stage2/generation_smoke/RESULT.json"), "grounded_cell_correctness": "unknown_without_independent_cell_truth",
               "execution_status": "ran", "generation_interface_valid": states["valid_value"] > 0,
               "scientific_validity": "engineering_only", "raw_completions": file_record(directory / "raw_completions.jsonl")}
    _json(directory / "SUMMARY.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(run()))
