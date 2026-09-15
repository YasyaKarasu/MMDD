"""Restore the five missing full-lake table hidden states from frozen visible inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch

from cache_stage1_features import _source_fingerprint
from mmdd_stage1.features import FeatureStore
from prepare_stage1_r26 import ROOT,OUT,file_record
from run_r26_followups import await_artifact,execute
from run_stage1_r21 import read_rows,write_rows
from run_stage1_r25 import _json,_r25_teacher_feature_paths


def run(device_name: str) -> dict:
    audit_path = OUT / "acceptance/TEACHER_MISSING_TABLE_HIDDEN.json"
    missing = set(json.loads(audit_path.read_text())["missing"])
    base = ROOT / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b"
    source = base.parent / "stage1_data/stage1_objects.jsonl"
    records = {}
    for row in read_rows(source):
        if row["object_id"] in missing:
            records[row["object_id"]] = row
        if records.keys() == missing:
            break
    if records.keys() != missing:
        raise ValueError("Missing frozen source records for Teacher supplement")
    manifest = {r["object_id"]:r for r in read_rows(base / "manifest.jsonl") if r["object_id"] in missing}
    directory = OUT / "teacher_backfill"
    shadow = directory / "base_reference"
    for oid,entry in manifest.items():
        if _source_fingerprint(records[oid]) != entry["source_fingerprint"]:
            raise ValueError("Teacher input differs from actual frozen retrieval source")
        target = shadow / entry["feature_path"]
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(base / entry["feature_path"],target)
    write_rows(shadow / "manifest.jsonl",[manifest[q] for q in sorted(missing)])
    write_rows(directory / "inputs.jsonl",[records[q] for q in sorted(missing)])
    write_rows(directory / "required_ids.jsonl",[{"query_id":min(missing),"candidate_ids":sorted(missing),"split":"train"}])
    _json(directory / "INPUT_AUDIT.json",{"missing_audit":file_record(audit_path),"source":file_record(source),
          "inputs":file_record(directory / "inputs.jsonl"),"base_manifest":file_record(base / "manifest.jsonl"),
          "verified_source_fingerprints":{q:manifest[q]["source_fingerprint"] for q in sorted(missing)},
          "semantics":"Frozen visible object-only target-table inputs; no query, qrels or new task label enters the embedding model"})
    # Generation's image reader can occupy22GB. The five table encodings can
    # share GPU1 with the small C2 trainers once that image workload is finished.
    if device_name == "cuda:1":
        await_artifact(OUT / "stage2/pilot/B13/PILOT_RECEIPT.json")
    execute("cache_stage1_features.py",["--input-jsonl",str(directory / "inputs.jsonl"),"--output-dir",str(shadow),
            "--teacher-output-dir",str(directory),"--teacher-data",str(directory / "required_ids.jsonl"),"--teacher-split","all",
            "--model-dir",str(ROOT / "hf_models/Qwen3-VL-Embedding-8B"),"--device",device_name,"--dtype","bf16",
            "--table-tokens-per-group","1"],"teacher_hidden_backfill.log")
    store = FeatureStore.from_path(base,teacher_paths=[*_r25_teacher_feature_paths(ROOT),directory])
    checks = []
    for oid in sorted(missing):
        features = store.get(oid,include_hidden=True)
        if features.hidden_states is None or not torch.isfinite(features.hidden_states).all():
            raise ValueError("Teacher supplement did not produce finite actual hidden states")
        checks.append({"object_id":oid,"hidden_shape":list(features.hidden_states.shape),
                       "groups":None if features.token_groups is None else features.token_groups.tolist()})
    result = {"execution_status":"ran","scientific_validity":"valid","cache_runner_device":device_name,"checks":checks,"manifest":file_record(directory / "teacher_manifest.jsonl"),
              "device_scope":"Requested device for this cache invocation; a cache hit performs no encoding. See ENCODING_AUDIT.json for the original generation device.",
              "inputs":file_record(directory / "INPUT_AUDIT.json"),"code":file_record(Path(__file__)),
              "base_features_untouched":"Only five small reference copies were used by the cache writer; canonical retrieval cache was not opened for writing"}
    _json(directory / "BACKFILL_RECEIPT.json",result)
    execute("evaluate_stage1_r26_teacher.py",["--device","cpu"],"teacher_all_cpu_after_backfill.log")
    _json(OUT / "TEACHER_BACKFILL_EVALUATION_RECEIPT.json",{"execution_status":"ran","backfill":file_record(directory / "BACKFILL_RECEIPT.json"),
          "evaluation_log":file_record(OUT / "logs/teacher_all_cpu_after_backfill.log")})
    execute("report_stage1_r26_recall.py",[],"stage1_requested_k_after_teacher_backfill.log")
    return {"backfilled":len(checks)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",default="cuda:1")
    print(json.dumps(run(parser.parse_args().device)))
