"""Verify the frozen hidden-state supplement and repair the six interrupted logical jobs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from mmdd_stage1.features import FeatureStore
from prepare_stage1_r27 import record, rows, sha, write_json
from prepare_stage1_r28 import ROOT, OUT, inputs
from run_stage1_r25 import _r25_teacher_feature_paths


def verify() -> dict:
    directory = OUT / "teacher_hidden_backfill"
    audit = json.loads((OUT / "TEACHER_HIDDEN_PREFLIGHT.json").read_text())
    missing = set(audit["missing_hidden_ids"])
    manifest = list(rows(directory / "teacher_manifest.jsonl"))
    assert {r["object_id"] for r in manifest} == missing and len(manifest) == len(missing)
    store = FeatureStore.from_path(Path(inputs()["feature_manifest"]["path"]).parent,
                                   teacher_paths=[*_r25_teacher_feature_paths(ROOT),directory])
    checks = []
    for r in manifest:
        f = store.get(r["object_id"],include_hidden=True)
        assert f.hidden_states is not None and torch.isfinite(f.hidden_states).all()
        checks.append({"object_id":r["object_id"],"shape":list(f.hidden_states.shape),
                       "feature":record(directory / r["teacher_feature_path"])})
    assert sha(Path(inputs()["feature_manifest"]["path"])) == inputs()["feature_manifest"]["expected_sha256"]
    result = {"status":"completed","manifest":record(directory / "teacher_manifest.jsonl"),
              "checks":checks,"source_audit":record(OUT / "TEACHER_HIDDEN_PREFLIGHT.json"),
              "model_config":record(ROOT / "hf_models/Qwen3-VL-Embedding-8B/config.json"),
              "cache_runner":record(ROOT / "src/backfill_stage1_teacher.py"),
              "source_fingerprints":"verified against original frozen embedding manifest before encoding",
              "base_embedding_manifest_unchanged":True}
    write_json(directory / "BACKFILL_RECEIPT.json",result)
    frozen = json.loads((OUT / "INPUT_HASHES.json").read_text())
    frozen["r28_hidden_supplement"] = result["manifest"]
    frozen["r28_hidden_receipt"] = record(directory / "BACKFILL_RECEIPT.json")
    write_json(OUT / "INPUT_HASHES.json",frozen)
    return {"status":"completed","backfilled":len(checks)}


def resume() -> None:
    assert json.loads((OUT / "teacher_hidden_backfill/BACKFILL_RECEIPT.json").read_text())["status"] == "completed"
    jobs, handles = [], []
    for gpu,seed in enumerate((13,29)):
        for arm in ("T-EDGE-CONT","T-PATH-SPLIT-LSE","T-PATH-SPLIT-COV"):
            job = OUT / "teacher" / arm / f"seed{seed}"
            execution = json.loads((job / "EXECUTION.json").read_text())
            assert execution["status"] == "failed" and execution["exception_type"] == "ValueError"
            try:
                os.kill(execution["pid"],0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError("Original process still exists; do not restart")
            archived = OUT / "failed_attempts/teacher_missing_hidden" / arm / f"seed{seed}"
            archived.parent.mkdir(parents=True,exist_ok=True)
            job.rename(archived)
            command = [sys.executable,str(ROOT/"src/train_stage1_r28.py"),"--arm",arm,"--seed",str(seed),"--device",f"cuda:{gpu}"]
            log = (OUT / "logs" / f"{arm}-seed{seed}-hidden-repaired.log").open("w")
            process = subprocess.Popen(command,cwd="/tmp/mmdd-r28-checks",stdout=log,stderr=subprocess.STDOUT)
            handles.append((process,log))
            jobs.append({"arm":arm,"seed":seed,"pid":process.pid,"status":"running", "archived_attempt":str(archived),
                         "discarded_updates":execution["updates"],"restart_checkpoint":"same epoch0 locked parent",
                         "reason":"full historical evidence graph had missing frozen Teacher hidden tensors; no objective change"})
    while any(j["status"] == "running" for j in jobs):
        for job,(process,log) in zip(jobs,handles):
            code = process.poll()
            if code is not None and job["status"] == "running":
                log.close()
                job.update({"status":"completed" if code == 0 else "failed","returncode":code})
        write_json(OUT / "TEACHER_RECOVERY_LEDGER.json",{"queue_pid":os.getpid(),"jobs":jobs,
                   "scientific_jobs":6,"extra_arms_or_seeds":0,"failed_updates_discarded":sum(j["discarded_updates"] for j in jobs)})
        if any(j["status"] == "running" for j in jobs):
            time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify",action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify()))
    else:
        resume()
