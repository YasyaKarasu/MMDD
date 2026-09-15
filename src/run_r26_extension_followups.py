"""Finish fusion, fixed T0 and train neighbors for the conditional C2 models."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from mmdd_stage1.checkpoints import load_student
from prepare_stage1_r26 import ROOT,OUT,file_record,parameter_sha
from run_r26_followups import await_artifact,execute
from run_stage1_r25 import _json,sha256


def register_aliases() -> list[dict]:
    """Reuse only a completed generator with numerically identical P/R tensors."""
    inventory = json.loads((OUT / "MODEL_INVENTORY.json").read_text())
    canonical = {}
    for spec in inventory:
        receipt = OUT / "rankings" / spec["generator_id"] / "RETRIEVAL_RECEIPT.json"
        if receipt.exists():
            data = json.loads(receipt.read_text())
            canonical.setdefault(data["signature"]["parameter_sha256"],spec["generator_id"])
    aliases = []
    for spec in inventory:
        name = spec["generator_id"]
        destination = OUT / "rankings" / name
        if (destination / "RETRIEVAL_RECEIPT.json").exists():
            continue
        checkpoint = Path(spec["checkpoint"])
        fingerprint = parameter_sha(load_student(checkpoint,torch.device("cpu")))
        if fingerprint not in canonical:
            raise ValueError(f"Missing own retrieval for distinct model: {name}")
        source_name = canonical[fingerprint]
        source = OUT / "rankings" / source_name
        receipt = source / "RETRIEVAL_RECEIPT.json"
        data = json.loads(receipt.read_text())
        if sha256(source / "rankings.jsonl.gz") != data["rankings"]["sha256"]:
            raise ValueError(f"Canonical retrieval changed: {source_name}")
        alias = {"generator_id":name,"canonical_generator":source_name,"parameter_sha256":fingerprint,
                 "checkpoint":file_record(checkpoint),"canonical_receipt":file_record(receipt),
                 "rankings":file_record(source / "rankings.jsonl.gz"),
                 "reason":"Identical actual P/R tensors on the same frozen lake and retrieval protocol",
                 "execution_status":"verified_parameter_identical_alias","scientific_validity":"valid"}
        old = destination / "ALIAS_RECEIPT.json"
        if old.exists():
            previous = json.loads(old.read_text())
            if previous["parameter_sha256"] != fingerprint or previous["canonical_generator"] != source_name:
                raise ValueError(f"Existing alias identity differs: {name}")
        else:
            _json(old,alias)
        aliases.append(alias)
    # Downstream aliases explicitly point to canonical outputs; they never
    # relabel cached ranks as independent Teacher measurements.
    for alias in aliases:
        for module,filename in (("teacher","TEACHER_RECEIPT.json"),("fusion","COLUMN_RECEIPT.json"),
                                ("statistics/train_hubs","HUB_RECEIPT.json")):
            receipt = OUT / module / alias["canonical_generator"] / filename
            if not receipt.exists():
                raise ValueError(f"Incomplete canonical downstream result: {receipt}")
            _json(OUT / module / alias["generator_id"] / "ALIAS_RECEIPT.json",
                  {**alias,"module":module,"canonical_module_receipt":file_record(receipt)})
    return aliases


def run() -> dict:
    torch.set_num_threads(2)
    await_artifact(OUT / "EXTENSION_EVALUATION_RECEIPT.json")
    await_artifact(OUT / "TEACHER_BACKFILL_EVALUATION_RECEIPT.json")
    names = json.loads((OUT / "EXTENSION_EVALUATION_RECEIPT.json").read_text())["generators"]
    # The original queues own the shared T0 sqlite writer and initial fusion
    # output. Wait for every original canonical result before starting ours.
    for spec in json.loads((OUT / "MODEL_INVENTORY.json").read_text()):
        name = spec["generator_id"]
        if name in names or not (OUT / "rankings" / name / "RETRIEVAL_RECEIPT.json").exists():
            continue
        for module,filename in (("teacher","TEACHER_RECEIPT.json"),("fusion","COLUMN_RECEIPT.json"),
                                ("statistics/train_hubs","HUB_RECEIPT.json")):
            await_artifact(OUT / module / name / filename)
    arguments = [arg for name in names for arg in ("--generator",name)]
    execute("evaluate_stage1_r26_column.py",["--device","cpu",*arguments],"extension_column.log")
    execute("evaluate_stage1_r26_teacher.py",["--device","cpu",*arguments],"extension_teacher.log")
    execute("audit_stage1_r26_train_hubs.py",arguments,"extension_train_hubs.log")
    aliases = register_aliases()
    execute("reconcile_stage1_r26_evaluations.py",[],"extension_reconciliation.log")
    execute("analyze_stage1_r26.py",[],"statistics_all_complete_endpoints.log")
    execute("analyze_stage1_r26_postprocessing.py",[],"postprocessing_all_complete_endpoints.log")
    execute("report_stage1_r26_recall.py",[],"stage1_requested_k_final.log")
    execute("report_r26_b13_teacher_fusion.py",[],"B13_teacher_fusion_final_report.log")
    result = {"execution_status":"ran","scientific_validity":"valid","generators":names,"aliases":aliases,
              "code":file_record(Path(__file__))}
    _json(OUT / "EXTENSION_FOLLOWUPS_RECEIPT.json",result)
    return {"generators":len(names),"aliases":len(aliases)}


if __name__ == "__main__":
    print(json.dumps(run()))
