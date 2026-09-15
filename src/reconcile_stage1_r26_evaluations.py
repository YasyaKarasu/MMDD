"""Resolve queue-era missing/failed entries using verified completed retrieval artifacts."""
from __future__ import annotations

import json
from pathlib import Path

from prepare_stage1_r26 import ROOT,OUT,file_record
from run_stage1_r25 import _json,sha256


def run() -> dict:
    results = []
    for spec in json.loads((OUT / "MODEL_INVENTORY.json").read_text()):
        directory = OUT / "rankings" / spec["generator_id"]
        receipt = directory / "RETRIEVAL_RECEIPT.json"
        alias = directory / "ALIAS_RECEIPT.json"
        if receipt.exists():
            data = json.loads(receipt.read_text())
            checkpoint_sha = sha256(Path(spec["checkpoint"])) if spec["checkpoint"] else None
            if checkpoint_sha != data["signature"]["checkpoint_sha256"]:
                raise ValueError("Current checkpoint differs from evaluated checkpoint")
            if sha256(directory / "rankings.jsonl.gz") != data["rankings"]["sha256"]:
                raise ValueError("Raw retrieval ranks changed")
            results.append({"generator_id":spec["generator_id"],"execution_status":"ran","scientific_validity":"valid","receipt":file_record(receipt)})
        elif alias.exists():
            data = json.loads(alias.read_text())
            canonical = json.loads(Path(data["canonical_receipt"]["path"]).read_text())
            if data["parameter_sha256"] != canonical["signature"]["parameter_sha256"]:
                raise ValueError("Alias differs from executed parameters")
            if sha256(Path(data["rankings"]["path"])) != data["rankings"]["sha256"]:
                raise ValueError("Alias ranks changed")
            results.append({"generator_id":spec["generator_id"],"execution_status":"verified_parameter_identical_alias","scientific_validity":"valid","receipt":file_record(alias)})
        else:
            results.append({"generator_id":spec["generator_id"],"execution_status":"pending","scientific_validity":"unassessable"})
    summary = {"models":len(results),"valid":sum(r["scientific_validity"] == "valid" for r in results),"results":results,
               "original_queue":file_record(OUT / "EVALUATION_QUEUE_RESULT.json"),
               "resolved_events":["Historical B13 recovered byte-exact, built new own indexes and evaluated on GPU0",
                                  "pre-B13 C1 recovered from exact B13 step0 and evaluated",
                                  "R25 SPLIT-UQTKD seed13 initial GPU1 OOM; actual evaluation retried on GPU0"]}
    _json(OUT / "EVALUATION_RECONCILIATION.json",summary)
    return {"models":summary["models"],"valid":summary["valid"],"pending":[r["generator_id"] for r in results if r["scientific_validity"] != "valid"]}


if __name__ == "__main__":
    print(json.dumps(run()))
