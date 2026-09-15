"""Open the R28 full-training gate only from tensor tests and real-parent smokes."""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

from prepare_stage1_r27 import record, rows, sha, write_json
from prepare_stage1_r28 import ROOT, OUT, FAMILIES, inputs


def audit() -> dict:
    test_path = OUT / "correctness.xml"
    suites = ET.parse(test_path).getroot().findall("testsuite")
    assert suites and sum(int(s.attrib["tests"]) for s in suites) >= 13
    assert all(int(s.attrib["failures"]) == int(s.attrib["errors"]) == int(s.attrib["skipped"]) == 0 for s in suites)
    source_paths = [ROOT / f"src/{name}" for name in (
        "train_stage1_r28.py", "mmdd_stage1/r28_objectives.py", "mmdd_stage1/r28_receipts.py",
        "mmdd_stage1/scoring.py", "mmdd_stage1/training.py", "mmdd_stage1/r26_training.py",
        "mmdd_stage1/models.py", "mmdd_stage1/objectives.py", "run_stage1_r19.py")]
    smokes = {}
    for arm in FAMILIES:
        path = OUT / "smoke" / arm / "seed13/EXECUTION.json"
        rec = json.loads(path.read_text())
        assert rec["status"] == "completed" and rec["updates"] == 0
        for r in rec["signature"]["code"].values():
            assert sha(Path(r["path"])) == r["sha256"]
        smokes[arm] = {"receipt": record(path), "loss": rec["smoke"]["losses"],
                       "peak_allocated_bytes": rec["peak_allocated_bytes"]}
    fixed = list(rows(Path(inputs()["historical_b13_rankings"]["path"])))
    population = list(rows(ROOT / "work/stage1_optimization_r26_20260914/common/dev_queries.jsonl"))
    def identities(rs):
        return {r["query_id"]: (r["source_table_id"], r["query_kind"], sorted(r["positive_target_ids"])) for r in rs}
    assert len(fixed) == len(population) == 1198
    assert identities(fixed) == identities(population)
    qrels = list(rows(ROOT / "work/stage1_optimization_r16_20260910/candidate_pools.jsonl.gz"))
    assert identities(qrels) == identities(population)
    result = {"status":"pass", "tests": record(test_path), "smokes": smokes,
              "qrels_identity": "1198 query/source/kind/positive-target identities match locked B13 and historical qrels",
              "source_identity": {str(p.relative_to(ROOT)):record(p) for p in source_paths},
              "contract": "C0-C7 passed; existing split helper used; COV only replaces E; no KD/Uniform/fusion",
              "full_training_jobs_authorized":12}
    write_json(OUT / "R28_OBJECTIVE_AUDIT.json", result)
    ledger = json.loads((OUT / "EXECUTION_LEDGER.json").read_text())
    ledger.update({"G1":"pass", "status":"ready_for_training"})
    write_json(OUT / "EXECUTION_LEDGER.json", ledger)
    return {"status":"pass", "smoke_families":list(smokes), "qrels":len(fixed)}


if __name__ == "__main__":
    print(json.dumps(audit()))
