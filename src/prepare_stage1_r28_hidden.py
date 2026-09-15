"""Audit every train/evaluation object and resolve missing frozen Teacher tensors."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import torch

from cache_stage1_features import _source_fingerprint
from prepare_stage1_r27 import record, rows, write_json
from prepare_stage1_r28 import ROOT, OUT, inputs, feature_store
from run_stage1_r13 import _merge_witness_metadata
from run_stage1_r21 import write_rows


def run() -> dict:
    torch.set_num_threads(2)
    store = feature_store(True)
    required, queries = set(), set()
    for ex in _merge_witness_metadata(ROOT):
        required.add(ex.query_id)
        queries.add(ex.query_id)
        for c in ex.candidates:
            required.add(c.target_id)
            required.update(c.evidence_ids)
    for r in rows(Path(inputs()["historical_b13_rankings"]["path"])):
        required.add(r["query_id"])
        queries.add(r["query_id"])
        required.update(r["U"])
        for c in r["E_paths"]:
            required.update(c["selected_evidence_ids"])
    missing, row_missing = [], []
    for i,oid in enumerate(sorted(required)):
        f = store.get(oid, include_hidden=True)
        if f.hidden_states is None:
            missing.append(oid)
        if oid in queries and (f.row_embeddings is None or not len(f.row_embeddings)):
            row_missing.append(oid)
        if i % 5000 == 0:
            print(json.dumps({"checked":i,"total":len(required),"missing":len(missing)}),flush=True)
    assert not row_missing, "Missing frozen query row embeddings"
    source = ROOT / "work/stage1_optimization_r10_20260907/stage1_data/stage1_objects.jsonl"
    selected = [r for r in rows(source) if r["object_id"] in set(missing)]
    assert {r["object_id"] for r in selected} == set(missing)
    base = {r["object_id"]: r for r in rows(Path(inputs()["feature_manifest"]["path"])) if r["object_id"] in set(missing)}
    for r in selected:
        assert _source_fingerprint(r) == base[r["object_id"]]["source_fingerprint"], r["object_id"]
    write_rows(OUT / "teacher_hidden_backfill/inputs.jsonl", selected)
    result = {"required_objects":len(required),"query_count":len(queries),"missing_hidden_ids":missing,
              "missing_by_type":dict(Counter(r["object_type"] for r in selected)),"row_missing":row_missing,
              "source":record(source),"inputs":record(OUT / "teacher_hidden_backfill/inputs.jsonl"),
              "source_fingerprints_match":True,"policy":"supplement hidden states only; canonical frozen embeddings untouched"}
    write_json(OUT / "TEACHER_HIDDEN_PREFLIGHT.json", result)
    return {k:v for k,v in result.items() if k != "missing_hidden_ids"}


if __name__ == "__main__":
    print(json.dumps(run()))
