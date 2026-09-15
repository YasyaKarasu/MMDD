"""Replay exact full-lake scores and historical D1 on a fixed three-query panel."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from audit_stage1_r26_rankings import require
from evaluate_stage1_r26 import retain_evidence
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.row_support import load_evidence_content_keys
from prepare_stage1_r26 import OUT, ROOT, file_record, parameter_sha
from run_stage1_r21 import paths, read_rows
from run_stage1_r25 import _json, sha256


@torch.inference_mode()
def run(device_name: str) -> dict:
    torch.set_num_threads(4)
    device = torch.device(device_name)
    destination = OUT / "acceptance/numerical_replay"
    destination.mkdir(parents=True,exist_ok=True)
    population_path = OUT / "common/dev_queries.jsonl"
    panel = list(itertools.islice(read_rows(population_path),3))
    query_ids = [r["query_id"] for r in panel]
    protocol = {"selection":"First three rows of the preexisting frozen query order; identical for every model; no metric-based selection",
                "query_ids":query_ids,"population":file_record(population_path),"score_atol":2e-5,
                "exact_boundary_atol":2e-5,"D1_atol":1e-10,"device":device_name}
    protocol_path = destination / "PROTOCOL.json"
    if protocol_path.exists():
        require(json.loads(protocol_path.read_text()) == protocol, "Replay protocol changed")
    else:
        _json(protocol_path,protocol)
    ps = paths(ROOT)
    store = FeatureStore.from_path(ps["features"],cache_size=30000)
    raw_index = json.loads((OUT / "rankings/Qwen-Raw/INDEX_RECEIPT.json").read_text())
    tables_path = Path(raw_index["index_dir"]) / "table_ids.json"
    table_ids = json.loads(tables_path.read_text())
    require(len(table_ids) == 22886 and not set(table_ids).intersection(query_ids), "Replay lake/self-exclusion changed")
    positions = {t:i for i,t in enumerate(table_ids)}
    store.preload_embeddings([*table_ids,*query_ids])
    embeddings = torch.stack([store.embedding_features(t).embedding for t in table_ids]).to(device)
    queries = torch.stack([store.embedding_features(q).embedding for q in query_ids]).to(device)
    content_path = ROOT / "work/stage1_optimization_r10_20260907/taskB_g5/evidence_content_keys.jsonl"
    content_keys,_ = load_evidence_content_keys(content_path)
    results = []
    for spec in json.loads((OUT / "MODEL_INVENTORY.json").read_text()):
        name = spec["generator_id"]
        directory = OUT / "rankings" / name
        receipt_path = directory / "RETRIEVAL_RECEIPT.json"
        if not receipt_path.exists():
            continue
        receipt = json.loads(receipt_path.read_text())
        checkpoint = Path(spec["checkpoint"]) if spec["checkpoint"] else None
        require(receipt["signature"]["checkpoint_sha256"] == (sha256(checkpoint) if checkpoint else None), "Replay checkpoint changed")
        model = load_student(checkpoint,device).eval() if checkpoint else None
        require(receipt["signature"]["parameter_sha256"] == (parameter_sha(model) if model else "raw_no_parameters"), "Replay parameters changed")
        transformed = torch.cat([model.index_vector(block,"table") for block in embeddings.split(4096)]) if model else embeddings
        q = model.relation_query(queries,"table","table") if model else queries
        matrix = (q @ transformed.T).cpu()
        saved = list(itertools.islice(read_rows(directory / "rankings.jsonl.gz"),3))
        require([r["query_id"] for r in saved] == query_ids, "Numerical replay panel differs")
        observed = []
        for row,scores in zip(saved,matrix):
            m = row["M_exact"]
            m_positions = [positions[t] for t in m]
            replay_m = scores[m_positions]
            exact_error = float((replay_m-torch.tensor(row["exact_scores"])).abs().max())
            require(exact_error <= 2e-5,"Saved exact score differs from actual full-lake replay")
            qt_error = max(abs(float(scores[positions[t]])-v) for t,v in row["QT_OVER_U_scores"].items())
            require(qt_error <= 2e-5,"Saved QT(U) differs from actual full-lake replay")
            excluded = scores.clone()
            excluded[m_positions] = -torch.inf
            boundary_gap = float(excluded.max()-replay_m.min())
            require(boundary_gap <= 2e-5,"M contains a non-top full-lake target")
            order_error = float((replay_m[1:]-replay_m[:-1]).max())
            require(order_error <= 2e-5,"M rank differs beyond numeric tie tolerance")
            replay_e = retain_evidence(row["query_id"],{"evidence":row["E_pre_retention"]},store,content_keys)
            require([r["target_id"] for r in replay_e] == row["E_target_ids"], "Actual D1 replay ranking differs")
            coverage_error = 0.
            for before,after in zip(row["E_paths"],replay_e):
                for key in ("selected_evidence_ids","retained_paths","routed_rows"):
                    require(before[key] == after[key], f"Actual D1 replay differs: {key}")
                coverage_error = max(coverage_error,abs(before["evidence_score"]-after["evidence_score"]))
            require(coverage_error <= 1e-10,"D1 coverage replay differs")
            observed.append({"query_id":row["query_id"],"lake_targets_scored":len(table_ids),"M_size":len(m),
                             "max_exact_score_error":exact_error,"max_qt_score_error":qt_error,
                             "excluded_max_minus_M_min":boundary_gap,"max_order_violation":order_error,
                             "D1_targets":len(replay_e),"max_D1_coverage_error":coverage_error})
        result = {"generator":name,"execution_status":"ran","scientific_validity":"valid",
                  "observed":observed,"retrieval_receipt":file_record(receipt_path),"checkpoint":file_record(checkpoint) if checkpoint else None,
                  "code":file_record(Path(__file__)),"protocol":file_record(protocol_path)}
        _json(destination / name / "AUDIT.json",result)
        results.append(result)
        print(json.dumps({"numerical_replay_verified":name,"queries":len(observed)}),flush=True)
        del model,transformed,matrix
    summary = {"execution_status":"ran","scientific_validity":"valid_for_fixed_panel","test_ids":["G02","G04"],
               "models":results,"panel_queries":3,"table_ids":file_record(tables_path),"features_manifest":file_record(ps["features"] / "manifest.jsonl"),
               "content_keys":file_record(content_path),"scope":"Actual checkpoint/full-lake direct numerical replay and actual D1 selection on fixed first3 queries for every canonical model; full-population raw-rank invariant audit is separate."}
    _json(destination / "AUDIT.json",summary)
    return {"models":len(results),"queries":3*len(results)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",default="cuda:1")
    args = parser.parse_args()
    print(json.dumps(run(args.device)))
