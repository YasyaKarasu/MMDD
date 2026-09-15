"""Natural train-fit D/E/U for the label-free Column reference and FB mining."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from evaluate_stage1_r26 import retain_evidence
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.r26_metrics import fuse_channels
from mmdd_stage1.retrieval import RawEmbeddingANNIndices, StudentANNIndices, load_corpus_ids, retrieve_zero_one_hop_detailed_many
from mmdd_stage1.row_support import load_evidence_content_keys
from prepare_stage1_r26 import ROOT, OUT, file_record, parameter_sha, stable_sha
from run_stage1_r21 import paths, read_rows, write_rows
from run_stage1_r25 import _json, sha256


@torch.inference_mode()
def run(generator: str, population_name: str, device_name: str) -> dict:
    torch.set_num_threads(4)
    device = torch.device(device_name)
    ps = paths(ROOT)
    train_path = ROOT / "work/stage1_optimization_r12_20260908/taskA_correctness/supervision/target_lists.train_fit.jsonl"
    all_rows = list(read_rows(train_path))
    train = {r["query_id"]: r for r in all_rows}
    if len(train) != len(all_rows):
        raise ValueError("Training reference must have unique query IDs")
    ids = sorted(train, key=lambda q: (hashlib.sha256(q.encode()).hexdigest(), q))
    if population_name == "column256":
        ids = ids[:256]
    dev = {r["query_id"] for r in read_rows(OUT / "common/dev_queries.jsonl")}
    if dev.intersection(ids):
        raise ValueError("Train-fit/dev overlap")
    population = [{key: train[q][key] for key in ("query_id", "query_kind", "positive_target_ids")} for q in ids]
    population_path = OUT / "common" / (population_name + "_queries.jsonl")
    if population_path.exists() and list(read_rows(population_path)) != population:
        raise ValueError("Frozen training population changed")
    write_rows(population_path, population)
    spec = next(r for r in json.loads((OUT / "MODEL_INVENTORY.json").read_text()) if r["generator_id"] == generator)
    checkpoint = Path(spec["checkpoint"]) if spec["checkpoint"] else None
    model = load_student(checkpoint,device).eval() if checkpoint else None
    index_dir = (ROOT / "work/stage1_optimization_r25_final_20260914/common/raw_qwen_index" if model is None else OUT / "indexes" / generator)
    signature = {"generator_id": generator, "parameter_sha": parameter_sha(model) if model else "raw_no_parameters",
        "checkpoint": file_record(checkpoint) if checkpoint else None, "corpus": file_record(ps["corpus"]),
        "feature_manifest": file_record(ps["features"] / "manifest.jsonl"), "index_manifest": file_record(index_dir / "manifest.json"),
        "train_source": file_record(train_path), "population": file_record(population_path),
        "selection": "SHA256(query_id) ascending; first256 for column256, all for feedback; no label-based selection",
        "protocol": json.loads((OUT / "PROTOCOL.json").read_text())["stage1"],
        "code": {name: sha256(ROOT / "src" / name) for name in ("retrieve_stage1_r26_train.py", "evaluate_stage1_r26.py", "mmdd_stage1/retrieval.py", "mmdd_stage1/row_support.py")}}
    destination = OUT / "train_retrieval" / population_name / generator
    receipt = destination / "RETRIEVAL_RECEIPT.json"
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous["signature"] != signature or sha256(destination / "rankings.jsonl.gz") != previous["rankings"]["sha256"]:
            raise ValueError("Train retrieval identity changed")
        return {"status": "verified_cached", "generator": generator}
    started = time.monotonic()
    store = FeatureStore.from_path(ps["features"],cache_size=300000)
    corpus = load_corpus_ids(ps["corpus"],store)
    if set(ids).intersection(corpus["table"]):
        raise ValueError("Training query objects overlap target lake")
    store.preload_embeddings([*(t for values in corpus.values() for t in values), *ids])
    if model is None:
        indices = RawEmbeddingANNIndices(store,index_dir,corpus_sha256=sha256(ps["corpus"]))
    else:
        indices = StudentANNIndices(model,store,index_dir,device=device,checkpoint_sha256=sha256(checkpoint),corpus_sha256=sha256(ps["corpus"]))
    for index in indices.indices.values():
        index.set_num_threads(4)
    for kind, object_ids in corpus.items():
        if indices.object_ids[kind] != object_ids:
            raise ValueError("Frozen index object list mismatch")
    content_keys,_ = load_evidence_content_keys(ROOT / "work/stage1_optimization_r10_20260907/taskB_g5/evidence_content_keys.jsonl")
    result = []
    for start in range(0,len(ids),16):
        batch = ids[start:start+16]
        if model is not None:
            indices.clear_query_cache()
        details = retrieve_zero_one_hop_detailed_many(batch,indices,direct_k=100,evidence_k=20,targets_per_evidence=20,
                                                     evidence_aggregation="logsumexp",query_batch_size=16)
        for q,detail in zip(batch,details):
            evidence = retain_evidence(q,detail,store,content_keys)
            direct = detail["direct"]
            d = [r["target_id"] for r in direct]
            e = [r["target_id"] for r in evidence]
            union = sorted(set(d)|set(e))
            fusion = fuse_channels(direct,evidence)
            result.append({"query_id":q, "generator_id":generator, "query_kind":train[q]["query_kind"],
                "positive_target_ids":train[q]["positive_target_ids"], "D100_ANN":direct, "E_target_ids":e,
                "E_paths":evidence, "E_pre_retention":detail["evidence"], "U":union,
                "rankings":{"D100_ANN":d,"E_ONLY":e,**fusion["rankings"]},
                "candidate_pool_id":stable_sha({"q":q,"D":d,"E":e}), "positive_injection":False})
        print(json.dumps({"generator":generator,"population":population_name,"queries":len(result),"total":len(ids)}),flush=True)
    destination.mkdir(parents=True,exist_ok=True)
    write_rows(destination / "rankings.jsonl.gz",result)
    summary = {"signature":signature,"execution_status":"ran","scientific_validity":"valid",
        "queries":len(result),"rankings":file_record(destination / "rankings.jsonl.gz"),"elapsed_seconds":time.monotonic()-started,
        "object_counts":{k:len(v) for k,v in corpus.items()},"positive_injection_count":0}
    _json(receipt,summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator",required=True)
    parser.add_argument("--population",choices=("column256","feedback"),required=True)
    parser.add_argument("--device",default="cpu")
    args = parser.parse_args()
    print(json.dumps(run(args.generator,args.population,args.device)))
