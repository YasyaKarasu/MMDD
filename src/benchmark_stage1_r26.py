"""Measure real per-query ANN, D1 retention and Equal prequeue without rank caching."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from evaluate_stage1_r26 import retain_evidence
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.r26_metrics import fuse_channels
from mmdd_stage1.retrieval import RawEmbeddingANNIndices,StudentANNIndices,load_corpus_ids,retrieve_zero_one_hop_detailed_many
from mmdd_stage1.row_support import load_evidence_content_keys
from prepare_stage1_r26 import ROOT,OUT,file_record,parameter_sha
from run_stage1_r21 import paths,read_rows,write_rows
from run_stage1_r25 import _json,sha256


@torch.inference_mode()
def run(device_name: str, generators: list[str] | None) -> dict:
    torch.set_num_threads(4)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    ps = paths(ROOT)
    store = FeatureStore.from_path(ps["features"],cache_size=300000)
    corpus = load_corpus_ids(ps["corpus"],store)
    population_path = OUT / "common/dev_queries.jsonl"
    queries = sorted((r["query_id"] for r in read_rows(population_path)),key=lambda q:(hashlib.sha256(q.encode()).hexdigest(),q))[:32]
    started = time.monotonic()
    store.preload_embeddings([*(t for ids in corpus.values() for t in ids),*queries])
    preload_seconds = time.monotonic()-started
    content_keys,_ = load_evidence_content_keys(ROOT / "work/stage1_optimization_r10_20260907/taskB_g5/evidence_content_keys.jsonl")
    completed = []
    for spec in json.loads((OUT / "MODEL_INVENTORY.json").read_text()):
        name = spec["generator_id"]
        if generators is not None and name not in generators:
            continue
        own = OUT / "rankings" / name
        if not (own / "RETRIEVAL_RECEIPT.json").exists():
            continue
        own_receipt = json.loads((own / "RETRIEVAL_RECEIPT.json").read_text())
        if sha256(own / "rankings.jsonl.gz") != own_receipt["rankings"]["sha256"]:
            raise ValueError("Benchmark reference ranks changed")
        reference = {r["query_id"]:r for r in read_rows(own / "rankings.jsonl.gz") if r["query_id"] in queries}
        checkpoint = Path(spec["checkpoint"]) if spec["checkpoint"] else None
        model = load_student(checkpoint,device).eval() if checkpoint else None
        fingerprint = parameter_sha(model) if model else "raw_no_parameters"
        if fingerprint != own_receipt["signature"]["parameter_sha256"]:
            raise ValueError("Benchmark model differs from own evaluation")
        index_receipt = json.loads((own / "INDEX_RECEIPT.json").read_text())
        index_path = Path(index_receipt["index_dir"])
        started = time.monotonic()
        indices = (StudentANNIndices(model,store,index_path,device=device,checkpoint_sha256=sha256(checkpoint),corpus_sha256=sha256(ps["corpus"])) if model else
                   RawEmbeddingANNIndices(store,index_path,corpus_sha256=sha256(ps["corpus"])))
        for index in indices.indices.values():
            index.set_num_threads(4)
        index_load_seconds = time.monotonic()-started
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        rows = []
        for query in queries:
            if model:
                indices.clear_query_cache()
            previous = None
            for mode in ("cold_relation_vectors","warm_relation_vectors"):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.monotonic()
                details = retrieve_zero_one_hop_detailed_many([query],indices,direct_k=100,evidence_k=20,targets_per_evidence=20,
                                                            evidence_aggregation="logsumexp",query_batch_size=1)[0]
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                ann_seconds = time.monotonic()-start
                start = time.monotonic()
                evidence = retain_evidence(query,details,store,content_keys)
                fusion = fuse_channels(details["direct"],evidence)
                post_seconds = time.monotonic()-start
                ranks = {"D":[r["target_id"] for r in details["direct"]],"E":[r["target_id"] for r in evidence],"Equal_BT100":fusion["rankings"]["Equal"][:100]}
                if previous is not None and ranks != previous:
                    raise ValueError("Clearing derived relation vectors changed repeated query rankings")
                previous = ranks
                direct_reference = reference[query]["rankings"]["D100_ANN"]
                rows.append({"query_id":query,"mode":mode,"ann_seconds":ann_seconds,"retention_and_fusion_seconds":post_seconds,
                             "total_seconds":ann_seconds+post_seconds,"rankings":ranks,
                             "direct_membership_overlap_with_batched_evaluation":len(set(ranks["D"]) & set(direct_reference))/len(direct_reference),
                             "U_candidates":len(set(ranks["D"])|set(ranks["E"]))})
        directory = OUT / "statistics/student_latency" / device_name.replace(":","_") / name
        directory.mkdir(parents=True,exist_ok=True)
        write_rows(directory / "queries.jsonl",rows)
        summary = {mode:{field:{"n":32,"p50":float(np.quantile([r[field] for r in rows if r["mode"] == mode],.5)),
                               "p95":float(np.quantile([r[field] for r in rows if r["mode"] == mode],.95))}
                         for field in ("ann_seconds","retention_and_fusion_seconds","total_seconds")}
                   for mode in ("cold_relation_vectors","warm_relation_vectors")}
        _json(directory / "LATENCY_RECEIPT.json",{"execution_status":"ran","scientific_validity":"valid","generator":name,"device":device_name,
              "parameter_sha":fingerprint,"own_retrieval":file_record(own / "RETRIEVAL_RECEIPT.json"),"index":file_record(index_path / "manifest.json"),
              "population":file_record(population_path),"selection":"first32 SHA256(query_id) ascending, label-free","queries":file_record(directory / "queries.jsonl"),
              "latency_seconds":summary,"index_load_seconds":index_load_seconds,"feature_preload_seconds_once":preload_seconds,
              "peak_allocated_bytes":torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              "scope":"Actual ANN+all paths+D1+Equal100, one query at a time; HNSW CPU, P/R on named device. Local frozen embeddings and resident indexes; cold/warm only derived relation vectors. Every mode reruns HNSW and path aggregation; no rank/result cache. Excludes backbone feature extraction, index construction, exact diagnostics, Column and Teacher.",
              "code":file_record(Path(__file__))})
        completed.append(name)
        print(json.dumps({"latency_complete":name,"device":device_name}),flush=True)
    return {"completed":completed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--generator",action="append")
    args = parser.parse_args()
    print(json.dumps(run(args.device,args.generator)))
