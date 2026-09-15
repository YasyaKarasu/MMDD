"""Measure real fixed-T0 C100/C18 scoring with cold/warm compression on a GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from mmdd_stage1.features import FeatureStore
from mmdd_stage1.teacher_rerank import _teacher_scores
from prepare_stage1_r26 import ROOT, OUT, file_record
from prepare_stage1_r26_refinement import T0
from run_stage1_r19 import load_r19_checkpoint
from run_stage1_r21 import paths, read_rows, write_rows
from run_stage1_r25 import _json, _r25_teacher_feature_paths, sha256


@torch.inference_mode()
def run(device_name: str, generators: list[str] | None) -> dict:
    torch.set_num_threads(2)
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    started = time.monotonic()
    _,_,_,teacher,_ = load_r19_checkpoint(T0,device)
    teacher.eval()
    store = FeatureStore.from_path(paths(ROOT)["features"],cache_size=30000,cache_bytes=3*1024**3,
                                   teacher_paths=_r25_teacher_feature_paths(ROOT))
    initialize_seconds = time.monotonic()-started
    population = OUT / "common/dev_queries.jsonl"
    queries = sorted((r["query_id"] for r in read_rows(population)),key=lambda q:(hashlib.sha256(q.encode()).hexdigest(),q))[:32]
    completed = []
    for spec in json.loads((OUT / "MODEL_INVENTORY.json").read_text()):
        name = spec["generator_id"]
        if generators is not None and name not in generators:
            continue
        own_path = OUT / "rankings" / name / "RETRIEVAL_RECEIPT.json"
        if not own_path.exists():
            continue
        own = json.loads(own_path.read_text())
        rank_path = Path(own["rankings"]["path"])
        if sha256(rank_path) != own["rankings"]["sha256"]:
            raise ValueError("Teacher benchmark own ranks changed")
        selected = {r["query_id"]:r for r in read_rows(rank_path) if r["query_id"] in queries}
        if len(selected)!=32:
            raise ValueError("Incomplete Teacher benchmark population")
        teacher_path = OUT / "teacher" / name / "rankings.jsonl.gz"
        reference = {}
        if teacher_path.exists():
            teacher_receipt = json.loads((teacher_path.parent / "TEACHER_RECEIPT.json").read_text())
            if sha256(teacher_path) != teacher_receipt["rankings"]["sha256"]:
                raise ValueError("CPU Teacher reference changed")
            reference = {r["query_id"]:r["teacher_scores"] for r in read_rows(teacher_path) if r["query_id"] in queries}
        rows = []
        torch.cuda.reset_peak_memory_stats(device)
        for q in queries:
            for pool,targets in (("Equal_C100",selected[q]["rankings"]["Equal"][:100]),
                                 ("Direct_C100",selected[q]["rankings"]["D100_ANN"][:100]),
                                 ("Equal_C18",selected[q]["rankings"]["Equal"][:18])):
                # Disk/object loading is measured separately from T0 arithmetic.
                started = time.monotonic()
                for target in [q,*targets]:
                    store.get(target,include_hidden=True)
                feature_seconds = time.monotonic()-started
                cache = teacher.new_compression_cache()
                previous = None
                for mode in ("cold_compression","warm_compression"):
                    torch.cuda.synchronize(device)
                    started = time.monotonic()
                    scores = _teacher_scores(teacher,q,targets,store,device,batch_size=64,compression_cache=cache)
                    torch.cuda.synchronize(device)
                    elapsed = time.monotonic()-started
                    difference = max((abs(v-reference[q][t]) for t,v in zip(targets,scores)),default=0.) if q in reference else None
                    if difference is not None and difference > .001:
                        raise ValueError("GPU T0 scores differ from fixed CPU T0 beyond preregistered.001 tolerance")
                    repeat_difference = max((abs(a-b) for a,b in zip(previous,scores)),default=0.) if previous is not None else None
                    if repeat_difference is not None and repeat_difference > .001:
                        raise ValueError("Compression reuse changes the Teacher score")
                    rows.append({"query_id":q,"pool":pool,"mode":mode,"target_ids":targets,"scores":scores,
                                 "pairs":len(targets),"seconds":elapsed,"local_feature_load_seconds":feature_seconds,
                                 "max_difference_vs_CPU":difference,"max_difference_vs_cold":repeat_difference})
                    previous = scores
        destination = OUT / "statistics/teacher_latency" / device_name.replace(":","_") / name
        destination.mkdir(parents=True,exist_ok=True)
        write_rows(destination / "queries.jsonl",rows)
        latency = {}
        for pool in ("Equal_C100","Direct_C100","Equal_C18"):
            for mode in ("cold_compression","warm_compression"):
                values = [r["seconds"] for r in rows if r["pool"] == pool and r["mode"] == mode]
                latency[pool+"/"+mode] = {"n":len(values),"p50":float(np.quantile(values,.5)),"p95":float(np.quantile(values,.95))}
        _json(destination / "LATENCY_RECEIPT.json",{"execution_status":"ran","scientific_validity":"valid",
              "generator":name,"device":device_name,"teacher":file_record(T0),"own_retrieval":file_record(own_path),
              "population":file_record(population),"query_selection":"first32 SHA256(query_id) ascending, no label selection",
              "latency_seconds":latency,"initialization_seconds_once":initialize_seconds,
              "peak_allocated_bytes":torch.cuda.max_memory_allocated(device),"queries":file_record(destination / "queries.jsonl"),
              "scope":"Actual per-query QT scoring, no pair-score/query-result cache. Local frozen features; cold/warm Teacher compression only. Equal18 is the actual Stage2 T0 prequeue budget. Excludes backbone, Student and Stage2 generation. Measurements may share GPU with small R26 jobs.",
              "code":file_record(Path(__file__))})
        completed.append(name)
        print(json.dumps({"teacher_latency_complete":name}),flush=True)
    return {"completed":completed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",default="cuda:1")
    parser.add_argument("--generator",action="append")
    args = parser.parse_args()
    print(json.dumps(run(args.device,args.generator)))
