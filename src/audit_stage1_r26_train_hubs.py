"""Fixed train-panel actual QE/ET neighbors and hub counts for existing checkpoints."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import torch

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import RawEmbeddingANNIndices,StudentANNIndices,load_corpus_ids
from prepare_stage1_r26 import ROOT,OUT,file_record,parameter_sha
from run_stage1_r21 import paths,read_rows,write_rows
from run_stage1_r25 import _json,sha256


def hub_summary(values: list[str]) -> dict:
    counts = Counter(values)
    return {"occurrences":len(values),"unique_objects":len(counts),"top1_share":max(counts.values(),default=0)/len(values) if values else 0.,
            "top10_share":sum(n for _,n in counts.most_common(10))/len(values) if values else 0.,"top20":counts.most_common(20)}


@torch.inference_mode()
def run(generators: list[str] | None = None) -> dict:
    torch.set_num_threads(4)
    ps = paths(ROOT)
    store = FeatureStore.from_path(ps["features"],cache_size=300000)
    corpus = load_corpus_ids(ps["corpus"],store)
    panel_path = OUT / "common/column256_queries.jsonl"
    queries = [r["query_id"] for r in read_rows(panel_path)]
    store.preload_embeddings([*(t for values in corpus.values() for t in values),*queries])
    completed = []
    for entry in json.loads((OUT / "MODEL_INVENTORY.json").read_text()):
        name = entry["generator_id"]
        if generators is not None and name not in generators:
            continue
        own_receipt = OUT / "rankings" / name / "RETRIEVAL_RECEIPT.json"
        if not own_receipt.exists():
            continue
        destination = OUT / "statistics/train_hubs" / name
        receipt = destination / "HUB_RECEIPT.json"
        checkpoint = Path(entry["checkpoint"]) if entry["checkpoint"] else None
        model = load_student(checkpoint,torch.device("cpu")).eval() if checkpoint else None
        index_path = OUT / "indexes" / name if model else ROOT / "work/stage1_optimization_r25_final_20260914/common/raw_qwen_index"
        started = time.monotonic()
        indices = (StudentANNIndices(model,store,index_path,device=torch.device("cpu"),checkpoint_sha256=sha256(checkpoint),corpus_sha256=sha256(ps["corpus"])) if model else
                   RawEmbeddingANNIndices(store,index_path,corpus_sha256=sha256(ps["corpus"])))
        for index in indices.indices.values():
            index.set_num_threads(4)
        load_seconds = time.monotonic()-started
        started = time.monotonic()
        qe = {kind:indices.search_many(queries,kind,20) for kind in ("text","image")}
        et = {}
        for kind in ("text","image"):
            evidence = sorted({e for row in qe[kind] for e,_ in row})
            neighbors = indices.search_many(evidence,"table",20)
            et[kind] = dict(zip(evidence,neighbors))
        rows = [{"query_id":q,"generator":name,"QE":{kind:qe[kind][i] for kind in qe},
                 "ET":{kind:{e:et[kind][e] for e,_ in qe[kind][i]} for kind in qe}} for i,q in enumerate(queries)]
        destination.mkdir(parents=True,exist_ok=True)
        write_rows(destination / "neighbors.jsonl.gz",rows)
        summary = {"generator":name,"execution_status":"ran","scientific_validity":"valid","queries":len(queries),
            "panel":file_record(panel_path),"whole_lake_index":file_record(index_path / "manifest.json"),"own_dev_receipt":file_record(own_receipt),
            "parameter_sha":parameter_sha(model) if model else "raw_no_parameters","neighbors":file_record(destination / "neighbors.jsonl.gz"),
            "hub_counts":{kind:{"QE":hub_summary([e for r in qe[kind] for e,_ in r]),
                "ET_per_query_path_occurrence":hub_summary([t for r in qe[kind] for e,_ in r for t,_ in et[kind][e]]),
                "ET_per_unique_evidence":hub_summary([t for r in et[kind].values() for t,_ in r])} for kind in qe},
            "cost":{"device":"cpu","index_load_seconds":load_seconds,"panel_seconds":time.monotonic()-started},
            "semantics":"Actual five-relation Student graph neighbors; no known-positive injection, no dev sampling; ET occurrence weighting and unique-E weighting separate"}
        _json(receipt,summary)
        completed.append(name)
        print(json.dumps({"train_hub_complete":name}),flush=True)
    audit_name = "TRAIN_HUB_AUDIT.json" if generators is None else "TRAIN_HUB_EXTENSION_AUDIT.json"
    _json(OUT / "statistics" / audit_name,{"execution_status":"ran","generators":completed})
    return {"generators":len(completed)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator",action="append")
    print(json.dumps(run(parser.parse_args().generator)))
