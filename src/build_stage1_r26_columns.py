"""Audit/reuse frozen Qwen columns and encode missing R26 natural-pool inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from build_r25_column_repr import _column_text
from cache_stage1_features import _load_embedder_class, encode_inputs
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import load_corpus_ids
from mmdd_stage2.data import load_stage2_index
from prepare_stage1_r26 import ROOT, OUT, file_record, stable_sha
from run_stage1_r21 import paths, read_rows, write_rows
from run_stage1_r25 import _json

DATASET = ROOT / "output_mm_joinability_entitables_20000_retry100_rounds5_qwen35_final_survivor_context_gaussian_v9"
INSTRUCTION = "Represent this visible table column for joinability comparison. Preserve the column name and visible cell values; do not infer hidden attributes."


def run(prepare_only: bool, device_name: str, batch_size: int) -> dict:
    torch.set_num_threads(4)
    destination = OUT / "fusion/columns"
    destination.mkdir(parents=True,exist_ok=True)
    cache_paths = [ROOT / f"work/stage1_optimization_r25_final_20260914/fusion/column_repr_gpu{s}.pt" for s in (0,1)]
    vectors = {}
    for path in cache_paths:
        payload = torch.load(path,map_location="cpu",weights_only=False)
        if payload["instruction"] != INSTRUCTION or payload["max_values"] != 5 or Path(payload["model_dir"]) != ROOT / "hf_models/Qwen3-VL-Embedding-8B":
            raise ValueError("Cached column encoding recipe changed")
        for key,value in payload["vectors"].items():
            if key in vectors and not torch.equal(value,vectors[key]):
                raise ValueError("Conflicting cached column tensors")
            if value.shape != (4096,) or not torch.isfinite(value).all() or abs(float(value.float().norm())-1) > .001:
                raise ValueError("Invalid frozen column vector")
            vectors[key] = value
    query_ids = {r["query_id"] for name in ("dev_queries.jsonl","column256_queries.jsonl") for r in read_rows(OUT / "common" / name)}
    store = FeatureStore.from_path(paths(ROOT)["features"])
    target_ids = set(load_corpus_ids(paths(ROOT)["corpus"],store)["table"])
    objects = load_stage2_index(DATASET,query_ids=query_ids,target_ids=target_ids,evidence_ids=set())
    inputs = []
    for role,tables in (("query",objects.queries),("target",objects.targets)):
        for table_id,table in sorted(tables.items()):
            for column in table["columns"]:
                key = f"{role}:{table_id}:{column['column_index']}"
                text = _column_text(table,column,5)
                inputs.append({"key":key,"role":role,"table_id":table_id,"column_index":column["column_index"],
                    "column_name":column.get("column_name",""),"text":text,"text_sha":hashlib.sha256(text.encode()).hexdigest(),
                    "cache_present":key in vectors})
    expected = {r["key"] for r in inputs}
    # Hash selection precedes similarity checks, independently of labels/scores.
    probes = sorted(expected & vectors.keys(),key=lambda k:hashlib.sha256(k.encode()).hexdigest())[:32]
    pending = [r for r in inputs if r["key"] not in vectors or r["key"] in probes]
    input_path = destination / "visible_column_inputs.jsonl.gz"
    if input_path.exists():
        if list(read_rows(input_path)) != inputs:
            raise ValueError("Frozen visible column inputs changed")
    else:
        write_rows(input_path,inputs)
    signature = {"sources":[file_record(p) for p in cache_paths],"visible_inputs":file_record(destination / "visible_column_inputs.jsonl.gz"),
        "instruction":INSTRUCTION,"max_values":5,"model_config":file_record(ROOT / "hf_models/Qwen3-VL-Embedding-8B/config.json"),
        "encoding_source":file_record(ROOT / "src/cache_stage1_features.py"),"builder_source":file_record(Path(__file__))}
    summary = {"signature":signature,"queries":len(query_ids),"targets":len(target_ids),"expected_columns":len(expected),
        "reused_columns":len(expected & vectors.keys()),"missing_columns":len(expected-vectors.keys()),"probe_keys":probes,
        "identity_scope":"Every selected column ID, current visible text and vector norm checked; 32 hash-fixed reused vectors require numerical reencoding",
        "type_policy":"No reliable CTA; max cosine across all visible columns; generic-column risk reported downstream"}
    _json(destination / "PREPARATION.json",summary)
    if prepare_only:
        return summary
    supplements = {}
    partial = destination / "supplement.partial.pt"
    if partial.exists():
        payload = torch.load(partial,map_location="cpu",weights_only=False)
        if payload["signature"] != stable_sha(signature):
            raise ValueError("Column supplement identity changed")
        supplements = payload["vectors"]
    missing = [r for r in pending if r["key"] not in supplements]
    model_dir = ROOT / "hf_models/Qwen3-VL-Embedding-8B"
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    embedder = _load_embedder_class(model_dir)(model_name_or_path=str(model_dir),torch_dtype=torch.bfloat16)
    embedder.model.to(device).eval()
    started = time.monotonic()
    for start in range(0,len(missing),batch_size):
        batch = missing[start:start+batch_size]
        encoded = encode_inputs(embedder,[{"text":r["text"],"instruction":INSTRUCTION} for r in batch],include_hidden=False)
        for row,(embedding,_,_) in zip(batch,encoded,strict=True):
            supplements[row["key"]] = F.normalize(embedding.float(),dim=0).half().cpu()
        if start % (batch_size*20) == 0:
            torch.save({"signature":stable_sha(signature),"vectors":supplements},partial)
            print(json.dumps({"encoded":min(start+batch_size,len(missing)),"pending":len(missing)}),flush=True)
    probe_results = [{"key":key,"cosine":float(F.cosine_similarity(vectors[key].float(),supplements[key].float(),dim=0)),
                      "max_abs_difference":float((vectors[key].float()-supplements[key].float()).abs().max())} for key in probes]
    _json(destination / "REENCODING_PROBES.json",probe_results)
    if any(r["cosine"] < .999 for r in probe_results):
        raise ValueError("Cached column reencoding differs; investigate before reuse")
    vectors = {key:value for key,value in vectors.items() if key in expected}
    vectors.update({key:value for key,value in supplements.items() if key not in vectors})
    if vectors.keys() != expected:
        raise ValueError("Column coverage incomplete")
    torch.save({"signature":signature,"instruction":INSTRUCTION,"max_values":5,"vectors":vectors},destination / "columns.pt")
    summary.update({"execution_status":"ran","scientific_validity":"valid_with_bounded_reencoding_audit",
        "elapsed_seconds":time.monotonic()-started,"output":file_record(destination / "columns.pt"),"probe_results":probe_results})
    _json(destination / "COLUMN_RECEIPT.json",summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--device",default="cuda:1")
    parser.add_argument("--batch-size",type=int,default=8)
    args = parser.parse_args()
    print(json.dumps(run(args.prepare_only,args.device,args.batch_size)))
