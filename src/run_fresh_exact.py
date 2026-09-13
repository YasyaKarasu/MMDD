"""Exact full-lake direct retrieval for fresh-lineage checkpoints."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path
import torch
from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from run_stage1_r21 import paths as r21_paths, out as r21_out, read_rows, write_rows
from run_stage1_r22 import paths as r22_paths, out as r22_out

ROOT = Path(__file__).resolve().parents[1]

@torch.inference_mode()
def run(arm: str, seed: int, step: int, device_name: str) -> dict:
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device = torch.device(device_name); ps = r21_paths(ROOT)
    source = r22_out(ROOT)/"fresh_lineage"/arm/f"seed{seed}"/"full_lake"/f"step_{step:06d}"
    rows = list(read_rows(source/"rankings.jsonl.gz"))
    store = FeatureStore.from_path(r22_paths(ROOT)["features"], cache_size=50000)
    table_ids = json.loads((r21_out(ROOT)/"indexes"/"Qwen-Raw"/"table_ids.json").read_text())
    targets = torch.stack([store.embedding_features(str(v)).embedding for v in table_ids]).to(device=device, dtype=torch.float32)
    ck = r22_out(ROOT)/"fresh_lineage"/arm/f"seed{seed}"/"edge"/"checkpoints"/f"step_{step:06d}.pt"
    model = load_student(ck, device).eval(); tv = model.project(targets,"table",role="target")
    rel = model.relations[model.relation_key("table","table")]
    out=[]
    for st in range(0,len(rows),32):
        batch=rows[st:st+32]; q=torch.stack([store.embedding_features(str(r["query_id"])).embedding for r in batch]).to(device=device,dtype=torch.float32)
        scores=model.project(q,"table",role="query") @ rel @ tv.T; vals,idx=scores.topk(k=100,dim=1)
        for r,ii,vv in zip(batch,idx.cpu(),vals.cpu()):
            ids=[str(table_ids[int(i)]) for i in ii]; pos=set(map(str,r["positive_target_ids"]))
            rec={"query_id":str(r["query_id"]),"query_kind":r["query_kind"],"positive_target_ids":sorted(pos),"exact_ids":ids,"exact_scores":[float(x) for x in vv]}
            for k in (10,20,50,100): rec[f"exact_recall@{k}"]=len(pos&set(ids[:k]))/len(pos) if pos else 0.0
            out.append(rec)
    path=source/"direct_exact_full_lake.jsonl.gz"; write_rows(path,out)
    result={"format_version":1,"status":"complete","retriever":arm,"seed":seed,"step":step,"corpus_tables":len(table_ids),"queries":len(out),"exact":{f"recall@{k}":statistics.fmean(r[f"exact_recall@{k}"] for r in out) for k in (10,20,50,100)},"checkpoint_sha256":checkpoint_fingerprint(ck),"rankings":str(path.resolve())}
    write_json(source/"EXACT.json",result)
    metrics_path = source/"metrics.json"
    if metrics_path.is_file():
        merged = json.loads(metrics_path.read_text()); merged["direct_exact_full_lake"] = result; write_json(metrics_path, merged)
    return result

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--arm",required=True); ap.add_argument("--seed",type=int,required=True); ap.add_argument("--step",type=int,default=1318); ap.add_argument("--device",default="cuda:0"); a=ap.parse_args(); print(json.dumps(run(a.arm,a.seed,a.step,a.device)))
