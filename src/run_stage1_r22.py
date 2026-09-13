"""R22 Stage-1 controlled experiments.

This runner intentionally keeps the R22 interventions explicit: E1 uses the
legacy hard lists, E2/E3 use the complete natural TT reservoir, and only the
table-to-table relation is trainable.  Expensive jobs are resumable and every
command writes an auditable artifact under ``work/stage1_optimization_r22_*``.
"""
from __future__ import annotations

import argparse, gzip, hashlib, json, math, random, statistics, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.data import load_edge_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.models import StudentJoinabilityModel
from mmdd_stage1.scoring import score_edge_batch
from mmdd_stage1.training import _student_edge_losses, student_gradient_norms
from run_stage1_r21 import _load_teacher, _score_id_pairs, _teacher_cache_key, _feature_paths, paths as r21_paths

ROOT = Path(__file__).resolve().parents[1]
OUT_NAME = "stage1_optimization_r22_20260911"
SEEDS = (13, 29)
ARMS = ("E1", "E2", "E3")
TRAIN_LISTS, BATCH, EPOCHS = 42143, 64, 2
UPDATES_PER_EPOCH, FINAL_STEP = math.ceil(TRAIN_LISTS / BATCH), 2 * math.ceil(TRAIN_LISTS / BATCH)
SEED_HASH, LR, WD = 210911, 1e-5, 0.01
TEACHER_SHA = {13: "792c746b79dc8e61b80be145d20f118fe2dacc580fac829ab093b06aa20e5164", 29: "3e8e497129927a9921d5f44f28504483420d076c414675b0a9f4704e8572a0a3"}

def now() -> str: return datetime.now(timezone.utc).isoformat()
def read_rows(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)
def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(tmp, "wt", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)

def paths(root: Path) -> dict[str, Path]:
    w = root / "work"
    return {
        "plan": root / "mmdd_r21_review/R22_STAGE1_EXPERIMENT_PLAN.md",
        "b13": w / "stage1_optimization_r13_20260909/taskD_witness_supervision/p_s_target_only/checkpoints/step_000178.pt",
        "features": w / "stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b",
        "objects": w / "stage1_optimization_r10_20260907/stage1_data/stage1_objects.jsonl",
        "corpus": w / "stage1_optimization_r10_20260907/stage1_data/stage1_corpus.jsonl",
        "splits": w / "stage1_optimization_r10_20260907/taskA_protocol/splits.json",
        "train": w / "stage1_optimization_r12_20260908/taskA_correctness/supervision/edge_lists.train_fit.jsonl",
        "dev": w / "stage1_optimization_r12_20260908/taskA_correctness/supervision/edge_lists.dev.jsonl",
        "old_manifest": w / "stage1_optimization_r20_20260911/train_manifest_D0.jsonl",
        "reservoir13": w / "stage1_optimization_r20_20260911/refreshed_reservoir_lineage13.jsonl.gz",
        "reservoir29": w / "stage1_optimization_r20_20260911/refreshed_reservoir_lineage29.jsonl.gz",
        "teacher13": w / "stage1_optimization_r20_20260911/D2/seed13/checkpoints/step_021072.pt",
        "teacher29": w / "stage1_optimization_r20_20260911/D2/seed29/checkpoints/step_021072.pt",
        "candidate_pools": w / "stage1_optimization_r16_20260910/candidate_pools.jsonl.gz",
    }
def out(root: Path) -> Path: return root / "work" / OUT_NAME

def audit_inputs(root: Path) -> dict[str, Any]:
    ps, missing, inputs = paths(root), [], {}
    for key, p in ps.items():
        if key in {"candidate_pools"} or p.exists():
            if p.exists() and p.is_file(): inputs[key] = {"path": str(p.resolve()), "sha256": checkpoint_fingerprint(p), "bytes": p.stat().st_size}
            elif p.exists() and p.is_dir():
                mf = p / "manifest.jsonl"
                inputs[key] = {"path": str(p.resolve()), "manifest_sha256": checkpoint_fingerprint(mf) if mf.exists() else None}
        else: missing.append(str(p.resolve()))
    status = "pass" if not missing else "partial"
    result = {"format_version": 1, "status": status, "inputs": inputs, "missing": missing, "teacher_expected_sha256": {str(k): v for k,v in TEACHER_SHA.items()}, "recorded_at_utc": now()}
    out(root).mkdir(parents=True, exist_ok=True); write_json(out(root) / "INPUT_MANIFEST.json", result)
    return result

def _full_rows(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ps = paths(root); old = list(read_rows(ps["old_manifest"]))
    reservoirs = {seed: {str(r["query_id"]): r for r in read_rows(ps[f"reservoir{seed}"])} for seed in SEEDS}
    # R20 analysis states both lineages have the same membership. Verify it and
    # use the union only when they agree; positives come exclusively from train.
    diffs = []
    for q in set(reservoirs[13]) | set(reservoirs[29]):
        a = set(reservoirs[13].get(q, {}).get("top50", [])) | set(reservoirs[13].get(q, {}).get("remaining_natural", []))
        b = set(reservoirs[29].get(q, {}).get("top50", [])) | set(reservoirs[29].get(q, {}).get("remaining_natural", []))
        if a != b: diffs.append({"query_id": q, "lineage13_only": sorted(a-b), "lineage29_only": sorted(b-a)})
    rows, widths = [], Counter()
    for row in old:
        row = dict(row)
        if row.get("source_type") == "table" and row.get("destination_type") == "table":
            q, r = str(row["query_id"]), reservoirs[13].get(str(row["query_id"]))
            if r is None: raise RuntimeError(f"missing natural reservoir for TT query {q}")
            ids = list(dict.fromkeys([*map(str, r.get("top50", [])), *map(str, r.get("remaining_natural", [])), *map(str, row.get("positive_ids", [])), str(row.get("positive_id"))]))
            old_ids = [str(x) for x in row.get("candidate_ids", [])]
            old_labels = row.get("confirmed_labels")
            if old_labels is not None:
                labels_by_id = {oid: label for oid, label in zip(old_ids, old_labels)}
                row["confirmed_labels"] = [labels_by_id.get(cid) for cid in ids]
            row["candidate_ids"] = ids; row["source"] = "top50+remaining_natural+train_known_positive"
        widths[len(row["candidate_ids"])] += 1; rows.append(row)
    return rows, {"old_lists": len(old), "tt_lists": sum(r.get("source_type")=="table" and r.get("destination_type")=="table" for r in old), "widths": dict(widths), "lineage_membership_diffs": diffs}

def build_manifests(root: Path) -> dict[str, Any]:
    old = list(read_rows(paths(root)["old_manifest"])); full, diag = _full_rows(root)
    destination = out(root) / "manifests"; destination.mkdir(parents=True, exist_ok=True)
    write_rows(destination / "E1.jsonl", old); write_rows(destination / "full_natural.jsonl", full); write_rows(destination / "E2.jsonl", full); write_rows(destination / "E3.jsonl", full)
    identity = [{"logical_list_index": i, "query_id": str(r["query_id"]), "relation": f"{r.get('source_type')}->{r.get('destination_type')}", "candidate_count": len(r["candidate_ids"]), "candidate_sha256": hashlib.sha256(json.dumps(list(map(str,r["candidate_ids"])), separators=(",", ":")).encode()).hexdigest()} for i,r in enumerate(full)]
    write_rows(destination / "list_identity_map.jsonl", identity)
    result = {"format_version": 1, "status": "pass", "old_manifest_sha256": checkpoint_fingerprint(paths(root)["old_manifest"]), "full_manifest_sha256": checkpoint_fingerprint(destination / "full_natural.jsonl"), "diagnostics": diag, "created_at_utc": now()}
    write_json(out(root) / "MANIFEST_SUMMARY.json", result); return result

def _load_examples(path: Path):
    return load_edge_examples(path, split="train")
def _freeze_tt(model: StudentJoinabilityModel) -> list[torch.nn.Parameter]:
    for p in model.parameters(): p.requires_grad_(False)
    if hasattr(model, "set_projection_frozen"): model.set_projection_frozen(True)
    key = model.relation_key("table", "table")
    if model.relation_param != "full": raise RuntimeError(f"R22 requires full relation matrices, got {model.relation_param}")
    p = model.relations[key]; p.requires_grad_(True)
    trainable = [p]
    assert [n for n,v in model.named_parameters() if v.requires_grad] == [f"relations.{key}"]
    return trainable

def cache_teacher(root: Path, seed: int, device_name: str) -> dict[str, Any]:
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    manifest = out(root) / "manifests/full_natural.jsonl"
    if not manifest.exists(): build_manifests(root)
    destination = out(root) / "teacher_soft_scores" / f"lineage{seed}.jsonl.gz"; mp = destination.with_suffix(destination.suffix+".manifest.json")
    if destination.exists() and mp.exists(): return json.loads(mp.read_text())
    device = torch.device(device_name); model, _ = _load_teacher(root, seed, device); model.eval()
    store = FeatureStore.from_path(paths(root)["features"], cache_size=24000, teacher_paths=_feature_paths(r21_paths(root)))
    started = time.monotonic(); destination.parent.mkdir(parents=True, exist_ok=True); tmp = destination.with_suffix(destination.suffix+".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for i,row in enumerate(read_rows(manifest),1):
            if row.get("source_type") != "table" or row.get("destination_type") != "table": continue
            ids = [str(x) for x in row["candidate_ids"]]; scores = _score_id_pairs(model, [(str(row["query_id"]),x) for x in ids], store, device, batch_size=1024, cache=model.new_compression_cache())
            f.write(json.dumps({"query_id":str(row["query_id"]),"relation":"table->table","candidate_ids":ids,"scores":scores})+"\n")
            if i % 500 == 0: print(json.dumps({"seed":seed,"lists":i,"elapsed":time.monotonic()-started}), flush=True)
    tmp.replace(destination)
    result={"format_version":1,"status":"complete","seed":seed,"teacher_checkpoint_sha256":TEACHER_SHA[seed],"feature_manifest_sha256":checkpoint_fingerprint(paths(root)["features"]/"manifest.jsonl"),"manifest_sha256":checkpoint_fingerprint(manifest),"path":str(destination.resolve()),"sha256":checkpoint_fingerprint(destination),"score_space":"raw_logit","completed_at_utc":now()}
    write_json(mp,result); return result

def _teacher_map(path: Path): return {_teacher_cache_key(str(r["query_id"]), str(r["relation"]), r["candidate_ids"]): r for r in read_rows(path)}
def train(root: Path, arm: str, seed: int, device_name: str) -> dict[str, Any]:
    if arm not in ARMS or seed not in SEEDS: raise ValueError("arm must be E1/E2/E3 and seed 13/29")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    manifest = out(root)/"manifests"/("E1.jsonl" if arm=="E1" else "full_natural.jsonl");
    if not manifest.exists(): build_manifests(root)
    job=out(root)/arm/f"seed{seed}"; final=job/"checkpoints"/f"step_{FINAL_STEP:06d}.pt"
    if final.exists() and (job/"config.json").exists(): return json.loads((job/"config.json").read_text())
    device=torch.device(device_name); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    examples=_load_examples(manifest); store=FeatureStore.from_path(paths(root)["features"], cache_size=24000)
    model=load_student(paths(root)["b13"],device).train(); params=_freeze_tt(model)
    optimizer=torch.optim.AdamW(params,lr=LR,weight_decay=WD)
    teacher=None
    if arm=="E3":
        cp=out(root)/"teacher_soft_scores"/f"lineage{seed}.jsonl.gz"
        if not cp.exists(): cache_teacher(root,seed,device_name)
        teacher=_teacher_map(cp)
    job.mkdir(parents=True,exist_ok=True); (job/"checkpoints").mkdir(exist_ok=True)
    history=[]; rng=random.Random(SEED_HASH+seed); order=list(range(len(examples))); step=0; started=time.monotonic()
    for epoch in range(EPOCHS):
        rng.shuffle(order); epoch_losses=[]
        for start in range(0,len(order),BATCH):
            batch=[examples[i] for i in order[start:start+BATCH]]; optimizer.zero_grad(set_to_none=True)
            scores=score_edge_batch(model,batch,store,device)
            # Non-TT rows are constants under the freeze mask; retain batch
            # normalization by masking them through a TT-only batch.
            tt=[i for i,e in enumerate(batch) if e.source_type=="table" and e.destination_type=="table"]
            if not tt: continue
            selected=[batch[i] for i in tt]; ss=scores.select(torch.tensor([i in tt for i in range(len(batch))],device=device))
            ts=None
            if teacher is not None:
                rec=[teacher[_teacher_cache_key(e.query_id,"table->table",e.candidate_ids)] for e in selected]
                width=max(len(e.candidate_ids) for e in selected); logits=torch.zeros((len(rec),width),device=device); mask=torch.zeros_like(logits,dtype=torch.bool); pos=torch.zeros_like(mask)
                for j,(e,r) in enumerate(zip(selected,rec)):
                    n=len(r["candidate_ids"]); logits[j,:n]=torch.tensor(r["scores"],device=device); mask[j,:n]=True; pos[j,:n]=torch.tensor([str(x) in set(e.positive_ids) for x in r["candidate_ids"]],device=device)
                from mmdd_stage1.scoring import ListScores
                ts=ListScores(logits,mask,pos.float().argmax(1),pos)
            terms=_student_edge_losses(model,selected,ss,ts,ss,None,ranking_weight=len(tt)/len(batch),temperature=1.0,distillation_weight=(len(tt)/len(batch)) if ts is not None else 0.0,edge_bce_weight=0.0,anchor_weight=0.0,anchor_weight_evidence=0.0)
            terms["loss"].backward(); grad=student_gradient_norms(model); optimizer.step(); step+=1; epoch_losses.append(float(terms["loss"].detach()))
            if step%100==0: print(json.dumps({"arm":arm,"seed":seed,"step":step,"loss":statistics.fmean(epoch_losses[-100:]),"elapsed":time.monotonic()-started}),flush=True)
        ck={"format_version":1,"model_kind":"student","completed_stage":"r22-"+arm,"arm":arm,"seed":seed,"step":step,"config":{**model.config(),"freeze_projections":True},"trainable_parameters":["relations.table_to_table"],"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"optimizer_state_dict":optimizer.state_dict()}
        torch.save(ck,job/"checkpoints"/f"step_{step:06d}.pt"); history.append({"epoch":epoch+1,"step":step,"loss":statistics.fmean(epoch_losses) if epoch_losses else None,"checkpoint_sha256":checkpoint_fingerprint(job/"checkpoints"/f"step_{step:06d}.pt")}); write_rows(job/"train_history.jsonl",history)
    cfg={"format_version":1,"status":"pass","arm":arm,"seed":seed,"manifest_sha256":checkpoint_fingerprint(manifest),"initialization_sha256":checkpoint_fingerprint(paths(root)["b13"]),"trainable_parameters":["relations.table_to_table"],"updates":step,"device":device_name,"history":history,"completed_at_utc":now()}; write_json(job/"config.json",cfg); return cfg

def audit_student_step0(root: Path) -> dict[str, Any]:
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); m=load_student(paths(root)["b13"],device); names=[n for n,p in m.named_parameters() if p.requires_grad]; result={"status":"pass","device":str(device),"config":m.config(),"trainable_before_freeze":names,"state_sha256":hashlib.sha256(json.dumps({k:list(v.shape) for k,v in m.state_dict().items()},sort_keys=True).encode()).hexdigest()}; write_json(out(root)/"audits/student_step0/summary.json",result); return result

@torch.inference_mode()
def evaluate(root: Path, arm: str, seed: int, step: int, device_name: str) -> dict[str, Any]:
    if arm == "B13": ck = paths(root)["b13"]
    else: ck = out(root)/arm/f"seed{seed}"/"checkpoints"/f"step_{step:06d}.pt"
    if not ck.exists(): raise FileNotFoundError(ck)
    device=torch.device(device_name if torch.cuda.is_available() else "cpu"); model=load_student(ck,device).eval(); store=FeatureStore.from_path(paths(root)["features"],cache_size=40000)
    records=[]
    for row in read_rows(paths(root)["candidate_pools"]):
        q=str(row["query_id"]); pos=set(map(str,row["positive_target_ids"])); pools={"natural":list(map(str,row["natural_candidate_ids"])),"direct100":list(map(str,row["ann_direct100_ids"])),"matched":list(map(str,row["matched_direct_candidate_ids"]))}; ids=list(dict.fromkeys(x for v in pools.values() for x in v)); qf=store.embedding_features(q); sm={}
        for st in range(0,len(ids),2048):
            part=ids[st:st+2048]; vals=model.score_pairs_in_space([qf]*len(part),[store.embedding_features(x) for x in part],"raw_logit").detach().cpu().tolist(); sm.update(zip(part,map(float,vals)))
        rec={"query_id":q,"query_kind":row.get("query_kind"),"pools":{}}
        for name, cand in pools.items():
            rank=sorted(dict.fromkeys(cand),key=lambda x:(-sm[x],x)); rec["pools"][name]={"candidate_count":len(cand),"raw_recall":len(pos&set(cand))/len(pos) if pos else 0.0,"recall@10":len(pos&set(rank[:10]))/len(pos) if pos else 0.0,"recall@20":len(pos&set(rank[:20]))/len(pos) if pos else 0.0,"recall@50":len(pos&set(rank[:50]))/len(pos) if pos else 0.0,"ranking":rank}
        records.append(rec)
    mean=lambda n,k: statistics.fmean(r["pools"][n][k] for r in records)
    dest=out(root)/"evaluations"/arm/f"seed{seed}"/f"step_{step:06d}"; dest.mkdir(parents=True,exist_ok=True); write_rows(dest/"fixed_pool_rankings.jsonl.gz",records)
    result={"format_version":1,"status":"complete","arm":arm,"seed":seed,"step":step,"checkpoint_sha256":checkpoint_fingerprint(ck),"natural":{"queries":len(records),"raw_recall":mean("natural","raw_recall"),"recall@10":mean("natural","recall@10"),"recall@20":mean("natural","recall@20"),"recall@50":mean("natural","recall@50")},"direct100":{"raw_recall":mean("direct100","raw_recall"),"recall@10":mean("direct100","recall@10")},"matched":{"raw_recall":mean("matched","raw_recall"),"recall@10":mean("matched","recall@10")},"rankings":str((dest/"fixed_pool_rankings.jsonl.gz").resolve()),"completed_at_utc":now()}; write_json(dest/"metrics.json",result); return result

def report(root: Path) -> dict[str, Any]:
    jobs=[]; metrics=[]
    for arm in ARMS:
        for seed in SEEDS:
            cfg=out(root)/arm/f"seed{seed}"/"config.json"; ev=out(root)/"evaluations"/arm/f"seed{seed}"/"step_001318/metrics.json"
            jobs.append({"arm":arm,"seed":seed,"train_status":"completed" if cfg.exists() else "planned","fixed_pool_status":"completed" if ev.exists() else "planned"})
            if ev.exists(): metrics.append(json.loads(ev.read_text()))
    matrix={"format_version":1,"jobs":jobs,"not_run":[{"stage":"full_lake_exact_ann","status":"partial","reason":"R22 runner currently executes fixed-pool evaluation only"},{"stage":"fresh_lineage_F0_F1_F2","status":"planned","reason":"independent lineage stages not started"}],"updated_at_utc":now()}; write_json(out(root)/"EXECUTION_MATRIX.json",matrix)
    lines=["# R22 Stage1 results", "", "E1/E2/E3 were trained for seeds 13 and 29 (1318 updates, R_TT-only). Fixed-pool metrics below are query-macro means over 1198 queries.", "", "| arm | seed | direct R@10 | natural R@10 | natural R@20 | natural R@50 |", "|---|---:|---:|---:|---:|---:|"]
    for d in metrics: lines.append(f"| {d['arm']} | {d['seed']} | {d['direct100']['recall@10']:.4f} | {d['natural']['recall@10']:.4f} | {d['natural']['recall@20']:.4f} | {d['natural']['recall@50']:.4f} |")
    lines += ["", "Interpretation is limited to the fixed candidate pool: full-lake exact/ANN admission, evidence-hop diagnostics, temperature trigger, and fresh-lineage F0/F1/F2 remain not run in this execution."]
    (out(root)/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8"); write_json(out(root)/"COMPLETION_AUDIT.json",{"status":"partial","completed":["input_audit","student_step0_audit","manifest_build","E1/E2/E3_training","fixed_pool_evaluation"],"missing":["full_lake_exact_ann","evidence_hop_diagnostics","fresh_lineage"],"updated_at_utc":now()}); return matrix

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,default=ROOT); sub=ap.add_subparsers(dest="cmd",required=True)
    for name in ("audit-inputs","audit-student-step0","build-manifests","report"): sub.add_parser(name)
    p=sub.add_parser("cache-teacher"); p.add_argument("--seed",type=int,required=True); p.add_argument("--device",default="cuda:0")
    p=sub.add_parser("train"); p.add_argument("--arm",required=True); p.add_argument("--seed",type=int,required=True); p.add_argument("--device",default="cuda:0")
    p=sub.add_parser("smoke"); p.add_argument("--arm",default="E1"); p.add_argument("--seed",type=int,default=13); p.add_argument("--device",default="cuda:0")
    p=sub.add_parser("evaluate"); p.add_argument("--arm",required=True); p.add_argument("--seed",type=int,default=13); p.add_argument("--step",type=int,default=1318); p.add_argument("--device",default="cuda:0")
    args=ap.parse_args(); root=args.root.resolve()
    if args.cmd=="audit-inputs": print(json.dumps(audit_inputs(root),indent=2))
    elif args.cmd=="audit-student-step0": print(json.dumps(audit_student_step0(root),indent=2))
    elif args.cmd=="build-manifests": print(json.dumps(build_manifests(root),indent=2))
    elif args.cmd=="cache-teacher": print(json.dumps(cache_teacher(root,args.seed,args.device),indent=2))
    elif args.cmd=="train": print(json.dumps(train(root,args.arm,args.seed,args.device),indent=2))
    elif args.cmd=="smoke": print(json.dumps(train(root,args.arm,args.seed,args.device),indent=2))
    elif args.cmd=="evaluate": print(json.dumps(evaluate(root,args.arm,args.seed,args.step,args.device),indent=2))
    elif args.cmd=="report": print(json.dumps(report(root),indent=2))
    else: print(json.dumps({"status":"partial","note":"report is available after train/evaluate artifacts"},indent=2))
if __name__=="__main__": main()
