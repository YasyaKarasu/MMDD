"""Execute the eight C2 extensions only after the preregistered budget gate passes."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import torch

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.data import load_target_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.objectives import PathAggregator
from mmdd_stage1.r26_extension_objectives import objective
from mmdd_stage1.r26_statistics import loss_extension_gate
from mmdd_stage1.r26_training import graph_edges
from mmdd_stage1.scoring import score_target_batch,score_edge_batch
from mmdd_stage1.training import _anchor_losses,student_gradient_norms
from prepare_stage1_r26 import ROOT,OUT,file_record,parameter_sha
from run_stage1_r21 import paths,read_rows,write_rows
from run_stage1_r22_f0 import _optimizer
from run_stage1_r25 import _json,_load_target_cache,_target_cache_scores,_r25_path_pool,out as r25_out,sha256
from train_stage1_r26 import known_edges

ARMS = ("O-QTKD","O-U","O-UQTKD","O-LSE-QTKD")


def train(arm: str,seed: int,device_name: str) -> dict:
    torch.set_num_threads(3)
    gate_path = OUT / "acceptance/LOSS_EXTENSION_GATE_FROZEN.json"
    if not gate_path.exists():
        names = ["B13"]+[f"{a}/seed{s}/step178" for a in ("R26-O-SUP","R25-SPLIT-SUP") for s in (13,29)]
        sources = {n:OUT / "rankings" / n / "metrics.json" for n in names}
        gate = loss_extension_gate({n:json.loads(p.read_text()) for n,p in sources.items() if p.exists()})
        if gate["status"] != "triggered":
            raise ValueError("Conditional loss-extension budget gate did not pass")
        _json(gate_path,{**gate,"inputs":{n:file_record(p) for n,p in sources.items()}})
    gate = json.loads(gate_path.read_text())
    if gate["status"] != "triggered":
        raise ValueError("Conditional budget gate is not triggered")
    for source in gate["inputs"].values():
        if sha256(Path(source["path"])) != source["sha256"]:
            raise ValueError("Frozen gate input changed")
    if not json.loads((OUT / "acceptance/C1_TENSOR_AUDIT.json").read_text())["c1"][str(seed)]["reuse_valid"]:
        raise ValueError("Parent C1 not validated")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    parent = r25_out(ROOT) / f"training/C1/seed{seed}/checkpoints/step_000659.pt"
    graph = _r25_path_pool(ROOT)
    order_path = OUT / "common/c2_order.jsonl"
    teacher_path = ROOT / "work/stage1_optimization_r24_20260913/path_pool/teacher_target_seed13.jsonl.gz"
    kd = arm != "O-U"
    uniform = arm in ("O-U","O-UQTKD")
    signature = {"arm":arm,"seed":seed,"parent":file_record(parent),"gate":file_record(gate_path),
        "features":file_record(paths(ROOT)["features"] / "manifest.jsonl"),"protocol":file_record(OUT / "PROTOCOL.json"),
        "graph":file_record(graph),"order":file_record(order_path),"registry":file_record(paths(ROOT)["train"]),
        "teacher_cache":file_record(teacher_path) if kd else None,"teacher_T0":file_record(ROOT / "work/stage1_optimization_r22_20260911/fresh_lineage/T1-B/seed13/checkpoints/step_010536.pt") if kd else None,
        "objective":{"family":"lse" if arm == "O-LSE-QTKD" else "split","kd_weight":.3 if kd else 0.,"uniform_weight":.3 if uniform else 0.},
        "optimizer":{"projection_lr":1e-6,"relation_lr":1e-5,"weight_decay":.01,"batch_size":64,"coverage":1},
        "code":{name:sha256(ROOT / "src" / name) for name in ("train_stage1_r26_extension.py","mmdd_stage1/r26_extension_objectives.py","mmdd_stage1/r26_training.py","mmdd_stage1/scoring.py","mmdd_stage1/models.py","run_stage1_r22_f0.py")}}
    job = OUT / "training" / arm / f"seed{seed}"
    receipt = job / "C2_COMPLETION_RECEIPT.json"
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous["signature"] != signature:
            raise ValueError("Existing extension training identity changed")
        return previous
    job.mkdir(parents=True,exist_ok=True)
    identity = job / "RUN_IDENTITY.json"
    if identity.exists() and json.loads(identity.read_text()) != signature:
        raise ValueError("Incomplete extension identity changed")
    _json(identity,signature)
    source_examples = load_target_examples(graph,split="train")
    order = list(read_rows(order_path))
    examples = [source_examples[r["source_row"]] for r in order]
    if any(e.query_id != r["query_id"] for e,r in zip(examples,order)):
        raise ValueError("Query order changed")
    store = FeatureStore.from_path(paths(ROOT)["features"],cache_size=120000)
    registry = known_edges()
    violations = [{"query":e.query_id,"missing":sorted((registry[e.query_id,"table","table"] & {c.target_id for c in e.candidates})-set(e.positive_target_ids))} for e in examples]
    violations = [r for r in violations if r["missing"]]
    _json(job / "TARGET_CLOSURE.json",{"checked_queries":len(examples),"violations":violations})
    if violations:
        raise ValueError("Known positive closure failed")
    aux = [graph_edges(e,registry,lambda oid:store.embedding_features(oid).object_type) for e in examples]
    write_rows(job / "query_graph_edges.jsonl.gz",({"query_id":e.query_id,"lists":[asdict(x) for x in rows]} for e,rows in zip(examples,aux)))
    cache = _load_target_cache(teacher_path) if kd else None
    model = load_student(parent,device).train()
    optimizer = _optimizer(model)
    initial = parameter_sha(model)
    steps = math.ceil(len(examples)/64)
    (job / "checkpoints").mkdir(exist_ok=True)
    def save(step):
        path = job / "checkpoints" / f"step_{step:06d}.pt"
        torch.save({"format_version":1,"model_kind":"student","completed_stage":"r26-extension-c2","arm":arm,"seed":seed,"step":step,
                    "config":model.config(),"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    "optimizer_state_dict":optimizer.state_dict(),"r26_signature":signature},path)
        return {"step":step,"parameter_sha256":parameter_sha(model),**file_record(path)}
    checkpoints = [save(0)]
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    with (job / "train_history.jsonl").open("w") as handle:
        for step,start in enumerate(range(0,len(examples),64),1):
            batch = examples[start:start+64]
            optimizer.zero_grad(set_to_none=True)
            scores = score_target_batch(model,batch,store,device,PathAggregator("logsumexp",8,path_combination="sum"),student_score_space="raw_logit")
            teacher = _target_cache_scores(batch,cache,device,qt_evidence=True) if kd else None
            edge_scores = relations = owners = None
            if uniform:
                edges,owners = [],[]
                for owner,rows in enumerate(aux[start:start+len(batch)]):
                    edges.extend(rows)
                    owners.extend([owner]*len(rows))
                relations = [f"{e.source_type}->{e.destination_type}" for e in edges]
                edge_scores = score_edge_batch(model,edges,store,device,student_score_space="raw_logit")
            _,anchor = _anchor_losses(model,.1,.1)
            terms = objective(scores,teacher=teacher,uniform_scores=edge_scores,uniform_relations=relations,owners=owners,anchor=anchor,**signature["objective"])
            terms["loss"].backward()
            gradients = student_gradient_norms(model)
            optimizer.step()
            row = {"step":step,"query_ids":[e.query_id for e in batch],"terms":{k:float(v.detach().cpu()) if isinstance(v,torch.Tensor) else v for k,v in terms.items()},
                   "gradient":gradients,"elapsed_seconds":time.monotonic()-started}
            handle.write(json.dumps(row)+"\n"); handle.flush()
            if step in (89,178):
                checkpoints.append(save(step))
            if step % 10 == 0:
                print(json.dumps({"arm":arm,"seed":seed,"step":step,"loss":row["terms"]["loss"]}),flush=True)
    result = {"signature":signature,"execution_status":"ran","scientific_validity":"valid","optimizer_initial_state":"fresh",
        "optimizer_updates":steps,"coverage_lists":len(examples),"initial_parameter_sha256":initial,"final_parameter_sha256":parameter_sha(model),
        "checkpoints":checkpoints,"elapsed_seconds":time.monotonic()-started,"peak_allocated_bytes":torch.cuda.max_memory_allocated(device),
        "history":file_record(job / "train_history.jsonl"),"graph_edges":file_record(job / "query_graph_edges.jsonl.gz")}
    _json(receipt,result)
    return {"arm":arm,"seed":seed,"steps":steps}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm",choices=ARMS,required=True)
    parser.add_argument("--seed",type=int,choices=(13,29),required=True)
    parser.add_argument("--device",default="cuda:1")
    args = parser.parse_args()
    print(json.dumps(train(args.arm,args.seed,args.device)))
