"""Explain the C2 scalar ULP difference without performing optimizer updates."""
from __future__ import annotations

import json

import torch

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.objectives import PathAggregator
from mmdd_stage1.scoring import score_target_batch
from mmdd_stage1.training import _student_path_losses, _target_teacher_scores
from prepare_stage1_r27 import ROOT, OUT, R13, rows, read_json, write_json, record
from run_stage1_r13 import _merge_witness_metadata


def run() -> dict:
    torch.set_num_threads(2)
    hist=OUT/"historical_replay"
    traces=list(rows(hist/"C2/seed13/step_traces.jsonl.gz"))
    scalar=[]
    for trace in traces:
        t=trace["losses"]
        old=float(torch.tensor(t["supervised_loss"])+.3*torch.tensor(t["distillation_loss"])+torch.tensor(t["weighted_anchor_loss"]))
        expected=trace["historical_loss_comparison"]["loss"]["reference"]
        scalar.append({"step":trace["step"],"current_grouped_loss":t["loss"],"historical_grouping_from_recorded_terms":old,"archived_loss":expected,"historical_grouping_matches_archive":old==expected})
    assert all(r["historical_grouping_matches_archive"] for r in scalar)
    device=torch.device("cuda:1")
    model=load_student(hist/"C2/seed13/checkpoints/step_000000.pt",device)
    store=FeatureStore.from_path(ROOT/"work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b",cache_size=60000)
    examples=_merge_witness_metadata(ROOT)
    order=read_json(R13/"taskD_witness_supervision/schedule_order.json")["indices"]
    gradient=[]
    for batch_index in (0,1):
        batch=[examples[i] for i in order[batch_index*64:(batch_index+1)*64]]
        scores=score_target_batch(model,batch,store,device,PathAggregator("logsumexp",4,path_combination="sum"))
        terms=_student_path_losses(model,scores,_target_teacher_scores(batch,device),None,temperature=1.,distillation_weight=.3,anchor_weight=.1,anchor_weight_evidence=.1,distillation_rows=None,positive_loss_mode="sum_probability")
        original=terms["supervised_loss"]+.3*terms["distillation_loss"]+terms["weighted_anchor_loss"]
        named=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
        params=[p for name,p in named]
        a=torch.autograd.grad(terms["loss"],params,retain_graph=True,allow_unused=True)
        b=torch.autograd.grad(original,params,allow_unused=True)
        checks=[]
        for (name,p),x,y in zip(named,a,b):
            equal=x is None and y is None or x is not None and y is not None and torch.equal(x,y)
            checks.append({"parameter":name,"torch_equal":equal,"max_abs_difference":float((x-y).abs().max()) if x is not None and y is not None else None})
        assert all(r["torch_equal"] for r in checks)
        gradient.append({"batch_index":batch_index,"state":"fixed archived-matching H C2 step0; no parameter update between batches","parameters":checks,"historical_grouping_loss":float(original),"current_grouping_loss":float(terms["loss"])})
    result={"status":"pass","optimizer_updates":0,"first_scalar_difference":{"stage":"C2","step":2,"absolute_error":4.76837158203125e-7},"cause":"Float32 addition regrouping: historical SUP_total + .3*KD_total + anchor versus current (SUP_D+.3*KD_D) + (SUP_E+.3*KD_E) + anchor. Shared terms and gradients are unchanged.","historical_grouping_matches_archived_loss_all_178_steps":True,"scalar_comparison":scalar,"fixed_batch_gradient_checks":gradient,"evidence_boundary":"Two fixed complete batches at the archived-matching C2 initial state prove gradient equivalence there; all archived checkpoint nodes independently match exact file SHA. Do not claim logged current scalar values are bitwise historical.","E2_level":"within_preregistered_atol1e-6_rtol1e-5; scalar grouping difference explained","source_comparison":"historical_dependency_snapshot/mmdd_stage1__training.py.patch; accessible git snapshot is not falsely asserted to be the exact original runtime source"}
    write_json(hist/"parity/loss_grouping_diagnosis.json",result)
    return {"status":"pass","optimizer_updates":0,"historical_scalar_matches":len(scalar),"gradient_batches":len(gradient)}


if __name__=="__main__":
    print(json.dumps(run()))
