"""Frozen 32-case Real/NoE local diagnostic, with isolated evaluation truth."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from mmdd_dataset.wdc_runtime import iter_dataset_artifact
from mmdd_stage1.features import FeatureStore
from mmdd_stage2.checkpoints import load_candidate_scorer
from mmdd_stage2.data import load_stage2_index, row_values, column_values
from mmdd_stage2.pipeline import Stage2Verifier, LocalizedEvidence
from mmdd_stage2.r26_generation import R26QwenBackend, VALUE_PROMPT
from mmdd_stage2.routing import SimilarityEvidenceRouter
from mmdd_stage2.verifier import EvidenceBundle
from audit_stage2_r26_engineering import DATASET
from prepare_stage1_r27 import ROOT, OUT, R26, rows, read_json, write_json, record, stable_sha
from prepare_stage2_r27 import write_rows

PANEL=OUT/"witness_panel"


def visible_table(table: dict) -> dict:
    return {"table_id":table["table_id"],"columns":[{k:c[k] for k in ("column_index","column_name")} for c in table["columns"]],"rows":[{"row_id":r["row_id"],"cells":[{k:c[k] for k in ("column_index","text")} for c in r["cells"]]} for r in table["rows"]]}


def prepare_inputs() -> dict:
    cases=list(rows(PANEL/"frozen_case_retrieval.jsonl.gz"))
    objects=load_stage2_index(DATASET,query_ids={c["query_id"] for c in cases},target_ids={c["target_id"] for c in cases},evidence_ids={e for c in cases for e in c["retained_evidence"]["selected_evidence_ids"]})
    source_ids={q["source_table_id"] for q in objects.queries.values()}
    sources={r["source_table_id"]:r for r in iter_dataset_artifact(DATASET,"source_tables") if r["source_table_id"] in source_ids}
    truth=[]; runtime=[]
    for case in cases:
        q,t=case["query_id"],case["target_id"]
        query=objects.queries[q]
        truth.append({"query_id":q,"target_id":t,"original_query":query,"source_table":sources.get(query["source_table_id"]),"original_target":objects.targets[t],"source_provenance":"independent original source table archive; never passed to generation"})
        clean={"query_id":q,"target_id":t,"query":visible_table(query),"target":visible_table(objects.targets[t]),"retained_evidence":case["retained_evidence"],"evidence":{e:{k:v for k,v in objects.evidence[e].items() if k in ("asset_id","asset_type","content","local_path","relative_path")} for e in case["retained_evidence"]["selected_evidence_ids"]},"diagnostic_outside_deployment_queue":case["diagnostic_outside_deployment_queue"],"local_pair_diagnostic":True}
        runtime.append(clean)
    write_rows(PANEL/"eval_truth_sidecar.jsonl.gz",truth)
    write_rows(PANEL/"inputs_sanitized.jsonl.gz",runtime)
    forbidden={"positive_target_ids","query_kind","hidden_attributes","source_row_id","source_column_index","ground_truth_join","source_table_id"}
    def keys(value):
        if isinstance(value,dict):
            return set(value)|{k for v in value.values() for k in keys(v)}
        if isinstance(value,list):
            return {k for v in value for k in keys(v)}
        return set()
    assert not (keys(runtime)&forbidden)
    write_json(OUT/"audit/prompt_leakage_audit.json",{"status":"pass","sanitization":"allow-listed visible query/target tables and evidence; independent truth in separate file","forbidden_keys":sorted(forbidden),"runtime":record(PANEL/"inputs_sanitized.jsonl.gz"),"truth_sidecar":record(PANEL/"eval_truth_sidecar.jsonl.gz"),"target_content_policy":"allowed only to frozen column reader; never passed to value generator"})
    return {"cases":len(runtime),"source_truth_tables":len(sources)}


def save_localized(local: LocalizedEvidence, q: str, t: str, position: int) -> dict:
    result=local.record()
    if local.image is not None:
        dest=PANEL/"grounding_evidence"/f"{q}_{t}_{position}.png"
        dest.parent.mkdir(parents=True,exist_ok=True)
        local.image.save(dest)
        result["crop_file"]=record(dest)
    return result


def restore_localized(rec: dict) -> LocalizedEvidence:
    from PIL import Image
    return LocalizedEvidence(rec["evidence_id"],rec["evidence_type"],text=rec.get("text_span"),image=Image.open(rec["crop_file"]["path"]).copy() if rec.get("crop_file") else None,box=tuple(rec["image_box"]) if rec.get("image_box") else None,text_span_relevance=rec.get("text_span_relevance"),image_presence_probability=rec.get("image_presence_probability"))


@torch.inference_mode()
def run(device: str) -> dict:
    torch.set_num_threads(2)
    assert read_json(OUT/"audit/prompt_leakage_audit.json")["status"]=="pass"
    assert read_json(OUT/"audit/G4.json")["status"]=="pass"
    model_dir=ROOT/"hf_models/Qwen3.5-9B"
    scorer_path=ROOT/"work/stage1_optimization_r25_final_20260914/stage2/r25_b13_column_scorer.pt"
    signature={"inputs":record(PANEL/"inputs_sanitized.jsonl.gz"),"scorer":record(scorer_path),"model_config":record(model_dir/"config.json"),"chat_template":record(model_dir/"chat_template.jinja"),"prompt":VALUE_PROMPT,"reader_generation_source":record(ROOT/"src/mmdd_stage2/r26_generation.py"),"runner":record(Path(__file__)),"device":device,"conditions":["Real-crop","NoE-fill"],"max_cell_opportunities":4,"local_pair_scope":"column argmax is identical to full-pool argmax because softmax denominator is shared across columns; no claims about acceptance priority or system Recall"}
    sig=stable_sha(signature)
    identity=PANEL/"RUN_IDENTITY.json"
    if identity.exists():
        assert read_json(identity)==signature,"Resume identity changed"
    else:
        write_json(identity,signature)
    cases=list(rows(PANEL/"inputs_sanitized.jsonl.gz"))
    backend=R26QwenBackend(model_dir,device=device,dtype="bf16")
    scorer=load_candidate_scorer(scorer_path,torch.device("cpu"),expected_model_dir=model_dir).to(backend.device)
    router=SimilarityEvidenceRouter(FeatureStore.from_path(ROOT/"work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b"))
    verifier=Stage2Verifier(backend,scorer,evidence_router=router)
    attempts_path=PANEL/"generation_attempts.jsonl.gz"
    backend.generation_records=list(rows(attempts_path)) if attempts_path.exists() else []
    outputs={condition:list(rows(PANEL/(name+".jsonl.gz"))) if (PANEL/(name+".jsonl.gz")).exists() else [] for condition,name in (("Real-crop","Real"),("NoE-fill","NoE"))}
    completed={(c,r["query_id"],r["target_id"]) for c,rr in outputs.items() for r in rr}
    for index,case in enumerate(cases,1):
        q,t=case["query_id"],case["target_id"]
        if all((c,q,t) in completed for c in outputs):
            continue
        query,target=case["query"],case["target"]
        opportunity_path=PANEL/"opportunities"/f"{q}_{t}.json"
        if opportunity_path.exists():
            opportunity=read_json(opportunity_path)
        else:
            try:
                bundle=EvidenceBundle(target_id=t,retrieval_score=0.,evidence_ids=tuple(case["retained_evidence"]["selected_evidence_ids"]))
                scores=verifier.score_candidates(query,[bundle],{t:target},case["evidence"])
                selection=scores[0].selection
                assignments=router.assign(q,bundle.evidence_ids,row_count=len(query["rows"]))
                assert assignments==case["retained_evidence"]["routed_rows"],"Original row routing changed"
                localized={}
                for position in sorted(set(assignments.values()))[:4]:
                    visible=row_values(query,query["rows"][position])
                    local=[backend.localize_evidence(visible,attribute_name=selection.column_name,evidence=case["evidence"][e]) for e in bundle.evidence_ids if assignments[e]==position]
                    best=local[0] if len(local)==1 else local[int(backend.evidence_logits(visible,attribute_name=selection.column_name,candidates=local).argmax())]
                    localized[str(position)]=save_localized(best,q,t,position)
                opportunity={"status":"ready","selection":asdict(selection),"column_scores":asdict(scores[0]),"assignments":assignments,"localized":localized,"row_count":len(query["rows"]),"coverage_upper_bound":len(localized)/len(query["rows"]),"query_id":q,"target_id":t,"run_signature":sig}
            except (RuntimeError,ValueError,OSError) as exc:
                opportunity={"status":"failed","error_type":type(exc).__name__,"query_id":q,"target_id":t,"run_signature":sig}
                torch.cuda.empty_cache()
            write_json(opportunity_path,opportunity)
        assert opportunity["run_signature"]==sig
        for condition,name in (("Real-crop","Real"),("NoE-fill","NoE")):
            if (condition,q,t) in completed:
                continue
            result={"query_id":q,"target_id":t,"condition":condition,"run_signature":sig,"opportunity_sha256":stable_sha(opportunity),"diagnostic_outside_deployment_queue":case["diagnostic_outside_deployment_queue"],"status":opportunity["status"],"cells":[],"local_pair_diagnostic":True}
            if opportunity["status"]=="ready":
                selection=opportunity["selection"]
                for position,row in enumerate(query["rows"]):
                    saved=opportunity["localized"].get(str(position))
                    cell={"row_id":row["row_id"],"position":position,"attribute_name":selection["column_name"],"value":"","status":"not_routed"}
                    if saved:
                        local=restore_localized(saved)
                        backend.condition=condition
                        backend.generation_context={"query_id":q,"target_id":t,"row_id":row["row_id"],"request_id":stable_sha([q,t,row["row_id"],condition]),"run_signature":sig}
                        previous=[r for r in backend.generation_records if r.get("request_id")==backend.generation_context["request_id"]]
                        if previous:
                            last=previous[-1]
                            cell.update(value=last.get("value") or "",status=last["status"],request_id=last["request_id"])
                        else:
                            try:
                                value=backend.generate_value(row_values(query,row),attribute_name=selection["column_name"],evidence=local,original_evidence=case["evidence"][local.evidence_id])
                                last=backend.generation_records[-1]
                                cell.update(value=value,status=last["status"],request_id=last["request_id"])
                            except (RuntimeError,ValueError,OSError) as exc:
                                cell.update(status="backend_error",error_type=type(exc).__name__)
                                torch.cuda.empty_cache()
                            write_rows(attempts_path,backend.generation_records)
                        cell["localized_evidence"]=saved
                    result["cells"].append(cell)
                target_values=column_values(target,selection["column_index"])
                result["verification"]=asdict(verifier._semantic_check([c["value"] for c in result["cells"]],target_values))
                result["status"]="completed" if all(c["status"] in ("valid_value","valid_abstain","not_routed") for c in result["cells"]) else "failed"
                result["selection"]=selection
                result["coverage_upper_bound"]=opportunity["coverage_upper_bound"]
            outputs[condition].append(result)
            write_rows(PANEL/(name+".jsonl.gz"),outputs[condition])
            completed.add((condition,q,t))
        assert len({r["request_id"] for r in backend.generation_records})<=256
        assert len(backend.generation_records)<=512
        write_json(PANEL/"EXECUTION.json",{"planned":True,"implemented":True,"executed":True,"evaluated":False,"status":"running","completed_cases":index,"first_requests":len({r["request_id"] for r in backend.generation_records}),"physical_attempts":len(backend.generation_records),"runtime_identity":sig})
        print(json.dumps({"panel_case":index,"total":32,"query_id":q,"target_id":t,"opportunity_status":opportunity["status"]}),flush=True)
    receipt={"planned":True,"implemented":True,"executed":True,"evaluated":False,"status":"generation_completed_truth_pending","cases":32,"first_requests":len({r["request_id"] for r in backend.generation_records}),"physical_attempts":len(backend.generation_records),"conditions":{c:len(rr) for c,rr in outputs.items()},"template_audit":backend.template_audit,"runtime_identity":sig}
    write_json(PANEL/"EXECUTION.json",receipt)
    return receipt


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--prepare",action="store_true");p.add_argument("--device",default="cuda:1")
    a=p.parse_args();print(json.dumps(prepare_inputs() if a.prepare else run(a.device)))
