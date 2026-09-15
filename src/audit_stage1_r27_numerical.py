"""Three label-free hash-selected queries: own QT/ANN/QE/ET and D1 checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from evaluate_stage1_r26 import retain_evidence
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import StudentANNIndices, RawEmbeddingANNIndices
from mmdd_stage1.row_support import load_evidence_content_keys
from prepare_stage1_r26 import parameter_sha
from prepare_stage1_r27 import ROOT, OUT, R26, rows, read_json, record, write_json, sha
from run_stage1_r27_scores import retained_lse


@torch.inference_mode()
def run(device_name: str) -> dict:
    torch.set_num_threads(2)
    device = torch.device(device_name)
    pop = list(rows(R26 / "common/dev_queries.jsonl"))
    panel = sorted((r["query_id"] for r in pop),key=lambda q:hashlib.sha256((q+"|R27-numerical").encode()).hexdigest())[:3]
    write_json(OUT / "audit/numerical_spotcheck/PROTOCOL.json", {"queries":panel,"selection":"sha256(query_id|R27-numerical) ascending, no labels","gpu_atol":1e-5,"float64_atol":1e-8,"scope":"three-query correctness; not global ANN error estimation"})
    features = ROOT / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b"
    store = FeatureStore.from_path(features,cache_size=60000)
    content,_ = load_evidence_content_keys(ROOT / "work/stage1_optimization_r10_20260907/taskB_g5/evidence_content_keys.jsonl")
    table_path = Path(read_json(R26 / "rankings/Qwen-Raw/INDEX_RECEIPT.json")["index_dir"]) / "table_ids.json"
    tables = read_json(table_path)
    assert len(tables)==22886 and not set(panel)&set(tables)
    store.preload_embeddings([*tables,*panel])
    embeddings = torch.stack([store.embedding_features(t).embedding for t in tables]).to(device)
    qemb = torch.stack([store.embedding_features(q).embedding for q in panel]).to(device)
    positions = {t:i for i,t in enumerate(tables)}
    outputs = []
    for spec in read_json(ROOT / "mmdd_r26_review/R27_INPUT_LOCK.json")["models"]:
        name = spec["generator_id"]
        dest = OUT / "audit/numerical_spotcheck" / name
        rr = read_json(R26 / "rankings" / name / "RETRIEVAL_RECEIPT.json")
        ir = read_json(R26 / "rankings" / name / "INDEX_RECEIPT.json")
        model = load_student(Path(spec["checkpoint"]),device).eval() if spec["checkpoint"] else None
        assert rr["signature"]["parameter_sha256"] == (parameter_sha(model) if model else "raw_no_parameters")
        vectors = torch.cat([model.index_vector(b,"table") for b in embeddings.split(4096)]) if model else embeddings
        qvec = model.relation_query(qemb,"table","table") if model else qemb
        matrix = (qvec@vectors.T).cpu()
        selected = {r["query_id"]:r for r in rows(R26 / "rankings" / name / "rankings.jsonl.gz") if r["query_id"] in panel}
        indices = StudentANNIndices(model,store,Path(ir["index_dir"]),device=device,checkpoint_sha256=rr["signature"]["checkpoint_sha256"],corpus_sha256=rr["signature"]["corpus_sha256"]) if model else RawEmbeddingANNIndices(store,Path(ir["index_dir"]),corpus_sha256=rr["signature"]["corpus_sha256"])
        for idx in indices.indices.values():
            idx.set_num_threads(2)
        checked=[]
        for qi,q in enumerate(panel):
            r=selected[q]
            qt=matrix[qi]
            qt_error=max(abs(float(qt[positions[t]])-score) for t,score in r["QT_OVER_U_scores"].items())
            mpos=[positions[t] for t in r["M_exact"]]
            m_error=float((qt[mpos]-torch.tensor(r["exact_scores"])).abs().max())
            excluded=qt.clone();excluded[mpos]=-torch.inf
            boundary=float(excluded.max()-qt[mpos].min())
            ann = indices.search(q,"table",100)
            assert [t for t,s in ann] == [x["target_id"] for x in r["D100_ANN"]], (name,q,"ANN IDs")
            ann_error=max(abs(s-x["direct_score"]) for (t,s),x in zip(ann,r["D100_ANN"]))
            retained=retain_evidence(q,{"evidence":r["E_pre_retention"]},store,content)
            assert [x["target_id"] for x in retained] == r["E_target_ids"]
            d1_error=0.;lse_error=0.;path_checks=[]
            for before,after in zip(r["E_paths"],retained):
                assert before["selected_evidence_ids"]==after["selected_evidence_ids"] and before["routed_rows"]==after["routed_rows"] and before["retained_paths"]==after["retained_paths"]
                d1_error=max(d1_error,abs(before["evidence_score"]-after["evidence_score"]))
                lse_error=max(lse_error,abs(before["retained_path_lse"]-retained_lse(before["retained_paths"])))
            # Fixed target-hash-selected examples, checking every retained witness there.
            for target in sorted(r["E_paths"],key=lambda x:hashlib.sha256(x["target_id"].encode()).hexdigest())[:3]:
                for path in target["retained_paths"]:
                    def score(a,b,at,bt):
                        aa=store.embedding_features(a).embedding.unsqueeze(0).to(device)
                        bb=store.embedding_features(b).embedding.unsqueeze(0).to(device)
                        return float(((model.relation_query(aa,at,bt) if model else aa)*(model.index_vector(bb,bt) if model else bb)).sum())
                    qe=score(q,path["evidence_id"],"table",path["evidence_type"])
                    et=score(path["evidence_id"],target["target_id"],path["evidence_type"],"table")
                    err=max(abs(qe-path["query_evidence_score"]),abs(et-path["evidence_target_score"]),abs(qe+et-path["path_score"]))
                    path_checks.append({"evidence_id":path["evidence_id"],"target_id":target["target_id"],"QE":qe,"ET":et,"path":qe+et,"max_abs_error":err})
            checks={"QT":qt_error,"M_scores":m_error,"M_boundary":max(0,boundary),"ANN":ann_error,"QE_ET_path":max((x["max_abs_error"] for x in path_checks),default=0.)}
            assert max(checks.values())<=1e-5,(name,q,checks)
            assert d1_error<=1e-8 and lse_error<=1e-8
            checked.append({"query_id":q,"full_lake_tables":len(tables),"errors":checks,"D1_error":d1_error,"retained_LSE_error":lse_error,"path_checks":path_checks,"ANN_ids_equal":True})
        result={"status":"pass","generator":name,"queries":checked,"checkpoint":record(Path(spec["checkpoint"])) if spec["checkpoint"] else None,"retrieval_receipt":record(R26/"rankings"/name/"RETRIEVAL_RECEIPT.json"),"script":record(Path(__file__))}
        write_json(dest/"AUDIT.json",result)
        outputs.append({"generator":name,"status":"pass"})
        print(json.dumps(outputs[-1]),flush=True)
        del model,indices,vectors,matrix
    write_json(OUT/"audit/numerical_spotcheck.json",{"status":"pass","models":outputs,"features":record(features/"manifest.jsonl")})
    return {"models":len(outputs)}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--device",default="cuda:0")
    print(json.dumps(run(p.parse_args().device)))
