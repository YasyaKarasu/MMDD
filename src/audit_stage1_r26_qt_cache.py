"""Recompute fixed hash-selected QT training-cache pairs with the actual T0 model."""
import hashlib
import json

import torch

from mmdd_stage1.features import FeatureStore
from mmdd_stage1.teacher_rerank import _teacher_scores
from prepare_stage1_r26 import ROOT,OUT,file_record
from run_stage1_r19 import load_r19_checkpoint
from run_stage1_r21 import paths,read_rows,write_rows
from run_stage1_r25 import _json,_r25_teacher_feature_paths


@torch.inference_mode()
def run():
    torch.set_num_threads(4)
    source = ROOT / "work/stage1_optimization_r24_20260913/path_pool/teacher_target_seed13.jsonl.gz"
    rows = sorted(read_rows(source),key=lambda r:hashlib.sha256(r["query_id"].encode()).hexdigest())[:16]
    checkpoint = ROOT / "work/stage1_optimization_r22_20260911/fresh_lineage/T1-B/seed13/checkpoints/step_010536.pt"
    _,_,_,teacher,_ = load_r19_checkpoint(checkpoint,torch.device("cpu"))
    teacher.eval()
    store = FeatureStore.from_path(paths(ROOT)["features"],cache_size=1000,teacher_paths=_r25_teacher_feature_paths(ROOT))
    compression = teacher.new_compression_cache()
    results = []
    for row in rows:
        pairs = sorted(zip(row["candidate_ids"],row["direct_logits"]),key=lambda x:hashlib.sha256(x[0].encode()).hexdigest())[:4]
        values = _teacher_scores(teacher,row["query_id"],[t for t,_ in pairs],store,torch.device("cpu"),batch_size=4,compression_cache=compression)
        results.extend({"query_id":row["query_id"],"target_id":t,"cached":old,"recomputed":new,"absolute_difference":abs(old-new)} for (t,old),new in zip(pairs,values))
    maximum = max(r["absolute_difference"] for r in results)
    write_rows(OUT / "acceptance/QT_CACHE_RECOMPUTED_PAIRS.jsonl",results)
    result = {"execution_status":"ran","scientific_validity":"valid" if maximum < .001 else "invalid",
        "cache":file_record(source),"teacher":file_record(checkpoint),"pairs":len(results),"queries":len(rows),
        "max_absolute_difference":maximum,"tolerance":.001,"sampling":"SHA256(query_id) first16, SHA256(target_id) first4; independent of labels/scores",
        "scope":"Bounded numerical reexecution, not an assertion that every cached pair was recomputed"}
    _json(OUT / "acceptance/QT_CACHE_AUDIT.json",result)
    if maximum >= .001:
        raise ValueError("Cached training QT values differ from actual fixed T0")
    return result


if __name__ == "__main__":
    print(json.dumps(run()))
