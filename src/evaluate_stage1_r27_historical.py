"""Own-pool evaluation of preregistered H nodes, using the frozen R26 protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import evaluate_stage1_r26 as retrieval
import evaluate_stage1_r26_teacher as teacher
from prepare_stage1_r27 import ROOT, OUT, R26, read_json, write_json, record, sha


def run(nodes: list[str], device: str, score_teacher: bool) -> dict:
    dest = OUT / "historical_replay/own_evaluation"
    write_json(dest / "PROTOCOL.json",read_json(R26 / "PROTOCOL.json"))
    (dest / "common").mkdir(parents=True,exist_ok=True)
    source=R26/"common/dev_queries.jsonl"
    target=dest/"common/dev_queries.jsonl"
    if not target.exists():
        target.write_bytes(source.read_bytes())
    assert sha(target)==sha(source)
    specs=[]
    for stage,steps in (("C1",(0,178,356)),("C2",(0,89,178))):
        for step in steps:
            p=OUT/f"historical_replay/{stage}/seed13/checkpoints/step_{step:06d}.pt"
            specs.append({"generator_id":f"H-{stage}-step{step:06d}","checkpoint":str(p),"seed":13,"step":step})
    write_json(dest/"MODEL_INVENTORY.json",specs)
    retrieval.OUT=dest
    teacher.OUT=dest
    results=[]
    for name in nodes:
        spec=next(s for s in specs if s["generator_id"]==name)
        checkpoint=Path(spec["checkpoint"])
        assert checkpoint.exists(),checkpoint
        node=read_json(checkpoint.with_suffix(".json"))
        assert sha(checkpoint)==node["checkpoint"]["sha256"]
        if score_teacher:
            result=teacher.run([name],device,0)
        else:
            result=retrieval.evaluate(name,device,index_threads=2)
        write_json(dest/"node_receipts"/(name+("_teacher" if score_teacher else "")+".json"),{"status":"completed","H_node":node["checkpoint"],"source_protocol":record(R26/"PROTOCOL.json"),"evaluation_type":"own_C100_T0" if score_teacher else "own_new_index_D_E_U_M", "result":result})
        results.append(name)
        print(json.dumps({"completed":name,"teacher":score_teacher}),flush=True)
    return {"completed":results}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--node",action="append",required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--teacher",action="store_true")
    a=p.parse_args();print(json.dumps(run(a.node,a.device,a.teacher)))
