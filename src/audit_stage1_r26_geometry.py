"""Actual checkpoint geometry and observed five-relation gradients over R26 trajectories."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from mmdd_stage1.checkpoints import load_student
from prepare_stage1_r26 import ROOT,OUT,file_record
from run_stage1_r21 import read_rows,write_rows
from run_stage1_r25 import _json,_r25_path_pool


def run() -> dict:
    torch.set_num_threads(4)
    directory = OUT / "statistics"
    directory.mkdir(parents=True,exist_ok=True)
    inventory = json.loads((OUT / "MODEL_INVENTORY.json").read_text())
    geometry = []
    for entry in inventory:
        if not entry["checkpoint"]:
            continue
        path = Path(entry["checkpoint"])
        if not path.exists():
            continue
        model = load_student(path,torch.device("cpu"))
        start_path = path.parent / "step_000000.pt"
        start = load_student(start_path,torch.device("cpu")) if start_path.exists() else None
        start_parameters = dict(start.named_parameters()) if start is not None else {}
        values = {}
        for key,parameter in model.named_parameters():
            if not key.startswith(("projections.","relations.")):
                continue
            if key.startswith("projections."):
                index = model.projection_keys.index(key.split(".")[1])
                anchor = model.initial_projection_weights[index]
            else:
                anchor = torch.eye(model.student_dim)
            values[key] = {"frobenius_norm":float(parameter.norm()),"distance_from_full_chain_PCA_or_identity":float((parameter-anchor).norm()),
                           "distance_from_available_stage0":float((parameter-start_parameters[key]).norm()) if key in start_parameters else None}
        geometry.append({"generator":entry["generator_id"],"checkpoint":file_record(path),"stage0":file_record(start_path),"parameters":values})
    write_rows(directory / "parameter_geometry.jsonl",geometry)
    histories = []
    for base in (ROOT / "work/stage1_optimization_r25_final_20260914/training",OUT / "training"):
        for path in sorted(base.glob("**/train_history.jsonl")):
            rows = list(read_rows(path))
            gradient_names = sorted({k for r in rows for k in r["gradient"]})
            summary = {}
            for phase,selected in (("all",rows),("first90",rows[:90]),("last88",rows[-88:])):
                summary[phase] = {name:{"observed":len(values),"mean":sum(values)/len(values),"max":max(values)}
                    for name in gradient_names for values in [[r["gradient"][name] for r in selected if r["gradient"].get(name) is not None]] if values}
            histories.append({"job":str(path.parent),"source":file_record(path),"steps":len(rows),"gradients":summary,
                              "fixed_snapshots":[r for r in rows if r["step"] in (1,89,90,91,178,330,659)]})
    _json(directory / "gradient_trajectories.json",histories)
    graph = list(read_rows(_r25_path_pool(ROOT)))
    def active(row):
        positives = set(row["positive_target_ids"])
        return any(c["target_id"] in positives and c.get("evidence_ids") for c in row["candidates"])
    activity = [bool(active(r)) for r in graph]
    order = list(read_rows(OUT / "common/c2_order.jsonl"))
    shuffled = [activity[r["source_row"]] for r in order]
    batches = [{"step":start//64+1,"size":len(activity[start:start+64]),"sequential_E_positive_active":sum(activity[start:start+64]),
                "shuffled_E_positive_active":sum(shuffled[start:start+64])} for start in range(0,len(graph),64)]
    write_rows(directory / "graph_batch_activity.jsonl",batches)
    summary = {"execution_status":"ran","scientific_validity":"valid","geometry_checkpoints":len(geometry),"gradient_histories":len(histories),
        "graph_queries":len(graph),"first5707_E_positive":sum(activity[:5707]),"last5683_E_positive":sum(activity[5707:]),
        "last88_sequential_E_positive":sum(r["sequential_E_positive_active"] for r in batches[-88:]),
        "last88_shuffled_E_positive":sum(r["shuffled_E_positive_active"] for r in batches[-88:]),
        "interpretation":"These are labels/gradients/parameter drift. Own-retrieval trajectories must establish whether admission or ranking actually worsens; absence of E positives is not absence of E candidates."}
    _json(directory / "GEOMETRY_AUDIT.json",summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(run()))
