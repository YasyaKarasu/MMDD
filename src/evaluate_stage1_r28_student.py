"""R28 own ANN/exact/E/U/M followed by the same frozen T0 on each own pool."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import evaluate_stage1_r26 as retrieval
import evaluate_stage1_r26_teacher as rerank
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.r28_receipts import own_index_receipt
from prepare_stage1_r27 import record, rows, sha, write_json
from prepare_stage1_r28 import ROOT, OUT, inputs
from run_stage1_r21 import write_rows

OWN = OUT / "student/own"


def prepare() -> dict:
    manifest = json.loads((OUT / "R28_PREPARED_MANIFEST.json").read_text())
    inventory = []
    for arm in manifest["arms"]:
        if not arm.startswith("S-"):
            continue
        for seed in (13,29):
            for epoch in manifest["trajectory_epochs"]:
                step = int(epoch*manifest["student"]["updates_per_epoch"])
                ck = OUT / "student" / arm / f"seed{seed}/checkpoints/step_{step:06d}.pt"
                inventory.append({"generator_id": f"{arm}/seed{seed}/epoch{epoch:g}",
                                  "checkpoint":str(ck), "arm":arm, "seed":seed, "epoch":epoch})
    write_json(OWN / "MODEL_INVENTORY.json", inventory)
    protocol = json.loads((ROOT / "work/stage1_optimization_r26_20260914/PROTOCOL.json").read_text())
    write_json(OWN / "PROTOCOL.json", {"stage1":protocol["stage1"]})
    population = ROOT / "work/stage1_optimization_r26_20260914/common/dev_queries.jsonl"
    write_rows(OWN / "common/dev_queries.jsonl", rows(population))
    return {"nodes":len(inventory), "own_output":str(OWN)}


def run(generator: str, device: str) -> dict:
    retrieval.OUT = OWN
    rerank.OUT = OWN
    result = retrieval.evaluate(generator, device, index_threads=2, legacy_diagnostics=False)
    spec = next(r for r in json.loads((OWN / "MODEL_INVENTORY.json").read_text()) if r["generator_id"] == generator)
    checkpoint = Path(spec["checkpoint"])
    model = load_student(checkpoint, torch.device("cpu"))
    receipt = own_index_receipt(checkpoint, model, OWN / "indexes" / generator,
                                Path(inputs()["feature_manifest"]["path"]))
    write_json(OUT / "student/own_pool_receipts" / (generator + ".json"), receipt)
    del model
    fixed = {r["query_id"]: set(r["positive_target_ids"]) & (set(r["E_target_ids"]) -
             (set(r["rankings"]["D100_ANN"]) | set(r["D100_EXACT"])))
             for r in rows(Path(inputs()["historical_b13_rankings"]["path"]))}
    assert sum(map(len, fixed.values())) == 207
    funnel = []
    for row in rows(OWN / "rankings" / generator / "rankings.jsonl.gz"):
        truth, q = set(row["positive_target_ids"]), row["query_id"]
        direct = set(row["rankings"]["D100_ANN"]) | set(row["D100_EXACT"])
        own = truth & (set(row["E_target_ids"]) - direct)
        sets = {"D_ANN100":set(row["rankings"]["D100_ANN"]), "D_exact100":set(row["D100_EXACT"]),
                "E":set(row["E_target_ids"]), "U":set(row["U"]), "M":set(row["M_exact"]),
                **{f"C{k}":set(row["rankings"]["Equal"][:k]) for k in (100,150,200)}}
        funnel.append({"generator_id":generator, "query_id":q, "source_table_id":row["source_table_id"],
                       "query_kind":row["query_kind"], "fixed_EO_STRICT":sorted(fixed[q]), "own_EO_STRICT":sorted(own),
                       "fixed_tracking":{k:sorted(fixed[q]&v) for k,v in sets.items()},
                       "own_tracking":{k:sorted(own&v) for k,v in sets.items()}})
    write_rows(OWN / "rankings" / generator / "eo_strict_funnel.jsonl.gz", funnel)
    teacher = rerank.run([generator], device, 0, cache_name=f"T0_pairs_{device.replace(':','')}.sqlite")
    write_json(OWN / "rankings" / generator / "R28_EVALUATION_RECEIPT.json",
               {"spec":spec, "own_index":receipt, "own_rankings":record(OWN / "rankings" / generator / "rankings.jsonl.gz"),
                "teacher":record(OWN / "teacher" / generator / "rankings.jsonl.gz"), "status":"completed"})
    return {"generator":generator, "status":"completed"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--generator")
    parser.add_argument("--device",default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(prepare() if args.prepare else run(args.generator,args.device)))
