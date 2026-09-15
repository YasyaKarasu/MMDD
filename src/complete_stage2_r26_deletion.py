"""Recompute fixed semantic verification after deleting all source-confirmed new cells."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
import json

import torch

from mmdd_stage1.r26_metrics import population_metrics
from mmdd_stage2.data import column_values,load_stage2_index
from mmdd_stage2.r26_generation import R26QwenBackend
from mmdd_stage2.verifier import semantic_joinability
from audit_stage2_r26_engineering import DATASET
from prepare_stage1_r26 import ROOT,OUT,file_record
from run_r26_followups import await_artifact
from run_stage1_r21 import read_rows,write_rows
from run_stage1_r25 import _json

KS = (1,3,5,7,9)


def rerank(candidates: list[dict]) -> list[str]:
    return [c["target_id"] for c in sorted(candidates,key=lambda c:(-c["verification"]["coverage"],-c["verification"]["mean_similarity"],c["stage1_rank"])
                                         if c["verification"] else (0.,0.,c["stage1_rank"]))]


@torch.inference_mode()
def run() -> dict:
    torch.set_num_threads(2)
    # Avoid the real image-reader memory spikes and remaining C2 jobs.
    await_artifact(OUT / "stage2/pilot/B13/PILOT_RECEIPT.json")
    await_artifact(OUT / "EXTENSION_TRAINING_RECEIPT.json")
    for generator in ("Qwen-Raw","B13"):
        await_artifact(OUT / "stage2/cell_audit" / generator / "CELL_AUDIT.json")
    backend = R26QwenBackend(ROOT / "hf_models/Qwen3.5-9B",device="cuda:1",dtype="bf16")
    population = {r["query_id"]:r for r in read_rows(OUT / "common/dev_queries.jsonl")}
    results = {}
    for generator in ("Qwen-Raw","B13"):
        directory = OUT / "stage2/cell_audit" / generator
        cells = [r for r in read_rows(directory / "all_added_cells_with_independent_truth.jsonl") if r["correctness"] == "confirmed_exact"]
        by_candidate = defaultdict(set)
        for c in cells:
            by_candidate[c["query_id"],c["target_id"]].add(c["row_id"])
        objects = load_stage2_index(DATASET,query_ids=set(),target_ids={t for q,t in by_candidate},evidence_ids=set())
        rows = [r for r in read_rows(OUT / "stage2/pilot" / generator / "results.jsonl") if r["condition"] == "Real-crop"]
        if len(rows) != 64:
            raise ValueError("Real-crop denominator changed")
        interventions = []
        for row in rows:
            candidates = deepcopy(row["diagnostic_candidates"])
            if row["status"] == "ran" and rerank(candidates) != row["ranking"]:
                raise ValueError("Frozen candidate scores do not reproduce original ranking")
            changed = []
            for candidate in candidates:
                removed = by_candidate.get((row["query_id"],candidate["target_id"]),set())
                if not removed:
                    continue
                branch = candidate["branches"]["evidence"]
                values = [r["value"] for r in branch["rows"]]
                targets = column_values(objects.targets[candidate["target_id"]],candidate["selection"]["column_index"])
                embeddings = backend.embed_texts([*values,*targets])
                before = semantic_joinability(values,targets,query_embeddings=embeddings[:len(values)],target_embeddings=embeddings[len(values):])
                difference = max(abs(before.coverage-branch["verification"]["coverage"]),abs(before.mean_similarity-branch["verification"]["mean_similarity"]))
                if difference > .001:
                    raise ValueError("Semantic backend did not reproduce baseline verification")
                after_values = ["" if r["row_id"] in removed else r["value"] for r in branch["rows"]]
                embeddings = backend.embed_texts([*after_values,*targets])
                after = semantic_joinability(after_values,targets,query_embeddings=embeddings[:len(values)],target_embeddings=embeddings[len(values):])
                changed.append({"target_id":candidate["target_id"],"deleted_rows":sorted(removed),"before":asdict(before),"after":asdict(after),"baseline_max_error":difference})
                branch["verification"] = asdict(after)
                for prediction in branch["rows"]:
                    if prediction["row_id"] in removed:
                        prediction["value"] = ""
                checks = [(name,b["verification"]) for name,b in candidate["branches"].items() if b.get("verification") is not None]
                name,best = min(checks,key=lambda item:(-item[1]["coverage"],-item[1]["mean_similarity"],item[0] != "direct"))
                candidate["final_branch"],candidate["verification"] = name,best
            interventions.append({"query_id":row["query_id"],"before":row["ranking"],"after":rerank(candidates) if row["status"] == "ran" else [],
                                  "changed_candidates":changed,"candidates":candidates})
        metrics = {kind:{method:population_metrics({r["query_id"]:r[method] for r in interventions},
            {r["query_id"]:population[r["query_id"]]["positive_target_ids"] for r in rows if kind == "overall" or population[r["query_id"]]["query_kind"] == kind},KS)
            for method in ("before","after")} for kind in ("overall","implicit","explicit")}
        write_rows(directory / "all_confirmed_cell_deletion.jsonl.gz",interventions)
        result = {"generator":generator,"execution_status":"ran","scientific_validity":"valid_source_correctness_intervention",
            "confirmed_cells_deleted":len(cells),"queries":64,"metrics":metrics,"raw":file_record(directory / "all_confirmed_cell_deletion.jsonl.gz"),
            "cell_audit":file_record(directory / "CELL_AUDIT.json"),"script":file_record(ROOT / "src/complete_stage2_r26_deletion.py"),
            "semantics":"Only independently source-confirmed Real-added values removed; actual frozen semantic backend recomputed, other candidate scores/pool/ordering fixed. Correctness of a value does not prove grounding."}
        _json(directory / "ALL_CONFIRMED_DELETION_RECEIPT.json",result)
        results[generator] = result
    return results


if __name__ == "__main__":
    print(json.dumps(run()))
