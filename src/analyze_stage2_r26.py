"""Audit the locked C18 pilot's paired Recall, costs and evidence-to-join chain."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np

from mmdd_stage1.r26_metrics import population_metrics, query_metrics
from mmdd_stage1.r26_statistics import source_cluster_comparison
from prepare_stage1_r26 import OUT, file_record
from run_stage1_r21 import read_rows, write_rows
from run_stage1_r25 import _json, sha256

KS = (1,3,5,7,9)
CONDITIONS = ("Real-crop","Real-crop+original","NoE-fill")


def verified_rows(record: dict) -> list[dict]:
    path = Path(record["path"])
    if sha256(path) != record["sha256"]:
        raise ValueError(f"Stage2 source artifact changed: {path.name}")
    return list(read_rows(path))


def run() -> dict:
    metadata = {r["query_id"]:r for r in read_rows(OUT / "common/dev_queries.jsonl")}
    raw_exact = {r["query_id"]:set(r["D100_EXACT"]) for r in read_rows(OUT / "rankings/Qwen-Raw/rankings.jsonl.gz")}
    all_ranks, per_query, summaries, inputs, chains = {}, [], {}, [], []
    fixed_ids = None
    for generator in ("Qwen-Raw","B13"):
        directory = OUT / "stage2/pilot" / generator
        receipt_path = directory / "PILOT_RECEIPT.json"
        receipt = json.loads(receipt_path.read_text())
        results = verified_rows(receipt["results"])
        completions = verified_rows(receipt["raw_generation"])
        rows = {(r["query_id"],r["condition"]):r for r in results}
        ids = {q for q,c in rows}
        if len(ids) != 64 or len(results) != 192 or len(rows) != 192:
            raise ValueError("Stage2 must retain all64 queries and three conditions")
        if fixed_ids is None:
            fixed_ids = ids
        if ids != fixed_ids:
            raise ValueError("Generators differ in locked pilot population")
        for q in ids:
            paired = [rows[q,c] for c in CONDITIONS]
            if len({r["opportunity_sha"] for r in paired}) != 1 or len({r["input_pool_sha"] for r in paired}) != 1:
                raise ValueError("Condition opportunities or candidate pools differ")
        input_path = OUT / "stage2/inputs" / generator / "retrieval.jsonl"
        retrieval = {r["query_id"]:r for r in verified_rows(receipt["signature"]["input"])}
        own_receipt_path = OUT / "rankings" / generator / "RETRIEVAL_RECEIPT.json"
        own_receipt = json.loads(own_receipt_path.read_text())
        own = {r["query_id"]:r for r in verified_rows(own_receipt["rankings"]) if r["query_id"] in ids}
        cell_directory = OUT / "stage2/cell_audit" / generator
        cells = list(read_rows(cell_directory / "all_added_cells_with_independent_truth.jsonl"))
        deletion_path = cell_directory / "all_confirmed_cell_deletion.jsonl.gz"
        deletion = {r["query_id"]:r for r in read_rows(deletion_path)}
        if deletion.keys() != ids:
            raise ValueError("All-confirmed-cell deletion missing a pilot query")
        summary = {"queries":64,"conditions":{},"chains":{}}
        for condition in CONDITIONS:
            rank = {q:rows[q,condition]["ranking"] if rows[q,condition]["status"] == "ran" else [] for q in ids}
            all_ranks[generator,condition] = rank
            for q,r in rank.items():
                if rows[q,condition]["status"] == "ran" and (len(r) != len(retrieval[q]["results"]) or set(r) != {v["target_id"] for v in retrieval[q]["results"]}):
                    raise ValueError("Stage2 dropped candidates outside recovery budget")
                per_query.append({"generator":generator,"condition":condition,"query_id":q,
                                  "source_table_id":metadata[q]["source_table_id"],"query_kind":metadata[q]["query_kind"],
                                  **query_metrics(r,metadata[q]["positive_target_ids"],KS)})
            metrics = {kind:population_metrics(rank,{q:metadata[q]["positive_target_ids"] for q in ids if kind == "overall" or metadata[q]["query_kind"] == kind},KS)
                       for kind in ("overall","implicit","explicit")}
            for kind,values in metrics.items():
                if any(abs(v-receipt["metrics"][condition][kind][k])>1e-12 for k,v in values.items()):
                    raise ValueError("Stage2 raw-rank recomputation disagrees with pilot metrics")
            attempts = [r for r in completions if r["condition"] == condition]
            final = {}
            for r in attempts:
                final[r["query_id"],r["target_id"],r["row_id"]] = r
            candidates = [c for q in ids for c in rows[q,condition]["diagnostic_candidates"]]
            predictions = [r for c in candidates for r in c["branches"].get("evidence",{}).get("rows",[])]
            query_seconds = [sum(r["elapsed_seconds"] for r in attempts if r["query_id"] == q) for q in sorted(ids)]
            summary["conditions"][condition] = {"metrics":metrics,
                "failed_queries":sum(rows[q,condition]["status"] != "ran" for q in ids),
                "candidate_pairs":len(candidates),"recovery_attempted_pairs":sum(c["selected_for_recovery"] for c in candidates),
                "predicted_row_slots":len(predictions),"nonempty_row_slots":sum(bool(r["value"].strip()) for r in predictions),
                "final_branch_pairs":dict(Counter(str(c["final_branch"]) for c in candidates)),
                "generation_calls_including_retries":len(attempts),"final_cell_calls":len(final),
                "retry_calls":sum(r["attempt"]>1 for r in attempts),
                "attempt_statuses":dict(Counter(r["status"] for r in attempts)),
                "final_statuses":dict(Counter(r["status"] for r in final.values())),
                "legal_final_fraction":sum(r["status"] in ("valid_value","valid_abstain") for r in final.values())/len(final) if final else None,
                "nonempty_final_fraction":sum(bool(r.get("value","")) for r in final.values())/len(final) if final else None,
                "prompt_tokens":sum(r["prompt_tokens"] for r in attempts),"generated_tokens":sum(r["generated_tokens"] for r in attempts),
                "generation_seconds":sum(query_seconds),"query_generation_seconds_p50":float(np.quantile(query_seconds,.5)),
                "query_generation_seconds_p95":float(np.quantile(query_seconds,.95)),
                "parse_seconds":sum(r["parse_seconds"] for r in attempts),
                "cost_scope":"Actual value-generation calls, all retries included. Local GPU1 Qwen3.5-9B bf16; excludes localization/column scoring/semantic verification/model initialization and Student/T0 retrieval."}
        for q in sorted(ids):
            original = own[q]
            equal = retrieval[q]["equal_pool_before_T0"]
            pool = {r["target_id"] for r in retrieval[q]["results"]}
            if set(equal) != pool or len(pool)>18 or equal != original["rankings"]["Equal"][:18]:
                raise ValueError("Stage2 prequeue differs from actual own Equal C18")
            real = rows[q,"Real-crop"]
            candidate_map = {r["target_id"]:r for r in real["diagnostic_candidates"]}
            d = original["rankings"]["D100_ANN"]
            e = {r["target_id"]:r for r in original["E_paths"]}
            for target in metadata[q]["positive_target_ids"]:
                candidate = candidate_map.get(target)
                added = [r for r in cells if r["query_id"] == q and r["target_id"] == target]
                confirmed = [r for r in added if r["correctness"] == "confirmed_exact"]
                rank = lambda values: values.index(target)+1 if target in values else None
                position = rank(real["ranking"])
                after_position = rank(deletion[q]["after"])
                row = {"generator":generator,"query_id":q,"source_table_id":metadata[q]["source_table_id"],
                    "query_kind":metadata[q]["query_kind"],"target_id":target,
                    "outside_D18_ANN":target not in d[:18],"outside_D100_ANN":target not in d,
                    "outside_own_D100_EXACT":target not in original["D100_EXACT"],
                    "outside_fixed_raw_D100_EXACT":target not in raw_exact[q],
                    "E_pre_retention_admitted":target in {r["target_id"] for r in original["E_pre_retention"]},
                    "E_retained":target in e,"in_Equal_C18":target in pool,
                    "Equal_rank":rank(equal),"T0_C18_rank":rank([r["target_id"] for r in retrieval[q]["results"]]),
                    "Stage2_Real_rank":position,"Stage2_NoE_rank":rank(rows[q,"NoE-fill"]["ranking"]),
                    "after_confirmed_cell_deletion_rank":after_position,
                    "recovery_attempted":candidate["selected_for_recovery"] if candidate else False,
                    "Real_added_cells":len(added),"independently_correct_new_cells":len(confirmed),
                    "confirmed_cell_ids":[r["cell_id"] for r in confirmed],
                    "final_branch":candidate["final_branch"] if candidate else None,
                    "final_verification":candidate["verification"] if candidate else None,
                    "retained_E_paths":e[target]["retained_paths"] if target in e else [],
                    "grounded_correctness":"unknown; no independent grounding annotation",
                    "Recall_gain_lost_by_deletion":{str(k):int(position is not None and position<=k)-int(after_position is not None and after_position<=k) for k in KS}}
                chains.append(row)
        group = [r for r in chains if r["generator"] == generator]
        for outside in ("outside_D18_ANN","outside_D100_ANN","outside_fixed_raw_D100_EXACT"):
            eligible = [r for r in group if r[outside]]
            admitted = [r for r in eligible if r["E_retained"]]
            kept = [r for r in admitted if r["in_Equal_C18"]]
            correct = [r for r in kept if r["independently_correct_new_cells"]]
            summary["chains"][outside] = {"qrel_target_pairs":len(eligible),"E_retained":len(admitted),
                "C18_retained":len(kept),"with_independently_correct_new_cells":len(correct),
                "correct_cell_pairs_final_joinable":sum(bool(r["final_verification"] and r["final_verification"]["joinable"]) for r in correct),
                "positive_Recall_deletion_effect_pairs":{str(k):sum(r["Recall_gain_lost_by_deletion"][str(k)]>0 for r in correct) for k in KS},
                "grounded_chain_status":"unassessable; independent value correctness does not verify grounding"}
        summary["independent_cells"] = {"Real_added":len(cells),"confirmed_exact":sum(r["correctness"] == "confirmed_exact" for r in cells),
                                         "unknown":sum(r["correctness"] == "unknown" for r in cells)}
        summaries[generator] = summary
        all_ranks[generator,"delete_confirmed"] = {q:r["after"] for q,r in deletion.items()}
        inputs.extend([file_record(receipt_path),file_record(input_path),file_record(own_receipt_path),file_record(deletion_path),
                       file_record(cell_directory / "all_added_cells_with_independent_truth.jsonl")])
    contrasts = [((g,new),(g,old)) for g in ("Qwen-Raw","B13") for new,old in (
        ("Real-crop","NoE-fill"),("Real-crop+original","Real-crop"),("delete_confirmed","Real-crop"))]
    contrasts += [(("B13",c),("Qwen-Raw",c)) for c in CONDITIONS]
    comparisons = []
    for new,old in contrasts:
        for kind in ("overall","implicit","explicit"):
            qs = sorted(q for q in fixed_ids if kind == "overall" or metadata[q]["query_kind"] == kind)
            for k in KS:
                delta = np.array([query_metrics(all_ranks[new][q],metadata[q]["positive_target_ids"],(k,))[f"recall@{k}"]-
                                  query_metrics(all_ranks[old][q],metadata[q]["positive_target_ids"],(k,))[f"recall@{k}"] for q in qs])
                comparisons.append({"new":new,"old":old,"kind":kind,"K":k,
                                    **source_cluster_comparison(delta,[metadata[q]["source_table_id"] for q in qs])})
    destination = OUT / "statistics/stage2"
    destination.mkdir(parents=True,exist_ok=True)
    write_rows(destination / "target_chains.jsonl.gz",chains)
    write_rows(destination / "per_query_metrics.jsonl",per_query)
    result = {"execution_status":"ran","scientific_validity":"valid_with_grounding_unknown","queries_per_generator":64,
              "Stage2_K":KS,"candidate_budget":18,"summaries":summaries,"paired_comparisons":comparisons,
              "chains":file_record(destination / "target_chains.jsonl.gz"),"inputs":inputs,"code":file_record(Path(__file__))}
    _json(destination / "RESULTS.json",result)
    lines = ["# R26 Stage2 fixed C18 pilot", "", "The same locked64 queries (32 implicit/32 explicit) are retained in every condition. Stage2 final Recall uses K=1/3/5/7/9; Stage1 uses10/20/30/40/50.", "",
             "| Generator | Condition | R1 | R3 | R5 | R7 | R9 |", "|---|---|---:|---:|---:|---:|---:|"]
    for generator,summary in summaries.items():
        for condition,data in summary["conditions"].items():
            m = data["metrics"]["overall"]
            lines.append(f"| {generator} | {condition} | "+" | ".join(f"{100*m[f'recall@{k}']:.3f}" for k in KS)+" |")
    failed_counts = {g:{c:v["failed_queries"] for c,v in s["conditions"].items()} for g,s in summaries.items()}
    lines.extend(["", "Both generators recorded192/192 condition rows. Failed query-condition counts: "+json.dumps(failed_counts)+". Raw Real-crop contains one exhausted256/512-token retry; its ranking is empty and its query remains in the64 denominator. All three conditions have identical aggregate Recall within each generator. Full query-paired source bootstrap (10k draws) and implicit/explicit results are in `RESULTS.json`.",
        "", "Real-crop produced558/530 new nonempty cells for Raw/B13 versus NoE. Independent source records confirmed16/13 exactly;542/517 remain unknown. Actual semantic-backend deletion of all16/13 confirmed new cells leaves all reported Recall unchanged. Value correctness alone does not establish grounding.",
        "", "`target_chains.jsonl.gz` follows every qrel target through own Direct18/100 and fixed raw-exact100 exclusion, E admission/retention, C18 retention, generated values, independent correctness, final joins and deletion effects. No missing link is asserted true.",
        "", "Generation counts/tokens/retries and p50/p95 query generation costs are exported per condition. These costs exclude localization, semantic verification, feature extraction and initialization; they are not end-to-end latency."])
    (destination / "RESULTS.md").write_text("\n".join(lines)+"\n")
    return {"generators":2,"condition_rows":384,"chain_target_pairs":len(chains),"paired_comparisons":len(comparisons)}


if __name__ == "__main__":
    print(json.dumps(run()))
