"""Blind cell export, independent source-cell checks and exact-match deletion."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import unicodedata

from mmdd_dataset.wdc_runtime import iter_dataset_artifact
from mmdd_stage1.r26_metrics import population_metrics
from mmdd_stage2.data import column_values, load_stage2_index, row_values
from mmdd_stage2.verifier import values_match
from audit_stage2_r26_engineering import DATASET
from prepare_stage1_r26 import ROOT, OUT, file_record, stable_sha
from run_stage1_r21 import read_rows,write_rows
from run_stage1_r25 import _json

KS = (1,3,5,7,9)


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC",str(value)).casefold().split())


def source_cell_truth(query: dict, source: dict, row_id: int, attribute: str) -> dict:
    """Only unambiguous hidden-column names and explicitly linked source rows."""
    rows = [r for r in query["rows"] if r["row_id"] == row_id]
    attributes = [a for a in query["hidden_attributes"] if normalize(a["column_name"]) == normalize(attribute)]
    if len(rows) != 1 or len(attributes) != 1:
        return {"status":"unknown","reason":"predicted_attribute_not_unique_hidden_source_attribute"}
    query_row = rows[0]
    originals = [r for r in source["rows"] if r["row_id"] == query_row.get("source_row_id")]
    if len(originals) != 1:
        return {"status":"unknown","reason":"missing_unique_source_row_link"}
    original = originals[0]
    source_cells = {c["column_index"]:c for c in original["cells"]}
    for cell in query_row["cells"]:
        index = cell.get("source_column_index",-1)
        if index >= 0 and (index not in source_cells or normalize(cell["text"]) != normalize(source_cells[index]["text"])):
            return {"status":"unknown","reason":"visible_source_row_alignment_mismatch"}
    column = attributes[0]["source_column_index"]
    source_columns = [c for c in source["columns"] if c["column_index"] == column]
    if len(source_columns) != 1 or normalize(source_columns[0]["column_name"]) != normalize(attribute) or column not in source_cells:
        return {"status":"unknown","reason":"source_column_semantics_mismatch"}
    value = source_cells[column]["text"]
    if not normalize(value):
        return {"status":"unknown","reason":"empty_source_cell"}
    return {"status":"independent_source_value","value":value,"source_table_id":source["source_table_id"],
            "source_row_id":original["row_id"],"source_column_index":column,
            "source_record_sha":stable_sha(source),"query_record_sha":stable_sha(query)}


def run(generator: str) -> dict:
    directory = OUT / "stage2/pilot" / generator
    receipt = directory / "PILOT_RECEIPT.json"
    if not receipt.exists():
        raise ValueError("Finish the complete paired pilot before the cell audit")
    records = list(read_rows(directory / "results.jsonl"))
    grouped = {}
    for row in records:
        key = (row["query_id"],row["condition"])
        if key in grouped:
            raise ValueError("Duplicate pilot condition")
        grouped[key] = row
    ids = {q for q,c in grouped}
    if len(ids) != 64 or len(grouped) != 192:
        raise ValueError("Pilot denominator incomplete")
    for q in ids:
        paired = [grouped[q,c] for c in ("Real-crop","Real-crop+original","NoE-fill")]
        if len({r["opportunity_sha"] for r in paired}) != 1 or len({r["input_pool_sha"] for r in paired}) != 1:
            raise ValueError("Paired opportunity or pool mismatch")
    objects = load_stage2_index(DATASET,query_ids=ids,
        target_ids={c["target_id"] for r in records for c in r["diagnostic_candidates"]},evidence_ids=set())
    source_ids = {q["source_table_id"] for q in objects.queries.values()}
    sources = {s["source_table_id"]:s for s in iter_dataset_artifact(DATASET,"source_tables") if s["source_table_id"] in source_ids}
    if sources.keys() != source_ids:
        raise ValueError("Source records missing")
    population = {r["query_id"]:r for r in read_rows(OUT / "common/dev_queries.jsonl") if r["query_id"] in ids}
    cells = []
    for q in sorted(ids):
        real, no_e = grouped[q,"Real-crop"], grouped[q,"NoE-fill"]
        if real["status"] != "ran" or no_e["status"] != "ran":
            continue
        no_values = {(c["target_id"],r["row_id"]):r["value"] for c in no_e["diagnostic_candidates"]
                     for r in c["branches"].get("evidence",{}).get("rows",[])}
        query = objects.queries[q]
        for candidate in real["diagnostic_candidates"]:
            for row in candidate["branches"].get("evidence",{}).get("rows",[]):
                value = row["value"]
                no_value = no_values.get((candidate["target_id"],row["row_id"]),"")
                if not value.strip() or normalize(value) == normalize(no_value):
                    continue
                attribute = candidate["selection"]["column_name"]
                truth = source_cell_truth(query,sources[query["source_table_id"]],row["row_id"],attribute)
                confirmed = truth["status"] == "independent_source_value" and normalize(value) == normalize(truth["value"])
                target_values = column_values(objects.targets[candidate["target_id"]],candidate["selection"]["column_index"])
                cells.append({"cell_id":stable_sha([generator,q,candidate["target_id"],row["row_id"],attribute]),
                    "generator_id":generator,"query_id":q,"target_id":candidate["target_id"],"row_id":row["row_id"],
                    "attribute":attribute,"value":value,"NoE_value":no_value,"localized_evidence":row["evidence"],
                    "visible_query_row":row_values(query,next(r for r in query["rows"] if r["row_id"] == row["row_id"])),
                    "truth":truth,"correctness":"confirmed_exact" if confirmed else "unknown",
                    "matches_selected_target_value":any(values_match(value,v) for v in target_values),
                    "target_is_qrel_positive":candidate["target_id"] in population[q]["positive_target_ids"],
                    "grounding":"unknown; source correctness does not establish use of evidence"})
    cells.sort(key=lambda r:hashlib.sha256(r["cell_id"].encode()).hexdigest())
    destination = OUT / "stage2/cell_audit" / generator
    destination.mkdir(parents=True,exist_ok=True)
    write_rows(destination / "all_added_cells_with_independent_truth.jsonl",cells)
    blind_keys = ("cell_id","visible_query_row","attribute","value","localized_evidence","target_id")
    write_rows(destination / "blind_review_32.jsonl",[{k:r[k] for k in blind_keys} for r in cells[:32]])
    confirmed = [r for r in cells if r["correctness"] == "confirmed_exact"]
    exact = [r for r in confirmed if r["matches_selected_target_value"]]
    # In production semantic_joinability, an exact target match contributes 1
    # to similarity and coverage. Removing it contributes 0, with the same row
    # denominator; all other row contributions are independent and unchanged.
    to_delete = {}
    for cell in exact:
        to_delete.setdefault((cell["query_id"],cell["target_id"]),set()).add(cell["row_id"])
    interventions = []
    for q in sorted(ids):
        row = grouped[q,"Real-crop"]
        candidates = deepcopy(row["diagnostic_candidates"])
        for candidate in candidates:
            removed = to_delete.get((q,candidate["target_id"]),set())
            if not removed:
                continue
            branch = candidate["branches"]["evidence"]
            delta = len(removed)/len(branch["rows"])
            check = branch["verification"]
            check["coverage"] = max(0.,check["coverage"]-delta)
            check["mean_similarity"] -= delta
            check["joinable"] = check["coverage"] >= .6
            for prediction in branch["rows"]:
                if prediction["row_id"] in removed:
                    prediction["value"] = ""
            checks = [(name,b["verification"]) for name,b in candidate["branches"].items() if b.get("verification") is not None]
            name,best = min(checks,key=lambda item:(-item[1]["coverage"],-item[1]["mean_similarity"],item[0] != "direct"))
            candidate["final_branch"],candidate["verification"] = name,best
        candidates.sort(key=lambda c:(-c["verification"]["coverage"],-c["verification"]["mean_similarity"],c["stage1_rank"]) if c["verification"] else (0.,0.,c["stage1_rank"]))
        interventions.append({"query_id":q,"before":row["ranking"],"after":[] if row["status"] != "ran" else [c["target_id"] for c in candidates],
                              "deleted_cell_ids":[c["cell_id"] for c in exact if c["query_id"] == q],"candidates":candidates})
    write_rows(destination / "exact_cell_deletion.jsonl.gz",interventions)
    metrics = {kind:{method:population_metrics({r["query_id"]:r[method] for r in interventions},
        {q:r["positive_target_ids"] for q,r in population.items() if kind == "overall" or r["query_kind"] == kind},KS)
        for method in ("before","after")} for kind in ("overall","implicit","explicit")}
    summary = {"generator":generator,"execution_status":"ran","paired_opportunities_valid":True,"queries":64,
        "added_cells":len(cells),"independently_confirmed_exact_cells":len(confirmed),"correctness_counts":dict(Counter(r["correctness"] for r in cells)),
        "blind_export_cells":min(32,len(cells)),"deleted_exact_target_match_cells":len(exact),
        "remaining_confirmed_nonexact_target_cells":[r["cell_id"] for r in confirmed if not r["matches_selected_target_value"]],
        "deletion_metrics":metrics,"grounded_correctness":"unknown; no automated judger treated as GT",
        "deletion_scope":"Analytically exact production-score deletion for independently correct cells that exactly match a selected target value; remaining semantic-only matches require backend recomputation",
        "sources":{"pilot":file_record(receipt),"results":file_record(directory / "results.jsonl"),"script":file_record(ROOT / "src/audit_stage2_r26_cells.py")}}
    _json(destination / "CELL_AUDIT.json",summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator",choices=("Qwen-Raw","B13"),required=True)
    print(json.dumps(run(parser.parse_args().generator)))
