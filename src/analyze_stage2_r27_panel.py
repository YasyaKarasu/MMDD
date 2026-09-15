"""Independent truth/grounding audit; unknown aliases remain unknown."""
from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
import json
import re

from audit_stage2_r26_cells import source_cell_truth, normalize
from mmdd_stage2.data import row_values, column_values
from mmdd_stage2.verifier import values_match
from prepare_stage1_r27 import OUT, rows, read_json, write_json, stable_sha, record
from prepare_stage2_r27 import write_rows

PANEL=OUT/"witness_panel"


def numeric(value: str):
    clean=value.strip().replace(",","").rstrip("%").strip()
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?",clean):
        return None
    return Decimal(clean)


def value_truth(value: str, source_truth: dict) -> dict:
    if not value:
        return {"value_truth":"unknown","truth_match_type":"unknown","reason":"abstention_no_claim"}
    if source_truth["status"]!="independent_source_value":
        return {"value_truth":"unknown","truth_match_type":"unknown","reason":source_truth.get("reason")}
    truth=source_truth["value"]
    if value==truth:
        return {"value_truth":"correct","truth_match_type":"exact"}
    if normalize(value)==normalize(truth):
        return {"value_truth":"correct","truth_match_type":"normalized_exact"}
    a,b=numeric(value),numeric(truth)
    if a is not None and b is not None:
        return {"value_truth":"correct" if a==b else "incorrect","truth_match_type":"independently_verified_alias" if a==b else "unknown","reason":"numeric_format_equivalence" if a==b else "distinct_numeric_values"}
    return {"value_truth":"unknown","truth_match_type":"unknown","reason":"nonmatching_surface_form_alias_not_adjudicated"}


def analyze() -> dict:
    side={(r["query_id"],r["target_id"]):r for r in rows(PANEL/"eval_truth_sidecar.jsonl.gz")}
    real={(r["query_id"],r["target_id"]):r for r in rows(PANEL/"Real.jsonl.gz")}
    noe={(r["query_id"],r["target_id"]):r for r in rows(PANEL/"NoE.jsonl.gz")}
    assert set(real)==set(noe)==set(side) and len(real)==32
    cells=[]; cases=[];deletable=[]
    manual=read_json(PANEL/"grounding_manual_review.json") if (PANEL/"grounding_manual_review.json").exists() else {}
    for key,r in real.items():
        n=noe[key];truth=side[key]
        assert r["opportunity_sha256"]==n["opportunity_sha256"]
        query=truth["original_query"];target=truth["original_target"]
        local=[]
        col_state="unknown"
        if r.get("selection"):
            col=next(c for c in target["columns"] if c["column_index"]==r["selection"]["column_index"])
            col_state="correct_attribute" if normalize(col["column_name"])==normalize(target["join_col_name"]) else "wrong_attribute"
        for condition,result in (("Real",r),("NoE",n)):
            for cell in result["cells"]:
                if cell["status"]=="not_routed":
                    continue
                row_id=cell["row_id"]
                source=source_cell_truth(query,truth["source_table"],row_id,cell["attribute_name"]) if truth["source_table"] else {"status":"unknown","reason":"source_table_missing"}
                audit=value_truth(cell["value"],source)
                source_row=next(x for x in query["rows"] if x["row_id"]==row_id)
                visible=row_values(query,source_row)
                entity=visible.get(query["query_entity_col_name"],"")
                localized=cell.get("localized_evidence",{})
                text=localized.get("text_span","")
                normalized_text=normalize(text)
                entity_present=bool(entity and normalize(entity) in normalized_text)
                value_present=bool(cell["value"] and normalize(cell["value"]) in normalized_text)
                truth_present=source["status"]=="independent_source_value" and normalize(source["value"]) in normalized_text
                # Literal presence is inspectable evidence, not automatic relation grounding.
                grounding="unknown" if localized.get("evidence_type")=="image" else "not_found" if cell["value"] and not value_present else "unknown"
                row_state="unknown"
                cell_id=stable_sha([*key,row_id,condition])
                review=manual.get(cell_id,{})
                if review:
                    grounding=review.get("grounding",grounding)
                    row_state=review.get("row_routing",row_state)
                if condition=="NoE":
                    grounding="not_found"
                rec={"cell_id":cell_id,"query_id":key[0],"target_id":key[1],"condition":condition,"row_id":row_id,"value":cell["value"],"generation_status":cell["status"],"attribute_name":cell["attribute_name"],"source_truth":source,**audit,"grounding":grounding,"column_compatibility":col_state,"row_routing":row_state,"entity":entity,"literal_entity_present":entity_present,"literal_value_present":value_present,"literal_truth_present":truth_present,"localized_evidence":localized,"grounding_review":review,"matches_target":any(values_match(cell["value"],v) for v in column_values(target,result["selection"]["column_index"])) if cell["value"] else False}
                other=next(c for c in n["cells"] if c["row_id"]==row_id) if condition=="Real" else None
                rec["new_over_NoE"]=condition=="Real" and bool(cell["value"]) and normalize(cell["value"])!=normalize(other["value"])
                if rec["new_over_NoE"] and audit["value_truth"]=="correct" and value_truth(other["value"],source)["value_truth"]=="correct":
                    rec["new_over_NoE"]=False
                if rec["new_over_NoE"] and rec["value_truth"]=="correct":
                    deletable.append(rec)
                cells.append(rec);local.append(rec)
        cases.append({"query_id":key[0],"target_id":key[1],"Real_status":r["status"],"NoE_status":n["status"],"column_compatibility":col_state,"routed_cells":sum(c["condition"]=="Real" for c in local),"coverage_upper_bound":r.get("coverage_upper_bound"),"independently_correct_Real":sum(c["condition"]=="Real" and c["value_truth"]=="correct" for c in local),"independently_correct_NoE":sum(c["condition"]=="NoE" and c["value_truth"]=="correct" for c in local),"correct_grounded_new_cells":sum(c["new_over_NoE"] and c["value_truth"]=="correct" and c["grounding"]=="supported_by_supplied_evidence" for c in local),"witness_contains_truth_literal":sum(c["condition"]=="Real" and c["literal_truth_present"] for c in local),"unknown_truth_cells":sum(c["value_truth"]=="unknown" for c in local),"diagnostic_outside_deployment_queue":r["diagnostic_outside_deployment_queue"],"Real_verification":r.get("verification"),"NoE_verification":n.get("verification")})
    attempts=list(rows(PANEL/"generation_attempts.jsonl.gz"))
    requests=Counter(r["request_id"] for r in attempts)
    assert len(requests)<=256 and len(attempts)<=512 and max(requests.values())<=2
    for req,count in requests.items():
        rr=[r for r in attempts if r["request_id"]==req]
        assert rr[0]["max_new_tokens"]==256
        if count==2:
            assert rr[0]["finish_reason"]=="length" and rr[1]["max_new_tokens"]==512
    write_rows(PANEL/"cell_truth_audit.jsonl.gz",cells)
    write_rows(PANEL/"case_funnel.jsonl.gz",cases)
    write_rows(PANEL/"deletable_cells.jsonl.gz",deletable)
    summary={"status":"truth_audited_deletion_pending" if deletable else "truth_audited_no_deletion_trigger", "cases":32,"opportunities":sum(c["condition"]=="Real" for c in cells),"generation_attempts":len(attempts),"first_requests":len(requests),"retry_count":len(attempts)-len(requests),"parse_states":dict(Counter(a["status"] for a in attempts)),"column_compatibility_cases":dict(Counter(c["column_compatibility"] for c in cases)),"cell_counts":{condition:{field:dict(Counter(c[field] for c in cells if c["condition"]==condition)) for field in ("value_truth","grounding","row_routing")} for condition in ("Real","NoE")},"independently_correct_new_cells":len(deletable),"grounded_correct_new_cells":sum(c["grounding"]=="supported_by_supplied_evidence" for c in deletable),"source_truth_policy":"exact, NFKC/case/whitespace normalization, numeric format equivalence; unresolved nonnumeric aliases remain unknown","unknown_not_scored_wrong":True,"not_population_recall":True}
    write_json(PANEL/"TRUTH_AUDIT.json",summary)
    return {**summary,"deletable":[{k:c[k] for k in ("cell_id","query_id","target_id","row_id","value","entity","localized_evidence")} for c in deletable]}


if __name__=="__main__":
    print(json.dumps(analyze(),ensure_ascii=False))
