"""The R27 intervention changes only retained E rank and preserves controls."""
from run_stage1_r27_scores import arms, equal_rank, retained_lse


def test_r27_retained_score_changes_without_membership_change():
    row = {"D100_ANN": [{"target_id":"d"}], "E_paths":[
        {"target_id":"e1","evidence_score":.9,"retained_paths":[{"path_score":.1}]},
        {"target_id":"e2","evidence_score":.2,"retained_paths":[{"path_score":2.}]},
    ]}
    result=arms(row,{"d":0.,"e1":1.,"e2":2.})
    assert result["A0"]["E_rank"]==["e1","e2"]
    assert result["A1"]["E_rank"]==["e2","e1"]
    row["E_paths"][0]["retained_paths"][0]["path_score"]=3.
    changed=arms(row,{"d":0.,"e1":1.,"e2":2.})
    assert changed["A0"]==result["A0"]
    assert changed["A1"]["E_rank"]==["e1","e2"]
    assert set(changed["A1"]["Equal"])==set(result["A1"]["Equal"])


def test_r27_empty_evidence_and_evidence_only_target():
    assert equal_rank(["d2","d1"],[])[0]==["d2","d1"]
    rank,scores=equal_rank(["d"],["e"])
    assert rank==["d","e"]
    assert scores["d"]==scores["e"]==1/61


def test_r27_qt_negative_control_when_exact_topk_is_in_union():
    scores={"t1":3.,"t2":2.,"t3":1.,"extra":-10.}
    direct=["t1","t2","t3"]
    union=direct+["extra"]
    assert sorted(union,key=lambda t:-scores[t])[:3]==direct


def test_r27_lse_retains_path_multiplicity():
    import math
    assert abs(retained_lse([{"path_score":1.},{"path_score":1.}])-(1+math.log(2)))<1e-12


def test_r27_stage2_evidence_changes_branch_and_direct_can_dominate():
    import torch
    from mmdd_stage2.verifier import semantic_joinability
    from mmdd_stage2.pipeline import CandidateResult, ColumnSelection, EvidenceVerification, DirectVerification
    def check(values):
        return semantic_joinability(values,["correct"],query_embeddings=torch.tensor([[1.,0.],[0.,0.]]),target_embeddings=torch.tensor([[1.,0.]]))
    recovered=EvidenceVerification(ColumnSelection("t",0,"attribute"),(),check(["correct",""]))
    removed=EvidenceVerification(ColumnSelection("t",0,"attribute"),(),check(["",""]))
    before=CandidateResult("t",1,0.,0.,None,None,True,None,recovered)
    after=CandidateResult("t",1,0.,0.,None,None,True,None,removed)
    assert before.semantic_joinability.coverage==.5
    assert after.semantic_joinability.coverage==0.
    direct_check=semantic_joinability(["correct","correct"],["correct"],query_embeddings=torch.tensor([[1.,0.],[1.,0.]]),target_embeddings=torch.tensor([[1.,0.]]))
    direct=DirectVerification("t",0,0,direct_check)
    dominated_before=CandidateResult("t",1,0.,0.,None,None,True,direct,recovered)
    dominated_after=CandidateResult("t",1,0.,0.,None,None,True,direct,removed)
    assert dominated_before.final_branch==dominated_after.final_branch=="direct"
    assert dominated_before.semantic_joinability.coverage==dominated_after.semantic_joinability.coverage==1.


def test_r27_truth_keeps_unadjudicated_aliases_unknown():
    from analyze_stage2_r27_panel import value_truth
    assert value_truth("17.10%",{"status":"independent_source_value","value":"17.1"})["value_truth"]=="correct"
    assert value_truth("1975",{"status":"independent_source_value","value":"1974"})["value_truth"]=="incorrect"
    assert value_truth("NYC",{"status":"independent_source_value","value":"New York City"})["value_truth"]=="unknown"
    assert value_truth("",{"status":"independent_source_value","value":"Turku"})["reason"]=="abstention_no_claim"


def test_r27_runtime_table_omits_truth_metadata():
    from run_stage2_r27_panel import visible_table
    table={"table_id":"q","hidden_attributes":["secret"],"source_table_id":"s","columns":[{"column_index":0,"column_name":"Name","source_column_index":9}],"rows":[{"row_id":0,"source_row_id":10,"cells":[{"column_index":0,"text":"Visible","source_column_index":9}]}]}
    clean=visible_table(table)
    assert clean=={"table_id":"q","columns":[{"column_index":0,"column_name":"Name"}],"rows":[{"row_id":0,"cells":[{"column_index":0,"text":"Visible"}]}]}
