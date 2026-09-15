"""Scientific correctness checks for the frozen R26 verifier experiment."""
import pytest
import torch
import json

from mmdd_stage2.join_diagnostic import column_metrics, multiple_positive_loss, recall, spearman
from analyze_r26_joinability_diagnostic import Scores


def test_tied_gold_column_does_not_receive_index_advantage():
    metrics = column_metrics([1., 1., 1.], [True, False, False])
    assert metrics["top1"] == pytest.approx(1/3)
    assert metrics["mrr"] == pytest.approx((1 + 1/2 + 1/3)/3)
    assert metrics["auc"] == .5
    assert column_metrics([1., 1., 1.], [False, False, True]) == metrics


def test_multiple_known_positives_and_masked_negatives():
    logits = torch.tensor([[0., 0., 10.]], requires_grad=True)
    positives = torch.tensor([[True, True, False]])
    allowed = torch.tensor([[True, True, False]])
    loss = multiple_positive_loss(logits, positives, allowed)
    assert loss == 0
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_cell_coverage_preserves_missing_rows_and_exact_override():
    scorer = Scores.__new__(Scores)
    scorer.columns = {"q": {"values": ["alpha", "", "unrelated"]}, "t": {"values": ["ALPHA", ""]}}
    scorer.cell_index = {"alpha":0, "":1, "unrelated":2, "ALPHA":3}
    scorer.cell_vectors = torch.tensor([[1.,0.],[0.,0.],[1.,0.],[0.,1.]])
    scorer.cache = {}
    result = scorer.cell("q","t")
    assert result["coverage"] == pytest.approx(1/3)
    assert result["best"] == [1.,0.,0.]
    assert result["exact"] == pytest.approx(1/3)


def test_empty_recovered_column_has_no_schema_only_support():
    scorer = Scores.__new__(Scores)
    scorer.columns = {"q": {"values": ["",""]}, "t": {"values": ["value"]}}
    assert scorer.cosine("q","t","B") == 0


def test_recall_uses_full_gold_denominator_and_counts_failures():
    assert recall(["wrong","gold1"],["gold1","gold2"],9) == .5
    assert recall([], ["gold1","gold2"],9) == 0


def test_auc_and_mrr_for_strict_column_order():
    metrics = column_metrics([.7,.8,.6],[True,False,False])
    assert metrics == {"top1":0.,"mrr":.5,"auc":.5}


def test_spearman_uses_average_tie_ranks():
    assert spearman([1,1,2],[4,4,8]) == pytest.approx(1.)
    assert spearman([1,1,1],[1,2,3]) is None


def test_calibration_ceiling_still_selects_a_trained_checkpoint(tmp_path):
    from run_r26_joinability_diagnostic import fit
    torch.set_num_threads(2)
    torch.manual_seed(17)
    vectors = {k:torch.randn(4096) for k in ("q1","q2","q3","t1","t2","t3")}
    columns = {k:{"values":[k],"source_table_id":k[-1]} for k in vectors}
    examples = [{"anchor":f"q{i}","gold":f"t{i}","candidates":[f"t{i}"],"source_table_id":str(i),
                 "split":"calibration" if i == 3 else "train"} for i in (1,2,3)]
    (tmp_path / "prepared.json").write_text(json.dumps({"columns":columns,"examples":examples}))
    torch.save(vectors,tmp_path / "column_reused.pt")
    torch.save({},tmp_path / "column_new.pt")
    fit(tmp_path,"cpu",13)
    selected = torch.load(tmp_path / "projection_seed13.pt",weights_only=False)
    initial = torch.load(tmp_path / "projection_random_seed13.pt",weights_only=False)
    assert selected["epoch"] == 1
    assert initial["epoch"] == 0
    assert not torch.equal(selected["state_dict"]["projection.weight"],initial["state_dict"]["projection.weight"])
