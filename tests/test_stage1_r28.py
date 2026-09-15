"""Tensor-level checks for the preregistered R28 actual computation path."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmdd_stage1 import r28_objectives as r28
from mmdd_stage1.data import TargetCandidate, TargetExample
from mmdd_stage1.features import FeatureStore, ObjectFeatures
from mmdd_stage1.models import StudentJoinabilityModel
from mmdd_stage1.objectives import PathAggregator, listwise_cross_entropy
from mmdd_stage1.r26_training import graph_edges
from mmdd_stage1.scoring import ListScores, TargetScores, score_target_batch, edge_positive_mask
from mmdd_stage1.training import _path_supervised_losses
from run_stage1_r19 import R19GlobalResidualTeacher


@pytest.fixture
def toy():
    torch.set_num_threads(1)
    torch.manual_seed(28)
    features = {}
    for name, kind in (("q", "table"), ("q2", "table"), ("p", "table"), ("n", "table"),
                       ("a", "text"), ("b", "text"), ("c", "image"), ("d", "image")):
        features[name] = ObjectFeatures(name, kind, torch.randn(8), torch.randn(4,8),
            torch.arange(4) if kind == "table" else None, torch.randn(3,8) if kind == "table" else None)
    store = FeatureStore(eager_features=features)
    example = TargetExample("q", (TargetCandidate("p", ("a", "c")), TargetCandidate("n", ("b", "d"))),
                            0, 0, positive_target_ids=("p",))
    known = {("q","table","table"): {"p"}, ("q","table","text"): {"a"},
             ("q","table","image"): {"c"}, ("a","text","table"): {"p"}}
    return store, example, known


def teacher():
    model = R19GlobalResidualTeacher(input_dim=8, model_dim=8, num_heads=2,
                                    num_layers=1, text_latents=2, image_latents=2, dropout=0.)
    model.cache_identity = "synthetic-r28"
    return model


@pytest.mark.parametrize("is_student", [False, True])
@pytest.mark.parametrize("family", ["SPLIT-LSE", "SPLIT-COV"])
def test_split_actual_objective_equals_branch_sum_and_differs_from_fused(toy, is_student, family):
    store, example, known = toy
    model = StudentJoinabilityModel(input_dim=8, student_dim=4) if is_student else teacher()
    scores = score_target_batch(model, [example], store, torch.device("cpu"), PathAggregator("logsumexp"))
    if family == "SPLIT-COV":
        cov = r28.coverage_scores(scores, [example], store)
        assert cov.direct is scores.direct
        scores = cov
    expected, direct, evidence = _path_supervised_losses(scores)
    terms = r28.objective(model, [example], store, torch.device("cpu"), family, known, student=is_student)
    assert torch.allclose(terms["loss"], expected + terms["weighted_anchor_loss"])
    assert torch.equal(terms["direct_supervised_loss"], direct)
    assert torch.equal(terms["evidence_supervised_loss"], evidence)
    fused = torch.logsumexp(torch.stack([scores.direct.logits, scores.evidence.logits]), dim=0)
    wrong = listwise_cross_entropy(fused, scores.direct.positive_indices, scores.direct.candidate_mask, scores.direct.positive_mask)
    assert not torch.allclose(terms["loss"] - terms["weighted_anchor_loss"], wrong)


def test_cov_detaches_support_and_backpropagates_qe_et():
    rows = torch.tensor([[1.,0.], [0.,1.]], requires_grad=True)
    ev = torch.eye(2, requires_grad=True)
    support = r28.row_support(rows, ev)
    assert not support.requires_grad
    qe = torch.tensor([.2, -.1], requires_grad=True)
    et = torch.tensor([.3, .4], requires_grad=True)
    r28.coverage_logit(qe+et, support).backward()
    assert qe.grad.abs().sum() > 0 and et.grad.abs().sum() > 0
    assert rows.grad is None and ev.grad is None


@pytest.mark.parametrize("is_student", [False, True])
def test_edge_does_not_execute_target_path_loss(toy, monkeypatch, is_student):
    store, example, known = toy
    def forbidden(*args, **kwargs):
        raise AssertionError("EDGE entered target/path computation")
    monkeypatch.setattr(r28, "score_target_batch", forbidden)
    monkeypatch.setattr(r28, "_path_supervised_losses", forbidden)
    model = StudentJoinabilityModel(input_dim=8, student_dim=4) if is_student else teacher()
    terms = r28.objective(model, [example], store, torch.device("cpu"), "EDGE", known, student=is_student)
    terms["loss"].backward()
    assert "evidence_supervised_loss" not in terms
    assert sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None) > 0


def test_present_registry_positive_is_not_negative(toy):
    store, example, known = toy
    known[("q", "table", "text")] = {"a", "b", "outside"}
    edges = graph_edges(example, known, lambda oid: store.embedding_features(oid).object_type)
    mask = edge_positive_mask(edges, max(len(e.candidate_ids) for e in edges), torch.device("cpu"))
    for i,e in enumerate(edges):
        positives = known.get((e.query_id,e.source_type,e.destination_type), set())
        for j,c in enumerate(e.candidate_ids):
            if c in positives:
                assert mask[i,j]
        assert "outside" not in e.candidate_ids


def test_shuffle_keeps_qt_scores_and_candidates(toy):
    store, example, _ = toy
    other = replace(example, query_id="q2", candidates=tuple(replace(c,evidence_ids=("b","d")) for c in example.candidates))
    original = [example, other]
    shuffled, receipt = r28.shuffled_examples(original, {"q":"g1","q2":"g2"},
        {e:store.embedding_features(e).object_type for e in ("a","b","c","d")})
    assert all(r["source_group"] != r["donor_source_group"] for r in receipt)
    model = teacher().eval()
    with torch.no_grad():
        real = score_target_batch(model, original, store, torch.device("cpu"), PathAggregator("logsumexp"))
        changed = score_target_batch(model, shuffled, store, torch.device("cpu"), PathAggregator("logsumexp"))
    torch.testing.assert_close(real.direct.logits, changed.direct.logits, rtol=0, atol=0)
    assert [[c.target_id for c in e.candidates] for e in original] == [[c.target_id for c in e.candidates] for e in shuffled]


@pytest.mark.parametrize("family", ["EDGE", "SPLIT-LSE", "SPLIT-COV"])
def test_microbatch_matches_logical_gradient_with_optional_evidence(toy, family):
    store, example, known = toy
    # Second query has no Evidence positive; this detects wrong microbatch normalization.
    inactive = replace(example, candidates=(replace(example.candidates[0],evidence_ids=()), example.candidates[1]))
    model = teacher()
    full = r28.objective(model, [example,inactive], store, torch.device("cpu"), family, known, student=False)
    full["loss"].backward()
    gradients = {n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    terms = r28.backward_batch(model, [example,inactive], store, torch.device("cpu"), family, known, student=False, microbatch=1)
    assert terms["loss"] == pytest.approx(float(full["loss"].detach()), rel=1e-5)
    for n,p in model.named_parameters():
        if n in gradients:
            torch.testing.assert_close(p.grad, gradients[n], rtol=2e-4, atol=2e-6)


def test_own_index_binds_checkpoint_bytes_and_parameters(toy, tmp_path):
    import json
    from mmdd_stage1.retrieval import build_indices
    from mmdd_stage1.r28_receipts import own_index_receipt
    from prepare_stage1_r27 import sha
    store, _, _ = toy
    model = StudentJoinabilityModel(input_dim=8, student_dim=4)
    checkpoint = tmp_path / "student.pt"
    torch.save({"format_version":1, "model_kind":"student", "config":model.config(), "state_dict":model.state_dict()}, checkpoint)
    feature = tmp_path / "features.jsonl"
    feature.write_text('{"synthetic":true}\n')
    index = tmp_path / "index"
    build_indices(model, store, {"table":["p","n"], "text":["a","b"], "image":["c","d"]},
                  index, device=torch.device("cpu"), checkpoint_sha256=sha(checkpoint), num_threads=1)
    receipt = own_index_receipt(checkpoint, model, index, feature)
    assert receipt["checkpoint"]["sha256"] == sha(checkpoint)
    with torch.no_grad():
        next(model.parameters()).add_(.01)
    with pytest.raises(ValueError, match="parameters differ"):
        own_index_receipt(checkpoint, model, index, feature)
    manifest = json.loads((index / "manifest.json").read_text())
    manifest["student_checkpoint_sha256"] = "archived-wrong-checkpoint"
    (index / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="actual checkpoint"):
        own_index_receipt(checkpoint, model, index, feature)


def test_teacher_metrics_use_macro_target_recall_and_strict_direct_union():
    from evaluate_stage1_r28_teacher import metric_rows
    meta = {"query_id":"q","source_table_id":"s","query_kind":"implicit", "positive_target_ids":["p","e"],
            "E_target_ids":["e","p"],"U":["p","e","n"], "D100_EXACT":["p"],
            "rankings":{"D100_ANN":["p"],"Equal":["p","e","n"]}}
    scores = {"D":{"p":2.,"e":1.,"n":0.}, "E-LSE":{"e":2.}, "E-COV":{"e":.3}}
    metrics = metric_rows(meta,scores,"m","Real")
    e = next(r for r in metrics if r["budget"] == "Full-U" and r["view"] == "E-LSE")
    assert e["recall"]["10"] == .5
    assert e["strict_EO_total"] == e["strict_EO_hits"]["10"] == 1
    assert e["candidate_count"] == 3 and e["rankable_count"] == 1


def test_evaluation_shuffle_preserves_modalities_and_is_gt_independent(toy):
    from evaluate_stage1_r28_teacher import shuffle_bundles
    store,ex,_ = toy
    other = replace(ex,query_id="q2",candidates=tuple(replace(c,evidence_ids=("b","d")) for c in ex.candidates))
    types = {e:store.embedding_features(e).object_type for e in ("a","b","c","d")}
    before,receipt = shuffle_bundles([ex,other],{"q":"g1","q2":"g2"},types)
    after,changed_receipt = shuffle_bundles([replace(ex,positive_target_ids=("n",)),other],{"q":"g1","q2":"g2"},types)
    assert receipt == changed_receipt
    assert all(r["source_group"] != r["donor_source_group"] for r in receipt)
    for a,b in zip([ex,other],before):
        assert [(c.target_id,sorted(types[e] for e in c.evidence_ids)) for c in a.candidates] == [(c.target_id,sorted(types[e] for e in c.evidence_ids)) for c in b.candidates]


def test_teacher_inference_chunk_without_evidence_is_finite(toy):
    from evaluate_stage1_r28_teacher import score_example
    store,ex,_ = toy
    ex = replace(ex,candidates=tuple(replace(c,evidence_ids=()) for c in ex.candidates))
    result = score_example(teacher().eval(),ex,store,torch.device("cpu"))
    assert set(result["D"]) == {"p","n"}
    assert not result["E-LSE"] and not result["E-COV"]
    assert all(torch.isfinite(torch.tensor(v)) for v in result["D"].values())


def test_bootstrap_pairs_seed_means_within_query_and_rejects_missing_queries():
    from analyze_stage1_r28 import paired_queries
    population = {"a":{"query_kind":"implicit","source_table_id":"shared"},
                  "b":{"query_kind":"explicit","source_table_id":"shared"}}
    left = {(13,"a"):1.,(29,"a"):0.,(13,"b"):0.,(29,"b"):0.}
    right = {(13,"a"):0.,(29,"a"):0.,(13,"b"):0.,(29,"b"):0.}
    delta,sources = paired_queries(left,right,population,(13,29),"overall")
    assert delta.tolist() == [.5,0.] and sources == ["shared","shared"]
    constant = {(0,"a"):0.,(0,"b"):0.}
    assert paired_queries(left,constant,population,(13,29),"overall")[0].tolist() == [.5,0.]
    del left[29,"b"]
    with pytest.raises(ValueError,match="Missing query/seed"):
        paired_queries(left,right,population,(13,29),"overall")


def test_family_summary_averages_seeds_per_query():
    from analyze_stage1_r28 import summarize
    records = [{"section":"student","arm":"S-PATH-LONG","epoch":5,"budget":"own","view":"U",
                "condition":"Real","query_kind":"implicit","source_table_id":"g",
                "query_id":q,"seed":seed,"metrics":{"RawRecall":v}}
               for q,seed,v in (("a",13,1.),("a",29,0.),("b",13,1.),("b",29,1.))]
    result = summarize(records)
    family = next(r for r in result if r["seed"] == "13+29" and r["kind"] == "overall")
    assert family["queries"] == 2 and family["value"] == .75
