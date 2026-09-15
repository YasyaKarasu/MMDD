"""Behavior checks for the production R26 metrics and retrieval wiring."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmdd_stage1.r26_metrics import fuse_channels, population_metrics, query_metrics


def test_recall_differs_from_hit_and_failed_queries_keep_denominator():
    result = population_metrics({"q1": ["a"]}, {"q1": ["a", "b"], "failed": ["c"]}, (1, 3, 5, 7, 9))
    assert result["recall@1"] == .25
    assert result["any_hit_at_1"] == .5
    assert result["recall@9"] == .25


def test_duplicates_short_lists_and_empty_qrels():
    assert query_metrics(["a", "a", "b"], ["a", "b"], (2, 50))["recall@2"] == 1
    assert query_metrics(["a", "a"], ["a", "b"], (50,))["recall@50"] == .5
    with pytest.raises(ValueError):
        query_metrics([], [], (1,))


def test_stage1_reporting_computes_requested_k_and_keeps_failed_queries():
    from report_stage1_r26_recall import KS,summarize
    assert KS == (10,20,30,40,50)
    ranking = [f"n{i}" for i in range(50)]
    ranking[29],ranking[39] = "a","b"
    rows = [{"query_id":"q","query_kind":"implicit","positive_target_ids":["a","b"],"rankings":{"QT_OVER_U":ranking}},
            {"query_id":"failed","query_kind":"explicit","positive_target_ids":["c"],"rankings":{}}]
    result = summarize(rows,{r["query_id"]:r for r in rows})["overall"]["QT_OVER_U"]
    assert [result[f"recall@{k}"] for k in KS] == [0.,0.,.25,.5,.5]
    assert "recall@1" not in result


def test_teacher_cache_reuses_pairs_but_invalidates_changed_features_and_teacher(tmp_path):
    import torch
    from mmdd_stage1.features import FeatureStore, ObjectFeatures
    from mmdd_stage1.r26_teacher import TeacherPairCache

    class Teacher(torch.nn.Module):
        compute_dtype = torch.float32

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.calls = 0

        def new_compression_cache(self):
            return {}

        def score_pairs(self, queries, targets, compression_cache):
            self.calls += len(targets)
            return torch.tensor([float((q.hidden_states*t.hidden_states).sum()) for q,t in zip(queries,targets)])

    def store(value):
        return FeatureStore({name: ObjectFeatures(name, "table", torch.ones(2), torch.full((1,2),v))
                             for name,v in (("q",1.),("a",value),("b",3.))})

    teacher = Teacher()
    path = tmp_path / "pairs.sqlite"
    cache = TeacherPairCache(path, "T0", teacher, store(2.), torch.device("cpu"))
    assert cache.score("q", ["a"])[0] == {"a": 4.}
    assert cache.score("q", ["b", "a"])[1]["new_pairs"] == 1
    assert teacher.calls == 2
    cache.db.close()
    cache = TeacherPairCache(path, "T0", teacher, store(4.), torch.device("cpu"))
    scores, cost = cache.score("q", ["a", "b"])
    assert scores == {"a": 8., "b": 6.}
    assert cost["new_pairs"] == 1
    cache.db.close()
    cache = TeacherPairCache(path, "T1", teacher, store(4.), torch.device("cpu"))
    assert cache.score("q", ["a", "b"])[1]["new_pairs"] == 2
    cache.db.close()


def test_teacher_uses_each_own_pool_and_keeps_prequeue_membership():
    from mmdd_stage1.r26_teacher import rerank_pools
    row = {"rankings": {"Equal": ["d", "e"], "D100_ANN": ["d"]}, "U": ["d", "e"], "M_exact": ["d", "m"]}
    ranks = rerank_pools(row, {"d": 0., "e": 3., "m": 2.})
    assert ranks["BT100_NO_T0"] == ["d", "e"]
    assert ranks["BT100_T0"] == ["e", "d"]
    assert ranks["D100_T0"] == ["d"]
    assert ranks["M_OFFLINE_T0"] == ["m", "d"]
    row["U"] = ["d", "new"]
    row["rankings"]["Equal"] = ["d", "new"]
    assert rerank_pools(row, {"d":0., "new":5., "m":2.})["U_OFFLINE_T0"] == ["new", "d"]


def test_extension_gate_requires_both_seeds_and_preserves_missing_inputs():
    from mmdd_stage1.r26_statistics import loss_extension_gate

    def result(e, u=.4):
        bucket = {"admission":{"EO_ANN":e},"QT_OVER_U":{"recall@10":.3},"U":{"raw_recall":u}}
        return {"overall":bucket,"implicit":bucket}

    inputs = {f"R25-SPLIT-SUP/seed{s}/step178":result(.1) for s in (13,29)}
    inputs["R26-O-SUP/seed13/step178"] = result(.12)
    assert loss_extension_gate(inputs)["status"] == "unassessable"
    inputs["R26-O-SUP/seed29/step178"] = result(.1)
    assert loss_extension_gate(inputs)["status"] == "unassessable"
    inputs["B13"] = result(.2,.8)
    assert loss_extension_gate(inputs)["status"] == "not_triggered"
    inputs["R26-O-SUP/seed29/step178"] = result(.12)
    assert loss_extension_gate(inputs)["status"] == "triggered"


def test_feedback_gate_respects_priority_health_and_actual_membership():
    from mmdd_stage1.r26_feedback import feedback_gate,PRIORITY
    def metric(u=.7,e=.2):
        return {"overall":{"U":{"raw_recall":u}},"implicit":{"admission":{"EO_ANN":e}}}
    metrics = {"B13":metric()}
    for arm in PRIORITY[1:]:
        for seed in (13,29):
            metrics[f"R26-{arm}/seed{seed}/step178"] = metric()
    changed = {name:1 for name in metrics if name != "B13"}
    assert feedback_gate(metrics,changed)["reason"] == "missing_priority_own_retrieval"
    for seed in (13,29):
        metrics[f"R26-O-UQTKD/seed{seed}/step178"] = metric()
    assert feedback_gate(metrics,changed)["reason"] == "missing_actual_H_lists"
    changed.update({f"R26-O-UQTKD/seed{s}/step178":0 for s in (13,29)})
    assert feedback_gate(metrics,changed)["selected_arm"] == "O-QTKD"
    changed["R26-O-UQTKD/seed29/step178"] = 1
    assert feedback_gate(metrics,changed)["selected_arm"] == "O-UQTKD"
    metrics["R26-O-UQTKD/seed13/step178"] = metric(.679)
    assert feedback_gate(metrics,changed)["selected_arm"] == "O-QTKD"


def test_feedback_augmentation_preserves_repeated_queries_relations_and_closure():
    from mmdd_stage1.data import EdgeExample
    from mmdd_stage1.r26_feedback import augment_teacher_lists
    examples = [EdgeExample("q",("p","n"),0,source_type="table",destination_type="table",positive_ids=("p",)),
                EdgeExample("q",("p","m"),0,source_type="table",destination_type="table",positive_ids=("p",)),
                EdgeExample("q",("e","unknown"),0,source_type="table",destination_type="text",positive_ids=("e",))]
    rows,audit = augment_teacher_lists(examples,{"q":["p","new","other_positive"]},{("q","table","table"):{"p","n","other_positive"}})
    assert len(rows) == 3
    assert rows[0].candidate_ids == ("p","n","new")
    assert rows[1].candidate_ids == ("p","m","new")
    assert rows[0].positive_ids == ("p","n")
    assert rows[0].confirmed_labels == (1,1,None)
    assert rows[2].candidate_ids == examples[2].candidate_ids
    assert audit["historical_or_mined_hard_positives_filtered"] == 4
    with pytest.raises(ValueError,match="Every train QT"):
        augment_teacher_lists(examples,{}, {})


def test_actual_refinement_preparation_roundtrips_all_six_training_inputs(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    import prepare_stage1_r26_refinement as driver
    from mmdd_stage1.data import load_edge_examples
    from run_stage1_r21 import write_rows
    from run_stage1_r25 import sha256

    r22,output = tmp_path/"r22",tmp_path/"output"
    monkeypatch.setattr(driver,"R22",r22)
    monkeypatch.setattr(driver,"OUT",output)
    examples = [{"query_id":q,"candidate_ids":[p,n],"positive_id":p,"positive_ids":[p],
                 "source_type":src,"destination_type":dst,"split":"train"}
                for q,p,n,src,dst in [("q","p","n","table","table"),
                    ("q","e","en","table","text"),("e","p","n","text","table"),
                    ("q","i","in","table","image"),("i","p","n","image","table")]]
    write_rows(r22/"manifests/full_natural.jsonl",examples)
    historical = r22/"fresh_lineage/S0_mining/seed13/hard_negatives.jsonl.gz"
    write_rows(historical,[{"query_id":"q","hard_candidate_ids":["old"]}])
    write_rows(output/"common/feedback_queries.jsonl",[{"query_id":"q"}])
    names = ["new/seed13","new/seed29"]
    gate = output/"feedback/REFINEMENT_GATE_FROZEN.json"
    gate.parent.mkdir(parents=True,exist_ok=True)
    gate.write_text(json.dumps({"status":"triggered","selected_generators":names}))
    protocol = {"list_budget":5,"historical_hard":driver.file_record(historical)}
    (output/"feedback/REFINEMENT_PROTOCOL.json").write_text(json.dumps(protocol))
    monkeypatch.setattr(driver,"freeze_protocol",lambda:protocol)
    known = {(r["query_id"],r["source_type"],r["destination_type"]):{r["positive_id"]} for r in examples}
    monkeypatch.setattr(driver,"known_edges",lambda:known)
    for name in ["B13",*names]:
        write_rows(output/"feedback"/name/"hard_lists.jsonl.gz",[{"query_id":"q","hard32":[name+"-hard"]}])
    assert driver.prepare(False) == {"prepared_jobs":6}
    orders = {}
    for arm in ("Tcont","Told","Tnew"):
        for seed in (13,29):
            path = output/"feedback/refinement"/arm/f"seed{seed}"/"LISTS_RECEIPT.json"
            receipt = json.loads(path.read_text())
            lists_path = Path(receipt["lists"]["path"])
            assert sha256(lists_path) == receipt["lists"]["sha256"]
            rows = load_edge_examples(lists_path,split="train")
            assert receipt["list_count"] == len(rows) == 5
            assert rows[0].candidate_ids[:2] == ("p","n") and len(rows[0].candidate_ids) == 3
            assert all(row.candidate_ids == tuple(raw["candidate_ids"]) for row,raw in zip(rows[1:],examples[1:]))
            assert receipt["non_QT_candidate_lists_changed"] == 0
            orders.setdefault(seed,receipt["order"]["sha256"])
            assert orders[seed] == receipt["order"]["sha256"]


def test_source_bootstrap_preserves_query_weight_with_unequal_cluster_sizes():
    import numpy as np
    from mmdd_stage1.r26_statistics import source_cluster_comparison
    # Two sources, one with three queries. Query-macro mean is .25, not .5.
    result = source_cluster_comparison(np.array([1.,0.,0.,0.]),["a","b","b","b"],replicates=1000)
    assert result["mean_delta"] == .25
    assert result["source_clusters"] == 2
    assert result["queries"] == 4
    assert result["wins"] == 1 and result["ties"] == 3
    assert result["bootstrap_95ci"] == [0.,1.]


def test_teacher_then_fusion_changes_e_order_without_changing_pool():
    from evaluate_r26_b13_teacher_fusion import teacher_then_fusion
    row = {"D100_ANN":[{"target_id":"d","direct_score":2.}],
           "E_paths":[{"target_id":"e1","evidence_score":3.},{"target_id":"e2","evidence_score":1.}],
           "QT_OVER_U_scores":{"d":2.,"e1":0.,"e2":1.}}
    result = teacher_then_fusion(row,{"d":0.,"e1":1.,"e2":3.},["d","e1","e2"],{"e1":.5,"e2":.5})
    assert result["E_ids_before_T0"] == ["e1","e2"]
    assert result["E_ids_after_T0"] == ["e2","e1"]
    for ranking in result["rankings"].values():
        assert set(ranking) == {"d","e1","e2"}
    assert result["fusion_scores"]["T0_D_T0_E"]["Equal"]["e2"] > result["fusion_scores"]["T0_D_original_E"]["Equal"]["e2"]


def test_column_reference_gives_equal_weight_to_queries_and_preserves_missing():
    import torch
    from mmdd_stage1.r26_column import ColumnSimilarity,QueryBalancedCDF
    cdf = QueryBalancedCDF([[0.],[1.,1.,1.]])
    assert cdf.alpha(.5) == .5
    assert cdf.alpha(1.) == pytest.approx(0.)
    assert cdf.alpha(None) == 1.
    scorer = ColumnSimilarity({"query:q:0":torch.tensor([1.,0.]),"target:t:0":torch.tensor([0.,2.]),
                               "target:t:1":torch.tensor([3.,0.])},torch.device("cpu"))
    assert scorer.scores("q",["t","missing"]) == {"t":1.,"missing":None}


@pytest.mark.parametrize("e_positive",([True,True],[False,False],[True,False]))
@pytest.mark.parametrize("kd_weight",[0.,.3])
def test_extension_lse_weights_each_query_before_mean(e_positive,kd_weight):
    import torch
    from mmdd_stage1.scoring import ListScores,TargetScores
    from mmdd_stage1.r26_extension_objectives import objective
    d = torch.tensor([[3.,0.],[0.,3.]],requires_grad=True)
    e = torch.tensor([[1.,0.],[1.,4.]],requires_grad=True)
    mask = torch.ones_like(d,dtype=torch.bool)
    pos = torch.tensor([[True,False],[True,False]])
    ep = pos & torch.tensor(e_positive)[:,None]
    indices = torch.zeros(2,dtype=torch.long)
    scores = TargetScores(ListScores(d,mask,indices,pos),ListScores(e,mask,indices,ep))
    teacher_logits = torch.tensor([[0.,2.],[3.,0.]])
    teacher = TargetScores(*(ListScores(teacher_logits,mask,indices,p) for p in (pos,ep)))
    terms = objective(scores,family="lse",teacher=teacher,kd_weight=kd_weight)
    fused = torch.logsumexp(torch.stack((d,e),-1),-1)
    losses = torch.logsumexp(fused,-1)-fused[:,0]
    units = 1+torch.tensor(e_positive).float()
    logt = torch.log_softmax(teacher_logits,-1)
    per_kd = (logt.exp()*(logt-torch.log_softmax(fused,-1))).sum(-1)
    expected = (units*losses).mean()+kd_weight*(units*per_kd).mean()
    assert torch.allclose(terms["fused_kd_loss"],(units*per_kd).mean())
    assert torch.allclose(terms["fused_supervised_loss"],(units*losses).mean())
    assert float(terms["loss"].detach()) == pytest.approx(float(expected.detach()))
    actual_grad = torch.autograd.grad(terms["loss"],(d,e),retain_graph=True)
    expected_grad = torch.autograd.grad(expected,(d,e))
    for actual,want in zip(actual_grad,expected_grad):
        assert torch.allclose(actual,want)


def test_extension_uniform_keeps_qt_zero_and_per_query_frozen_denominator():
    import torch
    from mmdd_stage1.scoring import ListScores
    from mmdd_stage1.r26_extension_objectives import query_uniform
    logits = torch.tensor([[3.,0.],[2.,0.],[2.,0.],[2.,0.],[2.,0.]],requires_grad=True)
    mask = torch.ones_like(logits,dtype=torch.bool)
    scores = ListScores(logits,mask,torch.zeros(5,dtype=torch.long),torch.tensor([[True,False]]*5))
    # q0 has QT plus QE/ET lists (3 denominator); q1 has two lists (2 denominator).
    value = query_uniform(scores,["table->table","table->text","text->table","table->image","image->table"],[0,0,0,1,1],2)
    single = -torch.log_softmax(logits[1],0).mean()-torch.log(torch.tensor(2.))
    assert float(value.detach()) == pytest.approx(float((single*(2/3+1)/2).detach()))
    value.backward()
    assert torch.count_nonzero(logits.grad[0]) == 0
    assert all(torch.count_nonzero(g) > 0 for g in logits.grad[1:])


@pytest.mark.parametrize("family",["split","lse"])
def test_extension_multiple_positives_use_total_probability_and_ignore_padding(family):
    import torch
    from mmdd_stage1.r26_extension_objectives import objective
    from mmdd_stage1.scoring import ListScores,TargetScores
    d = torch.tensor([[2.,1.,-1.,99.]],requires_grad=True)
    e = torch.tensor([[0.,3.,1.,99.]],requires_grad=True)
    mask = torch.tensor([[True,True,True,False]])
    pos = torch.tensor([[True,True,False,False]])
    scores = TargetScores(*(ListScores(x,mask,torch.tensor([0]),pos) for x in (d,e)))
    loss = objective(scores,family=family)["loss"]
    logits = [d[:,:3],e[:,:3]] if family == "split" else [torch.logsumexp(torch.stack((d[:,:3],e[:,:3])) ,0)]*2
    expected = sum((torch.logsumexp(x,1)-torch.logsumexp(x[:,:2],1)).mean() for x in logits)
    assert torch.allclose(loss,expected)
    actual = torch.autograd.grad(loss,(d,e),retain_graph=True)
    wanted = torch.autograd.grad(expected,(d,e))
    for got,want in zip(actual,wanted):
        assert torch.allclose(got,want)
        assert got[0,3] == 0


@pytest.mark.parametrize("family",["split","lse"])
@pytest.mark.parametrize("positive",[[True,True,False],[False,False,False]])
def test_extension_no_negative_or_no_positive_has_zero_loss_and_gradient(family,positive):
    import torch
    from mmdd_stage1.r26_extension_objectives import objective
    from mmdd_stage1.scoring import ListScores,TargetScores
    d = torch.tensor([[2.,-1.,100.]],requires_grad=True)
    e = torch.tensor([[1.,3.,100.]],requires_grad=True)
    mask = torch.tensor([[True,True,False]])
    pos = torch.tensor([positive])
    branch = lambda x: ListScores(x,mask,torch.tensor([0 if any(positive) else -1]),pos)
    teacher = TargetScores(branch(torch.tensor([[3.,2.,0.]])),branch(torch.tensor([[3.,2.,0.]])))
    terms = objective(TargetScores(branch(d),branch(e)),family=family,teacher=teacher,kd_weight=.3)
    assert terms["loss"] == 0
    assert terms["direct_active"] == terms["evidence_active"] == 0
    terms["loss"].backward()
    assert torch.isfinite(d.grad).all() and torch.isfinite(e.grad).all()
    assert not d.grad.any() and not e.grad.any()


def test_path_reassignment_with_fixed_edge_membership_changes_path_not_edge():
    import torch
    from mmdd_stage1.data import TargetExample,TargetCandidate
    from mmdd_stage1.features import ObjectFeatures
    from mmdd_stage1.models import StudentJoinabilityModel
    from mmdd_stage1.objectives import PathAggregator
    from mmdd_stage1.r26_extension_objectives import objective
    from mmdd_stage1.r26_training import graph_edges,edge_query_loss
    from mmdd_stage1.scoring import score_edge_batch,score_target_batch

    class Store:
        def embedding_features(self,oid):
            values = {"q":[1.,.1],"p":[1.,.2],"n":[.1,1.],"a":[1.,.1],"b":[.1,1.]}
            return ObjectFeatures(oid,"text" if oid in ("a","b") else "table",torch.tensor(values[oid]))

    model = StudentJoinabilityModel(2,2,initialization="identity",freeze_projections=False)
    device = torch.device("cpu")
    known = {("q","table","table"):{"p"},("q","table","text"):{"a"},
             ("a","text","table"):{"p"},("b","text","table"):{"n"}}
    # Reassign duplicate path occurrences. Simple directed edge memberships,
    # QT positives and total path count remain fixed; LSE consumes all paths.
    graphs = [TargetExample("q",(TargetCandidate("p",p),TargetCandidate("n",n)),0,0,positive_target_ids=("p",))
              for p,n in ((("a","b","b"),("a","a","b")),(("a","a","b"),("a","b","b")))]
    edges = [graph_edges(g,known,lambda _:"text") for g in graphs]
    assert edges[0] == edges[1]
    edge_losses,path_losses = [],[]
    for graph,lists in zip(graphs,edges):
        scored = score_edge_batch(model,lists,Store(),device,student_score_space="raw_logit")
        edge_losses.append(edge_query_loss(scored,lists,[0]*len(lists),1,scored.logits.sum()*0)["loss"])
        paths = score_target_batch(model,[graph],Store(),device,PathAggregator("logsumexp",8,path_combination="sum"))
        path_losses.append(objective(paths,family="split")["loss"])
    assert torch.equal(edge_losses[0],edge_losses[1])
    assert not torch.allclose(path_losses[0],path_losses[1])
    first = torch.autograd.grad(edge_losses[0],tuple(model.parameters()),allow_unused=True)
    second = torch.autograd.grad(edge_losses[1],tuple(model.parameters()),allow_unused=True)
    assert all((a is None and b is None) or torch.equal(a,b) for a,b in zip(first,second))


def test_real_evidence_changes_all_fusions_and_column_keeps_union():
    d = [{"target_id": "d", "direct_score": 1.0}]
    e = [{"target_id": "x", "evidence_score": 2.0}, {"target_id": "y", "evidence_score": 1.0}]
    before = fuse_channels(d, e, {"x": .8, "y": .8})
    after = fuse_channels(d, list(reversed(e)), {"x": .8, "y": .8})
    for method in ("Equal", "Conf", "Column"):
        assert set(before["rankings"][method]) == {"d", "x", "y"}
        assert before["scores"][method]["x"] != after["scores"][method]["x"]
    with pytest.raises(ValueError, match="missing_evidence_channel"):
        fuse_channels(d, None)


def test_rebuilt_production_student_index_changes_neighbor(tmp_path):
    import torch
    from mmdd_stage1.features import ObjectFeatures
    from mmdd_stage1.models import StudentJoinabilityModel
    from mmdd_stage1.retrieval import build_indices, StudentANNIndices

    class Store:
        def embedding_features(self, oid):
            vectors = {"q": [1., 1.], "a": [1., 0.], "b": [0., 1.]}
            return ObjectFeatures(oid, "table", torch.tensor(vectors[oid]))

    model = StudentJoinabilityModel(2, 2, initialization="identity", freeze_projections=False)
    store = Store()
    rankings = []
    for version, weights in enumerate(([2., 1.], [1., 2.])):
        with torch.no_grad():
            model.projections["table"].weight.copy_(torch.diag(torch.tensor(weights)))
        path = tmp_path / str(version)
        build_indices(model, store, {"table": ["a", "b"]}, path, device=torch.device("cpu"),
                      checkpoint_sha256=str(version), corpus_sha256="synthetic", num_threads=1)
        index = StudentANNIndices(model, store, path, device=torch.device("cpu"), checkpoint_sha256=str(version), corpus_sha256="synthetic")
        ranks = index.search("q", "table", 2)
        rankings.append(ranks[0][0])
        q = model.relation_query(store.embedding_features("q").embedding[None], "table", "table")
        targets = model.index_vector(torch.stack([store.embedding_features(t).embedding for t in ("a", "b")]), "table")
        scores = (q @ targets.T).flatten().tolist()
        assert dict(ranks) == pytest.approx(dict(zip(("a", "b"), scores)))
    assert rankings == ["a", "b"]


def test_query_bound_graph_does_not_label_all_positive_target_paths():
    from mmdd_stage1.data import TargetExample, TargetCandidate
    from mmdd_stage1.r26_training import graph_edges
    from mmdd_stage1.scoring import edge_positive_mask
    import torch
    graph = TargetExample("q", (TargetCandidate("p", ("e1", "e2")), TargetCandidate("n", ("e1",))), 0, 0, positive_target_ids=("p",))
    known = {("q", "table", "table"): {"p"}, ("q", "table", "text"): {"e1"}, ("e1", "text", "table"): {"p"}}
    edges = graph_edges(graph, known, lambda _: "text")
    unlabeled = next(e for e in edges if e.query_id == "e2")
    assert unlabeled.candidate_ids == ("p",)
    assert unlabeled.positive_index == -1
    assert not edge_positive_mask([unlabeled], 1, torch.device("cpu")).any()
    assert all(set(e.positive_ids).issubset(e.candidate_ids) for e in edges)


def test_edge_loss_preserves_each_query_aux_denominator():
    import torch
    from mmdd_stage1.data import EdgeExample
    from mmdd_stage1.r26_training import edge_query_loss
    from mmdd_stage1.scoring import ListScores
    examples = [EdgeExample(str(i), ("p", "n"), 0, source_type="table", destination_type="text", positive_ids=("p",)) for i in range(3)]
    logits = torch.tensor([[1., 0.], [0., 1.], [2., 0.]], requires_grad=True)
    mask = torch.ones_like(logits, dtype=torch.bool)
    positive = torch.tensor([[True, False], [False, False], [True, False]])
    terms = edge_query_loss(ListScores(logits, mask, torch.tensor([0, -1, 0]), positive), examples, [0, 0, 1], 2, logits.sum() * 0)
    expected = .5 * (torch.nn.functional.softplus(-logits[0, 0]) / 2 + torch.nn.functional.softplus(-logits[2, 0])) / 2
    assert torch.allclose(terms["loss"], expected)
    terms["loss"].backward()
    assert torch.isfinite(logits.grad).all()
    assert not logits.grad[1].any()


@pytest.mark.parametrize("arm", ["O-NATIVE", "O-SUP", "O-EXT-SUP", "O-EXT-SUP-ZERO"])
def test_order_only_production_factory_matches_old_loss_grad_and_adamw(arm):
    import copy
    import torch
    from mmdd_stage1.b13_recipe import path_objective
    from mmdd_stage1.data import TargetExample, TargetCandidate
    from mmdd_stage1.features import ObjectFeatures
    from mmdd_stage1.models import StudentJoinabilityModel
    from mmdd_stage1.objectives import PathAggregator
    from mmdd_stage1.r25_objectives import split_objective
    from mmdd_stage1.scoring import score_target_batch, TargetScores, ListScores
    from mmdd_stage1.training import _anchor_losses
    from run_stage1_r22_f0 import _optimizer
    from train_stage1_r26 import path_terms

    class Store:
        def embedding_features(self, oid):
            vectors = {"q": [1., .2], "p": [.8, .1], "n": [.1, 1.], "e": [.7, .3]}
            return ObjectFeatures(oid, "text" if oid == "e" else "table", torch.tensor(vectors[oid]))

    torch.manual_seed(1)
    original = StudentJoinabilityModel(2, 2, initialization="identity", freeze_projections=False)
    models = [copy.deepcopy(original), copy.deepcopy(original)]
    example = TargetExample("q", (TargetCandidate("p", ("e",)), TargetCandidate("n", ("e",))), 0, 0, positive_target_ids=("p",))
    outputs = []
    for new, model in zip((False, True), models):
        optimizer = _optimizer(model)
        scores = score_target_batch(model, [example], Store(), torch.device("cpu"), PathAggregator("logsumexp", 4 if arm == "O-NATIVE" else 8))
        teacher = TargetScores(*[ListScores(s.logits.detach() * 2, s.candidate_mask, s.positive_indices, s.positive_mask) for s in (scores.direct, scores.evidence)])
        if arm == "O-EXT-SUP-ZERO":
            teacher = TargetScores(teacher.direct,ListScores(teacher.direct.logits,
                teacher.evidence.candidate_mask,teacher.evidence.positive_indices,teacher.evidence.positive_mask))
        if new and arm.startswith("O-EXT-SUP"):
            from mmdd_stage1.r26_extension_objectives import objective
            _, anchor = _anchor_losses(model, .1, .1)
            extras = {"teacher":teacher,"kd_weight":0.,"uniform_weight":0.,
                      "uniform_scores":scores.direct,"uniform_relations":["table->table"],
                      "owners":[0]} if arm.endswith("-ZERO") else {}
            loss = objective(scores, family="split", anchor=anchor,**extras)["loss"]
        elif new:
            loss = path_terms(model, scores, arm, teacher if arm == "O-NATIVE" else None)["loss"]
        elif arm == "O-NATIVE":
            loss = path_objective(model, scores, teacher)["loss"]
        else:
            _, anchor = _anchor_losses(model, .1, .1)
            loss = split_objective(scores, arm="SPLIT-SUP", anchor_loss=anchor)["loss"]
        loss.backward()
        gradients = {k: None if p.grad is None else p.grad.clone() for k, p in model.named_parameters()}
        optimizer.step()
        outputs.append((loss.detach(), gradients, dict(model.named_parameters())))
    assert torch.equal(outputs[0][0], outputs[1][0])
    for name in outputs[0][1]:
        left, right = outputs[0][1][name], outputs[1][1][name]
        assert (left is None and right is None) or torch.equal(left, right)
        assert torch.equal(outputs[0][2][name], outputs[1][2][name])


@pytest.mark.parametrize("module_name,arm",[("train_stage1_r26","O-NATIVE"),
                                          ("train_stage1_r26_extension","O-UQTKD")])
def test_actual_student_resume_rejects_changed_identity_inputs(tmp_path,monkeypatch,module_name,arm):
    """Exercise production identity/resume branches before any model or GPU work."""
    import importlib
    import json
    import torch

    driver = importlib.import_module(module_name)
    root,output = tmp_path/"root",tmp_path/"output"
    r25 = root/"r25"
    graph,registry,features = root/"graph.jsonl",root/"train.jsonl",root/"features"
    source_names = ["train_stage1_r26.py","train_stage1_r26_extension.py",
        "mmdd_stage1/r26_training.py","mmdd_stage1/r26_extension_objectives.py",
        "mmdd_stage1/r25_objectives.py","mmdd_stage1/b13_recipe.py","mmdd_stage1/scoring.py",
        "mmdd_stage1/models.py","mmdd_stage1/objectives.py","mmdd_stage1/training.py","run_stage1_r22_f0.py"]
    files = [r25/"training/C1/seed13/checkpoints/step_000659.pt",graph,registry,
        features/"manifest.jsonl",output/"common/c2_order.jsonl",output/"PROTOCOL.json",
        r25/"common/teacher_native_path_cache.jsonl.gz",
        root/"work/stage1_optimization_r24_20260913/path_pool/teacher_target_seed13.jsonl.gz",
        root/"work/stage1_optimization_r22_20260911/fresh_lineage/T1-B/seed13/checkpoints/step_010536.pt",
        *[root/"src"/name for name in source_names]]
    for path in files:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text("synthetic identity input\n")
    (output/"acceptance").mkdir()
    (output/"acceptance/C1_TENSOR_AUDIT.json").write_text(json.dumps({"c1":{"13":{"reuse_valid":True}}}))
    (output/"acceptance/LOSS_EXTENSION_GATE_FROZEN.json").write_text(json.dumps({"status":"triggered","inputs":{}}))
    monkeypatch.setattr(driver,"ROOT",root)
    monkeypatch.setattr(driver,"OUT",output)
    monkeypatch.setattr(driver,"r25_out",lambda _:r25)
    monkeypatch.setattr(driver,"_r25_path_pool",lambda _:graph)
    monkeypatch.setattr(driver,"paths",lambda _:{"train":registry,"features":features})
    monkeypatch.setattr(torch.cuda,"set_device",lambda _:None)

    class ReachedTraining(Exception):
        pass

    def stop_before_training(*args,**kwargs):
        raise ReachedTraining

    monkeypatch.setattr(driver,"load_target_examples",stop_before_training)
    with pytest.raises(ReachedTraining):
        driver.train(arm,13,"cpu")
    job = output/"training"/arm/"seed13"
    signature = json.loads((job/"RUN_IDENTITY.json").read_text())
    receipt = {"signature":signature,"test_sentinel":"cached completion"}
    receipt_path = job/"C2_COMPLETION_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt))
    assert driver.train(arm,13,"cpu") == receipt
    identities = [files[0],graph,registry,features/"manifest.jsonl",output/"common/c2_order.jsonl",output/"PROTOCOL.json"]
    identities += [root/"src"/f"{module_name}.py"]
    identities += [files[6]] if arm == "O-NATIVE" else [files[7],files[8]]
    for path in identities:
        original = path.read_bytes()
        path.write_bytes(original+b"changed")
        with pytest.raises(ValueError,match="identity"):
            driver.train(arm,13,"cpu")
        path.write_bytes(original)
    assert driver.train(arm,13,"cpu") == receipt
    receipt_path.unlink()
    graph.write_text("changed incomplete-run graph")
    with pytest.raises(ValueError,match="identity"):
        driver.train(arm,13,"cpu")
