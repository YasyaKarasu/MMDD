"""Independent CPU reference contracts for the Stage-1 CQET math and audit tools.

``stage1_reference_contracts`` is a standalone re-statement of the formulas, not the
training code; ``run_stage1.py validate`` runs this file next to ``test_stage1_cqet.py``.
"""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage1_reference_contracts as r
from mmdd_stage1 import independent_metrics
from mmdd_stage1.provenance import verify_file_manifest

class Contracts(unittest.TestCase):
    def test_ann_nonsymmetric_relation(self):
        q=torch.tensor([1.,2.]);R=torch.tensor([[1.,4.],[-3.,2.]])
        T=torch.tensor([[3.,1.],[2.,-4.]])
        self.assertTrue(torch.equal(r.student_score(q,R,T),T@r.ann_query(q,R)))
        self.assertFalse(torch.equal(T@q,T@r.ann_query(q,R)))
        self.assertFalse(torch.equal(T@(q@R.T),T@r.ann_query(q,R)))
    def test_empty_E_exact(self):
        q,t,p=torch.randn(3,5)
        out=r.global_features(q,t,p)
        self.assertEqual(len(out),55)
        self.assertEqual(torch.count_nonzero(out[25:]).item(),0)
    def test_E_gradient(self):
        q,t,p=torch.randn(3,5);e=torch.randn(5,requires_grad=True)
        r.global_features(q,t,p,e,torch.ones(5)).square().sum().backward()
        self.assertGreater(e.grad.norm().item(),0)
    def test_cqet_nonempty_no_f0_competition(self):
        f0=torch.tensor([100.,2.],requires_grad=True);p=torch.tensor([-20.,-18.],requires_grad=True)
        v=r.aggregate_cqet(f0,p,torch.tensor([0,0]));v.sum().backward()
        self.assertTrue(torch.equal(f0.grad,torch.tensor([0.,1.])))
        self.assertAlmostEqual(p.grad.sum().item(),1,places=6)
        self.assertEqual(v[1].item(),2.)
    def test_aggregation_permutation(self):
        f0=torch.randn(3);p=torch.randn(6);idx=torch.tensor([2,0,1,1,0,2]);order=torch.tensor([4,2,0,5,1,3])
        self.assertTrue(torch.allclose(r.aggregate_cqet(f0,p,idx),r.aggregate_cqet(f0,p[order],idx[order])))
    def test_lme_no_count_bonus(self):
        f0=torch.tensor([1.,1.]);p=torch.tensor([3.,3.,3.]);idx=torch.tensor([0,1,1])
        self.assertTrue(torch.allclose(r.aggregate_cqet(f0,p,idx),torch.tensor([3.,3.])))
    def test_path_index_rejected(self):
        with self.assertRaises(ValueError):r.aggregate_cqet(torch.ones(1),torch.ones(1),torch.tensor([2]))
    def test_multi_positive_and_ignore(self):
        s=torch.tensor([2.,1.,20.,0.],requires_grad=True)
        loss=r.rank_mass(s,torch.tensor([True,True,False,False]),torch.tensor([True,True,False,True]))
        loss.backward();self.assertEqual(s.grad[2].item(),0)
        self.assertLess(s.grad[0].item(),0);self.assertLess(s.grad[1].item(),0)
    def test_no_competitor_skip(self):
        self.assertIsNone(r.rank_mass(torch.ones(2),torch.ones(2,dtype=torch.bool)))
    def test_kd_grad_and_detached_teacher(self):
        s=torch.tensor([.1,.2,.3],requires_grad=True);t=torch.tensor([3.,0.,-1.],requires_grad=True)
        r.list_kd(s,t).backward();self.assertGreater(s.grad.norm().item(),0);self.assertIsNone(t.grad)
    def test_nograd_KD_fails(self):
        s=torch.ones(3,requires_grad=True)
        with torch.no_grad():
            with self.assertRaises(RuntimeError):r.list_kd(s,torch.zeros(3))
    def test_support_common_shift_gradient(self):
        bias=torch.tensor(2.,requires_grad=True)
        loss=r.support_pair(torch.tensor([1.,3.])+bias,torch.tensor([0.,2.])+bias)
        loss.backward();self.assertAlmostEqual(bias.grad.item(),0,places=6)
    def test_hierarchical_denominator(self):
        got=r.hierarchical_mean([[torch.tensor(2.),torch.tensor(4.)],[torch.tensor(9.)]])
        self.assertEqual(got.item(),6.)
    def test_alias_id_rejected(self):
        with self.assertRaises(ValueError):r.normalize_content_hash('asset1','asset1')
    def test_sampling_protect_and_no_overlap(self):
        got=r.sample_competitors(['p','a','b','c'],['p','a','b','c','d','e'],{'p'},2,3,13,'q')
        self.assertEqual(got['hard'],['a','b']);self.assertNotIn('p',got['uniform'])
        self.assertFalse(set(got['hard'])&set(got['uniform']))
    def test_coverage_not_hit(self):
        v=r.pool_metrics(['a','x'],{'a','b','c','d'})
        self.assertEqual(v['coverage'],.25);self.assertEqual(v['hit_rate'],1.)
    def test_oracle_target_recall(self):
        v=r.pool_metrics([str(i) for i in range(15)],set(str(i) for i in range(20)))
        self.assertEqual(v['oracle'],.5);self.assertEqual(v['coverage'],.75)
    def test_bootstrap_query_weighting(self):
        got=r.source_group_bootstrap([1.,0.,0.,0.],['a','b','b','b'],replicates=100)
        self.assertEqual(got['query_macro_delta'],.25)
        self.assertNotEqual(got['query_macro_delta'],.5)
    def test_ranking_duplicate_rejected(self):
        with self.assertRaises(ValueError):r.rank_metrics(['a','a'],{'a'},10)
    def test_parent_hash(self):
        model=torch.nn.Linear(3,2);parent={k:torch.ones_like(v) for k,v in model.state_dict().items()}
        self.assertEqual(r.load_stage_parent(model,parent),r.state_digest(parent))
    def test_D1_admission_not_maxpath(self):
        order=r.p3_admission({'a':1.,'b':.9},{'a':0.,'b':1.},budget=2)
        # Sum ranks (1,2) vs (2,1) ties; UTF8 rule wins; reverse D1 has an observable effect below.
        self.assertEqual(order,['a','b'])
        self.assertEqual(r.p3_admission({'a':1.,'b':.9},{'b':1.},1),['b'])
    def test_seeded_order_repeatable(self):
        self.assertEqual(r.hash_order(['a','b','c'],'X',13),r.hash_order(['c','b','a'],'X',13))
    def test_score_vjp_recomputation(self):
        torch.manual_seed(13);m=torch.nn.Linear(3,1);X=torch.randn(8,3)
        original=m(X).flatten();full=torch.logsumexp(original,0)-original[0]
        full.backward();expected=[v.grad.clone() for v in m.parameters()];m.zero_grad()
        with torch.no_grad():first=m(X).flatten()
        leaf=first.detach().requires_grad_(True);loss=torch.logsumexp(leaf,0)-leaf[0];(coef,)=torch.autograd.grad(loss,leaf)
        for i in range(0,8,2):m(X[i:i+2]).flatten().backward(coef[i:i+2])
        for p,g in zip(m.parameters(),expected):self.assertTrue(torch.allclose(p.grad,g,atol=1e-6))
    def test_resume_optimizer_rng(self):
        torch.manual_seed(10);m=torch.nn.Linear(3,1);o=torch.optim.AdamW(m.parameters(),lr=.01)
        def step(model,opt):
            opt.zero_grad();loss=model(torch.randn(4,3)).square().mean();loss.backward();opt.step()
        step(m,o);step(m,o)
        import copy
        saved=copy.deepcopy(m.state_dict());optim=copy.deepcopy(o.state_dict());rng=torch.get_rng_state()
        step(m,o);step(m,o);expected=r.state_digest(m.state_dict())
        n=torch.nn.Linear(3,1);no=torch.optim.AdamW(n.parameters(),lr=.01);n.load_state_dict(saved);no.load_state_dict(optim);torch.set_rng_state(rng)
        step(n,no);step(n,no);self.assertEqual(r.state_digest(n.state_dict()),expected)


class AuditTools(unittest.TestCase):
    def test_raw_metrics_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            def save(name,rows):
                (p/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            save('queries.jsonl',[{'query_id':'q1','source_group':'s','query_kind':'implicit'}, {'query_id':'q2','source_group':'s','query_kind':'explicit'}])
            save('qrels.jsonl',[{'query_id':'q1','target_id':'t1','rel':1},{'query_id':'q1','target_id':'t2','rel':1},{'query_id':'q2','target_id':'t3','rel':1}])
            save('ranks.jsonl',[{'query_id':'q1','target_ids':['t1'],'scores':[1.]},{'query_id':'q2','target_ids':['t3'],'scores':[2.]}])
            got=independent_metrics.evaluate(p/'queries.jsonl',p/'qrels.jsonl',p/'ranks.jsonl',p/'out')
            self.assertEqual(got['segments']['overall']['R10'],.75)
            self.assertEqual(got['segments']['overall']['QueryHitRate_pool'],1.)
    def test_manifest_tamper_detected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);f=p/'x';f.write_bytes(b'ok')
            manifest=p/'FILE_MANIFEST.jsonl'
            manifest.write_text(json.dumps({'path':'x','bytes':2,'sha256':hashlib.sha256(b'ok').hexdigest(),'role':'test'})+'\n')
            self.assertEqual(verify_file_manifest(p,manifest)['checked_files'],1)
            f.write_bytes(b'no')
            with self.assertRaises(ValueError):verify_file_manifest(p,manifest)
    def test_manifest_unsafe_path(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);manifest=p/'FILE_MANIFEST.jsonl'
            manifest.write_text(json.dumps({'path':'../elsewhere','bytes':0,'sha256':'bad','role':'test'})+'\n')
            with self.assertRaises(ValueError):verify_file_manifest(p,manifest)
