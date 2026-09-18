import unittest
import torch
from reference_core import (CompactStudent, UnifiedTeacher, feature_bytes, kd_loss,
                            rank_loss, round_robin, support_label)

torch.set_num_threads(1)

class CoreTests(unittest.TestCase):
    def test_all_positive_gradient(self):
        s = torch.tensor([10., -2., 0.], requires_grad=True)
        loss = rank_loss(s, torch.tensor([True, True, False]), torch.tensor([False, False, True]))
        loss.backward()
        self.assertLess(float(s.grad[0]), 0)
        self.assertLess(float(s.grad[1]), 0)
        self.assertGreater(float(s.grad[2]), 0)

    def test_unknown_is_masked(self):
        p = torch.tensor([True, False, False]); n = torch.tensor([False, False, True])
        a = rank_loss(torch.tensor([1., -999., 0.]), p, n)
        b = rank_loss(torch.tensor([1., 999., 0.]), p, n)
        torch.testing.assert_close(a, b)

    def test_empty_loss(self):
        s = torch.ones(3, requires_grad=True)
        v = rank_loss(s, torch.zeros(3, dtype=torch.bool), torch.ones(3, dtype=torch.bool))
        self.assertEqual(v.item(), 0)
        v.backward()
        torch.testing.assert_close(s.grad, torch.zeros(3))

    def test_kd_no_teacher_gradient(self):
        s = torch.tensor([0., 1., 80.], requires_grad=True)
        t = torch.tensor([1., 0., -80.], requires_grad=True)
        mask = torch.tensor([True, True, False])
        loss = kd_loss(s, t, mask)
        loss.backward()
        self.assertIsNone(t.grad)
        self.assertEqual(s.grad[-1].item(), 0)
        self.assertGreater(s.grad[1].item(), 0)
        self.assertLess(s.grad[0].item(), 0)

    def test_support_semantics(self):
        direct, implicit = {'d'}, {'i'}
        w = {'i': {'e'}}
        self.assertEqual(support_label('d', set(), direct, implicit, w, set()), 1)
        self.assertEqual(support_label('i', set(), direct, implicit, w, set()), 0)
        self.assertEqual(support_label('i', {'e'}, direct, implicit, w, set()), 1)
        self.assertIsNone(support_label('i', {'unjudged'}, direct, implicit, w, set()))
        self.assertIsNone(support_label('unknown_t', set(), direct, implicit, w, set()))

    def test_rr(self):
        streams = [['a','b','c'], ['a','d','b','e']]
        self.assertEqual(round_robin(streams), ['a','d','b','e','c'])
        self.assertEqual(round_robin(streams, 3), ['a','d','b'])
        self.assertEqual(round_robin([[],['a','a']]), ['a'])

    def test_student_static_index_and_query_condition(self):
        torch.manual_seed(13)
        m = CompactStudent(input_dim=16, d=8, rank=3)
        x = torch.randn(4,9,16); valid = torch.ones(4,9,dtype=torch.bool)
        kind = torch.tensor([[0,1,2,2,2,2,2,2,2]]*4)
        modality = torch.tensor([0,0,1,2])
        u = m.encode(x,valid,modality,kind)
        torch.testing.assert_close(u.norm(dim=-1), torch.ones(4))
        a = m.query_next(u[:1],u[2:3]); b=m.query_next(u[1:2],u[2:3])
        self.assertFalse(torch.allclose(a,b))
        torch.testing.assert_close(m.logits(a,u), (a[:,None,:]*u[None]).sum(-1)/0.07)
        self.assertTrue(torch.isfinite(m.logits(a,u)).all())

    def teacher_fixture(self):
        torch.manual_seed(13)
        m=UnifiedTeacher(input_dim=16,d=16,heads=4,ffn=32).eval()
        x=torch.randn(2,5,9,16); valid=torch.ones(2,5,9,dtype=torch.bool)
        # one wholly padded E block; partial slot padding in another object
        valid[0,4]=False; valid[1,2,6:]=False
        modality=torch.tensor([[0,0,1,2,1],[0,0,1,2,1]])
        role=torch.tensor([[0,1,2,2,2],[0,1,2,2,2]])
        kind=torch.tensor([0,3,4,5,6,7,8,9,10])[None,None].expand(2,5,9).clone()
        mode=torch.ones(2,dtype=torch.long)
        return m,x,valid,modality,role,kind,mode

    def test_teacher_permutation(self):
        m,x,v,mod,role,kind,mode=self.teacher_fixture()
        with torch.no_grad():
            a=m(x,v,mod,role,kind,mode)
            p=torch.tensor([0,1,4,2,3])
            b=m(x[:,p],v[:,p],mod[:,p],role[:,p],kind[:,p],mode)
        self.assertTrue(torch.isfinite(a).all())
        torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)

    def test_teacher_padding(self):
        m,x,v,mod,role,kind,mode=self.teacher_fixture()
        y=x.clone();y[~v]=1e5
        with torch.no_grad():
            a=m(x,v,mod,role,kind,mode);b=m(y,v,mod,role,kind,mode)
        torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)

    def test_teacher_one_head_and_mode(self):
        m,x,v,mod,role,kind,mode=self.teacher_fixture()
        self.assertEqual(len([n for n,_ in m.named_modules() if n=='readout']),1)
        with torch.no_grad():
            a=m(x,v,mod,role,kind,mode);b=m(x,v,mod,role,kind,torch.zeros_like(mode))
        self.assertFalse(torch.allclose(a,b))

    def test_storage(self):
        self.assertEqual(feature_bytes(1),81920)
        self.assertAlmostEqual(feature_bytes(200000)/(1024**3),15.2587890625)

if __name__ == '__main__':
    unittest.main(verbosity=2)
