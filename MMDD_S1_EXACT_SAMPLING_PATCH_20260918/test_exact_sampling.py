import itertools
import random
import unittest
from exact_sampling import (Corpus, PrefixTask, build_prefix, exact_topk,
                            ordered_prefixes, resolve_negatives,
                            smallest_records, stable_order_reference)


def original_negatives(corpus, namespace, excluded, hard_rank, budget=31, hard_budget=16):
    blocked = set(excluded)
    hard = []
    for x in hard_rank:
        if len(hard) == hard_budget:
            break
        if x in corpus.members and x not in blocked and x not in hard:
            hard.append(x)
    rest = corpus.members - blocked - set(hard)
    return tuple(hard + stable_order_reference(rest, namespace)[:budget-len(hard)])


class ExactSamplingTests(unittest.TestCase):
    def setUp(self):
        self.c = Corpus.build([f'id_{i:04d}' for i in range(300)])
        self.ns = 'clean-r1:13:student:S:packet=E:epoch=1:q=Q:anchor=none:uniform'

    def test_literal_reference(self):
        for k in [0, 1, 15, 31, 400]:
            self.assertEqual(exact_topk(self.c, self.ns, k),
                             tuple(stable_order_reference(self.c.ids, self.ns)[:k]))

    def test_duplicates_and_unicode(self):
        values = ['', '中', 'é', 'e\u0301', 'a', '中', '日本', '\0', '😀']
        c = Corpus.build(values)
        self.assertEqual(exact_topk(c, '测试\0namespace', 99),
                         tuple(stable_order_reference(values, '测试\0namespace')))

    def test_empty_and_all_excluded(self):
        self.assertEqual(exact_topk(Corpus.build([]), self.ns, 31), ())
        self.assertEqual(exact_topk(self.c, self.ns, 31, self.c.ids), ())

    def test_exclusion(self):
        blocked = ['id_0000', 'id_0100', 'not-in-corpus']
        self.assertEqual(exact_topk(self.c, self.ns, 31, blocked),
                         tuple(stable_order_reference(self.c.members-set(blocked), self.ns)[:31]))

    def test_hard_counts_zero_to_sixteen(self):
        blocked = ('id_0000', 'id_0001', 'id_0002')
        prefix = build_prefix(self.c, PrefixTask(self.ns, blocked))
        for n in range(17):
            hard = list(prefix.ids[:n])
            self.assertEqual(resolve_negatives(self.c, prefix, hard),
                             original_negatives(self.c, self.ns, blocked, hard))

    def test_reuse_prefix_two_arms_and_final_order(self):
        prefix = build_prefix(self.c, PrefixTask(self.ns, ('id_0000',)))
        for hard in [self.c.ids[30:46], prefix.ids[:16], self.c.ids[-16:]]:
            actual = resolve_negatives(self.c, prefix, hard)
            expected = original_negatives(self.c, self.ns, ('id_0000',), hard)
            self.assertEqual(actual, expected)
            shuffle_ns = self.ns.replace(':uniform', ':list-order')
            self.assertEqual(stable_order_reference(['id_0000', *actual], shuffle_ns),
                             stable_order_reference(['id_0000', *expected], shuffle_ns))

    def test_small_legal_corpus(self):
        for n in range(35):
            c = Corpus.build([str(i) for i in range(n)])
            blocked = ('0', '2')
            p = build_prefix(c, PrefixTask(self.ns, blocked))
            hard = c.ids[::2]
            self.assertEqual(resolve_negatives(c, p, hard),
                             original_negatives(c, self.ns, blocked, hard))

    def test_invalid_and_repeated_hard_ids(self):
        blocked = ('id_0000',)
        hard = ['absent', 'id_0000', 'id_0001', 'id_0001', *self.c.ids[50:80]]
        p = build_prefix(self.c, PrefixTask(self.ns, blocked))
        self.assertEqual(resolve_negatives(self.c, p, hard),
                         original_negatives(self.c, self.ns, blocked, hard))

    def test_full_digest_and_tie_break(self):
        # Identical first 64 bits must NOT collapse different full SHA values.
        d1 = b'\0'*8 + b'\x01' + b'\0'*23
        d2 = b'\0'*8 + b'\x02' + b'\0'*23
        records = [(d2, 0), (d1, 9), (d1, 2), (b'\xff'*32, 1)]
        self.assertEqual(smallest_records(records, 2), sorted(records)[:2])

    def test_global_rng_not_consumed(self):
        before = random.getstate()
        exact_topk(self.c, self.ns, 31)
        self.assertEqual(random.getstate(), before)

    def test_namespace_sensitive_no_mixed_epoch_cache(self):
        p1 = build_prefix(self.c, PrefixTask(self.ns, ()))
        p2 = build_prefix(self.c, PrefixTask(self.ns.replace('epoch=1', 'epoch=2'), ()))
        self.assertNotEqual(p1.task.namespace, p2.task.namespace)
        self.assertEqual(p2.ids, tuple(stable_order_reference(self.c.ids, p2.task.namespace)[:31]))

    def test_mismatched_corpus_rejected(self):
        p = build_prefix(self.c, PrefixTask(self.ns, ()))
        with self.assertRaises(ValueError):
            resolve_negatives(Corpus.build(['different']), p, [])

    def test_parallel_order_equals_serial(self):
        tasks = [PrefixTask(self.ns.replace('q=Q:', f'q=Q{i}:'), ('id_0000',)) for i in range(8)]
        expected = [build_prefix(self.c, task) for task in tasks]
        actual = list(ordered_prefixes(self.c, tasks, workers=2, max_pending=3))
        self.assertEqual(actual, expected)

    def test_prefix_identity_exhaustive_small_subsets(self):
        c = Corpus.build([str(i) for i in range(8)])
        p = build_prefix(c, PrefixTask(self.ns, (), negative_budget=5))
        for n in range(4):
            for subset in itertools.combinations(c.ids, n):
                self.assertEqual(resolve_negatives(c, p, subset, hard_budget=3),
                                 original_negatives(c, self.ns, (), subset, budget=5, hard_budget=3))


if __name__ == '__main__':
    unittest.main(verbosity=2)
