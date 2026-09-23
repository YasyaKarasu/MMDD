"""S0 / SPEC 12 integration acceptance: production probes A11-A19 and the real small chain A20.

Every assertion calls the functions that the formal stages import.  The
small chain uses a handful of real train queries, temporary models and a
temporary directory; its state is discarded afterwards.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import graphs, lists as lists_mod, raw as raw_mod, runlog, student as student_mod, teacher as teacher_mod
from .config import Paths
from .data import Labels, load_basis, load_labels, load_row_store, load_split_gt, load_z, make_bank, utf8_sorted
from .io import write_json
from .losses import kd_loss, rank_ce, student_anchor, sup_transform
from .pools import PathEntry, PoolRecord, assemble_pool, d1_retain
from .retrieval import OwnRetriever

REFERENCE_DIR = Path(__file__).resolve().parents[2] / "MMDD_FRESH_REBUILD_AUDIT_20260922" / "reference"


def _kernels():
    if str(REFERENCE_DIR) not in sys.path:
        sys.path.insert(0, str(REFERENCE_DIR))
    import kernels  # type: ignore

    return kernels


class Probe:
    def __init__(self) -> None:
        self.results: list[dict] = []

    def run(self, item: str, name: str, fn) -> None:
        started = time.time()
        try:
            detail = fn() or {}
            status = "PASS"
        except Exception as error:  # record, do not hide
            detail = {"error": repr(error), "traceback": traceback.format_exc()[-2000:]}
            status = "FAIL"
        self.results.append({"item": item, "name": name, "status": status, "elapsed": round(time.time() - started, 2), **detail})
        print(json.dumps({"probe": item, "name": name, "status": status}), flush=True)


class _Bank:
    """Tiny synthetic z bank with the production interface."""

    def __init__(self, vals: dict[str, torch.Tensor], kinds: dict[str, str]) -> None:
        self.vals, self.kinds = vals, kinds

    def z(self, i):
        return self.vals[i]

    def z_many(self, ids):
        return torch.stack([self.vals[i] for i in ids])

    def kind(self, i):
        return self.kinds[i]


# ------------------------------------------------------------- synthetic ---


def probe_a12_c1_formula(p: Probe) -> None:
    def fn():
        torch.manual_seed(0)
        basis = torch.randn(4, 6)
        model = student_mod.NativeStudent(basis)
        with torch.no_grad():
            for r in model.relations.values():
                r.add_(0.1 * torch.randn_like(r))
            for w in model.projections.values():
                w.add_(0.05 * torch.randn_like(w))
        opt = student_mod.adamw(model)
        groups = {g["name"]: g["lr"] for g in opt.param_groups}
        assert groups == {"P": 1e-6, "R": 1e-5}, groups
        vals = {f"q{i}": torch.randn(6) for i in range(3)} | {f"t{i}": torch.randn(6) for i in range(5)} | {f"e{i}": torch.randn(6) for i in range(3)}
        kinds = {k: ("table" if k[0] in "qt" else ("text" if k == "e0" else "image")) for k in vals}
        bank = _Bank(vals, kinds)
        rows = [
            {"item_id": "QT:q0", "relation": "QT", "anchor_id": "q0", "positives": ["t0", "t1"], "candidates": ["t3", "t0", "t2", "t1"], "active": True, "teacher_logits": [0.2, 1.5, -0.3, 0.9]},
            {"item_id": "Q_text:q1", "relation": "Q_text", "anchor_id": "q1", "positives": ["e0"], "candidates": ["e0"], "active": True, "teacher_logits": [0.0]},
            {"item_id": "image_T:e1", "relation": "image_T", "anchor_id": "e1", "positives": ["t4"], "candidates": ["t2", "t4"], "active": True, "teacher_logits": [1.0, 0.4]},
            {"item_id": "image_T:e2", "relation": "image_T", "anchor_id": "e2", "positives": ["t1"], "candidates": ["t1", "t0", "t3"], "active": True, "teacher_logits": [0.1, 0.2, 0.3]},
        ]
        rows[1]["active"] = False  # no negative -> inactive, must be skipped by construction
        kernels = _kernels()
        active_rows = [r for r in rows if r["active"]]
        # reference: one logical batch of the 3 active lists, KD arm
        ref = None
        for r in active_rows:
            rel = student_mod.list_relation(r)
            s = model.score(rel, bank.z(r["anchor_id"]), bank.z_many(r["candidates"]))
            pmask = torch.tensor([c in set(r["positives"]) for c in r["candidates"]])
            term = kernels.edge_loss(s, torch.tensor(r["teacher_logits"]), pmask, torch.ones_like(pmask), kd_weight=0.3)
            ref = term if ref is None else ref + term
        ref = ref / len(active_rows) + 0.1 * kernels.anchor(dict(model.projections.items()), dict(model.relations.items()), basis)
        ref_grads = torch.autograd.grad(ref, list(model.parameters()))
        # production: micro-batched accumulation exactly as train_c1 does (micro=2 of 3 lists) + anchor once
        model.zero_grad()
        n = len(active_rows)
        for start in range(0, n, 2):
            total = None
            for r in active_rows[start : start + 2]:
                loss, parts = student_mod.c1_list_loss(model, bank, r, "cpu", kd=True)
                total = loss if total is None else total + loss
            (total / n).backward()
        (0.1 * model.anchor()).backward()
        for g, prm in zip(ref_grads, model.parameters()):
            assert torch.allclose(g, prm.grad, atol=1e-6), "C1 micro-batch gradient differs from reference"
        # SUP arm: kd flag off must not read teacher logits
        loss_sup, parts = student_mod.c1_list_loss(model, bank, dict(active_rows[0], teacher_logits=None), "cpu", kd=False)
        s = model.score("QT", bank.z("q0"), bank.z_many(active_rows[0]["candidates"]))
        pmask = torch.tensor([c in {"t0", "t1"} for c in active_rows[0]["candidates"]])
        assert torch.allclose(loss_sup, kernels.rank_ce(10 * torch.sigmoid(s), pmask, torch.ones_like(pmask)))
        # inactive list: production returns None (never falls back to G)
        assert student_mod.c1_list_loss(model, bank, rows[1], "cpu", kd=True)[0] is None
        return {"param_groups": groups, "lists_checked": n, "kd_weight": 0.3, "temperature": 1.0}
    p.run("A12", "C1 P/R groups, 10*sigmoid SUP, raw KD tau=1 lambda=.3, anchor once, micro-batch gradient", fn)


def _synthetic_c2_item(kernels):
    torch.manual_seed(1)
    basis = torch.randn(4, 6)
    model = student_mod.NativeStudent(basis)
    with torch.no_grad():
        for r in model.relations.values():
            r.add_(0.2 * torch.randn_like(r))
    vals = {"q": torch.randn(6)} | {f"t{i}": torch.randn(6) for i in range(5)} | {"e1": torch.randn(6), "e2": torch.randn(6), "e3": torch.randn(6)}
    kinds = {k: ("table" if k[0] in "qt" else ("text" if k in ("e1", "e3") else "image")) for k in vals}
    bank = _Bank(vals, kinds)
    # targets in reverse order, duplicated e across targets, t2 has no path, t4 has one
    targets = ["t3", "t1", "t0", "t2", "t4"]
    natural = {"t3": ["e1", "e2"], "t1": ["e2", "e1", "e3"], "t0": ["e1"], "t4": ["e3"]}
    gold = ["t1", "t0", "t2"]
    witness = {"t1": ["e2"], "t0": ["e9"]}  # t0 in G without support -> ignored in E view
    item = student_mod.build_c2_item("q", targets=targets, gold=gold, witness=witness, natural_bags=natural, augmented_e="e2")
    return model, bank, item, basis


def probe_a14_slot_order(p: Probe) -> None:
    def fn():
        kernels = _kernels()
        model, bank, item, basis = _synthetic_c2_item(kernels)
        terms = student_mod.c2_query_terms(model, bank, item, "cpu", kd=False, teacher=None)
        # loop reference per view
        for view_no, view in enumerate(item["views"]):
            bags = view["bags"]
            ref = []
            for t in item["targets"]:
                bag = bags.get(t, [])
                if not bag:
                    ref.append(torch.tensor(float("-inf")))
                    continue
                vals = []
                for e in bag:
                    k = bank.kind(e)
                    first = model.score(f"Q_{k}", bank.z("q"), bank.z(e).unsqueeze(0))[0]
                    second = model.score(f"{k}_T", bank.z(e), bank.z(t).unsqueeze(0))[0]
                    vals.append(first + second)
                ref.append(torch.logsumexp(torch.stack(vals), 0))
            ref = torch.stack(ref)
            # recompute production E_S for this view to compare value + gradient
            z_q, z_t = bank.z("q"), bank.z_many(item["targets"])
            u_q, u_t = model.u("table", z_q), model.u("table", z_t)
            first = torch.empty(len(item["evidence"]))
            second = torch.empty(len(item["evidence"]), len(item["targets"]))
            for kind in ("text", "image"):
                idx = [i for i, e in enumerate(item["evidence"]) if bank.kind(e) == kind]
                if idx:
                    u_e = model.u(kind, bank.z_many([item["evidence"][i] for i in idx]))
                    first[idx] = (u_q @ model.relations[f"Q_{kind}"]) @ u_e.T
                    second[idx] = (u_e @ model.relations[f"{kind}_T"]) @ u_t.T
            e_index = torch.tensor(view["e_index"]); t_index = torch.tensor(view["t_index"])
            prod = student_mod.segment_lse(first[e_index] + second[e_index, t_index], t_index, len(item["targets"]), view["counts"])
            finite = torch.isfinite(ref)
            assert torch.equal(finite, torch.isfinite(prod)) and torch.allclose(ref[finite], prod[finite], atol=1e-6)
            g_ref = torch.autograd.grad(ref[finite].sum(), model.relations["text_T"], retain_graph=True)[0]
            g_prod = torch.autograd.grad(prod[finite].sum(), model.relations["text_T"], retain_graph=True)[0]
            assert torch.allclose(g_ref, g_prod, atol=1e-6)
            # masks: t2 (no natural path / unsupported G in aug) not allowed; t0 (G without support) ignored;
            # t4 competitor; t1 positive; t3 competitor
            expect_allowed = [True, True, False, False, True]
            assert view["e_allowed"] == expect_allowed, view["e_allowed"]
            assert view["e_positive"] == [False, True, False, False, False]
        assert len(terms["e_terms"]) == 2
        # augmented witness e2 reaches every target, including t2 which had no natural path
        assert item["views"][1]["bags"]["t2"] == ["e2"]
        return {"targets": item["targets"], "views": len(item["views"]), "slots": [len(v["e_index"]) for v in item["views"]]}
    p.run("A14", "duplicate E across targets, reversed order, empty bag, augmented witness: values/gradients per slot", fn)


def probe_a13_c2_formula(p: Probe) -> None:
    def fn():
        kernels = _kernels()
        model, bank, item, basis = _synthetic_c2_item(kernels)
        # a second query without any valid E view (all G targets unsupported)
        item2 = student_mod.build_c2_item("q", targets=["t0", "t1", "t2"], gold=["t2"], witness={}, natural_bags={"t0": ["e1"]}, augmented_e=None)
        assert not any(v["active"] for v in item2["views"])
        teacher = {"D": torch.randn(5), "QE": torch.randn(len(item["evidence"])), "ET": [torch.randn(len(v["e_index"])) for v in item["views"]]}
        teacher2 = {"D": torch.randn(3)}
        items = {"a": item, "b": item2}
        tl = {"a": teacher, "b": teacher2}
        # production reduction for one logical batch of two queries (micro=1)
        model.zero_grad()
        n, n_e = 2, 1
        for q in ("a", "b"):
            terms = student_mod.c2_query_terms(model, bank, items[q], "cpu", kd=True, teacher=tl[q])
            part = terms["d_sup"] / n + 0.3 * terms["d_kd"] / n
            if terms["e_terms"]:
                part = part + torch.stack([t["sup"] for t in terms["e_terms"]]).mean() / n_e
                part = part + 0.3 * torch.stack([t["kd"] for t in terms["e_terms"]]).mean() / n_e
            part.backward()
        (0.1 * model.anchor()).backward()
        prod = [prm.grad.clone() for prm in model.parameters()]
        # reference kernel native_c2_batch
        model.zero_grad()
        d_student, d_teacher, d_pos, d_all, ev = [], [], [], [], []
        for q in ("a", "b"):
            it = items[q]
            d_student.append(model.score("QT", bank.z("q"), bank.z_many(it["targets"])))
            d_teacher.append(tl[q]["D"]); d_pos.append(torch.tensor(it["d_positive"])); d_all.append(torch.ones(len(it["targets"]), dtype=torch.bool))
            views = []
            for view_no, view in enumerate(it["views"]):
                if not view["active"]:
                    continue
                terms = student_mod.c2_query_terms(model, bank, it, "cpu", kd=False, teacher=None)
                # rebuild E_S with a plain loop (independent of segment_lse)
                scores = []
                for t in it["targets"]:
                    bag = view["bags"].get(t, [])
                    if not bag:
                        scores.append(torch.tensor(float("-inf"))); continue
                    vals = [model.score(f"Q_{bank.kind(e)}", bank.z("q"), bank.z(e).unsqueeze(0))[0] + model.score(f"{bank.kind(e)}_T", bank.z(e), bank.z(t).unsqueeze(0))[0] for e in bag]
                    scores.append(torch.logsumexp(torch.stack(vals), 0))
                t_scores = student_mod.segment_lse(tl[q]["ET"][view_no], torch.tensor(view["t_index"]), len(it["targets"]), view["counts"])
                views.append(kernels.EvidenceView(torch.stack(scores), t_scores, torch.tensor(view["e_positive"]), torch.tensor(view["e_allowed"])))
            ev.append(views)
        anchor_value = kernels.anchor(dict(model.projections.items()), dict(model.relations.items()), basis)
        ref = kernels.native_c2_batch(d_student, d_teacher, d_pos, d_all, ev, kd_weight=0.3, anchor_value=anchor_value)
        ref.backward()
        for a, prm in zip(prod, model.parameters()):
            assert torch.allclose(a, prm.grad, atol=1e-6), "C2 production reduction differs from kernel reference"
        return {"queries": 2, "e_valid_queries": 1, "kd_weight": 0.3}
    p.run("A13", "C2 separate D/E CE+KD, view mean then valid-query mean, anchor once: gradient equals kernel", fn)


def probe_a15_read_q(p: Probe) -> None:
    def fn():
        import unittest.mock as mock

        from fresh_path import train_student as ts
        from fresh_path.models import PCAStudent

        m = PCAStudent(torch.eye(2), adapter=True, adapter_hidden=3)
        bank = _Bank({"q": torch.tensor([.4, .2]), "e1": torch.tensor([1., 0.]), "e2": torch.tensor([0., 2.]), "t1": torch.tensor([1., 2.]), "t2": torch.tensor([3., 4.])},
                     {"q": "table", "e1": "text", "e2": "image", "t1": "table", "t2": "table"})
        reads = []
        orig = ts.adapter_vector

        def spy(*args, **kwargs):
            reads.append(kwargs.get("read_q", True))
            return orig(*args, **kwargs)
        flat = [("e1", "t1"), ("e2", "t1"), ("e1", "t2"), ("e2", "t2")]
        with mock.patch.object(ts, "adapter_vector", side_effect=spy):
            ts._conditional_path_scores(m, bank, "q", flat, "cpu", read_q=False)
        assert reads and all(r is False for r in reads), reads
        expected = torch.stack([(ts.et_logits(m, bank, e, [t], "cpu") + ts.first_hop_logits(m, bank, "q", bank.kind(e), [e], "cpu"))[0] for e, t in flat])
        assert torch.allclose(ts._native_path_scores(m, bank, "q", flat, "cpu"), expected)
        return {"read_q_calls": reads, "note": "legacy helper only; v3 trains no Student adapter (not a main-method metric)"}
    p.run("A15", "legacy EONLY helper: read_q=False reaches every adapter forward; grouped scorer restores slots", fn)


def probe_a17_pool_identity(p: Probe) -> None:
    def fn():
        entry = PathEntry("e1", 0.5, 0.4, "text")
        own = PoolRecord(split="dev", query_id="q", generator_id="S_KD_NATIVE_C2:frac100", model_sha="abc", target_index_sha="t", evidence_index_sha="e",
                         direct=[("t1", 1.0)], direct_exact=["t1"], first_hop={"text": [("e1", 0.5)], "image": []}, evidence=[("t1", 0.3)], U=["t1"], C150=["t1"],
                         pre_paths={"t1": [entry]}, retained_paths={"t1": ["e1"]}, retained_coverage={"t1": 0.3})
        raw = PoolRecord(**{**own.__dict__, "generator_id": "raw", "retained_paths": {"t1": ["e2"]}})
        own.validate(expected_generator="S_KD_NATIVE_C2:frac100")
        try:
            raw.validate(expected_generator="S_KD_NATIVE_C2:frac100")
            raise AssertionError("raw pool accepted for an own-generator rerank")
        except ValueError as error:
            assert "foreign generator" in str(error)
        try:
            own.validate(expected_generator="S_KD_NATIVE_C2:frac100", expected_model_sha="other")
            raise AssertionError("foreign checkpoint accepted")
        except ValueError:
            pass
        # the evaluation rerank must raise on a foreign pool and must forward the own E ids
        from . import evaluate as ev

        class Scorer:
            name = "probe"
            forwards = {"pairs": 0, "triplets": 0}

            def f0(self, qid, targets):
                return {t: 0.0 for t in targets}

            def qet(self, qid, slots):
                self.slots = list(slots)
                return {s: 1.0 for s in slots}
        scorer = Scorer()
        try:
            ev.teacher_readouts(scorer, {"q": raw}, generator_id="S_KD_NATIVE_C2:frac100", with_paths=True, swap=None, queries=["q"], log=lambda *_: None)
            raise AssertionError("teacher_readouts accepted a foreign generator pool")
        except ValueError:
            pass
        ev.teacher_readouts(scorer, {"q": own}, generator_id="S_KD_NATIVE_C2:frac100", with_paths=True, swap=None, queries=["q"], log=lambda *_: None)
        assert scorer.slots == [("e1", "t1")], scorer.slots
        return {"foreign_generator": "rejected", "foreign_checkpoint": "rejected", "forwarded_slots": scorer.slots}
    p.run("A17", "own PoolRecord identity: foreign generator/checkpoint raise; Teacher receives own evidence", fn)


def probe_a18_d1(p: Probe) -> None:
    def fn():
        kernels = _kernels()
        rng = np.random.default_rng(3)
        entries = [PathEntry(f"e{i}", float(rng.normal()), float(rng.normal()), "text") for i in range(30)]
        entries += [PathEntry("dupA", 2.0, 1.0, "text"), PathEntry("dupB", 1.5, 1.0, "text")]  # same content, different id
        content_key = {e.evidence_id: e.evidence_id for e in entries} | {"dupA": "same", "dupB": "same"}
        support = {e.evidence_id: rng.uniform(0, 1, size=5) for e in entries}
        sel, cov = d1_retain(entries, support, content_key=content_key)
        ref_sel, ref_cov = kernels.d1_greedy([(e.evidence_id, e.raw_path_score, content_key[e.evidence_id]) for e in entries],
                                             {k: list(v) for k, v in support.items()})
        assert sel == ref_sel and abs(cov - ref_cov) < 1e-9, (sel, ref_sel)
        assert "dupB" not in sel
        return {"selected": sel, "coverage": cov}
    p.run("A18", "D1 greedy retention matches the independent reference (dedup, tie, gain, coverage)", fn)


def probe_a19_chunks(p: Probe) -> None:
    def fn():
        torch.manual_seed(5)
        logits = torch.randn(9, requires_grad=True)
        pos = torch.tensor([True, False, False, True, False, False, False, False, False])
        allowed = torch.tensor([True, True, False, True, True, True, False, True, True])
        dense = rank_ce(logits, pos, allowed)
        # streamed LSE over chunks of 4
        alls, poss = [], []
        for s in range(0, 9, 4):
            a = allowed[s:s + 4]; pp = pos[s:s + 4]; sc = logits[s:s + 4]
            if a.any():
                alls.append(torch.logsumexp(sc[a], 0))
            if pp.any():
                poss.append(torch.logsumexp(sc[pp], 0))
        chunked = torch.logsumexp(torch.stack(alls), 0) - torch.logsumexp(torch.stack(poss), 0)
        assert torch.allclose(dense, chunked, atol=1e-6)
        assert torch.allclose(torch.autograd.grad(dense, logits, retain_graph=True)[0], torch.autograd.grad(chunked, logits)[0], atol=1e-6)
        # KD masked equals kernel
        kernels = _kernels()
        s = torch.randn(6, requires_grad=True); t = torch.randn(6); a = torch.tensor([True, False, True, True, False, True])
        assert torch.allclose(kd_loss(s, t, a, 1.0), kernels.kd(s, t, a, 1.0))
        # empty P stays empty: rank_ce returns None, never falls back
        assert rank_ce(torch.randn(3), torch.tensor([False, False, False])) is None
        # tail micro-batch weighting: 5 lists, logical 8 -> each list weight 1/5 (handled by /n in the loop)
        return {"chunk_equivalence": True, "empty_P": "None"}
    p.run("A19", "dense vs chunked CE/KL, masked KD, empty P stays empty", fn)


# --------------------------------------------------------- real small chain ---


def _subset_labels(labels: Labels, queries: list[str], legal: list[str], text: list[str], image: list[str]) -> Labels:
    qset = set(queries)
    legal_set = set(legal)
    sub_queries = {}
    for q in queries:
        e = dict(labels.queries[q])
        e["G"] = [t for t in e["G"] if t in legal_set]
        e["D"] = [t for t in e["D"] if t in legal_set]
        e["W"] = {t: v for t, v in e["W"].items() if t in legal_set}
        e["Qpos"] = {m: utf8_sorted({a for t in e["W"] for a in e["W"][t] if labels.modality[a] == m}) for m in ("text", "image")}
        sub_queries[q] = e
    epos = {}
    for q, e in sub_queries.items():
        for t, assets in e["W"].items():
            for a in assets:
                epos.setdefault(a, set()).add(t)
    epos = {a: utf8_sorted(v) for a, v in epos.items()}
    anchors = []
    for q, e in sub_queries.items():
        anchors.append({"item_id": f"QT:{q}", "relation": "QT", "anchor_id": q, "positive_ids": e["G"], "ignore_ids": []})
        for m in ("text", "image"):
            if e["Qpos"][m]:
                anchors.append({"item_id": f"Q_{m}:{q}", "relation": f"Q_{m}", "anchor_id": q, "positive_ids": e["Qpos"][m], "ignore_ids": []})
    for a, targets in epos.items():
            modality = labels.modality[a]
            anchors.append({"item_id": f"E_{modality}:{a}", "relation": f"{modality}_T", "anchor_id": a,
                            "positive_ids": targets, "ignore_ids": []})
    return Labels(queries=sub_queries, epos=epos, legal=utf8_sorted(legal), edge_anchors=anchors, canonical=labels.canonical,
                  modality=labels.modality, canonical_text=utf8_sorted(text), canonical_image=utf8_sorted(image), stats={}, files=labels.files)


def small_chain(paths: Paths, *, seed: int, device: str, out_dir: Path, p: Probe) -> None:
    labels = load_labels(paths)
    rows = load_row_store(paths)
    basis, basis_sha = load_basis(paths)
    root = lists_mod.data_root_id(paths)
    # choose queries: with both witness modalities, sharing evidence across targets when possible
    chosen = []
    for q in labels.query_ids:
        e = labels.queries[q]
        if e["Qpos"]["text"] and e["Qpos"]["image"] and len(e["G"]) >= 2:
            chosen.append(q)
        if len(chosen) >= 4:
            break
    chosen += [q for q in labels.query_ids if q not in chosen and labels.queries[q]["Qpos"]["image"] and not labels.queries[q]["Qpos"]["text"]][:1]
    chosen += [q for q in labels.query_ids if q not in chosen and not labels.queries[q]["W"]][:1]
    dev_gt = load_split_gt(paths, "dev")
    dev_q = utf8_sorted(dev_gt)[:3]
    z = load_z(paths)
    full_index = raw_mod.RawIndex(z, labels, device)
    legal: set[str] = set()
    text: set[str] = set()
    image: set[str] = set()
    with torch.no_grad():
        for q in chosen + dev_q:
            qz = full_index.vector(q)
            legal |= {t for t, _ in full_index.topk_targets(qz, 60)}
            text |= {e for e, _ in full_index.topk_evidence(qz, "text", 25)}
            image |= {e for e, _ in full_index.topk_evidence(qz, "image", 25)}
        for q in chosen:
            legal |= set(labels.queries[q]["G"])
            for assets in labels.queries[q]["W"].values():
                for a in assets:
                    (text if labels.modality[a] == "text" else image).add(a)
        for q in dev_q:
            legal |= set(dev_gt[q]["G"])
    del full_index
    sub = _subset_labels(labels, chosen, utf8_sorted(legal), utf8_sorted(text), utf8_sorted(image))
    universe = set(chosen) | set(dev_q) | legal | text | image
    bank = make_bank(paths, device=device, only=universe, gpu_token_bytes=256 * 2**20)
    sub_z = bank.z_store
    index = raw_mod.RawIndex(sub_z, sub, device)
    report: dict = {"queries": chosen, "dev_queries": dev_q, "legal": len(legal), "text": len(text), "image": len(image)}

    def step_raw():
        pools, reservoirs = {}, {}
        for q in chosen:
            pool, res = raw_mod.raw_query_pool(index, rows, "train", q, model_sha="probe_z")
            pools[q], reservoirs[q] = pool, res
            pool.validate(expected_generator="raw", query_id=q)
        et = {}
        for e in utf8_sorted(sub.epos):
            et[e] = [t for t, _ in index.second_hops([e], 256)[e]]
        dev_pools = {q: raw_mod.raw_query_pool(index, rows, "dev", q, model_sha="probe_z")[0] for q in dev_q}
        dup = sum(1 for pl in pools.values() for t, bag in pl.retained_paths.items() if len(bag) > 1)
        shared = {}
        for pl in pools.values():
            for t, bag in pl.retained_paths.items():
                for e in bag:
                    shared.setdefault((pl.query_id, e), set()).add(t)
        report["raw"] = {"pools": len(pools), "multi_path_targets": dup, "evidence_shared_across_targets": sum(1 for v in shared.values() if len(v) > 1)}
        assert report["raw"]["evidence_shared_across_targets"] > 0, "small chain must contain duplicated E across targets"
        return pools, reservoirs, et, dev_pools
    pools, reservoirs, et, dev_pools = step_raw()
    report["raw"]["status"] = "PASS"

    lists = {}
    def step_l0():
        nonlocal lists
        # build_l0 writes to work/lists; replicate its per-anchor construction on the subset
        for record in sub.edge_anchors:
            positives = utf8_sorted(record["positive_ids"])
            reservoir = lists_mod._reservoir_for(record, reservoirs, et, qt_source="qt_reservoir")
            hard = [x for x in reservoir if x not in set(positives)][:lists_mod.HARD_N]
            ro = lists_mod._random_order(sub.library(record["relation"]), set(positives) | set(hard), (root, "probe-L0", seed, record["relation"], record["anchor_id"]))
            lists[record["item_id"]] = lists_mod._finish_list(record, positives, hard, ro, order_namespace=(root, "probe-L0-order", seed, record["relation"], record["anchor_id"]))
        relations = {r["relation"] for r in lists.values()}
        assert relations == {"QT", "Q_text", "Q_image", "text_T", "image_T"}, relations
        assert all(r["positives"] for r in lists.values())
        return {"lists": len(lists), "relations": sorted(relations), "active": sum(r["active"] for r in lists.values())}
    p.run("A20", "small chain: raw pools + L0 five relations", lambda: (step_l0()))

    def step_fast_scoring():
        torch.manual_seed(7)
        model = teacher_mod.make_teacher(seed + 999).to(device).eval()
        with torch.no_grad():
            for prm in model.parameters():
                prm.add_(0.01 * torch.randn_like(prm))
        report_rows = {}
        by_relation = {}
        for row in lists.values():
            by_relation.setdefault(row["relation"], row)
        for relation, row in by_relation.items():
            ref = teacher_mod.pair_logits_reference(model, bank, row["anchor_id"], row["candidates"], device)
            fast = teacher_mod.pair_logits(model, bank, row["anchor_id"], row["candidates"], device)
            g_ref = torch.autograd.grad(ref.sum(), list(model.parameters()), allow_unused=True)
            g_fast = torch.autograd.grad(fast.sum(), list(model.parameters()), allow_unused=True)
            val = float((ref - fast).abs().max()); scale = float(ref.abs().max())
            gd = max(float((a - b).abs().max()) for a, b in zip(g_ref, g_fast) if a is not None)
            gs = max(float(a.abs().max()) for a in g_ref if a is not None)
            assert val <= 1e-4 * max(1.0, scale) and gd <= 1e-4 * max(1.0, gs), (relation, val, gd)
            report_rows[relation] = {"value_max_abs_diff": val, "value_scale": scale, "grad_max_abs_diff": gd, "grad_scale": gs}
        q = chosen[0]; targets = sub.legal[:5]
        evidence = [sub.canonical_text[0], sub.canonical_image[0]]
        flat = [(e, t) for t in targets for e in evidence]
        ref = teacher_mod.triplet_logits_reference(model, bank, q, flat, device)
        fast = teacher_mod.triplet_logits(model, bank, q, flat, device)
        g_ref = torch.autograd.grad(ref.sum(), list(model.parameters()), allow_unused=True)
        g_fast = torch.autograd.grad(fast.sum(), list(model.parameters()), allow_unused=True)
        val = float((ref - fast).abs().max()); gd = max(float((a - b).abs().max()) for a, b in zip(g_ref, g_fast) if a is not None)
        gs = max(float(a.abs().max()) for a in g_ref if a is not None)
        assert val <= 1e-4 * max(1.0, float(ref.abs().max())) and gd <= 1e-4 * max(1.0, gs), (val, gd)
        report_rows["triplet"] = {"value_max_abs_diff": val, "grad_max_abs_diff": gd, "grad_scale": gs}
        return report_rows
    p.run("A16/OPT", "batched Teacher scoring equals per-pair reference (values and gradients, five relations + QET)", step_fast_scoring)

    tdir = out_dir / "tmp"
    tdir.mkdir(parents=True, exist_ok=True)
    t_init = teacher_mod.make_teacher(seed + 1000)
    init_sha = runlog.state_sha(t_init.state_dict())
    t_boot_result = {}
    def step_tboot():
        model = teacher_mod.FreshPathTeacher(**teacher_mod.TEACHER_KWARGS).float()
        model.load_state_dict(t_init.state_dict())
        before = {k: v.clone() for k, v in model.state_dict().items()}
        res = teacher_mod.train_lists(model, bank, lists, paths=paths, seed=seed, stage="T_BOOT", stage_dir=tdir / "T_BOOT", root=root,
                                      epochs=1, parents={"T_INIT": init_sha}, device=device, inputs={"probe": True}, log=lambda *_: None)
        after = model.state_dict()
        changed = [k for k in before if not torch.equal(before[k], after[k].cpu())]
        assert any("poolers" in k for k in changed), "learned poolers must be trainable in T_BOOT"
        assert res["counters"]["updates"] == -(-sum(r["active"] for r in lists.values()) // 8)
        t_boot_result["model"] = model.eval()
        return {"updates": res["counters"]["updates"], "changed_tensors": len(changed), "total_tensors": len(before)}
    p.run("A20", "small chain: T_INIT -> T_BOOT one real epoch (all params incl. poolers update)", step_tboot)

    l1 = {}
    def step_l1():
        nonlocal l1
        l1 = lists_mod.refresh_lists(t_boot_result["model"], bank, sub, lists, reservoirs, et, seed=seed, root=root, stage_tag="probe-L1",
                                     qt_source="qt_reservoir", device=device, pair_scorer=lambda m, b, a, pool, d: teacher_mod.pair_logits(m, b, a, pool, d), log=lambda *_: None)
        row = next(iter(l1.values()))
        assert len(row["teacher_logits"]) == len(row["candidates"])
        # teacher logits equal a fresh forward
        with torch.no_grad():
            fresh = teacher_mod.pair_logits(t_boot_result["model"].eval(), bank, row["anchor_id"], row["candidates"], device).cpu()
        diff = float((fresh - torch.tensor(row["teacher_logits"])).abs().max())
        assert diff < 1e-3, f"stored T_BOOT logits differ from a fresh forward by {diff}"
        return {"lists": len(l1), "hard_changed": sum(1 for k in lists if lists[k]["hard"] != l1[k]["hard"]), "max_abs_diff": diff}
    p.run("A20", "small chain: L1 refresh with T_BOOT logits attached and verified", step_l1)

    students = {}
    def step_c1():
        for arm in ("S_SUP_NATIVE", "S_KD_NATIVE"):
            model = student_mod.make_student(basis, qt_only=False)
            students[arm] = model
        assert runlog.state_sha(students["S_SUP_NATIVE"].state_dict()) == runlog.state_sha(students["S_KD_NATIVE"].state_dict())
        res = {}
        for arm, model in students.items():
            r = student_mod.train_c1(model, bank, l1, paths=paths, seed=seed, stage=f"{arm}_C1", stage_dir=tdir / f"{arm}_C1", root=root, scope="native",
                                     kd=("KD" in arm), parents={}, device=device, inputs={"probe": True}, log=lambda *_: None)
            res[arm] = r["counters"]["updates"]
        assert runlog.state_sha(students["S_SUP_NATIVE"].state_dict()) != runlog.state_sha(students["S_KD_NATIVE"].state_dict())
        return {"updates": res, "snapshots": [x["snapshot"] for x in r["snapshots"]]}
    p.run("A20", "small chain: paired S_SUP/S_KD C1 from identical init (real update)", step_c1)

    own_pools = {}
    def step_own():
        for arm, model in students.items():
            model.eval()
            ret = OwnRetriever(model, bank, sub, rows, device=device, seed=seed, generator_id=f"probe:{arm}", model_sha=runlog.state_sha(model.state_dict()))
            own_pools[arm] = {q: ret.pool("train", q) for q in chosen}
            # A16: training score == exact == ANN query @ keys
            q = chosen[0]
            exact = ret.direct_exact(q, 5)
            z_q = bank.z(q).to(device)
            with torch.no_grad():
                train_scores = model.score("QT", z_q, bank.z_many([t for t, _ in exact]).to(device)).cpu()
            assert torch.allclose(train_scores, torch.tensor([s for _, s in exact]), atol=1e-4)
            ann, _ = ret.direct(q, 5)
            ann_ids = [t for t, _ in ann]
            assert set(ann_ids) & set(t for t, _ in exact), "ANN top5 shares nothing with exact top5"
            for kind, rel in (("text", "Q_text"), ("image", "Q_image")):
                hits = ret.first_hop(q, kind, 3)
                with torch.no_grad():
                    s = model.score(rel, z_q, bank.z_many([e for e, _ in hits]).to(device)).cpu()
                assert torch.allclose(s, torch.tensor([v for _, v in hits]), atol=1e-4)
                e = hits[0][0]
                sh = ret.second_hop(e, kind, 3)
                with torch.no_grad():
                    s2 = model.score(f"{kind}_T", bank.z(e).to(device), bank.z_many([t for t, _ in sh]).to(device)).cpu()
                assert torch.allclose(s2, torch.tensor([v for _, v in sh]), atol=1e-4)
        return {"A16": "train == exact == ANN rescored for QT/Q_text/Q_image/text_T/image_T"}
    p.run("A16/A20", "small chain: own HNSW retrieval; non-identity P/R scores identical across train/exact/ANN", step_own)

    c2_items = {}
    c2_logits = {}
    def step_c2():
        nonlocal c2_items, c2_logits
        c2_items = graphs.build_native_c2_graph(sub, {"raw": pools, **own_pools}, root=root, seed=seed)
        res = graphs.precompute_c2_teacher(t_boot_result["model"], bank, c2_items, device=device, teacher_sha="probe", store_path=tdir / "c2_store.pt", log=lambda *_: None)
        c2_logits = res["logits"]
        # verify one ET entry against a fresh pair forward
        q = next(q for q in c2_items if c2_items[q]["views"] and c2_items[q]["views"][0]["e_index"])
        it = c2_items[q]; v = it["views"][0]
        e = it["evidence"][v["e_index"][0]]; t = it["targets"][v["t_index"][0]]
        with torch.no_grad():
            fresh = teacher_mod.pair_logits(t_boot_result["model"], bank, q, [e], device)[0] + teacher_mod.pair_logits(t_boot_result["model"], bank, e, [t], device)[0]
        assert abs(float(fresh) - float(c2_logits[q]["ET"][0][0])) < 1e-3
        updates = {}
        for arm, model in students.items():
            r = student_mod.train_c2(model.train(), bank, c2_items, paths=paths, seed=seed, stage=f"{arm}_C2", stage_dir=tdir / f"{arm}_C2", root=root, scope="native",
                                     kd=("KD" in arm), teacher_logits=c2_logits if "KD" in arm else None, parents={}, device=device, inputs={"probe": True}, log=lambda *_: None)
            updates[arm] = r["counters"]
        return {"graph": graphs.graph_stats(c2_items), "updates": {k: v["updates"] for k, v in updates.items()}, "e_queries": {k: v["e_queries"] for k, v in updates.items()}}
    p.run("A20", "small chain: C2 graph from three generators, T_BOOT logits dedup-verified, real C2 step", step_c2)

    hard = {}
    kd_pools = {}
    def step_hard():
        nonlocal hard, kd_pools
        model = students["S_KD_NATIVE"].eval()
        ret = OwnRetriever(model, bank, sub, rows, device=device, seed=seed, generator_id="probe:S_KD_NATIVE_C2", model_sha=runlog.state_sha(model.state_dict()))
        kd_pools = {q: ret.pool("train", q) for q in chosen}
        hard = graphs.mine_hard32(sub, kd_pools, generator_id="probe:S_KD_NATIVE_C2")
        assert all(set(v["hard32"]).isdisjoint(sub.queries[q]["G"]) for q, v in hard.items())
        return {"mean_hard": float(np.mean([len(v["hard32"]) for v in hard.values()]))}
    p.run("A20", "small chain: hard32 from S_KD own Direct ANN (non-G)", step_hard)

    t_qt_result = {}
    def step_tqt():
        merged = graphs.apply_hard32(lists, hard, root=root, seed=seed)
        model = teacher_mod.FreshPathTeacher(**teacher_mod.TEACHER_KWARGS).float()
        model.load_state_dict(t_init.state_dict())
        assert runlog.state_sha(model.state_dict()) == init_sha, "T_QT must start from untrained T_INIT"
        res = teacher_mod.train_lists(model, bank, merged, paths=paths, seed=seed, stage="T_QT", stage_dir=tdir / "T_QT", root=root, epochs=1,
                                      parents={"T_INIT": init_sha}, device=device, inputs={"probe": True}, log=lambda *_: None)
        t_qt_result["model"] = model.eval()
        return {"updates": res["counters"]["updates"], "qt_extra": sum(len(r.get("hard32_extra", ())) for r in merged.values())}
    p.run("A20/A23", "small chain: T_QT from untrained T_INIT with hard32 appended", step_tqt)

    t_path_result = {}
    def step_tpath():
        items = graphs.build_path_items(sub, kd_pools, pools, own_generator="probe:S_KD_NATIVE_C2", reservoirs=reservoirs, et_reservoir=et, root=root, seed=seed)
        anchors = graphs.precompute_anchor_logits(t_qt_result["model"], bank, items, device=device, teacher_sha="probe_qt", store_path=tdir / "anchor_store.pt", log=lambda *_: None)
        model = teacher_mod.FreshPathTeacher(**teacher_mod.TEACHER_KWARGS).float()
        model.load_state_dict(t_qt_result["model"].state_dict())
        model = model.to(device)
        before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        # A24: view_scores really consumes three objects; slot order check against a loop
        q = next(q for q in items if items[q]["natural"])
        it = items[q]
        model.eval()
        with torch.no_grad():
            S, f0, f = teacher_mod.view_scores(model, bank, q, it["targets"], it["natural"], device)
            flat = [(e, t) for t in it["targets"] for e in it["natural"].get(t, ())]
            loop = torch.stack([teacher_mod.triplet_logits(model, bank, q, [(e, t)], device)[0] for e, t in flat])
            assert torch.allclose(f.cpu(), loop.cpu(), atol=1e-3), "triplet slots differ from per-slot loop"
            # E content matters: swapping the evidence object changes the score
            e0, t0 = flat[0]
            other = next(e for e in it["natural"][t0] if e != e0) if len(it["natural"][t0]) > 1 else None
            if other is None:
                other = next(e for tt in it["natural"] for e in it["natural"][tt] if e != e0)
            a = teacher_mod.triplet_logits(model, bank, q, [(e0, t0)], device)[0]
            b = teacher_mod.triplet_logits(model, bank, q, [(other, t0)], device)[0]
            assert not torch.allclose(a, b), "E content did not change the QET score"
        res = teacher_mod.train_path(model, bank, items, paths=paths, seed=seed, stage_dir=tdir / "T_PATH", root=root, parents={}, device=device,
                                     inputs={"probe": True}, anchor_logits=anchors, log=lambda *_: None)
        after = model.state_dict()
        changed = {k for k in before if not torch.equal(before[k], after[k].cpu())}
        frozen_changed = [k for k in changed if k.startswith(teacher_mod.PATH_FROZEN_PREFIXES)]
        assert not frozen_changed, frozen_changed
        assert any(k.startswith("relation.") for k in changed) and any(k.startswith("scoring_head.") for k in changed)
        t_path_result["model"] = model.eval()
        return {"updates": res["counters"]["updates"], "conditional_lists": sum(len(i["conditions"]) for i in items.values()), "changed": len(changed), "frozen_untouched": True}
    p.run("A20/A24/A25", "small chain: T_PATH from T_QT; real QET forward; frozen encoders; no presence margin", step_tpath)

    def step_eval():
        from . import evaluate as ev, metrics

        model = students["S_KD_NATIVE"].eval()
        ret = OwnRetriever(model, bank, sub, rows, device=device, seed=seed, generator_id="probe:S_KD_NATIVE_C2", model_sha=runlog.state_sha(model.state_dict()))
        dev_own = {q: ret.pool("dev", q) for q in dev_q}
        scorer = ev.TeacherScorer(t_path_result["model"], bank, device, name="probe_path")
        used = [e for q in dev_q for bag in dev_own[q].retained_paths.values() for e in bag]
        swap = ev.swap_map(used, {})
        res = ev.teacher_readouts(scorer, dev_own, generator_id="probe:S_KD_NATIVE_C2", with_paths=True, swap=swap, queries=dev_q, log=lambda *_: None)
        gold = {q: [t for t in dev_gt[q]["G"] if t in legal] for q in dev_q}
        gold = {q: g for q, g in gold.items() if g}
        # independent recomputation from saved rankings: recall of Real|C150 via kernel
        kernels = _kernels()
        rk = res["rankings"]["Real|C150"]
        r10 = float(np.mean([kernels.recall_at_k(gold[q], rk[q], 10) for q in gold]))
        r10_prod = metrics.grouped(gold, {q: "implicit" for q in gold}, rk)["overall"]["R10"]
        assert abs(r10 - r10_prod) < 1e-12
        # Real ranking uses the own path E ids only
        for q in dev_q:
            for t, slots in res["logits"][q].get("paths", {}).items():
                assert [e for e, _ in slots] == dev_own[q].retained_paths[t]
        try:
            ev.teacher_readouts(scorer, dev_pools, generator_id="probe:S_KD_NATIVE_C2", with_paths=True, swap=None, queries=dev_q, log=lambda *_: None)
            raise AssertionError("raw dev pools were accepted for the own generator")
        except ValueError:
            pass
        return {"dev_queries": len(gold), "R10_real_C150": r10, "swap_effective": res["swap"]["effective"], "forwards": scorer.forwards}
    p.run("A20/A17/A29", "small chain: own ANN -> T_PATH real paths -> rankings -> independent metric; raw pool rejected", step_eval)
    write_json(out_dir / "SMALL_CHAIN_REPORT.json", report)
    shutil.rmtree(tdir, ignore_errors=True)


def cmd_verify_integration(paths: Paths, *, seed: int, device: str) -> dict:
    runlog.enforce_precision()
    out_dir = paths.work_dir / f"seed{seed}" / "VERIFY_INTEGRATION"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = Probe()
    probe_a12_c1_formula(p)
    probe_a13_c2_formula(p)
    probe_a14_slot_order(p)
    probe_a15_read_q(p)
    probe_a17_pool_identity(p)
    probe_a18_d1(p)
    probe_a19_chunks(p)
    started = time.time()
    try:
        small_chain(paths, seed=seed, device=device, out_dir=out_dir, p=p)
    except Exception as error:
        p.results.append({"item": "A20", "name": "small chain aborted", "status": "FAIL", "error": repr(error),
                          "traceback": traceback.format_exc()[-3000:]})
    status = "PASS" if all(r["status"] == "PASS" for r in p.results) else "FAIL"
    report = {"status": status, "seed": seed, "device": device, "code_lock_sha256": runlog.code_lock_sha(),
              "code_lock": runlog.code_lock(), "environment": runlog.gpu_info(), "elapsed_seconds": time.time() - started,
              "results": p.results, "note": "probe models/state discarded; formal stages start from their own init"}
    write_json(out_dir / "VERIFY_INTEGRATION.json", report)
    if status != "PASS":
        raise RuntimeError("integration acceptance failed; formal training must not start")
    return {"status": status, "checks": [(r["item"], r["name"], r["status"]) for r in p.results]}
