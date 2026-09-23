"""Stage implementations behind ``run_fresh_recovery.py`` (SPEC 16).

Each stage reads only this run's outputs (plus the approved pure caches),
writes its own directory under ``work/seed{seed}/<STAGE>/`` with PRE_RUN /
POST_RUN receipts, and never opens a historical task artifact.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from . import graphs, lists as lists_mod, raw as raw_mod, runlog, student as student_mod, teacher as teacher_mod
from .config import Paths, RELATIONS
from .data import (Labels, load_basis, load_labels, load_row_store, load_split_gt, load_z, make_bank,
                   split_query_ids, utf8_sorted)
from .io import sha256_file, sha256_json, write_json
from .pools import PoolRecord, pool_summary
from .retrieval import OwnRetriever

NATIVE_ARMS = ("S_SUP_NATIVE", "S_KD_NATIVE")
QT_ARMS = ("S_QT_SUP", "S_QT_KD")
ARMS = NATIVE_ARMS + QT_ARMS
D_FLOOR = -0.02
U_FLOOR = -0.03
CATASTROPHIC = -0.10


class Runtime:
    def __init__(self, paths: Paths, *, seed: int, device: str | None = None) -> None:
        self.paths = paths
        self.seed = seed
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.labels = load_labels(paths)
        self.root = lists_mod.data_root_id(paths)
        self.basis, self.basis_sha = load_basis(paths)
        self.rows = load_row_store(paths)
        self._bank = None
        self.log = print

    def bank(self, *, gpu_token_bytes: int = 2 * 2**30):
        if self._bank is None:
            self._bank = make_bank(self.paths, device=self.device, gpu_token_bytes=gpu_token_bytes)
        return self._bank

    def seed_dir(self) -> Path:
        return self.paths.work_dir / f"seed{self.seed}"

    def stage_dir(self, stage: str) -> Path:
        return self.seed_dir() / stage

    def lists_dir(self) -> Path:
        return self.paths.work_dir / "lists" / f"seed{self.seed}"

    def raw_pools(self, split: str) -> dict[str, PoolRecord]:
        return raw_mod.load_pools(self.paths.work_dir / "raw" / split / "pools.pkl")

    def dev_gt(self, split: str = "dev") -> dict[str, dict]:
        return load_split_gt(self.paths, split)

    def checkpoint_sha(self, stage: str, name: str = "checkpoint.pt") -> str:
        path = self.stage_dir(stage) / name
        return sha256_file(path.resolve())

    def stage_log(self, stage: str):
        self.log = runlog.StageLog(self.stage_dir(stage))
        return self.log


# ------------------------------------------------------------- preparation ---


def cmd_build_rows(paths: Paths) -> dict:
    from .data import build_row_store

    return build_row_store(paths)


def cmd_build_raw(paths: Paths, *, split: str, device: str) -> dict:
    if split == "test":
        verify_model_lock(paths, seed=13)
    return raw_mod.build_raw(paths, split=split, device=device)


def verify_model_lock(paths: Paths, *, seed: int) -> dict:
    """Reject test reads if any frozen source, configuration, or model changed."""
    lock_path = paths.work_dir / f"seed{seed}" / "MODEL_LOCK.json"
    if not lock_path.exists():
        raise RuntimeError(f"test firewall: seed{seed} MODEL_LOCK.json is required")
    locked = json.loads(lock_path.read_text(encoding="utf-8"))
    if locked["protocol_sha256"] != sha256_file(paths.protocol_path) or locked["config"]["code_lock_sha256"] != runlog.code_lock_sha():
        raise RuntimeError("test firewall: source or protocol changed after MODEL_LOCK")
    for selection, name in (("selection_native", "NATIVE"), ("selection_qt", "QT")):
        if locked["config"][selection] != sha256_file(paths.work_dir / f"seed{seed}" / f"SELECTION_{name}_C2.json"):
            raise RuntimeError(f"test firewall: {name} selection changed after MODEL_LOCK")
    for name, expected in locked["models"].items():
        checkpoint = paths.work_dir / f"seed{seed}" / name / ("selected.pt" if name.startswith("S_") else
                                                                 "state.pt" if name == "T_INIT" else "checkpoint.pt")
        if sha256_file(checkpoint) != expected:
            raise RuntimeError(f"test firewall: {name} checkpoint changed after MODEL_LOCK")
    return locked


def cmd_build_lists(paths: Paths, *, seed: int) -> dict:
    labels = load_labels(paths)
    reservoirs = raw_mod.load_reservoirs(paths, "train")
    et = raw_mod.load_et_reservoir(paths)
    return lists_mod.build_l0(paths, labels, reservoirs, et, seed=seed)


def cmd_init_teacher(paths: Paths, *, seed: int) -> dict:
    model = teacher_mod.make_teacher(seed)
    stage_dir = paths.work_dir / f"seed{seed}" / "T_INIT"
    stage_dir.mkdir(parents=True, exist_ok=True)
    sha = runlog.save_checkpoint(stage_dir / "state.pt", model=model, optimizer=None, stage="T_INIT", seed=seed,
                                 paths=paths, parents={}, counters={},
                                 extra={"init": "xavier_uniform Linear/zeros bias; LN ones/zeros; Embedding/REL/SEP/pool queries N(0,0.02)",
                                        "kwargs": teacher_mod.TEACHER_KWARGS})
    report = {"stage": "T_INIT", "seed": seed, "state_sha256": runlog.state_sha(model.state_dict()),
              "checkpoint_sha256": sha, "parameters": sum(p.numel() for p in model.parameters()),
              "torch_seed": "SHA256(namespace('init','teacher',seed))[:4]"}
    write_json(stage_dir / "INIT_LINEAGE.json", report)
    return report


def _dev_raw_rerank_hook(rt: Runtime, model, bank, raw_dev: Mapping[str, PoolRecord], gt: Mapping[str, dict]):
    """Per-epoch Teacher probe: Raw C150 + QT rerank on dev (R@10)."""
    from .evaluate import TeacherScorer, rank_by

    def hook(epoch: int) -> dict:
        scorer = TeacherScorer(model, bank, rt.device, name=f"epoch{epoch}")
        rankings = {}
        for qid in gt:
            f0 = scorer.f0(qid, raw_dev[qid].C150)
            rankings[qid] = rank_by(f0)
        gold = {q: gt[q]["G"] for q in gt}
        kinds = {q: gt[q]["kind"] for q in gt}
        from . import metrics

        table = metrics.grouped(gold, kinds, rankings, pools={q: raw_dev[q].C150 for q in gt}, ks=(10, 20, 50))
        return {"system": "Raw_C150_plus_teacher", "metrics": table}
    return hook


def cmd_train_bootstrap(paths: Paths, *, seed: int, device: str) -> dict:
    rt = Runtime(paths, seed=seed, device=device)
    stage = "T_BOOT"
    log = rt.stage_log(stage)
    lists = lists_mod.load_lists(rt.lists_dir() / "L0.pkl")
    init_path = rt.stage_dir("T_INIT") / "state.pt"
    payload = runlog.load_checkpoint(init_path, expect_stage="T_INIT")
    model = teacher_mod.FreshPathTeacher(**teacher_mod.TEACHER_KWARGS).float()
    model.load_state_dict(payload["state_dict"])
    bank = rt.bank()
    raw_dev = rt.raw_pools("dev")
    gt = rt.dev_gt()
    hook = _dev_raw_rerank_hook(rt, model, bank, raw_dev, gt)
    return teacher_mod.train_lists(
        model, bank, lists, paths=paths, seed=seed, stage=stage, stage_dir=rt.stage_dir(stage), root=rt.root,
        epochs=2, parents={"T_INIT": sha256_file(init_path)}, device=rt.device,
        inputs={"lists": "L0.pkl", "lists_sha256": sha256_file(rt.lists_dir() / "L0.pkl"), "labels": rt.labels.files},
        epoch_hook=hook, log=log)


def _pair_scorer(model, bank, anchor, pool, device):
    return teacher_mod.pair_logits(model, bank, anchor, pool, device, chunk=256)


def cmd_refresh_lists(paths: Paths, *, seed: int, device: str) -> dict:
    rt = Runtime(paths, seed=seed, device=device)
    stage = "L1"
    log = rt.stage_log(stage)
    stage_dir = rt.stage_dir(stage)
    model, payload = teacher_mod.load_teacher(rt.stage_dir("T_BOOT") / "checkpoint.pt", rt.device, expect_stage="T_BOOT")
    teacher_sha = runlog.state_sha(model.state_dict())
    bank = rt.bank()
    reservoirs = raw_mod.load_reservoirs(paths, "train")
    et = raw_mod.load_et_reservoir(paths)
    l0 = lists_mod.load_lists(rt.lists_dir() / "L0.pkl")
    ldir = lists_mod.load_lists(rt.lists_dir() / "Ldir.pkl")
    runlog.pre_run(stage_dir, stage=stage, seed=seed, paths=paths,
                   parents={"T_BOOT": rt.checkpoint_sha("T_BOOT"), "T_BOOT_state": teacher_sha},
                   inputs={"L0": sha256_file(rt.lists_dir() / "L0.pkl"), "Ldir": sha256_file(rt.lists_dir() / "Ldir.pkl")},
                   initial_state_sha=teacher_sha, optimizer_state="none_frozen_teacher",
                   config={"hard": lists_mod.HARD_N, "random": lists_mod.RANDOM_N, "scoring": "T_BOOT pair logits on reservoir u P u random_order"})
    started = time.time()
    l1 = lists_mod.refresh_lists(model, bank, rt.labels, l0, reservoirs, et, seed=seed, root=rt.root, stage_tag="L1",
                                 qt_source="qt_reservoir", device=rt.device, pair_scorer=_pair_scorer, log=log)
    ldir1 = lists_mod.refresh_lists(model, bank, rt.labels, ldir, reservoirs, et, seed=seed, root=rt.root, stage_tag="Ldir1",
                                    qt_source="qt_top256", device=rt.device, pair_scorer=_pair_scorer, log=log)
    l1_sha = lists_mod.save_lists(rt.lists_dir() / "L1.pkl", l1)
    ldir1_sha = lists_mod.save_lists(rt.lists_dir() / "Ldir1.pkl", ldir1)
    changed = sum(1 for k in l0 if l0[k]["hard"] != l1[k]["hard"])
    report = {"L1": lists_mod.list_stats(l1), "Ldir1": lists_mod.list_stats(ldir1), "L1_sha256": l1_sha,
              "Ldir1_sha256": ldir1_sha, "teacher_sha256": teacher_sha, "hard_lists_changed": changed,
              "elapsed_seconds": time.time() - started}
    write_json(rt.lists_dir() / "L1_REPORT.json", report)
    runlog.post_run(stage_dir, status="COMPLETE", counters={"lists": len(l1) + len(ldir1), "elapsed": report["elapsed_seconds"]},
                    outputs={"L1.pkl": l1_sha, "Ldir1.pkl": ldir1_sha}, notes=report)
    return report


# ---------------------------------------------------------- student stages ---


def _student_dev_eval(rt: Runtime, model, bank, *, generator_id: str, model_sha: str, qt_only: bool,
                      gt: Mapping[str, dict], out_dir: Path, tag: str, save_pools: bool = False) -> dict:
    """Own retrieval on full dev for one Student state; returns the selection metrics."""
    from . import metrics

    retriever = OwnRetriever(model, bank, rt.labels, rt.rows, device=rt.device, seed=rt.seed,
                             generator_id=generator_id, model_sha=model_sha, qt_only=qt_only)
    queries = utf8_sorted(gt)
    pools = {q: retriever.pool("dev", q) for q in queries}
    gold = {q: gt[q]["G"] for q in queries}
    kinds = {q: gt[q]["kind"] for q in queries}
    direct = {q: pools[q].direct_ids for q in queries}
    exact = {q: pools[q].direct_exact for q in queries}
    evidence = {q: pools[q].evidence_ids for q in queries}
    c150 = {q: pools[q].C150 for q in queries}
    u = {q: pools[q].U for q in queries}
    table = {
        "Direct": metrics.grouped(gold, kinds, direct, pools=direct, ks=(10, 20, 50)),
        "Direct_exact": metrics.grouped(gold, kinds, exact, pools=exact, ks=(10, 20, 50)),
        "C150": metrics.grouped(gold, kinds, c150, pools=c150, ks=(10, 20, 50)),
        "U": metrics.grouped(gold, kinds, u, pools=u, ks=(10, 20, 50)),
    }
    if not qt_only:
        table["Evidence"] = metrics.grouped(gold, kinds, evidence, pools=evidence, ks=(10, 20, 50))
    summary = pool_summary(pools, gold)
    r10_direct = table["Direct"]["overall"]["R10"]
    r10_evidence = table["Evidence"]["overall"]["R10"] if not qt_only else None
    result = {
        "tag": tag, "generator_id": generator_id, "model_sha": model_sha, "retrieval": retriever.meta,
        "metrics": table, "pool_summary": summary,
        "summary": {"R10_direct": r10_direct, "R10_evidence": r10_evidence,
                    "coverage_D100": summary["coverage"]["D100"], "coverage_U": summary["coverage"]["U"],
                    "coverage_C150": summary["coverage"]["C150"],
                    "coverage_U": summary["coverage"]["U"],
                    "mean_C150_coverage_overall": table["C150"]["overall"].get("coverage"),
                    "mean_C150_coverage_implicit": table["C150"].get("implicit", {}).get("coverage"),
                    "mean_U_coverage_overall": table["U"]["overall"].get("coverage"),
                    "mean_Direct100_R10_overall": table["Direct"]["overall"]["R10"],
                    "native_key": (r10_direct + r10_evidence) / 2 if r10_evidence is not None else None},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"dev_{tag}.json", result)
    from .evaluate import save_rankings

    save_rankings(out_dir / f"dev_{tag}_rankings.json.gz",
                  {"Direct": direct, "Direct_exact": exact, "Evidence": evidence, "C150": c150, "U": u})
    if save_pools:
        raw_mod.save_pools(out_dir / f"dev_{tag}_pools.pkl", pools)
    del retriever
    return result


def cmd_eval_init(paths: Paths, *, seed: int, device: str) -> dict:
    """PCA-init reference (P0, identity R): the coverage baseline for selection floors."""
    rt = Runtime(paths, seed=seed, device=device)
    bank = rt.bank()
    gt = rt.dev_gt()
    out = {}
    for qt_only, name in ((False, "PCA_INIT_NATIVE"), (True, "PCA_INIT_QT")):
        model = student_mod.make_student(rt.basis, qt_only=qt_only).to(rt.device).eval()
        sha = runlog.state_sha(model.state_dict())
        out[name] = _student_dev_eval(rt, model, bank, generator_id=name, model_sha=f"init:{sha[:16]}", qt_only=qt_only,
                                      gt=gt, out_dir=rt.stage_dir("INIT_REFERENCE"), tag=name.lower(), save_pools=True)
    write_json(rt.stage_dir("INIT_REFERENCE") / "INIT_REFERENCE.json",
               {name: r["summary"] for name, r in out.items()} | {"basis_sha256": rt.basis_sha})
    return {name: r["summary"] for name, r in out.items()}


def _arm_flags(arm: str) -> tuple[bool, bool]:
    return ("KD" in arm), arm.startswith("S_QT")


def cmd_train_c1(paths: Paths, *, seed: int, arm: str, device: str) -> dict:
    rt = Runtime(paths, seed=seed, device=device)
    kd, qt_only = _arm_flags(arm)
    stage = f"{arm}_C1"
    log = rt.stage_log(stage)
    list_name = "Ldir1.pkl" if qt_only else "L1.pkl"
    lists = lists_mod.load_lists(rt.lists_dir() / list_name)
    model = student_mod.make_student(rt.basis, qt_only=qt_only)
    bank = rt.bank()
    gt = rt.dev_gt()

    def hook(tag: str, m) -> dict:
        sha = runlog.state_sha(m.state_dict())
        return _student_dev_eval(rt, m, bank, generator_id=f"{stage}:{tag}", model_sha=sha, qt_only=qt_only, gt=gt,
                                 out_dir=rt.stage_dir(stage) / "dev", tag=tag)

    parents = {"PCA_basis": rt.basis_sha, "L1": sha256_file(rt.lists_dir() / list_name),
               "T_BOOT": rt.checkpoint_sha("T_BOOT") if kd else "not_used_by_SUP"}
    return student_mod.train_c1(model, bank, lists, paths=paths, seed=seed, stage=stage, stage_dir=rt.stage_dir(stage),
                                root=rt.root, scope="qt" if qt_only else "native", kd=kd, parents=parents,
                                device=rt.device, inputs={"lists": list_name, "labels": rt.labels.files},
                                snapshot_hook=hook, log=log)


def _select(records_by_arm: Mapping[str, list[dict]], init: Mapping[str, float], *, native: bool) -> dict:
    """SPEC 8.4 / 9.3 paired common-fraction selection with coverage floors."""
    tags = [r["snapshot"] for r in next(iter(records_by_arm.values()))]
    rows = []
    stop = None
    for tag in tags:
        arms = {arm: next(r for r in recs if r["snapshot"] == tag)["dev"]["summary"] for arm, recs in records_by_arm.items()}
        eligible = True
        reasons = []
        for arm, s in arms.items():
            d_delta = s["coverage_D100"] - init["coverage_D100"]
            if d_delta < CATASTROPHIC:
                stop = stop or {"code": "STOP_GEOMETRY_FAILURE", "arm": arm, "tag": tag, "coverage_D100": s["coverage_D100"],
                                "init": init["coverage_D100"]}
            if d_delta < D_FLOOR:
                eligible = False
                reasons.append(f"{arm}: D100 coverage {s['coverage_D100']:.4f} < init{D_FLOOR:+.2f}")
            if native and s["coverage_U"] - init["coverage_U"] < U_FLOOR:
                eligible = False
                reasons.append(f"{arm}: U coverage {s['coverage_U']:.4f} < init{U_FLOOR:+.2f}")
        # Selection is coverage-first and uses the same C150/U/Direct keys for
        # both paired native arms.  The final tuple element is the snapshot
        # order, so an exact tie keeps the earlier cumulative point.
        key = [
            float(np.mean([s["mean_C150_coverage_overall"] for s in arms.values()])),
            float(np.mean([s.get("mean_C150_coverage_implicit", 0.0) or 0.0 for s in arms.values()])),
            float(np.mean([s["mean_U_coverage_overall"] for s in arms.values()])),
            float(np.mean([s["mean_Direct100_R10_overall"] for s in arms.values()])),
        ]
        rows.append({"tag": tag, "eligible": eligible, "reasons": reasons, "key": key, "arms": arms})
    chosen = None
    for row in rows:  # earlier fraction wins exact ties
        if row["eligible"] and (chosen is None or tuple(row["key"]) > tuple(chosen["key"])):
            chosen = row
    if stop is not None:
        chosen = None  # the paired subtree stops; no fraction may be selected
    return {"rows": rows, "selected": chosen["tag"] if chosen else None,
            "status": "STOP_GEOMETRY_FAILURE" if stop else ("STOP_NO_HEALTHY_C1" if chosen is None else "SELECTED"),
            "stop": stop, "init_reference": dict(init),
            "key": ["mean_arms_C150_coverage_overall", "mean_arms_C150_coverage_implicit",
                    "mean_arms_U_coverage_overall", "mean_arms_Direct100_R10_overall", "earlier_snapshot"],
            "floors": {"D100": D_FLOOR, "U": U_FLOOR if native else None, "catastrophic": CATASTROPHIC}}


def cmd_select(paths: Paths, *, seed: int, pair: str, phase: str) -> dict:
    rt = Runtime(paths, seed=seed)
    arms = NATIVE_ARMS if pair == "native" else QT_ARMS
    init = json.loads((rt.stage_dir("INIT_REFERENCE") / "INIT_REFERENCE.json").read_text())
    init = init["PCA_INIT_NATIVE" if pair == "native" else "PCA_INIT_QT"]
    records = {}
    for arm in arms:
        stage = f"{arm}_{phase}"
        post = rt.stage_dir(stage) / "POST_RUN.json"
        if not post.exists() or json.loads(post.read_text())["status"] != "COMPLETE":
            raise RuntimeError(f"{stage} has not completed")
        records[arm] = json.loads((rt.stage_dir(stage) / "SNAPSHOTS.json").read_text())
    if phase == "C2":
        expected = {"frac050", "frac100", "frac150", "frac200"}
        for arm in arms:
            got = {r["snapshot"] for r in records[arm]}
            if got != expected:
                raise RuntimeError(f"{arm}_C2 snapshots {sorted(got)!r} != {sorted(expected)!r}")
    result = _select(records, init, native=(pair == "native"))
    result.update({"pair": pair, "phase": phase, "arms": list(arms), "seed": seed})
    if result["selected"]:
        for arm in arms:
            stage = f"{arm}_{phase}"
            student_mod._link(rt.stage_dir(stage), "selected.pt", f"{result['selected']}.pt")
            result.setdefault("checkpoints", {})[stage] = sha256_file(rt.stage_dir(stage) / f"{result['selected']}.pt")
    write_json(rt.seed_dir() / f"SELECTION_{pair.upper()}_{phase}.json", result)
    return result


def _selected_student(rt: Runtime, stage: str):
    path = rt.stage_dir(stage) / "selected.pt"
    if not path.exists():
        raise RuntimeError(f"{stage}: no selected checkpoint (selection not done)")
    model, payload = student_mod.load_student(path, rt.basis, rt.device)
    return model, payload, sha256_file(path.resolve())


def _own_train_pools(rt: Runtime, model, bank, *, generator_id: str, model_sha: str, qt_only: bool, out_path: Path,
                     log=print) -> dict[str, PoolRecord]:
    if out_path.exists():
        pools = raw_mod.load_pools(out_path)
        if next(iter(pools.values())).model_sha == model_sha:
            log({"event": "own_pools_reused", "path": str(out_path)})
            return pools
    retriever = OwnRetriever(model, bank, rt.labels, rt.rows, device=rt.device, seed=rt.seed,
                             generator_id=generator_id, model_sha=model_sha, qt_only=qt_only)
    pools = {}
    started = time.time()
    for n, qid in enumerate(rt.labels.query_ids, 1):
        pools[qid] = retriever.pool("train", qid)
        if n % 1000 == 0:
            log({"event": "own_train_pools", "generator": generator_id, "done": n, "total": len(rt.labels.query_ids),
                 "elapsed": round(time.time() - started, 1)})
    raw_mod.save_pools(out_path, pools)
    write_json(out_path.with_suffix(".json"), {"generator_id": generator_id, "model_sha": model_sha, "retrieval": retriever.meta,
                                              "summary": pool_summary(pools, {q: rt.labels.queries[q]["G"] for q in pools}),
                                              "elapsed_seconds": time.time() - started})
    return pools


def cmd_build_c2_graphs(paths: Paths, *, seed: int, pair: str, device: str) -> dict:
    rt = Runtime(paths, seed=seed, device=device)
    stage = f"{pair.upper()}_C1_SELECTION_GRAPH"
    log = rt.stage_log(stage)
    stage_dir = rt.stage_dir(stage)
    bank = rt.bank()
    arms = NATIVE_ARMS if pair == "native" else QT_ARMS
    selection = json.loads((rt.seed_dir() / f"SELECTION_{pair.upper()}_C1.json").read_text())
    if selection["status"] != "SELECTED":
        raise RuntimeError(f"C1 selection for {pair} is {selection['status']}")
    parents = {"T_BOOT": rt.checkpoint_sha("T_BOOT")}
    pools_by_generator: dict[str, dict[str, PoolRecord]] = {}
    for arm in arms:
        model, payload, sha = _selected_student(rt, f"{arm}_C1")
        parents[f"{arm}_C1"] = sha
        pools_by_generator[arm] = _own_train_pools(rt, model, bank, generator_id=f"{arm}_C1:{selection['selected']}",
                                                   model_sha=runlog.state_sha(model.state_dict()), qt_only=(pair == "qt"),
                                                   out_path=stage_dir / f"{arm}_train_pools.pkl", log=log)
        del model
    runlog.pre_run(stage_dir, stage=stage, seed=seed, paths=paths, parents=parents,
                   inputs={"selection": selection["selected"], "raw_train": sha256_file(paths.work_dir / "raw" / "train" / "pools.pkl")},
                   initial_state_sha="n/a", optimizer_state="none", config={"pair": pair})
    reservoirs = raw_mod.load_reservoirs(paths, "train")
    if pair == "native":
        raw_pools = rt.raw_pools("train")
        items = graphs.build_native_c2_graph(rt.labels, {"raw": raw_pools, **pools_by_generator}, root=rt.root, seed=seed)
    else:
        items = graphs.build_qt_c2_graph(rt.labels, reservoirs,
                                         {arm: {q: p.direct_ids for q, p in pools.items()} for arm, pools in pools_by_generator.items()})
    graph_sha = graphs.save_graph(stage_dir / "graph.pkl", items)
    stats = graphs.graph_stats(items)
    teacher, _ = teacher_mod.load_teacher(rt.stage_dir("T_BOOT") / "checkpoint.pt", rt.device, expect_stage="T_BOOT")
    teacher_sha = runlog.state_sha(teacher.state_dict())
    result = graphs.precompute_c2_teacher(teacher, bank, items, device=rt.device, teacher_sha=teacher_sha,
                                          store_path=stage_dir / "teacher_logit_store.pt", log=log)
    logits_sha = graphs.save_teacher_logits(stage_dir / "teacher_logits.pt", result["logits"])
    report = {"pair": pair, "graph_sha256": graph_sha, "graph": stats, "teacher_sha256": teacher_sha,
              "teacher_logits_sha256": logits_sha, "unique_et_pairs": result["unique_et_pairs"],
              "evidence_anchors": result["evidence_anchors"], "elapsed_seconds": result["elapsed_seconds"]}
    write_json(stage_dir / "GRAPH_REPORT.json", report)
    runlog.post_run(stage_dir, status="COMPLETE", counters=stats, outputs={"graph.pkl": graph_sha, "teacher_logits.pt": logits_sha}, notes=report)
    return report


def cmd_train_c2(paths: Paths, *, seed: int, arm: str, device: str) -> dict:
    rt = Runtime(paths, seed=seed, device=device)
    kd, qt_only = _arm_flags(arm)
    pair = "qt" if qt_only else "native"
    stage = f"{arm}_C2"
    log = rt.stage_log(stage)
    graph_dir = rt.stage_dir(f"{pair.upper()}_C1_SELECTION_GRAPH")
    items = graphs.load_graph(graph_dir / "graph.pkl")
    teacher_logits = torch.load(graph_dir / "teacher_logits.pt", map_location="cpu", weights_only=False) if kd else None
    model, payload, c1_sha = _selected_student(rt, f"{arm}_C1")
    model = model.train()
    bank = rt.bank()
    gt = rt.dev_gt()

    def hook(tag: str, m) -> dict:
        sha = runlog.state_sha(m.state_dict())
        return _student_dev_eval(rt, m, bank, generator_id=f"{stage}:{tag}", model_sha=sha, qt_only=qt_only, gt=gt,
                                 out_dir=rt.stage_dir(stage) / "dev", tag=tag)

    parents = {f"{arm}_C1_selected": c1_sha, "graph": sha256_file(graph_dir / "graph.pkl"),
               "T_BOOT": rt.checkpoint_sha("T_BOOT") if kd else "not_used_by_SUP",
               "teacher_logits": sha256_file(graph_dir / "teacher_logits.pt") if kd else "not_used_by_SUP"}
    return student_mod.train_c2(model, bank, items, paths=paths, seed=seed, stage=stage, stage_dir=rt.stage_dir(stage),
                                root=rt.root, scope=pair, kd=kd, teacher_logits=teacher_logits, parents=parents,
                                device=rt.device, inputs={"graph": "graph.pkl", "labels": rt.labels.files},
                                snapshot_hook=hook, epochs=2, log=log)


# ------------------------------------------------- hard32 / own train pools ---


def cmd_mine_hard(paths: Paths, *, seed: int, device: str) -> dict:
    rt = Runtime(paths, seed=seed, device=device)
    stage = "HARD32"
    log = rt.stage_log(stage)
    stage_dir = rt.stage_dir(stage)
    selection = json.loads((rt.seed_dir() / "SELECTION_NATIVE_C2.json").read_text())
    if selection["status"] != "SELECTED":
        raise RuntimeError(f"native C2 selection is {selection['status']}")
    model, payload, sha = _selected_student(rt, "S_KD_NATIVE_C2")
    model_sha = runlog.state_sha(model.state_dict())
    generator_id = f"S_KD_NATIVE_C2:{selection['selected']}"
    runlog.pre_run(stage_dir, stage=stage, seed=seed, paths=paths, parents={"S_KD_NATIVE_C2_selected": sha},
                   inputs={"selection": selection["selected"]}, initial_state_sha=model_sha, optimizer_state="none",
                   config={"hard_n": 32, "source": "own Direct ANN top100, first 32 non-G legal targets"})
    pools = _own_train_pools(rt, model, rt.bank(), generator_id=generator_id, model_sha=model_sha, qt_only=False,
                             out_path=rt.stage_dir("OWN_TRAIN_POOLS") / "S_KD_NATIVE_train_pools.pkl", log=log)
    hard = graphs.mine_hard32(rt.labels, pools, generator_id=generator_id)
    path = stage_dir / "hard32.json.gz"
    from .evaluate import save_json_gz

    sha_hard = save_json_gz(path, hard)
    counts = [len(v["hard32"]) for v in hard.values()]
    report = {"queries": len(hard), "mean_hard": float(np.mean(counts)), "min_hard": min(counts),
              "queries_below_32": sum(1 for c in counts if c < 32), "generator_id": generator_id, "model_sha": model_sha,
              "hard32_sha256": sha_hard}
    write_json(stage_dir / "HARD32_REPORT.json", report)
    runlog.post_run(stage_dir, status="COMPLETE", counters={"queries": len(hard)}, outputs={"hard32.json.gz": sha_hard}, notes=report)
    return report


def cmd_train_qt_teacher(paths: Paths, *, seed: int, device: str) -> dict:
    rt = Runtime(paths, seed=seed, device=device)
    stage = "T_QT"
    log = rt.stage_log(stage)
    import gzip

    with gzip.open(rt.stage_dir("HARD32") / "hard32.json.gz", "rt") as handle:
        hard = json.load(handle)
    l0 = lists_mod.load_lists(rt.lists_dir() / "L0.pkl")
    lists = graphs.apply_hard32(l0, hard, root=rt.root, seed=seed)
    lists_sha = lists_mod.save_lists(rt.stage_dir(stage) / "L0_plus_hard32.pkl", lists)
    init_path = rt.stage_dir("T_INIT") / "state.pt"
    payload = runlog.load_checkpoint(init_path, expect_stage="T_INIT")
    model = teacher_mod.FreshPathTeacher(**teacher_mod.TEACHER_KWARGS).float()
    model.load_state_dict(payload["state_dict"])
    bank = rt.bank()
    hook = _dev_raw_rerank_hook(rt, model, bank, rt.raw_pools("dev"), rt.dev_gt())
    return teacher_mod.train_lists(
        model, bank, lists, paths=paths, seed=seed, stage=stage, stage_dir=rt.stage_dir(stage), root=rt.root,
        epochs=2, parents={"T_INIT": sha256_file(init_path), "HARD32": sha256_file(rt.stage_dir("HARD32") / "hard32.json.gz")},
        device=rt.device,
        inputs={"lists": "L0_plus_hard32.pkl", "lists_sha256": lists_sha, "initialization": "same_run_untrained_T_INIT",
                "extra_QT_candidates": sum(len(r.get("hard32_extra", ())) for r in lists.values())},
        epoch_hook=hook, log=log)


def cmd_train_qt_teacher_from_boot(paths: Paths, *, seed: int, device: str) -> dict:
    """Experimental branch: train the QT teacher from this run's T_BOOT state."""
    rt = Runtime(paths, seed=seed, device=device)
    stage = "T_QT_FROM_BOOT"
    log = rt.stage_log(stage)
    import gzip

    with gzip.open(rt.stage_dir("HARD32") / "hard32.json.gz", "rt") as handle:
        hard = json.load(handle)
    l0 = lists_mod.load_lists(rt.lists_dir() / "L0.pkl")
    lists = graphs.apply_hard32(l0, hard, root=rt.root, seed=seed)
    lists_sha = lists_mod.save_lists(rt.stage_dir(stage) / "L0_plus_hard32.pkl", lists)
    boot_path = rt.stage_dir("T_BOOT") / "checkpoint.pt"
    payload = runlog.load_checkpoint(boot_path, expect_stage="T_BOOT")
    model = teacher_mod.FreshPathTeacher(**teacher_mod.TEACHER_KWARGS).float()
    model.load_state_dict(payload["state_dict"])
    bank = rt.bank()
    hook = _dev_raw_rerank_hook(rt, model, bank, rt.raw_pools("dev"), rt.dev_gt())
    return teacher_mod.train_lists(
        model, bank, lists, paths=paths, seed=seed, stage=stage, stage_dir=rt.stage_dir(stage), root=rt.root,
        epochs=2, parents={"T_BOOT": sha256_file(boot_path), "HARD32": sha256_file(rt.stage_dir("HARD32") / "hard32.json.gz")},
        device=rt.device,
        inputs={"lists": "L0_plus_hard32.pkl", "lists_sha256": lists_sha,
                "initialization": "same_run_T_BOOT_experimental_branch",
                "extra_QT_candidates": sum(len(r.get("hard32_extra", ())) for r in lists.values())},
        epoch_hook=hook, log=log)


def cmd_train_path_teacher(paths: Paths, *, seed: int, device: str) -> dict:
    rt = Runtime(paths, seed=seed, device=device)
    stage = "T_PATH"
    log = rt.stage_log(stage)
    stage_dir = rt.stage_dir(stage)
    bank = rt.bank()
    own_path = rt.stage_dir("OWN_TRAIN_POOLS") / "S_KD_NATIVE_train_pools.pkl"
    own_pools = raw_mod.load_pools(own_path)
    raw_pools = rt.raw_pools("train")
    generator = next(iter(own_pools.values())).generator_id
    items = graphs.build_path_items(rt.labels, own_pools, raw_pools, own_generator=generator,
                                    reservoirs=raw_mod.load_reservoirs(paths, "train"),
                                    et_reservoir=raw_mod.load_et_reservoir(paths), root=rt.root, seed=seed)
    items_sha = graphs.save_graph(stage_dir / "items.pkl", items)
    t_qt, _ = teacher_mod.load_teacher(rt.stage_dir("T_QT") / "checkpoint.pt", rt.device, expect_stage="T_QT")
    t_qt_sha = runlog.state_sha(t_qt.state_dict())
    support = graphs.build_support_records(rt.labels, items, own_pools, raw_pools, t_qt, bank,
                                           device=rt.device, root=rt.root, seed=seed)
    with (stage_dir / "support_records.json").open("w", encoding="utf-8") as handle:
        json.dump(support, handle, ensure_ascii=False, sort_keys=True)
    anchors = graphs.precompute_anchor_logits(t_qt, bank, items, device=rt.device, teacher_sha=t_qt_sha,
                                              store_path=stage_dir / "anchor_logit_store.pt", log=log)
    model = teacher_mod.FreshPathTeacher(**teacher_mod.TEACHER_KWARGS).float()
    model.load_state_dict(t_qt.state_dict())
    del t_qt
    stats = {"queries": len(items), "active": sum(1 for it in items.values() if it["active"]),
             "mean_targets": float(np.mean([len(it["targets"]) for it in items.values()])),
             "mean_natural_slots": float(np.mean([sum(len(v) for v in it["natural"].values()) for it in items.values()])),
             "queries_with_support": sum(1 for it in items.values() if it.get("support_records")),
             "conditional_lists": sum(len(it["conditions"]) for it in items.values()),
             "mean_conditional_candidates": float(np.mean([len(c["candidates"]) for it in items.values() for c in it["conditions"]] or [0]))}
    write_json(stage_dir / "ITEMS_REPORT.json", {**stats, "items_sha256": items_sha, "own_generator": generator})
    return teacher_mod.train_path(
        model, bank, items, paths=paths, seed=seed, stage_dir=stage_dir, root=rt.root,
        parents={"T_QT": rt.checkpoint_sha("T_QT"), "T_QT_state": t_qt_sha, "own_train_pools": sha256_file(own_path),
                 "raw_train_pools": sha256_file(paths.work_dir / "raw" / "train" / "pools.pkl")},
        device=rt.device, inputs={"items_sha256": items_sha, **stats}, anchor_logits=anchors, log=log)


def cmd_train_qt_cont_teacher(paths: Paths, *, seed: int, device: str) -> dict:
    """Train the matched target-only continuation from this run's T_QT."""
    rt = Runtime(paths, seed=seed, device=device)
    stage_dir = rt.stage_dir("T_QT_CONT")
    own_path = rt.stage_dir("OWN_TRAIN_POOLS") / "S_KD_NATIVE_train_pools.pkl"
    own_pools = raw_mod.load_pools(own_path)
    raw_pools = rt.raw_pools("train")
    generator = next(iter(own_pools.values())).generator_id
    items = graphs.build_path_items(rt.labels, own_pools, raw_pools, own_generator=generator,
                                    reservoirs=raw_mod.load_reservoirs(paths, "train"),
                                    et_reservoir=raw_mod.load_et_reservoir(paths), root=rt.root, seed=seed)
    t_qt, _ = teacher_mod.load_teacher(rt.stage_dir("T_QT") / "checkpoint.pt", rt.device, expect_stage="T_QT")
    t_qt_sha = runlog.state_sha(t_qt.state_dict())
    anchors = graphs.precompute_anchor_logits(t_qt, rt.bank(), items, device=rt.device, teacher_sha=t_qt_sha,
                                              store_path=stage_dir / "anchor_logit_store.pt")
    model = teacher_mod.FreshPathTeacher(**teacher_mod.TEACHER_KWARGS).float()
    model.load_state_dict(t_qt.state_dict())
    items_sha = graphs.save_graph(stage_dir / "items.pkl", items)
    return teacher_mod.train_qt_cont(
        model, rt.bank(), items, paths=paths, seed=seed, stage_dir=stage_dir, root=rt.root,
        parents={"T_QT": rt.checkpoint_sha("T_QT"), "T_QT_state": t_qt_sha}, device=rt.device,
        inputs={"items_sha256": items_sha, "target_table_sha256": items_sha}, anchor_logits=anchors,
        log=rt.stage_log("T_QT_CONT"))


def cmd_freeze_models(paths: Paths, *, seed: int) -> dict:
    """Write the immutable per-seed model/config lock before any test read."""
    seed_dir = paths.work_dir / f"seed{seed}"
    required = {
        "T_INIT": seed_dir / "T_INIT" / "state.pt",
        "T_BOOT": seed_dir / "T_BOOT" / "checkpoint.pt",
        "S_SUP_NATIVE_C1": seed_dir / "S_SUP_NATIVE_C1" / "selected.pt",
        "S_KD_NATIVE_C1": seed_dir / "S_KD_NATIVE_C1" / "selected.pt",
        "S_QT_SUP_C1": seed_dir / "S_QT_SUP_C1" / "selected.pt",
        "S_QT_KD_C1": seed_dir / "S_QT_KD_C1" / "selected.pt",
        "T_QT": seed_dir / "T_QT" / "checkpoint.pt",
        "T_QT_FROM_BOOT": seed_dir / "T_QT_FROM_BOOT" / "checkpoint.pt",
        "T_QT_CONT": seed_dir / "T_QT_CONT" / "checkpoint.pt",
        "T_PATH": seed_dir / "T_PATH" / "checkpoint.pt",
        "S_SUP_NATIVE_C2": seed_dir / "S_SUP_NATIVE_C2" / "selected.pt",
        "S_KD_NATIVE_C2": seed_dir / "S_KD_NATIVE_C2" / "selected.pt",
        "S_QT_SUP_C2": seed_dir / "S_QT_SUP_C2" / "selected.pt",
        "S_QT_KD_C2": seed_dir / "S_QT_KD_C2" / "selected.pt",
        "EVAL_DEV": seed_dir / "EVAL_DEV" / "RESULTS.json",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"MODEL_LOCK blocked; missing {missing}")
    code_lock = runlog.code_lock_sha()
    lock = {"protocol_sha256": sha256_file(paths.protocol_path), "code_lock_sha256": code_lock, "seed": seed,
            "models": {name: sha256_file(path) for name, path in required.items() if name != "EVAL_DEV"},
            "config": {"protocol": str(paths.protocol_path), "code_lock_sha256": code_lock,
                        "selection_native": sha256_file(seed_dir / "SELECTION_NATIVE_C2.json"),
                        "selection_qt": sha256_file(seed_dir / "SELECTION_QT_C2.json")},
            "dev_results_sha256": sha256_file(required["EVAL_DEV"]), "test_read": False}
    write_json(seed_dir / "MODEL_LOCK.json", lock)
    return lock
