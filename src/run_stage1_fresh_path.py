#!/usr/bin/env python
"""Thin CLI for the FRESH-PATH v2.1 run (SPEC 14.2).

No flag accepts a historical checkpoint, teacher parent or old training list;
the only initialisers are the original dataset and the public backbone.
"""
from __future__ import annotations

import argparse
import os
import json
import sys
from pathlib import Path
from typing import Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fresh_path import (candidates, config, features, inputs, lineage, pca, pipeline,  # noqa: E402
                        prepare, train_student, train_teacher)
from fresh_path.teacher_cache import FrozenTeacherLogitCache  # noqa: E402


def _paths(args) -> config.Paths:
    return config.resolve_paths(args.dataset_root, args.backbone_dir, args.protocol, args.work_dir)


def _protocol(args) -> dict:
    return config.load_protocol(args.protocol)


def _teacher_cache(rt: pipeline.Runtime, model, seed: int, branch: str) -> FrozenTeacherLogitCache:
    """Open a run-local frozen-logit cache for one immutable Teacher branch."""
    path = rt.paths.work_dir / "teacher_logits" / f"seed{seed}_{branch}.pt"
    return FrozenTeacherLogitCache.for_model(model, rt.paths.protocol, path=path).load_existing()


def _adamw(parameters, lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(parameters, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)


def _resume_stage(model, *, stage_dir: Path, stage: str, seed: int, epochs: int,
                  protocol_path: Path, make_optimizer: Callable[[], torch.optim.Optimizer]):
    """Load one incomplete same-run stage at an epoch boundary, if present."""
    checkpoint = Path(stage_dir) / "checkpoint.pt"
    if not checkpoint.exists():
        return 1, None, None
    payload = lineage.load_checkpoint(checkpoint)
    if payload.get("stage") != stage or payload.get("model_seed") != seed:
        raise ValueError(f"{checkpoint}: stage/seed does not match requested {stage}/seed{seed}")
    if payload.get("protocol_hash") != lineage.protocol_hash(protocol_path):
        raise ValueError(f"{checkpoint}: protocol hash differs; same-run resume is forbidden")
    required = {"epoch", "state_dict", "optimizer_state_dict", "counters", "rng_state", "data_state"}
    missing = required - set(payload)
    if missing:
        raise ValueError(
            f"{checkpoint}: no strict-resume metadata ({', '.join(sorted(missing))}); "
            "this legacy checkpoint must not be resumed"
        )
    data_state = payload["data_state"]
    order = data_state.get("order", [])
    if data_state.get("order_cursor") != len(order):
        raise ValueError(f"{checkpoint}: only fully committed epoch boundaries are resumable")
    start_epoch = int(payload["epoch"]) + 1
    if start_epoch > epochs:
        print(json.dumps({"event": "skip_completed", "stage": stage, "epoch": payload["epoch"]}), flush=True)
        return None
    model.load_state_dict(payload["state_dict"])
    optimizer = make_optimizer()
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    lineage.restore_rng_state(payload["rng_state"])
    print(json.dumps({"event": "resume", "stage": stage, "from_epoch": payload["epoch"],
                      "next_epoch": start_epoch, "order_hash": data_state.get("order_hash")}), flush=True)
    return start_epoch, optimizer, dict(payload["counters"])


def cmd_prepare_data(args) -> int:
    paths = _paths(args)
    protocol = _protocol(args)
    out = paths.work_dir
    audit: list[dict] = []
    view = inputs.load_dataset(
        paths.dataset_root,
        audit=audit,
        content_key_path=Path(args.content_keys) if args.content_keys else None,
    )
    labels = inputs.build_train_labels(view, audit=audit)
    prepare.write_root_inputs(paths, protocol, out)
    prepare.write_schema_map(view, out)
    prepare.write_split_report(view, out)
    prepare.write_labels(labels, out / "labels")
    probe = inputs.label_isolation_probe(view, labels, out / "labels")
    inputs.write_jsonl_gz(out / "READ_AUDIT.jsonl.gz", audit)
    with (out / "READ_AUDIT.jsonl").open("w") as fh:
        for row in audit:
            fh.write(json.dumps(row) + "\n")
    (out / "label_isolation_probe.json").write_text(json.dumps(probe, indent=1))
    print(json.dumps({"event": "prepare_data", "stats": labels.stats,
                      "conflicts": len(labels.conflicts), "label_stable": probe["stable"]}, indent=1))
    return 0


def cmd_prepare_raw(args) -> int:
    paths = _paths(args)
    labels = prepare.load_labels(paths.work_dir / "labels")
    store = candidates.load_z(paths.work_dir / "features" / "z")
    query_ids = sorted(labels.queries, key=lambda x: x.encode("utf-8"))
    raw = candidates.build_raw(store, labels, query_ids=query_ids, device=args.device, progress=args.progress)
    candidates.save_pickle(paths.work_dir / "raw" / "raw_train.pkl", raw)
    print(json.dumps({"event": "prepare_raw", "queries": len(query_ids),
                      "targets": len(raw["targets"]), "text": len(raw["text_assets"]),
                      "image": len(raw["image_assets"])}))
    return 0


def cmd_prepare_features(args) -> int:
    paths = _paths(args)
    root = paths.work_dir / "features"
    sources = [root / "content_cache" / "chunks"] + sorted((root / "gen").glob("*/chunks"))
    index = features.merge_chunk_dirs(sources, root / "content" / "chunks")
    coverage = json.loads((root / "content" / "coverage.json").read_text())
    print(json.dumps({"event": "prepare_features", "coverage": coverage, "sources": [str(s) for s in sources]}))
    return 0


def cmd_prepare_pca(args) -> int:
    paths = _paths(args)
    labels = prepare.load_labels(paths.work_dir / "labels")
    z_dir = paths.work_dir / "features" / "z"
    index = json.loads((z_dir / "z_index.json").read_text())
    content_keys = None
    if args.content_keys and Path(args.content_keys).exists():
        content_keys = {}
        for row in inputs.iter_jsonl(Path(args.content_keys)):
            content_keys[str(row["object_id"])] = str(row["content_key"])
    fit_ids = pca.fit_scope(labels, content_keys)
    stats = pca.compute_pca(z_dir / "z.f32.npy", index["ids"], fit_ids,
                            paths.work_dir / "pca" / "basis.pt", components=args.components)
    (paths.work_dir / "pca" / "PCA_REPORT.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({"event": "prepare_pca", **stats}))
    return 0


def cmd_post_edge(args) -> int:
    paths = _paths(args)
    rt = pipeline.Runtime(paths, _protocol(args))
    rt.bank.attach_device(args.device)
    pipeline.run_post_edge(rt, seed=args.seed, device=args.device)
    return 0


def cmd_train_teacher(args) -> int:
    paths = _paths(args)
    protocol = _protocol(args)
    rt = pipeline.Runtime(paths, protocol)
    rt.bank.attach_device(args.device)
    seed, stage = args.seed, args.stage
    teacher = protocol["teacher"]
    if stage == "edge":
        path = paths.work_dir / "raw" / "edge_initial.pkl"
        payload = candidates.load_pickle(path) if path.exists() else pipeline.build_initial_edges(rt)
        model = rt.make_teacher(seed).to(args.device)
        resume = _resume_stage(
            model, stage_dir=rt.stage_dir(seed, "T_EDGE"), stage="T_EDGE", seed=seed,
            epochs=teacher["edge_epochs"], protocol_path=paths.protocol,
            make_optimizer=lambda: _adamw(model.parameters(), teacher["lr"]),
        )
        if resume is None:
            return 0
        start_epoch, optimizer, counters = resume
        train_teacher.train_edge(model, rt.bank, rt.labels, payload["edge"], payload["positives"],
                                 seed=seed, epochs=teacher["edge_epochs"], lr=teacher["lr"],
                                 logical_batch=teacher["logical_batch_queries"], device=args.device,
                                 out_dir=rt.stage_dir(seed, "T_EDGE"), protocol_path=paths.protocol,
                                 optimizer=optimizer, start_epoch=start_epoch, counters=counters,
                                 batch=args.path_batch)
    else:
        post = rt.post_edge
        model = rt.make_teacher(seed)
        parent = lineage.load_checkpoint(rt.stage_dir(seed, "T_EDGE") / "checkpoint.pt")
        model.load_state_dict(parent["state_dict"])
        model = model.to(args.device)
        if stage == "path":
            resume = _resume_stage(
                model, stage_dir=rt.stage_dir(seed, "T_PATH"), stage="T_PATH", seed=seed,
                epochs=teacher["path_epochs"], protocol_path=paths.protocol,
                make_optimizer=lambda: _adamw(model.parameters(), teacher["lr"]),
            )
            if resume is None:
                return 0
            start_epoch, optimizer, counters = resume
            train_teacher.train_path(model, rt.bank, rt.labels, post["edge"], post["positives"], post["graph"],
                                     rt.raw, seed=seed, epochs=teacher["path_epochs"], lr=teacher["lr"],
                                     logical_batch=teacher["logical_batch_queries"], device=args.device,
                                     out_dir=rt.stage_dir(seed, "T_PATH"), protocol_path=paths.protocol,
                                     parent_dirs=[rt.stage_dir(seed, "T_EDGE")], batch=args.path_batch,
                                     optimizer=optimizer, start_epoch=start_epoch, counters=counters,
                                     conditional_registry=post.get("conditional_registry"))
        else:
            resume = _resume_stage(
                model, stage_dir=rt.stage_dir(seed, "T_QT"), stage="T_QT", seed=seed,
                epochs=teacher["qt_control"]["epochs"], protocol_path=paths.protocol,
                make_optimizer=lambda: _adamw(model.parameters(), teacher["lr"]),
            )
            if resume is None:
                return 0
            start_epoch, optimizer, counters = resume
            train_teacher.train_qt(model, rt.bank, rt.labels, post["edge"], post["positives"], post["graph"],
                                   seed=seed, epochs=teacher["qt_control"]["epochs"], lr=teacher["lr"],
                                   logical_batch=teacher["logical_batch_queries"], device=args.device,
                                   out_dir=rt.stage_dir(seed, "T_QT"), protocol_path=paths.protocol,
                                   parent_dirs=[rt.stage_dir(seed, "T_EDGE")], batch=args.path_batch,
                                   optimizer=optimizer, start_epoch=start_epoch, counters=counters)
    return 0


STUDENT_C1 = {"S_SUP_C1": False, "S_KD_C1": True}
CONDITIONAL_C2 = {"S_SUP_QE_C2": (False, False), "S_KD_QE_C2": (True, False), "S_KD_EONLY_C2": (True, True)}
QT_STUDENTS = {"S_QT_SUP_C1", "S_QT_KD_C1", "S_QT_SUP_C2", "S_QT_KD_C2"}


def cmd_train_student(args) -> int:
    paths = _paths(args)
    protocol = _protocol(args)
    rt = pipeline.Runtime(paths, protocol)
    rt.bank.attach_device(args.device)
    seed, stage = args.seed, args.stage
    student = protocol["student"]
    # post-edge artifacts are only needed by stages that consume the common
    # target graph; the QT-SUP branch must be runnable before T_EDGE finishes.
    post = rt.post_edge if stage in set(STUDENT_C1) | set(CONDITIONAL_C2) | {"S_KD_NATIVE_C2"} else None
    if stage in STUDENT_C1:
        teacher = None
        cache = None
        if STUDENT_C1[stage]:
            teacher = rt.load_teacher(seed, "T_PATH", args.device)
            teacher.requires_grad_(False)
            cache = _teacher_cache(rt, teacher, seed, f"T_PATH_{stage}")
        else:
            # Historically C1 construction followed ``make_teacher(seed)`` even
            # though SUP never consumed its logits.  Preserve that model-RNG
            # setup while removing the frozen checkpoint dependency, so SUP can
            # start as soon as the refreshed edge lists are sealed.
            rt.make_teacher(seed)
        model = rt.make_student(seed).to(args.device)
        resume = _resume_stage(
            model, stage_dir=rt.stage_dir(seed, stage), stage=stage, seed=seed,
            epochs=student["C1_epochs"], protocol_path=paths.protocol,
            make_optimizer=lambda: _adamw(model.parameters(), student["C1_lr"]),
        )
        if resume is None:
            return 0
        start_epoch, optimizer, counters = resume
        train_student.train_c1(model, teacher, rt.bank, rt.labels, post["edge"], post["positives"],
                               seed=seed, epochs=student["C1_epochs"], lr=student["C1_lr"],
                               logical_batch=student["C1_logical_batch_queries"], device=args.device,
                               out_dir=rt.stage_dir(seed, stage), protocol_path=paths.protocol,
                               kd=STUDENT_C1[stage], batch=args.path_batch, teacher_cache=cache,
                               optimizer=optimizer, start_epoch=start_epoch, counters=counters)
    elif stage in CONDITIONAL_C2:
        kd, eonly = CONDITIONAL_C2[stage]
        c1 = "S_KD_C1" if kd else "S_SUP_C1"
        model = rt.load_student(seed, c1, args.device)
        teacher = None
        cache = None
        if kd:
            teacher = rt.load_teacher(seed, "T_PATH", args.device)
            teacher.requires_grad_(False)
            cache = _teacher_cache(rt, teacher, seed, f"T_PATH_{stage}")
        resume = _resume_stage(
            model, stage_dir=rt.stage_dir(seed, stage), stage=stage, seed=seed,
            epochs=student["C2_epochs"], protocol_path=paths.protocol,
            make_optimizer=lambda: _adamw(model.adapter.parameters(), student["C2_lr"]),
        )
        if resume is None:
            return 0
        start_epoch, optimizer, counters = resume
        train_student.train_conditional_c2(model, teacher, rt.bank, rt.labels, rt.raw, post["graph"],
                                           seed=seed, epochs=student["C2_epochs"], lr=student["C2_lr"],
                                           logical_batch=student["C2_logical_batch_queries"], device=args.device,
                                           out_dir=rt.stage_dir(seed, stage), protocol_path=paths.protocol,
                                           kd=kd, eonly=eonly, parent_dirs=[rt.stage_dir(seed, c1)],
                                           full_chunk=student["full_target_chunk"], batch=args.path_batch,
                                           teacher_cache=cache, optimizer=optimizer, start_epoch=start_epoch,
                                           counters=counters, conditional_registry=post.get("conditional_registry"))
    elif stage == "S_KD_NATIVE_C2":
        model = rt.load_student(seed, "S_KD_C1", args.device, adapter=False)
        teacher = rt.load_teacher(seed, "T_QT", args.device)
        teacher.requires_grad_(False)
        cache = _teacher_cache(rt, teacher, seed, "T_QT_S_KD_NATIVE_C2")
        native = protocol["controls"]["native"]
        resume = _resume_stage(
            model, stage_dir=rt.stage_dir(seed, stage), stage=stage, seed=seed,
            epochs=native["epochs"], protocol_path=paths.protocol,
            make_optimizer=lambda: _adamw(model.parameters(), native["lr"]),
        )
        if resume is None:
            return 0
        start_epoch, optimizer, counters = resume
        train_student.train_native(model, teacher, rt.bank, rt.labels, rt.raw, post["graph"],
                                   seed=seed, epochs=native["epochs"], lr=native["lr"],
                                   logical_batch=native["logical_batch_queries"], device=args.device,
                                   out_dir=rt.stage_dir(seed, stage), protocol_path=paths.protocol,
                                   parent_dirs=[rt.stage_dir(seed, "S_KD_C1")], batch=args.path_batch,
                                   teacher_cache=cache, optimizer=optimizer, start_epoch=start_epoch,
                                   counters=counters)
    elif stage in QT_STUDENTS:
        # QT-SUP needs no Teacher at all, so this branch can run while T_EDGE is
        # still training (SPEC 13.5 item 1).  Only the QT-KD stages distil T_QT.
        teacher = None
        if "KD" in stage:
            teacher = rt.load_teacher(seed, "T_QT", args.device)
            teacher.requires_grad_(False)
            cache = _teacher_cache(rt, teacher, seed, f"T_QT_{stage}")
        else:
            cache = None
        if stage.endswith("_C1"):
            model = rt.make_qt_student(seed).to(args.device)
            parent_dirs = []
        else:
            c1 = "S_QT_SUP_C1" if "SUP" in stage else "S_QT_KD_C1"
            model = rt.make_qt_student(seed)
            payload = lineage.load_checkpoint(rt.stage_dir(seed, c1) / "checkpoint.pt")
            model.load_state_dict(payload["state_dict"])
            model = model.to(args.device)
            parent_dirs = [rt.stage_dir(seed, c1)]
        direct = protocol["controls"]["direct_only"]
        epochs = direct["C1_epochs"] if stage.endswith("_C1") else direct["C2_epochs"]
        resume = _resume_stage(
            model, stage_dir=rt.stage_dir(seed, stage), stage=stage, seed=seed,
            epochs=epochs, protocol_path=paths.protocol,
            make_optimizer=lambda: _adamw(model.parameters(), direct["lr"]),
        )
        if resume is None:
            return 0
        start_epoch, optimizer, counters = resume
        train_student.train_qt_student(model, teacher, rt.bank, rt.labels, rt.raw, seed=seed, stage=stage,
                                       epochs=epochs,
                                       lr=direct["lr"],
                                       logical_batch=direct["logical_batch_queries"], device=args.device,
                                       out_dir=rt.stage_dir(seed, stage), protocol_path=paths.protocol,
                                       parent_dirs=parent_dirs, batch=args.path_batch,
                                       full_chunk=student["full_target_chunk"], teacher_cache=cache,
                                       optimizer=optimizer, start_epoch=start_epoch, counters=counters)
    else:
        raise ValueError(f"unknown student stage {stage}")
    return 0


def cmd_train_students(args) -> int:
    for stage in ("S_SUP_C1", "S_KD_C1", "S_SUP_QE_C2", "S_KD_QE_C2", "S_KD_EONLY_C2",
                  "S_KD_NATIVE_C2", "S_QT_SUP_C1", "S_QT_KD_C1", "S_QT_SUP_C2", "S_QT_KD_C2"):
        rc = cmd_train_student(argparse.Namespace(**{**vars(args), "stage": stage}))
        if rc:
            return rc
    return 0


def _subset_labels(labels, query_ids):
    out = inputs.TrainLabels()
    out.queries = {q: labels.queries[q] for q in query_ids}
    out.epos = labels.epos
    out.legal = labels.legal
    out.stats = labels.stats
    return out


def cmd_smoke(args) -> int:
    """P1 real end-to-end smoke: raw objects -> paths -> forward/backward -> index -> metric.

    Uses temporary models with at most two logical updates per stage; nothing it
    produces is a formal stage artifact (SPEC 14.3).
    """
    import tempfile

    paths = _paths(args)
    protocol = _protocol(args)
    rt = pipeline.Runtime(paths, protocol)
    rt.bank.attach_device(args.device)
    qids = sorted(rt.labels.queries, key=lambda x: x.encode("utf-8"))[: args.queries]
    labels = _subset_labels(rt.labels, qids)
    raw = rt.raw
    initial = candidates.load_pickle(paths.work_dir / "raw" / "edge_initial.pkl") \
        if (paths.work_dir / "raw" / "edge_initial.pkl").exists() else pipeline.build_initial_edges(rt)
    edge = {q: initial["edge"][q] for q in qids}
    positives = {q: initial["positives"][q] for q in qids}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        teacher = rt.make_teacher(args.seed).to(args.device)
        train_teacher.train_edge(teacher, rt.bank, labels, edge, positives, seed=args.seed, epochs=1, lr=5e-5,
                                 logical_batch=2, device=args.device, out_dir=tmp / "edge",
                                 protocol_path=paths.protocol)
        refresh = train_teacher.refreshed_hard(teacher, rt.bank, labels, raw, device=args.device)
        graph = train_teacher.build_train_graph(teacher, rt.bank, labels, raw, device=args.device)
        edge_ref = candidates.build_edge_lists(labels, raw, hard_scores=refresh)
        train_teacher.train_path(teacher, rt.bank, labels, edge_ref, positives, graph, raw, seed=args.seed,
                                 epochs=1, lr=5e-5, logical_batch=2, device=args.device, out_dir=tmp / "path",
                                 protocol_path=paths.protocol, parent_dirs=[], batch=args.path_batch)
        path_model = teacher
        qt_model = rt.make_teacher(args.seed).to(args.device)
        qt_model.load_state_dict(path_model.state_dict())
        train_teacher.train_qt(qt_model, rt.bank, labels, edge_ref, positives, graph, seed=args.seed, epochs=1,
                               lr=5e-5, logical_batch=2, device=args.device, out_dir=tmp / "qt",
                               protocol_path=paths.protocol, parent_dirs=[], batch=args.path_batch)
        student = rt.make_student(args.seed).to(args.device)
        train_student.train_c1(student, path_model, rt.bank, labels, edge_ref, positives, seed=args.seed,
                               epochs=1, lr=1e-4, logical_batch=2, device=args.device, out_dir=tmp / "supc1",
                               protocol_path=paths.protocol, kd=False)
        train_student.train_conditional_c2(student, path_model, rt.bank, labels, raw, graph, seed=args.seed,
                                           epochs=1, lr=1e-4, logical_batch=2, device=args.device,
                                           out_dir=tmp / "supqe", protocol_path=paths.protocol, kd=False,
                                           eonly=False, parent_dirs=[], batch=args.path_batch)
        native = rt.make_student(args.seed, adapter=False).to(args.device)
        train_student.train_native(native, qt_model, rt.bank, labels, raw, graph, seed=args.seed, epochs=1,
                                   lr=1e-4, logical_batch=2, device=args.device, out_dir=tmp / "native",
                                   protocol_path=paths.protocol, parent_dirs=[], batch=args.path_batch)
        qt_student = rt.make_qt_student(args.seed).to(args.device)
        train_student.train_qt_student(qt_student, qt_model, rt.bank, labels, raw, seed=args.seed,
                                       stage="QT_SUP_C1", epochs=1, lr=1e-4, logical_batch=2,
                                       device=args.device, out_dir=tmp / "qtsup", protocol_path=paths.protocol,
                                       parent_dirs=[])

        # real index + retrieval + rerank + metric on the train queries used above
        from fresh_path import evaluate, retrieval

        retriever = retrieval.OwnRetriever(student, rt.bank, rt.labels.legal, raw["text_assets"],
                                           raw["image_assets"], device=args.device)
        rankings = {}
        coverages = []
        for qid in qids:
            two = retriever.two_way(qid)
            coverages.append(len(set(two["C100"]) & set(rt.labels.queries[qid]["G"])))
            scores = evaluate.rerank_path(path_model, rt.bank, qid, two["C100"], two["paths"], args.device)
            rankings[qid] = sorted(scores, key=lambda t: (-scores[t], t.encode("utf-8")))
        gold = {q: rt.labels.queries[q]["G"] for q in qids}
        report = {"queries": len(qids),
                  "R10": evaluate.macro_recall(gold, rankings, 10),
                  "R20": evaluate.macro_recall(gold, rankings, 20),
                  "R50": evaluate.macro_recall(gold, rankings, 50),
                  "mean_C100_gold_coverage": float(sum(coverages) / len(qids))}
        (paths.work_dir / "tests" / "receipts").mkdir(parents=True, exist_ok=True)
        (paths.work_dir / "tests" / "receipts" / "SMOKE_RECEIPT.json").write_text(json.dumps(report, indent=1))
        print(json.dumps({"event": "smoke", **report}))
    return 0


def cmd_evaluate_dev(args) -> int:
    """Own retrieval + fixed-teacher-pool tables on all original dev queries (SPEC 12.1)."""
    from fresh_path import evaluate, retrieval

    paths = _paths(args)
    protocol = _protocol(args)
    rt = pipeline.Runtime(paths, protocol)
    rt.bank.attach_device(args.device)
    dev_queries = sorted([q for q, s in rt.view_query_split().items() if s == "dev"], key=lambda x: x.encode("utf-8"))
    gold = inputs.load_split_gt(paths.dataset_root, "dev")
    kinds = rt.dev_query_kinds(paths.dataset_root, dev_queries)
    raw_dev_path = paths.work_dir / "raw" / "raw_dev.pkl"
    if raw_dev_path.exists():
        raw_dev = candidates.load_pickle(raw_dev_path)
    else:
        raw_dev = candidates.build_raw(rt.z, rt.labels, query_ids=dev_queries, device=args.device,
                                       progress=250, et_anchors=[])
        candidates.save_pickle(raw_dev_path, raw_dev)

    out_dir = paths.work_dir / "seed13" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    def _try_teacher(stage):
        try:
            return rt.load_teacher(args.seed, stage, args.device)
        except FileNotFoundError:
            print(json.dumps({"event": "teacher_unavailable", "stage": stage,
                              "effect": "rows depending on it are skipped"}))
            return None

    teacher_qt = _try_teacher("T_QT")
    teacher_path = _try_teacher("T_PATH")
    teacher_edge = _try_teacher("T_EDGE")

    rows: dict[str, dict[str, object]] = {}
    rankings: dict[str, dict[str, list[str]]] = {}

    # Raw rows share the raw admission candidate pool.
    raw_direct = {q: list(raw_dev["admission"][q]["direct"]) for q in dev_queries}
    rankings["Raw_Direct"] = raw_direct
    raw_two = {q: list(raw_dev["admission"][q]["C100"]) for q in dev_queries}
    rankings["Raw_two_way"] = raw_two
    if teacher_qt is not None:
        rankings["Raw_plus_T_QT"] = {q: _rank_from_scores(teacher_qt, rt, q, raw_two[q], args.device, paths=None)
                                     for q in dev_queries}
    if teacher_path is not None:
        rankings["Raw_plus_T_PATH_Real"] = {
            q: _rank_from_paths(teacher_path, rt, raw_dev, q, raw_two[q], args.device) for q in dev_queries}

    endpoints = ["S_KD_QE_C2", "S_KD_NATIVE_C2", "S_SUP_QE_C2", "S_KD_EONLY_C2"]
    for stage in endpoints:
        ckpt = rt.stage_dir(args.seed, stage) / "checkpoint.pt"
        if not ckpt.exists():
            continue
        name = stage.replace("S_", "").replace("_C2", "")
        student = (rt.load_student(args.seed, stage, args.device, adapter=False)
                   if stage == "S_KD_NATIVE_C2" else rt.load_student(args.seed, stage, args.device))
        retriever = retrieval.OwnRetriever(student, rt.bank, rt.labels.legal, raw_dev["text_assets"],
                                           raw_dev["image_assets"], device=args.device)
        two = {q: retriever.two_way(q) for q in dev_queries}
        rankings[f"{name}_own_C100"] = {q: two[q]["C100"] for q in dev_queries}
        rankings[f"{name}_own_D100"] = {q: [t for t, _ in two[q]["direct"]] for q in dev_queries}
        if teacher_qt is not None:
            rankings[f"{name}_plus_T_QT"] = {q: _rank_from_scores(teacher_qt, rt, q, two[q]["C100"], args.device,
                                                                  paths=None) for q in dev_queries}
        if teacher_path is not None:
            rankings[f"{name}_plus_T_PATH"] = {q: _rank_from_paths(teacher_path, rt, raw_dev, q, two[q]["C100"],
                                                                   args.device) for q in dev_queries}
        if stage in ("S_KD_QE_C2", "S_KD_NATIVE_C2") and teacher_qt is not None:
            direct_rank = {q: [t for t, _ in retriever.direct(q)] for q in dev_queries}
            rankings[f"{name}_D100_plus_T_QT"] = {
                q: _rank_from_scores(teacher_qt, rt, q, direct_rank[q], args.device, paths=None) for q in dev_queries}

    for name in ("QT_SUP", "QT_KD"):
        ckpt = rt.stage_dir(args.seed, f"S_{name}_C2") / "checkpoint.pt"
        if not ckpt.exists():
            continue
        model = rt.make_qt_student(args.seed)
        payload = lineage.load_checkpoint(ckpt)
        model.load_state_dict(payload["state_dict"])
        model = model.to(args.device)
        direct = retrieval.direct_only(model, rt.bank, rt.labels.legal, dev_queries, device=args.device)
        rankings[f"{name}_D100"] = {q: [t for t, _ in direct[q]] for q in dev_queries}
        if teacher_qt is not None:
            rankings[f"{name}_plus_T_QT"] = {q: _rank_from_scores(teacher_qt, rt, q, rankings[f"{name}_D100"][q],
                                                                  args.device, paths=None) for q in dev_queries}

    for name, ranking in rankings.items():
        rows[name] = evaluate.grouped_metrics(gold, ranking, kinds)
    (out_dir / "DEV_METRICS.json").write_text(json.dumps(rows, indent=1))
    import gzip

    with gzip.open(out_dir / "DEV_RANKINGS.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(rankings, fh)
    probe_pairs = rt.dev_probe_pairs(paths.dataset_root, dev_queries, raw_dev)
    if probe_pairs:
        qe = rt.load_student(args.seed, "S_KD_QE_C2", args.device)
        eonly = rt.load_student(args.seed, "S_KD_EONLY_C2", args.device)
        (out_dir / "PROBE.json").write_text(json.dumps({
            "KD_QE": evaluate.probe_et(qe, rt.bank, rt.labels.legal, probe_pairs, args.device),
            "KD_EONLY": evaluate.probe_et(eonly, rt.bank, rt.labels.legal, probe_pairs, args.device),
        }, indent=1))
    print(json.dumps({"event": "evaluate_dev", "systems": len(rows), "dev_queries": len(dev_queries)}))
    return 0


def _rank_from_scores(teacher, rt, qid, candidates, device, *, paths):
    from fresh_path.evaluate import rerank_zero_hop

    scores = rerank_zero_hop(teacher, rt.bank, qid, candidates, device)
    return sorted(scores, key=lambda t: (-scores[t], t.encode("utf-8")))


def _rank_from_paths(teacher, rt, raw_dev, qid, candidates, device):
    from fresh_path.evaluate import rerank_path

    paths = raw_dev["admission"][qid]["paths"]
    scores = rerank_path(teacher, rt.bank, qid, candidates, paths, device)
    return sorted(scores, key=lambda t: (-scores[t], t.encode("utf-8")))


# This is the executable subset of EXECUTION_DAG.json.  The initial P0/P1,
# raw-list and PCA inputs are validated by the invoked stage commands; these
# edges describe only artifacts produced by this queue.
RUN_DEPENDENCIES = {
    "T_EDGE": (),
    "post_edge": ("T_EDGE",),
    "T_PATH": ("post_edge",),
    "T_QT": ("post_edge",),
    "S_SUP_C1": ("post_edge",),
    "S_KD_C1": ("post_edge", "T_PATH"),
    "S_SUP_QE_C2": ("post_edge", "S_SUP_C1"),
    "S_KD_QE_C2": ("post_edge", "S_KD_C1", "T_PATH"),
    "S_KD_EONLY_C2": ("post_edge", "S_KD_C1", "T_PATH"),
    "S_KD_NATIVE_C2": ("post_edge", "S_KD_C1", "T_QT"),
    "S_QT_SUP_C1": (),
    "S_QT_KD_C1": ("T_QT",),
    "S_QT_SUP_C2": ("S_QT_SUP_C1",),
    "S_QT_KD_C2": ("S_QT_KD_C1", "T_QT"),
}
RUN_KIND = {
    "T_EDGE": "teacher", "T_PATH": "teacher", "T_QT": "teacher",
    "post_edge": "post",
    **{stage: "student" for stage in STUDENT_C1},
    **{stage: "student" for stage in CONDITIONAL_C2},
    "S_KD_NATIVE_C2": "student",
    **{stage: "student" for stage in QT_STUDENTS},
}


def _expected_epochs(protocol: dict) -> dict:
    t, s = protocol["teacher"], protocol["student"]
    n, d = protocol["controls"]["native"], protocol["controls"]["direct_only"]
    return {
        "T_EDGE": t["edge_epochs"], "T_PATH": t["path_epochs"], "T_QT": t["qt_control"]["epochs"],
        "S_SUP_C1": s["C1_epochs"], "S_KD_C1": s["C1_epochs"],
        "S_SUP_QE_C2": s["C2_epochs"], "S_KD_QE_C2": s["C2_epochs"], "S_KD_EONLY_C2": s["C2_epochs"],
        "S_KD_NATIVE_C2": n["epochs"],
        "S_QT_SUP_C1": d["C1_epochs"], "S_QT_KD_C1": d["C1_epochs"],
        "S_QT_SUP_C2": d["C2_epochs"], "S_QT_KD_C2": d["C2_epochs"],
    }


def _stage_completed(paths, seed: int, stage: str, expected) -> bool:
    """A same-run stage is reusable only if its receipt reached the fixed endpoint."""
    if not expected:
        return False
    d = paths.stage_dir(seed, stage)
    receipt, checkpoint = d / "receipt.json", d / "checkpoint.pt"
    if not (receipt.exists() and checkpoint.exists()):
        return False
    try:
        payload = json.loads(receipt.read_text())
        checkpoint_payload = lineage.load_checkpoint(checkpoint)
    except Exception:
        return False
    required = {"rng_state", "data_state", "optimizer_state_dict"}
    return (
        int(payload.get("epoch", -1)) == int(expected)
        and checkpoint_payload.get("stage") == stage
        and checkpoint_payload.get("model_seed") == seed
        and int(checkpoint_payload.get("epoch", -1)) == int(expected)
        and required <= set(checkpoint_payload)
    )


def _post_edge_completed(paths) -> bool:
    return (paths.work_dir / "raw" / "post_edge.pkl").is_file()


def _stage_command(base: list[str], stage: str, seed: int, path_batch: int) -> list[str]:
    kind = RUN_KIND[stage]
    if kind == "teacher":
        return base + ["train-teacher", "--seed", str(seed), "--stage",
                       {"T_EDGE": "edge", "T_PATH": "path", "T_QT": "qt"}[stage],
                       "--device", "cuda:0", "--path-batch", str(path_batch)]
    if kind == "post":
        return base + ["post-edge", "--seed", str(seed), "--device", "cuda:0"]
    return base + ["train-student", "--seed", str(seed), "--stage", stage,
                   "--device", "cuda:0", "--path-batch", str(path_batch)]


def cmd_run(args) -> int:
    """Dependency-ready, one-process-per-GPU queue (SPEC 13.5)."""
    import subprocess
    import time

    paths = _paths(args)
    protocol = _protocol(args)
    base = [sys.executable, "-u", str(Path(__file__).resolve()), "--dataset-root", str(args.dataset_root),
            "--backbone-dir", str(args.backbone_dir), "--protocol", str(args.protocol),
            "--work-dir", str(args.work_dir)]
    logs = paths.work_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    expected = _expected_epochs(protocol)
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices or len(devices) > 2 or len(set(devices)) != len(devices):
        raise ValueError("--devices must name one or two distinct physical GPU IDs")

    completed = {
        stage for stage in RUN_DEPENDENCIES
        if stage != "post_edge" and _stage_completed(paths, args.seed, stage, expected.get(stage))
    }
    if _post_edge_completed(paths) and "T_EDGE" in completed:
        completed.add("post_edge")
    for stage in sorted(completed):
        print(json.dumps({"event": "skip_completed", "stage": stage}), flush=True)

    pending = set(RUN_DEPENDENCIES) - completed
    running: dict[str, tuple[str, object, object, list[str], dict]] = {}
    while pending or running:
        ready = sorted(
            (stage for stage in pending if set(RUN_DEPENDENCIES[stage]) <= completed),
            key=lambda stage: list(RUN_DEPENDENCIES).index(stage),
        )
        for gpu in devices:
            if not ready or gpu in running:
                continue
            stage = ready.pop(0)
            pending.remove(stage)
            cmd = _stage_command(base, stage, args.seed, args.path_batch)
            # expandable_segments curbs allocator fragmentation from variable
            # path lengths; it does not alter inputs, operations or precision.
            env = dict(**os.environ, CUDA_VISIBLE_DEVICES=gpu,
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            log = (logs / f"seed{args.seed}_{stage}.log").open("w")
            print(json.dumps({"event": "launch", "stage": stage, "gpu": gpu}), flush=True)
            running[gpu] = (stage, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log, cmd, env)

        if not running:
            waiting = {stage: RUN_DEPENDENCIES[stage] for stage in sorted(pending)}
            raise RuntimeError(f"no runnable stage; unresolved dependencies: {waiting}")

        finished = [(gpu, job) for gpu, job in running.items() if job[1].poll() is not None]
        if not finished:
            time.sleep(0.2)
            continue
        for gpu, (stage, proc, log, cmd, env) in finished:
            del running[gpu]
            rc = proc.returncode
            log.close()
            print(json.dumps({"event": "finish", "stage": stage, "returncode": rc, "attempt": 1}), flush=True)
            if rc != 0:
                # Epoch-boundary checkpoints include RNG and optimizer state, so
                # this retry replays only uncommitted work under the same setup.
                print(json.dumps({"event": "retry", "stage": stage, "first_returncode": rc}), flush=True)
                with (logs / f"seed{args.seed}_{stage}.retry.log").open("w") as retry_log:
                    again = subprocess.run(cmd, env=env, stdout=retry_log, stderr=subprocess.STDOUT)
                print(json.dumps({"event": "finish", "stage": stage, "returncode": again.returncode,
                                  "attempt": 2}), flush=True)
                if again.returncode != 0:
                    return again.returncode
            completed.add(stage)
    return 0




def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="run_stage1_fresh_path")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--backbone-dir", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--work-dir", required=True)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare-data")
    p.add_argument("--content-keys", default=None)
    p.set_defaults(func=cmd_prepare_data)

    p = sub.add_parser("prepare-raw")
    p.add_argument("--device", default="cpu")
    p.add_argument("--progress", type=int, default=500)
    p.set_defaults(func=cmd_prepare_raw)

    p = sub.add_parser("prepare-features")
    p.set_defaults(func=cmd_prepare_features)

    p = sub.add_parser("prepare-pca")
    p.add_argument("--components", type=int, default=1024)
    p.add_argument("--content-keys", default=None)
    p.set_defaults(func=cmd_prepare_pca)

    p = sub.add_parser("post-edge")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--device", default="cuda:0")
    p.set_defaults(func=cmd_post_edge)

    p = sub.add_parser("train-teacher")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--stage", choices=("edge", "path", "qt"), required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--path-batch", type=int, default=1024)
    p.set_defaults(func=cmd_train_teacher)

    p = sub.add_parser("train-student")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--stage", required=True,
                   choices=tuple(STUDENT_C1) + tuple(CONDITIONAL_C2) + ("S_KD_NATIVE_C2",) + tuple(QT_STUDENTS))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--path-batch", type=int, default=1024)
    p.set_defaults(func=cmd_train_student)

    p = sub.add_parser("train-students")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--path-batch", type=int, default=1024)
    p.set_defaults(func=cmd_train_students)

    p = sub.add_parser("run")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--devices", default="0,1")
    p.add_argument("--path-batch", type=int, default=1024)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("smoke")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--queries", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--path-batch", type=int, default=1024)
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("evaluate-dev")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--device", default="cuda:0")
    p.set_defaults(func=cmd_evaluate_dev)

    return ap


def main(argv=None) -> int:
    # The training pipeline's heavy maths runs on the GPU; the CPU side only
    # marshals tiny per-candidate slices.  Letting torch fan those out over every
    # core costs far more in thread synchronisation than it saves, so cap the
    # intra-op pool for stage processes.  Elementwise copies and GPU matmuls are
    # unaffected numerically.
    torch.set_num_threads(int(os.environ.get("MMDD_CPU_THREADS", "8")))
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
