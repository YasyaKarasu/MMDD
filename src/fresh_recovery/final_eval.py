"""Final dev/test evaluation and the SPEC 15 decision (A26-A30, A34)."""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from . import evaluate as ev, metrics, raw as raw_mod, runlog, stages, student as student_mod, teacher as teacher_mod
from .config import Paths
from .data import split_query_ids, utf8_sorted
from .io import sha256_file, write_json
from .pools import PoolRecord, assemble_pool, pool_summary
from .retrieval import OwnRetriever

GATES = {
    "overall_real_minus_qtcont_min": -0.005,
    "implicit_real_minus_qtcont_min": 0.0,
    "implicit_real_minus_f0_min": 0.0,
    "implicit_real_minus_swap_min": 0.005,
    "effective_swap_min": 0.95,
}
STUDENT_GENERATORS = {"S_SUP_NATIVE": "S_SUP_NATIVE_C2", "S_KD_NATIVE": "S_KD_NATIVE_C2",
                      "S_QT_SUP": "S_QT_SUP_C2", "S_QT_KD": "S_QT_KD_C2"}
K20_POLICIES = ("P0_K20_C100_OLD", "P1_K20_C150_OLD")


def _content_key(rt: stages.Runtime) -> dict[str, str]:
    from .io import read_jsonl_gz

    return {row["asset_id"]: row["content_key"] for row in read_jsonl_gz(rt.paths.work_dir / "CONTENT_ALIASES.jsonl.gz")}


def _k20_pools(rt: stages.Runtime, source: Mapping[str, PoolRecord], retriever, raw_index,
               queries: Sequence[str], bank) -> dict[str, PoolRecord]:
    """Keep D100 and requery ET second-hop20 for the read-only P0/P1 controls."""
    target_ids = raw_index.targets if retriever is None else retriever.legal
    out = {}
    for qid in queries:
        main = source[qid]
        if retriever is None:
            direct = main.direct
            all_scores = raw_index.target_scores(raw_index.vector(qid)).detach().cpu().tolist()
            evidence = [e for modality in ("text", "image") for e, _ in main.first_hop[modality]]
            second = raw_index.second_hops(evidence, 20)
            second_hop = lambda e, modality: second[e]
            ef = None
        else:
            direct, ef = main.direct, main.retrieval_meta.get("ef_direct")
            qvec = torch.from_numpy(retriever.query_vector("QT", qid)).to(rt.device)
            all_scores = (retriever.target_vectors_gpu @ qvec).detach().cpu().tolist()
            second_hop = lambda e, modality: retriever.second_hop(e, modality, 20)
        qt_scores = dict(zip(target_ids, all_scores))
        out[qid] = assemble_pool(
            split=main.split, query_id=qid, generator_id=main.generator_id, model_sha=main.model_sha,
            target_index_sha=main.target_index_sha, evidence_index_sha=main.evidence_index_sha,
            direct=direct, direct_exact=[t for t, _ in direct], first_hop=main.first_hop,
            second_hop=second_hop, query_rows=rt.rows.get(qid),
            evidence_z=lambda ids: bank.z_many(list(ids)).float().cpu().numpy(),
            content_key=None, qt_scores=qt_scores,
            retrieval_meta={"policy": "K20_actual_ET_requery", "ef_direct100": ef, "second_hop_k": 20},
        )
    return out


def cmd_evaluate(paths: Paths, *, seed: int, split: str, device: str,
                 teachers: Sequence[str] = ("T_QT", "T_QT_FROM_BOOT", "T_QT_CONT", "T_PATH")) -> dict:
    if split == "test":
        stages.verify_model_lock(paths, seed=seed)
    rt = stages.Runtime(paths, seed=seed, device=device)
    stage = f"EVAL_{split.upper()}"
    log = rt.stage_log(stage)
    out_dir = rt.stage_dir(stage)
    raw_path = paths.work_dir / "raw" / split / "pools.pkl"
    parents = {"raw_pools": sha256_file(raw_path)}
    parents.update({name: sha256_file(rt.stage_dir(stage_name) / "selected.pt")
                    for name, stage_name in STUDENT_GENERATORS.items()})
    parents.update({name: rt.checkpoint_sha(name) for name in teachers})
    model_lock = paths.work_dir / f"seed{seed}" / "MODEL_LOCK.json"
    runlog.pre_run(out_dir, stage=stage, seed=seed, paths=paths, parents=parents,
                   inputs={"split": split, "model_lock_sha256": sha256_file(model_lock) if model_lock.exists() else None},
                   initial_state_sha="evaluation_only", optimizer_state="none",
                   config={"teachers": list(teachers), "generators": list(STUDENT_GENERATORS),
                           "policies": ["P0_K20_C100_OLD", "P1_K20_C150_OLD",
                                        "P2_K50_C150_OLD", "P3_K50_C150_QTALL"]})
    gt = rt.dev_gt(split)
    queries = utf8_sorted(gt)
    gold = {q: gt[q]["G"] for q in queries}
    kinds = {q: gt[q]["kind"] for q in queries}
    groups = {q: gt[q]["source_group"] for q in queries}
    bank = rt.bank()
    content_key = _content_key(rt)
    started = time.time()

    # ---- generators: raw + Students --------------------------------------------------
    pools: dict[str, dict[str, PoolRecord]] = {"Raw": rt.raw_pools(split)}
    retrievers: dict[str, OwnRetriever] = {}
    rankings: dict[str, dict[str, list[str]]] = {}
    pools_for: dict[str, dict[str, list[str]]] = {}
    direct150: dict[str, dict[str, list[str]]] = {}
    direct150_exact: dict[str, dict[str, list[str]]] = {}
    generator_meta: dict[str, dict] = {"Raw": {"model_sha": next(iter(pools["Raw"].values())).model_sha}}
    parents["raw_pools"] = sha256_file(paths.work_dir / "raw" / split / "pools.pkl")
    student_models = {}
    for name, stage_name in STUDENT_GENERATORS.items():
        model, payload, sha = stages._selected_student(rt, stage_name)
        parents[stage_name] = sha
        qt_only = name.startswith("S_QT")
        model_sha = runlog.state_sha(model.state_dict())
        retriever = OwnRetriever(model, bank, rt.labels, rt.rows, device=rt.device, seed=seed,
                                 generator_id=f"{stage_name}:{payload.get('snapshot')}", model_sha=model_sha, qt_only=qt_only)
        retrievers[name] = retriever
        pools[name] = {q: retriever.pool(split, q) for q in queries}
        direct150[name] = {q: [t for t, _ in retriever.direct_m(q, 150)[0]] for q in queries}
        direct150_exact[name] = {q: [t for t, _ in retriever.direct_exact(q, 150)] for q in queries}
        generator_meta[name] = {"model_sha": model_sha, "checkpoint": payload.get("snapshot"), "retrieval": retriever.meta}
        student_models[name] = model
        log({"event": "generator_pools", "generator": name, "elapsed": round(time.time() - started, 1)})
    init_model = student_mod.make_student(rt.basis, qt_only=False).to(rt.device).eval()
    init_ret = OwnRetriever(init_model, bank, rt.labels, rt.rows, device=rt.device, seed=seed, generator_id="PCA_INIT",
                            model_sha=f"init:{runlog.state_sha(init_model.state_dict())[:16]}", qt_only=False)
    pools["PCA_INIT"] = {q: init_ret.pool(split, q) for q in queries}
    generator_meta["PCA_INIT"] = {"model_sha": init_ret.model_sha, "retrieval": init_ret.meta}
    raw_index = raw_mod.RawIndex(rt.bank().z_store, rt.labels, rt.device)
    direct150["Raw"] = {q: [t for t, _ in raw_index.topk_targets(raw_index.vector(q), 150)] for q in queries}
    direct150_exact["Raw"] = direct150["Raw"]
    k20 = {gen: _k20_pools(rt, pools[gen], retrievers.get(gen), raw_index, queries, bank)
           for gen in ("Raw", "S_SUP_NATIVE", "S_KD_NATIVE")}
    k20_files = {}
    k20_stats = {}
    for gen, by_query in k20.items():
        path = out_dir / f"pools_{gen}_K20.pkl"
        k20_files[path.name] = raw_mod.save_pools(path, by_query)
        k20_stats[gen] = {
            "D100_changed_queries": sum(by_query[q].direct_ids != pools[gen][q].direct_ids for q in queries),
            "evidence20_non_nested_queries": sum(not set(by_query[q].evidence_ids).issubset(pools[gen][q].evidence_ids)
                                                 for q in queries),
            "policy_outside_main_U_queries": sum(any(set(by_query[q].policies[p]) - set(pools[gen][q].U)
                                                       for p in K20_POLICIES) for q in queries),
        }
        for q in queries:
            for policy in K20_POLICIES:
                pools[gen][q].policies[policy] = by_query[q].policies[policy]
    write_json(out_dir / "POLICY20_REPORT.json", {"actual_K20_requery": True, "generators": k20_stats,
                                                   "pool_files": k20_files})
    del raw_index
    for name, gpools in pools.items():
        raw_mod.save_pools(out_dir / f"pools_{name}.pkl", gpools)
        rankings[f"{name}|Direct"] = {q: gpools[q].direct_ids for q in queries}
        pools_for[f"{name}|Direct"] = rankings[f"{name}|Direct"]
        rankings[f"{name}|Direct_exact"] = {q: gpools[q].direct_exact for q in queries}
        pools_for[f"{name}|Direct_exact"] = rankings[f"{name}|Direct_exact"]
        if name in direct150:
            rankings[f"{name}|Direct150"] = direct150[name]
            rankings[f"{name}|Direct150_exact"] = direct150_exact[name]
            pools_for[f"{name}|Direct150"] = direct150[name]
            pools_for[f"{name}|Direct150_exact"] = direct150_exact[name]
        if not name.startswith("S_QT"):
            rankings[f"{name}|Evidence"] = {q: gpools[q].evidence_ids for q in queries}
            pools_for[f"{name}|Evidence"] = rankings[f"{name}|Evidence"]
            rankings[f"{name}|C150"] = {q: gpools[q].C150 for q in queries}
            pools_for[f"{name}|C150"] = rankings[f"{name}|C150"]
            for policy_id in ("P0_K20_C100_OLD", "P1_K20_C150_OLD", "P2_K50_C150_OLD", "P3_K50_C150_QTALL"):
                if policy_id in K20_POLICIES and name not in k20:
                    continue
                rankings[f"{name}|{policy_id}"] = {q: gpools[q].policies.get(policy_id, gpools[q].C150) for q in queries}
                pools_for[f"{name}|{policy_id}"] = rankings[f"{name}|{policy_id}"]
            pools_for[f"{name}|U"] = {q: gpools[q].U for q in queries}

    # ---- Teachers ---------------------------------------------------------------------
    teacher_models = {}
    for tname in teachers:
        model, payload = teacher_mod.load_teacher(rt.stage_dir(tname) / "checkpoint.pt", rt.device, expect_stage=tname)
        teacher_models[tname] = model
        parents[tname] = rt.checkpoint_sha(tname)
    swap_reports = {}
    teacher_logits: dict[str, dict] = {}
    two_way_generators = [g for g in pools if not g.startswith("S_QT")]
    donor_ids = {e for gen in ("Raw", "S_SUP_NATIVE", "S_KD_NATIVE") for q in queries
                 for t in pools[gen][q].C150 for e in pools[gen][q].path_bag(t)}
    formal_swap = ev.swap_map(utf8_sorted(donor_ids), content_key, rt.labels.modality)
    write_json(out_dir / "SWAP_DONORS.json", {"scope": "Raw_SUP_KD_P3_retained_E_same_split",
                                               "split": split, "donors": formal_swap})
    for tname, model in teacher_models.items():
        scorer = ev.TeacherScorer(model, bank, rt.device, name=tname)
        if tname in ("T_QT", "T_QT_CONT"):
            for gen in ("Raw", "S_SUP_NATIVE", "S_KD_NATIVE", "S_QT_SUP", "S_QT_KD"):
                ids_by_query = direct150[gen]
                key = f"{gen}+{tname}|D150"
                rankings[key] = {q: ev.rank_by(scorer.f0(q, ids_by_query[q])) for q in queries}
                pools_for[key] = ids_by_query
        for gen in two_way_generators:
            with_paths = tname == "T_PATH"
            swap = None
            if with_paths:
                swap = formal_swap
            result = ev.teacher_readouts(scorer, pools[gen], generator_id=pools[gen][queries[0]].generator_id,
                                         with_paths=with_paths, swap=swap, queries=queries,
                                         exclude_policies=K20_POLICIES, log=log)
            for key, rows in result["rankings"].items():
                variant, budget = key.split("|")
                label = f"{gen}+{tname}" if variant == "f0" and not with_paths else f"{gen}+{tname}_{variant}"
                rankings[f"{label}|{budget}"] = rows
                if budget in ("C150", "C100"):
                    pool_key = f"{gen}|C150"
                elif budget in pools[gen][queries[0]].policies:
                    pool_key = f"{gen}|{budget}"
                elif budget == "FullU":
                    pool_key = f"{gen}|U"
                else:
                    pool_key = f"{gen}|Direct"
                pools_for[f"{label}|{budget}"] = pools_for[pool_key]
            if gen in k20:
                label = f"{gen}+{tname}_f0" if with_paths else f"{gen}+{tname}"
                for policy in K20_POLICIES:
                    ids = {q: k20[gen][q].policies[policy] for q in queries}
                    rankings[f"{label}|{policy}"] = {}
                    for q in queries:
                        scores = scorer.f0(q, ids[q])
                        rankings[f"{label}|{policy}"][q] = ev.rank_by(scores)
                        result["logits"][q].setdefault("qt_scores_k20", {}).update(scores)
                    pools_for[f"{label}|{policy}"] = ids
            teacher_logits[f"{tname}|{gen}"] = result["logits"]
            if with_paths:
                swap_reports[gen] = result["swap"]
        for gen in ("S_QT_SUP", "S_QT_KD"):
            if tname not in ("T_QT", "T_QT_FROM_BOOT", "T_QT_CONT"):
                continue
            result = ev.teacher_readouts(scorer, pools[gen], generator_id=pools[gen][queries[0]].generator_id,
                                         with_paths=False, swap=None, queries=queries, log=log)
            rankings[f"{gen}+{tname}|D100"] = result["rankings"]["f0|D100"]
            pools_for[f"{gen}+{tname}|D100"] = pools_for[f"{gen}|Direct"]
            teacher_logits[f"{tname}|{gen}"] = result["logits"]
        if tname == "T_QT_CONT":
            matched = {}
            for gen in ("S_SUP_NATIVE", "S_KD_NATIVE"):
                m = ev.matched_direct_m(scorer, retrievers[gen], pools[gen], queries=queries)
                rankings[f"{gen}+T_QT_CONT|MatchedDirectM"] = m.pop("rankings")
                matched[gen] = m
            raw_index = raw_mod.RawIndex(rt.bank().z_store, rt.labels, rt.device)

            class _RawM:
                def direct_m(self, qid, m):
                    return raw_index.topk_targets(raw_index.vector(qid), m), None
            m = ev.matched_direct_m(scorer, _RawM(), pools["Raw"], queries=queries)
            rankings["Raw+T_QT_CONT|MatchedDirectM"] = m.pop("rankings")
            matched["Raw"] = m
            del raw_index
        log({"event": "teacher_done", "teacher": tname, "forwards": scorer.forwards, "elapsed": round(time.time() - started, 1)})

    # ---- metrics ----------------------------------------------------------------------
    table = ev.system_table(gold, kinds, rankings, pools_for)
    contrasts = _contrasts(gold, kinds, groups, rankings)
    strict = _strict(gold, pools, rankings)
    strict["direct150"] = _strict_150(gold, pools, rankings, direct150, direct150_exact)
    raw_index = raw_mod.RawIndex(rt.bank().z_store, rt.labels, rt.device)
    witness_funnels = {}
    for gen in ("Raw", "S_SUP_NATIVE", "S_KD_NATIVE"):
        retriever = retrievers.get(gen)
        ids = raw_index.targets if retriever is None else retriever.legal
        target_vectors = raw_index.z_targets if retriever is None else retriever.target_vectors_gpu
        positions = {target: i for i, target in enumerate(ids)}
        cached_q = None
        cached_scores = {}

        @torch.no_grad()
        def exact_rank(qid, evidence, target):
            nonlocal cached_q, cached_scores
            if cached_q != qid:
                cached_q, cached_scores = qid, {}
            key = evidence or "QT"
            if key not in cached_scores:
                if retriever is None:
                    vector = raw_index.vector(evidence or qid)
                else:
                    relation = "QT" if evidence is None else f"{rt.labels.modality[evidence]}_T"
                    vector = torch.from_numpy(retriever.query_vector(relation, evidence or qid)).to(rt.device)
                cached_scores[key] = target_vectors @ vector
            scores = cached_scores[key]
            position = positions[target]
            score = scores[position]
            return int((scores > score).sum().item() + (scores[:position] == score).sum().item() + 1)

        real = rankings[f"{gen}+T_PATH_Real|C150"]
        witness_funnels[gen] = _witness_funnel(gt, pools[gen], real, exact_rank)
        log({"event": "witness_funnel", "generator": gen,
             "pairs": witness_funnels[gen]["summary"]["I"]["pairs"],
             "visible": witness_funnels[gen]["summary"]["D"]["pairs"]})
    del raw_index
    write_json(out_dir / "WITNESS_FUNNELS.json", witness_funnels)
    if split == "dev":
        trajectory = _trajectory(rt, pools["S_KD_NATIVE"], gt, formal_swap, bank, log)
        write_json(out_dir / "TRAJECTORY.json", trajectory)
    rankings_sha = ev.save_rankings(out_dir / "rankings.json.gz", rankings)
    logits_sha = ev.save_json_gz(out_dir / "teacher_logits.json.gz", teacher_logits)
    gt_sha = ev.save_json_gz(out_dir / "gt.json.gz", gt)
    write_json(out_dir / "STRICT.json", strict)
    results = {
        "split": split, "seed": seed, "queries": len(queries), "test_status": "historically_exposed_regression" if split == "test" else "dev",
        "generators": generator_meta, "teachers": {t: parents[t] for t in teachers}, "systems": table,
        "contrasts": contrasts, "swap": swap_reports, "matched_direct_m": matched if "T_QT_CONT" in teachers else None,
        "strict_summary": {k: v for k, v in strict.items() if k != "per_query"},
        "files": {"rankings": rankings_sha, "teacher_logits": logits_sha, "gt": gt_sha, "K20_pools": k20_files},
        "policy20": k20_stats,
        "witness_funnel_summary": {gen: record["summary"] for gen, record in witness_funnels.items()},
        "elapsed_seconds": time.time() - started,
    }
    write_json(out_dir / "RESULTS.json", results)
    _write_csv(out_dir / "RESULTS.csv", table)
    runlog.post_run(out_dir, status="COMPLETE", counters={"queries": len(queries), "systems": len(table)},
                    outputs={name: sha256_file(out_dir / name) for name in
                             ("RESULTS.json", "RESULTS.csv", "rankings.json.gz", "teacher_logits.json.gz",
                              "gt.json.gz", "STRICT.json", "WITNESS_FUNNELS.json", "SWAP_DONORS.json", "POLICY20_REPORT.json",
                              *(f"pools_{gen}_K20.pkl" for gen in k20),
                              *(["TRAJECTORY.json"] if split == "dev" else []))})
    return results


def _trajectory(rt: stages.Runtime, pools: Mapping[str, PoolRecord], gt: Mapping[str, dict],
                donors: Mapping[str, str | None], bank, log) -> dict:
    """Frozen KD P3 candidates and donor IDs for both arms at init/half/end."""
    timeline = {"T_QT_CONT": ("init.pt", "half.pt", "end.pt"),
                "T_PATH": ("init.pt", "frac050.pt", "frac100.pt")}
    result = {"scope": "S_KD_NATIVE_P3_C150_fixed_dev", "snapshots": {}}
    gold = {q: row["G"] for q, row in gt.items()}
    kinds = {q: row["kind"] for q, row in gt.items()}
    queries = utf8_sorted(gt)
    for arm, snapshots in timeline.items():
        result["snapshots"][arm] = {}
        for tag, filename in zip(("init", "half", "end"), snapshots):
            checkpoint = rt.stage_dir(arm) / filename
            model, payload = teacher_mod.load_teacher(checkpoint, rt.device, expect_stage=arm)
            scorer = ev.TeacherScorer(model, bank, rt.device, name=f"{arm}_{tag}")
            ranks = {name: {} for name in (("f0", "Real", "Swap") if arm == "T_PATH" else ("f0",))}
            per_query_logits = {}
            scope = {"all_with_path": {"count": 0, "gap_sum": 0.0, "alpha0_sum": 0.0},
                     "verified_witness": {"count": 0, "gap_sum": 0.0, "alpha0_sum": 0.0}}
            witnessed = hit10 = slots = replaced = 0
            for qid in queries:
                pool = pools[qid]
                candidate_ids = pool.C150
                f0 = scorer.f0(qid, candidate_ids)
                scores = {"f0": f0}
                q_logits = {"f0": f0}
                if arm == "T_PATH":
                    natural = [(e, t) for t in candidate_ids for e in pool.path_bag(t)]
                    original = scorer.qet(qid, natural)
                    changed = [(donors.get(e) or e, t) for e, t in natural]
                    swapped = scorer.qet(qid, changed)
                    real, swap = {}, {}
                    for t in candidate_ids:
                        bag = pool.path_bag(t)
                        real[t] = ev.lse([f0[t], *(original[(e, t)] for e in bag)]) if bag else f0[t]
                        swap[t] = ev.lse([f0[t], *(swapped[(donors.get(e) or e, t)] for e in bag)]) if bag else f0[t]
                        if bag:
                            delta = real[t] - f0[t]
                            alpha = math.exp(-delta)
                            for name in ("all_with_path", "verified_witness"):
                                if name == "verified_witness" and not (set(bag) & set(gt[qid]["W"].get(t, ()))):
                                    continue
                                scope[name]["count"] += 1
                                scope[name]["gap_sum"] += delta
                                scope[name]["alpha0_sum"] += alpha
                    for e, _ in natural:
                        slots += 1
                        replaced += donors.get(e) is not None
                    scores.update(Real=real, Swap=swap)
                    q_logits.update(Real=real, Swap=swap,
                                    paths={t: [[e, original[(e, t)]] for e in pool.path_bag(t)]
                                           for t in candidate_ids if pool.path_bag(t)},
                                    swap_paths={t: [[donors.get(e) or e, swapped[(donors.get(e) or e, t)]]
                                                    for e in pool.path_bag(t)]
                                                for t in candidate_ids if pool.path_bag(t)})
                per_query_logits[qid] = q_logits
                for name, values in scores.items():
                    ranks[name][qid] = ev.rank_by(values)
                if arm == "T_PATH":
                    verified = {t for t in gt[qid]["G"] if set(pool.path_bag(t)) & set(gt[qid]["W"].get(t, ()))}
                    witnessed += len(verified)
                    hit10 += len(verified & set(ranks["Real"][qid][:10]))
            result["snapshots"][arm][tag] = {
                "checkpoint_sha256": sha256_file(checkpoint), "updates": payload["counters"]["updates"],
                "metrics": {name: metrics.grouped(gold, kinds, rows,
                                                   pools={q: pools[q].C150 for q in queries})
                            for name, rows in ranks.items()},
                "rankings": ranks, "forward_counts": scorer.forwards,
                "logits": per_query_logits,
                "witness_pairs_in_candidates_with_verified_retained_path": witnessed if arm == "T_PATH" else None,
                "witness_pairs_real_top10": hit10 if arm == "T_PATH" else None,
                "effective_swap": replaced / slots if slots else None,
                "path_scopes": {name: {"targets": agg["count"],
                                       "mean_f_minus_f0": agg["gap_sum"] / agg["count"] if agg["count"] else None,
                                       "mean_alpha0": agg["alpha0_sum"] / agg["count"] if agg["count"] else None}
                                for name, agg in scope.items()},
            }
            log({"event": "trajectory", "arm": arm, "snapshot": tag,
                 "queries": len(queries), "forwards": scorer.forwards})
            del model, scorer
    return result


def _contrasts(gold, kinds, groups, rankings) -> dict:
    pairs = {
        "BOOT_init_vs_fresh_same_pool": ("S_KD_NATIVE+T_QT_FROM_BOOT|C150", "S_KD_NATIVE+T_QT|C150"),
        "BOOT_init_vs_fresh_Raw": ("Raw+T_QT_FROM_BOOT|C150", "Raw+T_QT|C150"),
        "P_path_vs_QT_CONT": ("S_KD_NATIVE+T_PATH_Real|C150", "S_KD_NATIVE+T_QT_CONT|C150"),
        "P_real_vs_swap": ("S_KD_NATIVE+T_PATH_Real|C150", "S_KD_NATIVE+T_PATH_Swap|C150"),
        "P_real_vs_f0": ("S_KD_NATIVE+T_PATH_Real|C150", "S_KD_NATIVE+T_PATH_f0|C150"),
        "P_real_vs_structure": ("S_KD_NATIVE+T_PATH_Real|C150", "S_KD_NATIVE+T_PATH_Struct|C150"),
        "P_path_vs_QT_FullU": ("S_KD_NATIVE+T_PATH_Real|FullU", "S_KD_NATIVE+T_QT|FullU"),
        "P_real_vs_swap_FullU": ("S_KD_NATIVE+T_PATH_Real|FullU", "S_KD_NATIVE+T_PATH_Swap|FullU"),
        "KD_vs_SUP_same_TQT": ("S_KD_NATIVE+T_QT|C150", "S_SUP_NATIVE+T_QT|C150"),
        "KD_vs_SUP_same_TQT_CONT": ("S_KD_NATIVE+T_QT_CONT|C150", "S_SUP_NATIVE+T_QT_CONT|C150"),
        "KD_vs_SUP_direct": ("S_KD_NATIVE|Direct", "S_SUP_NATIVE|Direct"),
        "KD_C150_vs_D150_same_QT_CONT": ("S_KD_NATIVE+T_QT_CONT|C150", "S_KD_NATIVE+T_QT_CONT|D150"),
        "KD_vs_Raw_same_QT_CONT": ("S_KD_NATIVE+T_QT_CONT|C150", "Raw+T_QT_CONT|C150"),
        "R_KD_vs_raw_same_TQT": ("S_KD_NATIVE+T_QT|C150", "Raw+T_QT|C150"),
        "R_KD_vs_raw_same_TQT_FullU": ("S_KD_NATIVE+T_QT|FullU", "Raw+T_QT|FullU"),
        "Raw_path_vs_QT": ("Raw+T_PATH_Real|C150", "Raw+T_QT|C150"),
        "Raw_real_vs_swap": ("Raw+T_PATH_Real|C150", "Raw+T_PATH_Swap|C150"),
        "Raw_real_vs_f0": ("Raw+T_PATH_Real|C150", "Raw+T_PATH_f0|C150"),
        "MatchedDirectM_KD": ("S_KD_NATIVE+T_QT_CONT|MatchedDirectM", "S_KD_NATIVE+T_QT_CONT|FullU"),
        "QT_KD_vs_QT_SUP": ("S_QT_KD+T_QT|D100", "S_QT_SUP+T_QT|D100"),
    }
    out = {}
    implicit = [q for q in gold if kinds.get(q) == "implicit"]
    for name, (a, b) in pairs.items():
        if a not in rankings or b not in rankings:
            out[name] = {"status": "NOT_AVAILABLE", "systems": [a, b]}
            continue
        out[name] = {"systems": [a, b],
                     "overall": metrics.paired_bootstrap(gold, rankings[a], rankings[b], groups, k=10, seed=20260923),
                     "implicit": metrics.paired_bootstrap(gold, rankings[a], rankings[b], groups, k=10,
                                                          seed=20260923, queries=implicit)}
    return out


def _strict(gold, pools, rankings) -> dict:
    own = {}
    for gen in ("Raw", "S_SUP_NATIVE", "S_KD_NATIVE"):
        gp = pools[gen]
        finals = {k: v for k, v in rankings.items() if k.startswith(f"{gen}+") and k.endswith("|C150")}
        own[gen] = metrics.strict_own(gold, {q: gp[q].evidence_ids for q in gold}, {q: gp[q].direct_ids for q in gold},
                                      {q: gp[q].direct_exact for q in gold},
                                      {"U": {q: gp[q].U for q in gold}, "C150": {q: gp[q].C150 for q in gold}}, finals)
    direct_sets = []
    for gen in ("Raw", "S_SUP_NATIVE", "S_KD_NATIVE"):
        direct_sets.append({q: pools[gen][q].direct_ids for q in gold})
        direct_sets.append({q: pools[gen][q].direct_exact for q in gold})
    cohort = metrics.strict_fixed_cohort(gold, direct_sets)
    fixed = {}
    for gen in ("Raw", "S_SUP_NATIVE", "S_KD_NATIVE"):
        gp = pools[gen]
        finals = {k: v for k, v in rankings.items() if k.startswith(f"{gen}+") and k.endswith("|C150")}
        fixed[gen] = metrics.strict_own(cohort, {q: gp[q].evidence_ids for q in cohort}, {q: [] for q in cohort},
                                        {q: [] for q in cohort},
                                        {"U": {q: gp[q].U for q in cohort}, "C150": {q: gp[q].C150 for q in cohort}}, finals)
    summary = {"own": {g: {k: v for k, v in r.items() if k != "per_query"} for g, r in own.items()},
               "fixed_cohort": {"queries": len(cohort), "pairs": sum(len(v) for v in cohort.values()),
                                "definition": "G minus ANN/exact Direct100 of Raw, S_SUP_NATIVE, S_KD_NATIVE",
                                **{g: {k: v for k, v in r.items() if k != "per_query"} for g, r in fixed.items()}}}
    return {**summary, "per_query": {"own": {g: r["per_query"] for g, r in own.items()},
                                     "fixed_cohort_queue": cohort,
                                     "fixed": {g: r["per_query"] for g, r in fixed.items()}}}


def _strict_150(gold, pools, rankings, direct150, direct150_exact) -> dict:
    own = {}
    for gen in ("Raw", "S_SUP_NATIVE", "S_KD_NATIVE"):
        gp = pools[gen]
        finals = {k: v for k, v in rankings.items() if k.startswith(f"{gen}+") and k.endswith("|C150")}
        own[gen] = metrics.strict_own(gold, {q: gp[q].evidence_ids for q in gold}, direct150[gen],
                                      direct150_exact[gen],
                                      {"U": {q: gp[q].U for q in gold}, "C150": {q: gp[q].C150 for q in gold}}, finals)
    direct_sets = [direct for gen in ("Raw", "S_SUP_NATIVE", "S_KD_NATIVE")
                   for direct in (direct150[gen], direct150_exact[gen])]
    cohort = metrics.strict_fixed_cohort(gold, direct_sets)
    return {"own": {g: {k: v for k, v in r.items() if k != "per_query"} for g, r in own.items()},
            "per_query": {g: r["per_query"] for g, r in own.items()}, "fixed_cohort_queue": cohort,
            "definition": "G minus ANN/exact Direct150 of Raw, S_SUP_NATIVE, S_KD_NATIVE"}


def _witness_funnel(gt: Mapping[str, dict], pools: Mapping[str, PoolRecord],
                    real: Mapping[str, Sequence[str]], exact_rank) -> dict:
    """Implicit (q,t) witness arrivals, with ANN and full-lake exact ranks kept separate."""
    names = ("I", "A", "B", "C", "D", "F10", "F50")
    pairs: dict[str, dict[str, list[str]]] = {name: {} for name in names}
    target_pairs: dict[str, dict[str, list[str]]] = {name: {} for name in
                                                      ("eligible", "any_E", "U", "C150", "Top10")}
    rank_rows = []
    remaining = 0.0
    for qid, row in gt.items():
        pool = pools[qid]
        implicit = set(row["implicit_G"])
        first = {e for modality in ("text", "image") for e, _ in pool.first_hop[modality]}
        ranks = set(real[qid])
        top10, top50 = set(real[qid][:10]), set(real[qid][:50])
        pair_sets = {name: set() for name in names}
        pair_sets["I"] = implicit
        for tid in implicit:
            verified = set(row["W"].get(tid, ())) & first
            if not verified:
                continue
            pair_sets["A"].add(tid)
            ann = verified & set(pool.arrivals(tid))
            if ann:
                pair_sets["B"].add(tid)
            retained = ann & set(pool.path_bag(tid))
            if retained:
                pair_sets["C"].add(tid)
            if tid in pool.C150 and retained:
                pair_sets["D"].add(tid)
                if tid in top10:
                    pair_sets["F10"].add(tid)
                if tid in top50:
                    pair_sets["F50"].add(tid)
            per_e = {e: exact_rank(qid, e, tid) for e in utf8_sorted(verified)}
            best_e = min(per_e, key=lambda e: (per_e[e], e.encode("utf-8")))
            rank_rows.append({"query_id": qid, "target_id": tid, "et_ranks": per_e,
                              "best_witness": best_e, "best_et_rank": per_e[best_e],
                              "qt_exact_rank": exact_rank(qid, None, tid),
                              "et50_ann_arrived": bool(ann)})
        for name in names:
            pairs[name][qid] = utf8_sorted(pair_sets[name])
        stages = {"eligible": implicit, "any_E": implicit & set(pool.evidence_ids),
                  "U": implicit & set(pool.U), "C150": implicit & set(pool.C150),
                  "Top10": implicit & top10}
        for name, members in stages.items():
            target_pairs[name][qid] = utf8_sorted(members)
        if row["G"] and implicit:
            remaining += len((set(row["G"]) & set(pool.C150) & pair_sets["C"]) - top10) / len(row["G"])
        if not ranks.issubset(set(pool.U)):
            raise ValueError(f"Real ranked IDs outside own pool for {qid}")
        for below, above in (("A", "I"), ("B", "A"), ("C", "B"), ("D", "C"),
                             ("F10", "D"), ("F50", "D")):
            if not pair_sets[below] <= pair_sets[above]:
                raise ValueError(f"witness funnel not nested: {qid} {below} vs {above}")
    eligible = sum(len(ids) for ids in pairs["I"].values())
    active_queries = [q for q in gt if gt[q]["implicit_G"]]
    counts = {name: sum(map(len, by_q.values())) for name, by_q in pairs.items()}
    summary = {name: {"pairs": counts[name], "pair_micro": counts[name] / eligible if eligible else None,
                      "query_macro": (sum(len(pairs[name][q]) / len(gt[q]["implicit_G"])
                                          for q in active_queries) / len(active_queries)) if active_queries else None}
               for name in names}
    for current, previous in (("A", "I"), ("B", "A"), ("C", "B"), ("D", "C"),
                              ("F10", "D"), ("F50", "D")):
        summary[current]["lost_from_previous"] = counts[previous] - counts[current]
    rank_summary = {}
    for name, field in (("ET_exact", "best_et_rank"), ("QT_exact", "qt_exact_rank")):
        values = [record[field] for record in rank_rows]
        rank_summary[name] = {"median": float(np.median(values)) if values else None,
                              **{f"at_{k}": sum(value <= k for value in values) for k in (20, 50, 100)}}
    return {"pairs": pairs, "summary": summary, "target_pairs": target_pairs,
            "target_counts": {name: sum(map(len, by_q.values())) for name, by_q in target_pairs.items()},
            "exact_ranks_for_A": rank_rows, "exact_rank_summary": rank_summary,
            "witness_visible_C_remaining_macro_all_implicit_queries": remaining / len(active_queries) if active_queries else None,
            "definitions": "B is actual ET50 ANN arrival; C and D require the same verified E retained in the target bag"}


def _write_csv(path: Path, table: Mapping[str, dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["system", "group", "queries", "R10", "R20", "R30", "R40", "R50", "coverage", "Oracle10", "Oracle50"])
        for system, groups in table.items():
            for group, row in groups.items():
                writer.writerow([system, group, row["queries"], *[round(row.get(k, float("nan")), 6) for k in
                                                                  ("R10", "R20", "R30", "R40", "R50", "coverage", "Oracle10", "Oracle50")]])


# ---------------------------------------------------------------- decision ---


def cmd_decide(paths: Paths, *, seed: int) -> dict:
    rt = stages.Runtime(paths, seed=seed)
    results = json.loads((rt.stage_dir("EVAL_DEV") / "RESULTS.json").read_text())
    systems, contrasts = results["systems"], results["contrasts"]

    def m(system: str, group: str = "overall", key: str = "R10"):
        return systems.get(system, {}).get(group, {}).get(key)

    checks = []

    def check(name, value, threshold, *, op=">=", note=""):
        ok = value is not None and ((value >= threshold - 1e-12) if op == ">=" else (value > threshold + 1e-12))
        checks.append({"name": name, "value": value, "threshold": threshold, "op": op, "passed": bool(ok), "note": note})
        return bool(ok)

    qtcont = contrasts.get("P_path_vs_QT_CONT", {})
    swap = results.get("swap", {}).get("S_KD_NATIVE", {})
    p_ok = all([
        check("P_real_minus_QT_CONT_overall", qtcont.get("overall", {}).get("point_estimate"), GATES["overall_real_minus_qtcont_min"]),
        check("P_real_minus_QT_CONT_implicit", qtcont.get("implicit", {}).get("point_estimate"), GATES["implicit_real_minus_qtcont_min"]),
        check("P_real_minus_swap_implicit", contrasts.get("P_real_vs_swap", {}).get("implicit", {}).get("point_estimate"), GATES["implicit_real_minus_swap_min"]),
        check("P_swap_effective", swap.get("effective"), GATES["effective_swap_min"]),
    ])
    kd = contrasts.get("KD_vs_SUP_same_TQT", {"overall": {"point_estimate": None, "ci95": [None, None]}, "implicit": {}})
    kd_point = kd.get("overall", {}).get("point_estimate")
    kd_ci = kd.get("overall", {}).get("ci95", [None])[0]
    kd_ok = kd_point is not None and kd_ci is not None and kd_point > 0 and kd_ci > 0
    decision = {
        "seed": seed, "checks": checks,
        "R": "RECOVERY_REFERENCE_UNAVAILABLE",
        "R_note": "same-protocol historical weights are isolated and optional",
        "P": "SINGLE_SEED_PATH_PROMISING_NOT_SIGNIFICANCE_PROOF" if p_ok else "PATH_GAIN_NOT_ESTABLISHED",
        "KD": {"status": "SINGLE_SEED_KD_CI_POSITIVE_NOT_TWO_SEED_SUPPORT" if kd_ok else "KD_ADVANTAGE_NOT_ESTABLISHED",
               "overall": kd["overall"], "implicit": kd["implicit"]},
        "seed29_status": "NOT_RUN_PER_LATER_USER_INSTRUCTION",
        "main_system": "S_KD_NATIVE+T_PATH_Real|P3_K50_C150_QTALL",
        "rule": "single-seed result per later user instruction; QT never replaces Path; no two-seed conclusion",
        "gates": GATES,
    }
    write_json(rt.seed_dir() / "DECISION.json", decision)
    return decision


def cmd_latency(paths: Paths, *, seed: int, device: str, n_queries: int = 200) -> dict:
    rt = stages.Runtime(paths, seed=seed, device=device)
    out_dir = rt.stage_dir("EVAL_DEV")
    pools = raw_mod.load_pools(out_dir / "pools_S_KD_NATIVE.pkl")
    queries = utf8_sorted(pools)[:n_queries]
    bank = rt.bank()
    report = {"device": device, "note": "exclusive card; each query scored with fresh forwards, P3 C150 candidates"}
    for tname in ("T_QT", "T_QT_CONT", "T_PATH"):
        model, _ = teacher_mod.load_teacher(rt.stage_dir(tname) / "checkpoint.pt", rt.device, expect_stage=tname)
        factory = lambda m=model, t=tname: ev.TeacherScorer(m, bank, rt.device, name=t)
        for _ in range(5):  # warm-up
            factory().f0(queries[0], pools[queries[0]].C150)
        report[tname] = ev.measure_latency(factory, pools, queries=queries, with_paths=(tname == "T_PATH"), device=rt.device)
    write_json(out_dir / "LATENCY.json", report)
    return report
