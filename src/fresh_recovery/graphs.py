"""Model-independent graph construction for the v3.1 DAG.

C2 keeps its shared witness augmentation; final Teacher target items use only
natural retained bags and keep verified witness supervision in separate records.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .config import Paths
from .data import Labels, local_rng, utf8_sorted
from .io import sha256_file, write_json
from .pools import PoolRecord
from .student import build_c2_item
from .teacher import LogitStore, pair_logits


def augmented_witness(labels: Labels, query_id: str, *, root: str, seed: int, stage: str) -> str | None:
    """SPEC 9.1 / 11.2: one witness e+ per query, fixed by namespace before training."""
    assets = labels.witness_assets(query_id)
    if not assets:
        return None
    return assets[local_rng(root, stage, "augment", seed, query_id).randrange(len(assets))]


def build_native_c2_graph(labels: Labels, pools_by_generator: Mapping[str, Mapping[str, PoolRecord]], *,
                          root: str, seed: int) -> dict[str, dict]:
    """C_q = raw U u SUP U u KD U u G; bag = union of the generators' actual arrivals."""
    items = {}
    for qid in labels.query_ids:
        entry = labels.queries[qid]
        targets: set[str] = set(entry["G"])
        bags: dict[str, set[str]] = {}
        sources: dict[str, dict[str, list[str]]] = {}
        for name, pools in pools_by_generator.items():
            pool = pools[qid]
            targets |= set(pool.U)
            for t in pool.pre_paths:
                arrivals = pool.arrivals(t)
                bags.setdefault(t, set()).update(arrivals)
                sources.setdefault(t, {})[name] = arrivals
        target_list = utf8_sorted(targets)
        natural = {t: utf8_sorted(bags[t]) for t in target_list if t in bags}
        aug = augmented_witness(labels, qid, root=root, seed=seed, stage="C2")
        item = build_c2_item(qid, targets=target_list, gold=entry["G"], witness=entry["W"],
                             natural_bags=natural, augmented_e=aug)
        item["sources"] = sources
        items[qid] = item
    return items


def build_qt_c2_graph(labels: Labels, reservoirs: Mapping[str, dict],
                      direct_by_generator: Mapping[str, Mapping[str, Sequence[str]]]) -> dict[str, dict]:
    """SPEC 9.4: raw QT256 u both QT-C1 own Direct100 u G; Direct term only."""
    items = {}
    for qid in labels.query_ids:
        entry = labels.queries[qid]
        targets = set(entry["G"]) | set(reservoirs[qid]["qt_top256"])
        for direct in direct_by_generator.values():
            targets |= set(direct[qid])
        item = build_c2_item(qid, targets=utf8_sorted(targets), gold=entry["G"], witness={}, natural_bags={},
                             augmented_e=None)
        item["views"] = []
        items[qid] = item
    return items


def save_graph(path: Path, items: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(items, handle, protocol=4)
    tmp.replace(path)
    return sha256_file(path)


def load_graph(path: Path) -> dict:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def graph_stats(items: Mapping[str, dict]) -> dict:
    n = len(items)
    return {
        "queries": n,
        "active_direct": sum(1 for it in items.values() if it["active"]),
        "mean_targets": sum(len(it["targets"]) for it in items.values()) / max(n, 1),
        "mean_evidence": sum(len(it["evidence"]) for it in items.values()) / max(n, 1),
        "mean_path_slots": sum(len(v["e_index"]) for it in items.values() for v in it["views"][:1]) / max(n, 1),
        "queries_with_active_e_view": sum(1 for it in items.values() if any(v["active"] for v in it["views"])),
        "queries_with_augmented": sum(1 for it in items.values() if it.get("augmented_e")),
    }


# --------------------------------------------------- frozen Teacher logits ---


@torch.no_grad()
def precompute_c2_teacher(model, bank, items: Mapping[str, dict], *, device: str, teacher_sha: str,
                          store_path: Path, chunk: int = 256, log=print) -> dict:
    """T_BOOT pair logits for D_T (q,t), QE (q,e) and ET (e,t) on the C2 graph.

    Unique (e,t) pairs are scored once across all queries (evidence anchor ->
    target destination), which is exactly the pair forward the per-query loop
    would run; only redundant repeats are removed.
    """
    model.eval()
    store = LogitStore(store_path, teacher_sha=teacher_sha, stage="C2_TEACHER")
    started = time.time()
    # 1. (e -> t) pairs grouped by evidence anchor.
    by_evidence: dict[str, set[str]] = {}
    for item in items.values():
        for view in item["views"]:
            for e_i, t_i in zip(view["e_index"], view["t_index"]):
                by_evidence.setdefault(item["evidence"][e_i], set()).add(item["targets"][t_i])
    et_scores: dict[tuple[str, str], float] = {}
    for n, e in enumerate(utf8_sorted(by_evidence), 1):
        targets = utf8_sorted(by_evidence[e])
        key = LogitStore.key("pair", e, targets)
        values = store.get(key)
        if values is None:
            values = pair_logits(model, bank, e, targets, device, chunk=chunk)
            store.put(key, values)
        for t, v in zip(targets, values.tolist()):
            et_scores[(e, t)] = v
        if n % 2000 == 0:
            log({"event": "c2_teacher_et", "done": n, "total": len(by_evidence), "elapsed": round(time.time() - started, 1)})
    # 2. per query: D_T over targets, QE over evidence.
    out: dict[str, dict] = {}
    for n, (qid, item) in enumerate(sorted(items.items()), 1):
        cache: dict = {}
        d_key = LogitStore.key("pair", qid, item["targets"], "D")
        d_values = store.get(d_key)
        if d_values is None:
            d_values = pair_logits(model, bank, qid, item["targets"], device, chunk=chunk, cache=cache)
            store.put(d_key, d_values)
        record = {"D": d_values.detach().float().cpu()}
        if item["evidence"]:
            qe_key = LogitStore.key("pair", qid, item["evidence"], "QE")
            qe_values = store.get(qe_key)
            if qe_values is None:
                qe_values = pair_logits(model, bank, qid, item["evidence"], device, chunk=chunk, cache=cache)
                store.put(qe_key, qe_values)
            qe = qe_values.detach().float().cpu()
            record["QE"] = qe
            record["ET"] = []
            for view in item["views"]:
                flat = torch.tensor(
                    [float(qe[e_i]) + et_scores[(item["evidence"][e_i], item["targets"][t_i])]
                     for e_i, t_i in zip(view["e_index"], view["t_index"])], dtype=torch.float32)
                record["ET"].append(flat)
        out[qid] = record
        if n % 1000 == 0:
            log({"event": "c2_teacher_query", "done": n, "total": len(items), "elapsed": round(time.time() - started, 1)})
            store.save()
    store_sha = store.save()
    return {"logits": out, "store_sha256": store_sha, "unique_et_pairs": len(et_scores),
            "evidence_anchors": len(by_evidence), "elapsed_seconds": time.time() - started}


def save_teacher_logits(path: Path, logits: Mapping[str, dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(logits), tmp)
    tmp.replace(path)
    return sha256_file(path)


# ---------------------------------------------------------------- hard32 ----


def mine_hard32(labels: Labels, pools: Mapping[str, PoolRecord], *, generator_id: str, hard_n: int = 32) -> dict:
    """SPEC 10.1: hard32 = first 32 non-G legal targets of S_KD_NATIVE own Direct ANN top100."""
    legal = set(labels.legal)
    out = {}
    for qid in labels.query_ids:
        pool = pools[qid]
        pool.validate(expected_generator=generator_id, query_id=qid)
        g = set(labels.queries[qid]["G"])
        top100 = pool.direct_ids
        hard, filtered = [], []
        for t in top100:
            if t in g:
                filtered.append({"target": t, "reason": "in_G"})
            elif t not in legal:
                filtered.append({"target": t, "reason": "not_legal"})
            elif len(hard) < hard_n:
                hard.append(t)
        out[qid] = {"top100": top100, "filtered": filtered, "hard32": hard,
                    "model_sha": pool.model_sha, "target_index_sha": pool.target_index_sha,
                    "gt_sha": labels.files["train_queries.jsonl.gz"]}
    return out


def apply_hard32(lists: Mapping[str, dict], hard32: Mapping[str, dict], *, root: str, seed: int) -> dict[str, dict]:
    """SPEC 10.2: append hard32 to the QT lists of L0 (dedup, GT-protected), reshuffle by namespace."""
    out = {}
    for item_id, row in lists.items():
        new = dict(row)
        if row["relation"] == "QT":
            extra = [t for t in hard32[row["anchor_id"]]["hard32"] if t not in set(row["positives"])]
            candidates = list(dict.fromkeys([*row["candidates"], *extra]))
            local_rng(root, "T_QT-order", seed, "QT", row["anchor_id"]).shuffle(candidates)
            new["candidates"] = candidates
            new["hard32_extra"] = extra
            new["active"] = bool(row["positives"]) and any(c not in set(row["positives"]) for c in candidates)
        out[item_id] = new
    return out


# ------------------------------------------------------------ T_PATH items ---


def conditional_candidates(labels: Labels, qid: str, asset: str, *, reservoirs: Mapping[str, dict],
                           et_reservoir: Mapping[str, list[str]], own_direct: Sequence[str]) -> dict | None:
    """SPEC 11.4 L_cond list: P[q,e]={t: e in W[q,t]}, I=(G u Epos[e]) \\ P, C_cond = rawQT256 u rawET256(e) u own D100 u P."""
    entry = labels.queries[qid]
    positives = utf8_sorted(t for t, assets in entry["W"].items() if asset in assets)
    if not positives:
        return None
    pset = set(positives)
    ignore = (set(entry["G"]) | set(labels.epos.get(asset, []))) - pset
    candidates = utf8_sorted(set(reservoirs[qid]["qt_top256"]) | set(et_reservoir.get(asset, [])) | set(own_direct) | pset)
    allowed = [t not in ignore for t in candidates]
    active = any(t in pset for t in candidates) and any(a and t not in pset for t, a in zip(candidates, allowed))
    if not active:
        return None
    return {"asset": asset, "candidates": candidates, "positives": positives, "ignore": utf8_sorted(ignore), "allowed": allowed}


def build_path_items(labels: Labels, own_pools: Mapping[str, PoolRecord], raw_pools: Mapping[str, PoolRecord], *,
                     own_generator: str, reservoirs: Mapping[str, dict], et_reservoir: Mapping[str, list[str]],
                     root: str, seed: int) -> dict[str, dict]:
    items = {}
    for qid in labels.query_ids:
        entry = labels.queries[qid]
        own, raw = own_pools[qid], raw_pools[qid]
        own.validate(expected_generator=own_generator, query_id=qid)
        raw.validate(expected_generator="raw", query_id=qid)
        targets = utf8_sorted(set(own.C150) | set(raw.C150) | set(entry["G"]))
        natural: dict[str, list[str]] = {}
        for t in targets:
            bag = utf8_sorted(set(own.path_bag(t)) | set(raw.path_bag(t)))
            if bag:
                natural[t] = bag
        conditions = []
        for asset in labels.witness_assets(qid):
            cond = conditional_candidates(labels, qid, asset, reservoirs=reservoirs, et_reservoir=et_reservoir,
                                          own_direct=own.direct_ids)
            if cond is not None:
                conditions.append(cond)
        positives = [t for t in targets if t in set(entry["G"])]
        active = bool(positives) and len(positives) < len(targets)
        items[qid] = {"query_id": qid, "targets": targets, "positives": positives, "natural": natural,
                      "conditions": conditions, "support_records": [], "active": active,
                      "sources": {"own": own_generator, "raw": "raw", "own_model_sha": own.model_sha}}
    return items


@torch.no_grad()
def build_support_records(labels: Labels, items: Mapping[str, dict], own_pools: Mapping[str, PoolRecord],
                          raw_pools: Mapping[str, PoolRecord], model, bank, *, device: str,
                          root: str, seed: int, max_hard: int = 4, max_random: int = 4) -> dict[str, list[dict]]:
    """Build independent verified-witness records and protected competitors."""
    from .teacher import triplet_logits

    model.eval()
    out: dict[str, list[dict]] = {}
    for qid in labels.query_ids:
        entry = labels.queries[qid]
        natural_assets = set()
        for pool in (own_pools[qid], raw_pools[qid]):
            for target in pool.pre_paths:
                natural_assets.update(pool.arrivals(target))
        records: list[dict] = []
        for target in utf8_sorted(entry["G"]):
            positive = utf8_sorted(entry["W"].get(target, ()))
            if not positive:
                continue
            protected = set(positive)
            protected.update(labels.canonical.get(asset, asset) for asset in positive)
            for other_target, assets in entry["W"].items():
                if other_target != target:
                    protected.update(assets)
            protected.update(asset for asset, targets in labels.epos.items() if target in targets)
            legal_pool = [asset for asset in utf8_sorted(natural_assets)
                          if asset not in protected and labels.canonical.get(asset, asset) not in protected]
            pairs = [(asset, target) for asset in legal_pool]
            scores = triplet_logits(model, bank, qid, pairs, device) if pairs else torch.empty(0)
            scored = sorted(zip(legal_pool, scores.detach().cpu().tolist()),
                            key=lambda pair: (-float(pair[1]), pair[0].encode("utf-8")))
            hard = [asset for asset, _ in scored[:max_hard]]
            remaining = [asset for asset, _ in scored[max_hard:]]
            rng = local_rng(root, "FINAL_SUPPORT", seed, 0, qid, target)
            random_ids = rng.sample(remaining, min(max_random, len(remaining)))
            records.append({"query_id": qid, "target_id": target, "positive_evidence": positive,
                            "hard_competitors": hard, "random_competitors": random_ids,
                            "competitors": hard + random_ids, "label": "auxiliary_verified_witness",
                            "protected_ids": utf8_sorted(protected), "scorer": "T_QT_initial_QET_detached",
                            "margin": 0.5})
        out[qid] = records
    for qid, records in out.items():
        items[qid]["support_records"] = records
    return out


@torch.no_grad()
def precompute_anchor_logits(model, bank, items: Mapping[str, dict], *, device: str, teacher_sha: str,
                             store_path: Path, chunk: int = 256, log=print) -> dict[str, torch.Tensor]:
    """Frozen T_QT f0 over C_q for the SPEC 11.4 KL anchor."""
    model.eval()
    store = LogitStore(store_path, teacher_sha=teacher_sha, stage="T_PATH_ANCHOR")
    out = {}
    started = time.time()
    for n, (qid, item) in enumerate(sorted(items.items()), 1):
        key = LogitStore.key("pair", qid, item["targets"], "anchor")
        values = store.get(key)
        if values is None:
            values = pair_logits(model, bank, qid, item["targets"], device, chunk=chunk)
            store.put(key, values)
        out[qid] = values.detach().float().cpu()
        if n % 1000 == 0:
            log({"event": "anchor_logits", "done": n, "total": len(items), "elapsed": round(time.time() - started, 1)})
            store.save()
    store.save()
    return out
