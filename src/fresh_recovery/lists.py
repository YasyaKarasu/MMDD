"""Edge lists L0 / Ldir (SPEC 6.2-6.3) and the single T_BOOT refresh L1 / Ldir1 (SPEC 8.1).

Every list is one ``(relation, anchor)`` record with all train-known positives,
32 hard competitors from the relation's reservoir and 32 uniform random
competitors from the relation's legal library.  Candidate order is shuffled by
a namespace that never contains the SUP/KD branch name.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Sequence

import torch

from .config import Paths, RELATIONS
from .data import Labels, local_rng, utf8_sorted
from .io import sha256_file, write_json

HARD_N = 32
RANDOM_N = 32
RANDOM_ORDER_N = 64


def data_root_id(paths: Paths) -> str:
    provenance = json.loads((paths.work_dir / "LABEL_PROVENANCE.json").read_text(encoding="utf-8"))
    return str(provenance["semantic_fingerprint"])[:16]


def _reservoir_for(record: dict, reservoirs: dict[str, dict], et_reservoir: dict[str, list[str]],
                   *, qt_source: str) -> list[str]:
    relation, anchor = record["relation"], record["anchor_id"]
    if relation == "QT":
        return list(reservoirs[anchor][qt_source])
    if relation in ("Q_text", "Q_image"):
        return list(reservoirs[anchor]["qe_reservoir"][relation[2:]])
    return list(et_reservoir.get(anchor, []))


def _random_order(library: Sequence[str], exclude: set[str], namespace_parts: tuple) -> list[str]:
    remaining = [x for x in library if x not in exclude]
    rng = local_rng(*namespace_parts)
    n = min(RANDOM_ORDER_N, len(remaining))
    return rng.sample(remaining, n)


def _finish_list(record: dict, positives: list[str], hard: list[str], random_order: list[str],
                 *, order_namespace: tuple, random_n: int = RANDOM_N) -> dict:
    hard_set = set(hard)
    random_ids = [x for x in random_order if x not in hard_set and x not in set(positives)][:random_n]
    candidates = list(dict.fromkeys([*positives, *hard, *random_ids]))
    rng = local_rng(*order_namespace)
    rng.shuffle(candidates)
    pset = set(positives)
    return {
        "item_id": record["item_id"],
        "relation": record["relation"],
        "anchor_id": record["anchor_id"],
        "positives": list(positives),
        "ignore": [],
        "hard": list(hard),
        "random": random_ids,
        "random_order": list(random_order),
        "candidates": candidates,
        "active": bool(pset) and any(c not in pset for c in candidates),
    }


def build_l0(paths: Paths, labels: Labels, reservoirs: dict[str, dict], et_reservoir: dict[str, list[str]],
             *, seed: int, log=print) -> dict:
    """L0 (five relations) and Ldir (QT-only control) for one seed."""
    root = data_root_id(paths)
    started = time.time()
    lists: dict[str, dict] = {}
    direct_lists: dict[str, dict] = {}
    for n, record in enumerate(labels.edge_anchors, 1):
        relation, anchor = record["relation"], record["anchor_id"]
        positives = utf8_sorted(record["positive_ids"])
        pset = set(positives)
        library = labels.library(relation)
        reservoir = _reservoir_for(record, reservoirs, et_reservoir, qt_source="qt_reservoir")
        hard = [x for x in reservoir if x not in pset][:HARD_N]
        random_order = _random_order(library, pset | set(hard), (root, "L0", seed, relation, anchor))
        lists[record["item_id"]] = _finish_list(
            record, positives, hard, random_order, order_namespace=(root, "L0-order", seed, relation, anchor))
        if relation == "QT":
            reservoir_dir = _reservoir_for(record, reservoirs, et_reservoir, qt_source="qt_top256")
            hard_dir = [x for x in reservoir_dir if x not in pset][:HARD_N]
            random_dir = _random_order(library, pset | set(hard_dir), (root, "Ldir", seed, relation, anchor))
            direct_lists[record["item_id"]] = _finish_list(
                record, positives, hard_dir, random_dir, order_namespace=(root, "Ldir-order", seed, relation, anchor))
        if n % 5000 == 0:
            log(json.dumps({"event": "l0_progress", "done": n, "total": len(labels.edge_anchors),
                            "elapsed": round(time.time() - started, 1)}))
    out_dir = paths.work_dir / "lists" / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    l0_sha = save_lists(out_dir / "L0.pkl", lists)
    ldir_sha = save_lists(out_dir / "Ldir.pkl", direct_lists)
    report = {
        "seed": seed, "data_root_id": root, "L0": list_stats(lists), "Ldir": list_stats(direct_lists),
        "L0_sha256": l0_sha, "Ldir_sha256": ldir_sha, "hard_n": HARD_N, "random_n": RANDOM_N,
        "namespace": "SHA256(prefix|data_root|stage|seed|relation|anchor)[:8] big-endian; random.sample without replacement",
        "elapsed_seconds": time.time() - started,
    }
    write_json(out_dir / "L0_REPORT.json", report)
    return report


def list_stats(lists: dict[str, dict]) -> dict:
    by_relation: dict[str, dict[str, float]] = {}
    for row in lists.values():
        stat = by_relation.setdefault(row["relation"], {"lists": 0, "active": 0, "candidates": 0, "positives": 0})
        stat["lists"] += 1
        stat["active"] += int(row["active"])
        stat["candidates"] += len(row["candidates"])
        stat["positives"] += len(row["positives"])
    return {r: by_relation[r] for r in RELATIONS if r in by_relation}


def save_lists(path: Path, lists: dict[str, dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(lists, handle, protocol=4)
    tmp.replace(path)
    return sha256_file(path)


def load_lists(path: Path) -> dict[str, dict]:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


# ------------------------------------------------------------ T_BOOT refresh ---


def refresh_lists(
    model,
    bank,
    labels: Labels,
    lists: dict[str, dict],
    reservoirs: dict[str, dict],
    et_reservoir: dict[str, list[str]],
    *,
    seed: int,
    root: str,
    stage_tag: str,
    qt_source: str,
    device: str,
    pair_scorer,
    log=print,
) -> dict[str, dict]:
    """SPEC 8.1: score reservoir u P with the frozen Teacher, take 32 hard, keep the
    L0 random draw, and attach the Teacher logits for the final list."""
    out: dict[str, dict] = {}
    started = time.time()
    model.eval()
    with torch.no_grad():
        for n, (item_id, row) in enumerate(lists.items(), 1):
            relation, anchor = row["relation"], row["anchor_id"]
            positives = list(row["positives"])
            pset = set(positives)
            reservoir = _reservoir_for(row, reservoirs, et_reservoir, qt_source=qt_source)
            pool = list(dict.fromkeys([*positives, *reservoir, *row["random_order"]]))
            scores = pair_scorer(model, bank, anchor, pool, device)
            values = scores.detach().float().cpu().tolist()
            score_of = dict(zip(pool, values))
            competitors = [x for x in reservoir if x not in pset]
            competitors.sort(key=lambda x: (-score_of[x], x.encode("utf-8")))
            hard = competitors[:HARD_N]
            new = _finish_list(row, positives, hard, row["random_order"],
                               order_namespace=(root, f"{stage_tag}-order", seed, relation, anchor))
            new["teacher_logits"] = [score_of[c] for c in new["candidates"]]
            out[item_id] = new
            if n % 2000 == 0:
                log(json.dumps({"event": "refresh_progress", "stage": stage_tag, "done": n, "total": len(lists),
                                "elapsed": round(time.time() - started, 1)}))
    return out
