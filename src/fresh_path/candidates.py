"""Raw retrieval, admission and candidate lists (SPEC 5).

Everything here is produced from this run's frozen ``z`` and the raw labels:
no historical ranking byte is read, no calibration mapping is consulted.
Matrices and id lists are kept in UTF-8 ascending order so that a stable
descending ``argsort`` implements the required ``(-score, object_id)`` order.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from . import config
from .contracts import local_rng
from .inputs import TrainLabels


def utf8_sorted(items: Iterable[str]) -> list[str]:
    return sorted(items, key=lambda x: x.encode("utf-8"))


@dataclass
class RawStore:
    ids: list[str]
    types: list[str]
    z: torch.Tensor
    index: dict[str, int]

    @property
    def dim(self) -> int:
        return self.z.shape[1]

    def rows(self, ids: Sequence[str]) -> torch.Tensor:
        return self.z[torch.tensor([self.index[i] for i in ids], dtype=torch.long)]

    def positions(self, ids: Sequence[str]) -> torch.Tensor:
        return torch.tensor([self.index[i] for i in ids], dtype=torch.long)


def load_z(features_dir: Path, *, dtype=torch.float32) -> RawStore:
    index = __import__("json").loads((Path(features_dir) / "z_index.json").read_text())
    ids = [str(x) for x in index["ids"]]
    types = [str(x) for x in index["types"]]
    order = np.argsort(np.array([x.encode("utf-8") for x in ids], dtype=object))
    ids = [ids[int(i)] for i in order]
    types = [types[int(i)] for i in order]
    raw = np.load(Path(features_dir) / "z.f32.npy", mmap_mode="r")
    z = torch.from_numpy(np.asarray(raw)[order]).to(dtype)
    return RawStore(ids=ids, types=types, z=z, index={oid: i for i, oid in enumerate(ids)})


def ordered_ids(store: RawStore, object_type: str) -> list[str]:
    return [oid for oid, t in zip(store.ids, store.types) if t == object_type]


def stable_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Scores over a UTF-8 ordered axis -> top-k positions in (-score, id) order."""
    k = min(k, scores.numel())
    return torch.argsort(scores, descending=True, stable=True)[:k]


def _lse(values: Sequence[float]) -> float:
    return float(torch.logsumexp(torch.tensor(list(values), dtype=torch.float64), 0))


def build_raw(
    store: RawStore,
    labels: TrainLabels,
    *,
    query_ids: Sequence[str],
    direct: int = 100,
    hard_pool: int = 256,
    first_hop: int = 20,
    targets_per_evidence: int = 20,
    retained_paths: int = 4,
    rrf_k: int = 60,
    device: str = "cpu",
    chunk: int = 4096,
    progress: int = 0,
    et_anchors: Sequence[str] | None = None,
) -> dict:
    torch_device = torch.device(device)
    z = store.z.to(torch_device)
    target_ids = utf8_sorted(labels.legal)
    text_ids = ordered_ids(store, "text")
    image_ids = ordered_ids(store, "image")
    target_pos = store.positions(target_ids).to(torch_device)
    text_pos = store.positions(text_ids).to(torch_device)
    image_pos = store.positions(image_ids).to(torch_device)
    z_target = z[target_pos]
    z_text = z[text_pos]
    z_image = z[image_pos]

    qt_reservoir: dict[str, list[str]] = {}
    qt_top256: dict[str, list[str]] = {}
    qe_reservoir: dict[str, dict[str, list[str]]] = {}
    admission: dict[str, dict] = {}
    for n, qid in enumerate(query_ids):
        qz = z[store.index[qid]].to(torch_device)
        s_target = z_target @ qz
        d256 = stable_topk(s_target, hard_pool)
        d100 = d256[:direct]
        direct_ids = [target_ids[int(i)] for i in d100]
        s_text = z_text @ qz
        s_image = z_image @ qz
        t256 = stable_topk(s_text, hard_pool)
        i256 = stable_topk(s_image, hard_pool)
        t20 = t256[:first_hop]
        i20 = i256[:first_hop]
        qe_reservoir[qid] = {
            "text": [text_ids[int(i)] for i in t256],
            "image": [image_ids[int(i)] for i in i256],
        }

        # ---- natural evidence paths: each first-hop e proposes its top targets
        path_scores: dict[str, list[tuple[float, str]]] = {}
        for eid, is_text in [(text_ids[int(i)], True) for i in t20] + [(image_ids[int(i)], False) for i in i20]:
            ez = z[store.index[eid]].to(torch_device)
            first = float(ez @ qz)
            s_e = z_target @ ez
            top = stable_topk(s_e, targets_per_evidence)
            for i in top:
                tid = target_ids[int(i)]
                path_scores.setdefault(tid, []).append((first + float(s_e[int(i)]), eid))
        evidence_ranked: list[tuple[str, float]] = []
        paths: dict[str, list[list[object]]] = {}
        for tid, entries in path_scores.items():
            entries.sort(key=lambda p: (-p[0], p[1].encode("utf-8")))
            kept = entries[:retained_paths]
            paths[tid] = [[e, score] for score, e in kept]
            evidence_ranked.append((tid, _lse([s for s, _ in kept])))
        evidence_ranked.sort(key=lambda p: (-p[1], p[0].encode("utf-8")))
        evidence_ids = [t for t, _ in evidence_ranked]

        rrf: dict[str, float] = {}
        for rank, tid in enumerate(direct_ids, 1):
            rrf[tid] = rrf.get(tid, 0.0) + 1.0 / (rrf_k + rank)
        for rank, tid in enumerate(evidence_ids, 1):
            rrf[tid] = rrf.get(tid, 0.0) + 1.0 / (rrf_k + rank)
        u_ids = utf8_sorted(rrf)
        c100 = sorted(rrf, key=lambda t: (-rrf[t], t.encode("utf-8")))[:direct]

        reservoir = list(dict.fromkeys(direct_ids + [target_ids[int(i)] for i in d256] + u_ids))
        qt_reservoir[qid] = reservoir
        qt_top256[qid] = [target_ids[int(i)] for i in d256]
        admission[qid] = {
            "direct": direct_ids,
            "direct_scores": [float(s_target[int(i)]) for i in d100],
            "evidence": evidence_ids,
            "first_hop": {
                "text": [(text_ids[int(i)], float(s_text[int(i)])) for i in t20],
                "image": [(image_ids[int(i)], float(s_image[int(i)])) for i in i20],
            },
            "U": u_ids,
            "C100": c100,
            "paths": paths,
        }
        if progress and n % progress == 0:
            print(f"raw {n}/{len(query_ids)}", flush=True)

    et_reservoir: dict[str, list[str]] = {}
    anchors = list(labels.epos) if et_anchors is None else list(et_anchors)
    for eid in utf8_sorted(anchors):
        if eid not in store.index:
            continue
        ez = z[store.index[eid]].to(torch_device)
        scores = z_target @ ez
        top = stable_topk(scores, hard_pool)
        et_reservoir[eid] = [target_ids[int(i)] for i in top]

    return {
        "legal": list(labels.legal),
        "targets": target_ids,
        "text_assets": text_ids,
        "image_assets": image_ids,
        "qt_reservoir": qt_reservoir,
        "qt_top256": qt_top256,
        "qe_reservoir": qe_reservoir,
        "et_reservoir": et_reservoir,
        "admission": admission,
        "params": {"direct": direct, "hard_pool": hard_pool, "first_hop": first_hop,
                   "targets_per_evidence": targets_per_evidence, "retained_paths": retained_paths,
                   "rrf_k": rrf_k},
    }


# ------------------------------------------------------------- edge lists ----


def _sample_random(pool: Sequence[str], exclude: set[str], n: int, namespace: str) -> list[str]:
    remaining = [x for x in pool if x not in exclude]
    if n >= len(remaining):
        return list(remaining)
    rng = local_rng(namespace)
    return rng.sample(remaining, n)


def edge_candidates(
    positives: Sequence[str],
    ignore: Sequence[str],
    reservoir: Sequence[str],
    legal: Sequence[str],
    *,
    query_id: str,
    relation: str,
    hard_n: int = 32,
    random_n: int = 32,
    namespace_phase: str = "edge",
) -> list[str]:
    """All positives + up to hard_n + up to random_n, positive/ignore protected."""
    p = utf8_sorted(set(positives))
    ig = set(ignore)
    pset = set(p)
    hard: list[str] = []
    for tid in reservoir:
        if tid in pset or tid in ig:
            continue
        hard.append(tid)
        if len(hard) >= hard_n:
            break
    pool = [x for x in legal if x not in pset and x not in ig]
    random = _sample_random(pool, set(hard), random_n, config.namespace(namespace_phase, query_id, relation))
    return list(dict.fromkeys([*p, *hard, *random]))


def build_edge_lists(
    labels: TrainLabels,
    raw: dict,
    *,
    hard_scores: dict[tuple[str, str], list[str]] | None = None,
    namespaces: str = "edge",
) -> dict[str, dict[str, list[str]]]:
    """Per-query edge items for the five relations (SPEC 3.4).

    ``hard_scores`` optionally replaces the raw hard order with a refreshed
    ranking (``{(query_id, relation): ordered_ids}``), as produced once by
    T_EDGE (SPEC 5.5).
    """
    edge: dict[str, dict[str, list[str]]] = {}
    for qid in sorted(labels.queries, key=lambda x: x.encode("utf-8")):
        entry = labels.queries[qid]
        groups: dict[str, list[str]] = {}
        reservoir = raw["qt_reservoir"][qid]
        if hard_scores is not None and (qid, "QT") in hard_scores:
            reservoir = hard_scores[(qid, "QT")]
        groups["QT"] = edge_candidates(entry["G"], [], reservoir, raw["legal"],
                                       query_id=qid, relation="QT", namespace_phase=namespaces)
        for modality in ("text", "image"):
            positives = entry["Qpos"][modality]
            if not positives:
                continue
            key = f"Q_{modality}"
            res = raw["qe_reservoir"][qid][modality]
            if hard_scores is not None and (qid, key) in hard_scores:
                res = hard_scores[(qid, key)]
            groups[key] = edge_candidates(positives, [], res, raw[f"{modality}_assets"],
                                          query_id=qid, relation=key, namespace_phase=namespaces)
        for tid, evidence in sorted(entry["W"].items(), key=lambda kv: kv[0].encode("utf-8")):
            for asset in evidence:
                positives = [t for t in labels.epos.get(asset, []) if t in set(raw["legal"])]
                if not positives:
                    continue
                key = f"E_{asset}"
                res = raw["et_reservoir"].get(asset, [])
                if hard_scores is not None and (qid, key) in hard_scores:
                    res = hard_scores[(qid, key)]
                groups[key] = edge_candidates(positives, [], res, raw["legal"],
                                              query_id=qid, relation=key, namespace_phase=namespaces)
        edge[qid] = groups
    return edge


def save_pickle(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(payload, fh, protocol=4)
    tmp.replace(path)


def load_pickle(path: Path) -> dict:
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def refresh_reservoirs(
    model,
    store: RawStore,
    raw: dict,
    labels: TrainLabels,
    *,
    device: str,
    batch: int = 32,
    progress: int = 0,
) -> dict[tuple[str, str], list[str]]:
    """One authorised hard refresh using the finished T_EDGE scorer (SPEC 5.5).

    Only targets inside each anchor's raw reservoir can be scored; the QT
    reservoir is the true union of raw Direct256 and raw U.
    """
    from .score import TeacherScorer  # local import to avoid a cycle

    scorer = TeacherScorer(model, store, device=device)
    out: dict[tuple[str, str], list[str]] = {}
    qids = sorted(labels.queries, key=lambda x: x.encode("utf-8"))
    for n, qid in enumerate(qids):
        entry = labels.queries[qid]
        pairs = [(("QT", raw["qt_reservoir"][qid]), entry["G"])]
        for modality in ("text", "image"):
            positives = entry["Qpos"][modality]
            if positives:
                pairs.append(((f"Q_{modality}", raw["qe_reservoir"][qid][modality]), positives))
        assets = sorted({a for evidence in entry["W"].values() for a in evidence}, key=lambda x: x.encode("utf-8"))
        for asset in assets:
            positives = [t for t in labels.epos.get(asset, []) if t in set(raw["legal"])]
            if positives and raw["et_reservoir"].get(asset):
                pairs.append(((f"E_{asset}", raw["et_reservoir"][asset]), positives))
        for key, reservoir in pairs:
            scores = scorer.pair_scores(qid, reservoir, batch=batch)
            order = sorted(range(len(reservoir)), key=lambda i: (-scores[i], reservoir[i].encode("utf-8")))
            out[(qid, key)] = [reservoir[i] for i in order]
        if progress and n % progress == 0:
            print(f"refresh {n}/{len(qids)}", flush=True)
    return out
