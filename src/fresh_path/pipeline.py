"""Stage orchestration: assemble this run's objects, then run one stage.

Each stage runs in its own process, loads only parents inside the same run
(SPEC 2.4) and writes its own checkpoint + receipt.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Sequence

import torch

from . import candidates, config, features, inputs, lineage, pca, prepare, score, train_student, train_teacher
from .candidates import load_pickle, save_pickle
from .models import FreshPathTeacher, PCAStudent, QTOnlyStudent
from .score import ObjectBank

TEACHER_KWARGS = dict(input_dim=4096, width=512, heads=8, layers=3, ffn=2048, text_slots=16, image_slots=24, dropout=0.1)
STUDENT_DIM = 1024


def _seed_int(*parts) -> int:
    import hashlib

    return int.from_bytes(hashlib.sha256(config.namespace(*parts).encode()).digest()[:4], "big")


class Runtime:
    def __init__(self, paths: config.Paths, protocol: dict, *, lru_bytes: int = 8 * 2**30) -> None:
        self.paths = paths
        self.protocol = protocol
        self.labels = prepare.load_labels(paths.work_dir / "labels")
        self.z = candidates.load_z(paths.work_dir / "features" / "z")
        self.content = features.ContentStore(paths.work_dir / "features" / "content", lru_bytes=lru_bytes)
        # GPU token storage is a pure cache.  Allow a stage process to lower
        # its capacity when a large differentiable path graph needs the VRAM;
        # the cached tensors and their ordering never affect the computation.
        gpu_token_bytes = int(os.environ.get("MMDD_GPU_TOKEN_BYTES", str(2**30)))
        self.bank = ObjectBank(self.z, self.content, lru_bytes=lru_bytes,
                               gpu_token_bytes=gpu_token_bytes)
        self._raw = None
        self._graph = None
        self._view = None

    @property
    def view(self):
        if self._view is None:
            self._view = inputs.load_dataset(self.paths.dataset_root)
        return self._view

    def view_query_split(self) -> dict[str, str]:
        return dict(self.view.query_split)

    def dev_query_kinds(self, dataset_root: Path, queries: Sequence[str]) -> dict[str, str]:
        kinds = inputs.split_query_kinds(Path(dataset_root), "dev")
        return {q: kinds.get(q, "unknown") for q in queries}

    def dev_probe_pairs(self, dataset_root: Path, queries: Sequence[str], raw_dev: dict, *, split: str = "dev") -> list:
        witness, modality = inputs.load_split_witness(Path(dataset_root), split)
        by_query: dict[str, dict[str, set[str]]] = {}
        for (q, t), assets in witness.items():
            by_query.setdefault(q, {})[t] = assets
        pairs = []
        for q in queries:
            entry = by_query.get(q)
            if not entry:
                continue
            seen: set[str] = set()
            for modality_name in ("text", "image"):
                for eid, _ in raw_dev["admission"][q].get("first_hop", {}).get(modality_name, []):
                    if eid in seen:
                        continue
                    if any(eid in assets for assets in entry.values()):
                        seen.add(eid)
                        positive = [t for t, assets in entry.items() if eid in assets]
                        pairs.append((q, eid, positive))
        return pairs


    @property
    def raw(self) -> dict:
        if self._raw is None:
            self._raw = load_pickle(self.paths.work_dir / "raw" / "raw_train.pkl")
        return self._raw

    @property
    def post_edge(self) -> dict:
        if self._graph is None:
            self._graph = load_pickle(self.paths.work_dir / "raw" / "post_edge.pkl")
        return self._graph

    def make_teacher(self, seed: int) -> FreshPathTeacher:
        torch.manual_seed(_seed_int("init", "teacher", seed))
        return FreshPathTeacher(**TEACHER_KWARGS).float()

    def basis(self, seed: int) -> torch.Tensor:
        payload = torch.load(self.paths.work_dir / "pca" / "basis.pt", map_location="cpu", weights_only=False)
        return payload["basis"]

    def make_student(self, seed: int, *, adapter: bool = True) -> PCAStudent:
        return PCAStudent(self.basis(seed), adapter=adapter)

    def make_qt_student(self, seed: int) -> QTOnlyStudent:
        return QTOnlyStudent(self.basis(seed))

    def stage_dir(self, seed: int, stage: str) -> Path:
        return self.paths.stage_dir(seed, stage)

    def load_teacher(self, seed: int, stage: str, device: str):
        model = self.make_teacher(seed)
        payload = lineage.load_checkpoint(self.stage_dir(seed, stage) / "checkpoint.pt")
        model.load_state_dict(payload["state_dict"])
        return model.to(device).eval()

    def load_student(self, seed: int, stage: str, device: str, *, adapter: bool = True) -> PCAStudent:
        model = self.make_student(seed, adapter=adapter)
        payload = lineage.load_checkpoint(self.stage_dir(seed, stage) / "checkpoint.pt")
        state = {k: v for k, v in payload["state_dict"].items()
                 if adapter or not k.startswith("adapter.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(f"{stage}: state mismatch missing={missing} unexpected={unexpected}")
        return model.to(device)


# ------------------------------------------------------------- data helpers ---


def edge_positives(labels, raw) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    legal = set(raw["legal"])
    for qid, entry in labels.queries.items():
        groups: dict[str, list[str]] = {"QT": list(entry["G"])}
        for modality in ("text", "image"):
            groups[f"Q_{modality}"] = list(entry["Qpos"][modality])
        for asset in sorted({a for ev in entry["W"].values() for a in ev}, key=lambda x: x.encode("utf-8")):
            groups[f"E_{asset}"] = [t for t in labels.epos.get(asset, []) if t in legal]
        out[qid] = groups
    return out


def build_initial_edges(rt: Runtime) -> dict:
    edge = candidates.build_edge_lists(rt.labels, rt.raw, hard_scores=None)
    payload = {"edge": edge, "positives": edge_positives(rt.labels, rt.raw)}
    save_pickle(rt.paths.work_dir / "raw" / "edge_initial.pkl", payload)
    return payload


def run_post_edge(rt: Runtime, *, seed: int, device: str, log=print) -> dict:
    model = rt.load_teacher(seed, "T_EDGE", device)
    refresh = train_teacher.refreshed_hard(model, rt.bank, rt.labels, rt.raw, device=device, log=log)
    graph = train_teacher.build_train_graph(model, rt.bank, rt.labels, rt.raw, device=device, log=log)
    edge = candidates.build_edge_lists(rt.labels, rt.raw, hard_scores=refresh)
    conditional_registry = train_teacher.build_conditional_registry(rt.labels, rt.raw)
    registry_epochs = max(
        rt.protocol["teacher"]["path_epochs"],
        rt.protocol["student"]["C2_epochs"],
        rt.protocol["controls"]["native"]["epochs"],
    )
    anchor_registry = train_teacher.build_anchor_registry(rt.labels, registry_epochs)
    payload = {
        "graph": graph,
        "hard_refresh": {f"{q}|{k}": v[:96] for (q, k), v in refresh.items()},
        "edge": edge,
        "positives": edge_positives(rt.labels, rt.raw),
        "conditional_registry": conditional_registry,
        "anchor_registry": anchor_registry,
    }
    save_pickle(rt.paths.work_dir / "raw" / "post_edge.pkl", payload)
    changed = sum(1 for (q, k), v in refresh.items() if v[:32] != rt.raw["qt_reservoir"].get(q, [])[:32])
    log(json.dumps({"event": "post_edge", "relations_refreshed": len(refresh), "graph_queries": len(graph),
                    "conditional_lists": sum(len(v) for v in conditional_registry.values()),
                    "anchor_epochs": len(anchor_registry),
                    "qt_hard_members_changed": changed}))
    return payload
