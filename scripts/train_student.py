#!/usr/bin/env python
"""Distill teacher scores into type projections and relation matrices."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from stage1_io import iter_jsonl, update_stage1_manifest, write_json, write_jsonl

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only in minimal envs.
    tqdm = None  # type: ignore[assignment]

TYPES = ["table_fragment", "text_asset", "image_asset"]
TYPE_TO_ID = {name: idx for idx, name in enumerate(TYPES)}


def progress_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "progress", True))


def load_embeddings(embedding_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, str], int, dict[str, list[str]]]:
    vectors: dict[str, np.ndarray] = {}
    types: dict[str, str] = {}
    ids_by_type: dict[str, list[str]] = {}
    dim = 0
    for object_type in TYPES:
        npy = embedding_dir / f"{object_type}.npy"
        ids_path = embedding_dir / f"{object_type}_ids.json"
        if not npy.exists():
            continue
        arr = np.load(npy).astype("float32")
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        ids_by_type[object_type] = ids
        for i, oid in enumerate(ids):
            vectors[oid] = arr[i]
            types[oid] = object_type
        if arr.size:
            dim = int(arr.shape[1])
    return vectors, types, dim, ids_by_type


class Student(nn.Module):
    def __init__(self, in_dim: int, student_dim: int) -> None:
        super().__init__()
        self.proj = nn.ModuleDict({t: nn.Linear(in_dim, student_dim, bias=False) for t in TYPES})
        self.rel = nn.ParameterDict(
            {f"{a}__{b}": nn.Parameter(torch.eye(student_dim) + 0.01 * torch.randn(student_dim, student_dim)) for a in TYPES for b in TYPES}
        )

    def project(self, z: torch.Tensor, object_type: str) -> torch.Tensor:
        u = self.proj[object_type](z)
        return torch.nn.functional.normalize(u, p=2, dim=-1)

    def score(self, za: torch.Tensor, ta: str, zb: torch.Tensor, tb: str) -> torch.Tensor:
        ua = self.project(za, ta)
        ub = self.project(zb, tb)
        r = self.rel[f"{ta}__{tb}"]
        return torch.sum((ua @ r) * ub, dim=1)

    def relation_query(self, za: torch.Tensor, ta: str, tb: str) -> torch.Tensor:
        ua = self.project(za, ta)
        query = ua @ self.rel[f"{ta}__{tb}"]
        return torch.nn.functional.normalize(query, p=2, dim=-1)


def build_distill_records(stage1_dir: Path, teacher_scores: Path) -> list[dict[str, Any]]:
    records = []
    for score in iter_jsonl(teacher_scores):
        if score.get("sample_kind") == "pair_score":
            records.append(
                {
                    "kind": "pair",
                    "a": score["query_id"],
                    "ta": "table_fragment",
                    "b": score["target_id"],
                    "tb": "table_fragment",
                    "target": float(score["score"]),
                    "group_id": f"pair:{score['query_id']}:table_fragment",
                }
            )
        elif score.get("sample_kind") == "path_score":
            records.append(
                {
                    "kind": "pair",
                    "a": score["query_fragment_id"],
                    "ta": "table_fragment",
                    "b": score["asset_id"],
                    "tb": score["asset_object_type"],
                    "target": float(score["score_Q_asset"]),
                    "group_id": f"pair:{score['query_fragment_id']}:{score['asset_object_type']}",
                }
            )
            records.append(
                {
                    "kind": "pair",
                    "a": score["asset_id"],
                    "ta": score["asset_object_type"],
                    "b": score["target_fragment_id"],
                    "tb": "table_fragment",
                    "target": float(score["score_asset_T"]),
                    "group_id": f"pair:{score['asset_id']}:table_fragment",
                }
            )
            records.append({"kind": "path", **score, "target": float(score["path_score"]), "group_id": f"path:{score['query_fragment_id']}:table_fragment"})
    human = stage1_dir / "human_labeled_paths.jsonl"
    if human.exists():
        for rec in iter_jsonl(human):
            if rec.get("human_label") == 2:
                target = 1.0
            elif rec.get("human_label") == 1:
                target = 0.6
            else:
                target = 0.0
            records.append(
                {
                    "kind": "path",
                    "query_fragment_id": rec["query_fragment_id"],
                    "asset_id": rec["asset_id"],
                    "asset_object_type": f"{rec['asset_type']}_asset",
                    "target_fragment_id": rec["target_fragment_id"],
                    "target": target,
                    "group_id": f"path:{rec['query_fragment_id']}:table_fragment",
                }
            )
    return records


def record_score(model: Student, rec: dict[str, Any], vectors: dict[str, np.ndarray], device: torch.device) -> torch.Tensor | None:
    if rec["kind"] == "pair":
        if rec["a"] not in vectors or rec["b"] not in vectors:
            return None
        za = torch.tensor(vectors[rec["a"]], dtype=torch.float32, device=device).unsqueeze(0)
        zb = torch.tensor(vectors[rec["b"]], dtype=torch.float32, device=device).unsqueeze(0)
        return model.score(za, rec["ta"], zb, rec["tb"]).squeeze(0)
    q, m, t = rec["query_fragment_id"], rec["asset_id"], rec["target_fragment_id"]
    mt = rec["asset_object_type"]
    if q not in vectors or m not in vectors or t not in vectors:
        return None
    zq = torch.tensor(vectors[q], dtype=torch.float32, device=device).unsqueeze(0)
    zm = torch.tensor(vectors[m], dtype=torch.float32, device=device).unsqueeze(0)
    zt = torch.tensor(vectors[t], dtype=torch.float32, device=device).unsqueeze(0)
    q_m = model.score(zq, "table_fragment", zm, mt).squeeze(0)
    m_t = model.score(zm, mt, zt, "table_fragment").squeeze(0)
    return torch.minimum(q_m, m_t)


def build_ranking_groups(records: list[dict[str, Any]], vectors: dict[str, np.ndarray], min_group_size: int = 2) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if rec.get("kind") == "pair":
            if rec.get("a") not in vectors or rec.get("b") not in vectors:
                continue
        else:
            if rec.get("query_fragment_id") not in vectors or rec.get("asset_id") not in vectors or rec.get("target_fragment_id") not in vectors:
                continue
        buckets[str(rec.get("group_id") or "ungrouped")].append(rec)
    groups = []
    for group_id, items in buckets.items():
        targets = {round(float(item.get("target", 0.0)), 6) for item in items}
        if len(items) >= min_group_size and len(targets) > 1:
            groups.append({"group_id": group_id, "items": items})
    return groups


def listwise_group_loss(
    model: Student,
    group: dict[str, Any],
    vectors: dict[str, np.ndarray],
    device: torch.device,
    temperature: float,
) -> torch.Tensor | None:
    preds = []
    targets = []
    for rec in group["items"]:
        score = record_score(model, rec, vectors, device)
        if score is None:
            continue
        preds.append(score)
        targets.append(float(rec["target"]))
    if len(preds) < 2:
        return None
    pred_tensor = torch.stack(preds) / max(1e-6, temperature)
    target_tensor = torch.tensor(targets, dtype=torch.float32, device=device) / max(1e-6, temperature)
    teacher_dist = torch.softmax(target_tensor, dim=0)
    student_log_dist = torch.log_softmax(pred_tensor, dim=0)
    return -(teacher_dist * student_log_dist).sum()


def pairwise_group_loss(
    model: Student,
    group: dict[str, Any],
    vectors: dict[str, np.ndarray],
    device: torch.device,
    max_pairs: int,
    min_delta: float,
) -> torch.Tensor | None:
    preds = []
    targets = []
    for rec in group["items"]:
        score = record_score(model, rec, vectors, device)
        if score is None:
            continue
        preds.append(score)
        targets.append(float(rec["target"]))
    if len(preds) < 2:
        return None
    candidate_pairs = [(i, j) for i in range(len(preds)) for j in range(i + 1, len(preds)) if abs(targets[i] - targets[j]) >= min_delta]
    if not candidate_pairs:
        return None
    if max_pairs > 0 and len(candidate_pairs) > max_pairs:
        random.shuffle(candidate_pairs)
        candidate_pairs = candidate_pairs[:max_pairs]
    losses = []
    for i, j in candidate_pairs:
        sign = 1.0 if targets[i] > targets[j] else -1.0
        diff = preds[i] - preds[j]
        weight = abs(targets[i] - targets[j])
        losses.append(torch.nn.functional.softplus(-sign * diff) * weight)
    return torch.stack(losses).mean()


def train_loss(model: Student, batch: list[dict[str, Any]], vectors: dict[str, np.ndarray], device: torch.device, args: argparse.Namespace) -> torch.Tensor:
    losses = []
    for group in batch:
        if args.distill_loss == "listwise":
            loss = listwise_group_loss(model, group, vectors, device, args.ranking_temperature)
        else:
            loss = pairwise_group_loss(model, group, vectors, device, args.max_pairs_per_group, args.pairwise_min_delta)
        if loss is not None:
            losses.append(loss)
    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(losses).mean()


def export_index_embeddings(model: Student, vectors: dict[str, np.ndarray], ids_by_type: dict[str, list[str]], out_dir: Path, device: torch.device) -> None:
    index_dir = out_dir / "index_embeddings"
    index_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for object_type, ids in ids_by_type.items():
        projected = []
        with torch.no_grad():
            for oid in ids:
                z = torch.tensor(vectors[oid], dtype=torch.float32, device=device).unsqueeze(0)
                projected.append(model.project(z, object_type).cpu().numpy()[0])
        arr = np.vstack(projected).astype("float32") if projected else np.zeros((0, 0), dtype="float32")
        np.save(index_dir / f"{object_type}.npy", arr)
        write_json(index_dir / f"{object_type}_ids.json", ids)


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    stage1_dir = Path(args.stage1_dir)
    vectors, _, in_dim, ids_by_type = load_embeddings(Path(args.embedding_dir))
    records = build_distill_records(stage1_dir, Path(args.teacher_scores))
    write_jsonl(stage1_dir / "student_train_pairs.jsonl", records)
    groups = build_ranking_groups(records, vectors)
    write_jsonl(stage1_dir / "student_train_groups.jsonl", [{"group_id": group["group_id"], "records": len(group["items"])} for group in groups])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Student(in_dim, args.student_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = DataLoader(groups, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x)
    history = []
    for epoch in range(args.epochs):
        total = 0.0
        steps = 0
        batches = loader
        if tqdm is not None and progress_enabled(args):
            batches = tqdm(
                loader,
                desc=f"Student distill epoch {epoch + 1}/{args.epochs}",
                total=len(loader),
                unit="batch",
            )
        for batch in batches:
            opt.zero_grad()
            loss = train_loss(model, batch, vectors, device, args)
            loss.backward()
            opt.step()
            loss_value = float(loss.detach().cpu())
            total += loss_value
            steps += 1
            if tqdm is not None and progress_enabled(args):
                batches.set_postfix(loss=f"{total / max(1, steps):.4f}")  # type: ignore[attr-defined]
        history.append({"epoch": epoch + 1, "loss": total / max(1, steps)})
        print(history[-1])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "in_dim": in_dim, "student_dim": args.student_dim, "types": TYPES, "args": vars(args)}, out_dir / "student.pt")
    write_json(out_dir / "train_history.json", history)
    export_index_embeddings(model, vectors, ids_by_type, out_dir, device)
    update_stage1_manifest(stage1_dir, "student", {"output_dir": str(out_dir), "records": len(records), "groups": len(groups), "history": history, "args": vars(args)})
    print(json.dumps({"records": len(records), "groups": len(groups), "output_dir": str(out_dir)}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_dir", default="output_stage1_logic")
    parser.add_argument("--embedding_dir", default="output_stage1_logic/embeddings")
    parser.add_argument("--teacher_scores", default="output_stage1_logic/teacher_scores.jsonl")
    parser.add_argument("--output_dir", default="output_stage1_logic/student")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--student_dim", type=int, default=128)
    parser.add_argument("--distill_loss", choices=["pairwise", "listwise"], default="pairwise")
    parser.add_argument("--ranking_temperature", type=float, default=1.0)
    parser.add_argument("--max_pairs_per_group", type=int, default=2048)
    parser.add_argument("--pairwise_min_delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--no_progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
