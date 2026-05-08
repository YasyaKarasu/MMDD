#!/usr/bin/env python
"""Train a single pairwise MLP teacher and score evidence paths."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from stage1_io import iter_jsonl, setup_logging, update_stage1_manifest, write_json, write_jsonl

TYPES = ["table_fragment", "text_asset", "image_asset"]
TYPE_TO_ID = {name: idx for idx, name in enumerate(TYPES)}


def load_embeddings(embedding_dir: Path) -> tuple[dict[str, np.ndarray], int]:
    vectors: dict[str, np.ndarray] = {}
    dim = 0
    for object_type in TYPES:
        npy = embedding_dir / f"{object_type}.npy"
        ids_path = embedding_dir / f"{object_type}_ids.json"
        if not npy.exists() or not ids_path.exists():
            continue
        arr = np.load(npy).astype("float32")
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        for idx, object_id in enumerate(ids):
            vectors[str(object_id)] = arr[idx]
        if arr.size:
            dim = int(arr.shape[1])
    return vectors, dim


class TeacherMLP(nn.Module):
    def __init__(self, dim: int, type_dim: int = 16, hidden: int = 512) -> None:
        super().__init__()
        self.type_emb = nn.Embedding(len(TYPES), type_dim)
        in_dim = dim * 4 + type_dim * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, za: torch.Tensor, zb: torch.Tensor, ta: torch.Tensor, tb: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([za, zb, za * zb, torch.abs(za - zb), self.type_emb(ta), self.type_emb(tb)], dim=1)
        return self.net(feat).squeeze(1)


class PairPathDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], vectors: dict[str, np.ndarray]) -> None:
        self.records = records
        self.vectors = vectors

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        return rec


def collate(batch: list[dict[str, Any]], vectors: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    return batch


def pair_score(model: TeacherMLP, vectors: dict[str, np.ndarray], a: str, ta: str, b: str, tb: str, device: torch.device) -> float | None:
    if a not in vectors or b not in vectors:
        return None
    with torch.no_grad():
        za = torch.tensor(vectors[a], dtype=torch.float32, device=device).unsqueeze(0)
        zb = torch.tensor(vectors[b], dtype=torch.float32, device=device).unsqueeze(0)
        logits = model(
            za,
            zb,
            torch.tensor([TYPE_TO_ID[ta]], device=device),
            torch.tensor([TYPE_TO_ID[tb]], device=device),
        )
        return float(torch.sigmoid(logits)[0].cpu())


def train_batch(model: TeacherMLP, batch: list[dict[str, Any]], vectors: dict[str, np.ndarray], device: torch.device, composition: str) -> torch.Tensor:
    losses = []
    bce_logits = nn.BCEWithLogitsLoss(reduction="none")
    bce_prob = nn.BCELoss(reduction="none")
    for rec in batch:
        if rec.get("sample_kind") == "pair":
            a, b = rec["object_id_a"], rec["object_id_b"]
            ta, tb = rec["object_type_a"], rec["object_type_b"]
            if a not in vectors or b not in vectors:
                continue
            za = torch.tensor(vectors[a], dtype=torch.float32, device=device).unsqueeze(0)
            zb = torch.tensor(vectors[b], dtype=torch.float32, device=device).unsqueeze(0)
            logits = model(za, zb, torch.tensor([TYPE_TO_ID[ta]], device=device), torch.tensor([TYPE_TO_ID[tb]], device=device))
            label = torch.tensor([float(rec["label"])], dtype=torch.float32, device=device)
            loss = bce_logits(logits, label) * float(rec.get("weight", 1.0))
            losses.append(loss.mean())
        else:
            q, m, t = rec["query_fragment_id"], rec["asset_id"], rec["target_fragment_id"]
            mt = rec["asset_object_type"]
            if q not in vectors or m not in vectors or t not in vectors:
                continue
            zq = torch.tensor(vectors[q], dtype=torch.float32, device=device).unsqueeze(0)
            zm = torch.tensor(vectors[m], dtype=torch.float32, device=device).unsqueeze(0)
            zt = torch.tensor(vectors[t], dtype=torch.float32, device=device).unsqueeze(0)
            tq = torch.tensor([TYPE_TO_ID["table_fragment"]], device=device)
            tm = torch.tensor([TYPE_TO_ID[mt]], device=device)
            tt = torch.tensor([TYPE_TO_ID["table_fragment"]], device=device)
            sq = torch.sigmoid(model(zq, zm, tq, tm))
            st = torch.sigmoid(model(zm, zt, tm, tt))
            path_score = torch.minimum(sq, st) if composition == "min" else sq * st
            label = torch.tensor([float(rec["label"])], dtype=torch.float32, device=device)
            loss = bce_prob(path_score.clamp(1e-6, 1 - 1e-6), label) * float(rec.get("weight", 1.0))
            losses.append(loss.mean())
    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(losses).mean()


def score_paths(stage1_dir: Path, model: TeacherMLP, vectors: dict[str, np.ndarray], device: torch.device, composition: str) -> list[dict[str, Any]]:
    path_file = stage1_dir / "hitl_pool.jsonl"
    if not path_file.exists():
        path_file = stage1_dir / "evidence_paths.jsonl"
    scores = []
    for path in iter_jsonl(path_file):
        mt = f"{path['asset_type']}_asset"
        q_asset = pair_score(model, vectors, path["query_fragment_id"], "table_fragment", path["asset_id"], mt, device)
        asset_t = pair_score(model, vectors, path["asset_id"], mt, path["target_fragment_id"], "table_fragment", device)
        if q_asset is None or asset_t is None:
            continue
        path_score = min(q_asset, asset_t) if composition == "min" else q_asset * asset_t
        scores.append(
            {
                "sample_kind": "path_score",
                "path_id": path["path_id"],
                "split": path.get("split"),
                "chain_id": path.get("chain_id"),
                "query_fragment_id": path["query_fragment_id"],
                "asset_id": path["asset_id"],
                "asset_object_type": mt,
                "target_fragment_id": path["target_fragment_id"],
                "score_Q_asset": q_asset,
                "score_asset_T": asset_t,
                "path_score": path_score,
            }
        )
    for qrel in iter_jsonl(stage1_dir / "qrels.jsonl"):
        score = pair_score(model, vectors, qrel["query_id"], "table_fragment", qrel["target_id"], "table_fragment", device)
        if score is not None:
            scores.append({"sample_kind": "pair_score", "query_id": qrel["query_id"], "target_id": qrel["target_id"], "score": score, **qrel})
    return scores


def run(args: argparse.Namespace) -> None:
    setup_logging()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    stage1_dir = Path(args.stage1_dir)
    vectors, dim = load_embeddings(Path(args.embedding_dir))
    records = [rec for rec in iter_jsonl(Path(args.train_pairs)) if rec.get("split") == "train"]
    if not records:
        records = list(iter_jsonl(Path(args.train_pairs)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TeacherMLP(dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = DataLoader(PairPathDataset(records, vectors), batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(b, vectors))
    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        steps = 0
        for batch in loader:
            opt.zero_grad()
            loss = train_batch(model, batch, vectors, device, args.path_composition)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
            steps += 1
        history.append({"epoch": epoch + 1, "loss": total / max(1, steps)})
        print(history[-1])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "dim": dim, "types": TYPES, "args": vars(args)}, out_dir / "teacher.pt")
    write_json(out_dir / "train_history.json", history)
    model.eval()
    scores = score_paths(stage1_dir, model, vectors, device, args.path_composition)
    count = write_jsonl(stage1_dir / "teacher_scores.jsonl", scores)
    update_stage1_manifest(stage1_dir, "teacher", {"output_dir": str(out_dir), "scores": count, "history": history, "args": vars(args)})
    print(json.dumps({"scores": count, "output_dir": str(out_dir)}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_dir", default="output_stage1_logic")
    parser.add_argument("--embedding_dir", default="output_stage1_logic/embeddings")
    parser.add_argument("--train_pairs", default="output_stage1_logic/train_pairs.jsonl")
    parser.add_argument("--output_dir", default="output_stage1_logic/teacher")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--path_composition", choices=["min", "product"], default="min")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
