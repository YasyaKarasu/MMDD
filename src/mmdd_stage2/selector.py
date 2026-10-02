"""Column selector head and the recovery plan it schedules.

Head: a fresh two-layer MLP over the reader's [OPEN; CLOSE] column feature, trained with a
multi-positive softmax loss over the target's columns (gold = the hidden recoverable column).
Pairs are weighted so every training query contributes equally; train views alternate per epoch.
The epoch count is fixed in the config and the final epoch is used: the holdout split only
monitors training, it never chooses a checkpoint.

Plan: for every C30 target, ``pair_score = log_softmax_C30(stage1 table logit) +
log_softmax_columns(head logit)``. Pairs are taken in score order with at most ``per_table_cap``
columns per target until ``branch_budget`` pairs are chosen. Chosen pairs that share a column name
and the same evidence bag become one recovery view.
"""
from __future__ import annotations

import collections
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .catalog import Catalog
from .common import digest, iter_jsonl, read_json, write_json, write_jsonl
from .reader import feature_path
from .stage1 import load_stage1
from .values import norm


class Head(nn.Module):
    def __init__(self, input_dim: int, width: int, dropout: float) -> None:
        super().__init__()
        self.weight = nn.Sequential(nn.Linear(input_dim, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1))

    def forward(self, opened: torch.Tensor, closed: torch.Tensor) -> torch.Tensor:
        return self.weight(torch.cat([opened, closed], dim=-1)).squeeze(-1)


def column_loss(logits: torch.Tensor, columns: list[int], gold: list[int]) -> torch.Tensor:
    positive = [i for i, column in enumerate(columns) if column in gold]
    return torch.logsumexp(logits, 0) - torch.logsumexp(logits[positive], 0)


def load_features(run: Path, split: str, view: int, pair_id: str) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    with np.load(feature_path(run, split, view, pair_id)) as data:
        return torch.from_numpy(data["open"]), torch.from_numpy(data["close"]), data["columns"].tolist()


def train_head(config: dict[str, Any], run: Path) -> None:
    h, seed = config["head"], config["seed"]
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    jobs = list(iter_jsonl(run / "jobs" / "train.jsonl"))
    gold = read_json(run / "jobs" / "train_labels.json")
    fit = [j for j in jobs if j["partition"] == "fit"]
    holdout = [j for j in jobs if j["partition"] == "holdout"]
    features = {(view, j["pair_id"]): load_features(run, "train", view, j["pair_id"])
                for view in range(len(config["reader"]["column_permutation_seeds"])) for j in jobs}

    torch.manual_seed(seed)
    head = Head(h["input_dim"], h["hidden_dim"], h["dropout"])
    optimizer = torch.optim.AdamW(head.parameters(), lr=h["lr"], weight_decay=h["weight_decay"],
                                  betas=tuple(h["betas"]), eps=h["eps"])
    per_query = collections.Counter(j["query_id"] for j in fit)
    weight = {j["pair_id"]: len(fit) / (len(per_query) * per_query[j["query_id"]]) for j in fit}
    views = len(config["reader"]["column_permutation_seeds"])
    history = []
    for epoch in range(1, h["epochs"] + 1):
        head.train()
        view = (epoch - 1) % views
        order = list(range(len(fit)))
        random.Random(seed * 1000 + epoch).shuffle(order)
        losses = []
        for start in range(0, len(order), h["batch_pairs"]):
            optimizer.zero_grad(set_to_none=True)
            torch.manual_seed(seed * 1_000_000 + epoch * 10_000 + start // h["batch_pairs"])  # dropout masks
            terms = []
            for i in order[start:start + h["batch_pairs"]]:
                job = fit[i]
                opened, closed, columns = features[view, job["pair_id"]]
                loss = column_loss(head(opened, closed), columns, gold[job["pair_id"]])
                terms.append(loss * weight[job["pair_id"]])
                losses.append(float(loss.detach()))
            loss = torch.stack(terms).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), h["gradient_clip"])
            optimizer.step()
        record = {"epoch": epoch, "view": view, "loss": float(np.mean(losses))}
        if epoch == 1 or epoch % 5 == 0:
            record.update(holdout_metrics(head, holdout, features, gold))
        history.append(record)
        print(json.dumps(record), flush=True)
    (run / "head").mkdir(exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "epoch": h["epochs"]}, run / "head" / "head.pt")
    write_json(run / "head" / "history.json", {"fit_pairs": len(fit), "fit_queries": len(per_query),
                                               "holdout_pairs": len(holdout), "epochs": history})


def holdout_metrics(head: Head, jobs: list[dict], features: dict, gold: dict) -> dict[str, float]:
    """Query-averaged MRR / Hit@1 / Hit@3 of the first gold column (view 0)."""
    head.eval()
    per_query = collections.defaultdict(list)
    with torch.inference_mode():
        for job in jobs:
            opened, closed, columns = features[0, job["pair_id"]]
            scores = head(opened, closed).tolist()
            order = sorted(range(len(columns)), key=lambda i: (-scores[i], columns[i]))
            rank = min(position + 1 for position, i in enumerate(order) if columns[i] in gold[job["pair_id"]])
            per_query[job["query_id"]].append((1 / rank, float(rank <= 1), float(rank <= 3)))
    values = np.array([np.mean(v, axis=0) for v in per_query.values()])
    return {"holdout_MRR": float(values[:, 0].mean()), "holdout_Hit1": float(values[:, 1].mean()),
            "holdout_Hit3": float(values[:, 2].mean())}


def log_softmax(values: list[float]) -> list[float]:
    top = max(values)
    total = top + math.log(sum(math.exp(v - top) for v in values))
    return [v - total for v in values]


def build_plan(query_id: str, candidates: list[str], table_logits: dict[str, float],
               column_logits: dict[str, dict[int, float]], evidence: dict[str, list[str]],
               tables: dict[str, dict[str, Any]], *, branch_budget: int, per_table_cap: int
               ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(views, selected_pairs)`` for one query."""
    table_prior = dict(zip(candidates, log_softmax([table_logits[t] for t in candidates])))
    stage1_rank = {t: i + 1 for i, t in enumerate(candidates)}
    pairs = []
    for target in candidates:
        logits = column_logits[target]
        if not logits:
            continue
        names = {c["column_index"]: c["column_name"] for c in tables[target]["columns"]}
        ids = sorted(logits)
        for column, column_prior in zip(ids, log_softmax([logits[i] for i in ids])):
            pairs.append({"target_id": target, "column_id": column, "column_name": names[column],
                          "attribute": norm(names[column]), "stage1_rank": stage1_rank[target],
                          "pair_score": table_prior[target] + column_prior})
    pairs.sort(key=lambda p: (-p["pair_score"], p["stage1_rank"], p["column_id"]))
    per_table, selected = collections.Counter(), []
    for pair in pairs:
        if per_table[pair["target_id"]] >= per_table_cap:
            continue
        per_table[pair["target_id"]] += 1
        selected.append({**pair, "pair_rank": len(selected) + 1})
        if len(selected) == branch_budget:
            break
    views: list[dict[str, Any]] = []
    for pair in selected:
        bag = list(evidence.get(pair["target_id"], []))
        link = {k: pair[k] for k in ("target_id", "column_id", "column_name", "stage1_rank", "pair_rank")}
        view = next((v for v in views if v["column_name"] == pair["column_name"] and v["evidence_ids"] == bag), None)
        if view:
            view["donor_links"].append(link)
        else:
            views.append({"view_id": digest([query_id, pair["column_name"], bag]), "attribute": pair["attribute"],
                          "column_name": pair["column_name"], "evidence_ids": bag,
                          "priority_pair_rank": pair["pair_rank"], "donor_links": [link]})
    return views, selected


def make_plans(config: dict[str, Any], run: Path) -> None:
    """Score every dev/test C30 column with the trained head and write ``plans/<split>.jsonl``."""
    h = config["head"]
    head = Head(h["input_dim"], h["hidden_dim"], h["dropout"])
    head.load_state_dict(torch.load(run / "head" / "head.pt", map_location="cpu", weights_only=True)["state_dict"])
    head.eval()
    catalog = Catalog(run)
    for split in ("dev", "test"):
        stage1 = load_stage1(Path(config["paths"]["stage1_handoff"]), split, config["candidate_scope"],
                             config["output_depth"])
        by_query = collections.defaultdict(list)
        for job in iter_jsonl(run / "jobs" / f"{split}.jsonl"):
            by_query[job["query_id"]].append(job)
        plans = []
        with torch.inference_mode():
            for query_id, jobs in by_query.items():
                column_logits = {}
                for job in jobs:
                    opened, closed, columns = load_features(run, split, 0, job["pair_id"])
                    column_logits[job["target_id"]] = dict(zip(columns, head(opened, closed).tolist() if columns else []))
                record = stage1[query_id]
                tables = {t: catalog.get("target", t) for t in record["candidates"]}
                views, selected = build_plan(query_id, record["candidates"], record["table_logits"], column_logits,
                                             record["evidence"], tables, branch_budget=config["plan"]["branch_budget"],
                                             per_table_cap=config["plan"]["per_table_cap"])
                plans.append({"query_id": query_id, "views": views, "selected_pairs": selected,
                              "column_logits": {t: [[c, v] for c, v in d.items()] for t, d in column_logits.items()}})
        write_jsonl(run / "plans" / f"{split}.jsonl", plans)
        print(json.dumps({"split": split, "plans": len(plans),
                          "mean_pairs": float(np.mean([len(p["selected_pairs"]) for p in plans]))}), flush=True)
