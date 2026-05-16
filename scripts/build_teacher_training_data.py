#!/usr/bin/env python
"""Build pair and path samples for the pairwise teacher."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from stage1_io import iter_jsonl, setup_logging, stable_hash, update_stage1_manifest, write_jsonl


def bool_arg(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y"}


def load_paths(stage1_dir: Path) -> list[dict[str, Any]]:
    path = stage1_dir / "hitl_pool.jsonl"
    paths = {rec["path_id"]: rec for rec in iter_jsonl(path)} if path.exists() else {}
    human = stage1_dir / "human_labeled_paths.jsonl"
    if human.exists():
        for rec in iter_jsonl(human):
            paths[rec["path_id"]] = rec
    return list(paths.values())


def path_label(path: dict[str, Any]) -> tuple[float, float, str] | None:
    if path.get("human_label") is not None:
        label = int(path["human_label"])
        if label == 2:
            return 1.0, 1.0, "human"
        if label == 1:
            return 0.6, 0.8, "human"
        return 0.0, 1.0, "human"
    weak = path.get("weak_label")
    if weak == "weak_direct":
        return 0.8, 0.4, "weak"
    if weak == "weak_indirect":
        return 0.5, 0.3, "weak"
    if weak == "weak_negative":
        return 0.0, 0.4, "weak"
    return None


def int_set(values: list[Any] | None) -> set[int]:
    cols: set[int] = set()
    for value in values or []:
        try:
            cols.add(int(value))
        except (TypeError, ValueError):
            continue
    return cols


def query_bridge_columns(fragment: dict[str, Any]) -> set[int]:
    bridge_cols: set[int] = set()
    for key in ("visible_bridge_col", "hidden_bridge_col"):
        value = fragment.get(key)
        if value is None:
            continue
        try:
            bridge_cols.add(int(value))
        except (TypeError, ValueError):
            continue
    return bridge_cols


def add_pair_sample(
    samples: list[dict[str, Any]],
    emitted: set[tuple[str, str, str]],
    *,
    sample_id: str,
    qid: str,
    tid: str,
    label: float,
    weight: float,
    split: str | None,
    chain_id: str | None,
    label_source: str,
    reason: str | None,
) -> None:
    key = (qid, tid, label_source)
    if key in emitted:
        return
    emitted.add(key)
    samples.append(
        {
            "sample_id": sample_id,
            "sample_kind": "pair",
            "object_id_a": qid,
            "object_type_a": "table_fragment",
            "object_id_b": tid,
            "object_type_b": "table_fragment",
            "label": label,
            "weight": weight,
            "split": split,
            "chain_id": chain_id,
            "label_source": label_source,
            "reason": reason,
        }
    )


def sample_negative_targets(
    pool: list[dict[str, Any]],
    positive_pairs: set[tuple[Any, Any]],
    *,
    qid: str,
    source_table_id: Any,
    chain_id: Any,
    rng: random.Random,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0 or not pool:
        return []
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def valid(candidate: dict[str, Any]) -> bool:
        target_id = candidate.get("fragment_id")
        return (
            target_id not in selected_ids
            and candidate.get("source_table_id") != source_table_id
            and candidate.get("chain_id") != chain_id
            and (qid, target_id) not in positive_pairs
        )

    attempts = min(len(pool), max(100, limit * 50))
    for _ in range(attempts):
        candidate = rng.choice(pool)
        if not valid(candidate):
            continue
        selected.append(candidate)
        selected_ids.add(candidate["fragment_id"])
        if len(selected) >= limit:
            return selected

    for candidate in pool:
        if not valid(candidate):
            continue
        selected.append(candidate)
        selected_ids.add(candidate["fragment_id"])
        if len(selected) >= limit:
            break
    return selected


def run(args: argparse.Namespace) -> None:
    setup_logging()
    rng = random.Random(args.seed)
    stage1_dir = Path(args.stage1_dir)
    fragments = {rec["fragment_id"]: rec for rec in iter_jsonl(stage1_dir / "logic_fragments.jsonl")}
    targets_by_split: dict[str, list[dict[str, Any]]] = {}
    targets_by_source_split: dict[tuple[str | None, str], list[dict[str, Any]]] = {}
    all_targets: list[dict[str, Any]] = []
    for frag in fragments.values():
        if frag.get("role") == "right_target":
            all_targets.append(frag)
            targets_by_split.setdefault(frag.get("split", "unknown"), []).append(frag)
            source_split = (frag.get("source_table_id"), frag.get("split", "unknown"))
            targets_by_source_split.setdefault(source_split, []).append(frag)
    samples: list[dict[str, Any]] = []
    logic_pairs = list(iter_jsonl(stage1_dir / "logic_pairs.jsonl"))
    positive_pairs = {(pair.get("query_fragment_id"), pair.get("target_fragment_id")) for pair in logic_pairs}
    emitted_pairs: set[tuple[str, str, str]] = set()
    for pair in logic_pairs:
        q = fragments.get(pair["query_fragment_id"])
        t = fragments.get(pair["target_fragment_id"])
        if not q or not t:
            continue
        if getattr(args, "table_only", False) and q.get("role") != "left_visible":
            continue
        weight = float(pair.get("weight", 1.0))
        label_source = "self_supervised" if q.get("role") == "left_visible" else "weak"
        if q.get("role") == "left_hidden":
            weight = min(weight, 0.4)
        add_pair_sample(
            samples,
            emitted_pairs,
            sample_id=f"train_{stable_hash(pair['pair_id'])}",
            qid=q["fragment_id"],
            tid=t["fragment_id"],
            label=float(pair["label"]),
            weight=weight,
            split=pair.get("split"),
            chain_id=pair.get("chain_id"),
            label_source=label_source,
            reason=pair.get("reason"),
        )
        negative_pool = (
            targets_by_split.get(pair.get("split", "unknown"))
            or targets_by_split.get("corpus")
            or all_targets
        )
        for neg in sample_negative_targets(
            negative_pool,
            positive_pairs,
            qid=q["fragment_id"],
            source_table_id=pair.get("source_table_id"),
            chain_id=pair.get("chain_id"),
            rng=rng,
            limit=1,
        ):
            add_pair_sample(
                samples,
                emitted_pairs,
                sample_id=f"train_neg_{stable_hash(q['fragment_id'], neg['fragment_id'])}",
                qid=q["fragment_id"],
                tid=neg["fragment_id"],
                label=0.0,
                weight=1.0,
                split=pair.get("split"),
                chain_id=pair.get("chain_id"),
                label_source="negative",
                reason="same_split_different_source_chain_target",
            )
        bridge_cols = query_bridge_columns(q)
        if bridge_cols:
            hard_negatives = [
                cand
                for cand in targets_by_source_split.get((pair.get("source_table_id"), pair.get("split", "unknown")), [])
                if (q["fragment_id"], cand["fragment_id"]) not in positive_pairs
                and not bridge_cols.intersection(int_set(cand.get("source_column_indices")))
            ]
            rng.shuffle(hard_negatives)
            for neg in hard_negatives[: int(getattr(args, "hard_negatives_per_positive", 1))]:
                add_pair_sample(
                    samples,
                    emitted_pairs,
                    sample_id=f"train_hard_neg_{stable_hash(q['fragment_id'], neg['fragment_id'])}",
                    qid=q["fragment_id"],
                    tid=neg["fragment_id"],
                    label=0.0,
                    weight=1.0,
                    split=pair.get("split"),
                    chain_id=pair.get("chain_id"),
                    label_source="hard_negative",
                    reason="same_source_non_joinable_target",
                )

    if not getattr(args, "table_only", False):
        for path in load_paths(stage1_dir):
            mapped = path_label(path)
            if mapped is None:
                continue
            label, weight, source = mapped
            samples.append(
                {
                    "sample_id": f"train_path_{stable_hash(path['path_id'], source)}",
                    "sample_kind": "path",
                    "query_fragment_id": path["query_fragment_id"],
                    "query_object_type": "table_fragment",
                    "asset_id": path["asset_id"],
                    "asset_object_type": f"{path['asset_type']}_asset",
                    "target_fragment_id": path["target_fragment_id"],
                    "target_object_type": "table_fragment",
                    "label": label,
                    "weight": weight,
                    "split": path.get("split"),
                    "chain_id": path.get("chain_id"),
                    "path_id": path["path_id"],
                    "label_source": source,
                    "reason": path.get("reason"),
                }
            )

    if not getattr(args, "table_only", False) and bool_arg(str(args.include_pseudo_labels)) and (stage1_dir / "teacher_scores.jsonl").exists():
        for score in iter_jsonl(stage1_dir / "teacher_scores.jsonl"):
            path_score = score.get("path_score")
            if score.get("path_id") is None or path_score is None:
                continue
            path_score = float(path_score)
            if path_score > args.pseudo_pos_threshold:
                label = 1.0
            elif path_score < args.pseudo_neg_threshold:
                label = 0.0
            else:
                continue
            samples.append(
                {
                    "sample_id": f"train_pseudo_{stable_hash(score['path_id'], path_score)}",
                    "sample_kind": "path",
                    "query_fragment_id": score["query_fragment_id"],
                    "query_object_type": "table_fragment",
                    "asset_id": score["asset_id"],
                    "asset_object_type": score["asset_object_type"],
                    "target_fragment_id": score["target_fragment_id"],
                    "target_object_type": "table_fragment",
                    "label": label,
                    "weight": 0.2,
                    "split": score.get("split"),
                    "chain_id": score.get("chain_id"),
                    "path_id": score["path_id"],
                    "label_source": "pseudo",
                    "reason": "high_confidence_teacher_path_score",
                }
            )

    out = Path(args.output)
    count = write_jsonl(out, samples)
    update_stage1_manifest(stage1_dir, "teacher_training_data", {"output": str(out), "records": count, "args": vars(args)})
    print(json.dumps({"records": count, "output": str(out)}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_dir", default="output_stage1_logic")
    parser.add_argument("--output", default="output_stage1_logic/train_pairs.jsonl")
    parser.add_argument("--include_pseudo_labels", default="false")
    parser.add_argument("--pseudo_pos_threshold", type=float, default=0.9)
    parser.add_argument("--pseudo_neg_threshold", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--hard_negatives_per_positive", type=int, default=1)
    parser.add_argument("--table_only", action="store_true", help="Only emit table-table pair samples; skip path, HITL, and pseudo-label samples.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
