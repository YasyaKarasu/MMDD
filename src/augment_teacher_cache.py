"""Extend an existing full-natural Teacher cache to an augmented manifest.

Only newly added candidate IDs are scored.  This preserves the original cache
values and avoids re-running tens of thousands of unchanged logical lists.
"""
from __future__ import annotations

import argparse, gzip, json
from pathlib import Path
import torch

from mmdd_stage1.artifacts import checkpoint_fingerprint
from mmdd_stage1.features import FeatureStore
from run_stage1_r21 import _teacher_cache_key, _score_id_pairs, _feature_paths, paths as r21_paths
from run_stage1_r22 import out as r22_out, read_rows, write_rows
from run_stage1_r22_f1 import _load_t0, r22_paths
from run_stage1_r19 import R19GlobalResidualTeacher


def _load_teacher(root: Path, stage: str, seed: int, device: torch.device):
    if stage == "T0":
        return _load_t0(root, seed, device)
    job = r22_out(root) / "fresh_lineage" / stage / f"seed{seed}"
    ck = sorted((job / "checkpoints").glob("step_*.pt"))[-1]
    from mmdd_stage1.checkpoints import load_checkpoint
    payload = load_checkpoint(ck)
    model = R19GlobalResidualTeacher(**payload["config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--stage", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    root, device = args.root, torch.device(args.device)
    base_manifest = r22_out(root) / "manifests" / "full_natural.jsonl"
    base_cache = r22_out(root) / "fresh_lineage" / args.stage / f"seed{args.seed}" / "teacher_soft_scores.jsonl.gz"
    base_rows, aug_rows = list(read_rows(base_manifest)), list(read_rows(args.manifest))
    if len(base_rows) != len(aug_rows):
        raise ValueError("augmented manifest must preserve row count")
    cache_rows = list(read_rows(base_cache))
    by_key = {_teacher_cache_key(str(r["query_id"]), "table->table", r["candidate_ids"]): r for r in cache_rows}
    store = FeatureStore.from_path(r22_paths(root)["features"], cache_size=4096,
                                   teacher_paths=_feature_paths(r21_paths(root)))
    model = _load_teacher(root, args.stage, args.seed, device)
    extra_pairs, extra_keys = [], []
    for b, a in zip(base_rows, aug_rows):
        if a.get("source_type") == "table" and a.get("destination_type") == "table":
            old = [str(x) for x in b["candidate_ids"]]
            new = [str(x) for x in a["candidate_ids"]]
            extras = [x for x in new if x not in set(old)]
            for cid in extras:
                extra_pairs.append((str(a["query_id"]), cid))
                extra_keys.append((str(a["query_id"]), cid))
    scores = _score_id_pairs(model, extra_pairs, store, device, batch_size=512) if extra_pairs else []
    extra_score = dict(zip(extra_keys, scores))
    out_rows = []
    for b, a in zip(base_rows, aug_rows):
        if a.get("source_type") != "table" or a.get("destination_type") != "table":
            continue
        old = [str(x) for x in b["candidate_ids"]]
        new = [str(x) for x in a["candidate_ids"]]
        rec = by_key[_teacher_cache_key(str(b["query_id"]), "table->table", old)]
        old_scores = {str(cid): float(s) for cid, s in zip(rec["candidate_ids"], rec["scores"])}
        values = []
        for cid in new:
            if cid in old_scores:
                values.append(old_scores[cid])
            else:
                values.append(extra_score[(str(a["query_id"]), cid)])
        out_rows.append({"query_id": str(a["query_id"]), "relation": "table->table",
                         "candidate_ids": new, "scores": values})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_rows(args.output, out_rows)
    print(json.dumps({"status": "complete", "rows": len(out_rows), "new_pairs": len(extra_pairs),
                      "output": str(args.output), "sha256": checkpoint_fingerprint(args.output)}))


if __name__ == "__main__":
    main()
