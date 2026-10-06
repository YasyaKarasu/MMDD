"""Inspect the actual cached Teacher distributions used by R4/R5's inherited KD Student."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from audit_r5_artifacts import records


def audit(seed_root: Path, output: Path) -> None:
    torch.set_num_threads(4)
    cache = torch.load(seed_root / "teacher_logits_cache/scores.pt", map_location="cpu", weights_only=True)
    values = {"direct": [], "evidence": []}
    correlations = []
    for row in records(seed_root / "training_records/C2_SHARED.jsonl.gz"):
        entry = cache[row["query_id"]]
        targets, positive = row["targets"], set(row["positives"])
        bags = row.get("natural_bags", {})
        bag_indices = [i for i, t in enumerate(targets) if bags.get(t)]
        bag_targets = [targets[i] for i in bag_indices]
        for channel, candidates in (("direct", targets), ("evidence", bag_targets)):
            scores = entry[channel]
            if scores is None or scores.numel() < 2:
                continue
            indices = [i for i, t in enumerate(candidates) if t in positive]
            if not indices:
                continue
            p = torch.softmax(scores.double() / 10, dim=0)
            entropy = float(-(p * p.log()).sum())
            values[channel].append({"n": len(scores), "std": float(scores.std(unbiased=False)),
                "top1_probability_T10": float(p.max()), "gold_mass_T10": float(p[indices].sum()),
                "entropy_fraction_T10": entropy / np.log(len(scores)),
                "effective_support_fraction_T10": float(np.exp(entropy) / len(scores)),
                "top1_is_gold": int(int(scores.argmax()) in indices)})
        if len(bag_indices) > 1 and entry["evidence"] is not None:
            direct = entry["direct"][bag_indices].numpy()
            evidence = entry["evidence"][:len(bag_indices)].numpy()
            if direct.std() > 0 and evidence.std() > 0:
                correlations.append(float(np.corrcoef(direct, evidence)[0, 1]))
    summary = {k: {"lists": len(rs), **{field: float(np.mean([r[field] for r in rs])) for field in rs[0]}}
               for k, rs in values.items()}
    summary["original_bags_direct_evidence_correlation_mean"] = float(np.mean(correlations))
    summary["interpretation"] = "Training lists, not held-out retrieval; p_T=softmax(logit/10). Correlation is descriptive."
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.seed_root, args.output)
