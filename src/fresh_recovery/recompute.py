"""Independent metric recomputation from saved rankings + GT (A29).

Uses only the saved ID rankings, original split GT, and reported results.
The calculation is intentionally separate from the production metrics module.
"""
from __future__ import annotations

import gzip
import json
from collections.abc import Sequence

import numpy as np

from .config import Paths
from .io import write_json


def _unique(gold: Sequence[str], ranking: Sequence[str]) -> tuple[set[str], list[str]]:
    positives = set(gold)
    if not positives:
        raise ValueError("empty GT must be explicitly handled")
    if len(ranking) != len(set(ranking)):
        raise ValueError("duplicate ranked target IDs")
    return positives, list(ranking)


def recall_from_ids(gold: Sequence[str], ranking: Sequence[str], k: int) -> float:
    positives, ordered = _unique(gold, ranking)
    return len(positives.intersection(ordered[:k])) / len(positives)


def coverage_from_ids(gold: Sequence[str], candidates: Sequence[str]) -> float:
    positives, pool = _unique(gold, candidates)
    return len(positives.intersection(pool)) / len(positives)


def oracle_from_ids(gold: Sequence[str], candidates: Sequence[str], k: int) -> float:
    positives, pool = _unique(gold, candidates)
    return min(k, len(positives.intersection(pool))) / len(positives)


def recompute(paths: Paths, *, seed: int, split: str) -> dict:
    out_dir = paths.work_dir / f"seed{seed}" / f"EVAL_{split.upper()}"
    from .compact import load_any

    rankings = load_any(out_dir, "rankings")
    with gzip.open(out_dir / "gt.json.gz", "rt") as handle:
        gt = json.load(handle)
    results = json.loads((out_dir / "RESULTS.json").read_text())
    groups = {"overall": list(gt)}
    for kind in ("implicit", "explicit", "mixed"):
        members = [q for q in gt if gt[q]["kind"] == kind]
        if members:
            groups[kind] = members
    max_error = 0.0
    rows = {}
    for system, ranking in rankings.items():
        rows[system] = {}
        for group, members in groups.items():
            if not members:
                continue
            for k in (10, 20, 30, 40, 50):
                value = float(np.mean([recall_from_ids(gt[q]["G"], ranking.get(q, []), k) for q in members]))
                reported = results["systems"][system][group][f"R{k}"]
                max_error = max(max_error, abs(value - reported))
                rows[system][f"{group}/R{k}"] = value
            # MatchedDirectM has rankings only; pool metrics exist for systems
            # whose reported candidate pool is exactly their ranked ID list.
            if "coverage" in results["systems"][system][group]:
                coverage = float(np.mean([coverage_from_ids(gt[q]["G"], ranking.get(q, [])) for q in members]))
                reported = results["systems"][system][group]["coverage"]
                max_error = max(max_error, abs(coverage - reported))
                rows[system][f"{group}/coverage"] = coverage
                for k in (10, 50):
                    oracle = float(np.mean([oracle_from_ids(gt[q]["G"], ranking.get(q, []), k) for q in members]))
                    reported = results["systems"][system][group][f"Oracle{k}"]
                    max_error = max(max_error, abs(oracle - reported))
                    rows[system][f"{group}/Oracle{k}"] = oracle
    report = {"split": split, "systems": len(rankings), "queries": len(gt), "max_abs_error_vs_RESULTS": max_error,
              "status": "PASS" if max_error < 1e-9 else "FAIL", "recomputed": rows}
    write_json(out_dir / "RECOMPUTED_METRICS.json", report)
    return {k: v for k, v in report.items() if k != "recomputed"}
