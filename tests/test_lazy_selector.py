from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmdd_stage2.lazy_selector import select_lazy


def exhaustive(base, logits, columns):
    # Independent probability-space oracle, avoiding the pruning implementation.
    maximum = max(logits.values())
    denominator = sum(math.exp(logits[t] - maximum) for t in base)
    pairs = []
    for rank, t in enumerate(base, 1):
        if not columns[t]:
            continue
        cm = max(columns[t])
        cd = sum(math.exp(v - cm) for v in columns[t])
        for cid, v in enumerate(columns[t]):
            p = math.exp(logits[t] - maximum) / denominator * math.exp(v - cm) / cd
            pairs.append((p, rank, cid, t))
    used = dict.fromkeys(base, 0)
    result = []
    for p, rank, cid, t in sorted(pairs, key=lambda p: (-p[0], p[1], p[2])):
        if used[t] < 3:
            used[t] += 1
            result.append((t, cid))
        if len(result) == 10:
            break
    return result


def run(base, logits, columns):
    calls = []

    def score(t):
        calls.append(t)
        return {"eligible_column_ids": list(range(len(columns[t]))),
                "column_logits": [{"column_id": i, "logit": v} for i, v in enumerate(columns[t])]}

    result = select_lazy(base, logits, {t: list(range(len(columns[t]))) for t in base}, score)
    actual = [(p["target_id"], p["column_id"]) for p in result["selected_pairs"]]
    assert actual == exhaustive(base, logits, columns)
    assert result["certificate"]["normalization_candidates"] == base
    bound = result["certificate"]
    assert all(v < bound["threshold"] for v in bound["pruned_upper_bounds"].values())
    assert set(calls) == set(result["selector"])
    return result


def test_exact_lazy_selection_randomized_and_unsorted():
    rng = random.Random(13)
    for _ in range(200):
        base = [str(i) for i in range(50)]
        rng.shuffle(base)
        logits = {t: rng.uniform(-8, 8) for t in base}
        columns = {t: [rng.uniform(-5, 5) for _ in range(rng.randrange(12))] for t in base}
        run(base, logits, columns)


def test_lazy_selection_preserves_ties_and_max_three_cap():
    base = [str(i) for i in range(50)]
    result = run(base, dict.fromkeys(base, 0.0), {t: [0.0] * 8 for t in base})
    assert len(result["selector"]) == 50
    assert [p["target_id"] for p in result["selected_pairs"]] == ["0"] * 3 + ["1"] * 3 + ["2"] * 3 + ["3"]


def test_lazy_selection_prunes_with_proof_and_handles_fewer_than_ten_pairs():
    base = [str(i) for i in range(50)]
    result = run(base, {t: -float(i) for i, t in enumerate(base)}, {t: [0.0] * 3 for t in base})
    assert len(result["selector"]) < 10
    result = run(base, dict.fromkeys(base, 0.0), {t: [0.0] if int(t) < 4 else [] for t in base})
    assert len(result["selected_pairs"]) == 4
    assert len(result["selector"]) == 50


def test_rejects_nonfinite_scores_and_eligibility_drift():
    with pytest.raises(ValueError, match="finite"):
        select_lazy(["a"], {"a": float("nan")}, {"a": []}, lambda t: {})
    with pytest.raises(ValueError, match="eligibility"):
        select_lazy(["a"], {"a": 0.0}, {"a": [0]}, lambda t: {"column_logits": [], "eligible_column_ids": []})
