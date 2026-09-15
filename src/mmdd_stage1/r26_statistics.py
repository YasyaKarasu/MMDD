"""Preregistered R26 budget gates and source-cluster paired statistics."""
from __future__ import annotations

import numpy as np


def loss_extension_gate(metrics: dict[str, dict]) -> dict:
    comparisons = []
    for seed in (13,29):
        new = metrics.get(f"R26-O-SUP/seed{seed}/step178")
        old = metrics.get(f"R25-SPLIT-SUP/seed{seed}/step178")
        b13 = metrics.get("B13")
        order_delta = None if new is None or old is None else {
            "E_addition": new["overall"]["admission"]["EO_ANN"]-old["overall"]["admission"]["EO_ANN"],
            "U_R10": new["overall"]["QT_OVER_U"]["recall@10"]-old["overall"]["QT_OVER_U"]["recall@10"],
            "implicit_U_R10": new["implicit"]["QT_OVER_U"]["recall@10"]-old["implicit"]["QT_OVER_U"]["recall@10"]}
        baseline_delta = None if new is None or b13 is None else {
            "U_raw":new["overall"]["U"]["raw_recall"]-b13["overall"]["U"]["raw_recall"],
            "implicit_E_addition":new["implicit"]["admission"]["EO_ANN"]-b13["implicit"]["admission"]["EO_ANN"],
            "implicit_U_R10":new["implicit"]["QT_OVER_U"]["recall@10"]-b13["implicit"]["QT_OVER_U"]["recall@10"]}
        comparisons.append({"seed":seed,"order_delta":order_delta,"baseline_delta":baseline_delta,
            "order_pass": None if order_delta is None else order_delta["E_addition"] >= .005 and min(order_delta["U_R10"],order_delta["implicit_U_R10"]) >= -.01,
            "baseline_pass": None if baseline_delta is None else min(baseline_delta.values()) >= -.02})
    passes = any(all(r[key] is True for r in comparisons) for key in ("order_pass","baseline_pass"))
    ruled_out = all(any(r[key] is False for r in comparisons) for key in ("order_pass","baseline_pass"))
    return {"status":"triggered" if passes else "not_triggered" if ruled_out else "unassessable",
            "comparisons":comparisons,"interpretation":"One-time budget gate, not statistical significance or scientific rejection"}


def source_cluster_comparison(deltas: np.ndarray, sources: list[str], *, replicates: int = 10000, seed: int = 260914) -> dict:
    """Average seeds per query before calling; sample whole sources with replacement."""
    if len(deltas) != len(sources) or len(deltas) == 0:
        raise ValueError("Nonempty paired queries and complete source mapping required")
    _,groups = np.unique(sources,return_inverse=True)
    n = int(groups.max())+1
    counts = np.bincount(groups)
    sums = np.bincount(groups,weights=deltas)
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0,replicates,250):
        selected = rng.integers(0,n,size=(min(250,replicates-start),n))
        draws.extend((sums[selected].sum(1)/counts[selected].sum(1)).tolist())
    return {"queries":len(deltas),"source_clusters":n,"mean_delta":float(np.mean(deltas)),
            "bootstrap_95ci":[float(v) for v in np.quantile(draws,[.025,.975])],
            "wins":int(np.sum(deltas>1e-12)),"losses":int(np.sum(deltas < -1e-12)),"ties":int(np.sum(np.abs(deltas)<=1e-12)),
            "replicates":replicates,"seed":seed}
