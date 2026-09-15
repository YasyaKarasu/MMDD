"""Score R26 A/B/C diagnostics and replay immutable Stage2 candidates."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from mmdd_dataset.utils import values_match
from mmdd_stage2.join_diagnostic import ColumnProjection, column_metrics, recall, spearman, write_json
from run_r26_joinability_diagnostic import OUT, load_columns


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Scores:
    def __init__(self, out: Path, plan: dict):
        self.columns = plan["columns"]
        vectors = load_columns(out)
        self.keys = sorted(vectors)
        assert set(self.keys) == set(self.columns)
        self.index = {k: i for i,k in enumerate(self.keys)}
        raw = F.normalize(torch.stack([vectors[k] for k in self.keys]).float(), dim=1)
        self.matrices = {"B": raw}
        self.checkpoints = {}
        for seed in (13,29):
            ckpt = torch.load(out / f"projection_seed{seed}.pt", map_location="cpu", weights_only=False)
            assert ckpt["epoch"] >= 1, "Arm C must be trained; epoch 0 is a separate ablation"
            model = ColumnProjection()
            model.load_state_dict(ckpt["state_dict"])
            with torch.inference_mode():
                self.matrices[f"C{seed}"] = model(raw)
            self.checkpoints[f"C{seed}"] = {k:v for k,v in ckpt.items() if k != "state_dict"}
        cells = torch.load(out / "cells.pt", map_location="cpu", weights_only=False)
        assert cells["values"] == plan["cell_values"]
        self.cell_index = {v:i for i,v in enumerate(cells["values"])}
        self.cell_vectors = F.normalize(cells["vectors"].float(), dim=1)
        self.cache = {}

    def cosine(self, left: str, right: str, arm: str) -> float:
        if not any(v.strip() for v in self.columns[left]["values"]) or not any(v.strip() for v in self.columns[right]["values"]):
            return 0.
        matrix = self.matrices[arm]
        return float(matrix[self.index[left]] @ matrix[self.index[right]])

    def cell(self, left: str, right: str) -> dict:
        key = (left,right)
        if key in self.cache:
            return self.cache[key]
        q = self.columns[left]["values"]
        t = [v for v in self.columns[right]["values"] if v.strip()]
        if not q or not t:
            result = {"coverage": 0., "mean": 0., "exact": 0., "best": [0.] * len(q)}
        else:
            qv = self.cell_vectors[[self.cell_index[v] for v in q]]
            tv = self.cell_vectors[[self.cell_index[v] for v in t]]
            best = (qv @ tv.T).amax(dim=1).tolist()
            exact = [bool(v.strip()) and any(values_match(v,w) for w in t) for v in q]
            best = [0. if not v.strip() else 1. if ex else s for v, ex, s in zip(q,exact,best)]
            result = {"coverage": sum(bool(v.strip()) and s >= .8 for v,s in zip(q,best))/len(q),
                      "mean": float(np.mean(best)), "exact": float(np.mean(exact)), "best": best}
        self.cache[key] = result
        return result

    def scalar(self, left: str, right: str, arm: str) -> float:
        if arm != "A":
            return self.cosine(left,right,arm)
        result = self.cell(left,right)
        return result["coverage"] + result["mean"] / (2 * (len(self.columns[left]["values"]) + 1))


def aggregate(rows: list[dict], fields: list[str]) -> dict:
    return {f: float(np.mean([r[f] for r in rows if r[f] is not None])) if any(r[f] is not None for r in rows) else None for f in fields}


def bootstrap_delta(rows: list[dict], field: str, reference: str, reps: int = 3000) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[r["source_table_id"]].append(r[field]-r[reference])
    chunks = list(groups.values())
    if not chunks:
        return {}
    rng = np.random.default_rng(260915)
    sums = np.array([sum(c) for c in chunks])
    counts = np.array([len(c) for c in chunks])
    choice = rng.integers(len(chunks), size=(reps,len(chunks)))
    values = sums[choice].sum(1)/counts[choice].sum(1)
    return {"delta": float(sums.sum()/counts.sum()), "ci95": np.quantile(values,[.025,.975]).tolist(), "source_groups": len(chunks)}


def analyze(out: Path) -> None:
    torch.set_num_threads(4)
    plan = json.loads((out / "prepared.json").read_text())
    scorer = Scores(out,plan)
    arms = ["A", *scorer.matrices]
    report = {"checkpoints": scorer.checkpoints}
    # Known query join column is ONLY supplied for this offline diagnostic.
    column_rows, distributions = [], []
    for example in plan["examples"]:
        if example["split"] == "train":
            continue
        for arm in arms:
            values = [scorer.scalar(example["anchor"],k,arm) for k in example["candidates"]]
            metrics = column_metrics(values,[k == example["gold"] for k in example["candidates"]])
            column_rows.append({"split": example["split"], "scope": "known_join_column", "arm": arm,
                "query_id": example["query_id"], "target_id": example["target_id"], "kind": example["kind"],
                "source_table_id": example["source_table_id"], "columns": len(values), **metrics})
            for key, value in zip(example["candidates"], values):
                distributions.append({"split": example["split"], "arm": arm, "query_id": example["query_id"],
                    "kind": example["kind"], "positive": key == example["gold"], "score": value})
    report["column_metrics"] = {}
    for split in ("calibration", "eval"):
        report["column_metrics"][split] = {}
        for kind in ("overall","explicit","implicit"):
            report["column_metrics"][split][kind] = {}
            for arm in arms:
                subset = [r for r in column_rows if r["split"] == split and r["arm"] == arm and (kind == "overall" or r["kind"] == kind)]
                report["column_metrics"][split][kind][arm] = {"pairs": len(subset), **aggregate(subset,["top1","mrr","auc"])}
    thresholds = {}
    for arm in scorer.matrices:
        positive = [r["score"] for r in distributions if r["split"] == "calibration" and r["kind"] == "explicit" and r["positive"] and r["arm"] == arm]
        thresholds[arm] = float(np.quantile(positive,.05,method="lower"))
    report["thresholds"] = thresholds
    print("Known-column diagnostics complete", flush=True)

    labels = defaultdict(list)
    for e in plan["examples"]:
        if e["split"] == "eval":
            labels[e["query_id"], e["target_id"]].append(e)
    # Unique query-target cells across both generators/conditions: no duplicate width observations.
    target_pairs = {}
    for generator, rows in plan["pilots"].items():
        for row in rows:
            for c in row["diagnostic_candidates"]:
                key = (row["query_id"],c["target_id"])
                target_pairs.setdefault(key, {"direct_eligible": False})
                target_pairs[key]["direct_eligible"] |= "direct" in c["branches"]
    # Include all gold targets for diagnostic evaluation even if C18 missed them. Never insert in replay.
    for key in labels:
        target_pairs.setdefault(key,{"direct_eligible": False})
    width_rows, pair_rows, subsample_rows, target_max, visible_metrics = [], [], [], {}, []
    rng = np.random.default_rng(260915)
    for num, ((qid,tid), eligibility) in enumerate(sorted(target_pairs.items())):
        pop = plan["population"][qid]
        qkeys, tkeys = plan["tables"]["query"][qid], plan["tables"]["target"][tid]
        combinations = [(q,t) for q in qkeys for t in tkeys]
        n = len(combinations)
        all_scores = {}
        for arm in arms:
            vals = [scorer.scalar(q,t,arm) for q,t in combinations]
            all_scores[arm] = vals
            if (qid,tid) in labels:
                by_column = [max(vals[i*len(tkeys)+j] for i in range(len(qkeys))) for j in range(len(tkeys))]
                gold_keys = {e["gold"] for e in labels[qid,tid]}
                visible_metrics.append({"query_id":qid,"target_id":tid,"kind":pop["query_kind"],"arm":arm,
                    **column_metrics(by_column,[k in gold_keys for k in tkeys])})
            winner = int(np.argmax(vals))
            q,t = combinations[winner]
            cell = scorer.cell(q,t) if arm == "A" else {}
            maximum = cell["coverage"] if arm == "A" else vals[winner]
            accepted = maximum >= (.6 if arm == "A" else thresholds[arm])
            target_max[qid,tid,arm] = {"score": vals[winner], "query_key": q, "target_key": t,
                "maximum": maximum, "accepted": accepted, "mean": cell.get("mean",vals[winner])}
            width_rows.append({"query_id": qid, "target_id": tid, "arm": arm, "kind": pop["query_kind"],
                "positive": tid in pop["positive_target_ids"], "source_table_id": pop["source_table_id"],
                "query_columns": len(qkeys), "target_columns": len(tkeys), "column_pairs": n,
                "target_max_rows": max(len(plan["columns"][k]["values"]) for k in tkeys),
                "score": maximum, "mean_similarity": cell.get("mean"), "exact_coverage_winner": cell.get("exact"),
                "accepted": accepted, "direct_eligible": eligibility["direct_eligible"],
                "query_column_name": plan["columns"][q]["name"], "target_column_name": plan["columns"][t]["name"]})
            # Nested random subsets control content/width confounding within the same table pair.
            for draw in range(16):
                order = rng.permutation(n)
                for budget in sorted({1,min(2,n),min(4,n),min(8,n),min(16,n),n}):
                    subset = [vals[i] for i in order[:budget]]
                    idx = order[int(np.argmax(subset))]
                    maximum_sub = scorer.cell(*combinations[idx])["coverage"] if arm == "A" else vals[idx]
                    subsample_rows.append({"query_id": qid,"target_id": tid,"arm": arm,"kind": pop["query_kind"],
                        "positive": tid in pop["positive_target_ids"], "draw": draw,"available_pairs": n,"budget": budget,"score": maximum_sub})
        for (q,t), *scores in zip(combinations, *[all_scores[a] for a in arms]):
            cell = scorer.cell(q,t)
            qvalues = plan["columns"][q]["values"]
            pair_rows.append({"query_id": qid, "target_id": tid, "query_key": q, "target_key": t,
                "kind": pop["query_kind"], "positive_target": tid in pop["positive_target_ids"],
                "coverage": cell["coverage"], "mean_similarity": cell["mean"], "exact_coverage": cell["exact"],
                "best_similarities": json.dumps(cell["best"]), **dict(zip(arms, scores))})
        if num % 200 == 0:
            print(f"Table-pair diagnostics {num}/{len(target_pairs)}", flush=True)
    write_csv(out / "column_pairs.csv",pair_rows)
    write_csv(out / "table_width.csv",width_rows)
    write_csv(out / "width_subsampling.csv",subsample_rows)
    write_csv(out / "visible_max_column_metrics.csv",visible_metrics)
    report["visible_max_column_metrics"] = []
    for kind in ("overall","implicit","explicit"):
        for arm in arms:
            subset = [r for r in visible_metrics if r["arm"] == arm and (kind == "overall" or r["kind"] == kind)]
            report["visible_max_column_metrics"].append({"kind":kind,"arm":arm,"pairs":len(subset),**aggregate(subset,["top1","mrr","auc"])})

    # FDR is labeled as a proxy: implicit GT does not prove absence of incidental direct joins.
    report["false_direct"] = {}
    for scope in ("implicit_positive_all", "implicit_positive_direct_eligible", "implicit_nonpositive_direct_eligible"):
        report["false_direct"][scope] = {}
        for arm in arms:
            subset = [r for r in width_rows if r["arm"] == arm and r["kind"] == "implicit"
                and (r["positive"] if "nonpositive" not in scope else not r["positive"])
                and (scope.endswith("all") or r["direct_eligible"])]
            report["false_direct"][scope][arm] = {"pairs": len(subset), "accept_rate": float(np.mean([r["accepted"] for r in subset])) if subset else None,
                "exact_ge_06": float(np.mean([r["exact_coverage_winner"] >= .6 for r in subset])) if arm == "A" and subset else None}
    report["width_correlations"] = []
    for arm in arms:
        for kind in ("implicit","explicit"):
            for positive in (False,True):
                subset = [r for r in width_rows if r["arm"] == arm and r["kind"] == kind and r["positive"] == positive]
                x, y = [r["column_pairs"] for r in subset], [r["score"] for r in subset]
                rho = spearman(x,y)
                report["width_correlations"].append({"arm":arm,"kind":kind,"positive":positive,"pairs":len(subset),"spearman":rho})
    report["threshold_sensitivity"] = []
    for threshold in (.7,.75,.8,.85,.9,.95):
        by_pair = defaultdict(list)
        for row in pair_rows:
            qs = plan["columns"][row["query_key"]]["values"]
            sims = json.loads(row["best_similarities"])
            coverage = sum(bool(v.strip()) and s >= threshold for v,s in zip(qs,sims))/len(qs)
            by_pair[row["query_id"],row["target_id"]].append(coverage)
        selected = [max(vals) >= .6 for (qid,tid),vals in by_pair.items() if plan["population"][qid]["query_kind"] == "implicit" and tid in plan["population"][qid]["positive_target_ids"]]
        report["threshold_sensitivity"].append({"cosine_threshold":threshold,"implicit_positive_pairs":len(selected),"false_direct_proxy":float(np.mean(selected))})
    report["A_winning_column_names"] = dict(Counter((r["query_column_name"]+" -> "+r["target_column_name"]) for r in width_rows if r["arm"] == "A" and r["kind"] == "implicit" and r["accepted"]).most_common(20))

    replay_rows, rankings, parity, recovered_metrics = [], [], [], []
    for generator, rows in plan["pilots"].items():
        for row in rows:
            qid = row["query_id"]
            pop = plan["population"][qid]
            positives = pop["positive_target_ids"]
            initial = next(r for r in plan["inputs"][generator] if r["query_id"] == qid)
            baseline = [c["target_id"] for c in initial["results"]]
            assert len(baseline) == 18 and len(set(baseline)) == 18
            record = {"generator":generator,"condition":row["condition"],"query_id":qid,"kind":pop["query_kind"],"source_table_id":pop["source_table_id"],"status":row["status"]}
            rank_by_arm = {"T0":baseline,"A":row["ranking"]}
            chosen_branches = {"A":{c["target_id"]:c["final_branch"] for c in row["diagnostic_candidates"]}}
            for arm in scorer.matrices:
                scored = []
                branches = {}
                for c in row["diagnostic_candidates"] if row["status"] == "ran" else []:
                    tid = c["target_id"]
                    direct = target_max[qid,tid,arm] if "direct" in c["branches"] else None
                    rk = c.get("recovered_key")
                    evidence = scorer.cosine(rk,f"target:{tid}:{c['selection']['column_index']}",arm) if rk else None
                    options = [(direct["score"],"direct")] if direct else []
                    if evidence is not None:
                        options.append((evidence,"evidence"))
                    value, branch = max(options,key=lambda x:x[0]) if options else (0.,None)
                    scored.append((value,c["stage1_rank"],tid))
                    branches[tid] = branch
                rank_by_arm[arm] = [t for _,_,t in sorted(scored,key=lambda x:(-x[0],x[1]))]
                chosen_branches[arm] = branches
            for arm, ranking in rank_by_arm.items():
                if row["status"] == "ran":
                    assert set(ranking) == set(baseline)
                for k in (1,3,5,7,9,18):
                    record[f"{arm}_R{k}"] = recall(ranking,positives,k)
                record[f"{arm}_E9"] = sum(chosen_branches.get(arm,{}).get(t) == "evidence" for t in ranking[:9])
                rankings.append({"generator":generator,"condition":row["condition"],"query_id":qid,"arm":arm,"ranking":ranking,
                    "branches":chosen_branches.get(arm,{})})
            replay_rows.append(record)
            if row["status"] != "ran":
                continue
            for c in row["diagnostic_candidates"]:
                tid = c["target_id"]
                direct = c["branches"].get("direct")
                if direct:
                    check = scorer.cell(f"query:{qid}:{direct['query_column']}",f"target:{tid}:{direct['target_column']}")
                    old = direct["verification"]
                    parity.append({"generator":generator,"query_id":qid,"target_id":tid,"condition":row["condition"],
                        "coverage_delta":check["coverage"]-old["coverage"],"mean_delta":check["mean"]-old["mean_similarity"],
                        "max_coverage_delta":target_max[qid,tid,"A"]["maximum"]-old["coverage"]})
                if c.get("recovered_key") and (qid,tid) in labels:
                    golds = {e["gold"] for e in labels[qid,tid]}
                    keys = plan["tables"]["target"][tid]
                    for arm in arms:
                        scores = [scorer.scalar(c["recovered_key"],k,arm) for k in keys]
                        metrics = column_metrics(scores,[k in golds for k in keys])
                        recovered_metrics.append({"generator":generator,"condition":row["condition"],"query_id":qid,"target_id":tid,"arm":arm,
                            "predicted_column_correct":f"target:{tid}:{c['selection']['column_index']}" in golds,
                            "nonempty_rows":sum(bool(v.strip()) for v in plan["columns"][c["recovered_key"]]["values"]),**metrics})
    report["replay"] = []
    for generator in plan["pilots"]:
        for condition in ("Real-crop","Real-crop+original","NoE-fill"):
            for kind in ("overall","implicit","explicit"):
                subset = [r for r in replay_rows if r["generator"] == generator and r["condition"] == condition and (kind == "overall" or r["kind"] == kind)]
                for arm in ("T0",*arms):
                    values = aggregate(subset,[f"{arm}_R{k}" for k in (1,3,5,7,9,18)])
                    report["replay"].append({"generator":generator,"condition":condition,"kind":kind,"arm":arm,"queries":len(subset),
                        **{k.split("_",1)[1]:v for k,v in values.items()},"evidence_top9":sum(r[f"{arm}_E9"] for r in subset),
                        "delta_vs_A_R9":bootstrap_delta(subset,f"{arm}_R9","A_R9"),"delta_vs_T0_R9":bootstrap_delta(subset,f"{arm}_R9","T0_R9")})
    report["cell_recompute_parity"] = {"comparisons":len(parity),"coverage_disagreements":sum(abs(r["coverage_delta"]) > 1e-6 for r in parity),
        "max_coverage_disagreements":sum(abs(r["max_coverage_delta"]) > 1e-6 for r in parity),
        "mean_absolute_delta":float(np.mean([abs(r["mean_delta"]) for r in parity])),"max_absolute_delta":max(abs(r["mean_delta"]) for r in parity),
        "policy":"A replay always uses original cached ranking/scores, because deduplicated bf16 cell batching can change numerical ties."}
    report["recovered_column_metrics"] = []
    for generator in plan["pilots"]:
        for condition in ("Real-crop","Real-crop+original","NoE-fill"):
            for arm in arms:
                subset = [r for r in recovered_metrics if r["generator"] == generator and r["condition"] == condition and r["arm"] == arm]
                report["recovered_column_metrics"].append({"generator":generator,"condition":condition,"arm":arm,"pairs":len(subset),
                    **aggregate(subset,["top1","mrr","auc","predicted_column_correct","nonempty_rows"])})
    write_csv(out / "column_metrics.csv",column_rows)
    write_csv(out / "score_distributions.csv",distributions)
    write_csv(out / "replay_per_query.csv",replay_rows)
    write_csv(out / "cell_parity.csv",parity)
    write_csv(out / "recovered_column_metrics.csv",recovered_metrics)
    with (out / "replay_rankings.jsonl").open("w") as handle:
        for r in rankings:
            handle.write(json.dumps(r)+"\n")
    write_json(out / "SUMMARY.json",report)
    print(json.dumps({"output":str(out / "SUMMARY.json"),"parity":report["cell_recompute_parity"]}),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUT)
    args = parser.parse_args()
    analyze(args.output)
