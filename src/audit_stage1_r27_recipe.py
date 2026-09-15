"""Describe historical/modern recipe differences without changing training inputs."""
from __future__ import annotations

from collections import Counter

import numpy as np

from prepare_stage1_r27 import ROOT, OUT, R12, R13, R26, rows, record, read_json, write_json, stable_sha


def audit() -> dict:
    old_path = R12 / "taskC_training/candidates_seed13_steps356/candidates.jsonl.gz"
    modern_path = ROOT / "work/stage1_optimization_r25_final_20260914/common/c1_selective_hard_seed13.jsonl.gz"
    old = list(rows(old_path))
    new = list(rows(modern_path))
    counts = Counter()
    exposures = {}
    for name, batches in (("historical", old), ("modern_first356", new[:356]), ("modern_full", new)):
        flat = [e for b in batches for e in b["examples"]]
        exposures[name] = {"batches": len(batches), "lists": len(flat), "query_order_sha256": stable_sha([e["query_id"] for e in flat]), "relations": dict(Counter(f"{e['source_type']}_to_{e['destination_type']}" for e in flat))}
    for b1, b2 in zip(old, new):
        for e1, e2 in zip(b1["examples"], b2["examples"]):
            counts["compared_list_positions"] += 1
            key = ("query_id", "source_type", "destination_type")
            aligned = all(e1[k] == e2[k] for k in key)
            counts["same_source_relation"] += aligned
            if aligned:
                a, b = set(e1["candidate_ids"]), set(e2["candidate_ids"])
                pa, pb = set(e1["positive_ids"]), set(e2["positive_ids"])
                counts["candidate_order_equal"] += e1["candidate_ids"] == e2["candidate_ids"]
                counts["candidate_membership_equal"] += a == b
                counts["positive_membership_equal"] += pa == pb
                counts["negative_membership_equal"] += (a-pa) == (b-pb)
                counts["added_candidates"] += len(b-a)
                counts["removed_candidates"] += len(a-b)
                counts["added_positives"] += len(pb-pa)
    old_graph_path = R12 / "taskC_training/c2_candidates_seed13/path_hard.jsonl"
    new_graph_path = ROOT / "work/stage1_optimization_r24_20260913/path_pool/common_seed13.jsonl"
    teacher_path = ROOT / "work/stage1_optimization_r25_final_20260914/common/teacher_native_path_cache.jsonl.gz"
    graph = list(rows(old_graph_path))
    modern = {r["query_id"]: r for r in rows(new_graph_path)}
    teachers = {r["query_id"]: r for r in rows(teacher_path)}
    witness = {r["query_id"]: r for r in rows(R12 / "taskA_correctness/supervision/target_lists.train_fit.jsonl")}
    path_counts = {"positive": [], "negative": []}
    bag_stats = Counter()
    logits = {"direct": [], "evidence": []}
    ranking_agreement = Counter()
    for row in graph:
        q = row["query_id"]
        nc = {c["target_id"]: c for c in modern[q]["candidates"]}
        tc = {t: i for i, t in enumerate(teachers[q]["candidate_ids"])}
        positives = set(row["positive_target_ids"])
        for i, candidate in enumerate(row["candidates"]):
            t = candidate["target_id"]
            group = "positive" if t in positives else "negative"
            paths = candidate["evidence_ids"]
            after = nc[t]["evidence_ids"]
            path_counts[group].append(len(paths))
            lost = set(paths) - set(after)
            bag_stats[f"{group}/targets"] += 1
            bag_stats[f"{group}/targets_with_removed_paths"] += bool(lost)
            bag_stats[f"{group}/removed_paths"] += len(lost)
            bag_stats[f"{group}/prefix8_equal"] += after == paths[:8]
            known = witness[q].get("positive_evidence_by_target", {}).get(t, [])
            bag_stats[f"{group}/removed_known_witnesses"] += len(lost & set(known))
            for e in lost:
                bag_stats[f"removed_modality/{'image' if e.startswith('asset_img') else 'text'}"] += 1
            mapping = witness[q].get("positive_evidence_rows_by_target", {}).get(t, {})
            before_rows = {r for e in paths for r in mapping.get(e, [])}
            after_rows = {r for e in after for r in mapping.get(e, [])}
            bag_stats[f"{group}/removed_known_row_coverage"] += len(before_rows-after_rows)
            for channel in logits:
                a = row[f"teacher_{channel}_logits"][i]
                b = teachers[q][f"{channel}_logits"][tc[t]]
                if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
                    logits[channel].append((a,b))
                else:
                    bag_stats[f"{channel}/masked_or_nonfinite"] += 1
                bag_stats[f"{channel}/none_mask_disagreement"] += (a is None) != (b is None)
        for channel in logits:
            a = row[f"teacher_{channel}_logits"]
            b = teachers[q][f"{channel}_logits"]
            ids = [c["target_id"] for c in row["candidates"]]
            eligible = [t for i,t in enumerate(ids) if a[i] is not None and b[tc[t]] is not None]
            av = dict(zip(ids,a))
            ra = sorted(eligible, key=lambda t:(-av[t],t))
            rb = sorted(eligible, key=lambda t:(-b[tc[t]],t))
            ranking_agreement[f"{channel}/queries"] += 1
            ranking_agreement[f"{channel}/same_order"] += ra == rb
    order = read_json(R13 / "taskD_witness_supervision/schedule_order.json")["indices"]
    historical_ids = [graph[i]["query_id"] for i in order]
    current_ids = [r["query_id"] for r in rows(R26 / "common/c2_order.jsonl")]
    result = {"status": "completed_readonly_no_training", "C1": {"historical": record(old_path), "modern": record(modern_path), "exposures": exposures, "aligned_position_differences": counts}, "C2": {"historical": record(old_graph_path), "modern": record(new_graph_path), "path_count_quantiles": {k: dict(zip(("min","median","p95","max"), np.quantile(v,[0,.5,.95,1]).tolist())) for k,v in path_counts.items()}, "bag_differences": bag_stats, "teacher_cache": record(teacher_path), "teacher_differences": {k:{"aligned_pairs": len(v), "old_mean": float(np.mean(v,axis=0)[0]), "modern_mean": float(np.mean(v,axis=0)[1]), "mean_abs_difference": float(np.abs(np.array(v)[:,0]-np.array(v)[:,1]).mean()), "max_abs_difference": float(np.abs(np.array(v)[:,0]-np.array(v)[:,1]).max())} for k,v in logits.items()}, "teacher_ranking_agreement": ranking_agreement, "consumed_query_order": {"historical_sha256": stable_sha(historical_ids), "modern_sha256": stable_sha(current_ids), "equal": historical_ids == current_ids, "same_positions": sum(a==b for a,b in zip(historical_ids,current_ids))}}, "interpretation": "Descriptive recipe differences only; no individual causal attribution. Historical runtime and historical dependency snapshot remain to be independently recovered."}
    write_json(OUT / "historical_replay/H_RECIPE_DIFF.json", result)
    return {"C1": counts, "C2_bags": bag_stats}


if __name__ == "__main__":
    import json
    print(json.dumps(audit()))
