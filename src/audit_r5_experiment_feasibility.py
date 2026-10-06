"""Read frozen R5 artifacts to size evidence-selection and text-span experiments.

No model inference or training. Source-table labels are used only in the post-hoc
recovery audit, never in evidence selection. Run from an isolated working directory.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from audit_r5_artifacts import paired, rank_metrics
from mmdd_stage2.values import norm, value_info


def records(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def artifact(root: Path, name: str):
    for path in sorted((root / name).glob("*.jsonl")):
        yield from records(path)


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def teacher_vs_student(root: Path, output: Path) -> None:
    """Use actual SUP Q-T scores on precisely the C150 scored by the Teacher."""
    audit = root / "work/r5_independent_audit_20261006"
    stage1 = root / "work/stage1_entitables_r5_s13_sup_rerank"
    with (audit / "teacher_controls/PER_QUERY.csv").open() as handle:
        controls = list(csv.DictReader(handle))
    contrasts, summaries = [], []
    for split in ("dev", "test"):
        gold = defaultdict(set)
        for record in records(stage1 / f"eval_labels/{split}/qrels.jsonl"):
            if record["rel"] > 0:
                gold[record["query_id"]].add(record["target_id"])
        metrics = {}
        for record in records(stage1 / f"seed13/eval/{split}/native_sup/pools.jsonl.gz"):
            order = sorted(record["C150"], key=lambda t: (-record["all_U_QT_scores"][t], t))
            metrics[record["query_id"]] = rank_metrics(order, gold[record["query_id"]])
        for population in ("overall", "implicit", "explicit"):
            selected = {policy: [r for r in controls if r["split"] == split and r["scope"] == "C150"
                                and r["policy"] == policy and (population == "overall" or r["kind"] == population)]
                        for policy in ("FULL", "F0")}
            for metric in ("R10", "NDCG10"):
                for policy, rows in selected.items():
                    contrasts.append({"split": split, "population": population, "method": policy,
                                      "reference": "SUP_DIRECT_SAME_C150", "metric": metric,
                                      **paired([r["source_group"] for r in rows],
                                               [float(r[metric]) - metrics[r["query_id"]][metric] for r in rows])})
            rows = selected["FULL"]
            summaries.append({"split": split, "population": population, "queries": len(rows),
                              "SUP_DIRECT_R10": float(np.mean([metrics[r["query_id"]]["R10"] for r in rows])),
                              "TEACHER_FULL_R10": float(np.mean([float(r["R10"]) for r in rows]))})
    save_csv(output / "TEACHER_VS_SUP_SAME_C150.csv", contrasts)
    save_csv(output / "TEACHER_VS_SUP_METRICS.csv", summaries)


def run(root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    stage1 = root / "work/stage1_entitables_r5_s13_sup_rerank"
    stage2 = root / "work/stage2_entitables_r5_sup_crop_on"
    config = json.loads((stage2 / "config.json").read_text())
    dataset = Path(config["paths"]["dataset_root"])
    protocol = json.loads((stage1 / "protocol.json").read_text())
    vectors = Path(protocol["paths"]["pure_cache_dir"]) / "z"
    index = {key: i for i, key in enumerate(json.loads((vectors / "z_index.json").read_text())["ids"])}
    z = np.load(vectors / "z.f32.npy", mmap_mode="r")
    db = sqlite3.connect(f"file:{stage2 / 'catalog.sqlite'}?mode=ro", uri=True)
    assets = {}
    plans = {split: list(records(stage2 / f"plans/{split}.jsonl")) for split in ("dev", "test")}
    ids = {p["query_id"] for ps in plans.values() for p in ps}
    for eid in sorted({e for ps in plans.values() for p in ps for v in p["views"] for e in v["evidence_ids"]}):
        assets[eid] = json.loads(db.execute("SELECT payload FROM objects WHERE kind='asset' AND id=?", (eid,)).fetchone()[0])
    db.close()
    tokenizer = Tokenizer.from_file(str(Path(config["paths"]["qwen_model"]) / "tokenizer.json"))
    token_lengths = {eid: len(tokenizer.encode(a["content"][:3000], add_special_tokens=False).ids)
                     for eid, a in assets.items() if a["asset_type"] == "text"}
    queries = {q["table_id"]: q for q in artifact(dataset, "query_tables") if q["table_id"] in ids}
    source_ids = {q["source_table_id"] for q in queries.values()}
    sources = {s["source_table_id"]: s for s in artifact(dataset, "source_tables") if s["source_table_id"] in source_ids}
    witnesses = defaultdict(set)
    for record in artifact(dataset, "evidence_recoveries"):
        if record["query_table_id"] in ids:
            witnesses[record["query_table_id"], record["query_row_id"], norm(record["recovered_attribute"]["column_name"])].add(record["evidence"]["asset_id"])
    summary, view_rows, slot_rows = {}, [], []
    for split, split_plans in plans.items():
        selected_targets = {p["query_id"]: {link["target_id"] for v in p["views"] for link in v["donor_links"]} for p in split_plans}
        path_scores = defaultdict(dict)
        for record in records(stage1 / f"seed13/eval/{split}/native_sup/logits.TB_CQET.Real.jsonl.gz"):
            if record["target_id"] in selected_targets.get(record["query_id"], set()):
                path_scores[record["query_id"]][record["target_id"]] = {
                    p["evidence_id"]: p["raw_QET"] - record["f0"] for p in record["paths"]}
        count = Counter(queries=len(split_plans))
        modality_tasks, gates, statuses = Counter(), Counter(), Counter()
        changed_qe, changed_path, multi_queries = set(), set(), set()
        localization_keys, seen_text = set(), set()
        text_token_total, text_token_max_saving = 0, 0
        choice_witness = {name: Counter() for name in ("teacher", "cosine_qe", "cosine_path")}
        recovery_seconds, model_inputs = 0.0, 0
        for plan in split_plans:
            qid = plan["query_id"]
            query = queries[qid]
            query_eids = sorted({e for v in plan["views"] for e in v["evidence_ids"]})
            # Normalize the frozen pooled vectors once per query. No new encoding.
            involved = [qid] + sorted(set(query_eids) | selected_targets[qid])
            values = np.array(z[[index[i] for i in involved]], dtype=np.float32)
            values /= np.linalg.norm(values, axis=1, keepdims=True)
            vi = {key: i for i, key in enumerate(involved)}
            cosine_qe = {eid: float(values[vi[eid]] @ values[vi[qid]]) for eid in query_eids}
            for view in plan["views"]:
                count["views"] += 1
                evidence = view["evidence_ids"]
                chosen = {k: [] for k in ("teacher", "cosine_qe", "cosine_path")}
                choices = False
                for modality in ("text", "image"):
                    eligible = [e for e in evidence if assets[e]["asset_type"] == modality]
                    count[f"views_with_{modality}"] += bool(eligible)
                    count[f"views_multiple_{modality}"] += len(eligible) > 1
                    choices |= len(eligible) > 1
                    if not eligible:
                        continue
                    score_teacher = {e: float(np.mean([path_scores[qid][l["target_id"]][e] for l in view["donor_links"]])) for e in eligible}
                    score_path = {e: cosine_qe[e] + float(np.mean([values[vi[e]] @ values[vi[l["target_id"]]] for l in view["donor_links"]])) for e in eligible}
                    for name, scores in (("teacher", score_teacher), ("cosine_qe", cosine_qe), ("cosine_path", score_path)):
                        chosen[name].append(min(eligible, key=lambda e: (-scores[e], e)))
                different_qe = set(chosen["teacher"]) != set(chosen["cosine_qe"])
                different_path = set(chosen["teacher"]) != set(chosen["cosine_path"])
                count["views_with_choice"] += choices
                count["views_teacher_differs_cosine_qe"] += different_qe
                count["views_teacher_differs_cosine_path"] += different_path
                if choices:
                    multi_queries.add(qid)
                if different_qe:
                    changed_qe.add(qid)
                if different_path:
                    changed_path.add(qid)
                view_rows.append({"split": split, "query_id": qid, "view_id": view["view_id"],
                                  "attribute": view["attribute"], "bag_size": len(evidence),
                                  "has_choice": choices, "teacher_differs_qe": different_qe,
                                  "teacher_differs_path": different_path,
                                  **{k: json.dumps(v) for k, v in chosen.items()}})
                if choices:
                    for row_id in range(len(query["rows"])):
                        known = witnesses[qid, row_id, view["attribute"]]
                        if known & set(evidence):
                            for name, selected in chosen.items():
                                choice_witness[name]["eligible_view_rows"] += 1
                                choice_witness[name]["retained_known_witness"] += bool(known & set(selected))
            recovery = json.loads((stage2 / f"recovery/{split}/{qid}.json").read_text())
            recovery_seconds += recovery.get("seconds", 0)
            model_inputs += recovery.get("model_inputs", 0)
            for task in recovery.get("tasks", []):
                mods = sorted({assets[e]["asset_type"] for e in task["evidence_ids"]})
                modality_tasks["+".join(mods) or "none"] += 1
                gates[task["gate"]] += 1
                statuses[task["status"]] += 1
                for eid in task["evidence_ids"]:
                    if eid in token_lengths:
                        seen_text.add(eid)
                        # Estimate localization opportunity on tasks that actually reached generation.
                        if task["gate"] != "OTHER_ROWS_ONLY" and task["status"] not in {"NO_EVIDENCE", "GATED_OUT"}:
                            localization_keys.add((qid, task["row_id"], task["attribute"], eid))
                            text_token_total += token_lengths[eid]
                            text_token_max_saving += max(0, token_lengths[eid] - 192)
            gold_attributes = query.get("hidden_attributes", [])
            if gold_attributes:
                count["implicit_queries"] += 1
            bridges = {b["attribute"]: b for b in recovery["bridges"]}
            source = sources[query["source_table_id"]]
            source_rows = {r["row_id"]: r for r in source["rows"]}
            for hidden in gold_attributes:
                attr, cid = norm(hidden["column_name"]), hidden["source_column_index"]
                matching_views = [v for v in plan["views"] if v["attribute"] == attr]
                planned_evidence = {e for v in matching_views for e in v["evidence_ids"]}
                bridge_slots = {s["row_id"]: s for s in bridges.get(attr, {}).get("slots", [])}
                for row_id, row in enumerate(query["rows"]):
                    source_row = source_rows[row["source_row_id"]]
                    gold = next(c["text"] for c in source_row["cells"] if c["column_index"] == cid)
                    gold_key = value_info(gold)["key"]
                    slot = bridge_slots.get(row_id, {})
                    pred = slot.get("value_key") if slot.get("status") == "VALUE" else None
                    correct = pred is not None and pred == gold_key
                    slot_rows.append({"split": split, "query_id": qid, "source_group": query["source_table_id"],
                                      "row_id": row_id, "attribute": attr, "gold": gold, "gold_key": gold_key,
                                      "planned": bool(matching_views), "has_evidence": bool(planned_evidence),
                                      "annotated_witness_in_plan": bool(witnesses[qid, row_id, attr] & planned_evidence),
                                      "status": slot.get("status", "NO_BRIDGE"), "predicted_key": pred,
                                      "exact_match": correct})
        lengths = np.array([token_lengths[e] for e in sorted(seen_text)])
        long_keys = [key for key in localization_keys if token_lengths[key[-1]] > 192]
        localization_windows = sum(1 + max(0, (token_lengths[key[-1]] - 1024 + 895) // 896) for key in long_keys)
        slots = [s for s in slot_rows if s["split"] == split and s["gold_key"] is not None]
        count.update(queries_with_choice=len(multi_queries), queries_teacher_differs_qe=len(changed_qe),
                     queries_teacher_differs_path=len(changed_path))
        summary[split] = {"selection": dict(count), "tasks_by_modality": dict(modality_tasks),
                          "task_gate": dict(gates), "task_status": dict(statuses),
                          "recovery_seconds": recovery_seconds, "model_inputs": model_inputs,
                          "text": {"unique_assets": len(lengths), "tokens_quantiles_0_50_90_95_100": np.quantile(lengths, [0, .5, .9, .95, 1]).tolist(),
                                   "assets_over_192": int((lengths > 192).sum()), "assets_over_1024": int((lengths > 1024).sum()),
                                   "candidate_row_attribute_asset_localizations": len(localization_keys),
                                   "candidate_localizations_over_192": len(long_keys), "estimated_window_forwards": localization_windows,
                                   "task_text_tokens_before_truncation": text_token_total,
                                   "task_text_tokens_removed_at_192": text_token_max_saving},
                          "selection_known_witness_diagnostic": {name: dict(c) for name, c in choice_witness.items()},
                          "hidden_attribute_slots": {"nonempty_gold": len(slots),
                              **{key: sum(bool(s[key]) for s in slots) for key in ("planned", "has_evidence", "annotated_witness_in_plan", "exact_match")},
                              "emitted_value": sum(s["status"] == "VALUE" for s in slots),
                              "conflict": sum(s["status"] == "CONFLICT" for s in slots),
                              "queries_with_exact_match": len({s["query_id"] for s in slots if s["exact_match"]})}}
    save_csv(output / "VIEW_SELECTION.csv", view_rows)
    save_csv(output / "HIDDEN_ATTRIBUTE_SLOTS.csv", slot_rows)
    (output / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    teacher_vs_student(root, output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.root, args.output)
