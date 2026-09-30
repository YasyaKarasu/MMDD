"""Audit a source-rebuilt AbeBooks experiment and independently recompute recall."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from mmdd_dataset.abebooks_ablation import read_rows
from mmdd_dataset.abebooks_source_rebuild import natural_author_names, recovery_key
from run_abebooks_fresh import write_json
from summarize_abebooks_data_ablation import diagnostics
from summarize_abebooks_balanced import learning_diagnostics, training_audit


def compressed(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def summarize(root: Path) -> dict:
    run, dataset = root / "main", root / "dataset_view"
    queries = {q["table_id"]: q for q in read_rows(dataset / "query_tables/part-00000.jsonl")}
    targets = {t["table_id"]: t for t in read_rows(dataset / "data_lake_tables/part-00000.jsonl")}
    sources = {t["source_table_id"]: t for t in read_rows(dataset / "source_tables/part-00000.jsonl")}
    qrels = read_rows(dataset / "qrels.jsonl")
    recoveries = read_rows(dataset / "evidence_recoveries/part-00000.jsonl")
    gold, source_splits, source_kinds = defaultdict(set), defaultdict(set), defaultdict(set)
    query_kinds = defaultdict(set)
    for relation in qrels:
        q, t = queries[relation["query_table_id"]], targets[relation["target_table_id"]]
        assert q["source_table_id"] == t["source_table_id"] == relation["source_table_id"]
        assert relation["split"] == q["split"]
        name = relation["join_attribute"]["column_name"]
        assert name in {c["column_name"] for c in t["columns"]}
        visible = name in {c["column_name"] for c in q["columns"]}
        assert visible == (relation["reason"] == "explicit_visible_join_column")
        source_splits[q["source_table_id"]].add(q["split"])
        source_kinds[q["source_table_id"]].add(relation["reason"])
        query_kinds[q["table_id"]].add("explicit" if visible else "implicit")
        gold[q["table_id"]].add(t["table_id"])
    assert all(len(v) == 1 for v in source_splits.values())
    assert all(len(v) == 1 for v in source_kinds.values())
    assert set(queries) == set(gold)
    assert all(len(v) == 1 for v in query_kinds.values())
    counts = Counter(next(iter(v)) for v in query_kinds.values())
    assert counts["implicit"] == counts["explicit"]
    split_counts = {split: dict(Counter(next(iter(query_kinds[q])) for q in queries
                   if queries[q]["split"] == split)) for split in ("train", "dev", "test")}
    assert all(set(q["target_table_ids"]) == gold[qid] for qid, q in queries.items())
    originals = {}
    for recovery in recoveries:
        origin = Path(recovery["annotation_provenance"]["source_dataset"])
        if origin not in originals:
            originals[origin] = {recovery_key(r) for r in read_rows(origin / "evidence_recoveries/part-00000.jsonl")}
        assert recovery_key(recovery) in originals[origin]
        original_name, original_value = recovery_key(recovery)[3:]
        current_value = recovery["recovered_attribute"]["value"]
        if current_value != original_value:
            assert original_name == "authors" and current_value == natural_author_names(original_value)
        q, t = queries[recovery["query_table_id"]], targets[recovery["target_table_id"]]
        assert recovery["target_table_id"] in gold[q["table_id"]]
        assert q["rows"][recovery["query_row_id"]]["source_row_id"] == recovery["source_row_id"]
        for target_row in recovery["target_row_ids"]:
            values = {c["column_name"]: c["text"] for c in t["rows"][target_row]["cells"]}
            attr = recovery["recovered_attribute"]
            assert values[attr["column_name"]] == attr["value"]
    aliases = {a["asset_id"]: a["canonical_evidence_id"]
               for a in compressed(run / "CONTENT_ALIASES.jsonl.gz")}
    evidence_splits = defaultdict(set)
    for recovery in recoveries:
        evidence_splits[aliases[recovery["evidence"]["asset_id"]]].add(recovery["split"])
    assert all(len(v) == 1 for v in evidence_splits.values())
    plan = json.loads((root / "EXPERIMENT_PLAN.json").read_text())
    repo = Path(__file__).resolve().parents[1]
    for relative, digest in plan["method_source_hashes"].items():
        assert hashlib.sha256((repo / relative).read_bytes()).hexdigest() == digest
    rebuild = json.loads((root / "DATA_REBUILD.json").read_text())
    for key, origin in (("input_hashes", rebuild["source"]),
                        ("source_reference_hashes", rebuild.get("source_reference"))):
        for relative, digest in rebuild.get(key, {}).items():
            assert hashlib.sha256((Path(origin) / relative).read_bytes()).hexdigest() == digest
    summary = json.loads((run / "recall_summary.json").read_text())
    per_query = read_rows(run / "recall_per_query.jsonl")
    splits = sorted({r["split"] for r in summary})
    generators = sorted({r["generator"] for r in summary})
    recomputed = {}
    evidence = {}
    for split in splits:
        for generator in generators:
            path = run / "eval" / split / generator
            pools = compressed(path / "pools.jsonl.gz")
            rankings = {"Direct": {p["query_id"]: p["D100_ANN"] for p in pools},
                        "Multimodal_RRF": {p["query_id"]: p["C150"] for p in pools}}
            for view in ("Real", "f0", "Swap"):
                rankings[f"Teacher_{view}"] = {r["query_id"]: r["target_ids"]
                    for r in compressed(path / f"rankings.TB_CQET.{view}.jsonl.gz")}
            for mode, orders in rankings.items():
                assert set(orders) == {q for q in queries if queries[q]["split"] == split}
                for q, order in orders.items():
                    assert len(order) == len(set(order))
                    for k in (5, 10, 15, 20):
                        recomputed[split, generator, mode, q, k] = len(set(order[:k]) & gold[q]) / len(gold[q])
            evidence[f"{split}/{generator}"] = diagnostics(run, generator, split)
    for row in per_query:
        for k in (5, 10, 15, 20):
            assert abs(row[f"R@{k}"] - recomputed[
                row["split"], row["generator"], row["mode"], row["query_id"], k]) < 1e-12
    for row in summary:
        selected = [r for r in per_query if all(r[key] == row[key] for key in ("split", "generator", "mode"))
                    and (row["segment"] == "overall" or r["kind"] == row["segment"])]
        assert len(selected) == row["queries"]
        for k in (5, 10, 15, 20):
            assert abs(row[f"R@{k}"] - sum(r[f"R@{k}"] for r in selected) / len(selected)) < 1e-12
    target = [r for r in summary if r["split"] == "test" and r["generator"] == "selected_kd"
              and r["mode"] == "Multimodal_RRF" and r["segment"] == "overall"]
    implicit_target = [r for r in summary if r["split"] == "test" and r["generator"] == "selected_kd"
                       and r["mode"] == "Multimodal_RRF" and r["segment"] == "implicit"]
    contribution = {}
    for split in splits:
        rows = [r for r in per_query if r["split"] == split and r["generator"] == "selected_kd"]
        direct = {r["query_id"]: r["R@10"] for r in rows if r["mode"] == "Direct"}
        rrf = {r["query_id"]: r["R@10"] for r in rows if r["mode"] == "Multimodal_RRF"}
        kinds = {r["query_id"]: r["kind"] for r in rows}
        contribution[split] = {
            "RRF_only_hits": [{"query_id": q, "kind": kinds[q]} for q in direct if rrf[q] > 0 and direct[q] == 0],
            "Direct_only_hits": [{"query_id": q, "kind": kinds[q]} for q in direct if direct[q] > 0 and rrf[q] == 0],
            "macro_recall_difference": sum(rrf[q] - direct[q] for q in direct) / len(direct)}
    schedules = {"main": (plan["student_epochs"], plan["student_batch"])}
    train_audit = training_audit(root, schedules)
    learning = learning_diagnostics(root, schedules)
    write_json(root / "TRAINING_AUDIT.json", train_audit)
    write_json(root / "LEARNING_DIAGNOSTICS.json", learning)
    result = {"audit": "PASS", "queries": len(queries), "targets": len(targets), "source_tables": len(sources),
        "split_counts": split_counts, "query_kind_counts": dict(counts), "original_inputs_unchanged": True,
        "source_split_overlap": 0, "implicit_explicit_source_overlap": 0, "gold_evidence_split_overlap": 0,
        "new_implicit_facts": 0, "independently_recomputed_recall_values": len(recomputed),
        "test_target_met": bool(target and implicit_target and target[0]["R@10"] >= 0.4
                                and implicit_target[0]["R@10"] >= 0.1),
        "required_test_recall": {"overall": 0.4, "implicit": 0.1},
        "primary_test_implicit": implicit_target,
        "primary_test": target, "metrics": summary, "evidence": evidence, "rrf_contribution": contribution,
        "selection": json.loads((run / "SELECTION_FREEZE.json").read_text()),
        "training": json.loads((run / "TRAINING_COMPLETE.json").read_text()),
        "training_audit": train_audit, "learning": learning}
    write_json(root / "RESULTS.json", result)
    lines = ["# AbeBooks：原表字段精简与重建", "",
        "主指标为 dev 选定 KD 的多模态 RRF Recall@10；Teacher 评分仅作诊断。",
        f"当前已评测划分：{', '.join(splits)}。测试集目标是否达到：{result['test_target_met']}。",
        "模型、损失、蒸馏、路径聚合及 checkpoint 选择沿用原方法。全部生成的 train queries 参与训练。",
        "原始数据及其独立备份保留；本实验使用独立 dataset_view。输入数据表由维护中的构造函数生成，未人工挑选查询或目标表。",
        "所有隐式事实来自当前数据已有的已审核恢复标注；没有新增模型标注。新布局未重新调用模型审核。",
        "train/dev/test 按源表分组；全数据 implicit/explicit 各半。各划分实际数量以 DATA_REBUILD.json 为准。",
        "重构后查询集合及候选湖变化，历史数据已多次评测；当前 test 是回归测试集，不能声称是未触碰的泛化评估。", "",
        "| split | generator | 排序 | R@5 | R@10 | R@15 | R@20 |",
        "|---|---|---|---:|---:|---:|---:|"]
    for row in summary:
        if row["segment"] == "overall":
            lines.append(f"| {row['split']} | {row['generator']} | {row['mode']} | " +
                         " | ".join(f"{row[f'R@{k}']:.6f}" for k in (5, 10, 15, 20)) + " |")
    lines += ["", "正确证据、行/属性/已知值覆盖、Real/f0/Swap 对照及逐 query 独立复算见 RESULTS.json。",
              "RRF 相对 Direct 的新增命中与丢失命中逐查询记录在 rrf_contribution；命中不等于完成值抽取与 join 验证。",
              "没有重新执行二阶段值抽取和 join 验证；已知值证据覆盖不能解释为新值恢复率。"]
    (root / "REPORT.zh-CN.md").write_text("\n".join(lines) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run_root.resolve())
    print(json.dumps({k: result[k] for k in ("audit", "test_target_met", "primary_test")}, indent=2))


if __name__ == "__main__":
    main()
