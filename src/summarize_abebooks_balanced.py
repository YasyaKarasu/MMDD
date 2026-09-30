"""Summarize balanced AbeBooks training budgets, recall, and evidence coverage."""
from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from mmdd_cqet_v4_1.data import iter_jsonl, read_json, write_json
from run_abebooks_balanced import SCHEDULES
from summarize_abebooks_data_ablation import diagnostics

KS = (5, 10, 15, 20)


def state_delta(left: Path, right: Path) -> float:
    a = torch.load(left, map_location="cpu", weights_only=False)["model"]
    b = torch.load(right, map_location="cpu", weights_only=False)["model"]
    return math.sqrt(sum(float((a[k].double() - b[k].double()).square().sum()) for k in a))


def training_audit(root: Path, schedules: dict | None = None) -> dict:
    results = {}
    teachers = []
    for name, (epochs, batch) in (SCHEDULES if schedules is None else schedules).items():
        run = root / name
        complete = read_json(run / "TRAINING_COMPLETE.json")
        freeze = read_json(run / "SELECTION_FREEZE.json")
        logs = {stage: list(iter_jsonl(run / f"{stage}.jsonl"))
                for stage in ("TA", "TB_CQET", "C1", "C2_SUP", "C2_KD")}
        expected = {"TA": 2 * math.ceil(complete["counts"]["TA"] / 8),
                    "TB_CQET": math.ceil(complete["counts"]["TB"] / 8),
                    "C1": epochs * math.ceil(complete["counts"]["C1"] / batch),
                    "C2_SUP": epochs * math.ceil(complete["counts"]["C2"] / batch),
                    "C2_KD": epochs * math.ceil(complete["counts"]["C2"] / batch)}
        for stage, records in logs.items():
            assert len(records) == expected[stage]
            assert [r["step"] for r in records] == list(range(1, len(records) + 1))
            assert all(math.isfinite(r["loss"]) and math.isfinite(r["grad_norm_preclip"]) for r in records)
        assert [r["record_ids"] for r in logs["C2_SUP"]] == [r["record_ids"] for r in logs["C2_KD"]]
        assert all(r["kd_weight"] == 0.3 for r in logs["C2_KD"])
        for stage, filename, key in (("C1", "C1", "item_id"), ("C2_SUP", "C2_SHARED", "query_id"),
                                     ("C2_KD", "C2_SHARED", "query_id")):
            unique = Counter(r[key] for r in iter_jsonl(run / f"training_records/{filename}.jsonl.gz"))
            for epoch in range(1, epochs + 1):
                assert Counter(q for r in logs[stage] if r["epoch"] == epoch for q in r["record_ids"]) == unique
        sup0, kd0 = run / "C2_SUP/snapshot_frac000.pt", run / "C2_KD/snapshot_frac000.pt"
        assert state_delta(sup0, kd0) == 0
        epoch_losses = {}
        for stage, records in logs.items():
            epoch_losses[stage] = []
            for epoch in sorted({r["epoch"] for r in records}):
                rows = [r for r in records if r["epoch"] == epoch]
                weights = [len(r["record_ids"]) for r in rows]
                epoch_losses[stage].append({"epoch": epoch,
                    "loss": sum(r["loss"] * w for r, w in zip(rows, weights)) / sum(weights),
                    "weighting": "number of query/edge records in logical batch"})
        deltas = {arm: state_delta(run / f"C2_{arm}/snapshot_frac000.pt", Path(freeze["endpoints"][arm]))
                  for arm in ("SUP", "KD")}
        assert all(delta > 0 for delta in deltas.values())
        results[name] = {"status": "PASS", "steps": expected, "epoch_losses": epoch_losses,
            "endpoint_L2_from_parent": deltas,
            "SUP_KD_endpoint_L2": state_delta(Path(freeze["endpoints"]["SUP"]), Path(freeze["endpoints"]["KD"])),
            "paired_parent_and_order_equal": True, "each_record_seen_once_per_epoch": True}
        teachers.append(Path(freeze["teacher"]))
    teacher_delta = [state_delta(teachers[0], path) for path in teachers[1:]]
    assert max(teacher_delta, default=0) == 0
    return {"schedules": results, "teachers_identical_across_schedules": True}


def learning_diagnostics(root: Path, schedules: dict | None = None) -> dict:
    """Full-lake exact direct ranking on train/dev, independent of ANN approximation."""
    ids = read_json(root / "features/z/z_index.json")["ids"]
    positions = {oid: i for i, oid in enumerate(ids)}
    z = np.load(root / "features/z/z.f32.npy", mmap_mode="r")
    targets = sorted(r["table_id"] for r in iter_jsonl(root / "dataset_view/data_lake_tables/part-00000.jsonl"))
    gold = defaultdict(set)
    splits = {}
    for row in iter_jsonl(root / "dataset_view/qrels.jsonl"):
        if row["split"] in {"train", "dev"} and row["rel"] > 0:
            gold[row["query_table_id"]].add(row["target_table_id"])
            splits[row["query_table_id"]] = row["split"]
    queries = sorted(gold)
    zq = torch.tensor(np.asarray(z[[positions[q] for q in queries]]))
    zt = torch.tensor(np.asarray(z[[positions[t] for t in targets]]))
    results = {}
    for name in (SCHEDULES if schedules is None else schedules):
        run = root / name
        selection = read_json(run / "selection_C1.json")
        freeze = read_json(run / "SELECTION_FREEZE.json")
        checkpoints = {"C1_init": run / "C1/snapshot_frac000.pt",
                       "C1_endpoint": Path(selection["points"][-1]["checkpoint"]),
                       "C2_SUP_endpoint": Path(freeze["endpoints"]["SUP"]),
                       "C2_KD_endpoint": Path(freeze["endpoints"]["KD"])}
        results[name] = {}
        for stage, path in checkpoints.items():
            state = torch.load(path, map_location="cpu", weights_only=False)["model"]
            with torch.no_grad():
                uq = (zq - state["pca_mean"]) @ state["P.table"].T
                ut = (zt - state["pca_mean"]) @ state["P.table"].T
                scores = ((uq @ state["R.QT"]) @ ut.T).numpy()
            rows = []
            for i, q in enumerate(queries):
                ranked = sorted(range(len(targets)), key=lambda j: (-float(scores[i, j]), targets[j]))
                ordered = [targets[j] for j in ranked]
                gold_ranks = [ordered.index(t) + 1 for t in gold[q]]
                rows.append({"query_id": q, "split": splits[q], "MRR": 1 / min(gold_ranks),
                    "best_gold_rank": min(gold_ranks),
                    **{f"R@{k}": len(set(ordered[:k]) & gold[q]) / len(gold[q]) for k in KS}})
            results[name][stage] = {split: {metric: float(np.mean([r[metric] for r in rows if r["split"] == split]))
                for metric in ("MRR", "best_gold_rank", *(f"R@{k}" for k in KS))} for split in ("train", "dev")}
    return results


def raw_matched_control(root: Path) -> dict:
    """Retrospective raw exact control: same new test queries, same 137 targets."""
    old = root.parent / "abebooks_data_ablation_20260930/columns/features/z"
    target_ids = sorted(r["table_id"] for r in iter_jsonl(root / "dataset_view/data_lake_tables/part-00000.jsonl"))
    gold = defaultdict(set)
    for row in iter_jsonl(root / "dataset_view/qrels.jsonl"):
        if row["split"] == "test" and row["rel"] > 0:
            gold[row["query_table_id"]].add(row["target_table_id"])
    results = {}
    for name, directory in (("before_four_columns_removed", old), ("after_four_columns_removed", root / "features/z")):
        ids = read_json(directory / "z_index.json")["ids"]
        positions = {oid: i for i, oid in enumerate(ids)}
        z = np.load(directory / "z.f32.npy", mmap_mode="r")
        targets = np.asarray(z[[positions[t] for t in target_ids]], dtype=np.float64)
        targets /= np.linalg.norm(targets, axis=1, keepdims=True)
        rows = []
        for q, g in sorted(gold.items()):
            query = np.asarray(z[positions[q]], dtype=np.float64)
            query /= np.linalg.norm(query)
            scores = targets @ query
            order = sorted(range(len(target_ids)), key=lambda i: (-scores[i], target_ids[i]))
            ranked = [target_ids[i] for i in order]
            rows.append({"query_id": q, "gold_ranks": {t: ranked.index(t) + 1 for t in sorted(g)},
                         **{f"R@{k}": len(set(ranked[:k]) & g) / len(g) for k in KS}})
        results[name] = {"queries": len(rows), **{f"R@{k}": sum(r[f"R@{k}"] for r in rows) / len(rows) for k in KS},
                         "per_query": rows}
    return {"scope": "exact cosine, raw frozen backbone, fixed new test queries and target set; no old learned model used",
            "old_features_used_only_for_retrospective_raw_control": True, "results": results}


def summarize(root: Path) -> None:
    assert read_json(root / "SUITE_COMPLETE.json")["status"] == "COMPLETE"
    curation = read_json(root / "dataset_view/dataset_manifest.json")["rebalanced"]
    audit = training_audit(root)
    write_json(root / "TRAINING_AUDIT.json", audit)
    recall, evidence, selection = [], {}, {}
    for name in SCHEDULES:
        run = root / name
        recall.extend({"schedule": name, **r} for r in read_json(run / "recall_summary.json"))
        evidence[name] = {g: diagnostics(run, g) for g in ("raw", "selected_sup", "selected_kd", "endpoint_sup", "endpoint_kd")}
        selection[name] = {stage: read_json(run / f"selection_{stage}.json") for stage in ("C1", "C2_SUP")}
    matched = raw_matched_control(root)
    learning = learning_diagnostics(root)
    write_json(root / "LEARNING_DIAGNOSTICS.json", learning)
    write_json(root / "COMPARISON.json", {"curation": curation, "recall": recall, "evidence": evidence,
               "selection": selection, "training": audit, "raw_matched_control": matched,
               "learning_diagnostics": learning})
    lookup = {(r["schedule"], r["split"], r["generator"], r["mode"], r["segment"]): r for r in recall}

    def values(row: dict) -> str:
        return " | ".join(f"{row[f'R@{k}'] * 100:.2f}%" for k in KS)

    primary = lookup["ten_epochs_b16", "test", "endpoint_sup", "Teacher_Real", "overall"]
    lines = ["# AbeBooks：删除四列、平衡重划分与训练预算对照", "",
        "## 结论", "",
        "已完成数据整理和三组完整训练。新数据为 118 个 Query，implicit/explicit 各 59。",
        "三组 SUP/KD 的 Student Direct 与 Multimodal_RRF 在 Test Recall@5/10/15/20 上仍全为 0；未发现 KD 的聚合指标优势。",
        f"主设置 10 轮/batch16 的完整 C2 终点，Teacher Real Recall@5/10/15/20 为 {values(primary)}。SUP/KD 相同；这属于 Teacher 对候选评分后的效果。",
        "固定新测试 Query 和 137 张 target 的 Raw 精确对照中，删四列前后四项 Recall 都为 0，尚无直接检索改善证据。",
        "更多更新确实发生了，但没有解决训练不足：C2 SUP 的训练集 Direct R@10 仍为 2/94=2.13%，dev/test 为 0；不能把问题只归结为测试泛化或标注数量。",
        "所有 dev 选择仍停在 C1/C2 的第 0 轮。因此各组 C2 都从 C1 初始投影开始，未承接完成 190 次更新的 C1 终点；报告同时保留 C1 终点的独立诊断。", "",
        "## 数据", "",
        "在上一轮 columns 数据版上继续删除 availability_quantity、seller_rating、copy_condition_grade、binding。",
        "任何正例 join 使用这四列的 Query 整体删除（包括其余关联标注）；相应 join target 删除。",
        f"删除 {len(curation['removed_join_queries'])} 个 Query 和 {len(curation['removed_join_targets'])} 个 Target 后，implicit=59、explicit=61。",
        "固定 seed=13 下采样 2 个 explicit Query，得到 59 implicit + 59 explicit。其余正常 target 保留为共享数据湖候选。", "",
        "| Split | implicit | explicit | 合计 |", "|---|---:|---:|---:|"]
    for split, counts in curation["split_counts"].items():
        lines.append(f"| {split} | {counts['implicit']} | {counts['explicit']} | {sum(counts.values())} |")
    lines += ["", f"最终 {curation['queries']} 个 Query、{curation['targets']} 个 Target、{curation['qrels']} 个正例关系、"
        f"{curation['evidence_recoveries']} 条证据恢复记录、{curation['assets']} 个可编码素材。",
        "所有源表、Query、Target 一致投影，并同步索引、gold 与恢复记录；保留值未改变。同一源表的 Query 不跨 split。",
        "数据目录：`dataset/abebooks_joinability_no4_balanced_20260930`。候选湖跨 split 共享，遵循原实验设置。", "",
        "## 训练步数与受控设置", "",
        "Student 原实现只有一轮，batch=64。新增 epochs 支持，每轮独立确定性打乱，尾批单独更新，AdamW 状态跨轮连续保留。默认 epochs=1 的行为保持。",
        "三组均完整执行新训练的 TA → TB_CQET → C1 → C2 SUP/KD；Teacher 保持 2/1 epochs、batch=8，不同时改动 Teacher 预算。",
        "所有 5,696 个对象的 embedding/content 为本实验新生成，三组共享这批固定输入；PCA 和模型重新拟合。核对三组 Teacher 参数完全相同。", "",
        "| 设置 | C1 更新 | C2 SUP 更新 | C2 KD 更新 | TA / TB 更新 |", "|---|---:|---:|---:|---|"]
    for name, data in audit["schedules"].items():
        s = data["steps"]
        lines.append(f"| {name} | {s['C1']} | {s['C2_SUP']} | {s['C2_KD']} | {s['TA']} / {s['TB_CQET']} |")
    lines += ["", "10 轮设置在每轮末保存模型，由 dev 的 C150 覆盖率、U 覆盖率、Direct R@10、较早轮次依次选择；KD 使用 SUP 选中的轮次。全部设置在测试前冻结。",
        "C150 上限大于本数据湖的 137 张表，候选覆盖容易饱和，不能只凭它宣称排序改善。", "",
        "| 设置 | C1 选中轮次 | C2 选中轮次 |", "|---|---:|---:|"]
    for name, (epochs, _) in SCHEDULES.items():
        a = selection[name]
        lines.append(f"| {name} | {a['C1']['selected']['fraction'] * epochs:g} | {a['C2_SUP']['selected']['fraction'] * epochs:g} |")
    for generator_prefix, title in (("selected", "Dev 选中模型"), ("endpoint", "固定训练终点（确实完成全部更新）")):
        lines += ["", f"## Test Recall：{title}", "",
            "Direct 为 Student 直接检索；Multimodal_RRF 为两跳融合；Teacher_Real 为同一候选上的 Teacher 评分，不能混称为 Student 的直接 Recall。",
            "Recall 为逐 Query 的 gold 覆盖比例再取平均。", "",
            "| 设置 | Student | 排序 | R@5 | R@10 | R@15 | R@20 |", "|---|---|---|---:|---:|---:|---:|"]
        for name in SCHEDULES:
            for arm in ("sup", "kd"):
                for mode in ("Direct", "Multimodal_RRF", "Teacher_Real"):
                    row = lookup[name, "test", f"{generator_prefix}_{arm}", mode, "overall"]
                    lines.append(f"| {name} | {arm.upper()} | {mode} | {values(row)} |")
    lines += ["", "## 主设置 ten_epochs_b16：implicit / explicit", "",
        "| 模型 | 排序 | 类型 | Query 数 | R@5 | R@10 | R@15 | R@20 |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for arm in ("sup", "kd"):
        for mode in ("Direct", "Multimodal_RRF", "Teacher_Real"):
            for kind in ("implicit", "explicit"):
                row = lookup["ten_epochs_b16", "test", f"selected_{arm}", mode, kind]
                lines.append(f"| {arm.upper()} | {mode} | {kind} | {row['queries']} | {values(row)} |")
    lines += ["", "## 正确证据与行/属性覆盖", "",
        "| 设置 | 模型 | QE20 命中对 | ET50 命中对 | D1 保留对 | D1 覆盖行/属性/值单元 |", "|---|---|---:|---:|---:|---:|"]
    for name in SCHEDULES:
        for g in ("raw", "selected_sup", "selected_kd", "endpoint_sup", "endpoint_kd"):
            e = evidence[name][g]
            lines.append(f"| {name} | {g} | {e['any_QE20_hit']}/{e['witness_pairs']} | {e['any_ET50_hit']}/{e['witness_pairs']} | {e['any_D1_retained']}/{e['witness_pairs']} | {e['D1_covered_units']}/{e['annotated_row_attribute_value_units']} |")
    lines += ["", "这里只统计已标注恢复单元对应证据的覆盖；没有重新调用属性抽取或二阶段 join 验证，不代表新的恢复值准确率。", "",
        "## 固定 Query/Target 的 Raw 对照", "",
        "旧实验重新划分后有些训练 Query 成为新 test，旧的训练模型不能作为无泄漏对照。这里仅比较未训练的 Raw backbone 精确余弦排名，固定新 test Query 与同样的 137 张 target；旧 embedding 只用于此回顾性对照，不输入本轮训练。", "",
        "| 表内容 | R@5 | R@10 | R@15 | R@20 |", "|---|---:|---:|---:|---:|"]
    for name, row in matched["results"].items():
        lines.append(f"| {name} | {values(row)} |")
    lines += ["", "## Train / dev 是否真正学到排序", "",
        "全 137 张目标表上的直接精确评分，排除 ANN 近似影响；MRR 使用最近的 gold 名次。", "",
        "| 设置 | 阶段 | Train R@10 | Dev R@10 | Train MRR | Dev MRR |", "|---|---|---:|---:|---:|---:|"]
    for name, stages in learning.items():
        for stage, segments in stages.items():
            a, b = segments["train"], segments["dev"]
            lines.append(f"| {name} | {stage} | {100*a['R@10']:.2f}% | {100*b['R@10']:.2f}% | {a['MRR']:.4f} | {b['MRR']:.4f} |")
    lines += ["", "## 如何理解增加步数", "",
        "- one_epoch_b64 → ten_epochs_b64：主要观察更多训练轮数的效果。ten_epochs_b64 → ten_epochs_b16：观察同样标签曝光轮数下，较小 batch 和更多更新的效果。C1 的 dev 选择也可能改变后续 C2 的父模型和候选图，因而这是训练流程设置的对照。",
        "- 重复十轮仍只有 94 个独立训练 Query，不等于新增了标注；训练 loss 降低不保证 dev/test 改善。",
        "- 主设置 C2 SUP 的逐记录加权训练 loss 仅从第 1 轮 9.8772 降至第 10 轮 9.8523；C1 完整终点将训练 gold 平均最近名次从 110.17 改至 102.65，仍远离前 20。当前 P/R 学习率为 1e-6/1e-5。",
        "- 建议下一步先在 8–16 个 train Query 上做过拟合诊断，核对监督对齐、分数/梯度尺度和训练可拟合性，再仅用 train/dev 比较更合适的学习率与总更新预算。本次尚未执行这个诊断或学习率搜索。",
        "- 模型选择也需另做对照：本轮 C150/U 覆盖率饱和且 R@10 一直为 0，小幅排序进步无法被当前规则识别。可用 dev MRR/平均 gold 名次作细粒度辅助，同时保留证据覆盖和最终 Recall 的检查，不能单独凭训练 loss 选模型。",
        "- 后续若增加监督数据，应优先扩充不同实体/源表的真实标注，或在 train 源表内部构造有正确恢复证据的不同子表视图；先分组划分，再生成视图，不能把同源变体散到测试集。",
        "- 现有 C1 已包含 QT、QE、ET 多种关系的监督，不能把这些边数量与独立 Query 数混为一谈。",
        "- Test 只有 12 个 Query（每类 6 个），单 seed 结果用于探索；类别内一个 Query 就影响约 16.7 个百分点。不得直接与旧的 15-query 测试结果作等价比较。", "",
        "Dev 的 12 个 Query 来自 4 个源表组，Test 来自 8 个源表组；有效独立样本数更小。扩大标注时应增加不同源表，并扩大 dev 的源表覆盖。", "",
        "## 核验与复现", "",
        "`TRAINING_AUDIT.json` 记录实际步数、逐轮 loss、参数变化、SUP/KD 相同父状态与样本顺序；`COMPARISON.json` 含全量分类型指标、dev 轨迹与证据漏斗。",
        "多轮/恢复/原训练内核测试：`cd /tmp && PYTHONPATH=/home/oycy/MMDD/src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 conda run -n MMDD python -m pytest /home/oycy/MMDD/tests/test_mmdd_cqet_v4_1.py -q`，26 passed。",
        "数据测试单独运行（训练 provenance 测试会检查进程已导入模块）：`PYTHONPATH=/home/oycy/MMDD/src conda run -n MMDD python -m pytest /home/oycy/MMDD/tests/test_abebooks_rebalance.py /home/oycy/MMDD/tests/test_abebooks_ablation.py -q`，5 passed。",
        "复现入口：`build_abebooks_balanced.py` → `run_abebooks_fresh.py prepare-data/encode` → `run_abebooks_balanced.py run` → `summarize_abebooks_balanced.py`。运行环境 MMDD、工作目录 /tmp；具体路径与命令见各组 COMMANDS.jsonl。", ""]
    (root / "REPORT.zh-CN.md").write_text("\n".join(lines))
    print("Wrote COMPARISON.json, TRAINING_AUDIT.json, REPORT.zh-CN.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    summarize(parser.parse_args().run_root.resolve())
