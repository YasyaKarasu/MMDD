"""Audit all R28 requirements before writing the final scientific readout."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from prepare_stage1_r27 import record, rows, sha, write_json
from prepare_stage1_r28 import ROOT, OUT, FAMILIES, inputs
from evaluate_stage1_r28_student import OWN
from audit_stage1_r28_training import audit_job
from audit_stage1_r28_evaluations import audit_available, audit_t0_parity


def audit() -> dict:
    resolved = json.loads((OUT / "R28_RESOLVED_INPUTS.json").read_text())
    assert sha(Path(resolved["lock"]["path"])) == resolved["lock"]["sha256"]
    objective = json.loads((OUT / "R28_OBJECTIVE_AUDIT.json").read_text())
    assert objective["status"] == "pass" and set(objective["smokes"]) == set(FAMILIES)
    for rec in [objective["tests"], *objective["source_identity"].values()]:
        assert sha(Path(rec["path"])) == rec["sha256"]
    for arm, smoke in objective["smokes"].items():
        rec = smoke["receipt"]
        assert sha(Path(rec["path"])) == rec["sha256"]
        execution = json.loads(Path(rec["path"]).read_text())
        assert execution["status"] == "completed" and execution["updates"] == 0
        assert execution["signature"]["smoke_no_optimizer_updates"]
        assert execution["signature"]["arm"] == arm
    frozen_inputs = json.loads((OUT / "INPUT_HASHES.json").read_text())
    for name,rec in frozen_inputs.items():
        assert sha(Path(rec["path"])) == rec["sha256"],name
    tests = ET.parse(OUT / "evaluation_correctness.xml").getroot().findall("testsuite")
    assert tests and all(int(s.attrib["failures"]) == int(s.attrib["errors"]) == int(s.attrib["skipped"]) == 0 for s in tests)
    assert sum(int(s.attrib["tests"]) for s in tests) >= 54
    for name in ("student_trajectory.png","student_trajectory.pdf","teacher_trajectory.png","teacher_trajectory.pdf"):
        assert (OUT / "figures" / name).stat().st_size > 1000
    extra_audits = []
    for name in ("INDEPENDENT_COVERAGE_AUDIT.json", "STUDENT_EXPORT_NUMERICAL_AUDIT.json", "STUDENT_ADMISSION_DECOMPOSITION_AUDIT.json"):
        path = OUT / name
        value = json.loads(path.read_text())
        assert value["status"] == "pass", name
        assert sha(Path(value["code"]["path"])) == value["code"]["sha256"]
        extra_audits.append(record(path))
    jobs = [audit_job(arm, seed) for arm in FAMILIES for seed in (13, 29)]
    write_json(OUT / "TRAINING_TENSOR_AUDIT_all.json", {"status": "pass", "jobs": jobs})
    content_audit = audit_available(require_complete=True)
    evaluations = []
    for spec in json.loads((OWN / "MODEL_INVENTORY.json").read_text()):
        gid = spec["generator_id"]
        path = OWN / "rankings" / gid / "R28_EVALUATION_RECEIPT.json"
        receipt = json.loads(path.read_text())
        assert receipt["status"] == "completed"
        assert receipt["own_index"]["checkpoint"]["sha256"] == sha(Path(spec["checkpoint"]))
        for name in ("own_rankings","teacher"):
            assert sha(Path(receipt[name]["path"])) == receipt[name]["sha256"]
        assert len(list(rows(Path(receipt["teacher"]["path"])))) == 1198
        evaluations.append(record(path))
    for gid in ["T0"] + [f"{a}/seed{s}/epoch{e:g}" for a in FAMILIES if a.startswith("T-") for s in (13,29) for e in (.5,1,2,3,5)]:
        path = OUT / "teacher/evaluation" / gid / "EVALUATION_RECEIPT.json"
        receipt = json.loads(path.read_text())
        assert receipt["status"] == "completed"
        assert sha(Path(receipt["checkpoint"]["path"])) == receipt["checkpoint"]["sha256"]
        assert sha(Path(receipt["per_query"]["path"])) == receipt["per_query"]["sha256"]
        items = list(rows(Path(receipt["per_query"]["path"])))
        assert len(items) == 1198*12*(2 if receipt["shuffle"] else 1)
        assert {r["view"] for r in items} == {"D","E-LSE","E-COV"}
        assert {str(r["budget"]) for r in items} == {"100","150","200","Full-U"}
        if receipt["shuffle"]:
            assert receipt["shuffle_max_QT_delta"] <= 1e-6
        evaluations.append(record(path))
    analysis = json.loads((OUT / "ANALYSIS_STATUS.json").read_text())
    assert not analysis["missing"] and analysis["bootstrap_comparisons"] > 0
    bootstrap = list(rows(OUT / "statistics/source_group_bootstrap.jsonl"))
    assert all(r["replicates"] == 10000 for r in bootstrap)
    assert {tuple(r["seeds"]) for r in bootstrap} >= {(13,),(29,),(13,29),(0,)}
    assert {r["comparison"] for r in bootstrap} >= {"Path vs Edge","COV vs LSE","Epoch2 vs Epoch1","Epoch3 vs Epoch1","Epoch5 vs Epoch1","Real vs Shuffled"}
    parity = audit_t0_parity()
    result = {"status":"complete","jobs":jobs,"evaluation_receipts":evaluations,
              "training_jobs":12,"checkpoint_receipts":72,"student_own_evaluations":36,
              "teacher_unique_checkpoints":31,"teacher_shuffle_checkpoints":7,
              "T0_historical_Full_U_querywise_recall_parity":parity["queries_with_identical_Recall_at_10_20_50"],
              "bootstrap_comparisons":len(bootstrap),"source_bootstrap_replicates":10000,
              "discarded_failed_attempt_updates":249,"failed_attempts_retained":True,
              "evaluation_content_audited": len(content_audit["completed"]), "additional_numerical_audits": extra_audits,
              "stage2_jobs":0,"remining_jobs":0,"redistillation_jobs":0}
    write_json(OUT / "COMPLETION_AUDIT.json",result)
    return result


def report(completion: dict) -> None:
    with (OUT / "statistics/main_table.csv").open() as handle:
        table = list(csv.DictReader(handle))
    stats = list(rows(OUT / "statistics/source_group_bootstrap.jsonl"))
    def get(section,arm,seed,epoch,view,metric,kind="overall",condition="Real"):
        rs = [r for r in table if r["section"] == section and r["arm"] == arm and r["seed"] == str(seed)
              and float(r["epoch"]) == epoch and r["view"] == view and r["metric"] == metric and r["kind"] == kind
              and r["condition"] == condition and (section != "teacher" or r["budget"] == "Full-U")]
        assert len(rs) == 1,(section,arm,seed,epoch,view,metric)
        return float(rs[0]["value"])
    text = ["# R28 split-path 实验结果\n", "本轮两个主要假设均未获支持：Teacher split-path 没有带来预注册的排序收益，Student 长训练出现明显退化。COV 仍保留部分证据内容依赖，但未转化为相对 Edge 的排序收益。结论、完整轨迹及下一步依据见 [独立结果解读](SCIENTIFIC_REVIEW.md)。\n", "本轮完成预注册的 12 个 continuation 任务；epoch5 为主端点，没有按最好 checkpoint 选结果。Teacher 的主要系统结果使用 D/QT-only，两个 E scorer 分开报告。所有 Student 节点完成 own ANN、Direct exact、Evidence/U/M 与同一冻结 T0 评测。\n",
            "E-only 只排序具有 retained evidence 的 target；因此它与 D view 的可排序集合不同。表中的 RawRecall 表示冻结 candidate membership，另用 rankable_RawRecall / rankable_count 记录各 view 实际可排序集合的上限。\n",
            "## Teacher Full-U 主端点\n", "| Arm | seed | D R@10 | E-LSE R@10 | E-COV R@10 |", "|---|---:|---:|---:|---:|"]
    text.append(f"| T0 | frozen | {get('teacher','T0',0,0,'D','R@10'):.2%} | {get('teacher','T0',0,0,'E-LSE','R@10'):.2%} | {get('teacher','T0',0,0,'E-COV','R@10'):.2%} |")
    for arm in FAMILIES:
        if arm.startswith("T-"):
            for seed in (13,29):
                text.append(f"| {arm} | {seed} | " + " | ".join(f"{get('teacher',arm,seed,5,v,'R@10'):.2%}" for v in ("D","E-LSE","E-COV")) + " |")
    text += ["\n## Student 长训练轨迹\n", "| Arm | seed | epoch1 U RawRecall | epoch5 U RawRecall | epoch1 Full-U+T0 R@10 | epoch5 Full-U+T0 R@10 |", "|---|---:|---:|---:|---:|---:|"]
    for arm in FAMILIES:
        if arm.startswith("S-"):
            for seed in (13,29):
                vals = [get("student",arm,seed,e,v,m) for v,m in (("U","RawRecall"),("U_OFFLINE_T0","R@10")) for e in (1,5)]
                text.append(f"| {arm} | {seed} | " + " | ".join(f"{v:.2%}" for v in vals) + " |")
    text += ["\n## 冻结历史参考与当前 Student parent\n",
             "历史参考直接读取已有排名；当前 C1 parent 使用本轮 epoch0 own ANN。ANN 的重建噪声可能使相同参数得到略不同候选。\n",
             "| Reference | own U RawRecall | own Full-U+T0 R@10 |", "|---|---:|---:|"]
    for arm in ("Qwen-Raw", "B13"):
        text.append(f"| {arm} | {get('historical_reference',arm,0,0,'U','RawRecall'):.2%} | {get('historical_reference',arm,0,0,'U_OFFLINE_T0','R@10'):.2%} |")
    text.append(f"| R28 C1 parent（S-EDGE seed13 epoch0） | {get('student','S-EDGE-LONG',13,0,'U','RawRecall'):.2%} | {get('student','S-EDGE-LONG',13,0,'U_OFFLINE_T0','R@10'):.2%} |")
    text += ["\n## 预注册主要配对比较\n", "family 统计先在同一 query 内平均两个 seed，再按 source group 整组 bootstrap 10,000 次。Recall 差以百分点表示，EO 命中差以每 query 命中数表示；W/L/T 为 query 数。\n",
             "| 比较 | arm | view / metric | Δ | 95% CI | 单位 | W/L/T |", "|---|---|---|---:|---|---|---|"]
    selected = [r for r in stats if r["kind"] == "overall" and r["seeds"] == [13,29]
                and r["comparison"] in ("Path vs Edge","COV vs Edge","COV vs LSE","Epoch5 vs Epoch1","Real vs Shuffled")
                and ((r["left"][0] == "teacher" and r["left"][3] in ("D","E-LSE","E-COV") and r["left"][-1] in ("R@10","EO_STRICT_hits@10"))
                     or (r["left"][0] == "student" and (r["left"][3],r["left"][-1]) in (("U","RawRecall"),("U_OFFLINE_T0","R@10"))))]
    for r in selected:
        lo,hi = r["bootstrap_95ci"]
        scale, unit = (1, "命中数/query") if "EO" in r["left"][-1] else (100, "pp")
        text.append(f"| {r['comparison']} | {r['left'][1]} | {r['left'][3]} / {r['left'][-1]} | {scale*r['mean_delta']:+.5f} | [{scale*lo:+.5f}, {scale*hi:+.5f}] | {unit} | {r['wins']}/{r['losses']}/{r['ties']} |")
    text += ["\n完整 overall/implicit/explicit、R@10/20/50、C100/150/200/Full-U 见 [main_table.csv](statistics/main_table.csv)。EO admission/retention 分母见 [eo_admission_retention.csv](statistics/eo_admission_retention.csv)，每 seed 与 epoch1→2/3/5 的 CI/W/L/T 见 [source_group_bootstrap.jsonl](statistics/source_group_bootstrap.jsonl)。EO hits 差不能读成条件 EO Recall 百分比。\n",
             "## 分组与 seed 方向\n",
             "下面全部使用 R@10；Student 比较 epoch5−epoch1，Teacher 比较 epoch5 Path−Edge。单位为百分点，分组均在同 query 内先平均 seed。\n",
             "| Arm / view | overall Δ [95% CI] | implicit Δ [95% CI] | explicit Δ [95% CI] | seed13 / seed29 overall Δ |",
             "|---|---|---|---|---|"]
    for section, arms, views, label in (
        ("teacher", ("T-PATH-SPLIT-LSE", "T-PATH-SPLIT-COV"), ("D", "E-LSE", "E-COV"), None),
        ("student", ("S-EDGE-LONG", "S-PATH-LONG", "S-COV-LONG"), ("U_OFFLINE_T0",), "Epoch5 vs Epoch1")):
        for arm in arms:
            comparison = label or ("Path vs Edge" if arm.endswith("LSE") else "COV vs Edge")
            for view in views:
                matches = [r for r in stats if r["left"] == [section,arm,5,view,"Real","R@10"] and r["comparison"] == comparison]
                cells = []
                for kind in ("overall", "implicit", "explicit"):
                    r = next(r for r in matches if r["seeds"] == [13,29] and r["kind"] == kind)
                    lo, hi = r["bootstrap_95ci"]
                    cells.append(f"{100*r['mean_delta']:+.3f} [{100*lo:+.3f}, {100*hi:+.3f}]")
                directions = [next(r["mean_delta"] for r in matches if r["seeds"] == [s] and r["kind"] == "overall") for s in (13,29)]
                text.append(f"| {arm} / {view} | " + " | ".join(cells) + f" | {100*directions[0]:+.3f} / {100*directions[1]:+.3f} |")
    with (OUT / "statistics/eo_admission_retention.csv").open() as handle:
        retention = list(csv.DictReader(handle))
    text += ["\n## Strict EO 主端点\n", "family 数量是两 seed 的平均对数。Teacher 的历史 strict EO 固定为207对；Student 同时保留 fixed 与 own 定义，own 数量变化须结合 Direct 变化解释。\n",
             "| Arm | view | EO 定义 | EO 对数 | admitted | Top10 / Top20 / Top50 |", "|---|---|---|---:|---:|---|"]
    for row in retention:
        if row["kind"] != "overall" or row["seed"] != "13+29" or row["condition"] != "Real" or float(row["epoch"]) != 5 or row["K"] != "10":
            continue
        if row["section"] == "teacher" and row["budget"] != "Full-U":
            continue
        if row["section"] == "student" and row["view"] != "U_OFFLINE_T0":
            continue
        companions = [r for r in retention if all(r[k] == row[k] for k in ("section","arm","seed","epoch","budget","view","condition","kind","EO_definition"))]
        hits = [float(next(r["topK_hits"] for r in companions if r["K"] == str(k))) for k in (10,20,50)]
        text.append(f"| {row['arm']} | {row['view']} | {row['EO_definition']} | {float(row['EO_pairs']):g} | {float(row['admitted_pairs']):g} | " + " / ".join(f"{v:g}" for v in hits) + " |")
    text += ["\n## Evidence 内容诊断：Real / Shuffled\n",
             "只使用冻结 T0 和六个 epoch5 端点；QT logits 对 shuffle 不变。下表 strict EO 为固定207对的 Top10 命中数，CI 为每 query EO 命中数差，未乘100。完整分组和 Top20/50 见逐 query 与 bootstrap 文件。\n",
             "| Arm / seed | E view | Real / Shuffled R@10 | Real / Shuffled EO Top10 | EO Δ/query 95% CI |",
             "|---|---|---|---|---|"]
    endpoints = [("T0",0,0)] + [(arm,seed,5) for arm in FAMILIES if arm.startswith("T-") for seed in (13,29)]
    for arm,seed,epoch in endpoints:
        for view in ("E-LSE","E-COV"):
            recalls = [get("teacher",arm,seed,epoch,view,"R@10",condition=c) for c in ("Real","Shuffled")]
            hits = [get("teacher",arm,seed,epoch,view,"EO_STRICT_hits@10",condition=c)*1198 for c in ("Real","Shuffled")]
            comparison = "T0 Real vs Shuffled" if arm == "T0" else "Real vs Shuffled"
            r = next(r for r in stats if r["left"] == ["teacher",arm,epoch,view,"Real","EO_STRICT_hits@10"]
                     and r["comparison"] == comparison and r["seeds"] == [seed] and r["kind"] == "overall")
            lo, hi = r["bootstrap_95ci"]
            text.append(f"| {arm} / {seed} | {view} | {recalls[0]:.2%} / {recalls[1]:.2%} | {hits[0]:.0f} / {hits[1]:.0f} | [{lo:+.5f}, {hi:+.5f}] |")
    overlap = json.loads((OUT / "teacher/evidence_shuffle/CONTENT_OVERLAP_DIAGNOSTIC.json").read_text())
    text.append(f"\nT0 的 cross-source donor 抽样中，{overlap['counts']['identical_content_sets']:,}/{overlap['counts']['bundles']:,} 个 bundle 内容偶然完全相同（{overlap['identical_bundle_fraction']:.2%}）；证据出现次数的重叠率为 {overlap['evidence_overlap_fraction']:.2%}。这些样本保留原抽样，未根据结果重抽；详见 [CONTENT_OVERLAP_DIAGNOSTIC.json](teacher/evidence_shuffle/CONTENT_OVERLAP_DIAGNOSTIC.json)。\n")
    with (OUT / "statistics/student_admission_decomposition.csv").open() as handle:
        accounting = list(csv.DictReader(handle))
    text += ["\n## Student 的 Direct / Evidence 贡献核算\n",
             "下表是同 query 内平均两个 seed 的描述性分解。Evidence-added 指实际 Direct ANN100 以外的 Evidence 目标；两部分严格加总回 U RawRecall 和最终 T0 R@10。它不是新增假设检验，也不是固定目标集合上的因果归因。\n",
             "| Arm | epoch | Direct ANN RawRecall | Evidence-added RawRecall | U RawRecall | T0 R@10：Direct / Evidence-added |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in accounting:
        if row["kind"] == "overall" and row["seed"] == "13+29":
            text.append(f"| {row['arm']} | {row['epoch']} | {float(row['Direct_ANN_RawRecall']):.2%} | {float(row['Evidence_added_outside_ANN_RawRecall']):.2%} | {float(row['U_RawRecall']):.2%} | {float(row['T0_U_R10_from_Direct_ANN_targets']):.2%} / {float(row['T0_U_R10_from_Evidence_added_targets']):.2%} |")
    text.append("\nDirect 召回下降时，原本已被 Direct 召回的目标也可能转入 Evidence-added / own EO 集合；不能把这类集合扩大读成 evidence quality 提升。Direct exact、严格排除 ANN 与 exact 的子集及 implicit/explicit 分组见 [student_admission_decomposition.csv](statistics/student_admission_decomposition.csv)。\n")
    conclusions = []
    for arm in ("T-EDGE-CONT", "T-PATH-SPLIT-LSE", "T-PATH-SPLIT-COV"):
        baseline = next(r for r in stats if r["left"] == ["teacher",arm,5,"D","Real","R@10"]
                        and r["comparison"] == "Epoch5 vs fixed T0" and r["kind"] == "overall" and r["seeds"] == [13,29])
        lo, hi = baseline["bootstrap_95ci"]
        conclusions.append(f"{arm} 相对冻结 T0 的 QT R@10: Δ={100*baseline['mean_delta']:+.3f} pp，95% CI [{100*lo:+.3f}, {100*hi:+.3f}] pp。"
                           + ("该 continuation 的区间完全低于0，不能用相对其他 continuation 的比较掩盖对原始 T0 的退化。" if hi < 0 else ""))
    def contrast(section,arm,view,metric,label,kind="overall"):
        return next(r for r in stats if r["left"] == [section,arm,5,view,"Real",metric]
                    and r["comparison"] == label and r["kind"] == kind and r["seeds"] == [13,29])
    for arm,view,label in (("T-PATH-SPLIT-LSE","E-LSE","Path vs Edge"),("T-PATH-SPLIT-COV","E-COV","COV vs Edge")):
        d = contrast("teacher",arm,"D","R@10",label)
        e = contrast("teacher",arm,view,"EO_STRICT_hits@10",label)
        e_order = contrast("teacher",arm,view,"R@10",label)
        shuffle = contrast("teacher",arm,view,"EO_STRICT_hits@10","Real vs Shuffled")
        directions = [r["mean_delta"] for r in stats if r["comparison"] == label and r["kind"] == "overall"
                      and r["seeds"] in ([13],[29]) and r["left"] == ["teacher",arm,5,"D","Real","R@10"]]
        evidence_directions = [r["mean_delta"] for r in stats if r["comparison"] in (label,"Real vs Shuffled") and r["kind"] == "overall"
                               and r["seeds"] in ([13],[29]) and r["left"] == ["teacher",arm,5,view,"Real","EO_STRICT_hits@10"]]
        order_directions = [r["mean_delta"] for r in stats if r["comparison"] == label and r["kind"] == "overall"
                            and r["seeds"] in ([13],[29]) and r["left"] == ["teacher",arm,5,view,"Real","R@10"]]
        explicit = contrast("teacher",arm,"D","R@10",label,"explicit")
        non_strict = contrast("teacher",arm,"D","non_strict_contribution@10",label)
        e_explicit = contrast("teacher",arm,view,"R@10",label,"explicit")
        e_non_strict = contrast("teacher",arm,view,"non_strict_contribution@10",label)
        evidence_supported = min(e["bootstrap_95ci"][0], shuffle["bootstrap_95ci"][0]) > 0 and len(evidence_directions) == 4 and min(evidence_directions) > 0
        ordering_supported = e_order["bootstrap_95ci"][0] > 0 and len(order_directions) == 2 and min(order_directions) > 0
        guardrail_harm = any(r["bootstrap_95ci"][1] < 0 for r in (explicit,non_strict,e_explicit,e_non_strict))
        if (ordering_supported or evidence_supported) and d["bootstrap_95ci"][0] <= 0:
            conclusions.append(f"{arm}: Path evidence ordering improved, but deployable D/E handoff remains unresolved.")
            if not evidence_supported:
                conclusions.append(f"{arm} 的 E-only ordering 提升尚未同时获得 strict EO 改善及 Real/Shuffled 内容作用的稳定支持，不能把 E-only 增益直接解释成完整 evidence 机制。")
        elif d["bootstrap_95ci"][0] > 0 and evidence_supported and len(directions) == 2 and min(directions) > 0 and not guardrail_harm:
            conclusions.append(f"{arm} 同时支持 QT ordering、strict EO ordering 和真实 evidence 内容作用，两个 seed 方向一致；D 与 E 的 explicit/non-strict 指标未发现区间完全低于0的退化，但这不等于已证明非劣效。")
        else:
            conclusions.append(f"{arm} 未满足完整强证据条件（QT、strict EO、Real/Shuffled、seed 一致性及抵消检查），不宣称完整机制成立。")
        for name, r in (("explicit QT R@10",explicit),("non-strict 对 QT R@10 的贡献",non_strict),
                        (f"explicit {view} R@10",e_explicit),(f"non-strict 对 {view} R@10 的贡献",e_non_strict)):
            lo, hi = r["bootstrap_95ci"]
            conclusions.append(f"{arm} 的 {name}: Δ={100*r['mean_delta']:+.3f} pp，95% CI [{100*lo:+.3f}, {100*hi:+.3f}] pp。" + ("存在明确抵消，不能只报 overall 增益。" if hi < 0 else ""))
    for arm in ("S-EDGE-LONG","S-PATH-LONG","S-COV-LONG"):
        u = contrast("student",arm,"U","RawRecall","Epoch5 vs Epoch1")
        t = contrast("student",arm,"U_OFFLINE_T0","R@10","Epoch5 vs Epoch1")
        directions = [r["mean_delta"] for r in stats if r["comparison"] == "Epoch5 vs Epoch1" and r["kind"] == "overall"
                      and r["seeds"] in ([13],[29]) and r["left"] == ["student",arm,5,"U_OFFLINE_T0","Real","R@10"]]
        u_directions = [r["mean_delta"] for r in stats if r["comparison"] == "Epoch5 vs Epoch1" and r["kind"] == "overall"
                        and r["seeds"] in ([13],[29]) and r["left"] == ["student",arm,5,"U","Real","RawRecall"]]
        supported = min(u["bootstrap_95ci"][0],t["bootstrap_95ci"][0]) > 0 and len(directions) == len(u_directions) == 2 and min(directions+u_directions) > 0
        declining = max(u["bootstrap_95ci"][1],t["bootstrap_95ci"][1]) < 0 and len(directions) == len(u_directions) == 2 and max(directions+u_directions) < 0
        conclusions.append(f"{arm}: " + ("epoch5 相对 epoch1 的 own U 和最终 T0 均有配对支持的提升；训练长度解释得到支持。" if supported else
            "epoch5 相对 epoch1 的 own U 与最终 T0 均明确退化，两个 seed 方向一致；当前 recipe 的 undertraining 解释未获支持。" if declining else
            "epoch5 相对 epoch1 的 own U 与最终 T0 未同时获得明确提升支持，不能仅凭 train loss 下降声称解决 undertraining。"))
    text += ["\n## 论文问题的结论\n", *[f"- {c}" for c in conclusions]]
    text += ["\n## 计算成本与可复核产物\n",
             "[costs.csv](statistics/costs.csv) 给出12个训练任务及各评测的实际耗时、显存和 T0 cache 命中量；并行任务的 wall time 不能相加当作总历时，cache 命中也不能视为新的推理。Student ANN/exact/retention 的摊销延迟在各 RETRIEVAL_RECEIPT 中。本轮没有隔离运行的单查询在线延迟测量。\n",
             "[TRAINING_TENSOR_AUDIT_all.json](TRAINING_TENSOR_AUDIT_all.json) 逐 checkpoint 核验实际参数及 optimizer，并逐 batch 检查数据消费和 loss；[EVALUATION_CONTENT_AUDIT.json](EVALUATION_CONTENT_AUDIT.json) 从分数重建 ranking/Recall，检查 candidate membership、shuffle donors 与两套 EO 集合。\n"]
    text.append("针对大幅退化另做实际数值核对：[STUDENT_EXPORT_NUMERICAL_AUDIT.json](STUDENT_EXPORT_NUMERICAL_AUDIT.json) 检查六个 epoch1 与六个 epoch5 checkpoint 的全部表索引向量，并对三个固定 query 比较模型 bilinear 全库分数与保存的 exact/U 分数；抽样 Top100 集合完全一致。[INDEPENDENT_COVERAGE_AUDIT.json](INDEPENDENT_COVERAGE_AUDIT.json) 对全部31个 Teacher 节点的三个固定 query 独立重算30,324个 COV bags。检查未发现这些导出或公式错误；不据此臆断训练退化的具体内部原因。\n")
    text.append("[INDEPENDENT_STATISTICS_AUDIT.json](INDEPENDENT_STATISTICS_AUDIT.json) 从1,135,704条逐 query 记录独立重建68,238个汇总单元格，并重算84项主要 family 区间与 W/L/T，全部一致。\n")
    parity = json.loads((OUT / "T0_HISTORICAL_PARITY.json").read_text())
    text.append(f"T0 对历史 B13 的 1,198 个 query 的 R@10/20/50 逐 query 完全一致，但不具有历史 scalar 的逐 bit 一致性：最大 QT 差为 {parity['max_QT_score_delta']:.9f}，{len(parity['queries_with_full_rank_order_differences'])} 个 query 的完整排序有差异。三个最大差异样例的实际特征哈希与历史缓存一致，当前历史 scorer 与 R28 scorer 重算一致；历史缓存差异的具体运行时原因未确定。详见 [T0_HISTORICAL_PARITY.json](T0_HISTORICAL_PARITY.json)，未替换分数或修改排序来强行对齐。\n")
    text += ["\n## 完整轨迹与验证\n", "![Student trajectories](figures/student_trajectory.png)\n",
             "![Teacher trajectories](figures/teacher_trajectory.png)\n",
             "数值/回归验证从隔离目录 `/tmp/mmdd-r28-checks` 执行：\n",
             "```bash\nconda run -n MMDD python -m pytest /home/oycy/MMDD/tests/test_stage1_r28.py /home/oycy/MMDD/tests/test_stage1_r26.py -q --junitxml=/home/oycy/MMDD/work/stage1_optimization_r28_split_path_20260915/evaluation_correctness.xml\n```\n",
             "54 项通过。训练前的 13 项 objective/ANN correctness tests、六个零更新真实 parent smoke、补充 hidden-state 缓存审计和完整交付审计均保留机器记录。\n"]
    (OUT / "RESULTS.md").write_text("\n".join(text)+"\n")
    (OUT / "LIMITATIONS.md").write_text("# R28 limitations\n\n"
        "- 两 seed 都从相同 parent continuation，并非独立 fresh lineage；source bootstrap 不能替代更多训练 seed。\n"
        "- 历史 T0 checkpoint 与 runner 的学习率实际为 5e-5，按计划的核实要求沿用；文档中的 1e-5 是未核实建议。\n"
        "- Student 故意移除历史 KD，保留 full bag、parent、seed13 epoch1 order、split SUP 和 anchor；不应声称复现完整 B13 C2。\n"
        "- Teacher 首次尝试因 116 个缺失 hidden-state cache 停止。输入指纹核实后由冻结本地 Qwen 补齐 48 text/68 image 张量，原 embedding manifest 未变。六个逻辑任务从同一 parent 重跑，249 个失败尝试更新不进入正式结果；原记录保留。\n"
        "- COV 的 row support 是冻结 Qwen cosine heuristic，没有训练支持模型；COV 阴性不能靠本轮调温度或权重改写。\n"
        "- Teacher inference 使用历史 B13 retained evidence bundles。Real/Shuffled 匹配确切模态组成和数量，donor 按其他 source group 抽样，不读取 GT；这不是所有潜在证据的质量估计。\n"
        "- 跨 source group 仍可能检索到同一证据：T0 shuffle 有约0.64%的 bundle 内容未变，证据出现次数重叠约0.80%；保留这些偶然重叠，没有事后重抽 donor。\n"
        "- Teacher query-wide batching 的三查询工程审计保留完整排名，分数存在 float32 级差异；T0 的全量 QT query-macro Recall 另与历史协议核对。\n"
        "- T0 与历史缓存的 QT scalar 最大差约0.000310421；103个 query 的完整排序不同，但1,198个 query 的 R@10/20/50 全部完全一致。三例实际特征身份匹配，当前 legacy/R28 scorer 重算也一致；具体历史运行时原因尚未定位，不能声称 scalar 逐 bit 复现。\n"
        "- 独立 ANN 构建存在近邻搜索噪声；每个 checkpoint 记录实际 own index 身份，同时给出 Direct exact。\n"
        "- 所有 CI 为预注册对比的逐项95%区间，没有做多重比较校正；不能按跨零与否从大量诊断中挑选单一正结果。\n"
        "- 多任务并行且 T0 pair cache 复用，因此本轮记录的 wall time 与 batch 摊销延迟不是隔离在线服务延迟。\n"
        "- 本轮没有 Stage2 值恢复实验，不能把 Stage1 的改进直接宣称为正确属性恢复或最终端到端 join discovery 的提升。\n")
    (OUT / "NEXT_DECISION.md").write_text("# R28 下一步决策\n\n"
        "本轮没有支持 Teacher split-path 排序收益或 Student 长训练不足的解释。三个 Teacher continuation 均弱于原始 T0，三个 Student 臂在 epoch5 均明确退化。完整区间、seed 方向、内容诊断及贡献分解见 [独立结果解读](SCIENTIFIC_REVIEW.md) 和 [RESULTS.md](RESULTS.md)。\n\n"
        "1. 保留历史 B13 / frozen T0 作为现有参照，不以本轮 continuation 替换它们。\n"
        "2. 停止在当前 loss/训练长度上继续自动扩展网格，也不从中间 checkpoint 挑选高点改写 epoch5 主端点结论。\n"
        "3. 后续先离线定界关系参数、投影及目标得分分布的变化，再单独预注册 candidate representation / evidence retrieval mechanism 的改动。数值检查未发现已检查的导出或公式错误，具体退化原因仍待定位。\n\n"
        "COV 的 Real/Shuffled 内容信号应保留，但它没有改善相对 Edge 的排序，不能作为补救性 fusion 搜索的依据。Edge 的 own EO 扩大伴随 Direct 召回下降，不能视为 evidence quality 提升。\n\n"
        "本轮不自动追加 fusion、权重/温度搜索、seed、KD、Uniform、remining、redistillation 或 Stage2；没有验证属性值恢复或端到端 join discovery 改善。结论限于本轮锁定配置。\n")
    write_json(OUT / "EXECUTION_LEDGER.json",{**completion,"G0":"pass","G1":"pass","G2":"completed","G3":"completed",
        "G4":"completed_own_pool_all_nodes","G5":"no_matrix_expansion","scientific_jobs":12,
        "training_process_attempts":18,"status":"completed","reports":[record(OUT/p) for p in ("RESULTS.md","LIMITATIONS.md","NEXT_DECISION.md","SCIENTIFIC_REVIEW.md")],
        "statistics_and_figures": [record(OUT / p) for p in (
            "statistics/main_table.csv", "statistics/per_query.jsonl.gz", "statistics/source_group_bootstrap.jsonl",
            "statistics/eo_admission_retention.csv", "statistics/costs.csv", "statistics/win_loss_tie.jsonl", "statistics/student_admission_decomposition.csv",
            "student/eo_strict_funnel.jsonl.gz", "figures/student_trajectory.png", "figures/student_trajectory.pdf",
            "figures/teacher_trajectory.png", "figures/teacher_trajectory.pdf")],
        "analysis_code": [record(ROOT / "src" / p) for p in (
            "analyze_stage1_r28.py", "plot_stage1_r28.py", "finalize_stage1_r28.py",
            "audit_stage1_r28_training.py", "audit_stage1_r28_evaluations.py", "mmdd_stage1/r26_statistics.py",
            "verify_stage1_r28_statistics.py", "verify_stage1_r28_coverage.py", "verify_stage1_r28_student_export.py", "review_stage1_r28_admission.py")]})


if __name__ == "__main__":
    completion = audit()
    report(completion)
    print(json.dumps({"status":"completed","jobs":12,"evaluations":67,"output":str(OUT)}))
