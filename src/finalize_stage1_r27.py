"""Finalize the R27 reports, execution ledger, and inspectable compact manifest."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

from prepare_stage1_r27 import ROOT, OUT, R26, rows, read_json, write_json, record, sha
from prepare_stage2_r27 import write_rows


def historical_query_deltas() -> dict:
    """Export all shared H/B13 endpoints with the locked 1e-12 W/L/T rule."""
    baseline = {r["query_id"]: r for r in rows(OUT/"score_handoff/B13/baseline_per_query.jsonl.gz")}
    summary = []
    total = 0
    for spec in read_json(OUT/"historical_replay/own_evaluation/MODEL_INVENTORY.json"):
        name = spec["generator_id"]
        directory = OUT/"historical_replay/own_evaluation/rankings"/name
        comparisons = []
        for current in rows(directory/"r27_per_query_metrics.jsonl.gz"):
            reference = baseline[current["query_id"]]
            assert current["positive_target_ids"] == reference["positive_target_ids"]
            endpoints = {}
            for method, metrics in current["metrics"].items():
                for metric, value in metrics.items():
                    old = reference["metrics"][method][metric]
                    delta = value - old
                    endpoints[method+"/"+metric] = {
                        "reference": old, "observed": value, "delta": delta,
                        "outcome": "win" if delta > 1e-12 else "loss" if delta < -1e-12 else "tie",
                    }
            comparisons.append({
                **{k: current[k] for k in ("query_id", "source_table_id", "query_kind")},
                "generator": name, "reference": "historical_B13_R26_own_pool", "endpoints": endpoints,
            })
        assert len(comparisons) == len(baseline) == 1198
        write_rows(directory/"per_query_vs_B13.jsonl.gz", comparisons)
        total += len(comparisons)
        for kind in ("overall", "implicit", "explicit"):
            chosen = [r for r in comparisons if kind == "overall" or r["query_kind"] == kind]
            for endpoint in comparisons[0]["endpoints"]:
                values = [r["endpoints"][endpoint] for r in chosen]
                summary.append({
                    "generator": name, "reference": "historical_B13_R26_own_pool",
                    "query_kind": kind, "endpoint": endpoint, "queries": len(values),
                    "mean_delta": sum(v["delta"] for v in values)/len(values),
                    **{plural: sum(v["outcome"] == label for v in values)
                       for label, plural in (("win", "wins"), ("loss", "losses"), ("tie", "ties"))},
                })
    # Cross-check the independently produced bootstrap table on its shared endpoints.
    indexed = {(r["generator"], r["query_kind"], r["endpoint"]): r for r in summary}
    for result in rows(OUT/"statistics/H_paired_source_bootstrap.jsonl"):
        match = indexed[result["generator"], result["query_kind"], result["endpoint"]]
        assert abs(match["mean_delta"]-result["mean_delta"]) < 1e-12
        assert all(match[k] == result[k] for k in ("wins", "losses", "ties"))
    write_rows(OUT/"statistics/H_query_win_loss_tie.jsonl", summary)
    return {"status": "pass", "query_comparisons": total, "summary_endpoints": len(summary), "tolerance": 1e-12}


def finalize() -> dict:
    hist=OUT/"historical_replay"
    h=read_json(hist/"H_VERDICT.json")
    assert h["status"]=="completed" and h["total_optimizer_updates"]==534
    grouping=read_json(hist/"parity/loss_grouping_diagnosis.json")
    assert grouping["status"]=="pass" and grouping["optimizer_updates"]==0
    h["numerical_parity_level"]="within_locked_tolerance_with_explained_float32_addition_grouping"
    h["loss_grouping_diagnosis"]=record(hist/"parity/loss_grouping_diagnosis.json")
    write_json(hist/"H_VERDICT.json",h)
    first=read_json(hist/"parity/first_divergence.json")
    first["diagnosed_cause"]=grouping["cause"]
    first["zero_update_gradient_diagnostic"]=h["loss_grouping_diagnosis"]
    write_json(hist/"parity/first_divergence.json",first)
    recipe=read_json(hist/"H_RECIPE_DIFF.json")
    recipe["posthoc_arithmetic_grouping_diagnosis"]=h["loss_grouping_diagnosis"]
    write_json(hist/"H_RECIPE_DIFF.json",recipe)
    preflight=hist/"preflight_zero_update/EXECUTION.json"
    pre=read_json(preflight);assert pre["updates"]==0
    pre.update(status="failed",reason="zero-update serialization preflight; resolved before actual C1 training",resolved_stage=str(hist/"C1/seed13/EXECUTION.json"))
    write_json(preflight,pre)
    specs=read_json(ROOT/"mmdd_r26_review/R27_INPUT_LOCK.json")
    a=[]
    for spec in specs["models"]:
        dest=OUT/"score_handoff"/spec["generator_id"]
        assert read_json(dest/"EXECUTION.json")["status"]=="completed"
        assert read_json(dest/"BASELINE_REPLAY.json")["status"]=="pass"
        assert read_json(dest/"diagnostics.json")["retained_scalar_distributions"]["counts"]["queries"]==1198
        duplicate_checked=0
        for row in rows(dest/"raw_candidates_paths.jsonl.gz"):
            for target in row["E_pre_retention"]:
                assert all(p["kind"] in ("direct", "evidence") for p in target["paths"])
                ids=[p["evidence_id"] for p in target["paths"] if p["kind"] == "evidence"]
                assert len(ids)==len(set(ids)),(spec["generator_id"],row["query_id"],target["target_id"])
                duplicate_checked+=len(ids)
        write_json(dest/"PATH_DUPLICATE_AUDIT.json",{"status":"pass","pre_retention_paths_checked":duplicate_checked,"duplicate_q_e_t":0,"dedup_changes_applied":False})
        a.append({"generator":spec["generator_id"],"status":"completed","receipt":record(dest/"EXECUTION.json")})
    assert read_json(OUT/"audit/numerical_spotcheck.json")["status"]=="pass"
    b=read_json(OUT/"witness_panel/TRUTH_AUDIT.json")
    b2=read_json(OUT/"witness_panel/B2_EXECUTION.json")
    assert b["cases"]==32 and b2["status"] in ("completed","not_triggered")
    tests=[]
    for name in ("historical_tests.xml","stage2_tests.xml","final_r27_tests.xml"):
        path=OUT/"audit"/name
        suite=list(ET.parse(path).getroot())[0]
        assert int(suite.attrib["failures"])==0 and int(suite.attrib["errors"])==0
        tests.append({"tests":int(suite.attrib["tests"]),"receipt":record(path)})
    resolved=read_json(hist/"H_RESOLVED_INPUTS.json")
    # Keep actual historical manifests alongside their source hashes.
    for rec in resolved:
        path=Path(rec["path"])
        if path.is_file() and path.suffix==".json":
            dest=hist/"input_manifests"/(rec["logical_id"]+".json")
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    for spec in specs["models"]:
        if spec["checkpoint"]:
            rec=record(Path(spec["checkpoint"]));assert rec["sha256"]==spec["sha256"]
            resolved.append({"logical_id":spec["generator_id"],**rec,"source_stage":"R25/R26 frozen A/B model","schema_summary":"Student checkpoint","locally_recheckable":True})
    from run_stage1_r21 import paths
    ps=paths(ROOT)
    extras={"T0":ROOT/specs["teacher"]["checkpoint"],"corpus":ps["corpus"],"features_manifest":ps["features"]/"manifest.jsonl","dev_queries":R26/"common/dev_queries.jsonl","content_keys":ROOT/"work/stage1_optimization_r10_20260907/taskB_g5/evidence_content_keys.jsonl","column_scorer":ROOT/"work/stage1_optimization_r25_final_20260914/stage2/r25_b13_column_scorer.pt","generation_config":ROOT/"hf_models/Qwen3.5-9B/config.json","generation_template":ROOT/"hf_models/Qwen3.5-9B/chat_template.jinja"}
    for name,path in extras.items():
        resolved.append({"logical_id":name,**record(path),"schema_summary":path.suffix,"source_stage":"frozen shared evaluation input","locally_recheckable":True})
    for spec in specs["models"]:
        receipt=read_json(R26/"rankings"/spec["generator_id"]/"RETRIEVAL_RECEIPT.json")
        assert receipt["signature"]["corpus_sha256"]==sha(extras["corpus"])
        assert receipt["signature"]["feature_manifest_sha256"]==sha(extras["features_manifest"])
    for path in sorted((ROOT/"hf_models/Qwen3.5-9B").glob("*.safetensors")):
        resolved.append({"logical_id":"generation_weight/"+path.name,**record(path),"schema_summary":"safetensors local frozen generator weight","source_stage":"local Qwen3.5-9B","historical_tensor_hash_provenance":"configuration/template/scorer match R26; shard hash recorded now, no historical shard hash asserted","locally_recheckable":True})
    write_json(OUT/"R27_RESOLVED_INPUTS.json",resolved)
    write_json(OUT/"MODEL_LOCK.json",specs)
    for filename in ("R27_CODEX_PROMPT.md","R27_EXPERIMENT_PLAN.md","R27_REVISION_NOTES.md","R27_INPUT_LOCK.json","R27_HISTORICAL_REPLAY_INPUT_LOCK.json","R26_INDEPENDENT_REVIEW.md"):
        dest=OUT/"protocol"/filename;dest.parent.mkdir(exist_ok=True);shutil.copyfile(ROOT/"mmdd_r26_review"/filename,dest)
    runtime={stage:read_json(hist/stage/"seed13/RUNTIME_LOCK.json") for stage in ("C1","C2")}
    write_json(hist/"H_RUNTIME_LOCK.json",runtime)
    (hist/"H_SOURCE_DIFF.patch").write_text("\n".join((hist/stage/"seed13/H_SOURCE_DIFF.patch").read_text() for stage in ("C1","C2")))
    for stage in ("C1","C2"):
        path=hist/stage/"seed13/EXECUTION.json";rec=read_json(path);rec["evaluated"]=True;rec["evaluation_summary"]=str(hist/"own_evaluation/SUMMARY.json");write_json(path,rec)
    execution=read_json(OUT/"witness_panel/EXECUTION.json")
    execution.update(status="completed",evaluated=True,truth_audit=record(OUT/"witness_panel/TRUTH_AUDIT.json"),deletion=record(OUT/"witness_panel/B2_EXECUTION.json"))
    write_json(OUT/"witness_panel/EXECUTION.json",execution)
    b["status"]="completed_with_unknown_truth_retained";b["deletion_status"]=b2["status"];write_json(OUT/"witness_panel/TRUTH_AUDIT.json",b)
    complete={"planned":True,"implemented":True,"executed":True,"evaluated":True,"status":"completed"}
    ledger={"version":"R27-rev2-B13-EXACT","P0":dict(complete,models=9),"H0":complete,"H1":dict(complete,updates=356),"H2":dict(complete,updates=178),"A0":dict(complete,models=9,queries_per_model=1198),"A1":dict(complete,models=9,queries_per_model=1198),"B0":dict(complete,queries_per_generator=64,candidate_budget=18,new_generation=0),"B1":dict(complete,cases=32,first_requests=execution["first_requests"],physical_attempts=execution["physical_attempts"]),"B2":dict(complete,status=b2["status"],deleted_cells=b2.get("deleted_cells",0)),"additional_training":{"planned":False,"executed":False,"status":"not_planned"},"new_C50_pilot":{"planned":False,"executed":False,"status":"not_planned"},"limitations":["historical complete runtime provenance unknown","unresolved independent truth/grounding explicitly unknown","B2 ranks are a local diagnostic with fixed Direct competitors, not full system rankings"]}
    write_json(OUT/"EXECUTION_LEDGER.json",ledger)
    write_json(OUT/"audit/own_identity.json",{"status":"pass","models":[read_json(OUT/"score_handoff"/s["generator_id"]/"INPUT_IDENTITY.json") for s in specs["models"]]})
    write_json(OUT/"audit/baseline_replay.json",{"status":"pass","models":[{"generator":s["generator_id"],"receipt":record(OUT/"score_handoff"/s["generator_id"]/"BASELINE_REPLAY.json")} for s in specs["models"]]})
    write_json(OUT/"audit/single_factor_assertions.json",{"status":"pass","queries":9*1198,"files":[record(OUT/"score_handoff"/s["generator_id"]/"single_factor_assertions.jsonl.gz") for s in specs["models"]]})
    write_json(OUT/"teacher_scores/TEACHER_RECEIPT.json",{"status":"reused_frozen_scores","source_identity":read_json(R26/"teacher/CACHE_IDENTITY.json"),"score_tables":[record(OUT/"score_handoff"/s["generator_id"]/"teacher_scores_refs.jsonl.gz") for s in specs["models"]],"online_budget":100,"offline_cache_reuse_does_not_make_online_cost_zero":True})
    summary=read_json(hist/"own_evaluation/SUMMARY.json")
    write_json(OUT/"audit/H_query_comparisons.json", historical_query_deltas())
    with (OUT/"statistics/metrics.csv").open() as f:
        all_metrics=[r for r in csv.DictReader(f) if not r["generator"].startswith("H-")]
    for name,s in summary.items():
        for source in ("retrieval","teacher"):
            for kind,methods in s[source].items():
                for method,values in methods.items():
                    if not isinstance(values,dict):continue
                    for metric,value in values.items():
                        all_metrics.append({"generator":name,"query_kind":kind,"method":source+"/"+method,"metric":metric,"value":value})
    with (OUT/"statistics/metrics.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=["generator","query_kind","method","metric","value"]);writer.writeheader();writer.writerows(all_metrics)
    decisions=read_json(OUT/"statistics/decision_table.json")
    transfers=read_json(OUT/"statistics/eo_strict_transfers.json")
    reports=["# R27-rev2 结果", "", "## H：历史 fresh C1→C2 可以重现", "",f"本轮从历史 PCA init 出发执行 C1 356 步，再以本轮 step356 为 parent、fresh AdamW 执行 C2 178 步。总计 534 次更新，没有新增 Teacher 训练、训练候选重挖、历史训练 logits 重算或额外训练臂。10 个归档节点全部达到 tensor exact；仅移除无计算影响的现代默认 metadata 后，文件 SHA 全部匹配历史归档。逐步 loss 分量最大绝对误差为 {h['max_per_step_loss_component_error']:.3g}。原 H checkpoint 永久保留。", "", "初始 preflight 的差异是 R12 缺少 projection_mode 默认字段，以及历史 tensor device 为 cuda:1。修正独立导出的序列化后，第 0 步 SHA 对齐；此前没有执行 optimizer update。R26 历史 parent 出发的 C2 恢复文件也重新 hash 验证过，本轮并未用它替代 fresh parent。", "", "recipe replay、state parity 与 retrieval parity 分开报告；历史完整 runtime/依赖源码快照无法逐项还原，不能称环境 provenance 全部 exact。H 成功只证明整份历史 recipe 可重现，不证明预算、初始化、closure 或完整 path bag 中哪个因素造成优势。", "", "| H 节点 | Direct ANN R10 | Equal C100+T0 R10 | implicit T0 R10 | explicit T0 R10 | Uraw | strict EO / C100 / T0@10 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name,s in summary.items():
        reports.append(f"| {name} | {100*s['retrieval']['overall']['D100_ANN']['recall@10']:.4f}% | {100*s['teacher']['overall']['BT100_T0']['recall@10']:.4f}% | {100*s['teacher']['implicit']['BT100_T0']['recall@10']:.4f}% | {100*s['teacher']['explicit']['BT100_T0']['recall@10']:.4f}% | {100*s['retrieval']['overall']['U']['raw_recall']:.4f}% | {s['strict_EO_pairs']} / {s['strict_C100_pairs']} / {s['strict_T0_10_pairs']} |")
    reports += ["",f"H final 与历史 B13 的 1,198-query 独立排名比较：`{json.dumps(h['retrieval_equality_queries'])}`。共同 QT scalar 最大差异 {h['max_common_QT_score_error']:.3g}。新的 ANN index 独立构建，membership 差异不能误称为已确认的 tensor 分叉。C1 step356/C2 step0 的推理参数相同，stage anchor 不同，不视为两个独立模型或 seed。", "", "## A：单改 retained E 排序未改善部署", "", "A0 已在九个模型上复现所有归档 Recall（容差 1e-10）。A1 严格固定 D/E/U/M、retained witnesses、QE/ET/path scalar、row support、T0 字典，仅替换 E 排序。每模型 1,198 个 query 的原始 scalar 与排名均随包交付。", "", "| 冻结模型 | A0 T0 R10 | A1 T0 R10 | ΔR10 pp [source-paired 95% CI] | strict C100 A0→A1 |", "|---|---:|---:|---:|---:|"]
    reports.insert(reports.index("## A：单改 retained E 排序未改善部署"),"C2 首个标量差异在 step2：4.768×10⁻⁷，来自 float32 加法分组。历史为 SUP_total + 0.3×KD_total + anchor；当前 helper 先在 D/E 分支各自组合 SUP/KD 后相加。按历史分组重算已记录项，178 步 total loss 全部与归档逐bit一致；在相同 C2 step0 上对两个完整固定 batch 做零更新检查，两种分组的所有参数梯度逐bit一致。原训练 trace 和 checkpoint 没有被修改。E2 按预锁容差通过，不宣称当前记录的 loss scalar 原本逐bit一致。\n")
    for spec in specs["models"]:
        name=spec["generator_id"];dest=OUT/"score_handoff"/name
        base=read_json(dest/"BASELINE_REPLAY.json")["metrics"]["overall"]["BT100_T0"]["recall@10"]
        decision=next(d for d in decisions if d["generator"]==name)["T0_R10"]
        funnel=next(d for d in transfers if d["generator"]==name)["counts"]
        ci=decision["bootstrap_95ci"]
        reports.append(f"| {name} | {100*base:.4f}% | {100*(base+decision['mean_delta']):.4f}% | {100*decision['mean_delta']:+.4f} [{100*ci[0]:+.4f}, {100*ci[1]:+.4f}] | {funnel['overall/C100/A0']}→{funnel['overall/C100/A1']} |")
    reports += ["", "所有九个模型的 T0 R10 和 strict EO C100 admission 均下降。B13 的 strict C100 为 101→65，T0 Top10 为 75→42；不能把 LSE 训练分数直接替换为部署 E 排序当作修复。source-group bootstrap 使用 10,000 次、seed=260914；既有双 seed 家族先按 query 平均后整 source 重采样。固定 seed13 的 H 不声称新增 seed 稳健性。", "", "overall/implicit/explicit、R10/R20/R50、D/E/U/M、RawUnionRecall、ANN/exact、U-only 与 M-only、EO_ANN/EXACT/STRICT/FIXED_RAW 的完整机器表见 statistics 与逐 query 文件；原 Direct100+T0、QT-over-U 等控制保留于 baseline 指标。", "", "## B：值确有局部作用，但覆盖和属性仍不足", "", "旧 Raw/B13 的 64-query own C18 三条件原样回放，B13 T0 R9=39.0625%、Stage2 Real R9=32.2917%。既有 Raw/B13 的 16/13 source-confirmed 新 cell 删除结果随包保存。这是零新生成回放，C18 没有改名为 C50。", "", f"新 panel 按原 B13 的 207 对 EO_STRICT 分层、hash 锁定 32 病例：16 已入 A0 C100、16 未入。Real/NoE 共 {execution['first_requests']} 次首次请求、{execution['physical_attempts']} 次物理 attempt；53 个相同路由机会/条件，97 个 valid_value、9 个 valid_abstain，无 parse failure、无重试。所有病例均保留分母。", "", "预测列 20/32 与 GT 属性一致，12/32 错误。路由 coverage 上限为 0.2/0.4/0.6/0.8 的病例分别是 19/7/4/2；26/32 病例即便所有路由值正确也只覆盖≤2/5 行。独立 source truth：Real correct/incorrect/unknown=14/1/38；NoE=13/4/36。未核定的文本别名保留 unknown，不按字符串差异硬判 wrong。", "", "Real 相对 NoE 有 3 个独立确证正确新 cell（2 个病例），其中 1 个有明确 supplied-span grounding：Chandigarh 2001–2011 增长率 17.10%。另两个正确值的裁剪片段语义联系不足，grounding 保留 unknown。部分同样被 NoE 正确输出的球队名称，图像 crop 只含局部人脸/空白；不能据 source-exact 就声称图像证据提供了答案。", "", "删除 3 个正确新值并实际重新编码、重新语义验证：两病例 coverage 0.4→0、0.4→0.2；局部诊断 rank 54→73、35→43。第一例原本已在 C100；第二例是 outside-deployment-queue 局部诊断。排名比较使用固定 Direct verification 竞争者，不是全量 Stage2 recovery 的系统排名，不能当作端到端 Recall 改善。", "", "## 当前研究判断", "", "**首要瓶颈：可用证据到正确属性、足够 query 行覆盖的恢复链。** 当前列错误和稀疏 routing 已形成可观测限制。B2 表明至少这些正确值会影响真实验证分数，不能再笼统说后端完全忽略证据。**第二竞争解释：候选准入与 Direct/coverage 竞争仍压制稀少的正确证据。** B13 的 strict EO 在 C100 仅保留 101/207；换 retained LSE 更差。局部 rank 变化尚未证明正式系统 TopK 收益。", "", "停止扩大 shuffle、KD/Uniform/LSE 训练矩阵和第三种部署融合搜索；保留历史 B13 为对照。H 为后续单因素实验提供可信起点，但完整 bag→前8条、预算356→659、初始化、closure 插入必须另行预注册且逐项控制。本轮未追加这些训练，也未启动新 C50 pilot。"]
    (OUT/"RESULTS.md").write_text("\n".join(reports)+"\n")
    (OUT/"LIMITATIONS.md").write_text("# 限制与证据边界\n\n- 历史 runtime 与完整原始 dependency closure 缺独立快照；当前 closure、输入 hash、534 步 loss 和10个归档节点匹配均已保存。不得把环境 provenance 写成全部 exact。\n- Checkpoint 原文件与默认 metadata 导出的文件分别保存；未修改 tensor 追 SHA。C1 中间原始权重在服务器已被清理，比较使用历史归档 SHA；C2 另有实际恢复文件逐 tensor 对比。\n- H 自身 ANN 索引重建可产生 membership 差异；按 exact score 和 query-level 集合分别判断，不能混为训练误差。\n- H 只有 seed13；既有双 seed A 家族先按 query 平均。不由 H 成功归因到单个历史因素。\n- B 是使用 qrels 选择的机制 panel，不报告代表总体的 Recall/precision；未入队病例仅局部诊断。\n- B 的列选择在单(q,t)上运行原 reader/scorer；列内 argmax 不受跨 target softmax 公共分母影响，但不将单pair priority 当线上 recovery priority。\n- B2 的 rank 是固定 Direct 竞争者的局部 counterfactual，其他候选没有新增 recovery；不是新的 C100/C50 端到端 pilot。\n- 很多独立值 truth、row/entity grounding 与别名仍 unknown，不能当成错误或成功。图像 crop 的 source-exact 正确值不自动具有 grounding。\n- 文本 grounding 的人工式片段核对由执行代理完成，引用与判定理由均公开；独立正确性来自 source table，而非生成模型或 judge 自评。\n- 统计均保留原分母；source-paired CI 不能纠正诊断 panel 的选择性。\n")
    (OUT/"NEXT_DECISION.md").write_text("# 下一步决策\n\n1. 保留历史 B13 与 D1 部署。九个模型的 retained-LSE 替换均为负，不继续搜索第三种融合公式。\n2. 将有限后续预算优先用于列/行/witness 质量及覆盖诊断；先说明哪些值可由 supplied evidence 支持，哪些来自模型知识。\n3. 若开展训练因果实验，以本轮可重现的历史 recipe 为基线，一次只改变一个因素。完整bag→前8条时须区分只改 Student bag 与同时改 Teacher bag；356→659 需要共同冻结前缀，不能更换 schedule 后归因预算。\n4. 继续停止 shuffle seed 搜索、KD/Uniform/LSE 大矩阵和自动扩大的 C50 pilot。新增训练或正式端到端实验需另行立项，本轮不执行。\n\n首要瓶颈：正确属性与足够行覆盖的证据恢复。第二竞争解释：候选准入及 Direct coverage 竞争。H 的复现成功与 A/B 的机制证据分开陈述。\n")
    # Snapshot only source and explicit non-secret protocol files.
    for source in (ROOT/"src").rglob("*.py"):
        dest=OUT/"source_snapshot"/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    dest=OUT/"source_snapshot/tests/test_stage1_r27.py";dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/"tests/test_stage1_r27.py",dest)
    write_json(OUT/"audit/COMPLETION_AUDIT.json",{"status":"pass","H":{"nodes":10,"updates":534,"own_evaluation_nodes":len(summary),"reference":record(hist/"H_VERDICT.json")},"A":{"models":9,"queries_per_model":1198,"paired_bootstrap_contrasts":252},"B":{"legacy_queries_per_model":64,"panel_cases":32,"first_requests":execution["first_requests"],"deletion":b2},"tests":tests,"unknown_scientific_results_are_retained":True,"scope_exclusions":"no additional training, no seed29 H, no new C50 pilot","required_deliverables":["RESULTS.md","LIMITATIONS.md","NEXT_DECISION.md","historical_replay/H_VERDICT.json","R27_RESOLVED_INPUTS.json","EXECUTION_LEDGER.json"]})
    return {"status":"reports_finalized","tests":sum(r["tests"] for r in tests),"H":h["historical_fresh_reproduction_status"],"models":len(a)}


def package() -> dict:
    assert read_json(OUT/"audit/COMPLETION_AUDIT.json")["status"]=="pass"
    omitted=[];included=[]
    excluded_names={"PACKAGE_MANIFEST.json","OMITTED_SERVER_FILES.json"}
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.name in excluded_names or "__pycache__" in path.parts:
            continue
        rel=path.relative_to(OUT)
        if path.suffix in (".hnsw",".sqlite") or (path.suffix==".pt" and path.stat().st_size>10*1024*1024):
            omitted.append({**record(path),"reason":"large generated checkpoint/index/cache; all necessary scalar ranks and parity receipts included"})
        else:
            included.append({"relative_path":str(rel),**record(path)})
    external=read_json(OUT/"R27_RESOLVED_INPUTS.json")
    write_json(OUT/"OMITTED_SERVER_FILES.json",{"generated_large_files":omitted,"external_input_files":external,"additional_feature_file_receipts":"historical_replay/consumed_feature_files.jsonl.gz","additional_R26_index_receipts":"audit/own_identity.json","note":"Large weights/features are server-local, with paths/hashes preserved. All required scalar raw ranks and actual generation replies are included."})
    included.append({"relative_path":"OMITTED_SERVER_FILES.json",**record(OUT/"OMITTED_SERVER_FILES.json")})
    manifest={"format_version":1,"run":str(OUT),"status":"completed","included_files":included,"included_bytes":sum(r["bytes"] for r in included),"omitted_generated_files":len(omitted),"manifest_self_hash":"stored in external delivery receipt; not recursively included"}
    write_json(OUT/"PACKAGE_MANIFEST.json",manifest)
    archive=ROOT/"R27_results_compact_20260915.tar.gz"
    with tarfile.open(archive,"w:gz",compresslevel=1) as tf:
        for rec in included:
            path=OUT/rec["relative_path"]
            assert sha(path)==rec["sha256"],path
            tf.add(path,arcname="R27/"+rec["relative_path"],recursive=False)
        tf.add(OUT/"PACKAGE_MANIFEST.json",arcname="R27/PACKAGE_MANIFEST.json",recursive=False)
    verified=0
    expected={"R27/"+r["relative_path"]:r["sha256"] for r in included}
    with tarfile.open(archive,"r:gz") as tf:
        for member in tf:
            if member.name not in expected:continue
            handle=tf.extractfile(member);digest=hashlib.sha256()
            for block in iter(lambda:handle.read(8*1024*1024),b""):digest.update(block)
            assert digest.hexdigest()==expected[member.name],member.name
            verified+=1
    assert verified==len(included)
    delivery={"status":"verified","archive":record(archive),"manifest":record(OUT/"PACKAGE_MANIFEST.json"),"verified_archive_members":verified,"results":str(OUT/"RESULTS.md")}
    write_json(ROOT/"R27_results_compact_20260915.DELIVERY.json",delivery)
    return delivery


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--package",action="store_true")
    print(json.dumps(package() if p.parse_args().package else finalize()))
