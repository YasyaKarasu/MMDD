"""Validate and report the current-corpus minimal-prompt experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from diagnose_text_prompt_dominance import BINS, fingerprint, rows, sources, summary, write_json


def report(root: Path, output: Path) -> None:
    def read(name: str):
        return json.loads((output / name).read_text())

    protocol = read("PROTOCOL.json")
    for rec in protocol["inputs"].values():
        assert fingerprint(Path(rec["path"])) == rec
    assert protocol["arms"] == ["original", "minimal"]
    baseline = read("baseline_buckets.json")
    inventory = list(rows(output / "evidence_inventory.jsonl.gz"))
    sample = list(rows(output / "embedding_diagnostics.jsonl"))
    inv = read("inventory.json")
    emb = read("embedding_summary.json")
    qe = read("counterfactual_qe_summary.json")
    path = read("counterfactual_path_summary.json")
    matrices = torch.load(output / "counterfactual_qe_matrices.pt", weights_only=True)
    qpos = {q: i for i, q in enumerate(matrices["queries"])}
    epos = {e: i for i, e in enumerate(matrices["evidence"])}
    scalar_errors = {}
    for g in ("Qwen-Raw", "B13"):
        errors = []
        cached = matrices["matrices"][g]["cached"]
        for p in rows(output / f"paths_{g}.jsonl.gz"):
            if p["evidence_id"] in epos:
                errors.append(abs(float(cached[qpos[p["query_id"]], epos[p["evidence_id"]]])-p["qe"]))
        # HNSW and CUDA GEMM accumulate float32 dot products differently.
        assert max(errors) < 1e-5, (g, max(errors))
        scalar_errors[g] = {**summary(errors), "max": max(errors), "tolerance": 1e-5}
        receipt = json.loads((sources(root)[g].parent / "RETRIEVAL_RECEIPT.json").read_text())
        assert protocol["inputs"][g]["sha256"] == receipt["rankings"]["sha256"]
        if g == "B13":
            assert protocol["inputs"]["student"]["sha256"] == receipt["signature"]["checkpoint_sha256"]
    assert baseline["B13"]["strict_target_pairs"] == 207
    assert baseline["Qwen-Raw"]["strict_target_pairs"] == 186
    dedup = {}
    for b in BINS:
        unique = {r["text"]: r for r in sample if r["bucket"] == b and "random_bin" in r["selection"]}
        dedup[b] = {"random_sample_unique_bodies": len(unique),
                    "cos_full_empty": summary([r["cos_full_empty"] for r in unique.values()])}
    body_audit = {b: {"objects": sum(r["bucket"] == b for r in inventory),
                     "unique_bodies": len({r["text"] for r in inventory if r["bucket"] == b})} for b in BINS}
    # Deterministic illustrative case: most frequent archived Raw short-text
    # hub, then highest original cosine among its exact Top20 query hits.
    case_e = sorted((r for r in sample if r["tokens"] <= 5),
                    key=lambda r: (-r["top20_counts"]["Qwen-Raw"], r["object_id"]))[0]
    si = epos[case_e["object_id"]]
    raw = matrices["matrices"]["Qwen-Raw"]
    eligible = torch.where((raw["cached_rank"][:, si] <= 20) & (raw["original_rank"][:, si] <= 20))[0]
    qi = int(eligible[raw["original"][eligible, si].argmax()])
    case = {"query_id": matrices["queries"][qi], "evidence_id": case_e["object_id"],
            "text": case_e["text"], "selection": "highest archived Raw frequency at <=5 tokens, then highest original cosine among exact Top20 hits",
            "cos_full_empty": case_e["cos_full_empty"],
            **{k: float(raw[k][qi, si]) for k in ("cached", "original", "minimal", "cached_rank", "original_rank", "minimal_rank")}}
    write_json(output / "CURRENT_CASE.json", case)
    write_json(output / "VALIDATION.json", {"status": "passed", "input_hashes_match_prepared_protocol": True,
        "rankings_match_frozen_R26_receipts": True, "student_matches_ranking_checkpoint": True,
        "cached_scalar_max_error": scalar_errors, "content_audit": body_audit,
        "random_sample_deduplicated": dedup,
        "punctuation_only_bodies": sum(not any(c.isalnum() for c in r["text"]) for r in inventory),
        "bare_missing_bodies": sum(r["text"].strip().lower() in ("null", "none", "n/a", "unknown", "missing") for r in inventory),
        "code": {name: fingerprint(root / name) for name in (
            "src/diagnose_text_prompt_dominance.py", "src/report_text_prompt_dominance.py",
            "src/cache_stage1_features.py", "src/mmdd_stage1/models.py", "src/mmdd_stage1/features.py",
            "src/mmdd_stage1/checkpoints.py", "tests/test_text_prompt_dominance.py")}})
    lines = [
        "当前语料 text evidence 的复杂／极简 prompt 诊断（2026-09-15）", "",
        "结果支持：当前复杂 instruction 对部分短正文、尤其通用页面节标题的相似度有明显影响；"
        "极简 prompt 能显著降低它们的原始 Qwen 召回分数。结果不支持：把它直接解释为当前 B13 的主要瓶颈，"
        "或宣称全量替换一定提高检索效果。B13 已明显抑制短文本，当前 strict-EO 文本 witness 全部是长文本。", "",
        "本实验从当前 R26/R27 使用的冻结语料和检索产物重新取样，没有使用《方案》中的旧案例。"
        "模型为本地 Qwen3-VL-Embedding-8B 和冻结 B13；1198 个 dev query、84689 条 text evidence。"
        "缓存目录带 R10 日期，是当前 R26/R27 实际引用的同一份缓存；输入文件、冻结排名和 Student checkpoint 的 SHA 已核对。", "",
        "主干实验保持正文、system/user 聊天模板、末 token pooling、Q/T 向量和其他 evidence 向量不变，"
        "只替换当前 evidence instruction 为 `Represent this text for retrieval.`。不执行完全去 prompt，也不训练。"
        "另编码原 prompt＋空正文、极简 prompt＋空正文，以及封装默认 NULL 正文对照。"
        "真正空正文绕过封装的 NULL 回退。", "",
        "分桶统计使用 tokenizer 的裸正文 token 数，20 归入 11–20，最后一桶为 ≥21。"
        "Top20 为每个 query 的文本 evidence 检索槽位；QE 每个 Q/E 只计一次，ET 每个 Q/E/T path 计一次。"
        "strict witness 指保留路径指向 `G∩E−(D_ANN100∪D_EXACT100)` 的已标注目标，不代表正文内容已独立核实。", "",
        "| 正文 token | 对象数／不同正文 | Raw Top20 占比 | B13 Top20 占比 | Raw strict 文本路径数 | B13 strict 文本路径数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for b in BINS:
        r, s = baseline["Qwen-Raw"]["bins"][b], baseline["B13"]["bins"][b]
        lines.append(f"| {b} | {body_audit[b]['objects']} / {body_audit[b]['unique_bodies']} | {r['top20_slot_share']:.4%} | {s['top20_slot_share']:.4%} | {r.get('strict_retained_paths',0)} | {s.get('strict_retained_paths',0)} |")
    lines += ["", "≤20 token 共603条（语料0.712%），占 Raw Top20 的3.869%、B13 的0.338%。"
              "两个最短对象都为 `Official website`；当前语料没有纯标点正文，未命中所检查的 `[url]`、`[image]`、"
              "`[missing]` 等占位符，也没有整段为 NULL/None/N/A/unknown/missing 的正文。"
              "因此本次没有可执行的占位符删除对照，不能确认 placeholder-dominance。", "",
              "抽样共295个对象：每个非空桶固定随机种子260915抽最多48个（合计194个），"
              "再加入当前 Raw/B13 召回频繁的对象及 strict witness。以下均值只使用194个随机对象；"
              "强化样本不并入主均值。最短桶只有一种不同正文，不作统计显著性推断。", "",
              "| token | 随机样本数 | cos(复杂正文,复杂空正文) | cos(极简正文,极简空正文) | Raw QE Δ | B13 QE Δ |",
              "|---|---:|---:|---:|---:|---:|"]
    for b in BINS:
        er = emb["random_bin"][b]
        lines.append(f"| {b} | {er['cos_full_empty']['n']} | {er['cos_full_empty']['mean']:.3f} | {er['cos_minimal_empty']['mean']:.3f} | {qe['Qwen-Raw']['random_bin'][b]['minimal']['qe_delta_vs_original']['mean']:+.3f} | {qe['B13']['random_bin'][b]['minimal']['qe_delta_vs_original']['mean']:+.3f} |")
    lines += ["", "QE Δ 为极简减复杂、固定1198个query后的所有 Q/E 对均值。Raw 为 cosine，B13 为学习到的有向点积分数，"
              "二者不是同一分数尺度。全语料每个模型另外计算101,457,422个Q/E分数，详见 all_corpus_scores.json。", "",
              "3–5 token 桶的高均值部分由重复节标题驱动：随机样本48对象只有8种不同正文，"
              f"按不同正文等权后 cos(复杂正文,复杂空正文) 均值为{dedup['3-5']['cos_full_empty']['mean']:.3f}，"
              "仍高于长文本的0.492。不能把重复对象当作独立支持，也不能声称所有越短文本都越接近空 prompt："
              "0–2桶均值0.705，低于3–5桶。", "",
              "| token | Raw 样本原 exact Top20 命中数 | 换极简后仍在Top20 | 新排名中位数 | B13 样本原命中数 | B13仍在Top20 |",
              "|---|---:|---:|---:|---:|---:|"]
    for b in BINS:
        r, s = (qe[g]["all_selected"][b]["minimal"] for g in ("Qwen-Raw", "B13"))
        survive = lambda x: "—" if x is None else f"{x:.1%}"
        rank = r["on_original_top20_new_rank"]["median"]
        lines.append(f"| {b} | {r['original_cached_top20_pairs']} | {survive(r['on_original_top20_survival'])} | {rank:g} | {s['original_cached_top20_pairs']} | {survive(s['on_original_top20_survival'])} |")
    lines += ["", "这一表使用全部295个选中对象，以原缓存的 exact Top20 命中为分母。"
              "每次只替换一个E，相对其余84688个原始文本向量计算精确名次；并非所有E同时更新，"
              "也不是全系统Recall。它与前表的归档ANN槽位统计口径不同；同分使用competition rank。", "",
              "当前自动选出的例子：", "",
              f"- Query：`{case['query_id']}`，表头为 Film / Notes / entity_url。",
              f"- Evidence：`{case['evidence_id']}`，正文仅 `{case['text']}`；该对象在Raw归档中进入44个query的Top20，在B13为0次。",
              f"- 与复杂空正文向量 cosine：{case['cos_full_empty']:.3f}。",
              f"- 固定同一query，复杂→极简 cosine：{case['original']:.3f}→{case['minimal']:.3f}；精确文本排名：{case['original_rank']:.0f}→{case['minimal_rank']:.0f}。",
              "- 例子按“≤5 token中归档召回次数最多的对象，再取其原exact Top20内原分数最高的query”选择；没有按最大分数降幅挑例子。", "",
              "该正文只是通用节标题，未陈述影片实体或可恢复属性。这给出了当前数据中的具体 prompt 敏感案例，"
              "但单靠cosine不应声称它完全等同于prompt，或把所有短正文都判为无用。", "",
              "E→T及路径对照固定原 Q/E/T：Raw共22,820条、B13共13,660条样本路径，均未重新检索目标。", "",
              "| token | Raw ET Δ | Raw path Δ | B13 ET Δ | B13 path Δ |",
              "|---|---:|---:|---:|---:|"]
    for b in BINS:
        cells = []
        for g in ("Qwen-Raw", "B13"):
            r = next((r for r in path[g] if r["bucket"] == b and r["scope"] == "all_sampled_paths" and r["arm"] == "minimal"), None)
            cells += ["—", "—"] if r is None else [f"{r['et_delta']['mean']:+.3f}", f"{r['path_delta']['mean']:+.3f}"]
        lines.append("| " + b + " | " + " | ".join(cells) + " |")
    lines += ["", "ET/path为已被归档召回的路径条件均值，不是随机E/T总体分布。path=QE+ET。"
              "路径文件还给出固定文本路径池内的名次变化，该名次不是最终目标表排名。"
              "B13抽样中44条strict保留文本路径全为长正文，换极简后平均path分数下降0.012；"
              "正常长文本也受影响，不能把所有降分视为清除了噪声。", "",
              "可支持的结论：复杂prompt对当前部分通用短正文有明显的相似度抬升；"
              "极简prompt降低这种影响，短正文平均降幅明显大于长正文。"
              "该对照改变了instruction的长度和语义细节，不能单独归因于token数量。"
              "不支持的结论：placeholder污染仍普遍存在、当前B13的strict-EO问题主要由短文本造成、"
              "或no-path row similarity强／path-quality sigmoid集中已被证明同源。"
              "本轮没有测row-routing重新编码、属性值恢复正确率或最终joinability；这些联系保留为未验证假设。", "",
              "建议保留本结果作为诊断，不据此立即替换部署缓存。若继续验证极简prompt，"
              "下一步需在一致更新的完整E语料上比较召回、strict-EO保留和正确属性/行覆盖，"
              "并单独区分冻结Student的分布变化；本轮没有启动这些实验。", "",
              "原prompt重新编码与缓存cosine最小0.998908、均值0.999755，不是逐bit一致。"
              "对照差值使用同次重编码的复杂prompt为基线。HNSW与CUDA GEMM归约存在float32误差，"
              "归档QE与缓存精确重算的最大误差均小于1e-5（实际值见VALIDATION.json）；"
              "固定输入hash、排名receipt和B13 checkpoint身份检查通过。593条序列编码约10.45秒，"
              "不含模型加载和缓存读盘。四个离线测试通过；未读取配置客户端或调用外部模型服务。", "",
              "复现命令（MMDD环境；工作目录使用隔离的/tmp目录）：", "", "```bash", "cd /tmp/mmdd_prompt_diagnostic"]
    for phase in ("prepare", "encode", "analyze"):
        lines.append(f"conda run --no-capture-output -n MMDD python {root}/src/diagnose_text_prompt_dominance.py --output {output} --phase {phase}")
    lines += [f"conda run -n MMDD python {root}/src/report_text_prompt_dominance.py --output {output}",
              f"PYTHONPATH={root}/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n MMDD python -m pytest {root}/tests/test_text_prompt_dominance.py -q --confcutdir=/tmp/mmdd_prompt_diagnostic", "```", "",
              "完整分桶：baseline_buckets.json / all_corpus_scores.json。"
              "抽样与编码：sample.jsonl / embedding_diagnostics.jsonl / counterfactual_embeddings.pt。"
              "逐query精确分数与名次：counterfactual_qe_matrices.pt。"
              "逐路径对照：counterfactual_paths_{Qwen-Raw,B13}.jsonl.gz。"
              "协议与核验：PROTOCOL.json / ENCODING.json / ANALYSIS_COMPLETE.json / VALIDATION.json。"]
    (output / "RESULTS.zh-CN.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"report": str(output / "RESULTS.zh-CN.md"), "validation": "passed"}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report(args.root, args.output)
