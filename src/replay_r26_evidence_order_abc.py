"""CPU-only frozen R26 replay: coverage vs retained LSE vs pre-retention LSE.

No training, retrieval, feature/model loading, environment loading, or cache writes.
C uses pre-retention scores restricted to the SAME retained target membership.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from mmdd_stage1.r26_metrics import fuse_channels, query_metrics
from mmdd_stage1.r26_statistics import source_cluster_comparison


GENERATORS = ["B13", "Qwen-Raw", "PCA"] + [
    f"{model}/seed{seed}/step{step}"
    for model, step in (("R25-C1", 659), ("R26-O-SUP", 178), ("R26-E-GRAPH", 178))
    for seed in (13, 29)
]
ARMS = ("A_coverage", "B_retained_lse", "C_pre_lse")
KS = (10, 20, 50, 100)


def read_rows(path: Path):
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as handle:
        for line in handle:
            yield json.loads(line)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def record(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)}


def lse(paths: list[dict]) -> float:
    values = [p["path_score"] for p in paths]
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("LSE requires nonempty finite path scores")
    top = max(values)
    return top + math.log(sum(math.exp(v - top) for v in values))


def replay_arms(row: dict, teacher_scores: dict[str, float]) -> dict:
    """Change E order only; preserve witnesses and use target-ID tie breaking."""
    retained = row["E_paths"]
    pre = {r["target_id"]: r for r in row["E_pre_retention"]}
    scores = {
        ARMS[0]: {r["target_id"]: r["evidence_score"] for r in retained},
        ARMS[1]: {r["target_id"]: lse(r["retained_paths"]) for r in retained},
        ARMS[2]: {r["target_id"]: lse([p for p in pre[r["target_id"]]["paths"]
                                     if p["kind"] == "evidence"]) for r in retained},
    }
    result = {}
    for arm, score in scores.items():
        evidence = sorted(retained, key=lambda r: (-score[r["target_id"]], r["target_id"]))
        equal = fuse_channels(row["D100_ANN"], evidence)["rankings"]["Equal"]
        c100 = equal[:100]
        final = sorted(c100, key=lambda t: (-teacher_scores[t], t))
        result[arm] = {"E": [r["target_id"] for r in evidence], "E_scores": score,
                       "Equal": equal, "C100": c100, "T0": final}
    return result


def validate_paths(row: dict) -> dict:
    """Recompute both LSEs and check exact retained-path provenance."""
    pre = {r["target_id"]: r for r in row["E_pre_retention"]}
    counts = Counter()
    truth = set(row["positive_target_ids"])
    for before in pre.values():
        paths = [p for p in before["paths"] if p["kind"] == "evidence"]
        for p in paths:
            assert math.isclose(p["path_score"], p["query_evidence_score"] + p["evidence_target_score"], abs_tol=1e-10)
        assert math.isclose(lse(paths), before["evidence_score"], abs_tol=1e-10)
        counts["pre_paths"] += len(paths)
    for after in row["E_paths"]:
        target = after["target_id"]
        before = pre[target]
        selected = after["selected_evidence_ids"]
        assert 0 < len(selected) <= 4 and len(set(selected)) == len(selected)
        assert after["paths"] == before["paths"]
        assert after["retained_paths"] == [p for p in before["paths"] if p.get("evidence_id") in selected]
        assert set(after["routed_rows"]) == set(selected)
        assert math.isclose(lse(after["retained_paths"]), after["retained_path_lse"], abs_tol=1e-10)
        paths = [p for p in before["paths"] if p["kind"] == "evidence"]
        best_kept = max(p["path_score"] for p in after["retained_paths"])
        best_pre = max(p["path_score"] for p in paths)
        for group in ("all", "qrel_positive" if target in truth else "unlabeled"):
            counts[group + "/targets"] += 1
            counts[group + "/best_path_score_lost"] += best_kept < best_pre - 1e-10
            counts[group + "/routed_row_count"] += len(set(after["routed_rows"].values()))
            counts[group + "/retained_paths"] += len(after["retained_paths"])
    retained_ids = {r["target_id"] for r in row["E_paths"]}
    counts["targets_dropped_by_retention"] = len(pre.keys() - retained_ids)
    counts["qrel_positives_dropped_by_retention"] = len(truth & (pre.keys() - retained_ids))
    return dict(counts)


def run_generator(source: Path, output: Path, generator: str, population: dict, namespace: str) -> dict:
    started = time.monotonic()
    dest = output / "models" / generator
    dest.mkdir(parents=True, exist_ok=False)
    rp = source / "rankings" / generator / "rankings.jsonl.gz"
    tp = source / "teacher" / generator / "rankings.jsonl.gz"
    rr_path = rp.with_name("RETRIEVAL_RECEIPT.json")
    tr_path = tp.with_name("TEACHER_RECEIPT.json")
    rr, tr = read_json(rr_path), read_json(tr_path)
    inputs = {"retrieval": record(rp), "teacher": record(tp),
              "retrieval_receipt": record(rr_path), "teacher_receipt": record(tr_path)}
    assert inputs["retrieval"]["sha256"] == rr["rankings"]["sha256"] == tr["signature"]["own_rankings"]["sha256"]
    assert inputs["teacher"]["sha256"] == tr["rankings"]["sha256"]
    assert tr["signature"]["namespace"] == namespace
    assert rr["signature"]["query_sha256"] == sha(source / "common/dev_queries.jsonl")
    write_json(dest / "INPUTS.json", inputs)
    teacher = {r["query_id"]: r for r in read_rows(tp)}
    assert teacher.keys() == population.keys()
    all_metrics, seen = [], set()
    counts, path_counts = Counter(), Counter()
    with ExitStack() as stack:
        handles = {name: stack.enter_context(gzip.open(dest / f"{name}.jsonl.gz", "wt", compresslevel=1))
                   for name in ("rankings", "per_query", "strict_evidence_only_pairs")}
        def emit(name, value):
            handles[name].write(json.dumps(value, separators=(",", ":")) + "\n")
        for row in read_rows(rp):
            q = row["query_id"]
            assert q not in seen
            seen.add(q)
            meta = {k: row[k] for k in population[q]}
            assert meta == population[q]
            t = teacher[q]
            assert all(t[k] == v for k, v in meta.items())
            assert t["candidate_pool_id"] == row["candidate_pool_id"]
            assert t["teacher_namespace"] == namespace
            assert row["parameter_sha"] == rr["signature"]["parameter_sha256"]
            assert row["index_id"] == stable_sha(rr["signature"])
            d = [r["target_id"] for r in row["D100_ANN"]]
            e = [r["target_id"] for r in row["E_paths"]]
            union, truth = set(d) | set(e), set(row["positive_target_ids"])
            assert len(d) == len(set(d)) == 100 and len(e) == len(set(e))
            assert union == set(row["U"]) and len(row["M_exact"]) == len(union)
            assert row["candidate_pool_id"] == stable_sha({"q": q, "D": d, "E": e})
            assert all(math.isfinite(t["teacher_scores"][target]) for target in union)
            strict = truth & (set(e) - set(d) - set(row["D100_EXACT"]))
            path_counts.update(validate_paths(row))
            invariant = {"D": row["D100_ANN"], "E": row["E_paths"], "pre": row["E_pre_retention"],
                         "U": row["U"], "T0": t["teacher_scores"]}
            before_hash = stable_sha(invariant)
            arms = replay_arms(row, t["teacher_scores"])
            assert stable_sha(invariant) == before_hash
            assert arms[ARMS[0]]["E"] == row["rankings"]["E_ONLY"] == e
            assert arms[ARMS[0]]["Equal"] == row["rankings"]["Equal"]
            assert arms[ARMS[0]]["C100"] == t["rankings"]["BT100_NO_T0"]
            assert arms[ARMS[0]]["T0"] == t["rankings"]["BT100_T0"]
            diagnostics = {"D100": d, "QT_OVER_U": row["rankings"]["QT_OVER_U"],
                           "D100_T0": t["rankings"]["D100_T0"]}
            assert diagnostics["QT_OVER_U"] == sorted(union, key=lambda target: (-row["QT_OVER_U_scores"][target], target))
            ranks = {**diagnostics, **{arm + "/" + stage: a[stage] for arm, a in arms.items()
                                      for stage in ("E", "Equal", "C100", "T0")}}
            metrics = {name: query_metrics(rank, list(truth), KS) for name, rank in ranks.items()}
            funnel = {name: {str(k): len(strict & set(rank[:k])) for k in KS} for name, rank in ranks.items()}
            for arm, a in arms.items():
                assert set(a["E"]) == set(e) and set(a["Equal"]) == union
                assert len(a["C100"]) == len(set(a["C100"])) == 100
                assert set(a["T0"]) == set(a["C100"])
                counts[arm + "/changed_E_queries"] += a["E"] != arms[ARMS[0]]["E"]
                counts[arm + "/changed_C100_queries"] += set(a["C100"]) != set(arms[ARMS[0]]["C100"])
                cset = set(a["C100"])
                paths = [p for p in row["E_paths"] if p["target_id"] in cset]
                metrics[arm + "/support"] = {
                    "E_targets_in_C100": len(paths),
                    "retained_evidence_in_C100": sum(len(p["retained_paths"]) for p in paths),
                    "routed_rows_in_C100": sum(len(set(p["routed_rows"].values())) for p in paths),
                    "D1_score_sum_in_C100": sum(p["evidence_score"] for p in paths),
                }
            per_query = {**meta, "metrics": metrics, "strict_pairs": len(strict), "strict_hits": funnel}
            all_metrics.append(per_query)
            emit("per_query", per_query)
            emit("rankings", {**meta, "candidate_pool_id": row["candidate_pool_id"],
                              "fixed_inputs_sha256": before_hash, "arms": arms, "diagnostics": diagnostics})
            retained = {p["target_id"]: p for p in row["E_paths"]}
            positions = {name: {target: i for i, target in enumerate(rank, 1)} for name, rank in ranks.items()}
            for target in sorted(strict):
                p = retained[target]
                emit("strict_evidence_only_pairs", {**meta, "target_id": target,
                     "ranks": {name: pos.get(target) for name, pos in positions.items()},
                     "E_scores": {arm: a["E_scores"][target] for arm, a in arms.items()},
                     "teacher_score": t["teacher_scores"][target],
                     "retained_paths": p["retained_paths"], "routed_rows": p["routed_rows"],
                     "pre_path_count": len(p["paths"])})
            counts["queries"] += 1
            counts["strict_pairs"] += len(strict)
            counts["strict_queries"] += bool(strict)
            if counts["queries"] % 300 == 0:
                print(json.dumps({"generator": generator, "queries": counts["queries"]}), flush=True)
    assert seen == population.keys()
    summary, comparisons = {}, []
    for kind in ("overall", "implicit", "explicit"):
        subset = [r for r in all_metrics if kind == "overall" or r["query_kind"] == kind]
        means = {name: {metric: sum(r["metrics"][name][metric] for r in subset) / len(subset)
                        for metric in subset[0]["metrics"][name]} for name in subset[0]["metrics"]}
        totals = {name: {str(k): sum(r["strict_hits"][name][str(k)] for r in subset) for k in KS}
                  for name in subset[0]["strict_hits"]}
        summary[kind] = {"queries": len(subset), "metrics": means, "strict_hits": totals,
                         "strict_pairs": sum(r["strict_pairs"] for r in subset)}
        for new, old in ((ARMS[1], ARMS[0]), (ARMS[2], ARMS[0]), (ARMS[2], ARMS[1])):
            for stage, metric in (("E", "recall@10"), ("E", "recall@50"), ("Equal", "recall@10"),
                                  ("Equal", "recall@50"), ("C100", "raw_recall"), ("T0", "recall@10"), ("T0", "recall@50")):
                delta = np.array([r["metrics"][new + "/" + stage][metric] - r["metrics"][old + "/" + stage][metric] for r in subset])
                comparisons.append({"kind": kind, "new": new, "old": old, "stage": stage, "metric": metric,
                                    **source_cluster_comparison(delta, [r["source_table_id"] for r in subset])})
        for own, original, receipt in ((ARMS[0] + "/E", "E_ONLY", rr), (ARMS[0] + "/Equal", "Equal", rr),
                                      (ARMS[0] + "/T0", "BT100_T0", tr), ("D100", "D100_ANN", rr),
                                      ("QT_OVER_U", "QT_OVER_U", rr)):
            for metric, expected in receipt["metrics"][kind][original].items():
                assert abs(means[own][metric] - expected) <= 1e-10, (generator, kind, original, metric)
    if generator == "B13":
        assert counts["strict_pairs"] == 207
        assert abs(summary["overall"]["metrics"][ARMS[0] + "/T0"]["recall@10"] - .4732888146911519) < 1e-10
    # Re-hash only frozen input artifacts, never unrelated workspace files.
    for rec in inputs.values():
        assert sha(Path(rec["path"])) == rec["sha256"]
    result = {"generator": generator, "status": "completed", "baseline_reproduced": True,
              "counts": dict(counts), "retention_diagnostics": dict(path_counts), "summary": summary,
              "comparisons": comparisons, "elapsed_seconds": time.monotonic() - started,
              "new_teacher_inferences": 0, "online_teacher_pairs_per_query": 100,
              "output_files": {name: record(dest / f"{name}.jsonl.gz") for name in handles}}
    write_json(dest / "RESULTS.json", result)
    print(json.dumps({"generator": generator, "status": "completed", "T0_R10": {
        arm: summary["overall"]["metrics"][arm + "/T0"]["recall@10"] for arm in ARMS}}), flush=True)
    return result


def write_report(output: Path, results: list[dict]) -> None:
    lines = ["# R26 固定证据三种排序重放", "",
             "A = D1 coverage；B = retained Path-LSE；C = pre-retention Path-LSE（仅在相同 retained E targets 上排序）。",
             "每个模型 1198 条冻结 dev queries。Direct100、E/U membership、retained witnesses、row routing、路径分数、Teacher T0 均固定；只改变 E rank，经 Equal RRF(k=60) 截断 C100 后用原 T0 分数排序。",
             "C100 的成员随排序改变是实验结果；固定的是截断前候选及 100 对 Teacher 预算。QT_OVER_U 保留为诊断。", "",
             "## 全量结果（query-macro recall，%）", "",
             "| 模型 | 排序 | E R10 | Equal R10 | C100 recall | T0 R10 | T0 R50 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in results:
        m = r["summary"]["overall"]["metrics"]
        for arm in ARMS:
            values = [m[arm + "/" + stage][metric] * 100 for stage, metric in
                      (("E", "recall@10"), ("Equal", "recall@10"), ("C100", "raw_recall"), ("T0", "recall@10"), ("T0", "recall@50"))]
            lines.append(f"| {r['generator']} | {arm} | " + " | ".join(f"{v:.3f}" for v in values) + " |")
    for r in results:
        if r["generator"] != "B13":
            continue
        lines += ["", "## B13：207 个 strict evidence-only positives", "",
                  "定义：G ∩ E − (D_ANN100 ∪ D_EXACT100)。下表是 query-target 对数量，分母 207，不是 query-macro recall。", "",
                  "| 排序 | E Top50 | Equal Top50 | C100 | T0 Top10 | T0 Top20 | T0 Top50 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        h = r["summary"]["overall"]["strict_hits"]
        for arm in ARMS:
            cells = [h[arm + "/" + stage][str(k)] for stage, k in (("E", 50), ("Equal", 50), ("C100", 100), ("T0", 10), ("T0", 20), ("T0", 50))]
            lines.append(f"| {arm} | " + " | ".join(map(str, cells)) + " |")
        lines += ["", f"QT_OVER_U Top10/20/50：{[h['QT_OVER_U'][str(k)] for k in (10, 20, 50)]}。", "",
                  "## B13 配对差异：T0 Recall@10", "",
                  "95% CI 按 source_table_id 整簇 bootstrap，10,000 次，seed=260914；差异单位百分点。多模型/分组结果属探索性比较，未做多重检验校正。", "",
                  "| 分组 | 比较 | 差异(pp) | 95% CI(pp) | 胜/负/平 queries |",
                  "|---|---|---:|---|---|"]
        for c in r["comparisons"]:
            if c["stage"] == "T0" and c["metric"] == "recall@10":
                lo, hi = c["bootstrap_95ci"]
                lines.append(f"| {c['kind']} | {c['new']} − {c['old']} | {100*c['mean_delta']:.3f} | [{100*lo:.3f}, {100*hi:.3f}] | {c['wins']}/{c['losses']}/{c['ties']} |")
    lines += ["", "## 范围与解释边界", "",
              "- A 完整排名及已存指标必须复现 R26；输入 SHA256 和每条 query 的固定证据不变性均校验。",
              "- 三臂共享 retained witness 内容和 row routing。C 仅用原始路径计算 target 排序分数，不把已删除路径重新交给下游。",
              "- pre/post LSE 差异同时包含路径数量和路径强度影响。C 优于 B 不能单独证明 D1 删除了语义正确证据；best-path-score-lost 也仅是分数诊断。",
              "- per_query 保存 C100 内 retained evidence 数、routed row 数及 D1 score sum；它们不是经标注验证的属性覆盖率或正确填充值。",
              "- T0 为固定 QT pair scorer（无 path 输入）；排序更改通过 C100 admission 影响 T0 结果。未新增属性生成或最终 semantic-joinability 验证，不能据此声称正确值恢复或端到端 joinability 已改善。",
              "- 仅 CPU 单进程顺序读取冻结缓存，0 次训练、0 次新 Teacher inference；线上预算仍为每 query 100 对。未测新线上延迟。",
              "- 不修改 R26/R27 原始代码、缓存或实验目录。运行代码快照、参数、输入哈希、逐 query 排名、证据专属漏斗、分组指标及 bootstrap 见本目录。", ""]
    (output / "REPORT.zh-CN.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generator", action="append")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    assert not output.is_relative_to(source), "Outputs must be separate from frozen R26"
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "PROTOCOL.json").exists(), "Choose a fresh output directory"
    population_path = source / "common/dev_queries.jsonl"
    population = {r["query_id"]: r for r in read_rows(population_path)}
    assert len(population) == 1198
    identity = read_json(source / "teacher/CACHE_IDENTITY.json")
    assert sha(Path(identity["teacher"]["path"])) == identity["teacher"]["sha256"]
    generators = args.generator or GENERATORS
    write_json(output / "PROTOCOL.json", {
        "hypothesis": "Replacing coverage E order by trained path LSE improves ranking and evidence-only survival under fixed witnesses.",
        "arms": dict(zip(ARMS, ("D1 coverage", "LSE retained paths", "LSE original paths restricted to same retained targets"))),
        "generators": generators, "source": str(source), "output": str(output),
        "population": record(population_path), "teacher": identity, "runner": record(Path(__file__)),
        "ks": KS, "rrf_k": 60, "teacher_budget": 100, "bootstrap_replicates": 10000,
        "primary": "B versus A; B13 T0 R10 and strict-evidence-only C100/T0 survival",
        "secondary": "C versus B/A; all models and implicit/explicit strata exploratory",
        "generation_and_final_joinability": "not rerun; rank-only intervention", "device": "CPU"})
    results = []
    for generator in generators:
        results.append(run_generator(source, output, generator, population, identity["namespace"]))
        write_report(output, results)
    write_json(output / "RESULTS.json", {"status": "completed", "models": results})


if __name__ == "__main__":
    main()
