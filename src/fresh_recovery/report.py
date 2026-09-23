"""Human-readable report (RESULTS.zh-CN.md) assembled only from saved JSON artifacts."""
from __future__ import annotations

import json
from pathlib import Path

from .config import Paths

MAIN_ROWS = [
    ("Raw|Direct", "Raw Direct (exact z)"), ("Raw|C150", "Raw 两路 C150"), ("Raw+T_QT|C150", "Raw + T_QT (C150)"),
    ("Raw+T_QT_FROM_BOOT|C150", "Raw + T_QT_FROM_BOOT (C150, 额外对照)"),
    ("Raw+T_QT_CONT|C150", "Raw + T_QT_CONT (C150)"),
    ("Raw+T_QT|FullU", "Raw + T_QT (Full-U)"), ("Raw+T_PATH_f0|C150", "Raw + T_PATH f0 (C150)"),
    ("Raw+T_PATH_Real|C150", "Raw + T_PATH Real (C150)"), ("Raw+T_PATH_Swap|C150", "Raw + T_PATH E-swap (C150)"),
    ("Raw+T_PATH_Real|FullU", "Raw + T_PATH Real (Full-U)"),
    ("PCA_INIT|Direct", "PCA init Direct"), ("PCA_INIT|C150", "PCA init C150"),
    ("S_SUP_NATIVE|Direct", "S_SUP_NATIVE Direct (ANN)"), ("S_SUP_NATIVE|Direct_exact", "S_SUP_NATIVE Direct (exact)"),
    ("S_SUP_NATIVE|C150", "S_SUP_NATIVE 两路 C150"), ("S_SUP_NATIVE+T_QT|C150", "S_SUP_NATIVE + T_QT (C150)"),
    ("S_SUP_NATIVE+T_QT_CONT|C150", "S_SUP_NATIVE + T_QT_CONT (C150)"),
    ("S_SUP_NATIVE+T_QT|FullU", "S_SUP_NATIVE + T_QT (Full-U)"), ("S_SUP_NATIVE+T_PATH_Real|C150", "S_SUP_NATIVE + T_PATH Real (C150)"),
    ("S_SUP_NATIVE+T_PATH_Real|FullU", "S_SUP_NATIVE + T_PATH Real (Full-U)"),
    ("S_KD_NATIVE|Direct", "S_KD_NATIVE Direct (ANN)"), ("S_KD_NATIVE|Direct_exact", "S_KD_NATIVE Direct (exact)"),
    ("S_KD_NATIVE|Evidence", "S_KD_NATIVE Evidence 通道"), ("S_KD_NATIVE|C150", "S_KD_NATIVE 两路 C150"),
    ("S_KD_NATIVE+T_QT|C150", "S_KD_NATIVE + T_QT (C150) [恢复对照]"), ("S_KD_NATIVE+T_QT|FullU", "S_KD_NATIVE + T_QT (Full-U)"),
    ("S_KD_NATIVE+T_QT_CONT|C150", "S_KD_NATIVE + T_QT_CONT (C150, 匹配控制)"),
    ("S_KD_NATIVE+T_QT_CONT|MatchedDirectM", "S_KD_NATIVE MatchedDirectM + T_QT_CONT"),
    ("S_KD_NATIVE+T_QT_CONT|P0_K20_C100_OLD", "S_KD_NATIVE + T_QT_CONT (P0 K20/C100)"),
    ("S_KD_NATIVE+T_QT_CONT|P1_K20_C150_OLD", "S_KD_NATIVE + T_QT_CONT (P1 K20/C150)"),
    ("S_KD_NATIVE+T_QT_CONT|P2_K50_C150_OLD", "S_KD_NATIVE + T_QT_CONT (P2 K50/C150)"),
    ("S_KD_NATIVE+T_QT_CONT|P3_K50_C150_QTALL", "S_KD_NATIVE + T_QT_CONT (P3 K50/C150/QTALL)"),
    ("S_KD_NATIVE+T_QT_FROM_BOOT|C150", "S_KD_NATIVE + T_QT_FROM_BOOT (C150, 额外对照)"),
    ("S_KD_NATIVE+T_QT|D100", "S_KD_NATIVE Direct100 + T_QT"), ("S_KD_NATIVE+T_QT|MatchedDirectM", "S_KD_NATIVE MatchedDirectM + T_QT"),
    ("S_KD_NATIVE+T_PATH_f0|C150", "S_KD_NATIVE + T_PATH f0 (C150)"),
    ("S_KD_NATIVE+T_PATH_Struct|C150", "S_KD_NATIVE + T_PATH 纯结构 (C150)"),
    ("S_KD_NATIVE+T_PATH_Real|C150", "**S_KD_NATIVE + T_PATH Real (C150) [主系统]**"),
    ("S_KD_NATIVE+T_PATH_Swap|C150", "S_KD_NATIVE + T_PATH E-swap (C150)"),
    ("S_KD_NATIVE+T_PATH_Real|FullU", "S_KD_NATIVE + T_PATH Real (Full-U)"), ("S_KD_NATIVE+T_PATH_Swap|FullU", "S_KD_NATIVE + T_PATH E-swap (Full-U)"),
    ("S_QT_SUP|Direct", "S_QT_SUP Direct"), ("S_QT_SUP+T_QT|D100", "S_QT_SUP + T_QT"),
    ("S_QT_KD|Direct", "S_QT_KD Direct"), ("S_QT_KD+T_QT|D100", "S_QT_KD + T_QT"),
]


def _pct(value) -> str:
    return "—" if value is None else f"{100 * value:.2f}"


def _table(systems: dict) -> str:
    lines = ["| 系统 | R@10 | implicit | explicit | R@20 | R@50 | 覆盖 | Oracle@10 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, label in MAIN_ROWS:
        row = systems.get(key)
        if not row:
            continue
        o = row["overall"]
        lines.append(f"| {label} | {_pct(o['R10'])} | {_pct(row.get('implicit', {}).get('R10'))} | {_pct(row.get('explicit', {}).get('R10'))} | "
                     f"{_pct(o['R20'])} | {_pct(o['R50'])} | {_pct(o.get('coverage'))} | {_pct(o.get('Oracle10'))} |")
    return "\n".join(lines)


def _contrast(name: str, c: dict) -> str:
    if c.get("status"):
        return f"| {name} | {c['status']} | | | |"
    o, i = c["overall"], c["implicit"]
    return (f"| {name} | {100 * o['point_estimate']:+.2f} [{100 * o['ci95'][0]:+.2f}, {100 * o['ci95'][1]:+.2f}] W/L/T {o['win']}/{o['loss']}/{o['tie']} "
            f"| {100 * i['point_estimate']:+.2f} [{100 * i['ci95'][0]:+.2f}, {100 * i['ci95'][1]:+.2f}] | {c['systems'][0]} | {c['systems'][1]} |")


def _path_mechanism(seed_dir: Path) -> dict | None:
    import gzip

    from .compact import load_any

    if not (seed_dir / "EVAL_DEV").exists():
        return None
    try:
        rec = load_any(seed_dir / "EVAL_DEV", "teacher_logits").get("T_PATH|S_KD_NATIVE")
    except FileNotFoundError:
        return None
    if not rec:
        return None
    gaps, diffs = [], []
    for r in rec.values():
        f0 = r["f0"]
        swap = r.get("swap_paths", {})
        for t, slots in r.get("paths", {}).items():
            for i, (e, v) in enumerate(slots):
                gaps.append(v - f0[t])
                if t in swap:
                    diffs.append(abs(v - swap[t][i][1]))
    if not gaps:
        return None
    return {"slots": len(gaps), "frac_above_f0": sum(g > 0 for g in gaps) / len(gaps), "mean_gap": sum(gaps) / len(gaps),
            "swap_abs_diff": (sum(diffs) / len(diffs)) if diffs else 0.0, "swap_max_diff": max(diffs) if diffs else 0.0}


def write_report(paths: Paths, seed: int) -> Path:
    s = paths.work_dir / f"seed{seed}"
    dev = json.loads((s / "EVAL_DEV" / "RESULTS.json").read_text()) if (s / "EVAL_DEV" / "RESULTS.json").exists() else None
    test = json.loads((s / "EVAL_TEST" / "RESULTS.json").read_text()) if (s / "EVAL_TEST" / "RESULTS.json").exists() else None
    decision = json.loads((s / "DECISION.json").read_text()) if (s / "DECISION.json").exists() else None
    ledger = json.loads((s / "ACCEPTANCE_LEDGER.json").read_text()) if (s / "ACCEPTANCE_LEDGER.json").exists() else []
    latency = json.loads((s / "EVAL_DEV" / "LATENCY.json").read_text()) if (s / "EVAL_DEV" / "LATENCY.json").exists() else None
    out = [f"# MMDD FRESH-RECOVERY v3.1 — seed {seed} 结果", "",
           "所有数字均由保存的 target-ID 排名与完整原始 GT 重新计算（`EVAL_*/rankings.cjson.xz`（无损字典编码，`fresh_recovery/compact.py` 读取）+ `gt.json.gz`），分母为全部 G[q]。",
           "历史 47%/49% 只是带协议标签的外部参照（`HISTORICAL_COMPARABILITY.json`），未在本轮重评。",
           "本次依用户最新要求仅完成 seed13、仅使用 GPU0；T_QT_FROM_BOOT 是追加对照，不能替代 T_PATH 主方法。单 seed 不支持原协议的双 seed 结论。", ""]
    for split, res in (("dev", dev), ("test", test)):
        if not res:
            out.append(f"## {split}: NOT_REACHED\n")
            continue
        out += [f"## {split} ({res['queries']} queries{'；既有test回归集，非新盲测' if split == 'test' else ''})", "", _table(res["systems"]), ""]
        out += ["### 配对 source-group bootstrap（R@10，×100，10,000 次，seed 20260923）", "",
                "| 对比 | overall Δ [95% CI] | implicit Δ [95% CI] | A | B |", "|---|---|---|---|---|"]
        out += [_contrast(k, v) for k, v in res["contrasts"].items()]
        out += ["", f"E-swap 有效替换比例：{json.dumps(res.get('swap'))}", f"MatchedDirectM：{json.dumps(res.get('matched_direct_m'))}",
                f"strict 汇总：{json.dumps(res.get('strict_summary'))}", ""]
        funnels = res.get("witness_funnel_summary", {})
        if funnels:
            out += ["### Implicit witness 逐对漏斗（pair 数；原始 ID 与 exact rank 见 WITNESS_FUNNELS.json）", "",
                    "| Generator | I | A | B (ET50 ANN) | C (D1) | D (C150) | F10 | F50 |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|"]
            out += ["| " + gen + " | " + " | ".join(str(rows[s]["pairs"]) for s in
                                                    ("I", "A", "B", "C", "D", "F10", "F50")) + " |"
                    for gen, rows in funnels.items()]
            out.append("")
    trajectory_path = s / "EVAL_DEV" / "TRAJECTORY.json"
    if trajectory_path.exists():
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))["snapshots"]
        out += ["## 固定 KD/P3 轨迹（init/half/end，非选模依据）", "",
                "| Teacher | 时点 | 更新 | f0 R@10 | Real R@10 | Swap R@10 | 有效替换 |", "|---|---|---:|---:|---:|---:|---:|"]
        for arm, snaps in trajectory.items():
            for tag, item in snaps.items():
                grouped = item["metrics"]
                out.append(f"| {arm} | {tag} | {item['updates']} | {_pct(grouped['f0']['overall']['R10'])} | "
                           f"{_pct(grouped.get('Real', {}).get('overall', {}).get('R10'))} | "
                           f"{_pct(grouped.get('Swap', {}).get('overall', {}).get('R10'))} | "
                           f"{_pct(item['effective_swap'])} |")
        out.append("")
    if decision:
        out += ["## 判定（DECISION.json）", "", f"- R（工程恢复）：**{decision['R']}** — {decision['R_note']}",
                f"- P（真实路径贡献）：**{decision['P']}**", f"- KD：**{decision['KD']['status']}**",
                "- seed29：本轮按用户最新要求不继续调度（原规范曾授权）。", "", "| 检查 | 值 | 阈值 | 通过 |", "|---|---:|---:|---|"]
        out += [f"| {c['name']} | {c['value'] if c['value'] is None else round(c['value'], 4)} | {c['op']} {c['threshold']} | {'PASS' if c['passed'] else 'FAIL'} |" for c in decision["checks"]]
        out.append("")
    mech = _path_mechanism(s)
    if mech:
        out += ["## Path 机制核查（来自 EVAL_DEV/teacher_logits.json.gz，非解释性猜测）", "",
                f"- S_KD_NATIVE own 路径槽位 {mech['slots']:,}：f(q,e,t) 高于 f0(q,t) 的比例 = {mech['frac_above_f0']:.4f}；f−f0 均值 {mech['mean_gap']:+.2f} nats。",
                f"- Real 与 E‑swap 的逐槽 |Δf| 均值 {mech['swap_abs_diff']:.3f}、最大 {mech['swap_max_diff']:.2f}：三元组前向确实读取了 E 内容。",
                "- 内容贡献须结合本轮同池 Real、Swap、f0、QT_CONT 排名差异与置信区间判读；单次槽位分数差不能证明最终检索收益。", ""]
    if latency:
        out += ["## 独占时延（S_KD_NATIVE own C150，未缓存前向）", "", f"```\n{json.dumps({k: v for k, v in latency.items() if k in ('T_QT', 'T_PATH')}, indent=1)}\n```", ""]
    out += ["## 验收条款", "", "| ID | 条款 | 状态 | 证据 |", "|---|---|---|---|"]
    out += [f"| {r['id']} | {r['item']} | {r['status']} | {r['evidence']} {('— ' + r['note']) if r.get('note') else ''} |" for r in ledger]
    path = s / "RESULTS.zh-CN.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path
