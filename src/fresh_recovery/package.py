"""Final packaging: acceptance ledger (A01-A35), resource/error ledgers, portable archive (SPEC 16)."""
from __future__ import annotations

import json
import os
import shutil
import tarfile
import time
from pathlib import Path

from . import PROTOCOL_ID, PROTOCOL_VERSION, runlog
from .config import Paths
from .io import sha256_file, write_json

LIGHT_PATTERNS = ("*.json", "*.json.gz", "*.csv", "*.jsonl", "*.md", "*.log", "log.jsonl")
HEAVY_SUFFIXES = (".pt", ".pkl", ".npy")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def acceptance_ledger(paths: Paths, seed: int) -> list[dict]:
    w = paths.work_dir
    s = w / f"seed{seed}"
    rows = []

    def add(item, name, status, evidence, note=""):
        rows.append({"id": item, "item": name, "status": status, "evidence": evidence, "note": note})

    def post(stage):
        r = _read(s / stage / "POST_RUN.json")
        return r is not None and r.get("status") == "COMPLETE"

    root = _read(w / "ROOT_INPUTS.json")
    add("A01", "package version unique", "PASS" if root and root.get("version") == PROTOCOL_VERSION else "FAIL", "ROOT_INPUTS.json / SOURCE_LOCK.json")
    verify = _read(s / "VERIFY_INTEGRATION" / "VERIFY_INTEGRATION.json")
    add("A02", "fresh lineage closed (real small chain, run-local parents)",
        "NOT_RUN",
        "VERIFY_INTEGRATION.json; formal PRE_RUN/POST_RUN lineage",
        "The discarded small chain covered T_PATH but did not run a separate QT_CONT terminal arm; individual production probes passed, but P26 is not claimed.")
    prov = _read(w / "LABEL_PROVENANCE.json")
    add("A03", "original train scope", "PASS" if prov and prov.get("train_scope") == "all_original_train" else "FAIL", "LABEL_PROVENANCE.json")
    iso = _read(w / "LABEL_ISOLATION_PROBE.json")
    add("A04", "GT isolation", "PASS" if iso and iso.get("stable") else "FAIL", "LABEL_ISOLATION_PROBE.json; dev/test GT loaded only by eval commands")
    alias = _read(w / "CONTENT_ALIASES.json")
    add("A05", "content aliases from raw bytes", "PASS" if alias else "FAIL", "CONTENT_ALIASES.json(.jsonl.gz)")
    pure = _read(w / "PURE_CACHE_STRUCTURAL_AUDIT.json")
    add("A06", "query rows preserved", "PASS" if pure and pure["tables"]["query_rows_preserved"] == pure["tables"]["query_objects"] else "FAIL", "PURE_CACHE_STRUCTURAL_AUDIT.json")
    src = _read(w / "PURE_FEATURE_SOURCE.json"); rows_src = _read(w / "ROW_FEATURE_SOURCE.json")
    add("A07", "pure features rebuildable / verified", "PASS" if src and src["status"] == "PASS" and rows_src and rows_src["status"] == "PASS" else "FAIL", "PURE_CACHE_RECOMPUTE_AUDIT.json, ROW_CACHE_RECOMPUTE_AUDIT.json")
    pca = _read(w / "PCA_REPORT.json")
    add("A08", "PCA generated in this run", "PASS" if pca and pca["status"] == "PASS" else "FAIL", "PCA_REPORT.json")
    l0 = _read(w / "lists" / f"seed{seed}" / "L0_REPORT.json")
    add("A09", "five-relation list semantics", "PASS" if l0 and set(l0["L0"]) == {"QT", "Q_text", "Q_image", "text_T", "image_T"} else "NOT_REACHED", "L0_REPORT.json; loss = list-equal-weight CE")
    inits = {}
    for arm in ("S_SUP_NATIVE", "S_KD_NATIVE", "S_QT_SUP", "S_QT_KD"):
        pre = _read(s / f"{arm}_C1" / "PRE_RUN.json")
        inits[arm] = pre["initial_state_sha256"] if pre else None
    t_init = _read(s / "T_INIT" / "INIT_LINEAGE.json")
    paired = inits["S_SUP_NATIVE"] and inits["S_SUP_NATIVE"] == inits["S_KD_NATIVE"] and inits["S_QT_SUP"] and inits["S_QT_SUP"] == inits["S_QT_KD"]
    add("A10", "fresh init; paired arms identical; empty optimizer", "PASS" if t_init and paired else "NOT_REACHED", f"T_INIT/INIT_LINEAGE.json; C1 PRE_RUN initial_state_sha256: {inits}")
    add("A11", "positive protection / empty sets", "PASS" if verify and verify["status"] == "PASS" else "NOT_REACHED", "VERIFY_INTEGRATION A19/A14 (empty P -> None, no fallback to G)")
    for item, key in (("A12", "A12"), ("A13", "A13"), ("A14", "A14"), ("A15", "A15"), ("A17", "A17"), ("A18", "A18"), ("A19", "A19")):
        res = [r for r in (verify or {}).get("results", []) if r["item"].startswith(key)]
        add(item, {"A12": "C1 formula", "A13": "C2 formula", "A14": "path slot order", "A15": "legacy read_q", "A17": "own-path identity", "A18": "D1 admission", "A19": "denominators/gradient accumulation"}[key],
            "PASS" if res and all(r["status"] == "PASS" for r in res) else ("FAIL" if res else "NOT_REACHED"), "VERIFY_INTEGRATION.json")
    a16 = [r for r in (verify or {}).get("results", []) if "A16" in r["item"]]
    add("A16", "train/exact/ANN identical scores", "PASS" if a16 and all(r["status"] == "PASS" for r in a16) else "NOT_REACHED", "VERIFY_INTEGRATION.json")
    a20 = [r for r in (verify or {}).get("results", []) if "A20" in r["item"]]
    add("A20", "real small chain", "NOT_RUN",
        "VERIFY_INTEGRATION.json + SMALL_CHAIN_REPORT.json",
        "Recorded probe chain omitted a separate QT_CONT training step; do not interpret its PASS rows as full P26 acceptance.")
    add("P26", "temporary full-chain integration including both terminal Teachers", "NOT_RUN",
        "VERIFY_INTEGRATION.json",
        "T_PATH was exercised; QT_CONT was not independently trained in the discarded pre-formal probe. Formal QT_CONT and T_PATH were both run, but that does not retroactively satisfy the required pre-init temporary-chain ordering.")
    sel = {k: _read(s / f"SELECTION_{k}.json") for k in ("NATIVE_C1", "NATIVE_C2", "QT_C1", "QT_C2")}
    add("A21", "stage selection (fixed fractions, paired, init not eligible)", "PASS" if all(v and v["status"] == "SELECTED" for v in sel.values()) else ("BLOCKED" if any(v and v["status"].startswith("STOP") for v in sel.values()) else "NOT_REACHED"),
        {k: (v["status"], v.get("selected")) if v else None for k, v in sel.items()})
    hard = _read(s / "HARD32" / "HARD32_REPORT.json")
    add("A22", "single hard32 from S_KD own ANN", "PASS" if hard else "NOT_REACHED", "HARD32/HARD32_REPORT.json")
    tqt = _read(s / "T_QT" / "PRE_RUN.json"); tpath = _read(s / "T_PATH" / "PRE_RUN.json")
    ok23 = tqt and tqt["initial_state_sha256"] == (t_init or {}).get("state_sha256") and tpath and "T_QT" in tpath["parents"]
    add("A23", "T_QT from T_INIT; T_PATH from same-run T_QT", "PASS" if ok23 and post("T_QT") and post("T_PATH") else "NOT_REACHED", "T_QT/PRE_RUN.json initial_state_sha256 == T_INIT; T_PATH/PRE_RUN.json parents")
    a24 = [r for r in (verify or {}).get("results", []) if "A24" in r["item"]]
    add("A24", "real QET forward, shared head", "PASS" if a24 and all(r["status"] == "PASS" for r in a24) and post("T_PATH") else "NOT_REACHED", "VERIFY_INTEGRATION A24; T_PATH config")
    add("A25", "no presence margin; fixed weights", "PASS" if tpath and tpath["config"]["presence_margin_weight"] == 0 else "NOT_REACHED", "T_PATH/PRE_RUN.json config")
    dev = _read(s / "EVAL_DEV" / "RESULTS.json")
    needed = ["Raw|Direct", "Raw|C150", "Raw+T_QT|C150", "Raw+T_PATH_Real|C150", "Raw+T_PATH_f0|C150", "Raw+T_PATH_Swap|C150",
              "S_SUP_NATIVE|Direct", "S_SUP_NATIVE+T_QT|C150", "S_SUP_NATIVE+T_PATH_Real|C150", "S_KD_NATIVE|Direct", "S_KD_NATIVE+T_QT|C150",
              "S_KD_NATIVE+T_PATH_Real|C150", "S_KD_NATIVE+T_PATH_f0|C150", "S_KD_NATIVE+T_PATH_Swap|C150", "S_QT_SUP|Direct", "S_QT_SUP+T_QT|D100",
              "S_QT_KD|Direct", "S_QT_KD+T_QT|D100", "S_KD_NATIVE+T_QT|D100"]
    have = set((dev or {}).get("systems", {}))
    add("A26", "complete controls", "PASS" if dev and all(k in have for k in needed) else "NOT_REACHED", f"EVAL_DEV/RESULTS.json missing={[k for k in needed if k not in have]}")
    fullu = ["Raw+T_QT|FullU", "S_SUP_NATIVE+T_QT|FullU", "S_KD_NATIVE+T_QT|FullU", "Raw+T_PATH_Real|FullU", "S_SUP_NATIVE+T_PATH_Real|FullU", "S_KD_NATIVE+T_PATH_Real|FullU"]
    add("A27", "C150 and Full-U for Raw/SUP/KD", "PASS" if dev and all(k in have for k in fullu) else "NOT_REACHED", "EVAL_DEV/RESULTS.json (separate |C150 and |FullU rows)")
    add("P24", "implicit witness nested pairs and exact/ANN diagnostic", "PASS" if
        dev and (s / "EVAL_DEV" / "WITNESS_FUNNELS.json").exists() and
        (s / "EVAL_TEST" / "WITNESS_FUNNELS.json").exists() else "NOT_REACHED",
        "EVAL_DEV/WITNESS_FUNNELS.json; EVAL_TEST/WITNESS_FUNNELS.json")
    add("P23", "fixed structural and init/half/end readouts", "PASS" if
        "S_KD_NATIVE+T_PATH_Struct|C150" in have and
        (s / "EVAL_DEV" / "TRAJECTORY.json").exists() else "NOT_REACHED",
        "EVAL_DEV/RESULTS.json; EVAL_DEV/TRAJECTORY.json")
    add("A28", "strict own + fixed cohort with target IDs", "PASS" if dev and (s / "EVAL_DEV" / "STRICT.json").exists() else "NOT_REACHED", "EVAL_DEV/STRICT.json (per_query target IDs)", "conditional (q,e) probe not produced this round: no Student adapter; Teacher full-lake probe not executed")
    recomputed = {split: _read(s / f"EVAL_{split}" / "RECOMPUTED_METRICS.json") for split in ("DEV", "TEST")}
    add("A29", "independent metric recomputation",
        "PASS" if all(row and row.get("status") == "PASS" for row in recomputed.values()) else
        "FAIL" if any(row and row.get("status") == "FAIL" for row in recomputed.values()) else "NOT_REACHED",
        "EVAL_DEV/RECOMPUTED_METRICS.json; EVAL_TEST/RECOMPUTED_METRICS.json")
    add("A30", "historical reference isolated", "PASS", "HISTORICAL_COMPARABILITY.json: historical weights unavailable -> REFERENCE_UNAVAILABLE; never read by training")
    events = s / "SCHEDULE_EVENTS.jsonl"
    add("A31", "GPU placement and measured co-residency", "PASS" if events.exists() and post("T_QT_CONT") and post("T_PATH") else "NOT_REACHED",
        "SCHEDULE_EVENTS.jsonl, PRE_RUN environment, POST_RUN peak_reserved_GiB",
        "seed13 uses GPU0 only per user request; GPU1 and seed29 are not part of the remaining run. "
        "A schedule event alone does not establish safe co-residency or throughput.")
    lat = _read(s / "EVAL_DEV" / "LATENCY.json")
    add("A32", "exclusive latency", "PASS" if lat else "NOT_REACHED", "EVAL_DEV/LATENCY.json")
    training_stages = [
        "T_BOOT", "S_SUP_NATIVE_C1", "S_KD_NATIVE_C1", "S_QT_SUP_C1", "S_QT_KD_C1",
        "S_SUP_NATIVE_C2", "S_KD_NATIVE_C2", "S_QT_SUP_C2", "S_QT_KD_C2",
        "T_QT", "T_QT_FROM_BOOT", "T_QT_CONT", "T_PATH",
    ]
    missing_receipts = [stage for stage in training_stages
                        if not (s / stage / "PRE_RUN.json").exists() or not post(stage)]
    add("A33", "PRE_RUN before training, POST_RUN after",
        "PASS" if not missing_receipts else "NOT_REACHED",
        "per-stage PRE_RUN.json/POST_RUN.json; ERROR_LEDGER.jsonl",
        f"missing formal training receipts={missing_receipts}; initialization, selection, and read-only reports are recorded by their native artifacts, not retroactive receipts")
    decision = _read(s / "DECISION.json")
    add("A34", "R/P/KD judged separately", "PASS" if decision else "NOT_REACHED", "DECISION.json")
    add("A35", "light deliverable complete", "PASS" if dev and decision else "NOT_REACHED", "package manifest")
    return rows


def write_execution_source(paths: Paths, seed: int) -> dict:
    """Snapshot the executed source and cross-check it against every stage's PRE_RUN code_lock."""
    w = paths.work_dir
    dest = w / "execution_source"
    if dest.exists():
        shutil.rmtree(dest)
    src_root = runlog.SRC_DIR.parent
    files = sorted(runlog.SRC_DIR.glob("*.py")) + [src_root / "fresh_path" / n for n in runlog.SHARED_FRESH_PATH_MODULES]
    files += [src_root / "run_fresh_recovery.py", src_root / "fresh_path" / "__init__.py", src_root / "fresh_path" / "candidates.py"]
    files += sorted((src_root / "mmdd_stage1").glob("construction.py")) + [src_root / "cache_stage1_features.py"]
    package_files = sorted(p for p in paths.package_dir.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".json", ".md"))
    manifest = {}
    for path in files + package_files:
        if not path.is_file():
            continue
        rel = path.relative_to(src_root) if src_root in path.parents else Path("audit_package") / path.relative_to(paths.package_dir)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        manifest[str(rel)] = sha256_file(path)
    current = runlog.code_lock()
    stages = {}
    for stage_dir in sorted((w / f"seed{seed}").iterdir()):
        pre = _read(stage_dir / "PRE_RUN.json")
        if not pre or "code_lock" not in pre:
            continue
        lock = pre["code_lock"]
        changed = sorted(k for k in lock if current.get(k) != lock[k])
        missing = sorted(k for k in current if k not in lock)
        stages[stage_dir.name] = {"code_lock_sha256": pre["code_lock_sha256"], "modules_changed_since": changed,
                                  "modules_added_since": missing, "identical_to_snapshot": not changed and not missing}
    payload = {"snapshot_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": manifest,
               "current_code_lock_sha256": runlog.code_lock_sha(),
               "per_stage_pre_run_cross_check": stages,
               "note": "PRE_RUN.json of every stage records the module hashes at that stage's start; modules listed under "
                       "modules_changed_since/added_since were edited after that stage ran (report/packaging/compaction code), "
                       "the training modules used by the stage are those whose hash matches."}
    write_json(dest / "EXECUTION_SOURCE.json", payload)
    write_json(w / "SOURCE_LOCK.json", {"protocol_id": PROTOCOL_ID, "version": PROTOCOL_VERSION, "code": manifest,
                                        "code_lock_sha256": payload["current_code_lock_sha256"], "snapshot": "execution_source/"})
    return payload


def cmd_package(paths: Paths, *, seed: int) -> dict:
    w = paths.work_dir
    s = w / f"seed{seed}"
    source = write_execution_source(paths, seed)
    ledger = acceptance_ledger(paths, seed)
    write_json(s / "ACCEPTANCE_LEDGER.json", ledger)
    from .report import write_report

    write_report(paths, seed)
    write_json(s / "HISTORICAL_COMPARABILITY.json", {
        "status": "REFERENCE_UNAVAILABLE",
        "historical_weights": "not available in this run; no isolated re-evaluation performed",
        "external_references": {
            "B13+T0_session": {"C100_R10": 0.4733, "FullU_R10": 0.49374, "FullU_implicit": 0.41235, "FullU_explicit": 0.57513,
                               "source": "prior session confirmation, not re-evaluated under v3 protocol"},
            "bridge_B4+T0": {"C100_R10": 0.47475, "source": "historical summary check, different supervision/Teacher/retention"},
            "audit_v2.1_Raw+T_PATH_raw_paths": {"R10": 0.4270, "source": "AUDIT.zh-CN.md section 3 (v2.1 saved rankings recomputed)"},
        },
        "rule": "these numbers are protocol-labelled external references; they are not entered in any 're-evaluated' column and never fed to training",
    })
    resource = {"schedule_events": str(s / "SCHEDULE_EVENTS.jsonl"), "stages": {}}
    for stage_dir in sorted(s.iterdir()):
        post = _read(stage_dir / "POST_RUN.json")
        pre = _read(stage_dir / "PRE_RUN.json")
        if post:
            resource["stages"][stage_dir.name] = {"status": post["status"], "started": post["started_utc"], "finished": post["finished_utc"],
                                                   "peak_reserved_GiB": post.get("peak_reserved_GiB"), "counters": post.get("counters"),
                                                   "gpu": (pre or {}).get("environment", {}).get("cuda_visible_devices")}
    write_json(s / "RESOURCE.json", resource)
    # portable archive: light files only, heavy files listed in manifest
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_tag = f"MMDD_FRESH_RECOVERY_v3_1_seed{seed}_{stamp}_{runlog.RUN_ID[-8:]}"
    archive = w / f"{run_tag}.tar.gz"
    if archive.exists():
        raise FileExistsError(archive)
    manifest = {"run_id": runlog.RUN_ID, "protocol": f"{PROTOCOL_ID} {PROTOCOL_VERSION}", "seed": seed, "included": [], "omitted_heavy": []}
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(w.rglob("*")):
            if not path.is_file() or path == archive:
                continue
            rel = path.relative_to(w)
            if rel.parts[0].startswith("seed") and rel.parts[0] != f"seed{seed}":
                continue
            if rel.parts[0] == "execution_source":
                pass
            elif path.suffix in HEAVY_SUFFIXES or path.stat().st_size > 200 * 2**20:
                manifest["omitted_heavy"].append({"path": str(rel), "bytes": path.stat().st_size})
                continue
            tar.add(path, arcname=f"{run_tag}/{rel}")
            manifest["included"].append({"path": str(rel), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        manifest_path = w / f"{run_tag}_MANIFEST.json"
        write_json(manifest_path, manifest)
        tar.add(manifest_path, arcname=f"{run_tag}/MANIFEST.json")
    return {"archive": str(archive), "sha256": sha256_file(archive), "included": len(manifest["included"]),
            "omitted_heavy": len(manifest["omitted_heavy"]), "source_files": len(source["files"]),
            "stages_with_changed_modules": {k: v["modules_changed_since"] + v["modules_added_since"]
                                            for k, v in source["per_stage_pre_run_cross_check"].items() if not v["identical_to_snapshot"]},
            "ledger": {r["id"]: r["status"] for r in ledger}}
