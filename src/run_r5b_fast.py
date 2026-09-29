"""Archived R5b calibration helpers; experiment CLI retired on 2026-09-28.

The user dropped the calibration model and its train generation workload.
Helpers remain importable for the existing lazy-selector audit and historical
engineering tests. Active selector training lives in run_stage2_columns_r2.py.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

from mmdd_stage2.calibration_cohort import choose_cohort
from mmdd_stage2.lazy_selector import build_lazy_plan, select_lazy

REPO = Path(__file__).resolve().parents[1]
NAME = "MMDD_S2_R5b_EvidenceCalibration_FrozenQET_S13_GPU1_v1"


def read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def immutable_json(path: Path, value: object) -> None:
    if path.exists():
        require(read(path) == value, f"Identity changed; use a new output directory: {path}")
    else:
        write(path, value)


def link_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        require(destination.resolve() == source.resolve(), f"Unexpected existing artifact: {destination}")
    else:
        destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def initialize(args: argparse.Namespace) -> dict:
    package, source, out = args.package.resolve(), args.source.resolve(), args.out.resolve()
    for protected in (package, source):
        require(not out.is_relative_to(protected) and not protected.is_relative_to(out), "Output overlaps an input")
    snapshot = source / "train_snapshot"
    c = read(snapshot / "RUNTIME_CONFIG.json")
    pop = read(snapshot / "population/POPULATION.json")
    parts = read(package / "reference/TRAIN_PARTITIONS.json")
    # Full train remains the default. Sampling is an explicit experiment option.
    fit_budget = args.fit_queries if args.fit_queries is not None else sum(v == "fit" for v in parts.values())
    val_budget = args.validation_queries if args.validation_queries is not None else sum(v == "inner_val" for v in parts.values())
    cohort = choose_cohort(pop, parts, fit_budget, val_budget, args.seed)
    parent = Path(read(package / "config/EXPERIMENT.json")["parent_root"])
    eval_pop = read(parent / "population/POPULATION.json")
    require(not {r["source_group"] for r in pop} & {r["source_group"] for r in eval_pop}, "Train/evaluation source group overlap")
    preflight = read(snapshot / "PREFLIGHT_SEAL.json")
    for rel in ("population/POPULATION.json", "population/VISIBLE_QUERIES.json", "population/train_ids.json", "RUNTIME_CONFIG.json"):
        require(sha(snapshot / rel) == preflight[rel], "Changed source snapshot: " + rel)
    require(sha(Path(c["selector_checkpoint"])) == c["selector_sha256"], "Changed selector checkpoint")
    sources = [Path(__file__), REPO / "src/mmdd_stage2/lazy_selector.py", REPO / "src/mmdd_stage2/calibration_cohort.py"]
    sources += list((REPO / "src/mmdd_stage2").glob("*.py"))
    for folder in (package / "code", package / "train_runtime/code", package / "train_runtime/server_binding", package / "train_runtime/prompts"):
        sources.extend(p for p in folder.rglob("*") if p.suffix in {".py", ".txt"})
    files = {str(p.resolve()): sha(p) for p in sorted(set(sources))}
    original_lock = read(package / "SOURCE_SHA256.json")
    deviations = {str(p.relative_to(package)): {"original": original_lock.get(str(p.relative_to(package))), "current": files[str(p.resolve())]}
                  for p in sources if p.is_relative_to(package)
                  and original_lock.get(str(p.relative_to(package))) != files[str(p.resolve())]}
    manifest = {"version": "R5b_FAST_v1", "source_snapshot": str(snapshot), "package": str(package),
                "source_stage1_seal": sha(snapshot / "STAGE1_SEAL.json"), "source_population": sha(snapshot / "population/POPULATION.json"),
                "config": sha(package / "config/EXPERIMENT.json"), "runtime_config": sha(snapshot / "RUNTIME_CONFIG.json"),
                "partitions": sha(package / "reference/TRAIN_PARTITIONS.json"), "source_files": files,
                "inherited_package_deviations": deviations, "selector": "historical_scalar_reader_exact_upper_bound",
                "generation_batch_size": args.generation_batch_size,
                "changed_requirements": ["unselected full-pair logits replaced by certified upper bounds"],
                "optional_experiments": ["nested source-group calibration subset only when explicit query budgets are passed"],
                "unchanged": ["Qwen3.5-9B", "C50", "top10_max3", "ROW1", "gate", "singleton", "generation limits", "full dev/test"]}
    immutable_json(out / "MANIFEST.json", manifest)
    run = out / "cohorts" / f"fit{fit_budget}_val{val_budget}_s{args.seed}"
    immutable_json(run / "COHORT.json", cohort)
    selected = set(cohort["query_ids"])
    experiment = run / "experiment"
    config = read(package / "config/EXPERIMENT.json")
    config.update(experiment_id="R5b_FAST_v1_" + run.name, output=str(run), train_count=len(selected))
    immutable_json(experiment / "config/EXPERIMENT.json", config)
    for split in ("dev", "test"):
        link_file(package / "reference" / f"{split}_ids.json", experiment / "reference" / f"{split}_ids.json")
    immutable_json(experiment / "reference/train_ids.json", cohort["query_ids"])
    immutable_json(experiment / "reference/TRAIN_PARTITIONS.json", cohort["partitions"])
    link_file(package / "code", experiment / "code")
    snap = run / "snapshot"
    immutable_json(snap / "population/POPULATION.json", [r for r in pop if r["query_id"] in selected])
    immutable_json(snap / "population/train_ids.json", cohort["query_ids"])
    cache = out / "cache"
    (cache / "query_runs").mkdir(parents=True, exist_ok=True)
    link_file(cache / "query_runs", snap / "query_runs")
    runtime = {**c, "output": str(cache), "cohorts": {"train": len(selected)}, "population_count": len(selected),
               "batch_size": args.generation_batch_size,
               "prepared_functional_hash": digest(manifest),
               "population_regime": "FULL_TRAIN" if len(selected) == len(pop) else "SOURCE_GROUP_CALIBRATION_SUBSET"}
    immutable_json(snap / "RUNTIME_CONFIG.json", runtime)
    return dict(package=package, source=source, original=snapshot, out=out, run=run, snap=snap,
                cache=cache, config=runtime, cohort=cohort, identity=digest(manifest), parent=parent)


def bind_runtime(state: dict):
    sys.path.insert(0, str(state["package"] / "train_runtime/code"))
    import common
    # Hash once per immutable process, and bind the new adapter into raw identity.
    original_hash = common.functional_hash()
    code_hash = digest({"original_runtime": original_hash, "fast_manifest": state["identity"]})
    common.functional_hash = functools.cache(lambda: code_hash)
    return common


def prepare(state: dict) -> None:
    rt = bind_runtime(state)
    from canonical import CanonicalReader, full_table, comparable, query_rows
    from e2e_plan import build_plan
    from e2e_utils import bind_module, gpu_guard
    gpu_guard()
    sys.path.insert(0, str(state["package"] / "train_runtime/code/legacy_runtime"))
    from io_utils import model_identity
    require(model_identity(Path(state["config"]["generator_model"]))["full_bytes_fingerprint"] == state["config"]["generator_fingerprint"],
            "Changed 9B model bytes")
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(13)
    torch.cuda.manual_seed_all(13)
    c, snap, cache = state["config"], state["snap"], state["cache"]
    wanted = state["cohort"]["query_ids"]
    query = read(state["original"] / "population/VISIBLE_QUERIES.json")
    seals = read(state["original"] / "STAGE1_SEAL.json")
    stages, needed = {}, set()
    for q in wanted:
        rel = f"stage1/train/{q}.json"
        source = state["original"] / rel
        require(sha(source) == seals[rel], "Changed sealed Stage1 query: " + q)
        stages[q] = read(source)
        needed.update(stages[q]["C50"])
        link_file(source, snap / rel)
    reader = CanonicalReader(c["original_dataset"])
    raw, tables = {}, {}
    for kind in ("target_tables", "lake_tables"):
        for record in reader.scan(kind, optional=(kind == "target_tables")):
            t = str(record["table_id"])
            if t not in needed:
                continue
            table = full_table(record)
            if t in tables:
                require(comparable(table) == comparable(tables[t]), "Conflicting canonical target: " + t)
            else:
                raw[t] = {"table_id": t, "columns": record["columns"], "rows": record["rows"]}
                tables[t] = table
    require(set(tables) == needed, "Missing canonical targets")
    hashes = {t: digest(raw[t]) for t in raw}
    binding, adapter = bind_module(), None
    assets, plans, prepare_records = {}, [], {}
    d = {"cohort": "train", "baselines": {}, "query_rows": {}, "source_groups": {}, "tables": tables, "assets": {}}
    started = time.perf_counter()
    for index, q in enumerate(wanted):
        stage = stages[q]
        base, retained = stage["C50"], stage["pool"]["retained_paths"]
        signature = digest({"code": state["identity"], "query": query[q], "stage": stage,
                            "tables": {t: hashes[t] for t in base}})
        path = cache / "selector" / (q + ".json")
        if path.exists():
            record = read(path)
            require(record["signature"] == signature and record["payload_hash"] == digest(record["payload"]), "Stale selector cache: " + q)
            payload = record["payload"]
        else:
            if adapter is None:
                adapter = binding.OriginalSelectorAdapter(c["selector_checkpoint"], "cuda:0")
            def score(t):
                return adapter._score_current_scalar_legacy(q, query[q], t, raw[t], list(retained.get(t, [])), {})
            result = select_lazy(base, stage["path_logits"], {t: [int(col["column_index"]) for col in raw[t]["columns"]] for t in base}, score)
            plan, pairs = build_lazy_plan(q, base, stage["path_logits"], retained, tables, result, build_plan)
            payload = {**result, "plan": plan, "evaluated_pairs": pairs}
            write(path, {"signature": signature, "payload_hash": digest(payload), "payload": payload})
        plan = payload["plan"]
        used = {e for view in plan["A_P0"]["views"] for e in view["evidence_ids"]}
        missing = sorted(used - assets.keys())
        if missing:
            hydrated = binding.hydrate_evidence(missing, c["original_dataset"])
            require(set(hydrated) == set(missing), "Evidence hydration changed the source set")
            for e, asset in hydrated.items():
                require(asset["asset_id"] == e and bool(asset.get("source_record_sha256")), "Missing evidence provenance")
                if asset["asset_type"] == "image":
                    src = Path(asset["local_path"])
                    require(sha(src) == asset["source_sha256"], "Changed original image")
                    dest = cache / "input_images" / (asset["source_sha256"] + src.suffix.lower())
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if not dest.exists():
                        shutil.copy2(src, dest)
                    require(sha(dest) == asset["source_sha256"], "Changed staged image")
                    asset = {**asset, "local_path": str(dest)}
                assets[e] = asset
        d["assets"].update({e: assets[e] for e in used})
        d["baselines"][q], d["query_rows"][q] = base, query_rows(query[q])
        d["source_groups"][q] = query[q]["source_table_id"]
        plans.append(plan)
        prepare_records[q] = digest({"selector_signature": signature, "plan": plan, "assets": {e: assets[e] for e in sorted(used)}})
        if (index + 1) % 10 == 0 or index + 1 == len(wanted):
            print(json.dumps({"stage": "selector", "completed": index + 1, "total": len(wanted),
                              "tables_this_query": len(payload["selector"]), "seconds": time.perf_counter() - started}), flush=True)
    (cache / "input_images").mkdir(parents=True, exist_ok=True)
    link_file(cache / "input_images", snap / "prepared/train/input_images")
    write(snap / "prepared/train/INPUTS.json", d)
    write(snap / "prepared/train/PLANS.json", plans)
    write(snap / "PREPARATION.json", {"query_signatures": prepare_records, "sources": reader.receipts, "selector_seconds": time.perf_counter() - started})
    rt.seal(snap, "FAST_PREPARE_SEAL.json", [snap / "prepared/train/INPUTS.json", snap / "prepared/train/PLANS.json", snap / "PREPARATION.json"])


def generate(state: dict) -> None:
    rt = bind_runtime(state)
    from pipeline import QueryRunner
    snap, cache, c = state["snap"], state["cache"], state["config"]
    rt.verify(snap, "FAST_PREPARE_SEAL.json")
    d = read(snap / "prepared/train/INPUTS.json")
    plans = {p["query_id"]: p for p in read(snap / "prepared/train/PLANS.json")}
    signatures = read(snap / "PREPARATION.json")["query_signatures"]
    wanted = state["cohort"]["query_ids"]
    require(set(plans) == set(d["baselines"]) == set(wanted), "Incomplete cohort preparation")
    os.environ["MMDD_RUNTIME_CONFIG"] = str(snap / "RUNTIME_CONFIG.json")
    sys.path.insert(0, str(state["package"] / "train_runtime/code/legacy_runtime"))
    import engine as engine_module
    from mmdd_stage2.generation_batching import microbatches
    engine_module.microbatches = microbatches
    Engine = engine_module.Engine
    for start_file in (cache / "attempts").glob("*/START.json"):
        require((start_file.parent / "END.json").exists(), "Unclosed generation attempt; preserve and investigate: " + str(start_file))
    historical_seconds = sum(read(p)["elapsed"] for p in (cache / "attempts").glob("*/END.json"))
    attempt = cache / "attempts" / uuid.uuid4().hex
    started = time.perf_counter()
    write(attempt / "START.json", {"cohort": str(state["run"]), "time_unix": time.time()})
    engine, completed, resumed = None, 0, 0
    def budget_check():
        require(historical_seconds + time.perf_counter() - started < c["total_gpu_budget_seconds"], "Generation GPU budget exhausted")
    try:
        for q in sorted(wanted, key=lambda q: (digest([c["query_order_salt"], "train", q]), q)):
            dest = cache / "query_runs/train" / q / "N_MISSING_ALL"
            signature = {"code": state["identity"], "input": signatures[q]}
            if (dest / "COMPLETE.json").exists():
                require(read(dest / "COMPLETE.json")["signature"] == signature, "Changed generation query: " + q)
                rt.verify(dest, "QUERY_SEAL.json")
                for rel, fingerprint in read(dest / "RAW_REFERENCES.json").items():
                    require(sha(cache / rel) == fingerprint, "Changed generation raw batch: " + rel)
                resumed += 1
                continue
            if engine is None:
                engine = Engine(snap / "prepared", cache)
            result = QueryRunner(engine, d, q, "N_MISSING_ALL", dest, c, budget_check).execute(plans[q])
            refs = {r["raw_batch_ref"] for r in result["logical_outputs"] if r.get("raw_batch_ref")}
            write(dest / "RAW_REFERENCES.json", {r: sha(cache / r) for r in refs})
            rt.seal(dest, "QUERY_SEAL.json", [p for p in dest.glob("*.json") if p.name not in {"COMPLETE.json", "QUERY_SEAL.json"}])
            write(dest / "COMPLETE.json", {"signature": signature, "attempt": attempt.name})
            completed += 1
            if completed % 10 == 0:
                print(json.dumps({"stage": "generation", "fresh": completed, "resumed": resumed, "total": len(wanted), "seconds": time.perf_counter() - started}), flush=True)
        write(attempt / "END.json", {"status": "COMPLETE", "fresh": completed, "resumed": resumed, "elapsed": time.perf_counter() - started})
        write(snap / "GENERATION_RECEIPT.json", {"queries": len(wanted), "fresh": completed, "resumed": resumed, "elapsed": time.perf_counter() - started,
                                               "full_train_census": len(wanted) == state["cohort"]["full_train_queries"]})
    except BaseException as error:
        write(attempt / "END.json", {"status": "INCOMPLETE", "error_type": type(error).__name__, "elapsed": time.perf_counter() - started})
        raise


def cpu_stage(state: dict, action: str) -> None:
    sys.path.insert(0, str(state["package"] / "code"))
    import r5b_common
    # Bind explicit cohort references/config without changing the frozen package.
    r5b_common.ROOT = state["run"] / "experiment"
    run, features = state["run"], state["run"] / "features"
    label_root = state["source"] / "labels"
    r5b_common.verify_seal(label_root)
    gold = read(label_root / "TRAIN_GOLD.json")
    immutable_json(run / "labels/TRAIN_GOLD.json", {q: gold[q] for q in state["cohort"]["query_ids"]})
    for split in ("DEV", "TEST"):
        link_file(label_root / f"{split}_LABELS.json", run / "labels" / f"{split}_LABELS.json")
    if action == "features":
        from build_features import run as build
        require((state["snap"] / "GENERATION_RECEIPT.json").exists(), "Generation has not completed")
        # The native builder checks C50 and reproduces A/PURE full rankings.
        build(state["snap"], "train", features)
        shared = state["out"] / "evaluation_features"
        for split in ("dev", "test"):
            existing = state["source"] / "features"
            use = existing if (existing / f"{split}_SEAL.json").exists() else shared
            if use == shared:
                build(state["parent"], split, shared)
            r5b_common.verify_seal(use, f"{split}_SEAL.json")
            for suffix in (".npz", "_meta.json", "_evidence_details.jsonl.gz", "_timing.csv", "_receipt.json", "_SEAL.json"):
                link_file(use / (split + suffix), features / (split + suffix))
    elif action == "fit":
        from train_calibrators import run as fit
        fit(features, run / "models", run / "labels")
    elif action == "evaluate":
        from evaluate import run as evaluate
        evaluate(features, run / "models", run / "labels", run / "evaluation")
        from make_review import run as review
        review(features, run / "evaluation", state["parent"], run / "review")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', help='Retired experiment action')
    parser.parse_args()
    parser.error('R5b calibration and train generation were cancelled. '
                 'Use run_stage2_columns_r2.py train for selector training; '
                 'see SELECTOR_TRAINING.md. No calibration work was started.')


if __name__ == "__main__":
    main()
