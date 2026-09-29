"""Replay exact lazy plans against complete logits; optionally time fresh GPU calls."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import statistics
import sys
import time

from mmdd_stage2.lazy_selector import build_lazy_plan, select_lazy
from run_r5b_fast import NAME, REPO, digest, read, require, sha, write


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", type=Path, default=REPO / "audit" / (NAME + "_PACKAGE"))
    p.add_argument("--train", type=Path, default=REPO / "work" / NAME / "train_snapshot")
    p.add_argument("--parent", type=Path, default=REPO / "work/MMDD_E2E_FULL_CLEAN_QET_V4_S13_N_GPU1_v1")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpu-probe", type=int, default=0, help="Fresh exhaustive/lazy scalar comparisons on this many train queries")
    a = p.parse_args()
    sys.path.insert(0, str(a.package / "train_runtime/code"))
    from canonical import CanonicalReader
    from e2e_plan import build_plan
    c = read(a.train / "RUNTIME_CONFIG.json")
    sources = [(a.parent, "dev"), (a.parent, "test"), (a.train, "train")]
    files = {(str(root), split): sorted((root / "selector_outputs" / split).glob("*.json")) for root, split in sources}
    tables, raw = {}, {}
    probe_files = sorted(files[str(a.train), "train"], key=lambda f: digest(f.stem))[:a.gpu_probe]
    probe_targets = {t for f in probe_files for t in read(a.train / "stage1/train" / f.name)["C50"]}
    reader = CanonicalReader(c["original_dataset"])
    for kind in ("target_tables", "lake_tables"):
        for record in reader.scan(kind, optional=(kind == "target_tables")):
            t = str(record["table_id"])
            columns = [{"column_id": int(x["column_index"]), "column_name": x["column_name"]} for x in record["columns"]]
            if t in tables:
                require(tables[t]["columns"] == columns, "Conflicting target schema")
            tables[t] = {"columns": columns}
            if t in probe_targets:
                raw[t] = {"table_id": t, "columns": record["columns"], "rows": record["rows"]}
    report = {"algorithm_source_sha256": sha(REPO / "src/mmdd_stage2/lazy_selector.py"), "splits": [], "gpu_probe": []}
    for root, split in sources:
        count, mismatches = [], []
        for f in files[str(root), split]:
            cr, stage, q = read(f), read(root / "stage1" / split / f.name), f.stem
            base, retained = stage["C50"], stage["pool"]["retained_paths"]
            expected, _ = build_plan(q, base, stage["path_logits"], cr, retained, tables)
            lazy = select_lazy(base, stage["path_logits"], {t: cr[t]["eligible_column_ids"] for t in base}, lambda t: cr[t])
            actual, _ = build_lazy_plan(q, base, stage["path_logits"], retained, tables, lazy, build_plan)
            if actual != expected:
                mismatches.append(q)
            count.append(len(lazy["selector"]))
        row = {"split": split, "queries": len(count), "mean_tables": statistics.mean(count),
               "median_tables": statistics.median(count), "max_tables": max(count),
               "exhaustive_calls": 50 * len(count), "lazy_calls": sum(count), "plan_mismatches": mismatches,
               "calls_saved_fraction": 1 - sum(count) / (50 * len(count)), "histogram": dict(Counter(count))}
        report["splits"].append(row)
        write(a.out, report)
        print(json.dumps(row), flush=True)
        require(not mismatches, "Lazy plan differs from exhaustive plan")
    if a.gpu_probe:
        os.environ["CUDA_VISIBLE_DEVICES"] = c["gpu_uuid"]
        os.environ["OMP_NUM_THREADS"] = "4"
        os.environ["MKL_NUM_THREADS"] = "4"
        import torch
        from e2e_utils import bind_module, gpu_guard
        gpu_guard()
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        adapter = bind_module().OriginalSelectorAdapter(c["selector_checkpoint"], "cuda:0")
        query = read(a.train / "population/VISIBLE_QUERIES.json")
        # Warm model loading and kernels outside either measured schedule.
        q = probe_files[0].stem
        t = read(a.train / "stage1/train" / probe_files[0].name)["C50"][0]
        adapter._score_current_scalar_legacy(q, query[q], t, raw[t], [], {})
        for index, f in enumerate(probe_files):
            q, stage = f.stem, read(a.train / "stage1/train" / f.name)
            base, retained = stage["C50"], stage["pool"]["retained_paths"]
            eligible = {t: [int(col["column_index"]) for col in raw[t]["columns"]] for t in base}
            def score(t):
                return adapter._score_current_scalar_legacy(q, query[q], t, raw[t], retained.get(t, []), {})
            outputs, elapsed = {}, {}
            # Alternate order to avoid consistently favouring warm follow-up calls.
            for mode in (["full", "lazy"] if index % 2 == 0 else ["lazy", "full"]):
                torch.cuda.synchronize()
                start = time.perf_counter()
                outputs[mode] = {t: score(t) for t in base} if mode == "full" else select_lazy(base, stage["path_logits"], eligible, score)
                torch.cuda.synchronize()
                elapsed[mode] = time.perf_counter() - start
            expected, _ = build_plan(q, base, stage["path_logits"], outputs["full"], retained, tables)
            actual, _ = build_lazy_plan(q, base, stage["path_logits"], retained, tables, outputs["lazy"], build_plan)
            errors = [abs(x["logit"] - y["logit"]) for t, r in outputs["lazy"]["selector"].items()
                      for x, y in zip(r["column_logits"], outputs["full"][t]["column_logits"])]
            record = {"query_id": q, "full_seconds": elapsed["full"], "lazy_seconds": elapsed["lazy"],
                      "speedup": elapsed["full"] / elapsed["lazy"], "full_calls": 50,
                      "lazy_calls": len(outputs["lazy"]["selector"]), "plan_equal": actual == expected,
                      "max_abs_logit_error": max(errors, default=0)}
            report["gpu_probe"].append(record)
            write(a.out, report)
            print(json.dumps(record), flush=True)
            require(actual == expected, "Fresh scalar lazy/exhaustive plan mismatch")


if __name__ == "__main__":
    main()
