"""Run the current CQET implementation on fresh AbeBooks features.

This adapter binds dataset/output paths only. The current training, retrieval,
selection, and feature-provenance checks are called without algorithm changes.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def environment(gpu: int | str) -> dict[str, str]:
    """Child-process environment pinned to one GPU (physical index in PCI bus order, or a UUID)."""
    return {**os.environ, "PYTHONPATH": str(ROOT / "src"),
            "CUDA_VISIBLE_DEVICES": str(gpu), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"}


def launch(run: Path, name: str, arguments: list[str], gpu: int | str) -> subprocess.Popen:
    log = run / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, *arguments]
    with (run / "COMMANDS.jsonl").open("a") as handle:
        handle.write(json.dumps({"name": name, "command": command, "cwd": str(run / "isolated_cwd"), "gpu": str(gpu)}) + "\n")
    print(f"START {name}: {log}", flush=True)
    with log.open("w") as handle:
        return subprocess.Popen(command, cwd=run / "isolated_cwd", env=environment(gpu), stdout=handle, stderr=subprocess.STDOUT)


def finish(name: str, process: subprocess.Popen) -> None:
    code = process.wait()
    if code:
        raise RuntimeError(f"{name} exited {code}; inspect its log")
    print(f"DONE {name}", flush=True)


def prepare_data(run: Path, dataset: Path, gpu: int) -> None:
    """Copy and filter the dataset into ``run/dataset_view``, bind the protocol, build Stage-1 data.

    Features go to ``work/stage1_features/<run name>``: the view is filtered per run, so the
    encoding is per run as well, but it still lives outside the run directory.
    """
    if run.exists():
        raise FileExistsError(f"Fresh run requires a new output directory: {run}")
    (run / "isolated_cwd").mkdir(parents=True)
    view = run / "dataset_view"
    manifest = json.loads((dataset / "dataset_manifest.json").read_text())
    files = ["dataset_manifest.json", *manifest["single_files"].values()]
    for artifact in manifest["artifacts"].values():
        files.extend(shard["path"] for shard in artifact["shards"])
    for relative in files:
        target = view / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dataset / relative, target)
    assets = read_rows(view / "bridge_assets/part-00000.jsonl")
    rejected, kept = [], []
    for asset in assets:
        kind = asset["asset_type"]
        reason = None
        if kind == "text" and not str(asset.get("content") or "").strip():
            reason = "empty_text"
        elif kind == "image":
            raw_path = asset.get("local_path") or asset.get("relative_path")
            if not raw_path or not Path(raw_path).is_file():
                reason = "missing_local_image"
        (rejected if reason else kept).append({**asset, "exclusion_reason": reason} if reason else asset)
    rejected_ids = {row["asset_id"] for row in rejected}
    witness_ids = {row["evidence"]["asset_id"] for row in read_rows(view / "evidence_recoveries/part-00000.jsonl")}
    if rejected_ids & witness_ids:
        raise ValueError("An unusable asset is a positive witness; cannot omit it")
    write_rows(run / "unusable_assets.jsonl", rejected)
    write_rows(view / "bridge_assets/part-00000.jsonl", kept)
    manifest["artifacts"]["bridge_assets"]["total_records"] = len(kept)
    manifest["artifacts"]["bridge_assets"]["shards"][0]["records"] = len(kept)
    write_json(view / "dataset_manifest.json", manifest)
    write_json(run / "FRESH_INPUTS.json", {"dataset": str(dataset), "old_features_reused": False,
               "old_checkpoints_reused": False, "excluded_assets": len(rejected),
               "excluded_positive_witnesses": 0, "feature_backbone": str(ROOT / "hf_models/Qwen3-VL-Embedding-8B")})
    import run_stage1

    template = json.loads(run_stage1.TEMPLATE.read_text())
    template["seeds"] = [13]
    template["evaluation"]["k"] = [5, 10, 15, 20]
    protocol = run_stage1.build_protocol(
        template, dataset_root=view, run_root=run, features_dir=run_stage1.FEATURES_ROOT / run.name,
        backbone_dir=run_stage1.DEFAULT_BACKBONE, hardware=run_stage1.query_gpu(gpu),
    )
    protocol["paths"]["package_dir"] = str(run / "protocol_package")
    write_json(run / "protocol.json", protocol)
    write_json(run / "protocol_package/next_round/protocol.json", protocol)
    run_stage1.build_data(protocol, dataset_name=dataset.name)
    print(json.dumps({"usable_assets": len(kept), "excluded_assets": len(rejected),
                      "features_dir": protocol["paths"]["features_dir"]}), flush=True)


def encode(run: Path, gpus: list[int]) -> None:
    import run_stage1

    run_stage1.encode(run, json.loads((run / "protocol.json").read_text()), gpus)


def bind(run: Path):
    from mmdd_stage1 import pipeline, preflight, provenance
    from mmdd_stage1.config import resolve_default_paths

    original_sources = provenance.source_files
    provenance.source_files = lambda paths: [*original_sources(paths), Path(__file__).resolve()]
    return pipeline, preflight, resolve_default_paths(run / "protocol.json", run)


def endpoint_evaluation(run: Path) -> None:
    from mmdd_stage1.artifacts import save_pool_bundle
    from mmdd_stage1.data import load_split_gt
    from mmdd_stage1.evaluate import evaluate_student_retrieval, evaluate_teacher_matrix

    pipeline, _preflight, _paths = bind(run)
    freeze = json.loads((run / "GLOBAL_SELECTION_FREEZE.json").read_text())
    if freeze["status"] != "ALL_SELECTIONS_FROZEN_BEFORE_TEST":
        raise ValueError("Endpoint diagnostics require completed selection")
    rt = pipeline.load_runtime(run / "protocol.json", run)
    teacher = pipeline._load_teacher(run / "seed13/TB_CQET/checkpoints/end.pt")
    for arm in ("SUP", "KD"):
        checkpoint = run / f"seed13/NATIVE_C2_{arm}/checkpoints/snapshot_frac100.pt"
        model = pipeline._load_native(checkpoint, rt)
        for split in ("dev", "test"):
            gt = load_split_gt(rt.paths, split, rt.labels.canonical_map)
            generator = f"endpoint_{arm.lower()}"
            output = run / f"seed13/eval/{split}/{generator}"
            pools = evaluate_student_retrieval(model, rt.z_store, rt.row_store, sorted(gt), rt.labels,
                                               split, hnsw_seed=13, generator_id=generator)
            save_pool_bundle(output, pools, rt.labels, seed=13, generator=generator)
            evaluate_teacher_matrix({"TB_CQET": teacher}, rt.bank, pools, rt.labels, seed=13,
                                    generator=generator, split=split, output_dir=output, split_gt=gt)
    print("DONE endpoint diagnostics (read-only; no checkpoint selection)", flush=True)


def summarize(run: Path) -> None:
    from collections import defaultdict
    from statistics import mean

    def compressed_rows(path: Path) -> list[dict]:
        with gzip.open(path, "rt") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    ks = (5, 10, 15, 20)
    summaries, per_query = [], []
    for split in ("dev", "test"):
        queries = {r["query_id"]: r for r in read_rows(run / f"eval_labels/{split}/queries.jsonl")}
        gold = defaultdict(set)
        for row in read_rows(run / f"eval_labels/{split}/qrels.jsonl"):
            if row["rel"] > 0:
                gold[row["query_id"]].add(row["target_id"])
        for generator in ("raw", "native_sup", "native_kd", "endpoint_sup", "endpoint_kd"):
            directory = run / f"seed13/eval/{split}/{generator}"
            pools = compressed_rows(directory / "pools.jsonl.gz")
            rankings = {
                "Direct": {r["query_id"]: r["D100_ANN"] for r in pools},
                "Multimodal_RRF": {r["query_id"]: r["C150"] for r in pools},
            }
            for view in ("Real", "f0", "Swap"):
                rankings[f"Teacher_{view}"] = {r["query_id"]: r["target_ids"]
                    for r in compressed_rows(directory / f"rankings.TB_CQET.{view}.jsonl.gz")}
            for mode, orders in rankings.items():
                if set(orders) != set(gold):
                    raise ValueError(f"Evaluation query mismatch: {split}/{generator}/{mode}")
                scores = {}
                for query, order in orders.items():
                    if len(order) != len(set(order)):
                        raise ValueError("Duplicate ranked targets")
                    scores[query] = {f"R@{k}": len(set(order[:k]) & gold[query]) / len(gold[query]) for k in ks}
                    per_query.append({"split": split, "generator": generator, "mode": mode,
                                      **queries[query], "gold_count": len(gold[query]), **scores[query]})
                for segment in ("overall", "implicit", "explicit"):
                    ids = [q for q in gold if segment == "overall" or queries[q]["query_kind"] == segment]
                    summaries.append({"split": split, "generator": generator, "mode": mode,
                                      "segment": segment, "queries": len(ids),
                                      **{f"R@{k}": mean(scores[q][f"R@{k}"] for q in ids) if ids else None for k in ks}})
    selection = json.loads((run / "seed13/SELECTION_FREEZE.json").read_text())
    write_json(run / "recall_summary.json", {"definition": "query macro |top-k intersect gold| / |gold|",
               "seed": 13, "selection": selection, "rows": summaries})
    write_rows(run / "recall_per_query.jsonl", per_query)
    lines = ["# AbeBooks fresh SUP/KD experiment", "", "Fresh features and training from current src; seed 13.",
             "Recall is query-macro target recall. The existing test split has been evaluated historically.",
             "Selected checkpoints use only dev; endpoint rows are fixed full-epoch diagnostics, not new selection.", ""]
    for split in ("test", "dev"):
        lines += [f"## {split}", "", "| Model | Ranking | R@5 | R@10 | R@15 | R@20 |", "|---|---|---:|---:|---:|---:|"]
        for row in summaries:
            if row["split"] == split and row["segment"] == "overall":
                lines.append(f"| {row['generator']} | {row['mode']} | " + " | ".join(f"{100*row[f'R@{k}']:.2f}%" for k in ks) + " |")
        lines.append("")
    (run / "RECALL_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps([r for r in summaries if r["split"] == "test" and r["segment"] == "overall"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare-data", "encode", "pack", "lock", "verify-features", "prepare", "validate", "smoke", "all", "train", "endpoints", "summarize"])
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "abebooks_joinability_bal04_clean")
    parser.add_argument("--gpu", type=int, default=0, help="physical GPU index (PCI bus order) the run is pinned to")
    parser.add_argument("--gpus", type=int, nargs="+", help="encode: GPU indices for the two shards (default: --gpu)")
    args = parser.parse_args()
    run = args.run_root.resolve()
    if args.command == "prepare-data":
        prepare_data(run, args.dataset_root.resolve(), args.gpu)
    elif args.command == "encode":
        encode(run, args.gpus or [args.gpu])
    elif args.command == "pack":
        import run_stage1

        run_stage1.pack(run_stage1.features_root(json.loads((run / "protocol.json").read_text())))
    elif args.command == "train":
        gpu = json.loads((run / "protocol.json").read_text())["hardware"]["uuid"]
        for stage in ("lock", "verify-features", "prepare", "validate", "smoke", "all", "endpoints", "summarize"):
            finish(stage, launch(run, stage, [str(Path(__file__).resolve()), stage, "--run-root", str(run)], gpu))
    elif args.command == "endpoints":
        endpoint_evaluation(run)
    elif args.command == "summarize":
        summarize(run)
    else:
        pipeline, preflight, paths = bind(run)
        if args.command == "lock":
            preflight.run_lock(paths)
        elif args.command == "verify-features":
            preflight.verify_feature_provenance(paths)
        else:
            function = pipeline.run_formal if args.command == "all" else getattr(pipeline, args.command)
            function(run / "protocol.json", run)


if __name__ == "__main__":
    main()
