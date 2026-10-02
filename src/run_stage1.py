#!/usr/bin/env python
"""Stage-1 main flow: dataset artifact -> frozen features -> CQET Teacher/Student -> Stage 2.

Every command takes ``--run-root``; the run is described by ``<run-root>/protocol.json``,
written once by ``init`` from ``configs/mmdd_stage1_cqet_protocol.json``. The frozen
feature layer lives in its own directory (``--features-dir``, default
``work/stage1_features/<dataset name>``) so several runs can train on one encoding:

    <features-dir>/data/      stage1_objects, edge_lists, target_lists, stage1_corpus (build-data)
    <features-dir>/encoder/   two-tier Qwen cache merged from two shards (encode)
    <features-dir>/features/  packed retrieval z and content tokens (encode / pack)

Order:

    init --dataset-root D --gpu N   bind dataset, backbone, features dir, run root and GPU
         [--side-gpu M]             optional second GPU for train-side
    build-data                      write <features-dir>/data (refuses to overwrite)
    encode                          write <features-dir>/{encoder,features} (refuses to overwrite)
    lock                            hash dataset/backbone/cache, content-alias archive
    verify-features                 re-encode a fixed sample and compare with the cache (GPU)
    prepare                         train labels, row store, run PCA
    validate                        CPU reference/contract tests bound to the source hash
    smoke                           every stage on 8 train queries (GPU)
    train                           TA, TB_{CQET,QT}, Native/QT C1, Native C2 SUP/KD, QT C2,
                                    dev selection freeze, then frozen dev/test evaluation (GPU)
    train-side                      with --side-gpu, run next to train on the same run root: TB_QT,
                                    QT C1/C2, the Teacher trajectory and the test split (side GPU)
    export                          retrieval.{train,dev,test}.jsonl + stage1_gate.json for Stage 2

GPU commands set CUDA_VISIBLE_DEVICES to the protocol's ``hardware.uuid`` (``train-side``:
``hardware.side_uuid``) before torch is imported, so the process sees exactly that device as
``cuda:0``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs" / "mmdd_stage1_cqet_protocol.json"
DEFAULT_BACKBONE = ROOT / "hf_models" / "Qwen3-VL-Embedding-8B"
FEATURES_ROOT = ROOT / "work" / "stage1_features"
# The feature-provenance probe replays the original batches, which assume exactly two
# encoder shards (stage1_objects position mod 2; content by object-id hash mod 2).
ENCODER_SHARDS = 2
TARGET_MAX_ROWS = 20
GPU_COMMANDS = {"verify-features", "smoke", "train", "train-side", "export"}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def feature_paths(features_dir: Path) -> dict[str, str]:
    """Protocol ``paths`` entries that name one frozen-feature directory."""
    return {
        "features_dir": str(features_dir),
        "pure_cache_dir": str(features_dir / "features"),
        "row_cache_manifest": str(features_dir / "encoder" / "manifest.jsonl"),
        "upstream_cache_dir": str(features_dir / "encoder"),
        "upstream_data_dir": str(features_dir / "data"),
    }


def features_root(protocol: dict) -> Path:
    """The feature directory a protocol binds (``upstream_data_dir``'s parent for older layouts)."""
    paths = protocol["paths"]
    return Path(paths.get("features_dir") or Path(paths["upstream_data_dir"]).parent)


def build_protocol(
    template: dict,
    *,
    dataset_root: Path,
    run_root: Path,
    features_dir: Path,
    backbone_dir: Path,
    hardware: dict,
) -> dict:
    """Bind one run's inputs, feature directory, outputs and GPU into a copy of the template."""
    protocol = json.loads(json.dumps(template))
    protocol["hardware"].update(hardware)
    protocol["paths"] = {
        "dataset_root": str(dataset_root),
        "backbone_dir": str(backbone_dir),
        **feature_paths(features_dir),
        "run_root": str(run_root),
    }
    protocol["max_registered_stages"] = protocol["stages_per_seed"] * len(protocol["seeds"])
    return protocol


def query_gpu(index: int) -> dict:
    """Read the UUID and model of one physical GPU (PCI bus order) from nvidia-smi."""
    output = subprocess.check_output(
        ["nvidia-smi", "-i", str(index), "--query-gpu=uuid,name", "--format=csv,noheader"],
        text=True,
    )
    uuid, name = (value.strip() for value in output.strip().split(",", 1))
    return {"physical_index": index, "uuid": uuid, "model": name}


def init(run: Path, dataset_root: Path, features_dir: Path, backbone_dir: Path, gpu: int,
         seeds: list[int] | None, side_gpu: int | None = None) -> None:
    protocol_path = run / "protocol.json"
    if protocol_path.exists():
        raise FileExistsError(f"run already initialised: {protocol_path}")
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    if seeds:
        template["seeds"] = seeds
    hardware = query_gpu(gpu)
    if side_gpu is not None:
        side = query_gpu(side_gpu)
        if side["model"] != hardware["model"]:
            raise ValueError(f"side GPU {side['model']} differs from main GPU {hardware['model']}")
        hardware.update(gpu_processes=2, side_physical_index=side_gpu, side_uuid=side["uuid"])
    protocol = build_protocol(
        template, dataset_root=dataset_root, run_root=run, features_dir=features_dir,
        backbone_dir=backbone_dir, hardware=hardware,
    )
    from mmdd_stage1.config import validate_protocol

    validate_protocol(protocol)
    write_json(protocol_path, protocol)
    encoded = (features_dir / "features" / "z" / "z.f32.npy").is_file()
    print(json.dumps({"protocol": str(protocol_path), "paths": protocol["paths"],
                      "hardware": protocol["hardware"],
                      "features": "existing encoding found; skip build-data and encode" if encoded
                      else "not encoded yet; run build-data then encode"}, indent=2), flush=True)


def build_data(protocol: dict, dataset_name: str | None = None) -> None:
    from mmdd_stage1.construction import build_stage1_training_artifacts

    data = features_root(protocol) / "data"
    if (data / "stage1_objects.jsonl").exists():
        raise FileExistsError(f"{data} already holds Stage-1 data; it is shared across runs, choose another --features-dir to rebuild")
    dataset_root = Path(protocol["paths"]["dataset_root"])
    artifacts = build_stage1_training_artifacts(
        dataset_root, dataset_name=dataset_name or dataset_root.name,
        max_rows=TARGET_MAX_ROWS, seed=13,
    )
    counts = {}
    for name, records in artifacts.items():
        write_rows(data / f"{name}.jsonl", records)
        counts[name] = len(records)
    print(json.dumps({"data": str(data), "records": counts}, indent=2), flush=True)


def _environment(gpu: int | str) -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(gpu)}


def _run_jobs(run: Path, jobs: list[tuple[str, list[str], int | str]]) -> None:
    """Run ``(name, python arguments, gpu)`` jobs; distinct GPUs run concurrently, a shared GPU in order."""
    logs = run / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    pending = list(jobs)
    while pending:
        batch = []
        for job in list(pending):
            if job[2] not in {gpu for _name, _arguments, gpu in batch}:
                batch.append(job)
                pending.remove(job)
        processes = []
        for name, arguments, gpu in batch:
            print(f"START {name} on GPU {gpu}: {logs / (name + '.log')}", flush=True)
            with (logs / f"{name}.log").open("w") as handle:
                processes.append((name, subprocess.Popen(
                    [sys.executable, *arguments], cwd=run, env=_environment(gpu),
                    stdout=handle, stderr=subprocess.STDOUT,
                )))
        failed = [name for name, process in processes if process.wait()]
        if failed:
            raise RuntimeError(f"{failed} failed; inspect their logs in {logs}")
        print("DONE " + " ".join(name for name, _process in processes), flush=True)


def encode(run: Path, protocol: dict, gpus: list[int | str]) -> None:
    """Two-shard frozen encoding, content tokens for evidence, then pack ``features/``.

    Everything is written under the protocol's feature directory; ``run`` only receives logs.
    """
    from cache_stage1_features import teacher_object_ids

    root = features_root(protocol)
    data = root / "data"
    if (root / "features" / "z" / "z.f32.npy").exists():
        raise FileExistsError(f"{root} is already encoded; it is shared across runs, choose another --features-dir to re-encode")
    backbone = protocol["paths"]["backbone_dir"]
    objects = read_rows(data / "stage1_objects.jsonl")
    teacher_ids = teacher_object_ids([data / "edge_lists.jsonl", data / "target_lists.jsonl"], split=None)
    # Teacher reranking scores every lake table, including tables absent from the lists.
    teacher_ids.update(row["object_id"] for row in objects if row["object_type"] == "table")
    jobs = []
    for shard in range(ENCODER_SHARDS):
        shard_objects = objects[shard::ENCODER_SHARDS]
        write_rows(data / f"objects_shard{shard}.jsonl", shard_objects)
        write_rows(data / f"teacher_shard{shard}.jsonl", [
            {"object_id": object_id}
            for object_id in sorted(teacher_ids & {row["object_id"] for row in shard_objects})
        ])
        jobs.append((f"encode_base_{shard}", [
            str(ROOT / "src" / "cache_stage1_features.py"),
            "--input-jsonl", str(data / f"objects_shard{shard}.jsonl"),
            "--output-dir", str(root / f"encoder_shard{shard}"), "--model-dir", backbone,
            "--device", "cuda:0", "--object-batch-size", "4",
            "--teacher-data", str(data / f"teacher_shard{shard}.jsonl"), "--teacher-split", "all",
        ], gpus[shard % len(gpus)]))
    _run_jobs(run, jobs)

    merged = root / "encoder"
    for tier in ("objects", "teacher_objects"):
        (merged / tier).mkdir(parents=True, exist_ok=True)
        for shard in range(ENCODER_SHARDS):
            for source in (root / f"encoder_shard{shard}" / tier).glob("*.pt"):
                target = merged / tier / source.name
                if not target.exists():
                    os.link(source, target)
    for filename in ("manifest.jsonl", "teacher_manifest.jsonl"):
        records = [row for shard in range(ENCODER_SHARDS)
                   for row in read_rows(root / f"encoder_shard{shard}" / filename)]
        records.sort(key=lambda row: row["object_id"].encode("utf-8"))
        if len({row["object_id"] for row in records}) != len(records):
            raise ValueError(f"duplicate object across encoder shards in {filename}")
        write_rows(merged / filename, records)
    (merged / "metadata.json").write_bytes((root / "encoder_shard0" / "metadata.json").read_bytes())

    _run_jobs(run, [
        (f"encode_content_{shard}", [
            "-m", "mmdd_stage1.content_encoder",
            "--dataset-root", protocol["paths"]["dataset_root"], "--backbone-dir", backbone,
            "--cache-dir", str(merged), "--out-dir", str(root / f"content_shard{shard}"),
            "--shard", str(shard), "--num-shards", str(ENCODER_SHARDS), "--kinds", "text,image",
        ], gpus[shard % len(gpus)])
        for shard in range(ENCODER_SHARDS)
    ])
    pack(root)


def pack(root: Path) -> None:
    """Pack retrieval ``z`` and per-object content tokens into ``<features-dir>/features``."""
    import numpy as np
    import torch

    from mmdd_stage1 import content

    content.build_z_memmap(root / "encoder", root / "features" / "z")
    rows = []
    for entry in content.read_manifest(root / "encoder" / "teacher_manifest.jsonl", "teacher_feature_path"):
        if entry.object_type == "table":
            payload = torch.load(root / "encoder" / entry.path, map_location="cpu", weights_only=False)
            rows.append((entry.object_id, entry.object_type,
                         content.table_tokens(payload["hidden_states"]).numpy()))
    content.write_chunk(root / "table_content" / "chunks", 0, rows)
    sources = [root / "table_content" / "chunks",
               *(root / f"content_shard{shard}" / "chunks" for shard in range(ENCODER_SHARDS))]
    index = content.merge_chunk_dirs(sources, root / "features" / "content" / "chunks")
    expected = {row["object_id"] for row in read_rows(root / "data" / "stage1_objects.jsonl")}
    missing = expected - set(index["ids"])
    if missing:
        raise ValueError(f"content tokens missing for {len(missing)} objects; first={sorted(missing)[:5]}")
    if not np.isfinite(np.load(root / "features" / "z" / "z.f32.npy", mmap_mode="r")).all():
        raise ValueError("non-finite frozen embeddings")
    print(json.dumps({"content_objects": len(index["ids"]), "status": "PASS"}), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    commands = ("init", "build-data", "encode", "pack", "lock", "verify-features", "prepare",
                "validate", "smoke", "train", "train-side", "export", "amend-source")
    for command in commands:
        subparsers.add_parser(command).add_argument("--run-root", type=Path, required=True)
    init_parser = subparsers.choices["init"]
    init_parser.add_argument("--dataset-root", type=Path, required=True)
    init_parser.add_argument("--gpu", type=int, required=True, help="physical GPU index (PCI bus order)")
    init_parser.add_argument("--features-dir", type=Path,
                             help=f"frozen-feature directory, shared across runs (default: {FEATURES_ROOT}/<dataset name>)")
    init_parser.add_argument("--backbone-dir", type=Path, default=DEFAULT_BACKBONE)
    init_parser.add_argument("--seeds", type=int, nargs="+")
    init_parser.add_argument("--side-gpu", type=int,
                             help="physical index of a second GPU of the same model for train-side")
    subparsers.choices["lock"].add_argument(
        "--workers", type=int, default=32, help="processes decoding evidence images for content aliases")
    subparsers.choices["encode"].add_argument(
        "--gpus", type=int, nargs="+", help="physical GPU indices for the two shards (default: protocol GPU)")
    export_parser = subparsers.choices["export"]
    export_parser.add_argument("--output-dir", type=Path, help="default: <run-root>/stage2_handoff")
    export_parser.add_argument("--seed", type=int, help="default: first protocol seed")
    export_parser.add_argument("--arm", choices=("SUP", "KD"), default="KD",
                               help="which dev-selected Native C2 Student to export (protocol primary: KD)")
    export_parser.add_argument("--top-k", type=int, default=50)
    export_parser.add_argument("--evidence-path-k", type=int, default=4)
    amend = subparsers.choices["amend-source"]
    amend.add_argument("--amendment-id", required=True)
    amend.add_argument("--carry", nargs="+", required=True)
    amend.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    run = args.run_root.resolve()

    if args.command == "init":
        dataset_root = args.dataset_root.resolve()
        features_dir = (args.features_dir or FEATURES_ROOT / dataset_root.name).resolve()
        init(run, dataset_root, features_dir, args.backbone_dir.resolve(), args.gpu, args.seeds, args.side_gpu)
        return
    protocol_path = run / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if args.command in GPU_COMMANDS:
        # Must happen before the first torch import in this process.
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = protocol["hardware"][
            "side_uuid" if args.command == "train-side" else "uuid"]

    if args.command == "build-data":
        build_data(protocol)
    elif args.command == "encode":
        encode(run, protocol, args.gpus or [protocol["hardware"]["physical_index"]])
    elif args.command == "pack":
        pack(features_root(protocol))
    elif args.command in ("lock", "verify-features"):
        from mmdd_stage1 import preflight
        from mmdd_stage1.config import resolve_default_paths

        paths = resolve_default_paths(protocol_path, run)
        if args.command == "lock":
            preflight.run_lock(paths, args.workers)
        else:
            preflight.verify_feature_provenance(paths)
    elif args.command == "export":
        from mmdd_stage1.export import export_stage2

        manifest = export_stage2(
            protocol_path, run, (args.output_dir or run / "stage2_handoff").resolve(),
            seed=args.seed if args.seed is not None else protocol["seeds"][0], arm=args.arm,
            top_k=args.top_k, evidence_path_k=args.evidence_path_k,
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    else:
        from mmdd_stage1 import pipeline

        if args.command == "prepare":
            pipeline.prepare(protocol_path, run)
        elif args.command == "validate":
            pipeline.validate(protocol_path, run)
        elif args.command == "smoke":
            pipeline.smoke(protocol_path, run)
        elif args.command == "train":
            pipeline.run_formal(protocol_path, run)
        elif args.command == "train-side":
            pipeline.run_side(protocol_path, run)
        else:
            unknown = set(args.carry) - set(pipeline.STAGES)
            if unknown:
                parser.error(f"unknown stages for --carry: {sorted(unknown)}")
            pipeline.amend_source(protocol_path, run, amendment_id=args.amendment_id,
                                  carried_stages=args.carry, reason=args.reason)


if __name__ == "__main__":
    main()
