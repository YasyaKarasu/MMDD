#!/usr/bin/env python
"""Continue the TB_CQET Teacher on the candidate lists of the dev-selected Student.

Hypothesis (docs/entitables_kd_distillation_diagnosis_20261001.zh-CN.md, section 7): TB_CQET
only ever saw Raw (frozen Qwen) candidate lists and loses about 7pp R@10 once it reranks the
pool of a Student that actually learned. Rebuilding the T_B lists from the Student's own
two-hop retrieval of the train queries (``build_tb_records`` on Student pools instead of Raw
pools) and training one more T_B epoch from ``TB_CQET/checkpoints/end.pt`` gives the Teacher
the hard negatives it is later asked to rerank. Acceptance: on the Student's dev pool the
continued Teacher's ``Real`` R@10 should be at least the Student's own ``Direct_ANN_R10`` and
above the initial Teacher's.

The script reads a ``run_stage1`` run root whose Native C2 selection exists and writes only
under ``--out-dir`` (the run root is never modified):

    mine      Student train pools (U, D150, bags, exact top-128 lists) -> records.jsonl.gz
    train     train_tb from --init-checkpoint on those records -> checkpoints/{init,half,end}.pt
    evaluate  init / half / end Teacher on the Student's dev pool and on the Raw dev pool
              (f0, Real, Swap R@10) -> eval/dev/<generator>/METRICS.json, eval/dev/SUMMARY.json

A step whose outputs exist is skipped, so a killed job restarts at the last unfinished step.
``--records`` replaces mining by an existing record file, e.g. the run's
``training_records/TB_SHARED.jsonl.gz`` for the "one more epoch on the Raw lists" control.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from mmdd_stage1.artifacts import load_pool_bundle, save_pool_bundle, save_training_records  # noqa: E402
from mmdd_stage1.data import iter_jsonl, load_split_gt, read_json, sha256_file, utf8_sorted, write_json  # noqa: E402
from mmdd_stage1.evaluate import evaluate_teacher_matrix  # noqa: E402
from mmdd_stage1.labels import Labels  # noqa: E402
from mmdd_stage1.lists import build_tb_records  # noqa: E402
from mmdd_stage1.metrics import bootstrap_contrast, candidate_metrics, evaluate_matrix, summarize  # noqa: E402
from mmdd_stage1.models import model_state_sha, state_sha  # noqa: E402
from mmdd_stage1.pipeline import _gpu_guard, _load_native, _load_teacher, _set_seed, load_runtime  # noqa: E402
from mmdd_stage1.provenance import source_identity  # noqa: E402
from mmdd_stage1.retrieval import PoolRecord, build_pools  # noqa: E402
from mmdd_stage1.train import train_tb  # noqa: E402

TEACHER_POINTS = ("init", "half", "end")


def gpu_uuid(physical_index: int) -> str:
    output = subprocess.check_output(
        ["nvidia-smi", "-i", str(physical_index), "--query-gpu=uuid", "--format=csv,noheader"], text=True,
    )
    return output.strip()


def mine_records(
    z_store, row_store, labels: Labels, student, query_ids: Sequence[str], seed: int, index_dir: Path,
    device: str = "cuda:0",
) -> tuple[list[dict], dict]:
    """T_B records whose candidate lists come from the Student's retrieval instead of the Raw pools.

    ``build_pools`` with ``training_exact`` also fills the exact top-128 QT/QE lists (under the
    Raw-named keys) that ``build_tb_records`` draws the support competitors from, so the record
    builder is reused verbatim: ``targets = U | D150 | G | U32`` with the Student's retained bags.
    """
    pools = build_pools(
        z_store, row_store, query_ids, labels, "train", student=student, generator_id="student_c2",
        hnsw_seed=seed, device=device, index_dir=index_dir, training_exact=True,
    )
    records = [build_tb_records(query_id, pools[query_id], labels, seed) for query_id in query_ids]
    coverage = summarize(
        {q: candidate_metrics(pools[q], set(labels.queries[q]["G"])) for q in query_ids},
        {q: {"kind": "unknown"} for q in query_ids},
    )["overall"]
    summary = {
        "queries": len(records),
        "mean_targets": sum(len(row["targets"]) for row in records) / len(records),
        "mean_paths": sum(len(bag) for row in records for bag in row["natural_bags"].values()) / len(records),
        "mean_support_records": sum(len(row["support_records"]) for row in records) / len(records),
        **{key: coverage[key] for key in ("U_target_coverage", "D150_target_coverage", "C150_target_coverage",
                                          "U_size", "Direct_ANN_R10")},
    }
    return records, summary


def _checkpoint_identity(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {"path": str(path), "sha256": sha256_file(path), "state_sha256": state_sha(payload["model"])}


def train_continuation(rt, records: list[dict], init_checkpoint: Path, out: Path, *, seed: int,
                       epochs: int, lr: float, metadata: dict) -> Path:
    tb = rt.protocol["teacher"]["TB"]
    checkpoints = out / "checkpoints"
    if (checkpoints / "end.pt").exists():
        return checkpoints / "end.pt"
    for stale in (checkpoints, out / "train.jsonl"):
        if stale.exists():  # an interrupted attempt; the loop has no mid-epoch resume
            stale.rename(stale.with_name(f"{stale.name}.failed.{int(time.time())}"))
    _set_seed(seed, "TB_SHARED")
    model = _load_teacher(init_checkpoint)
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    train_tb(
        model, rt.bank, records, mode="cqet", save_dir=checkpoints,
        epochs=epochs, lr=lr, weight_decay=float(tb["wd"]), logical_batch=int(tb["logical_batch_queries"]),
        direct_weight=float(tb["path_direct_weight"]), aggregate_weight=float(tb["path_aggregate_weight"]),
        support_weight=float(tb["support_weight"]), seed=seed, metadata=metadata, log_path=out / "train.jsonl",
    )
    torch.cuda.synchronize()
    write_json(out / "TRAIN_TIMING.json", {
        "wall_seconds": time.time() - started,
        "optimizer_steps": sum(1 for _ in iter_jsonl(out / "train.jsonl")),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    })
    return checkpoints / "end.pt"


def dev_pools(rt, generator: str, student, dev_ids: Sequence[str], run_eval_dir: Path, out_dir: Path,
              seed: int) -> tuple[dict[str, PoolRecord], str]:
    """The pipeline's frozen dev pools when it already wrote them (same model, same seed), else rebuilt."""
    bundle = run_eval_dir / generator
    if (bundle / "POOL_MANIFEST.json").exists():
        pools = load_pool_bundle(bundle)
        return {q: pools[q] for q in dev_ids}, str(bundle)
    pools = build_pools(
        rt.z_store, rt.row_store, dev_ids, rt.labels, "dev", student=student, generator_id=generator,
        hnsw_seed=seed, training_exact=False,
    )
    save_pool_bundle(out_dir, pools, rt.labels, seed=seed, generator=generator)
    return pools, str(out_dir)


def acceptance_summary(metrics: dict, gt: Mapping[str, dict], pools: Mapping[str, PoolRecord],
                       teachers: Sequence[str]) -> dict:
    """Student-side candidate metrics next to each Teacher point's f0/Real/Swap R@10, plus paired contrasts."""
    segments = ("overall", "implicit", "explicit")
    student = {
        q: candidate_metrics(pools[q], set(gt[q]["G"]))["Direct_ANN_R10"] for q in utf8_sorted(pools)
    }
    per_query = metrics["per_query"]
    summary = {
        "candidate": {
            segment: {key: metrics["candidate"][segment].get(key) for key in
                      ("queries", "Direct_ANN_R10", "C150_target_coverage", "U_target_coverage", "D150_target_coverage")}
            for segment in segments
        },
        "teacher": {
            name: {view: {segment: metrics["teacher"][name][view][segment]["R@10"] for segment in segments}
                   for view in ("f0", "Real", "Swap")}
            for name in teachers
        },
        "contrasts": {
            f"{name}.Real_minus_student_Direct_ANN_R10": bootstrap_contrast(
                {q: per_query[q][f"{name}.Real.R@10"] for q in student}, student, gt)
            for name in teachers
        },
    }
    first = teachers[0]
    for name in teachers[1:]:
        for view in ("f0", "Real"):
            summary["contrasts"][f"{name}.{view}_minus_{first}.{view}"] = bootstrap_contrast(
                {q: per_query[q][f"{name}.{view}.R@10"] for q in student},
                {q: per_query[q][f"{first}.{view}.R@10"] for q in student}, gt)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, help="physical GPU index (default: the protocol's side GPU, else its main GPU)")
    parser.add_argument("--student", choices=("KD", "SUP"), default="KD",
                        help="dev-selected Native C2 arm whose retrieval mines the lists (protocol primary: KD)")
    parser.add_argument("--init-checkpoint", type=Path, help="default: <run>/seed<seed>/TB_CQET/checkpoints/end.pt")
    parser.add_argument("--records", type=Path, help="train on these T_B records instead of mining (control arm)")
    parser.add_argument("--seed", type=int, help="default: first protocol seed")
    parser.add_argument("--epochs", type=int, help="default: protocol teacher.TB.epochs")
    parser.add_argument("--lr", type=float, help="default: protocol teacher.TB.lr")
    parser.add_argument("--limit", type=int, default=0, help="smoke: only the first N train and dev queries")
    args = parser.parse_args(argv)

    run = args.run_root.resolve()
    out = args.out_dir.resolve()
    protocol = read_json(run / "protocol.json")
    hardware = protocol["hardware"]
    physical = args.gpu if args.gpu is not None else hardware.get("side_physical_index", hardware["physical_index"])
    uuid = gpu_uuid(physical)
    # Before the first CUDA call: the process then sees exactly this device as cuda:0.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    _gpu_guard(uuid)

    seed = args.seed if args.seed is not None else int(protocol["seeds"][0])
    seed_dir = run / f"seed{seed}"
    tb = protocol["teacher"]["TB"]
    epochs = args.epochs if args.epochs is not None else int(tb["epochs"])
    lr = args.lr if args.lr is not None else float(tb["lr"])
    init_checkpoint = (args.init_checkpoint or seed_dir / "TB_CQET" / "checkpoints" / "end.pt").resolve()
    selection = read_json(seed_dir / "selections" / "NATIVE_C2_COMMON.json")
    student_checkpoint = Path(selection[f"{args.student}_checkpoint"])
    out.mkdir(parents=True, exist_ok=True)

    rt = load_runtime(run / "protocol.json", run)
    rt.bank.attach_device("cuda:0")
    student = _load_native(student_checkpoint, rt).eval()
    if model_state_sha(student) != selection[f"{args.student}_state_sha256"]:
        raise RuntimeError(f"{student_checkpoint} is not the selected {args.student} Student")

    records_path = out / "records.jsonl.gz"
    if args.records is not None:
        records_path = args.records.resolve()
        records = list(iter_jsonl(records_path))
        if args.limit:
            records = records[: args.limit]
    elif records_path.exists():
        records = list(iter_jsonl(records_path))
    else:
        train_ids = rt.labels.query_ids[: args.limit or None]
        print(f"[mine] {len(train_ids)} train queries with the {args.student} Student", flush=True)
        started = time.time()
        records, mined = mine_records(rt.z_store, rt.row_store, rt.labels, student, train_ids, seed, out / "indices")
        mined["wall_seconds"] = time.time() - started
        write_json(out / "records_summary.json", mined)
        save_training_records(records_path, records)
        print(json.dumps(mined, indent=2), flush=True)

    identity = {
        "run_root": str(run),
        "protocol_sha256": sha256_file(run / "protocol.json"),
        "seed": seed,
        "student_arm": args.student,
        "student": {"path": str(student_checkpoint), "sha256": sha256_file(student_checkpoint),
                    "state_sha256": model_state_sha(student), "selected_fraction": selection["selected_fraction"]},
        "init_checkpoint": _checkpoint_identity(init_checkpoint),
        "records": {"path": str(records_path), "sha256": sha256_file(records_path), "count": len(records),
                    "source": "student_pool" if args.records is None else "given"},
        "config": {**tb, "epochs": epochs, "lr": lr, "mode": "cqet", "limit": args.limit},
        "gpu_uuid": uuid,
        "source_identity_sha256": source_identity(rt.paths),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(out / "IDENTITY.json", identity)

    end_checkpoint = train_continuation(
        rt, records, init_checkpoint, out, seed=seed, epochs=epochs, lr=lr,
        metadata={"init_checkpoint": identity["init_checkpoint"], "records_sha256": identity["records"]["sha256"]},
    )
    del records

    summary_path = out / "eval" / "dev" / "SUMMARY.json"
    if summary_path.exists():
        print(f"[evaluate] {summary_path} exists; done", flush=True)
        return
    teachers = {
        "init": _load_teacher(init_checkpoint),
        "half": _load_teacher(end_checkpoint.with_name("half.pt")),
        "end": _load_teacher(end_checkpoint),
    }
    gt = load_split_gt(rt.paths, "dev", rt.labels.canonical_map)
    dev_ids = utf8_sorted(gt)[: args.limit or None]
    gt = {q: gt[q] for q in dev_ids}
    generator = f"native_{args.student.lower()}"
    summary = {"teacher_points": {name: _checkpoint_identity(path) for name, path in (
        ("init", init_checkpoint), ("half", end_checkpoint.with_name("half.pt")), ("end", end_checkpoint))},
        "query_count": len(dev_ids), "generators": {}}
    for name, model in ((generator, student), ("raw", None)):
        directory = out / "eval" / "dev" / name
        pools, source = dev_pools(rt, name, model, dev_ids, seed_dir / "eval" / "dev", directory, seed)
        print(f"[evaluate] {name} dev pools from {source}", flush=True)
        matrix = evaluate_teacher_matrix(
            teachers, rt.bank, pools, rt.labels, seed=seed, generator=name, split="dev",
            output_dir=directory, split_gt=gt, teacher_modes={point: "cqet" for point in teachers},
        )
        metrics = evaluate_matrix(pools, matrix, gt, directory)
        summary["generators"][name] = {"pools": source, **acceptance_summary(metrics, gt, pools, TEACHER_POINTS)}
        del pools, matrix
        torch.cuda.empty_cache()
    write_json(summary_path, summary)
    table = {name: {point: {view: round(block["teacher"][point][view]["overall"], 4) for view in ("f0", "Real")}
                    for point in TEACHER_POINTS} | {"student_Direct_ANN_R10": round(block["candidate"]["overall"]["Direct_ANN_R10"], 4),
                                                   "C150_coverage": round(block["candidate"]["overall"]["C150_target_coverage"], 4)}
             for name, block in summary["generators"].items()}
    print(json.dumps(table, indent=2), flush=True)


if __name__ == "__main__":
    main()
