#!/usr/bin/env python
"""Train a Teacher chain from fresh init (T_A -> T_B_CQET) on a run's Raw training lists and
evaluate every checkpoint on the run's Raw dev pools, with no Student in the loop.

This is the architecture A/B: ``--path-mode pairwise_residual`` (f0 + zero-initialised
path_head(q, e) + path_head(e, t)), ``--path-loss-scope bagged`` and
``--witness-target-weight`` against the run's own TA -> TB_CQET chain, on the same
``TA.jsonl.gz`` / ``TB_SHARED.jsonl.gz`` lists, seed and dev pools. T_A already trains on one
witness-conditioned target list per witness; those lists are merged into the T_B records by
query so T_B can keep that supervision (the stored T_B lists predate it).

Steps, each skipped when its outputs exist (a killed job restarts at the unfinished step):

    TA        train_ta from fresh init            -> TA/{init,epoch<n>}.pt, TA/train.jsonl
    TB        train_tb (cqet) from TA/epoch<E>    -> TB_CQET/{init,half,end}.pt, TB_CQET/train.jsonl
    evaluate  TA_epoch*, TB_half, TB_end on the Raw dev pools: f0 / Real / Swap R@10 per segment,
              strict-pair funnels, paired contrasts Real - f0 and Real - Swap, and TB_end against
              the run's TB_CQET end on the same pools  -> eval/dev/raw/*, eval/dev/SUMMARY.json

Only ``--out-dir`` is written; the run root is read.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from mmdd_stage1.artifacts import load_pool_bundle  # noqa: E402
from mmdd_stage1.data import iter_jsonl, load_split_gt, read_json, sha256_file, utf8_sorted, write_json  # noqa: E402
from mmdd_stage1.evaluate import evaluate_teacher_matrix  # noqa: E402
from mmdd_stage1.metrics import bootstrap_contrast, evaluate_matrix, export_funnels  # noqa: E402
from mmdd_stage1.models import model_state_sha, state_sha  # noqa: E402
from mmdd_stage1.pipeline import _gpu_guard, _load_teacher, _set_seed, _teacher, load_runtime  # noqa: E402
from mmdd_stage1.provenance import source_identity  # noqa: E402
from mmdd_stage1.train import train_ta, train_tb  # noqa: E402

SEGMENTS = ("overall", "implicit", "explicit")
VIEWS = ("f0", "Real", "Swap")


def gpu_uuid(physical_index: int) -> str:
    output = subprocess.check_output(
        ["nvidia-smi", "-i", str(physical_index), "--query-gpu=uuid", "--format=csv,noheader"], text=True,
    )
    return output.strip()


def merge_witness_lists(ta_records: Sequence[dict], tb_records: Sequence[dict]) -> list[dict]:
    """T_B records with the ``qet_lists`` of the same query's T_A record (both come from the
    same Raw pool, so the lists are the ones T_A trained on)."""
    by_query = {row["query_id"]: row for row in ta_records}
    if set(by_query) != {row["query_id"] for row in tb_records}:
        raise ValueError("T_A and T_B records cover different queries")
    return [{**row, "qet_lists": list(by_query[row["query_id"]].get("qet_lists", []))} for row in tb_records]


def _checkpoint_identity(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {"path": str(path), "sha256": sha256_file(path), "state_sha256": state_sha(payload["model"])}


def _fresh_stage_dir(out: Path, stage: str, done: Path) -> Path | None:
    """The stage directory to train into, or None when ``done`` already exists. A partial
    directory from an interrupted attempt is set aside (the Teacher loop has no mid-stage resume)."""
    directory = out / stage
    if done.exists():
        return None
    for stale in (directory, out / f"{stage}.train.jsonl"):
        if stale.exists():
            stale.rename(stale.with_name(f"{stale.name}.failed.{int(time.time())}"))
    return directory


def _timed(out: Path, stage: str, run) -> None:
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    run()
    torch.cuda.synchronize()
    write_json(out / f"{stage}.TIMING.json", {
        "wall_seconds": time.time() - started,
        "optimizer_steps": sum(1 for _ in iter_jsonl(out / f"{stage}.train.jsonl")),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    })


def train_chain(rt, ta_records: list[dict], tb_records: list[dict], out: Path, *, seed: int, path_mode: str,
                ta_epochs: int, tb_epochs: int, support_weight: float | None, path_loss_scope: str,
                witness_target_weight: float, metadata: dict) -> dict[str, Path]:
    """TA from fresh init then TB_CQET from its last epoch; returns the checkpoints to evaluate."""
    ta, tb = rt.protocol["teacher"]["TA"], rt.protocol["teacher"]["TB"]
    ta_end = out / "TA" / f"epoch{ta_epochs}.pt"
    directory = _fresh_stage_dir(out, "TA", ta_end)
    if directory is not None:
        _set_seed(seed, "TA")
        model = _teacher(path_mode)
        _timed(out, "TA", lambda: train_ta(
            model, rt.bank, ta_records, rt.labels, save_dir=directory, epochs=ta_epochs, lr=float(ta["lr"]),
            weight_decay=float(ta["wd"]), logical_batch=int(ta["logical_batch_queries"]),
            support_weight=float(ta["support_weight"] if support_weight is None else support_weight),
            seed=seed, metadata=metadata, log_path=out / "TA.train.jsonl",
        ))
        del model
    tb_end = out / "TB_CQET" / "end.pt"
    directory = _fresh_stage_dir(out, "TB_CQET", tb_end)
    if directory is not None:
        _set_seed(seed, "TB_SHARED")
        model = _load_teacher(ta_end)
        _timed(out, "TB_CQET", lambda: train_tb(
            model, rt.bank, tb_records, mode="cqet", save_dir=directory, epochs=tb_epochs, lr=float(tb["lr"]),
            weight_decay=float(tb["wd"]), logical_batch=int(tb["logical_batch_queries"]),
            direct_weight=float(tb["path_direct_weight"]), aggregate_weight=float(tb["path_aggregate_weight"]),
            support_weight=float(tb["support_weight"] if support_weight is None else support_weight),
            path_loss_scope=path_loss_scope, witness_target_weight=witness_target_weight,
            seed=seed, metadata=metadata, log_path=out / "TB_CQET.train.jsonl",
        ))
        del model
    points = {f"TA_epoch{n}": out / "TA" / f"epoch{n}.pt" for n in range(1, ta_epochs + 1)}
    return {**points, "TB_half": out / "TB_CQET" / "half.pt", "TB_end": tb_end}


def chain_summary(metrics: dict, gt: Mapping[str, dict], points: Sequence[str],
                  reference: Mapping[str, Mapping[str, float]] | None, strict: Mapping[str, dict]) -> dict:
    """Per point: R@10 of every view and segment, strict-pair funnel counts, and paired contrasts
    Real - f0 and Real - Swap; ``reference`` (per-query R@10 of the run's own TB_CQET end on the
    same pools, keyed ``TB_CQET.<view>.R@10``) adds TB_end - reference contrasts."""
    per_query = metrics["per_query"]
    queries = utf8_sorted(per_query)
    column = lambda name, view: {q: per_query[q][f"{name}.{view}.R@10"] for q in queries}  # noqa: E731
    summary = {
        "teacher": {name: {view: {segment: metrics["teacher"][name][view][segment]["R@10"] for segment in SEGMENTS}
                           for view in VIEWS} for name in points},
        "strict_pairs": {name: strict[name] for name in points if name in strict},
        "contrasts": {},
    }
    for name in points:
        summary["contrasts"][f"{name}.Real_minus_f0"] = bootstrap_contrast(column(name, "Real"), column(name, "f0"), gt)
        summary["contrasts"][f"{name}.Real_minus_Swap"] = bootstrap_contrast(column(name, "Real"), column(name, "Swap"), gt)
    if reference is not None and set(reference) >= set(queries):
        for view in ("f0", "Real"):
            summary["contrasts"][f"TB_end.{view}_minus_reference.{view}"] = bootstrap_contrast(
                column("TB_end", view), {q: reference[q][f"TB_CQET.{view}.R@10"] for q in queries}, gt)
    return summary


def _strict_counts(path: Path) -> dict:
    stages = read_json(path)["stages"]
    return {stage: stages[stage]["micro_pairs"] for stage in ("in_C150", "teacher_top10", "teacher_top50")}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, help="physical GPU index (default: the protocol's main GPU)")
    parser.add_argument("--seed", type=int, help="default: first protocol seed")
    parser.add_argument("--path-mode", choices=("triplet", "pairwise_residual"), default="triplet")
    parser.add_argument("--path-loss-scope", choices=("all", "bagged"), default="all")
    parser.add_argument("--witness-target-weight", type=float, default=0.0,
                        help="T_B weight of the witness-conditioned target lists merged from T_A (0 = off)")
    parser.add_argument("--support-weight", type=float, help="T_A and T_B support weight (default: protocol)")
    parser.add_argument("--ta-epochs", type=int, help="default: protocol teacher.TA.epochs")
    parser.add_argument("--tb-epochs", type=int, help="default: protocol teacher.TB.epochs")
    parser.add_argument("--records-dir", type=Path, help="default: <run>/seed<seed>/training_records")
    parser.add_argument("--dev-pools", type=Path, help="default: <run>/seed<seed>/eval/dev/raw")
    parser.add_argument("--limit", type=int, default=0, help="smoke: only the first N train and dev queries")
    args = parser.parse_args(argv)

    run, out = args.run_root.resolve(), args.out_dir.resolve()
    protocol = read_json(run / "protocol.json")
    physical = args.gpu if args.gpu is not None else protocol["hardware"]["physical_index"]
    uuid = gpu_uuid(physical)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    _gpu_guard(uuid)

    seed = args.seed if args.seed is not None else int(protocol["seeds"][0])
    seed_dir = run / f"seed{seed}"
    records_dir = (args.records_dir or seed_dir / "training_records").resolve()
    dev_pools_dir = (args.dev_pools or seed_dir / "eval" / "dev" / "raw").resolve()
    ta_epochs = args.ta_epochs if args.ta_epochs is not None else int(protocol["teacher"]["TA"]["epochs"])
    tb_epochs = args.tb_epochs if args.tb_epochs is not None else int(protocol["teacher"]["TB"]["epochs"])
    out.mkdir(parents=True, exist_ok=True)

    rt = load_runtime(run / "protocol.json", run)
    rt.bank.attach_device("cuda:0")
    ta_records = list(iter_jsonl(records_dir / "TA.jsonl.gz"))[: args.limit or None]
    tb_records = list(iter_jsonl(records_dir / "TB_SHARED.jsonl.gz"))[: args.limit or None]
    tb_records = merge_witness_lists(ta_records, tb_records)
    identity = {
        "run_root": str(run), "protocol_sha256": sha256_file(run / "protocol.json"), "seed": seed,
        "records": {name: {"path": str(records_dir / f"{name}.jsonl.gz"), "sha256": sha256_file(records_dir / f"{name}.jsonl.gz")}
                    for name in ("TA", "TB_SHARED")},
        "witness_lists": "T_A qet_lists merged into T_B records by query",
        "config": {"path_mode": args.path_mode, "path_loss_scope": args.path_loss_scope,
                   "witness_target_weight": args.witness_target_weight, "support_weight": args.support_weight,
                   "ta_epochs": ta_epochs, "tb_epochs": tb_epochs, "limit": args.limit,
                   "TA": protocol["teacher"]["TA"], "TB": protocol["teacher"]["TB"]},
        "dev_pools": str(dev_pools_dir), "gpu_uuid": uuid,
        "source_identity_sha256": source_identity(rt.paths),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(out / "IDENTITY.json", identity)

    points = train_chain(
        rt, ta_records, tb_records, out, seed=seed, path_mode=args.path_mode, ta_epochs=ta_epochs,
        tb_epochs=tb_epochs, support_weight=args.support_weight, path_loss_scope=args.path_loss_scope,
        witness_target_weight=args.witness_target_weight,
        metadata={"records": identity["records"], "config": identity["config"]},
    )
    del ta_records, tb_records

    summary_path = out / "eval" / "dev" / "SUMMARY.json"
    if summary_path.exists():
        print(f"[evaluate] {summary_path} exists; done", flush=True)
        return
    gt = load_split_gt(rt.paths, "dev", rt.labels.canonical_map)
    dev_ids = utf8_sorted(gt)[: args.limit or None]
    gt = {q: gt[q] for q in dev_ids}
    pools = load_pool_bundle(dev_pools_dir)
    pools = {q: pools[q] for q in dev_ids}
    directory = out / "eval" / "dev" / "raw"
    teachers = {name: _load_teacher(path) for name, path in points.items()}
    matrix = evaluate_teacher_matrix(
        teachers, rt.bank, pools, rt.labels, seed=seed, generator="raw", split="dev", output_dir=directory,
        split_gt=gt, teacher_modes={name: "cqet" for name in teachers},
    )
    metrics = evaluate_matrix(pools, matrix, gt, directory)
    strict = {}
    for name in points:
        export_funnels(pools, matrix[name]["Real"], gt, rt.labels, directory / "funnels" / name,
                       seed=seed, generator=f"raw.{name}")
        strict[name] = _strict_counts(directory / "funnels" / name / "strict_EO_SUMMARY.json")
    reference_path = seed_dir / "eval" / "dev" / "SUMMARY.json"
    reference = read_json(reference_path)["raw"]["per_query"] if reference_path.exists() else None
    summary = {
        "points": {name: _checkpoint_identity(path) for name, path in points.items()},
        "query_count": len(dev_ids), "dev_pools": str(dev_pools_dir),
        "reference": None if reference is None else {"path": str(reference_path), "teacher": "TB_CQET end of the run"},
        **chain_summary(metrics, gt, list(points), reference, strict),
    }
    write_json(summary_path, summary)
    table = {name: {view: round(summary["teacher"][name][view]["overall"], 4) for view in VIEWS}
             | {"strict_top10": strict[name]["teacher_top10"]} for name in points}
    print(json.dumps(table, indent=2), flush=True)
    for key in ("TB_end.Real_minus_f0", "TB_end.Real_minus_Swap", "TB_end.Real_minus_reference.Real", "TB_end.f0_minus_reference.f0"):
        if key in summary["contrasts"]:
            c = summary["contrasts"][key]
            print(f"{key}: {c['mean_delta_pp']:+.2f} pp, 95% CI [{c['ci_95'][0]:+.2f}, {c['ci_95'][1]:+.2f}]", flush=True)


if __name__ == "__main__":
    main()
