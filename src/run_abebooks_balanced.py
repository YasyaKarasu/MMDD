"""Compare Student update budgets on one fresh, balanced AbeBooks dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_abebooks_fresh import ROOT, finish, launch, write_json

SCHEDULES = {"one_epoch_b64": (1, 64), "ten_epochs_b64": (10, 64), "ten_epochs_b16": (10, 16)}


def prepare_schedules(root: Path) -> None:
    if not (root / "features/content/index.json").exists():
        raise ValueError("Finish fresh feature encoding before training")
    plan = {"dataset": str(root / "dataset_view"), "seed": 13,
        "hypothesis": "Additional shuffled epochs address undertraining; smaller logical batches increase updates per epoch",
        "schedules": {name: {"epochs": epochs, "batch": batch} for name, (epochs, batch) in SCHEDULES.items()},
        "primary_schedule": "ten_epochs_b16", "teacher_epochs": {"TA": 2, "TB_CQET": 1},
        "teacher_training": "fresh deterministic training per schedule, same seed and records",
        "features": "newly encoded for this experiment; shared unchanged across schedules",
        "student_selection": "dev each epoch; existing candidate-coverage/R10 ordering; KD follows SUP epoch",
        "test_policy": "all schedules and checkpoints frozen before any test evaluation",
        "labels": "repeated epochs reuse labels; they do not create additional annotations"}
    write_json(root / "SUITE_PROTOCOL.json", plan)
    for name, (epochs, batch) in SCHEDULES.items():
        run = root / name
        run.mkdir(exist_ok=False)
        (run / "isolated_cwd").mkdir()
        for shared in ("dataset_view", "features", "encoder", "data"):
            (run / shared).symlink_to(root / shared, target_is_directory=True)
        protocol = json.loads((root / "protocol.json").read_text())
        protocol["paths"] = {k: v.replace(str(root), str(run)) for k, v in protocol["paths"].items()}
        for stage, key in (("C1", "logical_batch_edge_lists"), ("C2", "logical_batch_queries")):
            protocol["student"][stage].update(epochs=epochs)
            protocol["student"][stage][key] = batch
            if epochs > 1:
                protocol["student"][stage]["checkpoints"] = [i / epochs for i in range(epochs + 1)]
        write_json(run / "protocol.json", protocol)


def run_phase(root: Path, phase: str, gpus: tuple[int, ...] = (0, 1)) -> None:
    schedules = list(SCHEDULES.items())
    for offset in range(0, len(schedules), len(gpus)):
        jobs = []
        for gpu, (name, (epochs, batch)) in zip(gpus, schedules[offset:offset + len(gpus)]):
            run = root / name
            arguments = [str(ROOT / "src/run_abebooks_data_ablation.py"), phase, "--run-root", str(run)]
            if phase == "train":
                arguments += ["--student-epochs", str(epochs), "--student-batch", str(batch)]
            jobs.append((name, launch(run, phase, arguments, gpu)))
        for name, process in jobs:
            finish(f"{phase}/{name}", process)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "evaluate"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", choices=(0, 1), default=[0, 1],
                        help="GPU indices available to this experiment; one index runs schedules sequentially")
    args = parser.parse_args()
    root = args.run_root.resolve()
    if args.command == "run":
        prepare_schedules(root)
        run_phase(root, "train", tuple(args.gpus))
        freezes = {name: json.loads((root / name / "SELECTION_FREEZE.json").read_text()) for name in SCHEDULES}
        write_json(root / "ALL_SELECTIONS_FROZEN.json", {"status": "FROZEN_BEFORE_TEST", "schedules": freezes})
    run_phase(root, "evaluate", tuple(args.gpus))
    write_json(root / "SUITE_COMPLETE.json", {"status": "COMPLETE", "schedules": list(SCHEDULES)})


if __name__ == "__main__":
    main()
