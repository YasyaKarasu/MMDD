#!/usr/bin/env python3
"""Materialize the rank-only Equal-RRF control for legacy and R25 rankings."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def read(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ranks(row):
    if "scorer_ids" in row:
        d = row["scorer_ids"]["DIRECT_ANN"]
        e = row["scorer_ids"].get("QT_OVER_U", row["scorer_ids"].get("QT_OVER_M", {}))
        return list(map(str, d["ranking"])), list(map(str, e.get("ranking", [])))
    return list(map(str, row.get("direct_ann", row.get("U", [])))), list(map(str, row.get("evidence_ann", [])))


def run(args):
    sources = [("B13-FULL/seed13", args.r25_b13_13), ("B13-FULL/seed29", args.r25_b13_29), ("SPLIT-QTKD/seed13", args.r25_split_13), ("SPLIT-QTKD/seed29", args.r25_split_29), ("Qwen-Raw/seed13", args.raw), ("B13/seed13", args.b13), ("N-U/seed13", args.nu13), ("N-U/seed29", args.nu29)]
    # Include the remaining frozen R25 arms without duplicating the explicit
    # reference/control paths above.
    output_root = Path(args.output_root).resolve()
    seen = {name for name, _ in sources}
    ranking_root = output_root.parent / "rankings"
    for path in sorted(ranking_root.glob("*/seed*/query_rankings.jsonl.gz")):
        name = f"{path.parent.parent.name}/{path.parent.name}"
        if name not in seen:
            sources.append((name, str(path)))
            seen.add(name)
    all_results = []
    for name, path in sources:
        rows = read(Path(path).resolve())
        output_rows = []
        for row in rows:
            direct, evidence = ranks(row)
            rd = {target: i + 1 for i, target in enumerate(direct)}
            re = {target: i + 1 for i, target in enumerate(evidence)}
            candidates = list(dict.fromkeys(direct + evidence))
            ranking = sorted(candidates, key=lambda target: (-(1.0 / (60 + rd[target]) if target in rd else 0.0) - (1.0 / (60 + re[target]) if target in re else 0.0), target))
            output_rows.append({"query_id": str(row["query_id"]), "query_kind": row.get("query_kind"), "positive_target_ids": list(map(str, row.get("positive_target_ids", []))), "fusion_method": "Equal", "ranking": ranking, "source_scorers": ["DIRECT_ANN", "QT_OVER_U"], "alpha": 1.0, "online_gt_inputs": False})
        destination = output_root / name
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / "Equal.jsonl.gz"
        with gzip.open(output, "wt", encoding="utf-8") as handle:
            for row in output_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        metrics = {"queries": len(output_rows), "R@10": sum(bool(set(row["positive_target_ids"]) & set(row["ranking"][:10])) for row in output_rows) / len(output_rows), "R@20": sum(bool(set(row["positive_target_ids"]) & set(row["ranking"][:20])) for row in output_rows) / len(output_rows), "R@50": sum(bool(set(row["positive_target_ids"]) & set(row["ranking"][:50])) for row in output_rows) / len(output_rows)}
        (destination / "Equal.metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        all_results.append({"source": name, "output": str(output), "metrics": metrics})
    print(json.dumps({"status": "complete", "sources": all_results}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("r25-b13-13", "r25-b13-29", "r25-split-13", "r25-split-29", "raw", "b13", "nu13", "nu29"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--output-root", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
