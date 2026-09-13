"""Build immutable fresh-lineage augmented candidate manifests.

The miner outputs are query-level hard candidates.  This utility applies them
to every logical TT list for that query while leaving non-TT rows and list
identity untouched, then records provenance and hashes in a sidecar manifest.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from run_stage1_r22 import out as r22_out, read_rows, write_rows


def build(root: Path, source: str, output_name: str, seed: int = 13) -> Path:
    base = r22_out(root) / "manifests" / "full_natural.jsonl"
    hard_path = r22_out(root) / "fresh_lineage" / source / "seed13" / "hard_negatives.jsonl.gz"
    if not hard_path.exists():
        raise FileNotFoundError(hard_path)
    # The two seed miners should produce the same candidate membership.  Use
    # seed13 as the deterministic pool and report any seed29 differences.
    hard_by_seed: dict[int, dict[str, list[str]]] = {}
    for seed in (13, 29):
        path = r22_out(root) / "fresh_lineage" / source / f"seed{seed}" / "hard_negatives.jsonl.gz"
        rows = read_rows(path)
        hard_by_seed[seed] = {str(r["query_id"]): [str(x) for x in r["hard_candidate_ids"]] for r in rows}
    if seed not in hard_by_seed:
        raise ValueError("seed must be 13 or 29")
    hard = hard_by_seed[seed]
    hard_scores = {
        str(r["query_id"]): list(r.get("hard_scores", []))
        for r in read_rows(r22_out(root) / "fresh_lineage" / source / f"seed{seed}" / "hard_negatives.jsonl.gz")
    }
    differing = [q for q in set(hard_by_seed[13]) | set(hard_by_seed[29])
                 if hard_by_seed[13].get(q, []) != hard_by_seed[29].get(q, [])]
    rows_out = []
    changed = 0
    for row in read_rows(base):
        row = dict(row)
        if row.get("source_type") == "table" and row.get("destination_type") == "table":
            q = str(row["query_id"])
            extra = hard.get(q, [])
            old = [str(x) for x in row.get("candidate_ids", [])]
            new = list(dict.fromkeys([*old, *extra]))
            if len(new) != len(old):
                changed += 1
                labels = list(row.get("confirmed_labels") or [None] * len(old))
                labels.extend([None] * (len(new) - len(labels)))
                row["candidate_ids"] = new
                row["confirmed_labels"] = labels
                row["candidate_provenance"] = {
                    "base": "full_natural",
                    "hard_source": source,
                    "hard_candidate_ids": extra,
                    "hard_scores": hard_scores.get(q, []),
                }
        rows_out.append(row)
    output = r22_out(root) / "manifests" / output_name
    write_rows(output, rows_out)
    write_json(output.with_suffix(output.suffix + ".manifest.json"), {
        "format_version": 1, "status": "pass", "source": source, "seed": seed,
        "base_manifest_sha256": checkpoint_fingerprint(base),
        "hard_seed13_sha256": checkpoint_fingerprint(r22_out(root) / "fresh_lineage" / source / "seed13" / "hard_negatives.jsonl.gz"),
        "hard_seed29_sha256": checkpoint_fingerprint(r22_out(root) / "fresh_lineage" / source / "seed29" / "hard_negatives.jsonl.gz"),
        "seed_membership_differences": len(differing),
        "changed_logical_lists": changed, "total_rows": len(rows_out),
        "output_sha256": checkpoint_fingerprint(output),
    })
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--source", required=True, help="Miner directory under fresh_lineage")
    ap.add_argument("--output", required=True, help="Output JSONL filename under manifests")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    print(json.dumps({"output": str(build(args.root, args.source, args.output, args.seed))}))


if __name__ == "__main__":
    main()
