"""Measure the strongest Stage-1 checkpoint on the pre-migration per-split data lakes.

The 2026-09-16 query-only migration collapsed the train/dev/test data lakes into
one shared 22,886-table lake, and every Stage-1 number reported since then
retrieves from that shared lake. The four files that migration rewrote were backed
up next to the dataset, so the pre-migration retrieval scopes can be rebuilt
exactly rather than approximated.

This probe replays the frozen R26 own-pool protocol -- B13 own ANN -> Equal C100
-> frozen T0 -- on those legacy scopes, and on the shared lake for contrast. Only
the retrieval scope changes: the checkpoint, the query population of each split,
the positives, the feature cache and the T0 teacher are the ones R26 already
froze. A legacy scope is one split's slice of the old data lake plus the bridge
assets reachable from that split's source tables.

Subcommands, in order:

    build    materialise the two legacy corpora and the two query populations
    run      own-pool retrieval for every (population, corpus) pair
    teacher  frozen T0 rerank over each run's own candidate pools
    report   side-by-side table against the R26 shared-lake baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import evaluate_stage1_r26 as retrieval
import evaluate_stage1_r26_teacher as teacher
from run_stage1_r21 import paths as historical_paths, write_rows


ROOT = Path(__file__).resolve().parent.parent
DATASET = (
    ROOT
    / "output_mm_joinability_entitables_20000_retry100_rounds5_qwen35_final_survivor_context_gaussian_v9"
)
# The migration's own backup of the four files it rewrote. `splits.json` here
# still carries the per-split `data_lake_table_ids` the shared format dropped.
LEGACY = DATASET.parent / (DATASET.name + ".pre_query_only")
R10 = ROOT / "work/stage1_optimization_r10_20260907"
R26 = ROOT / "work/stage1_optimization_r26_20260914"
OUT = ROOT / "work/stage1_split_lake_probe_20260917"

CORPUS = R10 / "stage1_data/stage1_corpus.jsonl"
TARGET_LISTS = R10 / "stage1_data/target_lists.jsonl"
B13 = R26 / "recovered/B13/step_000178.pt"
# R26 already froze these two; reusing them is what makes the contrast meaningful.
R26_PROTOCOL = R26 / "PROTOCOL.json"
R26_B13_METRICS = R26 / "rankings/B13/metrics.json"
R26_TEACHER_METRICS = R26 / "teacher/B13/metrics.json"
R26_TEACHER_CACHE = R26 / "teacher/T0_pairs.sqlite"

SPLITS = ("train", "dev", "test")
# run name -> (query population, legacy lake name or None for the shared lake)
RUNS: dict[str, tuple[str, str | None]] = {
    "dev_shared": ("dev", None),
    "dev_devlake": ("dev", "dev"),
    "test_shared": ("test", None),
    "test_testlake": ("test", "test"),
}


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_rows_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    temporary.replace(path)
    return count


def legacy_scope(split: str) -> tuple[set[str], set[str]]:
    """Return the split's legacy lake table IDs and its source table IDs."""
    payload = json.loads((LEGACY / "splits.json").read_text(encoding="utf-8"))
    if "data_lake_table_ids" not in payload.get(split, {}):
        raise ValueError(f"{LEGACY}/splits.json is not the pre-migration layout: {split}")
    return set(payload[split]["data_lake_table_ids"]), set(payload[split]["source_table_ids"])


def legacy_assets() -> dict[str, set[str]]:
    """Bridge assets reachable from each split's source tables, in one pass.

    The legacy format never tagged assets directly: an asset belonged to a split
    through the source tables that linked it, which is also how the pre-migration
    construction bucketed evidence. Entity-level assets are therefore reachable
    from more than one split, and the three legacy asset pools overlap.

    This is the scope-consistent reading of "the split's multimodal cues": the
    same rule the table side uses, everything belonging to the split. A narrower
    alternative -- only the assets that appear as evidence in
    ``evidence_recoveries`` -- is a strict subset (dev 1,656; test 1,654; train
    10,524 of 225,720) and is reported by ``evidence_asset_counts`` for contrast,
    but it is not the retrieval scope: pre-migration, that set only governed
    training-time evidence negative buckets, not the index that was searched.
    """
    owners: dict[str, list[str]] = {}
    for split in SPLITS:
        _, source_ids = legacy_scope(split)
        for source_id in source_ids:
            owners.setdefault(source_id, []).append(split)
    assets: dict[str, set[str]] = {split: set() for split in SPLITS}
    for path in sorted((DATASET / "table_asset_links").glob("part-*.jsonl")):
        for record in _rows(path):
            for split in owners.get(str(record.get("source_table_id")), ()):
                assets[split].update(record.get("asset_ids") or ())
    return assets


def evidence_asset_counts() -> dict[str, int]:
    """Distinct evidence assets per split, for the receipt's sensitivity note."""
    counts = {split: set() for split in SPLITS}
    for path in sorted((DATASET / "evidence_recoveries").glob("part-*.jsonl")):
        for record in _rows(path):
            asset_id = (record.get("evidence") or {}).get("asset_id")
            if asset_id and record.get("split") in counts:
                counts[record["split"]].add(str(asset_id))
    return {split: len(values) for split, values in counts.items()}


def query_kinds() -> dict[str, str]:
    """Derive each query's kind from its qrel reasons.

    Same rule as ``mmdd_stage1.construction``: a query whose every qrel is
    ``model_recoverable_join_column`` is implicit, one with none of them is
    explicit, and a mixture is mixed. Verified to reproduce all 1,198 frozen dev
    labels before this probe was run.
    """
    reasons: dict[str, set[str]] = {}
    for record in _rows(DATASET / "qrels.jsonl"):
        reasons.setdefault(str(record["query_table_id"]), set()).add(str(record["reason"]))
    marker = "model_recoverable_join_column"
    return {
        query_id: (
            "implicit" if values == {marker} else "explicit" if marker not in values else "mixed"
        )
        for query_id, values in reasons.items()
    }


def population(split: str) -> list[dict[str, Any]]:
    """The split's query population, in the frozen ``dev_queries.jsonl`` shape."""
    kinds = query_kinds()
    sources = {
        str(record["table_id"]): str(record["source_table_id"])
        for record in _rows(DATASET / "query_tables/part-00000.jsonl")
        if record.get("split") == split
    }
    rows = []
    for record in _rows(TARGET_LISTS):
        if record.get("split") != split:
            continue
        query_id = str(record["query_id"])
        positives = [str(value) for value in record["positive_target_ids"]]
        if not positives:
            raise ValueError(f"Frozen population includes a query without qrels: {query_id}")
        rows.append(
            {
                "query_id": query_id,
                "query_kind": kinds[query_id],
                "positive_target_ids": positives,
                "source_table_id": sources[query_id],
            }
        )
    rows.sort(key=lambda row: row["query_id"])
    return rows


def build() -> dict[str, Any]:
    corpora, populations, receipt = OUT / "corpora", OUT / "populations", {}
    pool = legacy_assets()
    evidence_counts = evidence_asset_counts()
    allowed: dict[str, set[str]] = {}
    for split in SPLITS:
        tables, sources = legacy_scope(split)
        assets = pool[split]
        allowed[split] = tables | assets
        path = corpora / f"stage1_corpus.{split}_lake.jsonl"
        written = _write_rows_atomic(
            path, ({"object_id": str(record["object_id"])} for record in _rows(CORPUS) if str(record["object_id"]) in allowed[split])
        )
        receipt[split] = {
            "lake_tables": len(tables),
            "source_tables": len(sources),
            "assets": len(assets),
            "assets_via_evidence_recoveries": evidence_counts[split],
            "corpus_objects": written,
        }
    # The shared scope is the untouched frozen corpus; refer to it by path so the
    # index and receipt hashes still line up with R26's own evaluation.
    for split in SPLITS:
        rows = population(split)
        _write_rows_atomic(populations / f"{split}_queries.jsonl", rows)
        receipt[split]["queries"] = len(rows)
        receipt[split]["queries_implicit"] = sum(row["query_kind"] == "implicit" for row in rows)
        receipt[split]["queries_explicit"] = sum(row["query_kind"] == "explicit" for row in rows)
        missing = {
            target
            for row in rows
            for target in row["positive_target_ids"]
            if target not in allowed[split]
        }
        if missing:
            raise ValueError(f"{split}: {len(missing)} positives fall outside the legacy lake")
        receipt[split]["positives_outside_legacy_lake"] = 0
    _json(OUT / "BUILD_RECEIPT.json", receipt)
    return receipt


def run_dir(name: str) -> Path:
    return OUT / "runs" / name


def corpus_for(lake: str | None) -> Path:
    return CORPUS if lake is None else OUT / "corpora" / f"stage1_corpus.{lake}_lake.jsonl"


def index_dir(name: str, lake: str | None) -> Path:
    """Shared runs reuse R26's verified index; a legacy scope needs its own.

    The loader checks an index against the corpus it was built from, so a
    restricted corpus cannot reuse the full-lake index.
    """
    return R26 / "indexes/B13" if lake is None else run_dir(name) / "indexes/B13"


def prepare(name: str) -> tuple[Path, str, str | None]:
    """Lay out one run's output directory in the shape the R26 evaluator expects."""
    split, lake = RUNS[name]
    destination = run_dir(name)
    (destination / "common").mkdir(parents=True, exist_ok=True)
    _json(destination / "MODEL_INVENTORY.json", [{"generator_id": "B13", "checkpoint": str(B13)}])
    _json(destination / "PROTOCOL.json", {"stage1": json.loads(R26_PROTOCOL.read_text())["stage1"]})
    write_rows(destination / "common/dev_queries.jsonl", population(split))
    return destination, split, lake


def scoped_paths(name: str, lake: str | None):
    """A ``paths()`` replacement that swaps only the retrieval scope."""

    def resolve(root: Path) -> dict[str, Path]:
        return {
            **historical_paths(root),
            "corpus": corpus_for(lake),
            "b13_index": index_dir(name, lake),
        }

    return resolve


def run_own(names: list[str], device: str) -> dict[str, Any]:
    completed = []
    for name in names:
        destination, split, lake = prepare(name)
        retrieval.OUT = destination
        retrieval.paths = scoped_paths(name, lake)
        result = retrieval.evaluate("B13", device, index_threads=2)
        _json(destination / "node_receipt.json", {"name": name, "split": split, "lake": lake or "shared", "result": result})
        completed.append({"name": name, "status": result.get("status")})
    return {"completed": completed}


def run_teacher(names: list[str], device: str, benchmark_queries: int = 32) -> dict[str, Any]:
    completed = []
    for name in names:
        destination, _, _ = prepare(name)
        # The pair cache is keyed by teacher identity and actual Q/T feature
        # digests, so the frozen R26 cache is shareable across every scope here
        # and turns most of this into cache hits.
        cache_target = destination / "teacher/T0_pairs.sqlite"
        if R26_TEACHER_CACHE.exists() and not cache_target.exists():
            cache_target.parent.mkdir(parents=True, exist_ok=True)
            cache_target.symlink_to(R26_TEACHER_CACHE)
        teacher.OUT = destination
        completed.append({"name": name, "result": teacher.run(["B13"], device, benchmark_queries)})
    return {"completed": completed}


def _channel(metrics: dict[str, Any], kind: str, channel: str) -> dict[str, float]:
    return metrics[kind][channel]


def report() -> dict[str, Any]:
    """Compare every scope against the R26 shared-lake baseline."""
    baseline = {
        "own": json.loads(R26_B13_METRICS.read_text()),
        "teacher": json.loads(R26_TEACHER_METRICS.read_text()),
    }
    sources = {"dev_shared": baseline["own"], "dev_shared_teacher": baseline["teacher"]}
    for name in RUNS:
        own = run_dir(name) / "rankings/B13/metrics.json"
        if own.exists():
            sources[name] = json.loads(own.read_text())
        rerank = run_dir(name) / "teacher/B13/metrics.json"
        if rerank.exists():
            sources[f"{name}_teacher"] = json.loads(rerank.read_text())
    table = {}
    for name, metrics in sources.items():
        for kind in ("overall", "implicit", "explicit"):
            channels = metrics.get(kind) or {}
            for channel in ("D100_ANN", "U", "Equal"):
                if channel in channels:
                    table.setdefault(name, {}).setdefault(kind, {})[channel] = round(
                        _channel(metrics, kind, channel)["recall@10"] * 100, 2
                    )
            for channel in ("BT100_T0", "D100_T0"):
                if channel in channels:
                    table.setdefault(name, {}).setdefault(kind, {})[channel] = round(
                        _channel(metrics, kind, channel)["recall@10"] * 100, 2
                    )
    _json(OUT / "REPORT.json", table)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("build", "run", "teacher", "report"))
    parser.add_argument("--run", action="append", choices=tuple(RUNS))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--benchmark-queries", type=int, default=32)
    args = parser.parse_args()
    names = args.run or list(RUNS)
    if args.stage == "build":
        print(json.dumps(build(), ensure_ascii=False, indent=2))
    elif args.stage == "run":
        print(json.dumps(run_own(names, args.device), ensure_ascii=False))
    elif args.stage == "teacher":
        print(json.dumps(run_teacher(names, args.device, args.benchmark_queries), ensure_ascii=False))
    else:
        print(json.dumps(report(), ensure_ascii=False, indent=2))
