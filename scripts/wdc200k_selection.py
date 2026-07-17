"""Deterministic table selection from preserved WDC statistics archives."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import zipfile
from collections import defaultdict
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

try:
    from wdc200k_io import (
        AtomicJsonlShard,
        CompletedShard,
        StageFingerprint,
        StageManifest,
        external_unique_jsonl,
        stable_hash,
        validate_completed_shard,
    )
except ModuleNotFoundError as error:
    if error.name != "wdc200k_io":
        raise
    from scripts.wdc200k_io import (
        AtomicJsonlShard,
        CompletedShard,
        StageFingerprint,
        StageManifest,
        external_unique_jsonl,
        stable_hash,
        validate_completed_shard,
    )


SUBSETS = ("top100", "minimum3", "rest")
_SUBSET_PRIORITY = {subset: index for index, subset in enumerate(SUBSETS)}


@dataclass(frozen=True)
class TableCandidate:
    schema_class: str
    subset: str
    host: str
    relative_path: str
    rows: int
    columns: int


@dataclass(frozen=True)
class SelectionPolicy:
    target_tables: int = 200_000
    seed: int = 13
    minimum3_fraction: float = 0.90
    minimum3_base_per_class: int = 250
    rest_base_per_class: int = 50
    class_cap: int = 40_000

    def __post_init__(self) -> None:
        if self.target_tables < 0:
            raise ValueError("target_tables must be non-negative")
        if not 0.0 <= self.minimum3_fraction <= 1.0:
            raise ValueError("minimum3_fraction must be between zero and one")
        if self.minimum3_base_per_class < 0:
            raise ValueError("minimum3_base_per_class must be non-negative")
        if self.rest_base_per_class < 0:
            raise ValueError("rest_base_per_class must be non-negative")
        if self.class_cap <= 0:
            raise ValueError("class_cap must be positive")


@dataclass(frozen=True)
class SelectionResult:
    selected: list[TableCandidate]
    reserve: list[TableCandidate]
    quotas: dict[tuple[str, str], int]


def read_statistics_catalog(archive: Path) -> Iterator[TableCandidate]:
    """Yield candidates whose sibling extracted data file exists."""
    schema_class = archive.stem.removesuffix("_statistics")
    with zipfile.ZipFile(archive) as zipped:
        members = set(zipped.namelist())
        for subset in ("top100", "minimum3", "rest"):
            member = (
                f"table_statistics/{schema_class}_October2023"
                f"_statistics_{subset}.csv"
            )
            if member not in members:
                continue
            with zipped.open(member) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
                for row in csv.DictReader(text):
                    host = row["host"]
                    filename = (
                        f"{schema_class}_{host}_October2023.json.gz"
                    )
                    if not (archive.parent / filename).is_file():
                        continue
                    yield TableCandidate(
                        schema_class=schema_class,
                        subset=subset,
                        host=host,
                        relative_path=f"{schema_class}/{filename}",
                        rows=int(row["number_of_rows"]),
                        columns=int(row["column_count"]),
                    )


def _canonical_catalog(
    catalog: Iterable[TableCandidate],
) -> tuple[TableCandidate, ...]:
    by_path: dict[str, TableCandidate] = {}
    for candidate in catalog:
        if candidate.subset not in _SUBSET_PRIORITY:
            raise ValueError(f"unknown WDC subset: {candidate.subset}")
        existing = by_path.get(candidate.relative_path)
        if existing is None or (
            _SUBSET_PRIORITY[candidate.subset],
            candidate.schema_class,
            candidate.host,
        ) < (
            _SUBSET_PRIORITY[existing.subset],
            existing.schema_class,
            existing.host,
        ):
            by_path[candidate.relative_path] = candidate
    return tuple(by_path.values())


def _largest_remainder(
    capacities: dict[str, int],
    slots: int,
    *,
    square_root_weights: bool,
) -> dict[str, int]:
    allocations = {key: 0 for key in capacities}
    active = {
        key: capacity
        for key, capacity in capacities.items()
        if capacity > 0
    }
    remaining = min(max(0, slots), sum(active.values()))
    while remaining and active:
        weights = {
            key: (
                math.sqrt(capacity)
                if square_root_weights
                else float(capacity)
            )
            for key, capacity in active.items()
        }
        weight_total = sum(weights.values())
        exact = {
            key: remaining * weights[key] / weight_total
            for key in active
        }
        saturated = [
            key for key, capacity in active.items()
            if exact[key] >= capacity
        ]
        if saturated:
            for key in sorted(saturated):
                capacity = active.pop(key)
                allocations[key] += capacity
                remaining -= capacity
            continue

        floors = {
            key: min(active[key], math.floor(exact[key]))
            for key in active
        }
        for key, amount in floors.items():
            allocations[key] += amount
            active[key] -= amount
            remaining -= amount
        if remaining:
            remainder_order = sorted(
                active,
                key=lambda key: (-(exact[key] - floors[key]), key),
            )
            for key in remainder_order:
                if not remaining:
                    break
                if active[key] <= 0:
                    continue
                allocations[key] += 1
                active[key] -= 1
                remaining -= 1
        break
    return allocations


def _apply_subset_allocation(
    subset: str,
    slots: int,
    base_per_class: int,
    availability: dict[tuple[str, str], int],
    quotas: dict[tuple[str, str], int],
    class_used: dict[str, int],
    class_cap: int,
) -> int:
    capacities = {
        schema_class: min(
            availability.get((schema_class, subset), 0)
            - quotas.get((schema_class, subset), 0),
            class_cap - class_used[schema_class],
        )
        for schema_class in class_used
    }
    capacities = {
        key: max(0, capacity) for key, capacity in capacities.items()
    }
    base_capacities = {
        key: min(base_per_class, capacity)
        for key, capacity in capacities.items()
    }
    base_slots = min(slots, sum(base_capacities.values()))
    base_allocations = _largest_remainder(
        base_capacities,
        base_slots,
        square_root_weights=False,
    )
    allocated = 0
    for schema_class, amount in base_allocations.items():
        quotas[(schema_class, subset)] += amount
        class_used[schema_class] += amount
        capacities[schema_class] -= amount
        allocated += amount

    extra_allocations = _largest_remainder(
        capacities,
        slots - allocated,
        square_root_weights=True,
    )
    for schema_class, amount in extra_allocations.items():
        quotas[(schema_class, subset)] += amount
        class_used[schema_class] += amount
        allocated += amount
    return allocated


def _apply_global_allocation(
    slots: int,
    availability: dict[tuple[str, str], int],
    quotas: dict[tuple[str, str], int],
    class_used: dict[str, int],
    class_cap: int,
) -> int:
    remaining_by_stratum = {
        key: max(0, count - quotas.get(key, 0))
        for key, count in availability.items()
        if key[1] != "top100"
    }
    class_capacities = {
        schema_class: min(
            class_cap - class_used[schema_class],
            sum(
                count
                for (candidate_class, _subset), count
                in remaining_by_stratum.items()
                if candidate_class == schema_class
            ),
        )
        for schema_class in class_used
    }
    class_allocations = _largest_remainder(
        class_capacities,
        slots,
        square_root_weights=True,
    )
    allocated = 0
    for schema_class in sorted(class_allocations):
        class_slots = class_allocations[schema_class]
        subset_capacities = {
            subset: remaining_by_stratum.get((schema_class, subset), 0)
            for subset in ("minimum3", "rest")
        }
        subset_allocations = _largest_remainder(
            subset_capacities,
            class_slots,
            square_root_weights=True,
        )
        for subset, amount in subset_allocations.items():
            quotas[(schema_class, subset)] += amount
            class_used[schema_class] += amount
            allocated += amount
    return allocated


def allocate_strata(
    catalog: Iterable[TableCandidate],
    policy: SelectionPolicy = SelectionPolicy(),
) -> dict[tuple[str, str], int]:
    """Allocate exact deterministic class/subset quotas when capacity permits."""
    candidates = _canonical_catalog(catalog)
    availability: dict[tuple[str, str], int] = defaultdict(int)
    for candidate in candidates:
        availability[(candidate.schema_class, candidate.subset)] += 1
    return _allocate_from_availability(availability, policy)


def _allocate_from_availability(
    availability: dict[tuple[str, str], int],
    policy: SelectionPolicy,
) -> dict[tuple[str, str], int]:
    classes = sorted({key[0] for key in availability})
    quotas = {
        (schema_class, subset): 0
        for schema_class in classes
        for subset in SUBSETS
    }
    class_used = {schema_class: 0 for schema_class in classes}

    mandatory = 0
    for schema_class in classes:
        top100 = availability.get((schema_class, "top100"), 0)
        if top100 > policy.class_cap:
            raise ValueError(
                f"top100 candidates exceed class cap for {schema_class}"
            )
        quotas[(schema_class, "top100")] = top100
        class_used[schema_class] = top100
        mandatory += top100
    if mandatory > policy.target_tables:
        raise ValueError("top100 candidates exceed target_tables")

    feasible_total = sum(
        min(
            policy.class_cap,
            sum(
                availability.get((schema_class, subset), 0)
                for subset in SUBSETS
            ),
        )
        for schema_class in classes
    )
    target = min(policy.target_tables, feasible_total)
    remaining = target - mandatory
    minimum3_goal = round(policy.minimum3_fraction * remaining)
    rest_goal = remaining - minimum3_goal

    minimum3_allocated = _apply_subset_allocation(
        "minimum3",
        minimum3_goal,
        policy.minimum3_base_per_class,
        availability,
        quotas,
        class_used,
        policy.class_cap,
    )
    rest_allocated = _apply_subset_allocation(
        "rest",
        rest_goal,
        policy.rest_base_per_class,
        availability,
        quotas,
        class_used,
        policy.class_cap,
    )
    rest_shortfall = rest_goal - rest_allocated
    if rest_shortfall:
        minimum3_allocated += _apply_subset_allocation(
            "minimum3",
            rest_shortfall,
            0,
            availability,
            quotas,
            class_used,
            policy.class_cap,
        )

    allocated = mandatory + minimum3_allocated + rest_allocated
    if allocated < target:
        allocated += _apply_global_allocation(
            target - allocated,
            availability,
            quotas,
            class_used,
            policy.class_cap,
        )
    if allocated != target:
        raise RuntimeError("unable to allocate feasible selection target")
    return {key: amount for key, amount in quotas.items() if amount}


def _candidate_rank(
    candidate: TableCandidate,
    seed: int,
) -> tuple[str, str]:
    return stable_hash(seed, candidate.relative_path), candidate.relative_path


def select_tables(
    catalog: Iterable[TableCandidate],
    policy: SelectionPolicy = SelectionPolicy(),
) -> SelectionResult:
    """Select candidates by allocated stratum and stable path hash."""
    candidates = _canonical_catalog(catalog)
    quotas = allocate_strata(candidates, policy)
    by_stratum: dict[tuple[str, str], list[TableCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_stratum[(candidate.schema_class, candidate.subset)].append(
            candidate
        )

    selected: list[TableCandidate] = []
    reserve: list[TableCandidate] = []
    for stratum, records in by_stratum.items():
        records.sort(key=lambda item: _candidate_rank(item, policy.seed))
        selected_count = quotas.get(stratum, 0)
        selected.extend(records[:selected_count])
        reserve.extend(records[selected_count:])
    selected.sort(key=lambda item: _candidate_rank(item, policy.seed))
    reserve.sort(key=lambda item: _candidate_rank(item, policy.seed))
    return SelectionResult(
        selected=selected,
        reserve=reserve,
        quotas=quotas,
    )


def replace_invalid_selection(
    invalid_candidate: TableCandidate,
    reserve: Iterable[TableCandidate],
    is_invalid: Callable[[TableCandidate], bool] | None = None,
    *,
    used_paths: Collection[str] = (),
) -> TableCandidate | None:
    """Choose a valid same-stratum reserve before the global reserve."""
    invalid_test = is_invalid or (lambda _candidate: False)
    global_fallback: TableCandidate | None = None
    for candidate in reserve:
        if (
            candidate.relative_path in used_paths
            or candidate.relative_path == invalid_candidate.relative_path
            or invalid_test(candidate)
        ):
            continue
        if global_fallback is None:
            global_fallback = candidate
        if (
            candidate.schema_class == invalid_candidate.schema_class
            and candidate.subset == invalid_candidate.subset
        ):
            return candidate
    return global_fallback


def _candidate_record(
    candidate: TableCandidate,
    policy: SelectionPolicy,
) -> dict[str, object]:
    return {
        "schema_class": candidate.schema_class,
        "subset": candidate.subset,
        "host": candidate.host,
        "relative_path": candidate.relative_path,
        "rows": candidate.rows,
        "columns": candidate.columns,
        "rank": stable_hash(policy.seed, candidate.relative_path),
        "selection_seed": policy.seed,
    }


def _statistics_archives(input_dir: Path) -> list[Path]:
    archives = []
    for class_dir in input_dir.iterdir():
        if not class_dir.is_dir():
            continue
        archive = class_dir / f"{class_dir.name}_statistics.zip"
        if archive.is_file():
            archives.append(archive)
    return sorted(archives)


def _iter_input_catalog(archives: Iterable[Path]) -> Iterator[TableCandidate]:
    for archive in archives:
        yield from read_statistics_catalog(archive)


def _write_selection_outputs(
    archives: list[Path],
    selection_dir: Path,
    policy: SelectionPolicy,
    sort_chunk_records: int,
) -> tuple[CompletedShard, CompletedShard]:
    unsorted_path = selection_dir / "catalog-unsorted.jsonl"
    unique_path = selection_dir / "catalog-unique.jsonl"
    sorted_path = selection_dir / "catalog-ranked.jsonl"
    unsorted = AtomicJsonlShard(unsorted_path)
    selected: AtomicJsonlShard | None = None
    reserve: AtomicJsonlShard | None = None
    try:
        for candidate in _iter_input_catalog(archives):
            unsorted.write(_candidate_record(candidate, policy))
        unsorted.commit()
        external_unique_jsonl(
            [unsorted_path],
            unique_path,
            key_fn=lambda record: record["relative_path"],
            chunk_records=sort_chunk_records,
        )
        availability: dict[tuple[str, str], int] = defaultdict(int)
        with unique_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                availability[
                    (str(record["schema_class"]), str(record["subset"]))
                ] += 1
        quotas = _allocate_from_availability(availability, policy)
        external_unique_jsonl(
            [unique_path],
            sorted_path,
            key_fn=lambda record: [
                record["rank"],
                record["relative_path"],
            ],
            chunk_records=sort_chunk_records,
        )

        selected = AtomicJsonlShard(
            selection_dir / "selected_tables.jsonl"
        )
        reserve = AtomicJsonlShard(
            selection_dir / "reserve_tables.jsonl"
        )
        selected_by_stratum: dict[tuple[str, str], int] = defaultdict(int)
        with sorted_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                stratum = (
                    str(record["schema_class"]),
                    str(record["subset"]),
                )
                if selected_by_stratum[stratum] < quotas.get(stratum, 0):
                    selected.write(record)
                    selected_by_stratum[stratum] += 1
                else:
                    reserve.write(record)
        selected_completed = selected.commit()
        reserve_completed = reserve.commit()
        if selected_completed.records != sum(quotas.values()):
            raise RuntimeError("selection output did not satisfy its quotas")
    except BaseException:
        unsorted.abort()
        if selected is not None:
            selected.abort()
        if reserve is not None:
            reserve.abort()
        raise
    finally:
        unsorted_path.unlink(missing_ok=True)
        unique_path.unlink(missing_ok=True)
        sorted_path.unlink(missing_ok=True)
    return selected_completed, reserve_completed


def run_selection(
    input_dir: Path,
    work_dir: Path,
    policy: SelectionPolicy,
    *,
    sort_chunk_records: int = 100_000,
) -> tuple[int, int]:
    """Create or resume the durable provisional selection stage."""
    if sort_chunk_records <= 0:
        raise ValueError("sort_chunk_records must be positive")
    archives = _statistics_archives(input_dir)
    if not archives:
        raise ValueError(f"no statistics archives found under {input_dir}")
    selection_dir = work_dir / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    input_fingerprint = stable_hash(
        *(
            f"{archive.relative_to(input_dir)}:"
            f"{archive.stat().st_size}:{archive.stat().st_mtime_ns}"
            for archive in archives
        ),
        length=40,
    )
    parameter_fingerprint = stable_hash(
        json.dumps(
            {
                "policy": policy.__dict__,
                "sort_chunk_records": sort_chunk_records,
            },
            sort_keys=True,
        ),
        length=40,
    )
    manifest = StageManifest(
        selection_dir / "manifest.json",
        StageFingerprint(
            stage="wdc200k_selection",
            input_fingerprint=input_fingerprint,
            parameter_fingerprint=parameter_fingerprint,
        ),
    )
    if manifest.complete:
        if not all(
            validate_completed_shard(shard, selection_dir)
            for shard in manifest.completed_shards
        ):
            raise RuntimeError("completed selection output failed validation")
        counts = {
            shard.path: shard.records for shard in manifest.completed_shards
        }
        return (
            counts.get("selected_tables.jsonl", 0),
            counts.get("reserve_tables.jsonl", 0),
        )

    selected_completed, reserve_completed = _write_selection_outputs(
        archives,
        selection_dir,
        policy,
        sort_chunk_records,
    )
    manifest.record_shard(selected_completed)
    manifest.record_shard(reserve_completed)
    manifest.mark_complete()
    return selected_completed.records, reserve_completed.records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select deterministic WDC tables from statistics ZIPs."
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--work_dir", type=Path, required=True)
    parser.add_argument("--target_tables", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--minimum3_fraction", type=float, default=0.90)
    parser.add_argument("--minimum3_base_per_class", type=int, default=250)
    parser.add_argument("--rest_base_per_class", type=int, default=50)
    parser.add_argument("--class_cap", type=int, default=40_000)
    parser.add_argument("--sort_chunk_records", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = SelectionPolicy(
        target_tables=args.target_tables,
        seed=args.seed,
        minimum3_fraction=args.minimum3_fraction,
        minimum3_base_per_class=args.minimum3_base_per_class,
        rest_base_per_class=args.rest_base_per_class,
        class_cap=args.class_cap,
    )
    selected, reserve = run_selection(
        args.input_dir,
        args.work_dir,
        policy,
        sort_chunk_records=args.sort_chunk_records,
    )
    print(json.dumps({"selected": selected, "reserve": reserve}))


if __name__ == "__main__":
    main()
