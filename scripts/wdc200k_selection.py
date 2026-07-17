"""Deterministic table selection from preserved WDC statistics archives."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sqlite3
import time
import uuid
import zipfile
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
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
    """Yield provisional candidates declared by a statistics archive."""
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
        if existing is None or _candidate_canonical_key(
            candidate
        ) < _candidate_canonical_key(existing):
            by_path[candidate.relative_path] = candidate
    return tuple(by_path.values())


def _candidate_canonical_key(
    candidate: TableCandidate,
) -> tuple[int, str, str, str, int, int]:
    return (
        _SUBSET_PRIORITY[candidate.subset],
        candidate.schema_class,
        candidate.subset,
        candidate.host,
        candidate.rows,
        candidate.columns,
    )


def _record_canonical_key(record: dict[str, object]) -> list[object]:
    subset = str(record["subset"])
    return [
        record["relative_path"],
        _SUBSET_PRIORITY[subset],
        record["schema_class"],
        subset,
        record["host"],
        _nonnegative_integer_sort_key(record["rows"]),
        _nonnegative_integer_sort_key(record["columns"]),
    ]


def _nonnegative_integer_sort_key(value: object) -> str:
    number = int(value)
    if number < 0:
        raise ValueError("row and column counts must be non-negative")
    text = str(number)
    return f"{len(text):08d}:{text}"


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
    if feasible_total < policy.target_tables:
        raise ValueError(
            f"requested {policy.target_tables} tables but only "
            f"{feasible_total} feasible under the class cap"
        )
    target = policy.target_tables
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


class ReserveExhaustedError(RuntimeError):
    """Raised when no unused reserve can preserve the selection constraints."""

    def __init__(self, claim: ReplacementClaim) -> None:
        self.claim = claim
        super().__init__(
            f"no reserve candidate can replace {claim.invalid_path}; "
            f"operation {claim.operation_key} exhausted: {claim.reason}"
        )


class ReplacementOperationStateError(RuntimeError):
    """Raised when a terminal operation cannot yield expansion work."""


@dataclass(frozen=True)
class ReplacementClaim:
    operation_key: str
    invalid_path: str
    replacement: TableCandidate | None
    reason: str
    status: str
    created_at: float
    acknowledged_at: float | None
    terminal_at: float | None
    successor_operation_key: str | None

    @property
    def replacement_path(self) -> str | None:
        return (
            None
            if self.replacement is None
            else self.replacement.relative_path
        )


_CLAIM_SELECT = """
    SELECT
        operations.operation_key,
        operations.invalid_path,
        operations.replacement_path,
        operations.reason,
        operations.status,
        operations.created_at,
        operations.acknowledged_at,
        operations.terminal_at,
        operations.successor_operation_key,
        candidates.schema_class AS replacement_schema_class,
        candidates.subset_name AS replacement_subset,
        candidates.host AS replacement_host,
        candidates.rows_count AS replacement_rows,
        candidates.columns_count AS replacement_columns
    FROM replacement_operations AS operations
    LEFT JOIN candidates
      ON candidates.relative_path = operations.replacement_path
"""


class ReserveManager:
    """Persist reserve cursors, used paths, and active class counts in SQLite."""

    def __init__(self, database_path: Path, policy: SelectionPolicy) -> None:
        self.database_path = database_path
        self.policy = policy

    @classmethod
    def create(
        cls,
        database_path: Path,
        *,
        reserve: Iterable[TableCandidate],
        selected: Iterable[TableCandidate],
        policy: SelectionPolicy,
    ) -> ReserveManager:
        """Atomically build indexed reserve state without materializing input."""
        if database_path.exists():
            raise FileExistsError(database_path)
        database_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = database_path.parent / (
            f".{database_path.name}.{uuid.uuid4().hex}.tmp"
        )
        connection = sqlite3.connect(temporary_path)
        try:
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript(
                """
                CREATE TABLE metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE candidates (
                    ordinal INTEGER PRIMARY KEY,
                    relative_path TEXT NOT NULL UNIQUE,
                    schema_class TEXT NOT NULL,
                    subset_name TEXT NOT NULL,
                    host TEXT NOT NULL,
                    rows_count INTEGER NOT NULL,
                    columns_count INTEGER NOT NULL,
                    used INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE active_selections (
                    relative_path TEXT PRIMARY KEY,
                    schema_class TEXT NOT NULL,
                    subset_name TEXT NOT NULL,
                    host TEXT NOT NULL,
                    rows_count INTEGER NOT NULL,
                    columns_count INTEGER NOT NULL
                );
                CREATE TABLE used_paths (
                    relative_path TEXT PRIMARY KEY
                );
                CREATE TABLE class_counts (
                    schema_class TEXT PRIMARY KEY,
                    active_count INTEGER NOT NULL
                );
                CREATE TABLE replacement_operations (
                    operation_key TEXT PRIMARY KEY,
                    invalid_path TEXT NOT NULL UNIQUE,
                    invalid_schema_class TEXT NOT NULL,
                    invalid_subset TEXT NOT NULL,
                    invalid_host TEXT NOT NULL,
                    invalid_rows INTEGER NOT NULL,
                    invalid_columns INTEGER NOT NULL,
                    replacement_path TEXT UNIQUE,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL
                        CHECK(status IN (
                            'pending', 'acked',
                            'superseded', 'exhausted'
                        )),
                    created_at REAL NOT NULL,
                    acknowledged_at REAL,
                    terminal_at REAL,
                    successor_operation_key TEXT,
                    FOREIGN KEY(replacement_path)
                        REFERENCES candidates(relative_path),
                    FOREIGN KEY(successor_operation_key)
                        REFERENCES replacement_operations(operation_key)
                );
                CREATE INDEX candidates_same_stratum
                ON candidates (
                    schema_class, subset_name, used, ordinal
                );
                CREATE INDEX candidates_class_order
                ON candidates (schema_class, used, ordinal);
                CREATE INDEX replacement_operations_pending
                ON replacement_operations (
                    status, created_at, operation_key
                );
                """
            )
            connection.execute(
                "INSERT INTO metadata (key, value) VALUES ('policy', ?)",
                (json.dumps(asdict(policy), sort_keys=True),),
            )
            for ordinal, candidate in enumerate(reserve):
                _validate_candidate_subset(candidate)
                try:
                    connection.execute(
                        """
                        INSERT INTO candidates (
                            ordinal, relative_path, schema_class,
                            subset_name, host, rows_count, columns_count
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            ordinal,
                            candidate.relative_path,
                            candidate.schema_class,
                            candidate.subset,
                            candidate.host,
                            candidate.rows,
                            candidate.columns,
                        ),
                    )
                except sqlite3.IntegrityError as error:
                    raise ValueError(
                        "reserve contains a duplicate relative_path: "
                        f"{candidate.relative_path}"
                    ) from error
                connection.execute(
                    """
                    INSERT OR IGNORE INTO class_counts (
                        schema_class, active_count
                    ) VALUES (?, 0)
                    """,
                    (candidate.schema_class,),
                )

            selected_count = 0
            for candidate in selected:
                _validate_candidate_subset(candidate)
                try:
                    connection.execute(
                        """
                        INSERT INTO active_selections (
                            relative_path, schema_class, subset_name,
                            host, rows_count, columns_count
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            candidate.relative_path,
                            candidate.schema_class,
                            candidate.subset,
                            candidate.host,
                            candidate.rows,
                            candidate.columns,
                        ),
                    )
                except sqlite3.IntegrityError as error:
                    raise ValueError(
                        "selected candidates contain a duplicate "
                        f"relative_path: {candidate.relative_path}"
                    ) from error
                selected_count += 1
                connection.execute(
                    """
                    INSERT INTO used_paths (relative_path)
                    VALUES (?)
                    ON CONFLICT(relative_path) DO NOTHING
                    """,
                    (candidate.relative_path,),
                )
                connection.execute(
                    """
                    INSERT INTO class_counts (
                        schema_class, active_count
                    ) VALUES (?, 1)
                    ON CONFLICT(schema_class) DO UPDATE SET
                        active_count = active_count + 1
                    """,
                    (candidate.schema_class,),
                )
                connection.execute(
                    """
                    UPDATE candidates SET used = 1
                    WHERE relative_path = ?
                    """,
                    (candidate.relative_path,),
                )

            if selected_count != policy.target_tables:
                raise ValueError(
                    f"selected count {selected_count} does not match "
                    f"target_tables {policy.target_tables}"
                )
            over_cap = connection.execute(
                """
                SELECT schema_class, active_count
                FROM class_counts
                WHERE active_count > ?
                ORDER BY schema_class
                LIMIT 1
                """,
                (policy.class_cap,),
            ).fetchone()
            if over_cap is not None:
                raise ValueError(
                    f"selected class {over_cap[0]} has {over_cap[1]} "
                    f"tables above class cap {policy.class_cap}"
                )
            connection.commit()
        except BaseException:
            connection.rollback()
            connection.close()
            temporary_path.unlink(missing_ok=True)
            raise
        else:
            connection.close()
        try:
            os.link(temporary_path, database_path)
        except FileExistsError:
            raise FileExistsError(
                f"reserve database was concurrently published: "
                f"{database_path}"
            ) from None
        finally:
            temporary_path.unlink(missing_ok=True)
        _fsync_parent(database_path.parent)
        return cls.open(database_path, policy)

    @classmethod
    def create_from_jsonl(
        cls,
        database_path: Path,
        *,
        reserve_path: Path,
        selected_path: Path,
        policy: SelectionPolicy,
    ) -> ReserveManager:
        """Build persistent reserve state by streaming CLI selection JSONL."""
        return cls.create(
            database_path,
            reserve=_iter_candidate_jsonl(reserve_path),
            selected=_iter_candidate_jsonl(selected_path),
            policy=policy,
        )

    @classmethod
    def open(
        cls,
        database_path: Path,
        policy: SelectionPolicy,
    ) -> ReserveManager:
        """Open existing reserve state and verify its selection policy."""
        if not database_path.is_file():
            raise FileNotFoundError(database_path)
        connection = _connect_sqlite_read_write(database_path)
        try:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'policy'"
            ).fetchone()
        finally:
            connection.close()
        expected = json.dumps(asdict(policy), sort_keys=True)
        if row is None or str(row[0]) != expected:
            raise ValueError("reserve database policy does not match")
        return cls(database_path, policy)

    def class_counts(self) -> dict[str, int]:
        connection = self._connect()
        try:
            return {
                str(row["schema_class"]): int(row["active_count"])
                for row in connection.execute(
                    """
                    SELECT schema_class, active_count
                    FROM class_counts
                    ORDER BY schema_class
                    """
                )
            }
        finally:
            connection.close()

    def used_paths(self) -> set[str]:
        connection = self._connect()
        try:
            return {
                str(row["relative_path"])
                for row in connection.execute(
                    "SELECT relative_path FROM used_paths"
                )
            }
        finally:
            connection.close()

    def replace(
        self,
        invalid_candidate: TableCandidate,
        *,
        operation_key: str,
        reason: str,
        is_invalid: Callable[[TableCandidate], bool] | None = None,
    ) -> TableCandidate:
        """Compatibility helper returning the claimed replacement candidate."""
        claim = self.claim_replacement(
            operation_key=operation_key,
            invalid_candidate=invalid_candidate,
            reason=reason,
            is_invalid=is_invalid,
        )
        if claim.status != "pending" or claim.replacement is None:
            raise ReplacementOperationStateError(
                f"replacement operation {claim.operation_key} is "
                f"{claim.status}; no expansion work may be replayed"
            )
        return claim.replacement

    def claim_replacement(
        self,
        *,
        operation_key: str,
        invalid_candidate: TableCandidate,
        reason: str,
        is_invalid: Callable[[TableCandidate], bool] | None = None,
    ) -> ReplacementClaim:
        """Persist and return an idempotent pending replacement operation."""
        if not operation_key:
            raise ValueError("operation_key must be non-empty")
        if not reason:
            raise ValueError("replacement reason must be non-empty")
        invalid_test = is_invalid or (lambda _candidate: False)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing_operation = self._claim_by_operation(
                connection,
                operation_key,
            )
            if existing_operation is not None:
                if (
                    existing_operation.invalid_path
                    != invalid_candidate.relative_path
                ):
                    raise ValueError(
                        f"operation key belongs to invalid path "
                        f"{existing_operation.invalid_path}"
                    )
                self._validate_invalid_metadata(
                    connection,
                    existing_operation.operation_key,
                    invalid_candidate,
                )
                if existing_operation.reason != reason:
                    raise ValueError(
                        "operation key replacement reason does not match"
                    )
                connection.commit()
                if existing_operation.status == "exhausted":
                    raise ReserveExhaustedError(existing_operation)
                return existing_operation

            existing_invalid = self._claim_by_invalid(
                connection,
                invalid_candidate.relative_path,
            )
            if existing_invalid is not None:
                self._validate_invalid_metadata(
                    connection,
                    existing_invalid.operation_key,
                    invalid_candidate,
                )
                connection.commit()
                if existing_invalid.status == "exhausted":
                    raise ReserveExhaustedError(existing_invalid)
                return existing_invalid

            active = connection.execute(
                """
                SELECT *
                FROM active_selections
                WHERE relative_path = ?
                """,
                (invalid_candidate.relative_path,),
            ).fetchone()
            if active is None:
                raise ValueError(
                    "invalid candidate is not an active selection: "
                    f"{invalid_candidate.relative_path}"
                )
            stored_candidate = _candidate_from_active_sqlite(active)
            if stored_candidate != invalid_candidate:
                raise ValueError(
                    "active candidate metadata does not match persisted state"
                )
            predecessor = self._claim_by_replacement(
                connection,
                invalid_candidate.relative_path,
            )
            connection.execute(
                """
                DELETE FROM active_selections
                WHERE relative_path = ?
                """,
                (invalid_candidate.relative_path,),
            )
            connection.execute(
                """
                UPDATE class_counts
                SET active_count = active_count - 1
                WHERE schema_class = ?
                """,
                (invalid_candidate.schema_class,),
            )

            replacement = self._consume_same_stratum(
                connection,
                invalid_candidate,
                invalid_test,
            )
            if replacement is None:
                replacement = self._consume_global(
                    connection,
                    invalid_test,
                )
            if replacement is None:
                now = time.time()
                connection.execute(
                    """
                    INSERT INTO replacement_operations (
                        operation_key, invalid_path,
                        invalid_schema_class, invalid_subset,
                        invalid_host, invalid_rows, invalid_columns,
                        replacement_path, reason, status,
                        created_at, terminal_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?,
                        NULL, ?, 'exhausted', ?, ?
                    )
                    """,
                    (
                        operation_key,
                        invalid_candidate.relative_path,
                        invalid_candidate.schema_class,
                        invalid_candidate.subset,
                        invalid_candidate.host,
                        invalid_candidate.rows,
                        invalid_candidate.columns,
                        reason,
                        now,
                        now,
                    ),
                )
                self._supersede_pending_predecessor(
                    connection,
                    predecessor,
                    operation_key,
                    now,
                )
                claim = self._claim_by_operation(connection, operation_key)
                if claim is None:
                    raise RuntimeError(
                        "exhausted replacement operation was not visible"
                    )
                connection.commit()
                raise ReserveExhaustedError(claim)
            connection.execute(
                """
                INSERT INTO active_selections (
                    relative_path, schema_class, subset_name,
                    host, rows_count, columns_count
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    replacement.relative_path,
                    replacement.schema_class,
                    replacement.subset,
                    replacement.host,
                    replacement.rows,
                    replacement.columns,
                ),
            )
            connection.execute(
                """
                UPDATE class_counts
                SET active_count = active_count + 1
                WHERE schema_class = ?
                """,
                (replacement.schema_class,),
            )
            connection.execute(
                """
                INSERT INTO replacement_operations (
                    operation_key, invalid_path,
                    invalid_schema_class, invalid_subset,
                    invalid_host, invalid_rows, invalid_columns,
                    replacement_path, reason, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    operation_key,
                    invalid_candidate.relative_path,
                    invalid_candidate.schema_class,
                    invalid_candidate.subset,
                    invalid_candidate.host,
                    invalid_candidate.rows,
                    invalid_candidate.columns,
                    replacement.relative_path,
                    reason,
                    time.time(),
                ),
            )
            self._supersede_pending_predecessor(
                connection,
                predecessor,
                operation_key,
                time.time(),
            )
            claim = self._claim_by_operation(connection, operation_key)
            if claim is None:
                raise RuntimeError("replacement journal insert was not visible")
            connection.commit()
            return claim
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def pending_claims(self) -> list[ReplacementClaim]:
        """Return committed operations awaiting durable-output acknowledgment."""
        connection = self._connect()
        try:
            return [
                _replacement_claim_from_row(row)
                for row in connection.execute(
                    _CLAIM_SELECT
                    + """
                    WHERE operations.status = 'pending'
                    ORDER BY operations.created_at, operations.operation_key
                    """
                )
            ]
        finally:
            connection.close()

    def terminal_claims(self) -> list[ReplacementClaim]:
        """Return acked, superseded, and exhausted operation history."""
        connection = self._connect()
        try:
            return [
                _replacement_claim_from_row(row)
                for row in connection.execute(
                    _CLAIM_SELECT
                    + """
                    WHERE operations.status != 'pending'
                    ORDER BY operations.created_at, operations.operation_key
                    """
                )
            ]
        finally:
            connection.close()

    def acknowledge(
        self,
        *,
        operation_key: str,
        replacement_path: str,
    ) -> ReplacementClaim:
        """Mark a claim acked after Task 3 durably commits replacement output."""
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            claim = self._claim_by_operation(connection, operation_key)
            if claim is None:
                raise KeyError(
                    f"unknown replacement operation: {operation_key}"
                )
            if claim.status in {"superseded", "exhausted"}:
                raise ValueError(
                    f"cannot acknowledge {claim.status} replacement "
                    f"operation {operation_key}"
                )
            if claim.replacement_path != replacement_path:
                raise ValueError(
                    "replacement path does not match operation "
                    f"{operation_key}"
                )
            if claim.status == "pending":
                connection.execute(
                    """
                    UPDATE replacement_operations
                    SET status = 'acked', acknowledged_at = ?
                    WHERE operation_key = ? AND status = 'pending'
                    """,
                    (time.time(), operation_key),
                )
                claim = self._claim_by_operation(connection, operation_key)
                if claim is None:
                    raise RuntimeError(
                        "acknowledged replacement operation disappeared"
                    )
            connection.commit()
            return claim
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _claim_by_operation(
        connection: sqlite3.Connection,
        operation_key: str,
    ) -> ReplacementClaim | None:
        row = connection.execute(
            _CLAIM_SELECT + "WHERE operations.operation_key = ?",
            (operation_key,),
        ).fetchone()
        return None if row is None else _replacement_claim_from_row(row)

    @staticmethod
    def _claim_by_invalid(
        connection: sqlite3.Connection,
        invalid_path: str,
    ) -> ReplacementClaim | None:
        row = connection.execute(
            _CLAIM_SELECT + "WHERE operations.invalid_path = ?",
            (invalid_path,),
        ).fetchone()
        return None if row is None else _replacement_claim_from_row(row)

    @staticmethod
    def _claim_by_replacement(
        connection: sqlite3.Connection,
        replacement_path: str,
    ) -> ReplacementClaim | None:
        row = connection.execute(
            _CLAIM_SELECT + "WHERE operations.replacement_path = ?",
            (replacement_path,),
        ).fetchone()
        return None if row is None else _replacement_claim_from_row(row)

    @staticmethod
    def _supersede_pending_predecessor(
        connection: sqlite3.Connection,
        predecessor: ReplacementClaim | None,
        successor_operation_key: str,
        terminal_at: float,
    ) -> None:
        if predecessor is None or predecessor.status != "pending":
            return
        cursor = connection.execute(
            """
            UPDATE replacement_operations
            SET status = 'superseded',
                successor_operation_key = ?,
                terminal_at = ?
            WHERE operation_key = ? AND status = 'pending'
            """,
            (
                successor_operation_key,
                terminal_at,
                predecessor.operation_key,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(
                "pending predecessor changed during replacement transaction"
            )

    @staticmethod
    def _validate_invalid_metadata(
        connection: sqlite3.Connection,
        operation_key: str,
        candidate: TableCandidate,
    ) -> None:
        row = connection.execute(
            """
            SELECT
                invalid_schema_class, invalid_subset, invalid_host,
                invalid_rows, invalid_columns
            FROM replacement_operations
            WHERE operation_key = ?
            """,
            (operation_key,),
        ).fetchone()
        stored = (
            str(row["invalid_schema_class"]),
            str(row["invalid_subset"]),
            str(row["invalid_host"]),
            int(row["invalid_rows"]),
            int(row["invalid_columns"]),
        )
        requested = (
            candidate.schema_class,
            candidate.subset,
            candidate.host,
            candidate.rows,
            candidate.columns,
        )
        if stored != requested:
            raise ValueError(
                "invalid candidate metadata does not match "
                f"operation {operation_key}"
            )

    def _consume_same_stratum(
        self,
        connection: sqlite3.Connection,
        invalid_candidate: TableCandidate,
        is_invalid: Callable[[TableCandidate], bool],
    ) -> TableCandidate | None:
        while True:
            row = connection.execute(
                """
                SELECT *
                FROM candidates
                WHERE schema_class = ?
                  AND subset_name = ?
                  AND used = 0
                ORDER BY ordinal
                LIMIT 1
                """,
                (
                    invalid_candidate.schema_class,
                    invalid_candidate.subset,
                ),
            ).fetchone()
            if row is None:
                return None
            candidate = _candidate_from_sqlite(row)
            self._mark_consumed(connection, candidate.relative_path)
            if not is_invalid(candidate):
                return candidate

    def _consume_global(
        self,
        connection: sqlite3.Connection,
        is_invalid: Callable[[TableCandidate], bool],
    ) -> TableCandidate | None:
        while True:
            feasible_classes = [
                str(row["schema_class"])
                for row in connection.execute(
                    """
                    SELECT schema_class
                    FROM class_counts
                    WHERE active_count < ?
                    ORDER BY schema_class
                    """,
                    (self.policy.class_cap,),
                )
            ]
            heads = []
            for schema_class in feasible_classes:
                row = connection.execute(
                    """
                    SELECT *
                    FROM candidates
                    WHERE schema_class = ? AND used = 0
                    ORDER BY ordinal
                    LIMIT 1
                    """,
                    (schema_class,),
                ).fetchone()
                if row is not None:
                    heads.append(row)
            if not heads:
                return None
            row = min(heads, key=lambda item: int(item["ordinal"]))
            candidate = _candidate_from_sqlite(row)
            self._mark_consumed(connection, candidate.relative_path)
            if not is_invalid(candidate):
                return candidate

    @staticmethod
    def _mark_consumed(
        connection: sqlite3.Connection,
        relative_path: str,
    ) -> None:
        connection.execute(
            "UPDATE candidates SET used = 1 WHERE relative_path = ?",
            (relative_path,),
        )
        connection.execute(
            """
            INSERT INTO used_paths (relative_path)
            VALUES (?)
            ON CONFLICT(relative_path) DO NOTHING
            """,
            (relative_path,),
        )

    def _connect(self) -> sqlite3.Connection:
        connection = _connect_sqlite_read_write(
            self.database_path,
            timeout=30.0,
        )
        connection.row_factory = sqlite3.Row
        return connection


def _validate_candidate_subset(candidate: TableCandidate) -> None:
    if candidate.subset not in _SUBSET_PRIORITY:
        raise ValueError(f"unknown WDC subset: {candidate.subset}")


def _candidate_from_sqlite(row: sqlite3.Row) -> TableCandidate:
    return TableCandidate(
        schema_class=str(row["schema_class"]),
        subset=str(row["subset_name"]),
        host=str(row["host"]),
        relative_path=str(row["relative_path"]),
        rows=int(row["rows_count"]),
        columns=int(row["columns_count"]),
    )


def _candidate_from_active_sqlite(row: sqlite3.Row) -> TableCandidate:
    return TableCandidate(
        schema_class=str(row["schema_class"]),
        subset=str(row["subset_name"]),
        host=str(row["host"]),
        relative_path=str(row["relative_path"]),
        rows=int(row["rows_count"]),
        columns=int(row["columns_count"]),
    )


def _replacement_claim_from_row(row: sqlite3.Row) -> ReplacementClaim:
    replacement = (
        None
        if row["replacement_path"] is None
        else TableCandidate(
            schema_class=str(row["replacement_schema_class"]),
            subset=str(row["replacement_subset"]),
            host=str(row["replacement_host"]),
            relative_path=str(row["replacement_path"]),
            rows=int(row["replacement_rows"]),
            columns=int(row["replacement_columns"]),
        )
    )
    return ReplacementClaim(
        operation_key=str(row["operation_key"]),
        invalid_path=str(row["invalid_path"]),
        replacement=replacement,
        reason=str(row["reason"]),
        status=str(row["status"]),
        created_at=float(row["created_at"]),
        acknowledged_at=(
            None
            if row["acknowledged_at"] is None
            else float(row["acknowledged_at"])
        ),
        terminal_at=(
            None
            if row["terminal_at"] is None
            else float(row["terminal_at"])
        ),
        successor_operation_key=(
            None
            if row["successor_operation_key"] is None
            else str(row["successor_operation_key"])
        ),
    )


def _iter_candidate_jsonl(path: Path) -> Iterator[TableCandidate]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            yield TableCandidate(
                schema_class=str(record["schema_class"]),
                subset=str(record["subset"]),
                host=str(record["host"]),
                relative_path=str(record["relative_path"]),
                rows=int(record["rows"]),
                columns=int(record["columns"]),
            )


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _connect_sqlite_read_write(
    path: Path,
    *,
    timeout: float = 5.0,
) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(path)
    return sqlite3.connect(
        path.resolve().as_uri() + "?mode=rw",
        uri=True,
        timeout=timeout,
    )


def replace_invalid_selection(
    invalid_candidate: TableCandidate,
    reserve: ReserveManager,
    *,
    operation_key: str,
    reason: str,
    is_invalid: Callable[[TableCandidate], bool] | None = None,
) -> TableCandidate:
    """Replace an invalid active candidate through persistent reserve state."""
    if not isinstance(reserve, ReserveManager):
        raise TypeError("reserve must be a ReserveManager")
    return reserve.replace(
        invalid_candidate,
        operation_key=operation_key,
        reason=reason,
        is_invalid=is_invalid,
    )


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
    canonical_path = selection_dir / "catalog-canonical.jsonl"
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
            canonical_path,
            key_fn=_record_canonical_key,
            chunk_records=sort_chunk_records,
        )
        unique = AtomicJsonlShard(unique_path)
        try:
            previous_path: str | None = None
            with canonical_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    relative_path = str(record["relative_path"])
                    if relative_path == previous_path:
                        continue
                    unique.write(record)
                    previous_path = relative_path
            unique.commit()
        except BaseException:
            unique.abort()
            raise
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
        canonical_path.unlink(missing_ok=True)
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
