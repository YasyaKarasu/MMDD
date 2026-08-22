from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import wdc200k_selection as selection_module  # noqa: E402
from wdc200k_selection import (  # noqa: E402
    ReserveExhaustedError,
    ReserveManager,
    SelectionPolicy,
    TableCandidate,
    allocate_strata,
    read_statistics_catalog,
    replace_invalid_selection,
    run_selection,
    select_tables,
)


def make_statistics_zip(
    root: Path,
    schema_class: str,
    subsets: dict[str, list[tuple[str, int, int]]],
) -> Path:
    class_dir = root / schema_class
    class_dir.mkdir(parents=True)
    archive = class_dir / f"{schema_class}_statistics.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for subset, records in subsets.items():
            member = (
                f"table_statistics/{schema_class}_October2023"
                f"_statistics_{subset}.csv"
            )
            rows = ["host,number_of_rows,column_count\n"]
            for host, row_count, column_count in records:
                rows.append(f"{host},{row_count},{column_count}\n")
                (class_dir / (
                    f"{schema_class}_{host}_October2023.json.gz"
                )).touch()
            zipped.writestr(member, "".join(rows))
    return archive


def test_statistics_zip_restores_subset_and_filename(tmp_path: Path) -> None:
    archive = make_statistics_zip(
        tmp_path,
        schema_class="Product",
        subsets={"top100": [("shop.test", 10, 4)]},
    )

    records = list(read_statistics_catalog(archive))

    assert records == [
        TableCandidate(
            schema_class="Product",
            subset="top100",
            host="shop.test",
            relative_path="Product/Product_shop.test_October2023.json.gz",
            rows=10,
            columns=4,
        )
    ]


def test_statistics_catalog_keeps_missing_gzip_as_provisional(
    tmp_path: Path,
) -> None:
    archive = make_statistics_zip(
        tmp_path,
        schema_class="Product",
        subsets={"top100": [("missing.test", 10, 4)]},
    )
    (
        archive.parent / "Product_missing.test_October2023.json.gz"
    ).unlink()

    records = list(read_statistics_catalog(archive))

    assert [record.relative_path for record in records] == [
        "Product/Product_missing.test_October2023.json.gz"
    ]
    assert "exists" not in (read_statistics_catalog.__doc__ or "")


def test_selection_module_imports_through_scripts_package() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from scripts.wdc200k_selection import SelectionPolicy; "
            "assert SelectionPolicy().seed == 13",
        ],
        cwd=SCRIPTS_DIR.parent,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def synthetic_catalog(
    *,
    classes: int,
    top100_per_class: int,
    minimum3_per_class: int,
    rest_per_class: int,
) -> list[TableCandidate]:
    records: list[TableCandidate] = []
    counts = {
        "top100": top100_per_class,
        "minimum3": minimum3_per_class,
        "rest": rest_per_class,
    }
    for class_index in range(classes):
        schema_class = f"Class{class_index:02d}"
        for subset, count in counts.items():
            for index in range(count):
                host = f"{subset}-{index:05d}.class{class_index:02d}.test"
                records.append(
                    TableCandidate(
                        schema_class=schema_class,
                        subset=subset,
                        host=host,
                        relative_path=(
                            f"{schema_class}/{schema_class}_{host}"
                            "_October2023.json.gz"
                        ),
                        rows=index + 1,
                        columns=3,
                    )
                )
    return records


def test_allocate_strata_spills_shortfall_and_obeys_class_cap() -> None:
    catalog = synthetic_catalog(
        classes=2,
        top100_per_class=1,
        minimum3_per_class=2,
        rest_per_class=10,
    )
    policy = SelectionPolicy(
        target_tables=10,
        minimum3_fraction=0.75,
        minimum3_base_per_class=1,
        rest_base_per_class=1,
        class_cap=6,
    )

    quotas = allocate_strata(catalog, policy)

    assert sum(quotas.values()) == 10
    assert sum(
        quota
        for (schema_class, _subset), quota in quotas.items()
        if schema_class == "Class00"
    ) <= 6
    assert sum(
        quota
        for (_schema_class, subset), quota in quotas.items()
        if subset == "minimum3"
    ) == 4


def test_allocate_strata_applies_exact_split_and_per_class_bases() -> None:
    catalog = synthetic_catalog(
        classes=3,
        top100_per_class=0,
        minimum3_per_class=2_000,
        rest_per_class=2_000,
    )
    quotas = allocate_strata(
        catalog,
        SelectionPolicy(target_tables=2_000),
    )

    assert sum(
        quota for (_schema_class, subset), quota in quotas.items()
        if subset == "minimum3"
    ) == 1_800
    assert sum(
        quota for (_schema_class, subset), quota in quotas.items()
        if subset == "rest"
    ) == 200
    assert all(
        quotas[(f"Class{index:02d}", "minimum3")] >= 250
        for index in range(3)
    )
    assert all(
        quotas[(f"Class{index:02d}", "rest")] >= 50
        for index in range(3)
    )


def test_allocate_strata_spills_unfilled_rest_to_minimum3_first() -> None:
    catalog = [
        *synthetic_catalog(
            classes=1,
            top100_per_class=0,
            minimum3_per_class=20,
            rest_per_class=1,
        )
    ]
    quotas = allocate_strata(
        catalog,
        SelectionPolicy(
            target_tables=10,
            minimum3_fraction=0.50,
            minimum3_base_per_class=0,
            rest_base_per_class=0,
        ),
    )

    assert quotas[("Class00", "minimum3")] == 9
    assert quotas[("Class00", "rest")] == 1


def test_allocate_strata_rejects_insufficient_capacity() -> None:
    catalog = synthetic_catalog(
        classes=1,
        top100_per_class=1,
        minimum3_per_class=1,
        rest_per_class=1,
    )

    with pytest.raises(ValueError, match="only 3 feasible"):
        allocate_strata(
            catalog,
            SelectionPolicy(target_tables=4),
        )


def test_allocate_strata_rejects_top100_above_target_or_cap() -> None:
    catalog = synthetic_catalog(
        classes=1,
        top100_per_class=3,
        minimum3_per_class=0,
        rest_per_class=0,
    )

    with pytest.raises(ValueError, match="exceed target_tables"):
        allocate_strata(
            catalog,
            SelectionPolicy(target_tables=2),
        )
    with pytest.raises(ValueError, match="exceed class cap"):
        allocate_strata(
            catalog,
            SelectionPolicy(target_tables=3, class_cap=2),
        )


def test_mixed_allocation_is_exact_capped_and_reproducible() -> None:
    catalog = synthetic_catalog(
        classes=42,
        top100_per_class=100,
        minimum3_per_class=10_000,
        rest_per_class=10_000,
    )

    first = select_tables(
        catalog,
        SelectionPolicy(target_tables=200_000, seed=13),
    )
    second = select_tables(
        reversed(catalog),
        SelectionPolicy(target_tables=200_000, seed=13),
    )

    assert [item.relative_path for item in first.selected] == [
        item.relative_path for item in second.selected
    ]
    assert len(first.selected) == 200_000
    assert (
        max(Counter(item.schema_class for item in first.selected).values())
        <= 40_000
    )
    selected_paths = {item.relative_path for item in first.selected}
    assert all(
        item.relative_path in selected_paths
        for item in catalog
        if item.subset == "top100"
    )


def test_cost_aware_policy_bounds_top100_and_excludes_rest_from_reserve() -> None:
    catalog = synthetic_catalog(
        classes=2,
        top100_per_class=20,
        minimum3_per_class=30,
        rest_per_class=30,
    )
    policy = SelectionPolicy(
        target_tables=20,
        top100_policy="bounded",
        top100_per_class=3,
        include_rest=False,
        min_candidate_rows=5,
        min_candidate_columns=3,
        minimum3_fraction=1.0,
        rest_base_per_class=0,
    )

    result = select_tables(catalog, policy)

    selected_top100 = [
        item for item in result.selected if item.subset == "top100"
    ]
    assert len(selected_top100) == 6
    assert Counter(item.schema_class for item in selected_top100) == {
        "Class00": 3,
        "Class01": 3,
    }
    assert not any(item.subset == "rest" for item in result.selected)
    assert not any(item.subset == "rest" for item in result.reserve)
    assert not any(item.subset == "top100" for item in result.reserve)
    assert all(item.rows >= 5 and item.columns >= 3 for item in result.reserve)


def test_streaming_selection_never_persists_excluded_candidates(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    make_statistics_zip(
        input_dir,
        schema_class="Product",
        subsets={
            "top100": [
                (f"top-{index}.test", 100 + index, 4)
                for index in range(5)
            ],
            "minimum3": [
                (f"minimum-{index}.test", 10 + index, 4)
                for index in range(10)
            ],
            "rest": [
                (f"rest-{index}.test", 10 + index, 4)
                for index in range(10)
            ],
        },
    )
    work_dir = tmp_path / "work"
    policy = SelectionPolicy(
        target_tables=4,
        top100_policy="bounded",
        top100_per_class=2,
        include_rest=False,
        min_candidate_rows=5,
        min_candidate_columns=3,
        minimum3_fraction=1.0,
        rest_base_per_class=0,
    )

    run_selection(input_dir, work_dir, policy, sort_chunk_records=3)

    selection_root = work_dir / "selection"
    selected = [
        json.loads(line)
        for line in (selection_root / "selected_tables.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    reserve = [
        json.loads(line)
        for line in (selection_root / "reserve_tables.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert sum(record["subset"] == "top100" for record in selected) == 2
    assert {record["subset"] for record in reserve} == {"minimum3"}
    assert len(reserve) == 8


def test_selection_progress_is_monotonic_and_names_long_phases(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    for schema_class in ("Event", "Product"):
        make_statistics_zip(
            input_dir,
            schema_class=schema_class,
            subsets={
                "minimum3": [
                    (f"{schema_class.lower()}-{index}.test", 10, 4)
                    for index in range(4)
                ]
            },
        )
    updates: list[dict[str, object]] = []

    run_selection(
        input_dir,
        tmp_path / "work",
        SelectionPolicy(
            target_tables=4,
            top100_policy="exclude",
            include_rest=False,
            min_candidate_rows=1,
            min_candidate_columns=1,
            minimum3_fraction=1.0,
            rest_base_per_class=0,
        ),
        sort_chunk_records=2,
        progress_callback=updates.append,
    )

    assert all(
        0 <= int(update["completed"]) <= int(update["total"])
        for update in updates
    )
    assert int(updates[-1]["completed"]) == int(updates[-1]["total"])
    assert {str(update["phase"]) for update in updates} >= {
        "scan_statistics_archives",
        "canonicalize_candidates_read_records",
        "canonicalize_candidates_write_records",
        "deduplicate_candidates",
        "apply_candidate_policy",
        "allocate_candidate_quotas",
        "rank_candidates_read_records",
        "rank_candidates_write_records",
        "write_selection_outputs",
    }


def test_conflicting_duplicate_path_is_canonical_across_input_order() -> None:
    first_record = TableCandidate(
        schema_class="Product",
        subset="minimum3",
        host="duplicate.test",
        relative_path="Product/Product_duplicate.test_October2023.json.gz",
        rows=2,
        columns=4,
    )
    canonical_record = TableCandidate(
        schema_class="Product",
        subset="minimum3",
        host="duplicate.test",
        relative_path="Product/Product_duplicate.test_October2023.json.gz",
        rows=10,
        columns=5,
    )
    other = TableCandidate(
        schema_class="Product",
        subset="rest",
        host="other.test",
        relative_path="Product/Product_other.test_October2023.json.gz",
        rows=3,
        columns=2,
    )
    policy = SelectionPolicy(target_tables=2)

    forward = select_tables(
        [first_record, canonical_record, other],
        policy,
    )
    backward = select_tables(
        [other, canonical_record, first_record],
        policy,
    )

    assert forward == backward
    assert first_record in forward.selected
    assert canonical_record not in forward.selected


def test_replacement_prefers_same_stratum_then_global_reserve(
    tmp_path: Path,
) -> None:
    invalid = TableCandidate(
        schema_class="Product",
        subset="minimum3",
        host="bad.test",
        relative_path="Product/Product_bad.test_October2023.json.gz",
        rows=1,
        columns=3,
    )
    earlier_global = TableCandidate(
        schema_class="Event",
        subset="rest",
        host="global.test",
        relative_path="Event/Event_global.test_October2023.json.gz",
        rows=2,
        columns=3,
    )
    invalid_same = TableCandidate(
        schema_class="Product",
        subset="minimum3",
        host="also-bad.test",
        relative_path=(
            "Product/Product_also-bad.test_October2023.json.gz"
        ),
        rows=2,
        columns=3,
    )
    valid_same = TableCandidate(
        schema_class="Product",
        subset="minimum3",
        host="reserve.test",
        relative_path="Product/Product_reserve.test_October2023.json.gz",
        rows=2,
        columns=3,
    )
    policy = SelectionPolicy(target_tables=1)
    first_manager = ReserveManager.create(
        tmp_path / "first.sqlite",
        reserve=[earlier_global, invalid_same, valid_same],
        selected=[invalid],
        policy=policy,
    )
    first = replace_invalid_selection(
        invalid,
        first_manager,
        operation_key="replace-bad",
        reason="malformed gzip",
        is_invalid=lambda item: item.host == "also-bad.test",
    )
    second_manager = ReserveManager.create(
        tmp_path / "second.sqlite",
        reserve=[earlier_global],
        selected=[invalid],
        policy=policy,
    )
    second = replace_invalid_selection(
        invalid,
        second_manager,
        operation_key="replace-bad-global",
        reason="malformed gzip",
        is_invalid=lambda _item: False,
    )

    assert first == valid_same
    assert second == earlier_global


def test_replacement_global_fallback_skips_class_at_cap(
    tmp_path: Path,
) -> None:
    invalid = TableCandidate(
        "Event",
        "minimum3",
        "invalid.test",
        "Event/Event_invalid.test_October2023.json.gz",
        1,
        3,
    )
    selected_products = [
        TableCandidate(
            "Product",
            "minimum3",
            f"selected-{index}.test",
            f"Product/Product_selected-{index}.test_October2023.json.gz",
            1,
            3,
        )
        for index in range(2)
    ]
    capped_product = TableCandidate(
        "Product",
        "rest",
        "reserve.test",
        "Product/Product_reserve.test_October2023.json.gz",
        1,
        3,
    )
    feasible_event = TableCandidate(
        "Event",
        "rest",
        "reserve.test",
        "Event/Event_reserve.test_October2023.json.gz",
        1,
        3,
    )
    policy = SelectionPolicy(target_tables=3, class_cap=2)
    manager = ReserveManager.create(
        tmp_path / "reserve.sqlite",
        reserve=[capped_product, feasible_event],
        selected=[*selected_products, invalid],
        policy=policy,
    )

    replacement = replace_invalid_selection(
        invalid,
        manager,
        operation_key="replace-event",
        reason="unreadable gzip",
    )

    assert replacement == feasible_event
    assert manager.class_counts() == {"Event": 1, "Product": 2}


def test_replacement_raises_when_no_candidate_fits_class_cap(
    tmp_path: Path,
) -> None:
    invalid = TableCandidate(
        "Event",
        "minimum3",
        "invalid.test",
        "Event/Event_invalid.test_October2023.json.gz",
        1,
        3,
    )
    selected_product = TableCandidate(
        "Product",
        "minimum3",
        "selected.test",
        "Product/Product_selected.test_October2023.json.gz",
        1,
        3,
    )
    reserve_product = TableCandidate(
        "Product",
        "rest",
        "reserve.test",
        "Product/Product_reserve.test_October2023.json.gz",
        1,
        3,
    )
    policy = SelectionPolicy(target_tables=2, class_cap=1)
    manager = ReserveManager.create(
        tmp_path / "reserve.sqlite",
        reserve=[reserve_product],
        selected=[selected_product, invalid],
        policy=policy,
    )

    with pytest.raises(
        ReserveExhaustedError,
        match="no reserve candidate can replace",
    ):
        replace_invalid_selection(
            invalid,
            manager,
            operation_key="replace-exhausted",
            reason="unreadable gzip",
        )

    assert manager.class_counts() == {"Event": 0, "Product": 1}


def test_reserve_manager_consumes_consecutively_and_resumes(
    tmp_path: Path,
) -> None:
    invalid_product = TableCandidate(
        "Product",
        "minimum3",
        "invalid-product.test",
        "Product/Product_invalid-product.test_October2023.json.gz",
        1,
        3,
    )
    invalid_event = TableCandidate(
        "Event",
        "minimum3",
        "invalid-event.test",
        "Event/Event_invalid-event.test_October2023.json.gz",
        1,
        3,
    )
    product_reserves = [
        TableCandidate(
            "Product",
            "minimum3",
            f"reserve-{index}.test",
            f"Product/Product_reserve-{index}.test_October2023.json.gz",
            1,
            3,
        )
        for index in range(2)
    ]
    event_reserve = TableCandidate(
        "Event",
        "minimum3",
        "reserve.test",
        "Event/Event_reserve.test_October2023.json.gz",
        1,
        3,
    )
    policy = SelectionPolicy(target_tables=2, class_cap=4)
    database = tmp_path / "reserve.sqlite"
    manager = ReserveManager.create(
        database,
        reserve=[
            product_reserves[0],
            event_reserve,
            product_reserves[1],
        ],
        selected=[invalid_product, invalid_event],
        policy=policy,
    )

    first = replace_invalid_selection(
        invalid_product,
        manager,
        operation_key="replace-product-1",
        reason="invalid table",
    )
    second = replace_invalid_selection(
        invalid_event,
        manager,
        operation_key="replace-event-1",
        reason="invalid table",
    )
    resumed = ReserveManager.open(database, policy)
    third = replace_invalid_selection(
        first,
        resumed,
        operation_key="replace-product-2",
        reason="invalid replacement",
    )

    assert first == product_reserves[0]
    assert second == event_reserve
    assert third == product_reserves[1]
    assert resumed.used_paths() == {
        invalid_product.relative_path,
        invalid_event.relative_path,
        product_reserves[0].relative_path,
        product_reserves[1].relative_path,
        event_reserve.relative_path,
    }


def test_reserve_manager_streams_selection_jsonl_from_cli_schema(
    tmp_path: Path,
) -> None:
    invalid = TableCandidate(
        "Product",
        "minimum3",
        "invalid.test",
        "Product/Product_invalid.test_October2023.json.gz",
        1,
        3,
    )
    replacement = TableCandidate(
        "Product",
        "minimum3",
        "replacement.test",
        "Product/Product_replacement.test_October2023.json.gz",
        2,
        4,
    )
    selected_path = tmp_path / "selected_tables.jsonl"
    reserve_path = tmp_path / "reserve_tables.jsonl"
    selected_path.write_text(
        json.dumps(
            {
                **invalid.__dict__,
                "rank": "001",
                "selection_seed": 13,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    reserve_path.write_text(
        json.dumps(
            {
                **replacement.__dict__,
                "rank": "002",
                "selection_seed": 13,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    policy = SelectionPolicy(target_tables=1)

    manager = ReserveManager.create_from_jsonl(
        tmp_path / "reserve.sqlite",
        reserve_path=reserve_path,
        selected_path=selected_path,
        policy=policy,
    )

    assert (
        replace_invalid_selection(
            invalid,
            manager,
            operation_key="replace-jsonl",
            reason="invalid table",
        )
        == replacement
    )


def test_reserve_database_guard_failure_preserves_selection_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    work_dir = tmp_path / "work"
    make_statistics_zip(
        input_dir,
        schema_class="Product",
        subsets={
            "minimum3": [
                (f"host-{index}.test", index + 1, 3)
                for index in range(8)
            ]
        },
    )
    policy = SelectionPolicy(target_tables=2)
    run_selection(
        input_dir,
        work_dir,
        policy,
        sort_chunk_records=2,
    )
    selection_dir = work_dir / "selection"
    checkpoint_paths = [
        selection_dir / "manifest.json",
        selection_dir / "selected_tables.jsonl",
        selection_dir / "reserve_tables.jsonl",
    ]
    checkpoint_bytes = {
        path: path.read_bytes() for path in checkpoint_paths
    }
    monkeypatch.setattr(
        selection_module,
        "_RESERVE_CREATE_BATCH_RECORDS",
        2,
        raising=False,
    )
    monkeypatch.setattr(
        selection_module.GuardedWriteTracker,
        "DEFAULT_INTERVAL_BYTES",
        16 * 1024,
    )
    calls: list[tuple[Path, int]] = []
    commit_checks = 0

    def guard(path: Path, estimated_bytes: int = 0) -> None:
        nonlocal commit_checks
        calls.append((path, estimated_bytes))
        if estimated_bytes == 0:
            commit_checks += 1
            if commit_checks == 3:
                raise OSError("reserve exhausted mid-batch")

    database_path = selection_dir / "reserve.sqlite3"
    with pytest.raises(OSError, match="reserve exhausted mid-batch"):
        ReserveManager.create_from_jsonl(
            database_path,
            reserve_path=selection_dir / "reserve_tables.jsonl",
            selected_path=selection_dir / "selected_tables.jsonl",
            policy=policy,
            pre_write_guard=guard,
        )

    assert not database_path.exists()
    assert {
        path: path.read_bytes() for path in checkpoint_paths
    } == checkpoint_bytes
    assert not list(selection_dir.glob(".reserve.sqlite3.*.tmp"))
    guarded_paths = {path for path, _estimated in calls}
    assert len(guarded_paths) == 1
    guarded_path = guarded_paths.pop()
    assert guarded_path.parent == selection_dir
    assert guarded_path.name.startswith(".reserve.sqlite3.")
    assert guarded_path.name.endswith(".tmp")
    positive_checks = [size for _path, size in calls if size > 0]
    assert 0 < len(positive_checks) < 4


def test_replacement_claim_guard_failure_rolls_back_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = TableCandidate(
        "Product",
        "minimum3",
        "invalid.test",
        "Product/Product_invalid.test_October2023.json.gz",
        1,
        3,
    )
    replacement = TableCandidate(
        "Product",
        "minimum3",
        "replacement.test",
        "Product/Product_replacement.test_October2023.json.gz",
        2,
        4,
    )
    policy = SelectionPolicy(target_tables=1)
    database = tmp_path / "reserve.sqlite3"
    ReserveManager.create(
        database,
        reserve=[replacement],
        selected=[invalid],
        policy=policy,
    )
    monkeypatch.setattr(
        selection_module.GuardedWriteTracker,
        "DEFAULT_INTERVAL_BYTES",
        1,
    )

    zero_checks = 0

    def reject_commit(_path: Path, estimated_bytes: int = 0) -> None:
        nonlocal zero_checks
        if estimated_bytes == 0:
            zero_checks += 1
            if zero_checks == 2:
                raise OSError(
                    "replacement claim commit reserve exhausted"
                )

    guarded = ReserveManager.open(
        database,
        policy,
        pre_write_guard=reject_commit,
    )
    with pytest.raises(OSError, match="claim commit reserve"):
        guarded.claim_replacement(
            operation_key="replace-one",
            invalid_candidate=invalid,
            reason="invalid gzip",
        )

    resumed = ReserveManager.open(database, policy)
    assert resumed.pending_claims() == []
    assert replacement.relative_path not in resumed.used_paths()
    claim = resumed.claim_replacement(
        operation_key="replace-one",
        invalid_candidate=invalid,
        reason="invalid gzip",
    )
    assert claim.replacement == replacement


def test_replacement_ack_guard_failure_preserves_pending_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = TableCandidate(
        "Product",
        "minimum3",
        "invalid.test",
        "Product/Product_invalid.test_October2023.json.gz",
        1,
        3,
    )
    replacement = TableCandidate(
        "Product",
        "minimum3",
        "replacement.test",
        "Product/Product_replacement.test_October2023.json.gz",
        2,
        4,
    )
    policy = SelectionPolicy(target_tables=1)
    database = tmp_path / "reserve.sqlite3"
    manager = ReserveManager.create(
        database,
        reserve=[replacement],
        selected=[invalid],
        policy=policy,
    )
    claim = manager.claim_replacement(
        operation_key="replace-one",
        invalid_candidate=invalid,
        reason="invalid gzip",
    )
    monkeypatch.setattr(
        selection_module.GuardedWriteTracker,
        "DEFAULT_INTERVAL_BYTES",
        1,
    )

    zero_checks = 0

    def reject_commit(_path: Path, estimated_bytes: int = 0) -> None:
        nonlocal zero_checks
        if estimated_bytes == 0:
            zero_checks += 1
            if zero_checks == 2:
                raise OSError(
                    "replacement ack commit reserve exhausted"
                )

    guarded = ReserveManager.open(
        database,
        policy,
        pre_write_guard=reject_commit,
    )
    with pytest.raises(OSError, match="ack commit reserve"):
        guarded.acknowledge(
            operation_key=claim.operation_key,
            replacement_path=replacement.relative_path,
        )

    resumed = ReserveManager.open(database, policy)
    assert resumed.pending_claims() == [claim]
    assert resumed.acknowledge(
        operation_key=claim.operation_key,
        replacement_path=replacement.relative_path,
    ).status == "acked"


def test_reserve_manager_create_does_not_reuse_fixed_temporary_path(
    tmp_path: Path,
) -> None:
    database = tmp_path / "reserve.sqlite"
    legacy_temporary = database.with_suffix(database.suffix + ".tmp")
    legacy_temporary.write_text("owned by another creator", encoding="utf-8")
    selected = TableCandidate(
        "Product",
        "minimum3",
        "selected.test",
        "Product/Product_selected.test_October2023.json.gz",
        1,
        3,
    )
    policy = SelectionPolicy(target_tables=1)

    ReserveManager.create(
        database,
        reserve=[],
        selected=[selected],
        policy=policy,
    )

    assert legacy_temporary.read_text(encoding="utf-8") == (
        "owned by another creator"
    )


def test_reserve_manager_open_missing_does_not_create_database(
    tmp_path: Path,
) -> None:
    database = tmp_path / "missing.sqlite"

    with pytest.raises(FileNotFoundError):
        ReserveManager.open(
            database,
            SelectionPolicy(target_tables=1),
        )

    assert not database.exists()


def test_replacement_claim_survives_crash_and_replays_idempotently(
    tmp_path: Path,
) -> None:
    invalid = TableCandidate(
        "Product",
        "minimum3",
        "invalid.test",
        "Product/Product_invalid.test_October2023.json.gz",
        1,
        3,
    )
    reserves = [
        TableCandidate(
            "Product",
            "minimum3",
            f"reserve-{index}.test",
            f"Product/Product_reserve-{index}.test_October2023.json.gz",
            2,
            4,
        )
        for index in range(2)
    ]
    policy = SelectionPolicy(target_tables=1)
    database = tmp_path / "reserve.sqlite"
    manager = ReserveManager.create(
        database,
        reserve=reserves,
        selected=[invalid],
        policy=policy,
    )

    committed = manager.claim_replacement(
        operation_key="structural-shard-7-row-3",
        invalid_candidate=invalid,
        reason="gzip JSON decode failed",
    )
    counts_after_commit = manager.class_counts()
    used_after_commit = manager.used_paths()

    resumed = ReserveManager.open(database, policy)
    pending = resumed.pending_claims()
    replay_by_key = resumed.claim_replacement(
        operation_key="structural-shard-7-row-3",
        invalid_candidate=invalid,
        reason="gzip JSON decode failed",
    )
    replay_by_invalid = resumed.claim_replacement(
        operation_key="retry-with-new-process-key",
        invalid_candidate=invalid,
        reason="process restarted",
    )

    assert committed.status == "pending"
    assert committed.operation_key == "structural-shard-7-row-3"
    assert committed.invalid_path == invalid.relative_path
    assert committed.replacement == reserves[0]
    assert committed.reason == "gzip JSON decode failed"
    assert committed.created_at > 0
    assert committed.acknowledged_at is None
    assert pending == [committed]
    assert replay_by_key == committed
    assert replay_by_invalid == committed
    assert resumed.class_counts() == counts_after_commit
    assert resumed.used_paths() == used_after_commit
    assert reserves[1].relative_path not in resumed.used_paths()


def test_acknowledge_is_persistent_idempotent_and_validated(
    tmp_path: Path,
) -> None:
    invalid = TableCandidate(
        "Product",
        "minimum3",
        "invalid.test",
        "Product/Product_invalid.test_October2023.json.gz",
        1,
        3,
    )
    replacement = TableCandidate(
        "Product",
        "minimum3",
        "replacement.test",
        "Product/Product_replacement.test_October2023.json.gz",
        2,
        4,
    )
    policy = SelectionPolicy(target_tables=1)
    database = tmp_path / "reserve.sqlite"
    manager = ReserveManager.create(
        database,
        reserve=[replacement],
        selected=[invalid],
        policy=policy,
    )
    pending = manager.claim_replacement(
        operation_key="operation-1",
        invalid_candidate=invalid,
        reason="invalid table",
    )

    with pytest.raises(KeyError, match="unknown replacement operation"):
        manager.acknowledge(
            operation_key="missing-operation",
            replacement_path=replacement.relative_path,
        )
    with pytest.raises(ValueError, match="replacement path does not match"):
        manager.acknowledge(
            operation_key=pending.operation_key,
            replacement_path="Product/wrong.json.gz",
        )
    with pytest.raises(ValueError, match="operation key belongs"):
        manager.claim_replacement(
            operation_key=pending.operation_key,
            invalid_candidate=replacement,
            reason="different invalid path",
        )
    conflicting_invalid = TableCandidate(
        "Event",
        invalid.subset,
        invalid.host,
        invalid.relative_path,
        invalid.rows,
        invalid.columns,
    )
    with pytest.raises(ValueError, match="invalid candidate metadata"):
        manager.claim_replacement(
            operation_key="operation-with-conflicting-metadata",
            invalid_candidate=conflicting_invalid,
            reason="invalid table",
        )

    acknowledged = manager.acknowledge(
        operation_key=pending.operation_key,
        replacement_path=replacement.relative_path,
    )
    replayed_ack = manager.acknowledge(
        operation_key=pending.operation_key,
        replacement_path=replacement.relative_path,
    )
    resumed = ReserveManager.open(database, policy)

    assert acknowledged.status == "acked"
    assert acknowledged.acknowledged_at is not None
    assert replayed_ack == acknowledged
    assert resumed.pending_claims() == []
    assert resumed.claim_replacement(
        operation_key=pending.operation_key,
        invalid_candidate=invalid,
        reason="invalid table",
    ) == acknowledged


def test_replacement_chain_supersedes_predecessor_across_restart(
    tmp_path: Path,
) -> None:
    invalid = TableCandidate(
        "Product",
        "minimum3",
        "invalid.test",
        "Product/Product_invalid.test_October2023.json.gz",
        1,
        3,
    )
    reserves = [
        TableCandidate(
            "Product",
            "minimum3",
            f"reserve-{index}.test",
            f"Product/Product_reserve-{index}.test_October2023.json.gz",
            2,
            4,
        )
        for index in range(2)
    ]
    policy = SelectionPolicy(target_tables=1)
    database = tmp_path / "reserve.sqlite"
    manager = ReserveManager.create(
        database,
        reserve=reserves,
        selected=[invalid],
        policy=policy,
    )
    predecessor = manager.claim_replacement(
        operation_key="operation-0",
        invalid_candidate=invalid,
        reason="invalid selected table",
    )

    resumed = ReserveManager.open(database, policy)
    successor = resumed.claim_replacement(
        operation_key="operation-1",
        invalid_candidate=predecessor.replacement,
        reason="invalid replacement table",
    )
    reopened = ReserveManager.open(database, policy)
    replayed_predecessor = reopened.claim_replacement(
        operation_key="operation-0",
        invalid_candidate=invalid,
        reason="invalid selected table",
    )

    assert successor.status == "pending"
    assert successor.replacement == reserves[1]
    assert reopened.pending_claims() == [successor]
    assert replayed_predecessor.status == "superseded"
    assert replayed_predecessor.successor_operation_key == "operation-1"
    assert reopened.class_counts() == {"Product": 1}
    with pytest.raises(ValueError, match="cannot acknowledge superseded"):
        reopened.acknowledge(
            operation_key="operation-0",
            replacement_path=reserves[0].relative_path,
        )
    with pytest.raises(RuntimeError, match="superseded"):
        replace_invalid_selection(
            invalid,
            reopened,
            operation_key="operation-0",
            reason="invalid selected table",
        )


def test_exhausted_operation_is_terminal_and_replays_after_restart(
    tmp_path: Path,
) -> None:
    invalid = TableCandidate(
        "Product",
        "minimum3",
        "invalid.test",
        "Product/Product_invalid.test_October2023.json.gz",
        1,
        3,
    )
    policy = SelectionPolicy(target_tables=1)
    database = tmp_path / "reserve.sqlite"
    manager = ReserveManager.create(
        database,
        reserve=[],
        selected=[invalid],
        policy=policy,
    )

    with pytest.raises(ReserveExhaustedError) as first:
        manager.claim_replacement(
            operation_key="operation-exhausted",
            invalid_candidate=invalid,
            reason="unreadable gzip",
        )

    resumed = ReserveManager.open(database, policy)
    with pytest.raises(ReserveExhaustedError) as replayed:
        resumed.claim_replacement(
            operation_key="operation-exhausted",
            invalid_candidate=invalid,
            reason="unreadable gzip",
        )
    with pytest.raises(ReserveExhaustedError) as aliased:
        resumed.claim_replacement(
            operation_key="retry-after-restart",
            invalid_candidate=invalid,
            reason="worker restarted",
        )

    terminal = resumed.terminal_claims()
    assert str(first.value) == str(replayed.value) == str(aliased.value)
    assert first.value.claim == terminal[0]
    assert terminal[0].status == "exhausted"
    assert terminal[0].replacement is None
    assert resumed.pending_claims() == []
    assert resumed.class_counts() == {"Product": 0}
    with pytest.raises(ValueError, match="cannot acknowledge exhausted"):
        resumed.acknowledge(
            operation_key="operation-exhausted",
            replacement_path="Product/no-replacement.json.gz",
        )


def test_recovery_exhaustion_can_retain_the_active_slot(tmp_path: Path) -> None:
    active = TableCandidate(
        "Product",
        "minimum3",
        "active.test",
        "Product/Product_active.test_October2023.json.gz",
        7,
        4,
    )
    policy = SelectionPolicy(target_tables=1)
    database = tmp_path / "reserve.sqlite"
    manager = ReserveManager.create(
        database,
        reserve=[],
        selected=[active],
        policy=policy,
    )

    with pytest.raises(ReserveExhaustedError):
        manager.claim_replacement(
            operation_key="recovery-exhausted",
            invalid_candidate=active,
            reason="unrecoverable after auto-check",
            retain_on_exhaustion=True,
        )

    resumed = ReserveManager.open(database, policy)
    assert resumed.class_counts() == {"Product": 1}
    assert resumed.used_paths() == {active.relative_path}
    assert resumed.terminal_claims()[0].status == "exhausted"


def test_first_claim_requires_full_active_candidate_metadata(
    tmp_path: Path,
) -> None:
    active = TableCandidate(
        "Product",
        "minimum3",
        "active.test",
        "Product/Product_active.test_October2023.json.gz",
        7,
        4,
    )
    reserve = TableCandidate(
        "Product",
        "rest",
        "reserve.test",
        "Product/Product_reserve.test_October2023.json.gz",
        2,
        3,
    )
    forged = TableCandidate(
        active.schema_class,
        "rest",
        active.host,
        active.relative_path,
        active.rows,
        active.columns,
    )
    policy = SelectionPolicy(target_tables=1)
    database = tmp_path / "reserve.sqlite"
    manager = ReserveManager.create(
        database,
        reserve=[reserve],
        selected=[active],
        policy=policy,
    )

    with pytest.raises(ValueError, match="active candidate metadata"):
        manager.claim_replacement(
            operation_key="forged-operation",
            invalid_candidate=forged,
            reason="forged subset",
        )

    reopened = ReserveManager.open(database, policy)
    with pytest.raises(ValueError, match="active candidate metadata"):
        reopened.claim_replacement(
            operation_key="forged-operation",
            invalid_candidate=forged,
            reason="forged subset",
        )
    assert reopened.class_counts() == {"Product": 1}
    assert reopened.used_paths() == {active.relative_path}
    assert reopened.pending_claims() == []


def test_selection_cli_writes_exact_stable_provisional_and_reserve(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    for schema_class in ("Event", "Product"):
        make_statistics_zip(
            input_dir,
            schema_class=schema_class,
            subsets={
                "top100": [
                    (f"top-{index}.{schema_class}.test", index + 1, 4)
                    for index in range(2)
                ],
                "minimum3": [
                    (f"minimum-{index}.{schema_class}.test", index + 1, 4)
                    for index in range(10)
                ],
                "rest": [
                    (f"rest-{index}.{schema_class}.test", index + 1, 4)
                    for index in range(10)
                ],
            },
        )
    work_dir = tmp_path / "work"
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "wdc200k_selection.py"),
        "--input_dir",
        str(input_dir),
        "--work_dir",
        str(work_dir),
        "--target_tables",
        "20",
        "--seed",
        "13",
        "--top100_policy",
        "all",
        "--include_rest",
        "--min_candidate_rows",
        "0",
        "--min_candidate_columns",
        "0",
        "--minimum3_fraction",
        "0.9",
        "--rest_base_per_class",
        "50",
    ]

    subprocess.run(command, check=True, capture_output=True, text=True)
    selected_path = work_dir / "selection" / "selected_tables.jsonl"
    reserve_path = work_dir / "selection" / "reserve_tables.jsonl"
    first_selected = selected_path.read_text(encoding="utf-8")
    first_reserve = reserve_path.read_text(encoding="utf-8")
    subprocess.run(command, check=True, capture_output=True, text=True)

    selected = [
        json.loads(line) for line in first_selected.splitlines()
    ]
    reserve = [json.loads(line) for line in first_reserve.splitlines()]
    assert len(selected) == 20
    assert len(reserve) == 24
    assert first_selected == selected_path.read_text(encoding="utf-8")
    assert first_reserve == reserve_path.read_text(encoding="utf-8")
    assert [item["rank"] for item in reserve] == sorted(
        item["rank"] for item in reserve
    )


def test_selection_cli_does_not_complete_when_capacity_is_insufficient(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    make_statistics_zip(
        input_dir,
        schema_class="Product",
        subsets={
            "top100": [("top.test", 10, 4)],
            "minimum3": [("minimum.test", 8, 4)],
        },
    )
    work_dir = tmp_path / "work"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "wdc200k_selection.py"),
            "--input_dir",
            str(input_dir),
            "--work_dir",
            str(work_dir),
            "--target_tables",
            "3",
        ],
        capture_output=True,
        text=True,
    )

    manifest = json.loads(
        (work_dir / "selection" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert result.returncode != 0
    assert "only 2 feasible" in result.stderr
    assert manifest["complete"] is False
    assert not (
        work_dir / "selection" / "selected_tables.jsonl"
    ).exists()


def test_selection_cli_deduplicates_paths_before_allocating(
    tmp_path: Path,
) -> None:
    duplicate_rows = [
        ("duplicate.test", 2, 4),
        ("duplicate.test", 10, 5),
        ("minimum.test", 8, 4),
    ]
    outputs = []
    for name, rows in (
        ("forward", duplicate_rows),
        ("backward", list(reversed(duplicate_rows))),
    ):
        input_dir = tmp_path / name / "input"
        make_statistics_zip(
            input_dir,
            schema_class="Product",
            subsets={
                "top100": [("top.test", 10, 4)],
                "minimum3": rows,
                "rest": [("rest.test", 6, 4)],
            },
        )
        work_dir = tmp_path / name / "work"
        subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "wdc200k_selection.py"),
                "--input_dir",
                str(input_dir),
                "--work_dir",
                str(work_dir),
                "--target_tables",
                "4",
                "--top100_policy",
                "all",
                "--include_rest",
                "--min_candidate_rows",
                "0",
                "--min_candidate_columns",
                "0",
                "--minimum3_fraction",
                "0.9",
                "--rest_base_per_class",
                "50",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(
            (
                work_dir / "selection" / "selected_tables.jsonl"
            ).read_text(encoding="utf-8")
        )

    selected = [json.loads(line) for line in outputs[0].splitlines()]
    assert outputs[0] == outputs[1]
    assert len(selected) == 4
    assert len({item["relative_path"] for item in selected}) == 4
    duplicate = next(
        item for item in selected
        if item["relative_path"].endswith("duplicate.test_October2023.json.gz")
    )
    assert (duplicate["rows"], duplicate["columns"]) == (2, 4)


def test_reserve_manager_reads_one_claim_by_operation_or_invalid_path(
    tmp_path: Path,
) -> None:
    selected = TableCandidate(
        schema_class="Product",
        subset="minimum3",
        host="bad.test",
        relative_path="Product/Product_bad.test_October2023.json.gz",
        rows=3,
        columns=2,
    )
    reserve = TableCandidate(
        schema_class="Product",
        subset="minimum3",
        host="good.test",
        relative_path="Product/Product_good.test_October2023.json.gz",
        rows=3,
        columns=2,
    )
    manager = ReserveManager.create(
        tmp_path / "reserve.sqlite3",
        reserve=[reserve],
        selected=[selected],
        policy=SelectionPolicy(target_tables=1),
    )
    claim = manager.claim_replacement(
        operation_key="replace-bad",
        invalid_candidate=selected,
        reason="broken",
    )

    assert manager.get_claim_by_operation("replace-bad") == claim
    assert (
        manager.get_claim_by_invalid_path(selected.relative_path)
        == claim
    )
    assert manager.get_claim_by_operation("missing") is None
    assert manager.get_claim_by_invalid_path("missing") is None
