from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from wdc200k_selection import (  # noqa: E402
    SelectionPolicy,
    TableCandidate,
    allocate_strata,
    read_statistics_catalog,
    replace_invalid_selection,
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


def test_replacement_prefers_same_stratum_then_global_reserve() -> None:
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
    reserve = [earlier_global, invalid_same, valid_same]

    first = replace_invalid_selection(
        invalid,
        reserve,
        is_invalid=lambda item: item.host == "also-bad.test",
    )
    second = replace_invalid_selection(
        invalid,
        reserve[:1],
        is_invalid=lambda _item: False,
    )

    assert first == valid_same
    assert second == earlier_global


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


def test_selection_cli_deduplicates_paths_before_allocating(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    make_statistics_zip(
        input_dir,
        schema_class="Product",
        subsets={
            "top100": [("duplicate.test", 10, 4)],
            "minimum3": [
                ("duplicate.test", 10, 4),
                ("minimum.test", 8, 4),
            ],
            "rest": [("rest.test", 6, 4)],
        },
    )
    work_dir = tmp_path / "work"

    subprocess.run(
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
        check=True,
        capture_output=True,
        text=True,
    )

    selected = [
        json.loads(line)
        for line in (
            work_dir / "selection" / "selected_tables.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert len(selected) == 3
    assert len({item["relative_path"] for item in selected}) == 3
