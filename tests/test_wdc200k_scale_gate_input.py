from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from create_wdc200k_scale_gate_input import create_scale_gate_input  # noqa: E402
from wdc200k_selection import read_statistics_catalog  # noqa: E402


def _make_real_input(
    root: Path,
    *,
    classes: tuple[str, ...] = ("Event", "Product"),
) -> None:
    for schema_class in classes:
        class_dir = root / schema_class
        class_dir.mkdir(parents=True)
        archive = class_dir / f"{schema_class}_statistics.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            for subset in ("top100", "minimum3", "rest"):
                member = (
                    f"table_statistics/{schema_class}_October2023"
                    f"_statistics_{subset}.csv"
                )
                rows = ["host,number_of_rows,column_count\n"]
                for row_count in (100, 1):
                    host = f"{subset}-{row_count}.{schema_class.lower()}.test"
                    rows.append(f"{host},{row_count},3\n")
                    (
                        class_dir
                        / f"{schema_class}_{host}_October2023.json.gz"
                    ).write_bytes(f"{schema_class}:{subset}:{row_count}".encode())
                zipped.writestr(member, "".join(rows))


def _read_manifest(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_scale_gate_input_round_robins_lowest_row_from_each_bucket(
    tmp_path: Path,
) -> None:
    source = tmp_path / "real-wdc"
    target = tmp_path / "gate"
    _make_real_input(source)

    result = create_scale_gate_input(
        source_dir=source,
        target_dir=target,
        table_count=6,
        seed=13,
    )

    records = _read_manifest(result.manifest_path)
    assert len(records) == 6
    assert {int(record["rows"]) for record in records} == {1}
    assert {
        (str(record["schema_class"]), str(record["subset"]))
        for record in records
    } == {
        (schema_class, subset)
        for schema_class in ("Event", "Product")
        for subset in ("top100", "minimum3", "rest")
    }
    for record in records:
        link = target / str(record["target"])
        assert link.is_symlink()
        assert link.resolve() == Path(str(record["source"]))

    filtered = []
    for schema_class in ("Event", "Product"):
        filtered.extend(
            read_statistics_catalog(
                target
                / schema_class
                / f"{schema_class}_statistics.zip"
            )
        )
    assert {candidate.relative_path for candidate in filtered} == {
        str(record["target"]) for record in records
    }


def test_scale_gate_input_rejects_broken_symlink_target(
    tmp_path: Path,
) -> None:
    source = tmp_path / "real-wdc"
    target = tmp_path / "gate"
    _make_real_input(source)
    os.symlink(tmp_path / "missing-target", target)

    with pytest.raises(ValueError, match="target"):
        create_scale_gate_input(
            source_dir=source,
            target_dir=target,
            table_count=1,
        )

    assert target.is_symlink()


def test_scale_gate_input_rejects_target_inside_source_before_writing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "real-wdc"
    target = source / "gate"
    _make_real_input(source)
    before = sorted(
        path.relative_to(source)
        for path in source.rglob("*")
    )

    with pytest.raises(ValueError, match="overlap"):
        create_scale_gate_input(
            source_dir=source,
            target_dir=target,
            table_count=1,
        )

    assert not target.exists()
    assert sorted(
        path.relative_to(source)
        for path in source.rglob("*")
    ) == before


def test_scale_gate_input_rejects_source_inside_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "outer"
    source = target / "real-wdc"
    _make_real_input(source)

    with pytest.raises(ValueError, match="overlap"):
        create_scale_gate_input(
            source_dir=source,
            target_dir=target,
            table_count=1,
        )


def test_scale_gate_input_rejects_target_equal_to_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "real-wdc"
    _make_real_input(source)

    with pytest.raises(ValueError, match="overlap"):
        create_scale_gate_input(
            source_dir=source,
            target_dir=source,
            table_count=1,
        )


def test_scale_gate_input_resolves_symlink_ancestor_before_overlap_check(
    tmp_path: Path,
) -> None:
    source = tmp_path / "real-wdc"
    _make_real_input(source)
    source_alias = tmp_path / "source-alias"
    source_alias.symlink_to(source, target_is_directory=True)
    target = source_alias / "gate"

    with pytest.raises(ValueError, match="overlap"):
        create_scale_gate_input(
            source_dir=source,
            target_dir=target,
            table_count=1,
        )

    assert not (source / "gate").exists()


def test_scale_gate_input_checksums_are_reproducible(
    tmp_path: Path,
) -> None:
    source = tmp_path / "real-wdc"
    _make_real_input(source)
    first = create_scale_gate_input(
        source_dir=source,
        target_dir=tmp_path / "gate-a",
        table_count=4,
    )
    second = create_scale_gate_input(
        source_dir=source,
        target_dir=tmp_path / "gate-b",
        table_count=4,
    )

    first_checksums = json.loads(
        first.checksums_path.read_text(encoding="utf-8")
    )
    second_checksums = json.loads(
        second.checksums_path.read_text(encoding="utf-8")
    )
    assert first_checksums["files"] == second_checksums["files"]
    for relative_path, expected in first_checksums["files"].items():
        assert _sha256(first.target_dir / relative_path) == expected


def test_scale_gate_input_cli_help_is_read_only_by_default() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "create_wdc200k_scale_gate_input.py"),
            "--help",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--source_dir" in completed.stdout
    assert "--target_dir" in completed.stdout
    assert "--table_count" in completed.stdout
