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

import create_wdc200k_scale_gate_input as gate_module  # noqa: E402
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


def _tree_snapshot(root: Path) -> list[tuple[str, str, str]]:
    snapshot = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            snapshot.append((relative, "symlink", os.readlink(path)))
        elif path.is_file():
            snapshot.append((relative, "file", _sha256(path)))
        else:
            snapshot.append((relative, "directory", ""))
    return snapshot


def _write_statistics_rows(
    root: Path,
    *,
    schema_class: str,
    subset: str,
    rows: list[tuple[str, int, int]],
) -> Path:
    class_dir = root / schema_class
    class_dir.mkdir(parents=True, exist_ok=True)
    archive = class_dir / f"{schema_class}_statistics.zip"
    member = (
        f"table_statistics/{schema_class}_October2023"
        f"_statistics_{subset}.csv"
    )
    text = ["host,number_of_rows,column_count\n"]
    text.extend(
        f"{host},{row_count},{column_count}\n"
        for host, row_count, column_count in rows
    )
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(member, "".join(text))
    return archive


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
    assert not (target / ".candidate-index.sqlite3").exists()
    checksums = json.loads(result.checksums_path.read_text(encoding="utf-8"))
    assert ".candidate-index.sqlite3" not in checksums["files"]


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


def test_scale_gate_input_rejects_traversal_statistics_before_publish(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-root" / "real-wdc"
    target = tmp_path / "target-root" / "gate"
    archive = _write_statistics_rows(
        source,
        schema_class="Product",
        subset="top100",
        rows=[("../../../../escape.test", 1, 3)],
    )
    candidate = next(read_statistics_catalog(archive))
    (source / "Product" / "Product_..").mkdir()
    escaped_source = (source / candidate.relative_path).resolve()
    escaped_source.parent.mkdir(parents=True, exist_ok=True)
    escaped_source.write_bytes(b"malicious-source")
    escaped_target = (target / candidate.relative_path).resolve()
    source_before = _tree_snapshot(source)

    with pytest.raises(ValueError, match="relative_path"):
        create_scale_gate_input(
            source_dir=source,
            target_dir=target,
            table_count=1,
        )

    assert not target.exists()
    assert not escaped_target.is_symlink()
    assert _tree_snapshot(source) == source_before


def test_scale_gate_input_rejects_duplicate_candidate_path_atomically(
    tmp_path: Path,
) -> None:
    source = tmp_path / "real-wdc"
    target = tmp_path / "gate"
    archive = _write_statistics_rows(
        source,
        schema_class="Product",
        subset="top100",
        rows=[("duplicate.test", 1, 3), ("duplicate.test", 1, 3)],
    )
    candidate = next(read_statistics_catalog(archive))
    source_path = source / candidate.relative_path
    source_path.write_bytes(b"source")
    source_before = _tree_snapshot(source)

    with pytest.raises(ValueError, match="duplicate"):
        create_scale_gate_input(
            source_dir=source,
            target_dir=target,
            table_count=1,
        )

    assert not target.exists()
    assert _tree_snapshot(source) == source_before


def test_scale_gate_input_link_failure_leaves_no_partial_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "real-wdc"
    target = tmp_path / "gate"
    _make_real_input(source)
    source_before = _tree_snapshot(source)
    real_symlink = gate_module.os.symlink
    calls = 0

    def fail_second_link(source_path: Path, target_path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected link failure")
        real_symlink(source_path, target_path)

    monkeypatch.setattr(gate_module.os, "symlink", fail_second_link)

    with pytest.raises(RuntimeError, match="injected link failure"):
        create_scale_gate_input(
            source_dir=source,
            target_dir=target,
            table_count=3,
        )

    assert not target.exists()
    assert not list(tmp_path.glob(".gate.*.tmp"))
    assert _tree_snapshot(source) == source_before


def test_scale_gate_input_zip_failure_preserves_precreated_empty_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "real-wdc"
    target = tmp_path / "gate"
    _make_real_input(source)
    target.mkdir()
    source_before = _tree_snapshot(source)

    def fail_archive(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected zip failure")

    monkeypatch.setattr(
        gate_module,
        "_write_statistics_archive",
        fail_archive,
    )

    with pytest.raises(RuntimeError, match="injected zip failure"):
        create_scale_gate_input(
            source_dir=source,
            target_dir=target,
            table_count=3,
        )

    assert target.is_dir()
    assert not list(target.iterdir())
    assert not list(tmp_path.glob(".gate.*.tmp"))
    assert _tree_snapshot(source) == source_before


def test_scale_gate_input_atomically_replaces_precreated_empty_target(
    tmp_path: Path,
) -> None:
    source = tmp_path / "real-wdc"
    target = tmp_path / "gate"
    _make_real_input(source)
    target.mkdir()

    result = create_scale_gate_input(
        source_dir=source,
        target_dir=target,
        table_count=3,
    )

    assert result.target_dir == target
    assert result.manifest_path.is_file()
    assert len(_read_manifest(result.manifest_path)) == 3


def test_scale_gate_input_does_not_overwrite_raced_nonempty_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "real-wdc"
    target = tmp_path / "gate"
    _make_real_input(source)
    target.mkdir()
    real_validate = gate_module._validate_staging

    def race_after_validation(*args: object, **kwargs: object) -> None:
        real_validate(*args, **kwargs)
        (target / "external-race.txt").write_text(
            "external",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        gate_module,
        "_validate_staging",
        race_after_validation,
    )

    with pytest.raises(ValueError, match="non-empty"):
        create_scale_gate_input(
            source_dir=source,
            target_dir=target,
            table_count=3,
        )

    assert (target / "external-race.txt").read_text() == "external"
    assert not list(tmp_path.glob(".gate.*.tmp"))


def test_scale_gate_input_manifest_failure_leaves_no_partial_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "real-wdc"
    target = tmp_path / "gate"
    _make_real_input(source)
    source_before = _tree_snapshot(source)

    def fail_manifest(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected manifest failure")

    monkeypatch.setattr(
        gate_module,
        "_write_manifest",
        fail_manifest,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="injected manifest failure"):
        create_scale_gate_input(
            source_dir=source,
            target_dir=target,
            table_count=3,
        )

    assert not target.exists()
    assert not list(tmp_path.glob(".gate.*.tmp"))
    assert _tree_snapshot(source) == source_before


def test_candidate_pool_retention_is_global_table_count_plus_buckets() -> None:
    pool = gate_module._BoundedCandidatePool(table_count=7)
    bucket_count = 80
    for bucket_index in range(bucket_count):
        schema_class = f"Class{bucket_index:03d}"
        for row_index in range(100):
            relative_path = (
                f"{schema_class}/{schema_class}_host-{row_index}.test"
                "_October2023.json.gz"
            )
            candidate = gate_module.TableCandidate(
                schema_class=schema_class,
                subset="top100",
                host=f"host-{row_index}.test",
                relative_path=relative_path,
                rows=row_index,
                columns=3,
            )
            pool.add(
                gate_module._RankedCandidate(
                    key=(row_index, relative_path, relative_path),
                    candidate=candidate,
                    source_path=Path("/") / relative_path,
                )
            )

    assert pool.bucket_count == bucket_count
    assert pool.retained_count <= 7 + bucket_count
