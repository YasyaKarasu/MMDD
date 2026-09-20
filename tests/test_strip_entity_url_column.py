"""The synthetic ``entity_url`` strip is a byte-exact projection repair."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_old"))

from strip_entity_url_column import strip_dataset  # noqa: E402
from mmdd_dataset.wdc_runtime import iter_dataset_artifact  # noqa: E402

ENTITY_URL = {
    "column_index": 1,
    "source_column_index": -1,
    "column_name": "entity_url",
}
TITLE = {"column_index": 0, "source_column_index": 0, "column_name": "Title"}


def write_dataset(
    tmp_path: Path,
    *,
    columns: list[dict],
    rows: list[dict],
    declared_records: int | None = None,
) -> Path:
    (tmp_path / "query_tables").mkdir()
    record = {"table_id": "query_abc", "role": "query", "columns": columns, "rows": rows}
    (tmp_path / "query_tables/part-00000.jsonl").write_text(
        json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (tmp_path / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "artifacts": {
                    "query_tables": {
                        "shards": [
                            {
                                "path": "query_tables/part-00000.jsonl",
                                "records": declared_records
                                if declared_records is not None
                                else len(rows),
                            }
                        ]
                    }
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return tmp_path


def read_records(dataset_dir: Path) -> list[dict]:
    path = dataset_dir / "query_tables/part-00000.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_strip_drops_only_the_trailing_column_and_keeps_the_query_id(
    tmp_path: Path,
) -> None:
    dataset = write_dataset(
        tmp_path,
        columns=[TITLE, ENTITY_URL],
        rows=[
            {
                "row_id": 0,
                "cells": [
                    {"column_name": "Title", "text": "Gas Chamber"},
                    {
                        "column_name": "entity_url",
                        "text": "https://en.wikipedia.org/wiki/wdc_deadbeef",
                        "synthetic": True,
                    },
                ],
            }
        ],
    )
    before = read_records(dataset)[0]

    report = strip_dataset(dataset)

    assert report["stripped_query_tables"] == 1
    assert report["already_absent_query_tables"] == 0
    assert Path(report["backup_dir"]).is_dir()
    after = list(iter_dataset_artifact(dataset, "query_tables"))[0]
    assert after["table_id"] == before["table_id"]
    assert after["columns"] == before["columns"][:-1]
    assert after["rows"][0]["cells"] == before["rows"][0]["cells"][:-1]
    assert "entity_url" not in json.dumps(after)


def test_the_backup_keeps_the_pre_repair_shard(tmp_path: Path) -> None:
    dataset = write_dataset(
        tmp_path,
        columns=[TITLE, ENTITY_URL],
        rows=[{"row_id": 0, "cells": [{"column_name": "Title"}, dict(ENTITY_URL)]}],
    )
    before = (dataset / "query_tables/part-00000.jsonl").read_bytes()

    report = strip_dataset(dataset)

    assert (Path(report["backup_dir"]) / "query_tables/part-00000.jsonl").read_bytes() == before
    assert (Path(report["backup_dir"]) / "dataset_manifest.json").is_file()


def test_an_already_stripped_dataset_is_left_alone(tmp_path: Path) -> None:
    dataset = write_dataset(
        tmp_path,
        columns=[TITLE],
        rows=[{"row_id": 0, "cells": [{"column_name": "Title"}]}],
    )
    before = (dataset / "query_tables/part-00000.jsonl").read_bytes()

    report = strip_dataset(dataset)

    assert report["stripped_query_tables"] == 0
    assert report["already_absent_query_tables"] == 1
    assert report["backup_dir"] == ""
    assert (dataset / "query_tables/part-00000.jsonl").read_bytes() == before


@pytest.mark.parametrize(
    ("columns", "rows", "message"),
    [
        # The synthetic column exists but is not the trailing one.
        (
            [ENTITY_URL, TITLE],
            [{"row_id": 0, "cells": [dict(ENTITY_URL), {"column_name": "Title"}]}],
            "is not the last column",
        ),
        # The trailing column is the synthetic one, but a row disagrees.
        (
            [TITLE, ENTITY_URL],
            [{"row_id": 0, "cells": [{"column_name": "Title"}]}],
            "does not end in a synthetic entity_url cell",
        ),
    ],
)
def test_a_dataset_of_an_unexpected_shape_is_not_written(
    tmp_path: Path, columns: list[dict], rows: list[dict], message: str
) -> None:
    dataset = write_dataset(tmp_path, columns=columns, rows=rows)
    before = (dataset / "query_tables/part-00000.jsonl").read_bytes()

    with pytest.raises(ValueError, match=message):
        strip_dataset(dataset)

    assert (dataset / "query_tables/part-00000.jsonl").read_bytes() == before


def test_a_record_count_that_disagrees_with_the_manifest_is_refused(
    tmp_path: Path,
) -> None:
    dataset = write_dataset(
        tmp_path,
        columns=[TITLE, ENTITY_URL],
        rows=[{"row_id": 0, "cells": [{"column_name": "Title"}, dict(ENTITY_URL)]}],
        declared_records=7,
    )

    with pytest.raises(ValueError, match="manifest says 7"):
        strip_dataset(dataset)


def test_a_dry_run_changes_nothing(tmp_path: Path) -> None:
    dataset = write_dataset(
        tmp_path,
        columns=[TITLE, ENTITY_URL],
        rows=[{"row_id": 0, "cells": [{"column_name": "Title"}, dict(ENTITY_URL)]}],
    )
    before = (dataset / "query_tables/part-00000.jsonl").read_bytes()

    report = strip_dataset(dataset, dry_run=True)

    assert report["stripped_query_tables"] == 1
    assert (dataset / "query_tables/part-00000.jsonl").read_bytes() == before
    assert not (dataset / "backups").exists()
