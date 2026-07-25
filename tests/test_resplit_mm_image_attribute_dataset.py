import sys
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_select_table_groups_hits_exact_label_target_without_splitting():
    from resplit_mm_image_attribute_dataset import select_table_groups

    groups = {
        "a": [{"extractable": True}, {"extractable": True}],
        "b": [{"extractable": True}],
        "c": [{"extractable": False}],
        "d": [{"extractable": True}, {"extractable": False}],
    }

    selected = select_table_groups(groups, positive_target=2, negative_target=1)

    assert sum(row["extractable"] for key in selected for row in groups[key]) == 2
    assert sum(not row["extractable"] for key in selected for row in groups[key]) == 1


def test_select_table_groups_rejects_impossible_target():
    from resplit_mm_image_attribute_dataset import select_table_groups

    with pytest.raises(ValueError, match="cannot satisfy"):
        select_table_groups(
            {"a": [{"extractable": True}, {"extractable": True}]},
            positive_target=1,
            negative_target=0,
        )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_resplit_dataset_preserves_records_and_keeps_tables_whole(tmp_path):
    from resplit_mm_image_attribute_dataset import resplit_dataset

    dataset_dir = tmp_path / "dataset"
    specs = {
        "a": [True],
        "b": [False],
        "c": [True],
        "d": [False],
        "e": [True, True, True],
        "f": [False, False],
    }
    samples = []
    tables = []
    for table_id, labels in specs.items():
        tables.append({"source_table_id": table_id, "rows": []})
        for index, label in enumerate(labels):
            sample_id = f"{table_id}-{index}"
            samples.append(
                {
                    "sample_id": sample_id,
                    "source_table_id": table_id,
                    "entity_id": f"entity-{sample_id}",
                    "extractable": label,
                    "image_path": f"images/{sample_id}.jpg",
                }
            )
            image = dataset_dir / "images" / f"{sample_id}.jpg"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"image")

    _write_jsonl(dataset_dir / "samples" / "train.jsonl", samples)
    _write_jsonl(dataset_dir / "samples" / "val.jsonl", [])
    _write_jsonl(dataset_dir / "samples" / "test.jsonl", [])
    _write_jsonl(dataset_dir / "tables" / "train.jsonl", tables)
    _write_jsonl(dataset_dir / "tables" / "val.jsonl", [])
    _write_jsonl(dataset_dir / "tables" / "test.jsonl", [])
    (dataset_dir / "stats.json").write_text("{}\n", encoding="utf-8")

    before_sample_ids = {row["sample_id"] for row in samples}
    before_table_ids = set(specs)
    before_images = {path.name for path in (dataset_dir / "images").iterdir()}

    report = resplit_dataset(
        dataset_dir,
        targets={
            "train": (3, 2),
            "val": (1, 1),
            "test": (1, 1),
        },
    )

    split_samples = {
        split: [
            json.loads(line)
            for line in (dataset_dir / "samples" / f"{split}.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        for split in ("train", "val", "test")
    }
    assert report["counts"] == {
        "train": {"positive": 3, "negative": 2, "total": 5},
        "val": {"positive": 1, "negative": 1, "total": 2},
        "test": {"positive": 1, "negative": 1, "total": 2},
    }
    assert {row["sample_id"] for rows in split_samples.values() for row in rows} == before_sample_ids
    assert {
        json.loads(line)["source_table_id"]
        for split in ("train", "val", "test")
        for line in (dataset_dir / "tables" / f"{split}.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    } == before_table_ids
    table_sets = {
        split: {row["source_table_id"] for row in rows}
        for split, rows in split_samples.items()
    }
    assert not table_sets["train"] & table_sets["val"]
    assert not table_sets["train"] & table_sets["test"]
    assert not table_sets["val"] & table_sets["test"]
    assert {path.name for path in (dataset_dir / "images").iterdir()} == before_images
