import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_old"))


def test_parse_args_uses_2k_defaults(tmp_path):
    from build_mm_image_attribute_dataset import parse_args

    args = parse_args(["--input_dir", str(tmp_path / "input")])

    assert args.output_dir == "output_mm_image_attribute_5k"
    assert (args.total_samples, args.positive_samples, args.negative_samples) == (2000, 1369, 631)
    assert (args.train_samples, args.val_samples, args.test_samples) == (1600, 200, 200)


def test_take_zero_returns_no_rows():
    from build_mm_image_attribute_dataset import _take

    row = {"source_table_id": "table", "entity_id": "entity"}
    assert _take([row], 0, seed=1, table_cap=20, entity_cap=1) == []


def test_public_sample_omits_asset_metadata_and_uses_sample_image_name():
    from build_mm_image_attribute_dataset import public_sample

    sample = {
        "sample_id": "sample-1",
        "asset_id": "asset_img_private",
        "entity_id": "entity-1",
        "ground_truth_value": "2008",
    }

    result = public_sample(sample, ".jpg")

    assert result == {
        "sample_id": "sample-1",
        "entity_id": "entity-1",
        "ground_truth_value": "2008",
        "image_path": "images/sample-1.jpg",
    }
