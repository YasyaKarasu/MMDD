import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_mm_table_dataset as mm_table_dataset


@pytest.mark.parametrize(
    "target",
    [
        "//commons.wikimedia.org/w/index.php?title=Special:UploadWizard",
        "http://example.test/entity",
        "HTTPS://example.test/entity",
    ],
)
def test_parse_wiki_cell_rejects_external_link_targets(target):
    parsed = mm_table_dataset.parse_wiki_cell(f"[{target}|External label]")

    assert parsed["text"] == "External label"
    assert parsed["wiki_title"] is None
    assert parsed["has_wiki_link"] is True


def test_parse_wiki_cell_keeps_internal_entity_targets():
    parsed = mm_table_dataset.parse_wiki_cell("[Alpha_Page|Alpha]")

    assert parsed["text"] == "Alpha"
    assert parsed["wiki_title"] == "Alpha Page"
    assert parsed["has_wiki_link"] is True
