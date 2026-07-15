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


class FakeMediaWikiResponse:
    headers = {}

    def __init__(self, titles, status_code):
        self.titles = titles
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return {
            "query": {
                "pages": [
                    {
                        "pageid": index,
                        "title": title,
                        "extract": f"{title} extract",
                        "images": [],
                        "imageinfo": [{"url": f"https://example.test/{index}.jpg"}],
                    }
                    for index, title in enumerate(self.titles, start=1)
                ]
            }
        }


class SplittingSession:
    def __init__(self, always_414=False):
        self.headers = {}
        self.calls = []
        self.always_414 = always_414

    def get(self, url, params=None, timeout=30):
        titles = params["titles"].split("|")
        self.calls.append(params["titles"])
        status_code = 414 if self.always_414 or len(titles) > 2 else 200
        return FakeMediaWikiResponse(titles, status_code)


def make_client(tmp_path):
    client = mm_table_dataset.WikipediaClient(
        cache_dir=tmp_path / "cache",
        image_output_dir=tmp_path / "images",
        output_dir=tmp_path,
        sleep=0,
        user_agent="test",
    )
    client.retry_base_sleep = 0
    client.retry_max_sleep = 0
    return client


def test_wikipedia_client_splits_page_batch_on_414(tmp_path):
    client = make_client(tmp_path)
    client.session = SplittingSession()

    pages = client.get_pages(["Alpha", "Beta", "Gamma", "Delta"])

    assert client.session.calls == [
        "Alpha|Beta|Gamma|Delta",
        "Alpha|Beta",
        "Gamma|Delta",
    ]
    assert set(pages) == {"Alpha", "Beta", "Gamma", "Delta"}
    assert client.api_failures == 0


def test_wikipedia_client_stops_after_single_title_414(tmp_path):
    client = make_client(tmp_path)
    client.session = SplittingSession(always_414=True)

    pages = client.get_pages(["Unsendable"])

    assert client.session.calls == ["Unsendable"]
    assert pages == {}
    assert client.api_failures == 1


def test_wikipedia_client_splits_imageinfo_batch_on_414(tmp_path):
    client = make_client(tmp_path)
    client.session = SplittingSession()

    infos = client.get_imageinfos(["File:A.jpg", "File:B.jpg", "File:C.jpg"])

    assert client.session.calls == [
        "File:A.jpg|File:B.jpg|File:C.jpg",
        "File:A.jpg",
        "File:B.jpg|File:C.jpg",
    ]
    assert set(infos) == {"File:A.jpg", "File:B.jpg", "File:C.jpg"}
    assert client.api_failures == 0
