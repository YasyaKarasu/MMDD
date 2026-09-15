"""Tests for the AbeBooks image downloader.

Network is faked throughout -- the point of these tests is the three properties
that are easy to get wrong and expensive to notice: the saved filename must not
leak the identifier the image is meant to make recoverable, a rerun must resume
rather than refetch, and a refusal must stop the run instead of being retried.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

import download_abebooks_images as dl


class FakeResponse:
    def __init__(self, status: int, content: bytes = b"", mime: str = "image/jpeg") -> None:
        self.status_code = status
        self.content = content
        self.headers = {"content-type": mime}


class FakeSession:
    """Serves ``responses`` by URL and records every call."""

    def __init__(self, responses: dict[str, FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def get(self, url: str, **_: object) -> FakeResponse:
        self.calls.append(url)
        return self.responses[url]

    def close(self) -> None:
        pass


def png_bytes() -> bytes:
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", (3, 2), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    data = tmp_path / "dataset"
    data.mkdir()
    write_jsonl(data / "evidence_asset.jsonl", [
        {"asset_type": "catalogue_cover", "uri": "https://pictures.abebooks.com/isbn/9780201616477-us.jpg"},
        {"asset_type": "seller_cover", "uri": "https://pictures.abebooks.com/inventory/31595123263.jpg"},
        {"asset_type": "seller_cover", "uri": "https://pictures.abebooks.com/inventory/31595123263.jpg"},
        {"asset_type": "detail_page", "uri": "https://www.abebooks.com/Some-Title/31595123263/bd"},
        {"asset_type": "synopsis", "uri": "https://www.abebooks.com/Some-Title/31595123263/bd"},
    ])
    return data


def run_download(urls: list[str], out: Path, session: FakeSession, **overrides) -> dict:
    options = {"delay": 0.0, "jitter": 0.0, "timeout": 5.0, "limit": None,
               "rng": __import__("random").Random(0)}
    options.update(overrides)
    original = dl.requests.Session
    dl.requests.Session = lambda: session
    try:
        return dl.download(urls, out, **options)
    finally:
        dl.requests.Session = original


def manifest_of(out: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in
            (out / "image_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    return {row["url"]: row for row in rows}


def test_only_image_rows_are_collected(dataset: Path) -> None:
    urls = dl.evidence_image_urls(dataset / "evidence_asset.jsonl")
    assert urls == ["https://pictures.abebooks.com/isbn/9780201616477-us.jpg",
                    "https://pictures.abebooks.com/inventory/31595123263.jpg"]


def test_card_urls_cover_every_card_plus_the_catalogue_image(tmp_path: Path) -> None:
    full = tmp_path / "full.jsonl"
    write_jsonl(full, [{
        "search": {"listings": [{"image_url": "https://pictures.abebooks.com/inventory/1.jpg"},
                                {"image_url": "https://pictures.abebooks.com/inventory/2.jpg"}]},
        "book": {"catalogue_image_url": "https://pictures.abebooks.com/isbn/9780201616477-us.jpg"},
    }])
    assert dl.card_image_urls(full) == [
        "https://pictures.abebooks.com/inventory/1.jpg",
        "https://pictures.abebooks.com/inventory/2.jpg",
        "https://pictures.abebooks.com/isbn/9780201616477-us.jpg",
    ]


def test_saved_filename_carries_neither_isbn_nor_title(tmp_path: Path) -> None:
    """The whole point of the download: the file must not hand over the answer.

    Every source URL leaks something -- ``/isbn/9780201616477-us`` spells the
    ISBN, the detail-page slug spells the title.  Naming files by content hash
    is what keeps the image from being a shortcut around the join.
    """
    url = "https://pictures.abebooks.com/isbn/9780201616477-us._SL300_.jpg"
    session = FakeSession({url: FakeResponse(200, png_bytes(), "image/png")})
    out = tmp_path / "imgs"
    run_download([url], out, session)

    row = manifest_of(out)[url]
    assert row["status"] == "ok"
    assert row["width"] == 3 and row["height"] == 2
    name = Path(row["local_path"]).name
    assert name == row["sha256"] + ".png"
    assert "9780201616477" not in name and "isbn" not in name.lower()
    assert Path(row["local_path"]).read_bytes() == png_bytes()


def test_a_rerun_resumes_instead_of_refetching(tmp_path: Path) -> None:
    urls = [f"https://pictures.abebooks.com/inventory/{n}.jpg" for n in (1, 2)]
    out = tmp_path / "imgs"
    first = FakeSession({u: FakeResponse(200, png_bytes()) for u in urls})
    run_download(urls, out, first)
    assert first.calls == urls

    second = FakeSession({u: FakeResponse(200, png_bytes()) for u in urls})
    summary = run_download(urls, out, second)
    assert second.calls == []          # nothing refetched
    assert summary["urls"] == 2 and summary["ok"] == 2


def test_a_refusal_stops_the_run_instead_of_retrying(tmp_path: Path) -> None:
    urls = [f"https://pictures.abebooks.com/inventory/{n}.jpg" for n in (1, 2, 3)]
    session = FakeSession({
        urls[0]: FakeResponse(200, png_bytes()),
        urls[1]: FakeResponse(429),
        urls[2]: FakeResponse(200, png_bytes()),
    })
    with pytest.raises(SystemExit, match="stopped"):
        run_download(urls, tmp_path / "imgs", session)
    assert session.calls == urls[:2]           # never reached the third


def test_identical_bytes_are_stored_once(tmp_path: Path) -> None:
    urls = [f"https://pictures.abebooks.com/inventory/{n}.jpg" for n in (1, 2)]
    session = FakeSession({u: FakeResponse(200, png_bytes()) for u in urls})
    out = tmp_path / "imgs"
    summary = run_download(urls, out, session)
    assert summary["ok"] == 2 and summary["distinct_sha256"] == 1
    assert len(list((out / "images").iterdir())) == 1


def test_a_non_image_response_is_recorded_and_does_not_stop_the_run(tmp_path: Path) -> None:
    urls = ["https://pictures.abebooks.com/inventory/1.jpg",
            "https://pictures.abebooks.com/inventory/2.jpg"]
    session = FakeSession({
        urls[0]: FakeResponse(200, b"<html>nope</html>", "text/html"),
        urls[1]: FakeResponse(200, png_bytes()),
    })
    summary = run_download(urls, tmp_path / "imgs", session)
    rows = manifest_of(tmp_path / "imgs")
    assert rows[urls[0]]["error"] == "not_an_image:text/html"
    assert summary["failed"] == 1 and summary["ok"] == 1


def test_manifest_keeps_one_row_per_url(tmp_path: Path) -> None:
    urls = ["https://pictures.abebooks.com/inventory/1.jpg"]
    out = tmp_path / "imgs"
    run_download(urls, out, FakeSession({urls[0]: FakeResponse(200, png_bytes())}))
    run_download(urls, out, FakeSession({urls[0]: FakeResponse(200, png_bytes())}))
    assert len((out / "image_manifest.jsonl").read_text(encoding="utf-8").splitlines()) == 1
