# MediaWiki External-Link Filtering and HTTP 414 Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reject external-link markup as entity titles and recover valid MediaWiki titles by splitting batches rejected with HTTP 414.

**Architecture:** Keep entity-link classification in `parse_wiki_cell`. Add a distinct internal exception for HTTP 414 and a `WikipediaClient` batch generator that recursively halves only the rejected title batch; both page and image-info requests consume that generator.

**Tech Stack:** Python 3.10+, `requests`, `pytest`, existing `WikipediaClient` and JSONL caches.

## Global Constraints

- External targets beginning with `//`, `http://`, or `https://`, case-insensitively, must return `wiki_title=None` and `has_wiki_link=True`.
- HTTP 414 must not retry the identical multi-title URL; it must split the batch into two non-empty halves.
- A one-title HTTP 414 must increment `api_failures` once and terminate.
- Existing retry behavior for HTTP 429, HTTP 503, and transient request failures must remain unchanged.
- Do not interrupt or modify the currently running `build` tmux process.

---

### Task 1: Reject external links during cell parsing

**Files:**
- Create: `tests/test_mediawiki_client.py`
- Modify: `scripts/build_mm_table_dataset.py:251-286`

**Interfaces:**
- Consumes: `parse_wiki_cell(cell: str) -> dict[str, Any]`
- Produces: unchanged return schema with external targets represented by `wiki_title=None`, `has_wiki_link=True`

- [ ] **Step 1: Write the failing parser regression tests**

```python
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
```

- [ ] **Step 2: Run the parser tests and verify RED**

Run: `conda run -n MMDD python -m pytest tests/test_mediawiki_client.py -q`

Expected: the three external-link cases fail because their `wiki_title` values are currently non-null; the internal-link case passes.

- [ ] **Step 3: Implement minimal external-target classification**

Add near the title-related constants:

```python
EXTERNAL_LINK_PREFIXES = ("//", "http://", "https://")
```

Update the entity decision in `parse_wiki_cell`:

```python
        page = normalize_title(match.group(1))
        label = clean_text(match.group(2) if match.group(2) is not None else page)
        namespace = page.split(":", 1)[0].lower() if ":" in page else ""
        is_external = page.lower().startswith(EXTERNAL_LINK_PREFIXES)
        wiki_title = None if is_external or namespace in NON_ENTITY_NAMESPACES else page
```

- [ ] **Step 4: Run the parser tests and verify GREEN**

Run: `conda run -n MMDD python -m pytest tests/test_mediawiki_client.py -q`

Expected: `4 passed`.

- [ ] **Step 5: Commit the isolated parser change**

```bash
git add scripts/build_mm_table_dataset.py tests/test_mediawiki_client.py
git commit -m "Reject external links as Wikipedia entities"
```

### Task 2: Split MediaWiki title batches on HTTP 414

**Files:**
- Modify: `tests/test_mediawiki_client.py`
- Modify: `scripts/build_mm_table_dataset.py:1050-1240`

**Interfaces:**
- Produces: `MediaWikiURITooLong(Exception)` as an internal control-flow signal
- Produces: `WikipediaClient._get_title_batches(titles: list[str], base_params: dict[str, Any]) -> Iterable[tuple[list[str], dict[str, Any]]]`
- Consumes: `WikipediaClient._get(params: dict[str, Any]) -> dict[str, Any] | None`

- [ ] **Step 1: Write failing batch-splitting tests**

Append to `tests/test_mediawiki_client.py`:

```python
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
    return mm_table_dataset.WikipediaClient(
        cache_dir=tmp_path / "cache",
        image_output_dir=tmp_path / "images",
        output_dir=tmp_path,
        sleep=0,
        user_agent="test",
    )


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
```

- [ ] **Step 2: Run the 414 tests and verify RED**

Run: `conda run -n MMDD python -m pytest tests/test_mediawiki_client.py -q`

Expected: all three 414 tests fail because `_get` currently retries the identical request six times and never splits it.

- [ ] **Step 3: Expose HTTP 414 without exponential retry**

Add an internal exception near the MediaWiki constants:

```python
class MediaWikiURITooLong(Exception):
    """Signal that a MediaWiki title batch must be split."""
```

In `_get`, immediately after receiving the response and before retry-status handling, add:

```python
                if response.status_code == 414:
                    raise MediaWikiURITooLong
```

Add a dedicated exception branch before the general `except Exception` branch:

```python
            except MediaWikiURITooLong:
                raise
```

- [ ] **Step 4: Implement recursive title-batch splitting**

Add this method before `get_pages`:

```python
    def _get_title_batches(
        self,
        titles: list[str],
        base_params: dict[str, Any],
    ) -> Iterable[tuple[list[str], dict[str, Any]]]:
        for start in range(0, len(titles), MEDIAWIKI_BATCH_TITLE_LIMIT):
            batch = titles[start : start + MEDIAWIKI_BATCH_TITLE_LIMIT]
            yield from self._get_title_batch(batch, base_params)

    def _get_title_batch(
        self,
        batch: list[str],
        base_params: dict[str, Any],
    ) -> Iterable[tuple[list[str], dict[str, Any]]]:
        if not batch:
            return
        params = dict(base_params)
        params["titles"] = "|".join(batch)
        try:
            payload = self._get(params)
        except MediaWikiURITooLong:
            if len(batch) == 1:
                self.api_failures += 1
                logging.warning("MediaWiki API rejected one title with HTTP 414: %s", batch[0])
                return
            midpoint = len(batch) // 2
            yield from self._get_title_batch(batch[:midpoint], base_params)
            yield from self._get_title_batch(batch[midpoint:], base_params)
            return
        if payload:
            yield batch, payload
```

Change both `get_pages` and `get_imageinfos` to construct their existing request parameters without `titles`, then iterate as follows:

```python
        for batch, payload in self._get_title_batches(missing_titles, base_params):
            query = payload.get("query", {})
```

Keep each method's existing response normalization, cache writes, and result assembly inside that loop.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `conda run -n MMDD python -m pytest tests/test_mediawiki_client.py tests/test_stage1_pipeline.py::test_wikipedia_client_get_pages_batches_titles_with_pipe tests/test_stage1_pipeline.py::test_wikipedia_client_retries_rate_limit_with_retry_after -q`

Expected: all focused tests pass; the rate-limit test confirms non-414 retry behavior is unchanged.

- [ ] **Step 6: Run the full repository test command**

Run: `conda run -n MMDD python -m pytest tests/test_stage1_pipeline.py tests/test_mediawiki_client.py -q`

Expected: all tests pass with exit code 0.

- [ ] **Step 7: Inspect scope and commit the 414 recovery**

Run: `git diff --check` and `git diff -- scripts/build_mm_table_dataset.py tests/test_mediawiki_client.py`.

Expected: no whitespace errors; diff contains only external-link filtering, HTTP 414 signaling/splitting, and their tests.

```bash
git add scripts/build_mm_table_dataset.py tests/test_mediawiki_client.py
git commit -m "Split MediaWiki batches rejected with HTTP 414"
```
