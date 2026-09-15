#!/usr/bin/env python
"""Download the AbeBooks cover images that the evidence table references.

The four-table build records image evidence as URLs only (``fetch_status`` is
``url_only``); this fills in the binaries.  Two properties matter beyond just
fetching bytes:

**Filenames must not carry the identifier.**  Every source URL leaks something
-- ``/isbn/9780201616477-us._SL300_.jpg`` spells out the ISBN, and the detail
page URL carries a title/author slug.  A file named after its URL would hand a
model the answer to the very join the image is supposed to make recoverable, so
files are named by the SHA-256 of their content instead.  That also dedupes for
free, and the url -> file mapping lives in ``image_manifest.jsonl``.

**It stops instead of retrying.**  A 403/429/503 halts the run, matching
``abebooks_scraper.py``: a refusal is a refusal, not something to work around.

Usage::

    python scripts_old/download_abebooks_images.py --limit 10          # pilot
    python scripts_old/download_abebooks_images.py --include-cards     # all 596 URLs
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Running this file directly puts ``scripts_old`` on ``sys.path``, not ``src``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import requests
from mmdd_progress import progress

from mmdd_dataset.utils import read_jsonl

#: Asset types in ``evidence_asset.jsonl`` whose ``uri`` is an image.
IMAGE_ASSET_TYPES = ("catalogue_cover", "seller_cover")
#: Statuses that mean "stop the run", not "retry this one".
BLOCK_STATUSES = frozenset({403, 429, 503})
#: A real browser UA, as in the scraper -- disguise is not the point, a
#: bare ``python-requests`` UA attracts blocks for no benefit.
USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
EXTENSIONS = {"image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif",
              "image/webp": ".webp"}


class BlockedError(RuntimeError):
    """Raised when the CDN refuses to serve; the run halts rather than retries."""


def evidence_image_urls(path: Path) -> list[str]:
    """Distinct image URLs referenced by the evidence table, in file order."""
    seen: set[str] = set()
    urls: list[str] = []
    for record in read_jsonl(path):
        uri = record.get("uri")
        if record.get("asset_type") in IMAGE_ASSET_TYPES and uri and uri not in seen:
            seen.add(uri)
            urls.append(uri)
    return urls


def card_image_urls(path: Path) -> list[str]:
    """Distinct image URLs from the raw scrape -- every card, not just the pick.

    The four-table build only keeps the representative listing, so its evidence
    rows reach 573 of the 596 distinct images the scrape already paid for.
    """
    seen: set[str] = set()
    urls: list[str] = []
    for record in read_jsonl(path):
        listings = (record.get("search") or {}).get("listings") or []
        candidates = [card.get("image_url") for card in listings]
        candidates.append((record.get("book") or {}).get("catalogue_image_url"))
        for url in candidates:
            if url and url not in seen:
                seen.add(url)
                urls.append(url)
    return urls


def load_done(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """URLs already fetched successfully, so a rerun resumes instead of redoing."""
    if not manifest_path.exists():
        return {}
    return {row["url"]: row for row in read_jsonl(manifest_path)
            if row.get("status") == "ok"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dimensions(blob: bytes) -> tuple[int | None, int | None]:
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow is in the env
        return None, None
    try:
        with Image.open(io.BytesIO(blob)) as image:
            return image.width, image.height
    except Exception:
        return None, None


def fetch_one(session: requests.Session, url: str, timeout: float) -> dict[str, Any]:
    """Fetch one image.  Returns a manifest row; raises :class:`BlockedError`."""
    try:
        response = session.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    except requests.RequestException as exc:
        return {"url": url, "status": "error", "error": type(exc).__name__,
                "fetched_at": _now()}
    if response.status_code in BLOCK_STATUSES:
        raise BlockedError(f"{response.status_code} for {url}")
    if response.status_code != 200:
        return {"url": url, "status": "error", "error": f"http_{response.status_code}",
                "fetched_at": _now()}

    blob = response.content
    if not blob:
        return {"url": url, "status": "error", "error": "empty_body", "fetched_at": _now()}
    mime = (response.headers.get("content-type") or "").split(";")[0].strip()
    if not mime.startswith("image/"):
        return {"url": url, "status": "error", "error": f"not_an_image:{mime}",
                "fetched_at": _now()}
    width, height = _dimensions(blob)
    return {
        "url": url,
        "status": "ok",
        "sha256": hashlib.sha256(blob).hexdigest(),
        "mime_type": mime,
        "bytes": len(blob),
        "width": width,
        "height": height,
        "fetched_at": _now(),
        "_blob": blob,
    }


def download(
    urls: list[str],
    out_dir: Path,
    *,
    delay: float,
    jitter: float,
    timeout: float,
    limit: int | None,
    rng: random.Random,
) -> dict[str, Any]:
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "image_manifest.jsonl"

    done = load_done(manifest_path)
    rows: list[dict[str, Any]] = []
    pending = [url for url in urls if url not in done]
    if limit is not None:
        pending = pending[:limit]

    session = requests.Session()
    try:
        for index, url in enumerate(progress(pending, desc="Download images", unit="img")):
            row = fetch_one(session, url, timeout)
            blob = row.pop("_blob", None)
            if row["status"] == "ok" and blob is not None:
                name = row["sha256"] + EXTENSIONS.get(row["mime_type"], ".bin")
                path = image_dir / name
                if not path.exists():
                    path.write_bytes(blob)
                row["local_path"] = str(path)
                row["relative_path"] = path.relative_to(out_dir).as_posix()
            rows.append(row)
            if index + 1 < len(pending):
                time.sleep(delay + rng.uniform(0, jitter))
    except BlockedError as exc:
        _write(out_dir, rows, manifest_path, done)
        raise SystemExit(f"stopped: {exc} -- the CDN refused service; not retrying")
    finally:
        session.close()

    summary = _write(out_dir, rows, manifest_path, done)
    return summary


def _write(
    out_dir: Path,
    rows: list[dict[str, Any]],
    manifest_path: Path,
    done: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Merge this run into the manifest, keeping one row per URL."""
    merged = {url: row for url, row in done.items()}
    for row in rows:
        merged[row["url"]] = row
    ordered = sorted(merged.values(), key=lambda row: row["url"])
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    ok = [row for row in ordered if row.get("status") == "ok"]
    summary = {
        "manifest": str(manifest_path),
        "urls": len(ordered),
        "ok": len(ok),
        "failed": len(ordered) - len(ok),
        "bytes": sum(row.get("bytes") or 0 for row in ok),
        "distinct_sha256": len({row["sha256"] for row in ok}),
        "images_dir": str(out_dir / "images"),
    }
    return summary


def build(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir)
    urls = evidence_image_urls(dataset_dir / "evidence_asset.jsonl")
    if args.include_cards:
        seen = set(urls)
        urls += [url for url in card_image_urls(Path(args.full)) if url not in seen]

    print(f"{len(urls)} distinct image URLs ({'evidence + cards' if args.include_cards else 'evidence only'})")
    summary = download(urls, Path(args.output_dir), delay=args.delay, jitter=args.jitter,
                       timeout=args.timeout, limit=args.limit,
                       rng=random.Random(args.seed))
    print(f"ok {summary['ok']}/{summary['urls']}  failed {summary['failed']}  "
          f"{summary['bytes'] / 1e6:.1f} MB  "
          f"({summary['distinct_sha256']} distinct files)")
    print(f"  {summary['manifest']}")
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--dataset-dir", default="output/abebooks_dataset",
                        help="directory holding evidence_asset.jsonl")
    result.add_argument("--full", default="output/abebooks_full.jsonl",
                        help="raw scrape, read with --include-cards")
    result.add_argument("--include-cards", action="store_true",
                        help="also fetch card images the representative pick dropped")
    result.add_argument("--output-dir", default="output/abebooks_images")
    result.add_argument("--delay", type=float, default=0.8, help="seconds between requests")
    result.add_argument("--jitter", type=float, default=0.6, help="added at random, 0..jitter")
    result.add_argument("--timeout", type=float, default=30.0)
    result.add_argument("--limit", type=int, default=None, help="fetch at most N (pilot)")
    result.add_argument("--seed", type=int, default=13)
    return result


def main(argv: list[str] | None = None) -> int:
    build(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
