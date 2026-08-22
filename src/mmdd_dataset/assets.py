from __future__ import annotations

import mimetypes
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests

from .utils import clean_text, stable_hash


WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"


class PageParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.text: list[str] = []
        self.images: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._ignored_depth += 1
        if tag == "img":
            src = dict(attrs).get("src")
            if src:
                self.images.append(urljoin(self.base_url, src))

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            text = clean_text(data)
            if text:
                self.text.append(text)


def _download_image(
    session: requests.Session,
    url: str,
    image_dir: Path,
    entity_id: str,
    source: str,
) -> dict[str, Any]:
    response = session.get(url, timeout=30)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").split(";", 1)[0]
    suffix = Path(urlparse(url).path).suffix or mimetypes.guess_extension(content_type) or ".img"
    asset_id = f"asset_img_{stable_hash(entity_id, url)}"
    image_dir.mkdir(parents=True, exist_ok=True)
    path = image_dir / f"{asset_id}{suffix[:8]}"
    path.write_bytes(response.content)
    return {
        "asset_id": asset_id,
        "entity_id": entity_id,
        "asset_type": "image",
        "source": source,
        "image_url": url,
        "local_path": str(path.resolve()),
        "relative_path": path.relative_to(image_dir.parent).as_posix(),
        "bytes": len(response.content),
    }


def _text_asset(entity: dict[str, Any], content: str, source: str, page_url: str) -> dict[str, Any]:
    entity_id = entity["entity_id"]
    return {
        "asset_id": f"asset_txt_{stable_hash(entity_id, page_url, content)}",
        "entity_id": entity_id,
        "entity_wiki_title": entity["wiki_title"],
        "asset_type": "text",
        "source": source,
        "page_url": page_url,
        "content": content,
    }


def _wikipedia_assets(
    entity: dict[str, Any],
    session: requests.Session,
    image_dir: Path,
    max_images: int,
) -> list[dict[str, Any]]:
    response = session.get(
        WIKIPEDIA_API,
        params={
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "prop": "extracts|pageimages|info",
            "explaintext": 1,
            "piprop": "original",
            "inprop": "url",
            "titles": entity["wiki_title"],
        },
        timeout=30,
    )
    response.raise_for_status()
    page = response.json()["query"]["pages"][0]
    if page.get("missing"):
        return []

    assets: list[dict[str, Any]] = []
    page_url = clean_text(page.get("fullurl"))
    content = clean_text(page.get("extract"))
    if content:
        assets.append(_text_asset(entity, content, "wikipedia", page_url))
    image_url = clean_text((page.get("original") or {}).get("source"))
    if image_url and max_images > 0:
        assets.append(_download_image(session, image_url, image_dir, entity["entity_id"], "wikipedia"))
    return assets


def _wdc_assets(
    entity: dict[str, Any],
    session: requests.Session,
    image_dir: Path,
    max_images: int,
) -> list[dict[str, Any]]:
    page_url = clean_text(entity.get("page_url"))
    page_images: list[str] = []
    assets: list[dict[str, Any]] = []
    if page_url:
        response = session.get(page_url, timeout=30)
        response.raise_for_status()
        parser = PageParser(response.url)
        parser.feed(response.text)
        content = clean_text(" ".join(parser.text))
        if content:
            assets.append(_text_asset(entity, content, "wdc_page", response.url))
        page_images = parser.images

    image_urls = list(entity.get("image_urls", []))
    image_urls.extend(url for url in page_images if url not in image_urls)
    for url in image_urls[:max_images]:
        assets.append(_download_image(session, url, image_dir, entity["entity_id"], "wdc_page"))
    return assets


def fetch_assets(
    entities: list[dict[str, Any]],
    output_dir: Path,
    *,
    max_entities: int | None,
    max_images_per_entity: int,
    user_agent: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    session = requests.Session()
    session.headers["User-Agent"] = user_agent
    image_dir = output_dir / "images"
    selected = entities if max_entities is None else entities[:max_entities]
    assets: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for entity in selected:
        try:
            if entity["source"] == "entitables":
                assets.extend(_wikipedia_assets(entity, session, image_dir, max_images_per_entity))
            else:
                assets.extend(_wdc_assets(entity, session, image_dir, max_images_per_entity))
        except requests.RequestException as exc:
            failures.append({"entity_id": entity["entity_id"], "error": str(exc)})
    return assets, failures
