"""Build a browsable data browser for the AbeBooks MMDD dataset.

Writes ``index.html`` + ``app.css`` + ``app.js`` + ``data/*.json`` + ``images/``
into a dedicated directory that contains *only* those files, so serving it never
exposes the repository. No CDN, no external font, no outbound request at all --
it renders on an intranet host with no internet.

    python scripts_old/build_abebooks_showcase.py
    python scripts_old/serve_showcase.py --port 8899

Every table is emitted whole -- all rows, all columns. Nothing is sampled.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmdd_dataset.utils import read_jsonl  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "showcase"

# The dataset marks a missing value with the *string* "None" (as well as with a
# real null). Rendering both as an em dash is what a reader expects from a data
# browser; the stored value is unchanged on disk either way.
NULLS = {"None", "none", "", "null"}


def cell(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    text = str(value)
    return None if text in NULLS else text


def table(columns, rows):
    return {"columns": list(columns), "rows": [[cell(v) for v in row] for row in rows]}


class Images:
    """Copy referenced images in and hand back a page-relative path."""

    def __init__(self, manifest: Path, out_dir: Path):
        self.out = out_dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.by_url: dict[str, Path] = {}
        self.copied: set[str] = set()
        if manifest.exists():
            for row in read_jsonl(manifest):
                local = row.get("local_path")
                if row.get("status") == "ok" and local and (REPO / local).exists():
                    self.by_url[row["url"]] = REPO / local

    def link(self, path: Path | None) -> str | None:
        if path is None or not path.exists():
            return None
        target = self.out / path.name
        if path.name not in self.copied:
            shutil.copy2(path, target)
            self.copied.add(path.name)
        return f"images/{path.name}"

    def from_url(self, url: str | None) -> str | None:
        return self.link(self.by_url.get(url)) if url else None


def scrape_records(full: Path):
    """One row per ISBN -- the last record wins, since degraded runs are retried."""
    rows = []
    latest: dict[str, dict] = {}
    for record in read_jsonl(full):
        latest[record["isbn"]] = record
    for isbn, r in latest.items():
        search = r.get("search") or {}
        book = r.get("book") or {}
        rows.append([
            isbn, r.get("status"), r.get("retrieved_at"), r.get("http_status"),
            search.get("listing_count"), book.get("title"), book.get("authors"),
            book.get("publisher"), book.get("publication_year"), book.get("binding"),
            book.get("edition_number"), book.get("series"), book.get("isbn13"),
            book.get("copy_condition_grade"), r.get("source_url"),
            r.get("html_path"), r.get("degraded_url"), r.get("error"),
        ])
    columns = ["isbn", "status", "retrieved_at", "http_status", "listing_count",
               "title", "authors", "publisher", "publication_year", "binding",
               "edition_number", "series", "isbn13", "copy_condition_grade",
               "source_url", "search_html", "degraded_url", "error"]
    return table(columns, rows)


def dataset_table(path: Path, derived=None):
    """A dataset table verbatim, optionally with extra derived columns appended."""
    rows = list(read_jsonl(path))
    columns = list(rows[0].keys()) if rows else []
    body = [[r.get(c) for c in columns] for r in rows]
    if derived:
        for name, fn in derived:
            columns.append(name)
            for row, source in zip(body, rows):
                row.append(fn(source))
    return table(columns, body)


def lake_tables(lake_dir: Path):
    """Every source table, cells flattened to their text."""
    out = []
    for t in read_jsonl(lake_dir / "source_tables.jsonl"):
        columns = [c["column_name"] for c in t["columns"]]
        rows = []
        for r in t["rows"]:
            cells = {c["column_name"]: c.get("text") for c in r["cells"]}
            rows.append([r["row_id"]] + [cells.get(c) for c in columns])
        out.append({
            "id": t["source_table_id"],
            "label": f"{t['source_table_id']} · {t['source_name']}",
            "table": table(["row_id"] + columns, rows),
        })
    return out


def bridge_assets(lake_dir: Path, images: Images):
    rows = list(read_jsonl(lake_dir / "bridge_assets.jsonl"))
    columns = ["asset_id", "asset_type", "source", "source_column", "row_id",
               "content", "url", "sha256", "mime_type", "width", "height"]
    body = []
    for r in rows:
        body.append([
            r.get("asset_id"), r.get("asset_type"), r.get("source"),
            r.get("source_column"), r.get("row_id"), r.get("content"),
            r.get("url"), r.get("sha256"), r.get("mime_type"),
            r.get("width"), r.get("height"),
            images.link(REPO / r["local_path"]) if r.get("local_path") else None,
        ])
    return table(columns + ["本地图片"], body)


def plan_candidates(plan: dict):
    rows = []
    for t in plan["tables"]:
        for c in t["columns"]:
            rows.append([
                t["source_table_id"], t["source_name"], t["num_rows"],
                c["column"], ", ".join(c["families"]), ", ".join(c["copy_families"]),
                ", ".join(c["open_families"]), c["max_value_share"], c["target_rows"],
            ])
    columns = ["source_table_id", "source_name", "num_rows", "column", "families",
               "copy_families", "open_families", "max_value_share", "target_rows"]
    return table(columns, rows)


def build(args) -> dict:
    data_dir = Path(args.data_dir)
    lake_dir = Path(args.lake_dir)
    out_dir = Path(args.output_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)

    for name in ("index.html", "app.css", "app.js"):
        shutil.copy2(ASSETS / name, out_dir / name)

    images = Images(Path(args.image_manifest), out_dir / "images")
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    manifest = json.loads((lake_dir / "lake_manifest.json").read_text(encoding="utf-8"))

    groups: list[dict] = []
    emitted = 0
    total_rows = 0

    def emit(group, tid, label, payload):
        nonlocal emitted, total_rows
        (out_dir / "data" / f"{tid}.json").write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8")
        group["tables"].append({
            "id": tid, "label": label,
            "rows": len(payload["rows"]), "cols": len(payload["columns"]),
        })
        emitted += 1
        total_rows += len(payload["rows"])

    g = {"label": "采集原始记录", "tables": []}
    emit(g, "scrape_records", "abebooks 采集记录", scrape_records(Path(args.full)))
    groups.append(g)

    g = {"label": "数据集", "tables": []}
    for name in ("book_edition", "seller", "book_listing"):
        emit(g, name, name, dataset_table(data_dir / f"{name}.jsonl"))
    emit(g, "evidence_asset", "evidence_asset",
         dataset_table(data_dir / "evidence_asset.jsonl",
                       derived=[("本地图片", lambda r: images.from_url(r.get("uri")))]))
    groups.append(g)

    g = {"label": "数据湖 source tables", "tables": []}
    for entry in lake_tables(lake_dir):
        emit(g, entry["id"], entry["label"], entry["table"])
    groups.append(g)

    g = {"label": "桥接素材 / 规划", "tables": []}
    emit(g, "bridge_assets", "bridge_assets", bridge_assets(lake_dir, images))
    emit(g, "plan_candidates", "joinability 候选（936）", plan_candidates(plan))
    groups.append(g)

    (out_dir / "data" / "manifest.json").write_text(json.dumps({
        "generated_at": manifest.get("generated_at"),
        "groups": groups,
        "total_tables": emitted,
        "total_rows": total_rows,
    }, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    return {
        "output": str(out_dir / "index.html"),
        "tables": emitted,
        "rows": total_rows,
        "images": len(images.copied),
        "bytes": size,
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", default="output/abebooks_dataset_full")
    p.add_argument("--lake-dir", default="output/abebooks_lake_full")
    p.add_argument("--full", default="output/abebooks_full.jsonl")
    p.add_argument("--plan", default="output/abebooks_joinability_full/plan.json")
    p.add_argument("--image-manifest",
                   default="output/abebooks_images_full/image_manifest.jsonl")
    p.add_argument("--output-dir", default="output/abebooks_showcase")
    return p


def main(argv: list[str] | None = None) -> int:
    result = build(parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
