#!/usr/bin/env python
"""Build -- and optionally serve -- a self-contained HTML viewer for the four tables.

The page embeds every row of every table as JSON and renders it client-side, so
it has **no external dependency at all**: no CDN, no font, no API call.  That
matters here because the point of the page is to be readable from other machines
on the LAN, and those are not guaranteed to have internet access (an offline
machine would otherwise render a blank page against a CDN).  It also keeps the
scraped content on the local network, which the plan doc's compliance section
requires.

Usage::

    python scripts_old/build_abebooks_viewer.py                 # write output/abebooks_viewer/index.html
    python scripts_old/build_abebooks_viewer.py --serve --lan   # ...and serve it to the LAN

``--serve`` blocks until Ctrl-C.  ``--lan`` binds ``0.0.0.0``; without it only
this machine can reach the page.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# Running this file directly puts ``scripts_old`` on ``sys.path``, not ``src``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmdd_dataset.utils import read_jsonl

TEMPLATE = Path(__file__).resolve().parent / "abebooks_viewer_template.html"
PLACEHOLDER = "__PAYLOAD__"

#: The four tables, in the order the plan doc introduces them.
TABLES = ("book_edition", "seller", "book_listing", "evidence_asset")
#: Single-file extras, shown as their own tabs when non-empty.
EXTRAS = ("unresolved", "superseded_records")

# Mirrors scripts_old/stage1_gui.py.  Duplicated rather than imported because
# ``scripts_old`` is only on the path under pytest, not when this file is run
# directly, and pulling it in would make this script depend on Flask's siblings.
LOCALHOST_HOST = "127.0.0.1"
LAN_BIND_HOST = "0.0.0.0"
DEFAULT_PORT = 8765


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _table(path: Path) -> dict[str, Any]:
    """Read one JSONL into a column list plus rows of value arrays.

    Rows are arrays rather than objects because the payload repeats every column
    name 2,305 times otherwise -- on the largest table that is a third of the
    page.  The column order is the union of every row's keys, first-seen first,
    so a file that gained a column partway through still renders it.
    """
    columns: list[str] = []
    seen: set[str] = set()
    records = list(read_jsonl(path))
    for record in records:
        for key in record:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    rows = [[record.get(key) for key in columns] for record in records]
    return {"name": path.stem, "columns": columns, "rows": rows, "count": len(rows)}


def build(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = []
    missing = []
    for name in TABLES + EXTRAS:
        path = data_dir / f"{name}.jsonl"
        if not path.exists():
            missing.append(name)
            continue
        table = _table(path)
        if table["rows"]:
            tables.append(table)
    if not any(t["name"] == "book_edition" for t in tables):
        raise SystemExit(f"no book_edition.jsonl under {data_dir} -- run "
                         f"src/build_abebooks_dataset.py first")

    stats = {}
    stats_path = data_dir / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))

    manifest = {}
    manifest_path = data_dir / "dataset_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "input": manifest.get("input") or {"path": str(data_dir)},
        "stats": stats,
        "tables": tables,
    }
    # ``</`` can only ever close the script element early; escaping the slash is
    # valid JSON and makes a cell containing "</script>" harmless.
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    if PLACEHOLDER not in TEMPLATE.read_text(encoding="utf-8"):
        raise SystemExit(f"{TEMPLATE} lost its {PLACEHOLDER} placeholder")

    index = output_dir / "index.html"
    index.write_text(TEMPLATE.read_text(encoding="utf-8").replace(PLACEHOLDER, blob),
                     encoding="utf-8")

    rows = {t["name"]: t["count"] for t in tables}
    print(f"wrote {index} ({index.stat().st_size / 1e6:.1f} MB) -- " +
          ", ".join(f"{n} {c}" for n, c in rows.items()))
    if missing:
        print(f"  skipped (not present): {', '.join(missing)}")
    return {"index": str(index), "rows": rows, "bytes": index.stat().st_size}


class _Handler(SimpleHTTPRequestHandler):
    """Serve the viewer; never cache, so a rebuild is visible on reload."""

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(fmt, *args)  # keep 404s visible, drop the rest


def _lan_ipv4() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))  # no packet sent; just picks the route
            return str(sock.getsockname()[0])
    except OSError:
        return None


def serve(directory: Path, host: str, port: int) -> None:
    httpd = ThreadingHTTPServer((host, port), partial(_Handler, directory=str(directory)))
    bound = httpd.socket.getsockname()
    print(f"serving {directory} on {bound[0]}:{bound[1]}  (Ctrl-C to stop)")
    print(f"  this machine:  http://127.0.0.1:{bound[1]}")
    if host == LAN_BIND_HOST:
        ip = _lan_ipv4()
        print(f"  LAN devices:   http://{ip}:{bound[1]}" if ip
              else "  LAN devices:   http://<this-machine-LAN-IP>:" + str(bound[1]))
    else:
        print("  LAN devices:   not reachable -- pass --lan to bind 0.0.0.0")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--data-dir", default="output/abebooks_dataset",
                        help="directory holding the JSONL tables and stats.json")
    result.add_argument("--output-dir", default="output/abebooks_viewer",
                        help="where index.html is written (and served from)")
    result.add_argument("--serve", action="store_true", help="serve the page until Ctrl-C")
    result.add_argument("--lan", action="store_true",
                        help="bind 0.0.0.0 so other devices on the LAN can open the page")
    result.add_argument("--host", default=None, help="bind host; overrides --lan")
    result.add_argument("--port", type=int, default=DEFAULT_PORT)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    build(args)
    if args.serve:
        serve(Path(args.output_dir), args.host or (LAN_BIND_HOST if args.lan else LOCALHOST_HOST),
              args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
