"""Serve the showcase directory, declaring UTF-8 in the Content-Type header.

``python -m http.server`` sends ``Content-type: text/html`` with no charset, so
the browser has to guess -- and a browser whose locale defaults to GBK renders
the page's UTF-8 Chinese as mojibake. The page carries a ``<meta charset>`` too,
but declaring it in the header removes the guess entirely.

    python scripts_old/serve_showcase.py --port 8899
"""
from __future__ import annotations

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TEXT_SUFFIXES = {".html", ".css", ".js", ".json", ".jsonl", ".txt", ".svg"}


class Utf8Handler(SimpleHTTPRequestHandler):
    def guess_type(self, path):
        base, _, _ = super().guess_type(path).partition(";")
        if Path(path).suffix.lower() in TEXT_SUFFIXES:
            return f"{base}; charset=utf-8"
        return base

    def log_message(self, fmt, *args):
        # Flush each line: the stdlib default buffers through stderr and a
        # nohup'd server would show nothing until it exits.
        print(f"{self.address_string()} [{self.log_date_time_string()}] {fmt % args}",
              flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--directory", default="output/abebooks_showcase")
    args = parser.parse_args()

    root = Path(args.directory).resolve()
    if not (root / "index.html").exists():
        raise SystemExit(f"no index.html in {root} -- run build_abebooks_showcase.py first")

    handler = functools.partial(Utf8Handler, directory=str(root))
    server = ThreadingHTTPServer((args.bind, args.port), handler)
    print(f"serving {root} at http://{args.bind}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
