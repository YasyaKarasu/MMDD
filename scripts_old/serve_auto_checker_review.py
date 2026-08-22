#!/usr/bin/env python
"""Serve an auto-check review site and persist one shared review state."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


MAX_REQUEST_BYTES = 2 * 1024 * 1024
ALLOWED_VERDICTS = {"", "supported", "contradicted", "insufficient", "schema_bad"}
ALLOWED_CONFIDENCE = {"", "high", "medium", "low"}
ALLOWED_ISSUES = {
    "",
    "model_false_positive",
    "model_false_negative",
    "normalization",
    "attribute_ambiguous",
    "evidence_ambiguous",
    "entity_mismatch",
    "other",
}


def _bounded_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text[:limit]


def validate_review_payload(
    payload: Any,
    *,
    dataset_id: str,
    allowed_review_ids: set[str],
) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("dataset_id") != dataset_id:
        raise ValueError("review dataset identity does not match")
    raw_reviews = payload.get("reviews")
    if not isinstance(raw_reviews, dict) or len(raw_reviews) > len(allowed_review_ids):
        raise ValueError("invalid review collection")
    reviews: dict[str, dict[str, Any]] = {}
    for review_id, raw_review in raw_reviews.items():
        if review_id not in allowed_review_ids or not isinstance(raw_review, dict):
            raise ValueError("invalid review item")
        verdict = _bounded_text(raw_review.get("human_verdict"), 32)
        confidence = _bounded_text(raw_review.get("confidence"), 16)
        issue_type = _bounded_text(raw_review.get("issue_type"), 64)
        if verdict not in ALLOWED_VERDICTS:
            raise ValueError("invalid human verdict")
        if confidence not in ALLOWED_CONFIDENCE:
            raise ValueError("invalid review confidence")
        if issue_type not in ALLOWED_ISSUES:
            raise ValueError("invalid review issue type")
        reviews[review_id] = {
            "human_verdict": verdict,
            "human_value": _bounded_text(raw_review.get("human_value"), 500),
            "confidence": confidence,
            "issue_type": issue_type,
            "notes": _bounded_text(raw_review.get("notes"), 4000),
            "locked": bool(raw_review.get("locked")),
            "completed": bool(raw_review.get("completed")),
            "updated_at": _bounded_text(raw_review.get("updated_at"), 64),
        }
    return {
        "schema_version": "mmdd-auto-check-human-review-state-v1",
        "dataset_id": dataset_id,
        "reviewer": _bounded_text(payload.get("reviewer"), 200),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "reviews": reviews,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class ReviewRequestHandler(SimpleHTTPRequestHandler):
    server_version = "MMDDReview/1.0"

    def __init__(
        self,
        *args: Any,
        directory: str,
        dataset_id: str,
        allowed_review_ids: set[str],
        save_path: Path,
        **kwargs: Any,
    ) -> None:
        self.dataset_id = dataset_id
        self.allowed_review_ids = allowed_review_ids
        self.save_path = save_path
        super().__init__(*args, directory=directory, **kwargs)

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; connect-src 'self'; base-uri 'none'; "
            "form-action 'self'; frame-ancestors 'none'",
        )
        super().end_headers()

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/api/health":
            self._send_json(
                HTTPStatus.OK,
                {"status": "ok", "dataset_id": self.dataset_id},
            )
            return
        if path == "/api/reviews":
            if self.save_path.is_file():
                try:
                    payload = json.loads(self.save_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    self._send_json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"error": "saved review state is unreadable"},
                    )
                    return
            else:
                payload = {
                    "schema_version": "mmdd-auto-check-human-review-state-v1",
                    "dataset_id": self.dataset_id,
                    "reviewer": "",
                    "saved_at": "",
                    "reviews": {},
                }
            self._send_json(HTTPStatus.OK, payload)
            return
        if path == "/":
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/review.html")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if urlsplit(self.path).path != "/api/reviews":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if not 0 < content_length <= MAX_REQUEST_BYTES:
            self._send_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": "invalid request size"},
            )
            return
        try:
            raw = self.rfile.read(content_length)
            request_payload = json.loads(raw.decode("utf-8"))
            saved_payload = validate_review_payload(
                request_payload,
                dataset_id=self.dataset_id,
                allowed_review_ids=self.allowed_review_ids,
            )
            write_json_atomic(self.save_path, saved_payload)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid review payload"})
            return
        except OSError:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "could not persist review state"},
            )
            return
        self._send_json(
            HTTPStatus.OK,
            {
                "status": "saved",
                "saved_at": saved_payload["saved_at"],
                "review_count": len(saved_payload["reviews"]),
            },
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the MMDD auto-check human review site."
    )
    parser.add_argument("--site_dir", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    site_dir = Path(args.site_dir).resolve()
    manifest_path = site_dir / "review_manifest.json"
    if not (site_dir / "review.html").is_file() or not manifest_path.is_file():
        raise ValueError("review site is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_id = str(manifest.get("dataset_id") or "")
    allowed_review_ids = {
        str(item.get("review_id") or "")
        for item in manifest.get("items") or []
        if isinstance(item, dict) and item.get("review_id")
    }
    if not dataset_id or not allowed_review_ids:
        raise ValueError("review manifest is invalid")
    handler = partial(
        ReviewRequestHandler,
        directory=str(site_dir),
        dataset_id=dataset_id,
        allowed_review_ids=allowed_review_ids,
        save_path=Path(args.save_path).resolve(),
    )
    server = ThreadingHTTPServer((args.bind, args.port), handler)
    print(
        f"Serving {len(allowed_review_ids)} review items on "
        f"http://{args.bind}:{args.port}/review.html",
        flush=True,
    )
    server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_args(argv))
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(
            f"ERROR: review server stopped ({type(error).__name__})",
            file=sys.stderr,
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
