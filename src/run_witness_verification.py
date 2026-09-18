#!/usr/bin/env python
"""Independent second-opinion pass over the Experiment-1 witness labels.

A local Qwen3.5-9B (a different model family from the labeling pipeline's
gpt-5.6 reviewers) is shown the frozen object serialization only -- no model
score, no rank, no candidate list -- and asked whether the evidence establishes
a concrete shared attribute value between the query row and the target table.

This is model-assisted annotation, not ground truth, and is reported as such.
Its job is to find systematic disagreement in either direction:

* `retained_witness` / `unretrieved_witness` -- do the verified_positive labels
  survive an independent read?
* `unknown_positive` -- is the 53% "no witness label" really unknown, or is
  there visible witness evidence the builder missed?  (If yes, the retention
  rates are a lower bound and the true retention could be worse.)
* `top10_competitor` -- is any unlabelled competitor actually a valid join?
  (If yes, the qrels are incomplete and some "false positives" are not false.)
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "work/witness_diagnostic_20260916"

SYSTEM = (
    "You audit table joins. You decide only from the supplied material. "
    "Never use outside knowledge, a URL, a file name, or the shape of the target schema. "
    "Answer with one JSON object and nothing else."
)

INSTRUCTIONS = """You are given a QUERY TABLE, a TARGET TABLE, and one piece of EVIDENCE (a text extract or an image) about an entity that appears in the query table.

Decide whether the EVIDENCE lets you establish a concrete attribute value for the query-table entity that ALSO APPEARS in the target table, so that the query table and the target table can be joined on that attribute.

Rules:
- Entity relevance alone is NOT a join. You need a concrete shared value.
- Generic topical similarity, shared category, or a shared URL is NOT a join.
- A value that appears only in the evidence and not in the target table does NOT support a join.
- The query table shows only some of the source columns; the joining attribute may be missing from it. That is expected.
- Judge only what the supplied evidence states. If the evidence does not state a usable attribute value, answer false.

Return exactly this JSON object:
{"join_supported": true|false,
 "shared_value": "<the value that appears in both, or null>",
 "target_column": "<target column name holding that value, or null>",
 "evidence_quote": "<an exact substring of the evidence text, or a short description of the image region, or null>",
 "reason": "<one sentence, evidence-based>"}"""


def image_part(path: str) -> dict[str, Any] | None:
    source = Path(path)
    if not source.is_file():
        return None
    mime = mimetypes.guess_type(source.name)[0] or "image/jpeg"
    encoded = base64.b64encode(source.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def build_messages(packet: dict[str, Any]) -> list[dict[str, Any]]:
    body = {
        "query_table": packet["query_columns_rows"],
        "target_table": packet["target_columns_rows"],
    }
    content: list[dict[str, Any]] = [
        {"type": "text", "text": INSTRUCTIONS},
        {"type": "text", "text": json.dumps(body, ensure_ascii=False, indent=1)},
    ]
    if packet["evidence_modality"] == "text":
        content.append(
            {
                "type": "text",
                "text": "EVIDENCE (text extract):\n" + str(packet["evidence_text"] or ""),
            }
        )
    else:
        part = image_part(packet["evidence_image"]) if packet.get("evidence_image") else None
        content.append({"type": "text", "text": "EVIDENCE (image):"})
        if part is None:
            content.append({"type": "text", "text": "(image file unavailable)"})
        else:
            content.append(part)
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content},
    ]


def call(url: str, model: str, messages: list[dict[str, Any]], timeout: int = 180) -> tuple[str, dict]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 320,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    return body["choices"][0]["message"]["content"], body.get("usage", {})


def parse(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        # Fall back to the first balanced {...} block in case the model still
        # emitted a preamble despite thinking being disabled.
        start = text.find("{")
        value = None
        while start != -1:
            depth = 0
            for index in range(start, len(text)):
                if text[index] == "{":
                    depth += 1
                elif text[index] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            value = json.loads(text[start : index + 1])
                        except json.JSONDecodeError:
                            value = None
                        break
            if value is not None:
                break
            start = text.find("{", start + 1)
        if value is None:
            return {"parse_status": "invalid_json", "join_supported": None}
    if not isinstance(value, dict) or not isinstance(value.get("join_supported"), bool):
        return {"parse_status": "invalid_shape", "join_supported": None}
    return {"parse_status": "valid", **value}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8123")
    parser.add_argument("--model", default="qwen35")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    packets = [
        json.loads(line)
        for line in (IN / "VERIFICATION_PACKETS.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        packets = packets[: args.limit]

    out = IN / "VERIFICATION_RESULTS.jsonl"
    started = time.monotonic()
    records: list[dict[str, Any] | None] = [None] * len(packets)
    done = 0
    lock = threading.Lock()

    def judge(index: int, packet: dict[str, Any]) -> None:
        nonlocal done
        messages = build_messages(packet)
        last_error = None
        for attempt in range(args.retries + 1):
            try:
                text, usage = call(args.url, args.model, messages)
                last_error = None
                break
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as error:
                last_error = repr(error)
                time.sleep(2 * (attempt + 1))
        if last_error is not None:
            record = {**packet, "parse_status": "request_failed", "error": last_error,
                      "join_supported": None}
        else:
            record = {**packet, **parse(text), "raw_response": text, "usage": usage}
        with lock:
            records[index] = record
            done += 1
            if done % 25 == 0:
                print(json.dumps({"event": "progress", "done": done,
                                  "total": len(packets),
                                  "elapsed": round(time.monotonic() - started, 1)}), flush=True)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(judge, index, packet) for index, packet in enumerate(packets)]
        for future in futures:
            future.result()

    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    # ---- agreement tables -------------------------------------------------- #
    rows = []
    by_group: dict[str, Counter] = {}
    for record in records:
        group = record["group"]
        counter = by_group.setdefault(group, Counter())
        counter["total"] += 1
        counter[f"verdict={record.get('join_supported')}"] += 1
        counter[f"parse={record.get('parse_status')}"] += 1
        counter[f"modality={record.get('evidence_modality')}"] += 1
    for group, counter in sorted(by_group.items()):
        total = counter["total"]
        yes = counter["verdict=True"]
        rows.append(
            {
                "group": group,
                "packets": total,
                "judge_join_supported": yes,
                "judge_rate_pct": round(100 * yes / total, 3) if total else None,
                "judge_not_supported": counter["verdict=False"],
                "unparseable": counter["parse=invalid_json"] + counter["parse=invalid_shape"]
                + counter["parse=request_failed"],
            }
        )
    with (IN / "verification_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"event": "done", "packets": len(records),
                      "elapsed": round(time.monotonic() - started, 1)}), flush=True)
    for row in rows:
        print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
