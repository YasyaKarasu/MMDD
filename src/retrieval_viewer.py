"""Export existing AbeBooks retrieval traces and serve an offline LAN viewer.

Only exported viewer data and dataset-registered images are exposed by HTTP.
No models, APIs, or project configuration are loaded.
"""
from __future__ import annotations

import argparse
import gzip
import json
import mimetypes
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ASSETS = Path(__file__).with_name("retrieval_viewer_assets")
ARMS = {"baseline": "原始数据", "hubs": "仅过滤素材", "columns": "仅精简列", "both": "两者同时"}
GENERATORS = {"raw": "Raw embedding", "endpoint_sup": "SUP · 训练终点",
              "endpoint_kd": "KD · 训练终点", "selected_sup": "SUP · Dev 选中",
              "selected_kd": "KD · Dev 选中"}


def read_rows(path: Path):
    with (gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(text)
    else:
        path.write_text(text, encoding="utf-8")


def compact_table(table: dict) -> dict:
    return {"id": table["table_id"], "source": table["source_table_id"],
            "columns": [c["column_name"] for c in table["columns"]],
            "rows": [[next((c.get("text", "") for c in r["cells"] if c["column_index"] == col["column_index"]), "")
                      for col in table["columns"]] for r in table["rows"]],
            "row_ids": [r["row_id"] for r in table["rows"]],
            "join_column": table.get("join_col_name"),
            "hidden": [a["column_name"] for a in table.get("hidden_attributes", [])]}


def table_label(table: dict) -> str:
    cols = table["columns"]
    if "title" in cols:
        i = cols.index("title")
        titles = [str(row[i]) for row in table["rows"] if row[i]]
        if titles:
            return titles[0]
    return " / ".join(cols[:4])


def assemble_query(pool: dict, first_hop: list[dict], paths: list[dict],
                   gold: list[dict], teacher_ranks: dict[str, dict[str, int]]) -> dict:
    """Keep query/evidence/target identity and the two rank spaces explicit."""
    evidence = {r["evidence_id"]: {"id": r["evidence_id"], "modality": r["modality"],
                "rank": r["rank"], "score": r["score"], "targets": [], "gold_target_ids": []} for r in first_hop}
    gold_by_target = {r["target_id"]: set(r["evidence_ids"]) for r in gold}
    for e in evidence.values():
        e["gold_target_ids"] = [t for t, ids in gold_by_target.items() if e["id"] in ids]
    seen = set()
    for r in paths:
        key = r["evidence_id"], r["target_id"]
        if key in seen:
            raise ValueError(f"Duplicate second-hop edge: {key}")
        seen.add(key)
        e = evidence[r["evidence_id"]]
        assert r["query_id"] == pool["query_id"] and r["first_rank"] == e["rank"]
        e["targets"].append({"id": r["target_id"], "rank": r["second_rank"],
            "score": r["second_raw_score"], "path_score": r["path_raw_score"],
            "retained": r["retained"], "gold_path": r["evidence_id"] in gold_by_target.get(r["target_id"], set()),
            "support": r.get("row_support"), "gain": r.get("D1_marginal_gain")})
    for e in evidence.values():
        e["targets"].sort(key=lambda x: (x["rank"], x["id"]))
    direct = pool["D100_ANN"]
    c150 = pool["C150"]
    all_targets = set(pool["U"]) | set(gold_by_target)
    targets = {t: {"gold": t in gold_by_target,
                  "direct_rank": direct.index(t) + 1 if t in direct else None,
                  "c150_rank": c150.index(t) + 1 if t in c150 else None,
                  "qt_rank": pool["QT_ranks"].get(t), "d1_rank": pool["D1_ranks"].get(t),
                  "direct_score": pool["all_U_QT_scores"].get(t),
                  "teacher_rank": teacher_ranks.get("Real", {}).get(t),
                  "f0_rank": teacher_ranks.get("f0", {}).get(t),
                  "bag": pool["retained_bags"].get(t, [])} for t in all_targets}
    return {"query_id": pool["query_id"], "evidence": sorted(evidence.values(), key=lambda e: (e["rank"], e["modality"])),
            "targets": targets, "gold": gold, "direct": direct, "c150": c150,
            "teacher_available": bool(teacher_ranks)}


def build(root: Path, output: Path) -> None:
    assets, image_paths = {}, {}
    for a in read_rows(root / "baseline/dataset_view/bridge_assets/part-00000.jsonl"):
        assets[a["asset_id"]] = {"id": a["asset_id"], "type": a["asset_type"],
            "content": a.get("content") or "", "source": a.get("source"),
            "source_table": a.get("source_table_id"), "source_row": a.get("source_row_id")}
        if a["asset_type"] == "image":
            image_paths[a["asset_id"]] = a.get("local_path") or a.get("relative_path")
    write_json(output / "data/assets.json.gz", assets)
    write_json(output / "image_paths.json", image_paths)
    catalog = {"title": "AbeBooks · 检索路径", "arms": ARMS, "generators": GENERATORS,
               "queries": [], "available": {}, "combination_queries": {}, "payload_count": 0}
    for arm in ARMS:
        run = root / arm
        dataset = run / "dataset_view"
        raw_queries = list(read_rows(dataset / "query_tables/part-00000.jsonl"))
        tables = {t["table_id"]: compact_table(t) for part in ("query_tables", "data_lake_tables")
                  for t in read_rows(dataset / part / "part-00000.jsonl")}
        for table in tables.values():
            table["label"] = table_label(table)
        write_json(output / f"data/{arm}/tables.json.gz", tables)
        canonical = {r["asset_id"]: r["canonical_evidence_id"] for r in read_rows(run / "CONTENT_ALIASES.jsonl.gz")}
        recovery_by_pair = defaultdict(list)
        for r in read_rows(dataset / "evidence_recoveries/part-00000.jsonl"):
            attr = r["recovered_attribute"]
            recovery_by_pair[r["query_table_id"], r["target_table_id"]].append({
                "row": r["query_row_id"], "attribute": attr["column_name"], "value": attr.get("value"),
                "evidence_id": canonical[r["evidence"]["asset_id"]]})
        gold_by_query = defaultdict(list)
        for r in read_rows(dataset / "qrels.jsonl"):
            if r["rel"] > 0:
                q, t = r["query_table_id"], r["target_table_id"]
                recoveries = recovery_by_pair[q, t]
                gold_by_query[q].append({"target_id": t, "join_column": r["join_attribute"]["column_name"],
                    "reason": r["reason"], "evidence_ids": sorted({x["evidence_id"] for x in recoveries}),
                    "recoveries": recoveries})
        if arm == "baseline":
            for q in raw_queries:
                table = tables[q["table_id"]]
                kind = "implicit" if any(g["reason"] == "model_recoverable_join_column" for g in gold_by_query[q["table_id"]]) else "explicit"
                catalog["queries"].append({"id": table["id"], "label": table["label"], "split": q["split"],
                    "kind": kind, "gold_count": len(gold_by_query[table["id"]]), "rows": len(table["rows"])})
        for split in ("test", "dev", "train"):
            catalog["available"][f"{arm}/{split}"] = []
            for generator in GENERATORS:
                directory = run / "eval" / split / generator
                if split == "train":
                    if generator != "raw":
                        continue
                    directory = run / "training_records/raw_train"
                if not (directory / "pools.jsonl.gz").exists():
                    continue
                first, paths, teacher = defaultdict(list), defaultdict(list), defaultdict(dict)
                for r in read_rows(directory / "first_hop.jsonl.gz"):
                    first[r["query_id"]].append(r)
                for r in read_rows(directory / "prepaths.jsonl.gz"):
                    paths[r["query_id"]].append(r)
                for view in ("Real", "f0"):
                    filename = directory / f"rankings.TB_CQET.{view}.jsonl.gz"
                    if filename.exists():
                        for r in read_rows(filename):
                            teacher[r["query_id"]][view] = {t: i + 1 for i, t in enumerate(r["target_ids"])}
                query_ids = []
                for pool in read_rows(directory / "pools.jsonl.gz"):
                    q = pool["query_id"]
                    payload = assemble_query(pool, first[q], paths.pop(q, []), gold_by_query[q], teacher[q])
                    write_json(output / f"data/{arm}/{split}/{generator}/{q}.json.gz", payload)
                    query_ids.append(q)
                    catalog["payload_count"] += 1
                catalog["available"][f"{arm}/{split}"].append(generator)
                catalog["combination_queries"][f"{arm}/{split}/{generator}"] = query_ids
                print(f"EXPORTED {arm}/{split}/{generator}: {len(query_ids)} queries", flush=True)
    write_json(output / "data/catalog.json.gz", catalog)
    print(f"READY {output} ({catalog['payload_count']} query views)", flush=True)


def make_handler(output: Path):
    images = json.loads((output / "image_paths.json").read_text())
    data_root = (output / "data").resolve()
    public_assets = {"/": (ASSETS / "index.html", "text/html; charset=utf-8"),
                     "/app.js": (ASSETS / "app.js", "text/javascript; charset=utf-8"),
                     "/style.css": (ASSETS / "style.css", "text/css; charset=utf-8"),
                     "/font.otf": (output / "viewer-sc.otf", "font/otf")}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = unquote(urlsplit(self.path).path)
            compressed = False
            if url in public_assets:
                path, content_type = public_assets[url]
            elif url.startswith("/image/"):
                asset_id = url[len("/image/"):]
                if asset_id not in images:
                    self.send_error(404)
                    return
                path = Path(images[asset_id])
                content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
            elif url.startswith("/data/") and url.endswith(".json.gz"):
                path = (output / url.lstrip("/")).resolve()
                if not path.is_relative_to(data_root):
                    self.send_error(404)
                    return
                content_type, compressed = "application/json; charset=utf-8", True
            else:
                self.send_error(404)
                return
            if not path.is_file():
                self.send_error(404)
                return
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-cache" if url in ("/", "/app.js", "/style.css") else "public, max-age=3600")
            if compressed:
                self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "serve"))
    parser.add_argument("--run-root", type=Path, default=Path(__file__).resolve().parents[1] / "work/abebooks_data_ablation_20260930")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    output = args.output or args.run_root / "viewer"
    if args.command == "build":
        build(args.run_root.resolve(), output.resolve())
    else:
        server = ThreadingHTTPServer((args.host, args.port), make_handler(output.resolve()))
        print(f"Retrieval viewer listening on http://{args.host}:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
