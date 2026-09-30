"""AbeBooks dataset views for global column pruning and nongold hub removal."""
from __future__ import annotations

import copy
import json
import math
import re
from collections import Counter
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def global_join_columns(qrels: list[dict], recoveries: list[dict]) -> set[str]:
    pairs = {(r["query_table_id"], r["target_table_id"]) for r in qrels if r["rel"] > 0}
    names = {r["join_attribute"]["column_name"] for r in qrels if r["rel"] > 0}
    names.update(r["recovered_attribute"]["column_name"] for r in recoveries
                 if (r["query_table_id"], r["target_table_id"]) in pairs)
    return names


def nongold_hubs(popularity: list[dict], canonical: dict[str, str], gold_ids: set[str],
                 threshold: int) -> set[str]:
    """Remove entire frequent content classes, protecting all aliases of gold."""
    protected = {canonical[e] for e in gold_ids}
    hubs = {canonical[r["evidence_id"]] for r in popularity
            if r["train_queries"] >= threshold} - protected
    return {e for e, representative in canonical.items() if representative in hubs}


def nongold_text_hubs(assets: list[dict], popularity: list[dict], canonical: dict[str, str],
                      gold_ids: set[str], threshold: int) -> set[str]:
    """Remove frequent unlabelled text classes while retaining all images."""
    text_classes = {canonical[a["asset_id"]] for a in assets if a["asset_type"] == "text"}
    text_popularity = [r for r in popularity if canonical[r["evidence_id"]] in text_classes]
    return nongold_hubs(text_popularity, canonical, gold_ids, threshold)


def unlabelled_text_assets(assets: list[dict], recoveries: list[dict],
                           canonical: dict[str, str]) -> set[str]:
    """Catalog ablation: retain images and every labelled text content class."""
    protected = {canonical[r["evidence"]["asset_id"]] for r in recoveries}
    return {a["asset_id"] for a in assets if a["asset_type"] == "text"
            and canonical[a["asset_id"]] not in protected}


def duplicate_book_image_assets(sources: list[dict], assets: list[dict], recoveries: list[dict],
                                canonical: dict[str, str]) -> set[str]:
    """Remove pixel-near copies within identical four-field book records.

    Compare complete RGB images resized to 128x128, without cropping. A mean
    absolute channel difference <= 3/255 counts as a near copy. All labelled
    content classes remain, even when several labelled covers look alike.
    """
    import numpy as np
    from PIL import Image

    fields = ("title", "authors", "publisher", "publication_year")
    books = {}
    for table in sources:
        for row in table["rows"]:
            cells = {c["column_name"]: " ".join(c["text"].casefold().split()) for c in row["cells"]}
            if cells.get("title") and cells.get("authors"):
                books[table["source_table_id"], row["row_id"]] = tuple(cells.get(f, "") for f in fields)
    groups = {}
    for asset in assets:
        key = books.get((asset["source_table_id"], asset["source_row_id"]))
        if asset["asset_type"] == "image" and key is not None:
            groups.setdefault(key, {})[canonical[asset["asset_id"]]] = asset
    protected = {canonical[r["evidence"]["asset_id"]] for r in recoveries}
    pixels, removed = {}, set()
    for group in groups.values():
        retained = []
        for eid in sorted(group, key=lambda e: (e not in protected, e)):
            if eid in removed:
                continue
            if eid not in pixels:
                with Image.open(group[eid]["local_path"]) as im:
                    pixels[eid] = np.asarray(im.convert("RGB").resize((128, 128), Image.Resampling.LANCZOS),
                                             dtype=np.float32)
            if eid not in protected and any(np.abs(pixels[eid] - pixels[other]).mean() <= 3.0
                                            for other in retained):
                removed.add(eid)
            else:
                retained.append(eid)
    return {a["asset_id"] for a in assets if canonical[a["asset_id"]] in removed}


def unsupported_source_assets(assets: list[dict], recoveries: list[dict],
                              canonical: dict[str, str]) -> set[str]:
    """Keep whole supported-source asset groups, including unlabelled assets."""
    sources = {r["source_table_id"] for r in recoveries}
    protected = {canonical[r["evidence"]["asset_id"]] for r in recoveries}
    return {a["asset_id"] for a in assets if a["source_table_id"] not in sources
            and canonical[a["asset_id"]] not in protected}


def nonpositive_source_assets(assets: list[dict], qrels: list[dict], recoveries: list[dict],
                              canonical: dict[str, str]) -> set[str]:
    """Keep all evidence from sources with any existing explicit or implicit positive."""
    sources = {r["source_table_id"] for r in qrels if r["rel"] > 0}
    protected = {canonical[r["evidence"]["asset_id"]] for r in recoveries}
    return {a["asset_id"] for a in assets if a["source_table_id"] not in sources
            and canonical[a["asset_id"]] not in protected}


def commercial_text_assets(assets: list[dict], recoveries: list[dict],
                           canonical: dict[str, str]) -> set[str]:
    """Remove seller descriptions/policies, preserving every labelled content class."""
    commercial = {"abebooks_description", "abebooks_seller_policy", "abebooks_shipping_policy"}
    protected = {canonical[r["evidence"]["asset_id"]] for r in recoveries}
    return {a["asset_id"] for a in assets if a["source"] in commercial
            and canonical[a["asset_id"]] not in protected}


def unanchored_text_assets(sources: list[dict], assets: list[dict], recoveries: list[dict],
                           canonical: dict[str, str]) -> set[str]:
    """Keep text mentioning distinctive source-title words, plus all gold aliases.

    A title word is distinctive when it occurs in at most 1% of book rows.
    Require two of the three rarest words (one for a one-word anchor). The
    rule uses source cells and corpus frequencies, never query/test rankings.
    """
    def words(text: str) -> set[str]:
        return {w for w in re.findall(r"\w+", text.casefold()) if len(w) > 2 and w.isalpha()}

    titles = {(t["source_table_id"], row["row_id"]): words(cell["text"])
              for t in sources for row in t["rows"] for cell in row["cells"]
              if cell["column_name"] == "title"}
    frequency = Counter(w for tokens in titles.values() for w in tokens)
    threshold = math.ceil(len(titles) * 0.01)
    anchors = {key: sorted((w for w in tokens if frequency[w] <= threshold),
                          key=lambda w: (frequency[w], w))[:3]
               for key, tokens in titles.items()}
    protected = {canonical[r["evidence"]["asset_id"]] for r in recoveries}
    removed = set()
    for asset in assets:
        if asset["asset_type"] != "text" or canonical[asset["asset_id"]] in protected:
            continue
        anchor = anchors.get((asset["source_table_id"], asset["source_row_id"]), [])
        if not anchor or len(set(anchor) & words(asset.get("content") or "")) < min(2, len(anchor)):
            removed.add(asset["asset_id"])
    return removed


def project_table(table: dict, keep: set[str], source_map: dict[int, int]) -> dict:
    out = copy.deepcopy(table)
    old_columns = [c for c in table["columns"] if c["column_name"] in keep]
    if not old_columns:
        raise ValueError(f"Projection produces an empty table: {table.get('table_id', table.get('source_table_id'))}")
    local_map = {c["column_index"]: i for i, c in enumerate(old_columns)}
    columns = []
    for c in old_columns:
        c = dict(c)
        c["column_index"] = local_map[c["column_index"]]
        if "source_column_index" in c:
            c["source_column_index"] = source_map[c["source_column_index"]]
        columns.append(c)
    out["columns"] = columns
    for row in out["rows"]:
        cells = []
        for cell in row["cells"]:
            if cell["column_index"] not in local_map:
                continue
            cell["column_index"] = local_map[cell["column_index"]]
            if "source_column_index" in cell:
                cell["source_column_index"] = source_map[cell["source_column_index"]]
            cells.append(cell)
        row["cells"] = cells
    if "num_cols" in out:
        out["num_cols"] = len(columns)
    if "source_column_indices" in out:
        out["source_column_indices"] = [c["source_column_index"] for c in columns]
    # AbeBooks join_col and query_entity_col refer to SOURCE column indices.
    for field in ("join_col", "query_entity_col"):
        if field in out:
            old = out[field]
            out[field] = source_map.get(old)
            if out[field] is None:
                out.setdefault("ablation_removed_entity", {})[field] = old
                out[field + "_name"] = None
    for field in ("query_context_col_names", "target_context_col_names"):
        if field in out:
            out[field] = [name for name in out[field] if name in keep]
    for attr in out.get("hidden_attributes", []):
        attr["source_column_index"] = source_map[attr["source_column_index"]]
    meta = out.get("metadata", {})
    if "candidate_entity_columns" in meta:
        meta["candidate_entity_columns"] = [local_map[i] for i in meta["candidate_entity_columns"] if i in local_map]
    if "column_profiles" in meta:
        meta["column_profiles"] = [{**p, "column_index": local_map[p["column_index"]]}
                                   for p in meta["column_profiles"] if p["column_index"] in local_map]
    return out


def _entity_reference(record: dict, source_map: dict[int, int], keep: set[str]) -> None:
    if "entity_column_index" in record:
        record["entity_column_index"] = source_map.get(record["entity_column_index"])
        if record["entity_column_index"] is None:
            record["entity_column_name"] = None
    if "row_attributes" in record:
        record["row_attributes"] = [a for a in record["row_attributes"] if a["name"] in keep]


def make_view(source: Path, destination: Path, *, prune_columns: bool,
              removed_assets: set[str]) -> dict:
    """Write independent files; preserve IDs, split, positive pairs and witnesses."""
    destination.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source / "dataset_manifest.json").read_text())
    artifacts = {name: sum((read_rows(source / s["path"]) for s in spec["shards"]), [])
                 for name, spec in manifest["artifacts"].items()}
    qrels = read_rows(source / manifest["single_files"]["qrels"])
    recoveries = artifacts["evidence_recoveries"]
    gold_ids = {r["evidence"]["asset_id"] for r in recoveries}
    if removed_assets & gold_ids:
        raise ValueError("Cannot remove gold evidence")
    all_columns = {c["column_name"] for t in artifacts["source_tables"] for c in t["columns"]}
    keep = global_join_columns(qrels, recoveries) | {"title"} if prune_columns else all_columns
    maps = {t["source_table_id"]: {c["column_index"]: i for i, c in
            enumerate(c for c in t["columns"] if c["column_name"] in keep)} for t in artifacts["source_tables"]}
    stats = {}
    for name in ("source_tables", "query_tables", "data_lake_tables"):
        before = artifacts[name]
        # Raw lake records can reference a source instead of duplicating its cells.
        after = [project_table(t, keep, maps[t["source_table_id"]])
                 if prune_columns and "columns" in t else copy.deepcopy(t) for t in before]
        stats[name] = {"tables": len(before), "columns_before": sum(len(t.get("columns", [])) for t in before),
                       "columns_after": sum(len(t.get("columns", [])) for t in after)}
        artifacts[name] = after
    for row in qrels:
        attr = row["join_attribute"]
        attr["source_column_index"] = maps[row["source_table_id"]][attr["source_column_index"]]
    for row in recoveries:
        mapping = maps[row["source_table_id"]]
        attr = row["recovered_attribute"]
        for key in ("column_index", "source_column_index"):
            if key in attr:
                attr[key] = mapping[attr[key]]
        _entity_reference(row["query_entity"], mapping, keep)
    for entity in artifacts["entities"]:
        entity["appears_in"] = [{**r, "column_index": maps[r["source_table_id"]][r["column_index"]]}
            for r in entity["appears_in"] if r["column_index"] in maps[r["source_table_id"]]]
    assets = [a for a in artifacts["bridge_assets"] if a["asset_id"] not in removed_assets]
    available = {a["asset_id"] for a in assets}
    assert gold_ids <= available
    artifacts["bridge_assets"] = assets
    links = []
    for row in artifacts["table_asset_links"]:
        # If a seller-name anchor was pruned, retain a row-level asset link.
        row["column_index"] = maps[row["source_table_id"]].get(row["column_index"])
        row["asset_ids"] = [a for a in row["asset_ids"] if a in available]
        if row["asset_ids"]:
            links.append(row)
    artifacts["table_asset_links"] = links
    extractions = []
    for row in artifacts["attribute_extractions"]:
        if row["asset_id"] not in available:
            continue
        _entity_reference(row, maps[row["source_table_id"]], keep)
        row["candidate_attribute_names"] = [n for n in row["candidate_attribute_names"] if n in keep]
        extractions.append(row)
    artifacts["attribute_extractions"] = extractions
    for name, rows in artifacts.items():
        relative = f"{name}/part-00000.jsonl"
        write_rows(destination / relative, rows)
        manifest["artifacts"][name].update(total_records=len(rows), shards=[{"path": relative, "records": len(rows)}])
    write_rows(destination / manifest["single_files"]["qrels"], qrels)
    for name, relative in list(manifest["single_files"].items()):
        if name == "qrels":
            continue
        # Construction decisions and original statistics are historical provenance,
        # explicitly in the original index space; never feed them to training.
        target = relative if name == "splits" else f"provenance/original_{Path(relative).name}"
        (destination / target).parent.mkdir(parents=True, exist_ok=True)
        (destination / target).write_bytes((source / relative).read_bytes())
        manifest["single_files"][name] = target
    report = {"source": str(source), "rule": "global join column names plus title",
              "prune_columns": prune_columns, "kept_columns": sorted(keep),
              "removed_columns": sorted(all_columns - keep), "tables": stats,
              "removed_assets": sorted(removed_assets), "assets_after": len(assets),
              "qrels": len(qrels), "recoveries": len(recoveries), "gold_evidence_removed": 0,
              "source_column_maps": maps,
              "historical_diagnostics_index_space": "original source; see provenance/",
              "attribute_extractions": "historical model responses, retained without API regeneration"}
    manifest["ablation"] = report
    (destination / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return report
