from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

import requests
from mmdd_progress import progress

from .utils import clean_text, get_cell, get_column_name, stable_hash, values_match

PROMPT_VERSION = "leave_one_attribute_out_v4_entity_evidence_grounding"
AUTO_CHECK_PROMPT_VERSION = "query_visible_row_raw_evidence_only_v2"
AUTO_CHECK_CASCADE_POLICY = "local_luna_consensus_terra_adjudication_v1"
AUTO_CHECK_LOCAL_POLICY = "local_only"


class OpenAICompatibleExtractor:
    """Minimal client shared by local vLLM and OpenAI-compatible endpoints."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key_env: str = "VLLM_API_KEY",
        timeout: float = 120,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()
        api_key = os.environ.get(api_key_env)
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def extract(
        self,
        *,
        attribute: str,
        visible_cells: list[dict[str, str]],
        asset: dict[str, Any],
    ) -> dict[str, str]:
        prompt = (
            "Extract one attribute from the evidence. The target attribute is omitted "
            "from the supplied table cells, so do not infer it from a supplied answer. "
            "The evidence may be unrelated to the entity in the table row. First verify "
            "that the evidence itself explicitly and unambiguously refers to that entity. "
            "If this connection cannot be established from the supplied row and evidence "
            "alone, return an empty value. Do not extract a plausible target-attribute "
            "value while ignoring or merely assuming the entity-evidence relationship. "
            "Return JSON only as {\"value\": \"...\", \"evidence\": \"...\"}. Use an "
            "empty value when the evidence does not state the answer for that entity.\n\n"
            f"Target attribute: {attribute}\n"
            f"Visible query-table row: {json.dumps(visible_cells, ensure_ascii=False)}\n"
        )
        content: str | list[dict[str, Any]] = prompt + "Evidence:\n" + clean_text(asset.get("content"))
        if asset["asset_type"] == "image":
            path = Path(asset["local_path"])
            media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{encoded}"}},
            ]
        response = self.session.post(
            self.url,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"]
        if isinstance(answer, list):
            answer = "".join(item.get("text", "") for item in answer if isinstance(item, dict))
        parsed = _json_object(answer)
        return {
            "value": clean_text(parsed.get("value")),
            "evidence": clean_text(parsed.get("evidence")),
        }


def _json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start < 0:
        return {}
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def build_extractions(
    tables: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    assets: list[dict[str, Any]],
    text_extractor: OpenAICompatibleExtractor | None,
    *,
    image_extractor: OpenAICompatibleExtractor | None = None,
    min_column_non_empty_ratio: float,
) -> list[dict[str, Any]]:
    entity_by_title = {entity["wiki_title"]: entity for entity in entities}
    assets_by_entity: dict[str, list[dict[str, Any]]] = {}
    for asset in assets:
        assets_by_entity.setdefault(asset["entity_id"], []).append(asset)

    records: list[dict[str, Any]] = []
    for table in progress(tables, desc="Extract attributes", unit="table"):
        profiles = {
            profile["column_index"]: profile
            for profile in table["metadata"]["column_profiles"]
        }
        for entity_col in table["metadata"]["candidate_entity_columns"]:
            attribute_cols = [
                column["column_index"]
                for column in table["columns"]
                if column["column_index"] != entity_col
                and profiles[column["column_index"]]["non_empty_ratio"]
                >= min_column_non_empty_ratio
            ]
            for row in table["rows"]:
                entity_cell = get_cell(row, entity_col)
                entity = entity_by_title.get(clean_text(entity_cell.get("wiki_title")))
                if entity is None:
                    continue
                row_assets = assets_by_entity.get(entity["entity_id"], [])
                for attribute_col in attribute_cols:
                    attribute_name = get_column_name(table, attribute_col)
                    visible_cells = [
                        {
                            "name": get_column_name(table, column["column_index"]),
                            "value": clean_text(get_cell(row, column["column_index"]).get("text")),
                        }
                        for column in table["columns"]
                        if column["column_index"] != attribute_col
                        and clean_text(get_cell(row, column["column_index"]).get("text"))
                    ]
                    for asset in row_assets:
                        extractor = (
                            image_extractor
                            if asset["asset_type"] == "image"
                            else text_extractor
                        )
                        if extractor is None:
                            continue
                        result = extractor.extract(
                            attribute=attribute_name,
                            visible_cells=visible_cells,
                            asset=asset,
                        )
                        records.append(
                            {
                                "extraction_id": "ext_"
                                + stable_hash(
                                    table["source_table_id"],
                                    row["row_id"],
                                    asset["asset_id"],
                                    attribute_name,
                                ),
                                "source_table_id": table["source_table_id"],
                                "source_row_id": row["row_id"],
                                "entity_id": entity["entity_id"],
                                "asset_id": asset["asset_id"],
                                "asset_type": asset["asset_type"],
                                "attribute_name": attribute_name,
                                "value": result["value"],
                                "evidence": result["evidence"],
                                "prompt_version": PROMPT_VERSION,
                            }
                        )
    return records


def auto_check_recoveries(
    artifacts: dict[str, list[dict[str, Any]]],
    assets: list[dict[str, Any]],
    text_extractor: OpenAICompatibleExtractor | None,
    *,
    image_extractor: OpenAICompatibleExtractor | None = None,
    review_mode: str = "cascade",
    luna_extractor: OpenAICompatibleExtractor | None = None,
    terra_extractor: OpenAICompatibleExtractor | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Fail closed using local-only or local/Luna/Terra review."""
    if review_mode not in {"cascade", "local"}:
        raise ValueError("review_mode must be 'cascade' or 'local'")
    if review_mode == "cascade" and (
        luna_extractor is None or terra_extractor is None
    ):
        raise ValueError("cascade review requires both Luna and Terra extractors")

    def results_agree(left: str, right: str) -> bool:
        left = clean_text(left)
        right = clean_text(right)
        if not left or not right:
            return not left and not right
        return values_match(left, right)

    query_by_id = {
        query["table_id"]: query for query in artifacts["query_tables"]
    }
    asset_by_id = {asset["asset_id"]: asset for asset in assets}
    checked_recoveries: list[dict[str, Any]] = []
    supported_rows: dict[str, set[int]] = {}

    for recovery in progress(
        artifacts["evidence_recoveries"],
        desc="Check recoveries",
        unit="recovery",
    ):
        query_id = recovery["query_table_id"]
        query = query_by_id.get(query_id)
        asset = asset_by_id.get(recovery["evidence"]["asset_id"])
        if query is None or asset is None:
            continue
        query_row_id = int(recovery["query_row_id"])
        query_row = next(
            (
                row
                for row in query["rows"]
                if int(row["row_id"]) == query_row_id
            ),
            None,
        )
        if query_row is None:
            continue
        visible_cells = [
            {
                "name": clean_text(cell.get("column_name")),
                "value": clean_text(cell.get("text")),
            }
            for cell in query_row["cells"]
            if clean_text(cell.get("column_name"))
        ]
        extractor = (
            image_extractor
            if asset.get("asset_type") == "image"
            else text_extractor
        )
        if extractor is None:
            continue
        recovered = recovery["recovered_attribute"]
        local_result = extractor.extract(
            attribute=clean_text(recovered.get("column_name")),
            visible_cells=visible_cells,
            asset=asset,
        )
        local_value = clean_text(local_result.get("value"))
        luna_value: str | None = None
        terra_value: str | None = None
        terra_triggered = False
        if review_mode == "local":
            extracted_value = local_value
            decision_source = "primary_local"
        else:
            luna_result = luna_extractor.extract(
                attribute=clean_text(recovered.get("column_name")),
                visible_cells=visible_cells,
                asset=asset,
            )
            luna_value = clean_text(luna_result.get("value"))
            if results_agree(local_value, luna_value):
                extracted_value = luna_value
                decision_source = "local_luna_consensus"
            else:
                terra_triggered = True
                terra_result = terra_extractor.extract(
                    attribute=clean_text(recovered.get("column_name")),
                    visible_cells=visible_cells,
                    asset=asset,
                )
                terra_value = clean_text(terra_result.get("value"))
                extracted_value = terra_value
                decision_source = "terra_adjudication"
        claimed_value = clean_text(recovered.get("value"))
        if not values_match(extracted_value, claimed_value):
            continue
        checked_recoveries.append(
            {
                **recovery,
                "auto_check": {
                    "prompt_version": AUTO_CHECK_PROMPT_VERSION,
                    "input_policy": "query_visible_row_and_raw_evidence_only",
                    "review_policy": (
                        AUTO_CHECK_CASCADE_POLICY
                        if review_mode == "cascade"
                        else AUTO_CHECK_LOCAL_POLICY
                    ),
                    "decision_source": decision_source,
                    "extracted_value": extracted_value,
                    "primary_extracted_value": local_value,
                    "luna_triggered": review_mode == "cascade",
                    "luna_extracted_value": luna_value,
                    "luna_agrees_with_local": (
                        results_agree(local_value, luna_value)
                        if luna_value is not None
                        else None
                    ),
                    "terra_triggered": terra_triggered,
                    "terra_extracted_value": terra_value,
                    "verdict": "supported",
                },
            }
        )
        supported_rows.setdefault(query_id, set()).add(query_row_id)

    retained_query_ids: set[str] = set()
    retained_queries: list[dict[str, Any]] = []
    for query in artifacts["query_tables"]:
        hidden = list(query.get("hidden_attributes") or [])
        required = int(hidden[0].get("required_recovered_rows", 0)) if hidden else 0
        recovered_rows = len(supported_rows.get(query["table_id"], set()))
        if recovered_rows < required:
            continue
        updated_hidden = [
            {
                **item,
                "recovered_rows": recovered_rows,
                "recovered_value_ratio": (
                    recovered_rows / len(query["rows"])
                    if query["rows"]
                    else 0.0
                ),
            }
            for item in hidden
        ]
        retained_query_ids.add(query["table_id"])
        retained_queries.append({**query, "hidden_attributes": updated_hidden})

    retained_qrels = [
        qrel
        for qrel in artifacts["qrels"]
        if qrel["query_table_id"] in retained_query_ids
    ]
    retained_target_ids = {qrel["target_table_id"] for qrel in retained_qrels}
    return {
        **artifacts,
        "query_tables": retained_queries,
        "data_lake_tables": [
            table
            for table in artifacts["data_lake_tables"]
            if table["table_id"] in retained_target_ids
        ],
        "qrels": retained_qrels,
        "evidence_recoveries": [
            recovery
            for recovery in checked_recoveries
            if recovery["query_table_id"] in retained_query_ids
        ],
    }
