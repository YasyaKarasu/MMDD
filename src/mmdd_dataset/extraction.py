from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

import requests

from .utils import clean_text, get_cell, get_column_name, stable_hash


PROMPT_VERSION = "leave_one_attribute_out_v1"


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
        entity: str,
        attribute: str,
        context: list[dict[str, str]],
        asset: dict[str, Any],
    ) -> dict[str, str]:
        prompt = (
            "Extract one attribute from the evidence. The target attribute is omitted "
            "from the row context, so do not infer it from a supplied answer. Return JSON "
            "only as {\"value\": \"...\", \"evidence\": \"...\"}. Use an empty value "
            "when the evidence does not state the answer.\n\n"
            f"Entity: {entity}\nTarget attribute: {attribute}\n"
            f"Other row attributes: {json.dumps(context, ensure_ascii=False)}\n"
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
    for table in tables:
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
                    context = [
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
                            entity=clean_text(entity_cell.get("text")),
                            attribute=attribute_name,
                            context=context,
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
