#!/usr/bin/env python
"""Cache frozen Qwen3-VL features in separate retrieval and Teacher tiers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch
from mmdd_stage1.features import normalize_object_type
from mmdd_stage1.models import structural_table_pool
from torch.nn import functional as F

PROMPT_VERSION = "role_modality_v2_object_only"
TEACHER_MANIFEST = "teacher_manifest.jsonl"
EMBEDDING_INSTRUCTIONS = {
    ("query", "table"): (
        "Represent this query table for directed multimodal joinability retrieval. "
        "Emphasize its visible entity and key columns, row values, and schema "
        "that identify what each row is about, so compatible target tables and bridge "
        "evidence can be found. Do not infer missing attributes or relationships not "
        "present in the table."
    ),
    ("query_row", "table"): (
        "Represent this single query row together with its table schema for "
        "assigning relevant multimodal evidence to that row. Emphasize the row's entity "
        "identity, key values, and qualifiers present in the cells. Do not infer "
        "missing attributes."
    ),
    ("target", "table"): (
        "Represent this candidate target table for directed joinability retrieval. "
        "Emphasize the entities and join-key values it covers, its schema and cell values, "
        "and the factual attributes it can provide to a compatible query table. Preserve "
        "distinctions between columns and do not assume a relationship to any particular "
        "query."
    ),
    ("evidence", "text"): (
        "Represent this text as independent bridge evidence for directed multimodal "
        "joinability retrieval. Emphasize explicitly stated entities, aliases, attributes, "
        "values, and relations that can connect a query-table row to a compatible target "
        "table. Do not assume a relationship to any particular table or add facts not "
        "expressed in the text."
    ),
    ("evidence", "image"): (
        "Represent this image as independent bridge evidence for directed multimodal "
        "joinability retrieval. Emphasize visually grounded entities, objects, scenes, "
        "text, attributes, and relations that can connect a query-table row to a "
        "compatible target table. Do not infer identities or facts that are not visible "
        "in the image."
    ),
}


def embedding_instructions(
    record: dict[str, Any],
    object_type: str,
    instruction_override: str | None = None,
) -> tuple[str, str | None, str]:
    """Return the object instruction, optional query-row instruction, and role."""

    role = record.get("embedding_role")
    if role is None:
        if object_type == "table":
            raise ValueError(
                f"{record.get('object_id')}: table objects must declare embedding_role"
            )
        else:
            role = "evidence"
    role = str(role)
    key = (role, object_type)
    if key not in EMBEDDING_INSTRUCTIONS:
        expected = ", ".join(
            f"{expected_role}/{expected_type}"
            for expected_role, expected_type in EMBEDDING_INSTRUCTIONS
            if expected_role != "query_row"
        )
        raise ValueError(
            f"{record.get('object_id')}: unsupported embedding role/modality "
            f"{role}/{object_type}; expected one of: {expected}"
        )

    object_instruction = str(
        record.get("instruction") or instruction_override or EMBEDDING_INSTRUCTIONS[key]
    )
    row_instruction = None
    if role == "query" and object_type == "table":
        row_instruction = str(
            record.get("row_instruction")
            or instruction_override
            or EMBEDDING_INSTRUCTIONS[("query_row", "table")]
        )
    return object_instruction, row_instruction, role


def _load_embedder_class(model_dir: Path):
    script = model_dir / "scripts" / "qwen3_vl_embedding.py"
    if not script.is_file():
        raise FileNotFoundError(f"Missing official Qwen embedding wrapper: {script}")
    spec = importlib.util.spec_from_file_location("_mmdd_qwen3_vl_embedding", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Qwen3VLEmbedder


@torch.inference_mode()
def encode_inputs(
    embedder: Any, items: list[dict[str, Any]]
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return embeddings, unpooled valid hidden states, and their token IDs."""

    conversations = [
        embedder.format_model_input(
            text=item.get("text"),
            image=item.get("image"),
            instruction=item.get("instruction"),
        )
        for item in items
    ]
    inputs = embedder._preprocess_inputs(conversations)
    inputs = {name: tensor.to(embedder.model.device) for name, tensor in inputs.items()}
    outputs = embedder.forward(inputs)
    hidden_states = outputs["last_hidden_state"]
    attention_mask = outputs["attention_mask"].bool()
    pooled = embedder._pooling_last(
        hidden_states,
        attention_mask.to(dtype=torch.long),
    )
    embeddings = F.normalize(pooled.float(), p=2, dim=-1)
    return [
        (
            embeddings[index].cpu(),
            hidden_states[index][attention_mask[index]].cpu(),
            inputs["input_ids"][index][attention_mask[index]].cpu(),
        )
        for index in range(hidden_states.shape[0])
    ]


def _table_token_groups(
    embedder: Any,
    item: dict[str, Any],
    parts: list[str],
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    text = item.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("A table requires non-empty text containing all table_parts")

    part_spans = []
    cursor = 0
    for part in parts:
        start = text.find(part, cursor)
        if start < 0:
            raise ValueError("table_parts must occur in order within the table text")
        part_spans.append((start, start + len(part)))
        cursor = start + len(part)

    conversation = embedder.format_model_input(
        text=text,
        image=item.get("image"),
        instruction=item.get("instruction"),
    )
    rendered = embedder.processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    text_start = rendered.rfind(text)
    if text_start < 0:
        raise ValueError("Qwen chat template did not preserve the serialized table text")
    rendered_spans = [(text_start + start, text_start + end) for start, end in part_spans]

    tokenized = None
    for add_special_tokens in (False, True):
        candidate = embedder.processor.tokenizer(
            rendered,
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=embedder.max_length,
            return_offsets_mapping=True,
        )
        if list(candidate["input_ids"]) == input_ids.tolist():
            tokenized = candidate
            break
    if tokenized is None:
        raise ValueError("Tokenizer offsets do not align with Qwen preprocessing")

    selected_indices = []
    groups = []
    for token_index, (start, end) in enumerate(tokenized["offset_mapping"]):
        if end <= start:
            continue
        for group, (part_start, part_end) in enumerate(rendered_spans):
            if end > part_start and start < part_end:
                selected_indices.append(token_index)
                groups.append(group)
                break
    if set(groups) != set(range(len(parts))):
        raise ValueError("Table truncation removed all tokens from at least one schema/row group")
    return torch.tensor(selected_indices, dtype=torch.long), torch.tensor(groups, dtype=torch.long)


def _resolve_image(record: dict[str, Any], input_dir: Path) -> str | None:
    value = record.get("image")
    if value is None:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = input_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    return str(path)


def build_object_features(
    embedder: Any,
    record: dict[str, Any],
    *,
    input_dir: Path,
    instruction: str | None,
    storage_dtype: torch.dtype,
    include_hidden: bool = True,
    include_row_embeddings: bool = True,
) -> dict[str, torch.Tensor]:
    object_type = normalize_object_type(str(record["object_type"]))
    object_instruction, row_instruction, embedding_role = embedding_instructions(
        record, object_type, instruction
    )
    parts = record.get("table_parts") if object_type == "table" else None
    if object_type == "table" and (
        not isinstance(parts, list)
        or not parts
        or not all(isinstance(part, str) and part.strip() for part in parts)
    ):
        raise ValueError(
            f"{record.get('object_id')}: table_parts must contain schema text followed by example-row texts"
        )
    item = {
        "text": "\n".join(parts) if parts is not None else record.get("text"),
        "image": _resolve_image(record, input_dir),
        "instruction": object_instruction,
    }
    embedding, hidden_states, input_ids = encode_inputs(embedder, [item])[0]
    payload = {"embedding": embedding.float()}

    if object_type != "table":
        if include_hidden:
            payload["hidden_states"] = hidden_states.to(dtype=storage_dtype)
        return payload

    assert isinstance(parts, list)
    encoded_parts = parts
    if include_hidden:
        try:
            indices, groups = _table_token_groups(embedder, item, encoded_parts, input_ids)
        except ValueError as error:
            if str(error) != "Table truncation removed all tokens from at least one schema/row group":
                raise
            for max_chars in (4096, 2048, 1024, 512, 256, 128):
                encoded_parts = [part[:max_chars].rstrip() for part in parts]
                if encoded_parts == parts:
                    continue
                item["text"] = "\n".join(encoded_parts)
                embedding, hidden_states, input_ids = encode_inputs(embedder, [item])[0]
                try:
                    indices, groups = _table_token_groups(
                        embedder, item, encoded_parts, input_ids
                    )
                except ValueError as retry_error:
                    if str(retry_error) == str(error):
                        continue
                    raise
                print(
                    json.dumps(
                        {
                            "object_id": str(record["object_id"]),
                            "event": "table_parts_truncated",
                            "max_chars_per_part": max_chars,
                        }
                    )
                )
                break
            else:
                raise
            payload["embedding"] = embedding.float()
        # Legacy caches loaded storage-dtype tokens as float32 before pooling.
        # Preserve that numerical order, then keep the much smaller pooled table
        # representation in float32 so no second quantization is introduced.
        selected_hidden = hidden_states.index_select(0, indices).to(
            dtype=storage_dtype
        ).float()
        pooled_hidden = structural_table_pool(selected_hidden, groups)
        payload["hidden_states"] = pooled_hidden
        payload["token_groups"] = torch.arange(len(pooled_hidden), dtype=torch.long)

    if embedding_role == "query" and include_row_embeddings:
        routing_outputs = encode_inputs(
            embedder,
            [
                {
                    "text": text,
                    "instruction": row_instruction,
                }
                for text in (
                    f"{encoded_parts[0]}\n{row}" for row in encoded_parts[1:]
                )
            ],
        )
        payload["row_embeddings"] = torch.stack(
            [embedding for embedding, _, _ in routing_outputs]
        )

    return payload


def teacher_object_ids(
    paths: list[Path], *, split: str | None = "train"
) -> set[str]:
    """Collect every object referenced by edge-list or target-list JSONL files."""

    object_ids: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError(f"{path}:{line_number}: expected a JSON object")
                if split is not None and record.get("split") != split:
                    continue
                object_ids.add(str(record["query_id"]))
                if "candidate_ids" in record:
                    object_ids.update(str(value) for value in record["candidate_ids"])
                    continue
                for candidate in record.get("candidates", []):
                    object_ids.add(str(candidate["target_id"]))
                    object_ids.update(
                        str(value) for value in candidate.get("evidence_ids", [])
                    )
    return object_ids


def _completed_records(manifest: Path) -> dict[str, dict[str, Any]]:
    if not manifest.exists():
        return {}
    records = {}
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                record = json.loads(line)
                object_id = str(record["object_id"])
                if object_id in records:
                    raise ValueError(
                        f"{manifest}:{line_number}: duplicate object_id {object_id!r}"
                    )
                records[object_id] = record
    return records


def _source_fingerprint(record: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input_jsonl).resolve()
    output_dir = Path(args.output_dir)
    object_dir = output_dir / "objects"
    object_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.jsonl"
    completed = _completed_records(manifest)
    teacher_dir = output_dir / "teacher_objects"
    teacher_manifest = output_dir / TEACHER_MANIFEST
    completed_teacher = _completed_records(teacher_manifest)
    teacher_paths = [Path(value).resolve() for value in args.teacher_data]
    selected_teacher_ids = teacher_object_ids(
        teacher_paths,
        split=(
            None
            if args.teacher_split == "all"
            else args.teacher_split
        ),
    )
    if selected_teacher_ids:
        teacher_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.model_dir).resolve()
    metadata = {
        "format_version": 5,
        "model_dir": str(model_dir),
        "dtype": args.dtype,
        "prompt_version": PROMPT_VERSION,
        "embedding_instructions": {
            f"{role}_{object_type}": instruction
            for (role, object_type), instruction in EMBEDDING_INSTRUCTIONS.items()
        },
        "instruction_override": args.instruction,
        "feature_tiers": ["retrieval", "teacher"],
        "table_pooling": "prepooled_schema_rows",
    }
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing_metadata != metadata:
            raise ValueError(f"{metadata_path}: cache settings differ from this run")
    else:
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending_base_ids = set()
    pending_teacher_ids = set()
    base_skipped = 0
    teacher_skipped = 0
    seen = set()
    with input_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            object_id = str(record["object_id"])
            if object_id in seen:
                raise ValueError(f"{input_path}:{line_number}: duplicate object_id {object_id!r}")
            seen.add(object_id)
            object_type = normalize_object_type(str(record["object_type"]))
            source_fingerprint = _source_fingerprint(record)
            if object_id in completed:
                completed_record = completed[object_id]
                completed_path = output_dir / completed_record["feature_path"]
                if not completed_path.is_file():
                    raise FileNotFoundError(f"Manifest references a missing feature file: {completed_path}")
                if completed_record.get("source_fingerprint") != source_fingerprint:
                    raise ValueError(f"{object_id}: input changed after this feature was cached")
                if completed_record.get("object_type") != object_type:
                    raise ValueError(f"{object_id}: object type changed after this feature was cached")
                base_skipped += 1
            else:
                pending_base_ids.add(object_id)
            if object_id in selected_teacher_ids:
                if object_id in completed_teacher:
                    completed_record = completed_teacher[object_id]
                    completed_path = output_dir / completed_record["teacher_feature_path"]
                    if not completed_path.is_file():
                        raise FileNotFoundError(
                            f"Teacher manifest references a missing feature file: {completed_path}"
                        )
                    if completed_record.get("source_fingerprint") != source_fingerprint:
                        raise ValueError(
                            f"{object_id}: input changed after its Teacher feature was cached"
                        )
                    if completed_record.get("object_type") != object_type:
                        raise ValueError(
                            f"{object_id}: object type changed after its Teacher feature was cached"
                        )
                    teacher_skipped += 1
                else:
                    pending_teacher_ids.add(object_id)

    missing_teacher_ids = selected_teacher_ids - seen
    if missing_teacher_ids:
        preview = ", ".join(sorted(missing_teacher_ids)[:10])
        raise KeyError(f"Teacher data references objects absent from {input_path}: {preview}")

    if not pending_base_ids and not pending_teacher_ids:
        print(
            json.dumps(
                {
                    "base_objects_written": 0,
                    "base_objects_skipped": base_skipped,
                    "teacher_objects_written": 0,
                    "teacher_objects_skipped": teacher_skipped,
                    "output_dir": str(output_dir),
                },
                indent=2,
            )
        )
        return

    embedder_class = _load_embedder_class(model_dir)
    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    embedder = embedder_class(model_name_or_path=str(model_dir), torch_dtype=torch_dtype)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    embedder.model.to(torch.device(device))
    embedder.model.eval()

    base_written = 0
    teacher_written = 0
    with (
        input_path.open(encoding="utf-8") as source,
        manifest.open("a", encoding="utf-8") as manifest_handle,
        teacher_manifest.open("a", encoding="utf-8") as teacher_manifest_handle,
    ):
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            object_id = str(record["object_id"])
            needs_base = object_id in pending_base_ids
            needs_teacher = object_id in pending_teacher_ids
            if not needs_base and not needs_teacher:
                continue
            object_type = normalize_object_type(str(record["object_type"]))
            source_fingerprint = _source_fingerprint(record)
            payload = build_object_features(
                embedder,
                record,
                input_dir=input_path.parent,
                instruction=args.instruction,
                storage_dtype=torch_dtype,
                include_hidden=needs_teacher,
                include_row_embeddings=needs_base,
            )
            name = hashlib.sha256(object_id.encode("utf-8")).hexdigest() + ".pt"
            if needs_base:
                base_payload = {"embedding": payload["embedding"]}
                if "row_embeddings" in payload:
                    base_payload["row_embeddings"] = payload["row_embeddings"]
                relative_path = Path("objects") / name
                destination = output_dir / relative_path
                temporary = destination.with_suffix(".pt.tmp")
                torch.save(base_payload, temporary)
                temporary.replace(destination)
                manifest_record = {
                    "object_id": object_id,
                    "object_type": object_type,
                    "feature_path": relative_path.as_posix(),
                    "source_fingerprint": source_fingerprint,
                }
                manifest_handle.write(
                    json.dumps(manifest_record, ensure_ascii=False) + "\n"
                )
                manifest_handle.flush()
                base_written += 1
            if needs_teacher:
                teacher_payload = {"hidden_states": payload["hidden_states"]}
                if "token_groups" in payload:
                    teacher_payload["token_groups"] = payload["token_groups"]
                relative_path = Path("teacher_objects") / name
                destination = output_dir / relative_path
                temporary = destination.with_suffix(".pt.tmp")
                torch.save(teacher_payload, temporary)
                temporary.replace(destination)
                manifest_record = {
                    "object_id": object_id,
                    "object_type": object_type,
                    "teacher_feature_path": relative_path.as_posix(),
                    "source_fingerprint": source_fingerprint,
                }
                teacher_manifest_handle.write(
                    json.dumps(manifest_record, ensure_ascii=False) + "\n"
                )
                teacher_manifest_handle.flush()
                teacher_written += 1
            print(
                json.dumps(
                    {
                        "object_id": object_id,
                        "base_written": base_written,
                        "base_skipped": base_skipped,
                        "teacher_written": teacher_written,
                        "teacher_skipped": teacher_skipped,
                    }
                )
            )

    print(
        json.dumps(
            {
                "base_objects_written": base_written,
                "base_objects_skipped": base_skipped,
                "teacher_objects_written": teacher_written,
                "teacher_objects_skipped": teacher_skipped,
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", default="hf_models/Qwen3-VL-Embedding-8B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--teacher-data",
        nargs="+",
        default=[],
        help=(
            "Edge-list and target-list JSONL files whose referenced objects need "
            "Teacher hidden features. Re-run with new hard-negative files to add this tier."
        ),
    )
    parser.add_argument(
        "--teacher-split",
        default="train",
        help="Only cache Teacher features referenced by this split; use 'all' for every record.",
    )
    parser.add_argument(
        "--instruction",
        help="Explicitly override all role- and modality-specific embedding instructions.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
