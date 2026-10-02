#!/usr/bin/env python
"""Cache frozen Qwen3-VL features in separate retrieval and Teacher tiers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.nn import functional as F

from mmdd_progress import progress

TYPE_ALIASES = {
    "table": "table",
    "table_fragment": "table",
    "text": "text",
    "text_asset": "text",
    "image": "image",
    "image_asset": "image",
}


def normalize_object_type(value: str) -> str:
    try:
        return TYPE_ALIASES[value]
    except KeyError as exc:
        raise ValueError(f"Unknown object type {value!r}; expected one of: {', '.join(sorted(TYPE_ALIASES))}") from exc


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


def structural_table_pool_with_groups(
    hidden_states: torch.Tensor,
    token_groups: torch.Tensor | None,
    tokens_per_group: int = 1,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pool each schema/row into ordered contiguous semantic segments."""

    if hidden_states.shape[0] == 0:
        raise ValueError("A table must contain at least one hidden-state token")
    if tokens_per_group <= 0:
        raise ValueError("tokens_per_group must be positive")
    if token_groups is None:
        return hidden_states, None
    pooled = []
    pooled_groups = []
    for group in torch.unique(token_groups, sorted=True):
        values = hidden_states[token_groups == group]
        chunks = torch.tensor_split(values, min(tokens_per_group, values.shape[0]))
        pooled.extend(chunk.mean(dim=0) for chunk in chunks)
        pooled_groups.extend([int(group)] * len(chunks))
    return torch.stack(pooled), torch.tensor(
        pooled_groups,
        dtype=token_groups.dtype,
        device=token_groups.device,
    )


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
def encode_preprocessed_inputs(
    embedder: Any,
    inputs: dict[str, torch.Tensor],
    *,
    include_hidden: bool = True,
) -> list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]]:
    """Run already-preprocessed inputs and return their requested CPU payload."""

    inputs = {name: tensor.to(embedder.model.device) for name, tensor in inputs.items()}
    outputs = embedder.forward(inputs)
    hidden_states = outputs["last_hidden_state"]
    attention_mask = outputs["attention_mask"].bool()
    pooled = embedder._pooling_last(
        hidden_states,
        attention_mask.to(dtype=torch.long),
    )
    embeddings = F.normalize(pooled.float(), p=2, dim=-1)
    if not include_hidden:
        return [
            (embeddings[index].cpu(), None, None)
            for index in range(hidden_states.shape[0])
        ]
    return [
        (
            embeddings[index].cpu(),
            hidden_states[index][attention_mask[index]].cpu(),
            inputs["input_ids"][index][attention_mask[index]].cpu(),
        )
        for index in range(hidden_states.shape[0])
    ]


def preprocess_input_items(
    embedder: Any, items: list[dict[str, Any]]
) -> dict[str, torch.Tensor]:
    """Format and preprocess model items on CPU."""

    conversations = [
        embedder.format_model_input(
            text=item.get("text"),
            image=item.get("image"),
            instruction=item.get("instruction"),
        )
        for item in items
    ]
    return embedder._preprocess_inputs(conversations)


def encode_inputs(
    embedder: Any,
    items: list[dict[str, Any]],
    *,
    include_hidden: bool = True,
) -> list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]]:
    """Return embeddings and, when requested, valid hidden states and token IDs."""

    return encode_preprocessed_inputs(
        embedder,
        preprocess_input_items(embedder, items),
        include_hidden=include_hidden,
    )


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
    table_row_batch_size: int = 8,
    table_tokens_per_group: int = 1,
) -> dict[str, torch.Tensor]:
    if table_row_batch_size <= 0:
        raise ValueError("table_row_batch_size must be positive")
    if table_tokens_per_group <= 0:
        raise ValueError("table_tokens_per_group must be positive")
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
    embedding, hidden_states, input_ids = encode_inputs(
        embedder,
        [item],
        include_hidden=include_hidden,
    )[0]
    payload = {"embedding": embedding.float()}

    if object_type != "table":
        if include_hidden:
            payload["hidden_states"] = hidden_states.to(dtype=storage_dtype)
        return payload

    assert isinstance(parts, list)
    if include_hidden:
        assert hidden_states is not None
        assert input_ids is not None
        try:
            indices, groups = _table_token_groups(embedder, item, parts, input_ids)
        except ValueError as error:
            if (
                str(error)
                != "Table truncation removed all tokens from at least one schema/row group"
            ):
                raise
            for max_chars in (4096, 2048, 1024, 512, 256, 128):
                truncated_parts = [part[:max_chars].rstrip() for part in parts]
                if truncated_parts == parts:
                    continue
                teacher_item = {**item, "text": "\n".join(truncated_parts)}
                _, hidden_states, input_ids = encode_inputs(embedder, [teacher_item])[0]
                assert hidden_states is not None
                assert input_ids is not None
                try:
                    indices, groups = _table_token_groups(
                        embedder, teacher_item, truncated_parts, input_ids
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
        # Legacy caches loaded storage-dtype tokens as float32 before pooling.
        # Preserve that numerical order, then keep the much smaller pooled table
        # representation in float32 so no second quantization is introduced.
        selected_hidden = hidden_states.index_select(0, indices).to(
            dtype=storage_dtype
        ).float()
        pooled_hidden, pooled_groups = structural_table_pool_with_groups(
            selected_hidden,
            groups,
            table_tokens_per_group,
        )
        payload["hidden_states"] = pooled_hidden
        if table_tokens_per_group > 1:
            assert pooled_groups is not None
            payload["token_groups"] = pooled_groups

    if embedding_role == "query" and include_row_embeddings:
        routing_items = [
            {
                "text": f"{parts[0]}\n{row}",
                "instruction": row_instruction,
            }
            for row in parts[1:]
        ]
        routing_outputs = [
            output
            for start in range(0, len(routing_items), table_row_batch_size)
            for output in encode_inputs(
                embedder,
                routing_items[start : start + table_row_batch_size],
                include_hidden=False,
            )
        ]
        payload["row_embeddings"] = torch.stack(
            [embedding for embedding, _, _ in routing_outputs]
        )

    return payload


def build_base_object_features_batch(
    embedder: Any,
    records: list[dict[str, Any]],
    *,
    input_dir: Path,
    instruction: str | None,
    preprocessed_inputs: dict[str, torch.Tensor] | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Build retrieval-only features for a batch of non-table objects."""

    items = []
    for record in records:
        object_type = normalize_object_type(str(record["object_type"]))
        if object_type == "table":
            raise ValueError("Object batching only supports non-table retrieval features")
        object_instruction, _row_instruction, _embedding_role = embedding_instructions(
            record, object_type, instruction
        )
        items.append(
            {
                "text": record.get("text"),
                "image": _resolve_image(record, input_dir),
                "instruction": object_instruction,
            }
        )

    outputs = (
        encode_inputs(embedder, items, include_hidden=False)
        if preprocessed_inputs is None
        else encode_preprocessed_inputs(
            embedder,
            preprocessed_inputs,
            include_hidden=False,
        )
    )
    if len(outputs) != len(records):
        # The official wrapper converts a whole malformed vision batch into one
        # NULL item. Retry those rare batches individually so one bad image does
        # not change the features of its valid neighbors.
        print(
            json.dumps(
                {
                    "event": "base_batch_fell_back_to_individual",
                    "batch_size": len(records),
                }
            )
        )
        if preprocessed_inputs is not None:
            return []
        return [
            build_object_features(
                embedder,
                record,
                input_dir=input_dir,
                instruction=instruction,
                storage_dtype=torch.float32,
                include_hidden=False,
                include_row_embeddings=False,
            )
            for record in records
        ]
    return [{"embedding": embedding.float()} for embedding, _, _ in outputs]


def preprocess_base_object_batch(
    embedder: Any,
    records: list[dict[str, Any]],
    *,
    input_dir: Path,
    instruction: str | None,
) -> dict[str, torch.Tensor]:
    """Prepare a retrieval-only non-table batch without touching the GPU."""

    items = []
    for record in records:
        object_type = normalize_object_type(str(record["object_type"]))
        object_instruction, _row_instruction, _embedding_role = embedding_instructions(
            record, object_type, instruction
        )
        items.append(
            {
                "text": record.get("text"),
                "image": _resolve_image(record, input_dir),
                "instruction": object_instruction,
            }
        )
    return preprocess_input_items(embedder, items)


def _base_object_batch_cost(
    record: dict[str, Any],
    *,
    input_dir: Path,
    max_image_pixels: int,
) -> int:
    """Estimate padded sequence cost for grouping similarly sized objects."""

    object_type = normalize_object_type(str(record["object_type"]))
    if object_type == "text":
        return len(str(record.get("text") or ""))
    if object_type != "image":
        raise ValueError("Batch cost is only defined for text and image objects")
    image_path = _resolve_image(record, input_dir)
    assert image_path is not None
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except (OSError, Image.DecompressionBombError):
        return -1
    return min(width * height, max_image_pixels)


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
                if "object_id" in record and "query_id" not in record:
                    object_ids.add(str(record["object_id"]))
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
    teacher_output_dir = (
        Path(args.teacher_output_dir)
        if getattr(args, "teacher_output_dir", None)
        else output_dir
    )
    teacher_dir = teacher_output_dir / "teacher_objects"
    teacher_manifest = teacher_output_dir / TEACHER_MANIFEST
    completed_teacher: dict[str, tuple[dict[str, Any], Path]] = {}
    for root in dict.fromkeys([output_dir, teacher_output_dir]):
        for object_id, record in _completed_records(
            root / TEACHER_MANIFEST
        ).items():
            if object_id in completed_teacher:
                previous, _previous_root = completed_teacher[object_id]
                if previous != record:
                    raise ValueError(
                        f"Teacher staging record for {object_id!r} conflicts with the main cache"
                    )
                continue
            completed_teacher[object_id] = (record, root)
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
    table_tokens_per_group = getattr(args, "table_tokens_per_group", 1)
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
    if table_tokens_per_group > 1:
        metadata.update(
            {
                "format_version": 6,
                "table_pooling": "contiguous_mean_segments",
                "table_tokens_per_group": table_tokens_per_group,
            }
        )
    for metadata_root in dict.fromkeys([output_dir, teacher_output_dir]):
        metadata_root.mkdir(parents=True, exist_ok=True)
        metadata_path = metadata_root / "metadata.json"
        if metadata_path.exists():
            existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if existing_metadata != metadata:
                raise ValueError(f"{metadata_path}: cache settings differ from this run")
        else:
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    pending_base_ids = set()
    pending_teacher_ids = set()
    base_skipped = 0
    teacher_skipped = 0
    seen = set()
    with input_path.open(encoding="utf-8") as source:
        lines = progress(source, desc="Validate feature cache", unit="object")
        for line_number, line in enumerate(lines, 1):
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
                    completed_record, completed_root = completed_teacher[object_id]
                    completed_path = completed_root / completed_record["teacher_feature_path"]
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
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(torch.device(device))
    embedder = embedder_class(model_name_or_path=str(model_dir), torch_dtype=torch_dtype)
    embedder.model.to(torch.device(device))
    embedder.model.eval()

    base_written = 0
    teacher_written = 0
    object_batch_size = getattr(args, "object_batch_size", 1)
    if object_batch_size <= 0:
        raise ValueError("object_batch_size must be positive")
    object_batch_buffer_size = getattr(args, "object_batch_buffer_size", 32)
    if object_batch_buffer_size < object_batch_size:
        raise ValueError("object_batch_buffer_size must be at least object_batch_size")
    prefetch_base_objects = getattr(args, "prefetch_base_objects", False)
    base_prefetch_workers = getattr(args, "base_prefetch_workers", 1)
    if base_prefetch_workers <= 0:
        raise ValueError("base_prefetch_workers must be positive")
    if prefetch_base_objects and object_batch_buffer_size < 2:
        raise ValueError("prefetch_base_objects requires object_batch_buffer_size>=2")
    async_write_workers = getattr(args, "async_write_workers", 0)
    async_write_queue_size = getattr(args, "async_write_queue_size", 4)
    if async_write_workers < 0:
        raise ValueError("async_write_workers cannot be negative")
    if async_write_queue_size <= 0:
        raise ValueError("async_write_queue_size must be positive")
    writer_context = (
        ThreadPoolExecutor(max_workers=async_write_workers)
        if async_write_workers
        else nullcontext(None)
    )
    with (
        writer_context as writer_executor,
        input_path.open(encoding="utf-8") as source,
        manifest.open("a", encoding="utf-8") as manifest_handle,
        teacher_manifest.open("a", encoding="utf-8") as teacher_manifest_handle,
    ):
        lines = progress(
            (line for line in source if line.strip()),
            total=len(seen),
            desc="Cache Stage-1 features",
            unit="object",
        )
        pending_writes = []

        def save_payload_files(
            writes: list[tuple[Path, dict[str, torch.Tensor]]],
        ) -> None:
            for destination, saved_payload in writes:
                temporary = destination.with_suffix(".pt.tmp")
                torch.save(saved_payload, temporary)
                temporary.replace(destination)

        def commit_write(
            base_record: dict[str, Any] | None,
            teacher_record: dict[str, Any] | None,
        ) -> None:
            nonlocal base_written, teacher_written
            if base_record is not None:
                manifest_handle.write(json.dumps(base_record, ensure_ascii=False) + "\n")
                manifest_handle.flush()
                base_written += 1
            if teacher_record is not None:
                teacher_manifest_handle.write(
                    json.dumps(teacher_record, ensure_ascii=False) + "\n"
                )
                teacher_manifest_handle.flush()
                teacher_written += 1
            lines.set_postfix(
                base=base_written,
                teacher=teacher_written,
                skipped=base_skipped + teacher_skipped,
            )

        def finish_oldest_write() -> None:
            future, base_record, teacher_record = pending_writes.pop(0)
            future.result()
            commit_write(base_record, teacher_record)

        def write_payload(
            record: dict[str, Any],
            object_type: str,
            source_fingerprint: str,
            *,
            needs_base: bool,
            needs_teacher: bool,
            payload: dict[str, torch.Tensor],
        ) -> None:
            object_id = str(record["object_id"])
            name = hashlib.sha256(object_id.encode("utf-8")).hexdigest() + ".pt"
            writes = []
            base_record = None
            teacher_record = None
            if needs_base:
                base_payload = {"embedding": payload["embedding"]}
                if "row_embeddings" in payload:
                    base_payload["row_embeddings"] = payload["row_embeddings"]
                relative_path = Path("objects") / name
                destination = output_dir / relative_path
                writes.append((destination, base_payload))
                base_record = {
                    "object_id": object_id,
                    "object_type": object_type,
                    "feature_path": relative_path.as_posix(),
                    "source_fingerprint": source_fingerprint,
                }
            if needs_teacher:
                teacher_payload = {"hidden_states": payload["hidden_states"]}
                if "token_groups" in payload:
                    teacher_payload["token_groups"] = payload["token_groups"]
                relative_path = Path("teacher_objects") / name
                destination = teacher_output_dir / relative_path
                writes.append((destination, teacher_payload))
                teacher_record = {
                    "object_id": object_id,
                    "object_type": object_type,
                    "teacher_feature_path": relative_path.as_posix(),
                    "source_fingerprint": source_fingerprint,
                }
            if writer_executor is None:
                save_payload_files(writes)
                commit_write(base_record, teacher_record)
                return
            pending_writes.append(
                (writer_executor.submit(save_payload_files, writes), base_record, teacher_record)
            )
            if len(pending_writes) >= async_write_queue_size:
                finish_oldest_write()

        base_batch: list[tuple[dict[str, Any], str, str]] = []

        def flush_base_batch() -> None:
            if not base_batch:
                return
            ordered = sorted(
                base_batch,
                key=lambda entry: _base_object_batch_cost(
                    entry[0],
                    input_dir=input_path.parent,
                    max_image_pixels=int(getattr(embedder, "max_pixels", 2**63 - 1)),
                ),
            )
            chunks = [
                ordered[start : start + object_batch_size]
                for start in range(0, len(ordered), object_batch_size)
            ]

            def prepare(entries: list[tuple[dict[str, Any], str, str]]):
                return preprocess_base_object_batch(
                    embedder,
                    [entry[0] for entry in entries],
                    input_dir=input_path.parent,
                    instruction=args.instruction,
                )

            with ThreadPoolExecutor(max_workers=base_prefetch_workers) as executor:
                futures = {}
                next_to_submit = 0

                def fill_prefetch_queue() -> None:
                    nonlocal next_to_submit
                    while (
                        prefetch_base_objects
                        and next_to_submit < len(chunks)
                        and len(futures) < base_prefetch_workers + 1
                    ):
                        futures[next_to_submit] = executor.submit(
                            prepare, chunks[next_to_submit]
                        )
                        next_to_submit += 1

                fill_prefetch_queue()
                for index, entries in enumerate(chunks):
                    if not prefetch_base_objects:
                        prepared_inputs = None
                    else:
                        prepared_inputs = futures.pop(index).result()
                        fill_prefetch_queue()
                    payloads = build_base_object_features_batch(
                        embedder,
                        [entry[0] for entry in entries],
                        input_dir=input_path.parent,
                        instruction=args.instruction,
                        preprocessed_inputs=prepared_inputs,
                    )
                    if not payloads:
                        # A malformed image makes the official wrapper collapse
                        # a whole batch into one NULL item. Finish queued CPU
                        # preprocessing before retrying this batch individually.
                        for pending in futures.values():
                            pending.result()
                        payloads = [
                            build_object_features(
                                embedder,
                                entry[0],
                                input_dir=input_path.parent,
                                instruction=args.instruction,
                                storage_dtype=torch_dtype,
                                include_hidden=False,
                                include_row_embeddings=False,
                            )
                            for entry in entries
                        ]
                    for (record, object_type, source_fingerprint), payload in zip(
                        entries, payloads, strict=True
                    ):
                        write_payload(
                            record,
                            object_type,
                            source_fingerprint,
                            needs_base=True,
                            needs_teacher=False,
                            payload=payload,
                        )
            base_batch.clear()

        for line in lines:
            record = json.loads(line)
            object_id = str(record["object_id"])
            needs_base = object_id in pending_base_ids
            needs_teacher = object_id in pending_teacher_ids
            if not needs_base and not needs_teacher:
                continue
            object_type = normalize_object_type(str(record["object_type"]))
            source_fingerprint = _source_fingerprint(record)
            batchable = (
                (object_batch_size > 1 or prefetch_base_objects)
                and needs_base
                and not needs_teacher
                and object_type != "table"
            )
            if batchable:
                if base_batch and base_batch[-1][1] != object_type:
                    flush_base_batch()
                base_batch.append((record, object_type, source_fingerprint))
                if len(base_batch) == object_batch_buffer_size:
                    flush_base_batch()
                continue
            payload = build_object_features(
                embedder,
                record,
                input_dir=input_path.parent,
                instruction=args.instruction,
                storage_dtype=torch_dtype,
                include_hidden=needs_teacher,
                include_row_embeddings=needs_base,
                table_row_batch_size=getattr(args, "table_row_batch_size", 8),
                table_tokens_per_group=table_tokens_per_group,
            )
            write_payload(
                record,
                object_type,
                source_fingerprint,
                needs_base=needs_base,
                needs_teacher=needs_teacher,
                payload=payload,
            )
        flush_base_batch()
        while pending_writes:
            finish_oldest_write()

    print(
        json.dumps(
            {
                "base_objects_written": base_written,
                "base_objects_skipped": base_skipped,
                "teacher_objects_written": teacher_written,
                "teacher_objects_skipped": teacher_skipped,
                "output_dir": str(output_dir),
                "teacher_output_dir": str(teacher_output_dir),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--teacher-output-dir",
        help=(
            "Optional staging directory for Teacher-only features. The base cache "
            "is read from --output-dir and can be merged after parallel GPU runs."
        ),
    )
    parser.add_argument("--model-dir", default="hf_models/Qwen3-VL-Embedding-8B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--table-row-batch-size",
        type=int,
        default=8,
        help="Maximum query-table rows encoded together for routing embeddings.",
    )
    parser.add_argument(
        "--object-batch-size",
        type=int,
        default=1,
        help="Batch retrieval-only non-table objects; Teacher and table objects remain individual.",
    )
    parser.add_argument(
        "--object-batch-buffer-size",
        type=int,
        default=32,
        help="Sort this many retrieval-only objects by estimated sequence cost before batching.",
    )
    parser.add_argument(
        "--prefetch-base-objects",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overlap CPU preprocessing of the next base-only object with the current GPU forward.",
    )
    parser.add_argument(
        "--base-prefetch-workers",
        type=int,
        default=1,
        help="CPU preprocessing workers used by --prefetch-base-objects.",
    )
    parser.add_argument(
        "--async-write-workers",
        type=int,
        default=0,
        help="CPU workers that overlap atomic feature writes with GPU inference.",
    )
    parser.add_argument(
        "--async-write-queue-size",
        type=int,
        default=4,
        help="Maximum number of scheduled feature writes kept in flight.",
    )
    parser.add_argument(
        "--table-tokens-per-group",
        type=int,
        default=1,
        help="Keep this many contiguous pooled tokens per table schema/row group.",
    )
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
    args = parser.parse_args()
    if args.table_row_batch_size <= 0:
        parser.error("--table-row-batch-size must be positive")
    if args.object_batch_size <= 0:
        parser.error("--object-batch-size must be positive")
    if args.object_batch_buffer_size < args.object_batch_size:
        parser.error("--object-batch-buffer-size must be at least --object-batch-size")
    if args.base_prefetch_workers <= 0:
        parser.error("--base-prefetch-workers must be positive")
    if args.async_write_workers < 0:
        parser.error("--async-write-workers cannot be negative")
    if args.async_write_queue_size <= 0:
        parser.error("--async-write-queue-size must be positive")
    if args.prefetch_base_objects and args.object_batch_buffer_size < 2:
        parser.error("--prefetch-base-objects requires --object-batch-buffer-size >= 2")
    if args.table_tokens_per_group <= 0:
        parser.error("--table-tokens-per-group must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
