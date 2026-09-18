"""Compressed frozen Qwen cache: z (float32 global) + C (float16, 8 summaries).

Spec section 4.  One frozen Qwen forward per object yields both the official
last-valid-token pooled vector and the eight structural summaries; full-length
hidden states are never persisted.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .config import ConfigError
from .data import IMAGE_KINDS, KIND_GLOBAL, KIND_ROW_GROUP, KIND_SCHEMA, clean_text
from .timing import Timing
from .util import (
    byte_order,
    log_line,
    read_jsonl,
    sha256_path,
    sha256_text,
    stable_digest,
    write_json,
    write_jsonl,
)

PROMPT_VERSION = "role_modality_v2_object_only"
# SOURCE_AUDIT.md pins this hash for the repository's cache_stage1_features.py,
# whose EMBEDDING_INSTRUCTIONS text and role rules are copied into this module.
# The official Qwen wrapper is a separate file: its hash is measured at runtime
# and recorded in the cache fingerprint (never a hardcoded constant).
SOURCE_AUDIT_INSTRUCTIONS_ORIGIN_SHA256 = (
    "1e2a8fa2b1280a873dd72e7ad579d3e3a16cebc66803a5c0335f700722231ba2"
)

MODALITY_TABLE = 0
MODALITY_TEXT = 1
MODALITY_IMAGE = 2

TABLE_KINDS = (KIND_GLOBAL, KIND_SCHEMA) + (KIND_ROW_GROUP,) * 7
TEXT_KINDS = (KIND_GLOBAL,) + IMAGE_KINDS
IMAGE_KINDS_FULL = (KIND_GLOBAL,) + IMAGE_KINDS

# Copied verbatim from the frozen prompt contract (spec section 4.2).  The run
# records the SHA256 of this mapping in the cache receipt.
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
INSTRUCTIONS_SHA256 = sha256_text(
    json.dumps(
        {f"{k[0]}/{k[1]}": v for k, v in sorted(EMBEDDING_INSTRUCTIONS.items())},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
)


# --------------------------------------------------------------------------
# object enumeration
# --------------------------------------------------------------------------


def enumerate_objects(
    lake: Any,
    gt: dict[str, Any],
    *,
    table_max_rows: int,
    table_max_cell_chars: int,
    table_row_format: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Full object set from the raw lake, never from a historical manifest.

    Returns the canonical, byte-ordered object list and a ``kind`` map.  Target
    and query tables are disjoint by id; assets are canonicalized by content.
    """
    objects: list[dict[str, Any]] = []
    for table_id in byte_order(lake.lake_tables):
        objects.append(
            {
                "object_id": table_id,
                "object_type": "table",
                "modality": "table",
                "roles": ["target"],
                "stable_key": sha256_text("target\0" + table_id),
            }
        )
    for query_id in byte_order(lake.query_tables):
        objects.append(
            {
                "object_id": query_id,
                "object_type": "table",
                "modality": "table",
                "roles": ["query"],
                "stable_key": sha256_text("query\0" + query_id),
            }
        )
    for canon in byte_order(gt["canonical"]):
        entry = gt["canonical"][canon]
        objects.append(
            {
                "object_id": canon,
                "object_type": "evidence",
                "modality": entry["asset_type"],
                "roles": ["evidence"],
                "stable_key": sha256_text("evidence\0" + canon),
            }
        )
    objects.sort(key=lambda r: r["object_id"].encode("utf-8"))
    ids = [r["object_id"] for r in objects]
    if len(set(ids)) != len(ids):
        duplicates = [i for i in ids if ids.count(i) > 1]
        raise ConfigError(f"object ids are not globally unique: {sorted(set(duplicates))[:10]}")
    corpus_kind = {
        "target": "target",
        "query": "query",
        "text": "evidence_text",
        "image": "evidence_image",
    }
    kinds = {
        r["object_id"]: corpus_kind[r["modality"] if r["object_type"] != "table" else r["roles"][0]]
        for r in objects
    }
    return objects, kinds


# --------------------------------------------------------------------------
# prompts and token spans
# --------------------------------------------------------------------------


def _load_embedder_class(model_dir: Path):
    script = model_dir / "scripts" / "qwen3_vl_embedding.py"
    if not script.is_file():
        raise FileNotFoundError(f"missing official Qwen embedding wrapper: {script}")
    spec = importlib.util.spec_from_file_location("_clean_r1_qwen3_vl_embedding", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Qwen3VLEmbedder


class FrozenEncoder:
    """Frozen, eval, inference-mode Qwen encoder for compact object features."""

    def __init__(
        self,
        backbone: Path,
        cache_config: dict[str, Any],
        device: str = "cuda",
        timing: "Timing | None" = None,
    ):
        self.timing: Timing = timing if timing is not None else Timing()
        # ``_active`` is the accumulator the phase helpers write to; it is swapped
        # per object so each modality gets its own attributed phase totals.
        self._active: Timing = self.timing
        wrapper = _load_embedder_class(Path(backbone))
        self.max_length = int(cache_config["max_length"])
        self.part_token_limit = int(cache_config["part_token_limit"])
        self.text_token_limit = int(cache_config["text_token_limit"])
        self.image_max_pixels = int(cache_config["image_max_pixels"])
        self.embedder = wrapper(
            model_name_or_path=str(backbone),
            max_length=self.max_length,
            max_pixels=self.image_max_pixels,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.embedder.model.eval()
        for parameter in self.embedder.model.parameters():
            parameter.requires_grad_(False)
        self.processor = self.embedder.processor
        self.tokenizer = self.processor.tokenizer
        self.device = device
        self.merge_size = int(self.processor.image_processor.merge_size)
        self.image_token_id = int(self.processor.image_token_id)
        self.model_fingerprint = {
            "backbone_dir": str(backbone),
            "config_sha256": sha256_path(Path(backbone) / "config.json"),
            "index_sha256": sha256_path(Path(backbone) / "model.safetensors.index.json"),
            "preprocessor_sha256": sha256_path(Path(backbone) / "preprocessor_config.json"),
            "wrapper_path": str(Path(backbone) / "scripts" / "qwen3_vl_embedding.py"),
            "wrapper_sha256": sha256_path(Path(backbone) / "scripts" / "qwen3_vl_embedding.py"),
            "instructions_origin_repo_sha256": SOURCE_AUDIT_INSTRUCTIONS_ORIGIN_SHA256,
            "tokenizer_sha256": sha256_path(Path(backbone) / "tokenizer.json")
            if (Path(backbone) / "tokenizer.json").is_file()
            else None,
            "prompt_version": PROMPT_VERSION,
            "instructions_sha256": INSTRUCTIONS_SHA256,
            "max_length": self.max_length,
            "image_max_pixels": self.image_max_pixels,
            "part_token_limit": self.part_token_limit,
            "text_token_limit": self.text_token_limit,
            "dtype": str(cache_config["qwen_dtype"]),
        }

    # -- prompt construction ---------------------------------------------

    def _conversation(self, *, text: str | None, image: str | None, instruction: str):
        return self.embedder.format_model_input(
            text=text, image=image, instruction=instruction
        )

    def _render(self, conversation: list[dict[str, Any]]) -> str:
        return self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

    def _encode(self, conversation: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        with self._active.stage("preprocess"):
            return self._encode_inner(conversation)

    def _encode_inner(self, conversation: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        rendered = self._render(conversation)
        from qwen_vl_utils.vision_process import process_vision_info

        try:
            images, video_inputs, video_kwargs = process_vision_info(
                [conversation], image_patch_size=16,
                return_video_metadata=True, return_video_kwargs=True,
            )
        except Exception as error:  # pragma: no cover - defensive
            raise ConfigError(f"vision preprocessing failed: {error!r}") from error
        inputs = self.processor(
            text=[rendered] if isinstance(rendered, str) else rendered,
            images=images,
            videos=None,
            video_metadata=None,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _offsets(self, rendered: str, raw_text: str) -> dict[str, Any]:
        """Tokenize the exact rendered prompt and keep the real offset mapping.

        The chat template already emits the special tokens, and the processor
        tokenizes the rendered string with ``add_special_tokens=True``; using
        ``False`` here would drop one leading token and shift every span.
        """
        with self._active.stage("token_spans"):
            encoded = self.tokenizer(
                rendered,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=True,
            )
        return {
            "input_ids": encoded["input_ids"],
            "offsets": encoded["offset_mapping"],
            "text_start": rendered.rfind(raw_text) if raw_text else -1,
        }

    def _forward(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # A GPU model is asynchronous, so the forward is bracketed by explicit
        # synchronisation; otherwise this would measure enqueue time, not compute.
        with self._active.stage("qwen_forward"):
            return self._forward_inner(inputs)

    def _forward_inner(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        with torch.inference_mode():
            outputs = self.embedder.forward(inputs)
        hidden = outputs["last_hidden_state"]
        mask = outputs["attention_mask"].bool()
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        if hidden.shape[1] > self.max_length:
            raise ConfigError(
                f"forward produced {hidden.shape[1]} positions, above max_length "
                f"{self.max_length}; the pre-forward check should have caught this"
            )
        return hidden[0], mask[0]

    # -- per modality ------------------------------------------------------

    def encode_table(
        self,
        *,
        object_id: str,
        parts: list[str],
        instruction: str,
        clipped_parts: list[str],
        token_counts: list[int],
    ) -> dict[str, Any]:
        raw_text = "\n".join(clipped_parts)
        conversation = self._conversation(text=raw_text, image=None, instruction=instruction)
        inputs = self._encode(conversation)
        if int(inputs["input_ids"].shape[1]) + 64 > self.max_length:
            raise ConfigError(
                f"{object_id}: pre-forward length "
                f"{int(inputs['input_ids'].shape[1]) + 64} exceeds max_length "
                f"{self.max_length}; CPU pre-check must fail rather than re-forward shorter."
            )
        rendered = self._render(conversation)
        info = self._offsets(rendered, raw_text)
        hidden, mask = self._forward(inputs)
        ids = inputs["input_ids"][0].tolist()
        if info["input_ids"] != ids:
            raise ConfigError(
                f"{object_id}: tokenizer offsets do not align with the Qwen forward; "
                "the part spans cannot be trusted"
            )
        if info["text_start"] < 0:
            raise ConfigError(f"{object_id}: chat template did not preserve table text")

        selected: list[int] = []
        groups: list[int] = []
        cursor = 0
        spans = []
        for part in clipped_parts:
            start = rendered.find(part, info["text_start"] if not spans else cursor)
            if start < 0:
                raise ConfigError(
                    f"{object_id}: a serialized part was not preserved verbatim in the "
                    "rendered prompt; token spans cannot be located"
                )
            spans.append((start, start + len(part)))
            cursor = start + len(part)
        for token_index, (start, end) in enumerate(info["offsets"]):
            if end <= start or token_index >= hidden.shape[0]:
                continue
            for group, (part_start, part_end) in enumerate(spans):
                if end > part_start and start < part_end:
                    selected.append(token_index)
                    groups.append(group)
                    break
        if set(groups) != set(range(len(clipped_parts))):
            missing = sorted(set(range(len(clipped_parts))) - set(groups))
            raise ConfigError(
                f"{object_id}: truncation removed every token of part(s) {missing}; "
                "the input capacity must be checked on CPU before the forward"
            )
        index = torch.tensor(selected, dtype=torch.long, device=hidden.device)
        group = torch.tensor(groups, dtype=torch.long, device=hidden.device)
        # Spec 4.3: the part means are accumulated in float32; the forward itself
        # ran in the configured bfloat16, and only the pooled results are stored.
        content = hidden.index_select(0, index).float()
        pooled = torch.zeros(len(clipped_parts), hidden.shape[-1], device=hidden.device)
        counts = torch.zeros(len(clipped_parts), device=hidden.device)
        pooled.index_add_(0, group, content)
        counts.index_add_(0, group, torch.ones_like(group, dtype=torch.float))
        pooled = pooled / counts.clamp(min=1)[:, None]

        z = self._pool(hidden, mask)
        schema = pooled[0]
        rows = pooled[1:]
        slots = self._group_rows(rows)
        return {"z": z, "summary": torch.stack([schema] + slots, dim=0),
                "token_counts": token_counts}

    @staticmethod
    def _group_rows(rows: torch.Tensor) -> list[torch.Tensor]:
        """C[1..7]: one slot per row when m<=7, else seven equal contiguous groups."""
        m = rows.shape[0]
        if m == 0:
            return [torch.zeros_like(rows[0]) for _ in range(7)]
        if m <= 7:
            slots = [rows[i] for i in range(m)]
            slots += [torch.zeros_like(rows[0]) for _ in range(7 - m)]
            return slots
        slots = []
        for j in range(7):
            start = (j * m) // 7
            stop = ((j + 1) * m) // 7
            slots.append(rows[start:stop].mean(dim=0))
        return slots

    def encode_text(
        self, *, object_id: str, content: str, instruction: str
    ) -> dict[str, Any]:
        token_ids = self.tokenizer(content, add_special_tokens=False)["input_ids"]
        if len(token_ids) > self.text_token_limit:
            content = self.tokenizer.decode(
                token_ids[: self.text_token_limit], skip_special_tokens=False
            )
            token_ids = token_ids[: self.text_token_limit]
        conversation = self._conversation(text=content, image=None, instruction=instruction)
        inputs = self._encode(conversation)
        if int(inputs["input_ids"].shape[1]) + 64 > self.max_length:
            raise ConfigError(
                f"{object_id}: text object exceeds max_length after pre-check"
            )
        rendered = self._render(conversation)
        info = self._offsets(rendered, content)
        hidden, mask = self._forward(inputs)
        if info["input_ids"] != inputs["input_ids"][0].tolist():
            raise ConfigError(f"{object_id}: text offsets do not align with the forward")
        text_start = info["text_start"]
        if text_start < 0:
            raise ConfigError(f"{object_id}: chat template did not preserve text content")
        text_end = text_start + len(content)
        selected = [
            i
            for i, (start, end) in enumerate(info["offsets"])
            if end > start and end > text_start and start < text_end and i < hidden.shape[0]
        ]
        if not selected:
            raise ConfigError(f"{object_id}: no content tokens located for the text object")
        index = torch.tensor(selected, dtype=torch.long, device=hidden.device)
        body = hidden.index_select(0, index).float()
        z = self._pool(hidden, mask)
        length = body.shape[0]
        if length >= 8:
            slots = []
            for j in range(8):
                start = (j * length) // 8
                stop = ((j + 1) * length) // 8
                slots.append(body[start:stop].mean(dim=0))
        else:
            slots = [body[j] for j in range(length)]
            slots += [torch.zeros_like(body[0]) for _ in range(8 - length)]
        return {
            "z": z,
            "summary": torch.stack(slots, dim=0),
            "token_counts": [len(token_ids)],
            "content_tokens": length,
        }

    def encode_image(
        self, *, object_id: str, image_path: str, instruction: str
    ) -> dict[str, Any]:
        conversation = self._conversation(text=None, image=image_path, instruction=instruction)
        inputs = self._encode(conversation)
        ids = inputs["input_ids"][0]
        placeholders = int((ids == self.image_token_id).sum())
        grid = inputs.get("image_grid_thw")
        if grid is None or grid.numel() == 0:
            raise ConfigError(f"{object_id}: processor returned no image_grid_thw")
        merge = self.merge_size
        expected = int(
            (grid[0][1] // merge) * (grid[0][2] // merge) * int(grid[0][0])
        )
        if placeholders != expected:
            raise ConfigError(
                f"image_token_mapping_mismatch: {object_id} has {placeholders} image "
                f"placeholder tokens but grid {grid.tolist()} with merge {merge} implies "
                f"{expected}; refusing to fall back to averaging every prompt token."
            )
        if int(ids.shape[0]) + 64 > self.max_length:
            raise ConfigError(f"{object_id}: image prompt exceeds max_length")
        hidden, mask = self._forward(inputs)
        positions = (ids == self.image_token_id).nonzero(as_tuple=False).flatten()
        positions = positions[positions < hidden.shape[0]]
        if int(positions.numel()) != expected:
            raise ConfigError(
                f"image_token_mapping_mismatch: {object_id} visual positions "
                f"{int(positions.numel())} != merged grid tokens {expected}"
            )
        body = hidden.index_select(0, positions).float()
        z = self._pool(hidden, mask)
        length = body.shape[0]
        if length >= 8:
            slots = []
            for j in range(8):
                start = (j * length) // 8
                stop = ((j + 1) * length) // 8
                slots.append(body[start:stop].mean(dim=0))
        else:
            slots = [body[j] for j in range(length)]
            slots += [torch.zeros_like(body[0]) for _ in range(8 - length)]
        return {
            "z": z,
            "summary": torch.stack(slots, dim=0),
            "token_counts": [int(ids.shape[0])],
            "visual_tokens": length,
        }

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Official last-valid-token pooling, then float32 L2 normalization."""
        flipped = mask.flip(dims=[0]).to(dtype=torch.long)
        last = flipped.argmax(dim=0)
        column = mask.shape[0] - last - 1
        pooled = hidden[column]
        return F.normalize(pooled.float(), p=2, dim=-1)


def clip_table_parts(
    parts: list[str], tokenizer: Any, part_token_limit: int
) -> tuple[list[str], list[int], list[int]]:
    """Clamp every part to the first ``part_token_limit`` tokens (spec 4.2)."""
    clipped: list[str] = []
    original: list[int] = []
    kept: list[int] = []
    for part in parts:
        ids = tokenizer(part, add_special_tokens=False)["input_ids"]
        original.append(len(ids))
        if len(ids) > part_token_limit:
            clipped.append(
                tokenizer.decode(ids[:part_token_limit], skip_special_tokens=False)
            )
            kept.append(part_token_limit)
        else:
            clipped.append(part)
            kept.append(len(ids))
    return clipped, original, kept


# --------------------------------------------------------------------------
# shard storage (memory-mapped, one shard per 4096 objects)
# --------------------------------------------------------------------------


def load_shard_vectors(
    root: Path, entries: list[dict[str, Any]], dim: int, slots: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Memory-map the shards backing ``entries`` and return stacked arrays.

    ``np.memmap`` has no context-manager protocol, so each mapping is released
    explicitly; leaving them open exhausts file descriptors on a full lake.  The
    persisted mask covers every slot, so it is ``slots + 1`` wide: one global slot
    plus the summaries.
    """
    z = np.zeros((len(entries), dim), dtype=np.float32)
    c = np.zeros((len(entries), slots, dim), dtype=np.float16)
    m = np.zeros((len(entries), slots + 1), dtype=np.uint8)
    by_shard: dict[str, list[tuple[int, int]]] = {}
    for position, entry in enumerate(entries):
        by_shard.setdefault(str(entry["shard"]), []).append((position, int(entry["offset"])))
    for shard, items in by_shard.items():
        directory = _shard_directory(Path(root), shard)
        zm = np.memmap(directory / "z.f32", dtype=np.float32, mode="r").reshape(-1, dim)
        cm = np.memmap(directory / "C.f16", dtype=np.float16, mode="r").reshape(-1, slots, dim)
        mm = np.memmap(directory / "mask.u8", dtype=np.uint8, mode="r").reshape(
            -1, slots + 1
        )
        try:
            for position, offset in items:
                z[position] = zm[offset]
                c[position] = cm[offset]
                m[position] = mm[offset]
        finally:
            del zm, cm, mm
    return z, c, m


def _shard_directory(root: Path, shard: str) -> Path:
    """Resolve a manifest ``shard`` value to the directory holding its arrays.

    A merged manifest records ``<source>/<shard>`` (for example
    ``gpu-1/shard-00026``) relative to the cache root, while the files live under
    ``<source>/shards/<shard>``.  A single-process build records the bare shard
    name and keeps its files under ``shards/``.  Both layouts are accepted so the
    merged index and a raw build resolve to the same bytes.
    """
    candidates = [
        root / shard,
        root / shard.replace("/shard-", "/shards/shard-", 1),
        root / "shards" / shard,
    ]
    for candidate in candidates:
        if (candidate / "z.f32").is_file():
            return candidate
    raise ConfigError(
        f"no shard arrays found for {shard!r}; tried "
        + ", ".join(str(c) for c in candidates)
    )


def read_cache_index(feature_dir: Path) -> dict[str, dict[str, Any]]:
    path = Path(feature_dir) / "manifest.jsonl"
    if not path.is_file():
        raise ConfigError(f"cache manifest missing: {path}")
    index: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        index[row["object_id"]] = row
    return index


# --------------------------------------------------------------------------
# per-object payload helpers
# --------------------------------------------------------------------------


def cleaned_text_content(asset: dict[str, Any]) -> str:
    """Normalize text evidence the same way its content key is computed.

    Unicode NFC and CRLF->LF are applied, and outer whitespace is stripped; inner
    whitespace is preserved so no entity or value is rewritten (spec section 3).
    """
    import unicodedata

    text = clean_text(asset.get("content"))
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def row_group_token_counts(row_counts: list[int]) -> list[int]:
    """Token totals of the seven row slots, mirroring ``FrozenEncoder._group_rows``."""
    m = len(row_counts)
    if m == 0:
        return [0] * 7
    if m <= 7:
        return list(row_counts) + [0] * (7 - m)
    groups = []
    for j in range(7):
        start = (j * m) // 7
        stop = ((j + 1) * m) // 7
        groups.append(sum(row_counts[start:stop]))
    return groups


def summary_mask(payload: dict[str, Any], summary_slots: int) -> np.ndarray:
    """Validity over all object slots: 1 global plus ``summary_slots`` summaries.

    Slot 0 is the global vector and is always valid -- the reference requires a
    valid global slot for every object.  Slot 1 is the schema summary for a table,
    or the first content bin for text and image; the remaining slots are the row
    groups (tables) or the remaining bins.  A slot is valid exactly when the
    content it averages is non-empty, so a padded table row slot or a bin beyond
    the token count is masked out rather than averaged as zeros.
    """
    mask = np.zeros(summary_slots + 1, dtype=np.uint8)
    mask[0] = 1
    if payload["modality_id"] == MODALITY_TABLE:
        # part_token_kept is [schema, row_1, ...]; C is [schema, row_slot_1..7].
        # The schema keeps a slot of its own -- dropping it would misalign every
        # later slot -- and row_group_token_counts already pads to seven entries,
        # so a table with fewer than seven visible rows leaves the tail masked.
        parts = list(payload["part_token_kept"])
        counts = parts[:1] + row_group_token_counts(parts[1:])
    else:
        count = int(payload.get("content_tokens") or payload.get("visual_tokens") or 0)
        counts = [1] * min(count, summary_slots)
    for index in range(1, summary_slots + 1):
        if index - 1 < len(counts) and counts[index - 1] > 0:
            mask[index] = 1
    return mask


def encode_one(
    encoder: FrozenEncoder,
    lake: Any,
    entry: dict[str, Any],
    cache_config: dict[str, Any],
    gt: dict[str, Any],
) -> dict[str, Any]:
    """One frozen forward for one object, with per-phase and per-modality timing.

    Exactly one Qwen call is made per object; OOM is handled by the caller's batch
    splitting, never by silently substituting a shorter input.
    """
    object_id = entry["object_id"]
    modality = "table" if entry["object_type"] == "table" else entry["modality"]
    phase = Timing()
    previous, encoder._active = encoder._active, phase
    try:
        with phase.stage("resolve_content"):
            if entry["object_type"] == "table":
                max_rows = int(cache_config["table_max_rows"])
                max_cell = int(cache_config["table_max_cell_chars"])
                row_format = str(cache_config["table_row_format"])
                content = lake.visible_content(object_id, max_rows, max_cell, row_format)
                parts = content["table_parts"]
                clipped, original, kept = clip_table_parts(
                    parts, encoder.tokenizer, encoder.part_token_limit
                )
            else:
                canonical = gt["canonical"][object_id]
                asset = lake.assets[object_id]
                content = parts = clipped = original = kept = None

        if entry["object_type"] == "table":
            payload = encoder.encode_table(
                object_id=object_id,
                parts=parts,
                instruction=EMBEDDING_INSTRUCTIONS[(content["embedding_role"], "table")],
                clipped_parts=clipped,
                token_counts=kept,
            )
            payload["kind_ids"] = list(TABLE_KINDS)
            payload["modality_id"] = MODALITY_TABLE
            payload["part_token_original"] = original
            payload["part_token_kept"] = kept
            payload["split"] = content["split"]
        elif canonical["asset_type"] == "text":
            text = cleaned_text_content(asset)
            payload = encoder.encode_text(
                object_id=object_id,
                content=text,
                instruction=EMBEDDING_INSTRUCTIONS[("evidence", "text")],
            )
            payload["kind_ids"] = list(TEXT_KINDS)
            payload["modality_id"] = MODALITY_TEXT
            payload["part_token_original"] = [len(text)]
            payload["part_token_kept"] = [payload.get("content_tokens", 0)]
        else:
            path = str(asset.get("local_path") or asset.get("relative_path"))
            payload = encoder.encode_image(
                object_id=object_id,
                image_path=path,
                instruction=EMBEDDING_INSTRUCTIONS[("evidence", "image")],
            )
            payload["kind_ids"] = list(IMAGE_KINDS_FULL)
            payload["modality_id"] = MODALITY_IMAGE
            payload["part_token_original"] = [0]
            payload["part_token_kept"] = [payload.get("visual_tokens", 0)]
    finally:
        encoder._active = previous
    for stage, samples in phase.samples.items():
        for value in samples:
            encoder.timing.record(f"{stage}:modality={modality}", value)
    return payload


# --------------------------------------------------------------------------
# object selection
# --------------------------------------------------------------------------


def select_objects(
    objects: list[dict[str, Any]],
    limit_per_modality: int | None,
    limit_tables: int | None = None,
    limit_evidence: int | None = None,
) -> list[dict[str, Any]]:
    """Deterministic (byte-order) subset used by the cache dry run and tests."""
    if limit_per_modality is None and limit_tables is None and limit_evidence is None:
        return list(objects)
    seen: dict[str, int] = {}
    tables = 0
    evidence = 0
    selected: list[dict[str, Any]] = []
    for entry in objects:
        modality = entry["modality"]
        if limit_per_modality is not None:
            if seen.get(modality, 0) >= limit_per_modality:
                continue
        if entry["object_type"] == "table":
            if limit_tables is not None and tables >= limit_tables:
                continue
            tables += 1
        else:
            if limit_evidence is not None and evidence >= limit_evidence:
                continue
            evidence += 1
        seen[modality] = seen.get(modality, 0) + 1
        selected.append(entry)
    return selected


def shard_objects(
    objects: list[dict[str, Any]],
    *,
    shard_id: int = 0,
    shard_count: int = 1,
    range_id: int = 0,
    range_count: int = 1,
    rank_offset: int = 0,
    rank_limit: int | None = None,
) -> list[dict[str, Any]]:
    """A disjoint contiguous slice of the byte-ordered object list.

    The slice boundaries come from the *full* list length rather than from a
    shrinking remainder, so consecutive shards tile the list exactly and a given
    shard id always selects the same objects regardless of how many are already
    encoded.  ``range_*`` splits one shard's tiling again, and
    ``rank_offset``/``rank_limit`` take a sub-block of the resulting slice.
    """
    for name, value, bound in (
        ("shard_id", shard_id, shard_count),
        ("range_id", range_id, range_count),
    ):
        if not 0 <= value < bound:
            raise ConfigError(f"{name} {value} outside [0, {bound})")
    if shard_count < 1 or range_count < 1:
        raise ConfigError("shard_count and range_count must be at least 1")

    total = len(objects)
    span = (total + shard_count - 1) // shard_count
    start = min(shard_id * span, total)
    stop = min(start + span, total)
    selected = list(objects[start:stop])

    if range_count > 1:
        r_span = (len(selected) + range_count - 1) // range_count
        r_start = min(range_id * r_span, len(selected))
        r_stop = min(r_start + r_span, len(selected))
        selected = selected[r_start:r_stop]

    if rank_offset:
        selected = selected[rank_offset:]
    if rank_limit is not None:
        selected = selected[: int(rank_limit)]
    return selected


# --------------------------------------------------------------------------
# cache build
# --------------------------------------------------------------------------


class CacheWriter:
    """Writes compact features into 4096-object shards plus a resumable index.

    ``manifest.jsonl`` is appended and fsynced per object, so an interrupted build
    resumes at the first object that has no row yet.  No Qwen state, optimizer
    state or full-length hidden states are ever persisted.
    """

    def __init__(
        self,
        feature_dir: Path,
        *,
        shard_objects: int,
        dim: int,
        slots: int,
        cache_fingerprint: str,
        accepted_fingerprints: set[str] | None = None,
        tag: str = "shards",
        timing: "Timing | None" = None,
    ) -> None:
        self.timing = timing if timing is not None else Timing()
        # The canonical value is written on new rows; the accepted set may also
        # hold older formula variants so an existing manifest stays resumable.
        self.accepted_fingerprints = set(accepted_fingerprints or ()) | {cache_fingerprint}
        self.cache_fingerprint = cache_fingerprint
        self.feature_dir = Path(feature_dir)
        self.tag = tag
        self.shards_dir = self.feature_dir / tag / "shards"
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.shard_objects = int(shard_objects)
        self.dim = int(dim)
        self.slots = int(slots)
        self.manifest_path = self.feature_dir / tag / "manifest.jsonl"
        self.rows: list[dict[str, Any]] = []
        self.index: dict[str, dict[str, Any]] = {}
        if self.manifest_path.is_file():
            for row in read_jsonl(self.manifest_path):
                if row.get("cache_fingerprint") not in self.accepted_fingerprints:
                    raise ConfigError(
                        "cache/manifest.jsonl was built with a different model, prompt or "
                        "input fingerprint; a partial cache from another configuration "
                        "cannot be resumed. "
                        f"row={row.get('cache_fingerprint')} "
                        f"accepted={sorted(self.accepted_fingerprints)}"
                    )
                self.rows.append(row)
                self.index[row["object_id"]] = row
        self._shard_index = -1
        self._offset = 0
        self._handle = None
        self._handles: dict[str, Any] = {}

    @property
    def count(self) -> int:
        return len(self.rows)

    def _rotate(self) -> None:
        if self._handle is not None:
            import os

            for handle in self._handles.values():
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
        self._shard_index += 1
        self._offset = 0
        directory = self.shards_dir / f"shard-{self._shard_index:05d}"
        directory.mkdir(parents=True, exist_ok=True)
        self._handles = {
            "z": (directory / "z.f32").open("ab"),
            "C": (directory / "C.f16").open("ab"),
            "mask": (directory / "mask.u8").open("ab"),
        }
        self._handle = self._handles["z"]

    def add(self, item: dict[str, Any]) -> None:
        if self._handle is None or self._offset >= self.shard_objects:
            self._rotate()
        z, summary, mask = item["z"], item["summary"], item["mask"]
        if z.shape != (self.dim,) or z.dtype != np.float32:
            raise ConfigError(f"{item['object_id']}: global vector must be float32[{self.dim}]")
        if summary.shape != (self.slots, self.dim) or summary.dtype != np.float16:
            raise ConfigError(
                f"{item['object_id']}: summary must be float16[{self.slots},{self.dim}]"
            )
        # The mask covers every slot: 1 global plus `slots` summaries.  The global
        # slot is always valid (spec Eq. 11-12), so the mask is never all-zero.
        if mask.shape != (self.slots + 1,) or mask.dtype != np.uint8:
            raise ConfigError(
                f"{item['object_id']}: mask must be uint8[{self.slots + 1}]"
            )
        if not mask[0]:
            raise ConfigError(f"{item['object_id']}: the global slot must be valid")
        if not np.isfinite(z).all() or not np.isfinite(summary.astype(np.float32)).all():
            raise ConfigError(f"{item['object_id']}: non-finite cached features")
        with self.timing.stage("shard_write"):
            self._handles["z"].write(z.tobytes())
            self._handles["C"].write(summary.tobytes())
            self._handles["mask"].write(mask.tobytes())
        row = {
            "cache_fingerprint": self.cache_fingerprint,
            "object_id": item["object_id"],
            "shard": f"shard-{self._shard_index:05d}",
            "offset": self._offset,
            "modality": item["modality"],
            "object_type": item["object_type"],
            "modality_id": item["modality_id"],
            "kind_ids": item["kind_ids"],
            "mask_popcount": int(mask.sum()),
            "z_norm": float(np.linalg.norm(z)),
            "part_token_original": item.get("part_token_original"),
            "part_token_kept": item.get("part_token_kept"),
            "content_tokens": item.get("content_tokens"),
            "visual_tokens": item.get("visual_tokens"),
            "split": item.get("split"),
        }
        self.rows.append(row)
        self.index[item["object_id"]] = row
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            import os

            os.fsync(handle.fileno())
        self._offset += 1

    def close(self) -> None:
        if self._handle is None:
            return
        import os

        for handle in self._handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        self._handle = None

    def vectors(self, object_ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return load_shard_vectors(
            self.shards_dir, [self.index[o] for o in object_ids], self.dim, self.slots
        )


def cache_fingerprint(
    resolved: dict[str, Any], encoder: "FrozenEncoder", object_count: int | None = None
) -> str:
    """Identity of the encoding itself: model, prompt and cache configuration.

    ``object_count`` is accepted only to reproduce an older formula that folded a
    slice's size into the hash.  Passing it is how a manifest written by that
    revision is recognised as the same encoding; new rows always carry the
    canonical form, which deliberately excludes the size.
    """
    payload: dict[str, Any] = {
        "prompt_version": PROMPT_VERSION,
        "instructions_sha256": INSTRUCTIONS_SHA256,
        "model": encoder.model_fingerprint,
        "cache_config": resolved["cache"],
    }
    if object_count is not None:
        payload["object_count"] = int(object_count)
    return stable_digest(payload)


def acceptable_fingerprints(
    resolved: dict[str, Any],
    encoder: "FrozenEncoder",
    observed: set[str],
    candidate_counts: Sequence[int] = (),
) -> set[str]:
    """Fingerprints this process may resume from.

    The canonical value is always accepted.  A value produced by the older
    size-dependent formula is accepted when one of ``candidate_counts`` reproduces
    it exactly.  Recognising such a value never weakens the check on the encoding:
    a wrong count cannot reproduce the hash, and the per-object metadata is
    verified independently, so only features that are in fact identical are kept.
    """
    accepted = {cache_fingerprint(resolved, encoder)}
    if not observed:
        return accepted
    for count in sorted({int(c) for c in candidate_counts if c is not None and c >= 0}):
        candidate = cache_fingerprint(resolved, encoder, count)
        if candidate in observed:
            accepted.add(candidate)
        if observed <= accepted:
            break
    return accepted


def build_object_records(
    lake: Any,
    gt: dict[str, Any],
    objects: list[dict[str, Any]],
    cache_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Object metadata (no features), in canonical order."""
    max_rows = int(cache_config["table_max_rows"])
    max_cell = int(cache_config["table_max_cell_chars"])
    row_format = str(cache_config["table_row_format"])
    records: list[dict[str, Any]] = []
    for entry in objects:
        object_id = entry["object_id"]
        if entry["object_type"] == "table":
            content = lake.visible_content(object_id, max_rows, max_cell, row_format)
            table = lake.query_tables.get(object_id) or lake.lake_tables.get(object_id, {})
            records.append(
                {
                    "object_id": object_id,
                    "object_type": "table",
                    "modality": "table",
                    "role": content["embedding_role"],
                    "content_sha256": sha256_text("\n".join(content["table_parts"])),
                    "num_rows_visible": content["num_rows_visible"],
                    "num_rows_total": content["num_rows_total"],
                    "truncated_rows": content["num_rows_total"] > max_rows,
                    "part_count": len(content["table_parts"]),
                    "split": content["split"],
                    "source_table_id": str(table.get("source_table_id", "")),
                }
            )
            continue
        canonical = gt["canonical"][object_id]
        record = {
            "object_id": object_id,
            "object_type": "evidence",
            "modality": canonical["asset_type"],
            "role": "evidence",
            "content_sha256": canonical["content_sha256"],
            "alias_count": canonical["alias_count"],
            "part_count": 1,
        }
        asset = lake.assets[object_id]
        if canonical["asset_type"] == "text":
            record["char_length"] = len(cleaned_text_content(asset))
        else:
            record["image_path"] = str(
                asset.get("local_path") or asset.get("relative_path")
            )
        records.append(record)
    return records


def build_cache(
    *,
    output_root: Path,
    resolved: dict[str, Any],
    lake: Any,
    gt: dict[str, Any],
    objects: list[dict[str, Any]],
    limit_per_modality: int | None = None,
    limit_tables: int | None = None,
    limit_evidence: int | None = None,
    device: str = "cuda",
    tag: str = "shards",
    timing: Timing | None = None,
    object_total: int | None = None,
) -> dict[str, Any]:
    """One-pass frozen encoding of the object set into compact shards.

    ``object_total`` is the size of the *whole* object set, which callers pass when
    they hand in a filtered subset (an ``--only-missing`` resume).  It matters
    because an older cache fingerprint folded in the size of the slice its writer
    was given, and that size is reconstructed from the total rather than from the
    subset actually being encoded.
    """
    timing = timing if timing is not None else Timing()
    cache_config = resolved["cache"]
    dim = int(cache_config["global_dimension"])
    slots = int(cache_config["summary_slots"])
    selected = select_objects(objects, limit_per_modality, limit_tables, limit_evidence)
    feature_dir = Path(output_root) / "cache"

    with timing.stage("model_load"):
        encoder = FrozenEncoder(
            Path(resolved["paths"]["backbone_dir"]), cache_config, device, timing=timing
        )
    fingerprint = cache_fingerprint(resolved, encoder)
    if object_total is None:
        object_total = len(objects)

    # Resume recognition.  A manifest may hold rows written by an older revision of
    # the fingerprint formula, and a surviving manifest's row count is not the slice
    # size that formula recorded (a resumed run appends only what was missing).  So
    # candidate sizes are reconstructed from the object total under the partition
    # rules this project uses -- a contiguous split into k shards holds ceil(N/k) --
    # and from each manifest's own row count.
    observed: set[str] = set()
    candidate_counts: set[int] = {len(selected), len(objects), object_total}
    for divisor in range(1, 129):
        candidate_counts.add(-(-object_total // divisor))  # ceil
    for manifest in sorted(feature_dir.glob("*/manifest.jsonl")):
        rows_here = 0
        for row in read_jsonl(manifest):
            rows_here += 1
            value = row.get("cache_fingerprint")
            if value:
                observed.add(str(value))
        candidate_counts.add(rows_here)
    accepted = acceptable_fingerprints(
        resolved, encoder, observed, candidate_counts=sorted(candidate_counts)
    )
    if len(accepted) > 1:
        log_line(
            f"cache: resuming a manifest written by an older fingerprint formula "
            f"({len(observed)} distinct stored value(s)); the encoding is verified "
            "per object instead of by hash equality"
        )
    writer = CacheWriter(
        feature_dir,
        shard_objects=int(cache_config["shard_objects"]),
        dim=dim,
        slots=slots,
        cache_fingerprint=fingerprint,
        accepted_fingerprints=accepted,
        tag=tag,
        timing=timing,
    )
    started = time.time()
    failures: list[dict[str, Any]] = []
    encoded_now = 0
    for entry in selected:
        object_id = entry["object_id"]
        if object_id in writer.index:
            continue
        try:
            with timing.stage("encode_total"):
                payload = encode_one(encoder, lake, entry, cache_config, gt)
        except Exception as error:
            failures.append({"object_id": object_id, "error": repr(error)})
            log_line(f"cache: FAILED {object_id}: {error!r}")
            continue
        writer.add(
            {
                "object_id": object_id,
                "modality": entry["modality"],
                "object_type": entry["object_type"],
                "modality_id": payload["modality_id"],
                "kind_ids": payload["kind_ids"],
                "z": payload["z"].detach().float().cpu().numpy().astype(np.float32),
                "summary": payload["summary"].detach().float().cpu().numpy().astype(np.float16),
                "mask": summary_mask(payload, slots),
                "part_token_original": payload.get("part_token_original"),
                "part_token_kept": payload.get("part_token_kept"),
                "content_tokens": payload.get("content_tokens"),
                "visual_tokens": payload.get("visual_tokens"),
                "split": payload.get("split"),
            }
        )
        encoded_now += 1
        if encoded_now % 500 == 0:
            elapsed = time.time() - started
            rate = encoded_now / max(elapsed, 1e-6)
            remaining = max(0, len(selected) - encoded_now)
            log_line(
                # Progress is against this pass, not the manifest total: a resumed
                # run appends to a manifest that already holds earlier rows, so the
                # manifest size is not the work to be done.
                f"cache: {encoded_now}/{len(selected)} objects this pass "
                f"({rate:.2f}/s, ~{remaining / max(rate, 1e-6) / 60:.1f} min left)"
            )
    writer.close()
    if failures:
        write_jsonl(feature_dir / "encode_failures.jsonl", failures)
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed = time.time() - started
    by_modality: dict[str, int] = {"table": 0, "text": 0, "image": 0}
    for entry in selected:
        key = "table" if entry["object_type"] == "table" else entry["modality"]
        by_modality[key] = by_modality.get(key, 0) + 1
    attributed: dict[str, float] = {m: 0.0 for m in by_modality}
    for stage, samples in timing.samples.items():
        for modality in by_modality:
            if stage.endswith(f"modality={modality}"):
                attributed[modality] += sum(samples)
                break
    report = timing.report(name=f"cache_build[{tag}]", total_seconds=elapsed)
    report["objects_per_modality"] = by_modality
    report["objects_encoded_this_pass"] = encoded_now
    report["seconds_per_object_by_modality"] = {
        modality: attributed[modality] / count
        for modality, count in by_modality.items()
        if count
    }
    (feature_dir / tag).mkdir(parents=True, exist_ok=True)
    write_json(feature_dir / tag / "TIMING.json", report)
    return {
        "objects_in_cache": writer.count,
        "objects_selected": len(selected),
        "objects_this_pass": encoded_now,
        "failures": len(failures),
        "feature_bytes": writer.count * dim * (4 + slots * 2),
        "elapsed_seconds": elapsed,
        "cache_fingerprint": fingerprint,
        "shards": writer._shard_index + 1,
        "objects_per_modality": by_modality,
        "seconds_per_object": elapsed / max(1, encoded_now),
        "seconds_per_object_by_modality": report["seconds_per_object_by_modality"],
        "timing_report": f"cache/{tag}/TIMING.json",
    }
