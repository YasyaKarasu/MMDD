#!/usr/bin/env python
"""Build frozen Qwen3-VL embeddings for stage-1 objects."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from image_preprocessing import DEFAULT_MAX_IMAGE_PIXELS, ensure_image_within_pixel_limit
from qwen3_vl_embedding import Qwen3VLEmbeddingEncoder
from stage1_io import iter_jsonl, iter_manifest_records, setup_logging, update_stage1_manifest, write_json
from stage1_serialization import (
    image_local_path,
    serialize_image_asset_prompt,
    serialize_table_for_embedding,
    serialize_text_asset_for_embedding,
)

EMBEDDING_INSTRUCTIONS = {
    "connectivity": {
        "table": "Represent only this table's intrinsic schema, values, and factual meaning for retrieval. Do not assume any relationship to another object.",
        "text": "Represent only this independent text's intrinsic content and factual meaning for retrieval. Do not assume it belongs to any table or entity.",
        "image": "Represent only this independent image's intrinsic visual content and factual meaning for retrieval. Do not assume it belongs to any table or entity.",
    },
    "content_only": {
        "table": "Represent the intrinsic content, schema, values, and factual meaning of this table for retrieval.",
        "text": "Represent the intrinsic content, facts, entities, and semantics of this text for retrieval.",
        "image": "Represent the intrinsic visual content, entities, objects, scene, and factual attributes visible in this image for retrieval.",
    },
}
DEFAULT_EMBEDDING_PROMPT_MODE = "connectivity"
EMBEDDING_PROMPT_MODE_CHOICES = tuple(EMBEDDING_INSTRUCTIONS)
TABLE_SERIALIZATION_VERSION = "intrinsic_table_and_asset_content_v2"


def embedding_prompt_mode(args: argparse.Namespace) -> str:
    mode = getattr(args, "embedding_prompt_mode", DEFAULT_EMBEDDING_PROMPT_MODE)
    if mode not in EMBEDDING_INSTRUCTIONS:
        choices = ", ".join(EMBEDDING_PROMPT_MODE_CHOICES)
        raise ValueError(f"Unsupported embedding_prompt_mode {mode!r}; use one of: {choices}")
    return mode


def embedding_instructions(args: argparse.Namespace) -> dict[str, str]:
    return EMBEDDING_INSTRUCTIONS[embedding_prompt_mode(args)]


def save_embeddings(out_dir: Path, object_type: str, ids: list[str], embeddings: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{object_type}.npy", embeddings.astype("float32"))
    write_json(out_dir / f"{object_type}_ids.json", ids)


def prompt_cache_compatible(out_dir: Path, mode: str) -> bool:
    stats_path = out_dir / "embedding_stats.json"
    if not stats_path.exists():
        return False
    try:
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if payload.get("embedding_prompt_mode", DEFAULT_EMBEDDING_PROMPT_MODE) != mode:
        return False
    return payload.get("table_serialization_version") == TABLE_SERIALIZATION_VERSION


def existing(out_dir: Path, object_type: str, force: bool, cache_compatible: bool = True) -> bool:
    if not cache_compatible:
        return False
    return not force and (out_dir / f"{object_type}.npy").exists() and (out_dir / f"{object_type}_ids.json").exists()


def encode_table_fragments(args: argparse.Namespace, encoder: Qwen3VLEmbeddingEncoder, emb_dir: Path) -> dict[str, Any]:
    object_type = "table_fragment"
    mode = embedding_prompt_mode(args)
    if existing(emb_dir, object_type, args.force_recompute, prompt_cache_compatible(emb_dir, mode)):
        ids = json.loads((emb_dir / f"{object_type}_ids.json").read_text(encoding="utf-8"))
        arr = np.load(emb_dir / f"{object_type}.npy", mmap_mode="r")
        return {"object_type": object_type, "count": len(ids), "dimension": int(arr.shape[1]), "cached": True}
    fragments = list(iter_jsonl(Path(args.stage1_dir) / "logic_fragments.jsonl"))
    ids = [frag["fragment_id"] for frag in fragments]
    texts = [serialize_table_for_embedding(frag, max_rows=args.max_table_rows) for frag in fragments]
    embeddings = encoder.encode_tables(texts, instruction=embedding_instructions(args)["table"])
    save_embeddings(emb_dir, object_type, ids, embeddings)
    return {"object_type": object_type, "count": len(ids), "dimension": int(embeddings.shape[1]), "cached": False}


def encode_assets(args: argparse.Namespace, encoder: Qwen3VLEmbeddingEncoder, emb_dir: Path) -> list[dict[str, Any]]:
    stats = []
    text_ids: list[str] = []
    text_inputs: list[str] = []
    image_ids: list[str] = []
    image_paths: list[str] = []
    image_prompts: list[str] = []
    skipped_images: list[dict[str, str]] = []
    resized_images: list[dict[str, Any]] = []
    mode = embedding_prompt_mode(args)
    cache_compatible = prompt_cache_compatible(emb_dir, mode)
    instructions = embedding_instructions(args)
    need_text = not existing(emb_dir, "text_asset", args.force_recompute, cache_compatible)
    need_image = not existing(emb_dir, "image_asset", args.force_recompute, cache_compatible)
    if not need_text and not need_image:
        for object_type in ("text_asset", "image_asset"):
            ids = json.loads((emb_dir / f"{object_type}_ids.json").read_text(encoding="utf-8"))
            arr = np.load(emb_dir / f"{object_type}.npy", mmap_mode="r")
            stats.append({"object_type": object_type, "count": len(ids), "dimension": int(arr.shape[1]), "cached": True})
        return stats

    input_dir = Path(args.input_dir)
    for asset in iter_manifest_records(input_dir, "bridge_assets", log_every=50000):
        if asset.get("asset_type") == "text" and need_text:
            text_ids.append(asset["asset_id"])
            text_inputs.append(serialize_text_asset_for_embedding(asset, max_text_chars=args.max_text_chars))
        elif asset.get("asset_type") == "image" and need_image:
            path = image_local_path(input_dir, asset)
            if path is None or not path.exists():
                skipped_images.append({"asset_id": asset.get("asset_id", ""), "reason": "missing_file"})
                continue
            try:
                path, resize_record = ensure_image_within_pixel_limit(
                    path,
                    emb_dir / "resized_images",
                    max_pixels=getattr(args, "max_image_pixels", DEFAULT_MAX_IMAGE_PIXELS),
                )
            except Exception as exc:
                skipped_images.append({"asset_id": asset.get("asset_id", ""), "reason": f"image_resize_failed:{exc}"})
                logging.warning("Skipping image asset %s after resize failure: %s", asset.get("asset_id", ""), exc)
                continue
            if resize_record:
                resize_record["asset_id"] = asset.get("asset_id", "")
                resized_images.append(resize_record)
            image_ids.append(asset["asset_id"])
            image_paths.append(str(path))
            image_prompts.append(serialize_image_asset_prompt(asset))

    if need_text:
        embeddings = encoder.encode_texts(text_inputs, instruction=instructions["text"])
        save_embeddings(emb_dir, "text_asset", text_ids, embeddings)
        stats.append({"object_type": "text_asset", "count": len(text_ids), "dimension": int(embeddings.shape[1]), "cached": False})
    if need_image:
        embeddings = encoder.encode_images(image_paths, prompts=image_prompts, instruction=instructions["image"])
        save_embeddings(emb_dir, "image_asset", image_ids, embeddings)
        write_json(emb_dir / "skipped_images.json", skipped_images)
        write_json(emb_dir / "resized_images.json", resized_images)
        stats.append(
            {
                "object_type": "image_asset",
                "count": len(image_ids),
                "dimension": int(embeddings.shape[1]),
                "cached": False,
                "skipped_images": len(skipped_images),
                "resized_images": len(resized_images),
                "max_image_pixels": getattr(args, "max_image_pixels", DEFAULT_MAX_IMAGE_PIXELS),
            }
        )
    return stats


def run(args: argparse.Namespace) -> None:
    setup_logging()
    stage1_dir = Path(args.stage1_dir)
    emb_dir = stage1_dir / "embeddings"
    encoder = Qwen3VLEmbeddingEncoder(
        encoder_path=args.encoder_path,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        image_resize_cache_dir=str(emb_dir / "resized_images"),
        progress=bool(getattr(args, "progress", True)),
    )
    stats = [encode_table_fragments(args, encoder, emb_dir)]
    if not getattr(args, "table_only", False):
        stats.extend(encode_assets(args, encoder, emb_dir))
    payload = {
        "encoder_path": str(encoder.model_dir),
        "device": args.device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "embedding_prompt_mode": embedding_prompt_mode(args),
        "embedding_instructions": embedding_instructions(args),
        "table_serialization_version": TABLE_SERIALIZATION_VERSION,
        "max_image_pixels": getattr(args, "max_image_pixels", DEFAULT_MAX_IMAGE_PIXELS),
        "table_only": bool(getattr(args, "table_only", False)),
        "objects": stats,
    }
    write_json(emb_dir / "embedding_stats.json", payload)
    update_stage1_manifest(stage1_dir, "embeddings", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="output_medium")
    parser.add_argument("--stage1_dir", default="output_stage1_logic")
    parser.add_argument("--encoder_path", default="./Qwen3-VL-Embedding-2B")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--max_table_rows", type=int, default=5)
    parser.add_argument("--max_text_chars", type=int, default=2048)
    parser.add_argument("--max_image_pixels", type=int, default=DEFAULT_MAX_IMAGE_PIXELS)
    parser.add_argument("--embedding_prompt_mode", choices=EMBEDDING_PROMPT_MODE_CHOICES, default=DEFAULT_EMBEDDING_PROMPT_MODE)
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument("--table_only", action="store_true", help="Only encode table fragments; skip text/image asset embeddings.")
    parser.add_argument("--no_progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
