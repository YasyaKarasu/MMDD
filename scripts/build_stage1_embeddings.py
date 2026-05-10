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

TABLE_INSTRUCTION = "Represent the logical statement of this table for retrieval. Focus on how this table can connect to another table or multimodal evidence."
TEXT_INSTRUCTION = "Represent this text as evidence for recovering hidden table attributes and logical connections."
IMAGE_INSTRUCTION = "Represent this image as evidence for multimodal table discovery. Focus on what factual attributes about the entity can be inferred from the image."


def save_embeddings(out_dir: Path, object_type: str, ids: list[str], embeddings: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{object_type}.npy", embeddings.astype("float32"))
    write_json(out_dir / f"{object_type}_ids.json", ids)


def existing(out_dir: Path, object_type: str, force: bool) -> bool:
    return not force and (out_dir / f"{object_type}.npy").exists() and (out_dir / f"{object_type}_ids.json").exists()


def encode_table_fragments(args: argparse.Namespace, encoder: Qwen3VLEmbeddingEncoder, emb_dir: Path) -> dict[str, Any]:
    object_type = "table_fragment"
    if existing(emb_dir, object_type, args.force_recompute):
        ids = json.loads((emb_dir / f"{object_type}_ids.json").read_text(encoding="utf-8"))
        arr = np.load(emb_dir / f"{object_type}.npy", mmap_mode="r")
        return {"object_type": object_type, "count": len(ids), "dimension": int(arr.shape[1]), "cached": True}
    fragments = list(iter_jsonl(Path(args.stage1_dir) / "logic_fragments.jsonl"))
    ids = [frag["fragment_id"] for frag in fragments]
    texts = [serialize_table_for_embedding(frag, max_rows=args.max_table_rows) for frag in fragments]
    embeddings = encoder.encode_tables(texts, instruction=TABLE_INSTRUCTION)
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
    need_text = not existing(emb_dir, "text_asset", args.force_recompute)
    need_image = not existing(emb_dir, "image_asset", args.force_recompute)
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
        embeddings = encoder.encode_texts(text_inputs, instruction=TEXT_INSTRUCTION)
        save_embeddings(emb_dir, "text_asset", text_ids, embeddings)
        stats.append({"object_type": "text_asset", "count": len(text_ids), "dimension": int(embeddings.shape[1]), "cached": False})
    if need_image:
        embeddings = encoder.encode_images(image_paths, prompts=image_prompts, instruction=IMAGE_INSTRUCTION)
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
    )
    stats = [encode_table_fragments(args, encoder, emb_dir)]
    stats.extend(encode_assets(args, encoder, emb_dir))
    payload = {
        "encoder_path": str(encoder.model_dir),
        "device": args.device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "max_image_pixels": getattr(args, "max_image_pixels", DEFAULT_MAX_IMAGE_PIXELS),
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
    parser.add_argument("--force_recompute", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
