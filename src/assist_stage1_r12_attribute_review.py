#!/usr/bin/env python
"""Produce local model-assisted annotations, never independent confirmed labels."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json


LABELS = {"correct_support", "same_entity_wrong_attribute", "entity_or_value_conflict", "insufficient_information"}


def review_prompt(packet: dict) -> str:
    material = {
        "visible_row": packet["visible_row"], "requested_attribute": packet["requested_attribute"],
        "target_schema": packet["target_schema"], "evidence_text": packet["evidence_text"],
    }
    return (
        "Audit whether the supplied evidence independently supports the requested attribute for the exact entity in the row. "
        "Use only the supplied text or image. Do not infer an answer from world knowledge, a URL, an image filename, "
        "the target schema, or the row itself. Entity relevance alone is not attribute support. "
        "For an image, visible text, logos, or identifiable depicted facts must establish the requested value; "
        "a generic portrait/building or recognition from memory is not sufficient. "
        "Labels: correct_support requires same entity AND a concrete attribute value AND a locatable basis; "
        "same_entity_wrong_attribute requires clear same-entity evidence about other attributes, not the requested one; "
        "entity_or_value_conflict requires explicit incompatible entity or value evidence; "
        "insufficient_information for all other cases. A missing fact is not a conflict. "
        "Return one JSON object with label, entity_match (true/false/null), extracted_value (string/null), "
        "evidence_quote (exact text substring or null), image_location (description of specific visual region or null), "
        "reason (brief evidence-based explanation). No markdown.\n" + json.dumps(material, ensure_ascii=False)
    )


def parse_assistance(response: str, packet: dict) -> dict:
    try:
        value = json.loads(response)
    except json.JSONDecodeError:
        return {"parse_status": "invalid_json", "proposed_label": None, "basis_check": False}
    if not isinstance(value, dict) or value.get("label") not in LABELS:
        return {"parse_status": "invalid_label", "proposed_label": None, "basis_check": False}
    quote = value.get("evidence_quote")
    basis_check = bool(
        isinstance(quote, str) and quote and quote in (packet["evidence_text"] or "")
        if packet["modality"] == "text" else value.get("image_location")
    )
    return {"parse_status": "valid", "proposed_label": value["label"], "basis_check": basis_check, "annotation": value}


def run(args):
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    torch.set_num_threads(2)
    started = time.monotonic()
    packets = [json.loads(line) for line in args.packets.open()]
    output = args.output_root / "taskB_attribute_audit/model_assistance.jsonl"
    if output.exists():
        raise FileExistsError("Preserve existing model assistance; do not silently overwrite it")
    model_dir = args.root / "hf_models/Qwen3.5-9B"
    configuration = {
        "role": "local model assistance only, not independent human review or ground truth",
        "model_dir": str(model_dir), "model_config_sha256": checkpoint_fingerprint(model_dir / "config.json"),
        "packets": str(args.packets), "packets_sha256": checkpoint_fingerprint(args.packets),
        "device": args.device, "dtype": "bfloat16", "do_sample": False, "max_new_tokens": 320,
        "enable_thinking": False, "image_max_pixels": 589824, "remote_services": [],
        "protected_env_file_loaded": False, "command": [sys.executable, *sys.argv],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output.with_suffix(".config.json"), configuration)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(args.device).eval()
    model.requires_grad_(False)
    processed = 0
    with output.open("w") as handle, torch.inference_mode():
        for packet in packets:
            begin = time.monotonic()
            prompt = review_prompt(packet)
            content = []
            if packet["modality"] == "image":
                content.append({"type": "image", "image": packet["image_path"], "max_pixels": 589824})
            content.append({"type": "text", "text": prompt})
            inputs = processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", enable_thinking=False,
            )
            inputs = {key: value.to(args.device) for key, value in inputs.items()}
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=320)
            response = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
            record = {"case_id": packet["case_id"], "role": "model_assistance_only", "prompt": prompt,
                      "raw_response": response, **parse_assistance(response, packet),
                      "input_tokens": int(inputs["input_ids"].shape[1]),
                      "output_tokens": int(generated.shape[1] - inputs["input_ids"].shape[1]),
                      "elapsed_seconds": time.monotonic() - begin, "independently_confirmed": False}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            processed += 1
            if processed % 16 == 0:
                print(json.dumps({"processed": processed, "total": len(packets), "elapsed_seconds": time.monotonic() - started}), flush=True)
            del inputs, generated
    write_json(output.with_suffix(".summary.json"), {"status": "model_assistance_complete_human_review_pending",
               "records": processed, "elapsed_seconds": time.monotonic() - started, "configuration": configuration,
               "output_sha256": checkpoint_fingerprint(output)})
    with (args.output_root / "runs.jsonl").open("a") as handle:
        handle.write(json.dumps({"task": "B model assistance", "status": "assistance_only",
                                 "command": configuration["command"], "output": str(output),
                                 "elapsed_seconds": time.monotonic() - started}) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    run(parser.parse_args())
