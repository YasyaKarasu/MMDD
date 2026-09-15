"""Exercise the real R26 template and backend on deterministic engineering probes."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from mmdd_stage2.pipeline import LocalizedEvidence
from mmdd_stage2.r26_generation import R26QwenBackend
from prepare_stage1_r26 import ROOT, OUT, file_record
from run_stage1_r25 import _json


def run() -> dict:
    torch.set_num_threads(2)
    directory = OUT / "stage2/generation_smoke"
    directory.mkdir(parents=True, exist_ok=True)
    image_path = directory / "engineering_label.png"
    image = Image.new("RGB", (640, 240), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
    draw.multiline_text((24, 30), "ITEM: ZX-314\nCOLOR: VIOLET\nBATCH: 2026", fill="black", font=font, spacing=15)
    image.save(image_path)
    cases = [
        {"id": "known_text", "row": {"item": "ZX-314"}, "attribute": "color", "truth": "violet",
         "evidence": LocalizedEvidence("synthetic_text", "text", text="Item ZX-314 has color violet and batch 2026.")},
        {"id": "known_image", "row": {"item": "ZX-314"}, "attribute": "color", "truth": "violet",
         "evidence": LocalizedEvidence("synthetic_image", "image", image=image)},
        {"id": "unknown_text", "row": {"item": "ZX-314"}, "attribute": "private calibration password", "truth": "",
         "evidence": LocalizedEvidence("synthetic_no_answer", "text", text="Item ZX-314 has color violet. Its private calibration password is not provided.")},
    ]
    backend = R26QwenBackend(ROOT / "hf_models/Qwen3.5-9B", device="cuda:1", dtype="bf16")
    _json(directory / "TEMPLATE_AUDIT.json", backend.template_audit)
    outcomes = []
    for case in cases:
        backend.generation_context = {"engineering_probe_id": case["id"], "source": "synthetic_correctness_only_not_32_train_records"}
        try:
            value = backend.generate_value(case["row"], attribute_name=case["attribute"], evidence=case["evidence"])
            outcome = {"id": case["id"], "value": value, "truth": case["truth"], "exact_match_casefold": value.casefold() == case["truth"]}
        except (ValueError, RuntimeError) as exc:
            outcome = {"id": case["id"], "error_type": type(exc).__name__, "exact_match_casefold": False}
        outcomes.append(outcome)
        _json(directory / "RAW_COMPLETIONS.json", backend.generation_records)
        print(json.dumps(outcome), flush=True)
    result = {"outcomes": outcomes, "passed": all(r["exact_match_casefold"] for r in outcomes),
              "template": backend.template_audit, "image": file_record(image_path),
              "scope": "synthetic text/image/abstain backend probes; actual 32 train records still required"}
    _json(directory / "RESULT.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(run()))
