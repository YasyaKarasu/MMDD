"""Resume the fixed pilot with a bounded decoder limit for its large source image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

from PIL import Image

from audit_stage2_r26_engineering import DATASET
from mmdd_stage2.data import load_stage2_index
from mmdd_stage2.verifier import build_evidence_bundles
from prepare_stage1_r26 import OUT,file_record
from run_stage1_r21 import read_rows
from run_stage1_r25 import _json
from run_stage2_r26 import run


def resume(generator: str) -> dict:
    source = OUT / "stage2/inputs" / generator / "retrieval.jsonl"
    inputs = list(read_rows(source))
    bundles = [b for row in inputs for b in build_evidence_bundles(row["results"],top_k_evidence=4)]
    objects = load_stage2_index(DATASET,query_ids={r["query_id"] for r in inputs},
        target_ids={r["target_id"] for row in inputs for r in row["results"]},
        evidence_ids={e for b in bundles for e in b.evidence_ids})
    original_limit = Image.MAX_IMAGE_PIXELS
    # Pillow warns above this threshold and raises above twice this threshold.
    # The known 273,040,200-pixel local source fits below the finite 300M cap.
    Image.MAX_IMAGE_PIXELS = 150_000_000
    large = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",Image.DecompressionBombWarning)
        for oid,evidence in objects.evidence.items():
            if evidence.get("asset_type") != "image":
                continue
            path = Path(evidence["local_path"])
            with Image.open(path) as image:
                width,height = image.size
            if width*height > original_limit:
                large.append({"evidence_id":oid,"width":width,"height":height,"source":file_record(path)})
    directory = OUT / "stage2/pilot" / generator
    _json(directory / "LARGE_IMAGE_DECODE_RECEIPT.json",{
        "execution_status":"audited_headers_before_resume","input":file_record(source),
        "prior_results":file_record(directory / "results.jsonl"),"source_images":large,
        "prior_pillow_warning_pixels":original_limit,"pillow_warning_pixels":150_000_000,"pillow_error_pixels":300_000_000,
        "pixel_transformation":"none; same original bytes, first-frame RGB, existing processor and existing reader OOM resize policy",
        "resume_scope":"Only the input decoder acceptance limit changes; completed query-condition results remain intact under their original signatures.",
        "code":file_record(Path(__file__))})
    return run(generator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator",choices=("Qwen-Raw","B13"),default="B13")
    print(json.dumps(resume(parser.parse_args().generator)))
