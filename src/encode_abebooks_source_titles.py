"""Encode all source book titles locally for label-independent table grouping."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from mmdd_dataset.abebooks_source_rebuild import without_series_notes
from run_abebooks_fresh import ROOT, finish, launch, read_rows, write_json, write_rows


def encode_titles(source: Path, output: Path, model: Path) -> None:
    """Encode title text only; these vectors are never retrieval/training inputs."""
    (output / "isolated_cwd").mkdir(parents=True, exist_ok=False)
    entries = sorted((table["source_table_id"], row["row_id"], cell["text"])
                     for table in read_rows(source) for row in table["rows"]
                     for cell in row["cells"] if cell["column_name"] == "title")
    records = [{"object_id": f"source-title:{sid}:{row}", "object_type": "text",
                "text": without_series_notes(title)} for sid, row, title in entries]
    write_rows(output / "titles.jsonl", records)
    instruction = "Represent the subject and topic of this book title for organizing a library catalog. Use only the given title."
    finish("encode_titles", launch(output, "encode_titles", [str(ROOT / "src/cache_stage1_features.py"),
        "--input-jsonl", str(output / "titles.jsonl"), "--output-dir", str(output / "encoder"),
        "--model-dir", str(model), "--device", "cuda:0", "--object-batch-size", "16",
        "--instruction", instruction], gpu=1))
    manifest = {r["object_id"]: r for r in read_rows(output / "encoder/manifest.jsonl")}
    vectors = np.stack([torch.load(output / "encoder" / manifest[r["object_id"]]["feature_path"],
                                   weights_only=True)["embedding"].numpy() for r in records])
    np.savez(output / "title_embeddings.npz", source_ids=np.array([e[0] for e in entries]),
             row_ids=np.array([e[1] for e in entries]), titles=np.array([e[2] for e in entries]),
             vectors=vectors)
    write_json(output / "TITLE_EMBEDDINGS.json", {"source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "rows": len(entries),
        "model": str(model), "instruction": instruction, "series_notes_removed": True,
        "labels_used": False, "gpu": 1, "use": "Source grouping only; retrieval method unchanged"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tables", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "hf_models/Qwen3-VL-Embedding-8B")
    args = parser.parse_args()
    encode_titles(args.source_tables.resolve(), args.output_dir.resolve(), args.model_dir.resolve())


if __name__ == "__main__":
    main()
