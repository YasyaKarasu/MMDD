#!/usr/bin/env python
"""Read AbeBooks evidence with the local model, without supplying source answers."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from mmdd_dataset.abebooks_ablation import read_rows, write_rows
from mmdd_dataset.abebooks_curation import file_hash, load_artifacts


PROMPT = '''Read only the supplied book cover or author-biography text. Do not use outside knowledge.
Transcribe the complete author names actually printed or explicitly stated. Keep initials as printed.
Do not infer missing coauthors. Separate editors from authors. Do not include foreword writers,
reviewers, endorsers, publishers, series editors, or a person mentioned only incidentally.
For a cover also transcribe its main book title. If illegible or absent, use an empty string/list.
For biography text, list the people whose author biographies it presents, and quote their names.
Return JSON only: {"title":"", "authors":["Full name"], "editors":[], "basis":"short visible quote or image region"}'''

PUBLISHER_PROMPT = '''Read only this book cover. Transcribe its main title and the publisher or
imprint text actually visible on this image. Do not infer a publisher from the title, author,
series, book design, or outside knowledge. If a logo contains only letters such as MK, AP,
or BH, transcribe those letters without expanding them to a publisher name. A parent-company
logo does not establish a specific subsidiary imprint. Ignore authors, endorsements and
software product logos. Use an empty list when no publisher text or readable logo lettering
is visible. Return JSON only:
{"title":"", "publisher_text":[], "logo_text":[], "basis":"visible quote and image region"}'''


def prepare(dataset: Path, output: Path) -> None:
    from PIL import Image

    _, data = load_artifacts(dataset)
    books = {s["source_table_id"] for s in data["source_tables"] if s["source_file"] == "book"}
    by_row: dict[tuple, list[dict]] = {}
    for asset in data["bridge_assets"]:
        if asset["source_table_id"] in books:
            by_row.setdefault((asset["source_table_id"], asset["source_row_id"]), []).append(asset)
    packets = []
    for loc, assets in sorted(by_row.items()):
        images = []
        texts = []
        for a in assets:
            if a["asset_type"] == "image":
                with Image.open(a["local_path"]) as im:
                    # Prefer the full, legible cover; selection does not use labels.
                    images.append((min(im.width, im.height), im.width * im.height, a["asset_id"], a))
            elif a.get("source") == "abebooks_about_author":
                texts.append(a)
        chosen = [max(images)[-1]] if images else []
        chosen += sorted(texts, key=lambda a: a["asset_id"])[:1]
        for a in chosen:
            packets.append({"asset_id": a["asset_id"], "source_table_id": loc[0], "source_row_id": loc[1],
                            "modality": a["asset_type"], "image": a.get("local_path"),
                            "text": a.get("content") if a["asset_type"] == "text" else None,
                            "content_sha256": file_hash(Path(a["local_path"])) if a["asset_type"] == "image" else hashlib.sha256(a["content"].encode()).hexdigest()})
    if output.exists():
        raise FileExistsError(output)
    write_rows(output, packets)
    print(json.dumps({"packets": len(packets), "rows": len(by_row),
                      "images": sum(p["modality"] == "image" for p in packets)}, indent=2))


def infer(args: argparse.Namespace) -> None:
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    packets = read_rows(args.packets)
    prompt_text = PUBLISHER_PROMPT if args.attribute == "publisher" else PROMPT
    if args.modality:
        packets = [p for p in packets if p["modality"] == args.modality]
    if args.limit:
        packets = packets[:args.limit]
    done = {r["asset_id"] for r in read_rows(args.output)} if args.output.exists() else set()
    remaining = [p for p in packets if p["asset_id"] not in done]
    if not remaining:
        print("All requested evidence already read.")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = LLM(model=str(args.model), tensor_parallel_size=args.tensor_parallel_size,
                dtype="bfloat16", max_model_len=4096, max_num_seqs=16,
                gpu_memory_utilization=0.85, enforce_eager=True,
                limit_mm_per_prompt={"image": 1, "video": 0},
                mm_processor_kwargs={"max_pixels": 589824})
    sampling = SamplingParams(temperature=0, max_tokens=220, seed=13)
    started = time.monotonic()
    with args.output.open("a") as handle:
        for start in range(0, len(remaining), 32):
            batch = remaining[start:start + 32]
            requests = []
            for packet in batch:
                content = [{"type": "text", "text": prompt_text}]
                image = None
                if packet["modality"] == "image":
                    content.insert(0, {"type": "image"})
                    with Image.open(packet["image"]) as im:
                        image = im.convert("RGB")
                    image.thumbnail((896, 896))
                else:
                    content.append({"type": "text", "text": packet["text"]})
                prompt = processor.apply_chat_template([{"role": "user", "content": content}],
                            tokenize=False, add_generation_prompt=True, enable_thinking=False)
                request = {"prompt": prompt}
                if image is not None:
                    request["multi_modal_data"] = {"image": image}
                requests.append(request)
            responses = model.generate(requests, sampling, use_tqdm=False)
            for packet, response in zip(batch, responses):
                raw = response.outputs[0].text
                try:
                    parsed = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
                except json.JSONDecodeError:
                    parsed = None
                result = {"asset_id": packet["asset_id"], "source_table_id": packet["source_table_id"],
                          "source_row_id": packet["source_row_id"], "modality": packet["modality"],
                          "content_sha256": packet["content_sha256"], "response": parsed, "raw_response": raw,
                          "model": str(args.model), "annotation_status": "local_model_proposal",
                          "source_answer_provided": False, "attribute": args.attribute,
                          "prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
                          "output_tokens": len(response.outputs[0].token_ids)}
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"completed": min(start + len(batch), len(remaining)), "total": len(remaining),
                              "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--dataset", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("infer")
    run.add_argument("--packets", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--tensor-parallel-size", type=int, default=2)
    run.add_argument("--modality", choices=("image", "text"))
    run.add_argument("--limit", type=int)
    run.add_argument("--attribute", choices=("authors", "publisher"), default="authors")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.dataset, args.output)
    else:
        infer(args)


if __name__ == "__main__":
    main()
