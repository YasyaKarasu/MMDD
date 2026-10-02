"""Frozen pure-feature layer packed from the encoder cache.

Two kinds of pure content exist and nothing else:

* ``z``: the L2-normalised last-valid-token embedding of every dataset object.
* ``content``: per-object token summaries.  Tables keep one token per
  schema/row group; text/image keep up to 64 consecutive-token bin means.

No learned task tensor is ever stored here.  ``content`` summaries are float16;
the retrieval ``z`` stays float32.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

CONTENT_BINS = 64
EMBED_DIM = 4096

# ---------------------------------------------------------------- manifest ---


@dataclass(frozen=True)
class CacheEntry:
    object_id: str
    object_type: str
    path: str


def read_manifest(manifest: Path, key: str) -> list[CacheEntry]:
    out: list[CacheEntry] = []
    with manifest.open() as fh:
        for line in fh:
            row = json.loads(line)
            out.append(CacheEntry(str(row["object_id"]), str(row["object_type"]), str(row[key])))
    return out


# ---------------------------------------------------------------------- z ----


def build_z_memmap(cache_dir: Path, out_dir: Path, *, dtype=np.float32) -> dict[str, object]:
    """Materialise the retrieval tier into one memmap + an id index."""
    entries = read_manifest(cache_dir / "manifest.jsonl", "feature_path")
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(
        out_dir / "z.f32.npy", mode="w+", dtype=dtype, shape=(len(entries), EMBED_DIM)
    )
    for i, entry in enumerate(entries):
        payload = torch.load(cache_dir / entry.path, map_location="cpu", weights_only=False)
        emb = payload["embedding"].to(torch.float32).numpy()
        if emb.shape != (EMBED_DIM,):
            raise ValueError(f"{entry.object_id}: unexpected embedding shape {emb.shape}")
        arr[i] = emb
    arr.flush()
    index = {
        "ids": [e.object_id for e in entries],
        "types": [e.object_type for e in entries],
        "dtype": np.dtype(dtype).name,
        "shape": [len(entries), EMBED_DIM],
    }
    (out_dir / "z_index.json").write_text(json.dumps(index))
    return index


# ------------------------------------------------------------- compression ---


def compress_bins(hidden: torch.Tensor, limit: int = CONTENT_BINS) -> torch.Tensor:
    """SPEC 4.2: keep all tokens when L<=limit, else ``limit`` consecutive bin means."""
    if hidden.ndim != 2 or len(hidden) == 0:
        raise ValueError("non-empty (L, D) token matrix required")
    n = len(hidden)
    if n <= limit:
        return hidden.float()
    edges = [(j * n) // limit for j in range(limit + 1)]
    return torch.stack([hidden[edges[j] : edges[j + 1]].float().mean(0) for j in range(limit)])


def table_tokens(hidden: torch.Tensor) -> torch.Tensor:
    """Tables keep the upstream per schema/row pooled tokens (SPEC 4.2)."""
    if hidden.ndim != 2 or len(hidden) == 0:
        raise ValueError("non-empty (rows+1, D) table token matrix required")
    return hidden.float()


# ------------------------------------------------------------ packed store ---


CHUNK_OBJECTS = 1000


def chunk_stem(chunk_dir: Path, index: int) -> Path:
    return Path(chunk_dir) / f"chunk_{index:06d}"


def chunk_files(chunk_dir: Path, index: int) -> dict[str, Path]:
    stem = chunk_stem(chunk_dir, index)
    return {name: Path(f"{stem}.{name}.npy") for name in ("ids", "types", "lens", "tokens")}


def write_chunk(chunk_dir: Path, index: int, rows: list[tuple[str, str, np.ndarray]]) -> Path:
    """Atomically publish one chunk of ``(object_id, object_type, tokens)`` rows.

    Stored as separate uncompressed ``.npy`` arrays so readers can mmap the
    token block instead of paying a ZIP CRC pass on every random access.
    """
    chunk_dir = Path(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    files = chunk_files(chunk_dir, index)
    ids = np.array([r[0] for r in rows], dtype="<U96")
    types = np.array([r[1] for r in rows], dtype="<U16")
    lens = np.array([len(r[2]) for r in rows], dtype=np.int64)
    tokens = (
        np.concatenate([r[2].astype(np.float16) for r in rows], axis=0)
        if rows
        else np.zeros((0, EMBED_DIM), np.float16)
    )
    for name, array in (("ids", ids), ("types", types), ("lens", lens), ("tokens", tokens)):
        target = files[name]
        tmp = target.with_suffix(".tmp.npy")
        np.save(tmp, array)
        tmp.replace(target)
    return chunk_stem(chunk_dir, index)


def build_chunk_index(chunk_dir: Path) -> dict[str, object]:
    """Index every published chunk; first occurrence of an object id wins."""
    ids: list[str] = []
    types: list[str] = []
    chunks: list[int] = []
    rows: list[int] = []
    seen: set[str] = set()
    duplicates = 0
    for ids_path in sorted(Path(chunk_dir).glob("chunk_*.ids.npy")):
        index = int(ids_path.name.split("_")[1].split(".")[0])
        cids = [str(x) for x in np.load(ids_path, allow_pickle=False)]
        ctypes = [str(x) for x in np.load(chunk_files(chunk_dir, index)["types"], allow_pickle=False)]
        for r, (oid, otype) in enumerate(zip(cids, ctypes)):
            if oid in seen:
                duplicates += 1
                continue
            seen.add(oid)
            ids.append(oid)
            types.append(otype)
            chunks.append(index)
            rows.append(r)
    payload = {
        "ids": ids,
        "types": types,
        "chunks": chunks,
        "rows": rows,
    }
    (Path(chunk_dir).parent / "index.json").write_text(json.dumps(payload))
    (Path(chunk_dir).parent / "coverage.json").write_text(
        json.dumps({"objects": len(ids), "types": dict(Counter(types)), "duplicate_objects_skipped": duplicates})
    )
    return payload


def merge_chunk_dirs(sources: Sequence[Path], dest: Path) -> dict[str, object]:
    """Rebuild one canonical chunk directory from every published chunk dir.

    Destination chunk files are recreated (hard links) and stale links from a
    previous merge are removed first, so a re-merge after a source rewrite
    cannot leave old inodes behind.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for stale in list(dest.glob("chunk_*.npy")) + list(dest.glob("chunk_*.npz")):
        stale.unlink()
    counter = 0
    for source in sources:
        if not Path(source).exists():
            continue
        for ids_path in sorted(Path(source).glob("chunk_*.ids.npy")):
            index = int(ids_path.name.split("_")[1].split(".")[0])
            files = chunk_files(source, index)
            for name, path in files.items():
                target = chunk_files(dest, counter)[name]
                if target.exists():
                    target.unlink()
                os.link(path, target)
            counter += 1
    return build_chunk_index(dest)


class ContentStore:
    """Chunked pure-content token store with random access by object id.

    Layout under ``root``: ``index.json`` plus ``chunks/chunk_*.{ids,lens,tokens}.npy``. Every
    chunk's token block is memory-mapped once on first use and stays mapped; the OS page cache
    is the only token cache. Tokens stay float16 on the host, ``ObjectBank`` expands them to
    float32 after the device transfer.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        payload = json.loads((self.root / "index.json").read_text())
        self.ids: list[str] = payload["ids"]
        self.types: list[str] = payload["types"]
        self.index = {oid: i for i, oid in enumerate(self.ids)}
        self.chunk_dir = self.root / "chunks"
        chunk_of = np.asarray(payload["chunks"], dtype=np.int64)
        row_of = np.asarray(payload["rows"], dtype=np.int64)
        lens = {index: np.load(chunk_files(self.chunk_dir, index)["lens"], allow_pickle=False)
                for index in np.unique(chunk_of).tolist()}
        offsets = {index: np.concatenate([[0], np.cumsum(value)]) for index, value in lens.items()}
        self.chunk_of = chunk_of
        self.length = np.array([lens[c][r] for c, r in zip(chunk_of.tolist(), row_of.tolist())], dtype=np.int64)
        self.start = np.array([offsets[c][r] for c, r in zip(chunk_of.tolist(), row_of.tolist())], dtype=np.int64)
        self._tokens: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.ids)

    def _chunk_tokens(self, index: int) -> np.ndarray:
        if index not in self._tokens:
            self._tokens[index] = np.load(chunk_files(self.chunk_dir, index)["tokens"], mmap_mode="r", allow_pickle=False)
        return self._tokens[index]

    def locate(self, object_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """``(positions, lengths)`` of ``object_ids`` in index order."""
        positions = np.fromiter((self.index[oid] for oid in object_ids), dtype=np.int64, count=len(object_ids))
        return positions, self.length[positions]

    def copy_rows(self, positions: np.ndarray, out: np.ndarray) -> None:
        """Concatenate the token rows of ``positions`` into ``out`` (float16, rows x EMBED_DIM)."""
        cursor = 0
        for position in positions.tolist():
            start, length = int(self.start[position]), int(self.length[position])
            out[cursor : cursor + length] = self._chunk_tokens(int(self.chunk_of[position]))[start : start + length]
            cursor += length

    def get(self, object_id: str) -> torch.Tensor:
        positions, lengths = self.locate([object_id])
        out = np.empty((int(lengths[0]), EMBED_DIM), dtype=np.float16)
        self.copy_rows(positions, out)
        return torch.from_numpy(out)


# --------------------------------------------------------------- generation ---


def load_embedder_class(backbone_dir: Path):
    import importlib.util
    import sys

    script = Path(backbone_dir) / "scripts" / "qwen3_vl_embedding.py"
    if not script.is_file():
        raise FileNotFoundError(f"missing public Qwen wrapper: {script}")
    spec = importlib.util.spec_from_file_location("_mmdd_qwen_embedding", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Qwen3VLEmbedder


def encode_hidden(embedder, items: list[dict]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """One frozen forward; per item the last-layer valid-token states and their input ids."""
    conversations = [
        embedder.format_model_input(
            text=item.get("text"), image=item.get("image"), instruction=item["instruction"]
        )
        for item in items
    ]
    inputs = embedder._preprocess_inputs(conversations)
    inputs = {k: v.to(embedder.model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = embedder.forward(inputs)
    hidden = outputs["last_hidden_state"]
    mask = outputs["attention_mask"].bool()
    return [(hidden[i][mask[i]].cpu(), inputs["input_ids"][i][mask[i]].cpu()) for i in range(hidden.shape[0])]


def table_token_groups(embedder, item: dict, parts: list[str], input_ids: torch.Tensor):
    """Alignment of serialized table parts to token spans (SPEC 4.2)."""
    text = item["text"]
    part_spans = []
    cursor = 0
    for part in parts:
        start = text.find(part, cursor)
        if start < 0:
            raise ValueError("table_parts must occur in order within the table text")
        part_spans.append((start, start + len(part)))
        cursor = start + len(part)
    conversation = embedder.format_model_input(
        text=text, image=None, instruction=item["instruction"]
    )
    rendered = embedder.processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    text_start = rendered.rfind(text)
    if text_start < 0:
        raise ValueError("Qwen chat template did not preserve the serialized table text")
    rendered_spans = [(text_start + s, text_start + e) for s, e in part_spans]
    tokenized = None
    for add_special in (False, True):
        candidate = embedder.processor.tokenizer(
            rendered,
            add_special_tokens=add_special,
            truncation=True,
            max_length=embedder.max_length,
            return_offsets_mapping=True,
        )
        if list(candidate["input_ids"]) == input_ids.tolist():
            tokenized = candidate
            break
    if tokenized is None:
        raise ValueError("tokenizer offsets do not align with Qwen preprocessing")
    selected, groups = [], []
    for token_index, (start, end) in enumerate(tokenized["offset_mapping"]):
        if end <= start:
            continue
        for group, (part_start, part_end) in enumerate(rendered_spans):
            if end > part_start and start < part_end:
                selected.append(token_index)
                groups.append(group)
                break
    if set(groups) != set(range(len(parts))):
        raise ValueError("table truncation removed all tokens from a schema/row group")
    return torch.tensor(selected, dtype=torch.long), torch.tensor(groups, dtype=torch.long)


def pool_table_groups(hidden: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """One mean token per schema/row group (matches upstream table pooling)."""
    out = []
    for group in range(int(groups.max().item()) + 1):
        members = hidden[groups == group]
        if len(members) == 0:
            raise ValueError("empty table token group")
        out.append(members.float().mean(0))
    return torch.stack(out)
