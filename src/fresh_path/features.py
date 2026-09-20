"""Frozen pure-feature layer (SPEC 4).

Two kinds of pure content exist and nothing else:

* ``z``: the L2-normalised last-valid-token embedding of every dataset object.
* ``content``: per-object token summaries.  Tables keep one token per
  schema/row group; text/image keep up to 64 consecutive-token bin means.

No learned task tensor is ever stored here.  ``content`` summaries produced by
the current run are float16 (SPEC 4.2); the retrieval ``z`` stays float32.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

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


def object_order(cache_dir: Path) -> list[str]:
    """Canonical object order: manifest order (itself UTF-8 id sorted upstream)."""
    ids = [e.object_id for e in read_manifest(cache_dir / "manifest.jsonl", "feature_path")]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate object ids in the retrieval manifest")
    return ids


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


def build_chunk_index(chunk_dir: Path, *, compact: bool = False) -> dict[str, object]:
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
        json.dumps({"objects": len(ids), "types": compact_types(types), "duplicate_objects_skipped": duplicates})
    )
    return payload


def merge_chunk_dirs(sources: Sequence[Path], dest: Path) -> dict[str, object]:
    """Rebuild one canonical chunk directory from every published chunk dir.

    Destination chunk files are recreated (hard links) and stale links from a
    previous merge are removed first, so a re-merge after a source rewrite
    cannot leave old inodes behind.
    """
    import os

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

    Layout under ``root``: ``index.json`` plus ``chunks/chunk_*.{ids,lens,tokens}.npy``.  Chunks
    are immutable once published; readers never look at a partial file.
    """

    def __init__(self, root: Path, *, lru_bytes: int = 0) -> None:
        self.root = Path(root)
        payload = json.loads((self.root / "index.json").read_text())
        self.ids: list[str] = payload["ids"]
        self.types: list[str] = payload["types"]
        self.chunks = payload["chunks"]
        self.rows = payload["rows"]
        self.index = {oid: i for i, oid in enumerate(self.ids)}
        self.chunk_dir = self.root / "chunks"
        self._open: dict[int, dict[str, np.ndarray]] = {}
        self._lru: dict[str, torch.Tensor] = {}
        self._lru_used = 0
        self._lru_limit = lru_bytes

    def has(self, object_id: str) -> bool:
        return object_id in self.index

    def __len__(self) -> int:
        return len(self.ids)

    def _chunk(self, index: int) -> dict[str, np.ndarray]:
        if index not in self._open:
            files = chunk_files(self.chunk_dir, index)
            lens = np.load(files["lens"], allow_pickle=False)
            self._open[index] = {
                "ids": np.load(files["ids"], allow_pickle=False),
                "lens": lens,
                "offsets": np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
                "tokens": np.load(files["tokens"], mmap_mode="r", allow_pickle=False),
            }
            if len(self._open) > 8:
                self._open.pop(next(iter(self._open)))
        return self._open[index]

    def get(self, object_id: str) -> torch.Tensor:
        cached = self._lru.pop(object_id, None)
        if cached is not None:
            self._lru[object_id] = cached
            return cached
        i = self.index[object_id]
        chunk = self._chunk(int(self.chunks[i]))
        row = int(self.rows[i])
        start = int(chunk["offsets"][row])
        length = int(chunk["lens"][row])
        # Keep the contracted float16 representation on the CPU.  ObjectBank
        # expands it only after host-to-device transfer, preserving the exact
        # float32 model input while halving transfer and CPU-cache bytes.
        tensor = torch.from_numpy(np.array(chunk["tokens"][start : start + length], copy=True))
        if self._lru_limit:
            nbytes = tensor.numel() * tensor.element_size()
            while self._lru and self._lru_used + nbytes > self._lru_limit:
                victim_id = next(iter(self._lru))
                victim = self._lru.pop(victim_id)
                self._lru_used -= victim.numel() * victim.element_size()
            if nbytes <= self._lru_limit:
                self._lru[object_id] = tensor
                self._lru_used += nbytes
        return tensor

    def ids_by_type(self, object_type: str) -> list[str]:
        return [oid for oid, t in zip(self.ids, self.types) if t == object_type]


# -------------------------------------------------- reuse of pure token cache ---


def extract_teacher_content(
    cache_dir: Path,
    chunk_dir: Path,
    *,
    limit: int = CONTENT_BINS,
    start_chunk: int = 0,
    only: set[str] | None = None,
) -> dict[str, object]:
    """Compress the upstream full-token pure cache into this run's contract.

    Tables keep the upstream per schema/row pooled tokens unchanged; text and
    image tokens are binned to at most ``limit`` consecutive bin means.
    """
    entries = read_manifest(cache_dir / "teacher_manifest.jsonl", "teacher_feature_path")
    if only is not None:
        entries = [e for e in entries if e.object_id in only]
    rows: list[tuple[str, str, np.ndarray]] = []
    chunk = start_chunk
    total_tokens = 0
    for entry in entries:
        payload = torch.load(cache_dir / entry.path, map_location="cpu", weights_only=False)
        hidden = payload["hidden_states"].float()
        tokens = table_tokens(hidden) if entry.object_type == "table" else compress_bins(hidden, limit)
        arr = tokens.numpy().astype(np.float16)
        rows.append((entry.object_id, entry.object_type, arr))
        total_tokens += len(arr)
        if len(rows) >= CHUNK_OBJECTS:
            write_chunk(chunk_dir, chunk, rows)
            chunk += 1
            rows = []
    if rows:
        write_chunk(chunk_dir, chunk, rows)
        chunk += 1
    return {"objects": len(entries), "tokens": total_tokens, "chunks": chunk - start_chunk}


# --------------------------------------------------------------- generation ---


def load_embedder_class(backbone_dir: Path):
    import importlib.util
    import sys

    script = Path(backbone_dir) / "scripts" / "qwen3_vl_embedding.py"
    if not script.is_file():
        raise FileNotFoundError(f"missing public Qwen wrapper: {script}")
    spec = importlib.util.spec_from_file_location("_fresh_path_qwen", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Qwen3VLEmbedder


def encode_hidden(embedder, items: list[dict]) -> list[torch.Tensor]:
    """One frozen forward; return the last-layer valid-token states per item."""
    return [h for h, _ in _encode(embedder, items, with_ids=False)]


def encode_hidden_ids(embedder, items: list[dict]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return _encode(embedder, items, with_ids=True)


def _encode(embedder, items: list[dict], *, with_ids: bool):
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
    out = []
    for i in range(hidden.shape[0]):
        valid = hidden[i][mask[i]].cpu()
        ids = inputs["input_ids"][i][mask[i]].cpu() if with_ids else None
        out.append((valid, ids))
    return out


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



def compact_types(types: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in types:
        counts[t] = counts.get(t, 0) + 1
    return counts


def content_key_index(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with Path(path).open() as fh:
        for line in fh:
            row = json.loads(line)
            out[str(row["object_id"])] = str(row["content_key"])
    return out


def alias_groups(content_keys: dict[str, str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for oid, key in content_keys.items():
        groups.setdefault(key, []).append(oid)
    return {k: sorted(v, key=lambda x: x.encode("utf-8")) for k, v in groups.items()}


def canonical_alias(content_keys: dict[str, str]) -> dict[str, str]:
    """Map every object id to the UTF-8 smallest id sharing its content key."""
    canon: dict[str, str] = {}
    for members in alias_groups(content_keys).values():
        first = members[0]
        for m in members:
            canon[m] = first
    return canon


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def prune_shards(shard_dir: Path) -> None:
    if shard_dir.exists():
        shutil.rmtree(shard_dir)


def iter_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open() as fh:
        for line in fh:
            yield json.loads(line)
