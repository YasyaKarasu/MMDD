"""Controlled text localization within the recovery reader's existing character prefix.

The model arm shares the frozen recovery model. Evidence precedes the marked row/attribute,
so causal value features at the condition tokens can incorporate the evidence. No generation,
target cells, labels, or auxiliary model are used by the localizer.
"""
from __future__ import annotations

import re
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .common import digest
from .localizer import ATTR_CLOSE, ATTR_OPEN, ROW_CLOSE, ROW_OPEN, escape


def token_windows(length: int, size: int, overlap: int) -> list[tuple[int, int]]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("window size must exceed overlap >= 0")
    windows = []
    for start in range(0, length, size - overlap):
        end = min(length, start + size)
        windows.append((start, end))
        if end == length:
            break
    return windows


def peak_span(scores: np.ndarray, budget: int) -> tuple[int, int]:
    """Grow a contiguous span from the first maximum towards the stronger adjacent token."""
    if budget < 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("span needs finite nonempty scores and a positive budget")
    left = int(np.argmax(scores))
    right = left + 1
    while right - left < min(budget, len(scores)):
        if left > 0 and (right == len(scores) or scores[left - 1] >= scores[right]):
            left -= 1
        else:
            right += 1
    return left, right


def conditional_relevance(layers: list[torch.Tensor], evidence_count: int, row_count: int,
                          mode: str = "joint") -> np.ndarray:
    """Per layer: cosine to mean row / attribute V, minmax separately, then multiply."""
    maps = []
    for values in layers:
        evidence = F.normalize(values[:evidence_count].float(), dim=-1)
        row = values[evidence_count:evidence_count + row_count].float().mean(0)
        attribute = values[evidence_count + row_count:].float().mean(0)
        def relevance(condition):
            x = evidence @ F.normalize(condition, dim=0)
            return (x - x.min()) / (x.max() - x.min()).clamp_min(1e-8)
        entity_map, attribute_map = relevance(row), relevance(attribute)
        maps.append(entity_map * attribute_map if mode == "joint"
                    else entity_map if mode == "entity" else attribute_map)
    return torch.stack(maps).mean(0).cpu().numpy()


class TextSpanSelector:
    def __init__(self, processor: Any, model: Any, policy: dict[str, Any]) -> None:
        self.tokenizer, self.model = processor.tokenizer, model
        self.policy = {"mode": "prefix", "span_tokens": 192, "window_tokens": 1024,
                       "overlap_tokens": 128, "layers": [15, 23, 27], "max_input_tokens": 8192, **policy}
        if self.policy["mode"] not in {"prefix", "lexical", "joint", "entity", "attribute"}:
            raise ValueError("unknown text span mode")
        token_windows(1, self.policy["window_tokens"], self.policy["overlap_tokens"])
        if self.policy["span_tokens"] < 1 or self.policy["span_tokens"] > self.policy["window_tokens"]:
            raise ValueError("span budget must be positive and fit within a window")
        self.cache: dict[str, dict[str, Any]] = {}
        self.forwards = 0
        self.seconds = 0.0

    def select(self, row: dict, attribute: str, content: str) -> dict[str, Any]:
        key = digest([row, attribute, content, self.policy])
        if key in self.cache:
            return {**self.cache[key], "cache_hit": True, "seconds": 0.0, "forwards": 0}
        started, initial = time.perf_counter(), self.forwards
        # Escape control tokens before making spans; offsets below refer to this sanitized prefix.
        text = escape(content, self.tokenizer.all_special_tokens)
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        mode = self.policy["mode"]
        result = {"text": text, "mode": mode, "source_sha256": digest(content),
                  "input_tokens": len(offsets), "start_char": 0, "end_char": len(text),
                  "selected_tokens": len(offsets), "windows": 0, "reason": "SHORT_TEXT"}
        budget = self.policy["span_tokens"]
        if len(offsets) > budget and mode != "prefix":
            candidates = []
            for first, last in token_windows(len(offsets), self.policy["window_tokens"],
                                            self.policy["overlap_tokens"]):
                lo, hi = offsets[first][0], offsets[last - 1][1]
                if mode == "lexical":
                    condition = " ".join(f"{c['column_name']} {c['text']}" for c in row["cells"])
                    terms = set(re.findall(r"\w+", (condition + " " + attribute).casefold()))
                    word_spans = [(m.start(), m.end()) for m in re.finditer(r"\w+", text[lo:hi])
                                  if m.group().casefold() in terms]
                    scores = np.array([sum(a < end - lo and b > start - lo for a, b in word_spans)
                                       for start, end in offsets[first:last]], dtype=float)
                    local_offsets = [(a - lo, b - lo) for a, b in offsets[first:last]]
                else:
                    scores, local_offsets = self._model_scores(row, attribute, text[lo:hi])
                left, right = peak_span(scores, budget)
                candidates.append((float(scores[left:right].sum()), lo + local_offsets[left][0],
                                   lo + local_offsets[right - 1][1], right - left))
            score, start, end, count = max(candidates, key=lambda x: (x[0], -x[1]))
            result.update(text=text[start:end], start_char=start, end_char=end, selected_tokens=count,
                          windows=len(candidates), relevance_sum=score,
                          reason="SELECTED" if score > 0 else "FLAT_PREFIX_SPAN")
        result.update(seconds=time.perf_counter() - started, forwards=self.forwards - initial, cache_hit=False)
        self.seconds += result["seconds"]
        self.cache[key] = result
        return result

    def _model_scores(self, row: dict, attribute: str, text: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
        tokenizer, model = self.tokenizer, self.model
        row_text = escape(" | ".join(f"{c['column_name']}={c['text']}" for c in row["cells"]),
                          tokenizer.all_special_tokens)
        attribute = escape(attribute, tokenizer.all_special_tokens)
        prefix = "Locate evidence for the query row and requested attribute. Do not generate an answer.\nEVIDENCE:\n"
        content = (prefix + text + "\nENTITY: " + ROW_OPEN + row_text + ROW_CLOSE
                   + "\nATTRIBUTE: " + ATTR_OPEN + attribute + ATTR_CLOSE)
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                               add_generation_prompt=False, enable_thinking=False)
        inputs = tokenizer(prompt, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=False)
        offsets = inputs.pop("offset_mapping")[0].tolist()
        ids = inputs["input_ids"][0].tolist()
        if len(ids) > self.policy["max_input_tokens"]:
            raise ValueError("text-localizer input exceeds declared token budget")
        start = prompt.index(prefix) + len(prefix)
        evidence = [i for i, (a, b) in enumerate(offsets) if a < start + len(text) and b > start]
        def marked(opening, closing):
            a, b = ids.index(tokenizer.convert_tokens_to_ids(opening)), ids.index(tokenizer.convert_tokens_to_ids(closing))
            if b <= a + 1:
                raise ValueError("empty or invalid localizer condition")
            return list(range(a + 1, b))
        entity = marked(ROW_OPEN, ROW_CLOSE)
        attr = marked(ATTR_OPEN, ATTR_CLOSE)
        device = next(model.parameters()).device
        selected = torch.tensor(evidence + entity + attr, device=device)
        captured, handles = {}, []
        layers = model.model.language_model.layers
        previous_rope = getattr(model.model, "rope_deltas", None)
        try:
            for layer in self.policy["layers"]:
                def hook(_module, _args, output, index=layer):
                    captured[index] = output.detach()[0].index_select(0, selected).float()
                handles.append(layers[layer].self_attn.v_proj.register_forward_hook(hook))
            with torch.inference_mode():
                model.model(**{k: v.to(device) for k, v in inputs.items()}, use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
            model.model.rope_deltas = previous_rope
        self.forwards += 1
        scores = conditional_relevance([captured[i] for i in self.policy["layers"]], len(evidence), len(entity),
                                       self.policy["mode"])
        return scores, [(max(0, offsets[i][0] - start), min(len(text), offsets[i][1] - start)) for i in evidence]
