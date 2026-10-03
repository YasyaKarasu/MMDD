"""Attribute-conditioned image crops for value recovery (RAEA-Attr + ConsensusMask).

For one (query row, requested attribute, image) the frozen generator reads the image with a prompt
holding the full serialized row and the attribute between ``<|box_start|>``/``<|box_end|>``. For each
attribute token and each fixed full-attention layer (15/23/27, 0-based) the cosine between the
token's v_proj value and every image token's value is a spatial map; maps are averaged over layers
and min-max normalized per token. RAEA weights the token maps by reliability
(``concentration + 1.2 * consistency + 0.6 * peak prominence``, softmaxed), ConsensusMask
multiplies that with the spatial median raised to 0.8.

Only the top local ROI proposal is used, and only when the map is not diffuse and the ROI's mean
density reaches 1.5x uniform; otherwise the original image stays alone. An accepted ROI yields a
padded context crop and a tight zoom, both shown to the generator after the original image as
extra views of the same source. Nothing here is trained or label-aware.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image, ImageOps

ROW_OPEN, ROW_CLOSE = "<|object_ref_start|>", "<|object_ref_end|>"
ATTR_OPEN, ATTR_CLOSE = "<|box_start|>", "<|box_end|>"


def decode_original(path: str) -> Image.Image:
    with Image.open(path) as image:
        image.seek(0)
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        rgb.load()
    return rgb


def bounded_rgb(image: Image.Image, max_pixels: int) -> Image.Image:
    image = image.convert("RGB")
    if image.width * image.height > max_pixels:
        r = math.sqrt(max_pixels / (image.width * image.height))
        image = image.resize((max(1, int(image.width * r)), max(1, int(image.height * r))), Image.Resampling.LANCZOS)
    return image


# ---------------------------------------------------------------- map statistics

def probability(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or (x < 0).any() or float(x.sum()) <= 1e-12:
        raise ValueError("non-finite, negative or empty heatmap")
    return x / x.sum()


def minmax_map(similarity: np.ndarray, epsilon: float) -> np.ndarray | None:
    x = np.asarray(similarity, dtype=np.float64)
    spread = float(x.max() - x.min())
    return None if spread <= epsilon else probability((x - x.min()) / spread)


def concentration(p: np.ndarray) -> float:
    """Normalized inverse entropy."""
    p = probability(p)
    return 1.0 if p.size == 1 else float(np.clip(1.0 + np.sum(p * np.log(np.maximum(p, 1e-300))) / math.log(p.size), 0, 1))


def centroid(p: np.ndarray) -> np.ndarray:
    p = probability(p)
    h, w = p.shape
    yy, xx = np.mgrid[0:h, 0:w]
    return np.array([(p * yy).sum() / max(h - 1, 1), (p * xx).sum() / max(w - 1, 1)], dtype=np.float64)


def peak_prominence(p: np.ndarray, eps: float) -> float:
    p = probability(p)
    top, median = float(p.max()), float(np.median(p))
    return float(np.clip((top - median) / (top + eps), 0, 1))


def aggregate_raea(token_maps: list[np.ndarray], epsilon: float) -> dict[str, Any]:
    if not token_maps:
        return {"status": "NO_ATTRIBUTE_TOKENS", "consensus": None}
    ps = [probability(x) for x in token_maps]
    centers = np.stack([centroid(x) for x in ps])
    middle = np.median(centers, axis=0)
    c = np.array([concentration(x) for x in ps])
    a = np.array([np.clip(1 - np.linalg.norm(mu - middle) / math.sqrt(2), 0, 1) for mu in centers])
    p = np.array([peak_prominence(x, epsilon) for x in ps])
    reliability = c + 1.2 * a + 0.6 * p
    weights = np.exp(reliability - reliability.max())
    weights /= weights.sum()
    raea = probability(sum(w * x for w, x in zip(weights, ps)))
    median = probability(np.median(np.stack(ps), axis=0))
    consensus = probability((raea + epsilon) * np.power(median + epsilon, 0.8))
    return {"status": "OK", "consensus": consensus, "weights": weights.tolist(), "reliability": reliability.tolist()}


# ---------------------------------------------------------------- local ROI proposal

@dataclass(frozen=True)
class RegionOfInterest:
    box: tuple[float, float, float, float]
    relevance: float


def gaussian_smooth(relevance_map: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    radius = max(1, math.ceil(2 * sigma))
    coordinates = torch.arange(-radius, radius + 1, device=relevance_map.device, dtype=relevance_map.dtype)
    kernel = torch.exp(-(coordinates ** 2) / (2 * sigma ** 2))
    kernel = torch.outer(kernel, kernel)
    kernel /= kernel.sum()
    smoothed = F.conv2d(relevance_map.reshape(1, 1, *relevance_map.shape), kernel.reshape(1, 1, *kernel.shape),
                        padding=radius).reshape_as(relevance_map)
    return smoothed / smoothed.sum().clamp_min(torch.finfo(smoothed.dtype).tiny)


def _iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-12)


def propose_image_regions(relevance_map: torch.Tensor, image_size: tuple[int, int], *, anchors: int,
                          min_anchor_distance: float, min_side: int, max_side: int | None,
                          expansion_threshold: float, nms_threshold: float) -> list[RegionOfInterest]:
    """FOCUS-style anchors on the smoothed map, grown while denser than ``expansion_threshold`` x mean, then NMS."""
    relevance_map = gaussian_smooth(relevance_map)
    height, width = relevance_map.shape
    image_width, image_height = image_size
    max_side = max_side or max(height, width)
    chosen: list[tuple[int, int]] = []
    for flat_index in torch.argsort(relevance_map.flatten(), descending=True).tolist():
        row, column = divmod(flat_index, width)
        if all(math.hypot(row - r, column - c) >= min_anchor_distance for r, c in chosen):
            chosen.append((row, column))
        if len(chosen) >= anchors:
            break
    proposals = []
    threshold = float(relevance_map.mean()) * expansion_threshold
    for row, column in chosen:
        side, best = min_side, (column, row, column + 1, row + 1)
        while side <= max_side:
            half = side / 2
            left, top = max(0, math.floor(column + 0.5 - half)), max(0, math.floor(row + 0.5 - half))
            right, bottom = min(width, math.ceil(column + 0.5 + half)), min(height, math.ceil(row + 0.5 + half))
            if float(relevance_map[top:bottom, left:right].mean()) < threshold and side > min_side:
                break
            best = (left, top, right, bottom)
            side += 2
        left, top, right, bottom = best
        box = (image_width * left / width, image_height * top / height,
               image_width * right / width, image_height * bottom / height)
        proposals.append(RegionOfInterest(box=box, relevance=float(relevance_map[row, column])))
    retained: list[RegionOfInterest] = []
    for proposal in sorted(proposals, key=lambda item: item.relevance, reverse=True):
        if all(_iou(proposal.box, item.box) <= nms_threshold for item in retained):
            retained.append(proposal)
    return retained


def crop_box(heatmap: np.ndarray, size: tuple[int, int], policy: dict[str, Any]) -> dict[str, Any]:
    """Context box + tight box (original pixels) from the top local ROI, or a reason for no crop."""
    width, height = map(int, size)
    if width < 2 or height < 2:
        return {"box": None, "tight_box": None, "reason": "INVALID_SMALL_IMAGE"}
    p = probability(heatmap)
    gh, gw = p.shape
    if concentration(p) < policy["min_joint_concentration"]:
        return {"box": None, "tight_box": None, "reason": "DIFFUSE_HEATMAP"}
    roi = policy["roi"]
    t = torch.as_tensor(np.array(p, copy=True), dtype=torch.float64)
    regions = propose_image_regions(t, (width, height), **roi)
    if not regions:
        return {"box": None, "tight_box": None, "reason": "NO_LOCAL_ROI"}
    rx1, ry1, rx2, ry2 = regions[0].box
    gx0, gx1 = int(round(rx1 / width * gw)), int(round(rx2 / width * gw))
    gy0, gy1 = int(round(ry1 / height * gh)), int(round(ry2 / height * gh))
    density = float(gaussian_smooth(t).numpy()[gy0:gy1, gx0:gx1].mean() * p.size)
    if density + 1e-12 < roi["expansion_threshold"]:
        return {"box": None, "tight_box": None, "reason": "WEAK_TOP_LOCAL_ROI_NO_FORCED_CROP"}
    tight = [max(0, math.floor(rx1)), max(0, math.floor(ry1)), min(width, math.ceil(rx2)), min(height, math.ceil(ry2))]
    cx, cy = (rx1 + rx2) / 2, (ry1 + ry2) / 2
    w = min(width, max((rx2 - rx1) * (1 + 2 * policy["padding_ratio"]), width * policy["min_width_ratio"]))
    h = min(height, max((ry2 - ry1) * (1 + 2 * policy["padding_ratio"]), height * policy["min_height_ratio"]))
    x1, y1 = min(max(0.0, cx - w / 2), width - w), min(max(0.0, cy - h / 2), height - h)
    box = [max(0, math.floor(x1)), max(0, math.floor(y1)), min(width, math.ceil(x1 + w)), min(height, math.ceil(y1 + h))]
    if min(box[2] - box[0], box[3] - box[1], tight[2] - tight[0], tight[3] - tight[1]) < 2:
        return {"box": None, "tight_box": None, "reason": "INVALID_SMALL_BOX"}
    if (box[2] - box[0]) * (box[3] - box[1]) / (width * height) >= policy["fallback_area_ratio"]:
        return {"box": None, "tight_box": None, "reason": "DEGENERATE_FULL_IMAGE"}
    return {"box": box, "tight_box": tight, "reason": None}


# ---------------------------------------------------------------- localizer forward

def escape(value: Any, tokens: list[str]) -> str:
    text = str(value)
    for token in sorted(set(tokens), key=len, reverse=True):
        if token:
            text = text.replace(token, token.replace("<", "&lt;", 1) if "<" in token else "[escaped-control-token]")
    return text


def localization_prompt(row: dict[str, Any], attribute: str, tokenizer: Any) -> str:
    """Full, untruncated serialized row as entity context; the attribute alone is marked."""
    special = tokenizer.all_special_tokens
    cells = [f"{escape(c['column_name'], special)}={escape(c['text'], special)}" for c in row["cells"]]
    anchor = " | ".join(tokenizer.decode(tokenizer.encode(x, add_special_tokens=False), skip_special_tokens=False)
                        for x in cells)
    return ("Task: localize image evidence for the requested attribute of this query-row entity.\n"
            "ENTITY/CONTEXT: " + ROW_OPEN + anchor + ROW_CLOSE + "\nATTRIBUTE: " + ATTR_OPEN + escape(attribute, special)
            + ATTR_CLOSE + "\nTASK: locate explicit support for this same entity/event/version/time and requested "
            "relation. No answer generation.\n")


def attribute_consensus(layer_values: Sequence[torch.Tensor], grid: tuple[int, int], epsilon: float) -> np.ndarray | None:
    """RAEA maps from compact CPU V rows: image tokens first, then attribute tokens.

    Image vectors are normalized once per layer. Keep the original per-token matvec and
    layer-mean order so the crop statistics retain their floating-point behavior.
    """
    image_count = grid[0] * grid[1]
    normalized_images = [F.normalize(values[:image_count], dim=-1) for values in layer_values]
    token_maps = []
    for token in range(image_count, len(layer_values[0])):
        per_layer = [(images @ F.normalize(values[token], dim=0)).reshape(grid).numpy()
                     for images, values in zip(normalized_images, layer_values)]
        token_map = minmax_map(np.stack(per_layer).mean(0), epsilon)
        if token_map is not None:
            token_maps.append(token_map)
    return aggregate_raea(token_maps, epsilon)["consensus"]


class ImageLocalizer:
    """Shares the recovery generator's model; one extra forward per (row, attribute, image).

    Crops are cached per query (the singleton phase reuses the packet phase's crops); the caller
    clears ``cache`` between queries.
    """

    def __init__(self, processor: Any, model: Any, policy: dict[str, Any]) -> None:
        self.processor, self.model, self.policy = processor, model, policy
        tokenizer = processor.tokenizer
        self.attr_open = tokenizer.convert_tokens_to_ids(ATTR_OPEN)
        self.attr_close = tokenizer.convert_tokens_to_ids(ATTR_CLOSE)
        self.layers = model.model.language_model.layers
        self.cache: dict[tuple, dict[str, Any]] = {}
        self.stats = {"forwards": 0, "crops": 0, "no_crop": 0}

    def crop(self, row: dict[str, Any], attribute: str, path: str) -> dict[str, Any]:
        """``{box, tight_box, reason, context, tight}``; ``context``/``tight`` are PIL crops or None."""
        key = (json.dumps(row, sort_keys=True, ensure_ascii=False), attribute, path)
        if key in self.cache:
            return self.cache[key]
        original = decode_original(path)
        image = bounded_rgb(original, self.policy["localizer_max_pixels"])
        consensus = self._consensus(image, localization_prompt(row, attribute, self.processor.tokenizer))
        result = {"box": None, "tight_box": None, "reason": "NO_ATTRIBUTE_TOKENS"}
        if consensus is not None:
            result = crop_box(consensus, original.size, self.policy)
        result["context"] = result["tight"] = None
        if result["box"] is not None:
            result["context"] = bounded_rgb(original.crop(tuple(result["box"])), self.policy["crop_max_pixels"])
            result["tight"] = bounded_rgb(original.crop(tuple(result["tight_box"])), self.policy["crop_max_pixels"])
        self.stats["crops" if result["box"] is not None else "no_crop"] += 1
        self.cache[key] = result
        return result

    def _consensus(self, image: Image.Image, prompt: str) -> np.ndarray | None:
        model = self.model
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}],
            tokenize=True, add_generation_prompt=False, enable_thinking=False, return_dict=True, return_tensors="pt")
        ids = inputs["input_ids"][0].tolist()
        start, end = ids.index(self.attr_open), ids.index(self.attr_close)
        attribute_tokens = list(range(start + 1, end))
        image_tokens = [i for i, t in enumerate(ids) if t == int(model.config.image_token_id)]
        _, h, w = map(int, inputs["image_grid_thw"][0].tolist())
        merge = int(model.config.vision_config.spatial_merge_size)
        gh, gw = h // merge, w // merge
        if gh * gw != len(image_tokens):
            raise RuntimeError("localizer image grid does not match the image token count")
        # Text outside the attribute never participates in the maps. Keep just these rows
        # on device during the forward, then copy them after all hooks have completed.
        selected = torch.tensor([*image_tokens, *attribute_tokens], device="cuda:0")
        captured: dict[int, torch.Tensor] = {}
        handles = []
        for layer in self.policy["layers"]:
            def hook(_module, _args, output, index=layer):
                captured[index] = output.detach()[0].index_select(0, selected).float()
            handles.append(self.layers[layer].self_attn.v_proj.register_forward_hook(hook))
        previous_rope = getattr(model.model, "rope_deltas", None)
        try:
            with torch.inference_mode():
                model.model(**{k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in inputs.items()},
                            use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
            model.model.rope_deltas = previous_rope
        self.stats["forwards"] += 1
        return attribute_consensus([captured[layer].cpu() for layer in self.policy["layers"]],
                                   (gh, gw), self.policy["flat_epsilon"])
