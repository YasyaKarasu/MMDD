"""Frozen Qwen3.5 v-projection localizer for MMDD-R4c Adapt-v1."""

from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Iterator, Sequence

import torch
from PIL import Image, ImageOps
from torch.nn import functional as F

from .data import ATTRIBUTE_CLOSE, ATTRIBUTE_OPEN, ROW_ANCHOR_CLOSE, ROW_ANCHOR_OPEN, serialize_localization_prompt
from .qwen import QwenStage2Backend

EPS = 1e-12
RAEA_A_COEF = 1.2
RAEA_P_COEF = 0.6
CONSENSUS_POWER = 0.8
MASS_FRACTION = 0.60
PADDING_RATIO = 0.10
MIN_WIDTH_RATIO = 0.15
MIN_HEIGHT_RATIO = 0.15
FALLBACK_AREA_RATIO = 0.85
MIN_JOINT_CONCENTRATION = 0.05


def norm_prob(x: torch.Tensor) -> torch.Tensor:
    x = x.float().clamp_min(0)
    total = x.sum()
    if not torch.isfinite(x).all() or not torch.isfinite(total) or float(total) <= EPS:
        raise ValueError("LOCALIZER_INVALID_MAP")
    return x / total


def concentration(p: torch.Tensor) -> torch.Tensor:
    p = norm_prob(p)
    count = p.numel()
    if count == 1:
        return torch.tensor(1.0)
    entropy = -(p * (p + EPS).log()).sum()
    return (1 - entropy / math.log(count)).clamp(0, 1)


def spatial_centroid(p: torch.Tensor) -> torch.Tensor:
    p = norm_prob(p)
    height, width = p.shape
    ys = (torch.arange(height, dtype=torch.float32) + 0.5) / height
    xs = (torch.arange(width, dtype=torch.float32) + 0.5) / width
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([(p * xx).sum(), (p * yy).sum()])


def agreement_scores(maps: torch.Tensor) -> torch.Tensor:
    centers = torch.stack([spatial_centroid(item) for item in maps])
    median = centers.median(dim=0).values
    distance = torch.linalg.vector_norm(centers - median, dim=1)
    return (1 - distance / math.sqrt(2)).clamp(0, 1)


def prominence(p: torch.Tensor) -> torch.Tensor:
    p = norm_prob(p)
    peak = p.max()
    background = p.flatten().median()
    return ((peak - background) / (peak + EPS)).clamp(0, 1)


@dataclass(frozen=True)
class RAEAResult:
    heatmap: torch.Tensor
    weights: torch.Tensor
    c: torch.Tensor
    a: torch.Tensor
    p: torch.Tensor
    z: torch.Tensor


def raea_attribute_map(maps: torch.Tensor) -> RAEAResult:
    probabilities = torch.stack([norm_prob(item) for item in maps])
    c = torch.stack([concentration(item) for item in probabilities])
    a = agreement_scores(probabilities)
    p = torch.stack([prominence(item) for item in probabilities])
    z = c + RAEA_A_COEF * a + RAEA_P_COEF * p
    weights = torch.softmax(z, dim=0)
    heatmap = norm_prob((weights[:, None, None] * probabilities).sum(dim=0))
    return RAEAResult(heatmap, weights, c, a, p, z)


def consensus_attribute_map(maps: torch.Tensor, raea: torch.Tensor) -> torch.Tensor:
    probabilities = torch.stack([norm_prob(item) for item in maps])
    median = norm_prob(probabilities.median(dim=0).values)
    return norm_prob((norm_prob(raea) + EPS) * (median + EPS).pow(CONSENSUS_POWER))


def row_map(layer_maps: torch.Tensor) -> torch.Tensor:
    return norm_prob(torch.stack([norm_prob(item) for item in layer_maps]).mean(dim=0))


def joint_map(row: torch.Tensor, attribute: torch.Tensor) -> torch.Tensor:
    return norm_prob((norm_prob(row) + EPS) * (norm_prob(attribute) + EPS))


def tight_crop_box(
    heatmap: torch.Tensor, image_width: int, image_height: int
) -> tuple[tuple[int, int, int, int] | None, str | None]:
    probability = norm_prob(heatmap)
    if float(concentration(probability)) < MIN_JOINT_CONCENTRATION:
        return None, "DIFFUSE_HEATMAP"
    height, width = probability.shape
    cells = sorted(
        ((float(probability[row, column]), row, column)
         for row in range(height) for column in range(width)),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    mass = 0.0
    chosen: list[tuple[int, int]] = []
    for value, row, column in cells:
        chosen.append((row, column))
        mass += value
        if mass + 1e-15 >= MASS_FRACTION:
            break
    row0 = min(row for row, _ in chosen)
    row1 = max(row for row, _ in chosen) + 1
    column0 = min(column for _, column in chosen)
    column1 = max(column for _, column in chosen) + 1
    x1, x2 = image_width * column0 / width, image_width * column1 / width
    y1, y2 = image_height * row0 / height, image_height * row1 / height
    box_width, box_height = x2 - x1, y2 - y1
    x1, x2 = x1 - PADDING_RATIO * box_width, x2 + PADDING_RATIO * box_width
    y1, y2 = y1 - PADDING_RATIO * box_height, y2 + PADDING_RATIO * box_height
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    needed_width = max(x2 - x1, MIN_WIDTH_RATIO * image_width)
    needed_height = max(y2 - y1, MIN_HEIGHT_RATIO * image_height)
    x1, x2 = center_x - needed_width / 2, center_x + needed_width / 2
    y1, y2 = center_y - needed_height / 2, center_y + needed_height / 2
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(image_width), x2), min(float(image_height), y2)
    if x2 - x1 < MIN_WIDTH_RATIO * image_width:
        if x1 == 0:
            x2 = min(float(image_width), MIN_WIDTH_RATIO * image_width)
        elif x2 == image_width:
            x1 = max(0.0, image_width - MIN_WIDTH_RATIO * image_width)
    if y2 - y1 < MIN_HEIGHT_RATIO * image_height:
        if y1 == 0:
            y2 = min(float(image_height), MIN_HEIGHT_RATIO * image_height)
        elif y2 == image_height:
            y1 = max(0.0, image_height - MIN_HEIGHT_RATIO * image_height)
    left, top = max(0, math.floor(x1)), max(0, math.floor(y1))
    right, bottom = min(image_width, math.ceil(x2)), min(image_height, math.ceil(y2))
    if right - left < 2 or bottom - top < 2:
        return None, "INVALID_SMALL_BOX"
    area_ratio = ((right - left) * (bottom - top)) / (image_width * image_height)
    if area_ratio >= FALLBACK_AREA_RATIO:
        return None, "DEGENERATE_FULL_IMAGE"
    return (left, top, right, bottom), None


def resolve_layer_indices(layers: Sequence[Any]) -> tuple[tuple[int, int, int], dict[str, Any]]:
    valid = [
        index for index, layer in enumerate(layers)
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "v_proj")
    ]
    count = len(layers)
    anchors = [0.50 * (count - 1), 0.75 * (count - 1), 0.90 * (count - 1)]
    selected: list[int] = []
    for anchor in anchors:
        choices = [index for index in valid if index not in selected]
        if not choices:
            raise ValueError("BLOCKED_INSUFFICIENT_FULL_ATTENTION_LAYERS")
        selected.append(min(choices, key=lambda index: (abs(index - anchor), index)))
    if len(selected) != 3:
        raise ValueError("BLOCKED_INSUFFICIENT_FULL_ATTENTION_LAYERS")
    return tuple(selected), {
        "num_language_layers": count,
        "valid_vproj_layers": valid,
        "anchor_depths": anchors,
        "resolved_layers": selected,
    }


@contextmanager
def capture_selected_vproj(
    model: Any, absolute_layer_indices: Sequence[int]
) -> Iterator[dict[int, torch.Tensor | None]]:
    layers = model.model.language_model.layers
    captured: dict[int, torch.Tensor | None] = {index: None for index in absolute_layer_indices}
    handles = []
    for index in absolute_layer_indices:
        projection = layers[index].self_attn.v_proj

        def hook(_module: Any, _inputs: Any, output: torch.Tensor, layer: int = index) -> None:
            captured[layer] = output.detach()[0].float().cpu()

        handles.append(projection.register_forward_hook(hook))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


@dataclass
class LocalizerForward:
    values: dict[int, torch.Tensor]
    input_ids: torch.Tensor
    row_indices: torch.Tensor
    attribute_indices: torch.Tensor
    image_indices: torch.Tensor
    grid_thw: tuple[int, int, int]
    merge_size: int
    image_size: tuple[int, int]
    image_file_sha256: str
    elapsed_seconds: float
    peak_gpu_memory_bytes: int


@dataclass
class LocalizerMaps:
    row: torch.Tensor
    attribute_maps: torch.Tensor
    raea: torch.Tensor
    consensus: torch.Tensor
    joint_raea: torch.Tensor
    joint_consensus: torch.Tensor
    metadata: dict[str, Any]


def _tensor_sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class R4CFastImageLocalizer(QwenStage2Backend):
    formula_version = "r4c-adapt-v1"

    def resolve_layers(self) -> tuple[int, int, int]:
        selected, audit = resolve_layer_indices(self.model.model.language_model.layers)
        self.localizer_config = audit
        return selected

    @torch.inference_mode()
    def localizer_forward(
        self, row: dict[str, str], attribute_name: str, evidence: dict[str, Any]
    ) -> LocalizerForward:
        path = Path(str(evidence.get("local_path", "")))
        if not path.is_file():
            raise FileNotFoundError(f"BLOCKED_IMAGE_DECODE:{path}")
        image_file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as source:
            source.seek(0)
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
        prompt = serialize_localization_prompt(row, attribute_name)
        inputs = self._inputs(
            [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
            generation_prompt=False,
        )
        selected = self.resolve_layers()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        started = monotonic()
        with capture_selected_vproj(self.model, selected) as captured:
            self.model.model(**inputs, use_cache=False, return_dict=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = monotonic() - started
        if any(captured[index] is None for index in selected):
            raise RuntimeError("LOCALIZER_HOOK_ERROR")
        input_ids = inputs["input_ids"][0].detach().cpu()
        row_indices = self._marker_range(
            input_ids, self.marker_ids[ROW_ANCHOR_OPEN], self.marker_ids[ROW_ANCHOR_CLOSE]
        )
        attribute_indices = self._marker_range(
            input_ids, self.marker_ids[ATTRIBUTE_OPEN], self.marker_ids[ATTRIBUTE_CLOSE]
        )
        image_indices = (input_ids == self.image_token_id).nonzero().flatten()
        if not len(row_indices) or not len(attribute_indices) or not len(image_indices):
            raise ValueError("LOCALIZER_MARKER_ERROR")
        grid = inputs["image_grid_thw"][0].detach().cpu().long()
        temporal, raw_height, raw_width = map(int, grid.tolist())
        merge = int(self.model.config.vision_config.spatial_merge_size)
        if raw_height % merge or raw_width % merge:
            raise ValueError("LOCALIZER_GRID_ERROR")
        height, width = raw_height // merge, raw_width // merge
        if len(image_indices) != temporal * height * width:
            raise ValueError(
                f"LOCALIZER_GRID_ERROR:image_tokens={len(image_indices)} grid="
                f"{temporal}x{height}x{width}"
            )
        return LocalizerForward(
            values={index: captured[index] for index in selected if captured[index] is not None},
            input_ids=input_ids,
            row_indices=row_indices,
            attribute_indices=attribute_indices,
            image_indices=image_indices,
            grid_thw=(temporal, raw_height, raw_width),
            merge_size=merge,
            image_size=image.size,
            image_file_sha256=image_file_sha,
            elapsed_seconds=elapsed,
            peak_gpu_memory_bytes=(
                int(torch.cuda.max_memory_allocated(self.device)) if self.device.type == "cuda" else 0
            ),
        )

    def build_maps(self, forward: LocalizerForward) -> LocalizerMaps:
        temporal, raw_height, raw_width = forward.grid_thw
        height, width = raw_height // forward.merge_size, raw_width // forward.merge_size
        row_layers = []
        attribute_maps = []
        map_units = []
        token_strings = [
            self.processor.tokenizer.decode([int(forward.input_ids[index])], skip_special_tokens=False)
            for index in forward.attribute_indices
        ]
        for layer_index, values in forward.values.items():
            image_values = F.normalize(values.index_select(0, forward.image_indices).float(), dim=-1)
            row_query = F.normalize(
                values.index_select(0, forward.row_indices).float().mean(dim=0), dim=0
            )
            row_probability = torch.softmax(image_values @ row_query, dim=0)
            row_probability = norm_prob(
                row_probability.reshape(temporal, height, width).mean(dim=0)
            )
            row_layers.append(row_probability)
            for token_position, token_index in enumerate(forward.attribute_indices):
                query = F.normalize(values[int(token_index)].float(), dim=0)
                probability = torch.softmax(image_values @ query, dim=0)
                probability = norm_prob(
                    probability.reshape(temporal, height, width).mean(dim=0)
                )
                attribute_maps.append(probability)
                map_units.append({
                    "layer": layer_index,
                    "token_index": int(token_index),
                    "token_id": int(forward.input_ids[token_index]),
                    "token_string": token_strings[token_position],
                })
        attribute_tensor = torch.stack(attribute_maps)
        row_heatmap = row_map(torch.stack(row_layers))
        raea_result = raea_attribute_map(attribute_tensor)
        consensus = consensus_attribute_map(attribute_tensor, raea_result.heatmap)
        joint_raea = joint_map(row_heatmap, raea_result.heatmap)
        joint_consensus = joint_map(row_heatmap, consensus)
        for index, unit in enumerate(map_units):
            unit.update({
                "c": float(raea_result.c[index]),
                "a": float(raea_result.a[index]),
                "p": float(raea_result.p[index]),
                "z": float(raea_result.z[index]),
                "w": float(raea_result.weights[index]),
            })
        metadata = {
            "resolved_layers": list(forward.values),
            "image_grid_thw": list(forward.grid_thw),
            "merge_size": forward.merge_size,
            "row_token_count": int(len(forward.row_indices)),
            "attribute_token_ids": [int(forward.input_ids[index]) for index in forward.attribute_indices],
            "attribute_token_strings": token_strings,
            "per_map": map_units,
            "H_row_sha256": _tensor_sha(row_heatmap),
            "H_attr_raea_sha256": _tensor_sha(raea_result.heatmap),
            "H_attr_cons_sha256": _tensor_sha(consensus),
            "H_joint_raea_sha256": _tensor_sha(joint_raea),
            "H_joint_consensus_sha256": _tensor_sha(joint_consensus),
            "image_size": list(forward.image_size),
            "image_file_sha256": forward.image_file_sha256,
            "elapsed_localizer_seconds": forward.elapsed_seconds,
            "peak_localizer_gpu_memory_bytes": forward.peak_gpu_memory_bytes,
        }
        return LocalizerMaps(
            row_heatmap, attribute_tensor, raea_result.heatmap, consensus,
            joint_raea, joint_consensus, metadata,
        )

    def crop_for_arm(
        self, maps: LocalizerMaps, arm: str, image_size: tuple[int, int]
    ) -> dict[str, Any]:
        heatmap = maps.joint_raea if arm == "V2_RAEA_DUAL" else maps.joint_consensus
        box, reason = tight_crop_box(heatmap, *image_size)
        area = None
        if box is not None:
            area = ((box[2] - box[0]) * (box[3] - box[1])) / (image_size[0] * image_size[1])
        return {
            "arm": arm,
            "pixel_box": list(box) if box is not None else None,
            "area_ratio": area,
            "crop_fallback": box is None,
            "fallback_reason": reason,
            "joint_concentration": float(concentration(heatmap)),
        }
