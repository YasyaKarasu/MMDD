"""Local Qwen3.5 backend for the Stage-2 research pipeline."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from PIL import Image
from torch.nn import functional as F

from mmdd_dataset.utils import clean_text

from .data import (
    ATTRIBUTE_CLOSE,
    ATTRIBUTE_OPEN,
    CANDIDATE_CLOSE,
    CANDIDATE_OPEN,
    ENTITY_CLOSE,
    ENTITY_OPEN,
    EVIDENCE_CLOSE,
    EVIDENCE_OPEN,
    serialize_localization_prompt,
    serialize_table,
)
from .pipeline import LocalizedEvidence
from .verifier import best_text_span, focus_relevance, propose_image_regions


class QwenStage2Backend:
    """Expose Qwen3.5 reader states, FOCUS features, generation, and embeddings."""

    def __init__(
        self,
        model_dir: Path,
        *,
        device: str = "auto",
        dtype: str = "bf16",
        focus_start_layer: int = 14,
        max_text_evidence_tokens: int = 1024,
        text_overlap_tokens: int = 128,
        max_span_tokens: int = 192,
        roi_candidates: int = 4,
        max_new_tokens: int = 64,
    ) -> None:
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        if max_text_evidence_tokens <= 0 or not 0 <= text_overlap_tokens < max_text_evidence_tokens:
            raise ValueError("Text evidence width must be positive and exceed its overlap")
        if min(max_span_tokens, roi_candidates, max_new_tokens) <= 0:
            raise ValueError("Span, ROI, and generation limits must be positive")
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_dir,
            local_files_only=True,
            dtype=dtype_map[dtype],
            attn_implementation="sdpa",
        ).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.hidden_dim = int(self.model.config.text_config.hidden_size)
        if not 0 <= focus_start_layer < len(self.model.model.language_model.layers):
            raise ValueError("focus_start_layer is outside the Qwen language model")
        self.focus_start_layer = focus_start_layer
        self.max_text_evidence_tokens = max_text_evidence_tokens
        self.text_overlap_tokens = text_overlap_tokens
        self.max_span_tokens = max_span_tokens
        self.roi_candidates = roi_candidates
        self.max_new_tokens = max_new_tokens
        tokenizer = self.processor.tokenizer
        self.marker_ids = {
            marker: tokenizer.convert_tokens_to_ids(marker)
            for marker in (
                CANDIDATE_OPEN,
                CANDIDATE_CLOSE,
                ATTRIBUTE_OPEN,
                ATTRIBUTE_CLOSE,
                EVIDENCE_OPEN,
                EVIDENCE_CLOSE,
            )
        }
        self.image_token_id = int(self.model.config.image_token_id)

    def _inputs(self, content: list[dict[str, Any]], *, generation_prompt: bool) -> dict[str, torch.Tensor]:
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=generation_prompt,
            return_dict=True,
            return_tensors="pt",
        )
        return {name: value.to(self.device) for name, value in inputs.items()}

    @staticmethod
    def _marker_range(input_ids: torch.Tensor, open_id: int, close_id: int) -> torch.Tensor:
        opens = (input_ids == open_id).nonzero().flatten()
        closes = (input_ids == close_id).nonzero().flatten()
        if opens.numel() != 1 or closes.numel() != 1 or int(opens[0]) >= int(closes[0]) - 1:
            raise ValueError("Expected exactly one non-empty marked token range")
        return torch.arange(int(opens[0]) + 1, int(closes[0]), dtype=torch.long)

    @contextmanager
    def _capture_values(self) -> Iterator[list[torch.Tensor | None]]:
        # Qwen3.5 interleaves linear-attention and full-attention layers. FOCUS
        # uses v_proj features from the later full-attention layers only.
        layers = self.model.model.language_model.layers
        value_projections = [
            layer.self_attn.v_proj
            for layer in layers[self.focus_start_layer :]
            if hasattr(layer, "self_attn")
        ]
        if not value_projections:
            raise ValueError("No Qwen3.5 full-attention layers remain after focus_start_layer")
        captured: list[torch.Tensor | None] = [None] * len(value_projections)
        handles = []
        for output_index, projection in enumerate(value_projections):
            def hook(_module: Any, _inputs: Any, output: torch.Tensor, index: int = output_index) -> None:
                captured[index] = output.detach()[0].float().cpu()

            handles.append(projection.register_forward_hook(hook))
        try:
            yield captured
        finally:
            for handle in handles:
                handle.remove()

    @torch.inference_mode()
    def _value_forward(
        self, content: list[dict[str, Any]]
    ) -> tuple[list[torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        inputs = self._inputs(content, generation_prompt=False)
        with self._capture_values() as captured:
            self.model.model(**inputs, use_cache=False, return_dict=True)
        if any(layer is None for layer in captured):
            raise RuntimeError("Qwen value-feature hooks did not run")
        return [layer for layer in captured if layer is not None], inputs["input_ids"][0].cpu(), inputs

    @torch.inference_mode()
    def reader_states(
        self,
        query: dict[str, Any],
        target: dict[str, Any],
        evidence: Sequence[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read query, all selected evidence, and target in one context."""

        if not evidence:
            raise ValueError("Candidate-column reading requires at least one evidence object")
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Select the target column that supplies the missing bridge attribute for the example rows.\n"
                    f"Query table:\n{serialize_table(query)}\n\nRetrieved evidence:\n"
                ),
            }
        ]
        text_limit = max(1, 12000 // len(evidence))
        for index, item in enumerate(evidence, 1):
            label = f"\nEvidence {index} ({item['asset_id']}):"
            if item.get("asset_type") == "image":
                content.extend(
                    [{"type": "text", "text": label}, {"type": "image", "image": self._image_path(item)}]
                )
            else:
                text = clean_text(item.get("content"))[:text_limit]
                content.append({"type": "text", "text": f"{label}\n{text}"})
        content.append(
            {
                "type": "text",
                "text": f"\nCandidate target table:\n{serialize_table(target, mark_candidates=True)}",
            }
        )
        inputs = self._inputs(content, generation_prompt=False)
        outputs = self.model.model(**inputs, use_cache=False, return_dict=True)
        input_ids = inputs["input_ids"][0]
        open_positions = (input_ids == self.marker_ids[CANDIDATE_OPEN]).nonzero().flatten()
        close_positions = (input_ids == self.marker_ids[CANDIDATE_CLOSE]).nonzero().flatten()
        if open_positions.numel() != len(target["columns"]) or close_positions.numel() != len(target["columns"]):
            raise ValueError("Candidate marker count changed during Qwen preprocessing")
        hidden = outputs.last_hidden_state[0]
        return hidden[open_positions].float().cpu(), hidden[close_positions].float().cpu()

    @staticmethod
    def _image_path(evidence: dict[str, Any]) -> str:
        path = Path(str(evidence.get("local_path", "")))
        if not path.is_file():
            raise FileNotFoundError(f"Missing evidence image: {path}")
        return str(path)

    def _text_chunks(self, text: str) -> Iterator[str]:
        token_ids = self.processor.tokenizer.encode(text, add_special_tokens=False)
        width = self.max_text_evidence_tokens
        step = max(1, width - self.text_overlap_tokens)
        for start in range(0, len(token_ids), step):
            chunk = token_ids[start : start + width]
            if chunk:
                yield self.processor.tokenizer.decode(chunk, skip_special_tokens=True)
            if start + width >= len(token_ids):
                break

    def _localize_text(
        self,
        row: dict[str, str],
        entity_column: str,
        attribute_name: str,
        evidence: dict[str, Any],
    ) -> LocalizedEvidence:
        best: LocalizedEvidence | None = None
        prompt = serialize_localization_prompt(row, entity_column, attribute_name)
        for chunk in self._text_chunks(clean_text(evidence.get("content"))):
            marked = f"{prompt}\nEvidence: {EVIDENCE_OPEN}{chunk}{EVIDENCE_CLOSE}"
            layers, input_ids, _ = self._value_forward([{"type": "text", "text": marked}])
            entity_indices = self._marker_range(
                input_ids, self.marker_ids[ENTITY_OPEN], self.marker_ids[ENTITY_CLOSE]
            )
            attribute_indices = self._marker_range(
                input_ids, self.marker_ids[ATTRIBUTE_OPEN], self.marker_ids[ATTRIBUTE_CLOSE]
            )
            evidence_indices = self._marker_range(
                input_ids, self.marker_ids[EVIDENCE_OPEN], self.marker_ids[EVIDENCE_CLOSE]
            )
            relevance = focus_relevance(layers, entity_indices, attribute_indices, evidence_indices)
            start, end = best_text_span(relevance, self.max_span_tokens)
            selected_ids = input_ids[evidence_indices[start:end]].tolist()
            span = clean_text(self.processor.tokenizer.decode(selected_ids, skip_special_tokens=True))
            candidate = LocalizedEvidence(
                evidence_id=str(evidence["asset_id"]),
                evidence_type="text",
                relevance=float(relevance[start:end].sum()),
                text=span,
            )
            if best is None or candidate.relevance > best.relevance:
                best = candidate
        return best or LocalizedEvidence(str(evidence["asset_id"]), "text", 0.0, text="")

    def _localize_image(
        self,
        row: dict[str, str],
        entity_column: str,
        attribute_name: str,
        evidence: dict[str, Any],
    ) -> LocalizedEvidence:
        image_path = self._image_path(evidence)
        prompt = serialize_localization_prompt(row, entity_column, attribute_name)
        layers, input_ids, inputs = self._value_forward(
            [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]
        )
        entity_indices = self._marker_range(input_ids, self.marker_ids[ENTITY_OPEN], self.marker_ids[ENTITY_CLOSE])
        attribute_indices = self._marker_range(
            input_ids, self.marker_ids[ATTRIBUTE_OPEN], self.marker_ids[ATTRIBUTE_CLOSE]
        )
        image_indices = (input_ids == self.image_token_id).nonzero().flatten()
        relevance = focus_relevance(layers, entity_indices, attribute_indices, image_indices)
        grid = inputs["image_grid_thw"][0].detach().cpu().long()
        merge = int(self.model.config.vision_config.spatial_merge_size)
        temporal, height, width = int(grid[0]), int(grid[1] // merge), int(grid[2] // merge)
        if relevance.numel() != temporal * height * width:
            raise ValueError("Qwen image token count does not match image_grid_thw")
        relevance_map = relevance.reshape(temporal, height, width).mean(dim=0)
        image = Image.open(image_path).convert("RGB")
        regions = propose_image_regions(relevance_map, image.size)[: self.roi_candidates]
        ranked = []
        for region in regions:
            crop = image.crop(tuple(round(value) for value in region.box))
            confidence = self._presence_confidence(crop, row[entity_column], attribute_name)
            ranked.append((confidence, region.relevance, region, crop))
        confidence, _, region, crop = max(ranked, key=lambda item: item[:2])
        return LocalizedEvidence(
            evidence_id=str(evidence["asset_id"]),
            evidence_type="image",
            relevance=confidence,
            image=crop,
            box=region.box,
        )

    def localize_evidence(
        self,
        row: dict[str, str],
        *,
        entity_column: str,
        attribute_name: str,
        evidence: dict[str, Any],
    ) -> LocalizedEvidence:
        if evidence.get("asset_type") == "image":
            return self._localize_image(row, entity_column, attribute_name, evidence)
        return self._localize_text(row, entity_column, attribute_name, evidence)

    @torch.inference_mode()
    def _presence_confidence(self, image: Image.Image, entity: str, attribute: str) -> float:
        content = [
            {"type": "image", "image": image},
            {
                "type": "text",
                "text": f"Does this crop contain visual evidence about {entity}'s {attribute}? Answer yes or no.",
            },
        ]
        inputs = self._inputs(content, generation_prompt=True)
        logits = self.model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True).logits[0, -1].float()
        yes_id = self.processor.tokenizer.encode(" yes", add_special_tokens=False)[0]
        no_id = self.processor.tokenizer.encode(" no", add_special_tokens=False)[0]
        return float(torch.softmax(logits[[yes_id, no_id]], dim=0)[0])

    @torch.inference_mode()
    def _generate(self, content: list[dict[str, Any]]) -> str:
        inputs = self._inputs(content, generation_prompt=True)
        generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens)
        answer_ids = generated[0, inputs["input_ids"].shape[1] :]
        return self.processor.tokenizer.decode(answer_ids, skip_special_tokens=True)

    def generate_value(
        self,
        row: dict[str, str],
        *,
        attribute_name: str,
        evidence: LocalizedEvidence,
    ) -> str:
        prompt = (
            "Extract the requested bridge value from the localized evidence. Do not infer a value not stated "
            "by the evidence. Return JSON only as {\"value\": \"...\"}; use an empty value when unsupported.\n"
            f"Example row: {json.dumps(row, ensure_ascii=False)}\nAttribute: {attribute_name}"
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if evidence.image is not None:
            content.insert(0, {"type": "image", "image": evidence.image})
        else:
            content.append({"type": "text", "text": f"Localized evidence: {evidence.text or ''}"})
        answer = self._generate(content)
        start = answer.find("{")
        if start < 0:
            return ""
        try:
            value, _ = json.JSONDecoder().raw_decode(answer[start:])
        except json.JSONDecodeError:
            return ""
        return clean_text(value.get("value")) if isinstance(value, dict) else ""

    @torch.inference_mode()
    def embed_texts(self, values: Sequence[str]) -> torch.Tensor:
        if not values:
            return torch.empty((0, self.hidden_dim))
        inputs = self.processor(text=list(values), padding=True, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        outputs = self.model.model(**inputs, use_cache=False, return_dict=True)
        positions = inputs["attention_mask"].sum(dim=-1) - 1
        pooled = outputs.last_hidden_state[torch.arange(len(values), device=self.device), positions]
        return F.normalize(pooled.float(), dim=-1).cpu()
