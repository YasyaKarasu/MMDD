"""Neural models from the directed joinability Teacher/Student formulation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from contextlib import nullcontext
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from .features import OBJECT_TYPES, ObjectFeatures, normalize_object_type

TYPE_TO_ID = {name: index for index, name in enumerate(OBJECT_TYPES)}
STUDENT_INITIALIZATIONS = (
    "random",
    "identity",
    "identity_noise",
    "orthogonal",
    "random_orthogonal",
    "pca",
)
STUDENT_RELATION_PARAMS = ("full", "lowrank")
STUDENT_SCORE_SPACES = ("raw_logit", "confidence_logit", "confidence")
STUDENT_PROJECTION_MODES = ("shared", "split")
TABLE_ROLES = ("query", "target")


def structural_table_pool_with_groups(
    hidden_states: torch.Tensor,
    token_groups: torch.Tensor | None,
    tokens_per_group: int = 1,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pool each schema/row into ordered contiguous semantic segments."""

    if hidden_states.shape[0] == 0:
        raise ValueError("A table must contain at least one hidden-state token")
    if tokens_per_group <= 0:
        raise ValueError("tokens_per_group must be positive")
    if token_groups is None:
        return hidden_states, None
    pooled = []
    pooled_groups = []
    for group in torch.unique(token_groups, sorted=True):
        values = hidden_states[token_groups == group]
        chunks = torch.tensor_split(values, min(tokens_per_group, values.shape[0]))
        pooled.extend(chunk.mean(dim=0) for chunk in chunks)
        pooled_groups.extend([int(group)] * len(chunks))
    return torch.stack(pooled), torch.tensor(
        pooled_groups,
        dtype=token_groups.dtype,
        device=token_groups.device,
    )


def structural_table_pool(
    hidden_states: torch.Tensor,
    token_groups: torch.Tensor | None,
    tokens_per_group: int = 1,
) -> torch.Tensor:
    """Return up to ``tokens_per_group`` tokens per schema/example-row group."""

    pooled, _ = structural_table_pool_with_groups(
        hidden_states,
        token_groups,
        tokens_per_group,
    )
    return pooled


class LearnedQueryPooler(nn.Module):
    """Compress variable-length text/image tokens into learned semantic slots."""

    def __init__(self, model_dim: int, num_heads: int, num_latents: int) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.empty(num_latents, model_dim))
        self.attention = nn.MultiheadAttention(model_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(model_dim)
        nn.init.normal_(self.queries, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            raise ValueError("Cannot pool an object with no hidden-state tokens")
        return self.forward_batch(
            hidden_states.unsqueeze(0),
            torch.zeros(
                (1, hidden_states.shape[0]),
                dtype=torch.bool,
                device=hidden_states.device,
            ),
        ).squeeze(0)

    def forward_batch(
        self, hidden_states: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool a padded batch of objects into learned semantic slots."""

        if hidden_states.ndim != 3 or hidden_states.shape[1] == 0:
            raise ValueError("Batched hidden states must have shape [batch, tokens, dim]")
        if padding_mask.shape != hidden_states.shape[:2]:
            raise ValueError("Pooler padding mask must match batch and token dimensions")
        queries = self.queries.unsqueeze(0).expand(hidden_states.shape[0], -1, -1)
        pooled, _ = self.attention(
            query=queries,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        return self.norm(pooled + queries)


class TeacherJoinabilityModel(nn.Module):
    """Shared cross-object Relation Transformer over frozen Qwen tokens."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        text_latents: int = 16,
        image_latents: int = 24,
        table_tokens_per_group: int = 1,
        dropout: float = 0.1,
        confidence_transform: bool = False,
        confidence_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if table_tokens_per_group <= 0:
            raise ValueError("table_tokens_per_group must be positive")
        if not 0 < confidence_epsilon < 1:
            raise ValueError("confidence_epsilon must be between 0 and 1")
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.text_latents = text_latents
        self.image_latents = image_latents
        self.table_tokens_per_group = table_tokens_per_group
        self.dropout = dropout
        self.confidence_transform = False
        self.confidence_epsilon = float(confidence_epsilon)
        self.compute_dtype: torch.dtype | None = None

        self.adapters = nn.ModuleDict(
            {object_type: nn.Linear(input_dim, model_dim) for object_type in OBJECT_TYPES}
        )
        self.poolers = nn.ModuleDict(
            {
                "text": LearnedQueryPooler(model_dim, num_heads, text_latents),
                "image": LearnedQueryPooler(model_dim, num_heads, image_latents),
            }
        )
        self.modality_embeddings = nn.Embedding(len(OBJECT_TYPES), model_dim)
        self.role_embeddings = nn.Embedding(2, model_dim)
        self.table_token_embeddings = nn.Embedding(2, model_dim)
        self.type_pair_embeddings = nn.Embedding(len(OBJECT_TYPES) ** 2, model_dim)
        self.rel_token = nn.Parameter(torch.empty(model_dim))
        self.sep_token = nn.Parameter(torch.empty(model_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.relation_transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(model_dim),
            enable_nested_tensor=False,
        )
        self.scoring_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )
        self.confidence_alphas = nn.ParameterDict()
        self.confidence_biases = nn.ParameterDict()
        nn.init.normal_(self.rel_token, std=0.02)
        nn.init.normal_(self.sep_token, std=0.02)
        self.set_confidence_transform(confidence_transform)

    @staticmethod
    def relation_key(source_type: str, destination_type: str) -> str:
        return (
            f"{normalize_object_type(source_type)}_to_"
            f"{normalize_object_type(destination_type)}"
        )

    def config(self) -> dict[str, int | float | bool]:
        return {
            "input_dim": self.input_dim,
            "model_dim": self.model_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "text_latents": self.text_latents,
            "image_latents": self.image_latents,
            "table_tokens_per_group": self.table_tokens_per_group,
            "dropout": self.dropout,
            "confidence_transform": self.confidence_transform,
            "confidence_epsilon": self.confidence_epsilon,
        }

    def set_confidence_transform(self, enabled: bool) -> None:
        """Enable monotonic type-pair confidence logits for Teacher edges."""

        enabled = bool(enabled)
        if enabled == self.confidence_transform:
            return
        if not enabled:
            self.confidence_alphas = nn.ParameterDict()
            self.confidence_biases = nn.ParameterDict()
            self.confidence_transform = False
            return
        parameter = next(self.parameters())
        initial_alpha = math.log(math.expm1(1.0 - self.confidence_epsilon))
        self.confidence_alphas = nn.ParameterDict(
            {
                self.relation_key(source_type, destination_type): nn.Parameter(
                    parameter.new_tensor(initial_alpha)
                )
                for source_type in OBJECT_TYPES
                for destination_type in OBJECT_TYPES
            }
        )
        self.confidence_biases = nn.ParameterDict(
            {
                self.relation_key(source_type, destination_type): nn.Parameter(
                    parameter.new_tensor(0.0)
                )
                for source_type in OBJECT_TYPES
                for destination_type in OBJECT_TYPES
            }
        )
        self.confidence_transform = True

    def confidence_scale(
        self, source_type: str, destination_type: str
    ) -> torch.Tensor:
        key = self.relation_key(source_type, destination_type)
        if self.confidence_transform:
            return F.softplus(self.confidence_alphas[key]) + self.confidence_epsilon
        return next(self.parameters()).new_tensor(1.0)

    def transform_edge_scores(
        self,
        raw_scores: torch.Tensor,
        source_type: str,
        destination_type: str,
        score_space: str,
    ) -> torch.Tensor:
        """Map raw Teacher scores to confidence logits or confidences."""

        if score_space not in STUDENT_SCORE_SPACES:
            raise ValueError(f"score_space must be one of {STUDENT_SCORE_SPACES}")
        if score_space == "raw_logit":
            return raw_scores
        logits = raw_scores
        if self.confidence_transform:
            key = self.relation_key(source_type, destination_type)
            logits = (
                self.confidence_scale(source_type, destination_type) * raw_scores
                + self.confidence_biases[key]
            )
        return logits if score_space == "confidence_logit" else torch.sigmoid(logits)

    def transform_pair_scores(
        self,
        raw_scores: torch.Tensor,
        source_types: Sequence[str],
        destination_types: Sequence[str],
        score_space: str,
    ) -> torch.Tensor:
        """Transform a heterogeneous batch without changing pair order."""

        if raw_scores.shape != (len(source_types),) or len(source_types) != len(
            destination_types
        ):
            raise ValueError("Pair score and type inputs must have equal lengths")
        if score_space == "raw_logit" or not source_types:
            return raw_scores
        transformed = torch.empty_like(raw_scores)
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, pair in enumerate(zip(source_types, destination_types)):
            groups[pair].append(index)
        for (source_type, destination_type), indices in groups.items():
            index_tensor = torch.tensor(indices, device=raw_scores.device)
            transformed[index_tensor] = self.transform_edge_scores(
                raw_scores.index_select(0, index_tensor),
                source_type,
                destination_type,
                score_space,
            )
        return transformed

    def set_compute_dtype(self, dtype: torch.dtype | None) -> None:
        """Select optional autocast compute without changing checkpoint weights."""

        if dtype not in {None, torch.bfloat16}:
            raise ValueError("Teacher compute dtype must be None or torch.bfloat16")
        self.compute_dtype = dtype

    def _autocast_context(self):
        device_type = self.rel_token.device.type
        if self.compute_dtype is None or device_type != "cuda":
            return nullcontext()
        return torch.autocast(device_type, dtype=self.compute_dtype)

    def compress(self, features: ObjectFeatures) -> torch.Tensor:
        object_type = normalize_object_type(features.object_type)
        if features.hidden_states is None:
            raise ValueError(f"{features.object_id}: Teacher requires hidden_states")
        if object_type == "table":
            # Segment pooling commutes with the affine adapter. Pool first to
            # avoid projecting table-token detail that is discarded.
            tokens, groups = structural_table_pool_with_groups(
                features.hidden_states,
                features.token_groups,
                self.table_tokens_per_group,
            )
            tokens = self.adapters[object_type](tokens)
            if groups is None:
                token_kinds = torch.ones(
                    tokens.shape[0], dtype=torch.long, device=tokens.device
                )
                token_kinds[0] = 0
            else:
                token_kinds = groups.ne(0).long()
            return tokens + self.table_token_embeddings(token_kinds)
        hidden = self.adapters[object_type](features.hidden_states)
        return self.poolers[object_type](hidden)

    @staticmethod
    def _compression_buckets(
        features: Sequence[ObjectFeatures], max_padded_tokens: int = 8192
    ) -> list[list[ObjectFeatures]]:
        """Group similarly sized objects without creating oversized padded tensors."""

        ordered = sorted(
            features,
            key=lambda item: int(item.hidden_states.shape[0]),
        )
        buckets: list[list[ObjectFeatures]] = []
        current: list[ObjectFeatures] = []
        current_max = 0
        for item in ordered:
            length = int(item.hidden_states.shape[0])
            next_max = max(current_max, length)
            if current and next_max * (len(current) + 1) > max_padded_tokens:
                buckets.append(current)
                current = []
                current_max = 0
            current.append(item)
            current_max = max(current_max, length)
        if current:
            buckets.append(current)
        return buckets

    def compress_many(
        self,
        features: Sequence[ObjectFeatures],
        compression_cache: dict[str, torch.Tensor],
    ) -> None:
        """Compress missing objects in modality and length batches."""

        grouped: dict[str, list[ObjectFeatures]] = defaultdict(list)
        for item in features:
            if item.hidden_states is None:
                raise ValueError(f"{item.object_id}: Teacher requires hidden_states")
            grouped[normalize_object_type(item.object_type)].append(item)

        table_features = grouped.pop("table", [])
        if table_features:
            pooled_with_groups = [
                structural_table_pool_with_groups(
                    item.hidden_states,
                    item.token_groups,
                    self.table_tokens_per_group,
                )
                for item in table_features
            ]
            pooled = [tokens for tokens, _groups in pooled_with_groups]
            lengths = [tokens.shape[0] for tokens in pooled]
            projected = self.adapters["table"](
                pad_sequence(pooled, batch_first=True)
            )
            token_kinds = pad_sequence(
                [
                    groups.ne(0).long()
                    if groups is not None
                    else torch.cat(
                        [
                            torch.zeros(1, dtype=torch.long, device=tokens.device),
                            torch.ones(
                                tokens.shape[0] - 1,
                                dtype=torch.long,
                                device=tokens.device,
                            ),
                        ]
                    )
                    for tokens, groups in pooled_with_groups
                ],
                batch_first=True,
                padding_value=1,
            )
            projected = projected + self.table_token_embeddings(token_kinds)
            for item, tokens, length in zip(table_features, projected, lengths):
                compression_cache[item.object_id] = tokens[:length]

        for object_type, items in grouped.items():
            for bucket in self._compression_buckets(items):
                lengths = torch.tensor(
                    [item.hidden_states.shape[0] for item in bucket],
                    device=bucket[0].hidden_states.device,
                )
                hidden = pad_sequence(
                    [item.hidden_states for item in bucket], batch_first=True
                )
                hidden = self.adapters[object_type](hidden)
                positions = torch.arange(
                    hidden.shape[1], device=hidden.device
                ).unsqueeze(0)
                padding_mask = positions >= lengths.unsqueeze(1)
                pooled = self.poolers[object_type].forward_batch(
                    hidden, padding_mask
                )
                for item, tokens in zip(bucket, pooled):
                    compression_cache[item.object_id] = tokens

    def score_compressed_pairs(
        self,
        source_tokens: Sequence[torch.Tensor],
        source_types: Sequence[str],
        destination_tokens: Sequence[torch.Tensor],
        destination_types: Sequence[str],
    ) -> torch.Tensor:
        if not source_tokens:
            return self.rel_token.new_empty(0)
        if not (
            len(source_tokens)
            == len(source_types)
            == len(destination_tokens)
            == len(destination_types)
        ):
            raise ValueError("Pair inputs must have equal lengths")

        device = source_tokens[0].device
        batch_size = len(source_tokens)
        source_lengths = torch.tensor(
            [tokens.shape[0] for tokens in source_tokens], device=device
        )
        destination_lengths = torch.tensor(
            [tokens.shape[0] for tokens in destination_tokens], device=device
        )
        source_ids = torch.tensor(
            [TYPE_TO_ID[normalize_object_type(value)] for value in source_types],
            device=device,
        )
        destination_ids = torch.tensor(
            [
                TYPE_TO_ID[normalize_object_type(value)]
                for value in destination_types
            ],
            device=device,
        )
        lengths = source_lengths + destination_lengths + 2
        input_dtype = self.compute_dtype or self.rel_token.dtype
        inputs = torch.zeros(
            (batch_size, int(lengths.max()), self.model_dim),
            dtype=input_dtype,
            device=device,
        )
        rows = torch.arange(batch_size, device=device)
        pair_ids = source_ids * len(OBJECT_TYPES) + destination_ids
        inputs[:, 0] = (
            self.rel_token + self.type_pair_embeddings(pair_ids)
        ).to(input_dtype)

        padded_sources = pad_sequence(source_tokens, batch_first=True).to(input_dtype)
        padded_sources = padded_sources + (
            self.modality_embeddings(source_ids) + self.role_embeddings.weight[0]
        ).to(input_dtype).unsqueeze(1)
        inputs[:, 1 : 1 + padded_sources.shape[1]] = padded_sources
        inputs[rows, source_lengths + 1] = self.sep_token.to(input_dtype)

        padded_destinations = pad_sequence(
            destination_tokens, batch_first=True
        ).to(input_dtype)
        padded_destinations = padded_destinations + (
            self.modality_embeddings(destination_ids)
            + self.role_embeddings.weight[1]
        ).to(input_dtype).unsqueeze(1)
        destination_offsets = torch.arange(
            padded_destinations.shape[1], device=device
        ).unsqueeze(0)
        destination_positions = source_lengths.unsqueeze(1) + 2 + destination_offsets
        destination_mask = destination_offsets < destination_lengths.unsqueeze(1)
        destination_rows = rows.unsqueeze(1).expand_as(destination_positions)
        inputs[
            destination_rows[destination_mask],
            destination_positions[destination_mask],
        ] = padded_destinations[destination_mask]

        positions = torch.arange(inputs.shape[1], device=inputs.device).unsqueeze(0)
        padding_mask = positions >= lengths.unsqueeze(1)
        encoded = self.relation_transformer(inputs, src_key_padding_mask=padding_mask)
        return self.scoring_head(encoded[:, 0]).squeeze(-1)

    def score_pairs(
        self,
        sources: Sequence[ObjectFeatures],
        destinations: Sequence[ObjectFeatures],
        *,
        compression_cache: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if len(sources) != len(destinations):
            raise ValueError("Pair inputs must have equal lengths")
        if not sources:
            return self.rel_token.new_empty(0)
        compressed = compression_cache if compression_cache is not None else {}
        missing = []
        seen = set(compressed)
        for features in (*sources, *destinations):
            if features.object_id not in seen:
                missing.append(features)
                seen.add(features.object_id)
        with self._autocast_context():
            self.compress_many(missing, compressed)
            scores = self.score_compressed_pairs(
                [compressed[features.object_id] for features in sources],
                [features.object_type for features in sources],
                [compressed[features.object_id] for features in destinations],
                [features.object_type for features in destinations],
            )
        return scores.float()


class StudentJoinabilityModel(nn.Module):
    """Independent type projections with an ordered relation per type pair."""

    def __init__(
        self,
        input_dim: int,
        student_dim: int = 128,
        initialization: str = "random",
        initialization_noise_std: float = 0.01,
        initialization_basis: torch.Tensor | None = None,
        freeze_projections: bool = False,
        relation_param: str = "full",
        relation_rank: int = 16,
        confidence_transform: bool = False,
        confidence_epsilon: float = 1e-6,
        projection_mode: str = "shared",
    ) -> None:
        super().__init__()
        if input_dim <= 0 or student_dim <= 0:
            raise ValueError("input_dim and student_dim must be positive")
        if initialization not in STUDENT_INITIALIZATIONS:
            raise ValueError(
                f"initialization must be one of {STUDENT_INITIALIZATIONS}"
            )
        if initialization_noise_std < 0:
            raise ValueError("initialization_noise_std must be non-negative")
        if relation_param not in STUDENT_RELATION_PARAMS:
            raise ValueError(
                f"relation_param must be one of {STUDENT_RELATION_PARAMS}"
            )
        if relation_rank <= 0:
            raise ValueError("relation_rank must be positive")
        if not 0 < confidence_epsilon < 1:
            raise ValueError("confidence_epsilon must be between 0 and 1")
        if projection_mode not in STUDENT_PROJECTION_MODES:
            raise ValueError(
                f"projection_mode must be one of {STUDENT_PROJECTION_MODES}"
            )
        if initialization in {"identity", "identity_noise"} and student_dim != input_dim:
            raise ValueError(
                f"{initialization} initialization requires student_dim == input_dim"
            )
        if initialization in {"orthogonal", "random_orthogonal", "pca"} and student_dim > input_dim:
            raise ValueError(
                f"{initialization} initialization requires student_dim <= input_dim"
            )
        if initialization == "pca":
            if initialization_basis is None:
                raise ValueError("pca initialization requires initialization_basis")
            if initialization_basis.shape != (student_dim, input_dim):
                raise ValueError(
                    "initialization_basis must have shape [student_dim, input_dim]"
                )
            if not torch.isfinite(initialization_basis).all():
                raise ValueError("initialization_basis must be finite")
            gram = initialization_basis @ initialization_basis.T
            if not torch.allclose(
                gram,
                torch.eye(student_dim, device=gram.device, dtype=gram.dtype),
                atol=1e-4,
                rtol=1e-4,
            ):
                raise ValueError("initialization_basis rows must be orthonormal")
        self.input_dim = input_dim
        self.student_dim = student_dim
        self.initialization = initialization
        self.initialization_noise_std = initialization_noise_std
        self.freeze_projections = bool(freeze_projections)
        self.relation_param = relation_param
        self.relation_rank = relation_rank
        self.confidence_transform = False
        self.confidence_epsilon = float(confidence_epsilon)
        self.projection_mode = projection_mode
        projection_keys = (
            ("table_query", "table_target", "text", "image")
            if projection_mode == "split"
            else OBJECT_TYPES
        )
        self.projections = nn.ModuleDict(
            {key: nn.Linear(input_dim, student_dim, bias=False) for key in projection_keys}
        )

        if initialization in {"identity", "identity_noise"}:
            with torch.no_grad():
                for projection in self.projections.values():
                    nn.init.eye_(projection.weight)
                    if initialization == "identity_noise":
                        projection.weight.add_(
                            initialization_noise_std
                            * torch.randn_like(projection.weight)
                        )
        elif initialization in {"orthogonal", "random_orthogonal", "pca"}:
            if initialization in {"orthogonal", "random_orthogonal"}:
                basis, _ = torch.linalg.qr(
                    torch.randn(input_dim, student_dim), mode="reduced"
                )
                projection_weight = basis.T
            else:
                assert initialization_basis is not None
                projection_weight = initialization_basis
            with torch.no_grad():
                for projection in self.projections.values():
                    projection.weight.copy_(projection_weight)

        self.register_buffer(
            "initial_projection_weights",
            torch.stack(
                [self.projections[key].weight.detach().clone() for key in self.projection_keys]
            ),
            persistent=True,
        )
        self.register_buffer(
            "stage_initial_projection_weights",
            self.initial_projection_weights.detach().clone(),
        )
        self.projection_reference_origin = "initialization"
        self.set_projection_frozen(self.freeze_projections)

        self.relations = nn.ParameterDict()
        self.relation_as = nn.ParameterDict()
        self.relation_bs = nn.ParameterDict()
        self.confidence_alphas = nn.ParameterDict()
        self.confidence_biases = nn.ParameterDict()
        for source_type in OBJECT_TYPES:
            for destination_type in OBJECT_TYPES:
                key = self.relation_key(source_type, destination_type)
                if relation_param == "full":
                    relation = torch.eye(student_dim)
                    if initialization not in {
                        "identity",
                        "orthogonal",
                        "random_orthogonal",
                        "pca",
                    }:
                        relation = relation + 0.01 * torch.randn(
                            student_dim, student_dim
                        )
                    self.relations[key] = nn.Parameter(relation)
                else:
                    self.relation_as[key] = nn.Parameter(
                        0.01 * torch.randn(student_dim, relation_rank)
                    )
                    self.relation_bs[key] = nn.Parameter(
                        torch.zeros(student_dim, relation_rank)
                    )
        self.set_confidence_transform(confidence_transform)

    @staticmethod
    def relation_key(source_type: str, destination_type: str) -> str:
        return f"{normalize_object_type(source_type)}_to_{normalize_object_type(destination_type)}"

    def config(self) -> dict[str, int | float | str | bool]:
        return {
            "input_dim": self.input_dim,
            "student_dim": self.student_dim,
            "initialization": self.initialization,
            "initialization_noise_std": self.initialization_noise_std,
            "freeze_projections": self.freeze_projections,
            "relation_param": self.relation_param,
            "relation_rank": self.relation_rank,
            "confidence_transform": self.confidence_transform,
            "confidence_epsilon": self.confidence_epsilon,
            "projection_mode": self.projection_mode,
        }

    @property
    def projection_keys(self) -> tuple[str, ...]:
        return tuple(self.projections.keys())

    def projection_key(self, object_type: str, role: str | None = None) -> str:
        """Resolve an object projection from its runtime endpoint role."""

        object_type = normalize_object_type(object_type)
        if object_type != "table" or self.projection_mode == "shared":
            return object_type
        if role not in TABLE_ROLES:
            raise ValueError("Split table projection requires role='query' or 'target'")
        return f"table_{role}"

    @property
    def ann_dim(self) -> int:
        """Vector dimension used by exact inner-product ANN retrieval."""

        if self.relation_param == "lowrank":
            return self.student_dim + self.relation_rank
        return self.student_dim

    def relation_parameters(self) -> list[nn.Parameter]:
        """Return the trainable parameters of all directed relations."""

        relation_parameters = (
            list(self.relations.parameters())
            if self.relation_param == "full"
            else [
                *self.relation_as.parameters(),
                *self.relation_bs.parameters(),
            ]
        )
        return [*relation_parameters, *self.confidence_parameters()]

    def confidence_parameters(self) -> list[nn.Parameter]:
        """Return the optional monotonic type-pair calibration parameters."""

        return [
            *self.confidence_alphas.parameters(),
            *self.confidence_biases.parameters(),
        ]

    def set_confidence_transform(self, enabled: bool) -> None:
        """Enable or disable trainable monotonic type-pair calibration."""

        enabled = bool(enabled)
        if enabled == self.confidence_transform:
            return
        if not enabled:
            self.confidence_alphas = nn.ParameterDict()
            self.confidence_biases = nn.ParameterDict()
            self.confidence_transform = False
            return
        parameter = next(self.parameters())
        initial_alpha = math.log(math.expm1(1.0 - self.confidence_epsilon))
        self.confidence_alphas = nn.ParameterDict(
            {
                self.relation_key(source_type, destination_type): nn.Parameter(
                    parameter.new_tensor(initial_alpha)
                )
                for source_type in OBJECT_TYPES
                for destination_type in OBJECT_TYPES
            }
        )
        self.confidence_biases = nn.ParameterDict(
            {
                self.relation_key(source_type, destination_type): nn.Parameter(
                    parameter.new_tensor(0.0)
                )
                for source_type in OBJECT_TYPES
                for destination_type in OBJECT_TYPES
            }
        )
        self.confidence_transform = True

    def confidence_scale(
        self, source_type: str, destination_type: str
    ) -> torch.Tensor:
        """Return the positive affine scale for one ordered type pair."""

        key = self.relation_key(source_type, destination_type)
        if self.confidence_transform:
            return F.softplus(self.confidence_alphas[key]) + self.confidence_epsilon
        return next(self.parameters()).new_tensor(1.0)

    def transform_edge_scores(
        self,
        raw_scores: torch.Tensor,
        source_type: str,
        destination_type: str,
        score_space: str,
    ) -> torch.Tensor:
        """Map raw bilinear scores to explicit logits or sigmoid confidence."""

        if score_space not in STUDENT_SCORE_SPACES:
            raise ValueError(f"score_space must be one of {STUDENT_SCORE_SPACES}")
        if score_space == "raw_logit":
            return raw_scores
        key = self.relation_key(source_type, destination_type)
        logits = raw_scores
        if self.confidence_transform:
            logits = (
                self.confidence_scale(source_type, destination_type) * raw_scores
                + self.confidence_biases[key]
            )
        return logits if score_space == "confidence_logit" else torch.sigmoid(logits)

    def relation_residual_squared_norm(self, key: str) -> torch.Tensor:
        """Return ``||R - I||_F^2`` without materializing a low-rank matrix."""

        if self.relation_param == "full":
            relation = self.relations[key]
            identity = torch.eye(
                self.student_dim,
                device=relation.device,
                dtype=relation.dtype,
            )
            return (relation - identity).square().sum()
        left_gram = self.relation_as[key].T @ self.relation_as[key]
        right_gram = self.relation_bs[key].T @ self.relation_bs[key]
        return (left_gram * right_gram).sum()

    def set_projection_frozen(self, frozen: bool) -> None:
        """Freeze or unfreeze the object-type projections explicitly."""

        self.freeze_projections = bool(frozen)
        for projection in self.projections.values():
            projection.weight.requires_grad_(not self.freeze_projections)

    @torch.no_grad()
    def reset_projection_anchors(self) -> None:
        """Record a stage boundary without changing the full-chain reference."""

        self.stage_initial_projection_weights.copy_(
            torch.stack(
                [self.projections[key].weight for key in self.projection_keys]
            )
        )

    def project(
        self,
        embedding: torch.Tensor,
        object_type: str,
        *,
        role: str | None = None,
    ) -> torch.Tensor:
        return self.projections[self.projection_key(object_type, role)](embedding)

    def _score_projected_pairs(
        self,
        sources: torch.Tensor,
        destinations: torch.Tensor,
        relation_key: str,
    ) -> torch.Tensor:
        if self.relation_param == "full":
            return (
                (sources @ self.relations[relation_key]) * destinations
            ).sum(dim=-1)
        direct = (sources * destinations).sum(dim=-1)
        residual = (
            (sources @ self.relation_as[relation_key])
            * (destinations @ self.relation_bs[relation_key])
        ).sum(dim=-1)
        return direct + residual

    def score_embeddings(
        self,
        source_embedding: torch.Tensor,
        source_type: str,
        destination_embedding: torch.Tensor,
        destination_type: str,
        *,
        source_role: str | None = None,
        destination_role: str | None = None,
    ) -> torch.Tensor:
        source = self.project(
            source_embedding,
            source_type,
            role=source_role or ("query" if normalize_object_type(source_type) == "table" else None),
        )
        destination = self.project(
            destination_embedding,
            destination_type,
            role=destination_role or ("target" if normalize_object_type(destination_type) == "table" else None),
        )
        key = self.relation_key(source_type, destination_type)
        return self._score_projected_pairs(source, destination, key)

    def raw_score_embeddings(
        self,
        source_embedding: torch.Tensor,
        source_type: str,
        destination_embedding: torch.Tensor,
        destination_type: str,
    ) -> torch.Tensor:
        """Return the unbounded bilinear logit used for ANN ordering."""

        return self.score_embeddings(
            source_embedding,
            source_type,
            destination_embedding,
            destination_type,
        )

    def confidence_logit_embeddings(
        self,
        source_embedding: torch.Tensor,
        source_type: str,
        destination_embedding: torch.Tensor,
        destination_type: str,
    ) -> torch.Tensor:
        raw_scores = self.raw_score_embeddings(
            source_embedding,
            source_type,
            destination_embedding,
            destination_type,
        )
        return self.transform_edge_scores(
            raw_scores, source_type, destination_type, "confidence_logit"
        )

    def confidence_embeddings(
        self,
        source_embedding: torch.Tensor,
        source_type: str,
        destination_embedding: torch.Tensor,
        destination_type: str,
    ) -> torch.Tensor:
        raw_scores = self.raw_score_embeddings(
            source_embedding,
            source_type,
            destination_embedding,
            destination_type,
        )
        return self.transform_edge_scores(
            raw_scores, source_type, destination_type, "confidence"
        )

    def score_embedding_matrix(
        self,
        source_embeddings: torch.Tensor,
        source_type: str,
        destination_embeddings: torch.Tensor,
        destination_type: str,
        *,
        source_role: str | None = None,
        destination_role: str | None = None,
    ) -> torch.Tensor:
        """Score every source/destination pair in two embedding batches."""

        sources = self.project(
            source_embeddings,
            source_type,
            role=source_role or ("query" if normalize_object_type(source_type) == "table" else None),
        )
        destinations = self.project(
            destination_embeddings,
            destination_type,
            role=destination_role or ("target" if normalize_object_type(destination_type) == "table" else None),
        )
        key = self.relation_key(source_type, destination_type)
        if self.relation_param == "full":
            return sources @ self.relations[key] @ destinations.T
        scores = sources @ destinations.T
        return scores + (
            (sources @ self.relation_as[key])
            @ (destinations @ self.relation_bs[key]).T
        )

    def raw_score_embedding_matrix(
        self,
        source_embeddings: torch.Tensor,
        source_type: str,
        destination_embeddings: torch.Tensor,
        destination_type: str,
    ) -> torch.Tensor:
        """Return raw bilinear logits for every source/destination pair."""

        return self.score_embedding_matrix(
            source_embeddings,
            source_type,
            destination_embeddings,
            destination_type,
        )

    def confidence_logit_embedding_matrix(
        self,
        source_embeddings: torch.Tensor,
        source_type: str,
        destination_embeddings: torch.Tensor,
        destination_type: str,
    ) -> torch.Tensor:
        raw_scores = self.raw_score_embedding_matrix(
            source_embeddings,
            source_type,
            destination_embeddings,
            destination_type,
        )
        return self.transform_edge_scores(
            raw_scores, source_type, destination_type, "confidence_logit"
        )

    def confidence_embedding_matrix(
        self,
        source_embeddings: torch.Tensor,
        source_type: str,
        destination_embeddings: torch.Tensor,
        destination_type: str,
    ) -> torch.Tensor:
        raw_scores = self.raw_score_embedding_matrix(
            source_embeddings,
            source_type,
            destination_embeddings,
            destination_type,
        )
        return self.transform_edge_scores(
            raw_scores, source_type, destination_type, "confidence"
        )

    def score_embedding_matrix_in_space(
        self,
        source_embeddings: torch.Tensor,
        source_type: str,
        destination_embeddings: torch.Tensor,
        destination_type: str,
        score_space: str,
    ) -> torch.Tensor:
        raw_scores = self.raw_score_embedding_matrix(
            source_embeddings,
            source_type,
            destination_embeddings,
            destination_type,
        )
        return self.transform_edge_scores(
            raw_scores, source_type, destination_type, score_space
        )

    def score_pairs(
        self,
        sources: Sequence[ObjectFeatures],
        destinations: Sequence[ObjectFeatures],
    ) -> torch.Tensor:
        if len(sources) != len(destinations):
            raise ValueError("Pair inputs must have equal lengths")
        parameter = next(self.parameters())
        if not sources:
            return parameter.new_empty(0)

        projected: dict[tuple[str, str], torch.Tensor] = {}
        features_by_projection: dict[str, dict[str, ObjectFeatures]] = defaultdict(dict)
        for features in sources:
            key = self.projection_key(
                features.object_type,
                "query" if features.object_type == "table" else None,
            )
            features_by_projection[key].setdefault(features.object_id, features)
        for features in destinations:
            key = self.projection_key(
                features.object_type,
                "target" if features.object_type == "table" else None,
            )
            features_by_projection[key].setdefault(features.object_id, features)

        for projection_key, by_id in features_by_projection.items():
            object_ids = list(by_id)
            embeddings = torch.stack(
                [by_id[object_id].embedding for object_id in object_ids]
            ).to(device=parameter.device, dtype=torch.float32)
            vectors = self.projections[projection_key](embeddings)
            projected.update(
                ((projection_key, object_id), vectors[row])
                for row, object_id in enumerate(object_ids)
            )

        pair_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, (source, destination) in enumerate(zip(sources, destinations)):
            pair_groups[(source.object_type, destination.object_type)].append(index)

        scores = parameter.new_empty(len(sources))
        for (source_type, destination_type), pair_indices in pair_groups.items():
            source_vectors = torch.stack(
                [
                    projected[(self.projection_key(source_type, "query" if source_type == "table" else None), sources[index].object_id)]
                    for index in pair_indices
                ]
            )
            destination_vectors = torch.stack(
                [
                    projected[(self.projection_key(destination_type, "target" if destination_type == "table" else None), destinations[index].object_id)]
                    for index in pair_indices
                ]
            )
            key = self.relation_key(source_type, destination_type)
            values = self._score_projected_pairs(
                source_vectors, destination_vectors, key
            )
            indices = torch.tensor(pair_indices, device=scores.device)
            scores = scores.index_copy(0, indices, values)
        return scores

    def raw_score_pairs(
        self,
        sources: Sequence[ObjectFeatures],
        destinations: Sequence[ObjectFeatures],
    ) -> torch.Tensor:
        """Return raw bilinear logits for directed object pairs."""

        return self.score_pairs(sources, destinations)

    def _transform_pair_scores(
        self,
        raw_scores: torch.Tensor,
        sources: Sequence[ObjectFeatures],
        destinations: Sequence[ObjectFeatures],
        score_space: str,
    ) -> torch.Tensor:
        if score_space not in STUDENT_SCORE_SPACES:
            raise ValueError(f"score_space must be one of {STUDENT_SCORE_SPACES}")
        if len(sources) != len(destinations) or len(sources) != len(raw_scores):
            raise ValueError("Pair inputs and raw_scores must have equal lengths")
        if score_space == "raw_logit" or not sources:
            return raw_scores
        transformed = raw_scores.new_empty(raw_scores.shape)
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, (source, destination) in enumerate(zip(sources, destinations)):
            groups[(source.object_type, destination.object_type)].append(index)
        for (source_type, destination_type), pair_indices in groups.items():
            indices = torch.tensor(pair_indices, device=raw_scores.device)
            values = raw_scores.index_select(0, indices)
            values = self.transform_edge_scores(
                values, source_type, destination_type, score_space
            )
            transformed = transformed.index_copy(0, indices, values)
        return transformed

    def confidence_logit_pairs(
        self,
        sources: Sequence[ObjectFeatures],
        destinations: Sequence[ObjectFeatures],
    ) -> torch.Tensor:
        raw_scores = self.raw_score_pairs(sources, destinations)
        return self._transform_pair_scores(
            raw_scores, sources, destinations, "confidence_logit"
        )

    def confidence_pairs(
        self,
        sources: Sequence[ObjectFeatures],
        destinations: Sequence[ObjectFeatures],
    ) -> torch.Tensor:
        raw_scores = self.raw_score_pairs(sources, destinations)
        return self._transform_pair_scores(
            raw_scores, sources, destinations, "confidence"
        )

    def score_pairs_in_space(
        self,
        sources: Sequence[ObjectFeatures],
        destinations: Sequence[ObjectFeatures],
        score_space: str,
    ) -> torch.Tensor:
        raw_scores = self.raw_score_pairs(sources, destinations)
        return self._transform_pair_scores(
            raw_scores, sources, destinations, score_space
        )

    def relation_query(
        self,
        source_embedding: torch.Tensor,
        source_type: str,
        destination_type: str,
        *,
        source_role: str | None = None,
    ) -> torch.Tensor:
        source = self.project(
            source_embedding,
            source_type,
            role=source_role or ("query" if normalize_object_type(source_type) == "table" else None),
        )
        key = self.relation_key(source_type, destination_type)
        if self.relation_param == "full":
            return source @ self.relations[key]
        return torch.cat([source, source @ self.relation_as[key]], dim=-1)

    def index_vector(
        self,
        destination_embedding: torch.Tensor,
        destination_type: str,
        source_type: str | None = None,
        *,
        destination_role: str | None = None,
    ) -> torch.Tensor:
        destination = self.project(
            destination_embedding,
            destination_type,
            role=destination_role or ("target" if normalize_object_type(destination_type) == "table" else None),
        )
        if self.relation_param == "full":
            return destination
        if source_type is None:
            raise ValueError("lowrank ANN index vectors require source_type")
        key = self.relation_key(source_type, destination_type)
        return torch.cat(
            [destination, destination @ self.relation_bs[key]], dim=-1
        )


def split_table_projection(
    shared: StudentJoinabilityModel,
) -> StudentJoinabilityModel:
    """Create a step-0-equivalent Q/T table split from a shared Student."""

    if shared.projection_mode != "shared":
        raise ValueError("Only a shared Student can be migrated to split table roles")
    table_index = shared.projection_keys.index("table")
    basis = shared.initial_projection_weights[table_index].detach().cpu()
    if not bool(torch.isfinite(basis).all()):
        raise ValueError("Role migration requires the persisted PCA reference")
    split = StudentJoinabilityModel(
        shared.input_dim,
        shared.student_dim,
        initialization="pca",
        initialization_basis=basis,
        freeze_projections=shared.freeze_projections,
        relation_param=shared.relation_param,
        relation_rank=shared.relation_rank,
        confidence_transform=shared.confidence_transform,
        confidence_epsilon=shared.confidence_epsilon,
        projection_mode="split",
    )
    split = split.to(next(shared.parameters()).device)
    with torch.no_grad():
        table = shared.projections["table"].weight
        split.projections["table_query"].weight.copy_(table)
        split.projections["table_target"].weight.copy_(table)
        for object_type in ("text", "image"):
            split.projections[object_type].weight.copy_(
                shared.projections[object_type].weight
            )
        for name, parameter in shared.relations.items():
            split.relations[name].copy_(parameter)
        for name, parameter in shared.relation_as.items():
            split.relation_as[name].copy_(parameter)
        for name, parameter in shared.relation_bs.items():
            split.relation_bs[name].copy_(parameter)
        for name, parameter in shared.confidence_alphas.items():
            split.confidence_alphas[name].copy_(parameter)
        for name, parameter in shared.confidence_biases.items():
            split.confidence_biases[name].copy_(parameter)
        shared_initial = {
            key: value
            for key, value in zip(
                shared.projection_keys, shared.initial_projection_weights
            )
        }
        split.initial_projection_weights.copy_(
            torch.stack(
                [
                    shared_initial["table"],
                    shared_initial["table"],
                    shared_initial["text"],
                    shared_initial["image"],
                ]
            )
        )
        split.reset_projection_anchors()
    split.projection_reference_origin = shared.projection_reference_origin
    return split


class IdentityStudentJoinabilityModel(nn.Module):
    """Zero-training Student with every type projection and relation equal to I."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.input_dim = embedding_dim
        self.student_dim = embedding_dim

    def config(self) -> dict[str, int | str]:
        return {
            "input_dim": self.input_dim,
            "student_dim": self.student_dim,
            "projection": "identity",
            "relation": "identity",
        }

    def project(self, embedding: torch.Tensor, object_type: str) -> torch.Tensor:
        normalize_object_type(object_type)
        if embedding.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected embedding dimension {self.input_dim}, got {embedding.shape[-1]}"
            )
        return embedding

    def relation_query(
        self,
        source_embedding: torch.Tensor,
        source_type: str,
        destination_type: str,
    ) -> torch.Tensor:
        normalize_object_type(destination_type)
        return self.project(source_embedding, source_type)

    def index_vector(
        self,
        destination_embedding: torch.Tensor,
        destination_type: str,
        source_type: str | None = None,
    ) -> torch.Tensor:
        if source_type is not None:
            normalize_object_type(source_type)
        return self.project(destination_embedding, destination_type)


class ProjectedIdentityStudentJoinabilityModel(nn.Module):
    """Zero-training shared projection P with every relation fixed to identity."""

    def __init__(self, projection: torch.Tensor) -> None:
        super().__init__()
        if projection.ndim != 2 or min(projection.shape) <= 0:
            raise ValueError("projection must have shape [student_dim, input_dim]")
        projection = projection.detach().float()
        if not torch.isfinite(projection).all():
            raise ValueError("projection must be finite")
        gram = projection @ projection.T
        if not torch.allclose(
            gram,
            torch.eye(projection.shape[0], dtype=projection.dtype),
            atol=1e-4,
            rtol=1e-4,
        ):
            raise ValueError("projection rows must be orthonormal")
        self.input_dim = int(projection.shape[1])
        self.student_dim = int(projection.shape[0])
        self.register_buffer("projection", projection)

    def config(self) -> dict[str, int | str]:
        return {
            "input_dim": self.input_dim,
            "student_dim": self.student_dim,
            "projection": "shared_pca",
            "relation": "identity",
        }

    def project(self, embedding: torch.Tensor, object_type: str) -> torch.Tensor:
        normalize_object_type(object_type)
        if embedding.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected embedding dimension {self.input_dim}, got {embedding.shape[-1]}"
            )
        return torch.nn.functional.linear(embedding, self.projection)

    def relation_query(
        self,
        source_embedding: torch.Tensor,
        source_type: str,
        destination_type: str,
    ) -> torch.Tensor:
        normalize_object_type(destination_type)
        return self.project(source_embedding, source_type)

    def index_vector(
        self,
        destination_embedding: torch.Tensor,
        destination_type: str,
        source_type: str | None = None,
    ) -> torch.Tensor:
        if source_type is not None:
            normalize_object_type(source_type)
        return self.project(destination_embedding, destination_type)


StudentANNModel = (
    StudentJoinabilityModel
    | IdentityStudentJoinabilityModel
    | ProjectedIdentityStudentJoinabilityModel
)
