"""Neural models from the directed joinability Teacher/Student formulation."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .features import OBJECT_TYPES, ObjectFeatures, normalize_object_type

TYPE_TO_ID = {name: index for index, name in enumerate(OBJECT_TYPES)}


def structural_table_pool(
    hidden_states: torch.Tensor, token_groups: torch.Tensor | None
) -> torch.Tensor:
    """Return one token per table schema/example-row group."""

    if hidden_states.shape[0] == 0:
        raise ValueError("A table must contain at least one hidden-state token")
    if token_groups is None:
        return hidden_states
    groups = torch.unique(token_groups, sorted=True)
    return torch.stack([hidden_states[token_groups == group].mean(dim=0) for group in groups])


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
        queries = self.queries.unsqueeze(0)
        pooled, _ = self.attention(
            query=queries,
            key=hidden_states.unsqueeze(0),
            value=hidden_states.unsqueeze(0),
            need_weights=False,
        )
        return self.norm(pooled + queries).squeeze(0)


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
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.text_latents = text_latents
        self.image_latents = image_latents
        self.dropout = dropout

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
        nn.init.normal_(self.rel_token, std=0.02)
        nn.init.normal_(self.sep_token, std=0.02)

    def config(self) -> dict[str, int | float]:
        return {
            "input_dim": self.input_dim,
            "model_dim": self.model_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "text_latents": self.text_latents,
            "image_latents": self.image_latents,
            "dropout": self.dropout,
        }

    def compress(self, features: ObjectFeatures) -> torch.Tensor:
        object_type = normalize_object_type(features.object_type)
        if features.hidden_states is None:
            raise ValueError(f"{features.object_id}: Teacher requires hidden_states")
        if object_type == "table":
            # Group pooling is a mean, so it commutes with the affine adapter.
            # Pool first to avoid projecting table-token detail that is discarded.
            tokens = structural_table_pool(features.hidden_states, features.token_groups)
            tokens = self.adapters[object_type](tokens)
            token_kinds = torch.ones(tokens.shape[0], dtype=torch.long, device=tokens.device)
            token_kinds[0] = 0
            return tokens + self.table_token_embeddings(token_kinds)
        hidden = self.adapters[object_type](features.hidden_states)
        return self.poolers[object_type](hidden)

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

        sequences = []
        for source, source_type, destination, destination_type in zip(
            source_tokens, source_types, destination_tokens, destination_types
        ):
            source_type = normalize_object_type(source_type)
            destination_type = normalize_object_type(destination_type)
            source_id = TYPE_TO_ID[source_type]
            destination_id = TYPE_TO_ID[destination_type]
            rel = self.rel_token + self.type_pair_embeddings.weight[source_id * len(OBJECT_TYPES) + destination_id]
            source_with_identity = source + self.modality_embeddings.weight[source_id] + self.role_embeddings.weight[0]
            destination_with_identity = (
                destination + self.modality_embeddings.weight[destination_id] + self.role_embeddings.weight[1]
            )
            sequences.append(
                torch.cat(
                    [rel.unsqueeze(0), source_with_identity, self.sep_token.unsqueeze(0), destination_with_identity],
                    dim=0,
                )
            )

        lengths = torch.tensor([sequence.shape[0] for sequence in sequences], device=sequences[0].device)
        inputs = pad_sequence(sequences, batch_first=True)
        positions = torch.arange(inputs.shape[1], device=inputs.device).unsqueeze(0)
        padding_mask = positions >= lengths.unsqueeze(1)
        encoded = self.relation_transformer(inputs, src_key_padding_mask=padding_mask)
        return self.scoring_head(encoded[:, 0]).squeeze(-1)

    def score_pairs(
        self,
        sources: Sequence[ObjectFeatures],
        destinations: Sequence[ObjectFeatures],
    ) -> torch.Tensor:
        if len(sources) != len(destinations):
            raise ValueError("Pair inputs must have equal lengths")
        compressed: dict[str, torch.Tensor] = {}
        for features in (*sources, *destinations):
            if features.object_id not in compressed:
                compressed[features.object_id] = self.compress(features)
        return self.score_compressed_pairs(
            [compressed[features.object_id] for features in sources],
            [features.object_type for features in sources],
            [compressed[features.object_id] for features in destinations],
            [features.object_type for features in destinations],
        )


class StudentJoinabilityModel(nn.Module):
    """Independent type projections with an ordered relation per type pair."""

    def __init__(self, input_dim: int, student_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.student_dim = student_dim
        self.projections = nn.ModuleDict(
            {object_type: nn.Linear(input_dim, student_dim, bias=False) for object_type in OBJECT_TYPES}
        )
        self.relations = nn.ParameterDict()
        for source_type in OBJECT_TYPES:
            for destination_type in OBJECT_TYPES:
                relation = torch.eye(student_dim) + 0.01 * torch.randn(student_dim, student_dim)
                self.relations[self.relation_key(source_type, destination_type)] = nn.Parameter(relation)

    @staticmethod
    def relation_key(source_type: str, destination_type: str) -> str:
        return f"{normalize_object_type(source_type)}_to_{normalize_object_type(destination_type)}"

    def config(self) -> dict[str, int]:
        return {"input_dim": self.input_dim, "student_dim": self.student_dim}

    def project(self, embedding: torch.Tensor, object_type: str) -> torch.Tensor:
        return self.projections[normalize_object_type(object_type)](embedding)

    def score_embeddings(
        self,
        source_embedding: torch.Tensor,
        source_type: str,
        destination_embedding: torch.Tensor,
        destination_type: str,
    ) -> torch.Tensor:
        source = self.project(source_embedding, source_type)
        destination = self.project(destination_embedding, destination_type)
        relation = self.relations[self.relation_key(source_type, destination_type)]
        return ((source @ relation) * destination).sum(dim=-1)

    def score_pairs(
        self,
        sources: Sequence[ObjectFeatures],
        destinations: Sequence[ObjectFeatures],
    ) -> torch.Tensor:
        if len(sources) != len(destinations):
            raise ValueError("Pair inputs must have equal lengths")
        if not sources:
            parameter = next(self.parameters())
            return parameter.new_empty(0)
        scores = [
            self.score_embeddings(source.embedding, source.object_type, destination.embedding, destination.object_type)
            for source, destination in zip(sources, destinations)
        ]
        return torch.stack(scores)

    def relation_query(
        self,
        source_embedding: torch.Tensor,
        source_type: str,
        destination_type: str,
    ) -> torch.Tensor:
        source = self.project(source_embedding, source_type)
        relation = self.relations[self.relation_key(source_type, destination_type)]
        return source @ relation

    def index_vector(self, destination_embedding: torch.Tensor, destination_type: str) -> torch.Tensor:
        return self.project(destination_embedding, destination_type)
