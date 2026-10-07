"""AMBER event-tokenizer component used by the Qwen-based research adapter."""

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


FEATURE_ROLES = frozenset({"user", "item", "context", "outcome"})
TOKEN_ROLES = {
    "ranking_context": frozenset({"user", "item", "context"}),
    "ranking_label": frozenset({"outcome"}),
    "retrieval_history": FEATURE_ROLES,
    "retrieval_target": frozenset({"item"}),
}


@dataclass
class AMBERConfig:
    categorical_cardinalities: Sequence[int]
    categorical_roles: Sequence[str]
    numerical_roles: Sequence[str] = ()
    d_model: int = 128
    feature_layers: int = 2
    decoder_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.0
    rope_base: float = 10_000.0

    def __post_init__(self) -> None:
        if len(self.categorical_cardinalities) != len(self.categorical_roles):
            raise ValueError("Every categorical field needs a cardinality and role")
        if any(n < 1 for n in self.categorical_cardinalities):
            raise ValueError("Categorical cardinalities must be positive")
        if not set((*self.categorical_roles, *self.numerical_roles)) <= FEATURE_ROLES:
            raise ValueError(f"Feature roles must belong to {sorted(FEATURE_ROLES)}")
        if self.n_heads < 1 or self.d_model % (2 * self.n_heads):
            raise ValueError("d_model must be divisible by 2 * n_heads for RoPE")
        if min(self.feature_layers, self.decoder_layers) < 1:
            raise ValueError("Both transformer stacks require at least one layer")

    @property
    def num_outcomes(self) -> int:
        """Binary prediction heads, in categorical-then-numerical field order."""
        return sum(r == "outcome" for r in (*self.categorical_roles, *self.numerical_roles))


class EventTokenizer(nn.Module):
    """Shared bidirectional feature encoder with one learnable CLS summary.

    ``cats`` and ``nums`` have shapes ``[..., F]`` and ``[..., G]``. A role
    selects fields *before* their values are read; hidden features have no path
    to the returned token. Categorical ID 0 is an ordinary trainable ID; the
    dataset may reserve it for unknown values. Padding is handled by the decoder.
    """

    def __init__(self, config: AMBERConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        n_cats = len(config.categorical_cardinalities)
        n_nums = len(config.numerical_roles)
        self.categorical = nn.ModuleList(
            nn.Embedding(n, d) for n in config.categorical_cardinalities
        )
        self.numerical_weight = nn.Parameter(torch.empty(n_nums, d))
        self.numerical_bias = nn.Parameter(torch.zeros(n_nums, d))
        self.feature_identity = nn.Parameter(torch.empty(n_cats + n_nums, d))
        self.cls = nn.Parameter(torch.empty(1, 1, d))
        nn.init.normal_(self.numerical_weight, std=0.02)
        nn.init.normal_(self.feature_identity, std=0.02)
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.n_heads,
            dim_feedforward=4 * d,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, config.feature_layers, norm=nn.LayerNorm(d), enable_nested_tensor=False
        )
        self.projection = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, cats: Tensor, nums: Tensor, role: str) -> Tensor:
        if role not in TOKEN_ROLES:
            raise ValueError(f"Unknown token role: {role}")
        if cats.shape[:-1] != nums.shape[:-1]:
            raise ValueError("Categorical and numerical event dimensions must match")
        if cats.shape[-1] != len(self.categorical) or nums.shape[-1] != len(self.config.numerical_roles):
            raise ValueError("Input feature counts do not match AMBERConfig")
        visible = TOKEN_ROLES[role]
        event_shape = cats.shape[:-1]
        cats = cats.reshape(-1, cats.shape[-1]) if cats.shape[-1] else cats.flatten(0, -2)
        nums = nums.reshape(-1, nums.shape[-1]) if nums.shape[-1] else nums.flatten(0, -2)
        fields = []
        for i, field_role in enumerate(self.config.categorical_roles):
            if field_role in visible:
                fields.append(self.categorical[i](cats[:, i]) + self.feature_identity[i])
        offset = len(self.categorical)
        for i, field_role in enumerate(self.config.numerical_roles):
            if field_role in visible:
                fields.append(
                    nums[:, i, None] * self.numerical_weight[i]
                    + self.numerical_bias[i]
                    + self.feature_identity[offset + i]
                )
        cls = self.cls.expand(cats.shape[0], -1, -1)
        features = torch.cat([torch.stack(fields, dim=1), cls], dim=1) if fields else cls
        summary = self.encoder(features)[:, -1]
        return self.projection(summary).reshape(*event_shape, self.config.d_model)

