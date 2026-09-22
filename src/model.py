"""SBD-INR network definition used by the released Kodak checkpoint."""

from __future__ import annotations

import math

import torch
from torch import nn


class SineLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(inputs))


class BitConditionFiLM(nn.Module):
    """Convert shared spatial features into plane-conditioned features."""

    def __init__(
        self,
        num_bits: int,
        hidden_features: int,
        bit_index_std: float = 0.12,
    ) -> None:
        super().__init__()
        self.num_bits = num_bits
        self.hidden_features = hidden_features
        self.scale = hidden_features**0.5

        self.bit_index_embed = nn.Embedding(num_bits, hidden_features)
        nn.init.normal_(self.bit_index_embed.weight, mean=0.0, std=bit_index_std)

        self.cond_proj = nn.Sequential(
            nn.LayerNorm(hidden_features),
            nn.Linear(hidden_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, hidden_features),
        )
        self.film = nn.Sequential(
            nn.LayerNorm(hidden_features),
            nn.Linear(hidden_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, hidden_features * 2),
        )
        nn.init.normal_(self.film[-1].weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.film[-1].bias, mean=0.0, std=1e-2)

    def forward(self, shared_feature: torch.Tensor) -> torch.Tensor:
        if shared_feature.ndim == 2:
            batch_size, hidden_features = shared_feature.shape
            base_feature = shared_feature.unsqueeze(1)
        elif shared_feature.ndim == 3:
            batch_size, num_bits, hidden_features = shared_feature.shape
            if num_bits != self.num_bits:
                raise ValueError(
                    f"Expected {self.num_bits} bit planes, received {num_bits}."
                )
            base_feature = shared_feature
        else:
            raise ValueError("Expected [N,D] or [N,B,D] input features.")

        if hidden_features != self.hidden_features:
            raise ValueError(
                f"Expected hidden size {self.hidden_features}, received "
                f"{hidden_features}."
            )

        bit_ids = torch.arange(self.num_bits, device=shared_feature.device)
        cond = self.bit_index_embed(bit_ids).unsqueeze(0)
        cond = cond + self.cond_proj(cond)
        gamma, beta = torch.chunk(self.film(cond), chunks=2, dim=-1)
        features = base_feature * (1.0 + gamma / self.scale) + beta / self.scale
        return features + cond.expand(batch_size, -1, -1)


class BitPlaneSiren(nn.Module):
    """2D decoupled SIREN backbone with the FiLM-PBD reconstruction module."""

    def __init__(
        self,
        hidden_features: int = 512,
        hidden_layers: int = 5,
        out_features: int = 3,
        num_bits: int = 8,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        bit_index_std: float = 0.12,
        use_db: bool = True,
        use_film_pbd: bool = True,
    ) -> None:
        super().__init__()
        self.num_bits = num_bits
        self.use_db = use_db
        self.use_film_pbd = use_film_pbd
        self.out_features = out_features
        self.hidden_features = hidden_features

        if use_db and not use_film_pbd:
            raise ValueError("DB requires the FiLM-PBD reconstruction module.")

        in_features = 2 if use_db else 3
        layers: list[nn.Module] = [
            SineLayer(
                in_features,
                hidden_features,
                is_first=True,
                omega_0=first_omega_0,
            )
        ]
        for _ in range(hidden_layers):
            layers.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )
        self.feature_extractor = nn.Sequential(*layers)

        if use_film_pbd:
            self.bit_condition = BitConditionFiLM(
                num_bits=num_bits,
                hidden_features=hidden_features,
                bit_index_std=bit_index_std,
            )
            self.decoder_weight = nn.Parameter(
                torch.empty(num_bits, hidden_features, out_features)
            )
            self.decoder_bias = nn.Parameter(torch.zeros(num_bits, out_features))
            with torch.no_grad():
                bound = math.sqrt(6.0 / hidden_features) / hidden_omega_0
                gain = bound * math.sqrt(hidden_features / out_features)
                for bit in range(num_bits):
                    weight = torch.empty(out_features, hidden_features)
                    nn.init.orthogonal_(weight)
                    self.decoder_weight[bit].copy_(weight.t() * gain)
        else:
            self.final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                bound = math.sqrt(6.0 / hidden_features) / hidden_omega_0
                self.final_linear.weight.uniform_(-bound, bound)
                self.final_linear.bias.zero_()

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        batch_size, num_bits, coord_dim = coords.shape

        if not self.use_db:
            flat_features = self.feature_extractor(coords.reshape(-1, coord_dim))
            if self.use_film_pbd:
                backbone_features = flat_features.reshape(
                    batch_size, num_bits, self.hidden_features
                )
                features = self.bit_condition(backbone_features)
                logits = torch.einsum(
                    "nbd,bdo->nbo", features, self.decoder_weight
                )
                return logits + self.decoder_bias.unsqueeze(0)
            logits = self.final_linear(flat_features)
            return logits.reshape(batch_size, num_bits, self.out_features)

        shared_feature = self.feature_extractor(coords[:, 0, :2])
        features = self.bit_condition(shared_feature)
        logits = torch.einsum("nbd,bdo->nbo", features, self.decoder_weight)
        return logits + self.decoder_bias.unsqueeze(0)

