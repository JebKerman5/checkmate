from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from lattice.config import ModelConfig


@dataclass(frozen=True)
class LatticeOutput:
    from_features: torch.Tensor
    to_features: torch.Tensor
    promo_bias: torch.Tensor
    wdl_logits: torch.Tensor
    value: torch.Tensor
    moves_left: torch.Tensor


def _line_indices(anti: bool) -> tuple[torch.Tensor, torch.Tensor]:
    lines: list[list[int]] = []
    masks: list[list[bool]] = []
    for line_id in range(15):
        squares: list[int] = []
        for rank in range(8):
            for file in range(8):
                key = rank + file if anti else rank - file + 7
                if key == line_id:
                    squares.append(rank * 8 + file)

        mask = [True] * len(squares)
        while len(squares) < 8:
            squares.append(64)
            mask.append(False)

        lines.append(squares)
        masks.append(mask)

    return torch.tensor(lines, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class AxisMix(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 4 != 0:
            raise ValueError("AxisMix requires d_model divisible by 4.")

        group = dim // 4
        self.group = group
        self.file_weight = nn.Parameter(torch.eye(8))
        self.rank_weight = nn.Parameter(torch.eye(8))
        self.diag_weight = nn.Parameter(torch.eye(8))
        self.anti_diag_weight = nn.Parameter(torch.eye(8))
        self.out = nn.Linear(dim, dim, bias=False)

        diag_indices, diag_mask = _line_indices(anti=False)
        anti_indices, anti_mask = _line_indices(anti=True)
        self.register_buffer("diag_indices", diag_indices, persistent=False)
        self.register_buffer("diag_mask", diag_mask, persistent=False)
        self.register_buffer("anti_indices", anti_indices, persistent=False)
        self.register_buffer("anti_mask", anti_mask, persistent=False)

    def _mix_indexed_lines(
        self,
        x: torch.Tensor,
        indices: torch.Tensor,
        mask: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, channels = x.shape
        pad = x.new_zeros(batch, 1, channels)
        x_pad = torch.cat([x, pad], dim=1)

        flat_indices = indices.reshape(-1)
        lines = x_pad[:, flat_indices].reshape(batch, 15, 8, channels)
        mixed = torch.einsum("blic,ij->bljc", lines, weight)
        mixed = mixed * mask.view(1, 15, 8, 1).to(dtype=mixed.dtype)

        out = x.new_zeros(batch, 65, channels)
        out.index_add_(1, flat_indices, mixed.reshape(batch, -1, channels))
        return out[:, :64]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        file_x, rank_x, diag_x, anti_x = x.split(self.group, dim=-1)

        file_board = file_x.reshape(x.shape[0], 8, 8, self.group)
        file_mixed = torch.einsum("brfc,rs->bsfc", file_board, self.file_weight)
        file_mixed = file_mixed.reshape_as(file_x)

        rank_board = rank_x.reshape(x.shape[0], 8, 8, self.group)
        rank_mixed = torch.einsum("brfc,fs->brsc", rank_board, self.rank_weight)
        rank_mixed = rank_mixed.reshape_as(rank_x)

        diag_mixed = self._mix_indexed_lines(
            diag_x, self.diag_indices, self.diag_mask, self.diag_weight
        )
        anti_mixed = self._mix_indexed_lines(
            anti_x, self.anti_indices, self.anti_mask, self.anti_diag_weight
        )

        return self.out(torch.cat([file_mixed, rank_mixed, diag_mixed, anti_mixed], dim=-1))


class LatticeBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = cfg.mlp_multiplier * cfg.d_model
        self.axis_norm = RMSNorm(cfg.d_model)
        self.axis = AxisMix(cfg.d_model)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.axis(self.axis_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class LatticeModel(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.encoder = nn.Linear(self.cfg.input_features, self.cfg.d_model, bias=False)
        self.blocks = nn.ModuleList([LatticeBlock(self.cfg) for _ in range(self.cfg.blocks)])
        self.final_norm = RMSNorm(self.cfg.d_model)

        self.from_head = nn.Linear(self.cfg.d_model, self.cfg.q_rank, bias=False)
        self.to_head = nn.Linear(self.cfg.d_model, self.cfg.q_rank, bias=False)
        self.promo_bias = nn.Parameter(torch.zeros(5))
        self.wdl_head = nn.Linear(2 * self.cfg.d_model, 3)
        self.value_head = nn.Linear(2 * self.cfg.d_model, 1)
        self.moves_left_head = nn.Linear(2 * self.cfg.d_model, 1)

    def forward(self, features: torch.Tensor) -> LatticeOutput:
        if features.ndim != 3:
            raise ValueError("Expected features with shape [batch, 64, channels].")
        if features.shape[1] != self.cfg.squares:
            raise ValueError(f"Expected {self.cfg.squares} squares, got {features.shape[1]}.")
        if features.shape[2] != self.cfg.input_features:
            raise ValueError(
                f"Expected {self.cfg.input_features} input features, got {features.shape[2]}."
            )

        x = self.encoder(features)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)

        pooled = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)
        return LatticeOutput(
            from_features=self.from_head(x),
            to_features=self.to_head(x),
            promo_bias=self.promo_bias,
            wdl_logits=self.wdl_head(pooled),
            value=torch.tanh(self.value_head(pooled)).squeeze(-1),
            moves_left=F.softplus(self.moves_left_head(pooled)).squeeze(-1),
        )


def score_moves(
    output: LatticeOutput,
    from_squares: torch.Tensor,
    to_squares: torch.Tensor,
    promotion: torch.Tensor | None = None,
) -> torch.Tensor:
    if from_squares.shape != to_squares.shape:
        raise ValueError("from_squares and to_squares must have the same shape.")

    batch = output.from_features.shape[0]
    if from_squares.shape[0] != batch:
        raise ValueError("Move tensors must use the same batch size as model output.")

    batch_index = torch.arange(batch, device=from_squares.device).unsqueeze(-1)
    from_vec = output.from_features[batch_index, from_squares]
    to_vec = output.to_features[batch_index, to_squares]
    scores = (from_vec * to_vec).sum(dim=-1)

    if promotion is not None:
        promo = promotion.clamp(min=0, max=4)
        scores = scores + output.promo_bias[promo] * (promotion >= 0).to(scores.dtype)

    return scores
