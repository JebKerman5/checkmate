from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape.")
    if prediction.shape != mask.shape:
        raise ValueError("mask must have the same shape as prediction.")

    mask_f = mask.to(dtype=prediction.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    loss = F.huber_loss(prediction, target, delta=delta, reduction="none")
    return (loss * mask_f).sum() / denom
