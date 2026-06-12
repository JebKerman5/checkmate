from __future__ import annotations

from copy import deepcopy

import torch


def clone_ema_model(model: torch.nn.Module) -> torch.nn.Module:
    ema = deepcopy(model)
    ema.eval()
    for param in ema.parameters():
        param.requires_grad_(False)
    return ema


@torch.no_grad()
def update_ema_model(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    if not 0.0 <= decay <= 1.0:
        raise ValueError("decay must be in [0, 1].")

    model_state = model.state_dict()
    ema_state = ema.state_dict()
    for name, ema_value in ema_state.items():
        source = model_state[name].detach()
        if torch.is_floating_point(ema_value):
            ema_value.mul_(decay).add_(source, alpha=1.0 - decay)
        else:
            ema_value.copy_(source)
