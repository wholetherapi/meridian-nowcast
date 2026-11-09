"""Reference spatio-temporal encoder used by Meridian-Nowcast."""

from __future__ import annotations

import torch
from torch import nn


class SpatialMixer(nn.Module):
    """Cheap graph-style mixing across neighbouring stations."""

    def __init__(self, channels: int, neighbours: int = 4) -> None:
        super().__init__()
        self.neighbours = neighbours
        self.proj = nn.Linear(channels * 2, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, stations, channels)
        left = torch.roll(x, shifts=1, dims=2)
        right = torch.roll(x, shifts=-1, dims=2)
        mixed = torch.cat([left, right], dim=-1)
        return x + self.proj(mixed)


class MeridianEncoder(nn.Module):
    def __init__(self, channels: int = 6, hidden: int = 64, horizon: int = 6) -> None:
        super().__init__()
        self.input_proj = nn.Linear(channels, hidden)
        self.mixer = SpatialMixer(hidden)
        self.temporal = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, channels * horizon)
        self.horizon = horizon
        self.channels = channels

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        x = self.mixer(self.input_proj(history))
        batch, time, stations, hidden = x.shape
        x = x.reshape(batch * stations, time, hidden)
        _, state = self.temporal(x)
        out = self.head(state[-1]).reshape(batch, stations, self.horizon, self.channels)
        return out.permute(0, 2, 1, 3)
