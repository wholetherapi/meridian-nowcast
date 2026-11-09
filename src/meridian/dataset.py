"""Dataset readers for the Meridian-Nowcast benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np


@dataclass(frozen=True)
class Window:
    """A single nowcasting example."""

    station_ids: List[str]
    history: np.ndarray  # shape (T, S, C)
    target: np.ndarray  # shape (H, S, C)


class MeridianDataset:
    """Lazy reader over the standardised measurement networks."""

    def __init__(self, root: str | Path, split: str = "test") -> None:
        self.root = Path(root)
        self.split = split
        self.manifest = json.loads((self.root / "manifest.json").read_text())

    def stations(self) -> List[str]:
        return list(self.manifest["stations"])

    def __len__(self) -> int:
        return int(self.manifest["splits"][self.split]["count"])

    def __iter__(self) -> Iterable[Window]:
        for shard in self.manifest["splits"][self.split]["shards"]:
            payload = np.load(self.root / shard)
            for i in range(payload["history"].shape[0]):
                yield Window(
                    station_ids=list(self.manifest["stations"]),
                    history=payload["history"][i],
                    target=payload["future"][i],
                )
