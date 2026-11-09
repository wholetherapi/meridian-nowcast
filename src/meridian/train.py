"""Training entry point for the Meridian-Nowcast reference model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import MeridianDataset, Window
from .model import MeridianEncoder


def collate(windows):
    history = torch.tensor([w.history for w in windows], dtype=torch.float32)
    target = torch.tensor([w.target for w in windows], dtype=torch.float32)
    return history, target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    dataset = MeridianDataset(cfg["data_root"], split="train")
    loader = DataLoader(
        list(dataset), batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate
    )

    model = MeridianEncoder(
        channels=cfg["channels"], hidden=cfg["hidden"], horizon=cfg["horizon"]
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    loss_fn = torch.nn.L1Loss()

    for epoch in range(cfg["epochs"]):
        running = 0.0
        for history, target in loader:
            optimiser.zero_grad()
            prediction = model(history)
            loss = loss_fn(prediction, target)
            loss.backward()
            optimiser.step()
            running += float(loss)
        print(f"epoch {epoch + 1}: loss={running / max(1, len(loader)):.4f}")

    out = Path(cfg["output_dir"]) / "model.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)


if __name__ == "__main__":
    main()
