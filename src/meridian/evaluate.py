"""Scoring utilities for Meridian-Nowcast."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .dataset import MeridianDataset, Window
from .model import MeridianEncoder


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - target)))


def evaluate(model_path: str, data_root: str, split: str = "test") -> dict:
    model = MeridianEncoder()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    dataset = MeridianDataset(data_root, split=split)
    errors = []
    with torch.no_grad():
        for window in dataset:
            history = torch.tensor(window.history, dtype=torch.float32).unsqueeze(0)
            target = np.asarray(window.target)
            prediction = model(history).squeeze(0).numpy()
            errors.append(mae(prediction, target))
    return {"split": split, "mae": float(np.mean(errors)), "n": len(errors)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    report = evaluate(str(Path(args.run_dir) / "model.pt"), "data")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
