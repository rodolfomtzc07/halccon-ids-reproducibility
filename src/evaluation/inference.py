# src/evaluation/inference.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def predict(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[List[int], List[int]]:
    model.eval()

    all_targets: List[int] = []
    all_preds: List[int] = []

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        preds = torch.argmax(logits, dim=1)

        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    return all_targets, all_preds