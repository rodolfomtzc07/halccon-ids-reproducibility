# src/training/loops.py
from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import DataLoader


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, List[int], List[int]]:
    model.train()

    running_loss = 0.0
    all_targets: List[int] = []
    all_preds: List[int] = []

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(inputs)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        preds = torch.argmax(logits, dim=1)
        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss, all_targets, all_preds


@torch.no_grad()
def evaluate_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, List[int], List[int]]:
    model.eval()

    running_loss = 0.0
    all_targets: List[int] = []
    all_preds: List[int] = []

    # Guardar estado original del criterio, por si tiene comportamiento dependiente de training/eval
    criterion_was_training = getattr(criterion, "training", False)
    if hasattr(criterion, "eval"):
        criterion.eval()

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = criterion(logits, targets)

        running_loss += loss.item() * inputs.size(0)

        preds = torch.argmax(logits, dim=1)
        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    # Restaurar estado original
    if hasattr(criterion, "train") and criterion_was_training:
        criterion.train()

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss, all_targets, all_preds