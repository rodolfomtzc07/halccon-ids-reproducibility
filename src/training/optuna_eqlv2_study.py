# src/training/optuna_eqlv2_study.py
import os
import json
import copy
import random
import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
    )
from pathlib import Path
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from src.models.loss_eqlv2 import EQLv2Loss
# from src.models.halccon import HALCCONModel
# from src.data.loaders import get_dataloaders
# from src.training.metrics import evaluate_model


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_eqlv2_trial_criterion(trial, num_classes: int) -> EQLv2Loss:
    gamma = trial.suggest_float("gamma", 8.0, 16.0)
    mu = trial.suggest_float("mu", 0.15, 0.45)
    alpha = trial.suggest_float("alpha", 1.0, 3.0)

    return EQLv2Loss(
        num_classes=num_classes,
        gamma=gamma,
        mu=mu,
        alpha=alpha,
        loss_weight=1.0,
        eps=1e-10,
        vis_grad=False,
    )


def train_one_epoch(model, dataloader, optimizer, criterion, device) -> float:
    model.train()
    criterion.train()

    if hasattr(criterion, "on_epoch_start"):
        criterion.on_epoch_start()

    running_loss = 0.0
    n_samples = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

        batch_size = y.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

        loss.backward()
        optimizer.step()

    if hasattr(criterion, "on_epoch_end"):
        criterion.on_epoch_end()

    return running_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate_one_epoch(model, dataloader, criterion, device) -> dict:
    model.eval()
    criterion.eval()

    running_loss = 0.0
    n_samples = 0

    all_preds = []
    all_targets = []

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        batch_size = y.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()

    metrics = compute_metrics_multiclass(y_true, y_pred)
    metrics["loss"] = running_loss / max(n_samples, 1)
    return metrics


def compute_metrics_multiclass(y_true, y_pred) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]

    fpr_per_class = []
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fpr_per_class.append(fpr)

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "fpr_macro": float(np.mean(fpr_per_class)),
        "confusion_matrix": cm.tolist(),
    }

