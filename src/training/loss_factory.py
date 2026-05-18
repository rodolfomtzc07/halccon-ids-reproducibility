# src/training/loss_factory.py
from src.models.loss_eqlv2 import EQLv2Loss

def build_eqlv2_from_trial(trial, num_classes: int):
    gamma = trial.suggest_float("gamma", 8.0, 16.0)
    mu = trial.suggest_float("mu", 0.15, 0.45)
    alpha = trial.suggest_float("alpha", 1.0, 3.0)

    criterion = EQLv2Loss(
        num_classes=num_classes,
        gamma=gamma,
        mu=mu,
        alpha=alpha,
        loss_weight=1.0,
        eps=1e-10,
        vis_grad=False,
    )
    return criterion