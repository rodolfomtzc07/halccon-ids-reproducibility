# src/training/scheduler_factory.py 
import torch


def build_scheduler(optimizer: torch.optim.Optimizer, scheduler_config: dict):
    if not scheduler_config.get("enabled", False):
        return None

    scheduler_name = scheduler_config["name"]
    params = scheduler_config.get("params", {})

    if scheduler_name is None:
        return None

    scheduler_name = scheduler_name.lower()

    if scheduler_name == "steplr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params["step_size"],
            gamma=params["gamma"],
        )

    if scheduler_name == "reducelronplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=params.get("mode", "min"),
            factor=params.get("factor", 0.1),
            patience=params.get("patience", 10),
            min_lr=params.get("min_lr", 0.0),
        )

    raise ValueError(f"Scheduler no soportado: {scheduler_name}")