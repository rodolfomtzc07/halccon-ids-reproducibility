# src/training/optimizer_factory.py 
import torch


def build_optimizer(model: torch.nn.Module, optimizer_config: dict) -> torch.optim.Optimizer:
    optimizer_name = optimizer_config["name"].lower()
    lr = optimizer_config["lr"]
    weight_decay = optimizer_config.get("weight_decay", 0.0)

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Optimizador no soportado: {optimizer_name}")