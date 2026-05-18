# src/training/trainer.py 
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch

from src.evaluation.metrics import compute_classification_metrics
from src.training.loops import evaluate_one_epoch, train_one_epoch
from src.training.optimizer_factory import build_optimizer
from src.training.scheduler_factory import build_scheduler
from src.utils.io import write_json


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        train_loader,
        val_loader,
        train_config: Dict[str, Any],
        run_dir: str | Path,
        device: torch.device,
        logger,
    ):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_config = train_config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.device = device
        self.logger = logger

        self.optimizer = build_optimizer(self.model, train_config["optimizer"])
        self.scheduler = build_scheduler(self.optimizer, train_config["scheduler"])

        self.epochs = train_config["training"]["epochs"]
        if self.epochs is None:
            raise ValueError("train_config['training']['epochs'] no puede ser None.")

        self.monitor = train_config["checkpoint"]["monitor"]
        self.mode = train_config["checkpoint"]["mode"]

        if self.mode not in ["min", "max"]:
            raise ValueError("checkpoint.mode debe ser 'min' o 'max'.")

        self.best_metric = float("inf") if self.mode == "min" else float("-inf")
        self.history: List[Dict[str, Any]] = []

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best_metric
        return value > self.best_metric

    def _save_checkpoint(self, checkpoint_name: str, epoch: int, monitored_value: float) -> None:
        checkpoint_path = self.run_dir / checkpoint_name
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "monitored_value": monitored_value,
            },
            checkpoint_path,
        )

    def _call_loss_hook(self, hook_name: str) -> None:
        """
        Permite que pérdidas especiales como EQLv2 ejecuten lógica por época
        sin acoplar el Trainer a una sola implementación.
        """
        hook = getattr(self.criterion, hook_name, None)
        if callable(hook):
            hook()

    def fit(self) -> pd.DataFrame:
        for epoch in range(1, self.epochs + 1):
            # Alineado al notebook/EQLv2: reinicio de gradientes acumulados por época
            self._call_loss_hook("on_epoch_start")

            train_loss, y_train, yhat_train = train_one_epoch(
                model=self.model,
                dataloader=self.train_loader,
                criterion=self.criterion,
                optimizer=self.optimizer,
                device=self.device,
            )

            val_loss, y_val, yhat_val = evaluate_one_epoch(
                model=self.model,
                dataloader=self.val_loader,
                criterion=self.criterion,
                device=self.device,
            )

            train_metrics = compute_classification_metrics(y_train, yhat_train)
            val_metrics = compute_classification_metrics(y_val, yhat_val)

            row = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                **{f"train_{k}": float(v) for k, v in train_metrics.items()},
                **{f"val_{k}": float(v) for k, v in val_metrics.items()},
            }
            self.history.append(row)

            if self.monitor not in row:
                raise KeyError(
                    f"La métrica monitor='{self.monitor}' no existe en history row. "
                    f"Métricas disponibles: {list(row.keys())}"
                )

            monitored_value = row[self.monitor]

            if self._is_improvement(monitored_value):
                self.best_metric = monitored_value
                if self.train_config["checkpoint"].get("save_best", True):
                    self._save_checkpoint("best.ckpt", epoch, monitored_value)

            if self.train_config["checkpoint"].get("save_last", True):
                self._save_checkpoint("last.ckpt", epoch, monitored_value)

            if self.scheduler is not None:
                scheduler_name = self.train_config["scheduler"].get("name")
                if scheduler_name and scheduler_name.lower() == "reducelronplateau":
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Hook de cierre de época para pérdidas tipo EQLv2
            self._call_loss_hook("on_epoch_end")

            self.logger.info(
                "Epoch %d/%d | train_loss=%.6f | val_loss=%.6f | val_f1_macro=%.6f | monitor(%s)=%.6f",
                epoch,
                self.epochs,
                train_loss,
                val_loss,
                val_metrics["f1_macro"],
                self.monitor,
                monitored_value,
            )

        history_df = pd.DataFrame(self.history)
        history_df.to_csv(self.run_dir / "train_history.csv", index=False)

        write_json(
            {
                "best_metric_name": self.monitor,
                "best_metric_value": float(self.best_metric),
                "mode": self.mode,
                "epochs": int(self.epochs),
            },
            self.run_dir / "best_summary.json",
        )

        return history_df