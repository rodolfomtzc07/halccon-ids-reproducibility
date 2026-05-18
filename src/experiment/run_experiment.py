#04_src/experiment/run_experiment.py Este será el orquestador mínimo del experimento: cargar configs, fijar semilla, preparar directorio de corrida y guardar metadatos base.
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch

from src.data.loaders import build_dataloaders_from_variant
from src.evaluation.inference import load_checkpoint, predict
from src.evaluation.report import save_evaluation_report
from src.models.factory import build_loss, build_model
from src.training.seed import set_seed
from src.training.trainer import Trainer
from src.utils.git_utils import get_git_info
from src.utils.io import read_yaml, write_json, write_yaml
from src.utils.logging_utils import get_logger
from src.utils.paths import PROJECT_ROOT, get_run_dir


def load_experiment_config(experiment_config_path: str | Path) -> Dict[str, Any]:
    experiment_config_path = Path(experiment_config_path)
    exp_cfg = read_yaml(experiment_config_path)

    config_refs = exp_cfg["config_refs"]

    data_cfg = read_yaml(PROJECT_ROOT / config_refs["data"])
    model_cfg = read_yaml(PROJECT_ROOT / config_refs["model"])
    train_cfg = read_yaml(PROJECT_ROOT / config_refs["train"])

    return {
        "experiment": exp_cfg,
        "data": data_cfg,
        "model": model_cfg,
        "train": train_cfg,
    }


def prepare_run_directory(configs: Dict[str, Any]) -> Path:
    exp_name = configs["experiment"]["experiment_name"]
    variant_name = configs["experiment"].get("preprocessing_variant", "default")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{exp_name}_{variant_name}_{timestamp}"
    return get_run_dir(run_name)


def save_resolved_configs(configs: Dict[str, Any], run_dir: Path) -> None:
    write_yaml(configs["experiment"], run_dir / "experiment_config.yaml")
    write_yaml(configs["data"], run_dir / "data_config.yaml")
    write_yaml(configs["model"], run_dir / "model_config.yaml")
    write_yaml(configs["train"], run_dir / "train_config.yaml")


def build_run_metadata(configs: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    exp_cfg = configs["experiment"]
    train_cfg = configs["train"]

    return {
        "experiment_name": exp_cfg["experiment_name"],
        "preprocessing_variant": exp_cfg.get("preprocessing_variant"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "task": exp_cfg.get("task"),
        "dataset": exp_cfg.get("dataset"),
        "seed": train_cfg["reproducibility"]["seed"],
        "git": get_git_info(PROJECT_ROOT),
    }


def resolve_device(train_cfg: Dict[str, Any]) -> torch.device:
    requested_device = train_cfg["runtime"]["device"]

    if requested_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def run_experiment(experiment_config_path: str | Path) -> Path:
    configs = load_experiment_config(experiment_config_path)
    run_dir = prepare_run_directory(configs)

    logger = get_logger("halccon.experiment", run_dir / "run.log")
    logger.info("Initializing experiment...")

    reproducibility_cfg = configs["train"]["reproducibility"]
    set_seed(
        seed=reproducibility_cfg["seed"],
        deterministic=reproducibility_cfg["deterministic"],
        benchmark=reproducibility_cfg["benchmark"],
    )

    save_resolved_configs(configs, run_dir)
    metadata = build_run_metadata(configs, run_dir)
    write_json(metadata, run_dir / "metadata.json")

    logger.info("Run directory ready: %s", run_dir)

    processed_dir = PROJECT_ROOT / configs["data"]["paths"]["processed_dir"]
    variant_name = configs["experiment"].get("preprocessing_variant", "catboost_label")
    variant_dir = processed_dir / variant_name

    if not variant_dir.exists():
        raise FileNotFoundError(f"No se encontró el directorio de variante preprocesada: {variant_dir}")

    train_loader, val_loader, test_loader = build_dataloaders_from_variant(
        processed_dir=processed_dir,
        variant_name=variant_name,
        train_config=configs["train"],
    )

    sample_inputs, _ = next(iter(train_loader))
    input_dim = sample_inputs.shape[-1]

    configs["model"]["architecture"]["input_dim"] = input_dim

    num_classes = configs["model"]["architecture"]["num_classes"]
    if num_classes is None:
        raise ValueError("model_config['architecture']['num_classes'] no puede ser None.")

    write_yaml(configs["model"], run_dir / "model_config.yaml")

    model = build_model(configs["model"])
    criterion = build_loss(configs["model"]["loss"], num_classes=num_classes)
    device = resolve_device(configs["train"])

    logger.info("Device selected: %s", device)
    logger.info("Preprocessing variant: %s", variant_name)
    logger.info("Input dim resolved from dataloader: %d", input_dim)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        train_config=configs["train"],
        run_dir=run_dir,
        device=device,
        logger=logger,
    )

    logger.info("Starting training...")
    trainer.fit()
    logger.info("Training finished.")

    best_checkpoint_path = run_dir / "best.ckpt"
    if not best_checkpoint_path.exists():
        raise FileNotFoundError(f"No se encontró el checkpoint best.ckpt en {best_checkpoint_path}")

    model = load_checkpoint(model, best_checkpoint_path, device)
    y_true, y_pred = predict(model, test_loader, device)

    test_metrics = save_evaluation_report(
        y_true=y_true,
        y_pred=y_pred,
        output_dir=run_dir,
    )

    logger.info("Test evaluation finished.")
    logger.info("Test metrics: %s", test_metrics)

    return run_dir


if __name__ == "__main__":
    experiment_path = PROJECT_ROOT / "01_configs" / "experiment" / "exp_litnet_multiclass.yaml"
    run_directory = run_experiment(experiment_path)
    print(f"Run directory created at: {run_directory}")