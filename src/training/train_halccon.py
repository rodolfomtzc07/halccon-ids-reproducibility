# src/training/train_halccon.py
import json
from pathlib import Path
import os
import time
import warnings
import random
import numpy as np
import optuna
import psutil
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from src.evaluation.inference import load_checkpoint
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_per_class_metrics,
)
from src.models.halccon import HALCCONMulticlass
from src.models.halccon_no_attention import HALCCONMulticlassNoAttention
from src.models.loss_eqlv2 import EQLv2Loss
from src.training.loops import evaluate_one_epoch
from src.training.trainer import Trainer
import optuna.visualization as vis
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=FutureWarning)


# ------------------
# CONFIG BASELINE (alineado al notebook)
# ------------------
dataset_name = "unsw_nb15"
variant = "catboost_label__clean"
base = Path(f"02_data/processed/{dataset_name}/{variant}")
use_attention = True
batch_size = 256
num_epochs = 30
learning_rate = 5e-4
loss_name = "eqlv2"   # Para estudio Optuna debe quedar fijo en eqlv2
random_seed = 42

# baseline canónico EQLv2
baseline_eql_gamma = 12.0
baseline_eql_mu = 0.3
baseline_eql_alpha = 2.0
eql_loss_weight = 1.0
eql_vis_grad = False


study_name = f"halccon_{dataset_name}_eqlv2_{'attn' if use_attention else 'noattn'}_{variant}"
study_dir = Path(f"06_experiments/{study_name}")
study_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RUN_BASELINE = True
RUN_KFOLD = False
RUN_OPTUNA = False

# ------------------
# REPRODUCIBILIDAD
# ------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(random_seed)


# ------------------
# LOGGER SIMPLE
# ------------------
class SimpleLogger:
    def info(self, msg, *args):
        print(msg % args)


logger = SimpleLogger()


# ------------------
# LOAD DATA (UNA SOLA VEZ)
# ------------------
X_train = torch.load(base / "X_train.pt")
y_train = torch.load(base / "y_train.pt")

X_val = torch.load(base / "X_val.pt")
y_val = torch.load(base / "y_val.pt")

X_test = torch.load(base / "X_test.pt")
y_test = torch.load(base / "y_test.pt")

input_dim = X_train.shape[1]
num_classes = int(torch.unique(y_train).numel())

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ------------------
# INFERENCE PERFORMANCE
# ------------------
def measure_inference_performance(model, dataloader, device):
    model.eval()
    model.to(device)

    start_time = time.time()

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(device)
            else:
                inputs = batch.to(device)
            _ = model(inputs)

    end_time = time.time()

    total_time = end_time - start_time
    avg_latency = total_time / len(dataloader)

    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 ** 2)

    return total_time, avg_latency, memory_usage


# ------------------
# MODEL FACTORY
# ------------------
def build_model():
    if use_attention:
        model = HALCCONMulticlass(input_dim=input_dim, num_classes=num_classes)
    else:
        model = HALCCONMulticlassNoAttention(input_dim=input_dim, num_classes=num_classes)
    return model


# ------------------
# LOSS FACTORY
# ------------------
def build_eqlv2(gamma, mu, alpha):
    criterion = EQLv2Loss(
        num_classes=num_classes,
        gamma=gamma,
        mu=mu,
        alpha=alpha,
        loss_weight=eql_loss_weight,
        vis_grad=eql_vis_grad,
    )
    if hasattr(criterion, "reset_all_statistics"):
        criterion.reset_all_statistics()
    return criterion


# ------------------
# TRAIN CONFIG FACTORY
# ------------------
def build_train_config():
    return {
        "optimizer": {
            "name": "adam",
            "lr": learning_rate,
            "weight_decay": 0.0,
        },
        "scheduler": {
            "name": None,
        },
        "training": {
            "epochs": num_epochs,
        },
        "checkpoint": {
            # alineado al notebook: mejor modelo por val_loss
            "monitor": "val_loss",
            "mode": "min",
            "save_best": True,
            "save_last": True,
        },
    }


# ------------------
# SAVE EXTRA ARTIFACTS
# ------------------
def save_trial_artifacts(run_dir, history_df, best_model, criterion):
    test_loss, y_test_true, y_test_pred = evaluate_one_epoch(
        model=best_model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    test_metrics = compute_classification_metrics(y_test_true, y_test_pred)

    total_time, avg_latency, memory_usage = measure_inference_performance(
        best_model, test_loader, device
    )

    report_dict = classification_report(
        y_test_true,
        y_test_pred,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(run_dir / "classification_report.csv", index=True)

    cm = confusion_matrix(y_test_true, y_test_pred)
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(run_dir / "confusion_matrix.csv", index=True)

    per_class_results = compute_per_class_metrics(y_test_true, y_test_pred)
    per_class_df = pd.DataFrame(per_class_results)
    per_class_df.to_csv(run_dir / "per_class_metrics.csv", index=False)

    best_row = history_df.loc[history_df["val_loss"].idxmin()]

    results = {
        "dataset_name": dataset_name,
        "variant": variant,
        "use_attention": use_attention,
        "model": "halccon",
        "loss_name": loss_name,
        "seed": random_seed,
        "num_classes": num_classes,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,

        "best_epoch": int(best_row["epoch"]),
        "best_val_loss": float(best_row["val_loss"]),
        "best_val_f1_macro": float(best_row["val_f1_macro"]),
        "best_val_accuracy": float(best_row["val_accuracy"]),

        "final_val_loss": float(history_df.iloc[-1]["val_loss"]),
        "final_val_f1_macro": float(history_df.iloc[-1]["val_f1_macro"]),
        "final_val_accuracy": float(history_df.iloc[-1]["val_accuracy"]),

        "test_loss": float(test_loss),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_fpr_macro": float(test_metrics["fpr_macro"]),
        "test_precision_macro": float(test_metrics["precision_macro"]),
        "test_recall_macro": float(test_metrics["recall_macro"]),
        "test_f1_macro": float(test_metrics["f1_macro"]),
        "test_precision_weighted": float(test_metrics["precision_weighted"]),
        "test_recall_weighted": float(test_metrics["recall_weighted"]),
        "test_f1_weighted": float(test_metrics["f1_weighted"]),

        "inference_total_time_sec": float(total_time),
        "inference_avg_latency_sec": float(avg_latency),
        "memory_usage_mb": float(memory_usage),
    }

    if hasattr(criterion, "get_hparams"):
        results["eqlv2_hparams"] = criterion.get_hparams()

    with open(run_dir / "results_summary.json", "w") as f:
        json.dump(results, f, indent=4)

    return results


# ------------------
# BASELINE RUN
# ------------------
def run_baseline():
    run_name = f"baseline_halccon_{dataset_name}_{'attn' if use_attention else 'noattn'}_{variant}_eqlv2"
    run_dir = study_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(random_seed)

    model = build_model()
    criterion = build_eqlv2(
        gamma=baseline_eql_gamma,
        mu=baseline_eql_mu,
        alpha=baseline_eql_alpha,
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        train_config=build_train_config(),
        run_dir=run_dir,
        device=device,
        logger=logger,
    )

    history_df = trainer.fit()
    history_df.to_csv(run_dir / "history.csv", index=False)
    
    best_ckpt_path = run_dir / "best.ckpt"
    best_model = load_checkpoint(model, best_ckpt_path, device)

    results = save_trial_artifacts(run_dir, history_df, best_model, criterion)
    print("Baseline complete")
    print(json.dumps(results, indent=4))
    return results

def run_stratified_kfold_validation(n_splits=5):
    kfold_name = f"kfold5_halccon_{'attn' if use_attention else 'noattn'}_{variant}_eqlv2"
    kfold_dir = study_dir / kfold_name
    kfold_dir.mkdir(parents=True, exist_ok=True)

    set_seed(random_seed)

    # Unificar dataset del 10% ya preprocesado
    X_all = torch.cat([X_train, X_val, X_test], dim=0)
    y_all = torch.cat([y_train, y_val, y_test], dim=0)

    # sklearn necesita numpy para stratify
    y_all_np = y_all.cpu().numpy()

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_seed,
    )

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_all_np)), y_all_np), start=1):
        print(f"\n===== Fold {fold_idx}/{n_splits} =====")

        fold_dir = kfold_dir / f"fold_{fold_idx:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        set_seed(random_seed + fold_idx)

        train_idx_t = torch.tensor(train_idx, dtype=torch.long)
        val_idx_t = torch.tensor(val_idx, dtype=torch.long)

        X_train_fold = X_all[train_idx_t]
        y_train_fold = y_all[train_idx_t]

        X_val_fold = X_all[val_idx_t]
        y_val_fold = y_all[val_idx_t]

        train_dataset_fold = TensorDataset(X_train_fold, y_train_fold)
        val_dataset_fold = TensorDataset(X_val_fold, y_val_fold)

        train_loader_fold = DataLoader(
            train_dataset_fold,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader_fold = DataLoader(
            val_dataset_fold,
            batch_size=batch_size,
            shuffle=False
        )

        model = build_model()
        criterion = build_eqlv2(
            gamma=baseline_eql_gamma,
            mu=baseline_eql_mu,
            alpha=baseline_eql_alpha,
        )

        trainer = Trainer(
            model=model,
            criterion=criterion,
            train_loader=train_loader_fold,
            val_loader=val_loader_fold,
            train_config=build_train_config(),
            run_dir=fold_dir,
            device=device,
            logger=logger,
        )

        history_df = trainer.fit()
        history_df.to_csv(fold_dir / "history.csv", index=False)

        best_ckpt_path = fold_dir / "best.ckpt"
        best_model = load_checkpoint(model, best_ckpt_path, device)

        # Evaluar sobre el fold de validación
        fold_loss, y_true, y_pred = evaluate_one_epoch(
            model=best_model,
            dataloader=val_loader_fold,
            criterion=criterion,
            device=device,
        )

        fold_metrics = compute_classification_metrics(y_true, y_pred)

        report_dict = classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(report_dict).transpose().to_csv(
            fold_dir / "classification_report.csv", index=True
        )

        cm = confusion_matrix(y_true, y_pred)
        pd.DataFrame(cm).to_csv(fold_dir / "confusion_matrix.csv", index=True)

        per_class_results = compute_per_class_metrics(y_true, y_pred)
        pd.DataFrame(per_class_results).to_csv(
            fold_dir / "per_class_metrics.csv", index=False
        )

        best_row = history_df.loc[history_df["val_loss"].idxmin()]

        result_row = {
            "fold": fold_idx,
            "best_epoch": int(best_row["epoch"]),
            "best_val_loss": float(best_row["val_loss"]),
            "best_val_f1_macro": float(best_row["val_f1_macro"]),
            "best_val_accuracy": float(best_row["val_accuracy"]),
            "eval_loss": float(fold_loss),
            "eval_accuracy": float(fold_metrics["accuracy"]),
            "eval_fpr_macro": float(fold_metrics["fpr_macro"]),
            "eval_precision_macro": float(fold_metrics["precision_macro"]),
            "eval_recall_macro": float(fold_metrics["recall_macro"]),
            "eval_f1_macro": float(fold_metrics["f1_macro"]),
            "eval_precision_weighted": float(fold_metrics["precision_weighted"]),
            "eval_recall_weighted": float(fold_metrics["recall_weighted"]),
            "eval_f1_weighted": float(fold_metrics["f1_weighted"]),
        }

        with open(fold_dir / "results_summary.json", "w") as f:
            json.dump(result_row, f, indent=4)

        fold_results.append(result_row)

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(kfold_dir / "kfold_results.csv", index=False)

    summary = {
        "n_splits": n_splits,
        "variant": variant,
        "use_attention": use_attention,
        "model": "halccon",
        "loss_name": loss_name,
        "seed": random_seed,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,

        "accuracy_mean": float(results_df["eval_accuracy"].mean()),
        "accuracy_std": float(results_df["eval_accuracy"].std(ddof=1)),
        "fpr_macro_mean": float(results_df["eval_fpr_macro"].mean()),
        "fpr_macro_std": float(results_df["eval_fpr_macro"].std(ddof=1)),
        "precision_macro_mean": float(results_df["eval_precision_macro"].mean()),
        "precision_macro_std": float(results_df["eval_precision_macro"].std(ddof=1)),
        "recall_macro_mean": float(results_df["eval_recall_macro"].mean()),
        "recall_macro_std": float(results_df["eval_recall_macro"].std(ddof=1)),
        "f1_macro_mean": float(results_df["eval_f1_macro"].mean()),
        "f1_macro_std": float(results_df["eval_f1_macro"].std(ddof=1)),
        "precision_weighted_mean": float(results_df["eval_precision_weighted"].mean()),
        "precision_weighted_std": float(results_df["eval_precision_weighted"].std(ddof=1)),
        "recall_weighted_mean": float(results_df["eval_recall_weighted"].mean()),
        "recall_weighted_std": float(results_df["eval_recall_weighted"].std(ddof=1)),
        "f1_weighted_mean": float(results_df["eval_f1_weighted"].mean()),
        "f1_weighted_std": float(results_df["eval_f1_weighted"].std(ddof=1)),
    }

    with open(kfold_dir / "kfold_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\nK-fold complete")
    print(json.dumps(summary, indent=4))

    return results_df, summary

# ------------------
# OPTUNA OBJECTIVE
# ------------------
def objective(trial):
    set_seed(random_seed)

    gamma = trial.suggest_float("gamma", 8.0, 16.0)
    mu = trial.suggest_float("mu", 0.15, 0.45)
    alpha = trial.suggest_float("alpha", 1.0, 3.0)

    trial_name = f"trial_{trial.number:03d}"
    run_dir = study_dir / trial_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_model()
    criterion = build_eqlv2(gamma=gamma, mu=mu, alpha=alpha)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        train_config=build_train_config(),
        run_dir=run_dir,
        device=device,
        logger=logger,
    )

    history_df = trainer.fit()

    # criterio canónico de selección: mejor val_loss
    best_row = history_df.loc[history_df["val_loss"].idxmin()]
    objective_value = float(best_row["val_f1_macro"])

    best_ckpt_path = run_dir / "best.ckpt"
    best_model = load_checkpoint(model, best_ckpt_path, device)

    results = save_trial_artifacts(run_dir, history_df, best_model, criterion)

    trial.set_user_attr("run_dir", str(run_dir))
    trial.set_user_attr("best_epoch", int(results["best_epoch"]))
    trial.set_user_attr("best_val_loss", float(results["best_val_loss"]))
    trial.set_user_attr("best_val_f1_macro", float(results["best_val_f1_macro"]))
    trial.set_user_attr("best_val_accuracy", float(results["best_val_accuracy"]))
    trial.set_user_attr("test_loss", float(results["test_loss"]))
    trial.set_user_attr("test_accuracy", float(results["test_accuracy"]))
    trial.set_user_attr("test_fpr_macro", float(results["test_fpr_macro"]))
    trial.set_user_attr("test_precision_macro", float(results["test_precision_macro"]))
    trial.set_user_attr("test_recall_macro", float(results["test_recall_macro"]))
    trial.set_user_attr("test_f1_macro", float(results["test_f1_macro"]))
    trial.set_user_attr("eqlv2_hparams", results.get("eqlv2_hparams", {}))

    history_df.to_csv(run_dir / "history.csv", index=False)

    return objective_value


# ------------------
# EXPORT STUDY RESULTS
# ------------------
def export_study_results(study):
    rows = []

    for t in study.trials:
        row = {
            "trial_number": t.number,
            "state": str(t.state),
            "objective_value": t.value,
        }

        for k, v in t.params.items():
            row[k] = v

        for k, v in t.user_attrs.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                row[k] = v

        rows.append(row)

    trials_df = pd.DataFrame(rows)
    trials_df.to_csv(study_dir / "optuna_trials_summary.csv", index=False)

    best_payload = {
        "best_trial_number": study.best_trial.number,
        "best_value": study.best_trial.value,
        "best_params": study.best_trial.params,
        "best_user_attrs": study.best_trial.user_attrs,
    }

    with open(study_dir / "optuna_best_trial.json", "w") as f:
        json.dump(best_payload, f, indent=4)

    print("Study exported")

def save_optuna_optimization_history(study, output_dir: Path) -> None:
    fig = vis.plot_optimization_history(study)
    fig.write_html(str(output_dir / "optuna_optimization_history.html"))

# ------------------
# MAIN
# ------------------
if __name__ == "__main__":
    if RUN_BASELINE:
        run_baseline()

    if RUN_KFOLD:
        run_stratified_kfold_validation(n_splits=5)

    if RUN_OPTUNA:
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=TPESampler(seed=random_seed),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=3),
        )

        study.optimize(objective, n_trials=50)

        export_study_results(study)
        save_optuna_optimization_history(study, study_dir)

        print("Best trial:")
        print("  number:", study.best_trial.number)
        print("  value:", study.best_trial.value)
        print("  params:", study.best_trial.params)