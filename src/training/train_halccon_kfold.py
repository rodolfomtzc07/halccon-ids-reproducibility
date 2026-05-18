# src/training/train_halccon_kfold.py
import json
from pathlib import Path
import os
import time
import warnings
import random
import numpy as np
import psutil
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

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

warnings.filterwarnings("ignore", category=FutureWarning)


# ------------------
# CONFIG K-FOLD VALIDATION
# ------------------
variant = "catboost_label__ieee_exact"
use_attention = True
batch_size = 256
num_epochs = 10
learning_rate = 5e-4
loss_name = "eqlv2"
num_classes = 13
random_seed = 42

# baseline canónico EQLv2
baseline_eql_gamma = 12.0
baseline_eql_mu = 0.3
baseline_eql_alpha = 2.0
eql_loss_weight = 1.0
eql_vis_grad = False

base = f"02_data/processed/litnet_10pct/{variant}"
study_name = f"halccon_kfold_{'attn' if use_attention else 'noattn'}_{variant}_{loss_name}"
study_dir = Path(f"06_experiments/{study_name}")
study_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RUN_BASELINE = False
RUN_KFOLD = True


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
X_train = torch.load(f"{base}/X_train.pt")
y_train = torch.load(f"{base}/y_train.pt")

X_val = torch.load(f"{base}/X_val.pt")
y_val = torch.load(f"{base}/y_val.pt")

X_test = torch.load(f"{base}/X_test.pt")
y_test = torch.load(f"{base}/y_test.pt")

input_dim = X_train.shape[1]

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
        "eqlv2_gamma": baseline_eql_gamma,
        "eqlv2_mu": baseline_eql_mu,
        "eqlv2_alpha": baseline_eql_alpha,
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
    run_name = f"baseline_halccon_{'attn' if use_attention else 'noattn'}_{variant}_{loss_name}"
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
    best_model = build_model()
    best_model = load_checkpoint(best_model, best_ckpt_path, device)

    results = save_trial_artifacts(run_dir, history_df, best_model, criterion)
    print("Baseline complete")
    print(json.dumps(results, indent=4))
    return results


# ------------------
# STRATIFIED K-FOLD VALIDATION
# ------------------
def run_stratified_kfold_validation(n_splits=5):
    kfold_name = f"kfold{n_splits}_halccon_{'attn' if use_attention else 'noattn'}_{variant}_{loss_name}"
    kfold_dir = study_dir / kfold_name
    kfold_dir.mkdir(parents=True, exist_ok=True)

    set_seed(random_seed)

    # Validación adicional sobre el dataset completo ya preprocesado
    X_all = torch.cat([X_train, X_val, X_test], dim=0)
    y_all = torch.cat([y_train, y_val, y_test], dim=0)

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
            shuffle=True,
        )
        val_loader_fold = DataLoader(
            val_dataset_fold,
            batch_size=batch_size,
            shuffle=False,
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
        best_model = build_model()
        best_model = load_checkpoint(best_model, best_ckpt_path, device)

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
            fold_dir / "classification_report.csv",
            index=True,
        )

        cm = confusion_matrix(y_true, y_pred)
        pd.DataFrame(cm).to_csv(
            fold_dir / "confusion_matrix.csv",
            index=True,
        )

        per_class_results = compute_per_class_metrics(y_true, y_pred)
        pd.DataFrame(per_class_results).to_csv(
            fold_dir / "per_class_metrics.csv",
            index=False,
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
            "eqlv2_gamma": baseline_eql_gamma,
            "eqlv2_mu": baseline_eql_mu,
            "eqlv2_alpha": baseline_eql_alpha,
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
        "eqlv2_gamma": baseline_eql_gamma,
        "eqlv2_mu": baseline_eql_mu,
        "eqlv2_alpha": baseline_eql_alpha,
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
# MAIN
# ------------------
if __name__ == "__main__":
    if RUN_BASELINE:
        run_baseline()

    if RUN_KFOLD:
        run_stratified_kfold_validation(n_splits=5)