import json
from pathlib import Path
import os
import time
import warnings

import psutil
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
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
# CONFIG BASELINE (alineado al notebook)
# ------------------
variant = "label_encoding__ieee_exact"  
use_attention = True
batch_size = 256
num_epochs = 10
learning_rate = 5e-4
loss_name = "eqlv2"          # "crossentropy" o "eqlv2"
num_classes = 13             # fijo según notebook canónico
random_seed = 42

# Hiperparámetros EQLv2 baseline del notebook
eql_gamma = 12.0
eql_mu = 0.3
eql_alpha = 2.0
eql_loss_weight = 1.0
eql_vis_grad = False

base = f"02_data/processed/litnet_10pct/{variant}"
run_name = f"halccon_{'attn' if use_attention else 'noattn'}_{variant}_{loss_name}"
run_dir = Path(f"06_experiments/{run_name}")
run_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------
# REPRODUCIBILIDAD
# ------------------
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Si luego agregas numpy/random, aquí también los fijas


set_seed(random_seed)


# ------------------
# LOGGER SIMPLE
# ------------------
class SimpleLogger:
    def info(self, msg, *args):
        print(msg % args)


logger = SimpleLogger()


# ------------------
# LOAD DATA
# ------------------
X_train = torch.load(f"{base}/X_train.pt")
y_train = torch.load(f"{base}/y_train.pt")

X_val = torch.load(f"{base}/X_val.pt")
y_val = torch.load(f"{base}/y_val.pt")

input_dim = X_train.shape[1]

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# ------------------
# MODEL
# ------------------
if use_attention:
    model = HALCCONMulticlass(input_dim=input_dim, num_classes=num_classes)
else:
    model = HALCCONMulticlassNoAttention(input_dim=input_dim, num_classes=num_classes)

if loss_name == "crossentropy":
    criterion = nn.CrossEntropyLoss()
elif loss_name == "eqlv2":
    criterion = EQLv2Loss(
        num_classes=num_classes,
        gamma=eql_gamma,
        mu=eql_mu,
        alpha=eql_alpha,
        loss_weight=eql_loss_weight,
        vis_grad=eql_vis_grad,
    )
else:
    raise ValueError(f"Loss no soportada: {loss_name}")


# ------------------
# TRAIN CONFIG
# ------------------
train_config = {
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
# TRAIN
# ------------------
trainer = Trainer(
    model=model,
    criterion=criterion,
    train_loader=train_loader,
    val_loader=val_loader,
    train_config=train_config,
    run_dir=run_dir,
    device=device,
    logger=logger,
)

history_df = trainer.fit()


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
# TEST EVALUATION
# ------------------
X_test = torch.load(f"{base}/X_test.pt")
y_test = torch.load(f"{base}/y_test.pt")

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

best_ckpt_path = run_dir / "best.ckpt"
best_model = load_checkpoint(model, best_ckpt_path, device)

test_loss, y_test_true, y_test_pred = evaluate_one_epoch(
    model=best_model,
    dataloader=test_loader,
    criterion=criterion,
    device=device,
)

test_metrics = compute_classification_metrics(y_test_true, y_test_pred)

print("Test metrics:")
print(test_metrics)


# ------------------
# INFERENCE PERFORMANCE
# ------------------
total_time, avg_latency, memory_usage = measure_inference_performance(
    best_model, test_loader, device
)


# ------------------
# SAVE CLASSIFICATION REPORT
# ------------------
report_dict = classification_report(
    y_test_true,
    y_test_pred,
    output_dict=True,
    zero_division=0,
)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(run_dir / "classification_report.csv", index=True)


# ------------------
# SAVE CONFUSION MATRIX
# ------------------
cm = confusion_matrix(y_test_true, y_test_pred)
cm_df = pd.DataFrame(cm)
cm_df.to_csv(run_dir / "confusion_matrix.csv", index=True)


# ------------------
# SAVE PER-CLASS METRICS
# ------------------
per_class_results = compute_per_class_metrics(y_test_true, y_test_pred)
per_class_df = pd.DataFrame(per_class_results)
per_class_df.to_csv(run_dir / "per_class_metrics.csv", index=False)


# ------------------
# SAVE SUMMARY
# ------------------
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
}

if loss_name == "eqlv2" and hasattr(criterion, "get_hparams"):
    results["eqlv2_hparams"] = criterion.get_hparams()

with open(run_dir / "results_summary.json", "w") as f:
    json.dump(results, f, indent=4)

print("Training complete")
print(json.dumps(results, indent=4))