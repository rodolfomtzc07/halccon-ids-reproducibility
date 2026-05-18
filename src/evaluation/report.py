# src/evaluation/report.py 
# src/evaluation/report.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.evaluation.confusion import compute_confusion_matrix, save_confusion_matrix_csv
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_notebook_metrics,
    compute_per_class_metrics,
)
from src.utils.io import write_json


def save_evaluation_report(
    y_true: List[int],
    y_pred: List[int],
    output_dir: str | Path,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Extended framework metrics
    # -------------------------------------------------------------
    global_metrics = compute_classification_metrics(y_true, y_pred)

    # -------------------------------------------------------------
    # 2. Canonical notebook-equivalent metrics
    # -------------------------------------------------------------
    notebook_metrics = compute_notebook_metrics(y_true, y_pred)

    # -------------------------------------------------------------
    # 3. Per-class metrics
    # -------------------------------------------------------------
    per_class_metrics = compute_per_class_metrics(y_true, y_pred)

    # -------------------------------------------------------------
    # 4. Confusion matrix
    # -------------------------------------------------------------
    confusion_df = compute_confusion_matrix(y_true, y_pred)

    # -------------------------------------------------------------
    # 5. Save JSON reports
    # -------------------------------------------------------------
    write_json(global_metrics, output_dir / "test_metrics_extended.json")
    write_json(notebook_metrics, output_dir / "test_metrics_notebook.json")

    # Archivo legacy si quieres mantener compatibilidad:
    # aquí recomiendo que "test_metrics.json" sea el canónico
    write_json(notebook_metrics, output_dir / "test_metrics.json")

    # -------------------------------------------------------------
    # 6. Save per-class metrics
    # -------------------------------------------------------------
    per_class_df = pd.DataFrame(per_class_metrics)
    per_class_df.to_csv(output_dir / "per_class_metrics.csv", index=False)

    # -------------------------------------------------------------
    # 7. Save confusion matrix
    # -------------------------------------------------------------
    save_confusion_matrix_csv(confusion_df, output_dir / "confusion_matrix.csv")

    # -------------------------------------------------------------
    # 8. Save predictions
    # -------------------------------------------------------------
    predictions_df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)

    # -------------------------------------------------------------
    # 9. Save notebook FPR per class as separate CSV
    # -------------------------------------------------------------
    fpr_per_class_df = pd.DataFrame(
        {
            "class_id": list(range(len(notebook_metrics["target_names"]))),
            "class_name": notebook_metrics["target_names"],
            "fpr": [
                notebook_metrics["fpr_per_class"][class_name]
                for class_name in notebook_metrics["target_names"]
            ],
            "class_weight": [
                notebook_metrics["class_weights"][class_name]
                for class_name in notebook_metrics["target_names"]
            ],
            "support": [
                notebook_metrics["class_distribution"][class_name]
                for class_name in notebook_metrics["target_names"]
            ],
        }
    )
    fpr_per_class_df.to_csv(output_dir / "fpr_per_class_notebook.csv", index=False)

    # -------------------------------------------------------------
    # 10. Return both metric families
    # -------------------------------------------------------------
    return {
        "extended_metrics": global_metrics,
        "notebook_metrics": notebook_metrics,
    }