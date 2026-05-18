# src/evaluation/metrics.py 
# src/evaluation/metrics.py
from __future__ import annotations

from typing import Dict, List, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------
# Canonical class definition from the notebook
# ---------------------------------------------------------------------
TARGET_NAMES = [
    "http_f",
    "icmp_f",
    "icmp_smf",
    "Normal",
    "smtp_b",
    "tcp_land",
    "tcp_red_w",
    "tcp_syn_f",
    "tcp_udp_win_p",
    "tcp_w32_w",
    "udp_0",
    "udp_f",
    "udp_reaper_w",
]

CLASS_IDS = list(range(len(TARGET_NAMES)))


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _to_numpy(
    y_true: List[int] | np.ndarray,
    y_pred: List[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true and y_pred must have the same length. "
            f"Got {y_true.shape[0]} and {y_pred.shape[0]}."
        )

    return y_true, y_pred


def _compute_confusion_components(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    tn = cm.sum() - (fp + fn + tp)
    return tp, fp, fn, tn


def _safe_divide(numerator: np.ndarray | float, denominator: np.ndarray | float) -> np.ndarray | float:
    numerator_arr = np.asarray(numerator, dtype=float)
    denominator_arr = np.asarray(denominator, dtype=float)

    out = np.zeros_like(numerator_arr, dtype=float)
    mask = denominator_arr > 0
    out[mask] = numerator_arr[mask] / denominator_arr[mask]

    if np.isscalar(numerator) and np.isscalar(denominator):
        return float(out.item())
    return out


# ---------------------------------------------------------------------
# Current extended metrics (your current framework style)
# ---------------------------------------------------------------------
def compute_classification_metrics(
    y_true: List[int] | np.ndarray,
    y_pred: List[int] | np.ndarray,
) -> Dict[str, float]:
    """
    Extended metrics used in the current framework.

    Notes:
    - Uses fixed CLASS_IDS to keep metric dimensionality stable across runs.
    - Keeps macro and weighted metrics.
    - Includes macro FPR.
    """
    y_true, y_pred = _to_numpy(y_true, y_pred)

    accuracy = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        average="macro",
        zero_division=0,
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        average="weighted",
        zero_division=0,
    )

    fpr_macro = compute_macro_fpr(y_true, y_pred)

    return {
        "accuracy": float(accuracy),
        "fpr_macro": float(fpr_macro),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }


def compute_macro_fpr(
    y_true: List[int] | np.ndarray,
    y_pred: List[int] | np.ndarray,
) -> float:
    """
    Macro-average FPR across fixed classes (one-vs-rest style).
    """
    y_true, y_pred = _to_numpy(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)
    tp, fp, fn, tn = _compute_confusion_components(cm)

    fpr_per_class = _safe_divide(fp, (fp + tn))
    return float(np.mean(fpr_per_class))


def compute_per_class_metrics(
    y_true: List[int] | np.ndarray,
    y_pred: List[int] | np.ndarray,
) -> List[Dict[str, float]]:
    """
    Per-class metrics using fixed class IDs and target names.
    """
    y_true, y_pred = _to_numpy(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)
    tp, fp, fn, tn = _compute_confusion_components(cm)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        average=None,
        zero_division=0,
    )

    fpr_per_class = _safe_divide(fp, (fp + tn))

    results: List[Dict[str, float]] = []
    for i, class_id in enumerate(CLASS_IDS):
        results.append(
            {
                "class_id": int(class_id),
                "class_name": TARGET_NAMES[i],
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "fpr": float(fpr_per_class[i]),
                "support": int(support[i]),
                "tp": int(tp[i]),
                "fp": int(fp[i]),
                "fn": int(fn[i]),
                "tn": int(tn[i]),
            }
        )

    return results


# ---------------------------------------------------------------------
# Canonical notebook-equivalent metrics
# ---------------------------------------------------------------------
def compute_notebook_metrics(
    y_true: List[int] | np.ndarray,
    y_pred: List[int] | np.ndarray,
) -> Dict[str, Any]:
    """
    Reproduces the notebook-style evaluation as closely as possible.

    Includes:
    - Accuracy
    - Precision weighted (zero_division=1)
    - Recall / Detection Rate
    - F1 weighted (zero_division=1)
    - Confusion matrix
    - FPR global
    - FPR per class
    - Weighted FPR
    - FPR excluding majority class
    - Classification report
    """
    y_true, y_pred = _to_numpy(y_true, y_pred)

    # -------------------------------------------------------------
    # Global metrics exactly aligned with notebook logic
    # -------------------------------------------------------------
    test_total_predictions = len(y_true)
    test_correct_predictions = int(np.sum(y_true == y_pred))
    test_accuracy = test_correct_predictions / test_total_predictions

    precision = precision_score(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        average="weighted",
        zero_division=1,
    )

    recall = recall_score(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        average="weighted",
        zero_division=1,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        average="weighted",
        zero_division=1,
    )

    # -------------------------------------------------------------
    # Classification report
    # -------------------------------------------------------------
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        target_names=TARGET_NAMES,
        zero_division=1,
        output_dict=True,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        target_names=TARGET_NAMES,
        zero_division=1,
    )

    # -------------------------------------------------------------
    # Confusion matrix and standard components
    # -------------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)
    tp, fp, fn, tn = _compute_confusion_components(cm)

    dr_value = np.sum(tp) / (np.sum(tp) + np.sum(fn))
    fpr_value = np.sum(fp) / (np.sum(fp) + np.sum(tn))

    # -------------------------------------------------------------
    # FPR per class
    # Matches your requested snippet:
    # fpr_per_class = fp / (fp + tn)
    # -------------------------------------------------------------
    fpr_per_class = _safe_divide(fp, (fp + tn))

    fpr_per_class_named = {}
    for i, class_name in enumerate(TARGET_NAMES):
        fpr_per_class_named[class_name] = float(fpr_per_class[i])

    # -------------------------------------------------------------
    # Majority class distribution
    # -------------------------------------------------------------
    counts = np.bincount(y_true, minlength=len(CLASS_IDS))
    class_distribution = {
        TARGET_NAMES[i]: int(counts[i]) for i in range(len(TARGET_NAMES))
    }
    majority_class = int(np.argmax(counts))
    majority_class_name = TARGET_NAMES[majority_class]

    # -------------------------------------------------------------
    # FPR excluding majority class
    # Following your requested logic as closely as possible
    # -------------------------------------------------------------
    mask_filtered = y_true != majority_class
    y_true_filtered = y_true[mask_filtered]
    y_pred_filtered = y_pred[mask_filtered]

    if len(y_true_filtered) > 0:
        # To preserve notebook-like behavior, use union of labels present
        filtered_labels = np.unique(np.concatenate([y_true_filtered, y_pred_filtered]))
        cm_filtered = confusion_matrix(y_true_filtered, y_pred_filtered, labels=filtered_labels)

        tp_filtered = np.diag(cm_filtered)
        fp_filtered = np.sum(cm_filtered, axis=0) - tp_filtered
        fn_filtered = np.sum(cm_filtered, axis=1) - tp_filtered
        tn_filtered = cm_filtered.sum() - (fp_filtered + fn_filtered + tp_filtered)

        denom_filtered = np.sum(fp_filtered) + np.sum(tn_filtered)
        fpr_filtered = float(np.sum(fp_filtered) / denom_filtered) if denom_filtered > 0 else 0.0
    else:
        cm_filtered = np.zeros((0, 0), dtype=int)
        fpr_filtered = 0.0

    # -------------------------------------------------------------
    # Weighted FPR
    # class_weights = counts / np.sum(counts)
    # fpr_weighted = np.sum(fpr_per_class * class_weights)
    # -------------------------------------------------------------
    class_weights = counts / np.sum(counts)
    fpr_weighted = float(np.sum(fpr_per_class * class_weights))

    return {
        # canonical notebook-like globals
        "accuracy": float(test_accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "detection_rate": float(dr_value),
        "fpr_global": float(fpr_value),

        # requested additional FPR views
        "fpr_weighted": float(fpr_weighted),
        "fpr_excluding_majority_class": float(fpr_filtered),

        # class information
        "target_names": TARGET_NAMES,
        "class_ids": CLASS_IDS,
        "class_distribution": class_distribution,
        "majority_class_id": majority_class,
        "majority_class_name": majority_class_name,
        "class_weights": {
            TARGET_NAMES[i]: float(class_weights[i]) for i in range(len(TARGET_NAMES))
        },

        # matrix and derived components
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_filtered": cm_filtered.tolist(),
        "tp": tp.tolist(),
        "fp": fp.tolist(),
        "fn": fn.tolist(),
        "tn": tn.tolist(),

        # FPR per class
        "fpr_per_class": {
            TARGET_NAMES[i]: float(fpr_per_class[i]) for i in range(len(TARGET_NAMES))
        },

        # reports
        "classification_report": report_dict,
        "classification_report_text": report_text,
    }


# ---------------------------------------------------------------------
# Optional pretty-printer for notebook-style console output
# ---------------------------------------------------------------------
def print_notebook_metrics(
    y_true: List[int] | np.ndarray,
    y_pred: List[int] | np.ndarray,
) -> Dict[str, Any]:
    """
    Prints notebook-style metrics, including the extra FPR outputs you requested.
    Returns the same dict as compute_notebook_metrics().
    """
    results = compute_notebook_metrics(y_true, y_pred)

    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Test Precision: {results['precision_weighted']:.4f}")
    print(f"Test Recall: {results['recall_weighted']:.4f}")
    print(f"Test F1-Score: {results['f1_weighted']:.4f}")

    print("\nClassification Report:\n")
    print(results["classification_report_text"])

    print(f"Final Accuracy: {results['accuracy']:.6f}")
    print(f"Final Detection Rate (Recall): {results['detection_rate']:.6f}")
    print(f"Final False Positive Rate: {results['fpr_global']:.8f}")
    print(f"Final F1 Score: {results['f1_weighted']:.6f}")

    print("\nFPR por clase:")
    for class_name in TARGET_NAMES:
        print(f"FPR para {class_name}: {results['fpr_per_class'][class_name]:.8f}")

    print(f"\nClase mayoritaria: {results['majority_class_name']} (id={results['majority_class_id']})")
    print(f"FPR sin clase mayoritaria: {results['fpr_excluding_majority_class']:.8f}")
    print(f"FPR ponderada: {results['fpr_weighted']:.8f}")

    return results