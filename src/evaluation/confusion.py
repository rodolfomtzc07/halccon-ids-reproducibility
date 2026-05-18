# src/evaluation/confusion.py
# src/evaluation/confusion.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from sklearn.metrics import confusion_matrix

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


def compute_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Iterable[int] | None = None,
    use_target_names: bool = True,
) -> pd.DataFrame:
    """
    Compute confusion matrix using canonical class order by default.

    Parameters
    ----------
    y_true : Sequence[int]
        Ground-truth labels.
    y_pred : Sequence[int]
        Predicted labels.
    labels : Iterable[int] | None
        Class IDs to use in the matrix. If None, canonical CLASS_IDS are used.
    use_target_names : bool
        If True, use class names in row/column labels. Otherwise use numeric IDs.

    Returns
    -------
    pd.DataFrame
        Confusion matrix as a DataFrame.
    """
    if labels is None:
        labels = CLASS_IDS
    else:
        labels = list(labels)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if use_target_names:
        row_labels = [f"true_{TARGET_NAMES[label]}" for label in labels]
        col_labels = [f"pred_{TARGET_NAMES[label]}" for label in labels]
    else:
        row_labels = [f"true_{label}" for label in labels]
        col_labels = [f"pred_{label}" for label in labels]

    return pd.DataFrame(cm, index=row_labels, columns=col_labels)


def save_confusion_matrix_csv(
    confusion_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    confusion_df.to_csv(output_path, index=True)