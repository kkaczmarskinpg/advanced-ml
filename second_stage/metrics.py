from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class ConfusionCounts:
    tn: int
    fp: int
    fn: int
    tp: int


def _to_numpy(array_like: Iterable) -> np.ndarray:
    return np.asarray(list(array_like))


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def confusion_counts(
    y_true: Iterable,
    y_pred: Iterable,
    positive_label: int | float | str = 1,
) -> ConfusionCounts:
    """Compute binary confusion matrix counts.

    Treats any label equal to positive_label as positive, all others as negative.
    """
    y_true_arr = _to_numpy(y_true)
    y_pred_arr = _to_numpy(y_pred)

    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    true_pos = y_true_arr == positive_label
    pred_pos = y_pred_arr == positive_label

    tp = int(np.sum(true_pos & pred_pos))
    tn = int(np.sum(~true_pos & ~pred_pos))
    fp = int(np.sum(~true_pos & pred_pos))
    fn = int(np.sum(true_pos & ~pred_pos))

    return ConfusionCounts(tn=tn, fp=fp, fn=fn, tp=tp)


def accuracy(counts: ConfusionCounts) -> float:
    return _safe_div(counts.tp + counts.tn, counts.tp + counts.tn + counts.fp + counts.fn)


def precision(counts: ConfusionCounts) -> float:
    return _safe_div(counts.tp, counts.tp + counts.fp)


def recall(counts: ConfusionCounts) -> float:
    return _safe_div(counts.tp, counts.tp + counts.fn)


def f1_score(counts: ConfusionCounts) -> float:
    prec = precision(counts)
    rec = recall(counts)
    return _safe_div(2.0 * prec * rec, prec + rec)


def specificity(counts: ConfusionCounts) -> float:
    return _safe_div(counts.tn, counts.tn + counts.fp)


def compute_metrics(
    y_true: Iterable,
    y_pred: Iterable,
    positive_label: int | float | str = 1,
) -> Dict[str, float]:
    """Compute core metrics for binary classification.

    Returns a dict with keys: accuracy, precision, recall, f1, specificity.
    """
    counts = confusion_counts(y_true, y_pred, positive_label=positive_label)
    return {
        "accuracy": accuracy(counts),
        "precision": precision(counts),
        "recall": recall(counts),
        "f1": f1_score(counts),
        "specificity": specificity(counts),
    }


def confusion_matrix_tuple(
    y_true: Iterable,
    y_pred: Iterable,
    positive_label: int | float | str = 1,
) -> Tuple[int, int, int, int]:
    """Return confusion matrix counts as (tn, fp, fn, tp)."""
    counts = confusion_counts(y_true, y_pred, positive_label=positive_label)
    return counts.tn, counts.fp, counts.fn, counts.tp


def confusion_table(
    y_true: Iterable,
    y_pred: Iterable,
    positive_label: int | float | str = 1,
):
    """Return confusion matrix as a pandas DataFrame.

    Rows are actual classes, columns are predicted classes.
    """
    import pandas as pd

    counts = confusion_counts(y_true, y_pred, positive_label=positive_label)
    return pd.DataFrame(
        [[counts.tn, counts.fp], [counts.fn, counts.tp]],
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    )


def show_confusion(
    y_true: Iterable,
    y_pred: Iterable,
    positive_label: int | float | str = 1,
    title: str | None = None,
):
    """Return confusion table and optionally print a title.

    Use in notebooks with display(show_confusion(...)).
    """
    if title:
        print(f"Confusion matrix: {title}")
    return confusion_table(y_true, y_pred, positive_label=positive_label)


def plot_confusion(
    y_true: Iterable,
    y_pred: Iterable,
    positive_label: int | float | str = 1,
    title: str | None = None,
    normalize: bool = False,
    ax=None,
):
    """Plot a confusion matrix using matplotlib.

    If normalize=True, rows are normalized to sum to 1.
    Returns the matplotlib Axes.
    """
    import matplotlib.pyplot as plt

    table = confusion_table(y_true, y_pred, positive_label=positive_label)
    values = table.values.astype(float)

    if normalize:
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums != 0)

    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))

    im = ax.imshow(values, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred_0", "pred_1"])
    ax.set_yticklabels(["actual_0", "actual_1"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title or "Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{values[i, j]:.2f}" if normalize else f"{int(values[i, j])}",
                    ha="center", va="center", color="black")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


__all__ = [
    "ConfusionCounts",
    "confusion_counts",
    "confusion_matrix_tuple",
    "confusion_table",
    "show_confusion",
    "plot_confusion",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "specificity",
    "compute_metrics",
]
