"""
evaluation_metrics.py — מדדי הצלחה לסיווג רגשות.

מחשב accuracy, precision, recall, F1 (לכל מחלקה וממוצעים macro/weighted),
support, מטריצת בלבול ודוח sklearn.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> dict[str, Any]:
    """
    מחשב מדדי סיווג עבור תוויות אמת וחיזוי.

    מחזיר מילון עם accuracy, per_class, averages, confusion_matrix ו-report.
  """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    label_list = list(labels)
    if not label_list:
        raise ValueError("labels must not be empty.")

    if not y_true:
        return {
            "n_samples": 0,
            "labels": label_list,
            "accuracy": None,
            "per_class": {},
            "averages": {},
            "confusion_matrix": [],
            "classification_report": "",
        }

    y_true_norm = [str(label).lower() for label in y_true]
    y_pred_norm = [str(label).lower() for label in y_pred]

    accuracy = float(accuracy_score(y_true_norm, y_pred_norm))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_norm,
        y_pred_norm,
        labels=label_list,
        average=None,
        zero_division=0,
    )

    avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(
        y_true_norm,
        y_pred_norm,
        labels=label_list,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true_norm,
        y_pred_norm,
        labels=label_list,
        average="weighted",
        zero_division=0,
    )

    per_class: dict[str, dict[str, float | int]] = {}
    for idx, label in enumerate(label_list):
        per_class[label] = {
            "precision": round(float(precision[idx]), 4),
            "recall": round(float(recall[idx]), 4),
            "f1": round(float(f1[idx]), 4),
            "support": int(support[idx]),
        }

    matrix = confusion_matrix(y_true_norm, y_pred_norm, labels=label_list)
    report = classification_report(
        y_true_norm,
        y_pred_norm,
        labels=label_list,
        target_names=label_list,
        zero_division=0,
    )

    return {
        "n_samples": len(y_true_norm),
        "labels": label_list,
        "accuracy": round(accuracy, 4),
        "per_class": per_class,
        "averages": {
            "macro": {
                "precision": round(float(avg_precision), 4),
                "recall": round(float(avg_recall), 4),
                "f1": round(float(avg_f1), 4),
            },
            "weighted": {
                "precision": round(float(weighted_precision), 4),
                "recall": round(float(weighted_recall), 4),
                "f1": round(float(weighted_f1), 4),
            },
        },
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }


def format_metrics_summary(metrics: Mapping[str, Any]) -> str:
    """מעצב מילון מדדים לטקסט קריא בטרמינל."""
    lines: list[str] = []
    lines.append(f"Samples evaluated: {metrics.get('n_samples', 0)}")

    accuracy = metrics.get("accuracy")
    if accuracy is not None:
        lines.append(f"Accuracy: {accuracy * 100:.2f}%")

    averages = metrics.get("averages") or {}
    for avg_name in ("macro", "weighted"):
        avg = averages.get(avg_name) or {}
        if not avg:
            continue
        lines.append(
            f"{avg_name.title()} — "
            f"Precision: {avg.get('precision', 0.0):.4f}, "
            f"Recall: {avg.get('recall', 0.0):.4f}, "
            f"F1: {avg.get('f1', 0.0):.4f}"
        )

    lines.append("")
    lines.append("Per class (precision / recall / F1 / support):")
    per_class = metrics.get("per_class") or {}
    for label in metrics.get("labels", []):
        stats = per_class.get(label)
        if not stats:
            continue
        lines.append(
            f"  {label:>10}: "
            f"P={stats['precision']:.4f}  "
            f"R={stats['recall']:.4f}  "
            f"F1={stats['f1']:.4f}  "
            f"n={stats['support']}"
        )

    matrix = metrics.get("confusion_matrix") or []
    labels = list(metrics.get("labels") or [])
    if matrix and labels:
        lines.append("")
        lines.append("Confusion matrix (rows=true, cols=pred):")
        header = " " * 12 + "  ".join(f"{label[:8]:>8}" for label in labels)
        lines.append(header)
        for label, row in zip(labels, matrix):
            row_txt = "  ".join(f"{value:8d}" for value in row)
            lines.append(f"{label:>12}  {row_txt}")

    report = metrics.get("classification_report")
    if report:
        lines.append("")
        lines.append(str(report).rstrip())

    return "\n".join(lines)
