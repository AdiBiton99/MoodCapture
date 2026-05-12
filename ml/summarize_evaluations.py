"""
summarize_evaluations.py — מסכם את כל קבצי ה-JSON של ההערכות לטבלת השוואה.

קלט: כל קובצי .json בתיקייה (ברירת מחדל: reports/)
פלט: טבלת השוואה + per-class F1 + טבלת ranking

הרצה:
    python ml/summarize_evaluations.py
    python ml/summarize_evaluations.py --reports-dir reports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_metrics(reports_dir: Path) -> list[dict]:
    metrics_list = []
    for json_path in sorted(reports_dir.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            data["_filename"] = json_path.name
            metrics_list.append(data)
    return metrics_list


def _format_mode(metrics: dict) -> str:
    mode = metrics.get("mode", "unknown")
    if mode == "ensemble":
        weight = metrics.get("ensemble_weight", 0.5)
        return f"Ensemble (DF={weight:.2f})"
    return mode.capitalize()


def _print_main_table(metrics_list: list[dict]) -> None:
    print("=" * 95)
    print(f"{'File':<35} {'Mode':<20} {'N':>6} {'Acc':>8} {'F1-mac':>8} {'F1-wei':>8}")
    print("-" * 95)

    rows = []
    for metrics in metrics_list:
        filename = metrics.get("_filename", "")
        mode_label = _format_mode(metrics)
        n = metrics.get("n_samples", 0)
        accuracy = metrics.get("accuracy", 0.0) or 0.0
        macro = (metrics.get("averages") or {}).get("macro") or {}
        weighted = (metrics.get("averages") or {}).get("weighted") or {}
        f1_macro = macro.get("f1", 0.0)
        f1_weighted = weighted.get("f1", 0.0)

        rows.append(
            (filename, mode_label, n, accuracy, f1_macro, f1_weighted)
        )
        print(
            f"{filename:<35} {mode_label:<20} {n:>6} "
            f"{accuracy * 100:>7.2f}% {f1_macro:>8.4f} {f1_weighted:>8.4f}"
        )

    print("=" * 95)

    if rows:
        best_acc = max(rows, key=lambda row: row[3])
        best_f1 = max(rows, key=lambda row: row[4])
        print(
            f"Best accuracy : {best_acc[1]:<25} -> {best_acc[3] * 100:.2f}%  ({best_acc[0]})"
        )
        print(
            f"Best F1 macro : {best_f1[1]:<25} -> {best_f1[4]:.4f}  ({best_f1[0]})"
        )


def _print_per_class_table(metrics_list: list[dict]) -> None:
    if not metrics_list:
        return

    emotions = metrics_list[0].get("labels") or []
    if not emotions:
        return

    print("\nPer-class F1 (rows = emotion, cols = mode):")
    print("-" * 95)
    header = f"{'Emotion':<12}" + "".join(f"{_format_mode(m)[:18]:>20}" for m in metrics_list)
    print(header)
    for emotion in emotions:
        line = f"{emotion:<12}"
        for metrics in metrics_list:
            per_class = metrics.get("per_class") or {}
            stats = per_class.get(emotion) or {}
            f1 = stats.get("f1", 0.0)
            line += f"{f1:>20.4f}"
        print(line)
    print("-" * 95)


def summarize(reports_dir: Path) -> None:
    metrics_list = _load_metrics(reports_dir)
    if not metrics_list:
        print(f"No JSON reports found in {reports_dir}")
        return

    print(f"Loaded {len(metrics_list)} report(s) from {reports_dir}\n")
    _print_main_table(metrics_list)
    _print_per_class_table(metrics_list)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize evaluation JSON reports.")
    parser.add_argument("--reports-dir", default="reports")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    summarize(Path(args.reports_dir))
