"""
evaluate_emotion_model.py — הערכת מודל רגשות על דאטה מתויג.

מחשב accuracy, precision, recall, F1, support ומטריצת בלבול.

מבנה דאטה מצופה:
    <dataset-dir>/<split>/<emotion>/*.png|*.jpg
    או
    <dataset-dir>/<emotion>/*.png|*.jpg   (כשלא מועבר --split)

דוגמאות:
    python ml/evaluate_emotion_model.py --dataset-dir data/fer2013 --split test
    python ml/evaluate_emotion_model.py --dataset-dir data/fer2013 --split test --mode finetuned
    python ml/evaluate_emotion_model.py --dataset-dir data/fer2013 --split test --mode ensemble --ensemble-weight 0.6
    python ml/evaluate_emotion_model.py --dataset-dir data/eval --source pipeline
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from screen_emotion.emotion_analysis_service import EmotionAnalysisService
from screen_emotion.emotion_predictor import EMOTION_NAMES, EmotionPredictor
from screen_emotion.evaluation_metrics import compute_classification_metrics, format_metrics_summary
from screen_emotion.face_detection import MTCNNFaceDetector
from screen_emotion.image_preprocessing import ImagePreprocessor
from screen_emotion.multi_face_aggregator import MultiFaceEmotionAggregator


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
DEFAULT_FINETUNED_MODEL_PATH = "models/finetuned_emotion.keras"


def _load_image_rgb(path: str) -> np.ndarray | None:
    image_bgr = cv2.imread(path)
    if image_bgr is None:
        return None
    if image_bgr.ndim == 2:
        return cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _collect_labeled_samples(
    dataset_dir: Path,
    split: str | None,
    limit: int | None,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """
    אוסף תמונות מתויגות מתיקיות לפי רגש. עם --limit, מבצע דגימה
    מאוזנת בין המחלקות (stratified): מנסה לקחת limit / 7 מכל רגש.
    """
    root = dataset_dir / split if split else dataset_dir
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    rng = random.Random(seed)
    per_class: dict[str, list[str]] = {}
    for emotion_name in EMOTION_NAMES:
        class_dir = root / emotion_name
        if not class_dir.is_dir():
            continue
        class_paths = [
            str(image_path)
            for image_path in sorted(class_dir.iterdir())
            if image_path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if class_paths:
            per_class[emotion_name] = class_paths

    if not per_class:
        raise FileNotFoundError(
            f"No labeled images found under {root}. "
            f"Expected subfolders named: {', '.join(EMOTION_NAMES)}"
        )

    samples: list[tuple[str, str]] = []
    if limit is None:
        for emotion_name, paths in per_class.items():
            for path in paths:
                samples.append((path, emotion_name))
        return samples

    per_class_quota = max(1, limit // len(per_class))
    for emotion_name, paths in per_class.items():
        shuffled = paths.copy()
        rng.shuffle(shuffled)
        take = shuffled[:per_class_quota]
        for path in take:
            samples.append((path, emotion_name))

    rng.shuffle(samples)
    return samples[:limit]


def _build_emotion_model(
    mode: str,
    ensemble_weight: float,
    finetuned_path: str,
):
    """בונה את המודל לפי mode — מתאים את הלוגיקה של main.py."""
    if mode == "deepface":
        print("Model: DeepFace")
        return EmotionPredictor(model_path=None)

    if mode == "finetuned":
        from screen_emotion.finetuned_emotion_model import FinetunedEmotionModel
        print(f"Model: Fine-tuned ({finetuned_path})")
        return FinetunedEmotionModel(finetuned_path)

    if mode == "ensemble":
        from screen_emotion.finetuned_emotion_model import FinetunedEmotionModel
        from screen_emotion.ensemble_emotion_model import EnsembleEmotionModel
        print(
            f"Model: Ensemble (DeepFace weight={ensemble_weight:.2f}, "
            f"fine-tuned weight={1.0 - ensemble_weight:.2f})"
        )
        deepface = EmotionPredictor(model_path=None)
        finetuned = FinetunedEmotionModel(finetuned_path)
        return EnsembleEmotionModel(deepface, finetuned, weight_deepface=ensemble_weight)

    raise ValueError(f"Unknown evaluation mode: {mode}")


def _build_analysis_service(
    mode: str,
    ensemble_weight: float,
    finetuned_path: str,
) -> EmotionAnalysisService:
    return EmotionAnalysisService(
        preprocessor=ImagePreprocessor(),
        face_detector=MTCNNFaceDetector(),
        emotion_model=_build_emotion_model(mode, ensemble_weight, finetuned_path),
        aggregator=MultiFaceEmotionAggregator(),
    )


def _evaluate_crops(
    predictor,
    samples: list[tuple[str, str]],
) -> tuple[list[str], list[str], int]:
    y_true: list[str] = []
    y_pred: list[str] = []
    skipped = 0

    total = len(samples)
    for index, (image_path, label) in enumerate(samples, start=1):
        image_rgb = _load_image_rgb(image_path)
        if image_rgb is None:
            skipped += 1
            continue

        try:
            emotion, _, _ = predictor.predict(image_rgb)
        except Exception:
            skipped += 1
            continue

        y_true.append(label)
        y_pred.append(emotion)

        if index % 100 == 0 or index == total:
            print(f"  [{index}/{total}] evaluated={len(y_true)} skipped={skipped}", flush=True)

    return y_true, y_pred, skipped


def _evaluate_pipeline(
    service: EmotionAnalysisService,
    samples: list[tuple[str, str]],
) -> tuple[list[str], list[str], int]:
    y_true: list[str] = []
    y_pred: list[str] = []
    skipped = 0

    total = len(samples)
    for index, (image_path, label) in enumerate(samples, start=1):
        image_rgb = _load_image_rgb(image_path)
        if image_rgb is None:
            skipped += 1
            continue

        result = service.analyze(image_rgb)
        predicted = result.get("final_emotion")
        if not result.get("faces") or not predicted:
            skipped += 1
            continue

        y_true.append(label)
        y_pred.append(str(predicted).lower())

        if index % 50 == 0 or index == total:
            print(f"  [{index}/{total}] evaluated={len(y_true)} skipped={skipped}", flush=True)

    return y_true, y_pred, skipped


def evaluate(
    dataset_dir: str,
    split: str | None,
    source: str,
    mode: str,
    ensemble_weight: float,
    finetuned_path: str,
    limit: int | None,
    output_json: str | None,
) -> dict:
    samples = _collect_labeled_samples(Path(dataset_dir), split, limit)
    print(f"Loaded {len(samples)} labeled images from {dataset_dir}")
    print(f"Evaluation source: {source}")

    if source == "crops":
        predictor = _build_emotion_model(mode, ensemble_weight, finetuned_path)
        y_true, y_pred, skipped = _evaluate_crops(predictor, samples)
    elif source == "pipeline":
        service = _build_analysis_service(mode, ensemble_weight, finetuned_path)
        y_true, y_pred, skipped = _evaluate_pipeline(service, samples)
    else:
        raise ValueError(f"Unsupported source: {source}")

    metrics = compute_classification_metrics(y_true, y_pred, EMOTION_NAMES)
    metrics["dataset_dir"] = dataset_dir
    metrics["split"] = split
    metrics["source"] = source
    metrics["mode"] = mode
    metrics["ensemble_weight"] = ensemble_weight if mode == "ensemble" else None
    metrics["finetuned_path"] = finetuned_path if mode in ("finetuned", "ensemble") else None
    metrics["skipped_samples"] = skipped
    metrics["requested_samples"] = len(samples)

    print()
    print(format_metrics_summary(metrics))
    print(f"\nSkipped samples (unreadable image / no face / prediction error): {skipped}")

    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
        print(f"Metrics saved to: {output_json}")

    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate emotion model metrics on labeled data.")
    parser.add_argument("--dataset-dir", default="data/eval", help="Root dataset directory.")
    parser.add_argument("--split", default=None, help="Optional split folder, e.g. test or train.")
    parser.add_argument(
        "--source",
        choices=("crops", "pipeline"),
        default="crops",
        help="crops = face images; pipeline = full MoodCapture analysis service.",
    )
    parser.add_argument(
        "--mode",
        choices=("deepface", "finetuned", "ensemble"),
        default="deepface",
        help="Which emotion model to evaluate. Mirrors main.py.",
    )
    parser.add_argument(
        "--ensemble-weight",
        type=float,
        default=0.5,
        help="DeepFace weight in ensemble mode (0.0–1.0). Default: 0.5.",
    )
    parser.add_argument(
        "--finetuned-model-path",
        default=DEFAULT_FINETUNED_MODEL_PATH,
        help=f"Path to fine-tuned .keras model. Default: {DEFAULT_FINETUNED_MODEL_PATH}.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N images.")
    parser.add_argument("--output-json", default=None, help="Optional path to save metrics as JSON.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not 0.0 <= args.ensemble_weight <= 1.0:
        raise SystemExit("--ensemble-weight must be between 0.0 and 1.0")
    evaluate(
        dataset_dir=args.dataset_dir,
        split=args.split,
        source=args.source,
        mode=args.mode,
        ensemble_weight=args.ensemble_weight,
        finetuned_path=args.finetuned_model_path,
        limit=args.limit,
        output_json=args.output_json,
    )
