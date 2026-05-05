"""
build_fusion_dataset.py - Build fusion feature dataset from FER2013.

For each image:
    1. Run DeepFace  -> 7 emotion probabilities
    2. Run GeometricEmotionFeatures -> ~20 geometric features
    3. If both succeed -> save row to X matrix + label to y

Output:
    data/fusion_features_train.npy  (N, 27)
    data/fusion_labels_train.npy    (N,)

Setup:
    Download FER2013 from https://www.kaggle.com/datasets/msambare/fer2013
    Place train/ and test/ folders under data/fer2013/

Usage:
    python ml/build_fusion_dataset.py
    python ml/build_fusion_dataset.py --split test
    python ml/build_fusion_dataset.py --limit 500   # quick test
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import cv2

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from screen_emotion.emotion_predictor import EmotionPredictor, EMOTION_NAMES
from screen_emotion.geometric_emotion_features import GeometricEmotionFeatures, FEATURE_DIM

LABEL_MAP = {name: idx for idx, name in enumerate(EMOTION_NAMES)}


def _load_image_rgb(path: str) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_CUBIC)
    return img


def build_dataset(fer2013_dir, split, output_dir, limit):
    split_dir = Path(fer2013_dir) / split
    if not split_dir.exists():
        print(f"[ERROR] Directory not found: {split_dir}")
        print("  Make sure FER2013 is at data/fer2013/")
        sys.exit(1)

    print("Loading models (DeepFace + MediaPipe)...")
    predictor = EmotionPredictor(model_path=None)
    geo       = GeometricEmotionFeatures()
    print("Models loaded.")

    image_paths = []
    for emotion_name in EMOTION_NAMES:
        class_dir = split_dir / emotion_name
        if not class_dir.exists():
            print(f"  [WARN] Emotion folder missing: {class_dir} -- skipping")
            continue
        for img_path in class_dir.glob("*.png"):
            image_paths.append((str(img_path), LABEL_MAP[emotion_name]))
        for img_path in class_dir.glob("*.jpg"):
            image_paths.append((str(img_path), LABEL_MAP[emotion_name]))

    if limit:
        image_paths = image_paths[:limit]

    total             = len(image_paths)
    all_features      = []
    all_labels        = []
    skipped_no_mesh   = 0
    skipped_deepface  = 0

    print(f"Processing {total} images from {split_dir}...")

    for i, (img_path, label) in enumerate(image_paths):
        if i % 500 == 0 and i > 0:
            pct = 100.0 * i / total
            print(f"  [{i}/{total}] {pct:.1f}%  saved={len(all_features)}"
                  f"  skip_mesh={skipped_no_mesh}  skip_df={skipped_deepface}",
                  flush=True)

        image_rgb = _load_image_rgb(img_path)
        if image_rgb is None:
            continue

        try:
            _, _, deepface_all = predictor.predict(image_rgb)
        except Exception:
            skipped_deepface += 1
            continue

        geo_features = geo.extract(image_rgb)
        if geo_features is None:
            skipped_no_mesh += 1
            continue

        deepface_vec = np.array(
            [deepface_all.get(name, 0.0) for name in EMOTION_NAMES],
            dtype=np.float32,
        )
        all_features.append(np.concatenate([deepface_vec, geo_features]))
        all_labels.append(label)

    print(f"\nSummary:")
    print(f"  Total images:    {total}")
    print(f"  Saved:           {len(all_features)}")
    print(f"  Skipped no mesh: {skipped_no_mesh}")
    print(f"  Skipped DeepFace:{skipped_deepface}")

    if not all_features:
        print("[ERROR] No rows saved -- check FER2013 path.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels,   dtype=np.int32)

    features_path = os.path.join(output_dir, f"fusion_features_{split}.npy")
    labels_path   = os.path.join(output_dir, f"fusion_labels_{split}.npy")

    np.save(features_path, X)
    np.save(labels_path,   y)

    print(f"\nSaved:")
    print(f"  {features_path}  shape={X.shape}")
    print(f"  {labels_path}  shape={y.shape}")
    print(f"\nNext: python ml/train_fusion_model.py")


def _parse_args():
    parser = argparse.ArgumentParser(description="Build fusion dataset from FER2013")
    parser.add_argument("--fer2013-dir", default="data/fer2013")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_dataset(
        fer2013_dir=args.fer2013_dir,
        split=args.split,
        output_dir=args.output_dir,
        limit=args.limit,
    )
