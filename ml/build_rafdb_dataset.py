"""
build_rafdb_dataset.py - Build fusion feature dataset from RAF-DB.

RAF-DB folder structure:
    data/DATASET/
        train/  1/ 2/ 3/ 4/ 5/ 6/ 7/
        test/   1/ 2/ 3/ 4/ 5/ 6/ 7/

RAF-DB label mapping -> our EMOTION_NAMES index:
    1=Surprise -> 6,  2=Fear -> 2,  3=Disgust -> 1
    4=Happy    -> 3,  5=Sad  -> 5,  6=Angry   -> 0,  7=Neutral -> 4

Output:
    data/rafdb_features_train.npy
    data/rafdb_labels_train.npy
    data/rafdb_features_test.npy
    data/rafdb_labels_test.npy

Usage:
    python ml/build_rafdb_dataset.py
    python ml/build_rafdb_dataset.py --split test
    python ml/build_rafdb_dataset.py --limit 200
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
from screen_emotion.geometric_emotion_features import GeometricEmotionFeatures

# RAF-DB folder number -> our emotion index
RAFDB_LABEL_MAP = {
    1: 6,  # Surprise
    2: 2,  # Fear
    3: 1,  # Disgust
    4: 3,  # Happy
    5: 5,  # Sad
    6: 0,  # Angry
    7: 4,  # Neutral
}


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


def build_dataset(rafdb_dir, split, output_dir, limit):
    split_dir = Path(rafdb_dir) / split
    if not split_dir.exists():
        print(f"[ERROR] Directory not found: {split_dir}")
        sys.exit(1)

    print("Loading models (DeepFace + MediaPipe)...")
    predictor = EmotionPredictor(model_path=None)
    geo       = GeometricEmotionFeatures()
    print("Models loaded.")

    image_paths = []
    for folder_num, emotion_idx in RAFDB_LABEL_MAP.items():
        class_dir = split_dir / str(folder_num)
        if not class_dir.exists():
            print(f"  [WARN] Folder missing: {class_dir}")
            continue
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            for img_path in class_dir.glob(ext):
                image_paths.append((str(img_path), emotion_idx))

    if limit:
        image_paths = image_paths[:limit]

    total            = len(image_paths)
    all_features     = []
    all_labels       = []
    skipped_no_mesh  = 0
    skipped_deepface = 0

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
    print(f"  Total:           {total}")
    print(f"  Saved:           {len(all_features)}")
    print(f"  Skipped no mesh: {skipped_no_mesh}")
    print(f"  Skipped DeepFace:{skipped_deepface}")

    if not all_features:
        print("[ERROR] No rows saved.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels,   dtype=np.int32)

    features_path = os.path.join(output_dir, f"rafdb_features_{split}.npy")
    labels_path   = os.path.join(output_dir, f"rafdb_labels_{split}.npy")
    np.save(features_path, X)
    np.save(labels_path,   y)

    print(f"\nSaved:")
    print(f"  {features_path}  shape={X.shape}")
    print(f"  {labels_path}  shape={y.shape}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Build fusion dataset from RAF-DB")
    parser.add_argument("--rafdb-dir", default="data/DATASET")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_dataset(
        rafdb_dir=args.rafdb_dir,
        split=args.split,
        output_dir=args.output_dir,
        limit=args.limit,
    )
