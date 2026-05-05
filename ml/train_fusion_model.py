"""
train_fusion_model.py - Train Fusion classifier on FER2013 + RAF-DB features.

Steps:
    1. Load FER2013 train features (fusion_features_train.npy)
    2. Load RAF-DB train features if available (rafdb_features_train.npy)
    3. Combine both datasets
    4. Normalize with StandardScaler
    5. Train MLPClassifier with class weights (balanced)
    6. Evaluate on test sets
    7. Save model to models/fusion_model.pkl

Usage:
    python ml/train_fusion_model.py
    python ml/train_fusion_model.py --classifier lr
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from screen_emotion.emotion_predictor import EMOTION_NAMES


def _load_split(data_dir, prefix, split):
    features_path = os.path.join(data_dir, f"{prefix}_features_{split}.npy")
    labels_path   = os.path.join(data_dir, f"{prefix}_labels_{split}.npy")
    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        return None, None
    X = np.load(features_path).astype(np.float32)
    y = np.load(labels_path).astype(np.int32)
    return X, y


def _print_distribution(y, title):
    print(f"\n  {title}:")
    for idx, name in enumerate(EMOTION_NAMES):
        count = int(np.sum(y == idx))
        bar   = "=" * (count // 200)
        print(f"    {name:>10}: {count:5d}  {bar}")


def train(data_dir, model_out, classifier_type):
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.utils.class_weight import compute_sample_weight

    # --- Load FER2013 train ---
    X_fer, y_fer = _load_split(data_dir, "fusion", "train")
    if X_fer is None:
        print("[ERROR] FER2013 train features not found.")
        print("  Run: python ml/build_fusion_dataset.py --split train")
        sys.exit(1)
    print(f"FER2013 train: {X_fer.shape[0]} samples")

    # --- Load RAF-DB train (optional) ---
    X_raf, y_raf = _load_split(data_dir, "rafdb", "train")
    if X_raf is not None:
        print(f"RAF-DB  train: {X_raf.shape[0]} samples")
        X_train = np.concatenate([X_fer, X_raf], axis=0)
        y_train = np.concatenate([y_fer, y_raf], axis=0)
        print(f"Combined:      {X_train.shape[0]} samples")
    else:
        print("RAF-DB train not found -- using FER2013 only")
        X_train, y_train = X_fer, y_fer

    _print_distribution(y_train, "Train distribution")

    # --- Load test sets ---
    X_test_fer, y_test_fer = _load_split(data_dir, "fusion", "test")
    X_test_raf, y_test_raf = _load_split(data_dir, "rafdb", "test")

    if X_test_fer is not None and X_test_raf is not None:
        X_test = np.concatenate([X_test_fer, X_test_raf], axis=0)
        y_test = np.concatenate([y_test_fer, y_test_raf], axis=0)
        print(f"\nTest set: FER2013({X_test_fer.shape[0]}) + RAF-DB({X_test_raf.shape[0]}) = {X_test.shape[0]}")
    elif X_test_fer is not None:
        X_test, y_test = X_test_fer, y_test_fer
        print(f"\nTest set: FER2013 only ({X_test.shape[0]})")
    else:
        X_test, y_test = None, None
        print("\nNo test set found.")

    # --- Normalize ---
    print("\nNormalizing features (StandardScaler)...")
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test = scaler.transform(X_test)

    # --- Class weights (balanced) ---
    sample_weights = compute_sample_weight("balanced", y_train)

    # --- Build classifier ---
    if classifier_type == "mlp":
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            verbose=False,
        )
    else:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                 multi_class="multinomial", random_state=42)

    print(f"\nTraining {classifier_type.upper()} with class weights (balanced)...")
    clf.fit(X_train, y_train, sample_weight=sample_weights)

    # --- Evaluate ---
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    print(f"\nTrain accuracy: {train_acc * 100:.2f}%")

    if X_test is not None:
        test_pred = clf.predict(X_test)
        test_acc  = accuracy_score(y_test, test_pred)
        print(f"Test  accuracy: {test_acc * 100:.2f}%")
        print(f"\nDetailed report (test):")
        print(classification_report(y_test, test_pred,
                                    target_names=EMOTION_NAMES, zero_division=0))

    # --- DeepFace baseline ---
    if X_test is not None:
        X_test_orig = scaler.inverse_transform(X_test)
        df_pred = np.argmax(X_test_orig[:, :7], axis=1)
        df_acc  = accuracy_score(y_test, df_pred)
        print(f"DeepFace-only baseline: {df_acc * 100:.2f}%")
        print(f"Improvement:           +{(test_acc - df_acc) * 100:.2f}%")

    # --- Save ---
    import joblib
    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    joblib.dump({"model": clf, "scaler": scaler,
                 "emotion_names": EMOTION_NAMES,
                 "feature_dim": X_train.shape[1],
                 "classifier_type": classifier_type}, model_out)
    print(f"\nModel saved: {model_out}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Train Fusion model")
    parser.add_argument("--data-dir",   default="data")
    parser.add_argument("--model-out",  default="models/fusion_model.pkl")
    parser.add_argument("--classifier", choices=["mlp", "lr"], default="mlp")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(data_dir=args.data_dir,
          model_out=args.model_out,
          classifier_type=args.classifier)
