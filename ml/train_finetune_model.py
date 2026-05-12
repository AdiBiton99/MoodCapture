"""
train_finetune_model.py — Fine-tuning של MobileNetV2 על FER2013.

תהליך:
    1. טעינת FER2013 מ-data/fer2013/train ו-data/fer2013/test (RGB 96x96)
    2. בניית מודל: MobileNetV2 (ImageNet, פאטון מקופא) + ראש 7 רגשות
    3. שלב 1 — אימון הראש בלבד עם class weights
    4. שלב 2 — שחרור 30 שכבות אחרונות + אימון ב-LR נמוך
    5. הערכה על test (accuracy, F1 macro)
    6. שמירה ל-models/finetuned_emotion.keras

הרצה:
    python ml/train_finetune_model.py
    python ml/train_finetune_model.py --epochs-head 8 --epochs-finetune 4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from screen_emotion.emotion_predictor import EMOTION_NAMES
from screen_emotion.evaluation_metrics import (
    compute_classification_metrics,
    format_metrics_summary,
)


IMAGE_SIZE = 96
DEFAULT_BATCH = 64


def _load_datasets(dataset_dir: Path, batch_size: int):
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Expected FER2013 folders at {train_dir} and {test_dir}.\n"
            "Run: python ml/prepare_fer2013.py"
        )

    train_raw = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        class_names=EMOTION_NAMES,
        color_mode="rgb",
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )

    test_raw = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        class_names=EMOTION_NAMES,
        color_mode="rgb",
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=batch_size,
        shuffle=False,
    )

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.05),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )

    def _prep_train(images, labels):
        images = augmentation(images, training=True)
        images = preprocess_input(images)
        return images, labels

    def _prep_eval(images, labels):
        return preprocess_input(images), labels

    autotune = tf.data.AUTOTUNE
    train_ds = train_raw.map(_prep_train, num_parallel_calls=autotune).prefetch(autotune)
    test_ds = test_raw.map(_prep_eval, num_parallel_calls=autotune).prefetch(autotune)

    return train_ds, test_ds, train_raw, test_raw


def _compute_class_weights(train_raw) -> dict[int, float]:
    from sklearn.utils.class_weight import compute_class_weight

    labels: list[int] = []
    for _, batch_labels in train_raw.unbatch():
        labels.append(int(batch_labels.numpy()))

    labels_arr = np.asarray(labels)
    classes = np.arange(len(EMOTION_NAMES))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels_arr)
    return {int(idx): float(w) for idx, w in zip(classes, weights)}


def _build_model():
    import tensorflow as tf

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    features = backbone(inputs, training=False)
    pooled = tf.keras.layers.GlobalAveragePooling2D()(features)
    dropped = tf.keras.layers.Dropout(0.3)(pooled)
    outputs = tf.keras.layers.Dense(len(EMOTION_NAMES), activation="softmax")(dropped)

    model = tf.keras.Model(inputs, outputs)
    return model, backbone


def _evaluate_test(model, test_raw) -> dict:
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    y_true: list[str] = []
    y_pred: list[str] = []

    for images, labels in test_raw:
        preprocessed = preprocess_input(tf.cast(images, tf.float32))
        probs = model.predict(preprocessed, verbose=0)
        predicted = np.argmax(probs, axis=1)
        for true_idx, pred_idx in zip(labels.numpy(), predicted):
            y_true.append(EMOTION_NAMES[int(true_idx)])
            y_pred.append(EMOTION_NAMES[int(pred_idx)])

    return compute_classification_metrics(y_true, y_pred, EMOTION_NAMES)


def train(
    dataset_dir: Path,
    output_path: Path,
    batch_size: int,
    epochs_head: int,
    epochs_finetune: int,
    unfreeze_top: int,
) -> None:
    import tensorflow as tf

    print(f"TensorFlow: {tf.__version__}")
    print(f"FER2013 dataset directory: {dataset_dir}")

    train_ds, test_ds, train_raw, test_raw = _load_datasets(dataset_dir, batch_size)

    print("Computing class weights...")
    class_weights = _compute_class_weights(train_raw)
    for idx, name in enumerate(EMOTION_NAMES):
        print(f"  {name:>10}: weight={class_weights[idx]:.3f}")

    print("\nBuilding MobileNetV2 + classification head...")
    model, backbone = _build_model()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=2,
        restore_best_weights=True,
    )

    print(f"\n=== Stage 1: training the classification head ({epochs_head} epochs) ===")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs_head,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=2,
    )

    if epochs_finetune > 0:
        print(f"\n=== Stage 2: fine-tuning top {unfreeze_top} layers ({epochs_finetune} epochs) ===")
        backbone.trainable = True
        for layer in backbone.layers[:-unfreeze_top]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=epochs_finetune,
            class_weight=class_weights,
            callbacks=[early_stopping],
            verbose=2,
        )

    print("\n=== Evaluating on FER2013 test ===")
    metrics = _evaluate_test(model, test_raw)
    print(format_metrics_summary(metrics))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"\nModel saved to: {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune MobileNetV2 on FER2013.")
    parser.add_argument("--dataset-dir", default="data/fer2013")
    parser.add_argument("--output", default="models/finetuned_emotion.keras")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--epochs-head", type=int, default=8)
    parser.add_argument("--epochs-finetune", type=int, default=4)
    parser.add_argument("--unfreeze-top", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        dataset_dir=Path(args.dataset_dir),
        output_path=Path(args.output),
        batch_size=args.batch_size,
        epochs_head=args.epochs_head,
        epochs_finetune=args.epochs_finetune,
        unfreeze_top=args.unfreeze_top,
    )
